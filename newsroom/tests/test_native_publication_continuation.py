from __future__ import annotations

import json
from types import SimpleNamespace
from dataclasses import replace

import pytest

from newsroom.authority import AuthorityEvents, EventId, ObjectAdmissionId, UtcTimestamp
from newsroom.control_plane.native_evidence import (
    DependencyAssessment,
    NativeEvidenceController,
    NativeEvidenceHold,
    NativeEvidenceSource,
    PublicationRightsAssessment,
    AcquiredEvidence,
)
from newsroom.control_plane.graphiti_operational_readiness import _source_requests
from newsroom.control_plane.native_progress import NativeRevisionJournal
from newsroom.control_plane.native_assessor import (
    RetainedAssessorContractFailure,
    RetainedAssessorPreDispatchFailure,
)
from newsroom.control_plane.native_publication import NativePublicationContinuation
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.control_plane.store import connect
from newsroom.increment10.editorial import (
    EditorialPolicyDecision,
    SourceCurrentness,
    SourceIntegrity,
)
from newsroom.tests.authority_helpers import proof
from newsroom.tests.test_native_graphiti import _native
from newsroom.tests.test_graphiti_operational_readiness import _rights
from newsroom.sources.record_models import SourceDefinitionVersion


_DIGEST = "sha256:" + "a" * 64
_CHECKS = (
    "ACCESS_COMPLETE",
    "ENCODING_VALID",
    "EXTRACTION_COMPLETE",
    "NOT_PAYWALL_FRAGMENT",
    "NOT_TRUNCATED",
    "VERSION_UNAMBIGUOUS",
)


def _decision(package_id):
    return EditorialPolicyDecision.create(
        candidate_version_id="candidate-version",
        candidate_version_digest=_DIGEST,
        governing_manifest_digest=_DIGEST,
        package_admission_id=package_id,
        package_digest=_DIGEST,
        policy_bundle_digest=_DIGEST,
        evaluated_at="2026-09-08T12:03:00Z",
        currentness=(SourceCurrentness(
            "source", "definition", _DIGEST, "CURRENT_VERSION",
            "2026-09-08T12:00:00Z", "2026-09-08T12:01:00Z", None,
            "version", _DIGEST, _DIGEST, "PASS", "CURRENT_VERSION_CONFIRMED",
        ),),
        integrity=(SourceIntegrity(
            "source", ObjectAdmissionId.new(), _DIGEST, _CHECKS,
            "PASS", "INDEPENDENT_ACQUISITION_VERIFIED",
        ),),
        evidence_gate_results=(
            ("CLAIM_TRACEABILITY", "PASS"),
            ("EVIDENCE_SUFFICIENCY", "PASS"),
            ("SOURCE_AUTHORITY", "PASS"),
        ),
    )


def _source(unit):
    request = _source_requests(unit, _rights())[1]
    return NativeEvidenceSource(
        unit,
        SourceDefinitionVersion(
            request,
            EventId.new(),
            1,
            UtcTimestamp.parse("2026-09-08T12:00:00Z"),
            request.digest,
        ),
        PublicationRightsAssessment.create(
            decision="PERMITTED",
            permitted_use="PUBLICATION_EVIDENCE",
            policy_digest=_DIGEST,
            evidence_digest=_DIGEST,
        ),
        DependencyAssessment.create(
            dependency_status="RESOLVED",
            evidential_origin_id="origin",
            originating_report_id="origin",
            evidence_digest=_DIGEST,
        ),
    )


class _Authority:
    def __init__(self, events=None):
        self.receives = 0
        self.events = events

    def candidate_version(self, _version_id):
        return SimpleNamespace(
            candidate_id="candidate",
            governing_manifest=SimpleNamespace(canonical_digest=_DIGEST)
        )

    def receive_evidence_intake(self, _ingress, **_request):
        self.receives += 1
        return SimpleNamespace(receipt_id="intake-receipt")


class _Publication:
    def reconcile_stale_intent(self, *_args, **_kwargs):
        return None

    def __init__(self):
        self.calls = 0

    def advance(self, *_args, **_kwargs):
        self.calls += 1
        self.requests = getattr(self, "requests", ()) + (_kwargs,)
        receipt = lambda name: SimpleNamespace(event_id=name)
        return SimpleNamespace(
            story_receipt=receipt("story-event"),
            publication_receipt=receipt("publication-event"),
            attempt_receipt=receipt("attempt-event"),
            evidence_receipt=receipt("evidence-event"),
            read_proof=object(),
            writer_id="newsroom.offline-exact-copy.v3",
        )


class _Reader:
    def acknowledged_rows(self):
        return SimpleNamespace(rows=(
            SimpleNamespace(surface_kind="ARTICLE"),
            SimpleNamespace(surface_kind="FEED_CARD"),
        ))

    def close(self):
        return None


@pytest.mark.parametrize("legacy_resume", (False, True))
def test_continuation_retains_times_and_replays_without_evidence_redispatch(
    tmp_path, monkeypatch, legacy_resume
) -> None:
    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    journal.advance(
        unit.revision_id,
        stage="CANDIDATE_ADMITTED",
        facts={"candidate_version_id": "candidate-version", "graphiti_receipts": [{}]},
    )
    from newsroom.tests.test_native_progress import _retrieval_facts
    pair = _retrieval_facts()
    pair.update(journal.current(unit.revision_id)["facts"], unknown_inline={"retained": True})
    journal.advance(unit.revision_id, stage="CANDIDATE_ADMITTED", facts=pair)
    package_id = ObjectAdmissionId.new()
    decision = _decision(package_id)
    evidence_calls = []
    evidence = object.__new__(NativeEvidenceController)

    def acquire(_self, **_request):
        evidence_calls.append("acquired")
        _request["before_assessment"]()
        return SimpleNamespace(
            retained=SimpleNamespace(package_admission_id=package_id),
            editorial_decision=decision,
            acquisition_receipt_digests=(_DIGEST,),
        )

    monkeypatch.setattr(NativeEvidenceController, "acquire_and_retain", acquire)
    monkeypatch.setattr(
        "newsroom.control_plane.native_publication.open_private_serving_read_port",
        lambda *_args, **_kwargs: _Reader(),
    )
    authority, publication = _Authority(), _Publication()
    apply = publication.advance

    def apply_with_retained_inline_fact(*args, **kwargs):
        # An effect may retain a newer inline fact before the continuation ACK.
        # The ACK must refresh full facts, not overwrite it with an old snapshot.
        current = journal.current(unit.revision_id)
        journal.advance(unit.revision_id, stage=current["stage"], facts={
            **current["facts"], "effect_marker": {"completed": True},
        })
        return apply(*args, **kwargs)

    monkeypatch.setattr(publication, "advance", apply_with_retained_inline_fact)
    runtime = SimpleNamespace(
        authority=authority,
        ingress=object(),
        publication=publication,
        proof=proof(),
        policies=SimpleNamespace(publication=SimpleNamespace(
            target_path=tmp_path / "serving.sqlite3",
            target_id="private",
            target_context_digest=_DIGEST,
        )),
    )
    times = iter((
        UtcTimestamp.parse("2026-09-08T12:00:00Z"),
        UtcTimestamp.parse("2026-09-08T12:01:00Z"),
        UtcTimestamp.parse("2026-09-08T12:04:00Z"),
        UtcTimestamp.parse("2026-09-08T12:05:00Z"),
    ))
    continuation = NativePublicationContinuation(
        journal=journal,
        runtime=runtime,
        evidence_controller=evidence,
        sources={unit.revision_id: (_source(unit),)},
        clock=lambda: next(times),
    )

    if legacy_resume:
        original = publication.advance

        def interrupted(*_args, **_kwargs):
            raise RuntimeError("pre-effect interruption")

        monkeypatch.setattr(publication, "advance", interrupted)
        with pytest.raises(RuntimeError, match="pre-effect interruption"):
            continuation.advance(revision_id=unit.revision_id, candidate_version_id="candidate-version")
        legacy = dict(journal.current(unit.revision_id)["facts"])
        legacy.pop("publication_started_at")
        legacy.update(publication_applied_at="2026-09-08T12:04:00.000000Z",
                      publication_observed_at="2026-09-08T12:05:00.000000Z")
        journal.advance(unit.revision_id, stage="PUBLICATION_STARTED", facts=legacy)
        monkeypatch.setattr(publication, "advance", original)

    first = continuation.advance(
        revision_id=unit.revision_id, candidate_version_id="candidate-version"
    )
    replay = continuation.advance(
        revision_id=unit.revision_id, candidate_version_id="candidate-version"
    )

    assert first.state == replay.state == "ACKNOWLEDGED"
    assert authority.receives == 1
    assert evidence_calls == ["acquired"]
    assert publication.calls == 2
    assert {
        (request["expected_story_version"],
         request["expected_publication_version"],
         request["expected_delivery_evidence_version"])
        for request in publication.requests
    } == {(0, 0, 0)}
    retained = journal.current(unit.revision_id)
    assert retained["stage"] == "ACKNOWLEDGED"
    for key in ("retrieval_binding", "retrieval_rights_inventory", "unknown_inline"):
        assert retained["facts"][key] == pair[key]
    assert retained["facts"]["effect_marker"] == {"completed": True}
    assert retained["facts"]["intake_received_epoch_seconds"] == 1788868800
    assert retained["facts"]["assessment_started_at"] == "2026-09-08T12:01:00.000000Z"
    assert retained["facts"]["editorial_decision"] == json.loads(
        decision.canonical_bytes()
    )
    if legacy_resume:
        assert retained["facts"]["publication_applied_at"] == legacy["publication_applied_at"]
        assert retained["facts"]["publication_observed_at"] == legacy["publication_observed_at"]
    else:
        assert retained["facts"]["publication_started_at"] == "2026-09-08T12:04:00.000000Z"
        assert "publication_applied_at" not in retained["facts"]
        assert "publication_observed_at" not in retained["facts"]
    assert all("applied_at" not in request and "observed_at" not in request
               for request in publication.requests)
    assert retained["facts"]["graphiti_receipts"] == [{}]
    connection.close()


@pytest.mark.parametrize(
    "reason", ["GOVUK_ACQUISITION_UNAVAILABLE", "WEATHER_ACQUISITION_UNAVAILABLE"]
)
def test_wrapped_transport_failure_retries_before_assessment_then_acknowledges(
    tmp_path, monkeypatch, reason
) -> None:
    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="CANDIDATE_ADMITTED", facts={
        "candidate_version_id": "candidate-version", "graphiti_receipts": [{}],
    })
    package_id = ObjectAdmissionId.new()
    decision = _decision(package_id)
    calls = []

    def acquire(_self, **request):
        calls.append("acquire")
        if len(calls) == 1:
            raise NativeEvidenceHold(reason, "source")
        request["before_assessment"]()
        return SimpleNamespace(
            retained=SimpleNamespace(package_admission_id=package_id),
            editorial_decision=decision,
            acquisition_receipt_digests=(_DIGEST,),
        )

    monkeypatch.setattr(NativeEvidenceController, "acquire_and_retain", acquire)
    monkeypatch.setattr(
        "newsroom.control_plane.native_publication.open_private_serving_read_port",
        lambda *_args, **_kwargs: _Reader(),
    )
    runtime = SimpleNamespace(
        authority=_Authority(), ingress=object(), publication=_Publication(),
        proof=proof(), policies=SimpleNamespace(publication=SimpleNamespace(
            target_path=tmp_path / "serving.sqlite3", target_id="private",
            target_context_digest=_DIGEST,
        )),
    )
    continuation = NativePublicationContinuation(
        journal=journal, runtime=runtime,
        evidence_controller=object.__new__(NativeEvidenceController),
        sources={unit.revision_id: (_source(unit),)},
        clock=lambda: UtcTimestamp.parse("2026-09-08T12:00:00Z"),
    )

    first = continuation.advance(
        revision_id=unit.revision_id, candidate_version_id="candidate-version"
    )
    second = continuation.advance(
        revision_id=unit.revision_id, candidate_version_id="candidate-version"
    )

    assert first.state == "EVIDENCE_HOLD"
    assert first.reason == "ACQUISITION_TRANSPORT_RETRY"
    assert second.state == "ACKNOWLEDGED"
    assert calls == ["acquire", "acquire"]
    assert journal.current(unit.revision_id)["facts"]["acquisition_attempt_count"] == 2
    connection.close()


def test_transport_retry_is_bounded_and_preserves_failed_attempt_count(
    tmp_path, monkeypatch
) -> None:
    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="CANDIDATE_ADMITTED", facts={
        "candidate_version_id": "candidate-version", "graphiti_receipts": [{}],
    })
    calls = []

    def acquire(_self, **_request):
        calls.append("acquire")
        raise OSError("source unavailable before model dispatch")

    monkeypatch.setattr(NativeEvidenceController, "acquire_and_retain", acquire)
    continuation = NativePublicationContinuation(
        journal=journal,
        runtime=SimpleNamespace(
            authority=_Authority(), ingress=object(), publication=_Publication(),
            proof=proof(), policies=SimpleNamespace(publication=object()),
        ),
        evidence_controller=object.__new__(NativeEvidenceController),
        sources={unit.revision_id: (_source(unit),)},
        clock=lambda: UtcTimestamp.parse("2026-09-08T12:00:00Z"),
    )

    results = tuple(
        continuation.advance(
            revision_id=unit.revision_id,
            candidate_version_id="candidate-version",
        )
        for _ in range(4)
    )

    assert [item.state for item in results] == ["EVIDENCE_HOLD"] * 4
    assert results[-1].reason == "ACQUISITION_TRANSPORT_RETRY_EXHAUSTED"
    assert calls == ["acquire"] * 3
    facts = journal.current(unit.revision_id)["facts"]
    assert facts["acquisition_attempt_count"] == 3
    assert facts["acquisition_retryable"] is False
    connection.close()


def test_deterministic_acquisition_hold_is_not_retried(
    tmp_path, monkeypatch
) -> None:
    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="CANDIDATE_ADMITTED", facts={
        "candidate_version_id": "candidate-version", "graphiti_receipts": [{}],
    })
    calls = []

    def acquire(_self, **_request):
        calls.append("acquire")
        raise NativeEvidenceHold("GOVUK_EVIDENCE_METADATA_HOLD", "source")

    monkeypatch.setattr(NativeEvidenceController, "acquire_and_retain", acquire)
    continuation = NativePublicationContinuation(
        journal=journal,
        runtime=SimpleNamespace(
            authority=_Authority(), ingress=object(), publication=_Publication(),
            proof=proof(), policies=SimpleNamespace(publication=object()),
        ),
        evidence_controller=object.__new__(NativeEvidenceController),
        sources={unit.revision_id: (_source(unit),)},
        clock=lambda: UtcTimestamp.parse("2026-09-08T12:00:00Z"),
    )

    first = continuation.advance(
        revision_id=unit.revision_id, candidate_version_id="candidate-version"
    )
    replay = continuation.advance(
        revision_id=unit.revision_id, candidate_version_id="candidate-version"
    )

    assert first.state == replay.state == "EVIDENCE_HOLD"
    assert first.reason == replay.reason == "GOVUK_EVIDENCE_METADATA_HOLD"
    assert calls == ["acquire"]
    assert journal.current(unit.revision_id)["facts"]["acquisition_retryable"] is False
    connection.close()


def test_superseded_assessment_revalidation_keeps_intake_and_prior_evidence(tmp_path, monkeypatch):
    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    old_package = str(ObjectAdmissionId.new())
    journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts={
        "candidate_version_id": "candidate-version", "graphiti_receipts": [{}],
        "intake_receipt_id": "already-acknowledged", "reason": "INVALID_GOVERNED_CLAIM_EVIDENCE",
        "package_admission_id": old_package, "editorial_decision": {"decision_id": "old-decision"},
        "acquisition_attempt_count": 3, "acquisition_retryable": False,
        "publication_applied_at": "old-time", "publication_observed_at": "old-time",
        "assessment_contract_version": (
            "newsroom.native-evidence-assessor.v12+newsroom.named-entity.v10+"
            "newsroom.zh-hant-hk-shape.v14"
        ),
    })
    calls = []

    def acquire(_self, **request):
        calls.append(request["intake_receipt_id"])
        assert request["assessment_cached_only"] is True
        assert "package_admission_id" not in journal.current(unit.revision_id)["facts"]
        raise NativeEvidenceHold("NO_QUALIFYING_NEW_INFORMATION", unit.source_id)

    monkeypatch.setattr(NativeEvidenceController, "acquire_and_retain", acquire)
    authority, publication = _Authority(), _Publication()
    continuation = NativePublicationContinuation(
        journal=journal,
        runtime=SimpleNamespace(authority=authority, ingress=object(), publication=publication,
                                proof=proof(), policies=SimpleNamespace(publication=object())),
        evidence_controller=object.__new__(NativeEvidenceController),
        sources={unit.revision_id: (_source(unit),)},
        assessment_contract_version=(
            "newsroom.native-evidence-assessor.v12+newsroom.named-entity.v12+"
            "newsroom.zh-hant-hk-shape.v14"
        ),
        clock=lambda: UtcTimestamp.parse("2026-09-08T12:00:00Z"),
    )
    try:
        for _ in range(2):
            result = continuation.advance(revision_id=unit.revision_id, candidate_version_id="candidate-version")
            assert result.reason == "NO_QUALIFYING_NEW_INFORMATION"
        facts = journal.current(unit.revision_id)["facts"]
        assert facts["assessment_superseded"]["package_admission_id"] == old_package
        assert facts["assessment_contract_version"] == (
            "newsroom.native-evidence-assessor.v12+newsroom.named-entity.v12+"
            "newsroom.zh-hant-hk-shape.v14"
        )
        assert calls == ["already-acknowledged"]
        assert authority.receives == publication.calls == 0
    finally:
        connection.close()


def test_producer_contract_revalidation_keeps_the_normal_assessment_path(
    tmp_path, monkeypatch,
) -> None:
    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts={
        "candidate_version_id": "candidate-version", "graphiti_receipts": [{}],
        "intake_receipt_id": "already-acknowledged",
        "reason": "ASSESSOR_RENDERING_CONTRACT_HOLD",
        "assessment_contract_version": "newsroom.native-evidence-assessor.v11",
        "acquisition_attempt_count": 1, "acquisition_retryable": False,
    })
    calls = []

    def acquire(_self, **request):
        calls.append(request["assessment_cached_only"])
        raise NativeEvidenceHold("NO_QUALIFYING_NEW_INFORMATION", unit.source_id)

    monkeypatch.setattr(NativeEvidenceController, "acquire_and_retain", acquire)
    continuation = NativePublicationContinuation(
        journal=journal,
        runtime=SimpleNamespace(
            authority=_Authority(), ingress=object(), publication=_Publication(),
            proof=proof(), policies=SimpleNamespace(publication=object()),
        ),
        evidence_controller=object.__new__(NativeEvidenceController),
        sources={unit.revision_id: (_source(unit),)},
        assessment_contract_version=(
            "newsroom.native-evidence-assessor.v12+consumer-contract"
        ),
        clock=lambda: UtcTimestamp.parse("2026-09-08T12:00:00Z"),
    )
    try:
        result = continuation.advance(
            revision_id=unit.revision_id,
            candidate_version_id="candidate-version",
        )
        assert result.reason == "NO_QUALIFYING_NEW_INFORMATION"
        assert calls == [False]
    finally:
        connection.close()


def test_consumer_only_cache_mode_survives_transport_retry_and_reopen(
    tmp_path, monkeypatch,
) -> None:
    unit = _native()
    path = tmp_path / "private.sqlite3"
    connection = connect(str(path))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts={
        "candidate_version_id": "candidate-version", "graphiti_receipts": [{}],
        "intake_receipt_id": "already-acknowledged",
        "reason": "ASSESSOR_RENDERING_CONTRACT_HOLD",
        "assessment_contract_version": "newsroom.native-evidence-assessor.v12",
        "acquisition_attempt_count": 0, "acquisition_retryable": False,
    })
    cached_modes = []

    def acquire(_self, **request):
        cached_modes.append(request["assessment_cached_only"])
        raise OSError("transport unavailable")

    monkeypatch.setattr(NativeEvidenceController, "acquire_and_retain", acquire)

    def continuation(retained_journal):
        return NativePublicationContinuation(
            journal=retained_journal,
            runtime=SimpleNamespace(
                authority=_Authority(), ingress=object(), publication=_Publication(),
                proof=proof(), policies=SimpleNamespace(publication=object()),
            ),
            evidence_controller=object.__new__(NativeEvidenceController),
            sources={unit.revision_id: (_source(unit),)},
            assessment_contract_version=(
                "newsroom.native-evidence-assessor.v12+consumer-contract"
            ),
            clock=lambda: UtcTimestamp.parse("2026-09-08T12:00:00Z"),
        )

    first = continuation(journal).advance(
        revision_id=unit.revision_id,
        candidate_version_id="candidate-version",
    )
    assert first.reason == "ACQUISITION_TRANSPORT_RETRY"
    connection.close()

    reopened_connection = connect(str(path))
    reopened_journal = NativeRevisionJournal(reopened_connection)
    try:
        second = continuation(reopened_journal).advance(
            revision_id=unit.revision_id,
            candidate_version_id="candidate-version",
        )
        assert second.reason == "ACQUISITION_TRANSPORT_RETRY"
        assert cached_modes == [True, True]
    finally:
        reopened_connection.close()


def test_consumer_only_cache_mode_survives_reopen_from_acquisition_started(
    tmp_path, monkeypatch,
) -> None:
    unit = _native()
    path = tmp_path / "private.sqlite3"
    connection = connect(str(path))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="ACQUISITION_STARTED", facts={
        "candidate_version_id": "candidate-version", "candidate_id": "candidate",
        "graphiti_receipts": [{}], "intake_receipt_id": "already-acknowledged",
        "assessment_contract_version": (
            "newsroom.native-evidence-assessor.v12+consumer-contract"
        ),
        "assessment_superseded": {
            "contract_version": "newsroom.native-evidence-assessor.v12",
            "reason": "ASSESSOR_RENDERING_CONTRACT_HOLD",
            "package_admission_id": None,
            "editorial_decision_id": None,
            "acquisition_attempt_count": 0,
        },
        "acquisition_attempt_count": 1,
        "acquisition_started_at": "2026-09-08T12:00:00.000000Z",
    })
    connection.close()
    cached_modes = []

    def acquire(_self, **request):
        cached_modes.append(request["assessment_cached_only"])
        raise NativeEvidenceHold("NO_QUALIFYING_NEW_INFORMATION", unit.source_id)

    monkeypatch.setattr(NativeEvidenceController, "acquire_and_retain", acquire)
    reopened_connection = connect(str(path))
    reopened_journal = NativeRevisionJournal(reopened_connection)
    continuation = NativePublicationContinuation(
        journal=reopened_journal,
        runtime=SimpleNamespace(
            authority=_Authority(), ingress=object(), publication=_Publication(),
            proof=proof(), policies=SimpleNamespace(publication=object()),
        ),
        evidence_controller=object.__new__(NativeEvidenceController),
        sources={unit.revision_id: (_source(unit),)},
        assessment_contract_version=(
            "newsroom.native-evidence-assessor.v12+consumer-contract"
        ),
        clock=lambda: UtcTimestamp.parse("2026-09-08T12:00:00Z"),
    )
    try:
        result = continuation.advance(
            revision_id=unit.revision_id,
            candidate_version_id="candidate-version",
        )
        assert result.reason == "NO_QUALIFYING_NEW_INFORMATION"
        assert cached_modes == [True]
    finally:
        reopened_connection.close()


def test_post_assessment_dispatch_ambiguity_is_not_redispatched(
    tmp_path, monkeypatch
) -> None:
    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="CANDIDATE_ADMITTED", facts={
        "candidate_version_id": "candidate-version", "graphiti_receipts": [{}],
    })
    calls = []

    def acquire(_self, **request):
        calls.append("assessor-dispatch")
        request["before_assessment"]()
        raise OSError("provider outcome ambiguous")

    monkeypatch.setattr(NativeEvidenceController, "acquire_and_retain", acquire)
    continuation = NativePublicationContinuation(
        journal=journal,
        runtime=SimpleNamespace(
            authority=_Authority(), ingress=object(), publication=_Publication(),
            proof=proof(), policies=SimpleNamespace(publication=object()),
        ),
        evidence_controller=object.__new__(NativeEvidenceController),
        sources={unit.revision_id: (_source(unit),)},
        clock=lambda: UtcTimestamp.parse("2026-09-08T12:00:00Z"),
    )

    first = continuation.advance(
        revision_id=unit.revision_id, candidate_version_id="candidate-version"
    )
    second = continuation.advance(
        revision_id=unit.revision_id, candidate_version_id="candidate-version"
    )

    assert first.state == second.state == "ASSESSMENT_INTERRUPTED"
    assert calls == ["assessor-dispatch"]
    assert journal.current(unit.revision_id)["stage"] == "ASSESSMENT_INTERRUPTED"
    connection.close()


def test_retained_assessor_contract_failure_becomes_typed_hold(tmp_path) -> None:
    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="ASSESSMENT_INTERRUPTED", facts={
        "candidate_id": "candidate",
        "candidate_version_id": "candidate-version",
        "graphiti_receipts": [{}],
        "failure_class": "EvidencePackageError",
        "reason": "ACQUISITION_RESULT_NOT_RETAINED",
    })
    calls: list[str] = []
    retained = RetainedAssessorContractFailure(
        "envelope", "invocation", _DIGEST, _DIGEST, _DIGEST
    )

    def recover(_version):
        calls.append("recover")
        return retained

    continuation = NativePublicationContinuation(
        journal=journal,
        runtime=SimpleNamespace(
            authority=_Authority(), ingress=object(), publication=_Publication(),
            proof=proof(), policies=SimpleNamespace(publication=object()),
        ),
        evidence_controller=object.__new__(NativeEvidenceController),
        sources={unit.revision_id: (_source(unit),)},
        assessment_contract_failure=recover,
    )

    first = continuation.advance(
        revision_id=unit.revision_id, candidate_version_id="candidate-version"
    )
    replay = continuation.advance(
        revision_id=unit.revision_id, candidate_version_id="candidate-version"
    )

    assert first.state == replay.state == "EVIDENCE_HOLD"
    assert first.reason == replay.reason == "ASSESSOR_OUTPUT_CONTRACT_HOLD"
    assert calls == ["recover"]
    facts = journal.current(unit.revision_id)["facts"]
    assert facts["assessment_failure_envelope_id"] == "envelope"
    assert facts["assessment_failure_invocation_id"] == "invocation"
    assert facts["assessment_failure_allocation_digest"] == _DIGEST
    assert facts["assessment_failure_terminal_digest"] == _DIGEST
    assert facts["assessment_failure_context_manifest_digest"] == _DIGEST
    assert facts["acquisition_retryable"] is False
    connection.close()


@pytest.mark.parametrize(
    ("proof_candidate_id", "attempt_count", "expected_state", "retryable"),
    (
        ("candidate", 0, "EVIDENCE_HOLD", True),
        ("candidate", 3, "EVIDENCE_HOLD", False),
        ("other-candidate", 0, "ASSESSMENT_INTERRUPTED", False),
    ),
)
@pytest.mark.parametrize(("initial_stage", "failure_class"), (("ASSESSMENT_INTERRUPTED", "NativeEvidenceError"), ("EVIDENCE_HOLD", "ModelUsageAdmissionError")))
def test_retained_zero_dispatch_assessor_failure_requires_exact_candidate(
    tmp_path,
    monkeypatch,
    proof_candidate_id,
    attempt_count,
    expected_state,
    retryable,
    initial_stage,
    failure_class,
) -> None:
    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    journal.advance(unit.revision_id, stage=initial_stage, facts={
        "candidate_id": "candidate",
        "candidate_version_id": "candidate-version",
        "failure_class": failure_class,
        "reason": "ACQUISITION_RESULT_NOT_RETAINED",
        "acquisition_attempt_count": attempt_count,
    })
    calls, acquisitions = [], []
    retained = RetainedAssessorPreDispatchFailure(
        proof_candidate_id, "candidate-version", _DIGEST, _DIGEST,
    )

    evidence = object.__new__(NativeEvidenceController)

    def acquire(_self, **_request):
        acquisitions.append("attempted")
        raise NativeEvidenceHold("SOURCE_AUTHORITY_HOLD", "source")

    monkeypatch.setattr(NativeEvidenceController, "acquire_and_retain", acquire)
    continuation = NativePublicationContinuation(
        journal=journal,
        runtime=SimpleNamespace(
            authority=_Authority(), ingress=object(), publication=_Publication(),
            proof=proof(), policies=SimpleNamespace(publication=object()),
        ),
        evidence_controller=evidence,
        sources={unit.revision_id: (_source(unit),)},
        assessment_pre_dispatch_failure=lambda _version: (
            calls.append("checked") or retained
        ),
    )
    first = continuation.advance(
        revision_id=unit.revision_id, candidate_version_id="candidate-version"
    )
    if expected_state == "ASSESSMENT_INTERRUPTED":
        assert first.state == initial_stage
        assert journal.current(unit.revision_id)["stage"] == initial_stage
        assert calls == ["checked"]
        connection.close()
        return
    assert first.state == expected_state
    assert first.reason == "ASSESSOR_PRE_DISPATCH_HOLD"
    assert calls == ["checked"]
    facts = journal.current(unit.revision_id)["facts"]
    assert facts["assessment_pre_dispatch_candidate_id"] == "candidate"
    assert facts["assessment_pre_dispatch_candidate_version_id"] == "candidate-version"
    assert facts["assessment_pre_dispatch_manifest_digest"] == _DIGEST
    assert facts["assessment_pre_dispatch_inventory_digest"] == _DIGEST
    assert facts["acquisition_retryable"] is retryable
    second = continuation.advance(
        revision_id=unit.revision_id, candidate_version_id="candidate-version"
    )
    assert second.state == "EVIDENCE_HOLD"
    assert acquisitions == (["attempted"] if retryable else [])
    if retryable:
        assert second.reason == "SOURCE_AUTHORITY_HOLD"
        assert journal.current(unit.revision_id)["facts"][
            "acquisition_attempt_count"
        ] == 1
    else:
        assert second.reason == "ASSESSOR_PRE_DISPATCH_HOLD"
    connection.close()


@pytest.mark.parametrize(
    ("failure_class", "expected_recovery_calls"),
    (("EvidencePackageError", 1), ("OSError", 0)),
)
def test_unproved_assessment_interruption_has_no_follow_on_effect(
    tmp_path, failure_class, expected_recovery_calls,
) -> None:
    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="ASSESSMENT_INTERRUPTED", facts={
        "candidate_id": "candidate",
        "candidate_version_id": "candidate-version",
        "failure_class": failure_class,
        "reason": "ACQUISITION_RESULT_NOT_RETAINED",
    })
    retained_ordinal = journal.current(unit.revision_id)["ordinal"]
    recovery_calls: list[str] = []

    def no_proof(_version):
        recovery_calls.append("checked")
        return None

    authority, publication = _Authority(), _Publication()
    continuation = NativePublicationContinuation(
        journal=journal,
        runtime=SimpleNamespace(
            authority=authority, ingress=object(), publication=publication,
            proof=proof(), policies=SimpleNamespace(publication=object()),
        ),
        evidence_controller=object.__new__(NativeEvidenceController),
        sources={},
        assessment_contract_failure=no_proof,
    )

    result = continuation.advance(
        revision_id=unit.revision_id, candidate_version_id="candidate-version"
    )

    assert result.state == "ASSESSMENT_INTERRUPTED"
    assert len(recovery_calls) == expected_recovery_calls
    assert authority.receives == 0
    assert publication.calls == 0
    assert journal.current(unit.revision_id)["ordinal"] == retained_ordinal
    connection.close()


def test_typed_editorial_hold_is_durable_and_not_repeated(
    tmp_path
) -> None:
    from newsroom.increment10.editorial import EditorialHold

    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    package_id = ObjectAdmissionId.new()
    journal.advance(unit.revision_id, stage="EVIDENCE_RETAINED", facts={
        "candidate_id": "candidate",
        "candidate_version_id": "candidate-version",
        "intake_receipt_id": "intake-receipt",
        "package_admission_id": str(package_id),
        "editorial_decision": json.loads(_decision(package_id).canonical_bytes()),
    })

    class HeldPublication:
        calls = 0

        def advance(self, *_args, **_kwargs):
            self.calls += 1
            raise EditorialHold(SimpleNamespace(
                stable_reason_codes=("FRESHNESS_NOT_PASS",)
            ))

    publication = HeldPublication()
    continuation = NativePublicationContinuation(
        journal=journal,
        runtime=SimpleNamespace(
            authority=_Authority(), ingress=object(), publication=publication,
            proof=proof(), policies=SimpleNamespace(publication=object()),
        ),
        evidence_controller=object.__new__(NativeEvidenceController),
        sources={unit.revision_id: (_source(unit),)},
        clock=lambda: UtcTimestamp.parse("2026-09-08T12:00:00Z"),
    )

    first = continuation.advance(
        revision_id=unit.revision_id, candidate_version_id="candidate-version"
    )
    replay = continuation.advance(
        revision_id=unit.revision_id, candidate_version_id="candidate-version"
    )

    assert first.state == replay.state == "EVIDENCE_HOLD"
    assert first.reason == replay.reason == "FRESHNESS_NOT_PASS"
    assert publication.calls == 1
    retained = journal.current(unit.revision_id)
    assert retained["stage"] == "EVIDENCE_HOLD"
    assert retained["facts"]["editorial_hold_reason_codes"] == [
        "FRESHNESS_NOT_PASS"
    ]
    connection.close()


def test_generic_editorial_error_preserves_publication_intent_for_replay(
    tmp_path
) -> None:
    from newsroom.increment10.editorial import EditorialError

    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    package_id = ObjectAdmissionId.new()
    journal.advance(unit.revision_id, stage="EVIDENCE_RETAINED", facts={
        "candidate_id": "candidate",
        "candidate_version_id": "candidate-version",
        "intake_receipt_id": "intake-receipt",
        "package_admission_id": str(package_id),
        "editorial_decision": json.loads(_decision(package_id).canonical_bytes()),
    })

    class AmbiguousPublication:
        calls = 0

        def advance(self, *_args, **_kwargs):
            self.calls += 1
            raise EditorialError("publication result ambiguous")

    publication = AmbiguousPublication()
    continuation = NativePublicationContinuation(
        journal=journal,
        runtime=SimpleNamespace(
            authority=_Authority(), ingress=object(), publication=publication,
            proof=proof(), policies=SimpleNamespace(publication=object()),
        ),
        evidence_controller=object.__new__(NativeEvidenceController),
        sources={unit.revision_id: (_source(unit),)},
        clock=lambda: UtcTimestamp.parse("2026-09-08T12:00:00Z"),
    )

    for _ in range(2):
        with pytest.raises(EditorialError, match="publication result ambiguous"):
            continuation.advance(
                revision_id=unit.revision_id,
                candidate_version_id="candidate-version",
            )
        assert journal.current(unit.revision_id)["stage"] == "PUBLICATION_STARTED"

    assert publication.calls == 2
    connection.close()


def _events(records):
    return AuthorityEvents(
        policy_id="native-publication-test-read",
        read=lambda *_args: (),
        provenance=lambda event_id, _proof: records[event_id],
        result=lambda *_args: None,
    )


def _provenance(
    *, command, event, aggregate_type, aggregate_id, version, definition=_DIGEST
):
    return SimpleNamespace(
        command_definition=SimpleNamespace(
            command_type=command, definition_digest=definition
        ),
        event=SimpleNamespace(
            event_type=event,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            aggregate_version=version,
            command_definition_digest=definition,
        ),
    )


def test_same_candidate_successor_uses_authenticated_prior_versions_and_replays(
    tmp_path, monkeypatch
) -> None:
    from newsroom.control_plane.native_publication import _aggregate
    from newsroom.increment10.editorial import STORY_COMMAND, STORY_EVENT
    from newsroom.increment10.private_serving import ATTEMPT_COMMAND, ATTEMPT_EVENT

    first = _native()
    second = _native("two")
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((first,))
    journal.land((second,))
    prior_facts = {
        "candidate_id": "candidate",
        "candidate_version_id": "candidate-version-1",
        "story_event_id": "story-event-1",
        "publication_event_id": "publication-event-1",
        "delivery_attempt_event_id": "attempt-event-1",
        "delivery_evidence_event_id": "evidence-event-1",
    }
    journal.advance(first.revision_id, stage="ACKNOWLEDGED", facts=prior_facts)
    package_id = ObjectAdmissionId.new()
    decision = _decision(package_id)
    journal.advance(second.revision_id, stage="EVIDENCE_RETAINED", facts={
        "candidate_id": "candidate",
        "candidate_version_id": "candidate-version-2",
        "intake_receipt_id": "intake-receipt-2",
        "package_admission_id": str(package_id),
        "editorial_decision": json.loads(decision.canonical_bytes()),
    })
    story_id = str(_aggregate("story", "candidate"))
    publication_id = str(_aggregate("publication", "candidate"))
    records = {
        "story-event-1": _provenance(
            command=STORY_COMMAND, event=STORY_EVENT, aggregate_type="story",
            aggregate_id=story_id, version=1,
        ),
        "attempt-event-1": _provenance(
            command=ATTEMPT_COMMAND, event=ATTEMPT_EVENT,
            aggregate_type="publication", aggregate_id=publication_id, version=2,
        ),
    }
    publication = _Publication()
    runtime = SimpleNamespace(
        authority=_Authority(_events(records)), ingress=object(),
        publication=publication, proof=proof(),
        policies=SimpleNamespace(publication=SimpleNamespace(
            target_path=tmp_path / "serving.sqlite3", target_id="private",
            target_context_digest=_DIGEST,
            editorial_story_command_definition_digest=_DIGEST,
            serving_attempt_command_definition_digest=_DIGEST,
        )),
    )
    monkeypatch.setattr(
        "newsroom.control_plane.native_publication.open_private_serving_read_port",
        lambda *_args, **_kwargs: _Reader(),
    )
    continuation = NativePublicationContinuation(
        journal=journal, runtime=runtime,
        evidence_controller=object.__new__(NativeEvidenceController),
        sources={first.revision_id: (_source(first),), second.revision_id: (_source(second),)},
        clock=lambda: UtcTimestamp.parse("2026-09-08T12:05:00Z"),
    )

    continuation.advance(
        revision_id=second.revision_id, candidate_version_id="candidate-version-2"
    )
    first_ordinal = journal.current(second.revision_id)["ordinal"]
    continuation.advance(
        revision_id=second.revision_id, candidate_version_id="candidate-version-2"
    )

    assert publication.calls == 2
    assert {
        (request["expected_story_version"], request["expected_publication_version"])
        for request in publication.requests
    } == {(1, 2)}
    facts = journal.current(second.revision_id)["facts"]
    assert facts["candidate_id"] == "candidate"
    assert facts["expected_story_version"] == 1
    assert facts["expected_publication_version"] == 2
    assert facts["expected_delivery_evidence_version"] == 0
    assert journal.current(second.revision_id)["ordinal"] == first_ordinal
    connection.close()


def test_same_candidate_successor_rejects_wrong_prior_attempt_event(
    tmp_path, monkeypatch
) -> None:
    from newsroom.control_plane.native_publication import _aggregate
    from newsroom.increment10.editorial import STORY_COMMAND, STORY_EVENT
    from newsroom.increment10.private_serving import ATTEMPT_COMMAND, ATTEMPT_EVENT

    first = _native()
    second = _native("two")
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((first,))
    journal.land((second,))
    journal.advance(first.revision_id, stage="ACKNOWLEDGED", facts={
        "candidate_id": "candidate", "candidate_version_id": "candidate-version-1",
        "story_event_id": "story-event-1", "publication_event_id": "publication-event-1",
        "delivery_attempt_event_id": "attempt-event-1",
        "delivery_evidence_event_id": "evidence-event-1",
    })
    package_id = ObjectAdmissionId.new()
    decision = _decision(package_id)
    journal.advance(second.revision_id, stage="EVIDENCE_RETAINED", facts={
        "candidate_id": "candidate", "candidate_version_id": "candidate-version-2",
        "intake_receipt_id": "intake-receipt-2",
        "package_admission_id": str(package_id),
        "editorial_decision": json.loads(decision.canonical_bytes()),
    })
    records = {
        "story-event-1": _provenance(
            command=STORY_COMMAND, event=STORY_EVENT, aggregate_type="story",
            aggregate_id=str(_aggregate("story", "candidate")), version=1,
        ),
        "attempt-event-1": _provenance(
            command=ATTEMPT_COMMAND, event=ATTEMPT_EVENT,
            aggregate_type="publication", aggregate_id=str(_aggregate("publication", "other")),
            version=2,
        ),
    }
    publication = _Publication()
    runtime = SimpleNamespace(
        authority=_Authority(_events(records)), ingress=object(), publication=publication,
        proof=proof(), policies=SimpleNamespace(publication=SimpleNamespace(
            target_path=tmp_path / "serving.sqlite3", target_id="private",
            target_context_digest=_DIGEST,
            editorial_story_command_definition_digest=_DIGEST,
            serving_attempt_command_definition_digest=_DIGEST,
        )),
    )
    monkeypatch.setattr(
        "newsroom.control_plane.native_publication.open_private_serving_read_port",
        lambda *_args, **_kwargs: _Reader(),
    )
    continuation = NativePublicationContinuation(
        journal=journal, runtime=runtime,
        evidence_controller=object.__new__(NativeEvidenceController),
        sources={first.revision_id: (_source(first),), second.revision_id: (_source(second),)},
    )

    with pytest.raises(ValueError, match="prior publication authority differs"):
        continuation.advance(
            revision_id=second.revision_id, candidate_version_id="candidate-version-2"
        )

    assert publication.calls == 0
    assert journal.current(second.revision_id)["stage"] == "EVIDENCE_RETAINED"
    connection.close()


@pytest.mark.parametrize("owner_stop", [False, True])
def test_started_acquisition_without_result_holds_without_redispatch(
    tmp_path, monkeypatch, owner_stop
) -> None:
    from newsroom.control_plane.veto import VetoError
    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="CANDIDATE_ADMITTED" if owner_stop else "ASSESSMENT_STARTED", facts={
        "candidate_version_id": "candidate-version",
        "intake_receipt_id": "intake-receipt",
        "assessment_started_at": "2026-09-08T12:01:00Z",
    })
    evidence = object.__new__(NativeEvidenceController)
    monkeypatch.setattr(
        NativeEvidenceController,
        "acquire_and_retain",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            VetoError("owner stop") if owner_stop else AssertionError("redispatched")
        ),
    )
    runtime = SimpleNamespace(
        authority=_Authority(), ingress=object(), publication=_Publication(),
        proof=proof(), policies=SimpleNamespace(publication=object()),
    )
    continuation = NativePublicationContinuation(
        journal=journal, runtime=runtime, evidence_controller=evidence,
        sources={unit.revision_id: (_source(unit),)},
    )

    if owner_stop:
        with pytest.raises(VetoError, match="owner stop"):
            continuation.advance(revision_id=unit.revision_id, candidate_version_id="candidate-version")
        assert journal.current(unit.revision_id)["stage"] == "ACQUISITION_STARTED"
    else:
        result = continuation.advance(
            revision_id=unit.revision_id, candidate_version_id="candidate-version"
        )
        assert result.state == "ASSESSMENT_INTERRUPTED"
        assert journal.current(unit.revision_id)["stage"] == "ASSESSMENT_INTERRUPTED"
    assert runtime.publication.calls == 0
    connection.close()


def test_ack_history_scans_inline_summaries_without_expanding_pairs(tmp_path, monkeypatch):
    from newsroom.tests.test_native_progress import _retrieval_facts

    connection = connect(str(tmp_path / "ack-history.sqlite3"))
    journal = NativeRevisionJournal(connection)
    units = tuple(_native(f"prior-{index}") for index in range(3))
    for index, unit in enumerate(units):
        journal.land((unit,))
        journal.advance(unit.revision_id, stage="ACKNOWLEDGED", facts={
            **_retrieval_facts(), "candidate_id": "candidate" if index < 2 else "unrelated",
            "story_event_id": f"story-{index}", "delivery_attempt_event_id": f"attempt-{index}",
        })
    continuation = NativePublicationContinuation(
        journal=journal, runtime=SimpleNamespace(authority=object(), ingress=object(),
            publication=object(), policies=SimpleNamespace(publication=SimpleNamespace(
                editorial_story_command_definition_digest=_DIGEST,
                serving_attempt_command_definition_digest=_DIGEST)), proof=proof()),
        evidence_controller=object.__new__(NativeEvidenceController), sources={},
    )
    monkeypatch.setattr(journal, "current", lambda _: pytest.fail("ACK scan expanded a cold pair"))
    calls = []

    def prior_event(event_id, **_):
        calls.append(event_id)
        return SimpleNamespace(aggregate_version=int(event_id.rsplit("-", 1)[1]) + 1)

    monkeypatch.setattr(continuation, "_prior_event", prior_event)
    try:
        assert continuation._prior_acknowledged_versions(
            revision_id="new-revision", candidate_id="candidate",
        ) == (2, 2)
        assert calls == ["story-0", "attempt-0", "story-1", "attempt-1"]
    finally:
        connection.close()


@pytest.mark.parametrize("remaining_hold", (False, True))
def test_inventory_consumer_reuses_retained_package_once_without_assessor(tmp_path, monkeypatch, remaining_hold):
    from newsroom.control_plane.admission import WRITE_ADMISSION_POLICY_VERSION, write_admission_revalidation_due
    from newsroom.increment10.editorial import EditorialHold
    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    package_id = ObjectAdmissionId.new()
    decision = _decision(package_id)
    prior_facts = {"candidate_version_id": "candidate-version", "graphiti_receipts": [{}],
        "intake_receipt_id": "already-acknowledged", "reason": "INVALID_SUBSTANTIVE_CLAIM_INVENTORY",
        "package_admission_id": str(package_id), "editorial_decision": json.loads(decision.canonical_bytes()),
        "acquisition_attempt_count": 3, "acquisition_retryable": False,
        "assessment_contract_version": "unchanged-producer-and-consumer",
        "expected_story_version": 0, "expected_publication_version": 0,
        "expected_delivery_evidence_version": 0, "publication_started_at": "2026-09-08T12:00:00Z"}
    journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts=prior_facts)
    monkeypatch.setattr(NativeEvidenceController, "acquire_and_retain",
        lambda *_, **__: pytest.fail("retained inventory repair reached acquisition/assessor"))
    monkeypatch.setattr("newsroom.control_plane.native_publication.open_private_serving_read_port",
        lambda *_, **__: _Reader())
    authority, publication = _Authority(), _Publication()
    if remaining_hold:
        def held(*_, **__):
            publication.calls += 1
            raise EditorialHold(reason="INVALID_SUBSTANTIVE_CLAIM_INVENTORY")
        publication.advance = held
    continuation = NativePublicationContinuation(journal=journal,
        runtime=SimpleNamespace(authority=authority, ingress=object(), publication=publication,
            proof=proof(), policies=SimpleNamespace(publication=SimpleNamespace(
                target_path=tmp_path / "serving.sqlite3", target_id="private", target_context_digest=_DIGEST))),
        evidence_controller=object.__new__(NativeEvidenceController),
        sources={unit.revision_id: (_source(unit),)},
        assessment_contract_version="unchanged-producer-and-consumer",
        clock=lambda: UtcTimestamp.parse("2026-09-08T12:00:00Z"))
    try:
        assert write_admission_revalidation_due(journal.summary(unit.revision_id)["facts"])
        first = continuation.advance(revision_id=unit.revision_id, candidate_version_id="candidate-version")
        assert first.state == ("EVIDENCE_HOLD" if remaining_hold else "ACKNOWLEDGED")
        facts = journal.current(unit.revision_id)["facts"]
        assert facts["write_admission_policy_version"] == WRITE_ADMISSION_POLICY_VERSION
        assert not write_admission_revalidation_due(facts)
        for key in ("package_admission_id", "editorial_decision", "intake_receipt_id", "acquisition_attempt_count", "assessment_contract_version"):
            assert facts[key] == prior_facts[key]
        assert publication.calls == 1 and authority.receives == 0
        if remaining_hold:
            continuation._journal = NativeRevisionJournal(connection)
            continuation.advance(revision_id=unit.revision_id, candidate_version_id="candidate-version")
            assert publication.calls == 1
    finally:
        connection.close()


@pytest.mark.parametrize('scenario', ['eligible', 'missing', 'same-producer', 'journal-mismatch',
    'validation-result', 'reader-error', 'unknown-current', 'stop'])
def test_accounted_old_provider_failure_reclassifies_interrupted_through_normal_current_producer(tmp_path, monkeypatch, scenario):
    from datetime import UTC, datetime
    from newsroom.control_plane.native_assessor import RetainedAssessorResult
    from newsroom.control_plane.veto import VetoError
    unit = _native()
    connection = connect(str(tmp_path / 'private.sqlite3'))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    journal.advance(unit.revision_id, stage='ASSESSMENT_INTERRUPTED', facts={
        'candidate_id': 'candidate', 'candidate_version_id': 'candidate-version',
        'graphiti_receipts': [{}], 'intake_receipt_id': 'already-acknowledged',
        'assessment_contract_version': ('newsroom.native-evidence-assessor.v23+consumer.v1'
            if scenario == 'same-producer' else 'newsroom.native-evidence-assessor.v21+consumer.v1'),
        'failure_class': 'CliTimeoutError', 'reason': 'ACQUISITION_RESULT_NOT_RETAINED',
        'acquisition_attempt_count': 1,
    })
    retained = RetainedAssessorResult(
        RetainedAssessorContractFailure('old-envelope', 'old-invocation', _DIGEST, _DIGEST, _DIGEST),
        'newsroom.native-evidence-assessor.v20' if scenario == 'journal-mismatch' else 'newsroom.native-evidence-assessor.v21',
        _DIGEST, 'ASSESSOR_VALIDATION_FAILED' if scenario == 'validation-result' else 'ASSESSOR_PROVIDER_FAILED',
        datetime(2026, 9, 8, tzinfo=UTC), None,
    )
    before = journal.current(unit.revision_id)
    def recover(_candidate):
        if scenario == 'reader-error': raise OSError('read failed')
        if scenario == 'stop': raise VetoError('owner stop')
        return None if scenario == 'missing' else retained
    calls = []
    def acquire(_self, **request):
        calls.append(request['assessment_cached_only'])
        assert request['intake_receipt_id'] == 'already-acknowledged'
        request['before_assessment']()
        raise NativeEvidenceHold('NO_QUALIFYING_NEW_INFORMATION', unit.source_id)
    monkeypatch.setattr(NativeEvidenceController, 'acquire_and_retain', acquire)
    continuation = NativePublicationContinuation(
        journal=journal, runtime=SimpleNamespace(authority=_Authority(), ingress=object(),
            publication=_Publication(), proof=proof(), policies=SimpleNamespace(publication=object())),
        evidence_controller=object.__new__(NativeEvidenceController), sources={unit.revision_id: (_source(unit),)},
        assessment_old_provider_failure=recover,
        assessment_contract_version=('newsroom.native-evidence-assessor.v24+consumer.v1'
            if scenario == 'unknown-current' else 'newsroom.native-evidence-assessor.v23+consumer.v1'),
        clock=lambda: UtcTimestamp.parse('2026-09-08T12:00:00Z'),
    )
    try:
        if scenario != 'eligible':
            if scenario == 'stop':
                with pytest.raises(VetoError):
                    continuation.advance(revision_id=unit.revision_id, candidate_version_id='candidate-version')
            else:
                result = continuation.advance(revision_id=unit.revision_id, candidate_version_id='candidate-version')
                assert result.state == 'ASSESSMENT_INTERRUPTED'
            assert calls == []
            assert journal.current(unit.revision_id) == before
            return
        for _ in range(2):
            result = continuation.advance(revision_id=unit.revision_id, candidate_version_id='candidate-version')
            assert result.reason == 'NO_QUALIFYING_NEW_INFORMATION'
        assert calls == [False]
        facts = journal.current(unit.revision_id)['facts']
        assert facts['assessment_superseded']['failure_class'] == 'CliTimeoutError'
        assert facts['assessment_superseded']['provider_failure']['invocation_id'] == 'old-invocation'
        assert facts['assessment_contract_version'] == 'newsroom.native-evidence-assessor.v23+consumer.v1'
    finally:
        connection.close()


@pytest.mark.parametrize('later_sibling', (False, True))
def test_stale_prepared_intent_retains_paired_ack_proof_across_reopen(tmp_path, monkeypatch, later_sibling):
    from newsroom.increment10.editorial import EditorialError

    first, incoming, later = _native(), _native('stale-successor'), _native('later-sibling')
    path = str(tmp_path / 'stale-progress.sqlite3')
    connection = connect(path)
    journal = NativeRevisionJournal(connection)
    journal.land((first,))
    journal.land((incoming,))
    journal.land((later,))
    sibling = dict(candidate_id='candidate', candidate_version_id='prior-version',
        story_event_id='corrected-story', publication_event_id='corrected-publication',
        delivery_attempt_event_id='corrected-attempt', delivery_evidence_event_id='corrected-evidence')
    journal.advance(first.revision_id, stage='ACKNOWLEDGED', facts=sibling)
    package_id = ObjectAdmissionId.new()
    original_decision = json.loads(_decision(package_id).canonical_bytes())
    journal.advance(incoming.revision_id, stage='PUBLICATION_STARTED', facts=dict(
        candidate_id='candidate', candidate_version_id='candidate-version', intake_receipt_id='incoming-intake',
        package_admission_id=str(package_id), editorial_decision=original_decision,
        expected_story_version=1, expected_publication_version=2, expected_delivery_evidence_version=0,
        publication_started_at='2026-09-08T12:04:00Z'))

    class Publication(_Publication):
        reconciliations = 0
        interrupted = True

        def reconcile_stale_intent(self, selected, **request):
            self.reconciliations += 1
            assert selected == package_id
            if (request['expected_story_version'], request['expected_publication_version']) == (2, 4):
                if not later_sibling:
                    return None  # Own partial slot; no newer authenticated sibling.
                selected = next(row for row in request['acknowledged'] if row['story_event_id'] == 'later-story')
                versions = (3, 6)
            else:
                assert (request['expected_story_version'], request['expected_publication_version']) == (1, 2)
                assert request['acknowledged'] == (sibling,)
                selected, versions = sibling, (2, 4)
            return SimpleNamespace(story_receipt=SimpleNamespace(aggregate_version=versions[0]),
                attempt_receipt=SimpleNamespace(aggregate_version=versions[1])), {key: selected[key] for key in (
                    'story_event_id', 'publication_event_id', 'delivery_attempt_event_id', 'delivery_evidence_event_id')}

        def advance(self, *args, **request):
            assert (request['expected_story_version'], request['expected_publication_version'],
                    request['expected_delivery_evidence_version']) == ((3, 6, 0) if later_sibling and not self.interrupted else (2, 4, 0))
            assert request['reconciled_predecessor']['story_event_id'] == ('later-story' if later_sibling and not self.interrupted else 'corrected-story')
            if self.interrupted:
                self.interrupted = False
                raise EditorialError('interrupted before publication')
            return super().advance(*args, **request)

    publication = Publication()
    runtime = SimpleNamespace(authority=_Authority(), ingress=object(), publication=publication, proof=proof(),
        policies=SimpleNamespace(publication=SimpleNamespace(target_path=tmp_path / 'serving.sqlite3',
            target_id='private', target_context_digest=_DIGEST)))
    monkeypatch.setattr('newsroom.control_plane.native_publication.open_private_serving_read_port',
        lambda *_args, **_kwargs: _Reader())
    build = lambda journal: NativePublicationContinuation(journal=journal, runtime=runtime,
        evidence_controller=object.__new__(NativeEvidenceController), sources={incoming.revision_id: (_source(incoming),)},
        clock=lambda: UtcTimestamp.parse('2026-09-08T12:05:00Z'))
    with pytest.raises(EditorialError, match='interrupted'):
        build(journal).advance(revision_id=incoming.revision_id, candidate_version_id='candidate-version')
    frozen = journal.current(incoming.revision_id)
    assert frozen['facts']['editorial_decision'] == original_decision
    assert frozen['facts']['package_admission_id'] == str(package_id)
    if later_sibling:
        journal.advance(later.revision_id, stage='ACKNOWLEDGED', facts={**sibling,
            'story_event_id': 'later-story', 'publication_event_id': 'later-publication',
            'delivery_attempt_event_id': 'later-attempt', 'delivery_evidence_event_id': 'later-evidence'})
    connection.close()
    connection = connect(path)
    journal = NativeRevisionJournal(connection)
    result = build(journal).advance(revision_id=incoming.revision_id, candidate_version_id='candidate-version')
    assert result.state == 'ACKNOWLEDGED'
    assert publication.reconciliations == 2 and publication.calls == 1
    assert journal.current(first.revision_id)['facts'] == sibling
    connection.close()


@pytest.mark.parametrize('resumed', [False, True, 'completed-v1', 'settled-consumer-failure', 'consumer-failure-loop'])
def test_distinct_context_package_preserves_old_writer_and_semantic_intent(tmp_path,monkeypatch,resumed):
    unit=_native();connection=connect(str(tmp_path/'context-purpose.sqlite3'))
    journal=NativeRevisionJournal(connection);journal.land((unit,))
    old_package=ObjectAdmissionId.new();new_package=ObjectAdmissionId.new()
    semantic={'contract':'old-semantic','input_digest':_DIGEST,'origin_invocation_id':'original-unknown'}
    old={'candidate_id':'candidate','candidate_version_id':'candidate-version','graphiti_receipts':[{}],
        'intake_receipt_id':'original-intake','package_admission_id':str(old_package),
        'editorial_decision':json.loads(_decision(old_package).canonical_bytes()),
        'semantic_assessment_intent':semantic,'reason':'NATIVE_STORY_FACTUAL_ENTITIES',
        'publication_started_at':'original-writer-purpose','acquisition_attempt_count':2}
    if resumed in (True, 'settled-consumer-failure', 'consumer-failure-loop'):
        old.update(context_enrichment_intent={'contract':'newsroom.native-context-package.v2',
            'original_package_admission_id':str(old_package),
            'original_journal':{'publication_started_at':old.pop('publication_started_at')}},
            reason='ACQUISITION_RESULT_NOT_RETAINED', failure_class='EvidencePackageError',
            assessment_contract_version='newsroom.native-evidence-assessor.v23+previous-consumer')
        old.pop('package_admission_id');old.pop('editorial_decision')
        if resumed == 'settled-consumer-failure':
            old.update(context_enrichment_settled='newsroom.native-context-package.v2',
                reason='SEMANTIC_INTENT_INPUT_CHANGED_HOLD')
    if resumed == 'completed-v1':
        old.update(context_enrichment_intent={'contract':'newsroom.native-context-package.v1',
            'original_package_admission_id':'original-before-context'},
            context_enrichment_completed=True)
    journal.advance(unit.revision_id,stage='ASSESSMENT_INTERRUPTED' if resumed in (True, 'settled-consumer-failure', 'consumer-failure-loop') else 'EVIDENCE_HOLD',facts=old)
    requests=[]
    def acquire(_self,**request):
        requests.append(request)
        assert request['assessment_context_only']is True
        assert request['assessment_cached_only']is False
        assert not request.get('assessment_qualification_cached_only')
        assert 'assessment_semantic_only'not in request
        request['before_assessment']()
        if resumed == 'consumer-failure-loop':
            from newsroom.increment10.evidence import EvidencePackageError
            raise EvidencePackageError('fixture unchanged consumer rejection')
        return SimpleNamespace(retained=SimpleNamespace(package_admission_id=new_package),
            editorial_decision=_decision(new_package),acquisition_receipt_digests=(_DIGEST,))
    monkeypatch.setattr(NativeEvidenceController,'acquire_and_retain',acquire)
    monkeypatch.setattr('newsroom.control_plane.native_publication.open_private_serving_read_port',lambda *_a,**_k:_Reader())
    publication=_Publication();runtime=SimpleNamespace(authority=_Authority(),ingress=object(),
        publication=publication,proof=proof(),policies=SimpleNamespace(publication=SimpleNamespace(
            target_path=tmp_path/'serving.sqlite3',target_id='private',target_context_digest=_DIGEST)))
    with pytest.raises(ValueError,match='composition differs'):
        NativePublicationContinuation(journal=journal,runtime=runtime,
            evidence_controller=object.__new__(NativeEvidenceController),sources={unit.revision_id:(_source(unit),)},
            context_enrichment_contract='newsroom.native-context-package.v99')
    continuation=NativePublicationContinuation(journal=journal,runtime=runtime,
        evidence_controller=object.__new__(NativeEvidenceController),sources={unit.revision_id:(_source(unit),)},
        context_enrichment_contract='newsroom.native-context-package.v2',
        semantic_origin_failure=lambda _:None, semantic_intent_contract='old-semantic',
        assessment_contract_version='newsroom.native-evidence-assessor.v23+new-consumer')
    continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
    facts=journal.current(unit.revision_id)['facts']
    if resumed == 'consumer-failure-loop':
        assert facts['context_enrichment_settled'] == 'newsroom.native-context-package.v2'
        assert facts['failure_class'] == 'EvidencePackageError'
        assert not continuation.context_enrichment_due(facts)
        assert facts['semantic_assessment_intent'] == semantic
        assert len(requests) == 1 and publication.calls == 0
        connection.close()
        return
    assert facts['context_enrichment_intent']['original_package_admission_id']==str(old_package)
    assert facts['context_enrichment_intent']['original_journal']['publication_started_at']=='original-writer-purpose'
    assert facts['semantic_assessment_intent']==semantic and facts['acquisition_attempt_count']==2
    assert facts['context_enrichment_completed']is True and facts['package_admission_id']==str(new_package)
    if resumed == 'completed-v1':
        prior = facts['context_enrichment_intent_history'][0]
        assert prior['intent']['contract'] == 'newsroom.native-context-package.v1'
        assert prior['intent']['original_package_admission_id'] == 'original-before-context'
        assert prior['completed'] is True and prior['package_admission_id'] == str(old_package)
        assert facts['context_enrichment_intent']['contract'] == 'newsroom.native-context-package.v2'
    assert journal.current(unit.revision_id)['stage']=='ACKNOWLEDGED'
    assert len(requests)==publication.calls==1
    assert runtime.authority.receives==0
    connection.close()


@pytest.mark.parametrize('already_checked',[False,True])
def test_story_consumer_revalidation_only_reads_retained_copy_once(tmp_path,monkeypatch,already_checked):
    from newsroom.control_plane.native_story_writer import CONSUMER_VERSION
    unit=_native();connection=connect(str(tmp_path/'writer-consumer.sqlite3'))
    journal=NativeRevisionJournal(connection);journal.land((unit,))
    package_id=ObjectAdmissionId.new();decision=_decision(package_id)
    facts={'candidate_id':'candidate','candidate_version_id':'candidate-version',
        'graphiti_receipts':[{}],'intake_receipt_id':'old-intake',
        'package_admission_id':str(package_id),'editorial_decision':json.loads(decision.canonical_bytes()),
        'reason':'NATIVE_STORY_SENTENCE_SUPPORT','expected_story_version':0,
        'expected_publication_version':0,'expected_delivery_evidence_version':0}
    if already_checked:facts['writer_support_checked_version']=CONSUMER_VERSION
    journal.advance(unit.revision_id,stage='EVIDENCE_HOLD',facts=facts)
    class CachedPublication(_Publication):
        def advance(self,*args,**kwargs):
            assert kwargs['story_cached_only']is True
            return super().advance(*args,**kwargs)
    publication=CachedPublication();runtime=SimpleNamespace(authority=_Authority(),ingress=object(),
        publication=publication,proof=proof(),policies=SimpleNamespace(publication=SimpleNamespace(
            target_path=tmp_path/'serving.sqlite3',target_id='private',target_context_digest=_DIGEST)))
    monkeypatch.setattr(NativeEvidenceController,'acquire_and_retain',lambda *_args,**_kwargs:pytest.fail('reacquired package'))
    monkeypatch.setattr('newsroom.control_plane.native_publication.open_private_serving_read_port',lambda *_args,**_kwargs:_Reader())
    continuation=NativePublicationContinuation(journal=journal,runtime=runtime,
        evidence_controller=object.__new__(NativeEvidenceController),sources={unit.revision_id:(_source(unit),)})
    continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
    assert publication.calls==(0 if already_checked else 1)
    if not already_checked:
        assert journal.current(unit.revision_id)['stage']=='ACKNOWLEDGED'
        assert journal.current(unit.revision_id)['facts']['writer_support_checked_version']==CONSUMER_VERSION
    assert runtime.authority.receives==0
    connection.close()


@pytest.mark.parametrize('fault', [None, 'accepted', 'changed-clock', 'structural-value', 'rendering', 'unknown', 'same-consumer', 'already-checked', 'pending-effect'])
def test_known_qualification_consumer_resume_never_restarts_semantic_input(tmp_path, monkeypatch, fault):
    from newsroom.control_plane.native_evidence import NativeEvidenceHold
    unit = _native()
    connection = connect(str(tmp_path / 'known-qualification.sqlite3'))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    contract='newsroom.native-assessor-judgments.v2+newsroom.native-source-qualification.v2'
    current='newsroom.native-evidence-assessor.v23+source-qualification-consumer.v4'
    intent={'contract':contract,'input_digest':_DIGEST,'origin_invocation_id':'original-unknown',
            'origin_journal':{'reason':'ACQUISITION_RESULT_NOT_RETAINED'}}
    facts={'candidate_id':'candidate','candidate_version_id':'candidate-version',
        'graphiti_receipts':[{}],'intake_receipt_id':'existing-intake',
        'semantic_assessment_intent':intent,'assessment_contract_version':current.replace('source-qualification-consumer.v4','source-qualification-consumer.v1'),
        'reason':'ACQUISITION_RESULT_NOT_RETAINED','failure_class':'EvidencePackageError',
        'assessment_started_at':'2026-10-05T07:33:07Z',
        'semantic_acquisition_attempt_count':2,'acquisition_attempt_count':3}
    if fault=='unknown':facts['failure_class']='CliTimeoutError'
    if fault=='same-consumer':facts['assessment_contract_version']=current
    if fault=='already-checked':facts['retained_qualification_checked_contract']=current
    if fault=='pending-effect':facts['publication_started_at']='pending'
    if fault=='changed-clock':facts['reason']='SEMANTIC_INTENT_INPUT_CHANGED_HOLD'
    if fault=='structural-value':facts.update(reason='SEMANTIC_INTENT_INPUT_CHANGED_HOLD',failure_class='ValueError')
    if fault=='rendering':facts.update(reason='ASSESSOR_RENDERING_CONTRACT_HOLD',failure_class='stale-ModelUsageAdmissionError')
    journal.advance(unit.revision_id,stage='EVIDENCE_HOLD'if fault in {'changed-clock','structural-value','rendering'}else'ASSESSMENT_INTERRUPTED',facts=facts)
    calls=[]
    package_id=ObjectAdmissionId.new()
    def acquire(_self, **request):
        calls.append(request)
        assert request['assessment_cached_only'] is True
        assert request['assessment_qualification_cached_only'] is True
        assert 'assessment_semantic_only' not in request
        assert 'before_semantic_assessment' not in request
        request['before_assessment']()
        if fault=='accepted':
            return SimpleNamespace(retained=SimpleNamespace(package_admission_id=package_id),
                editorial_decision=_decision(package_id),acquisition_receipt_digests=(_DIGEST,))
        raise NativeEvidenceHold('QUALIFICATION_RETAINED_RESULT_HOLD',unit.source_id)
    monkeypatch.setattr(NativeEvidenceController,'acquire_and_retain',acquire)
    monkeypatch.setattr('newsroom.control_plane.native_publication.open_private_serving_read_port',lambda *_args,**_kwargs:_Reader())
    runtime=SimpleNamespace(authority=_Authority(),ingress=object(),publication=_Publication(),
        policies=SimpleNamespace(publication=SimpleNamespace(target_path=tmp_path/'serving.sqlite3',
            target_id='private',target_context_digest=_DIGEST)),proof=proof())
    continuation=NativePublicationContinuation(journal=journal,runtime=runtime,
        evidence_controller=object.__new__(NativeEvidenceController),sources={unit.revision_id:(_source(unit),)},
        semantic_origin_failure=lambda _:None,semantic_intent_contract=contract,assessment_contract_version=current)
    continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
    after=journal.current(unit.revision_id)['facts']
    assert len(calls)==(1 if fault in (None,'accepted','changed-clock','structural-value','rendering') else 0), after
    assert after['semantic_assessment_intent']==intent
    assert after['semantic_acquisition_attempt_count']==2
    assert after['assessment_started_at']==facts['assessment_started_at']
    if fault in (None,'structural-value','rendering'):
        assert after['retained_qualification_checked_contract']==current
        continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
        assert len(calls)==1
    if fault=='accepted':
        assert journal.current(unit.revision_id)['stage']=='ACKNOWLEDGED'
    assert runtime.authority.receives==0 and runtime.publication.calls==(1 if fault=='accepted'else 0)
    connection.close()


@pytest.mark.parametrize('scenario', ['eligible', 'missing-origin', 'missing-graph', 'pending-publication', 'wrong-origin', 'changed-input', 'same-input', 'question-upgrade'])
def test_separate_semantic_continuation_retains_original_interruption_and_never_retries_legacy(tmp_path, monkeypatch, scenario):
    from datetime import UTC, datetime
    from newsroom.authority.canonical import digest_bytes
    from newsroom.control_plane.native_assessor import RetainedAssessorResult
    unit = _native()
    path = str(tmp_path / 'separate-intent.sqlite3')
    connection = connect(path)
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    old = {'candidate_id':'candidate','candidate_version_id':'candidate-version',
           'graphiti_receipts':[{}], 'intake_receipt_id':'already-acknowledged',
           'assessment_contract_version':'newsroom.native-evidence-assessor.v23+consumer.v1',
           'assessment_started_at':'2026-09-08T12:00:00Z',
           'acquisition_started_at':'2026-09-08T11:59:00Z', 'acquisition_attempt_count':3,
           'reason':'ACQUISITION_RESULT_NOT_RETAINED','failure_class':'CliTimeoutError'}
    if scenario == 'missing-graph': old['graphiti_receipts'] = []
    if scenario == 'pending-publication': old['publication_started_at']='pending-effect'
    journal.advance(unit.revision_id,stage='ASSESSMENT_INTERRUPTED',facts=old)
    origin=RetainedAssessorResult(RetainedAssessorContractFailure('original-envelope','original-invocation',_DIGEST,_DIGEST,_DIGEST),
        'newsroom.native-evidence-assessor.v21' if scenario=='wrong-origin' else 'newsroom.native-evidence-assessor.v23',
        _DIGEST,'ASSESSOR_PROVIDER_FAILED',datetime(2026,9,8,tzinfo=UTC),None)
    if scenario=='question-upgrade':
        old_intent={'contract':'newsroom.native-assessor-judgments.v1','candidate_version_id':'candidate-version',
            'origin_envelope_id':origin.proof.envelope_id,'origin_invocation_id':origin.proof.invocation_id,
            'origin_allocation_digest':origin.proof.allocation_digest,'origin_terminal_digest':origin.proof.terminal_digest,
            'origin_context_manifest_digest':origin.proof.context_manifest_digest,
            'origin_journal':dict(old),'input_digest':_DIGEST}
        journal.advance(unit.revision_id,stage='ASSESSMENT_INTERRUPTED',facts={**old,'semantic_assessment_intent':old_intent})
    calls=[]
    paid_calls=[]
    def acquire(_self, **request):
        assert request['assessment_semantic_only'] is True
        assert request['assessment_cached_only'] is False
        assert request['intake_receipt_id'] == old['intake_receipt_id']
        retained=journal.current(unit.revision_id)['facts']['semantic_assessment_intent']
        assert retained['origin_invocation_id'] == 'original-invocation'
        assert retained['origin_journal']['assessment_started_at'] == old['assessment_started_at']
        changed = scenario == 'changed-input' and bool(calls)
        acquired = _meaning_acquisition(unit)
        if changed:
            acquired = replace(acquired, body=b'Changed official source.', body_digest=digest_bytes(b'Changed official source.'),
                receipt_digest=digest_canonical({**json.loads(acquired.receipt_bytes), 'body_digest':digest_bytes(b'Changed official source.')}))
        request['before_semantic_assessment'](SimpleNamespace(digest=_DIGEST), (acquired,))
        request['before_assessment']()
        calls.append(request)
        if scenario in {'changed-input','same-input'}:
            if not paid_calls:
                paid_calls.append('one stable compound intent')
            raise RuntimeError('new semantic allocation remains unknown; no dispatch on replay')
        # The real semantic assessor's negative path is tested with actual
        # ModelUsage+CAS; this seam proves it never falls back to legacy dispatch.
        raise NativeEvidenceHold('SEMANTIC_INTENT_FALLBACK_HOLD',unit.source_id)
    monkeypatch.setattr(NativeEvidenceController,'acquire_and_retain',acquire)
    def continuation():
        return NativePublicationContinuation(journal=journal,
            runtime=SimpleNamespace(authority=_Authority(),ingress=object(),publication=_Publication(),
                proof=proof(),policies=SimpleNamespace(publication=object())),
            evidence_controller=object.__new__(NativeEvidenceController),sources={unit.revision_id:(_source(unit),)},
            semantic_origin_failure=lambda _:None if scenario=='missing-origin' else origin,
            semantic_intent_contract='newsroom.native-assessor-judgments.v2' if scenario=='question-upgrade' else 'newsroom.native-assessor-judgments.v1',
            assessment_contract_version=old['assessment_contract_version'],
            clock=lambda:UtcTimestamp.parse('2026-09-08T12:30:00Z'))
    result=continuation().advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
    if scenario in {'changed-input','same-input'}:
        assert result.state == 'ASSESSMENT_INTERRUPTED' and len(paid_calls)==1
        connection.close()
        connection=connect(path)
        journal=NativeRevisionJournal(connection)
        result=continuation().advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
        assert len(paid_calls)==1
        if scenario=='changed-input':
            assert result.reason=='SEMANTIC_INTENT_INPUT_CHANGED_HOLD'
            assert len(calls)==1
        else:
            assert result.reason=='SEMANTIC_ASSESSMENT_ALREADY_ATTEMPTED_HOLD' and len(calls)==1
    elif scenario not in {'eligible','question-upgrade'}:
        assert result.state=='ASSESSMENT_INTERRUPTED' and calls==[]
        assert journal.current(unit.revision_id)['facts']==old
    else:
        assert result.reason=='SEMANTIC_INTENT_FALLBACK_HOLD' and len(calls)==1
        assert journal.current(unit.revision_id)['facts']['acquisition_attempt_count']==3
        assert journal.current(unit.revision_id)['facts']['semantic_acquisition_attempt_count']==1
        if scenario=='question-upgrade':
            current=journal.current(unit.revision_id)['facts']
            assert current['semantic_assessment_intent_history']==[old_intent]
            assert current['semantic_assessment_intent']['contract']=='newsroom.native-assessor-judgments.v2'
        connection.close()
        connection=connect(path)
        journal=NativeRevisionJournal(connection)
        assert continuation().advance(revision_id=unit.revision_id,candidate_version_id='candidate-version').reason=='SEMANTIC_INTENT_FALLBACK_HOLD'
        assert len(calls)==1
    connection.close()


@pytest.mark.parametrize('previous,current,due',[
    ('newsroom.native-assessor-judgments.v1','newsroom.native-assessor-judgments.v2+newsroom.native-source-qualification.v2',True),
    ('newsroom.native-assessor-judgments.v2','newsroom.native-assessor-judgments.v2+newsroom.native-source-qualification.v2',True),
    ('newsroom.native-assessor-judgments.v2+newsroom.native-source-qualification.v2','newsroom.native-assessor-judgments.v2+newsroom.native-source-qualification.v2',False),
    ('newsroom.native-assessor-judgments.v2','unqualified-or-arbitrary-contract',False),
])
def test_public_context_compound_upgrade_is_exact_and_stable(previous,current,due):
    facts={'semantic_assessment_intent':{'contract':previous},'graphiti_receipts':[{}],
           'intake_receipt_id':'original-intake','reason':'NO_QUALIFYING_NEW_INFORMATION'}
    assert NativePublicationContinuation.semantic_intent_revalidation_due(facts,current)is due
    facts['publication_started_at']='pending'
    assert not NativePublicationContinuation.semantic_intent_revalidation_due(facts,current)


@pytest.mark.parametrize('stage_reason', ['ACQUISITION_RESULT_NOT_RETAINED', 'CONTEXT_PROVIDER_UNKNOWN_HOLD'])
def test_old_unresolved_context_does_not_receive_a_new_purpose(stage_reason):
    facts = {'context_enrichment_intent': {'contract': 'newsroom.native-context-package.v1',
             'original_package_admission_id': 'old'}, 'reason': stage_reason,
             'graphiti_receipts': [{}], 'intake_receipt_id': 'receipt',
             'semantic_assessment_intent': {'origin_invocation_id': 'unknown'}}
    before = json.loads(json.dumps(facts))
    assert not NativePublicationContinuation.context_enrichment_due(facts)
    assert facts == before


@pytest.mark.parametrize('reason,checked,due', [('CONTEXT_SUPPORT_UNPROVEN_HOLD', None, True),
    ('CONTEXT_SUPPORT_UNPROVEN_HOLD', 'newsroom.native-context-support.assembled.v1', False),
    ('CONTEXT_SELECTION_UNCERTAIN_HOLD', None, False)])
def test_only_closed_old_support_judgement_is_scheduled_for_assembled_input(reason, checked, due):
    facts = {'context_enrichment_intent': {'contract':'newsroom.native-context-package.v2',
             'original_package_admission_id':'original'}, 'context_enrichment_settled':'newsroom.native-context-package.v2',
             'reason':reason,'context_support_checked_contract':checked}
    assert NativePublicationContinuation.context_enrichment_due(facts) is due
    facts['publication_event_id'] = 'pending-publication'
    assert not NativePublicationContinuation.context_enrichment_due(facts)


@pytest.mark.parametrize('override,due', [({}, True),
    ({'context_consumer_checked_version': 'newsroom.native-context-materialisation.v3'}, False),
    ({'context_enrichment_completed': True}, False),
    ({'failure_class': 'TimeoutError'}, False),
    ({'reason': 'CONTEXT_SUPPORT_UNPROVEN_HOLD', 'context_support_checked_contract': 'newsroom.native-context-support.assembled.v1'}, False),
    ({'reason': 'CONTEXT_SELECTION_UNCERTAIN_HOLD'}, False),
    ({'publication_event_id': 'pending'}, False)])
def test_settled_context_consumer_failure_replays_only_changed_consumer(override, due):
    facts = {'context_enrichment_intent': {'contract': 'newsroom.native-context-package.v2',
             'original_package_admission_id': 'retained-original'},
             'context_enrichment_settled': 'newsroom.native-context-package.v2',
             'failure_class': 'EvidencePackageError', 'reason': 'SEMANTIC_INTENT_INPUT_CHANGED_HOLD',
             **override}
    before = json.loads(json.dumps(facts))
    assert NativePublicationContinuation.context_enrichment_due(facts) is due
    assert facts == before


@pytest.mark.parametrize('checked,due', [('newsroom.native-story-support.v3', True),
                                       ('newsroom.native-story-support.v4', False)])
def test_source_publisher_display_consumer_revalidates_retained_writer_once(checked, due):
    facts = {'reason': 'NATIVE_STORY_FACTUAL_ENTITIES',
             'editorial_hold_reason_codes': ['NATIVE_STORY_FACTUAL_ENTITIES'],
             'package_admission_id': 'original-retained-package', 'editorial_decision': {},
             'writer_support_checked_version': checked}
    before = json.loads(json.dumps(facts))
    assert NativePublicationContinuation.writer_revalidation_due(facts) is due
    assert facts == before
    facts['publication_event_id'] = 'pending'
    assert not NativePublicationContinuation.writer_revalidation_due(facts)


@pytest.mark.parametrize('scenario', ['eligible', 'v17', 'missing-origin', 'wrong-producer',
    'unknown-origin', 'missing-raw', 'missing-graph', 'pending-effect', 'settled-intent', 'source-hold'])
def test_reported_validation_origin_survives_allocation_denial_and_uses_distinct_semantics(tmp_path, monkeypatch, scenario):
    from datetime import UTC, datetime
    from newsroom.control_plane.native_assessor import RetainedAssessorResult
    unit = _native()
    connection = connect(str(tmp_path / 'validation-semantic.sqlite3'))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    old_contract = 'newsroom.native-evidence-assessor.v17' if scenario == 'v17' else 'newsroom.native-evidence-assessor.v15'
    original = {'candidate_id': 'candidate', 'candidate_version_id': 'candidate-version',
        'graphiti_receipts': [{}], 'intake_receipt_id': 'original-intake',
        'assessment_contract_version': 'newsroom.native-evidence-assessor.v19+consumer.v1',
        'assessment_superseded': {'contract_version': old_contract + '+consumer.v1',
            'reason': 'ASSESSOR_RENDERING_CONTRACT_HOLD', 'package_admission_id': None,
            'editorial_decision_id': None, 'acquisition_attempt_count': 1},
        'acquisition_attempt_count': 1, 'reason': 'ACQUISITION_RESULT_NOT_RETAINED',
        'failure_class': 'ModelUsageAdmissionError'}
    if scenario == 'wrong-producer': original['assessment_superseded']['contract_version'] = 'newsroom.native-evidence-assessor.v16'
    if scenario == 'missing-graph': original['graphiti_receipts'] = []
    if scenario == 'pending-effect': original['publication_started_at'] = 'unknown-effect'
    journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=original)
    origin = RetainedAssessorResult(RetainedAssessorContractFailure('old-envelope', 'old-invocation', _DIGEST, _DIGEST, _DIGEST),
        old_contract, _DIGEST, 'ASSESSOR_PROVIDER_FAILED' if scenario == 'unknown-origin' else 'ASSESSOR_VALIDATION_FAILED',
        datetime(2026, 9, 8, tzinfo=UTC), None, result_digest=None if scenario == 'missing-raw' else _DIGEST,
        result_receipt_digest=_DIGEST)
    calls = []
    def acquire(_self, **request):
        assert request['assessment_semantic_only'] is True and request['assessment_cached_only'] is False
        intent = journal.current(unit.revision_id)['facts']['semantic_assessment_intent']
        assert intent['origin_invocation_id'] == 'old-invocation'
        assert intent['origin_outcome'] == 'ASSESSOR_VALIDATION_FAILED'
        assert intent['origin_result_digest'] == _DIGEST and intent['origin_result_receipt_digest'] == _DIGEST
        assert intent['origin_journal']['assessment_superseded'] == original['assessment_superseded']
        calls.append(request)
        request['before_semantic_assessment'](SimpleNamespace(digest=_DIGEST), (_meaning_acquisition(unit),))
        raise NativeEvidenceHold('CURRENT_RIGHTS_HOLD' if scenario == 'source-hold' else 'NO_QUALIFYING_NEW_INFORMATION', unit.source_id)
    monkeypatch.setattr(NativeEvidenceController, 'acquire_and_retain', acquire)
    runtime = SimpleNamespace(authority=_Authority(), ingress=object(), publication=_Publication(),
        proof=proof(), policies=SimpleNamespace(publication=object()))
    continuation = NativePublicationContinuation(journal=journal, runtime=runtime,
        evidence_controller=object.__new__(NativeEvidenceController), sources={unit.revision_id: (_source(unit),)},
        semantic_origin_failure=lambda _: None if scenario == 'missing-origin' else origin,
        semantic_intent_contract='newsroom.native-assessor-judgments.v2+newsroom.native-source-qualification.v2',
        assessment_contract_version='newsroom.native-evidence-assessor.v23+consumer.v1',
        clock=lambda: UtcTimestamp.parse('2026-09-08T12:30:00Z'))
    try:
        before = journal.current(unit.revision_id)
        checked = continuation.recover_pre_dispatch((unit.revision_id,),
            denial_many=lambda *_args, **_kwargs: (True,),
            failure_many=lambda _: pytest.fail('allocated origin was mistaken for zero dispatch'),
            before_revision=lambda: True)
        # The denial is preserved; prospective validation is authenticated only
        # by ordinary continuation, not manufactured by the journal shortcut.
        assert checked == (() if scenario not in {'missing-graph', 'pending-effect'} else (unit.revision_id,))
        assert journal.current(unit.revision_id) == before
        result = continuation.advance(revision_id=unit.revision_id, candidate_version_id='candidate-version')
        if scenario in {'eligible', 'v17', 'settled-intent', 'source-hold'}:
            assert len(calls) == 1
            assert result.reason == ('CURRENT_RIGHTS_HOLD' if scenario == 'source-hold' else 'NO_QUALIFYING_NEW_INFORMATION')
            retained = journal.current(unit.revision_id)['facts']
            assert retained['assessment_superseded'] == original['assessment_superseded']
            assert retained['acquisition_attempt_count'] == 1 and retained['semantic_acquisition_attempt_count'] == 1
            if scenario != 'source-hold':
                assert continuation.advance(revision_id=unit.revision_id, candidate_version_id='candidate-version').reason == 'NO_QUALIFYING_NEW_INFORMATION'
                assert len(calls) == 1
        else:
            assert calls == [] and result.reason == 'ACQUISITION_RESULT_NOT_RETAINED'
            assert journal.current(unit.revision_id)['facts'] == original
        assert runtime.authority.receives == 0 and runtime.publication.calls == 0
    finally:
        connection.close()


@pytest.mark.parametrize('boundary', ['missing-source', 'stop', 'origin-mismatch'])
def test_reported_validation_semantic_origin_preserves_live_source_stop_and_exact_intent(tmp_path, monkeypatch, boundary):
    from datetime import UTC, datetime
    from newsroom.control_plane.native_assessor import RetainedAssessorResult
    from newsroom.control_plane.native_publication import NativePublicationError
    from newsroom.control_plane.veto import VetoError
    unit = _native()
    connection = connect(str(tmp_path / 'validation-boundaries.sqlite3'))
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    origin = RetainedAssessorResult(RetainedAssessorContractFailure('old-envelope', 'old-invocation', _DIGEST, _DIGEST, _DIGEST),
        'newsroom.native-evidence-assessor.v15', _DIGEST, 'ASSESSOR_VALIDATION_FAILED',
        datetime(2026, 9, 8, tzinfo=UTC), None, _DIGEST, _DIGEST)
    contract = 'newsroom.native-assessor-judgments.v2+newsroom.native-source-qualification.v2'
    facts = {'candidate_id': 'candidate', 'candidate_version_id': 'candidate-version',
        'graphiti_receipts': [{}], 'intake_receipt_id': 'original-intake',
        'assessment_contract_version': 'newsroom.native-evidence-assessor.v19+consumer.v1',
        'assessment_superseded': {'contract_version': origin.contract_version,
            'reason': 'ASSESSOR_RENDERING_CONTRACT_HOLD', 'package_admission_id': None, 'editorial_decision_id': None},
        'reason': 'ACQUISITION_RESULT_NOT_RETAINED', 'failure_class': 'ModelUsageAdmissionError'}
    if boundary == 'origin-mismatch':
        facts['semantic_assessment_intent'] = {'contract': contract,
            'candidate_version_id': 'candidate-version', 'origin_envelope_id': origin.proof.envelope_id,
            'origin_invocation_id': 'different-old-invocation',
            'origin_allocation_digest': _DIGEST, 'origin_terminal_digest': _DIGEST,
            'origin_context_manifest_digest': _DIGEST, 'origin_outcome': origin.outcome,
            'origin_contract_version': origin.contract_version, 'origin_base_digest': _DIGEST,
            'origin_result_digest': _DIGEST, 'origin_result_receipt_digest': _DIGEST,
            'origin_journal': dict(facts)}
    journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=facts)
    def sources_for(_revision):
        if boundary == 'stop': raise VetoError('owner stop')
        return ()  # No full current Source/rights means no acquisition/model call.
    monkeypatch.setattr(NativeEvidenceController, 'acquire_and_retain',
        lambda *_args, **_kwargs: pytest.fail('Source/stop/origin boundary was bypassed'))
    continuation = NativePublicationContinuation(journal=journal,
        runtime=SimpleNamespace(authority=_Authority(), ingress=object(), publication=_Publication(),
            proof=proof(), policies=SimpleNamespace(publication=object())),
        evidence_controller=object.__new__(NativeEvidenceController), sources={},
        evidence_sources_for=sources_for, semantic_origin_failure=lambda _: origin,
        semantic_intent_contract=contract, assessment_contract_version='newsroom.native-evidence-assessor.v23+consumer.v1',
        clock=lambda: UtcTimestamp.parse('2026-09-08T12:30:00Z'))
    try:
        if boundary in {'stop', 'origin-mismatch'}:
            with pytest.raises(VetoError if boundary == 'stop' else NativePublicationError):
                continuation.advance(revision_id=unit.revision_id, candidate_version_id='candidate-version')
        else:
            result = continuation.advance(revision_id=unit.revision_id, candidate_version_id='candidate-version')
            assert result.state == 'EVIDENCE_HOLD'
            assert journal.current(unit.revision_id)['facts']['failure_class'] == 'NativePublicationError'
        if boundary == 'origin-mismatch':
            assert journal.current(unit.revision_id)['facts'] == facts
        else:
            retained = journal.current(unit.revision_id)['facts']['semantic_assessment_intent']
            assert retained['origin_invocation_id'] == 'old-invocation'
            assert retained['origin_journal']['assessment_superseded'] == facts['assessment_superseded']
    finally:
        connection.close()


@pytest.mark.parametrize('choice',['NO','UNCERTAIN'])
def test_semantic_witness_disposition_is_retained_without_retry_or_free_text(tmp_path,monkeypatch,choice):
    from newsroom.control_plane.native_evidence import NativeEvidenceHold
    from newsroom.tests.test_qualification_semantic_witness import _witness_case
    from newsroom.control_plane.native_source_qualification_consumer import CONSUMER_VERSION
    with _witness_case(tmp_path,monkeypatch,choice=choice)as(w,q,c,p,b,usage,calls,stopped):
        unit=_native();connection=connect(str(tmp_path/'disposition.sqlite3'))
        journal=NativeRevisionJournal(connection);journal.land((unit,))
        current='newsroom.native-evidence-assessor.v23+'+CONSUMER_VERSION
        contract='newsroom.native-assessor-judgments.v2+newsroom.native-source-qualification.v2'
        intent={'contract':contract,'input_digest':_DIGEST,'origin_invocation_id':'protected-original'}
        facts={'candidate_id':p.candidate_id,'candidate_version_id':'candidate-version','graphiti_receipts':[{}],
            'intake_receipt_id':'protected-intake','semantic_assessment_intent':intent,
            'assessment_contract_version':current.replace('source-qualification-consumer.v4','source-qualification-consumer.v1'),
            'reason':'SEMANTIC_INTENT_INPUT_CHANGED_HOLD','failure_class':'ValueError','assessment_started_at':'protected-start'}
        journal.advance(unit.revision_id,stage='EVIDENCE_HOLD',facts=facts)
        class CurrentCandidate(_Authority):
            def candidate_version(self,_):return SimpleNamespace(candidate_id=p.candidate_id,governing_manifest=SimpleNamespace(canonical_digest=_DIGEST))
        requests=[]
        def acquire(_self,**request):
            requests.append(request);assert request['assessment_qualification_cached_only']is True
            request['before_assessment']()
            try:w.evaluate(q,c,p,b)
            except NativeEvidenceHold as exc:
                exc.free_model_text='This arbitrary field must not enter journal facts.'
                raise
        monkeypatch.setattr(NativeEvidenceController,'acquire_and_retain',acquire)
        monkeypatch.setattr('newsroom.control_plane.native_publication.open_private_serving_read_port',lambda *_a,**_k:_Reader())
        runtime=SimpleNamespace(authority=CurrentCandidate(),ingress=object(),publication=_Publication(),proof=proof(),
            policies=SimpleNamespace(publication=SimpleNamespace(target_path=tmp_path/'serving.sqlite3',target_id='private',target_context_digest=_DIGEST)))
        continuation=NativePublicationContinuation(journal=journal,runtime=runtime,evidence_controller=object.__new__(NativeEvidenceController),
            sources={unit.revision_id:(_source(unit),)},semantic_origin_failure=lambda _:None,semantic_intent_contract=contract,
            assessment_contract_version=current)
        try:
            result=continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
            after=journal.current(unit.revision_id)['facts']
            assert result.reason==after['reason']=='QUALIFICATION_SEMANTIC_WITNESS_'+choice
            assert after['semantic_assessment_intent']==intent and after['assessment_started_at']=='protected-start'
            assert set(after['semantic_witness_disposition'])=={'reference','confidence_ppm','probabilities_ppm'}
            assert 'free_model_text'not in after
            reference=after['semantic_witness_disposition']['reference']
            terminal=usage.terminal(reference['invocation_id'])
            assert terminal.usage_status.value=='REPORTED'and terminal.outcome=='TYPESAFE_COMPLETE'
            continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
            assert len(requests)==len(calls)==1 and runtime.publication.calls==0
            assert usage.terminal(reference['invocation_id'])==terminal
        finally:connection.close()


@pytest.mark.parametrize('fault',['NO','UNCERTAIN','missing-ref','source-change','unknown','stop','deadline'])
def test_old_valueerror_is_reclassified_only_by_current_authenticated_existing_witness(tmp_path,monkeypatch,fault):
    import sqlite3
    from newsroom.authority.canonical import digest_canonical
    from newsroom.tests.test_qualification_semantic_witness import _selected_qualification_case
    from newsroom.control_plane.native_source_qualification_replay import read_current_result
    from newsroom.control_plane.native_evidence import NativeEvidenceHold
    with _selected_qualification_case(tmp_path,monkeypatch,fault='NO'if fault!='UNCERTAIN'else None,retained_current=True)as(q,w,old,c,b,s,a,scope,auth,usage,qa,jev,render):
        if fault in {'UNCERTAIN','unknown'}:
            original_transport=w.judgments.transport
            def transport(request,**kw):
                if 'criterion'not in __import__('json').loads(request.data)['questions']:return original_transport(request,**kw)
                if fault=='unknown':jev.append('unknown fixture');raise TimeoutError('fixture unknown')
                status,url,raw=original_transport(request,**kw);value=__import__('json').loads(raw)
                answer=value['answers']['criterion'];answer['choice']='UNCERTAIN';answer['probabilities']={'YES':0,'NO':0,'UNCERTAIN':1}
                return status,url,__import__('json').dumps(value).encode()
            w.judgments.transport=transport
        original=read_current_result(q.qualifier,c,b,(s,),(a,),scope=scope,proof=auth)
        if fault!='missing-ref':
            with pytest.raises(NativeEvidenceHold):q.compose_selected(original,c,b,(s,),(a,),proof=auth)
        with sqlite3.connect(usage.path)as db:pins=db.execute('SELECT invocation_id,record_json FROM model_invocation_terminals ORDER BY invocation_id').fetchall()
        counts=(len(qa),len(jev),len(render))
        if fault=='source-change':s.unit.body+='\nChanged CURRENT Source.'
        monkeypatch.setattr(w.judgments,'evaluate',lambda **_k:pytest.fail('proof-only evaluated model'))
        monkeypatch.setattr(q.qualifier,'qualify',lambda *_a,**_k:pytest.fail('proof-only requalified QA'))
        q.localise=lambda *_a:pytest.fail('proof-only localised')
        unit=_native();connection=connect(str(tmp_path/'reclassification.sqlite3'));journal=NativeRevisionJournal(connection);journal.land((unit,))
        intent={'contract':'protected-contract','input_digest':_DIGEST}
        facts={'candidate_id':c.candidate_id,'candidate_version_id':c.version_id,'semantic_assessment_intent':intent,
            'reason':'SEMANTIC_INTENT_INPUT_CHANGED_HOLD','failure_class':'ValueError','assessment_contract_version':'protected-v2',
            'retained_qualification_checked_contract':'protected-v2','original_fee':'retained'}
        journal.advance(unit.revision_id,stage='EVIDENCE_HOLD',facts=facts)
        before=journal.summary(unit.revision_id)
        class Candidate(_Authority):
            def candidate_version(self,_):return c
        runtime=SimpleNamespace(authority=Candidate(),ingress=object(),publication=_Publication(),proof=auth,policies=SimpleNamespace(publication=object()))
        continuation=NativePublicationContinuation(journal=journal,runtime=runtime,evidence_controller=object.__new__(NativeEvidenceController),
            sources={},evidence_sources_for=lambda _:(s,),semantic_witness_disposition_reader=lambda candidate,sources:q.read_current_disposition(candidate,b,sources,proof=auth))
        try:
            checks=[]
            def before_revision():
                checks.append('checked')
                if fault=='stop'and len(checks)>1:
                    from newsroom.control_plane.veto import VetoError
                    raise VetoError('owner stop after authenticated read')
                return not(fault=='deadline'and len(checks)>1)
            if fault=='stop':
                from newsroom.control_plane.veto import VetoError
                with pytest.raises(VetoError,match='after authenticated read'):
                    continuation.recover_pre_dispatch((unit.revision_id,),failure_many=lambda *_:pytest.fail('unknown invocation recovery'),before_revision=before_revision)
                result=()
            else:
                result=continuation.recover_pre_dispatch((unit.revision_id,),failure_many=lambda *_:pytest.fail('unknown invocation recovery'),before_revision=before_revision)
            after=journal.summary(unit.revision_id)
            if fault in {'NO','UNCERTAIN'}:
                assert result==(unit.revision_id,)
                assert after['facts']['reason']=='QUALIFICATION_SEMANTIC_WITNESS_'+fault
                assert after['facts']['semantic_witness_previous_hold']['prior_state_ordinal']==before['ordinal']
                assert after['facts']['semantic_witness_previous_hold']['prior_summary_digest']==digest_canonical(before)
                assert after['facts']['semantic_witness_previous_hold']['failure_class']=='ValueError'
                assert after['facts']['semantic_assessment_intent']==intent and after['facts']['original_fee']=='retained'
            else:assert result==()and after==before
            if fault not in {'stop','deadline'}:assert continuation.recover_pre_dispatch((unit.revision_id,),failure_many=lambda *_:pytest.fail('retry'),before_revision=lambda:True)==()
            assert (len(qa),len(jev),len(render))==counts
            with sqlite3.connect(usage.path)as db:assert db.execute('SELECT invocation_id,record_json FROM model_invocation_terminals ORDER BY invocation_id').fetchall()==pins
        finally:connection.close()


def _meaning_acquisition(unit, *, refreshed=False, **changes):
    values = dict(request_digest=_DIGEST, outcome='COMPLETE',
        canonical_url=unit.canonical_url, body=b'Exact official source.', body_digest=digest_bytes(b'Exact official source.'),
        publisher='Official Department', responsible_body='Official Department', source_type='PRIMARY_OFFICIAL',
        publication_time='2026-09-08T10:00:00Z', source_updated_time='2026-09-08T11:00:00Z',
        retrieval_time='2026-09-08T14:00:00Z' if refreshed else '2026-09-08T12:00:00Z',
        geography='UK', language='en', transport_evidence_digest=digest_bytes(b'fresh' if refreshed else b'first'),
        rights_eligibility_digest=digest_bytes(b'fresh-rights' if refreshed else b'first-rights'),
        currentness_basis='AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT', text_only=True)
    values.update(changes)
    return AcquiredEvidence.create(**values)


@pytest.mark.parametrize('legacy', (False, True))
@pytest.mark.parametrize('changed', (False, True))
def test_future_semantic_input_is_stable_across_fresh_receipts_without_recovering_legacy(tmp_path, monkeypatch, legacy, changed):
    from datetime import UTC, datetime
    from newsroom.control_plane.native_assessor import RetainedAssessorResult
    unit = _native(); source = _source(unit); acquired = [_meaning_acquisition(unit)]
    connection = connect(str(tmp_path / 'meaning.sqlite3')); journal = NativeRevisionJournal(connection); journal.land((unit,))
    old = {'candidate_id':'candidate', 'candidate_version_id':'candidate-version','graphiti_receipts':[{}],
        'intake_receipt_id':'original-intake','assessment_contract_version':'newsroom.native-evidence-assessor.v23+consumer.v1',
        'reason':'ACQUISITION_RESULT_NOT_RETAINED','failure_class':'CliTimeoutError','assessment_started_at':'old-start'}
    origin = RetainedAssessorResult(RetainedAssessorContractFailure('old-envelope','old-invocation',_DIGEST,_DIGEST,_DIGEST),
        'newsroom.native-evidence-assessor.v23',_DIGEST,'ASSESSOR_PROVIDER_FAILED',datetime(2026,9,8,tzinfo=UTC),None)
    intent = {'contract':'newsroom.native-assessor-judgments.v1','candidate_version_id':'candidate-version',
        'origin_envelope_id':origin.proof.envelope_id,'origin_invocation_id':origin.proof.invocation_id,
        'origin_allocation_digest':_DIGEST,'origin_terminal_digest':_DIGEST,'origin_context_manifest_digest':_DIGEST,
        'origin_journal':dict(old)}
    if legacy:
        intent['input_digest'] = digest_canonical({'base_digest':_DIGEST,
            'acquired':[(acquired[0].receipt_digest,acquired[0].body_digest)]})
        old['semantic_assessment_intent'] = intent
    journal.advance(unit.revision_id,stage='ASSESSMENT_INTERRUPTED',facts=old)
    accepted = []
    def acquire(_self, **request):
        request['before_semantic_assessment'](SimpleNamespace(digest=_DIGEST),tuple(acquired))
        accepted.append(acquired[0].receipt_digest)
        request['before_assessment']()
        raise RuntimeError('Existing semantic accounting remains unresolved; no provider in this fixture')
    monkeypatch.setattr(NativeEvidenceController,'acquire_and_retain',acquire)
    continuation = NativePublicationContinuation(journal=journal,
        runtime=SimpleNamespace(authority=_Authority(),ingress=object(),publication=_Publication(),proof=proof(),
            policies=SimpleNamespace(publication=object())), evidence_controller=object.__new__(NativeEvidenceController),
        sources={unit.revision_id:(source,)}, semantic_origin_failure=lambda _:origin,
        semantic_intent_contract=intent['contract'], assessment_contract_version=old['assessment_contract_version'],
        clock=lambda:UtcTimestamp.parse('2026-09-08T15:00:00Z'))
    try:
        if not legacy: continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
        acquired[0] = _meaning_acquisition(unit,refreshed=True,
            **({'body':b'Changed Source.','body_digest':digest_bytes(b'Changed Source.')}if changed else {}))
        result = continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
        retained = journal.current(unit.revision_id)['facts']['semantic_assessment_intent']
        if legacy:
            assert result.reason=='SEMANTIC_INTENT_INPUT_CHANGED_HOLD' and accepted==[]
            assert retained['input_digest']==intent['input_digest'] and 'semantic_input_fingerprint_v2' not in retained
        else:
            assert result.reason==('SEMANTIC_INTENT_INPUT_CHANGED_HOLD'if changed else 'SEMANTIC_ASSESSMENT_ALREADY_ATTEMPTED_HOLD')and len(accepted)==1
            assert retained['semantic_input_fingerprint_v2']['version']=='newsroom.semantic-input-fingerprint.v2'
            assert retained['assessment_started_at']=='2026-09-08T15:00:00.000000Z'
            assert 'input_digest' not in retained
            if not changed:
                after=journal.summary(unit.revision_id)
                continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
                assert journal.summary(unit.revision_id)==after and len(accepted)==1
        assert retained['origin_invocation_id']=='old-invocation'
        if legacy: assert retained['origin_journal']==intent['origin_journal']
        else: assert all(retained['origin_journal'][key]==old.get(key) for key in retained['origin_journal'])
    finally:connection.close()


@pytest.mark.parametrize('failure',['acquisition','stop'])
def test_semantic_source_failure_before_producer_turn_remains_bounded_retry(tmp_path,monkeypatch,failure):
    from datetime import UTC,datetime
    from newsroom.control_plane.native_assessor import RetainedAssessorResult
    from newsroom.control_plane.veto import VetoError
    unit=_native();source=_source(unit);item=_meaning_acquisition(unit)
    connection=connect(str(tmp_path/'pre-producer.sqlite3'));journal=NativeRevisionJournal(connection);journal.land((unit,))
    old={'candidate_id':'candidate','candidate_version_id':'candidate-version','graphiti_receipts':[{}],
        'intake_receipt_id':'original-intake','assessment_contract_version':'newsroom.native-evidence-assessor.v23+consumer.v1',
        'reason':'ACQUISITION_RESULT_NOT_RETAINED','failure_class':'CliTimeoutError','assessment_started_at':'old-origin-start'}
    journal.advance(unit.revision_id,stage='ASSESSMENT_INTERRUPTED',facts=old)
    origin=RetainedAssessorResult(RetainedAssessorContractFailure('old-envelope','old-invocation',_DIGEST,_DIGEST,_DIGEST),
        'newsroom.native-evidence-assessor.v23',_DIGEST,'ASSESSOR_PROVIDER_FAILED',datetime(2026,9,8,tzinfo=UTC),None)
    calls=[]
    def acquire(_self,**request):
        calls.append('Source acquisition')
        if len(calls)==1:
            if failure=='stop':raise VetoError('fixture owner stop before semantic producer')
            raise TimeoutError('fixture acquisition timeout before semantic producer')
        request['before_semantic_assessment'](SimpleNamespace(digest=_DIGEST),(item,))
        assert journal.summary(unit.revision_id)['facts']['semantic_assessment_intent']['assessment_started_at']
        request['before_assessment']()
        raise NativeEvidenceHold('NO_QUALIFYING_NEW_INFORMATION',unit.source_id)
    monkeypatch.setattr(NativeEvidenceController,'acquire_and_retain',acquire)
    continuation=NativePublicationContinuation(journal=journal,
        runtime=SimpleNamespace(authority=_Authority(),ingress=object(),publication=_Publication(),proof=proof(),policies=SimpleNamespace(publication=object())),
        evidence_controller=object.__new__(NativeEvidenceController),sources={unit.revision_id:(source,)},
        semantic_origin_failure=lambda _:origin,semantic_intent_contract='newsroom.native-assessor-judgments.v2',assessment_contract_version=old['assessment_contract_version'])
    try:
        if failure=='stop':
            with pytest.raises(VetoError):continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
        else:assert continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version').reason=='ACQUISITION_TRANSPORT_RETRY'
        first=journal.summary(unit.revision_id)['facts']
        assert 'assessment_started_at'not in first['semantic_assessment_intent']
        assert first['semantic_assessment_intent']['origin_journal']['assessment_started_at']=='old-origin-start'
        assert continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version').reason=='NO_QUALIFYING_NEW_INFORMATION'
        assert len(calls)==2
        settled=journal.summary(unit.revision_id)
        assert continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version').reason=='NO_QUALIFYING_NEW_INFORMATION'
        assert journal.summary(unit.revision_id)==settled and len(calls)==2
    finally:connection.close()


@pytest.mark.parametrize('stage',['ASSESSMENT_INTERRUPTED','ASSESSMENT_STARTED'])
def test_existing_accounted_sourceqa_timeout_cannot_alias_a_fresh_observation(tmp_path,monkeypatch,stage):
    import sqlite3
    from datetime import UTC,datetime
    from newsroom.control_plane.native_assessor import RetainedAssessorResult
    from newsroom.control_plane.native_publication import _semantic_input_fingerprint_v2
    from newsroom.tests.test_native_source_qualification import _retained_role_bound_recipe
    with _retained_role_bound_recipe(tmp_path,monkeypatch,failure='timeout',first_publication=True)as(qualifier,consumer,candidate,base,source,fresh,scope,_old,usage,qa,jev):
        unit=_native();source.unit.revision_id=unit.revision_id;source.unit.revision_digest=_DIGEST
        source.source_version.version_id='definition-version'
        source=NativeEvidenceSource(unit=source.unit,source_version=source.source_version,dependency=source.dependency,rights=source.rights)
        fresh.request_digest=_DIGEST;fresh.body_origin=''
        origin=RetainedAssessorResult(RetainedAssessorContractFailure('origin-envelope','origin-invocation',_DIGEST,_DIGEST,_DIGEST),
            'newsroom.native-evidence-assessor.v23',base.digest,'ASSESSOR_PROVIDER_FAILED',datetime(2026,9,8,tzinfo=UTC),None)
        intent={'contract':'newsroom.native-assessor-judgments.v2','candidate_version_id':candidate.version_id,
            'origin_envelope_id':origin.proof.envelope_id,'origin_invocation_id':origin.proof.invocation_id,
            'origin_allocation_digest':_DIGEST,'origin_terminal_digest':_DIGEST,'origin_context_manifest_digest':_DIGEST,
            'origin_journal':{'assessment_started_at':'old-origin-start'},
            'semantic_input_fingerprint_v2':_semantic_input_fingerprint_v2(base,(fresh,),(source,))}
        connection=connect(str(tmp_path/'qa-alias.sqlite3'));journal=NativeRevisionJournal(connection);journal.land((unit,))
        facts={'candidate_id':candidate.candidate_id,'candidate_version_id':candidate.version_id,'graphiti_receipts':[{}],
            'intake_receipt_id':'protected-intake','semantic_assessment_intent':intent,'semantic_acquisition_attempt_count':1,
            'assessment_started_at':'new-semantic-start','assessment_contract_version':'newsroom.native-evidence-assessor.v23+consumer.v2',
            'reason':'ACQUISITION_RESULT_NOT_RETAINED','failure_class':'TimeoutError'}
        journal.advance(unit.revision_id,stage=stage,facts=facts)
        with sqlite3.connect(usage.path)as db:
            pins=db.execute('SELECT invocation_id,record_json FROM model_invocation_terminals ORDER BY invocation_id').fetchall()
            allocations=db.execute('SELECT invocation_id,record_json FROM model_invocation_allocations ORDER BY invocation_id').fetchall()
        counts=(len(qa),len(jev));acquired=[]
        def acquire(_self,**request):
            acquired.append('fresh Source only')
            request['before_semantic_assessment'](base,(fresh,))
            request['before_assessment']()
            consumer.scope_for=lambda *_:scope
            fallback=consumer.assess(candidate,base,(source,),(fresh,))
            qualifier.assess(candidate,base,(source,),(fresh,),fallback,scope=scope,proof=consumer.proof)
        monkeypatch.setattr(NativeEvidenceController,'acquire_and_retain',acquire)
        class CurrentCandidate(_Authority):
            def candidate_version(self,_):return candidate
        continuation=NativePublicationContinuation(journal=journal,
            runtime=SimpleNamespace(authority=CurrentCandidate(),ingress=object(),publication=_Publication(),proof=proof(),policies=SimpleNamespace(publication=object())),
            evidence_controller=object.__new__(NativeEvidenceController),sources={unit.revision_id:(source,)},
            semantic_origin_failure=lambda _:origin,semantic_intent_contract=intent['contract'],assessment_contract_version=facts['assessment_contract_version'])
        try:
            result=continuation.advance(revision_id=unit.revision_id,candidate_version_id=candidate.version_id)
            assert result.reason=='SEMANTIC_ASSESSMENT_ALREADY_ATTEMPTED_HOLD'
            assert acquired==['fresh Source only']and(len(qa),len(jev))==counts
            with sqlite3.connect(usage.path)as db:
                assert db.execute('SELECT invocation_id,record_json FROM model_invocation_terminals ORDER BY invocation_id').fetchall()==pins
                assert db.execute('SELECT invocation_id,record_json FROM model_invocation_allocations ORDER BY invocation_id').fetchall()==allocations
        finally:connection.close()


@pytest.mark.parametrize('changed', ['body', 'request', 'revision', 'definition', 'publisher',
    'publication', 'update', 'category', 'geography', 'permission', 'rights-policy', 'base'])
def test_semantic_meaning_fingerprint_changes_for_true_source_or_permission_inputs(changed):
    from newsroom.control_plane.native_publication import _semantic_input_fingerprint_v2
    unit = _native(); source = _source(unit); base = SimpleNamespace(digest=_DIGEST)
    acquired = _meaning_acquisition(unit)
    before = _semantic_input_fingerprint_v2(base,(acquired,),(source,))
    if changed == 'body': acquired = _meaning_acquisition(unit,body=b'Changed.',body_digest=digest_bytes(b'Changed.'))
    elif changed == 'request': acquired = _meaning_acquisition(unit,request_digest=digest_bytes(b'changed request'))
    elif changed == 'revision':
        from newsroom.tests.test_graphiti_operational_readiness import _next_revision
        source = replace(source,unit=_next_revision(unit))
    elif changed == 'definition':
        from newsroom.sources import SourceDefinitionVersionId
        request = replace(source.source_version.request,version_id=SourceDefinitionVersionId.new())
        source = replace(source,source_version=replace(source.source_version,request=request,canonical_digest=request.digest))
    elif changed == 'publisher': acquired = _meaning_acquisition(unit,publisher='Different Department')
    elif changed == 'publication': acquired = _meaning_acquisition(unit,publication_time='2026-09-09T10:00:00Z')
    elif changed == 'update': acquired = _meaning_acquisition(unit,source_updated_time='2026-09-09T11:00:00Z')
    elif changed == 'category': acquired = _meaning_acquisition(unit,source_type='SECONDARY_REPORT')
    elif changed == 'geography': acquired = _meaning_acquisition(unit,geography='HK')
    elif changed in {'permission','rights-policy'}:
        rights = PublicationRightsAssessment.create(decision='HOLD' if changed=='permission' else 'PERMITTED',
            permitted_use=source.rights.permitted_use,policy_digest=digest_bytes(b'changed policy') if changed=='rights-policy' else _DIGEST,
            evidence_digest=_DIGEST)
        source = replace(source,rights=rights)
    else: base=SimpleNamespace(digest=digest_bytes(b'changed base'))
    assert canonical_json_bytes(_semantic_input_fingerprint_v2(base,(acquired,),(source,)))!=canonical_json_bytes(before)


def test_semantic_base_constructor_has_no_retrieval_or_rights_receipt_fields():
    from newsroom.control_plane.evidence import EvidencePackage
    unit=_native();first=_meaning_acquisition(unit);fresh=_meaning_acquisition(unit,refreshed=True)
    def base(item):
        return EvidencePackage(candidate_id='candidate',hypothesis_id='hypothesis',signal_ids=('signal',),
            lead_ids=('lead',),source_ids=(unit.source_id,),observation_digests=(item.body_digest,),
            passages=(item.body.decode(),))
    assert first.receipt_digest!=fresh.receipt_digest and base(first).digest==base(fresh).digest


def test_semantic_meaning_ignores_fresh_permission_evidence_identity_only():
    from newsroom.control_plane.native_publication import _semantic_input_fingerprint_v2
    source=_source(_native());base=SimpleNamespace(digest=_DIGEST)
    refreshed=replace(source,rights=PublicationRightsAssessment.create(decision='PERMITTED',
        permitted_use=source.rights.permitted_use,policy_digest=source.rights.policy_digest,
        evidence_digest=digest_bytes(b'new authenticated observation')))
    assert refreshed.rights.record_id!=source.rights.record_id
    assert canonical_json_bytes(_semantic_input_fingerprint_v2(base,(_meaning_acquisition(source.unit),),(source,)))==canonical_json_bytes(
        _semantic_input_fingerprint_v2(base,(_meaning_acquisition(source.unit,refreshed=True),),(refreshed,)))


@pytest.mark.parametrize('failure',['exception','stdlib-exception','typed-hold','no-result','logging-failure'])
def test_witness_recovery_observes_suppressed_boundary_without_changing_authority(tmp_path,monkeypatch,failure):
    from newsroom.control_plane import diagnostic_logging
    unit=_native();connection=connect(str(tmp_path/'recovery-observation.sqlite3'));journal=NativeRevisionJournal(connection);journal.land((unit,))
    facts={'candidate_id':'candidate','candidate_version_id':'candidate-version','semantic_assessment_intent':{'contract':'protected'},
        'reason':'SEMANTIC_INTENT_INPUT_CHANGED_HOLD','failure_class':'ValueError'}
    journal.advance(unit.revision_id,stage='EVIDENCE_HOLD',facts=facts);before=journal.summary(unit.revision_id);events=[]
    def observe(event,data):
        if failure=='logging-failure':raise OSError('diagnostic sink unavailable')
        events.append((event,data))
    monkeypatch.setattr(diagnostic_logging,'emit_diagnostic',observe)
    def reader(*_a):
        if failure=='stdlib-exception':json.loads('INVALID TOKEN=private-secret')
        if failure=='typed-hold':
            from newsroom.control_plane.native_evidence import NativeEvidenceHold
            raise NativeEvidenceHold('QUALIFICATION_SEMANTIC_WITNESS_UNKNOWN_HOLD',unit.source_id)
        if failure!='no-result':raise ValueError('TOKEN=private-secret and full provider response must not be logged')
    candidate=SimpleNamespace(candidate_id='candidate',version_id='candidate-version')
    continuation=NativePublicationContinuation(journal=journal,
        runtime=SimpleNamespace(authority=SimpleNamespace(candidate_version=lambda _:candidate),ingress=object(),publication=_Publication(),policies=SimpleNamespace(publication=object()),proof=proof()),
        evidence_controller=object.__new__(NativeEvidenceController),sources={},
        evidence_sources_for=lambda _:(),semantic_witness_disposition_reader=reader)
    try:
        assert continuation.recover_pre_dispatch((unit.revision_id,),failure_many=lambda *_:pytest.fail('provider recovery'),before_revision=lambda:True)==()
        assert journal.summary(unit.revision_id)==before
        if failure=='logging-failure':assert events==[]
        else:
            assert len(events)==1 and events[0][0]=='native_witness_recovery_observation'
            data=events[0][1];assert data['revision_id']==unit.revision_id
            assert data['summary_digest']==digest_canonical(before)
            assert data['outcome']==('NO_DISPOSITION'if failure=='no-result'else'READ_FAILED')
            assert 'private-secret'not in json.dumps(events) and 'TOKEN='not in json.dumps(events)
            if failure in {'exception','stdlib-exception','typed-hold'}:
                assert data['failure_class']==('JSONDecodeError'if failure=='stdlib-exception'else'NativeEvidenceHold'if failure=='typed-hold'else'ValueError')
                assert data['function']=='reader'and data['file']=='test_native_publication_continuation.py'
    finally:connection.close()


def _known_witness_hold_facts(choice='NO'):
    from newsroom.control_plane.evidence import SEMANTIC_WITNESS_CONTRACT
    return {'candidate_id':'candidate','candidate_version_id':'candidate-version',
        'graphiti_receipts':[{}],'intake_receipt_id':'retained-intake',
        'assessment_contract_version':'newsroom.native-evidence-assessor.v23+newsroom.source-qualification-consumer.v4',
        'reason':'QUALIFICATION_SEMANTIC_WITNESS_'+choice,
        'assessment_started_at':'2026-10-09T14:31:20Z','acquisition_attempt_count':1,
        'semantic_witness_disposition':{'reference':{'contract':SEMANTIC_WITNESS_CONTRACT,
            'question_id':'criterion','invocation_id':_DIGEST,
            'raw_admission_id':str(ObjectAdmissionId.new()),'receipt_admission_id':str(ObjectAdmissionId.new())},
            'confidence_ppm':330000,'probabilities_ppm':{key:(550000 if key==choice else 220000 if key=='YES' else 230000)
                for key in ('YES','NO','UNCERTAIN')}}}


@pytest.mark.parametrize('choice',['NO','UNCERTAIN'])
def test_known_disagreement_uses_once_only_retained_consumer_continuation(tmp_path, monkeypatch, choice):
    from newsroom.control_plane.native_source_qualification_consumer import RESOLUTION_CONSUMER_VERSION
    unit=_native();connection=connect(str(tmp_path/'disagreement.sqlite3'))
    journal=NativeRevisionJournal(connection);journal.land((unit,))
    facts=_known_witness_hold_facts(choice);current=facts['assessment_contract_version']+'+'+RESOLUTION_CONSUMER_VERSION
    journal.advance(unit.revision_id,stage='EVIDENCE_HOLD',facts=facts)
    calls=[]
    def acquire(_self, **request):
        calls.append(request)
        assert request['assessment_cached_only'] is True
        assert request['assessment_qualification_cached_only'] is True
        assert 'assessment_semantic_only' not in request
        request['before_assessment']()
        raise NativeEvidenceHold('QUALIFICATION_RESOLUTION_UNPROVEN_HOLD',unit.source_id)
    monkeypatch.setattr(NativeEvidenceController,'acquire_and_retain',acquire)
    monkeypatch.setattr('newsroom.control_plane.native_publication.open_private_serving_read_port',lambda *_a,**_k:_Reader())
    runtime=SimpleNamespace(authority=_Authority(),ingress=object(),publication=_Publication(),
        policies=SimpleNamespace(publication=SimpleNamespace(target_path=tmp_path/'serving.sqlite3',target_id='private',target_context_digest=_DIGEST)),proof=proof())
    continuation=NativePublicationContinuation(journal=journal,runtime=runtime,
        evidence_controller=object.__new__(NativeEvidenceController),sources={unit.revision_id:(_source(unit),)},
        assessment_contract_version=current)
    assert continuation.qualification_resolution_due(facts)
    continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
    after=journal.current(unit.revision_id)['facts']
    assert len(calls)==1
    assert after['retained_qualification_checked_contract']==current
    assert after['assessment_started_at']==facts['assessment_started_at']
    assert after['acquisition_attempt_count']==facts['acquisition_attempt_count']+1
    assert after['semantic_witness_disposition']==facts['semantic_witness_disposition']
    assert not continuation.qualification_resolution_due(after)
    continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
    assert len(calls)==1 and runtime.authority.receives==0 and runtime.publication.calls==0
    connection.close()


@pytest.mark.parametrize('defect',['disabled','same-contract','different-producer','checked','missing-disposition',
    'bad-reference','bad-confidence','unsettled','unknown','missing-intake','missing-graph','committed'])
def test_disagreement_schedule_does_not_rescue_unknown_or_unbound_work(defect):
    from newsroom.control_plane.native_source_qualification_consumer import RESOLUTION_CONSUMER_VERSION
    facts=_known_witness_hold_facts();current=facts['assessment_contract_version']+'+'+RESOLUTION_CONSUMER_VERSION
    if defect=='disabled':current+='-not-enabled'
    elif defect=='same-contract':facts['assessment_contract_version']=current
    elif defect=='different-producer':facts['assessment_contract_version']='newsroom.native-evidence-assessor.v22'
    elif defect=='checked':facts['retained_qualification_checked_contract']=current
    elif defect=='missing-disposition':facts.pop('semantic_witness_disposition')
    elif defect=='bad-reference':facts['semantic_witness_disposition']['reference']['invocation_id']='wrong'
    elif defect=='bad-confidence':facts['semantic_witness_disposition']['confidence_ppm']=True
    elif defect=='unsettled':facts['reason']='QUALIFICATION_SEMANTIC_WITNESS_UNKNOWN_HOLD'
    elif defect=='unknown':facts.update(reason='ACQUISITION_RESULT_NOT_RETAINED',failure_class='CliTimeoutError')
    elif defect=='missing-intake':facts.pop('intake_receipt_id')
    elif defect=='missing-graph':facts.pop('graphiti_receipts')
    elif defect=='committed':facts['package_admission_id']='already-admitted'
    continuation=object.__new__(NativePublicationContinuation);continuation._assessment_contract_version=current
    assert not continuation.qualification_resolution_due(facts)


def test_disagreement_schedule_uses_authenticated_verdict_not_probability_ranking():
    from newsroom.control_plane.native_source_qualification_consumer import RESOLUTION_CONSUMER_VERSION
    facts=_known_witness_hold_facts()
    facts['semantic_witness_disposition']['probabilities_ppm']={'YES':550000,'NO':220000,'UNCERTAIN':230000}
    continuation=object.__new__(NativePublicationContinuation)
    continuation._assessment_contract_version=facts['assessment_contract_version']+'+'+RESOLUTION_CONSUMER_VERSION
    assert continuation.qualification_resolution_due(facts)


@pytest.mark.parametrize('reason',[
    'QUALIFICATION_RESOLUTION_NOT_AFFIRMATIVE_HOLD',
    'ACQUISITION_RESULT_NOT_RETAINED',
])
def test_corrected_resolution_upgrade_does_not_reopen_prior_negative_or_unknown(reason):
    from newsroom.control_plane.native_composition import ASSESSMENT_CONTRACT_VERSION
    facts=_known_witness_hold_facts()
    facts['assessment_contract_version']+='+newsroom.source-qualification-resolution-consumer.v1'
    facts['retained_qualification_checked_contract']=facts['assessment_contract_version']
    facts['reason']=reason
    continuation=object.__new__(NativePublicationContinuation)
    continuation._assessment_contract_version=ASSESSMENT_CONTRACT_VERSION
    assert continuation._assessment_contract_version!=facts['assessment_contract_version']
    assert not continuation.qualification_resolution_due(facts)


@pytest.mark.parametrize('rendering_repair',[False,True])
def test_known_qualification_repair_uses_once_only_cached_consumer(tmp_path, monkeypatch, rendering_repair):
    from newsroom.control_plane.native_source_qualification_consumer import REPLAY_CONSUMER_VERSION
    unit=_native();connection=connect(str(tmp_path/'recipe.sqlite3'))
    journal=NativeRevisionJournal(connection);journal.land((unit,))
    facts=_known_witness_hold_facts()
    facts.pop('semantic_witness_disposition')
    facts.update(reason='QUALIFICATION_ORIGINAL_RECIPE_UNSUPPORTED',failure_class='QualificationHold')
    current=facts['assessment_contract_version']+'+'+REPLAY_CONSUMER_VERSION
    if rendering_repair:
        facts.update(reason='QUALIFICATION_RESOLUTION_RENDERING_HOLD',failure_class=None)
        current=facts['assessment_contract_version']+'+newsroom.source-qualification-rendering-repair.v1'
    journal.advance(unit.revision_id,stage='EVIDENCE_HOLD',facts=facts)
    calls=[]
    def acquire(_self,**request):
        calls.append(request)
        assert request['assessment_cached_only'] is True
        assert request['assessment_qualification_cached_only'] is True
        request['before_assessment']()
        raise NativeEvidenceHold('KNOWN_RECIPE_TEST_STOP',unit.source_id)
    monkeypatch.setattr(NativeEvidenceController,'acquire_and_retain',acquire)
    monkeypatch.setattr('newsroom.control_plane.native_publication.open_private_serving_read_port',lambda *_a,**_k:_Reader())
    runtime=SimpleNamespace(authority=_Authority(),ingress=object(),publication=_Publication(),
        policies=SimpleNamespace(publication=SimpleNamespace(target_path=tmp_path/'serving.sqlite3',target_id='private',target_context_digest=_DIGEST)),proof=proof())
    continuation=NativePublicationContinuation(journal=journal,runtime=runtime,
        evidence_controller=object.__new__(NativeEvidenceController),sources={unit.revision_id:(_source(unit),)},
        assessment_contract_version=current)
    try:
        assert continuation.qualification_resolution_due(facts)
        continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
        after=journal.current(unit.revision_id)['facts']
        assert after['retained_qualification_checked_contract']==current
        assert after['assessment_started_at']==facts['assessment_started_at']
        assert after['acquisition_attempt_count']==facts['acquisition_attempt_count']+1
        continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
        assert len(calls)==1 and runtime.authority.receives==0 and runtime.publication.calls==0
        assert not continuation.qualification_resolution_due(after)
    finally:connection.close()


@pytest.mark.parametrize('fault',['disabled','unknown','missing-intake','checked'])
def test_recipe_repair_does_not_reopen_unproved_or_already_checked_work(fault):
    from newsroom.control_plane.native_source_qualification_consumer import REPLAY_CONSUMER_VERSION
    facts=_known_witness_hold_facts();facts.pop('semantic_witness_disposition')
    facts.update(reason='QUALIFICATION_ORIGINAL_RECIPE_UNSUPPORTED',failure_class='QualificationHold')
    current=facts['assessment_contract_version']+'+'+REPLAY_CONSUMER_VERSION
    if fault=='disabled':current+='-disabled'
    elif fault=='unknown':facts['failure_class']='CliTimeoutError'
    elif fault=='missing-intake':facts.pop('intake_receipt_id')
    else:facts['retained_qualification_checked_contract']=current
    continuation=object.__new__(NativePublicationContinuation);continuation._assessment_contract_version=current
    assert not continuation.qualification_resolution_due(facts)


@pytest.mark.parametrize('fault',['disabled','unknown','checked','committed','declined'])
def test_rendering_repair_does_not_reset_unknown_or_semantic_decisions(fault):
    facts=_known_witness_hold_facts();facts.pop('semantic_witness_disposition')
    facts.update(reason='QUALIFICATION_RESOLUTION_RENDERING_HOLD',failure_class=None)
    current=facts['assessment_contract_version']+'+newsroom.source-qualification-rendering-repair.v1'
    if fault=='disabled':current+='-disabled'
    elif fault=='unknown':facts['failure_class']='CliTimeoutError'
    elif fault=='checked':facts['retained_qualification_checked_contract']=current
    elif fault=='committed':facts['package_admission_id']='already-admitted'
    else:facts['reason']='QUALIFICATION_RESOLUTION_NOT_AFFIRMATIVE_HOLD'
    continuation=object.__new__(NativePublicationContinuation);continuation._assessment_contract_version=current
    assert not continuation.qualification_resolution_due(facts)


def test_missing_cache_consumer_upgrade_preserves_generic_cached_reader_selection(tmp_path,monkeypatch):
    unit=_native();connection=connect(str(tmp_path/'cached-origin.sqlite3'))
    journal=NativeRevisionJournal(connection);journal.land((unit,))
    prior='newsroom.native-evidence-assessor.v23+consumer.v1'
    current=prior+'+newsroom.cached-assessment-origin.v1'
    journal.advance(unit.revision_id,stage='EVIDENCE_HOLD',facts={
        'candidate_version_id':'candidate-version','graphiti_receipts':[{}],
        'intake_receipt_id':'retained-intake','reason':'ASSESSOR_REVALIDATION_CACHE_MISSING_HOLD',
        'assessment_contract_version':prior,'acquisition_retryable':False})
    calls=[]
    def acquire(_self,**request):
        calls.append(request)
        assert request['assessment_cached_only'] is True
        assert not request.get('assessment_qualification_cached_only')  # Adapter authenticates the origin.
        assert request['intake_receipt_id']=='retained-intake'
        raise NativeEvidenceHold('NO_QUALIFYING_NEW_INFORMATION',unit.source_id)
    monkeypatch.setattr(NativeEvidenceController,'acquire_and_retain',acquire)
    runtime=SimpleNamespace(authority=_Authority(),ingress=object(),publication=_Publication(),
                            policies=SimpleNamespace(publication=object()),proof=proof())
    continuation=NativePublicationContinuation(journal=journal,runtime=runtime,
        evidence_controller=object.__new__(NativeEvidenceController),sources={unit.revision_id:(_source(unit),)},
        assessment_contract_version=current)
    try:
        for _ in range(2):continuation.advance(revision_id=unit.revision_id,candidate_version_id='candidate-version')
        facts=journal.current(unit.revision_id)['facts']
        assert len(calls)==1
        assert facts['assessment_contract_version']==current
        assert facts['assessment_superseded']['reason']=='ASSESSOR_REVALIDATION_CACHE_MISSING_HOLD'
        assert facts['intake_receipt_id']=='retained-intake'
        assert runtime.authority.receives==runtime.publication.calls==0
    finally:connection.close()
