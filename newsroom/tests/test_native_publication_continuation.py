from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from newsroom.authority import AuthorityEvents, EventId, ObjectAdmissionId, UtcTimestamp
from newsroom.control_plane.native_evidence import (
    DependencyAssessment,
    NativeEvidenceController,
    NativeEvidenceHold,
    NativeEvidenceSource,
    PublicationRightsAssessment,
)
from newsroom.control_plane.graphiti_operational_readiness import _source_requests
from newsroom.control_plane.native_progress import NativeRevisionJournal
from newsroom.control_plane.native_assessor import (
    RetainedAssessorContractFailure,
    RetainedAssessorPreDispatchFailure,
)
from newsroom.control_plane.native_publication import NativePublicationContinuation
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
