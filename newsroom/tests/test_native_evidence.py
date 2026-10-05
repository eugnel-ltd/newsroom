from __future__ import annotations

import json

from copy import deepcopy
from dataclasses import replace

import pytest

from newsroom.authority import EventId, UtcTimestamp
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.control_plane.graphiti_operational_readiness import _source_requests
from newsroom.control_plane.admission import DeterministicWriteAdmission
from newsroom.control_plane.native_evidence import (
    _admit_record,
    AcquiredEvidence,
    AcquiredSourceAssessment,
    DependencyAssessment,
    EvidenceAcquisitionRequest,
    EvidenceAssessor,
    EvidenceTransport,
    IndependentEvidenceAssessment,
    NativeEvidenceController,
    NativeEvidenceHold,
    NativeEvidenceError,
    NativeEvidenceSource,
    PublicationRightsAssessment,
)
from newsroom.control_plane.evidence import (
    EVIDENCE_GATE_POLICY_VERSION,
    EvidenceGateEvidence,
)
from newsroom.control_plane.native_assessor import (
    AutonomousNativeEvidenceAssessor,
    NativeAssessmentExecution,
)
from newsroom.control_plane.native_assessor_references import build_source_view
from newsroom.control_plane.native_publication import NativePublicationController
from newsroom.control_plane.writer import (
    WriterCopy,
    required_surface_copy,
    validate_writer_copy,
)
from newsroom.increment10.editorial import SourceCurrentness
from newsroom.increment10.evidence import EvidencePackageError, _base_package
from newsroom.increment10.ingress import open_evidence_intake_ingress
from newsroom.increment10.private_serving import open_private_serving_read_port
from newsroom.sources.record_models import SourceDefinitionVersion
from newsroom.tests.authority_helpers import FIXED_NOW, proof
from newsroom.tests.test_graphiti_operational_readiness import _rights, _unit
from newsroom.tests.test_increment10_editorial import _evidence_facade, _ready_package
from newsroom.tests.test_increment10_ingress import _candidate, _receive
from newsroom.tests.test_increment10_private_serving import _open
from newsroom.tests.test_native_assessor import (
    _model_package_value, _current_wire_from_v17,
)
from newsroom.tests.test_native_publication import _bindings

NOW = UtcTimestamp.parse("2026-09-02T12:02:00.000000Z")
INTEGRITY = tuple(
    (name, "PASS")
    for name in (
        "ACCESS_COMPLETE",
        "ENCODING_VALID",
        "EXTRACTION_COMPLETE",
        "NOT_PAYWALL_FRAGMENT",
        "NOT_TRUNCATED",
        "VERSION_UNAMBIGUOUS",
    )
)


def test_independent_source_evidence_holds_then_reaches_private_ack(tmp_path) -> None:
    candidate_connection, candidate_port, version = _candidate(tmp_path)
    ingress = open_evidence_intake_ingress(tmp_path / "intake.sqlite3")
    acknowledgement = _receive(
        ingress,
        candidate_connection,
        candidate_port,
        version,
        request_id="request-1",
    )

    def candidate_version(version_id):
        candidate_connection.execute("BEGIN")
        try:
            return candidate_port.require_retained_version_in_transaction(version_id)
        finally:
            candidate_connection.rollback()

    candidate_port = candidate_port._with_bounded_version(candidate_version)
    system, registries, hydration, definitions, commands = _open(
        tmp_path / "objects.sqlite3"
    )
    packages = _evidence_facade(system, ingress, registries)
    passage, assessed_package, records = _ready_package(version)
    source_id = _unit().source_id
    source_record = next(
        item for item in records if item["record_type"] == "SOURCE_RECORD"
    )
    unit = replace(
        _unit(),
        headline="",
        body=passage,
        canonical_url=source_record["canonical_url"],
    )
    unit = replace(
        unit,
        effective_revision=replace(
            unit.effective_revision,
            revision_digest=unit.revision_digest
        ),
    )
    version_request = _source_requests(unit, _rights())[1]
    source_version = SourceDefinitionVersion(
        version_request,
        EventId.new(),
        1,
        NOW,
        version_request.digest,
    )
    transport_digest = "sha256:" + "7" * 64
    currentness_evidence = "sha256:" + "c" * 64
    calls = []
    assessment_markers = []

    def acquired(request):
        body = passage.encode()
        rights_receipt = PublicationRightsAssessment.create(
            decision="PERMITTED",
            permitted_use="PUBLICATION_EVIDENCE",
            policy_digest="sha256:" + "1" * 64,
            evidence_digest="sha256:" + "2" * 64,
        ).record_id
        return AcquiredEvidence.create(
            request_digest=request.digest,
            outcome="COMPLETE",
            canonical_url=unit.canonical_url,
            body=body,
            body_digest=digest_bytes(body),
            publisher=source_record["publisher"],
            responsible_body=source_record["responsible_body"],
            source_type=source_record["source_type"],
            publication_time=source_record["publication_time"],
            source_updated_time=source_record["publication_time"],
            retrieval_time=source_record["retrieval_time"],
            geography=source_record["geography"],
            language=source_record["language"],
            transport_evidence_digest=currentness_evidence,
            currentness_basis="AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT",
            rights_eligibility_digest=digest_canonical(
                {
                    "rights_receipt": rights_receipt,
                    "body_digest": digest_bytes(body),
                    "transport": currentness_evidence,
                    "exclusion_signals": (),
                    "text_only": True,
                }
            ),
            licence_attribution="Contains public sector information licensed",
            exclusion_signals=(),
            text_only=True,
        )

    request = EvidenceAcquisitionRequest(
        source_id,
        unit.authority.definition_id,
        unit.authority.definition_version_id,
        source_version.canonical_digest,
        unit.authority.revision_id,
        unit.canonical_url,
        transport_digest,
    )
    acquisition = acquired(request)

    def acquire(request):
        calls.append(request.digest)
        return acquired(request)

    rights = PublicationRightsAssessment.create(
        decision="PERMITTED",
        permitted_use="PUBLICATION_EVIDENCE",
        policy_digest="sha256:" + "1" * 64,
        evidence_digest="sha256:" + "2" * 64,
    )
    dependency = DependencyAssessment.create(
        dependency_status="RESOLVED",
        evidential_origin_id="origin-1",
        originating_report_id="origin-1",
        evidence_digest="sha256:" + "3" * 64,
    )
    currentness = SourceCurrentness(
        source_id,
        str(version_request.definition_id),
        source_version.canonical_digest,
        "CURRENT_VERSION",
        source_record["publication_time"],
        source_record["retrieval_time"],
        None,
        str(version_request.version_id),
        "sha256:" + "d" * 64,
        currentness_evidence,
        "PASS",
        "CURRENT_VERSION_CONFIRMED",
    )
    source = NativeEvidenceSource(
        unit,
        source_version,
        rights,
        dependency,
    )
    assessed_package = replace(
        assessed_package,
        source_ids=(source_id,),
        governed_claims=tuple(
            replace(
                claim,
                source_ids=(source_id,),
            )
            for claim in assessed_package.governed_claims
        ),
    )
    model_package = _model_package_value(assessed_package)
    model_claims = model_package["governed_claims"]
    assert [item["claim"] for item in model_claims] == list(
        model_package["substantive_new_information"]
    )
    assert sum(item["claim_role"] == "HEADLINE" for item in model_claims) == 1
    assert {
        item["governed_claim_id"]
        for item in model_package["qualification_evidence"]
    } == {
        item["claim_id"] for item in model_claims if item["claim_role"] == "HEADLINE"
    }
    assert model_package["geography"] == ["UK"]
    assert model_package["categories"] == ["Politics and law"]
    assert all(
        item["claim"] in item["supporting_excerpt"]
        and item["supporting_excerpt"] in passage
        for item in model_claims
    )

    view = build_source_view((passage,), (source_id,))
    spans = {
        view.resolve_range({"first_span_id": segment.span_id,
                            "last_span_id": segment.span_id})[0]: segment.span_id
        for segment in view.segments
    }

    def reference_value(value):
        claim_indexes = {item["claim_id"]: index
                         for index, item in enumerate(value["governed_claims"])}
        claims = []
        for item in value["governed_claims"]:
            claim_span = spans.get(item["claim"], "S99L1")
            support_span = spans.get(item["supporting_excerpt"], "S99L1")
            if item["source_ids"] != [source_id]:
                claim_span = "S99L1"
            claims.append({
                **{key: field for key, field in item.items() if key not in {
                    "claim_id", "claim", "passage_index", "supporting_excerpt",
                    "source_ids", "rendered_assertion_zh_hant_hk",
                }},
                "claim_range": {"first_span_id": claim_span, "last_span_id": claim_span},
                "support_range": {"first_span_id": support_span, "last_span_id": support_span},
                "rendered_fragments": [item["rendered_assertion_zh_hant_hk"]],
            })
        return {"package": {
            "substantive_claim_indexes": [
                next((index for index, item in enumerate(value["governed_claims"])
                      if item["claim"] == text), 99)
                for text in value["substantive_new_information"]
            ],
            "governed_claims": claims,
            "qualification_evidence": [
                {**{key: field for key, field in item.items()
                    if key != "governed_claim_id"},
                 "claim_index": claim_indexes.get(item["governed_claim_id"], 99)}
                for item in value["qualification_evidence"]
            ],
            "selection_rationale": value["selection_rationale"],
            "geography": value["geography"], "categories": value["categories"],
            "explicit_exclusions": value["explicit_exclusions"],
        }}

    def model_assessor(value):
        def dispatch(prompt):
            request = json.loads(prompt)
            assert "body" not in request["sources"][0]
            assert "passages" not in request["base_package"]
            assert "".join(segment["text"] for segment in request["sources"][0]["segments"]) == passage
            return NativeAssessmentExecution(
                canonical_json_bytes(
                    _current_wire_from_v17(reference_value(value))
                ).decode(),
                {},
            )
        return AutonomousNativeEvidenceAssessor(
            dispatch
        )

    assessor = model_assessor(model_package)
    assessed = assessor(
        version, _base_package(assessed_package), (source,), (acquisition,)
    )
    assert assessed.governed_claims
    assert any(
        claim.rendered_assertion_zh_hant_hk != claim.claim
        for claim in assessed.governed_claims
    )
    assert all(claim.source_authority_decision_ids for claim in assessed.governed_claims)
    assert all(
        record["record_id"] and record.get("claim_digest")
        for record in assessed.assessment_records
        if record["record_type"] == "SEMANTIC_RELATION_EVIDENCE"
    )

    def paraphrased_claim(value):
        value["governed_claims"][0]["claim"] = "The deadline was revised."

    def unrecognised_latin_term(value):
        value["governed_claims"][0][
            "rendered_assertion_zh_hant_hk"
        ] = "unsupported term限期安排已經更新。"

    malformed = deepcopy(model_package)
    paraphrased_claim(malformed)
    with pytest.raises(EvidencePackageError, match="source references differ"):
        model_assessor(malformed)(
            version,
            _base_package(assessed_package),
            (source,),
            (acquisition,),
        )

    wrong_source = replace(
        assessed_package,
        governed_claims=(
            replace(assessed_package.governed_claims[0], source_ids=("wrong-source",)),
            *assessed_package.governed_claims[1:],
        ),
    )
    with pytest.raises(EvidencePackageError, match="source references differ"):
        model_assessor(_model_package_value(wrong_source))(
            version, _base_package(assessed_package), (source,), (acquisition,))
    missing_claim = _model_package_value(assessed_package)
    missing_claim["qualification_evidence"][0][
        "governed_claim_id"
    ] = "missing-claim"
    with pytest.raises(EvidencePackageError, match="source references differ"):
        model_assessor(missing_claim)(
            version, _base_package(assessed_package), (source,), (acquisition,))

    def evidence_controller(selected_assessor):
        return NativeEvidenceController(
            objects=system.objects,
            candidate_port=candidate_port,
            evidence_packages=packages,
            transport=EvidenceTransport(acquire),
            assessor=EvidenceAssessor(selected_assessor),
            policy_bundle_digest="sha256:" + "a" * 64,
            transport_policy_digest=transport_digest,
            clock=lambda: UtcTimestamp.parse("2026-09-08T12:02:00Z"),
        )

    candidate_connection.commit()
    bound_inputs = []

    def deny_changed_semantic_input(base, acquired):
        bound_inputs.append((base.digest, tuple(item.receipt_digest for item in acquired)))
        raise NativeEvidenceHold('SEMANTIC_INTENT_INPUT_CHANGED_HOLD', source_id)

    with pytest.raises(NativeEvidenceHold, match='SEMANTIC_INTENT_INPUT_CHANGED_HOLD'):
        evidence_controller(lambda *_: pytest.fail('input must bind before assessment')).acquire_and_retain(
            candidate_version_id=version.version_id,
            intake_receipt_id=acknowledgement.receipt_id, sources=(source,),
            assessment_semantic_only=True, before_semantic_assessment=deny_changed_semantic_input,
            proof=proof(),
        )
    assert len(bound_inputs) == 1 and bound_inputs[0][1] == (acquisition.receipt_digest,)
    negative_value = _model_package_value(assessed_package)
    # Semantic relation is a controller constant in v18; exercise a
    # remaining provider choice rather than pretending it can be emitted.
    negative_value["governed_claims"][0]["status"] = "UNSUPPORTED"
    negative_assessor = model_assessor(negative_value)
    with pytest.raises(EvidencePackageError, match="source references differ"):
        negative_assessor(
            version, _base_package(assessed_package), (source,), (acquisition,)
        )
    # Keep the downstream defence independent of the now-stricter producer.
    negative = replace(assessed, assessment_records=tuple(
        {**record, "source_polarity": "NEGATED"}
        if record["record_type"] == "SEMANTIC_RELATION_EVIDENCE" else record
        for record in assessed.assessment_records
    ))
    with pytest.raises(NativeEvidenceHold, match="EVIDENCE_VALIDATION_HOLD"):
        evidence_controller(lambda *_args: negative).acquire_and_retain(
            candidate_version_id=version.version_id,
            intake_receipt_id=acknowledgement.receipt_id,
            sources=(source,),
            proof=proof(),
        )
    calls.clear()

    controller = evidence_controller(assessor)
    with pytest.raises(NativeEvidenceHold, match="PUBLICATION_RIGHTS_HOLD"):
        controller.acquire_and_retain(
            candidate_version_id=version.version_id,
            intake_receipt_id=acknowledgement.receipt_id,
            sources=(
                source,
                replace(
                    source,
                    rights=PublicationRightsAssessment.create(
                        decision="HOLD",
                        permitted_use=rights.permitted_use,
                        policy_digest=rights.policy_digest,
                        evidence_digest=rights.evidence_digest,
                    ),
                ),
            ),
            proof=proof(),
        )
    assert calls == []
    evidence = controller.acquire_and_retain(
        candidate_version_id=version.version_id,
        intake_receipt_id=acknowledgement.receipt_id,
        sources=(source,),
        before_assessment=lambda: assessment_markers.append("started"),
        proof=proof(),
    )
    assert len(calls) == 1
    assert assessment_markers == ["started"]
    assert evidence.editorial_decision.evaluated_at == "2026-09-08T12:02:00.000000Z"

    bindings = _bindings(tmp_path, registries, hydration, definitions, commands)
    publisher = NativePublicationController(
        objects=system.objects,
        commands=system.commands,
        events=system.events,
        candidate_port=candidate_port,
        evidence_packages=packages,
        bindings=bindings, clock=lambda: FIXED_NOW,
    )

    def free_slug_category(value):
        value["categories"] = ["immigration"]

    def invented_substantive_summary(value):
        value["substantive_new_information"] = [
            "A summary which is not an exact governed claim."
        ]

    def unqualified_headline(value):
        value["qualification_evidence"] = []

    def standing_information(value):
        value["substantive_new_information"] = []
        value["qualification_evidence"] = []

    for mutation, expected_reason in (
        (standing_information, "NO_SUBSTANTIVE_NEW_INFORMATION"),
        (free_slug_category, "UNRECOGNISED_CATEGORY"),
        (
            invented_substantive_summary,
            "INVALID_SUBSTANTIVE_CLAIM_INVENTORY",
        ),
        (unqualified_headline, "UNQUALIFIED_HEADLINE_CLAIM"),
        (unrecognised_latin_term, "INVALID_GOVERNED_CLAIM_EVIDENCE"),
    ):
        malformed = deepcopy(model_package)
        mutation(malformed)
        if mutation is unrecognised_latin_term:
            with pytest.raises(NativeEvidenceHold, match="ASSESSOR_RENDERING_CONTRACT_HOLD"):
                model_assessor(malformed)(
                    version, _base_package(assessed_package), (source,), (acquisition,),
                )
            # Keep the consumer defence independent of producer pre-validation.
            malformed_assessment = replace(
                assessed,
                governed_claims=(
                    replace(
                        assessed.governed_claims[0],
                        rendered_assertion_zh_hant_hk=(
                            malformed["governed_claims"][0]["rendered_assertion_zh_hant_hk"]
                        ),
                    ),
                    *assessed.governed_claims[1:],
                ),
            )
        else:
            # The v18 provider cannot emit these invalid free-text values.
            # Preserve the separate downstream-admission defence by mutating
            # the already materialised assessment, never bypassing the positive
            # current producer path above.
            if mutation is free_slug_category:
                malformed_assessment = replace(assessed, categories=("immigration",))
            elif mutation is invented_substantive_summary:
                malformed_assessment = replace(
                    assessed,
                    substantive_new_information=(
                        "A summary which is not an exact governed claim.",
                    ),
                )
            elif mutation is unqualified_headline:
                malformed_assessment = replace(assessed, qualification_evidence=())
            else:
                malformed_assessment = replace(
                    assessed, substantive_new_information=(), qualification_evidence=(),
                )
        resolved_records = dict(evidence.retained.package.resolved_evidence_records)
        resolved_records.update(
            (
                record["record_id"],
                digest_bytes(canonical_json_bytes(record)),
            )
            for record in malformed_assessment.assessment_records
        )
        malformed_package = replace(
            evidence.retained.package,
            substantive_new_information=(
                malformed_assessment.substantive_new_information
            ),
            governed_claims=malformed_assessment.governed_claims,
            qualification_evidence=malformed_assessment.qualification_evidence,
            selection_rationale=malformed_assessment.selection_rationale,
            geography=malformed_assessment.geography,
            categories=malformed_assessment.categories,
            explicit_exclusions=malformed_assessment.explicit_exclusions,
            resolved_evidence_records=tuple(sorted(resolved_records.items())),
            evidence_gate_results=evidence.editorial_decision.evidence_gate_results,
            evidence_gate_evidence=tuple(
                EvidenceGateEvidence(
                    gate,
                    result,
                    tuple(
                        claim.claim_id
                        for claim in malformed_assessment.governed_claims
                    ),
                    EVIDENCE_GATE_POLICY_VERSION,
                )
                for gate, result in evidence.editorial_decision.evidence_gate_results
            ),
            freshness_result="PASS",
            integrity_result="PASS",
        )
        decision = DeterministicWriteAdmission().decide_candidate_identity(
            candidate_id=malformed_package.candidate_id,
            hypothesis_id=malformed_package.hypothesis_id,
            package=malformed_package,
            decided_at=evidence.editorial_decision.evaluated_at,
        )
        assert decision.decision != "WRITE_READY"
        assert expected_reason in decision.stable_reason_codes

    month_source = "The deadline changed after 30 months."
    month_claim = replace(
        evidence.retained.package.governed_claims[0],
        claim=month_source,
        supporting_excerpt=month_source,
        rendered_assertion_zh_hant_hk="限期安排在30個月後更新。",
        localised_factual_expressions=(("30 months", "30個月"),),
    )
    month_package = replace(
        evidence.retained.package,
        passages=(f"{passage}\n{month_source}",),
        substantive_new_information=(
            month_claim.claim,
            evidence.retained.package.governed_claims[1].claim,
        ),
        governed_claims=(
            month_claim,
            evidence.retained.package.governed_claims[1],
        ),
    )
    title, body, links = required_surface_copy(month_package)
    month_validators = validate_writer_copy(
        WriterCopy(
            title,
            body,
            "newsroom.offline-exact-copy.v1",
            month_package.digest,
            links,
        ),
        month_package,
    )
    assert next(
        item
        for item in month_validators
        if item.validator == "NUMERIC_AND_DATE_FIDELITY"
    ).result == "PASS"
    for source_fact, rendered_fact in (
        ("30 months", "31個月"),
        ("missing months", "30個月"),
        ("30 months", "1296000 minutes"),
    ):
        with pytest.raises(ValueError, match="equivalent exact claim facts"):
            replace(
                month_claim,
                localised_factual_expressions=((source_fact, rendered_fact),),
            )

    year_source = "The applicant will be granted permission for 5 years."
    year_claim = replace(
        evidence.retained.package.governed_claims[0],
        claim=year_source,
        supporting_excerpt=year_source,
        rendered_assertion_zh_hant_hk="申請人會獲批為期5年的許可。",
        localised_factual_expressions=(("5 years", "5年"),),
    )
    assert year_claim.localised_factual_expressions == (("5 years", "5年"),)
    for wrong_unit in ("60個月", "1825日", "2628000分鐘"):
        with pytest.raises(ValueError, match="equivalent exact claim facts"):
            replace(
                year_claim,
                localised_factual_expressions=(("5 years", wrong_unit),),
            )

    published = publisher.advance(
        evidence.retained.package_admission_id,
        evidence.editorial_decision,
        expected_story_version=0,
        expected_publication_version=0,
        expected_delivery_evidence_version=0,
        proof=proof(),
    )
    reader = open_private_serving_read_port(
        bindings.target_path,
        target_id=bindings.target_id,
        target_context_digest=bindings.target_context_digest,
        proof=published.read_proof,
    )
    assert tuple(row.surface_kind for row in reader.acknowledged_rows().rows) == (
        "ARTICLE",
        "FEED_CARD",
    )
    reader.close()
    publisher.close()
    system.close()
    ingress.close()
    candidate_connection.close()


def test_retained_record_replay_binds_exact_revision_bytes(tmp_path, monkeypatch):
    system, *_ = _open(tmp_path / "records.sqlite3")
    try:
        first = canonical_json_bytes({
            "record_id": "stable-rights-policy", "record_type": "RIGHTS_DECISION",
            "base_package_digest": "sha256:" + "1" * 64,
        })
        second = canonical_json_bytes({
            "record_id": "stable-rights-policy", "record_type": "RIGHTS_DECISION",
            "base_package_digest": "sha256:" + "2" * 64,
        })
        first_id = _admit_record(system.objects, first, proof())
        second_id = _admit_record(system.objects, second, proof())
        assert first_id != second_id
        assert _admit_record(system.objects, first, proof()) == first_id
        assert _admit_record(system.objects, second, proof()) == second_id
        from newsroom.authority import ObjectAdmissionRequest
        stale = system.objects.admit(
            ObjectAdmissionRequest("evidence.record", f"record:{digest_bytes(first)}"),
            first, proof=proof(),
        )
        monkeypatch.setattr(type(system.objects), "admit", lambda *_args, **_kwargs: stale)
        with pytest.raises(NativeEvidenceError, match="record content differs"):
            _admit_record(system.objects, second, proof())
    finally:
        system.close()


def test_historical_observation_receipt_is_explicit_and_legacy_bytes_stay_exact():
    body = b'official captured source'
    values = dict(
        request_digest='sha256:' + '1' * 64, outcome='COMPLETE',
        canonical_url='https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=warnsum&lang=tc',
        body=body, body_digest=digest_bytes(body), publisher='Hong Kong Observatory',
        responsible_body='Hong Kong Observatory', source_type='PRIMARY_OFFICIAL',
        publication_time='2026-09-19T22:00:00.000000Z',
        source_updated_time='2026-09-20T14:20:00.000000Z',
        retrieval_time='2026-09-20T17:00:00.000000Z', geography='Hong Kong', language='zh-HK',
        transport_evidence_digest='sha256:' + '2' * 64,
        currentness_basis='AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT',
    )
    legacy = AcquiredEvidence.create(**values)
    assert 'body_origin' not in json.loads(legacy.receipt_bytes)
    sourced = AcquiredEvidence.create(**{**values,'body_origin':'GOVUK_CONTENT_API_PAGE_TEXT'})
    assert json.loads(sourced.receipt_bytes)['body_origin']=='GOVUK_CONTENT_API_PAGE_TEXT'
    assert sourced.receipt_digest!=legacy.receipt_digest
    assert 'source_observed_time' not in json.loads(legacy.receipt_bytes)
    assert AcquiredEvidence(**{**json.loads(legacy.receipt_bytes), 'exclusion_signals': (),
                               'body': body, 'receipt_digest': legacy.receipt_digest}) == legacy
    historical = AcquiredEvidence.create(**{**values,
        'currentness_basis': 'RETAINED_AUTHORITATIVE_COMPLETED_EVENT',
        'source_observed_time': '2026-09-20T14:22:35.000000Z',
    })
    assert json.loads(historical.receipt_bytes)['source_observed_time'] == '2026-09-20T14:22:35.000000Z'
    for stamp in ('', '2026-09-20T14:00:00.000000Z', '2026-09-21T14:22:35.000000Z'):
        with pytest.raises(ValueError):
            AcquiredEvidence.create(**{**values,
                'currentness_basis': 'RETAINED_AUTHORITATIVE_COMPLETED_EVENT',
                'source_observed_time': stamp,
            })


def test_completed_observation_cannot_be_promoted_to_current_warning():
    from types import SimpleNamespace as NS
    from newsroom.increment10.editorial import SourceCurrentness
    from newsroom.control_plane.native_evidence import _INTEGRITY_CHECKS

    digest = 'sha256:' + 'a' * 64
    source = NS(unit=NS(source_id='HK-02', authority=NS(definition_id='definition-hko')),
                source_version=NS(canonical_digest=digest))
    acquired = NS(currentness_basis='RETAINED_AUTHORITATIVE_COMPLETED_EVENT',
                  publication_time='2026-09-19T22:00:00.000000Z',
                  source_updated_time='2026-09-20T14:20:00.000000Z',
                  retrieval_time='2026-09-20T17:00:00.000000Z', transport_evidence_digest=digest)
    currentness = SourceCurrentness(
        'HK-02', 'definition-hko', digest, 'COMPLETED_HISTORICAL_EVENT',
        acquired.publication_time, acquired.retrieval_time, None,
        acquired.source_updated_time, None, digest, 'PASS', 'RETAINED_COMPLETED_EVENT_CONFIRMED',
    )
    assessment = AcquiredSourceAssessment('HK-02', currentness,
                                         tuple((name, 'PASS') for name in _INTEGRITY_CHECKS))
    assert NativeEvidenceController._validated_source_assessments(
        (source,), (acquired,), (assessment,),
    ) == (assessment,)
    for changed in (
        replace(currentness, currency_family='CURRENT_VERSION', supersession_evidence_digest=digest),
        replace(currentness, version_reference='different source update'),
    ):
        with pytest.raises(NativeEvidenceHold, match='SOURCE_ASSESSMENT_HOLD'):
            NativeEvidenceController._validated_source_assessments(
                (source,), (acquired,), (replace(assessment, currentness=changed),),
            )


@pytest.fixture
def retained_page_text_source_package(tmp_path):
    """One real disposable controller/package path, with no transport effects."""
    from types import SimpleNamespace
    candidate_connection, candidate_port, version = _candidate(tmp_path)
    ingress = open_evidence_intake_ingress(tmp_path / 'body-intake.sqlite3')
    acknowledgement = _receive(ingress, candidate_connection, candidate_port, version,
        request_id='page-body-provenance')
    def current(version_id):
        candidate_connection.execute('BEGIN')
        try:
            return candidate_port.require_retained_version_in_transaction(version_id)
        finally:
            candidate_connection.rollback()
    port = candidate_port._with_bounded_version(current)
    path = tmp_path / 'body-objects.sqlite3'
    system, registries, *_ = _open(path)
    packages = _evidence_facade(system, ingress, registries)
    passage, ready, records = _ready_package(version)
    source_record = next(r for r in records if r['record_type'] == 'SOURCE_RECORD')
    unit = replace(_unit(), headline='', body=passage, canonical_url=source_record['canonical_url'])
    unit = replace(unit, effective_revision=replace(unit.effective_revision, revision_digest=unit.revision_digest))
    request = _source_requests(unit, _rights())[1]
    source_version = SourceDefinitionVersion(request, EventId.new(), 1, NOW, request.digest)
    rights = PublicationRightsAssessment.create(decision='PERMITTED', permitted_use='PUBLICATION_EVIDENCE',
        policy_digest='sha256:' + '1' * 64, evidence_digest='sha256:' + '2' * 64)
    dependency = DependencyAssessment.create(dependency_status='RESOLVED', evidential_origin_id='origin-1',
        originating_report_id='origin-1', evidence_digest='sha256:' + '3' * 64)
    source = NativeEvidenceSource(unit, source_version, rights, dependency)
    wire = _model_package_value(replace(ready, source_ids=(unit.source_id,),
        governed_claims=tuple(replace(c, source_ids=(unit.source_id,)) for c in ready.governed_claims)))
    acquisitions = []
    def acquire(request):
        result = AcquiredEvidence.create(request_digest=request.digest, outcome='COMPLETE',
            canonical_url=unit.canonical_url, body=passage.encode(), body_digest=digest_bytes(passage.encode()),
            **{name: source_record[name] for name in ('publisher', 'responsible_body', 'source_type',
                'publication_time', 'retrieval_time', 'geography', 'language')},
            source_updated_time=source_record['publication_time'],
            transport_evidence_digest='sha256:' + 'c' * 64,
            currentness_basis='AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT', body_origin='GOVUK_CONTENT_API_PAGE_TEXT')
        acquisitions.append(result)
        return result
    def assess(candidate, base, sources, acquired):
        return AutonomousNativeEvidenceAssessor._validated_execution(
            NativeAssessmentExecution(canonical_json_bytes({'package': wire}).decode(), {}),
            candidate, base, sources, acquired)
    controller = NativeEvidenceController(objects=system.objects, candidate_port=port,
        evidence_packages=packages, transport=EvidenceTransport(acquire), assessor=EvidenceAssessor(assess),
        policy_bundle_digest='sha256:' + 'a' * 64, transport_policy_digest='sha256:' + '7' * 64,
        clock=lambda: UtcTimestamp.parse('2026-09-08T12:02:00Z'))
    candidate_connection.commit()
    try:
        result = controller.acquire_and_retain(candidate_version_id=version.version_id,
            intake_receipt_id=acknowledgement.receipt_id, sources=(source,), proof=proof())
        yield SimpleNamespace(system=system, path=path, packages=packages, candidate_port=port,
            registries=registries, ingress=ingress, retained=result.retained, acquisitions=acquisitions)
    finally:
        system.close()
        ingress.close()
        candidate_connection.close()


def test_page_text_source_provenance_survives_package_read_and_reopen(retained_page_text_source_package):
    from newsroom.authority import HydrationRequest
    from newsroom.increment10.evidence import _PACKAGE_PURPOSE
    case = retained_page_text_source_package
    retained = case.retained
    assert len(case.acquisitions) == 1
    acquired = case.acquisitions[0]
    expected = {'version': 'newsroom.acquisition-body-provenance.v1', 'kind': 'GOVUK_CONTENT_API_PAGE_TEXT',
        'body_digest': acquired.body_digest, 'acquisition_receipt_digest': acquired.receipt_digest,
        'transport_evidence_digest': acquired.transport_evidence_digest}
    assert retained.source_records[0]['body_provenance'] == expected
    assert retained.source_records[0]['record_id'] == acquired.receipt_digest
    raw = case.system.objects.rehydrate(HydrationRequest(retained.package_admission_id, _PACKAGE_PURPOSE), proof=proof()).data
    envelope = json.loads(raw)
    assert 'source_records' not in envelope and 'body_provenance' not in envelope['package']
    reread = case.packages.read(retained.package_admission_id, candidate_port=case.candidate_port, proof=proof())
    assert reread == retained
    case.system.close()
    reopened, registries, *_ = _open(case.path)
    try:
        packages = _evidence_facade(reopened, case.ingress, registries)
        assert packages.read(retained.package_admission_id, candidate_port=case.candidate_port, proof=proof()) == retained
        assert reopened.objects.rehydrate(HydrationRequest(retained.package_admission_id, _PACKAGE_PURPOSE), proof=proof()).data == raw
        assert len(case.acquisitions) == 1
    finally:
        reopened.close()


@pytest.mark.parametrize('tamper', ('body_digest', 'acquisition_receipt_digest', 'unknown_field'))
def test_page_text_source_provenance_tamper_denies_retained_package(retained_page_text_source_package, tamper):
    from newsroom.authority import HydrationRequest
    case = retained_page_text_source_package
    retained = case.retained
    original = deepcopy(retained.source_records[0])
    malformed = deepcopy(original)
    if tamper == 'unknown_field':
        malformed['body_provenance']['unsupported_schema_extension'] = True
    else:
        malformed['body_provenance'][tamper] = 'sha256:' + 'f' * 64
    forged_id = _admit_record(case.system.objects, canonical_json_bytes(malformed), proof())
    ids = []
    for admission_id in retained.record_admission_ids:
        record = json.loads(case.system.objects.rehydrate(HydrationRequest(admission_id, 'evidence.record'), proof=proof()).data)
        ids.append(forged_id if record['record_type'] == 'SOURCE_RECORD' else admission_id)
    with pytest.raises(EvidencePackageError, match='governed evidence records differ'):
        case.packages.retain(retained.package, receipt_id=retained.receipt_id,
            candidate_port=case.candidate_port, source_admission_ids=retained.source_admission_ids,
            record_admission_ids=tuple(ids), proof=proof())
    assert case.packages.read(retained.package_admission_id, candidate_port=case.candidate_port, proof=proof()) == retained
    assert retained.source_records[0] == original
