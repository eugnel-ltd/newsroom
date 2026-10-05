"""Complete one exact retained HKO witness without a new provider attempt."""

import json
import sqlite3
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from newsroom.authority import UtcTimestamp
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane import native_assessor as module
from newsroom.control_plane.admission import _qualification_relation_is_proven
from newsroom.control_plane.evidence import validate_governed_evidence_records
from newsroom.control_plane.graphiti_operational_readiness import (
    OPERATOR_AUTHORITY_DOMAIN, OPERATOR_PRINCIPAL_ID,
)
from newsroom.control_plane.model_usage import ModelUsageService
from newsroom.control_plane.native_assessor import (
    AutonomousNativeEvidenceAssessor, NativeAssessmentExecution, NativeAssessmentUsage,
)
from newsroom.control_plane.native_evidence import (
    AcquiredEvidence, EvidenceAcquisitionRequest, EvidenceAssessor, NativeEvidenceController, NativeEvidenceHold,
    PublicationRightsAssessment,
)
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.native_source_intake import native_evidence_sources
from newsroom.increment10.evidence import _base_package
from newsroom.tests.test_increment10_editorial import _ready_package
from newsroom.tests.test_increment10_ingress import _candidate
from newsroom.tests.test_native_assessor import _usage
from newsroom.tests.test_native_runtime import _args
from newsroom.tests.test_native_weather_evidence import (
    CONTROLLER_POLICY_DIGEST, _Response, _acquisition, _portfolio,
)
from newsroom.tests.test_native_weather_sources import _poll

NOW = datetime(2026, 10, 4, 9, 7, 33, tzinfo=UTC)
# The exact source fields and v23 output of revision 294ba394. No prompt or secrets.
RAW_SOURCE = {"WTS": {
    "actionCode": "CANCEL", "code": "WTS", "expireTime": "2026-10-04T15:15:00+08:00",
    "issueTime": "2026-10-04T13:03:00+08:00", "name": "雷暴警告",
    "updateTime": "2026-10-04T15:15:00+08:00",
}}
RAW_RESPONSE = {"package": {
    "categories": ["Weather and disasters"], "explicit_exclusions": [],
    "geography": ["Hong Kong"], "select_new_information": True,
    "governed_claims": [{
        "claim_role": "HEADLINE",
        "factual_localisations": [{"rendered_expression": "2026年10月4日15時15分",
                                   "source_lookup_key": "4 October 2026 at 15:15"}],
        "quotation_source_keys": [],
        "rendered_assertion_zh_hant_hk_fragments": [
            "", "天文台已取消雷暴警告；官方紀錄於", "時間2026年10月4日15時15分更新。",
        ],
        "source_range": {"first_span_id": "S1L6", "last_span_id": "S1L6"},
        "status": "CONFIRMED_FACT",
    }],
    "qualification_evidence": [{"claim_index": 0, "test": "LAW_RIGHT_STATUS_POLICY",
        "test_evidence": {"change_kind": "STATUS", "change_relation": "NEW_OR_CHANGED_STATE",
            "event_polarity": "AFFIRMED",
            "material_relation_span_source_lookup_key": "香港天文台 cancelled the 雷暴警告",
            "new_state_source_lookup_key": "cancelled the 雷暴警告"}}],
    "selection_rationale": (
        "S1L6 is the supplied required historical headline. It affirms that official status "
        "changed when 香港天文台 cancelled the 雷暴警告, and it states the warning-record update "
        "time as 4 October 2026 at 15:15 (香港時間). The Hong Kong Chinese rendering and the "
        "date pair follow that supplied headline. The cancellation is an affirmed STATUS "
        "change, so the headline qualifies."
    ),
}}
USAGE = {"usage_basis": "PROVIDER_REPORTED", "input_tokens": 1, "output_tokens": 1,
         "cached_read_tokens": 0, "cached_write_tokens": 0, "reasoning_tokens": 0,
         "context_tokens": 1, "total_tokens": 2}


@contextmanager
def _inputs(tmp_path, monkeypatch):
    monkeypatch.setattr("newsroom.tests.test_native_weather_sources.NOW", NOW)
    monkeypatch.setattr("newsroom.tests.test_native_weather_evidence.NOW", NOW)
    args = _args(tmp_path, monkeypatch)
    args.update(principal_id=OPERATOR_PRINCIPAL_ID,
                authority_domain=OPERATOR_AUTHORITY_DOMAIN,
                clock=lambda: UtcTimestamp.parse("2026-10-04T09:07:33.000000Z"))
    with open_native_runtime(**args) as runtime:
        rights = _portfolio(runtime, monkeypatch)
        _, disposition = _poll(runtime, "HK-02", canonical_json_bytes(RAW_SOURCE), rights, [])
        unit = next(item for item in disposition.units if item.item_key == "WTS")
        observations = {item[1]: item for item in disposition.observations}
        source, = native_evidence_sources(
            units=(unit,), sources=runtime.authority.sources, objects=runtime.authority.objects,
            observations=observations, licence=rights, proof=runtime.proof,
        )
        request = EvidenceAcquisitionRequest(
            unit.source_id, unit.authority.definition_id, unit.authority.definition_version_id,
            source.source_version.canonical_digest, unit.authority.revision_id,
            unit.canonical_url, CONTROLLER_POLICY_DIGEST,
        )
        class Opener:
            def open(self, http, timeout):
                return _Response(b"{}", http.full_url, "application/json")
        monkeypatch.setattr("urllib.request.build_opener", lambda *_: Opener())
        acquired = _acquisition(runtime, rights, [],
            retained_units={unit.revision_id: (unit,)}, observations=observations)(request)
        assert acquired.currentness_basis == "RETAINED_AUTHORITATIVE_COMPLETED_EVENT"
        path = tmp_path / "candidate"
        path.mkdir()
        connection, _, candidate = _candidate(path)
        try:
            base = replace(_base_package(_ready_package(candidate)[1]),
                source_ids=("HK-02",), passages=(acquired.body.decode(),),
                observation_digests=(acquired.body_digest,))
            yield candidate, base, source, acquired
        finally:
            connection.close()


def _retained_rows(path):
    with sqlite3.connect(path) as connection:
        return {table: connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                for table in ("model_invocation_allocations", "model_invocation_terminals",
                              "model_invocation_policies", "model_usage_current", "ledger")}


def test_exact_failed_v23_hko_response_replays_without_allocation_or_terminal_rewrite(
    tmp_path, monkeypatch,
):
    with _inputs(tmp_path, monkeypatch) as (candidate, base, source, acquired):
        service, usage = _usage(tmp_path, monkeypatch)
        usage._clock = lambda: NOW
        raw = canonical_json_bytes(RAW_RESPONSE).decode()
        calls = []
        def dispatch(request):
            calls.append(request)
            return NativeAssessmentExecution(raw, dict(USAGE))
        assessor = AutonomousNativeEvidenceAssessor(dispatch, usage=usage, dispatch_fence=nullcontext)
        # Seed the settled failure under the old consumer, never alter retained bytes.
        with monkeypatch.context() as legacy:
            legacy.setattr(module, "_complete_hko_qualification_clause",
                           lambda item, claim, result, base: item, raising=False)
            with pytest.raises(NativeEvidenceHold, match="ASSESSOR_QUALIFICATION_CONTRACT_HOLD"):
                EvidenceAssessor(assessor).assess(candidate, base, (source,), (acquired,))
        before = _retained_rows(service.path)
        assert len(before["model_invocation_allocations"]) == 1
        assert before["model_invocation_terminals"][0][3:5] == (
            "ASSESSOR_VALIDATION_FAILED", "ASSESSMENT_VALIDATION_FAILED")
        reopened = ModelUsageService(service.path)
        retained_usage = NativeAssessmentUsage(reopened, usage._policy, clock=lambda: NOW)
        replay = AutonomousNativeEvidenceAssessor(
            lambda _: pytest.fail("settled consumer repair must not dispatch"),
            usage=retained_usage, dispatch_fence=nullcontext,
        )
        repaired = EvidenceAssessor(replay).assess(
            candidate, base, (source,), (acquired,), cached_only=True)
        assert len(calls) == 1
        assert _retained_rows(service.path) == before
        original, = retained_usage.retained_assessments(candidate, base)
        original_claim = json.loads(original.execution.text)["package"]["governed_claims"][0]
        claim = repaired.governed_claims[0]
        assert claim.claim == original_claim["claim"]
        assert claim.supporting_excerpt == original_claim["supporting_excerpt"]
        assert claim.rendered_assertion_zh_hant_hk == original_claim["rendered_assertion_zh_hant_hk"]
        evidence = repaired.qualification_evidence[0]
        assert dict(evidence.test_evidence)["material_relation_span"] == claim.supporting_excerpt
        assert dict(evidence.test_evidence)["new_state"] == "cancelled the 雷暴警告"
        old_evidence = json.loads(original.execution.text)["package"]["qualification_evidence"][0]
        assert evidence.qualification_record_id != module._qualification_record_id(
            old_evidence["governed_claim_id"], old_evidence["test"], old_evidence["test_evidence"])
        assert _qualification_relation_is_proven(evidence, claim, source_context=acquired.body.decode())
        governed = replace(base,
            substantive_new_information=repaired.substantive_new_information,
            governed_claims=repaired.governed_claims, qualification_evidence=repaired.qualification_evidence,
            selection_rationale=repaired.selection_rationale, geography=repaired.geography,
            categories=repaired.categories, explicit_exclusions=repaired.explicit_exclusions)
        records = NativeEvidenceController._records(base, governed, (source,), (acquired,), repaired)
        assert validate_governed_evidence_records(
            candidate_id=candidate.candidate_id, source_inventory=((source.unit.source_id, acquired.canonical_url),),
            base_package_digest=base.digest, package=governed,
            retained_records=tuple((item["record_id"], item["record_type"], canonical_json_bytes(item).decode(),
                                    digest_bytes(canonical_json_bytes(item))) for item in records),
        )
        changed_base = replace(base, observation_digests=(digest_bytes(b"different source"),))
        with pytest.raises(NativeEvidenceHold, match="ASSESSOR_REVALIDATION_INPUT_CHANGED_HOLD"):
            EvidenceAssessor(replay).assess(candidate, changed_base, (source,), (acquired,), cached_only=True)
        denied = replace(source, rights=PublicationRightsAssessment.create(
            decision="HOLD", permitted_use=source.rights.permitted_use,
            policy_digest=source.rights.policy_digest, evidence_digest=source.rights.evidence_digest))
        with pytest.raises(NativeEvidenceHold, match="SOURCE_POLICY_FACTS_HOLD"):
            EvidenceAssessor(replay).assess(candidate, base, (denied,), (acquired,), cached_only=True)
        assert _retained_rows(service.path) == before


def _changed_acquisition(acquired, **changes):
    values = {name: getattr(acquired, name) for name in acquired.__dataclass_fields__
              if name != "receipt_digest"}
    values.update(changes)
    return AcquiredEvidence.create(**values)


def test_completion_denies_unbound_ambiguous_quoted_or_changed_evidence(tmp_path, monkeypatch):
    from newsroom.control_plane.native_assessor_spans import build_lossless_source_view
    from newsroom.control_plane.native_assessor_wire import materialise
    from newsroom.increment10.evidence import EvidencePackageError

    with _inputs(tmp_path, monkeypatch) as (candidate, base, source, acquired):
        _, receipt = materialise(
            canonical_json_bytes(RAW_RESPONSE),
            build_lossless_source_view((acquired.body.decode(),), ("HK-02",)),
            "fixture-request", provider_schema=module.PROVIDER_SCHEMA,
            v17_schema=module._V17_PROVIDER_SCHEMA,
        )
        materialised = json.loads(receipt["materialised_text"])
        for label, changes in (
            ("wrong_classification", {"change_kind": "PUBLIC_POLICY"}),
            ("wrong_state", {"new_state": "issued the 雷暴警告"}),
            ("ambiguous_fragment", {"material_relation_span": "at"}),
            ("quoted_fragment", {"material_relation_span": '"香港天文台 cancelled the 雷暴警告"'}),
            ("negated_fragment", {"material_relation_span": "香港天文台 did not cancel the 雷暴警告"}),
            ("outside_selected_clause", {"material_relation_span": "no exact cancellation time is asserted"}),
        ):
            value = json.loads(canonical_json_bytes(materialised))
            value["package"]["qualification_evidence"][0]["test_evidence"].update(changes)
            with pytest.raises(EvidencePackageError, match="qualification"):
                module.AutonomousNativeEvidenceAssessor._validated_execution(
                    NativeAssessmentExecution(canonical_json_bytes(value).decode(), {}),
                    candidate, base, (source,), (acquired,),
                )
        for changed_base in (
            replace(base, observation_digests=(digest_bytes(b"unbound"),)),
            replace(base, passages=("Unbound source passage.",)),
        ):
            with pytest.raises(EvidencePackageError, match="qualification"):
                module.AutonomousNativeEvidenceAssessor._validated_execution(
                    NativeAssessmentExecution(receipt["materialised_text"], {}),
                    candidate, changed_base, (source,), (acquired,),
                )
        for changed_body in (
            acquired.body + b"\nQuoted status: cancelled.",
            acquired.body + b"\n" + module._required_historical_headline(acquired)["claim"].encode(),
            acquired.body.replace(b'"CANCEL"', b'"ISSUE"'),
        ):
            with pytest.raises((NativeEvidenceHold, ValueError),
                               match="HISTORICAL_TIME_RELATION_HOLD|explicit cancellation"):
                module.AutonomousNativeEvidenceAssessor._validated_execution(
                    NativeAssessmentExecution(receipt["materialised_text"], {}),
                    candidate, base, (source,), (_changed_acquisition(acquired, body=changed_body, body_digest=digest_bytes(changed_body)),),
                )
        # Ordinary/current sources retain their existing qualification contract.
        with pytest.raises(EvidencePackageError, match="qualification"):
            module.AutonomousNativeEvidenceAssessor._validated_execution(
                NativeAssessmentExecution(receipt["materialised_text"], {}),
                candidate, base, (source,),
                (_changed_acquisition(acquired, currentness_basis="AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT"),),
            )


def test_consumer_suffix_routes_cached_revalidation_without_changing_producer():
    from newsroom.control_plane.native_composition import ASSESSMENT_CONTRACT_VERSION
    parts = ASSESSMENT_CONTRACT_VERSION.split("+")
    assert parts.count(module.QUALIFICATION_CLAUSE_CONSUMER_VERSION) == 1
    previous = "+".join(part for part in parts if part != module.QUALIFICATION_CLAUSE_CONSUMER_VERSION)
    assert module.VERSION == "newsroom.native-evidence-assessor.v23"
    assert module.same_assessment_producer(previous, ASSESSMENT_CONTRACT_VERSION)
    assert module.assessment_revalidation_due(
        {"reason": "ASSESSOR_QUALIFICATION_CONTRACT_HOLD", "assessment_contract_version": previous},
        ASSESSMENT_CONTRACT_VERSION,
    )
