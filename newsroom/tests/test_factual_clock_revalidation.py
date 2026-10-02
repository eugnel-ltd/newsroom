"""Consumer-only clock revalidation preserves retained producer evidence."""

import json
import sqlite3
from contextlib import nullcontext
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane.admission import DeterministicWriteAdmission, WriteAdmissionDecision, _decision_id
from newsroom.control_plane.native_assessor import (
    AutonomousNativeEvidenceAssessor, NativeAssessmentExecution,
    assessment_revalidation_due, same_assessment_producer,
)
from newsroom.control_plane.native_composition import ASSESSMENT_CONTRACT_VERSION
from newsroom.control_plane.native_evidence import rights_eligibility_digest
from newsroom.tests.test_native_assessor import (
    _qualification_assessor_inputs, _usage, _use_historical_v16, retained_22589_assessment,
)
from newsroom.tests.test_zero_quota_write_loop import _candidate_package
OLD_ASSESSMENT_CONTRACT = (
    "newsroom.native-evidence-assessor.v20+newsroom.named-entity.v15+"
    "newsroom.zh-hant-hk-shape.v14+newsroom.factual-localisation.v1+"
    "newsroom.qualification-relation.v3+newsroom.retained-assessment.v1"
)
CLOCK_UPGRADED_CONTRACT = OLD_ASSESSMENT_CONTRACT.replace(
    "+newsroom.factual-localisation.v1+", "+newsroom.factual-localisation.v2+",
)
OLD_WRITE_POLICY = (
    "newsroom.write-admission.v9+newsroom.evid-012.v7+newsroom.evidence-approval.v8+"
    "newsroom.evidence-gates.v2+newsroom.governed-claim.v7+newsroom.governed-input.v10+"
    "newsroom.named-entity.v15+newsroom.cont-originality.v3+newsroom.zh-hant-hk-shape.v14+"
    "newsroom.factual-localisation.v1+newsroom.qualification-relation.v3"
)

def test_clock_consumer_upgrade_revalidates_cached_result_without_relabelling(
    tmp_path, monkeypatch, retained_22589_assessment,
):
    facts = {"reason": "ASSESSOR_LOCALISATION_CONTRACT_HOLD",
             "assessment_contract_version": OLD_ASSESSMENT_CONTRACT}
    # Freeze the consumer-only change; today's wire may use a newer producer.
    assert assessment_revalidation_due(facts, CLOCK_UPGRADED_CONTRACT)
    assert same_assessment_producer(OLD_ASSESSMENT_CONTRACT, CLOCK_UPGRADED_CONTRACT)
    assert "+newsroom.factual-localisation.v2+" in ASSESSMENT_CONTRACT_VERSION
    _use_historical_v16(monkeypatch)
    candidate, base, source, acquired, raw = _qualification_assessor_inputs(retained_22589_assessment)
    source_date, rendered_date = "1 October 2026 at 06:00", "2026年10月1日06:00"
    text = f"The deadline changed on {source_date}."
    raw["package"]["governed_claims"][0].update(
        claim=text, supporting_excerpt=text,
        rendered_assertion_zh_hant_hk=f"限期於{rendered_date}更改。",
        localised_factual_expressions=[[source_date, rendered_date]],
    )
    raw["package"]["substantive_new_information"] = [text]
    raw["package"]["qualification_evidence"][0]["test_evidence"].update(
        material_relation_span=text, reader_action=text,
    )
    body = text.encode()
    base = replace(base, passages=(text,), observation_digests=(digest_bytes(body),))
    acquired = SimpleNamespace(**{**vars(acquired), "body": body, "body_digest": digest_bytes(body),
        "rights_eligibility_digest": rights_eligibility_digest(source.rights, body_digest=digest_bytes(body),
            transport_digest=acquired.transport_evidence_digest, exclusion_signals=(), text_only=True)})
    service, usage = _usage(tmp_path, monkeypatch)
    execution = NativeAssessmentExecution(canonical_json_bytes(raw).decode(), {
        "usage_basis": "PROVIDER_REPORTED", "input_tokens": 1, "output_tokens": 1,
        "cached_read_tokens": 0, "cached_write_tokens": 0, "reasoning_tokens": 0,
        "context_tokens": 1, "total_tokens": 2,
    })
    allocation = usage.begin(candidate, base, "retained clock fixture")
    dispatch_at = usage.mark_dispatch(allocation)
    assert usage.retain_result(allocation, execution, dispatch_at=dispatch_at)
    usage.complete(allocation, outcome="ASSESSOR_VALIDATION_FAILED", execution=execution,
                   provider_dispatched=True, dispatch_at=dispatch_at, failure_class="ASSESSMENT_VALIDATION_FAILED")

    def retained_rows():
        with sqlite3.connect(service.path) as connection:
            return tuple(connection.execute(query).fetchall() for query in (
                "SELECT record_json FROM model_invocation_allocations",
                "SELECT record_json FROM model_invocation_terminals",
                "SELECT payload_digest,payload_json FROM ledger",
            ))
    before = retained_rows()
    assessor = AutonomousNativeEvidenceAssessor(lambda *_: pytest.fail("consumer-only revalidation dispatched"),
                                              usage=usage, dispatch_fence=nullcontext)
    for _ in range(2):
        result = assessor.assess_with_boundary(candidate, base, (source,), (acquired,),
                                              before_dispatch=None, cached_only=True)
        assert result.governed_claims[0].localised_factual_expressions == ((source_date, rendered_date),)
        assert len(result.qualification_evidence) == 1
    assert retained_rows() == before
    assert json.loads(before[0][0][0])["prompt_contract_version"] == "newsroom.native-evidence-assessor.v16"
    facts["assessment_contract_version"] = CLOCK_UPGRADED_CONTRACT
    assert not assessment_revalidation_due(facts, CLOCK_UPGRADED_CONTRACT)

def test_prior_factual_v1_write_admission_remains_readable():
    candidate, package = _candidate_package()
    values = asdict(DeterministicWriteAdmission().decide(candidate, package, decided_at="2026-10-01T06:00:00Z"))
    values.pop("decision_id")
    decided_at = values.pop("decided_at")
    values["policy_version"] = OLD_WRITE_POLICY
    old = WriteAdmissionDecision(decision_id=_decision_id(**values), decided_at=decided_at, **values)
    assert WriteAdmissionDecision.from_record(old.as_record()) == old
