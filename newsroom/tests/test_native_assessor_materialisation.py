"""Integration proof for reference-result materialisation into governed evidence."""

import json
import sqlite3
from contextlib import nullcontext

import pytest

from dataclasses import replace
from types import SimpleNamespace

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.control_plane.admission import _qualification_relation_is_proven
from newsroom.control_plane.evidence import EVID_012_POLICY_VERSION, validate_governed_evidence_records
from newsroom.control_plane.govuk_spreadsheet import _Text, _csv
from newsroom.control_plane.native_assessor import (
    AutonomousNativeEvidenceAssessor,
    NativeAssessmentExecution,
    SCHEMA, PROVIDER_SCHEMA, VERSION, _V17_PROVIDER_SCHEMA,
)
from newsroom.control_plane.native_assessor_references import (
    build_source_view,
    materialise as materialise_v17, SourceReferenceError,
)
from newsroom.control_plane.native_assessor_wire import materialise as materialise_v18
from newsroom.control_plane.native_evidence import NativeEvidenceController, NativeEvidenceHold, rights_eligibility_digest
from newsroom.tests.test_native_assessor import (
    _qualification_assessor_inputs, _usage, _v18_wire_from_v17,
    retained_22589_assessment,
)


def _reference_claim(span_id, role, fragments, *, localised=(), quotations=()):
    return {
        "claim_role": role,
        "claim_range": {"first_span_id": span_id, "last_span_id": span_id},
        "support_range": {"first_span_id": span_id, "last_span_id": span_id},
        "rendered_fragments": list(fragments),
        "status": "CONFIRMED_FACT",
        "semantic_relation": {
            "source_modality": "ASSERTED",
            "rendered_modality": "ASSERTED",
            "source_polarity": "AFFIRMED",
            "rendered_polarity": "AFFIRMED",
            "relation": "SEMANTICALLY_EQUIVALENT",
        },
        "localised_factual_expressions": [list(item) for item in localised],
        "quotations": list(quotations),
        "certainty": "CONFIRMED",
        "originality_basis": "FACTUAL_REWRITE_REQUIRED",
        "originality_policy_version": "newsroom.cont-originality.v3",
        "admitted_use": "PUBLICATION_EVIDENCE",
        "policy_version": "newsroom.governed-claim.v7",
    }


def _literal_reference_inputs():
    """Synthetic table evidence proves the interface, not a production news fact."""

    candidate, base, source, acquired, _raw = _qualification_assessor_inputs(
        retained_22589_assessment.__wrapped__()
    )
    title = "The department announced the application deadline changed to 30 June 2026"
    csv_bytes = (
        "Senior Official's Name ,Date ,Name of individual or organisation ,Purpose of Meeting\r\n"
        'Marianthi Leontaridi,2026-05-21,Boston Consulting Group,"'
        + title
        + '"\r\n'
    ).encode()
    output = _Text()
    _csv(csv_bytes, output)
    row = output.lines[-1]
    body = (
        title
        + "\n\nAttachment: https://assets.publishing.service.gov.uk/media/fixture/meetings.csv\n"
        + "\n".join(output.lines)
    )
    encoded = body.encode()
    base = replace(
        base,
        passages=(body,),
        observation_digests=(digest_bytes(encoded),),
    )
    acquired = SimpleNamespace(
        **{
            **vars(acquired),
            "body": encoded,
            "body_digest": digest_bytes(encoded),
            "rights_eligibility_digest": rights_eligibility_digest(
            source.rights,
            body_digest=digest_bytes(encoded),
            transport_digest=acquired.transport_evidence_digest,
            exclusion_signals=(),
            text_only=True,
            ),
        }
    )
    view = build_source_view((body,), (source.unit.source_id,))
    title_id = view.segments[0].span_id
    row_segment = view.segments[-1]
    assert row_segment.text == row
    assert row_segment.entities == (
        ("Marianthi Leontaridi", "PERSON"),
        ("Boston Consulting Group", "ORGANISATION"),
    )
    wire = {
        "package": {
            "substantive_claim_indexes": [0, 1],
            "governed_claims": [
                _reference_claim(
                    title_id,
                    "HEADLINE",
                    ("部門宣布申請限期改為2026年6月30日。",),
                    localised=(("30 June 2026", "2026年6月30日"),),
                ),
                _reference_claim(
                    row_segment.span_id,
                    "SUBSTANTIVE",
                    ("第2行紀錄：", "，日期2026-05-21，", "；部門宣布申請限期改為2026年6月30日。"),
                ),
            ],
            "qualification_evidence": [
                {
                    "test": "OFFICIAL_ACTION_OR_DEADLINE",
                    "claim_index": 0,
                    "test_evidence": {
                        "action_class": "OFFICIAL_DEADLINE",
                        "event_polarity": "AFFIRMED",
                        "action_relation": "NEW_OR_CHANGED_OFFICIAL_ACTION",
                        "material_relation_span": title,
                        "reader_action": title,
                    },
                    "policy_version": EVID_012_POLICY_VERSION,
                }
            ],
            "selection_rationale": "Exact title and complete canonical row selected.",
            "geography": ["UK"],
            "categories": ["Politics and law"],
            "explicit_exclusions": [],
        }
    }
    return candidate, base, source, acquired, view, wire, row, body


def test_reference_result_materialises_then_passes_full_governed_path():
    candidate, base, source, acquired, view, wire, row, body = _literal_reference_inputs()
    wire = _v18_wire_from_v17(wire)
    package_value, receipt = materialise_v18(
        canonical_json_bytes(wire),
        view,
        "request-digest",
        provider_schema=PROVIDER_SCHEMA,
        v17_schema=_V17_PROVIDER_SCHEMA,
    )
    assert receipt["raw_digest"] == digest_bytes(canonical_json_bytes(wire))
    assert receipt["manifest_digest"] == view.manifest_digest
    execution = NativeAssessmentExecution(
        canonical_json_bytes(package_value).decode(), {}
    )
    assessment = AutonomousNativeEvidenceAssessor._validated_execution(
        execution, candidate, base, (source,), (acquired,)
    )
    assert assessment.governed_claims[1].claim == row
    assert assessment.governed_claims[1].named_entities == (
        "Boston Consulting Group",
        "Marianthi Leontaridi",
    )
    assert (
        assessment.governed_claims[1].rendered_assertion_zh_hant_hk
        == "第2行紀錄：Marianthi Leontaridi，日期2026-05-21，"
        "Boston Consulting Group；部門宣布申請限期改為2026年6月30日。"
    )
    governed = replace(
        base,
        substantive_new_information=assessment.substantive_new_information,
        governed_claims=assessment.governed_claims,
        qualification_evidence=assessment.qualification_evidence,
        selection_rationale=assessment.selection_rationale,
        geography=assessment.geography,
        categories=assessment.categories,
        explicit_exclusions=assessment.explicit_exclusions,
    )
    records = NativeEvidenceController._records(
        base, governed, (source,), (acquired,), assessment
    )
    retained = tuple(
        (
            record["record_id"],
            record["record_type"],
            canonical_json_bytes(record).decode(),
            digest_bytes(canonical_json_bytes(record)),
        )
        for record in records
    )
    assert validate_governed_evidence_records(
        candidate_id=candidate.candidate_id,
        source_inventory=((source.unit.source_id, acquired.canonical_url),),
        base_package_digest=base.digest,
        package=governed,
        retained_records=retained,
    )
    assert _qualification_relation_is_proven(
        governed.qualification_evidence[0],
        governed.governed_claims[0],
        source_context=body,
    )


_USAGE = {"usage_basis": "PROVIDER_REPORTED", "input_tokens": 1,
          "output_tokens": 1, "cached_read_tokens": 0, "cached_write_tokens": 0,
          "reasoning_tokens": 0, "context_tokens": 1, "total_tokens": 2}


def _run_reference_assessment(tmp_path, monkeypatch, *, malformed=False, raw_override=None):
    candidate, base, source, acquired, view, wire, _row, _body = _literal_reference_inputs()
    wire = _v18_wire_from_v17(wire)
    service, usage = _usage(tmp_path, monkeypatch)
    # Fixture creates the ledger after the usage schema; exercise normal startup
    # with a pre-existing ledger, as in the private runtime.
    from newsroom.control_plane.model_usage import ModelUsageService
    ModelUsageService(service.path)
    calls = []
    if malformed:
        wire["package"]["governed_claims"][0]["source_range"]["first_span_id"] = "S99L1"
    raw = raw_override if raw_override is not None else canonical_json_bytes(wire).decode()

    def dispatch(prompt):
        calls.append(json.loads(prompt))
        return NativeAssessmentExecution(raw, dict(_USAGE))

    assessor = AutonomousNativeEvidenceAssessor(dispatch, usage=usage, dispatch_fence=nullcontext)
    if malformed or raw_override is not None:
        with pytest.raises(NativeEvidenceHold, match="ASSESSOR_SOURCE_REFERENCE_HOLD"):
            assessor(candidate, base, (source,), (acquired,))
        result = None
    else:
        result = assessor(candidate, base, (source,), (acquired,))
    return service, usage, assessor, candidate, base, source, acquired, calls, raw, result


def test_v18_raw_and_materialised_results_are_retained_once_and_cached(tmp_path, monkeypatch):
    service, usage, assessor, candidate, base, source, acquired, calls, raw, result = (
        _run_reference_assessment(tmp_path, monkeypatch)
    )
    request = calls[0]
    assert request["contract"] == VERSION
    assert "passages" not in request["base_package"]
    assert "body" not in request["sources"][0]
    assert "".join(item["text"] for item in request["sources"][0]["segments"]) == acquired.body.decode()
    assert assessor(candidate, base, (source,), (acquired,)) == result
    assert assessor.assess_with_boundary(candidate, base, (source,), (acquired,),
                                        before_dispatch=None, cached_only=True) == result
    assert len(calls) == 1
    retained_result, = usage.retained_assessments(candidate, base)
    assert retained_result.execution.text != raw
    assert json.loads(retained_result.execution.text)["package"]["governed_claims"][1]["claim"] == result.governed_claims[1].claim
    with sqlite3.connect(service.path) as connection:
        raw_record = json.loads(connection.execute(
            "SELECT payload_json FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'"
        ).fetchone()[0])
        record = json.loads(connection.execute(
            "SELECT payload_json FROM ledger WHERE kind='NATIVE_ASSESSMENT_MATERIALISATION'"
        ).fetchone()[0])
        assert raw_record["result_text"] == raw
        assert record["receipt"]["materialised_text"] == retained_result.execution.text
        assert record["receipt"]["raw_digest"] == raw_record["result_digest"]
        assert record["evidence_package_digest"] == base.digest
        assert connection.execute("SELECT count(*) FROM model_invocation_allocations").fetchone() == (1,)
        assert connection.execute("SELECT usage_status,outcome FROM model_invocation_terminals").fetchone() == ("REPORTED", "ASSESSOR_ACCEPTED")
        plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT payload_json FROM ledger WHERE kind=? "
            "AND json_extract(payload_json,'$.invocation_id')=?",
            ("NATIVE_ASSESSMENT_MATERIALISATION", retained_result.proof.invocation_id),
        ).fetchall()
        assert any("model_usage_assessor_materialisation" in item[3] for item in plan)


@pytest.mark.parametrize("mutation", ["raw", "allocation", "context", "body", "text", "missing", "duplicate"])
def test_v18_retained_materialisation_tamper_never_dispatches(tmp_path, monkeypatch, mutation):
    service, usage, assessor, candidate, base, source, acquired, calls, _raw, _result = (
        _run_reference_assessment(tmp_path, monkeypatch)
    )
    with sqlite3.connect(service.path) as connection:
        kind = "NATIVE_ASSESSMENT_RESULT" if mutation == "raw" else "NATIVE_ASSESSMENT_MATERIALISATION"
        seq, payload = connection.execute("SELECT seq,payload_json FROM ledger WHERE kind=?", (kind,)).fetchone()
        record = json.loads(payload)
        if mutation == "raw":
            record["result_text"] += " "
            record["result_bytes"] += 1
            record["result_digest"] = digest_bytes(record["result_text"].encode())
        elif mutation == "allocation":
            record["allocation_digest"] = "sha256:" + "a" * 64
        elif mutation == "context":
            record["context_manifest_digest"] = "sha256:" + "a" * 64
        elif mutation in {"body", "text"}:
            receipt = record["receipt"]
            if mutation == "body":
                receipt["body_digests"] = ["sha256:" + "a" * 64]
            else:
                receipt["materialised_text"] += " "
            receipt["receipt_digest"] = digest_canonical({k: v for k, v in receipt.items() if k != "receipt_digest"})
        if mutation == "missing":
            connection.execute("DELETE FROM ledger WHERE seq=?", (seq,))
        elif mutation == "duplicate":
            from newsroom.control_plane.store import append_ledger
            append_ledger(connection, kind, record)
        else:
            updated = canonical_json_bytes(record)
            connection.execute("UPDATE ledger SET payload_json=?,payload_digest=? WHERE seq=?",
                               (updated.decode(), digest_bytes(updated), seq))
    assert usage.retained_assessments(candidate, base) is None
    with pytest.raises(NativeEvidenceHold, match="ASSESSOR_REVALIDATION_UNRESOLVED_HOLD"):
        assessor(candidate, base, (source,), (acquired,))
    assert len(calls) == 1


def test_v18_bad_reference_is_accounted_and_not_retried(tmp_path, monkeypatch):
    service, usage, assessor, candidate, base, source, acquired, calls, raw, _result = (
        _run_reference_assessment(tmp_path, monkeypatch, malformed=True)
    )
    assert usage.retained_output_contract_failure(candidate) is not None
    with pytest.raises(NativeEvidenceHold):
        assessor(candidate, base, (source,), (acquired,))
    with sqlite3.connect(service.path) as connection:
        assert connection.execute("SELECT usage_status,outcome FROM model_invocation_terminals").fetchone() == ("REPORTED", "ASSESSOR_VALIDATION_FAILED")
        assert connection.execute("SELECT count(*) FROM ledger WHERE kind='NATIVE_ASSESSMENT_MATERIALISATION'").fetchone() == (0,)
        record = json.loads(connection.execute("SELECT payload_json FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'").fetchone()[0])
        assert record["result_text"] == raw
        assert connection.execute("SELECT count(*) FROM model_invocation_allocations").fetchone() == (1,)
    assert len(calls) == 1


def test_reference_expansion_cap_applies_even_when_wire_is_small():
    body = ("ordinary source words " * 2200 + "\n") * 4
    view = build_source_view((body,), ("UK-03",))
    claims = [_reference_claim(f"S1L{i}", "SUBSTANTIVE", ("原文。",)) for i in (1, 2)]
    for claim in claims:
        claim["support_range"] = {"first_span_id": "S1L1", "last_span_id": "S1L4"}
    wire = {"package": {"governed_claims": claims, "substantive_claim_indexes": [0, 1],
                        "qualification_evidence": [], "selection_rationale": "Bound test.",
                        "geography": ["UK"], "categories": ["Politics and law"], "explicit_exclusions": []}}
    assert len(canonical_json_bytes(wire)) < 4096
    with pytest.raises(SourceReferenceError, match="materialised package exceeds bound"):
        materialise_v17(wire, view, "request", provider_schema=_V17_PROVIDER_SCHEMA)


def test_reference_receipt_replay_is_idempotent_and_conflict_rolls_back(tmp_path, monkeypatch):
    from datetime import datetime
    from newsroom.control_plane.model_usage import _allocation_from_record
    from newsroom.control_plane.native_evidence import NativeEvidenceError

    service, usage, _assessor, _candidate, _base, _source, _acquired, _calls, raw, _result = (
        _run_reference_assessment(tmp_path, monkeypatch)
    )
    with sqlite3.connect(service.path) as connection:
        allocation = _allocation_from_record(json.loads(connection.execute(
            "SELECT record_json FROM model_invocation_allocations"
        ).fetchone()[0]))
        materialisation = json.loads(connection.execute(
            "SELECT payload_json FROM ledger WHERE kind='NATIVE_ASSESSMENT_MATERIALISATION'"
        ).fetchone()[0])["receipt"]
        dispatch_at = datetime.fromisoformat(connection.execute(
            "SELECT observed_at FROM model_transport_observations"
        ).fetchone()[0])
        before = connection.execute("SELECT kind,payload_json,payload_digest FROM ledger ORDER BY seq").fetchall()
    execution = NativeAssessmentExecution(raw, dict(_USAGE))
    assert usage.retain_result(allocation, execution, dispatch_at=dispatch_at,
                               materialisation=materialisation)
    with pytest.raises(NativeEvidenceError, match="materialisation"):
        usage.retain_result(allocation, execution, dispatch_at=dispatch_at,
                            materialisation=materialisation | {"request_identity": "wrong-request"})
    with sqlite3.connect(service.path) as connection:
        assert connection.execute("SELECT kind,payload_json,payload_digest FROM ledger ORDER BY seq").fetchall() == before


def test_current_v18_reads_frozen_v16_full_package_without_relabelling_or_dispatch(tmp_path, monkeypatch):
    from newsroom.tests.test_native_assessor import _use_historical_v16

    candidate, base, source, acquired, view, wire, _row, _body = _literal_reference_inputs()
    package, _receipt = materialise_v17(
        wire, view, 'historical-request', provider_schema=_V17_PROVIDER_SCHEMA,
    )
    historical_execution = NativeAssessmentExecution(canonical_json_bytes(package).decode(), dict(_USAGE))
    with monkeypatch.context() as historical:
        _use_historical_v16(historical)
        service, old_usage = _usage(tmp_path, historical)
        allocation = old_usage.begin(candidate, base, 'historical request')
        dispatch_at = old_usage.mark_dispatch(allocation)
        old_usage.retain_result(allocation, historical_execution, dispatch_at=dispatch_at)
        old_usage.complete(allocation, outcome='ASSESSOR_ACCEPTED', execution=historical_execution,
                           provider_dispatched=True, dispatch_at=dispatch_at)
    _service, current_usage = _usage(tmp_path, monkeypatch)
    assert current_usage._policy.prompt_contract_version == VERSION
    assessor = AutonomousNativeEvidenceAssessor(
        lambda _request: pytest.fail('cached v16 result must not dispatch v18'),
        usage=current_usage, dispatch_fence=nullcontext,
    )
    result = assessor.assess_with_boundary(candidate, base, (source,), (acquired,),
                                           before_dispatch=None, cached_only=True)
    retained, = current_usage.retained_assessments(candidate, base)
    assert retained.contract_version == 'newsroom.native-evidence-assessor.v16'
    assert retained.execution.text == historical_execution.text
    assert result.governed_claims[1].claim == package['package']['governed_claims'][1]['claim']
    with sqlite3.connect(service.path) as connection:
        assert connection.execute("SELECT count(*) FROM model_invocation_allocations").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM ledger WHERE kind='NATIVE_ASSESSMENT_MATERIALISATION'").fetchone() == (0,)


def test_current_v18_reads_frozen_v17_materialisation_without_dispatch(
    tmp_path, monkeypatch,
):
    from newsroom.control_plane import native_assessor as module

    candidate, base, source, acquired, view, wire, _row, _body = (
        _literal_reference_inputs()
    )
    raw = canonical_json_bytes(wire).decode()
    execution = NativeAssessmentExecution(raw, dict(_USAGE))
    with monkeypatch.context() as historical:
        historical.setattr(module, "VERSION", module._V17_PRODUCER_VERSION)
        historical.setattr(module, "SYSTEM", module._V17_SYSTEM)
        historical.setattr(module, "PROVIDER_SCHEMA", module._V17_PROVIDER_SCHEMA)
        historical.setattr(
            module,
            "PROVIDER_SCHEMA_DIGEST",
            module._V17_PROVIDER_SCHEMA_DIGEST,
        )
        service, old_usage = _usage(tmp_path, historical)
        allocation = old_usage.begin(
            candidate, base, "historical v17 request", source_view=view,
        )
        package, receipt = materialise_v17(
            wire, view, allocation.request_digest,
            provider_schema=_V17_PROVIDER_SCHEMA,
        )
        dispatch_at = old_usage.mark_dispatch(allocation)
        old_usage.retain_result(
            allocation, execution, dispatch_at=dispatch_at,
            materialisation=receipt,
        )
        old_usage.complete(
            allocation, outcome="ASSESSOR_ACCEPTED", execution=execution,
            provider_dispatched=True, dispatch_at=dispatch_at,
        )
    _service, current_usage = _usage(tmp_path, monkeypatch)
    assessor = AutonomousNativeEvidenceAssessor(
        lambda _request: pytest.fail("cached v17 result dispatched v18"),
        usage=current_usage, dispatch_fence=nullcontext,
    )
    result = assessor.assess_with_boundary(
        candidate, base, (source,), (acquired,),
        before_dispatch=None, cached_only=True,
    )
    retained, = current_usage.retained_assessments(candidate, base)
    assert retained.contract_version == module._V17_PRODUCER_VERSION
    assert retained.execution.text == canonical_json_bytes(package).decode()
    assert result.governed_claims[1].claim == (
        package["package"]["governed_claims"][1]["claim"]
    )
    with sqlite3.connect(service.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM model_invocation_allocations"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM ledger "
            "WHERE kind='NATIVE_ASSESSMENT_MATERIALISATION'"
        ).fetchone() == (1,)


def test_rehashed_materialisation_cannot_drop_a_provider_selected_claim(tmp_path, monkeypatch):
    service, usage, assessor, candidate, base, source, acquired, calls, raw, result = (
        _run_reference_assessment(tmp_path, monkeypatch)
    )
    assert len(result.governed_claims) == 2
    with sqlite3.connect(service.path) as connection:
        seq, text = connection.execute("SELECT seq,payload_json FROM ledger WHERE kind='NATIVE_ASSESSMENT_MATERIALISATION'").fetchone()
        record = json.loads(text)
        receipt = record['receipt']
        package = json.loads(receipt['materialised_text'])
        package['package']['governed_claims'] = package['package']['governed_claims'][:1]
        package['package']['substantive_new_information'] = package['package']['substantive_new_information'][:1]
        receipt['claim_entity_order'] = receipt['claim_entity_order'][:1]
        changed = canonical_json_bytes(package)
        receipt['materialised_text'] = changed.decode()
        receipt['package_digest'] = digest_bytes(changed)
        receipt['receipt_digest'] = digest_canonical({key: value for key, value in receipt.items() if key != 'receipt_digest'})
        updated = canonical_json_bytes(record)
        connection.execute('UPDATE ledger SET payload_json=?,payload_digest=? WHERE seq=?',
                           (updated.decode(), digest_bytes(updated), seq))
        assert json.loads(connection.execute("SELECT payload_json FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'").fetchone()[0])['result_text'] == raw
    assert usage.retained_assessments(candidate, base) is None
    with pytest.raises(NativeEvidenceHold, match='ASSESSOR_REVALIDATION_UNRESOLVED_HOLD'):
        assessor.assess_with_boundary(candidate, base, (source,), (acquired,),
                                      before_dispatch=None, cached_only=True)
    assert len(calls) == 1


@pytest.mark.parametrize('malformed', ['nested', 'surrogate'])
def test_bounded_decoder_failure_retains_raw_and_usage_without_retry(tmp_path, monkeypatch, malformed):
    if malformed == 'nested':
        raw = '{"package":' + '[' * 10_000 + '0' + ']' * 10_000 + '}'
    else:
        _candidate, _base, _source, _acquired, _view, wire, _row, _body = _literal_reference_inputs()
        wire['package']['selection_rationale'] = '\ud800'
        raw = json.dumps(wire, ensure_ascii=True)
    service, usage, assessor, candidate, base, source, acquired, calls, _raw, _result = (
        _run_reference_assessment(tmp_path, monkeypatch, raw_override=raw)
    )
    assert len(raw.encode()) < 256 * 1024
    assert usage.retained_output_contract_failure(candidate) is not None
    with pytest.raises(NativeEvidenceHold):
        assessor(candidate, base, (source,), (acquired,))
    with sqlite3.connect(service.path) as connection:
        stored, = connection.execute("SELECT payload_json FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'").fetchone()
        assert json.loads(stored)['result_text'] == raw
        assert connection.execute('SELECT usage_status,outcome FROM model_invocation_terminals').fetchone() == ('REPORTED', 'ASSESSOR_VALIDATION_FAILED')
        assert connection.execute("SELECT count(*) FROM ledger WHERE kind='NATIVE_ASSESSMENT_MATERIALISATION'").fetchone() == (0,)
        assert connection.execute('SELECT count(*) FROM model_invocation_allocations').fetchone() == (1,)
    assert len(calls) == 1
