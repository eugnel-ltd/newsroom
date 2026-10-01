"""Removing the independent assessor output guard preserves immutable failures."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from newsroom.tests.assessor_fixture_support import candidate_fixture

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.control_plane import native_assessor as native
from newsroom.control_plane.model_usage import (
    InvocationAllocation,
    InvocationEfficiencyPolicy,
    InvocationTerminal,
    ModelUsageAdmissionError,
    ModelUsageIntegrityError,
    ModelUsageService,
    UsageComponents,
    UsageStatus,
    WorkEnvelope,
    WorkloadClass,
    _allocation_from_record,
    _policy_from_record,
)
from newsroom.control_plane.native_assessor import (
    AutonomousNativeEvidenceAssessor, NativeAssessmentExecution, NativeAssessmentUsage,
)
from newsroom.control_plane.native_evidence import NativeEvidenceHold
from newsroom.control_plane.store import append_ledger, connect
from newsroom.control_plane.writer import CONT_DISABLED_CAPABILITIES, _grok_command_flags
from newsroom.tests.test_native_assessor import _base_package, _candidate, _ready_package

AT = datetime(2026, 9, 29, tzinfo=UTC)
KIND = "NATIVE_ASSESSOR_OUTPUT_GUARD_REQUALIFICATION"


def _changed(policy, **changes):
    values = asdict(policy)
    values.pop("canonical_digest")
    return InvocationEfficiencyPolicy.create(**(values | changes))


def _policy(**changes):
    return InvocationEfficiencyPolicy.create(**({
        "policy_id": "native-assessor-output-guard-policy", "version": "v20",
        "workload_class": WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
        "provider": "grok-build-cli", "route": native.ROUTE,
        "model": "grok-4.7", "reasoning": "high", "one_turn": True,
        "exact_input": True, "skills_enabled": False, "tools_enabled": False,
        "mcp_enabled": False, "prior_message_count": 0,
        "command_semantic_version": "1.0.10",
        "command_flags": _grok_command_flags("high", model="grok-4.7"),
        "context_manifest_schema_version": "newsroom.native-evidence-assessor.context-manifest.v3",
        "disabled_capabilities": CONT_DISABLED_CAPABILITIES,
        "implementation_revision": "1" * 40, "max_prompt_bytes": 56_464,
        "max_context_tokens": 100_000, "max_output_tokens": None,
        "max_total_tokens": 100_000,
        "prompt_contract_version": "newsroom.native-evidence-assessor.v20",
        "output_schema_digest": native._V19_PROVIDER_SCHEMA_DIGEST,
        "allowed_context_identities": (native.CONTEXT_IDENTITY,),
        "allowed_config_identities": (native.CONFIG_IDENTITY,),
        "hard_estimate_ceiling_tokens": 100_000,
        "evidence_digest": digest_bytes(b"qualified-v20"), "qualified": True,
    } | changes))


def _usage(tmp_path, monkeypatch, policy):
    monkeypatch.setattr(native, "read_grok_command_semantic_version", lambda: "1.0.10")
    monkeypatch.setattr(native, "cont_writer_implementation_identity", lambda: ("1" * 40, True))
    path = str(tmp_path / "output-guard.sqlite3")
    connect(path).close()
    service = ModelUsageService(path)
    return service, NativeAssessmentUsage(service, policy, clock=lambda: AT)


def _execution(**changes):
    return NativeAssessmentExecution('{"package":{}}', {
        "usage_basis": "PROVIDER_REPORTED", "input_tokens": 20_000,
        "output_tokens": 12_000, "cached_read_tokens": 50,
        "cached_write_tokens": 0, "reasoning_tokens": 1_000,
        "context_tokens": 20_050, "total_tokens": 32_050,
    } | changes)


def _fixture(tmp_path, monkeypatch, *, declared_cli="1.0.10", observed_cli="1.0.10", **usage_changes):
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    connection.close()
    old = _policy(
        version="v19", model="grok-4.6", reasoning="medium", command_semantic_version=declared_cli,
        command_flags=_grok_command_flags("medium", model="grok-4.6"),
        prompt_contract_version=native._V19_PRODUCER_VERSION,
        max_output_tokens=10_000, evidence_digest=digest_bytes(b"qualified-v19"),
    )
    with monkeypatch.context() as historical:
        historical.setattr(native, "VERSION", native._V19_PRODUCER_VERSION)
        historical.setattr(native, "MODEL", "grok-4.6")
        historical.setattr(native, "REASONING", "medium")
        historical.setattr(native, "COMMAND_FLAGS", old.command_flags)
        service, usage = _usage(tmp_path, historical, old)
        historical.setattr(native, "read_grok_command_semantic_version", lambda: observed_cli)
        allocation = usage.begin(candidate, base, "original exact request")
        dispatch_at = usage.mark_dispatch(allocation)
        execution = _execution(**usage_changes)
        usage.retain_result(allocation, execution, dispatch_at=dispatch_at)
        usage.complete(
            allocation, outcome="ASSESSOR_VALIDATION_FAILED", execution=execution,
            provider_dispatched=True, dispatch_at=dispatch_at,
            failure_class="ASSESSMENT_VALIDATION_FAILED",
        )
    new = _policy(command_semantic_version=declared_cli)
    service.register_policy(new)
    _, current = _usage(tmp_path, monkeypatch, new)
    return service, current, candidate, base, allocation, new


def _recover(service, allocation, policy):
    return service.requalify_native_assessor_output_guard(
        invocation_id=allocation.invocation_id,
        qualified_policy_digest=policy.canonical_digest,
        recorded_at=AT + timedelta(seconds=30),
    )


@pytest.mark.parametrize("declared_cli,observed_cli", [("1.0.10", "1.0.10"), ("1.0.30", "1.0.42")], ids=["same-version", "observed-newer"])
def test_output_guard_requalification_preserves_failure_and_releases_only_future_work(tmp_path, monkeypatch, declared_cli, observed_cli):
    service, usage, candidate, base, failed, policy = _fixture(tmp_path, monkeypatch, declared_cli=declared_cli, observed_cli=observed_cli)
    tables = ("model_invocation_allocations", "model_invocation_terminals", "model_invocation_policies",
              "model_work_envelopes", "model_invocation_context_manifests", "model_provider_telemetry")
    with sqlite3.connect(service.path) as c:
        original = {table: c.execute(f"SELECT * FROM {table}").fetchall() for table in tables}
        original_results = c.execute("SELECT * FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'").fetchall()
        assert service._route_state(c, policy.route)["state"] == "OPEN"
        assert json.loads(original["model_invocation_terminals"][0][-1])["policy_breach"] == "REQUESTED_MAX_OUTPUT_TOKENS_EXCEEDED"
    digest = _recover(service, failed, policy)
    assert _recover(service, failed, policy) == digest
    reopened = ModelUsageService(service.path)
    with sqlite3.connect(service.path) as c:
        assert {table: c.execute(f"SELECT * FROM {table}").fetchall() for table in tables} == original
        assert c.execute("SELECT * FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'").fetchall() == original_results
        assert reopened._route_state(c, policy.route)["state"] == "CLOSED"
        receipts = c.execute("SELECT payload_json FROM ledger WHERE kind=?", (KIND,)).fetchall()
        assert len(receipts) == 1
        receipt = json.loads(receipts[0][0])
        assert receipt["requalification_digest"] == digest
        assert receipt["original_candidate_retry"] is False
        assert receipt["qualified_policy_digest"] == policy.canonical_digest
    # Usage eligibility is not permission to retry the breached candidate.
    assert usage.retained_assessments(candidate, base) is None
    assert usage.retained_pre_dispatch_failure(candidate) is None
    calls = []
    assessor = AutonomousNativeEvidenceAssessor(
        lambda _prompt: calls.append("provider"), usage=usage, dispatch_fence=nullcontext,
    )
    with pytest.raises(NativeEvidenceHold, match="ASSESSOR_REVALIDATION_UNRESOLVED_HOLD"):
        assessor(candidate, base, (), ())
    assert calls == []
    fresh = SimpleNamespace(candidate_id="fresh", version_id="fresh-version", governing_manifest=candidate.governing_manifest)
    assert usage.begin(fresh, base, "next exact request").invocation_id != failed.invocation_id


@pytest.mark.parametrize("limit", [None, 10_000])
def test_policy_and_allocation_output_limit_roundtrip_preserves_hashes(tmp_path, monkeypatch, limit):
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    connection.close()
    policy = _policy()
    service, usage = _usage(tmp_path, monkeypatch, policy)
    allocation = usage.begin(candidate, base, "request")
    if limit is not None:
        policy = _changed(policy, max_output_tokens=limit)
        values = asdict(allocation)
        values.pop("canonical_digest"); values.pop("invocation_id")
        allocation = InvocationAllocation.create(**(values | {"invocation_policy_digest": policy.canonical_digest, "max_output_tokens": limit}))
    assert _policy_from_record(policy.as_record()) == policy
    assert _allocation_from_record(allocation.as_record()) == allocation
    assert canonical_json_bytes(_policy_from_record(policy.as_record()).as_record()) == canonical_json_bytes(policy.as_record())


@pytest.mark.parametrize("bad", [True, False, 0, -1, "10000", 1.5])
def test_output_guard_rejects_malformed_limit_and_retained_bool(bad):
    with pytest.raises(ValueError):
        _policy(max_output_tokens=bad)
    record = _policy().as_record() | {"max_output_tokens": bad}
    with pytest.raises(ModelUsageIntegrityError):
        _policy_from_record(record)


@pytest.mark.parametrize("changes", [
    {"provider": "other-provider"}, {"route": "CONT_WRITER"},
    {"workload_class": WorkloadClass.CONT_WRITER_PRIMARY},
])
def test_absent_output_limit_is_native_assessor_only(changes):
    with pytest.raises(ModelUsageIntegrityError):
        _policy(**changes)


@pytest.mark.parametrize(("changes", "breach"), [
    ({}, None),
    ({"context_tokens": 100_001}, "MAX_CONTEXT_TOKENS_EXCEEDED"),
    ({"input_tokens": 90_000, "context_tokens": 90_050, "total_tokens": 102_050}, "MAX_TOTAL_TOKENS_EXCEEDED"),
])
def test_new_output_above_ten_thousand_keeps_other_token_guards(tmp_path, monkeypatch, changes, breach):
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    connection.close()
    service, usage = _usage(tmp_path, monkeypatch, _policy())
    allocation = usage.begin(candidate, base, "request")
    assert allocation.max_output_tokens is None
    dispatch_at = usage.mark_dispatch(allocation)
    usage.complete(allocation, outcome="ASSESSOR_SUCCEEDED", execution=_execution(**changes), provider_dispatched=True, dispatch_at=dispatch_at)
    with sqlite3.connect(service.path) as c:
        terminal = json.loads(c.execute("SELECT record_json FROM model_invocation_terminals").fetchone()[0])
        assert terminal["components"]["output_tokens"] == 12_000
        assert terminal["policy_breach"] == breach


@pytest.mark.parametrize("bad", [True, False, 0, -1, "10000", 1.5])
def test_allocation_reader_rejects_invalid_output_guard(tmp_path, monkeypatch, bad):
    service, _usage_, _candidate_, _base_, allocation, _policy_ = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ModelUsageIntegrityError):
        _allocation_from_record(allocation.as_record() | {"max_output_tokens": bad})


@pytest.mark.parametrize(("requested", "policy_limit"), [(10_000, None), (None, 10_000)])
def test_output_guard_presence_must_match_at_admission_and_terminal(tmp_path, monkeypatch, requested, policy_limit):
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    connection.close()
    service, usage = _usage(tmp_path, monkeypatch, _policy())
    original = usage.begin(candidate, base, "request")
    policy = _changed(usage._policy, max_output_tokens=policy_limit)
    values = asdict(original)
    values.pop("canonical_digest"); values.pop("invocation_id")
    allocation = InvocationAllocation.create(**(values | {
        "invocation_policy_digest": policy.canonical_digest, "max_output_tokens": requested,
    }))
    with sqlite3.connect(service.path) as c, pytest.raises(ModelUsageAdmissionError):
        service._validate_preflight(c, allocation, policy)
    terminal = InvocationTerminal.create(
        invocation_id=allocation.invocation_id, outcome="ASSESSOR_SUCCEEDED", failure_class=None,
        usage_status=UsageStatus.REPORTED, components=UsageComponents(
            input_tokens=1, output_tokens=1, total_tokens=2, provenance="PROVIDER_REPORTED",
        ), dispatch_at=AT, completed_at=AT, observed_at=AT,
        subscription_cli_chat_not_cash_debited=True,
    )
    with pytest.raises(ModelUsageIntegrityError, match="output guard differs"):
        service._validate_terminal(terminal, policy.workload_class, policy, requested_max_output_tokens=requested)


@pytest.mark.parametrize("change", [
    {"max_total_tokens": 110_000, "hard_estimate_ceiling_tokens": 110_000},
    {"max_context_tokens": 110_000}, {"max_prompt_bytes": 56_465},
    {"max_output_tokens": 10_000}, {"prompt_contract_version": native._V19_PRODUCER_VERSION},
    {"reasoning": "medium", "command_flags": _grok_command_flags("medium", model="grok-4.7")},
    {"model": "grok-4.6", "command_flags": _grok_command_flags("high", model="grok-4.6")},
    {"qualified": False}, {"output_schema_digest": digest_bytes(b"other schema")},
])
def test_requalification_rejects_unqualified_or_wider_policy(tmp_path, monkeypatch, change):
    service, _usage_, _candidate_, _base_, allocation, policy = _fixture(tmp_path, monkeypatch)
    invalid = _changed(policy, **change)
    service.register_policy(invalid)
    with pytest.raises((ModelUsageAdmissionError, ModelUsageIntegrityError)):
        _recover(service, allocation, invalid)
    with sqlite3.connect(service.path) as c:
        assert c.execute("SELECT count(*) FROM ledger WHERE kind=?", (KIND,)).fetchone()[0] == 0


@pytest.mark.parametrize("usage_change", [
    {"context_tokens": 100_001}, {"context_tokens": None},
    {"input_tokens": 90_000, "context_tokens": 90_050, "total_tokens": 102_050},
    {"total_tokens": 1}, {"reasoning_tokens": 12_001},
    {"usage_basis": "UNREPORTED", "total_tokens": None},
])
def test_requalification_rejects_non_output_only_or_unknown_usage(tmp_path, monkeypatch, usage_change):
    service, _usage_, _candidate_, _base_, allocation, policy = _fixture(tmp_path, monkeypatch, **usage_change)
    with pytest.raises((ModelUsageAdmissionError, ModelUsageIntegrityError)):
        _recover(service, allocation, policy)


@pytest.mark.parametrize("unreported", [False, True])
def test_another_active_or_unknown_assessor_rolls_back_requalification(tmp_path, monkeypatch, unreported):
    service, _usage_, candidate, base, failed, policy = _fixture(tmp_path, monkeypatch)
    envelope = WorkEnvelope.create(
        cycle_id=digest_bytes(b"other cycle"), workload_class=failed.workload_class,
        admitted_at=AT, admission_decision_id=None, candidate_id="other",
        hypothesis_digest=candidate.governing_manifest.canonical_digest,
        evidence_package_digest=base.digest, ingest_id=None, graphiti_attempt_id=None,
    )
    service.open_envelope(envelope)
    values = asdict(failed)
    values.pop("canonical_digest"); values.pop("invocation_id")
    other = InvocationAllocation.create(**(values | {"envelope_id": envelope.envelope_id, "cycle_id": envelope.cycle_id}))
    with service._connection() as c:
        service._insert_allocation(c, other)
    if unreported:
        service.complete(InvocationTerminal.create(
            invocation_id=other.invocation_id, outcome="ASSESSOR_PROVIDER_FAILED", failure_class="TIMEOUT",
            usage_status=UsageStatus.UNREPORTED, components=UsageComponents(),
            dispatch_at=AT, completed_at=AT, observed_at=AT,
            subscription_cli_chat_not_cash_debited=True,
        ))
        with service._connection() as c:
            service._append_route_state(c, route=policy.route, state="OPEN",
                reason="REQUESTED_MAX_OUTPUT_TOKENS_EXCEEDED", invocation_id=failed.invocation_id,
                recorded_at=AT + timedelta(seconds=1))
    with pytest.raises(ModelUsageAdmissionError):
        _recover(service, failed, policy)
    with sqlite3.connect(service.path) as c:
        assert service._route_state(c, policy.route)["state"] == "OPEN"
        assert c.execute("SELECT count(*) FROM ledger WHERE kind=?", (KIND,)).fetchone()[0] == 0


@pytest.mark.parametrize("table", [
    "model_invocation_allocations", "model_invocation_terminals", "model_provider_telemetry",
    "model_work_envelopes", "model_invocation_context_manifests", "model_invocation_policies",
])
def test_requalification_authenticates_immutable_records(tmp_path, monkeypatch, table):
    service, _usage_, _candidate_, _base_, allocation, policy = _fixture(tmp_path, monkeypatch)
    with sqlite3.connect(service.path) as c:
        c.execute(f"UPDATE {table} SET record_json=json_set(record_json,'$.tampered',1)")
    with pytest.raises(ModelUsageIntegrityError):
        _recover(service, allocation, policy)


@pytest.mark.parametrize("kind", ["NATIVE_ASSESSMENT_RESULT", "NATIVE_ASSESSMENT_MATERIALISATION"])
def test_requalification_requires_one_raw_failed_result_and_no_materialisation(tmp_path, monkeypatch, kind):
    service, _usage_, _candidate_, _base_, allocation, policy = _fixture(tmp_path, monkeypatch)
    with sqlite3.connect(service.path) as c:
        append_ledger(c, kind, {"invocation_id": allocation.invocation_id})
    with pytest.raises(ModelUsageIntegrityError):
        _recover(service, allocation, policy)


@pytest.mark.parametrize("field", ["candidate_id", "qualified_policy_digest", "terminal_digest", "original_candidate_retry"])
def test_tampered_requalification_receipt_is_not_an_exemption(tmp_path, monkeypatch, field):
    service, _usage_, _candidate_, _base_, allocation, policy = _fixture(tmp_path, monkeypatch)
    _recover(service, allocation, policy)
    with sqlite3.connect(service.path) as c:
        seq, raw = c.execute("SELECT seq,payload_json FROM ledger WHERE kind=?", (KIND,)).fetchone()
        receipt = json.loads(raw)
        receipt[field] = True if field == "original_candidate_retry" else digest_bytes(b"tampered")
        receipt.pop("requalification_digest")
        receipt["requalification_digest"] = digest_canonical(receipt)
        raw = canonical_json_bytes(receipt).decode()
        c.execute("UPDATE ledger SET payload_json=?,payload_digest=? WHERE seq=?", (raw, digest_bytes(raw.encode()), seq))
    reopened = ModelUsageService(service.path)
    with sqlite3.connect(service.path) as c, pytest.raises(ModelUsageIntegrityError):
        reopened._route_state(c, policy.route)


def test_requalification_replay_rejects_another_qualified_policy(tmp_path, monkeypatch):
    service, _usage_, _candidate_, _base_, allocation, policy = _fixture(tmp_path, monkeypatch)
    _recover(service, allocation, policy)
    other = _changed(policy, version="another-qualified-v20", evidence_digest=digest_bytes(b"other qualification"))
    service.register_policy(other)
    with pytest.raises(ModelUsageIntegrityError, match="replay policy differs"):
        _recover(service, allocation, other)
    with sqlite3.connect(service.path) as c:
        assert c.execute("SELECT count(*) FROM ledger WHERE kind=?", (KIND,)).fetchone()[0] == 1


def test_duplicated_requalification_receipt_is_not_an_exemption(tmp_path, monkeypatch):
    service, _usage_, _candidate_, _base_, allocation, policy = _fixture(tmp_path, monkeypatch)
    _recover(service, allocation, policy)
    with sqlite3.connect(service.path) as c:
        raw = c.execute("SELECT payload_json FROM ledger WHERE kind=?", (KIND,)).fetchone()[0]
        append_ledger(c, KIND, json.loads(raw))
    with sqlite3.connect(service.path) as c, pytest.raises(ModelUsageIntegrityError):
        service._route_state(c, policy.route)


def test_output_requalification_never_changes_graphiti_circuit(tmp_path, monkeypatch):
    service, _usage_, _candidate_, _base_, allocation, policy = _fixture(tmp_path, monkeypatch)
    with service._connection() as c:
        service._append_route_state(c, route="GRAPHITI_CHAT_PRIMARY", state="OPEN", reason="UNKNOWN_USAGE",
            invocation_id=None, recorded_at=AT)
        before = [tuple(row) for row in c.execute("SELECT * FROM model_usage_route_circuit_events WHERE route='GRAPHITI_CHAT_PRIMARY'")]
    _recover(service, allocation, policy)
    with sqlite3.connect(service.path) as c:
        assert c.execute("SELECT * FROM model_usage_route_circuit_events WHERE route='GRAPHITI_CHAT_PRIMARY'").fetchall() == before
        assert service._route_state(c, "GRAPHITI_CHAT_PRIMARY")["state"] == "OPEN"


@pytest.mark.parametrize(("field", "value"), [
    ("VERSION", "newsroom.native-evidence-assessor.v99"),
    ("SYSTEM", "Future producer instructions"),
    ("PROVIDER_SCHEMA_DIGEST", digest_bytes(b"future schema")),
])
def test_historical_output_receipt_reader_uses_frozen_v19_wire_contract(tmp_path, monkeypatch, field, value):
    service, _usage_, _candidate_, _base_, allocation, policy = _fixture(tmp_path, monkeypatch)
    _recover(service, allocation, policy)
    monkeypatch.setattr(native, field, value)
    with sqlite3.connect(service.path) as c:
        assert service._route_state(c, policy.route)["state"] == "CLOSED"


def test_output_requalification_reads_use_scoped_ledger_index(tmp_path):
    path = str(tmp_path / "indexed.sqlite3")
    connect(path).close()
    ModelUsageService(path)
    with sqlite3.connect(path) as c:
        query = "SELECT payload_digest,payload_json FROM ledger WHERE kind=?"
        plan = " ".join(row[3] for row in c.execute("EXPLAIN QUERY PLAN " + query, (KIND,)))
        assert "SEARCH ledger USING INDEX model_usage_assessor_output_requalification" in plan


def test_missing_output_guard_field_is_not_nullable_policy():
    record = _policy().as_record()
    record.pop("max_output_tokens")
    with pytest.raises(ModelUsageIntegrityError):
        _policy_from_record(record)
