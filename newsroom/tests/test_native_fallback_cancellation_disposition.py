"""Cancelled native fallback accounting never releases a route or retries work."""

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.control_plane import model_usage as m
from newsroom.control_plane.cycle import _graphiti_usage_cycle_id
from newsroom.control_plane.graphiti import GraphitiModelUsageObserver
from newsroom.control_plane.graphiti_fallback_policy import load_checked_native_graphiti_fallback_circuit_policy
from newsroom.control_plane.graphiti_requests import load_checked_native_graphiti_call_shape_policy
from newsroom.control_plane.native_progress import NativeRevisionJournal
from newsroom.control_plane.native_qualification import NativeQualificationError, NativeQualificationPending, _invocations
from newsroom.control_plane.store import connect, insert_graphiti_attempt_receipt, reserve_graphiti_spend, reconcile_graphiti_spend
from newsroom.tests.test_native_graphiti import _native, _open

ROUTE = "GRAPHITI_CHAT_FALLBACK"
T0 = datetime(2026, 8, 24, 20, 0, tzinfo=UTC)
# Retained ExtractedEntities request schema: accounting fixtures must collect
# without importing the optional Graphiti runtime or its skipped test module.
EXTRACTED_ENTITIES_SCHEMA = json.dumps({
    "$defs": {"ExtractedEntity": {
        "properties": {
            "name": {"description": "Name of the extracted entity", "title": "Name", "type": "string"},
            "entity_type_id": {
                "description": "ID of the classified entity type. Must be one of the provided entity_type_id integers.",
                "title": "Entity Type Id", "type": "integer",
            },
            "episode_indices": {
                "description": "List of episode numbers (0-indexed) this entity was extracted from. When processing a single episode, this should be [0].",
                "items": {"type": "integer"}, "title": "Episode Indices", "type": "array",
            },
        },
        "required": ["name", "entity_type_id"], "title": "ExtractedEntity", "type": "object",
    }},
    "properties": {"extracted_entities": {
        "description": "List of extracted entities", "items": {"$ref": "#/$defs/ExtractedEntity"},
        "title": "Extracted Entities", "type": "array",
    }},
    "required": ["extracted_entities"], "title": "ExtractedEntities", "type": "object",
})


def _cancelled(tmp_path, monkeypatch, *, land=True, native=True, dispatched=True,
               outcome="CANCELLED", reported=False, connection=None, parented=False):
    monkeypatch.setattr("newsroom.control_plane.graphiti._graphiti_implementation_identity",
                        lambda: ("a" * 40, True))
    path = str(tmp_path / "private.sqlite3")
    connection = connection or connect(path)
    unit = _native("fallback-cancellation")
    if not native:
        unit = replace(unit, proving_run_id="native-source:" + digest_canonical({"unrelated": "research"}))
    journal = NativeRevisionJournal(connection)
    if land:
        journal.land((unit,))
    usage = m.ModelUsageService(path)
    envelope = m.WorkEnvelope.create(
        cycle_id=_graphiti_usage_cycle_id(unit, attempt_number=1, requested_cycle_id=None),
        workload_class=m.WorkloadClass.GRAPHITI_CHAT_PRIMARY, admitted_at=T0,
        admission_decision_id=None, candidate_id=None, hypothesis_digest=None,
        evidence_package_digest=None, ingest_id=unit.ingest_id,
        graphiti_attempt_id=f"{unit.ingest_id}:1",
    )
    usage.open_envelope(envelope)
    if not parented:
        usage.open_route_circuit(route="GRAPHITI_CHAT_PRIMARY", reason="QUOTA",
                                 invocation_id=None, recorded_at=T0 + timedelta(seconds=1))
    observer = GraphitiModelUsageObserver(
        service=usage, envelope=envelope, clock=lambda: T0 + timedelta(seconds=10),
        owner_stop_check=lambda: None, deadline=T0 + timedelta(minutes=3),
        effective_revision_digest=digest_canonical(asdict(unit.effective_revision)),
        ingest_obligation_id=unit.ingest_id,
        call_shape_policy=load_checked_native_graphiti_call_shape_policy(),
        fallback_policy=load_checked_native_graphiti_fallback_circuit_policy(),
    )
    request = dict(prompt="source-safe prompt", schema=EXTRACTED_ENTITIES_SCHEMA,
                   semantic_request_class="ExtractedEntities", max_tokens=77)
    primary_receipts = []
    if parented:
        primary = observer.before_cli_invocation(provider="cursor-agent-cli", model="composer-2.5", **request)
        observer.transport_dispatch_started(primary)
        primary_binding = observer.after_cli_invocation(primary, outcome="MALFORMED_OUTPUT", usage={
            "usage_basis": "PROVIDER_REPORTED", "input_tokens": 5, "output_tokens": 2,
            "total_tokens": 7, "cached_read_tokens": 0, "cached_write_tokens": 0, "reasoning_tokens": 0,
        })
        primary_receipts.append({**primary_binding, "outcome": "MALFORMED_OUTPUT", "provider": "cursor-agent-cli"})
    else:
        assert observer.use_direct_fallback(**request)
    allocation = observer.before_cli_invocation(provider="grok-build-cli", model="grok-4.6", **request)
    if dispatched:
        observer.transport_dispatch_started(allocation)
    binding = observer.after_cli_invocation(
        allocation, outcome=outcome,
        usage=({"usage_basis": "PROVIDER_REPORTED", "input_tokens": 1, "output_tokens": 0,
                "total_tokens": 1, "provider_telemetry": {"request_id": "fixture", "total_tokens": 1}}
               if reported else {"usage_basis": "UNREPORTED" if dispatched else "NO_PROVIDER_CALL"}),
    )
    terminal = usage.terminal(allocation.invocation_id)
    receipt = {
        "ingest_id": unit.ingest_id, "attempt_number": 1, "outcome": "TIMEOUT",
        "chat_invocations": [*primary_receipts, {**binding, "outcome": outcome, "provider": "grok-build-cli"}],
    }
    receipt_digest = insert_graphiti_attempt_receipt(
        connection, ingest_id=unit.ingest_id, attempt_number=1, outcome="TIMEOUT", receipt=receipt,
    )
    reserve_graphiti_spend(connection, spend_id="retained-reservation", ingest_id=unit.ingest_id,
                           attempt_number=1, proving_run_id=unit.proving_run_id,
                           generation_id="fixture", reserved_gbp_microunits=500_000,
                           ceiling_gbp_microunits=None)
    reconcile_graphiti_spend(connection, spend_id="retained-reservation",
                             embedding_usage={"usage_basis": "UNREPORTED"})
    connection.commit()
    usage.record_work_outcome(
        envelope_id=envelope.envelope_id, outcome="GRAPHITI_TIMEOUT",
        outcome_record_id=receipt_digest, payload_digest=None,
        terminal_at=T0 + timedelta(seconds=11),
    )
    return SimpleNamespace(connection=connection, usage=usage, unit=unit, journal=journal,
                           envelope=envelope, allocation=allocation, terminal=terminal,
                           policy=observer._policies[allocation.invocation_id])


def _dispose(case, *, usage=None):
    return (usage or case.usage).disposition_native_graphiti_fallback_cancellation(
        invocation_id=case.allocation.invocation_id,
        expected_terminal_digest=case.terminal.terminal_digest,
        expected_allocation_digest=case.allocation.canonical_digest,
        observed_at=T0 + timedelta(seconds=12),
    )


def _unchanged(case):
    return {table: case.connection.execute(f"SELECT * FROM {table}").fetchall() for table in (
        "model_invocation_allocations", "model_invocation_terminals", "model_work_outcomes",
        "unpublished_graphiti_attempt_receipts", "model_transport_observations",
        "model_provider_telemetry", "model_usage_reconciliations", "unpublished_graphiti_spend",
        "model_usage_route_circuit_events",
    )}


def test_cancelled_fallback_estimate_is_idempotent_and_preserves_unknown_usage_and_holds(tmp_path, monkeypatch):
    case = _cancelled(tmp_path, monkeypatch)
    try:
        with pytest.raises(NativeQualificationPending):
            _invocations(case.connection, case.journal)
        # Generic failed-call authority deliberately remains narrower.
        with pytest.raises(m.ModelUsageIntegrityError, match="ineligible"):
            case.usage.disposition_native_unreported_subscription_usage(
                invocation_id=case.allocation.invocation_id,
                expected_terminal_digest=case.terminal.terminal_digest,
                expected_allocation_digest=case.allocation.canonical_digest,
                observed_at=T0 + timedelta(seconds=12),
            )
        before = _unchanged(case)
        record = _dispose(case)
        assert record["usage_status"] == "ESTIMATED"
        assert record["components"] == m.UsageComponents(total_tokens=147_456, provenance="BOUNDED_ESTIMATE").as_record()
        assert record["exact_usage_remains_unknown"] is True
        assert record["provider_dispatch_preserved"] is True
        assert record["unknown_spend_released"] is False
        assert record["terminal_digest"] == case.terminal.terminal_digest
        assert record["allocation_digest"] == case.allocation.canonical_digest
        assert _dispose(case, usage=m.ModelUsageService(case.usage.path)) == record
        assert _unchanged(case) == before
        assert case.usage.route_state(ROUTE)["state"] == "OPEN"
        assert ROUTE not in m._usage_blocking_routes(case.connection)
        assert case.connection.execute("SELECT count(*) FROM model_usage_conservative_dispositions").fetchone() == (1,)
        assert _invocations(case.connection, case.journal) == (case.allocation.invocation_id,)
        current, = case.usage.query(start=T0, end=T0 + timedelta(minutes=1))["leaves"]
        assert current["usage_status"] == "ESTIMATED"
        assert current["terminal_usage_status"] == "UNREPORTED"
        assert current["total_tokens"] == 147_456
        assert current["provider_telemetry_digest"] is None
        proof = case.usage.native_graphiti_ingest_retry_evidence_many(
            failed_attempts={case.unit.ingest_id: 1}, max_attempts=6,
        )[case.unit.ingest_id]
        assert proof.settled_provider_attempts == (1,)
        assert proof.zero_dispatch_attempts == ()
        assert proof.unresolved_attempts == ()
    finally:
        case.connection.close()


def _rewrite_receipt(case, change):
    raw, = case.connection.execute("SELECT receipt_json FROM unpublished_graphiti_attempt_receipts").fetchone()
    receipt = json.loads(raw)
    receipt.pop("receipt_digest")
    change(receipt)
    receipt_digest = digest_bytes(canonical_json_bytes(receipt))
    receipt["receipt_digest"] = receipt_digest
    case.connection.execute("UPDATE unpublished_graphiti_attempt_receipts SET outcome=?,receipt_digest=?,receipt_json=?",
                            (receipt["outcome"], receipt_digest, canonical_json_bytes(receipt).decode()))
    raw, = case.connection.execute("SELECT record_json FROM model_work_outcomes").fetchone()
    work = json.loads(raw)
    work.pop("outcome_digest")
    work["outcome_record_id"] = receipt_digest
    work["outcome_digest"] = digest_canonical(work)
    case.connection.execute("UPDATE model_work_outcomes SET outcome_digest=?,record_json=?",
                            (work["outcome_digest"], canonical_json_bytes(work).decode()))


@pytest.mark.parametrize("defect", [
    "unlanded", "unrelated", "pre-dispatch", "ambiguous", "reported", "wrong-policy", "missing-request",
    "missing-authority", "wrong-terminal", "missing-work", "missing-receipt", "failed-receipt",
    "empty-leaves", "wrong-leaf-terminal", "wrong-leaf-outcome", "wrong-leaf-allocation", "wrong-leaf-envelope", "duplicate-leaf",
    "missing-dispatch", "duplicate-dispatch", "wrong-dispatch",
])
def test_cancelled_fallback_rejects_ineligible_or_changed_evidence(tmp_path, monkeypatch, defect):
    case = _cancelled(tmp_path, monkeypatch, land=defect != "unlanded", native=defect != "unrelated",
                      dispatched=defect != "pre-dispatch", reported=defect == "reported",
                      outcome="AMBIGUOUS_DISPATCH" if defect == "ambiguous" else "CANCELLED")
    connection = case.connection
    try:
        if defect == "wrong-policy":
            connection.execute("UPDATE model_invocation_policies SET qualified=0")
        elif defect == "missing-request":
            connection.execute("DELETE FROM graphiti_internal_requests")
        elif defect == "missing-authority":
            connection.execute("DELETE FROM model_usage_route_circuit_events WHERE route='GRAPHITI_CHAT_PRIMARY'")
        elif defect == "wrong-terminal":
            connection.execute("UPDATE model_invocation_terminals SET outcome='FAILED'")
        elif defect == "missing-work":
            connection.execute("DELETE FROM model_work_outcomes")
        elif defect == "missing-receipt":
            connection.execute("DELETE FROM unpublished_graphiti_attempt_receipts")
        elif defect in {"failed-receipt", "empty-leaves", "wrong-leaf-terminal", "wrong-leaf-outcome", "wrong-leaf-allocation", "wrong-leaf-envelope", "duplicate-leaf"}:
            def change(receipt):
                if defect == "failed-receipt":
                    receipt["outcome"] = "FAILED"
                elif defect == "empty-leaves":
                    receipt["chat_invocations"] = []
                elif defect == "duplicate-leaf":
                    receipt["chat_invocations"] *= 2
                else:
                    field = {
                        "wrong-leaf-terminal": "model_invocation_terminal_digest",
                        "wrong-leaf-outcome": "outcome",
                        "wrong-leaf-allocation": "model_invocation_allocation_digest",
                        "wrong-leaf-envelope": "model_work_envelope_id",
                    }[defect]
                    receipt["chat_invocations"][0][field] = "wrong"
            _rewrite_receipt(case, change)
        elif defect in {"missing-dispatch", "wrong-dispatch"}:
            connection.execute("DELETE FROM model_transport_observations")
        connection.commit()
        if defect in {"duplicate-dispatch", "wrong-dispatch"}:
            case.usage.observe_transport(
                invocation_id=case.allocation.invocation_id, state="DISPATCH_STARTED",
                observed_at=case.terminal.dispatch_at + timedelta(microseconds=1),
                evidence_digest=case.allocation.canonical_digest,
            )
        before = _unchanged(case)
        with pytest.raises(m.ModelUsageIntegrityError):
            _dispose(case)
        assert _unchanged(case) == before
        assert connection.execute("SELECT count(*) FROM model_usage_conservative_dispositions").fetchone() == (0,)
    finally:
        connection.close()


@pytest.mark.parametrize("field", ["authority_scope", "scope-downgrade", "native_scope_digest", "work_outcome_digest", "attempt_receipt_digest", "components"])
def test_fallback_cancellation_qualification_reproves_retained_authority(tmp_path, monkeypatch, field):
    case = _cancelled(tmp_path, monkeypatch)
    try:
        record = _dispose(case)
        record.pop("disposition_digest")
        if field == "scope-downgrade":
            record["authority_scope"] = m.NATIVE_AUTONOMOUS_USAGE_SCOPE
            case.connection.execute("UPDATE model_usage_conservative_dispositions SET approved_by=?",
                                    (m.NATIVE_AUTONOMOUS_USAGE_SCOPE,))
        else:
            record[field] = (m.UsageComponents(total_tokens=0, provenance="BOUNDED_ESTIMATE").as_record()
                             if field == "components" else digest_canonical({"wrong": field}))
        record["disposition_digest"] = digest_canonical(record)
        case.connection.execute("UPDATE model_usage_conservative_dispositions SET disposition_digest=?,record_json=?",
                                (record["disposition_digest"], canonical_json_bytes(record).decode()))
        case.connection.commit()
        with pytest.raises(NativeQualificationError):
            _invocations(case.connection, case.journal)
        # Qualification is a historical audit lane; new admission consumes the
        # already-settled current fact. Explicit settlement replay still checks.
        assert ROUTE not in m._usage_blocking_routes(case.connection)
        with pytest.raises((m.ModelUsageIntegrityError, m.ModelUsageAdmissionError)):
            _dispose(case)
    finally:
        case.connection.close()


def test_native_processor_settles_only_current_cancelled_fallback_without_retry(tmp_path, monkeypatch):
    observed_holds = []

    def held_ingest(*_args, **_kwargs):
        # The existing ingest boundary owns its dispatch gate; settlement never
        # changes that gate, submits another allocation or repeats provider I/O.
        observed_holds.append(case.usage.route_state(ROUTE)["state"])

    processor, connection, _ = _open(tmp_path, monkeypatch, ingest=held_ingest)
    case = _cancelled(tmp_path, monkeypatch, connection=connection)
    try:
        processor._usage = case.usage
        processor._clock = lambda: T0 + timedelta(seconds=12)
        before = _unchanged(case)
        processor._settle_missing_subscription_usage((_native("unrelated"),))
        assert connection.execute("SELECT count(*) FROM model_usage_conservative_dispositions").fetchone() == (0,)
        processor.advance((case.unit,), cycle_id="cancelled-fallback-settlement")
        processor.advance((case.unit,), cycle_id="cancelled-fallback-settlement-replay")
        assert connection.execute("SELECT count(*) FROM model_usage_conservative_dispositions").fetchone() == (1,)
        assert _unchanged(case) == before
        assert observed_holds == ["OPEN", "OPEN"]
        assert case.usage.route_state(ROUTE)["state"] == "OPEN"
    finally:
        connection.close()


@pytest.mark.parametrize("chat", [False, True])
def test_fallback_cancellation_does_not_admit_cash_embedding_or_primary_chat(tmp_path, monkeypatch, chat):
    from newsroom.tests.test_native_graphiti_embedding_disposition import _cancelled as other_leaf

    case = other_leaf(tmp_path, monkeypatch, chat=chat)
    try:
        before = _unchanged(case)
        with pytest.raises(m.ModelUsageIntegrityError, match="ineligible"):
            _dispose(case)
        assert _unchanged(case) == before
        assert case.connection.execute("SELECT count(*) FROM model_usage_conservative_dispositions").fetchone() == (0,)
    finally:
        case.connection.close()


@pytest.mark.parametrize("outcome", ("CANCELLED", "FAILED", "TIMEOUT"))
def test_parented_unknown_fallback_is_not_a_global_optional_settlement_failure(tmp_path, monkeypatch, outcome):
    processor, connection, _ = _open(
        tmp_path, monkeypatch, ingest=lambda *_a, **_k: pytest.fail("settlement must not dispatch"),
    )
    case = _cancelled(tmp_path, monkeypatch, connection=connection, parented=True, outcome=outcome)
    try:
        assert case.allocation.parent_invocation_id is not None
        processor._usage = case.usage
        processor._clock = lambda: T0 + timedelta(seconds=12)
        before = _unchanged(case)
        processor._settle_missing_subscription_usage((case.unit,))
        assert _unchanged(case) == before
        assert connection.execute("SELECT count(*) FROM model_usage_conservative_dispositions").fetchone() == (0,)
        assert case.usage.terminal(case.allocation.invocation_id).usage_status is m.UsageStatus.UNREPORTED
        assert case.usage.route_state(ROUTE)["state"] == "OPEN"
        assert case.usage.graphiti_work_route_state(ROUTE)["state"] == "CLOSED"
    finally:
        connection.close()


@pytest.mark.parametrize("tamper", ("identity", "allocation_header"))
def test_inapplicable_parented_fallback_still_authenticates_retained_binding(tmp_path, monkeypatch, tamper):
    processor, connection, _ = _open(
        tmp_path, monkeypatch, ingest=lambda *_a, **_k: pytest.fail("must not dispatch"),
    )
    case = _cancelled(tmp_path, monkeypatch, connection=connection, parented=True)
    try:
        processor._usage = case.usage
        processor._clock = lambda: T0 + timedelta(seconds=12)
        if tamper == "identity":
            connection.execute("UPDATE graphiti_internal_requests SET record_json=json_set(record_json,'$.semantic_request_class','changed') WHERE invocation_id=?", (case.allocation.invocation_id,))
        else:
            connection.execute("UPDATE model_invocation_allocations SET parent_invocation_id=NULL WHERE invocation_id=?", (case.allocation.invocation_id,))
        connection.commit()
        with pytest.raises(m.ModelUsageIntegrityError):
            processor._settle_missing_subscription_usage((case.unit,))
        assert connection.execute("SELECT count(*) FROM model_usage_conservative_dispositions").fetchone() == (0,)
    finally:
        connection.close()
