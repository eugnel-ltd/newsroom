"""Native cancellation estimates preserve unknown provider usage and cash."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import newsroom.control_plane.model_usage as model_usage_module
import newsroom.control_plane.native_progress as native_progress_module
from newsroom.authority.canonical import canonical_json_bytes, digest_canonical
from newsroom.control_plane.cycle import _graphiti_usage_cycle_id
from newsroom.control_plane.graphiti import GraphitiModelUsageObserver
from newsroom.control_plane.model_usage import (
    ModelUsageIntegrityError,
    ModelUsageService,
    UsageComponents,
    UsageStatus,
    WorkEnvelope,
    WorkloadClass,
)
from newsroom.control_plane.native_progress import NativeRevisionJournal
from newsroom.control_plane.native_qualification import (
    NativeQualificationError,
    NativeQualificationPending,
    _invocations,
)
from newsroom.control_plane.store import (
    connect,
    reconcile_graphiti_spend,
    reserve_graphiti_spend,
)
from newsroom.tests.test_native_graphiti import _native, _open

NOW = datetime(2026, 9, 12, 23, tzinfo=UTC)
ROUTE = "GRAPHITI_EMBEDDING"
MODEL = "openai/text-embedding-3-large"


def _cancelled(tmp_path, monkeypatch, *, connection=None, unit=None,
               land=True, reported=False, dispatched=True, chat=False,
               input_size=100, other_unresolved=False, other_inflight=False):
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti._graphiti_implementation_identity",
        lambda: ("a" * 40, True),
    )
    path = str(tmp_path / "private.sqlite3")
    usage = ModelUsageService(path)
    connection = connection or connect(path)
    unit = unit or _native("embedding-cancelled")
    journal = NativeRevisionJournal(connection)
    if land:
        journal.land((unit,))
    envelope = WorkEnvelope.create(
        cycle_id=_graphiti_usage_cycle_id(
            unit, attempt_number=1, requested_cycle_id=None,
        ),
        workload_class=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        admitted_at=NOW, admission_decision_id=None, candidate_id=None,
        hypothesis_digest=None, evidence_package_digest=None,
        ingest_id=unit.ingest_id, graphiti_attempt_id=f"{unit.ingest_id}:1",
    )
    usage.open_envelope(envelope)
    revision = unit.effective_revision
    observer = GraphitiModelUsageObserver(
        service=usage, envelope=envelope, clock=lambda: NOW,
        effective_revision_digest=digest_canonical({
            "source_id": revision.source_id, "item_key": revision.item_key,
            "revision_digest": revision.revision_digest,
            "first_observed_at": revision.first_observed_at,
        }),
        ingest_obligation_id=unit.ingest_id,
        dispatch_authority_digest=digest_canonical({"fixture": "native-rights"}),
        owner_stop_check=lambda: None,
    )
    allocation = (
        observer.before_cli_invocation(
            provider="cursor-agent-cli", model="composer-2.5",
            prompt="fixture chat", schema=None,
        ) if chat else observer.before_embedding_invocation(
            provider="openrouter", model=MODEL, input_data=["x" * input_size],
        )
    )
    other = (
        observer.before_embedding_invocation(
            provider="openrouter", model=MODEL, input_data=["other exact input"],
        ) if other_unresolved or other_inflight else None
    )
    if dispatched:
        observer.transport_dispatch_started(allocation)
        if other is not None:
            observer.transport_dispatch_started(other)
    value = (
        {"usage_basis": "PROVIDER_REPORTED", "input_tokens": 1,
         "output_tokens": 0, "total_tokens": 1,
         "provider_telemetry": {"request_id": "fixture", "total_tokens": 1}}
        if reported else {"usage_basis": "UNREPORTED"}
        if dispatched else {"usage_basis": "NO_PROVIDER_CALL"}
    )
    complete = observer.after_cli_invocation if chat else observer.after_embedding_invocation
    if other is not None and not other_inflight:
        complete(other, outcome="CANCELLED", usage=value)
    complete(allocation, outcome="CANCELLED", usage=value)
    terminal = usage.terminal(allocation.invocation_id)
    if not other_inflight:
        usage.record_work_outcome(
            envelope_id=envelope.envelope_id, outcome="GRAPHITI_FAILED",
            outcome_record_id="fixture-cancelled", payload_digest=None,
            terminal_at=NOW, stable_reason_codes=("CANCELLED",),
        )
    reserve_graphiti_spend(
        connection, spend_id="held-cancellation", ingest_id=unit.ingest_id,
        attempt_number=1, proving_run_id=unit.proving_run_id,
        generation_id="fixture-generation", reserved_gbp_microunits=500_000,
        ceiling_gbp_microunits=None,
    )
    reconcile_graphiti_spend(
        connection, spend_id="held-cancellation",
        embedding_usage={"usage_basis": "UNREPORTED"},
    )
    connection.commit()
    return SimpleNamespace(
        usage=usage, connection=connection, unit=unit, journal=journal,
        envelope=envelope, allocation=allocation, terminal=terminal,
        policy=observer._policies[allocation.invocation_id],
        observer=observer, other=other,
    )


def _dispose(case, *, usage=None):
    return (usage or case.usage).disposition_native_graphiti_embedding_cancellation(
        invocation_id=case.allocation.invocation_id,
        observed_at=NOW + timedelta(seconds=1),
    )


def _immutable_snapshot(case):
    return {
        table: case.connection.execute(f"SELECT * FROM {table}").fetchall()
        for table in (
            "model_invocation_allocations", "model_invocation_terminals",
            "model_transport_observations", "model_provider_telemetry",
            "model_usage_reconciliations", "unpublished_graphiti_spend",
        )
    }


def test_native_embedding_cancellation_estimates_once_without_releasing_cash(
    tmp_path, monkeypatch,
):
    case = _cancelled(tmp_path, monkeypatch)
    try:
        assert case.terminal.usage_status is UsageStatus.UNREPORTED
        assert case.terminal.outcome == "CANCELLED"
        assert case.terminal.failure_class == "MISSING_PROVIDER_TELEMETRY"
        assert case.usage.route_state(ROUTE)["state"] == "OPEN"
        before = _immutable_snapshot(case)
        with pytest.raises(NativeQualificationError, match="unresolved"):
            _invocations(case.connection, case.journal)

        record = _dispose(case)

        assert record["usage_status"] == "ESTIMATED"
        assert record["components"] == {
            "input_tokens": None, "output_tokens": None,
            "cached_read_tokens": None, "cached_write_tokens": None,
            "reasoning_tokens": None, "context_tokens": None,
            "total_tokens": case.policy.max_total_tokens,
            "provenance": "BOUNDED_ESTIMATE",
        }
        assert record["estimate_policy_digest"] == case.policy.canonical_digest
        assert record["terminal_digest"] == case.terminal.terminal_digest
        assert record["allocation_digest"] == case.allocation.canonical_digest
        assert record["exact_usage_remains_unknown"] is True
        assert record["provider_dispatch_preserved"] is True
        assert record["unknown_spend_released"] is False
        assert record["cash_spend_known"] is False
        assert _immutable_snapshot(case) == before
        assert case.connection.execute(
            "SELECT status,reserved_gbp_microunits,actual_gbp_microunits "
            "FROM unpublished_graphiti_spend"
        ).fetchone() == ("UNRECONCILED", 500_000, None)
        assert case.usage.route_state(ROUTE)["state"] == "CLOSED"
        assert _dispose(case, usage=ModelUsageService(case.usage.path)) == record
        assert case.connection.execute(
            "SELECT count(*) FROM model_usage_conservative_dispositions"
        ).fetchone() == (1,)
        assert _invocations(case.connection, case.journal) == (
            case.allocation.invocation_id,
        )
        retry = case.usage.native_graphiti_ingest_retry_evidence_many(
            failed_attempts={case.unit.ingest_id: 1}, max_attempts=6,
        )[case.unit.ingest_id]
        assert retry.zero_dispatch_attempts == ()
        assert retry.settled_provider_attempts == (1,)
        assert retry.unresolved_attempts == ()
    finally:
        case.connection.close()


def test_native_embedding_cancellation_uses_exact_input_bound_when_larger(
    tmp_path, monkeypatch,
):
    case = _cancelled(tmp_path, monkeypatch, input_size=140_000)
    try:
        assert case.policy.max_total_tokens < case.allocation.prompt_bytes
        record = _dispose(case)
        assert record["components"]["total_tokens"] == case.allocation.prompt_bytes
        assert record["estimated_policy_ceiling_exceeded"] is True
        assert record["exact_policy_compliance_unknown"] is True
        assert case.usage.terminal(case.allocation.invocation_id) == case.terminal
    finally:
        case.connection.close()


@pytest.mark.parametrize("field", ["native_scope_digest", "authority_scope"])
def test_native_qualification_reproves_embedding_disposition_authority(
    tmp_path, monkeypatch, field,
):
    case = _cancelled(tmp_path, monkeypatch)
    try:
        record = _dispose(case)
        changed = {**record, field: digest_canonical({"wrong": "scope"})}
        changed.pop("disposition_digest")
        changed["disposition_digest"] = digest_canonical(changed)
        case.connection.execute(
            "UPDATE model_usage_conservative_dispositions SET "
            "disposition_digest=?,record_json=? WHERE invocation_id=?",
            (changed["disposition_digest"], canonical_json_bytes(changed).decode(),
             case.allocation.invocation_id),
        )
        case.connection.commit()
        with pytest.raises(NativeQualificationError):
            _invocations(case.connection, case.journal)
        assert case.usage.route_state(ROUTE)["state"] == "CLOSED"
        with pytest.raises(ModelUsageIntegrityError):
            _dispose(case)
    finally:
        case.connection.close()


def test_native_embedding_cancellation_does_not_replay_native_progress(
    tmp_path, monkeypatch,
):
    case = _cancelled(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(
            NativeRevisionJournal, "__init__",
            lambda *_args, **_kwargs: pytest.fail("whole native journal replay"),
        )
        record = _dispose(case)
        assert _dispose(case) == record
        assert case.usage.route_state(ROUTE)["state"] == "CLOSED"
    finally:
        case.connection.close()


def test_route_state_snapshot_keeps_current_accounting_without_settled_history_replay(
    tmp_path, monkeypatch,
):
    case = _cancelled(tmp_path, monkeypatch)
    try:
        record = _dispose(case)
        before = _immutable_snapshot(case)
        calls = []
        original = model_usage_module._usage_blocking_routes

        def checked(connection):
            if len(calls) < 2:
                assert connection.in_transaction
            calls.append(connection)
            return original(connection)

        monkeypatch.setattr(model_usage_module, "_usage_blocking_routes", checked)
        with case.usage.route_state_snapshot() as read:
            assert read(ROUTE)["state"] == "CLOSED"
            assert read("GRAPHITI_CHAT_PRIMARY")["state"] == "CLOSED"
            assert len(calls) == 1
            assert _immutable_snapshot(case) == before
            changed = {**record, "exact_usage_remains_unknown": False}
            changed.pop("disposition_digest")
            changed["disposition_digest"] = digest_canonical(changed)
            case.connection.execute(
                "UPDATE model_usage_conservative_dispositions SET "
                "disposition_digest=?,record_json=? WHERE invocation_id=?",
                (changed["disposition_digest"], canonical_json_bytes(changed).decode(),
                 case.allocation.invocation_id),
            )
            case.connection.commit()
            # A single read transaction keeps the already proved state, not
            # a mixture of old blocker authority and newly changed records.
            assert read(ROUTE)["state"] == "CLOSED"
            assert len(calls) == 1
        assert case.connection.execute(
            "SELECT count(*) FROM model_usage_conservative_dispositions"
        ).fetchone() == (1,)
        assert _immutable_snapshot(case) == before
        with case.usage.route_state_snapshot() as read:
            assert read(ROUTE)["state"] == "CLOSED"
        assert len(calls) == 2
        assert case.usage.route_state("GRAPHITI_CHAT_PRIMARY")["state"] == "CLOSED"
        assert len(calls) == 3
        with pytest.raises(ModelUsageIntegrityError, match="native conservative disposition differs"):
            _dispose(case)
    finally:
        case.connection.close()


@pytest.mark.parametrize("leaf", ["chat", "embedding"])
def test_native_disposition_uses_request_revision_hint_before_landing_bodies(
    tmp_path, monkeypatch, leaf,
):
    case = _cancelled(tmp_path, monkeypatch, chat=leaf == "chat")
    try:
        unrelated = _native("unrelated-landing")
        case.journal.land((unrelated,))
        original_unit = native_progress_module._unit

        def selected_unit_only(value, bodies):
            if value["item_key"] == unrelated.item_key:
                pytest.fail("unrelated landing body was reconstructed")
            return original_unit(value, bodies)

        monkeypatch.setattr(native_progress_module, "_unit", selected_unit_only)
        if leaf == "embedding":
            record = _dispose(case)
            assert record["usage_status"] == "ESTIMATED"
        else:
            assert model_usage_module._native_envelope(
                case.connection,
                case.allocation,
            ) == case.envelope
    finally:
        case.connection.close()


@pytest.mark.parametrize("dispose_first", [False, True])
def test_exact_later_telemetry_is_not_replaced_or_blocked_by_an_estimate(
    tmp_path, monkeypatch, dispose_first,
):
    case = _cancelled(tmp_path, monkeypatch)
    try:
        if dispose_first:
            _dispose(case)
        case.usage.reconcile(
            invocation_id=case.allocation.invocation_id,
            components=UsageComponents(
                input_tokens=1, total_tokens=1, provenance="PROVIDER_REPORTED",
            ),
            provider_telemetry={"request_id": "later-fixture", "total_tokens": 1},
            observed_at=NOW + timedelta(seconds=2),
            raw_telemetry_pointer="fixture://later-provider-receipt",
        )
        if not dispose_first:
            with pytest.raises(ModelUsageIntegrityError):
                _dispose(case)
        assert case.usage.terminal(case.allocation.invocation_id) == case.terminal
        assert case.connection.execute(
            "SELECT count(*) FROM model_usage_conservative_dispositions"
        ).fetchone() == (int(dispose_first),)
        assert _invocations(case.connection, case.journal) == (case.allocation.invocation_id,)
        assert case.connection.execute(
            "SELECT status,reserved_gbp_microunits,actual_gbp_microunits "
            "FROM unpublished_graphiti_spend"
        ).fetchone() == ("UNRECONCILED", 500_000, None)
    finally:
        case.connection.close()


@pytest.mark.parametrize("case_kind", [
    "wrong-workload", "non-native", "pre-dispatch", "reported",
    "wrong-route-index", "wrong-model-index", "wrong-request-index",
    "corrupt-request",
    "missing-request", "bad-canonical-policy", "missing-dispatch",
    "multiple-dispatch", "wrong-dispatch-binding", "policy-breach",
])
def test_native_embedding_cancellation_rejects_ineligible_or_changed_evidence(
    tmp_path, monkeypatch, case_kind,
):
    case = _cancelled(
        tmp_path, monkeypatch, land=case_kind != "non-native",
        dispatched=case_kind != "pre-dispatch",
        reported=case_kind == "reported", chat=case_kind == "wrong-workload",
    )
    connection = case.connection
    invocation_id = case.allocation.invocation_id
    try:
        if case_kind in {"wrong-route-index", "wrong-model-index"}:
            column = "route" if case_kind == "wrong-route-index" else "model"
            connection.execute(
                f"UPDATE model_invocation_allocations SET {column}=? "
                "WHERE invocation_id=?", ("another-contract", invocation_id),
            )
        elif case_kind == "wrong-request-index":
            connection.execute(
                "UPDATE graphiti_internal_requests SET provider_attempt_id=? "
                "WHERE invocation_id=?",
                ("another-provider-attempt", invocation_id),
            )
        elif case_kind == "corrupt-request":
            raw, = connection.execute(
                "SELECT record_json FROM graphiti_internal_requests "
                "WHERE invocation_id=?", (invocation_id,),
            ).fetchone()
            record = json.loads(raw)
            record["ingest_obligation_id"] = "another-native-unit"
            connection.execute(
                "UPDATE graphiti_internal_requests SET record_json=? "
                "WHERE invocation_id=?",
                (canonical_json_bytes(record).decode(), invocation_id),
            )
        elif case_kind == "missing-request":
            connection.execute(
                "DELETE FROM graphiti_internal_requests WHERE invocation_id=?",
                (invocation_id,),
            )
        elif case_kind == "bad-canonical-policy":
            connection.execute(
                "UPDATE model_invocation_policies SET record_json=? "
                "WHERE canonical_digest=?",
                (json.dumps(case.policy.as_record(), indent=1),
                 case.policy.canonical_digest),
            )
        elif case_kind in {"missing-dispatch", "wrong-dispatch-binding"}:
            connection.execute(
                "DELETE FROM model_transport_observations WHERE invocation_id=?",
                (invocation_id,),
            )
        elif case_kind == "policy-breach":
            record = case.terminal.as_record()
            record["policy_breach"] = "MAX_TOTAL_TOKENS_EXCEEDED"
            record["terminal_digest"] = ""
            record["terminal_digest"] = digest_canonical(record)
            connection.execute(
                "UPDATE model_invocation_terminals SET terminal_digest=?,record_json=? "
                "WHERE invocation_id=?",
                (record["terminal_digest"], canonical_json_bytes(record).decode(),
                 invocation_id),
            )
        connection.commit()
        if case_kind in {"multiple-dispatch", "wrong-dispatch-binding"}:
            case.usage.observe_transport(
                invocation_id=invocation_id,
                observed_at=(NOW + timedelta(microseconds=1)
                             if case_kind == "multiple-dispatch" else NOW),
                state="DISPATCH_STARTED",
                evidence_digest=(
                    digest_canonical({
                        "invocation_id": invocation_id, "provider": "openrouter",
                        "route": ROUTE, "request_digest": case.allocation.request_digest,
                    }) if case_kind == "multiple-dispatch"
                    else digest_canonical({"unrelated": "request"})
                ),
            )
        before = _immutable_snapshot(case)
        with pytest.raises(ModelUsageIntegrityError):
            _dispose(case)
        assert _immutable_snapshot(case) == before
        assert connection.execute(
            "SELECT count(*) FROM model_usage_conservative_dispositions"
        ).fetchone() == (0,)
    finally:
        connection.close()


@pytest.mark.parametrize("blocker", ["explicit-other-cause", "other-unresolved", "other-inflight"])
def test_native_embedding_cancellation_does_not_release_other_route_causes(
    tmp_path, monkeypatch, blocker,
):
    case = _cancelled(
        tmp_path, monkeypatch, other_unresolved=blocker == "other-unresolved",
        other_inflight=blocker == "other-inflight",
    )
    try:
        if blocker == "explicit-other-cause":
            case.usage.open_route_circuit(
                route=ROUTE, reason="AUTHENTICATION", invocation_id=None,
                recorded_at=NOW + timedelta(microseconds=1),
            )
        before = _immutable_snapshot(case)
        _dispose(case)
        assert _immutable_snapshot(case) == before
        assert case.usage.route_state(ROUTE)["state"] == "OPEN"
    finally:
        case.connection.close()


@pytest.mark.parametrize("retained", [True, False])
def test_native_advance_settles_embedding_cancellation_without_provider_replay(
    tmp_path, monkeypatch, retained,
):
    cases = []
    observed = []
    unit = _native("processor-cancelled")

    def ingest(connection, **_kwargs):
        if retained:
            observed.append(cases[0].usage.route_state(ROUTE)["state"])
        else:
            cases.append(_cancelled(
                tmp_path, monkeypatch, connection=connection, unit=unit,
            ))
            observed.append(cases[0].usage.route_state(ROUTE)["state"])
        # No provider is executed, and no allocation is repeated.

    processor, connection, _ = _open(tmp_path, monkeypatch, ingest=ingest)
    try:
        if retained:
            cases.append(_cancelled(
                tmp_path, monkeypatch, connection=connection, unit=unit,
            ))
        processor._usage = ModelUsageService(str(tmp_path / "private.sqlite3"))
        processor._clock = lambda: NOW + timedelta(seconds=1)
        processor.advance((unit,), cycle_id="native-cancellation-settlement")
        assert observed == (["CLOSED"] if retained else ["OPEN"])
        case = cases[0]
        assert case.usage.route_state(ROUTE)["state"] == "CLOSED"
        assert case.usage.terminal(case.allocation.invocation_id) == case.terminal
        assert connection.execute(
            "SELECT count(*) FROM model_invocation_allocations"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT count(*) FROM model_usage_conservative_dispositions"
        ).fetchone() == (1,)
    finally:
        connection.close()


@pytest.mark.parametrize("corrupt_total", [False, True])
def test_qualification_only_defers_semantically_valid_unknown_terminal(
    tmp_path, monkeypatch, corrupt_total,
):
    case = _cancelled(tmp_path, monkeypatch)
    try:
        if corrupt_total:
            record = case.terminal.as_record()
            record["components"]["total_tokens"] = 1
            record["terminal_digest"] = ""
            record["terminal_digest"] = digest_canonical(record)
            case.connection.execute(
                "UPDATE model_invocation_terminals SET terminal_digest=?,record_json=? "
                "WHERE invocation_id=?",
                (record["terminal_digest"], canonical_json_bytes(record).decode(),
                 case.allocation.invocation_id),
            )
            case.connection.commit()
        with pytest.raises(NativeQualificationError) as failure:
            _invocations(case.connection, case.journal)
        assert isinstance(failure.value, NativeQualificationPending) is not corrupt_total
    finally:
        case.connection.close()


@pytest.mark.parametrize("later_circuit", [None, "AUTHENTICATION", "POLICY_BREACH"])
def test_native_settlement_revisits_only_own_pending_circuit(
    tmp_path, monkeypatch, later_circuit,
):
    processor, connection, _ = _open(tmp_path, monkeypatch, ingest=lambda *_a, **_k: None)
    try:
        case = _cancelled(tmp_path, monkeypatch, connection=connection, other_inflight=True)
        processor._usage = case.usage
        processor._clock = lambda: NOW + timedelta(seconds=1)
        calls = []
        original = case.usage.disposition_native_graphiti_embedding_cancellation

        def dispose(**kwargs):
            calls.append(kwargs["invocation_id"])
            return original(**kwargs)

        monkeypatch.setattr(case.usage, "disposition_native_graphiti_embedding_cancellation", dispose)
        processor._settle_missing_subscription_usage((case.unit,))
        assert case.usage.route_state(ROUTE)["state"] == "OPEN"
        assert calls == [case.allocation.invocation_id]
        case.observer.after_embedding_invocation(
            case.other, outcome="COMPLETE", usage={
                "usage_basis": "PROVIDER_REPORTED", "input_tokens": 1,
                "output_tokens": 0, "total_tokens": 1,
                "provider_telemetry": {"request_id": "other-completed", "total_tokens": 1},
            },
        )
        if later_circuit:
            case.usage.open_route_circuit(
                route=ROUTE, reason=later_circuit, invocation_id=None,
                recorded_at=NOW + timedelta(microseconds=1),
            )
        processor._settle_missing_subscription_usage((case.unit,))
        assert case.usage.route_state(ROUTE)["state"] == ("OPEN" if later_circuit else "CLOSED")
        assert calls == [case.allocation.invocation_id] * (1 if later_circuit else 2)
        processor._settle_missing_subscription_usage((case.unit,))
        assert calls == [case.allocation.invocation_id] * (1 if later_circuit else 2)
        assert case.usage.terminal(case.allocation.invocation_id) == case.terminal
        assert connection.execute(
            "SELECT count(*) FROM model_usage_conservative_dispositions"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT status,reserved_gbp_microunits,actual_gbp_microunits "
            "FROM unpublished_graphiti_spend"
        ).fetchone() == ("UNRECONCILED", 500_000, None)
    finally:
        connection.close()
