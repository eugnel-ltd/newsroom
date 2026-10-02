"""Admission consumes current facts, not settled source/debug history."""

from dataclasses import replace
from datetime import timedelta

import pytest

from newsroom.control_plane import model_usage as m
from newsroom.tests.test_model_usage_preflight_snapshot import _prepared
from newsroom.tests.test_model_usage_receipts import T0, _allocation, _reported


def _unknown(allocation):
    return replace(
        _reported(allocation), usage_status=m.UsageStatus.UNREPORTED,
        components=m.UsageComponents(provenance="UNAVAILABLE"),
        failure_class="MISSING_PROVIDER_TELEMETRY", provider_telemetry_digest=None,
        raw_telemetry_pointer=None, terminal_digest="",
    )


def _current(service):
    connection = service._connection()
    try:
        return connection.execute(
            "SELECT invocation_id,active,unresolved,policy_breach FROM model_usage_current ORDER BY invocation_id"
        ).fetchall()
    finally:
        connection.close()


def test_active_allocation_dispatch_and_exact_terminal_do_not_self_block(tmp_path):
    service, envelope, policy = _prepared(tmp_path)
    allocation = _allocation(envelope, policy)
    service.allocate(allocation, owner_emergency_stop=False)
    assert tuple(_current(service)[0])[1:] == (1, 0, 0)
    assert service.route_state(policy.route)["state"] == "CLOSED"
    service.observe_transport(invocation_id=allocation.invocation_id,
                              observed_at=allocation.allocated_at,
                              state="DISPATCH_STARTED", evidence_digest=allocation.request_digest)
    service.complete(_reported(allocation))
    assert _current(service) == []
    assert service.route_state(policy.route)["state"] == "CLOSED"


def test_multiple_unknowns_reconcile_independently_and_reopen(tmp_path):
    service, envelope, policy = _prepared(tmp_path)
    first = _allocation(envelope, policy)
    second = _allocation(envelope, policy, leaf_ordinal=2, request="second")
    for allocation in (first, second):
        service.allocate(allocation, owner_emergency_stop=False)
    for allocation in (first, second):
        service.complete(_unknown(allocation))
    assert len(_current(service)) == 2
    assert service.route_state(policy.route)["state"] == "OPEN"
    def reconcile(allocation):
        service.reconcile(invocation_id=allocation.invocation_id,
            components=m.UsageComponents(input_tokens=1, output_tokens=1, total_tokens=2,
                                        provenance="PROVIDER_REPORTED"),
            provider_telemetry={"total_tokens": 2}, observed_at=T0+timedelta(minutes=1),
            raw_telemetry_pointer="fixture://usage")
    reconcile(first)
    assert len(_current(service)) == 1
    assert service.route_state(policy.route)["state"] == "OPEN"
    reconcile(second)
    assert _current(service) == []
    assert service.route_state(policy.route)["state"] == "CLOSED"
    # Replaying the old unknown terminal does not resurrect its settled blocker.
    service.complete(_unknown(first))
    assert _current(service) == []
    third = _allocation(envelope, policy, leaf_ordinal=3, request="third")
    service.allocate(third, owner_emergency_stop=False)
    service.complete(_unknown(third))
    assert len(_current(service)) == 1
    assert service.route_state(policy.route)["state"] == "OPEN"


def test_current_row_tamper_fails_closed(tmp_path):
    service, envelope, policy = _prepared(tmp_path)
    allocation = _allocation(envelope, policy)
    service.allocate(allocation, owner_emergency_stop=False)
    service.complete(_unknown(allocation))
    connection = service._connection()
    try:
        connection.execute("UPDATE model_usage_current SET route='WRONG'")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(m.ModelUsageIntegrityError, match="current"):
        service.route_state(policy.route)


@pytest.mark.parametrize("unknown", (False, True))
def test_missing_active_or_unknown_current_row_denies_next_dispatch(tmp_path, unknown):
    service, envelope, policy = _prepared(tmp_path)
    allocation = _allocation(envelope, policy)
    service.allocate(allocation, owner_emergency_stop=False)
    if unknown:
        service.complete(_unknown(allocation))
    connection = service._connection()
    try:
        connection.execute("DELETE FROM model_usage_current")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(m.ModelUsageIntegrityError, match="inventory"):
        service.route_state(policy.route)
    with pytest.raises(m.ModelUsageIntegrityError, match="inventory"):
        service.allocate(_allocation(envelope, policy, leaf_ordinal=2, request="next"), owner_emergency_stop=False)


@pytest.mark.parametrize("mutation", ("terminal-status", "terminal-record", "allocation-route", "allocation-digest", "active-request"))
def test_current_source_binding_tamper_is_not_a_settled_history_waiver(tmp_path, mutation):
    service, envelope, policy = _prepared(tmp_path)
    allocation = _allocation(envelope, policy)
    service.allocate(allocation, owner_emergency_stop=False)
    if mutation != "active-request":
        service.complete(_unknown(allocation))
    statements = {
        "terminal-status": "UPDATE model_invocation_terminals SET usage_status='REPORTED'",
        "terminal-record": "UPDATE model_invocation_terminals SET record_json=json_set(record_json,'$.components.total_tokens',0)",
        "allocation-route": "UPDATE model_invocation_allocations SET route='WRONG'",
        "allocation-digest": "UPDATE model_invocation_allocations SET canonical_digest='wrong'",
        "active-request": "UPDATE model_invocation_allocations SET request_digest='wrong'",
    }
    connection = service._connection()
    try:
        connection.execute(statements[mutation])
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(m.ModelUsageIntegrityError):
        service.route_state(policy.route)


def test_current_refresh_failure_rolls_back_terminal_and_transport_accounting(tmp_path, monkeypatch):
    service, envelope, policy = _prepared(tmp_path)
    allocation = _allocation(envelope, policy)
    service.allocate(allocation, owner_emergency_stop=False)
    before = [tuple(row) for row in _current(service)]
    def fail(*_args, **_kwargs):
        raise m.ModelUsageIntegrityError("fixture current write failure")
    monkeypatch.setattr(m, "_refresh_current_usage", fail)
    with pytest.raises(m.ModelUsageIntegrityError, match="fixture"):
        service.complete(_unknown(allocation))
    assert service.terminal(allocation.invocation_id) is None
    assert [tuple(row) for row in _current(service)] == before
    assert service.route_state(policy.route)["state"] == "CLOSED"


def test_settled_history_not_replayed_on_default_route_reads(tmp_path, monkeypatch):
    from newsroom.tests.test_native_graphiti_embedding_disposition import _cancelled, _dispose, ROUTE
    case = _cancelled(tmp_path, monkeypatch)
    try:
        _dispose(case)
        assert case.usage.route_state(ROUTE)["state"] == "CLOSED"
        def historical_proof(*_args, **_kwargs):
            raise AssertionError("settled historical proof entered admission")
        for name in ("_valid_native_disposition", "_reported_output_disposition_authority",
                     "_assessor_requalification_authority", "_native_landed_source_unit"):
            monkeypatch.setattr(m, name, historical_proof)
        case.connection.execute("DELETE FROM model_transport_observations")
        case.connection.commit()
        assert case.usage.route_state(ROUTE)["state"] == "CLOSED"
        assert case.connection.execute(
            "SELECT status,reserved_gbp_microunits,actual_gbp_microunits FROM unpublished_graphiti_spend"
        ).fetchone() == ("UNRECONCILED", 500_000, None)
    finally:
        case.connection.close()


def test_populated_legacy_store_requires_explicit_transactional_import(tmp_path):
    from newsroom.control_plane.model_usage_current import CURRENT_MIGRATION_ID
    service, envelope, policy = _prepared(tmp_path)
    allocation = _allocation(envelope, policy)
    service.allocate(allocation, owner_emergency_stop=False)
    service.complete(_unknown(allocation))
    connection = service._connection()
    try:
        connection.execute("DELETE FROM model_usage_current")
        connection.execute("DELETE FROM model_usage_migrations WHERE migration_id=?", (CURRENT_MIGRATION_ID,))
        connection.commit()
    finally:
        connection.close()
    # Merely constructing the service does not replay or import old history.
    service = m.ModelUsageService(service.path)
    assert _current(service) == []
    with pytest.raises(m.ModelUsageIntegrityError, match="import"):
        service.route_state(policy.route)
    assert service.import_current_state() == 1
    assert service.import_current_state() == 0
    assert service.route_state(policy.route)["state"] == "OPEN"


def test_failed_reconciliation_rolls_back_current_fact_and_exact_usage(tmp_path, monkeypatch):
    service, envelope, policy = _prepared(tmp_path)
    allocation = _allocation(envelope, policy)
    service.allocate(allocation, owner_emergency_stop=False)
    service.complete(_unknown(allocation))
    def fail(*_args, **_kwargs):
        raise m.ModelUsageIntegrityError("fixture current write failure")
    monkeypatch.setattr(m, "_refresh_current_usage", fail)
    with pytest.raises(m.ModelUsageIntegrityError, match="fixture"):
        service.reconcile(invocation_id=allocation.invocation_id,
            components=m.UsageComponents(total_tokens=2, provenance="PROVIDER_REPORTED"),
            provider_telemetry={"total_tokens": 2}, observed_at=T0+timedelta(minutes=1),
            raw_telemetry_pointer="fixture://usage")
    assert len(_current(service)) == 1
    connection = service._connection()
    try:
        assert connection.execute("SELECT COUNT(*) FROM model_usage_reconciliations").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM model_provider_telemetry").fetchone()[0] == 0
    finally:
        connection.close()


def test_failed_legacy_import_rolls_back_all_rows_and_readiness(tmp_path, monkeypatch):
    from newsroom.control_plane import model_usage_current as current
    service, envelope, policy = _prepared(tmp_path)
    for ordinal in (1, 2):
        allocation = _allocation(envelope, policy, leaf_ordinal=ordinal, request=str(ordinal))
        service.allocate(allocation, owner_emergency_stop=False)
    connection = service._connection()
    try:
        connection.execute("DELETE FROM model_usage_current")
        connection.execute("DELETE FROM model_usage_migrations WHERE migration_id=?", (current.CURRENT_MIGRATION_ID,))
        connection.commit()
    finally:
        connection.close()
    original = current.refresh
    calls = []
    def fail_second(connection, invocation_id, **kwargs):
        calls.append(invocation_id)
        original(connection, invocation_id, **kwargs)
        if len(calls) == 2:
            raise m.ModelUsageIntegrityError("fixture import failure")
    monkeypatch.setattr(current, "refresh", fail_second)
    with pytest.raises(m.ModelUsageIntegrityError, match="fixture"):
        service.import_current_state()
    assert _current(service) == []
    connection = service._connection()
    try:
        assert not current.ready(connection)
    finally:
        connection.close()


def test_settled_history_growth_does_not_increase_default_guard_work(tmp_path):
    service, envelope, policy = _prepared(tmp_path)
    allocation = _allocation(envelope, policy)
    service.allocate(allocation, owner_emergency_stop=False)
    service.complete(_unknown(allocation))
    connection = service._connection()
    try:
        costs = []
        previous = 0
        for size in (0, 100, 2000):
            for ordinal in range(previous, size):
                identifier = f"settled-{ordinal}"
                connection.execute("INSERT INTO model_invocation_allocations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (identifier, envelope.envelope_id, envelope.cycle_id, ordinal+2,
                     policy.workload_class.value, policy.canonical_digest, policy.provider,
                     policy.route, policy.model, identifier, None, "2026-08-01T00:00:00Z", identifier, "{}"))
                connection.execute("INSERT INTO model_invocation_terminals VALUES(?,?,?,?,?,?,?)",
                    (identifier, identifier, "REPORTED", "ACCEPTED", None, "2026-08-01T00:00:01Z", "{}"))
                # Expired/malformed settled diagnostics are outside admission.
                connection.execute("INSERT INTO model_usage_conservative_dispositions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (identifier, identifier, identifier, identifier, policy.canonical_digest,
                     identifier, identifier, "fixture", "fixture", "2026-08-01T00:00:01Z",
                     "2026-08-01T00:00:01Z", "ESTIMATED", '{"authority_scope":"NATIVE_AUTONOMOUS_INTERNAL_PIPELINE","expired_debug":true}'))
            connection.commit()
            # Warm the statement after schema/planner changes, count deterministic
            # SQLite work rather than asserting noisy wall-clock thresholds.
            assert m._usage_blocking_routes(connection) == {policy.route}
            steps = [0]
            def count():
                steps[0] += 1
                return 0
            connection.set_progress_handler(count, 1)
            assert m._usage_blocking_routes(connection) == {policy.route}
            connection.set_progress_handler(None, 0)
            costs.append(steps[0])
            previous = size
        assert max(costs) - min(costs) <= 4
        assert len(_current(service)) == 1
    finally:
        connection.close()


def test_no_retry_reads_current_reservations_without_historical_source_proof(tmp_path, monkeypatch):
    from newsroom.tests.test_native_reported_output_disposition import _failed, _dispose
    case = _failed(tmp_path, monkeypatch)
    try:
        _dispose(case)
        def history(*_args, **_kwargs):
            raise AssertionError("historical source proof entered selected no-retry read")
        monkeypatch.setattr(m, "_reported_output_disposition_authority", history)
        assert m.reported_output_rejected_ingests(case.connection, ingest_ids=(case.unit.ingest_id,)) == {case.unit.ingest_id}
        case.connection.execute("UPDATE model_usage_reported_output_dispositions SET record_json='{}'")
        case.connection.commit()
        # A malformed current reservation has no trustworthy selector; it
        # must not disappear merely because a different ingest was requested.
        with pytest.raises(m.ModelUsageIntegrityError):
            m.reported_output_rejected_ingests(case.connection, ingest_ids=("unrelated",))
        with pytest.raises(m.ModelUsageIntegrityError):
            m.reported_output_rejected_ingests(case.connection, ingest_ids=(case.unit.ingest_id,))
        with pytest.raises(m.ModelUsageIntegrityError):
            m.reported_output_rejected_ingests(case.connection)
    finally:
        case.connection.close()


@pytest.mark.parametrize("defect", ("digest", "terminal", "policy", "components", "reconciliation", "unknown-scope", "wrong-plan"))
def test_legacy_import_authenticates_compact_settlement_not_membership(tmp_path, monkeypatch, defect):
    from newsroom.control_plane.model_usage_current import CURRENT_MIGRATION_ID
    from newsroom.authority.canonical import canonical_json_bytes, digest_canonical
    from newsroom.tests.test_native_graphiti_embedding_disposition import _cancelled, _dispose
    case = _cancelled(tmp_path, monkeypatch)
    try:
        record = _dispose(case)
        case.connection.execute("DELETE FROM model_usage_current")
        case.connection.execute("DELETE FROM model_usage_migrations WHERE migration_id=?", (CURRENT_MIGRATION_ID,))
        case.connection.execute("DELETE FROM model_usage_current_inventory")
        if defect == "reconciliation":
            case.connection.execute("DELETE FROM model_usage_conservative_dispositions")
            record = {"invocation_id": case.allocation.invocation_id, "usage_status": "REPORTED",
                      "provider_telemetry_digest": "missing", "policy_breach": None}
            record["reconciliation_digest"] = digest_canonical(record)
            case.connection.execute("INSERT INTO model_usage_reconciliations VALUES(?,?,?,?)",
                (record["reconciliation_digest"], case.allocation.invocation_id, T0.isoformat(), canonical_json_bytes(record).decode()))
        else:
            if defect == "digest":
                record["exact_usage_remains_unknown"] = False
            else:
                if defect == "components":
                    record["components"]["total_tokens"] = 0
                elif defect == "unknown-scope":
                    record["authority_scope"] = "UNRECOGNISED_BUSINESS_AUTHORITY"
                elif defect == "wrong-plan":
                    from newsroom.control_plane.issue_790_contract import issue_790_approved_plan_contracts
                    contract = issue_790_approved_plan_contracts()[0]
                    record.pop("authority_scope")
                    record.update(approved_plan_digest=contract.plan_digest,
                        approved_by=contract.approved_by, approval_reference=contract.approval_reference,
                        approved_at=contract.approved_at)
                    case.connection.execute(
                        "UPDATE model_usage_conservative_dispositions SET approved_plan_digest=?,approved_by=?,approval_reference=?,approved_at=?",
                        (contract.plan_digest, contract.approved_by, contract.approval_reference, contract.approved_at))
                else:
                    record[defect + "_digest"] = "wrong"
                record.pop("disposition_digest")
                record["disposition_digest"] = digest_canonical(record)
            case.connection.execute("UPDATE model_usage_conservative_dispositions SET disposition_digest=?,record_json=?",
                (record["disposition_digest"], canonical_json_bytes(record).decode()))
        case.connection.commit()
        with pytest.raises((m.ModelUsageIntegrityError, ValueError), match="current"):
            case.usage.import_current_state()
        assert case.connection.execute("SELECT COUNT(*) FROM model_usage_current").fetchone()[0] == 0
        assert case.connection.execute("SELECT 1 FROM model_usage_migrations WHERE migration_id=?", (CURRENT_MIGRATION_ID,)).fetchone() is None
    finally:
        case.connection.close()


@pytest.mark.parametrize('column', ('authority_digest', 'approved_plan_digest', 'approved_by', 'approval_reference', 'approved_at', 'observed_at'))
def test_native_legacy_import_binds_compact_sql_approval_fields(tmp_path, monkeypatch, column):
    from newsroom.control_plane.model_usage_current import CURRENT_MIGRATION_ID
    from newsroom.tests.test_native_graphiti_embedding_disposition import _cancelled, _dispose
    case = _cancelled(tmp_path, monkeypatch)
    try:
        _dispose(case)
        case.connection.execute(f'UPDATE model_usage_conservative_dispositions SET {column}=?', ('different-binding',))
        case.connection.execute('DELETE FROM model_usage_current')
        case.connection.execute('DELETE FROM model_usage_current_inventory')
        case.connection.execute('DELETE FROM model_usage_migrations WHERE migration_id=?', (CURRENT_MIGRATION_ID,))
        case.connection.commit()
        with pytest.raises((m.ModelUsageIntegrityError, ValueError), match='current'):
            case.usage.import_current_state()
        assert case.connection.execute('SELECT count(*) FROM model_usage_current').fetchone()[0] == 0
        assert case.connection.execute('SELECT 1 FROM model_usage_migrations WHERE migration_id=?', (CURRENT_MIGRATION_ID,)).fetchone() is None
    finally:
        case.connection.close()
