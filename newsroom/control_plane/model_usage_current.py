"""Durable, invocation-keyed current usage; historical proof stays off admission."""

from datetime import UTC, datetime
import json
import sqlite3

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.control_plane.cycle_governor import CONT_WRITER_ROUTE

CURRENT_MIGRATION_ID = "model-usage-current-v1"
SCHEMA = """
CREATE TABLE IF NOT EXISTS model_usage_current(
    invocation_id TEXT PRIMARY KEY,
    route TEXT NOT NULL,
    allocation_digest TEXT NOT NULL,
    terminal_digest TEXT,
    active INTEGER NOT NULL CHECK(active IN (0,1)),
    unresolved INTEGER NOT NULL CHECK(unresolved IN (0,1)),
    policy_breach INTEGER NOT NULL CHECK(policy_breach IN (0,1)),
    canonical_digest TEXT NOT NULL,
    CHECK(active=1 OR unresolved=1 OR policy_breach=1),
    CHECK(active=0 OR (unresolved=0 AND policy_breach=0)),
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS model_usage_current_route ON model_usage_current(route);
CREATE TABLE IF NOT EXISTS model_usage_current_inventory(
    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
    row_count INTEGER NOT NULL CHECK(row_count>=0),
    canonical_digest TEXT NOT NULL
);
"""
_FIELDS = ("invocation_id", "route", "allocation_digest", "terminal_digest",
           "active", "unresolved", "policy_breach")
_SELECT = ",".join(_FIELDS) + ",canonical_digest"


class CurrentUsageIntegrityError(ValueError):
    """Current usage is absent, inconsistent or changed outside its transaction."""


def canonical_route(route: str) -> str:
    return CONT_WRITER_ROUTE if route.startswith("CONT_") and route != "CONT_HEALTH_PROBE" else route


def ready(connection: sqlite3.Connection) -> bool:
    return connection.execute(
        "SELECT 1 FROM model_usage_migrations WHERE migration_id=?", (CURRENT_MIGRATION_ID,)
    ).fetchone() is not None


def _mark_ready(connection: sqlite3.Connection) -> None:
    connection.execute("INSERT INTO model_usage_migrations VALUES(?,?,?)",
        (CURRENT_MIGRATION_ID, "newsroom.model-usage.current.v1", datetime.now(UTC).isoformat()))


def _set_inventory(connection: sqlite3.Connection, count: int) -> None:
    connection.execute(
        "INSERT INTO model_usage_current_inventory VALUES(1,?,?) ON CONFLICT(singleton) DO UPDATE "
        "SET row_count=excluded.row_count,canonical_digest=excluded.canonical_digest",
        (count, digest_canonical({"row_count": count})),
    )


def _inventory(connection: sqlite3.Connection, actual: int | None = None) -> int:
    row = connection.execute("SELECT row_count,canonical_digest FROM model_usage_current_inventory WHERE singleton=1").fetchone()
    if row is None or digest_canonical({"row_count": row[0]}) != row[1]:
        raise CurrentUsageIntegrityError("current usage inventory differs")
    if actual is None:
        actual = connection.execute("SELECT COUNT(*) FROM model_usage_current").fetchone()[0]
    if actual != row[0]:
        raise CurrentUsageIntegrityError("current usage inventory has missing or extra rows")
    return actual


def initialise_empty(connection: sqlite3.Connection) -> None:
    """New stores are cheap; populated legacy stores require explicit import."""
    if not ready(connection) and connection.execute(
        "SELECT 1 FROM model_invocation_allocations LIMIT 1"
    ).fetchone() is None:
        if connection.execute("SELECT 1 FROM model_usage_current LIMIT 1").fetchone():
            raise CurrentUsageIntegrityError("current usage inventory is orphaned")
        _set_inventory(connection, 0)
        _mark_ready(connection)


def require_ready(connection: sqlite3.Connection) -> None:
    if not ready(connection):
        raise CurrentUsageIntegrityError("current usage requires explicit quiescent import")


def _checked(row) -> dict:
    record = dict(zip(_FIELDS, tuple(row)[:-1], strict=True))
    if digest_canonical(record) != row[-1]:
        raise CurrentUsageIntegrityError("current usage digest differs")
    if (record["active"] not in (0, 1) or record["unresolved"] not in (0, 1)
            or record["policy_breach"] not in (0, 1)
            or not any(record[key] for key in ("active", "unresolved", "policy_breach"))
            or bool(record["active"]) != (record["terminal_digest"] is None)):
        raise CurrentUsageIntegrityError("current usage state differs")
    return record


def _active_allocation(connection: sqlite3.Connection, invocation_id: str):
    from .model_usage import _allocation_from_record, _object, _utc_text

    row = connection.execute("SELECT * FROM model_invocation_allocations WHERE invocation_id=?", (invocation_id,)).fetchone()
    if row is None:
        raise CurrentUsageIntegrityError("current active allocation is absent")
    allocation = _allocation_from_record(_object(row[-1]))
    if tuple(row[:-1]) != (
        allocation.invocation_id, allocation.envelope_id, allocation.cycle_id, allocation.leaf_ordinal,
        allocation.workload_class.value, allocation.invocation_policy_digest, allocation.provider,
        allocation.route, allocation.model, allocation.request_digest, allocation.parent_invocation_id,
        _utc_text(allocation.allocated_at), allocation.canonical_digest,
    ):
        raise CurrentUsageIntegrityError("current active allocation SQL binding differs")
    return allocation


def blocking_routes(connection: sqlite3.Connection) -> set[str]:
    require_ready(connection)
    rows = connection.execute(f"SELECT {_SELECT} FROM model_usage_current").fetchall()
    _inventory(connection, len(rows))
    result = set()
    for row in rows:
        record = _checked(row)
        # Authenticate the selected current bindings, never settled history.
        binding = connection.execute(
            "SELECT a.route,a.canonical_digest,t.terminal_digest FROM model_invocation_allocations a "
            "LEFT JOIN model_invocation_terminals t USING(invocation_id) WHERE a.invocation_id=?",
            (record["invocation_id"],),
        ).fetchone()
        if binding is None or (canonical_route(binding[0]), binding[1], binding[2]) != (
            record["route"], record["allocation_digest"], record["terminal_digest"]
        ):
            raise CurrentUsageIntegrityError("current usage invocation binding differs")
        from .model_usage import _retained_terminal_allocation
        if record["terminal_digest"] is not None:
            _retained_terminal_allocation(connection, record["invocation_id"])
        else:
            allocation = _active_allocation(connection, record["invocation_id"])
            if (allocation.invocation_id != record["invocation_id"]
                    or allocation.canonical_digest != record["allocation_digest"]
                    or canonical_route(allocation.route) != record["route"]):
                raise CurrentUsageIntegrityError("current active allocation differs")
        if record["unresolved"] or record["policy_breach"]:
            result.add(record["route"])
    return result


def has_active(connection: sqlite3.Connection, route: str) -> bool:
    require_ready(connection)
    _inventory(connection)
    return any(_checked(row)["active"] for row in connection.execute(
        f"SELECT {_SELECT} FROM model_usage_current WHERE route=?", (canonical_route(route),)
    ))


def _table(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _bound_record(raw: str, digest: str, digest_field: str, expected: dict) -> dict:
    try:
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise ValueError("not an object")
        unsigned = dict(record)
        supplied = unsigned.pop(digest_field, None)
    except (ValueError, TypeError) as exc:
        raise CurrentUsageIntegrityError("current settlement record is malformed") from exc
    if (supplied != digest or digest_canonical(unsigned) != digest
            or canonical_json_bytes(record).decode() != raw
            or any(record.get(key) != value for key, value in expected.items())):
        raise CurrentUsageIntegrityError("current settlement binding differs")
    return record


def _approved_conservative_plan(connection: sqlite3.Connection, record: dict, allocation, terminal) -> None:
    """Authenticate retained business approval, not expired review/CLI evidence."""
    from .issue_790_contract import issue_790_approved_plan_contract

    plan_digest = record.get("approved_plan_digest")
    try:
        contract = issue_790_approved_plan_contract(plan_digest)
    except KeyError:
        activation = None
        if _table(connection, "issue_790_step16_activations"):
            activation = connection.execute(
                "SELECT activation_digest,record_json FROM issue_790_step16_activations WHERE plan_digest=?",
                (plan_digest,),
            ).fetchone()
        if activation is None:
            raise CurrentUsageIntegrityError("current conservative approved plan is absent") from None
        from .issue_790_step16_activation import activation_record_to_contract

        retained = _bound_record(activation[1], activation[0], "activation_digest", {"plan_digest": plan_digest})
        contract = activation_record_to_contract(retained)
    if (contract.plan_digest != plan_digest or contract.invocation_id != allocation.invocation_id
            or contract.allocation_digest != allocation.canonical_digest
            or contract.terminal_digest != terminal.terminal_digest
            or contract.terminal_outcome != terminal.outcome
            or any(record.get(key) != getattr(contract, key)
                   for key in ("approved_by", "approval_reference", "approved_at"))):
        raise CurrentUsageIntegrityError("current conservative approved plan binding differs")
    from .model_usage import CONSERVATIVE_DISPOSITION_AUTHORITY_SCHEMA_VERSION

    authority = {"schema_version": CONSERVATIVE_DISPOSITION_AUTHORITY_SCHEMA_VERSION,
        "approved_plan_digest": plan_digest, "approved_by": contract.approved_by,
        "approval_reference": contract.approval_reference, "approved_at": contract.approved_at,
        "invocation_id": allocation.invocation_id, "terminal_digest": terminal.terminal_digest,
        "allocation_digest": allocation.canonical_digest, "scope": contract.scope}
    if digest_canonical(authority) != record.get("authority_digest"):
        raise CurrentUsageIntegrityError("current conservative approved authority differs")


def refresh(connection: sqlite3.Connection, invocation_id: str, *, importing: bool = False) -> None:
    """Update one checked transition in its caller's existing write transaction.

    Settlement rows are validated by their creation/change API, not replayed here.
    Cash reservations and no-retry dispositions remain independently durable.
    """
    if not connection.in_transaction:
        raise CurrentUsageIntegrityError("current usage update requires a transaction")
    if not importing:
        require_ready(connection)
        count = _inventory(connection)
    else:
        count = connection.execute("SELECT COUNT(*) FROM model_usage_current").fetchone()[0]
    prior = connection.execute(f"SELECT {_SELECT} FROM model_usage_current WHERE invocation_id=?", (invocation_id,)).fetchone()
    if prior is not None:
        _checked(prior)
    row = connection.execute(
        "SELECT a.route,a.canonical_digest,a.cycle_id,t.terminal_digest,t.usage_status,"
        "json_extract(t.record_json,'$.policy_breach'),a.policy_digest FROM model_invocation_allocations a "
        "LEFT JOIN model_invocation_terminals t USING(invocation_id) WHERE a.invocation_id=?", (invocation_id,),
    ).fetchone()
    if row is None:
        raise CurrentUsageIntegrityError("current usage allocation is absent")
    active = row[3] is None
    unresolved = row[4] in {"UNREPORTED", "AMBIGUOUS", "INVALID"}
    breach = row[5] is not None
    reconciliations = [
        _bound_record(item[1], item[0], "reconciliation_digest", {"invocation_id": invocation_id, "usage_status": "REPORTED"})
        for item in connection.execute(
        "SELECT reconciliation_digest,record_json FROM model_usage_reconciliations WHERE invocation_id=?",
        (invocation_id,),
    )]
    for reconciliation in reconciliations:
        telemetry_digest = reconciliation.get("provider_telemetry_digest")
        telemetry = connection.execute(
            "SELECT telemetry_record_digest,record_json FROM model_provider_telemetry "
            "WHERE invocation_id=? AND provider_telemetry_digest=?", (invocation_id, telemetry_digest),
        ).fetchone()
        if telemetry is None:
            raise CurrentUsageIntegrityError("current reconciliation telemetry is absent")
        value = json.loads(telemetry[1])
        if (digest_canonical(value) != telemetry[0] or canonical_json_bytes(value).decode() != telemetry[1]
                or value.get("invocation_id") != invocation_id
                or value.get("provider_telemetry_digest") != telemetry_digest
                or digest_canonical(value.get("provider_telemetry")) != telemetry_digest):
            raise CurrentUsageIntegrityError("current reconciliation telemetry binding differs")
    disposition = connection.execute(
        "SELECT disposition_digest,terminal_digest,allocation_digest,policy_digest,usage_status,record_json,"
        "approved_plan_digest,authority_digest,approved_by,approval_reference,approved_at,observed_at "
        "FROM model_usage_conservative_dispositions WHERE invocation_id=?", (invocation_id,),
    ).fetchone()
    expected = {"invocation_id": invocation_id, "allocation_digest": row[1], "terminal_digest": row[3], "policy_digest": row[6]}
    if reconciliations or disposition is not None:
        from .model_usage import _retained_terminal_allocation, _policy_for_allocation, UsageComponents
        allocation, terminal = _retained_terminal_allocation(connection, invocation_id)
        policy = _policy_for_allocation(connection, allocation)
        for reconciliation in reconciliations:
            components = reconciliation.get("components", {})
            if (not isinstance(components, dict) or components.get("provenance") != "PROVIDER_REPORTED"
                    or type(components.get("total_tokens")) is not int):
                raise CurrentUsageIntegrityError("current reconciliation components differ")
            counters = ("input_tokens", "output_tokens", "cached_read_tokens", "cached_write_tokens",
                        "reasoning_tokens", "context_tokens", "total_tokens")
            if any(components.get(key) is not None and (type(components[key]) is not int or components[key] < 0)
                   for key in counters):
                raise CurrentUsageIntegrityError("current reconciliation counters differ")
            total = components["total_tokens"]
            context = components.get("context_tokens")
            output = components.get("output_tokens")
            expected_breach = ("MAX_TOTAL_TOKENS_EXCEEDED" if total > policy.max_total_tokens
                else "MAX_CONTEXT_TOKENS_EXCEEDED" if context is not None and context > policy.max_context_tokens
                else "MAX_OUTPUT_TOKENS_EXCEEDED" if policy.max_output_tokens is not None
                    and output is not None and output > policy.max_output_tokens else None)
            if reconciliation.get("policy_breach") != expected_breach:
                raise CurrentUsageIntegrityError("current reconciliation policy binding differs")
    if disposition is not None:
        if tuple(disposition[1:5]) != (row[3], row[1], row[6], "ESTIMATED"):
            raise CurrentUsageIntegrityError("current conservative settlement SQL binding differs")
        record = _bound_record(disposition[5], disposition[0], "disposition_digest", {**expected, "usage_status": "ESTIMATED",
            "exact_usage_remains_unknown": True, "unknown_spend_released": False})
        from .model_usage import (
            NATIVE_AUTONOMOUS_USAGE_SCOPE, NATIVE_EMBEDDING_TIMEOUT_USAGE_SCOPE,
            NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE, NATIVE_GRAPHITI_FALLBACK_CANCELLATION_USAGE_SCOPE,
        )
        scope = record.get("authority_scope")
        if scope is None:
            if tuple(disposition[6:11]) != tuple(record.get(key) for key in (
                "approved_plan_digest", "authority_digest", "approved_by", "approval_reference", "approved_at",
            )):
                raise CurrentUsageIntegrityError("current conservative approved plan SQL binding differs")
            _approved_conservative_plan(connection, record, allocation, terminal)
        elif scope not in {NATIVE_AUTONOMOUS_USAGE_SCOPE, NATIVE_EMBEDDING_TIMEOUT_USAGE_SCOPE,
                          NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE, NATIVE_GRAPHITI_FALLBACK_CANCELLATION_USAGE_SCOPE}:
            raise CurrentUsageIntegrityError("current conservative authority scope is unrecognised")
        else:
            from .model_usage import (_instant, _native_conservative_subscription_leaf,
                GraphitiLeafClass, UsageStatus, WorkloadClass)
            from newsroom.authority.canonical import validate_sha256_digest
            scope_digest = record.get("native_scope_digest")
            try:
                validate_sha256_digest(scope_digest)
                observed = _instant(str(record.get("observed_at")))
            except (TypeError, ValueError) as exc:
                raise CurrentUsageIntegrityError("current native approval value differs") from exc
            if (tuple(disposition[6:]) != (scope_digest, scope_digest, scope, scope,
                    record.get("observed_at"), record.get("observed_at"))
                    or record.get("authority_digest") != scope_digest
                    or observed < terminal.observed_at or not policy.qualified
                    or terminal.usage_status is not UsageStatus.UNREPORTED
                    or terminal.dispatch_at is None or terminal.pre_dispatch_zero_proved
                    or terminal.policy_breach is not None
                    or terminal.provider_telemetry_digest is not None
                    or terminal.raw_telemetry_pointer is not None
                    or terminal.components.total_tokens is not None):
                raise CurrentUsageIntegrityError("current native approval SQL binding differs")
            leaf = _native_conservative_subscription_leaf(allocation)
            if scope == NATIVE_AUTONOMOUS_USAGE_SCOPE:
                eligible = leaf is not None and terminal.subscription_cli_chat_not_cash_debited
            elif scope == NATIVE_GRAPHITI_FALLBACK_CANCELLATION_USAGE_SCOPE:
                eligible = (leaf is GraphitiLeafClass.FALLBACK
                    and terminal.outcome == "CANCELLED"
                    and terminal.failure_class == "MISSING_PROVIDER_TELEMETRY"
                    and terminal.subscription_cli_chat_not_cash_debited)
            else:
                graphiti = scope == NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE
                workload = WorkloadClass.GRAPHITI_EMBEDDING if graphiti else WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING
                eligible = (allocation.workload_class is workload
                    and allocation.route == workload.value and allocation.provider == "openrouter"
                    and allocation.model == "openai/text-embedding-3-large"
                    and terminal.outcome == ("CANCELLED" if graphiti else "NATIVE_EMBEDDING_FAILED")
                    and terminal.failure_class == ("MISSING_PROVIDER_TELEMETRY" if graphiti else "TimeoutError")
                    and not terminal.subscription_cli_chat_not_cash_debited)
            if not eligible:
                raise CurrentUsageIntegrityError("current native settlement scope target differs")
            expected_calculation = ("QUALIFIED_POLICY_MAX_TOTAL_TOKENS_CONSERVATIVE_UPPER_BOUND"
                if scope in {NATIVE_AUTONOMOUS_USAGE_SCOPE, NATIVE_GRAPHITI_FALLBACK_CANCELLATION_USAGE_SCOPE}
                else "MAX_QUALIFIED_POLICY_TOTAL_OR_EXACT_REQUEST_UTF8_BYTES")
            if record.get("estimate_calculation") != expected_calculation:
                raise CurrentUsageIntegrityError("current native estimate calculation differs")
        total = (max(policy.max_total_tokens, allocation.prompt_bytes)
                 if scope in {NATIVE_EMBEDDING_TIMEOUT_USAGE_SCOPE, NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE}
                 else policy.max_total_tokens)
        if (record.get("estimate_policy_digest") != row[6]
                or record.get("components") != UsageComponents(total_tokens=total, provenance="BOUNDED_ESTIMATE").as_record()
                or record.get("provider_dispatch_preserved") is not True):
            raise CurrentUsageIntegrityError("current conservative estimate binding differs")
    if reconciliations or disposition is not None:
        unresolved = False
    reported = connection.execute("SELECT disposition_digest,record_json FROM model_usage_reported_output_dispositions WHERE invocation_id=?", (invocation_id,)).fetchone()
    if reported is not None:
        from .model_usage import _retained_terminal_allocation
        _, terminal = _retained_terminal_allocation(connection, invocation_id)
        _bound_record(reported[1], reported[0], "disposition_digest", {**expected, "usage_status": "REPORTED",
            "retry_authorised": False, "components": terminal.components.as_record(),
            "policy_breach": terminal.policy_breach, "unknown_spend_released": False})
        if terminal.usage_status.value != "REPORTED":
            raise CurrentUsageIntegrityError("current reported settlement usage differs")
        breach = False
    if breach and _table(connection, "ledger"):
        for kind in ("NATIVE_ASSESSOR_INPUT_REQUALIFICATION", "NATIVE_ASSESSOR_OUTPUT_GUARD_REQUALIFICATION"):
            requalified_rows = connection.execute(
                "SELECT payload_digest,payload_json FROM ledger WHERE kind=? AND json_extract(payload_json,'$.invocation_id')=?",
                (kind, invocation_id),
            ).fetchall()
            if len(requalified_rows) > 1:
                raise CurrentUsageIntegrityError("current requalification settlement is duplicated")
            requalified = requalified_rows[0] if requalified_rows else None
            if requalified:
                raw = requalified[1]
                if digest_bytes(raw.encode()) != requalified[0]:
                    raise CurrentUsageIntegrityError("current requalification ledger binding differs")
                record = json.loads(raw)
                record = _bound_record(raw, record.get("requalification_digest"), "requalification_digest", {
                    "invocation_id": invocation_id, "allocation_digest": row[1], "terminal_digest": row[3],
                    "original_policy_digest": row[6], "original_candidate_retry": False})
                from .model_usage import _policy_from_record, _object
                qualified = connection.execute("SELECT record_json FROM model_invocation_policies WHERE canonical_digest=?",
                    (record.get("qualified_policy_digest"),)).fetchone()
                if qualified is None:
                    raise CurrentUsageIntegrityError("current requalification policy is absent")
                policy = _policy_from_record(_object(qualified[0]))
                if (not policy.qualified or policy.canonical_digest != record["qualified_policy_digest"]
                        or policy.evidence_digest != record.get("qualification_evidence_digest")):
                    raise CurrentUsageIntegrityError("current requalification policy binding differs")
                breach = False
                break
    if _table(connection, "issue_790_bounded_canary_consumptions") and _table(connection, "issue_790_bounded_canary_outcomes"):
        canary = connection.execute(
            "SELECT c.consumption_digest,c.record_json,o.outcome_digest,o.record_json FROM issue_790_bounded_canary_consumptions c "
            "JOIN issue_790_bounded_canary_outcomes o USING(consumption_digest) "
            "WHERE c.event_id=? AND json_extract(o.record_json,'$.result_class')!='TRUTHFUL_PROVIDER_SUCCESS' LIMIT 1",
            (row[2],),
        ).fetchone()
        if canary:
            consumption = _bound_record(canary[1], canary[0], "consumption_digest", {"event_id": row[2]})
            _bound_record(canary[3], canary[2], "outcome_digest", {
                "event_id": row[2], "consumption_digest": canary[0], "ledger_seq": consumption["ledger_seq"],
            })
            unresolved = breach = False
    breach = breach or any(item.get("policy_breach") is not None for item in reconciliations)
    if not (active or unresolved or breach):
        connection.execute("DELETE FROM model_usage_current WHERE invocation_id=?", (invocation_id,))
        if prior is not None:
            _set_inventory(connection, count - 1)
        return
    record = dict(zip(_FIELDS, (invocation_id, canonical_route(row[0]), row[1], row[3],
                              int(active), int(unresolved), int(breach)), strict=True))
    connection.execute(
        f"INSERT INTO model_usage_current({_SELECT}) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(invocation_id) DO UPDATE SET "
        "route=excluded.route,allocation_digest=excluded.allocation_digest,terminal_digest=excluded.terminal_digest,"
        "active=excluded.active,unresolved=excluded.unresolved,policy_breach=excluded.policy_breach,canonical_digest=excluded.canonical_digest",
        (*record.values(), digest_canonical(record)),
    )
    if prior is None:
        _set_inventory(connection, count + 1)


def import_legacy(connection: sqlite3.Connection) -> int:
    """Explicit quiescent cutover; never called by ordinary reads or boot."""
    if not connection.in_transaction:
        raise CurrentUsageIntegrityError("current usage import requires a transaction")
    if ready(connection):
        _inventory(connection)
        return 0
    if connection.execute("SELECT 1 FROM model_usage_current LIMIT 1").fetchone():
        raise CurrentUsageIntegrityError("current usage import requires an empty uninitialised inventory")
    _set_inventory(connection, 0)
    # Existing accepted settlement memberships are durable facts. Import does
    # not re-prove expired source bodies, provider diagnostics or LAND history.
    rows = connection.execute(
        "SELECT a.invocation_id FROM model_invocation_allocations a "
        "LEFT JOIN model_invocation_terminals t USING(invocation_id) WHERE t.invocation_id IS NULL "
        "OR t.usage_status IN ('UNREPORTED','AMBIGUOUS','INVALID') "
        "OR json_extract(t.record_json,'$.policy_breach') IS NOT NULL "
        "OR EXISTS(SELECT 1 FROM model_usage_reconciliations r WHERE r.invocation_id=a.invocation_id "
        "AND json_extract(r.record_json,'$.policy_breach') IS NOT NULL)"
    ).fetchall()
    for row in rows:
        terminal = connection.execute("SELECT 1 FROM model_invocation_terminals WHERE invocation_id=?", (row[0],)).fetchone()
        if terminal is not None:
            from .model_usage import _retained_terminal_allocation
            _retained_terminal_allocation(connection, row[0])
        else:
            _active_allocation(connection, row[0])
        refresh(connection, row[0], importing=True)
    _mark_ready(connection)
    return connection.execute("SELECT COUNT(*) FROM model_usage_current").fetchone()[0]
