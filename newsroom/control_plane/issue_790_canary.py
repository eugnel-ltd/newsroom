"""Durable coordinator for the single owner-approved issue #790 canary."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from newsroom.authority.canonical import canonical_json_bytes, digest_canonical
from newsroom.control_plane.issue_790_contract import (
    issue_790_owner_activated_sequence,
)
from newsroom.control_plane.issue_790_step16_activation import (
    ISSUE_790_STEP16_CIRCUIT_RELEASE_POLICY_VERSION,
    ISSUE_790_STEP16_CIRCUIT_RELEASE_SCHEMA,
    ISSUE_790_STEP16_EVENT_CIRCUIT_POLICY,
    STEP16_ACTIVATION_TABLE_SQL,
    STEP16_CIRCUIT_RELEASE_TABLE_SQL,
    effective_issue_790_invocation_plan_digests,
    effective_issue_790_plan_contract,
    validate_step16_activation_receipt,
    validate_step16_circuit_release_receipt,
)
from newsroom.control_plane.sqlite_profile import apply_control_plane_sqlite_profile
from newsroom.control_plane.veto import assert_private_store
from newsroom.graphiti_adapter.cli_process import validated_timeout_diagnostics

CANARY_PREFLIGHT_SCHEMA = "newsroom.graphiti-fresh-event-preflight.v1"
ITERATIVE_CANARY_PREFLIGHT_SCHEMA = (
    "newsroom.issue-790.iterative-fresh-event-preflight.v2"
)
CANARY_CONSUMPTION_SCHEMA = "newsroom.issue-790.canary-consumption.v2"
CANARY_OUTCOME_SCHEMA = "newsroom.issue-790.canary-outcome.v2"
ITERATIVE_CANARY_OUTCOME_SCHEMA = "newsroom.issue-790.canary-outcome.v3"
RETRY_EXCLUSION_SCHEMA = "newsroom.issue-790.graphiti-retry-exclusion.v1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS issue_790_graphiti_retry_exclusions(
    exclusion_digest TEXT PRIMARY KEY,
    approved_plan_digest TEXT NOT NULL,
    disposition_digest TEXT NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    ledger_seq INTEGER NOT NULL UNIQUE CHECK(ledger_seq > 0),
    reason TEXT NOT NULL,
    excluded_at TEXT NOT NULL,
    record_json TEXT NOT NULL,
    FOREIGN KEY(disposition_digest)
        REFERENCES model_usage_conservative_dispositions(disposition_digest)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS issue_790_bounded_canary_consumptions(
    consumption_digest TEXT PRIMARY KEY,
    approved_plan_digest TEXT NOT NULL,
    disposition_digest TEXT NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    ledger_seq INTEGER NOT NULL UNIQUE CHECK(ledger_seq > 0),
    owner_id TEXT NOT NULL UNIQUE,
    consumed_at TEXT NOT NULL,
    record_json TEXT NOT NULL,
    FOREIGN KEY(disposition_digest)
        REFERENCES model_usage_conservative_dispositions(disposition_digest)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS issue_790_bounded_canary_outcomes(
    outcome_digest TEXT PRIMARY KEY,
    consumption_digest TEXT NOT NULL UNIQUE,
    event_id TEXT NOT NULL UNIQUE,
    ledger_seq INTEGER NOT NULL UNIQUE CHECK(ledger_seq > 0),
    completed_at TEXT NOT NULL,
    record_json TEXT NOT NULL,
    FOREIGN KEY(consumption_digest)
        REFERENCES issue_790_bounded_canary_consumptions(consumption_digest)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
""" + STEP16_ACTIVATION_TABLE_SQL + STEP16_CIRCUIT_RELEASE_TABLE_SQL


def _unique_index_columns(
    connection: sqlite3.Connection, *, table: str
) -> frozenset[tuple[str, ...]]:
    columns: set[tuple[str, ...]] = set()
    for index in connection.execute(f"PRAGMA index_list({table})"):
        if not index[2]:
            continue
        name = str(index[1]).replace('"', '""')
        columns.add(
            tuple(
                str(column[2])
                for column in connection.execute(f'PRAGMA index_info("{name}")')
            )
        )
    return frozenset(columns)


def _rebuild_canary_consumptions_table(
    connection: sqlite3.Connection, *, message: str
) -> None:
    """Drop UNIQUE(approved_plan_digest). Keep UNIQUE event_id, ledger_seq, owner_id."""

    connection.commit()
    connection.execute("PRAGMA foreign_keys=OFF")
    connection.execute("PRAGMA legacy_alter_table=ON")
    try:
        connection.executescript(
            """
            BEGIN IMMEDIATE;
            ALTER TABLE issue_790_bounded_canary_consumptions
                RENAME TO issue_790_bounded_canary_consumptions_legacy;
            CREATE TABLE issue_790_bounded_canary_consumptions(
                consumption_digest TEXT PRIMARY KEY,
                approved_plan_digest TEXT NOT NULL,
                disposition_digest TEXT NOT NULL,
                event_id TEXT NOT NULL UNIQUE,
                ledger_seq INTEGER NOT NULL UNIQUE CHECK(ledger_seq > 0),
                owner_id TEXT NOT NULL UNIQUE,
                consumed_at TEXT NOT NULL,
                record_json TEXT NOT NULL,
                FOREIGN KEY(disposition_digest)
                    REFERENCES model_usage_conservative_dispositions(
                        disposition_digest
                    ) ON UPDATE RESTRICT ON DELETE RESTRICT
            );
            INSERT INTO issue_790_bounded_canary_consumptions
                SELECT * FROM issue_790_bounded_canary_consumptions_legacy;
            DROP TABLE issue_790_bounded_canary_consumptions_legacy;
            COMMIT;
            """
        )
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.execute("PRAGMA legacy_alter_table=OFF")
        connection.execute("PRAGMA foreign_keys=ON")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise Issue790CanaryIntegrityError(message)


def _allow_reused_dispositions(connection: sqlite3.Connection) -> None:
    if ("disposition_digest",) not in _unique_index_columns(
        connection, table="issue_790_bounded_canary_consumptions"
    ):
        return
    _rebuild_canary_consumptions_table(
        connection, message="bounded canary disposition migration differs"
    )


class Issue790CanaryIntegrityError(ValueError):
    """The bounded authority, event or retained state is contradictory."""


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise Issue790CanaryIntegrityError("canary timestamp lacks a timezone")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _instant(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise Issue790CanaryIntegrityError(f"{field} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Issue790CanaryIntegrityError(f"{field} is invalid") from exc
    if parsed.tzinfo is None:
        raise Issue790CanaryIntegrityError(f"{field} lacks a timezone")
    return parsed.astimezone(UTC)


class RetryForbiddenSafetyError(ValueError):
    """Missing row, safety-field contradiction, claim/lease, or missing exclusion."""

    classification = "PREDISPATCH_BINDING_FAILURE"
    failure_code = "RETRY_FORBIDDEN_SAFETY_STATE"


@dataclass(frozen=True, slots=True)
class RetryForbiddenSafetyState:
    """Canonical retry-forbidden identity. ``available_at`` is not authority."""

    event_id: str
    ledger_seq: int
    state: str
    attempt_count: int
    last_failure_code: str | None
    provider_dispatched: bool

    @classmethod
    def from_mapping(cls, item: Mapping[str, object]) -> RetryForbiddenSafetyState:
        event_id = item.get("event_id")
        ledger_seq = item.get("ledger_seq")
        state = item.get("state")
        attempt = item.get("attempt_count")
        failure = item.get("last_failure_code")
        dispatched = item.get("provider_dispatched")
        if (
            not isinstance(event_id, str)
            or not event_id.startswith("sha256:")
            or isinstance(ledger_seq, bool)
            or not isinstance(ledger_seq, int)
            or ledger_seq <= 0
            or not isinstance(state, str)
            or not state
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt < 0
            or not (failure is None or (isinstance(failure, str) and failure))
            or not isinstance(dispatched, bool)
        ):
            raise RetryForbiddenSafetyError(
                "retry-forbidden safety state is malformed"
            )
        return cls(
            event_id=event_id,
            ledger_seq=ledger_seq,
            state=state,
            attempt_count=attempt,
            last_failure_code=failure,
            provider_dispatched=dispatched,
        )


def retry_forbidden_observation(
    item: Mapping[str, object],
) -> dict[str, object]:
    """Audit-only scheduler and lease fields. Never retry authority."""

    return {
        "available_at": item.get("available_at"),
        "claim_expires_at": item.get("claim_expires_at"),
        "claim_owner": item.get("claim_owner"),
        "event_id": item.get("event_id"),
        "ledger_seq": item.get("ledger_seq"),
    }


def live_retry_forbidden_row_is_unclaimed(item: Mapping[str, object]) -> bool:
    return (
        item.get("claim_owner") is None and item.get("claim_expires_at") is None
    )


def retry_forbidden_has_claim_columns(connection: sqlite3.Connection) -> bool:
    columns = {
        str(row[1])
        for row in connection.execute(
            "PRAGMA table_info(unpublished_graphiti_revision_events)"
        )
    }
    return {"claim_owner", "claim_expires_at"} <= columns


def retry_forbidden_live_select(*, has_claims: bool) -> str:
    extra = ",claim_owner,claim_expires_at" if has_claims else ""
    return (
        "SELECT event_id,ledger_seq,state,attempt_count,available_at,"
        f"last_failure_code,provider_dispatched{extra} "
        "FROM unpublished_graphiti_revision_events"
    )


def retry_forbidden_live_snapshot(
    row: Sequence[object], *, has_claims: bool
) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "attempt_count": int(row[3]),
        "available_at": str(row[4]),
        "event_id": str(row[0]),
        "last_failure_code": None if row[5] is None else str(row[5]),
        "ledger_seq": int(row[1]),
        "provider_dispatched": bool(row[6]),
        "state": str(row[2]),
    }
    if has_claims:
        snapshot["claim_owner"] = None if row[7] is None else str(row[7])
        snapshot["claim_expires_at"] = None if row[8] is None else str(row[8])
    return snapshot


def validate_retry_forbidden_safety_state(
    *,
    expected: Mapping[str, object],
    live: Mapping[str, object] | None,
    excluded: bool,
) -> RetryForbiddenSafetyState:
    """Shared preflight and live-apply retry-forbidden validator."""

    if live is None:
        raise RetryForbiddenSafetyError("retry-forbidden row is absent")
    expected_state = RetryForbiddenSafetyState.from_mapping(expected)
    live_state = RetryForbiddenSafetyState.from_mapping(live)
    if expected_state != live_state:
        raise RetryForbiddenSafetyError("retry-forbidden safety state differs")
    if not live_retry_forbidden_row_is_unclaimed(live):
        raise RetryForbiddenSafetyError("retry-forbidden claim/lease is present")
    if not excluded:
        raise RetryForbiddenSafetyError(
            "retry-forbidden durable exclusion is absent"
        )
    return live_state


evaluate_retry_forbidden_safety = validate_retry_forbidden_safety_state


def retry_forbidden_safety_states_match(
    expected: object,
    retained: object,
    *,
    excluded_seqs: frozenset[int] | None = None,
) -> bool:
    if not isinstance(expected, (list, tuple)) or not isinstance(
        retained, (list, tuple)
    ):
        return False
    if len(expected) != len(retained):
        return False
    live_by_seq: dict[int, Mapping[str, object]] = {}
    for row in retained:
        if not isinstance(row, Mapping):
            return False
        seq = row.get("ledger_seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq in live_by_seq:
            return False
        live_by_seq[seq] = row
    if len(live_by_seq) != len(expected):
        return False
    try:
        for item in expected:
            if not isinstance(item, Mapping):
                return False
            seq = int(item["ledger_seq"])
            excluded = True if excluded_seqs is None else seq in excluded_seqs
            validate_retry_forbidden_safety_state(
                expected=item,
                live=live_by_seq.get(seq),
                excluded=excluded,
            )
    except (KeyError, TypeError, ValueError, RetryForbiddenSafetyError):
        return False
    return True


def _retry_exclusion_records_match(
    prior: Mapping[str, object], record: Mapping[str, object]
) -> bool:
    prior_rest = {
        key: value
        for key, value in prior.items()
        if key not in {"excluded_at", "exclusion_digest", "event_snapshot"}
    }
    record_rest = {
        key: value
        for key, value in record.items()
        if key not in {"excluded_at", "exclusion_digest", "event_snapshot"}
    }
    return prior_rest == record_rest and retry_forbidden_safety_states_match(
        (prior.get("event_snapshot"),),
        (record.get("event_snapshot"),),
    )


def _token(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value.encode("utf-8")) > 1024
    ):
        raise Issue790CanaryIntegrityError(f"{field} is invalid")
    return value


def _object(value: object, *, field: str) -> dict[str, object]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise Issue790CanaryIntegrityError(f"{field} is malformed") from exc
    if not isinstance(value, dict):
        raise Issue790CanaryIntegrityError(f"{field} is malformed")
    return dict(value)


def _json(value: Mapping[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_addressed_record(
    value: object,
    *,
    digest_field: str,
    field: str,
) -> dict[str, object]:
    record = _object(value, field=field)
    without_digest = dict(record)
    supplied = without_digest.pop(digest_field, None)
    if supplied != digest_canonical(without_digest):
        raise Issue790CanaryIntegrityError(f"{field} digest differs")
    return record


def _validated_controller_timeout_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    report = dict(value)
    supplied_digest = report.pop("report_digest", None)
    if supplied_digest != digest_canonical(report):
        raise Issue790CanaryIntegrityError("canary causal report digest differs")
    if set(report) != {
        "schema_version",
        "classification",
        "causal_constraint",
        "local_cause",
        "provider_cause",
        "diagnostic_reference",
        "diagnostic",
    }:
        raise Issue790CanaryIntegrityError("canary causal report fields differ")
    try:
        diagnostic = validated_timeout_diagnostics([report.get("diagnostic")])[0]
    except ValueError as exc:
        raise Issue790CanaryIntegrityError(
            "canary causal timeout diagnostic differs"
        ) from exc
    if (
        report.get("schema_version") != "newsroom.issue-790.causal-report.v1"
        or report.get("classification") != "CONTROLLER_TIMEOUT"
        or report.get("causal_constraint") != "CONTROLLER_TIMEOUT_MS"
        or report.get("local_cause") != diagnostic.get("cause")
        or report.get("provider_cause") != diagnostic.get("provider_cause")
        or diagnostic.get("boundary") != "CONTROLLER_DEADLINE"
        or diagnostic.get("phase") != "PRIMARY_TRANSPORT"
        or diagnostic.get("cause") != "CONFIGURED_TIMEOUT_EXPIRED"
        or diagnostic.get("provider_cause") != "UNOBSERVED"
        or diagnostic.get("process") != "CLI_CHILD"
        or int(diagnostic["elapsed_ms"])
        < int(diagnostic["configured_timeout_ms"])
    ):
        raise Issue790CanaryIntegrityError("canary causal report differs")
    _token(report.get("diagnostic_reference"), field="canary diagnostic reference")
    return {**report, "report_digest": supplied_digest}


def _bound_row_record(
    row: sqlite3.Row,
    *,
    digest_field: str,
    identity_fields: tuple[str, ...],
    field: str,
) -> dict[str, object]:
    record = _content_addressed_record(
        row["record_json"],
        digest_field=digest_field,
        field=field,
    )
    if any(record.get(name) != row[name] for name in identity_fields):
        raise Issue790CanaryIntegrityError(f"{field} SQL identity differs")
    return record


_CIRCUIT_RELEASE_SELECT = (
    "SELECT release_digest,activation_digest,plan_digest,event_id,"
    "ledger_seq,released_at,record_json "
    "FROM issue_790_step16_circuit_releases "
)


def _validated_circuit_release_row(row: sqlite3.Row) -> dict[str, object]:
    try:
        record = json.loads(str(row[6]))
    except json.JSONDecodeError as exc:
        raise Issue790CanaryIntegrityError(
            "issue #790 event circuit release differs"
        ) from exc
    if not isinstance(record, dict):
        raise Issue790CanaryIntegrityError(
            "issue #790 event circuit release differs"
        )
    try:
        return validate_step16_circuit_release_receipt(
            record,
            sql_identity={
                "release_digest": str(row[0]),
                "activation_digest": str(row[1]),
                "plan_digest": str(row[2]),
                "event_id": str(row[3]),
                "ledger_seq": int(row[4]),
                "released_at": str(row[5]),
            },
        )
    except Exception as exc:
        raise Issue790CanaryIntegrityError(str(exc)) from exc


def _validated_nested_circuit_release(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    try:
        return validate_step16_circuit_release_receipt(
            _object(value, field="circuit release")
        )
    except Issue790CanaryIntegrityError:
        raise
    except Exception as exc:
        raise Issue790CanaryIntegrityError(str(exc)) from exc


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        is not None
    )


def graphiti_excluded_event_ids(connection: sqlite3.Connection) -> frozenset[str]:
    """Return exact events which a generic Graphiti worker must never claim."""

    from newsroom.control_plane.graphiti_event_reconciliation import (
        retained_graphiti_event_holds,
    )

    values: set[str] = set()
    if _table_exists(connection, "issue_790_graphiti_retry_exclusions"):
        values.update(
            str(row[0])
            for row in connection.execute(
                "SELECT event_id FROM issue_790_graphiti_retry_exclusions"
            )
        )
    if _table_exists(connection, "issue_790_bounded_canary_consumptions"):
        values.update(
            str(row[0])
            for row in connection.execute(
                "SELECT event_id FROM issue_790_bounded_canary_consumptions"
            )
        )
    values.update(retained_graphiti_event_holds(connection))
    return frozenset(values)


def graphiti_retry_excluded(
    connection: sqlite3.Connection,
    *,
    event_id: str,
) -> bool:
    from newsroom.control_plane.graphiti_event_reconciliation import (
        retained_graphiti_event_holds,
    )

    return bool(
        _table_exists(connection, "issue_790_graphiti_retry_exclusions")
        and connection.execute(
            "SELECT 1 FROM issue_790_graphiti_retry_exclusions WHERE event_id=?",
            (event_id,),
        ).fetchone()
        is not None
    ) or event_id in retained_graphiti_event_holds(connection)


def validate_graphiti_canary_claim(
    connection: sqlite3.Connection,
    *,
    consumption_digest: str,
    event_id: str,
    owner_id: str,
) -> None:
    if not _table_exists(connection, "issue_790_bounded_canary_consumptions"):
        raise ValueError("bounded canary claim authority differs")
    row = connection.execute(
        "SELECT consumption_digest,approved_plan_digest,disposition_digest,"
        "event_id,ledger_seq,owner_id,consumed_at,record_json "
        "FROM issue_790_bounded_canary_consumptions "
        "WHERE consumption_digest=? AND event_id=? AND owner_id=?",
        (consumption_digest, event_id, owner_id),
    ).fetchone()
    if row is None:
        raise ValueError("bounded canary claim authority differs")
    identity_fields = (
        "consumption_digest",
        "approved_plan_digest",
        "disposition_digest",
        "event_id",
        "ledger_seq",
        "owner_id",
        "consumed_at",
    )
    record = _content_addressed_record(
        row[7],
        digest_field="consumption_digest",
        field="canary consumption",
    )
    if tuple(record.get(name) for name in identity_fields) != tuple(row[:7]):
        raise Issue790CanaryIntegrityError(
            "canary consumption SQL identity differs"
        )


def graphiti_event_has_canary_consumption(
    connection: sqlite3.Connection,
    *,
    event_id: str,
) -> bool:
    return bool(
        _table_exists(connection, "issue_790_bounded_canary_consumptions")
        and connection.execute(
            "SELECT 1 FROM issue_790_bounded_canary_consumptions WHERE event_id=?",
            (event_id,),
        ).fetchone()
        is not None
    )


def unmatched_bounded_canary_consumption(
    connection: sqlite3.Connection,
    *,
    approved_plan_digest: str | None = None,
) -> tuple[str, int] | None:
    """Return the unmatched consumption identity, if any.

    Unmatched means a consumption row with no outcome. A circuit_release does
    not substitute for an outcome; consume-once stays keyed on that pair.
    ``approved_plan_digest`` narrows to one bound plan when known. Preflight
    may omit it because the pending-family digest is not the activated digest.
    """

    if not _table_exists(connection, "issue_790_bounded_canary_consumptions"):
        return None
    clauses = []
    params: list[object] = []
    if approved_plan_digest is not None:
        clauses.append("consumed.approved_plan_digest=?")
        params.append(approved_plan_digest)
    if _table_exists(connection, "issue_790_bounded_canary_outcomes"):
        clauses.append(
            "NOT EXISTS ("
            "SELECT 1 FROM issue_790_bounded_canary_outcomes AS outcome "
            "WHERE outcome.consumption_digest=consumed.consumption_digest)"
        )
    where = " AND ".join(clauses)
    sql = (
        "SELECT consumed.event_id, consumed.ledger_seq "
        "FROM issue_790_bounded_canary_consumptions AS consumed"
    )
    if where:
        sql = f"{sql} WHERE {where}"
    row = connection.execute(f"{sql} LIMIT 1", params).fetchone()
    if row is None:
        return None
    return str(row[0]), int(row[1])


def validate_graphiti_canary_target_unused(
    connection: sqlite3.Connection,
    *,
    event_id: str,
    ingest_ids: tuple[str, ...],
) -> None:
    """Reject a canary target with consumption or execution evidence."""

    if graphiti_event_has_canary_consumption(connection, event_id=event_id):
        raise Issue790CanaryIntegrityError(
            "bounded canary target is already consumed"
        )
    if not ingest_ids:
        raise Issue790CanaryIntegrityError(
            "bounded canary target has no resolved ingest identities"
        )
    placeholders = ",".join("?" for _ in ingest_ids)
    prior_tables = (
        "unpublished_graphiti_ingest",
        "unpublished_graphiti_failures",
        "unpublished_graphiti_receipts",
        "unpublished_graphiti_attempt_receipts",
        "unpublished_graphiti_spend",
    )
    prior_ingest_evidence = any(
        _table_exists(connection, table)
        and connection.execute(
            f"SELECT 1 FROM {table} WHERE ingest_id IN ({placeholders}) LIMIT 1",
            ingest_ids,
        ).fetchone()
        is not None
        for table in prior_tables
    )
    if prior_ingest_evidence or (
        _table_exists(connection, "model_work_envelopes")
        and connection.execute(
            f"SELECT 1 FROM model_work_envelopes WHERE cycle_id=? "
            f"OR json_extract(record_json,'$.ingest_id') "
            f"IN ({placeholders}) LIMIT 1",
            (event_id, *ingest_ids),
        ).fetchone()
        is not None
    ):
        raise Issue790CanaryIntegrityError(
            "bounded canary target has prior execution evidence"
        )


def _stable_unit_ref(value: Mapping[str, object]) -> tuple[object, ...]:
    return (
        value.get("ingest_id"),
        value.get("revision_id"),
        value.get("representation_digest"),
        value.get("chunk_digest"),
        value.get("chunk_ordinal"),
        value.get("predecessor_ingest_id"),
    )


def _require_effective_plan_contract(
    plan_digest: str,
    connection: sqlite3.Connection,
    *,
    message: str,
):
    try:
        return effective_issue_790_plan_contract(
            plan_digest,
            connection=connection,
        )
    except KeyError as exc:
        raise Issue790CanaryIntegrityError(message) from exc


def _validated_preflight(
    value: Mapping[str, object],
    *,
    event_id: str,
    ledger_seq: int,
    manifest_digest: str,
    consumed_at: datetime,
    approved_plan_digest: str,
    connection: sqlite3.Connection,
) -> dict[str, object]:
    retained = dict(value)
    supplied_digest = retained.pop("evidence_digest", None)
    if supplied_digest != digest_canonical(retained):
        raise Issue790CanaryIntegrityError("bounded canary preflight digest differs")
    approved_contract = _require_effective_plan_contract(
        approved_plan_digest,
        connection,
        message="bounded canary preflight plan differs",
    )
    iterative = approved_contract.sequence_ordinal > 0
    expected_schema = (
        ITERATIVE_CANARY_PREFLIGHT_SCHEMA
        if iterative
        else CANARY_PREFLIGHT_SCHEMA
    )
    if (
        retained.get("schema_version") != expected_schema
        or retained.get("event_id") != event_id
        or retained.get("ledger_seq") != ledger_seq
        or retained.get("event_state") != "QUEUED"
        or retained.get("event_attempt_count") != 0
        or retained.get("event_manifest_digest") != manifest_digest
        or retained.get("owner_emergency_stop_clear") is not True
        or retained.get("provider_calls") != 0
        or retained.get("store_mutations") != 0
    ):
        raise Issue790CanaryIntegrityError("bounded canary preflight differs")
    if iterative and (
        retained.get("approved_plan_digest") != approved_plan_digest
        or retained.get("fallback_mode")
        != "DISABLED_BEFORE_PROVIDER_DISPATCH"
        or retained.get("fixed_constraints_digest")
        != approved_contract.fixed_constraints_digest
    ):
        raise Issue790CanaryIntegrityError(
            "bounded canary iterative preflight differs"
        )
    if approved_contract.sequence_ordinal >= 22 and any(
        not isinstance(retained.get(field), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(retained.get(field))) is None
        for field in (
            "pre_operation_snapshot_digest",
            "prepared_canary_decision_digest",
            "prepared_canary_record_digest",
        )
    ):
        raise Issue790CanaryIntegrityError(
            "bounded canary prepared artefact binding differs"
        )
    evaluated_at = _instant(retained.get("evaluated_at"), field="preflight time")
    if evaluated_at > consumed_at.astimezone(UTC):
        raise Issue790CanaryIntegrityError("bounded canary preflight follows consumption")
    resolved = retained.get("resolved_units")
    decisions = retained.get("rights_decision_digests")
    if (
        not isinstance(resolved, list)
        or not resolved
        or not all(isinstance(item, dict) for item in resolved)
        or not isinstance(decisions, list)
        or len(decisions) != len(resolved)
        or not all(
            isinstance(item, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", item)
            for item in decisions
        )
    ):
        raise Issue790CanaryIntegrityError("bounded canary preflight units differ")
    ingest_ids: list[str] = []
    ordinals: list[int] = []
    for raw in resolved:
        item = dict(raw)
        if set(item) != {
            "ingest_id",
            "revision_id",
            "representation_digest",
            "chunk_digest",
            "chunk_ordinal",
            "predecessor_ingest_id",
        }:
            raise Issue790CanaryIntegrityError(
                "bounded canary preflight unit fields differ"
            )
        ingest_ids.append(_token(item["ingest_id"], field="preflight ingest id"))
        ordinal = item["chunk_ordinal"]
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal <= 0:
            raise Issue790CanaryIntegrityError(
                "bounded canary preflight chunk ordinal is invalid"
            )
        ordinals.append(ordinal)
        for field in ("revision_id", "representation_digest", "chunk_digest"):
            _token(item[field], field=f"preflight {field}")
        predecessor = item["predecessor_ingest_id"]
        if predecessor is not None:
            _token(predecessor, field="preflight predecessor ingest id")
    if len(set(ingest_ids)) != len(ingest_ids) or ordinals != list(
        range(1, len(ordinals) + 1)
    ):
        raise Issue790CanaryIntegrityError("bounded canary preflight order differs")
    return {**retained, "evidence_digest": supplied_digest}


class Issue790CanaryRepository:
    """Own the #790 exclusion, consumption and zero-I/O finalisation state."""

    def __init__(self, path: str) -> None:
        assert_private_store(path)
        self.path = path
        connection = self._connection()
        try:
            _allow_reused_dispositions(connection)
            connection.executescript(_SCHEMA)
            connection.commit()
        finally:
            connection.close()

    @classmethod
    def open_existing(cls, path: str) -> Issue790CanaryRepository:
        """Open an already-installed coordinator without a write-capable setup."""

        assert_private_store(path)
        instance = cls.__new__(cls)
        instance.path = path
        connection = instance._read_connection()
        try:
            required = {
                "issue_790_graphiti_retry_exclusions",
                "issue_790_bounded_canary_consumptions",
                "issue_790_bounded_canary_outcomes",
            }
            installed = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if not required.issubset(installed):
                raise Issue790CanaryIntegrityError(
                    "bounded canary coordinator schema is absent"
                )
        finally:
            connection.close()
        return instance

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        apply_control_plane_sqlite_profile(connection)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _read_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"{Path(self.path).absolute().as_uri()}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        apply_control_plane_sqlite_profile(connection, query_only=True)
        return connection

    def retain_step16_activation(
        self,
        record: Mapping[str, object],
    ) -> dict[str, object]:
        try:
            retained = validate_step16_activation_receipt(record)
        except Exception as exc:
            raise Issue790CanaryIntegrityError(str(exc)) from exc
        digest = retained.get("activation_digest")
        checked = str(retained.get("checked_candidate_digest"))
        plan_digest = str(retained.get("plan_digest"))
        comment_id = retained.get("comment_id")
        payload_digest = str(retained.get("canonical_approval_payload_digest"))
        created_at = str(retained.get("created_at"))
        encoded = canonical_json_bytes(retained).decode("utf-8")
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT record_json FROM issue_790_step16_activations "
                "WHERE activation_digest=? OR checked_candidate_digest=? "
                "OR plan_digest=? OR comment_id=?",
                (digest, checked, plan_digest, comment_id),
            ).fetchall()
            if rows:
                for row in rows:
                    try:
                        prior = json.loads(str(row[0]))
                    except json.JSONDecodeError as exc:
                        raise Issue790CanaryIntegrityError(
                            "step 16 activation contradicts retained evidence"
                        ) from exc
                    if prior != retained:
                        raise Issue790CanaryIntegrityError(
                            "step 16 activation contradicts retained evidence"
                        )
                connection.commit()
                return retained
            connection.execute(
                "INSERT INTO issue_790_step16_activations("
                "activation_digest,checked_candidate_digest,plan_digest,comment_id,"
                "payload_digest,created_at,record_json) VALUES (?,?,?,?,?,?,?)",
                (
                    digest,
                    checked,
                    plan_digest,
                    comment_id,
                    payload_digest,
                    created_at,
                    encoded,
                ),
            )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
        return retained

    def existing_step16_circuit_release(
        self,
        *,
        plan_digest: str,
        event_id: str,
        ledger_seq: int,
    ) -> dict[str, object] | None:
        connection = self._read_connection()
        try:
            if not _table_exists(connection, "issue_790_step16_circuit_releases"):
                return None
            row = connection.execute(
                _CIRCUIT_RELEASE_SELECT
                + "WHERE plan_digest=? AND event_id=? AND ledger_seq=?",
                (plan_digest, event_id, ledger_seq),
            ).fetchone()
            if row is None:
                return None
            return _validated_circuit_release_row(row)
        finally:
            connection.close()

    def release_step16_expired_open_circuit(
        self,
        *,
        plan_digest: str,
        activation_digest: str,
        event_id: str,
        ledger_seq: int,
        prior_state: Mapping[str, object],
        observed_at: datetime,
        policy: str,
    ) -> dict[str, object]:
        plan_digest = _token(plan_digest, field="plan digest")
        activation_digest = _token(activation_digest, field="activation digest")
        event_id = _token(event_id, field="canary event id")
        if isinstance(ledger_seq, bool) or not isinstance(ledger_seq, int) or ledger_seq <= 0:
            raise Issue790CanaryIntegrityError("bounded canary ledger sequence is invalid")
        if policy != ISSUE_790_STEP16_EVENT_CIRCUIT_POLICY:
            raise Issue790CanaryIntegrityError("issue #790 event circuit policy differs")
        opened_at = prior_state.get("opened_at")
        available_at = prior_state.get("available_at")
        failure_code = prior_state.get("failure_code")
        if (
            prior_state.get("state") != "OPEN"
            or not isinstance(opened_at, str)
            or not isinstance(available_at, str)
            or not isinstance(failure_code, str)
            or not failure_code
            or failure_code != failure_code.strip()
        ):
            raise Issue790CanaryIntegrityError("issue #790 event circuit is malformed")
        opened = _instant(opened_at, field="circuit opened_at")
        available = _instant(available_at, field="circuit available_at")
        observed = observed_at.astimezone(UTC)
        if opened > available or available > observed:
            raise Issue790CanaryIntegrityError("issue #790 event circuit is malformed")
        released_at = _utc_text(observed)
        unsigned = {
            "schema_version": ISSUE_790_STEP16_CIRCUIT_RELEASE_SCHEMA,
            "policy_version": ISSUE_790_STEP16_CIRCUIT_RELEASE_POLICY_VERSION,
            "plan_digest": plan_digest,
            "activation_digest": activation_digest,
            "event_id": event_id,
            "ledger_seq": ledger_seq,
            "prior_state": {
                "state": "OPEN",
                "opened_at": opened_at,
                "available_at": available_at,
                "failure_code": failure_code,
            },
            "released_at": released_at,
            "effect": "IMMEDIATE_CLOSE_EXPIRED_OPEN",
            "provider_calls": 0,
            "cas_result": {
                "singleton": 1,
                "state": "OPEN",
                "opened_at": opened_at,
                "available_at": available_at,
                "failure_code": failure_code,
                "rowcount": 1,
            },
        }
        receipt = {**unsigned, "release_digest": digest_canonical(unsigned)}
        try:
            receipt = validate_step16_circuit_release_receipt(receipt)
        except Exception as exc:
            raise Issue790CanaryIntegrityError(str(exc)) from exc
        encoded = canonical_json_bytes(receipt).decode("utf-8")
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(STEP16_CIRCUIT_RELEASE_TABLE_SQL)
            prior = connection.execute(
                _CIRCUIT_RELEASE_SELECT
                + "WHERE plan_digest=? AND event_id=? AND ledger_seq=?",
                (plan_digest, event_id, ledger_seq),
            ).fetchone()
            if prior is not None:
                retained = _validated_circuit_release_row(prior)
                if (
                    retained.get("plan_digest") != plan_digest
                    or retained.get("activation_digest") != activation_digest
                    or retained.get("event_id") != event_id
                    or retained.get("ledger_seq") != ledger_seq
                    or retained.get("prior_state") != unsigned["prior_state"]
                    or retained.get("effect") != "IMMEDIATE_CLOSE_EXPIRED_OPEN"
                    or retained.get("provider_calls") != 0
                    or retained.get("schema_version")
                    != ISSUE_790_STEP16_CIRCUIT_RELEASE_SCHEMA
                ):
                    raise Issue790CanaryIntegrityError(
                        "issue #790 event circuit release differs"
                    )
                connection.commit()
                return retained
            cursor = connection.execute(
                "UPDATE unpublished_graphiti_event_circuit SET state='CLOSED',"
                "opened_at=NULL,available_at=NULL,failure_code=NULL "
                "WHERE singleton=1 AND state='OPEN' AND opened_at IS ? "
                "AND available_at IS ? AND failure_code IS ?",
                (opened_at, available_at, failure_code),
            )
            if cursor.rowcount != 1:
                raise Issue790CanaryIntegrityError(
                    "issue #790 event circuit release differs"
                )
            connection.execute(
                "INSERT INTO issue_790_step16_circuit_releases("
                "release_digest,activation_digest,plan_digest,event_id,ledger_seq,"
                "released_at,record_json) VALUES (?,?,?,?,?,?,?)",
                (
                    receipt["release_digest"],
                    activation_digest,
                    plan_digest,
                    event_id,
                    ledger_seq,
                    released_at,
                    encoded,
                ),
            )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
        return receipt

    def retain_retry_exclusions(
        self,
        *,
        approved_plan_digest: str,
        disposition_digest: str,
        events: Sequence[Mapping[str, object]],
        excluded_at: datetime,
    ) -> tuple[dict[str, object], ...]:
        disposition_digest = _token(
            disposition_digest, field="retry exclusion disposition digest"
        )
        excluded_at_text = _utc_text(excluded_at)
        expected = tuple(events)
        seqs = tuple(int(item.get("ledger_seq", 0)) for item in expected)
        extra = seqs[2:]
        if (
            seqs[:2] != (1932, 1972)
            or any(
                seq not in {
                    8834, 8835, 13284, 13337, 13361, 13362,
                    13665, 13671, 13677, 13683, 13689, 13690, 13696,
                }
                for seq in extra
            )
            or len(set(seqs)) != len(seqs)
        ):
            raise Issue790CanaryIntegrityError("retry exclusion targets differ")
        connection = self._connection()
        retained: list[dict[str, object]] = []
        try:
            connection.execute("BEGIN IMMEDIATE")
            approved_contract = _require_effective_plan_contract(
                approved_plan_digest,
                connection,
                message="retry exclusion plan differs",
            )
            disposition = connection.execute(
                "SELECT invocation_id FROM model_usage_conservative_dispositions "
                "WHERE disposition_digest=? AND approved_plan_digest=?",
                (disposition_digest, approved_plan_digest),
            ).fetchone()
            if (
                disposition is None
                or str(disposition[0]) != approved_contract.invocation_id
            ):
                raise Issue790CanaryIntegrityError("retry exclusion authority differs")
            has_claims = retry_forbidden_has_claim_columns(connection)
            live_select = retry_forbidden_live_select(has_claims=has_claims)
            for item in expected:
                event_id = _token(item.get("event_id"), field="retry exclusion event id")
                ledger_seq = int(item["ledger_seq"])
                row = connection.execute(
                    live_select + " WHERE event_id=? AND ledger_seq=?",
                    (event_id, ledger_seq),
                ).fetchone()
                snapshot = (
                    retry_forbidden_live_snapshot(row, has_claims=has_claims)
                    if row is not None
                    else None
                )
                if snapshot is None or not retry_forbidden_safety_states_match(
                    (item,), (snapshot,)
                ):
                    raise Issue790CanaryIntegrityError(
                        "retry exclusion event state differs"
                    )
                without_digest: dict[str, object] = {
                    "schema_version": RETRY_EXCLUSION_SCHEMA,
                    "approved_plan_digest": approved_plan_digest,
                    "disposition_digest": disposition_digest,
                    "event_id": event_id,
                    "ledger_seq": ledger_seq,
                    "reason": "ISSUE_790_RETRY_FORBIDDEN",
                    "event_snapshot": snapshot,
                    "excluded_at": excluded_at_text,
                }
                exclusion_digest = digest_canonical(without_digest)
                record = {**without_digest, "exclusion_digest": exclusion_digest}
                prior = connection.execute(
                    "SELECT * FROM issue_790_graphiti_retry_exclusions "
                    "WHERE event_id=?",
                    (event_id,),
                ).fetchone()
                if prior is not None:
                    prior_record = _bound_row_record(
                        prior,
                        digest_field="exclusion_digest",
                        identity_fields=(
                            "exclusion_digest",
                            "approved_plan_digest",
                            "disposition_digest",
                            "event_id",
                            "ledger_seq",
                            "reason",
                            "excluded_at",
                        ),
                        field="retry exclusion",
                    )
                    if not _retry_exclusion_records_match(prior_record, record):
                        raise Issue790CanaryIntegrityError(
                            "conflicting retry exclusion replay"
                        )
                    retained.append(prior_record)
                    continue
                connection.execute(
                    "INSERT INTO issue_790_graphiti_retry_exclusions("
                    "exclusion_digest,approved_plan_digest,disposition_digest,"
                    "event_id,ledger_seq,reason,excluded_at,record_json) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        exclusion_digest,
                        approved_plan_digest,
                        disposition_digest,
                        event_id,
                        ledger_seq,
                        "ISSUE_790_RETRY_FORBIDDEN",
                        excluded_at_text,
                        _json(record),
                    ),
                )
                retained.append(record)
            connection.commit()
            return tuple(retained)
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def retry_exclusions(self) -> tuple[dict[str, object], ...]:
        connection = self._read_connection()
        try:
            return tuple(
                _bound_row_record(
                    row,
                    digest_field="exclusion_digest",
                    identity_fields=(
                        "exclusion_digest",
                        "approved_plan_digest",
                        "disposition_digest",
                        "event_id",
                        "ledger_seq",
                        "reason",
                        "excluded_at",
                    ),
                    field="retry exclusion",
                )
                for row in connection.execute(
                    "SELECT * FROM issue_790_graphiti_retry_exclusions "
                    "ORDER BY ledger_seq"
                )
            )
        finally:
            connection.close()

    def existing_consumption(
        self,
        *,
        approved_plan_digest: str,
        event_id: str | None = None,
        ledger_seq: int | None = None,
    ) -> dict[str, object] | None:
        if (event_id is None) != (ledger_seq is None):
            raise Issue790CanaryIntegrityError(
                "bounded canary consumption identity is incomplete"
            )
        connection = self._read_connection()
        try:
            if event_id is None:
                row = connection.execute(
                    "SELECT * FROM issue_790_bounded_canary_consumptions "
                    "WHERE approved_plan_digest=? "
                    "ORDER BY ledger_seq ASC LIMIT 1",
                    (approved_plan_digest,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM issue_790_bounded_canary_consumptions "
                    "WHERE approved_plan_digest=? AND event_id=? AND ledger_seq=?",
                    (approved_plan_digest, event_id, ledger_seq),
                ).fetchone()
            return (
                None
                if row is None
                else _bound_row_record(
                    row,
                    digest_field="consumption_digest",
                    identity_fields=(
                        "consumption_digest",
                        "approved_plan_digest",
                        "disposition_digest",
                        "event_id",
                        "ledger_seq",
                        "owner_id",
                        "consumed_at",
                    ),
                    field="canary consumption",
                )
            )
        finally:
            connection.close()

    def existing_outcome(self, *, consumption_digest: str) -> dict[str, object] | None:
        connection = self._read_connection()
        try:
            row = connection.execute(
                "SELECT * FROM issue_790_bounded_canary_outcomes "
                "WHERE consumption_digest=?",
                (consumption_digest,),
            ).fetchone()
            if row is None:
                return None
            record = _bound_row_record(
                row,
                digest_field="outcome_digest",
                identity_fields=(
                    "outcome_digest",
                    "consumption_digest",
                    "event_id",
                    "ledger_seq",
                    "completed_at",
                ),
                field="canary outcome",
            )
            if "circuit_release" in record:
                _validated_nested_circuit_release(record.get("circuit_release"))
            return record
        finally:
            connection.close()

    def disposition_invocation(
        self,
        *,
        approved_plan_digest: str,
        disposition_digest: str,
    ) -> str | None:
        """Return the invocation bound by one exact retained disposition."""

        connection = self._read_connection()
        try:
            row = connection.execute(
                "SELECT invocation_id FROM model_usage_conservative_dispositions "
                "WHERE approved_plan_digest=? AND disposition_digest=?",
                (approved_plan_digest, disposition_digest),
            ).fetchone()
            return None if row is None else str(row[0])
        finally:
            connection.close()

    def invocation_terminal(
        self,
        *,
        invocation_id: str,
    ) -> dict[str, object] | None:
        """Return one content-addressed model terminal after SQL identity checks."""

        invocation_id = _token(invocation_id, field="model terminal invocation id")
        connection = self._read_connection()
        try:
            row = connection.execute(
                "SELECT terminal_digest,invocation_id,usage_status,outcome,"
                "failure_class,completed_at,record_json "
                "FROM model_invocation_terminals WHERE invocation_id=?",
                (invocation_id,),
            ).fetchone()
            if row is None:
                return None
            record = _object(row[6], field="model invocation terminal")
            digest = record.get("terminal_digest")
            unsigned = dict(record)
            unsigned["terminal_digest"] = ""
            if (
                not isinstance(digest, str)
                or digest != row[0]
                or digest_canonical(unsigned) != digest
                or record.get("invocation_id") != row[1]
                or record.get("usage_status") != row[2]
                or record.get("outcome") != row[3]
                or record.get("failure_class") != row[4]
                or record.get("completed_at") != row[5]
            ):
                raise Issue790CanaryIntegrityError(
                    "model invocation terminal identity differs"
                )
            return record
        finally:
            connection.close()

    def preflight_for_consumption(
        self,
        *,
        consumption_digest: str,
        event_id: str,
        owner_id: str,
    ) -> dict[str, object]:
        connection = self._read_connection()
        try:
            row = connection.execute(
                "SELECT * FROM issue_790_bounded_canary_consumptions "
                "WHERE consumption_digest=? AND event_id=? AND owner_id=?",
                (consumption_digest, event_id, owner_id),
            ).fetchone()
            if row is None:
                raise Issue790CanaryIntegrityError(
                    "bounded canary consumption authority differs"
                )
            record = _bound_row_record(
                row,
                digest_field="consumption_digest",
                identity_fields=(
                    "consumption_digest",
                    "approved_plan_digest",
                    "disposition_digest",
                    "event_id",
                    "ledger_seq",
                    "owner_id",
                    "consumed_at",
                ),
                field="canary consumption",
            )
            return _content_addressed_record(
                record.get("preflight_evidence"),
                digest_field="evidence_digest",
                field="canary preflight",
            )
        finally:
            connection.close()

    def consume(
        self,
        *,
        approved_plan_digest: str,
        disposition_digest: str,
        event_id: str,
        ledger_seq: int,
        owner_id: str,
        preflight_evidence: Mapping[str, object],
        consumed_at: datetime,
    ) -> dict[str, object]:
        approved_plan_digest = _token(
            approved_plan_digest, field="approved plan digest"
        )
        disposition_digest = _token(disposition_digest, field="disposition digest")
        event_id = _token(event_id, field="canary event id")
        owner_id = _token(owner_id, field="canary owner id")
        if isinstance(ledger_seq, bool) or not isinstance(ledger_seq, int) or ledger_seq <= 0:
            raise Issue790CanaryIntegrityError("bounded canary ledger sequence is invalid")
        if ledger_seq in {1932, 1972, 8834, 8835, 13284, 13337, 13361, 13362}:
            raise Issue790CanaryIntegrityError("bounded canary targeted a retained failure")
        consumed_at_text = _utc_text(consumed_at)

        connection = self._connection()
        try:
            _allow_reused_dispositions(connection)
            connection.execute("BEGIN IMMEDIATE")
            approved_contract = _require_effective_plan_contract(
                approved_plan_digest,
                connection,
                message="bounded canary approved plan differs",
            )
            if connection.execute(
                "SELECT 1 FROM issue_790_bounded_canary_consumptions "
                "WHERE approved_plan_digest=? LIMIT 1",
                (approved_plan_digest,),
            ).fetchone() is not None:
                raise Issue790CanaryIntegrityError(
                    "bounded canary authority is already consumed"
                )
            if connection.execute(
                "SELECT 1 FROM issue_790_bounded_canary_consumptions "
                "WHERE event_id=? OR ledger_seq=? LIMIT 1",
                (event_id, ledger_seq),
            ).fetchone() is not None:
                raise Issue790CanaryIntegrityError(
                    "bounded canary authority is already consumed"
                )
            if unmatched_bounded_canary_consumption(
                connection,
                approved_plan_digest=approved_plan_digest,
            ) is not None:
                raise Issue790CanaryIntegrityError(
                    "bounded canary authority is already consumed"
                )
            disposition_row = connection.execute(
                "SELECT invocation_id,approved_plan_digest,record_json "
                "FROM model_usage_conservative_dispositions "
                "WHERE disposition_digest=?",
                (disposition_digest,),
            ).fetchone()
            disposition_plan_digests = effective_issue_790_invocation_plan_digests(
                approved_contract.invocation_id,
                connection=connection,
            )
            if disposition_row is None:
                raise Issue790CanaryIntegrityError(
                    "bounded canary disposition authority is absent"
                )
            disposition = _object(disposition_row[2], field="canary disposition")
            if (
                str(disposition_row[0]) != approved_contract.invocation_id
                or str(disposition_row[1]) not in disposition_plan_digests
                or disposition.get("exact_usage_remains_unknown") is not True
                or disposition.get("unknown_spend_released") is not False
            ):
                raise Issue790CanaryIntegrityError(
                    "bounded canary disposition authority differs"
                )
            route = connection.execute(
                "SELECT state,reason FROM model_usage_route_circuit_events "
                "WHERE route='GRAPHITI_CHAT_PRIMARY' "
                "ORDER BY recorded_at DESC,rowid DESC LIMIT 1"
            ).fetchone()
            if route is None or tuple(route) != (
                "CLOSED",
                f"AUTHORISED_OPERATOR_RESET:{disposition_digest}",
            ):
                raise Issue790CanaryIntegrityError(
                    "bounded canary route release authority differs"
                )
            event_row = connection.execute(
                "SELECT state,attempt_count,available_at,claim_owner,claim_expires_at,"
                "manifest_json,unit_count,manifest_digest "
                "FROM unpublished_graphiti_revision_events "
                "WHERE event_id=? AND ledger_seq=?",
                (event_id, ledger_seq),
            ).fetchone()
            if (
                event_row is None
                or str(event_row[0]) != "QUEUED"
                or int(event_row[1]) != 0
                or str(event_row[2]) > consumed_at_text
                or event_row[3] is not None
                or event_row[4] is not None
            ):
                raise Issue790CanaryIntegrityError(
                    "bounded canary event is not fresh and claimable"
                )
            if graphiti_retry_excluded(connection, event_id=event_id):
                raise Issue790CanaryIntegrityError(
                    "bounded canary targeted a retry-excluded event"
                )
            event_circuit = connection.execute(
                "SELECT state,available_at FROM unpublished_graphiti_event_circuit "
                "WHERE singleton=1"
            ).fetchone()
            if (
                event_circuit is not None
                and str(event_circuit[0]) == "OPEN"
                and event_circuit[1] is not None
                and str(event_circuit[1]) > consumed_at_text
            ):
                raise Issue790CanaryIntegrityError(
                    "bounded canary event circuit is still open"
                )
            circuit_release = None
            if issue_790_owner_activated_sequence(approved_contract.sequence_ordinal):
                if event_circuit is not None and str(event_circuit[0]) == "OPEN":
                    raise Issue790CanaryIntegrityError(
                        "bounded canary event circuit is still open"
                    )
                release_row = None
                if _table_exists(connection, "issue_790_step16_circuit_releases"):
                    release_row = connection.execute(
                        _CIRCUIT_RELEASE_SELECT
                        + "WHERE plan_digest=? AND event_id=? AND ledger_seq=?",
                        (approved_plan_digest, event_id, ledger_seq),
                    ).fetchone()
                if release_row is not None:
                    circuit_release = _validated_circuit_release_row(release_row)
            manifest = _object(event_row[5], field="canary event manifest")
            unit_refs = manifest.get("unit_refs")
            if (
                not isinstance(unit_refs, list)
                or not all(isinstance(item, dict) for item in unit_refs)
                or len(unit_refs) != int(event_row[6])
                or manifest.get("ledger_seq") != ledger_seq
                or digest_canonical(manifest) != str(event_row[7])
            ):
                raise Issue790CanaryIntegrityError(
                    "bounded canary event manifest differs"
                )
            preflight = _validated_preflight(
                preflight_evidence,
                event_id=event_id,
                ledger_seq=ledger_seq,
                manifest_digest=str(event_row[7]),
                consumed_at=consumed_at,
                approved_plan_digest=approved_plan_digest,
                connection=connection,
            )
            resolved_units = preflight["resolved_units"]
            assert isinstance(resolved_units, list)
            if unit_refs and tuple(
                _stable_unit_ref(item) for item in unit_refs if isinstance(item, dict)
            ) != tuple(
                _stable_unit_ref(item)
                for item in resolved_units
                if isinstance(item, dict)
            ):
                raise Issue790CanaryIntegrityError(
                    "bounded canary retained and preflight units differ"
                )
            ingest_ids = tuple(str(item["ingest_id"]) for item in resolved_units)
            validate_graphiti_canary_target_unused(
                connection,
                event_id=event_id,
                ingest_ids=ingest_ids,
            )
            without_digest: dict[str, object] = {
                "schema_version": CANARY_CONSUMPTION_SCHEMA,
                "approved_plan_digest": approved_plan_digest,
                "disposition_digest": disposition_digest,
                "event_id": event_id,
                "ledger_seq": ledger_seq,
                "owner_id": owner_id,
                "preflight_evidence": preflight,
                "preflight_evidence_digest": preflight["evidence_digest"],
                "event_state_before": "QUEUED",
                "attempt_count_before": 0,
                "provider_io_authorised": True,
                "maximum_event_attempts": 1,
                "persistent_worker_must_remain_unloaded": True,
                "public_dispatch_authorised": False,
                "publication_authorised": False,
                "consumed_at": consumed_at_text,
            }
            if issue_790_owner_activated_sequence(approved_contract.sequence_ordinal):
                without_digest["circuit_release"] = circuit_release
            consumption_digest = digest_canonical(without_digest)
            record = {**without_digest, "consumption_digest": consumption_digest}
            connection.execute(
                "INSERT INTO issue_790_bounded_canary_consumptions("
                "consumption_digest,approved_plan_digest,disposition_digest,"
                "event_id,ledger_seq,owner_id,consumed_at,record_json) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    consumption_digest,
                    approved_plan_digest,
                    disposition_digest,
                    event_id,
                    ledger_seq,
                    owner_id,
                    consumed_at_text,
                    _json(record),
                ),
            )
            connection.commit()
            return record
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def complete(
        self,
        *,
        consumption_digest: str,
        event_id: str,
        ledger_seq: int,
        owner_id: str,
        process_result: Mapping[str, object] | None,
        completed_at: datetime,
        exception_code: str | None = None,
        completion_mode: str = "FOREGROUND",
        result_class: str | None = None,
        causal_report: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        consumption_digest = _token(
            consumption_digest, field="canary consumption digest"
        )
        event_id = _token(event_id, field="canary event id")
        owner_id = _token(owner_id, field="canary owner id")
        if completion_mode not in {"FOREGROUND", "ZERO_IO_RECOVERY"}:
            raise Issue790CanaryIntegrityError("bounded canary completion mode differs")
        if isinstance(ledger_seq, bool) or not isinstance(ledger_seq, int) or ledger_seq <= 0:
            raise Issue790CanaryIntegrityError("bounded canary ledger sequence is invalid")
        if process_result is not None and exception_code is not None:
            raise Issue790CanaryIntegrityError(
                "bounded canary outcome has both a result and an exception"
            )
        if exception_code is not None:
            exception_code = _token(exception_code, field="canary exception code")
        retained_result = None if process_result is None else dict(process_result)
        completed_at_text = _utc_text(completed_at)
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            consumption_row = connection.execute(
                "SELECT * FROM issue_790_bounded_canary_consumptions "
                "WHERE consumption_digest=? AND event_id=? AND ledger_seq=?",
                (consumption_digest, event_id, ledger_seq),
            ).fetchone()
            if consumption_row is None:
                raise Issue790CanaryIntegrityError(
                    "bounded canary consumption authority differs"
                )
            consumption = _bound_row_record(
                consumption_row,
                digest_field="consumption_digest",
                identity_fields=(
                    "consumption_digest",
                    "approved_plan_digest",
                    "disposition_digest",
                    "event_id",
                    "ledger_seq",
                    "owner_id",
                    "consumed_at",
                ),
                field="canary consumption",
            )
            approved_contract = _require_effective_plan_contract(
                str(consumption["approved_plan_digest"]),
                connection,
                message="bounded canary outcome plan differs",
            )
            iterative = approved_contract.sequence_ordinal > 0
            retained_causal_report: dict[str, object] | None = None
            if iterative:
                if result_class not in {
                    "TRUTHFUL_PROVIDER_SUCCESS",
                    "CONTROLLER_TIMEOUT_NON_SUCCESS",
                    "UNCLASSIFIED_NON_SUCCESS",
                }:
                    raise Issue790CanaryIntegrityError(
                        "bounded canary result class differs"
                    )
                if result_class == "CONTROLLER_TIMEOUT_NON_SUCCESS":
                    if causal_report is None:
                        raise Issue790CanaryIntegrityError(
                            "bounded canary causal report is absent"
                        )
                    retained_causal_report = _validated_controller_timeout_report(
                        causal_report
                    )
                elif causal_report is not None:
                    raise Issue790CanaryIntegrityError(
                        "bounded canary causal report is ineligible"
                    )
            elif result_class is not None or causal_report is not None:
                raise Issue790CanaryIntegrityError(
                    "legacy bounded canary result fields differ"
                )
            if consumption.get("owner_id") != owner_id:
                raise Issue790CanaryIntegrityError(
                    "bounded canary consumption owner differs"
                )
            prior = connection.execute(
                "SELECT * FROM issue_790_bounded_canary_outcomes "
                "WHERE consumption_digest=?",
                (consumption_digest,),
            ).fetchone()
            if prior is not None:
                retained_prior = _bound_row_record(
                    prior,
                    digest_field="outcome_digest",
                    identity_fields=(
                        "outcome_digest",
                        "consumption_digest",
                        "event_id",
                        "ledger_seq",
                        "completed_at",
                    ),
                    field="canary outcome",
                )
                if (
                    retained_prior.get("event_id") != event_id
                    or retained_prior.get("ledger_seq") != ledger_seq
                    or retained_prior.get("owner_id") != owner_id
                    or (
                        iterative
                        and (
                            retained_prior.get("result_class") != result_class
                            or retained_prior.get("causal_report")
                            != retained_causal_report
                        )
                    )
                ):
                    raise Issue790CanaryIntegrityError(
                        "conflicting bounded canary outcome replay"
                    )
                connection.rollback()
                return retained_prior
            event_row = connection.execute(
                "SELECT state,attempt_count,claim_owner,last_failure_code,"
                "provider_dispatched FROM unpublished_graphiti_revision_events "
                "WHERE event_id=? AND ledger_seq=?",
                (event_id, ledger_seq),
            ).fetchone()
            if event_row is None:
                raise Issue790CanaryIntegrityError("bounded canary event disappeared")
            state_before_seal = str(event_row[0])
            attempt_count = int(event_row[1])
            claim_owner = None if event_row[2] is None else str(event_row[2])
            failure_code = None if event_row[3] is None else str(event_row[3])
            event_provider_dispatched_before_seal = bool(event_row[4])
            marker = connection.execute(
                "SELECT EXISTS("
                "SELECT 1 FROM model_invocation_allocations AS allocation "
                "JOIN model_transport_observations AS observation "
                "ON observation.invocation_id=allocation.invocation_id "
                "WHERE allocation.cycle_id=? "
                "AND allocation.workload_class IN (?,?,?) "
                "AND observation.state='DISPATCH_STARTED')",
                (
                    event_id,
                    "GRAPHITI_CHAT_PRIMARY",
                    "GRAPHITI_CHAT_FALLBACK",
                    "GRAPHITI_EMBEDDING",
                ),
            ).fetchone()
            provider_dispatched = bool(marker and marker[0])
            if iterative and result_class == "TRUTHFUL_PROVIDER_SUCCESS" and (
                state_before_seal != "TERMINAL"
                or attempt_count != 1
                or not provider_dispatched
            ):
                raise Issue790CanaryIntegrityError(
                    "bounded canary truthful success evidence differs"
                )
            if iterative and result_class == "CONTROLLER_TIMEOUT_NON_SUCCESS" and (
                state_before_seal == "TERMINAL"
                or attempt_count != 1
                or not provider_dispatched
            ):
                raise Issue790CanaryIntegrityError(
                    "bounded canary controller timeout evidence differs"
                )
            if attempt_count not in {0, 1}:
                raise Issue790CanaryIntegrityError(
                    "bounded canary retained more than one attempt"
                )
            if state_before_seal == "TERMINAL" and attempt_count != 1:
                raise Issue790CanaryIntegrityError(
                    "bounded canary terminal attempt count differs"
                )
            if retained_result is not None and (
                retained_result.get("event_id") != event_id
                or retained_result.get("ledger_seq") != ledger_seq
                or retained_result.get("attempt_count") != 1
                or retained_result.get("state") != state_before_seal
                or attempt_count != 1
            ):
                raise Issue790CanaryIntegrityError(
                    "bounded canary process result differs from retained event"
                )
            if state_before_seal in {"CLAIMED", "RUNNING"} and claim_owner != owner_id:
                raise Issue790CanaryIntegrityError(
                    "bounded canary active claim has a different owner"
                )
            if state_before_seal not in {
                "QUEUED",
                "CLAIMED",
                "RUNNING",
                "RETRY_HELD",
                "RIGHTS_HELD",
                "CONFIGURATION_HELD",
                "DEAD_LETTER",
                "TERMINAL",
            }:
                raise Issue790CanaryIntegrityError("bounded canary event state differs")
            sealed_state = state_before_seal
            sealed_failure_code = failure_code
            if state_before_seal != "TERMINAL":
                sealed_state = "CONFIGURATION_HELD"
                detail = exception_code or failure_code or "NO_EVENT_RESULT"
                if not str(detail).startswith("BOUNDED_CANARY_AUTHORITY_EXHAUSTED:"):
                    sealed_failure_code = (
                        f"BOUNDED_CANARY_AUTHORITY_EXHAUSTED:{detail}"
                    )
                cursor = connection.execute(
                    "UPDATE unpublished_graphiti_revision_events SET "
                    "state='CONFIGURATION_HELD',claim_owner=NULL,"
                    "claim_expires_at=NULL,last_failure_code=?,"
                    "provider_dispatched=? "
                    "WHERE event_id=? AND ledger_seq=? AND state=? "
                    "AND attempt_count=?",
                    (
                        sealed_failure_code,
                        int(provider_dispatched),
                        event_id,
                        ledger_seq,
                        state_before_seal,
                        attempt_count,
                    ),
                )
                if cursor.rowcount != 1:
                    raise Issue790CanaryIntegrityError(
                        "bounded canary event seal lost its exact state"
                    )
            else:
                cursor = connection.execute(
                    "UPDATE unpublished_graphiti_revision_events "
                    "SET provider_dispatched=? WHERE event_id=? AND ledger_seq=? "
                    "AND state='TERMINAL' AND attempt_count=?",
                    (
                        int(provider_dispatched),
                        event_id,
                        ledger_seq,
                        attempt_count,
                    ),
                )
                if cursor.rowcount != 1:
                    raise Issue790CanaryIntegrityError(
                        "bounded canary terminal telemetry sync lost its state"
                    )
            without_digest: dict[str, object] = {
                "schema_version": (
                    ITERATIVE_CANARY_OUTCOME_SCHEMA
                    if iterative
                    else CANARY_OUTCOME_SCHEMA
                ),
                "consumption_digest": consumption_digest,
                "approved_plan_digest": consumption["approved_plan_digest"],
                "event_id": event_id,
                "ledger_seq": ledger_seq,
                "owner_id": owner_id,
                "process_result": retained_result,
                "exception_code": exception_code,
                "completion_mode": completion_mode,
                "state_before_seal": state_before_seal,
                "state_after_seal": sealed_state,
                "attempt_count": attempt_count,
                "event_provider_dispatched_before_seal": (
                    event_provider_dispatched_before_seal
                ),
                "provider_dispatched": provider_dispatched,
                "failure_code_before_seal": failure_code,
                "failure_code_after_seal": sealed_failure_code,
                "retry_authorised": False,
                "completed_at": completed_at_text,
            }
            if iterative:
                without_digest["result_class"] = result_class
                without_digest["causal_report"] = retained_causal_report
            if issue_790_owner_activated_sequence(approved_contract.sequence_ordinal):
                without_digest["circuit_release"] = _validated_nested_circuit_release(
                    consumption.get("circuit_release")
                )
            outcome_digest = digest_canonical(without_digest)
            record = {**without_digest, "outcome_digest": outcome_digest}
            connection.execute(
                "INSERT INTO issue_790_bounded_canary_outcomes("
                "outcome_digest,consumption_digest,event_id,ledger_seq,"
                "completed_at,record_json) VALUES(?,?,?,?,?,?)",
                (
                    outcome_digest,
                    consumption_digest,
                    event_id,
                    ledger_seq,
                    completed_at_text,
                    _json(record),
                ),
            )
            # A failed canary settles its legacy runtime blocker only after this
            # exact outcome commits. Keep current usage in the same transaction.
            from .model_usage import _refresh_current_usage

            for allocation in connection.execute(
                "SELECT invocation_id FROM model_invocation_allocations WHERE cycle_id=?",
                (event_id,),
            ).fetchall():
                _refresh_current_usage(connection, str(allocation[0]))
            connection.commit()
            return record
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def finalise_without_dispatch(
        self,
        *,
        consumption_digest: str,
        event_id: str,
        ledger_seq: int,
        owner_id: str,
        completed_at: datetime,
        result_class: str | None = None,
        causal_report: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Seal an interrupted authority without starting another provider call."""

        prior = self.existing_outcome(consumption_digest=consumption_digest)
        if prior is not None:
            if (
                prior.get("event_id") != event_id
                or prior.get("ledger_seq") != ledger_seq
                or prior.get("owner_id") != owner_id
            ):
                raise Issue790CanaryIntegrityError(
                    "conflicting bounded canary finalisation replay"
                )
            return prior
        connection = self._connection()
        try:
            row = connection.execute(
                "SELECT e.state,e.attempt_count,c.approved_plan_digest "
                "FROM unpublished_graphiti_revision_events e "
                "JOIN issue_790_bounded_canary_consumptions c "
                "ON c.consumption_digest=? "
                "WHERE e.event_id=? AND e.ledger_seq=?",
                (consumption_digest, event_id, ledger_seq),
            ).fetchone()
            if row is None:
                raise Issue790CanaryIntegrityError("bounded canary event disappeared")
            attempt_count = int(row[1])
            process_result = (
                None
                if attempt_count == 0
                else {
                    "event_id": event_id,
                    "ledger_seq": ledger_seq,
                    "state": str(row[0]),
                    "attempt_count": attempt_count,
                }
            )
            approved_contract = _require_effective_plan_contract(
                str(row[2]),
                connection,
                message="bounded canary approved plan differs",
            )
        finally:
            connection.close()
        if approved_contract.sequence_ordinal > 0 and result_class is None:
            raise Issue790CanaryIntegrityError(
                "iterative zero-I/O finalisation classification is absent"
            )
        return self.complete(
            consumption_digest=consumption_digest,
            event_id=event_id,
            ledger_seq=ledger_seq,
            owner_id=owner_id,
            process_result=process_result,
            completed_at=completed_at,
            completion_mode="ZERO_IO_RECOVERY",
            result_class=result_class,
            causal_report=causal_report,
        )


__all__ = [
    "CANARY_PREFLIGHT_SCHEMA",
    "ITERATIVE_CANARY_PREFLIGHT_SCHEMA",
    "Issue790CanaryIntegrityError",
    "Issue790CanaryRepository",
    "RetryForbiddenSafetyError",
    "RetryForbiddenSafetyState",
    "evaluate_retry_forbidden_safety",
    "validate_retry_forbidden_safety_state",
    "graphiti_event_has_canary_consumption",
    "unmatched_bounded_canary_consumption",
    "validate_graphiti_canary_target_unused",
    "graphiti_excluded_event_ids",
    "graphiti_retry_excluded",
    "live_retry_forbidden_row_is_unclaimed",
    "retry_forbidden_has_claim_columns",
    "retry_forbidden_live_select",
    "retry_forbidden_live_snapshot",
    "retry_forbidden_observation",
    "retry_forbidden_safety_states_match",
    "validate_graphiti_canary_claim",
]
