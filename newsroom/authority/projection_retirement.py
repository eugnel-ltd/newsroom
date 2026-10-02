"""Expire only retired, single-final optional no-op diagnostic chains."""
from __future__ import annotations

from dataclasses import asdict
import sqlite3

from .canonical import digest_canonical
from .projection_retirement_migrations import RETIRED_FIELDS, retired_record_digest


# These are provisional disposable rows, not reference roots. Any retained
# FK, causation, receipt or nested token below removes its candidate first.
_EXCLUSIONS = {
    "ledger_events": "event_id IN (SELECT event_id FROM _retirement_candidates)",
    "authority_commands": "command_id IN (SELECT command_id FROM _retirement_candidates)",
    "authority_audit_events": "command_id IN (SELECT command_id FROM _retirement_candidates)",
    "authority_aggregate_versions": "command_id IN (SELECT command_id FROM _retirement_candidates)",
    "authority_payloads": "payload_id IN (SELECT payload_id FROM _retirement_candidates)",
    "authorization_requests": "request_digest IN (SELECT authorization_request_digest FROM _retirement_candidates)",
    "authorization_decisions": "authorization_decision_id IN (SELECT authorization_decision_id FROM _retirement_candidates)",
    "authentication_contexts": "authentication_context_id IN (SELECT authentication_context_id FROM _retirement_candidates)",
    "projection_delivery_states": "(generation_id,ledger_seq) IN (SELECT generation_id,source_seq FROM _retirement_candidates)",
    "projection_delivery_attempts": "(generation_id,ledger_seq) IN (SELECT generation_id,source_seq FROM _retirement_candidates)",
    "projection_checkpoint_versions": "(generation_id,checkpoint_version) IN (SELECT generation_id,checkpoint_version FROM _retirement_checkpoints)",
}


def retained_condition(table: str) -> str:
    exclusion = _EXCLUSIONS.get(table)
    return "NOT (" + exclusion + ")" if exclusion else "1"


def select_candidates(conn: sqlite3.Connection) -> int:
    from .audit_retention import AuditRetentionError
    from ._projection_retention import retired_ignored_attempt
    from ._projection_store import _ProjectionAuthorityStore

    conn.execute("""CREATE TEMP TABLE _retirement_candidates AS
        SELECT e.event_id,e.command_id,e.payload_id,e.authentication_context_id,
               e.authorization_request_digest,e.authorization_decision_id,
               s.generation_id,s.ledger_seq AS source_seq
        FROM projection_generations g JOIN projection_delivery_states s ON s.generation_id=g.generation_id
        JOIN ledger_events e ON e.event_id=s.last_authority_event_id
        WHERE g.state='RETIRED' AND s.current_outcome='IGNORED_OPTIONAL'
          AND s.required=0 AND s.finalized=1 AND s.attempt_count=1 AND s.last_error_code IS NULL
          AND e.retired_header_digest IS NULL
          AND e.event_type='projection.delivery.recorded' AND e.aggregate_type='projection_generation'
          AND e.aggregate_id=g.generation_id
          AND NOT EXISTS(SELECT 1 FROM projection_families f
              JOIN projection_family_complete_contracts b ON b.definition_digest=f.definition_digest
              WHERE f.family_id=g.family_id)
          AND NOT EXISTS(SELECT 1 FROM projection_delivery_attempts a
              WHERE a.generation_id=s.generation_id AND a.ledger_seq=s.ledger_seq AND a.attempt_number<>1)
    """)
    conn.execute("CREATE UNIQUE INDEX _retirement_candidate_event ON _retirement_candidates(event_id)")
    conn.execute("CREATE UNIQUE INDEX _retirement_candidate_command ON _retirement_candidates(command_id)")
    conn.execute("CREATE UNIQUE INDEX _retirement_candidate_source ON _retirement_candidates(generation_id,source_seq)")
    conn.execute("""CREATE TEMP TABLE _retirement_checkpoints AS
        SELECT c.generation_id,c.checkpoint_version,c.authority_event_id
        FROM projection_checkpoint_versions c JOIN projection_generations g USING(generation_id)
        WHERE g.state='RETIRED' AND EXISTS(SELECT 1 FROM _retirement_candidates x WHERE x.generation_id=g.generation_id)
    """)
    conn.execute("CREATE UNIQUE INDEX _retirement_checkpoint_key ON _retirement_checkpoints(generation_id,checkpoint_version)")
    # Authenticate the exact retained reconstruction, whether the dispensable
    # attempt is still present or was already elided by earlier maintenance.
    previous_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        for event_id, generation_id, sequence in conn.execute(
            "SELECT event_id,generation_id,source_seq FROM _retirement_candidates"
        ):
            state = conn.execute("SELECT * FROM projection_delivery_states WHERE generation_id=? AND ledger_seq=?", (generation_id, sequence)).fetchone()
            _ProjectionAuthorityStore._require_delivery_source_integrity(conn, state)
            reconstructed = retired_ignored_attempt(conn, event_id)
            cursor = conn.execute(
                "SELECT * FROM projection_delivery_attempts WHERE generation_id=? AND ledger_seq=?",
                (generation_id, sequence),
            )
            names = tuple(column[0] for column in cursor.description)
            attempts = cursor.fetchall()
            if (reconstructed is None or len(attempts) > 1
                    or any(reconstructed.get(name) != value for row in attempts
                           for name, value in zip(names, row, strict=True) if name != "delivery_attempt_id")):
                raise AuditRetentionError("retired projection candidate differs from exact authority")
    finally:
        conn.row_factory = previous_factory
    return int(conn.execute("SELECT count(*) FROM _retirement_candidates").fetchone()[0])


def protect_candidates(conn: sqlite3.Connection) -> int:
    from .audit_retention import _children, _q

    initial = int(conn.execute("SELECT count(*) FROM _retirement_candidates").fetchone()[0])
    conn.execute("""DELETE FROM _retirement_candidates WHERE event_id IN (
        SELECT e.event_id FROM ledger_events e JOIN authority_aggregates a
        ON a.aggregate_type=e.aggregate_type AND a.aggregate_id=e.aggregate_id AND a.current_version=e.aggregate_version
        UNION SELECT event_id FROM ledger_events WHERE ledger_seq=(SELECT max(ledger_seq) FROM ledger_events)
        UNION SELECT event_id FROM ledger_events WHERE ledger_seq=(SELECT max(ledger_seq) FROM ledger_events WHERE aggregate_type NOT IN ('projection_family','projection_generation'))
        UNION SELECT e.event_id FROM ledger_events e JOIN projection_generation_validations v
        ON e.ledger_seq=json_extract(CAST(v.canonical_bytes AS TEXT),'$.source_watermark_ledger_seq')
        UNION SELECT e.event_id FROM ledger_events e JOIN hybrid_retrieval_attempts r ON e.ledger_seq=r.authority_watermark
    )""")
    # The protected tokens include external stores/CAS and retained business
    # rows, not candidates' obsolete copies of their own identities.
    conn.execute("""DELETE FROM _retirement_checkpoints WHERE authority_event_id IN (SELECT id FROM _audit_tokens)""")
    while True:
        before = int(conn.execute("SELECT count(*) FROM _retirement_candidates").fetchone()[0])
        conn.execute("""DELETE FROM _retirement_candidates WHERE
            event_id IN (SELECT id FROM _audit_tokens) OR command_id IN (SELECT id FROM _audit_tokens)
            OR payload_id IN (SELECT id FROM _audit_tokens)
            OR authorization_request_digest IN (SELECT id FROM _audit_tokens)
            OR authorization_request_digest IN (SELECT request_digest FROM authorization_requests WHERE canonical_record_digest IN (SELECT id FROM _audit_tokens))
            OR event_id IN (SELECT authority_event_id FROM projection_delivery_attempts WHERE delivery_attempt_id IN (SELECT id FROM _audit_tokens))
            OR command_id IN (SELECT command_id FROM authority_commands WHERE result_digest IN (SELECT id FROM _audit_tokens)
                OR stable_semantic_request_digest IN (SELECT id FROM _audit_tokens))
            OR command_id IN (SELECT command_id FROM authority_audit_events WHERE audit_id IN (SELECT id FROM _audit_tokens)
                OR detail_digest IN (SELECT id FROM _audit_tokens))
            OR payload_id IN (SELECT payload_id FROM authority_payloads WHERE payload_digest IN (SELECT id FROM _audit_tokens))
        """)
        for parent, key, candidate_key in (
            ("ledger_events", "event_id", "event_id"),
            ("authority_commands", "command_id", "command_id"),
            ("authorization_requests", "request_digest", "authorization_request_digest"),
        ):
            for table, column in _children(conn, parent, key=key):
                # An unmapped no-op delivery consumes only original routing and
                # header identity, not expired payload/security provenance.
                route = ""
                if parent == "ledger_events" and column == "source_event_id" and table in {"projection_delivery_states", "projection_delivery_attempts"}:
                    outcome = "current_outcome" if table.endswith("states") else "outcome"
                    route = f" AND NOT (required=0 AND {outcome}='IGNORED_OPTIONAL')"
                conn.execute(
                    f"DELETE FROM _retirement_candidates WHERE {_q(candidate_key)} IN "
                    f"(SELECT {_q(column)} FROM {_q(table)} WHERE {retained_condition(table)}{route})"
                )
        # A retained causal child pins full EVENT/COMMAND provenance. This is
        # not generation-scoped; removing one candidate can pin its predecessor.
        conn.execute("""DELETE FROM _retirement_candidates WHERE event_id IN (
            SELECT causation_identifier FROM ledger_events WHERE causation_kind='EVENT'
            AND event_id NOT IN (SELECT event_id FROM _retirement_candidates)) OR command_id IN (
            SELECT causation_identifier FROM ledger_events WHERE causation_kind='COMMAND'
            AND event_id NOT IN (SELECT event_id FROM _retirement_candidates))""")
        # Do not expire checkpoint history for a generation whose candidates
        # are all protected. Actual checkpoint consumers stay exact/full.
        conn.execute("DELETE FROM _retirement_checkpoints WHERE generation_id NOT IN (SELECT generation_id FROM _retirement_candidates)")
        if int(conn.execute("SELECT count(*) FROM _retirement_candidates").fetchone()[0]) == before:
            break
    return initial - int(conn.execute("SELECT count(*) FROM _retirement_candidates").fetchone()[0])


def _validation_reader(conn: sqlite3.Connection):
    """Reuse exact event closure checks over the already writer-owned transaction."""
    import threading
    from ._event_store import _EventAuthorityStore
    from .policy import CommandRegistry, PayloadSchemaRegistry
    from newsroom.projection.policy import projection_command_definitions, projection_payload_contracts

    reader = object.__new__(_EventAuthorityStore)
    reader._conn, reader._closed, reader._lock = conn, False, threading.RLock()
    reader._command_registry = CommandRegistry(projection_command_definitions())
    reader._payload_schemas = PayloadSchemaRegistry(projection_payload_contracts())
    return reader


def _authenticate_chain(conn: sqlite3.Connection, event_id: str, reader):
    reader._validate_retained_event(event_id)
    event = conn.execute("SELECT * FROM ledger_events WHERE event_id=?", (event_id,)).fetchone()
    command = conn.execute("SELECT * FROM authority_commands WHERE command_id=?", (event["command_id"],)).fetchone()
    return event, command, reader._event_from_row(event)


def expire_candidates(conn: sqlite3.Connection) -> dict[str, int]:
    from .audit_retention import _index_children, _q, _unreferenced

    # This operation is called only inside the writer-owned transaction and
    # checked reference closure. Ordinary writes retain all immutable guards.
    changed_tables = tuple(_EXCLUSIONS) + ("projection_generations",)
    guards = tuple(conn.execute(
        "SELECT name,sql FROM sqlite_schema WHERE type='trigger' AND tbl_name IN (" + ",".join("?" for _ in changed_tables) + ") AND (sql LIKE '%BEFORE DELETE%' OR name IN ('immutable_ledger_events_update','projection_generation_update_guard','projection_diagnostic_expiry_update_guard'))",
        changed_tables,
    ))
    for name, _ in guards:
        conn.execute(f"DROP TRIGGER {_q(name)}")
    columns = tuple(row[1] for row in conn.execute("PRAGMA table_info(ledger_events)"))
    removed = tuple(name for name in columns if name not in RETIRED_FIELDS and name not in {"retired_header_digest", "retired_record_digest"})
    conn.execute("PRAGMA defer_foreign_keys=ON")
    previous_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    reader = _validation_reader(conn)
    try:
        for (event_id,) in conn.execute("SELECT event_id FROM _retirement_candidates ORDER BY event_id"):
            event, command, header = _authenticate_chain(conn, event_id, reader)
            row = {name: event[name] for name in RETIRED_FIELDS if name not in {"retired_namespace", "retired_key"}}
            row.update(retired_namespace=command["idempotency_namespace"], retired_key=command["idempotency_key"], retired_header_digest=bytes.fromhex(digest_canonical(asdict(header))[7:]))
            conn.execute(
                "UPDATE ledger_events SET " + ",".join(name + "=NULL" for name in removed)
                + ",retired_header_digest=?,retired_namespace=?,retired_key=?,retired_record_digest=? WHERE event_id=?",
                (row["retired_header_digest"], row["retired_namespace"], row["retired_key"], retired_record_digest(row), event_id),
            )
    finally:
        conn.row_factory = previous_factory
    # Checkpoint histories may be sparse only for explicitly marked retired
    # generations. Keep every actual protected checkpoint consumer.
    conn.execute("UPDATE projection_generations SET diagnostic_history_expired=1 WHERE generation_id IN (SELECT generation_id FROM _retirement_candidates)")
    deleted = {}
    indexes = []
    for table in (
        "projection_delivery_attempts", "projection_delivery_states", "projection_checkpoint_versions",
        "authority_audit_events", "authority_aggregate_versions", "authority_commands", "authority_payloads",
        "authorization_decisions", "authorization_requests", "authentication_contexts",
    ):
        condition = "NOT (" + retained_condition(table) + ")"
        if table in {"authority_payloads", "authorization_decisions", "authorization_requests", "authentication_contexts"}:
            key = "payload_id" if table == "authority_payloads" else {
                "authorization_decisions": "authorization_decision_id", "authorization_requests": "request_digest", "authentication_contexts": "authentication_context_id",
            }[table]
            children = _index_children(conn, table, indexes, key=key)
            if table == "authority_payloads":
                condition += " AND NOT EXISTS(SELECT 1 FROM _audit_tokens t WHERE t.id=authority_payloads.payload_id OR t.id=authority_payloads.payload_digest)"
                condition += "".join(f" AND NOT EXISTS(SELECT 1 FROM {_q(child)} WHERE {_q(column)}=authority_payloads.payload_id)" for child, column in children)
            else:
                condition += " AND " + _unreferenced(table, children)
        conn.execute(f"DELETE FROM {_q(table)} WHERE {condition}")
        deleted[table] = int(conn.execute("SELECT changes()").fetchone()[0])
    for name in indexes:
        conn.execute(f"DROP INDEX {_q(name)}")
    for _, sql in guards:
        conn.execute(sql)
    return deleted
