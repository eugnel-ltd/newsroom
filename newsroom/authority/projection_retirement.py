"""Expire single-attempt successful retired delivery diagnostics, retaining proof roots."""
from __future__ import annotations

from dataclasses import asdict
import sqlite3

from .canonical import digest_canonical
from .projection_retirement_migrations import RETIRED_FIELDS, retired_record_digest


# These are retired delivery diagnostics, not the projected source authority.
# Current local FK, causation and watermark consumers still protect required
# rows; historical opaque diagnostic links expire.
# Generation/sequence keys are NOT NULL. EXISTS keeps unmatched composite keys
# on the exact index lookup instead of row-value NOT IN's fallback scan.
_EXCLUSIONS = {
    "ledger_events": "event_id IN (SELECT event_id FROM _retirement_candidates)",
    "authority_commands": "command_id IN (SELECT command_id FROM _retirement_candidates)",
    "authority_audit_events": "command_id IN (SELECT command_id FROM _retirement_candidates)",
    "authority_aggregate_versions": "command_id IN (SELECT command_id FROM _retirement_candidates)",
    "authority_payloads": "payload_id IN (SELECT payload_id FROM _retirement_candidates)",
    "authorization_requests": "request_digest IN (SELECT authorization_request_digest FROM _retirement_candidates)",
    "authorization_decisions": "authorization_decision_id IN (SELECT authorization_decision_id FROM _retirement_candidates)",
    "authentication_contexts": "authentication_context_id IN (SELECT authentication_context_id FROM _retirement_candidates)",
    "projection_delivery_states": "EXISTS (SELECT 1 FROM _retirement_candidates x WHERE x.generation_id=projection_delivery_states.generation_id AND x.source_seq=projection_delivery_states.ledger_seq)",
    "projection_delivery_attempts": "EXISTS (SELECT 1 FROM _retirement_candidates x WHERE x.generation_id=projection_delivery_attempts.generation_id AND x.source_seq=projection_delivery_attempts.ledger_seq)",
    "projection_checkpoint_versions": "EXISTS (SELECT 1 FROM _retirement_checkpoints x WHERE x.generation_id=projection_checkpoint_versions.generation_id AND x.checkpoint_version=projection_checkpoint_versions.checkpoint_version)",
}


def retained_condition(table: str) -> str:
    exclusion = _EXCLUSIONS.get(table)
    return "NOT (" + exclusion + ")" if exclusion else "1"


def select_candidates(conn: sqlite3.Connection) -> int:
    # Visit each generation's state range once, then its exact event-ID key.
    # An event-first plan repeats the whole state range for every ledger event.
    conn.execute("""CREATE TEMP TABLE _retirement_candidates AS
        SELECT e.event_id,e.command_id,e.payload_id,e.authentication_context_id,
               e.authorization_request_digest,e.authorization_decision_id,
               s.generation_id,s.ledger_seq AS source_seq
        FROM projection_generations g CROSS JOIN projection_delivery_states s ON s.generation_id=g.generation_id
        CROSS JOIN ledger_events e ON e.event_id=s.last_authority_event_id
        WHERE g.state='RETIRED' AND s.current_outcome IN ('APPLIED','IGNORED_OPTIONAL')
          AND s.finalized=1 AND s.attempt_count=1 AND s.last_error_code IS NULL
          AND (s.current_outcome='APPLIED' OR s.required=0)
          AND e.retired_header_digest IS NULL
          AND e.event_type='projection.delivery.recorded' AND e.aggregate_type='projection_generation'
          AND e.aggregate_id=g.generation_id
          AND NOT EXISTS(SELECT 1 FROM projection_delivery_attempts a
              WHERE a.generation_id=s.generation_id AND a.ledger_seq=s.ledger_seq AND a.attempt_number<>1)
    """)
    conn.execute("CREATE UNIQUE INDEX _retirement_candidate_event ON _retirement_candidates(event_id)")
    conn.execute("CREATE UNIQUE INDEX _retirement_candidate_command ON _retirement_candidates(command_id)")
    conn.execute("CREATE UNIQUE INDEX _retirement_candidate_source ON _retirement_candidates(generation_id,source_seq)")
    for column in ("payload_id", "authentication_context_id",
                   "authorization_request_digest", "authorization_decision_id"):
        conn.execute(f"CREATE INDEX _retirement_candidate_{column} ON _retirement_candidates({column})")
    # Successful applied generations retain their final checkpoint and exact
    # event envelope. The marker keeps that pin stable on repeat maintenance.
    # Previously qualified ignored-only expiry retains its existing semantics.
    conn.execute("""CREATE TEMP TABLE _retirement_checkpoints AS
        SELECT c.generation_id,c.checkpoint_version,c.authority_event_id
        FROM projection_checkpoint_versions c JOIN projection_generations g USING(generation_id)
        WHERE g.state='RETIRED' AND EXISTS(SELECT 1 FROM _retirement_candidates x WHERE x.generation_id=g.generation_id)
          AND NOT (c.checkpoint_version=(SELECT max(last.checkpoint_version)
                   FROM projection_checkpoint_versions last WHERE last.generation_id=c.generation_id)
              AND (g.diagnostic_history_expired=1 OR EXISTS(SELECT 1 FROM projection_delivery_states s
                  WHERE s.generation_id=g.generation_id AND s.current_outcome='APPLIED')))
    """)
    conn.execute("CREATE UNIQUE INDEX _retirement_checkpoint_key ON _retirement_checkpoints(generation_id,checkpoint_version)")
    return int(conn.execute("SELECT count(*) FROM _retirement_candidates").fetchone()[0])


def protect_candidates(conn: sqlite3.Connection) -> int:
    from .audit_retention import _children, _q

    initial = int(conn.execute("SELECT count(*) FROM _retirement_candidates").fetchone()[0])
    conn.execute("""DELETE FROM _retirement_candidates WHERE event_id IN (
        SELECT e.event_id FROM authority_aggregates a CROSS JOIN ledger_events e
        ON a.aggregate_type=e.aggregate_type AND a.aggregate_id=e.aggregate_id AND a.current_version=e.aggregate_version
        WHERE a.aggregate_type='projection_generation'
        UNION SELECT event_id FROM ledger_events WHERE ledger_seq=(SELECT max(ledger_seq) FROM ledger_events)
        UNION SELECT event_id FROM ledger_events WHERE ledger_seq=(SELECT max(ledger_seq) FROM ledger_events WHERE aggregate_type NOT IN ('projection_family','projection_generation'))
        UNION SELECT e.event_id FROM ledger_events e JOIN projection_generation_validations v
        ON e.ledger_seq=json_extract(CAST(v.canonical_bytes AS TEXT),'$.source_watermark_ledger_seq')
        UNION SELECT e.event_id FROM ledger_events e JOIN hybrid_retrieval_attempts r ON e.ledger_seq=r.authority_watermark
    )""")
    while True:
        before = int(conn.execute("SELECT count(*) FROM _retirement_candidates").fetchone()[0])
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


def expire_candidates(conn: sqlite3.Connection) -> dict[str, int]:
    from .audit_retention import _index_children, _q, _unreferenced
    from ._event_store_read import _EventStoreReadMixin

    # This operation is called only inside the writer-owned transaction and
    # local reference protection. Ordinary writes retain all immutable guards.
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
    try:
        # Stream original headers once; obsolete diagnostic payload/security
        # history is deliberately discarded rather than reconstructed or audited.
        for event in conn.execute("""SELECT e.*,c.idempotency_namespace AS expired_namespace,
                c.idempotency_key AS expired_key
            FROM ledger_events e CROSS JOIN _retirement_candidates x ON x.event_id=e.event_id
            CROSS JOIN authority_commands c ON c.command_id=e.command_id
            ORDER BY e.ledger_seq"""):
            header = _EventStoreReadMixin._event_from_row(event)
            row = {name: event[name] for name in RETIRED_FIELDS if name not in {"retired_namespace", "retired_key"}}
            row.update(retired_namespace=event["expired_namespace"], retired_key=event["expired_key"], retired_header_digest=bytes.fromhex(digest_canonical(asdict(header))[7:]))
            conn.execute(
                "UPDATE ledger_events SET " + ",".join(name + "=NULL" for name in removed)
                + ",retired_header_digest=?,retired_namespace=?,retired_key=?,retired_record_digest=? WHERE ledger_seq=?",
                (row["retired_header_digest"], row["retired_namespace"], row["retired_key"], retired_record_digest(row), event["ledger_seq"]),
            )
    finally:
        conn.row_factory = previous_factory
    # Checkpoint histories may be sparse only for explicitly marked retired
    # generations. Keep every actual protected checkpoint consumer.
    conn.execute("UPDATE projection_generations SET diagnostic_history_expired=1 WHERE generation_id IN (SELECT generation_id FROM _retirement_candidates)")
    deleted = {}
    indexes = []
    # Reservations leave live_command_id NULL, but native parent DELETE still
    # needs an indexed generated-child probe for each expired command.
    _index_children(conn, "authority_commands", indexes, key="command_id")
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
                condition += "".join(f" AND NOT EXISTS(SELECT 1 FROM {_q(child)} WHERE {_q(column)}=authority_payloads.payload_id)" for child, column in children)
            else:
                condition += " AND " + _unreferenced(table, children, opaque_tokens=False)
        conn.execute(f"DELETE FROM {_q(table)} WHERE {condition}")
        deleted[table] = int(conn.execute("SELECT changes()").fetchone()[0])
    for name in indexes:
        conn.execute(f"DROP INDEX {_q(name)}")
    for _, sql in guards:
        conn.execute(sql)
    return deleted
