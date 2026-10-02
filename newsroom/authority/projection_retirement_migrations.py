"""v42 permits compact reservations; expiry is separate checked maintenance."""
from __future__ import annotations

import re
import sqlite3

from .authorisation_scope_content_migrations import AuthorisationScopeContentMigrationRecord
from .canonical import digest_canonical

PROJECTION_RETIREMENT_SCHEMA_VERSION = 42
PROJECTION_RETIREMENT_MIGRATION_NAME = "retired_projection_diagnostic_reservations_v42"
PROJECTION_RETIREMENT_PREDECESSOR_FINGERPRINT = "sha256:94014382880d8d9cda0a7b727dea90ca522c65299345e5ea6f80cf857311a3e9"
RETIRED_FIELDS = (
    "ledger_seq", "event_id", "command_id", "event_type", "aggregate_type",
    "aggregate_id", "security_scope", "trust_scope", "retired_namespace", "retired_key",
)


def retired_record_digest(row) -> bytes:
    """Bind the reservation to routing, namespace/key and the original header."""
    return bytes.fromhex(digest_canonical({
        name: row[name] for name in RETIRED_FIELDS
    } | {"original_header_digest": bytes(row["retired_header_digest"]).hex()})[7:])


def migrate_projection_retirement(connection: sqlite3.Connection, *, expected_history) -> None:
    from .migrations import schema_fingerprint

    if (not connection.in_transaction
            or schema_fingerprint(connection) != PROJECTION_RETIREMENT_PREDECESSOR_FINGERPRINT
            or tuple(tuple(row) for row in connection.execute(
                "SELECT version,name,checksum FROM authority_migrations ORDER BY version"
            )) != expected_history):
        raise sqlite3.DatabaseError("v42 migration requires exact checked schema v41")
    if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise sqlite3.DatabaseError("v42 migration requires native foreign keys")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise sqlite3.IntegrityError("v41 predecessor foreign-key integrity differs")
    ddl = connection.execute("SELECT sql FROM sqlite_schema WHERE name='ledger_events'").fetchone()[0]
    columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(ledger_events)"))
    required = tuple(row[1] for row in connection.execute("PRAGMA table_info(ledger_events)") if row[3])
    retained = set(RETIRED_FIELDS) - {"retired_namespace", "retired_key"}
    removed = tuple(name for name in columns if name not in retained)
    ddl = re.sub(r'(command_id TEXT NOT NULL UNIQUE)\s+REFERENCES authority_commands\(command_id\)', r'\1', ddl)
    for name in removed:
        ddl = re.sub(r'\b' + name + r' (TEXT|INTEGER) NOT NULL', name + r' \1', ddl)
    position = ddl.index("FOREIGN KEY(")
    ddl = ddl[:position] + """retired_header_digest BLOB,
        retired_namespace TEXT,
        retired_key TEXT,
        retired_record_digest BLOB,
        live_command_id TEXT GENERATED ALWAYS AS
            (CASE WHEN retired_header_digest IS NULL THEN command_id END) VIRTUAL
            REFERENCES authority_commands(command_id),
        """ + ddl[position:]
    full = " AND ".join(name + " IS NOT NULL" for name in required)
    full += " AND retired_namespace IS NULL AND retired_key IS NULL AND retired_record_digest IS NULL"
    expired = " AND ".join(name + " IS NULL" for name in removed)
    expired += " AND event_type='projection.delivery.recorded' AND aggregate_type='projection_generation'"
    expired += " AND typeof(retired_header_digest)='blob' AND length(retired_header_digest)=32"
    expired += " AND retired_namespace IS NOT NULL AND retired_key IS NOT NULL"
    expired += " AND typeof(retired_record_digest)='blob' AND length(retired_record_digest)=32"
    ddl = ddl.rsplit(") STRICT", 1)[0] + ", CHECK((retired_header_digest IS NULL AND " + full + ") OR (retired_header_digest IS NOT NULL AND " + expired + "))) STRICT"
    indices = tuple(row[0] for row in connection.execute(
        "SELECT sql FROM sqlite_schema WHERE type='index' AND tbl_name='ledger_events' AND sql IS NOT NULL ORDER BY name"
    ))
    guards = tuple(connection.execute(
        "SELECT name,sql FROM sqlite_schema WHERE type='trigger' AND tbl_name='ledger_events' ORDER BY name"
    ))
    # Keep native FKs ON. Recreate the original parent name before reinserting
    # IDs: a renamed replacement leaves SQLite's deferred counter unresolved.
    connection.execute("PRAGMA defer_foreign_keys=ON")
    original_sequence = connection.execute("SELECT seq FROM sqlite_sequence WHERE name='ledger_events'").fetchone()
    connection.execute("CREATE TEMP TABLE _retirement_ledger_buffer AS SELECT * FROM ledger_events")
    connection.execute("DROP TABLE ledger_events")
    connection.execute(ddl)
    names = ",".join(columns)
    connection.execute(f"INSERT INTO ledger_events({names}) SELECT {names} FROM _retirement_ledger_buffer")
    connection.execute("DROP TABLE _retirement_ledger_buffer")
    if original_sequence is not None:
        connection.execute("UPDATE sqlite_sequence SET seq=? WHERE name='ledger_events'", (original_sequence[0],))
    for sql in indices:
        connection.execute(sql)
    for name, sql in guards:
        if name in {"ledger_event_payload_guard", "ledger_event_command_guard"}:
            sql = sql.replace("WHEN NOT EXISTS(", "WHEN NEW.retired_header_digest IS NULL AND NOT EXISTS(")
        if name == "ledger_event_event_causation_guard":
            sql = sql.replace("WHERE event_id=NEW.causation_identifier", "WHERE event_id=NEW.causation_identifier AND retired_header_digest IS NULL")
        connection.execute(sql)
    connection.execute("CREATE UNIQUE INDEX idx_retired_command_key ON ledger_events(retired_namespace,retired_key) WHERE retired_header_digest IS NOT NULL")
    connection.execute("""CREATE TRIGGER authority_commands_retired_key_guard BEFORE INSERT ON authority_commands
        WHEN EXISTS(SELECT 1 FROM ledger_events WHERE retired_header_digest IS NOT NULL
            AND retired_namespace=NEW.idempotency_namespace AND retired_key=NEW.idempotency_key)
            OR EXISTS(SELECT 1 FROM ledger_events WHERE command_id=NEW.command_id AND retired_header_digest IS NOT NULL)
        BEGIN SELECT RAISE(ABORT,'command diagnostic history expired; identity remains reserved'); END""")
    connection.execute("""CREATE TRIGGER ledger_retired_insert_guard BEFORE INSERT ON ledger_events
        WHEN NEW.retired_header_digest IS NOT NULL
        BEGIN SELECT RAISE(ABORT,'retirement requires checked maintenance'); END""")
    connection.execute("""ALTER TABLE projection_generations ADD COLUMN diagnostic_history_expired
        INTEGER NOT NULL DEFAULT 0 CHECK(diagnostic_history_expired IN (0,1)
            AND (diagnostic_history_expired=0 OR state='RETIRED'))""")
    connection.execute("""CREATE TRIGGER projection_diagnostic_expiry_insert_guard BEFORE INSERT ON projection_generations
        WHEN NEW.diagnostic_history_expired<>0
        BEGIN SELECT RAISE(ABORT,'retirement requires checked maintenance'); END""")
    connection.execute("""CREATE TRIGGER projection_diagnostic_expiry_update_guard BEFORE UPDATE OF diagnostic_history_expired ON projection_generations
        BEGIN SELECT RAISE(ABORT,'retirement requires checked maintenance'); END""")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise sqlite3.IntegrityError("v42 retained foreign-key integrity differs")


PROJECTION_RETIREMENT_MIGRATION_STATEMENTS = (
    "check exact v41 schema/history and native foreign keys",
    "stage only ledger rows; recreate original parent name and exact full constraints with an explicit retired variant",
    "retain native child FKs, original sequence and append-only guards; indexed reservations deny retired inserts, namespace/key reuse and new expired causation",
    "add guarded retired-generation diagnostic expiry marker; expire no records during schema migration",
)
PROJECTION_RETIREMENT_MIGRATION_CHECKSUM = digest_canonical({
    "version": PROJECTION_RETIREMENT_SCHEMA_VERSION,
    "name": PROJECTION_RETIREMENT_MIGRATION_NAME,
    "statements": list(PROJECTION_RETIREMENT_MIGRATION_STATEMENTS),
})
PROJECTION_RETIREMENT_MIGRATION = AuthorisationScopeContentMigrationRecord(
    PROJECTION_RETIREMENT_SCHEMA_VERSION, PROJECTION_RETIREMENT_MIGRATION_NAME,
    PROJECTION_RETIREMENT_MIGRATION_CHECKSUM,
)
