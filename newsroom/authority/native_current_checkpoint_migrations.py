"""Checked v44 source checkpoint identities; no expiry during migration."""
from __future__ import annotations
import re
import sqlite3
from .authorisation_scope_content_migrations import AuthorisationScopeContentMigrationRecord
from .canonical import digest_canonical

NATIVE_CHECKPOINT_SCHEMA_VERSION = 44
NATIVE_CHECKPOINT_MIGRATION_NAME = 'native_current_business_checkpoint_v44'
NATIVE_CHECKPOINT_SCHEMA_FINGERPRINT = 'sha256:140620ca77bb4c22f1064b45af7dd2aff44a12da999f53b6eb3ce8b624033627'
NATIVE_CHECKPOINT_PREDECESSOR_FINGERPRINT = 'sha256:a00dd159b3743d3e964c99a8fe4f59e3e772d8321cc472c779ded19d82779b0b'
NATIVE_CHECKPOINT_MIGRATION_STATEMENTS = (
    'require exact checked v43 history/schema and native foreign keys',
    'create immutable native checkpoint roots bound to governed object admission and original proof manifest',
    'preserve original routing/reservations; allow checkpoint-bound current Source/Discovery headers and expired historical keys',
    'preserve source command identity/result/payload; null obsolete security parents only for an exact native checkpoint',
    'remove only Discovery predecessor self-FKs; preserve canonical previous IDs, native FKs, immutable guards and old namespace/event reservations; expire no rows during empty-store creation',
)
NATIVE_CHECKPOINT_MIGRATION_CHECKSUM = digest_canonical({'version':44,
    'name':NATIVE_CHECKPOINT_MIGRATION_NAME,'statements':list(NATIVE_CHECKPOINT_MIGRATION_STATEMENTS)})
NATIVE_CHECKPOINT_MIGRATION = AuthorisationScopeContentMigrationRecord(44,
    NATIVE_CHECKPOINT_MIGRATION_NAME,NATIVE_CHECKPOINT_MIGRATION_CHECKSUM)


def _replace_parent(conn, table, ddl):
    columns = tuple(row[1] for row in conn.execute(f'PRAGMA table_info({table})'))
    indexes = tuple(row[0] for row in conn.execute(
        "SELECT sql FROM sqlite_schema WHERE type='index' AND tbl_name=? AND sql IS NOT NULL", (table,)))
    guards = tuple(row[0] for row in conn.execute(
        "SELECT sql FROM sqlite_schema WHERE type='trigger' AND tbl_name=?", (table,)))
    conn.execute(f'CREATE TEMP TABLE _checkpoint_{table} AS SELECT '+','.join(columns)+f' FROM {table}')
    conn.execute(f'DROP TABLE {table}')
    conn.execute(ddl)
    names=','.join(columns)
    conn.execute(f'INSERT INTO {table}({names}) SELECT {names} FROM _checkpoint_{table}')
    conn.execute(f'DROP TABLE _checkpoint_{table}')
    for sql in indexes: conn.execute(sql)
    for sql in guards: conn.execute(sql)


def migrate_native_checkpoint(conn:sqlite3.Connection, *, expected_history):
    from .migrations import schema_fingerprint
    if (not conn.in_transaction or conn.execute('PRAGMA foreign_keys').fetchone()[0] != 1
        or schema_fingerprint(conn) != NATIVE_CHECKPOINT_PREDECESSOR_FINGERPRINT
        or tuple(tuple(r) for r in conn.execute('SELECT version,name,checksum FROM authority_migrations ORDER BY version')) != expected_history):
        raise sqlite3.DatabaseError('v44 requires exact checked v43 under native foreign keys')
    if conn.execute('SELECT 1 FROM authority_commands LIMIT 1').fetchone() is not None or conn.execute('SELECT 1 FROM ledger_events LIMIT 1').fetchone() is not None:
        raise sqlite3.DatabaseError('checkpoint schema creation requires an empty NEW store; no live parent migration')
    conn.execute('''CREATE TABLE native_current_checkpoints(
        checkpoint_id TEXT PRIMARY KEY,
        admission_id TEXT NOT NULL REFERENCES object_admissions(admission_id),
        blob_digest TEXT NOT NULL REFERENCES blob_identities(blob_digest),
        manifest_digest TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL) STRICT''')
    command_ddl=conn.execute("SELECT sql FROM sqlite_schema WHERE name='authority_commands'").fetchone()[0]
    for field in ('authentication_context_id','authorization_request_digest','authorization_decision_id'):
        command_ddl=command_ddl.replace(field+' TEXT NOT NULL',field+' TEXT')
    command_ddl=command_ddl.replace('UNIQUE(idempotency_namespace', 'native_checkpoint_id TEXT REFERENCES native_current_checkpoints(checkpoint_id), UNIQUE(idempotency_namespace')
    command_ddl=command_ddl.rsplit(') STRICT',1)[0]+''',CHECK((native_checkpoint_id IS NULL AND authentication_context_id IS NOT NULL
            AND authorization_request_digest IS NOT NULL AND authorization_decision_id IS NOT NULL)
          OR (native_checkpoint_id IS NOT NULL AND authentication_context_id IS NULL
            AND authorization_request_digest IS NULL AND authorization_decision_id IS NULL))) STRICT'''
    ledger_ddl=conn.execute("SELECT sql FROM sqlite_schema WHERE name='ledger_events'").fetchone()[0]
    original="event_type='projection.delivery.recorded' AND aggregate_type='projection_generation'"
    qualified="("+original+" OR (event_type IN ('discovery.gate.decided','discovery.lead.disposition.recorded') AND aggregate_type IN ('gate_decision','lead_disposition_decision')) OR (native_checkpoint_id IS NOT NULL AND aggregate_type IN ('source_definition','source_definition_version','source_item','source_revision','discovery_representation','source_locator_continuity','discovery_occurrence','discovery_signal','gate_decision','news_lead','watch_condition','lead_disposition_decision')))"
    if original not in ledger_ddl:raise sqlite3.DatabaseError('v43 expiry predicate differs')
    ledger_ddl=ledger_ddl.replace(original,qualified)
    position=ledger_ddl.index('FOREIGN KEY(')
    ledger_ddl=ledger_ddl[:position]+'native_checkpoint_id TEXT REFERENCES native_current_checkpoints(checkpoint_id), '+ledger_ddl[position:]
    conn.execute('PRAGMA defer_foreign_keys=ON')
    _replace_parent(conn,'authority_commands',command_ddl)
    _replace_parent(conn,'ledger_events',ledger_ddl)
    # New-only native checkpoints bind the current row's original predecessor
    # identity/ordinal. Superseded diagnostic history is deliberately opaque.
    for table in ('discovery_gate_decisions','lead_disposition_decisions'):
        ddl=conn.execute('SELECT sql FROM sqlite_schema WHERE name=?',(table,)).fetchone()[0]
        ddl=re.sub(r'previous_decision_id TEXT\s+REFERENCES '+table+r'\(decision_id\)', 'previous_decision_id TEXT',ddl)
        _replace_parent(conn,table,ddl)
    conn.execute('''CREATE TABLE native_current_checkpoint_members(
        event_id TEXT PRIMARY KEY REFERENCES ledger_events(event_id),
        command_id TEXT NOT NULL UNIQUE REFERENCES authority_commands(command_id),
        checkpoint_id TEXT NOT NULL REFERENCES native_current_checkpoints(checkpoint_id),
        canonical_bytes BLOB NOT NULL,
        canonical_digest TEXT NOT NULL UNIQUE,
        CHECK(length(canonical_bytes)>0)) STRICT''')
    conn.execute('''CREATE TABLE native_expired_command_keys(
        namespace TEXT NOT NULL, key TEXT NOT NULL,
        command_id TEXT NOT NULL UNIQUE, event_id TEXT NOT NULL UNIQUE,
        ledger_seq INTEGER NOT NULL UNIQUE, security_scope TEXT NOT NULL,
        trust_scope TEXT NOT NULL, aggregate_type TEXT NOT NULL, aggregate_id TEXT NOT NULL,
        PRIMARY KEY(namespace,key)) WITHOUT ROWID, STRICT''')
    conn.execute('CREATE INDEX idx_native_expired_aggregate ON native_expired_command_keys(aggregate_type,aggregate_id)')
    for table in ('native_current_checkpoints','native_current_checkpoint_members','native_expired_command_keys'):
        for operation in ('UPDATE','DELETE'):
            conn.execute(f'''CREATE TRIGGER immutable_{table}_{operation.lower()} BEFORE {operation} ON {table}
                BEGIN SELECT RAISE(ABORT,'immutable native checkpoint'); END''')
    conn.execute('''CREATE TRIGGER native_checkpoint_command_insert_guard BEFORE INSERT ON authority_commands
        WHEN NEW.native_checkpoint_id IS NOT NULL BEGIN SELECT RAISE(ABORT,'checkpoint requires DEV maintenance'); END''')
    conn.execute('''CREATE TRIGGER native_expired_command_insert_guard BEFORE INSERT ON authority_commands
        WHEN EXISTS(SELECT 1 FROM native_expired_command_keys WHERE command_id=NEW.command_id)
          OR EXISTS(SELECT 1 FROM native_expired_command_keys WHERE namespace=NEW.idempotency_namespace AND key=NEW.idempotency_key)
        BEGIN SELECT RAISE(ABORT,'command diagnostic history expired; identity remains reserved'); END''')
    conn.execute('''CREATE TRIGGER native_expired_event_insert_guard BEFORE INSERT ON ledger_events
        WHEN EXISTS(SELECT 1 FROM native_expired_command_keys WHERE event_id=NEW.event_id OR ledger_seq=NEW.ledger_seq)
        BEGIN SELECT RAISE(ABORT,'event diagnostic history expired; identity remains reserved'); END''')
    conn.execute('''CREATE TRIGGER native_expired_aggregate_insert_guard BEFORE INSERT ON authority_aggregates
        WHEN EXISTS(SELECT 1 FROM native_expired_command_keys WHERE aggregate_type=NEW.aggregate_type AND aggregate_id=NEW.aggregate_id)
        BEGIN SELECT RAISE(ABORT,'aggregate diagnostic history expired; identity remains reserved'); END''')
    if conn.execute('PRAGMA foreign_key_check').fetchone() is not None:
        raise sqlite3.IntegrityError('v44 predecessor foreign-key integrity differs')


def initialise_empty_checkpoint_store(connection):
    """Explicit new-store schema; never upgrades any populated operational store."""
    from . import migrations
    if connection.execute("SELECT 1 FROM sqlite_schema WHERE type='table' LIMIT 1").fetchone() is not None:
        raise sqlite3.DatabaseError('native checkpoint requires an empty NEW database')
    migrations.apply_pending_migrations(connection,applied_at='1970-01-01T00:00:00.000000Z')
    try:
        connection.execute('BEGIN EXCLUSIVE')
        migrate_native_checkpoint(connection,expected_history=migrations.EXPECTED_MIGRATION_HISTORY)
        connection.execute('INSERT INTO authority_migrations VALUES(?,?,?,?)',
            (44,NATIVE_CHECKPOINT_MIGRATION_NAME,NATIVE_CHECKPOINT_MIGRATION_CHECKSUM,'1970-01-01T00:00:00.000000Z'))
        connection.execute('PRAGMA user_version=44')
        connection.commit()
    except BaseException:
        if connection.in_transaction:connection.rollback()
        raise


def require_checkpoint_schema(connection):
    from . import migrations
    expected=(*migrations.EXPECTED_MIGRATION_HISTORY,
              (44,NATIVE_CHECKPOINT_MIGRATION_NAME,NATIVE_CHECKPOINT_MIGRATION_CHECKSUM))
    version=connection.execute('PRAGMA user_version').fetchone()[0]
    fingerprint=NATIVE_CHECKPOINT_SCHEMA_FINGERPRINT
    if version==45:
        from .native_marker_layout_migrations import NAME,CHECKSUM,FINGERPRINT
        expected=(*expected,(45,NAME,CHECKSUM))
        fingerprint=FINGERPRINT
    if (version not in (44,45)
        or tuple(tuple(row) for row in connection.execute('SELECT version,name,checksum FROM authority_migrations ORDER BY version'))!=expected
        or migrations.schema_fingerprint(connection)!=fingerprint):
        raise sqlite3.DatabaseError('native checkpoint schema/history differs')
