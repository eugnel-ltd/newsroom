"""Index-only v43 preserves exact predecessor data and rolls back on interruption."""
import sqlite3

import pytest

from newsroom.authority import migrations
from .test_projection_chain_retirement import _fixture, _snapshot
from .graphiti_adapter_4d_migration_helpers import _drop_v43_retirement_lookup


def _v42(tmp_path):
    _, path, _, _ = _fixture(tmp_path)
    with sqlite3.connect(path) as conn:
        _drop_v43_retirement_lookup(conn)
    return path


def test_v43_adds_only_required_persistent_lookups_without_expiry(tmp_path):
    path = _v42(tmp_path)
    before = _snapshot(path)
    with sqlite3.connect(path, isolation_level=None) as conn:
        conn.execute('PRAGMA foreign_keys=ON')
        migrations.apply_pending_migrations(conn, applied_at='2026-10-03T00:00:00Z')
        assert conn.execute('PRAGMA user_version').fetchone()[0] == 43
        assert conn.execute('SELECT count(*) FROM sqlite_schema WHERE name LIKE "idx_retirement_%"').fetchone()[0] == 46
        assert conn.execute('SELECT count(*) FROM ledger_events WHERE retired_header_digest IS NOT NULL').fetchone()[0] == 0
        assert migrations.schema_fingerprint(conn) == migrations.EXPECTED_SCHEMA_FINGERPRINT
        assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
    assert _snapshot(path) == before


def test_interrupted_v43_restores_exact_rows_schema_and_history(tmp_path):
    path = _v42(tmp_path)
    before = _snapshot(path)
    class Interrupt(sqlite3.Connection):
        count = 0
        def execute(self, sql, parameters=()):
            if sql.startswith('CREATE INDEX "idx_retirement_'):
                self.count += 1
                if self.count == 3:
                    raise sqlite3.OperationalError('fixture migration interrupted')
            return super().execute(sql, parameters)
    with sqlite3.connect(path, isolation_level=None, factory=Interrupt) as conn:
        conn.execute('PRAGMA foreign_keys=ON')
        schema = migrations.schema_fingerprint(conn)
        with pytest.raises(sqlite3.DatabaseError, match='fixture migration interrupted'):
            migrations.apply_pending_migrations(conn, applied_at='2026-10-03T00:00:00Z')
        assert not conn.in_transaction
        assert conn.execute('PRAGMA user_version').fetchone()[0] == 42
        assert conn.execute('SELECT count(*) FROM authority_migrations WHERE version=43').fetchone()[0] == 0
        assert migrations.schema_fingerprint(conn) == schema
    assert _snapshot(path) == before


@pytest.mark.parametrize('fault', ['fingerprint', 'history', 'foreign-keys'])
def test_v43_rejects_unchecked_predecessor_without_any_schema_or_data_change(tmp_path, fault):
    path = _v42(tmp_path)
    if fault in {'fingerprint', 'history'}:
        with sqlite3.connect(path) as conn:
            if fault == 'fingerprint':
                conn.execute('DROP INDEX idx_ledger_events_recorded')
            else:
                guard = conn.execute("SELECT sql FROM sqlite_schema WHERE name='immutable_authority_migrations_update'").fetchone()[0]
                conn.execute('DROP TRIGGER immutable_authority_migrations_update')
                conn.execute("UPDATE authority_migrations SET checksum='wrong' WHERE version=42")
                conn.execute(guard)
    before = _snapshot(path)
    with sqlite3.connect(path, isolation_level=None) as conn:
        conn.execute('PRAGMA foreign_keys=' + ('OFF' if fault == 'foreign-keys' else 'ON'))
        schema = migrations.schema_fingerprint(conn)
        with pytest.raises(sqlite3.DatabaseError):
            if fault == 'foreign-keys':
                # The public wrapper enables foreign keys before its transaction;
                # exercise the checked migration's own rejection boundary.
                from newsroom.authority.projection_retirement_lookup_migrations import migrate_retirement_lookup
                conn.execute('BEGIN')
                try:
                    migrate_retirement_lookup(conn, expected_history=tuple(
                        row for row in migrations.EXPECTED_MIGRATION_HISTORY if row[0] <= 42))
                finally:
                    conn.rollback()
            else:
                migrations.apply_pending_migrations(conn, applied_at='2026-10-03T00:00:00Z')
        assert not conn.in_transaction
        assert migrations.schema_fingerprint(conn) == schema
        assert conn.execute('PRAGMA user_version').fetchone()[0] == 42
        assert conn.execute('SELECT count(*) FROM authority_migrations WHERE version=43').fetchone()[0] == 0
    assert _snapshot(path) == before


def test_v43_adds_only_missing_fk_prefixes_and_covers_every_expiry_probe(tmp_path):
    from newsroom.authority._foreign_keys import foreign_key_children
    from newsroom.authority.projection_retirement_lookup_migrations import _REQUIRED_CHILD_PREFIXES
    path = _v42(tmp_path)
    def indexed(conn, table, column):
        return any(not row[4] and next(iter(conn.execute(f'PRAGMA index_info("{row[1]}")')), (None,None,None))[2] == column
                   for row in conn.execute(f'PRAGMA index_list("{table}")'))
    with sqlite3.connect(path, isolation_level=None) as conn:
        conn.execute('PRAGMA foreign_keys=ON')
        assert len(_REQUIRED_CHILD_PREFIXES) == 42
        assert all(not indexed(conn, table, column) for table, column in _REQUIRED_CHILD_PREFIXES)
        existing = {row[0] for row in conn.execute("SELECT name FROM sqlite_schema WHERE type='index'")}
        migrations.apply_pending_migrations(conn, applied_at='2026-10-03T00:00:00Z')
        assert existing <= {row[0] for row in conn.execute("SELECT name FROM sqlite_schema WHERE type='index'")}
        for parent,key in [('ledger_events','event_id'),('authority_commands','command_id'),
            ('authority_payloads','payload_id'),('authorization_requests','request_digest'),
            ('authorization_decisions','authorization_decision_id'),('authentication_contexts','authentication_context_id')]:
            assert all(indexed(conn, table, columns[0]) for table,columns in foreign_key_children(conn,parent,key=key))
