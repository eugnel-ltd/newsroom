"""v41 upgrades atomically without expiry, copies or weakened native FKs."""
import sqlite3

import pytest

from newsroom.authority import migrations
from newsroom.authority.projection_retirement_migrations import PROJECTION_RETIREMENT_PREDECESSOR_FINGERPRINT
from .projection_b1_helpers import open_projection_system
from .test_retired_projection_audit import _seed
from .test_projection_chain_retirement import _snapshot


def _v41(path):
    _seed(path)
    from .graphiti_adapter_4d_migration_helpers import _drop_v42_projection_retirement
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        _drop_v42_projection_retirement(conn)
        assert migrations.schema_fingerprint(conn) == PROJECTION_RETIREMENT_PREDECESSOR_FINGERPRINT


def test_v42_migrates_exact_v41_with_native_foreign_keys_and_no_expiry(tmp_path):
    path = tmp_path / "authority.sqlite3"
    _v41(path)
    before = _snapshot(path)
    with open_projection_system(path):
        pass
    assert _snapshot(path) == before
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == migrations.SCHEMA_VERSION
        assert conn.execute("SELECT count(*) FROM ledger_events WHERE retired_header_digest IS NOT NULL").fetchone()[0] == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert migrations.schema_fingerprint(conn) == migrations.EXPECTED_SCHEMA_FINGERPRINT
    assert not list(tmp_path.glob("*backup*"))


@pytest.mark.parametrize("fault", ["schema", "history", "foreign-key"])
def test_v42_rejects_unchecked_predecessor_atomically(tmp_path, fault):
    path = tmp_path / "authority.sqlite3"
    _v41(path)
    with sqlite3.connect(path) as conn:
        if fault == "schema":
            conn.execute("DROP INDEX idx_ledger_events_recorded")
        elif fault == "history":
            sql = conn.execute("SELECT sql FROM sqlite_schema WHERE name='immutable_authority_migrations_update'").fetchone()[0]
            conn.execute("DROP TRIGGER immutable_authority_migrations_update")
            conn.execute("UPDATE authority_migrations SET checksum='broken' WHERE version=41")
            conn.execute(sql)
        else:
            sql = conn.execute("SELECT sql FROM sqlite_schema WHERE name='immutable_authority_audit_events_update'").fetchone()[0]
            conn.execute("DROP TRIGGER immutable_authority_audit_events_update")
            conn.execute("UPDATE authority_audit_events SET command_id='missing' WHERE rowid=(SELECT min(rowid) FROM authority_audit_events)")
            conn.execute(sql)
    with sqlite3.connect(path, isolation_level=None) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        schema = migrations.schema_fingerprint(conn)
        with pytest.raises(sqlite3.DatabaseError):
            migrations.apply_pending_migrations(conn, applied_at="2026-10-02T12:00:00.000000Z")
        assert not conn.in_transaction
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 41
        assert migrations.schema_fingerprint(conn) == schema
        assert conn.execute("SELECT count(*) FROM authority_migrations WHERE version=42").fetchone()[0] == 0
