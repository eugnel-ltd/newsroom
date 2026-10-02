"""Command expiry must probe the generated live-command FK by index."""
import sqlite3

from newsroom.authority import audit_retention
from .test_projection_chain_retirement import _fixture, _snapshot
from .projection_b1_helpers import open_projection_system


def _plan(connection):
    return tuple(row[3] for row in connection.execute("EXPLAIN QUERY PLAN DELETE FROM authority_commands WHERE command_id='fixture-command'"))


def test_real_command_expiry_indexes_generated_fk_and_restores_schema(tmp_path, monkeypatch, record_property):
    root, path, _, _ = _fixture(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        before_plan = _plan(connection)
        assert any(item.startswith("SCAN ledger_events") for item in before_plan)
    snapshot = _snapshot(path)[1]
    observed = []
    connect = sqlite3.connect
    class ObserveDelete(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql.startswith('DELETE FROM "authority_commands" WHERE'):
                plan = _plan(self)
                observed.append(plan)
                assert not any(item.startswith("SCAN ledger_events") for item in plan)
                assert super().execute("PRAGMA foreign_keys").fetchone()[0] == 1
            return super().execute(sql, parameters)
    def fixture_connect(database, *args, **kwargs):
        if str(database).startswith(path.as_uri()):
            kwargs["factory"] = ObserveDelete
        return connect(database, *args, **kwargs)
    monkeypatch.setattr(sqlite3, "connect", fixture_connect)
    report = audit_retention.retire_native_projection_diagnostics(root, apply=True)
    assert report["retired_projection_chains"] == 1 and observed
    assert _snapshot(path)[1] == snapshot
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT name FROM sqlite_schema WHERE name LIKE '_audit_maintenance_%'").fetchall() == []
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    with open_projection_system(path):
        pass
    record_property("command_expiry_plan_before", str(before_plan))
    record_property("command_expiry_plan_indexed", str(observed[0]))


def _scaled_expiry(count, indexed):
    with sqlite3.connect(":memory:", isolation_level=None) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript('''
            CREATE TABLE authority_commands(command_id TEXT PRIMARY KEY) STRICT;
            CREATE TABLE ledger_events(ledger_seq INTEGER PRIMARY KEY,command_id TEXT NOT NULL UNIQUE,
                retired_header_digest BLOB,live_command_id TEXT GENERATED ALWAYS AS
                (CASE WHEN retired_header_digest IS NULL THEN command_id END) VIRTUAL
                REFERENCES authority_commands(command_id)) STRICT;
            CREATE TEMP TABLE candidates(command_id TEXT PRIMARY KEY) WITHOUT ROWID;
        ''')
        for number in range(count * 5):
            key = f"command-{number}"
            connection.execute("INSERT INTO authority_commands VALUES(?)", (key,))
            connection.execute("INSERT INTO ledger_events(ledger_seq,command_id,retired_header_digest) VALUES(?,?,?)", (number, key, bytes(32) if number < count else None))
            if number < count:
                connection.execute("INSERT INTO candidates VALUES(?)", (key,))
        connection.execute("BEGIN")
        steps = [0]
        connection.set_progress_handler(lambda: steps.__setitem__(0, steps[0] + 100) or 0, 100)
        indexes = []
        if indexed:
            audit_retention._index_children(connection, "authority_commands", indexes, key="command_id")
        plan = _plan(connection)
        connection.execute("DELETE FROM authority_commands WHERE command_id IN (SELECT command_id FROM candidates)")
        for name in indexes:
            connection.execute(f'DROP INDEX "{name}"')
        connection.commit()
        connection.set_progress_handler(None, 0)
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT count(*) FROM authority_commands").fetchone()[0] == count * 4
        assert connection.execute("SELECT count(*) FROM ledger_events").fetchone()[0] == count * 5
        assert connection.execute("SELECT name FROM sqlite_schema WHERE name LIKE '_audit_maintenance_%'").fetchall() == []
        return steps[0], plan


def test_generated_command_fk_expiry_vm_scaling_includes_index_lifecycle(record_property):
    baseline = [_scaled_expiry(size, False) for size in (256, 512)]
    indexed = [_scaled_expiry(size, True) for size in (256, 512)]
    assert all(any(item.startswith("SCAN ledger_events") for item in plan) for _, plan in baseline)
    assert all(not any(item.startswith("SCAN ledger_events") for item in plan) for _, plan in indexed)
    assert baseline[1][0] > baseline[0][0] * 3
    assert indexed[1][0] < indexed[0][0] * 2.5
    assert indexed[1][0] * 20 < baseline[1][0]
    record_property("command_expiry_baseline_vm", str([steps for steps, _ in baseline]))
    record_property("command_expiry_indexed_vm", str([steps for steps, _ in indexed]))
