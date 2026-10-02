"""Native parent replacement needs indexed child probes, not disabled FKs."""
from __future__ import annotations

import sqlite3
import json
import signal
import subprocess
import sys

import pytest

from newsroom.authority._foreign_keys import index_foreign_key_children
from newsroom.authority import migrations
from newsroom.authority.projection_retirement_migrations import PROJECTION_RETIREMENT_MIGRATION_CHECKSUM
from .test_projection_retirement_migrations import _v41
from .test_projection_chain_retirement import _snapshot
from .projection_b1_helpers import open_projection_system


def _schema(connection):
    return tuple(connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"))


def _parent_plan(connection, parent, where):
    return tuple(row[3] for row in connection.execute(f'EXPLAIN QUERY PLAN DELETE FROM "{parent}" WHERE {where}'))


def test_real_v41_parent_drop_uses_no_child_table_or_child_index_scans(tmp_path, record_property):
    path = tmp_path / "authority.sqlite3"
    _v41(path)
    before = _snapshot(path)
    class ObserveDrop(sqlite3.Connection):
        before_plan = ()
        indexed_plan = ()
        created = ()
        def execute(self, sql, parameters=()):
            if sql == "DROP TABLE ledger_events":
                self.indexed_plan = _parent_plan(self, "ledger_events", "event_id='fixture-event'")
                self.created = tuple(row[0] for row in super().execute("SELECT name FROM sqlite_schema WHERE name LIKE '_projection_migration_fk_%' ORDER BY name"))
                assert super().execute("PRAGMA foreign_keys").fetchone()[0] == 1
                assert not any(plan.startswith("SCAN ") for plan in self.indexed_plan)
            return super().execute(sql, parameters)
    with sqlite3.connect(path, isolation_level=None, factory=ObserveDrop) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.before_plan = _parent_plan(connection, "ledger_events", "event_id='fixture-event'")
        assert sum(plan.startswith("SCAN ") for plan in connection.before_plan) >= 18
        old_sequence = connection.execute("SELECT seq FROM sqlite_sequence WHERE name='ledger_events'").fetchone()
        migrations.apply_pending_migrations(connection, applied_at="2026-10-02T19:00:00.000000Z")
        assert connection.created
        record_property("v41_parent_delete_plan_before", json.dumps(connection.before_plan))
        record_property("v41_parent_delete_plan_indexed", json.dumps(connection.indexed_plan))
        record_property("transient_child_indexes", json.dumps(connection.created))
        assert migrations.schema_fingerprint(connection) == migrations.EXPECTED_SCHEMA_FINGERPRINT
        assert connection.execute("SELECT seq FROM sqlite_sequence WHERE name='ledger_events'").fetchone() == old_sequence
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT name FROM sqlite_schema WHERE name LIKE '_projection_migration_fk_%'").fetchall() == []
        assert connection.execute("SELECT checksum FROM authority_migrations WHERE version=42").fetchone()[0] == PROJECTION_RETIREMENT_MIGRATION_CHECKSUM
    assert _snapshot(path) == before
    with open_projection_system(path):
        pass


@pytest.mark.parametrize("stage", ["before_drop", "after_drop", "after_create", "after_insert"])
def test_interruption_restores_exact_v41_rows_schema_and_transient_indexes(tmp_path, stage):
    path = tmp_path / "authority.sqlite3"
    _v41(path)
    before = _snapshot(path)
    class Interrupted(sqlite3.Connection):
        seen = False
        def execute(self, sql, parameters=()):
            if stage == "before_drop" and sql == "DROP TABLE ledger_events":
                assert super().execute("SELECT name FROM sqlite_schema WHERE name LIKE '_projection_migration_fk_%' LIMIT 1").fetchone() is not None
                self.seen = True
                raise sqlite3.OperationalError("fixture interruption")
            result = super().execute(sql, parameters)
            selected = ((stage == "after_drop" and sql == "DROP TABLE ledger_events")
                        or (stage == "after_create" and sql.startswith("CREATE TABLE ledger_events("))
                        or (stage == "after_insert" and sql.startswith("INSERT INTO ledger_events(")))
            if selected:
                self.seen = True
                raise sqlite3.OperationalError("fixture interruption")
            return result
    with sqlite3.connect(path, isolation_level=None, factory=Interrupted) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        original = _schema(connection)
        with pytest.raises(sqlite3.OperationalError, match="fixture interruption"):
            migrations.apply_pending_migrations(connection, applied_at="2026-10-02T19:00:00.000000Z")
        assert connection.seen and not connection.in_transaction
        assert _schema(connection) == original
    with sqlite3.connect(path) as reopened:
        assert _schema(reopened) == original
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 41
        assert reopened.execute("PRAGMA foreign_key_check").fetchall() == []
        assert reopened.execute("SELECT name FROM sqlite_schema WHERE name LIKE '_projection_migration_fk_%'").fetchall() == []
    assert _snapshot(path) == before


def _scaled_replace(count, indexed):
    with sqlite3.connect(":memory:", isolation_level=None) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript('''
            CREATE TABLE parent(id INTEGER PRIMARY KEY,a TEXT NOT NULL,b INTEGER NOT NULL,UNIQUE(a,b),UNIQUE(a)) STRICT;
            CREATE TABLE scalar_child(id INTEGER PRIMARY KEY,ref INTEGER REFERENCES parent(id)) STRICT;
            CREATE TABLE composite_child(id INTEGER PRIMARY KEY,x TEXT,y INTEGER,FOREIGN KEY(x,y) REFERENCES parent(a,b)) STRICT;
            CREATE TABLE generated_child(id INTEGER PRIMARY KEY,original TEXT,ref TEXT GENERATED ALWAYS AS(original) VIRTUAL REFERENCES parent(a)) STRICT;
        ''')
        parents = [(i, f"key-{i}", i) for i in range(1, count + 1)]
        connection.executemany("INSERT INTO parent VALUES(?,?,?)", parents)
        rows = [(i, 1 + i % count) for i in range(count * 4)]
        connection.executemany("INSERT INTO scalar_child VALUES(?,?)", rows)
        connection.executemany("INSERT INTO composite_child VALUES(?,?,?)", [(i, f"key-{ref}", ref) for i, ref in rows])
        connection.executemany("INSERT INTO generated_child(id,original) VALUES(?,?)", [(i, f"key-{ref}") for i, ref in rows])
        schema = _schema(connection)
        steps = [0]
        connection.set_progress_handler(lambda: steps.__setitem__(0, steps[0] + 100) or 0, 100)
        connection.execute("BEGIN EXCLUSIVE")
        connection.execute("PRAGMA defer_foreign_keys=ON")
        names = []
        if indexed:
            index_foreign_key_children(connection, "parent", names, prefix="_fixture_fk_")
        plans = _parent_plan(connection, "parent", "id=1")
        connection.execute("CREATE TEMP TABLE buffer AS SELECT * FROM parent")
        connection.execute("DROP TABLE parent")
        connection.execute("CREATE TABLE parent(id INTEGER PRIMARY KEY,a TEXT NOT NULL,b INTEGER NOT NULL,UNIQUE(a,b),UNIQUE(a)) STRICT")
        connection.execute("INSERT INTO parent SELECT * FROM buffer")
        connection.execute("DROP TABLE buffer")
        for name in names:
            connection.execute('DROP INDEX "' + name + '"')
        connection.commit()
        connection.set_progress_handler(None, 0)
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert _schema(connection) == schema
        assert connection.execute("SELECT * FROM parent ORDER BY id").fetchall() == parents
        return steps[0], plans


def test_native_fk_parent_replacement_vm_scaling_is_materially_improved(record_property):
    baseline = [_scaled_replace(count, False) for count in (256, 512)]
    indexed = [_scaled_replace(count, True) for count in (256, 512)]
    assert all(any(plan.startswith("SCAN ") for plan in plans) for _, plans in baseline)
    assert all(not any(plan.startswith("SCAN ") for plan in plans) for _, plans in indexed)
    assert baseline[1][0] > baseline[0][0] * 3
    assert indexed[1][0] < indexed[0][0] * 3
    assert indexed[1][0] * 20 < baseline[1][0]
    record_property("baseline_vm_steps_256_512", str([row[0] for row in baseline]))
    record_property("indexed_vm_steps_256_512", str([row[0] for row in indexed]))


def test_existing_prefix_indexes_are_reused_but_partial_and_wrong_prefix_are_not():
    with sqlite3.connect(":memory:", isolation_level=None) as connection:
        connection.executescript('''
            CREATE TABLE parent(a TEXT,b INTEGER,PRIMARY KEY(a,b)) WITHOUT ROWID,STRICT;
            CREATE TABLE child(x TEXT,y INTEGER,z INTEGER,FOREIGN KEY(x,y) REFERENCES parent(a,b)) STRICT;
            CREATE INDEX reused ON child(y,x,z);
            CREATE TABLE partial(x TEXT,y INTEGER,z INTEGER,FOREIGN KEY(x,y) REFERENCES parent(a,b)) STRICT;
            CREATE INDEX partial_only ON partial(x,y) WHERE z=1;
            CREATE TABLE wrong(x TEXT,y INTEGER,z INTEGER,FOREIGN KEY(x,y) REFERENCES parent(a,b)) STRICT;
            CREATE INDEX wrong_prefix ON wrong(z,x,y);
        ''')
        before = _schema(connection)
        connection.execute("BEGIN")
        indexes = []
        children = index_foreign_key_children(connection, "parent", indexes, prefix="_fixture_fk_")
        assert set(children) == {("child", ("x", "y")), ("partial", ("x", "y")), ("wrong", ("x", "y"))}
        assert len(indexes) == 2
        assert not any(plan.startswith("SCAN ") for plan in _parent_plan(connection, "parent", "a='x' AND b=1"))
        connection.rollback()
        assert _schema(connection) == before


def test_single_parent_key_audit_lookup_indexes_its_actual_scalar_predicate():
    with sqlite3.connect(":memory:", isolation_level=None) as connection:
        connection.executescript('''
            CREATE TABLE parent(a TEXT,b INTEGER,UNIQUE(a,b)) STRICT;
            CREATE TABLE child(x TEXT,y INTEGER,FOREIGN KEY(x,y) REFERENCES parent(a,b)) STRICT;
            CREATE INDEX existing_composite ON child(x,y);
        ''')
        indexes = []
        assert index_foreign_key_children(connection, "parent", indexes, key="b", prefix="_fixture_fk_") == [("child", ("y",))]
        assert len(indexes) == 1
        plan = tuple(row[3] for row in connection.execute("EXPLAIN QUERY PLAN SELECT 1 FROM child WHERE y=1"))
        assert any("SEARCH child" in item for item in plan)


@pytest.mark.parametrize("stage", ["before_drop", "after_create"])
def test_sigterm_of_fixture_migrator_restores_v41_native_journal(tmp_path, stage):
    path = tmp_path / "authority.sqlite3"
    _v41(path)
    before = _snapshot(path)
    with sqlite3.connect(path) as connection:
        original = _schema(connection)
    script = r"""
import os, signal, sqlite3, sys
from newsroom.authority.migrations import apply_pending_migrations
stage, path = sys.argv[1:]
class Terminated(sqlite3.Connection):
    def execute(self, sql, parameters=()):
        if stage == 'before_drop' and sql == 'DROP TABLE ledger_events':
            assert super().execute("SELECT name FROM sqlite_schema WHERE name LIKE '_projection_migration_fk_%' LIMIT 1").fetchone() is not None
            os.kill(os.getpid(), signal.SIGTERM)
        result = super().execute(sql, parameters)
        if stage == 'after_create' and sql.startswith('CREATE TABLE ledger_events('):
            assert super().execute('PRAGMA foreign_keys').fetchone()[0] == 1
            os.kill(os.getpid(), signal.SIGTERM)
        return result
connection = sqlite3.connect(path, isolation_level=None, factory=Terminated)
connection.execute('PRAGMA foreign_keys=ON')
apply_pending_migrations(connection, applied_at='2026-10-02T19:00:00.000000Z')
raise AssertionError('fixture SIGTERM boundary was not reached')
"""
    terminated = subprocess.run([sys.executable, "-c", script, stage, str(path)], capture_output=True, text=True, timeout=30)
    assert terminated.returncode == -signal.SIGTERM, terminated.stderr
    with sqlite3.connect(path) as reopened:
        assert _schema(reopened) == original
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 41
        assert reopened.execute("PRAGMA foreign_key_check").fetchall() == []
        assert reopened.execute("SELECT name FROM sqlite_schema WHERE name LIKE '_projection_migration_fk_%'").fetchall() == []
    assert _snapshot(path) == before
