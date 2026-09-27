from __future__ import annotations

import random
import sqlite3

import pytest

from newsroom.authority._foreign_keys import has_foreign_key_violation
from newsroom.authority.migrations import apply_pending_migrations


def _native(connection):
    return connection.execute("PRAGMA foreign_key_check").fetchone() is not None


@pytest.mark.parametrize("row_factory", [None, sqlite3.Row])
def test_composite_nullable_duplicate_and_missing_keys_match_sqlite(row_factory):
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = row_factory
        connection.executescript('''
            CREATE TABLE parent(a TEXT,b INTEGER,UNIQUE(a,b)) STRICT;
            CREATE TABLE child(a TEXT,b INTEGER,
                FOREIGN KEY(a,b) REFERENCES parent(a,b)) STRICT;
            INSERT INTO parent VALUES('alpha',1),('beta',2),('001',3);
        ''')
        candidates = [(None, 999), ("absent", None), (None, None),
                      ("alpha", 1), ("alpha", 2), ("beta", 2),
                      ("001", 3), ("1", 3), ("absent", 4)]
        randomiser = random.Random(981)
        for _ in range(100):
            connection.execute("DELETE FROM child")
            connection.executemany("INSERT INTO child VALUES(?,?)", [
                randomiser.choice(candidates) for _ in range(randomiser.randrange(8))
            ])
            assert has_foreign_key_violation(connection) is _native(connection)


@pytest.mark.parametrize("declaration,child_type,parent,child", [
    ("INTEGER PRIMARY KEY", "TEXT", 1, "001"),
    ("TEXT COLLATE NOCASE PRIMARY KEY", "TEXT", "alpha", "ALPHA"),
    ("TEXT PRIMARY KEY", "TEXT COLLATE NOCASE", "alpha", "ALPHA"),
])
def test_affinity_and_collation_use_native_comparison(declaration, child_type, parent, child):
    with sqlite3.connect(":memory:") as connection:
        connection.executescript(f'''
            CREATE TABLE parent(id {declaration}) STRICT;
            CREATE TABLE child(id {child_type} REFERENCES parent(id)) STRICT;
        ''')
        connection.execute("INSERT INTO parent VALUES(?)", (parent,))
        connection.execute("INSERT INTO child VALUES(?)", (child,))
        assert has_foreign_key_violation(connection) is _native(connection)
        connection.execute("DELETE FROM parent")
        assert has_foreign_key_violation(connection) is _native(connection) is True


def test_implicit_parent_key_and_non_strict_tables_keep_native_semantics():
    with sqlite3.connect(":memory:") as connection:
        connection.executescript('''
            CREATE TABLE parent(id INTEGER PRIMARY KEY);
            CREATE TABLE child(id TEXT REFERENCES parent);
            INSERT INTO parent VALUES(1);
            INSERT INTO child VALUES('001');
        ''')
        assert has_foreign_key_violation(connection) is _native(connection) is False
        connection.execute("INSERT INTO child VALUES('missing')")
        assert has_foreign_key_violation(connection) is _native(connection) is True


def test_multiple_foreign_keys_self_references_and_quoted_names():
    with sqlite3.connect(":memory:") as connection:
        connection.executescript('''
            CREATE TABLE "parent""key"(id TEXT PRIMARY KEY) STRICT;
            CREATE TABLE child(id TEXT PRIMARY KEY,
                parent TEXT REFERENCES "parent""key"(id),
                previous TEXT REFERENCES child(id)) STRICT;
            INSERT INTO "parent""key" VALUES('p');
            INSERT INTO child VALUES('first','p',NULL),('second','p','first');
        ''')
        assert has_foreign_key_violation(connection) is _native(connection) is False
        connection.execute("UPDATE child SET previous='absent' WHERE id='second'")
        assert has_foreign_key_violation(connection) is _native(connection) is True


def test_full_migrated_schema_and_persisted_orphan_are_checked(tmp_path):
    database = tmp_path / "authority.sqlite3"
    with sqlite3.connect(database) as connection:
        apply_pending_migrations(connection, applied_at="2026-09-19T00:00:00.000000Z")
        assert has_foreign_key_violation(connection) is _native(connection) is False
        # Small leaf with no payload/canonical machinery: the relational fault
        # must still be found independently of all retained-record decoders.
        connection.execute('''CREATE TABLE test_leaf(
            ref TEXT REFERENCES authentication_contexts(authentication_context_id)
        ) STRICT''')
        connection.execute("INSERT INTO test_leaf VALUES('absent')")
    with sqlite3.connect(database) as reopened:
        assert has_foreign_key_violation(reopened) is _native(reopened) is True


def test_supported_keys_use_set_scan_without_changing_connection_settings():
    with sqlite3.connect(":memory:") as connection:
        connection.executescript('''
            CREATE TABLE parent(id TEXT PRIMARY KEY) STRICT;
            CREATE TABLE child(id TEXT REFERENCES parent(id)) STRICT;
            INSERT INTO parent VALUES('p'); INSERT INTO child VALUES('p');
        ''')
        statements = []
        connection.set_trace_callback(statements.append)
        assert has_foreign_key_violation(connection) is False
        assert any(" EXCEPT " in statement for statement in statements)
        assert not any("PRAGMA foreign_key_check" in statement for statement in statements)
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 0


def test_selected_tables_keep_native_fk_semantics_and_all_parents():
    with sqlite3.connect(':memory:') as connection:
        connection.executescript('''
            CREATE TABLE parent(id TEXT PRIMARY KEY) STRICT;
            CREATE TABLE chosen(value TEXT REFERENCES parent(id)) STRICT;
            CREATE TABLE other(value TEXT REFERENCES parent(id)) STRICT;
            INSERT INTO parent VALUES('kept');
            INSERT INTO chosen VALUES('kept');
            INSERT INTO other VALUES('missing');
        ''')
        assert has_foreign_key_violation(connection) is True
        assert has_foreign_key_violation(connection, table_names=('chosen',)) is False
        connection.execute("INSERT INTO chosen VALUES('missing')")
        assert has_foreign_key_violation(connection, table_names=('chosen',)) is True
        with pytest.raises(ValueError, match='unknown'):
            has_foreign_key_violation(connection, table_names=('not_a_table',))


def test_selected_tables_keep_affinity_and_collation_fallback():
    with sqlite3.connect(':memory:') as connection:
        connection.executescript('''
            CREATE TABLE parent(id TEXT COLLATE NOCASE PRIMARY KEY);
            CREATE TABLE chosen(value TEXT REFERENCES parent(id));
            INSERT INTO parent VALUES('Kept');
            INSERT INTO chosen VALUES('kept');
        ''')
        assert has_foreign_key_violation(connection, table_names=('chosen',)) is False
        connection.execute("INSERT INTO chosen VALUES('missing')")
        assert has_foreign_key_violation(connection, table_names=('chosen',)) is True
