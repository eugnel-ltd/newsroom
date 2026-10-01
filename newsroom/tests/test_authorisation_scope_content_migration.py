from __future__ import annotations

import json
import sqlite3

import pytest

from newsroom.authority import AuthorityPersistenceError, canonical_json_bytes
from newsroom.authority import authorisation_scope_content_migrations as scope_migration
from newsroom.authority.canonical import digest_bytes
from newsroom.authority.migrations import SCHEMA_VERSION, apply_pending_migrations, schema_fingerprint

from .authority_event_helpers import open_test_system
from .authority_helpers import command, proof
from .graphiti_adapter_4d_migration_helpers import (
    _drop_v36_shared_scope_schema,
    _drop_v37_security_record_schema,
)


def _v35_path(tmp_path):
    path = tmp_path / "authority.sqlite3"
    with open_test_system(path) as system:
        first = system.commands.execute(command(key="scope-one"), proof=proof())
        second = system.commands.execute(command(key="scope-two"), proof=proof())
        originals = tuple(
            (
                item.event_id,
                system.events.provenance(
                    item.event_id, proof=proof()
                ).authorization_decision,
            )
            for item in (first, second)
        )
    with sqlite3.connect(path) as connection:
        _drop_v36_shared_scope_schema(connection)
        connection.commit()
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 35
    return path, originals


def _trigger_sql(connection: sqlite3.Connection, name: str) -> str:
    row = connection.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type='trigger' AND name=?",
        (name,),
    ).fetchone()
    assert row is not None
    return str(row[0])


def _v35_state(connection: sqlite3.Connection) -> tuple[object, ...]:
    return (
        connection.execute("PRAGMA user_version").fetchone()[0],
        schema_fingerprint(connection),
        tuple(connection.execute(
            "SELECT version,name,checksum,applied_at FROM authority_migrations "
            "ORDER BY version"
        )),
        tuple(connection.execute(
            "SELECT * FROM authorization_decisions ORDER BY authorization_decision_id"
        )),
    )


def test_v35_decisions_migrate_to_one_lossless_scope_content(tmp_path) -> None:
    path, originals = _v35_path(tmp_path)
    with open_test_system(path) as system:
        rebuilt = tuple(
            system.events.provenance(event_id, proof=proof()).authorization_decision
            for event_id, _item in originals
        )
    assert tuple(item.canonical_bytes for item in rebuilt) == tuple(
        item.canonical_bytes for _event_id, item in originals
    )
    assert tuple(item.canonical_digest for item in rebuilt) == tuple(
        item.canonical_digest for _event_id, item in originals
    )
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert connection.execute(
            "SELECT count(*) FROM authorization_scope_contents"
        ).fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("tamper", ("same_count_scope", "noncanonical_decision", "indexed_reason"))
def test_v35_conversion_rejects_contradiction_and_rolls_back(tmp_path, tamper: str) -> None:
    path, _ = _v35_path(tmp_path)
    with sqlite3.connect(path) as connection:
        fingerprint = schema_fingerprint(connection)
        trigger = _trigger_sql(connection, "immutable_authorization_decisions_update")
        connection.execute("DROP TRIGGER immutable_authorization_decisions_update")
        if tamper == "same_count_scope":
            raw = bytes(connection.execute(
                "SELECT effective_scopes FROM authorization_decisions LIMIT 1"
            ).fetchone()[0])
            scopes = json.loads(raw)
            assert isinstance(scopes, list) and scopes and isinstance(scopes[0], str)
            scopes[0] = scopes[0][:-1] + ("x" if scopes[0][-1] != "x" else "y")
            changed = canonical_json_bytes(scopes)
            assert changed != raw and len(changed) == len(raw)
            connection.execute(
                "UPDATE authorization_decisions SET effective_scopes=? "
                "WHERE rowid=(SELECT min(rowid) FROM authorization_decisions)",
                (changed,),
            )
        elif tamper == "noncanonical_decision":
            raw = bytes(connection.execute(
                "SELECT canonical_bytes FROM authorization_decisions LIMIT 1"
            ).fetchone()[0])
            assert raw != b"{ }"
            connection.execute(
                "UPDATE authorization_decisions SET canonical_bytes=X'7B207D' "
                "WHERE rowid=(SELECT min(rowid) FROM authorization_decisions)"
            )
        else:
            old_reason = connection.execute(
                "SELECT reason_code FROM authorization_decisions LIMIT 1"
            ).fetchone()[0]
            assert old_reason != "CHANGED"
            connection.execute(
                "UPDATE authorization_decisions SET reason_code='CHANGED' "
                "WHERE rowid=(SELECT min(rowid) FROM authorization_decisions)"
            )
        connection.execute(trigger)
        assert schema_fingerprint(connection) == fingerprint
        connection.commit()
        before = _v35_state(connection)
    with pytest.raises((sqlite3.IntegrityError, sqlite3.DatabaseError)):
        with open_test_system(path):
            pass
    with sqlite3.connect(path) as connection:
        assert _v35_state(connection) == before
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 35
        assert connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE name='authorization_scope_contents'"
        ).fetchone() is None


@pytest.mark.parametrize("numeric_allowed", (0, 1))
def test_v35_conversion_rejects_numeric_json_allowed_and_preserves_v35(
    tmp_path, numeric_allowed: int
) -> None:
    path, _ = _v35_path(tmp_path)
    with sqlite3.connect(path) as connection:
        fingerprint = schema_fingerprint(connection)
        trigger = _trigger_sql(connection, "immutable_authorization_decisions_update")
        row = connection.execute(
            "SELECT rowid,canonical_bytes FROM authorization_decisions LIMIT 1"
        ).fetchone()
        value = json.loads(bytes(row[1]))
        assert isinstance(value["allowed"], bool)
        value["allowed"] = numeric_allowed
        changed = canonical_json_bytes(value)
        assert changed != bytes(row[1])
        connection.execute("DROP TRIGGER immutable_authorization_decisions_update")
        connection.execute(
            "UPDATE authorization_decisions SET allowed=?,canonical_bytes=?,canonical_digest=? "
            "WHERE rowid=?",
            (numeric_allowed, changed, digest_bytes(changed), row[0]),
        )
        connection.execute(trigger)
        assert schema_fingerprint(connection) == fingerprint
        connection.commit()
        before = _v35_state(connection)
        history = tuple(item[:3] for item in before[2])
        connection.execute("BEGIN EXCLUSIVE")
        with pytest.raises(sqlite3.IntegrityError):
            scope_migration.migrate_authorisation_scope_content(
                connection, expected_history=history
            )
        connection.rollback()
        assert _v35_state(connection) == before


def test_v35_conversion_rolls_back_after_first_converted_decision(
    tmp_path, monkeypatch
) -> None:
    path, _ = _v35_path(tmp_path)
    with sqlite3.connect(path) as connection:
        before = _v35_state(connection)

    original_decode = scope_migration._decode_old_decision
    calls = 0

    def fail_during_second_conversion(row):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise sqlite3.IntegrityError("injected after first converted decision")
        return original_decode(row)

    monkeypatch.setattr(
        scope_migration, "_decode_old_decision", fail_during_second_conversion
    )
    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="injected after first"):
            apply_pending_migrations(
                connection, applied_at="2026-09-12T00:00:00.000000Z"
            )
        assert calls == 4
        assert _v35_state(connection) == before
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name='authorization_scope_contents'"
        ).fetchone() is None


def test_v35_conversion_requires_callers_atomic_transaction(tmp_path) -> None:
    path, _ = _v35_path(tmp_path)
    with sqlite3.connect(path) as connection:
        before = _v35_state(connection)
        history = tuple(item[:3] for item in before[2])
        with pytest.raises(sqlite3.DatabaseError, match="active transaction"):
            scope_migration.migrate_authorisation_scope_content(
                connection, expected_history=history
            )
        assert _v35_state(connection) == before


@pytest.mark.parametrize("tamper", ("shared_content", "retarget", "orphan"))
def test_reopen_rejects_shared_scope_corruption(tmp_path, tamper: str) -> None:
    path = tmp_path / "authority.sqlite3"
    with open_test_system(path) as system:
        first = system.commands.execute(command(key="scope-one"), proof=proof())
        system.commands.execute(command(key="scope-two"), proof=proof())
        before = system.events.provenance(first.event_id, proof=proof()).authorization_decision
    with sqlite3.connect(path) as connection:
        fingerprint = schema_fingerprint(connection)
        if tamper == "shared_content":
            trigger = _trigger_sql(
                connection, "immutable_authorization_scope_contents_update"
            )
            connection.execute("DROP TRIGGER immutable_authorization_scope_contents_update")
            row = connection.execute(
                "SELECT scope_content_digest,canonical_bytes "
                "FROM authorization_scope_contents LIMIT 1"
            ).fetchone()
            changed = bytes(row[1]).replace(b"admitted", b"admittxd", 1)
            assert len(changed) == len(row[1]) and changed != row[1]
            connection.execute(
                "UPDATE authorization_scope_contents SET canonical_bytes=? "
                "WHERE scope_content_digest=?", (changed, row[0]),
            )
            connection.execute(trigger)
        else:
            decision_id = str(before.authorization_decision_id)
            trigger = _trigger_sql(connection, "immutable_authorization_decisions_update")
            connection.execute("DROP TRIGGER immutable_authorization_decisions_update")
            target = "sha256:" + ("f" if tamper == "retarget" else "e") * 64
            if tamper == "retarget":
                content = canonical_json_bytes(["authority.different"])
                target = digest_bytes(content)
                connection.execute(
                    "INSERT INTO authorization_scope_contents VALUES(?,?)",
                    (target, content),
                )
            else:
                connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute(
                "UPDATE authorization_decisions SET scope_content_digest=? "
                "WHERE authorization_decision_id=?", (target, decision_id),
            )
            connection.execute(trigger)
        assert schema_fingerprint(connection) == fingerprint
    with pytest.raises((AuthorityPersistenceError, sqlite3.IntegrityError, sqlite3.DatabaseError)):
        with open_test_system(path):
            pass


def test_v36_fixture_downgrade_preserves_references_inside_rollback(tmp_path) -> None:
    path = tmp_path / "nested-downgrade.sqlite3"
    with open_test_system(path) as system:
        system.commands.execute(command(key="retained-downgrade"), proof=proof())
    with sqlite3.connect(path, isolation_level=None) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        # This inverse/rollback contract starts at v36, not the latest codec.
        _drop_v37_security_record_schema(connection)
        assert connection.execute("PRAGMA user_version").fetchone() == (36,)
        before = _v35_state(connection)
        retained_ids = tuple(connection.execute(
            "SELECT authorization_decision_id,canonical_digest "
            "FROM authorization_decisions ORDER BY authorization_decision_id"
        ))
        commands = tuple(connection.execute("SELECT * FROM authority_commands"))
        connection.execute("SAVEPOINT caller_downgrade")
        _drop_v36_shared_scope_schema(connection)
        assert connection.in_transaction
        assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
        assert connection.execute("PRAGMA user_version").fetchone() == (35,)
        assert schema_fingerprint(connection) == (
            scope_migration.AUTHORISATION_SCOPE_CONTENT_PREDECESSOR_FINGERPRINT
        )
        assert tuple(connection.execute(
            "SELECT authorization_decision_id,canonical_digest "
            "FROM authorization_decisions ORDER BY authorization_decision_id"
        )) == retained_ids
        assert tuple(connection.execute("SELECT * FROM authority_commands")) == commands
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        connection.execute("ROLLBACK TO SAVEPOINT caller_downgrade")
        connection.execute("RELEASE SAVEPOINT caller_downgrade")
        assert not connection.in_transaction
        assert _v35_state(connection) == before
