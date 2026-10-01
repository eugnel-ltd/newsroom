from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from newsroom.authority import AuthorityPersistenceError
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority.command_bound_storage import COMMAND_RESULT_PREFIX

from .authority_event_helpers import open_test_system
from .authority_helpers import command, proof


_IMMUTABLE_PAYLOAD_TRIGGER = """CREATE TRIGGER immutable_authority_payloads_update
BEFORE UPDATE ON authority_payloads BEGIN
SELECT RAISE(ABORT,'immutable authority payload'); END"""

_IMMUTABLE_EVENT_TRIGGER = """CREATE TRIGGER immutable_ledger_events_update
BEFORE UPDATE ON ledger_events BEGIN
SELECT RAISE(ABORT,'immutable ledger event'); END"""


def test_reopen_rehashes_exact_retained_payload_bytes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    with open_test_system(database) as system:
        system.commands.execute(command(), proof=proof())

    conn = sqlite3.connect(database)
    try:
        conn.execute("DROP TRIGGER immutable_authority_payloads_update")
        conn.execute(
            "UPDATE authority_payloads SET payload_bytes=?",
            (b'{"count":2,"headline":"Tampered"}',),
        )
        conn.execute(_IMMUTABLE_PAYLOAD_TRIGGER)
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(AuthorityPersistenceError, match="payload digest"):
        open_test_system(database)


def _invalid_event_identity_fixture(tmp_path: Path, *, rebind_result: bool) -> Path:
    database = tmp_path / "authority.sqlite3"
    with open_test_system(database) as system:
        system.commands.execute(command(), proof=proof())

    conn = sqlite3.connect(database)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("DROP TRIGGER immutable_ledger_events_update")
        conn.execute("UPDATE ledger_events SET event_id='not-a-uuid'")
        conn.execute(_IMMUTABLE_EVENT_TRIGGER)
        if rebind_result:
            # Keep the derived-result digest coherent to reach the distinct
            # typed ledger-identity validator, rather than its earlier digest denial.
            event = conn.execute("SELECT * FROM ledger_events").fetchone()
            result = conn.execute("SELECT result_bytes FROM authority_commands").fetchone()
            assert bytes(result[0]).startswith(COMMAND_RESULT_PREFIX)
            value = {key: event[key] for key in (
                "command_id", "aggregate_type", "aggregate_id", "aggregate_version",
                "ledger_seq", "event_id",
            )}
            guard = conn.execute(
                "SELECT sql FROM sqlite_schema WHERE name='immutable_authority_commands_update'"
            ).fetchone()[0]
            conn.execute("DROP TRIGGER immutable_authority_commands_update")
            conn.execute(
                "UPDATE authority_commands SET result_digest=? WHERE command_id=?",
                (digest_bytes(canonical_json_bytes(value)), event["command_id"]),
            )
            conn.execute(guard)
        conn.commit()
    finally:
        conn.close()
    return database


def test_reopen_revalidates_typed_event_identity(tmp_path: Path) -> None:
    database = _invalid_event_identity_fixture(tmp_path, rebind_result=True)
    with pytest.raises(ValueError, match="identifier"):
        open_test_system(database)


def test_reopen_rejects_event_identity_with_unbound_result_digest(tmp_path: Path) -> None:
    database = _invalid_event_identity_fixture(tmp_path, rebind_result=False)
    with pytest.raises(AuthorityPersistenceError, match="command-bound result"):
        open_test_system(database)
