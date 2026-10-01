"""Atomic v41 lossless command-bound request and result storage."""

from __future__ import annotations

import sqlite3

from .authorisation_scope_content_migrations import AuthorisationScopeContentMigrationRecord
from .canonical import digest_canonical
from .command_bound_storage import (
    COMMAND_REQUEST_MARKER, command_request_backing, command_request_bytes,
    compact_command_request, compact_command_result,
)
from .persistence import AuthorityPersistenceError

COMMAND_BOUND_STORAGE_SCHEMA_VERSION = 41
COMMAND_BOUND_STORAGE_MIGRATION_NAME = "command_bound_request_result_storage_v41"
COMMAND_BOUND_STORAGE_PREDECESSOR_FINGERPRINT = "sha256:1e6a22fbc1b755d1eebd957378dc691b4e41e0ca405f2ef00d7ddcbd629206c7"
REQUEST_STORAGE_GUARD = """CREATE TRIGGER authorization_request_storage_guard
    BEFORE INSERT ON authorization_requests
    WHEN NEW.storage_request_marker NOT IN (X'763338',X'763431')
    BEGIN SELECT RAISE(ABORT,'authorization request storage differs'); END"""
REQUEST_STORAGE_COLUMN = """ALTER TABLE authorization_requests ADD COLUMN
    storage_request_marker BLOB NOT NULL DEFAULT X'763338'
    CHECK(storage_request_marker IN (X'763338',X'763431'))"""


def _rows(connection, table, *, after_rowid=0):
    cursor = connection.execute(
        f"SELECT rowid,* FROM {table} WHERE rowid>? ORDER BY rowid LIMIT 256", (after_rowid,),
    )
    names = tuple(item[0] for item in cursor.description)
    for row in cursor:
        yield dict(zip(names, row, strict=True))


def _commands(connection):
    after_rowid = 0
    while True:
        seen = False
        for row in _rows(connection, "authority_commands", after_rowid=after_rowid):
            yield row
            after_rowid = int(row["rowid"])
            seen = True
        if not seen:
            return


def _request(connection, digest):
    cursor = connection.execute("SELECT * FROM authorization_requests WHERE request_digest=?", (digest,))
    row = cursor.fetchone()
    if row is None:
        raise AuthorityPersistenceError("command request is missing")
    return dict(zip((item[0] for item in cursor.description), row, strict=True))


def migrate_command_bound_storage(connection: sqlite3.Connection, *, expected_history) -> None:
    from .migrations import schema_fingerprint

    if not connection.in_transaction:
        raise sqlite3.DatabaseError("v41 migration requires an active transaction")
    if (
        connection.execute("PRAGMA user_version").fetchone()[0] not in (0, *range(34, 41))
        or schema_fingerprint(connection) != COMMAND_BOUND_STORAGE_PREDECESSOR_FINGERPRINT
        or tuple(tuple(row) for row in connection.execute(
            "SELECT version,name,checksum FROM authority_migrations ORDER BY version"
        )) != expected_history
    ):
        raise sqlite3.DatabaseError("v41 migration requires exact checked schema v40")

    try:
        # Authenticate standalone/evaluation rows as well as the command stream.
        after_rowid = 0
        while True:
            seen = False
            for row in _rows(connection, "authorization_requests", after_rowid=after_rowid):
                command_request_bytes(connection, row)
                after_rowid = int(row["rowid"])
                seen = True
            if not seen:
                break
        for command in _commands(connection):
            backing = command_request_backing(connection, command["command_id"])
            request = _request(connection, command["authorization_request_digest"])
            compact_command_request(request, **backing)
            compact_command_result(command, event_row=backing["event_row"])

        guards = tuple(connection.execute(
            "SELECT sql FROM sqlite_schema WHERE name IN "
            "('immutable_authorization_requests_update','immutable_authority_commands_update') ORDER BY name"
        ))
        if len(guards) != 2:
            raise sqlite3.IntegrityError("v41 immutable storage guards are missing")
        connection.execute("DROP TRIGGER authorization_request_storage_guard")
        connection.execute("DROP TRIGGER immutable_authorization_requests_update")
        connection.execute("DROP TRIGGER immutable_authority_commands_update")
        connection.execute("ALTER TABLE authorization_requests DROP COLUMN storage_request_marker")
        connection.execute(REQUEST_STORAGE_COLUMN)
        for command in _commands(connection):
            backing = command_request_backing(connection, command["command_id"])
            request = _request(connection, command["authorization_request_digest"])
            if bytes(request["storage_request_marker"]) == b"v38":
                stored = compact_command_request(request, **backing)
                connection.execute(
                    "UPDATE authorization_requests SET storage_request_residual=?,storage_request_marker=? WHERE request_digest=?",
                    (stored, COMMAND_REQUEST_MARKER, request["request_digest"]),
                )
            else:
                command_request_bytes(connection, request)
            stored_result = compact_command_result(command, event_row=backing["event_row"])
            connection.execute("UPDATE authority_commands SET result_bytes=? WHERE command_id=?", (stored_result, command["command_id"]))
        for (guard,) in guards:
            connection.execute(guard)
        connection.execute(REQUEST_STORAGE_GUARD)
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise sqlite3.IntegrityError("v41 retained foreign-key integrity differs")
    except AuthorityPersistenceError as exc:
        raise sqlite3.IntegrityError("stored command authority differs before v41 conversion") from exc


COMMAND_BOUND_STORAGE_MIGRATION_STATEMENTS = (
    "validate exact v40 schema, history and original requests/results before mutation",
    "extend the request marker CHECK to v38 and v41 without changing original indexed keys or foreign keys",
    "keyset-elide only command-bound request fields and exact six-field results; preserve original digests and standalone v38 requests",
    "restore immutable/storage guards and require retained foreign-key integrity",
)
COMMAND_BOUND_STORAGE_MIGRATION_CHECKSUM = digest_canonical({
    "version": COMMAND_BOUND_STORAGE_SCHEMA_VERSION,
    "name": COMMAND_BOUND_STORAGE_MIGRATION_NAME,
    "statements": list(COMMAND_BOUND_STORAGE_MIGRATION_STATEMENTS),
})
COMMAND_BOUND_STORAGE_MIGRATION = AuthorisationScopeContentMigrationRecord(
    COMMAND_BOUND_STORAGE_SCHEMA_VERSION, COMMAND_BOUND_STORAGE_MIGRATION_NAME,
    COMMAND_BOUND_STORAGE_MIGRATION_CHECKSUM,
)
