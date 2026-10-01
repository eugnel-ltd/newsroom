from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
import sqlite3

import pytest

from newsroom.authority import AuthorityPersistenceError, canonical_json_bytes
from newsroom.authority._event_store import _EventAuthorityStore
from newsroom.authority._event_store_payload_integrity import _PayloadAndEnvelopeIntegrity
from newsroom.authority.canonical import digest_bytes
from newsroom.authority.command_bound_storage import COMMAND_REQUEST_SELECT, COMMAND_RESULT_SELECT

from .authority_event_helpers import open_test_system
from .authority_helpers import FIXED_NOW, command, make_service, proof


_STREAMED_QUERIES = frozenset(
    {
        "SELECT * FROM payload_schema_contracts",
        "SELECT * FROM command_definitions",
        "SELECT * FROM authentication_contexts",
        COMMAND_REQUEST_SELECT,
        "SELECT * FROM authorization_scope_contents",
        "SELECT d.*,s.canonical_bytes AS selected_scope_canonical_bytes "
        "FROM authorization_decisions d LEFT JOIN authorization_scope_contents s "
        "ON s.scope_content_digest=d.scope_content_digest",
        COMMAND_RESULT_SELECT,
        "SELECT p.*,c.contract_digest AS selected_contract_digest,"
        "c.schema_version AS selected_schema_version,"
        "c.payload_mode AS selected_payload_mode,"
        "c.contract_version AS selected_contract_version,"
        "c.canonicalizer_implementation_version AS selected_canonicalizer_version "
        "FROM authority_payloads p LEFT JOIN payload_schema_contracts c "
        "ON c.contract_digest=p.schema_contract_digest",
        "SELECT * FROM ledger_events ORDER BY ledger_seq",
    }
)

_PER_DECISION_SCOPE_QUERY = (
    "SELECT scope_content_digest,canonical_bytes FROM authorization_scope_contents "
    "WHERE scope_content_digest=?"
)

_PAYLOAD_UPDATE_TRIGGER = """CREATE TRIGGER immutable_authority_payloads_update
BEFORE UPDATE ON authority_payloads BEGIN
SELECT RAISE(ABORT,'immutable authority payload'); END"""

_COMMAND_UPDATE_TRIGGER = """CREATE TRIGGER immutable_authority_commands_update
BEFORE UPDATE ON authority_commands BEGIN
SELECT RAISE(ABORT,'immutable authority command'); END"""


class _StreamingCursor:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    def __iter__(self) -> Iterator[sqlite3.Row]:
        return iter(self._cursor)

    @property
    def description(self):
        return self._cursor.description

    def fetchall(self) -> list[sqlite3.Row]:
        raise AssertionError("full-table integrity validation must stream rows")


class _StreamingConnection:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self.streamed_queries: list[str] = []

    def execute(
        self, sql: str, parameters: tuple[object, ...] = ()
    ) -> sqlite3.Cursor | _StreamingCursor:
        cursor = self._connection.execute(sql, parameters)
        normalised = " ".join(sql.split())
        if parameters and normalised == _PER_DECISION_SCOPE_QUERY:
            raise AssertionError("streamed OPEN cannot look up scopes per decision")
        if parameters and (
            "WHERE c.command_id=?" in normalised
            or normalised.startswith("SELECT * FROM ledger_events WHERE command_id=?")
            or normalised.startswith("SELECT * FROM authority_commands WHERE command_id=?")
        ):
            raise AssertionError("streamed OPEN cannot look up backing per request/result")
        if not parameters and normalised in _STREAMED_QUERIES:
            self.streamed_queries.append(normalised)
            return _StreamingCursor(cursor)
        return cursor


def _store(path: Path, service: object) -> _EventAuthorityStore:
    return _EventAuthorityStore(
        path,
        issuer=service._issuer,  # type: ignore[attr-defined]
        command_registry=service._registry,  # type: ignore[attr-defined]
        payload_schemas=service._payload_schemas,  # type: ignore[attr-defined]
        command_service_version="authority-command-v1",
        clock=lambda: FIXED_NOW,
    )


def test_core_immutable_record_validation_streams_every_full_table(
    tmp_path: Path,
) -> None:
    service = make_service()
    with _store(tmp_path / "authority.sqlite3", service) as store:
        for index in range(3):
            grant = service._authorize_for_commit(  # type: ignore[attr-defined]
                command(key=f"stream-{index}"), proof=proof()
            )
            store.commit(grant)

        connection = _StreamingConnection(store._connection)
        store._validate_immutable_records(connection)  # type: ignore[arg-type]

    assert frozenset(connection.streamed_queries) == _STREAMED_QUERIES


@pytest.mark.parametrize("tamper", ("missing", "mutated"))
def test_streamed_decision_validation_rejects_scope_corruption(
    tmp_path: Path, tamper: str
) -> None:
    service = make_service()
    with _store(tmp_path / "authority.sqlite3", service) as store:
        grant = service._authorize_for_commit(  # type: ignore[attr-defined]
            command(key="stream-corrupt"), proof=proof()
        )
        store.commit(grant)
        connection = store._connection
        if tamper == "missing":
            trigger = connection.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE name='immutable_authorization_decisions_update'"
            ).fetchone()[0]
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute("DROP TRIGGER immutable_authorization_decisions_update")
            connection.execute(
                "UPDATE authorization_decisions SET scope_content_digest=?",
                ("sha256:" + "f" * 64,),
            )
            connection.execute(trigger)
            expected = "effective scopes are missing"
        else:
            trigger = connection.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE name='immutable_authorization_scope_contents_update'"
            ).fetchone()[0]
            row = connection.execute(
                "SELECT scope_content_digest,canonical_bytes "
                "FROM authorization_scope_contents LIMIT 1"
            ).fetchone()
            scopes = json.loads(bytes(row[1]))
            scopes[0] = scopes[0][:-1] + ("x" if scopes[0][-1] != "x" else "y")
            changed = canonical_json_bytes(scopes)
            assert changed != bytes(row[1]) and len(changed) == len(row[1])
            connection.execute(
                "DROP TRIGGER immutable_authorization_scope_contents_update"
            )
            connection.execute(
                "UPDATE authorization_scope_contents SET canonical_bytes=? "
                "WHERE scope_content_digest=?",
                (changed, row[0]),
            )
            connection.execute(trigger)
            expected = "effective scopes digest differs"

        with pytest.raises(AuthorityPersistenceError, match=expected):
            store._validate_immutable_records(  # type: ignore[arg-type]
                _StreamingConnection(connection)
            )


@pytest.mark.parametrize("record_position", (0, 1, 2))
@pytest.mark.parametrize("record_kind", ("payload", "result"))
def test_reopen_streams_and_rejects_tamper_at_every_position(
    tmp_path: Path, record_position: int, record_kind: str
) -> None:
    database = tmp_path / f"authority-{record_kind}-{record_position}.sqlite3"
    with open_test_system(database) as system:
        for index in range(3):
            system.commands.execute(command(key=f"record-{index}"), proof=proof())

    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT c.command_id,c.payload_id FROM authority_commands c "
            "JOIN ledger_events e ON e.command_id=c.command_id "
            "ORDER BY e.ledger_seq"
        ).fetchall()
        command_id, payload_id = rows[record_position]
        if record_kind == "payload":
            connection.execute("DROP TRIGGER immutable_authority_payloads_update")
            connection.execute(
                "UPDATE authority_payloads SET payload_bytes=? WHERE payload_id=?",
                (b'{"count":2,"headline":"tampered"}', payload_id),
            )
            connection.execute(_PAYLOAD_UPDATE_TRIGGER)
        else:
            connection.execute("DROP TRIGGER immutable_authority_commands_update")
            connection.execute(
                "UPDATE authority_commands SET result_bytes=? WHERE command_id=?",
                (b"{}", command_id),
            )
            connection.execute(_COMMAND_UPDATE_TRIGGER)
        connection.commit()
    finally:
        connection.close()

    expected = "payload digest" if record_kind == "payload" else "result digest"
    with pytest.raises(AuthorityPersistenceError, match=expected):
        open_test_system(database)


@pytest.mark.parametrize("payload_count", (3, 11))
def test_streamed_payload_validation_does_not_query_each_schema_contract(
    tmp_path: Path, payload_count: int,
) -> None:
    service = make_service()
    with _store(tmp_path / "payload-contracts.sqlite3", service) as store:
        for index in range(payload_count):
            store.commit(service._authorize_for_commit(
                command(key=f"payload-contract-{index}"), proof=proof(),
            ))
        statements = []
        store._connection.set_trace_callback(statements.append)
        try:
            store._validate_immutable_records(store._connection)
        finally:
            store._connection.set_trace_callback(None)
        lookups = [statement for statement in statements if statement.startswith(
            "SELECT * FROM payload_schema_contracts WHERE contract_digest="
        )]
        assert len(lookups) == 0
        assert store._connection.execute(
            "SELECT COUNT(*) FROM authority_payloads"
        ).fetchone()[0] == payload_count


class _NoopIntegrity:
    def _validate_immutable_records(self, _connection: object) -> None:
        pass


class _PayloadIntegrityProbe(_PayloadAndEnvelopeIntegrity, _NoopIntegrity):
    pass


@pytest.mark.parametrize("streamed", (False, True))
@pytest.mark.parametrize(("changes", "message"), (
    ({"schema_contract_digest": "sha256:" + "f" * 64}, "schema contract is missing"),
    ({"schema_version": "different-v1"}, "immutable schema contract"),
    ({"schema_contract_version": "different-v1"}, "immutable schema contract"),
    ({"canonicalizer_implementation_version": "different-v1"}, "immutable schema contract"),
    ({"mode": "NO_PAYLOAD", "payload_bytes": b"", "payload_digest": digest_bytes(b"")}, "immutable schema contract"),
    ({"payload_bytes": None}, "retained payload bytes are missing"),
    ({"payload_bytes": b"tampered"}, "payload digest"),
    ({"payload_bytes": b"", "payload_digest": digest_bytes(b"")}, "INLINE authority cannot retain an empty payload"),
    ({"mode": "NO_PAYLOAD"}, "NO_PAYLOAD authority must retain exact empty bytes"),
))
def test_joined_and_standalone_payload_validation_reject_late_corruption(
    tmp_path: Path, streamed: bool, changes: dict, message: str,
) -> None:
    service = make_service()
    with _store(tmp_path / "payload-corruption.sqlite3", service) as store:
        for index in range(3):
            store.commit(service._authorize_for_commit(
                command(key=f"late-payload-{index}"), proof=proof(),
            ))
        connection = store._connection
        last_id = connection.execute(
            "SELECT payload_id FROM authority_payloads ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute("DROP TRIGGER immutable_authority_payloads_update")
        connection.execute(
            "UPDATE authority_payloads SET "
            + ",".join(f"{field}=?" for field in changes) + " WHERE payload_id=?",
            (*changes.values(), last_id),
        )
        connection.execute(_PAYLOAD_UPDATE_TRIGGER)
        probe = _PayloadIntegrityProbe()
        with pytest.raises(AuthorityPersistenceError, match=message):
            if streamed:
                probe._validate_immutable_records(_StreamingConnection(connection))
            else:
                row = connection.execute(
                    "SELECT * FROM authority_payloads WHERE payload_id=?", (last_id,),
                ).fetchone()
                probe._validate_payload_record(connection, row)


@pytest.mark.parametrize("field", (
    "selected_schema_version", "selected_payload_mode", "selected_contract_version",
    "selected_canonicalizer_version", "selected_contract_digest",
))
def test_present_joined_contract_does_not_treat_null_fields_as_absent(
    tmp_path: Path, field: str,
) -> None:
    service = make_service()
    with _store(tmp_path / "null-contract-field.sqlite3", service) as store:
        store.commit(service._authorize_for_commit(command(), proof=proof()))
        query = next(query for query in _STREAMED_QUERIES if query.startswith("SELECT p.*"))
        row = store._connection.execute(query).fetchone()
        selected = dict(row)
        selected[field] = "sha256:" + "f" * 64 if field == "selected_contract_digest" else None
        with pytest.raises(AuthorityPersistenceError, match="immutable schema contract"):
            store._validate_payload_record(store._connection, row, selected_contract_row=selected)


def test_standalone_payload_validation_rechecks_contract_on_every_call(tmp_path: Path) -> None:
    service = make_service()
    with _store(tmp_path / "standalone-contract.sqlite3", service) as store:
        store.commit(service._authorize_for_commit(command(), proof=proof()))
        connection = store._connection
        row = connection.execute("SELECT * FROM authority_payloads").fetchone()
        statements = []
        connection.set_trace_callback(statements.append)
        try:
            for _ in range(2):
                store._validate_payload_record(connection, row)
        finally:
            connection.set_trace_callback(None)
        assert sum(statement.startswith(
            "SELECT * FROM payload_schema_contracts WHERE contract_digest="
        ) for statement in statements) == 2
        connection.execute("DROP TRIGGER immutable_payload_schema_contracts_update")
        connection.execute(
            "UPDATE payload_schema_contracts SET contract_version='different-v1' "
            "WHERE contract_digest=?", (row["schema_contract_digest"],),
        )
        with pytest.raises(AuthorityPersistenceError, match="immutable schema contract"):
            store._validate_payload_record(connection, row)
