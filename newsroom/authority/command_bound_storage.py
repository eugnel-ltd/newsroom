"""Lossless command-bound authority request and result storage."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping

from .authorization_request_storage_migrations import (
    AUTHORIZATION_REQUEST_INDEXED_FIELDS,
    authorization_request_bytes_from_v38_row,
    authorization_request_value_from_v38_row,
)
from .canonical import canonical_json_bytes, digest_bytes, digest_canonical
from .persistence import AuthorityPersistenceError

COMMAND_REQUEST_MARKER = b"v41"
COMMAND_RESULT_PREFIX = b"command-result-v1:"
_EVENT_DEFINITION_FIELDS = (
    "aggregate_type", "event_type", "event_schema_version", "payload_mode",
    "payload_schema_version", "payload_schema_contract_version",
    "payload_schema_contract_digest", "payload_canonicalizer_version",
    "trust_scope", "security_scope", "retention_scope",
)
_COMMAND_EVENT_FIELDS = (
    "command_id", "aggregate_type", "aggregate_id", "command_definition_version",
    "command_definition_digest", "authentication_context_id",
    "authorization_request_digest", "authorization_decision_id", "payload_id",
    "producer_version",
)
_RESULT_FIELDS = (
    "command_id", "aggregate_type", "aggregate_id", "aggregate_version",
    "ledger_seq", "event_id",
)
_DERIVED_REQUEST_FIELDS = frozenset((
    *AUTHORIZATION_REQUEST_INDEXED_FIELDS, *_EVENT_DEFINITION_FIELDS,
    "aggregate_id", "stable_semantic_request_digest", "command_definition_digest",
    "object_class", "allowed_use",
))
_BACKING_COLUMNS = {
    "c": _COMMAND_EVENT_FIELDS + ("command_type", "stable_semantic_request_digest"),
    "e": tuple(dict.fromkeys((*_COMMAND_EVENT_FIELDS, *_EVENT_DEFINITION_FIELDS, *_RESULT_FIELDS, "principal_id"))),
    "d": ("definition_digest", "command_type", "definition_version", "payload_schema_contract_digest", "canonical_bytes"),
    "s": ("contract_digest", "schema_version", "payload_mode", "contract_version", "canonicalizer_implementation_version", "canonical_bytes"),
}
_BACKING_SELECT = ",".join(
    f'{alias}.{name} AS "{alias}_{name}"'
    for alias, names in _BACKING_COLUMNS.items() for name in names
)
COMMAND_REQUEST_SELECT = (
    f"SELECT r.*,{_BACKING_SELECT} FROM authorization_requests r "
    "LEFT JOIN authority_commands c ON c.command_id=CASE "
    "WHEN r.storage_request_marker=X'763431' "
    "THEN json_extract(r.storage_request_residual,'$.command_id') END "
    "LEFT JOIN ledger_events e ON e.command_id=c.command_id "
    "LEFT JOIN command_definitions d ON d.definition_digest=c.command_definition_digest "
    "LEFT JOIN payload_schema_contracts s ON s.contract_digest=d.payload_schema_contract_digest"
)
COMMAND_RESULT_SELECT = (
    "SELECT c.command_id,c.aggregate_type,c.aggregate_id,c.result_digest,c.result_bytes,"
    "e.command_id selected_command_id,e.aggregate_type selected_aggregate_type,"
    "e.aggregate_id selected_aggregate_id,e.aggregate_version selected_aggregate_version,"
    "e.ledger_seq selected_ledger_seq,e.event_id selected_event_id "
    "FROM authority_commands c LEFT JOIN ledger_events e ON e.command_id=c.command_id"
)


def _object(data: object) -> dict[str, object]:
    if not isinstance(data, bytes):
        raise ValueError("canonical bytes are required")
    value = json.loads(data.decode("utf-8", errors="strict"))
    if not isinstance(value, dict) or canonical_json_bytes(value) != data:
        raise ValueError("canonical object is required")
    return value


def _equal(left: object, right: object, *, field: str = "backing") -> None:
    if type(left) is not type(right) or left != right:
        raise ValueError(f"command-bound {field} differs")


def _request_fields(request_row, command_row, event_row, definition_row, schema_row,
                    *, verified_definitions=None, verified_schemas=None):
    retained = None if verified_definitions is None else verified_definitions.get(definition_row["definition_digest"])
    if retained is None:
        definition = _object(definition_row["canonical_bytes"])
        _equal(digest_canonical(definition), definition_row["definition_digest"])
    else:
        original, definition = retained
        _equal(definition_row["canonical_bytes"], original, field="command definition bytes")
    for key in ("command_type", "definition_version", "payload_schema_contract_digest"):
        _equal(definition[key], definition_row[key])
    _equal(command_row["command_definition_digest"], definition_row["definition_digest"])
    _equal(command_row["command_definition_version"], definition["definition_version"])
    _equal(command_row["command_type"], definition["command_type"])
    for key in _COMMAND_EVENT_FIELDS:
        _equal(command_row[key], event_row[key], field=key)
    for key in _EVENT_DEFINITION_FIELDS:
        _equal(event_row[key], definition[key], field="payload schema contract" if key.startswith("payload_") else key)
    for request_key, event_key in (
        ("request_digest", "authorization_request_digest"),
        ("authentication_context_id", "authentication_context_id"),
        ("principal_id", "principal_id"),
    ):
        _equal(request_row[request_key], event_row[event_key])
    _equal(request_row["operation_type"], "command:" + definition["command_type"])
    _equal(request_row["required_scope"], definition["required_scope"])
    retained = None if verified_schemas is None else verified_schemas.get(schema_row["contract_digest"])
    if retained is None:
        schema = _object(schema_row["canonical_bytes"])
        _equal(digest_canonical(schema), schema_row["contract_digest"])
    else:
        original, schema = retained
        _equal(schema_row["canonical_bytes"], original, field="payload schema contract bytes")
    _equal(schema_row["contract_digest"], definition["payload_schema_contract_digest"])
    for schema_key, definition_key in (
        ("schema_version", "payload_schema_version"),
        ("payload_mode", "payload_mode"),
        ("contract_version", "payload_schema_contract_version"),
        ("canonicalizer_implementation_version", "payload_canonicalizer_version"),
    ):
        _equal(schema[schema_key], schema_row[schema_key])
        _equal(schema[schema_key], definition[definition_key])
    return {
        **{name: request_row[name] for name in AUTHORIZATION_REQUEST_INDEXED_FIELDS},
        **{name: definition[name] for name in _EVENT_DEFINITION_FIELDS},
        "aggregate_id": command_row["aggregate_id"],
        "stable_semantic_request_digest": command_row["stable_semantic_request_digest"],
        "command_definition_digest": definition_row["definition_digest"],
        "object_class": definition["required_object_class"],
        "allowed_use": definition["required_allowed_use"],
    }


def _authenticate_request(request_row, data, value=None):
    value = _object(data) if value is None else dict(value)
    _equal(digest_bytes(data), request_row["canonical_record_digest"])
    _equal(value.pop("request_digest"), request_row["request_digest"])
    _equal(digest_canonical(value), request_row["request_digest"])


def command_request_storage(value: Mapping[str, object], command_id: str) -> bytes:
    """Encode a verified pending grant; persisted backing is checked before commit."""
    return canonical_json_bytes({
        "storage_version": 1, "command_id": command_id,
        "residual": {name: item for name, item in value.items() if name not in _DERIVED_REQUEST_FIELDS},
    })


def pending_command_request_bytes(row, original: bytes, command_id: str) -> bytes:
    """Check the signed grant's pending row while its ledger FK is not yet present."""
    try:
        _equal(row["storage_request_marker"], COMMAND_REQUEST_MARKER)
        _authenticate_request(row, original)
        _equal(row["storage_request_residual"], command_request_storage(_object(original), command_id))
        return original
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthorityPersistenceError("pending command-bound request differs") from exc


def _row(connection, table: str, key: str, identity: object):
    cursor = connection.execute(f"SELECT * FROM {table} WHERE {key}=?", (identity,))
    row = cursor.fetchone()
    if row is None:
        raise AuthorityPersistenceError("command-bound backing is missing")
    return dict(zip((item[0] for item in cursor.description), row, strict=True))


def command_request_backing(connection: sqlite3.Connection, command_id: str) -> dict:
    """Resolve only the explicit command and its immutable unique/primary-key rows."""
    cursor = connection.execute(
        f"SELECT {_BACKING_SELECT} FROM authority_commands c "
        "LEFT JOIN ledger_events e ON e.command_id=c.command_id "
        "LEFT JOIN command_definitions d ON d.definition_digest=c.command_definition_digest "
        "LEFT JOIN payload_schema_contracts s ON s.contract_digest=d.payload_schema_contract_digest "
        "WHERE c.command_id=?", (command_id,),
    )
    raw = cursor.fetchone()
    if raw is None:
        raise AuthorityPersistenceError("command-bound backing is missing")
    row = dict(zip((item[0] for item in cursor.description), raw, strict=True))
    return selected_command_request_backing(row)


def selected_command_request_backing(row) -> dict:
    """Use the same exact backing fields from a single-row or streamed join."""
    if any(row[key] is None for key in ("c_command_id", "e_command_id", "d_definition_digest", "s_contract_digest")):
        raise AuthorityPersistenceError("command-bound backing is missing")
    return {
        f"{kind}_row": {name: row[f"{alias}_{name}"] for name in _BACKING_COLUMNS[alias]}
        for kind, alias in (("command", "c"), ("event", "e"), ("definition", "d"), ("schema", "s"))
    }


def command_request_rows(connection: sqlite3.Connection):
    """Stream bulk OPEN backing in one snapshot/query, with missing parents visible."""
    try:
        cursor = connection.execute(COMMAND_REQUEST_SELECT)
        names = tuple(item[0] for item in cursor.description)
        for raw in cursor:
            yield dict(zip(names, raw, strict=True))
    except sqlite3.DatabaseError as exc:
        raise AuthorityPersistenceError("stored authorization request representation differs") from exc


def validated_command_result_bytes(connection: sqlite3.Connection, row: Mapping[str, object]) -> bytes:
    """Authenticate either encoding for logical authority/retention comparisons."""
    data = bytes(row["result_bytes"])
    if data.startswith(COMMAND_RESULT_PREFIX):
        return command_result_bytes(connection, data, str(row["result_digest"]), command_id=str(row["command_id"]))
    event = _row(connection, "ledger_events", "command_id", row["command_id"])
    compact_command_result(row, event_row=event)
    return data


def command_request_value(connection: sqlite3.Connection, row: Mapping[str, object],
                          *, selected_backing=None, verified_definitions=None, verified_schemas=None) -> dict:
    """Restore physical storage; the caller authenticates the original request digests."""
    try:
        marker = bytes(row["storage_request_marker"])
        if marker == b"v38":
            return authorization_request_value_from_v38_row(row)
        _equal(marker, COMMAND_REQUEST_MARKER)
        stored = bytes(row["storage_request_residual"])
        command_id = _object(stored)["command_id"]
        if not isinstance(command_id, str):
            raise ValueError("command-bound request identity differs")
        backing = selected_backing if selected_backing is not None else command_request_backing(connection, command_id)
        return _restore_command_request_value(row, stored, **backing,
            verified_definitions=verified_definitions, verified_schemas=verified_schemas)
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthorityPersistenceError("stored command-bound request differs") from exc


def command_request_bytes(connection: sqlite3.Connection, row: Mapping[str, object]) -> bytes:
    """Return exact digest-authenticated bytes from either physical request codec."""
    value = command_request_value(connection, row)
    data = canonical_json_bytes(value)
    try:
        _authenticate_request(row, data, value)
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthorityPersistenceError("stored command-bound request digest differs") from exc
    return data


def command_result_bytes(connection: sqlite3.Connection, data: bytes, digest: str, *, command_id: str | None = None) -> bytes:
    """Restore a result marker; old canonical result bytes retain their old reader."""
    if not data.startswith(COMMAND_RESULT_PREFIX):
        return data
    try:
        stored_command_id = data[len(COMMAND_RESULT_PREFIX):].decode("utf-8", errors="strict")
        if command_id is not None:
            _equal(stored_command_id, command_id)
        command = _row(connection, "authority_commands", "command_id", stored_command_id)
        _equal(command["result_digest"], digest)
        _equal(command["result_bytes"], data)
        event = _row(connection, "ledger_events", "command_id", stored_command_id)
        return restore_command_result(command, data, event_row=event)
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthorityPersistenceError("stored command-bound result differs") from exc


def compact_command_request(
    request_row: Mapping[str, object], *, command_row, event_row, definition_row, schema_row,
) -> bytes:
    """Authenticate an original v38 row and elide only exact derived fields."""
    try:
        original = authorization_request_bytes_from_v38_row(request_row)
        _authenticate_request(request_row, original)
        value = _object(original)
        derived = _request_fields(request_row, command_row, event_row, definition_row, schema_row)
        for name, item in derived.items():
            _equal(value[name], item)
        return canonical_json_bytes({
            "storage_version": 1,
            "command_id": command_row["command_id"],
            "residual": {name: item for name, item in value.items() if name not in derived},
        })
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthorityPersistenceError("command-bound request conversion differs") from exc


def restore_command_request(
    request_row: Mapping[str, object], data: bytes, *, command_row, event_row, definition_row, schema_row,
) -> bytes:
    """Reconstruct exact original bytes; backing failures never use a fallback."""
    value = _restore_command_request_value(request_row, data, command_row=command_row,
        event_row=event_row, definition_row=definition_row, schema_row=schema_row)
    original = canonical_json_bytes(value)
    try:
        _authenticate_request(request_row, original, value)
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthorityPersistenceError("command-bound request digest differs") from exc
    return original


def _restore_command_request_value(
    request_row, data, *, command_row, event_row, definition_row, schema_row,
    verified_definitions=None, verified_schemas=None,
):
    try:
        stored = _object(data)
        if set(stored) != {"storage_version", "command_id", "residual"}:
            raise ValueError("command-bound request shape differs")
        _equal(stored["storage_version"], 1)
        _equal(stored["command_id"], command_row["command_id"])
        derived = _request_fields(request_row, command_row, event_row, definition_row, schema_row,
            verified_definitions=verified_definitions, verified_schemas=verified_schemas)
        residual = stored["residual"]
        if not isinstance(residual, dict) or set(residual).intersection(derived):
            raise ValueError("command-bound request residual differs")
        return {**residual, **derived}
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthorityPersistenceError(f"command-bound request representation differs: {exc}") from exc


def _result_bytes(command_row, event_row):
    for key in ("command_id", "aggregate_type", "aggregate_id"):
        _equal(command_row[key], event_row[key])
    original = canonical_json_bytes({name: event_row[name] for name in _RESULT_FIELDS})
    _equal(digest_bytes(original), command_row["result_digest"])
    return original


def compact_command_result(command_row: Mapping[str, object], *, event_row) -> bytes:
    """Authenticate the exact six-field result before replacing its encoding."""
    try:
        _equal(command_row["result_bytes"], _result_bytes(command_row, event_row))
        return COMMAND_RESULT_PREFIX + command_row["command_id"].encode("utf-8")
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthorityPersistenceError("command-bound result conversion differs") from exc


def restore_command_result(command_row: Mapping[str, object], data: bytes, *, event_row) -> bytes:
    """Return the original digest-bound result, not the physical marker bytes."""
    try:
        _equal(data, COMMAND_RESULT_PREFIX + command_row["command_id"].encode("utf-8"))
        return _result_bytes(command_row, event_row)
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthorityPersistenceError("command-bound result representation differs") from exc
