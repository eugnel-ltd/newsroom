from __future__ import annotations

import copy
import json
import sqlite3

import pytest

from newsroom.authority import AuthorityPersistenceError, canonical_json_bytes
from newsroom.authority._event_store import _EventAuthorityStore
from newsroom.authority.canonical import digest_bytes, digest_canonical
from newsroom.authority.command_bound_storage import (
    compact_command_request,
    compact_command_result,
    restore_command_request,
    restore_command_result,
)

from newsroom.tests.authority_event_helpers import open_test_system
from newsroom.tests.authority_helpers import command, proof


@pytest.fixture(scope="module")
def retained(tmp_path_factory):
    path = tmp_path_factory.mktemp("command-elision") / "fixture.sqlite3"
    semantic_command = command(key="command-elision")
    with open_test_system(path) as system:
        result = system.commands.execute(semantic_command, proof=proof())
        provenance = system.events.provenance(result.event_id, proof=proof())
        assert system.commands.execute(semantic_command, proof=proof()).replayed
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        tables = {
            "request": "authorization_requests",
            "command": "authority_commands",
            "event": "ledger_events",
            "definition": "command_definitions",
            "schema": "payload_schema_contracts",
        }
        rows = {
            name: dict(connection.execute(f"SELECT * FROM {table}").fetchone())
            for name, table in tables.items()
        }
    # Exercise conversion of the retained old v38 representation, even after
    # the production writer starts emitting compact command-bound rows.
    value = json.loads(provenance.authorization_request.canonical_bytes)
    from newsroom.authority.authorization_request_storage_migrations import AUTHORIZATION_REQUEST_INDEXED_FIELDS
    rows["request"]["storage_request_marker"] = b"v38"
    rows["request"]["storage_request_residual"] = canonical_json_bytes({
        key: item for key, item in value.items() if key not in AUTHORIZATION_REQUEST_INDEXED_FIELDS
    })
    rows["command"]["result_bytes"] = canonical_json_bytes({
        key: rows["event"][key] for key in ("command_id", "aggregate_type", "aggregate_id", "aggregate_version", "ledger_seq", "event_id")
    })
    return rows, result, provenance


def backing(rows):
    return {f"{name}_row": rows[name] for name in ("command", "event", "definition", "schema")}


def test_compact_request_and_result_preserve_exact_bytes_digests_and_replay(retained):
    rows, committed, provenance = retained
    request = compact_command_request(rows["request"], **backing(rows))
    result = compact_command_result(rows["command"], event_row=rows["event"])
    request_bytes = restore_command_request(rows["request"], request, **backing(rows))
    result_bytes = restore_command_result(rows["command"], result, event_row=rows["event"])
    assert request_bytes == provenance.authorization_request.canonical_bytes
    assert digest_bytes(request_bytes) == rows["request"]["canonical_record_digest"]
    assert result_bytes == rows["command"]["result_bytes"]
    assert digest_bytes(result_bytes) == committed.result_digest
    assert len(request) < len(rows["request"]["storage_request_residual"])
    assert len(result) < len(result_bytes)
    reader = object.__new__(_EventAuthorityStore)
    replay = reader._decode_result(result_bytes, committed.result_digest, replayed=True)
    assert replay == reader._decode_result(rows["command"]["result_bytes"], committed.result_digest, replayed=True)
    assert replay.replayed


def test_request_extensions_are_retained_not_assumed_or_discarded(retained):
    rows = copy.deepcopy(retained[0])
    value = json.loads(retained[2].authorization_request.canonical_bytes)
    value["future_extension"] = {"unicode": "證據", "nullable": None, "items": [1, True]}
    value.pop("request_digest")
    value["request_digest"] = digest_canonical(value)
    rows["request"]["request_digest"] = value["request_digest"]
    rows["request"]["canonical_record_digest"] = digest_canonical(value)
    residual = json.loads(rows["request"]["storage_request_residual"])
    residual["future_extension"] = value["future_extension"]
    rows["request"]["storage_request_residual"] = canonical_json_bytes(residual)
    for name in ("command", "event"):
        rows[name]["authorization_request_digest"] = value["request_digest"]
    compact = compact_command_request(rows["request"], **backing(rows))
    assert restore_command_request(rows["request"], compact, **backing(rows)) == canonical_json_bytes(value)


@pytest.mark.parametrize("missing", ("command", "event", "definition", "schema"))
def test_missing_backing_fails_closed_for_encoding_and_reading(retained, missing):
    rows = copy.deepcopy(retained[0])
    compact = compact_command_request(rows["request"], **backing(rows))
    rows[missing] = None
    with pytest.raises(AuthorityPersistenceError):
        compact_command_request(rows["request"], **backing(rows))
    with pytest.raises(AuthorityPersistenceError):
        restore_command_request(rows["request"], compact, **backing(rows))


@pytest.mark.parametrize(("table", "field"), (
    ("command", "command_id"), ("command", "command_type"),
    ("command", "aggregate_id"), ("command", "stable_semantic_request_digest"),
    ("command", "authorization_request_digest"), ("command", "authentication_context_id"),
    ("command", "command_definition_version"), ("command", "command_definition_digest"),
    ("event", "command_id"), ("event", "event_schema_version"),
    ("event", "aggregate_type"), ("event", "principal_id"),
    ("event", "payload_schema_version"), ("event", "security_scope"),
    ("event", "trust_scope"), ("definition", "canonical_bytes"),
    ("definition", "definition_version"), ("schema", "canonical_bytes"),
    ("schema", "schema_version"), ("request", "principal_id"),
    ("request", "canonical_record_digest"), ("request", "request_digest"),
))
def test_mutated_backing_or_request_fails_closed(retained, table, field):
    rows = copy.deepcopy(retained[0])
    compact = compact_command_request(rows["request"], **backing(rows))
    original = rows[table][field]
    rows[table][field] = original + 1 if type(original) is int else b"{}" if type(original) is bytes else "changed"
    with pytest.raises(AuthorityPersistenceError):
        restore_command_request(rows["request"], compact, **backing(rows))
    with pytest.raises(AuthorityPersistenceError):
        compact_command_request(rows["request"], **backing(rows))


@pytest.mark.parametrize("tamper", ("format", "command", "residual", "collision"))
def test_compact_request_encoding_is_strict(retained, tamper):
    rows = retained[0]
    value = json.loads(compact_command_request(rows["request"], **backing(rows)))
    if tamper == "format":
        value["storage_version"] = 2
    elif tamper == "command":
        value["command_id"] = "other-command"
    elif tamper == "residual":
        value["residual"] = []
    else:
        value["residual"]["aggregate_id"] = rows["command"]["aggregate_id"]
    with pytest.raises(AuthorityPersistenceError):
        restore_command_request(rows["request"], canonical_json_bytes(value), **backing(rows))


def test_noncommand_request_is_not_relabelled_as_command_bound(retained):
    rows = copy.deepcopy(retained[0])
    rows["request"]["operation_type"] = "evaluation:read"
    original = copy.deepcopy(rows)
    with pytest.raises(AuthorityPersistenceError):
        compact_command_request(rows["request"], **backing(rows))
    assert rows == original


@pytest.mark.parametrize("field", ("command_id", "aggregate_id", "aggregate_type", "event_id", "aggregate_version", "ledger_seq"))
def test_result_reconstruction_rejects_mutated_ledger(retained, field):
    rows = copy.deepcopy(retained[0])
    compact = compact_command_result(rows["command"], event_row=rows["event"])
    original = rows["event"][field]
    rows["event"][field] = original + 1 if type(original) is int else "changed"
    with pytest.raises(AuthorityPersistenceError):
        restore_command_result(rows["command"], compact, event_row=rows["event"])
    with pytest.raises(AuthorityPersistenceError):
        compact_command_result(rows["command"], event_row=rows["event"])


def test_result_reconstruction_rejects_missing_event_wrong_marker_and_digest(retained):
    rows = copy.deepcopy(retained[0])
    compact = compact_command_result(rows["command"], event_row=rows["event"])
    with pytest.raises(AuthorityPersistenceError):
        restore_command_result(rows["command"], compact, event_row=None)
    with pytest.raises(AuthorityPersistenceError):
        restore_command_result(rows["command"], b"wrong", event_row=rows["event"])
    rows["command"]["result_digest"] = "sha256:" + "f" * 64
    with pytest.raises(AuthorityPersistenceError):
        restore_command_result(rows["command"], compact, event_row=rows["event"])


@pytest.mark.parametrize("data", (b"{ }", b"[]", b"{\"storage_version\":true}", b"\xff"))
def test_noncanonical_compact_request_is_rejected(retained, data):
    rows = retained[0]
    with pytest.raises(AuthorityPersistenceError):
        restore_command_request(rows["request"], data, **backing(rows))


def test_coherently_changed_definition_still_breaks_original_request_binding(retained):
    rows = copy.deepcopy(retained[0])
    compact = compact_command_request(rows["request"], **backing(rows))
    definition = json.loads(rows["definition"]["canonical_bytes"])
    definition["required_scope"] = "authority.changed.write"
    definition_digest = digest_canonical(definition)
    rows["definition"].update(canonical_bytes=canonical_json_bytes(definition), definition_digest=definition_digest)
    rows["request"]["required_scope"] = definition["required_scope"]
    for table in ("command", "event"):
        rows[table]["command_definition_digest"] = definition_digest
    with pytest.raises(AuthorityPersistenceError):
        restore_command_request(rows["request"], compact, **backing(rows))


def test_result_with_extra_semantic_field_is_not_compacted(retained):
    rows = copy.deepcopy(retained[0])
    original = json.loads(rows["command"]["result_bytes"])
    original["future_semantic_field"] = "retain"
    rows["command"]["result_bytes"] = canonical_json_bytes(original)
    rows["command"]["result_digest"] = digest_canonical(original)
    with pytest.raises(AuthorityPersistenceError):
        compact_command_result(rows["command"], event_row=rows["event"])
