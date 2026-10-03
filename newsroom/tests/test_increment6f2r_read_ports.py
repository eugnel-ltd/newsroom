from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pytest

from newsroom.authority.migrations import apply_pending_migrations
from newsroom.authority.story_candidate_system import (
    _create_story_candidate_read_port,
)
from newsroom.increment6.candidates import (
    CandidateContractError,
    StoryCandidate,
    StoryCandidateReadPort,
    StoryCandidateVersion,
)
from newsroom.increment6.handoffs import (
    Acknowledgement,
    AcknowledgementOutcome,
    EvaluationHandoffReadPort,
    EvaluationHandoffStore,
    HandoffContractError,
    _create_evaluation_handoff_read_port,
    create_handoff,
)
from newsroom.tests import test_increment6e2_candidate_store as candidate_fixture


def _checked_connection(database: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def test_read_port_types_are_token_gated_and_expose_only_reads() -> None:
    with pytest.raises(CandidateContractError, match="authority-private"):
        StoryCandidateReadPort(object(), object())
    with pytest.raises(HandoffContractError, match="authority-private"):
        EvaluationHandoffReadPort(object(), object())

    candidate_names = {
        name
        for name, value in inspect.getmembers(StoryCandidateReadPort)
        if callable(value) and not name.startswith("_")
    }
    handoff_names = {
        name
        for name, value in inspect.getmembers(EvaluationHandoffReadPort)
        if callable(value) and not name.startswith("_")
    }
    assert candidate_names == {
        "require_current_head_in_transaction",
        "require_retained_candidate_in_transaction",
        "require_retained_version",
        "require_retained_version_in_transaction",
        "require_retained_versions_in_transaction",
        "verify_retained_integrity_in_transaction",
    }
    assert handoff_names == {
        "require_retained_handoff_in_transaction",
        "verify_retained_integrity_in_transaction",
    }


def test_handoff_port_reads_exact_aggregate_without_owning_transaction(
    tmp_path: Path,
) -> None:
    database = tmp_path / "handoff.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    connection.execute("PRAGMA foreign_keys=ON")
    apply_pending_migrations(connection, applied_at="2042-01-01T00:00:00.000000Z")
    handoff = create_handoff(
        "candidate-version:fixture",
        "sha256:" + "a" * 64,
        "evaluation-sink:fixture",
    )
    store = EvaluationHandoffStore(connection)
    retained = store.register(handoff)
    retained = store.persist_attempt(retained.handoff_id)
    retained = store.mark_attempt_sent(
        retained.handoff_id, retained.attempts[0].attempt_id
    )
    acknowledgement = Acknowledgement.create(
        handoff_id=retained.handoff_id,
        attempt_id=retained.attempts[0].attempt_id,
        candidate_version_id=retained.candidate_version_id,
        governing_manifest_digest=retained.governing_manifest_digest,
        sink_id=retained.sink_id,
        outcome=AcknowledgementOutcome.ACKNOWLEDGED,
        response_digest="sha256:" + "b" * 64,
    )
    retained = store.correlate_acknowledgement(retained.handoff_id, acknowledgement)
    connection.close()
    connection = _checked_connection(database)
    port = _create_evaluation_handoff_read_port(connection)

    with pytest.raises(HandoffContractError, match="active checked connection"):
        port.require_retained_handoff_in_transaction(handoff.handoff_id)

    connection.execute("BEGIN IMMEDIATE")
    assert port.require_retained_handoff_in_transaction(handoff.handoff_id) == retained
    with pytest.raises(HandoffContractError, match="unknown Handoff"):
        port.require_retained_handoff_in_transaction("handoff:sha256:" + "0" * 64)
    assert connection.in_transaction
    connection.execute("ROLLBACK")
    assert not connection.in_transaction
    connection.close()


@pytest.mark.parametrize(
    ("table", "column", "value"),
    (
        ("evaluation_handoffs", "schema_identity", "wrong"),
        ("evaluation_handoffs", "transport_state", "acknowledged"),
    ),
)
def test_handoff_port_fails_closed_on_retained_tamper(
    tmp_path: Path, table: str, column: str, value: str
) -> None:
    database = tmp_path / "handoff-tamper.sqlite3"
    connection = _checked_connection(database)
    apply_pending_migrations(connection, applied_at="2042-01-01T00:00:00.000000Z")
    handoff = EvaluationHandoffStore(connection).register(
        create_handoff(
            "candidate-version:fixture",
            "sha256:" + "a" * 64,
            "evaluation-sink:fixture",
        )
    )
    port = _create_evaluation_handoff_read_port(connection)
    connection.execute("PRAGMA ignore_check_constraints=ON")
    connection.execute("DROP TRIGGER evaluation_handoff_identity_guard")
    connection.execute("DROP TRIGGER evaluation_handoff_state_guard")
    connection.execute(
        f"UPDATE {table} SET {column}=? WHERE handoff_id=?", (value, handoff.handoff_id)
    )
    connection.execute("BEGIN IMMEDIATE")
    with pytest.raises(HandoffContractError):
        port.require_retained_handoff_in_transaction(handoff.handoff_id)
    assert connection.in_transaction
    connection.execute("ROLLBACK")
    connection.close()


@pytest.mark.parametrize("damage", ("missing", "altered"))
def test_handoff_port_rejects_missing_or_altered_v17_guard_without_effects(
    tmp_path: Path, damage: str
) -> None:
    database = tmp_path / "handoff-schema-tamper.sqlite3"
    connection = _checked_connection(database)
    apply_pending_migrations(connection, applied_at="2042-01-01T00:00:00.000000Z")
    handoff = EvaluationHandoffStore(connection).register(
        create_handoff(
            "candidate-version:fixture",
            "sha256:" + "a" * 64,
            "evaluation-sink:fixture",
        )
    )
    port = _create_evaluation_handoff_read_port(connection)
    connection.execute("DROP TRIGGER evaluation_handoff_identity_guard")
    if damage == "altered":
        connection.execute(
            "CREATE TRIGGER evaluation_handoff_identity_guard "
            "BEFORE UPDATE ON evaluation_handoffs BEGIN SELECT 1; END"
        )
    statements: list[str] = []
    connection.set_trace_callback(statements.append)
    connection.execute("BEGIN IMMEDIATE")
    statements.clear()

    with pytest.raises(HandoffContractError, match="schema objects differ"):
        port.require_retained_handoff_in_transaction(handoff.handoff_id)

    assert connection.in_transaction
    assert all(
        statement.lstrip().split(maxsplit=1)[0].upper() in {"SELECT", "PRAGMA"}
        for statement in statements
    )
    connection.set_trace_callback(None)
    connection.execute("ROLLBACK")
    connection.close()


def test_candidate_port_reads_exact_retained_and_current_values(
    tmp_path: Path,
) -> None:
    adapter = candidate_fixture._Adapter(tmp_path)
    location = adapter.create_location()
    handle = adapter.open_handle(location)
    handle.submit(candidate_fixture._generic("record-1"))
    row = handle._row("record-1")
    assert row is not None
    expected = handle._opened().load_version(str(row[1]))
    handle.close()

    args = candidate_fixture._collaborators(location.seed)
    connection = _checked_connection(location.seed[1])
    port = _create_story_candidate_read_port(
        connection,
        retrieval_authority=args["retrieval_authority"],
        authenticator=args["authenticator"],
        command_registry=args["command_registry"],
        payload_schemas=args["payload_schemas"],
        clock=args["clock"],
    )
    with pytest.raises(CandidateContractError, match="active checked connection"):
        port.require_retained_version_in_transaction(expected.version_id)

    connection.execute("BEGIN IMMEDIATE")
    assert port.require_retained_version_in_transaction(expected.version_id) == expected
    with pytest.raises(CandidateContractError, match="unknown Candidate Version"):
        port.require_retained_version_in_transaction("missing-version")
    assert (
        port.require_current_head_in_transaction(
            expected.candidate_id, proof=location.seed[0][3]
        )
        == expected
    )
    candidate = port.require_retained_candidate_in_transaction(expected.candidate_id)
    assert type(candidate) is StoryCandidate
    assert type(expected) is StoryCandidateVersion
    assert connection.in_transaction
    connection.execute("ROLLBACK")
    connection.close()


def test_candidate_historical_read_does_not_impose_current_upstream_state(
    tmp_path: Path,
) -> None:
    adapter = candidate_fixture._Adapter(tmp_path)
    location = adapter.create_location()
    handle = adapter.open_handle(location)
    handle.submit(candidate_fixture._generic("record-1"))
    row = handle._row("record-1")
    assert row is not None
    expected = handle._opened().load_version(str(row[1]))
    handle.close()
    candidate_fixture._advance_record_one(location)

    args = candidate_fixture._collaborators(location.seed)
    connection = _checked_connection(location.seed[1])
    port = _create_story_candidate_read_port(
        connection,
        retrieval_authority=args["retrieval_authority"],
        authenticator=args["authenticator"],
        command_registry=args["command_registry"],
        payload_schemas=args["payload_schemas"],
        clock=args["clock"],
    )
    connection.execute("BEGIN IMMEDIATE")
    assert port.require_retained_version_in_transaction(expected.version_id) == expected
    with pytest.raises(CandidateContractError):
        port.require_current_head_in_transaction(
            expected.candidate_id, proof=location.seed[0][3]
        )
    assert connection.in_transaction
    connection.execute("ROLLBACK")
    connection.close()


def test_candidate_bulk_snapshot_authenticates_selected_histories_once_and_reproves_next_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from newsroom.authority.story_candidate_system import _CandidateStore

    adapter = candidate_fixture._Adapter(tmp_path)
    location = adapter.create_location()
    handle = adapter.open_handle(location)
    handle.submit(candidate_fixture._generic("record-1"))
    handle.submit(candidate_fixture._generic("record-2"))
    rows = tuple(handle._row(name) for name in ("record-1", "record-2"))
    versions = tuple(handle._opened().load_version(str(row[1])) for row in rows)
    handle.close()
    args = candidate_fixture._collaborators(location.seed)
    connection = _checked_connection(location.seed[1])
    port = _create_story_candidate_read_port(
        connection, retrieval_authority=args["retrieval_authority"],
        authenticator=args["authenticator"], command_registry=args["command_registry"],
        payload_schemas=args["payload_schemas"], clock=args["clock"],
    )
    local_calls, upstream_calls = [], []
    verify_local, verify_upstream = _CandidateStore._verify_local, _CandidateStore._verify_upstream

    def local(store, **options):
        local_calls.append(options["candidate_ids"])
        return verify_local(store, **options)

    def upstream(store, selected, **options):
        upstream_calls.append(tuple(item[2].version_id for item in selected.values()))
        return verify_upstream(store, selected, **options)

    monkeypatch.setattr(_CandidateStore, "_verify_local", local)
    monkeypatch.setattr(_CandidateStore, "_verify_upstream", upstream)
    identities = (versions[0].version_id,) * 83 + ("missing", versions[1].version_id)
    with pytest.raises(CandidateContractError, match="active checked connection"):
        port.require_retained_versions_in_transaction(identities)
    connection.execute("BEGIN")
    assert port.require_retained_versions_in_transaction(identities) == (versions[0],) * 83 + (None, versions[1])
    assert len(local_calls) == len(upstream_calls) == 1
    assert set(upstream_calls[0]) == {item.version_id for item in versions}
    assert connection.in_transaction
    with pytest.raises(CandidateContractError, match="partition differs"):
        port.require_retained_versions_in_transaction((None,))
    connection.execute("ROLLBACK")
    # A retained selected head changed after the read. No verified state survives.
    connection.execute("BEGIN IMMEDIATE")
    connection.execute("DROP TRIGGER candidate_head_update_guard")
    connection.execute("UPDATE story_candidate_heads SET candidate_bytes=? WHERE candidate_id=?", (b"{}", versions[1].candidate_id))
    connection.execute("COMMIT")
    connection.execute("BEGIN")
    assert port.require_retained_versions_in_transaction((versions[0].version_id,)) == (versions[0],)
    assert len(local_calls) == 2
    assert local_calls[-1] == frozenset({versions[0].candidate_id})
    with pytest.raises(CandidateContractError):
        port.require_retained_versions_in_transaction((versions[1].version_id,))
    assert len(local_calls) == 3
    assert port.require_retained_versions_in_transaction(("missing",)) == (None,)
    assert len(local_calls) == 4
    assert local_calls[-1] == frozenset()
    connection.execute("ROLLBACK")
    connection.close()
