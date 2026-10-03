"""Selected Candidate reads retain whole target history, not unrelated candidates."""

import sqlite3

import pytest

from newsroom.authority.story_candidate_system import (
    _CandidateStore, _create_story_candidate_read_port,
)
from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.authority.types import UtcTimestamp
from newsroom.increment6.candidates import CandidateContractError
from newsroom.tests.authority_store_conformance import IntegrityViolation
from newsroom.tests.test_increment6e2_candidate_store import _Adapter, _generic


@pytest.fixture
def two_candidate_reads(tmp_path):
    adapter = _Adapter(tmp_path)
    location = adapter.create_location()
    handle = adapter.open_handle(location)
    versions = []
    try:
        for record in ("record-1", "record-2"):
            handle.submit(_generic(record))
            versions.append(handle._opened().load_version(str(handle._row(record)[1])))
    finally:
        handle.close()
    connection = sqlite3.connect(location.seed[1], isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    port = _create_story_candidate_read_port(
        connection, retrieval_authority=location.seed[0][1],
        authenticator=location.seed[0][2], command_registry=location.seed[4],
        payload_schemas=location.seed[5],
        clock=lambda: UtcTimestamp.parse("2042-01-02T00:00:00.000000Z"),
    )
    try:
        yield connection, port, versions, adapter, location
    finally:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        connection.close()


def test_runtime_candidate_read_verifies_only_the_selected_history(
    two_candidate_reads, monkeypatch,
):
    connection, port, (first, other), _adapter, _location = two_candidate_reads
    assert first.candidate_id != other.candidate_id
    verified = []
    original = _CandidateStore._verify_row

    def counted(store, digest):
        result = original(store, digest)
        verified.append(result[2].candidate_id)
        return result

    monkeypatch.setattr(_CandidateStore, "_verify_row", counted)
    changes = connection.total_changes
    connection.execute("BEGIN")
    assert port.require_retained_version_in_transaction(first.version_id) == first
    assert verified == [first.candidate_id]
    assert connection.total_changes == changes
    connection.execute("ROLLBACK")


def test_selected_bulk_preserves_duplicates_missing_values_and_empty_requests(
    two_candidate_reads, monkeypatch,
):
    connection, port, (first, other), _adapter, _location = two_candidate_reads
    checked = []
    original = _CandidateStore._verify_row

    def counted(store, digest):
        result = original(store, digest)
        checked.append(result[2].candidate_id)
        return result

    monkeypatch.setattr(_CandidateStore, "_verify_row", counted)
    missing = "00000000-0000-4000-8000-000000000001"
    changes = connection.total_changes
    connection.execute("BEGIN")
    assert port.require_retained_versions_in_transaction(
        (first.version_id, missing, first.version_id),
    ) == (first, None, first)
    assert checked == [first.candidate_id]
    checked.clear()
    assert port.require_retained_versions_in_transaction((missing, missing)) == (None, None)
    assert port.require_retained_versions_in_transaction(()) == ()
    assert checked == []
    assert port.require_retained_versions_in_transaction(
        (other.version_id, first.version_id),
    ) == (other, first)
    assert set(checked) == {first.candidate_id, other.candidate_id}
    assert len(checked) == 2
    with pytest.raises(CandidateContractError, match="unknown Candidate Version"):
        port.require_retained_version_in_transaction(missing)
    assert connection.total_changes == changes


@pytest.mark.parametrize("damage", ["receipt", "head", "collision", "foreign_key"])
def test_selected_history_fails_closed_on_its_own_damage(two_candidate_reads, damage):
    connection, port, (first, _other), _adapter, _location = two_candidate_reads
    connection.execute("PRAGMA foreign_keys=OFF")
    if damage == "receipt":
        connection.execute("DROP TRIGGER immutable_candidate_receipt")
        connection.execute("UPDATE story_candidate_admission_receipts_v2 SET admission_bytes=? WHERE candidate_id=?",
                           (b"{}", first.candidate_id))
    elif damage == "head":
        connection.execute("DROP TRIGGER retained_candidate_head")
        connection.execute("DELETE FROM story_candidate_heads WHERE candidate_id=?", (first.candidate_id,))
    elif damage == "collision":
        connection.execute("DROP TRIGGER retained_candidate_collision")
        connection.execute("DELETE FROM story_candidate_collision_bindings WHERE candidate_id=?", (first.candidate_id,))
    else:
        connection.execute("DROP TRIGGER immutable_candidate_receipt")
        connection.execute("UPDATE story_candidate_admission_receipts_v2 SET authority_event_id=? WHERE candidate_id=?",
                           ("00000000-0000-4000-8000-000000000001", first.candidate_id))
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("BEGIN")
    with pytest.raises((CandidateContractError, AuthorityPersistenceError)):
        port.require_retained_version_in_transaction(first.version_id)


def test_selected_event_coverage_rejects_an_extra_event(two_candidate_reads):
    connection, port, (first, other), _adapter, _location = two_candidate_reads
    connection.execute("DROP TRIGGER immutable_ledger_events_update")
    connection.execute(
        "UPDATE ledger_events SET aggregate_id=? WHERE event_id=("
        "SELECT authority_event_id FROM story_candidate_admission_receipts_v2 WHERE candidate_id=?)",
        (first.candidate_id, other.candidate_id),
    )
    connection.execute("BEGIN")
    with pytest.raises(CandidateContractError, match="event coverage"):
        port.require_retained_version_in_transaction(first.version_id)


@pytest.mark.parametrize("damage", ["receipt", "foreign_key"])
def test_unrelated_damage_is_not_a_target_gate_but_full_inventory_rejects_it(
    two_candidate_reads, damage,
):
    connection, port, (first, other), adapter, location = two_candidate_reads
    connection.execute("PRAGMA foreign_keys=OFF")
    connection.execute("DROP TRIGGER immutable_candidate_receipt")
    if damage == "receipt":
        connection.execute("UPDATE story_candidate_admission_receipts_v2 SET admission_bytes=? WHERE candidate_id=?",
                           (b"{}", other.candidate_id))
    else:
        connection.execute("UPDATE story_candidate_admission_receipts_v2 SET authority_event_id=? WHERE candidate_id=?",
                           ("00000000-0000-4000-8000-000000000001", other.candidate_id))
    connection.execute("PRAGMA foreign_keys=ON")
    changes = connection.total_changes
    connection.execute("BEGIN")
    assert port.require_retained_version_in_transaction(first.version_id) == first
    assert connection.total_changes == changes
    with pytest.raises((CandidateContractError, AuthorityPersistenceError)):
        port.verify_retained_integrity_in_transaction()
    connection.execute("ROLLBACK")
    with pytest.raises(IntegrityViolation, match="Candidate authority open failed"):
        adapter.open_handle(location)._opened()
