from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from newsroom.increment5.retrieval_context import (
    GovernedCasPassageHydrator,
    RetrievalContextBuilder,
    RetrievalContextError,
    RetrievalContextJournal,
)
from newsroom.tests.test_increment5d1_hybrid_composer import branch_inputs
from newsroom.tests.test_increment5d2_retrieval_context import _retained_complete_context


@pytest.fixture
def journal_connections(monkeypatch, tmp_path):
    connections = []
    connect = sqlite3.connect

    class TrackedConnection(sqlite3.Connection):
        close_count = 0

        def close(self):
            assert not self.in_transaction
            self.close_count += 1
            return super().close()

    def tracked_connect(database, *args, **kwargs):
        if Path(database).parent == tmp_path and Path(database).name.startswith("context"):
            kwargs["factory"] = TrackedConnection
            connection = connect(database, *args, **kwargs)
            connections.append(connection)
            return connection
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    return connections


def _assert_closed(connections):
    assert connections
    for connection in connections:
        assert connection.close_count == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")


def test_journal_initialisation_closes_connection(tmp_path, journal_connections):
    RetrievalContextJournal(tmp_path / "context.sqlite")

    _assert_closed(journal_connections)


def test_journal_write_and_replay_close_connections(
    tmp_path, branch_inputs, journal_connections,
):
    builder, composer, cas_root, _, path, request, receipt, _ = (
        _retained_complete_context(tmp_path, branch_inputs, name="lifetime")
    )

    assert builder.execute(request).canonical_bytes == receipt.canonical_bytes
    reopened = RetrievalContextBuilder(
        composition_replayer=composer,
        journal=RetrievalContextJournal(path),
        hydrator=GovernedCasPassageHydrator(cas_root),
    )
    assert reopened.execute(request).canonical_bytes == receipt.canonical_bytes
    _assert_closed(journal_connections)


def test_journal_producer_exception_rolls_back_and_closes(
    tmp_path, journal_connections,
):
    journal = RetrievalContextJournal(tmp_path / "context.sqlite")

    def failing_producer(_):
        raise RuntimeError("producer failed")

    with pytest.raises(RuntimeError, match="producer failed"):
        journal.execute(
            idempotency_key="context:failure",
            request_digest="sha256:" + "a" * 64,
            producer=failing_producer,
        )

    _assert_closed(journal_connections)


def test_journal_purge_and_tombstone_replay_close_connections(
    tmp_path, branch_inputs, journal_connections,
):
    builder, _, _, journal, path, request, receipt, _ = (
        _retained_complete_context(tmp_path, branch_inputs, name="purge-lifetime")
    )
    selection = {
        "admission_ids": (receipt.items[0].passage.admission_id,),
        "reason_code": "RIGHTS_WITHDRAWN",
    }

    purges = journal.purge_affected(**selection)
    assert len(purges) == 1
    replayed = RetrievalContextJournal(path).purge_affected(**selection)
    assert tuple(item.canonical_bytes for item in replayed) == (
        purges[0].canonical_bytes,
    )
    with pytest.raises(RetrievalContextError, match="purged"):
        builder.execute(request)
    _assert_closed(journal_connections)


def test_journal_purge_failure_restores_receipt_and_closes(
    tmp_path, branch_inputs, journal_connections, monkeypatch,
):
    builder, _, _, journal, _, request, receipt, _ = (
        _retained_complete_context(tmp_path, branch_inputs, name="purge-failure")
    )
    connect = journal._connect

    def deny_tombstone_insert():
        connection = connect()
        connection.set_authorizer(
            lambda action, table, _column, _database, _trigger: (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_INSERT
                and table == "increment5d2_retrieval_context_purges"
                else sqlite3.SQLITE_OK
            )
        )
        return connection

    with monkeypatch.context() as patch:
        patch.setattr(journal, "_connect", deny_tombstone_insert)
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            journal.purge_affected(
                admission_ids=(receipt.items[0].passage.admission_id,),
                reason_code="RIGHTS_WITHDRAWN",
            )

    assert builder.execute(request).canonical_bytes == receipt.canonical_bytes
    _assert_closed(journal_connections)


def test_journal_initialisation_exception_closes_connection(
    tmp_path, journal_connections, monkeypatch,
):
    def failing_initialisation(_connection):
        raise RuntimeError("initialisation failed")

    monkeypatch.setattr(
        RetrievalContextJournal, "_initialise_schema", staticmethod(failing_initialisation),
    )
    with pytest.raises(RuntimeError, match="initialisation failed"):
        RetrievalContextJournal(tmp_path / "context.sqlite")
    _assert_closed(journal_connections)
