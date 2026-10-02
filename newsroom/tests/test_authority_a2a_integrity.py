from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import sqlite3

import pytest

from newsroom.authority import (
    AuthorityPersistenceError,
    AuthoritySchemaError,
    AuthorityWriterBusy,
    CausationKind,
    CausationRef,
    CommandId,
    EventId,
    ExpectedVersionConflict,
    InlinePayload,
    SemanticCommand,
    UnknownCausation,
)
from newsroom.authority.migrations import apply_migration
from newsroom.authority._event_store_base import _INCOMPLETE_COMMAND_QUERY

from .authority_event_helpers import open_test_system
from .authority_helpers import command, proof


def test_authority_connection_uses_bounded_cache_without_changing_durability(
    tmp_path, monkeypatch,
) -> None:
    from newsroom.authority._event_store_base import _EventStoreBase

    original = _EventStoreBase._configure_connection
    observed = []

    def inspect_configuration(store):
        original(store)
        observed.append(tuple(
            store._connection.execute(f"PRAGMA {setting}").fetchone()[0]
            for setting in (
                "cache_size", "foreign_keys", "journal_mode", "synchronous", "busy_timeout",
            )
        ))

    monkeypatch.setattr(_EventStoreBase, "_configure_connection", inspect_configuration)
    database = tmp_path / "authority.sqlite3"
    for _ in range(2):
        with open_test_system(database):
            pass
    assert observed == [(-16384, 1, "wal", 2, 5000)] * 2
    # The cache budget belongs to this connection, not the database schema.
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        assert connection.execute("PRAGMA cache_size").fetchone()[0] == -2000


def test_reopen_logs_each_existing_validation_phase(tmp_path, caplog) -> None:
    database = tmp_path / "authority.sqlite3"
    with open_test_system(database) as system:
        system.commands.execute(command(), proof=proof())
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="newsroom.authority.open"):
        with open_test_system(database):
            pass
    records = [r for r in caplog.records if r.name == "newsroom.authority.open"]
    phases = (
        "schema", "quick_check", "foreign_key_check",
        "relational_invariants", "aggregate_heads", "command_completeness", "event_routing",
        "immutable_records", "connection_settings", "registry_coverage",
    )
    assert [r.getMessage() for r in records if "STARTED" in r.getMessage()] == [
        f"authority_open stage={stage} status=STARTED"
        for stage in ("validation", *phases)
    ]
    completed = [r for r in records if "COMPLETE" in r.getMessage()]
    # The relational phase encloses the existing three SQL subphases; these
    # elapsed values overlap and must not be added as independent wall time.
    assert [r.args[0] for r in completed] == [
        *phases[:3], *phases[4:7], "relational_invariants", *phases[7:], "validation",
    ]
    assert all(len(r.args) == 4 and all(type(v) is int and v >= 0 for v in r.args[2:])
               for r in completed)
    assert len(records) == 2 * (len(phases) + 1)


@pytest.mark.parametrize("failure", [None, ValueError("private detail"), KeyboardInterrupt()])
def test_validation_timing_preserves_original_failure(monkeypatch, caplog, failure) -> None:
    from newsroom.authority import _event_store_base as private

    ticks = iter((1_000_000, 3_999_999))
    cpu_ticks = iter((7_000_000, 8_999_999))
    monkeypatch.setattr(private, "perf_counter_ns", lambda: next(ticks), raising=False)
    monkeypatch.setattr(private, "process_time_ns", lambda: next(cpu_ticks), raising=False)
    with caplog.at_level(logging.INFO, logger="newsroom.authority.open"):
        if failure is None:
            with private._validation_stage("quick_check"):
                pass
        else:
            with pytest.raises(type(failure)) as raised:
                with private._validation_stage("quick_check"):
                    raise failure
            assert raised.value is failure
    status = "COMPLETE" if failure is None else "FAILED"
    assert [r.getMessage() for r in caplog.records] == [
        "authority_open stage=quick_check status=STARTED",
        f"authority_open stage=quick_check status={status} elapsed_ms=2 cpu_ms=1",
    ]


def test_partial_migration_failure_is_atomic() -> None:
    conn = sqlite3.connect(":memory:", isolation_level=None)
    try:
        faulty = (
            "CREATE TABLE partial_table(id INTEGER PRIMARY KEY) STRICT",
            "CREATE TABLE broken(",
        )
        with pytest.raises(sqlite3.DatabaseError):
            apply_migration(
                conn,
                applied_at="2026-07-17T00:00:00.000000Z",
                statements=faulty,
            )
        names = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "partial_table" not in names
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 0
    finally:
        conn.close()


def test_reopen_validates_history_schema_and_relational_integrity(
    tmp_path: Path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    with open_test_system(database) as system:
        system.commands.execute(command(), proof=proof())
    with open_test_system(database) as reopened:
        assert len(reopened.events.after(0, proof=proof())) == 1


def test_schema_tamper_fails_and_constructor_releases_writer_lock(
    tmp_path: Path, caplog,
) -> None:
    database = tmp_path / "authority.sqlite3"
    with open_test_system(database):
        pass

    conn = sqlite3.connect(database)
    try:
        conn.execute("DROP INDEX idx_ledger_events_recorded")
        conn.commit()
    finally:
        conn.close()

    with caplog.at_level(logging.INFO, logger="newsroom.authority.open"):
        with pytest.raises(AuthoritySchemaError, match="fingerprint"):
            open_test_system(database)
    failed = [r for r in caplog.records if "status=FAILED" in r.getMessage()]
    assert [r.args[0] for r in failed] == ["schema", "validation"]

    # A failed constructor must not retain the lifetime writer lock.
    conn = sqlite3.connect(database)
    try:
        conn.execute(
            "CREATE INDEX idx_ledger_events_recorded "
            "ON ledger_events(recorded_at, ledger_seq)"
        )
        conn.commit()
    finally:
        conn.close()
    with open_test_system(database):
        pass


def test_raw_sql_cannot_mutate_append_only_records(
    tmp_path: Path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    with open_test_system(database) as system:
        committed = system.commands.execute(command(), proof=proof())
        conn = sqlite3.connect(database, timeout=1)
        try:
            mutations = (
                (
                    "UPDATE authority_commands "
                    "SET command_type='tampered' WHERE command_id=?",
                    (committed.command_id,),
                ),
                (
                    "DELETE FROM authority_payloads WHERE payload_id=("
                    "SELECT payload_id FROM ledger_events WHERE event_id=?)",
                    (committed.event_id,),
                ),
                (
                    "UPDATE authority_audit_events "
                    "SET event_type='tampered' WHERE command_id=?",
                    (committed.command_id,),
                ),
                ("DELETE FROM authorization_requests", ()),
                (
                    "UPDATE payload_schema_contracts "
                    "SET contract_version='tampered'",
                    (),
                ),
            )
            for sql, parameters in mutations:
                with pytest.raises(sqlite3.DatabaseError, match="immutable"):
                    conn.execute(sql, parameters)
        finally:
            conn.close()


def test_aggregate_head_tamper_is_detected_on_restart(
    tmp_path: Path, caplog,
) -> None:
    database = tmp_path / "authority.sqlite3"
    with open_test_system(database) as system:
        system.commands.execute(command(), proof=proof())

    # Simulate an out-of-contract raw writer, then restore the exact schema.
    # The composite FK or the explicit head check must fail closed on reopen.
    conn = sqlite3.connect(database)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DROP TRIGGER authority_aggregates_update_guard")
        conn.execute(
            "UPDATE authority_aggregates SET current_version=current_version+1"
        )
        conn.execute(
            """CREATE TRIGGER authority_aggregates_update_guard
            BEFORE UPDATE ON authority_aggregates
            WHEN NEW.aggregate_type != OLD.aggregate_type
              OR NEW.aggregate_id != OLD.aggregate_id
              OR NEW.current_version != OLD.current_version + 1
              OR NEW.created_at != OLD.created_at
            BEGIN SELECT RAISE(ABORT,'invalid aggregate-head update'); END"""
        )
        conn.commit()
    finally:
        conn.close()
    with caplog.at_level(logging.INFO, logger="newsroom.authority.open"):
        with pytest.raises(
            AuthoritySchemaError, match="foreign-key|aggregate head"
        ):
            open_test_system(database)
    failed = [r for r in caplog.records if "status=FAILED" in r.getMessage()]
    assert [r.args[0] for r in failed] == ["foreign_key_check", "validation"]


def test_every_command_has_one_version_audit_and_event(
    tmp_path: Path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    with open_test_system(database) as system:
        for index in range(3):
            system.commands.execute(
                command(key=f"command-{index}"), proof=proof()
            )
    conn = sqlite3.connect(database)
    try:
        counts = conn.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM authority_commands),"
            "(SELECT COUNT(*) FROM authority_aggregate_versions),"
            "(SELECT COUNT(*) FROM authority_audit_events),"
            "(SELECT COUNT(*) FROM ledger_events)"
        ).fetchone()
        assert counts == (3, 3, 3, 3)
    finally:
        conn.close()


def test_command_completeness_uses_indexed_counts_without_temp_btrees() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(
            """
            CREATE TABLE authority_commands(command_id TEXT PRIMARY KEY);
            CREATE TABLE authority_aggregate_versions(command_id TEXT);
            CREATE TABLE ledger_events(command_id TEXT);
            CREATE TABLE authority_audit_events(command_id TEXT);
            CREATE INDEX versions_by_command
              ON authority_aggregate_versions(command_id);
            CREATE INDEX events_by_command ON ledger_events(command_id);
            CREATE INDEX audits_by_command ON authority_audit_events(command_id);
            """
        )
        for command_id in ("first", "middle", "last"):
            conn.execute("INSERT INTO authority_commands VALUES(?)", (command_id,))
            conn.execute(
                "INSERT INTO authority_aggregate_versions VALUES(?)", (command_id,)
            )
            conn.execute("INSERT INTO ledger_events VALUES(?)", (command_id,))
            conn.execute("INSERT INTO authority_audit_events VALUES(?)", (command_id,))

        assert conn.execute(_INCOMPLETE_COMMAND_QUERY).fetchone() is None
        plan = tuple(
            str(row[3])
            for row in conn.execute("EXPLAIN QUERY PLAN " + _INCOMPLETE_COMMAND_QUERY)
        )
        assert not any("TEMP B-TREE" in step for step in plan)
        assert all(
            any(index in step for step in plan)
            for index in (
                "versions_by_command",
                "events_by_command",
                "audits_by_command",
            )
        )

        # Exact counts, rather than current uniqueness constraints, keep this
        # check fail-closed if a future schema admits duplicate child rows.
        conn.execute("INSERT INTO ledger_events VALUES('middle')")
        assert conn.execute(_INCOMPLETE_COMMAND_QUERY).fetchone() == ("middle",)
        conn.execute(
            "DELETE FROM ledger_events WHERE rowid=(SELECT MAX(rowid) "
            "FROM ledger_events WHERE command_id='middle')"
        )
        conn.execute("DELETE FROM authority_audit_events WHERE command_id='last'")
        assert conn.execute(_INCOMPLETE_COMMAND_QUERY).fetchone() == ("last",)
    finally:
        conn.close()


def test_second_lifetime_writer_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "authority.sqlite3"
    with open_test_system(database):
        with pytest.raises(AuthorityWriterBusy):
            open_test_system(database)


def test_symlink_and_broad_parent_permissions_fail_closed(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real.sqlite3"
    with open_test_system(real):
        pass
    alias = tmp_path / "alias.sqlite3"
    alias.symlink_to(real)
    with pytest.raises(AuthoritySchemaError, match="symlink"):
        open_test_system(alias)

    broad = tmp_path / "broad"
    broad.mkdir(mode=0o700)
    os.chmod(broad, 0o755)
    with pytest.raises(AuthoritySchemaError, match="permissions"):
        open_test_system(broad / "authority.sqlite3")


def test_command_and_event_causation_must_resolve(
    tmp_path: Path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    with open_test_system(database) as system:
        first = system.commands.execute(command(key="first"), proof=proof())
        second = system.commands.execute(
            SemanticCommand(
                command_type="record.observed",
                aggregate_id=command().aggregate_id,
                expected_aggregate_version=0,
                payload=command().payload,
                idempotency_key="command-caused",
                causation=CausationRef(CausationKind.COMMAND, first.command_id),
            ),
            proof=proof(),
        )
        system.commands.execute(
            SemanticCommand(
                command_type="record.observed",
                aggregate_id=command().aggregate_id,
                expected_aggregate_version=0,
                payload=command().payload,
                idempotency_key="event-caused",
                causation=CausationRef(CausationKind.EVENT, second.event_id),
            ),
            proof=proof(),
        )

        with pytest.raises(UnknownCausation):
            system.commands.execute(
                SemanticCommand(
                    command_type="record.observed",
                    aggregate_id=command().aggregate_id,
                    expected_aggregate_version=0,
                    payload=command().payload,
                    idempotency_key="missing-command",
                    causation=CausationRef(
                        CausationKind.COMMAND, str(CommandId.new())
                    ),
                ),
                proof=proof(),
            )
        with pytest.raises(UnknownCausation):
            system.commands.execute(
                SemanticCommand(
                    command_type="record.observed",
                    aggregate_id=command().aggregate_id,
                    expected_aggregate_version=0,
                    payload=command().payload,
                    idempotency_key="missing-event",
                    causation=CausationRef(
                        CausationKind.EVENT, str(EventId.new())
                    ),
                ),
                proof=proof(),
            )


def test_expected_version_fencing_is_atomic(tmp_path: Path) -> None:
    database = tmp_path / "authority.sqlite3"
    aggregate_id = command().aggregate_id
    with open_test_system(database) as system:
        system.commands.execute(
            command(
                aggregate_id=aggregate_id,
                expected_version=0,
                key="create",
            ),
            proof=proof(),
        )
        system.commands.execute(
            SemanticCommand(
                command_type="record.observed",
                aggregate_id=aggregate_id,
                expected_aggregate_version=1,
                payload=InlinePayload({"headline": "Updated", "count": 2}),
                idempotency_key="update",
            ),
            proof=proof(),
        )
        with pytest.raises(ExpectedVersionConflict):
            system.commands.execute(
                command(
                    aggregate_id=aggregate_id,
                    expected_version=1,
                    key="stale",
                ),
                proof=proof(),
            )
        assert [
            event.aggregate_version
            for event in system.events.after(0, proof=proof())
        ] == [1, 2]


def test_shared_connection_reads_and_writes_are_serialized(
    tmp_path: Path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    with open_test_system(database) as system:
        def write(index: int) -> str:
            return system.commands.execute(
                command(key=f"parallel-{index}"),
                proof=proof(),
            ).command_id

        def read(_: int) -> int:
            return len(system.events.after(0, limit=100, proof=proof()))

        with ThreadPoolExecutor(max_workers=8) as executor:
            command_ids = list(executor.map(write, range(20)))
            read_counts = list(executor.map(read, range(20)))
        assert len(set(command_ids)) == 20
        assert len(system.events.after(0, limit=100, proof=proof())) == 20
        assert all(count >= 0 for count in read_counts)


def test_read_time_provenance_validation_fails_closed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    with open_test_system(database) as system:
        committed = system.commands.execute(command(), proof=proof())

        # Simulate an out-of-contract raw writer while the accepted process is
        # alive. Read-time canonical verification must still fail closed.
        conn = sqlite3.connect(database)
        try:
            conn.execute(
                "DROP TRIGGER immutable_authorization_requests_update"
            )
            conn.execute(
                "UPDATE authorization_requests SET storage_request_residual=?",
                (b'{"tampered":true}',),
            )
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(AuthorityPersistenceError, match="request"):
            system.events.provenance(committed.event_id, proof=proof())
