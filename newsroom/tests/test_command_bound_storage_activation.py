from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from newsroom.authority import AuthorityPersistenceError
from newsroom.authority.audit_retention import _scan_business
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.authority import command_bound_storage_migrations as migration
from newsroom.authority.migrations import SCHEMA_VERSION, EXPECTED_MIGRATION_HISTORY, apply_pending_migrations, schema_fingerprint
from newsroom.tests.authority_event_helpers import open_test_system
from newsroom.tests.authority_helpers import FIXED_NOW, command, proof, make_service
from newsroom.tests.test_authority_exact_event_closure import _store
from newsroom.tests.graphiti_adapter_4d_migration_helpers import _drop_v41_command_bound_storage


def test_new_command_writes_compact_requests_and_results_with_exact_public_reads(tmp_path):
    path = tmp_path / "compact.sqlite3"
    semantic = command(key="compact-write")
    with open_test_system(path) as system:
        committed = system.commands.execute(semantic, proof=proof())
        provenance = system.events.provenance(committed.event_id, proof=proof())
        result = system.events.command_result(committed.command_id, proof=proof())
        assert system.commands.execute(semantic, proof=proof()).replayed
    with sqlite3.connect(path) as connection:
        request = connection.execute("SELECT storage_request_marker,storage_request_residual FROM authorization_requests").fetchone()
        retained = connection.execute("SELECT result_bytes FROM authority_commands").fetchone()[0]
        assert SCHEMA_VERSION >= migration.COMMAND_BOUND_STORAGE_SCHEMA_VERSION
        assert request[0] == b"v41"
        assert len(request[1]) < 100
        assert len(retained) == 54
        assert retained != result.result_bytes
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert digest_bytes(result.result_bytes) == committed.result_digest
    with open_test_system(path) as system:
        assert system.events.provenance(committed.event_id, proof=proof()) == provenance
        assert system.events.command_result(committed.command_id, proof=proof()) == result
        assert system.commands.execute(semantic, proof=proof()).replayed


def _old_path(tmp_path, *, count=3):
    path = tmp_path / "v40.sqlite3"
    commands = tuple(command(key=f"old-{index}") for index in range(count))
    with open_test_system(path) as system:
        committed = tuple(system.commands.execute(item, proof=proof()) for item in commands)
        provenance = tuple(system.events.provenance(item.event_id, proof=proof()) for item in committed)
        results = tuple(system.events.command_result(item.command_id, proof=proof()) for item in committed)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _drop_v41_command_bound_storage(connection)
        assert schema_fingerprint(connection) == migration.COMMAND_BOUND_STORAGE_PREDECESSOR_FINGERPRINT
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    return path, commands, provenance, results


def _snapshot(connection):
    tables = ("authority_migrations", "authorization_requests", "authority_commands")
    return (
        connection.execute("PRAGMA user_version").fetchone()[0], schema_fingerprint(connection),
        tuple(tuple(connection.execute(f"SELECT * FROM {table} ORDER BY rowid")) for table in tables),
    )


def _change(connection, table, sql, parameters=()):
    name = f"immutable_{table}_update"
    guard = connection.execute("SELECT sql FROM sqlite_schema WHERE name=?", (name,)).fetchone()[0]
    connection.execute(f"DROP TRIGGER {name}")
    connection.execute(sql, parameters)
    connection.execute(guard)


def test_v40_upgrade_preserves_logical_authority_provenance_result_replay_and_new_write(tmp_path):
    path, commands, provenance, results = _old_path(tmp_path)
    with sqlite3.connect(path) as connection:
        old_logical = _scan_business(connection, logical_storage=True)
        old_physical = _scan_business(connection)
        before_ids = tuple(connection.execute("SELECT command_id,authorization_request_digest,result_digest,committed_at FROM authority_commands ORDER BY command_id"))
        apply_pending_migrations(connection, applied_at="2026-10-01T00:00:00.000000Z")
        assert _scan_business(connection, logical_storage=True) == old_logical
        assert _scan_business(connection) != old_physical
        assert tuple(connection.execute("SELECT command_id,authorization_request_digest,result_digest,committed_at FROM authority_commands ORDER BY command_id")) == before_ids
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT count(*) FROM authorization_requests WHERE storage_request_marker=?", (b"v41",)).fetchone()[0] == len(commands)
    with open_test_system(path) as system:
        for semantic, original_provenance, original_result in zip(commands, provenance, results, strict=True):
            assert system.events.provenance(original_provenance.event.event_id, proof=proof()) == original_provenance
            assert system.events.command_result(original_result.command_id, proof=proof()) == original_result
            assert system.commands.execute(semantic, proof=proof()).replayed
        system.commands.execute(command(key="post-upgrade"), proof=proof())


@pytest.mark.parametrize("table,field,value", (
    ("authority_commands", "result_digest", "sha256:" + "f" * 64),
    ("ledger_events", "event_schema_version", 2),
    ("command_definitions", "canonical_bytes", b"{}"),
    ("authorization_requests", "canonical_record_digest", "sha256:" + "f" * 64),
))
def test_v41_authenticates_all_before_first_update_and_rolls_back_on_corrupt_backing(tmp_path, table, field, value):
    path, _, _, _ = _old_path(tmp_path)
    with sqlite3.connect(path) as connection:
        _change(connection, table, f"UPDATE {table} SET {field}=? WHERE rowid=(SELECT max(rowid) FROM {table})", (value,))
        connection.commit()
        before = _snapshot(connection)
        updates = []
        connection.set_trace_callback(lambda sql: updates.append(sql) if sql.startswith("UPDATE ") else None)
        with pytest.raises(sqlite3.IntegrityError):
            apply_pending_migrations(connection, applied_at="2026-10-01T00:00:00.000000Z")
        assert updates == []
        assert _snapshot(connection) == before


def test_mid_conversion_failure_rolls_back_rows_guards_schema_and_history(tmp_path):
    class FailedUpdate(sqlite3.Connection):
        updates = 0

        def execute(self, sql, parameters=()):
            cursor = super().execute(sql, parameters)
            if sql.startswith("UPDATE authority_commands SET result_bytes="):
                self.updates += 1
                if self.updates == 2:
                    raise sqlite3.IntegrityError("injected compact result failure")
            return cursor

    path, _, _, _ = _old_path(tmp_path)
    with sqlite3.connect(path, factory=FailedUpdate) as connection:
        before = _snapshot(connection)
        with pytest.raises(sqlite3.IntegrityError, match="injected compact result"):
            apply_pending_migrations(connection, applied_at="2026-10-01T00:00:00.000000Z")
        assert connection.updates == 2
        assert _snapshot(connection) == before


@pytest.mark.parametrize("boundary", ("no_transaction", "history", "schema"))
def test_v41_requires_exact_checked_transactional_predecessor(tmp_path, boundary):
    path, _, _, _ = _old_path(tmp_path)
    with sqlite3.connect(path) as connection:
        history = tuple(connection.execute("SELECT version,name,checksum FROM authority_migrations ORDER BY version"))
        if boundary == "schema":
            connection.execute("CREATE INDEX unrecognised_storage_index ON authority_commands(command_type)")
            connection.commit()
        before = _snapshot(connection)
        if boundary != "no_transaction":
            connection.execute("BEGIN EXCLUSIVE")
        with pytest.raises(sqlite3.DatabaseError, match="active transaction|exact checked"):
            migration.migrate_command_bound_storage(connection, expected_history=() if boundary == "history" else history)
        connection.rollback()
        assert _snapshot(connection) == before


@pytest.mark.parametrize("table,field,value", (
    ("ledger_events", "aggregate_id", "changed"),
    ("command_definitions", "canonical_bytes", b"{}"),
    ("payload_schema_contracts", "canonical_bytes", b"{}"),
    ("authorization_requests", "storage_request_residual", b"{}"),
    ("authority_commands", "result_bytes", b"command-result-v1:missing"),
))
def test_compact_reopen_rejects_changed_backing_or_encoding(tmp_path, table, field, value):
    path = tmp_path / "corrupt.sqlite3"
    with open_test_system(path) as system:
        system.commands.execute(command(key="corrupt"), proof=proof())
    with sqlite3.connect(path) as connection:
        _change(connection, table, f"UPDATE {table} SET {field}=?", (value,))
    with pytest.raises((AuthorityPersistenceError, ValueError)):
        with open_test_system(path):
            pass


def test_keyset_conversion_does_not_fetchall_command_or_request_stream(tmp_path):
    class StreamingCursor(sqlite3.Cursor):
        guarded = False

        def execute(self, sql, parameters=()):
            self.guarded = sql.startswith("SELECT rowid,* FROM authority_commands") or sql.startswith("SELECT rowid,* FROM authorization_requests")
            return super().execute(sql, parameters)

        def fetchall(self):
            if self.guarded:
                raise AssertionError("fetched all authority storage rows")
            return super().fetchall()

    class StreamingConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            return self.cursor(factory=StreamingCursor).execute(sql, parameters)

    path, _, _, _ = _old_path(tmp_path)
    with sqlite3.connect(path, factory=StreamingConnection) as connection:
        apply_pending_migrations(connection, applied_at="2026-10-01T00:00:00.000000Z")
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_standalone_evaluation_request_remains_v38_on_new_writes_and_upgrade(tmp_path):
    path = tmp_path / "standalone.sqlite3"
    boundary = make_service()
    grant = boundary._authorize_for_commit(command(key="standalone"), proof=proof())
    unsigned = grant.authorization_request.unsigned_value()
    unsigned["operation_type"] = "evaluation:fixture"
    request = replace(grant.authorization_request, operation_type=unsigned["operation_type"], request_digest=digest_canonical(unsigned))
    decision = replace(grant.authorization, authorization_request_digest=request.request_digest)
    with _store(path, boundary) as store:
        with store._lock, store._transaction() as connection:
            store._persist_security_records(connection, authentication=grant.authentication, request=request, decision=decision, recorded_at=FIXED_NOW.to_text())
        store.commit(boundary._authorize_for_commit(command(key="ordinary"), proof=proof()))
    with sqlite3.connect(path) as connection:
        before = connection.execute("SELECT storage_request_residual,storage_request_marker FROM authorization_requests WHERE request_digest=?", (request.request_digest,)).fetchone()
        assert before[1] == b"v38"
        _drop_v41_command_bound_storage(connection)
        connection.commit()
        apply_pending_migrations(connection, applied_at=FIXED_NOW.to_text())
        assert connection.execute("SELECT storage_request_residual,storage_request_marker FROM authorization_requests WHERE request_digest=?", (request.request_digest,)).fetchone() == before
    with _store(path, boundary) as store:
        row = store._connection.execute("SELECT * FROM authorization_requests WHERE request_digest=?", (request.request_digest,)).fetchone()
        assert store._request_record_from_row(row).canonical_bytes == canonical_json_bytes(request.canonical_value())


def test_final_persisted_backing_validation_failure_rolls_back_entire_new_write(tmp_path, monkeypatch):
    path = tmp_path / "atomic-write.sqlite3"
    boundary = make_service()
    with _store(path, boundary) as store:
        calls = []

        def reject_after_ledger(row, *, connection=None):
            assert connection.execute("SELECT count(*) FROM ledger_events").fetchone()[0] == 1
            assert bytes(row["storage_request_marker"]) == b"v41"
            calls.append(row["request_digest"])
            raise AuthorityPersistenceError("injected final backing validation")

        monkeypatch.setattr(store, "_request_record_from_row", reject_after_ledger)
        with pytest.raises(AuthorityPersistenceError, match="final backing"):
            store.commit(boundary._authorize_for_commit(command(key="atomic"), proof=proof()))
        assert len(calls) == 1
        for table in ("authority_commands", "ledger_events", "authorization_requests", "authentication_contexts", "authority_payloads"):
            assert store._connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_physical_reference_scanner_preserves_v38_and_v41_tokens_without_redecoding(tmp_path, monkeypatch):
    from newsroom.authority import audit_retention

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("Physical reference collection must not replay historical codecs")

    monkeypatch.setattr(audit_retention, "_logical_storage_values", unexpected_decode)
    path, _, _, _ = _old_path(tmp_path, count=1)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TEMP TABLE _audit_tokens(id TEXT PRIMARY KEY) WITHOUT ROWID")
        original_physical = _scan_business(connection, tokens=connection)
        original_tokens = tuple(connection.execute("SELECT id FROM _audit_tokens ORDER BY id"))
        connection.execute("DELETE FROM _audit_tokens")
        connection.commit()
        apply_pending_migrations(connection, applied_at=FIXED_NOW.to_text())
        compact_physical = _scan_business(connection, tokens=connection)
        compact_tokens = tuple(connection.execute("SELECT id FROM _audit_tokens ORDER BY id"))
        # The new migration checksum is separately authenticated metadata.
        added = set(compact_tokens) - set(original_tokens)
        assert added == {(checksum,) for version, _, checksum in EXPECTED_MIGRATION_HISTORY if version > 40}
        assert set(original_tokens) <= set(compact_tokens)
        assert original_physical != compact_physical


def test_logical_scanner_rejects_corrupt_original_result_instead_of_hashing_it_as_authority(tmp_path):
    path, _, _, _ = _old_path(tmp_path, count=1)
    with sqlite3.connect(path) as connection:
        _change(connection, "authority_commands", "UPDATE authority_commands SET result_digest=?", ("sha256:" + "f" * 64,))
        with pytest.raises(AuthorityPersistenceError):
            _scan_business(connection, logical_storage=True)


@pytest.mark.parametrize("table", ("authority_commands", "ledger_events", "command_definitions", "payload_schema_contracts"))
def test_compact_request_direct_read_rejects_missing_persisted_backing(tmp_path, table):
    from newsroom.authority._event_store import _EventAuthorityStore

    path = tmp_path / "missing.sqlite3"
    with open_test_system(path) as system:
        system.commands.execute(command(key="missing"), proof=proof())
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM authorization_requests").fetchone()
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(f"DROP TRIGGER immutable_{table}_delete")
        connection.execute(f"DELETE FROM {table}")
        reader = object.__new__(_EventAuthorityStore)
        with pytest.raises(AuthorityPersistenceError, match="backing is missing"):
            reader._request_record_from_row(row, connection=connection)


def test_result_marker_cannot_rebind_a_different_command_row(tmp_path):
    from newsroom.authority._event_store import _EventAuthorityStore

    path = tmp_path / "swap.sqlite3"
    with open_test_system(path) as system:
        system.commands.execute(command(key="swap-one"), proof=proof())
        system.commands.execute(command(key="swap-two"), proof=proof())
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = tuple(connection.execute("SELECT * FROM authority_commands ORDER BY rowid"))
        _change(connection, "authority_commands", "UPDATE authority_commands SET result_bytes=?,result_digest=? WHERE command_id=?", (rows[1]["result_bytes"], rows[1]["result_digest"], rows[0]["command_id"]))
        reader = object.__new__(_EventAuthorityStore)
        with pytest.raises(AuthorityPersistenceError):
            reader._decode_result(bytes(rows[1]["result_bytes"]), str(rows[1]["result_digest"]), replayed=False, connection=connection, command_id=str(rows[0]["command_id"]))
        with pytest.raises(AuthorityPersistenceError):
            _scan_business(connection, logical_storage=True)

        # Retention preserves untouched bytes/references; it is not a repeat
        # audit of all old codecs. Normal consumers still reject this marker.
        before = _scan_business(connection)
        connection.execute("CREATE TEMP TABLE _audit_tokens(id TEXT PRIMARY KEY) WITHOUT ROWID")
        assert _scan_business(connection, tokens=connection) == before
        assert connection.execute(
            "SELECT result_bytes,result_digest FROM authority_commands WHERE command_id=?",
            (rows[0]["command_id"],),
        ).fetchone()[:] == (rows[1]["result_bytes"], rows[1]["result_digest"])
        tokens = {row[0] for row in connection.execute("SELECT id FROM _audit_tokens")}
        assert {row["command_id"] for row in rows} <= tokens
        with pytest.raises(AuthorityPersistenceError):
            reader._decode_result(bytes(rows[1]["result_bytes"]), str(rows[1]["result_digest"]), replayed=False, connection=connection, command_id=str(rows[0]["command_id"]))


def test_876_delivery_command_cohort_preserves_logical_authority_across_keyset_batches(tmp_path, record_property):
    from newsroom.authority import AggregateId, CommandRegistry, InlinePayload, PayloadSchemaRegistry, SemanticCommand
    from newsroom.projection.policy import projection_command_definitions, projection_payload_contracts
    from newsroom.tests.authority_event_helpers import fixture_read_policy

    definition = next(item for item in projection_command_definitions() if item.command_type == "projection.delivery.record")
    policy = fixture_read_policy(allowed_security_scopes=frozenset({definition.security_scope}), allowed_trust_scopes=frozenset({definition.trust_scope}))
    path = tmp_path / "delivery-cohort.sqlite3"
    generation = AggregateId.new()
    with open_test_system(
        path, registry=CommandRegistry([definition]),
        payload_schema_registry=PayloadSchemaRegistry(projection_payload_contracts()),
        scopes=frozenset({definition.required_scope, policy.required_scope}), read_policy=policy,
    ) as system:
        for index in range(876):
            system.commands.execute(SemanticCommand(
                command_type=definition.command_type, aggregate_id=generation,
                expected_aggregate_version=index,
                payload=InlinePayload({"generation_id": str(generation), "ledger_seq": index + 1, "outcome": "IGNORED_OPTIONAL", "error_code": None}),
                idempotency_key=f"cohort-{index}",
            ), proof=proof())
    with sqlite3.connect(path) as connection:
        _drop_v41_command_bound_storage(connection)
        connection.commit()
        old_logical = _scan_business(connection, logical_storage=True)
        old_bytes = connection.execute("SELECT sum(length(storage_request_residual)) FROM authorization_requests").fetchone()[0]
        old_results = connection.execute("SELECT sum(length(result_bytes)) FROM authority_commands").fetchone()[0]
        payloads = tuple(connection.execute("SELECT payload_id,payload_digest,payload_bytes,mode FROM authority_payloads ORDER BY payload_id"))
        apply_pending_migrations(connection, applied_at=FIXED_NOW.to_text())
        assert _scan_business(connection, logical_storage=True) == old_logical
        assert tuple(connection.execute("SELECT payload_id,payload_digest,payload_bytes,mode FROM authority_payloads ORDER BY payload_id")) == payloads
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        new_bytes = connection.execute("SELECT sum(length(storage_request_residual)) FROM authorization_requests").fetchone()[0]
        new_results = connection.execute("SELECT sum(length(result_bytes)) FROM authority_commands").fetchone()[0]
        assert new_bytes < old_bytes and new_results < old_results
        assert connection.execute("SELECT count(*) FROM authorization_requests WHERE storage_request_marker=?", (b"v41",)).fetchone()[0] == 876
        for key, value in (("request_old_bytes", old_bytes), ("request_compact_bytes", new_bytes), ("result_old_bytes", old_results), ("result_compact_bytes", new_results)):
            record_property(key, value)
