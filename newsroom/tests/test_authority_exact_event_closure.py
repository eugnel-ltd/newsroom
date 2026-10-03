from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from newsroom.authority import AuthorityPersistenceError
from newsroom.authority._event_store import _EventAuthorityStore

from .authority_helpers import FIXED_NOW, command, make_service, proof


def _store(path: Path, service: object) -> _EventAuthorityStore:
    return _EventAuthorityStore(
        path,
        issuer=service._issuer,  # type: ignore[attr-defined]
        command_registry=service._registry,  # type: ignore[attr-defined]
        payload_schemas=service._payload_schemas,  # type: ignore[attr-defined]
        command_service_version="authority-command-v1",
        clock=lambda: FIXED_NOW,
    )


def _commit(store: _EventAuthorityStore, service: object, key: str):
    grant = service._authorize_for_commit(  # type: ignore[attr-defined]
        command(key=key), proof=proof()
    )
    return store.commit(grant)


def _drop(store: _EventAuthorityStore, trigger: str) -> None:
    store._execute_test_sql(f"DROP TRIGGER {trigger}")


def _tamper_authentication(store: _EventAuthorityStore, command_id: str) -> None:
    _drop(store, "immutable_authentication_contexts_update")
    store._execute_test_sql(
        "UPDATE authentication_contexts SET storage_context_marker=? WHERE "
        "authentication_context_id=(SELECT authentication_context_id FROM "
        "authority_commands WHERE command_id=?)",
        (b"{}", command_id),
    )


def _tamper_request(store: _EventAuthorityStore, command_id: str) -> None:
    _drop(store, "immutable_authorization_requests_update")
    store._execute_test_sql(
        "UPDATE authorization_requests SET storage_request_residual=? WHERE "
        "request_digest=(SELECT authorization_request_digest FROM "
        "authority_commands WHERE command_id=?)",
        (b"{}", command_id),
    )


def _tamper_decision(store: _EventAuthorityStore, command_id: str) -> None:
    _drop(store, "immutable_authorization_decisions_update")
    store._execute_test_sql(
        "UPDATE authorization_decisions SET reason_code=? WHERE "
        "authorization_decision_id=(SELECT authorization_decision_id FROM "
        "authority_commands WHERE command_id=?)",
        ("TAMPERED", command_id),
    )


def _tamper_payload(store: _EventAuthorityStore, command_id: str) -> None:
    _drop(store, "immutable_authority_payloads_update")
    store._execute_test_sql(
        "UPDATE authority_payloads SET payload_bytes=? WHERE payload_id="
        "(SELECT payload_id FROM authority_commands WHERE command_id=?)",
        (b'{"count":2,"headline":"changed"}', command_id),
    )


def _tamper_result(store: _EventAuthorityStore, command_id: str) -> None:
    _drop(store, "immutable_authority_commands_update")
    store._execute_test_sql(
        "UPDATE authority_commands SET result_bytes=? WHERE command_id=?",
        (b"{}", command_id),
    )


def _tamper_audit(store: _EventAuthorityStore, command_id: str) -> None:
    _drop(store, "immutable_authority_audit_events_update")
    store._execute_test_sql(
        "UPDATE authority_audit_events SET detail_digest=? WHERE command_id=?",
        ("sha256:" + "0" * 64, command_id),
    )


def _tamper_cardinality(store: _EventAuthorityStore, command_id: str) -> None:
    _drop(store, "immutable_authority_audit_events_delete")
    store._execute_test_sql(
        "DELETE FROM authority_audit_events WHERE command_id=?", (command_id,)
    )


def _tamper_envelope(store: _EventAuthorityStore, command_id: str) -> None:
    _drop(store, "immutable_ledger_events_update")
    store._execute_test_sql(
        "UPDATE ledger_events SET producer_version=? WHERE command_id=?",
        ("other-producer-v1", command_id),
    )


def _tamper_head(store: _EventAuthorityStore, command_id: str) -> None:
    store._execute_test_sql("PRAGMA foreign_keys=OFF")
    _drop(store, "authority_aggregates_update_guard")
    store._execute_test_sql(
        "UPDATE authority_aggregates SET current_version=999 WHERE "
        "(aggregate_type,aggregate_id)=(SELECT aggregate_type,aggregate_id "
        "FROM authority_commands WHERE command_id=?)",
        (command_id,),
    )


@pytest.mark.parametrize(
    "tamper",
    (
        _tamper_authentication,
        _tamper_request,
        _tamper_decision,
        _tamper_payload,
        _tamper_result,
        _tamper_audit,
        _tamper_cardinality,
        _tamper_envelope,
        _tamper_head,
    ),
)
def test_exact_event_closure_rejects_target_tamper(
    tmp_path: Path,
    tamper: Callable[[_EventAuthorityStore, str], None],
) -> None:
    service = make_service()
    with _store(tmp_path / "authority.sqlite3", service) as store:
        retained = _commit(store, service, tamper.__name__)
        tamper(store, retained.command_id)

        with pytest.raises(AuthorityPersistenceError):
            store._validate_retained_event(retained.event_id)


def test_exact_event_closure_does_not_scan_unrelated_history(
    tmp_path: Path,
) -> None:
    service = make_service()
    with _store(tmp_path / "authority.sqlite3", service) as store:
        target = _commit(store, service, "target")
        unrelated = _commit(store, service, "unrelated")
        _tamper_payload(store, unrelated.command_id)

        store._validate_retained_event(target.event_id)
        with pytest.raises(AuthorityPersistenceError, match="payload digest"):
            store._validate_immutable_records(store._connection)


def test_transaction_event_scope_proves_complete_selected_closure_once(tmp_path, monkeypatch):
    service = make_service()
    with _store(tmp_path / 'scoped.sqlite3', service) as store:
        selected = _commit(store, service, 'scoped-selected')
        checks = []
        original = store._validate_payload_record
        def count(conn, row):
            checks.append(row['payload_id'])
            return original(conn, row)
        monkeypatch.setattr(store, '_validate_payload_record', count)
        with store._transaction() as conn:
            with store._exact_event_read_scope(conn):
                store._validate_retained_event(selected.event_id)
                store._validate_retained_event(selected.event_id)
                assert len(checks) == 1
        store._validate_retained_event(selected.event_id)
        assert len(checks) == 2


@pytest.mark.parametrize('tamper', (_tamper_authentication, _tamper_request, _tamper_decision,
    _tamper_payload, _tamper_result, _tamper_audit, _tamper_cardinality, _tamper_envelope, _tamper_head))
def test_transaction_event_scope_rejects_same_count_selected_tamper(tmp_path, tamper):
    service = make_service()
    with _store(tmp_path / 'tamper-scope.sqlite3', service) as store:
        selected = _commit(store, service, 'selected')
        if tamper is _tamper_head:
            store._connection.execute('PRAGMA foreign_keys=OFF')
        with store._transaction() as conn:
            with store._exact_event_read_scope(conn):
                store._validate_retained_event(selected.event_id)
                tamper(store, selected.command_id)
                with pytest.raises(AuthorityPersistenceError):
                    store._validate_retained_event(selected.event_id)


def test_transaction_event_scope_rejects_missing_or_replaced_transaction(tmp_path):
    service = make_service()
    with _store(tmp_path / 'transaction-scope.sqlite3', service) as store:
        selected = _commit(store, service, 'selected')
        with pytest.raises(AuthorityPersistenceError, match='transaction'):
            with store._exact_event_read_scope(store._connection):
                pass
        conn = store._connection
        conn.execute('BEGIN')
        with pytest.raises(AuthorityPersistenceError, match='transaction'):
            with store._exact_event_read_scope(conn):
                store._validate_retained_event(selected.event_id)
                conn.execute('ROLLBACK')
                conn.execute('BEGIN')
                store._validate_retained_event(selected.event_id)
        if conn.in_transaction:
            conn.execute('ROLLBACK')


def test_transaction_event_scope_is_connection_bound_and_rollback_invalidates(tmp_path, monkeypatch):
    import sqlite3
    service = make_service()
    path = tmp_path / 'rollback-scope.sqlite3'
    with _store(path, service) as store:
        selected = _commit(store, service, 'selected')
        checks = []
        original = store._validate_payload_record
        def count(conn, row):
            checks.append(row['payload_id'])
            return original(conn, row)
        monkeypatch.setattr(store, '_validate_payload_record', count)
        with sqlite3.connect(path) as other:
            other.execute('BEGIN')
            with pytest.raises(AuthorityPersistenceError, match='transaction'):
                with store._exact_event_read_scope(other):
                    pass
        with store._transaction() as conn:
            with store._exact_event_read_scope(conn):
                store._validate_retained_event(selected.event_id)
                conn.execute('SAVEPOINT fixture_mutation')
                _tamper_payload(store, selected.command_id)
                conn.execute('ROLLBACK TO fixture_mutation')
                conn.execute('RELEASE fixture_mutation')
                store._validate_retained_event(selected.event_id)
                assert len(checks) == 2
        _tamper_payload(store, selected.command_id)
        with store._transaction() as conn:
            with store._exact_event_read_scope(conn):
                with pytest.raises(AuthorityPersistenceError, match='payload digest'):
                    store._validate_retained_event(selected.event_id)


def test_transaction_event_scope_does_not_sweep_unrelated_history(tmp_path):
    service = make_service()
    with _store(tmp_path / 'selected-scope.sqlite3', service) as store:
        selected = _commit(store, service, 'selected')
        unrelated = _commit(store, service, 'unrelated')
        with store._transaction() as conn:
            with store._exact_event_read_scope(conn):
                store._validate_retained_event(selected.event_id)
                _tamper_payload(store, unrelated.command_id)
                store._validate_retained_event(selected.event_id)
                with pytest.raises(AuthorityPersistenceError, match='payload digest'):
                    store._validate_retained_event(unrelated.event_id)
