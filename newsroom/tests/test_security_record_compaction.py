from __future__ import annotations

import json
import sqlite3

import pytest

from newsroom.authority import AuthorityPersistenceError, canonical_json_bytes
from newsroom.authority import authorization_request_storage_migrations as request_migration
from newsroom.authority import command_bound_storage_migrations as command_migration
from newsroom.authority import graphiti_recovered_ambiguous_migrations as recovery_migration
from newsroom.authority import relationship_open_index_migrations as index_migration
from newsroom.authority import projection_retirement_migrations as retirement_migration
from newsroom.authority import security_record_migrations as migration
from newsroom.authority.canonical import digest_bytes
from newsroom.authority.migrations import (
    SCHEMA_VERSION,
    apply_pending_migrations,
    schema_fingerprint,
)

from .authority_event_helpers import open_test_system
from .authority_helpers import command, proof
from .graphiti_adapter_4d_migration_helpers import _drop_v37_security_record_schema
from .graphiti_adapter_4d_migration_helpers import (
    _drop_v38_authorization_request_storage,
)


def test_current_security_storage_is_compact_with_exact_public_provenance(tmp_path):
    path = tmp_path / 'authority.sqlite3'
    original_command = command(key='compact-one')
    with open_test_system(path) as system:
        result = system.commands.execute(original_command, proof=proof())
        before = system.events.provenance(result.event_id, proof=proof())
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        assert connection.execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION
        auth = connection.execute('SELECT * FROM authentication_contexts').fetchone()
        request = connection.execute('SELECT * FROM authorization_requests').fetchone()
        assert bytes(auth['storage_context_marker']) == b'v37'
        assert bytes(request['storage_request_marker']) == b'v41'
        residual = json.loads(bytes(request['storage_request_residual']))
        assert not {
            'request_digest', 'authentication_context_id', 'principal_id',
            'authority_domain', 'operation_type', 'required_scope',
        }.intersection(residual)
        assert len(request['storage_request_residual']) < len(
            before.authorization_request.canonical_bytes
        )
        history = tuple(
            tuple(row) for row in connection.execute(
                'SELECT version,name,checksum FROM authority_migrations '
                'WHERE version>=37 ORDER BY version'
            )
        )
        assert history == (
            (37, migration.SECURITY_RECORD_MIGRATION_NAME,
             migration.SECURITY_RECORD_MIGRATION_CHECKSUM),
            (38, request_migration.AUTHORIZATION_REQUEST_STORAGE_MIGRATION_NAME,
             request_migration.AUTHORIZATION_REQUEST_STORAGE_MIGRATION_CHECKSUM),
            (39, recovery_migration.GRAPHITI_RECOVERED_AMBIGUOUS_MIGRATION_NAME,
             recovery_migration.GRAPHITI_RECOVERED_AMBIGUOUS_MIGRATION_CHECKSUM),
            (40, index_migration.RELATIONSHIP_OPEN_INDEX_MIGRATION_NAME,
             index_migration.RELATIONSHIP_OPEN_INDEX_MIGRATION_CHECKSUM),
            (41, command_migration.COMMAND_BOUND_STORAGE_MIGRATION_NAME,
             command_migration.COMMAND_BOUND_STORAGE_MIGRATION_CHECKSUM),
            (42, retirement_migration.PROJECTION_RETIREMENT_MIGRATION_NAME,
             retirement_migration.PROJECTION_RETIREMENT_MIGRATION_CHECKSUM),
            (43, 'native_projection_retirement_lookup_indexes_v43',
             'sha256:f9aecc8859aec5b6a19acbacb8b1eea7e1a179c99a8d01b638976f4080b52832'),
        )
        assert connection.execute('PRAGMA foreign_key_check').fetchall() == []
    with open_test_system(path) as system:
        after = system.events.provenance(result.event_id, proof=proof())
        replay = system.commands.execute(original_command, proof=proof())
        system.commands.execute(command(key='compact-two'), proof=proof())
    assert replay.replayed
    assert after == before
    assert digest_bytes(after.authentication.canonical_bytes) == after.authentication.canonical_digest
    assert digest_bytes(after.authorization_request.canonical_bytes) == after.authorization_request.canonical_record_digest
    assert canonical_json_bytes(json.loads(after.authorization_request.canonical_bytes)) == before.authorization_request.canonical_bytes


def test_v38_request_decoder_does_not_reparse_reconstructed_bytes(
    tmp_path, monkeypatch
):
    from newsroom.authority._event_store import _EventAuthorityStore

    path = tmp_path / 'single-decode.sqlite3'
    with open_test_system(path) as system:
        result = system.commands.execute(command(key='single-decode'), proof=proof())
        expected = system.events.provenance(
            result.event_id, proof=proof()
        ).authorization_request.canonical_bytes
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = dict(connection.execute('SELECT * FROM authorization_requests').fetchone())
        # The historical decoder's no-reparse contract remains explicitly v38.
        row['storage_request_marker'] = b'v38'
        row['storage_request_residual'] = canonical_json_bytes({
            key: value for key, value in json.loads(expected).items()
            if key not in request_migration.AUTHORIZATION_REQUEST_INDEXED_FIELDS
        })
        reader = object.__new__(_EventAuthorityStore)
        monkeypatch.setattr(
            reader,
            '_decode_canonical',
            lambda _data: pytest.fail('reparsed reconstructed request bytes'),
        )
        assert request_migration.authorization_request_bytes_from_v38_row(row) == expected
        assert reader._request_record_from_row(row).canonical_bytes == expected


def _v36_path(tmp_path, *, count=2):
    path = tmp_path / 'old.sqlite3'
    commands = tuple(command(key=f'old-{index}') for index in range(count))
    with open_test_system(path) as system:
        results = tuple(system.commands.execute(item, proof=proof()) for item in commands)
        originals = tuple(system.events.provenance(item.event_id, proof=proof()) for item in results)
    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA foreign_keys=ON')
        _drop_v37_security_record_schema(connection)
        assert schema_fingerprint(connection) == migration.SECURITY_RECORD_PREDECESSOR_FINGERPRINT
        assert connection.execute('PRAGMA foreign_key_check').fetchall() == []
    return path, commands, originals


def _v37_path(tmp_path, *, count=3):
    path = tmp_path / 'request-v37.sqlite3'
    commands = tuple(command(key=f'request-old-{index}') for index in range(count))
    with open_test_system(path) as system:
        results = tuple(system.commands.execute(item, proof=proof()) for item in commands)
        originals = tuple(system.events.provenance(item.event_id, proof=proof()) for item in results)
    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA foreign_keys=ON')
        _drop_v38_authorization_request_storage(connection)
        assert schema_fingerprint(connection) == (
            request_migration.AUTHORIZATION_REQUEST_STORAGE_PREDECESSOR_FINGERPRINT
        )
        assert connection.execute('PRAGMA foreign_key_check').fetchall() == []
    return path, commands, originals


def _state(connection):
    return (connection.execute('PRAGMA user_version').fetchone()[0], schema_fingerprint(connection),
            tuple(connection.execute('SELECT * FROM authority_migrations ORDER BY version')),
            tuple(connection.execute('SELECT * FROM authentication_contexts ORDER BY authentication_context_id')),
            tuple(connection.execute('SELECT * FROM authorization_requests ORDER BY request_digest')))


def _change(connection, table, sql, parameters=()):
    name = f'immutable_{table}_update'
    guard = connection.execute('SELECT sql FROM sqlite_master WHERE name=?', (name,)).fetchone()[0]
    connection.execute(f'DROP TRIGGER {name}')
    connection.execute(sql, parameters)
    connection.execute(guard)


def test_v36_upgrade_preserves_provenance_replay_and_extension_residual(tmp_path):
    path, commands, originals = _v36_path(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = dict(connection.execute('SELECT * FROM authorization_requests LIMIT 1').fetchone())
        value = json.loads(row['canonical_bytes'])
        value['future_extension'] = {'unicode': '證據', 'nullable': None, 'items': [1, True]}
        value.pop('request_digest')
        unsigned = canonical_json_bytes(value)
        value['request_digest'] = digest_bytes(unsigned)
        full = canonical_json_bytes(value)
        row.update(request_digest=value['request_digest'], canonical_bytes=full, canonical_record_digest=digest_bytes(full))
        connection.execute('INSERT INTO authorization_requests VALUES(?,?,?,?,?,?,?,?,?)', tuple(row.values()))
    with open_test_system(path) as system:
        for original in originals:
            assert system.events.provenance(original.event.event_id, proof=proof()) == original
        assert system.commands.execute(commands[0], proof=proof()).replayed
    from newsroom.authority._event_store import _EventAuthorityStore
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        retained = connection.execute('SELECT * FROM authorization_requests WHERE request_digest=?', (value['request_digest'],)).fetchone()
        reader = object.__new__(_EventAuthorityStore)
        assert reader._request_record_from_row(retained).canonical_bytes == full
        assert bytes(retained['storage_request_marker']) == b'v38'
        assert retained['recorded_at'] == row['recorded_at']
        residual = json.loads(bytes(retained['storage_request_residual']))
        assert residual['future_extension'] == value['future_extension']
        assert not set(request_migration.AUTHORIZATION_REQUEST_INDEXED_FIELDS).intersection(residual)
        assert connection.execute('PRAGMA foreign_key_check').fetchall() == []


@pytest.mark.parametrize('tamper', ('context_json', 'context_index', 'context_digest'))
def test_v36_corruption_rejected_before_conversion_with_atomic_rollback(tmp_path, tamper):
    path, _, _ = _v36_path(tmp_path)
    with sqlite3.connect(path) as connection:
        table = 'authentication_contexts'
        if tamper == 'context_json':
            sql, parameters = f'UPDATE {table} SET canonical_bytes=?', (b'{ }',)
        elif tamper == 'context_index':
            sql, parameters = f'UPDATE {table} SET principal_id=?', ('changed',)
        else:
            sql, parameters = f'UPDATE {table} SET canonical_digest=?', ('sha256:' + 'f'*64,)
        _change(connection, table, sql + ' WHERE rowid=(SELECT max(rowid) FROM authentication_contexts)', parameters)
        connection.commit()
        before = _state(connection)
        updates = []
        connection.set_trace_callback(lambda sql: updates.append(sql) if sql.startswith(
            'UPDATE authentication_contexts SET canonical_bytes='
        ) else None)
        with pytest.raises(sqlite3.IntegrityError):
            apply_pending_migrations(connection, applied_at='2026-09-13T00:00:00.000000Z')
        assert updates == []
        assert _state(connection) == before


@pytest.mark.parametrize('counter', ('validations', 'updates'))
def test_v37_validates_once_then_compacts_with_one_update(tmp_path, monkeypatch, counter):
    path, _, _ = _v36_path(tmp_path, count=3)
    validations = []
    updates = []
    validate = migration._validate_old_context

    def counted(row):
        validate(row)
        validations.append(row['authentication_context_id'])

    monkeypatch.setattr(migration, '_validate_old_context', counted)
    with sqlite3.connect(path) as connection:
        connection.set_trace_callback(lambda sql: updates.append(sql) if sql.startswith(
            'UPDATE authentication_contexts SET canonical_bytes='
        ) else None)
        apply_pending_migrations(connection, applied_at='2026-09-13T00:00:00.000000Z')
        if counter == 'validations':
            assert len(validations) == 3
        else:
            assert len(updates) == 1
        assert len(set(validations)) == 3
        assert connection.execute(
            'SELECT count(*) FROM authentication_contexts WHERE storage_context_marker=?',
            (b'v37',),
        ).fetchone()[0] == 3


def test_v37_conversion_failure_after_update_rolls_back_exactly(tmp_path):
    class FailedUpdate(sqlite3.Connection):
        converted = False

        def execute(self, sql, parameters=()):
            cursor = super().execute(sql, parameters)
            if sql.startswith('UPDATE authentication_contexts SET canonical_bytes='):
                assert super().execute(
                    'SELECT count(*) FROM authentication_contexts WHERE canonical_bytes=?',
                    (b'v37',),
                ).fetchone()[0] > 0
                self.converted = True
                raise sqlite3.IntegrityError('injected after compacting contexts')
            return cursor

    path, _, _ = _v36_path(tmp_path)
    with sqlite3.connect(path, factory=FailedUpdate) as connection:
        before = _state(connection)
        with pytest.raises(sqlite3.IntegrityError, match='after compacting contexts'):
            apply_pending_migrations(connection, applied_at='2026-09-13T00:00:00.000000Z')
        assert connection.converted
        assert _state(connection) == before


def test_v38_validates_all_requests_before_the_first_update(tmp_path):
    path, _, _ = _v37_path(tmp_path)
    with sqlite3.connect(path) as connection:
        _change(
            connection,
            'authorization_requests',
            "UPDATE authorization_requests SET canonical_record_digest=? "
            "WHERE rowid=(SELECT max(rowid) FROM authorization_requests)",
            ('sha256:' + 'f' * 64,),
        )
        connection.commit()
        before = _state(connection)
        updates = []
        connection.set_trace_callback(
            lambda sql: updates.append(sql)
            if sql.startswith(
                'UPDATE authorization_requests SET canonical_bytes='
            )
            else None
        )
        with pytest.raises(sqlite3.IntegrityError):
            apply_pending_migrations(
                connection, applied_at='2026-09-13T00:00:00.000000Z'
            )
        assert updates == []
        assert _state(connection) == before


def test_v38_mid_update_failure_rolls_back_schema_rows_and_history(tmp_path):
    class FailedUpdate(sqlite3.Connection):
        updates = 0

        def execute(self, sql, parameters=()):
            cursor = super().execute(sql, parameters)
            if sql.startswith(
                'UPDATE authorization_requests SET canonical_bytes='
            ):
                self.updates += 1
                if self.updates == 2:
                    raise sqlite3.IntegrityError('injected request residual failure')
            return cursor

    path, _, _ = _v37_path(tmp_path)
    with sqlite3.connect(path, factory=FailedUpdate) as connection:
        before = _state(connection)
        with pytest.raises(sqlite3.IntegrityError, match='request residual'):
            apply_pending_migrations(
                connection, applied_at='2026-09-13T00:00:00.000000Z'
            )
        assert connection.updates == 2
        assert _state(connection) == before


def test_v38_request_conversion_streams_without_fetchall(tmp_path):
    class StreamingCursor(sqlite3.Cursor):
        request_stream = False

        def execute(self, sql, parameters=()):
            self.request_stream = sql.startswith(
                'SELECT rowid,* FROM authorization_requests'
            )
            return super().execute(sql, parameters)

        def fetchall(self):
            if self.request_stream:
                raise AssertionError('request migration fetched all rows')
            return super().fetchall()

    class StreamingConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            return self.cursor(factory=StreamingCursor).execute(sql, parameters)

    path, _, originals = _v37_path(tmp_path, count=4)
    with sqlite3.connect(path, factory=StreamingConnection) as connection:
        apply_pending_migrations(
            connection, applied_at='2026-09-13T00:00:00.000000Z'
        )
        assert connection.execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION
    with open_test_system(path) as system:
        for original in originals:
            assert system.events.provenance(
                original.event.event_id, proof=proof()
            ) == original


def test_v38_open_streams_request_validation_without_fetchall(
    tmp_path, monkeypatch
):
    path = tmp_path / 'streamed-open.sqlite3'
    with open_test_system(path) as system:
        system.commands.execute(command(key='streamed-open'), proof=proof())

    original_connect = sqlite3.connect

    class StreamingCursor(sqlite3.Cursor):
        request_stream = False

        def execute(self, sql, parameters=()):
            self.request_stream = sql == 'SELECT * FROM authorization_requests'
            return super().execute(sql, parameters)

        def fetchall(self):
            if self.request_stream:
                raise AssertionError('OPEN fetched all authorization requests')
            return super().fetchall()

    class StreamingConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            return self.cursor(factory=StreamingCursor).execute(sql, parameters)

    def streaming_connect(*args, **kwargs):
        kwargs['factory'] = StreamingConnection
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, 'connect', streaming_connect)
    with open_test_system(path):
        pass


@pytest.mark.parametrize(
    'tamper',
    ('marker', 'residual', 'reserved', 'index', 'record_digest', 'request_digest'),
)
def test_v38_reopen_rejects_request_representation_corruption(tmp_path, tamper):
    path = tmp_path / f'request-{tamper}.sqlite3'
    with open_test_system(path) as system:
        system.commands.execute(command(key=tamper), proof=proof())
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute('SELECT * FROM authorization_requests').fetchone()
        if tamper == 'marker':
            connection.execute('PRAGMA ignore_check_constraints=ON')
            sql, parameters = (
                'UPDATE authorization_requests SET storage_request_marker=?',
                (b'v37',),
            )
        elif tamper == 'residual':
            sql, parameters = (
                'UPDATE authorization_requests SET storage_request_residual=?',
                (b'{ }',),
            )
        elif tamper == 'reserved':
            residual = json.loads(bytes(row['storage_request_residual']))
            residual['principal_id'] = row['principal_id']
            sql, parameters = (
                'UPDATE authorization_requests SET storage_request_residual=?',
                (canonical_json_bytes(residual),),
            )
        elif tamper == 'index':
            sql, parameters = (
                'UPDATE authorization_requests SET principal_id=?',
                ('changed',),
            )
        elif tamper == 'record_digest':
            sql, parameters = (
                'UPDATE authorization_requests SET canonical_record_digest=?',
                ('sha256:' + 'f' * 64,),
            )
        else:
            residual = json.loads(bytes(row['storage_request_residual']))
            residual['aggregate_id'] = 'changed'
            reconstructed = {
                **residual,
                **{
                    name: str(row[name])
                    for name in request_migration.AUTHORIZATION_REQUEST_INDEXED_FIELDS
                },
            }
            sql, parameters = (
                'UPDATE authorization_requests SET storage_request_residual=?, '
                'canonical_record_digest=?',
                (canonical_json_bytes(residual), digest_bytes(canonical_json_bytes(reconstructed))),
            )
        _change(connection, 'authorization_requests', sql, parameters)
        connection.execute('PRAGMA ignore_check_constraints=OFF')
    with pytest.raises(AuthorityPersistenceError):
        with open_test_system(path):
            pass


@pytest.mark.parametrize('tamper', ('context_marker', 'context_index'))
def test_compact_reopen_preserves_all_integrity_checks(tmp_path, tamper):
    path = tmp_path / 'reopen.sqlite3'
    with open_test_system(path) as system:
        system.commands.execute(command(key='compact'), proof=proof())
    with sqlite3.connect(path) as connection:
        table = 'authentication_contexts'
        if tamper == 'context_marker':
            sql, parameters = 'UPDATE authentication_contexts SET storage_context_marker=?', (b'v36',)
        else:
            sql, parameters = f'UPDATE {table} SET principal_id=?', ('changed',)
        _change(connection, table, sql, parameters)
    with pytest.raises(AuthorityPersistenceError):
        with open_test_system(path):
            pass


@pytest.mark.parametrize('boundary', ('no_transaction', 'schema', 'history'))
def test_v37_requires_exact_atomic_predecessor(tmp_path, boundary):
    path, _, _ = _v36_path(tmp_path)
    with sqlite3.connect(path) as connection:
        history = tuple(tuple(row[:3]) for row in connection.execute('SELECT * FROM authority_migrations ORDER BY version'))
        if boundary == 'schema':
            connection.execute('CREATE INDEX unexpected_security_index ON authentication_contexts(principal_id)')
            connection.commit()
        before = _state(connection)
        if boundary != 'no_transaction':
            connection.execute('BEGIN EXCLUSIVE')
        with pytest.raises(sqlite3.DatabaseError, match='active transaction|exact checked'):
            migration.migrate_security_records(connection, expected_history=() if boundary == 'history' else history)
        connection.rollback()
        assert _state(connection) == before


@pytest.mark.parametrize('boundary', ('no_transaction', 'schema', 'history'))
def test_v38_requires_exact_atomic_predecessor(tmp_path, boundary):
    path, _, _ = _v37_path(tmp_path)
    with sqlite3.connect(path) as connection:
        history = tuple(
            tuple(row[:3])
            for row in connection.execute(
                'SELECT * FROM authority_migrations ORDER BY version'
            )
        )
        if boundary == 'schema':
            connection.execute(
                'CREATE INDEX unexpected_request_index '
                'ON authorization_requests(principal_id)'
            )
            connection.commit()
        before = _state(connection)
        if boundary != 'no_transaction':
            connection.execute('BEGIN EXCLUSIVE')
        with pytest.raises(sqlite3.DatabaseError, match='active transaction|exact checked'):
            request_migration.migrate_authorization_request_storage(
                connection,
                expected_history=() if boundary == 'history' else history,
            )
        connection.rollback()
        assert _state(connection) == before


@pytest.mark.parametrize(('version', 'expected_calls'), ((36, 0), (37, 1)))
def test_exact_prefix_invokes_v37_only_at_its_version(tmp_path, monkeypatch, version, expected_calls):
    from newsroom.authority import migrations
    from .authority_migration_compatibility import build_exact_prefix
    calls = 0
    original = migrations.migrate_security_records

    def counted(connection, *, expected_history):
        nonlocal calls
        calls += 1
        assert connection.in_transaction
        original(connection, expected_history=expected_history)
    monkeypatch.setattr(migrations, 'migrate_security_records', counted)
    build_exact_prefix(tmp_path / f'prefix-{version}.sqlite3', version)
    assert calls == expected_calls


@pytest.mark.parametrize(('version', 'expected_calls'), ((37, 0), (38, 1)))
def test_exact_prefix_invokes_v38_only_at_its_version(
    tmp_path, monkeypatch, version, expected_calls
):
    from newsroom.authority import migrations
    from .authority_migration_compatibility import build_exact_prefix

    calls = 0
    original = migrations.migrate_authorization_request_storage

    def counted(connection, *, expected_history):
        nonlocal calls
        calls += 1
        assert connection.in_transaction
        original(connection, expected_history=expected_history)

    monkeypatch.setattr(
        migrations, 'migrate_authorization_request_storage', counted
    )
    build_exact_prefix(tmp_path / f'request-prefix-{version}.sqlite3', version)
    assert calls == expected_calls


def test_v37_downgrade_is_lossless_inside_callers_transaction(tmp_path):
    path = tmp_path / 'rollback.sqlite3'
    with open_test_system(path) as system:
        system.commands.execute(command(key='nested'), proof=proof())
    with sqlite3.connect(path, isolation_level=None) as connection:
        connection.execute('PRAGMA foreign_keys=ON')
        before = _state(connection)
        connection.execute('SAVEPOINT caller')
        _drop_v37_security_record_schema(connection)
        assert connection.in_transaction
        assert connection.execute('PRAGMA foreign_keys').fetchone() == (1,)
        assert connection.execute('PRAGMA foreign_key_check').fetchall() == []
        assert schema_fingerprint(connection) == migration.SECURITY_RECORD_PREDECESSOR_FINGERPRINT
        connection.execute('ROLLBACK TO SAVEPOINT caller')
        connection.execute('RELEASE SAVEPOINT caller')
        assert _state(connection) == before
