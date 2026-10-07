"""Explicit owned native45→46 lossless marker storage; never a boot upgrade."""
from __future__ import annotations

import hashlib
from itertools import zip_longest
from uuid import UUID, RFC_4122

from .canonical import canonical_json_bytes, digest_canonical
from .persistence import AuthorityPersistenceError

VERSION = 46
NAME = 'native_expired_marker_dictionary_codec_v46'
STATEMENTS = (
    'retain every logical namespace/key/command/event/sequence/security/trust/aggregate identity',
    'normalise namespace independently of scope tuples; preserve opaque keys and noncanonical identifier text',
    'store only canonical UUIDv4 identities as sixteen-byte blobs; preserve indexed no-revival and provenance guards',
    'rewrite only markers under the owned idle writer transaction; prove ordered logical parity and exact schema history',
)
CHECKSUM = digest_canonical({'version': VERSION, 'name': NAME, 'statements': list(STATEMENTS)})
FINGERPRINT = 'sha256:66ef7f14560689fbbac6425117f286c3a6c9ddec811e17e0be5577d1d8e748ab'
_SHADOW = '_native_marker_codec46'


def encode_identity(value):
    if type(value) is not str:
        raise AuthorityPersistenceError('marker identity must retain its original text')
    try:
        identity = UUID(value)
    except ValueError:
        return value
    return identity.bytes if str(identity) == value and identity.version == 4 and identity.variant == RFC_4122 else value


def decode_identity(value):
    if type(value) is str:
        if type(encode_identity(value)) is not str:
            raise AuthorityPersistenceError('canonical marker identity has noncanonical storage')
        return value
    if type(value) is bytes and len(value) == 16:
        identity = UUID(bytes=value)
        if identity.version == 4 and identity.variant == RFC_4122:
            return str(identity)
    raise AuthorityPersistenceError('marker identifier codec differs')


def _canonical_uuid_text(expression):
    return (f"(typeof({expression})='text' AND length({expression})=36 "
        f"AND substr({expression},9,1)='-' AND substr({expression},14,1)='-' "
        f"AND substr({expression},19,1)='-' AND substr({expression},24,1)='-' "
        f"AND length(replace({expression},'-',''))=32 "
        f"AND replace({expression},'-','') NOT GLOB '*[^0-9a-f]*' "
        f"AND substr({expression},15,1)='4' AND substr({expression},20,1) IN ('8','9','a','b'))")


def _sql_identity(expression):
    return f"CASE WHEN {_canonical_uuid_text(expression)} THEN unhex(replace({expression},'-','')) ELSE {expression} END"


def _identity_column(name):
    return (f"{name} ANY NOT NULL CHECK((typeof({name})='blob' AND length({name})=16 "
        f"AND substr(hex({name}),13,1)='4' AND substr(hex({name}),17,1) IN ('8','9','A','B')) "
        f"OR (typeof({name})='text' AND NOT {_canonical_uuid_text(name)}))")


def require_codec_capability(connection):
    try:
        if connection.execute("SELECT unhex('00')").fetchone()[0] != b'\x00':
            raise ValueError
    except Exception as error:
        raise AuthorityPersistenceError('native46 requires SQLite unhex capability') from error


def marker_rows(connection, *, table='native_expired_command_keys'):
    if table not in {'native_expired_command_keys', _SHADOW}:
        raise ValueError('marker table differs')
    compact = table == _SHADOW or connection.execute('PRAGMA user_version').fetchone()[0] == VERSION
    if not compact:
        yield from (tuple(row) for row in connection.execute(
            'SELECT namespace,key,command_id,event_id,ledger_seq,security_scope,trust_scope,aggregate_type,aggregate_id '
            'FROM native_expired_command_keys ORDER BY namespace,key'))
        return
    for row in connection.execute(f'SELECT n.namespace,m.key,m.command_id,m.event_id,m.ledger_seq,'
            f's.security_scope,s.trust_scope,s.aggregate_type,m.aggregate_id FROM native_marker_namespaces n '
            f'CROSS JOIN {table} m ON m.namespace_id=n.namespace_id '
            'JOIN native_marker_scopes s ON s.scope_id=m.scope_id ORDER BY n.namespace,m.key'):
        values = list(row)
        for i in (2, 3, 8):
            values[i] = decode_identity(values[i])
        yield tuple(values)


def insert_markers(connection, rows):
    """Selected-store writer retains the same nine logical fields in either layout."""
    if connection.execute('PRAGMA user_version').fetchone()[0] != VERSION:
        connection.executemany('INSERT INTO native_expired_command_keys VALUES(?,?,?,?,?,?,?,?,?)', rows)
        return
    for namespace,key,command,event,seq,security,trust,aggregate_type,aggregate in rows:
        connection.execute('INSERT OR IGNORE INTO native_marker_namespaces(namespace) VALUES(?)', (namespace,))
        connection.execute('INSERT OR IGNORE INTO native_marker_scopes(security_scope,trust_scope,aggregate_type) VALUES(?,?,?)',
            (security,trust,aggregate_type))
        connection.execute('INSERT INTO native_expired_command_keys VALUES('
            '(SELECT namespace_id FROM native_marker_namespaces WHERE namespace=?),?,?,?,?, '
            '(SELECT scope_id FROM native_marker_scopes WHERE security_scope=? AND trust_scope=? AND aggregate_type=?),?)',
            (namespace,key,encode_identity(command),encode_identity(event),seq,security,trust,aggregate_type,encode_identity(aggregate)))


def _rewrite_codec(connection):
    require_codec_capability(connection)
    connection.execute('CREATE TABLE native_marker_namespaces(namespace_id INTEGER PRIMARY KEY,namespace TEXT NOT NULL UNIQUE) STRICT')
    connection.execute('CREATE TABLE native_marker_scopes(scope_id INTEGER PRIMARY KEY,security_scope TEXT NOT NULL,'
        'trust_scope TEXT NOT NULL,aggregate_type TEXT NOT NULL,UNIQUE(security_scope,trust_scope,aggregate_type)) STRICT')
    connection.execute('CREATE INDEX idx_native_marker_scope_type ON native_marker_scopes(aggregate_type)')
    connection.execute(f'CREATE TABLE {_SHADOW}(namespace_id INTEGER NOT NULL REFERENCES native_marker_namespaces(namespace_id),'
        f'key TEXT NOT NULL,{_identity_column("command_id")} UNIQUE,{_identity_column("event_id")} UNIQUE,'
        f'ledger_seq INTEGER NOT NULL UNIQUE,scope_id INTEGER NOT NULL REFERENCES native_marker_scopes(scope_id),'
        f'{_identity_column("aggregate_id")},PRIMARY KEY(namespace_id,key)) STRICT')
    connection.execute('INSERT INTO native_marker_namespaces(namespace) SELECT DISTINCT namespace FROM native_expired_command_keys')
    connection.execute('INSERT INTO native_marker_scopes(security_scope,trust_scope,aggregate_type) '
        'SELECT DISTINCT security_scope,trust_scope,aggregate_type FROM native_expired_command_keys')
    selected = connection.execute('SELECT n.namespace_id,m.key,m.command_id,m.event_id,m.ledger_seq,s.scope_id,m.aggregate_id '
        'FROM native_expired_command_keys m JOIN native_marker_namespaces n ON n.namespace=m.namespace '
        'JOIN native_marker_scopes s ON s.security_scope=m.security_scope AND s.trust_scope=m.trust_scope '
        'AND s.aggregate_type=m.aggregate_type ORDER BY m.namespace,m.key')
    try:
        while batch := selected.fetchmany(256):
            connection.executemany(f'INSERT INTO {_SHADOW} VALUES(?,?,?,?,?,?,?)',
                ((row[0],row[1],encode_identity(row[2]),encode_identity(row[3]),row[4],row[5],encode_identity(row[6])) for row in batch))
    finally:
        selected.close()
    digest = hashlib.sha256()
    count = 0
    for old,new in zip_longest(marker_rows(connection), marker_rows(connection,table=_SHADOW)):
        if old is None or old != new:
            raise AuthorityPersistenceError('marker codec logical parity differs')
        digest.update(canonical_json_bytes(list(old)))
        count += 1
    guards = tuple(connection.execute("SELECT name FROM sqlite_schema WHERE type='trigger' AND sql LIKE '%native_expired_command_keys%'"))
    for (name,) in guards:
        connection.execute('DROP TRIGGER "'+name+'"')
    connection.execute('DROP TABLE native_expired_command_keys')
    connection.execute(f'ALTER TABLE {_SHADOW} RENAME TO native_expired_command_keys')
    connection.execute('CREATE INDEX idx_native_expired_aggregate ON native_expired_command_keys(scope_id,aggregate_id)')
    for table in ('native_expired_command_keys','native_marker_namespaces','native_marker_scopes'):
        for operation in ('UPDATE','DELETE'):
            connection.execute(f"CREATE TRIGGER immutable_{table}_{operation.lower()} BEFORE {operation} ON {table} "
                "BEGIN SELECT RAISE(ABORT,'immutable native checkpoint'); END")
    connection.execute(f'''CREATE TRIGGER native_expired_command_insert_guard BEFORE INSERT ON authority_commands
        WHEN EXISTS(SELECT 1 FROM native_expired_command_keys WHERE command_id={_sql_identity('NEW.command_id')})
          OR EXISTS(SELECT 1 FROM native_marker_namespaces n CROSS JOIN native_expired_command_keys m
             ON m.namespace_id=n.namespace_id WHERE n.namespace=NEW.idempotency_namespace AND m.key=NEW.idempotency_key)
        BEGIN SELECT RAISE(ABORT,'command diagnostic history expired; identity remains reserved'); END''')
    connection.execute(f'''CREATE TRIGGER native_expired_event_insert_guard BEFORE INSERT ON ledger_events
        WHEN EXISTS(SELECT 1 FROM native_expired_command_keys WHERE event_id={_sql_identity('NEW.event_id')} OR ledger_seq=NEW.ledger_seq)
        BEGIN SELECT RAISE(ABORT,'event diagnostic history expired; identity remains reserved'); END''')
    connection.execute(f'''CREATE TRIGGER native_expired_aggregate_insert_guard BEFORE INSERT ON authority_aggregates
        WHEN EXISTS(SELECT 1 FROM native_marker_scopes s CROSS JOIN native_expired_command_keys m
          ON m.scope_id=s.scope_id WHERE s.aggregate_type=NEW.aggregate_type AND m.aggregate_id={_sql_identity('NEW.aggregate_id')})
        BEGIN SELECT RAISE(ABORT,'aggregate diagnostic history expired; identity remains reserved'); END''')
    if connection.execute('PRAGMA foreign_key_check').fetchone():
        raise AuthorityPersistenceError('marker codec foreign-key integrity differs')
    return {'rows': count, 'ordered_marker_digest': 'sha256:'+digest.hexdigest()}


def _apply_codec(connection, applied_at):
    result = _rewrite_codec(connection)
    connection.execute('INSERT INTO authority_migrations VALUES(?,?,?,?)', (VERSION,NAME,CHECKSUM,applied_at))
    connection.execute('PRAGMA user_version=46')
    from .native_current_checkpoint_migrations import require_checkpoint_schema
    require_checkpoint_schema(connection)
    return result


def initialise_empty_codec_store(connection):
    from .native_marker_layout_migrations import initialise_empty_marker_store
    initialise_empty_marker_store(connection)
    try:
        connection.execute('BEGIN EXCLUSIVE')
        _apply_codec(connection, '1970-01-01T00:00:00.000000Z')
        connection.commit()
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def migrate_native_marker_codec(store, *, dev_rebuild=False, before_commit=lambda:None):
    from .native_current_checkpoint_migrations import require_checkpoint_schema
    if not dev_rebuild or not store._current_state_only or not store._native_checkpoint_schema or store._lock_fd is None:
        raise AuthorityPersistenceError('marker codec migration requires explicit owned native DEV writer')
    with store._lock:
        connection = store._connection
        if connection.in_transaction:
            raise AuthorityPersistenceError('marker codec migration requires idle owned writer')
        require_checkpoint_schema(connection)
        version = connection.execute('PRAGMA user_version').fetchone()[0]
        if version == VERSION:
            return {'migrated': False}
        if version != 45:
            raise AuthorityPersistenceError('marker codec migration requires exact native45')
        require_codec_capability(connection)
        try:
            with store._transaction() as conn:
                if conn.execute('PRAGMA foreign_keys').fetchone()[0] != 1:
                    raise AuthorityPersistenceError('native foreign keys are required')
                result = _apply_codec(conn, store._clock().to_text())
                before_commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        store._native_marker_codec = True
    return {'migrated': True, **result, 'physical_file_reclaim': False}
