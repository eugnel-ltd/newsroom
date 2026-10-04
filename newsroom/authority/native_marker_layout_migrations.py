"""Explicit owned native44→45 marker-only physical rewrite; never auto-upgrade."""
from __future__ import annotations

import hashlib
from itertools import zip_longest

from .canonical import canonical_json_bytes,digest_canonical
from .persistence import AuthorityPersistenceError

VERSION=45
NAME='native_expired_marker_rowid_layout_v45'
STATEMENTS=(
    'require exact native44 schema/history under idle owned native writer',
    'retain identical nine marker columns, original identities, composite primary key, unique constraints and immutable guards in ordinary ROWID layout',
    'prove ordered marker parity and recreate original indexes/guards atomically; alter no other business rows',
)
CHECKSUM=digest_canonical({'version':VERSION,'name':NAME,'statements':list(STATEMENTS)})
FINGERPRINT='sha256:dbeca2d28e2f093d9ac7aaf5a6be68329dcb81c0421a33d88457a15e1c16bc78'
_COLUMNS=('namespace','key','command_id','event_id','ledger_seq','security_scope','trust_scope','aggregate_type','aggregate_id')
_SHADOW='_native_marker_rowid45'


def _rewrite_marker_layout(connection):
    """Internal transaction primitive; the public owned entry validates native44."""
    sql=connection.execute("SELECT sql FROM sqlite_schema WHERE name='native_expired_command_keys'").fetchone()[0]
    if 'WITHOUT ROWID, STRICT'not in sql:raise AuthorityPersistenceError('native44 marker layout differs')
    indexes=tuple(connection.execute("SELECT sql FROM sqlite_schema WHERE type='index' AND tbl_name='native_expired_command_keys' AND sql IS NOT NULL"))
    guards=tuple(connection.execute("SELECT name,sql FROM sqlite_schema WHERE type='trigger' AND sql LIKE '%native_expired_command_keys%' ORDER BY name"))
    ddl=sql.replace('CREATE TABLE native_expired_command_keys', 'CREATE TABLE '+_SHADOW,1).replace(' WITHOUT ROWID, STRICT',' STRICT',1)
    connection.execute(ddl)
    columns=','.join(_COLUMNS)
    connection.execute('INSERT INTO '+_SHADOW+'('+columns+') SELECT '+columns+' FROM native_expired_command_keys ORDER BY namespace,key')
    original=connection.execute('SELECT '+columns+' FROM native_expired_command_keys ORDER BY namespace,key')
    rewritten=connection.execute('SELECT '+columns+' FROM '+_SHADOW+' ORDER BY namespace,key')
    digest=hashlib.sha256();count=0
    try:
        for old,new in zip_longest(original,rewritten):
            if old is None or new is None or tuple(old)!=tuple(new):raise AuthorityPersistenceError('marker row parity differs')
            digest.update(canonical_json_bytes(list(old)));count+=1
    finally:original.close();rewritten.close()
    # External insert guards refer to the old table. Remove/recreate their exact
    # checked SQL inside this transaction so ALTER never sees a dangling name.
    for name,_ in guards:connection.execute('DROP TRIGGER "'+name+'"')
    connection.execute('DROP TABLE native_expired_command_keys')
    connection.execute('ALTER TABLE '+_SHADOW+' RENAME TO native_expired_command_keys')
    for (index,)in indexes:connection.execute(index)
    for _,guard in guards:connection.execute(guard)
    return {'rows':count,'ordered_marker_digest':'sha256:'+digest.hexdigest()}


def _apply_layout(connection,applied_at):
    result=_rewrite_marker_layout(connection)
    connection.execute('INSERT INTO authority_migrations VALUES(?,?,?,?)',(VERSION,NAME,CHECKSUM,applied_at))
    connection.execute('PRAGMA user_version=45')
    from .native_current_checkpoint_migrations import require_checkpoint_schema
    require_checkpoint_schema(connection)
    return result


def initialise_empty_marker_store(connection):
    """Explicit NEW45 creator; native44's existing default remains unchanged."""
    from .native_current_checkpoint_migrations import initialise_empty_checkpoint_store
    initialise_empty_checkpoint_store(connection)
    try:
        connection.execute('BEGIN EXCLUSIVE')
        _apply_layout(connection,'1970-01-01T00:00:00.000000Z')
        connection.commit()
    except BaseException:
        if connection.in_transaction:connection.rollback()
        raise


def migrate_native_marker_layout(store,*,dev_rebuild=False,before_commit=lambda:None):
    """Preserve all marker semantics; freed pages need separate physical reclaim."""
    from .native_current_checkpoint_migrations import require_checkpoint_schema
    if not dev_rebuild or not store._current_state_only or not store._native_checkpoint_schema or store._lock_fd is None:
        raise AuthorityPersistenceError('marker layout migration requires explicit owned native DEV writer')
    with store._lock:
        connection=store._connection
        if connection.in_transaction:raise AuthorityPersistenceError('marker layout migration requires idle owned writer')
        require_checkpoint_schema(connection)
        if connection.execute('PRAGMA user_version').fetchone()[0]==VERSION:return {'migrated':False}
        try:
            with store._transaction() as conn:
                if conn.execute('PRAGMA foreign_keys').fetchone()[0]!=1:raise AuthorityPersistenceError('native foreign keys are required')
                result=_apply_layout(conn,store._clock().to_text())
                before_commit()
        except BaseException:
            if connection.in_transaction:connection.rollback()
            raise
    return {'migrated':True,**result,'physical_file_reclaim':False}
