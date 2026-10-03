"""DEV-only verified CURRENT checkpoints; no grants or provider effects."""
from __future__ import annotations
from dataclasses import asdict
import json

from .canonical import canonical_json_bytes, digest_bytes, digest_canonical
from .command_bound_storage import validated_command_result_bytes
from .models import CommittedCommandIdentity
from .objects import ObjectAdmissionId, ObjectAdmissionRequest
from .persistence import AuthorityPersistenceError, DiagnosticHistoryExpired
from .projection_retirement_migrations import RETIRED_FIELDS, retired_record_digest
from ._source_registry_store_common import _RECORD_SPECS
from ._discovery_store import _DISCOVERY_RECORD_SPECS
BUSINESS_SPECS = {**_RECORD_SPECS, **_DISCOVERY_RECORD_SPECS}

VERSION = 'hermes-native-source-checkpoint-v1'
MAX_BYTES = 1_048_576


def _metadata_bytes(store, admission):
    store._current_admission_row(store._connection,str(admission.admission_id),now=store._clock())
    if admission.object_class != 'source.native-observation' or admission.allowed_use != 'native-source-parsing':
        raise AuthorityPersistenceError('checkpoint admission is not native metadata')
    if not 0 < admission.blob.size_bytes <= MAX_BYTES:
        raise AuthorityPersistenceError('checkpoint metadata exceeds bound')
    pinned=store._cas.pin(admission.blob)
    try:
        store._cas.verify_pinned(pinned)
        return store._cas.read_range(pinned,offset=0,length=admission.blob.size_bytes)
    finally:pinned.close()


def _capture(store,event_id):
    store._validate_retained_event(event_id)
    conn=store._connection
    event=conn.execute('SELECT * FROM ledger_events WHERE event_id=?',(event_id,)).fetchone()
    command=conn.execute('SELECT * FROM authority_commands WHERE command_id=?',(event['command_id'],)).fetchone()
    if command['command_type'] not in BUSINESS_SPECS or event['payload_mode'] != 'INLINE' or event['retired_header_digest'] is not None:
        raise AuthorityPersistenceError('checkpoint accepts original canonical source events only')
    auth=conn.execute('SELECT authority_domain FROM authentication_contexts WHERE authentication_context_id=?',
                      (command['authentication_context_id'],)).fetchone()
    return {'header':asdict(store._event_from_row(event)), 'command_type':command['command_type'],
        'namespace':command['idempotency_namespace'],'key':command['idempotency_key'],
        'semantic_digest':command['stable_semantic_request_digest'],'authority_domain':auth[0],
        'result':json.loads(validated_command_result_bytes(conn,command))}


def create_native_source_checkpoint(store,objects,*,event_ids,proof,dev_rebuild=False):
    """Capture trusted original rows, bind governed bytes, then atomically expire RPC.

    The caller supplies identities only, never a capsule or an asserted digest.
    The existing writer lock owns both admission and the final SQLite transaction.
    """
    if not dev_rebuild or not store._current_state_only or not store._native_checkpoint_schema or store._lock_fd is None:
        raise AuthorityPersistenceError('native checkpoint requires explicit DEV writer ownership')
    ids=tuple(sorted(set(event_ids)))
    if not ids:raise ValueError('checkpoint requires selected source events')
    with store._lock:
        with store._transaction():
            members={event_id:_capture(store,event_id) for event_id in ids}
        value={'version':VERSION,'members':members}
        raw=canonical_json_bytes(value)
        if len(raw)>MAX_BYTES:raise AuthorityPersistenceError('checkpoint metadata exceeds bound')
        blob_digest=digest_bytes(raw)
        admission=objects.admit(ObjectAdmissionRequest('source.native-observation',
            'native-dev-source-checkpoint:'+blob_digest),raw,proof=proof).admission
        if admission.blob.blob_digest != blob_digest or _metadata_bytes(store,admission) != raw:
            raise AuthorityPersistenceError('checkpoint admission differs from verified source closure')
        checkpoint_id=digest_canonical({'version':VERSION,'blob_digest':blob_digest,'admission_id':str(admission.admission_id)})
        conn=store._connection
        try:
            with store._transaction():
                # No stale creation-to-expiry window: exact original proof again.
                if {event_id:_capture(store,event_id) for event_id in ids} != members:
                    raise AuthorityPersistenceError('checkpoint original source changed')
                conn.execute('INSERT INTO native_current_checkpoints VALUES(?,?,?,?,?)',
                    (checkpoint_id,str(admission.admission_id),blob_digest,digest_canonical(value),store._clock().to_text()))
                for event_id,member in members.items():
                    encoded=canonical_json_bytes(member)
                    conn.execute('INSERT INTO native_current_checkpoint_members VALUES(?,?,?,?,?)',
                        (event_id,member['header']['command_id'],checkpoint_id,encoded,digest_bytes(encoded)))
                _expire_source_rpc(store,checkpoint_id,members)
                if conn.execute('PRAGMA foreign_key_check').fetchone() is not None:
                    raise AuthorityPersistenceError('checkpoint retained foreign-key closure differs')
        except BaseException:
            if conn.in_transaction:conn.rollback()
            raise
        return checkpoint_id


def _expire_source_rpc(store,checkpoint_id,members):
    conn=store._connection
    guards=tuple(conn.execute("SELECT name,sql FROM sqlite_schema WHERE type='trigger' AND "
        "(name IN ('immutable_ledger_events_update','immutable_authority_commands_update') OR "
        "tbl_name IN ('authority_audit_events','authorization_requests','authorization_decisions','authentication_contexts') AND sql LIKE '%BEFORE DELETE%')"))
    from ._foreign_keys import foreign_key_children
    keys={'authorization_decisions':'authorization_decision_id','authorization_requests':'request_digest','authentication_contexts':'authentication_context_id'}
    children={table:foreign_key_children(conn,table,key=key) for table,key in keys.items()}
    columns=tuple(row[1] for row in conn.execute('PRAGMA table_info(ledger_events)'))
    removed=tuple(name for name in columns if name not in RETIRED_FIELDS and name not in {'retired_header_digest','retired_record_digest','native_checkpoint_id'})
    for name,_ in guards:conn.execute('DROP TRIGGER "'+name+'"')
    try:
        for event_id,member in members.items():
            event=conn.execute('SELECT * FROM ledger_events WHERE event_id=?',(event_id,)).fetchone()
            parent_ids=(event['authorization_decision_id'],event['authorization_request_digest'],event['authentication_context_id'])
            thin={name:event[name] for name in RETIRED_FIELDS if name not in {'retired_namespace','retired_key'}}
            thin.update(retired_namespace=member['namespace'],retired_key=member['key'],
                retired_header_digest=bytes.fromhex(digest_canonical(member['header'])[7:]))
            conn.execute('UPDATE ledger_events SET '+','.join(name+'=NULL' for name in removed)+
                ',retired_header_digest=?,retired_namespace=?,retired_key=?,retired_record_digest=?,native_checkpoint_id=? WHERE event_id=?',
                (thin['retired_header_digest'],thin['retired_namespace'],thin['retired_key'],retired_record_digest(thin),checkpoint_id,event_id))
            command_id=member['header']['command_id']
            conn.execute('UPDATE authority_commands SET authentication_context_id=NULL,authorization_request_digest=NULL,'
                'authorization_decision_id=NULL,result_bytes=?,native_checkpoint_id=? WHERE command_id=?',
                (canonical_json_bytes(member['result']),checkpoint_id,command_id))
            conn.execute('DELETE FROM authority_audit_events WHERE command_id=?',(command_id,))
            # Only this event's obsolete security parents. Shared rights/pending
            # references remain protected by exact child-FK equality probes.
            for table,key,identity in zip(('authorization_decisions','authorization_requests','authentication_contexts'),
                ('authorization_decision_id','request_digest','authentication_context_id'),parent_ids):
                probes=children[table]
                if not any(conn.execute('SELECT 1 FROM "'+child+'" WHERE "'+cols[0]+'"=? LIMIT 1',(identity,)).fetchone()
                           for child,cols in probes):
                    conn.execute('DELETE FROM '+table+' WHERE '+key+'=?',(identity,))
    finally:
        for _,sql in guards:conn.execute(sql)


def checkpoint_member(store,event_id):
    if not store._current_state_only:raise DiagnosticHistoryExpired('source RPC provenance expired')
    conn=store._connection
    row=conn.execute('SELECT m.*,r.admission_id,r.blob_digest,r.manifest_digest FROM native_current_checkpoint_members m '
        'JOIN native_current_checkpoints r ON r.checkpoint_id=m.checkpoint_id WHERE m.event_id=?',(event_id,)).fetchone()
    event=conn.execute('SELECT * FROM ledger_events WHERE event_id=?',(event_id,)).fetchone()
    if row is None or event is None or event['native_checkpoint_id'] != row['checkpoint_id']:
        raise AuthorityPersistenceError('selected checkpoint root is absent')
    admission=store.admission_view(ObjectAdmissionId.parse(row['admission_id']))
    raw=_metadata_bytes(store,admission)
    encoded=bytes(row['canonical_bytes'])
    try:
        manifest=json.loads(raw)
        member=json.loads(encoded)
    except (ValueError,UnicodeDecodeError) as exc:
        raise AuthorityPersistenceError('selected checkpoint original proof differs') from exc
    if (type(manifest) is not dict or type(manifest.get('members')) is not dict
        or type(member) is not dict or type(member.get('header')) is not dict
        or not {'namespace','key'} <= member.keys() or event['retired_header_digest'] is None):
        raise AuthorityPersistenceError('selected checkpoint original proof differs')
    if (admission.blob.blob_digest != row['blob_digest'] or canonical_json_bytes(manifest)!=raw
        or manifest.get('version')!=VERSION or digest_canonical(manifest)!=row['manifest_digest']
        or digest_bytes(encoded)!=row['canonical_digest'] or canonical_json_bytes(member)!=encoded
        or manifest.get('members',{}).get(event_id)!=member
        or digest_canonical(member['header'])!='sha256:'+bytes(event['retired_header_digest']).hex()
        or retired_record_digest(event)!=bytes(event['retired_record_digest'])
        or member['namespace']!=event['retired_namespace'] or member['key']!=event['retired_key']):
        raise AuthorityPersistenceError('selected checkpoint original proof differs')
    for field in ('event_id','command_id','aggregate_type','aggregate_id','event_type','ledger_seq','security_scope','trust_scope'):
        if member['header'][field]!=event[field]:raise AuthorityPersistenceError('checkpoint routing differs')
    return member


def checkpoint_source_envelope(store,event_id,canonical_bytes):
    member=checkpoint_member(store,event_id)
    header=member['header']
    if digest_bytes(canonical_bytes)!=header['payload_digest']:
        raise AuthorityPersistenceError('checkpoint source canonical body differs')
    return {**header,'idempotency_key':member['key'],'payload_bytes':canonical_bytes}


def checkpoint_identity(store,namespace,key):
    row=store._connection.execute('SELECT event_id FROM ledger_events WHERE retired_namespace=? AND retired_key=? '
        'AND native_checkpoint_id IS NOT NULL',(namespace,key)).fetchone()
    if row is None:return None
    member=checkpoint_member(store,row['event_id']);header=member['header']
    return CommittedCommandIdentity(command_id=header['command_id'],authority_domain=member['authority_domain'],
        principal_id=header['principal_id'],command_type=member['command_type'],idempotency_namespace=namespace,
        idempotency_key=key,command_definition_version=header['command_definition_version'],
        command_definition_digest=header['command_definition_digest'],stable_semantic_request_digest=member['semantic_digest'],
        payload_mode=header['payload_mode'],payload_digest=header['payload_digest'],object_admission_id=None)


def discovery_current_root_ids(conn):
    """Existing recovery closure's exact current/promoting/Watch references only."""
    gate_sql="""SELECT current_decision_id FROM discovery_gate_decision_heads
        UNION SELECT promoting_gate_decision_id FROM news_leads
        UNION SELECT d.gate_decision_id FROM lead_disposition_heads h JOIN lead_disposition_decisions d ON d.decision_id=h.current_decision_id
        UNION SELECT gate_decision_id FROM discovery_watch_conditions"""
    gates=tuple(row[0] for row in conn.execute(gate_sql))
    dispositions=tuple(row[0] for row in conn.execute('SELECT current_decision_id FROM lead_disposition_heads'))
    return {'discovery_gate_decisions':gates,'lead_disposition_decisions':dispositions}


def prune_superseded_discovery(store,objects,*,proof,dev_rebuild=False):
    """Explicit DEV target: current recovery closure, not all decision history.

    Current canonical predecessor IDs remain opaque in the new-only schema.
    Original keys of removed decisions retain existing compact reservations.
    """
    if not dev_rebuild or not store._native_checkpoint_schema:
        raise AuthorityPersistenceError('discovery expiry requires explicit NEW native DEV store')
    conn=store._connection
    selected=discovery_current_root_ids(conn)
    event_ids=tuple(sorted({row[0] for table,ids in selected.items() for identity in ids
        for row in conn.execute('SELECT authority_event_id FROM '+table+' WHERE decision_id=?',(identity,))
        if conn.execute('SELECT retired_header_digest FROM ledger_events WHERE event_id=?',(row[0],)).fetchone()[0] is None}))
    # Each governed manifest stays within the existing 1 MiB metadata class.
    checkpoints=tuple(create_native_source_checkpoint(store,objects,event_ids=event_ids[offset:offset+64],
        proof=proof,dev_rebuild=True) for offset in range(0,len(event_ids),64))
    deleted={}
    try:
      with store._lock,store._transaction():
        from ._foreign_keys import foreign_key_children
        parent_keys={'discovery_gate_decisions':'decision_id','lead_disposition_decisions':'decision_id',
            'authority_payloads':'payload_id','authorization_decisions':'authorization_decision_id',
            'authorization_requests':'request_digest','authentication_contexts':'authentication_context_id'}
        children={parent:foreign_key_children(conn,parent,key=key) for parent,key in parent_keys.items()}
        event_children=foreign_key_children(conn,'ledger_events',key='event_id')
        columns=tuple(r[1] for r in conn.execute('PRAGMA table_info(ledger_events)'))
        removed=tuple(name for name in columns if name not in RETIRED_FIELDS and name not in {'retired_header_digest','retired_record_digest','native_checkpoint_id'})
        guards=tuple(conn.execute("SELECT name,sql FROM sqlite_schema WHERE type='trigger' AND tbl_name IN "
            "('discovery_gate_decisions','lead_disposition_decisions','authority_audit_events','authority_aggregate_versions','authority_aggregates','authority_commands','authorization_requests','authorization_decisions','authentication_contexts','authority_payloads','ledger_events') "
            "AND (sql LIKE '%BEFORE DELETE%' OR name='immutable_ledger_events_update')"))
        for name,_ in guards:conn.execute('DROP TRIGGER "'+name+'"')
        candidates=None
        try:
            for table in ('lead_disposition_decisions','discovery_gate_decisions'):
                ids=selected[table]
                conn.execute('CREATE TEMP TABLE _checkpoint_keep(decision_id TEXT PRIMARY KEY)')
                conn.executemany('INSERT INTO _checkpoint_keep VALUES(?)',((identity,) for identity in ids))
                conn.execute('CREATE TEMP TABLE _checkpoint_candidates(decision_id TEXT PRIMARY KEY,authority_event_id TEXT NOT NULL)')
                conn.execute('INSERT INTO _checkpoint_candidates SELECT decision_id,authority_event_id FROM '+table+
                    ' WHERE decision_id NOT IN (SELECT decision_id FROM _checkpoint_keep)')
                candidates=conn.execute('SELECT decision_id,authority_event_id FROM _checkpoint_candidates ORDER BY decision_id')
                count=0
                for identity,event_id in candidates:
                    # Canonical active/pending references remain exact. No opaque
                    # historical token scan or whole-ledger proof is performed.
                    if any(conn.execute('SELECT 1 FROM "'+child+'" WHERE "'+columns[0]+'"=? LIMIT 1',(identity,)).fetchone()
                           for child,columns in children[table]):
                        continue
                    event=conn.execute('SELECT * FROM ledger_events WHERE event_id=?',(event_id,)).fetchone()
                    if event['retired_header_digest'] is not None:continue
                    if any(conn.execute('SELECT 1 FROM "'+child+'" WHERE "'+cols[0]+'"=? LIMIT 1',(event_id,)).fetchone()
                           for child,cols in event_children if child not in {table,'authority_audit_events','authority_aggregate_versions'}):
                        continue
                    if any(conn.execute('SELECT 1 FROM ledger_events WHERE causation_kind=? AND causation_identifier=? '
                        'AND retired_header_digest IS NULL LIMIT 1',(kind,target)).fetchone()
                        for kind,target in (('EVENT',event_id),('COMMAND',event['command_id']))):
                        continue
                    command=conn.execute('SELECT * FROM authority_commands WHERE command_id=?',(event['command_id'],)).fetchone()
                    header=store._event_from_row(event)
                    thin={name:event[name] for name in RETIRED_FIELDS if name not in {'retired_namespace','retired_key'}}
                    thin.update(retired_namespace=command['idempotency_namespace'],retired_key=command['idempotency_key'],
                        retired_header_digest=bytes.fromhex(digest_canonical(asdict(header))[7:]))
                    conn.execute('DELETE FROM '+table+' WHERE decision_id=?',(identity,))
                    conn.execute('UPDATE ledger_events SET '+','.join(name+'=NULL' for name in removed)+
                        ',retired_header_digest=?,retired_namespace=?,retired_key=?,retired_record_digest=? WHERE event_id=?',
                        (thin['retired_header_digest'],thin['retired_namespace'],thin['retired_key'],retired_record_digest(thin),event_id))
                    conn.execute('DELETE FROM authority_audit_events WHERE command_id=?',(command['command_id'],))
                    conn.execute('DELETE FROM authority_aggregate_versions WHERE command_id=?',(command['command_id'],))
                    conn.execute('DELETE FROM authority_aggregates WHERE aggregate_type=? AND aggregate_id=?',
                        (event['aggregate_type'],event['aggregate_id']))
                    conn.execute('DELETE FROM authority_commands WHERE command_id=?',(command['command_id'],))
                    for parent,key,identity in (('authority_payloads','payload_id',command['payload_id']),
                        ('authorization_decisions','authorization_decision_id',command['authorization_decision_id']),
                        ('authorization_requests','request_digest',command['authorization_request_digest']),
                        ('authentication_contexts','authentication_context_id',command['authentication_context_id'])):
                        if identity is not None and not any(conn.execute('SELECT 1 FROM "'+child+'" WHERE "'+cols[0]+'"=? LIMIT 1',(identity,)).fetchone()
                            for child,cols in children[parent]):
                            conn.execute('DELETE FROM '+parent+' WHERE '+key+'=?',(identity,))
                    count+=1
                deleted[table]=count
                conn.execute('DROP TABLE _checkpoint_candidates')
                conn.execute('DROP TABLE _checkpoint_keep')
            if conn.execute('PRAGMA foreign_key_check').fetchone() is not None:
                raise AuthorityPersistenceError('discovery current reference closure differs')
        finally:
            if candidates is not None:candidates.close()
            conn.execute('DROP TABLE IF EXISTS temp._checkpoint_candidates')
            conn.execute('DROP TABLE IF EXISTS temp._checkpoint_keep')
            for _,sql in guards:conn.execute(sql)
    except BaseException:
        if conn.in_transaction:conn.rollback()
        raise
    return {'checkpoint_ids':checkpoints,'deleted':deleted}
