"""Owned marker46 preserves logical reservations, Source replay and indexed denial."""
import sqlite3
import uuid
from dataclasses import replace

import pytest

from newsroom.authority import DiagnosticHistoryExpired, EventReadPolicy, MetadataClass, TrustScope
from newsroom.authority.native_current_checkpoint_migrations import require_checkpoint_schema
from newsroom.authority.native_marker_layout_migrations import migrate_native_marker_layout
from newsroom.authority._graphiti_increment4_system import _AUTHORITY_COMPOSITION_TOKEN
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.tests.test_native_current_checkpoint import source_fixture


def _marker(i, *, opaque=False):
    return ('namespace', 'opaque key:£'+str(i), 'legacy-command-'+str(i) if opaque else str(uuid.uuid4()),
        str(uuid.uuid4()), 5000+i, 'authority.fixture', 'ADMITTED', 'fixture_aggregate',
        'legacy:aggregate' if opaque else str(uuid.uuid4()))


def test_native45_to46_preserves_real_source_all_logical_keys_and_reopens(tmp_path,monkeypatch):
    from newsroom.authority.native_marker_codec_migrations import migrate_native_marker_codec, marker_rows
    rows=(_marker(1),_marker(2,opaque=True))
    with source_fixture(tmp_path,monkeypatch)as(args,runtime,_,_,before,root):
        c=root._connection
        c.executemany('INSERT INTO native_expired_command_keys VALUES(?,?,?,?,?,?,?,?,?)',rows)
        migrate_native_marker_layout(root,dev_rebuild=True)
        before_source=tuple(c.execute('SELECT canonical_bytes FROM source_revisions ORDER BY revision_id'))
        result=migrate_native_marker_codec(root,dev_rebuild=True)
        assert result['migrated']and result['rows']==2
        assert tuple(marker_rows(c))==rows
        assert c.execute('PRAGMA user_version').fetchone()[0]==46
        require_checkpoint_schema(c)
        assert c.execute('PRAGMA foreign_key_check').fetchall()==[]
        assert tuple(c.execute('SELECT canonical_bytes FROM source_revisions ORDER BY revision_id'))==before_source
        assert runtime.authority.sources.revision(before.request.revision_id,proof=runtime.proof)==before
        assert runtime.authority.sources.record_revision(before.request,proof=runtime.proof).replayed
        for row in rows:
            with pytest.raises(DiagnosticHistoryExpired):root._require_unexpired_key(c,*row[:2])
        assert migrate_native_marker_codec(root,dev_rebuild=True)['migrated']is False
    with open_native_runtime(**args)as runtime:
        root=runtime.authority._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0]
        require_checkpoint_schema(root._connection)
        assert tuple(marker_rows(root._connection))==rows
        assert runtime.authority.sources.revision(before.request.revision_id,proof=runtime.proof)==before


@pytest.mark.parametrize('failure',(RuntimeError('before commit'),KeyboardInterrupt()))
def test_codec_failure_restores_exact45_history_guards_and_rows(tmp_path,monkeypatch,failure):
    from newsroom.authority.native_marker_codec_migrations import migrate_native_marker_codec
    with source_fixture(tmp_path,monkeypatch)as(_,_,_,_,_,root):
        c=root._connection;c.execute('INSERT INTO native_expired_command_keys VALUES(?,?,?,?,?,?,?,?,?)',_marker(1))
        migrate_native_marker_layout(root,dev_rebuild=True)
        before=tuple(c.iterdump())
        def interrupt():raise failure
        with pytest.raises(type(failure)):migrate_native_marker_codec(root,dev_rebuild=True,before_commit=interrupt)
        assert tuple(c.iterdump())==before and not c.in_transaction
        assert root._native_marker_codec is False
        require_checkpoint_schema(c)


def test_codec_preserves_all_no_revival_guards_and_scope_independent_key(tmp_path,monkeypatch):
    from newsroom.authority.native_marker_codec_migrations import migrate_native_marker_codec,insert_markers,marker_rows,encode_identity
    row=_marker(1)
    with source_fixture(tmp_path,monkeypatch)as(_,_,_,_,_,root):
        c=root._connection;c.execute('INSERT INTO native_expired_command_keys VALUES(?,?,?,?,?,?,?,?,?)',row)
        migrate_native_marker_layout(root,dev_rebuild=True);migrate_native_marker_codec(root,dev_rebuild=True)
        denied=(
            ('UPDATE native_expired_command_keys SET key=?',('other',)),
            ('DELETE FROM native_expired_command_keys',()),
            ('UPDATE native_marker_namespaces SET namespace=?',('other',)),
            ('DELETE FROM native_marker_scopes',()),
            ('INSERT INTO authority_commands(command_id,idempotency_namespace,idempotency_key) VALUES(?,?,?)',(row[2],'other','other')),
            ('INSERT INTO authority_commands(command_id,idempotency_namespace,idempotency_key) VALUES(?,?,?)',('other',*row[:2])),
            ('INSERT INTO ledger_events(event_id,ledger_seq) VALUES(?,?)',(row[3],5002)),
            ('INSERT INTO ledger_events(event_id,ledger_seq) VALUES(?,?)',('other',row[4])),
            ('INSERT INTO authority_aggregates(aggregate_type,aggregate_id) VALUES(?,?)',row[7:]),
        )
        for sql,args in denied:
            with pytest.raises(sqlite3.IntegrityError,match='immutable|identity remains reserved'):c.execute(sql,args)
        fresh=_marker(2)
        changed=(row[0],row[1],*fresh[2:5],'other.security','OBSERVED','other_aggregate',fresh[8])
        c.execute('SAVEPOINT duplicate')
        with pytest.raises(sqlite3.IntegrityError,match='UNIQUE'):insert_markers(c,(changed,))
        c.execute('ROLLBACK TO duplicate');c.execute('RELEASE duplicate')
        long=(row[0],'PAYLOAD:'+('£'*300),*fresh[2:])
        insert_markers(c,(long,))
        assert long in tuple(marker_rows(c))
        for column,bad in [('namespace_id',9999),('scope_id',9999),('command_id',b'x'*15),('command_id',fresh[2])]:
            values=[1,'unique-'+column,encode_identity(str(uuid.uuid4())),encode_identity(str(uuid.uuid4())),9999,1,encode_identity(str(uuid.uuid4()))]
            values[['namespace_id','key','command_id','event_id','ledger_seq','scope_id','aggregate_id'].index(column)]=bad
            with pytest.raises(sqlite3.IntegrityError):c.execute('INSERT INTO native_expired_command_keys VALUES(?,?,?,?,?,?,?)',values)


def test_codec_expired_provenance_scope_bounds_and_aliases_are_unchanged(tmp_path,monkeypatch):
    from newsroom.authority.native_marker_codec_migrations import migrate_native_marker_codec
    row=_marker(1)
    with source_fixture(tmp_path,monkeypatch)as(_,_,_,_,_,root):
        c=root._connection;c.execute('INSERT INTO native_expired_command_keys VALUES(?,?,?,?,?,?,?,?,?)',row)
        migrate_native_marker_layout(root,dev_rebuild=True);migrate_native_marker_codec(root,dev_rebuild=True)
        policy=EventReadPolicy(policy_id='fixture-codec',purpose='fixture',required_scope='fixture.read',
            allowed_principal_ids=frozenset({'fixture'}),allowed_security_scopes=frozenset({row[5]}),
            allowed_trust_scopes=frozenset({TrustScope.ADMITTED}),metadata_classes=frozenset({MetadataClass.PROVENANCE}),
            minimum_ledger_seq=row[4],maximum_ledger_seq=row[4],max_results=1)
        with pytest.raises(DiagnosticHistoryExpired):root.event_provenance(event_id=row[3],policy=policy)
        for changed in (replace(policy,allowed_security_scopes=frozenset({'other'})),
                replace(policy,allowed_trust_scopes=frozenset({TrustScope.OBSERVED})),
                replace(policy,minimum_ledger_seq=row[4]+1,maximum_ledger_seq=row[4]+1)):
            with pytest.raises(KeyError):root.event_provenance(event_id=row[3],policy=changed)
        with pytest.raises(KeyError):root.event_provenance(event_id=row[3].upper(),policy=policy)


def test_codec_real_denial_and_sequence_sql_remain_indexed(tmp_path,monkeypatch):
    from newsroom.authority.native_marker_codec_migrations import migrate_native_marker_codec,encode_identity
    from newsroom.authority import ObjectAdmissionRequest
    row=_marker(1)
    with source_fixture(tmp_path,monkeypatch)as(_,runtime,_,_,_,root):
        c=root._connection;c.execute('INSERT INTO native_expired_command_keys VALUES(?,?,?,?,?,?,?,?,?)',row)
        migrate_native_marker_layout(root,dev_rebuild=True);migrate_native_marker_codec(root,dev_rebuild=True)
        queries=(
            ('SELECT 1 FROM native_marker_namespaces n CROSS JOIN native_expired_command_keys m ON m.namespace_id=n.namespace_id WHERE n.namespace=? AND m.key=?',row[:2]),
            ('SELECT 1 FROM native_expired_command_keys WHERE command_id=?',(encode_identity(row[2]),)),
            ('SELECT 1 FROM native_expired_command_keys m JOIN native_marker_scopes s ON s.scope_id=m.scope_id WHERE m.event_id=? AND s.security_scope=? AND s.trust_scope=? AND m.ledger_seq BETWEEN ? AND ?',
                (encode_identity(row[3]),row[5],row[6],row[4],row[4])),
            ('SELECT MAX(ledger_seq) FROM native_expired_command_keys',()),
            ('SELECT 1 FROM native_marker_scopes s CROSS JOIN native_expired_command_keys m ON m.scope_id=s.scope_id WHERE s.aggregate_type=? AND m.aggregate_id=?',(row[7],encode_identity(row[8]))),
        )
        for sql,args in queries:
            plan=' '.join(str(part)for r in c.execute('EXPLAIN QUERY PLAN '+sql,args)for part in r)
            assert 'SEARCH'in plan and 'SCAN m'not in plan and 'SCAN native_expired_command_keys'not in plan,plan
        runtime.authority.objects.admit(ObjectAdmissionRequest('evidence.source','after-codec'),b'fresh authorised evidence',proof=runtime.proof)
        assert c.execute('SELECT max(ledger_seq) FROM ledger_events').fetchone()[0]>row[4]


@pytest.mark.parametrize('identifier',('legacy:opaque',str(uuid.uuid5(uuid.NAMESPACE_URL,'fixture')),'12345678-1234-4ABC-8ABC-123456789ABC'))
def test_codec_never_relabels_legally_stored_noncanonical_text(tmp_path,identifier):
    from newsroom.authority.native_marker_codec_migrations import initialise_empty_codec_store,insert_markers,marker_rows,encode_identity
    c=sqlite3.connect(tmp_path/'opaque.sqlite3',isolation_level=None)
    initialise_empty_codec_store(c)
    row=('namespace','long opaque key:'+('x'*1024),identifier,str(uuid.uuid4()),1,'authority.fixture','ADMITTED','fixture',identifier)
    insert_markers(c,(row,))
    assert tuple(marker_rows(c))==(row,)and encode_identity(identifier)==identifier
    assert c.execute('SELECT typeof(command_id),typeof(aggregate_id) FROM native_expired_command_keys').fetchone()==('text','text')
    c.close()


def test_codec_owned_idle_and_explicit45_preconditions(tmp_path,monkeypatch):
    from newsroom.authority.native_marker_codec_migrations import migrate_native_marker_codec
    from newsroom.authority import AuthorityPersistenceError
    with source_fixture(tmp_path,monkeypatch)as(_,_,_,_,_,root):
        c=root._connection
        with pytest.raises(AuthorityPersistenceError,match='DEV writer'):migrate_native_marker_codec(root)
        with pytest.raises(AuthorityPersistenceError,match='native45'):migrate_native_marker_codec(root,dev_rebuild=True)
        migrate_native_marker_layout(root,dev_rebuild=True)
        c.execute('BEGIN IMMEDIATE');c.execute('CREATE TABLE caller_owned(value TEXT)')
        with pytest.raises(AuthorityPersistenceError,match='idle'):migrate_native_marker_codec(root,dev_rebuild=True)
        assert c.in_transaction and c.execute("SELECT 1 FROM sqlite_schema WHERE name='caller_owned'").fetchone()
        c.rollback();require_checkpoint_schema(c)


@pytest.mark.parametrize('source_codec',(False,True))
@pytest.mark.parametrize('destination_codec',(False,True))
def test_selected_rebuild_keeps_original_reservations_across_both_layouts(tmp_path,monkeypatch,source_codec,destination_codec):
    from newsroom.authority.native_marker_codec_migrations import migrate_native_marker_codec,initialise_empty_codec_store,marker_rows
    from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store
    from newsroom.authority.native_current_rebuild import copy_selected_native_store
    row=_marker(1,opaque=True)
    with source_fixture(tmp_path,monkeypatch)as(_,_,_,_,before,root):
        c=root._connection;c.execute('INSERT INTO native_expired_command_keys VALUES(?,?,?,?,?,?,?,?,?)',row)
        migrate_native_marker_layout(root,dev_rebuild=True)
        if source_codec:migrate_native_marker_codec(root,dev_rebuild=True)
        with sqlite3.connect(tmp_path/'selected.sqlite3',isolation_level=None)as destination:
            (initialise_empty_codec_store if destination_codec else initialise_empty_checkpoint_store)(destination)
            copy_selected_native_store(root,destination,
                roots={'source_revisions':((str(before.request.revision_id),),)},dev_rebuild=True)
            assert row in tuple(marker_rows(destination))
            assert destination.execute('SELECT canonical_bytes FROM source_revisions WHERE revision_id=?',
                (str(before.request.revision_id),)).fetchone()[0]==c.execute('SELECT canonical_bytes FROM source_revisions WHERE revision_id=?',
                (str(before.request.revision_id),)).fetchone()[0]
            assert destination.execute('PRAGMA foreign_key_check').fetchall()==[]
            require_checkpoint_schema(destination)


def test_codec_keeps_every_business_row_pending_obligation_and_cas_bytes(tmp_path,monkeypatch):
    import hashlib
    from newsroom.authority import ObjectAdmissionRequest
    from newsroom.authority.native_marker_codec_migrations import migrate_native_marker_codec
    with source_fixture(tmp_path,monkeypatch)as(args,runtime,_,_,before,root):
        original=root.commit_admission
        def interrupted(*_a,**_k):raise KeyboardInterrupt('fixture before activation')
        monkeypatch.setattr(root,'commit_admission',interrupted)
        with pytest.raises(KeyboardInterrupt):runtime.authority.objects.admit(
            ObjectAdmissionRequest('evidence.source','pending-before46'),b'original pending authorised bytes',proof=runtime.proof)
        monkeypatch.setattr(root,'commit_admission',original)
        c=root._connection
        staged=c.execute("SELECT staged_name FROM object_staging_records WHERE state='STAGED'").fetchone()[0]
        path=root._cas.staging_root/staged
        cas_digest=hashlib.sha256(path.read_bytes()).hexdigest()
        migrate_native_marker_layout(root,dev_rebuild=True)
        def business_rows():
            return tuple(line for line in c.iterdump()if line.startswith('INSERT INTO ')and not any(
                line.startswith('INSERT INTO "'+table+'"')for table in ('authority_migrations','native_expired_command_keys','native_marker_namespaces','native_marker_scopes')))
        protected=business_rows()
        migrate_native_marker_codec(root,dev_rebuild=True)
        assert business_rows()==protected
        assert hashlib.sha256(path.read_bytes()).hexdigest()==cas_digest
    with open_native_runtime(**args)as runtime:
        assert runtime.authority.sources.revision(before.request.revision_id,proof=runtime.proof)==before
        assert hashlib.sha256(path.read_bytes()).hexdigest()==cas_digest


def test_codec_capability_failure_has_no_schema_or_business_effect(tmp_path,monkeypatch):
    from newsroom.authority import AuthorityPersistenceError
    from newsroom.authority import native_marker_codec_migrations as codec
    with source_fixture(tmp_path,monkeypatch)as(_,_,_,_,_,root):
        migrate_native_marker_layout(root,dev_rebuild=True)
        c=root._connection;before=tuple(c.iterdump())
        def unavailable(_):raise AuthorityPersistenceError('SQLite unhex unavailable')
        monkeypatch.setattr(codec,'require_codec_capability',unavailable)
        with pytest.raises(AuthorityPersistenceError,match='unhex'):codec.migrate_native_marker_codec(root,dev_rebuild=True)
        assert tuple(c.iterdump())==before and not c.in_transaction
        require_checkpoint_schema(c)  # Legacy45 does not demand46 capability.


def test_new46_literal_schema_contract_and_reopen_corruption_denial(tmp_path,monkeypatch):
    from newsroom.authority.native_marker_codec_migrations import migrate_native_marker_codec
    from newsroom.authority.migrations import schema_fingerprint
    with source_fixture(tmp_path,monkeypatch)as(args,_,_,_,_,root):
        migrate_native_marker_layout(root,dev_rebuild=True);migrate_native_marker_codec(root,dev_rebuild=True)
        c=root._connection
        assert schema_fingerprint(c)=='sha256:66ef7f14560689fbbac6425117f286c3a6c9ddec811e17e0be5577d1d8e748ab'
        assert tuple(c.execute('SELECT version,name,checksum FROM authority_migrations WHERE version=46').fetchone())==(
            46,'native_expired_marker_dictionary_codec_v46','sha256:65b03e61a09d6e67e3a6e89ed565a24d1ebae96f5162e6d9310b5d5cfd7e3f80')
        c.execute('CREATE TABLE forged46(value TEXT)')
    with pytest.raises(sqlite3.DatabaseError,match='schema/history differs'):
        with open_native_runtime(**args):pytest.fail('forged46 opened')
