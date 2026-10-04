"""Native45 marker layout keeps replay/guards and real Source reads unchanged."""
import sqlite3
import uuid

import pytest

from newsroom.authority.native_marker_layout_migrations import migrate_native_marker_layout
from newsroom.authority.native_current_checkpoint_migrations import require_checkpoint_schema
from newsroom.authority import DiagnosticHistoryExpired,ObjectAdmissionRequest
from newsroom.authority._graphiti_increment4_system import _AUTHORITY_COMPOSITION_TOKEN
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.tests.test_native_current_checkpoint import source_fixture

MARKER=('sha256:'+'a'*64,'expired-fixture-key',str(uuid.uuid4()),str(uuid.uuid4()),5000,
        'authority.fixture','ADMITTED','fixture_aggregate',str(uuid.uuid4()))


def test_native44_to45_real_source_replay_new_sequence_and_reopen(tmp_path,monkeypatch):
    with source_fixture(tmp_path,monkeypatch) as (args,runtime,_,_,before,root):
        c=root._connection;c.execute('INSERT INTO native_expired_command_keys VALUES(?,?,?,?,?,?,?,?,?)',MARKER)
        require_checkpoint_schema(c)
        result=migrate_native_marker_layout(root,dev_rebuild=True)
        assert result['migrated'] and result['rows']==1
        assert c.execute('PRAGMA user_version').fetchone()[0]==45
        require_checkpoint_schema(c)
        assert tuple(c.execute('SELECT * FROM native_expired_command_keys').fetchone())==MARKER
        assert 'WITHOUT ROWID'not in c.execute("SELECT sql FROM sqlite_schema WHERE name='native_expired_command_keys'").fetchone()[0]
        assert runtime.authority.sources.record_revision(before.request,proof=runtime.proof).replayed
        assert runtime.authority.sources.revision(before.request.revision_id,proof=runtime.proof)==before
        with pytest.raises(DiagnosticHistoryExpired):root._require_unexpired_key(c,*MARKER[:2])
        assert migrate_native_marker_layout(root,dev_rebuild=True)['migrated']is False
        runtime.authority.objects.admit(ObjectAdmissionRequest('evidence.source','new-qualified-key'),b'complete fresh evidence',proof=runtime.proof)
        assert c.execute('SELECT max(ledger_seq) FROM ledger_events').fetchone()[0]>MARKER[4]
    with open_native_runtime(**args) as reopened:
        root=reopened.authority._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0]
        require_checkpoint_schema(root._connection)
        assert reopened.authority.sources.revision(before.request.revision_id,proof=reopened.proof)==before
        assert migrate_native_marker_layout(root,dev_rebuild=True)['migrated']is False


@pytest.mark.parametrize('failure',(RuntimeError('injected before commit'),KeyboardInterrupt()))
def test_interrupt_restores_exact44_rows_history_and_guards(tmp_path,monkeypatch,failure):
    with source_fixture(tmp_path,monkeypatch) as (_,_,_,_,_,root):
        c=root._connection;c.execute('INSERT INTO native_expired_command_keys VALUES(?,?,?,?,?,?,?,?,?)',MARKER)
        before=tuple(c.iterdump())
        def interrupt():raise failure
        with pytest.raises(type(failure)):
            migrate_native_marker_layout(root,dev_rebuild=True,before_commit=interrupt)
        assert tuple(c.iterdump())==before
        assert not c.in_transaction and c.execute('PRAGMA user_version').fetchone()[0]==44
        require_checkpoint_schema(c)


def test_active_caller_transaction_and_unowned_migration_are_denied(tmp_path,monkeypatch):
    from newsroom.authority import AuthorityPersistenceError
    with source_fixture(tmp_path,monkeypatch) as (_,_,_,_,_,root):
        c=root._connection
        with pytest.raises(AuthorityPersistenceError):migrate_native_marker_layout(root)
        c.execute('BEGIN IMMEDIATE');c.execute('CREATE TABLE caller_owned(value TEXT)')
        with pytest.raises(AuthorityPersistenceError,match='idle'):
            migrate_native_marker_layout(root,dev_rebuild=True)
        assert c.in_transaction and c.execute("SELECT 1 FROM sqlite_schema WHERE name='caller_owned'").fetchone()
        c.rollback();require_checkpoint_schema(c)


@pytest.mark.parametrize('damage',('fingerprint','history','version'))
def test_bad44_contract_is_rejected_without_effect(tmp_path,monkeypatch,damage):
    with source_fixture(tmp_path,monkeypatch) as (_,_,_,_,_,root):
        c=root._connection
        if damage=='fingerprint':c.execute('CREATE TABLE unrelated_schema(value TEXT)')
        elif damage=='history':c.execute("INSERT INTO authority_migrations VALUES(45,'forged','wrong','1970-01-01T00:00:00.000000Z')")
        else:c.execute('PRAGMA user_version=46')
        before=tuple(c.iterdump())
        with pytest.raises(sqlite3.DatabaseError):migrate_native_marker_layout(root,dev_rebuild=True)
        assert tuple(c.iterdump())==before and not c.in_transaction


def test_native45_keeps_thin_source_checkpoint_and_all_reservation_guards(tmp_path,monkeypatch):
    from newsroom.authority.native_current_checkpoint import create_native_source_checkpoint
    with source_fixture(tmp_path,monkeypatch) as (args,runtime,_,_,before,root):
        c=root._connection;c.execute('INSERT INTO native_expired_command_keys VALUES(?,?,?,?,?,?,?,?,?)',MARKER)
        create_native_source_checkpoint(root,runtime.authority.objects,event_ids=(str(before.event_id),),proof=runtime.proof,dev_rebuild=True)
        migrate_native_marker_layout(root,dev_rebuild=True)
        assert runtime.authority.sources.revision(before.request.revision_id,proof=runtime.proof)==before
        assert runtime.authority.sources.record_revision(before.request,proof=runtime.proof).replayed
        schema=tuple(c.execute("SELECT name,sql FROM sqlite_schema WHERE type='trigger' AND sql LIKE '%native_expired_command_keys%' ORDER BY name"))
        assert len(schema)==5  # two immutable + command/event/aggregate insertion
        denied=(
            ('UPDATE native_expired_command_keys SET key=? WHERE namespace=? AND key=?',('other',*MARKER[:2])),
            ('UPDATE native_expired_command_keys SET rowid=rowid+1',()),
            ('DELETE FROM native_expired_command_keys WHERE namespace=? AND key=?',MARKER[:2]),
            # Trigger expressions can be checked with partial rows: BEFORE INSERT
            # must deny the protected identity before NOT NULL business columns.
            ('INSERT INTO authority_commands(command_id,idempotency_namespace,idempotency_key) VALUES(?,?,?)',(MARKER[2],'new','new')),
            ('INSERT INTO authority_commands(command_id,idempotency_namespace,idempotency_key) VALUES(?,?,?)',('new',*MARKER[:2])),
            ('INSERT INTO ledger_events(event_id,ledger_seq) VALUES(?,?)',(MARKER[3],5001)),
            ('INSERT INTO ledger_events(event_id,ledger_seq) VALUES(?,?)',('new',MARKER[4])),
            ('INSERT INTO authority_aggregates(aggregate_type,aggregate_id) VALUES(?,?)',MARKER[7:]),
        )
        for sql,params in denied:
            with pytest.raises(sqlite3.IntegrityError,match='immutable|identity remains reserved'):
                c.execute(sql,params)
        assert tuple(c.execute('SELECT * FROM native_expired_command_keys').fetchone())==MARKER
    with open_native_runtime(**args) as reopened:
        assert reopened.authority.sources.revision(before.request.revision_id,proof=reopened.proof)==before
        assert reopened.authority.sources.record_revision(before.request,proof=reopened.proof).replayed


def test_new45_creator_and_independent_literal_contract(tmp_path):
    from newsroom.authority.native_marker_layout_migrations import initialise_empty_marker_store
    from newsroom.authority.migrations import schema_fingerprint
    c=sqlite3.connect(tmp_path/'new45.sqlite3',isolation_level=None)
    initialise_empty_marker_store(c);require_checkpoint_schema(c)
    assert schema_fingerprint(c)=='sha256:dbeca2d28e2f093d9ac7aaf5a6be68329dcb81c0421a33d88457a15e1c16bc78'
    assert tuple(c.execute('SELECT version,name,checksum FROM authority_migrations WHERE version=45').fetchone())==(
        45,'native_expired_marker_rowid_layout_v45','sha256:6e30c5753eee04144586d27bae45b2eb4593b6ade8666b71442be6dacc592b92')
    c.close()


def test_existing_pending_staged_obligation_and_cas_survive45_reopen(tmp_path,monkeypatch):
    import hashlib
    with source_fixture(tmp_path,monkeypatch) as (args,runtime,_,_,before,root):
        original=root.commit_admission
        def interrupted(*args,**kwargs):raise KeyboardInterrupt('before admission activation')
        monkeypatch.setattr(root,'commit_admission',interrupted)
        with pytest.raises(KeyboardInterrupt):
            runtime.authority.objects.admit(ObjectAdmissionRequest('evidence.source','pending-original-identity'),
                b'original pending bytes',proof=runtime.proof)
        monkeypatch.setattr(root,'commit_admission',original)
        pending=tuple(tuple(row)for row in root._connection.execute("SELECT * FROM object_staging_records WHERE state='STAGED'"))
        assert len(pending)==1
        path=root._cas.staging_root/root._connection.execute("SELECT staged_name FROM object_staging_records WHERE state='STAGED'").fetchone()[0]
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        migrate_native_marker_layout(root,dev_rebuild=True)
        assert tuple(tuple(row)for row in root._connection.execute("SELECT * FROM object_staging_records WHERE state='STAGED'"))==pending
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    with open_native_runtime(**args) as reopened:
        store=reopened.authority._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0]
        assert tuple(tuple(row)for row in store._connection.execute("SELECT * FROM object_staging_records WHERE state='STAGED'"))==pending
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        assert reopened.authority.sources.revision(before.request.revision_id,proof=reopened.proof)==before


def test_bad45_fingerprint_rejected_by_actual_native_open(tmp_path,monkeypatch):
    with source_fixture(tmp_path,monkeypatch) as (args,_,_,_,_,root):
        migrate_native_marker_layout(root,dev_rebuild=True)
        root._connection.execute('CREATE TABLE unexpected_native45(value TEXT)')
    with pytest.raises(sqlite3.DatabaseError,match='schema/history differs'):
        with open_native_runtime(**args):pytest.fail('forged45 OPEN accepted')


def test_marker_migration_preserves_every_other_business_row(tmp_path,monkeypatch):
    with source_fixture(tmp_path,monkeypatch) as (_,_,_,_,_,root):
        c=root._connection
        c.executemany('INSERT INTO native_expired_command_keys VALUES(?,?,?,?,?,?,?,?,?)',
            [(MARKER[0],f'expired-key-{n}',str(uuid.uuid4()),str(uuid.uuid4()),5000+n,*MARKER[5:])for n in range(1000)])
        def other_rows():
            return tuple(line for line in c.iterdump()if line.startswith('INSERT INTO ')and
                not line.startswith('INSERT INTO "authority_migrations"')and
                not line.startswith('INSERT INTO "native_expired_command_keys"'))
        before=other_rows();old=tuple(tuple(row)for row in c.execute('SELECT * FROM native_expired_command_keys ORDER BY namespace,key'))
        result=migrate_native_marker_layout(root,dev_rebuild=True)
        assert result['rows']==1000 and result['physical_file_reclaim']is False
        assert other_rows()==before
        assert tuple(tuple(row)for row in c.execute('SELECT * FROM native_expired_command_keys ORDER BY namespace,key'))==old
        plan=c.execute('EXPLAIN QUERY PLAN SELECT 1 FROM native_expired_command_keys WHERE aggregate_type=? AND aggregate_id=?',MARKER[7:]).fetchone()[3]
        assert 'SEARCH'in plan and 'idx_native_expired_aggregate'in plan


def test_native45_marker_primary_key_and_unique_constraints_remain_exact(tmp_path):
    from newsroom.authority.native_marker_layout_migrations import initialise_empty_marker_store
    c=sqlite3.connect(tmp_path/'constraints45.sqlite3',isolation_level=None);initialise_empty_marker_store(c)
    c.execute('INSERT INTO native_expired_command_keys VALUES(?,?,?,?,?,?,?,?,?)',MARKER)
    for index in (0,2,3,4):
        candidate=('sha256:'+'b'*64,'fresh-key',str(uuid.uuid4()),str(uuid.uuid4()),6000,*MARKER[5:])
        candidate=list(candidate)
        if index==0:candidate[:2]=MARKER[:2]
        else:candidate[index]=MARKER[index]
        with pytest.raises(sqlite3.IntegrityError,match='UNIQUE'):
            c.execute('INSERT INTO native_expired_command_keys VALUES(?,?,?,?,?,?,?,?,?)',candidate)
    assert c.execute('SELECT count(*) FROM native_expired_command_keys').fetchone()[0]==1
    c.close()


from newsroom.tests.test_native_sparse_projection_checkpoint import sparse_current


def test_nonempty_native_graphiti45_replay_and_bad_graph_proof_stays_denied(sparse_current):
    from newsroom.increment4 import Increment4Neo4jCurrentBuildRequest
    from newsroom.projection import ProjectionGenerationId,ProjectionGenerationState
    from newsroom.projection.neo4j import Neo4jIdentityConflict
    from newsroom.tests.increment4e_governed_path_helpers import open_graphiti_path_increment4_neo4j_system
    from newsroom.tests.extraction_4a_helpers import extraction_proof
    relation,adapter,_,_,captured=sparse_current
    request=Increment4Neo4jCurrentBuildRequest(ProjectionGenerationId.new(),'CURRENT_AFTER_SELECTED_COPY',
        'marker45-required-current',allow_active_extension=True)
    with open_graphiti_path_increment4_neo4j_system(relation,adapter) as system:
        built=system.increment4.build_current_and_promote(request,proof=extraction_proof())
        assert built.generation.state is ProjectionGenerationState.ACTIVE
        root=captured[-1]
        required=tuple(tuple(row)for row in root._connection.execute(
            'SELECT * FROM projection_delivery_states WHERE generation_id=? ORDER BY ledger_seq',
            (str(built.generation.generation_id),)))
        assert required
        assert root._connection.execute("SELECT count(*) FROM projection_delivery_states WHERE generation_id=? AND required=1",
            (str(built.generation.generation_id),)).fetchone()[0]==0  # actual native mapping, not a fabricated flag
        before=system.increment4.generation_status(built.generation.generation_id,proof=extraction_proof())
        migrate_native_marker_layout(root,dev_rebuild=True)
        assert tuple(tuple(row)for row in root._connection.execute(
            'SELECT * FROM projection_delivery_states WHERE generation_id=? ORDER BY ledger_seq',
            (str(built.generation.generation_id),)))==required
    with open_graphiti_path_increment4_neo4j_system(relation,adapter) as system:
        assert system.increment4.generation_status(built.generation.generation_id,proof=extraction_proof())==before
        assert system.increment4.build_current_and_promote(request,proof=extraction_proof())==built
        sequence=next(seq for generation,seq in adapter.deliveries if generation==str(built.generation.generation_id))
        adapter.corrupt_delivery_digest(built.generation.generation_id,sequence)
        with pytest.raises(Neo4jIdentityConflict):
            system.increment4.build_current_and_promote(request,proof=extraction_proof())
