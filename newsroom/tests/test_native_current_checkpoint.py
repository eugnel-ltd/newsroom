"""Normal-FK DEV source checkpoint fixture; original source/pending rights survive."""
from contextlib import nullcontext,contextmanager
from datetime import UTC,datetime
import sqlite3
from dataclasses import replace
import pytest
from newsroom.authority import AuthorityPersistenceError, ObjectAdmissionId, ObjectAdmissionRequest, DiagnosticHistoryExpired, EventReadPolicy, MetadataClass, TrustScope
from newsroom.authority._graphiti_increment4_system import _AUTHORITY_COMPOSITION_TOKEN
from newsroom.authority.native_current_checkpoint import create_native_source_checkpoint
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.native_source_intake import NativeSourceIntake
from newsroom.control_plane.graphiti_operational_readiness import OPERATOR_PRINCIPAL_ID,OPERATOR_AUTHORITY_DOMAIN
from newsroom.increment9.proving import SOURCE_URLS
from newsroom.sources import SourceRevisionId
from .test_native_runtime import _args
from .test_native_source_intake import _seed_uk01,_licence,_document,ATOM


@contextmanager
def source_fixture(tmp_path,monkeypatch):
    args=_args(tmp_path,monkeypatch)
    args.update(principal_id=OPERATOR_PRINCIPAL_ID,authority_domain=OPERATOR_AUTHORITY_DOMAIN)
    from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store
    with sqlite3.connect(args['authority_path'],isolation_level=None) as connection:
        initialise_empty_checkpoint_store(connection)
    args['authority_path'].chmod(0o600)
    with open_native_runtime(**args) as runtime:
        definition=_seed_uk01(runtime)
        retained={}
        intake=NativeSourceIntake(sources=runtime.authority.sources,objects=runtime.authority.objects,
            proof=runtime.proof,definition_ids={'UK-01':definition},licence=_licence(),retained_units=retained,
            dispatch_fence=lambda *_:nullcontext(),fetch=lambda url:(200,ATOM if url==SOURCE_URLS['UK-01'] else _document()),
            clock=lambda:datetime(2026,9,8,12,tzinfo=UTC))
        first=intake.poll()[0];assert first.status=='READY',first
        unit,=first.units;retained[unit.revision_id]=first.units
        before=runtime.authority.sources.revision(SourceRevisionId.parse(unit.revision_id),proof=runtime.proof)
        root=runtime.authority._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0]
        conn=root._connection
        yield args,runtime,intake,first,before,root


def test_normal_fk_source_checkpoint_replays_same_source_without_rpc(tmp_path,monkeypatch):
    with source_fixture(tmp_path,monkeypatch) as (args,runtime,intake,first,before,root):
        conn=root._connection
        unit,=first.units
        original=conn.execute('SELECT * FROM ledger_events WHERE event_id=?',(str(before.event_id),)).fetchone()
        original_context=original['authentication_context_id']
        root_rights=conn.execute('SELECT count(*) FROM object_rights_decisions').fetchone()[0]
        checkpoint=create_native_source_checkpoint(root,runtime.authority.objects,event_ids=(str(before.event_id),),
            proof=runtime.proof,dev_rebuild=True)
        assert conn.execute('PRAGMA foreign_key_check').fetchall()==[]
        assert conn.execute('SELECT count(*) FROM authentication_contexts WHERE authentication_context_id=?',(original_context,)).fetchone()[0]==0
        assert conn.execute('SELECT native_checkpoint_id FROM ledger_events WHERE event_id=?',(str(before.event_id),)).fetchone()[0]==checkpoint
        assert runtime.authority.sources.revision(SourceRevisionId.parse(unit.revision_id),proof=runtime.proof)==before
        with pytest.raises(DiagnosticHistoryExpired):
            root.event_provenance(event_id=str(before.event_id),policy=EventReadPolicy(
                policy_id='fixture-source-provenance',purpose='fixture',required_scope='fixture.read',
                allowed_principal_ids=frozenset({OPERATOR_PRINCIPAL_ID}),
                allowed_security_scopes=frozenset({'authority.source_registry'}),
                allowed_trust_scopes=frozenset({TrustScope.OBSERVED}),
                metadata_classes=frozenset({MetadataClass.PROVENANCE}),max_results=1))
        replay=runtime.authority.sources.record_revision(before.request,proof=runtime.proof)
        assert replay.replayed and replay.event_id==before.event_id
        repeated=intake.poll()[0]
        assert repeated.units==first.units
        assert conn.execute('SELECT count(*) FROM object_rights_decisions').fetchone()[0]==root_rights+1 # checkpoint metadata only
    with open_native_runtime(**args) as runtime:
        assert runtime.authority.sources.revision(SourceRevisionId.parse(unit.revision_id),proof=runtime.proof)==before
        replay=runtime.authority.sources.record_revision(before.request,proof=runtime.proof)
        assert replay.replayed and replay.event_id==before.event_id


@pytest.mark.parametrize('failure',[RuntimeError('fixture failure'),KeyboardInterrupt('fixture interrupt')])
def test_expiry_rollback_deduplicates_prepared_metadata_without_source_mutation(tmp_path,monkeypatch,failure):
    from newsroom.authority import native_current_checkpoint as checkpoint
    with source_fixture(tmp_path,monkeypatch) as (_,runtime,_,_,before,root):
        conn=root._connection;event_id=str(before.event_id)
        original=dict(conn.execute('SELECT * FROM ledger_events WHERE event_id=?',(event_id,)).fetchone())
        count=conn.execute('SELECT count(*) FROM object_admissions').fetchone()[0]
        expire=checkpoint._expire_source_rpc
        def fail(*a,**kw):
            expire(*a,**kw);raise failure
        monkeypatch.setattr(checkpoint,'_expire_source_rpc',fail)
        for _ in range(2):
            with pytest.raises(type(failure)):
                create_native_source_checkpoint(root,runtime.authority.objects,event_ids=(event_id,),proof=runtime.proof,dev_rebuild=True)
            assert not conn.in_transaction
            assert dict(conn.execute('SELECT * FROM ledger_events WHERE event_id=?',(event_id,)).fetchone())==original
            assert conn.execute('SELECT count(*) FROM native_current_checkpoints').fetchone()[0]==0
            assert conn.execute('SELECT count(*) FROM object_admissions').fetchone()[0]==count+1
            assert conn.execute('PRAGMA foreign_key_check').fetchall()==[]


def test_forged_prepared_admission_is_rejected_before_checkpoint_commit(tmp_path,monkeypatch):
    from newsroom.authority.native_current_checkpoint import _capture,VERSION
    from newsroom.authority.canonical import canonical_json_bytes,digest_bytes
    with source_fixture(tmp_path,monkeypatch) as (_,runtime,_,_,before,root):
        event_id=str(before.event_id)
        with root._transaction():member=_capture(root,event_id)
        expected=canonical_json_bytes({'version':VERSION,'members':{event_id:member}})
        # Deliberately poison the right key with unrelated, self-consistent bytes.
        runtime.authority.objects.admit(ObjectAdmissionRequest('source.native-observation',
            'native-dev-source-checkpoint:'+digest_bytes(expected)),b'{"forged":"fresh self digest"}',proof=runtime.proof)
        with pytest.raises(AuthorityPersistenceError,match='admission differs'):
            create_native_source_checkpoint(root,runtime.authority.objects,event_ids=(event_id,),proof=runtime.proof,dev_rebuild=True)
        assert root._connection.execute('SELECT count(*) FROM native_current_checkpoints').fetchone()[0]==0
        assert root._connection.execute('SELECT retired_header_digest FROM ledger_events WHERE event_id=?',(event_id,)).fetchone()[0] is None
        assert root._connection.execute('PRAGMA foreign_key_check').fetchall()==[]


def test_original_header_mutation_or_missing_dev_authority_never_qualifies(tmp_path,monkeypatch):
    with source_fixture(tmp_path,monkeypatch) as (_,runtime,_,_,before,root):
        event_id=str(before.event_id);conn=root._connection
        with pytest.raises(AuthorityPersistenceError,match='DEV writer'):
            create_native_source_checkpoint(root,runtime.authority.objects,event_ids=(event_id,),proof=runtime.proof)
        with root._transaction():
            conn.execute('DROP TRIGGER immutable_ledger_events_update')
            conn.execute("UPDATE ledger_events SET payload_digest=? WHERE event_id=?",('sha256:'+'f'*64,event_id))
        with pytest.raises(AuthorityPersistenceError):
            create_native_source_checkpoint(root,runtime.authority.objects,event_ids=(event_id,),proof=runtime.proof,dev_rebuild=True)
        assert conn.execute('SELECT count(*) FROM native_current_checkpoints').fetchone()[0]==0


@pytest.mark.parametrize('history',[0,32,1000])
def test_discovery_current_head_survives_superseded_history_drop_and_next_ordinal(tmp_path,monkeypatch,history,record_property):
    from time import perf_counter
    import resource
    from newsroom.authority.native_current_checkpoint import prune_superseded_discovery
    from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store
    from .discovery_3d_authority_helpers import seed_check_lineage,exact_admission_request,exact_gate_request,exact_initial_disposition
    from newsroom.discovery import GateDecisionId,LeadDispositionDecisionId,DiscoveryVersionConflict
    args=_args(tmp_path,monkeypatch)
    with sqlite3.connect(args['authority_path'],isolation_level=None) as c:initialise_empty_checkpoint_store(c)
    args['authority_path'].chmod(0o600)
    with open_native_runtime(**args) as runtime:
        system=runtime.authority;seed_check_lineage(system)
        request=exact_admission_request()
        system.discovery.admit_signal_to_lead(request,proof=runtime.proof)
        gate=system.discovery.current_gate(request.signal.signal_id,proof=runtime.proof)
        disposition=system.discovery.current_disposition(request.lead.lead_id,proof=runtime.proof)
        for ordinal in range(2,history+2):
            gate_request=replace(exact_gate_request(),decision_id=GateDecisionId.new(),decision_ordinal=ordinal,
                previous_decision_id=gate.request.decision_id,idempotency_key=f'gate-history-{ordinal}')
            gate=system.discovery.decide_gate(gate_request,proof=runtime.proof)
            disposition_request=replace(exact_initial_disposition(),decision_id=LeadDispositionDecisionId.new(),
                decision_ordinal=ordinal,previous_decision_id=disposition.request.decision_id,
                gate_decision_id=gate.request.decision_id,idempotency_key=f'disposition-history-{ordinal}')
            disposition=system.discovery.record_lead_disposition(disposition_request,proof=runtime.proof)
        root=system._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0]
        conn=root._connection
        bytes_before=conn.execute('SELECT coalesce(sum(length(canonical_bytes)),0) FROM discovery_gate_decisions').fetchone()[0]+conn.execute('SELECT coalesce(sum(length(canonical_bytes)),0) FROM lead_disposition_decisions').fetchone()[0]
        started=perf_counter()
        result=prune_superseded_discovery(root,system.objects,proof=runtime.proof,dev_rebuild=True)
        elapsed=perf_counter()-started
        bytes_after=conn.execute('SELECT coalesce(sum(length(canonical_bytes)),0) FROM discovery_gate_decisions').fetchone()[0]+conn.execute('SELECT coalesce(sum(length(canonical_bytes)),0) FROM lead_disposition_decisions').fetchone()[0]
        record_property('history',history);record_property('prune_wall_seconds',elapsed)
        record_property('canonical_bytes_before_after',str((bytes_before,bytes_after)))
        record_property('fixture_maxrss_bytes',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        assert result['deleted']=={'lead_disposition_decisions':history,'discovery_gate_decisions':max(history-1,0)}
        assert conn.execute('PRAGMA foreign_key_check').fetchall()==[]
        assert system.discovery.current_gate(request.signal.signal_id,proof=runtime.proof)==gate
        assert system.discovery.current_disposition(request.lead.lead_id,proof=runtime.proof)==disposition
        assert (gate.request.previous_decision_id is not None)==bool(history)
        assert (disposition.request.previous_decision_id is not None)==bool(history)
        next_gate=replace(gate.request,decision_id=GateDecisionId.new(),decision_ordinal=history+2,
            previous_decision_id=gate.request.decision_id,idempotency_key='next-exact-gate')
        appended=system.discovery.decide_gate(next_gate,proof=runtime.proof)
        assert appended.request.decision_ordinal==history+2
        # A changed Gate correctly invalidates the old disposition's currency.
        with pytest.raises(LookupError):
            system.discovery.current_disposition(request.lead.lead_id,proof=runtime.proof)
        next_disposition=replace(disposition.request,decision_id=LeadDispositionDecisionId.new(),decision_ordinal=history+2,
            previous_decision_id=disposition.request.decision_id,gate_decision_id=appended.request.decision_id,
            idempotency_key='next-exact-disposition')
        updated_disposition=system.discovery.record_lead_disposition(next_disposition,proof=runtime.proof)
        with pytest.raises(DiscoveryVersionConflict,match="exact current head"):
            system.discovery.decide_gate(replace(next_gate,decision_id=GateDecisionId.new(),
                previous_decision_id=gate.request.previous_decision_id or gate.request.decision_id,idempotency_key='old-head-resurrection'),proof=runtime.proof)
    with open_native_runtime(**args) as runtime:
        assert runtime.authority.discovery.current_gate(request.signal.signal_id,proof=runtime.proof)==appended
        assert runtime.authority.discovery.current_disposition(request.lead.lead_id,proof=runtime.proof)==updated_disposition


def test_new_store_selects_current_discovery_without_copying_old_rpc(tmp_path,monkeypatch,record_property):
    from time import perf_counter
    from newsroom.authority.native_current_rebuild import copy_selected_native_store
    from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store
    from .discovery_3d_authority_helpers import seed_check_lineage,exact_admission_request,exact_gate_request,exact_initial_disposition
    from newsroom.discovery import GateDecisionId,LeadDispositionDecisionId
    args=_args(tmp_path,monkeypatch)
    destination=tmp_path/'selected.sqlite3'
    with open_native_runtime(**args) as runtime:
        system=runtime.authority;seed_check_lineage(system)
        request=exact_admission_request();system.discovery.admit_signal_to_lead(request,proof=runtime.proof)
        gate=system.discovery.current_gate(request.signal.signal_id,proof=runtime.proof)
        disposition=system.discovery.current_disposition(request.lead.lead_id,proof=runtime.proof)
        for ordinal in range(2,34):
            gate=system.discovery.decide_gate(replace(exact_gate_request(),decision_id=GateDecisionId.new(),decision_ordinal=ordinal,
                previous_decision_id=gate.request.decision_id,idempotency_key=f'gate-copy-{ordinal}'),proof=runtime.proof)
            disposition=system.discovery.record_lead_disposition(replace(exact_initial_disposition(),decision_id=LeadDispositionDecisionId.new(),
                decision_ordinal=ordinal,previous_decision_id=disposition.request.decision_id,gate_decision_id=gate.request.decision_id,
                idempotency_key=f'disposition-copy-{ordinal}'),proof=runtime.proof)
        root=system._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0]
        old_copy=root._connection.execute('SELECT * FROM authority_commands WHERE idempotency_key=?',('gate-copy-2',)).fetchone()
        old_gate=root._connection.execute('SELECT decision_id FROM discovery_gate_decisions WHERE authority_event_id=?',
            (root._connection.execute('SELECT event_id FROM ledger_events WHERE command_id=?',(old_copy['command_id'],)).fetchone()[0],)).fetchone()[0]
        with sqlite3.connect(destination,isolation_level=None) as c:
            initialise_empty_checkpoint_store(c)
            start=perf_counter()
            counts=copy_selected_native_store(root,c,roots={},dev_rebuild=True)
            record_property('copy_wall_seconds',perf_counter()-start)
            assert c.execute('PRAGMA foreign_key_check').fetchall()==[]
            assert counts['discovery_gate_decisions']==2
            assert counts['lead_disposition_decisions']==1
            assert counts['native_expired_command_keys']==63
            assert c.execute('SELECT 1 FROM authority_commands WHERE command_id=?',(old_copy['command_id'],)).fetchone() is None
        record_property('source_disk_bytes',args['authority_path'].stat().st_size+args['authority_path'].with_name('authority.sqlite3-wal').stat().st_size)
        record_property('new_disk_bytes',destination.stat().st_size)
        assert root._connection.execute('SELECT count(*) FROM discovery_gate_decisions').fetchone()[0]==33
    destination.chmod(0o600)
    args['authority_path']=destination
    with open_native_runtime(**args) as runtime:
        system=runtime.authority
        assert system.discovery.current_gate(request.signal.signal_id,proof=runtime.proof)==gate
        assert system.discovery.current_disposition(request.lead.lead_id,proof=runtime.proof)==disposition
        with pytest.raises(DiagnosticHistoryExpired):
            system._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0].find(idempotency_namespace=old_copy['idempotency_namespace'],idempotency_key='gate-copy-2')
        # A fresh key must not resurrect an expired immutable decision UUID.
        with pytest.raises(AuthorityPersistenceError,match='aggregate diagnostic history expired'):
            system.discovery.decide_gate(replace(gate.request,decision_id=GateDecisionId.parse(old_gate),decision_ordinal=34,
                previous_decision_id=gate.request.decision_id,idempotency_key='resurrect-old-gate-fresh-key'),proof=runtime.proof)
        next_gate=system.discovery.decide_gate(replace(gate.request,decision_id=GateDecisionId.new(),decision_ordinal=34,
            previous_decision_id=gate.request.decision_id,idempotency_key='next-copied-gate'),proof=runtime.proof)
        assert next_gate.request.decision_ordinal==34
    with open_native_runtime(**args) as runtime:
        assert runtime.authority.discovery.current_gate(request.signal.signal_id,proof=runtime.proof)==next_gate


def test_current_source_inventory_copy_preserves_rights_cas_and_repoll(tmp_path,monkeypatch):
    from newsroom.control_plane.native_progress import NativeRevisionJournal
    from newsroom.control_plane.store import connect
    from newsroom.authority.native_current_rebuild import copy_selected_native_store,source_roots_from_current
    from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store
    destination=tmp_path/'source-selected.sqlite3'
    journal_connection=connect(str(tmp_path/'current.sqlite3'))
    journal=NativeRevisionJournal(journal_connection)
    with source_fixture(tmp_path,monkeypatch) as (args,runtime,intake,first,before,root):
        journal.land(first.units);journal.sources((first,))
        journal.advance(first.units[0].revision_id,stage='GRAPHITI_HOLD',facts={'reason':'NOT_DISPATCHED'})
        current=journal.current(first.units[0].revision_id)
        roots=source_roots_from_current(journal)
        rights={identity:dict(root._connection.execute('SELECT r.* FROM object_admissions a JOIN object_rights_decisions r '
            'ON r.rights_decision_id=a.rights_decision_id WHERE a.admission_id=?',identity).fetchone()) for identity in roots['object_admissions']}
        with sqlite3.connect(destination,isolation_level=None) as c:
            initialise_empty_checkpoint_store(c)
            copy_selected_native_store(root,c,roots=roots,dev_rebuild=True)
            assert c.execute('PRAGMA foreign_key_check').fetchall()==[]
    destination.chmod(0o600);args['authority_path']=destination
    for reopen in range(2):
      with open_native_runtime(**args) as runtime:
        root=runtime.authority._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0]
        assert runtime.authority.sources.revision(before.request.revision_id,proof=runtime.proof)==before
        for identity,original in rights.items():
            assert dict(root._connection.execute('SELECT r.* FROM object_admissions a JOIN object_rights_decisions r '
                'ON r.rights_decision_id=a.rights_decision_id WHERE a.admission_id=?',identity).fetchone())==original
            admission=root.admission_view(ObjectAdmissionId.parse(identity[0]))
            pinned=root._cas.pin(admission.blob)
            try:root._cas.verify_pinned(pinned)
            finally:pinned.close()
        repeated=NativeSourceIntake(sources=runtime.authority.sources,objects=runtime.authority.objects,proof=runtime.proof,
            definition_ids=intake._definitions,licence=_licence(),retained_units=journal.units,
            dispatch_fence=lambda *_:nullcontext(),fetch=lambda url:(200,ATOM if url==SOURCE_URLS['UK-01'] else _document()),
            clock=lambda:datetime(2026,9,8,12,tzinfo=UTC)).poll()[0]
        assert repeated.status=='READY' and repeated.units==first.units
        assert journal.current(first.units[0].revision_id)==current
    journal_connection.close()


@pytest.mark.parametrize('failure',[RuntimeError('copy failure'),KeyboardInterrupt('copy interrupt')])
def test_selected_copy_rolls_back_all_destination_rows_and_temp_state(tmp_path,monkeypatch,failure):
    from newsroom.authority.native_current_rebuild import copy_selected_native_store
    from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store,require_checkpoint_schema
    with source_fixture(tmp_path,monkeypatch) as (_,runtime,_,_,before,root):
        original=root._validate_retained_event
        def interrupted(event_id):original(event_id);raise failure
        monkeypatch.setattr(root,'_validate_retained_event',interrupted)
        with sqlite3.connect(tmp_path/'interrupted.sqlite3',isolation_level=None) as c:
            initialise_empty_checkpoint_store(c)
            with pytest.raises(type(failure)):
                copy_selected_native_store(root,c,roots={'source_revisions':((str(before.request.revision_id),),)},dev_rebuild=True)
            assert not c.in_transaction and not root._connection.in_transaction
            assert c.execute('SELECT count(*) FROM ledger_events').fetchone()[0]==0
            assert c.execute('SELECT count(*) FROM source_revisions').fetchone()[0]==0
            assert root._connection.execute("SELECT count(*) FROM sqlite_temp_schema WHERE name LIKE '_native_rebuild_%'").fetchone()[0]==0
            assert c.execute('PRAGMA foreign_key_check').fetchall()==[]
            require_checkpoint_schema(c)


@pytest.mark.parametrize('damaged',[b'{}',b'[]',b'invalid json'])
def test_checkpoint_canonical_member_tamper_is_not_a_new_proof(tmp_path,monkeypatch,damaged):
    with source_fixture(tmp_path,monkeypatch) as (_,runtime,_,_,before,root):
        event_id=str(before.event_id)
        create_native_source_checkpoint(root,runtime.authority.objects,event_ids=(event_id,),proof=runtime.proof,dev_rebuild=True)
        conn=root._connection
        with pytest.raises(sqlite3.IntegrityError,match='immutable native checkpoint'):
            conn.execute("UPDATE native_current_checkpoint_members SET canonical_bytes=x'7b7d' WHERE event_id=?",(event_id,))
        with root._transaction():
            conn.execute('DROP TRIGGER immutable_native_current_checkpoint_members_update')
            conn.execute('UPDATE native_current_checkpoint_members SET canonical_bytes=? WHERE event_id=?',(damaged,event_id))
        with pytest.raises(AuthorityPersistenceError,match='original proof differs'):
            runtime.authority.sources.revision(before.request.revision_id,proof=runtime.proof)
        # Class-method decoders use the SQL envelope seam. SQLite must not mask
        # the same denial as an OperationalError from a user-defined function.
        with pytest.raises(AuthorityPersistenceError,match='original proof differs'):
            root._record_context(conn,event_id=event_id)


def test_new_only_schema_has_independent_literal_identity_and_never_upgrades_old_store(tmp_path):
    from newsroom.authority import migrations
    from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store,require_checkpoint_schema
    with sqlite3.connect(tmp_path/'new.sqlite3',isolation_level=None) as c:
        initialise_empty_checkpoint_store(c)
        assert migrations.schema_fingerprint(c)=='sha256:140620ca77bb4c22f1064b45af7dd2aff44a12da999f53b6eb3ce8b624033627'
        assert c.execute('SELECT version,name,checksum FROM authority_migrations ORDER BY version DESC LIMIT 1').fetchone()==(
            44,'native_current_business_checkpoint_v44','sha256:381cf0a4e32b1de3521b3c3387e2bde0a48ebfdbe3bcb490f7b86fac56cd1d90')
        require_checkpoint_schema(c)
        with pytest.raises(sqlite3.IntegrityError,match='immutable migration history'):
            c.execute("UPDATE authority_migrations SET checksum=? WHERE version=44",('sha256:'+'f'*64,))
        name,guard=c.execute("SELECT name,sql FROM sqlite_schema WHERE type='trigger' AND tbl_name='authority_migrations' AND sql LIKE '%BEFORE UPDATE%'").fetchone()
        c.execute('DROP TRIGGER '+name)
        c.execute("UPDATE authority_migrations SET checksum=? WHERE version=44",('sha256:'+'f'*64,))
        c.execute(guard)
        with pytest.raises(sqlite3.DatabaseError,match='schema/history differs'):require_checkpoint_schema(c)
        c.execute('DROP TRIGGER '+name)
        c.execute("UPDATE authority_migrations SET checksum=? WHERE version=44",('sha256:381cf0a4e32b1de3521b3c3387e2bde0a48ebfdbe3bcb490f7b86fac56cd1d90',))
        c.execute(guard)
        c.execute('DROP TRIGGER native_expired_command_insert_guard')
        with pytest.raises(sqlite3.DatabaseError,match='schema/history differs'):require_checkpoint_schema(c)
    with sqlite3.connect(tmp_path/'old.sqlite3',isolation_level=None) as c:
        migrations.apply_pending_migrations(c,applied_at='1970-01-01T00:00:00.000000Z')
        assert migrations.schema_fingerprint(c)=='sha256:a00dd159b3743d3e964c99a8fe4f59e3e772d8321cc472c779ded19d82779b0b'
        with pytest.raises(sqlite3.DatabaseError,match='empty NEW'):initialise_empty_checkpoint_store(c)
        assert c.execute('PRAGMA user_version').fetchone()[0]==43
        assert migrations.schema_fingerprint(c)=='sha256:a00dd159b3743d3e964c99a8fe4f59e3e772d8321cc472c779ded19d82779b0b'


def test_publication_current_roots_resolve_real_receipts_to_ledger_primary_keys(tmp_path):
    from newsroom.authority.native_current_rebuild import publication_event_roots
    from .test_increment10_private_serving import _context,_delivery,_close
    from .authority_helpers import proof
    context=_context(tmp_path);delivery=_delivery(tmp_path,context)
    try:
        request=dict(story_receipt=context[-2],candidate_port=context[1],proof=proof())
        attempt,_=delivery.begin(context[-1],**request)
        delivery.apply(attempt,publication_receipt=context[-1],applied_at='2026-07-16T11:00:00Z',**request)
        observed=delivery.observe(attempt,publication_receipt=context[-1],observed_at='2026-07-16T11:30:00Z',**request)
        receipt=delivery.record(observed,attempt,expected_version=0,proof=proof())
        refs={'story_event_id':context[-2].event_id,'publication_event_id':context[-1].event_id,
              'delivery_attempt_event_id':attempt.event_id,'delivery_evidence_event_id':receipt.event_id}
        with sqlite3.connect(tmp_path/'objects.sqlite3') as connection:
            package_id=connection.execute("SELECT admission_id FROM object_admissions WHERE object_class='evidence_package' LIMIT 1").fetchone()[0]
            facts={**refs,'package_admission_id':package_id,
                'factual_correction_intent':{'superseded':{'story_event_id':refs['story_event_id']},
                    'reviewed':{'publication_event_id':refs['publication_event_id']},'package_admission_id':package_id},
                'copy_correction_origin':{'story_event_id':refs['story_event_id']},
                'copy_correction_of':{'publication_event_id':refs['publication_event_id']},
                'factual_correction_result':{'delivery_attempt_event_id':refs['delivery_attempt_event_id'],
                    'delivery_evidence_event_id':refs['delivery_evidence_event_id']}}
            roots=publication_event_roots(connection,facts)
            expected=tuple(sorted((connection.execute('SELECT ledger_seq FROM ledger_events WHERE event_id=?',(eid,)).fetchone()[0],) for eid in refs.values()))
            assert roots['ledger_events']==expected
            assert roots['object_admissions']==((package_id,),)
            originals=publication_event_roots(connection,{'copy_correction_origin':facts['copy_correction_origin'],
                'copy_correction_of':facts['copy_correction_of'],'factual_correction_result':facts['factual_correction_result']})
            assert originals['ledger_events']==expected
            nested=publication_event_roots(connection,{'factual_correction_intent':facts['factual_correction_intent'],
                'factual_correction_result':facts['factual_correction_result']})
            assert nested==roots
            assert all(type(key[0]) is int for key in roots['ledger_events'])
            before=connection.total_changes
            with pytest.raises(AuthorityPersistenceError,match='publication event is absent'):
                publication_event_roots(connection,{**facts,'story_event_id':'missing-current-event'})
            assert connection.total_changes==before
    finally:_close(context,delivery)


def test_selected_graphiti_manifest_keeps_passages_proposals_and_replay_on_reopen(tmp_path):
    from .graphiti_adapter_4d_authority_helpers import seed_graphiti_authority_fixture,open_graphiti_system,fake_attempt,approval_from_authority,replay_attempt_for_next_version
    from .extraction_4a_helpers import extraction_proof
    from newsroom.extraction.types import FixtureExtractionCase
    from newsroom.authority.native_current_rebuild import copy_selected_native_store
    from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store
    from newsroom.authority._graphiti_adapter_store_common import _GraphitiAdapterStoreSupport
    state=seed_graphiti_authority_fixture(tmp_path/'origin',fixture_case=FixtureExtractionCase.BILINGUAL_PARTIAL)
    request=fake_attempt(state,fixture_case=FixtureExtractionCase.BILINGUAL_PARTIAL)
    destination=tmp_path/'graphiti-selected.sqlite3'
    with open_graphiti_system(state,workspace_root=(tmp_path/'workspaces').resolve()) as system:
        system.graphiti.register_configuration(request.configuration,proof=extraction_proof())
        attempt=system.graphiti.execute_attempt(request,proof=extraction_proof())
    approval_request=approval_from_authority(state,attempt)
    with open_graphiti_system(state,workspace_root=(tmp_path/'workspaces').resolve()) as system:
        approval=system.graphiti.approve_replay(approval_request,proof=extraction_proof())
        replay=replay_attempt_for_next_version(state,attempt,approval.source)
        system.graphiti.register_configuration(replay.configuration,proof=extraction_proof())
        retained=system.graphiti.execute_attempt(replay,proof=extraction_proof())
        # A fully verified real authority fixture explicitly opts into the
        # native DEV copy boundary; this is not a production activation.
        boundary=system.graphiti._GovernedGraphitiProposalAdapter__attempt.__self__
        root=boundary._store
        root._current_state_only=True
        with sqlite3.connect(destination,isolation_level=None) as connection:
            initialise_empty_checkpoint_store(connection)
            counts=copy_selected_native_store(root,connection,
                roots={'graphiti_adapter_attempts':((str(retained.attempt_id),),)},dev_rebuild=True)
            assert counts['graphiti_input_manifest_passages']==len(request.manifest.passages)+len(replay.manifest.passages)
            assert counts['extraction_proposal_evidence']>0
            assert counts['graphiti_adapter_attempt_replays']==1
            assert counts['graphiti_replay_sources']==1
            assert connection.execute('PRAGMA foreign_key_check').fetchall()==[]
    with sqlite3.connect(destination) as connection:
        connection.row_factory=sqlite3.Row
        for original in (request.manifest,replay.manifest):
            row=connection.execute('SELECT * FROM graphiti_input_manifests WHERE manifest_id=?',(str(original.manifest_id),)).fetchone()
            restored=_GraphitiAdapterStoreSupport._graphiti_manifest_from_row(connection,row)
            assert restored==original and restored.canonical_digest==original.canonical_digest


def test_current_projection_copy_uses_latest_checkpoint_and_appends_after_reopen(tmp_path,monkeypatch):
    from .projection_b2_helpers import open_b2_system,MemoryNeo4jAdapter,proof,source_command
    from .test_projection_b2_authority import _register_and_create,_source_event
    from .projection_b1_helpers import FAMILY_ID
    from newsroom.projection.neo4j import StructuralDeliveryRequest
    from newsroom.authority._increment4_projection_store import _Increment4ProjectionAuthorityStore
    from newsroom.authority.native_current_rebuild import projection_roots_from_current,copy_selected_native_store
    from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store
    monkeypatch.setattr(_Increment4ProjectionAuthorityStore,'_current_state_only',True)
    adapter=MemoryNeo4jAdapter();source_path=tmp_path/'projection-origin.sqlite3';destination=tmp_path/'projection-selected.sqlite3'
    with open_b2_system(source_path,adapter) as system:
        generation=_register_and_create(system)
        root=system.structural._Neo4jStructuralProjector__deliver.__self__._store
        for number in range(3):
            event=_source_event(system,key=f'checkpoint-source-{number}')
            current=root.projection_generation(generation.generation_id)
            system.structural.deliver(StructuralDeliveryRequest(generation.generation_id,current.authority_aggregate_version,
                event.ledger_seq,f'checkpoint-delivery-{number}'),proof=proof())
        roots=projection_roots_from_current(root._connection)
        assert len(roots['projection_checkpoint_versions'])==1
        assert root._connection.execute('SELECT count(*) FROM projection_checkpoint_versions').fetchone()[0]>1
        with sqlite3.connect(destination,isolation_level=None) as connection:
            initialise_empty_checkpoint_store(connection)
            copy_selected_native_store(root,connection,roots=roots,dev_rebuild=True)
            assert connection.execute('SELECT count(*) FROM projection_checkpoint_versions').fetchone()[0]==1
            assert connection.execute('PRAGMA foreign_key_check').fetchall()==[]
    destination.chmod(0o600)
    with open_b2_system(destination,adapter) as system:
        event=_source_event(system,key='after-copy-projection-source')
        root=system.structural._Neo4jStructuralProjector__deliver.__self__._store
        current=root.projection_generation(generation.generation_id)
        system.structural.deliver(StructuralDeliveryRequest(generation.generation_id,current.authority_aggregate_version,
            event.ledger_seq,'after-copy-projection-delivery'),proof=proof())
        assert root.projection_status(FAMILY_ID).contiguous_ledger_seq>=event.ledger_seq
    with open_b2_system(destination,adapter) as system:
        assert system.projections.status(FAMILY_ID,proof=proof()).contiguous_ledger_seq>=event.ledger_seq


@pytest.mark.parametrize('failure',[RuntimeError('prune failure'),KeyboardInterrupt('prune interrupt')])
def test_discovery_expiry_interrupt_restores_every_original_row_and_guard(tmp_path,monkeypatch,failure):
    from newsroom.authority.native_current_checkpoint import discovery_current_root_ids,prune_superseded_discovery
    from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store,require_checkpoint_schema
    from .discovery_3d_authority_helpers import seed_check_lineage,exact_admission_request,exact_gate_request,exact_initial_disposition
    from newsroom.discovery import GateDecisionId,LeadDispositionDecisionId
    args=_args(tmp_path,monkeypatch)
    with sqlite3.connect(args['authority_path'],isolation_level=None) as c:initialise_empty_checkpoint_store(c)
    args['authority_path'].chmod(0o600)
    with open_native_runtime(**args) as runtime:
        system=runtime.authority;seed_check_lineage(system)
        request=exact_admission_request();system.discovery.admit_signal_to_lead(request,proof=runtime.proof)
        gate=system.discovery.current_gate(request.signal.signal_id,proof=runtime.proof)
        disposition=system.discovery.current_disposition(request.lead.lead_id,proof=runtime.proof)
        for ordinal in range(2,5):
            gate=system.discovery.decide_gate(replace(exact_gate_request(),decision_id=GateDecisionId.new(),decision_ordinal=ordinal,
                previous_decision_id=gate.request.decision_id,idempotency_key=f'gate-interrupt-{ordinal}'),proof=runtime.proof)
            disposition=system.discovery.record_lead_disposition(replace(exact_initial_disposition(),decision_id=LeadDispositionDecisionId.new(),
                decision_ordinal=ordinal,previous_decision_id=disposition.request.decision_id,gate_decision_id=gate.request.decision_id,
                idempotency_key=f'disposition-interrupt-{ordinal}'),proof=runtime.proof)
        root=system._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0];conn=root._connection
        current=discovery_current_root_ids(conn)
        ids=tuple(row[0] for table,keys in current.items() for key in keys for row in conn.execute('SELECT authority_event_id FROM '+table+' WHERE decision_id=?',(key,)))
        create_native_source_checkpoint(root,system.objects,event_ids=ids,proof=runtime.proof,dev_rebuild=True)
        tables=('ledger_events','authority_commands','discovery_gate_decisions','lead_disposition_decisions','authority_audit_events')
        before={table:tuple(tuple(row) for row in conn.execute('SELECT * FROM '+table)) for table in tables}
        original=root._event_from_row;calls=0
        def interrupted(row):
            nonlocal calls
            calls+=1
            if calls==2:raise failure
            return original(row)
        monkeypatch.setattr(root,'_event_from_row',interrupted)
        with pytest.raises(type(failure)):prune_superseded_discovery(root,system.objects,proof=runtime.proof,dev_rebuild=True)
        assert not conn.in_transaction
        assert before=={table:tuple(tuple(row) for row in conn.execute('SELECT * FROM '+table)) for table in tables}
        assert conn.execute('PRAGMA foreign_key_check').fetchall()==[]
        assert conn.execute("SELECT count(*) FROM sqlite_temp_schema WHERE name LIKE '_checkpoint_%'").fetchone()[0]==0
        require_checkpoint_schema(conn)


def test_selected_ack_payload_keeps_typed_admission_and_initial_activation(tmp_path):
    from .test_increment10_private_serving import _context,_delivery,_close
    from .authority_helpers import proof
    from newsroom.authority.native_current_rebuild import publication_event_roots,copy_selected_native_store
    from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store
    from newsroom.authority._projection_store import _ProjectionAuthorityStore
    context=_context(tmp_path);delivery=_delivery(tmp_path,context)
    try:
        request=dict(story_receipt=context[-2],candidate_port=context[1],proof=proof())
        attempt,_=delivery.begin(context[-1],**request)
        delivery.apply(attempt,publication_receipt=context[-1],applied_at='2026-07-16T11:00:00Z',**request)
        observed=delivery.observe(attempt,publication_receipt=context[-1],observed_at='2026-07-16T11:30:00Z',**request)
        evidence=delivery.record(observed,attempt,expected_version=0,proof=proof())
        root=context[3].objects._GovernedObjects__hydrate.__self__._store
        root._current_state_only=True
        refs={'story_event_id':context[-2].event_id,'publication_event_id':context[-1].event_id,
            'delivery_attempt_event_id':attempt.event_id,'delivery_evidence_event_id':evidence.event_id}
        ids=publication_event_roots(root._connection,refs)
        with sqlite3.connect(tmp_path/'ack-selected.sqlite3',isolation_level=None) as c:
            c.row_factory=sqlite3.Row;initialise_empty_checkpoint_store(c)
            copy_selected_native_store(root,c,roots=ids,dev_rebuild=True)
            for event_id in refs.values():
                payload=c.execute('SELECT p.* FROM ledger_events e JOIN authority_payloads p USING(payload_id) WHERE e.event_id=?',(event_id,)).fetchone()
                original=root._connection.execute('SELECT * FROM object_admission_versions WHERE admission_id=? AND lifecycle_version=1',(payload['object_admission_id'],)).fetchone()
                retained=c.execute('SELECT * FROM object_admission_versions WHERE admission_id=? AND lifecycle_version=1',(payload['object_admission_id'],)).fetchone()
                assert retained is not None and tuple(retained)==tuple(original)
                _ProjectionAuthorityStore._validate_object_admission_payload_record(c,payload)
                assert c.execute('SELECT 1 FROM ledger_events WHERE event_id=?',(retained['event_id'],)).fetchone()
            assert c.execute('PRAGMA foreign_key_check').fetchall()==[]
    finally:_close(context,delivery)


def test_selected_older_projection_validation_keeps_exact_and_previous_versions(tmp_path,monkeypatch):
    from .projection_b1_helpers import open_projection_system,proof,FAMILY_ID
    from .test_projection_b3_authority import _register,_create,_validate
    from newsroom.authority._projection_store import _ProjectionAuthorityStore
    from newsroom.authority.native_current_rebuild import copy_selected_native_store
    from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store
    monkeypatch.setattr(_ProjectionAuthorityStore,'_current_state_only',True)
    origin=tmp_path/'validation-original.sqlite3';target=tmp_path/'validation-selected.sqlite3'
    with open_projection_system(origin) as system:
        _register(system);created=_create(system,'typed-validation-create')
        _validate(system,created,'typed-validation-2')
        root=system.projections._NativeProjections__validation.__self__._store
        generation=root.projection_generation(created.generation_id)
        selected=_validate(system,generation,'typed-validation-3')
        current=root.projection_generation(created.generation_id)
        _validate(system,current,'typed-validation-4')
        current=root.projection_generation(created.generation_id)
        assert selected.lifecycle_version==3 and current.lifecycle_version==4
        namespace=root._connection.execute("SELECT idempotency_namespace FROM authority_commands WHERE idempotency_key='typed-validation-3'").fetchone()[0]
        with sqlite3.connect(target,isolation_level=None) as c:
            c.row_factory=sqlite3.Row;initialise_empty_checkpoint_store(c)
            copy_selected_native_store(root,c,roots={'projection_generation_validations':((selected.validation_digest,),)},dev_rebuild=True)
            assert [tuple(row) for row in c.execute('SELECT lifecycle_version FROM projection_generation_versions WHERE generation_id=? AND lifecycle_version IN(2,3) ORDER BY lifecycle_version',(str(created.generation_id),))]==[(2,),(3,)]
            assert c.execute('PRAGMA foreign_key_check').fetchall()==[]
    target.chmod(0o600)
    with open_projection_system(target) as system:
        restored=system.projections._NativeProjections__validation.__self__._store
        assert restored.projection_generation_validation_for_key(namespace,'typed-validation-3')==selected
        assert restored.projection_generation(created.generation_id)==current
