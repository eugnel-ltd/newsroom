"""Bounded diagnostic inventory is observational, never a retry or provider port."""
import json
import sqlite3
from pathlib import Path

import pytest

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from scripts.diagnostics.native_assessment_inventory import inventory


def seed(path, *, corrupt=False):
    c=sqlite3.connect(path)
    c.executescript('''CREATE TABLE native_current_sources(revision_id TEXT PRIMARY KEY,land_seq INTEGER,content_json TEXT,content_digest TEXT);
    CREATE TABLE native_current_heads(revision_id TEXT PRIMARY KEY,ordinal INTEGER,state_json TEXT,state_digest TEXT,pair_digest TEXT);
    CREATE TABLE model_work_envelopes(envelope_id TEXT PRIMARY KEY,record_json TEXT);
    CREATE TABLE model_invocation_allocations(invocation_id TEXT PRIMARY KEY,envelope_id TEXT,record_json TEXT);
    CREATE INDEX alloc_envelope ON model_invocation_allocations(envelope_id);
    CREATE TABLE model_invocation_terminals(invocation_id TEXT PRIMARY KEY,record_json TEXT);
    CREATE TABLE model_invocation_context_manifests(context_manifest_digest TEXT PRIMARY KEY,record_json TEXT);
    CREATE TABLE ledger(seq INTEGER PRIMARY KEY,kind TEXT,payload_json TEXT,payload_digest TEXT);
    CREATE INDEX result_lookup ON ledger(kind,json_extract(payload_json,'$.invocation_id'));''')
    for n in (2,1,3,4):
        revision=f'revision-{n}'
        source={'revision_id':revision,'units':[{'source_id':'HK-04','effective_revision':{'first_observed_at':'2026-10-04T00:00:00Z'},'authority':{'revision_id':revision}}]}
        state={'revision_id':revision,'ordinal':1,'stage':'EVIDENCE_HOLD','facts':{'reason':'NO_NEW_INFORMATION','assessment_contract_version':'newsroom.native-evidence-assessor.v23+consumer.v1'}}
        raw=canonical_json_bytes(source).decode();head=canonical_json_bytes(state).decode()
        c.execute('INSERT INTO native_current_sources VALUES(?,?,?,?)',(revision,n,raw,'sha256:'+'0'*64 if corrupt and n==1 else digest_bytes(raw.encode())))
        c.execute('INSERT INTO native_current_heads VALUES(?,?,?,?,?)',(revision,1,head,digest_canonical({'state':state,'pair_digest':None}),None))
    c.commit();c.close()


def test_unknown_usage_group_is_deterministic_readonly_and_bounded(tmp_path,monkeypatch):
    path=tmp_path/'journal.sqlite3';seed(path);before=path.read_bytes()
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:pytest.fail('provider/source call'))
    result=inventory(path,limit=3,seconds=2)
    assert result['selected']==3 and result['truncated'] is True
    group=result['groups'][0]
    assert group['disposition']=='UNKNOWN_OR_NO_RETRY' and group['count']==3
    assert group['revision_ids']==['revision-1','revision-2','revision-3']
    assert len(group['representatives'])<=3
    assert result==inventory(path,limit=3,seconds=2)
    assert path.read_bytes()==before
    assert not Path(str(path)+'-wal').exists()
    assert 'body' not in json.dumps(result) and 'result_text' not in json.dumps(result)


def test_corrupt_current_hash_is_unknown_not_legitimate_hold(tmp_path):
    path=tmp_path/'journal.sqlite3';seed(path,corrupt=True)
    result=inventory(path,limit=10,seconds=2)
    group=next(g for g in result['groups']if 'revision-1'in g['revision_ids'])
    rep=next(r for r in group['representatives']if r['revision_id']=='revision-1')
    assert rep['availability']['current_source']=='UNKNOWN'
    assert rep['availability']['proof']=='INVALID_OR_MISSING'
    assert group['disposition']=='UNKNOWN_OR_NO_RETRY'
    assert all(g['disposition']!='LEGIT_HOLD'for g in result['groups'])


def test_missing_database_does_not_create_file(tmp_path):
    path=tmp_path/'absent.sqlite3'
    with pytest.raises((FileNotFoundError,sqlite3.OperationalError)):
        inventory(path,limit=5,seconds=1)
    assert not path.exists()


def test_real_usage_retained_negative_is_reported_not_editorial_approval(tmp_path,monkeypatch):
    from newsroom.tests.test_native_assessor import _usage,_empty_reference_result
    from newsroom.tests.assessor_fixture_support import candidate_fixture
    from newsroom.tests.test_increment10_editorial import _ready_package
    from newsroom.increment10.evidence import _base_package
    from newsroom.control_plane.native_assessor import NativeAssessmentExecution
    from scripts.diagnostics.native_assessment_inventory import _assessment
    from newsroom.control_plane.model_usage import _envelope_from_record
    conn,_,candidate=candidate_fixture(tmp_path)
    service,usage=_usage(tmp_path,monkeypatch);base=_base_package(_ready_package(candidate)[1])
    allocation=usage.begin(candidate,base,'exact diagnostic fixture')
    execution=NativeAssessmentExecution(canonical_json_bytes(_empty_reference_result()).decode(),{
        'usage_basis':'PROVIDER_REPORTED','input_tokens':1,'output_tokens':1,
        'cached_read_tokens':0,'cached_write_tokens':0,'reasoning_tokens':0,'context_tokens':1,'total_tokens':2})
    dispatch=usage.mark_dispatch(allocation);usage.retain_result(allocation,execution,dispatch_at=dispatch)
    usage.complete(allocation,outcome='ASSESSOR_VALIDATION_FAILED',execution=execution,
        provider_dispatched=True,dispatch_at=dispatch,failure_class='ASSESSMENT_VALIDATION_FAILED')
    path=Path(service.path)
    # The real service's constructor precedes connect() in this old helper;
    # mirror its existing operational expression indexes, never live migration.
    with sqlite3.connect(path) as c:
        for kind in ('NATIVE_ASSESSMENT_RESULT','NATIVE_ASSESSMENT_MATERIALISATION'):
            c.execute("CREATE INDEX fixture_"+kind+" ON ledger(kind,json_extract(payload_json,'$.invocation_id')) WHERE kind='"+kind+"'")
    before=path.read_bytes()
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:pytest.fail('provider/source call'))
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as c:
        c.execute('PRAGMA query_only=ON')
        envelope=_envelope_from_record(json.loads(c.execute('SELECT record_json FROM model_work_envelopes').fetchone()[0]))
        available,proof=_assessment(c,envelope,{},lambda:None)
        assert available['accounting']=='REPORTED_SETTLED' and available['retained_raw']=='HASH_VERIFIED'
        # Reference producer's missing deterministic materialisation cannot turn
        # a raw self-declared negative into cached or legitimate editorial proof.
        assert available['materialisation']=='MISSING' and proof['cached_candidate'] is False
        assert proof['model_decision_reported']=='NO_NEW_INFORMATION' and proof['editorial_acceptance']=='UNASSESSED'
    assert path.read_bytes()==before
    conn.close()


def test_representative_deadline_keeps_all_thin_group_ids(tmp_path,monkeypatch):
    import scripts.diagnostics.native_assessment_inventory as module
    path=tmp_path/'journal.sqlite3';seed(path)
    before=path.read_bytes()
    def bounded_failure(*a,**k):raise TimeoutError('injected proof deadline')
    monkeypatch.setattr(module,'checked_json',bounded_failure)
    result=inventory(path,limit=10,seconds=2)
    assert result['deadline_exceeded'] and result['groups_complete']
    assert result['selected']==4 and result['groups'][0]['revision_ids']==['revision-1','revision-2','revision-3','revision-4']
    assert result['groups'][0]['disposition']=='UNKNOWN_OR_NO_RETRY'
    assert path.read_bytes()==before


def test_no_index_retained_lookup_denies_instead_of_history_scan(tmp_path):
    from scripts.diagnostics.native_assessment_inventory import _retained
    path=tmp_path/'journal.sqlite3';seed(path)
    with sqlite3.connect(path) as c:
        c.execute('DROP INDEX result_lookup');c.commit()
        before=path.read_bytes()
        with pytest.raises(ValueError,match='no index'):
            _retained(c,'NATIVE_ASSESSMENT_RESULT','not-present')
    assert path.read_bytes()==before


def test_source_proof_count_is_not_full_assessment_proof_count(tmp_path):
    path=tmp_path/'journal.sqlite3';seed(path)
    group=inventory(path,limit=10,seconds=2)['groups'][0]
    assert group['source_representatives_verified']==3
    assert group['proof_representatives_checked']==0
    assert group['unverified_revisions']==1


def test_healthy_unassessed_rows_are_not_reported_as_retry_blockers(tmp_path):
    path=tmp_path/'journal.sqlite3';seed(path)
    with sqlite3.connect(path) as c:
        for revision,raw in c.execute('SELECT revision_id,state_json FROM native_current_heads').fetchall():
            value=json.loads(raw);value['stage']='SAME_STATE_ASSOCIATED';value['facts']={}
            c.execute('UPDATE native_current_heads SET state_json=?,state_digest=? WHERE revision_id=?',
                (canonical_json_bytes(value).decode(),digest_canonical({'state':value,'pair_digest':None}),revision))
    group=inventory(path,limit=10,seconds=2)['groups'][0]
    assert group['disposition']=='NOT_ASSESSOR_WORK'
    assert all(r['availability']['accounting']=='NOT_APPLICABLE'for r in group['representatives'])


def test_large_reason_is_bounded_and_digest_keeps_distinct_groups(tmp_path):
    path=tmp_path/'journal.sqlite3';seed(path)
    with sqlite3.connect(path) as c:
        for revision,raw in c.execute('SELECT revision_id,state_json FROM native_current_heads').fetchall():
            value=json.loads(raw);value['facts']['reason']='x'*2048+revision
            c.execute('UPDATE native_current_heads SET state_json=?,state_digest=? WHERE revision_id=?',
                (canonical_json_bytes(value).decode(),digest_canonical({'state':value,'pair_digest':None}),revision))
    groups=inventory(path,limit=10,seconds=2)['groups']
    assert len(groups)==4 and len({g['reason_digest']for g in groups})==4
    assert all(len(g['reason'])==512 and g['reason_truncated']for g in groups)
