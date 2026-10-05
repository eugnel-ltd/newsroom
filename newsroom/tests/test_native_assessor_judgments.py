"""Current codec consumes authenticated closed judgments; no live provider."""
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
import json
import sqlite3

import pytest
from newsroom.authority import ObjectAdmissionRequest
from newsroom.authority.canonical import digest_bytes, canonical_json_bytes
from newsroom.control_plane.model_usage import ModelUsageService
from newsroom.control_plane.native_assessor import AutonomousNativeEvidenceAssessor
from newsroom.control_plane.native_assessor_judgments import NativeAssessorJudgments, JudgedAssessment
from newsroom.control_plane.native_evidence import PublicationRightsAssessment, rights_eligibility_digest
from newsroom.control_plane.evidence import EvidencePackage
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.typesafe_judgment import TypesafeJudgment, judgment_policy
from newsroom.tests.test_native_runtime import _args

BODY = '當局現時推出新政策。\n政策適用於住戶。'


def _source(body, admission, source_id="HK-fixture"):
    role = SimpleNamespace(role=SimpleNamespace(value='ORIGINATING_AUTHORITY'), purpose='Own public policy',
        canonical_value=lambda: {'role':'ORIGINATING_AUTHORITY','purpose':'Own public policy'})
    rights = PublicationRightsAssessment.create(decision='PERMITTED', permitted_use='PUBLICATION_EVIDENCE',
        policy_digest=digest_bytes(b'policy'), evidence_digest=digest_bytes(b'rights'))
    source = SimpleNamespace(unit=SimpleNamespace(source_id=source_id, authority=SimpleNamespace(definition_id='definition-fixture')),
        source_version=SimpleNamespace(canonical_digest=digest_bytes(b'version'),request=SimpleNamespace(roles=(role,))),
        rights=rights,dependency=SimpleNamespace(record_id='dependency-fixture',dependency_status='RESOLVED',evidential_origin_id='origin-fixture',originating_report_id='report-fixture'))
    acquired=SimpleNamespace(body=body,body_digest=digest_bytes(body),receipt_digest=str(admission),
        currentness_basis='AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT',text_only=True,licence_attribution='Fixture attribution',exclusion_signals=(),
        transport_evidence_digest=digest_bytes(b'transport'),canonical_url='https://example.invalid/policy',publisher='當局',responsible_body='當局',source_type='PRIMARY_OFFICIAL',
        publication_time='2026-10-04T00:00:00.000000Z',source_updated_time='2026-10-04T00:00:00.000000Z',retrieval_time='2026-10-04T00:01:00.000000Z',geography='HK',language='zh-Hant-HK')
    acquired.rights_eligibility_digest=rights_eligibility_digest(rights,body_digest=acquired.body_digest,transport_digest=acquired.transport_evidence_digest,exclusion_signals=(),text_only=True)
    return source, acquired


@contextmanager
def _case(tmp_path, monkeypatch, *, answer_change=None, source_id="HK-fixture"):
    args=_args(tmp_path,monkeypatch);usage=ModelUsageService(str(tmp_path/'usage.sqlite3'));calls=[]
    with open_native_runtime(**args) as runtime:
        admission=runtime.authority.objects.admit(ObjectAdmissionRequest('evidence.source','chinese-source'),BODY.encode(),proof=runtime.proof).admission
        source,acquired=_source(BODY.encode(),admission.admission_id,source_id)
        base=EvidencePackage(candidate_id='candidate-fixture',hypothesis_id='hypothesis-fixture',signal_ids=('signal-fixture',),lead_ids=('lead-fixture',),source_ids=(source_id,),observation_digests=(digest_bytes(BODY.encode()),),passages=(BODY,))
        candidate=SimpleNamespace(candidate_id=base.candidate_id,version_id='candidate-version',canonical_bytes=canonical_json_bytes({'candidate':'fixture'}),governing_manifest=SimpleNamespace(canonical_digest=digest_bytes(b'hypothesis'),canonical_bytes=canonical_json_bytes({'hypothesis':'fixture'})))
        def transport(request,**_kw):
            value=json.loads(request.data);calls.append(value);answers={}
            for key,q in value['questions'].items():
                choices=q['criteria']
                if key=='S1L1':choice='MATERIAL'
                elif key=='S1L2':choice='SUPPORTING'
                elif key=='headline':choice='S1L1'
                elif key.endswith(':announced_event'):choice='YES'
                elif key=='S1L1:LAW_RIGHT_STATUS_POLICY':choice='YES'
                elif key.endswith(':change_kind'):choice='PUBLIC_POLICY'
                elif key.endswith('_source_lookup_key'):choice=key.split(':')[0]
                elif ':LAW_RIGHT_STATUS_POLICY:' in key:choice=next(c for c in choices if c!='NONE')
                else:choice='NO' if 'NO'in choices else 'NONE'
                answers[key]={'type':'choice','choice':choice,'confidence':1,'probabilities':{c:int(c==choice)for c in choices}}
            if answer_change is not None:
                answer_change(answers)
            raw=json.dumps({'model':'jev-1.13.0','answers':answers,'usage':{'input_tokens':100,'output_tokens':30}},ensure_ascii=False).encode()
            return 200,request.full_url,raw
        @contextmanager
        def fence(binding,proof):
            assert binding['content_digest']==base.digest
            yield
        service=TypesafeJudgment(usage=usage,objects=runtime.authority.objects,
            policy=judgment_policy(evidence_digest=digest_bytes(b'qualified fixture'),qualified=True),api_key=lambda:'fixture-not-live',source_fence=fence,transport=transport,implementation_worktree_clean=True,clock=lambda:datetime(2026,10,4,tzinfo=UTC))
        consumer=NativeAssessorJudgments(judgments=service,proof=runtime.proof,scope_for=lambda *_:{'coverage':'COMPLETE','newness':'KNOWN_CHANGE','current_scope':{'revision_digest':source.source_version.canonical_digest},'prior_scope':{'revision_digest':digest_bytes(b'prior')}})
        yield consumer,service,candidate,base,source,acquired,usage,calls


def test_missing_qualified_localisation_preserves_closed_judgments_and_requests_fallback(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch) as (consumer,service,candidate,base,source,acquired,usage,calls):
        result=consumer.assess(candidate,base,(source,),(acquired,))
        from newsroom.control_plane.native_assessor_judgments import JudgmentFallback
        assert type(result) is JudgmentFallback
        assert result.reason == 'QUALIFIED_LOCALISATION_REQUIRED'
        assert len(result.references) == 2
        assert consumer.assess(candidate,base,(source,),(acquired,)) == result
        assert len(calls)==2
        with sqlite3.connect(usage.path) as db:
            assert db.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]==2
            assert db.execute('SELECT count(*) FROM model_invocation_terminals').fetchone()[0]==2


@pytest.mark.parametrize('newness,reason',[('UNKNOWN','NEWNESS_BASELINE_UNKNOWN'),('KNOWN_UNCHANGED','EXISTING_REASONING_NEWNESS_REQUIRED')])
def test_unknown_newness_does_not_invent_negative_decision(tmp_path,monkeypatch,newness,reason):
    with _case(tmp_path,monkeypatch) as (consumer,_service,candidate,base,source,acquired,usage,calls):
        consumer.scope_for=lambda *_:{'coverage':'COMPLETE','newness':newness,'current_scope':{},'prior_scope':None}
        result=consumer.assess(candidate,base,(source,),(acquired,))
        assert result.reason==reason
        assert not result.references and not calls
        with sqlite3.connect(usage.path) as db:
            assert db.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]==0


def test_tampered_judgment_reference_is_rejected_by_actual_accounted_reader(tmp_path,monkeypatch):
    from newsroom.control_plane.typesafe_judgment import TypesafeJudgmentError
    with _case(tmp_path,monkeypatch) as (consumer,service,candidate,base,source,acquired,_usage,calls):
        evaluate=service.evaluate
        def tampered(**inputs):
            reference=evaluate(**inputs)
            return replace(reference,raw_admission_id=reference.receipt_admission_id)
        monkeypatch.setattr(service,'evaluate',tampered)
        with pytest.raises(TypesafeJudgmentError,match='TYPESAFE_REPLAY_BINDING_HOLD'):
            consumer.assess(candidate,base,(source,),(acquired,))
        assert len(calls)==1


def test_source_body_binding_change_denies_before_any_judgment(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch) as (consumer,_service,candidate,base,source,acquired,_usage,calls):
        acquired.body=b'changed'
        with pytest.raises(ValueError,match='acquired source bytes differ'):
            consumer.assess(candidate,base,(source,),(acquired,))
        assert not calls


def test_fresh_hook_and_separate_intent_fallback_preserve_old_gate_and_cached_only(tmp_path,monkeypatch):
    from newsroom.control_plane.native_evidence import NativeEvidenceHold
    from newsroom.control_plane.native_assessor_judgments import JudgmentFallback
    with _case(tmp_path,monkeypatch) as (_consumer,_service,candidate,base,source,acquired,_usage,_calls):
        entered=[]
        judgments=SimpleNamespace(get_decision_ref=lambda *_:None,assess=lambda *_:entered.append('fresh') or JudgmentFallback('fixture'))
        def legacy(_prompt):
            raise RuntimeError('legacy automatic fallback')
        assessor=AutonomousNativeEvidenceAssessor(legacy,judgments=judgments)
        before=[]
        with pytest.raises(RuntimeError,match='legacy automatic fallback'):
            assessor.assess_with_boundary(candidate,base,(source,),(acquired,),before_dispatch=lambda:before.append('once'))
        assert entered==['fresh'] and before==['once']
        assessor._usage=SimpleNamespace(retained_assessments=lambda *_:None)
        with pytest.raises(NativeEvidenceHold,match='ASSESSOR_REVALIDATION_UNRESOLVED_HOLD'):
            assessor(candidate,base,(source,),(acquired,))
        assert entered==['fresh','fresh']
        assessor._usage=SimpleNamespace(retained_assessments=lambda *_:())
        with pytest.raises(NativeEvidenceHold,match='ASSESSOR_REVALIDATION_CACHE_MISSING_HOLD'):
            assessor.assess_with_boundary(candidate,base,(source,),(acquired,),before_dispatch=None,cached_only=True)
        assert entered==['fresh','fresh']


def test_current_rights_hold_denies_before_judgment_call(tmp_path,monkeypatch):
    from newsroom.control_plane.native_evidence import NativeEvidenceHold
    with _case(tmp_path,monkeypatch) as (consumer,_service,candidate,base,source,acquired,_usage,calls):
        source.rights=PublicationRightsAssessment.create(decision='HOLD',permitted_use='PUBLICATION_EVIDENCE',
            policy_digest=digest_bytes(b'policy'),evidence_digest=digest_bytes(b'withdrawn'))
        assessor=AutonomousNativeEvidenceAssessor(lambda _:pytest.fail('legacy dispatch'),judgments=consumer)
        with pytest.raises(NativeEvidenceHold,match='SOURCE_POLICY_FACTS_HOLD'):
            assessor(candidate,base,(source,),(acquired,))
        assert not calls


def _localiser(consumer,service,usage,candidate,base,calls):
    from newsroom.control_plane.native_claim_localisation import NativeClaimLocaliser, localisation_policy
    from newsroom.control_plane.writer import WriterCliExecution
    scope=dict(candidate_id=candidate.candidate_id,hypothesis_digest=candidate.governing_manifest.canonical_digest,evidence_package_digest=base.digest,proof=consumer.proof)
    def runner(prompt):
        value=json.loads(prompt);calls.append(value)
        assert 'candidate_id' not in value and 'source_binding' not in value
        return WriterCliExecution(canonical_json_bytes({'renderings':{
            'S1L1':{'rendered_assertion_zh_hant_hk_fragments':['當局現時推出的是新政策。'],'factual_localisations':[],'quotation_source_keys':[]},
            'S1L2':{'rendered_assertion_zh_hant_hk_fragments':['住戶屬於此政策的適用對象。'],'factual_localisations':[],'quotation_source_keys':[]}}}).decode(),
            {'usage_basis':'PROVIDER_REPORTED','input_tokens':100,'output_tokens':30,'cached_read_tokens':0,'cached_write_tokens':0,'reasoning_tokens':0,'context_tokens':100,'total_tokens':130})
    localiser=NativeClaimLocaliser(usage=usage,objects=service.objects,
        policy=localisation_policy(evidence_digest=digest_bytes(b'qualified-localisation-fixture'),qualified=True),source_fence=service.fence,runner=runner,implementation_worktree_clean=True,clock=lambda:datetime(2026,10,4,tzinfo=UTC))
    consumer.localise=lambda request:localiser.localise(request,**scope)
    consumer.read_localisation=lambda reference,request:localiser.read_localisation(reference,request,**scope)
    return localiser,scope


def test_fresh_actual_consumer_uses_qualified_rendering_and_current_validator(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch) as (consumer,service,candidate,base,source,acquired,usage,calls):
        local_calls=[]
        _localiser(consumer,service,usage,candidate,base,local_calls)
        assessor=AutonomousNativeEvidenceAssessor(lambda _:pytest.fail('no legacy dispatch'),judgments=consumer)
        assessment=assessor(candidate,base,(source,),(acquired,))
        assert [c.claim for c in assessment.governed_claims]==BODY.splitlines()
        assert assessment.governed_claims[1].claim_role=='CONTEXT'
        assert len(assessment.qualification_evidence)==1
        assert assessor(candidate,base,(source,),(acquired,))==assessment
        assert len(calls)==2 and len(local_calls)==1
        with sqlite3.connect(usage.path) as db:
            assert db.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]==3
            terminals=[json.loads(r[0])for r in db.execute('SELECT record_json FROM model_invocation_terminals')]
            assert all(r['usage_status']=='REPORTED'for r in terminals)
        result=consumer.assess(candidate,base,(source,),(acquired,))
        record=json.loads(result.decision_record)
        assert record['source_binding']['evidence_package_digest']==base.digest
        assert len(record['judgments'])==2
        assert record['render_provenance']['mode']=='QUALIFIED_LOCALISATION'


def test_real_localisation_reference_tampering_denies_current_consumer(tmp_path,monkeypatch):
    from newsroom.control_plane.native_claim_localisation import LocalisationHold
    with _case(tmp_path,monkeypatch) as (consumer,service,candidate,base,source,acquired,usage,_calls):
        local_calls=[]
        _localiser(consumer,service,usage,candidate,base,local_calls)
        localise=consumer.localise
        def swapped(state):
            reference=localise(state)
            return replace(reference,raw_admission_id=reference.receipt_admission_id)
        consumer.localise=swapped
        with pytest.raises(LocalisationHold,match='LOCALISATION_REPLAY_BINDING_HOLD'):
            consumer.assess(candidate,base,(source,),(acquired,))
        assert len(local_calls)==1


def test_stop_propagates_before_any_stage_and_partial_source_falls_back(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch) as (consumer,_service,candidate,base,source,acquired,_usage,calls):
        consumer.scope_for=lambda *_:{'coverage':'PARTIAL'}
        assert consumer.assess(candidate,base,(source,),(acquired,)).reason=='SOURCE_COVERAGE_UNPROVEN'
        assert not calls
        consumer.scope_for=lambda *_:{'coverage':'COMPLETE','newness':'KNOWN_CHANGE','current_scope':{},'prior_scope':{}}
        consumer.require_current=lambda:(_ for _ in ()).throw(InterruptedError('stop fixture'))
        with pytest.raises(InterruptedError,match='stop fixture'):
            consumer.assess(candidate,base,(source,),(acquired,))
        assert not calls


@pytest.mark.parametrize('kind,reason',[('role','UNCERTAIN_SOURCE_ROLE'),('qualification','UNCERTAIN_QUALIFICATION'),('witness','QUALIFICATION_WITNESS_MISSING')])
def test_accounted_uncertainty_or_missing_witness_never_becomes_false_selection(tmp_path,monkeypatch,kind,reason):
    def change(answers):
        key='S1L1' if kind=='role' else 'S1L1:LAW_RIGHT_STATUS_POLICY' if kind=='qualification' else 'S1L1:LAW_RIGHT_STATUS_POLICY:new_state_source_lookup_key'
        if key not in answers:return
        choice='NONE' if kind=='witness' else 'UNCERTAIN'
        answers[key]['choice']=choice
        answers[key]['probabilities']={c:int(c==choice)for c in answers[key]['probabilities']}
    with _case(tmp_path,monkeypatch,answer_change=change) as (consumer,_service,candidate,base,source,acquired,_usage,calls):
        result=consumer.assess(candidate,base,(source,),(acquired,))
        assert result.reason==reason
        assert len(calls)==(1 if kind=='role' else 2)


def test_combined_current_decision_reader_checks_cas_refs_binding_and_replay(tmp_path,monkeypatch):
    from newsroom.control_plane.typesafe_judgment import TypesafeJudgmentError
    with _case(tmp_path,monkeypatch) as (consumer,service,candidate,base,source,acquired,usage,calls):
        local_calls=[]
        _localiser(consumer,service,usage,candidate,base,local_calls)
        result=consumer.assess(candidate,base,(source,),(acquired,))
        assert consumer.read(result.decision_admission_id,candidate,base,(source,),(acquired,))==result
        assert len(calls)==2 and len(local_calls)==1
        row=json.loads(result.decision_record)
        row['judgments'][0]['raw_admission_id']=row['judgments'][0]['receipt_admission_id']
        bad=service.objects.admit(ObjectAdmissionRequest('evidence.record','forged-decision'),canonical_json_bytes(row),proof=consumer.proof).admission.admission_id
        with pytest.raises(TypesafeJudgmentError,match='TYPESAFE_REPLAY_BINDING_HOLD'):
            consumer.read(bad,candidate,base,(source,),(acquired,))
        consumer.scope_for=lambda *_:{'coverage':'COMPLETE','newness':'KNOWN_CHANGE','current_scope':{'revision_digest':digest_bytes(b'other')},'prior_scope':{}}
        with pytest.raises(TypesafeJudgmentError):
            consumer.read(result.decision_admission_id,candidate,base,(source,),(acquired,))
        assert len(calls)==2 and len(local_calls)==1


def test_real_outer_native_usage_gate_replays_new_decision_after_reopen(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from newsroom.tests.test_native_assessor import _usage
    from newsroom.control_plane.native_assessor import NativeAssessmentUsage
    with _case(tmp_path,monkeypatch) as (consumer,service,candidate,base,source,acquired,usage,calls):
        _same_store,outer=_usage(tmp_path,monkeypatch)
        local_calls=[]
        _localiser(consumer,service,usage,candidate,base,local_calls)
        assessor=AutonomousNativeEvidenceAssessor(lambda _:pytest.fail('no legacy dispatch'),usage=outer,dispatch_fence=nullcontext,judgments=consumer)
        assert outer.retained_assessments(candidate,base)==()
        first=assessor(candidate,base,(source,),(acquired,))
        assert assessor(candidate,base,(source,),(acquired,))==first
        assert assessor.assess_with_boundary(candidate,base,(source,),(acquired,),before_dispatch=None,cached_only=True)==first
        assert len(calls)==2 and len(local_calls)==1
        with sqlite3.connect(usage.path) as db:
            allocation_rows=db.execute('SELECT record_json FROM model_invocation_allocations ORDER BY invocation_id').fetchall()
            terminal_rows=db.execute('SELECT record_json FROM model_invocation_terminals ORDER BY invocation_id').fetchall()
        args=_args(tmp_path,monkeypatch)
    with open_native_runtime(**args) as runtime:
        service.objects=runtime.authority.objects
        consumer.proof=runtime.proof
        reopened=ModelUsageService(usage.path)
        _localiser(consumer,service,reopened,candidate,base,local_calls)
        outer=NativeAssessmentUsage(reopened,outer._policy,clock=outer._clock)
        assessor=AutonomousNativeEvidenceAssessor(lambda _:pytest.fail('no reopened legacy dispatch'),usage=outer,dispatch_fence=nullcontext,judgments=consumer)
        assert assessor(candidate,base,(source,),(acquired,))==first
        assert len(calls)==2 and len(local_calls)==1
        with sqlite3.connect(usage.path) as db:
            assert db.execute('SELECT record_json FROM model_invocation_allocations ORDER BY invocation_id').fetchall()==allocation_rows
            assert db.execute('SELECT record_json FROM model_invocation_terminals ORDER BY invocation_id').fetchall()==terminal_rows


def test_completed_new_intent_replay_preserves_original_unknown_allocation(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from newsroom.tests.test_native_assessor import _usage
    with _case(tmp_path,monkeypatch) as (consumer,service,candidate,base,source,acquired,usage,calls):
        _,outer=_usage(tmp_path,monkeypatch)
        local_calls=[]
        _localiser(consumer,service,usage,candidate,base,local_calls)
        assessor=AutonomousNativeEvidenceAssessor(lambda _:pytest.fail('no legacy retry'),usage=outer,dispatch_fence=nullcontext,judgments=consumer)
        first=assessor(candidate,base,(source,),(acquired,))
        original=outer.begin(candidate,base,'original unknown source-bound request')
        outer.mark_dispatch(original)
        assert outer.retained_assessments(candidate,base) is None
        with sqlite3.connect(usage.path) as db:
            rows=db.execute('SELECT record_json FROM model_invocation_allocations ORDER BY invocation_id').fetchall()
            observations=db.execute('SELECT record_json FROM model_transport_observations ORDER BY observation_digest').fetchall()
        assert assessor(candidate,base,(source,),(acquired,))==first
        assert len(calls)==2 and len(local_calls)==1
        assert outer.retained_assessments(candidate,base) is None
        assert usage.terminal(original.invocation_id) is None
        with sqlite3.connect(usage.path) as db:
            assert db.execute('SELECT record_json FROM model_invocation_allocations ORDER BY invocation_id').fetchall()==rows
            assert db.execute('SELECT record_json FROM model_transport_observations ORDER BY observation_digest').fetchall()==observations


def test_source_definition_change_during_judgment_preserves_usage_and_never_localises(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch) as (consumer,service,candidate,base,source,acquired,usage,calls):
        current={'version':'source-version-1'}
        identity={'source_id':source.unit.source_id,'definition_id':source.unit.authority.definition_id,'version_id':current['version']}
        scope=consumer.scope_for(candidate,base,(source,),(acquired,))
        consumer.scope_for=lambda *_:{**scope,'source_currentness':[identity]}
        original=service.transport
        def changed(request,**kwargs):
            assert 'source_currentness' not in json.loads(request.data)
            response=original(request,**kwargs)
            current['version']='source-version-2'
            return response
        @contextmanager
        def fence(binding,proof):
            if binding['source_currentness'][0]['version_id']!=current['version']:
                raise InterruptedError('source definition changed')
            yield
        service.fence=fence;service.transport=changed
        consumer.localise=lambda _:pytest.fail('no localiser after current source changed')
        consumer.read_localisation=lambda *_:pytest.fail('no rendering reader')
        with pytest.raises(InterruptedError,match='source definition changed'):
            consumer.assess(candidate,base,(source,),(acquired,))
        assert len(calls)==1
        with sqlite3.connect(usage.path) as db:
            allocations=db.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]
            terminals=[json.loads(row[0])for row in db.execute('SELECT record_json FROM model_invocation_terminals')]
        assert allocations==1 and len(terminals)==1
        assert terminals[0]['usage_status']=='REPORTED'
        assert terminals[0]['components']['total_tokens']==130


def test_first_separate_qualified_intent_with_original_unknown_reopens_without_retry(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from newsroom.tests.test_native_assessor import _usage
    from newsroom.control_plane.native_assessor import NativeAssessmentUsage
    with _case(tmp_path,monkeypatch) as (consumer,service,candidate,base,source,acquired,usage,calls):
        _,outer=_usage(tmp_path,monkeypatch)
        original=outer.begin(candidate,base,'original unknown source-bound request')
        outer.mark_dispatch(original)
        assert outer.retained_assessments(candidate,base) is None
        with sqlite3.connect(usage.path) as db:
            old_alloc=db.execute('SELECT record_json FROM model_invocation_allocations WHERE invocation_id=?',(original.invocation_id,)).fetchone()[0]
            old_transport=db.execute('SELECT record_json FROM model_transport_observations WHERE invocation_id=?',(original.invocation_id,)).fetchall()
        local_calls=[]
        _localiser(consumer,service,usage,candidate,base,local_calls)
        assessor=AutonomousNativeEvidenceAssessor(lambda _:pytest.fail('old intent must never dispatch'),usage=outer,dispatch_fence=nullcontext,judgments=consumer)
        first=assessor(candidate,base,(source,),(acquired,))
        assert assessor(candidate,base,(source,),(acquired,))==first
        args=_args(tmp_path,monkeypatch)
    with open_native_runtime(**args) as runtime:
        service.objects=runtime.authority.objects
        consumer.proof=runtime.proof
        _localiser(consumer,service,usage,candidate,base,local_calls)
        outer=NativeAssessmentUsage(ModelUsageService(usage.path),outer._policy,clock=outer._clock)
        assessor=AutonomousNativeEvidenceAssessor(lambda _:pytest.fail('no reopened old intent dispatch'),usage=outer,dispatch_fence=nullcontext,judgments=consumer)
        assert assessor(candidate,base,(source,),(acquired,))==first
        assert len(calls)==2 and len(local_calls)==1
        assert outer.retained_assessments(candidate,base) is None
        assert usage.terminal(original.invocation_id) is None
        with sqlite3.connect(usage.path) as db:
            assert db.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]==4
            assert db.execute('SELECT record_json FROM model_invocation_allocations WHERE invocation_id=?',(original.invocation_id,)).fetchone()[0]==old_alloc
            assert db.execute('SELECT record_json FROM model_transport_observations WHERE invocation_id=?',(original.invocation_id,)).fetchall()==old_transport


def test_new_intent_fallback_still_denies_original_unknown_without_legacy_dispatch(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from newsroom.tests.test_native_assessor import _usage
    from newsroom.control_plane.native_evidence import NativeEvidenceHold
    with _case(tmp_path,monkeypatch) as (consumer,_service,candidate,base,source,acquired,usage,calls):
        _,outer=_usage(tmp_path,monkeypatch)
        original=outer.begin(candidate,base,'original unknown source-bound request')
        outer.mark_dispatch(original)
        consumer.scope_for=lambda *_:{'coverage':'COMPLETE','newness':'UNKNOWN','current_scope':{},'prior_scope':None}
        assessor=AutonomousNativeEvidenceAssessor(lambda _:pytest.fail('no old intent dispatch after fallback'),usage=outer,dispatch_fence=nullcontext,judgments=consumer)
        with pytest.raises(NativeEvidenceHold,match='ASSESSOR_REVALIDATION_UNRESOLVED_HOLD'):
            assessor(candidate,base,(source,),(acquired,))
        assert not calls and usage.terminal(original.invocation_id) is None
        with sqlite3.connect(usage.path) as db:
            assert db.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]==1


def _first_publication(consumer,source,acquired):
    source.unit.authority.definition_version_id='definition-version-fixture'
    source.unit.revision_digest=digest_bytes(b'current-source-revision')
    acquired.canonical_url='https://www.gov.uk/guidance/fixture-public-announcement'
    evidence={'source_id':source.unit.source_id,'definition_id':source.unit.authority.definition_id,
        'definition_version_id':source.unit.authority.definition_version_id,'source_revision_digest':source.unit.revision_digest,
        'acquisition_receipt_digest':acquired.receipt_digest,'first_published_at':acquired.publication_time}
    scope={'coverage':'COMPLETE','newness':'SOURCE_DECLARED_FIRST_PUBLICATION','prior_scope':None,
        'current_scope':{'sources':[{'source_id':source.unit.source_id,'published_at':acquired.publication_time,'body':acquired.body.decode()}]},
        'first_publication':[evidence]}
    consumer.scope_for=lambda *_:scope
    return scope


def test_source_declared_first_publication_has_distinct_bound_intent_and_exact_replay(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch,source_id='UK-03') as (consumer,service,candidate,base,source,acquired,usage,calls):
        scope=_first_publication(consumer,source,acquired)
        local_calls=[]
        _localiser(consumer,service,usage,candidate,base,local_calls)
        first=consumer.assess(candidate,base,(source,),(acquired,))
        assert type(first)is JudgedAssessment
        record=json.loads(first.decision_record)
        assert record['source_binding']['newness']=='SOURCE_DECLARED_FIRST_PUBLICATION'
        assert record['source_binding']['prior_scope'] is None
        assert record['source_binding']['first_publication']==scope['first_publication']
        assert consumer.get_decision_ref(candidate,base,(source,),(acquired,))==first.decision_admission_id
        assert consumer.read(first.decision_admission_id,candidate,base,(source,),(acquired,))==first
        assert len(calls)==2 and len(local_calls)==1
        assert 'newly announced' in calls[1]['questions']['S1L1:announced_event']['instructions']
        assessed=AutonomousNativeEvidenceAssessor._validated_execution(first.execution,candidate,base,(source,),(acquired,))
        assert len(assessed.governed_claims)==2


@pytest.mark.parametrize('choice',['NO','UNCERTAIN'])
def test_first_publication_without_affirmed_announcement_is_reasoning_fallback(tmp_path,monkeypatch,choice):
    def change(answers):
        key='S1L1:announced_event'
        if key in answers:
            answers[key]['choice']=choice
            answers[key]['probabilities']={c:int(c==choice)for c in answers[key]['probabilities']}
    with _case(tmp_path,monkeypatch,source_id='UK-03',answer_change=change) as (consumer,_service,candidate,base,source,acquired,_usage,calls):
        _first_publication(consumer,source,acquired)
        consumer.localise=lambda _:pytest.fail('no rendering for uncertain/admin/old assertion')
        result=consumer.assess(candidate,base,(source,),(acquired,))
        assert result.reason=='FIRST_PUBLICATION_ANNOUNCEMENT_UNPROVEN'
        assert len(result.references)==2 and len(calls)==2


def test_first_publication_provenance_drift_never_uses_updated_or_retrieved_time(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch,source_id='UK-03') as (consumer,_service,candidate,base,source,acquired,_usage,calls):
        scope=_first_publication(consumer,source,acquired)
        scope['first_publication'][0]['first_published_at']=acquired.retrieval_time
        result=consumer.assess(candidate,base,(source,),(acquired,))
        assert result.reason=='FIRST_PUBLICATION_PROVENANCE_UNPROVEN'
        assert not calls
