"""Current codec consumes authenticated closed judgments; no live provider."""
from contextlib import contextmanager
from dataclasses import replace,asdict
from datetime import UTC, datetime
from types import SimpleNamespace
import json
import sqlite3

import pytest
from newsroom.authority import ObjectAdmissionRequest
from newsroom.authority.canonical import digest_bytes, canonical_json_bytes
from newsroom.control_plane.model_usage import ModelUsageService,InvocationEfficiencyPolicy
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
def _case(tmp_path, monkeypatch, *, answer_change=None, source_id="HK-fixture", max_prompt_bytes=None, body=BODY):
    args=_args(tmp_path,monkeypatch);usage=ModelUsageService(str(tmp_path/'usage.sqlite3'));calls=[]
    with open_native_runtime(**args) as runtime:
        admission=runtime.authority.objects.admit(ObjectAdmissionRequest('evidence.source','chinese-source'),body.encode(),proof=runtime.proof).admission
        source,acquired=_source(body.encode(),admission.admission_id,source_id)
        base=EvidencePackage(candidate_id='candidate-fixture',hypothesis_id='hypothesis-fixture',signal_ids=('signal-fixture',),lead_ids=('lead-fixture',),source_ids=(source_id,),observation_digests=(digest_bytes(body.encode()),),passages=(body,))
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
        policy=judgment_policy(evidence_digest=digest_bytes(b'qualified fixture'),qualified=True)
        if max_prompt_bytes is not None:
            policy=InvocationEfficiencyPolicy.create(**{**asdict(policy),'max_prompt_bytes':max_prompt_bytes})
        service=TypesafeJudgment(usage=usage,objects=runtime.authority.objects,
            policy=policy,api_key=lambda:'fixture-not-live',source_fence=fence,transport=transport,implementation_worktree_clean=True,clock=lambda:datetime(2026,10,4,tzinfo=UTC))
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
        return WriterCliExecution(canonical_json_bytes({'renderings':[
            {'span_id':'S1L1','rendered_assertion_zh_hant_hk_fragments':['當局現時推出的是新政策。'],'factual_localisations':[],'quotation_source_keys':[]},
            {'span_id':'S1L2','rendered_assertion_zh_hant_hk_fragments':['住戶屬於此政策的適用對象。'],'factual_localisations':[],'quotation_source_keys':[]}] }).decode(),
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


def test_lossless_supporting_ranges_keep_full_roles_and_material_qualification(tmp_path,monkeypatch):
    from newsroom.control_plane.native_assessor_spans import build_lossless_source_view
    from newsroom.control_plane.native_claim_localisation import NativeClaimLocaliser,localisation_policy
    from newsroom.control_plane.writer import WriterCliExecution
    material={1:'當局現時推出新政策。',16:'當局現時推出教育新政策。',32:'當局現時推出醫療新政策。'}
    body='\n'.join(material.get(i,f'第{i}項政策適用於住戶。')if i<46 else '行政分隔。'for i in range(1,48))
    def choices(answers):
        for key,answer in answers.items():
            if ':'not in key and key!='headline':
                n=int(key.split('L')[1]);choice='MATERIAL'if n in material else 'SUPPORTING'if n<46 else 'BACKGROUND'
                answer.update(choice=choice,probabilities={x:int(x==choice)for x in answer['probabilities']})
            elif key.endswith(':LAW_RIGHT_STATUS_POLICY'):
                answer.update(choice='YES',probabilities={x:int(x=='YES')for x in answer['probabilities']})
    with _case(tmp_path,monkeypatch,body=body,answer_change=choices)as(consumer,service,candidate,base,source,acquired,usage,calls):
        rendered=[]
        def runner(prompt):
            claims=json.loads(prompt)['claims'];rendered.append(claims)
            assert len(claims)<=32
            assert all(not row['entities']for row in claims.values())
            return WriterCliExecution(canonical_json_bytes({'renderings':[{'span_id':key,
                'rendered_assertion_zh_hant_hk_fragments':[row['text'].replace('適用於住戶','以住戶為適用對象').replace('當局現時推出','當局現時推出的是')],
                'factual_localisations':[],'quotation_source_keys':[]}for key,row in claims.items()]}).decode(),
                {'usage_basis':'PROVIDER_REPORTED','input_tokens':100,'output_tokens':30,'total_tokens':130})
        localiser=NativeClaimLocaliser(usage=usage,objects=service.objects,policy=localisation_policy(evidence_digest=digest_bytes(b'packing fixture'),qualified=True),
            source_fence=service.fence,runner=runner,implementation_worktree_clean=True,clock=lambda:datetime(2026,10,4,tzinfo=UTC))
        scope=dict(candidate_id=candidate.candidate_id,hypothesis_digest=candidate.governing_manifest.canonical_digest,evidence_package_digest=base.digest,proof=consumer.proof)
        consumer.localise=lambda request:localiser.localise(request,**scope)
        consumer.read_localisation=lambda ref,request:localiser.read_localisation(ref,request,**scope)
        result=consumer.assess(candidate,base,(source,),(acquired,))
        assert type(result)is JudgedAssessment,result
        view=build_lossless_source_view(base.passages,base.source_ids)
        wire=json.loads(result.execution.text)['package']
        assert len(calls[0]['questions'])==47 and calls[0]['state']['candidates'].keys()=={s.span_id for s in view.segments}
        assert all(f'S1L{i}:LAW_RIGHT_STATUS_POLICY'in calls[1]['questions']for i in material)
        assert set(calls[1]['state']['witness_inventory'])=={s.span_id for s in view.segments}
        coverage=[]
        for row in rendered[0].values():
            text,_,_=view.resolve_range(row['source_range']);assert text==row['text']
            a=int(row['source_range']['first_span_id'].split('L')[1]);b=int(row['source_range']['last_span_id'].split('L')[1])
            coverage.extend(range(a,b+1))
            if a in material:assert a==b
        assert sorted(coverage)==list(range(1,46))and len(coverage)==len(set(coverage))
        assessment=AutonomousNativeEvidenceAssessor._validated_execution(result.execution,candidate,base,(source,),(acquired,))
        assert len(assessment.governed_claims)<=32
        assert len([c for c in assessment.governed_claims if c.claim_role!='CONTEXT'])==3
        assert len(assessment.qualification_evidence)==3
        with sqlite3.connect(usage.path)as db:before=db.execute('SELECT invocation_id,record_json FROM model_invocation_terminals ORDER BY invocation_id').fetchall()
        assert consumer.read(result.decision_admission_id,candidate,base,(source,),(acquired,))==result
        with sqlite3.connect(usage.path)as db:assert db.execute('SELECT invocation_id,record_json FROM model_invocation_terminals ORDER BY invocation_id').fetchall()==before
        assert len(calls)==2 and len(rendered)==1


def _packing_inputs(passages, choices):
    from newsroom.control_plane.native_assessor_spans import build_lossless_source_view
    view=build_lossless_source_view(tuple(passages),tuple(f'source-{i}'for i in range(len(passages))))
    candidates={s.span_id:{'source_id':s.source_id,'text':view.resolve_range({'first_span_id':s.span_id,'last_span_id':s.span_id})[0],
        'entities':[list(e)for e in s.entities],'rendering_fragment_count':len(s.entities)+1,
        'source_range':{'first_span_id':s.span_id,'last_span_id':s.span_id}}for s in view.segments}
    return view,candidates,{s.span_id:{'choice':choice}for s,choice in zip(view.segments,choices,strict=True)}


@pytest.mark.parametrize('barrier',['The policy may change.','The policy does not change.','If approved, the policy applies.',
    'Alice said: "The policy applies."',"Bob said: 'The policy applies.'",'We provide the service.',
    'The policy won’t change.',"The policy shouldn't change.",'‘The policy applies.’',"Businesses' records are available."])
def test_support_packing_never_hides_modality_or_attribution(barrier):
    from newsroom.control_plane.native_assessor_judgments import _packed_support_candidates
    view,candidates,roles=_packing_inputs((f'First detail.\n{barrier}\nLast detail.',),['SUPPORTING']*3)
    packed=_packed_support_candidates(view,candidates,roles)
    assert list(packed)==list(candidates)
    assert [x['text']for x in packed.values()]==[x['text']for x in candidates.values()]


def test_support_packing_does_not_skip_background_material_or_sources():
    from newsroom.control_plane.native_assessor_judgments import _packed_support_candidates
    view,candidates,roles=_packing_inputs(('First detail.\nBackground.\nMaterial fact.\nSecond detail.','Another source.'),
        ['SUPPORTING','BACKGROUND','MATERIAL','SUPPORTING','SUPPORTING'])
    packed=_packed_support_candidates(view,candidates,roles)
    assert list(packed)==['S1L1','S1L3','S1L4','S2L1']
    assert all(row['source_range']['first_span_id']==row['source_range']['last_span_id']for row in packed.values())


def test_support_packing_observes_entity_fragment_and_constituent_bounds(monkeypatch):
    from newsroom.control_plane.native_assessor_judgments import _packed_support_candidates
    from newsroom.control_plane import native_assessor_references as refs
    view,candidates,roles=_packing_inputs(('x'*2048+'\n'+'y'*2048,),['SUPPORTING']*2)
    assert len(_packed_support_candidates(view,candidates,roles))==2
    view,candidates,roles=_packing_inputs(('x'*4097,),['SUPPORTING'])
    assert _packed_support_candidates(view,candidates,roles)is None
    view,candidates,roles=_packing_inputs(('First.\nSecond.',),['SUPPORTING']*2)
    monkeypatch.setattr(refs,'_claim_entities',lambda text,*_a,**_k:tuple((f'Entity{i}','ORGANISATION')for i in range(66 if '\n'in text else 33)))
    assert len(_packed_support_candidates(view,candidates,roles))==2
    monkeypatch.setattr(refs,'_claim_entities',lambda *_a,**_k:())
    view,candidates,roles=_packing_inputs(('\n'.join('Detail.'for _ in range(33)),),['SUPPORTING']*33)
    packed=_packed_support_candidates(view,candidates,roles)
    assert len(packed)==2 and packed['S1L1']['source_range']['last_span_id']=='S1L32'


def test_packed_dates_and_source_entities_stay_exact_source_fragments():
    from newsroom.control_plane.native_assessor_judgments import _packed_support_candidates
    from newsroom.control_plane.native_assessor_references import _claim_entities
    view,candidates,roles=_packing_inputs(('The service starts on 12 October 2026.\nHome Office provides the service.',),['SUPPORTING']*2)
    packed=_packed_support_candidates(view,candidates,roles)
    assert len(packed)==1
    row=next(iter(packed.values()));text,passage,_=view.resolve_range(row['source_range'])
    assert row['text']==text and '12 October 2026'in text
    assert row['entities']==[list(x)for x in _claim_entities(text,view.passages[passage],policy_version=view.entity_policy_version)]
    assert row['rendering_fragment_count']==len(row['entities'])+1


@pytest.mark.parametrize('geometry',['material','sparse'])
def test_unbounded_material_or_sparse_context_never_truncates_or_dispatches_second_stage(tmp_path,monkeypatch,geometry):
    count=33 if geometry=='material'else 69
    body='\n'.join(f'當局现时推出第{i}項新政策。'for i in range(1,count+1))
    def choose(answers):
        for key,answer in answers.items():
            i=int(key.split('L')[1]);choice='MATERIAL'if geometry=='material'or i==1 else 'BACKGROUND'if i%2==0 else 'SUPPORTING'
            answer.update(choice=choice,probabilities={x:int(x==choice)for x in answer['probabilities']})
    with _case(tmp_path,monkeypatch,body=body,answer_change=choose)as(consumer,_service,candidate,base,source,acquired,usage,calls):
        consumer.localise=lambda _:pytest.fail('unbounded range reached rendering')
        result=consumer.assess(candidate,base,(source,),(acquired,))
        assert result.reason=='MISSING_OR_UNBOUNDED_MATERIAL_CLAIMS'
        assert len(result.references)==len(calls)==1
        assert len(calls[0]['questions'])==count
        with sqlite3.connect(usage.path)as db:assert db.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]==1


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


def test_large_prior_geometry_returns_zero_call_judgment_fallback(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch) as (consumer,service,candidate,base,source,acquired,usage,calls):
        consumer.scope_for=lambda *_:{'coverage':'COMPLETE','newness':'KNOWN_CHANGE',
            'current_scope':{'body':BODY},'prior_scope':{'body':'x'*(service.policy.max_prompt_bytes+1)}}
        result=consumer.assess(candidate,base,(source,),(acquired,))
        assert result.reason=='JUDGMENT_INPUT_BOUND' and result.references==()
        assert not calls
        with sqlite3.connect(usage.path) as db:
            assert db.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]==0


def test_second_stage_geometry_fallback_retains_exact_first_reference_without_second_dispatch(tmp_path,monkeypatch):
    from newsroom.control_plane.typesafe_judgment import MODEL,_json
    with _case(tmp_path,monkeypatch,max_prompt_bytes=4096) as (consumer,service,candidate,base,source,acquired,usage,calls):
        evaluate=service.evaluate;references=[]
        def first_then_bound(**inputs):
            reference=evaluate(**inputs)
            references.append(reference)
            bound=len(_json({'model':MODEL,'state':inputs['state'],'questions':inputs['questions']}))
            assert bound<=service.policy.max_prompt_bytes
            return reference
        monkeypatch.setattr(service,'evaluate',first_then_bound)
        result=consumer.assess(candidate,base,(source,),(acquired,))
        assert result.reason=='JUDGMENT_INPUT_BOUND' and result.references==tuple(references)
        assert len(references)==1 and len(calls)==1
        with sqlite3.connect(usage.path) as db:
            assert db.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]==1
            assert db.execute('SELECT count(*) FROM model_invocation_terminals').fetchone()[0]==1


@pytest.mark.parametrize('fallback', [False, True])
def test_semantic_only_intent_never_dispatches_legacy_reasoning(tmp_path, monkeypatch, fallback):
    from contextlib import nullcontext
    from newsroom.control_plane.native_evidence import NativeEvidenceHold
    with _case(tmp_path,monkeypatch) as (consumer,service,candidate,base,source,acquired,usage,calls):
        local_calls=[]
        _localiser(consumer,service,usage,candidate,base,local_calls)
        if fallback:
            consumer.scope_for=lambda *_:{'coverage':'COMPLETE','newness':'UNKNOWN','current_scope':{},'prior_scope':None}
        assessor=AutonomousNativeEvidenceAssessor(lambda _:pytest.fail('separate semantic purpose may not dispatch legacy'),
            dispatch_fence=nullcontext,judgments=consumer)
        for _ in range(2):
            if fallback:
                with pytest.raises(NativeEvidenceHold,match='SEMANTIC_INTENT_FALLBACK_HOLD'):
                    assessor.assess_with_boundary(candidate,base,(source,),(acquired,),before_dispatch=None,semantic_only=True)
            else:
                result=assessor.assess_with_boundary(candidate,base,(source,),(acquired,),before_dispatch=None,semantic_only=True)
                assert len(result.governed_claims)==2
        assert len(calls)==(0 if fallback else 2)
        assert len(local_calls)==(0 if fallback else 1)


@pytest.mark.parametrize('kind',['missing','uncertain'])
def test_auxiliary_witness_gap_retains_every_claim_and_only_omits_unproven_qualification(tmp_path,monkeypatch,kind):
    def change(answers):
        if 'S1L2'in answers:
            answers['S1L2']['choice']='MATERIAL'
            answers['S1L2']['probabilities']={key:int(key=='MATERIAL')for key in answers['S1L2']['probabilities']}
        key='S1L2:LAW_RIGHT_STATUS_POLICY'
        if key not in answers:return
        choice='UNCERTAIN'if kind=='uncertain'else'YES'
        answers[key]['choice']=choice
        answers[key]['probabilities']={value:int(value==choice)for value in answers[key]['probabilities']}
        if kind=='missing':
            key+=':new_state_source_lookup_key'
            answers[key]['choice']='NONE'
            answers[key]['probabilities']={value:int(value=='NONE')for value in answers[key]['probabilities']}
    with _case(tmp_path,monkeypatch,answer_change=change) as (consumer,service,candidate,base,source,acquired,usage,calls):
        local_calls=[]
        _localiser(consumer,service,usage,candidate,base,local_calls)
        result=consumer.assess(candidate,base,(source,),(acquired,))
        assert type(result)is JudgedAssessment
        assessment=AutonomousNativeEvidenceAssessor._validated_execution(result.execution,candidate,base,(source,),(acquired,))
        assert [claim.claim for claim in assessment.governed_claims]==BODY.splitlines()
        assert [claim.claim_role for claim in assessment.governed_claims]==['HEADLINE','SUBSTANTIVE']
        assert assessment.substantive_new_information==tuple(BODY.splitlines())
        assert len(assessment.qualification_evidence)==1
        assert assessment.qualification_evidence[0].governed_claim_id==assessment.governed_claims[0].claim_id
        assert consumer.read(result.decision_admission_id,candidate,base,(source,),(acquired,))==result
        assert len(calls)==2 and len(local_calls)==1


def test_v2_qualification_questions_supply_all_six_rubrics_and_complete_parent_witnesses(tmp_path,monkeypatch):
    from newsroom.control_plane.native_assessor_judgments import VERSION
    from newsroom.control_plane.qualification_rubrics import RUBRICS
    assert VERSION=='newsroom.native-assessor-judgments.v2'
    with _case(tmp_path,monkeypatch) as (consumer,_service,candidate,base,source,acquired,_usage,calls):
        result=consumer.assess(candidate,base,(source,),(acquired,))
        assert result.reason=='QUALIFIED_LOCALISATION_REQUIRED'
        for test,rubric in RUBRICS.items():
            actual=calls[1]['questions']['S1L1:'+test]['instructions']
            assert actual['definition']==rubric['definition']
            assert actual['requires']==rubric['requires'] and actual['excludes']==rubric['excludes']
            assert 'already-effective' in actual['temporal_and_source_rules']
        assert calls[1]['state']['witness_inventory']['S1L1']['parent_text']==BODY.splitlines()[0]


def test_witness_inventory_is_utf8_exact_and_never_clips_parent_negation_or_dates():
    from newsroom.control_plane.native_assessor_spans import build_lossless_source_view
    from newsroom.control_plane.qualification_rubrics import witness_inventory
    body='政府宣布新政策，但該政策並未生效。\nThe authority announced a deadline, but the report was not confirmed.'
    view=build_lossless_source_view((body,),('fixture',));inventory=witness_inventory(view)
    for parent in inventory.values():
        for value in parent['candidates'].values():
            assert body.encode()[value['start_byte']:value['end_byte']].decode()==value['text']
        assert parent['parent_text']
    assert '並未生效' in inventory['S1L1']['parent_text']
    assert 'not confirmed' in inventory['S1L2']['parent_text']


def test_complete_no_material_source_has_valid_negative_codec_without_renderer(tmp_path,monkeypatch):
    def change(answers):
        for key,value in answers.items():
            if key.startswith('S1L') and ':'not in key:
                value['choice']='BACKGROUND'
                value['probabilities']={option:int(option=='BACKGROUND')for option in value['probabilities']}
    with _case(tmp_path,monkeypatch,answer_change=change) as (consumer,_service,candidate,base,source,acquired,_usage,calls):
        consumer.localise=lambda _:pytest.fail('no renderer needed for zero selected news')
        result=consumer.assess(candidate,base,(source,),(acquired,))
        assert type(result)is JudgedAssessment
        assessment=AutonomousNativeEvidenceAssessor._validated_execution(result.execution,candidate,base,(source,),(acquired,))
        assert not assessment.substantive_new_information and not assessment.qualification_evidence
        assert len(calls)==1
        assert json.loads(result.decision_record)['render_provenance']['mode']=='NO_MATERIAL_CLAIMS'


def test_v2_fallback_carries_exact_authenticated_inputs_and_no_proof_or_private_state(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch) as (consumer,service,candidate,base,source,acquired,_usage,_calls):
        result=consumer.assess(candidate,base,(source,),(acquired,))
        assert len(result.details['judgment_inputs'])==len(result.references)==2
        for reference,inputs in zip(result.references,result.details['judgment_inputs'],strict=True):
            assert 'proof'not in inputs and inputs['source_binding']==result.details['source_binding']
            assert service.read(reference,**inputs,proof=consumer.proof)['answers']
        assert result.details['failed_questions']
        assert result.details['witness_inventory']


def test_v2_validation_mapper_reauthenticates_prior_decision_without_new_dispatch(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch) as (consumer,service,candidate,base,source,acquired,usage,calls):
        local_calls=[]
        _localiser(consumer,service,usage,candidate,base,local_calls)
        result=consumer.assess(candidate,base,(source,),(acquired,))
        fallback=consumer.validation_failure(result,candidate,base,(source,),(acquired,),'qualification relation failed')
        assert fallback.reason=='MATERIALISATION_VALIDATION_FAILED'
        assert len(fallback.references)==2 and len(fallback.details['judgment_inputs'])==2
        assert fallback.details['prior_decision_admission_id']==str(result.decision_admission_id)
        assert len(calls)==2 and len(local_calls)==1
        assert 'judgment_inputs'not in json.loads(result.decision_record)
        with pytest.raises(Exception):
            consumer.validation_failure(replace(result,decision_admission_id=result.judgment_inputs[0]['candidate_id']),candidate,base,(source,),(acquired,),'forged ref')


def test_overbound_complete_witness_clause_is_visible_and_escalates_without_clipping():
    from newsroom.control_plane.native_assessor_spans import build_lossless_source_view
    from newsroom.control_plane.qualification_rubrics import witness_inventory
    text='The authority introduced a new policy '+('x'*300)+'.'
    view=build_lossless_source_view((text,),('fixture',));inventory=witness_inventory(view)
    assert inventory['S1L1']['parent_text']==text
    assert inventory['S1L1']['uncovered_clause_ids']
    assert not inventory['S1L1']['candidates']


@pytest.mark.parametrize('name,test,expected,context',[
    ('policy-versus-safety','LAW_RIGHT_STATUS_POLICY',True,'new licensing policy'),
    ('conditional-risk','SAFETY_OR_PUBLIC_HEALTH',False,'conditional possible harm'),
    ('consultation','OFFICIAL_ACTION_OR_DEADLINE',True,'proposed rule is not thereby enacted'),
    ('announced-future','LAW_RIGHT_STATUS_POLICY',None,'not proof the rule is already in force'),
    ('auxiliary','LAW_RIGHT_STATUS_POLICY',True,'standing guidance'),
])
def test_promoted_rubrics_state_distinct_positive_negative_and_temporal_contrasts(name,test,expected,context):
    from newsroom.control_plane.qualification_rubrics import question,RUBRICS
    q=question('S1L1',test,{'material_relation_span':{}})
    assert context in (str(q['instructions'])+' '+str(q['criteria'])) or name in {'policy-versus-safety','auxiliary'}
    assert q['instructions']['definition']==RUBRICS[test]['definition']
    assert set(q['criteria'])=={'YES','NO','UNCERTAIN'}
    assert 'first-observation' in q['instructions']['temporal_and_source_rules']
    # Labels are fixture expectations, not a fabricated provider judgement.
    assert expected in (True,False,None)


def test_clause_selector_resolves_only_exact_inventory_keys_with_parent_context(tmp_path,monkeypatch):
    def change(answers):
        for key,value in answers.items():
            if key.startswith('S1L1:LAW_RIGHT_STATUS_POLICY:')and key.endswith('_source_lookup_key'):
                selected=next(option for option in value['probabilities']if option.endswith('C1'))
                value['choice']=selected
                value['probabilities']={option:int(option==selected)for option in value['probabilities']}
    with _case(tmp_path,monkeypatch,answer_change=change) as (consumer,service,candidate,base,source,acquired,usage,calls):
        local_calls=[];_localiser(consumer,service,usage,candidate,base,local_calls)
        result=consumer.assess(candidate,base,(source,),(acquired,))
        assessment=AutonomousNativeEvidenceAssessor._validated_execution(result.execution,candidate,base,(source,),(acquired,))
        witness=dict(assessment.qualification_evidence[0].test_evidence)['material_relation_span']
        assert witness==BODY.splitlines()[0].removesuffix('。')
        assert witness in assessment.governed_claims[0].claim
        assert len(calls)==2 and len(local_calls)==1
        assert all('proof'not in row for row in result.judgment_inputs)


def test_v2_old_v1_decision_key_is_not_relabelled_as_new_purpose(tmp_path,monkeypatch):
    from newsroom.authority.canonical import digest_canonical
    with _case(tmp_path,monkeypatch) as (consumer,service,candidate,base,source,acquired,usage,_calls):
        local_calls=[];_localiser(consumer,service,usage,candidate,base,local_calls)
        result=consumer.assess(candidate,base,(source,),(acquired,))
        binding=json.loads(result.decision_record)['source_binding']
        old_key='judgment-decision:'+digest_canonical(['newsroom.native-assessor-judgments.v1',binding])
        assert consumer._decision_key(binding)!=old_key
        assert service.objects.committed_admission(ObjectAdmissionRequest('evidence.record',old_key),proof=consumer.proof) is None
        assert consumer.get_decision_ref(candidate,base,(source,),(acquired,))==result.decision_admission_id


def test_unknown_complete_source_fallback_retains_source_bound_exception_context_without_paid_judgment(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch) as (consumer,_service,candidate,base,source,acquired,_usage,calls):
        consumer.scope_for=lambda *_:{'coverage':'COMPLETE','newness':'UNKNOWN','prior_scope':None,'current_scope':{}}
        result=consumer.assess(candidate,base,(source,),(acquired,))
        assert result.reason=='NEWNESS_BASELINE_UNKNOWN' and not calls
        assert result.details['source_binding']['content_digest']==base.digest
        assert result.details['state']['sources']==[{'source_id':base.source_ids[0],'text':BODY}]
        assert result.details['judgment_inputs']==[] and result.references==()


def test_selected_fragment_cannot_erase_negative_parent_context(tmp_path,monkeypatch):
    body='The authority introduced a new policy, but it was not confirmed.\nReaders may enquire.'
    def change(answers):
        for key,value in answers.items():
            if key.startswith('S1L1:LAW_RIGHT_STATUS_POLICY:')and key.endswith('_source_lookup_key'):
                value['choice']='S1L1C1'
                value['probabilities']={option:int(option=='S1L1C1')for option in value['probabilities']}
    with _case(tmp_path,monkeypatch,body=body,answer_change=change) as (consumer,_service,candidate,base,source,acquired,_usage,calls):
        consumer.localise=lambda _:pytest.fail('parent contradiction must escalate before rendering')
        result=consumer.assess(candidate,base,(source,),(acquired,))
        assert result.reason=='PARENT_MODALITY_REQUIRES_REASONING'
        assert 'not confirmed' in result.details['state']['witness_inventory']['S1L1']['parent_text']
        assert result.details['failed_questions'][0]['reason']=='PARENT_NEGATION_OR_MODALITY'
        assert len(calls)==2


@pytest.mark.parametrize("span_id,text", [('S1L7', 'For businesses, digital proof of age offers a way to verify a customer’s age with greater confidence than checking a physical document alone. '), ('S1L22', 'This is now possible thanks to the UK’s thriving digital verification sector, worth over £2 billion a year, and the UK’s DVS Trust Framework, which techUK worked closely with government to help develop. '), ('S1L27', 'Separate non-statutory guidance has also been published on GOV.UK to help businesses and digital verification providers understand what’s involved in adopting the technology, including requirements for using certified services. '), ('S1L32', 'Digital proof of age is separate from the Government’s digital driving licence and GOV.UK Wallet, although in time this will be one of the ways people can prove their age digitally to buy alcohol. ')])
def test_retained_source_internal_apostrophes_are_not_quotation_boundaries(span_id,text):
    from newsroom.control_plane.native_assessor_judgments import _packed_support_candidates
    view,candidates,roles=_packing_inputs((text+"\nThe service is available.",),["SUPPORTING"]*2)
    packed=_packed_support_candidates(view,candidates,roles)
    assert len(packed)==1,span_id
    row=next(iter(packed.values()))
    assert row["text"]==view.resolve_range(row["source_range"])[0]
    assert text in row["text"]
