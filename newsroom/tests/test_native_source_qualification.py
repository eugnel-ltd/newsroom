"""One separately bound exception leaf; no live provider or old-purpose retry."""
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
import json
import sqlite3

import pytest
from newsroom.authority import ObjectAdmissionRequest, HydrationRequest
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane.native_source_qualification import NativeSourceQualifier, qualification_policy, QualificationHold
from newsroom.control_plane.native_assessor import NativeAssessmentExecution
from newsroom.control_plane.model_usage import ModelUsageService
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.native_assessor_spans import build_lossless_source_view
from newsroom.tests.test_native_runtime import _args

NOW=datetime(2026,10,5,tzinfo=UTC)
BODY='The authority published a complete official record.'
WIRE={'package':{'select_new_information':False,'governed_claims':[], 'qualification_evidence':[],
    'selection_rationale':'The complete record contains no new qualifying material.',
    'geography':[],'categories':[],'explicit_exclusions':[]}}


def _state():
    view=build_lossless_source_view((BODY,),('UK-03',))
    return {'source_binding':{'content_digest':digest_bytes(BODY.encode()),'candidate_id':'candidate','hypothesis_digest':digest_bytes(b'hypothesis'),
        'evidence_package_digest':digest_bytes(BODY.encode()),'candidate_version_id':'private-candidate-id'},
        'source_view':{'passages':[BODY],'source_ids':['UK-03'],
            'sources':[{'source_id':'UK-03','segments':[{**s.request_record(),'rendering_fragment_count':len(s.entities)+1}for s in view.segments]}]},
        'issue':{'reason':'UNCERTAIN_QUALIFICATION','failed_questions':['S1L1:LAW_RIGHT_STATUS_POLICY']},
        'judgments':[{'answers':{'S1L1':{'choice':'UNCERTAIN'}},'outcome':'TYPESAFE_COMPLETE'}]}


@contextmanager
def _case(tmp_path,monkeypatch,*,failure=None):
    usage=ModelUsageService(str(tmp_path/'usage.sqlite3'));calls=[]
    @contextmanager
    def fence(binding,proof):
        assert binding['content_digest']==digest_bytes(BODY.encode())
        yield
    def runner(prompt):
        calls.append(prompt)
        assert 'private-candidate-id' not in prompt
        if failure=='timeout':raise TimeoutError('fixture')
        return NativeAssessmentExecution('bad JSON'if failure=='malformed'else canonical_json_bytes(WIRE).decode(),
            {'usage_basis':'PROVIDER_REPORTED','input_tokens':40,'output_tokens':10,'total_tokens':50})
    with open_native_runtime(**_args(tmp_path,monkeypatch))as runtime:
        qualifier=NativeSourceQualifier(usage=usage,objects=runtime.authority.objects,
            policy=qualification_policy(evidence_digest=digest_bytes(b'qualified fixture'),qualified=True),
            source_fence=fence,judgments=SimpleNamespace(read=lambda *_a,**_k:pytest.fail('direct leaf needs no fake judgment')),
            runner=runner,implementation_worktree_clean=True,clock=lambda:NOW)
        yield qualifier,usage,runtime.proof,calls


def test_exception_leaf_has_one_original_accounted_purpose_and_exact_cas_replay(tmp_path,monkeypatch):
    state=_state();ids=dict(candidate_id='candidate',hypothesis_digest=digest_bytes(b'hypothesis'),evidence_package_digest=state['source_binding']['content_digest'])
    with _case(tmp_path,monkeypatch)as(qualifier,usage,proof,calls):
        ref=qualifier.qualify(state,proof=proof,**ids)
        assert qualifier.qualify(state,proof=proof,**ids)==ref
        result=qualifier.read_qualification(ref,state,proof=proof,**ids)
        assert result['materialisation']['materialised_text']
        assert len(calls)==1
        with sqlite3.connect(usage.path)as db:
            assert db.execute('SELECT route FROM model_invocation_allocations').fetchall()==[('NATIVE_SOURCE_QUALIFICATION',)]
            assert db.execute('SELECT usage_status,outcome FROM model_invocation_terminals').fetchall()==[('REPORTED','QUALIFICATION_COMPLETE')]


@pytest.mark.parametrize('failure',['timeout','malformed'])
def test_exception_failure_never_redispatches_or_invents_usage(tmp_path,monkeypatch,failure):
    state=_state();ids=dict(candidate_id='candidate',hypothesis_digest=digest_bytes(b'hypothesis'),evidence_package_digest=state['source_binding']['content_digest'])
    with _case(tmp_path,monkeypatch,failure=failure)as(qualifier,usage,proof,calls):
        for _ in range(2):
            with pytest.raises((TimeoutError,ValueError,QualificationHold)):
                qualifier.qualify(state,proof=proof,**ids)
        assert len(calls)==1
        with sqlite3.connect(usage.path)as db:
            rows=db.execute('SELECT usage_status,outcome FROM model_invocation_terminals').fetchall()
        assert rows==[('ESTIMATED'if failure=='timeout'else'REPORTED','QUALIFICATION_FAILED')]
        if failure=='malformed':
            with sqlite3.connect(usage.path)as db:
                invocation=db.execute('SELECT invocation_id FROM model_invocation_allocations').fetchone()[0]
            admitted=qualifier.objects.committed_admission(ObjectAdmissionRequest('evidence.record','source-qualification-raw:'+invocation),proof=proof)
            assert qualifier.objects.rehydrate(HydrationRequest(admitted.admission.admission_id,'evidence.record'),proof=proof).data==b'bad JSON'


def test_forged_public_segments_are_rejected_before_exception_allocation(tmp_path,monkeypatch):
    state=_state();state['source_view']['sources'][0]['segments'][0]['text']='forged source'
    ids=dict(candidate_id='candidate',hypothesis_digest=digest_bytes(b'hypothesis'),evidence_package_digest=state['source_binding']['content_digest'])
    with _case(tmp_path,monkeypatch)as(qualifier,usage,proof,calls):
        with pytest.raises(QualificationHold,match='SOURCE_SEGMENTS'):
            qualifier.qualify(state,proof=proof,**ids)
        assert not calls
        with sqlite3.connect(usage.path)as db:
            assert db.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]==0


def test_source_bound_typed_fallback_uses_one_separate_exception_with_original_unknown_intact(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from newsroom.tests.test_native_assessor_judgments import _case as typed_case
    from newsroom.tests.test_native_assessor import _usage
    from newsroom.control_plane.native_assessor import AutonomousNativeEvidenceAssessor
    from newsroom.control_plane.native_assessor_judgments import JudgmentFallback
    from newsroom.control_plane.native_source_qualification import NativeSourceQualifier
    with typed_case(tmp_path,monkeypatch)as(consumer,service,candidate,base,source,acquired,usage,jev_calls):
        consumer.scope_for=lambda *_:{'coverage':'COMPLETE','newness':'UNKNOWN',
            'current_scope':{'sources':[{'source_id':source.unit.source_id,'body':base.passages[0]}]},'prior_scope':None}
        fallback=consumer.assess(candidate,base,(source,),(acquired,))
        assert type(fallback)is JudgmentFallback and fallback.details['source_binding']['content_digest']==base.digest
        _,outer=_usage(tmp_path,monkeypatch)
        original=outer.begin(candidate,base,'original unknown request');outer.mark_dispatch(original)
        calls=[]
        def runner(prompt):
            calls.append(prompt)
            return NativeAssessmentExecution(canonical_json_bytes(WIRE).decode(),
                {'usage_basis':'PROVIDER_REPORTED','input_tokens':40,'output_tokens':10,'total_tokens':50})
        qualifier=NativeSourceQualifier(usage=usage,objects=service.objects,
            policy=qualification_policy(evidence_digest=digest_bytes(b'qualified exception fixture'),qualified=True),
            source_fence=service.fence,judgments=service,runner=runner,implementation_worktree_clean=True,clock=lambda:NOW)
        def qualify(c,b,s,a,f):
            return qualifier.assess(c,b,s,a,f,scope=consumer.scope_for(c,b,s,a),proof=consumer.proof)
        assessor=AutonomousNativeEvidenceAssessor(lambda _:pytest.fail('original reasoning may not retry'),
            usage=outer,dispatch_fence=nullcontext,judgments=consumer,qualification=qualify)
        first=assessor.assess_with_boundary(candidate,base,(source,),(acquired,),before_dispatch=None,semantic_only=True)
        assert not first.governed_claims
        assert assessor.assess_with_boundary(candidate,base,(source,),(acquired,),before_dispatch=None,semantic_only=True)==first
        assert len(calls)==1 and not jev_calls
        assert usage.terminal(original.invocation_id)is None and outer.retained_assessments(candidate,base)is None


def _ids(state):
    return {key:state['source_binding'][key]for key in ('candidate_id','hypothesis_digest','evidence_package_digest')}


def _drop_fixture_guards(db,table):
    # Disposable test database only: demonstrate canonical decoder denials.
    for name, in db.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name=?",(table,)).fetchall():
        db.execute('DROP TRIGGER "'+name.replace('"','""')+'"')


@pytest.mark.parametrize('target',['context','terminal','telemetry'])
def test_reported_exception_replay_denies_tampered_accounting_and_context(tmp_path,monkeypatch,target):
    from newsroom.control_plane.model_usage import ModelUsageIntegrityError
    state=_state()
    with _case(tmp_path,monkeypatch)as(qualifier,usage,proof,calls):
        ref=qualifier.qualify(state,proof=proof,**_ids(state))
        with sqlite3.connect(usage.path)as db:
            if target=='context':
                _drop_fixture_guards(db,'model_invocation_context_manifests')
                db.execute("UPDATE model_invocation_context_manifests SET route='wrong-route'")
            elif target=='terminal':
                _drop_fixture_guards(db,'model_invocation_terminals')
                db.execute("UPDATE model_invocation_terminals SET outcome='QUALIFICATION_FAILED' WHERE invocation_id=?",(ref.invocation_id,))
            else:
                _drop_fixture_guards(db,'model_provider_telemetry')
                db.execute("UPDATE model_provider_telemetry SET provider_telemetry_digest=? WHERE invocation_id=?",(digest_bytes(b'wrong telemetry'),ref.invocation_id))
        with pytest.raises((QualificationHold,ModelUsageIntegrityError)):
            qualifier.read_qualification(ref,state,proof=proof,**_ids(state))
        assert len(calls)==1


def test_exception_candidate_hypothesis_and_package_scope_swaps_never_dispatch_again(tmp_path,monkeypatch):
    state=_state()
    with _case(tmp_path,monkeypatch)as(qualifier,_usage,proof,calls):
        ref=qualifier.qualify(state,proof=proof,**_ids(state))
        for field in _ids(state):
            altered={**_ids(state),field:'other-candidate'if field=='candidate_id'else digest_bytes(b'other binding')}
            with pytest.raises((QualificationHold,ValueError)):
                qualifier.read_qualification(ref,state,proof=proof,**altered)
        assert len(calls)==1


@pytest.mark.parametrize('point',['before_allocate','before_transport'])
def test_exception_stop_is_owned_and_does_not_invent_dispatched_zero_usage(tmp_path,monkeypatch,point):
    state=_state()
    with _case(tmp_path,monkeypatch)as(qualifier,usage,proof,calls):
        @contextmanager
        def stop(_binding,_proof):
            raise InterruptedError('owner stop fixture')
            yield
        if point=='before_allocate':
            qualifier.fence=stop
        else:
            allocate=usage.allocate
            def allocated(*args,**kwargs):
                result=allocate(*args,**kwargs)
                qualifier.fence=stop
                return result
            monkeypatch.setattr(usage,'allocate',allocated)
        with pytest.raises(InterruptedError,match='owner stop fixture'):
            qualifier.qualify(state,proof=proof,**_ids(state))
        assert not calls
        with sqlite3.connect(usage.path)as db:
            allocations=db.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]
            terminals=[json.loads(row[0])for row in db.execute('SELECT record_json FROM model_invocation_terminals')]
            transports=db.execute('SELECT count(*) FROM model_transport_observations').fetchone()[0]
        assert transports==0
        if point=='before_allocate':
            assert allocations==0 and not terminals
        else:
            assert allocations==1 and len(terminals)==1
            assert terminals[0]['pre_dispatch_zero_proved']is True and terminals[0]['dispatch_at']is None
            assert terminals[0]['components']['total_tokens']==0


def test_exception_preopened_envelope_resumes_exact_identity_with_later_clock(tmp_path,monkeypatch):
    from datetime import timedelta
    state=_state()
    with _case(tmp_path,monkeypatch)as(qualifier,usage,proof,calls):
        _prompt,_snapshot,envelope,_manifest=qualifier._input(state,**_ids(state))
        usage.open_envelope(envelope)
        qualifier.clock=lambda:NOW+timedelta(seconds=9)
        ref=qualifier.qualify(state,proof=proof,**_ids(state))
        assert qualifier.qualify(state,proof=proof,**_ids(state))==ref and len(calls)==1
        with sqlite3.connect(usage.path)as db:
            rows=db.execute('SELECT envelope_id,admitted_at FROM model_work_envelopes').fetchall()
        assert rows==[(envelope.envelope_id,'2026-10-05T00:00:00.000000Z')]


def test_reported_invalid_exception_raw_stays_retained_and_replay_denied(tmp_path,monkeypatch):
    state=_state()
    with _case(tmp_path,monkeypatch,failure='malformed')as(qualifier,usage,proof,calls):
        with pytest.raises(ValueError):qualifier.qualify(state,proof=proof,**_ids(state))
        with sqlite3.connect(usage.path)as db:
            before=db.execute('SELECT invocation_id,record_json FROM model_invocation_terminals').fetchone()
        raw=qualifier.objects.committed_admission(ObjectAdmissionRequest('evidence.record','source-qualification-raw:'+before[0]),proof=proof)
        assert qualifier.objects.rehydrate(HydrationRequest(raw.admission.admission_id,'evidence.record'),proof=proof).data==b'bad JSON'
        with pytest.raises(QualificationHold):qualifier.qualify(state,proof=proof,**_ids(state))
        with sqlite3.connect(usage.path)as db:
            assert db.execute('SELECT invocation_id,record_json FROM model_invocation_terminals').fetchone()==before
        assert len(calls)==1 and json.loads(before[1])['usage_status']=='REPORTED'


def test_exception_recomputed_current_binding_rejects_same_body_identity_and_scope_swaps(tmp_path,monkeypatch):
    from copy import deepcopy
    from newsroom.tests.test_native_assessor_judgments import _case as typed_case
    with typed_case(tmp_path,monkeypatch)as(consumer,service,candidate,base,source,acquired,usage,_jev_calls):
        consumer.scope_for=lambda *_:{'coverage':'COMPLETE','newness':'UNKNOWN',
            'current_scope':{'sources':[{'source_id':source.unit.source_id,'body':base.passages[0]}]},'prior_scope':None}
        scope=consumer.scope_for(candidate,base,(source,),(acquired,))
        fallback=consumer.assess(candidate,base,(source,),(acquired,))
        calls=[]
        def runner(prompt):
            calls.append(prompt)
            return NativeAssessmentExecution(canonical_json_bytes(WIRE).decode(),
                {'usage_basis':'PROVIDER_REPORTED','input_tokens':40,'output_tokens':10,'total_tokens':50})
        qualifier=NativeSourceQualifier(usage=usage,objects=service.objects,
            policy=qualification_policy(evidence_digest=digest_bytes(b'qualified current fixture'),qualified=True),
            source_fence=service.fence,judgments=service,runner=runner,implementation_worktree_clean=True,clock=lambda:NOW)
        assert qualifier.assess(candidate,base,(source,),(acquired,),fallback,scope=scope,proof=consumer.proof).execution
        monkeypatch.setattr(service,'read',lambda *_args,**_kw:pytest.fail('binding swap must fail before old JEV read'))
        monkeypatch.setattr(qualifier,'qualify',lambda *_args,**_kw:pytest.fail('binding swap must fail before exception allocation'))
        alternatives=[]
        changed=SimpleNamespace(**vars(candidate));changed.version_id='other-version';alternatives.append((changed,scope))
        changed=SimpleNamespace(**vars(candidate));changed.governing_manifest=SimpleNamespace(canonical_digest=digest_bytes(b'other hypothesis'));alternatives.append((changed,scope))
        changed=deepcopy(scope);changed['current_scope']={'body':'same bytes, changed scope'};alternatives.append((candidate,changed))
        changed=deepcopy(scope);changed['prior_scope']={'body':'different prior'};alternatives.append((candidate,changed))
        changed=deepcopy(scope);changed['source_currentness']=[{'definition_version_id':'different-source-version'}];alternatives.append((candidate,changed))
        for current,current_scope in alternatives:
            with pytest.raises(QualificationHold,match='CURRENT_SNAPSHOT'):
                qualifier.assess(current,base,(source,),(acquired,),fallback,scope=current_scope,proof=consumer.proof)
        assert len(calls)==1


def test_public_qualification_v2_prompt_preserves_rubrics_time_and_prior_without_private_refs():
    from newsroom.control_plane.native_source_qualification import _prompt, VERSION, SYSTEM
    from newsroom.control_plane.qualification_rubrics import RUBRICS
    state=_state();binding=state['source_binding']
    binding.update(coverage='COMPLETE',newness='SOURCE_DECLARED_FIRST_PUBLICATION',
        current_scope={'sources':[{'source_id':'UK-03','published_at':'2026-10-02T15:08:14Z','body':BODY}]},
        prior_scope=None,first_publication=[{'source_id':'UK-03','first_published_at':'2026-10-02T15:08:14Z',
            'definition_id':'private-definition','acquisition_receipt_digest':'private-receipt'}])
    state['judgments']=[{'questions':{'q':{'type':'choice','instructions':'Exact full source-supported process rubric.'}},
        'answers':{'q':{'choice':'UNCERTAIN'}},'outcome':'TYPESAFE_COMPLETE'}]
    raw=_prompt(state);value=json.loads(raw)
    assert VERSION=='newsroom.native-source-qualification.v2'
    assert value['qualification_rubrics']==RUBRICS
    assert value['scope']['newness']==binding['newness'] and value['scope']['prior']is None
    assert value['scope']['first_publication']==[{'source_id':'UK-03','first_published_at':'2026-10-02T15:08:14Z'}]
    assert value['judgments']==state['judgments']
    assert all(secret not in raw for secret in ('private-definition','private-receipt','private-candidate-id'))
    assert 'not the retrieval clock' in SYSTEM and 'mandatory obligation' in SYSTEM


def test_prior_public_body_occurs_once_in_qualification_prompt():
    from newsroom.control_plane.native_source_qualification import _prompt
    state=_state();old='An earlier uniquely identifiable public fact.'
    prior={'sources':[{'source_id':'UK-03','body':old,'published_at':'2026-10-01T00:00:00Z'}]}
    state['source_binding'].update(coverage='COMPLETE',newness='KNOWN_CHANGE',prior_scope=prior)
    state['issue']['prior_scope']=prior
    raw=_prompt(state);value=json.loads(raw)
    assert raw.count(old)==1
    assert value['scope']['prior']==prior and 'prior_scope'not in value['unresolved']
    assert state['issue']['prior_scope']==prior  # Local provenance is unchanged.


@contextmanager
def _retained_role_bound_recipe(tmp_path,monkeypatch,*,failure=None,first_publication=False):
    from copy import deepcopy
    from newsroom.tests.test_native_assessor_judgments import _case as typed_case, _first_publication
    from newsroom.control_plane.native_assessor_judgments import JudgmentFallback
    with typed_case(tmp_path,monkeypatch,max_prompt_bytes=6000,source_id='UK-03') as (consumer,service,candidate,base,source,acquired,usage,jev_calls):
        scope={'coverage':'COMPLETE','newness':'KNOWN_CHANGE','prior_scope':{'revision_digest':digest_bytes(b'prior')},
            'current_scope':{'sources':[{'source_id':source.unit.source_id,'body':base.passages[0],
                'published_at':acquired.publication_time,'updated_at':acquired.source_updated_time,'retrieved_at':acquired.retrieval_time}]}}
        if first_publication:
            _first_publication(consumer,source,acquired)
            first=consumer.scope_for(candidate,base,(source,),(acquired,))
            scope.update(newness=first['newness'],prior_scope=first['prior_scope'],first_publication=first['first_publication'])
        consumer.scope_for=lambda *_:scope
        fallback=consumer.assess(candidate,base,(source,),(acquired,))
        assert isinstance(fallback,JudgmentFallback) and fallback.reason=='JUDGMENT_INPUT_BOUND',fallback
        assert fallback.details['failed_questions']==[{'stage':'SELECTED_QUALIFICATION','reason':'INPUT_BOUND'}]
        calls=[]
        def runner(prompt):
            calls.append(prompt)
            if failure=='timeout':raise TimeoutError('fixture')
            return NativeAssessmentExecution(canonical_json_bytes(WIRE).decode(),
                {'usage_basis':'PROVIDER_REPORTED','input_tokens':40,'output_tokens':10,'total_tokens':50})
        qualifier=NativeSourceQualifier(usage=usage,objects=service.objects,
            policy=qualification_policy(evidence_digest=digest_bytes(b'retained-consumer fixture'),qualified=True),
            source_fence=service.fence,judgments=service,runner=runner,implementation_worktree_clean=True,clock=lambda:NOW)
        if failure=='timeout':
            with pytest.raises(TimeoutError):qualifier.assess(candidate,base,(source,),(acquired,),fallback,scope=scope,proof=consumer.proof)
            original=None
        else:
            original=qualifier.assess(candidate,base,(source,),(acquired,),fallback,scope=scope,proof=consumer.proof)
        fresh=SimpleNamespace(**{**vars(acquired),'retrieval_time':'2026-10-05T12:00:00Z','receipt_digest':digest_bytes(b'fresh observation')})
        current=deepcopy(scope);current['current_scope']['sources'][0]['retrieved_at']=fresh.retrieval_time
        if first_publication:current['first_publication'][0]['acquisition_receipt_digest']=fresh.receipt_digest
        yield qualifier,consumer,candidate,base,source,fresh,current,original,usage,calls,jev_calls


def test_known_role_bound_qualification_replays_two_observations_without_model(tmp_path,monkeypatch):
    from newsroom.control_plane.native_source_qualification_replay import read_current_result
    with _retained_role_bound_recipe(tmp_path,monkeypatch,first_publication=True) as (qualifier,consumer,candidate,base,source,fresh,scope,original,usage,calls,jev_calls):
        with sqlite3.connect(usage.path)as db:
            before=db.execute('SELECT invocation_id,record_json FROM model_invocation_terminals ORDER BY invocation_id').fetchall()
        result=read_current_result(qualifier,candidate,base,(source,),(fresh,),scope=scope,proof=consumer.proof)
        assert result==original
        assert len(calls)==len(jev_calls)==1
        with sqlite3.connect(usage.path)as db:
            assert db.execute('SELECT invocation_id,record_json FROM model_invocation_terminals ORDER BY invocation_id').fetchall()==before


@pytest.mark.parametrize('mutation',['body','definition','prior','newness','first-publication','missing'])
def test_known_qualification_reader_denies_current_binding_drift(tmp_path,monkeypatch,mutation):
    from newsroom.control_plane.native_source_qualification_replay import read_current_result
    with _retained_role_bound_recipe(tmp_path,monkeypatch,first_publication=True) as (qualifier,consumer,candidate,base,source,fresh,scope,_original,_usage,calls,jev_calls):
        if mutation=='body':fresh.body=b'changed current source'
        elif mutation=='definition':scope['first_publication'][0]['definition_id']='other-definition'
        elif mutation=='prior':scope['prior_scope']={'body':'unproved prior'}
        elif mutation=='newness':scope['newness']='KNOWN_CHANGE'
        elif mutation=='first-publication':scope['first_publication'][0]['first_published_at']='2026-10-02T00:00:00Z'
        else:candidate=SimpleNamespace(**{**vars(candidate),'candidate_id':'other-candidate'})
        with pytest.raises((QualificationHold,ValueError)):
            read_current_result(qualifier,candidate,base,(source,),(fresh,),scope=scope,proof=consumer.proof)
        assert len(calls)==len(jev_calls)==1


def test_unknown_qualification_reader_never_retries_original(tmp_path,monkeypatch):
    from newsroom.control_plane.native_source_qualification_replay import read_current_result
    with _retained_role_bound_recipe(tmp_path,monkeypatch,failure='timeout') as (qualifier,consumer,candidate,base,source,fresh,scope,_original,_usage,calls,jev_calls):
        with pytest.raises(QualificationHold,match='USAGE_HOLD'):
            read_current_result(qualifier,candidate,base,(source,),(fresh,),scope=scope,proof=consumer.proof)
        assert len(calls)==len(jev_calls)==1


def test_known_qualification_reader_denies_ambiguous_candidate_leaves(tmp_path,monkeypatch):
    from dataclasses import asdict
    from newsroom.control_plane.model_usage import WorkEnvelope,WorkloadClass,InvocationAllocation,_allocation_from_record
    from newsroom.control_plane.native_source_qualification_replay import read_current_result
    with _retained_role_bound_recipe(tmp_path,monkeypatch) as (qualifier,consumer,candidate,base,source,fresh,scope,_original,usage,calls,jev_calls):
        with sqlite3.connect(usage.path)as db:
            record=db.execute("SELECT record_json FROM model_invocation_allocations WHERE route='NATIVE_SOURCE_QUALIFICATION'").fetchone()[0]
        original=_allocation_from_record(json.loads(record))
        envelope=WorkEnvelope.create(cycle_id='another-retained-qualified-intent',workload_class=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
            admitted_at=NOW,admission_decision_id=None,candidate_id=candidate.candidate_id,
            hypothesis_digest=candidate.governing_manifest.canonical_digest,evidence_package_digest=base.digest,
            ingest_id=None,graphiti_attempt_id=None)
        usage.open_envelope(envelope)
        values=asdict(original);values.pop('invocation_id');values.pop('canonical_digest')
        second=InvocationAllocation.create(**{**values,'envelope_id':envelope.envelope_id,'cycle_id':envelope.cycle_id})
        usage.allocate(second,owner_emergency_stop=False)
        with pytest.raises(QualificationHold,match='ABSENT_OR_AMBIGUOUS'):
            read_current_result(qualifier,candidate,base,(source,),(fresh,),scope=scope,proof=consumer.proof)
        assert len(calls)==len(jev_calls)==1


@contextmanager
def _retained_closed_recipe(tmp_path, monkeypatch, recipe):
    from newsroom.tests.test_native_assessor_judgments import (
        _case as typed_case, _first_publication, _localiser,
    )
    from newsroom.control_plane.native_assessor_judgments import JudgmentFallback

    def answers(value):
        for key, answer in value.items():
            if recipe == 'material_bound' and 'MATERIAL' in answer['probabilities']:
                answer.update(choice='MATERIAL', probabilities={c: int(c == 'MATERIAL')
                              for c in answer['probabilities']})
            if ((recipe == 'witness_missing' and key.endswith('_source_lookup_key'))
                    or (recipe == 'announcement' and key.endswith(':announced_event'))):
                choice = 'NONE' if recipe == 'witness_missing' else 'NO'
                answer.update(choice=choice, probabilities={c: int(c == choice)
                              for c in answer['probabilities']})

    kwargs = dict(source_id='UK-03', answer_change=answers)
    if recipe in {'material_bound', 'candidate_bound'}:
        count = 33 if recipe == 'material_bound' else 254
        kwargs['body'] = '\n'.join(['Schools now receive fully funded practical education materials.',
            'Materials are now available to households.'] + ['The provision remains documented.'] * (count - 2))
    if recipe == 'coverage':
        kwargs['body'] = ('The authority confirms a material policy change '
                          + 'for affected residents ' * 16 + '.\nSupporting details apply.')
    with typed_case(tmp_path, monkeypatch, **kwargs) as (
            consumer, service, candidate, base, source, acquired, usage, jev_calls):
        scope = {'coverage': 'COMPLETE', 'newness': 'KNOWN_CHANGE',
                 'prior_scope': {'revision_digest': digest_bytes(b'prior')},
                 'current_scope': {'sources': [{'source_id': source.unit.source_id,
                     'body': base.passages[0], 'published_at': acquired.publication_time,
                     'updated_at': acquired.source_updated_time,
                     'retrieved_at': acquired.retrieval_time}]}}
        if recipe == 'announcement':
            _first_publication(consumer, source, acquired)
            declared = consumer.scope_for(candidate, base, (source,), (acquired,))
            scope.update(newness=declared['newness'], prior_scope=declared['prior_scope'],
                         first_publication=declared['first_publication'])
        consumer.scope_for = lambda *_: scope
        local_calls = []
        if recipe == 'typed_output':
            _localiser(consumer, service, usage, candidate, base, local_calls)
            judged = consumer.assess(candidate, base, (source,), (acquired,))
            fallback = consumer.validation_failure(judged, candidate, base,
                (source,), (acquired,), 'TYPED_OUTPUT_CONTRACT_UNPROVEN')
        else:
            fallback = consumer.assess(candidate, base, (source,), (acquired,))
        assert isinstance(fallback, JudgmentFallback)
        expected = {'coverage': 'QUALIFICATION_WITNESS_COVERAGE_UNPROVEN',
                    'witness_missing': 'QUALIFICATION_WITNESS_MISSING',
                    'announcement': 'FIRST_PUBLICATION_ANNOUNCEMENT_UNPROVEN',
                    'typed_output': 'MATERIALISATION_VALIDATION_FAILED',
                    'material_bound': 'MISSING_OR_UNBOUNDED_MATERIAL_CLAIMS',
                    'candidate_bound': 'CANDIDATE_COVERAGE_LIMIT'}
        assert fallback.reason == expected[recipe]
        calls = []
        wire = WIRE
        if recipe in {'material_bound', 'candidate_bound'}:
            wire = {'package': {'select_new_information': True, 'governed_claims': [
                {'claim_role': role, 'status': 'CONFIRMED_FACT',
                 'source_range': {'first_span_id': span, 'last_span_id': span},
                 'rendered_assertion_zh_hant_hk_fragments': [render],
                 'factual_localisations': [], 'quotation_source_keys': []}
                for role, span, render in [('HEADLINE', 'S1L1', '學校現時獲提供全額資助嘅實用教育教材。'),
                                          ('SUBSTANTIVE', 'S1L2', '教材現時可供家庭使用。')]],
                'qualification_evidence': [{'test': 'HOUSEHOLD_PRACTICAL_EFFECT', 'claim_index': 0,
                    'test_evidence': {'domain': 'EDUCATION', 'event_polarity': 'AFFIRMED',
                        'effect_relation': 'MATERIAL_PRACTICAL_EFFECT',
                        'material_relation_span_source_lookup_key': base.passages[0].split('\n')[0],
                        'practical_effect_source_lookup_key': base.passages[0].split('\n')[0]}}],
                'selection_rationale': 'An affirmed practical education provision.',
                'geography': [], 'categories': [], 'explicit_exclusions': []}}

        def runner(prompt):
            calls.append(prompt)
            return NativeAssessmentExecution(canonical_json_bytes(wire).decode(),
                {'usage_basis': 'PROVIDER_REPORTED', 'input_tokens': 40,
                 'output_tokens': 10, 'total_tokens': 50})

        qualifier = NativeSourceQualifier(usage=usage, objects=service.objects,
            policy=qualification_policy(evidence_digest=digest_bytes(b'closed recipe fixture'),
                                        qualified=True),
            source_fence=service.fence, judgments=service, runner=runner,
            implementation_worktree_clean=True, clock=lambda: NOW)
        original = qualifier.assess(candidate, base, (source,), (acquired,), fallback,
                                    scope=scope, proof=consumer.proof)
        yield qualifier, consumer, candidate, base, source, acquired, scope, original, usage, calls, jev_calls, local_calls


@pytest.mark.parametrize('recipe', ['coverage', 'witness_missing', 'announcement', 'typed_output',
                                    'material_bound', 'candidate_bound'])
def test_known_closed_qualification_recipes_replay_exact_original_without_dispatch(
        tmp_path, monkeypatch, recipe):
    from newsroom.control_plane.native_source_qualification_replay import (
        original_qualification_state, read_current_result,
    )
    with _retained_closed_recipe(tmp_path, monkeypatch, recipe) as (
            qualifier, consumer, candidate, base, source, acquired, scope, original,
            usage, calls, jev_calls, local_calls):
        binding = json.loads(original.decision_record)['source_binding']
        before = (len(calls), len(jev_calls), len(local_calls))
        with sqlite3.connect(usage.path) as db:
            retained = db.execute('SELECT invocation_id,record_json FROM model_invocation_terminals '
                                  'ORDER BY invocation_id').fetchall()
        state = original_qualification_state(qualifier, candidate, base, binding,
                                             proof=consumer.proof)
        prompt, _, _, _ = qualifier._input(state, **_ids(state))
        assert prompt == calls[0]
        if recipe in {'material_bound', 'candidate_bound'}:
            assert json.loads(original.execution.text)['package']['substantive_new_information'] == base.passages[0].split('\n')[:2]
            assert len(jev_calls) == (1 if recipe == 'material_bound' else 0)
        assert read_current_result(qualifier, candidate, base, (source,), (acquired,),
                                   scope=scope, proof=consumer.proof) == original
        assert (len(calls), len(jev_calls), len(local_calls)) == before
        with sqlite3.connect(usage.path) as db:
            assert db.execute('SELECT invocation_id,record_json FROM model_invocation_terminals '
                              'ORDER BY invocation_id').fetchall() == retained


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'swapped_receipt', 'source', 'inventory'])
def test_closed_recipe_replay_denies_missing_ambiguous_or_tampered_prior_evidence(
        tmp_path, monkeypatch, mutation):
    from copy import deepcopy
    from newsroom.authority import ObjectAdmissionId
    from newsroom.control_plane.native_source_qualification import QualificationReference
    from newsroom.control_plane.native_source_qualification_replay import original_qualification_state
    from newsroom.control_plane.typesafe_judgment import TypesafeJudgmentError

    with _retained_closed_recipe(tmp_path, monkeypatch, 'witness_missing') as (
            qualifier, consumer, candidate, base, _source, _acquired, _scope, original,
            usage, calls, jev_calls, local_calls):
        decision = json.loads(original.decision_record)
        binding = deepcopy(decision['source_binding'])
        refs = binding['prior_judgments']
        if mutation == 'missing':
            refs.pop()
        elif mutation == 'duplicate':
            refs[1] = deepcopy(refs[0])
        elif mutation == 'swapped_receipt':
            refs[0]['receipt_admission_id'] = refs[1]['receipt_admission_id']
        elif mutation == 'source':
            binding['current_scope']['sources'][0]['body'] = 'Changed source bytes.'
        else:
            binding['failure_inventory'][0]['question_id'] = 'not-the-original-question'
        before = (len(calls), len(jev_calls), len(local_calls))
        parent = decision['qualification_reference']
        reference = QualificationReference(parent['invocation_id'],
            ObjectAdmissionId.parse(parent['raw_admission_id']),
            ObjectAdmissionId.parse(parent['receipt_admission_id']))
        with pytest.raises((QualificationHold, TypesafeJudgmentError, ValueError)):
            state = original_qualification_state(qualifier, candidate, base, binding,
                                                 proof=consumer.proof)
            qualifier.read_qualification(reference, state, proof=consumer.proof, **_ids(state))
        assert (len(calls), len(jev_calls), len(local_calls)) == before


@pytest.mark.parametrize('recipe',['material_bound','candidate_bound'])
def test_plain_reason_replay_requires_the_original_authenticated_request(tmp_path,monkeypatch,recipe):
    from copy import deepcopy
    from newsroom.authority import ObjectAdmissionId
    from newsroom.control_plane.native_source_qualification import QualificationReference
    from newsroom.control_plane.native_source_qualification_replay import original_qualification_state
    with _retained_closed_recipe(tmp_path,monkeypatch,recipe) as (
            qualifier,consumer,candidate,base,_source,_acquired,_scope,original,
            _usage,calls,jev_calls,local_calls):
        decision=json.loads(original.decision_record);binding=deepcopy(decision['source_binding'])
        binding['failure_inventory']=[{'reason':'NOT_THE_ORIGINAL_REASON'}]
        state=original_qualification_state(qualifier,candidate,base,binding,proof=consumer.proof)
        ref=decision['qualification_reference'];before=(len(calls),len(jev_calls),len(local_calls))
        reference=QualificationReference(ref['invocation_id'],ObjectAdmissionId.parse(ref['raw_admission_id']),
                                        ObjectAdmissionId.parse(ref['receipt_admission_id']))
        with pytest.raises(QualificationHold,match='REPLAY_CONTEXT'):
            qualifier.read_qualification(reference,state,proof=consumer.proof,**_ids(state))
        assert (len(calls),len(jev_calls),len(local_calls))==before
