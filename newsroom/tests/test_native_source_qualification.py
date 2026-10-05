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
