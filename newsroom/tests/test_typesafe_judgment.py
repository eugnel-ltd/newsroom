"""One paid, source-bound semantic batch; no external provider calls."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime

from newsroom.authority.canonical import digest_bytes
from newsroom.control_plane.model_usage import ModelUsageService, ModelUsageIntegrityError
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.tests.test_native_runtime import _args
from newsroom.control_plane.typesafe_judgment import TypesafeJudgment, judgment_policy

NOW = datetime(2026, 10, 4, tzinfo=UTC)


def test_three_typed_questions_have_one_accounted_call_cas_and_exact_replay(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    usage = ModelUsageService(str(tmp_path / 'usage.sqlite3'))
    calls, fences = [], []
    state = {'source': 'The scheme is now open.', 'claim': 'The scheme is open.'}
    binding = {'source_id': 'synthetic', 'revision_id': 'revision-1', 'content_digest': digest_bytes(state['source'].encode())}
    questions = {
        'support': {'type': 'choice', 'instructions': 'Does source support claim?', 'criteria': {'yes': 'Supported', 'no': 'Unsupported'}},
        'present': {'type': 'noul', 'instructions': 'Is the scheme open?'},
        'importance': {'type': 'score', 'instructions': 'What practical significance?', 'criteria': ['Routine restatement', 'Small actionable change', 'Major actionable change']},
    }
    raw = json.dumps({'model': 'jev-1.13.0', 'answers': {
        'support': {'type': 'choice', 'choice': 'yes', 'confidence': 0.8, 'probabilities': {'yes': 0.9, 'no': 0.1}},
        'present': {'type': 'noul', 'noul': 0.95},
        'importance': {'type': 'score', 'score': 1.25, 'legend': {'0': 'Routine restatement', '1': 'Small actionable change', '2': 'Major actionable change'}, 'confidence': 0.5, 'probabilities': {'0': 0, '1': 0.75, '2': 0.25}},
    }, 'usage': {'input_tokens': 5085, 'output_tokens': 1147}}, separators=(',', ':')).encode()

    def transport(request, *, timeout, max_bytes):
        calls.append(request.data)
        assert request.full_url == 'https://api.typesafe.ai/v1/systemone'
        return 200, request.full_url, raw

    @contextmanager
    def fence(current, proof):
        assert current == binding
        fences.append(proof)
        yield

    with open_native_runtime(**args) as runtime:
        engine = TypesafeJudgment(usage=usage, objects=runtime.authority.objects,
            policy=judgment_policy(evidence_digest=digest_bytes(b'fixture qualified'), qualified=True),
            api_key=lambda: 'fixture-never-live', source_fence=fence, transport=transport, clock=lambda: NOW, implementation_worktree_clean=True)
        inputs = dict(state=state, questions=questions, source_binding=binding, cycle_id='cycle-1',
                      caller_identity='NATIVE_ASSESSOR',
                      candidate_id='candidate-1', hypothesis_digest=digest_bytes(b'hypothesis'), proof=runtime.proof)
        reference = engine.evaluate(**inputs)
        receipt = engine.read(reference, **inputs)
        assert receipt['answers']['present']['noul_ppm'] == 950000
        assert receipt['answers']['support']['probabilities_ppm'] == {'yes': 900000, 'no': 100000}
        assert receipt['answers']['importance']['score_ppm'] == 1250000
        assert receipt['tariff']['calculated_usd_microunits'] == 214
        assert receipt['tariff']['basis'] == 'CALCULATED_FROM_QUALIFIED_TARIFF'
        assert receipt['model_alias'] == 'jev-latest' and receipt['model_returned'] == 'jev-1.13.0'
        assert receipt['raw_response_digest'] == digest_bytes(raw)
        assert engine.evaluate(**inputs) == reference
        assert len(calls) == 1
    with sqlite3.connect(usage.path) as db:
        assert db.execute('SELECT COUNT(*) FROM model_invocation_allocations').fetchone()[0] == 1
        assert db.execute('SELECT COUNT(*) FROM model_transport_observations').fetchone()[0] == 1
        row = json.loads(db.execute('SELECT record_json FROM model_invocation_terminals').fetchone()[0])
        assert row['usage_status'] == 'REPORTED'
        assert row['components']['total_tokens'] == 6232
        assert row['subscription_cli_chat_not_cash_debited'] is False

# The same real usage/CAS seam drives every failure, never a live provider.
import copy
import pytest
from dataclasses import replace, asdict
from newsroom.control_plane.typesafe_judgment import TypesafeJudgmentError


@contextmanager
def _case(tmp_path, monkeypatch, *, mutate=None, transport_error=None, fence_error=None):
    args = _args(tmp_path, monkeypatch)
    usage = ModelUsageService(str(tmp_path/'usage.sqlite3'))
    calls=[]
    state={'source':'The scheme is open.', 'claim':'The scheme is open.'}
    questions={'support':{'type':'choice','instructions':'Does source support claim?', 'criteria':{'yes':'Supported','no':'Unsupported'}}}
    value={'model':'jev-1.13.0','answers':{'support':{'type':'choice','choice':'yes','confidence':1,'probabilities':{'yes':1,'no':0}}},'usage':{'input_tokens':40,'output_tokens':10}}
    raw=json.dumps(value).encode() if mutate is None else mutate(copy.deepcopy(value))
    def transport(request, **_kw):
        calls.append(request)
        if transport_error: raise transport_error
        return 200, request.full_url, raw
    @contextmanager
    def fence(binding, proof):
        if fence_error: raise fence_error
        yield
    with open_native_runtime(**args) as runtime:
        engine=TypesafeJudgment(usage=usage, objects=runtime.authority.objects,
            policy=judgment_policy(evidence_digest=digest_bytes(b'fixture-qualified'),qualified=True),
            api_key=lambda:'fixture-only-secret',source_fence=fence,transport=transport,clock=lambda:NOW,implementation_worktree_clean=True)
        inputs=dict(state=state,questions=questions,source_binding={'source_id':'synthetic','revision_id':'revision-1','content_digest':digest_bytes(state['source'].encode())},
            caller_identity='NATIVE_ASSESSOR',cycle_id='cycle-1',candidate_id='candidate-1',hypothesis_digest=digest_bytes(b'hypothesis'),proof=runtime.proof)
        yield engine,inputs,usage,calls,raw


def _bad(kind):
    def mutate(value):
        if kind=='ids':value['answers']={'wrong':value['answers']['support']}
        elif kind=='type':value['answers']['support']['type']='noul'
        elif kind=='model':value['model']='grok-4.7'
        elif kind=='member':value['answers']['support']['choice']='outside'
        elif kind=='sum':value['answers']['support']['probabilities']={'yes':0.2,'no':0.2}
        elif kind=='negative':value['answers']['support']['probabilities']={'yes':1.1,'no':-0.1}
        elif kind=='bool':value['answers']['support']['confidence']=True
        elif kind=='duplicate':return b'{"model":"jev-1.13.0","model":"jev-1.13.0","answers":{},"usage":{"input_tokens":40,"output_tokens":10}}'
        elif kind=='nonfinite':return b'{"model":"jev-1.13.0","answers":{"support":{"type":"choice","choice":"yes","confidence":NaN,"probabilities":{"yes":1,"no":0}}},"usage":{"input_tokens":40,"output_tokens":10}}'
        elif kind=='usage':value.pop('usage')
        return json.dumps(value).encode()
    return mutate


@pytest.mark.parametrize('kind',['ids','type','model','member','sum','negative','bool','duplicate','nonfinite','usage'])
def test_invalid_response_never_grants_a_second_call_or_invents_usage(tmp_path,monkeypatch,kind):
    with _case(tmp_path,monkeypatch,mutate=_bad(kind)) as (engine,inputs,usage,calls,_raw):
        with pytest.raises(TypesafeJudgmentError) as first:engine.evaluate(**inputs)
        with sqlite3.connect(usage.path)as db:
            retained_terminal=db.execute('SELECT record_json FROM model_invocation_terminals').fetchone()[0]
        with pytest.raises(TypesafeJudgmentError,match='TYPESAFE_REPLAY_USAGE_HOLD') as replay:engine.evaluate(**inputs)
        assert replay.value.reference == first.value.reference
        assert len(calls)==1
        with sqlite3.connect(usage.path)as db:
            after=db.execute('SELECT record_json FROM model_invocation_terminals').fetchone()[0]
        assert after == retained_terminal
        value=json.loads(after)
        assert value['outcome']=='TYPESAFE_FAILED'
        if kind in {'duplicate','nonfinite','usage'}:
            assert value['usage_status']=='UNREPORTED' and value['components']['total_tokens'] is None
        else:
            assert value['usage_status']=='REPORTED' and value['components']['total_tokens']==50


def test_timeout_is_unknown_once_and_never_cash_zero(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch,transport_error=TimeoutError('fixture')) as(engine,inputs,usage,calls,_):
        for _ in range(2):
            with pytest.raises(TypesafeJudgmentError):engine.evaluate(**inputs)
        assert len(calls)==1
        with sqlite3.connect(usage.path)as db:
            value=json.loads(db.execute('SELECT record_json FROM model_invocation_terminals').fetchone()[0])
        assert value['usage_status']=='UNREPORTED' and not value['pre_dispatch_zero_proved']
        assert value['components']['total_tokens'] is None


def test_replay_binds_allocation_source_question_and_cas_member(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch)as(engine,inputs,usage,calls,_):
        ref=engine.evaluate(**inputs)
        altered={**inputs,'source_binding':{**inputs['source_binding'],'revision_id':'other'}}
        with pytest.raises(TypesafeJudgmentError):engine.read(ref,**altered)
        altered={**inputs,'questions':{**inputs['questions'],'other':inputs['questions']['support']}}
        with pytest.raises(TypesafeJudgmentError):engine.read(ref,**altered)
        with pytest.raises(TypesafeJudgmentError):engine.read(replace(ref,raw_admission_id=ref.receipt_admission_id),**inputs)
        assert len(calls)==1


def test_private_binding_never_enters_request_and_cross_envelope_parent_is_denied(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch)as(engine,inputs,_usage,calls,_):
        with pytest.raises(TypesafeJudgmentError):engine.evaluate(**inputs,parent_invocation_id='unrelated-parent')
        assert calls==[]
        engine.evaluate(**inputs)
        sent=json.loads(calls[0].data)
        assert set(sent)=={'state','questions','model'}
        assert 'source_binding' not in sent and 'candidate_id' not in sent
        assert inputs['hypothesis_digest'].encode()not in calls[0].data


def test_replay_denies_indexed_allocation_header_drift(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch)as(engine,inputs,usage,_calls,_):
        ref=engine.evaluate(**inputs)
        with sqlite3.connect(usage.path)as db:
            db.execute('UPDATE model_invocation_allocations SET request_digest=? WHERE invocation_id=?',(digest_bytes(b'foreign'),ref.invocation_id))
        with pytest.raises((TypesafeJudgmentError,ModelUsageIntegrityError),match='ALLOCATION_HOLD|binding differs'):
            engine.read(ref,**inputs)


def test_replay_rechecks_current_source_fence(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch)as(engine,inputs,_usage,calls,_):
        ref=engine.evaluate(**inputs)
        @contextmanager
        def revoked(_binding,_proof):
            raise RuntimeError('fixture source revoked')
            yield
        engine.fence=revoked
        with pytest.raises(RuntimeError,match='source revoked'):
            engine.read(ref,**inputs)
        assert len(calls)==1


def test_call_local_snapshot_and_nonreentrant_replay_fence(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch)as(engine,inputs,_usage,calls,_):
        active=False
        @contextmanager
        def single(_binding,_proof):
            nonlocal active
            assert not active
            active=True
            try:yield
            finally:active=False
        engine.fence=single
        original=engine.transport
        def mutate_caller(request,**kwargs):
            result=original(request,**kwargs)
            inputs['questions']['support']['instructions']='caller changed after request'
            return result
        engine.transport=mutate_caller
        original_instruction=inputs['questions']['support']['instructions']
        reference=engine.evaluate(**inputs)
        inputs['questions']['support']['instructions']=original_instruction
        assert engine.read(reference,**inputs)['answers']['support']['choice']=='yes'
        assert engine.evaluate(**inputs)==reference
        assert len(calls)==1


def test_preopened_intent_resumes_later_clock_without_rewriting(tmp_path,monkeypatch):
    from datetime import timedelta
    with _case(tmp_path,monkeypatch)as(engine,inputs,usage,calls,_):
        _request,_snapshot,envelope,_manifest=engine._input(**inputs)
        usage.open_envelope(envelope)
        engine.clock=lambda:NOW+timedelta(seconds=10)
        ref=engine.evaluate(**inputs)
        assert engine.read(ref,**inputs)['outcome']=='TYPESAFE_COMPLETE'
        assert len(calls)==1
        with sqlite3.connect(usage.path)as db:
            assert db.execute('SELECT admitted_at FROM model_work_envelopes').fetchone()[0]==envelope.as_record()['admitted_at']


def test_implementation_only_policy_upgrade_replays_original_zero_dispatch(tmp_path,monkeypatch):
    import newsroom.control_plane.typesafe_judgment as module
    with _case(tmp_path,monkeypatch)as(engine,inputs,_usage,calls,_):
        ref=engine.evaluate(**inputs)
        newer=module.InvocationEfficiencyPolicy.create(**{**asdict(engine.policy),'implementation_revision':digest_bytes(b'qualified-new-implementation')})
        monkeypatch.setattr(module, 'implementation_digest', lambda: newer.implementation_revision)
        engine = module.TypesafeJudgment(usage=engine.usage, objects=engine.objects, policy=newer, api_key=engine.key, source_fence=engine.fence, transport=engine.transport, clock=engine.clock, implementation_worktree_clean=True)
        assert engine.read(ref,**inputs)['outcome']=='TYPESAFE_COMPLETE'
        assert engine.evaluate(**inputs)==ref
        assert len(calls)==1


@pytest.mark.parametrize('table,column,value',[
    ('model_invocation_allocations','route','FOREIGN_ROUTE'),
    ('model_invocation_terminals','outcome','FOREIGN_OUTCOME'),
    ('model_provider_telemetry','provider_telemetry_digest',digest_bytes(b'foreign')),
])
def test_replay_denies_original_indexed_route_terminal_or_telemetry_drift(tmp_path,monkeypatch,table,column,value):
    with _case(tmp_path,monkeypatch)as(engine,inputs,usage,_calls,_):
        ref=engine.evaluate(**inputs)
        with sqlite3.connect(usage.path)as db:
            db.execute('UPDATE '+table+' SET '+column+'=? WHERE invocation_id=?',(value,ref.invocation_id))
        with pytest.raises((TypesafeJudgmentError,ModelUsageIntegrityError)):
            engine.read(ref,**inputs)


def test_unclean_implementation_denies_constructor_without_model(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch)as(engine,_inputs,_usage,calls,_):
        with pytest.raises(TypesafeJudgmentError,match='IMPLEMENTATION_HOLD'):
            TypesafeJudgment(usage=engine.usage, objects=engine.objects, policy=engine.policy,
                api_key=engine.key, source_fence=engine.fence, transport=engine.transport,
                clock=engine.clock, implementation_worktree_clean=False)
        assert calls==[]
