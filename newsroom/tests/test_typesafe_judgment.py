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


_ROUNDED_ROLES={'BACKGROUND':0.16,'MATERIAL':0.02,'UNCERTAIN':0.0,'SUPPORTING':0.81}


def _rounded_role_response(value):
    value['answers']={'support':{'type':'choice','choice':'SUPPORTING','confidence':0.75,'probabilities':dict(_ROUNDED_ROLES)}}
    return json.dumps(value).encode()


@contextmanager
def _reported_old_decoder_failure(tmp_path,monkeypatch,*,failure=ValueError):
    from newsroom.control_plane import typesafe_judgment as module
    with _case(tmp_path,monkeypatch,mutate=_rounded_role_response) as (engine,inputs,usage,calls,raw):
        inputs['questions']={'support':{'type':'choice','instructions':'Classify source role.',
            'criteria':{key:key for key in _ROUNDED_ROLES}}}
        # Reproduce the old strict decoder failure without forging durable SQL.
        with monkeypatch.context() as old:
            def strict(_answers,_questions):
                raise failure('probabilities do not sum to one')
            old.setattr(module,'_answers',strict)
            with pytest.raises(TypesafeJudgmentError) as held:
                engine.evaluate(**inputs)
        reference=held.value.reference
        assert reference is not None and len(calls)==1
        yield engine,inputs,usage,calls,raw,reference


def test_rounded_reported_failure_is_locally_revalidated_without_rewriting_original_receipts(tmp_path,monkeypatch):
    from newsroom.authority import HydrationRequest
    with _reported_old_decoder_failure(tmp_path,monkeypatch) as (engine,inputs,usage,calls,raw,reference):
        original_receipt=engine.objects.rehydrate(HydrationRequest(reference.receipt_admission_id,'evidence.record'),proof=inputs['proof']).data
        with sqlite3.connect(usage.path) as db:
            before={table:db.execute(f'SELECT record_json FROM {table} ORDER BY record_json').fetchall()
                for table in ('model_invocation_allocations','model_invocation_terminals','model_transport_observations','model_provider_telemetry')}
        record=engine.read(reference,**inputs)
        assert record['answers']['support']['probabilities_ppm']=={key:int(value*1000000)for key,value in _ROUNDED_ROLES.items()}
        assert record['answers']['support']['choice']=='SUPPORTING'
        assert record['outcome']=='TYPESAFE_FAILED' and usage.terminal(reference.invocation_id).outcome=='TYPESAFE_FAILED'
        assert record['consumer_revalidation']=={'consumer_contract':'newsroom.typesafe-judgment.answers-consumer.v2','original_outcome':'TYPESAFE_FAILED',
            'original_terminal_digest':usage.terminal(reference.invocation_id).terminal_digest,'raw_response_digest':digest_bytes(raw)}
        assert engine.evaluate(**inputs)==reference
        reopened=TypesafeJudgment(usage=ModelUsageService(usage.path),objects=engine.objects,policy=engine.policy,
            api_key=lambda:pytest.fail('replay must not read credential'),source_fence=engine.fence,
            transport=lambda *_args,**_kw:pytest.fail('replay must not call provider'),clock=engine.clock,implementation_worktree_clean=True)
        assert reopened.evaluate(**inputs)==reference
        assert len(calls)==1
        assert engine.objects.rehydrate(HydrationRequest(reference.receipt_admission_id,'evidence.record'),proof=inputs['proof']).data==original_receipt
        assert json.loads(original_receipt)['answers'] is None
        with sqlite3.connect(usage.path) as db:
            after={table:db.execute(f'SELECT record_json FROM {table} ORDER BY record_json').fetchall()for table in before}
        assert after==before
        with pytest.raises(TypesafeJudgmentError):
            engine.read(replace(reference,raw_admission_id=reference.receipt_admission_id),**inputs)


@pytest.mark.parametrize('probabilities',[
    {'BACKGROUND':0.16,'MATERIAL':0.02,'UNCERTAIN':0.0,'SUPPORTING':0.81},
    {'BACKGROUND':0.17,'MATERIAL':0.02,'UNCERTAIN':0.01,'SUPPORTING':0.81},
])
def test_choice_supports_inclusive_one_percent_provider_rounding_without_normalising(probabilities):
    from newsroom.control_plane.typesafe_judgment import _answers,_decode
    value=_decode(json.dumps({'support':{'type':'choice','choice':'SUPPORTING','confidence':0.75,'probabilities':probabilities}}).encode())
    result=_answers(value,{'support':{'type':'choice','criteria':{key:key for key in probabilities}}})
    assert result['support']['probabilities_ppm']=={key:int(value*1000000)for key,value in probabilities.items()}
    assert sum(result['support']['probabilities_ppm'].values())!=1000000


def test_score_and_gross_nonunit_choice_distributions_remain_rejected():
    from newsroom.control_plane.typesafe_judgment import _answers,_decode
    value=_decode(b'{"support":{"type":"choice","choice":"yes","confidence":1,"probabilities":{"yes":0.2,"no":0.2}}}')
    with pytest.raises(ValueError,match='probabilities do not sum'):
        _answers(value,{'support':{'type':'choice','criteria':{'yes':'Yes','no':'No'}}})
    value=_decode(b'{"score":{"type":"score","score":0.5,"confidence":0.5,"probabilities":{"0":0.49,"1":0.5},"legend":{"0":"Low","1":"High"}}}')
    with pytest.raises(ValueError,match='probabilities do not sum'):
        _answers(value,{'score':{'type':'score','criteria':['Low','High']}})


def test_non_valueerror_failed_terminal_is_never_revalidated_even_with_valid_rounded_raw(tmp_path,monkeypatch):
    with _reported_old_decoder_failure(tmp_path,monkeypatch,failure=RuntimeError) as (engine,inputs,usage,calls,_raw,reference):
        with pytest.raises(TypesafeJudgmentError,match='TYPESAFE_REPLAY_USAGE_HOLD'):
            engine.evaluate(**inputs)
        assert len(calls)==1 and usage.terminal(reference.invocation_id).failure_class=='RuntimeError'


@pytest.mark.parametrize('new_caller', ['native', 'graphiti'])
def test_timeout_quarantines_its_work_item_not_independent_paid_work(tmp_path, monkeypatch, new_caller):
    from newsroom.control_plane.model_usage import ModelUsageAdmissionError
    with _case(tmp_path, monkeypatch, transport_error=TimeoutError('fixture')) as (engine, inputs, usage, calls, raw):
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**inputs)
        with sqlite3.connect(usage.path) as db:
            old = db.execute('SELECT invocation_id,record_json FROM model_invocation_terminals').fetchone()
        def successful(request, **_):
            calls.append(request.data)
            return 200, request.full_url, raw
        engine.transport = successful
        different = {**inputs, 'cycle_id': 'different-work'}
        if new_caller == 'native':
            different['candidate_id'] = 'candidate-2'
        else:
            different.update(caller_identity='GRAPHITI_VERIFIER', candidate_id=None, hypothesis_digest=None,
                             ingest_id='ingest-2', graphiti_attempt_id='ingest-2:1')
        reference = engine.evaluate(**different)
        assert engine.read(reference, **different)['answers']['support']['choice'] == 'yes'
        assert len(calls) == 2
        with sqlite3.connect(usage.path) as db:
            assert db.execute('SELECT record_json FROM model_invocation_terminals WHERE invocation_id=?', (old[0],)).fetchone()[0] == old[1]
            assert db.execute('SELECT unresolved FROM model_usage_current WHERE invocation_id=?', (old[0],)).fetchone() == (1,)
        assert usage.route_state('TYPESAFE_JUDGMENT')['state'] == 'OPEN'
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**inputs)
        with pytest.raises(ModelUsageAdmissionError):
            engine.evaluate(**{**inputs, 'cycle_id': 'disguised-retry'})
        assert len(calls) == 2


@pytest.mark.parametrize('fault', ['missing-context', 'context-role', 'missing-dispatch', 'envelope-header',
                                   'systemic-open', 'circuit-header', 'active', 'unqualified', 'dispatch-binding', 'context-header'])
def test_timeout_scope_never_relaxes_corrupt_or_unsettled_admission(tmp_path, monkeypatch, fault):
    from newsroom.control_plane.model_usage import ModelUsageAdmissionError, ModelUsageIntegrityError
    with _case(tmp_path, monkeypatch, transport_error=TimeoutError('fixture')) as (engine, inputs, usage, calls, raw):
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**inputs)
        with sqlite3.connect(usage.path) as db:
            allocation = json.loads(db.execute('SELECT record_json FROM model_invocation_allocations').fetchone()[0])
            invocation, envelope, manifest = (allocation[key] for key in ('invocation_id', 'envelope_id', 'context_manifest_digest'))
            if fault == 'missing-context':
                db.execute('DELETE FROM model_invocation_context_manifests WHERE context_manifest_digest=?', (manifest,))
            elif fault == 'context-header':
                db.execute("UPDATE model_invocation_context_manifests SET route='tampered' WHERE context_manifest_digest=?", (manifest,))
            elif fault == 'context-role':
                row = json.loads(db.execute('SELECT record_json FROM model_invocation_context_manifests WHERE context_manifest_digest=?', (manifest,)).fetchone()[0])
                row['caller_identity'] = 'GRAPHITI_VERIFIER'
                db.execute('UPDATE model_invocation_context_manifests SET record_json=? WHERE context_manifest_digest=?', (json.dumps(row), manifest))
            elif fault == 'missing-dispatch':
                db.execute('DELETE FROM model_transport_observations WHERE invocation_id=?', (invocation,))
            elif fault == 'dispatch-binding':
                from newsroom.authority.canonical import canonical_json_bytes, digest_canonical
                row = json.loads(db.execute('SELECT record_json FROM model_transport_observations WHERE invocation_id=?', (invocation,)).fetchone()[0])
                row.pop('observation_digest'); row['evidence_digest'] = digest_bytes(b'other request')
                digest = digest_canonical(row); row['observation_digest'] = digest
                db.execute('UPDATE model_transport_observations SET observation_digest=?,evidence_digest=?,record_json=? WHERE invocation_id=?',
                           (digest, row['evidence_digest'], canonical_json_bytes(row).decode(), invocation))
            elif fault == 'unqualified':
                from newsroom.authority.canonical import canonical_json_bytes
                policy = engine.policy.as_record(); policy['qualified'] = False
                db.execute('UPDATE model_invocation_policies SET qualified=0,record_json=? WHERE canonical_digest=?',
                           (canonical_json_bytes(policy).decode(), engine.policy.canonical_digest))

            elif fault == 'envelope-header':
                db.execute("UPDATE model_work_envelopes SET cycle_id='tampered' WHERE envelope_id=?", (envelope,))
            elif fault == 'systemic-open':
                usage._append_route_state(db, route='TYPESAFE_JUDGMENT', state='OPEN', reason='SYSTEMIC_TRANSPORT',
                                          invocation_id=None, recorded_at=NOW.replace(year=2027))
            elif fault == 'circuit-header':
                usage._append_route_state(db, route='TYPESAFE_JUDGMENT', state='OPEN', reason='SYSTEMIC_TRANSPORT',
                                          invocation_id=None, recorded_at=NOW.replace(year=2027))
                db.execute("UPDATE model_usage_route_circuit_events SET reason='TimeoutError',invocation_id=? WHERE reason='SYSTEMIC_TRANSPORT'", (invocation,))
        if fault == 'active':
            def interrupted(request, **_):
                calls.append(request.data)
                raise KeyboardInterrupt('fixture process interruption')
            engine.transport = interrupted
            with pytest.raises(KeyboardInterrupt):
                engine.evaluate(**{**inputs, 'cycle_id': 'active-work', 'candidate_id': 'candidate-2'})
        before = len(calls)
        engine.transport = lambda request, **_: (200, request.full_url, raw)
        with pytest.raises((ModelUsageAdmissionError, ModelUsageIntegrityError, ValueError)):
            engine.evaluate(**{**inputs, 'cycle_id': 'unrelated-work', 'candidate_id': 'candidate-3'})
        assert len(calls) == before


@pytest.mark.parametrize('new_role', ['native', 'graphiti'])
def test_unknown_known_ingest_remains_quarantined_across_roles(tmp_path, monkeypatch, new_role):
    from newsroom.control_plane.model_usage import ModelUsageAdmissionError
    with _case(tmp_path, monkeypatch, transport_error=TimeoutError('fixture')) as (engine, inputs, usage, calls, raw):
        original = {**inputs, 'ingest_id': 'shared-ingest'}
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**original)
        independent = {**inputs, 'cycle_id': 'different-cycle', 'candidate_id': 'candidate-2', 'ingest_id': 'shared-ingest'}
        if new_role == 'graphiti':
            independent.update(caller_identity='GRAPHITI_VERIFIER', candidate_id=None, hypothesis_digest=None,
                               graphiti_attempt_id='shared-ingest:2')
        with pytest.raises(ModelUsageAdmissionError):
            engine.evaluate(**independent)
        assert len(calls) == 1


@pytest.mark.parametrize('mixed', ['unknown-non-timeout', 'policy-breach'])
def test_timeout_exception_requires_every_current_blocker_to_be_eligible(tmp_path, monkeypatch, mixed):
    from newsroom.control_plane.model_usage import ModelUsageAdmissionError
    with _case(tmp_path, monkeypatch, transport_error=TimeoutError('fixture')) as (engine, inputs, usage, calls, raw):
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**inputs)
        with sqlite3.connect(usage.path) as db:
            old = db.execute('SELECT invocation_id FROM model_invocation_terminals').fetchone()[0]
        def mixed_failure(request, **_):
            calls.append(request.data)
            if mixed == 'unknown-non-timeout':
                raise ValueError('unknown provider failure')
            value = json.loads(raw); value['usage']['output_tokens'] = 1_000_000
            return 200, request.full_url, json.dumps(value).encode()
        engine.transport = mixed_failure
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**{**inputs, 'candidate_id': 'candidate-2', 'cycle_id': 'mixed-work'})
        # The latest selected circuit may still be the eligible timeout; current
        # liabilities, not that one event, must establish the scope exception.
        with sqlite3.connect(usage.path) as db:
            db.execute('DELETE FROM model_usage_route_circuit_events WHERE invocation_id != ?', (old,))
        with pytest.raises(ModelUsageAdmissionError):
            engine.evaluate(**{**inputs, 'candidate_id': 'candidate-3', 'cycle_id': 'third-work'})
        assert len(calls) == 2


def test_unknown_verifier_never_retries_its_ingest_under_a_new_attempt(tmp_path, monkeypatch):
    from newsroom.control_plane.model_usage import ModelUsageAdmissionError
    with _case(tmp_path, monkeypatch, transport_error=TimeoutError('fixture')) as (engine, inputs, usage, calls, raw):
        verifier = {**inputs, 'caller_identity': 'GRAPHITI_VERIFIER', 'candidate_id': None,
                    'hypothesis_digest': None, 'ingest_id': 'ingest-1', 'graphiti_attempt_id': 'ingest-1:1'}
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**verifier)
        with pytest.raises(ModelUsageAdmissionError):
            engine.evaluate(**{**verifier, 'cycle_id': 'changed', 'graphiti_attempt_id': 'ingest-1:2'})
        assert len(calls) == 1


@pytest.mark.parametrize('status', [400, 401, 402, 429, 503])
def test_http_failure_retains_bounded_response_status_without_inventing_usage(tmp_path, monkeypatch, status):
    from io import BytesIO
    from urllib.error import HTTPError
    from newsroom.authority import HydrationRequest
    from newsroom.control_plane.typesafe_judgment import URL
    body = json.dumps({'error': {'code': 'fixture-error', 'message': 'fixture response'}}).encode()
    with _case(tmp_path, monkeypatch,
               transport_error=HTTPError(URL, status, 'fixture status', {}, BytesIO(body))) as (engine, inputs, usage, calls, _):
        with pytest.raises(TypesafeJudgmentError) as held:
            engine.evaluate(**inputs)
        reference = held.value.reference
        assert reference is not None
        receipt = json.loads(engine.objects.rehydrate(
            HydrationRequest(reference.receipt_admission_id, 'evidence.record'), proof=inputs['proof']).data)
        assert receipt['transport_failure'] == {'status': status, 'endpoint_matches': True}
        assert engine.objects.rehydrate(HydrationRequest(reference.raw_admission_id, 'evidence.record'),
                                        proof=inputs['proof']).data == body
        terminal = usage.terminal(reference.invocation_id)
        assert terminal.usage_status.value == 'UNREPORTED'
        assert terminal.components.total_tokens is None and not terminal.pre_dispatch_zero_proved
        assert terminal.failure_class == f'HTTPError:{status}'
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**inputs)
        assert len(calls) == 1


@pytest.mark.parametrize('response', ['unreadable', 'oversized'])
def test_http_failure_diagnostic_survives_without_a_retainable_body(tmp_path, monkeypatch, response):
    from io import BytesIO
    from urllib.error import HTTPError
    from newsroom.authority import HydrationRequest, ObjectAdmissionRequest
    from newsroom.control_plane.typesafe_judgment import URL, MAX_RESPONSE_BYTES
    class Unreadable:
        def read(self, *_):
            raise OSError('fixture unreadable body')
        def close(self):
            pass
    stream = Unreadable() if response == 'unreadable' else BytesIO(b'x' * (MAX_RESPONSE_BYTES + 1))
    with _case(tmp_path, monkeypatch, transport_error=HTTPError(URL, 503, 'fixture', {}, stream)) as (engine, inputs, usage, calls, _):
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**inputs)
        with sqlite3.connect(usage.path) as db:
            invocation = db.execute('SELECT invocation_id FROM model_invocation_allocations').fetchone()[0]
        admission = engine.objects.committed_admission(ObjectAdmissionRequest('evidence.record',
            'typesafe-transport-failure:' + invocation), proof=inputs['proof']).admission
        diagnostic = json.loads(engine.objects.rehydrate(HydrationRequest(admission.admission_id, 'evidence.record'),
                                                        proof=inputs['proof']).data)
        assert diagnostic['transport_failure']['status'] == 503
        if response == 'unreadable':
            assert diagnostic['transport_failure']['response_read_failure'] == 'OSError'
        terminal = usage.terminal(invocation)
        assert terminal.usage_status.value == 'UNREPORTED' and terminal.components.total_tokens is None
        assert len(calls) == 1


@pytest.mark.parametrize('elapsed_seconds,allowed', [(0, False), (299, False), (300, True)])
def test_unknown_http_work_is_quarantined_with_bounded_availability_cooldown(tmp_path, monkeypatch, elapsed_seconds, allowed):
    from datetime import timedelta
    from urllib.error import HTTPError
    from newsroom.control_plane.model_usage import ModelUsageAdmissionError
    from newsroom.control_plane.typesafe_judgment import URL
    with _case(tmp_path, monkeypatch, transport_error=HTTPError(URL, 422, 'fixture', {}, None)) as (engine, inputs, usage, calls, raw):
        # This reproduces the already recorded legacy status-less HTTPError.
        monkeypatch.setattr('newsroom.control_plane.typesafe_judgment.urllib.error.HTTPError', type('DifferentError', (Exception,), {}))
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**inputs)
        engine.clock = lambda: NOW + timedelta(seconds=elapsed_seconds)
        def success(request, **_):
            calls.append(request.data)
            return 200, request.full_url, raw
        engine.transport = success
        other = {**inputs, 'cycle_id': 'new-independent-work', 'candidate_id': 'candidate-2',
                 'state': {'source':'A genuinely different Source has a new update.'},
                 'source_binding': {'content_digest': digest_bytes(b'new source')}}
        if allowed:
            reference = engine.evaluate(**other)
            assert usage.terminal(reference.invocation_id).usage_status.value == 'REPORTED'
            assert len(calls) == 2
            with pytest.raises(ModelUsageAdmissionError):
                engine.evaluate(**{**inputs, 'cycle_id': 'disguised-original-retry'})
        else:
            with pytest.raises(ModelUsageAdmissionError):
                engine.evaluate(**other)
            assert len(calls) == 1
        assert usage.route_state('TYPESAFE_JUDGMENT')['state'] == 'OPEN'


@pytest.mark.parametrize('status', [401, 402, 403, 429, 529])
def test_known_account_or_unapplied_rate_delay_never_uses_http_relaxation(tmp_path, monkeypatch, status):
    from datetime import timedelta
    from io import BytesIO
    from urllib.error import HTTPError
    from newsroom.control_plane.model_usage import ModelUsageAdmissionError
    from newsroom.control_plane.typesafe_judgment import URL
    with _case(tmp_path, monkeypatch,
               transport_error=HTTPError(URL, status, 'fixture', {}, BytesIO(b'{}'))) as (engine, inputs, usage, calls, raw):
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**inputs)
        engine.clock = lambda: NOW + timedelta(hours=1)
        with pytest.raises(ModelUsageAdmissionError):
            engine.evaluate(**{**inputs, 'candidate_id':'new-candidate', 'cycle_id':'fresh-work',
                               'state':{'source':'Different independent evidence'},
                               'source_binding':{'content_digest':digest_bytes(b'new evidence')}})
        assert len(calls) == 1


@pytest.mark.parametrize('change', ['candidate-only', 'binding-only'])
def test_unknown_http_cannot_launder_a_repeated_public_request(tmp_path, monkeypatch, change):
    from datetime import timedelta
    from urllib.error import HTTPError
    from newsroom.control_plane.model_usage import ModelUsageAdmissionError
    from newsroom.control_plane.typesafe_judgment import URL
    failure = HTTPError(URL, 422, 'fixture legacy status loss', {}, None)
    with _case(tmp_path, monkeypatch, transport_error=failure) as (engine, inputs, usage, calls, raw):
        monkeypatch.setattr('newsroom.control_plane.typesafe_judgment.urllib.error.HTTPError', type('OtherError', (Exception,), {}))
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**inputs)
        engine.clock = lambda: NOW + timedelta(minutes=6)
        repeated = {**inputs, 'candidate_id':'another-wrapper', 'cycle_id':'changed-wrapper'}
        if change == 'binding-only':
            repeated['source_binding'] = {'content_digest': digest_bytes(b'another private binding')}
        with pytest.raises(ModelUsageAdmissionError):
            engine.evaluate(**repeated)
        assert len(calls) == 1


def test_http_diagnostics_only_retain_documented_allowlisted_headers(tmp_path, monkeypatch):
    from io import BytesIO
    from urllib.error import HTTPError
    from newsroom.authority import HydrationRequest
    from newsroom.control_plane.typesafe_judgment import URL
    headers = {'x-typesafe-request-id':'req_123-safe', 'Retry-After':'600', 'Authorization':'private-never-retained'}
    with _case(tmp_path, monkeypatch, transport_error=HTTPError(URL, 429, 'fixture', headers, BytesIO(b'{}'))) as (engine, inputs, usage, calls, _):
        with pytest.raises(TypesafeJudgmentError) as held:
            engine.evaluate(**inputs)
        diagnostic = engine.objects.rehydrate(HydrationRequest(held.value.reference.receipt_admission_id, 'evidence.record'), proof=inputs['proof']).data
        record = json.loads(diagnostic)
        assert record['transport_failure']['provider_request_id'] == 'req_123-safe'
        assert record['transport_failure']['retry_after_seconds'] == 600
        assert b'private-never-retained' not in diagnostic and b'Authorization' not in diagnostic


@pytest.mark.parametrize('retry_after', ['600', 'Wed, 07 Oct 2026 12:00:00 GMT', 'invalid'])
def test_server_requested_503_delay_never_falls_back_to_shorter_cooldown(tmp_path, monkeypatch, retry_after):
    from datetime import timedelta
    from io import BytesIO
    from urllib.error import HTTPError
    from newsroom.control_plane.model_usage import ModelUsageAdmissionError
    from newsroom.control_plane.typesafe_judgment import URL
    with _case(tmp_path, monkeypatch, transport_error=HTTPError(URL, 503, 'fixture', {'Retry-After':retry_after}, BytesIO(b'{}'))) as (engine, inputs, usage, calls, _):
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**inputs)
        engine.clock = lambda: NOW + timedelta(minutes=5)
        with pytest.raises(ModelUsageAdmissionError):
            engine.evaluate(**{**inputs, 'candidate_id':'new-candidate', 'cycle_id':'new-source',
                               'state':{'source':'new independent source'},
                               'source_binding':{'content_digest':digest_bytes(b'different evidence')}})
        assert len(calls) == 1


@pytest.mark.parametrize('status,seconds,allowed', [(400, 0, True), (422, 0, True), (500, 299, False), (503, 300, True)])
def test_known_request_or_transient_server_error_admission_boundaries(tmp_path, monkeypatch, status, seconds, allowed):
    from datetime import timedelta
    from io import BytesIO
    from urllib.error import HTTPError
    from newsroom.control_plane.model_usage import ModelUsageAdmissionError
    from newsroom.control_plane.typesafe_judgment import URL
    with _case(tmp_path, monkeypatch, transport_error=HTTPError(URL, status, 'fixture', {}, BytesIO(b'{}'))) as (engine, inputs, usage, calls, raw):
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**inputs)
        engine.clock = lambda: NOW + timedelta(seconds=seconds)
        def success(request, **_):
            calls.append(request.data)
            return 200, request.full_url, raw
        engine.transport = success
        fresh = {**inputs, 'candidate_id':'independent-candidate', 'cycle_id':'new-source',
                 'state':{'source':'new unrelated source'}, 'source_binding':{'content_digest':digest_bytes(b'new unrelated source')}}
        if allowed:
            engine.evaluate(**fresh)
            assert len(calls) == 2
        else:
            with pytest.raises(ModelUsageAdmissionError):
                engine.evaluate(**fresh)
            assert len(calls) == 1


def test_new_unknown_server_failure_restarts_cooldown_without_clearing_old_liability(tmp_path, monkeypatch):
    from datetime import timedelta
    from io import BytesIO
    from urllib.error import HTTPError
    from newsroom.control_plane.model_usage import ModelUsageAdmissionError
    from newsroom.control_plane.typesafe_judgment import URL
    with _case(tmp_path, monkeypatch, transport_error=HTTPError(URL, 503, 'fixture', {}, BytesIO(b'{}'))) as (engine, inputs, usage, calls, raw):
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**inputs)
        with sqlite3.connect(usage.path) as db:
            old = db.execute('SELECT invocation_id,record_json FROM model_invocation_terminals').fetchone()
        engine.clock = lambda: NOW + timedelta(seconds=300)
        fresh = {**inputs, 'candidate_id':'candidate-2', 'cycle_id':'source-2',
                 'state':{'source':'second independent source'}, 'source_binding':{'content_digest':digest_bytes(b'source-2')}}
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**fresh)
        engine.clock = lambda: NOW + timedelta(seconds=599)
        with pytest.raises(ModelUsageAdmissionError):
            engine.evaluate(**{**fresh, 'candidate_id':'candidate-3', 'cycle_id':'source-3',
                               'state':{'source':'third independent source'}, 'source_binding':{'content_digest':digest_bytes(b'source-3')}})
        assert len(calls) == 2
        with sqlite3.connect(usage.path) as db:
            assert db.execute('SELECT record_json FROM model_invocation_terminals WHERE invocation_id=?', (old[0],)).fetchone()[0] == old[1]
            assert db.execute('SELECT unresolved FROM model_usage_current WHERE invocation_id=?', (old[0],)).fetchone() == (1,)


def test_latest_canonical_circuit_reason_must_match_its_actual_terminal(tmp_path, monkeypatch):
    from newsroom.authority.canonical import canonical_json_bytes, digest_canonical
    from newsroom.control_plane.model_usage import ModelUsageAdmissionError
    with _case(tmp_path, monkeypatch, transport_error=TimeoutError('fixture')) as (engine, inputs, usage, calls, _):
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**inputs)
        with sqlite3.connect(usage.path) as db:
            record = json.loads(db.execute('SELECT record_json FROM model_usage_route_circuit_events').fetchone()[0])
            record.pop('event_digest'); record['reason'] = 'HTTPError:422'
            digest = digest_canonical(record); record['event_digest'] = digest
            db.execute('UPDATE model_usage_route_circuit_events SET event_digest=?,reason=?,record_json=?',
                       (digest, record['reason'], canonical_json_bytes(record).decode()))
        with pytest.raises(ModelUsageAdmissionError):
            engine.evaluate(**{**inputs, 'candidate_id':'candidate-2', 'cycle_id':'other',
                               'state':{'source':'other source'}, 'source_binding':{'content_digest':digest_bytes(b'other source')}})
        assert len(calls) == 1


@pytest.mark.parametrize('url,status', [('https://other.invalid/', 422), ('https://api.typesafe.ai/v1/systemone', True),
                                        ('https://api.typesafe.ai/v1/systemone', None), ('https://api.typesafe.ai/v1/systemone', '503')])
def test_malformed_http_transport_never_becomes_legacy_unknown_retry_class(tmp_path, monkeypatch, url, status):
    from io import BytesIO
    from urllib.error import HTTPError
    with _case(tmp_path, monkeypatch, transport_error=HTTPError(url, status, 'fixture', {}, BytesIO(b'{}'))) as (engine, inputs, usage, calls, _):
        with pytest.raises(TypesafeJudgmentError) as held:
            engine.evaluate(**inputs)
        assert usage.terminal(held.value.reference.invocation_id).failure_class == 'HTTPError:INVALID_TRANSPORT'
