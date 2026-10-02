"""Native SDK reported consumption is separate from final-response validity."""
import asyncio
import json
from dataclasses import replace
from datetime import timedelta

import pytest

from newsroom.control_plane.graphiti import GraphitiModelUsageObserver
from newsroom.control_plane.graphiti_requests import load_checked_native_graphiti_call_shape_policy
from newsroom.control_plane.graphiti_fallback_policy import load_checked_native_graphiti_fallback_circuit_policy
from newsroom.graphiti_adapter.cli_client import run_cli_chain, run_cursor_agent_llm
from newsroom.tests.test_graphiti_cursor_sdk_transport import (
    FakeRuntime, FakeRun, FakeTerminal, FakeUsage, FakeMessage, _assistant, _bind, _observer_fixture,
)


def _case(tmp_path, monkeypatch, *, text='{"ok":true}', usage=None, schema=None, semantic='UNSTRUCTURED', legacy=False):
    monkeypatch.setattr('newsroom.control_plane.graphiti._graphiti_implementation_identity', lambda: ('a'*40, True))
    service, previous, at = _observer_fixture(tmp_path)
    shape=load_checked_native_graphiti_call_shape_policy()
    fallback=load_checked_native_graphiti_fallback_circuit_policy()
    if legacy:
        from pathlib import Path
        from newsroom.control_plane import graphiti_requests, graphiti_fallback_policy
        shape=graphiti_requests._load_checked_graphiti_call_shape_policy(
            Path(graphiti_requests.__file__).with_name('native_graphiti_call_shape_policy_v1.json'))
        fallback=graphiti_fallback_policy._load_checked_graphiti_fallback_circuit_policy(
            Path(graphiti_fallback_policy.__file__).with_name('native_graphiti_fallback_circuit_policy_v1.json'))
        assert shape.canonical_digest=='sha256:7362fe7ffede8c5ade4079e75a869d5d31ccac7170784e703ed66a603d4c1d3f'
    observer = GraphitiModelUsageObserver(
        service=service, envelope=previous._envelope, clock=lambda: at+timedelta(seconds=10),
        owner_stop_check=lambda: None,
        call_shape_policy=shape,
        fallback_policy=fallback,
    )
    runtime = _bind(monkeypatch, FakeRuntime(run=FakeRun(messages=(_assistant(text),),
        terminal=FakeTerminal(result=text,usage=usage))))
    calls=[]
    def run(**kwargs):
        return asyncio.run(run_cli_chain(prompt='exact native source input', schema=schema,semantic_request_class=semantic,
            cursor_runner=run_cursor_agent_llm,
            grok_runner=lambda *_a, **_kw: pytest.fail('fallback dispatched'),
            invocations=calls,invocation_observer=observer,max_tokens=16_384,
            fallback_permitted=False,**kwargs))
    return service,observer,runtime,calls,run


@pytest.mark.parametrize('reasoning', [None, 16_000])
def test_valid_small_final_answer_survives_reported_output_including_unknown_thinking(tmp_path, monkeypatch,reasoning):
    service,observer,runtime,calls,run=_case(tmp_path,monkeypatch,
        usage=FakeUsage(70_691,20_287,cache_read_tokens=60_544,reasoning_tokens=reasoning))
    assert run()=={'ok':True}
    assert calls[0]['outcome']=='COMPLETE'
    assert calls[0]['usage']['total_tokens']==151_522
    assert calls[0]['usage']['reasoning_tokens'] == reasoning
    assert calls[0]['response_quality']['final_utf8_bytes']==11
    assert calls[0]['response_quality']['json_parse']=='OBJECT'
    assert calls[0]['response_quality']['schema_status']=='NOT_REQUIRED'
    terminal=service.terminal(calls[0]['model_invocation_id'])
    assert terminal.outcome=='COMPLETE' and terminal.policy_breach is None
    assert terminal.components.context_tokens is None
    assert service.route_state('GRAPHITI_CHAT_PRIMARY')['state']=='CLOSED'
    assert len(runtime.requests)==1


@pytest.mark.parametrize('streamed', [False, True])
def test_actual_final_utf8_ceiling_applies_to_stream_and_terminal_fallback(monkeypatch, streamed):
    from newsroom.graphiti_adapter.cursor_transport import run_cursor_transport, CursorSdkBoundedFailure, cursor_output_limit
    cap=cursor_output_limit(1)
    text='€'*(cap//3+1)
    runtime=_bind(monkeypatch,FakeRuntime(run=FakeRun(
        messages=(_assistant(text),) if streamed else (),
        terminal=FakeTerminal(result=text,usage=FakeUsage(2,1)))))
    with pytest.raises(CursorSdkBoundedFailure) as error:
        run_cursor_transport(prompt='source',max_tokens=1,timeout=5,idempotency_key='sha256:'+'a'*64)
    assert error.value.error_class=='OUTPUT_BOUND'
    assert error.value.execution.cancelled is streamed
    assert runtime.run.cancel_count==(1 if streamed else 0)
    assert len(runtime.requests)==1


@pytest.mark.parametrize('text', ['not-json','[]'])
def test_advisory_tokens_do_not_hide_json_failure(tmp_path, monkeypatch, text):
    from newsroom.graphiti_adapter.cli_client import CliResponseError
    service,observer,runtime,calls,run=_case(tmp_path,monkeypatch,text=text,
        usage=FakeUsage(70_691,20_287,cache_read_tokens=60_544))
    with pytest.raises(CliResponseError):run()
    assert calls[0]['outcome']=='MALFORMED_OUTPUT'
    assert calls[0]['response_quality']['json_parse']=='NOT_OBJECT'
    assert calls[0]['response_quality']['schema_status']=='NOT_CHECKED'
    assert service.terminal(calls[0]['model_invocation_id']).policy_breach is None
    assert len(runtime.requests)==1


@pytest.mark.parametrize('usage', [None, FakeUsage(False,20_287,cache_read_tokens=60_544)])
def test_missing_or_false_sdk_usage_is_not_fabricated_zero(tmp_path, monkeypatch, usage):
    service,observer,runtime,calls,run=_case(tmp_path,monkeypatch,usage=usage)
    run()
    terminal=service.terminal(calls[0]['model_invocation_id'])
    assert terminal.usage_status.value=='UNREPORTED'
    assert terminal.components.total_tokens is None
    assert terminal.components.reasoning_tokens is None
    assert terminal.components.context_tokens is None
    assert service.route_state('GRAPHITI_CHAT_PRIMARY')['state']=='OPEN'


def test_native_response_contract_rejects_swapped_allocation_identity(tmp_path,monkeypatch):
    service,observer,runtime,calls,run=_case(tmp_path,monkeypatch,usage=FakeUsage(2,1))
    token=observer.before_cli_invocation(provider='cursor-agent-cli',model='composer-2.5',
        prompt='exact',schema=None,max_tokens=16_384)
    assert observer.reported_token_targets_are_advisory(token) is True
    from newsroom.control_plane.model_usage import ModelUsageIntegrityError
    with pytest.raises(ModelUsageIntegrityError,match='allocation differs'):
        observer.reported_token_targets_are_advisory(replace(token,invocation_policy_digest='sha256:'+'b'*64))
    assert runtime.requests==[]


def test_advisory_usage_keeps_response_schema_failure(tmp_path,monkeypatch):
    from newsroom.graphiti_adapter.cli_client import CliResponseError
    from newsroom.tests.test_native_fallback_cancellation_disposition import EXTRACTED_ENTITIES_SCHEMA
    service,observer,runtime,calls,run=_case(tmp_path,monkeypatch,text='{"wrong":true}',
        schema=EXTRACTED_ENTITIES_SCHEMA,semantic='ExtractedEntities',
        usage=FakeUsage(70_691,20_287,cache_read_tokens=60_544))
    with pytest.raises(CliResponseError):run()
    assert calls[0]['outcome']=='MALFORMED_OUTPUT'
    assert calls[0]['response_quality']['json_parse']=='OBJECT'
    assert calls[0]['response_quality']['schema_status']=='INVALID'
    assert len(runtime.requests)==1


def test_legacy_v12_output_gate_and_receipt_shape_are_frozen(tmp_path,monkeypatch):
    from newsroom.graphiti_adapter.cli_client import CliResponseError
    service,observer,runtime,calls,run=_case(tmp_path,monkeypatch,legacy=True,
        usage=FakeUsage(70_691,20_287,cache_read_tokens=60_544))
    with pytest.raises(CliResponseError,match='requested max_tokens'):run()
    assert calls[0]['outcome']=='OUTPUT_LIMIT_EXCEEDED'
    assert 'response_quality' not in calls[0]
    assert service.terminal(calls[0]['model_invocation_id']).policy_breach=='MAX_TOTAL_TOKENS_EXCEEDED'
    assert service.route_state('GRAPHITI_CHAT_PRIMARY')['state']=='OPEN'


def test_real_measured_context_remains_a_hard_breach(tmp_path,monkeypatch):
    service,observer,runtime,calls,run=_case(tmp_path,monkeypatch)
    token=observer.before_cli_invocation(provider='cursor-agent-cli',model='composer-2.5',
        prompt='source',schema=None,max_tokens=16_384)
    observer.transport_dispatch_started(token)
    observer.after_cli_invocation(token,outcome='COMPLETE',usage=dict(usage_basis='PROVIDER_REPORTED',
        input_tokens=70_691,output_tokens=20_287,cached_read_tokens=60_544,cached_write_tokens=0,
        total_tokens=151_522,reasoning_tokens=None,context_tokens=131_073))
    terminal=service.terminal(token.invocation_id)
    assert terminal.policy_breach=='MAX_CONTEXT_TOKENS_EXCEEDED'
    assert service.route_state('GRAPHITI_CHAT_PRIMARY')['state']=='OPEN'


@pytest.mark.parametrize('source_valid',[True,False])
def test_advisory_response_reaches_unchanged_source_evidence_validation(tmp_path,monkeypatch,source_valid):
    from newsroom.graphiti_adapter.combined_temporal_fixtures import fixture
    from newsroom.graphiti_adapter.combined_temporal_contract import SCHEMA,CONTRACT_NAME
    from newsroom.graphiti_adapter.combined_temporal_extraction import CombinedTemporalTransportResult,CombinedTemporalOutcome
    from newsroom.graphiti_adapter.combined_temporal_runtime import extract_combined_temporal_async
    from newsroom.tests.test_graphiti_combined_temporal_runtime import _Pipeline
    case=fixture('pair-current')
    raw=json.loads(json.dumps(case.gold))
    if not source_valid:raw['facts'][0]['evidence_segment_ids']=[999]
    service,observer,runtime,calls,_run=_case(tmp_path,monkeypatch,text=json.dumps(raw),
        usage=FakeUsage(70_691,20_287,cache_read_tokens=60_544))
    class Transport:
        async def generate_response(self,*,prompt,schema,response_model,max_tokens):
            result=await run_cli_chain(prompt=prompt,schema=json.dumps(schema),
                semantic_request_class=response_model,cursor_runner=run_cursor_agent_llm,
                grok_runner=lambda *_a,**_kw:pytest.fail('fallback'),invocations=calls,
                invocation_observer=observer,max_tokens=max_tokens,fallback_permitted=False)
            return CombinedTemporalTransportResult(raw=result,framework_version='fixture',model_version='composer-2.5',
                token_usage=calls[0]['usage'],provider_cost=None)
    class Pipeline(_Pipeline):
        executed=False
        async def _execute(self,**kwargs):
            self.executed=True
            return await super()._execute(**kwargs)
    pipeline=Pipeline()
    leaf=asyncio.run(extract_combined_temporal_async(case.revision,transport=Transport(),pipeline=pipeline))
    assert calls[0]['outcome']=='COMPLETE'
    assert calls[0]['response_quality']['schema_status']=='VALID'
    # Existing projection may retain valid entities while holding one bad fact.
    assert pipeline.executed is True
    assert len(leaf.edges)==(1 if source_valid else 0)
    assert len(leaf.payload['facts'])==(1 if source_valid else 0)
    assert len(runtime.requests)==1


def test_advisory_response_preserves_semantic_identity_rejection(tmp_path,monkeypatch):
    from newsroom.graphiti_adapter.combined_temporal_fixtures import fixture
    from newsroom.graphiti_adapter.combined_temporal_runtime import extract_combined_temporal_async
    from newsroom.graphiti_adapter.combined_temporal_extraction import CombinedTemporalTransportResult,CombinedTemporalOutcome
    from newsroom.tests.test_graphiti_combined_temporal_runtime import _Pipeline
    case=fixture('pair-current');raw=json.loads(json.dumps(case.gold))
    raw['entities'][1]['local_id']=raw['entities'][0]['local_id']
    service,observer,runtime,calls,_run=_case(tmp_path,monkeypatch,text=json.dumps(raw),
        usage=FakeUsage(70_691,20_287,cache_read_tokens=60_544))
    class Transport:
        async def generate_response(self,*,prompt,schema,response_model,max_tokens):
            result=await run_cli_chain(prompt=prompt,schema=json.dumps(schema),semantic_request_class=response_model,
                cursor_runner=run_cursor_agent_llm,grok_runner=lambda *_a,**_kw:pytest.fail('fallback'),
                invocations=calls,invocation_observer=observer,max_tokens=max_tokens,fallback_permitted=False)
            return CombinedTemporalTransportResult(raw=result,framework_version='fixture',model_version='composer-2.5',
                token_usage=calls[0]['usage'],provider_cost=None)
    class Pipeline(_Pipeline):
        async def _execute(self,**kwargs):pytest.fail('invalid identity reached graph mutation')
    leaf=asyncio.run(extract_combined_temporal_async(case.revision,transport=Transport(),pipeline=Pipeline()))
    assert calls[0]['response_quality']['schema_status']=='VALID'
    assert leaf.outcome is CombinedTemporalOutcome.TERMINAL_ATTEMPT_FAILURE
    assert len(runtime.requests)==1


def test_invalid_reported_counters_do_not_gain_the_advisory_exception(tmp_path,monkeypatch):
    from newsroom.graphiti_adapter.cli_client import CliResponseError
    service,observer,runtime,calls,run=_case(tmp_path,monkeypatch,
        usage=FakeUsage(70_691,20_287,cache_read_tokens=60_544,total_tokens=1))
    with pytest.raises(CliResponseError):run()
    terminal=service.terminal(calls[0]['model_invocation_id'])
    assert terminal.usage_status.value=='INVALID'
    assert service.route_state('GRAPHITI_CHAT_PRIMARY')['state']=='OPEN'
    assert calls[0]['usage']['total_tokens']==1


def test_actual_byte_breach_still_holds_the_native_v13_route(tmp_path,monkeypatch):
    from newsroom.graphiti_adapter.cli_client import CliResponseError
    from newsroom.graphiti_adapter.cursor_transport import cursor_output_limit
    service,observer,runtime,calls,run=_case(tmp_path,monkeypatch,
        text='x'*(cursor_output_limit(16_384)+1),usage=FakeUsage(70_691,20_287,cache_read_tokens=60_544))
    with pytest.raises(CliResponseError):run()
    assert calls[0]['sdk_terminal']['error_class']=='OUTPUT_BOUND'
    assert calls[0]['outcome']=='OUTPUT_LIMIT_EXCEEDED'
    assert service.route_state('GRAPHITI_CHAT_PRIMARY')['state']=='OPEN'


def test_new_protocol_policy_identity_and_bounds_are_exact(tmp_path,monkeypatch):
    from newsroom.control_plane.model_usage import native_sdk_reported_token_targets_are_advisory
    service,observer,runtime,calls,run=_case(tmp_path,monkeypatch,usage=FakeUsage(2,1))
    run();token=observer._allocations[0];policy=observer._policies[token.invocation_id]
    assert policy.command_semantic_version=='newsroom.graphiti-provider-dispatch.v13'
    assert (policy.max_prompt_bytes,policy.max_context_tokens,policy.max_output_tokens,policy.max_total_tokens)==(262_144,131_072,16_384,147_456)
    assert native_sdk_reported_token_targets_are_advisory(policy)
    for invalid in (replace(policy,command_semantic_version='newsroom.graphiti-provider-dispatch.v12'),
        replace(policy,model='grok-4.6'),replace(policy,route='GRAPHITI_CHAT_FALLBACK'),
        replace(policy,command_flags=tuple(x for x in policy.command_flags if not x.startswith('REPORTED_TOTAL_TOKENS=')))):
        assert not native_sdk_reported_token_targets_are_advisory(invalid)


@pytest.mark.parametrize('context', [None, 131_073])
def test_reconciled_future_usage_keeps_protocol_semantics_and_original_unknown(tmp_path,monkeypatch,context):
    from newsroom.control_plane.model_usage import UsageComponents
    service,observer,runtime,calls,run=_case(tmp_path,monkeypatch)
    run();ident=calls[0]['model_invocation_id'];original=service.terminal(ident)
    components=UsageComponents(input_tokens=70_691,output_tokens=20_287,cached_read_tokens=60_544,
        cached_write_tokens=0,reasoning_tokens=16_000,context_tokens=context,total_tokens=151_522,
        provenance='PROVIDER_REPORTED')
    service.reconcile(invocation_id=ident,components=components,provider_telemetry=components.as_record(),
        observed_at=original.observed_at+timedelta(seconds=1),raw_telemetry_pointer='fixture://exact-sdk-usage')
    assert service.terminal(ident).terminal_digest==original.terminal_digest
    assert service.terminal(ident).usage_status.value=='UNREPORTED'
    connection=service._connection()
    try:
        record=json.loads(connection.execute('SELECT record_json FROM model_usage_reconciliations WHERE invocation_id=?',(ident,)).fetchone()[0])
        assert record['components']==components.as_record()
        assert record['policy_breach']==('MAX_CONTEXT_TOKENS_EXCEEDED' if context else None)
    finally:connection.close()
    assert service.route_state('GRAPHITI_CHAT_PRIMARY')['state']==('OPEN' if context else 'CLOSED')


@pytest.mark.parametrize('streamed',[True,False])
def test_actual_utf8_ceiling_accepts_exact_boundary(monkeypatch,streamed):
    from newsroom.graphiti_adapter.cursor_transport import run_cursor_transport,cursor_output_limit
    cap=cursor_output_limit(1);text='€'*(cap//3)+'x'*(cap%3)
    assert len(text.encode())==65_600
    runtime=_bind(monkeypatch,FakeRuntime(run=FakeRun(messages=(_assistant(text),) if streamed else (),
        terminal=FakeTerminal(result=text,usage=FakeUsage(2,1)))))
    execution=run_cursor_transport(prompt='source',max_tokens=1,timeout=5,idempotency_key='sha256:'+'a'*64)
    assert execution.text==text and execution.cancelled is False
    assert runtime.run.cancel_count==0


@pytest.mark.parametrize('changes', [
    {'total_tokens':90_978}, {'total_tokens':167_522},
    {'cached_read_tokens':None,'total_tokens':90_978},
    {'cached_write_tokens':None}, {'input_tokens':None},
])
def test_v13_reconciliation_requires_exact_sdk_formula_atomically(tmp_path,monkeypatch,changes):
    from newsroom.control_plane.model_usage import UsageComponents,ModelUsageIntegrityError
    service,observer,runtime,calls,run=_case(tmp_path,monkeypatch)
    run();ident=calls[0]['model_invocation_id'];original=service.terminal(ident)
    values=dict(input_tokens=70_691,output_tokens=20_287,cached_read_tokens=60_544,
        cached_write_tokens=0,reasoning_tokens=16_000,context_tokens=None,total_tokens=151_522,
        provenance='PROVIDER_REPORTED')
    components=UsageComponents(**(values|changes))
    connection=service._connection()
    tables=('model_invocation_terminals','model_usage_current','model_provider_telemetry',
        'model_usage_reconciliations','model_usage_route_circuit_events','model_transport_observations')
    before={table:connection.execute(f'SELECT * FROM {table} ORDER BY rowid').fetchall() for table in tables}
    connection.close()
    with pytest.raises(ModelUsageIntegrityError):
        service.reconcile(invocation_id=ident,components=components,provider_telemetry=components.as_record(),
            observed_at=original.observed_at+timedelta(seconds=1),raw_telemetry_pointer='fixture://inconsistent-sdk-total')
    connection=service._connection()
    try:
        assert {table:connection.execute(f'SELECT * FROM {table} ORDER BY rowid').fetchall() for table in tables}==before
    finally:connection.close()
    assert service.terminal(ident).terminal_digest==original.terminal_digest
    assert service.terminal(ident).usage_status.value=='UNREPORTED'
    assert service.route_state('GRAPHITI_CHAT_PRIMARY')['state']=='OPEN'


@pytest.mark.parametrize('total',[90_978,167_522])
def test_fresh_v13_terminal_rejects_legacy_formula_aliases(tmp_path,monkeypatch,total):
    from newsroom.graphiti_adapter.cli_client import CliResponseError
    service,observer,runtime,calls,run=_case(tmp_path,monkeypatch,
        usage=FakeUsage(70_691,20_287,cache_read_tokens=60_544,reasoning_tokens=16_000,total_tokens=total))
    with pytest.raises(CliResponseError):run()
    terminal=service.terminal(calls[0]['model_invocation_id'])
    assert terminal.usage_status.value=='INVALID'
    assert terminal.failure_class=='REPORTED_COMPONENT_TOTAL_INVALID'
    assert terminal.components.total_tokens==total
    assert terminal.components.context_tokens is None
    assert service.route_state('GRAPHITI_CHAT_PRIMARY')['state']=='OPEN'


@pytest.mark.parametrize('total',[90_978,167_522])
def test_current_v13_reader_denies_resigned_incoherent_reconciliation(tmp_path,monkeypatch,total):
    from newsroom.control_plane.model_usage import UsageComponents,ModelUsageIntegrityError,_refresh_current_usage
    from newsroom.authority.canonical import digest_canonical,canonical_json_bytes
    service,observer,runtime,calls,run=_case(tmp_path,monkeypatch)
    run();ident=calls[0]['model_invocation_id'];original=service.terminal(ident)
    components=UsageComponents(input_tokens=70_691,output_tokens=20_287,cached_read_tokens=60_544,
        cached_write_tokens=0,reasoning_tokens=16_000,context_tokens=None,total_tokens=151_522,
        provenance='PROVIDER_REPORTED')
    service.reconcile(invocation_id=ident,components=components,provider_telemetry=components.as_record(),
        observed_at=original.observed_at+timedelta(seconds=1),raw_telemetry_pointer='fixture://exact-sdk-usage')
    connection=service._connection()
    try:
        digest,raw=connection.execute('SELECT reconciliation_digest,record_json FROM model_usage_reconciliations').fetchone()
        record=json.loads(raw);record['components']['total_tokens']=total
        record.pop('reconciliation_digest');record['reconciliation_digest']=digest_canonical(record)
        connection.execute('UPDATE model_usage_reconciliations SET reconciliation_digest=?,record_json=? WHERE reconciliation_digest=?',
            (record['reconciliation_digest'],canonical_json_bytes(record).decode(),digest));connection.commit()
        before=connection.execute('SELECT * FROM model_usage_current').fetchall()
        connection.execute('BEGIN IMMEDIATE')
        with pytest.raises(ModelUsageIntegrityError):_refresh_current_usage(connection,ident)
        connection.rollback()
        assert connection.execute('SELECT * FROM model_usage_current').fetchall()==before
    finally:connection.close()
