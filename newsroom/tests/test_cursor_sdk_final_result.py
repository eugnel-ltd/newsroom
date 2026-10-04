"""Final SDK answers, compact diagnostics and unchanged bounded failure handling."""
import json
from collections import Counter
import pytest
from newsroom.graphiti_adapter.cursor_transport import (
    CursorSdkError,CursorSdkBoundedFailure,CursorToolCallViolation,run_cursor_transport,
)
from newsroom.graphiti_adapter.cli_client import _parsed_object
from newsroom.graphiti_adapter.cli_process import validated_sdk_terminal
from newsroom.tests.test_graphiti_cursor_sdk_transport import (
    FakeRun,FakeTerminal,FakeRuntime,FakeMessage,FakeUsage,_assistant,_bind,
    _GRAPHITI_JSON,_TEST_IDEMPOTENCY_KEY,
)


def _run(monkeypatch,run,**changes):
    _bind(monkeypatch,FakeRuntime(run=run))
    return run_cursor_transport(prompt='extract',max_tokens=changes.get('max_tokens',64),
                                timeout=5,idempotency_key=_TEST_IDEMPOTENCY_KEY)


def _fragments(text,count):
    return tuple(_assistant(text[len(text)*i//count:len(text)*(i+1)//count])for i in range(count))


def test_finished_terminal_is_last_answer_after_seven_and_951_assistant_deltas(monkeypatch):
    # Exact retained class shape of105713, not a claim to recover its deleted text.
    initial='{"draft":1}'
    final='{'+' '*11409+'"entities":[],"entity_resolutions":[],"edges":[]}'
    usage=FakeUsage(11,7)
    messages=(FakeMessage('status'),*(FakeMessage('thinking')for _ in range(220)),
        *_fragments(initial,7),*(FakeMessage('thinking')for _ in range(68)),
        *_fragments(final,951),FakeMessage('usage',usage=usage),FakeMessage('status'))
    run=FakeRun(messages=messages,terminal=FakeTerminal(result=final,usage=usage))
    assert _parsed_object(initial+final) is None
    execution=_run(monkeypatch,run)
    assert execution.text==final
    assert _parsed_object(execution.text)==json.loads(_GRAPHITI_JSON)
    assert Counter(execution.stream_message_classes)=={'status':2,'thinking':288,'assistant':958,'usage':1}
    terminal=execution.terminal_record()
    assert terminal['stream_message_classes']==['status','thinking','assistant','usage']
    assert terminal['diagnostic_digest']==execution.diagnostic_digest
    assert validated_sdk_terminal(terminal)==terminal
    legacy={**terminal,'stream_message_classes':list(execution.stream_message_classes)}
    assert validated_sdk_terminal(legacy)==legacy
    assert len(json.dumps(terminal,separators=(',',':')).encode())<1500
    assert execution.usage['input_tokens']==11 and execution.usage['output_tokens']==7
    assert run.cancel_count==0 and run.close_count==1


@pytest.mark.parametrize('result',(None,'',{'unsupported':'typed non-string'}))
def test_terminal_missing_answer_keeps_existing_stream_only_fallback(monkeypatch,result):
    run=FakeRun(messages=(_assistant(_GRAPHITI_JSON),),terminal=FakeTerminal(result=result))
    assert _run(monkeypatch,run).text==_GRAPHITI_JSON


def test_externally_cancelled_valid_text_remains_failed_without_cancel_or_retry(monkeypatch):
    usage=FakeUsage(3,2)
    run=FakeRun(messages=(_assistant(_GRAPHITI_JSON),),
        terminal=FakeTerminal(status='cancelled',result=_GRAPHITI_JSON,usage=usage))
    with pytest.raises(CursorSdkError)as caught:_run(monkeypatch,run)
    assert caught.value.error_class=='CANCELLED'
    assert caught.value.execution.status=='cancelled'
    assert caught.value.execution.cancelled is True
    assert caught.value.usage['input_tokens']==3
    assert run.cancel_count==0 and run.close_count==1


@pytest.mark.parametrize('mode',('terminal_only','streamed'))
def test_output_bound_stays_strict_for_terminal_or_stream_before_acceptance(monkeypatch,mode):
    huge='x'*200_000
    run=FakeRun(messages=(_assistant(huge),)if mode=='streamed'else (),
        terminal=FakeTerminal(result=_GRAPHITI_JSON if mode=='streamed'else huge,usage=FakeUsage(2,1)))
    with pytest.raises(CursorSdkBoundedFailure)as caught:_run(monkeypatch,run,max_tokens=1)
    assert caught.value.error_class=='OUTPUT_BOUND'
    assert run.cancel_count==(1 if mode=='streamed'else 0)
    assert run.close_count==1


def test_tool_call_remains_stronger_than_valid_terminal_json(monkeypatch):
    run=FakeRun(messages=(FakeMessage('tool_call',call_id='one'),),
        terminal=FakeTerminal(result=_GRAPHITI_JSON,usage=FakeUsage(2,1)))
    with pytest.raises(CursorToolCallViolation):_run(monkeypatch,run)
    assert run.cancel_count==1 and run.close_count==1


def test_error_terminal_never_admits_its_valid_json(monkeypatch):
    run=FakeRun(messages=(_assistant('partial'),),
        terminal=FakeTerminal(status='error',result=_GRAPHITI_JSON,usage=FakeUsage(2,1)))
    with pytest.raises(CursorSdkError)as caught:_run(monkeypatch,run)
    assert caught.value.execution.status=='error'
    assert caught.value.execution.text!=''+_GRAPHITI_JSON
    assert run.cancel_count==0 and run.close_count==1
