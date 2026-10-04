"""Fixed setup codes distinguish credential read failures without secret data."""
import json
from types import SimpleNamespace
import subprocess

import pytest

from newsroom.control_plane import broker
from newsroom.graphiti_adapter.types import graphiti_setup_failure_detail,GraphitiAdapterContractError,GRAPHITI_EXTRA_REQUIRED,GRAPHITI_CORE_RELEASE_MISMATCH


@pytest.mark.parametrize('account,service,label',(
    (broker.OPENROUTER_ACCOUNT,broker.OPENROUTER_SERVICE,'OPENROUTER'),
    (broker.NEO4J_ACCOUNT,broker.NEO4J_SERVICE,'NEO4J_COMMUNITY'),
    (broker.NEO4J_PROJECTOR_ACCOUNT,broker.NEO4J_PROJECTOR_SERVICE,'NEO4J_PROJECTOR'),
))
@pytest.mark.parametrize('outcome',('LOOKUP_TIMEOUT','LOOKUP_FAILED','LOOKUP_UNAVAILABLE','EMPTY_OR_TOO_SHORT'))
def test_known_keychain_failure_propagates_fixed_class_and_outcome(monkeypatch,account,service,label,outcome):
    def run(*args,**kwargs):
        assert kwargs['timeout']==10
        if outcome=='LOOKUP_TIMEOUT':raise subprocess.TimeoutExpired('private-command',10,output=b'private-secret')
        if outcome=='LOOKUP_FAILED':raise OSError('private-secret-and-path')
        return SimpleNamespace(returncode=3 if outcome=='LOOKUP_UNAVAILABLE'else 0,
            stdout='private-short',stderr='private-sensitive-error')
    monkeypatch.setattr(broker.subprocess,'run',run)
    with pytest.raises(broker.BrokerError) as raised:
        broker._keychain_password(account=account,service=service)
    code='BROKER_'+label+'_'+outcome
    assert raised.value.reason_code==code
    assert graphiti_setup_failure_detail(raised.value)==code
    assert 'private-' not in str(raised.value)
    assert 'private-' not in code and 'ABSENT'not in code


def test_generic_old_broker_error_has_no_detail_and_dependency_codes_unchanged():
    assert graphiti_setup_failure_detail(broker.BrokerError('legacy fixed failure')) is None
    for code in (GRAPHITI_EXTRA_REQUIRED,GRAPHITI_CORE_RELEASE_MISMATCH):
        assert graphiti_setup_failure_detail(GraphitiAdapterContractError('dependency failure',reason_code=code))==code


@pytest.mark.parametrize('account,service',(
    ('unknown-private-account','unknown-private-service'),
    (broker.OPENROUTER_ACCOUNT,'wrong-service'),
))
def test_unknown_account_service_tuple_has_no_typed_detail(monkeypatch,account,service):
    monkeypatch.setattr(broker.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout='',stderr='secret'))
    with pytest.raises(broker.BrokerError) as raised:
        broker._keychain_password(account=account,service=service)
    assert raised.value.reason_code is None
    assert graphiti_setup_failure_detail(raised.value)is None


def test_unicode_failure_and_success_leave_secret_out_of_typed_metadata(monkeypatch):
    def broken(*a,**k):raise UnicodeDecodeError('utf8',b'private-output',0,1,'private reason')
    monkeypatch.setattr(broker.subprocess,'run',broken)
    with pytest.raises(broker.BrokerError) as raised:broker.openrouter_api_key()
    assert raised.value.reason_code=='BROKER_OPENROUTER_LOOKUP_FAILED'
    assert str(raised.value)=='Keychain class OPENROUTER_API lookup failed'
    monkeypatch.setattr(broker.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='x'*32+'\n',stderr='private'))
    assert broker.openrouter_api_key()=='x'*32


def test_broker_code_cannot_be_invented_or_used_as_dependency_error():
    with pytest.raises(ValueError,match='allow-listed'):
        broker.BrokerError('fixed',reason_code='TOKEN=private')
    with pytest.raises(ValueError,match='allow-listed'):
        GraphitiAdapterContractError('fixed',reason_code='BROKER_OPENROUTER_LOOKUP_TIMEOUT')


@pytest.mark.parametrize('code',('BROKER_OPENROUTER_LOOKUP_UNAVAILABLE',None))
def test_actual_producer_receipt_retains_only_fixed_broker_detail(monkeypatch,code):
    import json
    import newsroom.graphiti_adapter.real as real
    from newsroom.graphiti_adapter.real import RealGraphitiAdapter
    from newsroom.graphiti_adapter.evaluation_attempt import evaluation_attempt_for
    from newsroom.authority.types import UtcTimestamp
    from newsroom.extraction.types import ExtractionOutcome
    monkeypatch.setattr(real,'_load_graphiti',lambda:SimpleNamespace())
    def unavailable():raise broker.BrokerError('TOKEN=private must not appear',reason_code=code)
    monkeypatch.setattr(real,'openrouter_api_key',unavailable)
    monkeypatch.setattr(real,'neo4j_community_password',lambda:pytest.fail('additional credential read'))
    result=RealGraphitiAdapter()._produce(evaluation_attempt_for(('Retained exact source passage.',)),
        UtcTimestamp.parse('2026-08-20T00:00:00.000000Z'))
    receipt=result.attempt_receipt_value
    assert result.outcome is ExtractionOutcome.RETRYABLE_FAILURE
    assert receipt['setup_failure']=='BrokerError' and receipt['dispatch_state']=='NOT_DISPATCHED'
    assert receipt.get('setup_failure_detail')==code
    assert receipt['chat_invocation_count']==0 and receipt['embedding_usage']['request_count']==0
    assert 'TOKEN='not in json.dumps(receipt)


@pytest.mark.parametrize('detail,expected',(
    ('BROKER_OPENROUTER_LOOKUP_UNAVAILABLE','BROKER_OPENROUTER_LOOKUP_UNAVAILABLE'),
    ('UNREVIEWED_BROKER_PRIVATE_DETAIL',None),
))
def test_durable_cycle_receipt_validates_broker_detail_union(tmp_path,monkeypatch,detail,expected):
    from dataclasses import replace
    from newsroom.authority.canonical import canonical_json_bytes,digest_bytes
    from newsroom.control_plane import cycle
    from newsroom.tests.test_native_graphiti import _open,_native
    from newsroom.tests.test_native_graphiti_systemic_defer import setup_failure
    processor,connection,_=_open(tmp_path,monkeypatch,ingest=cycle._ingest)
    unit=_native('broker-detail');calls=[]
    def produce(selected):
        calls.append(selected.ingest_id)
        result=setup_failure(selected)
        raw=dict(result.raw_receipt);raw['setup_failure_detail']=detail
        raw.pop('raw_output_digest');raw['raw_output_digest']=digest_bytes(canonical_json_bytes(raw))
        return replace(result,receipt_digest=raw['raw_output_digest'],raw_receipt=raw)
    processor._runner=SimpleNamespace(ingest=produce)
    try:
        if expected is None:
            with pytest.raises(ValueError,match='setup failure detail is invalid'):
                processor.advance((unit,),cycle_id='broker-detail-retention')
            assert connection.execute('SELECT count(*) FROM unpublished_graphiti_attempt_receipts').fetchone()[0]==0
            assert calls==[unit.ingest_id]
            return
        processor.advance((unit,),cycle_id='broker-detail-retention')
        row=connection.execute('SELECT outcome,receipt_digest,receipt_json FROM unpublished_graphiti_attempt_receipts WHERE ingest_id=? AND attempt_number=1',
            (unit.ingest_id,)).fetchone()
        receipt=json.loads(row[2]);retained=receipt.pop('receipt_digest')
        assert retained==row[1]==digest_bytes(canonical_json_bytes(receipt))
        assert receipt['ingest_id']==unit.ingest_id and row[0]==receipt['outcome']=='FAILED'
        assert receipt['setup_failure_detail']==expected
        assert receipt['dispatch_state']=='NOT_DISPATCHED'
        assert cycle._validated_setup_failure_detail(receipt['setup_failure_detail'])==expected
        assert calls==[unit.ingest_id]
    finally:connection.close()
