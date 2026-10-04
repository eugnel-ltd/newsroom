"""Global verified setup refusal defers the native batch, not unrelated content."""
from dataclasses import replace
from types import SimpleNamespace
import json

import pytest

from newsroom.authority.canonical import canonical_json_bytes,digest_bytes
from newsroom.control_plane import cycle
from newsroom.tests.test_native_graphiti import _open,_native
from newsroom.tests.test_graphiti_corpus_ingest import _complete


def setup_failure(unit,*,code='BrokerError',state='NOT_DISPATCHED'):
    result=replace(_complete(unit,proposal_count=0,entity_count=0),outcome='FAILED',failure_code='PRODUCER_INTERNAL_ERROR')
    raw=dict(result.raw_receipt or {});raw.update(dispatch_state=state,setup_failure=code)
    raw.pop('raw_output_digest',None);raw['raw_output_digest']=digest_bytes(canonical_json_bytes(raw))
    return replace(result,receipt_digest=raw['raw_output_digest'],raw_receipt=raw)


def test_first_verified_broker_setup_failure_defers_remaining_and_next_cycle_recovers(tmp_path,monkeypatch):
    processor,connection,_=_open(tmp_path,monkeypatch,ingest=cycle._ingest)
    units=tuple(_native(str(n))for n in range(3));attempts=[];healthy=[False]
    def ingest(unit):
        attempts.append(unit.ingest_id)
        return _complete(unit,proposal_count=0,entity_count=0)if healthy[0]else setup_failure(unit)
    processor._runner=SimpleNamespace(ingest=ingest)
    try:
        outcomes=processor.advance(units,cycle_id='broker-failed')
        assert len(attempts)==1
        assert [r.state for r in outcomes].count('GRAPHITI_DEFERRED')==2
        assert all(r.reason=='SYSTEMIC_SETUP_UNAVAILABLE'for r in outcomes if r.state=='GRAPHITI_DEFERRED')
        assert connection.execute('SELECT count(*) FROM unpublished_graphiti_attempt_receipts').fetchone()[0]==1
        assert connection.execute("SELECT count(*) FROM ledger WHERE kind='GRAPHITI_SPEND_RESERVE'").fetchone()[0]==1
        receipt=json.loads(connection.execute('SELECT receipt_json FROM unpublished_graphiti_attempt_receipts').fetchone()[0])
        assert receipt['setup_failure']=='BrokerError' and receipt['dispatch_state']=='NOT_DISPATCHED'
        healthy[0]=True
        resumed=processor.advance(units,cycle_id='broker-healthy')
        assert all(r.state=='GRAPHITI_COMPLETE'for r in resumed)
        assert len(attempts)==4
        assert connection.execute('SELECT count(*) FROM unpublished_graphiti_ingest').fetchone()[0]==3
    finally:connection.close()


@pytest.mark.parametrize('code,state',(
    ('BrokerError','DISPATCHED'),('ContentError','NOT_DISPATCHED'),(None,'NOT_DISPATCHED'),
))
def test_unknown_after_dispatch_or_content_failure_never_defers_other_units(tmp_path,monkeypatch,code,state):
    processor,connection,_=_open(tmp_path,monkeypatch,ingest=cycle._ingest)
    units=tuple(_native(str(n))for n in range(3));attempts=[]
    def ingest(unit):
        attempts.append(unit.ingest_id)
        return setup_failure(unit,code=code,state=state)
    processor._runner=SimpleNamespace(ingest=ingest)
    try:
        outcomes=processor.advance(units,cycle_id='not-global')
        assert len(attempts)==3
        assert all(r.state!='GRAPHITI_DEFERRED'for r in outcomes)
    finally:connection.close()


def test_producer_broker_string_without_setup_record_does_not_defer(tmp_path,monkeypatch):
    processor,connection,_=_open(tmp_path,monkeypatch,ingest=cycle._ingest)
    units=tuple(_native(str(n))for n in range(3));attempts=[]
    def ingest(unit):
        attempts.append(unit.ingest_id)
        result=setup_failure(unit)
        raw=dict(result.raw_receipt);raw.pop('setup_failure');raw['producer_failure']='BrokerError'
        raw.pop('raw_output_digest');raw['raw_output_digest']=digest_bytes(canonical_json_bytes(raw))
        return replace(result,receipt_digest=raw['raw_output_digest'],raw_receipt=raw)
    processor._runner=SimpleNamespace(ingest=ingest)
    try:
        outcomes=processor.advance(units,cycle_id='content-broker-string')
        assert len(attempts)==3 and all(r.state!='GRAPHITI_DEFERRED'for r in outcomes)
    finally:connection.close()


def test_corrupt_exact_setup_receipt_never_becomes_global_defer(tmp_path,monkeypatch):
    processor,connection,_=_open(tmp_path,monkeypatch,ingest=cycle._ingest)
    units=tuple(_native(str(n))for n in range(3));attempts=[]
    original=cycle.insert_graphiti_attempt_receipt
    def corrupted(c,**values):
        digest=original(c,**values)
        c.execute('UPDATE unpublished_graphiti_attempt_receipts SET receipt_json=? WHERE ingest_id=? AND attempt_number=?',
            ('{}',values['ingest_id'],values['attempt_number']))
        return digest
    monkeypatch.setattr(cycle,'insert_graphiti_attempt_receipt',corrupted)
    processor._runner=SimpleNamespace(ingest=lambda unit:attempts.append(unit.ingest_id)or setup_failure(unit))
    try:
        outcomes=processor.advance(units,cycle_id='corrupt-setup-proof')
        assert len(attempts)==3 and all(r.state!='GRAPHITI_DEFERRED'for r in outcomes)
    finally:connection.close()


def test_pipeline_accepts_real_systemic_defer_and_preserves_pending_stage(tmp_path,monkeypatch):
    from newsroom.tests.test_native_pipeline import _open as pipeline_open
    pipeline,journal,journal_connection,_,calls,dispositions=pipeline_open(tmp_path,monkeypatch)
    processor,connection,_=_open(tmp_path,monkeypatch,ingest=cycle._ingest)
    units=tuple(_native(str(n))for n in range(3));healthy=[False];attempts=[]
    dispositions[0]=tuple(SimpleNamespace(source_id=u.source_id,status='READY',reason_code='RETAINED',units=(u,))for u in units)
    processor._runner=SimpleNamespace(ingest=lambda unit:attempts.append(unit.ingest_id)or(
        _complete(unit,proposal_count=0,entity_count=0)if healthy[0]else setup_failure(unit)))
    pipeline._graphiti=processor
    try:
        first=pipeline.tick(cycle_id='pipeline-broker-failed')
        assert len(attempts)==1
        untouched=[u for u in units if u.ingest_id not in attempts]
        assert all(journal.summary(u.revision_id)=={}for u in untouched)
        assert first.revision_states=={'GRAPHITI_HOLD':1,'QUEUED':2}
        assert not any(kind=='publish'for kind,_ in calls)
        healthy[0]=True
        second=pipeline.tick(cycle_id='pipeline-broker-recovered')
        assert second.revision_states=={'ACKNOWLEDGED':3}
        assert len(attempts)==4
    finally:connection.close();journal_connection.close()


@pytest.mark.parametrize('reason,receipt',(('UNREVIEWED_DEFER',None),('SYSTEMIC_SETUP_UNAVAILABLE','sha256:'+'a'*64)))
def test_pipeline_rejects_unknown_or_receipt_bearing_defer_without_publication(tmp_path,monkeypatch,reason,receipt):
    from newsroom.tests.test_native_pipeline import _open as pipeline_open
    from newsroom.control_plane.native_graphiti import NativeGraphitiOutcome
    pipeline,journal,connection,units,calls,_=pipeline_open(tmp_path,monkeypatch)
    pipeline._graphiti=SimpleNamespace(advance=lambda selected,**_:tuple(
        NativeGraphitiOutcome(u.ingest_id,'GRAPHITI_DEFERRED',receipt,reason)for u in selected))
    try:
        report=pipeline.tick(cycle_id='invalid-defer')
        assert report.revision_states=={'GRAPHITI_HOLD':2}
        assert all(journal.current(u.revision_id)['facts']['reason']=='ValueError'for u in units)
        assert not any(kind=='publish'for kind,_ in calls)
    finally:connection.close()


def test_noncanonical_numeric_setup_receipt_remains_per_unit_failure(tmp_path,monkeypatch):
    processor,connection,_=_open(tmp_path,monkeypatch,ingest=cycle._ingest)
    units=tuple(_native(str(n))for n in range(3));attempts=[]
    original=cycle.insert_graphiti_attempt_receipt
    def noncanonical(c,**values):
        digest=original(c,**values)
        row=c.execute('SELECT receipt_json FROM unpublished_graphiti_attempt_receipts WHERE ingest_id=? AND attempt_number=?',
            (values['ingest_id'],values['attempt_number'])).fetchone()
        receipt=json.loads(row[0]);receipt['invalid_number']=float('nan')
        c.execute('UPDATE unpublished_graphiti_attempt_receipts SET receipt_json=? WHERE ingest_id=? AND attempt_number=?',
            (json.dumps(receipt),values['ingest_id'],values['attempt_number']))
        return digest
    monkeypatch.setattr(cycle,'insert_graphiti_attempt_receipt',noncanonical)
    processor._runner=SimpleNamespace(ingest=lambda unit:attempts.append(unit.ingest_id)or setup_failure(unit))
    try:
        outcomes=processor.advance(units,cycle_id='noncanonical-setup-proof')
        assert len(attempts)==3 and all(r.state!='GRAPHITI_DEFERRED'for r in outcomes)
    finally:connection.close()
