"""Accounted output rejection isolates the failed unit, not its healthy peers."""
import json
from dataclasses import asdict
from datetime import timedelta
from types import SimpleNamespace

import pytest

from newsroom.authority.canonical import digest_canonical
from newsroom.control_plane import model_usage as m
from newsroom.control_plane.cycle import _graphiti_usage_cycle_id
from newsroom.control_plane.graphiti import GraphitiModelUsageObserver
from newsroom.control_plane.graphiti_fallback_policy import load_checked_native_graphiti_fallback_circuit_policy
from newsroom.control_plane.graphiti_requests import load_checked_native_graphiti_call_shape_policy
from newsroom.control_plane.native_progress import NativeRevisionJournal
from newsroom.control_plane.store import connect, insert_graphiti_attempt_receipt, reserve_graphiti_spend, reconcile_graphiti_spend
from newsroom.tests.test_native_graphiti import _native
from newsroom.tests.test_native_fallback_cancellation_disposition import EXTRACTED_ENTITIES_SCHEMA, T0, _rewrite_receipt

ROUTE = 'GRAPHITI_CHAT_PRIMARY'


def _failed(tmp_path, monkeypatch, *, output=20_336, total=None, context=None,
            reported=True, outcome='OUTPUT_LIMIT_EXCEEDED', settled=True, sdk_status='finished', active_peer=False):
    monkeypatch.setattr('newsroom.control_plane.graphiti._graphiti_implementation_identity', lambda: ('a' * 40, True))
    path = str(tmp_path/'private.sqlite3'); connection = connect(path)
    unit = _native('reported-output-rejection'); journal = NativeRevisionJournal(connection); journal.land((unit,))
    usage = m.ModelUsageService(path)
    envelope = m.WorkEnvelope.create(
        cycle_id=_graphiti_usage_cycle_id(unit, attempt_number=1, requested_cycle_id=None),
        workload_class=m.WorkloadClass.GRAPHITI_CHAT_PRIMARY, admitted_at=T0,
        admission_decision_id=None, candidate_id=None, hypothesis_digest=None, evidence_package_digest=None,
        ingest_id=unit.ingest_id, graphiti_attempt_id=f'{unit.ingest_id}:1',
    )
    usage.open_envelope(envelope)
    observer = GraphitiModelUsageObserver(
        service=usage, envelope=envelope, clock=lambda: T0+timedelta(seconds=10),
        owner_stop_check=lambda: None, deadline=T0+timedelta(minutes=3),
        effective_revision_digest=digest_canonical(asdict(unit.effective_revision)),
        ingest_obligation_id=unit.ingest_id,
        call_shape_policy=load_checked_native_graphiti_call_shape_policy(),
        fallback_policy=load_checked_native_graphiti_fallback_circuit_policy(),
    )
    allocation = observer.before_cli_invocation(
        provider='cursor-agent-cli', model='composer-2.5', prompt='source-safe prompt',
        schema=EXTRACTED_ENTITIES_SCHEMA, semantic_request_class='ExtractedEntities', max_tokens=16_384,
    )
    peer_observer = peer_allocation = None
    if active_peer:
        peer_envelope = m.WorkEnvelope.create(
            cycle_id=_graphiti_usage_cycle_id(unit, attempt_number=2, requested_cycle_id=None),
            workload_class=m.WorkloadClass.GRAPHITI_CHAT_PRIMARY, admitted_at=T0,
            admission_decision_id=None, candidate_id=None, hypothesis_digest=None,
            evidence_package_digest=None, ingest_id=unit.ingest_id,
            graphiti_attempt_id=f'{unit.ingest_id}:2',
        )
        usage.open_envelope(peer_envelope)
        peer_observer = GraphitiModelUsageObserver(
            service=usage, envelope=peer_envelope, clock=lambda: T0+timedelta(seconds=10),
            owner_stop_check=lambda: None, deadline=T0+timedelta(minutes=3),
            effective_revision_digest=digest_canonical(asdict(unit.effective_revision)),
            ingest_obligation_id=unit.ingest_id,
            call_shape_policy=load_checked_native_graphiti_call_shape_policy(),
            fallback_policy=load_checked_native_graphiti_fallback_circuit_policy(),
        )
        peer_allocation = peer_observer.before_cli_invocation(
            provider='cursor-agent-cli', model='composer-2.5', prompt='other source-safe prompt',
            schema=EXTRACTED_ENTITIES_SCHEMA, semantic_request_class='ExtractedEntities', max_tokens=16_384,
        )
        peer_observer.transport_dispatch_started(peer_allocation)
    observer.transport_dispatch_started(allocation)
    telemetry = dict(usage_basis='PROVIDER_REPORTED', input_tokens=41_710, output_tokens=output,
                     cached_read_tokens=25_376, cached_write_tokens=0, reasoning_tokens=None,
                     total_tokens=67_086+output if total is None else total)
    if context is not None: telemetry['context_tokens'] = context
    binding = observer.after_cli_invocation(allocation, outcome=outcome,
        usage=telemetry if reported else {'usage_basis': 'UNREPORTED'})
    terminal = usage.terminal(allocation.invocation_id)
    sdk = dict(schema_version='newsroom.cursor-sdk-terminal.v1', agent_id='agent-fixture', run_id='run-fixture',
               request_id='UNOBSERVED', status=sdk_status, cancelled=sdk_status=='cancelled',
               error_class='NONE', error_code='NONE', tool_call_count=0,
               resolved_model=allocation.model, duration_ms=1000, stream_message_classes=['assistant','usage','status'])
    from newsroom.graphiti_adapter.cursor_transport import _diagnostic_digest
    sdk['diagnostic_digest'] = _diagnostic_digest(**{key: sdk[key] for key in (
        'status','error_class','error_code','tool_call_count','cancelled','duration_ms',
    )})
    spend_id = f'{unit.ingest_id}:1'
    reserve_graphiti_spend(connection, spend_id=spend_id, ingest_id=unit.ingest_id, attempt_number=1,
                          proving_run_id=unit.proving_run_id, generation_id='fixture',
                          reserved_gbp_microunits=500_000, ceiling_gbp_microunits=None)
    accounting = reconcile_graphiti_spend(connection, spend_id=spend_id, embedding_usage=(
        dict(usage_basis='NO_EMBEDDING_CALL', request_count=0, requests=[], embedding_tokens=0, cost_usd_microunits=0)
        if settled else {'usage_basis':'UNREPORTED'}))
    receipt = dict(ingest_id=unit.ingest_id, attempt_number=1, outcome='FAILED',
                   failure_code='PRODUCER_INTERNAL_ERROR', accounting=accounting,
                   chat_invocations=[{**binding, 'outcome': outcome, 'provider': allocation.provider,
                                      'model': allocation.model, 'requested_max_tokens': allocation.max_output_tokens,
                                      'sdk_terminal': sdk, 'sdk_run_id': sdk['run_id'], 'sdk_agent_id': sdk['agent_id'],
                                      'usage': telemetry}])
    receipt_digest = insert_graphiti_attempt_receipt(connection, ingest_id=unit.ingest_id, attempt_number=1,
                                                    outcome='FAILED', receipt=receipt)
    connection.commit()
    usage.record_work_outcome(envelope_id=envelope.envelope_id, outcome='GRAPHITI_FAILED',
                             outcome_record_id=receipt_digest, payload_digest=None, terminal_at=T0+timedelta(seconds=11))
    return SimpleNamespace(connection=connection, usage=usage, unit=unit, journal=journal,
                           allocation=allocation, terminal=terminal, envelope=envelope,
                           peer_observer=peer_observer, peer_allocation=peer_allocation)


def _dispose(case, **changes):
    args = dict(invocation_id=case.allocation.invocation_id, revision_id=case.unit.revision_id,
                expected_terminal_digest=case.terminal.terminal_digest,
                expected_allocation_digest=case.allocation.canonical_digest, observed_at=T0+timedelta(seconds=12))
    return case.usage.disposition_native_reported_output_rejection(**(args | changes))


def _history(case):
    return {table: case.connection.execute(f'SELECT * FROM {table}').fetchall() for table in (
        'model_invocation_allocations','model_invocation_terminals','model_provider_telemetry',
        'model_work_outcomes','unpublished_graphiti_attempt_receipts','unpublished_graphiti_spend',
        'model_usage_reconciliations','model_usage_conservative_dispositions',
    )}


def test_reported_output_rejection_keeps_failure_and_usage_then_releases_only_primary(tmp_path, monkeypatch):
    case = _failed(tmp_path, monkeypatch)
    try:
        case.usage.open_route_circuit(route='GRAPHITI_CHAT_FALLBACK', reason='MISSING_PROVIDER_TELEMETRY',
                                      invocation_id=None, recorded_at=T0+timedelta(seconds=11))
        before = _history(case)
        assert case.usage.route_state(ROUTE)['state'] == 'OPEN'
        record = _dispose(case)
        assert record['components']['output_tokens'] == 20_336
        assert record['requested_max_output_tokens'] == 16_384
        assert record['retry_authorised'] is False and record['unknown_spend_released'] is False
        assert _history(case) == before
        assert case.usage.route_state(ROUTE)['state'] == 'CLOSED'
        assert case.usage.route_state('GRAPHITI_CHAT_FALLBACK')['state'] == 'OPEN'
        assert _dispose(case) == record
        assert case.connection.execute('SELECT count(*) FROM model_usage_reported_output_dispositions').fetchone() == (1,)
        assert case.connection.execute("SELECT count(*) FROM model_usage_route_circuit_events WHERE state='CLOSED'").fetchone() == (1,)
    finally:
        case.connection.close()


@pytest.mark.parametrize('changes', [
    {'reported': False}, {'outcome': 'CANCELLED'}, {'output': 16_384},
    {'context': 131_073}, {'total': 200_000, 'output': 132_914},
    {'total': 140_000, 'output': 72_914}, {'total': 87_423},
    {'settled': False}, {'sdk_status': 'cancelled'},
])
def test_other_failures_and_uncertainty_remain_blocked(tmp_path, monkeypatch, changes):
    case = _failed(tmp_path, monkeypatch, **changes)
    try:
        before = _history(case)
        with pytest.raises((m.ModelUsageAdmissionError, m.ModelUsageIntegrityError)):
            _dispose(case)
        assert _history(case) == before
        assert case.usage.route_state(ROUTE)['state'] == 'OPEN'
        assert case.connection.execute('SELECT count(*) FROM model_usage_reported_output_dispositions').fetchone() == (0,)
    finally:
        case.connection.close()


@pytest.mark.parametrize('defect', [
    'revision', 'terminal-digest', 'allocation-digest', 'receipt-failure',
    'receipt-sdk-status', 'receipt-sdk-run', 'receipt-output', 'receipt-basis', 'receipt-allocation',
    'receipt-accounting', 'telemetry', 'dispatch',
])
def test_changed_authority_fails_closed(tmp_path, monkeypatch, defect):
    case = _failed(tmp_path, monkeypatch)
    try:
        if defect.startswith('receipt-'):
            def change(receipt):
                leaf = receipt['chat_invocations'][0]
                if defect == 'receipt-failure':
                    receipt['failure_code'] = 'OTHER_FAILURE'
                elif defect == 'receipt-sdk-status':
                    leaf['sdk_terminal']['status'] = 'failed'
                elif defect == 'receipt-sdk-run':
                    leaf['sdk_run_id'] = 'different'
                elif defect == 'receipt-output':
                    leaf['usage']['output_tokens'] += 1
                elif defect == 'receipt-basis':
                    leaf['usage'] = {'usage_basis': 'UNREPORTED', 'provider_telemetry': leaf['usage']}
                elif defect == 'receipt-allocation':
                    leaf['model_invocation_allocation_digest'] = 'different'
                else:
                    receipt['accounting']['actual_gbp_microunits'] += 1
            _rewrite_receipt(case, change)
        elif defect == 'telemetry':
            case.connection.execute("UPDATE model_provider_telemetry SET record_json='{}'")
        elif defect == 'dispatch':
            case.connection.execute('DELETE FROM model_transport_observations')
        case.connection.commit()
        args = {'revision_id': 'different'} if defect == 'revision' else (
            {'expected_terminal_digest': 'different'} if defect == 'terminal-digest' else (
                {'expected_allocation_digest': 'different'} if defect == 'allocation-digest' else {}))
        with pytest.raises((m.ModelUsageAdmissionError, m.ModelUsageIntegrityError, ValueError, KeyError)):
            _dispose(case, **args)
        assert case.usage.route_state(ROUTE)['state'] == 'OPEN'
        assert case.connection.execute('SELECT count(*) FROM model_usage_reported_output_dispositions').fetchone() == (0,)
    finally:
        case.connection.close()


def test_reopen_reads_current_disposition_without_duplicate_closure_and_replay_rechecks(tmp_path, monkeypatch):
    case = _failed(tmp_path, monkeypatch)
    try:
        record = _dispose(case)
        reopened = m.ModelUsageService(case.usage.path)
        assert reopened.route_state(ROUTE)['state'] == 'CLOSED'
        assert _dispose(case) == record
        assert case.connection.execute("SELECT count(*) FROM model_usage_route_circuit_events WHERE state='CLOSED'").fetchone() == (1,)
        case.connection.execute("UPDATE model_usage_reported_output_dispositions SET record_json='{}'")
        case.connection.commit()
        assert reopened.route_state(ROUTE)['state'] == 'CLOSED'
        with pytest.raises((m.ModelUsageIntegrityError, ValueError, KeyError)):
            _dispose(case)
    finally:
        case.connection.close()


def test_unrelated_open_primary_circuit_is_not_closed(tmp_path, monkeypatch):
    case = _failed(tmp_path, monkeypatch)
    try:
        case.usage.open_route_circuit(route=ROUTE, reason='QUOTA', invocation_id=None,
                                      recorded_at=T0+timedelta(seconds=11))
        _dispose(case)
        assert case.usage.route_state(ROUTE)['state'] == 'OPEN'
        assert case.usage.route_state(ROUTE)['reason'] == 'QUOTA'
        assert case.connection.execute("SELECT count(*) FROM model_usage_route_circuit_events WHERE state='CLOSED'").fetchone() == (0,)
    finally:
        case.connection.close()


def test_active_primary_peer_prevents_route_closure(tmp_path, monkeypatch):
    case = _failed(tmp_path, monkeypatch, active_peer=True)
    try:
        _dispose(case)
        assert case.usage.route_state(ROUTE)['state'] == 'OPEN'
        assert case.connection.execute('SELECT count(*) FROM model_usage_reported_output_dispositions').fetchone() == (1,)
        assert case.connection.execute("SELECT count(*) FROM model_usage_route_circuit_events WHERE state='CLOSED'").fetchone() == (0,)
    finally:
        case.connection.close()


def test_processor_settlement_hook_is_current_and_stop_checked(tmp_path, monkeypatch):
    from newsroom.control_plane.native_graphiti import NativeGraphitiProcessor

    case = _failed(tmp_path, monkeypatch)
    try:
        processor = object.__new__(NativeGraphitiProcessor)
        processor._connection = case.connection
        processor._usage = case.usage
        processor._clock = lambda: T0+timedelta(seconds=12)
        checks = []
        processor._stop_check = lambda: checks.append('checked')
        processor._settle_missing_subscription_usage((_native('unrelated'),))
        assert checks == []
        assert case.usage.route_state(ROUTE)['state'] == 'OPEN'
        before = _history(case)
        processor._settle_missing_subscription_usage((case.unit,))
        assert checks == ['checked']
        assert case.usage.route_state(ROUTE)['state'] == 'CLOSED'
        assert _history(case) == before
        processor._settle_missing_subscription_usage((case.unit,))
        assert checks == ['checked']
        assert case.connection.execute('SELECT count(*) FROM model_usage_reported_output_dispositions').fetchone() == (1,)
        assert case.connection.execute("SELECT count(*) FROM model_usage_route_circuit_events WHERE state='CLOSED'").fetchone() == (1,)
    finally:
        case.connection.close()


def test_processor_stop_prevents_disposition(tmp_path, monkeypatch):
    from newsroom.control_plane.native_graphiti import NativeGraphitiProcessor

    case = _failed(tmp_path, monkeypatch)
    try:
        processor = object.__new__(NativeGraphitiProcessor)
        processor._connection = case.connection
        processor._usage = case.usage
        processor._clock = lambda: T0+timedelta(seconds=12)
        def stop():
            raise RuntimeError('stop')
        processor._stop_check = stop
        with pytest.raises(RuntimeError, match='stop'):
            processor._settle_missing_subscription_usage((case.unit,))
        assert case.usage.route_state(ROUTE)['state'] == 'OPEN'
        assert case.connection.execute('SELECT count(*) FROM model_usage_reported_output_dispositions').fetchone() == (0,)
    finally:
        case.connection.close()


@pytest.mark.parametrize('peer_kind', ['unknown', 'context'])
def test_other_terminal_usage_blockers_survive_exact_output_disposition(tmp_path, monkeypatch, peer_kind):
    case = _failed(tmp_path, monkeypatch, active_peer=True)
    try:
        usage = ({'usage_basis': 'UNREPORTED'} if peer_kind == 'unknown' else {
            'usage_basis': 'PROVIDER_REPORTED', 'input_tokens': 1, 'output_tokens': 1,
            'cached_read_tokens': 0, 'cached_write_tokens': 0, 'total_tokens': 2,
            'context_tokens': 131_073,
        })
        case.peer_observer.after_cli_invocation(case.peer_allocation, outcome='FAILED', usage=usage)
        case.usage.open_route_circuit(route=ROUTE, reason='CONTEXT_OUTPUT_BREACH',
                                      invocation_id=case.allocation.invocation_id,
                                      recorded_at=T0+timedelta(seconds=11))
        before = _history(case)
        _dispose(case)
        assert _history(case) == before
        assert case.usage.route_state(ROUTE)['state'] == 'OPEN'
        assert ROUTE in m._usage_blocking_routes(case.connection)
    finally:
        case.connection.close()


def test_disposition_rechecks_conflicting_landing_via_scoped_index(tmp_path, monkeypatch):
    from newsroom.control_plane.store import append_ledger
    case = _failed(tmp_path, monkeypatch)
    try:
        _dispose(case)
        plan = case.connection.execute(
            "EXPLAIN QUERY PLAN SELECT payload_digest,payload_json FROM ledger WHERE kind=? "
            "AND json_extract(payload_json,'$.revision_id')=?",
            ('NATIVE_REVISION_LANDED', case.unit.revision_id),
        ).fetchall()
        assert any('model_usage_native_landed_revision' in row[-1] for row in plan)
        raw, = case.connection.execute("SELECT payload_json FROM ledger WHERE kind='NATIVE_REVISION_LANDED'").fetchone()
        landing = json.loads(raw)
        landing['units'][0]['observed_at'] = '2026-09-01T00:00:00Z'
        append_ledger(case.connection, 'NATIVE_REVISION_LANDED', landing)
        case.connection.commit()
        assert case.usage.route_state(ROUTE)['state'] == 'CLOSED'
        with pytest.raises(m.ModelUsageIntegrityError):
            _dispose(case)
    finally:
        case.connection.close()


def test_reported_nested_receipt_telemetry_keeps_valid_binding(tmp_path, monkeypatch):
    case = _failed(tmp_path, monkeypatch)
    try:
        def nest(receipt):
            leaf = receipt['chat_invocations'][0]
            leaf['usage'] = {'usage_basis': 'PROVIDER_REPORTED', 'provider_telemetry': leaf['usage']}
        _rewrite_receipt(case, nest)
        case.connection.commit()
        before = _history(case)
        _dispose(case)
        assert _history(case) == before
        assert case.usage.route_state(ROUTE)['state'] == 'CLOSED'
    finally:
        case.connection.close()


@pytest.mark.parametrize('authority_failed', [False, True])
def test_full_advance_excludes_disposed_failed_ingest_but_dispatches_healthy_peer(tmp_path, monkeypatch, authority_failed):
    from newsroom.tests.test_native_graphiti import _open
    from newsroom.control_plane.store import record_graphiti_failure
    from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome
    from newsroom.graphiti_adapter.identity import typed_id
    from newsroom.extraction.types import ExtractionRunId

    case = _failed(tmp_path, monkeypatch)
    queued = []
    processor, connection, _calls = _open(
        tmp_path, monkeypatch,
        ingest=lambda _connection, **kw: queued.extend(unit.ingest_id for unit in kw['units']),
    )
    peer = _native('healthy-peer')
    try:
        _dispose(case)
        record_graphiti_failure(connection, ingest_id=case.unit.ingest_id,
                                source_id=case.unit.source_id, item_key=case.unit.item_key,
                                outcome='FAILED', failure_code='PRODUCER_INTERNAL_ERROR')
        connection.commit()
        bad_run = typed_id(ExtractionRunId, 'run', case.unit.ingest_id)
        processor._system.graphiti = SimpleNamespace(attempt_history=lambda run, **_kw: (
            (SimpleNamespace(outcome=GraphitiAdapterOutcome.FAILED,
                             failure_code='PRODUCER_INTERNAL_ERROR', attempt_number=1),)
            if authority_failed and run == bad_run else ()
        ))
        assert GraphitiAdapterOutcome.FAILED.terminal is False
        before = _history(case)
        for cycle in ('first', 'repeat'):
            result = {item.ingest_id: item for item in processor.advance((case.unit, peer), cycle_id=cycle)}
            assert result[case.unit.ingest_id].state == 'GRAPHITI_HOLD'
            assert result[case.unit.ingest_id].reason == 'REPORTED_OUTPUT_REJECTION_NO_RETRY'
        assert case.unit.ingest_id not in queued
        assert queued == [peer.ingest_id, peer.ingest_id]
        assert _history(case) == before
    finally:
        connection.close()
        case.connection.close()
