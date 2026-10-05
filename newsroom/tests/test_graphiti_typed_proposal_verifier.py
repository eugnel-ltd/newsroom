"""Typed semantic verification stays before all business graph mutation."""
from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from newsroom.control_plane.graphiti import EvaluationGraphitiRunner
from newsroom.control_plane.model_usage import ModelUsageService, native_graphiti_usage_cycle_id
from newsroom.graphiti_adapter import real
from newsroom.graphiti_adapter.combined_temporal_types import CombinedTemporalError, CombinedTemporalFailureCode
from newsroom.graphiti_adapter.evaluation_attempt import evaluation_attempt_for
from newsroom.graphiti_adapter.neo4j_guard import GuardState
from newsroom.authority.types import UtcTimestamp
from newsroom.extraction.types import ExtractionContractError
from newsroom.tests.test_graphiti_episode_after_validation import _run
from newsroom.tests.test_graphiti_governed_runner import _unit


def _with_verifier(monkeypatch, verifier):
    original = real._add_episode
    observer = SimpleNamespace(verify_typed_proposals=verifier)
    monkeypatch.setattr(real, '_add_episode', lambda **values: original(**values, invocation_observer=observer))


@pytest.mark.parametrize('case_name', ['pair-current', 'explicit-valid-at', 'null-temporal'])
def test_full_typed_verification_precedes_episode_and_resolution(monkeypatch, case_name):
    calls = []
    def verify(*, source_revision, proposal_receipt):
        assert seen.business == {}
        seen.events.append('verify')
        assert proposal_receipt['wire_payload']['facts']
        assert proposal_receipt['evidence_passages']
        calls.append((source_revision, copy.deepcopy(proposal_receipt)))
        # Detached inputs must not be able to replace accepted facts.
        proposal_receipt['wire_payload']['facts'].clear()
        return {'receipt_admission_id': 'fixture-retained-judgment'}

    _with_verifier(monkeypatch, verify)
    call, seen, revision, case = _run(monkeypatch, case_name=case_name)
    result = asyncio.run(call)
    assert len(calls) == 1 and calls[0][0] == revision
    assert seen.events.index('model') < seen.events.index('verify') < seen.events.index('episode')
    assert [str(edge.fact) for edge in result.edges] == [item['fact'] for item in case.gold['facts']]
    assert seen.snapshots[0]['combined']['typed_proposal_verification'] == {'receipt_admission_id': 'fixture-retained-judgment'}


@pytest.mark.parametrize('failure', ['semantic', 'unknown'])
def test_verifier_failure_has_no_business_mutation_and_retains_chat_receipt(monkeypatch, failure):
    def verify(**values):
        seen.events.append('verify')
        assert seen.business == {}
        if failure == 'semantic':
            raise CombinedTemporalError(CombinedTemporalFailureCode.EVIDENCE_UNRESOLVED, 'fixture unsupported')
        raise TimeoutError('fixture unknown judgment transport')

    _with_verifier(monkeypatch, verify)
    call, seen, _revision, _case = _run(monkeypatch)
    with pytest.raises(ExtractionContractError):
        asyncio.run(call)
    assert seen.business == {} and seen.rollback == 0
    assert seen.model_calls == 1 and 'episode' not in seen.events and 'context' not in seen.events
    receipt = seen.snapshots[0]['combined']
    assert receipt['raw_output_digest'] and len(receipt['transport_calls']) == 1
    assert receipt['failure_code'] == ('EVIDENCE_UNRESOLVED' if failure == 'semantic' else 'PIPELINE_FAILED')


def test_empty_verification_is_not_skipped_before_episode(monkeypatch):
    def verify(*, source_revision, proposal_receipt):
        assert proposal_receipt['wire_payload'] == {'entities': [], 'facts': []}
        assert seen.business == {}
        seen.events.append('verify')
        return {'receipt_admission_id': 'empty-judgment'}
    _with_verifier(monkeypatch, verify)
    call, seen, _revision, _case = _run(monkeypatch, empty=True)
    asyncio.run(call)
    assert seen.events.index('verify') < seen.events.index('episode')


def test_completed_guard_replay_does_not_repeat_verifier_or_chat(monkeypatch):
    calls = []
    def verify(**values):
        calls.append(values)
        return {'receipt_admission_id': 'retained-judgment'}
    _with_verifier(monkeypatch, verify)
    call, seen, revision, _case = _run(monkeypatch)
    asyncio.run(call)
    raw = copy.deepcopy(seen.snapshots[0])
    runtime = real._load_graphiti()
    async def completed(_self):
        return SimpleNamespace(state=GuardState.COMPLETE)
    async def completed_raw(_self):
        return raw
    monkeypatch.setattr(runtime.MutationGuard, 'begin', completed)
    monkeypatch.setattr(runtime.MutationGuard, 'completed_raw', completed_raw, raising=False)
    restored = []
    asyncio.run(real._add_episode(
        api_key='fixture', password='fixture', body=revision.body,
        name=revision.episode_uuid, episode_id=revision.episode_uuid,
        reference_time=UtcTimestamp.parse(revision.reference_time).value,
        telemetry=real._EpisodeTelemetry(), attempt_number=1,
        validate_result=lambda *_args: pytest.fail('replay must not seal again'),
        restore_result=lambda value, _telemetry: restored.append(value),
        configuration=evaluation_attempt_for((revision.body,)).configuration, revision=revision,
    ))
    assert len(calls) == seen.model_calls == 1
    assert restored == [raw]


def test_runner_binds_original_authoritative_envelope_and_unit(tmp_path):
    calls = []
    unit = replace(_unit(), attempt_number=2)
    cycle = native_graphiti_usage_cycle_id(ingest_id=unit.ingest_id, attempt_number=2)
    service = ModelUsageService(str(tmp_path / 'usage.sqlite3'))
    def verify(**values):
        calls.append(values)
        return {'receipt_admission_id': 'bound-judgment'}
    class Runner(EvaluationGraphitiRunner):
        def _ingest(self, selected, *, deadline, invocation_observer):
            result = invocation_observer.verify_typed_proposals(
                source_revision=SimpleNamespace(ingest_id='different-combined-contract-hash'),
                proposal_receipt={'payload_digest': 'full-typed-payload'},
            )
            assert result == {'receipt_admission_id': 'bound-judgment'}
            return SimpleNamespace(provider_attempt_number=2)
    runner = Runner(clock=lambda: datetime(2026, 10, 4, tzinfo=UTC), typed_proposal_verifier=verify)
    runner.ingest_with_usage(unit, model_usage=service, cycle_id=cycle,
        dispatch_authority={'current': True}, owner_stop_check=lambda: None)
    assert len(calls) == 1
    assert calls[0]['unit'] == unit
    assert calls[0]['envelope'].ingest_id == unit.ingest_id
    assert calls[0]['envelope'].graphiti_attempt_id == f'{unit.ingest_id}:2'
    assert calls[0]['envelope'].cycle_id == cycle


@pytest.mark.parametrize('deny_at', ['missing-evidence', 'stop'])
def test_configured_observer_denies_missing_evidence_or_owner_stop(tmp_path, deny_at):
    unit = _unit()
    calls = []
    service = ModelUsageService(str(tmp_path / 'usage.sqlite3'))
    def verify(**values):
        calls.append(values)
        return None
    def stop():
        if deny_at == 'stop':
            raise RuntimeError('fixture owner stopped')
    class Runner(EvaluationGraphitiRunner):
        def _ingest(self, selected, *, deadline, invocation_observer):
            invocation_observer.verify_typed_proposals(
                source_revision=SimpleNamespace(ingest_id='derived'), proposal_receipt={'facts': []})
            pytest.fail('configured missing evidence or stop must deny')
    runner = Runner(clock=lambda: datetime(2026, 10, 4, tzinfo=UTC), typed_proposal_verifier=verify)
    with pytest.raises((ValueError, RuntimeError), match='evidence is absent|owner stopped'):
        runner.ingest_with_usage(unit, model_usage=service, cycle_id='configured-verifier',
            dispatch_authority={'current': True}, owner_stop_check=stop)
    assert len(calls) == (0 if deny_at == 'stop' else 1)


def test_unconfigured_observer_has_no_receipt_or_behaviour_change(monkeypatch, tmp_path):
    from newsroom.tests.test_graphiti_cursor_sdk_transport import _observer_fixture
    _service, observer, _clock = _observer_fixture(tmp_path)
    _with_verifier(monkeypatch, observer.verify_typed_proposals)
    call, seen, revision, case = _run(monkeypatch)
    result = asyncio.run(call)
    assert 'typed_proposal_verification' not in seen.snapshots[0]['combined']
    assert seen.model_calls == 1 and seen.rollback == 0
    assert [str(edge.fact) for edge in result.edges] == [item['fact'] for item in case.gold['facts']]


def test_failed_verifier_retains_only_bounded_reason_and_exact_reference(monkeypatch):
    reference = SimpleNamespace(invocation_id='sha256:'+'a'*64,
        raw_admission_id='00000000-0000-4000-8000-000000009821',
        receipt_admission_id='00000000-0000-4000-8000-000000009822')
    failure = ValueError('SECRET raw provider text must never be retained')
    failure.reason_code = 'GRAPHITI_PROPOSAL_UNSUPPORTED'
    failure.reference = reference
    def verify(**values):
        raise failure
    _with_verifier(monkeypatch, verify)
    call, seen, _revision, _case = _run(monkeypatch)
    with pytest.raises(ExtractionContractError):
        asyncio.run(call)
    evidence = seen.snapshots[0]['combined']['typed_proposal_verification']
    assert evidence == {'reason_code':'GRAPHITI_PROPOSAL_UNSUPPORTED', 'judgment_reference':vars(reference)}
    assert 'SECRET' not in str(seen.snapshots) and seen.business == {} and seen.rollback == 0
