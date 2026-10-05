"""Selected claims are rendered once with authentic local accounting and replay."""
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
import json
import sqlite3

import pytest

from newsroom.authority.canonical import digest_bytes
from newsroom.control_plane.model_usage import ModelUsageService
from newsroom.control_plane.native_claim_localisation import NativeClaimLocaliser, localisation_policy, LocalisationHold
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.writer import WriterCliExecution
from newsroom.tests.test_native_runtime import _args

NOW = datetime(2026, 10, 5, tzinfo=UTC)


def case(tmp_path, monkeypatch, *, payload=None):
    args = _args(tmp_path, monkeypatch)
    usage = ModelUsageService(str(tmp_path/'usage.sqlite3'))
    state = {'source_binding': {'content_digest': digest_bytes(b'The scheme is now open.')},
        'claims': {'S1L1': {'source_id': 'fixture', 'text': 'The scheme is now open.',
            'entities': [], 'rendering_fragment_count': 1}}}
    payload = payload or {'renderings': {'S1L1': {'rendered_assertion_zh_hant_hk_fragments': ['計劃現已接受申請。'],
        'factual_localisations': [], 'quotation_source_keys': []}}}
    calls = []
    def runner(prompt):
        assert 'source_binding' not in prompt
        calls.append(prompt)
        return WriterCliExecution(json.dumps(payload, ensure_ascii=False), {
            'usage_basis': 'PROVIDER_REPORTED', 'input_tokens': 22, 'output_tokens': 34,
            'cached_read_tokens': 0, 'cached_write_tokens': 0, 'reasoning_tokens': 0,
            'context_tokens': 22, 'total_tokens': 56})
    @contextmanager
    def fence(binding, proof):
        assert binding == state['source_binding']
        yield
    return args, usage, state, runner, fence, calls


def test_localisation_has_one_qualified_leaf_and_authenticated_cas_replay(tmp_path, monkeypatch):
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        localiser = NativeClaimLocaliser(usage=usage, objects=runtime.authority.objects,
            policy=localisation_policy(evidence_digest=digest_bytes(b'qualified fixture'), qualified=True),
            source_fence=fence, runner=runner, implementation_worktree_clean=True, clock=lambda: NOW)
        scope = dict(candidate_id='candidate-1', hypothesis_digest=digest_bytes(b'hypothesis'),
                     evidence_package_digest=digest_bytes(b'package'), proof=runtime.proof)
        ref = localiser.localise(state, **scope)
        record = localiser.read_localisation(ref, state, **scope)
        assert record['renderings']['S1L1']['rendered_assertion_zh_hant_hk_fragments'] == ['計劃現已接受申請。']
        assert localiser.localise(state, **scope) == ref
        assert len(calls) == 1
        assert usage.terminal(ref.invocation_id).outcome == 'LOCALISATION_COMPLETE'
        with sqlite3.connect(usage.path) as c:
            assert c.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0] == 1
        changed = {**state, 'source_binding': {'content_digest': digest_bytes(b'changed source')}}
        with pytest.raises((LocalisationHold, AssertionError)):
            localiser.read_localisation(ref, changed, **scope)


@pytest.mark.parametrize('failure', ('fragment_count', 'timeout'))
def test_failed_rendering_preserves_usage_and_never_redispatches(tmp_path, monkeypatch, failure):
    invalid = {'renderings': {'S1L1': {'rendered_assertion_zh_hant_hk_fragments': ['one', 'extra'],
        'factual_localisations': [], 'quotation_source_keys': []}}}
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch, payload=invalid)
    if failure == 'timeout':
        def runner(prompt):
            calls.append(prompt)
            raise TimeoutError('fixture only')
    with open_native_runtime(**args) as runtime:
        localiser = NativeClaimLocaliser(usage=usage, objects=runtime.authority.objects,
            policy=localisation_policy(evidence_digest=digest_bytes(b'qualified fixture'), qualified=True),
            source_fence=fence, runner=runner, implementation_worktree_clean=True, clock=lambda: NOW)
        scope = dict(candidate_id='candidate-1', hypothesis_digest=digest_bytes(b'hypothesis'),
                     evidence_package_digest=digest_bytes(b'package'), proof=runtime.proof)
        with pytest.raises((LocalisationHold, TimeoutError)):
            localiser.localise(state, **scope)
        with sqlite3.connect(usage.path) as c:
            invocation = c.execute('SELECT invocation_id FROM model_invocation_allocations').fetchone()[0]
        terminal = usage.terminal(invocation)
        assert terminal.outcome == 'LOCALISATION_FAILED'
        assert terminal.usage_status.value == ('REPORTED' if failure == 'fragment_count' else 'ESTIMATED')
        assert terminal.components.total_tokens == (56 if failure == 'fragment_count' else 300000)
        with pytest.raises(LocalisationHold, match='PRIOR_RESULT_UNAVAILABLE'):
            localiser.localise(state, **scope)
        assert len(calls) == 1


def test_same_source_digest_does_not_allow_changed_claim_or_scope_replay(tmp_path, monkeypatch):
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        localiser = NativeClaimLocaliser(usage=usage, objects=runtime.authority.objects,
            policy=localisation_policy(evidence_digest=digest_bytes(b'qualified fixture'), qualified=True),
            source_fence=fence, runner=runner, implementation_worktree_clean=True, clock=lambda: NOW)
        scope = dict(candidate_id='candidate-1', hypothesis_digest=digest_bytes(b'hypothesis'),
                     evidence_package_digest=digest_bytes(b'package'), proof=runtime.proof)
        ref = localiser.localise(state, **scope)
        changed = {**state, 'claims': {'S1L1': {**state['claims']['S1L1'], 'text': 'An unsupported changed claim.'}}}
        with pytest.raises(LocalisationHold, match='REPLAY_BINDING'):
            localiser.read_localisation(ref, changed, **scope)
        with pytest.raises(LocalisationHold, match='REPLAY_BINDING'):
            localiser.read_localisation(ref, state, **{**scope, 'candidate_id': 'other-candidate'})
        assert len(calls) == 1


def test_preallocated_intent_resumes_original_admission_time_without_redispatch(tmp_path, monkeypatch):
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch)
    now = [NOW]
    with open_native_runtime(**args) as runtime:
        localiser = NativeClaimLocaliser(usage=usage, objects=runtime.authority.objects,
            policy=localisation_policy(evidence_digest=digest_bytes(b'qualified fixture'), qualified=True),
            source_fence=fence, runner=runner, implementation_worktree_clean=True, clock=lambda: now[0])
        scope = dict(candidate_id='candidate-1', hypothesis_digest=digest_bytes(b'hypothesis'),
                     evidence_package_digest=digest_bytes(b'package'), proof=runtime.proof)
        _, _, envelope, _ = localiser._input(state, **{k: v for k, v in scope.items() if k != 'proof'})
        usage.open_envelope(envelope)
        now[0] += timedelta(seconds=10)
        ref = localiser.localise(state, **scope)
        assert localiser.localise(state, **scope) == ref
        assert len(calls) == 1
        with sqlite3.connect(usage.path) as c:
            assert c.execute('SELECT admitted_at FROM model_work_envelopes WHERE envelope_id=?',
                (envelope.envelope_id,)).fetchone()[0] == envelope.as_record()['admitted_at']
