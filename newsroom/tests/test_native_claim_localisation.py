"""Selected claims are rendered once with authentic local accounting and replay."""
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
import json
import sqlite3

import pytest

from newsroom.authority import HydrationRequest, ObjectAdmissionRequest
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.control_plane.model_usage import (ModelUsageService, InvocationAllocation, InvocationEfficiencyPolicy, WorkEnvelope)
from newsroom.control_plane.cycle import _complete_writer_usage
from newsroom.control_plane import native_claim_localisation as module
from jsonschema import ValidationError
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
    payload = payload or {'renderings': [{'span_id': 'S1L1', 'rendered_assertion_zh_hant_hk_fragments': ['計劃現已接受申請。'],
        'factual_localisations': [], 'quotation_source_keys': []}]}
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
    invalid = {'renderings': [{'span_id': 'S1L1', 'rendered_assertion_zh_hant_hk_fragments': ['one', 'extra'],
        'factual_localisations': [], 'quotation_source_keys': []}]}
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
        with pytest.raises(LocalisationHold, match='REPLAY_BINDING|PRIOR_RESULT_UNAVAILABLE'):
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
        with pytest.raises(LocalisationHold, match='REPLAY_(?:BINDING|MANIFEST|SCOPE)'):
            localiser.read_localisation(ref, changed, **scope)
        with pytest.raises(LocalisationHold, match='REPLAY_(?:BINDING|MANIFEST|SCOPE)'):
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


def _localiser(usage, runtime, fence, runner):
    return NativeClaimLocaliser(usage=usage, objects=runtime.authority.objects,
        policy=localisation_policy(evidence_digest=digest_bytes(b'qualified fixture'), qualified=True),
        source_fence=fence, runner=runner, implementation_worktree_clean=True, clock=lambda: NOW)


def _scope(runtime):
    return dict(candidate_id='candidate-1', hypothesis_digest=digest_bytes(b'hypothesis'),
        evidence_package_digest=digest_bytes(b'package'), proof=runtime.proof)


def test_bad_v2_schema_retains_actual_raw_and_small_diagnostic_before_raise(tmp_path, monkeypatch):
    payload = {'renderings': [{'span_id': 'S1L1',
        'rendered_assertion_zh_hant_hk_fragments': ['計劃現已接受申請。'],
        'factual_localisations': [], 'quotation_source_keys': ['s' * 257]}]}
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch, payload=payload)
    with open_native_runtime(**args) as runtime:
        localiser = _localiser(usage, runtime, fence, runner)
        scope = _scope(runtime)
        with pytest.raises(ValidationError):
            localiser.localise(state, **scope)
        with sqlite3.connect(usage.path) as c:
            invocation = c.execute('SELECT invocation_id FROM model_invocation_allocations').fetchone()[0]
        ref = localiser._reference(invocation, proof=runtime.proof)
        receipt = json.loads(runtime.authority.objects.rehydrate(
            HydrationRequest(ref.receipt_admission_id, 'evidence.record'), proof=runtime.proof).data)
        raw = runtime.authority.objects.rehydrate(
            HydrationRequest(ref.raw_admission_id, 'evidence.record'), proof=runtime.proof).data
        assert json.loads(raw) == payload
        assert receipt['raw_digest'] == digest_bytes(raw)
        assert receipt['terminal_digest'] == usage.terminal(invocation).terminal_digest
        assert receipt['schema_digest'] == module.SCHEMA_DIGEST
        assert receipt['diagnostic']['validator'] == 'maxLength'
        assert receipt['diagnostic']['instance_path'] == ['renderings', '0', 'quotation_source_keys', '0']
        assert receipt['diagnostic']['value_digest'] == digest_bytes(canonical_json_bytes('s' * 257))
        assert 's' * 257 not in json.dumps(receipt)
        assert len(canonical_json_bytes(receipt['diagnostic'])) < 2048
        assert usage.terminal(invocation).usage_status.value == 'REPORTED'
        with pytest.raises(LocalisationHold, match='REPLAY_(?:BINDING|MANIFEST|SCOPE)'):
            localiser.localise(state, **scope)
        assert len(calls) == 1


def _legacy_result(usage, runtime, state, scope, *, outcome='LOCALISATION_COMPLETE',
                   failure_class=None, reported=True, active=False, breach=False, pre_dispatch=False):
    """Genuine old-policy/allocation/CAS records, independent of the v2 producer."""
    from dataclasses import asdict
    current = localisation_policy(evidence_digest=digest_bytes(b'legacy qualified fixture'), qualified=True)
    values = asdict(current)
    values.pop('canonical_digest')
    for key in ('policy_id', 'version', 'command_semantic_version',
                'context_manifest_schema_version', 'prompt_contract_version'):
        values[key] = 'newsroom.native-claim-localisation.v1'
    values.update(output_schema_digest='sha256:3039378d8ca625fa3c8bec97e4c89e63e4d2265b4282574cdf9dcda2400976d0',
        allowed_context_identities=('newsroom.native-claim-localisation.v1',),
        allowed_config_identities=('newsroom.native-claim-localisation.v1',),
        implementation_revision='sha256:236898455fe3839349b043beb8f461bf232c244ec963a177d1e2cc0f77c0723a')
    policy = InvocationEfficiencyPolicy.create(**values)
    usage.register_policy(policy)
    prompt = canonical_json_bytes({'contract': 'newsroom.native-claim-localisation.v1', 'claims': state['claims']})
    snapshot = digest_canonical({'state': state, **{key: value for key, value in scope.items() if key != 'proof'}})
    envelope = WorkEnvelope.create(cycle_id='claim-localisation:'+snapshot,
        workload_class=policy.workload_class, admitted_at=NOW, admission_decision_id=None,
        candidate_id=scope['candidate_id'], hypothesis_digest=scope['hypothesis_digest'],
        evidence_package_digest=scope['evidence_package_digest'], ingest_id=None, graphiti_attempt_id=None)
    usage.open_envelope(envelope)
    manifest = dict(schema_version=policy.version, provider=policy.provider, route=policy.route,
        model=policy.model, reasoning='high', command_semantic_version=policy.version,
        command_flags=list(policy.command_flags), disabled_capabilities=list(policy.disabled_capabilities),
        implementation_revision=policy.implementation_revision, implementation_worktree_clean=True,
        prompt_contract_version=policy.version, prompt_bytes=len(prompt), prompt_digest=digest_bytes(prompt),
        schema_digest=policy.output_schema_digest, output_schema_digest=policy.output_schema_digest,
        system_digest='sha256:cc606db25532f5bd68ecab110e4ba8bf7a5a241dd9e23b8e362d5542ea4cb60c',
        evidence_package_digest=scope['evidence_package_digest'], evidence_package_bytes=len(canonical_json_bytes(
            {'state': state, **{key: value for key, value in scope.items() if key != 'proof'}})),
        context_identity=policy.version, config_identity=policy.version, one_turn=True, exact_input=True,
        skills_enabled=False, tools_enabled=False, mcp_enabled=False, prior_message_count=0,
        skill_count=0, tool_count=0, mcp_server_count=0, mcp_tool_count=0, source_snapshot_digest=snapshot)
    manifest['request_digest'] = digest_canonical({key: manifest[key] for key in ('provider', 'route',
        'model', 'reasoning', 'command_semantic_version', 'command_flags', 'implementation_revision',
        'system_digest', 'prompt_digest', 'output_schema_digest')})
    manifest['context_manifest_digest'] = digest_canonical(manifest)
    usage.retain_context_manifest(manifest)
    allocation = InvocationAllocation.create(envelope_id=envelope.envelope_id, cycle_id=envelope.cycle_id,
        leaf_ordinal=1, workload_class=policy.workload_class, invocation_policy_digest=policy.canonical_digest,
        provider=policy.provider, route=policy.route, model=policy.model, reasoning='high',
        prompt_contract_version=policy.version, prompt_bytes=len(prompt), prompt_digest=digest_bytes(prompt),
        request_digest=manifest['request_digest'], output_schema_digest=policy.output_schema_digest,
        max_output_tokens=None, context_manifest_digest=manifest['context_manifest_digest'],
        context_identity=policy.version, config_identity=policy.version, one_turn=True, exact_input=True,
        skills_enabled=False, tools_enabled=False, mcp_enabled=False, prior_message_count=0,
        allocated_at=NOW, recovery_deadline_at=NOW+timedelta(seconds=305), parent_invocation_id=None)
    usage.allocate(allocation, owner_emergency_stop=False)
    if active:
        return allocation, None
    if not pre_dispatch:
        usage.observe_transport(invocation_id=allocation.invocation_id, observed_at=NOW,
                                state='DISPATCH_STARTED', evidence_digest=allocation.request_digest)
    telemetry = {'usage_basis': 'PROVIDER_REPORTED', 'input_tokens': 22, 'output_tokens': 34,
        'cached_read_tokens': 0, 'cached_write_tokens': 0, 'reasoning_tokens': 0,
        'context_tokens': 22, 'total_tokens': 56}
    if breach:
        telemetry['total_tokens'] = 300001
        telemetry['output_tokens'] = 299979
    _complete_writer_usage(usage, allocation, outcome=outcome, failure_class=failure_class,
        usage=None if pre_dispatch else telemetry if reported else None,
        dispatch_at=None if pre_dispatch else NOW, completed_at=NOW,
        provider_dispatched=not pre_dispatch, policy=policy)
    if outcome != 'LOCALISATION_COMPLETE':
        return allocation, None  # Old v1 really lost raw; do not fabricate a failure receipt.
    raw = canonical_json_bytes({'renderings': {'S1L1': {
        'rendered_assertion_zh_hant_hk_fragments': ['計劃現已接受申請。'],
        'factual_localisations': [], 'quotation_source_keys': []}}})
    raw_admission = runtime.authority.objects.admit(ObjectAdmissionRequest('evidence.record',
        'claim-localisation-raw:'+allocation.invocation_id), raw, proof=runtime.proof).admission
    terminal = usage.terminal(allocation.invocation_id)
    receipt = {'version': policy.version, 'invocation_id': allocation.invocation_id,
        'allocation_digest': allocation.canonical_digest, 'terminal_digest': terminal.terminal_digest,
        'source_snapshot_digest': snapshot, 'source_binding': state['source_binding'],
        'raw_admission_id': str(raw_admission.admission_id), 'raw_digest': digest_bytes(raw)}
    admission = runtime.authority.objects.admit(ObjectAdmissionRequest('evidence.record',
        'claim-localisation-receipt:'+allocation.invocation_id), canonical_json_bytes(receipt), proof=runtime.proof).admission
    return allocation, module.LocalisationReference(allocation.invocation_id,
        raw_admission.admission_id, admission.admission_id)


def test_authenticated_v1_success_is_read_and_reused_without_new_dispatch(tmp_path, monkeypatch):
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        scope = _scope(runtime)
        allocation, legacy = _legacy_result(usage, runtime, state, scope)
        localiser = _localiser(usage, runtime, fence, runner)
        assert localiser.read_localisation(legacy, state, **scope)['version'] == module.LEGACY_VERSION
        assert localiser.localise(state, **scope) == legacy
        assert calls == []
        assert usage.terminal(allocation.invocation_id).outcome == 'LOCALISATION_COMPLETE'


def test_reported_v1_schema_failure_has_one_distinct_v2_repair_and_keeps_old_accounting(tmp_path, monkeypatch):
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        scope = _scope(runtime)
        old, _ = _legacy_result(usage, runtime, state, scope,
            outcome='LOCALISATION_FAILED', failure_class='ValidationError')
        old_terminal = usage.terminal(old.invocation_id)
        localiser = _localiser(usage, runtime, fence, runner)
        ref = localiser.localise(state, **scope)
        record = localiser.read_localisation(ref, state, **scope)
        assert record['repair_of'] == old.invocation_id
        assert ref.invocation_id != old.invocation_id
        assert usage.terminal(old.invocation_id) == old_terminal
        assert localiser.localise(state, **scope) == ref
        assert len(calls) == 1
        with sqlite3.connect(usage.path) as c:
            assert c.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0] == 2


@pytest.mark.parametrize('condition', ('unknown', 'active', 'breach', 'other_hold'))
def test_legacy_uncertain_or_breached_or_unproved_failure_never_upgrades(tmp_path, monkeypatch, condition):
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        scope = _scope(runtime)
        old, _ = _legacy_result(usage, runtime, state, scope,
            outcome='LOCALISATION_FAILED', failure_class='LocalisationHold' if condition == 'other_hold' else 'ValidationError',
            reported=condition != 'unknown', active=condition == 'active', breach=condition == 'breach')
        with pytest.raises(LocalisationHold):
            _localiser(usage, runtime, fence, runner).localise(state, **scope)
        assert calls == []
        with sqlite3.connect(usage.path) as c:
            assert c.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0] == 1


@pytest.mark.parametrize('condition', ('duplicate', 'unknown', 'byte_bound'))
def test_v2_inventory_and_utf8_source_key_boundaries_hold(tmp_path, monkeypatch, condition):
    item = {'span_id': 'S1L1', 'rendered_assertion_zh_hant_hk_fragments': ['計劃現已接受申請。'],
        'factual_localisations': [], 'quotation_source_keys': []}
    payload = {'renderings': [item]}
    if condition == 'duplicate':
        payload['renderings'].append(dict(item))
    elif condition == 'unknown':
        item['span_id'] = 'NOT_SELECTED'
    else:
        item['quotation_source_keys'] = ['中' * 86]  # 258 bytes, only 86 characters.
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch, payload=payload)
    with open_native_runtime(**args) as runtime:
        localiser = _localiser(usage, runtime, fence, runner)
        with pytest.raises(LocalisationHold):
            localiser.localise(state, **_scope(runtime))
        assert len(calls) == 1


@pytest.mark.parametrize('corruption', ['context-header', 'context-snapshot', 'missing-dispatch', 'dispatch-binding'])
def test_legacy_schema_failure_never_repairs_unbound_evidence(tmp_path, monkeypatch, corruption):
    from newsroom.authority.canonical import canonical_json_bytes, digest_canonical
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        scope = _scope(runtime)
        old, _ = _legacy_result(usage, runtime, state, scope,
                               outcome='LOCALISATION_FAILED', failure_class='ValidationError')
        with sqlite3.connect(usage.path) as c:
            if corruption == 'context-header':
                c.execute("UPDATE model_invocation_context_manifests SET route='tampered' WHERE context_manifest_digest=?", (old.context_manifest_digest,))
            elif corruption == 'context-snapshot':
                manifest = json.loads(c.execute('SELECT record_json FROM model_invocation_context_manifests WHERE context_manifest_digest=?',
                                                (old.context_manifest_digest,)).fetchone()[0])
                manifest['source_snapshot_digest'] = digest_bytes(b'another source')
                c.execute('UPDATE model_invocation_context_manifests SET record_json=? WHERE context_manifest_digest=?',
                          (canonical_json_bytes(manifest).decode(), old.context_manifest_digest))
            elif corruption == 'dispatch-binding':
                record = json.loads(c.execute('SELECT record_json FROM model_transport_observations WHERE invocation_id=?',
                                              (old.invocation_id,)).fetchone()[0])
                record.pop('observation_digest'); record['evidence_digest'] = digest_bytes(b'other request')
                digest = digest_canonical(record); record['observation_digest'] = digest
                c.execute('UPDATE model_transport_observations SET observation_digest=?,evidence_digest=?,record_json=? WHERE invocation_id=?',
                          (digest, record['evidence_digest'], canonical_json_bytes(record).decode(), old.invocation_id))
            else:
                c.execute('DELETE FROM model_transport_observations WHERE invocation_id=?', (old.invocation_id,))
        with pytest.raises((LocalisationHold, ValueError)):
            _localiser(usage, runtime, fence, runner).localise(state, **scope)
        assert calls == []


def test_pre_dispatch_validation_error_is_not_output_wire_repair_credit(tmp_path, monkeypatch):
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        scope = _scope(runtime)
        old, _ = _legacy_result(usage, runtime, state, scope, outcome='LOCALISATION_FAILED',
                               failure_class='ValidationError', pre_dispatch=True)
        assert usage.terminal(old.invocation_id).pre_dispatch_zero_proved
        with pytest.raises(LocalisationHold):
            _localiser(usage, runtime, fence, runner).localise(state, **scope)
        assert calls == []


def test_recomputed_legacy_policy_breach_cannot_grant_repair_credit(tmp_path, monkeypatch):
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        scope = _scope(runtime)
        _legacy_result(usage, runtime, state, scope, outcome='LOCALISATION_FAILED', failure_class='ValidationError')
        monkeypatch.setattr(ModelUsageService, '_validate_terminal', staticmethod(lambda *_a, **_k: 'MAX_TOTAL_TOKENS_EXCEEDED'))
        with pytest.raises(LocalisationHold):
            _localiser(usage, runtime, fence, runner).localise(state, **scope)
        assert calls == []
