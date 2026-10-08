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


def test_future_policy_is_default_but_original_producer_bytes_stay_frozen():
    policy = localisation_policy(evidence_digest=digest_bytes(b'future fixture'), qualified=True)
    assert policy.prompt_contract_version == module.ALIGNED_VERSION
    assert policy.output_schema_digest == module.ALIGNED_SCHEMA_DIGEST
    assert digest_bytes(module.LEGACY_SYSTEM.encode()) == 'sha256:cc606db25532f5bd68ecab110e4ba8bf7a5a241dd9e23b8e362d5542ea4cb60c'
    assert digest_bytes(module.SYSTEM.encode()) == 'sha256:9e97c3ae97211f9443de1182d8a0cabf489f3e85bcf0096d16c894da085f0588'
    assert module.LEGACY_SCHEMA_DIGEST == 'sha256:3039378d8ca625fa3c8bec97e4c89e63e4d2265b4282574cdf9dcda2400976d0'
    assert module.SCHEMA_DIGEST == 'sha256:25829291c0174634a597794c513755278fbf0956c8c69553e310edac7632ea7d'


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
            policy=localisation_policy(evidence_digest=digest_bytes(b'qualified fixture'), qualified=True, version=module.VERSION),
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
            policy=localisation_policy(evidence_digest=digest_bytes(b'qualified fixture'), qualified=True, version=module.VERSION),
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
            policy=localisation_policy(evidence_digest=digest_bytes(b'qualified fixture'), qualified=True, version=module.VERSION),
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
            policy=localisation_policy(evidence_digest=digest_bytes(b'qualified fixture'), qualified=True, version=module.VERSION),
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
        policy=localisation_policy(evidence_digest=digest_bytes(b'qualified fixture'), qualified=True, version=module.VERSION),
        source_fence=fence, runner=runner, implementation_worktree_clean=True, clock=lambda: NOW)


def _scope(runtime):
    return dict(candidate_id='candidate-1', hypothesis_digest=digest_bytes(b'hypothesis'),
        evidence_package_digest=digest_bytes(b'package'), proof=runtime.proof)


def test_future_localisation_checks_content_before_complete_and_never_retries(tmp_path, monkeypatch):
    payload = {'renderings': [{'span_id': 'S1L1',
        'rendered_assertion_zh_hant_hk_fragments': ['計劃現已接受申請，new fact。'],
        'factual_localisations': [{'source_lookup_key': 'scheme', 'rendered_expression': '計劃'}],
        'quotation_source_keys': []}]}
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch, payload=payload)
    with open_native_runtime(**args) as runtime:
        localiser = NativeClaimLocaliser(usage=usage, objects=runtime.authority.objects,
            policy=localisation_policy(evidence_digest=digest_bytes(b'future fixture'), qualified=True,
                                       version='newsroom.native-claim-localisation.v3'),
            source_fence=fence, runner=runner, implementation_worktree_clean=True, clock=lambda: NOW)
        with pytest.raises(LocalisationHold) as held:
            localiser.localise(state, **_scope(runtime))
        assert set(held.value.reason_codes) >= {'LOCALISATION_FACT_EQUIVALENCE_HOLD', 'LOCALISATION_LANGUAGE_HOLD'}
        with sqlite3.connect(usage.path) as connection:
            invocation = connection.execute('SELECT invocation_id FROM model_invocation_allocations').fetchone()[0]
        terminal = usage.terminal(invocation)
        assert terminal.outcome == 'LOCALISATION_FAILED' and terminal.usage_status.value == 'REPORTED'
        with pytest.raises(LocalisationHold):
            localiser.localise(state, **_scope(runtime))
        assert len(calls) == 1 and usage.terminal(invocation) == terminal


@pytest.mark.parametrize(('source', 'rendered', 'pairs', 'names', 'fragments', 'reason'), [
    ('People aged 16 to 19 are eligible.', '16至20歲的人士符合資格。', [], [], None, 'LOCALISATION_NUMERIC_HOLD'),
    ('The scheme opens in 2026.', '計劃於2027年開放。', [], [], None, 'LOCALISATION_NUMERIC_HOLD'),
    ('People aged 16 to 19 are eligible.', '16至19歲的人士符合資格。', [], [], None, 'LOCALISATION_NUMERIC_HOLD'),
    ('Age 16, year 2026.', '年齡2026，年份16。', [], [], None, 'LOCALISATION_NUMERIC_HOLD'),
    ('關閉時間為2小時。', '關閉時間是2分鐘。', [], [], None, 'LOCALISATION_NUMERIC_HOLD'),
    ('金額為+20。', '金額是−20。', [], [], None, 'LOCALISATION_NUMERIC_HOLD'),
    ('金額為£20。', '金額是€20。', [], [], None, 'LOCALISATION_NUMERIC_HOLD'),
    ('The funding is £20 million.', '資助為二千萬英鎊。', [], [], None, 'LOCALISATION_NUMERIC_HOLD'),
    ('The deadline changed.', '英國的限期已更改。', [], [], None, 'LOCALISATION_ENTITY_HOLD'),
    ('The Home Office changed the deadline.', 'Home OfficeHome Office已更改限期。', [],
     [['Home Office', 'ORGANISATION']], ['Home Office', '已更改限期。'], 'LOCALISATION_ENTITY_HOLD'),
    ('The scheme opens.', '計劃現已開放。', [], [], None, 'LOCALISATION_QUOTATION_HOLD'),
])
def test_future_content_boundary_rejects_unproved_numbers_names_and_quotes(source, rendered, pairs, names, fragments, reason):
    state = {'claims': {'0': {'text': source, 'entities': names, 'rendering_fragment_count': len(names)+1}}}
    item = {'span_id': '0', 'rendered_assertion_zh_hant_hk_fragments': fragments or [rendered],
            'factual_localisations': pairs, 'quotation_source_keys': ['invented quotation'] if reason.endswith('QUOTATION_HOLD') else []}
    with pytest.raises(LocalisationHold) as held:
        module._renderings(canonical_json_bytes({'renderings': [item]}), state, version=module.ALIGNED_VERSION)
    assert reason in held.value.reason_codes


@pytest.mark.parametrize(('source', 'rendered', 'pair'), [
    ('The deadline changed on 1 October 2026.', '限期於2026年10月1日更改。', ('1 October 2026', '2026年10月1日')),
    ('The funding is £20 million.', '資助為二千萬英鎊。', ('£20 million', '二千萬英鎊')),
    ('The funding is HK$20.', '資助為二十港元。', ('HK$20', '二十港元')),
    ('The closure lasts 2 hours.', '關閉時間為120分鐘。', ('2 hours', '120分鐘')),
    ('The change affects 2 schools.', '改動影響2間學校。', ('2 schools', '2間學校')),
    ('The scheme lasts 2 years.', '計劃為期2年。', ('2 years', '2年')),
])
def test_future_supported_localisations_preserve_exact_fact_proof(source, rendered, pair):
    state = {'claims': {'0': {'text': source, 'entities': [], 'rendering_fragment_count': 1}}}
    item = {'span_id': '0', 'rendered_assertion_zh_hant_hk_fragments': [rendered],
            'factual_localisations': [{'source_lookup_key': pair[0], 'rendered_expression': pair[1]}], 'quotation_source_keys': []}
    assert module._renderings(canonical_json_bytes({'renderings': [item]}), state, version=module.ALIGNED_VERSION)['0']['factual_localisations'] == item['factual_localisations']


def test_future_numeric_literals_do_not_bind_unrelated_surrounding_prose():
    state = {'claims': {'0': {'text': '第2項政策適用於住戶。', 'entities': [], 'rendering_fragment_count': 1}}}
    item = {'span_id': '0', 'rendered_assertion_zh_hant_hk_fragments': ['第2項政策以住戶為適用對象。'],
            'factual_localisations': [], 'quotation_source_keys': []}
    assert module._renderings(canonical_json_bytes({'renderings': [item]}), state, version=module.ALIGNED_VERSION)['0']['factual_localisations'] == []


def test_future_forged_entity_input_holds_before_dispatch(tmp_path, monkeypatch):
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch)
    state['claims']['S1L1'].update(entities=[['invented actor', 'ORGANISATION']], rendering_fragment_count=2)
    with open_native_runtime(**args) as runtime:
        future = NativeClaimLocaliser(usage=usage, objects=runtime.authority.objects,
            policy=localisation_policy(evidence_digest=digest_bytes(b'future fixture'), qualified=True),
            source_fence=fence, runner=runner, implementation_worktree_clean=True, clock=lambda: NOW)
        with pytest.raises(LocalisationHold, match='ENTITY_INPUT'):
            future.localise(state, **_scope(runtime))
        assert calls == []
        with sqlite3.connect(usage.path) as connection:
            assert connection.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0] == 0


@pytest.mark.parametrize('quoted', [True, False])
def test_future_quotation_key_requires_an_actual_source_quote_not_substring_alone(quoted):
    key = 'The deadline changed.'
    source = 'Home Office said "The deadline changed."' if quoted else 'Home Office stated that The deadline changed.'
    state = {'claims': {'0': {'text': source, 'entities': [['Home Office', 'ORGANISATION']], 'rendering_fragment_count': 2}}}
    item = {'span_id': '0', 'rendered_assertion_zh_hant_hk_fragments': ['', '表示限期已更改。'],
            'factual_localisations': [], 'quotation_source_keys': [key]}
    raw = canonical_json_bytes({'renderings': [item]})
    if quoted:
        assert module._renderings(raw, state, version=module.ALIGNED_VERSION)['0']['quotation_source_keys'] == [key]
    else:
        with pytest.raises(LocalisationHold) as held:
            module._renderings(raw, state, version=module.ALIGNED_VERSION)
        assert 'LOCALISATION_QUOTATION_HOLD' in held.value.reason_codes


@pytest.mark.parametrize(('source', 'rendered', 'pair'), [
    ('People aged 16 to 19 are eligible.', '16至19歲的人士符合資格。', ('16 to 19', '16至19歲')),
    ('The change is for academic year 2026 to 2027.', '改動適用於2026至2027學年。', ('academic year 2026 to 2027', '2026至2027學年')),
    ('The funding is £20 million.', '資助為三千萬英鎊。', ('£20 million', '三千萬英鎊')),
    ('The deadline changed on 1 October 2026.', '限期於2027年10月1日更改。', ('1 October 2026', '2027年10月1日')),
])
def test_future_unsupported_or_changed_facts_are_never_dropped_as_glossary(source, rendered, pair):
    state = {'claims': {'0': {'text': source, 'entities': [], 'rendering_fragment_count': 1}}}
    item = {'span_id': '0', 'rendered_assertion_zh_hant_hk_fragments': [rendered],
            'factual_localisations': [{'source_lookup_key': pair[0], 'rendered_expression': pair[1]}], 'quotation_source_keys': []}
    with pytest.raises(LocalisationHold) as held:
        module._renderings(canonical_json_bytes({'renderings': [item]}), state, version=module.ALIGNED_VERSION)
    assert 'LOCALISATION_FACT_EQUIVALENCE_HOLD' in held.value.reason_codes


@pytest.mark.parametrize(('source', 'rendered', 'pairs'), [
    ('The course lasts 2 years and the other course lasts 3 years.', '甲課程為期3年，乙課程為期2年。', [('2 years', '2年'), ('3 years', '3年')]),
    ('The course lasts 2 years.', '課程為期2年，適用期亦為2年。', [('2 years', '2年')]),
    ('Funding is £20 million and the loan is £30 million.', '資助為三千萬英鎊，貸款為二千萬英鎊。', [('£20 million', '二千萬英鎊'), ('£30 million', '三千萬英鎊')]),
])
def test_future_typed_facts_preserve_occurrence_order_and_count(source, rendered, pairs):
    state = {'claims': {'0': {'text': source, 'entities': [], 'rendering_fragment_count': 1}}}
    item = {'span_id': '0', 'rendered_assertion_zh_hant_hk_fragments': [rendered],
            'factual_localisations': [{'source_lookup_key': source, 'rendered_expression': target} for source, target in pairs], 'quotation_source_keys': []}
    with pytest.raises(LocalisationHold) as held:
        module._renderings(canonical_json_bytes({'renderings': [item]}), state, version=module.ALIGNED_VERSION)
    assert 'LOCALISATION_NUMERIC_HOLD' in held.value.reason_codes


@pytest.mark.parametrize('pair', [True, False])
def test_future_partial_quantity_cannot_erase_unsupported_half_suffix(pair):
    state = {'claims': {'0': {'text': '計劃為期2年半。', 'entities': [], 'rendering_fragment_count': 1}}}
    item = {'span_id': '0', 'rendered_assertion_zh_hant_hk_fragments': ['計劃為期2年。'],
            'factual_localisations': [{'source_lookup_key': '2年', 'rendered_expression': '2年'}] if pair else [], 'quotation_source_keys': []}
    with pytest.raises(LocalisationHold) as held:
        module._renderings(canonical_json_bytes({'renderings': [item]}), state, version=module.ALIGNED_VERSION)
    assert 'LOCALISATION_NUMERIC_HOLD' in held.value.reason_codes


@pytest.mark.parametrize('quoted', [False, True])
def test_future_protected_name_apostrophe_is_not_a_quote_boundary(quoted):
    name = "Teachers' Pension Scheme"
    source = f'The provider said "{name}" accepts applications.' if quoted else f'{name} now accepts applications.'
    state = {'claims': {'0': {'text': source, 'entities': [[name, 'OFFICIAL_TERM']], 'rendering_fragment_count': 2}}}
    item = {'span_id': '0', 'rendered_assertion_zh_hant_hk_fragments': ['「' if quoted else '', '」現已接受申請。' if quoted else '現已接受申請。'],
            'factual_localisations': [], 'quotation_source_keys': [name] if quoted else []}
    raw = canonical_json_bytes({'renderings': [item]})
    decoded = module._renderings(raw, state, version=module.ALIGNED_VERSION)
    assert decoded['0'] == {key: value for key, value in item.items() if key != 'span_id'}
    assert canonical_json_bytes({'renderings': [item]}) == raw


@pytest.mark.parametrize('rendered', ['「計劃現已開放。」', '計劃現已「開放。', '計劃現已「開放』。'])
def test_future_target_quotes_need_complete_supported_source_bindings(rendered):
    state = {'claims': {'0': {'text': 'The scheme is now open.', 'entities': [], 'rendering_fragment_count': 1}}}
    item = {'span_id': '0', 'rendered_assertion_zh_hant_hk_fragments': [rendered],
            'factual_localisations': [], 'quotation_source_keys': []}
    with pytest.raises(LocalisationHold) as held:
        module._renderings(canonical_json_bytes({'renderings': [item]}), state, version=module.ALIGNED_VERSION)
    assert 'LOCALISATION_QUOTATION_HOLD' in held.value.reason_codes


@pytest.mark.parametrize('quoted', ['限期', '限期已更改。'])
def test_future_target_quote_must_be_completely_bound_not_a_partial_key(quoted):
    state = {'claims': {'0': {'text': 'Home Office said "限期已更改。"', 'entities': [['Home Office', 'ORGANISATION']], 'rendering_fragment_count': 2}}}
    item = {'span_id': '0', 'rendered_assertion_zh_hant_hk_fragments': ['', f'表示「{quoted}」。'],
            'factual_localisations': [], 'quotation_source_keys': ['限期已更改。']}
    raw = canonical_json_bytes({'renderings': [item]})
    if quoted == '限期':
        with pytest.raises(LocalisationHold) as held:
            module._renderings(raw, state, version=module.ALIGNED_VERSION)
        assert 'LOCALISATION_QUOTATION_HOLD' in held.value.reason_codes
    else:
        assert module._renderings(raw, state, version=module.ALIGNED_VERSION)['0']['quotation_source_keys'] == ['限期已更改。']


@pytest.mark.parametrize('capability', [None, 'terms', 'year'])
def test_future_renderer_reaches_governed_materialisation_and_admission_contract(tmp_path, monkeypatch, capability):
    from dataclasses import replace
    from newsroom.tests.test_qualification_semantic_witness import _selected_qualification_case
    from newsroom.control_plane.native_assessor import AutonomousNativeEvidenceAssessor
    from newsroom.control_plane.native_evidence import NativeEvidenceController
    from newsroom.control_plane.admission import DeterministicWriteAdmission
    from newsroom.control_plane.evidence import EvidenceGateEvidence, validate_governed_evidence_records

    original_policy = module.localisation_policy
    monkeypatch.setattr(module, 'localisation_policy', lambda **kwargs: original_policy(version=module.ALIGNED_VERSION, **kwargs))
    with _selected_qualification_case(tmp_path, monkeypatch, malformed_rendering=True, fault=capability) as (
            consumer, verifier, original, candidate, base, source, acquired, _scope, proof, usage, qa, jev, render):
        selected = consumer.compose_selected(original, candidate, base, (source,), (acquired,), proof=proof)
        assessment = AutonomousNativeEvidenceAssessor._validated_execution(selected.execution, candidate, base,
            (source,), (acquired,), semantic_witnesses=selected.semantic_witnesses,
            source_renderings=selected.source_renderings, semantic_witness_reader=verifier.read)
        package = replace(base, **{key: getattr(assessment, key) for key in (
            'governed_claims', 'qualification_evidence', 'substantive_new_information', 'selection_rationale', 'geography', 'categories')})
        records = NativeEvidenceController._records(base, package, (source,), (acquired,), assessment)
        retained = tuple((row['record_id'], row['record_type'], canonical_json_bytes(row).decode(),
                          digest_bytes(canonical_json_bytes(row))) for row in records)
        assert validate_governed_evidence_records(candidate_id=base.candidate_id,
            source_inventory=((source.unit.source_id, acquired.canonical_url),), base_package_digest=base.digest,
            package=package, retained_records=retained) is not None
        gates = ('CLAIM_TRACEABILITY', 'EVIDENCE_SUFFICIENCY', 'SOURCE_AUTHORITY')
        # These surrounding readiness gates are synthetic fixtures, not live Source acceptance.
        ready = replace(package, freshness_result='PASS', integrity_result='PASS',
            resolved_evidence_records=tuple((row['record_id'], digest_bytes(canonical_json_bytes(row))) for row in records),
            evidence_gate_results=tuple((gate, 'PASS') for gate in gates),
            evidence_gate_evidence=tuple(EvidenceGateEvidence(gate, 'PASS', tuple(claim.claim_id for claim in package.governed_claims)) for gate in gates))
        decision = DeterministicWriteAdmission(semantic_witness_reader=verifier.read).decide_candidate_identity(
            candidate_id=base.candidate_id, hypothesis_id=base.hypothesis_id, package=ready, decided_at='2026-10-05T12:00:00Z')
        assert decision.decision == 'WRITE_READY', decision.stable_reason_codes
        assert consumer.compose_selected(original, candidate, base, (source,), (acquired,), proof=proof) == selected
        assert len(qa) == 1 and len(jev) == 2 and len(render) == 1
        with sqlite3.connect(usage.path) as connection:
            row = connection.execute('SELECT record_json FROM model_invocation_allocations WHERE route=?', (module.ROUTE,)).fetchone()
        assert json.loads(row[0])['prompt_contract_version'] == module.ALIGNED_VERSION


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


@pytest.mark.parametrize('condition', ['complete', 'reported-failure', 'unknown', 'active', 'known-NO'])
def test_future_contract_never_reopens_an_older_paid_purpose(tmp_path, monkeypatch, condition):
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        scope = _scope(runtime)
        old, reference = _legacy_result(usage, runtime, state, scope,
            outcome='LOCALISATION_COMPLETE' if condition == 'complete' else 'LOCALISATION_FAILED',
            failure_class=None if condition == 'complete' else 'QUALIFICATION_NO' if condition == 'known-NO' else 'ValidationError',
            reported=condition != 'unknown', active=condition == 'active')
        before = usage.terminal(old.invocation_id)
        future = NativeClaimLocaliser(usage=usage, objects=runtime.authority.objects,
            policy=localisation_policy(evidence_digest=digest_bytes(b'future fixture'), qualified=True),
            source_fence=fence, runner=runner, implementation_worktree_clean=True, clock=lambda: NOW)
        if reference is not None:
            before_raw = runtime.authority.objects.rehydrate(HydrationRequest(reference.raw_admission_id, 'evidence.record'), proof=runtime.proof).data
            for _ in range(2):
                # Both ordinary consumers obtain a reference before calling the reader.
                selected = future.localise(state, **scope)
                assert selected == reference
                assert future.read_localisation(selected, state, **scope)['version'] == module.LEGACY_VERSION
            assert runtime.authority.objects.rehydrate(HydrationRequest(reference.raw_admission_id, 'evidence.record'), proof=runtime.proof).data == before_raw
        else:
            with pytest.raises(LocalisationHold):
                future.localise(state, **scope)
        assert calls == [] and usage.terminal(old.invocation_id) == before
        with sqlite3.connect(usage.path) as connection:
            assert connection.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0] == 1


@pytest.mark.parametrize('condition', ['complete', 'reported-failure', 'unknown'])
def test_future_contract_keeps_v2_paid_work_and_original_reader(tmp_path, monkeypatch, condition):
    payload = {'renderings': [{'span_id': 'S1L1', 'rendered_assertion_zh_hant_hk_fragments': ['計劃現已開放。'],
                              'factual_localisations': [], 'quotation_source_keys': []}]}
    if condition == 'reported-failure':
        payload['renderings'][0]['rendered_assertion_zh_hant_hk_fragments'].append('多餘片段。')
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch, payload=payload)
    if condition == 'unknown':
        def runner(prompt):
            calls.append(prompt)
            raise TimeoutError('synthetic uncertain transport')
    with open_native_runtime(**args) as runtime:
        old = _localiser(usage, runtime, fence, runner)
        scope = _scope(runtime)
        if condition == 'complete':
            reference = old.localise(state, **scope)
        else:
            with pytest.raises((LocalisationHold, TimeoutError)):
                old.localise(state, **scope)
        with sqlite3.connect(usage.path) as connection:
            invocation = connection.execute('SELECT invocation_id FROM model_invocation_allocations').fetchone()[0]
        before = usage.terminal(invocation)
        future = NativeClaimLocaliser(usage=usage, objects=runtime.authority.objects,
            policy=localisation_policy(evidence_digest=digest_bytes(b'future fixture'), qualified=True),
            source_fence=fence, runner=runner, implementation_worktree_clean=True, clock=lambda: NOW)
        if condition == 'complete':
            before_raw = runtime.authority.objects.rehydrate(HydrationRequest(reference.raw_admission_id, 'evidence.record'), proof=runtime.proof).data
            for _ in range(2):
                selected = future.localise(state, **scope)
                assert selected == reference
                assert future.read_localisation(selected, state, **scope)['version'] == module.VERSION
            assert runtime.authority.objects.rehydrate(HydrationRequest(reference.raw_admission_id, 'evidence.record'), proof=runtime.proof).data == before_raw
        else:
            with pytest.raises(LocalisationHold):
                future.localise(state, **scope)
        assert len(calls) == 1 and usage.terminal(invocation) == before
        with sqlite3.connect(usage.path) as connection:
            assert connection.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0] == 1


@pytest.mark.parametrize('fault', [None, 'missing-repair', 'missing-receipt', 'ancestor-breach'])
def test_future_callback_reuses_only_existing_authenticated_v2_repair_lineage(tmp_path, monkeypatch, fault):
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        scope = _scope(runtime)
        ancestor, _ = _legacy_result(usage, runtime, state, scope,
            outcome='LOCALISATION_FAILED', failure_class='ValidationError')
        reference = None
        if fault != 'missing-repair':
            old = _localiser(usage, runtime, fence, runner)
            if fault == 'missing-receipt':
                # Reproduce a crash after a settled COMPLETE and raw CAS write.
                original_admit = type(runtime.authority.objects).admit
                def interrupted_admit(self, request, *args, **kwargs):
                    if request.idempotency_key.startswith('claim-localisation-receipt:'):
                        raise RuntimeError('synthetic receipt-write interruption')
                    return original_admit(self, request, *args, **kwargs)
                with monkeypatch.context() as interrupted:
                    interrupted.setattr(type(runtime.authority.objects), 'admit', interrupted_admit)
                    with pytest.raises(RuntimeError, match='receipt-write'):
                        old.localise(state, **scope)
            else:
                reference = old.localise(state, **scope)
                assert old.read_localisation(reference, state, **scope)['repair_of'] == ancestor.invocation_id
        with sqlite3.connect(usage.path) as connection:
            before = tuple(connection.execute('SELECT record_json FROM model_invocation_terminals ORDER BY invocation_id').fetchall())
            allocation_count = connection.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]
        future = NativeClaimLocaliser(usage=usage, objects=runtime.authority.objects,
            policy=localisation_policy(evidence_digest=digest_bytes(b'future fixture'), qualified=True),
            source_fence=fence, runner=lambda _prompt: pytest.fail('historical work dispatched again'),
            implementation_worktree_clean=True, clock=lambda: NOW)
        if fault == 'ancestor-breach':
            monkeypatch.setattr(ModelUsageService, '_validate_terminal', staticmethod(lambda *_args, **_kwargs: 'MAX_TOTAL_TOKENS_EXCEEDED'))
        if fault is None:
            before_raw = runtime.authority.objects.rehydrate(HydrationRequest(reference.raw_admission_id, 'evidence.record'), proof=runtime.proof).data
            for _ in range(2):
                selected = future.localise(state, **scope)
                assert selected == reference
                assert future.read_localisation(selected, state, **scope)['repair_of'] == ancestor.invocation_id
            assert runtime.authority.objects.rehydrate(HydrationRequest(reference.raw_admission_id, 'evidence.record'), proof=runtime.proof).data == before_raw
        else:
            with pytest.raises(LocalisationHold):
                future.localise(state, **scope)
        with sqlite3.connect(usage.path) as connection:
            assert tuple(connection.execute('SELECT record_json FROM model_invocation_terminals ORDER BY invocation_id').fetchall()) == before
            assert connection.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0] == allocation_count
        assert len(calls) == int(fault != 'missing-repair')


def test_sourceqa_ordinary_callbacks_reuse_v2_success_under_a_v3_policy(tmp_path, monkeypatch):
    from newsroom.tests.test_qualification_semantic_witness import _selected_qualification_case
    policy_factory = module.localisation_policy
    monkeypatch.setattr(module, 'localisation_policy', lambda **kwargs: policy_factory(version=module.VERSION, **kwargs))
    with _selected_qualification_case(tmp_path, monkeypatch, malformed_rendering=True) as (
            consumer, verifier, original, candidate, base, source, acquired, _scope, proof, usage, qa, jev, render):
        selected = consumer.compose_selected(original, candidate, base, (source,), (acquired,), proof=proof)
        future = NativeClaimLocaliser(usage=usage, objects=consumer.objects,
            policy=policy_factory(evidence_digest=digest_bytes(b'future fixture'), qualified=True),
            source_fence=consumer.qualifier.fence, runner=lambda _prompt: pytest.fail('cached localisation was redispatched'),
            implementation_worktree_clean=True, clock=lambda: NOW)
        scope = dict(candidate_id=candidate.candidate_id, hypothesis_digest=candidate.governing_manifest.canonical_digest,
                     evidence_package_digest=base.digest, proof=proof)
        consumer.localise = lambda request: future.localise(request, **scope)
        consumer.read_localisation = lambda reference, request: future.read_localisation(reference, request, **scope)
        monkeypatch.setattr(module, '_validate_content', lambda *_args: pytest.fail('v3 producer checks retroactively applied'))
        assert consumer.compose_selected(original, candidate, base, (source,), (acquired,), proof=proof) == selected
        assert len(qa) == 1 and len(jev) == 2 and len(render) == 1


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


@contextmanager
def _retained_slot_alias_failure(tmp_path, monkeypatch, *, runtime_failure=False):
    """Record the original strict decoder through real accounting and CAS APIs."""
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch)
    state['claims'] = {'0': {**state['claims']['S1L1'], 'source_range': {
        'first_span_id': 'S1L1', 'last_span_id': 'S1L1'}}}
    decoder = module._renderings

    def original_decoder(raw, input_state, *, version=module.VERSION):
        if version == module.VERSION:
            returned = {row['span_id'] for row in json.loads(raw)['renderings']}
            if returned != set(input_state['claims']):
                if runtime_failure:
                    raise RuntimeError('Original runtime decoder failure')
                raise LocalisationHold('LOCALISATION_SPAN_PARTITION_HOLD')
        return decoder(raw, input_state, version=version)

    with open_native_runtime(**args) as runtime:
        localiser = _localiser(usage, runtime, fence, runner)
        scope = _scope(runtime)
        with monkeypatch.context() as previous:
            previous.setattr(module, '_renderings', original_decoder)
            with pytest.raises(RuntimeError if runtime_failure else LocalisationHold):
                localiser.localise(state, **scope)
        with sqlite3.connect(usage.path) as connection:
            invocation = connection.execute('SELECT invocation_id FROM model_invocation_allocations').fetchone()[0]
        reference = localiser._reference(invocation, proof=runtime.proof)
        yield localiser, usage, state, scope, reference, runtime, calls


def test_reported_slot_alias_failure_revalidates_same_paid_raw_without_redispatch(tmp_path, monkeypatch):
    with _retained_slot_alias_failure(tmp_path, monkeypatch) as (
            localiser, usage, state, scope, reference, runtime, calls):
        terminal = usage.terminal(reference.invocation_id)
        assert terminal.outcome == 'LOCALISATION_FAILED'
        assert terminal.failure_class == 'LocalisationHold'
        assert terminal.usage_status.value == 'REPORTED'
        before = runtime.authority.objects.rehydrate(HydrationRequest(
            reference.raw_admission_id, 'evidence.record'), proof=runtime.proof).data
        assert localiser.localise(state, **scope) == reference
        record = localiser.read_localisation(reference, state, **scope)
        assert record['outcome'] == 'LOCALISATION_FAILED'
        assert record['renderings']['0']['rendered_assertion_zh_hant_hk_fragments'] == ['計劃現已接受申請。']
        assert record['consumer_revalidation'] == {
            'consumer_contract': module.CONSUMER_VERSION,
            'original_outcome': 'LOCALISATION_FAILED',
            'original_terminal_digest': terminal.terminal_digest,
            'raw_response_digest': digest_bytes(before),
            'source_span_aliases': {'S1L1': '0'},
        }
        assert usage.terminal(reference.invocation_id) == terminal
        assert runtime.authority.objects.rehydrate(HydrationRequest(
            reference.raw_admission_id, 'evidence.record'), proof=runtime.proof).data == before
        assert len(calls) == 1
        with sqlite3.connect(usage.path) as connection:
            assert connection.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0] == 1


@pytest.mark.parametrize('condition', ['single_alias', 'direct', 'multi_span', 'duplicate_alias',
                                     'mixed_collision', 'duplicate_output', 'unknown', 'fragment_count', 'byte_bound'])
def test_source_span_aliases_preserve_exact_partition_and_existing_shape_guards(condition):
    from copy import deepcopy
    claim = {'text': 'The scheme is now open.', 'entities': [], 'rendering_fragment_count': 1,
             'source_range': {'first_span_id': 'S1L1', 'last_span_id': 'S1L1'}}
    state = {'claims': {'0': claim}}
    item = {'span_id': 'S1L1', 'rendered_assertion_zh_hant_hk_fragments': ['計劃現已接受申請。'],
            'factual_localisations': [], 'quotation_source_keys': []}
    value = {'renderings': [item]}
    if condition == 'direct':
        item['span_id'] = '0'
    elif condition == 'multi_span':
        claim['source_range']['last_span_id'] = 'S1L2'
    elif condition == 'duplicate_alias':
        state['claims']['1'] = deepcopy(claim)
        value['renderings'].append({**deepcopy(item), 'span_id': '1'})
    elif condition == 'mixed_collision':
        state['claims']['S1L1'] = {**deepcopy(claim), 'source_range': {
            'first_span_id': 'S1L2', 'last_span_id': 'S1L2'}}
        value['renderings'].append({**deepcopy(item), 'span_id': 'S1L2'})
    elif condition == 'duplicate_output':
        value['renderings'].append({**deepcopy(item), 'span_id': '0'})
    elif condition == 'unknown':
        item['span_id'] = 'S1L9'
    elif condition == 'fragment_count':
        item['rendered_assertion_zh_hant_hk_fragments'].append('多餘片段。')
    elif condition == 'byte_bound':
        item['quotation_source_keys'] = ['中' * 86]
    if condition in {'single_alias', 'direct'}:
        assert module._renderings(canonical_json_bytes(value), state) == {
            '0': {key: value for key, value in item.items() if key != 'span_id'}}
    else:
        with pytest.raises(LocalisationHold):
            module._renderings(canonical_json_bytes(value), state)


@pytest.mark.parametrize('corruption', ['claim', 'scope', 'raw_admission', 'receipt', 'recomputed_breach'])
def test_failed_alias_revalidation_authenticates_original_input_raw_and_usage(tmp_path, monkeypatch, corruption):
    from copy import deepcopy
    from dataclasses import replace
    with _retained_slot_alias_failure(tmp_path, monkeypatch) as (
            localiser, usage, state, scope, reference, runtime, calls):
        terminal = usage.terminal(reference.invocation_id)
        if corruption == 'claim':
            state = deepcopy(state)
            state['claims']['0']['text'] = 'A different assertion.'
        elif corruption == 'scope':
            scope = {**scope, 'candidate_id': 'other-candidate'}
        elif corruption == 'raw_admission':
            other = runtime.authority.objects.admit(ObjectAdmissionRequest('evidence.record', 'other-rendering'),
                canonical_json_bytes({'renderings': []}), proof=runtime.proof).admission
            reference = replace(reference, raw_admission_id=other.admission_id)
        elif corruption == 'receipt':
            receipt = json.loads(runtime.authority.objects.rehydrate(HydrationRequest(
                reference.receipt_admission_id, 'evidence.record'), proof=runtime.proof).data)
            receipt['raw_digest'] = digest_bytes(b'not the original rendering')
            other = runtime.authority.objects.admit(ObjectAdmissionRequest('evidence.record', 'other-rendering-receipt'),
                canonical_json_bytes(receipt), proof=runtime.proof).admission
            reference = replace(reference, receipt_admission_id=other.admission_id)
        else:
            monkeypatch.setattr(ModelUsageService, '_validate_terminal',
                                staticmethod(lambda *_a, **_k: 'MAX_TOTAL_TOKENS_EXCEEDED'))
        with pytest.raises(LocalisationHold):
            localiser.read_localisation(reference, state, **scope)
        assert len(calls) == 1
        assert usage.terminal(terminal.invocation_id) == terminal


def test_reported_runtime_failure_is_not_source_span_alias_revalidation_credit(tmp_path, monkeypatch):
    with _retained_slot_alias_failure(tmp_path, monkeypatch, runtime_failure=True) as (
            localiser, usage, state, scope, reference, _runtime, calls):
        terminal = usage.terminal(reference.invocation_id)
        assert terminal.usage_status.value == 'REPORTED'
        assert terminal.failure_class == 'RuntimeError'
        with pytest.raises(LocalisationHold, match='REPLAY_BINDING'):
            localiser.read_localisation(reference, state, **scope)
        assert len(calls) == 1 and usage.terminal(reference.invocation_id) == terminal


def test_retained_03df_alias_is_structural_not_rendering_admission_proof():
    # Exact retained 1,481-byte paid result, never a prescribed translation.
    raw = '{"renderings":[{"factual_localisations":[{"rendered_expression":"資訊：","source_lookup_key":"Information:"},{"rendered_expression":"2026至2027學年16至19歲資助大型課程額外資助的更改","source_lookup_key":"Change to the 16 to 19 funded large programme uplift for academic year 2026 to 2027"},{"rendered_expression":"16至19歲","source_lookup_key":"16 to 19"},{"rendered_expression":"資助","source_lookup_key":"funded"},{"rendered_expression":"大型課程額外資助","source_lookup_key":"large programme uplift"},{"rendered_expression":"2026至2027學年","source_lookup_key":"academic year 2026 to 2027"},{"rendered_expression":"我們已更改","source_lookup_key":"We have changed"},{"rendered_expression":"將該額外資助集中於","source_lookup_key":"to focus the uplift on"},{"rendered_expression":"數學及高價值A level課程","source_lookup_key":"maths and high value A level programmes"},{"rendered_expression":"以支援學生進入優先行業","source_lookup_key":"to support the progression of students into priority sectors"},{"rendered_expression":"優先行業","source_lookup_key":"priority sectors"}],"quotation_source_keys":[],"rendered_assertion_zh_hant_hk_fragments":["資訊：2026至2027學年16至19歲資助大型課程額外資助的更改。我們已更改2026至2027學年的大型課程額外資助，將該額外資助集中於數學及高價值A level課程，以支援學生進入優先行業。"],"span_id":"S1L4"}]}'.encode()
    assert digest_bytes(raw) == 'sha256:6f4103dfd5271201a19062d3f038261c8fcc86810582bd7ca152b59a3f7c8364'
    state = {'claims': {'0': {'text': 'Information: Change to the 16 to 19 funded large programme uplift for academic year 2026 to 2027 We have changed the large programme uplift for academic year 2026 to 2027 to focus the uplift on maths and high value A level programmes to support the progression of students into priority sectors. ', 'entities': [], 'rendering_fragment_count': 1,
        'source_range': {'first_span_id': 'S1L4', 'last_span_id': 'S1L4'}}}}
    decoded = module._renderings(raw, state, version=module.VERSION)
    assert set(decoded) == {'0'}
    with pytest.raises(LocalisationHold) as held:
        module._validate_content(decoded, state)
    assert set(held.value.reason_codes) >= {'LOCALISATION_FACT_EQUIVALENCE_HOLD', 'LOCALISATION_LANGUAGE_HOLD', 'LOCALISATION_NUMERIC_HOLD'}
