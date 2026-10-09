"""Exact retained v4 responses, not provider calls or admission claims."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane import native_claim_localisation as m

FIXTURES = json.loads((Path(__file__).parent / 'fixtures/source_occurrence_rendering_v4.json').read_text())


def test_occurrence_consumer_keeps_the_exact_qualified_v4_producer_contract():
    from dataclasses import asdict
    from newsroom.authority.canonical import digest_canonical
    policy = asdict(m.localisation_policy(evidence_digest=digest_bytes(b'producer-equivalence'), qualified=True))
    for key in ('canonical_digest', 'implementation_revision', 'evidence_digest'):
        policy.pop(key)
    # Captured independently from base 6db01642, before the consumer change.
    assert digest_canonical({'policy': policy, 'system_digest': digest_bytes(m.TYPED_SYSTEM.encode()),
                             'schema_digest': m.TYPED_SCHEMA_DIGEST}) == 'sha256:b971677993e7fc5beb3e1dad08585c492a922e036eac12483e7725fbbd20d6a9'


def test_indefinite_classifier_does_not_erase_a_post_nominal_restriction():
    from newsroom.control_plane.evidence import factual_rendering_is_bound_v3
    assert factual_rendering_is_bound_v3('Complete a short survey.', '填寫一份簡短調查。')
    assert not factual_rendering_is_bound_v3('Complete a short survey.', '填寫一份簡短調查而已。')
    assert not factual_rendering_is_bound_v3('Recruit 6,500 teachers.', '招聘6,500名教師而已。')


@pytest.mark.parametrize('source,rendered', [
    ('These changes are part of the programme.', '這些改變是計劃的二十一部分。'),
    ('Complete a short survey.', '填寫二十一個簡短調查。'),
    ('Complete a short survey.', '填寫第一個簡短調查。'),
    ('These changes are part of the programme.', '這些改變是計劃的第一部分。'),
    ('Recruit 6,500 teachers.', '招聘負6,500名教師。'),
    ('Recruit 6,500 teachers.', '招聘負 6,500名教師。'),
    ('Recruit 6,500 teachers.', '招聘負六千五百名教師。'),
    ('Complete a short survey.', '填寫一個半簡短調查。'),
    ('Claims are due at year-end.', '申索於明年終到期。'),
])
def test_review_counterexamples_preserve_complete_cardinality_and_sign(source, rendered):
    from newsroom.control_plane.evidence import factual_rendering_is_bound_v3
    assert not factual_rendering_is_bound_v3(source, rendered)


@pytest.mark.parametrize('prefix', ['第', '頭', '首', '只得', '僅得', '至少'])
@pytest.mark.parametrize('boundary', ['primitive', 'full_consumer'])
@pytest.mark.parametrize('fixture_index,span,old,new,source', [
    (0, '2', '一部分', '一部分', 'These changes are part of the programme.'),
    (1, '1', '一份簡短調查', '一個簡短調查', 'Complete a short survey.'),
])
def test_grammatical_exemptions_preserve_ordinal_and_restrictive_prefixes(prefix, boundary, fixture_index, span, old, new, source):
    from newsroom.control_plane.evidence import SOURCE_RENDERING_CONTRACT_V3, factual_rendering_is_bound_v3
    if boundary == 'primitive':
        assert not factual_rendering_is_bound_v3(source, '這是' + prefix + new + '。')
        return
    fixture = deepcopy(FIXTURES[fixture_index])
    row = next(row for row in fixture['raw']['renderings'] if row['span_id'] == span)
    fragments = row['rendered_assertion_zh_hant_hk_fragments']
    index = next(index for index, text in enumerate(fragments) if old in text)
    fragments[index] = fragments[index].replace(old, prefix + new, 1)
    with pytest.raises(m.LocalisationHold, match='LOCALISATION_CONTENT_CONTRACT_HOLD') as caught:
        m._renderings(canonical_json_bytes(fixture['raw']), fixture['state'], version=m.TYPED_VERSION,
                      consumer_contract=SOURCE_RENDERING_CONTRACT_V3)
    assert 'LOCALISATION_NUMERIC_HOLD' in caught.value.reason_codes


@pytest.mark.parametrize('fixture', FIXTURES, ids=lambda row: row['candidate_id'][:8])
def test_reported_v4_occurrence_consumer_preserves_wire_and_only_corrects_three(fixture):
    from newsroom.control_plane.evidence import SOURCE_RENDERING_CONTRACT_V3
    state, raw = deepcopy(fixture['state']), canonical_json_bytes(fixture['raw'])
    assert digest_bytes(m._prompt(state, version=m.TYPED_VERSION).encode()) == fixture['original_prompt_digest']
    original = deepcopy(state)
    with pytest.raises(m.LocalisationHold, match='LOCALISATION_CONTENT_CONTRACT_HOLD'):
        m._renderings(raw, state, version=m.TYPED_VERSION)
    if fixture['accepted']:
        assert m._renderings(raw, state, version=m.TYPED_VERSION, consumer_contract=SOURCE_RENDERING_CONTRACT_V3)
    else:
        with pytest.raises(m.LocalisationHold):
            m._renderings(raw, state, version=m.TYPED_VERSION, consumer_contract=SOURCE_RENDERING_CONTRACT_V3)
    assert state == original


@pytest.mark.parametrize('candidate,span,old,new', [
    (0, '0', '星期五', '星期四'),
    (0, '2', '6,500', '650'),
    (0, '2', '6,500名', '只有6,500名'),
    (0, '0', '12個月', '12年'),
    (0, '1', '課程', '三個月課程'),
    (0, '1', '課程', '一公升課程'),
    (0, '1', '課程', '「課程」'),
    (0, '2', '一部分', '二部分'),
    (0, '2', '一部分', '二十一部分'),
    (0, '2', '一部分', '第一部分'),
    (0, '2', '6,500名', '負6,500名'),
    (0, '2', '6,500名', '負 6,500名'),
    (0, '2', '6,500名', '負六千五百名'),
    (0, '2', '一部分', '只有一部分'),
    (0, '1', '課程', 'InventedAlias課程'),
    (1, '1', '一份簡短調查', '只有一份簡短調查'),
    (1, '1', '一份簡短調查', '僅一個教師'),
    (1, '1', '一份簡短調查', '最多一份簡短調查'),
    (1, '1', '一份簡短調查', '兩份簡短調查'),
    (1, '1', '一份簡短調查', '二十一個簡短調查'),
    (1, '1', '一份簡短調查', '第一個簡短調查'),
    (1, '1', '一份簡短調查', '一個半簡短調查'),
    (2, '0', 'Ofqual 特此', 'Ofqual Ofqual 特此'),
])
def test_occurrence_consumer_keeps_quantities_qualifiers_literals_and_quotes(candidate, span, old, new):
    from newsroom.control_plane.evidence import SOURCE_RENDERING_CONTRACT_V3
    fixture = deepcopy(FIXTURES[candidate])
    row = next(row for row in fixture['raw']['renderings'] if row['span_id'] == span)
    fragments = row['rendered_assertion_zh_hant_hk_fragments']
    index = next(index for index, text in enumerate(fragments) if old in text)
    fragments[index] = fragments[index].replace(old, new, 1)
    with pytest.raises(m.LocalisationHold):
        m._renderings(canonical_json_bytes(fixture['raw']), fixture['state'], version=m.TYPED_VERSION,
                      consumer_contract=SOURCE_RENDERING_CONTRACT_V3)


def test_occurrence_overlay_cannot_shrink_a_larger_relative_fact_in_full_rendering():
    from newsroom.control_plane.evidence import SOURCE_RENDERING_CONTRACT_V3
    source = 'Claims are due at year-end.'
    state = {'source_binding': {'content_digest': digest_bytes(source.encode())},
        'claims': {'0': {'source_id': 'fixture', 'text': source, 'entities': [], 'rendering_fragment_count': 1}}}
    raw = canonical_json_bytes({'renderings': [{'span_id': '0',
        'rendered_assertion_zh_hant_hk_fragments': ['申索於明年終到期。'],
        'factual_localisations': [], 'quotation_source_keys': []}]})
    with pytest.raises(m.LocalisationHold, match='LOCALISATION_CONTENT_CONTRACT_HOLD') as caught:
        m._renderings(raw, state, version=m.TYPED_VERSION, consumer_contract=SOURCE_RENDERING_CONTRACT_V3)
    assert caught.value.reason_codes == ('LOCALISATION_NUMERIC_HOLD',)


@pytest.mark.parametrize('fixture', [row for row in FIXTURES if row['accepted']], ids=lambda row: row['candidate_id'][:8])
def test_reported_v4_failure_reuses_exact_raw_and_terminal_without_another_call(tmp_path, monkeypatch, fixture):
    from newsroom.control_plane.evidence import SOURCE_RENDERING_CONTRACT_V3
    from newsroom.control_plane.native_runtime import open_native_runtime
    from newsroom.tests.test_native_claim_localisation import case, NOW, _scope
    from newsroom.authority import HydrationRequest
    import sqlite3
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch, payload=fixture['raw'])
    state.clear(); state.update(deepcopy(fixture['state']))
    with open_native_runtime(**args) as runtime:
        localiser = m.NativeClaimLocaliser(usage=usage, objects=runtime.authority.objects,
            policy=m.localisation_policy(evidence_digest=digest_bytes(b'occurrence fixture'), qualified=True),
            source_fence=fence, runner=runner, implementation_worktree_clean=True, clock=lambda: NOW)
        scope = _scope(runtime)
        reference = localiser.localise(state, **scope)
        before = usage.terminal(reference.invocation_id)
        assert before.outcome == 'LOCALISATION_FAILED' and before.failure_class == 'LocalisationHold'
        raw = runtime.authority.objects.rehydrate(HydrationRequest(reference.raw_admission_id, 'evidence.record'), proof=runtime.proof).data
        for _ in range(2):
            assert localiser.localise(state, **scope) == reference
            checked = localiser.read_localisation(reference, state, **scope)
            assert checked['outcome'] == 'LOCALISATION_FAILED'
            assert checked['consumer_revalidation']['consumer_contract'] == SOURCE_RENDERING_CONTRACT_V3
            assert checked['consumer_revalidation']['retry_authorised'] is False
            assert checked['original_state'] == state
            assert digest_bytes(m._prompt(state, version=m.TYPED_VERSION).encode()) == fixture['original_prompt_digest']
        assert usage.terminal(reference.invocation_id) == before and len(calls) == 1
        assert runtime.authority.objects.rehydrate(HydrationRequest(reference.raw_admission_id, 'evidence.record'), proof=runtime.proof).data == raw
        with sqlite3.connect(usage.path) as connection:
            assert connection.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0] == 1


@pytest.mark.parametrize('fixture', [row for row in FIXTURES if row['accepted']], ids=lambda row: row['candidate_id'][:8])
def test_three_occurrence_outputs_reach_governed_claim_admitted_reader_and_writer(tmp_path, monkeypatch, fixture):
    from dataclasses import replace
    from datetime import UTC, datetime
    from newsroom.control_plane.evidence import SOURCE_RENDERING_CONTRACT_V3
    from newsroom.control_plane.native_assessor_judgments import NativeSemanticWitnesses, source_rendering_projection
    from newsroom.control_plane.admission import source_rendering_is_admitted, _admitted_claim_names
    from newsroom.control_plane.writer import WriterCliExecution, _typed_claim_numeric_relation
    from newsroom.tests.test_native_assessor_judgments import _case
    from newsroom.tests.test_document_year_fidelity import package as template_package
    body = fixture['state']['source_binding']['current_scope']['sources'][0]['body']
    with _case(tmp_path, monkeypatch, body=body, source_id='UK-05') as (consumer, service, candidate, base, source, acquired, usage, votes):
        state = deepcopy(fixture['state'])
        state['source_binding'].update(content_digest=base.digest, candidate_id=candidate.candidate_id,
            candidate_version_id=candidate.version_id, hypothesis_digest=candidate.governing_manifest.canonical_digest,
            evidence_package_digest=base.digest)
        calls = []
        def runner(prompt):
            calls.append(prompt)
            return WriterCliExecution(canonical_json_bytes(fixture['raw']).decode(), {
                'usage_basis': 'PROVIDER_REPORTED', 'input_tokens': 100, 'output_tokens': 30, 'total_tokens': 130})
        localiser = m.NativeClaimLocaliser(usage=usage, objects=service.objects,
            policy=m.localisation_policy(evidence_digest=digest_bytes(b'occurrence fixture'), qualified=True),
            source_fence=service.fence, runner=runner, implementation_worktree_clean=True,
            clock=lambda: datetime(2026, 10, 4, tzinfo=UTC))
        scope = dict(candidate_id=candidate.candidate_id, hypothesis_digest=candidate.governing_manifest.canonical_digest,
            evidence_package_digest=base.digest, proof=consumer.proof)
        ref = localiser.localise(state, **scope)
        checked = localiser.read_localisation(ref, state, **scope)
        source_ref = tuple(sorted({'contract': SOURCE_RENDERING_CONTRACT_V3, 'operation': 'SOURCE_RENDERING',
            'invocation_id': ref.invocation_id, 'raw_admission_id': str(ref.raw_admission_id),
            'receipt_admission_id': str(ref.receipt_admission_id)}.items()))
        template = template_package().governed_claims[0]
        claims = []
        for identity, selected in source_rendering_projection(state)['claims'].items():
            item = checked['renderings'][identity]; names = tuple(name for name, _ in selected['entities'])
            fragments = item['rendered_assertion_zh_hant_hk_fragments']
            text = fragments[0] + ''.join(name + part for name, part in zip(names, fragments[1:], strict=True))
            claims.append(replace(template, claim_id='occurrence-' + identity, claim=selected['text'],
                supporting_excerpt=selected['text'], passage_index=0, source_ids=base.source_ids,
                rendered_assertion_zh_hant_hk=text, quotations=tuple(item['quotation_source_keys']),
                named_entities=names, rendered_named_entities=names,
                named_entity_evidence=tuple((name, kind, 'name-' + str(index)) for index, (name, kind) in enumerate(selected['entities'])),
                localised_factual_expressions=tuple((pair['source_lookup_key'], pair['rendered_expression']) for pair in item['factual_localisations']),
                source_rendering_ref=source_ref))
        value = replace(base, governed_claims=tuple(claims))
        verifier = NativeSemanticWitnesses(judgments=service, candidate_for=lambda _: candidate,
            proof=consumer.proof, require_current=lambda: None)
        verifier.rendering_reader = localiser.read_localisation
        for claim in claims:
            assert source_rendering_is_admitted(claim, value, semantic_witness_reader=verifier.read)
            assert _admitted_claim_names(claim, value) == frozenset((name, kind) for name, kind, _ in claim.named_entity_evidence)
            assert _typed_claim_numeric_relation(claim) is True
            forged = replace(claim, source_rendering_ref=tuple((key, digest_bytes(b'forged')) if key == 'invocation_id' else (key, part) for key, part in source_ref))
            with pytest.raises((ValueError, m.LocalisationHold)):
                source_rendering_is_admitted(forged, value, semantic_witness_reader=verifier.read)
            from newsroom.control_plane.evidence import SOURCE_RENDERING_CONTRACT_V2
            wrong_consumer = replace(claim, source_rendering_ref=tuple((key, SOURCE_RENDERING_CONTRACT_V2) if key == 'contract' else (key, part) for key, part in source_ref))
            with pytest.raises(ValueError, match='consumer identity'):
                source_rendering_is_admitted(wrong_consumer, value, semantic_witness_reader=verifier.read)
        assert len(calls) == 1 and votes == []


def test_sourceqa_composition_attaches_occurrence_identity_without_requalifying_selection(tmp_path, monkeypatch):
    from dataclasses import replace
    from newsroom.tests.test_qualification_semantic_witness import _selected_qualification_case
    from newsroom.control_plane.evidence import SOURCE_RENDERING_CONTRACT_V3
    from newsroom.control_plane.native_assessor import AutonomousNativeEvidenceAssessor
    from newsroom.control_plane.admission import source_rendering_is_admitted
    from newsroom.control_plane.writer import _typed_claim_numeric_relation
    with _selected_qualification_case(tmp_path, monkeypatch, fault='occurrence', typed_rendering=True, malformed_rendering=True) as (
            post, witness, original, candidate, base, source, acquired, scope, proof, usage, qa, votes, renders):
        original_bytes = original.decision_record
        selected = post.compose_selected(original, candidate, base, (source,), (acquired,), proof=proof)
        assert {dict(ref)['contract'] for _, ref in selected.source_renderings.references} == {SOURCE_RENDERING_CONTRACT_V3}
        assessment = AutonomousNativeEvidenceAssessor._validated_execution(selected.execution, candidate, base,
            (source,), (acquired,), semantic_witnesses=selected.semantic_witnesses,
            semantic_witness_reader=witness.read, source_renderings=selected.source_renderings)
        value = replace(base, governed_claims=assessment.governed_claims,
            qualification_evidence=assessment.qualification_evidence,
            substantive_new_information=assessment.substantive_new_information)
        for claim in value.governed_claims:
            assert source_rendering_is_admitted(claim, value, semantic_witness_reader=witness.read)
            assert _typed_claim_numeric_relation(claim) is True
        assert post.compose_selected(original, candidate, base, (source,), (acquired,), proof=proof) == selected
        assert original.decision_record == original_bytes
        assert len(qa) == len(renders) == 1 and len(votes) == 2  # Existing selection + criterion, no new vote.
        with usage._connection() as connection:
            terminal = connection.execute("SELECT t.record_json FROM model_invocation_terminals t JOIN model_invocation_allocations a USING(invocation_id) WHERE a.route='NATIVE_CLAIM_LOCALISATION'").fetchone()[0]
        assert json.loads(terminal)['outcome'] == 'LOCALISATION_FAILED'


@pytest.mark.parametrize('fault', ['unknown', 'prompt', 'breach', 'fence', 'schema', 'unsupported-clock', 'unsupported-title'])
def test_consumer_revalidation_never_grants_retry_for_unproved_history(tmp_path, monkeypatch, fault):
    from contextlib import contextmanager
    from newsroom.control_plane.native_runtime import open_native_runtime
    from newsroom.tests.test_native_claim_localisation import case, NOW, _scope
    from newsroom.control_plane.writer import WriterCliExecution
    from newsroom.control_plane.model_usage import ModelUsageService
    from jsonschema import ValidationError
    fixture = deepcopy(FIXTURES[2 if fault == 'unsupported-clock' else 3 if fault == 'unsupported-title' else 0])
    if fault == 'schema':
        fixture['raw']['renderings'][0]['span_id'] = 0
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch, payload=fixture['raw'])
    state.clear(); state.update(fixture['state'])
    if fault == 'unknown':
        def runner(prompt):
            calls.append(prompt)
            return WriterCliExecution(canonical_json_bytes(fixture['raw']).decode(), {'usage_basis': 'UNREPORTED'})
    with open_native_runtime(**args) as runtime:
        localiser = m.NativeClaimLocaliser(usage=usage, objects=runtime.authority.objects,
            policy=m.localisation_policy(evidence_digest=digest_bytes(b'occurrence fixture'), qualified=True),
            source_fence=fence, runner=runner, implementation_worktree_clean=True, clock=lambda: NOW)
        scope = _scope(runtime)
        if fault in {'unknown', 'schema', 'unsupported-clock', 'unsupported-title'}:
            with pytest.raises((m.LocalisationHold, ValidationError)):
                localiser.localise(state, **scope)
            with usage._connection() as connection:
                invocation = connection.execute('SELECT invocation_id FROM model_invocation_allocations').fetchone()[0]
            ref = localiser._reference(invocation, proof=runtime.proof)
        else:
            ref = localiser.localise(state, **scope)
            invocation = ref.invocation_id
        before = usage.terminal(invocation)
        if fault == 'unknown':
            assert before.usage_status.value in {'UNKNOWN', 'ESTIMATED'}
        supplied = deepcopy(state)
        if fault == 'prompt':
            supplied['claims']['0']['forged_prompt_field'] = 'not in paid input'
        elif fault == 'breach':
            monkeypatch.setattr(ModelUsageService, '_validate_terminal', staticmethod(lambda *_args, **_kw: 'MAX_TOTAL_TOKENS_EXCEEDED'))
        elif fault == 'fence':
            @contextmanager
            def stopped(_binding, _proof):
                raise m.LocalisationHold('fixture-currentness-fence')
                yield
            localiser.fence = stopped
        for _ in range(2):
            with pytest.raises(m.LocalisationHold):
                localiser.read_localisation(ref, supplied, **scope)
        assert usage.terminal(invocation) == before and len(calls) == 1
        with usage._connection() as connection:
            assert connection.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0] == 1
