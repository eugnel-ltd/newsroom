"""Pure composition over already read receipts; no provider/authority opener."""
from copy import deepcopy
from types import SimpleNamespace as N
import json

import pytest
from newsroom.authority import ObjectAdmissionId
from newsroom.authority.canonical import canonical_json_bytes, digest_canonical, digest_bytes
from newsroom.control_plane.native_assessor import (
    NativeAssessmentExecution, AutonomousNativeEvidenceAssessor, _materialise_reference_result,
    _reference_binding, VERSION as CODEC,
)
from newsroom.control_plane.native_assessor_judgments import JudgedAssessment
from newsroom.control_plane.native_assessor_spans import build_lossless_source_view
from newsroom.control_plane.evidence import EvidencePackage
from newsroom.sources.types import SourceRole, SourceRoleAssignment

BODY = ('The government launched a public consultation.\n'
        'The government is proposing to regulate 2 chemicals under UK law.\n'
        'Security Minister Dan Jarvis said: Protecting the public is our first duty.')


def _case():
    view = build_lossless_source_view((BODY,), ('UK-fixture',))
    base = EvidencePackage('candidate', 'hypothesis', ('signal',), ('lead',),
        ('UK-fixture',), (digest_bytes(BODY.encode()),), (BODY,))
    original_binding = {'content_digest': base.digest, 'source_reference_binding': _reference_binding(view),
        'candidate_id': 'candidate', 'candidate_version_id': 'version', 'hypothesis_digest': digest_bytes(b'hypothesis'),
        'source_currentness': [{'source_id': 'UK-fixture', 'definition_id': 'definition', 'definition_version_id': 'source-version'}]}
    head = {'claim_role': 'HEADLINE', 'status': 'CONFIRMED_FACT',
        'source_range': {'first_span_id': 'S1L1', 'last_span_id': 'S1L1'},
        'rendered_assertion_zh_hant_hk_fragments': ['政府已啟動公眾諮詢。'],
        'factual_localisations': [], 'quotation_source_keys': []}
    wire = {'package': {'select_new_information': True, 'governed_claims': [head],
        'qualification_evidence': [{'claim_index': 0, 'test': 'OFFICIAL_ACTION_OR_DEADLINE',
            'test_evidence': {'action_class': 'PROCESS', 'event_polarity': 'AFFIRMED',
                'action_relation': 'NEW_OR_CHANGED_OFFICIAL_ACTION',
                'material_relation_span_source_lookup_key': 'The government launched a public consultation.',
                'reader_action_source_lookup_key': 'The government launched a public consultation.'}}],
        'selection_rationale': 'The public consultation has launched.', 'geography': ['UK'],
        'categories': ['Politics and law'], 'explicit_exclusions': []}}
    _, receipt = _materialise_reference_result(canonical_json_bytes(wire), view, digest_canonical(original_binding), CODEC)
    decision = {'schema': 'newsroom.native-source-qualification.v2', 'source_binding': original_binding,
        'materialisation_receipt': receipt, 'qualification_reference': {'invocation_id': digest_bytes(b'original invocation')}}
    original = JudgedAssessment(NativeAssessmentExecution(receipt['materialised_text'], {}),
        canonical_json_bytes(decision), ObjectAdmissionId.new())
    ranges = {identity: {'first_span_id': identity, 'last_span_id': identity} for identity in ('S1L2', 'S1L3')}
    binding = {**original_binding, 'context_purpose': 'newsroom.native-context-package.v1',
        'context_original_receipt_digest': digest_canonical(receipt), 'context_ranges': ranges}
    renderings = {
        'S1L2': {'rendered_assertion_zh_hant_hk_fragments': ['政府正提出按', '法律管制2種化學物質。'],
            'factual_localisations': [], 'quotation_source_keys': []},
        'S1L3': {'rendered_assertion_zh_hant_hk_fragments': ['保安事務大臣', '表示，保障公眾是首要責任。'],
            'factual_localisations': [], 'quotation_source_keys': []},
    }
    claims = [{'claim_role': 'CONTEXT', 'status': 'CONFIRMED_FACT', 'source_range': ranges[k], **deepcopy(v)}
              for k, v in renderings.items()]
    context_wire = {'package': {'select_new_information': False, 'governed_claims': claims,
        'qualification_evidence': [], 'selection_rationale': 'Supported context only.',
        'geography': [], 'categories': [], 'explicit_exclusions': []}}
    localisation = {'source_binding': binding, 'renderings': renderings,
        'invocation_id': digest_bytes(b'localisation'), 'terminal_digest': digest_bytes(b'localisation terminal')}
    answers = {f'{identity}:{field}': {'choice': 'SUPPORTED' if field == 'support' else 'YES'}
               for identity in ranges for field in ('support', 'modality', 'attribution', 'entities')}
    support = {'snapshot': {'source_binding': binding}, 'answers': answers,
        'invocation_id': digest_bytes(b'support'), 'terminal_digest': digest_bytes(b'support terminal')}
    role = SourceRoleAssignment(SourceRole.ORIGINATING_AUTHORITY, 'Official public process', ())
    h = digest_bytes(b'disposable diagnostic provenance')
    source = N(unit=N(source_id='UK-fixture', authority=N(definition_id='definition')),
        source_version=N(canonical_digest=h, request=N(roles=(role,))), rights=N(record_id=h),
        dependency=N(record_id=h, evidential_origin_id=h))
    acquired = N(body=BODY.encode(), receipt_digest=h, publisher='Official authority',
        currentness_basis='AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT', publication_time='2026-10-04T00:00:00Z',
        source_updated_time='2026-10-04T00:00:00Z', retrieval_time='2026-10-04T00:01:00Z',
        transport_evidence_digest=h, body_origin='GOVUK_CONTENT_API_PAGE_TEXT')
    return original, context_wire, localisation, binding, view, support, base, source, acquired


def _inputs():
    """Shared orchestration fixture; no authority is implied by these labels."""
    original, wire, localisation, binding, view, support, *_ = _case()
    return original, view, binding, wire, localisation, support


def _compose(case):
    from newsroom.control_plane.native_context_materialisation import compose_context_execution
    original, wire, localisation, binding, view, support, *_ = case
    return compose_context_execution(original, wire, localisation, binding=binding, view=view, support_receipt=support)


def test_context_composition_preserves_original_headline_and_qualification_and_exact_context():
    case = _case()
    before = deepcopy(case[:6])
    execution, proof = _compose(case)
    original_package = json.loads(case[0].execution.text)['package']
    package = json.loads(execution.text)['package']
    assert package['governed_claims'][0] == original_package['governed_claims'][0]
    assert package['qualification_evidence'] == original_package['qualification_evidence']
    assert [c['claim_role'] for c in package['governed_claims']] == ['HEADLINE', 'CONTEXT', 'CONTEXT']
    assert package['governed_claims'][1]['claim'] == BODY.splitlines()[1]
    assert 'UK' in package['governed_claims'][1]['rendered_assertion_zh_hant_hk']
    assert 'Dan Jarvis' in package['governed_claims'][2]['rendered_assertion_zh_hant_hk']
    assert proof['original_receipt_digest'] == case[3]['context_original_receipt_digest']
    assert proof['combined_digest'] == digest_bytes(execution.text.encode())
    assert case[:6] == before
    _, _, _, _, _, _, base, source, acquired = case
    assessment = AutonomousNativeEvidenceAssessor._validated_execution(execution, N(), base, (source,), (acquired,))
    assert len(assessment.governed_claims) == 3
    assert len(assessment.qualification_evidence) == 1


@pytest.mark.parametrize('mutation', ('body', 'execution', 'nesting', 'rekey', 'candidate', 'duplicate_range'))
def test_context_composition_denies_original_tamper_nesting_and_rekey(mutation):
    from dataclasses import replace
    from newsroom.control_plane.native_context_materialisation import ContextCompositionError
    case = list(_case())
    decision = json.loads(case[0].decision_record)
    if mutation == 'body':
        decision['materialisation_receipt']['body_digests'] = [digest_bytes(b'other Source')]
        case[0] = replace(case[0], decision_record=canonical_json_bytes(decision))
    elif mutation == 'execution':
        case[0] = replace(case[0], execution=NativeAssessmentExecution('{}', {}))
    elif mutation == 'nesting':
        decision['schema'] = 'newsroom.native-context-package.v1'
        case[0] = replace(case[0], decision_record=canonical_json_bytes(decision))
    elif mutation == 'rekey':
        receipt = decision['materialisation_receipt']
        receipt['request_identity'] = digest_bytes(b'new original purpose')
        receipt.pop('receipt_digest')
        receipt['receipt_digest'] = digest_canonical(receipt)
        case[0] = replace(case[0], decision_record=canonical_json_bytes(decision))
    elif mutation == 'candidate':
        case[3]['candidate_id'] = 'other candidate'
    else:
        case[3]['context_ranges']['S1L3'] = deepcopy(case[3]['context_ranges']['S1L2'])
    with pytest.raises(ContextCompositionError):
        _compose(case)


@pytest.mark.parametrize('mutation', ('alias', 'inforce', 'uncertain', 'missing_answer', 'other_scope', 'other_localisation', 'headline', 'unbound_range'))
def test_context_requires_bound_assertion_modality_attribution_and_entity_results(mutation):
    from newsroom.control_plane.native_context_materialisation import ContextCompositionError
    case = list(_case())
    if mutation == 'alias':
        rendering = case[1]['package']['governed_claims'][1]['rendered_assertion_zh_hant_hk_fragments']
        rendering[0] = '英國內政部保安事務大臣'
        case[2]['renderings']['S1L3']['rendered_assertion_zh_hant_hk_fragments'] = rendering
        case[5]['answers']['S1L3:entities']['choice'] = 'NO'
    elif mutation == 'inforce':
        rendering = case[1]['package']['governed_claims'][0]['rendered_assertion_zh_hant_hk_fragments']
        rendering[0] = '政府現已按'
        case[2]['renderings']['S1L2']['rendered_assertion_zh_hant_hk_fragments'] = rendering
        case[5]['answers']['S1L2:modality']['choice'] = 'NO'
    elif mutation == 'uncertain':
        case[5]['answers']['S1L2:support']['choice'] = 'UNCERTAIN'
    elif mutation == 'missing_answer':
        del case[5]['answers']['S1L3:attribution']
    elif mutation == 'other_scope':
        case[5]['snapshot']['source_binding'] = {**case[3], 'candidate_id': 'another'}
    elif mutation == 'other_localisation':
        case[2]['renderings']['S1L3']['rendered_assertion_zh_hant_hk_fragments'][1] = '未經證明的新陳述。'
    elif mutation == 'headline':
        case[1]['package']['governed_claims'][0]['claim_role'] = 'HEADLINE'
    else:
        case[1]['package']['governed_claims'][0]['source_range'] = {'first_span_id': 'S99L1', 'last_span_id': 'S99L1'}
    with pytest.raises(ContextCompositionError):
        _compose(case)


def test_context_static_consumer_still_denies_extra_entity_and_unproved_fact_pair():
    from newsroom.increment10.evidence import EvidencePackageError
    case = _case()
    execution, _ = _compose(case)
    for mutation in ('entity', 'unproved_fact_pair'):
        document = json.loads(execution.text)
        claim = document['package']['governed_claims'][1]
        if mutation == 'entity':
            claim['rendered_assertion_zh_hant_hk'] = 'Home Office ' + claim['rendered_assertion_zh_hant_hk']
        else:
            claim['localised_factual_expressions'] = [['2 years', '3年']]
        with pytest.raises((EvidencePackageError, ValueError)):
            AutonomousNativeEvidenceAssessor._validated_execution(
                NativeAssessmentExecution(canonical_json_bytes(document).decode(), {}), N(), case[6], (case[7],), (case[8],))


def test_context_receipt_roundtrip_replay_is_identical_and_keeps_closed_original_bytes():
    case = list(_case())
    original_bytes = case[0].decision_record
    first = _compose(case)
    # An already read materialisation receipt is the second supported input
    # shape. This is a serialisation/replay proof, not an authority-store OPEN.
    case[0] = json.loads(original_bytes)['materialisation_receipt']
    for index in (1, 2, 3, 5):
        case[index] = json.loads(canonical_json_bytes(case[index]))
    assert _compose(case) == first
    assert json.loads(original_bytes)['materialisation_receipt']['receipt_digest'] == first[1]['original_materialisation_digest']


def test_context_multispan_ranges_use_existing_entity_order_and_source_decoder():
    case = list(_case())
    binding = case[3]
    binding['context_ranges'] = {'combined': {'first_span_id': 'S1L2', 'last_span_id': 'S1L3'}}
    rendering = {'rendered_assertion_zh_hant_hk_fragments': ['政府正提出按', '法律管制2種化學物質。保安事務大臣', '表示，保障公眾是首要責任。'],
        'factual_localisations': [], 'quotation_source_keys': []}
    case[1]['package']['governed_claims'] = [{'claim_role': 'CONTEXT', 'status': 'CONFIRMED_FACT',
        'source_range': binding['context_ranges']['combined'], **rendering}]
    case[2]['renderings'] = {'combined': deepcopy(rendering)}
    case[5]['answers'] = {f'combined:{field}': {'choice': 'SUPPORTED' if field == 'support' else 'YES'}
        for field in ('support', 'modality', 'attribution', 'entities')}
    execution, _ = _compose(case)
    result = json.loads(execution.text)['package']['governed_claims'][1]
    assert result['claim'] == '\n'.join(BODY.splitlines()[1:])
    assert result['rendered_assertion_zh_hant_hk'].count('UK') == 1
    assert result['rendered_assertion_zh_hant_hk'].count('Dan Jarvis') == 1
    assert AutonomousNativeEvidenceAssessor._validated_execution(execution, N(), case[6], (case[7],), (case[8],)).governed_claims
