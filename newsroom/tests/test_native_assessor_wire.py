"""The smaller v18 wire reuses the unchanged v17 source materialiser."""

import json

import pytest
from jsonschema import Draft202012Validator

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.control_plane.native_assessor import SCHEMA
from newsroom.control_plane.native_assessor_references import (
    MAX_RESULT_BYTES, SourceReferenceError, build_source_view,
    make_provider_schema as make_v17_schema,
    materialise as materialise_v17,
)
from newsroom.control_plane.native_assessor_wire import make_provider_schema, materialise

V17_SCHEMA = make_v17_schema(SCHEMA)
V18_SCHEMA = make_provider_schema(V17_SCHEMA)
BODY = 'Alice Smith said the deadline changed.\nNext line has 60 minutes.\n'


def _wire(first='S1L1'):
    return {'package': {
        'substantive_claim_indexes': [0],
        'governed_claims': [{
            'source_range': {'first_span_id': first, 'last_span_id': first},
            'rendered_assertion_zh_hant_hk_fragments': ['', ' 表示限期已更改。'],
            'factual_localisations': [], 'quotation_source_keys': [],
            'status': 'CONFIRMED_FACT', 'claim_role': 'HEADLINE',
        }],
        'qualification_evidence': [], 'selection_rationale': 'Exact source line',
        'geography': [], 'categories': [], 'explicit_exclusions': [],
    }}


def test_v18_schema_is_static_closed_and_drops_only_controller_constants():
    Draft202012Validator.check_schema(V18_SCHEMA)
    assert V18_SCHEMA == make_provider_schema(V17_SCHEMA)
    claim = V18_SCHEMA['properties']['package']['properties']['governed_claims']['items']
    assert claim['additionalProperties'] is False
    assert set(claim['properties']) == {
        'source_range', 'rendered_assertion_zh_hant_hk_fragments',
        'factual_localisations', 'quotation_source_keys', 'status', 'claim_role',
    }
    assert Draft202012Validator(V18_SCHEMA).is_valid(_wire())
    assert 'support_range' not in claim['properties']
    assert 'semantic_relation' not in claim['properties']


def test_v18_maps_one_range_to_claim_and_support_without_changing_names_or_bytes():
    view = build_source_view((BODY,), ('NEWS-1',))
    wire = _wire()
    raw = canonical_json_bytes(wire)
    package, receipt = materialise(
        raw, view, 'request-digest', provider_schema=V18_SCHEMA, v17_schema=V17_SCHEMA,
    )
    Draft202012Validator(SCHEMA).validate(package)
    claim = package['package']['governed_claims'][0]
    assert claim['claim'] == claim['supporting_excerpt'] == 'Alice Smith said the deadline changed.'
    assert claim['rendered_assertion_zh_hant_hk'] == 'Alice Smith 表示限期已更改。'
    assert claim['status'] == 'CONFIRMED_FACT'
    assert claim['semantic_relation'] == {
        'source_modality': 'ASSERTED', 'rendered_modality': 'ASSERTED',
        'source_polarity': 'AFFIRMED', 'rendered_polarity': 'AFFIRMED',
        'relation': 'SEMANTICALLY_EQUIVALENT',
    }
    assert receipt['raw_digest'] == digest_bytes(raw)
    assert receipt['provider_schema_digest'] == digest_bytes(canonical_json_bytes(V18_SCHEMA))
    assert receipt['manifest_digest'] == view.manifest_digest
    assert receipt['materialised_text'] == canonical_json_bytes(package).decode()
    assert receipt['receipt_digest'] == digest_canonical({
        key: value for key, value in receipt.items() if key != 'receipt_digest'
    })
    assert materialise(raw, view, 'request-digest',
                       provider_schema=V18_SCHEMA, v17_schema=V17_SCHEMA) == (package, receipt)


def test_v18_qualification_classifier_and_lookup_keys_reconstruct_exact_source():
    view = build_source_view((BODY,), ('NEWS-1',))
    wire = _wire()
    wire['package']['governed_claims'][0]['factual_localisations'] = [{
        'source_lookup_key': 'deadline', 'rendered_expression': '限期',
    }]
    wire['package']['governed_claims'][0]['quotation_source_keys'] = ['deadline']
    wire['package']['qualification_evidence'] = [{
        'test': 'OFFICIAL_ACTION_OR_DEADLINE', 'claim_index': 0,
        'test_evidence': {
            'action_class': 'OFFICIAL_DEADLINE', 'event_polarity': 'AFFIRMED',
            'action_relation': 'NEW_OR_CHANGED_OFFICIAL_ACTION',
            'material_relation_span_source_lookup_key': 'deadline changed',
            'reader_action_source_lookup_key': 'deadline',
        },
    }]
    package, _ = materialise(wire, view, 'request-digest',
                             provider_schema=V18_SCHEMA, v17_schema=V17_SCHEMA)
    claim = package['package']['governed_claims'][0]
    assert claim['localised_factual_expressions'] == [['deadline', '限期']]
    assert claim['quotations'] == ['deadline']
    qualification = package['package']['qualification_evidence'][0]
    assert qualification['test_evidence']['material_relation_span'] == 'deadline changed'
    assert qualification['test_evidence']['reader_action'] == 'deadline'
    assert qualification['policy_version'] == 'newsroom.evid-012.v7'


def test_disruption_duration_remains_classifier_not_source_lookup():
    variant = next(item for item in V18_SCHEMA['properties']['package']['properties'][
        'qualification_evidence']['items']['oneOf']
        if item['properties']['test']['const'] == 'ESSENTIAL_SERVICE_DISRUPTION')
    witnesses = variant['properties']['test_evidence']['properties']
    assert 'duration_minutes' in witnesses
    assert 'duration_minutes_source_lookup_key' not in witnesses
    assert 'affected_group_source_lookup_key' in witnesses
    view = build_source_view((BODY,), ('NEWS-1',))
    wire = _wire('S1L2')
    wire['package']['governed_claims'][0]['rendered_assertion_zh_hant_hk_fragments'] = [
        '下一行提及60分鐘。',
    ]
    wire['package']['qualification_evidence'] = [{
        'test': 'ESSENTIAL_SERVICE_DISRUPTION', 'claim_index': 0,
        'test_evidence': {
            'service_kind': 'TRANSPORT', 'event_polarity': 'AFFIRMED',
            'duration_relation': 'DISRUPTION_DURATION', 'duration_minutes': '60',
            'affected_group_source_lookup_key': 'Next line',
        },
    }]
    package, _ = materialise(wire, view, 'request-digest',
                             provider_schema=V18_SCHEMA, v17_schema=V17_SCHEMA)
    evidence = package['package']['qualification_evidence'][0]['test_evidence']
    assert evidence['duration_minutes'] == '60'
    assert evidence['affected_group'] == 'Next line'


@pytest.mark.parametrize('mutate', [
    lambda wire: wire['package']['governed_claims'][0].update(support_range={
        'first_span_id': 'S1L2', 'last_span_id': 'S1L2',
    }),
    lambda wire: wire['package']['governed_claims'][0].update(certainty='UNCERTAIN'),
    lambda wire: wire['package']['governed_claims'][0].update(
        rendered_assertion_zh_hant_hk_fragments=['缺少第二段'],
    ),
    lambda wire: wire['package']['governed_claims'][0]['factual_localisations'].append({
        'source_lookup_key': 'invented', 'rendered_expression': '虛構',
    }),
    lambda wire: wire['package']['governed_claims'][0]['quotation_source_keys'].append('invented'),
    lambda wire: wire['package']['governed_claims'][0].update(source_range={
        'first_span_id': 'S9L1', 'last_span_id': 'S9L1',
    }),
])
def test_v18_rejects_extra_choice_wrong_fragments_and_invented_source_keys(mutate):
    view = build_source_view((BODY,), ('NEWS-1',))
    wire = _wire()
    mutate(wire)
    with pytest.raises(SourceReferenceError):
        materialise(wire, view, 'request-digest',
                    provider_schema=V18_SCHEMA, v17_schema=V17_SCHEMA)


def test_v18_raw_duplicate_keys_and_caps_fail_closed():
    view = build_source_view((BODY,), ('NEWS-1',))
    with pytest.raises(SourceReferenceError):
        materialise(b'{"package":{},"package":{}}', view, 'request-digest',
                    provider_schema=V18_SCHEMA, v17_schema=V17_SCHEMA)
    with pytest.raises(SourceReferenceError):
        materialise(b'x' * (MAX_RESULT_BYTES + 1), view, 'request-digest',
                    provider_schema=V18_SCHEMA, v17_schema=V17_SCHEMA)
    with pytest.raises(SourceReferenceError):
        materialise(b'[' * 1200 + b'0' + b']' * 1200, view, 'request-digest',
                    provider_schema=V18_SCHEMA, v17_schema=V17_SCHEMA)
    wire = _wire()
    wire['package']['selection_rationale'] = 'x' * MAX_RESULT_BYTES
    with pytest.raises(SourceReferenceError):
        materialise(wire, view, 'request-digest',
                    provider_schema=V18_SCHEMA, v17_schema=V17_SCHEMA)


def test_v17_codec_schema_and_materialisation_remain_unchanged():
    view = build_source_view((BODY,), ('NEWS-1',))
    v17 = {'package': {
        **{key: _wire()['package'][key] for key in (
            'substantive_claim_indexes', 'selection_rationale', 'geography',
            'categories', 'explicit_exclusions', 'qualification_evidence',
        )},
        'governed_claims': [{
            'claim_range': {'first_span_id': 'S1L1', 'last_span_id': 'S1L1'},
            'support_range': {'first_span_id': 'S1L1', 'last_span_id': 'S1L1'},
            'rendered_fragments': ['', ' 表示限期已更改。'],
            'localised_factual_expressions': [], 'quotations': [],
            'status': 'CONFIRMED_FACT', 'claim_role': 'HEADLINE',
            'semantic_relation': {
                'source_modality': 'ASSERTED', 'rendered_modality': 'ASSERTED',
                'source_polarity': 'AFFIRMED', 'rendered_polarity': 'AFFIRMED',
                'relation': 'SEMANTICALLY_EQUIVALENT',
            },
            'certainty': 'CONFIRMED', 'originality_basis': 'FACTUAL_REWRITE_REQUIRED',
            'originality_policy_version': 'newsroom.cont-originality.v3',
            'admitted_use': 'PUBLICATION_EVIDENCE', 'policy_version': 'newsroom.governed-claim.v7',
        }],
    }}
    package, receipt = materialise_v17(v17, view, 'request-digest', provider_schema=V17_SCHEMA)
    assert package['package']['governed_claims'][0]['claim'] == 'Alice Smith said the deadline changed.'
    assert receipt['raw_digest'] == digest_bytes(canonical_json_bytes(v17))


def test_current_wire_derives_selected_news_once_from_claim_roles():
    from newsroom.control_plane.native_assessor import (
        VERSION, _materialise_reference_result,
    )

    wire = _wire()
    wire['package'].pop('substantive_claim_indexes')
    wire['package']['select_new_information'] = True
    wire['package']['governed_claims'].append({
        'source_range': {'first_span_id': 'S1L2', 'last_span_id': 'S1L2'},
        'rendered_assertion_zh_hant_hk_fragments': ['下一行提及60分鐘。'],
        'factual_localisations': [
            {'source_lookup_key': '60 minutes', 'rendered_expression': '60分鐘'},
        ],
        'quotation_source_keys': [], 'status': 'CONFIRMED_FACT',
        'claim_role': 'SUBSTANTIVE',
    })
    wire['package']['governed_claims'].append({
        'source_range': {'first_span_id': 'S1L3', 'last_span_id': 'S1L3'},
        'rendered_assertion_zh_hant_hk_fragments': ['原有指引維持不變。'],
        'factual_localisations': [], 'quotation_source_keys': [],
        'status': 'CONFIRMED_FACT', 'claim_role': 'CONTEXT',
    })
    raw = canonical_json_bytes(wire)
    view = build_source_view((BODY + 'Existing guidance remains.\n',), ('NEWS-1',))
    package, receipt = _materialise_reference_result(raw, view, 'request-digest', VERSION)

    assert package['package']['substantive_new_information'] == [
        'Alice Smith said the deadline changed.', 'Next line has 60 minutes.',
    ]
    assert [claim['claim_role'] for claim in package['package']['governed_claims']] == [
        'HEADLINE', 'SUBSTANTIVE', 'CONTEXT',
    ]
    assert receipt['raw_digest'] == digest_bytes(raw)


def _retained_headline_fixture():
    from pathlib import Path
    return json.loads((Path(__file__).parent / 'fixtures/native_assessor_v20_headline.json').read_text())


def _headline_wire(fixture, selected):
    wire = json.loads(fixture['raw_result_text'])
    wire['package'].pop('substantive_claim_indexes')
    wire['package']['select_new_information'] = selected
    return wire


@pytest.mark.parametrize('selected,expected', [(True, 'WRITE_READY'), (False, 'REJECT')])
def test_current_selection_reaches_existing_admission_without_duplicate_inventory(selected, expected):
    from dataclasses import replace
    from newsroom.control_plane.native_assessor import VERSION, _materialise_reference_result
    from newsroom.control_plane.admission import DeterministicWriteAdmission
    from newsroom.control_plane.evidence import EvidenceGateEvidence, EVIDENCE_GATE_POLICY_VERSION
    from newsroom.increment10.evidence import _package_from_value

    fixture = _retained_headline_fixture()
    wire = _headline_wire(fixture, selected)
    # Non-canonical whitespace must bind the actual supplied bytes, not an intermediate wire.
    raw = json.dumps(wire, ensure_ascii=False, indent=1).encode()
    view = build_source_view(tuple(fixture['source_passages']), tuple(fixture['source_ids']))
    materialised, receipt = _materialise_reference_result(raw, view, 'current-request', VERSION)
    from newsroom.control_plane.native_assessor import PROVIDER_SCHEMA_DIGEST
    assert receipt['raw_digest'] == digest_bytes(raw)
    assert receipt['provider_schema_digest'] == PROVIDER_SCHEMA_DIGEST
    assert receipt['request_identity'] == 'current-request'
    assert receipt['receipt_digest'] == digest_canonical({k:v for k,v in receipt.items() if k != 'receipt_digest'})
    previous = json.loads(fixture['materialisation_receipt']['materialised_text'])['package']['governed_claims']
    assert [{key:value for key,value in claim.items() if key != 'claim_id'}
            for claim in materialised['package']['governed_claims']] == [
        {key:value for key,value in claim.items() if key != 'claim_id'} for claim in previous]
    assert materialised['package']['qualification_evidence'][0]['governed_claim_id'] == materialised['package']['governed_claims'][0]['claim_id']
    retained = _package_from_value(fixture['retained_package'])
    policy = fixture['editorial_policy']
    claims = tuple(claim.claim_id for claim in retained.governed_claims)
    package = replace(retained,
        substantive_new_information=tuple(materialised['package']['substantive_new_information']),
        evidence_gate_results=tuple(map(tuple,policy['evidence_gate_results'])),
        evidence_gate_evidence=tuple(EvidenceGateEvidence(gate,result,claims,EVIDENCE_GATE_POLICY_VERSION)
                                    for gate,result in policy['evidence_gate_results']),
        freshness_result='PASS',integrity_result='PASS')
    admission = DeterministicWriteAdmission()
    def decide(value):
        return admission.decide_candidate_identity(candidate_id=value.candidate_id,
            hypothesis_id=value.hypothesis_id,package=value,decided_at=policy['evaluated_at'])
    result = decide(package)
    assert result.decision == expected
    if selected:
        assert package.substantive_new_information == (
            'Official status changed: 香港天文台 issued the 雷暴警告.',
            'The 雷暴警告 warning record was issued on 2 October 2026 at 15:55 (香港時間).',
        )
        assert decide(replace(package,qualification_evidence=())).stable_reason_codes == ('UNQUALIFIED_HEADLINE_CLAIM',)
        missing_body = replace(package,governed_claims=tuple(c for c in package.governed_claims if c.claim_role=='HEADLINE'),
            substantive_new_information=(package.substantive_new_information[0],),
            evidence_gate_evidence=tuple(replace(g,governed_claim_ids=(claims[0],))for g in package.evidence_gate_evidence))
        assert 'INVALID_SUBSTANTIVE_CLAIM_INVENTORY' in decide(missing_body).stable_reason_codes
    else:
        assert package.substantive_new_information == ()
        assert package.qualification_evidence
        assert result.stable_reason_codes == ('NO_SUBSTANTIVE_NEW_INFORMATION',)


def test_original_v20_result_and_receipt_replay_without_selection_reinterpretation():
    from newsroom.control_plane.native_assessor import _materialise_reference_result
    fixture = _retained_headline_fixture()
    view = build_source_view(tuple(fixture['source_passages']), tuple(fixture['source_ids']))
    materialised, receipt = _materialise_reference_result(fixture['raw_result_text'],view,
        fixture['materialisation_receipt']['request_identity'],'newsroom.native-evidence-assessor.v20')
    assert digest_bytes(fixture['raw_result_text'].encode()) == 'sha256:cd98d36fedbfa2fdacb8e5ce0d3da749ca2c2d4c84f45bab7f60ba7494e5a48c'
    assert receipt == fixture['materialisation_receipt']
    assert materialised['package']['substantive_new_information'] == [
        'The 雷暴警告 warning record was issued on 2 October 2026 at 15:55 (香港時間).',
    ]


@pytest.mark.parametrize('mutation', ['empty', 'integer', 'old_indexes', 'bad_source'])
def test_current_selection_preserves_no_news_and_closed_source_contract(mutation):
    from newsroom.control_plane.native_assessor import VERSION, _materialise_reference_result
    fixture = _retained_headline_fixture()
    wire = _headline_wire(fixture, False)
    if mutation == 'empty':
        wire['package']['governed_claims'] = []
        wire['package']['qualification_evidence'] = []
    elif mutation == 'integer':
        wire['package']['select_new_information'] = 1
    elif mutation == 'old_indexes':
        wire['package']['substantive_claim_indexes'] = [1]
    else:
        wire['package']['governed_claims'][0]['source_range']['first_span_id'] = 'S9L1'
    view = build_source_view(tuple(fixture['source_passages']),tuple(fixture['source_ids']))
    if mutation == 'empty':
        materialised,_ = _materialise_reference_result(wire,view,'current-request',VERSION)
        assert materialised['package']['substantive_new_information'] == []
        assert materialised['package']['governed_claims'] == []
    else:
        with pytest.raises(SourceReferenceError):
            _materialise_reference_result(wire,view,'current-request',VERSION)
