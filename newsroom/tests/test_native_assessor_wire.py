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
