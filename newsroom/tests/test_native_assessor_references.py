"""Static, lossless assessor source references and exact materialisation."""

from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane.evidence import EVID_012_POLICY_VERSION
from newsroom.control_plane.native_assessor import SCHEMA
from newsroom.control_plane.native_assessor_references import (
    MAX_RESULT_BYTES, SourceReferenceError, build_source_view, make_provider_schema,
    materialise,
)


PREFIX = (
    "Senior officials hospitality\n\n"
    "Attachment: https://assets.publishing.service.gov.uk/media/fixture/officials.csv\n"
    "Published CSV cells: Row and column identify each literal text cell. Whitespace, empty fields, quoted newlines and formula-like text are preserved; nothing is executed. No header or numeric types are inferred.\n"
    'Sheet "CSV"\n'
    'Row 1: A="Senior Official\'s Name "; B="Date"; C="Individual or organisation that provided hospitality "\n'
)
ROW = 'Row 2: A="Marianthi Leontaridi"; B="2026-01-14"; C="Boston Consulting Group"'


def _claim(first: str, *, fragments=None):
    return {
        'claim_role': 'HEADLINE',
        'claim_range': {'first_span_id': first, 'last_span_id': first},
        'support_range': {'first_span_id': first, 'last_span_id': first},
        'rendered_fragments': fragments if fragments is not None else ['', ' 與 ', ' 用餐。'],
        'status': 'CONFIRMED_FACT',
        'semantic_relation': {
            'source_modality': 'ASSERTED', 'rendered_modality': 'ASSERTED',
            'source_polarity': 'AFFIRMED', 'rendered_polarity': 'AFFIRMED',
            'relation': 'SEMANTICALLY_EQUIVALENT',
        },
        'localised_factual_expressions': [['2026-01-14', '2026年1月14日']],
        'quotations': ['2026-01-14'],
        'certainty': 'CONFIRMED', 'originality_basis': 'FACTUAL_REWRITE_REQUIRED',
        'originality_policy_version': 'newsroom.cont-originality.v3',
        'admitted_use': 'PUBLICATION_EVIDENCE',
        'policy_version': 'newsroom.governed-claim.v7',
    }


def _wire(first: str, *, claims=None):
    return {'package': {
        'substantive_claim_indexes': [0] if claims is None else [],
        'governed_claims': [_claim(first)] if claims is None else claims,
        'qualification_evidence': [], 'selection_rationale': 'Exact row',
        'geography': [], 'categories': [], 'explicit_exclusions': [],
    }}


def test_view_is_lossless_across_utf8_lines_and_preserves_whole_csv_row():
    prose = '香港第一句。\nSecond line.\r\n'
    view = build_source_view((prose, PREFIX + ROW), ('HK-01', 'UK-02'))
    assert tuple(view.source_ids) == ('HK-01', 'UK-02')
    for index, body in enumerate(view.passages):
        assert ''.join(item.text for item in view.segments if item.passage_index == index) == body
        assert view.body_digests[index] == digest_bytes(body.encode())
    assert view.segments[-1].text == ROW
    assert view.request_segments[-1]['text'] == ROW
    assert view.resolve_range({'first_span_id': view.segments[-1].span_id,
                               'last_span_id': view.segments[-1].span_id}) == (ROW, 1, 'UK-02')
    assert view.manifest_digest == build_source_view((prose, PREFIX + ROW), ('HK-01', 'UK-02')).manifest_digest
    assert ('Marianthi Leontaridi', 'PERSON') in view.source_entities[1]
    assert ('Boston Consulting Group', 'ORGANISATION') in view.source_entities[1]


def test_segment_entity_order_is_local_even_when_source_first_occurrence_differs():
    body = 'Alice Smith said yes; Bob Jones said no.\nBob Jones said yes; Alice Smith said no.\n'
    view = build_source_view((body,), ('NEWS-1',))
    assert [name for name, _ in view.source_entities[0]] == ['Alice Smith', 'Bob Jones']
    assert [name for name, _ in view.segments[1].entities] == ['Bob Jones', 'Alice Smith']


def test_advertised_entities_match_materialiser_dedup_and_boundaries():
    body = 'Alice Smith said yes; Alice Smith said no.\nThe Department for Education said yes.\n'
    view = build_source_view((body,), ('NEWS-1',))
    assert view.segments[0].entities == (('Alice Smith', 'PERSON'),)
    assert view.segments[1].entities == (('Department for Education', 'ORGANISATION'),)
    assert view.source_entities[0] == (
        ('Alice Smith', 'PERSON'), ('Department for Education', 'ORGANISATION'),
    )


def test_schema_is_static_closed_and_keeps_existing_classifier_contract():
    schema = make_provider_schema(SCHEMA)
    Draft202012Validator.check_schema(schema)
    assert schema == make_provider_schema(SCHEMA)
    assert schema['additionalProperties'] is False
    claim = schema['properties']['package']['properties']['governed_claims']['items']
    assert claim['additionalProperties'] is False
    assert {'claim_range', 'support_range', 'rendered_fragments'} <= set(claim['required'])
    assert 'claim' not in claim['properties'] and 'source_ids' not in claim['properties']
    assert 'status' in claim['properties'] and 'semantic_relation' in claim['properties']
    assert schema['properties']['package']['properties']['qualification_evidence']['items']['oneOf']
    assert Draft202012Validator(schema).is_valid(_wire('S1L6'))


def test_materialisation_reconstructs_exact_row_names_and_replays_identically():
    view = build_source_view((PREFIX + ROW,), ('UK-02',))
    first = view.segments[-1].span_id
    wire = _wire(first)
    raw = canonical_json_bytes(wire)
    package, receipt = materialise(raw, view, 'request-digest')
    Draft202012Validator(SCHEMA).validate(package)
    claim = package['package']['governed_claims'][0]
    assert claim['claim'] == claim['supporting_excerpt'] == ROW
    assert claim['source_ids'] == ['UK-02'] and claim['passage_index'] == 0
    assert claim['rendered_assertion_zh_hant_hk'] == 'Marianthi Leontaridi 與 Boston Consulting Group 用餐。'
    assert package['package']['substantive_new_information'] == [ROW]
    assert claim['quotations'] == ['2026-01-14']
    assert claim['localised_factual_expressions'] == [['2026-01-14', '2026年1月14日']]
    assert receipt['raw_digest'] == digest_bytes(raw)
    assert receipt['manifest_digest'] == view.manifest_digest
    assert receipt['body_digests'] == list(view.body_digests)
    assert receipt['materialised_text'] == canonical_json_bytes(package).decode()
    assert receipt['claim_entity_order'] == [[
        ['Marianthi Leontaridi', 'PERSON'], ['Boston Consulting Group', 'ORGANISATION'],
    ]]
    assert tuple(map(tuple, receipt['claim_entity_order'][0])) == view.segments[-1].entities
    assert materialise(raw, view, 'request-digest') == (package, receipt)


def test_repeated_entity_is_inserted_once_in_first_occurrence_order():
    body = 'Alice Smith said yes. Alice Smith said no.\n'
    view = build_source_view((body,), ('NEWS-1',))
    wire = _wire('S1L1')
    wire['package']['governed_claims'][0]['rendered_fragments'] = ['', ' 表示。']
    wire['package']['governed_claims'][0]['localised_factual_expressions'] = []
    wire['package']['governed_claims'][0]['quotations'] = []
    package, receipt = materialise(wire, view, 'request-digest')
    assert receipt['claim_entity_order'] == [[['Alice Smith', 'PERSON']]]
    assert package['package']['governed_claims'][0]['rendered_assertion_zh_hant_hk'] == 'Alice Smith 表示。'


def test_empty_no_news_materialises_without_entities():
    view = build_source_view(('No supported new information.\n',), ('S-1',))
    wire = _wire(view.segments[0].span_id, claims=[])
    wire['package']['selection_rationale'] = 'No supported new information.'
    package, _ = materialise(wire, view, 'request-digest')
    assert package['package']['governed_claims'] == []
    assert package['package']['substantive_new_information'] == []


@pytest.mark.parametrize('change', [
    lambda wire, view: wire['package']['governed_claims'][0]['claim_range'].update(first_span_id='S9L9'),
    lambda wire, view: wire['package']['governed_claims'][0]['claim_range'].update(first_span_id='S1L999'),
    lambda wire, view: wire['package']['governed_claims'][0].update(rendered_fragments=['少咗片段']),
    lambda wire, view: wire['package']['governed_claims'][0].update(rendered_fragments=['Marianthi Leontaridi', ' 與 ', ' 用餐。']),
    lambda wire, view: wire['package']['governed_claims'][0].update(quotations=['wrong quote']),
    lambda wire, view: wire['package']['governed_claims'][0].update(localised_factual_expressions=[['wrong date', '日期']]),
    lambda wire, view: wire['package']['governed_claims'].append(wire['package']['governed_claims'][0].copy()),
    lambda wire, view: wire['package']['substantive_claim_indexes'].__setitem__(0, 99),
])
def test_invalid_references_fragments_keys_or_duplicates_fail_closed(change):
    view = build_source_view((PREFIX + ROW,), ('UK-02',))
    wire = _wire(view.segments[-1].span_id)
    change(wire, view)
    with pytest.raises(SourceReferenceError):
        materialise(wire, view, 'request-digest')


def test_cross_source_reversed_and_uncontained_ranges_fail_closed():
    view = build_source_view(('First sentence.\nSecond sentence.\n', 'Other sentence.\n'), ('A', 'B'))
    range_one = {'first_span_id': 'S1L1', 'last_span_id': 'S1L2'}
    assert view.resolve_range(range_one)[0] == 'First sentence.\nSecond sentence.\n'
    for reference in (
        {'first_span_id': 'S1L2', 'last_span_id': 'S1L1'},
        {'first_span_id': 'S1L2', 'last_span_id': 'S2L1'},
    ):
        with pytest.raises(SourceReferenceError):
            view.resolve_range(reference)
    wire = _wire('S1L2')
    wire['package']['governed_claims'][0]['support_range'] = {
        'first_span_id': 'S1L1', 'last_span_id': 'S1L1',
    }
    with pytest.raises(SourceReferenceError):
        materialise(wire, view, 'request-digest')
    repeated = build_source_view(('Same line.\nSame line.\n',), ('A',))
    wire = _wire('S1L2')
    wire['package']['governed_claims'][0]['support_range'] = {
        'first_span_id': 'S1L1', 'last_span_id': 'S1L1',
    }
    with pytest.raises(SourceReferenceError):
        materialise(wire, repeated, 'request-digest')


def test_qualification_witness_reconstructs_source_and_tamper_fails():
    view = build_source_view((PREFIX + ROW,), ('UK-02',))
    wire = _wire(view.segments[-1].span_id)
    wire['package']['qualification_evidence'] = [{
        'test': 'OFFICIAL_ACTION_OR_DEADLINE', 'claim_index': 0,
        'policy_version': EVID_012_POLICY_VERSION,
        'test_evidence': {'action_class': 'OFFICIAL_DEADLINE', 'event_polarity': 'AFFIRMED',
                          'action_relation': 'NEW_OR_CHANGED_OFFICIAL_ACTION',
                          'material_relation_span': ROW, 'reader_action': '2026-01-14'},
    }]
    package, _ = materialise(wire, view, 'request-digest')
    assert package['package']['qualification_evidence'][0]['governed_claim_id'] == package['package']['governed_claims'][0]['claim_id']
    assert package['package']['qualification_evidence'][0]['test_evidence']['reader_action'] == '2026-01-14'
    wire['package']['qualification_evidence'][0]['test_evidence']['reader_action'] = '2026-01-15'
    with pytest.raises(SourceReferenceError):
        materialise(wire, view, 'request-digest')


@pytest.mark.parametrize(('test', 'evidence'), [
    ('LAW_RIGHT_STATUS_POLICY', {
        'change_kind': 'LAW', 'event_polarity': 'AFFIRMED',
        'change_relation': 'NEW_OR_CHANGED_STATE', 'material_relation_span': ROW,
        'new_state': '2026-01-14',
    }),
    ('ESSENTIAL_SERVICE_DISRUPTION', {
        'service_kind': 'SCHOOL', 'event_polarity': 'AFFIRMED',
        'duration_relation': 'DISRUPTION_DURATION', 'duration_minutes': '60',
        'affected_group': 'Boston Consulting Group',
    }),
    ('SAFETY_OR_PUBLIC_HEALTH', {
        'effect_class': 'INJURY_RISK', 'event_polarity': 'AFFIRMED',
        'effect_relation': 'MATERIAL_EFFECT', 'material_relation_span': ROW,
        'affected_group': 'Boston Consulting Group',
    }),
    ('HOUSEHOLD_PRACTICAL_EFFECT', {
        'domain': 'MONEY', 'event_polarity': 'AFFIRMED',
        'effect_relation': 'MATERIAL_PRACTICAL_EFFECT', 'material_relation_span': ROW,
        'practical_effect': '2026-01-14',
    }),
    ('OFFICIAL_ACTION_OR_DEADLINE', {
        'action_class': 'PROCESS', 'event_polarity': 'AFFIRMED',
        'action_relation': 'NEW_OR_CHANGED_OFFICIAL_ACTION', 'material_relation_span': ROW,
        'reader_action': '2026-01-14',
    }),
    ('EXCEPTIONAL_PUBLIC_IMPORTANCE', {
        'importance_class': 'HONG_KONG_WIDE', 'event_polarity': 'AFFIRMED',
        'importance_relation': 'CURRENT_EXCEPTIONAL_IMPORTANCE', 'material_relation_span': ROW,
        'affected_group': 'Boston Consulting Group',
    }),
])
def test_qualification_classifier_keys_are_not_source_lookup_keys(test, evidence):
    view = build_source_view((PREFIX + ROW,), ('UK-02',))
    wire = _wire(view.segments[-1].span_id)
    wire['package']['qualification_evidence'] = [{
        'test': test, 'claim_index': 0, 'policy_version': EVID_012_POLICY_VERSION,
        'test_evidence': evidence,
    }]
    package, _ = materialise(wire, view, 'request-digest',
                             provider_schema=make_provider_schema(SCHEMA))
    assert package['package']['qualification_evidence'][0]['test_evidence'] == evidence
    invalid = dict(evidence)
    classifier = next(key for key in invalid if key not in {
        'material_relation_span', 'new_state', 'affected_group',
        'practical_effect', 'reader_action',
    })
    invalid[classifier] = 'INVALID_CLASSIFIER'
    wire['package']['qualification_evidence'][0]['test_evidence'] = invalid
    with pytest.raises(SourceReferenceError):
        materialise(wire, view, 'request-digest',
                    provider_schema=make_provider_schema(SCHEMA))


@pytest.mark.parametrize('mutation', [
    lambda wire: wire['package'].update(unexpected=True),
    lambda wire: wire['package'].update(categories='not a list'),
    lambda wire: wire['package']['governed_claims'][0].update(extra='ignored'),
    lambda wire: wire['package']['governed_claims'][0].update(status=3),
    lambda wire: wire['package']['qualification_evidence'].append({
        'test': 'OFFICIAL_ACTION_OR_DEADLINE', 'claim_index': 0,
        'policy_version': EVID_012_POLICY_VERSION, 'test_evidence': {}, 'extra': 'ignored',
    }),
])
def test_closed_wire_rejects_extra_fields_and_bad_types(mutation):
    view = build_source_view((PREFIX + ROW,), ('UK-02',))
    wire = _wire(view.segments[-1].span_id)
    mutation(wire)
    with pytest.raises(SourceReferenceError):
        materialise(wire, view, 'request-digest', provider_schema=make_provider_schema(SCHEMA))


def test_source_tamper_and_raw_materialisation_caps():
    view = build_source_view((PREFIX + ROW,), ('UK-02',))
    wire = _wire(view.segments[-1].span_id)
    tampered = replace(view, passages=(PREFIX + ROW + ' changed',))
    with pytest.raises(SourceReferenceError):
        materialise(wire, tampered, 'request-digest')
    with pytest.raises(SourceReferenceError):
        materialise(b'x' * (MAX_RESULT_BYTES + 1), view, 'request-digest')
    wire['package']['selection_rationale'] = 'x' * MAX_RESULT_BYTES
    with pytest.raises(SourceReferenceError):
        materialise(wire, view, 'request-digest')
