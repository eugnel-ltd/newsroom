"""Consumer Source-term inputs preserve original bytes and calendar precision."""
import pytest

from newsroom.authority.canonical import digest_bytes
from newsroom.control_plane.native_source_term_bindings import source_term_bindings, derive_relative_year


def _scope(body, selected=None):
    selected = body if selected is None else selected
    start = body.index(selected)
    return dict(body_digest=digest_bytes(body.encode()), start_byte=len(body[:start].encode()),
                end_byte=len(body[:start + len(selected)].encode()))


def test_current_source_literals_are_not_actor_or_longform_certificates():
    body = ('Support is provided by Ambition Institute. Training is delivered by charity '
            'Dingley’s Promise for SEND. Children with SEND will benefit.')
    records = source_term_bindings(body, body, **_scope(body))
    assert [(literal, kind) for literal, kind, _, _ in records] == [
        ('Ambition Institute', 'SOURCE_LITERAL_ORGANISATION_NAME'),
        ('Dingley’s Promise', 'SOURCE_LITERAL_ORGANISATION_NAME'),
        ('SEND', 'SOURCE_LITERAL_ACRONYM'), ('SEND', 'SOURCE_LITERAL_ACRONYM')]
    for literal, _, start, end in records:
        assert body.encode()[start:end].decode() == literal
    assert records == source_term_bindings(body, body, **_scope(body))


def test_existing_extractor_is_reused_without_a_new_name_registry():
    body = 'The Home Office provides updates.'
    assert source_term_bindings(body, body, **_scope(body))[0][:2] == ('Home Office', 'SOURCE_LITERAL_NAME')


def test_v2_source_literals_preserve_contacts_and_opaque_labels_without_age_inference():
    from newsroom.control_plane.native_source_term_bindings import VERSION_V2
    body = 'For post-16 funding contact team.help@education.gov.uk. DfE publishes this form.'
    assert source_term_bindings(body, body, **_scope(body)) == ()
    records = source_term_bindings(body, body, **_scope(body), version=VERSION_V2)
    assert [r[:2] for r in records] == [
        ('post-16', 'SOURCE_LITERAL_LABEL'), ('team.help@education.gov.uk', 'SOURCE_LITERAL_EMAIL'),
        ('DfE', 'SOURCE_LITERAL_LABEL')]
    assert all(body.encode()[start:end].decode() == literal for literal, _, start, end in records)
    with pytest.raises(ValueError):
        source_term_bindings(body, body.replace('help', 'HELP'), **_scope(body), version=VERSION_V2)
    ordinary = 'NO funding is ON hold. Information is provided.'
    assert source_term_bindings(ordinary, ordinary, **_scope(ordinary), version=VERSION_V2) == ()


@pytest.mark.parametrize('damage', [None, 'range', 'source', 'forged-literal'])
def test_source_literal_projection_uses_bound_range_and_leaves_original_slots_immutable(damage):
    from copy import deepcopy
    from newsroom.control_plane.native_assessor_spans import build_lossless_source_view
    from newsroom.control_plane.native_assessor import _reference_binding
    from newsroom.control_plane.native_assessor_judgments import source_rendering_projection, original_rendering_slots
    body = 'Contact team.help@education.gov.uk for post-16 funding.'
    view = build_lossless_source_view((body,), ('UK-05',))
    original = {'source_binding': {'current_scope': {'sources': [{'source_id': 'UK-05', 'body': body}]},
                                  'source_reference_binding': _reference_binding(view)},
                'claims': {'S1L1': {'source_id': 'UK-05', 'text': body, 'entities': [], 'rendering_fragment_count': 1,
                                   'source_range': {'first_span_id': 'S1L1', 'last_span_id': 'S1L1'}}}}
    if damage == 'range': original['claims']['S1L1']['source_range']['last_span_id'] = 'S1L2'
    if damage == 'source': original['claims']['S1L1']['source_id'] = 'UK-01'
    if damage == 'forged-literal': original['claims']['S1L1'].update(entities=[['funding', 'SOURCE_LITERAL']], rendering_fragment_count=2)
    before = deepcopy(original)
    if damage:
        with pytest.raises(ValueError): source_rendering_projection(original)
    else:
        projected = source_rendering_projection(original)
        assert projected['claims']['S1L1']['entities'] == [
            ['team.help@education.gov.uk', 'SOURCE_LITERAL'], ['post-16', 'SOURCE_LITERAL']]
        assert projected['claims']['S1L1']['rendering_fragment_count'] == 3
        restored = original_rendering_slots(original['claims']['S1L1'], projected['claims']['S1L1'],
            {'rendered_assertion_zh_hant_hk_fragments': ['請聯絡', '處理', '資助。']})
        assert restored['rendered_assertion_zh_hant_hk_fragments'] == ['請聯絡team.help@education.gov.uk處理post-16資助。']
    assert original == before


@pytest.mark.parametrize('body', [
    'NATIONALTITLE\nRANDOM', 'RANDOM is an unsupported isolated label.',
    'The Ambition Institute slogan was discussed.', 'The Government’s Policy title was copied.',
])
def test_unproved_capitalisation_is_not_literal_name_authority(body):
    assert source_term_bindings(body, body, **_scope(body)) == ()


@pytest.mark.parametrize('change', ['body', 'selected', 'case', 'range', 'boolean'])
def test_changed_body_case_and_range_never_reuse_term_positions(change):
    body = '計劃 Support is provided by Ambition Institute.'
    selected = 'Support is provided by Ambition Institute.'
    scope = _scope(body, selected)
    if change == 'body': body += ' changed'
    elif change == 'selected': selected = '"' + selected + '"'
    elif change == 'case': selected = selected.replace('Institute', 'institute')
    elif change == 'range': scope['start_byte'] = 1
    else: scope['start_byte'] = True
    with pytest.raises(ValueError): source_term_bindings(body, selected, **scope)


def test_relative_year_uses_publisher_not_current_or_retrieval_clock():
    body = 'Earlier material. Funding starts early next year for schools.'
    selected = 'Funding starts early next year for schools.'
    result = derive_relative_year(body, selected, **_scope(body, selected),
        publication_time='2026-09-25T09:33:26Z', source_updated_time='2026-10-01T00:00:00Z')
    assert result[0:2] == ('next year', '2027年') and result[-1] == 2026
    assert body.encode()[result[2]:result[3]] == b'next year'


@pytest.mark.parametrize('selected', [
    'If funding is agreed, training starts next year.',
    'The minister could fund training next year.',
    'In 1880 the minister planned training next year.',
    'Training starts next academic year.', 'Training is planned for the financial year next year.',
    'Training starts within the next year.', 'Training starts over next year.',
    'Training starts next year and grows next year.',
    'The minister said "training starts next year".',
])
def test_ambiguous_conditional_historical_or_quoted_year_is_denied(selected):
    with pytest.raises(ValueError):
        derive_relative_year(selected, selected, **_scope(selected),
            publication_time='2026-09-25T09:33:26Z', source_updated_time='2026-09-25T09:33:26Z')


@pytest.mark.parametrize(('published', 'updated'), [
    ('2026-09-25', '2026-09-25'), ('2026-09-25T09:33:26Z', '2027-01-01T00:00:00Z'),
    ('2026-01-01T00:30:00+01:00', '2026-01-01T00:30:00+01:00'),
    ('invalid', '2026-09-25T09:33:26Z'), (None, '2026-09-25T09:33:26Z'),
])
def test_relative_year_requires_unambiguous_aware_publisher_chronology(published, updated):
    body = 'Training starts next year.'
    with pytest.raises(ValueError):
        derive_relative_year(body, body, **_scope(body), publication_time=published, source_updated_time=updated)
