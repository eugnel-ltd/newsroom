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
