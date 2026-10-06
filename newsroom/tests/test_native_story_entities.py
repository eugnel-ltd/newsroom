"""A display name is not permission to invent a place or a policy actor."""
from types import SimpleNamespace as N
import pytest
from newsroom.control_plane.native_story_entities import story_entity_names_are_bound


def _claim(name, kind='PLACE'):
    return N(named_entities=(name,), named_entity_evidence=((name, kind, 'record'),))


@pytest.mark.parametrize('source,rendered', [('UK', '英國'), ('英國', 'UK'),
                                           ('Hong Kong', '香港'), ('香港', 'Hong Kong')])
def test_admitted_typed_place_retains_identity_in_its_display_name(source, rendered):
    assert story_entity_names_are_bound(rendered, (_claim(source),))


@pytest.mark.parametrize('claims', [(), (_claim('Home Office', 'ORGANISATION'),),
                                    (_claim('UK', 'ORGANISATION'),), (_claim('香港'),)])
def test_publisher_or_unrelated_identity_does_not_authorise_country(claims):
    assert not story_entity_names_are_bound('英國', claims)


def test_localisation_never_admits_a_new_person_or_other_place():
    assert not story_entity_names_are_bound('John Smith said 英國 and Hong Kong', (_claim('UK'),))


@pytest.mark.parametrize('source', ['Security Minister, Dan Jarvis said: We will close the gaps.',
                                    'Dan Jarvis, Security Minister said: We will close the gaps.'])
@pytest.mark.parametrize('rendered,expected', [('Dan Jarvis表示將堵塞漏洞。', True),
                                            ('Home Office表示將堵塞漏洞。', False)])
def test_publisher_is_not_an_explicit_source_speaker(source, rendered, expected):
    from newsroom.control_plane.native_story_entities import declared_source_speakers_are_retained
    claim = N(claim_id='context', claim=source,
              named_entity_evidence=(('Dan Jarvis', 'PERSON', 'record'),), attribution='Home Office')
    link = N(governed_claim_id='context', rendered_assertion=rendered)
    assert declared_source_speakers_are_retained((claim,), (link,)) is expected


@pytest.mark.parametrize('name', ['Home Office', '英國內政部'])
def test_exact_governed_publisher_allows_only_reporting_display(name):
    claim = N(named_entities=(), named_entity_evidence=(), attribution='Home Office')
    assert story_entity_names_are_bound(name+'表示，有關改動須經諮詢。', (claim,))


@pytest.mark.parametrize('publisher', ['UKVI', 'Home Office; UKVI', 'Other authority',
    'home office', 'Home Office ', None])
def test_source_anchor_or_other_publisher_does_not_select_reporting_name(publisher):
    claim = N(named_entities=('UK',), named_entity_evidence=(('UK', 'PLACE', 'record'),),
        source_ids=('UK-01',), attribution=publisher)
    assert not story_entity_names_are_bound('英國內政部表示，有關改動須經諮詢。', (claim,))


@pytest.mark.parametrize('text', [
    '英國內政部推出新政策。',
    '英國內政部表示新政策已公布。英國內政部制定新政策。',
    'Home Office 表示有關改動須經諮詢。Home Office introduced a policy.',
    '其他機構引述英國內政部表示有關改動須經諮詢。',
    'Dan Jarvis表示：「英國內政部表示有關改動須經諮詢。」',
    'Dan Jarvis表示：「保障公眾。英國內政部表示有關改動須經諮詢。」',
    'Dan Jarvis表示：「保障公眾。英國內政部表示有關改動須經諮詢。',
    '英國內政部表示，有關改動須經諮詢。Housing Authority表示支持。',
])
def test_reporting_publisher_display_never_whitelists_actors_or_quoted_names(text):
    claim = N(named_entities=('Dan Jarvis',), named_entity_evidence=(('Dan Jarvis', 'PERSON', 'record'),),
        attribution='Home Office')
    assert not story_entity_names_are_bound(text, (claim,))


def test_only_actual_reporting_names_are_exposed_to_other_consumers():
    from newsroom.control_plane.native_story_entities import source_publisher_reporting_names
    claim = N(named_entities=(), named_entity_evidence=(), attribution='Home Office')
    assert source_publisher_reporting_names('英國內政部表示，有關改動須經諮詢。', (claim,)) == {'英國內政部'}
    assert source_publisher_reporting_names('政府表示，有關改動須經諮詢。', (claim,)) == set()
    assert source_publisher_reporting_names('英國內政部表示有關改動。英國內政部制定政策。', (claim,)) == set()


def test_exact_typed_organisation_actor_keeps_the_existing_literal_contract():
    assert story_entity_names_are_bound('Home Office制定政策。', (_claim('Home Office', 'ORGANISATION'),))
    assert not story_entity_names_are_bound('英國內政部制定政策。', (_claim('Home Office', 'ORGANISATION'),))


@pytest.mark.parametrize('publisher_name', ['Home Office', '英國內政部'])
def test_reporting_display_does_not_replace_declared_source_speaker(publisher_name):
    from newsroom.control_plane.native_story_entities import declared_source_speakers_are_retained
    claim = N(claim_id='context', claim='Security Minister, Dan Jarvis said: We will close the gaps.',
        named_entities=('Dan Jarvis',), named_entity_evidence=(('Dan Jarvis', 'PERSON', 'record'),),
        attribution='Home Office')
    replacement = N(governed_claim_id='context', rendered_assertion=publisher_name+'表示將堵塞漏洞。')
    assert not declared_source_speakers_are_retained((claim,), (replacement,))
    retained = N(governed_claim_id='context', rendered_assertion='Dan Jarvis表示將堵塞漏洞。')
    assert declared_source_speakers_are_retained((claim,), (retained,))


def test_admitted_country_does_not_hide_an_unbound_publisher_actor():
    claim = N(named_entities=('UK',), named_entity_evidence=(('UK', 'PLACE', 'record'),),
        attribution='Home Office')
    assert not story_entity_names_are_bound('英國內政部制定政策。', (claim,))
    assert story_entity_names_are_bound('英國內政部表示，有關改動須經諮詢。', (claim,))


@pytest.mark.parametrize('text,expected', [
    ('英國內政部表示，有關改動須經諮詢。', {'英國內政部'}),
    ('Home Office表示，有關改動須經諮詢。', {'Home Office'}),
    ('英國內政部稱霸有關政策。', set()),
    ('英國內政部指出有關政策。', set()),
    ("'申述。英國內政部表示，有關改動須經諮詢。'", set()),
    ("'The government's statement。\n英國內政部表示，有關改動須經諮詢。'", set()),
    ("The government's briefing。\n英國內政部表示，有關改動須經諮詢。", {'英國內政部'}),
    ("Officials' briefing。\n英國內政部表示，有關改動須經諮詢。", {'英國內政部'}),
])
def test_reporting_verb_boundary_and_ascii_quote_roles(text, expected):
    from newsroom.control_plane.native_story_entities import source_publisher_reporting_names
    claim = N(named_entities=(), named_entity_evidence=(), attribution='Home Office')
    assert source_publisher_reporting_names(text, (claim,)) == expected
