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


@pytest.mark.parametrize('rendered,expected', [('Dan Jarvis表示將堵塞漏洞。', True),
                                            ('Home Office表示將堵塞漏洞。', False)])
def test_publisher_is_not_an_explicit_source_speaker(rendered, expected):
    from newsroom.control_plane.native_story_entities import declared_source_speakers_are_retained
    claim = N(claim_id='context', claim='Security Minister, Dan Jarvis said: We will close the gaps.',
              named_entity_evidence=(('Dan Jarvis', 'PERSON', 'record'),), attribution='Home Office')
    link = N(governed_claim_id='context', rendered_assertion=rendered)
    assert declared_source_speakers_are_retained((claim,), (link,)) is expected
