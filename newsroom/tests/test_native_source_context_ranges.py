"""Speaker parents remain within exact, lossless Source context ranges."""
from dataclasses import replace

import pytest
from newsroom.control_plane.native_assessor_references import SourceReferenceError, _claim_entities

from newsroom.control_plane.native_assessor_spans import build_lossless_source_view
from newsroom.control_plane.native_source_context_ranges import context_candidates


@pytest.mark.parametrize('title', ('Security Minister, Dan Jarvis', 'Security Minister Dan Jarvis'))
def test_explicit_speaker_closes_three_first_person_spans_and_stops_at_third_person(title):
    body = ''.join(f'Earlier source paragraph {index}.\n' for index in range(1, 10)) + (
        f'{title} said: Protecting the public is our first duty.\n'
        'While these substances have legitimate uses, we must stop their misuse.\n'
        'We will close gaps in the law.\n'
        'As part of the move, the government will consult the public.'
    )
    view = build_lossless_source_view((body,), ('UK-01',))
    candidates = context_candidates(view)
    parent = candidates['S1L10']
    assert parent['source_range'] == {'first_span_id': 'S1L10', 'last_span_id': 'S1L12'}
    assert parent['text'] == view.resolve_range(parent['source_range'])[0]
    assert parent['entities'] == [list(entity) for entity in _claim_entities(
        parent['text'], body, policy_version=view.entity_policy_version,
    )]
    assert parent['rendering_fragment_count'] == 2
    assert parent['speaker_parent_hold'] is False
    assert 'S1L11' not in candidates and 'S1L12' not in candidates
    assert candidates['S1L13']['source_range'] == {'first_span_id': 'S1L13', 'last_span_id': 'S1L13'}
    assert ''.join(segment.text for segment in view.segments) == body
    assert [segment.ordinal for segment in view.segments] == list(range(1, 14))


def test_an_unsupported_new_speaker_breaks_the_previous_speaker_parent():
    body = ('Dan Jarvis said: Our first duty is protecting the public.\n'
            'the spokesperson said: We will propose other measures.\n'
            'We will announce details later.')
    view = build_lossless_source_view((body,), ('UK-01',))
    candidates = context_candidates(view)
    assert candidates['S1L1']['source_range']['last_span_id'] == 'S1L1'
    assert candidates['S1L1']['speaker_parent_hold'] is False
    assert candidates['S1L2']['speaker_parent_hold'] is True
    assert candidates['S1L3']['speaker_parent_hold'] is True


@pytest.mark.parametrize('verb', ('said', 'says', 'stated', 'announced'))
def test_explicit_source_named_speaker_verbs_close_the_next_first_person_span(verb):
    view = build_lossless_source_view((
        f'Dan Jarvis {verb}: Our first duty is protecting the public.\nWe will act.',
    ), ('UK-01',))
    parent = context_candidates(view)['S1L1']
    assert parent['source_range'] == {'first_span_id': 'S1L1', 'last_span_id': 'S1L2'}
    assert parent['speaker_parent_hold'] is False


def test_new_source_named_speaker_starts_a_separate_parent():
    view = build_lossless_source_view((
        'Dan Jarvis said: Our first duty is protecting the public.\n'
        'Alice Smith stated: We will consult residents.\n'
        'We will publish the responses.',
    ), ('UK-01',))
    candidates = context_candidates(view)
    assert candidates['S1L1']['source_range']['last_span_id'] == 'S1L1'
    assert candidates['S1L2']['source_range']['last_span_id'] == 'S1L3'
    assert all(not value['speaker_parent_hold'] for value in candidates.values())


def test_quoted_parent_never_reaches_a_different_acquired_source():
    view = build_lossless_source_view((
        '“Dan Jarvis said: Our first duty is protecting the public.\nWe will act.”',
        'We will publish more details.',
    ), ('UK-01', 'UK-02'))
    candidates = context_candidates(view)
    assert candidates['S1L1']['source_range']['last_span_id'] == 'S1L2'
    assert candidates['S1L1']['source_id'] == 'UK-01'
    assert candidates['S2L1']['source_id'] == 'UK-02'
    assert candidates['S2L1']['speaker_parent_hold'] is True
    with pytest.raises(SourceReferenceError, match='crosses an acquired body'):
        view.resolve_range({'first_span_id': 'S1L1', 'last_span_id': 'S2L1'})


@pytest.mark.parametrize('parent', (
    'We will act.', 'I will act.',
    'the spokesperson said: Our first duty is protecting the public.',
    '[Dan Jarvis] said: Our first duty is protecting the public.',
    '<Dan Jarvis> said: Our first duty is protecting the public.',
    'the spokesperson said: Our duty is to protect Dan Jarvis.',
))
def test_unresolved_or_template_speaker_is_flagged_without_publisher_inference(parent):
    view = build_lossless_source_view((parent + '\nWe will announce details later.',), ('Official-publisher',))
    candidates = context_candidates(view)
    assert len(candidates) == 2
    assert all(value['speaker_parent_hold'] for value in candidates.values())
    assert all(value['source_range']['first_span_id'] == value['source_range']['last_span_id']
               for value in candidates.values())


def test_grouped_entities_keep_repeated_occurrences_in_materialiser_order():
    body = ('Dan Jarvis said: Our first duty is protecting the public.\n'
            'We will act, Dan Jarvis said.')
    view = build_lossless_source_view((body,), ('UK-01',))
    parent = context_candidates(view)['S1L1']
    entities = _claim_entities(parent['text'], body, policy_version=view.entity_policy_version)
    assert parent['entities'] == [list(entity) for entity in entities]
    assert parent['entities'].count(['Dan Jarvis', 'PERSON']) == 2
    assert parent['rendering_fragment_count'] == len(entities) + 1


def test_ranges_cover_every_original_span_once_without_changing_source_identity():
    bodies = ('  Dan Jarvis said: Our first duty is protecting the public.\r\n'
              'We will act.\r\n\r\nThird-person transition.\nWe will announce later.',
              '香港首段。\nWe will respond.')
    view = build_lossless_source_view(bodies, ('UK-01', 'HK-01'))
    before = view.manifest, view.manifest_digest, view.body_digests, view.segments
    candidates = context_candidates(view)
    consumed = []
    positions = {segment.span_id: index for index, segment in enumerate(view.segments)}
    for identity, candidate in candidates.items():
        reference = candidate['source_range']
        assert identity == reference['first_span_id']
        assert candidate['text'] == view.resolve_range(reference)[0]
        consumed.extend(view.segments[positions[identity]:positions[reference['last_span_id']] + 1])
    assert tuple(consumed) == view.segments
    assert tuple(''.join(segment.text for segment in consumed if segment.passage_index == index)
                 for index in range(len(bodies))) == bodies
    assert (view.manifest, view.manifest_digest, view.body_digests, view.segments) == before


def test_candidate_range_uses_existing_selected_source_byte_validation():
    view = build_lossless_source_view(('We will act.',), ('UK-01',))
    changed = replace(view, segments=(replace(view.segments[0], start_byte=1),))
    with pytest.raises(SourceReferenceError, match='source range bytes differ'):
        context_candidates(changed)


def test_unbound_explicit_speaker_is_flagged_even_without_first_person_language():
    view = build_lossless_source_view((
        'the spokesperson said: The policy changed.',
    ), ('UK-01',))
    assert context_candidates(view)['S1L1']['speaker_parent_hold'] is True
