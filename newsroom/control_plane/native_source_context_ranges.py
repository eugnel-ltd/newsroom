"""Lossless Source candidate ranges with explicit first-person speaker parents."""
from __future__ import annotations

import re

from .native_assessor_references import SourceReferenceError, SourceView

_FIRST_PERSON = re.compile(r'\b(?:we|our|us|I)\b', re.IGNORECASE)
_SPEAKER = re.compile(r'''^\s*["“‘']?(?P<subject>[^:\n.!?;]{1,256}?)\s+(?:said|says|stated|announced)\s*:\s*\S''', re.IGNORECASE)


def declared_speakers(text, entities):
    """Use one Source-bound speaker grammar in grouping and copy validation."""
    match = _SPEAKER.match(text)
    if match is None:
        return ()
    subject = match['subject'].strip()
    if any(character in subject for character in '{}[]<>'):
        return ()
    return tuple(name for name, kind in entities if kind in {'PERSON', 'ORGANISATION'} and (
        subject == name or subject.endswith(' ' + name) or subject.startswith(name + ', ')
    ))


def _has_speaker(segment) -> bool:
    return bool(declared_speakers(segment.text, segment.entities))


def context_candidates(view: SourceView) -> dict[str, dict]:
    """Partition the original spans; a range is evidence, not a speaker grant."""
    if not isinstance(view, SourceView):
        raise SourceReferenceError('context Source view differs')
    candidates = {}
    index = 0
    while index < len(view.segments):
        first = view.segments[index]
        last = index
        speaker = _has_speaker(first)
        if speaker:
            while last + 1 < len(view.segments):
                following = view.segments[last + 1]
                if (following.passage_index != first.passage_index
                        or not _FIRST_PERSON.search(following.text) or _SPEAKER.match(following.text)):
                    break
                last += 1
        reference = {'first_span_id': first.span_id, 'last_span_id': view.segments[last].span_id}
        text, _, source_id = view.resolve_range(reference)
        entities = tuple(entity for segment in view.segments[index:last + 1] for entity in segment.entities)
        candidates[first.span_id] = {'source_id': source_id, 'text': text,
            'entities': [list(entity) for entity in entities], 'rendering_fragment_count': len(entities) + 1,
            'source_range': reference,
            'speaker_parent_hold': bool(_FIRST_PERSON.search(text) or _SPEAKER.match(first.text)) and not speaker}
        index = last + 1
    return candidates
