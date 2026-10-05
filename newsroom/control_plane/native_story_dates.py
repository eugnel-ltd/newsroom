"""A deterministic dated derivative of an unchanged, separately reviewed copy.

The caller authenticates the Evidence Package/currentness and validates the
original draft/review normally. This module grants no source or model authority.
The first contract supports one unquoted publisher-relative yesterday only.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import re
from zoneinfo import ZoneInfo

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from .admission import _qualification_text_is_negative
from .evidence import GovernedClaimStatus

VERSION = 'newsroom.native-story-date-derivation.v1'
_GOVUK = frozenset({'UK-01', 'UK-02', 'UK-03', 'UK-05'})
_INSTANT = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})')
_RELATIVE = re.compile(r'昨日|昨天|今日|今天|明日|明天|今晚|昨晚|前日|前天|明早')
_YESTERDAY = re.compile(r'\byesterday\b', re.IGNORECASE)
_SENTENCE = re.compile(r'[^。！？!?\n]+(?:[。！？!?][」』”’\"]*)?')


def _require(passed, reason):
    if not passed:
        from .native_story_writer import NativeStoryWriterHold
        raise NativeStoryWriterHold('NATIVE_STORY_DATE_'+reason+'_QUALITY_HOLD')


def _unquoted(text, start, end):
    for left, right in [('「', '」'), ('『', '』'), ('“', '”'), ('‘', '’')]:
        opening = text.rfind(left, 0, start)
        closing = text.find(right, opening + 1) if opening >= 0 else -1
        _require(opening < 0 or 0 <= closing < start, 'QUOTED_CLOCK')
    _require(text[:start].count('"') % 2 == 0 and not re.search(r'["“”「」『』]', text[start:end]), 'QUOTED_CLOCK')


def _anchor(currentness):
    published, updated = currentness.publication_time, currentness.version_reference
    _require(type(published) is str and type(updated) is str
             and _INSTANT.fullmatch(published) and _INSTANT.fullmatch(updated), 'CHRONOLOGY_INCOMPLETE')
    try:
        publication = datetime.fromisoformat(published.replace('Z', '+00:00'))
        update = datetime.fromisoformat(updated.replace('Z', '+00:00'))
    except ValueError:
        _require(False, 'CHRONOLOGY_INCOMPLETE')
    _require(publication == update, 'ANCHOR_AMBIGUOUS')
    calendars = {name: publication.astimezone(ZoneInfo(name)).date() for name in ('UTC', 'Europe/London')}
    _require(len(set(calendars.values())) == 1, 'CALENDAR_AMBIGUOUS')
    day = calendars['UTC'] - timedelta(days=1)
    return {'publication_time': published, 'source_updated_time': updated,
            'calendar_basis': 'GOVUK_UTC_LONDON_CALENDAR_AGREEMENT',
            'calendar_dates': {name: value.isoformat() for name, value in calendars.items()},
            'day_offset': -1, 'resolved_date': day.isoformat()}, f'{day.year}年{day.month}月{day.day}日'


def derive_and_verify(original_draft, original_review, package, source_currentness, *,
                      source_records=(), final_draft=None, date_derivation=None):
    """Return a detached (final draft, proof), or (unchanged draft, None).

    Supplying a final draft and proof verifies their exact recomputation. Both
    must be supplied together. Original review hashes cover only its unchanged
    provider fields, not subsequently attached model receipts or this proof.
    """
    from newsroom.increment10.editorial import SourceCurrentness
    from .native_story_writer import REVIEW_SCHEMA
    _require(type(original_draft) is dict and type(original_review) is dict, 'INPUT')
    _require((final_draft is None) == (date_derivation is None), 'PROOF_PAIR')
    paths = [('/title', original_draft['title']), ('/body', original_draft['body']),
             *((f'/evidence_links/{i}/rendered_assertion', link['rendered_assertion'])
               for i, link in enumerate(original_draft['evidence_links']))]
    relative = [(path, text) for path, text in paths if _RELATIVE.search(text)]
    if not relative:
        _require(final_draft is None, 'UNEXPECTED_PROOF')
        return deepcopy(original_draft), None
    _require(len(package.source_ids) == len(package.passages) == 1
             and package.source_ids[0] in _GOVUK, 'SOURCE_SCOPE')
    _require(type(source_currentness) is tuple and len(source_currentness) == 1
             and type(source_currentness[0]) is SourceCurrentness, 'CURRENTNESS')
    currentness = source_currentness[0]
    _require(currentness.source_id == package.source_ids[0] and currentness.result == 'PASS'
             and currentness.currency_family == 'CURRENT_VERSION', 'CURRENTNESS')
    from .evidence import _source_body_provenance_is_bound
    _require(type(source_records)is tuple and len(source_records)==1
        and type(source_records[0])is dict,'BODY_PROVENANCE')
    record=source_records[0];origin=record.get('body_provenance')
    _require(_source_body_provenance_is_bound(record) and type(origin)is dict
        and origin['kind']=='GOVUK_CONTENT_API_PAGE_TEXT'
        and record.get('source_id')==currentness.source_id
        and record.get('publication_time')==currentness.publication_time
        and record.get('language')=='en-GB'
        and origin['transport_evidence_digest']==currentness.evidence_digest
        and origin['body_digest']==digest_bytes(package.passages[0].encode())
        and dict(package.resolved_evidence_records).get(record['record_id'])==digest_canonical(record),
        'BODY_PROVENANCE')
    source = package.passages[0]
    occurrences = list(_YESTERDAY.finditer(source))
    _require(len(occurrences) == 1, 'SOURCE_TOKEN_AMBIGUOUS')
    occurrence = occurrences[0]
    _unquoted(source, occurrence.start(), occurrence.end())
    claims = [claim for claim in package.governed_claims if _YESTERDAY.search(claim.claim)]
    _require(len(claims) == 1, 'CLAIM_AMBIGUOUS')
    claim = claims[0]
    _require(claim.status is GovernedClaimStatus.CONFIRMED_FACT and not claim.quotations
             and claim.source_ids == package.source_ids and claim.passage_index == 0
             and claim.claim == claim.supporting_excerpt and source.count(claim.claim) == 1,
             'CLAIM_BINDING')
    claim_start = source.index(claim.claim)
    _require(claim_start <= occurrence.start() < occurrence.end() <= claim_start + len(claim.claim), 'SOURCE_TOKEN_BINDING')
    # Inspect the complete source parent, not a clipped model lookup. A future
    # policy target may coexist with a past launch, but the launch's own sentence
    # cannot be conditional, negated, provisional or quoted speech.
    parent = next((match.group() for match in re.finditer(r'[^.!?。！？\n]+(?:[.!?。！？]+)?', source)
                   if match.start() <= occurrence.start() < match.end()), '')
    _require(parent and not _qualification_text_is_negative(parent)
             and not _qualification_text_is_negative(claim.claim)
             and not re.search(r'\b(?:if|unless|conditional|pending approval|subject to approval|said|stated|quoted)\b', parent, re.I),
             'SOURCE_CONTEXT')
    core_review = {key: original_review.get(key) for key in REVIEW_SCHEMA['required']}
    _require(core_review['draft_digest'] == digest_canonical(original_draft)
             and core_review['source_package_digest'] == package.digest and core_review['verdict'] == 'PASS', 'ORIGINAL_REVIEW')
    sentences = [original_draft['title'], *[m.group().strip() for m in _SENTENCE.finditer(original_draft['body']) if m.group().strip()]]
    support = core_review['sentence_support']
    _require(type(support) is list and [item['sentence_index'] for item in support] == list(range(len(sentences)))
             and all(item['verdict'] == 'SUPPORTED' for item in support)
             and claim.claim_id in support[0]['claim_ids'], 'REVIEW_ALIGNMENT')
    links = original_draft['evidence_links']
    selected = [(i, link) for i, link in enumerate(links) if _RELATIVE.search(link['rendered_assertion'])]
    _require(len(selected) == 1 and selected[0][1]['governed_claim_id'] == claim.claim_id
             and selected[0][1]['rendered_assertion'] == claim.rendered_assertion_zh_hant_hk
             and original_draft['body'].count(selected[0][1]['rendered_assertion']) == 1, 'COPY_ALIGNMENT')
    expected_paths = {'/title', '/body', f'/evidence_links/{selected[0][0]}/rendered_assertion'}
    _require({path for path, _ in relative} == expected_paths, 'COPY_ALIGNMENT')
    anchor, target = _anchor(currentness)
    substitutions = []
    for path, text in relative:
        matches = list(_RELATIVE.finditer(text))
        _require(len(matches) == 1 and matches[0].group() == '昨日', 'COPY_TOKEN_AMBIGUOUS')
        match = matches[0]
        _unquoted(text, match.start(), match.end())
        if path == '/body':
            sentence_index = next(i + 1 for i, sentence in enumerate(_SENTENCE.finditer(text))
                                  if sentence.start() <= match.start() < sentence.end())
            _require(claim.claim_id in support[sentence_index]['claim_ids'], 'REVIEW_ALIGNMENT')
            link_start = text.index(selected[0][1]['rendered_assertion'])
            _require(link_start <= match.start() < link_start + len(selected[0][1]['rendered_assertion']), 'COPY_ALIGNMENT')
        substitutions.append({'path': path, 'start_byte': len(text[:match.start()].encode()),
                              'end_byte': len(text[:match.end()].encode()), 'original': '昨日', 'replacement': target,
                              'original_field_digest': digest_bytes(text.encode())})
    final = deepcopy(original_draft)
    final['title'] = original_draft['title'].replace('昨日', target)
    final['body'] = original_draft['body'].replace('昨日', target)
    final['evidence_links'][selected[0][0]]['rendered_assertion'] = selected[0][1]['rendered_assertion'].replace('昨日', target)
    proof = {'version': VERSION, 'original_draft': deepcopy(original_draft),
             'original_draft_digest': digest_canonical(original_draft), 'original_review_digest': digest_canonical(core_review),
             'original_review_applies_to': 'ORIGINAL_DRAFT_ONLY', 'final_draft_digest': digest_canonical(final),
             'source_package_digest': package.digest, 'source_id': currentness.source_id, 'claim_id': claim.claim_id,
             'source_body_digest': digest_bytes(source.encode()), 'source_claim_digest': digest_bytes(claim.claim.encode()),
             'source_currentness_digest': digest_canonical(currentness.value()), 'anchor': anchor,
             'source_record_digest':digest_canonical(record),
             'source_relative_range': {'token': occurrence.group(), 'start_byte': len(source[:occurrence.start()].encode()),
                                       'end_byte': len(source[:occurrence.end()].encode())},
             'substitutions': substitutions}
    if final_draft is not None:
        try:
            exact = (canonical_json_bytes(final_draft) == canonical_json_bytes(final)
                     and canonical_json_bytes(date_derivation) == canonical_json_bytes(proof))
        except (TypeError, ValueError):
            exact = False
        _require(exact, 'DERIVATION_DIFFERS')
    return final, proof
