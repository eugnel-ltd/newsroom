"""A dated derivative never relabels the original source-support review."""
from copy import deepcopy
from dataclasses import replace

import pytest

from newsroom.authority.canonical import digest_bytes, digest_canonical
from newsroom.control_plane.evidence import EvidencePackage, GovernedClaimStatus
from newsroom.control_plane.native_story_dates import derive_and_verify as _derive
from newsroom.control_plane.native_story_writer import NativeStoryWriterHold
from newsroom.increment10.editorial import SourceCurrentness
from newsroom.tests.test_factual_localisation import _claim


def _source_record(text):
    from newsroom.control_plane.evidence import _SOURCE_RECORD_FIELDS
    record={key:'fixture'for key in _SOURCE_RECORD_FIELDS}
    record.update(record_id=digest_bytes(b'acquisition'),source_id='UK-01',
        originating_artefact_digest=digest_bytes(text.encode()),language='en-GB',
        publication_time='2026-10-02T15:08:14.000000Z',
        body_provenance={'version':'newsroom.acquisition-body-provenance.v1',
            'kind':'GOVUK_CONTENT_API_PAGE_TEXT','body_digest':digest_bytes(text.encode()),
            'acquisition_receipt_digest':digest_bytes(b'acquisition'),
            'transport_evidence_digest':digest_bytes(b'transport')})
    return record


def derive_and_verify(draft,review,package,currentness,**kwargs):
    return _derive(draft,review,package,currentness,
        source_records=kwargs.pop('source_records',(_source_record(package.passages[0]),)),**kwargs)


def _fixture():
    text = ('The changes will be subject to a consultation which was launched yesterday. '
            'It seeks views from the public, industry and business to gauge how these changes would affect them.')
    rendered = '這些改動將須進行一項已於昨日展開的諮詢。該諮詢向公眾、業界及商界徵詢意見，以衡量這些改動會如何影響他們。'
    claim = replace(_claim('1 October 2026', '2026年10月1日'), claim_id='launch',
                    claim=text, supporting_excerpt=text, source_ids=('UK-01',),
                    rendered_assertion_zh_hant_hk=rendered, claim_role='HEADLINE',
                    localised_factual_expressions=())
    package = EvidencePackage('candidate', 'hypothesis', ('signal',), ('lead',),
                              ('UK-01',), (digest_bytes(text.encode()),), (text,),
                              governed_claims=(claim,))
    record=_source_record(text)
    package=replace(package,resolved_evidence_records=((record['record_id'],digest_canonical(record)),))
    draft = {'title': '英國內政部：改動須經昨日已展開的諮詢',
             'body': '英國內政部表示，'+rendered, 'format': 'BRIEF',
             'evidence_links': [{'governed_claim_id': 'launch', 'rendered_assertion': rendered}]}
    review = {'source_package_digest': package.digest, 'draft_digest': digest_canonical(draft),
              'verdict': 'PASS', 'covered_claim_ids': ['launch'],
              'sentence_support': [{'sentence_index': i, 'claim_ids': ['launch'], 'verdict': 'SUPPORTED'}
                                   for i in range(3)],
              'factual_checks': {key: 'PASS' for key in ('numbers', 'entities', 'modality', 'quotations')}}
    instant = '2026-10-02T15:08:14.000000Z'
    currentness = (SourceCurrentness('UK-01', 'definition', digest_bytes(b'definition'),
                                    'CURRENT_VERSION', instant, '2026-10-05T08:00:00Z', None,
                                    instant, digest_bytes(b'transport'), digest_bytes(b'transport'),
                                    'PASS', 'CURRENT'),)
    return draft, review, package, currentness


def test_source_anchored_derivative_preserves_original_review_and_all_other_copy():
    draft, review, package, currentness = _fixture()
    original = deepcopy((draft, review))
    final, proof = derive_and_verify(draft, review, package, currentness)
    assert final['title'] == '英國內政部：改動須經2026年10月1日已展開的諮詢'
    assert final['body'] == draft['body'].replace('昨日', '2026年10月1日')
    assert final['evidence_links'][0]['rendered_assertion'] == draft['evidence_links'][0]['rendered_assertion'].replace('昨日', '2026年10月1日')
    assert proof['original_draft'] == draft
    assert proof['original_draft_digest'] == review['draft_digest']
    assert proof['original_review_digest'] == digest_canonical(review)
    assert proof['final_draft_digest'] == digest_canonical(final)
    assert proof['anchor']['resolved_date'] == '2026-10-01'
    assert len(proof['substitutions']) == 3
    assert (draft, review) == original
    assert derive_and_verify(draft, review, package, currentness,
                             final_draft=final, date_derivation=proof) == (final, proof)


@pytest.mark.parametrize('case', [
    'different-update', 'date-only', 'missing-currentness', 'wrong-source', 'held-currentness',
    'multiple-sources', 'multiple-source-tokens', 'missing-source-token', 'quoted-source',
    'denied-parent', 'conditional-parent', 'provisional', 'quoted-copy', 'multiple-copy-tokens',
    'review-digest', 'calendar-disagreement', 'denied-selected-context',
])
def test_unproved_relative_clock_or_alignment_is_held(case):
    draft, review, package, currentness = _fixture()
    if case == 'different-update':
        currentness = (replace(currentness[0], version_reference='2026-10-03T15:08:14Z'),)
    elif case == 'date-only':
        currentness = (replace(currentness[0], version_reference='2026-10-02'),)
    elif case == 'missing-currentness':
        currentness = ()
    elif case == 'wrong-source':
        currentness = (replace(currentness[0], source_id='UK-05'),)
    elif case == 'held-currentness':
        currentness = (replace(currentness[0], result='HOLD'),)
    elif case == 'multiple-sources':
        package = replace(package, source_ids=('UK-01', 'UK-05'), passages=(*package.passages, 'Other source.'))
    elif case == 'multiple-source-tokens':
        package = replace(package, passages=(package.passages[0]+' Another event happened yesterday.',))
    elif case == 'missing-source-token':
        package = replace(package, passages=(package.passages[0].replace('yesterday', 'last week'),))
    elif case == 'quoted-source':
        package = replace(package, passages=('“'+package.passages[0]+'”',))
    elif case == 'denied-parent':
        package = replace(package, passages=('Officials deny that '+package.passages[0],))
    elif case == 'conditional-parent':
        package = replace(package, passages=('If approval is granted, '+package.passages[0],))
    elif case == 'denied-selected-context':
        text = package.governed_claims[0].claim.split('. ', 1)[0]+'. The consultation was not launched.'
        package = replace(package, passages=(text,),
                          governed_claims=(replace(package.governed_claims[0], claim=text, supporting_excerpt=text),))
    elif case == 'provisional':
        package = replace(package, governed_claims=(replace(package.governed_claims[0],
                          status=GovernedClaimStatus.EXPRESSLY_PROVISIONAL_FACT),))
    elif case == 'quoted-copy':
        draft['title'] = '「'+draft['title']+'」'
    elif case == 'multiple-copy-tokens':
        draft['body'] += '昨日'
    elif case == 'calendar-disagreement':
        currentness = (replace(currentness[0], publication_time='2026-10-02T23:30:00Z',
                              version_reference='2026-10-02T23:30:00Z'),)
    review.update(source_package_digest=package.digest, draft_digest=digest_canonical(draft))
    if case == 'review-digest':
        review['draft_digest'] = digest_bytes(b'different')
    with pytest.raises(NativeStoryWriterHold, match='QUALITY_HOLD'):
        derive_and_verify(draft, review, package, currentness)


@pytest.mark.parametrize('case', ['other-prose', 'final-digest', 'range', 'anchor', 'original-draft',
                                  'source-currentness', 'relabel-review', 'missing-proof'])
def test_retained_derivation_is_recomputed_not_trusted(case):
    draft, review, package, currentness = _fixture()
    final, proof = derive_and_verify(draft, review, package, currentness)
    if case == 'other-prose':
        final['body'] += '措施已經生效。'
    elif case == 'final-digest':
        proof['final_draft_digest'] = digest_bytes(b'different')
    elif case == 'range':
        proof['substitutions'][0]['start_byte'] += 1
    elif case == 'anchor':
        proof['anchor']['day_offset'] = -2
    elif case == 'original-draft':
        proof['original_draft']['title'] += '其他'
    elif case == 'source-currentness':
        currentness = (replace(currentness[0], evidence_digest=digest_bytes(b'other transport')) ,)
    elif case == 'relabel-review':
        review['draft_digest'] = proof['final_draft_digest']
    elif case == 'missing-proof':
        proof = None
    with pytest.raises(NativeStoryWriterHold, match='QUALITY_HOLD'):
        derive_and_verify(draft, review, package, currentness, final_draft=final, date_derivation=proof)


@pytest.mark.parametrize('instant,expected', [
    ('2026-01-01T15:00:00Z', '2025-12-31'), ('2024-03-01T15:00:00Z', '2024-02-29'),
])
def test_calendar_boundary_arithmetic_keeps_day_precision(instant, expected):
    draft, review, package, currentness = _fixture()
    currentness = (replace(currentness[0], publication_time=instant, version_reference=instant),)
    record=_source_record(package.passages[0]);record['publication_time']=instant
    package=replace(package,resolved_evidence_records=((record['record_id'],digest_canonical(record)),))
    review['source_package_digest']=package.digest
    _, proof = derive_and_verify(draft, review, package, currentness,source_records=(record,))
    assert proof['anchor']['resolved_date'] == expected


def test_review_metadata_is_not_relabelled_as_a_provider_answer():
    draft, review, package, currentness = _fixture()
    raw_review_digest = digest_canonical(review)
    review['model_receipts'] = {'REVIEW': {'response_digest': raw_review_digest}}
    _, proof = derive_and_verify(draft, review, package, currentness)
    assert proof['original_review_digest'] == raw_review_digest
    assert review['draft_digest'] == proof['original_draft_digest']


def test_no_relative_copy_is_detached_without_a_new_proof():
    draft, review, package, _ = _fixture()
    draft['title'] = '政府展開諮詢'
    draft['body'] = '政府展開諮詢。'
    draft['evidence_links'][0]['rendered_assertion'] = draft['body']
    final, proof = derive_and_verify(draft, review, package, ())
    assert final == draft and proof is None
    final['evidence_links'][0]['rendered_assertion'] = 'detached'
    assert draft['evidence_links'][0]['rendered_assertion'] == '政府展開諮詢。'


def test_byte_ranges_reject_boolean_equality_in_a_retained_proof():
    draft, review, package, currentness = _fixture()
    draft['title'] = '昨日已展開的諮詢'
    review['draft_digest'] = digest_canonical(draft)
    final, proof = derive_and_verify(draft, review, package, currentness)
    assert proof['substitutions'][0]['start_byte'] == 0
    proof['substitutions'][0]['start_byte'] = False
    with pytest.raises(NativeStoryWriterHold, match='QUALITY_HOLD'):
        derive_and_verify(draft, review, package, currentness, final_draft=final, date_derivation=proof)


@pytest.mark.parametrize('case',['missing','asset','body','receipt','transport','record-digest','unknown-field','kind-type'])
def test_publisher_clock_requires_authenticated_page_text_body_origin(case):
    draft,review,package,currentness=_fixture()
    record=_source_record(package.passages[0])
    if case=='missing':records=()
    else:
        if case=='asset':record['body_provenance']['kind']='GOVUK_DECLARED_ASSET_TEXT'
        elif case=='body':record['body_provenance']['body_digest']=digest_bytes(b'other')
        elif case=='receipt':record['body_provenance']['acquisition_receipt_digest']=digest_bytes(b'other')
        elif case=='transport':record['body_provenance']['transport_evidence_digest']=digest_bytes(b'other')
        elif case=='unknown-field':record['body_provenance']['trust_me']=True
        elif case=='kind-type':record['body_provenance']['kind']=[]
        records=(record,)
        if case!='record-digest':
            package=replace(package,resolved_evidence_records=((record['record_id'],digest_canonical(record)),))
            review['source_package_digest']=package.digest
        else:
            record['publisher']='changed'
    with pytest.raises(NativeStoryWriterHold,match='BODY_PROVENANCE'):
        _derive(draft,review,package,currentness,source_records=records)


@pytest.mark.parametrize('wrong_claim', [False, True])
def test_source_date_derivative_uses_reviewed_natural_paraphrase(wrong_claim):
    draft, review, package, currentness = _fixture()
    paraphrase = ('有關改動須經諮詢，該項諮詢已於昨日展開，正尋求公眾、業界及商界的意見，'
                  '以衡量這些改動會如何影響他們。')
    draft['body'] = '英國內政部表示，' + paraphrase
    draft['evidence_links'][0]['rendered_assertion'] = paraphrase
    if wrong_claim:
        draft['evidence_links'][0]['governed_claim_id'] = 'unrelated-claim'
    review['draft_digest'] = digest_canonical(draft)
    review['sentence_support'] = [{'sentence_index': index, 'claim_ids': ['launch'],
                                  'verdict': 'SUPPORTED'} for index in range(2)]
    before = deepcopy((draft, review))
    if wrong_claim:
        with pytest.raises(NativeStoryWriterHold, match='COPY_ALIGNMENT'):
            derive_and_verify(draft, review, package, currentness)
        return
    final, proof = derive_and_verify(draft, review, package, currentness)
    assert final['body'] == draft['body'].replace('昨日', '2026年10月1日')
    assert final['evidence_links'][0]['rendered_assertion'] == paraphrase.replace('昨日', '2026年10月1日')
    assert (draft, review) == before
    assert proof['claim_id'] == 'launch' and proof['anchor']['resolved_date'] == '2026-10-01'
    assert derive_and_verify(draft, review, package, currentness,
                            final_draft=final, date_derivation=proof) == (final, proof)
