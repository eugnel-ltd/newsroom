"""Natural copy needs a separately bound source-support review, not self-approval."""
from copy import deepcopy
from dataclasses import replace

import pytest

from newsroom.control_plane.evidence import GovernedClaimStatus
from newsroom.control_plane.native_story_writer import (
    NativeStoryWriterHold, validate_retained_story, write_native_story,
)
from newsroom.tests.test_zero_quota_write_loop import _candidate_package


def _package():
    _, package = _candidate_package()
    template = package.governed_claims[0]
    facts = (
        ("headline", "Two new community centres will open.", "政府將開設兩個社區中心。", "HEADLINE"),
        ("capacity", "Each centre has 100 places.", "每個中心提供100個名額。", "SUBSTANTIVE"),
        ("replacement", "The service will replace a temporary site.", "新服務將取代臨時服務點。", "CONTEXT"),
        ("estimate", "The cost estimate remains provisional.", "費用估算仍屬暫定。", "CONTEXT"),
    )
    claims = tuple(replace(template, claim_id=identity, claim=claim, supporting_excerpt=claim,
                           rendered_assertion_zh_hant_hk=rendering, claim_role=role,
                           status=GovernedClaimStatus.EXPRESSLY_PROVISIONAL_FACT if identity == "estimate"
                           else GovernedClaimStatus.CONFIRMED_FACT,
                           named_entities=(), rendered_named_entities=(), named_entity_evidence=())
                   for identity, claim, rendering, role in facts)
    return replace(package, governed_claims=claims, passages=("\n".join(item.claim for item in claims),),
                   substantive_new_information=(claims[1].claim,))


DRAFT = {
    "title": "兩個社區中心將啟用　各設100個名額",
    "body": "政府計劃開設兩個社區中心，每個中心提供100個名額。\n\n新服務將接替臨時服務點，惟費用估算仍屬暫定。",
    "format": "ARTICLE",
    "evidence_links": [
        {"governed_claim_id": "headline", "rendered_assertion": "兩個社區中心將啟用"},
        {"governed_claim_id": "capacity", "rendered_assertion": "每個中心提供100個名額。"},
        {"governed_claim_id": "replacement", "rendered_assertion": "新服務將接替臨時服務點"},
        {"governed_claim_id": "estimate", "rendered_assertion": "費用估算仍屬暫定。"},
    ],
}


def _review(request):
    return {
        "source_package_digest": request["source_package_digest"],
        "draft_digest": request["draft_digest"],
        "verdict": "PASS", "covered_claim_ids": ["headline", "capacity", "replacement", "estimate"],
        "sentence_support": [
            {"sentence_index": 0, "claim_ids": ["headline", "capacity"], "verdict": "SUPPORTED"},
            {"sentence_index": 1, "claim_ids": ["headline", "capacity"], "verdict": "SUPPORTED"},
            {"sentence_index": 2, "claim_ids": ["replacement", "estimate"], "verdict": "SUPPORTED"},
        ],
        "factual_checks": {key: "PASS" for key in ("numbers", "entities", "modality", "quotations")},
    }


def test_natural_multi_paragraph_report_is_separately_reviewed_against_compact_sources():
    package, calls = _package(), []

    def generate(request):
        calls.append("draft")
        assert request["source_package_digest"] == package.digest
        assert "passages" not in request["evidence"]
        return deepcopy(DRAFT)

    def review(request):
        calls.append("review")
        assert request["draft"] == DRAFT
        assert request["sentences"] == [DRAFT["title"], *DRAFT["body"].split("\n\n")]
        assert "passages" not in request["evidence"]
        return _review(request)

    result = write_native_story(package, generate=generate, review=review)

    assert calls == ["draft", "review"]
    assert result.copy.title == DRAFT["title"]
    assert result.copy.body == DRAFT["body"]
    assert result.copy.evidence_package_digest == package.digest
    assert result.format == "ARTICLE"


def test_one_verbatim_evidence_span_can_support_multiple_reviewed_sentences():
    draft=deepcopy(DRAFT)
    paragraph='政府計劃開設兩個社區中心。每個中心提供100個名額。'
    draft['body']=paragraph+'\n\n'+DRAFT['body'].split('\n\n')[1]
    draft['evidence_links'][1]['rendered_assertion']=paragraph
    def review(request):
        result=_review(request)
        result['sentence_support']=[
            {'sentence_index':0,'claim_ids':['headline','capacity'],'verdict':'SUPPORTED'},
            {'sentence_index':1,'claim_ids':['headline','capacity'],'verdict':'SUPPORTED'},
            {'sentence_index':2,'claim_ids':['capacity'],'verdict':'SUPPORTED'},
            {'sentence_index':3,'claim_ids':['replacement','estimate'],'verdict':'SUPPORTED'},
        ]
        return result
    result=write_native_story(_package(),generate=lambda _:draft,review=review)
    assert result.copy.body==draft['body']
    assert all(item.result=='PASS'for item in result.validators)


def test_source_dated_copy_retains_original_review_and_revalidates_derivation():
    from newsroom.authority.canonical import digest_canonical
    from newsroom.tests.test_native_story_dates import _fixture
    from newsroom.control_plane.native_story_writer import validate_retained_story
    draft,review,package,currentness=_fixture()
    # Date mechanics alone: publisher-name localisation requires its own
    # governed entity evidence and is not supplied by this disposable fixture.
    draft['title']=draft['title'].removeprefix('英國內政部：')
    draft['body']=draft['body'].removeprefix('英國內政部表示，')
    review['draft_digest']=digest_canonical(draft)
    result=write_native_story(package,generate=lambda _:deepcopy(draft),review=lambda _:deepcopy(review),
        source_currentness=currentness)
    record=result.review.as_record()
    assert record['draft_digest']==review['draft_digest']
    assert record['date_derivation']['original_draft']==draft
    assert '2026年10月1日'in result.copy.body
    assert all(item.result=='PASS'for item in validate_retained_story(result.copy,package,record,result.format,
        source_currentness=currentness))
    tampered=deepcopy(record);tampered['date_derivation']['anchor']['resolved_date']='2026-10-04'
    assert any(item.result=='FAIL'for item in validate_retained_story(result.copy,package,tampered,result.format,
        source_currentness=currentness))
    assert result.review.as_record()["verdict"] == "PASS"
    assert all(check.result == "PASS" for check in result.validators)


@pytest.mark.parametrize("mutation", ["number", "certainty", "entity", "quote", "unmapped", "missing-claim"])
def test_fabricated_or_unmapped_copy_is_held_even_if_a_review_asserts_pass(mutation):
    draft = deepcopy(DRAFT)
    if mutation == "number":
        draft["body"] = draft["body"].replace("100", "999")
        draft["evidence_links"][1]["rendered_assertion"] = "每個中心提供999個名額。"
    elif mutation == "certainty":
        draft["body"] = draft["body"].replace("費用估算仍屬暫定。", "費用已確定。")
        draft["evidence_links"][3]["rendered_assertion"] = "費用已確定。"
    elif mutation == "entity":
        draft["body"] = draft["body"].replace("政府計劃", "李小明公布政府計劃")
    elif mutation == "quote":
        draft["body"] = draft["body"].replace("惟費用", "當局稱「全部免費」，惟費用")
    elif mutation == "unmapped":
        draft["body"] += "\n\n服務亦設免費接送。"
    else:
        draft["evidence_links"].pop()
    with pytest.raises(NativeStoryWriterHold):
        write_native_story(_package(), generate=lambda _: draft, review=_review)


@pytest.mark.parametrize("mutation", ["unknown", "hold", "empty", "truncated", "package", "draft", "coverage", "sentence", "fact", "mapping"])
def test_partial_unknown_or_unbound_review_never_admits_copy(mutation):
    def review(request):
        result = _review(request)
        if mutation in {"unknown", "hold"}:
            result["verdict"] = mutation.upper()
        elif mutation == "empty":
            return {}
        elif mutation == "truncated":
            return '{"verdict":"PASS"'
        elif mutation in {"package", "draft"}:
            result["source_package_digest" if mutation == "package" else "draft_digest"] = "sha256:stale"
        elif mutation == "coverage":
            result["covered_claim_ids"].pop()
        elif mutation == "sentence":
            result["sentence_support"].pop()
        elif mutation == "fact":
            result["factual_checks"]["numbers"] = "UNKNOWN"
        else:
            result["sentence_support"][1]["claim_ids"] = ["estimate"]
        return result

    with pytest.raises(NativeStoryWriterHold):
        write_native_story(_package(), generate=lambda _: deepcopy(DRAFT), review=review)


def test_writer_self_review_is_denied_before_either_call():
    calls = []

    def self_review(request):
        calls.append(request)
        return deepcopy(DRAFT)

    with pytest.raises(NativeStoryWriterHold, match="SEPARATE_REVIEW"):
        write_native_story(_package(), generate=self_review, review=self_review)
    assert calls == []


def test_retained_review_revalidates_exact_copy_without_model_redispatch():
    package = _package()
    result = write_native_story(package, generate=lambda _: deepcopy(DRAFT), review=_review)
    record = result.review.as_record()
    record["review_invocation_id"] = "accounted-separate-review"
    assert all(check.result == "PASS" for check in validate_retained_story(result.copy, package, record, result.format))
    changed = replace(result.copy, body=result.copy.body.replace("接替", "替代"))
    assert any(check.result == "FAIL" for check in validate_retained_story(changed, package, record, result.format))
    record["verdict"] = "UNKNOWN"
    assert result.review.as_record()["verdict"] == "PASS"


def test_sparse_weather_brief_is_supported_without_a_fake_article_word_quota():
    package = _package()
    claims = tuple(replace(package.governed_claims[index], claim_id=identity, claim=source,
                           supporting_excerpt=source, rendered_assertion_zh_hant_hk=rendering)
                   for index, identity, source, rendering in (
                       (0, "headline", "The thunderstorm warning is cancelled.", "雷暴警告已取消。"),
                       (1, "capacity", "The warning no longer applies.", "雷暴警告已不再生效。")))
    package = replace(package, governed_claims=claims, passages=("\n".join(claim.claim for claim in claims),))
    draft = {"title": "雷暴警告取消", "body": "雷暴警告現已取消，不再生效。", "format": "BRIEF",
             "evidence_links": [{"governed_claim_id": "headline", "rendered_assertion": "雷暴警告取消"},
                                {"governed_claim_id": "capacity", "rendered_assertion": "不再生效。"}]}

    def review(request):
        result = _review(request)
        result["covered_claim_ids"] = ["headline", "capacity"]
        result["sentence_support"] = result["sentence_support"][:2]
        return result

    result = write_native_story(package, generate=lambda _: draft, review=review)
    assert result.format == "BRIEF"
    assert result.copy.body == draft["body"]
    assert len(result.copy.body) < 50
    assert all(check.result == "PASS" for check in result.validators)


def test_source_quotation_keeps_its_closing_mark_with_the_reviewed_sentence():
    package = _package()
    quoted = "新服務將取代臨時服務點。"
    claim = replace(package.governed_claims[2], claim=f"The statement says: {quoted}",
                    supporting_excerpt=f"The statement says: {quoted}", quotations=(quoted,),
                    rendered_assertion_zh_hant_hk=f"當局表示：「{quoted}」")
    package = replace(package, governed_claims=(*package.governed_claims[:2], claim, package.governed_claims[3]))
    draft = deepcopy(DRAFT)
    draft["body"] = DRAFT["body"].replace("新服務將接替臨時服務點，", f"當局表示：「{quoted}」")
    draft["evidence_links"][2]["rendered_assertion"] = f"當局表示：「{quoted}」"

    def review(request):
        assert request["sentences"][2:] == [f"當局表示：「{quoted}」", "惟費用估算仍屬暫定。"]
        result = _review(request)
        result["sentence_support"][2]["claim_ids"] = ["replacement"]
        result["sentence_support"].append({"sentence_index": 3, "claim_ids": ["estimate"], "verdict": "SUPPORTED"})
        return result

    assert write_native_story(package, generate=lambda _: draft, review=review).copy.body == draft["body"]


def test_reviewed_headline_does_not_require_duplicate_writer_span_metadata():
    draft = deepcopy(DRAFT)
    draft['evidence_links'][0]['rendered_assertion'] = '政府計劃開設兩個社區中心'
    result = write_native_story(_package(), generate=lambda _: draft, review=_review)
    assert result.copy.title == DRAFT['title']
    assert result.copy.evidence_links[0].rendered_assertion in result.copy.body
    assert all(item.result == 'PASS' for item in result.validators)


@pytest.mark.parametrize('failure', ['unknown', 'substantive_only', 'unreviewed', 'invented_number'])
def test_headline_paraphrase_still_requires_bound_headline_support(failure):
    draft = deepcopy(DRAFT)
    draft['evidence_links'][0]['rendered_assertion'] = '政府計劃開設兩個社區中心'
    if failure == 'invented_number':
        draft['title'] = draft['title'].replace('100', '999')
    def review(request):
        result = _review(request)
        first = result['sentence_support'][0]
        if failure == 'unknown': first['claim_ids'] = ['missing']
        elif failure == 'substantive_only': first['claim_ids'] = ['capacity']
        elif failure == 'unreviewed': first['verdict'] = 'UNKNOWN'
        return result
    with pytest.raises(NativeStoryWriterHold):
        write_native_story(_package(), generate=lambda _: draft, review=review)
