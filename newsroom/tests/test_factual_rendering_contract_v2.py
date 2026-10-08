"""Closed v4 factual lexemes, independent of translation/editorial judgement."""
import pytest

from newsroom.control_plane import evidence


def test_same_customer_is_prose_not_an_invented_count():
    assert evidence.factual_rendering_is_bound_v2(
        "same customer help email address", "同一客戶支援電郵地址",
    )


def test_unparsed_english_number_words_and_chinese_classifiers_remain_visible():
    assert not evidence.factual_rendering_is_bound_v2("There are two rooms.", "有七個房間。")
    assert not evidence.factual_rendering_is_bound_v2("There are two rooms.", "有兩個房間。")
    assert evidence.factual_rendering_is_bound_v2("the same customer", "同一名客戶")


@pytest.mark.parametrize("source,target", (
    ("長度三公里。", "長度四公里。"),
    ("兩公斤", "五公斤"),
    ("壹佰元", "貳佰元"),
    ("三分之二", "三分之一"),
))
def test_unsupported_han_numerical_forms_never_become_empty_fact_streams(source, target):
    check = evidence.factual_rendering_is_bound_v2
    assert not check(source, target)
    # Unknown forms remain a hold even when their literal text is unchanged.
    assert not check(source, source)
    assert not check(target, target)


def test_same_customer_exemption_does_not_hide_a_separate_han_quantity():
    check = evidence.factual_rendering_is_bound_v2
    assert check("The same customer waits 2 hours.", "同一名客戶等候120分鐘。")
    assert not check("同一客戶購買兩公斤。", "同一客戶購買五公斤。")
    assert not check("同一客戶購買兩公斤。", "同一客戶購買。")
    assert not check("同一客戶購買。", "同一客戶購買兩公斤。")


def test_source_literal_masking_preserves_financial_glyphs_but_not_other_quantities():
    check = evidence.factual_rendering_is_bound_v2
    source, target = "Contact 壹佰集團.", "請聯絡壹佰集團。"
    assert not check(source, target)
    assert check(source, target, literals=("壹佰集團",))
    assert not check(source, "請聯絡貳佰集團。", literals=("壹佰集團",))
    assert not check(source, "請聯絡。", literals=("壹佰集團",))
    assert not check("壹佰集團購買兩公斤。", "壹佰集團購買五公斤。", literals=("壹佰集團",))


@pytest.mark.parametrize("text,literal", (("兩公斤", "公斤"), ("兩年半", "半")))
def test_literal_masking_cannot_strip_a_suffix_from_an_uncovered_han_quantity(text, literal):
    assert not evidence.factual_rendering_is_bound_v2(text, text, literals=(literal,))


def test_one_of_is_selection_and_preserves_qualifiers():
    check = evidence.factual_rendering_is_bound_v2
    assert check("one of the following roles", "其中一個角色")
    assert not check("one of the following roles", "以下角色")
    assert not check("only one of the following roles", "其中一個角色")
    assert check("at least one of the following roles", "至少其中一個角色")
    assert not check("at least one of the following roles", "其中一個角色")


@pytest.mark.parametrize("source,target,unit", (
    ("academic year 2025 to 2026", "2025至2026學年", "ACADEMIC_YEAR"),
    ("financial year 2026 to 2027", "2026至2027財政年度", "FINANCIAL_YEAR"),
))
def test_year_ranges_require_the_complete_explicit_unit(source, target, unit):
    expected = ("YEAR_RANGE", unit, int(source[-12:-8]), int(source[-4:]))
    assert evidence.canonical_localised_fact_v2(source) == expected
    assert evidence.canonical_localised_fact_v2(target) == expected
    assert evidence.factual_rendering_is_bound_v2(source, target, ((source, target),))
    assert not evidence.localised_fact_is_bound_v2(
        source[-12:], target, source, source, target,
    )


def test_plain_and_typed_ranges_keep_endpoints_order_units_and_count():
    check = evidence.factual_rendering_is_bound_v2
    assert check("2025 to 2026", "2025至2026")
    assert check("6 to 8 months", "六至八個月")
    assert not check("6 to 8 months", "六至八年")
    assert not check("6 to 8 months", "八至六個月")
    assert not check("6 to 8 months; 6 to 8 months", "六至八個月")
    assert not check("academic year 2025 to 2026", "2025至2026財政年度")
    assert not check("academic year 2025 to 2026", "2025至2026")


@pytest.mark.parametrize("source,target", (
    ("6 months", "六個月"), ("2 years", "2年"),
    ("£2 million", "二百萬英鎊"), ("4 May 2026", "2026年5月4日"),
    ("May 2026", "2026年5月"),
    ("4 May 2026 at 15:00", "2026年5月4日下午3時00分"),
    ("2 buses", "兩輛巴士"),
    ("two schools", "兩間學校"), ("2 hours", "120分鐘"),
))
def test_legacy_supported_scalar_pairs_are_complete_v2_occurrences(source, target):
    assert evidence.localised_fact_is_bound_v2(source, target, source, source, target)
    assert evidence.factual_rendering_is_bound_v2(source, target, ((source, target),))


def test_embedded_units_and_explicit_bounds_are_not_erased_or_inferred():
    check = evidence.factual_rendering_is_bound_v2
    assert not check("計劃2年", "計劃3年")
    assert not check("計劃兩年", "計劃三年")
    assert check("at least 6 months", "至少六個月")
    assert not check("at least 6 months", "六個月")
    assert not check("6 months", "六年")


@pytest.mark.parametrize("source,target", (
    ("about 6 months", "六個月"),
    ("approximately 6 months", "六個月"),
    ("6 months", "約六個月"),
    ("within 6 months", "六個月"),
    ("around 6 months", "六個月"),
))
def test_approximation_and_deadline_qualifiers_are_never_erased_or_added(source, target):
    assert not evidence.factual_rendering_is_bound_v2(source, target)


@pytest.mark.parametrize("source,target", (
    ("about 6 months", "約六個月"),
    ("approximately 6 months", "約六個月"),
    ("around 6 months", "約六個月"),
    ("within 6 months", "六個月內"),
    ("within 6 months", "六個月期限內"),
))
def test_approximation_and_deadline_pairs_bind_the_complete_qualified_occurrence(source, target):
    assert evidence.localised_fact_is_bound_v2(source, target, source, source, target)
    assert evidence.factual_rendering_is_bound_v2(source, target, ((source, target),))


def test_qualified_occurrences_keep_their_operator_value_and_lookup_boundary():
    check = evidence.factual_rendering_is_bound_v2
    assert not check("6 months", "六個月期限內")
    assert not check("about 6 months", "約七個月")
    assert not check("within 6 months", "最多六個月")
    assert not check("within 6 months", "約六個月")
    for source, target in (("about 6 months", "約六個月"), ("within 6 months", "六個月期限內")):
        assert not evidence.localised_fact_is_bound_v2("6 months", "六個月", source, source, target)
        assert not check(source, target, (("6 months", "六個月"),))
    assert not check("about 6 months", "六個月", literals=("about",))


def test_partial_quantity_key_and_unknown_units_are_not_laundered():
    assert not evidence.localised_fact_is_bound_v2("£2", "二英鎊", "£2 million", "£2 million", "二英鎊")
    assert not evidence.factual_rendering_is_bound_v2("2 kg", "2 m", (("2", "2"),))
    assert not evidence.factual_rendering_is_bound_v2("-2", "2", (("2", "2"),))
    assert not evidence.factual_rendering_is_bound_v2("1/2", "1", (("1", "1"),))


@pytest.mark.parametrize("source,target", (
    ("2 years", "兩年半"),
    ("2°C", "華氏2度"),
    ("2噸", "2毫升"),
    ("2度", "2呎"),
))
def test_unsupported_suffixes_and_units_do_not_become_shorter_known_facts(source, target):
    assert not evidence.factual_rendering_is_bound_v2(source, target)
    assert not evidence.factual_rendering_is_bound_v2(target, target)
    assert not evidence.factual_rendering_is_bound_v2(source, target, (("2", "2"),))


@pytest.mark.parametrize("source,target", (
    ("next year", "下個月"),
    ("yesterday", "明天"),
))
def test_uncertified_relative_time_never_becomes_empty_fact_streams(source, target):
    assert not evidence.factual_rendering_is_bound_v2(source, target)
    assert not evidence.factual_rendering_is_bound_v2(source, source)
    assert not evidence.factual_rendering_is_bound_v2(target, target)
    assert not evidence.factual_rendering_is_bound_v2("The scheme is open.", target)


def test_opaque_literals_and_certified_derived_year_pairs():
    assert evidence.factual_rendering_is_bound_v2(
        "Use the same customer help email address contact.help@education.gov.uk for post-16 funding.",
        "請使用同一客戶支援電郵地址contact.help@education.gov.uk處理post-16資助。",
        literals=("contact.help@education.gov.uk", "post-16"),
    )
    assert evidence.factual_rendering_is_bound_v2(
        "post-16 contact@example.com", "post-16 contact@example.com",
        literals=("post-16", "contact@example.com"),
    )
    assert not evidence.factual_rendering_is_bound_v2(
        "post-16", "post-17", literals=("post-16",),
    )
    assert evidence.factual_rendering_is_bound_v2(
        "next year", "2027年", (("next year", "2027年"),),
        derived_pairs=(("next year", "2027年"),),
    )
    assert not evidence.factual_rendering_is_bound_v2(
        "next year", "2027年", (("next year", "2027年"),),
    )


def test_modal_may_is_not_a_calendar_month():
    assert evidence.factual_rendering_is_bound_v2("The customer may apply.", "客戶可以申請。")


def test_semiconductors_are_ordinary_prose_not_a_half_quantity():
    assert evidence.factual_rendering_is_bound_v2("The scheme supports semiconductors.", "計劃支援半導體。")


def _reference(contract):
    return tuple(sorted({"contract": contract, "operation": "SOURCE_RENDERING",
        "invocation_id": "sha256:" + "1" * 64,
        "raw_admission_id": "00000000-0000-4000-8000-000000000001",
        "receipt_admission_id": "00000000-0000-4000-8000-000000000002"}.items()))


@pytest.mark.parametrize("contract", (evidence.SOURCE_RENDERING_CONTRACT, evidence.SOURCE_RENDERING_CONTRACT_V2))
def test_rendering_locator_keeps_the_same_exact_five_fields(contract):
    reference = _reference(contract)
    assert evidence.source_rendering_reference(reference) == dict(reference)
    with pytest.raises(ValueError):
        evidence.source_rendering_reference((*reference, ("extra", "field")))


def test_only_v2_reference_opts_governed_claim_pairs_into_v2():
    values = dict(claim_id="claim", claim="The scheme lasts 2 to 3 months.", passage_index=0,
        supporting_excerpt="The scheme lasts 2 to 3 months.", source_ids=("source",),
        source_record_ids=("record",), source_authority_decision_ids=("authority",),
        rights_decision_ids=("rights",), dependency_evidence_ids=("dependency",),
        evidential_origin_ids=("origin",), authority_class=evidence.ClaimAuthorityClass.RESPONSIBLE_PRIMARY,
        authority_scope="the scheme", status=evidence.GovernedClaimStatus.CONFIRMED_FACT,
        attribution="the responsible body", rendered_assertion_zh_hant_hk="安排持續兩至三個月。",
        claim_role="SUBSTANTIVE", semantic_relation_evidence_id="semantic",
        localised_factual_expressions=(("2 to 3 months", "兩至三個月"),))
    for reference in ((), _reference(evidence.SOURCE_RENDERING_CONTRACT)):
        with pytest.raises(ValueError, match="equivalent exact claim facts"):
            evidence.GovernedClaimEvidence(**values, source_rendering_ref=reference)
    claim = evidence.GovernedClaimEvidence(**values, source_rendering_ref=_reference(evidence.SOURCE_RENDERING_CONTRACT_V2))
    assert claim.localised_factual_expressions == (("2 to 3 months", "兩至三個月"),)


def test_invalid_closed_values_return_unknown_and_never_hide_a_quantity():
    assert evidence.canonical_localised_fact_v2("financial year 0000 to 2027") is None
    assert evidence.canonical_localised_fact_v2("9" * 5_000) is None
    assert not evidence.factual_rendering_is_bound_v2("2 kg", "2", literals=("kg",))
    assert not evidence.factual_rendering_is_bound_v2("plain text", "普通內容", literals=([],))
