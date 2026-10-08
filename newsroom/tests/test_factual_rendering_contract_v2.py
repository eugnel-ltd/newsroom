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


def test_partial_quantity_key_and_unknown_units_are_not_laundered():
    assert not evidence.localised_fact_is_bound_v2("£2", "二英鎊", "£2 million", "£2 million", "二英鎊")
    assert not evidence.factual_rendering_is_bound_v2("2 kg", "2 m", (("2", "2"),))
    assert not evidence.factual_rendering_is_bound_v2("-2", "2", (("2", "2"),))
    assert not evidence.factual_rendering_is_bound_v2("1/2", "1", (("1", "1"),))


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
