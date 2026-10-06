"""Month-granularity localisation preserves retained factual precision."""

from dataclasses import replace

import pytest

from newsroom.control_plane.evidence import (
    ClaimAuthorityClass, GovernedClaimEvidence, GovernedClaimStatus,
)


def _claim(source: str, target: str) -> GovernedClaimEvidence:
    text = f"The programme changed in {source}."
    return GovernedClaimEvidence(
        claim_id="claim", claim=text, passage_index=0, supporting_excerpt=text,
        source_ids=("source",), source_record_ids=("record",),
        source_authority_decision_ids=("authority",), rights_decision_ids=("rights",),
        dependency_evidence_ids=("dependency",), evidential_origin_ids=("origin",),
        authority_class=ClaimAuthorityClass.RESPONSIBLE_PRIMARY,
        authority_scope="Programme changes", status=GovernedClaimStatus.CONFIRMED_FACT,
        attribution="The originating authority", claim_role="SUBSTANTIVE",
        rendered_assertion_zh_hant_hk=f"計劃於{target}修訂。",
        semantic_relation_evidence_id="semantic",
        localised_factual_expressions=((source, target),),
    )


@pytest.mark.parametrize(("source", "target"), (
    # Exact retained pairs from ledger results 25853 and 25848.
    ("February 2025", "2025年2月"),
    ("January", "一月"),
    ("October", "十月"),
    ("December 2026", "二零二六年十二月"),
    ("March", "3月"),
    ("May", "五月"),
))
def test_month_localisation_preserves_source_precision(source, target):
    claim = _claim(source, target)
    assert claim.localised_factual_expressions == ((source, target),)
    for changes in (
        {"claim": "The programme changed.", "supporting_excerpt": "The programme changed."},
        {"rendered_assertion_zh_hant_hk": "計劃已修訂。"},
    ):
        with pytest.raises(ValueError, match="equivalent exact claim facts"):
            replace(claim, **changes)


@pytest.mark.parametrize(("source", "target"), (
    ("January", "二月"),
    ("January", "2025年一月"),
    ("January 2025", "一月"),
    ("January 2025", "2026年一月"),
    ("January", "一月一日"),
    ("1 January", "一月"),
    ("January", "1個月"),
    ("1 month", "一月"),
    ("January 2025", "2025年0月"),
    ("January 2025", "2025年13月"),
    ("January 0000", "零年一月"),
    ("Januaryish", "一月"),
    ("Jan", "一月"),
))
def test_month_localisation_rejects_changed_precision_units_or_value(source, target):
    with pytest.raises(ValueError, match="equivalent exact claim facts"):
        _claim(source, target)


@pytest.mark.parametrize(("source", "target", "changes"), (
    ("may", "五月", {"claim": "The programme may change.", "supporting_excerpt": "The programme may change."}),
    ("May", "五月", {"claim": "May the programme change?", "supporting_excerpt": "May the programme change?"}),
    ("March", "三月", {"claim": "March against the reforms.", "supporting_excerpt": "March against the reforms."}),
    ("August", "八月", {"claim": "August issued the statement.", "supporting_excerpt": "August issued the statement."}),
    ("January", "一月", {"claim": "The programme changed in Januaryish.", "supporting_excerpt": "The programme changed in Januaryish."}),
    ("January", "一月", {"claim": "The programme changed in January 2025.", "supporting_excerpt": "The programme changed in January 2025."}),
    ("January", "一月", {"claim": "The programme changed on 21 January.", "supporting_excerpt": "The programme changed on 21 January."}),
    ("January", "一月", {"rendered_assertion_zh_hant_hk": "計劃於十一月修訂。"}),
    ("January", "一月", {"rendered_assertion_zh_hant_hk": "計劃於2025年一月修訂。"}),
    ("January", "一月", {"rendered_assertion_zh_hant_hk": "計劃於一月一日修訂。"}),
))
def test_month_expression_must_bind_a_complete_calendar_fact(source, target, changes):
    claim = _claim("January", "一月")
    text = f"The programme changed in {source}."
    values = {
        "claim": text, "supporting_excerpt": text,
        "rendered_assertion_zh_hant_hk": f"計劃於{target}修訂。",
        **changes,
    }
    with pytest.raises(ValueError, match="equivalent exact claim facts"):
        replace(claim, localised_factual_expressions=((source, target),), **values)


@pytest.mark.parametrize(("source", "target", "claim", "excerpt", "rendered"), (
    (
        "January", "一月",
        "Lifelong Learning Entitlement courses will start in January",
        "And Lifelong Learning Entitlement courses will start in January, enabling people to learn, upskill and retrain across their working lives.",
        "Lifelong Learning Entitlement課程將於一月開課",
    ),
    (
        "October", "十月",
        "A further round of expressions of interest will open in October.",
        "A further round of expressions of interest will open in October.",
        "新一輪意向表達將於十月開放。",
    ),
    (
        "February 2025", "2025年2月",
        "In February 2025, changes were announced to the way apprenticeships are assessed.",
        "In February 2025, changes were announced to the way apprenticeships are assessed.",
        "2025年2月，當局公布學徒制評核方式的改動。",
    ),
))
def test_retained_month_claim_contexts_are_calendar_facts(source, target, claim, excerpt, rendered):
    actual = replace(
        _claim(source, target), claim=claim, supporting_excerpt=excerpt,
        rendered_assertion_zh_hant_hk=rendered,
    )
    assert actual.localised_factual_expressions == ((source, target),)


@pytest.mark.parametrize(('source', 'target'), (
    ('£200 million', '二億英鎊'),
    ('GBP 200 million', '2億英鎊'),
    ('£150,000', '十五萬英鎊'),
    ('£1 billion', '十億英鎊'),
    ('200 million pounds sterling', '200000000英鎊'),
))
def test_pound_localisation_preserves_currency_and_integer_scale(source, target):
    claim = _claim(source, target)
    assert claim.localised_factual_expressions == ((source, target),)
    for changes in (
        {'claim': 'The grant changed.', 'supporting_excerpt': 'The grant changed.'},
        {'rendered_assertion_zh_hant_hk': '津貼已修訂。'},
    ):
        with pytest.raises(ValueError, match='equivalent exact claim facts'):
            replace(claim, **changes)


@pytest.mark.parametrize(('source', 'target'), (
    ('£200 million', '二億港元'),
    ('£200 million', '二千萬英鎊'),
    ('£200 million', '三億英鎊'),
    ('$200 million', '二億英鎊'),
    ('€200 million', '二億英鎊'),
    ('£20,00', '二千英鎊'),
    ('£2.5 million', '二百五十萬英鎊'),
    ('£-200', '二百英鎊'),
    ('£200 million', '2億美元'),
    ('200 million pounds', '二億英鎊'),
    ('200 million Egyptian pounds', '二億英鎊'),
))
def test_pound_localisation_rejects_unsupported_or_changed_money(source, target):
    with pytest.raises(ValueError, match='equivalent exact claim facts'):
        _claim(source, target)


def test_pound_weight_cannot_be_rendered_as_sterling():
    with pytest.raises(ValueError, match='equivalent exact claim facts'):
        replace(_claim('£200 million', '二億英鎊'),
            claim='The shipment weighs 200 million pounds.',
            supporting_excerpt='The shipment weighs 200 million pounds.',
            localised_factual_expressions=(('200 million pounds', '二億英鎊'),))


@pytest.mark.parametrize(('source_text', 'source_key', 'target_text', 'target_key'), (
    ('£200 million', '£200', '二百英鎊', '二百英鎊'),
    ('£200', '£20', '二十英鎊', '二十英鎊'),
    ('£2.5 million', '£2', '二英鎊', '二英鎊'),
    ('£200', '£200', '二千二百英鎊', '二百英鎊'),
    ('£2', '£2', '二十二英鎊', '二英鎊'),
    ('E£200 million', '£200 million', '二億英鎊', '二億英鎊'),
    ('-£200', '£200', '二百英鎊', '二百英鎊'),
    ('- £200', '£200', '二百英鎊', '二百英鎊'),
    ('£2½ million', '£2', '二英鎊', '二英鎊'),
    ('£2/5 million', '£2', '二英鎊', '二英鎊'),
    ('£200', '£200', '- 二百英鎊', '二百英鎊'),
))
def test_pound_fact_binding_rejects_partial_amount_or_scale(
    source_text, source_key, target_text, target_key,
):
    with pytest.raises(ValueError, match='equivalent exact claim facts'):
        replace(_claim('£200', '二百英鎊'),
            claim=f'The grant is {source_text}.',
            supporting_excerpt=f'The grant is {source_text}.',
            rendered_assertion_zh_hant_hk=f'資助額係{target_text}。',
            localised_factual_expressions=((source_key, target_key),))
