"""Equivalent Chinese date clocks preserve source precision and values."""
from dataclasses import replace
import pytest
from newsroom.tests.test_factual_localisation import _claim

@pytest.mark.parametrize(("source", "target"), (
    ("1 October 2026 at 06:00", "2026年10月1日06:00"),
    ("1 October 2026 at 00:05", "2026年10月1日00:05"),
    ("1 October 2026 at 23:59", "2026年10月1日23:59"),
    ("29 February 2024 at 06:00", "2024年2月29日06:00"),
    ("1 October 2026 at 00:00", "2026年10月1日上午12:00"),
    ("1 October 2026 at 12:00", "2026年10月1日下午12:00"),
    ("1 October 2026 at 18:00", "2026年10月1日下午6:00"),
    ("1 October 2026 at 06:00", "2026年10月1日上午6:00"),
))
def test_colon_clock_preserves_exact_date_and_minute(source, target):
    claim = _claim(source, target)
    assert claim.localised_factual_expressions == ((source, target),)
    with pytest.raises(ValueError, match="equivalent exact claim facts"):
        replace(claim, rendered_assertion_zh_hant_hk="計劃已修訂。")

@pytest.mark.parametrize("target", (
    "2026年10月1日06:01", "2026年10月2日06:00", "2026年11月1日06:00",
    "2027年10月1日06:00", "10月1日06:00", "2026年10月1日",
    "2026年10月1日24:00", "2026年10月1日06:60", "2026年10月1日06:00:00",
))
def test_colon_clock_rejects_changed_fact_or_precision(target):
    with pytest.raises(ValueError, match="equivalent exact claim facts"):
        _claim("1 October 2026 at 06:00", target)

@pytest.mark.parametrize(("source", "target"), (
    ("1 October 2026 at 18:00", "2026年10月1日上午18:00"),
    ("1 October 2026 at 00:05", "2026年10月1日上午00:05"),
    ("1 October 2026 at 13:05", "2026年10月1日下午13:05"),
    ("1 October 2026 at 12:05", "2026年10月1日下午00:05"),
))
def test_colon_clock_rejects_impossible_explicit_half_day_hour(source, target):
    with pytest.raises(ValueError, match="equivalent exact claim facts"):
        _claim(source, target)
