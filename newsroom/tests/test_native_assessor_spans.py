"""Concise sentence selection without changing acquired source bytes."""

from dataclasses import replace

import pytest

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.control_plane.native_assessor_references import (
    MAX_SEGMENTS, VERSION, SourceReferenceError, _claim_entities,
    build_source_view, materialise,
)
from newsroom.control_plane.native_assessor_spans import (
    PARTITION_VERSION, PARTITION_VERSION_V1, PartitionedSourceView, _chunks, build_lossless_source_view,
)


def _range(first="S1L1", last=None):
    return {"first_span_id": first, "last_span_id": last or first}


def _wire():
    return {"package": {
        "substantive_claim_indexes": [0],
        "governed_claims": [{
            "claim_role": "HEADLINE", "claim_range": _range(),
            "support_range": _range(last="S1L2"),
            "rendered_fragments": ["新安排明日開始。"],
            "status": "CONFIRMED_FACT",
            "semantic_relation": {
                "source_modality": "ASSERTED", "rendered_modality": "ASSERTED",
                "source_polarity": "AFFIRMED", "rendered_polarity": "AFFIRMED",
                "relation": "SEMANTICALLY_EQUIVALENT",
            },
            "localised_factual_expressions": [], "quotations": [],
            "certainty": "CONFIRMED", "originality_basis": "FACTUAL_REWRITE_REQUIRED",
            "originality_policy_version": "newsroom.cont-originality.v3",
            "admitted_use": "PUBLICATION_EVIDENCE",
            "policy_version": "newsroom.governed-claim.v7",
        }],
        "qualification_evidence": [], "selection_rationale": "Both clauses retained.",
        "geography": [], "categories": [], "explicit_exclusions": [],
    }}


def test_single_line_supports_concise_exact_claim_and_full_body_reconstruction():
    body = "new arrangements begin tomorrow. Existing restrictions remain unchanged."
    view = build_lossless_source_view((body,), ("SOURCE-1",))
    assert len(view.segments) == 2
    assert view.resolve_range(_range()) == (
        "new arrangements begin tomorrow. ", 0, "SOURCE-1",
    )
    assert "".join(segment.text for segment in view.segments).encode() == body.encode()
    assert view.body_digests == (digest_bytes(body.encode()),)


def test_partition_marker_adds_no_storage_or_historical_manifest_fields():
    body = "one sentence. Another sentence."
    legacy = build_source_view((body,), ("SOURCE-1",))
    view = build_lossless_source_view((body,), ("SOURCE-1",))
    assert isinstance(view, PartitionedSourceView)
    assert view.partition_version == PARTITION_VERSION
    assert not hasattr(view, "__dict__")
    assert not hasattr(legacy, "partition_version")
    old = build_lossless_source_view((body,), ("SOURCE-1",), version=PARTITION_VERSION_V1)
    assert set(old.manifest) == set(legacy.manifest)
    assert old.partition_version == PARTITION_VERSION_V1
    assert set(view.manifest) == set(legacy.manifest) | {"entity_policy_version"}
    assert replace(view).partition_version == PARTITION_VERSION


def test_adjacent_range_retains_negative_context_and_unchanged_v1_materialisation():
    body = "new arrangements begin tomorrow. Existing restrictions remain unchanged."
    view = build_lossless_source_view((body,), ("SOURCE-1",))
    wire = _wire()
    package, receipt = materialise(wire, view, "request-id")
    claim = package["package"]["governed_claims"][0]
    assert claim["claim"] == "new arrangements begin tomorrow. "
    assert claim["supporting_excerpt"] == body
    assert package["package"]["qualification_evidence"] == []
    assert receipt["version"] == view.manifest["version"] == VERSION
    assert PARTITION_VERSION != VERSION
    assert view.manifest_digest == digest_canonical(view.manifest)
    assert receipt["materialised_text"] == canonical_json_bytes(package).decode()
    assert materialise(wire, view, "request-id") == (package, receipt)
    with pytest.raises(SourceReferenceError, match="source view or acquired bytes differ"):
        materialise(wire, replace(view, passages=(body + " changed",)), "request-id")


def test_multibyte_emoji_crlf_and_unicode_paragraphs_preserve_all_offsets():
    body = "  香港👩🏽‍💻首句。\t第二句。 \r\n\r\n末段🧪。下一句！\u2029"
    view = build_lossless_source_view((body,), ("HK-1",))
    assert [item.text for item in view.segments] == [
        "  香港👩🏽‍💻首句。\t", "第二句。 \r\n", "\r\n", "末段🧪。", "下一句！\u2029",
    ]
    encoded = body.encode()
    offset = 0
    for segment in view.segments:
        raw = segment.text.encode()
        assert segment.start_byte == offset
        assert segment.end_byte == offset + len(raw)
        assert encoded[segment.start_byte:segment.end_byte] == raw
        assert segment.digest == digest_bytes(raw)
        expected = encoded[segment.start_byte:segment.content_end_byte].decode()
        assert view.resolve_range(_range(segment.span_id))[0] == expected
        offset = segment.end_byte
    assert offset == len(encoded)
    assert view.resolve_range(_range("S1L1", "S1L5"))[0] == body[:-1]


@pytest.mark.parametrize("first", [
    "Dr. Alice Smith said the U.K. figure was 3.14 on 14.10.2026 at 10:30 a.m. today.",
    "The deadline is Jan. 14 and the date is 14. 10. 2026.",
    "Visit https://example.test/v1.2/next?x=3.14 for details.",
    "Visit www.example.test/a.b for details.",
    "He said ‘the arrangements will begin.’",
    "Wait... the arrangements are not final!",
])
def test_abbreviations_decimals_dates_urls_and_quotation_closers_are_conservative(first):
    body = first + "  Next sentence."
    view = build_lossless_source_view((body,), ("SOURCE-1",))
    assert [item.text for item in view.segments] == [first + "  ", "Next sentence."]


def test_recognised_entity_internal_period_is_protected():
    text = "Example. Name Department announced changes. Next sentence."
    segment = build_source_view((text,), ("SOURCE-1",)).segments[0]
    segment = replace(segment, entities=(("Example. Name Department", "ORGANISATION"),))
    chunks = list(_chunks(segment))
    assert chunks == [
        "Example. Name Department announced changes. ",
        "Next sentence.",
    ]


def test_source_and_segment_entity_order_matches_existing_recogniser():
    body = (
        "Alice Smith said yes. Bob Jones said yes. Alice Smith said no.\n"
        "The Department for Education announced new arrangements.\n"
    )
    original = build_source_view((body,), ("SOURCE-1",))
    view = build_lossless_source_view((body,), ("SOURCE-1",))
    assert view.source_entities == original.source_entities
    assert [item.entities for item in view.segments] == [
        _claim_entities(item.text, body) for item in view.segments
    ]
    assert view.segments[0].entities == (("Alice Smith", "PERSON"),)
    assert view.segments[2].entities == (("Alice Smith", "PERSON"),)


def test_legacy_inventory_does_not_invent_contextual_entity_occurrences():
    body = "Home Secretary said yes. The Home Secretary approved changes."
    view = build_lossless_source_view((body,), ("SOURCE-1",))
    assert len(view.segments) == 1
    assert view.segments[0].entities == (("Home Secretary", "PERSON"),) * 2
    wire = _wire()
    claim = wire["package"]["governed_claims"][0]
    claim["claim_range"] = claim["support_range"] = _range()
    claim["rendered_fragments"] = ["", " 表示同意；", " 批准新安排。"]
    package, receipt = materialise(wire, view, "request-id")
    assert package["package"]["governed_claims"][0]["claim"] == body
    assert receipt["claim_entity_order"] == [[["Home Secretary", "PERSON"]] * 2]


@pytest.mark.parametrize("body", [
    "Home Secretary said yes. The Home Secretary approved changes. Other arrangements remain unchanged.",
    "The Home Secretary approved changes. Home Secretary said yes. Other arrangements remain unchanged.",
    "Alice Smith said yes. Alice Smith said no. Other arrangements remain unchanged.",
    "Alice Smith said yes. Minister Alice Smith approved changes. Minister Alice Smith said yes.",
    "Special Agency Authority (SAA) confirmed it. SAA changed its arrangements.",
    "張小明表示同意。張小明批准新安排。其他安排不變。",
    "Home Secretary said yes.\r\nThe Home Secretary approved changes. Other arrangements remain unchanged.",
])
def test_every_contiguous_range_advertises_authoritative_entity_order(body):
    view = build_lossless_source_view((body,), ("SOURCE-1",))
    for first in range(len(view.segments)):
        for last in range(first, len(view.segments)):
            selected = view.segments[first:last + 1]
            text = view.resolve_range(_range(selected[0].span_id, selected[-1].span_id))[0]
            advertised = tuple(entity for segment in selected for entity in segment.entities)
            assert advertised == _claim_entities(text, body)
            wire = _wire()
            claim = wire["package"]["governed_claims"][0]
            claim["claim_range"] = claim["support_range"] = _range(
                selected[0].span_id, selected[-1].span_id,
            )
            claim["rendered_fragments"] = [""] * (len(advertised) + 1)
            package, receipt = materialise(wire, view, "request-id")
            assert package["package"]["governed_claims"][0]["claim"] == text
            assert receipt["claim_entity_order"] == [[list(entity) for entity in advertised]]
    assert "".join(segment.text for segment in view.segments) == body


@pytest.mark.parametrize("sentence", ["他說：「新安排明日開始。」", "首句。）", "首句。』】"])
def test_terminal_cjk_closers_remain_attached_to_their_sentence(sentence):
    terminal = build_lossless_source_view((sentence,), ("SOURCE-1",))
    assert [segment.text for segment in terminal.segments] == [sentence]
    followed = build_lossless_source_view((sentence + "下一句。",), ("SOURCE-1",))
    assert [segment.text for segment in followed.segments] == [sentence, "下一句。"]


def test_source_declared_acronym_remains_bound_to_full_body_context():
    body = "Special Agency Authority (SAA) confirmed it. SAA changed its arrangements."
    view = build_lossless_source_view((body,), ("SOURCE-1",))
    assert ("SAA", "OFFICIAL_TERM") in view.segments[1].entities
    assert view.segments[1].entities == _claim_entities(view.segments[1].text, body)


def test_literal_csv_rows_remain_atomic_and_historical_builder_is_unchanged():
    row = 'Row 2: A="Alice Smith"; B="One sentence. Another sentence."\r\n'
    body = "First sentence. Second sentence.\n" + row
    original = build_source_view((body,), ("SOURCE-1",))
    view = build_lossless_source_view((body,), ("SOURCE-1",))
    assert [item.text for item in original.segments] == [
        "First sentence. Second sentence.\n", row,
    ]
    assert view.segments[-1].text == row
    assert view.resolve_range(_range("S1L3"))[0] == row[:-2]
    assert original == build_source_view((body,), ("SOURCE-1",))
    assert original.manifest_digest != view.manifest_digest


@pytest.mark.parametrize(("passages", "source_ids"), [
    ((), ()), (("",), ("SOURCE-1",)), (("body",), ("",)),
    (("body", "other"), ("same", "same")), (("body",), ("A", "B")),
    (("x" * 1_048_577,), ("SOURCE-1",)),
])
def test_legacy_input_bounds_still_fail_closed(passages, source_ids):
    with pytest.raises(SourceReferenceError):
        build_lossless_source_view(passages, source_ids)


def test_segment_count_bound_includes_new_sentence_splits(monkeypatch):
    from newsroom.control_plane import native_assessor_spans as spans
    assert spans.MAX_SEGMENTS == MAX_SEGMENTS == 8192
    monkeypatch.setattr(spans, "MAX_SEGMENTS", 4)
    body = "sentence. " * 4
    assert len(build_lossless_source_view((body,), ("SOURCE-1",)).segments) == 4
    with pytest.raises(SourceReferenceError, match="source segment count exceeds bound"):
        build_lossless_source_view((body + "another.",), ("SOURCE-1",))


def test_blank_source_lines_and_invalid_cross_source_boundaries():
    view = build_lossless_source_view(("\r\n\t\u2029", "first. second."), ("A", "B"))
    assert [item.text for item in view.segments] == ["\r\n", "\t\u2029", "first. ", "second."]
    for reference in (
        _range("S1L2", "S2L1"), _range("S2L2", "S2L1"),
        _range("S2L999"), _range("S0L1"),
    ):
        with pytest.raises(SourceReferenceError):
            view.resolve_range(reference)
