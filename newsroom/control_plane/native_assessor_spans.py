"""Conservative, lossless partitioning of exact assessor source bytes."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import replace
from itertools import pairwise

from newsroom.authority.canonical import digest_bytes, digest_canonical
from newsroom.control_plane.evidence import _entity_pattern, NAMED_ENTITY_POLICY_VERSION, NAMED_ENTITY_POLICY_VERSION_V15

from .native_assessor_references import (
    MAX_SEGMENTS, SourceReferenceError, SourceSegment, SourceView,
    _claim_entities, _manifest, build_source_view,
)

PARTITION_VERSION_V1 = "newsroom.native-assessor-spans.v1"
PARTITION_VERSION = "newsroom.native-assessor-spans.v2"
_CLOSERS = r"[\"'’”）)\]】」』]*+"
_BOUNDARY = re.compile(
    rf"(?P<latin>[.!?]){_CLOSERS}[ \t]+(?=\S)|"
    rf"(?P<cjk>[。！？]){_CLOSERS}[ \t]*(?=\S)"
)
_ABBREVIATION = re.compile(
    r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|a\.m|p\.m|"
    r"No|Fig|Art|para|cl|Vol|Inc|Ltd|Co|Jan|Feb|Mar|Apr|Jun|Jul|Aug|"
    r"Sep|Sept|Oct|Nov|Dec|[A-Za-z]|(?:[A-Za-z]\.)+[A-Za-z])\.$",
    re.IGNORECASE,
)


class PartitionedSourceView(SourceView):
    """Identify the partition policy without adding fields to the v1 manifest."""

    __slots__ = ()

    @property
    def partition_version(self) -> str:
        return PARTITION_VERSION_V1 if self.entity_policy_version == NAMED_ENTITY_POLICY_VERSION_V15 else PARTITION_VERSION


def _chunks(segment: SourceSegment) -> Iterator[str]:
    text = segment.text
    # Literal CSV rows remain one evidence unit, including quoted cell prose.
    if re.match(r"Row [1-9][0-9]*: ", text):
        yield text
        return
    protected = sorted(
        [match.span() for name, _ in dict.fromkeys(segment.entities)
         for match in re.finditer(_entity_pattern(name), text)]
        + [match.span() for match in re.finditer(r"(?:https?://|www\.)\S+", text)]
    )
    cuts = [0]
    protected_index = protected_end = 0
    for match in _BOUNDARY.finditer(text):
        stop = match.end("latin" if match.group("latin") else "cjk")
        while protected_index < len(protected) and protected[protected_index][0] < stop:
            protected_end = max(protected_end, protected[protected_index][1])
            protected_index += 1
        if stop < protected_end:
            continue
        if match.group("latin") == ".":
            token = text[max(0, stop - 64):stop].rsplit(None, 1)[-1].lstrip("\"'‘“([")
            if (_ABBREVIATION.fullmatch(token) or token.endswith("..")
                    or (len(token) > 1 and token[-2].isdigit()
                        and text[match.end()].isdigit())):
                continue
        if len(cuts) >= MAX_SEGMENTS:
            raise SourceReferenceError("source segment count exceeds bound")
        cuts.append(match.end())
    cuts.append(len(text))
    for first, last in pairwise(cuts):
        yield text[first:last]


def _range_closed_chunks(line: SourceSegment, body: str, *, policy_version: str = NAMED_ENTITY_POLICY_VERSION_V15) -> tuple[tuple[str, tuple], ...]:
    """Keep a context-dependent name's occurrences in one selectable unit."""
    texts = tuple(_chunks(line))
    if len(texts) == 1:
        return ((line.text, line.entities),)
    chunks = tuple((text, _claim_entities(text, body, policy_version=policy_version)) for text in texts)
    occurrences = sorted(
        [(match.start(), match.end(), name, kind)
         for name, kind in dict.fromkeys(line.entities)
         for match in re.finditer(_entity_pattern(name), line.text)],
        key=lambda item: (item[0], -(item[1] - item[0])),
    )
    selected = []
    for item in occurrences:
        if not selected or item[0] >= selected[-1][1]:
            selected.append(item)
    offset = 0
    sensitive = set()
    for text, actual in chunks:
        end = offset + len(text)
        expected = tuple((name, kind) for first, _, name, kind in selected if offset <= first < end)
        if actual != expected:
            counts = Counter(actual)
            counts.subtract(expected)
            sensitive.update(name for (name, _), count in counts.items() if count)
            if not any(counts.values()):
                sensitive.update(name for name, _ in (*actual, *expected))
        offset = end
    if not sensitive:
        return chunks
    neighbourhoods = []
    for name in sensitive:
        matches = tuple(re.finditer(_entity_pattern(name), line.text))
        neighbourhoods.append((matches[0].start(), matches[-1].end()))
    cuts = [0]
    offset = 0
    for text in texts:
        offset += len(text)
        if not any(first < offset < last for first, last in neighbourhoods):
            cuts.append(offset)
    merged = tuple((line.text[first:last], _claim_entities(line.text[first:last], body, policy_version=policy_version))
                   for first, last in pairwise(cuts))
    # Each surviving unit must expose exactly its full-line occurrence slice.
    # Ambiguous overlap retains the original physical-line contract instead.
    offset = 0
    for text, actual in merged:
        end = offset + len(text)
        expected = tuple((name, kind) for first, _, name, kind in selected if offset <= first < end)
        if actual != expected:
            return ((line.text, line.entities),)
        offset = end
    return merged


def build_lossless_source_view(passages: tuple[str, ...], source_ids: tuple[str, ...], *, version: str = PARTITION_VERSION) -> SourceView:
    """Refine the legacy view without changing its bytes or v1 range proof."""
    if version not in {PARTITION_VERSION_V1, PARTITION_VERSION}:
        raise SourceReferenceError("unsupported source partition version")
    policy = NAMED_ENTITY_POLICY_VERSION_V15 if version == PARTITION_VERSION_V1 else NAMED_ENTITY_POLICY_VERSION
    original = build_source_view(passages, source_ids, policy_version=policy)
    segments = []
    for passage_index, body in enumerate(passages):
        ordinal = 0
        for line in (item for item in original.segments if item.passage_index == passage_index):
            offset = line.start_byte
            for text, entities in _range_closed_chunks(line, body, policy_version=policy):
                if len(segments) >= MAX_SEGMENTS:
                    raise SourceReferenceError("source segment count exceeds bound")
                ordinal += 1
                raw = text.encode("utf-8")
                end = offset + len(raw)
                segments.append(replace(
                    line, span_id=f"S{passage_index + 1}L{ordinal}", ordinal=ordinal,
                    text=text, start_byte=offset, end_byte=end,
                    content_end_byte=min(end, line.content_end_byte),
                    digest=digest_bytes(raw), entities=entities,
                ))
                offset = end
            if offset != line.end_byte:
                raise SourceReferenceError("source segmentation lost exact bytes")
    exact = tuple(segments)
    return PartitionedSourceView(
        passages, source_ids, exact, original.body_digests, original.source_entities,
        digest_canonical(_manifest(source_ids, original.body_digests, exact, policy_version=policy)), policy,
    )
