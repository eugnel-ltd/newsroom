"""Pure Source-derived rendering inputs, never actor or admission authority.

The caller supplies already authenticated current Source bytes/chronology and
binds VERSION plus these immutable positions into its own consumer identity.
No paid Source view, entity inventory or producer receipt is changed here.
"""
from __future__ import annotations

from datetime import UTC, datetime
import re
from zoneinfo import ZoneInfo

from newsroom.authority.canonical import digest_bytes
from .evidence import bounded_named_entities, _entity_pattern
from .native_story_dates import _unquoted

VERSION = "newsroom.native-source-term-bindings.v1"
VERSION_V2 = "newsroom.native-source-term-bindings.v2"


def _selected(body, selected_text, body_digest, start_byte, end_byte):
    if (type(body) is not str or type(selected_text) is not str or not selected_text
            or type(start_byte) is not int or type(end_byte) is not int):
        raise ValueError("Source term input differs")
    raw = body.encode("utf-8")
    if digest_bytes(raw) != body_digest or not 0 <= start_byte < end_byte <= len(raw):
        raise ValueError("Source term body/range differs")
    try:
        prefix, selected = raw[:start_byte].decode("utf-8"), raw[start_byte:end_byte].decode("utf-8")
    except UnicodeError:
        raise ValueError("Source term range is not UTF-8 aligned") from None
    if selected != selected_text:
        raise ValueError("Source term selected text differs")
    return prefix, selected


def source_term_bindings(
    body: str, selected_text: str, *, body_digest: str, start_byte: int, end_byte: int,
    version: str = VERSION,
) -> tuple[tuple[str, str, int, int], ...]:
    """Return exact literal-preservation terms, not translations/actor proofs."""
    _prefix, selected = _selected(body, selected_text, body_digest, start_byte, end_byte)
    if version not in {VERSION, VERSION_V2}:
        raise ValueError("Source term contract differs")
    names = {(name, "SOURCE_LITERAL_NAME") for name, _kind in
             bounded_named_entities(selected, source_context=body)}
    # Adjuncts require ordinary Source actor syntax, not capitalisation alone.
    actor = re.compile(r"\bby(?:\s+[a-z][a-z-]*){0,4}\s*$|"
                       r"\b(?:charity|organisation|organization|training provider)\s*$|"
                       r"\b(?:chief executive|CEO|director)(?:\s+[a-z][a-z-]*){0,4}\s+(?:of|at)\s*$", re.I)
    adjunct = re.compile(r"(?<![A-Za-z'’])(?:[A-Z][a-z-]+(?:\s+[A-Z][a-z-]+){0,2}\s+Institute|"
                         r"[A-Z][a-z-]+['’]s\s+[A-Z][a-z-]+(?:\s+[A-Z][a-z-]+){0,2})(?![A-Za-z'’])")
    for match in adjunct.finditer(body):
        if actor.search(body[max(0, match.start() - 80):match.start()]):
            names.add((match.group(), "SOURCE_LITERAL_ORGANISATION_NAME"))
    # Literal acronym preservation does not establish an expanded long form.
    # Corroborated narrative tokens exclude isolated/all-capital headings.
    for match in re.finditer(r"(?<![A-Za-z0-9_./-])[A-Z]{4,10}(?![A-Za-z0-9_]|[./-][A-Za-z0-9_])", selected):
        literal = match.group()
        line = selected[selected.rfind("\n", 0, match.start()) + 1:selected.find("\n", match.end()) if "\n" in selected[match.end():] else len(selected)]
        if re.search(r"[a-z]{2}", line) and len(re.findall(_entity_pattern(literal), body)) >= 2:
            names.add((literal, "SOURCE_LITERAL_ACRONYM"))
    if version == VERSION_V2:
        # Exact contact/opaque label preservation is not actor or age authority.
        email = re.compile(r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~-])"
            r"[A-Za-z0-9][A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]*@"
            r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,63}"
            r"(?![A-Za-z0-9_-])")
        for pattern, kind in (
            (email, "SOURCE_LITERAL_EMAIL"),
            (re.compile(r"(?<![A-Za-z0-9_-])post-[0-9]{1,4}(?![A-Za-z0-9_-])", re.I), "SOURCE_LITERAL_LABEL"),
            (re.compile(r"(?<![A-Za-z0-9_./-])[A-Z][a-z]+[A-Z][A-Za-z]{0,8}(?![A-Za-z0-9_./-])"), "SOURCE_LITERAL_LABEL"),
        ):
            for match in pattern.finditer(selected):
                if len(match.group()) <= 80:
                    names.add((match.group(), kind))
    positions = []
    for name, kind in names:
        for match in re.finditer(_entity_pattern(name), selected):
            positions.append((name, kind, start_byte + len(selected[:match.start()].encode("utf-8")),
                              start_byte + len(selected[:match.end()].encode("utf-8"))))
    # Prefer an existing recognised name over an overlapping adjunct.
    result = []
    for term in sorted(positions, key=lambda item: (item[2], -(item[3] - item[2]), item[1])):
        if not result or term[2] >= result[-1][3]:
            result.append(term)
    return tuple(result)


def derive_relative_year(
    body: str, selected_text: str, *, body_digest: str, start_byte: int, end_byte: int,
    publication_time: str, source_updated_time: str,
) -> tuple[str, str, int, int, int]:
    """Bind one unquoted calendar 'next year' to publisher chronology only."""
    prefix, selected = _selected(body, selected_text, body_digest, start_byte, end_byte)
    if (re.search(r"\b(?:if|unless|would|could|might)\b|"
                  r"\b(?:financial|fiscal|academic|school|tax)\s+year\b|"
                  r"\b[1-9][0-9]{3}\b|\b(?:within|over|during)\s+(?:the\s+)?next year\b", selected, re.I)):
        raise ValueError("Source relative year is conditional or ambiguous")
    tokens = tuple(re.finditer(r"\bnext year\b", selected, re.I))
    if len(tokens) != 1:
        raise ValueError("Source relative year token differs")
    dates = []
    for value in (publication_time, source_updated_time):
        if type(value) is not str:
            raise ValueError("Source relative year chronology is absent")
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if instant.tzinfo is None:
            raise ValueError("Source relative year chronology is naive")
        dates.extend((instant.year, instant.astimezone(UTC).year,
                      instant.astimezone(ZoneInfo("Europe/London")).year))
    if len(set(dates)) != 1 or not 1 <= dates[0] < 9999:
        raise ValueError("Source relative year chronology conflicts")
    token = tokens[0]
    _unquoted(body, len(prefix) + token.start(), len(prefix) + token.end())
    return (token.group(), f"{dates[0] + 1}年",
            start_byte + len(selected[:token.start()].encode("utf-8")),
            start_byte + len(selected[:token.end()].encode("utf-8")), dates[0])
