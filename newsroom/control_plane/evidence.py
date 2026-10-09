"""CONT-001 Evidence Package for unpublished staging."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane.editorial import StoryCandidateRecord
from newsroom.control_plane.governed_context import GovernedContext

EVID_012_POLICY_VERSION = "newsroom.evid-012.v7"
GOVERNED_CLAIM_POLICY_VERSION = "newsroom.governed-claim.v7"
EVIDENCE_GATE_POLICY_VERSION = "newsroom.evidence-gates.v2"
GOVERNED_INPUT_SCHEMA_VERSION = "newsroom.governed-input.v10"
EVIDENCE_APPROVAL_POLICY_VERSION = "newsroom.evidence-approval.v8"
EVIDENCE_APPROVAL_PRINCIPAL = "HERMES_EVIDENCE_CONTROLLER"
ORIGINALITY_POLICY_VERSION = "newsroom.cont-originality.v3"
NAMED_ENTITY_POLICY_VERSION_V15 = "newsroom.named-entity.v15"
NAMED_ENTITY_POLICY_VERSION = "newsroom.named-entity.v16"
FACTUAL_LOCALISATION_POLICY_VERSION = "newsroom.factual-localisation.v2"

_SOURCE_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "record_type",
        "candidate_id",
        "base_package_digest",
        "status",
        "source_id",
        "canonical_url",
        "publisher",
        "responsible_body",
        "source_type",
        "authority_class",
        "publication_time",
        "retrieval_time",
        "geography",
        "language",
        "extraction_status",
        "rights_decision_id",
        "originating_report_id",
        "originating_artefact_digest",
        "dependency_evidence_ids",
    }
)


def _source_body_provenance_is_bound(record):
    if set(record)==_SOURCE_RECORD_FIELDS:
        return True
    if set(record)!=_SOURCE_RECORD_FIELDS|{'body_provenance'}:
        return False
    proof=record.get('body_provenance')
    return (type(proof)is dict and set(proof)=={'version','kind','body_digest',
        'acquisition_receipt_digest','transport_evidence_digest'}
        and proof['version']=='newsroom.acquisition-body-provenance.v1'
        and type(proof['kind'])is str
        and proof['kind']in {'GOVUK_CONTENT_API_PAGE_TEXT','GOVUK_DECLARED_ASSET_TEXT'}
        and proof['body_digest']==record.get('originating_artefact_digest')
        and proof['acquisition_receipt_digest']==record.get('record_id')
        and all(type(proof[key])is str and re.fullmatch(r'sha256:[0-9a-f]{64}',proof[key])
            for key in ('body_digest','acquisition_receipt_digest','transport_evidence_digest')))
_SOURCE_AUTHORITY_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "record_type",
        "candidate_id",
        "base_package_digest",
        "status",
        "source_id",
        "decision",
        "authority_class",
        "authority_scope",
        "governed_claim_id",
        "claim_digest",
    }
)
_RIGHTS_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "record_type",
        "candidate_id",
        "base_package_digest",
        "status",
        "source_id",
        "decision",
        "permitted_use",
    }
)
_DEPENDENCY_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "record_type",
        "candidate_id",
        "base_package_digest",
        "status",
        "source_id",
        "dependency_status",
        "evidential_origin_id",
        "originating_report_id",
    }
)
_QUALIFICATION_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "record_type",
        "candidate_id",
        "base_package_digest",
        "status",
        "governed_claim_id",
        "test",
        "test_evidence",
        "policy_version",
        "evidence_span_digest",
        "source_record_ids",
    }
)
_NAMED_ENTITY_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "record_type",
        "candidate_id",
        "base_package_digest",
        "status",
        "governed_claim_id",
        "text",
        "rendered_text",
        "entity_type",
        "canonical_entity_id",
        "policy_version",
        "evidence_span_digest",
        "rendered_span_digest",
        "source_record_ids",
    }
)
_SEMANTIC_RELATION_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "record_type",
        "candidate_id",
        "base_package_digest",
        "status",
        "governed_claim_id",
        "source_modality",
        "rendered_modality",
        "source_polarity",
        "rendered_polarity",
        "relation",
        "claim_digest",
        "rendered_assertion_digest",
    }
)
_RECORD_FIELDS_BY_TYPE = {
    "SOURCE_RECORD": _SOURCE_RECORD_FIELDS,
    "SOURCE_AUTHORITY_DECISION": _SOURCE_AUTHORITY_RECORD_FIELDS,
    "RIGHTS_DECISION": _RIGHTS_RECORD_FIELDS,
    "DEPENDENCY_EVIDENCE": _DEPENDENCY_RECORD_FIELDS,
    "QUALIFICATION_EVIDENCE": _QUALIFICATION_RECORD_FIELDS,
    "NAMED_ENTITY_EVIDENCE": _NAMED_ENTITY_RECORD_FIELDS,
    "SEMANTIC_RELATION_EVIDENCE": _SEMANTIC_RELATION_RECORD_FIELDS,
}
_PUBLICATION_EVIDENCE_SOURCE_TYPES = frozenset(
    {
        "PRIMARY_OFFICIAL",
        "ESTABLISHED_NEWS_ORGANISATION",
        "LOCAL_SPECIALIST_PUBLICATION",
    }
)
_ENGLISH_MONTHS = {
    month.casefold(): index
    for index, month in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}
_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "兩": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_OWNER_APPROVED_ENTITY_REGISTRY = {
    "里斯本": "PLACE",
    "里約熱內盧": "PLACE",
    "后海灣": "PLACE",
    "干邑": "PLACE",
    "干德道": "PLACE",
    "干諾道中": "PLACE",
    "香港": "PLACE",
    "中西區": "PLACE",
    "灣仔": "PLACE",
    "東區": "PLACE",
    "南區": "PLACE",
    "油尖旺": "PLACE",
    "深水埗": "PLACE",
    "九龍城": "PLACE",
    "黃大仙": "PLACE",
    "觀塘": "PLACE",
    "葵青": "PLACE",
    "荃灣": "PLACE",
    "屯門": "PLACE",
    "元朗": "PLACE",
    "北區": "PLACE",
    "大埔": "PLACE",
    "沙田": "PLACE",
    "北角": "PLACE",
    "太子": "PLACE",
    "旺角": "PLACE",
    "尖沙咀": "PLACE",
    "銅鑼灣": "PLACE",
    "佐敦": "PLACE",
    "西貢": "PLACE",
    "離島": "PLACE",
    "九龍": "PLACE",
    "倫敦": "PLACE",
    "深圳": "PLACE",
    "北京": "PLACE",
    "上海": "PLACE",
    "澳門": "PLACE",
    "廣州": "PLACE",
    "巴黎": "PLACE",
    "英國": "PLACE",
    "UK": "PLACE",
    "Hong Kong": "PLACE",
    "香港政府": "ORGANISATION",
    "運輸署": "ORGANISATION",
    "教育局": "ORGANISATION",
    "醫院管理局": "ORGANISATION",
    "Home Office": "ORGANISATION",
    "Hong Kong Authority": "ORGANISATION",
    "Hong Kong Monetary Authority": "ORGANISATION",
    "Housing Authority": "ORGANISATION",
    "EUSS": "OFFICIAL_TERM",
    "Universal Credit": "OFFICIAL_TERM",
    "British National (Overseas)": "OFFICIAL_TERM",
    "ATAS": "OFFICIAL_TERM",
    "eVisa": "OFFICIAL_TERM",
}
_ENGLISH_ORGANISATION_ACTION_WORDS = frozenset(
    {
        "announces",
        "backs",
        "confirms",
        "creates",
        "expands",
        "funds",
        "introduces",
        "launches",
        "new",
        "opens",
        "plans",
        "proposes",
        "says",
        "scraps",
        "supports",
        "unveils",
    }
)
_ENGLISH_ORGANISATION = re.compile(
    r"\b(?:Department|Ministry|Office)\s+(?:for|of)\s+(?:the\s+)?"
    r"[A-Z][A-Za-z&.-]+(?:\s+(?:and|of|for)\s+(?:the\s+)?"
    r"[A-Z][A-Za-z&.-]+)*\b(?!\s+[A-Z])|"
    r"\bNHS(?:\s+(?:England|Scotland|Wales))?\b|"
    r"\bTransport\s+for\s+[A-Z][A-Za-z&.-]+\b|"
    r"\b(?:[A-Z][A-Za-z&.-]+\s+){1,3}"
    r"(?:Authority|Directorate|Department|Ministry|Agency|Council|Commission|"
    r"Service|Police|University|Hospital|Bank)\b"
)
_ENGLISH_UNIVERSITY_OF = re.compile(
    r"(?<![A-Za-z0-9_.-])(University of [A-Z][a-z]+)"
    r"(?![A-Za-z0-9_-]|\.[A-Za-z0-9_]|/[A-Za-z0-9_]|"
    r"[ \t]+[A-Z][A-Za-z-]*\b)"
)
_LITERAL_CSV_PREAMBLE = (
    "Published CSV cells: Row and column identify each literal text cell. "
    "Whitespace, empty fields, quoted newlines and formula-like text are preserved; "
    "nothing is executed. No header or numeric types are inferred."
)
_LITERAL_CSV_NIL = frozenset({"nil return", "n/a", "none", "not applicable"})
# A rejected prefix such as "The Department" must not hide an overlapping
# complete name such as "Department for Education". Selection below still
# retains only the longest non-overlapping, independently checked spans.
_ENGLISH_ORGANISATION_CANDIDATES = re.compile(
    rf"(?=({_ENGLISH_ORGANISATION.pattern}))"
)
_ENGLISH_OFFICIAL_TERM_V15 = re.compile(
    r"\b(?:[A-Z][A-Za-z-]+(?:\s+(?:and|of|the|for|[A-Z][A-Za-z-]+)){1,7}"
    r"\s+Act|(?:[A-Z][A-Za-z-]+\s+){1,5}"
    r"(?:Authorisation|Credit|Scheme|Programme|Benefit|Visa|Permit|Status))\b"
)
_ENGLISH_OFFICIAL_TERM = re.compile(
    r"\b(?:[A-Z][A-Za-z-]+(?:['’]s|['’])?(?:\s+(?:and|of|the|for|[A-Z][A-Za-z-]+(?:['’]s|['’])?)){1,7}"
    r"\s+Act|(?:[A-Z][A-Za-z-]+\s+){1,5}"
    r"(?:Authorisation|Credit|Scheme|Programme|Benefit|Visa|Permit|Status))\b"
)
_ENGLISH_OFFICIAL_REFERENCE = re.compile(
    r"(?:\b(?:General Grounds for Refusal|Part Suitability)\b|"
    r"\bAppendix\s+(?:[A-Z]|[A-Z][a-z]+"
    r"(?:\s+(?:and|of|the|for|[A-Z][a-z]+)){0,6})\b|"
    r"\b[A-Z]{1,4}\([A-Z]{2,4}\)\d+(?:\.\d+)+\b)"
)
_BOUNDED_OFFICIAL_ABBREVIATIONS = frozenset({"DWP", "ECAA", "EPA", "ETA"})
_DECLARED_ACRONYM = re.compile(
    r"\b([A-Za-z]+(?:[ \t-]+[A-Za-z]+){1,9})[ \t]+"
    r"\(([A-Z][A-Za-z]{1,9})\)"
)


def _source_declared_acronyms(source_context: str) -> frozenset[str]:
    """Recognise literal declarations, not arbitrary capitalised prose."""
    declared = set()
    for match in _DECLARED_ACRONYM.finditer(source_context):
        long_form, acronym = match.groups()
        if sum(letter.isupper() for letter in acronym) < 2:
            continue
        words = re.findall(r"[A-Za-z]+", long_form)
        # The source may place ordinary prose before the declared long form.
        for start in range(len(words) - 1):
            expansion = words[start:]
            if len(" ".join(expansion)) > 80:
                continue
            initials = {
                "".join(word[0] for word in expansion).upper(),
                "".join(word[0] for word in expansion
                        if word.lower() not in {"and", "of", "the", "for"}).upper(),
            }
            if acronym.upper() in initials or (
                acronym.endswith("s") and expansion[-1].endswith("s")
                and acronym[:-1].upper() in initials
            ):
                declared.add(acronym)
                break
    return frozenset(declared)


_SOURCE_BOUND_ROUTE_TERM = re.compile(
    r"\b([A-Z]{2,5}(?:\s+[A-Z][A-Za-z-]+){1,5})(?=\s+route\b)"
)
_SOURCE_BOUND_TECHNICAL_FRAMEWORK = re.compile(
    r"\b([A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+){0,4}\s+Framework"
    r"\s+of\s+Reference\s+for\s+[A-Z][A-Za-z-]+)\b"
)
_SOURCE_BOUND_LEVEL_CODE = re.compile(r"\blevel\s+([A-C][12])\b")
_SOURCE_BOUND_IMMIGRATION_RULE_CITATION = re.compile(
    r"(?<![A-Za-z0-9_])([A-Z]{1,4}\d{1,3}\.\d{1,3}(?:/\d{1,3}){1,4})"
    r"(?![A-Za-z0-9_/-]|\.[A-Za-z0-9_])"
)
_SOURCE_BOUND_IMMIGRATION_PART_END = (
    r"(?![A-Za-z0-9_/-]|\.[A-Za-z0-9_]|[ \t]+[A-Za-z0-9_])"
)
_SOURCE_BOUND_IMMIGRATION_PART_REFERENCE = re.compile(
    r"\b(Part\s+\d{1,3}:\s+[a-z][a-z-]*(?:[ \t]+[a-z][a-z-]*){0,4})"
    + _SOURCE_BOUND_IMMIGRATION_PART_END
)
_SOURCE_BOUND_IMMIGRATION_PART_DECLARATION = re.compile(
    r"\bImmigration Rules\s+"
    r"(Part\s+\d{1,3}:\s+[a-z][a-z-]*(?:[ \t]+[a-z][a-z-]*){0,4})"
    + _SOURCE_BOUND_IMMIGRATION_PART_END,
    flags=re.IGNORECASE,
)
_SOURCE_BOUND_IMMIGRATION_PART_OCCURRENCE_END = (
    r"(?![A-Za-z0-9_/-]|\.[A-Za-z0-9_])"
)
_SOURCE_BOUND_IMMIGRATION_PART_INLINE_REFERENCE = re.compile(
    r"\bunder[ \t]+"
    r"(Part\s+\d{1,3}:\s+[a-z][a-z-]*(?:[ \t]+[a-z][a-z-]*){0,1}?)"
    r"(?=[ \t]+applying[ \t]+(?:on|before|after)\b)"
)
_SOURCE_BOUND_IMMIGRATION_RULES_DOCUMENT = re.compile(
    r"(?:\A|\n)Immigration Rules[ \t]+(?:Appendix|Part)\b",
    flags=re.IGNORECASE,
)


def _contextual_official_term_shapes(
    text: str,
) -> tuple[tuple[int, int, str], ...]:
    matches: list[tuple[int, int, str]] = []
    for pattern in (
        _SOURCE_BOUND_ROUTE_TERM,
        _SOURCE_BOUND_TECHNICAL_FRAMEWORK,
        _SOURCE_BOUND_LEVEL_CODE,
        _SOURCE_BOUND_IMMIGRATION_RULE_CITATION,
        _SOURCE_BOUND_IMMIGRATION_PART_REFERENCE,
    ):
        matches.extend(
            (match.start(1), match.end(1), match.group(1))
            for match in pattern.finditer(text)
        )
    return tuple(matches)


def _source_bound_official_terms(
    text: str,
    source_context: str,
) -> tuple[tuple[int, int, str], ...]:
    matches = []
    for acronym in _source_declared_acronyms(source_context):
        for occurrence in re.finditer(
            rf"(?<![A-Za-z0-9_./-]){re.escape(acronym)}"
            r"(?![A-Za-z0-9_]|[./-][A-Za-z0-9_])", text,
        ):
            matches.append((occurrence.start(), occurrence.end(), acronym))
    if _SOURCE_BOUND_IMMIGRATION_RULES_DOCUMENT.search(source_context):
        source_inline_terms = {
            match.group(1)
            for match in _SOURCE_BOUND_IMMIGRATION_PART_INLINE_REFERENCE.finditer(
                source_context
            )
        }
        matches.extend(
            (match.start(1), match.end(1), match.group(1))
            for match in _SOURCE_BOUND_IMMIGRATION_PART_INLINE_REFERENCE.finditer(text)
            if match.group(1) in source_inline_terms
        )
    for declaration in _SOURCE_BOUND_IMMIGRATION_PART_DECLARATION.finditer(
        source_context
    ):
        declared = "Part" + declaration.group(1)[4:]
        for occurrence in re.finditer(
            rf"(?<![A-Za-z0-9_])({re.escape(declared)})"
            + _SOURCE_BOUND_IMMIGRATION_PART_OCCURRENCE_END,
            text,
        ):
            matches.append(
                (occurrence.start(1), occurrence.end(1), occurrence.group(1))
            )
    for start, end, term in _contextual_official_term_shapes(text):
        if not re.search(_entity_pattern(term), source_context):
            continue
        reference_pattern = (
            _SOURCE_BOUND_IMMIGRATION_RULE_CITATION
            if _SOURCE_BOUND_IMMIGRATION_RULE_CITATION.fullmatch(term)
            else _SOURCE_BOUND_IMMIGRATION_PART_REFERENCE
            if _SOURCE_BOUND_IMMIGRATION_PART_REFERENCE.fullmatch(term)
            else None
        )
        if reference_pattern is not None and not any(
            match.group(1) == term for match in reference_pattern.finditer(source_context)
        ):
            continue
        if reference_pattern is _SOURCE_BOUND_IMMIGRATION_RULE_CITATION:
            declared = re.search(
                r"\bImmigration Rules Appendix\b", source_context,
                flags=re.IGNORECASE,
            )
        elif reference_pattern is _SOURCE_BOUND_IMMIGRATION_PART_REFERENCE:
            declared = re.search(
                rf"\bImmigration Rules\s+{re.escape(term)}"
                + _SOURCE_BOUND_IMMIGRATION_PART_END,
                source_context,
                flags=re.IGNORECASE,
            )
        elif re.search(rf"\b{re.escape(term)}\s+route\b", text):
            declared = re.search(
                rf"\b{re.escape(term)}\s+route\b", source_context
            )
        else:
            declared = True
        if declared:
            matches.append((start, end, term))
    return tuple(matches)


def _is_bounded_english_organisation(text: str) -> bool:
    if _ENGLISH_UNIVERSITY_OF.fullmatch(text):
        return True
    return (
        not re.fullmatch(
            r"(?:The|A|An) (?:Authority|Directorate|Department|Ministry|Agency|"
            r"Council|Commission|Service|Police|University|Hospital|Bank)",
            text,
        )
        and bool(_ENGLISH_ORGANISATION.fullmatch(text))
        and not any(
            token.casefold() in _ENGLISH_ORGANISATION_ACTION_WORDS
            for token in re.findall(r"[A-Za-z]+", text)
        )
    )


def _literal_csv_cells(line: str, number: int) -> dict[str, str] | None:
    prefix = f"Row {number}: "
    if not line.startswith(prefix):
        return None
    if line == prefix + "[empty record]":
        return {}
    cells: dict[str, str] = {}
    offset = len(prefix)
    decoder = json.JSONDecoder()
    column_number = 0
    while offset < len(line):
        start = offset
        column_number += 1
        value = column_number
        column = ""
        while value:
            value, digit = divmod(value - 1, 26)
            column = chr(65 + digit) + column
        marker = column + "="
        if not line.startswith(marker, offset):
            return None
        offset += len(marker)
        try:
            cell, offset = decoder.raw_decode(line, offset)
        except ValueError:
            return None
        if type(cell) is not str or line[start:offset] != marker + json.dumps(cell, ensure_ascii=False):
            return None
        cells[column] = cell
        if offset == len(line):
            break
        if not line.startswith("; ", offset):
            return None
        offset += 2
        if offset == len(line):
            return None
    return cells or None


def _table_group_organisation_shape(text: str, source_context: str) -> bool:
    if (
        len(text) > 80
        or not re.fullmatch(r"[A-Z][A-Za-z.-]+(?: [A-Z][A-Za-z.-]+){1,3} Group", text)
        or any(token.casefold() in _ENGLISH_ORGANISATION_ACTION_WORDS
               for token in text.split())
    ):
        return False
    return any(
        cells is not None and cells.get("C") == text
        for line in source_context.splitlines()
        if (match := re.fullmatch(r"Row ([2-9]|[1-9][0-9]+): .*", line))
        for cells in (_literal_csv_cells(line, int(match.group(1))),)
    )


def _source_bound_literal_csv_entities(
    text: str, source_context: str,
) -> tuple[tuple[int, int, str, str], ...]:
    # The converter caps the source body at 1 MiB; this parser never indexes
    # every cell against the full table for each proposed entity.
    if len(source_context.encode("utf-8")) > 1_048_576:
        return ()
    lines = source_context.splitlines()
    if len(lines) >= 7 and lines[0].strip() and lines[1] == "":
        # Independent acquisition retains one declared title before the
        # converter's otherwise exact attachment body.
        lines = lines[2:]
    if (
        len(lines) < 5
        or not re.fullmatch(
            r"Attachment: https://assets\.publishing\.service\.gov\.uk/\S+\.csv",
            lines[0],
        )
        or lines[1:3] != [_LITERAL_CSV_PREAMBLE, 'Sheet "CSV"']
    ):
        return ()
    parsed = []
    for number, line in enumerate(lines[3:], 1):
        cells = _literal_csv_cells(line, number)
        if cells is None:
            return ()
        parsed.append((line, cells))
    header = parsed[0][1]
    labels = [value.strip().casefold() for value in header.values()]
    if not all(labels) or len(labels) != len(set(labels)):
        return ()
    typed = []
    if header.get("A", "").strip().casefold() == "senior official's name":
        typed.append(("A", "PERSON"))
    if header.get("C", "").strip().casefold() in {
        "name of individual or organisation",
        "individual or organisation that provided hospitality",
    }:
        typed.append(("C", "ORGANISATION"))
    if not typed:
        return ()
    claimed_lines: dict[str, list[int]] = {}
    offset = 0
    for line in text.splitlines(keepends=True):
        claimed_lines.setdefault(line.rstrip("\r\n"), []).append(offset)
        offset += len(line)
    matches = []
    for line, cells in parsed[1:]:
        if line not in claimed_lines:
            continue
        for column, entity_type in typed:
            name = cells.get(column)
            if not name or name.strip().casefold() in _LITERAL_CSV_NIL:
                continue
            # Only literal unescaped name bytes can form the same exact span
            # in a claim and the retained JSON-string cell.
            if (name != name.strip()
                    or json.dumps(name, ensure_ascii=False) != '"' + name + '"'
                    or re.search(r"[\n\r;:！？!?]", name)):
                continue
            if not _has_bounded_named_entity_shape(
                name, entity_type, source_context=line,
            ):
                continue
            cell_start = line.find(column + "=" + json.dumps(name, ensure_ascii=False))
            if cell_start < 0:
                continue
            name_start = cell_start + len(column) + 2
            for line_start in claimed_lines[line]:
                matches.append((line_start + name_start, line_start + name_start + len(name),
                                name, entity_type))
    return tuple(matches)


def _has_bounded_named_entity_shape(
    text: str,
    entity_type: str,
    *,
    source_context: str = "",
) -> bool:
    if text in _OWNER_APPROVED_ENTITY_REGISTRY:
        return entity_type == _OWNER_APPROVED_ENTITY_REGISTRY[text]
    if re.search(r"[A-Za-z]", text):
        if entity_type == "OFFICIAL_TERM":
            return bool(
                _ENGLISH_OFFICIAL_TERM.fullmatch(text)
                or _ENGLISH_OFFICIAL_REFERENCE.fullmatch(text)
                or _SOURCE_BOUND_IMMIGRATION_RULE_CITATION.fullmatch(text)
                or _SOURCE_BOUND_IMMIGRATION_PART_REFERENCE.fullmatch(text)
                or text in _BOUNDED_OFFICIAL_ABBREVIATIONS
                # Shape only: admission independently requires an exact
                # declaration in the retained source, claim and excerpt binding.
                or (re.fullmatch(r"[A-Z][A-Za-z]{1,9}", text)
                    and sum(letter.isupper() for letter in text) >= 2)
                or any(
                    candidate == text
                    for _start, _end, candidate in _contextual_official_term_shapes(
                        source_context
                    )
                )
            )
        if entity_type == "ORGANISATION":
            return (_is_bounded_english_organisation(text)
                    or _table_group_organisation_shape(text, source_context))
        tokens = re.findall(r"[A-Za-z]+", text)
        if not tokens or any(
            not (
                token.isupper()
                or token[:1].isupper()
                or token.casefold() in {"of", "the", "and", "for"}
            )
            for token in tokens
        ):
            return False
        return entity_type == "PERSON" and 2 <= len(tokens) <= 3
    if not re.fullmatch(r"[\u3400-\u9fff《》〈〉]+", text):
        return False
    suffixes = {
        "ORGANISATION": (
            "政府",
            "署",
            "局",
            "部",
            "委員會",
            "協會",
            "公司",
            "大學",
            "學校",
            "法院",
            "警方",
            "醫院",
            "銀行",
            "管理局",
        ),
        "PLACE": (
            "市",
            "區",
            "國",
            "灣",
            "道",
            "路",
            "山",
            "河",
            "島",
            "州",
            "縣",
            "鎮",
            "角",
        ),
        "OFFICIAL_TITLE": ("長", "司", "官", "大臣", "主席", "總統"),
    }
    if entity_type == "PERSON":
        return 2 <= len(text) <= 4
    if entity_type == "PLACE":
        return 2 <= len(text) <= 5
    if entity_type == "PRODUCT":
        return text.startswith(("《", "〈")) and text.endswith(("》", "〉"))
    return text.endswith(suffixes.get(entity_type, ()))


def _entity_pattern(entity: str) -> str:
    # Latin identifiers may touch Chinese copy, but not a different Latin word.
    return (
        (r"(?<![A-Za-z0-9_])" if entity[0].isascii() else "")
        + re.escape(entity)
        + (r"(?![A-Za-z0-9_])" if entity[-1].isascii() else "")
    )


def rendered_named_entities(
    text: str, source_entities: frozenset[tuple[str, str]],
    *, policy_version: str = NAMED_ENTITY_POLICY_VERSION,
) -> frozenset[tuple[str, str]]:
    """Preserve exact source names without requiring their English verb context.

    The caller derives this inventory independently from the claim and excerpt.
    Other recognised entities remain visible; unrecognised Latin prose still
    fails the separate Hong Kong rendering check.
    """
    retained = set()
    for entity, kind in sorted(source_entities, key=lambda item: (-len(item[0]), item)):
        text, count = re.subn(_entity_pattern(entity), " ", text)
        if count:
            retained.add((entity, kind))
    return frozenset(retained) | bounded_named_entities(text, policy_version=policy_version)


def bounded_named_entities(
    text: str,
    *,
    source_context: str | None = None,
    policy_version: str = NAMED_ENTITY_POLICY_VERSION,
) -> frozenset[tuple[str, str]]:
    """Extract only closed, structurally recognisable entity spans."""

    if policy_version not in {NAMED_ENTITY_POLICY_VERSION_V15, NAMED_ENTITY_POLICY_VERSION}:
        raise ValueError("unsupported named-entity policy")

    candidates: list[tuple[int, int, str, str]] = []
    for entity, entity_type in _OWNER_APPROVED_ENTITY_REGISTRY.items():
        for match in re.finditer(_entity_pattern(entity), text):
            candidates.append((match.start(), match.end(), entity, entity_type))
    english_person = re.compile(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})"
        r"(?=\s+(?:said|says|announced|confirmed|stated|warned|told)\b)"
    )
    for match in english_person.finditer(text):
        candidates.append((match.start(1), match.end(1), match.group(1), "PERSON"))
    for match in _ENGLISH_ORGANISATION_CANDIDATES.finditer(text):
        organisation = match.group(1)
        if _is_bounded_english_organisation(organisation):
            candidates.append(
                (match.start(1), match.end(1), organisation, "ORGANISATION")
            )
    for match in _ENGLISH_UNIVERSITY_OF.finditer(text):
        candidates.append(
            (match.start(1), match.end(1), match.group(1), "ORGANISATION")
        )
    official_terms = _ENGLISH_OFFICIAL_TERM_V15 if policy_version == NAMED_ENTITY_POLICY_VERSION_V15 else _ENGLISH_OFFICIAL_TERM
    for match in official_terms.finditer(text):
        candidates.append((match.start(), match.end(), match.group(0), "OFFICIAL_TERM"))
    for match in _ENGLISH_OFFICIAL_REFERENCE.finditer(text):
        candidates.append((match.start(), match.end(), match.group(0), "OFFICIAL_TERM"))
    for abbreviation in _BOUNDED_OFFICIAL_ABBREVIATIONS:
        for match in re.finditer(_entity_pattern(abbreviation), text):
            candidates.append(
                (match.start(), match.end(), abbreviation, "OFFICIAL_TERM")
            )
    if source_context is not None:
        for start, end, term in _source_bound_official_terms(text, source_context):
            candidates.append((start, end, term, "OFFICIAL_TERM"))
        candidates.extend(_source_bound_literal_csv_entities(text, source_context))
    titled_chinese_person = re.compile(
        r"(行政長官|財政司司長|政務司司長|律政司司長|特首|司長|局長|署長)"
        r"([趙錢孫李周吳鄭王馮陳褚衛蔣沈韓楊朱秦尤許何呂施張孔曹嚴華金魏陶姜戚謝鄒喻柏水竇章雲蘇潘葛奚范彭郎魯韋昌馬苗鳳花方俞任袁柳唐羅薛伍余米貝姚孟顧尹江鍾蔡葉杜夏汪田]"
        r"[\u3400-\u9fff]{1,2})"
        r"(?=[\u3400-\u9fff]{0,8}(?:公布|宣佈|宣布|表示|指出|證實|確認|警告|"
        r"稱|說|指|主持|出席|會見))"
    )
    for match in titled_chinese_person.finditer(text):
        candidates.append(
            (match.start(1), match.end(1), match.group(1), "OFFICIAL_TITLE")
        )
        candidates.append((match.start(2), match.end(2), match.group(2), "PERSON"))
    chinese_person = re.compile(
        r"(?<![\u3400-\u9fff])"
        r"([趙錢孫李周吳鄭王馮陳褚衛蔣沈韓楊朱秦尤許何呂施張孔曹嚴華金魏陶姜戚謝鄒喻柏水竇章雲蘇潘葛奚范彭郎魯韋昌馬苗鳳花方俞任袁柳唐羅薛伍余米貝姚孟顧尹江鍾蔡葉杜夏汪田]"
        r"[\u3400-\u9fff]{1,2})"
        r"(?=(?:公布|宣佈|宣布|表示|指出|證實|確認|警告|"
        r"稱|說|指|主持|出席|會見|簽署|签署|視察|视察|任命|接見|接见|"
        r"辭職|辞职|請辭|请辞))"
    )
    for match in chinese_person.finditer(text):
        if match.group(1) not in {"方資料", "方代表", "方表示"} and not any(
            marker in match.group(1)
            for marker in ("任命", "委任", "出任", "局長", "署長", "司長")
        ):
            candidates.append((match.start(1), match.end(1), match.group(1), "PERSON"))
    appointed_chinese_person = re.compile(
        r"(?:任命|委任|提名|公布[：:]?|指[：:]?|由)"
        r"([趙錢孫李周吳鄭王馮陳褚衛蔣沈韓楊朱秦尤許何呂施張孔曹嚴華金魏陶姜戚謝鄒喻柏水竇章雲蘇潘葛奚范彭郎魯韋昌馬苗鳳花方俞任袁柳唐羅薛伍余米貝姚孟顧尹江鍾蔡葉杜夏汪田劉郭梁黃林]"
        r"[\u3400-\u9fff]{2})"
        r"(?=(?:出任|擔任|担任|任職|任职|獲委任|获委任|接任|升任))"
    )
    for match in appointed_chinese_person.finditer(text):
        person = match.group(1)
        if person[1:] not in {"表示", "安排", "措施", "政策", "案甲", "代表"}:
            candidates.append((match.start(1), match.end(1), person, "PERSON"))
    interaction_chinese_person = re.compile(
        r"(?:會見|会见|接見|接见|拘捕|起訴|起诉|邀請|邀请)"
        r"([趙錢孫李周吳鄭王馮陳褚衛蔣沈韓楊朱秦尤許何呂施張孔曹嚴華金魏陶姜戚謝鄒喻柏水竇章雲蘇潘葛奚范彭郎魯韋昌馬苗鳳花方俞任袁柳唐羅薛伍余米貝姚孟顧尹江鍾蔡葉杜夏汪田劉郭梁黃林]"
        r"[\u3400-\u9fff]{2})"
        r"(?=$|[，,。；;：:]|(?:後|后|時|时|並|并|出席|表示|獲准|获准))"
    )
    for match in interaction_chinese_person.finditer(text):
        person = match.group(1)
        if person[1:] not in {"表示", "安排", "措施", "政策", "案甲"}:
            candidates.append((match.start(1), match.end(1), person, "PERSON"))
    structural_chinese_place = re.compile(
        r"(?:公布|涉及|位於|位于|前往|覆蓋|覆盖|影響|影响)"
        r"([\u3400-\u9fff]{1,4}?(?:市|區|区|國|国|灣|湾|道|路|山|河|島|岛|"
        r"州|縣|县|鎮|镇|角|咀))"
        r"(?=(?:嘅|的)?(?:新安排|安排|措施|計劃|计划|服務|服务|居民|地區|地区))"
    )
    for match in structural_chinese_place.finditer(text):
        candidates.append((match.start(1), match.end(1), match.group(1), "PLACE"))
    action_context_structural_place = re.compile(
        r"(?:公布|指)[：:]?([\u3400-\u9fff]{1,4}?(?:市|區|区|國|国|灣|湾|"
        r"道|路|山|河|島|岛|州|縣|县|鎮|镇|角|咀))(?=(?:將|将)"
        r"(?:實施|实施|推行|設立|设立|開設|开设|啟用|启用))"
    )
    for match in action_context_structural_place.finditer(text):
        candidates.append((match.start(1), match.end(1), match.group(1), "PLACE"))
    chinese_organisation = re.compile(
        r"(?<![\u3400-\u9fff])"
        r"([\u3400-\u9fff]{2,16}(?:政府|醫院管理局|管理局|委員會|協會|"
        r"公司|大學|學校|法院|警方|醫院|銀行|署|局|部))"
        r"(?=(?:公布|宣佈|宣布|表示|指出|證實|確認|警告|稱|說|指|推出))"
    )
    for match in chinese_organisation.finditer(text):
        candidates.append(
            (match.start(1), match.end(1), match.group(1), "ORGANISATION")
        )
    for match in re.finditer(r"[《〈][^《》〈〉\n]{1,80}[》〉]", text):
        if policy_version != NAMED_ENTITY_POLICY_VERSION_V15 and not match.group(0)[1:-1].strip():
            continue
        candidates.append((match.start(), match.end(), match.group(0), "PRODUCT"))
    selected: list[tuple[int, int, str, str]] = []
    for candidate in sorted(candidates, key=lambda item: (-(item[1] - item[0]), item)):
        if not any(
            candidate[0] < existing[1] and existing[0] < candidate[1]
            for existing in selected
        ):
            selected.append(candidate)
    return frozenset(
        (text, entity_type) for _start, _end, text, entity_type in selected
    )


def _chinese_integer(value: str) -> int | None:
    def section(raw: str) -> int | None:
        if not raw:
            return 0
        if raw.isdigit():
            return int(raw)
        if all(character in _CHINESE_DIGITS for character in raw):
            return int("".join(str(_CHINESE_DIGITS[character]) for character in raw))
        small_units = {"十": 10, "百": 100, "千": 1_000}
        result = 0
        number = 0
        last_unit_value = 0
        last_unit_index = -1
        for character in raw:
            if character in _CHINESE_DIGITS:
                number = _CHINESE_DIGITS[character]
            elif character in small_units:
                result += (number or 1) * small_units[character]
                number = 0
                last_unit_value = small_units[character]
                last_unit_index = raw.index(character, last_unit_index + 1)
            else:
                return None
        if (
            number
            and last_unit_value >= 100
            and "零" not in raw[last_unit_index + 1 :]
            and "〇" not in raw[last_unit_index + 1 :]
        ):
            return None
        return result + number

    def below_yi(raw: str) -> int | None:
        separators = tuple(character for character in ("萬", "万") if character in raw)
        if len(separators) > 1 or (separators and raw.count(separators[0]) != 1):
            return None
        if not separators:
            return section(raw)
        left, right = raw.split(separators[0])
        if not left or (
            right
            and not right.startswith(("零", "〇"))
            and not any(unit in right for unit in ("十", "百", "千"))
        ):
            return None
        high = section(left)
        low = section(right)
        if high is None or low is None:
            return None
        return high * 10_000 + low

    yi_separators = tuple(character for character in ("億", "亿") if character in value)
    if len(yi_separators) > 1 or (yi_separators and value.count(yi_separators[0]) != 1):
        return None
    if not yi_separators:
        return below_yi(value)
    left, right = value.split(yi_separators[0])
    if not left or (
        right
        and not right.startswith(("零", "〇"))
        and not any(unit in right for unit in ("十", "百", "千", "萬", "万"))
    ):
        return None
    high = below_yi(left)
    low = below_yi(right)
    if high is None or low is None:
        return None
    return high * 100_000_000 + low


def _valid_canonical_date(value: tuple[object, ...]) -> bool:
    _kind, year, month, day, hour, minute = value
    if not isinstance(month, int) or not isinstance(day, int):
        return False
    if (hour is None) != (minute is None):
        return False
    if hour is not None and (
        not isinstance(hour, int)
        or not isinstance(minute, int)
        or not 0 <= hour <= 23
        or not 0 <= minute <= 59
    ):
        return False
    try:
        date(int(year) if isinstance(year, int) else 2000, int(month), int(day))
    except ValueError:
        return False
    return True


def _parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _has_valid_origin_independence(
    authority_class: ClaimAuthorityClass,
    source_records: Sequence[Mapping[str, object]],
    dependency_records: Sequence[Mapping[str, object]],
    expected_origins: tuple[str, ...],
) -> bool:
    report_to_origins: dict[str, set[str]] = {}
    origin_to_reports: dict[str, set[str]] = {}
    for record in dependency_records:
        report = record.get("originating_report_id")
        origin = record.get("evidential_origin_id")
        if (
            not isinstance(report, str)
            or not report
            or not isinstance(origin, str)
            or not origin
        ):
            return False
        report_to_origins.setdefault(report, set()).add(origin)
        origin_to_reports.setdefault(origin, set()).add(report)
    if (
        not report_to_origins
        or any(len(values) != 1 for values in report_to_origins.values())
        or any(len(values) != 1 for values in origin_to_reports.values())
        or set(origin_to_reports) != set(expected_origins)
    ):
        return False
    if authority_class is not ClaimAuthorityClass.INDEPENDENT_RELIABLE:
        return True
    source_ids = {record.get("source_id") for record in source_records}
    canonical_urls = {record.get("canonical_url") for record in source_records}
    source_reports = {record.get("originating_report_id") for record in source_records}
    artefact_digests = {
        record.get("originating_artefact_digest") for record in source_records
    }
    return (
        len(source_records) >= 2
        and len(source_ids) == len(source_records)
        and len(canonical_urls) == len(source_records)
        and len(source_reports) == len(source_records)
        and len(artefact_digests) == len(source_records)
        and len(report_to_origins) >= 2
    )


def _canonical_localised_fact(value: str) -> tuple[object, ...] | None:
    value = value.strip()
    english_month = re.fullmatch(r"([A-Za-z]+)(?:\s+(\d{4}))?", value)
    if english_month:
        month = _ENGLISH_MONTHS.get(english_month.group(1).casefold())
        year = int(english_month.group(2)) if english_month.group(2) else None
        if (
            month is not None and (year is None or 1 <= year <= 9999)
            and (year is not None or english_month.group(1).istitle())
        ):
            return ("CALENDAR_MONTH", year, month)
    chinese_month = re.fullmatch(
        r"(?:(\d{4}|[零〇一二三四五六七八九十]+)年)?"
        r"(\d{1,2}|[零〇一二三四五六七八九十]+)月", value,
    )
    if chinese_month:
        year = _chinese_integer(chinese_month.group(1)) if chinese_month.group(1) else None
        month = _chinese_integer(chinese_month.group(2))
        if (
            month is not None and 1 <= month <= 12
            and (chinese_month.group(1) is None or (year is not None and 1 <= year <= 9999))
        ):
            return ("CALENDAR_MONTH", year, month)
    english_date = re.fullmatch(
        r"(\d{1,2})\s+([A-Za-z]+)(?:\s+(\d{4}))?"
        r"(?:\s+at\s+(\d{1,2}):(\d{2}))?",
        value,
        flags=re.IGNORECASE,
    )
    if english_date:
        month = _ENGLISH_MONTHS.get(english_date.group(2).casefold())
        if month is not None:
            result = (
                "DATE_TIME",
                int(english_date.group(3)) if english_date.group(3) else None,
                month,
                int(english_date.group(1)),
                int(english_date.group(4)) if english_date.group(4) else None,
                int(english_date.group(5)) if english_date.group(5) else None,
            )
            if not _valid_canonical_date(result):
                return None
            return result
    chinese_date = re.fullmatch(
        r"(?:(\d{4}|[零〇一二三四五六七八九十]+)年)?"
        r"(\d{1,2}|[零〇一二三四五六七八九十]+)月"
        r"(\d{1,2}|[零〇一二三四五六七八九十]+)(?:日|號|号)"
        r"(?:(上午|下午)?(\d{1,2}|[零〇一二三四五六七八九十]+)"
        r"(?:時|时|點|点|:)(\d{1,2}|[零〇一二三四五六七八九十]+)分?)?",
        value,
    )
    if chinese_date:
        year = (
            _chinese_integer(chinese_date.group(1)) if chinese_date.group(1) else None
        )
        month = _chinese_integer(chinese_date.group(2))
        day = _chinese_integer(chinese_date.group(3))
        hour = (
            _chinese_integer(chinese_date.group(5)) if chinese_date.group(5) else None
        )
        minute = (
            _chinese_integer(chinese_date.group(6)) if chinese_date.group(6) else None
        )
        if (chinese_date.group(1) is not None and year is None) or (
            chinese_date.group(5) is not None and (hour is None or minute is None)
        ):
            return None
        if ":" in value and chinese_date.group(4) and not 1 <= hour <= 12:
            return None
        if hour is not None:
            if chinese_date.group(4) == "上午" and hour == 12:
                hour = 0
            elif chinese_date.group(4) == "下午" and hour < 12:
                hour += 12
        result = ("DATE_TIME", year, month, day, hour, minute)
        if not _valid_canonical_date(result):
            return None
        return result
    pound_number = r"([0-9]+|[1-9][0-9]{0,2}(?:,[0-9]{3})+)"
    pound_scale = r"(?:\s+(thousand|million|billion))?"
    pounds = re.fullmatch(
        r"(?:£|GBP\s+)\s*" + pound_number + pound_scale, value, re.IGNORECASE,
    )
    if pounds is None:
        pounds = re.fullmatch(pound_number + pound_scale + r"\s+pounds sterling", value, re.IGNORECASE)
    if pounds is not None:
        scale = {None: 1, 'thousand': 1_000, 'million': 1_000_000, 'billion': 1_000_000_000}
        multiplier = scale[pounds.group(2).lower() if pounds.group(2) else None]
        return ('MONEY', 'GBP', int(pounds.group(1).replace(',', '')) * multiplier)
    chinese_pounds = re.fullmatch(r"([0-9零〇一二三四五六七八九十百千萬万億亿兩两]+)英鎊", value)
    if chinese_pounds is not None:
        amount = _chinese_integer(chinese_pounds.group(1))
        if amount is not None:
            return ('MONEY', 'GBP', amount)
    english_money = re.fullmatch(r"HK\$\s*([\d,]+)", value, re.IGNORECASE)
    if english_money:
        return ("MONEY", "HKD", int(english_money.group(1).replace(",", "")))
    chinese_money = re.fullmatch(
        r"([零〇一二三四五六七八九十百千萬万億亿兩两]+)(?:港元|元)", value
    )
    if chinese_money:
        amount = _chinese_integer(chinese_money.group(1))
        if amount is None:
            return None
        return ("MONEY", "HKD", amount)
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    english_calendar_months = re.fullmatch(
        r"(\d+)\s+months?", value, flags=re.IGNORECASE,
    )
    if english_calendar_months:
        return ("DURATION_CALENDAR_MONTHS", int(english_calendar_months.group(1)))
    chinese_calendar_months = re.fullmatch(
        r"([零〇一二三四五六七八九十百千兩两\d]+)(?:個月|个月)", value,
    )
    if chinese_calendar_months:
        number = _chinese_integer(chinese_calendar_months.group(1))
        if number is not None:
            return ("DURATION_CALENDAR_MONTHS", number)
    english_calendar_years = re.fullmatch(
        r"(\d+)\s+years?", value, flags=re.IGNORECASE,
    )
    if english_calendar_years:
        return ("DURATION_CALENDAR_YEARS", int(english_calendar_years.group(1)))
    chinese_calendar_years = re.fullmatch(
        r"([零〇一二三四五六七八九十百千兩两\d]+)年", value,
    )
    if chinese_calendar_years:
        number = _chinese_integer(chinese_calendar_years.group(1))
        if number is not None:
            return ("DURATION_CALENDAR_YEARS", number)
    english_duration = re.fullmatch(
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(hours?|minutes?)",
        value,
        flags=re.IGNORECASE,
    )
    if english_duration:
        raw_number = english_duration.group(1).casefold()
        number = int(raw_number) if raw_number.isdigit() else number_words[raw_number]
        minutes = (
            number * 60
            if english_duration.group(2).casefold().startswith("hour")
            else number
        )
        return ("DURATION_MINUTES", minutes)
    chinese_duration = re.fullmatch(
        r"([零〇一二三四五六七八九十百千兩两\d]+)(小時|小时|分鐘|分钟)",
        value,
    )
    if chinese_duration:
        number = _chinese_integer(chinese_duration.group(1))
        if number is None:
            return None
        minutes = (
            number * 60 if chinese_duration.group(2) in {"小時", "小时"} else number
        )
        return ("DURATION_MINUTES", minutes)
    english_count = re.fullmatch(
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(schools?|hospitals?|clinics?|buses?|roads?)",
        value,
        flags=re.IGNORECASE,
    )
    if english_count:
        raw_number = english_count.group(1).casefold()
        number = int(raw_number) if raw_number.isdigit() else number_words[raw_number]
        objects = {
            "school": "SCHOOL",
            "hospital": "HOSPITAL",
            "clinic": "CLINIC",
            "bus": "BUS",
            "road": "ROAD",
        }
        return (
            "COUNT",
            objects[english_count.group(2).casefold().removesuffix("s")],
            number,
        )
    chinese_count = re.fullmatch(
        r"([零〇一二三四五六七八九十百千兩两\d]+)"
        r"(?:間|间|所|間|部|輛|辆|條|条)"
        r"(學校|学校|醫院|医院|診所|诊所|巴士|道路)",
        value,
    )
    if chinese_count:
        number = _chinese_integer(chinese_count.group(1))
        if number is None:
            return None
        objects = {
            "學校": "SCHOOL",
            "学校": "SCHOOL",
            "醫院": "HOSPITAL",
            "医院": "HOSPITAL",
            "診所": "CLINIC",
            "诊所": "CLINIC",
            "巴士": "BUS",
            "道路": "ROAD",
        }
        return ("COUNT", objects[chinese_count.group(2)], number)
    return None


def _calendar_month_occurs(expression: str, text: str) -> bool:
    # A calendar month must not be a substring of a word, another month or a
    # more precise date. A bare English month also needs calendar context: it
    # could otherwise be a modal, action or name (May, March, August).
    boundary = "A-Za-z0-9_零〇一二三四五六七八九十百千萬万億亿兩两年月日號号"
    for match in re.finditer(
        rf"(?<![{boundary}]){re.escape(expression)}(?![{boundary}])", text,
    ):
        if re.fullmatch(r"[A-Za-z]+(?:\s+\d{4})?", expression) and (
            re.search(r"\d\s+$", text[:match.start()])
            or re.match(r"\s+\d", text[match.end():])
        ):
            continue
        if re.fullmatch(r"[A-Za-z]+", expression) and not re.search(
            r"\b(?:in|during|from|until|through|by|before|after|since|between|for)\s+$",
            text[:match.start()], flags=re.IGNORECASE,
        ):
            continue
        return True
    return False


def _localised_fact_is_bound(
    source: str, target: str, claim: str, excerpt: str, rendered: str,
) -> bool:
    fact = _canonical_localised_fact(source)
    if fact is None or fact != _canonical_localised_fact(target):
        return False
    if fact[0] == "CALENDAR_MONTH":
        return (
            _calendar_month_occurs(source, claim)
            or _calendar_month_occurs(source, excerpt)
        ) and _calendar_month_occurs(target, rendered)
    if fact[:2] == ('MONEY', 'GBP'):
        # A lookup key must name the whole amount, not £20 in £200 million,
        # sterling inside E£, or 二百 inside 一千二百英鎊.
        digits = '0-9零〇一二三四五六七八九十百千萬万億亿兩两'
        pound_expression = re.compile(
            r'(?<![A-Za-z0-9£$€.,+−-])(?:£|GBP\s+)\s*[0-9]+(?:[.,][0-9]+)*'
            r'(?:\s+(?:thousand|million|billion))?'
            r'(?![A-Za-z0-9]|[.,][0-9]|\s+(?:thousand|million|billion)\b)|'
            r'(?<![A-Za-z0-9£$€.,+−-])[0-9]+(?:[.,][0-9]+)*'
            r'(?:\s+(?:thousand|million|billion))?\s+pounds sterling(?![A-Za-z])|'
            rf'(?<![{digits}點点.,+−負负-])[{digits}]+英鎊',
            re.IGNORECASE,
        )
        def occurs(expression, text):
            for match in pound_expression.finditer(text):
                if match.group() != expression or _canonical_localised_fact(match.group()) != fact:
                    continue
                before, after = text[:match.start()].rstrip(), text[match.end():].lstrip()
                if before.endswith(('-', '−', '+', '負', '负')):
                    continue
                if after and (after[0].isnumeric() or after[0] in '/⁄半'):
                    continue
                return True
            return False
        return (occurs(source, claim) or occurs(source, excerpt)) and occurs(target, rendered)
    return (source in claim or source in excerpt) and target in rendered


_FACT_NUMBER_V2 = r"(?:[+−-]?[0-9]+(?:[.,][0-9]+)*|[零〇一二三四五六七八九十百千萬万億亿兩两]+)"
_FACT_VALUE_V2 = rf"(?:{_FACT_NUMBER_V2}|one|two|three|four|five|six|seven|eight|nine|ten)"
_FACT_UNIT_V2 = (r"months?|years?|hours?|minutes?|schools?|hospitals?|clinics?|buses?|roads?|"
                 r"個月|个月|年|小時|小时|分鐘|分钟|(?:間|间|所|部|輛|辆|條|条)(?:學校|学校|醫院|医院|診所|诊所|巴士|道路)")
_FACT_QUALIFIER_V2 = (r"only\s+|exactly\s+|at least\s+|at most\s+|more than\s+|less than\s+|up to\s+|"
                      r"about\s+|approximately\s+|around\s+|within\s+|"
                      r"只限|只有|僅限|恰好|至少|最少|不少於|最多|不多於|超過|多於|少於|不足|大約|約")


def canonical_localised_fact_v2(value: str) -> tuple[object, ...] | None:
    """Closed factual expressions, not a translation or semantic proof."""
    if type(value) is not str or len(value) > 256:
        return None
    value = value.strip()
    bounds = ((r"only\s+|只限|只有|僅限", "ONLY"),
              (r"exactly\s+|恰好", "EXACT"),
              (r"at least\s+|至少|最少|不少於", "GE"),
              (r"at most\s+|up to\s+|最多|不多於", "LE"),
              (r"more than\s+|超過|多於", "GT"),
              (r"less than\s+|少於|不足", "LT"),
              (r"about\s+|approximately\s+|around\s+|大約|約", "APPROX"),
              (r"within\s+", "WITHIN"))
    for prefix, operator in bounds:
        match = re.fullmatch(rf"(?:{prefix})(.+)", value, re.I)
        if match:
            fact = canonical_localised_fact_v2(match.group(1))
            if operator == "WITHIN" and fact is not None and not fact[0].startswith("DURATION_"):
                return None
            return ("BOUND", operator, fact) if fact is not None else None
    deadline = re.fullmatch(r"(.+?)(?:期限)?內", value)
    if deadline:
        fact = canonical_localised_fact_v2(deadline[1])
        return ("BOUND", "WITHIN", fact) if fact is not None and fact[0].startswith("DURATION_") else None
    if re.fullmatch(r"one of(?: the following roles)?|其中一[個位名項種](?:角色)?", value, re.I):
        return ("SELECTION", 1)
    year = re.fullmatch(r"(academic|financial) year\s+([0-9]{4})\s*(?:to|[-–至])\s*([0-9]{4})", value, re.I)
    if year:
        return ("YEAR_RANGE", year[1].upper() + "_YEAR", int(year[2]), int(year[3])) if 1 <= int(year[2]) < int(year[3]) <= 9999 else None
    year = re.fullmatch(r"([0-9]{4})\s*(?:to|[-–至])\s*([0-9]{4})(學年|學年度|財政年度)", value, re.I)
    if year:
        return ("YEAR_RANGE", "FINANCIAL_YEAR" if year[3] == "財政年度" else "ACADEMIC_YEAR", int(year[1]), int(year[2])) if 1 <= int(year[1]) < int(year[2]) <= 9999 else None
    interval = re.fullmatch(rf"({_FACT_NUMBER_V2})\s*(?:to|[-–至])\s*({_FACT_NUMBER_V2})(?:\s*({_FACT_UNIT_V2}))?", value, re.I)
    if interval:
        suffix = " " + interval[3] if interval[3] else ""
        left, right = (canonical_localised_fact_v2(interval[index] + suffix) for index in (1, 2))
        if left is not None and right is not None and left[:-1] == right[:-1]:
            return ("RANGE", left, right)
        return None
    scalar = re.fullmatch(rf"({_FACT_VALUE_V2})\s*({_FACT_UNIT_V2})", value, re.I)
    if scalar:
        words = dict(zip("one two three four five six seven eight nine ten".split(), range(1, 11)))
        amount = str(words.get(scalar[1].casefold(), scalar[1]))
        value = amount + (" " if scalar[2].isascii() else "") + scalar[2]
        if scalar[2].casefold() == "buses":
            return ("COUNT", "BUS", int(amount)) if amount.isascii() and amount.isdigit() else None
    existing = _canonical_localised_fact(value)
    if existing is not None:
        return existing
    if re.fullmatch(r"[+−-]?(?:[0-9]+|[1-9][0-9]{0,2}(?:,[0-9]{3})+)(?:\.[0-9]+)?", value):
        number = Decimal(value.replace("−", "-").replace(",", ""))
        return ("NUMBER", int(number) if number == int(number) else str(number.normalize()))
    if re.fullmatch(r"[零〇一二三四五六七八九十百千萬万億亿兩两]+", value):
        number = _chinese_integer(value)
        if number is not None:
            return ("NUMBER", number)
    return None


def _factual_occurrences_v2(text: str):
    from .writer import _RELATIVE_TIME_FACT

    # Same-customer co-reference (同一客戶) is not an asserted count.
    boundary = "A-Za-z0-9_\\u3400-\\u9fff"
    pattern = rf"(?<![{boundary}]){_FACT_NUMBER_V2}(?![{boundary}])"
    selection = r"one of(?: the following roles)?|其中一[個位名項種](?:角色)?"
    interval = rf"{_FACT_NUMBER_V2}\s*(?:to|[-–至])\s*{_FACT_NUMBER_V2}"
    year = rf"(?:academic|financial) year\s+{interval}|{interval}(?:學年|學年度|財政年度)"
    quantity = rf"{_FACT_VALUE_V2}\s*(?:{_FACT_UNIT_V2})"
    month = "|".join(_ENGLISH_MONTHS)
    date = (rf"[0-9]{{1,2}}\s+(?:{month})(?:\s+[0-9]{{4}})?(?:\s+at\s+[0-9]{{1,2}}:[0-9]{{2}})?|"
            rf"(?:{_FACT_NUMBER_V2}年)?{_FACT_NUMBER_V2}月{_FACT_NUMBER_V2}(?:日|號|号)"
            rf"(?:(?:上午|下午)?{_FACT_NUMBER_V2}(?:時|时|點|点|:){_FACT_NUMBER_V2}分?)?")
    money = rf"(?:£|GBP\s+|HK\$)\s*{_FACT_NUMBER_V2}(?:\s+(?:thousand|million|billion))?|{_FACT_NUMBER_V2}(?:英鎊|港元|元)"
    calendar_month = rf"(?:{month})(?:\s+[0-9]{{4}})?|(?:{_FACT_NUMBER_V2}年)?{_FACT_NUMBER_V2}月"
    core = rf"{year}|{interval}(?:\s*(?:{_FACT_UNIT_V2}))?|{date}|{money}|{quantity}|{selection}|{calendar_month}"
    matches = [*re.finditer(rf"(?<![A-Za-z0-9_])(?:{_FACT_QUALIFIER_V2})?(?:{core})(?:半)?(?:(?:期限)?內)?(?![A-Za-z0-9_])", text, re.I), *re.finditer(pattern, text)]
    # Retain finite unparsed measurement units, not arbitrary following nouns.
    # Noun fidelity belongs to the separate semantic support check.
    raw_unit = (r"(?:kg|mg|g|km|cm|mm|m|ml|l|(?:kilo|milli)?grams?|"
                r"(?:kilo|centi|milli)?met(?:res?|ers?)|(?:milli)?lit(?:res?|ers?)|"
                r"tonnes?|tons?|degrees?|seconds?|days?|weeks?)(?![A-Za-z])|"
                r"%|％|公里|公斤|公噸|噸|吨|毫升|米|歲|度|呎|℃|℉|°(?:[CFcf])?|半")
    raw = rf"(?<![A-Za-z0-9_])[+−-]?[0-9]+(?:[.,][0-9]+)*(?:/[0-9]+|\s*(?:{raw_unit}))?"
    matches.extend(re.finditer(raw, text, re.I))
    matches.extend(re.finditer(rf"(?<![A-Za-z0-9_]){_FACT_NUMBER_V2}\s*(?:{raw_unit})", text, re.I))
    cardinal = (r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion")
    matches.extend(re.finditer(rf"(?<![A-Za-z])(?:{cardinal})(?:\s+[A-Za-z]+)?(?![A-Za-z])", text, re.I))
    matches.extend(re.finditer(rf"{_FACT_NUMBER_V2}(?:個|个|名|間|间|所|輛|辆|部|條|条|項|项|次|人|座|期|倍|成)", text))
    # Reuse the finite legacy lexicon, but retain unparsed relative facts as
    # unknowns. Only supplied derivation may replace a complete next-year span.
    matches.extend(_RELATIVE_TIME_FACT.finditer(text))
    matches.extend(re.finditer(r"[上下本]個月|明天|昨天", text))
    selected = []
    for match in sorted(matches, key=lambda item: (-(item.end() - item.start()), item.start())):
        if not any(match.start() < end and start < match.end() for start, end, _fact in selected):
            if match.start() and text[match.start()-1] == "同" and match.group().startswith("一"):
                continue  # 同一 is co-reference, not an asserted cardinality.
            fact = canonical_localised_fact_v2(match.group())
            if re.fullmatch(rf"(?:{month})", match.group(), re.I) and not _calendar_month_occurs(match.group(), text):
                continue
            if fact and fact[0] == "CALENDAR_MONTH" and not _calendar_month_occurs(match.group(), text):
                continue
            if match.start() and text[match.start()-1] in "+−-£$€¥負负" and not match.group().startswith(tuple("+−-")):
                fact = None
            selected.append((match.start(), match.end(), fact))
    for index, character in enumerate(text):
        if character == "一" and index and text[index - 1] == "同":
            continue  # Preserve only the explicit 同一 co-reference exemption.
        # Han/financial numerals must not disappear merely because a unit or
        # fraction form is outside the closed grammar. No value is inferred.
        if character.isnumeric() and not any(start <= index < end for start, end, _fact in selected):
            selected.append((index, index + 1, None))
    return tuple(sorted(selected))


def localised_fact_is_bound_v2(source, target, claim, excerpt, rendered) -> bool:
    fact = canonical_localised_fact_v2(source)
    if fact is None or fact != canonical_localised_fact_v2(target):
        return False
    def occurs(expression, text):
        return any(text[start:end] == expression and item == fact
                   for start, end, item in _factual_occurrences_v2(text))
    return (occurs(source, claim) or occurs(source, excerpt)) and occurs(target, rendered)


def factual_rendering_is_bound_v2(source, rendered, pairs=(), *, literals=(), derived_pairs=()) -> bool:
    """Compare ordered complete typed occurrences, never infer source units."""
    return _factual_rendering_is_bound(source, rendered, pairs, literals=literals,
        derived_pairs=derived_pairs, occurrences=lambda text, _source: _factual_occurrences_v2(text))


def _source_bound_occurrences(text, source, source_side):
    """Complete grammatical expressions over selected Source, not numeral waivers."""
    from datetime import date
    legacy = list(_factual_occurrences_v2(text))
    overlays, lexical = [], []
    qualifier = re.compile(r'(?:only|exactly|at least|at most|more than|less than|up to|about|approximately|around|'
        r'只有|只限|只|僅(?:限|僅)?|仅(?:限|仅)?|恰好|正好|至少|最少|不少於|最多|至多|不多於|超過|多於|少於|不足|大約|約)\s*$', re.I)
    def preceding(start):
        return qualifier.search(text[:start])
    def following(end):
        return re.match(r'[^，。；、\n]{0,24}(?:而已|為限|为限)(?=$|[，。；、\n])', text[end:])
    def complete_boundary(start, end):
        return (not (start and text[start - 1].isnumeric())
            and not text[:start].rstrip().endswith(('+', '−', '-', '負', '负', '£', '$', '€', '¥'))
            and not (end < len(text) and (text[end].isnumeric() or text[end] == '半')))
    def add(start, end, fact):
        if not complete_boundary(start, end):
            return
        prefix = preceding(start)
        if prefix:
            bound = canonical_localised_fact_v2(prefix.group() + '1')
            fact = ('BOUND', bound[1], fact) if bound and bound[0] == 'BOUND' and fact is not None else None
            start = prefix.start()
        if following(end):
            fact = None  # A post-nominal restriction is not an indefinite article.
        if not any(start < right and left < end for left, right, _ in overlays):
            overlays.append((start, end, fact))
    weekdays = 'Monday Tuesday Wednesday Thursday Friday Saturday Sunday'.split()
    pattern = re.compile(r'\b(?:' + '|'.join(weekdays) + r')\b|(?:星期|週|周)[一二三四五六日天]', re.I)
    weekday_rows, used = list(pattern.finditer(text)), set()
    def weekday(value):
        if value.isascii():
            return [word.casefold() for word in weekdays].index(value.casefold()) + 1
        return {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '日': 7, '天': 7}[value[-1]]
    for start, end, fact in legacy:
        if not fact or fact[0] != 'DATE_TIME':
            continue
        adjacent = [match for match in weekday_rows if
            (match.end() <= start and re.fullmatch(r'\s*', text[match.end():start])) or
            (match.start() >= end and re.fullmatch(r'\s*[（(]?\s*', text[end:match.start()]))]
        if len(adjacent) == 1:
            match = adjacent[0]; number = weekday(match.group()); used.add(match.span())
            try:
                consistent = None not in fact[1:4] and date(*fact[1:4]).isoweekday() == number
            except ValueError:
                consistent = False
            add(min(start, match.start()), max(end, match.end()), ('DATED_WEEKDAY', fact, number) if consistent else None)
    for match in weekday_rows:
        if match.span() not in used:
            add(match.start(), match.end(), ('WEEKDAY', weekday(match.group())))
    for match in re.finditer(r'\b(?:mid[- ]year|year[- ]end)\b|年中|年終', text, re.I):
        add(match.start(), match.end(), ('PERIOD_PHASE', 'MID_YEAR' if re.fullmatch(r'mid[- ]year|年中', match.group(), re.I) else 'YEAR_END'))
    number = _FACT_NUMBER_V2
    counts = (rf'(?P<number>{number})\s+(?:people|persons?|teachers?|students?|learners?|trainees?|employees?|workers?|patients?|children|adults?)(?![A-Za-z])'
        if source_side else rf'(?P<number>{number})(?:名|位|人)(?=[\u3400-\u9fff]|[，。；、）)])')
    for match in re.finditer(counts, text, re.I):
        fact = canonical_localised_fact_v2(match['number'])
        if fact and fact[0] == 'NUMBER':
            add(match.start(), match.end(), ('COUNT', 'PERSON', fact[1]))
    if source_side:
        for match in re.finditer(r'\b[0-9]+\s+(?P<unit>months?|years?)\s*\(currently\s+(?P<prior>[0-9]+)\)', text, re.I):
            add(match.start('prior'), match.end('prior'), canonical_localised_fact_v2(match['prior'] + ' ' + match['unit']))
    else:
        for match in list(re.finditer(r'一部分', text))[:len(re.findall(r'\bpart of\b', source, re.I))]:
            if complete_boundary(*match.span()) and not preceding(match.start()) and not following(match.end()):
                lexical.append(match.span())
        available = len(re.findall(r'\b(?:a|an)\s+[a-z][a-z-]*', source))
        for match in re.finditer(r'一(?:個|份|項|件|種)(?=[\u3400-\u9fff])', text):
            if (available and complete_boundary(*match.span()) and not preceding(match.start()) and not following(match.end())
                    and not any(match.start() < end and start < match.end() and fact is not None for start, end, fact in legacy)
                    and not any(match.start() < end and start < match.end() for start, end in lexical)):
                lexical.append(match.span()); available -= 1
    spans = [(start, end) for start, end, _ in overlays] + lexical
    retained = []
    for row in legacy:
        overlapping = [(start, end) for start, end in spans if row[0] < end and start < row[1]]
        if not overlapping:
            retained.append(row)
        elif not any(start <= row[0] and row[1] <= end for start, end in overlapping):
            # No new binding may erase the uncovered part of a quantity.
            return ((row[0], row[1], None),)
    return tuple(sorted(retained + overlays))


def factual_rendering_is_bound_v3(source, rendered, pairs=(), *, literals=(), derived_pairs=()) -> bool:
    """Source-bound occurrence consumer; historical v4 producer slots stay fixed."""
    if type(source) is not str:
        return False
    context = source
    if isinstance(literals, (tuple, list)):
        for literal in literals:
            if type(literal) is str and literal:
                context = re.sub(_entity_pattern(literal), lambda match: ' ' * len(match.group()), context)
    return _factual_rendering_is_bound(source, rendered, pairs, literals=literals,
        derived_pairs=derived_pairs, occurrences=lambda text, source_side: _source_bound_occurrences(text, context, source_side))


def _factual_rendering_is_bound(source, rendered, pairs, *, literals, derived_pairs, occurrences):
    if type(source) is not str or type(rendered) is not str:
        return False
    for rows in (pairs, derived_pairs):
        if not isinstance(rows, (tuple, list)) or any(not isinstance(row, (tuple, list)) or len(row) != 2
            or any(type(part) is not str or not part.strip() for part in row) for row in rows):
            return False
        if len({row[0] for row in rows}) != len(rows) or len({row[1] for row in rows}) != len(rows):
            return False
    if not isinstance(literals, (tuple, list)) or any(type(value) is not str or not value.strip() for value in literals) or len(set(literals)) != len(literals):
        return False
    original = (source, rendered)
    masked = [source, rendered]
    literal_spans = [[], []]
    for literal in literals:
        if type(literal) is not str or not literal.strip() or canonical_localised_fact_v2(literal) is not None:
            return False
        spans = [tuple(re.finditer(_entity_pattern(literal), text)) for text in original]
        if not spans[0] or len(spans[0]) != len(spans[1]):
            return False
        for column, matches in enumerate(spans):
            literal_spans[column].extend((match.start(), match.end()) for match in matches)
    for column, spans in enumerate(literal_spans):
        for start, end, fact in occurrences(original[column], column == 0):
            if any(left < end and start < right for left, right in spans) and not any(left <= start and end <= right for left, right in spans):
                if fact is not None or any((original[column][index].isnumeric() or original[column][index] == "半")
                    and not any(left <= index < right for left, right in spans) for index in range(start, end)):
                    return False
        for start, end in spans:
            masked[column] = masked[column][:start] + " " * (end-start) + masked[column][end:]
    source, rendered = masked
    derived = {tuple(row) for row in derived_pairs}
    if not derived.issubset({tuple(row) for row in pairs}):
        return False
    if any((left, right) not in derived and not localised_fact_is_bound_v2(left, right, source, source, rendered) for left, right in pairs):
        return False
    before, after = occurrences(source, True), occurrences(rendered, False)
    before, after = list(before), list(after)
    for left, right in derived:
        if not re.fullmatch(r"next year", left, re.I) or not re.fullmatch(r"[0-9]{4}年", right) or not 1 <= int(right[:-1]) <= 9999:
            return False
        source_matches = tuple(re.finditer(_entity_pattern(left), source))
        target_matches = tuple(re.finditer(_entity_pattern(right), rendered))
        if not source_matches or len(source_matches) != len(target_matches):
            return False
        for matches, stream in ((source_matches, before), (target_matches, after)):
            for match in matches:
                overlapping = [(start, end, fact) for start, end, fact in stream if match.start() < end and start < match.end()]
                if any(start < match.start() or end > match.end() for start, end, _fact in overlapping):
                    return False
                stream[:] = [row for row in stream if row not in overlapping]
                stream.append((match.start(), match.end(), ("SOURCE_DERIVED_YEAR", int(right[:-1]))))
        before.sort(); after.sort()
    return all(item is not None for *_bounds, item in (*before, *after)) and tuple(
        item for *_bounds, item in before) == tuple(item for *_bounds, item in after)


class Evid012QualificationTest(StrEnum):
    LAW_RIGHT_STATUS_POLICY = "LAW_RIGHT_STATUS_POLICY"
    SAFETY_OR_PUBLIC_HEALTH = "SAFETY_OR_PUBLIC_HEALTH"
    ESSENTIAL_SERVICE_DISRUPTION = "ESSENTIAL_SERVICE_DISRUPTION"
    HOUSEHOLD_PRACTICAL_EFFECT = "HOUSEHOLD_PRACTICAL_EFFECT"
    OFFICIAL_ACTION_OR_DEADLINE = "OFFICIAL_ACTION_OR_DEADLINE"
    EXCEPTIONAL_PUBLIC_IMPORTANCE = "EXCEPTIONAL_PUBLIC_IMPORTANCE"


class GovernedClaimStatus(StrEnum):
    CONFIRMED_FACT = "CONFIRMED_FACT"
    EXPRESSLY_PROVISIONAL_FACT = "EXPRESSLY_PROVISIONAL_FACT"
    ATTRIBUTED_CLAIM_OR_OPINION = "ATTRIBUTED_CLAIM_OR_OPINION"
    PUBLISHED_ANALYSIS_OR_FORECAST = "PUBLISHED_ANALYSIS_OR_FORECAST"
    CONTEXTUAL_BACKGROUND = "CONTEXTUAL_BACKGROUND"


class ClaimAuthorityClass(StrEnum):
    RESPONSIBLE_PRIMARY = "RESPONSIBLE_PRIMARY"
    INDEPENDENT_RELIABLE = "INDEPENDENT_RELIABLE"


SOURCE_RENDERING_CONTRACT = 'newsroom.source-qualified-rendering.v1+newsroom.native-source-term-bindings.v1'
SOURCE_RENDERING_CONTRACT_V2 = 'newsroom.source-qualified-rendering.v2+newsroom.native-source-term-bindings.v2'
SOURCE_RENDERING_CONTRACT_V3 = 'newsroom.source-qualified-rendering.v3+newsroom.native-source-term-bindings.v2'


def source_rendering_reference(ref):
    # The locator authenticates a SourceQA parent, never model-supplied truth.
    if type(ref) is not tuple or any(type(p) is not tuple or len(p) != 2 for p in ref):
        raise ValueError('Source rendering reference differs')
    value = dict(ref)
    if (len(value) != len(ref) or set(value) != {'contract','operation','invocation_id','raw_admission_id','receipt_admission_id'}
            or value.get('contract') not in {SOURCE_RENDERING_CONTRACT, SOURCE_RENDERING_CONTRACT_V2, SOURCE_RENDERING_CONTRACT_V3}
            or value.get('operation') != 'SOURCE_RENDERING'):
        raise ValueError('Source rendering reference differs')
    semantic_witness_reference(tuple(sorted({**{k:v for k,v in value.items() if k != 'operation'},
        'contract':SEMANTIC_WITNESS_CONTRACT,'question_id':'criterion'}.items())))
    return value


@dataclass(frozen=True, slots=True)
class GovernedClaimEvidence:
    claim_id: str
    claim: str
    passage_index: int
    supporting_excerpt: str
    source_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_authority_decision_ids: tuple[str, ...]
    rights_decision_ids: tuple[str, ...]
    dependency_evidence_ids: tuple[str, ...]
    evidential_origin_ids: tuple[str, ...]
    authority_class: ClaimAuthorityClass
    authority_scope: str
    status: GovernedClaimStatus
    attribution: str
    rendered_assertion_zh_hant_hk: str
    claim_role: Literal["HEADLINE", "SUBSTANTIVE", "CONTEXT"]
    semantic_relation_evidence_id: str
    localised_factual_expressions: tuple[tuple[str, str], ...] = ()
    named_entity_evidence: tuple[tuple[str, str, str], ...] = ()
    named_entities: tuple[str, ...] = ()
    rendered_named_entities: tuple[str, ...] = ()
    quotations: tuple[str, ...] = ()
    certainty: Literal["CONFIRMED"] = "CONFIRMED"
    originality_basis: Literal["FACTUAL_REWRITE_REQUIRED"] = "FACTUAL_REWRITE_REQUIRED"
    originality_policy_version: str = ORIGINALITY_POLICY_VERSION
    admitted_use: Literal["PUBLICATION_EVIDENCE"] = "PUBLICATION_EVIDENCE"
    policy_version: str = GOVERNED_CLAIM_POLICY_VERSION
    source_rendering_ref: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.source_rendering_ref:
            source_rendering_reference(self.source_rendering_ref)
        elif self.source_rendering_ref != ():
            raise ValueError('Source rendering reference must be immutable')
        required = (
            self.claim_id,
            self.claim,
            self.supporting_excerpt,
            self.authority_scope,
            self.attribution,
            self.rendered_assertion_zh_hant_hk,
            self.semantic_relation_evidence_id,
        )
        if any(not isinstance(value, str) or not value.strip() for value in required):
            raise ValueError("governed claim evidence fields are required")
        if (
            not isinstance(self.passage_index, int)
            or isinstance(self.passage_index, bool)
            or self.passage_index < 0
        ):
            raise ValueError("governed claim passage index must be non-negative")
        if any(
            not values
            for values in (
                self.source_ids,
                self.source_record_ids,
                self.source_authority_decision_ids,
                self.rights_decision_ids,
                self.dependency_evidence_ids,
                self.evidential_origin_ids,
            )
        ):
            raise ValueError(
                "governed claim requires source, authority, rights and dependency provenance"
            )
        if any(
            not isinstance(value, str) or not value.strip()
            for values in (
                self.source_ids,
                self.source_record_ids,
                self.source_authority_decision_ids,
                self.rights_decision_ids,
                self.dependency_evidence_ids,
                self.evidential_origin_ids,
                self.named_entities,
                self.quotations,
            )
            for value in values
        ):
            raise ValueError("governed claim provenance values must be strings")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("governed claim source IDs must be unique")
        if any(
            len(set(values)) != len(values)
            for values in (
                self.source_record_ids,
                self.source_authority_decision_ids,
                self.rights_decision_ids,
                self.dependency_evidence_ids,
            )
        ):
            raise ValueError("governed claim provenance IDs must be unique")
        if len(set(self.evidential_origin_ids)) != len(self.evidential_origin_ids):
            raise ValueError("governed claim evidential origins must be unique")
        entity_texts = tuple(item[0] for item in self.named_entity_evidence)
        entity_types = frozenset(
            {
                "PERSON",
                "ORGANISATION",
                "PLACE",
                "OFFICIAL_TITLE",
                "OFFICIAL_TERM",
                "PRODUCT",
                *({"SOURCE_LITERAL"} if self.source_rendering_ref else set()),
            }
        )
        if (
            entity_texts != self.named_entities
            or self.rendered_named_entities != self.named_entities
            or len(set(entity_texts)) != len(entity_texts)
            or any(
                len(item) != 3
                or any(
                    not isinstance(value, str) or not value.strip() for value in item
                )
                or item[1] not in entity_types
                for item in self.named_entity_evidence
            )
            or len({item[2] for item in self.named_entity_evidence})
            != len(self.named_entity_evidence)
            or any(
                text in {self.claim, self.supporting_excerpt}
                or len(text) > 80
                or (
                    re.search(r"[\n。！？!?；;：:]", text)
                    and not (
                        entity_type == "OFFICIAL_TERM"
                        and _SOURCE_BOUND_IMMIGRATION_PART_REFERENCE.fullmatch(text)
                    )
                )
                or (not (self.source_rendering_ref and entity_type == "SOURCE_LITERAL" and text in self.claim)
                    and not _has_bounded_named_entity_shape(
                        text, entity_type, source_context=self.claim))
                for text, entity_type, _record_id in self.named_entity_evidence
            )
            or any(
                not isinstance(text, str)
                or not text.strip()
                or len(text) > 80
                or (
                    re.search(r"[\n。！？!?；;：:]", text)
                    and not _SOURCE_BOUND_IMMIGRATION_PART_REFERENCE.fullmatch(text)
                )
                for text in self.rendered_named_entities
            )
        ):
            raise ValueError("named entities require exact typed retained evidence")
        if any(
            not isinstance(item, (tuple, list))
            or len(item) != 2
            or any(not isinstance(value, str) or not value.strip() for value in item)
            for item in self.localised_factual_expressions
        ):
            raise ValueError(
                "localised factual expressions must be source-target pairs"
            )
        localised_sources = tuple(
            source for source, _target in self.localised_factual_expressions
        )
        localised_targets = tuple(
            target for _source, target in self.localised_factual_expressions
        )
        fact_is_bound = (localised_fact_is_bound_v2 if self.source_rendering_ref
            and dict(self.source_rendering_ref)['contract'] in {SOURCE_RENDERING_CONTRACT_V2, SOURCE_RENDERING_CONTRACT_V3}
            else _localised_fact_is_bound)
        if (
            len(set(localised_sources)) != len(localised_sources)
            or len(set(localised_targets)) != len(localised_targets)
            or any(
                not fact_is_bound(
                    source, target, self.claim, self.supporting_excerpt,
                    self.rendered_assertion_zh_hant_hk,
                ) and not (self.source_rendering_ref
                    and re.fullmatch(r"next year", source, re.I)
                    and re.fullmatch(r"[0-9]{4}年", target)
                    and source in self.claim and target in self.rendered_assertion_zh_hant_hk)
                for source, target in self.localised_factual_expressions
            )
        ):
            raise ValueError(
                "localised factual expressions must bind equivalent exact claim facts"
            )
        if self.admitted_use != "PUBLICATION_EVIDENCE":
            raise ValueError("governed claim is not admitted for publication evidence")
        if self.claim_role not in {"HEADLINE", "SUBSTANTIVE", "CONTEXT"}:
            raise ValueError("governed claim role is not supported")
        if self.certainty != "CONFIRMED":
            raise ValueError("governed claim certainty is not supported")
        if self.originality_basis != "FACTUAL_REWRITE_REQUIRED":
            raise ValueError("governed claim originality basis is not supported")
        if self.policy_version != GOVERNED_CLAIM_POLICY_VERSION:
            raise ValueError("governed claim policy version is not supported")
        if self.originality_policy_version != ORIGINALITY_POLICY_VERSION:
            raise ValueError(
                "governed claim originality policy version is not supported"
            )
        if self.rendered_assertion_zh_hant_hk in {
            self.claim,
            self.supporting_excerpt,
        }:
            raise ValueError("governed claim rendering must be an original assertion")


@dataclass(frozen=True, slots=True)
class EvidenceGateEvidence:
    gate: Literal["CLAIM_TRACEABILITY", "EVIDENCE_SUFFICIENCY", "SOURCE_AUTHORITY"]
    result: Literal["PASS"]
    governed_claim_ids: tuple[str, ...]
    policy_version: str = EVIDENCE_GATE_POLICY_VERSION

    def __post_init__(self) -> None:
        if (
            self.gate
            not in {
                "CLAIM_TRACEABILITY",
                "EVIDENCE_SUFFICIENCY",
                "SOURCE_AUTHORITY",
            }
            or self.result != "PASS"
        ):
            raise ValueError("evidence gate or result is not supported")
        if not self.governed_claim_ids or any(
            not isinstance(value, str) or not value.strip()
            for value in self.governed_claim_ids
        ):
            raise ValueError("evidence gate requires governed claim provenance")
        if len(set(self.governed_claim_ids)) != len(self.governed_claim_ids):
            raise ValueError("evidence gate claim provenance must be unique")
        if self.policy_version != EVIDENCE_GATE_POLICY_VERSION:
            raise ValueError("evidence gate policy version is not supported")


SEMANTIC_WITNESS_CONTRACT = "newsroom.qualification-semantic-witness.v1"
SEMANTIC_RESOLUTION_CONTRACT = "newsroom.qualification-semantic-resolution.v1"
SEMANTIC_RESOLUTION_CONTRACT_V2 = "newsroom.qualification-semantic-resolution.v2"
_SEMANTIC_WITNESS_REF_FIELDS = frozenset({"contract", "invocation_id", "raw_admission_id", "receipt_admission_id", "question_id"})
_SEMANTIC_RESOLUTION_REF_FIELDS = _SEMANTIC_WITNESS_REF_FIELDS | frozenset({
    "resolution_invocation_id", "resolution_raw_admission_id", "resolution_receipt_admission_id"})


def semantic_witness_reference(ref: tuple[tuple[str, str], ...]) -> dict[str, str]:
    """A closed reference is a locator, never an authenticated YES."""
    if (type(ref) is not tuple or any(type(pair) is not tuple or len(pair) != 2
            or any(type(part) is not str for part in pair) for pair in ref)):
        raise ValueError('semantic witness reference differs')
    value = dict(ref)
    resolution = value.get('contract') == SEMANTIC_RESOLUTION_CONTRACT
    if (type(ref) is not tuple or len(value) != len(ref)
            or set(value) != (_SEMANTIC_RESOLUTION_REF_FIELDS if resolution else _SEMANTIC_WITNESS_REF_FIELDS)
            or value.get('contract') not in {SEMANTIC_WITNESS_CONTRACT, SEMANTIC_RESOLUTION_CONTRACT, SEMANTIC_RESOLUTION_CONTRACT_V2}
            or not all(type(v) is str and v for v in value.values())
            or not re.fullmatch(r'sha256:[0-9a-f]{64}', value['invocation_id'])
            or len(value['question_id'].encode()) > 256
            or any(not re.fullmatch(r'[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', value[k])
                   for k in ('raw_admission_id', 'receipt_admission_id'))
            or resolution and (not re.fullmatch(r'sha256:[0-9a-f]{64}', value['resolution_invocation_id'])
                or any(not re.fullmatch(r'[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', value[k])
                    for k in ('resolution_raw_admission_id', 'resolution_receipt_admission_id')))):
        raise ValueError('semantic witness reference differs')
    return value


@dataclass(frozen=True, slots=True)
class QualificationEvidence:
    test: Evid012QualificationTest
    governed_claim_id: str
    qualification_record_id: str
    test_evidence: tuple[tuple[str, str], ...]
    policy_version: str = EVID_012_POLICY_VERSION
    semantic_witness_ref: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        try:
            canonical_test = Evid012QualificationTest(self.test)
        except ValueError:
            raise ValueError("qualification test is not in EVID-012") from None
        object.__setattr__(self, "test", canonical_test)
        if (
            not isinstance(self.governed_claim_id, str)
            or not self.governed_claim_id.strip()
            or not isinstance(self.qualification_record_id, str)
            or not self.qualification_record_id.strip()
        ):
            raise ValueError("qualification governed claim is required")
        if self.policy_version != EVID_012_POLICY_VERSION:
            raise ValueError("qualification policy version is not supported")
        if self.semantic_witness_ref:
            semantic_witness_reference(self.semantic_witness_ref)
        elif self.semantic_witness_ref != ():
            raise ValueError("semantic witness reference must be immutable")
        evidence = dict(self.test_evidence)
        if len(evidence) != len(self.test_evidence) or any(
            not key.strip() or not value.strip() for key, value in self.test_evidence
        ):
            raise ValueError("qualification test evidence must be unique and complete")
        allowed: dict[Evid012QualificationTest, dict[str, frozenset[str] | None]] = {
            Evid012QualificationTest.LAW_RIGHT_STATUS_POLICY: {
                "change_kind": frozenset(
                    {"LAW", "RIGHT", "STATUS", "OFFICIAL_DEADLINE", "PUBLIC_POLICY"}
                ),
                "event_polarity": frozenset({"AFFIRMED"}),
                "change_relation": frozenset({"NEW_OR_CHANGED_STATE"}),
                "material_relation_span": None,
                "new_state": None,
            },
            Evid012QualificationTest.SAFETY_OR_PUBLIC_HEALTH: {
                "effect_class": frozenset(
                    {
                        "INJURY_RISK",
                        "PUBLIC_HEALTH_WARNING",
                        "EVACUATION",
                        "MATERIAL_EXPOSURE",
                    }
                ),
                "event_polarity": frozenset({"AFFIRMED"}),
                "effect_relation": frozenset({"MATERIAL_EFFECT"}),
                "material_relation_span": None,
                "affected_group": None,
            },
            Evid012QualificationTest.ESSENTIAL_SERVICE_DISRUPTION: {
                "service_kind": frozenset(
                    {"TRANSPORT", "UTILITY", "SCHOOL", "WORKPLACE", "LOCALITY"}
                ),
                "event_polarity": frozenset({"AFFIRMED"}),
                "duration_relation": frozenset({"DISRUPTION_DURATION"}),
                "duration_minutes": None,
                "affected_group": None,
            },
            Evid012QualificationTest.HOUSEHOLD_PRACTICAL_EFFECT: {
                "domain": frozenset(
                    {
                        "MONEY",
                        "WORK",
                        "HOUSING",
                        "EDUCATION",
                        "HEALTHCARE",
                        "UK_HONG_KONG_TRAVEL",
                    }
                ),
                "event_polarity": frozenset({"AFFIRMED"}),
                "effect_relation": frozenset({"MATERIAL_PRACTICAL_EFFECT"}),
                "material_relation_span": None,
                "practical_effect": None,
            },
            Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE: {
                "action_class": frozenset(
                    {"INSTRUCTION", "PROCESS", "OFFICIAL_DEADLINE"}
                ),
                "event_polarity": frozenset({"AFFIRMED"}),
                "action_relation": frozenset({"NEW_OR_CHANGED_OFFICIAL_ACTION"}),
                "material_relation_span": None,
                "reader_action": None,
            },
            Evid012QualificationTest.EXCEPTIONAL_PUBLIC_IMPORTANCE: {
                "importance_class": frozenset(
                    {
                        "HONG_KONG_WIDE",
                        "INTERNATIONAL_EMERGENCY",
                        "CONSTITUTIONAL_CHANGE",
                    }
                ),
                "event_polarity": frozenset({"AFFIRMED"}),
                "importance_relation": frozenset({"CURRENT_EXCEPTIONAL_IMPORTANCE"}),
                "material_relation_span": None,
                "affected_group": None,
            },
        }
        required = allowed[canonical_test]
        if set(evidence) != set(required) or any(
            permitted is not None and evidence[field] not in permitted
            for field, permitted in required.items()
        ):
            raise ValueError("qualification test evidence does not satisfy EVID-012")
        if canonical_test is Evid012QualificationTest.ESSENTIAL_SERVICE_DISRUPTION:
            try:
                duration_minutes = int(evidence["duration_minutes"])
            except ValueError:
                raise ValueError(
                    "qualification disruption duration must be an integer"
                ) from None
            if duration_minutes < 60:
                raise ValueError(
                    "qualification disruption is below the material duration floor"
                )


@dataclass(frozen=True, slots=True)
class EvidencePackage:
    candidate_id: str
    hypothesis_id: str
    signal_ids: tuple[str, ...]
    lead_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    observation_digests: tuple[str, ...]
    passages: tuple[str, ...]
    substantive_new_information: tuple[str, ...] = ()
    governed_claims: tuple[GovernedClaimEvidence, ...] = ()
    qualification_evidence: tuple[QualificationEvidence, ...] = ()
    selection_rationale: str = ""
    geography: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    evidence_gate_results: tuple[tuple[str, str], ...] = ()
    evidence_gate_evidence: tuple[EvidenceGateEvidence, ...] = ()
    freshness_result: str = "MISSING"
    integrity_result: str = "MISSING"
    explicit_exclusions: tuple[str, ...] = ()
    resolved_evidence_records: tuple[tuple[str, str], ...] = ()
    admitted_context: GovernedContext | None = None

    def __post_init__(self) -> None:
        if not self.signal_ids or not self.lead_ids or not self.observation_digests:
            raise ValueError(
                "Evidence Package requires Signal, Lead and retained observations"
            )
        if not self.passages:
            raise ValueError("Evidence Package requires at least one retained passage")
        gate_names = tuple(name for name, _result in self.evidence_gate_results)
        if len(set(gate_names)) != len(gate_names):
            raise ValueError("Evidence Package gate names must be unique")
        if any(
            result not in {"PASS", "HOLD", "FAIL", "MISSING"}
            for _name, result in self.evidence_gate_results
        ):
            raise ValueError("Evidence Package gate result is not canonical")
        claim_ids = tuple(item.claim_id for item in self.governed_claims)
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("Evidence Package governed claim IDs must be unique")
        qualification_record_ids = tuple(
            item.qualification_record_id for item in self.qualification_evidence
        )
        qualification_logical_ids = tuple(
            (item.test, item.governed_claim_id) for item in self.qualification_evidence
        )
        if len(set(qualification_record_ids)) != len(qualification_record_ids) or len(
            set(qualification_logical_ids)
        ) != len(qualification_logical_ids):
            raise ValueError("Evidence Package qualification evidence must be unique")
        if any(
            len(set(values)) != len(values)
            for values in (
                self.substantive_new_information,
                self.geography,
                self.categories,
                self.explicit_exclusions,
            )
        ):
            raise ValueError("Evidence Package governed inventories must be unique")

    @property
    def digest(self) -> str:
        base_digest = digest_bytes(canonical_json_bytes(evidence_package_value(self)))
        if self.admitted_context is None:
            return base_digest
        return digest_bytes(
            canonical_json_bytes(
                {
                    "base_evidence_package_digest": base_digest,
                    "admitted_context": self.admitted_context.canonical_value(),
                }
            )
        )


def evidence_package_value(package: EvidencePackage) -> dict[str, object]:
    """Return the established canonical package value without store authority."""

    if type(package) is not EvidencePackage:
        raise TypeError("package must be exact EvidencePackage")
    return {
        "candidate_id": package.candidate_id,
        "hypothesis_id": package.hypothesis_id,
        "signal_ids": list(package.signal_ids),
        "lead_ids": list(package.lead_ids),
        "source_ids": list(package.source_ids),
        "observation_digests": list(package.observation_digests),
        "passages": list(package.passages),
        "substantive_new_information": list(
            package.substantive_new_information
        ),
        "governed_claims": [
            {
                "claim_id": item.claim_id,
                "claim": item.claim,
                "passage_index": item.passage_index,
                "supporting_excerpt": item.supporting_excerpt,
                "source_ids": list(item.source_ids),
                "source_record_ids": list(item.source_record_ids),
                "source_authority_decision_ids": list(
                    item.source_authority_decision_ids
                ),
                "rights_decision_ids": list(item.rights_decision_ids),
                "dependency_evidence_ids": list(
                    item.dependency_evidence_ids
                ),
                "evidential_origin_ids": list(item.evidential_origin_ids),
                "authority_class": item.authority_class.value,
                "authority_scope": item.authority_scope,
                "status": item.status.value,
                "attribution": item.attribution,
                "rendered_assertion_zh_hant_hk": (
                    item.rendered_assertion_zh_hant_hk
                ),
                "claim_role": item.claim_role,
                "semantic_relation_evidence_id": (
                    item.semantic_relation_evidence_id
                ),
                "localised_factual_expressions": [
                    list(value)
                    for value in item.localised_factual_expressions
                ],
                "named_entity_evidence": [
                    list(value) for value in item.named_entity_evidence
                ],
                "named_entities": list(item.named_entities),
                "rendered_named_entities": list(
                    item.rendered_named_entities
                ),
                "quotations": list(item.quotations),
                "certainty": item.certainty,
                "originality_basis": item.originality_basis,
                "originality_policy_version": (
                    item.originality_policy_version
                ),
                "admitted_use": item.admitted_use,
                "policy_version": item.policy_version,
                **({"source_rendering_ref":source_rendering_reference(item.source_rendering_ref)} if item.source_rendering_ref else {}),
            }
            for item in package.governed_claims
        ],
        "qualification_evidence": [
            {
                "test": item.test.value,
                "governed_claim_id": item.governed_claim_id,
                "qualification_record_id": item.qualification_record_id,
                "test_evidence": [
                    list(value) for value in item.test_evidence
                ],
                "policy_version": item.policy_version,
                **({'semantic_witness_ref': semantic_witness_reference(item.semantic_witness_ref)} if item.semantic_witness_ref else {}),
            }
            for item in package.qualification_evidence
        ],
        "selection_rationale": package.selection_rationale,
        "geography": list(package.geography),
        "categories": list(package.categories),
        "evidence_gate_results": [
            list(item) for item in package.evidence_gate_results
        ],
        "evidence_gate_evidence": [
            {
                "gate": item.gate,
                "result": item.result,
                "governed_claim_ids": list(item.governed_claim_ids),
                "policy_version": item.policy_version,
            }
            for item in package.evidence_gate_evidence
        ],
        "freshness_result": package.freshness_result,
        "integrity_result": package.integrity_result,
        "explicit_exclusions": list(package.explicit_exclusions),
        "resolved_evidence_records": [
            list(item) for item in package.resolved_evidence_records
        ],
    }



def package_for(candidate: StoryCandidateRecord) -> EvidencePackage:
    passages = tuple(
        f"{item.source_id}: {item.headline}\n{item.body}".strip()
        for item in candidate.items
    )
    return EvidencePackage(
        candidate_id=candidate.candidate_id,
        hypothesis_id=candidate.hypothesis_id,
        signal_ids=tuple(signal.signal_id for signal in candidate.signals),
        lead_ids=tuple(lead.lead_id for lead in candidate.leads),
        source_ids=tuple(sorted({item.source_id for item in candidate.items})),
        observation_digests=tuple(
            signal.observation_digest for signal in candidate.signals
        ),
        passages=passages,
        admitted_context=candidate.governed_context,
    )


def _decode_governed_package(
    candidate: StoryCandidateRecord,
    base: EvidencePackage,
    raw: str,
) -> EvidencePackage:
    package_fields = {
        "schema_version",
        "candidate_id",
        "hypothesis_id",
        "base_package_digest",
        "governed_claims",
        "substantive_new_information",
        "qualification_evidence",
        "selection_rationale",
        "geography",
        "categories",
        "evidence_gate_results",
        "evidence_gate_evidence",
        "freshness_result",
        "integrity_result",
        "explicit_exclusions",
    }
    claim_fields = {
        "claim_id",
        "claim",
        "passage_index",
        "supporting_excerpt",
        "source_ids",
        "source_record_ids",
        "source_authority_decision_ids",
        "rights_decision_ids",
        "dependency_evidence_ids",
        "evidential_origin_ids",
        "authority_class",
        "authority_scope",
        "status",
        "attribution",
        "rendered_assertion_zh_hant_hk",
        "claim_role",
        "semantic_relation_evidence_id",
        "localised_factual_expressions",
        "named_entity_evidence",
        "named_entities",
        "rendered_named_entities",
        "quotations",
        "certainty",
        "originality_basis",
        "originality_policy_version",
        "admitted_use",
        "policy_version",
    }
    qualification_fields = {
        "test",
        "governed_claim_id",
        "qualification_record_id",
        "test_evidence",
        "policy_version",
    }
    gate_fields = {"gate", "result", "governed_claim_ids", "policy_version"}

    def string_list(item: object) -> bool:
        return isinstance(item, list) and all(isinstance(value, str) for value in item)

    try:
        value = json.loads(raw)
        if (
            not isinstance(value, dict)
            or set(value) != package_fields
            or value["schema_version"] != GOVERNED_INPUT_SCHEMA_VERSION
            or value["candidate_id"] != candidate.candidate_id
            or value["hypothesis_id"] != candidate.hypothesis_id
            or value["base_package_digest"] != base.digest
            or canonical_json_bytes(value).decode("utf-8") != raw
        ):
            return base
        if (
            not isinstance(value["governed_claims"], list)
            or not isinstance(value["qualification_evidence"], list)
            or not isinstance(value["evidence_gate_evidence"], list)
            or not string_list(value["substantive_new_information"])
            or not string_list(value["geography"])
            or not string_list(value["categories"])
            or not string_list(value["explicit_exclusions"])
            or not isinstance(value["selection_rationale"], str)
            or not isinstance(value["freshness_result"], str)
            or not isinstance(value["integrity_result"], str)
            or not isinstance(value["evidence_gate_results"], list)
            or any(
                not isinstance(item, list)
                or len(item) != 2
                or not all(isinstance(part, str) for part in item)
                for item in value["evidence_gate_results"]
            )
        ):
            return base
        if (
            any(
                not isinstance(item, dict) or set(item) != claim_fields
                for item in value["governed_claims"]
            )
            or any(
                not isinstance(item, dict) or set(item) != qualification_fields
                for item in value["qualification_evidence"]
            )
            or any(
                not isinstance(item, dict) or set(item) != gate_fields
                for item in value["evidence_gate_evidence"]
            )
        ):
            return base
        if (
            any(
                not string_list(item[field])
                for item in value["governed_claims"]
                for field in (
                    "source_ids",
                    "source_record_ids",
                    "source_authority_decision_ids",
                    "rights_decision_ids",
                    "dependency_evidence_ids",
                    "evidential_origin_ids",
                    "named_entities",
                    "rendered_named_entities",
                    "quotations",
                )
            )
            or any(
                not isinstance(item["localised_factual_expressions"], list)
                or any(
                    not isinstance(part, list)
                    or len(part) != 2
                    or not all(isinstance(value, str) for value in part)
                    for part in item["localised_factual_expressions"]
                )
                for item in value["governed_claims"]
            )
            or any(
                not isinstance(item["named_entity_evidence"], list)
                or any(
                    not isinstance(part, list)
                    or len(part) != 3
                    or not all(isinstance(value, str) for value in part)
                    for part in item["named_entity_evidence"]
                )
                for item in value["governed_claims"]
            )
            or any(
                not isinstance(item["test_evidence"], list)
                or any(
                    not isinstance(part, list)
                    or len(part) != 2
                    or not all(isinstance(value, str) for value in part)
                    for part in item["test_evidence"]
                )
                for item in value["qualification_evidence"]
            )
            or any(
                not string_list(item["governed_claim_ids"])
                for item in value["evidence_gate_evidence"]
            )
        ):
            return base
        claims = tuple(
            GovernedClaimEvidence(
                claim_id=item["claim_id"],
                claim=item["claim"],
                passage_index=item["passage_index"],
                supporting_excerpt=item["supporting_excerpt"],
                source_ids=tuple(item["source_ids"]),
                source_record_ids=tuple(item["source_record_ids"]),
                source_authority_decision_ids=tuple(
                    item["source_authority_decision_ids"]
                ),
                rights_decision_ids=tuple(item["rights_decision_ids"]),
                dependency_evidence_ids=tuple(item["dependency_evidence_ids"]),
                evidential_origin_ids=tuple(item["evidential_origin_ids"]),
                authority_class=ClaimAuthorityClass(item["authority_class"]),
                authority_scope=item["authority_scope"],
                status=GovernedClaimStatus(item["status"]),
                attribution=item["attribution"],
                rendered_assertion_zh_hant_hk=item["rendered_assertion_zh_hant_hk"],
                claim_role=item["claim_role"],
                semantic_relation_evidence_id=item["semantic_relation_evidence_id"],
                localised_factual_expressions=tuple(
                    tuple(value) for value in item["localised_factual_expressions"]
                ),
                named_entity_evidence=tuple(
                    tuple(value) for value in item["named_entity_evidence"]
                ),
                named_entities=tuple(item["named_entities"]),
                rendered_named_entities=tuple(item["rendered_named_entities"]),
                quotations=tuple(item["quotations"]),
                certainty=item["certainty"],
                originality_basis=item["originality_basis"],
                originality_policy_version=item["originality_policy_version"],
                admitted_use=item["admitted_use"],
                policy_version=item["policy_version"],
            )
            for item in value["governed_claims"]
        )
        return EvidencePackage(
            candidate_id=base.candidate_id,
            hypothesis_id=base.hypothesis_id,
            signal_ids=base.signal_ids,
            lead_ids=base.lead_ids,
            source_ids=base.source_ids,
            observation_digests=base.observation_digests,
            passages=base.passages,
            substantive_new_information=tuple(value["substantive_new_information"]),
            governed_claims=claims,
            qualification_evidence=tuple(
                QualificationEvidence(
                    Evid012QualificationTest(item["test"]),
                    item["governed_claim_id"],
                    item["qualification_record_id"],
                    tuple(tuple(value) for value in item["test_evidence"]),
                    item["policy_version"],
                )
                for item in value["qualification_evidence"]
            ),
            selection_rationale=value["selection_rationale"],
            geography=tuple(value["geography"]),
            categories=tuple(value["categories"]),
            evidence_gate_results=tuple(
                tuple(item) for item in value["evidence_gate_results"]
            ),
            evidence_gate_evidence=tuple(
                EvidenceGateEvidence(
                    item["gate"],
                    item["result"],
                    tuple(item["governed_claim_ids"]),
                    item["policy_version"],
                )
                for item in value["evidence_gate_evidence"]
            ),
            freshness_result=value["freshness_result"],
            integrity_result=value["integrity_result"],
            explicit_exclusions=tuple(value["explicit_exclusions"]),
            admitted_context=base.admitted_context,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return base


def retained_package_for(
    candidate: StoryCandidateRecord,
    *,
    proving_store: str,
) -> EvidencePackage:
    """Load one controller-approved sidecar package; source content cannot mint it."""

    base = package_for(candidate)
    approval_key = os.environ.get("NEWSROOM_EVIDENCE_APPROVAL_KEY", "").encode("utf-8")
    if len(approval_key) < 32:
        return base
    connection = sqlite3.connect(proving_store)
    try:
        connection.execute("PRAGMA query_only=ON")
        existing_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                "('proving_write_evidence_packages','proving_write_evidence_records')"
            )
        }
        if existing_tables != {
            "proving_write_evidence_packages",
            "proving_write_evidence_records",
        }:
            return base
        row = connection.execute(
            "SELECT package_json, package_json_digest, approval_status, "
            "approval_record_json, approval_signature "
            "FROM proving_write_evidence_packages WHERE candidate_id=?",
            (candidate.candidate_id,),
        ).fetchone()
        if row is None or row[2] != "APPROVED" or not isinstance(row[0], str):
            return base
        raw = row[0]
        approval_raw = row[3]
        if (
            not isinstance(approval_raw, str)
            or row[1] != digest_bytes(raw.encode("utf-8"))
            or not hmac.compare_digest(
                str(row[4]),
                hmac.new(
                    approval_key,
                    approval_raw.encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest(),
            )
        ):
            return base
        try:
            approval = json.loads(approval_raw)
        except json.JSONDecodeError:
            return base
        package = _decode_governed_package(candidate, base, raw)
        if package is base:
            return base
        records = _resolve_governed_records(connection, candidate, base, package)
        if records is None:
            return base
        record_set_digest = digest_bytes(
            canonical_json_bytes({"records": [list(item) for item in records]})
        )
        if canonical_json_bytes(approval).decode(
            "utf-8"
        ) != approval_raw or approval != {
            "base_package_digest": base.digest,
            "candidate_id": candidate.candidate_id,
            "controller_principal": EVIDENCE_APPROVAL_PRINCIPAL,
            "decision": "APPROVED",
            "evidence_record_set_digest": record_set_digest,
            "hypothesis_id": candidate.hypothesis_id,
            "package_json_digest": row[1],
            "policy_version": EVIDENCE_APPROVAL_POLICY_VERSION,
        }:
            return base
        return replace(package, resolved_evidence_records=records)
    finally:
        connection.close()


def _expected_governed_record_types(
    package: EvidencePackage,
) -> dict[str, str] | None:
    expected_types: dict[str, str] = {}
    for claim in package.governed_claims:
        for record_type, record_ids in (
            ("SOURCE_RECORD", claim.source_record_ids),
            ("SOURCE_AUTHORITY_DECISION", claim.source_authority_decision_ids),
            ("RIGHTS_DECISION", claim.rights_decision_ids),
            ("DEPENDENCY_EVIDENCE", claim.dependency_evidence_ids),
        ):
            for record_id in record_ids:
                existing_type = expected_types.setdefault(record_id, record_type)
                if existing_type != record_type:
                    return None
        existing_type = expected_types.setdefault(
            claim.semantic_relation_evidence_id, "SEMANTIC_RELATION_EVIDENCE"
        )
        if existing_type != "SEMANTIC_RELATION_EVIDENCE":
            return None
    for qualification in package.qualification_evidence:
        existing_type = expected_types.setdefault(
            qualification.qualification_record_id, "QUALIFICATION_EVIDENCE"
        )
        if existing_type != "QUALIFICATION_EVIDENCE":
            return None
    for claim in package.governed_claims:
        for _text, _entity_type, record_id in claim.named_entity_evidence:
            existing_type = expected_types.setdefault(
                record_id, "NAMED_ENTITY_EVIDENCE"
            )
            if existing_type != "NAMED_ENTITY_EVIDENCE":
                return None
    if not expected_types:
        return None
    return expected_types


def validate_governed_evidence_records(
    *,
    candidate_id: str,
    source_inventory: tuple[tuple[str, str], ...],
    base_package_digest: str,
    package: EvidencePackage,
    retained_records: Sequence[tuple[object, object, object, object]],
) -> tuple[tuple[str, str], ...] | None:
    """Validate canonical, independently governed records without store coupling."""

    def record_id_set(value: object) -> set[str] | None:
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            return None
        return set(value)

    def has_exact_source_ids(
        record_ids: tuple[str, ...], expected_source_ids: tuple[str, ...]
    ) -> bool:
        source_ids = tuple(
            records[record_id].get("source_id") for record_id in record_ids
        )
        return all(isinstance(source_id, str) for source_id in source_ids) and set(
            source_ids
        ) == set(expected_source_ids)

    expected_types = _expected_governed_record_types(package)
    if expected_types is None:
        return None
    rows = tuple(retained_records)
    if any(
        not isinstance(row, (tuple, list))
        or len(row) != 4
        or type(row[0]) is not str
        for row in rows
    ):
        return None
    row_ids = tuple(row[0] for row in rows)
    if (
        len(rows) != len(expected_types)
        or len(set(row_ids)) != len(row_ids)
        or set(row_ids) != set(expected_types)
    ):
        return None
    records: dict[str, dict[str, object]] = {}
    digests: list[tuple[str, str]] = []
    for record_id, record_type, record_raw, record_digest in rows:
        if (
            expected_types.get(record_id) != record_type
            or not isinstance(record_raw, str)
            or record_digest != digest_bytes(record_raw.encode("utf-8"))
        ):
            return None
        try:
            record = json.loads(record_raw)
        except json.JSONDecodeError:
            return None
        if (
            canonical_json_bytes(record).decode("utf-8") != record_raw
            or record.get("record_id") != record_id
            or record.get("record_type") != record_type
            or record.get("candidate_id") != candidate_id
            or record.get("base_package_digest") != base_package_digest
            or record.get("status") != "CURRENT"
            or (not _source_body_provenance_is_bound(record) if record_type=='SOURCE_RECORD'
                else set(record) != (_QUALIFICATION_RECORD_FIELDS | {'semantic_witness_ref'}
                    if record_type == 'QUALIFICATION_EVIDENCE' and record.get('semantic_witness_ref') is not None
                    else _RECORD_FIELDS_BY_TYPE.get(record_type)))
        ):
            return None
        records[record_id] = record
        digests.append((record_id, record_digest))
    source_urls = set(source_inventory)
    required_source_record_string_fields = (
        "source_id",
        "canonical_url",
        "publisher",
        "responsible_body",
        "source_type",
        "authority_class",
        "publication_time",
        "retrieval_time",
        "geography",
        "language",
        "rights_decision_id",
        "originating_report_id",
        "originating_artefact_digest",
    )
    for claim in package.governed_claims:
        if claim.passage_index >= len(source_inventory):
            return None
        source_records = [records[record_id] for record_id in claim.source_record_ids]
        if not has_exact_source_ids(claim.source_record_ids, claim.source_ids):
            return None
        passage_source_id, passage_url = source_inventory[claim.passage_index]
        if not any(
            record.get("source_id") == passage_source_id
            and record.get("canonical_url") == passage_url
            for record in source_records
        ):
            return None
        if any(
            (
                records[record_id].get("source_id"),
                records[record_id].get("canonical_url"),
            )
            not in source_urls
            or records[record_id].get("extraction_status") != "COMPLETE"
            or any(
                not isinstance(records[record_id].get(field), str)
                or not str(records[record_id][field]).strip()
                for field in required_source_record_string_fields
            )
            or not _source_body_provenance_is_bound(records[record_id])
            or records[record_id].get("source_type")
            not in _PUBLICATION_EVIDENCE_SOURCE_TYPES
            or records[record_id].get("authority_class") != claim.authority_class.value
            for record_id in claim.source_record_ids
        ):
            return None
        for source_record in source_records:
            publication_time = _parse_iso_datetime(source_record["publication_time"])
            retrieval_time = _parse_iso_datetime(source_record["retrieval_time"])
            if (
                publication_time is None
                or retrieval_time is None
                or (publication_time.tzinfo is None) != (retrieval_time.tzinfo is None)
                or retrieval_time < publication_time
                or source_record["geography"] not in {"UK", "Hong Kong", "Global"}
                or source_record["language"] not in {"en", "en-GB", "zh-Hant-HK"}
                or (
                    package.geography
                    and source_record["geography"] not in {*package.geography, "Global"}
                )
            ):
                return None
        source_rights_id_values = tuple(
            record.get("rights_decision_id") for record in source_records
        )
        if not all(isinstance(rights_id, str) for rights_id in source_rights_id_values):
            return None
        source_rights_ids = set(source_rights_id_values)
        source_dependency_ids: set[str] = set()
        for record in source_records:
            raw_dependency_ids = record.get("dependency_evidence_ids")
            dependency_ids = record_id_set(raw_dependency_ids)
            if (
                dependency_ids is None
                or not isinstance(raw_dependency_ids, list)
                or not dependency_ids
                or len(dependency_ids) != len(raw_dependency_ids)
                or any(not item.strip() for item in dependency_ids)
            ):
                return None
            source_dependency_ids.update(dependency_ids)
        if source_rights_ids != set(
            claim.rights_decision_ids
        ) or source_dependency_ids != set(claim.dependency_evidence_ids):
            return None
        if any(
            records[record_id].get("source_id") not in claim.source_ids
            or records[record_id].get("decision") != "ADMITTED"
            or records[record_id].get("authority_class") != claim.authority_class.value
            or records[record_id].get("authority_scope") != claim.authority_scope
            or records[record_id].get("governed_claim_id") != claim.claim_id
            or records[record_id].get("claim_digest")
            != digest_bytes(claim.claim.encode("utf-8"))
            for record_id in claim.source_authority_decision_ids
        ):
            return None
        if not has_exact_source_ids(
            claim.source_authority_decision_ids, claim.source_ids
        ):
            return None
        if any(
            records[record_id].get("source_id") not in claim.source_ids
            or records[record_id].get("decision") != "PERMITTED"
            or records[record_id].get("permitted_use") != "PUBLICATION_EVIDENCE"
            for record_id in claim.rights_decision_ids
        ):
            return None
        if not has_exact_source_ids(claim.rights_decision_ids, claim.source_ids):
            return None
        if any(
            records[record_id].get("source_id") not in claim.source_ids
            or records[record_id].get("dependency_status") != "RESOLVED"
            or records[record_id].get("evidential_origin_id")
            not in claim.evidential_origin_ids
            or not records[record_id].get("originating_report_id")
            for record_id in claim.dependency_evidence_ids
        ):
            return None
        for index, (text, entity_type, record_id) in enumerate(
            claim.named_entity_evidence
        ):
            rendered_text = claim.rendered_named_entities[index]
            record = records[record_id]
            entity_policy = record.get("policy_version")
            if entity_policy not in ({source_rendering_reference(claim.source_rendering_ref)['contract']} if entity_type == "SOURCE_LITERAL" and claim.source_rendering_ref
                    else {NAMED_ENTITY_POLICY_VERSION_V15, NAMED_ENTITY_POLICY_VERSION}):
                return None
            if record != {
                "base_package_digest": base_package_digest,
                "candidate_id": candidate_id,
                "canonical_entity_id": digest_bytes(f"{entity_type}:{text}".encode()),
                "entity_type": entity_type,
                "evidence_span_digest": digest_bytes(text.encode("utf-8")),
                "governed_claim_id": claim.claim_id,
                "policy_version": entity_policy,
                "record_id": record_id,
                "record_type": "NAMED_ENTITY_EVIDENCE",
                "rendered_span_digest": digest_bytes(rendered_text.encode("utf-8")),
                "rendered_text": rendered_text,
                "source_record_ids": list(claim.source_record_ids),
                "status": "CURRENT",
                "text": text,
            }:
                return None
        semantic_record = records[claim.semantic_relation_evidence_id]
        if semantic_record != {
            "base_package_digest": base_package_digest,
            "candidate_id": candidate_id,
            "claim_digest": digest_bytes(claim.claim.encode("utf-8")),
            "governed_claim_id": claim.claim_id,
            "record_id": claim.semantic_relation_evidence_id,
            "record_type": "SEMANTIC_RELATION_EVIDENCE",
            "relation": "SEMANTICALLY_EQUIVALENT",
            "rendered_assertion_digest": digest_bytes(
                claim.rendered_assertion_zh_hant_hk.encode("utf-8")
            ),
            "rendered_modality": "ASSERTED",
            "rendered_polarity": "AFFIRMED",
            "source_modality": "ASSERTED",
            "source_polarity": "AFFIRMED",
            "status": "CURRENT",
        }:
            return None
        if not has_exact_source_ids(claim.dependency_evidence_ids, claim.source_ids):
            return None
        for source_record in source_records:
            source_id = source_record.get("source_id")
            rights_id = source_record.get("rights_decision_id")
            if (
                not isinstance(rights_id, str)
                or records[rights_id].get("source_id") != source_id
            ):
                return None
            dependency_ids = record_id_set(source_record.get("dependency_evidence_ids"))
            if dependency_ids is None:
                return None
            for dependency_id in dependency_ids:
                dependency_record = records[dependency_id]
                if dependency_record.get(
                    "source_id"
                ) != source_id or dependency_record.get(
                    "originating_report_id"
                ) != source_record.get("originating_report_id"):
                    return None
        if not _has_valid_origin_independence(
            claim.authority_class,
            source_records,
            [records[record_id] for record_id in claim.dependency_evidence_ids],
            claim.evidential_origin_ids,
        ):
            return None
    governed_claims = {claim.claim_id: claim for claim in package.governed_claims}
    for qualification in package.qualification_evidence:
        claim = governed_claims.get(qualification.governed_claim_id)
        record = records[qualification.qualification_record_id]
        if (
            claim is None
            or record.get("governed_claim_id") != qualification.governed_claim_id
            or record.get("test") != qualification.test.value
            or record.get("test_evidence")
            != [list(item) for item in qualification.test_evidence]
            or record.get("policy_version") != qualification.policy_version
            or (record.get('semantic_witness_ref') != semantic_witness_reference(qualification.semantic_witness_ref)
                if qualification.semantic_witness_ref else 'semantic_witness_ref' in record)
            or record.get("evidence_span_digest")
            != digest_bytes(claim.supporting_excerpt.encode("utf-8"))
            or record_id_set(record.get("source_record_ids"))
            != set(claim.source_record_ids)
        ):
            return None
    return tuple(sorted(digests))


def _resolve_governed_records(
    connection: sqlite3.Connection,
    candidate: StoryCandidateRecord,
    base: EvidencePackage,
    package: EvidencePackage,
) -> tuple[tuple[str, str], ...] | None:
    expected_types = _expected_governed_record_types(package)
    if expected_types is None:
        return None
    record_ids = tuple(sorted(expected_types))
    placeholders = ",".join("?" for _ in record_ids)
    rows = connection.execute(
        "SELECT record_id, record_type, record_json, record_digest "
        "FROM proving_write_evidence_records "
        f"WHERE record_id IN ({placeholders})",
        record_ids,
    ).fetchall()
    return validate_governed_evidence_records(
        candidate_id=candidate.candidate_id,
        source_inventory=tuple(
            (item.source_id, item.canonical_url) for item in candidate.items
        ),
        base_package_digest=base.digest,
        package=package,
        retained_records=tuple(tuple(row) for row in rows),
    )
