"""Source-bound disposition of complete archival nil-return CSV disclosures."""
from datetime import date
import json
import re
from urllib.parse import urlsplit

from newsroom.authority import UtcTimestamp
from newsroom.authority.canonical import digest_canonical
from newsroom.sources import SourceTime, SourceRevisionRequest
from .corpus import CorpusIngestUnit, CorpusAuthorityBinding
from .govuk_spreadsheet import POLICY_DIGEST as PARSER_POLICY_DIGEST
from .native_source_intake import VERSION as SOURCE_VERSION

VERSION = "newsroom.archival-nil-return.v1"
_NOTICE = "Published CSV cells: Row and column identify each literal text cell. Whitespace, empty fields, quoted newlines and formula-like text are preserved; nothing is executed. No header or numeric types are inferred."
_HEADERS = {
    "meetings": (("Special adviser", "Date", "Name", "Media organisation represented", "Purpose of meeting"),
                 ("Special Adviser", "Date", "Name of organisation or individual", "Purpose of meeting")),
    "hospitality": (("Special adviser", "Date", "Person or organisation that hospitality was received from",
                     "Type of hospitality received", "Accompanied by spouse, family member(s) or friend?"),),
}
_PERIODS = {"January to March": (3, 31), "April to June": (6, 30),
            "July to September": (9, 30), "October to December": (12, 31)}
_TITLE = re.compile(r"Home Office's ministerial special advisers (meetings|hospitality), (January to March|April to June|July to September|October to December) ([0-9]{4})")
POLICY_DIGEST = digest_canonical({"version": VERSION, "parser": PARSER_POLICY_DIGEST,
    "headers": _HEADERS, "periods": _PERIODS, "dates": "report-year<publication-year<original-observation-year;publication=update",
    "source": SOURCE_VERSION, "observation_cells": "literal-Nil-Return-plus-whitespace"})


def archival_nil_return_candidate(unit):
    return (type(unit) is CorpusIngestUnit and type(unit.authority) is CorpusAuthorityBinding
            and _TITLE.fullmatch(unit.headline) is not None
            and unit.body.startswith("Attachment: https://assets.publishing.service.gov.uk/"))


def _rows(lines):
    rows = []
    for number, line in enumerate(lines, 1):
        prefix = f"Row {number}: "
        if not line.startswith(prefix):
            raise ValueError("CSV row coverage differs")
        encoded = line[len(prefix):]
        if encoded == "[empty record]":
            rows.append(())
            continue
        cells = re.findall(r'([A-E])=("(?:[^"\\]|\\.)*")', encoded)
        if (not cells or "; ".join(f"{column}={value}" for column, value in cells) != encoded
                or [column for column, _ in cells] != list("ABCDE"[:len(cells)])):
            raise ValueError("CSV cell coverage differs")
        rows.append(tuple(json.loads(value) for _, value in cells))
    return rows


def archival_nil_return_disposition(unit, original, *, now):
    if not archival_nil_return_candidate(unit) or type(original) is not SourceRevisionRequest:
        return None
    try:
        published, updated = UtcTimestamp.parse(unit.published_at), UtcTimestamp.parse(unit.updated_at)
    except (ValueError, TypeError):
        return None
    if (original.canonicalizer_version != SOURCE_VERSION
            or str(original.revision_id) != unit.revision_id
            or str(original.item_id) != unit.authority.item_id
            or str(original.definition_version_id) != unit.authority.definition_version_id
            or original.permitted_state_digest != unit.revision_digest
            or original.source_published_time != SourceTime.exact(published)
            or original.source_updated_time != SourceTime.exact(updated)
            or original.source_native_revision_token != unit.updated_at
            or original.observed_at.to_text() != unit.coverage_first_observed_at):
        return None
    try:
        role, period, year = _TITLE.fullmatch(unit.headline).groups()
        period_end = date(int(year), *_PERIODS[period])
        if not (published == updated and int(year) < published.value.year < original.observed_at.value.year
                and original.observed_at.value <= now.value and period_end < published.value.date()):
            return None
        lines = unit.body.splitlines()
        asset = lines[0].removeprefix("Attachment: ")
        location, parent = urlsplit(asset), urlsplit(unit.canonical_url)
        if (location.scheme != "https" or location.netloc != "assets.publishing.service.gov.uk"
                or not location.path.endswith(".csv") or location.query or location.fragment
                or not unit.item_key.endswith("|" + asset)
                or parent.scheme != "https" or parent.netloc != "www.gov.uk"
                or not parent.path.startswith("/government/publications/")
                or lines[1:3] != [_NOTICE, 'Sheet "CSV"']):
            return None
        rows = _rows(lines[3:])
        if not rows or rows[0] not in _HEADERS[role]:
            return None
        records = [row for row in rows[1:] if any(value.strip() for value in row)]
        if not records or any(len(row) != len(rows[0])
                or re.fullmatch(r"[A-Z][A-Za-z'’-]*(?: [A-Z][A-Za-z'’-]*){1,4}", row[0].strip()) is None
                or any(value.strip() != "Nil Return" for value in row[1:]) for row in records):
            return None
    except (ValueError, TypeError, IndexError):
        return None
    return {"policy_digest": POLICY_DIGEST, "parser_policy_digest": PARSER_POLICY_DIGEST,
        "source_revision_id": unit.revision_id, "source_revision_digest": original.digest,
        "source_body_digest": unit.revision_digest, "source_representation_digest": unit.representation_digest,
        "published_at": unit.published_at, "updated_at": unit.updated_at,
        "reporting_period_end": period_end.isoformat(), "header": list(rows[0]),
        "rows_digest": digest_canonical(rows), "observation_cell_count": sum(len(row)-1 for row in records),
        "identity_row_count": len(records), "zero_call": True}
