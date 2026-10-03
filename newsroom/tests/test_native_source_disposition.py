"""Only declared, unchanged archival nil disclosures avoid fresh model work."""
from dataclasses import replace
import json

import pytest

from newsroom.authority import UtcTimestamp
from newsroom.effective_revision import EffectiveRevisionIdentity
from newsroom.sources import SourceRevisionId, SourceItemId, SourceDefinitionVersionId, SourceTime, VersionedPolicyRef
from newsroom.sources.observation_models import SourceRevisionRequest
from newsroom.control_plane.native_source_intake import VERSION as SOURCE_VERSION
from newsroom.tests.test_native_graphiti import _native

NOW = UtcTimestamp.parse("2026-10-03T06:00:00Z")
CSV_NOTICE = "Published CSV cells: Row and column identify each literal text cell. Whitespace, empty fields, quoted newlines and formula-like text are preserved; nothing is executed. No header or numeric types are inferred."
HEADERS = (
    ("Special adviser", "Date", "Name", "Media organisation represented", "Purpose of meeting"),
    ("Special adviser", "Date", "Person or organisation that hospitality was received from", "Type of hospitality received", "Accompanied by spouse, family member(s) or friend?"),
    ("Special Adviser", "Date", "Name of organisation or individual", "Purpose of meeting"),
)


def _fixture(kind=0):
    base = _native("nil")
    role = "hospitality" if kind == 1 else "meetings"
    period = "October to December" if kind == 2 else "July to September"
    asset = "https://assets.publishing.service.gov.uk/media/known/table.csv"
    def row(number, cells):
        return f"Row {number}: " + "; ".join(f"{chr(65+i)}={json.dumps(value)}" for i, value in enumerate(cells))
    body = "\n".join(("Attachment: " + asset, CSV_NOTICE, 'Sheet "CSV"',
        row(1, HEADERS[kind]), row(2, ("Olivia Robey ", *("Nil Return " for _ in HEADERS[kind][1:])))))
    date = "2020-03-26T10:00:00.000000Z" if kind == 2 else "2020-01-23T10:00:00.000000Z"
    unit = replace(base, source_id="UK-01", item_key="declared|" + asset,
        headline=f"Home Office's ministerial special advisers {role}, {period} 2019", body=body,
        canonical_url="https://www.gov.uk/government/publications/special-advisers",
        published_at=date, updated_at=date, effective_pull_first_observed_at="2026-10-01T10:00:00.000000Z")
    unit = replace(unit, effective_revision=EffectiveRevisionIdentity(unit.source_id, unit.item_key, unit.revision_digest,
        unit.effective_pull_first_observed_at))
    original = SourceRevisionRequest(SourceRevisionId.parse(unit.revision_id), SourceItemId.parse(unit.authority.item_id),
        SourceDefinitionVersionId.parse(unit.authority.definition_version_id), None, date, unit.revision_digest,
        VersionedPolicyRef("native-revision", "v1"), SOURCE_VERSION,
        SourceTime.exact(UtcTimestamp.parse(date)), SourceTime.exact(UtcTimestamp.parse(date)),
        UtcTimestamp.parse(unit.effective_pull_first_observed_at), "original-native-revision")
    return unit, original


@pytest.mark.parametrize("kind", range(3))
def test_actual_archival_templates_have_source_bound_zero_call_disposition(kind):
    from newsroom.control_plane.native_source_disposition import archival_nil_return_disposition
    unit, original = _fixture(kind)
    disposition = archival_nil_return_disposition(unit, original, now=NOW)
    assert disposition is not None
    assert disposition["source_revision_digest"] == original.digest
    assert disposition["source_body_digest"] == unit.revision_digest
    assert disposition["observation_cell_count"] == len(HEADERS[kind]) - 1
    assert disposition["zero_call"] is True


@pytest.mark.parametrize("mutation", ("non_nil", "current_date", "changed_date", "unknown_template",
    "new_field", "narrative", "partial_row", "empty_data", "no_authority", "body_mismatch", "date_mismatch"))
def test_uncertain_or_material_source_keeps_ordinary_qualification(mutation):
    from newsroom.control_plane.native_source_disposition import archival_nil_return_disposition
    unit, original = _fixture()
    if mutation == "non_nil": unit = replace(unit, body=unit.body.replace('B="Nil Return "', 'B="2 August 2019"'))
    elif mutation == "current_date": unit = replace(unit, published_at=NOW.to_text(), updated_at=NOW.to_text())
    elif mutation == "changed_date": unit = replace(unit, updated_at="2021-01-23T10:00:00.000000Z")
    elif mutation == "unknown_template": unit = replace(unit, headline=unit.headline.replace("meetings", "gifts"))
    elif mutation == "new_field": unit = replace(unit, body=unit.body.replace('E="Purpose of meeting"', 'E="Purpose of meeting"; F="New instruction"'))
    elif mutation == "narrative": unit = replace(unit, body=unit.body + "\nAn official new deadline applies today.")
    elif mutation == "partial_row": unit = replace(unit, body=unit.body.rsplit('; E=', 1)[0])
    elif mutation == "empty_data": unit = replace(unit, body=unit.body.rsplit("\nRow 2:", 1)[0])
    elif mutation == "no_authority": unit = replace(unit, authority=None)
    elif mutation == "body_mismatch": original = replace(original, permitted_state_digest="sha256:" + "b" * 64)
    elif mutation == "date_mismatch": original = replace(original, source_updated_time=SourceTime.exact(NOW))
    if mutation not in {"body_mismatch", "date_mismatch", "no_authority"}:
        original = replace(original, permitted_state_digest=unit.revision_digest,
            source_published_time=SourceTime.exact(UtcTimestamp.parse(unit.published_at)),
            source_updated_time=SourceTime.exact(UtcTimestamp.parse(unit.updated_at)),
            source_native_revision_token=unit.updated_at)
    assert archival_nil_return_disposition(unit, original, now=NOW) is None
