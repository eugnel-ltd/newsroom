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


def test_closed_reporting_quarter_can_be_published_in_the_same_archival_year():
    from newsroom.control_plane.native_source_disposition import archival_nil_return_disposition
    unit, original = _fixture()
    published = "2018-10-18T10:00:00.000000Z"
    unit = replace(unit, headline=unit.headline.replace("July to September 2019", "April to June 2018"),
        published_at=published, updated_at=published)
    original = replace(original, permitted_state_digest=unit.revision_digest,
        source_published_time=SourceTime.exact(UtcTimestamp.parse(published)),
        source_updated_time=SourceTime.exact(UtcTimestamp.parse(published)), source_native_revision_token=published)
    witness = archival_nil_return_disposition(unit, original, now=NOW)
    assert witness is not None and witness["reporting_period_end"] == "2018-06-30"
    future_period = replace(unit, headline=unit.headline.replace("April to June", "October to December"))
    original = replace(original, permitted_state_digest=future_period.revision_digest)
    assert archival_nil_return_disposition(future_period, original, now=NOW) is None


# Exact retained f02bf9b5 CSV body (1,142 bytes); source clocks below are retained too.
MINISTERIAL_HOSPITALITY_CSV = '''Attachment: https://assets.publishing.service.gov.uk/media/60b0a384e90e0732a9461991/Ministerial_Hospitality_Oct-Dec_20.csv
Published CSV cells: Row and column identify each literal text cell. Whitespace, empty fields, quoted newlines and formula-like text are preserved; nothing is executed. No header or numeric types are inferred.
Sheet "CSV"
Row 1: A="Minister"; B="Date"; C="Person or organisation that offered hospitality"; D="Type of hospitality received"; E="Accompanied by spouse, family member(s) or friend?"
Row 2: A="Priti Patel"; B="nil return"; C="nil return"; D="nil return"; E="nil return"
Row 3: A="Kevin Foster"; B="nil return"; C="nil return"; D="nil return"; E="nil return"
Row 4: A="Kit Malthouse"; B="nil return"; C="nil return"; D="nil return"; E="nil return"
Row 5: A="Chris Philp"; B="nil return"; C="nil return"; D="nil return"; E="nil return"
Row 6: A="James Brokenshire"; B="nil return"; C="nil return"; D="nil return"; E="nil return"
Row 7: A="Victoria Atkins"; B="nil return"; C="nil return"; D="nil return"; E="nil return"
Row 8: A="Susan Williams"; B="nil return"; C="nil return"; D="nil return"; E="nil return"'''


def _ministerial_hospitality_fixture():
    unit, original = _fixture(1)
    published = "2021-05-28T10:14:00.000000Z"
    asset = MINISTERIAL_HOSPITALITY_CSV.splitlines()[0].removeprefix("Attachment: ")
    unit = replace(unit,
        headline="Home Office's ministerial hospitality, October to December 2020",
        canonical_url="https://www.gov.uk/government/publications/home-office-ministerial-gifts-hospitality-travel-and-meetings-october-to-december-2020",
        item_key="declared|" + asset, body=MINISTERIAL_HOSPITALITY_CSV,
        published_at=published, updated_at=published)
    unit = replace(unit, effective_revision=EffectiveRevisionIdentity(
        unit.source_id, unit.item_key, unit.revision_digest, unit.effective_pull_first_observed_at))
    original = replace(original, source_native_revision_token=published,
        permitted_state_digest=unit.revision_digest,
        source_published_time=SourceTime.exact(UtcTimestamp.parse(published)),
        source_updated_time=SourceTime.exact(UtcTimestamp.parse(published)))
    return unit, original


def test_actual_ministerial_hospitality_csv_has_source_bound_zero_call_disposition():
    from newsroom.authority.canonical import digest_bytes
    from newsroom.control_plane.native_source_disposition import archival_nil_return_disposition
    unit, original = _ministerial_hospitality_fixture()
    assert digest_bytes(unit.body.encode()) == "sha256:125dc3d60c4906a9597cc63ad6f782094502c08ce5c946ea46e6c0ef3a882ffd"
    witness = archival_nil_return_disposition(unit, original, now=NOW)
    assert witness is not None
    assert witness["zero_call"] is True
    assert witness["identity_row_count"] == 7
    assert witness["observation_cell_count"] == 28
    assert witness["reporting_period_end"] == "2020-12-31"
    assert witness["source_revision_digest"] == original.digest
    assert witness["source_body_digest"] == unit.revision_digest


@pytest.mark.parametrize("mutation", (
    "non_nil", "empty_cell", "unknown_nil_case", "wrong_header", "new_field",
    "narrative", "partial_row", "missing_row", "empty_data", "malformed_cell",
    "unknown_template", "ods", "asset_changed", "parent_changed", "current_date",
    "changed_date", "no_authority", "body_mismatch", "date_mismatch",
    "revision_mismatch", "item_mismatch", "definition_mismatch", "token_mismatch", "observation_mismatch",
    "canonicalizer_mismatch",
))
def test_ministerial_hospitality_uncertainty_keeps_ordinary_qualification(mutation):
    from newsroom.control_plane.native_source_disposition import archival_nil_return_disposition
    unit, original = _ministerial_hospitality_fixture()
    if mutation == "non_nil": unit = replace(unit, body=unit.body.replace('B="nil return"', 'B="14 December 2020"', 1))
    elif mutation == "empty_cell": unit = replace(unit, body=unit.body.replace('B="nil return"', 'B=""', 1))
    elif mutation == "unknown_nil_case": unit = replace(unit, body=unit.body.replace('B="nil return"', 'B="NIL RETURN"', 1))
    elif mutation == "wrong_header": unit = replace(unit, body=unit.body.replace('A="Minister"', 'A="Special adviser"', 1))
    elif mutation == "new_field": unit = replace(unit, body=unit.body.replace('?"\nRow 2:', '?"; F="New instruction"\nRow 2:', 1))
    elif mutation == "narrative": unit = replace(unit, body=unit.body + "\nAn official new deadline applies today.")
    elif mutation == "partial_row": unit = replace(unit, body=unit.body.replace('; E="nil return"', '', 1))
    elif mutation == "missing_row": unit = replace(unit, body="\n".join(line for line in unit.body.splitlines() if not line.startswith("Row 3: ")))
    elif mutation == "empty_data": unit = replace(unit, body=unit.body.split("\nRow 2:")[0])
    elif mutation == "malformed_cell": unit = replace(unit, body=unit.body.replace('B="nil return"', 'B="nil return', 1))
    elif mutation == "unknown_template": unit = replace(unit, headline=unit.headline.replace("hospitality", "travel"))
    elif mutation == "ods": unit = replace(unit, body=unit.body.replace('Sheet "CSV"', 'Sheet "ODS"'))
    elif mutation == "asset_changed": unit = replace(unit, body=unit.body.replace(".csv", ".ods"), item_key=unit.item_key.replace(".csv", ".ods"))
    elif mutation == "parent_changed": unit = replace(unit, canonical_url=unit.canonical_url.replace("www.gov.uk", "example.invalid"))
    elif mutation == "current_date": unit = replace(unit, published_at=NOW.to_text(), updated_at=NOW.to_text())
    elif mutation == "changed_date": unit = replace(unit, updated_at="2022-05-28T10:14:00.000000Z")
    elif mutation == "no_authority": unit = replace(unit, authority=None)
    else:
        if mutation == "body_mismatch": original = replace(original, permitted_state_digest="sha256:" + "b" * 64)
        elif mutation == "date_mismatch": original = replace(original, source_updated_time=SourceTime.exact(NOW))
        elif mutation == "revision_mismatch": original = replace(original, revision_id=SourceRevisionId.parse("00000000-0000-4000-8000-000000000001"))
        elif mutation == "item_mismatch": original = replace(original, item_id=SourceItemId.parse("00000000-0000-4000-8000-000000000001"))
        elif mutation == "definition_mismatch": original = replace(original, definition_version_id=SourceDefinitionVersionId.parse("00000000-0000-4000-8000-000000000001"))
        elif mutation == "token_mismatch": original = replace(original, source_native_revision_token="different-token")
        elif mutation == "observation_mismatch": original = replace(original, observed_at=NOW)
        elif mutation == "canonicalizer_mismatch": original = replace(original, canonicalizer_version="unknown-source-version")
        assert archival_nil_return_disposition(unit, original, now=NOW) is None
        return
    if unit.authority is not None:
        unit = replace(unit, effective_revision=EffectiveRevisionIdentity(
            unit.source_id, unit.item_key, unit.revision_digest, unit.effective_pull_first_observed_at))
        original = replace(original, permitted_state_digest=unit.revision_digest,
            source_published_time=SourceTime.exact(UtcTimestamp.parse(unit.published_at)),
            source_updated_time=SourceTime.exact(UtcTimestamp.parse(unit.updated_at)),
            source_native_revision_token=unit.updated_at)
    assert archival_nil_return_disposition(unit, original, now=NOW) is None


def test_known_uppercase_ministerial_nil_literal_and_original_clock_are_preserved():
    from newsroom.control_plane.native_source_disposition import archival_nil_return_disposition
    unit, original = _ministerial_hospitality_fixture()
    unit = replace(unit, body=unit.body.replace('="nil return"', '="Nil Return"'))
    original = replace(original, permitted_state_digest=unit.revision_digest)
    assert archival_nil_return_disposition(unit, original, now=NOW)["observation_cell_count"] == 28
    assert archival_nil_return_disposition(unit, original, now=UtcTimestamp.parse("2026-09-30T06:00:00Z")) is None


def test_lowercase_nil_does_not_expand_existing_special_adviser_contract():
    from newsroom.control_plane.native_source_disposition import archival_nil_return_disposition
    unit, original = _fixture(1)
    unit = replace(unit, body=unit.body.replace("Nil Return", "nil return"))
    original = replace(original, permitted_state_digest=unit.revision_digest)
    assert archival_nil_return_disposition(unit, original, now=NOW) is None
