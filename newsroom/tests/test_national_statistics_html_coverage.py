"""Provider-free coverage of declared national-statistics publication leaves."""

from __future__ import annotations

from collections import Counter
import json

import pytest

from newsroom.control_plane.govuk_evidence import (
    GovUkContentHold, parse_govuk_content_document,
)
from newsroom.control_plane.native_evidence import NativeEvidenceHold
from newsroom.control_plane.native_source_intake import native_evidence_sources
from newsroom.increment9.proving import SOURCE_URLS
from newsroom.tests.test_guidance_declared_html_coverage import (
    API, CANONICAL, NOW, _poll_fixture, _raw,
)
from newsroom.tests.test_native_source_intake import (
    _atom_for, _document, _licence, _parent_with_children, _spreadsheet_parent,
    _xlsx_asset,
)


PUBLICATION = (
    "/government/statistics/"
    "stop-and-search-arrests-and-mental-health-detentions-march-2026"
)
CHILDREN = (
    (PUBLICATION + "/statistics-report", "Statistics report"),
    (PUBLICATION + "/pre-release-access-list", "Pre-release access list"),
)
INTRODUCTION = "Statistics on stop and search, arrests and mental health detentions."
FULL_TEXT = "The complete published statistics report and its supporting definitions."
COLLECTION = "/government/collections/police-powers-and-procedures"
PDF_URL = "https://assets.publishing.service.gov.uk/media/statistics-report.pdf"


def _statistics_parent() -> dict:
    # Mirrors the retained publication/national_statistics shape: a non-empty
    # introduction and the same two HTML declarations in both inventories.
    value = json.loads(_parent_with_children(
        "national_statistics", PUBLICATION, CHILDREN,
    ))
    value["details"]["body"] = "<p>" + INTRODUCTION + "</p>"
    for child in value["links"]["children"]:
        child["document_type"] = "html_publication"
    return value


def _html_leaf(path: str) -> bytes:
    value = json.loads(_document(path=path, body=FULL_TEXT))
    value.update(document_type="html_publication", schema_name="html_publication")
    return _raw(value)


def test_feed_national_statistics_retains_two_declared_html_leaves_not_introduction(
    tmp_path, monkeypatch,
):
    bodies = {
        SOURCE_URLS["UK-01"]: _atom_for(PUBLICATION),
        API + PUBLICATION: _raw(_statistics_parent()),
        **{API + path: _html_leaf(path) for path, _ in CHILDREN},
    }
    with _poll_fixture(tmp_path, monkeypatch, bodies) as case:
        runtime, _, result, fetched = case
        assert result.status == "READY" and result.item_holds == ()
        assert [(unit.canonical_url, unit.body) for unit in result.units] == [
            (CANONICAL + path, FULL_TEXT) for path, _ in CHILDREN
        ]
        assert all(unit.item_key.endswith("|" + path)
                   for unit, (path, _) in zip(result.units, CHILDREN, strict=True))
        assert len(result.observations) == 4
        assert Counter(fetched) == Counter(bodies.keys())  # Each declared leaf once.
        assert len(native_evidence_sources(
            units=result.units, sources=runtime.authority.sources,
            objects=runtime.authority.objects,
            observations={value[1]: value for value in result.observations},
            licence=_licence(), proof=runtime.proof,
        )) == 2


def test_collection_statistics_html_has_full_ancestry_replay_and_pdf_hold(
    tmp_path, monkeypatch,
):
    parent = _statistics_parent()
    parent["details"]["attachments"].append({
        "attachment_type": "file", "url": PDF_URL, "title": "Uncovered report",
    })
    bodies = {
        SOURCE_URLS["UK-01"]: _atom_for(COLLECTION),
        API + COLLECTION: _parent_with_children(
            "document_collection", COLLECTION, ((PUBLICATION, "National statistics"),),
        ),
        API + PUBLICATION: _raw(parent),
        **{API + path: _html_leaf(path) for path, _ in CHILDREN},
    }
    with _poll_fixture(tmp_path, monkeypatch, bodies) as case:
        runtime, intake, result, fetched = case
        assert result.status == "HOLD"
        assert result.item_holds == ((
            CANONICAL + PUBLICATION, "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE",
        ),)
        assert [(unit.canonical_url, unit.body) for unit in result.units] == [
            (CANONICAL + path, FULL_TEXT) for path, _ in CHILDREN
        ]
        assert len(result.observations) == 5
        assert Counter(fetched) == Counter(bodies.keys())
        observations = {value[1]: value for value in result.observations}
        evidence_args = dict(
            units=result.units, sources=runtime.authority.sources,
            objects=runtime.authority.objects, licence=_licence(), proof=runtime.proof,
        )
        assert len(native_evidence_sources(observations=observations, **evidence_args)) == 2
        for ancestor in (SOURCE_URLS["UK-01"], API + COLLECTION, API + PUBLICATION):
            incomplete = {key: value for key, value in observations.items()
                          if value[0] != ancestor}
            with pytest.raises(NativeEvidenceHold, match="NATIVE_SOURCE_AUTHORITY_HOLD"):
                native_evidence_sources(observations=incomplete, **evidence_args)
        replay = next(value for value in intake.poll() if value.source_id == "UK-01")
        assert replay.status == "HOLD" and replay.item_holds == result.item_holds
        assert [unit.ingest_id for unit in replay.units] == [unit.ingest_id for unit in result.units]
        assert Counter(fetched) == Counter({url: 2 for url in bodies})


def test_statistics_inventory_is_complete_and_deduplicated():
    for inventory in ("attachments", "children", "both"):
        value = _statistics_parent()
        if inventory == "attachments":
            del value["links"]["children"]
        elif inventory == "children":
            del value["details"]["attachments"]
        with pytest.raises(GovUkContentHold) as caught:
            parse_govuk_content_document(
                CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW,
            )
        assert caught.value.reason_code == "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE"
        assert caught.value.child_items == CHILDREN
        assert caught.value.unsupported_attachments == ()
        assert caught.value.exclusion_signals == ()


def test_statistics_inventory_never_masks_invalid_schema_body_or_metadata():
    mutations = (
        ("unknown-type", "document_type", "unrecognised_statistics"),
        ("other-statistics-type", "document_type", "official_statistics"),
        ("wrong-schema", "schema_name", "html_publication"),
        ("wrong-path", "base_path", PUBLICATION + "-unrelated"),
        ("wrong-locale", "locale", "cy"),
        ("withdrawn", "withdrawn_notice", {"explanation": "Withdrawn"}),
        ("empty-title", "title", ""),
        ("future-publication", "first_published_at", "2026-09-09T00:00:00Z"),
        ("future-update", "public_updated_at", "2026-09-09T00:00:00Z"),
        ("unbounded-time", "public_updated_at", "2026-09-08T11:00:00"),
    )
    for label, field, replacement in mutations:
        value = _statistics_parent()
        value[field] = replacement
        with pytest.raises(ValueError) as caught:
            parse_govuk_content_document(
                CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW,
            )
        assert type(caught.value) is ValueError, label
    for body in ("", "<p> </p>", None, "<script>untrusted()</script>"):
        value = _statistics_parent()
        value["details"]["body"] = body
        with pytest.raises(ValueError) as caught:
            parse_govuk_content_document(
                CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW,
            )
        assert type(caught.value) is ValueError
    value = _statistics_parent()
    value["links"]["organisations"] = []
    with pytest.raises(ValueError, match="responsible publisher is absent"):
        parse_govuk_content_document(
            CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW,
        )


def test_statistics_unsafe_duplicate_or_malformed_inventories_fail_closed():
    for mutation in (
        "unsafe", "cross-origin", "dangling", "duplicate-attachments",
        "duplicate-children", "malformed-attachments", "malformed-children",
        "absent", "empty",
    ):
        value = _statistics_parent()
        if mutation in {"unsafe", "cross-origin"}:
            value["details"]["attachments"][0]["url"] = (
                PUBLICATION + "/%2e%2e/unrelated" if mutation == "unsafe"
                else "https://example.org/statistics-report"
            )
        elif mutation == "dangling":
            del value["details"]["attachments"][0]["url"]
        elif mutation.startswith("duplicate"):
            container, field = (("details", "attachments") if mutation.endswith("attachments")
                                else ("links", "children"))
            value[container][field] *= 2
        elif mutation.startswith("malformed"):
            container, field = (("details", "attachments") if mutation.endswith("attachments")
                                else ("links", "children"))
            value[container][field] = {"invalid": "inventory"}
        elif mutation == "absent":
            del value["details"]["attachments"]
            del value["links"]["children"]
        else:
            value["details"]["attachments"] = []
            value["links"]["children"] = []
        with pytest.raises(ValueError) as caught:
            parse_govuk_content_document(
                CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW,
            )
        assert type(caught.value) is ValueError, mutation


def test_statistics_and_html_leaf_rights_veto_retention(tmp_path, monkeypatch):
    for location in ("parent-body", "parent-metadata", "leaf-body", "leaf-metadata"):
        parent, leaf = _statistics_parent(), json.loads(_html_leaf(CHILDREN[0][0]))
        target = parent if location.startswith("parent-") else leaf
        if location.endswith("body"):
            target["details"]["body"] = "<p>All rights <em>reserved</em> for this text.</p>"
        else:
            target["details"]["copyright_notice"] = "Not covered by the Open Government Licence."
        bodies = {
            SOURCE_URLS["UK-01"]: _atom_for(PUBLICATION),
            API + PUBLICATION: _raw(parent),
            API + CHILDREN[0][0]: _raw(leaf),
            API + CHILDREN[1][0]: _html_leaf(CHILDREN[1][0]),
        }
        case_path = tmp_path / location
        case_path.mkdir(mode=0o700)
        with _poll_fixture(case_path, monkeypatch, bodies) as (_, _, result, fetched):
            parent_held = location.startswith("parent-")
            held_path = PUBLICATION if parent_held else CHILDREN[0][0]
            assert result.status == "HOLD"
            assert result.item_holds == ((
                CANONICAL + held_path, "SOURCE_ITEM_RIGHTS_EXCLUSION_HOLD",
            ),)
            assert [unit.canonical_url for unit in result.units] == (
                [] if parent_held else [CANONICAL + CHILDREN[1][0]]
            )
            assert Counter(fetched) == Counter(list(bodies)[:2] if parent_held else bodies.keys())


def test_collection_statistics_handoff_rejects_non_descendants_and_other_types(
    tmp_path, monkeypatch,
):
    for label, document_type, child_path in (
        ("non-descendant", "national_statistics", PUBLICATION + "-other/report"),
        ("other-publication-type", "corporate_report", CHILDREN[0][0]),
    ):
        parent = json.loads(_parent_with_children(
            document_type, PUBLICATION, ((child_path, "Declared report"),),
        ))
        bodies = {
            SOURCE_URLS["UK-01"]: _atom_for(COLLECTION),
            API + COLLECTION: _parent_with_children(
                "document_collection", COLLECTION, ((PUBLICATION, "Statistics"),),
            ),
            API + PUBLICATION: _raw(parent),
        }
        case_path = tmp_path / label
        case_path.mkdir(mode=0o700)
        with _poll_fixture(case_path, monkeypatch, bodies) as (_, _, result, fetched):
            assert result.status == "HOLD" and result.units == ()
            assert result.item_holds == ((
                CANONICAL + PUBLICATION, "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE",
            ),)
            assert Counter(fetched) == Counter(bodies.keys())


def test_statistics_terminal_leaf_never_crawls_a_nested_inventory(tmp_path, monkeypatch):
    nested = json.loads(_html_leaf(CHILDREN[0][0]))
    nested["details"]["attachments"] = [{
        "attachment_type": "html", "url": CHILDREN[0][0] + "/deeper",
        "title": "Undeclared deeper report",
    }]
    bodies = {
        SOURCE_URLS["UK-01"]: _atom_for(COLLECTION),
        API + COLLECTION: _parent_with_children(
            "document_collection", COLLECTION, ((PUBLICATION, "National statistics"),),
        ),
        API + PUBLICATION: _raw(_statistics_parent()),
        API + CHILDREN[0][0]: _raw(nested),
        API + CHILDREN[1][0]: _html_leaf(CHILDREN[1][0]),
    }
    with _poll_fixture(tmp_path, monkeypatch, bodies) as (_, _, result, fetched):
        assert result.status == "HOLD"
        assert [(unit.canonical_url, unit.body) for unit in result.units] == [
            (CANONICAL + CHILDREN[1][0], FULL_TEXT),
        ]
        assert result.item_holds == ((
            CANONICAL + CHILDREN[0][0], "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE",
        ),)
        assert len(result.observations) == 5
        assert Counter(fetched) == Counter(bodies.keys())


def test_binary_only_statistics_stays_held_and_mixed_assets_preserve_coverage(
    tmp_path, monkeypatch,
):
    asset_url = "https://assets.publishing.service.gov.uk/media/statistics.xlsx"
    ods_url = "https://assets.publishing.service.gov.uk/media/statistics.ods"
    asset = _xlsx_asset()
    for label in ("binary-only", "mixed-html-oversized-ods-xlsx"):
        parent = _statistics_parent()
        bodies = {
            SOURCE_URLS["UK-01"]: _atom_for(PUBLICATION),
        }
        if label == "binary-only":
            parent["details"]["attachments"] = [{
                "attachment_type": "file", "url": PDF_URL, "title": "Binary report",
            }]
            parent["links"]["children"] = []
        else:
            parent["details"]["attachments"].extend([
                {"attachment_type": "file", "url": ods_url, "title": "ODS data tables",
                 "filename": "statistics.ods", "content_type": "application/vnd.oasis.opendocument.spreadsheet",
                     # The retained release includes a 37 MB ODS; its existing
                     # acquisition bound still holds without fetching it.
                     "file_size": 37_364_527, "id": "asset-ods"},
                json.loads(_spreadsheet_parent(PUBLICATION, asset_url, asset))["details"]["attachments"][0],
            ])
        bodies[API + PUBLICATION] = _raw(parent)
        if label != "binary-only":
            bodies.update({API + path: _html_leaf(path) for path, _ in CHILDREN})
            bodies[asset_url] = asset
        case_path = tmp_path / label
        case_path.mkdir(mode=0o700)
        with _poll_fixture(case_path, monkeypatch, bodies) as case:
            runtime, _, result, fetched = case
            assert result.status == "HOLD"
            assert result.item_holds == ((
                CANONICAL + PUBLICATION, "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE",
            ),)
            assert Counter(fetched) == Counter(bodies.keys())
            if label == "binary-only":
                assert result.units == ()
            else:
                assert len(result.units) == 3
                assert [(unit.canonical_url, unit.body) for unit in result.units[:2]] == [
                    (CANONICAL + path, FULL_TEXT) for path, _ in CHILDREN
                ]
                assert "Example College" in result.units[2].body
                assert "125000" in result.units[2].body
                assert len(native_evidence_sources(
                    units=result.units, sources=runtime.authority.sources,
                    objects=runtime.authority.objects,
                    observations={value[1]: value for value in result.observations},
                    licence=_licence(), proof=runtime.proof,
                )) == 3
