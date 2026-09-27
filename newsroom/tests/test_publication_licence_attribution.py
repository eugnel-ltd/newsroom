import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from newsroom.control_plane.govuk_rights import ATTRIBUTION, LICENCE_URL
from newsroom.increment10.publication import PublicationError, _render
from newsroom.control_plane.writer import WriterCopy, WriterEvidenceLink


def _story():
    return SimpleNamespace(
        story_id="00000000-0000-4000-8000-000000000001", aggregate_version=1,
        digest="sha256:" + "a" * 64,
        copy=WriterCopy("【未出版】政府公布服務安排", "服務將於星期一開始。", "fixture-writer",
                        "sha256:" + "b" * 64,
                        (WriterEvidenceLink("claim-headline", "政府公布服務安排"),
                         WriterEvidenceLink("claim-body", "服務將於星期一開始。"))),
        write_admission=SimpleNamespace(geography=("UK",), categories=("public-service",)),
    )


def test_licence_is_bound_metadata_on_both_surfaces_not_editorial_copy():
    story = _story()
    sources = (("UK-01", "https://www.gov.uk/government/news/arrangements"),)
    legacy = _render(story, sources)
    licensed = _render(story, sources, (("www.gov.uk", ATTRIBUTION, LICENCE_URL),))
    assert all("licence_attributions" not in json.loads(item.canonical_bytes()) for item in legacy)
    assert tuple(item.payload_id for item in legacy) != tuple(item.payload_id for item in licensed)
    for surface in licensed:
        value = json.loads(surface.canonical_bytes())
        assert value["licence_attributions"] == [["UK-01", ATTRIBUTION, LICENCE_URL]]
        assert value["source_references"] == [list(sources[0])]
        assert ATTRIBUTION not in surface.headline + surface.body
        with pytest.raises(PublicationError, match="identity differs"):
            replace(surface, licence_attributions=())
    assert licensed[0].body == story.copy.body
    assert licensed[1].body == ""
    assert _render(story, sources, (("www.gov.uk", ATTRIBUTION, LICENCE_URL),)) == licensed


def test_licence_policy_never_labels_another_origin_or_source():
    story = _story()
    surfaces = _render(story, (("UK-01", "https://www.gov.uk.example.test/news"),),
                       (("www.gov.uk", ATTRIBUTION, LICENCE_URL),))
    assert all(not item.licence_attributions for item in surfaces)
    with pytest.raises(PublicationError, match="source binding"):
        replace(surfaces[0], licence_attributions=(("missing-source", ATTRIBUTION, LICENCE_URL),))


def test_spreadsheet_uses_govuk_landing_page_as_licensed_source_reference():
    landing = (
        "https://www.gov.uk/government/publications/"
        "funding-allocations-to-training-providers-2024-to-2025"
    )
    surfaces = _render(
        _story(),
        (("UK-05", landing),),
        (("www.gov.uk", ATTRIBUTION, LICENCE_URL),),
    )
    for surface in surfaces:
        assert surface.source_references == (("UK-05", landing),)
        assert surface.licence_attributions == (
            ("UK-05", ATTRIBUTION, LICENCE_URL),
        )
        assert "assets.publishing.service.gov.uk" not in surface.canonical_bytes().decode()
