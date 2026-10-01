"""Provider-free coverage of the observed form publication's declared HTML."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json

import pytest

from newsroom.control_plane.govuk_evidence import (
    GovUkContentHold, parse_govuk_content_document,
)
from newsroom.control_plane.native_evidence import NativeEvidenceHold
from newsroom.control_plane.native_source_intake import native_evidence_sources
from newsroom.increment9.proving import SOURCE_URLS
from newsroom.tests import test_guidance_declared_html_coverage as guidance
from newsroom.tests.test_guidance_declared_html_coverage import (
    API, CANONICAL, _poll_fixture, _raw,
)
from newsroom.tests.test_native_source_intake import (
    _atom_for, _document, _licence, _parent_with_children, _spreadsheet_parent,
    _xlsx_asset,
)


PUBLICATION = (
    "/government/publications/"
    "academy-financial-management-and-governance-self-assessment-guidance"
)
LEAF = PUBLICATION + "/list-of-questions-found-in-the-fmgs-online-form"
COLLECTION = "/government/collections/academy-financial-management"
FULL_TEXT = "The complete preview gives every published financial management question."
NOW = datetime(2026, 10, 2, 12, tzinfo=UTC)

# Contract projection of exact CAS sha256:92d1666dcdb28c804f07cb94c62445eb9
# 336c6fd94dd32ce277630e2d61d9727. Body, metadata and both declarations are
# unchanged; unrelated presentation metadata is omitted. The exact raw replay
# is retained with the change evidence, without a local-CAS dependency in CI.
_OBSERVED_FORM = {'base_path': '/government/publications/academy-financial-management-and-governance-self-assessment-guidance',
 'locale': 'en',
 'document_type': 'form',
 'schema_name': 'publication',
 'withdrawn_notice': {},
 'first_published_at': '2014-03-11T00:00:00+00:00',
 'public_updated_at': '2026-10-01T09:30:11+01:00',
 'title': 'Academies financial management and governance self-assessment',
 'details': {'body': '<div class="govspeak"><p>Newly operational academy trusts must submit an '
                     '<abbr title="financial management and governance '
                     'self-assessment">FMGS</abbr> return using the <abbr title="financial '
                     'management and governance self-assessment">FMGS</abbr> form sent to them '
                     'through Document Exchange (<abbr title="Document Exchange">DocEx</abbr>) '
                     'within 3 months of opening their first schools.</p>\n'
                     '\n'
                     '<p>The <abbr title="financial management and governance '
                     'self-assessment">FMGS</abbr> form will be collected through <abbr '
                     'title="Department for Education">DfE</abbr> Sign-in. If you’ve not already '
                     'registered, follow the <a rel="external" '
                     'href="https://services.signin.education.gov.uk/">create a <abbr '
                     'title="Department for Education">DfE</abbr> Sign-in account process</a> to '
                     'request access to <abbr title="Document Exchange">DocEx</abbr>. You will be '
                     'contacted if this applies to your trust.</p>\n'
                     '\n'
                     '<p>The guidance on this page allows you to preview the questions that appear '
                     'in the form before you start completing it.</p>\n'
                     '\n'
                     '<p>Contact us using the <a rel="external" '
                     'href="https://customerhelpportal.education.gov.uk/">customer help portal</a> '
                     'if you have any questions.</p>\n'
                     '\n'
                     '</div>',
             'attachments': [{'attachment_type': 'html',
                              'command_paper_number': '',
                              'hoc_paper_number': '',
                              'id': '9526204',
                              'isbn': '',
                              'title': 'Preview of FMGS questions',
                              'unique_reference': '',
                              'unnumbered_command_paper': False,
                              'unnumbered_hoc_paper': False,
                              'url': '/government/publications/academy-financial-management-and-governance-self-assessment-guidance/list-of-questions-found-in-the-fmgs-online-form'}]},
 'links': {'organisations': [{'title': 'Department for Education'}],
           'children': [{'base_path': '/government/publications/academy-financial-management-and-governance-self-assessment-guidance/list-of-questions-found-in-the-fmgs-online-form',
                         'title': 'Preview of FMGS questions',
                         'document_type': 'html_publication',
                         'schema_name': 'html_publication'}]}}


@pytest.fixture(autouse=True)
def _observed_clock(monkeypatch):
    monkeypatch.setattr(guidance, "NOW", NOW)


def _form_parent():
    return json.loads(_raw(_OBSERVED_FORM))


def _html_leaf(*, path=LEAF):
    value = json.loads(_document(path=path, body=FULL_TEXT))
    value.update(document_type="html_publication", schema_name="html_publication")
    return value


def _bodies(parent, *, collection=False, leaf=None):
    bodies = {SOURCE_URLS["UK-05"]: _atom_for(COLLECTION if collection else PUBLICATION)}
    if collection:
        bodies[API + COLLECTION] = _parent_with_children(
            "document_collection", COLLECTION, ((PUBLICATION, "Form publication"),),
        )
    bodies[API + PUBLICATION] = _raw(parent)
    bodies[API + LEAF] = _raw(_html_leaf() if leaf is None else leaf)
    return bodies


def _evidence_args(runtime, result):
    return dict(
        units=result.units, sources=runtime.authority.sources,
        objects=runtime.authority.objects, licence=_licence(), proof=runtime.proof,
    )


@pytest.mark.parametrize("inventory", ("attachments", "children", "both"))
def test_observed_form_exposes_exact_complete_html_inventory_not_summary(inventory):
    value = _form_parent()
    if inventory == "attachments":
        del value["links"]["children"]
    elif inventory == "children":
        del value["details"]["attachments"]
    with pytest.raises(GovUkContentHold) as caught:
        parse_govuk_content_document(CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW)
    assert caught.value.reason_code == "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE"
    assert caught.value.child_items == ((LEAF, "Preview of FMGS questions"),)
    assert caught.value.unsupported_attachments == ()
    assert caught.value.exclusion_signals == ()


@pytest.mark.parametrize("collection", (False, True))
def test_form_retains_full_declared_leaf_with_verified_ancestry_and_replay(
    tmp_path, monkeypatch, collection,
):
    bodies = _bodies(_form_parent(), collection=collection)
    with _poll_fixture(tmp_path, monkeypatch, bodies, source_id="UK-05") as case:
        runtime, intake, result, fetched = case
        assert result.status == "READY" and result.item_holds == ()
        assert [(unit.canonical_url, unit.body) for unit in result.units] == [
            (CANONICAL + LEAF, FULL_TEXT),
        ]
        assert result.units[0].item_key.endswith("|" + LEAF)
        assert len(result.observations) == len(bodies)
        assert Counter(fetched) == Counter(bodies.keys())
        observations = {value[1]: value for value in result.observations}
        evidence_args = _evidence_args(runtime, result)
        assert len(native_evidence_sources(observations=observations, **evidence_args)) == 1
        # Matching leaf bytes do not waive its exact feed and parent lineage.
        for ancestor in list(bodies)[:-1]:
            incomplete = {key: value for key, value in observations.items()
                          if value[0] != ancestor}
            with pytest.raises(NativeEvidenceHold, match="NATIVE_SOURCE_AUTHORITY_HOLD"):
                native_evidence_sources(observations=incomplete, **evidence_args)
        replay = next(value for value in intake.poll() if value.source_id == "UK-05")
        assert replay.status == "READY" and replay.item_holds == ()
        assert [unit.ingest_id for unit in replay.units] == [unit.ingest_id for unit in result.units]
        assert Counter(fetched) == Counter({url: 2 for url in bodies})


def test_form_does_not_mask_invalid_metadata_body_or_inventory():
    mutations = (
        ("schema_name", "consultation"), ("document_type", "unrecognised_form"),
        ("base_path", PUBLICATION + "-other"), ("locale", "cy"),
        ("withdrawn_notice", {"explanation": "Withdrawn"}), ("title", ""),
        ("first_published_at", "2026-10-03T00:00:00Z"),
        ("public_updated_at", "2026-10-03T00:00:00Z"),
        ("public_updated_at", "2026-10-01T11:00:00"),
    )
    for field, replacement in mutations:
        value = _form_parent()
        value[field] = replacement
        with pytest.raises(ValueError) as caught:
            parse_govuk_content_document(CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW)
        assert type(caught.value) is ValueError, field
    for body in ("", "<p> </p>", None, "<script>untrusted()</script>"):
        value = _form_parent()
        value["details"]["body"] = body
        with pytest.raises(ValueError) as caught:
            parse_govuk_content_document(CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW)
        assert type(caught.value) is ValueError
    value = _form_parent()
    value["links"]["organisations"] = []
    with pytest.raises(ValueError, match="responsible publisher is absent"):
        parse_govuk_content_document(CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW)
    for container, field in (("details", "attachments"), ("links", "children")):
        for replacement in ({"invalid": "inventory"}, [None]):
            value = _form_parent()
            value[container][field] = replacement
            with pytest.raises(ValueError) as caught:
                parse_govuk_content_document(CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW)
            assert type(caught.value) is ValueError
        value = _form_parent()
        value[container][field] *= 2
        with pytest.raises(ValueError, match="source attachment inventory is incomplete"):
            parse_govuk_content_document(CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW)
    for location in (LEAF + "/%2e%2e/other", "https://example.org/form", None):
        value = _form_parent()
        value["details"]["attachments"][0]["url"] = location
        with pytest.raises(ValueError, match="source attachment identity differs"):
            parse_govuk_content_document(CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW)
    for inventory in (None, []):
        value = _form_parent()
        value["details"]["attachments"] = value["links"]["children"] = inventory
        with pytest.raises(ValueError, match="source attachment inventory is absent"):
            parse_govuk_content_document(CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW)


@pytest.mark.parametrize("location", ("parent-body", "parent-metadata", "leaf-body", "leaf-metadata"))
def test_form_rights_exclusions_veto_parent_or_leaf_retention(tmp_path, monkeypatch, location):
    parent, leaf = _form_parent(), _html_leaf()
    target = parent if location.startswith("parent-") else leaf
    if location.endswith("body"):
        target["details"]["body"] = "<p>All rights <em>reserved</em> for this text.</p>"
    else:
        target["details"]["copyright_notice"] = "Not covered by the Open Government Licence."
    bodies = _bodies(parent, leaf=leaf)
    with _poll_fixture(tmp_path, monkeypatch, bodies, source_id="UK-05") as (_, _, result, fetched):
        parent_held = location.startswith("parent-")
        assert result.status == "HOLD" and result.units == ()
        assert result.item_holds == ((
            CANONICAL + (PUBLICATION if parent_held else LEAF), "SOURCE_ITEM_RIGHTS_EXCLUSION_HOLD",
        ),)
        assert Counter(fetched) == Counter(list(bodies)[:2] if parent_held else bodies.keys())


@pytest.mark.parametrize("binary_only", (False, True))
def test_form_unsupported_assets_stay_held_without_fetch_or_summary_waiver(
    tmp_path, monkeypatch, binary_only,
):
    parent = _form_parent()
    unsupported = [{
        "attachment_type": "file", "url": "https://assets.publishing.service.gov.uk/media/form." + suffix,
        "title": "Unsupported " + suffix.upper(),
    } for suffix in ("pdf", "odt")]
    if binary_only:
        parent["details"]["attachments"] = unsupported
        parent["links"]["children"] = []
    else:
        parent["details"]["attachments"].extend(unsupported)
    bodies = _bodies(parent)
    if binary_only:
        del bodies[API + LEAF]
    with _poll_fixture(tmp_path, monkeypatch, bodies, source_id="UK-05") as case:
        runtime, _, result, fetched = case
        assert result.status == "HOLD"
        assert result.item_holds == ((
            CANONICAL + PUBLICATION, ("SOURCE_ITEM_METADATA_HOLD" if binary_only
                                     else "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE"),
        ),)
        assert [(unit.canonical_url, unit.body) for unit in result.units] == (
            [] if binary_only else [(CANONICAL + LEAF, FULL_TEXT)]
        )
        assert Counter(fetched) == Counter(bodies.keys())
        if not binary_only:
            assert len(native_evidence_sources(
                observations={value[1]: value for value in result.observations},
                **_evidence_args(runtime, result),
            )) == 1


def test_binary_only_form_does_not_gain_spreadsheet_coverage(tmp_path, monkeypatch):
    asset_url = "https://assets.publishing.service.gov.uk/media/form.xlsx"
    asset = _xlsx_asset()
    parent = json.loads(_spreadsheet_parent(PUBLICATION, asset_url, asset))
    parent["document_type"] = "form"
    parent["details"]["body"] = _OBSERVED_FORM["details"]["body"]
    bodies = {
        SOURCE_URLS["UK-05"]: _atom_for(PUBLICATION),
        API + PUBLICATION: _raw(parent), asset_url: asset,
    }
    with _poll_fixture(tmp_path, monkeypatch, bodies, source_id="UK-05") as (_, _, result, fetched):
        assert result.status == "HOLD" and result.units == ()
        assert result.item_holds == ((CANONICAL + PUBLICATION, "SOURCE_ITEM_METADATA_HOLD"),)
        assert fetched == list(bodies)[:2]


@pytest.mark.parametrize("boundary", ("non-descendant", "other-route", "other-type", "nested", "missing"))
def test_collection_form_handoff_stays_strict_bounded_and_terminal(
    tmp_path, monkeypatch, boundary,
):
    parent, leaf = _form_parent(), _html_leaf()
    if boundary == "non-descendant":
        path = PUBLICATION + "-other/report"
        parent["details"]["attachments"][0]["url"] = path
        parent["links"]["children"][0]["base_path"] = path
    elif boundary == "other-route":
        parent["base_path"] = PUBLICATION.replace("publications", "statistics")
        parent["details"]["attachments"][0]["url"] = parent["base_path"] + "/report"
        parent["links"]["children"][0]["base_path"] = parent["base_path"] + "/report"
    elif boundary == "other-type":
        parent["document_type"] = "research"
    elif boundary == "nested":
        leaf["details"]["attachments"] = [{
            "attachment_type": "html", "url": LEAF + "/deeper", "title": "Deeper form",
        }]
    bodies = _bodies(parent, collection=True, leaf=leaf)
    if boundary == "other-route":
        path = parent["base_path"]
        bodies[API + COLLECTION] = _parent_with_children(
            "document_collection", COLLECTION, ((path, "Form publication"),),
        )
        bodies[API + path] = bodies.pop(API + PUBLICATION)
    if boundary in {"non-descendant", "other-route", "other-type", "missing"}:
        del bodies[API + LEAF]
    with _poll_fixture(tmp_path, monkeypatch, bodies, source_id="UK-05") as (_, _, result, fetched):
        assert result.status == "HOLD" and result.units == ()
        held_path = LEAF if boundary in {"nested", "missing"} else parent["base_path"]
        reason = ("SOURCE_ITEM_METADATA_HOLD" if boundary == "other-type"
                  else "SOURCE_ITEM_RETAIN_FAILED" if boundary == "missing"
                  else "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE")
        assert result.item_holds == ((CANONICAL + held_path, reason),)
        expected = list(bodies) + ([API + LEAF] if boundary == "missing" else [])
        if boundary not in {"nested", "missing"}:
            expected = expected[:3]
        assert Counter(fetched) == Counter(expected)
