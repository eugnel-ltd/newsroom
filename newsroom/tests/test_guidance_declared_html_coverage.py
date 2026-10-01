"""Provider-free coverage of a guidance publication's declared HTML evidence."""

from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
import json

import pytest

from newsroom.control_plane.govuk_evidence import (
    GovUkContentHold, parse_govuk_content_document,
)
from newsroom.control_plane.graphiti_operational_readiness import (
    OPERATOR_AUTHORITY_DOMAIN, OPERATOR_PRINCIPAL_ID,
)
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.native_evidence import NativeEvidenceHold
from newsroom.control_plane.native_source_intake import (
    NativeSourceIntake, native_evidence_sources,
)
from newsroom.increment9.proving import SOURCE_URLS
from newsroom.tests.test_native_source_intake import (
    _atom_for, _document, _licence, _parent_with_children, _seed_missing,
    _seed_uk01,
)
from newsroom.tests.test_native_runtime import _args


PUBLICATION = (
    "/government/publications/"
    "police-powers-and-procedures-in-england-and-wales-201112-user-guide"
)
LEAF = PUBLICATION + "/user-guide-to-police-powers-and-procedures"
COLLECTION = "/government/collections/police-powers-and-procedures"
API = "https://www.gov.uk/api/content"
CANONICAL = "https://www.gov.uk"
NOW = datetime(2026, 9, 8, 12, tzinfo=UTC)
INTRODUCTION = "Read the user guide for police powers and procedures statistics."
FULL_TEXT = "The complete user guide explains the scope, definitions and statistical methods."


def _raw(value):
    return json.dumps(value, separators=(",", ":")).encode()


def _guidance_parent(*, binary=False):
    # The observed publication declares one child in both inventories; its
    # introduction is not the complete user guide.
    value = json.loads(_parent_with_children(
        "guidance", PUBLICATION,
        ((LEAF, "User guide to police powers and procedures"),), binary=binary,
    ))
    value["details"]["body"] = "<p>" + INTRODUCTION + "</p>"
    value["links"]["children"][0]["document_type"] = "html_publication"
    return value


def _html_leaf(*, path=LEAF):
    value = json.loads(_document(path=path, body=FULL_TEXT))
    value.update(document_type="html_publication", schema_name="html_publication")
    return value


@contextmanager
def _poll_fixture(tmp_path, monkeypatch, bodies, *, source_id="UK-01"):
    args = _args(tmp_path, monkeypatch)
    args.update(
        principal_id=OPERATOR_PRINCIPAL_ID,
        authority_domain=OPERATOR_AUTHORITY_DOMAIN,
    )
    fetched = []

    def fetch(url):
        fetched.append(url)
        return 200, bodies[url]

    with open_native_runtime(**args) as runtime:
        definition_id = (
            _seed_uk01(runtime) if source_id == "UK-01"
            else _seed_missing(runtime, source_id)
        )
        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=runtime.proof, definition_ids={source_id: definition_id},
            licence=_licence(), dispatch_fence=lambda *_: nullcontext(),
            fetch=fetch, clock=lambda: NOW,
        )
        result = next(value for value in intake.poll() if value.source_id == source_id)
        yield runtime, intake, result, fetched


def test_feed_guidance_retains_the_declared_html_not_the_introduction(tmp_path, monkeypatch):
    bodies = {
        SOURCE_URLS["UK-01"]: _atom_for(PUBLICATION),
        API + PUBLICATION: _raw(_guidance_parent()),
        API + LEAF: _raw(_html_leaf()),
    }
    with _poll_fixture(tmp_path, monkeypatch, bodies) as (runtime, _, result, fetched):
        assert result.status == "READY" and result.item_holds == ()
        assert [(unit.canonical_url, unit.body) for unit in result.units] == [
            (CANONICAL + LEAF, FULL_TEXT),
        ]
        assert result.units[0].item_key.endswith("|" + LEAF)
        assert len(result.observations) == 3
        assert fetched == list(bodies)  # Duplicate inventories fetch the leaf once.
        assert len(native_evidence_sources(
            units=result.units, sources=runtime.authority.sources,
            objects=runtime.authority.objects,
            observations={value[1]: value for value in result.observations},
            licence=_licence(), proof=runtime.proof,
        )) == 1


def test_feed_collection_guidance_leaf_retains_full_ancestry_and_replay(tmp_path, monkeypatch):
    bodies = {
        SOURCE_URLS["UK-05"]: _atom_for(COLLECTION),
        API + COLLECTION: _parent_with_children(
            "document_collection", COLLECTION, ((PUBLICATION, "User guide"),),
        ),
        API + PUBLICATION: _raw(_guidance_parent()),
        API + LEAF: _raw(_html_leaf()),
    }
    with _poll_fixture(tmp_path, monkeypatch, bodies, source_id="UK-05") as case:
        runtime, intake, result, fetched = case
        assert result.status == "READY" and result.item_holds == ()
        assert [(unit.canonical_url, unit.body) for unit in result.units] == [
            (CANONICAL + LEAF, FULL_TEXT),
        ]
        assert len(result.observations) == 4
        assert fetched == list(bodies)
        observations = {value[1]: value for value in result.observations}
        evidence_args = dict(
            units=result.units, sources=runtime.authority.sources,
            objects=runtime.authority.objects, licence=_licence(), proof=runtime.proof,
        )
        assert len(native_evidence_sources(observations=observations, **evidence_args)) == 1
        # Both ancestor declarations are required, not just a matching leaf URL.
        for ancestor in (COLLECTION, PUBLICATION):
            incomplete = {
                key: value for key, value in observations.items()
                if value[0] != API + ancestor
            }
            with pytest.raises(NativeEvidenceHold, match="NATIVE_SOURCE_AUTHORITY_HOLD"):
                native_evidence_sources(observations=incomplete, **evidence_args)
        replay = next(value for value in intake.poll() if value.source_id == "UK-05")
        assert replay.status == "READY" and replay.item_holds == ()
        assert [unit.ingest_id for unit in replay.units] == [unit.ingest_id for unit in result.units]
        assert fetched == list(bodies) * 2


def test_complete_body_types_do_not_ignore_declared_attachment_coverage():
    for document_type in (
        "news_story", "press_release", "guidance", "detailed_guide",
        "html_publication", "notice", "policy_paper", "written_statement",
        "guide", "manual_section", "oral_statement", "statistics", "speech",
    ):
        value = _guidance_parent()
        value["document_type"] = document_type
        if document_type == "guide":
            value["details"]["parts"] = [{
                "slug": "overview", "title": "Overview",
                "body": "<p>" + INTRODUCTION + "</p>",
            }]
        with pytest.raises(GovUkContentHold) as caught:
            parse_govuk_content_document(
                CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW,
            )
        assert caught.value.reason_code == "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE", document_type
        assert caught.value.child_items == ((LEAF, "User guide to police powers and procedures"),), document_type


@pytest.mark.parametrize("inventory", ["attachments", "children", "both"])
def test_guidance_inventory_is_complete_and_deduplicated(inventory):
    value = _guidance_parent()
    if inventory == "attachments":
        del value["links"]["children"]
    elif inventory == "children":
        del value["details"]["attachments"]
    with pytest.raises(GovUkContentHold) as caught:
        parse_govuk_content_document(
            CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW,
        )
    assert caught.value.reason_code == "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE"
    assert caught.value.child_items == ((LEAF, "User guide to police powers and procedures"),)
    assert caught.value.unsupported_attachments == ()
    assert caught.value.exclusion_signals == ()


@pytest.mark.parametrize("document_type", ["guidance", "manual_section"])
def test_absent_or_empty_inventories_preserve_complete_body_behaviour(document_type):
    for inventory in ("absent", "empty"):
        value = _guidance_parent()
        value.update(document_type=document_type, schema_name=(
            "publication" if document_type == "guidance" else "manual_section"
        ))
        if inventory == "empty":
            value["details"]["attachments"] = []
            value["links"]["children"] = []
        else:
            del value["details"]["attachments"]
            del value["links"]["children"]
        document = parse_govuk_content_document(
            CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW,
        )
        assert document.document_type == document_type
        assert document.body_text == INTRODUCTION


@pytest.mark.parametrize("location", ["parent-body", "parent-metadata", "leaf-body", "leaf-metadata"])
def test_guidance_and_html_leaf_rights_veto_retention(tmp_path, monkeypatch, location):
    parent, leaf = _guidance_parent(), _html_leaf()
    target = parent if location.startswith("parent-") else leaf
    if location.endswith("body"):
        target["details"]["body"] = "<p>All rights <em>reserved</em> for this text.</p>"
    else:
        target["details"]["copyright_notice"] = "This is not covered by the Open Government Licence."
    bodies = {
        SOURCE_URLS["UK-01"]: _atom_for(PUBLICATION),
        API + PUBLICATION: _raw(parent),
        API + LEAF: _raw(leaf),
    }
    with _poll_fixture(tmp_path, monkeypatch, bodies) as (_, _, result, fetched):
        assert result.status == "HOLD" and result.units == ()
        held_path = PUBLICATION if location.startswith("parent-") else LEAF
        assert result.item_holds == ((
            CANONICAL + held_path, "SOURCE_ITEM_RIGHTS_EXCLUSION_HOLD",
        ),)
        assert fetched == list(bodies)[:2 if held_path == PUBLICATION else 3]


@pytest.mark.parametrize("mutation", [
    "dangling", "unsafe", "duplicate", "malformed-peer", "base-path", "empty-body",
])
def test_guidance_inventory_never_masks_invalid_metadata(mutation):
    # Malformed populated peers in either direction fail closed even when the
    # other inventory contains a valid declaration.
    for peer in (("attachments", "children") if mutation == "malformed-peer" else (None,)):
        value = _guidance_parent()
        if mutation == "dangling":
            del value["details"]["attachments"][0]["url"]
        elif mutation == "unsafe":
            value["details"]["attachments"][0]["url"] = "/government/publications/../unrelated"
        elif mutation == "duplicate":
            value["details"]["attachments"] *= 2
        elif mutation == "malformed-peer":
            value["details" if peer == "attachments" else "links"][peer] = {"invalid": "inventory"}
        elif mutation == "base-path":
            value["base_path"] = PUBLICATION + "-unrelated"
        else:
            value["details"]["body"] = ""
        with pytest.raises(ValueError) as caught:
            parse_govuk_content_document(
                CANONICAL + PUBLICATION, _raw(value), retrieved_at=NOW,
            )
        assert type(caught.value) is ValueError


def test_guidance_unsupported_pdf_remains_held_beside_retained_html(tmp_path, monkeypatch):
    bodies = {
        SOURCE_URLS["UK-01"]: _atom_for(PUBLICATION),
        API + PUBLICATION: _raw(_guidance_parent(binary=True)),
        API + LEAF: _raw(_html_leaf()),
    }
    with _poll_fixture(tmp_path, monkeypatch, bodies) as (_, _, result, fetched):
        assert result.status == "HOLD"
        assert [(unit.canonical_url, unit.body) for unit in result.units] == [
            (CANONICAL + LEAF, FULL_TEXT),
        ]
        assert result.item_holds == ((
            CANONICAL + PUBLICATION, "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE",
        ),)
        assert fetched == list(bodies)


def test_guidance_terminal_leaf_inventory_is_held_without_losing_complete_sibling(tmp_path, monkeypatch):
    sibling = PUBLICATION + "/definitions"
    parent = _guidance_parent()
    parent["details"]["attachments"].append({
        "attachment_type": "html", "url": sibling, "title": "Definitions",
    })
    parent["links"]["children"].append({"base_path": sibling, "title": "Definitions"})
    nested = _html_leaf()
    nested["details"]["attachments"] = [{
        "attachment_type": "html", "url": LEAF + "/deeper", "title": "Deeper text",
    }]
    bodies = {
        SOURCE_URLS["UK-05"]: _atom_for(COLLECTION),
        API + COLLECTION: _parent_with_children(
            "document_collection", COLLECTION, ((PUBLICATION, "User guide"),),
        ),
        API + PUBLICATION: _raw(parent),
        API + LEAF: _raw(nested),
        API + sibling: _raw(_html_leaf(path=sibling)),
    }
    with _poll_fixture(tmp_path, monkeypatch, bodies, source_id="UK-05") as case:
        _, _, result, fetched = case
        assert result.status == "HOLD"
        assert [(unit.canonical_url, unit.body) for unit in result.units] == [
            (CANONICAL + sibling, FULL_TEXT),
        ]
        assert result.item_holds == ((
            CANONICAL + LEAF, "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE",
        ),)
        assert len(result.observations) == 5
        assert sorted(fetched) == sorted(bodies)
