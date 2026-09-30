"""Provider-free bounds for the collection's additional publication-leaf hop."""

from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
import json
from types import SimpleNamespace

import pytest

from newsroom.authority import ObjectAdmissionId
from newsroom.authority.canonical import digest_bytes
from newsroom.control_plane import native_source_intake as intake_module
from newsroom.control_plane.items import SourceItem
from newsroom.control_plane.native_source_intake import NativeSourceIntake
from newsroom.control_plane.veto import VetoError
from newsroom.increment9.proving import SOURCE_URLS
from newsroom.tests.test_native_source_intake import (
    _atom_for, _document, _parent_with_children, _spreadsheet_parent, _xlsx_asset,
)

COLLECTION = "/government/collections/dfe-update"
PUBLICATION = "/government/publications/dfe-update-current"
LEAF = PUBLICATION + "/education"
API = "https://www.gov.uk/api/content"
NOW = datetime(2026, 9, 26, 12, tzinfo=UTC)


def _stubbed_intake(monkeypatch, responses, *, fence=lambda *_: nullcontext()):
    fetched = []

    def fetch(url):
        fetched.append(url)
        return responses[url]

    intake = NativeSourceIntake(
        sources=None, objects=None, proof=None, definition_ids={}, licence=None,
        dispatch_fence=fence, fetch=fetch, clock=lambda: NOW,
    )
    monkeypatch.setattr(intake, "_admit_observation", lambda *_: (
        SimpleNamespace(admission_id="admission"),
        SimpleNamespace(access_decision_id="access"),
    ))
    monkeypatch.setattr(intake, "_retain_item", lambda *args: (args[4],))
    return intake, fetched


def _settle_collection(intake):
    raw = _parent_with_children(
        "document_collection", COLLECTION, ((PUBLICATION, "Publication"),),
    )
    item = SourceItem("UK-05", COLLECTION, "Collection", "", "https://www.gov.uk" + COLLECTION)
    return intake._settle_item(
        "UK-05", None, None, None, item, API + COLLECTION, raw, NOW, None,
    )


@pytest.mark.parametrize("child", [
    PUBLICATION, COLLECTION, PUBLICATION + "-unrelated/education",
], ids=["self", "ancestor", "non-descendant"])
def test_publication_invalid_leaf_inventory_is_not_followed(monkeypatch, child):
    intake, fetched = _stubbed_intake(monkeypatch, {
        API + PUBLICATION: (200, _parent_with_children(
            "correspondence", PUBLICATION, ((child, "Invalid leaf"),),
        )),
    })
    units, observations, holds = _settle_collection(intake)
    assert units == ()
    assert len(observations) == 2
    assert holds == (("https://www.gov.uk" + PUBLICATION,
                      "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE"),)
    assert fetched == [API + PUBLICATION]


def test_terminal_html_leaf_does_not_follow_children_or_spreadsheet(monkeypatch):
    asset_url = "https://assets.publishing.service.gov.uk/media/asset/funding-values.xlsx"
    terminal = json.loads(_spreadsheet_parent(LEAF, asset_url, _xlsx_asset()))
    terminal["details"]["attachments"].append({
        "attachment_type": "html", "url": LEAF + "/deeper", "title": "Deeper child",
    })
    intake, fetched = _stubbed_intake(monkeypatch, {
        API + PUBLICATION: (200, _parent_with_children(
            "correspondence", PUBLICATION, ((LEAF, "Leaf"),),
        )),
        API + LEAF: (200, json.dumps(terminal).encode()),
    })
    units, observations, holds = _settle_collection(intake)
    assert units == ()
    assert len(observations) == 3
    assert holds == (("https://www.gov.uk" + LEAF,
                      "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE"),)
    assert fetched == [API + PUBLICATION, API + LEAF]


@pytest.mark.parametrize("failure, expected_hold", [
    ("fetch", "SOURCE_ITEM_FETCH_INCOMPLETE"),
    ("exclusion", "SOURCE_ITEM_RIGHTS_EXCLUSION_HOLD"),
    ("metadata", "SOURCE_ITEM_METADATA_HOLD"),
])
def test_partial_publication_leaves_keep_success_and_exact_hold(monkeypatch, failure, expected_hold):
    failed_leaf = PUBLICATION + "/academies"
    failed = json.loads(_document(path=failed_leaf))
    if failure == "exclusion":
        failed["details"]["copyright_notice"] = "Not covered by the Open Government Licence."
    elif failure == "metadata":
        failed["public_updated_at"] = "2026-09-27T12:00:00Z"
    intake, fetched = _stubbed_intake(monkeypatch, {
        API + PUBLICATION: (200, _parent_with_children(
            "correspondence", PUBLICATION, ((LEAF, "Complete"), (failed_leaf, "Failed")),
        )),
        API + LEAF: (200, _document(path=LEAF)),
        API + failed_leaf: (503, b"") if failure == "fetch" else (200, json.dumps(failed).encode()),
    })
    units, observations, holds = _settle_collection(intake)
    assert [item.canonical_url for item in units] == ["https://www.gov.uk" + LEAF]
    assert holds == (("https://www.gov.uk" + failed_leaf, expected_hold),)
    assert len(observations) == (3 if failure == "fetch" else 4)
    assert set(fetched) == {API + PUBLICATION, API + LEAF, API + failed_leaf}


def test_publication_leaf_stop_fence_precedes_fetch(monkeypatch):
    fenced = []

    @contextmanager
    def fence(_source_id, url):
        fenced.append(url)
        if url == API + LEAF:
            raise VetoError("owner stop")
        yield

    intake, fetched = _stubbed_intake(monkeypatch, {
        API + PUBLICATION: (200, _parent_with_children(
            "correspondence", PUBLICATION, ((LEAF, "Leaf"),),
        )),
    }, fence=fence)
    with pytest.raises(VetoError, match="owner stop"):
        _settle_collection(intake)
    assert fetched == [API + PUBLICATION]
    assert fenced == [API + PUBLICATION, API + LEAF]


@pytest.mark.parametrize("mismatch", ["unlisted-leaf", "unlisted-publication", "manual-parent"])
def test_additional_leaf_inventory_binding_rejects_unlisted_and_manual_parent(monkeypatch, mismatch):
    publication = _parent_with_children("correspondence", PUBLICATION, ((LEAF, "Leaf"),))
    collection = _parent_with_children(
        "manual" if mismatch == "manual-parent" else "document_collection",
        COLLECTION,
        ((PUBLICATION + "-other" if mismatch == "unlisted-publication" else PUBLICATION, "Publication"),),
    )
    raw_by_admission, observations = {}, {}
    for index, (url, raw) in enumerate((
        (SOURCE_URLS["UK-05"], _atom_for(COLLECTION)),
        (API + COLLECTION, collection), (API + PUBLICATION, publication),
    ), start=1):
        admission = ObjectAdmissionId.parse(f"00000000-0000-4000-8000-{index:012d}")
        raw_by_admission[str(admission)] = raw
        digest = digest_bytes(raw)
        observations[digest] = (url, digest, str(admission), "access")
    # Isolate declaration/ancestry bounds; existing native-runtime tests prove access authority.
    monkeypatch.setattr(intake_module, "_require_observation_access", lambda *, observation, **_: (
        ObjectAdmissionId.parse(observation[2]),
        SimpleNamespace(allowed_bytes=len(raw_by_admission[observation[2]])),
    ))
    objects = SimpleNamespace(rehydrate=lambda request, **_: SimpleNamespace(
        data=raw_by_admission[str(request.admission_id)],
    ))
    path = PUBLICATION + "/unlisted" if mismatch == "unlisted-leaf" else LEAF
    unit = SimpleNamespace(
        source_id="UK-05", item_key=digest_bytes(publication) + "|" + path,
        canonical_url="https://www.gov.uk" + path,
        source_definition_url=SOURCE_URLS["UK-05"], observed_at="2026-09-26T12:00:00.000000Z",
    )
    with pytest.raises(ValueError, match=(
        "child is outside its retained parent inventory" if mismatch == "unlisted-leaf"
        else "parent is outside its retained feed inventory"
    )):
        intake_module._require_parent_inventory_binding(
            unit=unit, observations=observations, objects=objects, proof=None,
        )
