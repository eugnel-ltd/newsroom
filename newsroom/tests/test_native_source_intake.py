from contextlib import contextmanager, nullcontext
import io
import json
import sqlite3
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from newsroom.authority import ObjectAdmissionId
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane.govuk_rights import GovUkLicenceEvidence, POLICY_DIGEST
from newsroom.control_plane.graphiti_operational_readiness import _source_requests
from newsroom.control_plane.graphiti_operational_readiness import (
    OPERATOR_AUTHORITY_DOMAIN,
    OPERATOR_PRINCIPAL_ID,
)
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.native_evidence import NativeEvidenceHold
from newsroom.control_plane.native_source_intake import (
    NativeSourceIntake, _fetch_exact, native_evidence_sources,
)
from newsroom.control_plane.native_source_definitions import (
    native_source_definition_requests,
)
from newsroom.increment9.proving import MAX_BODY_BYTES, SOURCE_IDS, SOURCE_URLS
from newsroom.sources import SourceItemId, SourceRevisionId
from newsroom.tests.test_graphiti_operational_readiness import _rights, _unit
from newsroom.tests.test_native_runtime import _args

ATOM = b'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><id>item-1</id><title>Visa rules updated</title><summary>The official deadline changed.</summary><link href="https://www.gov.uk/item-1"/><published>2026-09-08T10:00:00Z</published><updated>2026-09-08T11:00:00Z</updated></entry></feed>'''


def _source_read_audit_counts(path):
    with sqlite3.connect(path) as connection:
        return tuple(connection.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0] for table in (
            "object_access_decisions", "authentication_contexts",
            "authorization_requests", "authorization_decisions",
        ))


def _atom_for(path):
    return ATOM.replace(b"https://www.gov.uk/item-1", ("https://www.gov.uk" + path).encode())


def _document(
    *, body="The complete maintained-page text.",
    published="2026-09-08T10:00:00Z", updated="2026-09-08T11:00:00Z",
    path="/item-1",
):
    return json.dumps({
        "base_path": path,
        "locale": "en",
        "document_type": "news_story",
        "withdrawn_notice": None,
        "first_published_at": published,
        "public_updated_at": updated,
        "title": "Visa rules updated",
        "details": {"body": f"<p>{body}</p>"},
        "links": {"organisations": [{"title": "Home Office"}]},
    }, separators=(",", ":")).encode()


def _future_announcement(*, path="/item-2"):
    return json.dumps({
        "base_path": path,
        "locale": "en",
        "document_type": "official_statistics_announcement",
        "withdrawn_notice": None,
        "first_published_at": "2026-09-08T10:00:00Z",
        "public_updated_at": "2026-09-08T11:00:00Z",
        "title": "Future official statistics",
        "details": {
            "display_date": "10 September 2026 9:30am",
            "release_timestamp": "2026-09-10T09:30:00+01:00",
            "state": "confirmed",
        },
        "links": {"organisations": [{"title": "Home Office"}]},
    }, separators=(",", ":")).encode()


def _licence():
    return GovUkLicenceEvidence(
        (
            ObjectAdmissionId.parse("00000000-0000-4000-8000-000000000101"),
            ObjectAdmissionId.parse("00000000-0000-4000-8000-000000000102"),
        ),
        ("sha256:" + "1" * 64, "sha256:" + "2" * 64),
        "2026-09-08T09:00:00.000000Z",
        POLICY_DIGEST,
    )


def _seed_uk01(runtime):
    unit = replace(_unit(), source_definition_url=SOURCE_URLS["UK-01"])
    definition, version, *_ = _source_requests(unit, _rights())
    runtime.authority.sources.register_definition(definition, proof=runtime.proof)
    runtime.authority.sources.record_definition_version(version, proof=runtime.proof)
    return definition.definition_id


def _seed_missing(runtime, source_id):
    requests = native_source_definition_requests(
        source_id=source_id,
        rights=_licence().for_source(
            source_id=source_id, definition_url=SOURCE_URLS[source_id],
        ),
    )
    runtime.authority.sources.register_definition(requests.definition, proof=runtime.proof)
    runtime.authority.sources.record_definition_version(requests.version, proof=runtime.proof)
    return requests.definition.definition_id


def _guide():
    return json.dumps({
        "base_path": "/british-national-overseas-bno-visa", "locale": "en",
        "document_type": "guide", "title": "British National (Overseas) visa",
        "first_published_at": "2020-01-01T00:00:00Z",
        "public_updated_at": "2026-09-08T10:00:00Z", "withdrawn_notice": None,
        "details": {"parts": [
            {"title": f"Part {number}", "slug": f"part-{number}",
             "body": f"<p>Complete guide text {number}.</p>"}
            for number in range(1, 10)
        ]},
        "links": {"organisations": [{"title": "Home Office"}]},
    }).encode()


def _manual():
    return json.dumps({
        "base_path": "/guidance/immigration-rules", "locale": "en",
        "document_type": "manual", "title": "Immigration Rules",
        "first_published_at": "2020-01-01T00:00:00Z",
        "public_updated_at": "2026-09-08T10:00:00Z", "withdrawn_notice": None,
        "details": {"child_section_groups": [{"title": "Rules", "child_sections": [
            {"base_path": "/guidance/immigration-rules/part-1", "title": "Part 1"},
            {"base_path": "/guidance/immigration-rules/part-2", "title": "Part 2"},
        ]}]},
        "links": {"organisations": [{"title": "Home Office"}]},
    }).encode()


def _manual_section(number):
    return json.dumps({
        "base_path": f"/guidance/immigration-rules/part-{number}", "locale": "en",
        "document_type": "manual_section", "title": f"Part {number}",
        "first_published_at": "2020-01-01T00:00:00Z",
        "public_updated_at": "2026-09-08T10:00:00Z", "withdrawn_notice": None,
        "details": {"body": f"<p>Complete rules section {number}.</p>",
                    "manual": {"organisations": [{"title": "Home Office"}]}},
        "links": {},
    }).encode()


def _parent_with_children(document_type, path, children, *, binary=False):
    value = json.loads(_document(path=path))
    value.update(document_type=document_type, schema_name=(
        "document_collection" if document_type == "document_collection" else "publication"
    ))
    declared = [{"base_path": child, "title": title} for child, title in children]
    if document_type == "manual":
        value["details"] = {"child_section_groups": [{
            "title": "Sections", "child_sections": declared,
        }]}
    elif document_type == "document_collection":
        value["links"]["documents"] = declared
    else:
        attachments = [{
            "attachment_type": "html", "url": child, "title": title,
        } for child, title in children]
        if binary:
            attachments.append({
                "attachment_type": "file",
                "url": "https://assets.publishing.service.gov.uk/media/report.pdf",
                "title": "Signed circular",
            })
        value["details"]["attachments"] = attachments
        # The observed circular/corporate-report shape declares the same HTML
        # child in both inventories. Its exact path is still one child.
        value["links"]["children"] = declared
    return json.dumps(value, separators=(",", ":")).encode()


def _xlsx_asset() -> bytes:
    workbook = b'''<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <sheets><sheet name="Published" sheetId="1" r:id="rId1"/></sheets>
</workbook>'''
    relationships = b'''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>'''
    worksheet = b'''<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
 <sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Provider</t></is></c><c r="B1" t="inlineStr"><is><t>Allocation</t></is></c></row>
 <row r="2"><c r="A2" t="inlineStr"><is><t>Example College</t></is></c><c r="B2"><v>125000</v></c></row></sheetData>
</worksheet>'''
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, body in (
            ("xl/workbook.xml", workbook),
            ("xl/_rels/workbook.xml.rels", relationships),
            ("xl/worksheets/sheet1.xml", worksheet),
        ):
            info = zipfile.ZipInfo(name, (2026, 9, 25, 12, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, body)
    return output.getvalue()


def _spreadsheet_parent(path: str, asset_url: str, asset: bytes, *, pdf=False):
    value = json.loads(_document(path=path, updated="2026-09-25T12:00:00Z"))
    value.update(document_type="transparency", schema_name="publication")
    value["details"] = {"attachments": [{
        "attachment_type": "file",
        "url": asset_url,
        "title": "Final funding values",
        "filename": asset_url.rsplit("/", 1)[-1],
        "content_type": ("text/csv" if asset_url.endswith(".csv") else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        "file_size": len(asset),
        "id": "asset-xlsx",
        "locale": "en",
    }]}
    if pdf:
        value["details"]["attachments"].append({
            "attachment_type": "file",
            "url": "https://assets.publishing.service.gov.uk/media/report.pdf",
            "title": "Uncovered report",
        })
    return json.dumps(value, separators=(",", ":")).encode()


def test_native_source_poll_retains_real_lineage_replay_and_all_dispositions(
    tmp_path, monkeypatch,
):
    args = _args(tmp_path, monkeypatch)
    args["principal_id"] = OPERATOR_PRINCIPAL_ID
    args["authority_domain"] = OPERATOR_AUTHORITY_DOMAIN
    page = [_document(published="2026-09-08T11:00:12Z")]
    fences = []
    instant = [datetime(2026, 9, 8, 12, tzinfo=UTC)]

    with open_native_runtime(**args) as runtime:
        definition_id = _seed_uk01(runtime)
        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=runtime.proof, definition_ids={"UK-01": definition_id},
            licence=_licence(), dispatch_fence=lambda source, url: nullcontext(fences.append((source, url))),
            fetch=lambda url: (200, ATOM if url == SOURCE_URLS["UK-01"] else page[0]),
            clock=lambda: instant[0],
        )
        first = intake.poll()
        assert tuple(item.source_id for item in first) == SOURCE_IDS
        assert len(first) == 10
        assert first[0].status == "READY" and first[0].units, first[0]
        assert all(item.status == "HOLD" for item in first[1:])
        unit = first[0].units[0]
        assert unit.proving_run_id.startswith("native-source:sha256:")
        assert unit.authority is not None
        assert unit.body == "The complete maintained-page text."
        assert unit.published_at == "2026-09-08T11:00:12.000000Z"
        assert unit.updated_at == "2026-09-08T11:00:00.000000Z"
        assert len(first[0].observations) == 2
        evidence_sources = native_evidence_sources(
            units=first[0].units, sources=runtime.authority.sources,
            objects=runtime.authority.objects,
            observations={item[1]: item for item in first[0].observations},
            licence=_licence(), proof=runtime.proof,
        )
        assert len(evidence_sources) == 1
        assert evidence_sources[0].unit == unit
        assert evidence_sources[0].dependency.dependency_status == "RESOLVED"
        assert evidence_sources[0].dependency.evidential_origin_id == unit.observation_digest
        audit_counts = _source_read_audit_counts(args["authority_path"])
        assert native_evidence_sources(
            units=first[0].units, sources=runtime.authority.sources,
            objects=runtime.authority.objects,
            observations={item[1]: item for item in first[0].observations},
            licence=_licence(), proof=runtime.proof,
        ) == evidence_sources
        assert _source_read_audit_counts(args["authority_path"]) == audit_counts
        with pytest.raises(NativeEvidenceHold, match="NATIVE_SOURCE_CHUNK_BINDING_HOLD"):
            native_evidence_sources(
                units=(replace(unit, observation_digest="sha256:" + "f" * 64),),
                sources=runtime.authority.sources, objects=runtime.authority.objects,
                observations={item[1]: item for item in first[0].observations},
                licence=_licence(), proof=runtime.proof,
            )
        bad_observations = {item[1]: item for item in first[0].observations}
        bad_observations[unit.observation_digest] = (
            "https://www.gov.uk/api/content/unrelated",
            *bad_observations[unit.observation_digest][1:],
        )
        with pytest.raises(NativeEvidenceHold, match="NATIVE_SOURCE_AUTHORITY_HOLD"):
            native_evidence_sources(
                units=first[0].units, sources=runtime.authority.sources,
                objects=runtime.authority.objects, observations=bad_observations,
                licence=_licence(), proof=runtime.proof,
            )
        old_revision = unit.authority.revision_id

        replay = intake.poll()[0].units[0]
        assert replay.ingest_id == unit.ingest_id
        assert replay.authority.revision_id == old_revision

        page[0] = _document(body="A changed complete document.", updated="2026-09-08T12:30:00Z")
        instant[0] = datetime(2026, 9, 8, 13, tzinfo=UTC)
        changed = intake.poll()[0].units[0]
        assert changed.authority.revision_id != old_revision
        retained = runtime.authority.sources.revision(
            SourceRevisionId.parse(changed.authority.revision_id), proof=runtime.proof
        )
        assert str(retained.request.prior_revision_id) == old_revision
        latest = runtime.authority.sources.latest_revision(
            SourceItemId.parse(changed.authority.item_id), proof=runtime.proof
        )
        assert latest == retained
        # Re-observing the second revision must retain its predecessor, rather
        # than rewriting the original immutable request as a first revision.
        assert intake.poll()[0].units[0].authority.revision_id == changed.authority.revision_id

        page[0] = _document(updated="2026-09-08T14:00:00Z")
        instant[0] = datetime(2026, 9, 8, 15, tzinfo=UTC)
        returned = intake.poll()[0].units[0]
        assert returned.authority.revision_id not in {
            old_revision, changed.authority.revision_id,
        }
        returned_revision = runtime.authority.sources.revision(
            SourceRevisionId.parse(returned.authority.revision_id), proof=runtime.proof
        )
        assert str(returned_revision.request.prior_revision_id) == changed.authority.revision_id
        assert returned.revision_digest == unit.revision_digest
        assert returned_revision.request.observed_at.to_text() == "2026-09-08T15:00:00.000000Z"
        assert fences == [
            item
            for _ in range(5)
            for item in (
                ("UK-01", SOURCE_URLS["UK-01"]),
                ("UK-01", "https://www.gov.uk/api/content/item-1"),
            )
        ]
        # A new source version with unchanged text is still observed now; it
        # must not borrow the preceding version's observation timestamp.
        page[0] = _document(updated="2026-09-08T16:00:00Z")
        instant[0] = datetime(2026, 9, 8, 17, tzinfo=UTC)
        metadata_only = intake.poll()[0].units[0]
        metadata_revision = runtime.authority.sources.revision(
            SourceRevisionId.parse(metadata_only.authority.revision_id), proof=runtime.proof
        )
        assert str(metadata_revision.request.prior_revision_id) == returned.authority.revision_id
        assert metadata_revision.request.observed_at.to_text() == "2026-09-08T17:00:00.000000Z"
        assert metadata_only.revision_digest == returned.revision_digest


def test_native_source_reobservation_reuses_journal_units_after_current_checks(tmp_path, monkeypatch):
    import newsroom.control_plane.corpus as corpus
    import newsroom.control_plane.native_source_intake as intake_module

    args = _args(tmp_path, monkeypatch)
    args["principal_id"] = OPERATOR_PRINCIPAL_ID
    args["authority_domain"] = OPERATOR_AUTHORITY_DOMAIN
    retained, admitted, fetched = {}, [], []
    page = [_document(body="Complete retained text. " * 750)]
    permitted = [True]
    instant = [datetime(2026, 9, 8, 12, tzinfo=UTC)]
    licence = _licence()
    with open_native_runtime(**args) as runtime:
        def admit(request, data, **kwargs):
            admitted.append(request.admission_type)
            return runtime.authority.objects.admit(request, data, **kwargs)

        def fetch(url):
            fetched.append(url)
            return 200, ATOM if url == SOURCE_URLS["UK-01"] else page[0]

        intake = NativeSourceIntake(
            sources=runtime.authority.sources,
            objects=SimpleNamespace(
                admit=admit, hydrate=runtime.authority.objects.hydrate,
                latest_access_decision=runtime.authority.objects.latest_access_decision,
            ),
            proof=runtime.proof, definition_ids={"UK-01": _seed_uk01(runtime)},
            licence=SimpleNamespace(for_source=lambda **kw: (
                licence.for_source(**kw) if permitted[0] else SimpleNamespace(decision="DENIED")
            )), dispatch_fence=lambda *_: nullcontext(), fetch=fetch,
            retained_units=retained,
            clock=lambda: instant[0],
        )
        first = intake.poll()[0]
        assert first.status == "READY", first
        assert len(first.units) == 3
        retained[first.units[0].revision_id] = first.units
        admitted.clear()
        hashed_sizes = []
        original_digest = corpus.content_digest

        def count_digest(**value):
            hashed_sizes.append(len(canonical_json_bytes(value)))
            return original_digest(**value)

        unit = first.units[0]
        revision_bytes = len(canonical_json_bytes({
            "headline": unit.headline, "body": unit.body,
            "canonical_url": unit.canonical_url,
        }))
        with monkeypatch.context() as counting:
            counting.setattr(corpus, "content_digest", count_digest)
            counting.setattr(intake_module, "content_digest", count_digest)
            replay = intake.poll()[0]
        assert replay.units == first.units
        # One current observation, one integrity hash and one representation
        # binding per retained chunk; no repeated first-chunk baseline hash.
        assert hashed_sizes == [revision_bytes] * (1 + 2 * len(first.units))
        assert fetched == [SOURCE_URLS["UK-01"], "https://www.gov.uk/api/content/item-1"] * 2
        # Fresh observations remain governed; unchanged corpus chunks are not re-admitted.
        assert admitted == ["source.native-observation", "source.native-observation"]
        retained[first.units[0].revision_id] = tuple(
            replace(unit, authority=replace(unit.authority, definition_version_id="another-version"))
            for unit in first.units
        )
        held = intake.poll()[0]
        assert held.item_holds == (("https://www.gov.uk/item-1", "SOURCE_RETAINED_REVISION_BINDING_HOLD"),)
        retained[first.units[0].revision_id] = first.units
        page[0] = _document(body="A newly changed complete document.", updated="2026-09-08T13:00:00Z")
        instant[0] = datetime(2026, 9, 8, 14, tzinfo=UTC)
        changed = intake.poll()[0]
        assert changed.status == "READY" and changed.units[0].revision_id != first.units[0].revision_id
        permitted[0] = False
        before = len(fetched)
        assert intake.poll()[0].reason_code == "CURRENT_RIGHTS_HOLD"
        assert len(fetched) == before


def test_native_source_poll_holds_oversize_and_duplicate_native_revision(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    args["principal_id"] = OPERATOR_PRINCIPAL_ID
    args["authority_domain"] = OPERATOR_AUTHORITY_DOMAIN
    feed = [ATOM]
    page = [_document()]

    with open_native_runtime(**args) as runtime:
        definition_id = _seed_uk01(runtime)
        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=runtime.proof, definition_ids={"UK-01": definition_id},
            licence=_licence(), dispatch_fence=lambda *_: nullcontext(),
            fetch=lambda url: (200, feed[0] if url == SOURCE_URLS["UK-01"] else page[0]),
            clock=lambda: datetime(2026, 9, 8, 12, tzinfo=UTC),
        )
        assert intake.poll()[0].status == "READY"

        page[0] = _document(body="Different bytes.")
        duplicate = intake.poll()[0]
        assert duplicate.status == "HOLD"
        assert duplicate.reason_code == "SOURCE_ITEMS_HELD"
        assert duplicate.item_holds == ((
            "https://www.gov.uk/item-1", "SOURCE_NATIVE_REVISION_CONFLICT",
        ),)
        assert not duplicate.units

        page[0] = b"x" * (MAX_BODY_BYTES + 1)
        oversized_page = intake.poll()[0]
        assert oversized_page.reason_code == "SOURCE_ITEMS_HELD"
        assert oversized_page.item_holds[0][1] == "SOURCE_ITEM_BODY_TOO_LARGE"
        assert oversized_page.observation_admission_id is not None

        feed[0] = b"x" * (MAX_BODY_BYTES + 1)
        oversized = intake.poll()[0]
        assert oversized.status == "HOLD"
        assert oversized.reason_code == "SOURCE_BODY_TOO_LARGE"
        assert oversized.observation_admission_id is None


@pytest.mark.parametrize(("second_page", "reason_code"), [
    (b"{}", "SOURCE_ITEM_METADATA_HOLD"),
    (_future_announcement(), "SOURCE_ITEM_NOT_YET_PUBLISHED"),
])
def test_native_source_poll_preserves_completed_items_when_later_item_holds(
    tmp_path, monkeypatch, second_page, reason_code,
):
    args = _args(tmp_path, monkeypatch)
    args["principal_id"] = OPERATOR_PRINCIPAL_ID
    args["authority_domain"] = OPERATOR_AUTHORITY_DOMAIN
    second = ATOM.split(b"<entry>", 1)[1].replace(
        b"item-1", b"item-2"
    ).replace(b"/item-1", b"/item-2")
    body = [ATOM.replace(b"</feed>", b"<entry>" + second)]

    with open_native_runtime(**args) as runtime:
        definition_id = _seed_uk01(runtime)
        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=runtime.proof, definition_ids={"UK-01": definition_id},
            licence=_licence(), dispatch_fence=lambda *_: nullcontext(),
            fetch=lambda url: (
                (200, body[0]) if url == SOURCE_URLS["UK-01"]
                else (200, _document()) if url.endswith("/item-1")
                else (200, second_page)
            ),
            clock=lambda: datetime(2026, 9, 8, 12, tzinfo=UTC),
        )
        held = intake.poll()[0]
        assert held.status == "HOLD"
        assert held.reason_code == "SOURCE_ITEMS_HELD"
        assert held.item_holds == ((
            "https://www.gov.uk/item-2", reason_code,
        ),)
        assert held.units
        assert len(held.observations) == 3
        retained = held.units[0]
        latest = runtime.authority.sources.latest_revision(
            SourceItemId.parse(retained.authority.item_id), proof=runtime.proof
        )
        assert str(latest.request.revision_id) == retained.authority.revision_id

        body[0] = ATOM
        replay = intake.poll()[0]
        assert replay.status == "READY"
        assert replay.units[0].authority.revision_id == retained.authority.revision_id


def test_native_exact_fetch_rejects_any_endpoint_outside_fixed_portfolio():
    try:
        _fetch_exact("https://example.com/not-approved")
    except ValueError as exc:
        assert str(exc) == "native source endpoint is not approved"
    else:
        raise AssertionError("unapproved endpoint was fetched")


def test_native_exact_asset_fetch_rejects_response_identity_drift(monkeypatch):
    asset_url = (
        "https://assets.publishing.service.gov.uk/media/asset/"
        "funding-values.xlsx"
    )

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def geturl(self):
            return asset_url + "?redirected=1"

    class Opener:
        def open(self, _request, timeout):
            assert timeout > 0
            return Response(b"spreadsheet")

    monkeypatch.setattr(
        "newsroom.control_plane.native_source_intake.urllib.request.build_opener",
        lambda *_args: Opener(),
    )
    with pytest.raises(ValueError, match="response identity differs"):
        _fetch_exact(asset_url)


def test_native_bno_guide_retains_all_nine_parts_as_one_revision(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    args["principal_id"] = OPERATOR_PRINCIPAL_ID
    args["authority_domain"] = OPERATOR_AUTHORITY_DOMAIN
    with open_native_runtime(**args) as runtime:
        definition_id = _seed_missing(runtime, "UK-02")
        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=runtime.proof, definition_ids={"UK-02": definition_id},
            licence=_licence(), dispatch_fence=lambda *_: nullcontext(),
            fetch=lambda _: (200, _guide()),
            clock=lambda: datetime(2026, 9, 8, 12, tzinfo=UTC),
        )
        disposition = intake.poll()[SOURCE_IDS.index("UK-02")]
        assert disposition.status == "READY"
        assert len(disposition.observations) == 1
        text = disposition.units[0].body
        assert all(f"Part {number}\nComplete guide text {number}." in text
                   for number in range(1, 10))
        assert len({unit.authority.revision_id for unit in disposition.units}) == 1


def test_native_manual_retains_every_section_and_root_inventory(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    args["principal_id"] = OPERATOR_PRINCIPAL_ID
    args["authority_domain"] = OPERATOR_AUTHORITY_DOMAIN
    bodies = {
        SOURCE_URLS["UK-03"]: _manual(),
        "https://www.gov.uk/api/content/guidance/immigration-rules/part-1": _manual_section(1),
        "https://www.gov.uk/api/content/guidance/immigration-rules/part-2": _manual_section(2),
    }
    with open_native_runtime(**args) as runtime:
        definition_id = _seed_missing(runtime, "UK-03")
        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=runtime.proof, definition_ids={"UK-03": definition_id},
            licence=_licence(), dispatch_fence=lambda *_: nullcontext(),
            fetch=lambda url: (200, bodies[url]),
            clock=lambda: datetime(2026, 9, 8, 12, tzinfo=UTC),
        )
        disposition = intake.poll()[SOURCE_IDS.index("UK-03")]
        assert disposition.status == "READY"
        assert len(disposition.observations) == 3
        revision_units = {}
        for unit in disposition.units:
            revision_units.setdefault(unit.revision_id, []).append(unit)
        assert len(revision_units) == 2
        assert {units[0].canonical_url for units in revision_units.values()} == {
            "https://www.gov.uk/guidance/immigration-rules/part-1",
            "https://www.gov.uk/guidance/immigration-rules/part-2",
        }
        observations = {item[1]: item for item in disposition.observations}
        for units in revision_units.values():
            assert native_evidence_sources(
                units=tuple(units), sources=runtime.authority.sources,
                objects=runtime.authority.objects, observations=observations,
                licence=_licence(), proof=runtime.proof,
            )
        root = disposition.units[0].item_key.split("|", 1)[0]
        tampered = dict(observations)
        tampered.pop(root)
        with pytest.raises(NativeEvidenceHold, match="NATIVE_SOURCE_AUTHORITY_HOLD"):
            native_evidence_sources(
                units=tuple(next(iter(revision_units.values()))),
                sources=runtime.authority.sources, objects=runtime.authority.objects,
                observations=tampered, licence=_licence(), proof=runtime.proof,
            )
        unrelated_root = dict(observations)
        unrelated_root[root] = (
            "https://www.gov.uk/api/content/guidance/unrelated-manual",
            *unrelated_root[root][1:],
        )
        with pytest.raises(NativeEvidenceHold, match="NATIVE_SOURCE_AUTHORITY_HOLD"):
            native_evidence_sources(
                units=tuple(next(iter(revision_units.values()))),
                sources=runtime.authority.sources, objects=runtime.authority.objects,
                observations=unrelated_root, licence=_licence(), proof=runtime.proof,
            )


@pytest.mark.parametrize(("document_type", "parent_path", "child_path", "binary"), [
    (
        "correspondence",
        "/government/publications/circular-0122026-cpi-and-the-police-pension-scheme-2015",
        "/government/publications/circular-0122026-cpi-and-the-police-pension-scheme-2015/circular-0122026",
        True,
    ),
    (
        "document_collection", "/government/collections/visa-guidance",
        "/government/publications/visa-guidance", False,
    ),
    (
        "manual", "/guidance/visa-manual",
        "/guidance/visa-manual/section-one", False,
    ),
    (
        "corporate_report", "/government/publications/annual-report",
        "/government/publications/annual-report/accounts", False,
    ),
])
def test_feed_parent_settles_each_exact_declared_html_child(
    tmp_path, monkeypatch, document_type, parent_path, child_path, binary,
):
    args = _args(tmp_path, monkeypatch)
    args.update(principal_id=OPERATOR_PRINCIPAL_ID, authority_domain=OPERATOR_AUTHORITY_DOMAIN)
    parent = _parent_with_children(
        document_type, parent_path, ((child_path, "Declared child"),), binary=binary,
    )
    bodies = {
        SOURCE_URLS["UK-01"]: _atom_for(parent_path),
        "https://www.gov.uk/api/content" + parent_path: parent,
        "https://www.gov.uk/api/content" + child_path: _document(path=child_path),
    }
    fetched = []
    with open_native_runtime(**args) as runtime:
        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=runtime.proof, definition_ids={"UK-01": _seed_uk01(runtime)},
            licence=_licence(), dispatch_fence=lambda *_: nullcontext(),
            fetch=lambda url: (fetched.append(url), (200, bodies[url]))[1],
            clock=lambda: datetime(2026, 9, 8, 12, tzinfo=UTC),
        )
        disposition = intake.poll()[0]
        assert len(disposition.units) == 1
        unit = disposition.units[0]
        assert unit.canonical_url == "https://www.gov.uk" + child_path
        assert unit.item_key.endswith("|" + child_path)
        assert all(item.canonical_url != "https://www.gov.uk" + parent_path
                   for item in disposition.units)
        assert len(disposition.observations) == 3
        evidence_sources = native_evidence_sources(
            units=disposition.units, sources=runtime.authority.sources,
            objects=runtime.authority.objects,
            observations={item[1]: item for item in disposition.observations},
            licence=_licence(), proof=runtime.proof,
        )
        assert evidence_sources
        audit_counts = _source_read_audit_counts(args["authority_path"])
        assert native_evidence_sources(
            units=disposition.units, sources=runtime.authority.sources,
            objects=runtime.authority.objects,
            observations={item[1]: item for item in disposition.observations},
            licence=_licence(), proof=runtime.proof,
        ) == evidence_sources
        assert _source_read_audit_counts(args["authority_path"]) == audit_counts
        if binary:
            assert disposition.status == "HOLD"
            assert disposition.item_holds == ((
                "https://www.gov.uk" + parent_path,
                "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE",
            ),)
            assert all("assets.publishing.service.gov.uk" not in url for url in fetched)
        else:
            assert disposition.status == "READY"
            assert disposition.item_holds == ()


@pytest.mark.parametrize(("with_pdf", "expected_status"), [
    (False, "READY"),
    (True, "HOLD"),
])
@pytest.mark.parametrize("suffix", ("xlsx", "csv"))
def test_declared_table_is_retained_without_hiding_unsupported_peers(
    tmp_path, monkeypatch, with_pdf, expected_status, suffix,
) -> None:
    args = _args(tmp_path, monkeypatch)
    args.update(
        principal_id=OPERATOR_PRINCIPAL_ID,
        authority_domain=OPERATOR_AUTHORITY_DOMAIN,
    )
    parent_path = "/government/publications/funding-values"
    asset_url = (
        "https://assets.publishing.service.gov.uk/media/asset/"
        f"funding-values.{suffix}"
    )
    asset = _xlsx_asset() if suffix == "xlsx" else b"Provider,Funding\nExample College,125000\n"
    parent = _spreadsheet_parent(
        parent_path, asset_url, asset, pdf=with_pdf,
    )
    bodies = {
        SOURCE_URLS["UK-01"]: _atom_for(parent_path),
        "https://www.gov.uk/api/content" + parent_path: parent,
        asset_url: asset,
    }
    fetched = []
    fences = []
    with open_native_runtime(**args) as runtime:
        intake = NativeSourceIntake(
            sources=runtime.authority.sources,
            objects=runtime.authority.objects,
            proof=runtime.proof,
            definition_ids={"UK-01": _seed_uk01(runtime)},
            licence=_licence(),
            dispatch_fence=lambda source, url: nullcontext(
                fences.append((source, url))
            ),
            fetch=lambda url: (fetched.append(url), (200, bodies[url]))[1],
            clock=lambda: datetime(2026, 9, 26, 12, tzinfo=UTC),
        )
        disposition = intake.poll()[0]
        assert disposition.status == expected_status
        assert len(disposition.units) == 1
        unit = disposition.units[0]
        assert unit.canonical_url == "https://www.gov.uk" + parent_path
        assert unit.item_key.endswith("|" + asset_url)
        assert unit.observation_digest == digest_bytes(asset)
        assert 'Row 1: A="Provider"' in unit.body
        assert ('Row 2: A="Example College"; B=' + ('125000' if suffix == "xlsx" else '"125000"')) in unit.body
        assert len(disposition.observations) == 3
        observations = {item[1]: item for item in disposition.observations}
        evidence_sources = native_evidence_sources(
            units=disposition.units,
            sources=runtime.authority.sources,
            objects=runtime.authority.objects,
            observations=observations,
            licence=_licence(),
            proof=runtime.proof,
        )
        assert len(evidence_sources) == 1
        root_digest = unit.item_key.split("|", 1)[0]
        tampered = dict(observations)
        tampered[root_digest] = (
            "https://www.gov.uk/api/content/government/publications/unrelated",
            *tampered[root_digest][1:],
        )
        with pytest.raises(
            NativeEvidenceHold, match="NATIVE_SOURCE_AUTHORITY_HOLD"
        ):
            native_evidence_sources(
                units=disposition.units,
                sources=runtime.authority.sources,
                objects=runtime.authority.objects,
                observations=tampered,
                licence=_licence(),
                proof=runtime.proof,
            )
    assert fetched == [
        SOURCE_URLS["UK-01"],
        "https://www.gov.uk/api/content" + parent_path,
        asset_url,
    ]
    assert ("UK-01", asset_url) in fences
    assert disposition.item_holds == (
        (("https://www.gov.uk" + parent_path,
          "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE"),)
        if with_pdf else ()
    )


def test_declared_spreadsheet_parse_failure_retains_raw_and_parent_hold(
    tmp_path, monkeypatch,
) -> None:
    args = _args(tmp_path, monkeypatch)
    args.update(
        principal_id=OPERATOR_PRINCIPAL_ID,
        authority_domain=OPERATOR_AUTHORITY_DOMAIN,
    )
    parent_path = "/government/publications/funding-values"
    asset_url = (
        "https://assets.publishing.service.gov.uk/media/asset/"
        "funding-values.xlsx"
    )
    declared = _xlsx_asset()
    malformed = b"x" * len(declared)
    bodies = {
        SOURCE_URLS["UK-01"]: _atom_for(parent_path),
        "https://www.gov.uk/api/content" + parent_path: _spreadsheet_parent(
            parent_path, asset_url, declared,
        ),
        asset_url: malformed,
    }
    with open_native_runtime(**args) as runtime:
        disposition = NativeSourceIntake(
            sources=runtime.authority.sources,
            objects=runtime.authority.objects,
            proof=runtime.proof,
            definition_ids={"UK-01": _seed_uk01(runtime)},
            licence=_licence(),
            dispatch_fence=lambda *_: nullcontext(),
            fetch=lambda url: (200, bodies[url]),
            clock=lambda: datetime(2026, 9, 26, 12, tzinfo=UTC),
        ).poll()[0]
    assert disposition.status == "HOLD"
    assert disposition.units == ()
    assert disposition.item_holds == ((
        asset_url, "SOURCE_SPREADSHEET_RETAIN_FAILED",
    ),)
    assert any(item[:2] == (asset_url, digest_bytes(malformed))
               for item in disposition.observations)


def test_feed_parent_keeps_failed_and_excluded_children_visible(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    args.update(principal_id=OPERATOR_PRINCIPAL_ID, authority_domain=OPERATOR_AUTHORITY_DOMAIN)
    parent_path = "/government/collections/visa-guidance"
    missing_path = "/government/publications/missing-guidance"
    excluded_path = "/government/publications/excluded-guidance"
    parent = _parent_with_children(
        "document_collection", parent_path,
        ((missing_path, "Missing"), (excluded_path, "Excluded")),
    )
    excluded = json.loads(_document(path=excluded_path))
    excluded["details"]["copyright_notice"] = "This is not covered by the Open Government Licence."
    bodies = {
        SOURCE_URLS["UK-01"]: (200, _atom_for(parent_path)),
        "https://www.gov.uk/api/content" + parent_path: (200, parent),
        "https://www.gov.uk/api/content" + missing_path: (503, b""),
        "https://www.gov.uk/api/content" + excluded_path: (
            200, json.dumps(excluded, separators=(",", ":")).encode(),
        ),
    }
    with open_native_runtime(**args) as runtime:
        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=runtime.proof, definition_ids={"UK-01": _seed_uk01(runtime)},
            licence=_licence(), dispatch_fence=lambda *_: nullcontext(),
            fetch=lambda url: bodies[url],
            clock=lambda: datetime(2026, 9, 8, 12, tzinfo=UTC),
        )
        disposition = intake.poll()[0]
    assert disposition.status == "HOLD"
    assert disposition.units == ()
    assert disposition.item_holds == (
        ("https://www.gov.uk" + missing_path, "SOURCE_ITEM_FETCH_INCOMPLETE"),
        ("https://www.gov.uk" + excluded_path, "SOURCE_ITEM_RIGHTS_EXCLUSION_HOLD"),
    )
    assert len(disposition.observations) == 3


def test_feed_parent_does_not_recurse_beyond_its_direct_inventory(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    args.update(principal_id=OPERATOR_PRINCIPAL_ID, authority_domain=OPERATOR_AUTHORITY_DOMAIN)
    parent_path = "/government/collections/visa-guidance"
    child_path = "/government/collections/nested-guidance"
    grandchild_path = "/government/publications/nested-guidance"
    bodies = {
        SOURCE_URLS["UK-01"]: _atom_for(parent_path),
        "https://www.gov.uk/api/content" + parent_path: _parent_with_children(
            "document_collection", parent_path, ((child_path, "Direct child"),),
        ),
        "https://www.gov.uk/api/content" + child_path: _parent_with_children(
            "document_collection", child_path, ((grandchild_path, "Grandchild"),),
        ),
    }
    fetched = []
    with open_native_runtime(**args) as runtime:
        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=runtime.proof, definition_ids={"UK-01": _seed_uk01(runtime)},
            licence=_licence(), dispatch_fence=lambda *_: nullcontext(),
            fetch=lambda url: (fetched.append(url), (200, bodies[url]))[1],
            clock=lambda: datetime(2026, 9, 8, 12, tzinfo=UTC),
        )
        disposition = intake.poll()[0]
    assert disposition.units == ()
    assert disposition.item_holds == ((
        "https://www.gov.uk" + child_path,
        "SOURCE_ITEM_CHILD_COVERAGE_INCOMPLETE",
    ),)
    assert "https://www.gov.uk/api/content" + grandchild_path not in fetched


def test_feed_child_replay_and_parent_lineage_fail_closed(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    args.update(principal_id=OPERATOR_PRINCIPAL_ID, authority_domain=OPERATOR_AUTHORITY_DOMAIN)
    parent_path = "/government/collections/visa-guidance"
    child_path = "/government/publications/visa-guidance"
    parent = _parent_with_children(
        "document_collection", parent_path, ((child_path, "Declared child"),),
    )
    bodies = {
        SOURCE_URLS["UK-01"]: _atom_for(parent_path),
        "https://www.gov.uk/api/content" + parent_path: parent,
        "https://www.gov.uk/api/content" + child_path: _document(path=child_path),
    }
    retained = {}
    with open_native_runtime(**args) as runtime:
        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=runtime.proof, definition_ids={"UK-01": _seed_uk01(runtime)},
            licence=_licence(), dispatch_fence=lambda *_: nullcontext(),
            fetch=lambda url: (200, bodies[url]), retained_units=retained,
            clock=lambda: datetime(2026, 9, 8, 12, tzinfo=UTC),
        )
        first = intake.poll()[0]
        unit = first.units[0]
        observations = {item[1]: item for item in first.observations}
        root_digest = unit.item_key.split("|", 1)[0]
        feed_digest = next(
            digest for digest, value in observations.items()
            if value[0] == SOURCE_URLS["UK-01"]
        )

        for target_digest, replacement_digest, reason in (
            (unit.observation_digest, root_digest, "NATIVE_SOURCE_RAW_OBSERVATION_HOLD"),
            (root_digest, feed_digest, "NATIVE_SOURCE_AUTHORITY_HOLD"),
            (feed_digest, unit.observation_digest, "NATIVE_SOURCE_AUTHORITY_HOLD"),
        ):
            wrong_access = dict(observations)
            wrong_access[target_digest] = (
                *wrong_access[target_digest][:3], observations[replacement_digest][3],
            )
            with pytest.raises(NativeEvidenceHold, match=reason):
                native_evidence_sources(
                    units=first.units, sources=runtime.authority.sources,
                    objects=runtime.authority.objects, observations=wrong_access,
                    licence=_licence(), proof=runtime.proof,
                )

        changed_feed = _atom_for(parent_path).replace(
            b"The official deadline changed.", b"A later feed summary.",
        )
        bodies[SOURCE_URLS["UK-01"]] = changed_feed
        changed = intake.poll()[0]
        assert changed.status == "READY"
        feed_history = dict(observations)
        feed_history.update({item[1]: item for item in changed.observations})
        assert native_evidence_sources(
            units=first.units, sources=runtime.authority.sources,
            objects=runtime.authority.objects, observations=feed_history,
            licence=_licence(), proof=runtime.proof,
        )

        with pytest.raises(NativeEvidenceHold, match="NATIVE_SOURCE_AUTHORITY_HOLD"):
            native_evidence_sources(
                units=(replace(unit, item_key=root_digest + "|/government/unexpected"),),
                sources=runtime.authority.sources, objects=runtime.authority.objects,
                observations=observations, licence=_licence(), proof=runtime.proof,
            )
        wrong_origin = dict(observations)
        wrong_origin[root_digest] = (
            "https://example.test/api/content" + parent_path,
            *wrong_origin[root_digest][1:],
        )
        with pytest.raises(NativeEvidenceHold, match="NATIVE_SOURCE_AUTHORITY_HOLD"):
            native_evidence_sources(
                units=first.units, sources=runtime.authority.sources,
                objects=runtime.authority.objects, observations=wrong_origin,
                licence=_licence(), proof=runtime.proof,
            )
        unrelated_parent_path = "/government/collections/unrelated-guidance"
        unrelated_parent = _parent_with_children(
            "document_collection", unrelated_parent_path,
            ((child_path, "Declared child"),),
        )
        unrelated_admission, unrelated_access = intake._admit_observation(
            "UK-01", unrelated_parent,
        )
        unrelated_digest = digest_bytes(unrelated_parent)
        wrong_parent = dict(observations)
        wrong_parent[unrelated_digest] = (
            "https://www.gov.uk/api/content" + unrelated_parent_path,
            unrelated_digest, str(unrelated_admission.admission_id),
            str(unrelated_access.access_decision_id),
        )
        with pytest.raises(NativeEvidenceHold, match="NATIVE_SOURCE_AUTHORITY_HOLD"):
            native_evidence_sources(
                units=(replace(
                    unit, item_key=unrelated_digest + "|" + child_path,
                ),),
                sources=runtime.authority.sources, objects=runtime.authority.objects,
                observations=wrong_parent, licence=_licence(), proof=runtime.proof,
            )

        retained[unit.revision_id] = tuple(
            replace(item, authority=replace(
                item.authority, definition_version_id="different-version",
            )) for item in first.units
        )
        replay = intake.poll()[0]
        assert replay.status == "HOLD"
        assert replay.item_holds == ((
            "https://www.gov.uk" + child_path,
            "SOURCE_RETAINED_REVISION_BINDING_HOLD",
        ),)


def test_native_manual_keeps_successful_sections_when_one_child_holds(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    args["principal_id"] = OPERATOR_PRINCIPAL_ID
    args["authority_domain"] = OPERATOR_AUTHORITY_DOMAIN
    with open_native_runtime(**args) as runtime:
        definition_id = _seed_missing(runtime, "UK-03")
        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=runtime.proof, definition_ids={"UK-03": definition_id},
            licence=_licence(), dispatch_fence=lambda *_: nullcontext(),
            fetch=lambda url: (
                (200, _manual()) if url == SOURCE_URLS["UK-03"]
                else (200, _manual_section(1)) if url.endswith("part-1")
                else (503, b"")
            ),
            clock=lambda: datetime(2026, 9, 8, 12, tzinfo=UTC),
        )
        disposition = intake.poll()[SOURCE_IDS.index("UK-03")]
        assert disposition.status == "HOLD"
        assert disposition.item_holds == ((
            "https://www.gov.uk/guidance/immigration-rules/part-2",
            "SOURCE_ITEM_FETCH_INCOMPLETE",
        ),)
        assert disposition.units


@pytest.mark.parametrize("stop_at", [1, 2])
def test_native_source_stop_propagates_at_feed_and_item_boundaries(tmp_path, monkeypatch, stop_at):
    from newsroom.control_plane.veto import VetoError
    args = _args(tmp_path, monkeypatch)
    args["principal_id"] = OPERATOR_PRINCIPAL_ID
    args["authority_domain"] = OPERATOR_AUTHORITY_DOMAIN
    fences, fetches, held = [], [], []
    @contextmanager
    def fence(source_id, url):
        fences.append(url)
        if len(fences) == stop_at:
            raise VetoError("owner stop")
        held.append(url)
        try:
            yield
        finally:
            held.pop()
    def fetch(url):
        assert held == [url]
        fetches.append(url)
        return (200, ATOM if url == SOURCE_URLS["UK-01"] else _document())
    with open_native_runtime(**args) as runtime:
        definition_id = _seed_uk01(runtime)
        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=runtime.proof, definition_ids={"UK-01": definition_id},
            licence=_licence(), dispatch_fence=fence, fetch=fetch,
            clock=lambda: datetime(2026, 9, 8, 12, tzinfo=UTC),
        )
        with pytest.raises(VetoError, match="owner stop"):
            intake.poll()
    assert len(fences) == stop_at
    assert len(fetches) == stop_at - 1
    assert held == []


def test_manual_network_fetches_are_bounded_parallel_and_writes_stay_serial(tmp_path, monkeypatch):
    import threading
    args = _args(tmp_path, monkeypatch)
    args.update(principal_id=OPERATOR_PRINCIPAL_ID, authority_domain=OPERATOR_AUTHORITY_DOMAIN)
    barrier = threading.Barrier(2)
    workers = set()
    writes = []
    main_thread = threading.get_ident()
    fenced = []
    @contextmanager
    def fence(*args):
        assert threading.get_ident() == main_thread
        fenced.append(args)
        try:
            yield
        finally:
            fenced.pop()
    def fetch(url):
        if url == SOURCE_URLS['UK-03']:
            return 200, _manual()
        assert len(fenced) == 2
        workers.add(threading.get_ident())
        barrier.wait(timeout=2)
        return 200, _manual_section(int(url[-1]))
    with open_native_runtime(**args) as runtime:
        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=runtime.proof, definition_ids={'UK-03': _seed_missing(runtime, 'UK-03')},
            licence=_licence(), dispatch_fence=fence, fetch=fetch,
            clock=lambda: datetime(2026, 9, 8, 12, tzinfo=UTC),
        )
        original = intake._retain_item
        def retain(*a, **kw):
            assert not fenced
            writes.append(threading.get_ident())
            return original(*a, **kw)
        monkeypatch.setattr(intake, '_retain_item', retain)
        disposition = intake.poll()[SOURCE_IDS.index('UK-03')]
        assert disposition.status == 'READY', disposition.item_holds
        assert len(workers) == 2 and main_thread not in workers
        assert writes == [main_thread, main_thread]
        assert [unit.canonical_url for unit in disposition.units] == [
            'https://www.gov.uk/guidance/immigration-rules/part-1',
            'https://www.gov.uk/guidance/immigration-rules/part-2',
        ]


@pytest.mark.parametrize('failure', ('submit', 'wait'))
def test_manual_fetch_failure_settles_workers_before_releasing_owner_fence(monkeypatch, failure):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from newsroom.control_plane import native_source_intake as module

    started, release = threading.Event(), threading.Event()
    fenced, shutdown_depths, completion_depths = [], [], []

    @contextmanager
    def fence(*args):
        fenced.append(args)
        try:
            yield
        finally:
            fenced.pop()

    def fetch(_url):
        started.set()
        assert release.wait(timeout=5)
        completion_depths.append(len(fenced))
        return 200, b'{}'

    class InterruptedPool(ThreadPoolExecutor):
        submitted = 0

        def submit(self, fn, *args, **kwargs):
            self.submitted += 1
            if failure == 'submit' and self.submitted == 2:
                raise RuntimeError('submit interrupted')
            future = super().submit(fn, *args, **kwargs)
            assert started.wait(timeout=5)
            return future

        def shutdown(self, *args, **kwargs):
            shutdown_depths.append(len(fenced))
            release.set()
            return super().shutdown(*args, **kwargs)

    monkeypatch.setattr(module, 'ThreadPoolExecutor', InterruptedPool)
    if failure == 'wait':
        def interrupt(_futures):
            raise KeyboardInterrupt('wait interrupted')
        monkeypatch.setattr(module, 'wait', interrupt)
    intake = NativeSourceIntake(
        sources=None, objects=None, proof=None, definition_ids={}, licence=None,
        dispatch_fence=fence, fetch=fetch,
    )
    with pytest.raises(RuntimeError if failure == 'submit' else KeyboardInterrupt):
        list(intake._fetch_manual_sections('UK-03', 'sha256:' + '0' * 64, (
            ('/guidance/immigration-rules/part-1', 'Part 1'),
            ('/guidance/immigration-rules/part-2', 'Part 2'),
        )))
    assert shutdown_depths[0] == 2
    assert completion_depths and set(completion_depths) == {2}
    assert not fenced


def test_collection_body_rights_notice_vetoes_child_file_fetches(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    args.update(principal_id=OPERATOR_PRINCIPAL_ID, authority_domain=OPERATOR_AUTHORITY_DOMAIN)
    collection = '/government/collections/data'
    child = '/government/publications/data'
    value = json.loads(_parent_with_children('document_collection', collection, ((child, 'Data files'),)))
    value['details']['body'] = '<p>All rights <em>reserved</em> for the linked files.</p>'
    bodies = {
        SOURCE_URLS['UK-01']: _atom_for(collection),
        'https://www.gov.uk/api/content' + collection: json.dumps(value).encode(),
    }
    fetched = []
    with open_native_runtime(**args) as runtime:
        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=runtime.proof, definition_ids={'UK-01': _seed_uk01(runtime)},
            licence=_licence(), dispatch_fence=lambda *_: nullcontext(),
            fetch=lambda url: (fetched.append(url), (200, bodies[url]))[1],
            clock=lambda: datetime(2026, 9, 26, 12, tzinfo=UTC),
        )
        result = intake.poll()[0]
    assert result.units == ()
    assert result.item_holds == (('https://www.gov.uk' + collection, 'SOURCE_ITEM_RIGHTS_EXCLUSION_HOLD'),)
    assert fetched == list(bodies)


@contextmanager
def _changing_manual_child(tmp_path, monkeypatch, source_id="UK-01"):
    args = _args(tmp_path, monkeypatch)
    args.update(principal_id=OPERATOR_PRINCIPAL_ID, authority_domain=OPERATOR_AUTHORITY_DOMAIN)
    parent_path = "/guidance/immigration-rules" if source_id == "UK-03" else "/government/collections/visa-guidance"
    child_path = parent_path + "/part-1" if source_id == "UK-03" else "/government/publications/factsheets"
    parent_url, child_url = ("https://www.gov.uk/api/content" + path for path in (parent_path, child_path))
    parent = _parent_with_children(
        "manual" if source_id == "UK-03" else "document_collection",
        parent_path, ((child_path, "Declared child"),),
    )
    bodies = {SOURCE_URLS[source_id]: _atom_for(parent_path), parent_url: parent,
              child_url: _document(path=child_path)}
    retained, fetched = {}, []
    instant = [datetime(2026, 9, 8, 12, tzinfo=UTC)]
    with open_native_runtime(**args) as runtime:
        definition = _seed_missing(runtime, source_id) if source_id == "UK-03" else _seed_uk01(runtime)
        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=runtime.proof, definition_ids={source_id: definition},
            licence=_licence(), dispatch_fence=lambda *_: nullcontext(),
            fetch=lambda url: (fetched.append(url), (200, bodies[url]))[1],
            retained_units=retained, clock=lambda: instant[0],
        )
        first = intake.poll()[SOURCE_IDS.index(source_id)]
        assert first.status == "READY" and len(first.units) == 1
        retained[first.units[0].revision_id] = first.units
        instant[0] = datetime(2026, 9, 8, 14, tzinfo=UTC)
        changed = json.loads(parent)
        changed["title"] += " — revised parent inventory"
        bodies[parent_url] = json.dumps(changed).encode()
        yield runtime, intake, bodies, child_url, retained, fetched, first


@pytest.mark.parametrize("source_id", ["UK-01", "UK-03"])
def test_manual_child_parent_changes_reuse_retained_identity(tmp_path, monkeypatch, source_id):
    with _changing_manual_child(tmp_path, monkeypatch, source_id) as case:
        runtime, intake, _, _, _, _, first = case
        replay = intake.poll()[SOURCE_IDS.index(source_id)]
        assert replay.status == "READY"
        assert replay.units == first.units
        assert replay.units[0] is first.units[0]
        assert replay.units[0].ingest_id == first.units[0].ingest_id
        observations = {item[1]: item for result in (first, replay) for item in result.observations}
        assert len(observations) > len(first.observations)
        assert native_evidence_sources(
            units=replay.units, sources=runtime.authority.sources,
            objects=runtime.authority.objects, observations=observations,
            licence=_licence(), proof=runtime.proof,
        )
        original_root = first.units[0].item_key.split("|", 1)[0]
        observations.pop(original_root)
        with pytest.raises(NativeEvidenceHold, match="NATIVE_SOURCE_AUTHORITY_HOLD"):
            native_evidence_sources(
                units=replay.units, sources=runtime.authority.sources,
                objects=runtime.authority.objects, observations=observations,
                licence=_licence(), proof=runtime.proof,
            )


@pytest.mark.parametrize("change", ["updated", "conflict", "removed"])
def test_manual_child_updates_keep_namespace_and_current_coverage(tmp_path, monkeypatch, change):
    with _changing_manual_child(tmp_path, monkeypatch) as case:
        runtime, intake, bodies, child_url, _, fetched, first = case
        old = first.units[0]
        if change == "removed":
            parent_url = next(url for url in bodies if "/api/content/government/collections/" in url)
            value = json.loads(bodies[parent_url])
            value["links"]["documents"] = []
            bodies[parent_url] = json.dumps(value).encode()
        else:
            bodies[child_url] = _document(
                path=old.canonical_url.removeprefix("https://www.gov.uk"),
                body="The child now contains a substantively different rule.",
                updated="2026-09-08T13:00:00Z" if change == "updated" else "2026-09-08T11:00:00Z",
            )
        result = intake.poll()[0]
        if change == "updated":
            assert result.status == "READY" and len(result.units) == 1
            new = result.units[0]
            assert new.item_key == old.item_key
            assert new.authority.item_id == old.authority.item_id
            assert new.revision_id != old.revision_id and new.ingest_id != old.ingest_id
            revision = runtime.authority.sources.revision(SourceRevisionId.parse(new.revision_id), proof=runtime.proof)
            assert str(revision.request.prior_revision_id) == old.revision_id
        elif change == "conflict":
            assert result.units == ()
            assert result.item_holds == ((old.canonical_url, "SOURCE_NATIVE_REVISION_CONFLICT"),)
        else:
            assert not any(unit.canonical_url == old.canonical_url for unit in result.units)
            assert fetched.count(child_url) == 1


@pytest.mark.parametrize("mismatch", [None, "source", "version", "path", "url", "invalid-root", "file", "tagged"])
def test_manual_item_key_reuses_only_exact_retained_namespace(mismatch):
    from newsroom.control_plane.items import SourceItem

    path = "/government/publications/factsheets"
    old_key, current_key = "sha256:" + "1" * 64 + "|" + path, "sha256:" + "2" * 64 + "|" + path
    old = SimpleNamespace(source_id="UK-01", item_key=old_key, canonical_url="https://www.gov.uk" + path,
                          authority=SimpleNamespace(definition_version_id="version-1"))
    if mismatch == "source":
        old.source_id = "UK-03"
    elif mismatch == "version":
        old.authority.definition_version_id = "version-2"
    elif mismatch == "path":
        old.item_key = old_key + "-other"
    elif mismatch == "url":
        old.canonical_url += "-other"
    elif mismatch == "invalid-root":
        old.item_key = "sha256:invalid|" + path
    elif mismatch == "file":
        current_key = "sha256:" + "2" * 64 + "|https://assets.publishing.service.gov.uk/file.xlsx"
    elif mismatch == "tagged":
        current_key = "govuk-child-v1|" + path
    later = SimpleNamespace(**{**vars(old), "item_key": "sha256:" + "3" * 64 + "|" + path})
    intake = NativeSourceIntake(sources=None, objects=None, proof=None, definition_ids={}, licence=None,
                               dispatch_fence=lambda *_: nullcontext(),
                               retained_units={"first-land": (old,), "later-land": (later,)} if mismatch is None else {"first-land": (old,)})
    item = SourceItem("UK-01", current_key, "Factsheets", "Retained body", "https://www.gov.uk" + path)
    assert intake._manual_item_key("UK-01", "version-1", item) == (old_key if mismatch is None else current_key)


def test_feed_collection_retains_declared_publication_html_leaves_with_ancestry(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    args.update(principal_id=OPERATOR_PRINCIPAL_ID, authority_domain=OPERATOR_AUTHORITY_DOMAIN)
    collection = '/government/collections/dfe-update'
    publication = '/government/publications/dfe-update-current'
    leaves = tuple((publication + '/' + kind, kind.title()) for kind in ('education', 'academies', 'authorities'))
    bodies = {
        SOURCE_URLS['UK-05']: _atom_for(collection),
        'https://www.gov.uk/api/content' + collection: _parent_with_children('document_collection', collection, ((publication, 'DfE publication'),)),
        'https://www.gov.uk/api/content' + publication: _parent_with_children('correspondence', publication, leaves),
        **{'https://www.gov.uk/api/content' + path: _document(path=path) for path, title in leaves},
    }
    fetched = []
    with open_native_runtime(**args) as runtime:
        intake = NativeSourceIntake(
            sources=runtime.authority.sources, objects=runtime.authority.objects, proof=runtime.proof,
            definition_ids={'UK-05': _seed_missing(runtime, 'UK-05')}, licence=_licence(),
            dispatch_fence=lambda *_: nullcontext(),
            fetch=lambda url: (fetched.append(url), (200, bodies[url]))[1],
            clock=lambda: datetime(2026, 9, 8, 12, tzinfo=UTC),
        )
        result = next(value for value in intake.poll() if value.source_id == 'UK-05')
        assert result.status == 'READY' and not result.item_holds
        assert {unit.canonical_url for unit in result.units} == {'https://www.gov.uk' + path for path, _ in leaves}
        assert len(result.observations) == 6
        observations = {value[1]: value for value in result.observations}
        sources = native_evidence_sources(
            units=result.units, sources=runtime.authority.sources, objects=runtime.authority.objects,
            observations=observations, licence=_licence(), proof=runtime.proof,
        )
        assert len(sources) == 3
        # The publication itself is not a feed item; its collection proof is necessary.
        missing_collection = {key: value for key, value in observations.items()
                              if value[0] != 'https://www.gov.uk/api/content' + collection}
        with pytest.raises(NativeEvidenceHold, match='NATIVE_SOURCE_AUTHORITY_HOLD'):
            native_evidence_sources(
                units=result.units, sources=runtime.authority.sources, objects=runtime.authority.objects,
                observations=missing_collection, licence=_licence(), proof=runtime.proof,
            )
        before = len(fetched)
        replay = next(value for value in intake.poll() if value.source_id == 'UK-05')
        assert [unit.ingest_id for unit in replay.units] == [unit.ingest_id for unit in result.units]
        assert len(fetched) > before  # Fresh current observation is still acquired.
