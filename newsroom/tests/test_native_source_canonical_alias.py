"""Provider-free regressions for exact GOV.UK child revision aliases."""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import UTC, datetime
import json
from types import SimpleNamespace

import pytest

from newsroom.authority.canonical import digest_bytes
from newsroom.control_plane.graphiti_operational_readiness import (
    OPERATOR_AUTHORITY_DOMAIN, OPERATOR_PRINCIPAL_ID,
)
from newsroom.control_plane.native_evidence import NativeEvidenceHold
from newsroom.control_plane.native_progress import NativeRevisionJournal
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.native_source_intake import (
    NativeSourceIntake, native_evidence_sources,
)
from newsroom.control_plane.store import connect
from newsroom.control_plane.veto import VetoError
from newsroom.increment9.proving import SOURCE_IDS, SOURCE_URLS
from newsroom.sources import SourceDefinitionVersionId, SourceRevisionId
from newsroom.tests.test_native_source_intake import (
    _args, _atom_for, _document, _licence, _parent_with_children,
    _seed_missing, _seed_uk01, _spreadsheet_parent, _xlsx_asset,
)

CHILD = "/government/publications/factsheets"
PARENTS = (
    "/government/collections/immigration-guidance",
    "/government/collections/visa-guidance",
)
CANONICAL = "https://www.gov.uk" + CHILD
CHILD_API = "https://www.gov.uk/api/content" + CHILD
TAG_KEY = "tag:www.gov.uk,2005:" + CHILD


def _feed(*paths, child_key=TAG_KEY):
    entries = []
    for path in paths:
        key = child_key if path == CHILD else "tag:www.gov.uk,2005:" + path
        raw = _atom_for(path).replace(b"<id>item-1</id>", f"<id>{key}</id>".encode())
        entries.append(raw.split(b"<entry>", 1)[1].split(b"</entry>", 1)[0])
    return (b'<feed xmlns="http://www.w3.org/2005/Atom">' + b"".join(
        b"<entry>" + entry + b"</entry>" for entry in entries
    ) + b"</feed>")


@contextmanager
def _case(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    args.update(principal_id=OPERATOR_PRINCIPAL_ID, authority_domain=OPERATOR_AUTHORITY_DOMAIN)
    bodies = {CHILD_API: _document(path=CHILD)}
    for path in PARENTS:
        bodies["https://www.gov.uk/api/content" + path] = _parent_with_children(
            "document_collection", path, ((CHILD, "Declared child"),),
        )
    fetched = []
    instant = [datetime(2026, 9, 8, 12, tzinfo=UTC)]
    connection = connect(str(tmp_path / "control.sqlite3"))
    journal = NativeRevisionJournal(connection)
    try:
        with open_native_runtime(**args) as runtime:
            definition_id = _seed_uk01(runtime)

            def fetch(url):
                fetched.append(url)
                return 200, bodies[url]

            intake = NativeSourceIntake(
                sources=runtime.authority.sources, objects=runtime.authority.objects,
                proof=runtime.proof, definition_ids={"UK-01": definition_id},
                licence=_licence(), dispatch_fence=lambda *_: nullcontext(),
                fetch=fetch, retained_units=journal.units, clock=lambda: instant[0],
            )
            yield SimpleNamespace(
                runtime=runtime, intake=intake, bodies=bodies, fetched=fetched,
                instant=instant, journal=journal, connection=connection,
                definition_id=definition_id,
            )
    finally:
        connection.close()


def _evidence(case, disposition):
    return native_evidence_sources(
        units=disposition.units, sources=case.runtime.authority.sources,
        objects=case.runtime.authority.objects,
        observations={**case.journal.observations, **{
            observation[1]: observation for observation in disposition.observations
        }}, licence=_licence(), proof=case.runtime.proof,
    )


@pytest.mark.parametrize("paths", (
    (*PARENTS, CHILD), (CHILD, *reversed(PARENTS)),
))
def test_same_poll_parent_and_feed_aliases_retain_one_complete_revision(
    tmp_path, monkeypatch, paths,
):
    with _case(tmp_path, monkeypatch) as case:
        case.bodies[SOURCE_URLS["UK-01"]] = _feed(*paths)
        result = case.intake.poll()[0]
        assert result.status == "READY", result
        assert len(result.units) == 1
        unit = result.units[0]
        first_key = (
            TAG_KEY if paths[0] == CHILD
            else digest_bytes(case.bodies["https://www.gov.uk/api/content" + paths[0]]) + "|" + CHILD
        )
        assert unit.item_key == first_key
        assert unit.canonical_url == CANONICAL
        assert len({item.authority.item_id for item in result.units}) == 1
        assert len({item.authority.revision_id for item in result.units}) == 1
        assert len({item.ingest_id for item in result.units}) == 1
        assert len(result.observations) == 6
        assert {observation[0] for observation in result.observations} == {
            SOURCE_URLS["UK-01"], CHILD_API,
            *("https://www.gov.uk/api/content" + path for path in PARENTS),
        }
        assert case.fetched.count(CHILD_API) == 3
        assert len(_evidence(case, result)) == 1
        if paths[0] != CHILD:
            observations = {item[1]: item for item in result.observations}
            observations.pop(first_key.split("|", 1)[0])
            with pytest.raises(NativeEvidenceHold, match="NATIVE_SOURCE_AUTHORITY_HOLD"):
                native_evidence_sources(
                    units=result.units, sources=case.runtime.authority.sources,
                    objects=case.runtime.authority.objects, observations=observations,
                    licence=_licence(), proof=case.runtime.proof,
                )
        case.journal.sources((result,))
        case.journal.land(result.units)
        assert len(case.journal.units) == 1


@pytest.mark.parametrize("first_path", (PARENTS[0], CHILD))
@pytest.mark.parametrize("changed", (False, True))
def test_historical_namespace_survives_later_parent_and_feed_aliases(
    tmp_path, monkeypatch, first_path, changed,
):
    with _case(tmp_path, monkeypatch) as case:
        case.bodies[SOURCE_URLS["UK-01"]] = _feed(first_path)
        first = case.intake.poll()[0]
        assert first.status == "READY" and len(first.units) == 1
        old = first.units[0]
        case.journal.sources((first,))
        case.journal.land(first.units)
        original_ledger = case.connection.execute("SELECT * FROM ledger ORDER BY seq").fetchall()
        case.bodies[SOURCE_URLS["UK-01"]] = _feed(PARENTS[1], CHILD, PARENTS[0])
        case.instant[0] = datetime(2026, 9, 8, 14, tzinfo=UTC)
        if changed:
            case.bodies[CHILD_API] = _document(
                path=CHILD, body="The maintained child now contains a changed rule.",
                updated="2026-09-08T13:00:00Z",
            )
        result = case.intake.poll()[0]
        assert result.status == "READY", result
        assert len(result.units) == 1
        new = result.units[0]
        assert new.item_key == old.item_key
        assert new.authority.item_id == old.authority.item_id
        if changed:
            assert new.revision_id != old.revision_id
            assert new.ingest_id != old.ingest_id
            revision = case.runtime.authority.sources.revision(
                SourceRevisionId.parse(new.authority.revision_id), proof=case.runtime.proof,
            )
            assert str(revision.request.prior_revision_id) == old.authority.revision_id
        else:
            assert new is case.journal.units[old.revision_id][0]
            assert new == old
        assert _evidence(case, result)
        assert case.connection.execute("SELECT * FROM ledger ORDER BY seq").fetchall() == original_ledger
        assert case.journal.units[old.revision_id] == first.units


@pytest.mark.parametrize("boundary", ("source", "definition-version"))
def test_canonical_alias_does_not_cross_retained_authority_boundaries(
    tmp_path, monkeypatch, boundary,
):
    with _case(tmp_path, monkeypatch) as case:
        case.bodies[SOURCE_URLS["UK-01"]] = _feed(PARENTS[0])
        first = case.intake.poll()[0]
        old = first.units[0]
        case.journal.sources((first,))
        case.journal.land(first.units)
        source_id = "UK-01"
        if boundary == "source":
            source_id = "UK-05"
            case.intake.bind_definitions({source_id: _seed_missing(case.runtime, source_id)})
        else:
            summary = case.runtime.authority.sources.current_summary(
                case.definition_id, proof=case.runtime.proof,
            )
            version = case.runtime.authority.sources.version_details(
                summary.version_id, proof=case.runtime.proof,
            ).request
            case.runtime.authority.sources.record_definition_version(replace(
                version,
                version_id=SourceDefinitionVersionId.parse("00000000-0000-4000-8000-000000000099"),
                version_number=2, expected_previous_version_id=summary.version_id,
                extraction_scope=tuple(sorted((*version.extraction_scope, "fixture_annotation"))),
                change_reason="Provider-free canonical alias boundary fixture.",
                idempotency_key="canonical-alias-definition-version-2",
            ), proof=case.runtime.proof)
        case.bodies[SOURCE_URLS[source_id]] = _feed(PARENTS[1])
        result = case.intake.poll()[SOURCE_IDS.index(source_id)]
        assert result.status == "READY" and len(result.units) == 1, result
        new = result.units[0]
        assert new.canonical_url == old.canonical_url
        assert new.revision_digest == old.revision_digest
        assert new.item_key == digest_bytes(
            case.bodies["https://www.gov.uk/api/content" + PARENTS[1]]
        ) + "|" + CHILD
        assert new.item_key != old.item_key
        assert new.authority.item_id != old.authority.item_id
        assert new.authority.revision_id != old.authority.revision_id
        assert _evidence(case, result)


@pytest.mark.parametrize("key", (
    "item-1", "govuk-child-v1|" + CHILD,
    "sha256:invalid|" + CHILD,
    "tag:www.gov.uk,2005:/government/publications/different",
))
def test_unrecognised_feed_keys_keep_their_own_namespace(tmp_path, monkeypatch, key):
    with _case(tmp_path, monkeypatch) as case:
        case.bodies[SOURCE_URLS["UK-01"]] = _feed(PARENTS[0], CHILD, child_key=key)
        result = case.intake.poll()[0]
        assert result.status == "READY", result
        assert len(result.units) == 2
        assert key in {unit.item_key for unit in result.units}
        assert len({unit.authority.item_id for unit in result.units}) == 2
        assert len({unit.authority.revision_id for unit in result.units}) == 2
        assert len({unit.ingest_id for unit in result.units}) == 2


def test_two_declared_assets_at_one_citation_url_are_not_html_aliases(tmp_path, monkeypatch):
    with _case(tmp_path, monkeypatch) as case:
        path = "/government/publications/funding-values"
        urls = (
            "https://assets.publishing.service.gov.uk/media/asset/funding-one.xlsx",
            "https://assets.publishing.service.gov.uk/media/asset/funding-two.xlsx",
        )
        asset = _xlsx_asset()
        parent = json.loads(_spreadsheet_parent(path, urls[0], asset))
        attachment = parent["details"]["attachments"][0]
        parent["details"]["attachments"].append({
            **attachment, "url": urls[1], "filename": "funding-two.xlsx", "id": "asset-two",
        })
        case.bodies[SOURCE_URLS["UK-01"]] = _feed(path)
        case.bodies["https://www.gov.uk/api/content" + path] = json.dumps(parent).encode()
        case.bodies.update({url: asset for url in urls})
        case.instant[0] = datetime(2026, 9, 26, 12, tzinfo=UTC)
        result = case.intake.poll()[0]
        assert result.status == "READY", result
        assert len(result.units) == 2
        assert len({unit.canonical_url for unit in result.units}) == 1
        assert {unit.item_key.split("|", 1)[1] for unit in result.units} == set(urls)
        assert len({unit.authority.item_id for unit in result.units}) == 2
        assert len({unit.authority.revision_id for unit in result.units}) == 2
        assert len({unit.ingest_id for unit in result.units}) == 2


@pytest.mark.parametrize(("change", "reason"), (
    ("excluded", "SOURCE_ITEM_RIGHTS_EXCLUSION_HOLD"),
    ("removed", "SOURCE_ITEM_METADATA_HOLD"),
))
def test_same_poll_alias_does_not_skip_current_parent_checks(
    tmp_path, monkeypatch, change, reason,
):
    with _case(tmp_path, monkeypatch) as case:
        case.bodies[SOURCE_URLS["UK-01"]] = _feed(*PARENTS)
        second_api = "https://www.gov.uk/api/content" + PARENTS[1]
        parent = json.loads(case.bodies[second_api])
        if change == "excluded":
            parent["details"]["body"] = "<p>Third-party copyright: all rights reserved.</p>"
        else:
            parent["links"]["documents"] = []
        case.bodies[second_api] = json.dumps(parent).encode()
        result = case.intake.poll()[0]
        assert result.status == "HOLD"
        assert len(result.units) == 1
        assert result.item_holds == (("https://www.gov.uk" + PARENTS[1], reason),)
        assert len(result.observations) == 4
        assert case.fetched.count(CHILD_API) == 1
        assert _evidence(case, result)


def test_same_poll_alias_keeps_native_revision_conflict_fail_closed(tmp_path, monkeypatch):
    with _case(tmp_path, monkeypatch) as case:
        case.bodies[SOURCE_URLS["UK-01"]] = _feed(*PARENTS)
        fetch = case.intake._fetch
        child_reads = 0

        def changing_child(url):
            nonlocal child_reads
            status, raw = fetch(url)
            if url == CHILD_API:
                child_reads += 1
                if child_reads == 2:
                    raw = _document(path=CHILD, body="Conflicting bytes at the same native revision token.")
            return status, raw

        case.intake._fetch = changing_child
        result = case.intake.poll()[0]
        assert result.status == "HOLD"
        assert len(result.units) == 1
        assert result.item_holds == ((CANONICAL, "SOURCE_NATIVE_REVISION_CONFLICT"),)
        # The current conflict is admitted but never selected as evidence;
        # the completed revision and both observed parent inventories remain.
        assert len(result.observations) == 4
        assert _evidence(case, result)


def test_current_rights_and_second_parent_veto_still_precede_alias_reuse(tmp_path, monkeypatch):
    with _case(tmp_path, monkeypatch) as case:
        case.bodies[SOURCE_URLS["UK-01"]] = _feed(*PARENTS, CHILD)
        first = case.intake.poll()[0]
        case.journal.sources((first,))
        case.journal.land(first.units)
        licence = _licence()
        case.intake._licence = replace(licence, policy_digest="sha256:" + "f" * 64)
        fetched = tuple(case.fetched)
        held = case.intake.poll()[0]
        assert held.reason_code == "CURRENT_RIGHTS_HOLD" and held.units == ()
        assert tuple(case.fetched) == fetched
        case.intake._licence = licence

        @contextmanager
        def fence(_source_id, url):
            if url == "https://www.gov.uk/api/content" + PARENTS[1]:
                raise VetoError("fixture owner stop before current second parent")
            yield

        case.intake._fence = fence
        with pytest.raises(VetoError, match="fixture owner stop"):
            case.intake.poll()
        assert len(case.journal.units) == 1


def test_first_historical_land_is_future_identity_without_rewriting_three_old_aliases(
    tmp_path, monkeypatch,
):
    with _case(tmp_path, monkeypatch) as case:
        original_units = []
        # Independently retained single-alias polls reproduce the three old
        # authorities without mocking or mutating Source Registry records.
        for path in (PARENTS[1], CHILD, PARENTS[0]):
            case.bodies[SOURCE_URLS["UK-01"]] = _feed(path)
            intake = NativeSourceIntake(
                sources=case.runtime.authority.sources, objects=case.runtime.authority.objects,
                proof=case.runtime.proof, definition_ids={"UK-01": case.definition_id},
                licence=_licence(), dispatch_fence=lambda *_: nullcontext(),
                fetch=case.intake._fetch, clock=lambda: case.instant[0],
            )
            result = intake.poll()[0]
            assert result.status == "READY" and len(result.units) == 1
            original_units.append(result.units[0])
            case.journal.sources((result,))
            case.journal.land(result.units)
        assert len(case.journal.units) == 3
        assert len({unit.authority.item_id for unit in original_units}) == 3
        assert len({unit.authority.revision_id for unit in original_units}) == 3
        assert len({unit.observation_digest for unit in original_units}) == 1
        assert len({unit.revision_digest for unit in original_units}) == 1
        ledger = case.connection.execute("SELECT * FROM ledger ORDER BY seq").fetchall()
        revisions = tuple(case.runtime.authority.sources.revision(
            SourceRevisionId.parse(unit.authority.revision_id), proof=case.runtime.proof,
        ) for unit in original_units)
        case.bodies[SOURCE_URLS["UK-01"]] = _feed(CHILD, *PARENTS)
        result = case.intake.poll()[0]
        assert result.status == "READY" and len(result.units) == 1
        assert result.units[0] is case.journal.units[original_units[0].revision_id][0]
        assert _evidence(case, result)
        assert len(case.journal.units) == 3
        assert case.connection.execute("SELECT * FROM ledger ORDER BY seq").fetchall() == ledger
        assert tuple(case.runtime.authority.sources.revision(
            SourceRevisionId.parse(unit.authority.revision_id), proof=case.runtime.proof,
        ) for unit in original_units) == revisions
