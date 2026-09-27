from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import count

import pytest

from newsroom.control_plane.govuk_spreadsheet_evidence import (
    GovUkSpreadsheetEvidenceAcquisition,
    POLICY_DIGEST,
)
from newsroom.control_plane.graphiti_operational_readiness import (
    OPERATOR_AUTHORITY_DOMAIN,
    OPERATOR_PRINCIPAL_ID,
)
from newsroom.control_plane.native_evidence import (
    EvidenceAcquisitionRequest,
    EvidenceTransport,
    NativeEvidenceController,
    NativeEvidenceHold,
)
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.native_source_intake import (
    NativeSourceIntake,
    native_evidence_sources,
)
from newsroom.increment9.proving import SOURCE_URLS
from newsroom.sources import SourceDefinitionVersionId
from newsroom.tests.test_native_runtime import _args
from newsroom.tests.test_native_source_intake import (
    _atom_for,
    _parent_with_children,
    _licence,
    _seed_uk01,
    _spreadsheet_parent,
    _xlsx_asset,
)


def _retained_spreadsheet(tmp_path, monkeypatch, suffix="xlsx", *, nested=False, duplicate=False):
    args = _args(tmp_path, monkeypatch)
    args.update(
        principal_id=OPERATOR_PRINCIPAL_ID,
        authority_domain=OPERATOR_AUTHORITY_DOMAIN,
    )
    parent_path = "/government/publications/funding-values"
    parent_url = "https://www.gov.uk/api/content" + parent_path
    asset_url = (
        "https://assets.publishing.service.gov.uk/media/asset/"
        f"funding-values.{suffix}"
    )
    asset = _xlsx_asset() if suffix == "xlsx" else b"Provider,Funding\nExample College,125000\n"
    parent = _spreadsheet_parent(parent_path, asset_url, asset)
    bodies = {
        SOURCE_URLS["UK-01"]: _atom_for(parent_path),
        parent_url: parent,
        asset_url: asset,
    }
    if nested:
        collection_path = "/government/collections/funding"
        feed = _atom_for(collection_path)
        if duplicate:
            import xml.etree.ElementTree as ET
            first, second = ET.fromstring(feed), ET.fromstring(_atom_for(parent_path))
            first.extend(list(second)[-1:])
            feed = ET.tostring(first)
        bodies[SOURCE_URLS["UK-01"]] = feed
        bodies["https://www.gov.uk/api/content" + collection_path] = _parent_with_children(
            "document_collection", collection_path, ((parent_path, "Funding files"),),
        )
    runtime_context = open_native_runtime(**args)
    runtime = runtime_context.__enter__()
    ticks = count()
    intake = NativeSourceIntake(
        sources=runtime.authority.sources,
        objects=runtime.authority.objects,
        proof=runtime.proof,
        definition_ids={"UK-01": _seed_uk01(runtime)},
        licence=_licence(),
        dispatch_fence=lambda *_: nullcontext(),
        fetch=lambda url: (200, bodies[url]),
        clock=lambda: datetime(2026, 9, 26, 12, tzinfo=UTC) + timedelta(seconds=next(ticks)),
    )
    disposition = intake.poll()[0]
    assert disposition.status == "READY" and disposition.units
    unit = disposition.units[0]
    version = runtime.authority.sources.version_details(
        SourceDefinitionVersionId.parse(unit.authority.definition_version_id),
        proof=runtime.proof,
    )
    request = EvidenceAcquisitionRequest(
        source_id=unit.source_id,
        source_definition_id=unit.authority.definition_id,
        source_definition_version_id=unit.authority.definition_version_id,
        source_definition_version_digest=version.canonical_digest,
        source_revision_id=unit.revision_id,
        canonical_url=unit.canonical_url,
        transport_policy_digest=POLICY_DIGEST,
    )
    source = native_evidence_sources(
        units=disposition.units,
        sources=runtime.authority.sources,
        objects=runtime.authority.objects,
        observations={item[1]: item for item in disposition.observations},
        licence=_licence(),
        proof=runtime.proof,
    )[0]
    return (
        runtime_context, runtime, disposition, source, request, bodies,
        parent_url, asset_url,
    )


@pytest.mark.parametrize("nested,duplicate", ((False, False), (True, False), (True, True)))
@pytest.mark.parametrize("suffix", ("xlsx", "csv"))
def test_spreadsheet_acquisition_refetches_parent_and_asset_with_exact_binding(
    tmp_path, monkeypatch, suffix, nested, duplicate,
) -> None:
    context, runtime, disposition, source, request, bodies, parent_url, asset_url = (
        _retained_spreadsheet(tmp_path, monkeypatch, suffix, nested=nested, duplicate=duplicate)
    )
    fetched = []
    fences = []
    try:
        assert len(disposition.units) == 1
        # Duplicate feed/collection paths are one canonical revision, not two
        # chunk ordinals that would make journal LAND fail.
        from newsroom.control_plane.native_progress import NativeRevisionJournal
        from newsroom.control_plane.store import connect
        connection = connect(str(tmp_path / "journal.sqlite3"))
        journal = NativeRevisionJournal(connection)
        journal.land(disposition.units)
        journal.land(disposition.units)
        connection.close()
        acquisition = GovUkSpreadsheetEvidenceAcquisition(
            sources=runtime.authority.sources,
            objects=runtime.authority.objects,
            proof=runtime.proof,
            licence=_licence(),
            transport_policy_digest=POLICY_DIGEST,
            dispatch_fence=lambda value: nullcontext(fences.append(value.digest)),
            retained_units={request.source_revision_id: disposition.units},
            observations={item[1]: item for item in disposition.observations},
            fetch=lambda url: (fetched.append(url), (200, bodies[url]))[1],
            clock=lambda: datetime(2026, 9, 26, 13, tzinfo=UTC),
        )
        controller = object.__new__(NativeEvidenceController)
        controller._transport_policy_digest = POLICY_DIGEST
        controller._transport = EvidenceTransport(acquisition)
        assert controller._preflight(source) == request
        result = controller._acquire(source, request)
        assert result.outcome == "COMPLETE"
        assert result.canonical_url == source.unit.canonical_url
        assert result.canonical_url.startswith("https://www.gov.uk/")
        assert result.text_only is True
        assert result.rights_eligibility_digest
        assert result.licence_attribution
        assert b'Row 1: A="Provider"' in result.body
        assert fetched == [parent_url, asset_url]
        assert fences == [request.digest]
        assert acquisition(request) == result
    finally:
        context.__exit__(None, None, None)


@pytest.mark.parametrize("tamper", ["retained-parent", "fresh-parent", "fresh-asset"])
def test_spreadsheet_acquisition_tampering_fails_closed(
    tmp_path, monkeypatch, tamper,
) -> None:
    context, runtime, disposition, _source, request, bodies, parent_url, asset_url = (
        _retained_spreadsheet(tmp_path, monkeypatch)
    )
    observations = {item[1]: item for item in disposition.observations}
    if tamper == "retained-parent":
        root = disposition.units[0].item_key.split("|", 1)[0]
        observations[root] = (
            "https://www.gov.uk/api/content/government/publications/unrelated",
            *observations[root][1:],
        )
    current = dict(bodies)
    if tamper == "fresh-parent":
        current[parent_url] = current[parent_url].replace(asset_url.encode(), b"https://assets.publishing.service.gov.uk/media/asset/other.xlsx")
    if tamper == "fresh-asset":
        current[asset_url] = b"x" * len(current[asset_url])
    fetched = []
    try:
        acquisition = GovUkSpreadsheetEvidenceAcquisition(
            sources=runtime.authority.sources,
            objects=runtime.authority.objects,
            proof=runtime.proof,
            licence=_licence(),
            transport_policy_digest=POLICY_DIGEST,
            dispatch_fence=lambda _value: nullcontext(),
            retained_units={request.source_revision_id: disposition.units},
            observations=observations,
            fetch=lambda url: (fetched.append(url), (200, current[url]))[1],
            clock=lambda: datetime(2026, 9, 26, 13, tzinfo=UTC),
        )
        with pytest.raises(
            NativeEvidenceHold,
            match=(
                "GOVUK_SPREADSHEET_SOURCE_BINDING_HOLD"
                if tamper == "retained-parent"
                else "GOVUK_SPREADSHEET_EVIDENCE_METADATA_HOLD"
            ),
        ):
            acquisition(request)
        assert fetched == (
            []
            if tamper == "retained-parent"
            else [parent_url]
            if tamper == "fresh-parent"
            else [parent_url, asset_url]
        )
    finally:
        context.__exit__(None, None, None)


def test_controller_preflight_rejects_foreign_and_acquisition_rejects_undeclared_asset(
    tmp_path, monkeypatch,
) -> None:
    (
        context, runtime, disposition, source, _request, bodies,
        _parent_url, _asset_url,
    ) = _retained_spreadsheet(tmp_path, monkeypatch)
    controller = object.__new__(NativeEvidenceController)
    controller._transport_policy_digest = POLICY_DIGEST
    try:
        root = source.unit.item_key.split("|", 1)[0]
        foreign_url = "https://example.com/foreign.xlsx"
        foreign_unit = replace(
            source.unit,
            item_key=root + "|" + foreign_url,
            canonical_url=foreign_url,
        )
        foreign_unit = replace(
            foreign_unit,
            effective_revision=replace(
                foreign_unit.effective_revision,
                item_key=foreign_unit.item_key,
                revision_digest=foreign_unit.revision_digest,
            ),
        )
        with pytest.raises(NativeEvidenceHold, match="SOURCE_BINDING_HOLD"):
            controller._preflight(replace(source, unit=foreign_unit))

        undeclared_url = (
            "https://assets.publishing.service.gov.uk/media/asset/"
            "undeclared.xlsx"
        )
        undeclared_unit = replace(
            source.unit,
            item_key=root + "|" + undeclared_url,
        )
        undeclared_unit = replace(
            undeclared_unit,
            effective_revision=replace(
                undeclared_unit.effective_revision,
                item_key=undeclared_unit.item_key,
                revision_digest=undeclared_unit.revision_digest,
            ),
        )
        undeclared_source = replace(source, unit=undeclared_unit)
        undeclared_request = controller._preflight(undeclared_source)
        fetched = []
        acquisition = GovUkSpreadsheetEvidenceAcquisition(
            sources=runtime.authority.sources,
            objects=runtime.authority.objects,
            proof=runtime.proof,
            licence=_licence(),
            transport_policy_digest=POLICY_DIGEST,
            dispatch_fence=lambda _value: nullcontext(),
            retained_units={
                undeclared_request.source_revision_id: tuple(
                    replace(
                        unit,
                        item_key=root + "|" + undeclared_url,
                    )
                    for unit in disposition.units
                )
            },
            observations={
                item[1]: item for item in disposition.observations
            },
            fetch=lambda url: (fetched.append(url), (200, bodies[url]))[1],
            clock=lambda: datetime(2026, 9, 26, 13, tzinfo=UTC),
        )
        controller._transport = EvidenceTransport(acquisition)
        with pytest.raises(
            NativeEvidenceHold,
            match="GOVUK_SPREADSHEET_SOURCE_BINDING_HOLD",
        ):
            controller._acquire(undeclared_source, undeclared_request)
        assert fetched == []
    finally:
        context.__exit__(None, None, None)


def test_collection_attachment_requires_every_retained_ancestry_link(tmp_path, monkeypatch):
    import json
    from newsroom.authority.canonical import digest_bytes
    context, runtime, disposition, _source, request, bodies, parent_url, _asset_url = (
        _retained_spreadsheet(tmp_path, monkeypatch, 'csv', nested=True)
    )
    observations = {item[1]: item for item in disposition.observations}
    collection_url = 'https://www.gov.uk/api/content/government/collections/funding'
    collection_digest = next(key for key, value in observations.items() if value[0] == collection_url)
    intake = NativeSourceIntake(
        sources=runtime.authority.sources, objects=runtime.authority.objects,
        proof=runtime.proof, definition_ids={}, licence=_licence(),
        dispatch_fence=lambda *_: nullcontext(),
    )
    try:
        invalid = []
        for missing in (SOURCE_URLS['UK-01'], collection_url, parent_url):
            invalid.append({key: value for key, value in observations.items() if value[0] != missing})
        for kind in ('wrong-url', 'wrong-access', 'wrong-digest', 'undeclared-child', 'rights-exclusion'):
            changed = dict(observations)
            original = changed.pop(collection_digest)
            if kind == 'wrong-url':
                replacement = ('https://www.gov.uk/api/content/government/collections/unrelated', *original[1:])
            elif kind == 'wrong-access':
                replacement = (*original[:3], observations[disposition.units[0].observation_digest][3])
            elif kind == 'wrong-digest':
                replacement = (original[0], 'sha256:' + 'f' * 64, *original[2:])
            else:
                value = json.loads(bodies[collection_url])
                if kind == 'undeclared-child':
                    value['links']['documents'][0]['base_path'] = '/government/publications/unrelated'
                else:
                    value['details']['copyright_notice'] = 'All rights reserved'
                raw = json.dumps(value).encode()
                admission, access = intake._admit_observation('UK-01', raw)
                replacement = (collection_url, digest_bytes(raw), str(admission.admission_id), str(access.access_decision_id))
            changed[replacement[1]] = replacement
            invalid.append(changed)
        for changed in invalid:
            with pytest.raises(NativeEvidenceHold, match='NATIVE_SOURCE_AUTHORITY_HOLD'):
                native_evidence_sources(
                    units=disposition.units, sources=runtime.authority.sources,
                    objects=runtime.authority.objects, observations=changed,
                    licence=_licence(), proof=runtime.proof,
                )
    finally:
        context.__exit__(None, None, None)
