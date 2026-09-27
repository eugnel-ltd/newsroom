from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime

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
    _licence,
    _seed_uk01,
    _spreadsheet_parent,
    _xlsx_asset,
)


def _retained_spreadsheet(tmp_path, monkeypatch, suffix="xlsx"):
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
    runtime_context = open_native_runtime(**args)
    runtime = runtime_context.__enter__()
    intake = NativeSourceIntake(
        sources=runtime.authority.sources,
        objects=runtime.authority.objects,
        proof=runtime.proof,
        definition_ids={"UK-01": _seed_uk01(runtime)},
        licence=_licence(),
        dispatch_fence=lambda *_: nullcontext(),
        fetch=lambda url: (200, bodies[url]),
        clock=lambda: datetime(2026, 9, 26, 12, tzinfo=UTC),
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


@pytest.mark.parametrize("suffix", ("xlsx", "csv"))
def test_spreadsheet_acquisition_refetches_parent_and_asset_with_exact_binding(
    tmp_path, monkeypatch, suffix,
) -> None:
    context, runtime, disposition, source, request, bodies, parent_url, asset_url = (
        _retained_spreadsheet(tmp_path, monkeypatch, suffix)
    )
    fetched = []
    fences = []
    try:
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
