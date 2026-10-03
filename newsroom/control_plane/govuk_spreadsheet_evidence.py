"""Independent acquisition of one declared GOV.UK spreadsheet asset."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from datetime import UTC, datetime

from newsroom.authority import AuthenticationProof, GovernedObjects
from newsroom.authority.canonical import digest_bytes, digest_canonical, validate_sha256_digest
from newsroom.control_plane.corpus import CorpusIngestUnit
from newsroom.increment9.proving import MAX_BODY_BYTES

from .govuk_evidence import _utc
from .govuk_rights import ATTRIBUTION
from .govuk_spreadsheet import (
    POLICY_DIGEST as PARSER_POLICY_DIGEST,
    declared_spreadsheet,
    parse_govuk_spreadsheet,
)
from .native_evidence import (
    AcquiredEvidence,
    EvidenceAcquisitionRequest,
    NativeEvidenceHold,
    rights_eligibility_digest,
)
from .native_source_intake import (
    _canonical_url_from_api,
    _fetch_exact,
    native_evidence_sources,
    spreadsheet_asset_url,
)
from .veto import VetoError

VERSION = "hermes-govuk-spreadsheet-evidence-v1"
POLICY_DIGEST = digest_canonical(
    {
        "version": VERSION,
        "parser_policy": PARSER_POLICY_DIGEST,
        "method": "GET",
        "redirects": 0,
        "max_bytes": MAX_BODY_BYTES,
        "credentials": False,
        "parent_and_asset_required": True,
    }
)


class GovUkSpreadsheetEvidenceAcquisition:
    """Reacquire an exact parent declaration and its spreadsheet asset."""

    asset_url_for = staticmethod(spreadsheet_asset_url)
    declare = staticmethod(declared_spreadsheet)
    parse = staticmethod(parse_govuk_spreadsheet)
    asset_byte_limit = MAX_BODY_BYTES
    parser_policy_digest = PARSER_POLICY_DIGEST
    version = VERSION
    reason_prefix = "GOVUK_SPREADSHEET"
    require_raw_identity = False

    def __init__(
        self,
        *,
        sources,
        objects: GovernedObjects,
        proof: AuthenticationProof,
        licence,
        transport_policy_digest: str,
        dispatch_fence: Callable[
            [EvidenceAcquisitionRequest], AbstractContextManager[None]
        ],
        retained_units: Mapping[str, tuple[CorpusIngestUnit, ...]],
        observations: Mapping[str, tuple[str, str, str, str]],
        fetch: Callable[[str], tuple[int, bytes]] = _fetch_exact,
        clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
    ) -> None:
        validate_sha256_digest(transport_policy_digest)
        if (
            not callable(dispatch_fence)
            or not callable(fetch)
            or not callable(clock)
            or not isinstance(retained_units, Mapping)
            or not isinstance(observations, Mapping)
        ):
            raise ValueError("spreadsheet acquisition configuration differs")
        self._sources = sources
        self._objects = objects
        self._proof = proof
        self._licence = licence
        self._transport_policy_digest = transport_policy_digest
        self._fence = dispatch_fence
        self._retained_units = retained_units
        self._observations = observations
        self._fetch = fetch
        self._clock = clock

    def __call__(self, request: EvidenceAcquisitionRequest) -> AcquiredEvidence:
        if type(request) is not EvidenceAcquisitionRequest:
            raise TypeError("exact independent acquisition request required")

        def hold(reason: str) -> NativeEvidenceHold:
            return NativeEvidenceHold(reason.replace("GOVUK_SPREADSHEET", self.reason_prefix), request.source_id)

        if request.transport_policy_digest != self._transport_policy_digest:
            raise hold("TRANSPORT_POLICY_MISMATCH")
        retained = self._retained_units.get(request.source_revision_id)
        if type(retained) is not tuple or not retained:
            raise hold("GOVUK_SPREADSHEET_SOURCE_BINDING_HOLD")
        try:
            evidence_sources = native_evidence_sources(
                units=retained,
                sources=self._sources,
                objects=self._objects,
                observations=self._observations,
                licence=self._licence,
                proof=self._proof,
            )
            if len(evidence_sources) != 1:
                raise ValueError("one exact spreadsheet source is required")
            source = evidence_sources[0]
            unit = source.unit
            authority = unit.authority
            root_digest = unit.item_key.partition("|")[0]
            asset_url = self.asset_url_for(unit)
            parent_observation = self._observations[root_digest]
            if (
                asset_url is None
                or unit.canonical_url != request.canonical_url
                or unit.source_id != request.source_id
                or unit.revision_id != request.source_revision_id
                or authority is None
                or authority.definition_id != request.source_definition_id
                or authority.definition_version_id
                != request.source_definition_version_id
                or source.source_version.canonical_digest
                != request.source_definition_version_digest
                or parent_observation[1] != root_digest
                or _canonical_url_from_api(parent_observation[0])
                != unit.canonical_url
            ):
                raise ValueError("spreadsheet source binding differs")
            parent_url = parent_observation[0]
        except (KeyError, LookupError, PermissionError, TypeError, ValueError):
            raise hold("GOVUK_SPREADSHEET_SOURCE_BINDING_HOLD") from None

        try:
            with self._fence(request):
                parent_status, parent_raw = self._fetch(parent_url)
                if (
                    parent_status != 200
                    or not parent_raw
                    or len(parent_raw) > MAX_BODY_BYTES
                ):
                    raise hold("GOVUK_SPREADSHEET_ACQUISITION_INCOMPLETE")
                self.declare(
                    _canonical_url_from_api(parent_url),
                    parent_raw,
                    asset_url,
                    retrieved_at=self._clock(),
                )
                if self.require_raw_identity and digest_bytes(parent_raw) != root_digest:
                    raise hold("GOVUK_SPREADSHEET_PARENT_CHANGED_HOLD")
                asset_status, asset_raw = self._fetch(asset_url)
        except VetoError:
            raise
        except NativeEvidenceHold:
            raise
        except ValueError:
            raise hold("GOVUK_SPREADSHEET_EVIDENCE_METADATA_HOLD") from None
        except Exception:
            raise hold("GOVUK_SPREADSHEET_ACQUISITION_UNAVAILABLE") from None
        retrieved = self._clock()
        if (
            retrieved.tzinfo is None
            or asset_status != 200
            or not asset_raw
            or len(asset_raw) > self.asset_byte_limit
        ):
            raise hold("GOVUK_SPREADSHEET_ACQUISITION_INCOMPLETE")
        retrieved = retrieved.astimezone(UTC)
        if self.require_raw_identity and digest_bytes(asset_raw) != unit.observation_digest:
            raise hold("GOVUK_SPREADSHEET_RAW_CHANGED_HOLD")
        try:
            document = self.parse(
                _canonical_url_from_api(parent_url),
                parent_raw,
                asset_url,
                asset_raw,
                retrieved_at=retrieved,
            )
            if (
                document.title != unit.headline
                or document.body_text != unit.body
                or _utc(document.publication) != unit.published_at
                or _utc(document.updated) != unit.updated_at
            ):
                raise ValueError("spreadsheet current content differs")
        except (KeyError, TypeError, ValueError, UnicodeError):
            raise hold("GOVUK_SPREADSHEET_EVIDENCE_METADATA_HOLD") from None
        body = (document.title + "\n\n" + document.body_text).encode("utf-8")
        body_digest = digest_bytes(body)
        transport_digest = digest_canonical(
            {
                "version": self.version,
                "parser_policy": self.parser_policy_digest,
                "request_digest": request.digest,
                "parent_url": parent_url,
                "parent_response_digest": digest_bytes(parent_raw),
                "asset_url": asset_url,
                "asset_response_digest": digest_bytes(asset_raw),
                "body_digest": body_digest,
                "public_updated_at": _utc(document.updated),
                "retrieved_at": _utc(retrieved),
            }
        )
        rights_digest = rights_eligibility_digest(
            source.rights,
            body_digest=body_digest,
            transport_digest=transport_digest,
            exclusion_signals=document.exclusion_signals,
            text_only=True,
        )
        return AcquiredEvidence.create(
            request_digest=request.digest,
            outcome="COMPLETE",
            canonical_url=unit.canonical_url,
            body=body,
            body_digest=body_digest,
            publisher="; ".join(document.organisations),
            responsible_body="; ".join(document.organisations),
            source_type="PRIMARY_OFFICIAL",
            publication_time=_utc(document.publication),
            source_updated_time=_utc(document.updated),
            retrieval_time=_utc(retrieved),
            geography="UK",
            language="en-GB",
            transport_evidence_digest=transport_digest,
            currentness_basis="AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT",
            rights_eligibility_digest=rights_digest,
            licence_attribution=ATTRIBUTION,
            exclusion_signals=document.exclusion_signals,
            text_only=True,
        )


__all__ = [
    "GovUkSpreadsheetEvidenceAcquisition",
    "POLICY_DIGEST",
]
