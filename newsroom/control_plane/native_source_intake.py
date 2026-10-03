"""Native polling of current approved sources into governed corpus revisions."""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, ExitStack
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from itertools import chain
from datetime import UTC, datetime
from urllib.parse import urlsplit

from newsroom.authority import (
    AuthenticationProof, HydrationRequest, ObjectAccessDecisionId, ObjectAdmissionId,
    ObjectAdmissionRequest, UtcTimestamp, AuthorityPersistenceError,
)
from newsroom.authority.canonical import (
    digest_bytes, digest_canonical, validate_sha256_digest,
)
from newsroom.checks import deterministic_uuid4
from newsroom.control_plane.corpus import CorpusAuthorityBinding, CorpusIngestUnit, chunk_text
from newsroom.control_plane.graphiti_operational_readiness import (
    GRAPHITI_EVALUATION_HYDRATION_POLICY,
    OPERATIONAL_ADMISSION_TYPE,
)
from newsroom.control_plane.items import SourceItem, parse_observation
from newsroom.effective_revision import EffectiveRevisionIdentity
from newsroom.graphiti_adapter.identity import content_digest, representation_digest_for
from newsroom.increment9.proving import (
    MAX_BODY_BYTES,
    SOURCE_IDS,
    SOURCE_URLS,
    TIMEOUT_SECONDS,
    USER_AGENT,
)
from newsroom.sources import (
    DiscoveryRepresentationId, DiscoveryRepresentationRequest, IdentityComponent,
    SourceDefinitionId, SourceDefinitionVersionId, SourceItemId, SourceItemIdentityKind,
    SourceDependencyKind, SourceItemRequest, SourceRevisionId, SourceRevisionRequest,
    SourceRole, SourceTime,
)

from .govuk_rights import GovUkLicenceEvidence
from .govuk_evidence import (
    GovUkContentHold, _api_url, _utc, _require_attachment_inventory, parse_govuk_content_document,
    parse_govuk_manual_inventory,
)
from .govuk_spreadsheet import (
    declared_spreadsheet,
    is_spreadsheet_url,
    parse_govuk_spreadsheet,
)
from .govuk_pdf import (declared_pdf, is_pdf_url, parse_govuk_pdf, GovUkPdfHold,
    pdf_parse_binding, pdf_parse_receipt, read_pdf_parse_receipt,
    MAX_RAW_BYTES as MAX_PDF_RAW_BYTES)
from .native_policies import (
    NATIVE_SOURCE_OBSERVATION_ADMISSION_TYPE,
    NATIVE_SOURCE_OBSERVATION_PURPOSE,
    NATIVE_PDF_OBSERVATION_ADMISSION_TYPE, NATIVE_PDF_OBSERVATION_PURPOSE,
)
from .native_evidence import (
    DependencyAssessment, NativeEvidenceHold, NativeEvidenceSource,
)
from .native_progress import NativeRevisionJournal, iter_source_headers

from .veto import OperatorDrainRequested, VetoError

VERSION = "hermes-native-source-intake-v1"
SUPPORTED = frozenset({"UK-01", "UK-02", "UK-03", "UK-05"})


class NativeSourceIntakeHold(GovUkContentHold):
    """Keep typed GOV.UK coverage and historical-reference metadata at intake."""

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _canonical_url_from_api(url: str) -> str:
    prefix = "https://www.gov.uk/api/content"
    if not url.startswith(prefix + "/"):
        raise ValueError("GOV.UK Content API endpoint differs")
    return "https://www.gov.uk" + url.removeprefix(prefix)


def _fetch_exact(url: str) -> tuple[int, bytes]:
    """Fetch only the already-approved exact HTTPS endpoint, without redirects."""

    if (
        url not in SOURCE_URLS.values()
        and not url.startswith("https://www.gov.uk/api/content/")
        and not is_spreadsheet_url(url)
        and not is_pdf_url(url)
    ):
        raise ValueError("native source endpoint is not approved")
    request = urllib.request.Request(
        url, method="GET", headers={"User-Agent": USER_AGENT, "Accept": "*/*"}
    )
    body_limit = MAX_PDF_RAW_BYTES if is_pdf_url(url) else MAX_BODY_BYTES
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        _NoRedirect(),
    )
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            status = int(getattr(response, "status", 200))
            response_url = response.geturl()
            body = response.read(body_limit + 1)
    except urllib.error.HTTPError as exc:
        body = exc.read(body_limit + 1) if exc.fp else b""
        status = int(exc.code)
        response_url = exc.geturl()
    except (urllib.error.URLError, TimeoutError, ssl.SSLError, OSError) as exc:
        raise ValueError("native source transport failed") from exc
    if len(body) > body_limit:
        raise ValueError("native source response exceeds body bound")
    if response_url != url:
        raise ValueError("native source response identity differs")
    return status, body


@dataclass(frozen=True, slots=True)
class NativeSourceDisposition:
    source_id: str
    status: str
    reason_code: str
    units: tuple[CorpusIngestUnit, ...] = ()
    observation_admission_id: str | None = None
    observation_access_decision_id: str | None = None
    observations: tuple[tuple[str, str, str, str], ...] = ()
    item_holds: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class VerifiedNativeObservation:
    url: str
    digest: str
    admission_id: ObjectAdmissionId
    access_decision_id: str
    raw: bytes


def spreadsheet_asset_url(unit: CorpusIngestUnit) -> str | None:
    """Return the exact declared asset tail without treating it as citation URL."""

    root_digest, separator, asset_url = unit.item_key.partition("|")
    try:
        validate_sha256_digest(root_digest)
    except (TypeError, ValueError):
        return None
    return (
        asset_url
        if separator == "|" and is_spreadsheet_url(asset_url)
        else None
    )


def pdf_asset_url(unit: CorpusIngestUnit) -> str | None:
    root_digest, separator, asset_url = unit.item_key.partition("|")
    try:
        validate_sha256_digest(root_digest)
    except (TypeError, ValueError):
        return None
    return asset_url if separator == "|" and is_pdf_url(asset_url) else None


class NativeSourceIntake:
    """Poll eligible retained definitions; every portfolio member remains visible."""

    def __init__(
        self, *, sources, objects, proof: AuthenticationProof,
        definition_ids: Mapping[str, SourceDefinitionId],
        licence: GovUkLicenceEvidence,
        dispatch_fence: Callable[[str, str], AbstractContextManager[None]],
        other_source_poll: Callable[..., NativeSourceDisposition] | None = None,
        retained_units: Mapping[str, tuple[CorpusIngestUnit, ...]] | None = None,
        fetch: Callable[[str], tuple[int, bytes]] = _fetch_exact,
        clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
    ) -> None:
        if set(definition_ids) - set(SOURCE_IDS):
            raise ValueError("native source bindings exceed the approved portfolio")
        if not callable(dispatch_fence):
            raise ValueError("native observation authority and dispatch fence are required")
        self._sources, self._objects, self._proof = sources, objects, proof
        self._definitions, self._licence = {}, licence
        self.bind_definitions(definition_ids)
        self._fence, self._fetch, self._clock = dispatch_fence, fetch, clock
        self._other_source_poll = other_source_poll
        self._retained_units = {} if retained_units is None else retained_units
        self._pending_units = {}

    def bind_definitions(
        self, definition_ids: Mapping[str, SourceDefinitionId],
    ) -> None:
        """Add newly retained portfolio definitions without rebinding existing ones."""
        if set(definition_ids) - set(SOURCE_IDS) or any(
            type(value) is not SourceDefinitionId
            for value in definition_ids.values()
        ):
            raise ValueError("native source bindings exceed the approved portfolio")
        for source_id, definition_id in definition_ids.items():
            retained = self._definitions.get(source_id)
            if retained is not None and retained != definition_id:
                raise ValueError("native source binding differs")
        self._definitions.update(definition_ids)

    def poll(self) -> tuple[NativeSourceDisposition, ...]:
        self._pending_units.clear()
        results = []
        for source_id in SOURCE_IDS:
            try:
                results.append(self._poll_one(source_id))
            except VetoError:
                raise
            except Exception as exc:
                results.append(NativeSourceDisposition(
                    source_id, "HOLD", getattr(exc, "reason_code", "SOURCE_POLL_FAILED")
                ))
        return tuple(results)

    def _poll_one(self, source_id: str) -> NativeSourceDisposition:
        definition_id = self._definitions.get(source_id)
        if definition_id is None:
            from .native_source_rights import NativePortfolioRights
            if type(self._licence) is NativePortfolioRights and self._licence.for_source(
                source_id=source_id, definition_url=SOURCE_URLS[source_id],
            ).decision != "PERMITTED":
                return NativeSourceDisposition(
                    source_id, "HOLD", self._licence.reason_for(source_id),
                    observations=self._licence.observations_for(source_id),
                )
            return NativeSourceDisposition(source_id, "HOLD", "SOURCE_DEFINITION_MISSING")
        summary = self._sources.current_summary(definition_id, proof=self._proof)
        version = self._sources.version_details(summary.version_id, proof=self._proof).request
        expected_url = SOURCE_URLS[source_id]
        if version.locator != expected_url:
            return NativeSourceDisposition(source_id, "HOLD", "SOURCE_LOCATOR_MISMATCH")
        if source_id not in SUPPORTED:
            if self._other_source_poll is not None:
                return self._other_source_poll(
                    source_id=source_id, definition_id=definition_id,
                    version_id=summary.version_id, version=version,
                )
            return NativeSourceDisposition(source_id, "HOLD", "CURRENT_RIGHTS_UNSUPPORTED")
        rights = self._licence.for_source(source_id=source_id, definition_url=expected_url)
        if rights.decision != "PERMITTED":
            return NativeSourceDisposition(source_id, "HOLD", "CURRENT_RIGHTS_HOLD")
        if source_id in {"UK-02", "UK-03"}:
            return self._poll_direct_govuk(
                source_id, definition_id, summary.version_id, version, rights,
            )
        with self._fence(source_id, expected_url):
            status, raw = self._fetch(expected_url)
        if len(raw) > MAX_BODY_BYTES:
            return NativeSourceDisposition(source_id, "HOLD", "SOURCE_BODY_TOO_LARGE")
        if status != 200 or not raw:
            return NativeSourceDisposition(source_id, "HOLD", "SOURCE_FETCH_INCOMPLETE")
        raw_admission, raw_access = self._admit_observation(source_id, raw)
        observations = [(
            expected_url, digest_bytes(raw), str(raw_admission.admission_id),
            str(raw_access.access_decision_id),
        )]
        items = parse_observation(source_id=source_id, url=expected_url, body=raw)
        if not items:
            return NativeSourceDisposition(
                source_id, "HOLD", "SOURCE_PARSE_EMPTY", (),
                str(raw_admission.admission_id), str(raw_access.access_decision_id),
                tuple(observations),
            )
        units, item_holds = [], []
        for item in items:
            try:
                item_url, item_raw, retrieved = self._fetch_complete_item(source_id, item)
                settled_units, settled_observations, settled_holds = self._settle_item(
                    source_id, definition_id, summary.version_id, version, item,
                    item_url, item_raw, retrieved, rights.record_id,
                )
                units.extend(settled_units)
                observations.extend(settled_observations)
                item_holds.extend(settled_holds)
            except VetoError:
                raise
            except Exception as exc:
                item_holds.append((
                    item.canonical_url,
                    getattr(exc, "reason_code", "SOURCE_ITEM_RETAIN_FAILED"),
                ))
        unique_units = {}
        for unit in units:
            key = (unit.authority.revision_id, unit.chunk_ordinal)
            prior = unique_units.get(key)
            if prior is None:
                unique_units[key] = unit
            elif replace(unit, observed_at=prior.observed_at) != prior:
                raise NativeSourceIntakeHold("SOURCE_REOBSERVATION_CONFLICT")
        status = "HOLD" if item_holds else "READY"
        return NativeSourceDisposition(
            source_id, status,
            "SOURCE_ITEMS_HELD" if item_holds else "GOVERNED_REVISIONS_RETAINED",
            tuple(unique_units.values()),
            str(raw_admission.admission_id), str(raw_access.access_decision_id),
            tuple(observations),
            tuple(item_holds),
        )

    def _settle_item(
        self, source_id, definition_id, version_id, version, item,
        item_url, raw, observed, rights_id, *, follow_children=True,
        publication_leaves=False, terminal_html_leaf=False,
    ):
        admission, access = self._admit_observation(source_id, raw)
        observation_digest = digest_bytes(raw)
        observations = [(
            item_url, observation_digest, str(admission.admission_id),
            str(access.access_decision_id),
        )]
        try:
            complete = self._parse_complete_item(item, raw, observed)
        except NativeSourceIntakeHold as exc:
            if exc.exclusion_signals:
                return (), tuple(observations), ((
                    item.canonical_url, "SOURCE_ITEM_RIGHTS_EXCLUSION_HOLD",
                ),)
            if terminal_html_leaf or (not exc.child_items and not exc.unsupported_attachments):
                return (), tuple(observations), ((item.canonical_url, exc.reason_code), *(
                    (url, "SOURCE_ITEM_HISTORICAL_REFERENCE_COVERAGE_UNASSESSED") for url, _ in exc.archival_references
                ))
            units, holds = [], [
                (url, "SOURCE_ITEM_HISTORICAL_REFERENCE_COVERAGE_UNASSESSED") for url, _ in exc.archival_references
            ]
            leaf_handoff = publication_leaves and _declared_publication_leaves(
                raw, item.canonical_url,
            )
            follow_inventory = follow_children or leaf_handoff
            collection_children = follow_children and json.loads(raw).get("document_type") == "document_collection"
            if exc.child_items and not follow_inventory:
                holds.append((item.canonical_url, exc.reason_code))
            for child, fetched in self._fetch_manual_sections(
                source_id, observation_digest, exc.child_items if follow_inventory else (),
            ):
                try:
                    child_url, child_raw, child_observed = fetched.result()
                    child_units, child_observations, child_holds = self._settle_item(
                        source_id, definition_id, version_id, version, child,
                        child_url, child_raw, child_observed, rights_id,
                        follow_children=False, publication_leaves=collection_children,
                        terminal_html_leaf=leaf_handoff,
                    )
                    units.extend(child_units)
                    observations.extend(child_observations)
                    holds.extend(child_holds)
                except VetoError:
                    raise
                except Exception as child_exc:
                    holds.append((
                        child.canonical_url,
                        getattr(child_exc, "reason_code", "SOURCE_ITEM_RETAIN_FAILED"),
                    ))
            asset_declarations = []
            unsupported_attachments = []
            for asset_url, asset_title in exc.unsupported_attachments:
                if (asset_url, asset_title) in exc.archival_references:
                    unsupported_attachments.append((asset_url, asset_title))
                    continue
                try:
                    declare = declared_pdf if is_pdf_url(asset_url) else declared_spreadsheet
                    asset_declarations.append(declare(
                        item.canonical_url, raw, asset_url, retrieved_at=observed,
                    ))
                except (TypeError, ValueError, KeyError, UnicodeError):
                    unsupported_attachments.append((asset_url, asset_title))
            for declaration, fetched in self._fetch_declared_assets(
                source_id, asset_declarations,
            ):
                try:
                    asset_url, asset_raw, asset_observed = fetched.result()
                    asset_admission, asset_access = self._admit_observation(
                        source_id, asset_raw, pdf=is_pdf_url(asset_url),
                    )
                    asset_digest = digest_bytes(asset_raw)
                    observations.append((
                        asset_url, asset_digest,
                        str(asset_admission.admission_id),
                        str(asset_access.access_decision_id),
                    ))
                    if is_pdf_url(asset_url):
                        document = self._parse_pdf(
                            item.canonical_url, raw, asset_url, asset_raw,
                            retrieved_at=asset_observed, source_version=str(version_id))
                    else:
                        document = parse_govuk_spreadsheet(
                            item.canonical_url, raw, asset_url, asset_raw,
                            retrieved_at=asset_observed)
                    asset_item = SourceItem(
                        source_id, observation_digest + "|" + asset_url,
                        document.title, document.body_text, item.canonical_url,
                        _utc(document.publication), _utc(document.updated),
                        document.body_text,
                    )
                    units.extend(self._retain_item(
                        source_id, definition_id, version_id, version,
                        asset_item, asset_digest, _utc(asset_observed), rights_id,
                    ))
                except VetoError:
                    raise
                except Exception as asset_exc:
                    holds.append((
                        declaration.asset_url,
                        getattr(
                            asset_exc, "reason_code",
                            "SOURCE_PDF_RETAIN_FAILED" if is_pdf_url(declaration.asset_url) else "SOURCE_SPREADSHEET_RETAIN_FAILED",
                        ),
                    ))
            if unsupported_attachments:
                holds.append((item.canonical_url, exc.reason_code))
            return tuple(units), tuple(observations), tuple(holds)
        if terminal_html_leaf:
            value = json.loads(raw)
            if value.get("links", {}).get("children") or value.get("details", {}).get("attachments"):
                return (), tuple(observations), ((
                    item.canonical_url, "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE",
                ),)
        units = self._retain_item(
            source_id, definition_id, version_id, version, complete,
            observation_digest, _utc(observed), rights_id,
        )
        return units, tuple(observations), ()

    def _poll_direct_govuk(self, source_id, definition_id, version_id, version, rights):
        endpoint = SOURCE_URLS[source_id]
        with self._fence(source_id, endpoint):
            status, raw = self._fetch(endpoint)
        if len(raw) > MAX_BODY_BYTES:
            return NativeSourceDisposition(source_id, "HOLD", "SOURCE_BODY_TOO_LARGE")
        if status != 200 or not raw:
            return NativeSourceDisposition(source_id, "HOLD", "SOURCE_FETCH_INCOMPLETE")
        admission, access = self._admit_observation(source_id, raw)
        root_digest = digest_bytes(raw)
        observations = [(endpoint, root_digest, str(admission.admission_id),
                         str(access.access_decision_id))]
        retrieved = self._clock().astimezone(UTC)
        canonical_root = _canonical_url_from_api(endpoint)
        if source_id == "UK-02":
            try:
                document = parse_govuk_content_document(
                    canonical_root, raw, retrieved_at=retrieved,
                )
                if document.document_type != "guide":
                    raise ValueError("BN(O) source is not a complete guide")
                item = SourceItem(
                    source_id, urlsplit(canonical_root).path, document.title,
                    document.body_text, canonical_root,
                    _utc(document.publication), _utc(document.updated), document.body_text,
                )
                units = self._retain_item(
                    source_id, definition_id, version_id, version, item, root_digest,
                    _utc(retrieved), rights.record_id,
                )
            except (TypeError, ValueError, KeyError, UnicodeError):
                return NativeSourceDisposition(
                    source_id, "HOLD", "MULTIPART_COVERAGE_INCOMPLETE", (),
                    str(admission.admission_id), str(access.access_decision_id),
                    tuple(observations),
                )
            return NativeSourceDisposition(
                source_id, "READY", "GOVERNED_REVISIONS_RETAINED", units,
                str(admission.admission_id), str(access.access_decision_id),
                tuple(observations),
            )
        try:
            inventory = parse_govuk_manual_inventory(
                canonical_root, raw, retrieved_at=retrieved,
            )
        except (TypeError, ValueError, KeyError, UnicodeError):
            return NativeSourceDisposition(
                source_id, "HOLD", "MANUAL_CHILD_COVERAGE_INCOMPLETE", (),
                str(admission.admission_id), str(access.access_decision_id),
                tuple(observations),
            )
        units, item_holds = [], []
        for item, fetched in self._fetch_manual_sections(source_id, root_digest, inventory.sections):
            canonical_url = item.canonical_url
            try:
                item_url, item_raw, observed = fetched.result()
                item_admission, item_access = self._admit_observation(source_id, item_raw)
                observations.append((
                    item_url, digest_bytes(item_raw), str(item_admission.admission_id),
                    str(item_access.access_decision_id),
                ))
                item = self._parse_complete_item(item, item_raw, observed)
                units.extend(self._retain_item(
                    source_id, definition_id, version_id, version, item,
                    digest_bytes(item_raw), _utc(observed), rights.record_id,
                ))
            except VetoError:
                raise
            except Exception as exc:
                item_holds.append((
                    canonical_url,
                    getattr(exc, "reason_code", "SOURCE_ITEM_RETAIN_FAILED"),
                ))
        return NativeSourceDisposition(
            source_id, "HOLD" if item_holds else "READY",
            "SOURCE_ITEMS_HELD" if item_holds else "GOVERNED_REVISIONS_RETAINED",
            tuple(units), str(admission.admission_id), str(access.access_decision_id),
            tuple(observations), tuple(item_holds),
        )

    def _fetch_manual_sections(self, source_id, root_digest, sections):
        # Network only in workers; SQLite admission and deterministic coverage
        # stay on the owner thread. Four responses bound the live working set.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(0, len(sections), 4):
                items = tuple(SourceItem(
                    source_id, root_digest + "|" + path, title, title,
                    "https://www.gov.uk" + path,
                ) for path, title in sections[start:start + 4])
                # Owner-stop fences use a re-entrant transaction on this thread.
                # Hold it once for the batch, not competing writer locks in each
                # worker; no authority writes take place under the network fence.
                with ExitStack() as fences:
                    for item in items:
                        fences.enter_context(self._fence(source_id, _api_url(item.canonical_url)))
                    try:
                        pending = tuple((item, pool.submit(self._fetch_item_response, item)) for item in items)
                        wait(tuple(future for _, future in pending))
                    except BaseException:
                        # Settle even partially submitted work before the owner
                        # fence exits; the outer pool context unwinds too late.
                        pool.shutdown(wait=True, cancel_futures=True)
                        raise
                yield from pending

    def _fetch_complete_item(self, source_id, item):
        try:
            url = _api_url(item.canonical_url)
        except ValueError:
            raise NativeSourceIntakeHold("SOURCE_ITEM_CANONICAL_URL_HOLD") from None
        with self._fence(source_id, url):
            return self._fetch_item_response(item)

    def _fetch_declared_assets(self, source_id, declarations):
        declarations = tuple(declarations)
        if not declarations:
            return
        with ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(0, len(declarations), 4):
                batch = declarations[start:start + 4]
                with ExitStack() as fences:
                    for declaration in batch:
                        fences.enter_context(
                            self._fence(source_id, declaration.asset_url)
                        )
                    try:
                        pending = tuple(
                            (
                                declaration,
                                pool.submit(
                                    self._fetch_asset_response,
                                    declaration.asset_url,
                                ),
                            )
                            for declaration in batch
                        )
                        wait(tuple(future for _item, future in pending))
                    except BaseException:
                        pool.shutdown(wait=True, cancel_futures=True)
                        raise
                yield from pending

    def _fetch_asset_response(self, asset_url):
        pdf = is_pdf_url(asset_url)
        if not pdf and not is_spreadsheet_url(asset_url):
            raise NativeSourceIntakeHold("SOURCE_SPREADSHEET_URL_HOLD")
        status, raw = self._fetch(asset_url)
        if len(raw) > (MAX_PDF_RAW_BYTES if pdf else MAX_BODY_BYTES):
            raise NativeSourceIntakeHold("SOURCE_PDF_RAW_BOUND_HOLD" if pdf else "SOURCE_SPREADSHEET_BODY_TOO_LARGE")
        if status != 200 or not raw:
            raise NativeSourceIntakeHold("SOURCE_PDF_FETCH_INCOMPLETE_HOLD" if pdf else "SOURCE_SPREADSHEET_FETCH_INCOMPLETE")
        return asset_url, raw, self._clock().astimezone(UTC)

    def _fetch_item_response(self, item):
        url = _api_url(item.canonical_url)
        status, raw = self._fetch(url)
        if len(raw) > MAX_BODY_BYTES:
            raise NativeSourceIntakeHold("SOURCE_ITEM_BODY_TOO_LARGE")
        if status != 200 or not raw:
            raise NativeSourceIntakeHold("SOURCE_ITEM_FETCH_INCOMPLETE")
        return url, raw, self._clock().astimezone(UTC)

    @staticmethod
    def _parse_complete_item(item, raw, observed):
        try:
            document = parse_govuk_content_document(
                item.canonical_url, raw, retrieved_at=observed
            )
        except GovUkContentHold as exc:
            raise NativeSourceIntakeHold(
                exc.reason_code,
                child_items=exc.child_items,
                unsupported_attachments=exc.unsupported_attachments,
                exclusion_signals=exc.exclusion_signals,
            ) from None
        except (ValueError, TypeError, KeyError, UnicodeError):
            raise NativeSourceIntakeHold("SOURCE_ITEM_METADATA_HOLD") from None
        if document.exclusion_signals:
            raise NativeSourceIntakeHold(
                "SOURCE_ITEM_RIGHTS_EXCLUSION_HOLD",
                exclusion_signals=document.exclusion_signals,
            )
        return replace(
            item,
            headline=document.title,
            body=document.body_text,
            corpus_body=document.body_text,
            published_at=_utc(document.publication),
            updated_at=_utc(document.updated),
        )

    def _parse_pdf(self, parent_url, parent_raw, asset_url, raw, *, retrieved_at, source_version):
        binding = pdf_parse_binding(parent_url, parent_raw, asset_url, raw,
            retrieved_at=retrieved_at, source_version=source_version)
        request = ObjectAdmissionRequest(NATIVE_SOURCE_OBSERVATION_ADMISSION_TYPE,
            'native-pdf-parse:' + digest_canonical(binding))
        existing = None
        try:
            existing = self._objects.committed_admission(request, proof=self._proof)
            if existing is not None:
                admission = existing.admission
                if not 0 < admission.blob.size_bytes <= MAX_BODY_BYTES:
                    raise ValueError('PDF parse receipt exceeds metadata bound')
                receipt = self._objects.rehydrate(HydrationRequest(admission.admission_id,
                    NATIVE_SOURCE_OBSERVATION_PURPOSE, 0, admission.blob.size_bytes), proof=self._proof)
                outcome = read_pdf_parse_receipt(receipt.data, binding)
                if isinstance(outcome, str):
                    raise GovUkPdfHold(outcome)
                return outcome
        except GovUkPdfHold:
            raise
        except (VetoError, OperatorDrainRequested):
            raise
        except (TypeError, ValueError, LookupError, AuthorityPersistenceError):
            # Invalid optimisation metadata is not source authority. Reparse the
            # freshly acquired declared bytes; never rewrite an immutable receipt.
            existing = False
        try:
            document = parse_govuk_pdf(parent_url, parent_raw, asset_url, raw, retrieved_at=retrieved_at)
        except GovUkPdfHold as exc:
            encoded = pdf_parse_receipt(binding, exc.reason_code)
            if existing is None and encoded is not None:
                admission = self._objects.admit(request, encoded, proof=self._proof).admission
                self._hydrate(admission, NATIVE_SOURCE_OBSERVATION_PURPOSE, encoded)
            raise
        encoded = pdf_parse_receipt(binding, document)
        if existing is None and encoded is not None:
            admission = self._objects.admit(request, encoded, proof=self._proof).admission
            self._hydrate(admission, NATIVE_SOURCE_OBSERVATION_PURPOSE, encoded)
        return document

    def _admit_observation(self, source_id: str, raw: bytes, *, pdf=False):
        admission = self._objects.admit(ObjectAdmissionRequest(
            NATIVE_PDF_OBSERVATION_ADMISSION_TYPE if pdf else NATIVE_SOURCE_OBSERVATION_ADMISSION_TYPE,
            f"native-source-observation:{source_id}:{digest_bytes(raw)}"
        ), raw, proof=self._proof).admission
        access = self._hydrate(admission, NATIVE_PDF_OBSERVATION_PURPOSE if pdf else NATIVE_SOURCE_OBSERVATION_PURPOSE, raw)
        return admission, access

    @staticmethod
    def _govuk_item_path(item):
        path = urlsplit(item.canonical_url).path
        if not path.startswith("/") or item.canonical_url != "https://www.gov.uk" + path:
            return None
        if item.item_key in (item.canonical_url, "tag:www.gov.uk,2005:" + path):
            return path
        root, separator, tail = item.item_key.partition("|")
        if separator != "|" or tail != path:
            return None
        try:
            validate_sha256_digest(root)
        except ValueError:
            return None
        return path

    def _manual_item_key(self, source_id, version_id, item):
        path = self._govuk_item_path(item)
        if path is None:
            return item.item_key
        # First journal LAND wins, then the first successful retention this poll.
        # Keep that exact namespace and its original ancestry proof.
        for unit in chain(iter_source_headers(self._retained_units),
                          iter_source_headers(self._pending_units)):
            if (unit.source_id == source_id
                    and unit.definition_version_id == str(version_id)
                    and unit.canonical_url == item.canonical_url
                    and self._govuk_item_path(unit) == path):
                return unit.item_key
        return item.item_key

    def _retain_item(self, source_id, definition_id, version_id, version, item,
                     observation_digest, observed_at, rights_id):
        item = replace(item, item_key=self._manual_item_key(source_id, version_id, item))
        item_id = deterministic_uuid4(
            SourceItemId, namespace=f"{VERSION}:item",
            semantic_value=[str(version_id), source_id, item.item_key],
        )
        item_request = SourceItemRequest(
            item_id, definition_id, version_id, SourceItemIdentityKind.COMPOSITE,
            version.item_identity_policy, item.item_key,
            (IdentityComponent("item_key", item.item_key), IdentityComponent("source_id", source_id)),
            (), f"native-source-item:{item_id}",
        )
        body = item.retained_corpus_body
        revision_digest = content_digest(
            headline=item.headline, body=body, canonical_url=item.canonical_url
        )
        latest = self._sources.latest_revision(item_id, proof=self._proof)
        native_revision_token = item.updated_at
        if (
            native_revision_token is not None
            and latest is not None
            and latest.request.source_native_revision_token == native_revision_token
            and latest.request.permitted_state_digest != revision_digest
        ):
            raise NativeSourceIntakeHold("SOURCE_NATIVE_REVISION_CONFLICT")
        revision_id = deterministic_uuid4(
            SourceRevisionId, namespace=f"{VERSION}:revision",
            semantic_value=[str(item_id), native_revision_token, revision_digest],
        )
        try:
            retained_revision = self._sources.revision(revision_id, proof=self._proof)
        except LookupError:
            retained_revision = None
        replay = retained_revision is not None
        first_observed = (
            retained_revision.request.observed_at.to_text() if replay else observed_at
        )
        prior_revision_id = (
            retained_revision.request.prior_revision_id if replay
            else None if latest is None else latest.request.revision_id
        )
        revision_request = SourceRevisionRequest(
            revision_id, item_id, version_id, prior_revision_id,
            native_revision_token, revision_digest, version.revision_policy,
            VERSION, self._source_time(item.published_at), self._source_time(item.updated_at),
            UtcTimestamp.parse(first_observed), f"native-source-revision:{revision_id}",
        )
        representation_digest = representation_digest_for(
            source_id=source_id, item_key=item.item_key, revision_digest=revision_digest,
            published_at=item.published_at, updated_at=item.updated_at,
        )
        representation_id = deterministic_uuid4(
            DiscoveryRepresentationId, namespace=f"{VERSION}:representation",
            semantic_value=[str(revision_id), representation_digest],
        )
        fields_digest = digest_canonical({
            "headline": item.headline, "body": body, "canonical_url": item.canonical_url,
            "published_at": item.published_at, "updated_at": item.updated_at,
        })
        representation_request = DiscoveryRepresentationRequest(
            representation_id, revision_id, version_id, VERSION, VERSION, VERSION, VERSION,
            fields_digest, representation_digest, UtcTimestamp.parse(first_observed),
            f"native-source-representation:{representation_id}",
        )
        retained = self._retained_units.get(str(revision_id), self._pending_units.get(str(revision_id)))
        if retained is not None:
            # Polling already checked the current definition/rights and fetched
            # the complete bytes. Reuse the journal's original chunk receipts,
            # rather than minting ones journal.land would immediately discard.
            NativeRevisionJournal._validate_units(retained)
            representation = self._sources.representation(representation_id, proof=self._proof)
            if (
                retained_revision is None or retained_revision.request != revision_request
                or representation.request != representation_request
                or any(
                    unit.source_id != source_id or unit.item_key != item.item_key
                    or unit.representation_digest != representation_digest
                    or unit.source_definition_url != version.locator
                    or unit.authority.definition_id != str(definition_id)
                    or unit.authority.definition_version_id != str(version_id)
                    or unit.authority.item_id != str(item_id)
                    or unit.authority.revision_id != str(revision_id)
                    or unit.authority.representation_id != str(representation_id)
                    for unit in retained
                )
            ):
                raise NativeSourceIntakeHold("SOURCE_RETAINED_REVISION_BINDING_HOLD")
            return retained
        self._sources.register_item(item_request, proof=self._proof)
        self._sources.record_revision(revision_request, proof=self._proof)
        self._sources.record_representation(representation_request, proof=self._proof)
        base = CorpusIngestUnit(
            source_id, item.item_key, item.headline, body, item.canonical_url,
            observation_digest, observed_at, f"native-source:{observation_digest}",
            EffectiveRevisionIdentity(source_id, item.item_key, revision_digest, first_observed),
            item.published_at, item.updated_at, source_definition_url=version.locator,
            effective_pull_first_observed_at=first_observed,
        )
        chunks = chunk_text(base.full_text)
        result, predecessor = [], None
        source_records = (
            {"record_type": "SOURCE_DEFINITION", "record_id": str(definition_id), "source_id": source_id},
            {"record_type": "SOURCE_DEFINITION_VERSION", "record_id": str(version_id), "definition_id": str(definition_id), "source_id": source_id, "source_url": version.locator},
            {"record_type": "SOURCE_ITEM", "record_id": str(item_id), "definition_id": str(definition_id), "source_id": source_id, "item_key": item.item_key},
            {"record_type": "SOURCE_REVISION", "record_id": str(revision_id), "item_id": str(item_id), "source_id": source_id, "item_key": item.item_key, "revision_digest": revision_digest},
            {"record_type": "DISCOVERY_REPRESENTATION", "record_id": str(representation_id), "source_id": source_id, "item_key": item.item_key, "revision_id": str(revision_id), "representation_digest": representation_digest},
        )
        for ordinal, chunk in enumerate(chunks, 1):
            provisional = replace(base, chunk_ordinal=ordinal, chunk_count=len(chunks), predecessor_ingest_id=predecessor)
            data = " ".join(provisional.episode_body.split()).encode()
            admission = self._objects.admit(ObjectAdmissionRequest(
                OPERATIONAL_ADMISSION_TYPE, f"native-source-passage:{provisional.ingest_id}:{rights_id}"
            ), data, proof=self._proof).admission
            access = self._hydrate(admission, GRAPHITI_EVALUATION_HYDRATION_POLICY.purpose, data)
            binding = CorpusAuthorityBinding(
                str(admission.admission_id), str(access.access_decision_id), str(definition_id),
                str(version_id), str(item_id), str(revision_id), str(representation_id),
                (*source_records,
                 {"record_type": "OBJECT_ADMISSION", "record_id": str(admission.admission_id), "revision_id": str(revision_id), "decision": "ADMIT", "scope": "EVALUATION_CORPUS_INGEST"},
                 {"record_type": "OBJECT_ACCESS_DECISION", "record_id": str(access.access_decision_id), "revision_id": str(revision_id), "decision": "ALLOW", "principal_id": access.principal_id, "authority_domain": access.authority_domain, "purpose": access.purpose}),
            )
            unit = replace(provisional, authority=binding)
            result.append(unit)
            predecessor = unit.ingest_id
        settled = tuple(result)
        self._pending_units[str(revision_id)] = settled
        return settled

    def _hydrate(self, admission, purpose: str, expected: bytes):
        try:
            access = self._objects.latest_access_decision(
                admission.admission_id, purpose=purpose, proof=self._proof
            )
        except KeyError:
            hydrated = self._objects.hydrate(HydrationRequest(
                admission.admission_id, purpose, 0, len(expected)
            ), proof=self._proof)
            if hydrated.data != expected:
                raise ValueError("native source object hydration differs")
            access = hydrated.decision
        if access.admission_id != admission.admission_id or access.allowed_bytes != len(expected):
            raise ValueError("native source object access differs")
        return access

    @staticmethod
    def _source_time(value: str | None) -> SourceTime:
        return SourceTime.unknown() if not value else SourceTime.exact(UtcTimestamp.parse(value))


def native_evidence_sources(
    *, units: tuple[CorpusIngestUnit, ...], sources, objects,
    observations: Mapping[str, tuple[str, str, str, str]],
    licence: GovUkLicenceEvidence, proof: AuthenticationProof,
) -> tuple[NativeEvidenceSource, ...]:
    """Bind retained complete-page revisions to independent evidence inputs."""

    if not units or any(type(unit) is not CorpusIngestUnit for unit in units):
        raise ValueError("native evidence source units differ")
    grouped: dict[str, list[CorpusIngestUnit]] = {}
    for unit in units:
        grouped.setdefault(unit.revision_id, []).append(unit)
    result = []
    for revision_units in grouped.values():
        ordered = tuple(sorted(revision_units, key=lambda item: item.chunk_ordinal))
        unit = ordered[0]

        def hold(reason: str):
            return NativeEvidenceHold(reason, unit.source_id)

        authority = unit.authority
        weather = unit.source_id in {"HK-02", "UK-10"}
        if (
            authority is None
            or tuple(item.chunk_ordinal for item in ordered)
            != tuple(range(1, unit.chunk_count + 1))
            or any(
                item.chunk_count != unit.chunk_count
                or item.revision_id != unit.revision_id
                or item.body != unit.body
                or item.headline != unit.headline
                or item.canonical_url != unit.canonical_url
                for item in ordered
            )
            or unit.proving_run_id != "native-source:" + unit.observation_digest
        ):
            raise hold("NATIVE_SOURCE_CHUNK_BINDING_HOLD")
        try:
            validate_sha256_digest(unit.observation_digest)
            asset_url = spreadsheet_asset_url(unit) or pdf_asset_url(unit)
            expected_api_url = (
                SOURCE_URLS[unit.source_id]
                if weather
                else asset_url
                if asset_url is not None
                else _api_url(unit.canonical_url)
            )
            observation = observations[unit.observation_digest]
            if (
                type(observation) is not tuple
                or len(observation) != 4
                or observation[0] != expected_api_url
                or observation[1] != unit.observation_digest
            ):
                raise ValueError("raw observation reference differs")
            root_digest, separator, _child_path = unit.item_key.partition("|")
            parent_inventory = None
            if separator == "|" and root_digest.startswith("sha256:"):
                parent_inventory = _require_parent_inventory_binding(
                    unit=unit, observations=observations, objects=objects,
                    proof=proof,
                )
            version = sources.version_details(
                SourceDefinitionVersionId.parse(authority.definition_version_id),
                proof=proof,
            )
            revision = sources.revision(
                SourceRevisionId.parse(authority.revision_id), proof=proof,
            )
            representation = sources.representation(
                DiscoveryRepresentationId.parse(authority.representation_id),
                proof=proof,
            )
            current = sources.current_summary(
                SourceDefinitionId.parse(authority.definition_id), proof=proof,
            )
        except (TypeError, ValueError, LookupError, KeyError, PermissionError):
            raise hold("NATIVE_SOURCE_AUTHORITY_HOLD") from None
        request = version.request
        roles = tuple(
            role for role in request.roles
            if role.role in {SourceRole.ORIGINATING_AUTHORITY, SourceRole.RESPONSIBLE_OPERATOR}
        )
        dependencies = tuple(
            dependency for dependency in request.dependencies
            if dependency.kind is SourceDependencyKind.ORIGINATING_MATERIAL
        )
        if (
            current.version_id != version.version_id
            or str(request.definition_id) != authority.definition_id
            or request.locator != unit.source_definition_url
            or len(roles) != 1
            or (len(dependencies) != 1 and not (unit.source_id == "UK-10" and not dependencies))
            or revision.request.item_id != SourceItemId.parse(authority.item_id)
            or revision.request.definition_version_id != version.version_id
            or revision.request.permitted_state_digest != unit.revision_digest
            or representation.request.revision_id != revision.request.revision_id
            or representation.request.definition_version_id != version.version_id
            or representation.request.representation_digest != unit.representation_digest
        ):
            raise hold("NATIVE_SOURCE_AUTHORITY_HOLD")
        for item in ordered:
            expected = " ".join(item.episode_body.split()).encode()
            hydrated = objects.rehydrate(HydrationRequest(
                ObjectAdmissionId.parse(item.authority.admission_id),
                GRAPHITI_EVALUATION_HYDRATION_POLICY.purpose, 0, len(expected),
            ), proof=proof)
            if hydrated.data != expected:
                raise hold("NATIVE_SOURCE_CANONICAL_PAGE_HOLD")
        try:
            verified = verified_native_observation(
                unit=unit, observations=observations, objects=objects,
                proof=proof, expected_url=expected_api_url,
            )
            raw = verified.raw
            retrieved = datetime.fromisoformat(unit.observed_at.replace("Z", "+00:00"))
            if weather:
                from .native_weather_sources import weather_items
                items = tuple(item for item in weather_items(
                    unit.source_id, raw, observed_at=retrieved,
                ) if item.item_key == unit.item_key)
                matches = len(items) == 1 and (
                    items[0].canonical_url == unit.canonical_url
                    and items[0].headline == unit.headline
                    and items[0].retained_corpus_body == unit.body
                    and items[0].published_at == unit.published_at
                    and items[0].updated_at == unit.updated_at
                )
            else:
                if asset_url is not None:
                    if parent_inventory is None:
                        raise ValueError("spreadsheet parent inventory is absent")
                    parent_url, parent_raw = parent_inventory
                    parse = parse_govuk_pdf if is_pdf_url(asset_url) else parse_govuk_spreadsheet
                    document = parse(
                        parent_url, parent_raw, asset_url, raw,
                        retrieved_at=retrieved,
                    )
                else:
                    document = parse_govuk_content_document(
                        unit.canonical_url, raw, retrieved_at=retrieved,
                    )
                matches = (
                    document.title == unit.headline
                    and document.body_text == unit.body
                    and _utc(document.publication) == unit.published_at
                    and _utc(document.updated) == unit.updated_at
                )
        except GovUkContentHold as exc:
            raise hold(exc.reason_code) from None
        except (TypeError, ValueError, KeyError, UnicodeError, PermissionError):
            raise hold("NATIVE_SOURCE_RAW_OBSERVATION_HOLD") from None
        if (
            digest_bytes(raw) != unit.observation_digest
            or not matches
        ):
            raise hold("NATIVE_SOURCE_RAW_OBSERVATION_HOLD")
        rights = licence.for_source(
            source_id=unit.source_id, definition_url=request.locator,
        )
        if rights.decision != "PERMITTED":
            raise hold("NATIVE_SOURCE_RIGHTS_HOLD")
        dependency = dependencies[0] if dependencies else None
        dependency_receipt = DependencyAssessment.create(
            dependency_status="RESOLVED",
            evidential_origin_id=unit.observation_digest,
            originating_report_id=authority.revision_id,
            evidence_digest=digest_canonical({
                "source_definition_version": version.canonical_digest,
                "source_revision": revision.canonical_digest,
                "originating_material_dependency": None if dependency is None else dependency.canonical_value(),
                "canonical_url": unit.canonical_url,
                "observation_digest": unit.observation_digest,
                "canonical_page_digest": unit.revision_digest,
            }),
        )
        result.append(NativeEvidenceSource(unit, version, rights, dependency_receipt))
    return tuple(result)


def _require_parent_inventory_binding(
    *, unit: CorpusIngestUnit,
    observations: Mapping[str, tuple[str, str, str, str]], objects,
    proof: AuthenticationProof,
) -> tuple[str, bytes]:
    root_digest, separator, section_path = unit.item_key.partition("|")
    validate_sha256_digest(root_digest)
    root = observations[root_digest]
    if separator != "|" or root[1] != root_digest:
        raise ValueError("parent inventory reference differs")
    admission_id, access = _require_observation_access(
        observation=root, objects=objects, proof=proof,
    )
    raw = objects.rehydrate(HydrationRequest(
        admission_id, NATIVE_SOURCE_OBSERVATION_PURPOSE, 0, access.allowed_bytes,
    ), proof=proof).data
    if digest_bytes(raw) != root_digest:
        raise ValueError("parent inventory bytes differ")
    if unit.source_id == "UK-03":
        if root[0] != unit.source_definition_url:
            raise ValueError("manual root inventory reference differs")
    else:
        feed_observations = tuple(
            value for value in observations.values()
            if value[0] == unit.source_definition_url
        )
        if not feed_observations:
            raise ValueError("parent feed observation differs")
        parent_url = _canonical_url_from_api(root[0])
        parent_found = False
        for feed in feed_observations:
            try:
                feed_admission_id, feed_access = _require_observation_access(
                    observation=feed, objects=objects, proof=proof,
                )
                feed_raw = objects.rehydrate(HydrationRequest(
                    feed_admission_id, NATIVE_SOURCE_OBSERVATION_PURPOSE,
                    0, feed_access.allowed_bytes,
                ), proof=proof).data
                if digest_bytes(feed_raw) != feed[1]:
                    continue
                feed_items = parse_observation(
                    source_id=unit.source_id, url=unit.source_definition_url,
                    body=feed_raw,
                )
                parent_found = any(item.canonical_url == parent_url for item in feed_items)
                if not parent_found and (
                    spreadsheet_asset_url(unit) is not None or pdf_asset_url(unit) is not None
                    or _declared_publication_leaves(raw, parent_url)
                ):
                    parent_found = _declared_file_parent_in_feed_child(
                        unit=unit, parent_url=parent_url, feed_items=feed_items,
                        observations=observations, objects=objects, proof=proof,
                        collection_only=spreadsheet_asset_url(unit) is None and pdf_asset_url(unit) is None,
                    )
                if parent_found:
                    break
            except (TypeError, ValueError, LookupError, KeyError, PermissionError):
                continue
        if not parent_found:
            raise ValueError("parent is outside its retained feed inventory")
    parent_url = _canonical_url_from_api(root[0])
    try:
        parse_govuk_content_document(
            parent_url, raw,
            retrieved_at=datetime.fromisoformat(unit.observed_at.replace("Z", "+00:00")),
        )
    except GovUkContentHold as exc:
        child_paths = {path for path, _title in exc.child_items}
    else:
        raise ValueError("parent inventory is absent")
    asset_url = spreadsheet_asset_url(unit) or pdf_asset_url(unit)
    if asset_url is not None:
        if section_path != asset_url or parent_url != unit.canonical_url:
            raise ValueError("spreadsheet child identity differs")
        declare = declared_pdf if is_pdf_url(asset_url) else declared_spreadsheet
        declare(
            parent_url, raw, asset_url,
            retrieved_at=datetime.fromisoformat(
                unit.observed_at.replace("Z", "+00:00")
            ),
        )
    elif section_path not in child_paths:
        raise ValueError("child is outside its retained parent inventory")
    return parent_url, raw


def _declared_publication_leaves(raw, parent_url) -> bool:
    """Allow only a publication's strictly descended declared HTML inventory."""
    try:
        value = json.loads(raw)
        path = urlsplit(parent_url).path
        children, _unsupported = _require_attachment_inventory(value)
        return (
            value.get("schema_name") == "publication"
            and value.get("document_type") in {"correspondence", "corporate_report", "regulation", "transparency", "guidance", "national_statistics", "form"}
            and value.get("base_path") == path
            and path.startswith(
                "/government/statistics/" if value.get("document_type") == "national_statistics"
                else "/government/publications/"
            )
            and bool(children)
            and all(child.startswith(path + "/") for child, _title in children)
        )
    except (TypeError, ValueError, AttributeError):
        return False


def _declared_file_parent_in_feed_child(
    *, unit, parent_url, feed_items, observations, objects, proof,
    collection_only=False,
) -> bool:
    """Prove exactly feed -> direct inventory -> attachment parent; never crawl."""
    feed_urls = set()
    for item in feed_items:
        try:
            feed_urls.add(_api_url(item.canonical_url))
        except ValueError:
            continue
    for observation in observations.values():
        if observation[0] not in feed_urls:
            continue
        try:
            admission_id, access = _require_observation_access(
                observation=observation, objects=objects, proof=proof,
            )
            raw = objects.rehydrate(HydrationRequest(
                admission_id, NATIVE_SOURCE_OBSERVATION_PURPOSE, 0, access.allowed_bytes,
            ), proof=proof).data
            if digest_bytes(raw) != observation[1]:
                continue
            if collection_only and json.loads(raw).get("document_type") != "document_collection":
                continue
            try:
                parse_govuk_content_document(
                    _canonical_url_from_api(observation[0]), raw,
                    retrieved_at=datetime.fromisoformat(unit.observed_at.replace("Z", "+00:00")),
                )
            except GovUkContentHold as exc:
                if (not exc.exclusion_signals
                        and urlsplit(parent_url).path in {path for path, _title in exc.child_items}):
                    return True
        except (TypeError, ValueError, LookupError, KeyError, PermissionError):
            continue
    return False


def _require_observation_access(*, observation, objects, proof):
    purpose = NATIVE_PDF_OBSERVATION_PURPOSE if is_pdf_url(observation[0]) else NATIVE_SOURCE_OBSERVATION_PURPOSE
    admission_id = ObjectAdmissionId.parse(observation[2])
    retained = objects.access_decision(
        ObjectAccessDecisionId.parse(observation[3]),
        admission_id=admission_id,
        purpose=purpose,
        proof=proof,
    )
    current = objects.latest_access_decision(
        admission_id, purpose=purpose, proof=proof,
    )
    if (
        retained.admission_id != admission_id
        or retained.purpose != purpose
        or retained.offset != 0
        or retained.allowed_bytes != current.allowed_bytes
        or current.admission_id != admission_id
    ):
        raise ValueError("observation access differs")
    return admission_id, current


def verified_native_observation(
    *, unit: CorpusIngestUnit,
    observations: Mapping[str, tuple[str, str, str, str]],
    objects, proof: AuthenticationProof, expected_url: str,
) -> VerifiedNativeObservation:
    """Rehydrate the exact retained raw observation already bound to a unit."""

    validate_sha256_digest(unit.observation_digest)
    observation = observations[unit.observation_digest]
    if (
        type(observation) is not tuple
        or len(observation) != 4
        or observation[0] != expected_url
        or observation[1] != unit.observation_digest
    ):
        raise ValueError("raw observation reference differs")
    admission_id, access = _require_observation_access(
        observation=observation, objects=objects, proof=proof,
    )
    raw = objects.rehydrate(HydrationRequest(
        admission_id, NATIVE_PDF_OBSERVATION_PURPOSE if is_pdf_url(expected_url) else NATIVE_SOURCE_OBSERVATION_PURPOSE,
        0, access.allowed_bytes,
    ), proof=proof).data
    if digest_bytes(raw) != observation[1]:
        raise ValueError("raw observation digest differs")
    return VerifiedNativeObservation(
        observation[0], observation[1], admission_id,
        str(access.access_decision_id), raw,
    )


__all__ = [
    "NativeSourceDisposition", "NativeSourceIntake", "SOURCE_IDS",
    "VerifiedNativeObservation", "native_evidence_sources",
    "spreadsheet_asset_url", "verified_native_observation",
]
