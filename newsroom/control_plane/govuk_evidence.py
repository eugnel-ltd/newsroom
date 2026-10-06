"""Independent GOV.UK Content API acquisition for the approved source route.

The runtime binds the exact current Source Registry and its dispatch fence.
Only the fixed public HTTPS Content API is reachable; there are no credentials,
redirects, browser execution, model calls or caller-selected network backends.
See https://content-api.publishing.service.gov.uk/getting-started.html.
"""

from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.request
from contextlib import AbstractContextManager
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from urllib.parse import unquote, urlsplit

from lxml import etree, html

from newsroom.authority import AuthenticationProof
from newsroom.authority.canonical import (
    digest_bytes, digest_canonical, validate_sha256_digest,
)
from newsroom.sources import SourceDefinitionVersionId, SourceRevisionId

from .native_evidence import (
    AcquiredEvidence, EvidenceAcquisitionRequest, NativeEvidenceHold,
    rights_eligibility_digest,
)

VERSION = "hermes-govuk-evidence-v1"
MAX_BODY_BYTES = 1_048_576
TIMEOUT_SECONDS = 20
POLICY_DIGEST = digest_canonical({
    "version": VERSION, "origin": "https://www.gov.uk",
    "api_prefix": "/api/content", "method": "GET", "redirects": 0,
    "max_bytes": MAX_BODY_BYTES, "timeout_seconds": TIMEOUT_SECONDS,
    "credentials": False,
})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True, slots=True)
class GovUkAssetScopeExclusion:
    asset_url: str
    mime: str
    raw_root_digest: str
    definition_scope: tuple[str, ...]
    policy_version: str = "newsroom.govuk-text-body-image-scope.v1"
    disposition: str = "SOURCE_SCOPE_EXCLUDED"


@dataclass(frozen=True, slots=True)
class GovUkContentDocument:
    document_type: str
    title: str
    body_text: str
    publication: datetime
    updated: datetime
    organisations: tuple[str, ...]
    exclusion_signals: tuple[str, ...]
    scope_excluded_assets: tuple[GovUkAssetScopeExclusion, ...] = field(default=(), kw_only=True)


@dataclass(frozen=True, slots=True)
class GovUkManualInventory:
    title: str
    publication: datetime
    updated: datetime
    organisations: tuple[str, ...]
    sections: tuple[tuple[str, str], ...]


class GovUkContentHold(ValueError):
    """A valid known GOV.UK content shape needing a different coverage path."""

    def __init__(
        self,
        reason_code: str,
        *,
        child_items: tuple[tuple[str, str], ...] = (),
        unsupported_attachments: tuple[tuple[str, str], ...] = (),
        exclusion_signals: tuple[str, ...] = (),
        scope_excluded_assets: tuple[GovUkAssetScopeExclusion, ...] = (),
    ) -> None:
        self.reason_code = reason_code
        self.child_items = child_items
        self.unsupported_attachments = unsupported_attachments
        # References are retained metadata, never an executable fetch route or
        # proof that historical material is unnecessary for this source.
        self.archival_references = tuple(
            (url, title) for url, title in unsupported_attachments
            if _archival_reference_location(url)
        )
        self.historical_coverage_status = "UNASSESSED" if self.archival_references else None
        self.exclusion_signals = exclusion_signals
        self.scope_excluded_assets = scope_excluded_assets
        super().__init__(reason_code)


def _instant(value: object) -> datetime:
    if type(value) is not str:
        raise ValueError("source publication time is missing")
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("source publication time lacks offset")
    return instant.astimezone(UTC)


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _api_url(canonical_url: str) -> str:
    parsed = urlsplit(canonical_url)
    path = unquote(parsed.path)
    if (
        parsed.scheme != "https" or parsed.netloc != "www.gov.uk"
        or parsed.query or parsed.fragment or not path.startswith("/")
        or path.startswith("//") or "\\" in path
        or any(part in {".", ".."} for part in path.split("/"))
        or any(ord(character) < 32 for character in canonical_url + path)
        or parsed.path.startswith("/api/")
    ):
        raise ValueError("canonical source URL is outside the fixed GOV.UK route")
    return "https://www.gov.uk/api/content" + parsed.path


class GovUkEvidenceAcquisition:
    """A concrete bounded GET, with current rights/stop rechecked before I/O."""

    def __init__(
        self, *, sources, proof: AuthenticationProof,
        dispatch_fence: Callable[[EvidenceAcquisitionRequest], AbstractContextManager[None]],
        clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
        licence_evidence=None,
        transport_policy_digest: str = POLICY_DIGEST,
    ) -> None:
        validate_sha256_digest(transport_policy_digest)
        self._sources = sources
        self._proof = proof
        self._fence = dispatch_fence
        self._clock = clock
        self._licence = licence_evidence
        self._transport_policy_digest = transport_policy_digest

    def __call__(self, request: EvidenceAcquisitionRequest) -> AcquiredEvidence:
        def hold(reason: str):
            return NativeEvidenceHold(reason, request.source_id)

        if type(request) is not EvidenceAcquisitionRequest:
            raise TypeError("exact independent acquisition request required")
        if request.transport_policy_digest != self._transport_policy_digest:
            raise hold("TRANSPORT_POLICY_MISMATCH")
        try:
            url = _api_url(request.canonical_url)
            version = self._sources.version_details(
                SourceDefinitionVersionId.parse(request.source_definition_version_id),
                proof=self._proof,
            )
            revision = self._sources.revision(
                SourceRevisionId.parse(request.source_revision_id), proof=self._proof,
            )
            if (
                str(version.request.definition_id) != request.source_definition_id
                or version.canonical_digest != request.source_definition_version_digest
                or revision.request.definition_version_id != version.version_id
                or urlsplit(version.request.locator).netloc != "www.gov.uk"
            ):
                raise ValueError("exact source binding differs")
        except (ValueError, LookupError):
            raise hold("GOVUK_SOURCE_BINDING_HOLD") from None
        # The caller supplies the existing signed-stop/current-rights fence,
        # not a per-story human approval. No SQLite transaction spans this I/O.
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect(),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        )
        http_request = urllib.request.Request(url, method="GET", headers={
            "User-Agent": "Newsroom-Hermes/1.0", "Accept": "application/json",
            "Accept-Encoding": "identity",
        })
        try:
            with self._fence(request), opener.open(http_request, timeout=TIMEOUT_SECONDS) as response:
                status = response.status
                content_type = response.headers.get_content_type()
                response_url = response.geturl()
                raw = response.read(MAX_BODY_BYTES + 1)
        except (urllib.error.URLError, TimeoutError, OSError):
            raise hold("GOVUK_ACQUISITION_UNAVAILABLE") from None
        retrieved = self._clock()
        if (
            status != 200 or response_url != url or content_type != "application/json"
            or not raw or len(raw) > MAX_BODY_BYTES
        ):
            raise hold("GOVUK_ACQUISITION_INCOMPLETE")
        try:
            document = parse_govuk_content_document(
                request.canonical_url, raw, retrieved_at=retrieved,
                extraction_scope=getattr(version.request, "extraction_scope", ()),
            )
            body = (document.title + "\n\n" + document.body_text).encode("utf-8")
        except (ValueError, TypeError, KeyError, UnicodeError, etree.ParserError):
            raise hold("GOVUK_EVIDENCE_METADATA_HOLD") from None
        transport_digest = digest_canonical({
            "version": VERSION, "request_digest": request.digest,
            "url": url, "response_url": response_url, "http_status": status,
            "content_type": content_type, "response_digest": digest_bytes(raw),
            "extracted_body_digest": digest_bytes(body),
            "public_updated_at": _utc(document.updated), "retrieved_at": _utc(retrieved),
            **({"scope_excluded_assets": [asdict(item) for item in document.scope_excluded_assets]}
               if document.scope_excluded_assets else {}),
        })
        # These are observed acquisition facts, not six invented semantic PASS
        # decisions. Editorial claim checks still decide what may be rewritten.
        # Image/logo bytes are never acquired or retained by this text route.
        signals = document.exclusion_signals
        rights_digest = ""
        attribution = ""
        if self._licence is not None:
            from .govuk_rights import ATTRIBUTION, POLICY_DIGEST as RIGHTS_POLICY

            rights = self._licence.for_source(
                source_id=request.source_id, definition_url=version.request.locator,
            )
            if rights.decision == "PERMITTED" and rights.policy_digest == RIGHTS_POLICY:
                rights_digest = rights_eligibility_digest(
                    rights, body_digest=digest_bytes(body), transport_digest=transport_digest,
                    exclusion_signals=signals, text_only=True,
                )
                attribution = ATTRIBUTION
        return AcquiredEvidence.create(
            request_digest=request.digest, outcome="COMPLETE",
            canonical_url=request.canonical_url, body=body, body_digest=digest_bytes(body),
            publisher="; ".join(document.organisations),
            responsible_body="; ".join(document.organisations),
            source_type="PRIMARY_OFFICIAL",
            publication_time=_utc(document.publication),
            source_updated_time=_utc(document.updated),
            retrieval_time=_utc(retrieved), geography="UK", language="en-GB",
            transport_evidence_digest=transport_digest,
            currentness_basis="AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT",
            rights_eligibility_digest=rights_digest,
            licence_attribution=attribution,
            exclusion_signals=signals, text_only=True,
            body_origin='GOVUK_CONTENT_API_PAGE_TEXT',
        )


def _unique_object(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("source JSON has duplicate fields")
    return value


def parse_govuk_content_document(
    canonical_url: str, raw: bytes, *, retrieved_at: datetime,
    extraction_scope: tuple[str, ...] = (),
) -> GovUkContentDocument:
    """Validate and extract one complete current GOV.UK Content API document."""

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    if (
        type(value) is not dict
        or value.get("base_path") != urlsplit(canonical_url).path
        or value.get("locale") != "en"
        or type(value.get("document_type")) is not str
        or value.get("withdrawn_notice")
    ):
        raise ValueError("source schema or currentness differs")
    publication = _instant(value.get("first_published_at"))
    updated = _instant(value.get("public_updated_at"))
    if publication > retrieved_at or updated > retrieved_at:
        raise ValueError("source temporal order differs")
    title = value["title"]
    if type(title) is not str or not title.strip():
        raise ValueError("source title is absent")
    names = _organisation_names(value)
    document_type = value["document_type"]
    excluded = ()
    if document_type in {
        "news_story", "press_release", "guidance", "detailed_guide",
        "html_publication", "notice", "policy_paper", "written_statement",
        "guide", "manual_section", "oral_statement", "statistics",
        "speech",
    }:
        body_text = _document_text(value)
        if value.get("details", {}).get("attachments") or value.get("links", {}).get("children"):
            children, unsupported = _require_attachment_inventory(value)
            excluded = _text_body_image_exclusions(value, raw, extraction_scope)
            excluded_urls = {item.asset_url for item in excluded}
            unsupported = tuple(item for item in unsupported if item[0] not in excluded_urls)
            if children or unsupported:
                raise GovUkContentHold(
                    "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE",
                    child_items=children,
                    unsupported_attachments=unsupported,
                    exclusion_signals=_exclusion_signals(value, body_text),
                    scope_excluded_assets=excluded,
                )
    elif document_type == "official_statistics_announcement":
        _require_future_statistics_announcement(value, retrieved_at=retrieved_at)
        raise GovUkContentHold("SOURCE_ITEM_NOT_YET_PUBLISHED")
    elif document_type == "manual":
        inventory = parse_govuk_manual_inventory(
            canonical_url, raw, retrieved_at=retrieved_at,
        )
        raise GovUkContentHold(
            "SOURCE_ITEM_CHILD_COVERAGE_INCOMPLETE",
            child_items=inventory.sections,
            exclusion_signals=_exclusion_signals(value, ""),
        )
    elif document_type == "document_collection":
        children = _require_collection_inventory(value)
        body = value["details"].get("body")
        raise GovUkContentHold(
            "SOURCE_ITEM_CHILD_COVERAGE_INCOMPLETE", child_items=children,
            exclusion_signals=_exclusion_signals(value, _html_text(body, allow_empty=True) if body else ""),
        )
    elif document_type == "transparency":
        children, unsupported = _require_attachment_inventory(value)
        raise GovUkContentHold(
            "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE",
            child_items=children,
            unsupported_attachments=unsupported,
            exclusion_signals=_exclusion_signals(value, ""),
        )
    elif document_type in {"correspondence", "corporate_report", "regulation", "national_statistics", "form"}:
        if value.get("schema_name") != "publication":
            raise ValueError("source attachment-bearing schema differs")
        body_text = _document_text(value)
        children, unsupported = _require_attachment_inventory(value)
        if document_type == "form" and not children:
            raise ValueError("source form HTML inventory is absent")
        raise GovUkContentHold(
            "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE",
            child_items=children,
            unsupported_attachments=unsupported,
            exclusion_signals=_exclusion_signals(value, body_text),
        )
    elif document_type == "statutory_guidance":
        if value.get("schema_name") != "publication":
            raise ValueError("source statutory guidance schema differs")
        _document_text(value)
        children, unsupported = _require_attachment_inventory(value)
        raise GovUkContentHold(
            "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE",
            child_items=children,
            unsupported_attachments=unsupported,
            exclusion_signals=_exclusion_signals(value, ""),
        )
    elif document_type == "consultation_outcome":
        details = value.get("details")
        if value.get("schema_name") != "consultation" or type(details) is not dict:
            raise ValueError("source consultation outcome schema differs")
        _document_text(value)
        _html_text(details.get("final_outcome_detail"))
        outcome_attachments = details.get("final_outcome_attachments")
        if (
            type(outcome_attachments) is not list
            or not outcome_attachments
            or any(type(item) is not str or not item for item in outcome_attachments)
            or len(set(outcome_attachments)) != len(outcome_attachments)
        ):
            raise ValueError("source consultation outcome inventory differs")
        children, unsupported = _require_attachment_inventory(value)
        attachments = details["attachments"]
        attachment_ids = [item.get("id") for item in attachments]
        if (
            any(type(item) is not str or not item for item in attachment_ids)
            or len(set(attachment_ids)) != len(attachment_ids)
            or any(item not in attachment_ids for item in outcome_attachments)
        ):
            raise ValueError("source consultation outcome inventory differs")
        raise GovUkContentHold(
            "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE",
            child_items=children,
            unsupported_attachments=unsupported,
            exclusion_signals=_exclusion_signals(value, ""),
        )
    else:
        raise ValueError("source document type is unsupported")
    return GovUkContentDocument(
        document_type, title.strip(), body_text, publication, updated, names,
        _exclusion_signals(value, body_text),
        scope_excluded_assets=excluded,
    )


def _text_body_image_exclusions(value, raw, extraction_scope):
    """Scope metadata only: no claim about unseen image contents or extraction."""
    text_fields = {"body", "canonical_url", "headline", "published_at", "updated_at"}
    if (type(extraction_scope) is not tuple or not extraction_scope
            or any(type(field) is not str for field in extraction_scope)
            or "body" not in extraction_scope or not set(extraction_scope) <= text_fields):
        return ()
    details = value["details"]
    fragment = details.get("body")
    if type(fragment) is not str:
        return ()
    document = html.fragment_fromstring(fragment, create_parent="div")
    references = "\n".join(unquote(text) for text in (
        fragment, document.text_content(), *document.xpath(".//@href | .//@src | .//@data-src"),
    )).casefold()
    # A meaningful visual/data relationship requires evidence, not a format bypass.
    if (document.xpath(".//figure | .//figcaption | .//table | .//img")
            or re.search(r"\b(?:diagram|figure|image data|chart|table)\b", document.text_content(), re.I)
            or any(details.get(key) for key in ("figures", "diagrams", "charts", "tables"))):
        return ()
    images = details.get("images", [])
    if (type(images) is not list or any(type(image) is not dict
            or image.get("type") not in (None, "lead")
            or any(image.get(key) not in (None, "") for key in ("caption", "description", "alt_text")) for image in images)):
        return ()
    result = []
    for entry in details.get("attachments", []):
        url, mime, filename = entry.get("url"), entry.get("content_type"), entry.get("filename")
        extensions = {"image/jpeg": {".jpg", ".jpeg"}, "image/png": {".png"}}.get(mime, set())
        if (entry.get("attachment_type") != "file" or type(url) is not str
                or not _safe_attachment_location(url) or urlsplit(url).netloc != "assets.publishing.service.gov.uk"
                or type(filename) is not str or not filename
                or unquote(urlsplit(url).path).rsplit("/", 1)[-1] != filename
                or not any(filename.lower().endswith(extension) for extension in extensions)
                or any(entry.get(key) not in (None, "") for key in ("caption", "description", "alt_text"))
                or any(entry.get(key) for key in ("role", "type", "data_table", "diagram", "figure"))
                or unquote(url).casefold() in references or filename.casefold() in references):
            continue
        result.append(GovUkAssetScopeExclusion(url, mime, digest_bytes(raw), extraction_scope))
    return tuple(result)


def _require_future_statistics_announcement(
    value: dict, *, retrieved_at: datetime,
) -> None:
    details = value.get("details")
    if type(details) is not dict:
        raise ValueError("source announcement details are absent")
    release = _instant(details.get("release_timestamp"))
    if (
        details.get("state") not in {"confirmed", "provisional"}
        or type(details.get("display_date")) is not str
        or not details["display_date"].strip()
        or release <= retrieved_at
    ):
        raise ValueError("source announcement is not a future release")


def _require_link_inventory(
    value: dict, *, key: str,
) -> tuple[tuple[str, str], ...]:
    links = value.get("links")
    entries = links.get(key) if type(links) is dict else None
    if type(entries) is not list or not entries:
        raise ValueError("source child inventory is absent")
    items = []
    for entry in entries:
        if type(entry) is not dict:
            raise ValueError("source child inventory differs")
        path, title = entry.get("base_path"), entry.get("title")
        if (
            type(path) is not str
            or not path.startswith("/")
            or type(title) is not str
            or not title.strip()
        ):
            raise ValueError("source child identity differs")
        try:
            _api_url("https://www.gov.uk" + path)
        except ValueError:
            raise ValueError("source child identity differs") from None
        items.append((path, title.strip()))
    if len({path for path, _title in items}) != len(items):
        raise ValueError("source child inventory is incomplete")
    return tuple(items)


def _require_collection_inventory(value: dict) -> tuple[tuple[str, str], ...]:
    """Validate either structured children or an observed body-backed collection."""
    links = value.get("links")
    if type(links) is dict and links.get("documents"):
        return _require_link_inventory(value, key="documents")
    details = value.get("details")
    groups = details.get("collection_groups") if type(details) is dict else None
    if (
        value.get("schema_name") != "document_collection"
        or type(groups) is not list
        or not groups
    ):
        raise ValueError("source child inventory is absent")
    _document_text(value)
    for group in groups:
        if (
            type(group) is not dict
            or type(group.get("title")) is not str
            or not group["title"].strip()
            or type(group.get("body")) is not str
            or type(group.get("documents")) is not list
            or group["documents"]
        ):
            raise ValueError("source child inventory differs")
    return ()


def _require_attachment_inventory(
    value: dict,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    details = value.get("details")
    links = value.get("links")
    attachments = details.get("attachments") if type(details) is dict else None
    children = links.get("children") if type(links) is dict else None
    if any(entries is not None and type(entries) is not list for entries in (attachments, children)):
        raise ValueError("source attachment inventory differs")
    inventories = []
    if type(attachments) is list and attachments:
        inventories.append(attachments)
    if type(children) is list and children:
        inventories.append(children)
    if not inventories:
        raise ValueError("source attachment inventory is absent")
    items: dict[str, str] = {}
    for entries in inventories:
        paths = []
        for entry in entries:
            if type(entry) is not dict:
                raise ValueError("source attachment inventory differs")
            path = entry.get("url") or entry.get("base_path")
            title = entry.get("title")
            if (
                type(path) is not str
                or not (_safe_attachment_location(path)
                        or entry.get("attachment_type") == "external" and _archival_reference_location(path))
                or type(title) is not str
                or not title.strip()
            ):
                raise ValueError("source attachment identity differs")
            paths.append(path)
            items.setdefault(path, title.strip())
        if len(set(paths)) != len(paths):
            raise ValueError("source attachment inventory is incomplete")
    children = tuple(
        (path, title) for path, title in items.items() if path.startswith("/")
    )
    unsupported = tuple(
        (path, title) for path, title in items.items() if not path.startswith("/")
    )
    return children, unsupported


def _archival_reference_location(value: str) -> bool:
    """Recognise declared historical metadata without authorising its retrieval."""
    if not isinstance(value, str) or "\\" in value or any(ord(character) < 32 for character in value):
        return False
    parsed = urlsplit(value)
    prefix = "/ukgwa/timeline/"
    if (parsed.scheme != "https" or parsed.netloc != "webarchive.nationalarchives.gov.uk"
            or parsed.query or parsed.fragment or not parsed.path.startswith(prefix)):
        return False
    try:
        _api_url(parsed.path.removeprefix(prefix))
    except ValueError:
        return False
    return True


def _safe_attachment_location(value: str) -> bool:
    if value.startswith("/"):
        try:
            _api_url("https://www.gov.uk" + value)
        except ValueError:
            return False
        return True
    if any(ord(character) < 32 for character in value):
        return False
    parsed = urlsplit(value)
    path = unquote(parsed.path)
    return (
        parsed.scheme == "https"
        and parsed.netloc == "assets.publishing.service.gov.uk"
        and not parsed.query
        and not parsed.fragment
        and path.startswith("/")
        and not path.startswith("//")
        and "\\" not in path
        and not any(ord(character) < 32 for character in path)
        and all(part not in {".", ".."} for part in path.split("/"))
    )


def parse_govuk_manual_inventory(
    canonical_url: str, raw: bytes, *, retrieved_at: datetime,
) -> GovUkManualInventory:
    """Return every section declared by one current GOV.UK manual index."""

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    root = urlsplit(canonical_url).path.rstrip("/")
    try:
        _api_url(canonical_url)
    except ValueError:
        raise ValueError("source manual schema or currentness differs") from None
    if (
        type(value) is not dict
        or value.get("base_path") != root
        or value.get("locale") != "en"
        or value.get("document_type") != "manual"
        or value.get("withdrawn_notice")
    ):
        raise ValueError("source manual schema or currentness differs")
    publication = _instant(value.get("first_published_at"))
    updated = _instant(value.get("public_updated_at"))
    if publication > retrieved_at or updated > retrieved_at:
        raise ValueError("source temporal order differs")
    title = value.get("title")
    groups = value.get("details", {}).get("child_section_groups")
    if type(title) is not str or not title.strip() or type(groups) is not list or not groups:
        raise ValueError("source manual inventory is absent")
    sections = []
    for group in groups:
        if type(group) is not dict or type(group.get("title")) is not str:
            raise ValueError("source manual group differs")
        children = group.get("child_sections")
        if type(children) is not list:
            raise ValueError("source manual sections differ")
        for child in children:
            if type(child) is not dict:
                raise ValueError("source manual section differs")
            path, section_title = child.get("base_path"), child.get("title")
            if (
                type(path) is not str or not path.startswith(root + "/")
                or type(section_title) is not str or not section_title.strip()
            ):
                raise ValueError("source manual section identity differs")
            try:
                _api_url("https://www.gov.uk" + path)
            except ValueError:
                raise ValueError("source manual section identity differs") from None
            sections.append((path, section_title.strip()))
    if not sections or len({path for path, _ in sections}) != len(sections):
        raise ValueError("source manual inventory is incomplete")
    return GovUkManualInventory(
        title.strip(), publication, updated, _organisation_names(value), tuple(sections),
    )


def _organisation_names(value: dict) -> tuple[str, ...]:
    organisations = value.get("links", {}).get("organisations")
    if not organisations and value.get("document_type") in {"manual", "manual_section"}:
        organisations = value.get("details", {}).get("manual", {}).get("organisations")
    if type(organisations) is not list:
        raise ValueError("responsible publisher is absent")
    if any(type(item) is not dict or type(item.get("title")) is not str
           or not item["title"].strip() for item in organisations):
        raise ValueError("responsible publisher is absent")
    names = tuple(sorted({item["title"].strip() for item in organisations}))
    if not names:
        raise ValueError("responsible publisher is absent")
    return names


def _exclusion_signals(value: dict, body_text: str) -> tuple[str, ...]:
    """Retain explicit contrary rights signals; absence is not a legal finding."""
    details = value["details"]
    notices = " ".join(str(details.get(key, "")) for key in (
        "copyright_notice", "copyright", "licence", "license",
    ))
    text = (notices + " " + body_text).casefold()
    signals = set()
    if re.search(r"third[- ]party copyright|all rights reserved|permission.{0,30}copyright holder", text):
        signals.add("THIRD_PARTY_RIGHTS")
    if re.search(r"not (?:covered|available|licensed).{0,45}open government licen[cs]e", text):
        signals.add("NON_OGL_CONTENT")
    if details.get("personal_information") or details.get("identity_document"):
        signals.add("EXCLUDED_PERSONAL_OR_IDENTITY_CONTENT")
    return tuple(sorted(signals))


def _document_text(value: dict) -> str:
    """Read every supplied guide part, not just the first-page summary."""
    if value.get("document_type") == "guide":
        parts = value["details"]["parts"]
        if type(parts) is not list or not parts:
            raise ValueError("source guide parts are absent")
        slugs = set()
        sections = []
        for part in parts:
            if type(part) is not dict:
                raise ValueError("source guide part differs")
            slug, title = part.get("slug"), part.get("title")
            if (type(slug) is not str or not slug or slug in slugs
                    or type(title) is not str or not title.strip()):
                raise ValueError("source guide part identity differs")
            slugs.add(slug)
            sections.append(title.strip() + "\n" + _html_text(part.get("body")))
        return "\n\n".join(sections)
    return _html_text(value["details"].get("body"))


def _html_text(fragment: object, *, allow_empty: bool = False) -> str:
    if type(fragment) is not str or (not fragment.strip() and not allow_empty):
        raise ValueError("source document body is absent")
    if not fragment.strip():
        return ""
    document = html.fragment_fromstring(fragment, create_parent="div")
    if document.xpath(".//script | .//iframe | .//object"):
        raise ValueError("source body requires non-text resources")
    text = " ".join(document.text_content().split())
    if not text and not allow_empty:
        raise ValueError("source content is empty")
    return text
