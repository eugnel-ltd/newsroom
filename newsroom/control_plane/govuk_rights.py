"""Observed GOV.UK text-reuse policy for the autonomous private evidence route.

No fixture reviews or reviewer quorum are minted. The two official licence
pages were observed over HTTPS on 8 September 2026. A changed substantive terms
text is a source-local HOLD, not an implicit new licence or a daemon-wide stop.
"""

from __future__ import annotations

import re
import ssl
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ContextManager
from urllib.parse import urlsplit

from lxml import html

from newsroom.authority import AuthenticationProof, GovernedObjects, HydrationRequest, ObjectAdmissionRequest
from newsroom.authority.canonical import digest_bytes, digest_canonical
from newsroom.authority.types import ObjectAdmissionId

from .govuk_evidence import MAX_BODY_BYTES, TIMEOUT_SECONDS, _NoRedirect
from .native_evidence import NativeEvidenceHold, PublicationRightsAssessment
from .veto import VetoError

REUSE_URL = "https://www.gov.uk/help/reuse-govuk-content"
LICENCE_URL = "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"
REVIEWED_TEXT = {
    REUSE_URL: "sha256:b58998ef4b7ffc6754b6780bbec15f4b909f7662a47186329c175a86ea37d9ea",
    LICENCE_URL: "sha256:c9f8aa884c89702fc694ea97c91d8db796e6f2a966d93f08e05731c7ab088c61",
}
ATTRIBUTION = "Contains public sector information licensed under the Open Government Licence v3.0."
POLICY_VERSION = "hermes-govuk-text-ogl-v1"
POLICY_DIGEST = digest_canonical({
    "version": POLICY_VERSION, "reviewed_terms": REVIEWED_TEXT,
    "use": "PUBLICATION_EVIDENCE", "scope": "GOVUK_PUBLISHED_TEXT_ONLY",
    "required_attribution": ATTRIBUTION,
    "exclusions": ["personal_data", "third_party_rights", "logos_and_insignia",
                   "unpublished_information", "identity_documents", "endorsement"],
    "public_exposure": False,
})


def _govuk_semantic_evidence(*, source_id: str, definition_url: str) -> dict:
    if (source_id not in {"UK-01", "UK-02", "UK-03", "UK-05"}
            or urlsplit(definition_url).scheme != "https"
            or urlsplit(definition_url).netloc != "www.gov.uk"):
        raise ValueError("GOV.UK semantic rights source scope differs")
    return {
        "schema": "hermes-govuk-semantic-rights-v2",
        "source_id": source_id, "definition_url": definition_url,
        "reviewed_text": dict(REVIEWED_TEXT), "policy_digest": POLICY_DIGEST,
    }


@dataclass(frozen=True, slots=True)
class GovUkLicenceEvidence:
    admission_ids: tuple[ObjectAdmissionId, ObjectAdmissionId]
    raw_digests: tuple[str, str]
    observed_at: str
    policy_digest: str

    def require_retained(self, *, objects: GovernedObjects, proof: AuthenticationProof) -> None:
        """Validate actual retained terms, not merely a caller's policy string."""
        if self.policy_digest != POLICY_DIGEST or len(self.admission_ids) != 2 or len(self.raw_digests) != 2:
            raise NativeEvidenceHold("GOVUK_LICENCE_BINDING_HOLD", "UK-GOVUK")
        for url, admission_id, raw_digest in zip(
            (REUSE_URL, LICENCE_URL), self.admission_ids, self.raw_digests, strict=True,
        ):
            retained = objects.hydrate(HydrationRequest(admission_id, "evidence.source"), proof=proof)
            if digest_bytes(retained.data) != raw_digest or licence_text_digest(retained.data) != REVIEWED_TEXT[url]:
                raise NativeEvidenceHold("GOVUK_LICENCE_BINDING_HOLD", "UK-GOVUK")

    def for_source(self, *, source_id: str, definition_url: str) -> PublicationRightsAssessment:
        permitted = (
            self.policy_digest == POLICY_DIGEST
            and source_id in {"UK-01", "UK-02", "UK-03", "UK-05"}
            and urlsplit(definition_url).scheme == "https"
            and urlsplit(definition_url).netloc == "www.gov.uk"
        )
        return PublicationRightsAssessment.create(
            decision="PERMITTED" if permitted else "HOLD",
            permitted_use="PUBLICATION_EVIDENCE", policy_digest=POLICY_DIGEST,
            evidence_digest=digest_canonical(_govuk_semantic_evidence(
                source_id=source_id, definition_url=definition_url,
            ) if permitted else {
                "source_id": source_id, "definition_url": definition_url,
                "admission_ids": [str(item) for item in self.admission_ids],
                "raw_digests": self.raw_digests,
                "reviewed_text": REVIEWED_TEXT, "policy_digest": POLICY_DIGEST,
            }),
        )


def licence_text_digest(raw: bytes) -> str:
    tree = html.fromstring(raw)
    roots = tree.xpath("//main") or [tree]
    for root in roots:
        for node in (*root.find_class("gem-c-published-dates"), *root.find_class("gem-c-metadata")):
            # Canonicalise the display to the already reviewed text. A markup
            # repair is not a new permission contract or private destination.
            match = re.fullmatch(
                r"(?:Updates to this page )?Last updated:? (\d{1,2} [A-Z][a-z]+ \d{4})",
                " ".join(" ".join(node.itertext()).split()),
            )
            if match:
                node.clear(keep_tail=True)
                node.text = f"Updates to this page Last updated {match[1]}"
    text = " ".join(" ".join(root.text_content().split()) for root in roots)
    return digest_bytes(text.encode("utf-8"))


def retain_current_govuk_licence(
    *, objects: GovernedObjects, proof: AuthenticationProof,
    dispatch_fence: Callable[[], ContextManager[None]],
    clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
) -> GovUkLicenceEvidence:
    """Fetch one current observation and retain exact raw terms in governed CAS.

    This is acquisition permission for the reviewed text use, not blanket
    publication approval. The post-acquisition assessor must apply exclusions;
    the final private article/feed card must carry attribution and source links.
    """
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    observations = []
    for url in (REUSE_URL, LICENCE_URL):
        request = urllib.request.Request(url, method="GET", headers={
            "User-Agent": "Newsroom-Hermes-Rights-Review/1.0", "Accept-Encoding": "identity",
        })
        try:
            with dispatch_fence():
                with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
                    raw = response.read(MAX_BODY_BYTES + 1)
                    if response.status != 200 or response.geturl() != url:
                        raise ValueError("licence response identity differs")
            if not raw or len(raw) > MAX_BODY_BYTES:
                raise ValueError("licence response length differs")
            if licence_text_digest(raw) != REVIEWED_TEXT[url]:
                raise ValueError("substantive licence terms changed")
        except VetoError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            raise NativeEvidenceHold("GOVUK_LICENCE_REVIEW_HOLD", "UK-GOVUK") from None
        observations.append(raw)
    # Check every response before the first authority write; no partial licence.
    admissions = tuple(
        objects.admit(
            ObjectAdmissionRequest("evidence.source", f"govuk-licence:{digest_bytes(raw)}"),
            raw, proof=proof,
        ).admission for raw in observations
    )
    return GovUkLicenceEvidence(
        tuple(item.admission_id for item in admissions),
        tuple(item.blob.blob_digest for item in admissions),
        clock().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        POLICY_DIGEST,
    )
