"""Observed source terms for the fixed private Hermes portfolio.

This separates an observed reuse restriction from a missing adapter. No absent
permission is manufactured from owner approval or from an accessible feed.
The approved GOV.UK path retains its own exact OGL evidence unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from typing import ContextManager
from threading import RLock
import urllib.request

from lxml import html

from newsroom.authority import HydrationRequest, ObjectAdmissionId, ObjectAdmissionRequest
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.increment9.proving import SOURCE_URLS

from .govuk_evidence import _NoRedirect
from .govuk_rights import GovUkLicenceEvidence
from .native_evidence import PublicationRightsAssessment
from .veto import VetoError

VERSION = "hermes-observed-portfolio-rights-v1"
SOURCE_LICENCE_POLICY = (
    ("data.weather.gov.hk", "Weather information provided by the Hong Kong Observatory.",
     "https://data.gov.hk/en/terms-and-conditions"),
    ("www.metoffice.gov.uk", "Weather warnings provided by the Met Office.",
     "https://www.metoffice.gov.uk/policies/tandc"),
    ("weather.metoffice.gov.uk", "Weather warnings provided by the Met Office.",
     "https://www.metoffice.gov.uk/policies/tandc"),
)
# Reviewed from these exact official pages on 8 September 2026. Hashes cover
# substantive visible terms, excluding scripts/styles and site navigation.
TERMS = {
    "UK-10": (("https://www.metoffice.gov.uk/policies/tandc", "sha256:c746137861806d4e5f2aa77732df3969fa2c362081137be84a2362af1636dc33"),
              ("https://weather.metoffice.gov.uk/guides/rss", "sha256:cae7a443f0e2325eb508a3d22935c07f9f1644377a2b1b0f582bfda78713b793")),
    "HK-01": (("https://www.news.gov.hk/eng/about/", "sha256:d11337e4040770ec10cfa0a539072209a4d1ff244dda2824c143789b5241b6a0"),),
    "HK-02": (("https://data.gov.hk/en/terms-and-conditions", "sha256:6306fd063cbeb36d77e1a1275ab62421eabdc6ee0444083421f442d0514f80b4"),),
    "HK-04": (("https://www.edb.gov.hk/en/important-notices/index.html", "sha256:cc522228dbb3dc35d9f29e0f2b0ce5d3836e104be9b61e2ba8205a10e6cebfc7"),),
    "RAD-01": (("https://www.rthk.hk/copyright/index_e.html", "sha256:7cd591e4d0b4a8ded692b4f625d1778f113cb02e6942a020b4aa081a7a47d0a9"),),
    "RAD-02": (("https://www.bbc.co.uk/usingthebbc/terms-of-use/", "sha256:632a0e1e1f6fcc0506b8bf2f60ecd32baf1f2f73e07468d2f3d21253506728b6"),),
}
# The route neither assumes commercial permission nor treats private access as
# a copyright exception. These restrictions remain source-local, not a daemon
# stop. A separately retained licence can support a future policy revision.
RESTRICTIONS = {
    "HK-01": "MEDIA_REUSE_PERMISSION_SCOPE_NOT_ESTABLISHED",
    "HK-04": "NON_COMMERCIAL_INTERNAL_USE_ONLY",
    "RAD-01": "AUTOMATED_REUSE_PERMISSION_NOT_RETAINED",
    "RAD-02": "COMPUTER_ANALYSIS_PERMISSION_NOT_RETAINED",
}
POLICY_DIGEST = digest_canonical({
    "version": VERSION, "terms": TERMS, "restrictions": RESTRICTIONS,
    "permitted_scope": {"UK-10": "ATTRIBUTED_RSS_WITH_DIRECT_LINK", "HK-02": "ATTRIBUTED_OPEN_WARNING_DATA"},
    "source_licence_policy": SOURCE_LICENCE_POLICY,
    "public_exposure": False,
})


def terms_text_digest(source_id: str, raw: bytes) -> str:
    tree = html.fromstring(raw.decode("utf-8"))
    for element in tree.xpath("//script|//style|//template"):
        element.drop_tree()
    roots = tree.xpath(
        "//div[contains(concat(' ',normalize-space(@class),' '),' inner_page_content_container ')]"
        if source_id == "HK-04" else "//main"
    )
    if len(roots) != 1:
        raise ValueError("source terms content boundary differs")
    return digest_bytes(" ".join(roots[0].text_content().split()).encode())


def _fetch_terms(url: str) -> bytes:
    if url not in {url for terms in TERMS.values() for url, _ in terms}:
        raise ValueError("source terms endpoint differs")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(urllib.request.Request(url, headers={
        "User-Agent": "Newsroom-Rights-Review/1.0", "Accept-Encoding": "identity",
    }), timeout=20) as response:
        raw = response.read(1_048_577)
        if response.status != 200 or response.geturl() != url or not raw or len(raw) > 1_048_576:
            raise ValueError("source terms response differs")
    return raw


@dataclass(frozen=True, slots=True)
class SourceTermsEvidence:
    source_id: str
    observed_at: str
    reason: str
    # Exact URL, raw digest, governed admission and authenticated access ID.
    observations: tuple[tuple[str, str, str, str], ...]

    def __post_init__(self) -> None:
        if self.source_id not in TERMS or datetime.fromisoformat(self.observed_at).tzinfo is None:
            raise ValueError("source terms observation identity differs")
        expected = {url for url, _ in TERMS[self.source_id]}
        seen = [item[0] for item in self.observations]
        if len(seen) != len(set(seen)) or not set(seen).issubset(expected):
            raise ValueError("source terms observation inventory differs")
        if self.reason == "REVIEWED_REUSE_PERMITTED" and set(seen) != expected:
            raise ValueError("permitted source terms inventory is incomplete")
        if self.source_id in RESTRICTIONS and self.reason == "REVIEWED_REUSE_PERMITTED":
            raise ValueError("observed source restriction has no permission override")
        for _url, digest, admission, access in self.observations:
            from newsroom.authority.canonical import validate_sha256_digest
            validate_sha256_digest(digest)
            ObjectAdmissionId.parse(admission)
            if not access:
                raise ValueError("source terms access identity is absent")

    @property
    def digest(self) -> str:
        return digest_canonical({
            "policy": POLICY_DIGEST, "source_id": self.source_id,
            "reason": self.reason,
            "observations": tuple(
                (url, digest, admission) for url, digest, admission, _access in self.observations
            ),
        })


@dataclass(frozen=True, slots=True)
class RightsSnapshotReference:
    assessment_admission_id: str
    assessment_blob_digest: str
    observation_admission_id: str
    observation_blob_digest: str

    def __post_init__(self) -> None:
        from newsroom.authority.canonical import validate_sha256_digest
        ObjectAdmissionId.parse(self.assessment_admission_id)
        ObjectAdmissionId.parse(self.observation_admission_id)
        validate_sha256_digest(self.assessment_blob_digest)
        validate_sha256_digest(self.observation_blob_digest)


def retain_rights_snapshot(
    *, objects, proof, source_id: str, definition_url: str,
    assessment: PublicationRightsAssessment, observed_at: str,
    reason: str, observations: tuple[tuple[str, str, str, str], ...],
    govuk_semantic_evidence: dict | None = None,
) -> RightsSnapshotReference:
    """Retain the current observation and its stable semantic assessment."""
    if govuk_semantic_evidence is not None:
        from .govuk_rights import POLICY_DIGEST as GOVUK_POLICY, _govuk_semantic_evidence
        expected = _govuk_semantic_evidence(source_id=source_id, definition_url=definition_url)
        if (govuk_semantic_evidence != expected or type(assessment) is not PublicationRightsAssessment
                or assessment.decision != "PERMITTED" or assessment.permitted_use != "PUBLICATION_EVIDENCE"
                or assessment.policy_digest != GOVUK_POLICY
                or assessment.evidence_digest != digest_canonical(expected)):
            raise ValueError("GOV.UK semantic rights binding differs")
    observation_bytes = canonical_json_bytes({
        "schema": "hermes-native-rights-observation-v1",
        "source_id": source_id, "definition_url": definition_url,
        "observed_at": observed_at, "reason": reason,
        "observations": observations,
    })
    observation = objects.admit(ObjectAdmissionRequest(
        "evidence.source", f"native-rights-observation:{digest_bytes(observation_bytes)}",
    ), observation_bytes, proof=proof).admission
    assessment_value = {
        "schema": "hermes-native-rights-assessment-v1",
        "source_id": source_id, "definition_url": definition_url,
        "record_id": assessment.record_id, "decision": assessment.decision,
        "permitted_use": assessment.permitted_use,
        "policy_digest": assessment.policy_digest,
        "evidence_digest": assessment.evidence_digest,
        # Access/observation times are audit facts above, not semantic identity.
        "evidence": tuple(
            (url, digest, admission) for url, digest, admission, _access in observations
        ),
    }
    if govuk_semantic_evidence is not None:
        assessment_value["schema"] = "hermes-native-rights-assessment-v2"
        del assessment_value["evidence"]
        assessment_value["semantic_evidence"] = govuk_semantic_evidence
    assessment_bytes = canonical_json_bytes(assessment_value)
    retained = objects.admit(ObjectAdmissionRequest(
        "evidence.source", f"native-rights-assessment:{assessment.record_id}",
    ), assessment_bytes, proof=proof).admission
    for admission, raw in ((observation, observation_bytes), (retained, assessment_bytes)):
        hydrated = objects.rehydrate(
            HydrationRequest(admission.admission_id, "evidence.source"), proof=proof,
        )
        if hydrated.data != raw:
            raise ValueError("retained source rights snapshot differs")
    return RightsSnapshotReference(
        str(retained.admission_id), retained.blob.blob_digest,
        str(observation.admission_id), observation.blob.blob_digest,
    )


class NativePortfolioRights:
    def __init__(
        self, govuk: GovUkLicenceEvidence | None,
        evidence: dict[str, SourceTermsEvidence], *,
        refresh_current: Callable[
            [], tuple[
                GovUkLicenceEvidence | None, str, dict[str, SourceTermsEvidence],
                dict[str, RightsSnapshotReference],
            ]
        ] | None = None,
        govuk_reason: str = "GOVUK_LICENCE_UNOBSERVED",
        snapshots: dict[str, RightsSnapshotReference] | None = None,
    ):
        if (govuk is not None and type(govuk) is not GovUkLicenceEvidence) or any(
            type(value) is not SourceTermsEvidence or source_id != value.source_id
            for source_id, value in evidence.items()
        ) or not isinstance(govuk_reason, str) or not govuk_reason:
            raise ValueError("portfolio source terms binding differs")
        if refresh_current is not None and not callable(refresh_current):
            raise ValueError("portfolio source terms refresh differs")
        self.govuk, self.evidence = govuk, dict(evidence)
        self._snapshots = dict(snapshots or {})
        self._govuk_reason = "REVIEWED_REUSE_PERMITTED" if govuk else govuk_reason
        self._refresh_current = refresh_current
        self._lock = RLock()

    def refresh(self) -> None:
        """Replace the complete current snapshot once per bounded iteration."""
        if self._refresh_current is None:
            return
        govuk, govuk_reason, evidence, snapshots = self._refresh_current()
        if (
            (govuk is not None and type(govuk) is not GovUkLicenceEvidence)
            or not isinstance(govuk_reason, str) or not govuk_reason
            or set(evidence) != set(TERMS)
            or any(
                type(value) is not SourceTermsEvidence or source_id != value.source_id
                for source_id, value in evidence.items()
            )
            or set(snapshots) != set(SOURCE_URLS)
            or any(type(value) is not RightsSnapshotReference for value in snapshots.values())
        ):
            raise ValueError("refreshed portfolio source terms binding differs")
        with self._lock:
            self.govuk = govuk
            self._govuk_reason = (
                "REVIEWED_REUSE_PERMITTED" if govuk is not None else govuk_reason
            )
            self.evidence = dict(evidence)
            self._snapshots = dict(snapshots)

    def snapshot_for(self, source_id: str) -> RightsSnapshotReference | None:
        with self._lock:
            return self._snapshots.get(source_id)

    def for_source(self, *, source_id: str, definition_url: str) -> PublicationRightsAssessment:
        if source_id not in TERMS:
            with self._lock:
                govuk, reason = self.govuk, self._govuk_reason
            if govuk is not None:
                return govuk.for_source(source_id=source_id, definition_url=definition_url)
            return PublicationRightsAssessment.create(
                decision="HOLD", permitted_use="PUBLICATION_EVIDENCE",
                policy_digest=POLICY_DIGEST,
                evidence_digest=digest_canonical({
                    "source_id": source_id, "definition_url": definition_url,
                    "state": reason,
                }),
            )
        with self._lock:
            evidence = self.evidence.get(source_id)
        permitted = (evidence is not None and evidence.reason == "REVIEWED_REUSE_PERMITTED"
                     and definition_url == SOURCE_URLS[source_id])
        return PublicationRightsAssessment.create(
            decision="PERMITTED" if permitted else "HOLD", permitted_use="PUBLICATION_EVIDENCE",
            policy_digest=POLICY_DIGEST,
            evidence_digest=evidence.digest if evidence else digest_canonical({"source": source_id, "state": "UNOBSERVED"}),
        )

    def reason_for(self, source_id: str) -> str:
        if source_id not in TERMS:
            with self._lock:
                return self._govuk_reason
        with self._lock:
            evidence = self.evidence.get(source_id)
        return "SOURCE_TERMS_UNOBSERVED" if evidence is None else evidence.reason

    def observations_for(self, source_id: str):
        if source_id not in TERMS:
            return ()
        with self._lock:
            evidence = self.evidence.get(source_id)
        return () if evidence is None else evidence.observations

    def require_retained(self, *, objects, proof) -> None:
        with self._lock:
            govuk, evidence_snapshot = self.govuk, dict(self.evidence)
        if govuk is not None:
            govuk.require_retained(objects=objects, proof=proof)
        for source_id, evidence in evidence_snapshot.items():
            for url, digest, admission_id, _access in evidence.observations:
                retained = objects.rehydrate(HydrationRequest(ObjectAdmissionId.parse(admission_id), "evidence.source"), proof=proof)
                if digest_bytes(retained.data) != digest:
                    raise ValueError("retained source terms bytes differ")
                if evidence.reason == "REVIEWED_REUSE_PERMITTED" and terms_text_digest(source_id, retained.data) != dict(TERMS[source_id]).get(url):
                    raise ValueError("retained permitted terms differ")


def observe_portfolio_terms(*, objects, proof, stop_check,
                            stop_fence: Callable[[], ContextManager[None]], fetch=_fetch_terms,
                            clock=lambda: datetime.now(tz=UTC)) -> dict[str, SourceTermsEvidence]:
    """One bounded parallel observation, then serial governed retention."""
    def observe(source_id):
        bodies, reason = [], RESTRICTIONS.get(source_id, "REVIEWED_REUSE_PERMITTED")
        for url, expected in TERMS[source_id]:
            try:
                raw = fetch(url)
                bodies.append((url, raw))
                if terms_text_digest(source_id, raw) != expected:
                    reason = "SOURCE_TERMS_CHANGED"
            except VetoError:
                raise
            except Exception:
                reason = "SOURCE_TERMS_UNAVAILABLE"
        return source_id, bodies, reason, clock().astimezone(UTC).isoformat()
    stop_check()
    # One stable owner-stop decision covers the complete bounded network phase.
    # Governed-object writes remain outside the fence and are serial below.
    with stop_fence():
        with ThreadPoolExecutor(max_workers=len(TERMS)) as pool:
            results = tuple(pool.map(observe, TERMS))
    evidence = {}
    for source_id, bodies, reason, observed_at in results:
        stop_check()
        observations = []
        for url, raw in bodies:
            digest = digest_bytes(raw)
            admission = objects.admit(ObjectAdmissionRequest("evidence.source", f"native-source-terms:{source_id}:{digest}"), raw, proof=proof).admission
            access = objects.rehydrate(HydrationRequest(admission.admission_id, "evidence.source"), proof=proof).decision
            observations.append((url, digest, str(admission.admission_id), str(access.access_decision_id)))
        evidence[source_id] = SourceTermsEvidence(source_id, observed_at, reason, tuple(observations))
    return evidence
