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
import json
import urllib.request

from lxml import html

from newsroom.authority import HydrationRequest, ObjectAdmissionId, ObjectAdmissionRequest, UtcTimestamp, DiagnosticHistoryExpired
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical, validate_sha256_digest
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
    observation_source_id: str | None = None
    observation_member_digest: str | None = None

    def __post_init__(self) -> None:
        ObjectAdmissionId.parse(self.assessment_admission_id)
        ObjectAdmissionId.parse(self.observation_admission_id)
        validate_sha256_digest(self.assessment_blob_digest)
        validate_sha256_digest(self.observation_blob_digest)
        if self.observation_source_id is not None or self.observation_member_digest is not None:
            if self.observation_source_id not in SOURCE_URLS:
                raise ValueError("rights observation source selector differs")
            validate_sha256_digest(self.observation_member_digest)


def _rights_snapshot_values(
    *, source_id: str, definition_url: str,
    assessment: PublicationRightsAssessment, observed_at: str,
    reason: str, observations: tuple[tuple[str, str, str, str], ...],
    govuk_semantic_evidence: dict | None = None,
) -> tuple[dict, dict]:
    if govuk_semantic_evidence is not None:
        from .govuk_rights import POLICY_DIGEST as GOVUK_POLICY, _govuk_semantic_evidence
        expected = _govuk_semantic_evidence(source_id=source_id, definition_url=definition_url)
        if (govuk_semantic_evidence != expected or type(assessment) is not PublicationRightsAssessment
                or assessment.decision != "PERMITTED" or assessment.permitted_use != "PUBLICATION_EVIDENCE"
                or assessment.policy_digest != GOVUK_POLICY
                or assessment.evidence_digest != digest_canonical(expected)):
            raise ValueError("GOV.UK semantic rights binding differs")
    observation_value = {
        "schema": "hermes-native-rights-observation-v1",
        "source_id": source_id, "definition_url": definition_url,
        "observed_at": observed_at, "reason": reason,
        "observations": observations,
    }
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
    return observation_value, assessment_value


def _retain_assessment(*, objects, proof, value, reobservation_epoch=None):
    assessment_bytes = canonical_json_bytes(value)
    original = ObjectAdmissionRequest("evidence.source", f"native-rights-assessment:{value['record_id']}")
    fallback = None
    if reobservation_epoch is not None:
        validate_sha256_digest(reobservation_epoch)
        if value["decision"] == "HOLD":
            fallback = ObjectAdmissionRequest("evidence.source",
                f"{original.idempotency_key}:native-reobservation:{reobservation_epoch}")
    # Recover the current store's exact reobservation before touching the expired
    # original again. Current auth/rights and the bytes are still rechecked.
    recovered = None if fallback is None else objects.committed_admission(fallback, proof=proof)
    if recovered is not None:
        retained = recovered.admission
    else:
        try:
            retained = objects.admit(original, assessment_bytes, proof=proof).admission
        except DiagnosticHistoryExpired:
            if fallback is None:
                raise
            retained = objects.admit(fallback, assessment_bytes, proof=proof).admission
    hydrated = objects.rehydrate(
        HydrationRequest(retained.admission_id, "evidence.source"), proof=proof,
    )
    if hydrated.data != assessment_bytes:
        raise ValueError("retained source rights snapshot differs")
    return retained


def retain_rights_snapshot(
    *, objects, proof, source_id: str, definition_url: str,
    assessment: PublicationRightsAssessment, observed_at: str,
    reason: str, observations: tuple[tuple[str, str, str, str], ...],
    govuk_semantic_evidence: dict | None = None,
) -> RightsSnapshotReference:
    """Retain an individual observation, including the legacy read contract."""
    observation_value, assessment_value = _rights_snapshot_values(
        source_id=source_id, definition_url=definition_url, assessment=assessment,
        observed_at=observed_at, reason=reason, observations=observations,
        govuk_semantic_evidence=govuk_semantic_evidence,
    )
    observation_bytes = canonical_json_bytes(observation_value)
    observation = objects.admit(ObjectAdmissionRequest(
        "evidence.source", f"native-rights-observation:{digest_bytes(observation_bytes)}",
    ), observation_bytes, proof=proof).admission
    retained = _retain_assessment(objects=objects, proof=proof, value=assessment_value)
    hydrated = objects.rehydrate(
        HydrationRequest(observation.admission_id, "evidence.source"), proof=proof,
    )
    if hydrated.data != observation_bytes:
        raise ValueError("retained source rights snapshot differs")
    return RightsSnapshotReference(
        str(retained.admission_id), retained.blob.blob_digest,
        str(observation.admission_id), observation.blob.blob_digest,
    )


_OBSERVATION_SCHEMA = "hermes-native-rights-observation-v1"
_BUNDLE_SCHEMA = "hermes-native-rights-observation-bundle-v1"


def _validate_observation_member(value, *, source_id, definition_url):
    if (type(value) is not dict or set(value) != {
            "schema", "source_id", "definition_url", "observed_at", "reason", "observations",
        } or value["schema"] != _OBSERVATION_SCHEMA
        or value["source_id"] != source_id or value["definition_url"] != definition_url
        or type(value["reason"]) is not str or not value["reason"]
        or type(value["observations"]) not in (list, tuple)):
        raise ValueError("rights observation member binding differs")
    UtcTimestamp.parse(value["observed_at"])
    urls = []
    for item in value["observations"]:
        if (type(item) not in (list, tuple) or len(item) != 4
            or any(type(field) is not str for field in item)
            or not item[0]):
            raise ValueError("rights observation raw reference differs")
        validate_sha256_digest(item[1])
        ObjectAdmissionId.parse(item[2])
        urls.append(item[0])
    if len(urls) != len(set(urls)):
        raise ValueError("rights observation raw inventory differs")


def _bundle_members(value):
    if (type(value) is not dict or set(value) != {"schema", "members"}
        or value["schema"] != _BUNDLE_SCHEMA or type(value["members"]) is not list):
        raise ValueError("rights observation bundle shape differs")
    members = {}
    for entry in value["members"]:
        if (type(entry) is not dict or set(entry) != {"source_id", "member_digest", "member"}
            or type(entry["source_id"]) is not str
            or entry["source_id"] not in SOURCE_URLS or entry["source_id"] in members):
            raise ValueError("rights observation bundle inventory differs")
        source_id = entry["source_id"]
        _validate_observation_member(entry["member"], source_id=source_id,
                                     definition_url=SOURCE_URLS[source_id])
        if digest_bytes(canonical_json_bytes(entry["member"])) != entry["member_digest"]:
            raise ValueError("rights observation member digest differs")
        members[source_id] = entry
    if set(members) != set(SOURCE_URLS) or list(members) != sorted(members):
        raise ValueError("rights observation bundle inventory differs")
    return members


def retain_rights_snapshot_bundle(*, objects, proof, snapshots, stop_check, reobservation_epoch=None):
    """Retain every fresh source fact in one immutable portfolio envelope."""
    if type(snapshots) is not dict or set(snapshots) != set(SOURCE_URLS):
        raise ValueError("rights observation bundle inventory differs")
    values = {source: _rights_snapshot_values(source_id=source, **snapshots[source])
              for source in sorted(snapshots)}
    bundle = {"schema": _BUNDLE_SCHEMA, "members": [
        {"source_id": source, "member_digest": digest_bytes(canonical_json_bytes(observation)),
         "member": observation}
        for source, (observation, _) in values.items()
    ]}
    members = _bundle_members(bundle)
    stop_check()
    raw = canonical_json_bytes(bundle)
    admission = objects.admit(ObjectAdmissionRequest(
        "evidence.source", f"native-rights-observation-bundle:{digest_bytes(raw)}",
    ), raw, proof=proof).admission
    hydrated = objects.rehydrate(
        HydrationRequest(admission.admission_id, "evidence.source"), proof=proof,
    )
    if hydrated.data != raw:
        raise ValueError("retained rights observation bundle differs")
    result = {}
    for source, (_, assessment) in values.items():
        stop_check()
        retained = _retain_assessment(objects=objects, proof=proof, value=assessment,
                                      reobservation_epoch=reobservation_epoch)
        result[source] = RightsSnapshotReference(
            str(retained.admission_id), retained.blob.blob_digest,
            str(admission.admission_id), admission.blob.blob_digest,
            source, members[source]["member_digest"],
        )
    return result


def read_rights_observation(*, objects, proof, reference, source_id, definition_url):
    """Recheck current governed authority and the exact selected fresh fact."""
    if type(reference) is not RightsSnapshotReference:
        raise ValueError("rights observation reference differs")
    raw = objects.rehydrate(HydrationRequest(
        ObjectAdmissionId.parse(reference.observation_admission_id), "evidence.source",
    ), proof=proof).data
    if digest_bytes(raw) != reference.observation_blob_digest:
        raise ValueError("rights observation blob digest differs")
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("rights observation duplicate field")
            value[key] = item
        return value
    try:
        value = json.loads(raw, object_pairs_hook=unique_object)
        if canonical_json_bytes(value) != raw:
            raise ValueError("rights observation bytes are not canonical")
        if type(value) is dict and value.get("schema") == _BUNDLE_SCHEMA:
            members = _bundle_members(value)
            if (reference.observation_source_id != source_id
                or source_id not in members
                or members[source_id]["member_digest"] != reference.observation_member_digest):
                raise ValueError("rights observation selector differs")
            value = members[source_id]["member"]
        elif reference.observation_source_id is not None:
            raise ValueError("rights observation selector requires a bundle")
        _validate_observation_member(value, source_id=source_id, definition_url=definition_url)
    except (TypeError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("rights observation bytes differ") from exc
    return value


def validate_rights_observation_selector(value, *, source_id=None, definition_url=None):
    """Keep legacy packets valid; bind every new packet's member selector."""
    fields = {"observation_source_id", "observation_member_digest"}
    present = fields.intersection(value)
    if not present:
        return
    if (present != fields or type(value["observation_source_id"]) is not str
        or value["observation_source_id"] not in SOURCE_URLS
        or value["observation_source_id"] != value.get("source_id")
        or (source_id is not None and value["observation_source_id"] != source_id)
        or (definition_url is not None and value.get("source_url") != definition_url)):
        raise ValueError("rights observation selector differs")
    validate_sha256_digest(value["observation_member_digest"])


def require_rights_assessment(*, objects, proof, reference, assessment, observation):
    """A current rights packet must not reference revoked or rebound evidence."""
    raw = objects.rehydrate(HydrationRequest(
        ObjectAdmissionId.parse(reference.assessment_admission_id), "evidence.source",
    ), proof=proof).data
    if digest_bytes(raw) != reference.assessment_blob_digest:
        raise ValueError("rights assessment blob digest differs")
    try:
        value = json.loads(raw)
        semantic = None
        if type(value) is dict and value.get("schema") == "hermes-native-rights-assessment-v2":
            from .govuk_rights import _govuk_semantic_evidence
            semantic = _govuk_semantic_evidence(
                source_id=observation["source_id"], definition_url=observation["definition_url"],
            )
        _, expected = _rights_snapshot_values(
            source_id=observation["source_id"], definition_url=observation["definition_url"],
            assessment=assessment, observed_at=observation["observed_at"], reason=observation["reason"],
            observations=observation["observations"], govuk_semantic_evidence=semantic,
        )
        if canonical_json_bytes(expected) != raw:
            raise ValueError("rights assessment is not the current source assessment")
    except (TypeError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("rights assessment bytes differ") from exc


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
