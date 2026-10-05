"""Independent source acquisition for native publication evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Literal
from urllib.parse import urlsplit

from newsroom.authority import AuthenticationProof, GovernedObjects, ObjectAdmissionId, ObjectAdmissionRequest, UtcTimestamp
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical, validate_sha256_digest
from newsroom.control_plane.corpus import CorpusIngestUnit
from newsroom.control_plane.evidence import (
    EvidencePackage,
    GovernedClaimEvidence,
    QualificationEvidence,
    validate_governed_evidence_records,
)
from newsroom.increment6.candidates import StoryCandidateReadPort
from newsroom.increment6.candidates import StoryCandidateVersion
from newsroom.increment10.editorial import (
    EditorialPolicyDecision,
    SourceCurrentness,
    SourceIntegrity,
)
from newsroom.increment10.evidence import (
    GovernedEvidencePackage,
    GovernedEvidencePackages,
)
from newsroom.sources.record_models import SourceDefinitionVersion
from newsroom.sources.types import SourceLifecycleStage

_GATES = ("CLAIM_TRACEABILITY", "EVIDENCE_SUFFICIENCY", "SOURCE_AUTHORITY")
_INTEGRITY_CHECKS = (
    "ACCESS_COMPLETE",
    "ENCODING_VALID",
    "EXTRACTION_COMPLETE",
    "NOT_PAYWALL_FRAGMENT",
    "NOT_TRUNCATED",
    "VERSION_UNAMBIGUOUS",
)


class NativeEvidenceError(ValueError):
    """Raised when independent evidence bindings are malformed."""


class NativeEvidenceHold(NativeEvidenceError):
    """Typed non-publication outcome for incomplete independent evidence."""

    def __init__(self, reason_code: str, source_id: str) -> None:
        super().__init__(f"{reason_code}:{source_id}")
        self.reason_code = reason_code
        self.source_id = source_id


@dataclass(frozen=True, slots=True)
class EvidenceAcquisitionRequest:
    source_id: str
    source_definition_id: str
    source_definition_version_id: str
    source_definition_version_digest: str
    source_revision_id: str
    canonical_url: str
    transport_policy_digest: str

    @property
    def digest(self) -> str:
        return digest_bytes(canonical_json_bytes(self.__dict_value()))

    def __dict_value(self) -> dict[str, str]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class AcquiredEvidence:
    request_digest: str
    outcome: Literal["COMPLETE", "UNAVAILABLE", "AMBIGUOUS", "INCOMPLETE"]
    canonical_url: str
    body: bytes
    body_digest: str
    publisher: str
    responsible_body: str
    source_type: str
    publication_time: str
    source_updated_time: str
    retrieval_time: str
    geography: str
    language: str
    transport_evidence_digest: str
    receipt_digest: str
    currentness_basis: str = ""
    rights_eligibility_digest: str = ""
    licence_attribution: str = ""
    exclusion_signals: tuple[str, ...] = ()
    text_only: bool = False
    source_observed_time: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.text_only) is not bool
            or self.currentness_basis not in {
                "", "AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT",
                "RETAINED_AUTHORITATIVE_COMPLETED_EVENT",
            }
            or type(self.source_observed_time) is not str
            or type(self.exclusion_signals) is not tuple
            or any(type(item) is not str or not item for item in self.exclusion_signals)
            or tuple(sorted(set(self.exclusion_signals))) != self.exclusion_signals
        ):
            raise NativeEvidenceError("acquired source policy facts differ")
        if self.currentness_basis == "RETAINED_AUTHORITATIVE_COMPLETED_EVENT":
            if not self.source_observed_time or not (
                UtcTimestamp.parse(self.source_updated_time).value
                <= UtcTimestamp.parse(self.source_observed_time).value
                <= UtcTimestamp.parse(self.retrieval_time).value
            ):
                raise NativeEvidenceError("retained source observation time differs")
        if self.rights_eligibility_digest:
            validate_sha256_digest(self.rights_eligibility_digest)
        if type(self.body) is not bytes or self.body_digest != digest_bytes(self.body):
            raise NativeEvidenceError("acquired source body differs")
        if self.receipt_digest != digest_bytes(self.receipt_bytes):
            raise NativeEvidenceError("acquisition receipt differs")

    @property
    def receipt_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in {"body", "receipt_digest"}
                and not (name == "source_observed_time" and not self.source_observed_time)
            }
        )

    @classmethod
    def create(cls, **values: object) -> AcquiredEvidence:
        values = {
            "currentness_basis": "", "rights_eligibility_digest": "",
            "licence_attribution": "", "exclusion_signals": (), "text_only": False,
            **values,
        }
        value = {name: item for name, item in values.items() if name != "body"
                 and not (name == "source_observed_time" and not item)}
        return cls(
            **values,
            receipt_digest=digest_bytes(canonical_json_bytes(value)),
        )


class EvidenceTransport:
    """Provider-neutral transport bound by the runtime composition."""

    __slots__ = ("_acquire",)

    def __init__(
        self, acquire: Callable[[EvidenceAcquisitionRequest], AcquiredEvidence]
    ) -> None:
        if not callable(acquire):
            raise NativeEvidenceError("evidence transport is required")
        self._acquire = acquire

    def acquire(self, request: EvidenceAcquisitionRequest) -> AcquiredEvidence:
        result = self._acquire(request)
        if type(result) is not AcquiredEvidence:
            raise NativeEvidenceError("evidence transport result differs")
        return result


@dataclass(frozen=True, slots=True)
class PublicationRightsAssessment:
    record_id: str
    decision: Literal["PERMITTED", "HOLD"]
    permitted_use: str
    policy_digest: str
    evidence_digest: str

    def __post_init__(self) -> None:
        if self.record_id != _assessment_id(
            "RIGHTS",
            self.decision,
            self.permitted_use,
            self.policy_digest,
            self.evidence_digest,
        ):
            raise NativeEvidenceError("publication rights receipt differs")

    @classmethod
    def create(cls, **values: str) -> PublicationRightsAssessment:
        return cls(
            _assessment_id(
                "RIGHTS",
                values["decision"],
                values["permitted_use"],
                values["policy_digest"],
                values["evidence_digest"],
            ),
            **values,
        )


def rights_eligibility_digest(
    rights: PublicationRightsAssessment, *, body_digest: str,
    transport_digest: str, exclusion_signals: tuple[str, ...], text_only: bool,
) -> str:
    # The typed receipt already binds decision, use, policy AND rights evidence.
    # Both transports and the assessor add the same exact acquisition facts.
    if type(rights) is not PublicationRightsAssessment:
        raise NativeEvidenceError("publication rights authority differs")
    rights.__post_init__()
    return digest_canonical({
        "rights_receipt": rights.record_id, "body_digest": body_digest,
        "transport": transport_digest, "exclusion_signals": exclusion_signals,
        "text_only": text_only,
    })


@dataclass(frozen=True, slots=True)
class DependencyAssessment:
    record_id: str
    dependency_status: Literal["RESOLVED", "HOLD"]
    evidential_origin_id: str
    originating_report_id: str
    evidence_digest: str

    def __post_init__(self) -> None:
        if self.record_id != _assessment_id(
            "DEPENDENCY",
            self.dependency_status,
            self.evidential_origin_id,
            self.originating_report_id,
            self.evidence_digest,
        ):
            raise NativeEvidenceError("dependency receipt differs")

    @classmethod
    def create(cls, **values: str) -> DependencyAssessment:
        return cls(
            _assessment_id(
                "DEPENDENCY",
                values["dependency_status"],
                values["evidential_origin_id"],
                values["originating_report_id"],
                values["evidence_digest"],
            ),
            **values,
        )


@dataclass(frozen=True, slots=True)
class SourceAuthorityAssessment:
    record_id: str
    source_id: str
    governed_claim_id: str
    decision: Literal["ADMITTED", "HOLD"]
    authority_class: str
    authority_scope: str
    evidence_digest: str

    def __post_init__(self) -> None:
        if self.record_id != _assessment_id(
            "AUTHORITY",
            self.source_id,
            self.governed_claim_id,
            self.decision,
            self.authority_class,
            self.authority_scope,
            self.evidence_digest,
        ):
            raise NativeEvidenceError("source authority receipt differs")

    @classmethod
    def create(cls, **values: str) -> SourceAuthorityAssessment:
        return cls(
            _assessment_id(
                "AUTHORITY",
                values["source_id"],
                values["governed_claim_id"],
                values["decision"],
                values["authority_class"],
                values["authority_scope"],
                values["evidence_digest"],
            ),
            **values,
        )


@dataclass(frozen=True, slots=True)
class NativeEvidenceSource:
    unit: CorpusIngestUnit
    source_version: SourceDefinitionVersion
    rights: PublicationRightsAssessment
    dependency: DependencyAssessment


@dataclass(frozen=True, slots=True)
class AcquiredSourceAssessment:
    source_id: str
    currentness: SourceCurrentness
    integrity_results: tuple[tuple[str, Literal["PASS", "HOLD"]], ...]

@dataclass(frozen=True, slots=True)
class IndependentEvidenceAssessment:
    source_assessments: tuple[AcquiredSourceAssessment, ...]
    source_authority: tuple[SourceAuthorityAssessment, ...]
    substantive_new_information: tuple[str, ...]
    governed_claims: tuple[GovernedClaimEvidence, ...]
    qualification_evidence: tuple[QualificationEvidence, ...]
    assessment_records: tuple[dict[str, object], ...]
    selection_rationale: str
    geography: tuple[str, ...]
    categories: tuple[str, ...]
    explicit_exclusions: tuple[str, ...] = ()


class EvidenceAssessor:
    """Deterministic assessment of the independently acquired exact bytes."""

    __slots__ = ("_assess",)

    def __init__(
        self,
        assess: Callable[
            [
                StoryCandidateVersion,
                EvidencePackage,
                tuple[NativeEvidenceSource, ...],
                tuple[AcquiredEvidence, ...],
            ],
            IndependentEvidenceAssessment,
        ],
    ) -> None:
        if not callable(assess):
            raise NativeEvidenceError("evidence assessor is required")
        self._assess = assess

    def assess(
        self,
        candidate: StoryCandidateVersion,
        package: EvidencePackage,
        sources: tuple[NativeEvidenceSource, ...],
        acquired: tuple[AcquiredEvidence, ...],
        before_assessment: Callable[[], None] | None = None,
        cached_only: bool = False,
        semantic_only: bool = False,
        qualification_cached_only: bool = False,
    ) -> IndependentEvidenceAssessment:
        bounded = getattr(self._assess, "assess_with_boundary", None)
        if callable(bounded):
            result = bounded(
                candidate, package, sources, acquired,
                before_dispatch=before_assessment,
                cached_only=cached_only,
                **({'semantic_only': True} if semantic_only else {}),
                **({'qualification_cached_only': True} if qualification_cached_only else {}),
            )
        else:
            if cached_only or semantic_only or qualification_cached_only:
                source_id = (
                    sources[0].unit.source_id if sources else candidate.candidate_id
                )
                raise NativeEvidenceHold(
                    "ASSESSOR_REVALIDATION_CACHE_MISSING_HOLD", source_id
                )
            if before_assessment is not None:
                before_assessment()
            result = self._assess(candidate, package, sources, acquired)
        if type(result) is not IndependentEvidenceAssessment:
            raise NativeEvidenceError("evidence assessment differs")
        return result


@dataclass(frozen=True, slots=True)
class NativeEvidenceResult:
    retained: GovernedEvidencePackage
    editorial_decision: EditorialPolicyDecision
    acquisition_receipt_digests: tuple[str, ...]


def _admit_record(
    objects: GovernedObjects, body: bytes, proof: AuthenticationProof,
) -> ObjectAdmissionId:
    # Logical rights IDs survive revisions; governed bytes also bind the Candidate
    # and base package. Replay must therefore use the exact record content.
    digest = digest_bytes(body)
    admission = objects.admit(
        ObjectAdmissionRequest("evidence.record", f"record:{digest}"),
        body,
        proof=proof,
    ).admission
    if admission.blob.blob_digest != digest:
        raise NativeEvidenceError("retained evidence record content differs")
    return admission.admission_id


class NativeEvidenceController:
    """Acquire independently, validate exact records, then retain one package."""

    def __init__(
        self,
        *,
        objects: GovernedObjects,
        candidate_port: StoryCandidateReadPort,
        evidence_packages: GovernedEvidencePackages,
        transport: EvidenceTransport,
        assessor: EvidenceAssessor,
        policy_bundle_digest: str,
        transport_policy_digest: str,
        clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
    ) -> None:
        if not all(
            type(value) is expected
            for value, expected in (
                (objects, GovernedObjects),
                (candidate_port, StoryCandidateReadPort),
                (evidence_packages, GovernedEvidencePackages),
                (transport, EvidenceTransport),
                (assessor, EvidenceAssessor),
            )
        ):
            raise NativeEvidenceError("exact native evidence authorities required")
        self._objects = objects
        self._candidate_port = candidate_port
        self._packages = evidence_packages
        self._transport = transport
        self._assessor = assessor
        self._policy_bundle_digest = policy_bundle_digest
        self._transport_policy_digest = transport_policy_digest
        self._clock = clock

    def acquire_and_retain(
        self,
        *,
        candidate_version_id: str,
        intake_receipt_id: str,
        sources: tuple[NativeEvidenceSource, ...],
        evaluated_at: str | None = None,
        before_assessment: Callable[[], None] | None = None,
        assessment_cached_only: bool = False,
        assessment_semantic_only: bool = False,
        assessment_qualification_cached_only: bool = False,
        before_semantic_assessment: Callable[[EvidencePackage, tuple[AcquiredEvidence, ...]], None] | None = None,
        proof: AuthenticationProof,
    ) -> NativeEvidenceResult:
        if (
            type(assessment_cached_only) is not bool
            or type(assessment_semantic_only) is not bool
            or type(assessment_qualification_cached_only) is not bool
            or (assessment_qualification_cached_only and (not assessment_cached_only or assessment_semantic_only))
            or (before_semantic_assessment is not None and not callable(before_semantic_assessment))
            or type(sources) is not tuple
            or not sources
            or any(type(item) is not NativeEvidenceSource for item in sources)
        ):
            raise NativeEvidenceError("native evidence request differs")
        version = self._candidate_port.require_retained_version(
            candidate_version_id
        )
        requests = tuple(self._preflight(item) for item in sources)
        acquired = tuple(
            self._acquire(source, request)
            for source, request in zip(sources, requests, strict=True)
        )
        source_ids = tuple(item.unit.source_id for item in sources)
        base = EvidencePackage(
            candidate_id=version.candidate_id,
            hypothesis_id=version.governing_manifest.hypothesis_id,
            signal_ids=tuple(
                item.signal_id
                for item in version.governing_manifest.lead_signal_bindings
            ),
            lead_ids=tuple(
                item.lead_id
                for item in version.governing_manifest.lead_signal_bindings
            ),
            source_ids=source_ids,
            observation_digests=tuple(item.body_digest for item in acquired),
            passages=tuple(self._passage(item) for item in acquired),
        )
        if assessment_semantic_only and before_semantic_assessment is not None:
            before_semantic_assessment(base, acquired)
        assessment = self._assessor.assess(
            version, base, sources, acquired,
            before_assessment=before_assessment,
            cached_only=assessment_cached_only,
            **({'semantic_only': True} if assessment_semantic_only else {}),
            **({'qualification_cached_only': True} if assessment_qualification_cached_only else {}),
        )
        source_assessments = self._validated_source_assessments(
            sources, acquired, assessment.source_assessments
        )
        package = replace(
            base,
            substantive_new_information=assessment.substantive_new_information,
            governed_claims=assessment.governed_claims,
            qualification_evidence=assessment.qualification_evidence,
            selection_rationale=assessment.selection_rationale,
            geography=assessment.geography,
            categories=assessment.categories,
            explicit_exclusions=assessment.explicit_exclusions,
        )
        records = self._records(base, package, sources, acquired, assessment)
        inventory = tuple(
            (source.unit.source_id, result.canonical_url)
            for source, result in zip(sources, acquired, strict=True)
        )
        retained_rows = tuple(
            (
                str(record["record_id"]),
                str(record["record_type"]),
                canonical_json_bytes(record).decode(),
                digest_bytes(canonical_json_bytes(record)),
            )
            for record in records
        )
        if validate_governed_evidence_records(
            candidate_id=version.candidate_id,
            source_inventory=inventory,
            base_package_digest=base.digest,
            package=package,
            retained_records=retained_rows,
        ) is None:
            raise NativeEvidenceHold("EVIDENCE_VALIDATION_HOLD", source_ids[0])
        source_admissions = tuple(
            self._objects.admit(
                ObjectAdmissionRequest(
                    "evidence.source", f"source:{result.receipt_digest}"
                ),
                result.body,
                proof=proof,
            ).admission.admission_id
            for result in acquired
        )
        record_admissions = tuple(
            _admit_record(self._objects, row[2].encode(), proof)
            for row in retained_rows
        )
        retained = self._packages.retain(
            package,
            receipt_id=intake_receipt_id,
            candidate_port=self._candidate_port,
            source_admission_ids=source_admissions,
            record_admission_ids=record_admissions,
            proof=proof,
        )
        integrity = tuple(
            SourceIntegrity(
                source.unit.source_id,
                admission_id,
                result.body_digest,
                tuple(name for name, _ in source_assessment.integrity_results),
                "PASS",
                "INDEPENDENT_ACQUISITION_VERIFIED",
            )
            for source, result, admission_id, source_assessment in zip(
                sources,
                acquired,
                source_admissions,
                source_assessments,
                strict=True,
            )
        )
        decision = EditorialPolicyDecision.create(
            candidate_version_id=retained.candidate_version_id,
            candidate_version_digest=retained.candidate_version_digest,
            governing_manifest_digest=retained.governing_manifest_digest,
            package_admission_id=retained.package_admission_id,
            package_digest=retained.package.digest,
            policy_bundle_digest=self._policy_bundle_digest,
            # Live acquisition and assessment finish after entry to this
            # method. Timestamp the decision now, not before its own evidence.
            evaluated_at=evaluated_at or self._clock().to_text(),
            currentness=tuple(item.currentness for item in source_assessments),
            integrity=integrity,
            evidence_gate_results=tuple((gate, "PASS") for gate in _GATES),
        )
        return NativeEvidenceResult(
            retained,
            decision,
            tuple(item.receipt_digest for item in acquired),
        )

    def _preflight(
        self, source: NativeEvidenceSource
    ) -> EvidenceAcquisitionRequest:
        unit, version = source.unit, source.source_version
        authority = unit.authority
        if (
            authority is None
            or str(version.request.definition_id) != authority.definition_id
            or str(version.version_id) != authority.definition_version_id
            or unit.source_definition_url != version.request.locator
            or not _same_https_origin(unit.canonical_url, version.request.locator)
            or unit.effective_revision.source_id != unit.source_id
            or unit.revision_digest != unit.effective_revision.revision_digest
            or not {
                ("SOURCE_DEFINITION_VERSION", authority.definition_version_id),
                ("SOURCE_REVISION", authority.revision_id),
            }.issubset(
                {
                    (record.get("record_type"), record.get("record_id"))
                    for record in authority.records
                }
            )
        ):
            raise NativeEvidenceHold("SOURCE_BINDING_HOLD", unit.source_id)
        if version.request.lifecycle_stage not in {
            SourceLifecycleStage.SHADOW_SHORTLISTED,
            SourceLifecycleStage.PRODUCTION_ELIGIBLE,
        }:
            raise NativeEvidenceHold("SOURCE_LIFECYCLE_HOLD", unit.source_id)
        if (
            source.rights.decision != "PERMITTED"
            or source.rights.permitted_use != "PUBLICATION_EVIDENCE"
        ):
            raise NativeEvidenceHold("PUBLICATION_RIGHTS_HOLD", unit.source_id)
        if source.dependency.dependency_status != "RESOLVED":
            raise NativeEvidenceHold("SOURCE_DEPENDENCY_HOLD", unit.source_id)
        return EvidenceAcquisitionRequest(
            unit.source_id,
            authority.definition_id,
            authority.definition_version_id,
            version.canonical_digest,
            authority.revision_id,
            unit.canonical_url,
            self._transport_policy_digest,
        )

    def _acquire(
        self,
        source: NativeEvidenceSource,
        request: EvidenceAcquisitionRequest,
    ) -> AcquiredEvidence:
        unit = source.unit
        result = self._transport.acquire(request)
        if (
            result.request_digest != request.digest
            or result.outcome != "COMPLETE"
            or result.canonical_url != unit.canonical_url
            or any(
                type(value) is not str or not value.strip()
                for value in (
                    result.publisher,
                    result.responsible_body,
                    result.source_type,
                    result.publication_time,
                    result.source_updated_time,
                    result.retrieval_time,
                    result.geography,
                    result.language,
                )
            )
        ):
            raise NativeEvidenceHold("INDEPENDENT_ACQUISITION_HOLD", unit.source_id)
        return result

    @staticmethod
    def _validated_source_assessments(sources, acquired, assessments):
        if (
            type(assessments) is not tuple
            or len(assessments) != len(sources)
        ):
            raise NativeEvidenceHold(
                "SOURCE_ASSESSMENT_HOLD", sources[0].unit.source_id
            )
        for source, result, assessment in zip(
            sources, acquired, assessments, strict=True
        ):
            if type(assessment) is not AcquiredSourceAssessment:
                raise NativeEvidenceHold(
                    "SOURCE_ASSESSMENT_HOLD", source.unit.source_id
                )
            if type(assessment.currentness) is not SourceCurrentness:
                raise NativeEvidenceHold(
                    "SOURCE_ASSESSMENT_HOLD", source.unit.source_id
                )
            currentness = assessment.currentness
            if (
                assessment.source_id != source.unit.source_id
                or type(currentness) is not SourceCurrentness
                or currentness.source_id != source.unit.source_id
                or currentness.source_definition_id
                != source.unit.authority.definition_id
                or currentness.source_definition_revision_digest
                != source.source_version.canonical_digest
                or currentness.publication_time != result.publication_time
                or currentness.retrieval_time != result.retrieval_time
                or currentness.evidence_digest != result.transport_evidence_digest
                or (
                    result.currentness_basis == "AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT"
                    and (
                        currentness.currency_family != "CURRENT_VERSION"
                        or currentness.version_reference != result.source_updated_time
                        or currentness.supersession_evidence_digest != result.transport_evidence_digest
                    )
                )
                or (
                    result.currentness_basis == "RETAINED_AUTHORITATIVE_COMPLETED_EVENT"
                    and (
                        source.unit.source_id != "HK-02"
                        or currentness.currency_family != "COMPLETED_HISTORICAL_EVENT"
                        or currentness.version_reference != result.source_updated_time
                        or currentness.supersession_evidence_digest is not None
                        or currentness.currency_window_seconds is not None
                    )
                )
                or currentness.result != "PASS"
                or assessment.integrity_results
                != tuple((name, "PASS") for name in _INTEGRITY_CHECKS)
            ):
                raise NativeEvidenceHold(
                    "SOURCE_ASSESSMENT_HOLD", source.unit.source_id
                )
        return assessments

    @staticmethod
    def _passage(result: AcquiredEvidence) -> str:
        try:
            passage = result.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NativeEvidenceHold("SOURCE_ENCODING_HOLD", result.canonical_url) from exc
        if not passage.strip():
            raise NativeEvidenceHold("SOURCE_CONTENT_HOLD", result.canonical_url)
        return passage

    @staticmethod
    def _records(base, package, sources, acquired, assessment):
        if not (
            package.governed_claims
            or package.substantive_new_information
            or package.qualification_evidence
        ):
            raise NativeEvidenceHold(
                "NO_QUALIFYING_NEW_INFORMATION", sources[0].unit.source_id
            )
        common = {
            "candidate_id": package.candidate_id,
            "base_package_digest": base.digest,
            "status": "CURRENT",
        }
        records: list[dict[str, object]] = []
        for source, result in zip(sources, acquired, strict=True):
            authority = tuple(
                item
                for item in assessment.source_authority
                if item.source_id == source.unit.source_id
            )
            authority_classes = {item.authority_class for item in authority}
            if (
                not authority
                or len(authority_classes) != 1
                or any(item.decision != "ADMITTED" for item in authority)
            ):
                raise NativeEvidenceHold(
                    "SOURCE_AUTHORITY_HOLD", source.unit.source_id
                )
            records.extend(
                (
                    {
                        **common,
                        "record_id": result.receipt_digest,
                        "record_type": "SOURCE_RECORD",
                        "source_id": source.unit.source_id,
                        "canonical_url": result.canonical_url,
                        "publisher": result.publisher,
                        "responsible_body": result.responsible_body,
                        "source_type": result.source_type,
                        "authority_class": next(iter(authority_classes)),
                        "publication_time": result.publication_time,
                        "retrieval_time": result.retrieval_time,
                        "geography": result.geography,
                        "language": result.language,
                        "extraction_status": "COMPLETE",
                        "rights_decision_id": source.rights.record_id,
                        "originating_report_id": source.dependency.originating_report_id,
                        "originating_artefact_digest": result.body_digest,
                        "dependency_evidence_ids": [source.dependency.record_id],
                    },
                    {
                        **common,
                        "record_id": source.rights.record_id,
                        "record_type": "RIGHTS_DECISION",
                        "source_id": source.unit.source_id,
                        "decision": source.rights.decision,
                        "permitted_use": source.rights.permitted_use,
                    },
                    {
                        **common,
                        "record_id": source.dependency.record_id,
                        "record_type": "DEPENDENCY_EVIDENCE",
                        "source_id": source.unit.source_id,
                        "dependency_status": source.dependency.dependency_status,
                        "evidential_origin_id": source.dependency.evidential_origin_id,
                        "originating_report_id": source.dependency.originating_report_id,
                    },
                )
            )
            records.extend(
                {
                    **common,
                    "record_id": item.record_id,
                    "record_type": "SOURCE_AUTHORITY_DECISION",
                    "source_id": source.unit.source_id,
                    "decision": item.decision,
                    "authority_class": item.authority_class,
                    "authority_scope": item.authority_scope,
                    "governed_claim_id": item.governed_claim_id,
                    "claim_digest": digest_bytes(
                        next(
                            claim.claim
                            for claim in package.governed_claims
                            if claim.claim_id == item.governed_claim_id
                        ).encode()
                    ),
                }
                for item in authority
            )
        records.extend({**record, **common} for record in assessment.assessment_records)
        return tuple(records)


def _assessment_id(kind: str, *values: str) -> str:
    return digest_bytes(canonical_json_bytes([kind, *values]))


def _same_https_origin(left: str, right: str) -> bool:
    try:
        first, second = urlsplit(left), urlsplit(right)
        return not any((first.username, first.password, second.username, second.password)) and (
            first.scheme,
            first.hostname,
            first.port or 443,
        ) == (
            "https",
            second.hostname,
            second.port or 443,
        ) and second.scheme == "https"
    except ValueError:
        return False
