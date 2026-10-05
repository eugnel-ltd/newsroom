"""Offline native editorial admission and authoritative Story Versions."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Literal

from newsroom.authority import (
    AggregateId,
    AuthenticationProof,
    AuthorityCommands,
    AuthorityEvents,
    CommandId,
    EventId,
    GovernedObjects,
    HydrationRequest,
    ObjectAdmissionId,
    ObjectAdmissionPayload,
    ObjectAdmissionRequest,
    SemanticCommand,
    UtcTimestamp,
)
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    validate_sha256_digest,
)
from newsroom.authority.types import TrustScope
from newsroom.control_plane.admission import (
    DeterministicWriteAdmission,
    WriteAdmissionDecision,
)
from newsroom.control_plane.evidence import (
    EVIDENCE_GATE_POLICY_VERSION,
    EvidenceGateEvidence,
    EvidencePackage,
)
from newsroom.control_plane.writer import (
    WriterCopy,
    WriterEvidenceLink,
    WriterValidatorResult,
    required_surface_copy,
    validate_writer_copy,
)
from newsroom.increment6.candidates import StoryCandidateReadPort

from .evidence import GovernedEvidencePackage, GovernedEvidencePackages

DECISION_SCHEMA = "newsroom.increment10.editorial-policy-decision.v1"
STORY_VERSION_SCHEMA = "newsroom.increment10.story-version.v1"
DECISION_CLASS = "editorial_policy_decision"
DECISION_USE = "editorial_admission_input"
DECISION_PURPOSE = "editorial.policy-decision"
DECISION_ADMISSION_TYPE = "editorial.policy-decision"
DECISION_COMMAND = "editorial.package-decision.record"
DECISION_EVENT = "editorial.package-decision.recorded"
STORY_CLASS = "story_version"
STORY_USE = "editorial_story_version"
STORY_PURPOSE = "editorial.story-version"
STORY_ADMISSION_TYPE = "editorial.story-version"
STORY_COMMAND = "editorial.story-version.admit"
STORY_EVENT = "editorial.story-version.admitted"
EDITORIAL_SECURITY_SCOPE = "authority.internal"
EDITORIAL_RETENTION_SCOPE = "editorial.retained"

_GATES = ("CLAIM_TRACEABILITY", "EVIDENCE_SUFFICIENCY", "SOURCE_AUTHORITY")
_INTEGRITY_CHECKS = (
    "ACCESS_COMPLETE",
    "ENCODING_VALID",
    "EXTRACTION_COMPLETE",
    "NOT_PAYWALL_FRAGMENT",
    "NOT_TRUNCATED",
    "VERSION_UNAMBIGUOUS",
)


class EditorialError(ValueError):
    """Raised when exact editorial authority cannot be established."""


class EditorialHold(EditorialError):
    """Fail-closed non-error outcome with no Story Version effect."""

    def __init__(
        self,
        decision: WriteAdmissionDecision | None = None,
        *,
        reason: str = "",
    ) -> None:
        if decision is None and not reason:
            raise EditorialError("editorial hold reason is required")
        super().__init__(reason or ", ".join(decision.stable_reason_codes))
        self.decision = decision


@dataclass(frozen=True, slots=True)
class SourceCurrentness:
    source_id: str
    source_definition_id: str
    source_definition_revision_digest: str
    currency_family: Literal[
        "CURRENT_VERSION", "OBSERVED_STATE", "COMPLETED_HISTORICAL_EVENT"
    ]
    publication_time: str
    retrieval_time: str
    currency_window_seconds: int | None
    version_reference: str | None
    supersession_evidence_digest: str | None
    evidence_digest: str
    result: Literal["PASS", "HOLD"]
    reason_code: str

    def __post_init__(self) -> None:
        _text_fields(
            self,
            "source_id",
            "source_definition_id",
            "publication_time",
            "retrieval_time",
        )
        _token_fields(self, "currency_family", "result", "reason_code")
        published = _utc_timestamp(self.publication_time)
        retrieved = _utc_timestamp(self.retrieval_time)
        if retrieved.value < published.value:
            raise EditorialError("source retrieval precedes publication")
        if self.currency_family not in {
            "CURRENT_VERSION",
            "OBSERVED_STATE",
            "COMPLETED_HISTORICAL_EVENT",
        } or self.result not in {"PASS", "HOLD"}:
            raise EditorialError("source currentness result differs")
        if self.currency_family == "OBSERVED_STATE":
            if (
                (
                    self.currency_window_seconds is not None
                    and (
                        type(self.currency_window_seconds) is not int
                        or self.currency_window_seconds <= 0
                    )
                )
                or (self.result == "PASS" and self.currency_window_seconds is None)
                or self.version_reference is not None
                or self.supersession_evidence_digest is not None
            ):
                raise EditorialError("observed-state currency policy differs")
        elif (
            self.currency_window_seconds is not None
            or type(self.version_reference) is not str
            or not self.version_reference
        ):
            raise EditorialError("versioned source currency policy differs")
        _digests(
            self.source_definition_revision_digest,
            self.evidence_digest,
        )
        if self.currency_family == "CURRENT_VERSION":
            if self.supersession_evidence_digest is None:
                raise EditorialError("current-version supersession evidence is required")
            _digests(self.supersession_evidence_digest)
        elif self.supersession_evidence_digest is not None:
            raise EditorialError("source supersession evidence differs")

    def value(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class SourceIntegrity:
    source_id: str
    source_admission_id: ObjectAdmissionId
    source_digest: str
    checks: tuple[str, ...]
    result: Literal["PASS", "HOLD"]
    reason_code: str

    def __post_init__(self) -> None:
        _text_fields(self, "source_id")
        if type(self.source_admission_id) is not ObjectAdmissionId:
            raise EditorialError("source integrity admission identity differs")
        _digests(self.source_digest)
        if self.checks != _INTEGRITY_CHECKS:
            raise EditorialError("source integrity checks differ")
        _token_fields(self, "result", "reason_code")
        if self.result not in {"PASS", "HOLD"}:
            raise EditorialError("source integrity result differs")

    def value(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_admission_id": str(self.source_admission_id),
            "source_digest": self.source_digest,
            "checks": list(self.checks),
            "result": self.result,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class EditorialPolicyDecision:
    decision_id: str
    candidate_version_id: str
    candidate_version_digest: str
    governing_manifest_digest: str
    package_admission_id: ObjectAdmissionId
    package_digest: str
    policy_bundle_digest: str
    evaluated_at: str
    currentness: tuple[SourceCurrentness, ...]
    integrity: tuple[SourceIntegrity, ...]
    evidence_gate_results: tuple[tuple[str, Literal["PASS", "HOLD"]], ...]

    def __post_init__(self) -> None:
        _text_fields(self, "candidate_version_id", "evaluated_at")
        evaluated = _utc_timestamp(self.evaluated_at)
        if type(self.package_admission_id) is not ObjectAdmissionId:
            raise EditorialError("decision package admission identity differs")
        _digests(
            self.candidate_version_digest,
            self.governing_manifest_digest,
            self.package_digest,
            self.policy_bundle_digest,
        )
        if not self.currentness or not self.integrity:
            raise EditorialError("decision source provenance is required")
        for item in self.currentness:
            published = _utc_timestamp(item.publication_time)
            retrieved = _utc_timestamp(item.retrieval_time)
            if retrieved.value > evaluated.value or (
                item.currency_family == "OBSERVED_STATE"
                and item.result == "PASS"
                and (evaluated.value - published.value).total_seconds()
                > item.currency_window_seconds
            ):
                raise EditorialError("source currentness is stale or future-dated")
        if tuple(item[0] for item in self.evidence_gate_results) != _GATES or any(
            result not in {"PASS", "HOLD"}
            for _gate, result in self.evidence_gate_results
        ):
            raise EditorialError("decision evidence gates differ")
        if self.decision_id != digest_bytes(
            canonical_json_bytes(self.value(include_identity=False))
        ):
            raise EditorialError("editorial decision identity differs")

    def value(self, *, include_identity: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_identity": DECISION_SCHEMA,
            "candidate_version_id": self.candidate_version_id,
            "candidate_version_digest": self.candidate_version_digest,
            "governing_manifest_digest": self.governing_manifest_digest,
            "package_admission_id": str(self.package_admission_id),
            "package_digest": self.package_digest,
            "policy_bundle_digest": self.policy_bundle_digest,
            "evaluated_at": self.evaluated_at,
            "currentness": [item.value() for item in self.currentness],
            "integrity": [item.value() for item in self.integrity],
            "evidence_gate_results": [list(item) for item in self.evidence_gate_results],
        }
        if include_identity:
            value["decision_id"] = self.decision_id
        return value

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.value())

    @classmethod
    def create(cls, **values: Any) -> EditorialPolicyDecision:
        currentness = values["currentness"]
        integrity = values["integrity"]
        gates = values["evidence_gate_results"]
        identity = {
            "schema_identity": DECISION_SCHEMA,
            "candidate_version_id": values["candidate_version_id"],
            "candidate_version_digest": values["candidate_version_digest"],
            "governing_manifest_digest": values["governing_manifest_digest"],
            "package_admission_id": str(values["package_admission_id"]),
            "package_digest": values["package_digest"],
            "policy_bundle_digest": values["policy_bundle_digest"],
            "evaluated_at": values["evaluated_at"],
            "currentness": [item.value() for item in currentness],
            "integrity": [item.value() for item in integrity],
            "evidence_gate_results": [list(item) for item in gates],
        }
        return cls(
            decision_id=digest_bytes(canonical_json_bytes(identity)),
            **values,
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> EditorialPolicyDecision:
        value = _document(raw)
        if set(value) != {
            "schema_identity",
            "decision_id",
            "candidate_version_id",
            "candidate_version_digest",
            "governing_manifest_digest",
            "package_admission_id",
            "package_digest",
            "policy_bundle_digest",
            "evaluated_at",
            "currentness",
            "integrity",
            "evidence_gate_results",
        } or value["schema_identity"] != DECISION_SCHEMA:
            raise EditorialError("editorial decision schema differs")
        try:
            currentness = tuple(SourceCurrentness(**item) for item in value["currentness"])
            integrity = tuple(
                SourceIntegrity(
                    **{
                        **item,
                        "source_admission_id": ObjectAdmissionId.parse(
                            item["source_admission_id"]
                        ),
                        "checks": tuple(item["checks"]),
                    }
                )
                for item in value["integrity"]
            )
            decision = cls(
                decision_id=value["decision_id"],
                candidate_version_id=value["candidate_version_id"],
                candidate_version_digest=value["candidate_version_digest"],
                governing_manifest_digest=value["governing_manifest_digest"],
                package_admission_id=ObjectAdmissionId.parse(
                    value["package_admission_id"]
                ),
                package_digest=value["package_digest"],
                policy_bundle_digest=value["policy_bundle_digest"],
                evaluated_at=value["evaluated_at"],
                currentness=currentness,
                integrity=integrity,
                evidence_gate_results=tuple(
                    tuple(item) for item in value["evidence_gate_results"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EditorialError("editorial decision fields differ") from exc
        if decision.canonical_bytes() != raw:
            raise EditorialError("editorial decision is non-canonical")
        return decision


@dataclass(frozen=True, slots=True)
class DecisionReference:
    event_id: str
    admission_id: ObjectAdmissionId

    def __post_init__(self) -> None:
        EventId.parse(self.event_id)
        if type(self.admission_id) is not ObjectAdmissionId:
            raise EditorialError("decision reference admission differs")


@dataclass(frozen=True, slots=True)
class StoryVersionRequest:
    story_id: AggregateId
    expected_aggregate_version: int
    idempotency_key: str

    def __post_init__(self) -> None:
        if type(self.story_id) is not AggregateId:
            raise EditorialError("distinct Story aggregate identity is required")
        if (
            type(self.expected_aggregate_version) is not int
            or self.expected_aggregate_version < 0
        ):
            raise EditorialError("expected Story aggregate version differs")
        if type(self.idempotency_key) is not str or not self.idempotency_key.strip():
            raise EditorialError("Story idempotency key is required")


@dataclass(frozen=True, slots=True)
class StoryVersion:
    story_id: AggregateId
    aggregate_version: int
    candidate_version_id: str
    candidate_version_digest: str
    governing_manifest_digest: str
    package_admission_id: ObjectAdmissionId
    retained_package_digest: str
    admission_input_digest: str
    policy_decision_event_id: str
    policy_decision_admission_id: ObjectAdmissionId
    policy_decision_id: str
    write_admission: WriteAdmissionDecision
    copy: WriterCopy
    validators: tuple[WriterValidatorResult, ...]
    writer_review: dict | None = None
    story_format: str | None = None

    def __post_init__(self) -> None:
        if type(self.story_id) is not AggregateId or (
            type(self.aggregate_version) is not int or self.aggregate_version <= 0
        ):
            raise EditorialError("Story Version identity differs")
        _text_values(self.candidate_version_id, self.policy_decision_event_id)
        EventId.parse(self.policy_decision_event_id)
        if (
            type(self.package_admission_id) is not ObjectAdmissionId
            or type(self.policy_decision_admission_id) is not ObjectAdmissionId
        ):
            raise EditorialError("Story Version admission binding differs")
        _digests(
            self.candidate_version_digest,
            self.governing_manifest_digest,
            self.retained_package_digest,
            self.admission_input_digest,
            self.policy_decision_id,
        )
        if (
            type(self.write_admission) is not WriteAdmissionDecision
            or type(self.copy) is not WriterCopy
            or type(self.validators) is not tuple
            or not self.validators
            or any(type(item) is not WriterValidatorResult for item in self.validators)
        ):
            raise EditorialError("Story Version editorial values differ")
        if (self.writer_review is None) != (self.story_format is None):
            raise EditorialError("Story Version writer review binding differs")
        if (self.copy.writer_id == "newsroom.native-story-writer.v1") != (self.writer_review is not None):
            raise EditorialError("Story Version narrative review is required")
        if self.writer_review is not None and (
            type(self.writer_review) is not dict
            or self.story_format not in {"ARTICLE", "BRIEF"}
            or self.copy.writer_id != "newsroom.native-story-writer.v1"
        ):
            raise EditorialError("Story Version narrative identity differs")

    @property
    def digest(self) -> str:
        return digest_bytes(self.canonical_bytes())

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "schema_identity": STORY_VERSION_SCHEMA,
                "story_id": str(self.story_id),
                "aggregate_version": self.aggregate_version,
                "candidate_version_id": self.candidate_version_id,
                "candidate_version_digest": self.candidate_version_digest,
                "governing_manifest_digest": self.governing_manifest_digest,
                "package_admission_id": str(self.package_admission_id),
                "retained_package_digest": self.retained_package_digest,
                "admission_input_digest": self.admission_input_digest,
                "policy_decision_event_id": self.policy_decision_event_id,
                "policy_decision_admission_id": str(
                    self.policy_decision_admission_id
                ),
                "policy_decision_id": self.policy_decision_id,
                "write_admission": self.write_admission.as_record(),
                "copy": {
                    "title": self.copy.title,
                    "body": self.copy.body,
                    "writer_id": self.copy.writer_id,
                    "evidence_package_digest": self.copy.evidence_package_digest,
                    "evidence_links": [
                        {
                            "governed_claim_id": item.governed_claim_id,
                            "rendered_assertion": item.rendered_assertion,
                        }
                        for item in self.copy.evidence_links
                    ],
                },
                "validators": [
                    {
                        "validator": item.validator,
                        "result": item.result,
                        "reason_code": item.reason_code,
                    }
                    for item in self.validators
                ],
                **({"writer_review": self.writer_review, "story_format": self.story_format}
                   if self.writer_review is not None else {}),
            }
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> StoryVersion:
        value = _document(raw)
        try:
            copy = value["copy"]
            story = cls(
                story_id=AggregateId.parse(value["story_id"]),
                aggregate_version=value["aggregate_version"],
                candidate_version_id=value["candidate_version_id"],
                candidate_version_digest=value["candidate_version_digest"],
                governing_manifest_digest=value["governing_manifest_digest"],
                package_admission_id=ObjectAdmissionId.parse(
                    value["package_admission_id"]
                ),
                retained_package_digest=value["retained_package_digest"],
                admission_input_digest=value["admission_input_digest"],
                policy_decision_event_id=value["policy_decision_event_id"],
                policy_decision_admission_id=ObjectAdmissionId.parse(
                    value["policy_decision_admission_id"]
                ),
                policy_decision_id=value["policy_decision_id"],
                write_admission=WriteAdmissionDecision.from_record(
                    value["write_admission"]
                ),
                copy=WriterCopy(
                    title=copy["title"],
                    body=copy["body"],
                    writer_id=copy["writer_id"],
                    evidence_package_digest=copy["evidence_package_digest"],
                    evidence_links=tuple(
                        WriterEvidenceLink(**item) for item in copy["evidence_links"]
                    ),
                ),
                validators=tuple(
                    WriterValidatorResult(**item) for item in value["validators"]
                ),
                writer_review=value.get("writer_review"),
                story_format=value.get("story_format"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EditorialError("Story Version fields differ") from exc
        if story.canonical_bytes() != raw:
            raise EditorialError("Story Version is non-canonical")
        return story


@dataclass(frozen=True, slots=True)
class StoryVersionReceipt:
    command_id: str
    event_id: str
    story_id: AggregateId
    aggregate_version: int
    admission_id: ObjectAdmissionId
    story_version_digest: str

    def __post_init__(self) -> None:
        CommandId.parse(self.command_id)
        EventId.parse(self.event_id)
        if type(self.story_id) is not AggregateId:
            raise EditorialError("Story receipt aggregate differs")
        if type(self.admission_id) is not ObjectAdmissionId:
            raise EditorialError("Story receipt admission differs")
        _digests(self.story_version_digest)


class NativeEditorial:
    """Admit one offline-validated Story Version through the authority ledger."""

    def __init__(
        self,
        *,
        objects: GovernedObjects,
        commands: AuthorityCommands,
        events: AuthorityEvents,
        evidence: GovernedEvidencePackages,
        reader_principal_id: str,
        reader_authority_domain: str,
        controller_principal_id: str,
        story_principal_id: str,
        policy_bundle_digest: str,
        decision_hydration_policy_digest: str,
        story_hydration_policy_digest: str,
        decision_command_definition_digest: str,
        story_command_definition_digest: str,
        story_admission_definition_digest: str,
        story_writer=None, clock=UtcTimestamp.now, retained_policy_bundles=(),
    ) -> None:
        if not all(
            type(value) is expected
            for value, expected in (
                (objects, GovernedObjects),
                (commands, AuthorityCommands),
                (events, AuthorityEvents),
                (evidence, GovernedEvidencePackages),
            )
        ):
            raise EditorialError("exact native authorities are required")
        _text_values(
            reader_principal_id,
            reader_authority_domain,
            controller_principal_id,
            story_principal_id,
        )
        _digests(
            policy_bundle_digest,
            decision_hydration_policy_digest,
            story_hydration_policy_digest,
            decision_command_definition_digest,
            story_command_definition_digest,
            story_admission_definition_digest,
        )
        self._objects = objects
        self._commands = commands
        self._events = events
        self._evidence = evidence
        self._reader_principal = reader_principal_id
        self._authority_domain = reader_authority_domain
        self._controller_principal = controller_principal_id
        self._story_principal = story_principal_id
        self._policy_bundle_digest = policy_bundle_digest
        if type(retained_policy_bundles) is not tuple:
            raise EditorialError("retained editorial policies differ")
        _digests(*retained_policy_bundles)
        self._retained_policy_bundles = retained_policy_bundles
        self._decision_policy = decision_hydration_policy_digest
        self._story_policy = story_hydration_policy_digest
        self._decision_definition = decision_command_definition_digest
        self._story_definition = story_command_definition_digest
        self._story_admission_definition = story_admission_definition_digest
        self._story_writer = story_writer
        self._clock = clock

    def admit_story_version(
        self,
        request: StoryVersionRequest,
        *,
        package_admission_id: ObjectAdmissionId,
        decision_reference: DecisionReference,
        candidate_port: StoryCandidateReadPort,
        proof: AuthenticationProof,
        reviewed_copy_from: StoryVersionReceipt | None = None,
        story_cached_only: bool = False,
    ) -> tuple[StoryVersionReceipt, StoryVersion]:
        if type(request) is not StoryVersionRequest:
            raise EditorialError("immutable Story Version request is required")
        if type(story_cached_only) is not bool:
            raise EditorialError('story cache mode differs')
        if type(decision_reference) is not DecisionReference:
            raise EditorialError("immutable editorial decision reference is required")
        retained = self._evidence.read(
            package_admission_id, candidate_port=candidate_port, proof=proof
        )
        try:
            decision = self._read_policy_decision(
                decision_reference, retained=retained, proof=proof
            )
        except KeyError:
            raise EditorialHold(reason="EDITORIAL_POLICY_DECISION_MISSING") from None
        admission_request = ObjectAdmissionRequest(
            STORY_ADMISSION_TYPE,
            f"story-version:{request.story_id}:{request.expected_aggregate_version + 1}",
        )

        def legacy_story(blob_digest: str) -> StoryVersion | None:
            for legacy_writer_id in (
                "newsroom.offline-exact-copy.v2",
                "newsroom.offline-exact-copy.v1",
            ):
                legacy = self._build_story(
                    request,
                    retained,
                    decision,
                    decision_reference,
                    writer_id=legacy_writer_id,
                )
                if legacy.digest == blob_digest:
                    return legacy
            return None

        committed_admission = self._objects.committed_admission(
            admission_request,
            proof=proof,
        )
        if committed_admission is not None:
            admitted = committed_admission.admission
            hydrated = self._objects.rehydrate(
                HydrationRequest(admitted.admission_id, STORY_PURPOSE), proof=proof
            )
            self._verify_access(hydrated.decision, policy=self._story_policy,
                                object_class=STORY_CLASS, allowed_use=STORY_USE)
            original = StoryVersion.from_bytes(hydrated.data)
            story = self._build_story(
                request, retained, decision, decision_reference,
                **({'story_cached_only': True} if story_cached_only else {}),
                writer_id=original.copy.writer_id,
                retained_copy=original.copy if original.writer_review is not None else None,
                writer_review=original.writer_review, story_format=original.story_format,
                retained_admission=original.write_admission,
            )
        else:
            if decision.policy_bundle_digest != self._policy_bundle_digest:
                raise EditorialHold(reason="CURRENT_EDITORIAL_POLICY_REQUIRED")
            reviewed = None
            if reviewed_copy_from is not None:
                reviewed = self.read_story_version(reviewed_copy_from, candidate_port=candidate_port, proof=proof)
                if (reviewed.story_id != request.story_id or reviewed.package_admission_id != package_admission_id
                        or reviewed.retained_package_digest != retained.package.digest
                        or reviewed.candidate_version_id != retained.candidate_version_id
                        or reviewed.candidate_version_digest != retained.candidate_version_digest
                        or reviewed.governing_manifest_digest != retained.governing_manifest_digest):
                    raise EditorialError("reviewed correction copy belongs to another package")
            story = self._build_story(
                request, retained, decision, decision_reference,
                **({'story_cached_only': True} if story_cached_only else {}),
                **({"writer_id": reviewed.copy.writer_id,
                    "retained_copy": reviewed.copy if reviewed.writer_review is not None else None,
                    "writer_review": reviewed.writer_review, "story_format": reviewed.story_format,
                    "check_reused_source_currency": True}
                   if reviewed is not None else {}),
            )
            if story.writer_review is not None:
                refreshed = self._evidence.read(
                    package_admission_id, candidate_port=candidate_port, proof=proof
                )
                if (refreshed.package.digest != retained.package.digest
                        or refreshed.candidate_version_digest != retained.candidate_version_digest
                        or refreshed.governing_manifest_digest != retained.governing_manifest_digest):
                    raise EditorialHold(reason="NATIVE_STORY_SOURCE_DRIFT_HOLD")
                self._read_policy_decision(decision_reference, retained=refreshed, proof=proof)
            admission = self._objects.admit(
                admission_request,
                story.canonical_bytes(),
                proof=proof,
            )
            admitted = admission.admission
        if admitted.blob.blob_digest != story.digest:
            # Object admission can predate the writer upgrade or a crash before
            # its Story event. Accept only the exact old bytes reconstructed
            # from these same immutable package/decision/request identities.
            legacy = legacy_story(admitted.blob.blob_digest)
            if legacy is not None:
                story = legacy
        if (
            admitted.definition_digest != self._story_admission_definition
            or admitted.object_class != STORY_CLASS
            or admitted.allowed_use != STORY_USE
            or admitted.blob.blob_digest != story.digest
            or not admitted.active
        ):
            raise EditorialError("Story Version object admission differs")
        committed = self._commands.execute(
            SemanticCommand(
                command_type=STORY_COMMAND,
                aggregate_id=request.story_id,
                expected_aggregate_version=request.expected_aggregate_version,
                payload=ObjectAdmissionPayload(admitted.admission_id),
                idempotency_key=request.idempotency_key,
            ),
            proof=proof,
        )
        receipt = StoryVersionReceipt(
            command_id=committed.command_id,
            event_id=committed.event_id,
            story_id=request.story_id,
            aggregate_version=committed.aggregate_version,
            admission_id=admitted.admission_id,
            story_version_digest=story.digest,
        )
        self._verify_story_event(receipt, proof=proof)
        if committed.aggregate_version != story.aggregate_version:
            raise EditorialError("Story Version aggregate version differs")
        return receipt, story

    def read_story_version(
        self,
        receipt: StoryVersionReceipt,
        *,
        candidate_port: StoryCandidateReadPort,
        proof: AuthenticationProof,
    ) -> StoryVersion:
        if type(receipt) is not StoryVersionReceipt:
            raise EditorialError("immutable Story Version receipt is required")
        self._verify_story_event(receipt, proof=proof)
        hydrated = self._objects.rehydrate(
            HydrationRequest(receipt.admission_id, STORY_PURPOSE), proof=proof
        )
        self._verify_access(
            hydrated.decision,
            policy=self._story_policy,
            object_class=STORY_CLASS,
            allowed_use=STORY_USE,
        )
        story = StoryVersion.from_bytes(hydrated.data)
        if (
            story.digest != receipt.story_version_digest
            or story.story_id != receipt.story_id
            or story.aggregate_version != receipt.aggregate_version
        ):
            raise EditorialError("Story Version receipt binding differs")
        retained = self._evidence.read(
            story.package_admission_id, candidate_port=candidate_port, proof=proof
        )
        reference = DecisionReference(
            story.policy_decision_event_id, story.policy_decision_admission_id
        )
        decision = self._read_policy_decision(reference, retained=retained, proof=proof)
        rebuilt = self._build_story(
            StoryVersionRequest(
                story.story_id, story.aggregate_version - 1, "readback"
            ),
            retained,
            decision,
            reference,
            writer_id=story.copy.writer_id,
            retained_copy=story.copy if story.writer_review is not None else None,
            writer_review=story.writer_review, story_format=story.story_format,
            retained_admission=story.write_admission,
        )
        if rebuilt.canonical_bytes() != hydrated.data:
            raise EditorialError("Story Version replay differs")
        return story

    def _build_story(
        self,
        request: StoryVersionRequest,
        retained: GovernedEvidencePackage,
        policy: EditorialPolicyDecision,
        reference: DecisionReference,
        *, writer_id: str | None = None, retained_copy: WriterCopy | None = None,
        writer_review: dict | None = None, story_format: str | None = None,
        retained_admission: WriteAdmissionDecision | None = None,
        check_reused_source_currency: bool = False,
        story_cached_only: bool = False,
    ) -> StoryVersion:
        package = retained.package
        if str(request.story_id) == package.candidate_id:
            raise EditorialError("Story identity must differ from Candidate identity")
        gates = policy.evidence_gate_results
        governed_claim_ids = tuple(item.claim_id for item in package.governed_claims)
        evaluated = replace(
            package,
            evidence_gate_results=gates,
            evidence_gate_evidence=(
                tuple(
                    EvidenceGateEvidence(
                        gate,
                        result,
                        governed_claim_ids,
                        EVIDENCE_GATE_POLICY_VERSION,
                    )
                    for gate, result in gates
                )
                if all(result == "PASS" for _gate, result in gates)
                else ()
            ),
            freshness_result=(
                "PASS"
                if all(item.result == "PASS" for item in policy.currentness)
                else "HOLD"
            ),
            integrity_result=(
                "PASS"
                if all(item.result == "PASS" for item in policy.integrity)
                else "HOLD"
            ),
        )
        decision = DeterministicWriteAdmission().decide_candidate_identity(
            candidate_id=evaluated.candidate_id,
            hypothesis_id=evaluated.hypothesis_id,
            package=evaluated,
            decided_at=policy.evaluated_at,
        )
        if decision.decision != "WRITE_READY":
            raise EditorialHold(decision)
        if retained_admission is not None:
            # Recheck current readiness without rewriting immutable policy identity.
            # Only policy version and its derived ID may differ during read/replay.
            record = decision.as_record()
            record.update(policy_version=retained_admission.policy_version,
                          decision_id=retained_admission.decision_id)
            if record != retained_admission.as_record():
                raise EditorialError("Story Version object admission differs: retained write-admission")
            decision = WriteAdmissionDecision.from_record(record)
        def require_current_sources():
            # A model turn may cross the original observation's currency window.
            now = self._clock()
            now_value = now.value
            for item in policy.currentness:
                if item.currency_family == "OBSERVED_STATE":
                    elapsed = (now_value - _utc_timestamp(item.retrieval_time).value).total_seconds()
                    if elapsed < 0 or elapsed > item.currency_window_seconds:
                        raise EditorialHold(reason="NATIVE_STORY_SOURCE_CURRENCY_HOLD")
        if writer_id is None:
            writer_id = ("newsroom.native-story-writer.v1" if self._story_writer is not None
                         else "newsroom.offline-exact-copy.v3")
        if writer_id == "newsroom.native-story-writer.v1":
            from newsroom.control_plane.native_story_writer import validate_retained_story, NativeStoryWriterHold
            if retained_copy is None:
                if self._story_writer is None:
                    raise EditorialHold(reason="NATIVE_STORY_WRITER_UNAVAILABLE")
                try:
                    result = self._story_writer(
                        evaluated, candidate_id=retained.package.candidate_id,
                        hypothesis_digest=retained.governing_manifest_digest,
                        admission_decision_id=decision.decision_id,
                        require_current=require_current_sources,
                        source_currentness=policy.currentness,
                        **({'cached_only': True} if story_cached_only else {}),
                    )
                except NativeStoryWriterHold as exc:
                    raise EditorialHold(reason=str(exc)) from exc
                copy, writer_review, story_format = result.copy, result.review.as_record(), result.format
                require_current_sources()
            else:
                copy = retained_copy
                if check_reused_source_currency:
                    require_current_sources()
            validators = validate_retained_story(copy, evaluated, writer_review, story_format,
                source_currentness=policy.currentness)
            if not validators or any(item.result != "PASS" for item in validators):
                raise EditorialHold(reason="NATIVE_STORY_SOURCE_SUPPORT_HOLD")
            return StoryVersion(
                story_id=request.story_id, aggregate_version=request.expected_aggregate_version + 1,
                candidate_version_id=retained.candidate_version_id,
                candidate_version_digest=retained.candidate_version_digest,
                governing_manifest_digest=retained.governing_manifest_digest,
                package_admission_id=retained.package_admission_id,
                retained_package_digest=retained.package.digest, admission_input_digest=evaluated.digest,
                policy_decision_event_id=reference.event_id,
                policy_decision_admission_id=reference.admission_id, policy_decision_id=policy.decision_id,
                write_admission=decision, copy=copy, validators=validators,
                writer_review=writer_review, story_format=story_format,
            )
        if writer_id not in {
            "newsroom.offline-exact-copy.v1",
            "newsroom.offline-exact-copy.v2",
            "newsroom.offline-exact-copy.v3",
        }:
            raise EditorialError("native Story Version writer differs")
        try:
            title, body, links = required_surface_copy(
                evaluated,
                paragraphs=writer_id != "newsroom.offline-exact-copy.v1",
                context_preserving=writer_id == "newsroom.offline-exact-copy.v3",
            )
        except ValueError as exc:
            if (
                writer_id == "newsroom.offline-exact-copy.v3"
                and str(exc).startswith("governed claim source span order")
            ):
                raise EditorialHold(
                    reason="STORY_SOURCE_CONTEXT_ORDER_HOLD"
                ) from None
            raise
        copy = WriterCopy(
            title,
            body,
            writer_id,
            evaluated.digest,
            links,
        )
        validators = validate_writer_copy(copy, evaluated)
        if not validators or any(item.result != "PASS" for item in validators):
            raise EditorialError("offline Story Version validation failed")
        return StoryVersion(
            story_id=request.story_id,
            aggregate_version=request.expected_aggregate_version + 1,
            candidate_version_id=retained.candidate_version_id,
            candidate_version_digest=retained.candidate_version_digest,
            governing_manifest_digest=retained.governing_manifest_digest,
            package_admission_id=retained.package_admission_id,
            retained_package_digest=retained.package.digest,
            admission_input_digest=evaluated.digest,
            policy_decision_event_id=reference.event_id,
            policy_decision_admission_id=reference.admission_id,
            policy_decision_id=policy.decision_id,
            write_admission=decision,
            copy=copy,
            validators=validators,
        )

    def _read_policy_decision(
        self,
        reference: DecisionReference,
        *,
        retained: GovernedEvidencePackage,
        proof: AuthenticationProof,
    ) -> EditorialPolicyDecision:
        provenance = self._events.provenance(reference.event_id, proof=proof)
        self._verify_event(
            provenance,
            admission_id=reference.admission_id,
            command=DECISION_COMMAND,
            event=DECISION_EVENT,
            definition_digest=self._decision_definition,
            producer_principal=self._controller_principal,
        )
        hydrated = self._objects.rehydrate(
            HydrationRequest(reference.admission_id, DECISION_PURPOSE), proof=proof
        )
        self._verify_access(
            hydrated.decision,
            policy=self._decision_policy,
            object_class=DECISION_CLASS,
            allowed_use=DECISION_USE,
        )
        decision = EditorialPolicyDecision.from_bytes(hydrated.data)
        expected_integrity = tuple(
            (source_id, admission_id, digest)
            for source_id, admission_id, digest in zip(
                retained.package.source_ids,
                retained.source_admission_ids,
                retained.package.observation_digests,
                strict=True,
            )
        )
        actual_integrity = tuple(
            (item.source_id, item.source_admission_id, item.source_digest)
            for item in decision.integrity
        )
        if (
            provenance.event.payload_digest != digest_bytes(hydrated.data)
            or decision.candidate_version_id != retained.candidate_version_id
            or decision.candidate_version_digest != retained.candidate_version_digest
            or decision.governing_manifest_digest
            != retained.governing_manifest_digest
            or decision.package_admission_id != retained.package_admission_id
            or decision.package_digest != retained.package.digest
            or decision.policy_bundle_digest not in (
                self._policy_bundle_digest, *self._retained_policy_bundles
            )
            or tuple(item.source_id for item in decision.currentness)
            != retained.package.source_ids
            or actual_integrity != expected_integrity
        ):
            raise EditorialError("editorial decision binding differs")
        return decision

    def current_copy_decision(self, decision: EditorialPolicyDecision) -> EditorialPolicyDecision:
        """Rebind only an explicitly compatible immutable article decision."""
        if type(decision) is not EditorialPolicyDecision or decision.policy_bundle_digest not in (
            self._policy_bundle_digest, *self._retained_policy_bundles
        ):
            raise EditorialError("unknown retained editorial policy")
        if decision.policy_bundle_digest == self._policy_bundle_digest:
            return decision
        values = {name: getattr(decision, name) for name in decision.__dataclass_fields__
                  if name != "decision_id"}
        return EditorialPolicyDecision.create(**{**values, "policy_bundle_digest": self._policy_bundle_digest})

    def _verify_story_event(
        self, receipt: StoryVersionReceipt, *, proof: AuthenticationProof
    ) -> None:
        provenance = self._events.provenance(receipt.event_id, proof=proof)
        self._verify_event(
            provenance,
            admission_id=receipt.admission_id,
            command=STORY_COMMAND,
            event=STORY_EVENT,
            definition_digest=self._story_definition,
            producer_principal=self._story_principal,
        )
        event = provenance.event
        if (
            event.command_id != receipt.command_id
            or event.aggregate_id != str(receipt.story_id)
            or event.aggregate_version != receipt.aggregate_version
            or event.payload_digest != receipt.story_version_digest
        ):
            raise EditorialError("Story Version event binding differs")

    def _verify_event(
        self,
        provenance,
        *,
        admission_id: ObjectAdmissionId,
        command: str,
        event: str,
        definition_digest: str,
        producer_principal: str,
    ) -> None:
        retained = provenance.event
        if (
            provenance.command_definition.command_type != command
            or provenance.command_definition.definition_digest != definition_digest
            or retained.command_definition_digest != definition_digest
            or retained.event_type != event
            or retained.object_admission_id != str(admission_id)
            or retained.principal_id != producer_principal
            or provenance.authentication.principal_id != producer_principal
            or provenance.authentication.authority_domain != self._authority_domain
            or retained.trust_scope != TrustScope.ADMITTED.value
            or retained.security_scope != EDITORIAL_SECURITY_SCOPE
            or retained.retention_scope != EDITORIAL_RETENTION_SCOPE
        ):
            raise EditorialError("editorial authority event differs")

    def _verify_access(
        self, decision, *, policy: str, object_class: str, allowed_use: str
    ) -> None:
        if (
            decision.policy_contract_digest != policy
            or decision.principal_id != self._reader_principal
            or decision.authority_domain != self._authority_domain
            or decision.object_class != object_class
            or decision.allowed_use != allowed_use
        ):
            raise EditorialError("editorial object access differs")


def _document(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EditorialError("editorial object is malformed") from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise EditorialError("editorial object is non-canonical")
    return value


def _text_values(*values: object) -> None:
    if any(type(value) is not str or not value for value in values):
        raise EditorialError("editorial text value differs")


def _text_fields(instance: object, *fields: str) -> None:
    _text_values(*(getattr(instance, field) for field in fields))


def _token_fields(instance: object, *fields: str) -> None:
    values = tuple(getattr(instance, field) for field in fields)
    _text_values(*values)
    if any(any(character.isspace() for character in value) for value in values):
        raise EditorialError("editorial token differs")


def _utc_timestamp(value: str) -> UtcTimestamp:
    try:
        return UtcTimestamp.parse(value)
    except ValueError as exc:
        raise EditorialError("editorial timestamp differs") from exc


def _digests(*values: object) -> None:
    try:
        for value in values:
            if validate_sha256_digest(value, field="editorial_digest") != value:
                raise EditorialError("editorial digest differs")
    except (TypeError, ValueError) as exc:
        raise EditorialError("editorial digest differs") from exc


__all__ = [
    "DECISION_CLASS",
    "DECISION_ADMISSION_TYPE",
    "DECISION_COMMAND",
    "DECISION_EVENT",
    "DECISION_PURPOSE",
    "DECISION_SCHEMA",
    "DECISION_USE",
    "EDITORIAL_RETENTION_SCOPE",
    "EDITORIAL_SECURITY_SCOPE",
    "DecisionReference",
    "EditorialError",
    "EditorialHold",
    "EditorialPolicyDecision",
    "NativeEditorial",
    "STORY_ADMISSION_TYPE",
    "STORY_CLASS",
    "STORY_COMMAND",
    "STORY_EVENT",
    "STORY_PURPOSE",
    "STORY_USE",
    "STORY_VERSION_SCHEMA",
    "SourceCurrentness",
    "SourceIntegrity",
    "StoryVersion",
    "StoryVersionReceipt",
    "StoryVersionRequest",
]
