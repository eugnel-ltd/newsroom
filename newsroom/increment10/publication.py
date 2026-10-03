"""Offline publication decision transaction; no dispatch or public effect."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

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
from newsroom.increment6.candidates import StoryCandidateReadPort

from .editorial import NativeEditorial, StoryVersion, StoryVersionReceipt
from .evidence import GovernedEvidencePackages

PUBLICATION_SCHEMA = "newsroom.increment10.publication-transaction.v1"
SURFACE_SCHEMA = "newsroom.increment10.surface-payload.v1"
SURFACE_ADMISSION_TYPE = "publication.surface-payload"
SURFACE_CLASS = "surface_payload"
SURFACE_USE = "publication_candidate"
SURFACE_PURPOSE = "publication.surface-payload"
TRANSACTION_ADMISSION_TYPE = "publication.transaction"
TRANSACTION_CLASS = "publication_transaction"
TRANSACTION_USE = "publication_authority"
TRANSACTION_PURPOSE = "publication.transaction"
PUBLICATION_COMMAND = "publication.transaction.commit"
PUBLICATION_EVENT = "publication.transaction.committed"
PUBLICATION_SECURITY_SCOPE = "authority.internal"
PUBLICATION_RETENTION_SCOPE = "editorial.retained"
AUTOMATED_NEWSROOM_IDENTITY = "NEWSROOM_AUTOMATED_EDITORIAL"
CONTENT_LANGUAGE = "zh-Hant-HK"
LAUNCH_CAPABILITIES = (
    "CREATE",
    "DELETE",
    "EDIT",
    "OBSERVE",
    "QUERY_BY_IDEMPOTENCY_KEY",
    "TOMBSTONE",
)
_SURFACE_KINDS = ("ARTICLE", "FEED_CARD")


class PublicationError(ValueError):
    """Raised when exact offline publication authority cannot be established."""


@dataclass(frozen=True, slots=True)
class PublicationRequest:
    publication_id: AggregateId
    expected_aggregate_version: int
    idempotency_key: str
    outcome: Literal["AUTO_PUBLISH", "HOLD_FOR_REVIEW"]
    reason_codes: tuple[str, ...]
    decided_at: str

    def __post_init__(self) -> None:
        if type(self.publication_id) is not AggregateId:
            raise PublicationError("publication aggregate identity is required")
        if (
            type(self.expected_aggregate_version) is not int
            or self.expected_aggregate_version < 0
        ):
            raise PublicationError("publication aggregate version differs")
        _text(self.idempotency_key, self.decided_at)
        UtcTimestamp.parse(self.decided_at)
        if self.outcome not in {"AUTO_PUBLISH", "HOLD_FOR_REVIEW"}:
            raise PublicationError("publication outcome differs")
        if not self.reason_codes or self.reason_codes != tuple(
            sorted(set(self.reason_codes))
        ):
            raise PublicationError("publication reasons must be sorted and unique")
        _text(*self.reason_codes)


@dataclass(frozen=True, slots=True)
class SurfacePayload:
    payload_id: str
    kind: Literal["ARTICLE", "FEED_CARD"]
    story_id: str
    story_aggregate_version: int
    story_version_digest: str
    headline: str
    body: str
    geography: tuple[str, ...]
    categories: tuple[str, ...]
    source_references: tuple[tuple[str, str], ...]
    # Claim ID, field, exact assertion, materiality.
    claim_links: tuple[tuple[str, str, str, str], ...]
    newsroom_identity: str
    correction_status: Literal["ORIGINAL", "CORRECTED"]
    withdrawal_status: Literal["ACTIVE"]
    content_language: str
    renderer_version: str
    licence_attributions: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in _SURFACE_KINDS:
            raise PublicationError("surface kind differs")
        _text(
            self.story_id,
            self.headline,
            self.newsroom_identity,
            self.content_language,
            self.renderer_version,
        )
        if self.kind == "ARTICLE" and not self.body:
            raise PublicationError("article body is required")
        if self.kind == "FEED_CARD" and self.body:
            raise PublicationError("feed-card body must be empty")
        _digest(self.story_version_digest)
        if (
            not self.geography
            or not self.categories
            or not self.source_references
            or not self.claim_links
            or self.newsroom_identity != AUTOMATED_NEWSROOM_IDENTITY
            or self.correction_status not in {"ORIGINAL", "CORRECTED"}
            or (self.correction_status == "CORRECTED" and self.story_aggregate_version < 2)
            or self.withdrawal_status != "ACTIVE"
            or self.content_language != CONTENT_LANGUAGE
        ):
            raise PublicationError("surface publication fields differ")
        if (type(self.licence_attributions) is not tuple
            or any(type(item) is not tuple or len(item) != 3
                   or any(type(value) is not str for value in item)
                   for item in self.licence_attributions)
            or self.licence_attributions != tuple(sorted(set(self.licence_attributions)))):
            raise PublicationError("surface licence attributions differ")
        for source_id, statement, url in self.licence_attributions:
            _text(source_id, statement, url)
            if source_id not in {item[0] for item in self.source_references} or urlsplit(url).scheme != "https":
                raise PublicationError("surface licence source binding differs")
        if self.payload_id != digest_bytes(
            canonical_json_bytes(self.value(include_identity=False))
        ):
            raise PublicationError("surface identity differs")

    def value(self, *, include_identity: bool = True) -> dict[str, object]:
        value = {
            "schema_identity": SURFACE_SCHEMA,
            "kind": self.kind,
            "story_id": self.story_id,
            "story_aggregate_version": self.story_aggregate_version,
            "story_version_digest": self.story_version_digest,
            "headline": self.headline,
            "body": self.body,
            "geography": list(self.geography),
            "categories": list(self.categories),
            "source_references": [list(item) for item in self.source_references],
            "claim_links": [list(item) for item in self.claim_links],
            "newsroom_identity": self.newsroom_identity,
            "correction_status": self.correction_status,
            "withdrawal_status": self.withdrawal_status,
            "content_language": self.content_language,
            "renderer_version": self.renderer_version,
        }
        if self.licence_attributions:
            value["licence_attributions"] = [list(item) for item in self.licence_attributions]
        if include_identity:
            value["payload_id"] = self.payload_id
        return value

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.value())

    @classmethod
    def create(cls, **values) -> SurfacePayload:
        if not values.get("licence_attributions"):
            values.pop("licence_attributions", None)
        identity = {"schema_identity": SURFACE_SCHEMA, **_surface_value(values)}
        return cls(payload_id=digest_bytes(canonical_json_bytes(identity)), **values)


@dataclass(frozen=True, slots=True)
class PublicationBundle:
    bundle_id: str
    story_event_id: str
    story_id: str
    story_aggregate_version: int
    story_version_digest: str
    evidence_package_admission_id: str
    retained_evidence_package_digest: str
    admission_input_digest: str
    validator_digest: str
    target_id: str
    target_policy_digest: str
    surface_payloads: tuple[tuple[str, ObjectAdmissionId, str, str], ...]
    claim_surface_manifest: tuple[tuple[str, str, str, str, str], ...]

    def __post_init__(self) -> None:
        _text(self.story_event_id, self.story_id, self.target_id)
        EventId.parse(self.story_event_id)
        _digests(
            self.story_version_digest,
            self.retained_evidence_package_digest,
            self.admission_input_digest,
            self.validator_digest,
            self.target_policy_digest,
        )
        ObjectAdmissionId.parse(self.evidence_package_admission_id)
        if tuple(item[0] for item in self.surface_payloads) != _SURFACE_KINDS:
            raise PublicationError("bundle requires exact article and feed-card")
        if not self.claim_surface_manifest:
            raise PublicationError("bundle claim-to-surface manifest is required")
        if self.bundle_id != digest_bytes(
            canonical_json_bytes(self.value(include_identity=False))
        ):
            raise PublicationError("bundle identity differs")

    def value(self, *, include_identity: bool = True) -> dict[str, object]:
        value = {
            "story_event_id": self.story_event_id,
            "story_id": self.story_id,
            "story_aggregate_version": self.story_aggregate_version,
            "story_version_digest": self.story_version_digest,
            "evidence_package_admission_id": self.evidence_package_admission_id,
            "retained_evidence_package_digest": self.retained_evidence_package_digest,
            "admission_input_digest": self.admission_input_digest,
            "validator_digest": self.validator_digest,
            "target_id": self.target_id,
            "target_policy_digest": self.target_policy_digest,
            "surface_payloads": [
                [kind, str(admission_id), payload_id, digest]
                for kind, admission_id, payload_id, digest in self.surface_payloads
            ],
            "claim_surface_manifest": [
                list(item) for item in self.claim_surface_manifest
            ],
        }
        if include_identity:
            value["bundle_id"] = self.bundle_id
        return value


@dataclass(frozen=True, slots=True)
class PublicationDecision:
    decision_id: str
    outcome: Literal["AUTO_PUBLISH", "HOLD_FOR_REVIEW"]
    reason_codes: tuple[str, ...]
    story_event_id: str
    story_version_digest: str
    bundle_id: str | None
    authorisation_policy_digest: str
    target_policy_digest: str
    controller_principal_id: str
    decided_at: str

    def __post_init__(self) -> None:
        if (self.outcome == "AUTO_PUBLISH") != (self.bundle_id is not None):
            raise PublicationError("publication decision bundle binding differs")
        _text(
            self.story_event_id,
            self.controller_principal_id,
            self.decided_at,
            *self.reason_codes,
        )
        EventId.parse(self.story_event_id)
        UtcTimestamp.parse(self.decided_at)
        _digests(
            self.story_version_digest,
            self.authorisation_policy_digest,
            self.target_policy_digest,
        )
        if self.decision_id != digest_bytes(
            canonical_json_bytes(self.value(include_identity=False))
        ):
            raise PublicationError("publication decision identity differs")

    def value(self, *, include_identity: bool = True) -> dict[str, object]:
        value = {
            "outcome": self.outcome,
            "reason_codes": list(self.reason_codes),
            "story_event_id": self.story_event_id,
            "story_version_digest": self.story_version_digest,
            "bundle_id": self.bundle_id,
            "authorisation_policy_digest": self.authorisation_policy_digest,
            "target_policy_digest": self.target_policy_digest,
            "controller_principal_id": self.controller_principal_id,
            "decided_at": self.decided_at,
        }
        if include_identity:
            value["decision_id"] = self.decision_id
        return value


@dataclass(frozen=True, slots=True)
class TargetOperation:
    operation_id: str
    target_id: str
    bundle_id: str
    surface_kind: str
    surface_payload_id: str
    surface_admission_id: ObjectAdmissionId
    surface_digest: str
    action: Literal["CREATE"]
    state: Literal["PENDING"]
    semantic_idempotency_key: str

    def __post_init__(self) -> None:
        _text(
            self.target_id,
            self.bundle_id,
            self.surface_kind,
            self.surface_payload_id,
            self.semantic_idempotency_key,
        )
        _digest(self.surface_digest)
        if self.action != "CREATE" or self.state != "PENDING":
            raise PublicationError("target operation state differs")
        if self.operation_id != digest_bytes(
            canonical_json_bytes(self.value(include_identity=False))
        ):
            raise PublicationError("target operation identity differs")

    def value(self, *, include_identity: bool = True) -> dict[str, object]:
        value = {
            "target_id": self.target_id,
            "bundle_id": self.bundle_id,
            "surface_kind": self.surface_kind,
            "surface_payload_id": self.surface_payload_id,
            "surface_admission_id": str(self.surface_admission_id),
            "surface_digest": self.surface_digest,
            "action": self.action,
            "state": self.state,
            "semantic_idempotency_key": self.semantic_idempotency_key,
        }
        if include_identity:
            value["operation_id"] = self.operation_id
        return value


@dataclass(frozen=True, slots=True)
class PublicationTransaction:
    transaction_id: str
    publication_id: str
    aggregate_version: int
    decision: PublicationDecision
    audit: tuple[tuple[str, str], ...]
    bundle: PublicationBundle | None
    operations: tuple[TargetOperation, ...]

    def __post_init__(self) -> None:
        AggregateId.parse(self.publication_id)
        if type(self.audit) is not tuple or any(
            type(item) is not tuple or len(item) != 2
            or any(type(value) is not str for value in item)
            for item in self.audit
        ):
            raise PublicationError("publication audit must be immutable text pairs")
        if self.audit != tuple(sorted(dict(self.audit).items())):
            raise PublicationError("publication audit keys differ")
        if (self.bundle is None) != (not self.operations):
            raise PublicationError("publication transaction records differ")
        if self.bundle is not None and len(self.operations) != 2:
            raise PublicationError("publication transaction operations differ")
        if self.transaction_id != digest_bytes(
            canonical_json_bytes(self.value(include_identity=False))
        ):
            raise PublicationError("publication transaction identity differs")

    def value(self, *, include_identity: bool = True) -> dict[str, object]:
        value = {
            "schema_identity": PUBLICATION_SCHEMA,
            "publication_id": self.publication_id,
            "aggregate_version": self.aggregate_version,
            "decision": self.decision.value(),
            "audit": dict(self.audit),
            "bundle": None if self.bundle is None else self.bundle.value(),
            "operations": [item.value() for item in self.operations],
        }
        if include_identity:
            value["transaction_id"] = self.transaction_id
        return value

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.value())


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    command_id: str
    event_id: str
    publication_id: AggregateId
    aggregate_version: int
    admission_id: ObjectAdmissionId
    transaction_digest: str
    transaction_id: str
    decision_id: str
    bundle_id: str | None
    operation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        CommandId.parse(self.command_id)
        EventId.parse(self.event_id)
        if type(self.publication_id) is not AggregateId:
            raise PublicationError("publication receipt aggregate differs")
        if type(self.admission_id) is not ObjectAdmissionId:
            raise PublicationError("publication receipt admission differs")
        _digest(self.transaction_digest)


class OfflinePublication:
    """Trusted server facade for one offline publication transaction.

    Raw controller-scoped command authority must remain private to composition:
    typed readback revalidates the complete transaction before exposing records.
    """

    def __init__(
        self,
        *,
        objects: GovernedObjects,
        commands: AuthorityCommands,
        events: AuthorityEvents,
        editorial: NativeEditorial,
        evidence: GovernedEvidencePackages,
        reader_principal_id: str,
        authority_domain: str,
        controller_principal_id: str,
        authorisation_policy_digest: str,
        target_id: str,
        target_policy_digest: str,
        target_capabilities: tuple[str, ...],
        surface_hydration_policy_digest: str,
        transaction_hydration_policy_digest: str,
        surface_admission_definition_digest: str,
        transaction_admission_definition_digest: str,
        command_definition_digest: str,
        source_licence_policy: tuple[tuple[str, str, str], ...] = (),
    ) -> None:
        if not all(
            type(value) is expected
            for value, expected in (
                (objects, GovernedObjects),
                (commands, AuthorityCommands),
                (events, AuthorityEvents),
                (editorial, NativeEditorial),
                (evidence, GovernedEvidencePackages),
            )
        ):
            raise PublicationError("exact publication authorities are required")
        _text(reader_principal_id, authority_domain, controller_principal_id, target_id)
        _digests(
            authorisation_policy_digest,
            target_policy_digest,
            surface_hydration_policy_digest,
            transaction_hydration_policy_digest,
            surface_admission_definition_digest,
            transaction_admission_definition_digest,
            command_definition_digest,
        )
        if target_capabilities != LAUNCH_CAPABILITIES:
            raise PublicationError("launch target capabilities differ")
        if type(source_licence_policy) is not tuple or any(
            type(item) is not tuple or len(item) != 3
            or any(type(value) is not str or not value for value in item)
            or urlsplit(item[2]).scheme != "https"
            for item in source_licence_policy
        ):
            raise PublicationError("source licence policy differs")
        self._source_licence_policy = source_licence_policy
        self._objects = objects
        self._commands = commands
        self._events = events
        self._editorial = editorial
        self._evidence = evidence
        self._reader = reader_principal_id
        self._domain = authority_domain
        self._controller = controller_principal_id
        self._authorisation_policy = authorisation_policy_digest
        self._target = target_id
        self._target_policy = target_policy_digest
        self._surface_policy = surface_hydration_policy_digest
        self._transaction_policy = transaction_hydration_policy_digest
        self._surface_definition = surface_admission_definition_digest
        self._transaction_definition = transaction_admission_definition_digest
        self._command_definition = command_definition_digest

    def decide(
        self,
        request: PublicationRequest,
        *,
        story_receipt: StoryVersionReceipt,
        candidate_port: StoryCandidateReadPort,
        proof: AuthenticationProof,
    ) -> tuple[PublicationReceipt, PublicationTransaction]:
        if type(request) is not PublicationRequest:
            raise PublicationError("exact PublicationRequest is required")
        story, sources = self._context(story_receipt, candidate_port, proof)
        correction = "NATIVE_COPY_CORRECTION" in request.reason_codes
        if correction and (request.expected_aggregate_version < 1 or story.aggregate_version < 2):
            raise PublicationError("copy correction requires a prior Story and publication")
        surfaces: tuple[SurfacePayload, ...] = ()
        admissions: tuple[ObjectAdmissionId, ...] = ()
        if request.outcome == "AUTO_PUBLISH":
            surfaces = _render(story, sources, self._source_licence_policy, correction=correction)
            admissions = tuple(
                self._admit_surface(surface, proof=proof) for surface in surfaces
            )
        transaction = self._transaction(
            request, story_receipt, story, surfaces, admissions
        )
        raw = transaction.canonical_bytes()
        admitted = self._objects.admit(
            ObjectAdmissionRequest(
                TRANSACTION_ADMISSION_TYPE,
                f"publication:{request.publication_id}:"
                f"{request.expected_aggregate_version + 1}",
            ),
            raw,
            proof=proof,
        ).admission
        if (
            admitted.definition_digest != self._transaction_definition
            or admitted.object_class != TRANSACTION_CLASS
            or admitted.allowed_use != TRANSACTION_USE
            or admitted.blob.blob_digest != digest_bytes(raw)
            or not admitted.active
        ):
            raise PublicationError("publication transaction admission differs")
        committed = self._commands.execute(
            SemanticCommand(
                PUBLICATION_COMMAND,
                request.publication_id,
                request.expected_aggregate_version,
                ObjectAdmissionPayload(admitted.admission_id),
                request.idempotency_key,
            ),
            proof=proof,
        )
        receipt = PublicationReceipt(
            committed.command_id,
            committed.event_id,
            request.publication_id,
            committed.aggregate_version,
            admitted.admission_id,
            digest_bytes(raw),
            transaction.transaction_id,
            transaction.decision.decision_id,
            None if transaction.bundle is None else transaction.bundle.bundle_id,
            tuple(item.operation_id for item in transaction.operations),
        )
        self._verify_event(receipt, proof=proof)
        return receipt, transaction

    def read(
        self,
        receipt: PublicationReceipt,
        *,
        story_receipt: StoryVersionReceipt,
        candidate_port: StoryCandidateReadPort,
        proof: AuthenticationProof,
    ) -> PublicationTransaction:
        if type(receipt) is not PublicationReceipt:
            raise PublicationError("exact PublicationReceipt is required")
        self._verify_event(receipt, proof=proof)
        hydrated = self._objects.hydrate(
            HydrationRequest(receipt.admission_id, TRANSACTION_PURPOSE), proof=proof
        )
        self._access(
            hydrated.decision,
            self._transaction_policy,
            TRANSACTION_CLASS,
            TRANSACTION_USE,
        )
        value = _document(hydrated.data)
        transaction = _transaction_from_value(value)
        if (
            transaction.transaction_id != receipt.transaction_id
            or transaction.decision.decision_id != receipt.decision_id
            or (None if transaction.bundle is None else transaction.bundle.bundle_id)
            != receipt.bundle_id
            or tuple(item.operation_id for item in transaction.operations)
            != receipt.operation_ids
        ):
            raise PublicationError("publication receipt records differ")
        story, sources = self._context(story_receipt, candidate_port, proof)
        request = PublicationRequest(
            receipt.publication_id,
            receipt.aggregate_version - 1,
            "readback",
            transaction.decision.outcome,
            transaction.decision.reason_codes,
            transaction.decision.decided_at,
        )
        surfaces: tuple[SurfacePayload, ...] = ()
        admissions: tuple[ObjectAdmissionId, ...] = ()
        if transaction.bundle is not None:
            surfaces = _render(
                story, sources, self._source_licence_policy,
                correction="NATIVE_COPY_CORRECTION" in transaction.decision.reason_codes,
            )
            admissions = tuple(item[1] for item in transaction.bundle.surface_payloads)
            for surface, admission_id in zip(surfaces, admissions, strict=True):
                material = self._objects.hydrate(
                    HydrationRequest(admission_id, SURFACE_PURPOSE), proof=proof
                )
                self._access(
                    material.decision,
                    self._surface_policy,
                    SURFACE_CLASS,
                    SURFACE_USE,
                )
                if material.data != surface.canonical_bytes():
                    raise PublicationError("surface payload replay differs")
        rebuilt = self._transaction(
            request, story_receipt, story, surfaces, admissions
        )
        if rebuilt.canonical_bytes() != hydrated.data:
            raise PublicationError("publication transaction replay differs")
        return transaction

    def _context(self, receipt, candidate_port, proof):
        story = self._editorial.read_story_version(
            receipt, candidate_port=candidate_port, proof=proof
        )
        retained = self._evidence.read(
            story.package_admission_id, candidate_port=candidate_port, proof=proof
        )
        if retained.package.digest != story.retained_package_digest:
            raise PublicationError("Story Version Evidence Package differs")
        return story, retained.source_inventory

    def _admit_surface(
        self, surface: SurfacePayload, *, proof: AuthenticationProof
    ) -> ObjectAdmissionId:
        raw = surface.canonical_bytes()
        admitted = self._objects.admit(
            ObjectAdmissionRequest(SURFACE_ADMISSION_TYPE, surface.payload_id),
            raw,
            proof=proof,
        ).admission
        if (
            admitted.definition_digest != self._surface_definition
            or admitted.object_class != SURFACE_CLASS
            or admitted.allowed_use != SURFACE_USE
            or admitted.blob.blob_digest != digest_bytes(raw)
            or not admitted.active
        ):
            raise PublicationError("surface payload admission differs")
        return admitted.admission_id

    def _transaction(self, request, receipt, story, surfaces, admissions):
        validator_digest = digest_bytes(
            canonical_json_bytes(
                [
                    [item.validator, item.result, item.reason_code]
                    for item in story.validators
                ]
            )
        )
        bundle = None
        operations: tuple[TargetOperation, ...] = ()
        if request.outcome == "AUTO_PUBLISH":
            surface_rows = tuple(
                (
                    surface.kind,
                    admission,
                    surface.payload_id,
                    digest_bytes(surface.canonical_bytes()),
                )
                for surface, admission in zip(surfaces, admissions, strict=True)
            )
            manifest = tuple(
                (surface.payload_id, *link)
                for surface in surfaces
                for link in surface.claim_links
            )
            bundle_values = {
                "story_event_id": receipt.event_id,
                "story_id": str(story.story_id),
                "story_aggregate_version": story.aggregate_version,
                "story_version_digest": story.digest,
                "evidence_package_admission_id": str(story.package_admission_id),
                "retained_evidence_package_digest": story.retained_package_digest,
                "admission_input_digest": story.admission_input_digest,
                "validator_digest": validator_digest,
                "target_id": self._target,
                "target_policy_digest": self._target_policy,
                "surface_payloads": surface_rows,
                "claim_surface_manifest": manifest,
            }
            bundle = PublicationBundle(
                bundle_id=digest_bytes(
                    canonical_json_bytes(_bundle_value(bundle_values))
                ),
                **bundle_values,
            )
            operations = tuple(
                _operation(self._target, bundle, surface, admission)
                for surface, admission in zip(surfaces, admissions, strict=True)
            )
        decision_values = {
            "outcome": request.outcome,
            "reason_codes": request.reason_codes,
            "story_event_id": receipt.event_id,
            "story_version_digest": story.digest,
            "bundle_id": None if bundle is None else bundle.bundle_id,
            "authorisation_policy_digest": self._authorisation_policy,
            "target_policy_digest": self._target_policy,
            "controller_principal_id": self._controller,
            "decided_at": request.decided_at,
        }
        decision = PublicationDecision(
            decision_id=digest_bytes(
                canonical_json_bytes(_decision_value(decision_values))
            ),
            **decision_values,
        )
        audit = {
            "decision_id": decision.decision_id,
            "story_event_id": receipt.event_id,
            "story_version_digest": story.digest,
            "validator_digest": validator_digest,
            "authorisation_policy_digest": self._authorisation_policy,
            "target_policy_digest": self._target_policy,
            "controller_principal_id": self._controller,
            "outcome": request.outcome,
        }
        values = {
            "publication_id": str(request.publication_id),
            "aggregate_version": request.expected_aggregate_version + 1,
            "decision": decision,
            "audit": tuple(sorted(audit.items())),
            "bundle": bundle,
            "operations": operations,
        }
        return PublicationTransaction(
            transaction_id=digest_bytes(
                canonical_json_bytes(
                    {
                        "schema_identity": PUBLICATION_SCHEMA,
                        "publication_id": values["publication_id"],
                        "aggregate_version": values["aggregate_version"],
                        "decision": decision.value(),
                        "audit": audit,
                        "bundle": None if bundle is None else bundle.value(),
                        "operations": [item.value() for item in operations],
                    }
                )
            ),
            **values,
        )

    def _verify_event(self, receipt: PublicationReceipt, *, proof) -> None:
        provenance = self._events.provenance(receipt.event_id, proof=proof)
        event = provenance.event
        if (
            provenance.command_definition.command_type != PUBLICATION_COMMAND
            or provenance.command_definition.definition_digest
            != self._command_definition
            or event.command_definition_digest != self._command_definition
            or event.event_type != PUBLICATION_EVENT
            or event.object_admission_id != str(receipt.admission_id)
            or event.payload_digest != receipt.transaction_digest
            or event.command_id != receipt.command_id
            or event.aggregate_id != str(receipt.publication_id)
            or event.aggregate_version != receipt.aggregate_version
            or event.principal_id != self._controller
            or provenance.authentication.principal_id != self._controller
            or provenance.authentication.authority_domain != self._domain
            or event.trust_scope != TrustScope.ADMITTED.value
            or event.security_scope != PUBLICATION_SECURITY_SCOPE
            or event.retention_scope != PUBLICATION_RETENTION_SCOPE
        ):
            raise PublicationError("publication authority event differs")

    def _access(self, decision, policy, object_class, allowed_use):
        if (
            decision.policy_contract_digest != policy
            or decision.principal_id != self._reader
            or decision.authority_domain != self._domain
            or decision.object_class != object_class
            or decision.allowed_use != allowed_use
        ):
            raise PublicationError("publication object access differs")


def _render(
    story: StoryVersion, sources: tuple[tuple[str, str], ...],
    source_licence_policy: tuple[tuple[str, str, str], ...] = (),
    *, correction: bool = False,
) -> tuple[SurfacePayload, SurfacePayload]:
    prefix = "【未出版】"
    narrative = story.copy.writer_id == "newsroom.native-story-writer.v1"
    if not narrative and (not story.copy.title.startswith(prefix) or story.copy.title.count(prefix) != 1):
        raise PublicationError("Story Version lacks exact unpublished presentation")
    headline = story.copy.title if narrative else story.copy.title.removeprefix(prefix)
    common = {
        "story_id": str(story.story_id),
        "story_aggregate_version": story.aggregate_version,
        "story_version_digest": story.digest,
        "headline": headline,
        "geography": story.write_admission.geography,
        "categories": story.write_admission.categories,
        "source_references": sources,
        "newsroom_identity": AUTOMATED_NEWSROOM_IDENTITY,
        "correction_status": "CORRECTED" if correction else "ORIGINAL",
        "withdrawal_status": "ACTIVE",
        "content_language": CONTENT_LANGUAGE,
        "renderer_version": "newsroom.public-surface.v1",
    }
    attributions = tuple(sorted({
        (source_id, statement, licence_url)
        for source_id, source_url in sources
        for host, statement, licence_url in source_licence_policy
        if urlsplit(source_url).scheme == "https" and urlsplit(source_url).netloc == host
    }))
    if attributions:
        common["licence_attributions"] = attributions
        common["renderer_version"] = "newsroom.private-licensed-surface.v1"
    article_links = tuple(
        (
            item.governed_claim_id,
            "HEADLINE" if item.rendered_assertion in headline else "BODY",
            item.rendered_assertion,
            # The admitted copy contains only HEADLINE and SUBSTANTIVE claims;
            # neither is optional context for correction impact (PUBENG-004).
            "MATERIAL",
        )
        for item in story.copy.evidence_links
    )
    if story.copy.writer_id == "newsroom.offline-exact-copy.v3":
        # v3 deliberately retains the headline's source context in the body.
        article_links += tuple(
            (claim_id, "BODY", assertion, materiality)
            for claim_id, field, assertion, materiality in article_links
            if field == "HEADLINE"
        )
        common["renderer_version"] = "newsroom.context-preserving-surface.v1"
    if narrative:
        common["renderer_version"] = ("newsroom.source-grounded-brief.v1" if story.story_format == "BRIEF"
                                      else "newsroom.source-grounded-report.v1")
    feed_links = tuple(item for item in article_links if item[1] == "HEADLINE")
    return (
        SurfacePayload.create(
            kind="ARTICLE",
            body=story.copy.body,
            claim_links=article_links,
            **common,
        ),
        SurfacePayload.create(
            kind="FEED_CARD",
            body="",
            claim_links=feed_links,
            **common,
        ),
    )


def _operation(target, bundle, surface, admission):
    values = {
        "target_id": target,
        "bundle_id": bundle.bundle_id,
        "surface_kind": surface.kind,
        "surface_payload_id": surface.payload_id,
        "surface_admission_id": admission,
        "surface_digest": digest_bytes(surface.canonical_bytes()),
        "action": "CREATE",
        "state": "PENDING",
        "semantic_idempotency_key": (
            f"story:{bundle.story_id}:v{bundle.story_aggregate_version}:"
            f"target:{target}:surface:{surface.kind.lower()}"
        ),
    }
    return TargetOperation(
        operation_id=digest_bytes(canonical_json_bytes(_operation_value(values))),
        **values,
    )


def _transaction_from_value(value: dict[str, object]) -> PublicationTransaction:
    try:
        if type(value["audit"]) is not dict:
            raise PublicationError("publication audit fields differ")
        raw_decision = value["decision"]
        decision = PublicationDecision(
            **{**raw_decision, "reason_codes": tuple(raw_decision["reason_codes"])}
        )
        raw_bundle = value["bundle"]
        bundle = None
        if raw_bundle is not None:
            bundle = PublicationBundle(
                **{
                    **raw_bundle,
                    "surface_payloads": tuple(
                        (
                            item[0],
                            ObjectAdmissionId.parse(item[1]),
                            item[2],
                            item[3],
                        )
                        for item in raw_bundle["surface_payloads"]
                    ),
                    "claim_surface_manifest": tuple(
                        tuple(item) for item in raw_bundle["claim_surface_manifest"]
                    ),
                }
            )
        transaction = PublicationTransaction(
            transaction_id=value["transaction_id"],
            publication_id=value["publication_id"],
            aggregate_version=value["aggregate_version"],
            decision=decision,
            audit=tuple(sorted(value["audit"].items())),
            bundle=bundle,
            operations=tuple(
                TargetOperation(
                    **{
                        **item,
                        "surface_admission_id": ObjectAdmissionId.parse(
                            item["surface_admission_id"]
                        ),
                    }
                )
                for item in value["operations"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PublicationError("publication transaction fields differ") from exc
    if value.get("schema_identity") != PUBLICATION_SCHEMA:
        raise PublicationError("publication transaction schema differs")
    return transaction


def _surface_value(values):
    return {
        **values,
        "geography": list(values["geography"]),
        "categories": list(values["categories"]),
        "source_references": [list(item) for item in values["source_references"]],
        "claim_links": [list(item) for item in values["claim_links"]],
    }


def _bundle_value(values):
    return {
        **values,
        "surface_payloads": [
            [kind, str(admission), payload_id, digest]
            for kind, admission, payload_id, digest in values["surface_payloads"]
        ],
        "claim_surface_manifest": [
            list(item) for item in values["claim_surface_manifest"]
        ],
    }


def _decision_value(values):
    return {**values, "reason_codes": list(values["reason_codes"])}


def _operation_value(values):
    return {**values, "surface_admission_id": str(values["surface_admission_id"])}


def _document(raw: bytes) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationError("publication object is malformed") from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise PublicationError("publication object is non-canonical")
    return value


def _text(*values) -> None:
    if any(type(value) is not str or not value for value in values):
        raise PublicationError("publication text field differs")


def _digest(value) -> None:
    try:
        validate_sha256_digest(value)
    except (TypeError, ValueError) as exc:
        raise PublicationError("publication digest differs") from exc


def _digests(*values) -> None:
    for value in values:
        _digest(value)


__all__ = [
    "AUTOMATED_NEWSROOM_IDENTITY",
    "LAUNCH_CAPABILITIES",
    "OfflinePublication",
    "PUBLICATION_COMMAND",
    "PUBLICATION_EVENT",
    "PUBLICATION_RETENTION_SCOPE",
    "PUBLICATION_SCHEMA",
    "PUBLICATION_SECURITY_SCOPE",
    "PublicationBundle",
    "PublicationDecision",
    "PublicationError",
    "PublicationReceipt",
    "PublicationRequest",
    "PublicationTransaction",
    "SURFACE_ADMISSION_TYPE",
    "SURFACE_CLASS",
    "SURFACE_PURPOSE",
    "SURFACE_SCHEMA",
    "SURFACE_USE",
    "SurfacePayload",
    "TRANSACTION_ADMISSION_TYPE",
    "TRANSACTION_CLASS",
    "TRANSACTION_PURPOSE",
    "TRANSACTION_USE",
    "TargetOperation",
]
