from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
    validate_sha256_digest,
)
from newsroom.authority.types import (
    AggregateId,
    EventId,
    TrustScope,
    UUIDv4Id,
    UtcTimestamp,
    require_scope,
    require_token,
)


class ProjectionContractError(ValueError):
    """Raised when a projection contract or request is malformed."""


class ProjectionStateError(RuntimeError):
    """Raised when an authoritative projection transition is invalid."""


class ProjectionAuthorizationError(PermissionError):
    """Raised when the projection facade rejects an authenticated operation."""


@dataclass(frozen=True, slots=True)
class ProjectionGenerationId(UUIDv4Id):
    def as_aggregate_id(self) -> AggregateId:
        return AggregateId(self.value)


@dataclass(frozen=True, slots=True)
class ProjectionGapId(UUIDv4Id):
    pass


@dataclass(frozen=True, slots=True)
class ProjectionDeadLetterId(UUIDv4Id):
    pass


@dataclass(frozen=True, slots=True)
class ProjectionDeliveryAttemptId(UUIDv4Id):
    pass


class ProjectionFamilyKind(StrEnum):
    GRAPH = "GRAPH"
    VECTOR = "VECTOR"
    FULL_TEXT = "FULL_TEXT"


class ProjectionGenerationState(StrEnum):
    BUILDING = "BUILDING"
    VALIDATING = "VALIDATING"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"
    FAILED = "FAILED"


class ProjectionDeliveryOutcome(StrEnum):
    APPLIED = "APPLIED"
    IGNORED_OPTIONAL = "IGNORED_OPTIONAL"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    REQUIRED_UNSUPPORTED = "REQUIRED_UNSUPPORTED"


class ProjectionGapState(StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"


class GraphitiWorkspaceMode(StrEnum):
    PROPOSAL_ONLY = "PROPOSAL_ONLY"


@dataclass(frozen=True, slots=True)
class ProjectionFamilyDefinition:
    family_id: str
    authority_aggregate_id: AggregateId
    family_kind: ProjectionFamilyKind
    definition_version: str
    projector_version: str
    ontology_contract_digest: str
    mapping_contract_digest: str
    complete_projection_contract_digest: str | None = None
    max_delivery_attempts: int = 3
    max_gap_span: int = 1000
    required_manage_scope: str = "authority.projection.manage"
    required_write_scope: str = "authority.projection.write"
    required_read_scope: str = "authority.projection.read"
    security_scope: str = "authority.projection"
    retention_scope: str = "authority.audit"

    def __post_init__(self) -> None:
        require_token(self.family_id, field="projection_family_id")
        if not isinstance(self.authority_aggregate_id, AggregateId):
            raise ProjectionContractError("family aggregate identity must be AggregateId")
        if not isinstance(self.family_kind, ProjectionFamilyKind):
            raise ProjectionContractError("projection family kind must be typed")
        require_token(self.definition_version, field="projection_family_version")
        require_token(self.projector_version, field="projector_version")
        for field, value in (
            ("ontology_contract_digest", self.ontology_contract_digest),
            ("mapping_contract_digest", self.mapping_contract_digest),
        ):
            normalized = validate_sha256_digest(value, field=field)
            if normalized != value:
                raise ProjectionContractError(f"{field} must be canonical lowercase")
        if self.complete_projection_contract_digest is not None:
            normalized = validate_sha256_digest(
                self.complete_projection_contract_digest,
                field="complete_projection_contract_digest",
            )
            if normalized != self.complete_projection_contract_digest:
                raise ProjectionContractError(
                    "complete projection contract digest must be canonical lowercase"
                )
            if self.family_kind is not ProjectionFamilyKind.GRAPH:
                raise ProjectionContractError(
                    "complete projection contracts require a GRAPH family"
                )
        for field, value in (
            ("max_delivery_attempts", self.max_delivery_attempts),
            ("max_gap_span", self.max_gap_span),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ProjectionContractError(f"{field} must be a positive integer")
        require_scope(self.required_manage_scope, field="required_manage_scope")
        require_scope(self.required_write_scope, field="required_write_scope")
        require_scope(self.required_read_scope, field="required_read_scope")
        require_scope(self.security_scope, field="security_scope")
        require_scope(self.retention_scope, field="retention_scope")

    def canonical_value(self) -> dict[str, object]:
        value: dict[str, object] = {
            "family_id": self.family_id,
            "authority_aggregate_id": str(self.authority_aggregate_id),
            "family_kind": self.family_kind.value,
            "definition_version": self.definition_version,
            "projector_version": self.projector_version,
            "ontology_contract_digest": self.ontology_contract_digest,
            "mapping_contract_digest": self.mapping_contract_digest,
            "max_delivery_attempts": self.max_delivery_attempts,
            "max_gap_span": self.max_gap_span,
            "required_manage_scope": self.required_manage_scope,
            "required_write_scope": self.required_write_scope,
            "required_read_scope": self.required_read_scope,
            "security_scope": self.security_scope,
            "retention_scope": self.retention_scope,
        }
        # Omit the optional key for pre-Increment-2B definitions so all retained
        # Increment 1 family digests remain byte-for-byte stable.
        if self.complete_projection_contract_digest is not None:
            value["complete_projection_contract_digest"] = (
                self.complete_projection_contract_digest
            )
        return value

    @property
    def digest(self) -> str:
        return digest_bytes(canonical_json_bytes(self.canonical_value()))


@dataclass(frozen=True, slots=True)
class GraphitiProposalWorkspaceContract:
    workspace_id: str
    contract_version: str
    implementation_version: str
    endpoint_reference: str
    secret_reference: str
    mode: GraphitiWorkspaceMode = GraphitiWorkspaceMode.PROPOSAL_ONLY

    def __post_init__(self) -> None:
        require_token(self.workspace_id, field="graphiti_workspace_id")
        require_token(self.contract_version, field="graphiti_contract_version")
        require_token(self.implementation_version, field="graphiti_implementation_version")
        for field_name, value in (("graphiti_endpoint_reference", self.endpoint_reference), ("graphiti_secret_reference", self.secret_reference)):
            if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > 512:
                raise ProjectionContractError(f"{field_name} must be a bounded non-empty reference")
        if not isinstance(self.mode, GraphitiWorkspaceMode):
            raise ProjectionContractError("Graphiti workspace mode must be typed")
        if self.mode is not GraphitiWorkspaceMode.PROPOSAL_ONLY:
            raise ProjectionContractError("Graphiti workspace must remain proposal-only")

    def canonical_value(self) -> dict[str, object]:
        return {
            "workspace_id": self.workspace_id,
            "contract_version": self.contract_version,
            "implementation_version": self.implementation_version,
            "endpoint_reference": self.endpoint_reference,
            "secret_reference": self.secret_reference,
            "mode": self.mode.value,
        }

    @property
    def contract_digest(self) -> str:
        return digest_bytes(canonical_json_bytes(self.canonical_value()))


@dataclass(frozen=True, slots=True)
class ProjectionFamilyView:
    family_id: str
    definition_digest: str
    authority_aggregate_id: AggregateId
    family_kind: ProjectionFamilyKind
    authority_aggregate_version: int
    registered_event_id: EventId
    created_at: UtcTimestamp


@dataclass(frozen=True, slots=True)
class ProjectionGenerationView:
    generation_id: ProjectionGenerationId
    family_id: str
    state: ProjectionGenerationState
    lifecycle_version: int
    authority_aggregate_version: int
    validated_through_ledger_seq: int | None
    created_at: UtcTimestamp
    updated_at: UtcTimestamp


@dataclass(frozen=True, slots=True)
class ProjectionGenerationValidationView:
    validation_digest: str
    generation_id: ProjectionGenerationId
    validation_version: int
    lifecycle_version: int
    checkpoint_ledger_seq: int
    definition_digest: str
    ontology_contract_digest: str
    mapping_contract_digest: str
    projector_version: str
    service_compatibility_digest: str
    projection_state_digest: str
    authority_aggregate_version: int
    authority_event_id: EventId
    recorded_at: UtcTimestamp
    source_snapshot_digest: str | None = None
    source_watermark_ledger_seq: int | None = None
    source_request_digest: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("validation_digest", self.validation_digest),
            ("definition_digest", self.definition_digest),
            ("ontology_contract_digest", self.ontology_contract_digest),
            ("mapping_contract_digest", self.mapping_contract_digest),
            ("service_compatibility_digest", self.service_compatibility_digest),
            ("projection_state_digest", self.projection_state_digest),
        ):
            validate_sha256_digest(value, field=field_name)
        source_values = (
            self.source_snapshot_digest,
            self.source_watermark_ledger_seq,
            self.source_request_digest,
        )
        if any(value is not None for value in source_values):
            if any(value is None for value in source_values):
                raise ProjectionContractError("validation source binding must be complete")
            for field_name, value in (
                ("source_snapshot_digest", self.source_snapshot_digest),
                ("source_request_digest", self.source_request_digest),
            ):
                validate_sha256_digest(value, field=field_name)  # type: ignore[arg-type]
            require_non_negative_sequence(
                self.source_watermark_ledger_seq,  # type: ignore[arg-type]
                field="source_watermark_ledger_seq",
            )
            if self.source_watermark_ledger_seq > self.checkpoint_ledger_seq:  # type: ignore[operator]
                raise ProjectionContractError(
                    "validation source watermark exceeds checkpoint"
                )


@dataclass(frozen=True, slots=True)
class ProjectionGenerationPromotionView:
    promotion_digest: str
    family_id: str
    generation: ProjectionGenerationView
    prior_generation: ProjectionGenerationView | None
    checkpoint_ledger_seq: int
    validation_digest: str
    target_authority_event_id: EventId
    prior_authority_event_id: EventId | None
    recorded_at: UtcTimestamp

    def __post_init__(self) -> None:
        validate_sha256_digest(self.promotion_digest, field="promotion_digest")
        validate_sha256_digest(self.validation_digest, field="validation_digest")


@dataclass(frozen=True, slots=True)
class ProjectionCheckpointView:
    generation_id: ProjectionGenerationId
    checkpoint_version: int
    contiguous_ledger_seq: int
    authority_aggregate_version: int
    event_id: EventId
    recorded_at: UtcTimestamp


@dataclass(frozen=True, slots=True)
class ProjectionGapView:
    gap_id: ProjectionGapId
    generation_id: ProjectionGenerationId
    ledger_seq_start: int
    ledger_seq_end: int
    state: ProjectionGapState
    lifecycle_version: int
    required: bool
    reason_code: str
    opened_event_id: EventId
    resolved_event_id: EventId | None
    recorded_at: UtcTimestamp


@dataclass(frozen=True, slots=True)
class ProjectionDeadLetterView:
    dead_letter_id: ProjectionDeadLetterId
    generation_id: ProjectionGenerationId
    ledger_seq: int
    source_event_id: EventId
    attempts: int
    reason_code: str
    authority_event_id: EventId
    recorded_at: UtcTimestamp


@dataclass(frozen=True, slots=True)
class ProjectionStatusMetadata:
    family_id: str
    family_kind: ProjectionFamilyKind
    projector_version: str
    ontology_contract_digest: str
    mapping_contract_digest: str
    generation_id: ProjectionGenerationId | None
    generation_state: ProjectionGenerationState | None
    contiguous_ledger_seq: int
    open_gap_count: int
    dead_letter_count: int
    trust_scope: TrustScope
    serving_time: UtcTimestamp
    authority_watermark_ledger_seq: int = 0


@dataclass(frozen=True, slots=True)
class DeliveryRecordView:
    generation_id: ProjectionGenerationId
    ledger_seq: int
    source_event_id: EventId
    source_event_digest: str
    source_event_type: str
    outcome: ProjectionDeliveryOutcome
    required: bool
    attempt_count: int
    finalized: bool
    error_code: str | None
    authority_event_id: EventId
    recorded_at: UtcTimestamp

    def __post_init__(self) -> None:
        validate_sha256_digest(self.source_event_digest, field="source_event_digest")


@dataclass(frozen=True, slots=True)
class ProjectionFamilyRegistrationRequest:
    family_id: str
    idempotency_key: str

    def __post_init__(self) -> None:
        require_token(self.family_id, field="projection_family_id")
        require_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class ProjectionGenerationCreateRequest:
    generation_id: ProjectionGenerationId
    family_id: str
    reason_code: str
    idempotency_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.generation_id, ProjectionGenerationId):
            raise ProjectionContractError("generation identity must be typed")
        require_token(self.family_id, field="projection_family_id")
        require_reason(self.reason_code)
        require_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class ProjectionGenerationTransitionRequest:
    generation_id: ProjectionGenerationId
    expected_authority_version: int
    target_state: ProjectionGenerationState
    reason_code: str
    idempotency_key: str
    validated_through_ledger_seq: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.generation_id, ProjectionGenerationId):
            raise ProjectionContractError("generation transition identity must be typed")
        require_positive_sequence(
            self.expected_authority_version, field="expected_authority_version"
        )
        if not isinstance(self.target_state, ProjectionGenerationState):
            raise ProjectionContractError("generation target state must be typed")
        require_reason(self.reason_code)
        require_idempotency_key(self.idempotency_key)
        if self.validated_through_ledger_seq is not None:
            require_non_negative_sequence(
                self.validated_through_ledger_seq,
                field="validated_through_ledger_seq",
            )


@dataclass(frozen=True, slots=True)
class ProjectionGenerationValidationRequest:
    generation_id: ProjectionGenerationId
    expected_authority_version: int
    checkpoint_ledger_seq: int
    service_compatibility_digest: str
    projection_state_digest: str
    reason_code: str
    idempotency_key: str
    source_snapshot_digest: str | None = None
    source_watermark_ledger_seq: int | None = None
    source_request_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.generation_id, ProjectionGenerationId):
            raise ProjectionContractError("validation generation identity must be typed")
        require_positive_sequence(
            self.expected_authority_version, field="expected_authority_version"
        )
        require_non_negative_sequence(
            self.checkpoint_ledger_seq, field="checkpoint_ledger_seq"
        )
        for field_name, value in (
            ("service_compatibility_digest", self.service_compatibility_digest),
            ("projection_state_digest", self.projection_state_digest),
        ):
            normalized = validate_sha256_digest(value, field=field_name)
            if normalized != value:
                raise ProjectionContractError(f"{field_name} must be canonical lowercase")
        require_reason(self.reason_code)
        require_idempotency_key(self.idempotency_key)
        source_values = (
            self.source_snapshot_digest,
            self.source_watermark_ledger_seq,
            self.source_request_digest,
        )
        if any(value is not None for value in source_values):
            if any(value is None for value in source_values):
                raise ProjectionContractError("validation source binding must be complete")
            for field_name, value in (
                ("source_snapshot_digest", self.source_snapshot_digest),
                ("source_request_digest", self.source_request_digest),
            ):
                normalized = validate_sha256_digest(value, field=field_name)  # type: ignore[arg-type]
                if normalized != value:
                    raise ProjectionContractError(
                        f"{field_name} must be canonical lowercase"
                    )
            require_non_negative_sequence(
                self.source_watermark_ledger_seq,  # type: ignore[arg-type]
                field="source_watermark_ledger_seq",
            )
            if self.source_watermark_ledger_seq > self.checkpoint_ledger_seq:  # type: ignore[operator]
                raise ProjectionContractError(
                    "validation source watermark exceeds checkpoint"
                )


@dataclass(frozen=True, slots=True)
class ProjectionGenerationPromotionRequest:
    generation_id: ProjectionGenerationId
    expected_authority_version: int
    checkpoint_ledger_seq: int
    validation_digest: str
    reason_code: str
    idempotency_key: str
    prior_generation_id: ProjectionGenerationId | None = None
    expected_prior_authority_version: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.generation_id, ProjectionGenerationId):
            raise ProjectionContractError("promotion generation identity must be typed")
        require_positive_sequence(
            self.expected_authority_version, field="expected_authority_version"
        )
        require_non_negative_sequence(
            self.checkpoint_ledger_seq, field="checkpoint_ledger_seq"
        )
        normalized = validate_sha256_digest(
            self.validation_digest, field="validation_digest"
        )
        if normalized != self.validation_digest:
            raise ProjectionContractError("validation_digest must be canonical lowercase")
        require_reason(self.reason_code)
        require_idempotency_key(self.idempotency_key)
        paired = (
            self.prior_generation_id is None,
            self.expected_prior_authority_version is None,
        )
        if paired[0] != paired[1]:
            raise ProjectionContractError(
                "prior generation identity and expected version must be supplied together"
            )
        if self.prior_generation_id is not None:
            if not isinstance(self.prior_generation_id, ProjectionGenerationId):
                raise ProjectionContractError("prior generation identity must be typed")
            if self.prior_generation_id == self.generation_id:
                raise ProjectionContractError("promotion cannot replace itself")
            require_positive_sequence(
                self.expected_prior_authority_version,
                field="expected_prior_authority_version",
            )


@dataclass(frozen=True, slots=True)
class ProjectionDeliveryRequest:
    generation_id: ProjectionGenerationId
    expected_authority_version: int
    ledger_seq: int
    outcome: ProjectionDeliveryOutcome
    idempotency_key: str
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.generation_id, ProjectionGenerationId):
            raise ProjectionContractError("delivery generation identity must be typed")
        require_positive_sequence(
            self.expected_authority_version, field="expected_authority_version"
        )
        require_positive_sequence(self.ledger_seq, field="ledger_seq")
        if not isinstance(self.outcome, ProjectionDeliveryOutcome):
            raise ProjectionContractError("delivery outcome must be typed")
        require_idempotency_key(self.idempotency_key)
        if self.error_code is not None:
            require_reason(self.error_code, field="error_code")
        if self.outcome in {
            ProjectionDeliveryOutcome.RETRYABLE_FAILURE,
            ProjectionDeliveryOutcome.REQUIRED_UNSUPPORTED,
        } and self.error_code is None:
            raise ProjectionContractError("failure delivery requires error_code")
        if self.outcome in {
            ProjectionDeliveryOutcome.APPLIED,
            ProjectionDeliveryOutcome.IGNORED_OPTIONAL,
        } and self.error_code is not None:
            raise ProjectionContractError("successful delivery cannot include error_code")


@dataclass(frozen=True, slots=True)
class ProjectionGapResolutionRequest:
    generation_id: ProjectionGenerationId
    expected_authority_version: int
    gap_id: ProjectionGapId
    reason_code: str
    idempotency_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.generation_id, ProjectionGenerationId):
            raise ProjectionContractError("gap generation identity must be typed")
        if not isinstance(self.gap_id, ProjectionGapId):
            raise ProjectionContractError("gap identity must be typed")
        require_positive_sequence(
            self.expected_authority_version, field="expected_authority_version"
        )
        require_reason(self.reason_code)
        require_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class ProjectionReadPolicy:
    policy_id: str
    purpose: str
    required_scope: str
    allowed_principal_ids: frozenset[str]
    allowed_family_ids: frozenset[str]
    allowed_family_kinds: frozenset[ProjectionFamilyKind]
    max_results: int = 1000

    def __post_init__(self) -> None:
        require_token(self.policy_id, field="projection_read_policy_id")
        require_token(self.purpose, field="projection_read_purpose")
        require_scope(self.required_scope, field="projection_read_scope")
        for field_name, value in (
            ("allowed_principal_ids", self.allowed_principal_ids),
            ("allowed_family_ids", self.allowed_family_ids),
            ("allowed_family_kinds", self.allowed_family_kinds),
        ):
            if not isinstance(value, frozenset) or not value:
                raise ProjectionContractError(f"{field_name} must be a non-empty frozenset")
        for principal in self.allowed_principal_ids:
            require_token(principal, field="projection_reader_principal")
        for family_id in self.allowed_family_ids:
            require_token(family_id, field="projection_read_family")
        if any(
            not isinstance(kind, ProjectionFamilyKind)
            for kind in self.allowed_family_kinds
        ):
            raise ProjectionContractError("projection read family kinds must be typed")
        require_positive_sequence(self.max_results, field="projection_read_max_results")

    def canonical_value(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "purpose": self.purpose,
            "required_scope": self.required_scope,
            "allowed_principal_ids": sorted(self.allowed_principal_ids),
            "allowed_family_ids": sorted(self.allowed_family_ids),
            "allowed_family_kinds": sorted(item.value for item in self.allowed_family_kinds),
            "max_results": self.max_results,
        }

    @property
    def digest(self) -> str:
        return digest_canonical(self.canonical_value())

    def require_family(self, definition: ProjectionFamilyDefinition) -> None:
        if definition.family_id not in self.allowed_family_ids:
            raise ProjectionAuthorizationError("projection family is outside read policy")
        if definition.family_kind not in self.allowed_family_kinds:
            raise ProjectionAuthorizationError("projection family kind is outside read policy")

    def require_principal(self, principal_id: str) -> None:
        if principal_id not in self.allowed_principal_ids:
            raise ProjectionAuthorizationError("projection reader principal is not allowed")

    def require_limit(self, limit: int) -> None:
        require_positive_sequence(limit, field="projection_read_limit")
        if limit > self.max_results:
            raise ProjectionAuthorizationError("projection read limit exceeds policy")


@dataclass(frozen=True, slots=True)
class ProjectionPayload:
    values: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.values, Mapping):
            raise ProjectionContractError("projection payload must be a mapping")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(dict(self.values))


def require_positive_sequence(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProjectionContractError(f"{field} must be a positive integer")
    return value


def require_non_negative_sequence(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProjectionContractError(f"{field} must be a non-negative integer")
    return value


def require_reason(value: str, *, field: str = "reason_code") -> str:
    return require_token(value, field=field)


def require_idempotency_key(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectionContractError("idempotency_key must be non-empty")
    if len(value.encode("utf-8")) > 256:
        raise ProjectionContractError("idempotency_key exceeds 256 UTF-8 bytes")
    return value


def require_exact_keys(value: Any, expected: frozenset[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ProjectionContractError(f"{name} fields differ from retained schema")
    return value
