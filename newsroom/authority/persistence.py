from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, FrozenSet

from .auth import AuthenticationProof
from .canonical import digest_canonical
from .models import SemanticCommand
from .types import TrustScope, UUIDv4Id, require_scope, require_token


SUCCESSFUL_READ_AUDIT_STATUS = "DEFERRED"


@dataclass(frozen=True, slots=True)
class PayloadId(UUIDv4Id):
    """Typed identity for one retained authority payload."""


class AuthorityPersistenceError(RuntimeError):
    """Base failure for the authoritative SQLite event ledger."""


class DiagnosticHistoryExpired(AuthorityPersistenceError):
    """Full diagnostic provenance/result expired; its identity stays reserved."""


class AuthoritySchemaError(AuthorityPersistenceError):
    """The authority database does not match the accepted schema contract."""


class AuthorityWriterBusy(AuthorityPersistenceError):
    """Another lifetime authority writer already owns the database."""


class ExpectedVersionConflict(AuthorityPersistenceError):
    """The command's expected aggregate version does not match authority."""


class IdempotencyConflict(AuthorityPersistenceError):
    """A durable idempotency identity conflicts with committed authority."""


class UnknownCausation(AuthorityPersistenceError):
    """A command or event causation reference does not resolve."""


class UnsupportedPayloadMode(AuthorityPersistenceError):
    """The payload mode belongs to a later authority review unit."""


class ReadPolicyDenied(PermissionError):
    """A server-owned metadata policy does not permit the requested read."""


class MetadataClass(StrEnum):
    ROUTING = "ROUTING"
    PROVENANCE = "PROVENANCE"
    RESULT = "RESULT"


@dataclass(frozen=True, slots=True)
class EventReadPolicy:
    """Server-owned, purpose-bounded event metadata policy.

    A public caller never supplies security/trust filters. The composed system
    binds one immutable policy to a facade and the store applies its bounds and
    row filters. Every collection is required to be a ``frozenset`` so trusted
    composition cannot widen the policy in place after system startup.
    """

    policy_id: str
    purpose: str
    required_scope: str
    allowed_principal_ids: FrozenSet[str]
    allowed_security_scopes: FrozenSet[str]
    allowed_trust_scopes: FrozenSet[TrustScope]
    metadata_classes: FrozenSet[MetadataClass]
    minimum_ledger_seq: int = 1
    maximum_ledger_seq: int | None = None
    max_results: int = 100

    def __post_init__(self) -> None:
        require_token(self.policy_id, field="event_read_policy_id")
        require_token(self.purpose, field="event_read_purpose")
        require_scope(self.required_scope, field="event_read_required_scope")
        collection_fields = (
            "allowed_principal_ids",
            "allowed_security_scopes",
            "allowed_trust_scopes",
            "metadata_classes",
        )
        for field_name in collection_fields:
            if not isinstance(getattr(self, field_name), frozenset):
                raise ValueError(
                    f"{field_name} must be an immutable frozenset"
                )
        if not self.allowed_principal_ids:
            raise ValueError("event read policy requires at least one principal")
        for principal in self.allowed_principal_ids:
            require_token(principal, field="event_read_principal")
        if not self.allowed_security_scopes:
            raise ValueError("event read policy requires security scopes")
        for scope in self.allowed_security_scopes:
            require_scope(scope, field="event_read_security_scope")
        if not self.allowed_trust_scopes:
            raise ValueError("event read policy requires trust scopes")
        if not all(
            isinstance(value, TrustScope)
            for value in self.allowed_trust_scopes
        ):
            raise ValueError("event read trust scopes must be typed")
        if not self.metadata_classes:
            raise ValueError("event read policy requires metadata classes")
        if not all(
            isinstance(value, MetadataClass)
            for value in self.metadata_classes
        ):
            raise ValueError("metadata classes must be typed")
        if (
            isinstance(self.minimum_ledger_seq, bool)
            or not isinstance(self.minimum_ledger_seq, int)
            or self.minimum_ledger_seq <= 0
        ):
            raise ValueError("minimum ledger sequence must be positive")
        if self.maximum_ledger_seq is not None:
            if (
                isinstance(self.maximum_ledger_seq, bool)
                or not isinstance(self.maximum_ledger_seq, int)
                or self.maximum_ledger_seq < self.minimum_ledger_seq
            ):
                raise ValueError("maximum ledger sequence must follow minimum")
        if (
            isinstance(self.max_results, bool)
            or not isinstance(self.max_results, int)
            or not 1 <= self.max_results <= 1000
        ):
            raise ValueError(
                "event read result bound must be between 1 and 1000"
            )

    def canonical_value(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "purpose": self.purpose,
            "required_scope": self.required_scope,
            "allowed_principal_ids": sorted(self.allowed_principal_ids),
            "allowed_security_scopes": sorted(
                self.allowed_security_scopes
            ),
            "allowed_trust_scopes": sorted(
                value.value for value in self.allowed_trust_scopes
            ),
            "metadata_classes": sorted(
                value.value for value in self.metadata_classes
            ),
            "minimum_ledger_seq": self.minimum_ledger_seq,
            "maximum_ledger_seq": self.maximum_ledger_seq,
            "max_results": self.max_results,
        }

    @property
    def digest(self) -> str:
        return digest_canonical(self.canonical_value())

    def require_principal(self, principal_id: str) -> None:
        if principal_id not in self.allowed_principal_ids:
            raise ReadPolicyDenied(
                "principal is outside the server read policy"
            )

    def require_metadata_class(
        self, metadata_class: MetadataClass
    ) -> None:
        if metadata_class not in self.metadata_classes:
            raise ReadPolicyDenied(
                f"metadata class {metadata_class.value} is outside "
                "the server read policy"
            )

    def require_window(
        self, *, after_ledger_seq: int, limit: int
    ) -> None:
        if (
            isinstance(after_ledger_seq, bool)
            or not isinstance(after_ledger_seq, int)
            or after_ledger_seq < 0
        ):
            raise ValueError(
                "after_ledger_seq must be a non-negative integer"
            )
        if after_ledger_seq < self.minimum_ledger_seq - 1:
            raise ReadPolicyDenied(
                "requested sequence precedes the policy window"
            )
        if (
            self.maximum_ledger_seq is not None
            and after_ledger_seq >= self.maximum_ledger_seq
        ):
            raise ReadPolicyDenied(
                "requested sequence is outside the policy window"
            )
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
            or limit > self.max_results
        ):
            raise ReadPolicyDenied(
                "requested result limit exceeds server policy"
            )


def projector_service_read_policy(
    *,
    principal_id: str,
    allowed_security_scopes: FrozenSet[str],
    allowed_trust_scopes: FrozenSet[TrustScope],
    maximum_ledger_seq: int | None = None,
    max_results: int = 250,
) -> EventReadPolicy:
    """Dedicated policy seam for a future deterministic projector service."""

    return EventReadPolicy(
        policy_id="projector-structural-v1",
        purpose="projector.structural",
        required_scope="authority.projector.events.read",
        allowed_principal_ids=frozenset({principal_id}),
        allowed_security_scopes=allowed_security_scopes,
        allowed_trust_scopes=allowed_trust_scopes,
        metadata_classes=frozenset(
            {MetadataClass.ROUTING, MetadataClass.PROVENANCE}
        ),
        minimum_ledger_seq=1,
        maximum_ledger_seq=maximum_ledger_seq,
        max_results=max_results,
    )


@dataclass(frozen=True, slots=True)
class CommittedCommand:
    command_id: str
    aggregate_type: str
    aggregate_id: str
    aggregate_version: int
    ledger_seq: int
    event_id: str
    result_digest: str
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class CommandResultRecord:
    command_id: str
    result_digest: str
    result_bytes: bytes


@dataclass(frozen=True, slots=True)
class CommandDefinitionRecord:
    definition_digest: str
    command_type: str
    definition_version: str
    payload_schema_contract_digest: str
    canonical_bytes: bytes


@dataclass(frozen=True, slots=True)
class PayloadSchemaContractRecord:
    contract_digest: str
    schema_version: str
    payload_mode: str
    contract_version: str
    canonicalizer_implementation_version: str
    canonical_bytes: bytes


@dataclass(frozen=True, slots=True)
class LedgerEventRecord:
    ledger_seq: int
    event_id: str
    event_type: str
    event_schema_version: int
    aggregate_type: str
    aggregate_id: str
    aggregate_version: int
    recorded_at: str
    command_id: str
    producer_version: str
    command_definition_version: str
    command_definition_digest: str
    payload_id: str
    payload_mode: str
    payload_schema_version: str
    payload_schema_contract_version: str
    payload_schema_contract_digest: str
    payload_canonicalizer_version: str
    payload_digest: str
    object_admission_id: str | None
    principal_id: str
    authentication_context_id: str
    authorization_request_digest: str
    authorization_decision_id: str
    correlation_id: str | None
    causation_kind: str | None
    causation_identifier: str | None
    causation_external_system: str | None
    security_scope: str
    retention_scope: str
    trust_scope: str


@dataclass(frozen=True, slots=True)
class RetiredLedgerEventRecord:
    """Routing and exact original-header identity, never synthetic provenance."""
    ledger_seq: int
    event_id: str
    command_id: str
    event_type: str
    aggregate_type: str
    aggregate_id: str
    security_scope: str
    trust_scope: str
    original_header_digest: str


@dataclass(frozen=True, slots=True)
class AuthenticationContextRecord:
    authentication_context_id: str
    principal_id: str
    authority_domain: str
    authentication_method: str
    assurance_class: str
    credential_binding_digest: str
    authenticated_at: str
    expires_at: str
    canonical_digest: str
    canonical_bytes: bytes


@dataclass(frozen=True, slots=True)
class AuthorizationRequestRecord:
    request_digest: str
    authentication_context_id: str
    principal_id: str
    authority_domain: str
    operation_type: str
    required_scope: str
    canonical_record_digest: str
    canonical_bytes: bytes


@dataclass(frozen=True, slots=True)
class AuthorizationDecisionRecord:
    authorization_decision_id: str
    authentication_context_id: str
    authorization_request_digest: str
    authorization_policy_version: str
    effective_scopes: tuple[str, ...]
    effective_scope_digest: str
    allowed: bool
    reason_code: str
    decided_at: str
    canonical_digest: str
    canonical_bytes: bytes


@dataclass(frozen=True, slots=True)
class EventProvenanceRecord:
    event: LedgerEventRecord
    command_definition: CommandDefinitionRecord
    payload_schema_contract: PayloadSchemaContractRecord
    authentication: AuthenticationContextRecord
    authorization_request: AuthorizationRequestRecord
    authorization_decision: AuthorizationDecisionRecord


class AuthorityCommands:
    """Public command facade; no persistence object is exposed."""

    __slots__ = ("__execute",)

    def __init__(
        self,
        execute: Callable[
            [SemanticCommand, AuthenticationProof], CommittedCommand
        ],
    ) -> None:
        self.__execute = execute

    def execute(
        self, command: SemanticCommand, *, proof: AuthenticationProof
    ) -> CommittedCommand:
        return self.__execute(command, proof)


class AuthorityEvents:
    """Authenticated, policy-bound metadata facade.

    Successful-read audit retention is explicitly Deferred in Increment 1A2a.
    Authentication and authorization are checked for every call, but this facade
    does not claim that successful reads are durably audited.
    """

    __slots__ = ("__read", "__provenance", "__result", "policy_id")

    def __init__(
        self,
        *,
        policy_id: str,
        read: Callable[
            [int, int, AuthenticationProof],
            tuple[LedgerEventRecord | RetiredLedgerEventRecord, ...],
        ],
        provenance: Callable[
            [str, AuthenticationProof], EventProvenanceRecord
        ],
        result: Callable[
            [str, AuthenticationProof], CommandResultRecord
        ],
    ) -> None:
        require_token(policy_id, field="event_read_policy_id")
        self.policy_id = policy_id
        self.__read = read
        self.__provenance = provenance
        self.__result = result

    def after(
        self,
        ledger_seq: int,
        *,
        limit: int = 100,
        proof: AuthenticationProof,
    ) -> tuple[LedgerEventRecord | RetiredLedgerEventRecord, ...]:
        return self.__read(ledger_seq, limit, proof)

    def provenance(
        self, event_id: str, *, proof: AuthenticationProof
    ) -> EventProvenanceRecord:
        return self.__provenance(event_id, proof)

    def command_result(
        self, command_id: str, *, proof: AuthenticationProof
    ) -> CommandResultRecord:
        return self.__result(command_id, proof)
