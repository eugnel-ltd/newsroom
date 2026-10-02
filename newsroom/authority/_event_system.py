from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ._capability import _CapabilityIssuer
from ._event_store import _EventAuthorityStore
from ._security import _AuthorizationRequest
from .auth import AuthenticationProof
from .canonical import digest_canonical
from .models import SemanticCommand
from .persistence import (
    AuthorityCommands,
    AuthorityEvents,
    CommandResultRecord,
    CommittedCommand,
    EventProvenanceRecord,
    EventReadPolicy,
    LedgerEventRecord,
    RetiredLedgerEventRecord,
    MetadataClass,
)
from .policy import CommandRegistry, PayloadSchemaRegistry
from .service import CommandService
from .types import UtcTimestamp


_READ_SCHEMA_CONTRACT_DIGEST = digest_canonical(
    {
        "schema_version": "authority_read_v1",
        "payload_mode": "NO_PAYLOAD",
        "contract_version": "authority-read-contract-v1",
        "canonicalizer_implementation_version": "authority-read-none-v1",
        "golden_vectors": [{"name": "empty", "digest": "sha256-empty"}],
    }
)


class _ReadBoundary:
    """Authenticate and authorize one immutable server-owned read policy."""

    def __init__(
        self,
        *,
        store: _EventAuthorityStore,
        policy: EventReadPolicy,
        authenticator: Any,
        authorizer: Any,
        clock: Callable[[], UtcTimestamp],
    ) -> None:
        self._store = store
        self._policy = policy
        self._authenticator = authenticator
        self._authorizer = authorizer
        self._clock = clock

    def _authorize(
        self,
        proof: AuthenticationProof,
        *,
        metadata_class: MetadataClass,
        semantic_value: dict[str, object],
    ) -> None:
        self._policy.require_metadata_class(metadata_class)
        now = self._clock()
        authentication = self._authenticator.authenticate(proof, now=now)
        authentication.require_current(now)
        self._policy.require_principal(authentication.principal_id)

        operation_type = (
            f"read:{self._policy.purpose}:{metadata_class.value.lower()}"
        )
        stable_digest = digest_canonical(
            {
                "event_read_policy_digest": self._policy.digest,
                "metadata_class": metadata_class.value,
                "semantic_value": semantic_value,
            }
        )
        unsigned = {
            "authentication_context_id": str(
                authentication.authentication_context_id
            ),
            "principal_id": authentication.principal_id,
            "authority_domain": authentication.authority_domain,
            "operation_type": operation_type,
            "required_scope": self._policy.required_scope,
            "stable_semantic_request_digest": stable_digest,
            "command_definition_digest": self._policy.digest,
            "aggregate_type": "authority_event_metadata",
            "aggregate_id": self._policy.policy_id,
            "event_type": "authority.metadata.read",
            "event_schema_version": 1,
            "payload_mode": "NO_PAYLOAD",
            "payload_schema_version": "authority_read_v1",
            "payload_schema_contract_version": "authority-read-contract-v1",
            "payload_schema_contract_digest": _READ_SCHEMA_CONTRACT_DIGEST,
            "payload_canonicalizer_version": "authority-read-none-v1",
            "trust_scope": "OBSERVED",
            "security_scope": "authority.metadata",
            "retention_scope": "authority.audit",
            "object_class": None,
            "allowed_use": None,
        }
        request = _AuthorizationRequest(
            authentication_context_id=authentication.authentication_context_id,
            principal_id=authentication.principal_id,
            authority_domain=authentication.authority_domain,
            operation_type=operation_type,
            required_scope=self._policy.required_scope,
            stable_semantic_request_digest=stable_digest,
            command_definition_digest=self._policy.digest,
            aggregate_type="authority_event_metadata",
            aggregate_id=self._policy.policy_id,
            event_type="authority.metadata.read",
            event_schema_version=1,
            payload_mode="NO_PAYLOAD",
            payload_schema_version="authority_read_v1",
            payload_schema_contract_version="authority-read-contract-v1",
            payload_schema_contract_digest=_READ_SCHEMA_CONTRACT_DIGEST,
            payload_canonicalizer_version="authority-read-none-v1",
            trust_scope="OBSERVED",
            security_scope="authority.metadata",
            retention_scope="authority.audit",
            object_class=None,
            allowed_use=None,
            request_digest=digest_canonical(unsigned),
        )
        decision = self._authorizer.authorize(
            authentication, request, now=now
        )
        if (
            decision.authentication_context_id
            != authentication.authentication_context_id
        ):
            raise PermissionError("reader authorization context mismatch")
        if decision.authorization_request_digest != request.request_digest:
            raise PermissionError("reader authorization request mismatch")
        decision.require_allowed()
        # Successful-read audit retention is explicitly Deferred in A2a.

    def events_after(
        self,
        ledger_seq: int,
        limit: int,
        proof: AuthenticationProof,
    ) -> tuple[LedgerEventRecord | RetiredLedgerEventRecord, ...]:
        self._policy.require_window(
            after_ledger_seq=ledger_seq, limit=limit
        )
        self._authorize(
            proof,
            metadata_class=MetadataClass.ROUTING,
            semantic_value={
                "after_ledger_seq": ledger_seq,
                "limit": limit,
            },
        )
        return self._store.events_after(
            policy=self._policy,
            ledger_seq=ledger_seq,
            limit=limit,
        )

    def provenance(
        self,
        event_id: str,
        proof: AuthenticationProof,
    ) -> EventProvenanceRecord:
        self._authorize(
            proof,
            metadata_class=MetadataClass.PROVENANCE,
            semantic_value={"event_id": event_id},
        )
        value = self._store.event_provenance(
            event_id=event_id, policy=self._policy
        )
        if self._store._current_state_only:
            self._store._validate_retained_event(event_id)
        return value

    def command_result(
        self,
        command_id: str,
        proof: AuthenticationProof,
    ) -> CommandResultRecord:
        self._authorize(
            proof,
            metadata_class=MetadataClass.RESULT,
            semantic_value={"command_id": command_id},
        )
        return self._store.command_result(
            command_id=command_id, policy=self._policy
        )


class AuthorityEventSystem:
    """Composed A2a facades without a public mutation store."""

    __slots__ = ("commands", "events", "__close")

    def __init__(
        self,
        *,
        commands: AuthorityCommands,
        events: AuthorityEvents,
        close: Callable[[], None],
    ) -> None:
        self.commands = commands
        self.events = events
        self.__close = close

    def close(self) -> None:
        self.__close()

    def __enter__(self) -> AuthorityEventSystem:
        return self

    def __exit__(
        self, exc_type: object, exc: object, tb: object
    ) -> None:
        self.close()


def open_authority_event_system(
    *,
    path: Path,
    registry: CommandRegistry,
    payload_schemas: PayloadSchemaRegistry,
    authenticator: Any,
    authorizer: Any,
    event_read_policy: EventReadPolicy,
    command_service_version: str = "authority-command-v1",
    busy_timeout_ms: int = 5_000,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
) -> AuthorityEventSystem:
    """Open one lifetime-writer SQLite event authority system.

    The returned object exposes command and policy-bound metadata facades.
    SQLite connections, writer methods, capabilities and payload bytes remain
    private.
    """

    issuer = _CapabilityIssuer(
        command_registry=registry,
        payload_schemas=payload_schemas,
    )
    store: _EventAuthorityStore | None = None
    try:
        store = _EventAuthorityStore(
            path,
            issuer=issuer,
            command_registry=registry,
            payload_schemas=payload_schemas,
            command_service_version=command_service_version,
            busy_timeout_ms=busy_timeout_ms,
            clock=clock,
        )
        command_boundary = CommandService(
            registry=registry,
            payload_schemas=payload_schemas,
            authenticator=authenticator,
            authorizer=authorizer,
            committed_lookup=store,
            clock=clock,
            _issuer=issuer,
        )
        read_boundary = _ReadBoundary(
            store=store,
            policy=event_read_policy,
            authenticator=authenticator,
            authorizer=authorizer,
            clock=clock,
        )

        def execute(
            command: SemanticCommand,
            proof: AuthenticationProof,
        ) -> CommittedCommand:
            grant = command_boundary._authorize_for_commit(
                command, proof=proof
            )
            return store.commit(grant)  # type: ignore[union-attr]

        return AuthorityEventSystem(
            commands=AuthorityCommands(execute),
            events=AuthorityEvents(
                policy_id=event_read_policy.policy_id,
                read=read_boundary.events_after,
                provenance=read_boundary.provenance,
                result=read_boundary.command_result,
            ),
            close=store.close,
        )
    except Exception:
        if store is not None:
            store.close()
        raise
