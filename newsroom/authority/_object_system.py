from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, BinaryIO

from ._capability import _AuthorizedCommandGrant, _CapabilityIssuer, _derive_idempotency_namespace
from ._event_system import _ReadBoundary
from ._object_capability import (
    _AdmissionCommitGrant,
    _AdmissionPreflightGrant,
    _HydrationGrant,
    _MaintenanceGrant,
    _ObjectCapabilityIssuer,
    _RightsProposal,
)
from ._object_cas import _GovernedCAS
from ._object_store import _GovernedObjectAuthorityStore
from ._security import _AuthorizationRequest
from .auth import AuthenticationProof
from .canonical import digest_canonical
from .models import InlinePayload, SemanticCommand
from .object_policy import (
    HydrationPolicyRegistry,
    LIFECYCLE_COMMAND_TYPES,
    ObjectAdmissionRegistry,
    RightsPolicyRegistry,
    merge_authority_registries,
)
from .objects import (
    AdmissionRevocationView,
    BlobIdentity,
    GovernedDeletionId,
    GovernedDeletionView,
    HydrationRequest,
    ObjectAccessDecisionId,
    ObjectAccessDecisionView,
    ObjectAdmissionDenied,
    ObjectAdmissionId,
    ObjectAdmissionRequest,
    ObjectAdmissionView,
    ObjectLimits,
    ObjectOperationId,
    ObjectPreflightId,
    RecoveryPinId,
    RecoveryPinView,
)
from .persistence import (
    AuthorityCommands,
    AuthorityEvents,
    CommittedCommand,
    EventReadPolicy,
    IdempotencyConflict,
)
from .policy import CommandRegistry, PayloadSchemaRegistry
from .service import CommandService
from .types import AggregateId, RightsDecisionId, UtcTimestamp, UUIDv4Id


_Source = bytes | bytearray | memoryview | BinaryIO | Iterable[bytes]
_OBJECT_COMPOSITION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class ObjectAdmissionResult:
    admission: ObjectAdmissionView
    replayed: bool


@dataclass(frozen=True, slots=True)
class HydratedObject:
    data: bytes
    decision: ObjectAccessDecisionView


class GovernedObjects:
    """Authenticated governed-object facade; no direct writer is exposed."""

    __slots__ = (
        "__admit",
        "__committed_admission",
        "__hydrate",
        "__rehydrate",
        "__access_decision",
        "__latest_access_decision",
        "__revoke",
        "__request_deletion",
        "__tombstone",
        "__complete_deletion",
        "__create_pin",
        "__release_pin",
        "__collect_orphans",
    )

    def __init__(
        self,
        *,
        admit: Callable[[ObjectAdmissionRequest, _Source, AuthenticationProof], ObjectAdmissionResult],
        committed_admission: Callable[
            [ObjectAdmissionRequest, AuthenticationProof],
            ObjectAdmissionResult | None,
        ],
        hydrate: Callable[[HydrationRequest, AuthenticationProof], HydratedObject],
        rehydrate: Callable[[HydrationRequest, AuthenticationProof], HydratedObject],
        access_decision: Callable[
            [ObjectAccessDecisionId, ObjectAdmissionId, str, AuthenticationProof],
            ObjectAccessDecisionView,
        ],
        latest_access_decision: Callable[
            [ObjectAdmissionId, str, AuthenticationProof],
            ObjectAccessDecisionView,
        ],
        revoke: Callable[[ObjectAdmissionId, str, str, AuthenticationProof], AdmissionRevocationView],
        request_deletion: Callable[[str, str, str, AuthenticationProof], GovernedDeletionView],
        tombstone: Callable[[GovernedDeletionId, str, str, AuthenticationProof], GovernedDeletionView],
        complete_deletion: Callable[[GovernedDeletionId, str, AuthenticationProof], GovernedDeletionView],
        create_pin: Callable[[str, str, str, AuthenticationProof], RecoveryPinView],
        release_pin: Callable[[RecoveryPinId, str, str, AuthenticationProof], RecoveryPinView],
        collect_orphans: Callable[[AuthenticationProof], tuple[BlobIdentity, ...]],
    ) -> None:
        self.__admit = admit
        self.__committed_admission = committed_admission
        self.__hydrate = hydrate
        self.__rehydrate = rehydrate
        self.__access_decision = access_decision
        self.__latest_access_decision = latest_access_decision
        self.__revoke = revoke
        self.__request_deletion = request_deletion
        self.__tombstone = tombstone
        self.__complete_deletion = complete_deletion
        self.__create_pin = create_pin
        self.__release_pin = release_pin
        self.__collect_orphans = collect_orphans

    def admit(
        self,
        request: ObjectAdmissionRequest,
        source: _Source,
        *,
        proof: AuthenticationProof,
    ) -> ObjectAdmissionResult:
        return self.__admit(request, source, proof)

    def committed_admission(
        self,
        request: ObjectAdmissionRequest,
        *,
        proof: AuthenticationProof,
    ) -> ObjectAdmissionResult | None:
        """Return only an exact committed replay; never stage or admit bytes."""

        return self.__committed_admission(request, proof)

    def hydrate(
        self, request: HydrationRequest, *, proof: AuthenticationProof
    ) -> HydratedObject:
        return self.__hydrate(request, proof)

    def rehydrate(
        self, request: HydrationRequest, *, proof: AuthenticationProof
    ) -> HydratedObject:
        """Revalidate a read, retaining its prior exact access receipt if unchanged.

        Authentication, authorisation, current rights and bytes are checked on
        every call. This is receipt reuse, not an audit of a new delivery.
        """
        return self.__rehydrate(request, proof)

    def _bind_composed_hydrate(
        self,
        hydrate: Callable[[HydrationRequest, AuthenticationProof], HydratedObject],
        *,
        _token: object,
        rehydrate: Callable[[HydrationRequest, AuthenticationProof], HydratedObject],
    ) -> None:
        """Bind hydration to the private cumulative-store transaction seam."""
        if (_token is not _OBJECT_COMPOSITION_TOKEN
                or not callable(hydrate) or not callable(rehydrate)):
            raise TypeError("governed object composition is private")
        self.__hydrate = hydrate
        self.__rehydrate = rehydrate

    def latest_access_decision(
        self,
        admission_id: ObjectAdmissionId,
        *,
        purpose: str,
        proof: AuthenticationProof,
    ) -> ObjectAccessDecisionView:
        return self.__latest_access_decision(admission_id, purpose, proof)

    def access_decision(
        self,
        access_decision_id: ObjectAccessDecisionId,
        *,
        admission_id: ObjectAdmissionId,
        purpose: str,
        proof: AuthenticationProof,
    ) -> ObjectAccessDecisionView:
        """Authenticate an exact retained historical access receipt."""

        return self.__access_decision(
            access_decision_id, admission_id, purpose, proof,
        )

    def revoke(
        self,
        admission_id: ObjectAdmissionId,
        *,
        reason_code: str,
        idempotency_key: str,
        proof: AuthenticationProof,
    ) -> AdmissionRevocationView:
        return self.__revoke(
            admission_id, reason_code, idempotency_key, proof
        )

    def request_deletion(
        self,
        blob_digest: str,
        *,
        reason_code: str,
        idempotency_key: str,
        proof: AuthenticationProof,
    ) -> GovernedDeletionView:
        return self.__request_deletion(
            blob_digest, reason_code, idempotency_key, proof
        )

    def tombstone(
        self,
        deletion_id: GovernedDeletionId,
        *,
        reason_code: str,
        idempotency_key: str,
        proof: AuthenticationProof,
    ) -> GovernedDeletionView:
        return self.__tombstone(
            deletion_id, reason_code, idempotency_key, proof
        )

    def complete_deletion(
        self,
        deletion_id: GovernedDeletionId,
        *,
        idempotency_key: str,
        proof: AuthenticationProof,
    ) -> GovernedDeletionView:
        return self.__complete_deletion(deletion_id, idempotency_key, proof)

    def create_recovery_pin(
        self,
        blob_digest: str,
        *,
        reason_code: str,
        idempotency_key: str,
        proof: AuthenticationProof,
    ) -> RecoveryPinView:
        return self.__create_pin(
            blob_digest, reason_code, idempotency_key, proof
        )

    def release_recovery_pin(
        self,
        pin_id: RecoveryPinId,
        *,
        reason_code: str,
        idempotency_key: str,
        proof: AuthenticationProof,
    ) -> RecoveryPinView:
        return self.__release_pin(
            pin_id, reason_code, idempotency_key, proof
        )

    def collect_orphans(
        self, *, proof: AuthenticationProof
    ) -> tuple[BlobIdentity, ...]:
        return self.__collect_orphans(proof)


class GovernedObjectAuthoritySystem:
    """Composed A1/A2a/A2b facades; stores and capabilities remain internal."""

    __slots__ = ("commands", "events", "objects", "__close")

    def __init__(
        self,
        *,
        commands: AuthorityCommands,
        events: AuthorityEvents,
        objects: GovernedObjects,
        close: Callable[[], None],
    ) -> None:
        self.commands = commands
        self.events = events
        self.objects = objects
        self.__close = close

    def close(self) -> None:
        self.__close()

    def __enter__(self) -> GovernedObjectAuthoritySystem:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class _ObjectBoundary:
    def __init__(
        self,
        *,
        store: _GovernedObjectAuthorityStore,
        cas: _GovernedCAS,
        object_issuer: _ObjectCapabilityIssuer,
        admission_registry: ObjectAdmissionRegistry,
        rights_policies: RightsPolicyRegistry,
        hydration_policies: HydrationPolicyRegistry,
        authenticator: Any,
        authorizer: Any,
        command_service: CommandService,
        command_registry: CommandRegistry,
        clock: Callable[[], UtcTimestamp],
    ) -> None:
        self._store = store
        self._cas = cas
        self._object_issuer = object_issuer
        self._admission_registry = admission_registry
        self._rights_policies = rights_policies
        self._hydration_policies = hydration_policies
        self._authenticator = authenticator
        self._authorizer = authorizer
        self._command_service = command_service
        self._command_registry = command_registry
        self._clock = clock

    def _authenticate(self, proof: AuthenticationProof) -> tuple[Any, UtcTimestamp]:
        now = self._clock()
        authentication = self._authenticator.authenticate(proof, now=now)
        authentication.require_current(now)
        return authentication, now

    def _authorize(
        self,
        *,
        authentication: Any,
        now: UtcTimestamp,
        operation_type: str,
        required_scope: str,
        stable_digest: str,
        definition_digest: str,
        aggregate_type: str,
        aggregate_id: str,
        object_class: str | None,
        allowed_use: str | None,
        security_scope: str,
        retention_scope: str,
    ) -> tuple[Any, Any]:
        schema_contract_digest = digest_canonical(
            {
                "operation_type": operation_type,
                "definition_digest": definition_digest,
                "contract": "governed-object-authorization-request-v1",
            }
        )
        unsigned = {
            "authentication_context_id": str(
                authentication.authentication_context_id
            ),
            "principal_id": authentication.principal_id,
            "authority_domain": authentication.authority_domain,
            "operation_type": operation_type,
            "required_scope": required_scope,
            "stable_semantic_request_digest": stable_digest,
            "command_definition_digest": definition_digest,
            "aggregate_type": aggregate_type,
            "aggregate_id": aggregate_id,
            "event_type": "governed_object.authorization.evaluated",
            "event_schema_version": 1,
            "payload_mode": "NO_PAYLOAD",
            "payload_schema_version": "governed_object_authorization_v1",
            "payload_schema_contract_version": (
                "governed-object-authorization-contract-v1"
            ),
            "payload_schema_contract_digest": schema_contract_digest,
            "payload_canonicalizer_version": (
                "governed-object-authorization-none-v1"
            ),
            "trust_scope": "ADMITTED",
            "security_scope": security_scope,
            "retention_scope": retention_scope,
            "object_class": object_class,
            "allowed_use": allowed_use,
        }
        request = _AuthorizationRequest(
            authentication_context_id=authentication.authentication_context_id,
            principal_id=authentication.principal_id,
            authority_domain=authentication.authority_domain,
            operation_type=operation_type,
            required_scope=required_scope,
            stable_semantic_request_digest=stable_digest,
            command_definition_digest=definition_digest,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            event_type="governed_object.authorization.evaluated",
            event_schema_version=1,
            payload_mode="NO_PAYLOAD",
            payload_schema_version="governed_object_authorization_v1",
            payload_schema_contract_version=(
                "governed-object-authorization-contract-v1"
            ),
            payload_schema_contract_digest=schema_contract_digest,
            payload_canonicalizer_version=(
                "governed-object-authorization-none-v1"
            ),
            trust_scope="ADMITTED",
            security_scope=security_scope,
            retention_scope=retention_scope,
            object_class=object_class,
            allowed_use=allowed_use,
            request_digest=digest_canonical(unsigned),
        )
        decision = self._authorizer.authorize(
            authentication, request, now=now
        )
        if (
            decision.authentication_context_id
            != authentication.authentication_context_id
            or decision.authorization_request_digest != request.request_digest
        ):
            raise PermissionError(
                "object authorizer returned mismatched provenance"
            )
        decision.require_allowed()
        return request, decision

    def _lifecycle_command_grant(
        self,
        command_type: str,
        target: UUIDv4Id,
        payload: dict[str, object],
        *,
        proof: AuthenticationProof,
        idempotency_key: str,
    ) -> _AuthorizedCommandGrant:
        definition = self._command_registry.resolve(command_type)
        with self._store._lock:
            row = self._store._connection.execute(
                "SELECT current_version FROM authority_aggregates "
                "WHERE aggregate_type=? AND aggregate_id=?",
                (definition.aggregate_type, str(AggregateId(target.value))),
            ).fetchone()
            expected_version = 0 if row is None else int(row["current_version"])
        return self._command_service._authorize_for_commit(
            SemanticCommand(
                command_type=command_type,
                aggregate_id=AggregateId(target.value),
                expected_aggregate_version=expected_version,
                payload=InlinePayload(payload),
                idempotency_key=idempotency_key,
            ),
            proof=proof,
        )

    def _admission_preflight(
        self,
        request: ObjectAdmissionRequest,
        proof: AuthenticationProof,
    ) -> tuple[_AdmissionPreflightGrant, bool]:
        if not isinstance(request, ObjectAdmissionRequest):
            raise TypeError("request must be ObjectAdmissionRequest")
        authentication, checked_at = self._authenticate(proof)
        namespace = digest_canonical(
            {
                "authority_domain": authentication.authority_domain,
                "principal_id": authentication.principal_id,
                "operation": "object_admission",
                "admission_type": request.admission_type,
            }
        )
        replay_contract = self._store.committed_admission_replay_contract(
            idempotency_namespace=namespace,
            idempotency_key=request.idempotency_key,
        )
        if replay_contract is None:
            definition = self._admission_registry.resolve(
                request.admission_type
            )
            rights_policy = self._rights_policies.resolve_digest(
                definition.rights_policy_contract_digest
            )
        else:
            if replay_contract.admission_type != request.admission_type:
                raise IdempotencyConflict(
                    "admission idempotency identity belongs to another type"
                )
            definition = self._admission_registry.resolve_digest(
                replay_contract.admission_definition_digest
            )
            rights_policy = self._rights_policies.resolve_digest(
                replay_contract.rights_policy_contract_digest
            )
        semantic = digest_canonical(
            {
                "admission_type": request.admission_type,
                "definition_digest": definition.digest,
                "rights_policy_contract_digest": (
                    rights_policy.contract_digest
                ),
                "idempotency_key": request.idempotency_key,
            }
        )
        if (
            replay_contract is not None
            and semantic
            != replay_contract.stable_semantic_request_digest
        ):
            raise IdempotencyConflict(
                "retained admission semantics cannot be reconstructed"
            )
        preflight_id = ObjectPreflightId.new()
        authz_request, authz = self._authorize(
            authentication=authentication,
            now=checked_at,
            operation_type=(
                f"object:admission:preflight:{definition.admission_type}"
            ),
            required_scope=definition.required_write_scope,
            stable_digest=semantic,
            definition_digest=definition.digest,
            aggregate_type="governed_object_admission",
            aggregate_id=str(preflight_id),
            object_class=definition.object_class,
            allowed_use=definition.allowed_use,
            security_scope=definition.security_scope,
            retention_scope=definition.retention_scope,
        )
        # Known rights denial happens only after current authn/authz, but before
        # the input source is touched.
        if not rights_policy.preflight_allowed:
            raise ObjectAdmissionDenied(rights_policy.reason_code)
        if replay_contract is None:
            # An expired activation is not a fresh admission. Detect the exact
            # existing reservation before staging bytes or retaining preflight.
            self._store.find(
                idempotency_namespace=_derive_idempotency_namespace(
                    authentication, self._command_registry.resolve('object.admission.activate')),
                idempotency_key='lifecycle-activate:' + request.idempotency_key,
            )
        preflight = self._object_issuer.issue_preflight(
            preflight_id=preflight_id,
            request=request,
            definition=definition,
            rights_policy=rights_policy,
            authentication=authentication,
            authorization_request=authz_request,
            authorization=authz,
            idempotency_namespace=namespace,
            stable_semantic_request_digest=semantic,
            checked_at=checked_at,
            expires_at=UtcTimestamp(
                checked_at.value
                + timedelta(seconds=rights_policy.preflight_ttl_seconds)
            ),
        )
        return preflight, replay_contract is not None

    def committed_admission(
        self,
        request: ObjectAdmissionRequest,
        proof: AuthenticationProof,
    ) -> ObjectAdmissionResult | None:
        preflight, has_replay_contract = self._admission_preflight(request, proof)
        existing = self._store.find_admission_replay(preflight)
        if existing is not None:
            return ObjectAdmissionResult(existing, replayed=True)
        if has_replay_contract:
            raise IdempotencyConflict(
                "committed admission replay contract has no exact result"
            )
        return None

    def admit(
        self,
        request: ObjectAdmissionRequest,
        source: _Source,
        proof: AuthenticationProof,
    ) -> ObjectAdmissionResult:
        preflight, has_replay_contract = self._admission_preflight(request, proof)
        existing = self._store.find_admission_replay(preflight)
        if existing is not None:
            return ObjectAdmissionResult(existing, replayed=True)
        if has_replay_contract:
            raise IdempotencyConflict(
                "committed admission replay contract has no exact result"
            )
        definition = preflight.definition
        rights_policy = preflight.rights_policy

        staged = self._cas.stage(
            source, object_class=definition.object_class
        )
        try:
            self._store.record_stage(preflight, staged)
            final_authentication, final_now = self._authenticate(proof)
            final_semantic = digest_canonical(
                {
                    "preflight_digest": preflight.digest,
                    "blob": staged.identity.canonical_value(),
                    "stage_id": staged.stage_id,
                    "staged_name": staged.staged_name,
                }
            )
            final_request, final_authz = self._authorize(
                authentication=final_authentication,
                now=final_now,
                operation_type=(
                    f"object:admission:commit:{definition.admission_type}"
                ),
                required_scope=definition.required_write_scope,
                stable_digest=final_semantic,
                definition_digest=definition.digest,
                aggregate_type="governed_object_admission",
                aggregate_id=str(preflight.preflight_id),
                object_class=definition.object_class,
                allowed_use=definition.allowed_use,
                security_scope=definition.security_scope,
                retention_scope=definition.retention_scope,
            )
            valid_from = UtcTimestamp(
                final_now.value
                + timedelta(
                    seconds=rights_policy.valid_from_delay_seconds
                )
            )
            valid_until = (
                None
                if rights_policy.validity_seconds is None
                else UtcTimestamp(
                    valid_from.value
                    + timedelta(seconds=rights_policy.validity_seconds)
                )
            )
            rights_request_digest = digest_canonical(
                {
                    "preflight_digest": preflight.digest,
                    "blob": staged.identity.canonical_value(),
                    "final_authentication_context_digest": (
                        final_authentication.digest
                    ),
                    "final_authorization_request_digest": (
                        final_request.request_digest
                    ),
                    "final_authorization_decision_digest": final_authz.digest,
                    "rights_policy_contract_digest": (
                        rights_policy.contract_digest
                    ),
                }
            )
            rights = _RightsProposal(
                policy_contract_digest=rights_policy.contract_digest,
                admission_definition_digest=definition.digest,
                blob=staged.identity,
                object_class=definition.object_class,
                allowed_use=definition.allowed_use,
                security_scope=definition.security_scope,
                retention_scope=definition.retention_scope,
                allowed=rights_policy.preflight_allowed,
                reason_code=rights_policy.reason_code,
                decided_at=final_now,
                valid_from=valid_from,
                valid_until=valid_until,
                rights_request_digest=rights_request_digest,
            )
            grant = self._object_issuer.issue_admission(
                operation_id=ObjectOperationId.new(),
                preflight=preflight,
                blob=staged.identity,
                stage_id=staged.stage_id,
                staged_name=staged.staged_name,
                authentication=final_authentication,
                authorization_request=final_request,
                authorization=final_authz,
                rights=rights,
                stable_semantic_request_digest=final_semantic,
            )

            def lifecycle_factory(
                admission_id: ObjectAdmissionId,
                _rights_id: RightsDecisionId,
                payload: dict[str, object],
            ) -> _AuthorizedCommandGrant:
                return self._lifecycle_command_grant(
                    "object.admission.activate",
                    admission_id,
                    payload,
                    proof=proof,
                    idempotency_key=(
                        f"lifecycle-activate:{request.idempotency_key}"
                    ),
                )

            committed = self._store.commit_admission(
                grant, lifecycle_grant_factory=lifecycle_factory
            )
            return ObjectAdmissionResult(
                committed.admission, replayed=committed.replayed
            )
        except Exception:
            try:
                self._store.fail_stage(
                    staged.stage_id, failure_code="ADMISSION_FAILED"
                )
            finally:
                # A failure before the staging row is committed must not leak
                # bytes.  ``discard_stage`` is idempotent and removes only the
                # temporary path, never the installed content-addressed blob.
                if staged.path.exists():
                    self._cas.discard_stage(staged)
            raise

    def hydrate(
        self, request: HydrationRequest, proof: AuthenticationProof
    ) -> HydratedObject:
        return self._hydrate(request, proof, in_transaction=False)

    def rehydrate(
        self, request: HydrationRequest, proof: AuthenticationProof,
        *, in_transaction: bool = False,
    ) -> HydratedObject:
        return self._hydrate(
            request, proof, in_transaction=in_transaction, reuse_retained=True
        )

    def hydrate_in_transaction(
        self, request: HydrationRequest, proof: AuthenticationProof
    ) -> HydratedObject:
        """Hydrate inside the cumulative writer's existing transaction."""
        return self._hydrate(request, proof, in_transaction=True)

    def _hydrate(
        self,
        request: HydrationRequest,
        proof: AuthenticationProof,
        *,
        in_transaction: bool,
        reuse_retained: bool = False,
    ) -> HydratedObject:
        if not isinstance(request, HydrationRequest):
            raise TypeError("request must be HydrationRequest")
        authentication, now = self._authenticate(proof)
        policy = self._hydration_policies.resolve_for_purpose(
            request.purpose
        )
        semantic = digest_canonical(
            {
                "policy_contract_digest": policy.contract_digest,
                "admission_id": str(request.admission_id),
                "purpose": request.purpose,
                "offset": request.offset,
                "length": request.length,
            }
        )
        authz_request, authz = self._authorize(
            authentication=authentication,
            now=now,
            operation_type=f"object:hydrate:{policy.purpose}",
            required_scope=policy.required_scope,
            stable_digest=semantic,
            definition_digest=policy.contract_digest,
            aggregate_type="governed_object_hydration",
            aggregate_id=str(request.admission_id),
            object_class=None,
            allowed_use=None,
            security_scope="authority.object_hydration",
            retention_scope="authority.audit",
        )
        grant = self._object_issuer.issue_hydration(
            operation_id=ObjectOperationId.new(),
            request=request,
            policy=policy,
            authentication=authentication,
            authorization_request=authz_request,
            authorization=authz,
            stable_semantic_request_digest=semantic,
            decided_at=now,
        )
        data, decision = self._store.hydrate(
            grant, in_transaction=in_transaction, reuse_retained=reuse_retained
        )
        return HydratedObject(data=data, decision=decision)

    def latest_access_decision(
        self,
        admission_id: ObjectAdmissionId,
        purpose: str,
        proof: AuthenticationProof,
    ) -> ObjectAccessDecisionView:
        """Authenticate a read-only replay of retained exact access authority."""

        if not isinstance(admission_id, ObjectAdmissionId):
            raise TypeError("admission identity must be ObjectAdmissionId")
        authentication, now = self._authenticate(proof)
        policy = self._hydration_policies.resolve_for_purpose(purpose)
        semantic = digest_canonical(
            {
                "admission_id": str(admission_id),
                "policy_contract_digest": policy.contract_digest,
                "purpose": purpose,
            }
        )
        self._authorize(
            authentication=authentication,
            now=now,
            operation_type=f"object:access-decision:read:{purpose}",
            required_scope=policy.required_scope,
            stable_digest=semantic,
            definition_digest=policy.contract_digest,
            aggregate_type="governed_object_hydration",
            aggregate_id=str(admission_id),
            object_class=None,
            allowed_use=None,
            security_scope="authority.object_hydration",
            retention_scope="authority.audit",
        )
        return self._store.latest_access_decision(
            admission_id=admission_id,
            policy_contract_digest=policy.contract_digest,
            principal_id=authentication.principal_id,
            authority_domain=authentication.authority_domain,
            purpose=purpose,
        )

    def access_decision(
        self,
        access_decision_id: ObjectAccessDecisionId,
        admission_id: ObjectAdmissionId,
        purpose: str,
        proof: AuthenticationProof,
    ) -> ObjectAccessDecisionView:
        """Authenticate a read-only lookup of one retained access receipt."""

        if (
            not isinstance(access_decision_id, ObjectAccessDecisionId)
            or not isinstance(admission_id, ObjectAdmissionId)
        ):
            raise TypeError("access and admission identities are required")
        authentication, now = self._authenticate(proof)
        policy = self._hydration_policies.resolve_for_purpose(purpose)
        semantic = digest_canonical({
            "access_decision_id": str(access_decision_id),
            "admission_id": str(admission_id),
            "policy_contract_digest": policy.contract_digest,
            "purpose": purpose,
        })
        self._authorize(
            authentication=authentication,
            now=now,
            operation_type=f"object:access-decision:read:{purpose}",
            required_scope=policy.required_scope,
            stable_digest=semantic,
            definition_digest=policy.contract_digest,
            aggregate_type="governed_object_hydration",
            aggregate_id=str(admission_id),
            object_class=None,
            allowed_use=None,
            security_scope="authority.object_hydration",
            retention_scope="authority.audit",
        )
        view = self._store.access_decision_view(access_decision_id)
        if (
            view.admission_id != admission_id
            or view.policy_contract_digest != policy.contract_digest
            or view.principal_id != authentication.principal_id
            or view.authority_domain != authentication.authority_domain
            or view.purpose != purpose
        ):
            raise PermissionError("retained access decision binding differs")
        return view

    def _maintenance(
        self,
        *,
        operation_type: str,
        command_type: str,
        target_identity: str,
        reason_code: str,
        idempotency_key: str,
        proof: AuthenticationProof,
    ) -> tuple[_MaintenanceGrant, Callable[[str, UUIDv4Id, dict[str, object]], _AuthorizedCommandGrant]]:
        authentication, now = self._authenticate(proof)
        definition = self._command_registry.resolve(command_type)
        semantic = digest_canonical(
            {
                "operation_type": operation_type,
                "target_identity": target_identity,
                "reason_code": reason_code,
                "lifecycle_definition_digest": definition.digest,
                "idempotency_key": idempotency_key,
            }
        )
        namespace = digest_canonical(
            {
                "authority_domain": authentication.authority_domain,
                "principal_id": authentication.principal_id,
                "operation_type": operation_type,
            }
        )
        authz_request, authz = self._authorize(
            authentication=authentication,
            now=now,
            operation_type=f"object:maintenance:{operation_type}",
            required_scope=definition.required_scope,
            stable_digest=semantic,
            definition_digest=definition.digest,
            aggregate_type="governed_object_lifecycle",
            aggregate_id=target_identity,
            object_class=None,
            allowed_use=None,
            security_scope="authority.object_lifecycle",
            retention_scope="authority.audit",
        )
        grant = self._object_issuer.issue_maintenance(
            operation_id=ObjectOperationId.new(),
            operation_type=operation_type,
            target_identity=target_identity,
            reason_code=reason_code,
            lifecycle_definition=definition,
            authentication=authentication,
            authorization_request=authz_request,
            authorization=authz,
            idempotency_key=idempotency_key,
            idempotency_namespace=namespace,
            stable_semantic_request_digest=semantic,
            decided_at=now,
        )

        def factory(
            actual_command_type: str,
            target: UUIDv4Id,
            payload: dict[str, object],
        ) -> _AuthorizedCommandGrant:
            return self._lifecycle_command_grant(
                actual_command_type,
                target,
                payload,
                proof=proof,
                idempotency_key=(
                    f"lifecycle:{operation_type}:{idempotency_key}"
                ),
            )

        return grant, factory

    def revoke(
        self,
        admission_id: ObjectAdmissionId,
        reason_code: str,
        idempotency_key: str,
        proof: AuthenticationProof,
    ) -> AdmissionRevocationView:
        grant, factory = self._maintenance(
            operation_type="ADMISSION_REVOKE",
            command_type="object.admission.revoke",
            target_identity=str(admission_id),
            reason_code=reason_code,
            idempotency_key=idempotency_key,
            proof=proof,
        )
        return self._store.revoke_admission(
            grant, lifecycle_grant_factory=factory
        )

    def request_deletion(
        self,
        blob_digest: str,
        reason_code: str,
        idempotency_key: str,
        proof: AuthenticationProof,
    ) -> GovernedDeletionView:
        grant, factory = self._maintenance(
            operation_type="DELETION_REQUEST",
            command_type="object.deletion.request",
            target_identity=blob_digest,
            reason_code=reason_code,
            idempotency_key=idempotency_key,
            proof=proof,
        )
        return self._store.request_deletion(
            grant, lifecycle_grant_factory=factory
        )

    def tombstone(
        self,
        deletion_id: GovernedDeletionId,
        reason_code: str,
        idempotency_key: str,
        proof: AuthenticationProof,
    ) -> GovernedDeletionView:
        grant, factory = self._maintenance(
            operation_type="DELETION_TOMBSTONE",
            command_type="object.deletion.tombstone",
            target_identity=str(deletion_id),
            reason_code=reason_code,
            idempotency_key=idempotency_key,
            proof=proof,
        )
        return self._store.tombstone_deletion(
            grant, lifecycle_grant_factory=factory
        )

    def complete_deletion(
        self,
        deletion_id: GovernedDeletionId,
        idempotency_key: str,
        proof: AuthenticationProof,
    ) -> GovernedDeletionView:
        grant, factory = self._maintenance(
            operation_type="DELETION_COMPLETE",
            command_type="object.deletion.complete",
            target_identity=str(deletion_id),
            reason_code="DELETE_PHYSICAL_BYTES",
            idempotency_key=idempotency_key,
            proof=proof,
        )

        def failure_factory(
            command_type: str,
            target: UUIDv4Id,
            payload: dict[str, object],
        ) -> _AuthorizedCommandGrant:
            return self._lifecycle_command_grant(
                command_type,
                target,
                payload,
                proof=proof,
                idempotency_key=f"failure:{idempotency_key}",
            )

        return self._store.complete_deletion(
            grant,
            lifecycle_grant_factory=factory,
            failure_grant_factory=failure_factory,
        )

    def create_pin(
        self,
        blob_digest: str,
        reason_code: str,
        idempotency_key: str,
        proof: AuthenticationProof,
    ) -> RecoveryPinView:
        grant, factory = self._maintenance(
            operation_type="RECOVERY_PIN_CREATE",
            command_type="object.recovery_pin.create",
            target_identity=blob_digest,
            reason_code=reason_code,
            idempotency_key=idempotency_key,
            proof=proof,
        )
        return self._store.create_recovery_pin(
            grant, lifecycle_grant_factory=factory
        )

    def release_pin(
        self,
        pin_id: RecoveryPinId,
        reason_code: str,
        idempotency_key: str,
        proof: AuthenticationProof,
    ) -> RecoveryPinView:
        grant, factory = self._maintenance(
            operation_type="RECOVERY_PIN_RELEASE",
            command_type="object.recovery_pin.release",
            target_identity=str(pin_id),
            reason_code=reason_code,
            idempotency_key=idempotency_key,
            proof=proof,
        )
        return self._store.release_recovery_pin(
            grant, lifecycle_grant_factory=factory
        )

    def collect_orphans(
        self, proof: AuthenticationProof
    ) -> tuple[BlobIdentity, ...]:
        removed: list[BlobIdentity] = []
        for blob in self._store.orphan_candidates():
            grant, factory = self._maintenance(
                operation_type="ORPHAN_REMOVE",
                command_type="object.orphan.remove",
                target_identity=blob.blob_digest,
                reason_code="UNREFERENCED_ORPHAN",
                idempotency_key=f"orphan:{blob.blob_digest}",
                proof=proof,
            )
            self._store.remove_orphan(
                grant, lifecycle_grant_factory=factory
            )
            removed.append(blob)
        return tuple(removed)


def open_governed_object_authority_system(
    *,
    path: Path,
    object_root: Path,
    registry: CommandRegistry,
    payload_schemas: PayloadSchemaRegistry,
    admission_registry: ObjectAdmissionRegistry,
    rights_policies: RightsPolicyRegistry,
    hydration_policies: HydrationPolicyRegistry,
    authenticator: Any,
    authorizer: Any,
    event_read_policy: EventReadPolicy,
    object_limits: ObjectLimits,
    command_service_version: str = "authority-command-v1",
    busy_timeout_ms: int = 5_000,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
    cas_fault_hook: Callable[[str], None] | None = None,
    disk_usage: Callable[[Path], Any] | None = None,
) -> GovernedObjectAuthoritySystem:
    """Open the single-writer A2b SQLite/CAS authority system."""

    merged_registry, merged_schemas = merge_authority_registries(
        command_registry=registry, payload_schemas=payload_schemas
    )
    command_issuer = _CapabilityIssuer(
        command_registry=merged_registry,
        payload_schemas=merged_schemas,
    )
    object_issuer = _ObjectCapabilityIssuer(
        admission_registry=admission_registry,
        rights_policies=rights_policies,
        hydration_policies=hydration_policies,
        command_registry=merged_registry,
    )
    cas_kwargs: dict[str, Any] = {
        "limits": object_limits,
        "clock": clock,
        "fault_hook": cas_fault_hook,
    }
    if disk_usage is not None:
        cas_kwargs["disk_usage"] = disk_usage
    cas = _GovernedCAS(object_root, **cas_kwargs)
    store: _GovernedObjectAuthorityStore | None = None
    try:
        store = _GovernedObjectAuthorityStore(
            path,
            issuer=command_issuer,
            object_issuer=object_issuer,
            command_registry=merged_registry,
            payload_schemas=merged_schemas,
            admission_registry=admission_registry,
            rights_policies=rights_policies,
            hydration_policies=hydration_policies,
            cas=cas,
            command_service_version=command_service_version,
            busy_timeout_ms=busy_timeout_ms,
            clock=clock,
        )
        command_service = CommandService(
            registry=merged_registry,
            payload_schemas=merged_schemas,
            authenticator=authenticator,
            authorizer=authorizer,
            admission_lookup=store,
            committed_lookup=store,
            clock=clock,
            _issuer=command_issuer,
        )
        read_boundary = _ReadBoundary(
            store=store,
            policy=event_read_policy,
            authenticator=authenticator,
            authorizer=authorizer,
            clock=clock,
        )
        boundary = _ObjectBoundary(
            store=store,
            cas=cas,
            object_issuer=object_issuer,
            admission_registry=admission_registry,
            rights_policies=rights_policies,
            hydration_policies=hydration_policies,
            authenticator=authenticator,
            authorizer=authorizer,
            command_service=command_service,
            command_registry=merged_registry,
            clock=clock,
        )

        def execute(
            command: SemanticCommand, proof: AuthenticationProof
        ) -> CommittedCommand:
            if command.command_type in LIFECYCLE_COMMAND_TYPES:
                raise PermissionError(
                    "object lifecycle commands are internal authority operations"
                )
            grant = command_service._authorize_for_commit(
                command, proof=proof
            )
            return store.commit(grant)  # type: ignore[union-attr]

        return GovernedObjectAuthoritySystem(
            commands=AuthorityCommands(execute),
            events=AuthorityEvents(
                policy_id=event_read_policy.policy_id,
                read=read_boundary.events_after,
                provenance=read_boundary.provenance,
                result=read_boundary.command_result,
            ),
            objects=GovernedObjects(
                admit=boundary.admit,
                committed_admission=boundary.committed_admission,
                hydrate=boundary.hydrate,
                rehydrate=boundary.rehydrate,
                access_decision=boundary.access_decision,
                latest_access_decision=boundary.latest_access_decision,
                revoke=boundary.revoke,
                request_deletion=boundary.request_deletion,
                tombstone=boundary.tombstone,
                complete_deletion=boundary.complete_deletion,
                create_pin=boundary.create_pin,
                release_pin=boundary.release_pin,
                collect_orphans=boundary.collect_orphans,
            ),
            close=store.close,
        )
    except Exception:
        if store is not None:
            store.close()
        raise


__all__ = [
    "GovernedObjectAuthoritySystem",
    "GovernedObjects",
    "HydratedObject",
    "ObjectAdmissionResult",
    "open_governed_object_authority_system",
]
