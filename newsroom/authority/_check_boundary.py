from __future__ import annotations

from typing import Any, Callable, TypeVar

from newsroom.authority._check_store import _CheckAuthorityStore
from newsroom.authority._security import _AuthorizationRequest
from newsroom.authority.auth import AuthenticationProof
from newsroom.authority.canonical import digest_canonical
from newsroom.authority.models import InlinePayload, SemanticCommand
from newsroom.authority.service import CommandService
from newsroom.authority.types import AggregateId, UtcTimestamp
from newsroom.checks.baseline_models import BaselineDecisionRequest
from newsroom.checks.check_models import (
    CheckAttemptRequest,
    CheckOutcomeRequest,
    CheckRequestRequest,
)
from newsroom.checks.finding_models import (
    OperationalFindingOccurrenceRequest,
    OperationalFindingRequest,
)
from newsroom.checks.policy import (
    CHECK_ATTEMPT_START_COMMAND,
    CHECK_BASELINE_DECIDE_COMMAND,
    CHECK_OUTCOME_RECORD_COMMAND,
    CHECK_REQUEST_REGISTER_COMMAND,
    OBSERVABLE_TRANSITION_RECORD_COMMAND,
    OPERATIONAL_FINDING_OCCURRENCE_RECORD_COMMAND,
    OPERATIONAL_FINDING_OPEN_COMMAND,
)
from newsroom.checks.read_policy import DiscoveryCheckReadPolicy
from newsroom.checks.record_models import (
    BaselineDecision,
    CheckAttempt,
    CheckOutcome,
    CheckRequest,
    ObservableTransition,
    OperationalFinding,
    OperationalFindingOccurrence,
)
from newsroom.checks.transition_models import ObservableTransitionRequest
from newsroom.checks.types import (
    BaselineDecisionId,
    CheckAttemptId,
    CheckRequestId,
    ObservableTransitionId,
    OperationalFindingId,
)
from newsroom.sources import CheckOutcomeId, SourceDefinitionId, SourceItemId, SourceRevisionId


_Record = TypeVar("_Record")
_CHECK_READ_SCHEMA_DIGEST = digest_canonical(
    {
        "contract": "discovery-check-read-no-payload-v1",
        "payload_mode": "NO_PAYLOAD",
        "redaction": "typed-record-or-policy-bounded-list",
    }
)


class _CheckBoundary:
    def __init__(
        self,
        *,
        store: _CheckAuthorityStore,
        command_service: CommandService,
        authenticator: Any,
        authorizer: Any,
        read_policy: DiscoveryCheckReadPolicy,
        clock: Callable[[], UtcTimestamp],
    ) -> None:
        self._store = store
        self._command_service = command_service
        self._authenticator = authenticator
        self._authorizer = authorizer
        self._read_policy = read_policy
        self._clock = clock

    def _commit(
        self,
        request: Any,
        proof: AuthenticationProof,
        *,
        command_type: str,
        aggregate_id: AggregateId,
        commit: Callable[..., _Record],
    ) -> _Record:
        command = SemanticCommand(
            command_type=command_type,
            aggregate_id=aggregate_id,
            expected_aggregate_version=0,
            payload=InlinePayload(request.canonical_value()),
            idempotency_key=request.idempotency_key,
        )
        grant = self._command_service._authorize_for_commit(
            command,
            proof=proof,
        )
        return commit(grant, request=request)

    def register_request(
        self,
        request: CheckRequestRequest,
        proof: AuthenticationProof,
    ) -> CheckRequest:
        if not isinstance(request, CheckRequestRequest):
            raise TypeError("Check Request must be typed")
        return self._commit(
            request,
            proof,
            command_type=CHECK_REQUEST_REGISTER_COMMAND,
            aggregate_id=AggregateId(request.request_id.value),
            commit=self._store.commit_check_request,
        )

    def start_attempt(
        self,
        request: CheckAttemptRequest,
        proof: AuthenticationProof,
    ) -> CheckAttempt:
        if not isinstance(request, CheckAttemptRequest):
            raise TypeError("Check Attempt must be typed")
        return self._commit(
            request,
            proof,
            command_type=CHECK_ATTEMPT_START_COMMAND,
            aggregate_id=AggregateId(request.attempt_id.value),
            commit=self._store.commit_check_attempt,
        )

    def record_outcome(
        self,
        request: CheckOutcomeRequest,
        proof: AuthenticationProof,
    ) -> CheckOutcome:
        if not isinstance(request, CheckOutcomeRequest):
            raise TypeError("Check Outcome must be typed")
        return self._commit(
            request,
            proof,
            command_type=CHECK_OUTCOME_RECORD_COMMAND,
            aggregate_id=AggregateId(request.outcome_id.value),
            commit=self._store.commit_check_outcome,
        )

    def decide_baseline(
        self,
        request: BaselineDecisionRequest,
        proof: AuthenticationProof,
    ) -> BaselineDecision:
        if not isinstance(request, BaselineDecisionRequest):
            raise TypeError("Baseline Decision must be typed")
        return self._commit(
            request,
            proof,
            command_type=CHECK_BASELINE_DECIDE_COMMAND,
            aggregate_id=AggregateId(request.decision_id.value),
            commit=self._store.commit_baseline_decision,
        )

    def record_transition(
        self,
        request: ObservableTransitionRequest,
        proof: AuthenticationProof,
    ) -> ObservableTransition:
        if not isinstance(request, ObservableTransitionRequest):
            raise TypeError("Observable Transition must be typed")
        return self._commit(
            request,
            proof,
            command_type=OBSERVABLE_TRANSITION_RECORD_COMMAND,
            aggregate_id=AggregateId(request.transition_id.value),
            commit=self._store.commit_observable_transition,
        )

    def open_finding(
        self,
        request: OperationalFindingRequest,
        proof: AuthenticationProof,
    ) -> OperationalFinding:
        if not isinstance(request, OperationalFindingRequest):
            raise TypeError("Operational Finding must be typed")
        return self._commit(
            request,
            proof,
            command_type=OPERATIONAL_FINDING_OPEN_COMMAND,
            aggregate_id=AggregateId(request.finding_id.value),
            commit=self._store.commit_operational_finding,
        )

    def record_finding_occurrence(
        self,
        request: OperationalFindingOccurrenceRequest,
        proof: AuthenticationProof,
    ) -> OperationalFindingOccurrence:
        if not isinstance(request, OperationalFindingOccurrenceRequest):
            raise TypeError("Finding occurrence must be typed")
        return self._commit(
            request,
            proof,
            command_type=OPERATIONAL_FINDING_OCCURRENCE_RECORD_COMMAND,
            aggregate_id=AggregateId(request.occurrence_id.value),
            commit=self._store.commit_operational_finding_occurrence,
        )

    def _authorize_read(
        self,
        proof: AuthenticationProof,
        *,
        operation: str,
        aggregate_type: str,
        aggregate_id: str,
        sensitive: bool,
        limit: int | None = None,
    ) -> None:
        now = self._clock()
        authentication = self._authenticator.authenticate(proof, now=now)
        authentication.require_current(now)
        self._read_policy.require_principal(authentication.principal_id)
        if limit is not None:
            self._read_policy.require_limit(limit)
        required_scope = (
            self._read_policy.sensitive_required_scope
            if sensitive
            else self._read_policy.metadata_required_scope
        )
        stable = digest_canonical(
            {
                "contract": "discovery-check-read-v1",
                "policy_digest": self._read_policy.digest,
                "operation": operation,
                "aggregate_type": aggregate_type,
                "aggregate_id": aggregate_id,
                "sensitive": sensitive,
                "limit": limit,
            }
        )
        unsigned = {
            "authentication_context_id": str(
                authentication.authentication_context_id
            ),
            "principal_id": authentication.principal_id,
            "authority_domain": authentication.authority_domain,
            "operation_type": operation,
            "required_scope": required_scope,
            "stable_semantic_request_digest": stable,
            "command_definition_digest": _CHECK_READ_SCHEMA_DIGEST,
            "aggregate_type": aggregate_type,
            "aggregate_id": aggregate_id,
            "event_type": "discovery.check.read",
            "event_schema_version": 1,
            "payload_mode": "NO_PAYLOAD",
            "payload_schema_version": "discovery_check_read_v1",
            "payload_schema_contract_version": (
                "discovery-check-read-no-payload-v1"
            ),
            "payload_schema_contract_digest": _CHECK_READ_SCHEMA_DIGEST,
            "payload_canonicalizer_version": "discovery-check-none-v1",
            "trust_scope": "ADMITTED",
            "security_scope": "authority.discovery_checks",
            "retention_scope": "authority.audit",
            "object_class": None,
            "allowed_use": None,
        }
        request = _AuthorizationRequest(
            authentication_context_id=authentication.authentication_context_id,
            principal_id=authentication.principal_id,
            authority_domain=authentication.authority_domain,
            operation_type=operation,
            required_scope=required_scope,
            stable_semantic_request_digest=stable,
            command_definition_digest=_CHECK_READ_SCHEMA_DIGEST,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            event_type="discovery.check.read",
            event_schema_version=1,
            payload_mode="NO_PAYLOAD",
            payload_schema_version="discovery_check_read_v1",
            payload_schema_contract_version=(
                "discovery-check-read-no-payload-v1"
            ),
            payload_schema_contract_digest=_CHECK_READ_SCHEMA_DIGEST,
            payload_canonicalizer_version="discovery-check-none-v1",
            trust_scope="ADMITTED",
            security_scope="authority.discovery_checks",
            retention_scope="authority.audit",
            object_class=None,
            allowed_use=None,
            request_digest=digest_canonical(unsigned),
        )
        decision = self._authorizer.authorize(
            authentication,
            request,
            now=now,
        )
        if (
            decision.authentication_context_id
            != authentication.authentication_context_id
            or decision.authorization_request_digest
            != request.request_digest
        ):
            raise PermissionError(
                "Check read authorization provenance differs"
            )
        decision.require_allowed()

    def request(
        self, request_id: CheckRequestId, proof: AuthenticationProof
    ) -> CheckRequest:
        if not isinstance(request_id, CheckRequestId):
            raise TypeError("Check Request identity must be typed")
        self._authorize_read(
            proof,
            operation="read:discovery_checks:request",
            aggregate_type="check_request",
            aggregate_id=str(request_id),
            sensitive=True,
        )
        value = self._store.check_request(request_id)
        if value is None:
            raise LookupError("Check Request is not retained")
        return value

    def attempt(
        self, attempt_id: CheckAttemptId, proof: AuthenticationProof
    ) -> CheckAttempt:
        if not isinstance(attempt_id, CheckAttemptId):
            raise TypeError("Check Attempt identity must be typed")
        self._authorize_read(
            proof,
            operation="read:discovery_checks:attempt",
            aggregate_type="check_attempt",
            aggregate_id=str(attempt_id),
            sensitive=True,
        )
        value = self._store.check_attempt(attempt_id)
        if value is None:
            raise LookupError("Check Attempt is not retained")
        return value

    def outcome(
        self, outcome_id: CheckOutcomeId, proof: AuthenticationProof
    ) -> CheckOutcome:
        if not isinstance(outcome_id, CheckOutcomeId):
            raise TypeError("Check Outcome identity must be typed")
        self._authorize_read(
            proof,
            operation="read:discovery_checks:outcome",
            aggregate_type="check_outcome",
            aggregate_id=str(outcome_id),
            sensitive=True,
        )
        value = self._store.check_outcome(outcome_id)
        if value is None:
            raise LookupError("Check Outcome is not retained")
        return value

    def observed_prior_revision(
        self, item_id: SourceItemId, request_id: CheckRequestId,
        outcome_id: CheckOutcomeId, completed_at: UtcTimestamp,
        proof: AuthenticationProof,
    ) -> SourceRevisionId | None:
        if not isinstance(item_id, SourceItemId) or not isinstance(request_id, CheckRequestId):
            raise TypeError("observed prior requires typed Item and Check Request identities")
        if not isinstance(outcome_id, CheckOutcomeId) or not isinstance(completed_at, UtcTimestamp):
            raise TypeError("observed prior requires a typed Check Outcome and UTC boundary")
        self._authorize_read(
            proof, operation="read:discovery_checks:observed_prior_revision",
            aggregate_type="check_request",
            aggregate_id=digest_canonical({
                "request_id": str(request_id), "item_id": str(item_id),
                "outcome_id": str(outcome_id), "completed_at": completed_at.to_text(),
            }), sensitive=True,
        )
        return self._store.observed_prior_revision(
            item_id, request_id=request_id, outcome_id=outcome_id,
            completed_at=completed_at,
        )

    def attempts(
        self,
        request_id: CheckRequestId,
        limit: int,
        proof: AuthenticationProof,
    ) -> tuple[CheckAttempt, ...]:
        self._authorize_read(
            proof,
            operation="read:discovery_checks:attempts",
            aggregate_type="check_request",
            aggregate_id=str(request_id),
            sensitive=True,
            limit=limit,
        )
        return self._store.attempts_for_request(request_id, limit=limit)

    def outcomes(
        self,
        request_id: CheckRequestId,
        limit: int,
        proof: AuthenticationProof,
    ) -> tuple[CheckOutcome, ...]:
        self._authorize_read(
            proof,
            operation="read:discovery_checks:outcomes",
            aggregate_type="check_request",
            aggregate_id=str(request_id),
            sensitive=True,
            limit=limit,
        )
        return self._store.outcomes_for_request(request_id, limit=limit)

    def baseline(
        self,
        decision_id: BaselineDecisionId,
        proof: AuthenticationProof,
    ) -> BaselineDecision:
        self._authorize_read(
            proof,
            operation="read:discovery_checks:baseline",
            aggregate_type="baseline_decision",
            aggregate_id=str(decision_id),
            sensitive=True,
        )
        value = self._store.baseline_decision(decision_id)
        if value is None:
            raise LookupError("Baseline Decision is not retained")
        return value

    def current_baseline(
        self,
        definition_id: SourceDefinitionId,
        proof: AuthenticationProof,
    ) -> BaselineDecision:
        self._authorize_read(
            proof,
            operation="read:discovery_checks:current_baseline",
            aggregate_type="source_definition",
            aggregate_id=str(definition_id),
            sensitive=True,
        )
        value = self._store.current_baseline_decision(definition_id)
        if value is None:
            raise LookupError("current Baseline Decision is not retained")
        return value

    def transition(
        self,
        transition_id: ObservableTransitionId,
        proof: AuthenticationProof,
    ) -> ObservableTransition:
        self._authorize_read(
            proof,
            operation="read:discovery_checks:transition",
            aggregate_type="observable_transition",
            aggregate_id=str(transition_id),
            sensitive=True,
        )
        value = self._store.observable_transition(transition_id)
        if value is None:
            raise LookupError("Observable Transition is not retained")
        return value

    def finding(
        self,
        finding_id: OperationalFindingId,
        proof: AuthenticationProof,
    ) -> OperationalFinding:
        self._authorize_read(
            proof,
            operation="read:discovery_checks:finding",
            aggregate_type="operational_finding",
            aggregate_id=str(finding_id),
            sensitive=True,
        )
        value = self._store.operational_finding(finding_id)
        if value is None:
            raise LookupError("Operational Finding is not retained")
        return value

    def finding_occurrences(
        self,
        finding_id: OperationalFindingId,
        limit: int,
        proof: AuthenticationProof,
    ) -> tuple[OperationalFindingOccurrence, ...]:
        self._authorize_read(
            proof,
            operation="read:discovery_checks:finding_occurrences",
            aggregate_type="operational_finding",
            aggregate_id=str(finding_id),
            sensitive=True,
            limit=limit,
        )
        return self._store.finding_occurrences(finding_id, limit=limit)


__all__ = ["_CheckBoundary"]
