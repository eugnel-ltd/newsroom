from __future__ import annotations

from typing import Callable

from newsroom.authority.auth import AuthenticationProof
from newsroom.authority.types import UtcTimestamp
from newsroom.checks.admission_models import (
    ProposalAdmissionRequest,
    ProposalAdmissionResult,
)
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


class GovernedChecks:
    __slots__ = (
        "__register_request",
        "__start_attempt",
        "__record_outcome",
        "__decide_baseline",
        "__record_transition",
        "__open_finding",
        "__record_finding_occurrence",
        "__admit_proposal",
        "__request",
        "__attempt",
        "__outcome",
        "__observed_prior_revision",
        "__attempts",
        "__outcomes",
        "__baseline",
        "__current_baseline",
        "__transition",
        "__finding",
        "__finding_occurrences",
    )

    def __init__(
        self,
        *,
        register_request: Callable[..., CheckRequest],
        start_attempt: Callable[..., CheckAttempt],
        record_outcome: Callable[..., CheckOutcome],
        decide_baseline: Callable[..., BaselineDecision],
        record_transition: Callable[..., ObservableTransition],
        open_finding: Callable[..., OperationalFinding],
        record_finding_occurrence: Callable[
            ..., OperationalFindingOccurrence
        ],
        admit_proposal: Callable[..., ProposalAdmissionResult],
        request: Callable[..., CheckRequest],
        attempt: Callable[..., CheckAttempt],
        outcome: Callable[..., CheckOutcome],
        observed_prior_revision: Callable[..., SourceRevisionId | None],
        attempts: Callable[..., tuple[CheckAttempt, ...]],
        outcomes: Callable[..., tuple[CheckOutcome, ...]],
        baseline: Callable[..., BaselineDecision],
        current_baseline: Callable[..., BaselineDecision],
        transition: Callable[..., ObservableTransition],
        finding: Callable[..., OperationalFinding],
        finding_occurrences: Callable[
            ..., tuple[OperationalFindingOccurrence, ...]
        ],
    ) -> None:
        self.__register_request = register_request
        self.__start_attempt = start_attempt
        self.__record_outcome = record_outcome
        self.__decide_baseline = decide_baseline
        self.__record_transition = record_transition
        self.__open_finding = open_finding
        self.__record_finding_occurrence = record_finding_occurrence
        self.__admit_proposal = admit_proposal
        self.__request = request
        self.__attempt = attempt
        self.__outcome = outcome
        self.__observed_prior_revision = observed_prior_revision
        self.__attempts = attempts
        self.__outcomes = outcomes
        self.__baseline = baseline
        self.__current_baseline = current_baseline
        self.__transition = transition
        self.__finding = finding
        self.__finding_occurrences = finding_occurrences

    def register_request(
        self,
        request: CheckRequestRequest,
        *,
        proof: AuthenticationProof,
    ) -> CheckRequest:
        return self.__register_request(request, proof)

    def start_attempt(
        self,
        request: CheckAttemptRequest,
        *,
        proof: AuthenticationProof,
    ) -> CheckAttempt:
        return self.__start_attempt(request, proof)

    def record_outcome(
        self,
        request: CheckOutcomeRequest,
        *,
        proof: AuthenticationProof,
    ) -> CheckOutcome:
        return self.__record_outcome(request, proof)

    def decide_baseline(
        self,
        request: BaselineDecisionRequest,
        *,
        proof: AuthenticationProof,
    ) -> BaselineDecision:
        return self.__decide_baseline(request, proof)

    def record_transition(
        self,
        request: ObservableTransitionRequest,
        *,
        proof: AuthenticationProof,
    ) -> ObservableTransition:
        return self.__record_transition(request, proof)

    def open_finding(
        self,
        request: OperationalFindingRequest,
        *,
        proof: AuthenticationProof,
    ) -> OperationalFinding:
        return self.__open_finding(request, proof)

    def record_finding_occurrence(
        self,
        request: OperationalFindingOccurrenceRequest,
        *,
        proof: AuthenticationProof,
    ) -> OperationalFindingOccurrence:
        return self.__record_finding_occurrence(request, proof)

    def admit_proposal(
        self,
        request: ProposalAdmissionRequest,
        *,
        proof: AuthenticationProof,
    ) -> ProposalAdmissionResult:
        return self.__admit_proposal(request, proof)

    def request(
        self,
        request_id: CheckRequestId,
        *,
        proof: AuthenticationProof,
    ) -> CheckRequest:
        return self.__request(request_id, proof)

    def attempt(
        self,
        attempt_id: CheckAttemptId,
        *,
        proof: AuthenticationProof,
    ) -> CheckAttempt:
        return self.__attempt(attempt_id, proof)

    def outcome(
        self,
        outcome_id: CheckOutcomeId,
        *,
        proof: AuthenticationProof,
    ) -> CheckOutcome:
        return self.__outcome(outcome_id, proof)

    def attempts(
        self,
        request_id: CheckRequestId,
        *,
        limit: int,
        proof: AuthenticationProof,
    ) -> tuple[CheckAttempt, ...]:
        return self.__attempts(request_id, limit, proof)

    def observed_prior_revision(
        self, item_id: SourceItemId, *, request_id: CheckRequestId,
        outcome_id: CheckOutcomeId, completed_at: UtcTimestamp,
        proof: AuthenticationProof,
    ) -> SourceRevisionId | None:
        return self.__observed_prior_revision(
            item_id, request_id, outcome_id, completed_at, proof,
        )

    def outcomes(
        self,
        request_id: CheckRequestId,
        *,
        limit: int,
        proof: AuthenticationProof,
    ) -> tuple[CheckOutcome, ...]:
        return self.__outcomes(request_id, limit, proof)

    def baseline(
        self,
        decision_id: BaselineDecisionId,
        *,
        proof: AuthenticationProof,
    ) -> BaselineDecision:
        return self.__baseline(decision_id, proof)

    def current_baseline(
        self,
        definition_id: SourceDefinitionId,
        *,
        proof: AuthenticationProof,
    ) -> BaselineDecision:
        return self.__current_baseline(definition_id, proof)

    def transition(
        self,
        transition_id: ObservableTransitionId,
        *,
        proof: AuthenticationProof,
    ) -> ObservableTransition:
        return self.__transition(transition_id, proof)

    def finding(
        self,
        finding_id: OperationalFindingId,
        *,
        proof: AuthenticationProof,
    ) -> OperationalFinding:
        return self.__finding(finding_id, proof)

    def finding_occurrences(
        self,
        finding_id: OperationalFindingId,
        *,
        limit: int,
        proof: AuthenticationProof,
    ) -> tuple[OperationalFindingOccurrence, ...]:
        return self.__finding_occurrences(finding_id, limit, proof)


__all__ = ["GovernedChecks"]
