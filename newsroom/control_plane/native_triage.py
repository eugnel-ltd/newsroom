"""Advance committed discovery Leads through the native triage authorities."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from newsroom.authority.auth import AuthenticationProof
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.discovery import LeadDispositionDecision, NewsLead
from newsroom.increment6.dispositions import CurrentCandidateCitation, ProposalDisposition
from newsroom.increment6.autonomous_worker import (
    AUTONOMOUS_WORKER_VERSION,
    autonomous_worker_input_digest,
    build_autonomous_proposal,
)
from newsroom.increment6.candidates import (
    CandidateAdmission,
    CandidateAdmissionOutcome,
    CandidateAdmissionRequest,
    CandidateGoverningState,
    CandidateGoverningStateStatus,
    StoryCandidateVersion,
    evaluate_candidate_admission,
)
from newsroom.increment6.collision import (
    CurrentCollisionEligibilityDecision,
    CurrentCollisionEligibilityRequest,
)
from newsroom.increment6.execution import (
    ExecutionBatch,
    WorkerAttempt,
    WorkItemLease,
)
from newsroom.increment6.hypotheses import EventHypothesisVersion
from newsroom.increment6.outcomes import (
    CanonicalNextAction,
    CanonicalOutcome,
    DecisionTerminality,
    NextAction,
    OutcomeSelection,
    PriorityLane,
    PrioritySelection,
    ReasonCode,
    ReasonBasisClass,
    ReasonReference,
    StructuredReason,
)
from newsroom.increment6.proposals import (
    HypothesisRelationship,
    ProposalRoute,
    TriageProposal,
    WorkerKind,
)
from newsroom.increment6.relationships import (
    ComparatorEvidence,
    ComparatorSetManifest,
    HypothesisVersionBinding,
    RelationshipAssessment,
    assess_relationships,
)
from newsroom.increment6.scheduling import ReservedCapacityDecision
from newsroom.increment6.scheduling import (
    CapacityAllocationDisposition,
    CapacityPathState,
    CapacityPopulationItem,
    CapacitySnapshot,
    CapacityWorkState,
    DeadlineBoundary,
    DeadlineKind,
    LaneTimingRule,
    ReservedCapacityDisposition,
    ReservedCapacityPolicy,
    SchedulingEligibility,
    UrgencyDeadlineInput,
    UrgencyDeadlinePolicy,
    allocate_reserved_capacity,
    calculate_urgency_deadline,
)
from newsroom.increment6.work_items import (
    DecisionLeadBinding,
    RetrievalInputBinding,
    TriageWorkItem,
    TriageWorkItemVersion,
)


class NativeTriageError(RuntimeError):
    """Native triage inputs or retained authority state differ."""


_URGENCY_LANES = {
    "URGENT": PriorityLane.URGENT,
    "TIME_SENSITIVE": PriorityLane.TIME_SENSITIVE,
    "PLANNED": PriorityLane.PLANNED_WINDOW,
    "ROUTINE": PriorityLane.ROUTINE,
}


@dataclass(frozen=True, slots=True)
class NativeTriageWork:
    item: TriageWorkItem
    version: TriageWorkItemVersion
    leads: tuple[NewsLead, ...]


@dataclass(frozen=True, slots=True)
class NativeTriageResult:
    work: NativeTriageWork
    state: str
    batch: ExecutionBatch | None
    attempt: WorkerAttempt | None
    lease: WorkItemLease | None
    proposal: TriageProposal | None
    dispositions: tuple[ProposalDisposition, ...]
    hypothesis: EventHypothesisVersion | None
    relationship: RelationshipAssessment | None
    admission: CandidateAdmission | None
    candidate: StoryCandidateVersion | None

    @property
    def candidate_ready(self) -> bool:
        return self.hypothesis is not None and self.relationship is not None


@dataclass(frozen=True, slots=True)
class NativeSchedulePlan:
    state: str
    decision: ReservedCapacityDecision | None
    reason: str | None


def _scheduling_policy() -> UrgencyDeadlinePolicy:
    limits = {
        PriorityLane.CONTAINMENT: (60, 120),
        PriorityLane.URGENT: (120, 300),
        PriorityLane.TIME_SENSITIVE: (900, 1_800),
        PriorityLane.PLANNED_WINDOW: (1_800, 3_600),
        PriorityLane.ROUTINE: (3_600, 7_200),
        PriorityLane.OPTIONAL_EVALUATION: (7_200, 14_400),
    }
    return UrgencyDeadlinePolicy(
        "hermes-native-immediate-scheduling",
        "v1",
        "UTC",
        "WORK_ITEM_VERSION_ID_ASC",
        tuple(
            LaneTimingRule(lane, *limits[lane], False) for lane in PriorityLane
        ),
    )


def plan_native_schedule(work: NativeTriageWork) -> NativeSchedulePlan:
    """Allocate one just-retained Work Item with replay-stable native policy."""

    if type(work) is not NativeTriageWork:
        raise NativeTriageError("native triage work must be exact typed")
    priority = PrioritySelection.from_canonical_bytes(
        work.version.priority.selection_bytes
    )
    enqueued_at = min(
        (lead.recorded_at.value for lead in work.leads)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    hard_deadlines = tuple(
        lead.request.urgency.hard_deadline
        for lead in work.leads
        if lead.request.urgency.hard_deadline is not None
    )
    deadline = None
    if hard_deadlines:
        due = min(hard_deadlines, key=lambda value: value.value).value.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        deadline = DeadlineBoundary(
            DeadlineKind.HARD_ACTION, due, "UTC", due[:-1], 0, 0
        )
    policy = _scheduling_policy()
    observation = calculate_urgency_deadline(
        policy=policy,
        item=UrgencyDeadlineInput(
            work.version.work_item_id,
            work.version.version_id,
            work.version.canonical_digest,
            priority,
            priority.lane,
            enqueued_at,
            enqueued_at,
            deadline,
            SchedulingEligibility.CURRENT_ELIGIBLE,
            max(0, min(3, 4 - priority.lane.ordinal)),
            max(0, min(3, 4 - priority.lane.ordinal)),
            True,
        ),
    )
    if observation.revalidation_required:
        return NativeSchedulePlan(
            "SCHEDULING_HOLD", None, "CURRENTNESS_REVALIDATION_REQUIRED"
        )
    snapshot = CapacitySnapshot(
        enqueued_at,
        policy,
        (CapacityPopulationItem(observation, CapacityWorkState.PENDING, None),),
        CapacityPathState.AVAILABLE,
    )
    decision = allocate_reserved_capacity(
        policy=ReservedCapacityPolicy(
            "hermes-native-serial-capacity",
            "v1",
            2,
            1,
            1,
            ReservedCapacityDisposition.URGENT_VISIBLE_OPERATIONAL_HOLD,
        ),
        snapshot=snapshot,
    )
    if (
        len(decision.allocations) != 1
        or decision.allocations[0].disposition
        is not CapacityAllocationDisposition.GRANTED
    ):
        return NativeSchedulePlan("SCHEDULING_HOLD", None, "CAPACITY_DEFERRED")
    return NativeSchedulePlan("SCHEDULED", decision, None)


def build_native_triage_work(
    *,
    admitted_leads: tuple[tuple[NewsLead, LeadDispositionDecision], ...],
    retrieval: RetrievalInputBinding,
) -> NativeTriageWork:
    """Build the deterministic native Work Item from committed Lead authority."""

    if type(admitted_leads) is not tuple or not admitted_leads:
        raise NativeTriageError("admitted Leads must be a non-empty exact tuple")
    if type(retrieval) is not RetrievalInputBinding:
        raise NativeTriageError("retrieval binding must be exact typed")
    if any(
        type(pair) is not tuple
        or len(pair) != 2
        or type(pair[0]) is not NewsLead
        or type(pair[1]) is not LeadDispositionDecision
        for pair in admitted_leads
    ):
        raise NativeTriageError("admitted Lead authority must be exact committed records")
    pairs = sorted(admitted_leads, key=lambda pair: str(pair[0].request.lead_id))
    bindings = tuple(DecisionLeadBinding.from_authority(*pair) for pair in pairs)
    item = TriageWorkItem.create(bindings)
    version_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{item.work_item_id}|1"))
    lane = min(
        (_URGENCY_LANES[pair[0].request.urgency.route.value] for pair in pairs),
        key=lambda value: value.ordinal,
    )
    references = tuple(
        sorted(
            (
                ReasonReference("DISCOVERY_LEAD", binding.lead_id, binding.lead_digest)
                for binding in bindings
            ),
            key=lambda value: (value.reference_type, value.identifier, value.digest or ""),
        )
    )
    priority = PrioritySelection(item.work_item_id, version_id, lane, references)
    version = TriageWorkItemVersion.create(
        work_item_id=item.work_item_id,
        ordinal=1,
        previous_version_id=None,
        decision_leads=bindings,
        context_leads=(),
        retrieval=retrieval,
        priority=priority,
    )
    return NativeTriageWork(item, version, tuple(pair[0] for pair in pairs))


def _selection(proposal: TriageProposal, lead_id: str) -> OutcomeSelection:
    recommendation = next(
        item for item in proposal.recommendations if item.decision_lead_id == lead_id
    )
    route_digest = digest_bytes(canonical_json_bytes(recommendation.canonical_value()))
    reference = ReasonReference("TRIAGE_PROPOSAL", proposal.proposal_id, route_digest)
    if recommendation.route is ProposalRoute.NEW_EVENT_CANDIDATE:
        return OutcomeSelection(
            outcome=CanonicalOutcome.LEAD_ADMIT_NEW_CANDIDATE,
            terminality=DecisionTerminality.TERMINAL_EXACT_VERSION,
            primary_reason=StructuredReason(
                ReasonCode.REL_NO_ADEQUATE_PRIOR_MATCH,
                ReasonBasisClass.DETERMINISTIC_POLICY,
                (reference,),
                "The complete governed retrieval found no adequate prior match.",
            ),
            supporting_reasons=(),
            next_action=NextAction(
                CanonicalNextAction.HANDOFF_FOR_EVALUATION.kind,
                CanonicalNextAction.HANDOFF_FOR_EVALUATION,
                None,
            ),
        )
    if recommendation.route is ProposalRoute.DEVELOPMENT_CANDIDATE:
        return OutcomeSelection(
            outcome=CanonicalOutcome.LEAD_ADMIT_DEVELOPMENT_CANDIDATE,
            terminality=DecisionTerminality.TERMINAL_EXACT_VERSION,
            primary_reason=StructuredReason(
                ReasonCode.REL_DEVELOPMENT,
                ReasonBasisClass.DETERMINISTIC_POLICY,
                (reference,),
                "The direct retained source revision develops the current Candidate Hypothesis.",
            ),
            supporting_reasons=(),
            next_action=NextAction(
                CanonicalNextAction.HANDOFF_FOR_EVALUATION.kind,
                CanonicalNextAction.HANDOFF_FOR_EVALUATION,
                None,
            ),
        )
    if recommendation.route is ProposalRoute.ASSOCIATE_WITHOUT_CANDIDATE:
        return OutcomeSelection(
            outcome=CanonicalOutcome.LEAD_ASSOCIATE_WITHOUT_CANDIDATE,
            terminality=DecisionTerminality.TERMINAL_EXACT_VERSION,
            primary_reason=StructuredReason(
                ReasonCode.REL_SAME_STATE,
                ReasonBasisClass.DETERMINISTIC_POLICY,
                (reference,),
                "The direct retained source revision preserves the current Candidate state.",
            ),
            supporting_reasons=(),
            next_action=NextAction(
                CanonicalNextAction.CLOSE_DECISION.kind,
                CanonicalNextAction.CLOSE_DECISION,
                None,
            ),
        )
    action = recommendation.operational_action
    if action is None:
        raise NativeTriageError("operational hold lacks its exact action")
    action_code = {
        "RETRY_RETRIEVAL": CanonicalNextAction.RETRY_SAME_REQUEST,
        "WAIT_FOR_DEPENDENCY": CanonicalNextAction.WAIT_FOR_DEPENDENCY,
    }.get(action.action_kind)
    if action_code is None:
        raise NativeTriageError("operational hold action is unsupported")
    return OutcomeSelection(
        outcome=CanonicalOutcome.LEAD_OPERATIONAL_HOLD,
        terminality=(
            DecisionTerminality.RETRYABLE_SAME_REQUEST
            if action_code is CanonicalNextAction.RETRY_SAME_REQUEST
            else DecisionTerminality.PENDING_CONDITION
        ),
        primary_reason=StructuredReason(
            ReasonCode.OPS_RETRIEVAL,
            ReasonBasisClass.OPERATIONAL_ASSESSMENT,
            (reference,),
            "The governed retrieval does not yet support a deterministic relationship.",
        ),
        supporting_reasons=(),
        next_action=NextAction(action_code.kind, action_code, route_digest),
    )


def advance_native_triage(
    system: object,
    *,
    work: NativeTriageWork,
    scheduling_decision: ReservedCapacityDecision | None,
    proof: AuthenticationProof,
    collision_request: CurrentCollisionEligibilityRequest | None = None,
    collision_decision: CurrentCollisionEligibilityDecision | None = None,
    candidate_request: CandidateAdmissionRequest | None = None,
    current_candidate: CurrentCandidateCitation | None = None,
    revision_relationship: HypothesisRelationship | None = None,
    expected_target_version: EventHypothesisVersion | None = None,
    current_candidate_version: StoryCandidateVersion | None = None,
) -> NativeTriageResult:
    """Persist one Work Item and advance every currently available native stage."""

    if type(work) is not NativeTriageWork:
        raise NativeTriageError("native triage work must be exact typed")
    if type(proof) is not AuthenticationProof:
        raise NativeTriageError("authentication proof must be exact typed")
    candidate_inputs = (collision_request, collision_decision, candidate_request)
    if any(value is not None for value in candidate_inputs) and not (
        type(collision_request) is CurrentCollisionEligibilityRequest
        and type(collision_decision) is CurrentCollisionEligibilityDecision
        and type(candidate_request) is CandidateAdmissionRequest
    ):
        raise NativeTriageError("Candidate admission requires all exact typed inputs")
    if (current_candidate is None) != (revision_relationship is None):
        raise NativeTriageError("revision relationship requires its Candidate citation")
    if current_candidate is not None and type(current_candidate) is not CurrentCandidateCitation:
        raise NativeTriageError("current Candidate citation must be exact typed")
    if revision_relationship not in {
        None,
        HypothesisRelationship.DEVELOPMENT_OF,
        HypothesisRelationship.SAME_STATE,
    }:
        raise NativeTriageError("revision relationship is unsupported")
    if (expected_target_version is None) != (current_candidate is None):
        raise NativeTriageError("revision relationship requires the exact target head")
    version = system.work_items.create_or_replay(work.item, work.version)
    if version != work.version:
        raise NativeTriageError("retained Work Item Version differs")
    if not version.retrieval.usable:
        return NativeTriageResult(
            work,
            "RETRIEVAL_PENDING",
            None,
            None,
            None,
            None,
            (),
            None,
            None,
            None,
            None,
        )
    if type(scheduling_decision) is not ReservedCapacityDecision:
        raise NativeTriageError("usable Work Item requires an exact scheduling decision")
    batch = ExecutionBatch.create(
        scheduling_decision=scheduling_decision,
        work_item_versions=(version,),
    )
    batch = system.executions.register_batch(batch, proof=proof)
    attempt = WorkerAttempt.create(
        member=batch.members[0],
        ordinal=1,
        worker_kind=WorkerKind.AUTONOMOUS_DETERMINISTIC,
        worker_version=AUTONOMOUS_WORKER_VERSION,
        input_digest=autonomous_worker_input_digest(
            version,
            work.leads,
            current_candidate=current_candidate,
            revision_relationship=revision_relationship,
        ),
    )
    attempt = system.executions.register_attempt(batch.batch_id, attempt, proof=proof)
    lease = system.executions.claim(attempt.attempt_id, proof=proof)
    proposal = build_autonomous_proposal(
        work_item_version=version,
        attempt=attempt,
        decision_leads=work.leads,
        current_candidate=current_candidate,
        revision_relationship=revision_relationship,
    )
    lease = system.executions.complete(
        lease.lease_id, digest_bytes(proposal.canonical_bytes), proof=proof
    )
    selections = {
        lead_id: _selection(proposal, lead_id) for lead_id in proposal.decision_lead_ids
    }
    dispositions = tuple(
        system.dispositions.persist(
            proposal.canonical_bytes, selections, proof=proof
        )
    )
    if any(item.route is ProposalRoute.OPERATIONAL_HOLD for item in dispositions):
        return NativeTriageResult(
            work,
            "OPERATIONAL_HOLD",
            batch,
            attempt,
            lease,
            proposal,
            dispositions,
            None,
            None,
            None,
            None,
        )
    hypothesis = system.hypotheses.retain(
        proposal.canonical_bytes,
        dispositions,
        proof=proof,
        expected_target_version=expected_target_version,
    )
    subject = HypothesisVersionBinding.from_version(hypothesis)
    if expected_target_version is None:
        comparators = ComparatorSetManifest.complete(())
        evidence = ()
    else:
        comparator = HypothesisVersionBinding.from_version(expected_target_version)
        comparators = ComparatorSetManifest.complete((comparator,))
        evidence = (ComparatorEvidence(
            subject,
            comparator,
            100,
            0,
            100 if revision_relationship is HypothesisRelationship.DEVELOPMENT_OF else 0,
            100 if revision_relationship is HypothesisRelationship.SAME_STATE else 0,
            0,
        ),)
    assessment = assess_relationships(subject, comparators, evidence)
    relationship = system.relationships.retain(
        assessment.canonical_bytes,
        tuple(item.canonical_bytes for item in evidence),
        proof=proof,
    )
    state = (
        "SAME_STATE_ASSOCIATED"
        if revision_relationship is HypothesisRelationship.SAME_STATE
        else "CANDIDATE_READY"
    )
    result = NativeTriageResult(
        work,
        state,
        batch,
        attempt,
        lease,
        proposal,
        dispositions,
        hypothesis,
        relationship,
        None,
        None,
    )
    if all(value is None for value in candidate_inputs):
        return result
    return _admit_native_triage_candidate(
        system,
        triage=result,
        collision_request=collision_request,
        collision_decision=collision_decision,
        candidate_request=candidate_request,
        current_candidate_version=current_candidate_version,
        proof=proof,
    )


def _admit_native_triage_candidate(
    system: object,
    *,
    triage: NativeTriageResult,
    collision_request: CurrentCollisionEligibilityRequest,
    collision_decision: CurrentCollisionEligibilityDecision,
    candidate_request: CandidateAdmissionRequest,
    current_candidate_version: StoryCandidateVersion | None,
    proof: AuthenticationProof,
) -> NativeTriageResult:
    """Continue one exact prepared native result through Candidate admission."""

    if (
        type(triage) is not NativeTriageResult
        or triage.state not in {"CANDIDATE_READY", "SAME_STATE_ASSOCIATED"}
        or triage.hypothesis is None
        or triage.relationship is None
        or type(collision_request) is not CurrentCollisionEligibilityRequest
        or type(collision_decision) is not CurrentCollisionEligibilityDecision
        or type(candidate_request) is not CandidateAdmissionRequest
        or type(current_candidate_version) not in (type(None), StoryCandidateVersion)
        or type(proof) is not AuthenticationProof
    ):
        raise NativeTriageError("Candidate continuation requires exact typed inputs")
    manifest = system.build_candidate_manifest(
        triage.hypothesis.version_id,
        triage.relationship.canonical_digest,
        collision_decision,
        proof=proof,
    )
    admission = evaluate_candidate_admission(
        request=candidate_request,
        manifest=manifest,
        collision=collision_decision,
        current_version=current_candidate_version,
        governing_state=CandidateGoverningState(
            CandidateGoverningStateStatus.COMPLETE,
            manifest.governing_state_binding,
        ),
    )
    if admission.outcome is not CandidateAdmissionOutcome.ADMISSIBLE:
        return NativeTriageResult(
            triage.work,
            "CANDIDATE_HOLD",
            triage.batch,
            triage.attempt,
            triage.lease,
            triage.proposal,
            triage.dispositions,
            triage.hypothesis,
            triage.relationship,
            admission,
            None,
        )
    candidate = system.candidates.admit(
        admission.canonical_bytes,
        collision_request=collision_request,
        proof=proof,
    )
    return NativeTriageResult(
        triage.work,
        "CANDIDATE_ADMITTED",
        triage.batch,
        triage.attempt,
        triage.lease,
        triage.proposal,
        triage.dispositions,
        triage.hypothesis,
        triage.relationship,
        admission,
        candidate,
    )


__all__ = [
    "NativeTriageError",
    "NativeTriageResult",
    "NativeTriageWork",
    "NativeSchedulePlan",
    "advance_native_triage",
    "build_native_triage_work",
    "plan_native_schedule",
]
