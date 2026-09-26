"""Bounded autonomous advancement of current native Discovery Leads."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
from typing import ContextManager, Protocol

from newsroom.authority import AuthenticationProof, UtcTimestamp
from newsroom.authority.canonical import canonical_json_bytes, validate_sha256_digest
from newsroom.discovery import NewsLead, NewsLeadId
from newsroom.sources import SourceRevisionId
from newsroom.discovery.read_models import DiscoveryCurrentStatus
from newsroom.increment6.candidates import CandidateAdmissionRequest
from newsroom.increment6.candidates import StoryCandidateVersion
from newsroom.increment6.dispositions import CurrentCandidateCitation
from newsroom.increment6.hypotheses import EventHypothesisVersion
from newsroom.increment6.proposals import HypothesisRelationship
from newsroom.increment6.collision import (
    CandidateUseOperation,
    CurrentCollisionEligibilityBlocked,
    CurrentCollisionEligibilityRequest,
)
from newsroom.increment6.work_items import RetrievalInputBinding
from newsroom.increment5.native_retrieval import NativeRetrievalHold

from .corpus import CorpusIngestUnit
from .native_discovery import NativeDiscovery
from .native_collision import NativeCollisionHold
from .native_triage import (
    NativeTriageResult,
    _admit_native_triage_candidate,
    advance_native_triage,
    build_native_triage_work,
    plan_native_schedule,
)


class NativeRetrievalPort(Protocol):
    def retrieve(
        self, lead: NewsLead, *, proof: AuthenticationProof
    ) -> RetrievalInputBinding: ...


class NativeCollisionRequestPort(Protocol):
    def current_candidate_citation(
        self,
        lead: NewsLead,
        retrieval: RetrievalInputBinding,
        *,
        proof: AuthenticationProof,
    ) -> CurrentCandidateCitation | None: ...

    def retain_associated_candidate_citation(
        self,
        citation: CurrentCandidateCitation,
        candidate: StoryCandidateVersion,
        hypothesis: EventHypothesisVersion,
        *,
        lead: NewsLead,
        retrieval: RetrievalInputBinding,
        proof: AuthenticationProof,
    ) -> CurrentCandidateCitation: ...

    def request(
        self,
        triage: NativeTriageResult,
        retrieval: RetrievalInputBinding,
        *,
        proof: AuthenticationProof,
    ) -> CurrentCollisionEligibilityRequest | None: ...


@dataclass(frozen=True, slots=True)
class NativeCycleOutcome:
    revision_id: str
    state: str
    triage: NativeTriageResult | None
    reason: str | None


def _revision_successor(
    system: object,
    lead: NewsLead,
    citation: CurrentCandidateCitation,
    *,
    proof: AuthenticationProof,
    current_producers=None,
):
    current, target = (
        system.candidates._exact_current_producers(
            citation.candidate_id, proof=proof
        )
        if current_producers is None
        else current_producers
    )
    if (
        current.version_id != citation.candidate_version_id
        or current.canonical_digest != citation.candidate_version_digest
        or current.governing_manifest.hypothesis_id != citation.hypothesis_id
    ):
        raise NativeCollisionHold("CURRENT_CANDIDATE_CITATION_STALE")
    if (
        target.version_id != citation.hypothesis_version_id
        or target.canonical_digest != citation.hypothesis_version_digest
    ):
        raise NativeCollisionHold("CURRENT_HYPOTHESIS_CITATION_STALE")
    bindings = target.source_bindings
    if len(bindings) != 1:
        raise NativeCollisionHold("SOURCE_REVISION_RELATIONSHIP_AMBIGUOUS")
    prior_lead = system.discovery.lead(
        NewsLeadId.parse(bindings[0].decision_lead_id), proof=proof
    )
    previous = system.sources.revision(
        SourceRevisionId.parse(str(prior_lead.request.revision_id)), proof=proof
    ).request
    proposed = system.sources.revision(lead.request.revision_id, proof=proof).request
    if (
        str(lead.request.definition_id) != citation.source_definition_id
        or str(lead.request.item_id) != citation.source_item_id
        or prior_lead.request.definition_id != lead.request.definition_id
        or prior_lead.request.item_id != lead.request.item_id
        or proposed.item_id != previous.item_id
        or proposed.definition_version_id != previous.definition_version_id
    ):
        raise NativeCollisionHold("SOURCE_REVISION_RELATIONSHIP_AMBIGUOUS")
    cursor = proposed
    seen = set()
    same_state = True
    while cursor.revision_id != previous.revision_id:
        if (
            cursor.revision_id in seen
            or cursor.item_id != previous.item_id
            or cursor.definition_version_id != previous.definition_version_id
            or cursor.prior_revision_id is None
        ):
            raise NativeCollisionHold("SOURCE_REVISION_RELATIONSHIP_AMBIGUOUS")
        same_state = (
            same_state
            and cursor.permitted_state_digest == previous.permitted_state_digest
        )
        seen.add(cursor.revision_id)
        cursor = system.sources.revision(cursor.prior_revision_id, proof=proof).request
    if cursor != previous:
        raise NativeCollisionHold("SOURCE_REVISION_RELATIONSHIP_AMBIGUOUS")
    relationship = (
        HypothesisRelationship.SAME_STATE
        if same_state
        else HypothesisRelationship.DEVELOPMENT_OF
    )
    return current, target, relationship


def _uuid4_for(value: object) -> str:
    raw = bytearray(hashlib.sha256(canonical_json_bytes(value)).digest()[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(raw)))


def advance_native_cycle(
    system: object,
    statuses: tuple[DiscoveryCurrentStatus, ...],
    *,
    retrieval: NativeRetrievalPort,
    collision_requests: NativeCollisionRequestPort,
    actor_identity_digest: str,
    proof: AuthenticationProof,
    owner_stop_check: Callable[[], None],
    owner_stop_fence: Callable[[], ContextManager[None]],
) -> tuple[NativeCycleOutcome, ...]:
    """Advance each current Lead independently; retain visible per-revision holds."""

    validate_sha256_digest(actor_identity_digest, field="actor_identity_digest")
    if not callable(owner_stop_check) or not callable(owner_stop_fence):
        raise TypeError("native cycle owner-stop boundaries must be callable")
    outcomes: list[NativeCycleOutcome] = []
    for status in statuses:
        owner_stop_check()
        if type(status) is not DiscoveryCurrentStatus:
            raise TypeError("native cycle statuses must be exact typed")
        lead, disposition = status.lead, status.current_disposition
        revision_id = (
            str(status.signal.request.revision_id)
            if lead is None
            else str(lead.request.revision_id)
        )
        if lead is None or disposition is None:
            outcomes.append(
                NativeCycleOutcome(
                    revision_id, "DISCOVERY_HOLD", None, status.phase.value
                )
            )
            continue
        try:
            binding = retrieval.retrieve(lead, proof=proof)
        except NativeRetrievalHold as exc:
            outcomes.append(
                NativeCycleOutcome(revision_id, "RETRIEVAL_HOLD", None, exc.reason)
            )
            continue
        if type(binding) is not RetrievalInputBinding:
            raise TypeError("native retrieval returned a non-authoritative binding")
        work = build_native_triage_work(
            admitted_leads=((lead, disposition),), retrieval=binding
        )
        schedule = plan_native_schedule(work)
        if schedule.decision is None:
            outcomes.append(
                NativeCycleOutcome(
                    revision_id, schedule.state, None, schedule.reason
                )
            )
            continue
        try:
            citation_reader = getattr(
                collision_requests, "current_candidate_citation", None
            )
            citation = (
                None
                if citation_reader is None
                else citation_reader(lead, binding, proof=proof)
            )
            exact_replay = False
            if citation is None:
                current_candidate = target_hypothesis = relationship = None
            else:
                cited_current, cited_target = (
                    system.candidates._exact_associated_current_producers(
                        citation.candidate_id, proof=proof
                    )
                )
                retain_association = getattr(
                    collision_requests,
                    "retain_associated_candidate_citation",
                    None,
                )
                if retain_association is not None:
                    citation = retain_association(
                        citation,
                        cited_current,
                        cited_target,
                        lead=lead,
                        retrieval=binding,
                        proof=proof,
                    )
                elif (
                    citation.hypothesis_version_id != cited_target.version_id
                    or citation.hypothesis_version_digest
                    != cited_target.canonical_digest
                ):
                    raise NativeCollisionHold(
                        "CURRENT_CANDIDATE_ASSOCIATION_UNAVAILABLE"
                    )
                bindings = cited_current.governing_manifest.lead_signal_bindings
                exact_replay = any(
                    item.lead_id == str(lead.request.lead_id)
                    and item.lead_digest == lead.canonical_digest
                    for item in bindings
                )
                associated_replay = any(
                    item.decision_lead_id == str(lead.request.lead_id)
                    and item.decision_lead_digest == lead.canonical_digest
                    for item in cited_target.source_bindings
                )
                if exact_replay:
                    current_candidate = cited_current
                    citation = target_hypothesis = relationship = None
                elif associated_replay:
                    outcomes.append(
                        NativeCycleOutcome(
                            revision_id,
                            "SAME_STATE_ASSOCIATED",
                            None,
                            None,
                        )
                    )
                    continue
                else:
                    current_candidate, target_hypothesis, relationship = _revision_successor(
                        system,
                        lead,
                        citation,
                        proof=proof,
                        current_producers=(cited_current, cited_target),
                    )
        except NativeCollisionHold as exc:
            outcomes.append(
                NativeCycleOutcome(revision_id, "COLLISION_HOLD", None, exc.reason)
            )
            continue
        triage = advance_native_triage(
            system,
            work=work,
            scheduling_decision=schedule.decision,
            proof=proof,
            current_candidate=citation,
            revision_relationship=relationship,
            expected_target_version=target_hypothesis,
        )
        if triage.state == "SAME_STATE_ASSOCIATED":
            outcomes.append(NativeCycleOutcome(revision_id, triage.state, triage, None))
            continue
        if triage.state != "CANDIDATE_READY":
            outcomes.append(
                NativeCycleOutcome(revision_id, triage.state, triage, None)
            )
            continue
        try:
            collision_request = collision_requests.request(
                triage, binding, proof=proof
            )
        except NativeCollisionHold as exc:
            outcomes.append(
                NativeCycleOutcome(revision_id, "COLLISION_HOLD", triage, exc.reason)
            )
            continue
        if collision_request is None:
            outcomes.append(
                NativeCycleOutcome(
                    revision_id,
                    "COLLISION_HOLD",
                    triage,
                    "CURRENT_COLLISION_AUTHORITY_UNAVAILABLE",
                )
            )
            continue
        collision_binding = collision_request.binding
        if (
            collision_binding.subject_id != triage.hypothesis.hypothesis_id
            or collision_binding.subject_version_id != triage.hypothesis.version_id
            or collision_binding.subject_version_digest
            != triage.hypothesis.canonical_digest
            or (
                collision_binding.operation
                is CandidateUseOperation.ADMIT_NEW_CANDIDATE
            )
            != (collision_binding.expected_candidate_id is None)
        ):
            raise ValueError("current collision request differs from native Hypothesis")
        try:
            collision = system.collision.enforce(
                request=collision_request, effect=lambda decision: decision
            )
        except CurrentCollisionEligibilityBlocked as exc:
            outcomes.append(
                NativeCycleOutcome(
                    revision_id,
                    "COLLISION_HOLD",
                    triage,
                    f"{exc.decision.outcome.value}:{exc.decision.reason.value}",
                )
            )
            continue
        except NativeCollisionHold as exc:
            outcomes.append(
                NativeCycleOutcome(revision_id, "COLLISION_HOLD", triage, exc.reason)
            )
            continue
        if collision_binding.operation is CandidateUseOperation.USE_CURRENT_CANDIDATE:
            candidate_id = collision_binding.expected_candidate_id
            assert candidate_id is not None
            confirmed = system.candidates._exact_current_candidate(candidate_id)
            if current_candidate is None or confirmed != current_candidate:
                outcomes.append(NativeCycleOutcome(
                    revision_id, "COLLISION_HOLD", triage,
                    "CURRENT_SLOT_OCCUPIED_BY_OTHER_HYPOTHESIS",
                ))
                continue
            current = confirmed
            if exact_replay:
                if (
                    current.governing_manifest.hypothesis_id
                    != triage.hypothesis.hypothesis_id
                    or current.governing_manifest.hypothesis_version_id
                    != triage.hypothesis.version_id
                ):
                    outcomes.append(NativeCycleOutcome(
                        revision_id, "COLLISION_HOLD", triage,
                        "CURRENT_SLOT_OCCUPIED_BY_OTHER_HYPOTHESIS",
                    ))
                    continue
                outcomes.append(NativeCycleOutcome(
                    revision_id, "CANDIDATE_ADMITTED",
                    replace(triage, state="CANDIDATE_ADMITTED", candidate=current),
                    None,
                ))
                continue
            if citation is None or (
                current is None
                or current.version_id != citation.candidate_version_id
                or current.canonical_digest != citation.candidate_version_digest
                or current.governing_manifest.hypothesis_id != triage.hypothesis.hypothesis_id
                or citation.hypothesis_version_id
                != triage.hypothesis.previous_version_id
                or citation.hypothesis_version_digest
                != triage.hypothesis.previous_version_digest
            ):
                outcomes.append(NativeCycleOutcome(
                    revision_id, "COLLISION_HOLD", triage,
                    "CURRENT_SLOT_OCCUPIED_BY_OTHER_HYPOTHESIS",
                ))
                continue
        else:
            current = None
        owner_stop_check()
        manifest = system.build_candidate_manifest(
            triage.hypothesis.version_id,
            triage.relationship.canonical_digest,
            collision,
            proof=proof,
        )
        candidate_request = CandidateAdmissionRequest(
            _uuid4_for(
                {
                    "hypothesis_version_id": triage.hypothesis.version_id,
                    "relationship_digest": triage.relationship.canonical_digest,
                    "collision_request_digest": collision_request.request_digest,
                }
            ),
            actor_identity_digest,
            f"native-candidate:{triage.hypothesis.version_id}",
            None if current is None else current.version_id,
            None if current is None else current.canonical_digest,
            0 if current is None else current.ordinal,
            manifest.semantic_scope_digest,
            collision_request.request_digest,
            manifest.governing_state_binding.canonical_digest,
            None,
        )
        with owner_stop_fence():
            admitted = _admit_native_triage_candidate(
                system,
                triage=triage,
                collision_request=collision_request,
                collision_decision=collision,
                candidate_request=candidate_request,
                current_candidate_version=current,
                proof=proof,
            )
        outcomes.append(
            NativeCycleOutcome(revision_id, admitted.state, admitted, None)
        )
    return tuple(outcomes)


def advance_native_revisions(
    discovery: NativeDiscovery,
    system: object,
    units: tuple[CorpusIngestUnit, ...],
    *,
    now: UtcTimestamp,
    retrieval: NativeRetrievalPort,
    collision_requests: NativeCollisionRequestPort,
    actor_identity_digest: str,
    proof: AuthenticationProof,
    owner_stop_check: Callable[[], None],
    owner_stop_fence: Callable[[], ContextManager[None]],
) -> tuple[NativeCycleOutcome, ...]:
    """Deliver and advance revisions independently through the native authorities."""

    if type(discovery) is not NativeDiscovery or type(now) is not UtcTimestamp:
        raise TypeError("native revision cycle requires exact runtime inputs")
    if type(units) is not tuple or any(
        type(unit) is not CorpusIngestUnit for unit in units
    ):
        raise TypeError("native revision cycle units must be exact typed")
    outcomes: list[NativeCycleOutcome] = []
    for unit in units:
        revision_id = (
            unit.revision_digest
            if unit.authority is None
            else unit.authority.revision_id
        )
        try:
            delivered = discovery.deliver(unit, now=now, proof=proof)
            status = discovery.admit_lead(delivered, now=now, proof=proof)
        except ValueError as exc:
            outcomes.append(
                NativeCycleOutcome(
                    revision_id, "DISCOVERY_HOLD", None, str(exc)
                )
            )
            continue
        outcomes.extend(
            advance_native_cycle(
                system,
                (status,),
                retrieval=retrieval,
                collision_requests=collision_requests,
                actor_identity_digest=actor_identity_digest,
                proof=proof,
                owner_stop_check=owner_stop_check,
                owner_stop_fence=owner_stop_fence,
            )
        )
    return tuple(outcomes)


__all__ = [
    "NativeCollisionRequestPort",
    "NativeCycleOutcome",
    "NativeRetrievalPort",
    "advance_native_cycle",
    "advance_native_revisions",
]
