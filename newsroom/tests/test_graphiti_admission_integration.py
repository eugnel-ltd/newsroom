from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import sqlite3
from types import SimpleNamespace

import pytest

from newsroom.authority.auth import AuthenticationProof
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.authority.types import EventId, UtcTimestamp
from newsroom.control_plane.graphiti_admission import (
    GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION,
    GraphitiAdmissionConsumer,
    GraphitiAdmissionConsumerError,
    GraphitiAdmissionRequest,
    GraphitiGovernedDecision,
    GraphitiProposalAuthorityBinding,
    GraphitiProjectionRequest,
)
from newsroom.control_plane.store import connect
from newsroom.control_plane.graphiti_admission_integration import (
    ConservativeGraphitiRelationPlanBuilder,
    ExistingIncrement4GenerationProjector,
    ExistingGovernedGraphitiProposalAuthority,
    ExistingGovernedGraphitiRightsAuthority,
    ExistingGovernedGraphitiAdmissionAuthority,
    GraphitiEntityAdmissionPlan,
    GraphitiRelationAdmissionPlan,
    GraphitiRelationOperationalDecisionPlan,
    compose_existing_graphiti_admission_consumer,
    conservative_entity_mention_plan,
)
from newsroom.control_plane.governed_context import (
    GovernedContextHydrator,
    GovernedContextStatus,
)
from newsroom.entities.models import (
    EntityMentionAdmissionRequest,
    EntityResolutionDependencyRequest,
    EntityResolutionDecision,
    EntityResolutionDecisionRequest,
    EntityResolutionProposalRequest,
)
from newsroom.entities.types import (
    ENTITY_NORMALISATION_CONTRACT_DIGEST,
    CanonicalEntityId,
    CanonicalEntityLifecycle,
    CanonicalEntityVersionId,
    EntityAliasId,
    EntityAliasKind,
    EntityKind,
    EntityMentionId,
    EntityResolutionDecisionAction,
    EntityResolutionDecisionId,
    EntityResolutionDependencyId,
    EntityResolutionProposalId,
    EntityResolutionProposalKind,
    EntityResolutionProposalVersionId,
    EntityScript,
)
from newsroom.extraction.models import ProposalDraft, ProposalEnvelope
from newsroom.extraction.types import (
    EvidenceRange,
    ExtractionPassageId,
    ExtractionProposalKind,
    ProposalEnvelopeId,
    ProposalSetId,
    ExtractionOutputId,
    ExtractionRunId,
    ExtractionRunVersionId,
    ProposalPredicateHint,
)
from newsroom.graphiti_adapter.admission import GraphitiProposalAdmissionAction
from newsroom.graphiti_adapter.identity import attempt_ids, typed_id
from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome
from newsroom.increment4.neo4j import Increment4Neo4jCurrentBuildRequest
from newsroom.projection.models import (
    ProjectionGenerationId,
    ProjectionGenerationState,
)
from newsroom.relations.editorial_models import (
    CanonicalEntityRelationEndpoint,
    EditorialRelationTemporalScope,
)
from newsroom.relations.editorial_types import (
    EditorialPredicateCode,
    EditorialRelationAssertionLifecycle,
    EditorialRelationDecisionAction,
)

from .authority_a2b_helpers import admit, open_object_system
from .entity_4b_helpers import seed_homonym_entity_fixture
from .extraction_4a_helpers import extraction_proof
from .projection_b2_helpers import MemoryNeo4jAdapter
from .source_3a_helpers import SOURCE_NOW
from . import test_graphiti_increment4_system

DIGEST = "sha256:" + ("ab" * 32)


def _binding(
    draft: ProposalDraft,
    *,
    cohort_seed: str | None = None,
) -> GraphitiProposalAuthorityBinding:
    shared = cohort_seed or draft.digest
    proposal_id = typed_id(ProposalEnvelopeId, "proposal", shared, draft.digest)
    proposal_set_id = typed_id(ProposalSetId, "set", shared)
    output_id = typed_id(ExtractionOutputId, "output", shared)
    run_id = typed_id(ExtractionRunId, "run", shared)
    run_version_id = typed_id(ExtractionRunVersionId, "version", shared)
    producer = digest_canonical({"producer": "fixture"})
    envelope_digest = digest_canonical({
        "proposal_id": str(proposal_id), "proposal_set_id": str(proposal_set_id),
        "output_id": str(output_id), "run_id": str(run_id),
        "run_version_id": str(run_version_id), "draft": draft.canonical_value(),
        "producer_contract_digest": producer,
    })
    return GraphitiProposalAuthorityBinding(
        graphiti_attempt_id=str(typed_id(ProposalEnvelopeId, "attempt", shared)),
        graphiti_attempt_authority_event_id=str(typed_id(ProposalEnvelopeId, "event", shared)),
        proposal_envelope=ProposalEnvelope(
            proposal_id=proposal_id, proposal_set_id=proposal_set_id,
            output_id=output_id, run_id=run_id, run_version_id=run_version_id,
            local_id=draft.local_id, kind=draft.kind,
            subject_placeholder=draft.subject_placeholder,
            object_placeholder=draft.object_placeholder,
            predicate_hint=draft.predicate_hint,
            confidence_basis_points=draft.confidence_basis_points,
            uncertainty_codes=draft.uncertainty_codes,
            rationale_codes=draft.rationale_codes, evidence=draft.evidence,
            producer_contract_digest=producer, canonical_digest=envelope_digest,
            retained_at=UtcTimestamp.parse("2026-08-24T00:00:00Z"),
        ),
    )


def _request() -> GraphitiAdmissionRequest:
    proposal = ProposalDraft(
        local_id="entity.0001",
        kind=ExtractionProposalKind.ENTITY_MENTION,
        subject_placeholder="Alice",
        object_placeholder=None,
        predicate_hint=None,
        confidence_basis_points=9_000,
        uncertainty_codes=(),
        rationale_codes=("EXACT_EXTRACTION_EVIDENCE",),
        evidence=(
            EvidenceRange(
                passage_id=ExtractionPassageId.parse(
                    "00000000-0000-4000-8000-000000007601"
                ),
                start_byte=0,
                end_byte=5,
                evidence_text_digest=DIGEST,
            ),
        ),
    )
    return GraphitiAdmissionRequest(
        queue_seq=1,
        proposal_key="proposal-key",
        source_receipt_digest=DIGEST,
        proposal_authority_binding=_binding(proposal),
        proposal=proposal,
        proposal_payload=proposal.canonical_value(),
        evidence_passages=({"passage_id": str(proposal.evidence[0].passage_id)},),
        proposed_endpoints=None,
        relation_statement=None,
        relation_temporal_bounds=None,
        source_lineage={"revision_id": "fixture"},
    )


def _relation_request(
    predicate_hint: ProposalPredicateHint = ProposalPredicateHint.SAME_PROCESS_AS,
) -> GraphitiAdmissionRequest:
    entity = _request().proposal
    endpoints = (
        replace(entity, local_id="entity.0001", subject_placeholder="Alice"),
        replace(entity, local_id="entity.0002", subject_placeholder="Bob"),
    )
    proposal = ProposalDraft(
        local_id="relation.0001",
        kind=ExtractionProposalKind.RELATION,
        subject_placeholder="Alice",
        object_placeholder="Bob",
        predicate_hint=predicate_hint,
        confidence_basis_points=9_000,
        uncertainty_codes=("REQUIRES_RELATION_ADMISSION",),
        rationale_codes=("EXACT_EXTRACTION_EVIDENCE",),
        evidence=entity.evidence,
    )
    cohort_seed = "relation-plan"
    return GraphitiAdmissionRequest(
        queue_seq=3,
        proposal_key="relation-proposal-key",
        source_receipt_digest=DIGEST,
        proposal_authority_binding=_binding(
            proposal,
            cohort_seed=cohort_seed,
        ),
        proposal=proposal,
        proposal_payload=proposal.canonical_value(),
        evidence_passages=(
            {
                "passage_id": str(proposal.evidence[0].passage_id),
                "language": "en-GB",
            },
        ),
        proposed_endpoints=("Alice", "Bob"),
        relation_statement="Alice and Bob participate in the same process.",
        relation_temporal_bounds={
            "valid_at": "2026-08-24T00:00:00Z",
            "invalid_at": None,
            "expired_at": None,
        },
        source_lineage={
            "revision_id": "fixture",
            "reference_time": "2026-08-24T00:00:00Z",
        },
        relation_endpoint_bindings=tuple(
            _binding(item, cohort_seed=cohort_seed) for item in endpoints
        ),
    )


def _plan(request: GraphitiAdmissionRequest) -> GraphitiEntityAdmissionPlan:
    mention_id = EntityMentionId.parse("00000000-0000-4000-8000-000000007602")
    source_id = ProposalEnvelopeId.parse("00000000-0000-4000-8000-000000007603")
    proposal_id = EntityResolutionProposalId.parse(
        "00000000-0000-4000-8000-000000007604"
    )
    version_id = EntityResolutionProposalVersionId.parse(
        "00000000-0000-4000-8000-000000007605"
    )
    proposal_request = EntityResolutionProposalRequest(
        proposal_id=proposal_id,
        proposal_version_id=version_id,
        version_number=1,
        expected_previous_version_id=None,
        source_proposal_id=source_id,
        expected_source_proposal_digest=DIGEST,
        kind=EntityResolutionProposalKind.MENTION_TO_NEW_ENTITY,
        subject_mention_id=mention_id,
        object_mention_id=None,
        candidate_entity_id=None,
        candidate_entity_version_id=None,
        confidence_basis_points=9_000,
        uncertainty_codes=(),
        basis_codes=("EXACT_EXTRACTION_EVIDENCE",),
        idempotency_key="entity-proposal",
    )
    retained_digest = digest_canonical(proposal_request.canonical_value())
    return GraphitiEntityAdmissionPlan(
        graphiti_proposal_digest=request.proposal.digest,
        graphiti_proposal_local_id=request.proposal.local_id,
        mention_requests=(
            EntityMentionAdmissionRequest(
                mention_id=mention_id,
                source_proposal_id=source_id,
                expected_source_proposal_digest=DIGEST,
                entity_kind=EntityKind.PERSON,
                language="en-GB",
                script=EntityScript.LATIN,
                normalized_text="alice",
                normalization_contract_digest=ENTITY_NORMALISATION_CONTRACT_DIGEST,
                idempotency_key="entity-mention",
            ),
        ),
        proposal_request=proposal_request,
        decision_request=EntityResolutionDecisionRequest(
            proposal_id=proposal_id,
            expected_proposal_version_id=version_id,
            expected_proposal_digest=retained_digest,
            action=EntityResolutionDecisionAction.ACCEPT,
            expected_decision_version=0,
            expected_previous_decision_id=None,
            accepted_entity_id=CanonicalEntityId.parse(
                "00000000-0000-4000-8000-000000007606"
            ),
            accepted_entity_version_id=CanonicalEntityVersionId.parse(
                "00000000-0000-4000-8000-000000007607"
            ),
            alias_id=EntityAliasId.parse("00000000-0000-4000-8000-000000007608"),
            alias_kind=EntityAliasKind.PRIMARY_NAME,
            reason_code="FIXTURE_ACCEPT",
            decision_policy_version="entity-resolution-policy-v1",
            idempotency_key="entity-decision",
        ),
    )


class _Entities:
    def __init__(self, plan: GraphitiEntityAdmissionPlan) -> None:
        self.plan = plan
        self.calls: list[str] = []
        self.retained_decision: EntityResolutionDecision | None = None
        self.preferred_entity_id = plan.decision_request.accepted_entity_id

    def admit_mention(self, request, *, proof):
        assert isinstance(proof, AuthenticationProof)
        self.calls.append("mention")
        return SimpleNamespace()

    def propose_resolution(self, request, *, proof):
        self.calls.append("propose")
        return SimpleNamespace(
            proposal_id=request.proposal_id,
            proposal_version_id=request.proposal_version_id,
            version_number=request.version_number,
            previous_proposal_version_id=request.expected_previous_version_id,
            source_proposal_id=request.source_proposal_id,
            source_proposal_digest=request.expected_source_proposal_digest,
            kind=request.kind,
            subject_mention_id=request.subject_mention_id,
            object_mention_id=request.object_mention_id,
            candidate_entity_id=request.candidate_entity_id,
            candidate_entity_version_id=request.candidate_entity_version_id,
            confidence_basis_points=request.confidence_basis_points,
            uncertainty_codes=request.uncertainty_codes,
            basis_codes=request.basis_codes,
            stable_semantic_digest=request.stable_semantic_digest,
            canonical_digest=digest_canonical(
                {"retained_proposal": request.canonical_value()}
            ),
        )

    def decide_resolution(self, request, *, proof):
        self.calls.append("decide")
        self.retained_decision = EntityResolutionDecision(
            decision_id=EntityResolutionDecisionId.parse(
                "00000000-0000-4000-8000-000000007609"
            ),
            proposal_id=request.proposal_id,
            proposal_version_id=request.expected_proposal_version_id,
            proposal_digest=request.expected_proposal_digest,
            action=request.action,
            decision_version=1,
            previous_decision_id=None,
            accepted_entity_id=request.accepted_entity_id,
            accepted_entity_version_id=request.accepted_entity_version_id,
            alias_id=request.alias_id,
            reason_code=request.reason_code,
            decision_policy_version=request.decision_policy_version,
            authority_event_id=EventId.parse("00000000-0000-4000-8000-000000007610"),
            authority_ledger_seq=42,
            recorded_at=UtcTimestamp.parse("2026-08-24T00:00:00Z"),
        )
        return self.retained_decision

    def decision(self, proposal_id, *, proof):
        assert proposal_id == self.plan.proposal_request.proposal_id
        return self.retained_decision

    def entity_version(self, version_id, *, proof):
        assert version_id == self.plan.decision_request.accepted_entity_version_id
        return SimpleNamespace(
            entity_id=self.plan.decision_request.accepted_entity_id,
            version_number=1,
            lifecycle=CanonicalEntityLifecycle.ACTIVE,
            canonical_value=lambda: {
                "entity_id": str(self.plan.decision_request.accepted_entity_id),
                "entity_version_id": str(version_id),
                "lifecycle": "ACTIVE",
                "version_number": 1,
            },
        )

    def preferred(self, entity_id, *, proof):
        return SimpleNamespace(
            entity_id=entity_id,
            current_entity_version_id=(
                self.plan.decision_request.accepted_entity_version_id
            ),
            preferred_entity_id=self.preferred_entity_id,
            lifecycle=CanonicalEntityLifecycle.ACTIVE,
        )

    def aliases(self, entity_id, *, limit, proof):
        assert entity_id == self.plan.decision_request.accepted_entity_id
        assert limit == 16
        return ()


def test_existing_authority_executes_typed_entity_commands() -> None:
    request = _request()
    plan = _plan(request)
    entities = _Entities(plan)
    authority = ExistingGovernedGraphitiAdmissionAuthority(
        entities=entities,  # type: ignore[arg-type]
        relations=SimpleNamespace(),  # type: ignore[arg-type]
        proof=AuthenticationProof(method="STATIC_TOKEN", credential="fixture"),
        entity_plan=lambda *_: plan,
        relation_plan=lambda *_: pytest.fail("relation planner called"),
    )

    decision = authority.decide_entity_resolution(
        request,
        required_action=GraphitiProposalAdmissionAction.ADMIT,
        idempotency_key="graphiti-admit:proposal-key",
    )

    assert entities.calls == ["mention", "propose", "decide"]
    assert decision.action is GraphitiProposalAdmissionAction.ADMIT
    assert decision.authority_ledger_seq == 42

    context = authority.current_context(request, decision)

    assert context is not None
    assert context.currentness_state == "CURRENT"
    assert tuple(item.authority_kind for item in context.bindings) == (
        "CANONICAL_ENTITY",
        "ENTITY_RESOLUTION_DECISION",
    )
    assert context.admitted_temporal_fields == (
        ("admitted_at", "2026-08-24T00:00:00.000000Z"),
    )
    assert context.admitted_structured_value["authority_kind"] == "CANONICAL_ENTITY"


def test_existing_authority_rejects_substituted_retained_entity_proposal() -> None:
    request = _request()
    plan = _plan(request)
    entities = _Entities(plan)
    propose_resolution = entities.propose_resolution

    def substituted_proposal(proposal_request, *, proof):
        proposed = propose_resolution(proposal_request, proof=proof)
        proposed.source_proposal_digest = "sha256:" + ("cd" * 32)
        return proposed

    entities.propose_resolution = substituted_proposal  # type: ignore[method-assign]
    authority = ExistingGovernedGraphitiAdmissionAuthority(
        entities=entities,  # type: ignore[arg-type]
        relations=SimpleNamespace(),  # type: ignore[arg-type]
        proof=AuthenticationProof(method="STATIC_TOKEN", credential="fixture"),
        entity_plan=lambda *_: plan,
        relation_plan=lambda *_: pytest.fail("relation planner called"),
    )

    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="differs from the retained authority proposal",
    ):
        authority.decide_entity_resolution(
            request,
            required_action=GraphitiProposalAdmissionAction.ADMIT,
            idempotency_key="graphiti-admit:proposal-key",
        )

    assert entities.calls == ["mention", "propose"]


def test_actual_entity_authority_replays_partial_work_and_builds_four_item_generation(
    monkeypatch,
    tmp_path,
) -> None:
    state = seed_homonym_entity_fixture(tmp_path)
    sources = (
        state.en_transit_source,
        state.en_association_source,
        state.zh_transit_source,
        state.zh_association_source,
    )

    ingest_id = "00000000-0000-4000-8000-0000000076d0"
    unsigned_receipt = {
        "ingest_id": ingest_id,
        "outcome": "COMPLETE",
        "proposal_count": len(sources),
    }
    receipt_digest = digest_bytes(canonical_json_bytes(unsigned_receipt))
    input_binding = state.extraction.input_binding
    passage_by_id = {
        str(item.passage_id): item for item in input_binding.passages
    }

    def admission_request(source) -> GraphitiAdmissionRequest:
        proposal = ProposalDraft(
            local_id=source.local_id,
            kind=source.kind,
            subject_placeholder=source.subject_placeholder,
            object_placeholder=source.object_placeholder,
            predicate_hint=source.predicate_hint,
            confidence_basis_points=source.confidence_basis_points,
            uncertainty_codes=source.uncertainty_codes,
            rationale_codes=source.rationale_codes,
            evidence=source.evidence,
        )
        language = "zh-HK" if ".zh-hk" in source.local_id else "en-GB"
        seed = source.canonical_digest
        return GraphitiAdmissionRequest(
            queue_seq=1,
            proposal_key=f"proposal-{source.local_id}",
            source_receipt_digest=receipt_digest,
            proposal_authority_binding=GraphitiProposalAuthorityBinding(
                graphiti_attempt_id=str(
                    typed_id(ProposalEnvelopeId, "attempt", seed)
                ),
                graphiti_attempt_authority_event_id=str(
                    typed_id(ProposalEnvelopeId, "attempt-event", seed)
                ),
                proposal_envelope=source,
            ),
            proposal=proposal,
            proposal_payload=proposal.canonical_value(),
            evidence_passages=tuple(
                {
                    **passage_by_id[str(item.passage_id)].canonical_value(),
                    "language": language,
                }
                for item in source.evidence
            ),
            proposed_endpoints=None,
            relation_statement=None,
            relation_temporal_bounds=None,
            source_lineage={
                "ingest_id": ingest_id,
                "source_id": "UK-01",
                "item_key": "item-1",
                "revision_id": str(input_binding.revision_id),
                "reference_time": SOURCE_NOW.to_text(),
                "temporal_basis": "SOURCE_PUBLISHED",
            },
        )

    requests = tuple(
        admission_request(source) for source in sources
    )
    proof = extraction_proof()
    adapter = MemoryNeo4jAdapter()

    monkeypatch.setattr(test_graphiti_increment4_system, "FIXED_NOW", SOURCE_NOW)
    with test_graphiti_increment4_system._open(tmp_path, adapter) as system:
        connection = connect(str(tmp_path / "unpublished.sqlite3"))
        receipt = {**unsigned_receipt, "receipt_digest": receipt_digest}
        connection.execute(
            "INSERT INTO unpublished_graphiti_ingest("
            "ingest_id,source_id,item_key,outcome,proposal_count,entity_count,"
            "relation_count,failure_code,temporal_basis,reference_time,"
            "generation_id,receipt_digest,at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ingest_id,
                "UK-01",
                "item-1",
                "COMPLETE",
                4,
                4,
                0,
                "NONE",
                "SOURCE_PUBLISHED",
                "2026-08-24T00:00:00Z",
                "fixture-generation",
                receipt_digest,
                "2026-08-24T00:00:00Z",
            ),
        )
        connection.execute(
            "INSERT INTO unpublished_graphiti_receipts(ingest_id,receipt_json) "
            "VALUES(?,?)",
            (ingest_id, canonical_json_bytes(receipt).decode()),
        )
        queued_requests = []
        for request in requests:
            cursor = connection.execute(
                "INSERT INTO unpublished_graphiti_admission_queue("
                "proposal_key,ingest_id,source_revision_id,source_receipt_digest,"
                "proposal_digest,proposal_kind,request_json,request_digest,state,"
                "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,'READY',?,?)",
                (
                    request.proposal_key,
                    ingest_id,
                    str(input_binding.revision_id),
                    receipt_digest,
                    request.proposal.digest,
                    request.proposal.kind.value,
                    "{}",
                    request.proposal_key,
                    "2026-08-24T00:00:00Z",
                    "2026-08-24T00:00:00Z",
                ),
            )
            queued = replace(request, queue_seq=cursor.lastrowid)
            encoded = canonical_json_bytes(queued.canonical_value())
            connection.execute(
                "UPDATE unpublished_graphiti_admission_queue "
                "SET request_json=?,request_digest=? WHERE proposal_key=?",
                (encoded.decode(), digest_bytes(encoded), queued.proposal_key),
            )
            queued_requests.append(queued)
        connection.commit()

        first_plan = conservative_entity_mention_plan(
            queued_requests[0],
            GraphitiProposalAdmissionAction.ADMIT,
            f"graphiti-admit:{queued_requests[0].proposal_key}",
        )
        system.entities.admit_mention(first_plan.mention_requests[0], proof=proof)
        retained = system.entities.propose_resolution(
            first_plan.proposal_request,
            proof=proof,
        )
        assert retained.canonical_digest != (
            first_plan.decision_request.expected_proposal_digest
        )

        class CountingEntities:
            def __init__(self, wrapped) -> None:
                self.wrapped = wrapped
                self.decision_calls = 0

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def decide_resolution(self, request, *, proof):
                self.decision_calls += 1
                return self.wrapped.decide_resolution(request, proof=proof)

        entities = CountingEntities(system.entities)
        authority = ExistingGovernedGraphitiAdmissionAuthority(
            entities=entities,  # type: ignore[arg-type]
            relations=SimpleNamespace(),  # type: ignore[arg-type]
            proof=proof,
            entity_plan=conservative_entity_mention_plan,
            relation_plan=lambda *_: pytest.fail("relation planner called"),
        )
        rights = ExistingGovernedGraphitiRightsAuthority(
            objects=system.objects,
            proof=proof,
        )
        consumer = GraphitiAdmissionConsumer(
            connection,
            proposal_authority=SimpleNamespace(),  # type: ignore[arg-type]
            authority=authority,
            projector=ExistingIncrement4GenerationProjector(
                controller=system.increment4,
                proof=proof,
            ),
            rights=rights,
            clock=lambda: datetime(2026, 8, 24, 12, tzinfo=UTC),
        )
        decisions = consumer.drain(
            worker_id="fixture-worker",
            limit=4,
            ingest_ids=(ingest_id,),
        )

        assert decisions.decided == 4
        assert entities.decision_calls == 4
        with sqlite3.connect(state.extraction.database) as authority_connection:
            assert authority_connection.execute(
                "SELECT COUNT(*) FROM entity_mentions"
            ).fetchone() == (4,)
            assert authority_connection.execute(
                "SELECT COUNT(*) FROM entity_resolution_proposal_versions"
            ).fetchone() == (4,)
            assert authority_connection.execute(
                "SELECT COUNT(*) FROM entity_resolution_decisions"
            ).fetchone() == (4,)

        projection = consumer.finalise_decided_cohort(ingest_ids=(ingest_id,))

        assert projection.projected == 4
        assert connection.execute(
            "SELECT COUNT(*) FROM unpublished_graphiti_projection_receipts"
        ).fetchone() == (4,)
        assert connection.execute(
            "SELECT COUNT(*) FROM unpublished_graphiti_projection_reconciliations"
        ).fetchone() == (1,)
        assert len(adapter.deliveries) > 0
        context = GovernedContextHydrator(
            connection,
            authority=authority,
            rights=rights,
            clock=lambda: datetime(2042, 3, 12, 10, tzinfo=UTC),
        ).hydrate()
        assert context.status is GovernedContextStatus.READY
        assert len(context.items) == 4
        assert {item.projection_generation_id for item in context.items} == {
            context.projection_generation_id
        }
        connection.close()


def test_existing_authority_marks_merged_entity_head_stale() -> None:
    request = _request()
    plan = _plan(request)
    entities = _Entities(plan)
    entities.preferred_entity_id = CanonicalEntityId.parse(
        "00000000-0000-4000-8000-0000000076ff"
    )
    authority = ExistingGovernedGraphitiAdmissionAuthority(
        entities=entities,  # type: ignore[arg-type]
        relations=SimpleNamespace(),  # type: ignore[arg-type]
        proof=AuthenticationProof(method="STATIC_TOKEN", credential="fixture"),
        entity_plan=lambda *_: plan,
        relation_plan=lambda *_: pytest.fail("relation planner called"),
    )
    decision = authority.decide_entity_resolution(
        request,
        required_action=GraphitiProposalAdmissionAction.ADMIT,
        idempotency_key="graphiti-admit:proposal-key",
    )

    context = authority.current_context(request, decision)

    assert context is not None
    assert context.currentness_state == "STALE"


def test_existing_authority_hydrates_current_relation_facts() -> None:
    entity_request = _request()
    endpoint_drafts = (
        replace(
            entity_request.proposal,
            local_id="entity.0001",
            subject_placeholder="Alice",
        ),
        replace(
            entity_request.proposal,
            local_id="entity.0002",
            subject_placeholder="Bob",
        ),
    )
    proposal = ProposalDraft(
        local_id="relation.0001",
        kind=ExtractionProposalKind.RELATION,
        subject_placeholder="Alice",
        object_placeholder="Bob",
        predicate_hint=ProposalPredicateHint.SUPPORTS,
        confidence_basis_points=9_000,
        uncertainty_codes=(),
        rationale_codes=("EXACT_EXTRACTION_EVIDENCE",),
        evidence=entity_request.proposal.evidence,
    )
    request = GraphitiAdmissionRequest(
        queue_seq=1,
        proposal_key="relation-proposal-key",
        source_receipt_digest=DIGEST,
        proposal_authority_binding=_binding(proposal, cohort_seed="relation"),
        proposal=proposal,
        proposal_payload=proposal.canonical_value(),
        evidence_passages=entity_request.evidence_passages,
        proposed_endpoints=("Alice", "Bob"),
        relation_endpoint_bindings=tuple(
            _binding(item, cohort_seed="relation") for item in endpoint_drafts
        ),
        relation_statement="Alice supports Bob",
        relation_temporal_bounds={
            "valid_at": "2026-08-24T00:00:00Z",
            "invalid_at": None,
            "expired_at": None,
        },
        source_lineage=entity_request.source_lineage,
    )
    decision = GraphitiGovernedDecision(
        proposal_key=request.proposal_key,
        proposal_digest=proposal.digest,
        proposal_kind=proposal.kind,
        proposal_local_id=proposal.local_id,
        action=GraphitiProposalAdmissionAction.ADMIT,
        decision_id="00000000-0000-4000-8000-0000000076a1",
        authority_ledger_seq=43,
        reason_code="FIXTURE_ACCEPT",
        authority_receipt_digest=DIGEST,
        admitted_authority_id="00000000-0000-4000-8000-0000000076a6",
        endpoint_resolution_decision_ids=(
            "00000000-0000-4000-8000-0000000076b1",
            "00000000-0000-4000-8000-0000000076b2",
        ),
        resolved_endpoint_names=("Alice", "Bob"),
    )
    subject = CanonicalEntityRelationEndpoint(
        entity_id=CanonicalEntityId.parse("00000000-0000-4000-8000-0000000076a2"),
        entity_version_id=CanonicalEntityVersionId.parse(
            "00000000-0000-4000-8000-0000000076a3"
        ),
    )
    object_ = CanonicalEntityRelationEndpoint(
        entity_id=CanonicalEntityId.parse("00000000-0000-4000-8000-0000000076a4"),
        entity_version_id=CanonicalEntityVersionId.parse(
            "00000000-0000-4000-8000-0000000076a5"
        ),
    )
    temporal = EditorialRelationTemporalScope(
        valid_from=UtcTimestamp.parse("2026-08-24T00:00:00Z"),
        valid_until=None,
        observed_at=UtcTimestamp.parse("2026-08-24T00:00:00Z"),
    )
    assertion = SimpleNamespace(
        assertion_id="00000000-0000-4000-8000-0000000076a6",
        proposal_version_id="00000000-0000-4000-8000-0000000076a7",
        predicate=EditorialPredicateCode.SUPPORTS,
        subject=subject,
        object=object_,
        statement="Alice supports Bob",
        temporal_scope=temporal,
        uncertainty_codes=(),
        admitted_at=UtcTimestamp.parse("2026-08-24T00:00:00Z"),
    )
    retained = SimpleNamespace(
        action=EditorialRelationDecisionAction.ACCEPT,
        decision_id=decision.decision_id,
        decision_version=1,
        authority_ledger_seq=43,
        assertion_id=assertion.assertion_id,
    )
    relations = SimpleNamespace(
        decision=lambda *_args, **_kwargs: retained,
        current=lambda *_args, **_kwargs: SimpleNamespace(
            assertion=assertion,
            lifecycle=EditorialRelationAssertionLifecycle.ACTIVE,
            current_decision_id=decision.decision_id,
            current_decision_version=1,
        ),
        proposal_version=lambda *_args, **_kwargs: SimpleNamespace(version_number=1),
    )
    endpoint_proposal_ids = (
        EntityResolutionProposalId.parse("00000000-0000-4000-8000-0000000076c1"),
        EntityResolutionProposalId.parse("00000000-0000-4000-8000-0000000076c2"),
    )
    relation_envelope = request.proposal_authority_binding.proposal_envelope
    dependency_requests = tuple(
        EntityResolutionDependencyRequest(
            dependency_id=EntityResolutionDependencyId.parse(
                f"00000000-0000-4000-8000-{index:012d}"
            ),
            dependent_proposal_id=relation_envelope.proposal_id,
            expected_dependent_proposal_digest=relation_envelope.canonical_digest,
            resolution_proposal_id=proposal_id,
            expected_resolution_proposal_version_id=(
                EntityResolutionProposalVersionId.parse(
                    f"00000000-0000-4000-8000-{index + 10:012d}"
                )
            ),
            expected_resolution_proposal_digest=DIGEST,
            material=True,
            idempotency_key=f"dependency-{index}",
        )
        for index, proposal_id in enumerate(endpoint_proposal_ids, start=1)
    )
    endpoint_by_entity = {
        subject.entity_id: subject,
        object_.entity_id: object_,
    }
    endpoint_decision_by_proposal = dict(
        zip(
            endpoint_proposal_ids,
            decision.endpoint_resolution_decision_ids,
            strict=True,
        )
    )
    endpoint_state = {"stale": False}

    def endpoint_decision(
        proposal_id: EntityResolutionProposalId,
        *,
        proof: object,
    ) -> SimpleNamespace:
        del proof
        return SimpleNamespace(
            action=EntityResolutionDecisionAction.ACCEPT,
            decision_id=endpoint_decision_by_proposal[proposal_id],
        )

    def preferred_endpoint(
        entity_id: CanonicalEntityId,
        *,
        proof: object,
    ) -> SimpleNamespace:
        del proof
        endpoint = endpoint_by_entity[entity_id]
        preferred_entity_id = (
            object_.entity_id
            if endpoint_state["stale"] and entity_id == subject.entity_id
            else entity_id
        )
        return SimpleNamespace(
            entity_id=entity_id,
            current_entity_version_id=endpoint.entity_version_id,
            preferred_entity_id=preferred_entity_id,
            lifecycle=CanonicalEntityLifecycle.ACTIVE,
        )

    def endpoint_version(
        version_id: CanonicalEntityVersionId,
        *,
        proof: object,
    ) -> SimpleNamespace:
        del proof
        endpoint = next(
            item
            for item in endpoint_by_entity.values()
            if item.entity_version_id == version_id
        )
        return SimpleNamespace(
            entity_id=endpoint.entity_id,
            entity_version_id=endpoint.entity_version_id,
            version_number=1,
            lifecycle=CanonicalEntityLifecycle.ACTIVE,
        )

    entities = SimpleNamespace(
        decision=endpoint_decision,
        preferred=preferred_endpoint,
        entity_version=endpoint_version,
    )
    plan = GraphitiRelationAdmissionPlan(
        graphiti_proposal_digest=proposal.digest,
        graphiti_proposal_local_id=proposal.local_id,
        proposal_request=SimpleNamespace(proposal_id="relation-proposal"),  # type: ignore[arg-type]
        decision_request=SimpleNamespace(),  # type: ignore[arg-type]
        dependency_requests=dependency_requests,
        endpoint_resolution_proposal_ids=endpoint_proposal_ids,
        resolved_endpoint_names=("Alice", "Bob"),
    )
    authority = ExistingGovernedGraphitiAdmissionAuthority(
        entities=entities,  # type: ignore[arg-type]
        relations=relations,  # type: ignore[arg-type]
        proof=AuthenticationProof(method="STATIC_TOKEN", credential="fixture"),
        entity_plan=lambda *_: pytest.fail("entity planner called"),
        relation_plan=lambda *_: plan,
    )

    context = authority.current_context(request, decision)

    assert context is not None
    assert context.currentness_state == "CURRENT"
    structured = context.admitted_structured_value
    assert structured["authority_kind"] == "EDITORIAL_RELATION_ASSERTION"
    assert structured["assertion"]["predicate"] == "SUPPORTS"  # type: ignore[index]
    assert structured["assertion"]["statement"] == "Alice supports Bob"  # type: ignore[index]
    assert {
        item.authority_id
        for item in context.bindings
        if item.authority_kind == "CANONICAL_ENTITY"
    } == {str(subject.entity_id), str(object_.entity_id)}

    endpoint_state["stale"] = True
    stale_context = authority.current_context(request, decision)

    assert stale_context is not None
    assert stale_context.currentness_state == "STALE"


def test_existing_authority_rejects_unbound_graphiti_plan() -> None:
    request = _request()
    plan = _plan(request)
    unbound = GraphitiEntityAdmissionPlan(
        graphiti_proposal_digest=DIGEST,
        graphiti_proposal_local_id=plan.graphiti_proposal_local_id,
        mention_requests=plan.mention_requests,
        proposal_request=plan.proposal_request,
        decision_request=plan.decision_request,
    )
    authority = ExistingGovernedGraphitiAdmissionAuthority(
        entities=_Entities(plan),  # type: ignore[arg-type]
        relations=SimpleNamespace(),  # type: ignore[arg-type]
        proof=AuthenticationProof(method="STATIC_TOKEN", credential="fixture"),
        entity_plan=lambda *_: unbound,
        relation_plan=lambda *_: pytest.fail("relation planner called"),
    )

    with pytest.raises(GraphitiAdmissionConsumerError, match="exact Graphiti"):
        authority.decide_entity_resolution(
            request,
            required_action=None,
            idempotency_key="graphiti-admit:proposal-key",
        )


def test_required_rights_rejection_is_checked_before_authority_mutation() -> None:
    request = _request()
    plan = _plan(request)
    entities = _Entities(plan)
    authority = ExistingGovernedGraphitiAdmissionAuthority(
        entities=entities,  # type: ignore[arg-type]
        relations=SimpleNamespace(),  # type: ignore[arg-type]
        proof=AuthenticationProof(method="STATIC_TOKEN", credential="fixture"),
        entity_plan=lambda *_: plan,
        relation_plan=lambda *_: pytest.fail("relation planner called"),
    )

    with pytest.raises(GraphitiAdmissionConsumerError, match="required admission"):
        authority.decide_entity_resolution(
            request,
            required_action=GraphitiProposalAdmissionAction.REJECT,
            idempotency_key="graphiti-admit:proposal-key",
        )

    assert entities.calls == []


def test_proposal_authority_binds_exact_4d_attempt_and_4a_envelope() -> None:
    request = _request()
    binding = request.proposal_authority_binding
    envelope = binding.proposal_envelope
    ingest_id = "fixture"
    attempt_id = attempt_ids(ingest_id, 1)[0]
    passage = SimpleNamespace(
        passage_id=envelope.evidence[0].passage_id,
        text_digest=digest_bytes(
            b"Alice briefed Example Council in a longer retained passage."
        ),
    )
    attempt = SimpleNamespace(
        attempt_id=attempt_id,
        attempt_number=1,
        outcome=GraphitiAdapterOutcome.COMPLETE,
        authority_event_id=EventId.parse(binding.graphiti_attempt_authority_event_id),
        output_id=envelope.output_id,
        proposal_set_id=envelope.proposal_set_id,
        manifest_id="manifest",
        run_id=envelope.run_id,
        run_version_id=envelope.run_version_id,
    )
    manifest = SimpleNamespace(
        manifest_id="manifest",
        run_id=envelope.run_id,
        requested_run_version_id=envelope.run_version_id,
        passages=(passage,),
    )
    unsigned_raw = {
        "attempt_number": 1,
        "provider_attempt_number": 1,
        "generation_id": "newsroom-eval-generation-v1",
        "episode_uuid": ingest_id,
        "temporal_basis": "SOURCE_PUBLISHED",
        "reference_time": "2026-08-24T00:00:00Z",
        "proposals": [request.proposal.canonical_value()],
        "passages": [{"passage_id": str(passage.passage_id)}],
        "entities": [],
        "relations": [],
        "proposal_count": 1,
        "entity_count": 0,
        "relation_count": 0,
    }
    inner_digest = digest_bytes(canonical_json_bytes(unsigned_raw))
    raw_bytes = canonical_json_bytes(
        {**unsigned_raw, "raw_output_digest": inner_digest}
    )
    output_view = SimpleNamespace(
        output_id=envelope.output_id,
        canonical_digest=digest_bytes(raw_bytes),
    )
    adapter = SimpleNamespace(
        attempt=lambda *_args, **_kwargs: attempt,
        manifest_for_attempt=lambda *_args, **_kwargs: manifest,
    )
    extraction = SimpleNamespace(
        metadata=lambda *_args, **_kwargs: SimpleNamespace(
            terminal=True,
            run_id=envelope.run_id,
            output=output_view,
        ),
        raw_output=lambda *_args, **_kwargs: SimpleNamespace(
            view=output_view,
            canonical_bytes=raw_bytes,
        ),
        proposals=lambda *_args, **_kwargs: (envelope,),
    )
    authority = ExistingGovernedGraphitiProposalAuthority(
        adapter=adapter,  # type: ignore[arg-type]
        extraction=extraction,  # type: ignore[arg-type]
        proof=AuthenticationProof(method="STATIC_TOKEN", credential="fixture"),
    )

    retained = authority.bind_proposal(
        ingest_id=ingest_id,
        terminal_receipt={
            "ingest_id": ingest_id,
            "outcome": "COMPLETE",
            **unsigned_raw,
            "raw_output_digest": inner_digest,
        },
        proposal=request.proposal,
    )

    assert retained == GraphitiProposalAuthorityBinding(
        graphiti_attempt_id=str(attempt_id),
        graphiti_attempt_authority_event_id=(
            binding.graphiti_attempt_authority_event_id
        ),
        proposal_envelope=envelope,
    )
    assert authority.bind_proposal(
        ingest_id=ingest_id,
        terminal_receipt={
            "ingest_id": ingest_id,
            "outcome": "COMPLETE",
            **unsigned_raw,
            "raw_output_digest": DIGEST,
        },
        proposal=request.proposal,
    ) is None
    assert authority.bind_proposal(
        ingest_id=ingest_id,
        terminal_receipt={
            "ingest_id": ingest_id,
            "outcome": "COMPLETE",
            **unsigned_raw,
            "proposals": [],
            "proposal_count": 0,
            "raw_output_digest": inner_digest,
        },
        proposal=request.proposal,
    ) is None
    attempt.outcome = GraphitiAdapterOutcome.PARTIAL
    assert authority.bind_proposal(
        ingest_id=ingest_id,
        terminal_receipt={
            "ingest_id": ingest_id,
            "outcome": "PARTIAL",
            **unsigned_raw,
            "raw_output_digest": inner_digest,
        },
        proposal=request.proposal,
    ) is None
    attempt.outcome = GraphitiAdapterOutcome.COMPLETE
    assert authority.bind_proposal(
        ingest_id=ingest_id,
        terminal_receipt={
            "ingest_id": ingest_id,
            "outcome": "PARTIAL",
            **unsigned_raw,
            "raw_output_digest": inner_digest,
        },
        proposal=request.proposal,
    ) is None


def test_generation_projector_uses_one_current_increment4_snapshot() -> None:
    request = _request()
    decision = GraphitiGovernedDecision(
        proposal_key=request.proposal_key,
        proposal_digest=request.proposal.digest,
        proposal_kind=request.proposal.kind,
        proposal_local_id=request.proposal.local_id,
        action=GraphitiProposalAdmissionAction.ADMIT,
        decision_id="decision:proposal-key",
        authority_ledger_seq=42,
        reason_code="EXACT_MENTION_TO_NEW_UNKNOWN_ENTITY",
        authority_receipt_digest=DIGEST,
        admitted_authority_id="00000000-0000-4000-8000-0000000076f1",
    )
    generation_id = ProjectionGenerationId.parse(
        "00000000-0000-4000-8000-0000000076f2"
    )
    observed: list[tuple[object, object]] = []

    class Controller:
        def build_current_and_promote(self, build_request, *, proof):
            observed.append((build_request, proof))
            return SimpleNamespace(
                generation=SimpleNamespace(
                    generation_id=generation_id,
                    state=ProjectionGenerationState.ACTIVE,
                ),
                source_watermark_ledger_seq=42,
                checkpoint_ledger_seq=42,
                source_snapshot_digest=DIGEST,
                validation=SimpleNamespace(
                    validation_digest=DIGEST,
                    projection_state_digest=DIGEST,
                    source_snapshot_digest=None,
                    source_watermark_ledger_seq=None,
                    source_request_digest=None,
                ),
                promotion=SimpleNamespace(
                    promotion_digest=DIGEST,
                    validation_digest=DIGEST,
                ),
                projection_state_digest=DIGEST,
            )

    proof = AuthenticationProof(method="STATIC_TOKEN", credential="fixture")
    projector = ExistingIncrement4GenerationProjector(
        controller=Controller(),  # type: ignore[arg-type]
        proof=proof,
    )

    result = projector.build_and_promote_increment4_cohort(
        (GraphitiProjectionRequest(request=request, decision=decision),),
        cohort_digest=DIGEST,
        generation_id=str(generation_id),
        idempotency_key="graphiti-generation:fixture",
    )

    assert result.source_snapshot_digest == DIGEST
    assert result.admitted_authority_ids == (decision.admitted_authority_id,)
    assert observed[0][1] == proof
    build_request = observed[0][0]
    assert isinstance(build_request, Increment4Neo4jCurrentBuildRequest)
    assert build_request.generation_id == generation_id
    assert build_request.idempotency_key == "graphiti-generation:fixture"
    assert build_request.allow_active_extension is True


def test_generation_projector_accepts_only_complete_source_bound_active_extension() -> None:
    request = _request()
    decision = GraphitiGovernedDecision(
        proposal_key=request.proposal_key,
        proposal_digest=request.proposal.digest,
        proposal_kind=request.proposal.kind,
        proposal_local_id=request.proposal.local_id,
        action=GraphitiProposalAdmissionAction.ADMIT,
        decision_id="decision:active-extension",
        authority_ledger_seq=42,
        reason_code="EXACT_MENTION_TO_NEW_UNKNOWN_ENTITY",
        authority_receipt_digest=DIGEST,
        admitted_authority_id="00000000-0000-4000-8000-0000000076f1",
    )
    requested = ProjectionGenerationId.parse(
        "00000000-0000-4000-8000-0000000076f2"
    )
    active = ProjectionGenerationId.parse(
        "00000000-0000-4000-8000-0000000076f4"
    )

    class Controller:
        source_snapshot_digest = DIGEST
        validation_snapshot_digest = DIGEST
        source_watermark = 42
        source_request_digest = digest_canonical(
            {
                "generation_id": str(requested),
                "reason_code": "GRAPHITI_ADMISSION_COHORT",
                "idempotency_key": "graphiti-generation:active-extension",
                "purge_retired_generation": True,
                "allow_active_extension": True,
            }
        )

        def build_current_and_promote(self, build_request, *, proof):
            del build_request, proof
            return SimpleNamespace(
                generation=SimpleNamespace(
                    generation_id=active,
                    state=ProjectionGenerationState.ACTIVE,
                ),
                source_watermark_ledger_seq=self.source_watermark,
                checkpoint_ledger_seq=50,
                source_snapshot_digest=self.source_snapshot_digest,
                validation=SimpleNamespace(
                    validation_digest=digest_canonical({"validation": "cohort"}),
                    projection_state_digest=DIGEST,
                    source_snapshot_digest=self.validation_snapshot_digest,
                    source_watermark_ledger_seq=self.source_watermark,
                    source_request_digest=self.source_request_digest,
                ),
                promotion=SimpleNamespace(
                    promotion_digest=digest_canonical({"promotion": "original"}),
                    validation_digest=digest_canonical({"validation": "original"}),
                ),
                projection_state_digest=DIGEST,
            )

    controller = Controller()
    projector = ExistingIncrement4GenerationProjector(
        controller=controller,  # type: ignore[arg-type]
        proof=AuthenticationProof(method="STATIC_TOKEN", credential="fixture"),
    )
    result = projector.build_and_promote_increment4_cohort(
        (GraphitiProjectionRequest(request=request, decision=decision),),
        cohort_digest=DIGEST,
        generation_id=str(requested),
        idempotency_key="graphiti-generation:active-extension",
    )
    assert result.generation_id == str(active)
    assert result.schema_version == GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION
    assert result.source_request_digest == controller.source_request_digest

    controller.source_snapshot_digest = digest_canonical({"wrong": "snapshot"})
    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="exact cohort cutoff",
    ):
        projector.build_and_promote_increment4_cohort(
            (GraphitiProjectionRequest(request=request, decision=decision),),
            cohort_digest=DIGEST,
            generation_id=str(requested),
            idempotency_key="graphiti-generation:active-extension",
        )


def test_rights_authority_rehydrates_exact_current_source_bytes() -> None:
    data = b"Alice"
    admission_id = "00000000-0000-4000-8000-0000000076f3"
    request = replace(
        _request(),
        evidence_passages=(
            {
                "passage_id": "00000000-0000-4000-8000-000000007601",
                "admission_id": admission_id,
                "purpose": "graphiti.corpus-ingest",
                "byte_offset": 0,
                "byte_length": len(data),
                "blob_digest": digest_bytes(data),
                "allowed_use": "proposal.extraction",
                "security_scope": "evaluation",
                "retention_scope": "disposable-workspace",
            },
        ),
    )
    proof = AuthenticationProof(method="STATIC_TOKEN", credential="fixture")
    calls: list[object] = []

    class Objects:
        def rehydrate(self, hydration, *, proof):
            calls.append((hydration, proof))
            return SimpleNamespace(
                data=data,
                decision=SimpleNamespace(
                    admission_id=hydration.admission_id,
                    offset=0,
                    allowed_bytes=len(data),
                    purpose="graphiti.corpus-ingest",
                    allowed_use="proposal.extraction",
                    security_scope="evaluation",
                    retention_scope="disposable-workspace",
                ),
            )

    authority = ExistingGovernedGraphitiRightsAuthority(
        objects=Objects(),  # type: ignore[arg-type]
        proof=proof,
    )

    assert authority.is_current(request) is True
    assert len(calls) == 1
    assert authority.is_current(
        replace(
            request,
            evidence_passages=(
                {**request.evidence_passages[0], "blob_digest": DIGEST},
            ),
        )
    ) is False


def test_rights_authority_reuses_retained_access_for_sequential_rechecks(
    tmp_path,
) -> None:
    path = tmp_path / "authority.sqlite3"
    data = b"Alice"
    proof = extraction_proof()

    def counts() -> tuple[int, ...]:
        with sqlite3.connect(path) as connection:
            return tuple(
                int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
                for table in (
                    "authentication_contexts",
                    "authorization_requests",
                    "authorization_decisions",
                    "object_access_decisions",
                )
            )

    with open_object_system(path) as system:
        admission = admit(system, data=data).admission
        request = replace(
            _request(),
            evidence_passages=(
                {
                    "passage_id": "00000000-0000-4000-8000-000000007601",
                    "admission_id": str(admission.admission_id),
                    "purpose": "project.discovery",
                    "byte_offset": 0,
                    "byte_length": len(data),
                    "blob_digest": digest_bytes(data),
                    "allowed_use": "project.discovery",
                    "security_scope": "authority.protected",
                    "retention_scope": "source.short",
                },
            ),
        )
        authority = ExistingGovernedGraphitiRightsAuthority(
            objects=system.objects,
            proof=proof,
        )

        before = counts()
        assert authority.is_current(request) is True
        after_first = counts()
        assert after_first == tuple(value + 1 for value in before)

        assert authority.is_current(request) is True
        assert authority.is_current(request) is True
        assert counts() == after_first

    with open_object_system(path) as reopened:
        authority = ExistingGovernedGraphitiRightsAuthority(
            objects=reopened.objects,
            proof=proof,
        )
        assert authority.is_current(request) is True
        assert counts() == after_first


def test_conservative_entity_plan_allocates_separate_unknown_identity() -> None:
    request = _request()
    request = replace(
        request,
        evidence_passages=(
            {
                "passage_id": str(request.proposal.evidence[0].passage_id),
                "language": "en-GB",
            },
        ),
    )

    plan = conservative_entity_mention_plan(
        request,
        GraphitiProposalAdmissionAction.ADMIT,
        "graphiti-admit:proposal-key",
    )

    assert plan.mention_requests[0].entity_kind is EntityKind.UNKNOWN
    assert plan.proposal_request.kind is EntityResolutionProposalKind.MENTION_TO_NEW_ENTITY
    assert plan.proposal_request.candidate_entity_id is None
    assert plan.decision_request.action is EntityResolutionDecisionAction.ACCEPT
    assert plan.decision_request.accepted_entity_id is not None


def test_conservative_entity_plan_rejects_stale_rights_without_allocating_entity() -> None:
    request = _request()
    request = replace(
        request,
        evidence_passages=(
            {
                "passage_id": str(request.proposal.evidence[0].passage_id),
                "language": "en-GB",
            },
        ),
    )

    plan = conservative_entity_mention_plan(
        request,
        GraphitiProposalAdmissionAction.REJECT,
        "graphiti-admit:proposal-key",
    )

    assert plan.decision_request.action is EntityResolutionDecisionAction.REJECT
    assert plan.decision_request.accepted_entity_id is None
    assert plan.decision_request.accepted_entity_version_id is None
    assert plan.decision_request.alias_id is None


def test_entity_equivalence_reuses_exact_mentions_and_always_holds() -> None:
    first = _request().proposal
    second_evidence = EvidenceRange(
        passage_id=first.evidence[0].passage_id,
        start_byte=10,
        end_byte=13,
        evidence_text_digest=digest_canonical({"mention": "Bob"}),
    )
    endpoints = (
        first,
        replace(
            first,
            local_id="entity.0002",
            subject_placeholder="Bob",
            evidence=(second_evidence,),
        ),
    )
    proposal = ProposalDraft(
        local_id="equivalence.0001",
        kind=ExtractionProposalKind.ENTITY_EQUIVALENCE,
        subject_placeholder="Alice",
        object_placeholder="Bob",
        predicate_hint=None,
        confidence_basis_points=10_000,
        uncertainty_codes=("REQUIRES_EXPLICIT_RESOLUTION",),
        rationale_codes=("NAME_EQUALITY_ONLY",),
        evidence=(first.evidence[0], second_evidence),
    )
    cohort_seed = "entity-equivalence-plan"
    request = GraphitiAdmissionRequest(
        queue_seq=3,
        proposal_key="equivalence-proposal-key",
        source_receipt_digest=DIGEST,
        proposal_authority_binding=_binding(proposal, cohort_seed=cohort_seed),
        proposal=proposal,
        proposal_payload=proposal.canonical_value(),
        evidence_passages=(
            {
                "passage_id": str(first.evidence[0].passage_id),
                "language": "en-GB",
            },
        ),
        proposed_endpoints=("Alice", "Bob"),
        relation_statement=None,
        relation_temporal_bounds=None,
        source_lineage={"revision_id": "fixture"},
        relation_endpoint_bindings=tuple(
            _binding(item, cohort_seed=cohort_seed) for item in endpoints
        ),
    )

    plan = conservative_entity_mention_plan(
        request,
        GraphitiProposalAdmissionAction.HOLD,
        "graphiti-admit:equivalence-proposal-key",
    )

    assert plan.mention_requests == ()
    assert plan.proposal_request.kind is EntityResolutionProposalKind.MENTION_EQUIVALENCE
    assert plan.proposal_request.subject_mention_id == typed_id(
        EntityMentionId,
        "graphiti-v1-mention",
        request.relation_endpoint_bindings[0].proposal_envelope.canonical_digest,
    )
    assert plan.proposal_request.object_mention_id == typed_id(
        EntityMentionId,
        "graphiti-v1-mention",
        request.relation_endpoint_bindings[1].proposal_envelope.canonical_digest,
    )
    assert plan.decision_request.action is EntityResolutionDecisionAction.HOLD
    assert plan.decision_request.accepted_entity_id is None


class _RelationEndpointAuthorities:
    def __init__(
        self,
        request: GraphitiAdmissionRequest,
        *,
        second_action: EntityResolutionDecisionAction = (
            EntityResolutionDecisionAction.ACCEPT
        ),
    ) -> None:
        self.proposals: dict[EntityResolutionProposalId, object] = {}
        self.decisions: dict[EntityResolutionProposalId, object] = {}
        self.preferred_rows: dict[CanonicalEntityId, object] = {}
        self.version_rows: dict[CanonicalEntityVersionId, object] = {}
        self.dependency_calls: list[EntityResolutionDependencyRequest] = []
        for ordinal, binding in enumerate(
            request.relation_endpoint_bindings,
            start=1,
        ):
            envelope = binding.proposal_envelope
            proposal_id = typed_id(
                EntityResolutionProposalId,
                "graphiti-v1-resolution",
                envelope.canonical_digest,
            )
            proposal_version_id = typed_id(
                EntityResolutionProposalVersionId,
                "graphiti-v1-resolution-version",
                envelope.canonical_digest,
            )
            proposal_digest = digest_canonical(
                {"proposal": str(proposal_id), "ordinal": ordinal}
            )
            entity_id = typed_id(
                CanonicalEntityId,
                "graphiti-v1-entity",
                envelope.canonical_digest,
            )
            entity_version_id = typed_id(
                CanonicalEntityVersionId,
                "graphiti-v1-entity-version",
                envelope.canonical_digest,
            )
            action = (
                second_action
                if ordinal == 2
                else EntityResolutionDecisionAction.ACCEPT
            )
            self.proposals[proposal_id] = SimpleNamespace(
                proposal_id=proposal_id,
                proposal_version_id=proposal_version_id,
                source_proposal_id=envelope.proposal_id,
                source_proposal_digest=envelope.canonical_digest,
                canonical_digest=proposal_digest,
            )
            self.decisions[proposal_id] = SimpleNamespace(
                action=action,
                decision_id=typed_id(
                    EntityResolutionDecisionId,
                    "graphiti-v1-test-decision",
                    envelope.canonical_digest,
                ),
                proposal_id=proposal_id,
                proposal_version_id=proposal_version_id,
                proposal_digest=proposal_digest,
                accepted_entity_id=(
                    entity_id
                    if action is EntityResolutionDecisionAction.ACCEPT
                    else None
                ),
                accepted_entity_version_id=(
                    entity_version_id
                    if action is EntityResolutionDecisionAction.ACCEPT
                    else None
                ),
            )
            self.preferred_rows[entity_id] = SimpleNamespace(
                entity_id=entity_id,
                preferred_entity_id=entity_id,
                current_entity_version_id=entity_version_id,
                lifecycle=CanonicalEntityLifecycle.ACTIVE,
            )
            self.version_rows[entity_version_id] = SimpleNamespace(
                entity_id=entity_id,
                entity_version_id=entity_version_id,
                lifecycle=CanonicalEntityLifecycle.ACTIVE,
            )

    def proposal(self, proposal_id, *, proof):
        return self.proposals[proposal_id]

    def decision(self, proposal_id, *, proof):
        return self.decisions[proposal_id]

    def preferred(self, entity_id, *, proof):
        return self.preferred_rows[entity_id]

    def entity_version(self, version_id, *, proof):
        return self.version_rows[version_id]

    def bind_resolution_dependency(self, dependency, *, proof):
        self.dependency_calls.append(dependency)
        ordinal = len(self.dependency_calls)
        return SimpleNamespace(
            dependency_id=dependency.dependency_id,
            dependent_proposal_id=dependency.dependent_proposal_id,
            dependent_proposal_digest=dependency.expected_dependent_proposal_digest,
            resolution_proposal_id=dependency.resolution_proposal_id,
            proposal_version_id=dependency.expected_resolution_proposal_version_id,
            proposal_version_digest=dependency.expected_resolution_proposal_digest,
            material=dependency.material,
            authority_event_id=typed_id(
                EventId,
                "graphiti-v1-test-dependency-event",
                str(dependency.dependency_id),
            ),
            authority_ledger_seq=100 + ordinal,
            canonical_digest=digest_canonical(dependency.canonical_value()),
        )


class _RelationAuthorities:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def propose(self, proposal, *, proof):
        self.calls.append("propose")
        return SimpleNamespace(
            proposal_id=proposal.proposal_id,
            proposal_version_id=proposal.proposal_version_id,
            canonical_digest=proposal.canonical_digest,
        )

    def decide(self, decision, *, proof):
        self.calls.append("decide")
        return SimpleNamespace(
            action=decision.action,
            decision_id=decision.decision_id,
            authority_ledger_seq=199,
            reason_code=decision.reason_code,
            canonical_digest=DIGEST,
            assertion_id=decision.assertion_id,
        )


@pytest.mark.parametrize(
    ("predicate_hint", "second_action", "expected_reason"),
    (
        (
            ProposalPredicateHint.SUPERSEDES,
            EntityResolutionDecisionAction.ACCEPT,
            "RELATION_PREDICATE_UNSUPPORTED_FOR_CANONICAL_ENTITY_ENDPOINTS",
        ),
        (
            ProposalPredicateHint.SAME_PROCESS_AS,
            EntityResolutionDecisionAction.HOLD,
            "RELATION_ENDPOINT_RESOLUTION_NOT_ACCEPTED_OR_CURRENT",
        ),
    ),
)
def test_relation_without_admissible_predicate_or_endpoints_is_operational_hold(
    predicate_hint: ProposalPredicateHint,
    second_action: EntityResolutionDecisionAction,
    expected_reason: str,
) -> None:
    request = _relation_request(predicate_hint)
    proof = AuthenticationProof(method="STATIC_TOKEN", credential="fixture")
    entities = _RelationEndpointAuthorities(
        request,
        second_action=second_action,
    )
    relations = _RelationAuthorities()
    builder = ConservativeGraphitiRelationPlanBuilder(
        entities=entities,  # type: ignore[arg-type]
        proof=proof,
    )
    plan = builder(
        request,
        GraphitiProposalAdmissionAction.HOLD,
        "graphiti-admit:relation-proposal-key",
    )
    authority = ExistingGovernedGraphitiAdmissionAuthority(
        entities=entities,  # type: ignore[arg-type]
        relations=relations,  # type: ignore[arg-type]
        proof=proof,
        entity_plan=lambda *_: pytest.fail("entity planner called"),
        relation_plan=builder,
    )

    decision = authority.decide_relation_admission(
        request,
        required_action=GraphitiProposalAdmissionAction.HOLD,
        idempotency_key="graphiti-admit:relation-proposal-key",
    )

    assert isinstance(plan, GraphitiRelationOperationalDecisionPlan)
    assert all(not item.material for item in plan.dependency_requests)
    assert decision.action is GraphitiProposalAdmissionAction.HOLD
    assert decision.reason_code == expected_reason
    assert len(decision.relation_hold_basis) == 2
    assert relations.calls == []
    assert entities.dependency_calls == list(plan.dependency_requests)


def test_relation_with_missing_temporal_field_is_real_4c_hold() -> None:
    request = replace(
        _relation_request(),
        relation_temporal_bounds={"valid_at": "2026-08-24T00:00:00Z"},
    )
    proof = AuthenticationProof(method="STATIC_TOKEN", credential="fixture")
    entities = _RelationEndpointAuthorities(request)
    relations = _RelationAuthorities()
    builder = ConservativeGraphitiRelationPlanBuilder(
        entities=entities,  # type: ignore[arg-type]
        proof=proof,
    )
    plan = builder(
        request,
        GraphitiProposalAdmissionAction.HOLD,
        "graphiti-admit:relation-proposal-key",
    )
    authority = ExistingGovernedGraphitiAdmissionAuthority(
        entities=entities,  # type: ignore[arg-type]
        relations=relations,  # type: ignore[arg-type]
        proof=proof,
        entity_plan=lambda *_: pytest.fail("entity planner called"),
        relation_plan=builder,
    )

    decision = authority.decide_relation_admission(
        request,
        required_action=GraphitiProposalAdmissionAction.HOLD,
        idempotency_key="graphiti-admit:relation-proposal-key",
    )

    assert isinstance(plan, GraphitiRelationAdmissionPlan)
    assert plan.decision_request.action is EditorialRelationDecisionAction.HOLD
    assert decision.action is GraphitiProposalAdmissionAction.HOLD
    assert decision.relation_hold_basis == ()
    assert relations.calls == ["propose", "decide"]


def test_conservative_relation_plan_binds_two_exact_4b_dependencies() -> None:
    request = _relation_request()
    proof = AuthenticationProof(method="STATIC_TOKEN", credential="fixture")
    proposals: dict[EntityResolutionProposalId, object] = {}
    decisions: dict[EntityResolutionProposalId, object] = {}
    preferred: dict[CanonicalEntityId, object] = {}
    versions: dict[CanonicalEntityVersionId, object] = {}
    for ordinal, binding in enumerate(request.relation_endpoint_bindings, start=1):
        envelope = binding.proposal_envelope
        proposal_id = typed_id(
            EntityResolutionProposalId,
            "graphiti-v1-resolution",
            envelope.canonical_digest,
        )
        proposal_version_id = typed_id(
            EntityResolutionProposalVersionId,
            "graphiti-v1-resolution-version",
            envelope.canonical_digest,
        )
        entity_id = typed_id(
            CanonicalEntityId,
            "graphiti-v1-entity",
            envelope.canonical_digest,
        )
        entity_version_id = typed_id(
            CanonicalEntityVersionId,
            "graphiti-v1-entity-version",
            envelope.canonical_digest,
        )
        proposal_digest = digest_canonical(
            {"proposal": str(proposal_id), "ordinal": ordinal}
        )
        proposals[proposal_id] = SimpleNamespace(
            proposal_id=proposal_id,
            proposal_version_id=proposal_version_id,
            source_proposal_id=envelope.proposal_id,
            source_proposal_digest=envelope.canonical_digest,
            canonical_digest=proposal_digest,
        )
        decisions[proposal_id] = SimpleNamespace(
            action=EntityResolutionDecisionAction.ACCEPT,
            decision_id=typed_id(
                EntityResolutionDecisionId,
                "graphiti-v1-test-decision",
                envelope.canonical_digest,
            ),
            proposal_id=proposal_id,
            proposal_version_id=proposal_version_id,
            proposal_digest=proposal_digest,
            accepted_entity_id=entity_id,
            accepted_entity_version_id=entity_version_id,
        )
        preferred[entity_id] = SimpleNamespace(
            entity_id=entity_id,
            preferred_entity_id=entity_id,
            current_entity_version_id=entity_version_id,
            lifecycle=CanonicalEntityLifecycle.ACTIVE,
        )
        versions[entity_version_id] = SimpleNamespace(
            entity_id=entity_id,
            entity_version_id=entity_version_id,
            lifecycle=CanonicalEntityLifecycle.ACTIVE,
        )

    dependency_calls: list[EntityResolutionDependencyRequest] = []

    class Entities:
        def proposal(self, proposal_id, *, proof):
            return proposals[proposal_id]

        def decision(self, proposal_id, *, proof):
            return decisions[proposal_id]

        def preferred(self, entity_id, *, proof):
            return preferred[entity_id]

        def entity_version(self, version_id, *, proof):
            return versions[version_id]

        def bind_resolution_dependency(self, dependency, *, proof):
            dependency_calls.append(dependency)
            return SimpleNamespace(
                dependency_id=dependency.dependency_id,
                dependent_proposal_id=dependency.dependent_proposal_id,
                dependent_proposal_digest=(
                    dependency.expected_dependent_proposal_digest
                ),
                resolution_proposal_id=dependency.resolution_proposal_id,
                proposal_version_id=(
                    dependency.expected_resolution_proposal_version_id
                ),
                proposal_version_digest=(
                    dependency.expected_resolution_proposal_digest
                ),
                material=dependency.material,
            )

    class Relations:
        def propose(self, proposal, *, proof):
            return SimpleNamespace(
                proposal_id=proposal.proposal_id,
                proposal_version_id=proposal.proposal_version_id,
                canonical_digest=proposal.canonical_digest,
            )

        def decide(self, decision, *, proof):
            return SimpleNamespace(
                action=decision.action,
                decision_id=decision.decision_id,
                authority_ledger_seq=99,
                reason_code=decision.reason_code,
                canonical_digest=DIGEST,
                assertion_id=decision.assertion_id,
            )

    entities = Entities()
    builder = ConservativeGraphitiRelationPlanBuilder(
        entities=entities,  # type: ignore[arg-type]
        proof=proof,
    )
    plan = builder(
        request,
        GraphitiProposalAdmissionAction.ADMIT,
        "graphiti-admit:relation-proposal-key",
    )
    authority = ExistingGovernedGraphitiAdmissionAuthority(
        entities=entities,  # type: ignore[arg-type]
        relations=Relations(),  # type: ignore[arg-type]
        proof=proof,
        entity_plan=lambda *_: pytest.fail("entity planner called"),
        relation_plan=builder,
    )

    decision = authority.decide_relation_admission(
        request,
        required_action=GraphitiProposalAdmissionAction.ADMIT,
        idempotency_key="graphiti-admit:relation-proposal-key",
    )

    assert plan.proposal_request.predicate is EditorialPredicateCode.SAME_PROCESS_AS
    assert plan.decision_request.action is EditorialRelationDecisionAction.ACCEPT
    assert tuple(
        item.dependent_proposal_id for item in plan.dependency_requests
    ) == (request.proposal_authority_binding.proposal_envelope.proposal_id,) * 2
    assert dependency_calls == list(plan.dependency_requests)
    assert decision.action is GraphitiProposalAdmissionAction.ADMIT
    assert decision.admitted_authority_id == str(plan.decision_request.assertion_id)


def test_existing_runtime_composer_uses_only_existing_components() -> None:
    component = SimpleNamespace()
    consumer = compose_existing_graphiti_admission_consumer(
        SimpleNamespace(),  # type: ignore[arg-type]
        adapter=component,  # type: ignore[arg-type]
        extraction=component,  # type: ignore[arg-type]
        objects=component,  # type: ignore[arg-type]
        entities=component,  # type: ignore[arg-type]
        relations=component,  # type: ignore[arg-type]
        increment4=component,  # type: ignore[arg-type]
        proof=AuthenticationProof(method="STATIC_TOKEN", credential="fixture"),
    )

    assert consumer.__class__.__name__ == "GraphitiAdmissionConsumer"
