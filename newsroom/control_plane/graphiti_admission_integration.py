"""Concrete governed authority integration for Graphiti admission work.

The corpus receipt deliberately does not manufacture authority identities.  A
planner must bind it to proposal records already retained by the extraction
authority; this adapter then executes the existing authenticated entity and
editorial-relation commands and translates only their retained receipts.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from newsroom.authority.auth import AuthenticationProof
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.authority.editorial_relation_system import GovernedEditorialRelations
from newsroom.authority.entity_system import GovernedEntityRecords
from newsroom.authority.extraction_facade import GovernedExtractionRecords
from newsroom.authority.graphiti_adapter_facade import GovernedGraphitiProposalAdapter
from newsroom.authority.object_system import GovernedObjects
from newsroom.authority.objects import HydrationRequest
from newsroom.authority.types import ObjectAdmissionId, UtcTimestamp
from newsroom.control_plane.governed_context import (
    AuthorityContextBinding,
    GovernedAuthorityContext,
)
from newsroom.control_plane.graphiti_admission import (
    GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION,
    GRAPHITI_PROJECTION_GENERATION_SCHEMA_VERSION,
    GraphitiAdmissionConsumer,
    GraphitiAdmissionConsumerError,
    GraphitiAdmissionRequest,
    GraphitiGovernedDecision,
    GraphitiProposalAuthorityBinding,
    GraphitiRelationHoldBasis,
    GraphitiProjectionGenerationResult,
    GraphitiProjectionReceipt,
    GraphitiProjectionReconciliationReceipt,
    GraphitiProjectionRequest,
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
    CanonicalEntityVersionId,
    CanonicalEntityLifecycle,
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
    classify_entity_script,
    normalize_entity_text,
)
from newsroom.extraction.types import ExtractionProposalKind, ProposalPredicateHint
from newsroom.graphiti_adapter.admission import GraphitiProposalAdmissionAction
from newsroom.graphiti_adapter.identity import attempt_ids, typed_id
from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome
from newsroom.increment4.neo4j import (
    Increment4Neo4jController,
    Increment4Neo4jCurrentBuildRequest,
)
from newsroom.projection.models import (
    ProjectionGenerationId,
    ProjectionGenerationState,
)
from newsroom.relations.editorial_models import (
    EDITORIAL_PREDICATE_REGISTRY_V1,
    EDITORIAL_RELATION_ADMISSION_POLICY_VERSION,
    CanonicalEntityRelationEndpoint,
    EditorialRelationProducer,
    EditorialRelationDecisionRequest,
    EditorialRelationProposalRequest,
    EditorialRelationTemporalScope,
    ExtractionRelationEvidence,
    endpoint_canonical_value,
)
from newsroom.relations.editorial_types import (
    EditorialRelationAssertionId,
    EditorialRelationAssertionLifecycle,
    EditorialRelationDecisionAction,
    EditorialRelationDecisionId,
    EditorialPredicateCode,
    EditorialRelationProducerKind,
    EditorialRelationProposalId,
    EditorialRelationProposalVersionId,
)


class ExistingGovernedGraphitiProposalAuthority:
    """Bind a terminal raw proposal only when exact retained 4D/4A authority exists."""

    def __init__(
        self,
        *,
        adapter: GovernedGraphitiProposalAdapter,
        extraction: GovernedExtractionRecords,
        proof: AuthenticationProof,
    ) -> None:
        self._adapter = adapter
        self._extraction = extraction
        self._proof = proof

    def bind_proposal(
        self,
        *,
        ingest_id: str,
        terminal_receipt: Mapping[str, object],
        proposal: object,
    ) -> GraphitiProposalAuthorityBinding | None:
        attempt_number = terminal_receipt.get("attempt_number")
        if (
            isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number <= 0
        ):
            return None
        attempt_id = attempt_ids(ingest_id, attempt_number)[0]
        try:
            attempt = self._adapter.attempt(
                attempt_id, proof=self._proof
            )
            manifest = self._adapter.manifest_for_attempt(
                attempt.attempt_id, proof=self._proof
            )
            if (
                attempt.attempt_id != attempt_id
                or attempt.attempt_number != attempt_number
                or attempt.outcome is not GraphitiAdapterOutcome.COMPLETE
                or not attempt.outcome.terminal
                or terminal_receipt.get("outcome") != attempt.outcome.value
                or attempt.output_id is None
                or attempt.proposal_set_id is None
                or attempt.manifest_id != manifest.manifest_id
                or attempt.run_id != manifest.run_id
                or attempt.run_version_id != manifest.requested_run_version_id
            ):
                return None
            metadata = self._extraction.metadata(
                attempt.run_version_id, proof=self._proof
            )
            if (
                not metadata.terminal
                or metadata.run_id != attempt.run_id
                or metadata.output is None
                or metadata.output.output_id != attempt.output_id
            ):
                return None
            raw = self._extraction.raw_output(attempt.output_id, proof=self._proof)
            terminal_raw_digest = terminal_receipt.get("raw_output_digest")
            raw_value = json.loads(raw.canonical_bytes)
            if (
                raw.view != metadata.output
                or terminal_receipt.get("ingest_id") != ingest_id
                or not isinstance(terminal_raw_digest, str)
                or not isinstance(raw_value, dict)
                or canonical_json_bytes(raw_value) != raw.canonical_bytes
                or digest_bytes(raw.canonical_bytes) != raw.view.canonical_digest
            ):
                return None
            retained_raw_digest = raw_value.pop("raw_output_digest", None)
            if (
                retained_raw_digest != terminal_raw_digest
                or retained_raw_digest
                != digest_bytes(canonical_json_bytes(raw_value))
            ):
                return None
            exact_raw_fields = (
                "attempt_number",
                "provider_attempt_number",
                "generation_id",
                "episode_uuid",
                "temporal_basis",
                "reference_time",
                "proposals",
                "passages",
                "entities",
                "relations",
                "proposal_count",
                "entity_count",
                "relation_count",
            )
            if any(
                terminal_receipt.get(field) != raw_value.get(field)
                for field in exact_raw_fields
            ):
                return None
            envelopes = self._extraction.proposals(
                attempt.run_version_id, proof=self._proof
            )
        except Exception:  # noqa: BLE001 - any authority read fault fails closed
            return None
        matches = tuple(
            envelope
            for envelope in envelopes
            if envelope.proposal_set_id == attempt.proposal_set_id
            and envelope.output_id == attempt.output_id
            and envelope.run_id == attempt.run_id
            and envelope.run_version_id == attempt.run_version_id
            and envelope.local_id == getattr(proposal, "local_id", None)
        )
        if len(matches) != 1:
            return None
        envelope = matches[0]
        if any(
            getattr(envelope, field) != getattr(proposal, field, object())
            for field in (
                "kind",
                "subject_placeholder",
                "object_placeholder",
                "predicate_hint",
                "confidence_basis_points",
                "uncertainty_codes",
                "rationale_codes",
                "evidence",
            )
        ):
            return None
        manifest_passages = {item.passage_id: item for item in manifest.passages}
        if any(
            evidence.passage_id not in manifest_passages
            for evidence in envelope.evidence
        ):
            return None
        return GraphitiProposalAuthorityBinding(
            graphiti_attempt_id=str(attempt.attempt_id),
            graphiti_attempt_authority_event_id=str(attempt.authority_event_id),
            proposal_envelope=envelope,
        )


class ExistingGovernedGraphitiRightsAuthority:
    """Re-read every exact source object through existing governed rights."""

    def __init__(
        self,
        *,
        objects: GovernedObjects,
        proof: AuthenticationProof,
    ) -> None:
        self._objects = objects
        self._proof = proof

    def is_current(self, request: GraphitiAdmissionRequest) -> bool:
        try:
            for raw in request.evidence_passages:
                required = {
                    "admission_id",
                    "purpose",
                    "byte_offset",
                    "byte_length",
                    "blob_digest",
                    "allowed_use",
                    "security_scope",
                    "retention_scope",
                }
                if not required.issubset(raw):
                    return False
                offset = raw["byte_offset"]
                length = raw["byte_length"]
                if (
                    isinstance(offset, bool)
                    or not isinstance(offset, int)
                    or offset != 0
                    or isinstance(length, bool)
                    or not isinstance(length, int)
                    or length <= 0
                ):
                    return False
                hydrated = self._objects.rehydrate(
                    HydrationRequest(
                        admission_id=ObjectAdmissionId.parse(
                            str(raw["admission_id"])
                        ),
                        purpose=str(raw["purpose"]),
                        offset=offset,
                        length=length,
                    ),
                    proof=self._proof,
                )
                decision = hydrated.decision
                if (
                    len(hydrated.data) != length
                    or digest_bytes(hydrated.data) != str(raw["blob_digest"])
                    or str(decision.admission_id) != str(raw["admission_id"])
                    or decision.offset != offset
                    or decision.allowed_bytes < length
                    or decision.purpose != raw["purpose"]
                    or decision.allowed_use != raw["allowed_use"]
                    or decision.security_scope != raw["security_scope"]
                    or decision.retention_scope != raw["retention_scope"]
                ):
                    return False
        except Exception:  # noqa: BLE001 - any governed rights fault is HOLD
            return False
        return bool(request.evidence_passages)


class ExistingIncrement4GenerationProjector:
    """Reduce Graphiti admission to the existing full Increment 4 controller."""

    def __init__(
        self,
        *,
        controller: Increment4Neo4jController,
        proof: AuthenticationProof,
    ) -> None:
        self._controller = controller
        self._proof = proof

    def build_and_promote_increment4_cohort(
        self,
        requests: tuple[GraphitiProjectionRequest, ...],
        *,
        cohort_digest: str,
        generation_id: str,
        idempotency_key: str,
    ) -> GraphitiProjectionGenerationResult:
        if not requests:
            raise GraphitiAdmissionConsumerError(
                "Increment 4 generation requires one exact decided cohort"
            )
        admitted_ids = tuple(
            sorted(
                str(item.decision.admitted_authority_id)
                for item in requests
                if item.decision.action is GraphitiProposalAdmissionAction.ADMIT
            )
        )
        if len(admitted_ids) != len(set(admitted_ids)):
            raise GraphitiAdmissionConsumerError(
                "Increment 4 generation needs unique admitted authority identities"
            )
        required_watermark = max(
            item.decision.authority_ledger_seq for item in requests
        )
        result = self._controller.build_current_and_promote(
            Increment4Neo4jCurrentBuildRequest(
                generation_id=ProjectionGenerationId.parse(generation_id),
                reason_code="GRAPHITI_ADMISSION_COHORT",
                idempotency_key=idempotency_key,
                purge_retired_generation=True,
                allow_active_extension=True,
            ),
            proof=self._proof,
        )
        source_bound = (
            result.validation.source_snapshot_digest,
            result.validation.source_watermark_ledger_seq,
            result.validation.source_request_digest,
        )
        if any(item is not None for item in source_bound) and any(
            item is None for item in source_bound
        ):
            raise GraphitiAdmissionConsumerError(
                "Increment 4 generation result has partial source authority"
            )
        extension = all(item is not None for item in source_bound)
        expected_source_request_digest = digest_canonical(
            {
                "generation_id": generation_id,
                "reason_code": "GRAPHITI_ADMISSION_COHORT",
                "idempotency_key": idempotency_key,
                "purge_retired_generation": True,
                "allow_active_extension": True,
            }
        )
        if (
            (
                not extension
                and result.generation.generation_id
                != ProjectionGenerationId.parse(generation_id)
            )
            or result.generation.state is not ProjectionGenerationState.ACTIVE
            or result.source_watermark_ledger_seq < required_watermark
            or result.checkpoint_ledger_seq < result.source_watermark_ledger_seq
            or result.validation.projection_state_digest
            != result.projection_state_digest
            or (
                not extension
                and result.validation.validation_digest
                != result.promotion.validation_digest
            )
            or (
                extension
                and (
                    result.validation.source_snapshot_digest
                    != result.source_snapshot_digest
                    or result.validation.source_watermark_ledger_seq
                    != result.source_watermark_ledger_seq
                    or result.validation.source_request_digest
                    != expected_source_request_digest
                )
            )
        ):
            raise GraphitiAdmissionConsumerError(
                "Increment 4 full generation does not bind the exact cohort cutoff"
            )
        return GraphitiProjectionGenerationResult(
            cohort_digest=cohort_digest,
            generation_id=str(result.generation.generation_id),
            source_snapshot_digest=result.source_snapshot_digest,
            authority_watermark=result.source_watermark_ledger_seq,
            validation_digest=result.validation.validation_digest,
            promotion_digest=result.promotion.promotion_digest,
            reconciliation_digest=result.projection_state_digest,
            admitted_authority_ids=admitted_ids,
            schema_version=(
                GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION
                if extension
                else GRAPHITI_PROJECTION_GENERATION_SCHEMA_VERSION
            ),
            source_request_digest=(
                None
                if not extension
                else result.validation.source_request_digest
            ),
        )

    @staticmethod
    def recover_increment4_admitted_receipt(
        *, idempotency_key: str
    ) -> GraphitiProjectionReceipt | None:
        del idempotency_key
        return None

    @staticmethod
    def deliver_increment4_admitted(
        request: GraphitiProjectionRequest,
        *,
        idempotency_key: str,
    ) -> GraphitiProjectionReceipt:
        del request, idempotency_key
        raise GraphitiAdmissionConsumerError(
            "generation projector rejects legacy per-proposal delivery"
        )

    @staticmethod
    def tombstone_increment4_admitted(
        request: GraphitiProjectionRequest,
        *,
        idempotency_key: str,
    ) -> GraphitiProjectionReceipt:
        del request, idempotency_key
        raise GraphitiAdmissionConsumerError(
            "generation projector rejects legacy per-proposal tombstones"
        )

    @staticmethod
    def reconcile_increment4_admitted(
        expected: tuple[GraphitiProjectionReceipt, ...],
        *,
        generation_id: str,
    ) -> GraphitiProjectionReconciliationReceipt:
        del expected, generation_id
        raise GraphitiAdmissionConsumerError(
            "generation projector reconciles only complete Increment 4 snapshots"
        )


def conservative_entity_mention_plan(
    request: GraphitiAdmissionRequest,
    required_action: GraphitiProposalAdmissionAction | None,
    idempotency_key: str,
) -> GraphitiEntityAdmissionPlan:
    """Allocate a separate UNKNOWN entity; never infer an existing identity."""

    if request.proposal.kind not in {
        ExtractionProposalKind.ENTITY_MENTION,
        ExtractionProposalKind.ENTITY_EQUIVALENCE,
    }:
        raise GraphitiAdmissionConsumerError("unsupported entity proposal kind")
    mention_is_admissible = request.proposal.kind is ExtractionProposalKind.ENTITY_MENTION
    planned_action = (
        GraphitiProposalAdmissionAction.ADMIT
        if mention_is_admissible
        else GraphitiProposalAdmissionAction.HOLD
    )
    action = (
        GraphitiProposalAdmissionAction.REJECT
        if required_action is GraphitiProposalAdmissionAction.REJECT
        else planned_action
    )
    if required_action not in (None, action):
        raise GraphitiAdmissionConsumerError("entity mention action differs")
    envelope = request.proposal_authority_binding.proposal_envelope
    seed = envelope.canonical_digest
    endpoint_envelopes = tuple(
        item.proposal_envelope for item in request.relation_endpoint_bindings
    )
    if mention_is_admissible:
        mention_ids = (typed_id(EntityMentionId, "graphiti-v1-mention", seed),)
    else:
        if len(endpoint_envelopes) != 2:
            raise GraphitiAdmissionConsumerError(
                "entity equivalence lacks exact mention authorities"
            )
        mention_ids = tuple(
            typed_id(
                EntityMentionId,
                "graphiti-v1-mention",
                item.canonical_digest,
            )
            for item in endpoint_envelopes
        )
    proposal_id = typed_id(EntityResolutionProposalId, "graphiti-v1-resolution", seed)
    version_id = typed_id(
        EntityResolutionProposalVersionId, "graphiti-v1-resolution-version", seed
    )
    proposal_request = EntityResolutionProposalRequest(
        proposal_id=proposal_id,
        proposal_version_id=version_id,
        version_number=1,
        expected_previous_version_id=None,
        source_proposal_id=envelope.proposal_id,
        expected_source_proposal_digest=envelope.canonical_digest,
        kind=(
            EntityResolutionProposalKind.MENTION_TO_NEW_ENTITY
            if mention_is_admissible
            else EntityResolutionProposalKind.MENTION_EQUIVALENCE
        ),
        subject_mention_id=mention_ids[0],
        object_mention_id=None if mention_is_admissible else mention_ids[1],
        candidate_entity_id=None,
        candidate_entity_version_id=None,
        confidence_basis_points=envelope.confidence_basis_points,
        uncertainty_codes=envelope.uncertainty_codes,
        basis_codes=tuple(sorted(set((*envelope.rationale_codes, "EXACT_PROPOSAL_ENVELOPE")))),
        idempotency_key=f"{idempotency_key}:proposal",
    )
    mention_requests: tuple[EntityMentionAdmissionRequest, ...] = ()
    if mention_is_admissible:
        language_by_passage = {
            str(item["passage_id"]): item.get("language")
            for item in request.evidence_passages
            if "passage_id" in item
        }
        languages = {
            language_by_passage.get(str(item.passage_id)) for item in envelope.evidence
        }
        if len(languages) != 1 or not isinstance(next(iter(languages)), str):
            raise GraphitiAdmissionConsumerError("entity mention language is ambiguous")
        language = next(iter(languages))
        mention_requests = (
            EntityMentionAdmissionRequest(
                mention_id=mention_ids[0],
                source_proposal_id=envelope.proposal_id,
                expected_source_proposal_digest=envelope.canonical_digest,
                entity_kind=EntityKind.UNKNOWN,
                language=language,
                script=classify_entity_script(envelope.subject_placeholder),
                normalized_text=normalize_entity_text(envelope.subject_placeholder),
                normalization_contract_digest=ENTITY_NORMALISATION_CONTRACT_DIGEST,
                idempotency_key=f"{idempotency_key}:mention",
            ),
        )
    return GraphitiEntityAdmissionPlan(
        graphiti_proposal_digest=request.proposal.digest,
        graphiti_proposal_local_id=request.proposal.local_id,
        mention_requests=mention_requests,
        proposal_request=proposal_request,
        decision_request=EntityResolutionDecisionRequest(
            proposal_id=proposal_id,
            expected_proposal_version_id=version_id,
            expected_proposal_digest=digest_canonical(proposal_request.canonical_value()),
            action={
                GraphitiProposalAdmissionAction.ADMIT: (
                    EntityResolutionDecisionAction.ACCEPT
                ),
                GraphitiProposalAdmissionAction.REJECT: (
                    EntityResolutionDecisionAction.REJECT
                ),
                GraphitiProposalAdmissionAction.HOLD: EntityResolutionDecisionAction.HOLD,
            }[action],
            expected_decision_version=0,
            expected_previous_decision_id=None,
            accepted_entity_id=(
                typed_id(CanonicalEntityId, "graphiti-v1-entity", seed)
                if action is GraphitiProposalAdmissionAction.ADMIT else None
            ),
            accepted_entity_version_id=(
                typed_id(CanonicalEntityVersionId, "graphiti-v1-entity-version", seed)
                if action is GraphitiProposalAdmissionAction.ADMIT else None
            ),
            alias_id=(typed_id(EntityAliasId, "graphiti-v1-alias", seed)
                      if action is GraphitiProposalAdmissionAction.ADMIT else None),
            alias_kind=(
                EntityAliasKind.PRIMARY_NAME
                if action is GraphitiProposalAdmissionAction.ADMIT
                else None
            ),
            reason_code={
                GraphitiProposalAdmissionAction.ADMIT: (
                    "EXACT_MENTION_TO_NEW_UNKNOWN_ENTITY"
                ),
                GraphitiProposalAdmissionAction.REJECT: "CURRENT_RIGHTS_REJECT",
                GraphitiProposalAdmissionAction.HOLD: "AMBIGUOUS_IDENTITY_HOLD",
            }[action],
            decision_policy_version="graphiti-conservative-admission-v1",
            idempotency_key=f"{idempotency_key}:decision",
        ),
    )


@dataclass(frozen=True, slots=True)
class GraphitiEntityAdmissionPlan:
    """Typed commands bound to one exact retained Graphiti proposal."""

    graphiti_proposal_digest: str
    graphiti_proposal_local_id: str
    mention_requests: tuple[EntityMentionAdmissionRequest, ...]
    proposal_request: EntityResolutionProposalRequest
    decision_request: EntityResolutionDecisionRequest

    def __post_init__(self) -> None:
        if (
            not self.mention_requests
            and self.proposal_request.kind
            is not EntityResolutionProposalKind.MENTION_EQUIVALENCE
        ):
            raise GraphitiAdmissionConsumerError(
                "entity plan requires governed mention admission"
            )
        if (
            self.mention_requests
            and self.proposal_request.kind
            is EntityResolutionProposalKind.MENTION_EQUIVALENCE
        ):
            raise GraphitiAdmissionConsumerError(
                "entity equivalence must reuse exact governed mentions"
            )
        if any(
            not isinstance(item, EntityMentionAdmissionRequest)
            for item in self.mention_requests
        ):
            raise GraphitiAdmissionConsumerError(
                "entity plan mention commands must be typed"
            )


@dataclass(frozen=True, slots=True)
class GraphitiRelationAdmissionPlan:
    """Typed relation commands plus the two authority endpoint bindings."""

    graphiti_proposal_digest: str
    graphiti_proposal_local_id: str
    proposal_request: EditorialRelationProposalRequest
    decision_request: EditorialRelationDecisionRequest
    dependency_requests: tuple[EntityResolutionDependencyRequest, ...]
    endpoint_resolution_proposal_ids: tuple[EntityResolutionProposalId, ...]
    resolved_endpoint_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            len(self.dependency_requests) != 2
            or any(
                not isinstance(item, EntityResolutionDependencyRequest)
                for item in self.dependency_requests
            )
            or any(not item.material for item in self.dependency_requests)
            or tuple(
                sorted(
                    (item.dependency_id for item in self.dependency_requests),
                    key=str,
                )
            )
            != tuple(item.dependency_id for item in self.dependency_requests)
        ):
            raise GraphitiAdmissionConsumerError(
                "relation plan requires two sorted material entity dependencies"
            )


@dataclass(frozen=True, slots=True)
class GraphitiRelationOperationalDecisionPlan:
    """Non-authoritative HOLD/REJECT backed by exact non-material 4B links."""

    graphiti_proposal_digest: str
    graphiti_proposal_local_id: str
    action: GraphitiProposalAdmissionAction
    reason_code: str
    dependency_requests: tuple[EntityResolutionDependencyRequest, ...]
    endpoint_resolution_proposal_ids: tuple[EntityResolutionProposalId, ...]
    resolved_endpoint_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.action not in {
            GraphitiProposalAdmissionAction.HOLD,
            GraphitiProposalAdmissionAction.REJECT,
        }:
            raise GraphitiAdmissionConsumerError(
                "operational relation decision must be HOLD or REJECT"
            )
        if not self.reason_code:
            raise GraphitiAdmissionConsumerError(
                "operational relation decision needs an exact reason"
            )
        if (
            len(self.dependency_requests) != 2
            or any(item.material for item in self.dependency_requests)
            or tuple(
                sorted(
                    (item.dependency_id for item in self.dependency_requests),
                    key=str,
                )
            )
            != tuple(item.dependency_id for item in self.dependency_requests)
            or len(self.endpoint_resolution_proposal_ids) != 2
            or len(set(self.endpoint_resolution_proposal_ids)) != 2
            or len(self.resolved_endpoint_names) != 2
        ):
            raise GraphitiAdmissionConsumerError(
                "operational relation decision needs two exact non-material 4B links"
            )


class ConservativeGraphitiRelationPlanBuilder:
    """Bind one supported relation to exact current 4B endpoint authority."""

    def __init__(
        self,
        *,
        entities: GovernedEntityRecords,
        proof: AuthenticationProof,
    ) -> None:
        self._entities = entities
        self._proof = proof

    @staticmethod
    def _temporal_scope(
        request: GraphitiAdmissionRequest,
    ) -> tuple[EditorialRelationTemporalScope, bool]:
        try:
            observed_at = UtcTimestamp.parse(
                str(request.source_lineage["reference_time"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GraphitiAdmissionConsumerError(
                "relation lacks exact observed-at source authority"
            ) from exc
        raw = request.relation_temporal_bounds
        assert raw is not None
        try:
            valid_from = (
                None
                if raw["valid_at"] is None
                else UtcTimestamp.parse(str(raw["valid_at"]))
            )
            end_values = tuple(
                UtcTimestamp.parse(str(raw[field]))
                for field in ("invalid_at", "expired_at")
                if raw[field] is not None
            )
        except (KeyError, TypeError, ValueError):
            return (
                EditorialRelationTemporalScope(
                    valid_from=None,
                    valid_until=None,
                    observed_at=observed_at,
                ),
                False,
            )
        valid_until = None
        if end_values:
            if len({item.to_text() for item in end_values}) != 1:
                return (
                    EditorialRelationTemporalScope(
                        valid_from=None,
                        valid_until=None,
                        observed_at=observed_at,
                    ),
                    False,
                )
            valid_until = end_values[0]
        if valid_from is None or (
            valid_until is not None and valid_until.value <= valid_from.value
        ):
            return (
                EditorialRelationTemporalScope(
                    valid_from=None,
                    valid_until=None,
                    observed_at=observed_at,
                ),
                False,
            )
        return (
            EditorialRelationTemporalScope(
                valid_from=valid_from,
                valid_until=valid_until,
                observed_at=observed_at,
            ),
            True,
        )

    def __call__(
        self,
        request: GraphitiAdmissionRequest,
        required_action: GraphitiProposalAdmissionAction | None,
        idempotency_key: str,
    ) -> GraphitiRelationAdmissionPlan | GraphitiRelationOperationalDecisionPlan:
        if request.proposal.kind is not ExtractionProposalKind.RELATION:
            raise GraphitiAdmissionConsumerError("relation plan requires a relation")
        relation_envelope = request.proposal_authority_binding.proposal_envelope
        endpoint_rows: list[tuple[str, EntityResolutionProposalId, object, object | None]] = []
        canonical_endpoints: list[CanonicalEntityRelationEndpoint | None] = []
        for endpoint_binding in request.relation_endpoint_bindings:
            endpoint_envelope = endpoint_binding.proposal_envelope
            resolution_proposal_id = typed_id(
                EntityResolutionProposalId,
                "graphiti-v1-resolution",
                endpoint_envelope.canonical_digest,
            )
            try:
                proposal = self._entities.proposal(
                    resolution_proposal_id,
                    proof=self._proof,
                )
                if (
                    proposal.proposal_id != resolution_proposal_id
                    or proposal.source_proposal_id != endpoint_envelope.proposal_id
                    or proposal.source_proposal_digest
                    != endpoint_envelope.canonical_digest
                ):
                    raise GraphitiAdmissionConsumerError(
                        "relation endpoint lacks exact 4B proposal authority"
                    )
            except GraphitiAdmissionConsumerError:
                raise
            except Exception as exc:  # noqa: BLE001 - missing authority cannot be bound
                raise GraphitiAdmissionConsumerError(
                    "relation endpoint proposal authority is unavailable"
                ) from exc
            try:
                decision = self._entities.decision(
                    resolution_proposal_id,
                    proof=self._proof,
                )
            except Exception:  # noqa: BLE001 - unavailable current head remains HOLD
                decision = None
            if decision is not None and (
                decision.proposal_id != proposal.proposal_id
                or decision.proposal_version_id != proposal.proposal_version_id
                or decision.proposal_digest != proposal.canonical_digest
            ):
                raise GraphitiAdmissionConsumerError(
                    "relation endpoint 4B decision contradicts its proposal"
                )
            endpoint: CanonicalEntityRelationEndpoint | None = None
            if decision is not None and (
                decision.action is EntityResolutionDecisionAction.ACCEPT
            ):
                if (
                    decision.accepted_entity_id is None
                    or decision.accepted_entity_version_id is None
                ):
                    raise GraphitiAdmissionConsumerError(
                        "accepted relation endpoint lacks canonical identity"
                    )
                try:
                    preferred = self._entities.preferred(
                        decision.accepted_entity_id,
                        proof=self._proof,
                    )
                    version = self._entities.entity_version(
                        decision.accepted_entity_version_id,
                        proof=self._proof,
                    )
                    if (
                        preferred.entity_id == decision.accepted_entity_id
                        and preferred.preferred_entity_id
                        == decision.accepted_entity_id
                        and preferred.current_entity_version_id
                        == decision.accepted_entity_version_id
                        and preferred.lifecycle is CanonicalEntityLifecycle.ACTIVE
                        and version.entity_id == decision.accepted_entity_id
                        and version.entity_version_id
                        == decision.accepted_entity_version_id
                        and version.lifecycle is CanonicalEntityLifecycle.ACTIVE
                    ):
                        endpoint = CanonicalEntityRelationEndpoint(
                            entity_id=decision.accepted_entity_id,
                            entity_version_id=decision.accepted_entity_version_id,
                        )
                except Exception:  # noqa: BLE001 - unavailable current head remains HOLD
                    endpoint = None
            endpoint_rows.append(
                (
                    endpoint_envelope.subject_placeholder,
                    resolution_proposal_id,
                    proposal,
                    decision,
                )
            )
            canonical_endpoints.append(endpoint)

        if tuple(item[0] for item in endpoint_rows) != request.proposed_endpoints:
            raise GraphitiAdmissionConsumerError(
                "relation endpoint names differ from exact ProposalEnvelopes"
            )
        endpoints_current = all(item is not None for item in canonical_endpoints)
        predicate_supported = (
            request.proposal.predicate_hint is ProposalPredicateHint.SAME_PROCESS_AS
        )
        operational_reason: str | None = None
        if not endpoints_current:
            operational_reason = (
                "RELATION_ENDPOINT_RESOLUTION_NOT_ACCEPTED_OR_CURRENT"
            )
        elif not predicate_supported:
            operational_reason = (
                "RELATION_PREDICATE_UNSUPPORTED_FOR_CANONICAL_ENTITY_ENDPOINTS"
            )
        operational_action = (
            GraphitiProposalAdmissionAction.REJECT
            if required_action is GraphitiProposalAdmissionAction.REJECT
            else GraphitiProposalAdmissionAction.HOLD
        )
        if (
            operational_reason is not None
            and required_action is GraphitiProposalAdmissionAction.REJECT
        ):
            operational_reason = "CURRENT_RIGHTS_REJECT"
        if (
            operational_reason is not None
            and required_action not in (None, operational_action)
        ):
            raise GraphitiAdmissionConsumerError(
                "relation action differs from exact endpoint or predicate semantics"
            )

        dependency_requests = tuple(
            sorted(
                (
                    EntityResolutionDependencyRequest(
                        dependency_id=typed_id(
                            EntityResolutionDependencyId,
                            "graphiti-v1-relation-dependency",
                            relation_envelope.canonical_digest,
                            binding.proposal_envelope.canonical_digest,
                        ),
                        dependent_proposal_id=relation_envelope.proposal_id,
                        expected_dependent_proposal_digest=(
                            relation_envelope.canonical_digest
                        ),
                        resolution_proposal_id=row[2].proposal_id,
                        expected_resolution_proposal_version_id=(
                            row[2].proposal_version_id
                        ),
                        expected_resolution_proposal_digest=row[2].canonical_digest,
                        material=operational_reason is None,
                        idempotency_key=(
                            f"{idempotency_key}:dependency:"
                            f"{binding.proposal_envelope.local_id}"
                        ),
                    )
                    for row, binding in zip(
                        endpoint_rows,
                        request.relation_endpoint_bindings,
                        strict=True,
                    )
                ),
                key=lambda item: str(item.dependency_id),
            )
        )

        if operational_reason is not None:
            return GraphitiRelationOperationalDecisionPlan(
                graphiti_proposal_digest=request.proposal.digest,
                graphiti_proposal_local_id=request.proposal.local_id,
                action=operational_action,
                reason_code=operational_reason,
                dependency_requests=dependency_requests,
                endpoint_resolution_proposal_ids=tuple(
                    item[1] for item in endpoint_rows
                ),
                resolved_endpoint_names=tuple(item[0] for item in endpoint_rows),
            )

        temporal_scope, temporal_is_explicit = self._temporal_scope(request)
        planned_action = (
            GraphitiProposalAdmissionAction.ADMIT
            if temporal_is_explicit
            else GraphitiProposalAdmissionAction.HOLD
        )
        if required_action is GraphitiProposalAdmissionAction.REJECT:
            action = GraphitiProposalAdmissionAction.REJECT
        else:
            action = planned_action
            if required_action not in (None, planned_action):
                raise GraphitiAdmissionConsumerError(
                    "relation action differs from exact temporal semantics"
                )

        predicate = EditorialPredicateCode.SAME_PROCESS_AS
        contract = EDITORIAL_PREDICATE_REGISTRY_V1.contract(predicate)
        ordered_endpoints = tuple(
            sorted(
                (item for item in canonical_endpoints if item is not None),
                key=lambda item: canonical_json_bytes(item.canonical_value()),
            )
        )
        evidence = tuple(
            ExtractionRelationEvidence(
                source_proposal_id=relation_envelope.proposal_id,
                source_proposal_digest=relation_envelope.canonical_digest,
                run_id=relation_envelope.run_id,
                run_version_id=relation_envelope.run_version_id,
                output_id=relation_envelope.output_id,
                passage_id=item.passage_id,
                source_evidence_ordinal=ordinal,
                start_byte=item.start_byte,
                end_byte=item.end_byte,
                evidence_text_digest=item.evidence_text_digest,
            )
            for ordinal, item in enumerate(relation_envelope.evidence)
        )
        relation_seed = relation_envelope.canonical_digest
        proposal_id = typed_id(
            EditorialRelationProposalId,
            "graphiti-v1-editorial-relation",
            relation_seed,
        )
        proposal_version_id = typed_id(
            EditorialRelationProposalVersionId,
            "graphiti-v1-editorial-relation-version",
            relation_seed,
        )
        proposal_request = EditorialRelationProposalRequest(
            proposal_id=proposal_id,
            proposal_version_id=proposal_version_id,
            version_number=1,
            expected_previous_version_id=None,
            predicate_registry_digest=EDITORIAL_PREDICATE_REGISTRY_V1.digest,
            predicate_contract_digest=contract.digest,
            predicate=predicate,
            subject=ordered_endpoints[0],
            object=ordered_endpoints[1],
            temporal_scope=temporal_scope,
            evidence=evidence,
            resolution_dependency_ids=tuple(
                item.dependency_id for item in dependency_requests
            ),
            producer=EditorialRelationProducer(
                kind=EditorialRelationProducerKind.EXTRACTION_RUN,
                producer_id="graphiti-admission",
                producer_version="graphiti-conservative-admission-v1",
                contract_digest=relation_envelope.producer_contract_digest,
            ),
            statement=str(request.relation_statement),
            confidence_basis_points=relation_envelope.confidence_basis_points,
            uncertainty_codes=relation_envelope.uncertainty_codes,
            basis_codes=tuple(
                sorted(
                    set(
                        (
                            *relation_envelope.rationale_codes,
                            "EXACT_ENDPOINT_RESOLUTIONS",
                            "EXACT_PROPOSAL_ENVELOPE",
                        )
                    )
                )
            ),
            idempotency_key=f"{idempotency_key}:proposal",
        )
        relation_action = {
            GraphitiProposalAdmissionAction.ADMIT: (
                EditorialRelationDecisionAction.ACCEPT
            ),
            GraphitiProposalAdmissionAction.REJECT: (
                EditorialRelationDecisionAction.REJECT
            ),
            GraphitiProposalAdmissionAction.HOLD: EditorialRelationDecisionAction.HOLD,
        }[action]
        return GraphitiRelationAdmissionPlan(
            graphiti_proposal_digest=request.proposal.digest,
            graphiti_proposal_local_id=request.proposal.local_id,
            proposal_request=proposal_request,
            decision_request=EditorialRelationDecisionRequest(
                decision_id=typed_id(
                    EditorialRelationDecisionId,
                    "graphiti-v1-editorial-relation-decision",
                    relation_seed,
                    action.value,
                ),
                action=relation_action,
                proposal_id=proposal_id,
                proposal_version_id=proposal_version_id,
                expected_proposal_version_digest=proposal_request.canonical_digest,
                expected_previous_decision_id=None,
                expected_previous_decision_version=0,
                assertion_id=(
                    typed_id(
                        EditorialRelationAssertionId,
                        "graphiti-v1-editorial-relation-assertion",
                        relation_seed,
                    )
                    if action is GraphitiProposalAdmissionAction.ADMIT
                    else None
                ),
                target_assertion_id=None,
                successor_assertion_id=None,
                supersession_id=None,
                reason_code={
                    GraphitiProposalAdmissionAction.ADMIT: (
                        "EXACT_CURRENT_ENDPOINTS_AND_TEMPORAL_SEMANTICS"
                    ),
                    GraphitiProposalAdmissionAction.REJECT: "CURRENT_RIGHTS_REJECT",
                    GraphitiProposalAdmissionAction.HOLD: (
                        "EXPLICIT_TEMPORAL_SEMANTICS_REQUIRED"
                    ),
                }[action],
                decision_policy_version=(
                    EDITORIAL_RELATION_ADMISSION_POLICY_VERSION
                ),
                idempotency_key=f"{idempotency_key}:decision",
            ),
            dependency_requests=dependency_requests,
            endpoint_resolution_proposal_ids=tuple(
                item[1] for item in endpoint_rows
            ),
            resolved_endpoint_names=tuple(item[0] for item in endpoint_rows),
        )


EntityPlanBuilder = Callable[
    [
        GraphitiAdmissionRequest,
        GraphitiProposalAdmissionAction | None,
        str,
    ],
    GraphitiEntityAdmissionPlan,
]
RelationPlanBuilder = Callable[
    [
        GraphitiAdmissionRequest,
        GraphitiProposalAdmissionAction | None,
        str,
    ],
    GraphitiRelationAdmissionPlan | GraphitiRelationOperationalDecisionPlan,
]


_ENTITY_ACTIONS = {
    EntityResolutionDecisionAction.ACCEPT: GraphitiProposalAdmissionAction.ADMIT,
    EntityResolutionDecisionAction.REJECT: GraphitiProposalAdmissionAction.REJECT,
    EntityResolutionDecisionAction.HOLD: GraphitiProposalAdmissionAction.HOLD,
}
_RELATION_ACTIONS = {
    EditorialRelationDecisionAction.ACCEPT: GraphitiProposalAdmissionAction.ADMIT,
    EditorialRelationDecisionAction.REJECT: GraphitiProposalAdmissionAction.REJECT,
    EditorialRelationDecisionAction.HOLD: GraphitiProposalAdmissionAction.HOLD,
}


class ExistingGovernedGraphitiAdmissionAuthority:
    """Route admission through existing authenticated authority facades."""

    def __init__(
        self,
        *,
        entities: GovernedEntityRecords,
        relations: GovernedEditorialRelations,
        proof: AuthenticationProof,
        entity_plan: EntityPlanBuilder,
        relation_plan: RelationPlanBuilder,
    ) -> None:
        self._entities = entities
        self._relations = relations
        self._proof = proof
        self._entity_plan = entity_plan
        self._relation_plan = relation_plan

    @staticmethod
    def _require_binding(
        request: GraphitiAdmissionRequest,
        *,
        digest: str,
        local_id: str,
    ) -> None:
        if (
            digest != request.proposal.digest
            or local_id != request.proposal.local_id
        ):
            raise GraphitiAdmissionConsumerError(
                "authority plan does not bind the exact Graphiti proposal"
            )

    @staticmethod
    def _require_action(
        action: GraphitiProposalAdmissionAction,
        required: GraphitiProposalAdmissionAction | None,
    ) -> None:
        if required is not None and action is not required:
            raise GraphitiAdmissionConsumerError(
                "authority plan did not honour the required admission action"
            )

    def decide_entity_resolution(
        self,
        request: GraphitiAdmissionRequest,
        *,
        required_action: GraphitiProposalAdmissionAction | None,
        idempotency_key: str,
    ) -> GraphitiGovernedDecision:
        if request.proposal.kind is ExtractionProposalKind.RELATION:
            raise GraphitiAdmissionConsumerError(
                "relation proposal cannot use entity resolution authority"
            )
        plan = self._entity_plan(request, required_action, idempotency_key)
        self._require_binding(
            request,
            digest=plan.graphiti_proposal_digest,
            local_id=plan.graphiti_proposal_local_id,
        )
        planned_action = _ENTITY_ACTIONS.get(plan.decision_request.action)
        if planned_action is None:
            raise GraphitiAdmissionConsumerError(
                "entity plan contains a non-admission decision"
            )
        self._require_action(planned_action, required_action)
        for mention in plan.mention_requests:
            self._entities.admit_mention(mention, proof=self._proof)
        proposed = self._entities.propose_resolution(
            plan.proposal_request, proof=self._proof
        )
        proposal_request = plan.proposal_request
        if (
            proposed.proposal_id != proposal_request.proposal_id
            or proposed.proposal_version_id
            != proposal_request.proposal_version_id
            or proposed.version_number != proposal_request.version_number
            or proposed.previous_proposal_version_id
            != proposal_request.expected_previous_version_id
            or proposed.source_proposal_id != proposal_request.source_proposal_id
            or proposed.source_proposal_digest
            != proposal_request.expected_source_proposal_digest
            or proposed.kind is not proposal_request.kind
            or proposed.subject_mention_id != proposal_request.subject_mention_id
            or proposed.object_mention_id != proposal_request.object_mention_id
            or proposed.candidate_entity_id != proposal_request.candidate_entity_id
            or proposed.candidate_entity_version_id
            != proposal_request.candidate_entity_version_id
            or proposed.confidence_basis_points
            != proposal_request.confidence_basis_points
            or proposed.uncertainty_codes != proposal_request.uncertainty_codes
            or proposed.basis_codes != proposal_request.basis_codes
            or proposed.stable_semantic_digest
            != proposal_request.stable_semantic_digest
        ):
            raise GraphitiAdmissionConsumerError(
                "entity decision command differs from the retained authority proposal"
            )
        decision_request = replace(
            plan.decision_request,
            expected_proposal_digest=proposed.canonical_digest,
        )
        decided = self._entities.decide_resolution(
            decision_request, proof=self._proof
        )
        if not isinstance(decided, EntityResolutionDecision):
            raise GraphitiAdmissionConsumerError(
                "entity authority returned an untyped decision"
            )
        action = _ENTITY_ACTIONS.get(decided.action)
        if action is None:
            raise GraphitiAdmissionConsumerError(
                "entity authority returned a non-admission decision"
            )
        self._require_action(action, required_action)
        return GraphitiGovernedDecision(
            proposal_key=request.proposal_key,
            proposal_digest=request.proposal.digest,
            proposal_kind=request.proposal.kind,
            proposal_local_id=request.proposal.local_id,
            action=action,
            decision_id=str(decided.decision_id),
            authority_ledger_seq=decided.authority_ledger_seq,
            reason_code=decided.reason_code,
            authority_receipt_digest=digest_canonical(decided.canonical_value()),
            admitted_authority_id=(
                str(decided.accepted_entity_id)
                if action is GraphitiProposalAdmissionAction.ADMIT
                and decided.accepted_entity_id is not None
                else None
            ),
        )

    def decide_relation_admission(
        self,
        request: GraphitiAdmissionRequest,
        *,
        required_action: GraphitiProposalAdmissionAction | None,
        idempotency_key: str,
    ) -> GraphitiGovernedDecision:
        if request.proposal.kind is not ExtractionProposalKind.RELATION:
            raise GraphitiAdmissionConsumerError(
                "entity proposal cannot use relation admission authority"
            )
        plan = self._relation_plan(request, required_action, idempotency_key)
        self._require_binding(
            request,
            digest=plan.graphiti_proposal_digest,
            local_id=plan.graphiti_proposal_local_id,
        )
        if isinstance(plan, GraphitiRelationOperationalDecisionPlan):
            if (
                len(plan.endpoint_resolution_proposal_ids) != 2
                or len(set(plan.endpoint_resolution_proposal_ids)) != 2
                or plan.resolved_endpoint_names != request.proposed_endpoints
            ):
                raise GraphitiAdmissionConsumerError(
                    "operational relation decision lacks exact endpoint bindings"
                )
            self._require_action(plan.action, required_action)
            bases: list[GraphitiRelationHoldBasis] = []
            for dependency_request in plan.dependency_requests:
                retained = self._entities.bind_resolution_dependency(
                    dependency_request,
                    proof=self._proof,
                )
                if (
                    retained.dependency_id != dependency_request.dependency_id
                    or retained.dependent_proposal_id
                    != dependency_request.dependent_proposal_id
                    or retained.dependent_proposal_digest
                    != dependency_request.expected_dependent_proposal_digest
                    or retained.resolution_proposal_id
                    != dependency_request.resolution_proposal_id
                    or retained.proposal_version_id
                    != dependency_request.expected_resolution_proposal_version_id
                    or retained.proposal_version_digest
                    != dependency_request.expected_resolution_proposal_digest
                    or retained.material
                ):
                    raise GraphitiAdmissionConsumerError(
                        "operational relation dependency differs from exact 4B authority"
                    )
                bases.append(
                    GraphitiRelationHoldBasis(
                        dependency_id=str(retained.dependency_id),
                        authority_event_id=str(retained.authority_event_id),
                        authority_ledger_seq=retained.authority_ledger_seq,
                        authority_receipt_digest=retained.canonical_digest,
                    )
                )
            hold_basis = tuple(sorted(bases, key=lambda item: item.dependency_id))
            latest = max(hold_basis, key=lambda item: item.authority_ledger_seq)
            return GraphitiGovernedDecision(
                proposal_key=request.proposal_key,
                proposal_digest=request.proposal.digest,
                proposal_kind=request.proposal.kind,
                proposal_local_id=request.proposal.local_id,
                action=plan.action,
                decision_id=str(
                    typed_id(
                        EntityResolutionDecisionId,
                        "graphiti-v1-operational-relation-decision",
                        request.proposal_authority_binding.proposal_envelope.canonical_digest,
                        plan.action.value,
                        plan.reason_code,
                    )
                ),
                authority_ledger_seq=latest.authority_ledger_seq,
                reason_code=plan.reason_code,
                authority_receipt_digest=latest.authority_receipt_digest,
                relation_hold_basis=hold_basis,
            )
        if (
            len(plan.endpoint_resolution_proposal_ids) != 2
            or len(set(plan.endpoint_resolution_proposal_ids)) != 2
            or plan.resolved_endpoint_names != request.proposed_endpoints
        ):
            raise GraphitiAdmissionConsumerError(
                "relation plan lacks two exact governed endpoint bindings"
            )
        planned_action = _RELATION_ACTIONS.get(plan.decision_request.action)
        if planned_action is None:
            raise GraphitiAdmissionConsumerError(
                "relation plan contains a lifecycle decision"
            )
        self._require_action(planned_action, required_action)
        if planned_action is GraphitiProposalAdmissionAction.ADMIT:
            if (
                request.proposal.predicate_hint
                is not ProposalPredicateHint.SAME_PROCESS_AS
                or plan.proposal_request.predicate
                is not EditorialPredicateCode.SAME_PROCESS_AS
                or request.relation_temporal_bounds is None
            ):
                raise GraphitiAdmissionConsumerError(
                    "ambiguous or unsupported relation remains HOLD"
                )
        endpoint_decisions = []
        for proposal_id in plan.endpoint_resolution_proposal_ids:
            decision = self._entities.decision(proposal_id, proof=self._proof)
            if (
                decision is None
                or decision.action is not EntityResolutionDecisionAction.ACCEPT
                or decision.accepted_entity_id is None
                or decision.accepted_entity_version_id is None
            ):
                raise GraphitiAdmissionConsumerError(
                    "relation endpoint resolution is not governed-current"
                )
            endpoint_decisions.append(decision)
        expected_endpoints = {
            canonical_json_bytes(
                CanonicalEntityRelationEndpoint(
                    entity_id=decision.accepted_entity_id,
                    entity_version_id=decision.accepted_entity_version_id,
                ).canonical_value()
            )
            for decision in endpoint_decisions
        }
        actual_endpoints = {
            canonical_json_bytes(item.canonical_value())
            for item in (
                plan.proposal_request.subject,
                plan.proposal_request.object,
            )
            if isinstance(item, CanonicalEntityRelationEndpoint)
        }
        relation_envelope = request.proposal_authority_binding.proposal_envelope
        if (
            actual_endpoints != expected_endpoints
            or tuple(
                sorted(
                    (item.resolution_proposal_id for item in plan.dependency_requests),
                    key=str,
                )
            )
            != tuple(sorted(plan.endpoint_resolution_proposal_ids, key=str))
            or any(
                item.dependent_proposal_id != relation_envelope.proposal_id
                or item.expected_dependent_proposal_digest
                != relation_envelope.canonical_digest
                for item in plan.dependency_requests
            )
        ):
            raise GraphitiAdmissionConsumerError(
                "relation plan differs from exact endpoint authority dependencies"
            )
        for dependency_request in plan.dependency_requests:
            retained_dependency = self._entities.bind_resolution_dependency(
                dependency_request,
                proof=self._proof,
            )
            if (
                retained_dependency.dependency_id
                != dependency_request.dependency_id
                or retained_dependency.dependent_proposal_id
                != dependency_request.dependent_proposal_id
                or retained_dependency.dependent_proposal_digest
                != dependency_request.expected_dependent_proposal_digest
                or retained_dependency.resolution_proposal_id
                != dependency_request.resolution_proposal_id
                or retained_dependency.proposal_version_id
                != dependency_request.expected_resolution_proposal_version_id
                or retained_dependency.proposal_version_digest
                != dependency_request.expected_resolution_proposal_digest
                or not retained_dependency.material
            ):
                raise GraphitiAdmissionConsumerError(
                    "retained relation dependency differs from exact authority"
                )
        proposed = self._relations.propose(
            plan.proposal_request, proof=self._proof
        )
        if (
            proposed.proposal_id != plan.decision_request.proposal_id
            or proposed.proposal_version_id
            != plan.decision_request.proposal_version_id
            or proposed.canonical_digest
            != plan.decision_request.expected_proposal_version_digest
        ):
            raise GraphitiAdmissionConsumerError(
                "relation decision command differs from the retained authority proposal"
            )
        decided = self._relations.decide(
            plan.decision_request, proof=self._proof
        )
        action = _RELATION_ACTIONS.get(decided.action)
        if action is None:
            raise GraphitiAdmissionConsumerError(
                "relation authority returned a lifecycle decision"
            )
        self._require_action(action, required_action)
        endpoint_binding = tuple(
            (
                proposal_id,
                str(decision.decision_id),
                name,
            )
            for proposal_id, decision, name in zip(
                plan.endpoint_resolution_proposal_ids,
                endpoint_decisions,
                plan.resolved_endpoint_names,
                strict=True,
            )
        )
        return GraphitiGovernedDecision(
            proposal_key=request.proposal_key,
            proposal_digest=request.proposal.digest,
            proposal_kind=request.proposal.kind,
            proposal_local_id=request.proposal.local_id,
            action=action,
            decision_id=str(decided.decision_id),
            authority_ledger_seq=decided.authority_ledger_seq,
            reason_code=decided.reason_code,
            authority_receipt_digest=decided.canonical_digest,
            admitted_authority_id=(
                str(decided.assertion_id)
                if action is GraphitiProposalAdmissionAction.ADMIT
                and decided.assertion_id is not None
                else None
            ),
            endpoint_resolution_decision_ids=tuple(
                item[1] for item in endpoint_binding
            ) if action is GraphitiProposalAdmissionAction.ADMIT else (),
            resolved_endpoint_names=plan.resolved_endpoint_names
            if action is GraphitiProposalAdmissionAction.ADMIT
            else (),
        )

    def relation_endpoint_resolutions_current(
        self,
        request: GraphitiAdmissionRequest,
        decision: GraphitiGovernedDecision,
    ) -> bool:
        try:
            plan = self._relation_plan(
                request,
                decision.action,
                f"graphiti-admit:{request.proposal_key}",
            )
        except Exception:
            return False
        if (
            plan.graphiti_proposal_digest != request.proposal.digest
            or plan.graphiti_proposal_local_id != request.proposal.local_id
            or plan.resolved_endpoint_names != request.proposed_endpoints
            or len(plan.endpoint_resolution_proposal_ids) != 2
        ):
            return False
        current_ids: list[str] = []
        current_names: list[str] = []
        for proposal_id, expected_decision_id, name in zip(
            plan.endpoint_resolution_proposal_ids,
            decision.endpoint_resolution_decision_ids,
            plan.resolved_endpoint_names,
            strict=True,
        ):
            current = self._entities.decision(proposal_id, proof=self._proof)
            if (
                current is None
                or current.action is not EntityResolutionDecisionAction.ACCEPT
                or str(current.decision_id) != expected_decision_id
            ):
                return False
            current_ids.append(str(current.decision_id))
            current_names.append(name)
        return (
            tuple(current_ids) == decision.endpoint_resolution_decision_ids
            and tuple(current_names) == decision.resolved_endpoint_names
            and tuple(current_names) == request.proposed_endpoints
        )


    def current_context(
        self,
        request: GraphitiAdmissionRequest,
        decision: GraphitiGovernedDecision,
    ) -> GovernedAuthorityContext | None:
        """Hydrate the exact current authority behind one admitted receipt."""

        if decision.action is not GraphitiProposalAdmissionAction.ADMIT:
            return None
        try:
            if request.proposal.kind is ExtractionProposalKind.RELATION:
                plan = self._relation_plan(
                    request,
                    GraphitiProposalAdmissionAction.ADMIT,
                    f"graphiti-admit:{request.proposal_key}",
                )
                self._require_binding(
                    request,
                    digest=plan.graphiti_proposal_digest,
                    local_id=plan.graphiti_proposal_local_id,
                )
                retained = self._relations.decision(
                    plan.proposal_request.proposal_id,
                    proof=self._proof,
                )
                if (
                    retained is None
                    or retained.action is not EditorialRelationDecisionAction.ACCEPT
                    or str(retained.decision_id) != decision.decision_id
                    or retained.authority_ledger_seq != decision.authority_ledger_seq
                    or retained.assertion_id is None
                ):
                    return None
                current = self._relations.current(
                    retained.assertion_id,
                    proof=self._proof,
                )
                assertion = current.assertion
                version = self._relations.proposal_version(
                    assertion.proposal_version_id,
                    proof=self._proof,
                )
                endpoints_current = self.relation_endpoint_resolutions_current(
                    request,
                    decision,
                )
                endpoint_bindings: list[AuthorityContextBinding] = []
                for endpoint in (assertion.subject, assertion.object):
                    if not isinstance(endpoint, CanonicalEntityRelationEndpoint):
                        endpoints_current = False
                        continue
                    preferred = self._entities.preferred(
                        endpoint.entity_id,
                        proof=self._proof,
                    )
                    endpoint_version = self._entities.entity_version(
                        preferred.current_entity_version_id,
                        proof=self._proof,
                    )
                    if not (
                        preferred.entity_id == endpoint.entity_id
                        and preferred.preferred_entity_id == endpoint.entity_id
                        and preferred.current_entity_version_id
                        == endpoint.entity_version_id
                        and preferred.lifecycle is CanonicalEntityLifecycle.ACTIVE
                        and endpoint_version.entity_id == endpoint.entity_id
                        and endpoint_version.entity_version_id
                        == endpoint.entity_version_id
                        and endpoint_version.lifecycle
                        is CanonicalEntityLifecycle.ACTIVE
                    ):
                        endpoints_current = False
                    endpoint_bindings.append(
                        AuthorityContextBinding(
                            authority_kind="CANONICAL_ENTITY",
                            authority_id=str(endpoint.entity_id),
                            authority_version=str(endpoint_version.version_number),
                        )
                    )
                currentness = (
                    "CURRENT"
                    if current.lifecycle is EditorialRelationAssertionLifecycle.ACTIVE
                    and str(current.current_decision_id) == decision.decision_id
                    and current.current_decision_version == retained.decision_version
                    and endpoints_current
                    else "STALE"
                )
                bindings = tuple(
                    sorted(
                        {
                            *endpoint_bindings,
                            AuthorityContextBinding(
                                authority_kind="EDITORIAL_RELATION_ASSERTION",
                                authority_id=str(assertion.assertion_id),
                                authority_version=str(version.version_number),
                            ),
                            AuthorityContextBinding(
                                authority_kind="EDITORIAL_RELATION_DECISION",
                                authority_id=str(retained.decision_id),
                                authority_version=str(retained.decision_version),
                            ),
                        },
                        key=lambda item: (
                            item.authority_kind,
                            item.authority_id,
                            item.authority_version,
                        ),
                    )
                )
                temporal = tuple(
                    sorted(
                        {
                            "admitted_at": (
                                None
                                if assertion.admitted_at is None
                                else assertion.admitted_at.to_text()
                            ),
                            **assertion.temporal_scope.canonical_value(),
                        }.items()
                    )
                )
                return GovernedAuthorityContext(
                    bindings=bindings,
                    admitted_temporal_fields=temporal,
                    currentness_state=currentness,
                    admitted_structured_value_json=canonical_json_bytes(
                        {
                            "authority_kind": "EDITORIAL_RELATION_ASSERTION",
                            "assertion": {
                                "assertion_id": str(assertion.assertion_id),
                                "predicate": assertion.predicate.value,
                                "subject": endpoint_canonical_value(
                                    assertion.subject
                                ),
                                "object": endpoint_canonical_value(assertion.object),
                                "statement": assertion.statement,
                                "temporal_scope": (
                                    assertion.temporal_scope.canonical_value()
                                ),
                                "uncertainty_codes": list(
                                    assertion.uncertainty_codes
                                ),
                            },
                        }
                    ).decode(),
                )

            plan = self._entity_plan(
                request,
                GraphitiProposalAdmissionAction.ADMIT,
                f"graphiti-admit:{request.proposal_key}",
            )
            self._require_binding(
                request,
                digest=plan.graphiti_proposal_digest,
                local_id=plan.graphiti_proposal_local_id,
            )
            retained = self._entities.decision(
                plan.proposal_request.proposal_id,
                proof=self._proof,
            )
            if (
                retained is None
                or retained.action is not EntityResolutionDecisionAction.ACCEPT
                or str(retained.decision_id) != decision.decision_id
                or retained.authority_ledger_seq != decision.authority_ledger_seq
                or retained.accepted_entity_id is None
                or retained.accepted_entity_version_id is None
            ):
                return None
            preferred = self._entities.preferred(
                retained.accepted_entity_id,
                proof=self._proof,
            )
            version = self._entities.entity_version(
                preferred.current_entity_version_id,
                proof=self._proof,
            )
            aliases = self._entities.aliases(
                retained.accepted_entity_id,
                limit=16,
                proof=self._proof,
            )
            currentness = (
                "CURRENT"
                if version.entity_id == retained.accepted_entity_id
                and preferred.entity_id == retained.accepted_entity_id
                and preferred.preferred_entity_id == retained.accepted_entity_id
                and preferred.lifecycle is CanonicalEntityLifecycle.ACTIVE
                and version.lifecycle is CanonicalEntityLifecycle.ACTIVE
                else "STALE"
            )
            return GovernedAuthorityContext(
                bindings=(
                    AuthorityContextBinding(
                        authority_kind="CANONICAL_ENTITY",
                        authority_id=str(retained.accepted_entity_id),
                        authority_version=str(version.version_number),
                    ),
                    AuthorityContextBinding(
                        authority_kind="ENTITY_RESOLUTION_DECISION",
                        authority_id=str(retained.decision_id),
                        authority_version=str(retained.decision_version),
                    ),
                ),
                admitted_temporal_fields=(
                    ("admitted_at", retained.recorded_at.to_text()),
                ),
                currentness_state=currentness,
                admitted_structured_value_json=canonical_json_bytes(
                    {
                        "authority_kind": "CANONICAL_ENTITY",
                        "aliases": [alias.canonical_value() for alias in aliases],
                        "entity_version": version.canonical_value(),
                    }
                ).decode(),
            )
        except Exception:  # noqa: BLE001 - any authority read fault fails closed
            return None


def compose_existing_graphiti_admission_consumer(
    connection: sqlite3.Connection,
    *,
    adapter: GovernedGraphitiProposalAdapter,
    extraction: GovernedExtractionRecords,
    objects: GovernedObjects,
    entities: GovernedEntityRecords,
    relations: GovernedEditorialRelations,
    increment4: Increment4Neo4jController,
    proof: AuthenticationProof,
    max_attempts: int = 3,
) -> GraphitiAdmissionConsumer:
    """Compose the existing queue, authorities and full Increment 4 projector."""

    relation_plan = ConservativeGraphitiRelationPlanBuilder(
        entities=entities,
        proof=proof,
    )
    return GraphitiAdmissionConsumer(
        connection,
        proposal_authority=ExistingGovernedGraphitiProposalAuthority(
            adapter=adapter,
            extraction=extraction,
            proof=proof,
        ),
        rights=ExistingGovernedGraphitiRightsAuthority(
            objects=objects,
            proof=proof,
        ),
        authority=ExistingGovernedGraphitiAdmissionAuthority(
            entities=entities,
            relations=relations,
            proof=proof,
            entity_plan=conservative_entity_mention_plan,
            relation_plan=relation_plan,
        ),
        projector=ExistingIncrement4GenerationProjector(
            controller=increment4,
            proof=proof,
        ),
        max_attempts=max_attempts,
    )


__all__ = [
    "ConservativeGraphitiRelationPlanBuilder",
    "ExistingIncrement4GenerationProjector",
    "ExistingGovernedGraphitiProposalAuthority",
    "ExistingGovernedGraphitiRightsAuthority",
    "ExistingGovernedGraphitiAdmissionAuthority",
    "GraphitiEntityAdmissionPlan",
    "GraphitiRelationAdmissionPlan",
    "GraphitiRelationOperationalDecisionPlan",
    "compose_existing_graphiti_admission_consumer",
    "conservative_entity_mention_plan",
]
