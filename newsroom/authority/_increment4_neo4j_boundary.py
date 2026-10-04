from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import json
import uuid
from threading import RLock
from typing import Any, Protocol

from newsroom.authority.auth import AuthenticationProof
from newsroom.authority.canonical import digest_canonical, validate_sha256_digest
from newsroom.authority.persistence import (
    AuthorityPersistenceError,
    ExpectedVersionConflict,
)
from newsroom.authority.types import UtcTimestamp
from newsroom.increment4.contracts import INCREMENT4_ADMITTED_FAMILY_ID
from newsroom.increment4.neo4j import (
    Increment4Neo4jActiveReadRequest,
    Increment4Neo4jBuildRequest,
    Increment4Neo4jBuildResult,
    Increment4Neo4jCurrentBuildRequest,
    Increment4Neo4jGenerationStatus,
)
from newsroom.increment4.projection import build_increment4_admitted_batches
from newsroom.projection.models import (
    ProjectionDeliveryOutcome,
    ProjectionDeliveryRequest,
    ProjectionFamilyRegistrationRequest,
    ProjectionGenerationCreateRequest,
    ProjectionGenerationId,
    ProjectionGenerationPromotionRequest,
    ProjectionGenerationState,
    ProjectionGenerationTransitionRequest,
    ProjectionGenerationValidationRequest,
    ProjectionStateError,
)
from newsroom.projection.neo4j.models import (
    Neo4jAuthorityCommitPending,
    Neo4jIdentityConflict,
    Neo4jWriteError,
    StructuralActiveReadRequest,
    StructuralBatch,
    StructuralReadResponse,
    StructuralReconciliationView,
)
from newsroom.projection.neo4j.qualification import neo4j_compatibility_digest
from newsroom.projection.neo4j._state import _expected_projection_state_digest

from ._increment4_projection_store import (
    _Increment4CurrentBuildInputs, _Increment4ProjectionAuthorityStore,
)
from ._projection_system import _ProjectionBoundary


class _Increment4GraphAdapter(Protocol):
    def verify_compatibility(self) -> Any:
        ...

    def apply(self, batch: StructuralBatch) -> Any:
        ...

    def read(
        self,
        *,
        generation_id: str,
        canonical_ids: tuple[str, ...],
        maximum_ledger_seq: int,
        limit: int,
    ) -> Any:
        ...

    def reconcile_generation(
        self,
        *,
        generation_id: str,
        expected_batches: tuple[StructuralBatch, ...],
    ) -> str:
        ...

    def cleanup_generation(self, generation_id: str) -> int:
        ...


class _Increment4StructuralReader(Protocol):
    def read_active(
        self,
        request: StructuralActiveReadRequest,
        proof: AuthenticationProof,
    ) -> StructuralReadResponse:
        ...

class _Increment4Neo4jBoundary:
    """Private controller implementation over checked authority and graph seams."""

    _FAMILY_REGISTER_KEY = "increment4-admitted-family-register-v1"

    def __init__(
        self,
        *,
        store: _Increment4ProjectionAuthorityStore,
        projection_boundary: _ProjectionBoundary,
        structural_reader: _Increment4StructuralReader,
        adapter: _Increment4GraphAdapter,
        clock: Callable[[], UtcTimestamp],
        operation_lock: RLock,
    ) -> None:
        self._store = store
        self._projection_boundary = projection_boundary
        self._structural_reader = structural_reader
        self._adapter = adapter
        self._clock = clock
        self._operation_lock = operation_lock

    @staticmethod
    def _operation_key(request_key: str, operation: str, value: object) -> str:
        return f"increment4:{operation}:" + digest_canonical(
            {
                "request_key": request_key,
                "operation": operation,
                "value": value,
            }
        )

    def _register_family(self, proof: AuthenticationProof) -> None:
        self._projection_boundary.register_family(
            ProjectionFamilyRegistrationRequest(
                family_id=INCREMENT4_ADMITTED_FAMILY_ID,
                idempotency_key=self._FAMILY_REGISTER_KEY,
            ),
            proof,
        )

    def _authenticate_management(
        self,
        *,
        generation_id: ProjectionGenerationId,
        operation: str,
        semantic_value: dict[str, object],
        proof: AuthenticationProof,
    ) -> None:
        authenticated = self._projection_boundary._authenticate(proof)
        self._projection_boundary._authorize_management_operation(
            family_id=INCREMENT4_ADMITTED_FAMILY_ID,
            aggregate_id=str(generation_id),
            operation=operation,
            semantic_value=semantic_value,
            authenticated=authenticated,
        )

    def _metadata_or_none(self, generation_id: ProjectionGenerationId):
        try:
            return self._store.projection_generation_metadata(generation_id)
        except ProjectionStateError:
            return None

    def _active_metadata_or_none(self):
        try:
            return self._store.projection_active_generation_metadata(
                INCREMENT4_ADMITTED_FAMILY_ID
            )
        except ProjectionStateError:
            return None

    @staticmethod
    def _require_family(metadata: Any) -> None:
        if metadata.family.family_id != INCREMENT4_ADMITTED_FAMILY_ID:
            raise ProjectionStateError(
                "Increment 4 controller cannot operate another projection family"
            )

    def _require_source_snapshot(
        self,
        request: Increment4Neo4jBuildRequest,
    ):
        authoritative = self._store.increment4_admitted_snapshot()
        if (
            request.snapshot.canonical_digest != authoritative.canonical_digest
            or request.snapshot != authoritative
        ):
            raise ProjectionStateError(
                "Increment 4 snapshot differs from exact retained admitted authority"
            )
        return authoritative.through_ledger_seq, authoritative

    def _create_generation(
        self,
        *,
        request: Increment4Neo4jBuildRequest | Increment4Neo4jCurrentBuildRequest,
        snapshot_digest: str,
        proof: AuthenticationProof,
        legacy_identity: bool = False,
    ) -> None:
        creation_value: dict[str, object] = {
            "generation_id": str(request.generation_id),
            "snapshot_digest": snapshot_digest,
        }
        if not legacy_identity:
            creation_value["purge_retired_generation"] = (
                request.purge_retired_generation
            )
        self._projection_boundary.create_generation(
            ProjectionGenerationCreateRequest(
                generation_id=request.generation_id,
                family_id=INCREMENT4_ADMITTED_FAMILY_ID,
                reason_code=request.reason_code,
                idempotency_key=self._operation_key(
                    request.idempotency_key,
                    "create",
                    creation_value,
                ),
            ),
            proof,
        )

    def _retry_active_predecessor_cleanup(
        self,
        *,
        request: Increment4Neo4jBuildRequest | Increment4Neo4jCurrentBuildRequest,
        snapshot_digest: str,
        proof: AuthenticationProof,
    ) -> tuple[Any | None, int | None]:
        metadata = self._metadata_or_none(request.generation_id)
        if (
            metadata is None
            or metadata.generation.state is not ProjectionGenerationState.ACTIVE
        ):
            return None, None
        self._require_family(metadata)

        # Prove this is the exact immutable creation-command replay before any
        # graph cleanup. A changed request key or snapshot digest cannot attach to
        # an existing ACTIVE identity and trigger predecessor deletion.
        try:
            self._create_generation(
                request=request,
                snapshot_digest=snapshot_digest,
                proof=proof,
            )
        except ExpectedVersionConflict:
            if not request.purge_retired_generation:
                raise ProjectionStateError(
                    "Increment 4 ACTIVE retry differs from immutable build intent"
                ) from None
            try:
                # Parent-release creation identities predate the explicit purge
                # bit. Retired graph state is derivative and that release's
                # rights-safe/default contract is therefore migrated one-way to
                # purge=True. A false request never receives this fallback.
                self._create_generation(
                    request=request,
                    snapshot_digest=snapshot_digest,
                    proof=proof,
                    legacy_identity=True,
                )
            except ExpectedVersionConflict:
                raise ProjectionStateError(
                    "Increment 4 ACTIVE retry differs from immutable build intent"
                ) from None
        promotion = self._promotion_for_generation(request.generation_id)
        purged_prior = 0
        if (
            request.purge_retired_generation
            and promotion.prior_generation is not None
        ):
            # The target is already ACTIVE and the predecessor already RETIRED in
            # SQLite. Retry only that immutable retired namespace before the
            # current-source gate, because newer source authority must not strand
            # a failed post-promotion purge forever.
            purged_prior = self._adapter.cleanup_generation(
                str(promotion.prior_generation.generation_id)
            )
        return promotion, purged_prior

    @staticmethod
    def _batch_by_sequence(
        batches: tuple[StructuralBatch, ...],
    ) -> dict[int, StructuralBatch]:
        result: dict[int, StructuralBatch] = {}
        for batch in batches:
            if batch.ledger_seq in result:
                raise ProjectionStateError(
                    "Increment 4 admitted mapper emitted duplicate ledger sequence"
                )
            result[batch.ledger_seq] = batch
        return result

    def _require_batch_source(self, batch: StructuralBatch) -> None:
        source = self._store.projection_delivery_source(
            batch.generation_id,
            batch.ledger_seq,
        )
        if source.mapping is None:
            raise ProjectionStateError(
                "Increment 4 admitted batch source lacks retained mapping authority"
            )
        if (
            batch.family_id != INCREMENT4_ADMITTED_FAMILY_ID
            or batch.family_definition_version != source.family.definition_version
            or batch.projector_version != source.family.projector_version
            or batch.ontology_contract_digest
            != source.family.ontology_contract_digest
            or batch.mapping_contract_digest != source.family.mapping_contract_digest
            or batch.source_event_id != source.event.event_id
            or batch.source_event_type != source.event.event_type
            or batch.source_event_digest != source.source_event_digest
        ):
            raise ProjectionStateError(
                "Increment 4 admitted batch differs from retained source authority"
            )

    def _apply_and_record(
        self,
        *,
        batch: StructuralBatch,
        expected_authority_version: int,
        idempotency_key: str,
        proof: AuthenticationProof,
    ) -> None:
        request = ProjectionDeliveryRequest(
            generation_id=batch.generation_id,
            expected_authority_version=expected_authority_version,
            ledger_seq=batch.ledger_seq,
            outcome=ProjectionDeliveryOutcome.APPLIED,
            idempotency_key=idempotency_key,
        )
        grant = self._projection_boundary._authorize_delivery(request, proof)
        self._require_batch_source(batch)
        try:
            self._adapter.apply(batch)
        except (Neo4jIdentityConflict, Neo4jWriteError):
            raise
        try:
            self._projection_boundary._commit_delivery(grant, request)
        except Exception:
            raise Neo4jAuthorityCommitPending(
                "Increment 4 graph batch committed but delivery authority is pending"
            ) from None

    def _record_ignored(
        self,
        *,
        generation_id: ProjectionGenerationId,
        ledger_seq: int,
        expected_authority_version: int,
        idempotency_key: str,
        proof: AuthenticationProof,
    ) -> None:
        self._projection_boundary.record_delivery(
            ProjectionDeliveryRequest(
                generation_id=generation_id,
                expected_authority_version=expected_authority_version,
                ledger_seq=ledger_seq,
                outcome=ProjectionDeliveryOutcome.IGNORED_OPTIONAL,
                idempotency_key=idempotency_key,
            ),
            proof,
        )

    def _materialize_generation(
        self,
        *,
        request: Increment4Neo4jBuildRequest | Increment4Neo4jCurrentBuildRequest,
        snapshot_digest: str,
        batches: tuple[StructuralBatch, ...],
        source_watermark: int,
        proof: AuthenticationProof,
    ) -> tuple[int, int, int]:
        batch_by_seq = self._batch_by_sequence(batches)
        if batch_by_seq and max(batch_by_seq) > source_watermark:
            raise ProjectionStateError(
                "Increment 4 admitted batch exceeds source watermark"
            )
        deleted = self._adapter.cleanup_generation(str(request.generation_id))
        metadata = self._store.projection_generation_metadata(
            request.generation_id
        )
        if metadata.generation.state not in {
            ProjectionGenerationState.BUILDING,
            ProjectionGenerationState.VALIDATING,
        }:
            raise ProjectionStateError(
                "Increment 4 unfinished delivery requires a BUILDING generation"
            )
        # Every delivery key retains the prepared full-history snapshot identity.
        projected = 0
        states = self._store.projection_rebuild_delivery_states(
            request.generation_id
        )
        # Include sequence 1 so one delivery can advance the optional/unmapped
        # checkpoint across retained history. Remaining skippable sequences are
        # not written as IGNORED_OPTIONAL commands.
        work_seqs = set(batch_by_seq) | set(states)
        if source_watermark >= 1:
            work_seqs.add(1)
        for ledger_seq in sorted(work_seqs):
            batch = batch_by_seq.get(ledger_seq)
            state = states.get(ledger_seq)
            if state is not None and state.finalized:
                source = self._store.projection_delivery_source(
                    request.generation_id,
                    ledger_seq,
                )
                if (
                    str(state.source_event_id) != source.event.event_id
                    or state.source_event_digest != source.source_event_digest
                ):
                    raise ProjectionStateError(
                        "Increment 4 retained delivery provenance changed"
                    )
                if state.outcome is ProjectionDeliveryOutcome.APPLIED:
                    if batch is None:
                        raise ProjectionStateError(
                            "Increment 4 retained APPLIED delivery lacks current admitted batch"
                        )
                    self._require_batch_source(batch)
                    self._adapter.apply(batch)
                    projected += 1
                    continue
                if state.outcome is ProjectionDeliveryOutcome.IGNORED_OPTIONAL:
                    if batch is not None:
                        raise ProjectionStateError(
                            "Increment 4 admitted batch was previously ignored"
                        )
                    continue
                raise ProjectionStateError(
                    "Increment 4 retained delivery is finalized as a failure"
                )

            metadata = self._store.projection_generation_metadata(
                request.generation_id
            )
            if metadata.generation.state is not ProjectionGenerationState.BUILDING:
                raise ProjectionStateError(
                    "Increment 4 unfinished delivery requires a BUILDING generation"
                )
            key_value = {
                "generation_id": str(request.generation_id),
                "ledger_seq": ledger_seq,
                "snapshot_digest": snapshot_digest,
                "outcome": (
                    ProjectionDeliveryOutcome.APPLIED.value
                    if batch is not None
                    else ProjectionDeliveryOutcome.IGNORED_OPTIONAL.value
                ),
            }
            if batch is None:
                self._record_ignored(
                    generation_id=request.generation_id,
                    ledger_seq=ledger_seq,
                    expected_authority_version=(
                        metadata.generation.authority_aggregate_version
                    ),
                    idempotency_key=self._operation_key(
                        request.idempotency_key,
                        "delivery",
                        key_value,
                    ),
                    proof=proof,
                )
            else:
                self._apply_and_record(
                    batch=batch,
                    expected_authority_version=(
                        metadata.generation.authority_aggregate_version
                    ),
                    idempotency_key=self._operation_key(
                        request.idempotency_key,
                        "delivery",
                        key_value,
                    ),
                    proof=proof,
                )
                projected += 1

        metadata = self._store.projection_generation_metadata(
            request.generation_id
        )
        if metadata.contiguous_ledger_seq < source_watermark:
            raise ProjectionStateError(
                "Increment 4 build did not finalize the complete source watermark"
            )
        if metadata.open_gap_count or metadata.dead_letter_count:
            raise ProjectionStateError(
                "Increment 4 build retained gaps or dead letters"
            )
        return deleted, projected, source_watermark - projected

    def _transition_to_validating(
        self,
        *,
        request: Increment4Neo4jBuildRequest | Increment4Neo4jCurrentBuildRequest,
        source_watermark: int,
        proof: AuthenticationProof,
    ) -> Any:
        metadata = self._store.projection_generation_metadata(
            request.generation_id
        )
        if metadata.generation.state is ProjectionGenerationState.BUILDING:
            self._projection_boundary.transition_generation(
                ProjectionGenerationTransitionRequest(
                    generation_id=request.generation_id,
                    expected_authority_version=(
                        metadata.generation.authority_aggregate_version
                    ),
                    target_state=ProjectionGenerationState.VALIDATING,
                    validated_through_ledger_seq=metadata.contiguous_ledger_seq,
                    reason_code=request.reason_code,
                    idempotency_key=self._operation_key(
                        request.idempotency_key,
                        "transition-validating",
                        {
                            "generation_id": str(request.generation_id),
                            "source_watermark": source_watermark,
                        },
                    ),
                ),
                proof,
            )
            metadata = self._store.projection_generation_metadata(
                request.generation_id
            )
        if metadata.generation.state is not ProjectionGenerationState.VALIDATING:
            raise ProjectionStateError(
                "Increment 4 generation must be VALIDATING before reconciliation"
            )
        return metadata

    def _validate(
        self,
        *,
        request: Increment4Neo4jBuildRequest | Increment4Neo4jCurrentBuildRequest,
        snapshot_digest: str,
        batches: tuple[StructuralBatch, ...],
        source_watermark: int,
        proof: AuthenticationProof,
        logical_request=None,
    ):
        metadata = self._transition_to_validating(
            request=request,
            source_watermark=source_watermark,
            proof=proof,
        )
        state_digest = self._adapter.reconcile_generation(
            generation_id=str(request.generation_id),
            expected_batches=batches,
        )
        compatibility_digest = neo4j_compatibility_digest(
            self._adapter.verify_compatibility()
        )
        source_request = request if logical_request is None else logical_request
        try:
            validation = self._store.projection_generation_validation(
                request.generation_id
            )
        except (ProjectionStateError, AuthorityPersistenceError):
            validation = self._projection_boundary.validate_generation(
                ProjectionGenerationValidationRequest(
                    generation_id=request.generation_id,
                    expected_authority_version=(
                        metadata.generation.authority_aggregate_version
                    ),
                    checkpoint_ledger_seq=metadata.contiguous_ledger_seq,
                    service_compatibility_digest=compatibility_digest,
                    projection_state_digest=state_digest,
                    reason_code=request.reason_code,
                    idempotency_key=(
                        self._current_validation_key(source_request)
                        if isinstance(request, Increment4Neo4jCurrentBuildRequest)
                        and request.allow_active_extension
                        else self._operation_key(
                            request.idempotency_key, "validate",
                            {"generation_id": str(request.generation_id),
                             "source_watermark": source_watermark,
                             "snapshot_digest": snapshot_digest},
                        )
                    ),
                    **self._source_binding(source_request, snapshot_digest, source_watermark),
                ),
                proof,
                required_source_ledger_seq=source_watermark,
            )
        else:
            if (
                validation.checkpoint_ledger_seq != metadata.contiguous_ledger_seq
                or validation.service_compatibility_digest != compatibility_digest
                or validation.projection_state_digest != state_digest
            ):
                raise Neo4jIdentityConflict(
                    "Increment 4 retained validation differs from admitted graph state"
                )
        return validation, state_digest

    def _promotion_for_generation(self, generation_id: ProjectionGenerationId):
        matches = tuple(
            item
            for item in self._store.projection_promotions(
                INCREMENT4_ADMITTED_FAMILY_ID,
                1000,
            )
            if item.generation.generation_id == generation_id
        )
        if len(matches) != 1:
            raise ProjectionStateError(
                "Increment 4 generation lacks one exact promotion record"
            )
        return matches[0]

    def _promote(
        self,
        *,
        request: Increment4Neo4jBuildRequest | Increment4Neo4jCurrentBuildRequest,
        batches: tuple[StructuralBatch, ...],
        source_watermark: int,
        validation: Any,
        state_digest: str,
        proof: AuthenticationProof,
    ):
        metadata = self._store.projection_generation_metadata(
            request.generation_id
        )
        if metadata.generation.state is ProjectionGenerationState.ACTIVE:
            if validation.projection_state_digest != state_digest:
                raise Neo4jIdentityConflict(
                    "Increment 4 active generation differs from retained validation"
                )
            return self._promotion_for_generation(request.generation_id)
        if metadata.generation.state is not ProjectionGenerationState.VALIDATING:
            raise ProjectionStateError(
                "Increment 4 promotion requires VALIDATING generation state"
            )
        prior_metadata = self._active_metadata_or_none()
        if (
            prior_metadata is not None
            and prior_metadata.generation.generation_id == request.generation_id
        ):
            prior_metadata = None
        promotion_request = ProjectionGenerationPromotionRequest(
            generation_id=request.generation_id,
            expected_authority_version=(
                metadata.generation.authority_aggregate_version
            ),
            checkpoint_ledger_seq=metadata.contiguous_ledger_seq,
            validation_digest=validation.validation_digest,
            reason_code=request.reason_code,
            idempotency_key=self._operation_key(
                request.idempotency_key,
                "promote",
                {
                    "generation_id": str(request.generation_id),
                    "source_watermark": source_watermark,
                    "validation_digest": validation.validation_digest,
                    "prior_generation_id": (
                        None
                        if prior_metadata is None
                        else str(prior_metadata.generation.generation_id)
                    ),
                },
            ),
            prior_generation_id=(
                None
                if prior_metadata is None
                else prior_metadata.generation.generation_id
            ),
            expected_prior_authority_version=(
                None
                if prior_metadata is None
                else prior_metadata.generation.authority_aggregate_version
            ),
        )
        target_grant, prior_grant = self._projection_boundary._authorize_promotion(
            promotion_request,
            proof,
        )
        # Reconcile again immediately before the atomic SQLite promotion so a
        # graph mutation between validation and activation fails closed.
        current_state_digest = self._adapter.reconcile_generation(
            generation_id=str(request.generation_id),
            expected_batches=batches,
        )
        if current_state_digest != validation.projection_state_digest:
            raise Neo4jIdentityConflict(
                "Increment 4 graph changed after retained validation"
            )
        if (
            neo4j_compatibility_digest(self._adapter.verify_compatibility())
            != validation.service_compatibility_digest
        ):
            raise Neo4jIdentityConflict(
                "Increment 4 Neo4j compatibility changed after validation"
            )
        return self._projection_boundary._commit_promotion(
            target_grant,
            prior_grant,
            promotion_request,
            required_source_ledger_seq=source_watermark,
            expire_prior_diagnostics=isinstance(request, Increment4Neo4jCurrentBuildRequest),
        )

    def _result(
        self,
        *,
        request: Increment4Neo4jBuildRequest | Increment4Neo4jCurrentBuildRequest,
        snapshot_digest: str,
        source_watermark: int,
        batches: tuple[StructuralBatch, ...],
        deleted_target: int,
        ignored: int,
        validation: Any,
        promotion: Any,
        state_digest: str,
        purged_prior: int | None = None,
    ) -> Increment4Neo4jBuildResult:
        if purged_prior is None:
            purged_prior = 0
            if (
                request.purge_retired_generation
                and promotion.prior_generation is not None
            ):
                purged_prior = self._adapter.cleanup_generation(
                    str(promotion.prior_generation.generation_id)
                )
        metadata = self._store.projection_generation_metadata(
            request.generation_id
        )
        if metadata.generation.state is not ProjectionGenerationState.ACTIVE:
            raise ProjectionStateError(
                "Increment 4 build did not produce the ACTIVE generation"
            )
        return Increment4Neo4jBuildResult(
            family_id=INCREMENT4_ADMITTED_FAMILY_ID,
            generation=metadata.generation,
            prior_generation=promotion.prior_generation,
            validation=validation,
            promotion=promotion,
            source_watermark_ledger_seq=source_watermark,
            checkpoint_ledger_seq=metadata.contiguous_ledger_seq,
            projected_batch_count=len(batches),
            ignored_optional_count=ignored,
            deleted_target_graph_record_count=deleted_target,
            purged_retired_graph_record_count=purged_prior,
            source_snapshot_digest=snapshot_digest,
            projection_state_digest=state_digest,
            serving_time=metadata.serving_time,
        )

    def build_and_promote(
        self,
        request: Increment4Neo4jBuildRequest,
        proof: AuthenticationProof,
    ) -> Increment4Neo4jBuildResult:
        if not isinstance(request, Increment4Neo4jBuildRequest):
            raise TypeError("Increment 4 build requires a typed request")
        with self._operation_lock:
            self._register_family(proof)
            snapshot_digest = request.snapshot.canonical_digest
            self._authenticate_build(request, snapshot_digest, request.snapshot.through_ledger_seq, proof)
            # Exact ACTIVE replay may retry its immutable predecessor cleanup
            # before the current-source gate, as in the public tuple contract.
            active_promotion, pre_purged_prior = self._retry_active_predecessor_cleanup(
                request=request, snapshot_digest=snapshot_digest, proof=proof,
            )
            source_watermark, authoritative = self._require_source_snapshot(request)
            batches = build_increment4_admitted_batches(
                authoritative,
                generation_id=request.generation_id,
                family=self._store.projection_family_definition(INCREMENT4_ADMITTED_FAMILY_ID),
            )
            return self._build_prepared(
                request, _Increment4CurrentBuildInputs(source_watermark, snapshot_digest, batches),
                active_promotion, pre_purged_prior, proof,
            )

    def _authenticate_build(self, request, snapshot_digest, source_watermark, proof):
        self._authenticate_management(
            generation_id=request.generation_id,
            operation="increment4-build-and-promote",
            semantic_value={
                "generation_id": str(request.generation_id),
                "snapshot_digest": snapshot_digest,
                "source_watermark": source_watermark,
                "purge_retired_generation": request.purge_retired_generation,
            },
            proof=proof,
        )

    def _build_prepared(
        self,
        request: Increment4Neo4jBuildRequest | Increment4Neo4jCurrentBuildRequest,
        inputs: _Increment4CurrentBuildInputs,
        active_promotion,
        pre_purged_prior,
        proof: AuthenticationProof,
        logical_request=None,
    ) -> Increment4Neo4jBuildResult:
        source_watermark = inputs.source_watermark
        snapshot_digest = inputs.snapshot_digest
        batches = inputs.batches
        # New and unfinished generations replay the exact immutable
        # creation command only after the source gate. ACTIVE generations
        # already replayed it before their bounded predecessor cleanup.
        if active_promotion is None:
            self._create_generation(
                request=request,
                snapshot_digest=snapshot_digest,
                proof=proof,
            )
        metadata = self._store.projection_generation_metadata(
            request.generation_id
        )
        self._require_family(metadata)
        if metadata.generation.state in {
            ProjectionGenerationState.RETIRED,
            ProjectionGenerationState.FAILED,
        }:
            raise ProjectionStateError(
                "Increment 4 cannot rebuild a terminal generation identity"
            )

        if metadata.generation.state is ProjectionGenerationState.ACTIVE:
            validation = self._store.projection_generation_validation(
                request.generation_id
            )
            compatibility_digest = neo4j_compatibility_digest(
                self._adapter.verify_compatibility()
            )
            if (
                validation.checkpoint_ledger_seq
                != metadata.contiguous_ledger_seq
                or metadata.generation.validated_through_ledger_seq
                != validation.checkpoint_ledger_seq
                or validation.service_compatibility_digest
                != compatibility_digest
            ):
                raise Neo4jIdentityConflict(
                    "Increment 4 active generation differs from retained validation"
                )
            state_digest = self._adapter.reconcile_generation(
                generation_id=str(request.generation_id),
                expected_batches=batches,
            )
            if state_digest != validation.projection_state_digest:
                raise Neo4jIdentityConflict(
                    "Increment 4 active graph differs from retained validation"
                )
            promotion = (
                active_promotion
                if active_promotion is not None
                else self._promotion_for_generation(request.generation_id)
            )
            # The serving generation remains reconciliation-only. A retained
            # promotion may still name a retired predecessor whose requested
            # cleanup failed after the atomic SQLite promotion, so result
            # assembly must retry only that retired namespace.
            result = self._result(
                request=request,
                snapshot_digest=snapshot_digest,
                source_watermark=source_watermark,
                batches=batches,
                deleted_target=0,
                ignored=source_watermark - len(batches),
                validation=validation,
                promotion=promotion,
                state_digest=state_digest,
                purged_prior=pre_purged_prior,
            )
            # Authority commands do not share the graph operation lock. Re-read
            # after reconciliation and retired cleanup so a concurrent rights,
            # tombstone or lineage change cannot be reported as current.
            if isinstance(request, Increment4Neo4jBuildRequest):
                self._require_source_snapshot(request)
            elif self._store._increment4_current_build_inputs(
                generation_id=request.generation_id,
                family=self._store.projection_family_definition(INCREMENT4_ADMITTED_FAMILY_ID),
            ) != inputs:
                raise ProjectionStateError(
                    "Increment 4 snapshot differs from exact retained admitted authority"
                )
            return result

        deleted_target, _projected, ignored = self._materialize_generation(
            request=request,
            snapshot_digest=snapshot_digest,
            batches=batches,
            source_watermark=source_watermark,
            proof=proof,
        )
        validation, state_digest = self._validate(
            request=request,
            snapshot_digest=snapshot_digest,
            batches=batches,
            source_watermark=source_watermark,
            proof=proof,
            logical_request=logical_request,
        )
        promotion = self._promote(
            request=request,
            batches=batches,
            source_watermark=source_watermark,
            validation=validation,
            state_digest=state_digest,
            proof=proof,
        )
        return self._result(
            request=request,
            snapshot_digest=snapshot_digest,
            source_watermark=source_watermark,
            batches=batches,
            deleted_target=deleted_target,
            ignored=ignored,
            validation=validation,
            promotion=promotion,
            state_digest=state_digest,
        )

    @classmethod
    def _current_validation_key(cls, request):
        return cls._operation_key(request.idempotency_key, "validate-current", {})

    @staticmethod
    def _source_request_digest(request):
        return digest_canonical({
            "generation_id": str(request.generation_id),
            "reason_code": request.reason_code,
            "idempotency_key": request.idempotency_key,
            "purge_retired_generation": request.purge_retired_generation,
            "allow_active_extension": request.allow_active_extension,
        })

    @classmethod
    def _source_binding(cls, request, snapshot_digest, source_watermark):
        if not (
            isinstance(request, Increment4Neo4jCurrentBuildRequest)
            and request.allow_active_extension
        ):
            return {}
        return {
            "source_snapshot_digest": snapshot_digest,
            "source_watermark_ledger_seq": source_watermark,
            "source_request_digest": cls._source_request_digest(request),
        }

    def _creation_identity(self, request, generation_id):
        row = self._store._connection.execute(
            "SELECT v.authority_event_id,c.idempotency_key,p.payload_bytes FROM projection_generation_versions v "
            "JOIN ledger_events e ON e.event_id=v.authority_event_id "
            "JOIN authority_commands c ON c.command_id=e.command_id "
            "JOIN authority_payloads p ON p.payload_id=c.payload_id "
            "WHERE v.generation_id=? AND v.lifecycle_version=1 AND c.command_type='projection.generation.create'",
            (str(generation_id),),
        ).fetchone()
        if row is None:
            raise ProjectionStateError("CURRENT generation creation proof is absent")
        self._store._validate_retained_event(row[0])
        if json.loads(row[2]) != {"generation_id": str(generation_id),
                "family_id": INCREMENT4_ADMITTED_FAMILY_ID, "reason_code": request.reason_code}:
            raise ProjectionStateError("CURRENT generation creation intent differs")
        return row[0], row[1]

    def _replacement_id(self, request, creation_event, snapshot_digest):
        identity = digest_canonical({"request": self._source_request_digest(request),
            "creation_event": creation_event, "snapshot_digest": snapshot_digest})
        return ProjectionGenerationId.parse(str(uuid.UUID(hex=identity[7:39], version=4)))

    def _supersession_target(self, request, metadata):
        generation = metadata.generation
        creation, _ = self._creation_identity(request, generation.generation_id)
        row = self._store._connection.execute(
            "SELECT v.reason_code,v.authority_event_id,c.idempotency_key FROM projection_generation_versions v "
            "JOIN ledger_events e ON e.event_id=v.authority_event_id "
            "JOIN authority_commands c ON c.command_id=e.command_id "
            "WHERE v.generation_id=? AND v.lifecycle_version=? AND c.command_type='projection.generation.transition'",
            (str(generation.generation_id), generation.lifecycle_version),
        ).fetchone()
        if row is None or not row[0].startswith("CURRENT_NEXT:"):
            raise ProjectionStateError("CURRENT FAILED generation has no supersession intent")
        self._store._validate_retained_event(row[1])
        parts = row[0].split(":")
        if len(parts) != 3:
            raise ProjectionStateError("CURRENT supersession reason differs")
        target = ProjectionGenerationId.parse(parts[1]); snapshot = "sha256:" + parts[2]
        validate_sha256_digest(snapshot)
        expected = self._operation_key(request.idempotency_key, "supersede-current", {
            "request": self._source_request_digest(request), "creation_event": creation,
            "target": str(target), "snapshot_digest": snapshot})
        if target != self._replacement_id(request, creation, snapshot) or row[2] != expected:
            raise ProjectionStateError("CURRENT supersession request binding differs")
        return target, snapshot

    def _require_replacement_validation(self, request, validation):
        generation_id = request.generation_id; visited = set()
        while generation_id != validation.generation_id:
            if generation_id in visited:
                raise ProjectionStateError("CURRENT supersession cycle differs")
            visited.add(generation_id)
            metadata = self._store.projection_generation_metadata(generation_id)
            self._require_family(metadata)
            if metadata.generation.state is not ProjectionGenerationState.FAILED:
                raise ProjectionStateError("CURRENT replacement lacks FAILED predecessor")
            generation_id, snapshot = self._supersession_target(request, metadata)
        if snapshot != validation.source_snapshot_digest:
            raise ProjectionStateError("CURRENT replacement validation snapshot differs")

    def _prepare_current_generation(self, request, inputs, proof):
        if not request.allow_active_extension:
            return request, inputs
        logical = request; expected_snapshot = inputs.snapshot_digest; visited = set()
        while True:
            if request.generation_id in visited:
                raise ProjectionStateError("CURRENT supersession cycle differs")
            visited.add(request.generation_id)
            metadata = self._metadata_or_none(request.generation_id)
            if metadata is None and request.generation_id != logical.generation_id:
                # A normal FAILED transition already declared this exact next
                # namespace. Finish its create even after fail-before-create.
                self._create_generation(request=request, snapshot_digest=expected_snapshot, proof=proof)
                metadata = self._store.projection_generation_metadata(request.generation_id)
            if metadata is None or metadata.generation.state is ProjectionGenerationState.ACTIVE:
                break
            self._require_family(metadata)
            if metadata.generation.state is ProjectionGenerationState.FAILED:
                target, expected_snapshot = self._supersession_target(logical, metadata)
                request = replace(logical, generation_id=target)
                continue
            creation, creation_key = self._creation_identity(logical, request.generation_id)
            wanted = self._operation_key(request.idempotency_key, "create", {
                "generation_id": str(request.generation_id), "snapshot_digest": inputs.snapshot_digest,
                "purge_retired_generation": request.purge_retired_generation})
            if creation_key == wanted or metadata.generation.state is not ProjectionGenerationState.BUILDING:
                break
            if metadata.generation.validated_through_ledger_seq is not None or self._store._connection.execute(
                "SELECT 1 FROM projection_generation_validations WHERE generation_id=? LIMIT 1",
                (str(request.generation_id),),
            ).fetchone() is not None:
                raise ProjectionStateError("CURRENT validated intent cannot be superseded")
            target = self._replacement_id(logical, creation, inputs.snapshot_digest)
            self._projection_boundary.transition_generation(ProjectionGenerationTransitionRequest(
                generation_id=request.generation_id,
                expected_authority_version=metadata.generation.authority_aggregate_version,
                target_state=ProjectionGenerationState.FAILED,
                reason_code=f"CURRENT_NEXT:{target}:{inputs.snapshot_digest[7:]}",
                idempotency_key=self._operation_key(logical.idempotency_key, "supersede-current", {
                    "request": self._source_request_digest(logical), "creation_event": creation,
                    "target": str(target), "snapshot_digest": inputs.snapshot_digest}),
            ), proof)
            request = replace(logical, generation_id=target); expected_snapshot = inputs.snapshot_digest
        if request.generation_id == logical.generation_id:
            return request, inputs
        # Relation identities themselves bind the graph namespace. Re-derive
        # through the existing mapper, never relabel an old batch UUID.
        replacement = self._store._increment4_current_build_inputs(
            generation_id=request.generation_id,
            family=self._store.projection_family_definition(INCREMENT4_ADMITTED_FAMILY_ID),
        )
        if (replacement.snapshot_digest, replacement.source_watermark) != (inputs.snapshot_digest, inputs.source_watermark):
            raise ProjectionStateError("CURRENT source changed during supersession")
        return request, replacement

    def _replay_current_validation(self, request, validation, proof):
        if validation.source_request_digest != self._source_request_digest(request):
            raise ProjectionStateError("Increment 4 retained build request differs")
        metadata = self._store.projection_generation_metadata(validation.generation_id)
        self._require_family(metadata)
        if metadata.generation.state is ProjectionGenerationState.VALIDATING:
            # Validation committed before initial promotion. Resume the existing
            # full-build protocol, never attach this intent to another generation.
            if validation.generation_id != request.generation_id:
                self._require_replacement_validation(request, validation)
            return None
        if metadata.generation.state is not ProjectionGenerationState.ACTIVE:
            raise ProjectionStateError("Increment 4 retained build is no longer ACTIVE")
        self._authenticate_build(
            request, validation.source_snapshot_digest,
            validation.source_watermark_ledger_seq, proof,
        )
        # Replay the immutable cohort receipt, but prove the current graph first.
        # A newer ACTIVE validation is not a rewrite of an earlier cohort receipt.
        current = self._store.projection_generation_validation(validation.generation_id)
        if (
            metadata.contiguous_ledger_seq != current.checkpoint_ledger_seq
            or metadata.generation.validated_through_ledger_seq != current.checkpoint_ledger_seq
            or metadata.open_gap_count or metadata.dead_letter_count
            or neo4j_compatibility_digest(self._adapter.verify_compatibility())
            != current.service_compatibility_digest
        ):
            raise Neo4jIdentityConflict("Increment 4 ACTIVE replay lacks current validation")
        family = self._store.projection_family_definition(INCREMENT4_ADMITTED_FAMILY_ID)
        batches = self._store._increment4_current_batches(
            generation_id=validation.generation_id, family=family,
        )
        state_digest = self._adapter.reconcile_generation(
            generation_id=str(validation.generation_id), expected_batches=batches,
        )
        if state_digest != current.projection_state_digest:
            raise Neo4jIdentityConflict("Increment 4 ACTIVE replay differs from current authority")
        if batches != self._store._increment4_current_batches(
            generation_id=validation.generation_id, family=family,
        ):
            raise ProjectionStateError("Increment 4 source changed during ACTIVE replay")
        historical_batches = tuple(
            batch for batch in batches
            if batch.ledger_seq <= validation.checkpoint_ledger_seq
        )
        return self._result(
            request=replace(request, generation_id=validation.generation_id),
            snapshot_digest=validation.source_snapshot_digest,
            source_watermark=validation.source_watermark_ledger_seq,
            batches=historical_batches,
            deleted_target=0,
            ignored=validation.source_watermark_ledger_seq - len(historical_batches),
            validation=validation,
            promotion=self._promotion_for_generation(validation.generation_id),
            state_digest=validation.projection_state_digest,
        )

    def _extend_or_replay_current(self, request, proof):
        retained = self._store.projection_generation_validation_for_key(
            self._projection_boundary.validation_idempotency_namespace(proof),
            self._current_validation_key(request),
        )
        if retained is not None:
            return self._replay_current_validation(request, retained, proof)
        if self._metadata_or_none(request.generation_id) is not None:
            # Preserve old v2 creation/replay and unfinished full-build behaviour.
            return None
        metadata = self._active_metadata_or_none()
        if metadata is None:
            return None
        self._require_family(metadata)
        generation_id = metadata.generation.generation_id
        validation = self._store.projection_generation_validation(generation_id)
        if metadata.open_gap_count or metadata.dead_letter_count:
            raise ProjectionStateError("Increment 4 ACTIVE extension retained gaps or dead letters")
        compatibility = neo4j_compatibility_digest(self._adapter.verify_compatibility())
        if compatibility != validation.service_compatibility_digest:
            raise Neo4jIdentityConflict("Increment 4 ACTIVE service compatibility changed")
        family = self._store.projection_family_definition(INCREMENT4_ADMITTED_FAMILY_ID)
        inputs = self._store._increment4_current_build_inputs(
            generation_id=generation_id, family=family,
        )
        self._authenticate_build(request, inputs.snapshot_digest, inputs.source_watermark, proof)
        prefix = tuple(b for b in inputs.batches if b.ledger_seq <= validation.checkpoint_ledger_seq)
        if _expected_projection_state_digest(str(generation_id), prefix) != validation.projection_state_digest:
            # Changed rights, lineage or historical membership require the
            # established replacement-generation path. Graph-only drift below
            # never enters this fallback.
            return None

        by_seq = self._batch_by_sequence(inputs.batches)
        states = self._store.projection_rebuild_delivery_states(generation_id)
        delivered = set()
        for seq, state in states.items():
            if not state.finalized or state.outcome not in {
                ProjectionDeliveryOutcome.APPLIED, ProjectionDeliveryOutcome.IGNORED_OPTIONAL,
            }:
                raise ProjectionStateError("Increment 4 ACTIVE delivery retained a failure")
            source = self._store.projection_delivery_source(generation_id, seq)
            if str(state.source_event_id) != source.event.event_id or state.source_event_digest != source.source_event_digest:
                raise ProjectionStateError("Increment 4 retained delivery provenance changed")
            if state.outcome is ProjectionDeliveryOutcome.APPLIED:
                if seq not in by_seq:
                    raise ProjectionStateError("Increment 4 retained APPLIED delivery lacks current batch")
                delivered.add(seq)
            elif seq in by_seq:
                raise ProjectionStateError("Increment 4 admitted batch was previously ignored")
        if any(b.ledger_seq not in delivered for b in prefix):
            raise ProjectionStateError("Increment 4 validated prefix lacks delivery authority")
        committed = tuple(b for b in inputs.batches if b.ledger_seq in delivered)
        pending = tuple(b for b in inputs.batches if b.ledger_seq not in delivered)
        if pending and any(seq > pending[0].ledger_seq for seq in delivered):
            raise ProjectionStateError("Increment 4 ACTIVE delivery history has a gap")
        try:
            self._adapter.reconcile_generation(
                generation_id=str(generation_id), expected_batches=committed,
            )
        except Neo4jIdentityConflict:
            if not pending:
                raise
            # apply is one graph transaction and precedes SQLite delivery commit.
            # Exactly the next authorised batch may therefore already exist;
            # any other graph difference still fails before a write or cleanup.
            self._require_batch_source(pending[0])
            self._adapter.reconcile_generation(
                generation_id=str(generation_id),
                expected_batches=committed + (pending[0],),
            )
        for batch in pending:
            metadata = self._store.projection_generation_metadata(generation_id)
            if metadata.contiguous_ledger_seq < batch.ledger_seq - 1:
                checkpoint = metadata.contiguous_ledger_seq
                self._record_ignored(
                    generation_id=generation_id, ledger_seq=checkpoint + 1,
                    expected_authority_version=metadata.generation.authority_aggregate_version,
                    idempotency_key=self._operation_key(request.idempotency_key, "active-prefix", {
                        "generation_id": str(generation_id), "batch_digest": batch.batch_digest,
                        "checkpoint_ledger_seq": checkpoint,
                    }), proof=proof,
                )
                metadata = self._store.projection_generation_metadata(generation_id)
            self._apply_and_record(
                batch=batch,
                expected_authority_version=metadata.generation.authority_aggregate_version,
                idempotency_key=self._operation_key(request.idempotency_key, "active-delivery", {
                    "generation_id": str(generation_id), "batch_digest": batch.batch_digest,
                }), proof=proof,
            )
        metadata = self._store.projection_generation_metadata(generation_id)
        if metadata.contiguous_ledger_seq < inputs.source_watermark:
            self._record_ignored(
                generation_id=generation_id, ledger_seq=metadata.contiguous_ledger_seq + 1,
                expected_authority_version=metadata.generation.authority_aggregate_version,
                idempotency_key=self._operation_key(request.idempotency_key, "active-ignored", {
                    "generation_id": str(generation_id), "snapshot_digest": inputs.snapshot_digest,
                }), proof=proof,
            )
            metadata = self._store.projection_generation_metadata(generation_id)
        if (
            metadata.generation.state is not ProjectionGenerationState.ACTIVE
            or metadata.contiguous_ledger_seq < inputs.source_watermark
            or metadata.open_gap_count or metadata.dead_letter_count
        ):
            raise ProjectionStateError("Increment 4 ACTIVE extension lacks complete source watermark")
        state_digest = self._adapter.reconcile_generation(
            generation_id=str(generation_id), expected_batches=inputs.batches,
        )
        if inputs.batches != self._store._increment4_current_batches(
            generation_id=generation_id, family=family,
        ):
            raise ProjectionStateError("Increment 4 source changed during ACTIVE extension")
        if compatibility != neo4j_compatibility_digest(self._adapter.verify_compatibility()):
            raise Neo4jIdentityConflict("Increment 4 ACTIVE service compatibility changed")
        validation = self._projection_boundary.validate_generation(
            ProjectionGenerationValidationRequest(
                generation_id=generation_id,
                expected_authority_version=metadata.generation.authority_aggregate_version,
                checkpoint_ledger_seq=metadata.contiguous_ledger_seq,
                service_compatibility_digest=compatibility,
                projection_state_digest=state_digest,
                reason_code=request.reason_code,
                idempotency_key=self._current_validation_key(request),
                **self._source_binding(request, inputs.snapshot_digest, inputs.source_watermark),
            ), proof, required_source_ledger_seq=inputs.source_watermark,
        )
        return self._result(
            request=replace(request, generation_id=generation_id),
            snapshot_digest=inputs.snapshot_digest, source_watermark=inputs.source_watermark,
            batches=inputs.batches, deleted_target=0,
            ignored=inputs.source_watermark - len(inputs.batches),
            validation=validation, promotion=self._promotion_for_generation(generation_id),
            state_digest=state_digest,
        )

    def build_current_and_promote(
        self,
        request: Increment4Neo4jCurrentBuildRequest,
        proof: AuthenticationProof,
    ) -> Increment4Neo4jBuildResult:
        """Prepare current authority once, without retaining historical events."""
        if not isinstance(request, Increment4Neo4jCurrentBuildRequest):
            raise TypeError("Increment 4 current build requires a typed request")
        with self._operation_lock:
            self._register_family(proof)
            self._authenticate_management(
                generation_id=request.generation_id,
                operation="increment4-build-current-and-promote",
                semantic_value={
                    "generation_id": str(request.generation_id),
                    "purge_retired_generation": request.purge_retired_generation,
                },
                proof=proof,
            )
            if request.allow_active_extension:
                result = self._extend_or_replay_current(request, proof)
                if result is not None:
                    return result
            inputs = self._store._increment4_current_build_inputs(
                generation_id=request.generation_id,
                family=self._store.projection_family_definition(INCREMENT4_ADMITTED_FAMILY_ID),
            )
            self._authenticate_build(request, inputs.snapshot_digest, inputs.source_watermark, proof)
            logical_request = request
            request, inputs = self._prepare_current_generation(request, inputs, proof)
            active_promotion, pre_purged_prior = self._retry_active_predecessor_cleanup(
                request=request, snapshot_digest=inputs.snapshot_digest, proof=proof,
            )
            return self._build_prepared(
                request, inputs, active_promotion, pre_purged_prior, proof,
                logical_request=logical_request,
            )

    def generation_status(
        self,
        generation_id: ProjectionGenerationId,
        proof: AuthenticationProof,
    ) -> Increment4Neo4jGenerationStatus:
        if not isinstance(generation_id, ProjectionGenerationId):
            raise TypeError("Increment 4 generation status identity must be typed")
        authenticated = self._projection_boundary._authenticate_read(proof)
        self._projection_boundary._authorize_read(
            family_id=INCREMENT4_ADMITTED_FAMILY_ID,
            operation="increment4-generation-status",
            semantic_value={"generation_id": str(generation_id)},
            authenticated=authenticated,
        )
        metadata = self._store.projection_generation_metadata(generation_id)
        self._require_family(metadata)
        return Increment4Neo4jGenerationStatus(
            generation=metadata.generation,
            contiguous_ledger_seq=metadata.contiguous_ledger_seq,
            open_gap_count=metadata.open_gap_count,
            dead_letter_count=metadata.dead_letter_count,
            source_watermark_ledger_seq=(
                self._store.latest_projection_source_ledger_seq()
            ),
            serving_time=metadata.serving_time,
        )

    def reconcile_active(
        self,
        proof: AuthenticationProof,
    ) -> StructuralReconciliationView:
        with self._operation_lock:
            authenticated = self._projection_boundary._authenticate_read(proof)
            self._projection_boundary._authorize_read(
                family_id=INCREMENT4_ADMITTED_FAMILY_ID,
                operation="increment4-active-reconcile",
                semantic_value={"family_id": INCREMENT4_ADMITTED_FAMILY_ID},
                authenticated=authenticated,
            )
            metadata = self._store.projection_active_generation_metadata(
                INCREMENT4_ADMITTED_FAMILY_ID
            )
            expected = self._store._increment4_current_batches(
                generation_id=metadata.generation.generation_id,
                family=metadata.family,
            )
            digest = self._adapter.reconcile_generation(
                generation_id=str(metadata.generation.generation_id),
                expected_batches=expected,
            )
            validation = self._store.projection_generation_validation(
                metadata.generation.generation_id
            )
            if (
                validation.checkpoint_ledger_seq != metadata.contiguous_ledger_seq
                or validation.projection_state_digest != digest
                or validation.service_compatibility_digest
                != neo4j_compatibility_digest(self._adapter.verify_compatibility())
            ):
                raise Neo4jIdentityConflict(
                    "Increment 4 ACTIVE graph differs from retained validation"
                )
            return StructuralReconciliationView(
                family_id=INCREMENT4_ADMITTED_FAMILY_ID,
                generation_id=metadata.generation.generation_id,
                checkpoint_ledger_seq=metadata.contiguous_ledger_seq,
                projection_state_digest=digest,
                serving_time=metadata.serving_time,
            )

    def read_active(
        self,
        request: Increment4Neo4jActiveReadRequest,
        proof: AuthenticationProof,
    ) -> StructuralReadResponse:
        if not isinstance(request, Increment4Neo4jActiveReadRequest):
            raise TypeError("Increment 4 active read requires a typed request")
        return self._structural_reader.read_active(
            StructuralActiveReadRequest(
                family_id=INCREMENT4_ADMITTED_FAMILY_ID,
                canonical_ids=request.canonical_ids,
                query_valid_time=request.query_valid_time,
                limit=request.limit,
            ),
            proof,
        )


__all__ = ["_Increment4Neo4jBoundary"]
