from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from ._capability import _CapabilityIssuer
from ._event_system import _ReadBoundary
from ._increment4_neo4j_boundary import _Increment4Neo4jBoundary
from ._increment4_projection_store import _Increment4ProjectionAuthorityStore
from ._projection_store import (
    _ProjectionAuthorityStore,
    _ProjectionDeliverySource,
)
from ._projection_system import NativeProjections, _ProjectionBoundary
from .auth import AuthenticationProof
from .canonical import digest_canonical
from .models import SemanticCommand
from .persistence import (
    AuthorityCommands,
    AuthorityEvents,
    AuthorityPersistenceError,
    CommittedCommand,
    EventReadPolicy,
)
from .policy import CommandRegistry, PayloadSchemaRegistry
from .service import CommandService
from .types import TrustScope, UtcTimestamp
from .neo4j_fulltext_reader import (
    Neo4jFullTextReadError,
    Neo4jFullTextReadPhase,
    Neo4jFullTextReadRequest,
    Neo4jFullTextReadResult,
    Neo4jFullTextReadTimeout,
    Neo4jFullTextReader,
)
from newsroom.increment4.neo4j import Increment4Neo4jController
from newsroom.projection.mapping import (
    ProjectionIdentitySource,
    StructuralIdentityContext,
    canonical_identity_reference,
    canonical_node_id,
    canonical_node_identity_source,
    structural_node_identity_available,
)
from newsroom.projection.models import (
    DeliveryRecordView,
    ProjectionContractError,
    ProjectionDeliveryOutcome,
    ProjectionDeliveryRequest,
    ProjectionFamilyRegistrationRequest,
    ProjectionFamilyView,
    ProjectionGapResolutionRequest,
    ProjectionGapView,
    ProjectionGenerationCreateRequest,
    ProjectionGenerationId,
    ProjectionGenerationPromotionRequest,
    ProjectionGenerationPromotionView,
    ProjectionGenerationState,
    ProjectionGenerationTransitionRequest,
    ProjectionGenerationView,
    ProjectionGenerationValidationRequest,
    ProjectionGenerationValidationView,
    ProjectionReadPolicy,
    ProjectionStateError,
)
from newsroom.increment4.contracts import INCREMENT4_ADMITTED_FAMILY_ID
from newsroom.projection.neo4j._adapter import _open_neo4j_adapter
from newsroom.projection.neo4j.discovery_health_reads import (
    DiscoveryCoverageHealthReadRequest,
    DiscoveryHealthAuthorityFacade,
    DiscoveryHealthReadError,
    DiscoverySourceHealthReadRequest,
)
from newsroom.projection.neo4j.models import (
    Neo4jApplyResult,
    Neo4jAuthorityCommitPending,
    Neo4jCompatibility,
    Neo4jIdentityConflict,
    Neo4jProjectorConfig,
    Neo4jReadError,
    Neo4jStructuralRead,
    Neo4jWriteError,
    StructuralActiveReadRequest,
    StructuralActiveReconciliationRequest,
    StructuralBatch,
    StructuralDeliveryRequest,
    StructuralGenerationValidationRequest,
    StructuralNode,
    StructuralRebuildRequest,
    StructuralRebuildResult,
    StructuralReadAuthoritySelection,
    StructuralReadMetadata,
    StructuralReadRequest,
    StructuralReadResponse,
    StructuralReconciliationView,
    StructuralRelation,
)
from newsroom.projection.neo4j.qualification import (
    neo4j_compatibility_digest,
)
from newsroom.projection.policy import (
    PROJECTION_COMMAND_TYPES,
    ProjectionContractRegistry,
    merge_projection_authority_registries,
)


class _StructuralGraphAdapter(Protocol):
    def verify_compatibility(self) -> Neo4jCompatibility:
        ...

    def bootstrap_schema(self) -> None:
        ...

    def apply(self, batch: StructuralBatch) -> Neo4jApplyResult:
        ...

    def read(
        self,
        *,
        generation_id: str,
        canonical_ids: tuple[str, ...],
        maximum_ledger_seq: int,
        limit: int,
    ) -> Neo4jStructuralRead:
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

    @property
    def driver_version(self) -> str:
        ...

    def read_increment5_fulltext(
        self,
        *,
        phase: str,
        index_name: str | None,
        lucene_expression: str | None,
        generation_id: str | None,
        source_ids: tuple[str, ...],
        limit: int,
        timeout_ns: int,
        eligible_passage_ids: tuple[str, ...] | None = None,
    ) -> Any:
        ...

    def open_native_retrieval_projection(
        self,
        *,
        generation_id: str,
        fulltext_index: str,
        vector_index: str,
    ) -> Any:
        ...

    def close(self) -> None:
        ...


def _open_structural_graph_adapter(
    config: Neo4jProjectorConfig,
) -> _StructuralGraphAdapter:
    return _open_neo4j_adapter(config)


def _increment5_fulltext_record_mapping(
    value: Any,
    *,
    identity: str,
) -> dict[str, object]:
    if isinstance(value, Mapping):
        record = dict(value)
    else:
        try:
            record = dict(value.items())
        except Exception:
            raise Neo4jFullTextReadError(
                f"Neo4j full-text authority read returned malformed {identity}"
            ) from None
    if any(not isinstance(key, str) for key in record):
        raise Neo4jFullTextReadError(
            f"Neo4j full-text authority read returned malformed {identity}"
        )
    return record


def _open_neo4j_fulltext_reader_with_adapter(
    adapter: _StructuralGraphAdapter,
) -> Neo4jFullTextReader:
    """Reduce the private adapter to one owned, phased read capability."""

    def read(
        request: Neo4jFullTextReadRequest,
    ) -> Neo4jFullTextReadResult:
        try:
            value = adapter.read_increment5_fulltext(
                phase=request.phase.value,
                index_name=request.index_name,
                lucene_expression=request.lucene_expression,
                generation_id=(
                    None
                    if request.generation_id is None
                    else str(request.generation_id)
                ),
                source_ids=request.source_ids,
                eligible_passage_ids=request.eligible_passage_ids,
                limit=request.limit,
                timeout_ns=request.timeout_ns,
            )
        except Neo4jReadError as exc:
            if "timed out" in str(exc).casefold():
                raise Neo4jFullTextReadTimeout(
                    "Neo4j full-text authority read timed out"
                ) from None
            raise Neo4jFullTextReadError(
                "Neo4j full-text authority read is unavailable"
            ) from None
        except Exception:
            raise Neo4jFullTextReadError(
                "Neo4j full-text authority read is unavailable"
            ) from None
        if request.phase is Neo4jFullTextReadPhase.COMPONENT:
            component = (
                None
                if value is None
                else _increment5_fulltext_record_mapping(
                    value,
                    identity="component",
                )
            )
            return Neo4jFullTextReadResult(
                phase=request.phase,
                component=component,
                driver_version=adapter.driver_version,
            )
        if request.phase is Neo4jFullTextReadPhase.INDEX:
            try:
                records = tuple(
                    _increment5_fulltext_record_mapping(
                        item,
                        identity="row",
                    )
                    for item in value
                )
            except Neo4jFullTextReadError:
                raise
            except Exception:
                raise Neo4jFullTextReadError(
                    "Neo4j full-text authority read returned malformed rows"
                ) from None
            return Neo4jFullTextReadResult(
                phase=request.phase,
                indexes=records,
                driver_version=adapter.driver_version,
            )

        envelope = _increment5_fulltext_record_mapping(
            value,
            identity="query envelope",
        )
        if set(envelope) != {"candidate_overflow", "rows"}:
            raise Neo4jFullTextReadError(
                "Neo4j full-text query envelope keys are not exact"
            )
        candidate_overflow = envelope["candidate_overflow"]
        raw_rows = envelope["rows"]
        if type(candidate_overflow) is not bool:
            raise Neo4jFullTextReadError(
                "Neo4j full-text candidate overflow flag is malformed"
            )
        if not isinstance(raw_rows, (list, tuple)):
            raise Neo4jFullTextReadError(
                "Neo4j full-text query rows are malformed"
            )
        try:
            records = tuple(
                _increment5_fulltext_record_mapping(
                    item,
                    identity="row",
                )
                for item in raw_rows
            )
        except Neo4jFullTextReadError:
            raise
        except Exception:
            raise Neo4jFullTextReadError(
                "Neo4j full-text authority read returned malformed rows"
            ) from None
        return Neo4jFullTextReadResult(
            phase=request.phase,
            rows=records,
            candidate_overflow=candidate_overflow,
            driver_version=adapter.driver_version,
        )

    return Neo4jFullTextReader(
        driver_version=adapter.driver_version,
        read=read,
        close=adapter.close,
    )


class NativeRetrievalNeo4jResources:
    """Bounded native retrieval projection and reader sharing one driver."""

    __slots__ = ("projector", "fulltext", "__closed")

    def __init__(self, *, projector: Any, fulltext: Neo4jFullTextReader) -> None:
        self.projector = projector
        self.fulltext = fulltext
        self.__closed = False

    def close(self) -> None:
        if not self.__closed:
            self.__closed = True
            self.fulltext.close()


def open_native_retrieval_neo4j_resources(
    *,
    config: Neo4jProjectorConfig,
    generation_id: str,
    fulltext_index: str,
    vector_index: str,
) -> NativeRetrievalNeo4jResources:
    """Open one private adapter reduced to the two fixed native capabilities."""

    adapter = _open_structural_graph_adapter(config)
    try:
        projector = adapter.open_native_retrieval_projection(
            generation_id=generation_id,
            fulltext_index=fulltext_index,
            vector_index=vector_index,
        )
        fulltext = _open_neo4j_fulltext_reader_with_adapter(adapter)
        return NativeRetrievalNeo4jResources(
            projector=projector, fulltext=fulltext
        )
    except Exception:
        adapter.close()
        raise


class Neo4jStructuralProjector:
    """Public B2 facade: exact delivery plus bounded non-authoritative read."""

    __slots__ = (
        "__deliver",
        "__read",
        "__read_active",
        "__reconcile_active",
        "__rebuild",
        "__validate",
    )

    def __init__(
        self,
        *,
        deliver: Callable[
            [StructuralDeliveryRequest, AuthenticationProof],
            DeliveryRecordView,
        ],
        read: Callable[
            [StructuralReadRequest, AuthenticationProof],
            StructuralReadResponse,
        ],
        read_active: Callable[
            [StructuralActiveReadRequest, AuthenticationProof],
            StructuralReadResponse,
        ],
        reconcile_active: Callable[
            [StructuralActiveReconciliationRequest, AuthenticationProof],
            StructuralReconciliationView,
        ],
        rebuild: Callable[
            [StructuralRebuildRequest, AuthenticationProof],
            StructuralRebuildResult,
        ],
        validate_generation: Callable[
            [StructuralGenerationValidationRequest, AuthenticationProof],
            ProjectionGenerationValidationView,
        ],
    ) -> None:
        self.__deliver = deliver
        self.__read = read
        self.__read_active = read_active
        self.__reconcile_active = reconcile_active
        self.__rebuild = rebuild
        self.__validate = validate_generation

    def deliver(
        self,
        request: StructuralDeliveryRequest,
        *,
        proof: AuthenticationProof,
    ) -> DeliveryRecordView:
        return self.__deliver(request, proof)

    def read(
        self,
        request: StructuralReadRequest,
        *,
        proof: AuthenticationProof,
    ) -> StructuralReadResponse:
        return self.__read(request, proof)

    def read_active(
        self,
        request: StructuralActiveReadRequest,
        *,
        proof: AuthenticationProof,
    ) -> StructuralReadResponse:
        return self.__read_active(request, proof)

    def reconcile_active(
        self,
        request: StructuralActiveReconciliationRequest,
        *,
        proof: AuthenticationProof,
    ) -> StructuralReconciliationView:
        return self.__reconcile_active(request, proof)

    def rebuild(
        self,
        request: StructuralRebuildRequest,
        *,
        proof: AuthenticationProof,
    ) -> StructuralRebuildResult:
        return self.__rebuild(request, proof)

    def validate_generation(
        self,
        request: StructuralGenerationValidationRequest,
        *,
        proof: AuthenticationProof,
    ) -> ProjectionGenerationValidationView:
        return self.__validate(request, proof)


class Neo4jProjectionAuthoritySystem:
    __slots__ = (
        "commands",
        "events",
        "projections",
        "structural",
        "increment4",
        "health",
        "compatibility",
        "__close",
    )

    def __init__(
        self,
        *,
        commands: AuthorityCommands,
        events: AuthorityEvents,
        projections: NativeProjections,
        structural: Neo4jStructuralProjector,
        increment4: Increment4Neo4jController,
        health: DiscoveryHealthAuthorityFacade,
        compatibility: Neo4jCompatibility,
        close: Callable[[], None],
    ) -> None:
        self.commands = commands
        self.events = events
        self.projections = projections
        self.structural = structural
        self.increment4 = increment4
        self.health = health
        self.compatibility = compatibility
        self.__close = close

    def close(self) -> None:
        self.__close()

    def __enter__(self) -> Neo4jProjectionAuthoritySystem:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class _Neo4jProjectionBoundary:
    def __init__(
        self,
        *,
        store: _ProjectionAuthorityStore,
        projection_boundary: _ProjectionBoundary,
        adapter: _StructuralGraphAdapter,
        clock: Callable[[], UtcTimestamp],
        operation_lock: RLock | None = None,
    ) -> None:
        self._store = store
        self._projection_boundary = projection_boundary
        self._adapter = adapter
        self._clock = clock
        self._operation_lock = operation_lock or RLock()

    @staticmethod
    def _require_generic_structural_family(family_id: str) -> None:
        from newsroom.increment4.contracts import INCREMENT4_ADMITTED_FAMILY_ID

        if family_id == INCREMENT4_ADMITTED_FAMILY_ID:
            raise ProjectionStateError(
                "Increment 4 admitted projection requires its bounded controller"
            )

    def _require_generic_mutation_generation(
        self,
        generation_id: ProjectionGenerationId,
    ) -> None:
        metadata = self._store.projection_generation_metadata(generation_id)
        self._require_generic_structural_family(metadata.family.family_id)

    def _authorize_transition(
        self,
        request: ProjectionGenerationTransitionRequest,
        proof: AuthenticationProof,
    ):
        return self._projection_boundary._grant(
            command_type="projection.generation.transition",
            aggregate_id=request.generation_id.as_aggregate_id(),
            expected_version=request.expected_authority_version,
            payload={
                "generation_id": str(request.generation_id),
                "target_state": request.target_state.value,
                "validated_through_ledger_seq": (
                    request.validated_through_ledger_seq
                ),
                "reason_code": request.reason_code,
            },
            idempotency_key=request.idempotency_key,
            proof=proof,
        )

    def _authorize_validation(
        self,
        request: ProjectionGenerationValidationRequest,
        proof: AuthenticationProof,
    ):
        return self._projection_boundary._grant(
            command_type="projection.generation.validate",
            aggregate_id=request.generation_id.as_aggregate_id(),
            expected_version=request.expected_authority_version,
            payload={
                "generation_id": str(request.generation_id),
                "checkpoint_ledger_seq": request.checkpoint_ledger_seq,
                "service_compatibility_digest": (
                    request.service_compatibility_digest
                ),
                "projection_state_digest": request.projection_state_digest,
                "reason_code": request.reason_code,
            },
            idempotency_key=request.idempotency_key,
            proof=proof,
        )

    def _authorize_rebuild(
        self,
        request: StructuralRebuildRequest,
        proof: AuthenticationProof,
    ):
        return self._projection_boundary._grant(
            command_type="projection.generation.rebuild",
            aggregate_id=request.generation_id.as_aggregate_id(),
            expected_version=request.expected_authority_version,
            payload={
                "generation_id": str(request.generation_id),
                "through_ledger_seq": request.through_ledger_seq,
                "reason_code": request.reason_code,
            },
            idempotency_key=request.idempotency_key,
            proof=proof,
        )

    def _authorize_gap_resolution(
        self,
        request: ProjectionGapResolutionRequest,
        proof: AuthenticationProof,
    ):
        return self._projection_boundary._grant(
            command_type="projection.gap.resolve",
            aggregate_id=request.generation_id.as_aggregate_id(),
            expected_version=request.expected_authority_version,
            payload={
                "generation_id": str(request.generation_id),
                "gap_id": str(request.gap_id),
                "reason_code": request.reason_code,
            },
            idempotency_key=request.idempotency_key,
            proof=proof,
        )

    def register_family(
        self,
        request: ProjectionFamilyRegistrationRequest,
        proof: AuthenticationProof,
    ) -> ProjectionFamilyView:
        if not isinstance(request, ProjectionFamilyRegistrationRequest):
            raise TypeError("projection family registration requires a typed request")
        self._projection_boundary._authenticate(proof)
        self._require_generic_structural_family(request.family_id)
        return self._projection_boundary.register_family(request, proof)

    def create_generation(
        self,
        request: ProjectionGenerationCreateRequest,
        proof: AuthenticationProof,
    ) -> ProjectionGenerationView:
        if not isinstance(request, ProjectionGenerationCreateRequest):
            raise TypeError("projection generation creation requires a typed request")
        self._projection_boundary._authenticate(proof)
        self._require_generic_structural_family(request.family_id)
        return self._projection_boundary.create_generation(request, proof)

    def transition_generation(
        self,
        request: ProjectionGenerationTransitionRequest,
        proof: AuthenticationProof,
    ) -> ProjectionGenerationView:
        if not isinstance(request, ProjectionGenerationTransitionRequest):
            raise TypeError("projection generation transition requires a typed request")
        grant = self._authorize_transition(request, proof)
        self._require_generic_mutation_generation(request.generation_id)
        return self._store.transition_generation(
            grant,
            generation_id=request.generation_id,
            target_state=request.target_state,
            validated_through_ledger_seq=request.validated_through_ledger_seq,
            reason_code=request.reason_code,
        )

    def record_delivery(
        self,
        request: ProjectionDeliveryRequest,
        proof: AuthenticationProof,
    ) -> DeliveryRecordView:
        if not isinstance(request, ProjectionDeliveryRequest):
            raise TypeError("projection delivery requires a typed request")
        grant = self._projection_boundary._authorize_delivery(
            request,
            proof,
        )
        self._require_generic_mutation_generation(request.generation_id)
        return self._projection_boundary._commit_delivery(grant, request)

    def resolve_gap(
        self,
        request: ProjectionGapResolutionRequest,
        proof: AuthenticationProof,
    ) -> ProjectionGapView:
        if not isinstance(request, ProjectionGapResolutionRequest):
            raise TypeError("projection gap resolution requires a typed request")
        grant = self._authorize_gap_resolution(request, proof)
        self._require_generic_mutation_generation(request.generation_id)
        return self._store.resolve_gap(
            grant,
            generation_id=request.generation_id,
            gap_id=request.gap_id,
            reason_code=request.reason_code,
        )

    def deliver(
        self,
        request: StructuralDeliveryRequest,
        proof: AuthenticationProof,
    ) -> DeliveryRecordView:
        with self._operation_lock:
            return self._deliver_locked(request, proof)

    def _deliver_locked(
        self,
        request: StructuralDeliveryRequest,
        proof: AuthenticationProof,
    ) -> DeliveryRecordView:
        if not isinstance(request, StructuralDeliveryRequest):
            raise TypeError("structural delivery requires a typed request")

        applied_request = ProjectionDeliveryRequest(
            generation_id=request.generation_id,
            expected_authority_version=request.expected_authority_version,
            ledger_seq=request.ledger_seq,
            outcome=ProjectionDeliveryOutcome.APPLIED,
            idempotency_key=request.idempotency_key,
        )
        # Authentication and exact command authorization happen before any graph
        # effect.  The returned grant remains bound to the APPLIED transition.
        applied_grant = self._projection_boundary._authorize_delivery(
            applied_request,
            proof,
        )
        source = self._store.projection_delivery_source(
            request.generation_id,
            request.ledger_seq,
        )
        self._require_generic_structural_family(source.family.family_id)
        if (
            applied_grant.replay_of_command_id is None
            and source.generation.authority_aggregate_version
            != request.expected_authority_version
        ):
            raise ProjectionStateError(
                "projection generation authority version changed before graph apply"
            )
        if source.generation.state in {
            ProjectionGenerationState.RETIRED,
            ProjectionGenerationState.FAILED,
        }:
            raise ProjectionStateError(
                "terminal projection generation cannot accept graph delivery"
            )

        if source.mapping is None:
            return self._record_without_graph(
                request=request,
                proof=proof,
                outcome=ProjectionDeliveryOutcome.IGNORED_OPTIONAL,
                error_code=None,
            )

        try:
            batch = _build_structural_batch(source)
        except ProjectionContractError:
            if source.mapping.required:
                return self._record_without_graph(
                    request=request,
                    proof=proof,
                    outcome=ProjectionDeliveryOutcome.REQUIRED_UNSUPPORTED,
                    error_code="STRUCTURAL_MAPPING_UNSUPPORTED",
                )
            return self._record_without_graph(
                request=request,
                proof=proof,
                outcome=ProjectionDeliveryOutcome.IGNORED_OPTIONAL,
                error_code=None,
            )

        try:
            self._adapter.apply(batch)
        except Neo4jIdentityConflict:
            try:
                return self._record_without_graph(
                    request=request,
                    proof=proof,
                    outcome=ProjectionDeliveryOutcome.RETRYABLE_FAILURE,
                    error_code="NEO4J_IDENTITY_CONFLICT",
                )
            except ProjectionStateError as exc:
                if "already finalized" not in str(exc):
                    raise
                raise Neo4jIdentityConflict(
                    "Neo4j delivery identity conflicts with finalized B1 authority"
                ) from None
        except Neo4jWriteError:
            return self._record_without_graph(
                request=request,
                proof=proof,
                outcome=ProjectionDeliveryOutcome.RETRYABLE_FAILURE,
                error_code="NEO4J_WRITE_FAILURE",
            )

        try:
            return self._projection_boundary._commit_delivery(
                applied_grant,
                applied_request,
            )
        except Exception:
            # The graph marker is durable, but B1 did not confirm authoritative
            # progress.  A retry will observe a DUPLICATE graph delivery and
            # retry only the exact SQLite transition.
            raise Neo4jAuthorityCommitPending(
                "Neo4j delivery committed but B1 authority transition is pending"
            ) from None

    def rebuild(
        self,
        request: StructuralRebuildRequest,
        proof: AuthenticationProof,
    ) -> StructuralRebuildResult:
        with self._operation_lock:
            return self._rebuild_locked(request, proof)

    def _rebuild_locked(
        self,
        request: StructuralRebuildRequest,
        proof: AuthenticationProof,
    ) -> StructuralRebuildResult:
        if not isinstance(request, StructuralRebuildRequest):
            raise TypeError("structural rebuild requires a typed request")
        grant = self._authorize_rebuild(request, proof)
        self._require_generic_mutation_generation(request.generation_id)
        receipt = self._store.begin_projection_rebuild(
            grant,
            generation_id=request.generation_id,
            through_ledger_seq=request.through_ledger_seq,
        )
        if receipt.generation.state is not ProjectionGenerationState.BUILDING:
            raise ProjectionStateError(
                "only a building generation can be destructively rebuilt"
            )

        deleted = self._adapter.cleanup_generation(str(request.generation_id))
        reapplied = 0
        recorded = 0
        ignored = 0
        blocked = 0
        for ledger_seq in range(1, request.through_ledger_seq + 1):
            source = self._store.projection_delivery_source(
                request.generation_id, ledger_seq
            )
            state = self._store.projection_rebuild_delivery_state(
                request.generation_id, ledger_seq
            )
            if source.mapping is None:
                if not source.policy_omitted:
                    ignored += 1
                    continue
                current = self._store.projection_generation(
                    request.generation_id
                )
                result = self.deliver(
                    StructuralDeliveryRequest(
                        generation_id=request.generation_id,
                        expected_authority_version=(
                            current.authority_aggregate_version
                        ),
                        ledger_seq=ledger_seq,
                        idempotency_key=(
                            "rebuild-policy-omission:"
                            + digest_canonical(
                                {
                                    "rebuild_idempotency_key": (
                                        request.idempotency_key
                                    ),
                                    "generation_id": str(
                                        request.generation_id
                                    ),
                                    "ledger_seq": ledger_seq,
                                }
                            )
                        ),
                    ),
                    proof,
                )
                if (
                    result.outcome
                    is ProjectionDeliveryOutcome.IGNORED_OPTIONAL
                ):
                    ignored += 1
                else:
                    blocked += 1
                continue
            if state is not None:
                if (
                    state.finalized
                    and state.outcome is ProjectionDeliveryOutcome.APPLIED
                ):
                    self._adapter.apply(_build_structural_batch(source))
                    reapplied += 1
                    continue
                if state.outcome is ProjectionDeliveryOutcome.IGNORED_OPTIONAL:
                    ignored += 1
                    continue
                if state.finalized:
                    blocked += 1
                    continue
                attempt_number = state.attempt_count + 1
            else:
                attempt_number = 1

            current = self._store.projection_generation(request.generation_id)
            delivery_key = "rebuild-delivery:" + digest_canonical(
                {
                    "rebuild_idempotency_key": request.idempotency_key,
                    "generation_id": str(request.generation_id),
                    "ledger_seq": ledger_seq,
                    "attempt_number": attempt_number,
                }
            )
            result = self.deliver(
                StructuralDeliveryRequest(
                    generation_id=request.generation_id,
                    expected_authority_version=(
                        current.authority_aggregate_version
                    ),
                    ledger_seq=ledger_seq,
                    idempotency_key=delivery_key,
                ),
                proof,
            )
            if result.outcome is ProjectionDeliveryOutcome.APPLIED:
                recorded += 1
            elif result.outcome is ProjectionDeliveryOutcome.IGNORED_OPTIONAL:
                ignored += 1
            else:
                blocked += 1

        metadata = self._store.projection_generation_metadata(
            request.generation_id
        )
        return StructuralRebuildResult(
            generation_id=request.generation_id,
            through_ledger_seq=request.through_ledger_seq,
            checkpoint_ledger_seq=metadata.contiguous_ledger_seq,
            rebuild_authority_event_id=receipt.authority_event_id,
            authority_command_replayed=receipt.replayed,
            deleted_graph_record_count=deleted,
            reapplied_delivery_count=reapplied,
            recorded_delivery_count=recorded,
            ignored_optional_count=ignored,
            blocked_delivery_count=blocked,
            serving_time=metadata.serving_time,
        )

    def promote_generation(
        self,
        request: ProjectionGenerationPromotionRequest,
        proof: AuthenticationProof,
    ) -> ProjectionGenerationPromotionView:
        with self._operation_lock:
            return self._promote_generation_locked(request, proof)

    def _promote_generation_locked(
        self,
        request: ProjectionGenerationPromotionRequest,
        proof: AuthenticationProof,
    ) -> ProjectionGenerationPromotionView:
        if not isinstance(request, ProjectionGenerationPromotionRequest):
            raise TypeError("projection promotion requires a typed request")
        target_grant, prior_grant = (
            self._projection_boundary._authorize_promotion(request, proof)
        )
        self._require_generic_mutation_generation(request.generation_id)
        if request.prior_generation_id is not None:
            self._require_generic_mutation_generation(
                request.prior_generation_id
            )
        metadata = self._store.projection_generation_metadata(
            request.generation_id
        )
        self._require_generic_structural_family(metadata.family.family_id)
        replaying = target_grant.replay_of_command_id is not None
        if replaying:
            if metadata.generation.state not in {
                ProjectionGenerationState.ACTIVE,
                ProjectionGenerationState.RETIRED,
            }:
                raise ProjectionStateError(
                    "promotion replay requires an active or retired generation"
                )
            checkpoint_ledger_seq = metadata.contiguous_ledger_seq
        else:
            if metadata.generation.state is not ProjectionGenerationState.VALIDATING:
                raise ProjectionStateError(
                    "only a validating generation can be promoted"
                )
            if metadata.contiguous_ledger_seq != request.checkpoint_ledger_seq:
                raise ProjectionStateError(
                    "promotion must bind the exact authority checkpoint"
                )
            checkpoint_ledger_seq = request.checkpoint_ledger_seq
        if metadata.open_gap_count or metadata.dead_letter_count:
            raise ProjectionStateError(
                "promotion requires zero gaps and dead letters"
            )
        validation = self._store.projection_generation_validation(
            request.generation_id
        )
        if replaying:
            if (
                validation.checkpoint_ledger_seq != checkpoint_ledger_seq
                or metadata.generation.validated_through_ledger_seq
                != checkpoint_ledger_seq
            ):
                raise ProjectionStateError(
                    "promotion replay requires validation through the current authority checkpoint"
                )
        else:
            if validation.validation_digest != request.validation_digest:
                raise ProjectionStateError(
                    "promotion requires the exact retained validation evidence"
                )
            if validation.checkpoint_ledger_seq != request.checkpoint_ledger_seq:
                raise ProjectionStateError(
                    "promotion validation checkpoint is stale"
                )

        compatibility_digest = neo4j_compatibility_digest(
            self._adapter.verify_compatibility()
        )
        if (
            compatibility_digest
            != validation.service_compatibility_digest
        ):
            raise Neo4jIdentityConflict(
                "Neo4j compatibility differs from retained validation"
            )
        batches = self._expected_validation_batches(
            request.generation_id,
            checkpoint_ledger_seq,
        )
        state_digest = self._adapter.reconcile_generation(
            generation_id=str(request.generation_id),
            expected_batches=batches,
        )
        if state_digest != validation.projection_state_digest:
            raise Neo4jIdentityConflict(
                "Neo4j graph state differs from retained validation"
            )
        return self._projection_boundary._commit_promotion(
            target_grant,
            prior_grant,
            request,
        )

    def reject_direct_validation(
        self,
        request: ProjectionGenerationValidationRequest,
        proof: AuthenticationProof,
    ) -> ProjectionGenerationValidationView:
        if not isinstance(request, ProjectionGenerationValidationRequest):
            raise TypeError("projection validation requires a typed request")
        self._authorize_validation(request, proof)
        self._require_generic_mutation_generation(request.generation_id)
        raise ProjectionStateError(
            "Neo4j generation validation requires structural reconciliation"
        )

    def validate_generation(
        self,
        request: StructuralGenerationValidationRequest,
        proof: AuthenticationProof,
    ) -> ProjectionGenerationValidationView:
        with self._operation_lock:
            return self._validate_generation_locked(request, proof)

    def _validate_generation_locked(
        self,
        request: StructuralGenerationValidationRequest,
        proof: AuthenticationProof,
    ) -> ProjectionGenerationValidationView:
        if not isinstance(request, StructuralGenerationValidationRequest):
            raise TypeError("structural validation requires a typed request")
        authenticated = self._projection_boundary._authenticate(proof)
        metadata = self._store.projection_generation_metadata(
            request.generation_id
        )
        self._require_generic_structural_family(metadata.family.family_id)
        self._projection_boundary._authorize_management_operation(
            family_id=metadata.family.family_id,
            aggregate_id=str(request.generation_id),
            operation="neo4j-generation-reconcile",
            semantic_value={
                "generation_id": str(request.generation_id),
                "checkpoint_ledger_seq": request.checkpoint_ledger_seq,
                "reason_code": request.reason_code,
            },
            authenticated=authenticated,
        )
        if metadata.generation.state not in {
            ProjectionGenerationState.BUILDING,
            ProjectionGenerationState.VALIDATING,
            ProjectionGenerationState.ACTIVE,
        }:
            raise ProjectionStateError(
                "only building, validating or active generations can be reconciled"
            )
        if metadata.contiguous_ledger_seq != request.checkpoint_ledger_seq:
            raise ProjectionStateError(
                "structural validation must bind the exact authority checkpoint"
            )
        if metadata.open_gap_count or metadata.dead_letter_count:
            raise ProjectionStateError(
                "structural validation requires zero gaps and dead letters"
            )
        batches = self._expected_validation_batches(
            request.generation_id,
            request.checkpoint_ledger_seq,
        )
        compatibility = self._adapter.verify_compatibility()
        compatibility_digest = neo4j_compatibility_digest(compatibility)
        state_digest = self._adapter.reconcile_generation(
            generation_id=str(request.generation_id),
            expected_batches=batches,
        )
        authoritative_request = ProjectionGenerationValidationRequest(
            generation_id=request.generation_id,
            expected_authority_version=request.expected_authority_version,
            checkpoint_ledger_seq=request.checkpoint_ledger_seq,
            service_compatibility_digest=compatibility_digest,
            projection_state_digest=state_digest,
            reason_code=request.reason_code,
            idempotency_key=request.idempotency_key,
        )
        return self._projection_boundary.validate_generation(
            authoritative_request,
            proof,
        )

    def _expected_validation_batches(
        self,
        generation_id: ProjectionGenerationId,
        checkpoint_ledger_seq: int,
    ) -> tuple[StructuralBatch, ...]:
        batches: list[StructuralBatch] = []
        for ledger_seq in range(1, checkpoint_ledger_seq + 1):
            source = self._store.projection_delivery_source(
                generation_id,
                ledger_seq,
            )
            state = self._store.projection_rebuild_delivery_state(
                generation_id,
                ledger_seq,
            )
            if state is None:
                if source.mapping is None:
                    continue
                raise ProjectionStateError(
                    "authoritative checkpoint lacks a structural delivery state"
                )
            if (
                str(state.source_event_id) != source.event.event_id
                or state.source_event_digest != source.source_event_digest
            ):
                raise ProjectionStateError(
                    "structural delivery provenance differs from retained authority"
                )
            if not state.finalized:
                raise ProjectionStateError(
                    "structural validation encountered an unfinished delivery"
                )
            if state.outcome is ProjectionDeliveryOutcome.APPLIED:
                if source.mapping is None:
                    raise ProjectionStateError(
                        "applied delivery has no retained structural mapping"
                    )
                batches.append(_build_structural_batch(source))
                continue
            if state.outcome is ProjectionDeliveryOutcome.IGNORED_OPTIONAL:
                if source.mapping is not None and source.mapping.required:
                    raise ProjectionStateError(
                        "required structural mapping was ignored"
                    )
                continue
            raise ProjectionStateError(
                "structural validation encountered a failed delivery"
            )
        return tuple(batches)

    def read(
        self,
        request: StructuralReadRequest,
        proof: AuthenticationProof,
    ) -> StructuralReadResponse:
        if not isinstance(request, StructuralReadRequest):
            raise TypeError("structural read requires a typed request")
        self._projection_boundary._read_policy.require_limit(request.limit)
        authenticated = self._projection_boundary._authenticate_read(proof)
        metadata = self._store.projection_generation_metadata(
            request.generation_id
        )
        return self._read_with_metadata(
            metadata=metadata,
            canonical_ids=request.canonical_ids,
            query_valid_time=request.query_valid_time,
            limit=request.limit,
            operation="neo4j-structural",
            authenticated=authenticated,
            authority_selection=(
                StructuralReadAuthoritySelection.EXACT_GENERATION
            ),
        )

    def reconcile_active(
        self,
        request: StructuralActiveReconciliationRequest,
        proof: AuthenticationProof,
    ) -> StructuralReconciliationView:
        if not isinstance(request, StructuralActiveReconciliationRequest):
            raise TypeError("active reconciliation requires a typed request")
        with self._operation_lock:
            authenticated = self._projection_boundary._authenticate_read(proof)
            self._projection_boundary._authorize_read(
                family_id=request.family_id,
                operation="neo4j-structural-active-reconcile",
                semantic_value={"family_id": request.family_id},
                authenticated=authenticated,
            )
            metadata = self._store.projection_active_generation_metadata(
                request.family_id
            )
            self._require_generic_structural_family(metadata.family.family_id)
            expected = self._expected_validation_batches(
                metadata.generation.generation_id,
                metadata.contiguous_ledger_seq,
            )
            digest = self._adapter.reconcile_generation(
                generation_id=str(metadata.generation.generation_id),
                expected_batches=expected,
            )
            return StructuralReconciliationView(
                family_id=request.family_id,
                generation_id=metadata.generation.generation_id,
                checkpoint_ledger_seq=metadata.contiguous_ledger_seq,
                projection_state_digest=digest,
                serving_time=metadata.serving_time,
            )

    def read_active(
        self,
        request: StructuralActiveReadRequest,
        proof: AuthenticationProof,
    ) -> StructuralReadResponse:
        if not isinstance(request, StructuralActiveReadRequest):
            raise TypeError("active structural read requires a typed request")
        with self._operation_lock:
            self._projection_boundary._read_policy.require_limit(request.limit)
            authenticated = self._projection_boundary._authenticate_read(proof)
            # Authorize the family-scoped selection before resolving whether an
            # ACTIVE generation exists.  This prevents the serving facade from
            # becoming an existence oracle for callers without read authority.
            self._projection_boundary._authorize_read(
                family_id=request.family_id,
                operation="neo4j-structural-active-select",
                semantic_value={
                    "family_id": request.family_id,
                    "canonical_ids": list(request.canonical_ids),
                    "query_valid_time": request.query_valid_time.to_text(),
                    "limit": request.limit,
                },
                authenticated=authenticated,
            )
            metadata = self._store.projection_active_generation_metadata(
                request.family_id
            )
            return self._read_with_metadata(
                metadata=metadata,
                canonical_ids=request.canonical_ids,
                query_valid_time=request.query_valid_time,
                limit=request.limit,
                operation="neo4j-structural-active",
                authenticated=authenticated,
                authority_selection=(
                    StructuralReadAuthoritySelection.AUTHORITY_SELECTED_ACTIVE
                ),
            )

    def _read_with_metadata(
        self,
        *,
        metadata: Any,
        canonical_ids: tuple[str, ...],
        query_valid_time: UtcTimestamp,
        limit: int,
        operation: str,
        authenticated: tuple[UtcTimestamp, Any],
        authority_selection: StructuralReadAuthoritySelection,
    ) -> StructuralReadResponse:
        self._projection_boundary._authorize_read(
            family_id=metadata.family.family_id,
            operation=operation,
            semantic_value={
                "generation_id": str(metadata.generation.generation_id),
                "canonical_ids": list(canonical_ids),
                "query_valid_time": query_valid_time.to_text(),
                "limit": limit,
            },
            authenticated=authenticated,
        )
        if query_valid_time.value > metadata.serving_time.value:
            raise ProjectionContractError(
                "query_valid_time cannot be later than serving_time"
            )
        if (
            metadata.family.family_id == INCREMENT4_ADMITTED_FAMILY_ID
            and metadata.generation.state is ProjectionGenerationState.ACTIVE
            and (
                metadata.generation.validated_through_ledger_seq
                != metadata.contiguous_ledger_seq
                or metadata.open_gap_count or metadata.dead_letter_count
            )
        ):
            raise ProjectionStateError(
                "Increment 4 ACTIVE delivery is awaiting complete validation"
            )
        graph = self._adapter.read(
            generation_id=str(metadata.generation.generation_id),
            canonical_ids=canonical_ids,
            maximum_ledger_seq=metadata.contiguous_ledger_seq,
            limit=limit,
        )
        if any(
            relation.ledger_seq > metadata.contiguous_ledger_seq
            for relation in graph.relations
        ):
            raise Neo4jReadError(
                "Neo4j returned relation beyond the authoritative watermark"
            )
        return StructuralReadResponse(
            metadata=StructuralReadMetadata(
                family_id=metadata.family.family_id,
                family_definition_version=metadata.family.definition_version,
                projector_version=metadata.family.projector_version,
                ontology_contract_digest=(
                    metadata.family.ontology_contract_digest
                ),
                mapping_contract_digest=(
                    metadata.family.mapping_contract_digest
                ),
                generation_id=metadata.generation.generation_id,
                generation_state=metadata.generation.state,
                authority_selection=authority_selection,
                contiguous_ledger_seq=metadata.contiguous_ledger_seq,
                open_gap_count=metadata.open_gap_count,
                dead_letter_count=metadata.dead_letter_count,
                trust_scope=TrustScope.ADMITTED,
                query_valid_time=query_valid_time,
                serving_time=metadata.serving_time,
            ),
            nodes=graph.nodes,
            relations=graph.relations,
        )

    def require_lineage_eligible(
        self,
        identifiers: tuple[object, ...],
        proof: AuthenticationProof,
    ) -> None:
        if (
            not isinstance(identifiers, tuple)
            or not identifiers
            or len(identifiers) > 64
        ):
            raise TypeError(
                "lineage eligibility requires bounded governed identities"
            )
        authenticated = self._projection_boundary._authenticate_read(proof)
        self._projection_boundary._authorize_read(
            family_id="graph.discovery_lineage",
            operation="discovery-lineage-eligibility",
            semantic_value={
                "identifiers": [str(value) for value in identifiers],
            },
            authenticated=authenticated,
        )
        try:
            self._store.require_discovery_lineage_subjects_eligible(identifiers)
        except AuthorityPersistenceError as exc:
            raise ProjectionStateError(
                "discovery-lineage subject is not currently eligible"
            ) from exc

    def source_health(
        self,
        request: DiscoverySourceHealthReadRequest,
        proof: AuthenticationProof,
    ):
        from newsroom.projection.health import assess_source_health

        if not isinstance(request, DiscoverySourceHealthReadRequest):
            raise TypeError("source health read requires a typed request")
        authenticated = self._projection_boundary._authenticate_read(proof)
        self._projection_boundary._authorize_read(
            family_id="graph.discovery_lineage",
            operation="discovery-source-health",
            semantic_value={
                "definition_id": str(request.definition_id),
                "policy_id": request.policy.policy_id,
                "policy_version": request.policy.policy_version,
                "assessed_at": request.assessed_at.to_text(),
            },
            authenticated=authenticated,
        )
        try:
            source = self._store.discovery_source_health_input(
                request.definition_id
            )
        except (AuthorityPersistenceError, ProjectionStateError) as exc:
            raise DiscoveryHealthReadError(
                "discovery-lineage subject is not currently eligible"
            ) from exc
        return assess_source_health(
            source,
            policy=request.policy,
            assessed_at=request.assessed_at,
        )

    def coverage_health(
        self,
        request: DiscoveryCoverageHealthReadRequest,
        proof: AuthenticationProof,
    ):
        from newsroom.projection.health import (
            CoveragePathHealthInput,
            assess_coverage_availability,
            assess_source_health,
            summarize_source_path_state,
        )
        from newsroom.sources.types import (
            CoverageResponsibility,
            PortfolioFunction,
        )

        if not isinstance(request, DiscoveryCoverageHealthReadRequest):
            raise TypeError("coverage health read requires a typed request")
        authenticated = self._projection_boundary._authenticate_read(proof)
        self._projection_boundary._authorize_read(
            family_id="graph.discovery_lineage",
            operation="discovery-coverage-health",
            semantic_value={
                "obligation_id": request.obligation_id,
                "policy_id": request.policy.policy_id,
                "policy_version": request.policy.policy_version,
                "assessed_at": request.assessed_at.to_text(),
            },
            authenticated=authenticated,
        )
        paths = []
        for contract in self._store.discovery_coverage_path_contracts(
            request.obligation_id
        ):
            try:
                source = self._store.discovery_source_health_input(
                    contract.definition_id
                )
            except (AuthorityPersistenceError, ProjectionStateError) as exc:
                raise DiscoveryHealthReadError(
                    "discovery-lineage subject is not currently eligible"
                ) from exc
            assessments = assess_source_health(
                source,
                policy=request.policy,
                assessed_at=request.assessed_at,
            )
            paths.append(
                CoveragePathHealthInput(
                    path_id=contract.path_id,
                    obligation_id=contract.obligation_id,
                    responsibility=contract.responsibility,
                    contribution=contract.contribution,
                    portfolio_functions=contract.portfolio_functions,
                    state=summarize_source_path_state(assessments),
                    qualifies_as_substitute=(
                        PortfolioFunction.EXPLICIT_CONTINGENCY
                        in contract.portfolio_functions
                        and contract.responsibility
                        is CoverageResponsibility.OPERATIONAL_RESILIENCE
                    ),
                    evidence=source.evidence,
                )
            )
        return assess_coverage_availability(
            tuple(paths),
            obligation_id=request.obligation_id,
            policy=request.policy,
            assessed_at=request.assessed_at,
        )

    def _record_without_graph(
        self,
        *,
        request: StructuralDeliveryRequest,
        proof: AuthenticationProof,
        outcome: ProjectionDeliveryOutcome,
        error_code: str | None,
    ) -> DeliveryRecordView:
        return self._projection_boundary.record_delivery(
            ProjectionDeliveryRequest(
                generation_id=request.generation_id,
                expected_authority_version=(
                    request.expected_authority_version
                ),
                ledger_seq=request.ledger_seq,
                outcome=outcome,
                idempotency_key=request.idempotency_key,
                error_code=error_code,
            ),
            proof,
        )


def _build_structural_batch(
    source: _ProjectionDeliverySource,
) -> StructuralBatch:
    mapping = source.mapping
    if mapping is None:
        raise ProjectionContractError("structural source event is unmapped")
    if not source.payload_is_mapping and any(
        binding.identity_source is ProjectionIdentitySource.PAYLOAD_FIELD
        for binding in mapping.nodes
    ):
        raise ProjectionContractError(
            "structural event requires retained inline mapping payload"
        )
    event = source.event
    context = StructuralIdentityContext(
        aggregate_type=event.aggregate_type,
        aggregate_id=event.aggregate_id,
        aggregate_version=event.aggregate_version,
        event_id=event.event_id,
        payload_id=event.payload_id,
        payload=source.payload,
    )
    node_by_alias: dict[str, StructuralNode] = {}
    for binding in mapping.nodes:
        if not structural_node_identity_available(binding, context):
            continue
        canonical_id = canonical_node_id(binding, context)
        node_by_alias[binding.alias] = StructuralNode(
            canonical_id=canonical_id,
            node_type=binding.node_type,
            identity_source=canonical_node_identity_source(binding),
            identity_reference_digest=digest_canonical(
                canonical_identity_reference(binding, context)
            ),
            first_ledger_seq=event.ledger_seq,
            first_source_event_id=event.event_id,
            first_source_event_digest=source.source_event_digest,
        )
    recorded_at = UtcTimestamp.parse(event.recorded_at)
    trust_scope = TrustScope(event.trust_scope)
    relations: list[StructuralRelation] = []
    for binding in mapping.relations:
        source_node = node_by_alias.get(binding.source_alias)
        target_node = node_by_alias.get(binding.target_alias)
        if source_node is None or target_node is None:
            continue
        relation_key = digest_canonical(
            {
                "relation_contract": "newsroom-neo4j-structural-relation-v1",
                "generation_id": str(source.generation.generation_id),
                "relation_type": binding.relation_type.value,
                "source_canonical_id": source_node.canonical_id,
                "target_canonical_id": target_node.canonical_id,
                "ledger_seq": event.ledger_seq,
                "source_event_id": event.event_id,
                "source_event_digest": source.source_event_digest,
            }
        )
        relations.append(
            StructuralRelation(
                relation_key=relation_key,
                relation_type=binding.relation_type,
                source_canonical_id=source_node.canonical_id,
                target_canonical_id=target_node.canonical_id,
                ledger_seq=event.ledger_seq,
                source_event_id=event.event_id,
                source_event_type=event.event_type,
                source_event_digest=source.source_event_digest,
                aggregate_type=event.aggregate_type,
                aggregate_id=event.aggregate_id,
                aggregate_version=event.aggregate_version,
                payload_id=event.payload_id,
                payload_digest=event.payload_digest,
                object_admission_id=event.object_admission_id,
                principal_id=event.principal_id,
                trust_scope=trust_scope,
                security_scope=event.security_scope,
                retention_scope=event.retention_scope,
                recorded_at=recorded_at,
            )
        )
    return StructuralBatch(
        generation_id=source.generation.generation_id,
        family_id=source.family.family_id,
        family_definition_version=source.family.definition_version,
        projector_version=source.family.projector_version,
        ontology_contract_digest=source.family.ontology_contract_digest,
        mapping_contract_digest=source.family.mapping_contract_digest,
        ledger_seq=event.ledger_seq,
        source_event_id=event.event_id,
        source_event_type=event.event_type,
        source_event_digest=source.source_event_digest,
        nodes=tuple(node_by_alias[key] for key in sorted(node_by_alias)),
        relations=tuple(
            sorted(relations, key=lambda value: value.relation_key)
        ),
        tombstoned_object_admission_ids=(
            source.tombstoned_object_admission_ids
        ),
    )


def _open_with_adapter(
    *,
    path: Path,
    registry: CommandRegistry,
    payload_schemas: PayloadSchemaRegistry,
    contracts: ProjectionContractRegistry,
    authenticator: Any,
    authorizer: Any,
    event_read_policy: EventReadPolicy,
    projection_read_policy: ProjectionReadPolicy,
    adapter: _StructuralGraphAdapter,
    command_service_version: str = "authority-command-v1",
    busy_timeout_ms: int = 5_000,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
) -> Neo4jProjectionAuthoritySystem:
    merged_registry, merged_schemas = merge_projection_authority_registries(
        command_registry=registry,
        payload_schemas=payload_schemas,
    )
    issuer = _CapabilityIssuer(
        command_registry=merged_registry,
        payload_schemas=merged_schemas,
    )
    store: _Increment4ProjectionAuthorityStore | None = None
    try:
        compatibility = adapter.verify_compatibility()
        adapter.bootstrap_schema()
        store = _Increment4ProjectionAuthorityStore(
            path,
            issuer=issuer,
            command_registry=merged_registry,
            payload_schemas=merged_schemas,
            command_service_version=command_service_version,
            busy_timeout_ms=busy_timeout_ms,
            clock=clock,
            contracts=contracts,
        )
        command_service = CommandService(
            registry=merged_registry,
            payload_schemas=merged_schemas,
            authenticator=authenticator,
            authorizer=authorizer,
            committed_lookup=store,
            clock=clock,
            _issuer=issuer,
        )
        event_read_boundary = _ReadBoundary(
            store=store,
            policy=event_read_policy,
            authenticator=authenticator,
            authorizer=authorizer,
            clock=clock,
        )
        projection_boundary = _ProjectionBoundary(
            store=store,
            contracts=contracts,
            command_service=command_service,
            authenticator=authenticator,
            authorizer=authorizer,
            read_policy=projection_read_policy,
            clock=clock,
        )
        operation_lock = RLock()
        graph_boundary = _Neo4jProjectionBoundary(
            store=store,
            projection_boundary=projection_boundary,
            adapter=adapter,
            clock=clock,
            operation_lock=operation_lock,
        )
        increment4_boundary = _Increment4Neo4jBoundary(
            store=store,
            projection_boundary=projection_boundary,
            structural_reader=graph_boundary,
            adapter=adapter,
            clock=clock,
            operation_lock=operation_lock,
        )

        def execute(
            command: SemanticCommand,
            proof: AuthenticationProof,
        ) -> CommittedCommand:
            if command.command_type in PROJECTION_COMMAND_TYPES:
                raise PermissionError(
                    "projection commands are internal authority operations"
                )
            grant = command_service._authorize_for_commit(
                command,
                proof=proof,
            )
            return store.commit(grant)  # type: ignore[union-attr]

        closed = False

        def close() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            try:
                adapter.close()
            finally:
                store.close()  # type: ignore[union-attr]

        return Neo4jProjectionAuthoritySystem(
            commands=AuthorityCommands(execute),
            events=AuthorityEvents(
                policy_id=event_read_policy.policy_id,
                read=event_read_boundary.events_after,
                provenance=event_read_boundary.provenance,
                result=event_read_boundary.command_result,
            ),
            projections=NativeProjections(
                register_family=graph_boundary.register_family,
                create_generation=graph_boundary.create_generation,
                transition_generation=graph_boundary.transition_generation,
                validate_generation=graph_boundary.reject_direct_validation,
                promote_generation=graph_boundary.promote_generation,
                record_delivery=graph_boundary.record_delivery,
                resolve_gap=graph_boundary.resolve_gap,
                status=projection_boundary.status,
                generations=projection_boundary.generations,
                validation=projection_boundary.validation,
                promotions=projection_boundary.promotions,
                gaps=projection_boundary.gaps,
                dead_letters=projection_boundary.dead_letters,
            ),
            structural=Neo4jStructuralProjector(
                deliver=graph_boundary.deliver,
                read=graph_boundary.read,
                read_active=graph_boundary.read_active,
                reconcile_active=graph_boundary.reconcile_active,
                rebuild=graph_boundary.rebuild,
                validate_generation=graph_boundary.validate_generation,
            ),
            increment4=Increment4Neo4jController(
                build=increment4_boundary.build_and_promote,
                build_current=increment4_boundary.build_current_and_promote,
                status=increment4_boundary.generation_status,
                read_active=increment4_boundary.read_active,
                reconcile_active=increment4_boundary.reconcile_active,
            ),
            health=DiscoveryHealthAuthorityFacade(
                source=graph_boundary.source_health,
                coverage=graph_boundary.coverage_health,
                eligibility=graph_boundary.require_lineage_eligible,
            ),
            compatibility=compatibility,
            close=close,
        )
    except Exception:
        try:
            adapter.close()
        finally:
            if store is not None:
                store.close()
        raise


def open_neo4j_projection_authority_system(
    *,
    path: Path,
    registry: CommandRegistry,
    payload_schemas: PayloadSchemaRegistry,
    contracts: ProjectionContractRegistry,
    authenticator: Any,
    authorizer: Any,
    event_read_policy: EventReadPolicy,
    projection_read_policy: ProjectionReadPolicy,
    neo4j_config: Neo4jProjectorConfig,
    command_service_version: str = "authority-command-v1",
    busy_timeout_ms: int = 5_000,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
) -> Neo4jProjectionAuthoritySystem:
    adapter = _open_neo4j_adapter(neo4j_config)
    return _open_with_adapter(
        path=path,
        registry=registry,
        payload_schemas=payload_schemas,
        contracts=contracts,
        authenticator=authenticator,
        authorizer=authorizer,
        event_read_policy=event_read_policy,
        projection_read_policy=projection_read_policy,
        adapter=adapter,
        command_service_version=command_service_version,
        busy_timeout_ms=busy_timeout_ms,
        clock=clock,
    )


__all__ = [
    "NativeRetrievalNeo4jResources",
    "Neo4jProjectionAuthoritySystem",
    "Neo4jStructuralProjector",
    "open_native_retrieval_neo4j_resources",
    "open_neo4j_projection_authority_system",
]
