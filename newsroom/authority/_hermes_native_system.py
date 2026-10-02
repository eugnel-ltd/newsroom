"""One native Hermes authority writer composed over the cumulative store."""

from __future__ import annotations

from ._event_store_base import _validation_stage

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from newsroom.checks.policy import merge_discovery_check_authority_registries
from newsroom.checks.read_policy import DiscoveryCheckReadPolicy
from newsroom.discovery.policy import merge_discovery_signal_lead_registries
from newsroom.discovery.types import DiscoveryReadPolicy
from newsroom.increment6._execution_store import _open_on_connection
from newsroom.increment6.candidates import (
    _compose_story_candidate_authority,
    merge_candidate_authority_registries,
)
from newsroom.increment6.collision import CurrentCollisionEffectEnforcer
from newsroom.increment6.dispositions import (
    CurrentCandidateCitationReadPort,
    ProposalDispositionStore,
)
from newsroom.increment6.hypotheses import _compose_event_hypothesis_authority
from newsroom.increment6.lineage import (
    _compose_event_hypothesis_lineage_authority,
    merge_lineage_authority_registries,
)
from newsroom.increment6.relationships import (
    _compose_event_hypothesis_relationship_authority,
    merge_relationship_authority_registries,
)
from newsroom.increment6.work_items import RetrievalContextAuthority, TriageWorkItemStore
from newsroom.projection.neo4j.models import Neo4jProjectorConfig

from ._check_boundary import _CheckBoundary
from ._check_facade import GovernedChecks
from ._discovery_boundary import _DiscoveryBoundary
from ._discovery_facade import GovernedDiscovery
from ._event_hypothesis_lineage_system import (
    _LineageStore,
    _create_event_hypothesis_lineage_read_port,
)
from ._event_hypothesis_relationship_system import (
    _RelationshipEventStore,
    _create_event_hypothesis_relationship_read_port,
    _verify_relationship_event_coverage,
)
from ._event_hypothesis_system import _HypothesisStore
from ._event_system import _ReadBoundary
from ._graphiti_increment4_system import (
    _AUTHORITY_COMPOSITION_TOKEN,
    GovernedGraphitiIncrement4AuthoritySystem,
    open_governed_graphiti_increment4_authority_system,
)
from ._extraction_facade import GovernedExtractionRecords
from ._object_system import (
    _OBJECT_COMPOSITION_TOKEN,
    _ObjectBoundary,
    GovernedObjects,
)
from ._proposal_admission import _ProposalAdmissionBoundary
from ._signal_lead_admission import _SignalLeadAdmissionBoundary
from .auth import AuthenticationProof, StaticAuthenticator
from .models import SemanticCommand
from .persistence import AuthorityCommands, AuthorityEvents, CommittedCommand, EventReadPolicy
from .policy import CommandRegistry, PayloadSchemaRegistry
from .story_candidate_system import _CandidateStore, _create_story_candidate_read_port
from .types import UtcTimestamp


class _SharedAuthority:
    """Keep child facades from closing the sole authority writer."""

    __slots__ = ("_authority", "_lock")

    def __init__(self, authority: object, lock: Any) -> None:
        self._authority = authority
        self._lock = lock

    def __getattr__(self, name: str) -> object:
        value = getattr(self._authority, name)
        if not callable(value):
            return value

        def serialised(*args: object, **kwargs: object) -> object:
            with self._lock:
                return value(*args, **kwargs)

        return serialised

    def close(self) -> None:
        return None


class _SharedRelationshipStore(_RelationshipEventStore):
    def close(self) -> None:
        return None


class _SharedLineageStore(_LineageStore):
    def close(self) -> None:
        return None


class _SharedCandidateStore(_CandidateStore):
    def close(self) -> None:
        return None


def _share_store(store_type: type, root: object) -> Any:
    store = object.__new__(store_type)
    store.__dict__.update(root.__dict__)
    # Lineage/Candidate stores originate before object-admission payloads.
    # They share the cumulative writer, so dispatch this validation to that
    # writer's actual A2b implementation rather than the old A2a rejection.
    store._validate_object_admission_payload_record = root._validate_object_admission_payload_record
    return store


_SYSTEM_TOKEN = object()


class NativeDependencyFactory(Protocol):
    """Build native retrieval/collision dependencies from the opened base."""

    def __call__(
        self,
        *,
        objects: GovernedObjects,
        extraction: GovernedExtractionRecords,
        commands: AuthorityCommands,
        events: AuthorityEvents,
    ) -> tuple[
        RetrievalContextAuthority,
        CurrentCollisionEffectEnforcer,
        CurrentCandidateCitationReadPort,
    ]: ...


class HermesNativeAuthoritySystem:
    """Server-owned facades sharing one SQLite/CAS writer and lock."""

    def __init__(self, token: object, *, base: GovernedGraphitiIncrement4AuthoritySystem, **facets: object) -> None:
        if token is not _SYSTEM_TOKEN:
            raise TypeError("Hermes native authority systems require the production opener")
        self._base = base
        self.__dict__.update(facets)

    @property
    def sources(self): return self._base.sources
    @property
    def extraction(self): return self._base.extraction
    @property
    def objects(self): return self._base.objects
    @property
    def entities(self): return self._base.entities
    @property
    def relations(self): return self._base.relations
    @property
    def graphiti(self): return self._base.graphiti
    @property
    def structural(self): return self._base.structural
    @property
    def increment4(self): return self._base.increment4
    @property
    def compatibility(self): return self._base.compatibility
    @property
    def authority_store_path(self): return self._base.authority_store_path
    @property
    def graph_destination_id(self): return self._base.graph_destination_id

    def close(self) -> None:
        self._base.close()

    def __enter__(self): return self
    def __exit__(self, *_: object) -> None: self.close()


def open_hermes_native_authority_system(
    *,
    path: Path,
    object_root: Path,
    workspace_root: Path,
    registry: CommandRegistry,
    payload_schemas: PayloadSchemaRegistry,
    admission_registry: Any,
    rights_policies: Any,
    hydration_policies: Any,
    contracts: Any,
    authenticator: StaticAuthenticator,
    authorizer: Any,
    event_read_policy: EventReadPolicy,
    source_read_policy: Any,
    check_read_policy: DiscoveryCheckReadPolicy,
    discovery_read_policy: DiscoveryReadPolicy,
    extraction_read_policy: Any,
    entity_read_policy: Any,
    relation_read_policy: Any,
    graphiti_read_policy: Any,
    projection_read_policy: Any,
    object_limits: Any,
    neo4j_config: Neo4jProjectorConfig,
    retrieval_authority: RetrievalContextAuthority | None = None,
    collision_enforcer: CurrentCollisionEffectEnforcer | None = None,
    native_dependency_factory: NativeDependencyFactory | None = None,
    command_service_version: str = "hermes-native-authority-v1",
    busy_timeout_ms: int = 5_000,
    lease_ttl_seconds: int = 300,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
    cas_fault_hook: Callable[[str], None] | None = None,
    disk_usage: Callable[[Path], Any] | None = None,
) -> HermesNativeAuthoritySystem:
    """Open the sole native writer from current durable state.

    Historical physical/record sweeps are explicit maintenance. Selected
    business reads and effects still prove their exact retained dependencies.
    """

    explicit_dependencies = (
        retrieval_authority is not None or collision_enforcer is not None
    )
    current_candidate_citations = None
    if native_dependency_factory is None:
        if (
            type(retrieval_authority) is not RetrievalContextAuthority
            or type(collision_enforcer) is not CurrentCollisionEffectEnforcer
        ):
            raise TypeError("native retrieval and collision dependencies are required")
    elif explicit_dependencies or not callable(native_dependency_factory):
        raise TypeError("native dependency factory conflicts with explicit dependencies")

    commands, schemas = merge_discovery_check_authority_registries(
        command_registry=registry, payload_schemas=payload_schemas
    )
    commands, schemas = merge_discovery_signal_lead_registries(
        command_registry=commands, payload_schemas=schemas
    )
    commands, schemas = merge_relationship_authority_registries(commands, schemas)
    commands, schemas = merge_lineage_authority_registries(commands, schemas)
    commands, schemas = merge_candidate_authority_registries(commands, schemas)
    base = open_governed_graphiti_increment4_authority_system(
        path=path,
        object_root=object_root,
        workspace_root=workspace_root,
        registry=commands,
        payload_schemas=schemas,
        admission_registry=admission_registry,
        rights_policies=rights_policies,
        hydration_policies=hydration_policies,
        contracts=contracts,
        authenticator=authenticator,
        authorizer=authorizer,
        source_read_policy=source_read_policy,
        extraction_read_policy=extraction_read_policy,
        entity_read_policy=entity_read_policy,
        relation_read_policy=relation_read_policy,
        graphiti_read_policy=graphiti_read_policy,
        projection_read_policy=projection_read_policy,
        object_limits=object_limits,
        neo4j_config=neo4j_config,
        command_service_version=command_service_version,
        busy_timeout_ms=busy_timeout_ms,
        clock=clock,
        cas_fault_hook=cas_fault_hook,
        disk_usage=disk_usage,
        _native_current_state=True,
    )
    try:
        root, service, cas, _, commands, schemas = base._authority_composition(
            _AUTHORITY_COMPOSITION_TOKEN
        )
        connection, operation_lock = root._connection, root._lock
        read_boundary = _ReadBoundary(
            store=root, policy=event_read_policy, authenticator=authenticator,
            authorizer=authorizer, clock=clock,
        )

        def execute(command: SemanticCommand, proof: AuthenticationProof) -> CommittedCommand:
            grant = service._authorize_for_commit(command, proof=proof)
            return root.commit(grant)

        def with_native_rows(operation, *args, **kwargs):
            with operation_lock:
                prior = connection.row_factory
                connection.row_factory = sqlite3.Row
                try:
                    return operation(*args, **kwargs)
                finally:
                    connection.row_factory = prior

        authority_commands = AuthorityCommands(execute)
        authority_events = AuthorityEvents(
            policy_id=event_read_policy.policy_id,
            read=lambda *args, **kwargs: with_native_rows(
                read_boundary.events_after, *args, **kwargs
            ),
            provenance=lambda *args, **kwargs: with_native_rows(
                read_boundary.provenance, *args, **kwargs
            ),
            result=lambda *args, **kwargs: with_native_rows(
                read_boundary.command_result, *args, **kwargs
            ),
        )
        if native_dependency_factory is not None:
            object_boundary = _ObjectBoundary(
                store=root,
                cas=cas,
                object_issuer=root._object_issuer,
                admission_registry=root._admission_registry,
                rights_policies=root._rights_policies,
                hydration_policies=root._hydration_policies,
                authenticator=authenticator,
                authorizer=authorizer,
                command_service=service,
                command_registry=commands,
                clock=clock,
            )

            def composed_hydrate(request, proof):
                operation = (
                    object_boundary.hydrate_in_transaction
                    if connection.in_transaction
                    else object_boundary.hydrate
                )
                return with_native_rows(operation, request, proof)

            def composed_rehydrate(request, proof):
                return with_native_rows(
                    object_boundary.rehydrate, request, proof,
                    in_transaction=connection.in_transaction,
                )

            base.objects._bind_composed_hydrate(
                composed_hydrate, rehydrate=composed_rehydrate,
                _token=_OBJECT_COMPOSITION_TOKEN
            )
            with _validation_stage("native_dependencies"):
                dependencies = native_dependency_factory(
                    objects=base.objects,
                    extraction=base.extraction,
                    commands=authority_commands,
                    events=authority_events,
                )
                if (
                    type(dependencies) is not tuple
                    or len(dependencies) != 3
                    or type(dependencies[0]) is not RetrievalContextAuthority
                    or type(dependencies[1]) is not CurrentCollisionEffectEnforcer
                    or type(dependencies[2]) is not CurrentCandidateCitationReadPort
                ):
                    raise TypeError("native dependency factory result differs")
                retrieval_authority, collision_enforcer, current_candidate_citations = dependencies
        assert retrieval_authority is not None and collision_enforcer is not None

        check_boundary = _CheckBoundary(
            store=root, command_service=service, authenticator=authenticator,
            authorizer=authorizer, read_policy=check_read_policy, clock=clock,
        )
        proposal_admission = _ProposalAdmissionBoundary(store=root, command_service=service)
        discovery_boundary = _DiscoveryBoundary(
            store=root, command_service=service, authenticator=authenticator,
            authorizer=authorizer, read_policy=discovery_read_policy, clock=clock,
        )
        signal_admission = _SignalLeadAdmissionBoundary(store=root, command_service=service)
        checks = GovernedChecks(
            register_request=check_boundary.register_request,
            start_attempt=check_boundary.start_attempt,
            record_outcome=check_boundary.record_outcome,
            decide_baseline=check_boundary.decide_baseline,
            record_transition=check_boundary.record_transition,
            open_finding=check_boundary.open_finding,
            record_finding_occurrence=check_boundary.record_finding_occurrence,
            admit_proposal=proposal_admission.admit,
            request=check_boundary.request, attempt=check_boundary.attempt,
            outcome=check_boundary.outcome, attempts=check_boundary.attempts,
            observed_prior_revision=check_boundary.observed_prior_revision,
            outcomes=check_boundary.outcomes, baseline=check_boundary.baseline,
            current_baseline=check_boundary.current_baseline,
            transition=check_boundary.transition, finding=check_boundary.finding,
            finding_occurrences=check_boundary.finding_occurrences,
        )
        discovery = GovernedDiscovery(
            admit_signal=discovery_boundary.admit_signal,
            decide_gate=discovery_boundary.decide_gate,
            open_lead=discovery_boundary.open_lead,
            record_watch_condition=discovery_boundary.record_watch_condition,
            record_lead_disposition=discovery_boundary.record_lead_disposition,
            admit_signal_to_lead=signal_admission.admit,
            signal=discovery_boundary.signal, gate=discovery_boundary.gate,
            current_gate=discovery_boundary.current_gate, gates=discovery_boundary.gates,
            lead=discovery_boundary.lead, lead_for_signal=discovery_boundary.lead_for_signal,
            watch_condition=discovery_boundary.watch_condition,
            disposition=discovery_boundary.disposition,
            current_disposition=discovery_boundary.current_disposition,
            latest_disposition=discovery_boundary.latest_disposition,
            dispositions=discovery_boundary.dispositions,
            signals_for_revision=discovery_boundary.signals_for_revision,
            current_status=discovery_boundary.current_status,
        )

        with _validation_stage("native_semantic_stores"):
            work_items = TriageWorkItemStore(
                connection, retrieval_authority, _validate_on_open=False,
                _validate_retained_event=lambda event_id: with_native_rows(root._validate_retained_event, event_id),
            )
            executions = _open_on_connection(
                connection, retrieval_authority=retrieval_authority,
                authenticator=authenticator, clock=clock,
                lease_ttl_seconds=lease_ttl_seconds,
                work_items=work_items,
                _validate_on_open=False,
            )
            executions._TriageExecutionAuthority__store._transaction_lock = operation_lock
            dispositions = ProposalDispositionStore(
                connection,
                retrieval_authority,
                authenticator,
                current_candidate_citations,
                work_items=work_items,
                _validate_on_open=False,
            )
            hypothesis_store = _HypothesisStore(
                connection, retrieval_authority, authenticator, clock,
                current_candidate_citations,
                dispositions=dispositions,
                _validate_on_open=False,
            )
            hypothesis_store._lock = operation_lock

        relationship_store = _share_store(_SharedRelationshipStore, root)
        with relationship_store._hypothesis_rows():
            relationship_store._hypotheses = hypothesis_store
        relationship_store._hypotheses._lock = operation_lock
        relationship_store._command_service = service
        lineage_store = _share_store(_SharedLineageStore, root)
        lineage_store._port = _create_event_hypothesis_relationship_read_port(
            connection, retrieval_authority=retrieval_authority,
            authenticator=authenticator, command_registry=commands,
            payload_schemas=schemas, clock=clock,
            current_candidate_citations=current_candidate_citations,
            hypotheses=hypothesis_store,
        )
        lineage_store._service = service
        candidate_store = _share_store(_SharedCandidateStore, root)
        candidate_store._retrieval = retrieval_authority
        candidate_store._authenticator = authenticator
        candidate_store._collision = collision_enforcer
        candidate_store._lineage = _create_event_hypothesis_lineage_read_port(
            connection, retrieval_authority=retrieval_authority,
            authenticator=authenticator, command_registry=commands,
            payload_schemas=schemas, clock=clock,
            object_admission_payload_validator=root._validate_object_admission_payload_record,
            current_candidate_citations=current_candidate_citations,
            hypotheses=hypothesis_store,
        )
        candidate_store._dispositions = dispositions
        candidate_store._service = service
        # Construct facades without replaying unused history. Their existing
        # transaction-bound producer readers prove exact dependencies on use.
        from ._native_startup import validate_current_heads
        with operation_lock, relationship_store._transaction():
            validate_current_heads(connection)

        def validate_retained_history():
            with operation_lock, relationship_store._transaction():
                root.validate_retained_history()
                root._reconcile_blob_integrity(_include_historical_active=True)
                root._validate_projection_integrity()
                work_items._verify_integrity()
                executions._TriageExecutionAuthority__store._verify_integrity()
                verified_dispositions = dispositions._verify_integrity()
                verified_hypotheses = hypothesis_store._verify(verified_dispositions=verified_dispositions)
                relationship_store._adopt()
                try:
                    _verify_relationship_event_coverage(connection)
                    inputs = relationship_store._verify_relationships(verified_versions=verified_hypotheses)
                    lineage_store._verify_global_event_coverage()
                    lineage_store._verify(relationship_inputs=inputs)
                    candidate_store._verify_global_event_coverage()
                    candidate_store._verify(relationship_receipts=inputs[1])
                finally:
                    relationship_store._release()

        transaction_candidate_port = _create_story_candidate_read_port(
            connection,
            retrieval_authority=retrieval_authority,
            authenticator=authenticator,
            command_registry=commands,
            payload_schemas=schemas,
            clock=clock,
            command_service_version=command_service_version,
            current_candidate_citations=current_candidate_citations,
            hypotheses=hypothesis_store,
            dispositions=dispositions,
            object_admission_payload_validator=root._validate_object_admission_payload_record,
        )

        def candidate_version(version_id: str):
            with operation_lock:
                connection.execute("BEGIN")
                try:
                    value = transaction_candidate_port.require_retained_version_in_transaction(
                        version_id
                    )
                    connection.execute("COMMIT")
                    return value
                except BaseException:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                    raise

        def candidate_versions(version_ids: tuple[str, ...]):
            with operation_lock:
                connection.execute("BEGIN")
                try:
                    values = transaction_candidate_port.require_retained_versions_in_transaction(
                        version_ids
                    )
                    connection.execute("COMMIT")
                    return values
                except Exception:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                    raise

        candidate_read_port = transaction_candidate_port._with_bounded_version(
            candidate_version
        )

        def receive_evidence_intake(ingress, **request):
            from newsroom.increment10.ingress import EvidenceIntakeIngress
            if type(ingress) is not EvidenceIntakeIngress:
                raise TypeError("native Evidence Intake receiver must be typed")
            # Keep the exact Candidate snapshot stable only for the local
            # durable handoff, never across source/model I/O.
            with operation_lock:
                connection.execute("BEGIN")
                try:
                    acknowledgement = ingress.receive(
                        transaction_candidate_port, **request
                    )
                    connection.execute("COMMIT")
                    return acknowledgement
                except BaseException:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                    raise

        return HermesNativeAuthoritySystem(
            _SYSTEM_TOKEN, base=base, checks=checks, discovery=discovery,
            work_items=_SharedAuthority(work_items, operation_lock),
            executions=_SharedAuthority(executions, operation_lock),
            proposals=_SharedAuthority(executions, operation_lock),
            dispositions=_SharedAuthority(dispositions, operation_lock),
            hypotheses=_compose_event_hypothesis_authority(_SharedAuthority(hypothesis_store, operation_lock)),
            hypothesis_relationships=_compose_event_hypothesis_relationship_authority(_SharedAuthority(relationship_store, operation_lock)),
            relationships=_compose_event_hypothesis_relationship_authority(_SharedAuthority(relationship_store, operation_lock)),
            hypothesis_lineage=_compose_event_hypothesis_lineage_authority(_SharedAuthority(lineage_store, operation_lock)),
            lineage=_compose_event_hypothesis_lineage_authority(_SharedAuthority(lineage_store, operation_lock)),
            candidates=_compose_story_candidate_authority(_SharedAuthority(candidate_store, operation_lock)),
            build_candidate_manifest=_SharedAuthority(candidate_store, operation_lock).build_manifest,
            candidate_read_port=candidate_read_port,
            candidate_version=candidate_version,
            candidate_versions=candidate_versions,
            receive_evidence_intake=receive_evidence_intake,
            collision=collision_enforcer,
            commands=authority_commands,
            events=authority_events,
            validate_retained_history=validate_retained_history,
        )
    except BaseException:
        base.close()
        raise


__all__ = [
    "HermesNativeAuthoritySystem",
    "NativeDependencyFactory",
    "open_hermes_native_authority_system",
]
