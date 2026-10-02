from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import RLock
from typing import Any

from newsroom.authority.canonical import digest_canonical, validate_sha256_digest
from newsroom.entities.types import EntityReadPolicy
from newsroom.extraction.producer import DeterministicFixtureExtractor
from newsroom.extraction.types import ExtractionReadPolicy
from newsroom.graphiti_adapter import GraphitiAdapterReadPolicy
from newsroom.graphiti_adapter.policy import merge_graphiti_adapter_authority_registries
from newsroom.increment4.neo4j import Increment4Neo4jController
from newsroom.projection.models import ProjectionReadPolicy
from newsroom.projection.neo4j.models import Neo4jCompatibility, Neo4jProjectorConfig
from newsroom.projection.policy import (
    ProjectionContractRegistry,
    merge_projection_authority_registries,
)
from newsroom.relations.editorial_models import EditorialRelationReadPolicy
from newsroom.sources.policy import (
    SOURCE_REGISTRY_COMMAND_TYPES,
    merge_source_registry_authority_registries,
)
from newsroom.sources.types import SourceRegistryReadPolicy

from ._capability import _CapabilityIssuer
from ._check_store_support import _CHECK_RECORD_SPECS
from ._discovery_store import _DISCOVERY_RECORD_SPECS, _DiscoveryAuthorityStore
from ._editorial_relation_boundary import _EditorialRelationBoundary
from ._editorial_relation_facade import GovernedEditorialRelations
from ._entity_boundary import _EntityBoundary
from ._entity_facade import GovernedEntityRecords
from ._entity_store_common import _EntityStoreSupport
from ._extraction_boundary import _ExtractionBoundary
from ._extraction_facade import GovernedExtractionRecords
from ._extraction_store_common import _ExtractionStoreSupport
from ._graphiti_adapter_boundary import _GraphitiAdapterBoundary
from ._graphiti_adapter_facade import GovernedGraphitiProposalAdapter
from ._graphiti_adapter_store import _GraphitiAdapterAuthorityStore
from ._increment4_neo4j_boundary import _Increment4Neo4jBoundary
from ._increment4_projection_store import _Increment4ProjectionAuthorityStore
from ._neo4j_projection_system import (
    Neo4jStructuralProjector,
    _Neo4jProjectionBoundary,
    _StructuralGraphAdapter,
    _open_structural_graph_adapter,
)
from ._object_capability import _ObjectCapabilityIssuer
from ._object_cas import _GovernedCAS
from ._object_store import _GovernedObjectAuthorityStore
from ._object_system import GovernedObjects, _ObjectBoundary
from ._projection_system import _ProjectionBoundary
from ._source_registry_store_common import _SourceRegistryStoreSupport
from ._source_registry_system import GovernedSources, _SourceRegistryBoundary
from .object_policy import (
    HydrationPolicyRegistry,
    ObjectAdmissionRegistry,
    RightsPolicyRegistry,
    merge_authority_registries,
)
from .objects import ObjectLimits
from .policy import CommandRegistry, PayloadSchemaRegistry
from .service import CommandService
from .types import UtcTimestamp


class _GraphitiIncrement4AuthorityStore(
    _DiscoveryAuthorityStore,
    _GraphitiAdapterAuthorityStore,
    _Increment4ProjectionAuthorityStore,
    _GovernedObjectAuthorityStore,
):
    """One existing authority SQLite/CAS writer for Increment 4."""

    def __init__(self, *args, _native_current_state: bool = False, **kwargs):
        self._current_state_only = _native_current_state
        super().__init__(*args, **kwargs)

    def _signal_from_row(self, conn, row, *, replayed):
        self._prove_current_record(row)
        return _DiscoveryAuthorityStore._signal_from_row.__func__(type(self), conn, row, replayed=replayed)

    def _gate_from_row(self, conn, row, *, replayed):
        self._prove_current_record(row)
        return _DiscoveryAuthorityStore._gate_from_row.__func__(type(self), conn, row, replayed=replayed)

    def _lead_from_row(self, conn, row, *, replayed):
        self._prove_current_record(row)
        return _DiscoveryAuthorityStore._lead_from_row.__func__(type(self), conn, row, replayed=replayed)

    def _watch_from_row(self, conn, row, *, replayed):
        self._prove_current_record(row)
        return _DiscoveryAuthorityStore._watch_from_row.__func__(type(self), conn, row, replayed=replayed)

    def _disposition_from_row(self, conn, row, *, replayed):
        self._prove_current_record(row)
        return _DiscoveryAuthorityStore._disposition_from_row.__func__(type(self), conn, row, replayed=replayed)

    _SOURCE_TABLES = frozenset(
        {
            "discovery_occurrences",
            "discovery_representations",
            "source_locator_continuity_decisions",
            "source_definition_versions",
            "source_definitions",
            "source_items",
            "source_revisions",
        }
    )

    @staticmethod
    def _ensure_identifier_absent(
        conn: sqlite3.Connection,
        *,
        table: str,
        column: str,
        identifier: str,
        identity: str,
    ) -> None:
        support = (
            _SourceRegistryStoreSupport
            if table in _GraphitiIncrement4AuthorityStore._SOURCE_TABLES
            else _EntityStoreSupport
        )
        support._ensure_identifier_absent(
            conn,
            table=table,
            column=column,
            identifier=identifier,
            identity=identity,
        )

    @staticmethod
    def _ensure_semantic_absent(
        conn: sqlite3.Connection,
        *,
        table: str,
        identity: str,
        column: str | None = None,
        digest: str | None = None,
        predicate: str | None = None,
        parameters: tuple[object, ...] | None = None,
    ) -> None:
        if table in _GraphitiIncrement4AuthorityStore._SOURCE_TABLES:
            if predicate is None or parameters is None:
                raise TypeError("source semantic check requires a predicate")
            _SourceRegistryStoreSupport._ensure_semantic_absent(
                conn,
                table=table,
                predicate=predicate,
                parameters=parameters,
                identity=identity,
            )
            return
        if column is None or digest is None:
            raise TypeError("authority semantic check requires a digest column")
        _EntityStoreSupport._ensure_semantic_absent(
            conn,
            table=table,
            column=column,
            digest=digest,
            identity=identity,
        )

    @classmethod
    def _validate_record_envelope(
        cls,
        conn: sqlite3.Connection,
        row: Mapping[str, Any],
        *,
        command_type: str,
        aggregate_id: str,
        canonical_bytes: bytes,
        canonical_digest: str,
    ) -> sqlite3.Row:
        if command_type in _CHECK_RECORD_SPECS or command_type in _DISCOVERY_RECORD_SPECS:
            return _DiscoveryAuthorityStore._validate_record_envelope.__func__(
                cls,
                conn,
                row,
                command_type=command_type,
                aggregate_id=aggregate_id,
                canonical_bytes=canonical_bytes,
                canonical_digest=canonical_digest,
            )
        support = (
            _SourceRegistryStoreSupport
            if command_type in SOURCE_REGISTRY_COMMAND_TYPES
            else _ExtractionStoreSupport
        )
        return support._validate_record_envelope(
            conn,
            row,
            command_type=command_type,
            aggregate_id=aggregate_id,
            canonical_bytes=canonical_bytes,
            canonical_digest=canonical_digest,
        )


_SYSTEM_CONSTRUCTION_TOKEN = object()
_AUTHORITY_COMPOSITION_TOKEN = object()


class GovernedGraphitiIncrement4AuthoritySystem:
    __slots__ = (
        "__graphiti",
        "__sources",
        "__extraction",
        "__objects",
        "__entities",
        "__relations",
        "__increment4",
        "__structural",
        "__compatibility",
        "__authority_store_path",
        "__graph_destination_id",
        "__close",
        "__composition",
    )

    def __init__(
        self,
        *,
        graphiti: GovernedGraphitiProposalAdapter,
        sources: GovernedSources,
        extraction: GovernedExtractionRecords,
        objects: GovernedObjects,
        entities: GovernedEntityRecords,
        relations: GovernedEditorialRelations,
        increment4: Increment4Neo4jController,
        structural: Neo4jStructuralProjector,
        compatibility: Neo4jCompatibility,
        authority_store_path: Path,
        graph_destination_id: str,
        close: Callable[[], None],
        composition: tuple[object, ...] = (),
        _construction_token: object,
    ) -> None:
        if _construction_token is not _SYSTEM_CONSTRUCTION_TOKEN:
            raise TypeError("combined Increment 4 authority systems require the opener")
        self.__graphiti = graphiti
        self.__sources = sources
        self.__extraction = extraction
        self.__objects = objects
        self.__entities = entities
        self.__relations = relations
        self.__increment4 = increment4
        self.__structural = structural
        self.__compatibility = compatibility
        self.__authority_store_path = str(authority_store_path.expanduser().resolve())
        validate_sha256_digest(
            graph_destination_id,
            field="combined Increment 4 graph destination identity",
        )
        self.__graph_destination_id = graph_destination_id
        self.__close = close
        self.__composition = composition

    def _authority_composition(self, token: object) -> tuple[object, ...]:
        """Return authority-private components for the cumulative Hermes opener."""

        if token is not _AUTHORITY_COMPOSITION_TOKEN:
            raise TypeError("authority composition access is private")
        return self.__composition

    @property
    def graphiti(self) -> GovernedGraphitiProposalAdapter:
        return self.__graphiti

    @property
    def sources(self) -> GovernedSources:
        return self.__sources

    @property
    def extraction(self) -> GovernedExtractionRecords:
        return self.__extraction

    @property
    def objects(self) -> GovernedObjects:
        return self.__objects

    @property
    def entities(self) -> GovernedEntityRecords:
        return self.__entities

    @property
    def relations(self) -> GovernedEditorialRelations:
        return self.__relations

    @property
    def increment4(self) -> Increment4Neo4jController:
        return self.__increment4

    @property
    def structural(self) -> Neo4jStructuralProjector:
        return self.__structural

    @property
    def compatibility(self) -> Neo4jCompatibility:
        return self.__compatibility

    @property
    def authority_store_path(self) -> str:
        """Return the exact backing authority-store identity for runtime fencing."""

        return self.__authority_store_path

    @property
    def graph_destination_id(self) -> str:
        """Return the credential-free graph configuration identity."""

        return self.__graph_destination_id

    def close(self) -> None:
        self.__close()

    def __enter__(self) -> GovernedGraphitiIncrement4AuthoritySystem:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _open_with_adapter(
    *,
    path: Path,
    object_root: Path,
    workspace_root: Path,
    registry: CommandRegistry,
    payload_schemas: PayloadSchemaRegistry,
    admission_registry: ObjectAdmissionRegistry,
    rights_policies: RightsPolicyRegistry,
    hydration_policies: HydrationPolicyRegistry,
    contracts: ProjectionContractRegistry,
    authenticator: Any,
    authorizer: Any,
    source_read_policy: SourceRegistryReadPolicy,
    extraction_read_policy: ExtractionReadPolicy,
    entity_read_policy: EntityReadPolicy,
    relation_read_policy: EditorialRelationReadPolicy,
    graphiti_read_policy: GraphitiAdapterReadPolicy,
    projection_read_policy: ProjectionReadPolicy,
    object_limits: ObjectLimits,
    adapter: _StructuralGraphAdapter,
    graph_destination_id: str,
    command_service_version: str = "authority-command-v1",
    busy_timeout_ms: int = 5_000,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
    cas_fault_hook: Callable[[str], None] | None = None,
    disk_usage: Callable[[Path], Any] | None = None,
    _native_current_state: bool = False,
) -> GovernedGraphitiIncrement4AuthoritySystem:
    store: _GraphitiIncrement4AuthorityStore | None = None
    try:
        merged_registry, merged_schemas = merge_source_registry_authority_registries(
            command_registry=registry,
            payload_schemas=payload_schemas,
        )
        merged_registry, merged_schemas = merge_authority_registries(
            command_registry=merged_registry,
            payload_schemas=merged_schemas,
        )
        # These merges are cumulative. Keep the explicit order aligned with the
        # retained Source -> Object -> 4A/4B/4C/4D -> 4E authority dependencies.
        merged_registry, merged_schemas = merge_graphiti_adapter_authority_registries(
            command_registry=merged_registry,
            payload_schemas=merged_schemas,
        )
        merged_registry, merged_schemas = merge_projection_authority_registries(
            command_registry=merged_registry,
            payload_schemas=merged_schemas,
        )
        issuer = _CapabilityIssuer(
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
        compatibility = adapter.verify_compatibility()
        adapter.bootstrap_schema()
        store = _GraphitiIncrement4AuthorityStore(
            path,
            workspace_root=workspace_root,
            issuer=issuer,
            object_issuer=object_issuer,
            command_registry=merged_registry,
            payload_schemas=merged_schemas,
            admission_registry=admission_registry,
            rights_policies=rights_policies,
            hydration_policies=hydration_policies,
            cas=cas,
            contracts=contracts,
            command_service_version=command_service_version,
            busy_timeout_ms=busy_timeout_ms,
            clock=clock,
            _native_current_state=_native_current_state,
        )
        command_service = CommandService(
            registry=merged_registry,
            payload_schemas=merged_schemas,
            authenticator=authenticator,
            authorizer=authorizer,
            admission_lookup=store,
            committed_lookup=store,
            clock=clock,
            _issuer=issuer,
        )
        object_boundary = _ObjectBoundary(
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
        source_boundary = _SourceRegistryBoundary(
            store=store,
            command_service=command_service,
            authenticator=authenticator,
            authorizer=authorizer,
            read_policy=source_read_policy,
            clock=clock,
        )
        extraction_boundary = _ExtractionBoundary(
            store=store,
            command_service=command_service,
            authenticator=authenticator,
            authorizer=authorizer,
            read_policy=extraction_read_policy,
            producer=DeterministicFixtureExtractor(),
            clock=clock,
        )
        entity_boundary = _EntityBoundary(
            store=store,
            command_service=command_service,
            authenticator=authenticator,
            authorizer=authorizer,
            read_policy=entity_read_policy,
            clock=clock,
        )
        relation_boundary = _EditorialRelationBoundary(
            store=store,
            command_service=command_service,
            authenticator=authenticator,
            authorizer=authorizer,
            read_policy=relation_read_policy,
            clock=clock,
        )
        graphiti_boundary = _GraphitiAdapterBoundary(
            store=store,
            command_service=command_service,
            authenticator=authenticator,
            authorizer=authorizer,
            read_policy=graphiti_read_policy,
            workspace_root=workspace_root,
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

        closed = False

        def close() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            try:
                adapter.close()
            finally:
                assert store is not None
                store.close()

        return GovernedGraphitiIncrement4AuthoritySystem(
            sources=GovernedSources(
                register_definition=source_boundary.register_definition,
                record_version=source_boundary.record_version,
                register_item=source_boundary.register_item,
                decide_locator=source_boundary.decide_locator,
                record_revision=source_boundary.record_revision,
                record_representation=source_boundary.record_representation,
                record_occurrence=source_boundary.record_occurrence,
                definition=source_boundary.definition,
                current_summary=source_boundary.current_summary,
                version_details=source_boundary.version_details,
                item=source_boundary.item,
                revision=source_boundary.revision,
                latest_revision=source_boundary.latest_revision,
                representation=source_boundary.representation,
                occurrences=source_boundary.occurrences,
            ),
            graphiti=GovernedGraphitiProposalAdapter(
                register_configuration=graphiti_boundary.register_configuration,
                execute_attempt=graphiti_boundary.execute_attempt,
                approve_replay=graphiti_boundary.approve_replay,
                configuration=graphiti_boundary.configuration,
                attempt=graphiti_boundary.attempt,
                attempt_history=graphiti_boundary.attempt_history,
                manifest_for_attempt=graphiti_boundary.manifest_for_attempt,
                replay_source=graphiti_boundary.replay_source,
            ),
            extraction=GovernedExtractionRecords(
                register_contract=extraction_boundary.register_contract,
                execute=extraction_boundary.execute,
                contract=extraction_boundary.contract,
                metadata=extraction_boundary.metadata,
                run_history=extraction_boundary.run_history,
                proposals=extraction_boundary.proposals,
                raw_output=extraction_boundary.raw_output,
            ),
            objects=GovernedObjects(
                admit=object_boundary.admit,
                committed_admission=object_boundary.committed_admission,
                hydrate=object_boundary.hydrate,
                rehydrate=object_boundary.rehydrate,
                access_decision=object_boundary.access_decision,
                latest_access_decision=(
                    object_boundary.latest_access_decision
                ),
                revoke=object_boundary.revoke,
                request_deletion=object_boundary.request_deletion,
                tombstone=object_boundary.tombstone,
                complete_deletion=object_boundary.complete_deletion,
                create_pin=object_boundary.create_pin,
                release_pin=object_boundary.release_pin,
                collect_orphans=object_boundary.collect_orphans,
            ),
            entities=GovernedEntityRecords(
                admit_mention=entity_boundary.admit_mention,
                propose_resolution=entity_boundary.propose_resolution,
                decide_resolution=entity_boundary.decide_resolution,
                bind_resolution_dependency=entity_boundary.bind_resolution_dependency,
                merge_entities=entity_boundary.merge_entities,
                split_entity=entity_boundary.split_entity,
                reverse_lineage=entity_boundary.reverse_lineage,
                mention=entity_boundary.mention,
                proposal=entity_boundary.proposal,
                proposal_version=entity_boundary.proposal_version,
                decision=entity_boundary.decision,
                entity=entity_boundary.entity,
                entity_version=entity_boundary.entity_version,
                aliases=entity_boundary.aliases,
                preferred=entity_boundary.preferred,
                projection_events_after=entity_boundary.projection_events_after,
                admission_guard=entity_boundary.admission_guard,
                dependency=entity_boundary.dependency,
                dependent_admission_guard=entity_boundary.dependent_admission_guard,
                merge_decision=entity_boundary.merge_decision,
                split_decision=entity_boundary.split_decision,
                reversal_decision=entity_boundary.reversal_decision,
            ),
            relations=GovernedEditorialRelations(
                propose=relation_boundary.propose,
                decide=relation_boundary.decide,
                proposal=relation_boundary.proposal,
                proposal_version=relation_boundary.proposal_version,
                decision=relation_boundary.decision,
                assertion=relation_boundary.assertion,
                current=relation_boundary.current,
                current_relations=relation_boundary.current_relations,
                projection_events_after=relation_boundary.projection_events_after,
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
            compatibility=compatibility,
            authority_store_path=path,
            graph_destination_id=graph_destination_id,
            close=close,
            composition=(store, command_service, cas, issuer, merged_registry, merged_schemas),
            _construction_token=_SYSTEM_CONSTRUCTION_TOKEN,
        )
    except Exception:
        try:
            adapter.close()
        finally:
            if store is not None:
                store.close()
        raise


def open_governed_graphiti_increment4_authority_system(
    *,
    path: Path,
    object_root: Path,
    workspace_root: Path,
    registry: CommandRegistry,
    payload_schemas: PayloadSchemaRegistry,
    admission_registry: ObjectAdmissionRegistry,
    rights_policies: RightsPolicyRegistry,
    hydration_policies: HydrationPolicyRegistry,
    contracts: ProjectionContractRegistry,
    authenticator: Any,
    authorizer: Any,
    source_read_policy: SourceRegistryReadPolicy,
    extraction_read_policy: ExtractionReadPolicy,
    entity_read_policy: EntityReadPolicy,
    relation_read_policy: EditorialRelationReadPolicy,
    graphiti_read_policy: GraphitiAdapterReadPolicy,
    projection_read_policy: ProjectionReadPolicy,
    object_limits: ObjectLimits,
    neo4j_config: Neo4jProjectorConfig,
    command_service_version: str = "authority-command-v1",
    busy_timeout_ms: int = 5_000,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
    cas_fault_hook: Callable[[str], None] | None = None,
    disk_usage: Callable[[Path], Any] | None = None,
    _native_current_state: bool = False,
) -> GovernedGraphitiIncrement4AuthoritySystem:
    """Open the explicit, effectful Increment 4 authority and graph runtime."""

    graph_destination_id = digest_canonical(
        {
            "schema": "newsroom.increment4-graph-destination.v1",
            "uri": neo4j_config.uri,
            "database": neo4j_config.database,
            "username": neo4j_config.username,
        }
    )
    adapter = _open_structural_graph_adapter(neo4j_config)
    return _open_with_adapter(
        path=path,
        object_root=object_root,
        workspace_root=workspace_root,
        registry=registry,
        payload_schemas=payload_schemas,
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
        adapter=adapter,
        graph_destination_id=graph_destination_id,
        command_service_version=command_service_version,
        busy_timeout_ms=busy_timeout_ms,
        clock=clock,
        cas_fault_hook=cas_fault_hook,
        disk_usage=disk_usage,
        _native_current_state=_native_current_state,
    )


__all__ = [
    "GovernedGraphitiIncrement4AuthoritySystem",
    "open_governed_graphiti_increment4_authority_system",
]
