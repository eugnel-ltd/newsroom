from __future__ import annotations

from dataclasses import replace

import pytest

from newsroom.authority import AggregateId, PayloadMode
from newsroom.projection import (
    GraphitiProposalWorkspaceContract,
    GraphitiWorkspaceMode,
    OntologyContract,
    OntologyRegistry,
    ProjectionContractError,
    ProjectionFamilyDefinition,
    ProjectionFamilyKind,
    ProjectionFamilyRegistry,
    ProjectionIdentitySource,
    StructuralIdentityContext,
    canonical_node_id,
    ProjectionNodeType,
    ProjectionRelationType,
    StructuralEventMapping,
    StructuralMappingContract,
    StructuralMappingRegistry,
    StructuralNodeBinding,
    StructuralRelationBinding,
    native_ontology_v1,
    native_structural_mapping_v1,
)
from newsroom.projection.policy import (
    PROJECTION_COMMAND_TYPES,
    ProjectionContractRegistry,
    merge_projection_authority_registries,
    projection_command_definitions,
    projection_payload_contracts,
)

from .authority_event_helpers import payload_schemas, registry_v1


FAMILY_AGGREGATE = AggregateId.parse("11111111-1111-4111-8111-111111111111")


def family_definition(*, version: str = "family-v1") -> ProjectionFamilyDefinition:
    ontology = native_ontology_v1()
    mapping = native_structural_mapping_v1(ontology)
    return ProjectionFamilyDefinition(
        family_id="graph.structural",
        authority_aggregate_id=FAMILY_AGGREGATE,
        family_kind=ProjectionFamilyKind.GRAPH,
        definition_version=version,
        projector_version="structural-projector-v1",
        ontology_contract_digest=ontology.contract_digest,
        mapping_contract_digest=mapping.contract_digest,
    )


def test_native_ontology_is_stable_and_rejects_generic_relations() -> None:
    ontology = native_ontology_v1()
    assert ontology.contract_digest == native_ontology_v1().contract_digest
    assert ProjectionRelationType.PROJECTED_FROM_EVENT in ontology.relation_types
    assert "RELATED_TO" not in {item.value for item in ontology.relation_types}
    assert all(item.name == item.value for item in ProjectionRelationType)


def test_mapping_is_allow_listed_deterministic_and_validated() -> None:
    ontology = native_ontology_v1()
    mapping = native_structural_mapping_v1(ontology)
    mapping.validate_against(ontology)
    revised = mapping.resolve("source.item.revised")
    assert revised is not None
    assert any(
        node.identity_source is ProjectionIdentitySource.PAYLOAD_FIELD
        and node.payload_field == "source_item_id"
        for node in revised.nodes
    )
    assert mapping.resolve("caller.related") is None

    changed = replace(ontology, ontology_version="ontology-v2")
    with pytest.raises(ProjectionContractError, match="ontology digest"):
        mapping.validate_against(changed)


def test_mapping_rejects_unknown_alias_and_relation_direction() -> None:
    ontology = native_ontology_v1()
    with pytest.raises(ProjectionContractError, match="unknown node alias"):
        StructuralEventMapping(
            event_type="bad.alias",
            required=True,
            nodes=(
                StructuralNodeBinding(
                    "item",
                    ProjectionNodeType.SOURCE_ITEM,
                    ProjectionIdentitySource.AGGREGATE,
                ),
            ),
            relations=(
                StructuralRelationBinding(
                    ProjectionRelationType.HAS_VERSION,
                    "item",
                    "missing",
                ),
            ),
        )

    reversed_mapping = StructuralMappingContract(
        mapping_id="bad.direction",
        mapping_version="mapping-v1",
        implementation_version="mapping-python-v1",
        ontology_contract_digest=ontology.contract_digest,
        mappings=(
            StructuralEventMapping(
                event_type="bad.direction",
                required=True,
                nodes=(
                    StructuralNodeBinding(
                        "version",
                        ProjectionNodeType.AUTHORITY_VERSION,
                        ProjectionIdentitySource.AGGREGATE_VERSION,
                    ),
                    StructuralNodeBinding(
                        "aggregate",
                        ProjectionNodeType.AUTHORITY_AGGREGATE,
                        ProjectionIdentitySource.AGGREGATE,
                    ),
                ),
                relations=(
                    StructuralRelationBinding(
                        ProjectionRelationType.HAS_VERSION,
                        "version",
                        "aggregate",
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ProjectionContractError, match="source type"):
        reversed_mapping.validate_against(ontology)


def test_family_registry_binds_exact_contracts_and_retains_versions() -> None:
    ontology = native_ontology_v1()
    mapping = native_structural_mapping_v1(ontology)
    v1 = family_definition()
    v2 = replace(v1, definition_version="family-v2", projector_version="structural-projector-v2")
    registry = ProjectionFamilyRegistry(
        (v1, v2),
        ontologies=OntologyRegistry((ontology,)),
        mappings=StructuralMappingRegistry((mapping,)),
        current_versions={"graph.structural": "family-v2"},
    )
    assert registry.resolve("graph.structural").definition_version == "family-v2"
    assert registry.resolve("graph.structural", "family-v1").digest == v1.digest

    cross_family = replace(v1, family_id="vector.semantic", family_kind=ProjectionFamilyKind.VECTOR)
    with pytest.raises(ProjectionContractError, match="shared across families"):
        ProjectionFamilyRegistry(
            (v1, cross_family),
            ontologies=OntologyRegistry((ontology,)),
            mappings=StructuralMappingRegistry((mapping,)),
        )


def test_projection_authority_contracts_merge_without_caller_semantics() -> None:
    registry, schemas = merge_projection_authority_registries(
        command_registry=registry_v1(), payload_schemas=payload_schemas()
    )
    assert PROJECTION_COMMAND_TYPES <= {
        item.command_type for item in registry.definitions()
    }
    for definition in projection_command_definitions():
        selected = registry.resolve_exact(
            definition.command_type,
            definition.definition_version,
            definition.digest,
        )
        assert selected.trust_scope.value == "ADMITTED"
        assert selected.security_scope == "authority.projection"
        assert selected.payload_mode is PayloadMode.INLINE
    for contract in projection_payload_contracts():
        assert schemas.resolve_exact(
            contract.schema_version,
            contract.payload_mode,
            contract.contract_version,
            contract.contract_digest,
            contract.canonicalizer_implementation_version,
        ) is not None
    assert registry.resolve("projection.generation.validate").definition_version == (
        "projection-command-v1"
    )
    current = registry.resolve("projection.generation.validate-current")
    assert current.definition_version == "projection-command-v2"
    assert current.payload_schema_version == "projection_generation_validate_v2"


def test_graphiti_contract_is_proposal_only_and_has_no_execution_api() -> None:
    contract = GraphitiProposalWorkspaceContract(
        workspace_id="graphiti.proposals",
        contract_version="graphiti-workspace-v1",
        implementation_version="graphiti-seam-v1",
        endpoint_reference="config://graphiti/proposal-endpoint",
        secret_reference="secret://graphiti/proposal-token",
    )
    assert contract.mode is GraphitiWorkspaceMode.PROPOSAL_ONLY
    assert not hasattr(contract, "execute")
    assert not hasattr(contract, "write_graph")

    ontology = native_ontology_v1()
    mapping = native_structural_mapping_v1(ontology)
    contracts = ProjectionContractRegistry(
        ontologies=OntologyRegistry((ontology,)),
        mappings=StructuralMappingRegistry((mapping,)),
        families=ProjectionFamilyRegistry(
            (family_definition(),),
            ontologies=OntologyRegistry((ontology,)),
            mappings=StructuralMappingRegistry((mapping,)),
        ),
        graphiti_workspaces=(contract,),
    )
    assert contracts.graphiti_contracts() == (contract,)


def test_canonical_engine_neutral_node_ids_are_stable_and_typed() -> None:
    context = StructuralIdentityContext(
        aggregate_type="source_item",
        aggregate_id="11111111-1111-4111-8111-111111111111",
        aggregate_version=2,
        event_id="22222222-2222-4222-8222-222222222222",
        payload_id="33333333-3333-4333-8333-333333333333",
        payload={"source_item_id": "item-123"},
    )
    aggregate = StructuralNodeBinding(
        "item",
        ProjectionNodeType.SOURCE_ITEM,
        ProjectionIdentitySource.AGGREGATE,
    )
    first = canonical_node_id(aggregate, context)
    assert first == canonical_node_id(aggregate, context)
    assert first.startswith("npid:v1:source_item:")

    version = StructuralNodeBinding(
        "version",
        ProjectionNodeType.AUTHORITY_VERSION,
        ProjectionIdentitySource.AGGREGATE_VERSION,
    )
    assert canonical_node_id(version, context) != first

    payload_field = StructuralNodeBinding(
        "item",
        ProjectionNodeType.SOURCE_ITEM,
        ProjectionIdentitySource.PAYLOAD_FIELD,
        "source_item_id",
    )
    assert canonical_node_id(payload_field, context) != first
    with pytest.raises(ProjectionContractError, match="field is absent"):
        canonical_node_id(
            payload_field,
            replace(context, payload={}),
        )
