from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from newsroom.authority.canonical import canonical_json_bytes
from newsroom.authority.models import CommandDefinition
from newsroom.authority.policy import (
    CommandRegistry,
    PayloadGoldenVector,
    PayloadSchemaContract,
    PayloadSchemaRegistry,
    PayloadSchemaValidationError,
)
from newsroom.authority.types import PayloadMode, TrustScope

from .complete import CompleteProjectionContractRegistry
from .mapping import StructuralMappingRegistry
from .models import (
    GraphitiProposalWorkspaceContract,
    ProjectionContractError,
    ProjectionFamilyDefinition,
)
from .ontology import OntologyRegistry
from .registry import ProjectionFamilyRegistry


PROJECTION_COMMAND_TYPES = frozenset(
    {
        "projection.family.register",
        "projection.generation.create",
        "projection.generation.transition",
        "projection.generation.validate",
        "projection.generation.validate-current",
        "projection.generation.promote",
        "projection.generation.rebuild",
        "projection.delivery.record",
        "projection.gap.resolve",
    }
)


_PAYLOAD_SPECS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "projection_family_register_v1",
        frozenset({"family_id", "definition_digest"}),
    ),
    (
        "projection_generation_create_v1",
        frozenset({"generation_id", "family_id", "reason_code"}),
    ),
    (
        "projection_generation_transition_v1",
        frozenset(
            {
                "generation_id",
                "target_state",
                "validated_through_ledger_seq",
                "reason_code",
            }
        ),
    ),
    (
        "projection_generation_validate_v1",
        frozenset(
            {
                "generation_id",
                "checkpoint_ledger_seq",
                "service_compatibility_digest",
                "projection_state_digest",
                "reason_code",
            }
        ),
    ),
    (
        "projection_generation_validate_v2",
        frozenset(
            {
                "generation_id",
                "checkpoint_ledger_seq",
                "service_compatibility_digest",
                "projection_state_digest",
                "reason_code",
                "source_snapshot_digest",
                "source_watermark_ledger_seq",
                "source_request_digest",
            }
        ),
    ),
    (
        "projection_generation_promote_v1",
        frozenset(
            {
                "generation_id",
                "checkpoint_ledger_seq",
                "validation_digest",
                "prior_generation_id",
                "reason_code",
            }
        ),
    ),
    (
        "projection_generation_rebuild_v1",
        frozenset(
            {
                "generation_id",
                "through_ledger_seq",
                "reason_code",
            }
        ),
    ),
    (
        "projection_delivery_record_v1",
        frozenset(
            {
                "generation_id",
                "ledger_seq",
                "outcome",
                "error_code",
            }
        ),
    ),
    (
        "projection_gap_resolve_v1",
        frozenset({"generation_id", "gap_id", "reason_code"}),
    ),
)


_COMMAND_SPECS: tuple[tuple[str, str, str, str], ...] = (
    (
        "projection.family.register",
        "projection.family.registered",
        "projection_family_register_v1",
        "authority.projection.manage",
    ),
    (
        "projection.generation.create",
        "projection.generation.created",
        "projection_generation_create_v1",
        "authority.projection.manage",
    ),
    (
        "projection.generation.transition",
        "projection.generation.transitioned",
        "projection_generation_transition_v1",
        "authority.projection.manage",
    ),
    (
        "projection.generation.validate",
        "projection.generation.validated",
        "projection_generation_validate_v1",
        "authority.projection.manage",
    ),
    (
        "projection.generation.validate-current",
        "projection.generation.validated",
        "projection_generation_validate_v2",
        "authority.projection.manage",
    ),
    (
        "projection.generation.promote",
        "projection.generation.promoted",
        "projection_generation_promote_v1",
        "authority.projection.manage",
    ),
    (
        "projection.generation.rebuild",
        "projection.generation.rebuild.started",
        "projection_generation_rebuild_v1",
        "authority.projection.manage",
    ),
    (
        "projection.delivery.record",
        "projection.delivery.recorded",
        "projection_delivery_record_v1",
        "authority.projection.write",
    ),
    (
        "projection.gap.resolve",
        "projection.gap.resolved",
        "projection_gap_resolve_v1",
        "authority.projection.write",
    ),
)


def _canonicalizer(expected_keys: frozenset[str]):
    def canonicalize(value: Any) -> bytes:
        if not isinstance(value, dict) or set(value) != set(expected_keys):
            raise PayloadSchemaValidationError(
                "projection payload fields differ from retained schema"
            )
        return canonical_json_bytes(value)

    return canonicalize


def projection_payload_contracts() -> tuple[PayloadSchemaContract, ...]:
    contracts: list[PayloadSchemaContract] = []
    for schema_version, keys in _PAYLOAD_SPECS:
        vector_value = {key: None for key in sorted(keys)}
        canonicalizer = _canonicalizer(keys)
        contracts.append(
            PayloadSchemaContract(
                schema_version=schema_version,
                payload_mode=PayloadMode.INLINE,
                contract_version=(
                    "projection-schema-v2"
                    if schema_version == "projection_generation_validate_v2"
                    else "projection-schema-v1"
                ),
                canonicalizer_implementation_version=(
                    "projection-canonical-json-v1"
                ),
                canonicalizer=canonicalizer,
                golden_vectors=(
                    PayloadGoldenVector(
                        name="exact-fields",
                        input_identity=f"{schema_version}:exact-fields",
                        value=vector_value,
                        expected_bytes=canonicalizer(vector_value),
                    ),
                ),
            )
        )
    return tuple(contracts)


def projection_command_definitions() -> tuple[CommandDefinition, ...]:
    contracts = {
        item.schema_version: item for item in projection_payload_contracts()
    }
    definitions: list[CommandDefinition] = []
    for command_type, event_type, schema_version, required_scope in _COMMAND_SPECS:
        contract = contracts[schema_version]
        aggregate_type = (
            "projection_family"
            if command_type == "projection.family.register"
            else "projection_generation"
        )
        definitions.append(
            CommandDefinition(
                command_type=command_type,
                definition_version=(
                    "projection-command-v2"
                    if command_type == "projection.generation.validate-current"
                    else "projection-command-v1"
                ),
                aggregate_type=aggregate_type,
                event_type=event_type,
                event_schema_version=1,
                payload_mode=PayloadMode.INLINE,
                payload_schema_version=contract.schema_version,
                payload_schema_contract_version=contract.contract_version,
                payload_schema_contract_digest=contract.contract_digest,
                payload_canonicalizer_version=(
                    contract.canonicalizer_implementation_version
                ),
                trust_scope=TrustScope.ADMITTED,
                security_scope="authority.projection",
                retention_scope="authority.audit",
                required_scope=required_scope,
                max_inline_bytes=32 * 1024,
            )
        )
    return tuple(definitions)


def merge_projection_authority_registries(
    *,
    command_registry: CommandRegistry,
    payload_schemas: PayloadSchemaRegistry,
) -> tuple[CommandRegistry, PayloadSchemaRegistry]:
    definitions = list(command_registry.definitions())
    by_key = {
        (item.command_type, item.definition_version): item
        for item in definitions
    }
    for definition in projection_command_definitions():
        key = (definition.command_type, definition.definition_version)
        existing = by_key.get(key)
        if existing is not None and existing.digest != definition.digest:
            raise ProjectionContractError(
                f"projection command identity conflict: {definition.command_type}"
            )
        if existing is None:
            definitions.append(definition)
            by_key[key] = definition
    current_commands: dict[str, str] = {}
    for command_type in {item.command_type for item in definitions}:
        if command_type in PROJECTION_COMMAND_TYPES:
            current_commands[command_type] = (
                "projection-command-v2"
                if command_type == "projection.generation.validate-current"
                else "projection-command-v1"
            )
        else:
            current_commands[command_type] = command_registry.resolve(
                command_type
            ).definition_version

    contracts = list(payload_schemas.contracts())
    schema_keys = {
        (item.schema_version, item.payload_mode, item.contract_version): item
        for item in contracts
    }
    projection_contracts = projection_payload_contracts()
    for contract in projection_contracts:
        key = (
            contract.schema_version,
            contract.payload_mode,
            contract.contract_version,
        )
        existing = schema_keys.get(key)
        if (
            existing is not None
            and existing.contract_digest != contract.contract_digest
        ):
            raise ProjectionContractError(
                f"projection payload identity conflict: {contract.schema_version}"
            )
        if existing is None:
            contracts.append(contract)
            schema_keys[key] = contract
    projection_schema_versions = {
        item.schema_version for item in projection_contracts
    }
    current_schemas: dict[tuple[str, PayloadMode], str] = {}
    for schema_version, mode in {
        (item.schema_version, item.payload_mode) for item in contracts
    }:
        if schema_version in projection_schema_versions:
            current_schemas[(schema_version, mode)] = (
                "projection-schema-v2"
                if schema_version == "projection_generation_validate_v2"
                else "projection-schema-v1"
            )
        else:
            current_schemas[(schema_version, mode)] = payload_schemas.resolve(
                schema_version, mode
            ).contract_version
    return (
        CommandRegistry(definitions, current_versions=current_commands),
        PayloadSchemaRegistry(contracts, current_versions=current_schemas),
    )


@dataclass(frozen=True, slots=True)
class ProjectionContractRegistry:
    ontologies: OntologyRegistry
    mappings: StructuralMappingRegistry
    families: ProjectionFamilyRegistry
    graphiti_workspaces: tuple[GraphitiProposalWorkspaceContract, ...] = ()
    complete_projections: CompleteProjectionContractRegistry | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.graphiti_workspaces, tuple):
            raise ProjectionContractError(
                "Graphiti workspace contracts must be an immutable tuple"
            )
        keys = [
            (item.workspace_id, item.contract_version)
            for item in self.graphiti_workspaces
        ]
        if len(keys) != len(set(keys)):
            raise ProjectionContractError(
                "Graphiti workspace contracts must be unique"
            )
        if self.complete_projections is not None and not isinstance(
            self.complete_projections, CompleteProjectionContractRegistry
        ):
            raise ProjectionContractError(
                "complete projection contracts must use the typed registry"
            )

    def family(self, family_id: str) -> ProjectionFamilyDefinition:
        return self.families.resolve(family_id)

    def graphiti_contracts(
        self,
    ) -> tuple[GraphitiProposalWorkspaceContract, ...]:
        return tuple(
            sorted(
                self.graphiti_workspaces,
                key=lambda item: (item.workspace_id, item.contract_version),
            )
        )


__all__ = [
    "PROJECTION_COMMAND_TYPES",
    "ProjectionContractRegistry",
    "merge_projection_authority_registries",
    "projection_command_definitions",
    "projection_payload_contracts",
]
