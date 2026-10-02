from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import sqlite3

from newsroom.authority import (
    AggregateId,
    CommandDefinition,
    CommandRegistry,
    ObjectAdmissionDescriptor,
    ObjectAdmissionId,
    ObjectAdmissionPayload,
    PayloadGoldenVector,
    PayloadMode,
    PayloadSchemaContract,
    PayloadSchemaRegistry,
    PayloadSchemaValidationError,
    SemanticCommand,
    StaticAuthenticator,
    StaticAuthorizer,
    StaticPrincipal,
    TrustScope,
    canonical_json_bytes,
)
from newsroom.authority._neo4j_projection_system import _open_with_adapter
from newsroom.authority.neo4j_projection_system import (
    open_neo4j_projection_authority_system,
)
from newsroom.authority.object_policy import merge_authority_registries
from newsroom.authority.persistence import LedgerEventRecord
from newsroom.projection.neo4j import (
    Neo4jApplyResult,
    Neo4jProjectorConfig,
    StructuralBatch,
)

from .authority_a2b_helpers import admit, open_object_system
from .authority_event_helpers import payload_schemas
from .projection_b1_helpers import (
    event_read_policy,
    projection_contracts,
    projection_read_policy,
    proof,
    source_command_registry,
)
from .projection_b2_helpers import MemoryNeo4jAdapter


@dataclass(frozen=True, slots=True)
class TombstonedStructuralHistory:
    admission_id: ObjectAdmissionId
    blob_digest: str
    source_event: LedgerEventRecord
    tombstone_event: LedgerEventRecord


class TombstoneMemoryNeo4jAdapter(MemoryNeo4jAdapter):
    def apply(self, batch: StructuralBatch) -> Neo4jApplyResult:
        key = (str(batch.generation_id), batch.ledger_seq)
        if key in self.deliveries:
            result = super().apply(batch)
            self._apply_tombstone_cleanup(batch)
            return result
        self._apply_tombstone_cleanup(batch)
        return super().apply(batch)

    def _apply_tombstone_cleanup(self, batch: StructuralBatch) -> None:
        if not batch.tombstoned_object_admission_ids:
            return
        covered = set(batch.tombstoned_object_admission_ids)
        for key, prior in tuple(self.deliveries.items()):
            if key[0] != str(batch.generation_id):
                continue
            if any(
                relation.object_admission_id in covered
                for relation in prior.relations
            ):
                del self.deliveries[key]


def _object_reference_bytes(value: object) -> bytes:
    if not isinstance(value, ObjectAdmissionDescriptor):
        raise PayloadSchemaValidationError(
            "object-reference payload requires an admission descriptor"
        )
    return canonical_json_bytes(
        {
            "admission_id": str(value.admission_id),
            "blob_digest": value.blob_digest,
            "object_class": value.object_class,
            "allowed_use": value.allowed_use,
            "security_scope": value.security_scope,
            "retention_scope": value.retention_scope,
        }
    )


def _object_contract() -> PayloadSchemaContract:
    vector = ObjectAdmissionDescriptor(
        admission_id=ObjectAdmissionId.parse(
            "00000000-0000-4000-8000-000000000001"
        ),
        blob_digest="sha256:" + "a" * 64,
        object_class="source_capture",
        allowed_use="project.discovery",
        security_scope="authority.protected",
        retention_scope="source.short",
        active=True,
    )
    return PayloadSchemaContract(
        schema_version="projection_object_reference_v1",
        payload_mode=PayloadMode.OBJECT_ADMISSION,
        contract_version="projection-object-contract-v1",
        canonicalizer_implementation_version=(
            "projection-object-canonicalizer-v1"
        ),
        canonicalizer=_object_reference_bytes,
        golden_vectors=(
            PayloadGoldenVector(
                name="projection-object-reference",
                input_identity="projection-object-reference-v1",
                value=vector,
                expected_bytes=_object_reference_bytes(vector),
            ),
        ),
    )


def tombstone_registries() -> tuple[CommandRegistry, PayloadSchemaRegistry]:
    contract = _object_contract()
    definition = CommandDefinition(
        command_type="record.object_versioned",
        definition_version="projection-object-command-v1",
        aggregate_type="fixture_aggregate",
        event_type="authority.aggregate.versioned",
        event_schema_version=1,
        payload_mode=PayloadMode.OBJECT_ADMISSION,
        payload_schema_version=contract.schema_version,
        payload_schema_contract_version=contract.contract_version,
        payload_schema_contract_digest=contract.contract_digest,
        payload_canonicalizer_version=(
            contract.canonicalizer_implementation_version
        ),
        trust_scope=TrustScope.OBSERVED,
        security_scope="authority.protected",
        retention_scope="source.short",
        required_scope="authority.observed.write",
        required_object_class="source_capture",
        required_allowed_use="project.discovery",
    )
    commands = CommandRegistry(
        (*source_command_registry().definitions(), definition)
    )
    schemas = PayloadSchemaRegistry(
        (*payload_schemas().contracts(), contract)
    )
    return merge_authority_registries(
        command_registry=commands,
        payload_schemas=schemas,
    )


def create_tombstoned_structural_history(
    database: Path,
    object_root: Path,
) -> tuple[CommandRegistry, PayloadSchemaRegistry, TombstonedStructuralHistory]:
    commands, schemas = tombstone_registries()
    system = open_object_system(
        database,
        object_root=object_root,
        command_registry=commands,
        payload_schema_registry=schemas,
    )
    try:
        admission = admit(
            system,
            data=b"B3 authoritative object-backed structural payload",
        ).admission
        committed = system.commands.execute(
            SemanticCommand(
                command_type="record.object_versioned",
                aggregate_id=AggregateId.new(),
                expected_aggregate_version=0,
                payload=ObjectAdmissionPayload(admission.admission_id),
                idempotency_key="b3-object-structural-event",
            ),
            proof=proof(),
        )
        system.objects.revoke(
            admission.admission_id,
            reason_code="B3_TOMBSTONE_REVOKE",
            idempotency_key="b3-object-revoke",
            proof=proof(),
        )
        deletion = system.objects.request_deletion(
            admission.blob.blob_digest,
            reason_code="B3_TOMBSTONE_REQUEST",
            idempotency_key="b3-object-delete",
            proof=proof(),
        )
        tombstone = system.objects.tombstone(
            deletion.deletion_id,
            reason_code="B3_TOMBSTONE_COMMIT",
            idempotency_key="b3-object-tombstone",
            proof=proof(),
        )
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            source_row = connection.execute(
                "SELECT * FROM ledger_events WHERE event_id=?",
                (str(committed.event_id),),
            ).fetchone()
            tombstone_row = connection.execute(
                "SELECT * FROM ledger_events WHERE event_id=?",
                (str(tombstone.event_id),),
            ).fetchone()
        finally:
            connection.close()
        if source_row is None or tombstone_row is None:
            raise AssertionError("retained tombstone fixture event is absent")
        # Construct logical full events, excluding physical reservation columns.
        event_columns = tuple(field.name for field in fields(LedgerEventRecord))
        source_event = LedgerEventRecord(**{name: source_row[name] for name in event_columns})
        tombstone_event = LedgerEventRecord(**{name: tombstone_row[name] for name in event_columns})
        return (
            commands,
            schemas,
            TombstonedStructuralHistory(
                admission_id=admission.admission_id,
                blob_digest=admission.blob.blob_digest,
                source_event=source_event,
                tombstone_event=tombstone_event,
            ),
        )
    finally:
        system.close()


def _scopes() -> frozenset[str]:
    policy = event_read_policy()
    return frozenset(
        {
            "authority.observed.write",
            "authority.admitted.write",
            policy.required_scope,
            "authority.projection.manage",
            "authority.projection.write",
            "authority.projection.read",
        }
    )


def open_tombstone_memory_system(
    path: Path,
    adapter: TombstoneMemoryNeo4jAdapter,
    commands: CommandRegistry,
    schemas: PayloadSchemaRegistry,
):
    policy = event_read_policy()
    return _open_with_adapter(
        path=path,
        registry=commands,
        payload_schemas=schemas,
        contracts=projection_contracts(),
        authenticator=StaticAuthenticator(
            credentials={"token-1": StaticPrincipal("principal.alpha")},
            authority_domain="newsroom.authority",
        ),
        authorizer=StaticAuthorizer(
            policy_version="authz-v1",
            grants_by_principal={"principal.alpha": _scopes()},
        ),
        event_read_policy=policy,
        projection_read_policy=projection_read_policy(),
        adapter=adapter,
    )


def open_tombstone_service_system(
    path: Path,
    config: Neo4jProjectorConfig,
    commands: CommandRegistry,
    schemas: PayloadSchemaRegistry,
):
    policy = event_read_policy()
    return open_neo4j_projection_authority_system(
        path=path,
        registry=commands,
        payload_schemas=schemas,
        contracts=projection_contracts(),
        authenticator=StaticAuthenticator(
            credentials={"token-1": StaticPrincipal("principal.alpha")},
            authority_domain="newsroom.authority",
        ),
        authorizer=StaticAuthorizer(
            policy_version="authz-v1",
            grants_by_principal={"principal.alpha": _scopes()},
        ),
        event_read_policy=policy,
        projection_read_policy=projection_read_policy(),
        neo4j_config=config,
    )
