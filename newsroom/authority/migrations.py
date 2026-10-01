# ruff: noqa: I001
from __future__ import annotations
# fmt: off

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass

from .canonical import digest_canonical
from .bounded_search_migrations import (
    BOUNDED_SEARCH_MIGRATION,
    BOUNDED_SEARCH_MIGRATION_CHECKSUM,
    BOUNDED_SEARCH_MIGRATION_NAME,
    BOUNDED_SEARCH_MIGRATION_STATEMENTS,
    BOUNDED_SEARCH_SCHEMA_VERSION,
    BoundedSearchBackupReceipt,
    bounded_search_backup_paths,
    prepare_bounded_search_backup,
    require_bounded_search_backup,
)
from .coverage_audit_migrations import (
    COVERAGE_AUDIT_MIGRATION,
    COVERAGE_AUDIT_MIGRATION_CHECKSUM,
    COVERAGE_AUDIT_MIGRATION_NAME,
    COVERAGE_AUDIT_MIGRATION_STATEMENTS,
    COVERAGE_AUDIT_SCHEMA_VERSION,
    CoverageAuditBackupReceipt,
    coverage_audit_backup_paths,
    prepare_coverage_audit_backup,
    require_coverage_audit_backup,
)
from .local_watch_migrations import (
    LOCAL_WATCH_MIGRATION,
    LOCAL_WATCH_MIGRATION_CHECKSUM,
    LOCAL_WATCH_MIGRATION_NAME,
    LOCAL_WATCH_MIGRATION_STATEMENTS,
    LOCAL_WATCH_SCHEMA_VERSION,
    LocalWatchBackupReceipt,
    local_watch_backup_paths,
    prepare_local_watch_backup,
    require_local_watch_backup,
)
from .increment8_evaluation_migrations import (
    INCREMENT8_EVALUATION_MIGRATION,
    INCREMENT8_EVALUATION_MIGRATION_CHECKSUM,
    INCREMENT8_EVALUATION_MIGRATION_NAME,
    INCREMENT8_EVALUATION_MIGRATION_STATEMENTS,
    INCREMENT8_EVALUATION_SCHEMA_VERSION,
    Increment8EvaluationBackupReceipt,
    increment8_evaluation_backup_paths,
    prepare_increment8_evaluation_backup,
    require_increment8_evaluation_backup,
)
from .increment8_operational_migrations import (
    INCREMENT8_OPERATIONAL_MIGRATION,
    INCREMENT8_OPERATIONAL_MIGRATION_CHECKSUM,
    INCREMENT8_OPERATIONAL_MIGRATION_NAME,
    INCREMENT8_OPERATIONAL_MIGRATION_STATEMENTS,
    INCREMENT8_OPERATIONAL_SCHEMA_VERSION,
    Increment8OperationalBackupReceipt,
    increment8_operational_backup_paths,
    prepare_increment8_operational_backup,
    require_increment8_operational_backup,
)
from .increment8_recovery_migrations import (
    INCREMENT8_RECOVERY_MIGRATION,
    INCREMENT8_RECOVERY_MIGRATION_CHECKSUM,
    INCREMENT8_RECOVERY_MIGRATION_NAME,
    INCREMENT8_RECOVERY_MIGRATION_STATEMENTS,
    INCREMENT8_RECOVERY_SCHEMA_VERSION,
    Increment8RecoveryBackupReceipt,
    increment8_recovery_backup_paths,
    prepare_increment8_recovery_backup,
    require_increment8_recovery_backup,
)
from .check_migrations import (
    CHECK_AUTHORITY_MIGRATION,
    CHECK_AUTHORITY_MIGRATION_CHECKSUM,
    CHECK_AUTHORITY_MIGRATION_NAME,
    CHECK_AUTHORITY_MIGRATION_STATEMENTS,
    CHECK_AUTHORITY_SCHEMA_VERSION,
)
from .complete_projection_migrations import (
    COMPLETE_PROJECTION_MIGRATION,
    COMPLETE_PROJECTION_MIGRATION_CHECKSUM,
    COMPLETE_PROJECTION_MIGRATION_NAME,
    COMPLETE_PROJECTION_MIGRATION_STATEMENTS,
    COMPLETE_PROJECTION_SCHEMA_VERSION,
)
from .development_candidate_migrations import (
    DEVELOPMENT_CANDIDATE_MIGRATION,
    DEVELOPMENT_CANDIDATE_MIGRATION_CHECKSUM,
    DEVELOPMENT_CANDIDATE_MIGRATION_NAME,
    DEVELOPMENT_CANDIDATE_MIGRATION_STATEMENTS,
    DEVELOPMENT_CANDIDATE_SCHEMA_VERSION,
)
from .discovery_migrations import (
    DISCOVERY_AUTHORITY_MIGRATION,
    DISCOVERY_AUTHORITY_MIGRATION_CHECKSUM,
    DISCOVERY_AUTHORITY_MIGRATION_NAME,
    DISCOVERY_AUTHORITY_MIGRATION_STATEMENTS,
    DISCOVERY_AUTHORITY_SCHEMA_VERSION,
)
from .editorial_relation_migrations import (
    EDITORIAL_RELATION_MIGRATION,
    EDITORIAL_RELATION_MIGRATION_CHECKSUM,
    EDITORIAL_RELATION_MIGRATION_NAME,
    EDITORIAL_RELATION_MIGRATION_STATEMENTS,
    EDITORIAL_RELATION_SCHEMA_VERSION,
)
from .entity_migrations import (
    ENTITY_AUTHORITY_MIGRATION,
    ENTITY_AUTHORITY_MIGRATION_CHECKSUM,
    ENTITY_AUTHORITY_MIGRATION_NAME,
    ENTITY_AUTHORITY_MIGRATION_STATEMENTS,
    ENTITY_AUTHORITY_SCHEMA_VERSION,
)
from .evaluation_feedback_migrations import (
    EVALUATION_FEEDBACK_MIGRATION,
    EVALUATION_FEEDBACK_MIGRATION_CHECKSUM,
    EVALUATION_FEEDBACK_MIGRATION_NAME,
    EVALUATION_FEEDBACK_MIGRATION_STATEMENTS,
    EVALUATION_FEEDBACK_SCHEMA_VERSION,
    EvaluationFeedbackBackupReceipt,
    evaluation_feedback_backup_paths,
    prepare_evaluation_feedback_backup,
    require_evaluation_feedback_backup,
)
from .evaluation_handoff_migrations import (
    EVALUATION_HANDOFF_MIGRATION,
    EVALUATION_HANDOFF_MIGRATION_CHECKSUM,
    EVALUATION_HANDOFF_MIGRATION_NAME,
    EVALUATION_HANDOFF_MIGRATION_STATEMENTS,
    EVALUATION_HANDOFF_SCHEMA_VERSION,
    EvaluationHandoffBackupReceipt,
    evaluation_handoff_backup_paths,
    prepare_evaluation_handoff_backup,
    require_evaluation_handoff_backup,
)
from .planned_agenda_migrations import (
    PLANNED_AGENDA_MIGRATION,
    PLANNED_AGENDA_MIGRATION_CHECKSUM,
    PLANNED_AGENDA_MIGRATION_NAME,
    PLANNED_AGENDA_MIGRATION_STATEMENTS,
    PLANNED_AGENDA_SCHEMA_VERSION,
    PlannedAgendaBackupReceipt,
    planned_agenda_backup_paths,
    prepare_planned_agenda_backup,
    require_planned_agenda_backup,
)
from .event_hypothesis_lineage_migrations import (
    EVENT_HYPOTHESIS_LINEAGE_MIGRATION,
    EVENT_HYPOTHESIS_LINEAGE_MIGRATION_CHECKSUM,
    EVENT_HYPOTHESIS_LINEAGE_MIGRATION_NAME,
    EVENT_HYPOTHESIS_LINEAGE_MIGRATION_STATEMENTS,
    EVENT_HYPOTHESIS_LINEAGE_SCHEMA_VERSION,
    EventHypothesisLineageBackupReceipt,
    event_hypothesis_lineage_backup_paths,
    prepare_event_hypothesis_lineage_backup,
    require_event_hypothesis_lineage_backup,
)
from .event_hypothesis_migrations import (
    EVENT_HYPOTHESIS_MIGRATION,
    EVENT_HYPOTHESIS_MIGRATION_CHECKSUM,
    EVENT_HYPOTHESIS_MIGRATION_NAME,
    EVENT_HYPOTHESIS_MIGRATION_STATEMENTS,
    EVENT_HYPOTHESIS_SCHEMA_VERSION,
    EventHypothesisBackupReceipt,
    event_hypothesis_backup_paths,
    prepare_event_hypothesis_backup,
    require_event_hypothesis_backup,
)
from .event_hypothesis_relationship_migrations import (
    EVENT_HYPOTHESIS_RELATIONSHIP_MIGRATION,
    EVENT_HYPOTHESIS_RELATIONSHIP_MIGRATION_CHECKSUM,
    EVENT_HYPOTHESIS_RELATIONSHIP_MIGRATION_NAME,
    EVENT_HYPOTHESIS_RELATIONSHIP_MIGRATION_STATEMENTS,
    EVENT_HYPOTHESIS_RELATIONSHIP_SCHEMA_VERSION,
    EventHypothesisRelationshipBackupReceipt,
    event_hypothesis_relationship_backup_paths,
    prepare_event_hypothesis_relationship_backup,
    require_event_hypothesis_relationship_backup,
)
from .extraction_migrations import (
    EXTRACTION_AUTHORITY_MIGRATION,
    EXTRACTION_AUTHORITY_MIGRATION_CHECKSUM,
    EXTRACTION_AUTHORITY_MIGRATION_NAME,
    EXTRACTION_AUTHORITY_MIGRATION_STATEMENTS,
    EXTRACTION_AUTHORITY_SCHEMA_VERSION,
)
from .graphiti_adapter_migrations import (
    GRAPHITI_ADAPTER_MIGRATION,
    GRAPHITI_ADAPTER_MIGRATION_CHECKSUM,
    GRAPHITI_ADAPTER_MIGRATION_NAME,
    GRAPHITI_ADAPTER_MIGRATION_STATEMENTS,
    GRAPHITI_ADAPTER_SCHEMA_VERSION,
)
from .graphiti_accounted_zero_migrations import (
    GRAPHITI_ACCOUNTED_ZERO_MIGRATION,
    GRAPHITI_ACCOUNTED_ZERO_MIGRATION_CHECKSUM,
    GRAPHITI_ACCOUNTED_ZERO_MIGRATION_NAME,
    GRAPHITI_ACCOUNTED_ZERO_MIGRATION_STATEMENTS,
    GRAPHITI_ACCOUNTED_ZERO_SCHEMA_VERSION,
    GraphitiAccountedZeroBackupReceipt,
    graphiti_accounted_zero_backup_paths,
    prepare_graphiti_accounted_zero_backup,
    require_graphiti_accounted_zero_backup,
)
from .authorisation_scope_content_migrations import (
    AUTHORISATION_SCOPE_CONTENT_MIGRATION,
    AUTHORISATION_SCOPE_CONTENT_MIGRATION_CHECKSUM,
    AUTHORISATION_SCOPE_CONTENT_MIGRATION_NAME,
    AUTHORISATION_SCOPE_CONTENT_MIGRATION_STATEMENTS,
    AUTHORISATION_SCOPE_CONTENT_SCHEMA_VERSION,
    migrate_authorisation_scope_content,
)
from .security_record_migrations import (
    SECURITY_RECORD_MIGRATION,
    SECURITY_RECORD_MIGRATION_CHECKSUM,
    SECURITY_RECORD_MIGRATION_NAME,
    SECURITY_RECORD_MIGRATION_STATEMENTS,
    SECURITY_RECORD_SCHEMA_VERSION,
    migrate_security_records,
)
from .authorization_request_storage_migrations import (
    AUTHORIZATION_REQUEST_STORAGE_MIGRATION,
    AUTHORIZATION_REQUEST_STORAGE_MIGRATION_CHECKSUM,
    AUTHORIZATION_REQUEST_STORAGE_MIGRATION_NAME,
    AUTHORIZATION_REQUEST_STORAGE_MIGRATION_STATEMENTS,
    AUTHORIZATION_REQUEST_STORAGE_SCHEMA_VERSION,
    migrate_authorization_request_storage,
)
from .graphiti_recovered_ambiguous_migrations import (
    GRAPHITI_RECOVERED_AMBIGUOUS_MIGRATION,
    GRAPHITI_RECOVERED_AMBIGUOUS_MIGRATION_CHECKSUM,
    GRAPHITI_RECOVERED_AMBIGUOUS_MIGRATION_NAME,
    GRAPHITI_RECOVERED_AMBIGUOUS_MIGRATION_STATEMENTS,
    GRAPHITI_RECOVERED_AMBIGUOUS_SCHEMA_VERSION,
)
from .command_bound_storage_migrations import (
    COMMAND_BOUND_STORAGE_MIGRATION,
    COMMAND_BOUND_STORAGE_MIGRATION_CHECKSUM,
    COMMAND_BOUND_STORAGE_MIGRATION_NAME,
    COMMAND_BOUND_STORAGE_MIGRATION_STATEMENTS,
    COMMAND_BOUND_STORAGE_SCHEMA_VERSION,
    migrate_command_bound_storage,
)
from .relationship_open_index_migrations import (
    RELATIONSHIP_OPEN_INDEX_MIGRATION,
    RELATIONSHIP_OPEN_INDEX_MIGRATION_CHECKSUM,
    RELATIONSHIP_OPEN_INDEX_MIGRATION_NAME,
    RELATIONSHIP_OPEN_INDEX_MIGRATION_STATEMENTS,
    RELATIONSHIP_OPEN_INDEX_SCHEMA_VERSION,
)
from .graphiti_evaluation_migrations import (
    GRAPHITI_EVALUATION_MIGRATION,
    GRAPHITI_EVALUATION_MIGRATION_CHECKSUM,
    GRAPHITI_EVALUATION_MIGRATION_NAME,
    GRAPHITI_EVALUATION_MIGRATION_STATEMENTS,
    GRAPHITI_EVALUATION_SCHEMA_VERSION,
    GraphitiEvaluationBackupReceipt,
    graphiti_evaluation_backup_paths,
    prepare_graphiti_evaluation_backup,
    require_graphiti_evaluation_backup,
)
from .integrated_migrations import (
    INTEGRATED_FOUNDATION_MIGRATION,
    INTEGRATED_FOUNDATION_MIGRATION_CHECKSUM,
    INTEGRATED_FOUNDATION_MIGRATION_NAME,
    INTEGRATED_FOUNDATION_MIGRATION_STATEMENTS,
    INTEGRATED_FOUNDATION_SCHEMA_VERSION,
)
from .object_migrations import (
    OBJECT_MIGRATION,
    OBJECT_MIGRATION_CHECKSUM,
    OBJECT_MIGRATION_NAME,
    OBJECT_MIGRATION_STATEMENTS,
    OBJECT_SCHEMA_VERSION,
)
from .projection_migrations import (
    PROJECTION_MIGRATION,
    PROJECTION_MIGRATION_CHECKSUM,
    PROJECTION_MIGRATION_NAME,
    PROJECTION_MIGRATION_STATEMENTS,
    PROJECTION_SCHEMA_VERSION,
)
from .projection_promotion_migrations import (
    PROJECTION_PROMOTION_MIGRATION,
    PROJECTION_PROMOTION_MIGRATION_CHECKSUM,
    PROJECTION_PROMOTION_MIGRATION_NAME,
    PROJECTION_PROMOTION_MIGRATION_STATEMENTS,
    PROJECTION_PROMOTION_SCHEMA_VERSION,
)
from .relation_migrations import (
    RELATION_MIGRATION,
    RELATION_MIGRATION_CHECKSUM,
    RELATION_MIGRATION_NAME,
    RELATION_MIGRATION_STATEMENTS,
    RELATION_SCHEMA_VERSION,
)
from .retrieval_migrations import (
    HYBRID_RETRIEVAL_MIGRATION,
    HYBRID_RETRIEVAL_MIGRATION_CHECKSUM,
    HYBRID_RETRIEVAL_MIGRATION_NAME,
    HYBRID_RETRIEVAL_MIGRATION_STATEMENTS,
    HYBRID_RETRIEVAL_SCHEMA_VERSION,
)
from .source_registry_migrations import (
    SOURCE_REGISTRY_MIGRATION,
    SOURCE_REGISTRY_MIGRATION_CHECKSUM,
    SOURCE_REGISTRY_MIGRATION_NAME,
    SOURCE_REGISTRY_MIGRATION_STATEMENTS,
    SOURCE_REGISTRY_SCHEMA_VERSION,
)
from .story_candidate_migrations import (
    STORY_CANDIDATE_MIGRATION,
    STORY_CANDIDATE_MIGRATION_CHECKSUM,
    STORY_CANDIDATE_MIGRATION_NAME,
    STORY_CANDIDATE_MIGRATION_STATEMENTS,
    STORY_CANDIDATE_SCHEMA_VERSION,
    StoryCandidateBackupReceipt,
    prepare_story_candidate_backup,
    require_story_candidate_backup,
    story_candidate_backup_paths,
)
from .triage_disposition_migrations import (
    TRIAGE_DISPOSITION_MIGRATION,
    TRIAGE_DISPOSITION_MIGRATION_CHECKSUM,
    TRIAGE_DISPOSITION_MIGRATION_NAME,
    TRIAGE_DISPOSITION_MIGRATION_STATEMENTS,
    TRIAGE_DISPOSITION_SCHEMA_VERSION,
    TriageDispositionBackupReceipt,
    prepare_triage_disposition_backup,
    require_triage_disposition_backup,
    triage_disposition_backup_paths,
)
from .triage_execution_migrations import (
    TRIAGE_EXECUTION_MIGRATION,
    TRIAGE_EXECUTION_MIGRATION_CHECKSUM,
    TRIAGE_EXECUTION_MIGRATION_NAME,
    TRIAGE_EXECUTION_MIGRATION_STATEMENTS,
    TRIAGE_EXECUTION_SCHEMA_VERSION,
    TriageExecutionBackupReceipt,
    prepare_triage_execution_backup,
    require_triage_execution_backup,
    triage_execution_backup_paths,
)
from .triage_work_item_migrations import (
    TRIAGE_WORK_ITEM_MIGRATION,
    TRIAGE_WORK_ITEM_MIGRATION_CHECKSUM,
    TRIAGE_WORK_ITEM_MIGRATION_NAME,
    TRIAGE_WORK_ITEM_MIGRATION_STATEMENTS,
    TRIAGE_WORK_ITEM_SCHEMA_VERSION,
    TriageWorkItemBackupReceipt,
    prepare_triage_work_item_backup,
    require_triage_work_item_backup,
    triage_work_item_backup_paths,
)

BASE_SCHEMA_VERSION = 1
SCHEMA_VERSION = COMMAND_BOUND_STORAGE_SCHEMA_VERSION
ISOLATED_SCHEMA_VERSION_RESERVATIONS = frozenset({33})
MIGRATION_NAME = "authority_event_foundation_v1"


@dataclass(frozen=True, slots=True)
class MigrationRecord:
    version: int
    name: str
    checksum: str


MIGRATION_STATEMENTS: tuple[str, ...] = (
    """CREATE TABLE authority_migrations(
        version INTEGER PRIMARY KEY CHECK(version > 0),
        name TEXT NOT NULL UNIQUE,
        checksum TEXT NOT NULL,
        applied_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE payload_schema_contracts(
        contract_digest TEXT PRIMARY KEY,
        schema_version TEXT NOT NULL,
        payload_mode TEXT NOT NULL
            CHECK(payload_mode IN ('INLINE','OBJECT_ADMISSION','NO_PAYLOAD')),
        contract_version TEXT NOT NULL,
        canonicalizer_implementation_version TEXT NOT NULL,
        canonical_bytes BLOB NOT NULL,
        registered_at TEXT NOT NULL,
        UNIQUE(
            schema_version,
            payload_mode,
            contract_version,
            canonicalizer_implementation_version
        ),
        CHECK(length(canonical_bytes) > 0)
    ) STRICT""",
    """CREATE TABLE command_definitions(
        definition_digest TEXT PRIMARY KEY,
        command_type TEXT NOT NULL,
        definition_version TEXT NOT NULL,
        payload_schema_contract_digest TEXT NOT NULL
            REFERENCES payload_schema_contracts(contract_digest),
        canonical_bytes BLOB NOT NULL,
        registered_at TEXT NOT NULL,
        UNIQUE(command_type, definition_version),
        CHECK(length(canonical_bytes) > 0)
    ) STRICT""",
    """CREATE TABLE authentication_contexts(
        authentication_context_id TEXT PRIMARY KEY,
        principal_id TEXT NOT NULL,
        authority_domain TEXT NOT NULL,
        authentication_method TEXT NOT NULL,
        assurance_class TEXT NOT NULL,
        credential_binding_digest TEXT NOT NULL,
        authenticated_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        canonical_bytes BLOB NOT NULL,
        canonical_digest TEXT NOT NULL,
        CHECK(length(canonical_bytes) > 0)
    ) STRICT""",
    """CREATE TABLE authorization_requests(
        request_digest TEXT PRIMARY KEY,
        authentication_context_id TEXT NOT NULL
            REFERENCES authentication_contexts(authentication_context_id),
        principal_id TEXT NOT NULL,
        authority_domain TEXT NOT NULL,
        operation_type TEXT NOT NULL,
        required_scope TEXT NOT NULL,
        canonical_bytes BLOB NOT NULL,
        canonical_record_digest TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        UNIQUE(request_digest, authentication_context_id),
        CHECK(length(canonical_bytes) > 0)
    ) STRICT""",
    """CREATE TABLE authorization_decisions(
        authorization_decision_id TEXT PRIMARY KEY,
        authentication_context_id TEXT NOT NULL,
        authorization_request_digest TEXT NOT NULL,
        authorization_policy_version TEXT NOT NULL,
        effective_scopes BLOB NOT NULL,
        effective_scope_digest TEXT NOT NULL,
        allowed INTEGER NOT NULL CHECK(allowed IN (0,1)),
        reason_code TEXT NOT NULL,
        decided_at TEXT NOT NULL,
        canonical_bytes BLOB NOT NULL,
        canonical_digest TEXT NOT NULL,
        UNIQUE(
            authorization_decision_id,
            authentication_context_id,
            authorization_request_digest
        ),
        FOREIGN KEY(authentication_context_id)
            REFERENCES authentication_contexts(authentication_context_id),
        FOREIGN KEY(authorization_request_digest, authentication_context_id)
            REFERENCES authorization_requests(
                request_digest, authentication_context_id
            ),
        CHECK(length(canonical_bytes) > 0)
    ) STRICT""",
    """CREATE TABLE authority_payloads(
        payload_id TEXT PRIMARY KEY,
        mode TEXT NOT NULL
            CHECK(mode IN ('INLINE','OBJECT_ADMISSION','NO_PAYLOAD')),
        schema_version TEXT NOT NULL,
        schema_contract_version TEXT NOT NULL,
        schema_contract_digest TEXT NOT NULL
            REFERENCES payload_schema_contracts(contract_digest),
        canonicalizer_implementation_version TEXT NOT NULL,
        payload_digest TEXT NOT NULL,
        payload_bytes BLOB,
        object_admission_id TEXT,
        created_at TEXT NOT NULL,
        CHECK((mode='INLINE' AND payload_bytes IS NOT NULL
               AND length(payload_bytes) > 0 AND object_admission_id IS NULL)
           OR (mode='OBJECT_ADMISSION' AND payload_bytes IS NULL
               AND object_admission_id IS NOT NULL)
           OR (mode='NO_PAYLOAD' AND payload_bytes IS NOT NULL
               AND length(payload_bytes) = 0 AND object_admission_id IS NULL))
    ) STRICT""",
    """CREATE TABLE authority_aggregates(
        aggregate_type TEXT NOT NULL,
        aggregate_id TEXT NOT NULL,
        current_version INTEGER NOT NULL CHECK(current_version > 0),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(aggregate_type, aggregate_id),
        FOREIGN KEY(aggregate_type, aggregate_id, current_version)
            REFERENCES authority_aggregate_versions(
                aggregate_type, aggregate_id, aggregate_version
            ) DEFERRABLE INITIALLY DEFERRED
    ) WITHOUT ROWID, STRICT""",
    """CREATE TABLE authority_commands(
        command_id TEXT PRIMARY KEY,
        command_type TEXT NOT NULL,
        producer_version TEXT NOT NULL,
        command_definition_version TEXT NOT NULL,
        command_definition_digest TEXT NOT NULL
            REFERENCES command_definitions(definition_digest),
        aggregate_type TEXT NOT NULL,
        aggregate_id TEXT NOT NULL,
        expected_aggregate_version INTEGER NOT NULL
            CHECK(expected_aggregate_version >= 0),
        payload_id TEXT NOT NULL REFERENCES authority_payloads(payload_id),
        idempotency_namespace TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        stable_semantic_request_digest TEXT NOT NULL,
        authentication_context_id TEXT NOT NULL,
        authorization_request_digest TEXT NOT NULL,
        authorization_decision_id TEXT NOT NULL,
        result_digest TEXT NOT NULL,
        result_bytes BLOB NOT NULL,
        committed_at TEXT NOT NULL,
        UNIQUE(idempotency_namespace, idempotency_key),
        FOREIGN KEY(authentication_context_id)
            REFERENCES authentication_contexts(authentication_context_id),
        FOREIGN KEY(authorization_request_digest, authentication_context_id)
            REFERENCES authorization_requests(
                request_digest, authentication_context_id
            ),
        FOREIGN KEY(
            authorization_decision_id,
            authentication_context_id,
            authorization_request_digest
        ) REFERENCES authorization_decisions(
            authorization_decision_id,
            authentication_context_id,
            authorization_request_digest
        ),
        CHECK(length(result_bytes) > 0)
    ) STRICT""",
    """CREATE TABLE authority_aggregate_versions(
        aggregate_type TEXT NOT NULL,
        aggregate_id TEXT NOT NULL,
        aggregate_version INTEGER NOT NULL CHECK(aggregate_version > 0),
        command_id TEXT NOT NULL UNIQUE
            REFERENCES authority_commands(command_id),
        payload_id TEXT NOT NULL REFERENCES authority_payloads(payload_id),
        trust_scope TEXT NOT NULL
            CHECK(trust_scope IN ('OBSERVED','PROPOSED','ADMITTED')),
        recorded_at TEXT NOT NULL,
        PRIMARY KEY(aggregate_type, aggregate_id, aggregate_version),
        FOREIGN KEY(aggregate_type, aggregate_id)
            REFERENCES authority_aggregates(aggregate_type, aggregate_id)
            DEFERRABLE INITIALLY DEFERRED
    ) WITHOUT ROWID, STRICT""",
    """CREATE TABLE authority_audit_events(
        audit_id TEXT PRIMARY KEY,
        command_id TEXT NOT NULL UNIQUE
            REFERENCES authority_commands(command_id),
        authentication_context_id TEXT NOT NULL,
        authorization_request_digest TEXT NOT NULL,
        authorization_decision_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        detail_digest TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        FOREIGN KEY(authorization_request_digest, authentication_context_id)
            REFERENCES authorization_requests(
                request_digest, authentication_context_id
            ),
        FOREIGN KEY(
            authorization_decision_id,
            authentication_context_id,
            authorization_request_digest
        ) REFERENCES authorization_decisions(
            authorization_decision_id,
            authentication_context_id,
            authorization_request_digest
        )
    ) STRICT""",
    """CREATE TABLE ledger_events(
        ledger_seq INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL UNIQUE,
        event_type TEXT NOT NULL,
        event_schema_version INTEGER NOT NULL CHECK(event_schema_version > 0),
        aggregate_type TEXT NOT NULL,
        aggregate_id TEXT NOT NULL,
        aggregate_version INTEGER NOT NULL CHECK(aggregate_version > 0),
        recorded_at TEXT NOT NULL,
        command_id TEXT NOT NULL UNIQUE
            REFERENCES authority_commands(command_id),
        producer_version TEXT NOT NULL,
        command_definition_version TEXT NOT NULL,
        command_definition_digest TEXT NOT NULL
            REFERENCES command_definitions(definition_digest),
        payload_id TEXT NOT NULL REFERENCES authority_payloads(payload_id),
        payload_mode TEXT NOT NULL,
        payload_schema_version TEXT NOT NULL,
        payload_schema_contract_version TEXT NOT NULL,
        payload_schema_contract_digest TEXT NOT NULL
            REFERENCES payload_schema_contracts(contract_digest),
        payload_canonicalizer_version TEXT NOT NULL,
        payload_digest TEXT NOT NULL,
        object_admission_id TEXT,
        principal_id TEXT NOT NULL,
        authentication_context_id TEXT NOT NULL,
        authorization_request_digest TEXT NOT NULL,
        authorization_decision_id TEXT NOT NULL,
        correlation_id TEXT,
        causation_kind TEXT
            CHECK(causation_kind IN ('COMMAND','EVENT','EXTERNAL')),
        causation_identifier TEXT,
        causation_external_system TEXT,
        security_scope TEXT NOT NULL,
        retention_scope TEXT NOT NULL,
        trust_scope TEXT NOT NULL
            CHECK(trust_scope IN ('OBSERVED','PROPOSED','ADMITTED')),
        FOREIGN KEY(aggregate_type, aggregate_id, aggregate_version)
            REFERENCES authority_aggregate_versions(
                aggregate_type, aggregate_id, aggregate_version
            ),
        FOREIGN KEY(authorization_request_digest, authentication_context_id)
            REFERENCES authorization_requests(
                request_digest, authentication_context_id
            ),
        FOREIGN KEY(
            authorization_decision_id,
            authentication_context_id,
            authorization_request_digest
        ) REFERENCES authorization_decisions(
            authorization_decision_id,
            authentication_context_id,
            authorization_request_digest
        ),
        CHECK((causation_kind IS NULL AND causation_identifier IS NULL
               AND causation_external_system IS NULL)
           OR (causation_kind IN ('COMMAND','EVENT')
               AND causation_identifier IS NOT NULL
               AND causation_external_system IS NULL)
           OR (causation_kind='EXTERNAL'
               AND causation_identifier IS NOT NULL
               AND causation_external_system IS NOT NULL))
    ) STRICT""",
    "CREATE INDEX idx_ledger_events_aggregate ON ledger_events(aggregate_type, aggregate_id, aggregate_version)",
    "CREATE INDEX idx_ledger_events_recorded ON ledger_events(recorded_at, ledger_seq)",
    "CREATE INDEX idx_ledger_events_visibility ON ledger_events(security_scope, trust_scope, ledger_seq)",
    "CREATE INDEX idx_authorization_decisions_context ON authorization_decisions(authentication_context_id, decided_at)",
    """CREATE TRIGGER immutable_authority_migrations_update
        BEFORE UPDATE ON authority_migrations BEGIN
        SELECT RAISE(ABORT,'immutable migration history'); END""",
    """CREATE TRIGGER immutable_authority_migrations_delete
        BEFORE DELETE ON authority_migrations BEGIN
        SELECT RAISE(ABORT,'immutable migration history'); END""",
    """CREATE TRIGGER immutable_payload_schema_contracts_update
        BEFORE UPDATE ON payload_schema_contracts BEGIN
        SELECT RAISE(ABORT,'immutable payload schema contract'); END""",
    """CREATE TRIGGER immutable_payload_schema_contracts_delete
        BEFORE DELETE ON payload_schema_contracts BEGIN
        SELECT RAISE(ABORT,'immutable payload schema contract'); END""",
    """CREATE TRIGGER immutable_command_definitions_update
        BEFORE UPDATE ON command_definitions BEGIN
        SELECT RAISE(ABORT,'immutable command definition'); END""",
    """CREATE TRIGGER immutable_command_definitions_delete
        BEFORE DELETE ON command_definitions BEGIN
        SELECT RAISE(ABORT,'immutable command definition'); END""",
    """CREATE TRIGGER immutable_authentication_contexts_update
        BEFORE UPDATE ON authentication_contexts BEGIN
        SELECT RAISE(ABORT,'immutable authentication context'); END""",
    """CREATE TRIGGER immutable_authentication_contexts_delete
        BEFORE DELETE ON authentication_contexts BEGIN
        SELECT RAISE(ABORT,'immutable authentication context'); END""",
    """CREATE TRIGGER immutable_authorization_requests_update
        BEFORE UPDATE ON authorization_requests BEGIN
        SELECT RAISE(ABORT,'immutable authorization request'); END""",
    """CREATE TRIGGER immutable_authorization_requests_delete
        BEFORE DELETE ON authorization_requests BEGIN
        SELECT RAISE(ABORT,'immutable authorization request'); END""",
    """CREATE TRIGGER immutable_authorization_decisions_update
        BEFORE UPDATE ON authorization_decisions BEGIN
        SELECT RAISE(ABORT,'immutable authorization decision'); END""",
    """CREATE TRIGGER immutable_authorization_decisions_delete
        BEFORE DELETE ON authorization_decisions BEGIN
        SELECT RAISE(ABORT,'immutable authorization decision'); END""",
    """CREATE TRIGGER immutable_authority_payloads_update
        BEFORE UPDATE ON authority_payloads BEGIN
        SELECT RAISE(ABORT,'immutable authority payload'); END""",
    """CREATE TRIGGER immutable_authority_payloads_delete
        BEFORE DELETE ON authority_payloads BEGIN
        SELECT RAISE(ABORT,'immutable authority payload'); END""",
    """CREATE TRIGGER immutable_authority_commands_update
        BEFORE UPDATE ON authority_commands BEGIN
        SELECT RAISE(ABORT,'immutable authority command'); END""",
    """CREATE TRIGGER immutable_authority_commands_delete
        BEFORE DELETE ON authority_commands BEGIN
        SELECT RAISE(ABORT,'immutable authority command'); END""",
    """CREATE TRIGGER authority_aggregates_insert_guard
        BEFORE INSERT ON authority_aggregates
        WHEN NEW.current_version != 1 BEGIN
        SELECT RAISE(ABORT,'aggregate heads begin at version one'); END""",
    """CREATE TRIGGER authority_aggregates_update_guard
        BEFORE UPDATE ON authority_aggregates
        WHEN NEW.aggregate_type != OLD.aggregate_type
          OR NEW.aggregate_id != OLD.aggregate_id
          OR NEW.current_version != OLD.current_version + 1
          OR NEW.created_at != OLD.created_at
        BEGIN SELECT RAISE(ABORT,'invalid aggregate-head update'); END""",
    """CREATE TRIGGER authority_aggregates_delete_guard
        BEFORE DELETE ON authority_aggregates BEGIN
        SELECT RAISE(ABORT,'aggregate heads are retained'); END""",
    """CREATE TRIGGER immutable_aggregate_versions_update
        BEFORE UPDATE ON authority_aggregate_versions BEGIN
        SELECT RAISE(ABORT,'immutable aggregate version'); END""",
    """CREATE TRIGGER immutable_aggregate_versions_delete
        BEFORE DELETE ON authority_aggregate_versions BEGIN
        SELECT RAISE(ABORT,'immutable aggregate version'); END""",
    """CREATE TRIGGER immutable_authority_audit_events_update
        BEFORE UPDATE ON authority_audit_events BEGIN
        SELECT RAISE(ABORT,'immutable audit event'); END""",
    """CREATE TRIGGER immutable_authority_audit_events_delete
        BEFORE DELETE ON authority_audit_events BEGIN
        SELECT RAISE(ABORT,'immutable audit event'); END""",
    """CREATE TRIGGER immutable_ledger_events_update
        BEFORE UPDATE ON ledger_events BEGIN
        SELECT RAISE(ABORT,'immutable ledger event'); END""",
    """CREATE TRIGGER immutable_ledger_events_delete
        BEFORE DELETE ON ledger_events BEGIN
        SELECT RAISE(ABORT,'immutable ledger event'); END""",
    """CREATE TRIGGER ledger_event_payload_guard
        BEFORE INSERT ON ledger_events
        WHEN NOT EXISTS(
            SELECT 1 FROM authority_payloads p
            WHERE p.payload_id = NEW.payload_id
              AND p.mode = NEW.payload_mode
              AND p.schema_version = NEW.payload_schema_version
              AND p.schema_contract_version =
                    NEW.payload_schema_contract_version
              AND p.schema_contract_digest =
                    NEW.payload_schema_contract_digest
              AND p.canonicalizer_implementation_version =
                    NEW.payload_canonicalizer_version
              AND p.payload_digest = NEW.payload_digest
              AND p.object_admission_id IS NEW.object_admission_id
        )
        BEGIN SELECT RAISE(ABORT,'event payload envelope mismatch'); END""",
    """CREATE TRIGGER ledger_event_command_guard
        BEFORE INSERT ON ledger_events
        WHEN NOT EXISTS(
            SELECT 1 FROM authority_commands c
            WHERE c.command_id = NEW.command_id
              AND c.producer_version = NEW.producer_version
              AND c.command_definition_version =
                    NEW.command_definition_version
              AND c.command_definition_digest =
                    NEW.command_definition_digest
              AND c.payload_id = NEW.payload_id
              AND c.authentication_context_id =
                    NEW.authentication_context_id
              AND c.authorization_request_digest =
                    NEW.authorization_request_digest
              AND c.authorization_decision_id =
                    NEW.authorization_decision_id
        )
        BEGIN SELECT RAISE(ABORT,'event command envelope mismatch'); END""",
    """CREATE TRIGGER ledger_event_command_causation_guard
        BEFORE INSERT ON ledger_events
        WHEN NEW.causation_kind='COMMAND'
         AND NOT EXISTS(
            SELECT 1 FROM authority_commands
            WHERE command_id=NEW.causation_identifier
         )
        BEGIN SELECT RAISE(ABORT,'unknown command causation'); END""",
    """CREATE TRIGGER ledger_event_event_causation_guard
        BEFORE INSERT ON ledger_events
        WHEN NEW.causation_kind='EVENT'
         AND NOT EXISTS(
            SELECT 1 FROM ledger_events
            WHERE event_id=NEW.causation_identifier
         )
        BEGIN SELECT RAISE(ABORT,'unknown event causation'); END""",
)

MIGRATION_CHECKSUM = digest_canonical(
    {
        "version": BASE_SCHEMA_VERSION,
        "name": MIGRATION_NAME,
        "statements": list(MIGRATION_STATEMENTS),
    }
)
MIGRATION = MigrationRecord(
    version=BASE_SCHEMA_VERSION,
    name=MIGRATION_NAME,
    checksum=MIGRATION_CHECKSUM,
)


def _normalize(sql: str | None) -> str:
    return " ".join((sql or "").split())


def schema_fingerprint(conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
    ).fetchall()
    return digest_canonical(
        [[str(row[0]), str(row[1]), str(row[2]), _normalize(row[3])] for row in rows]
    )


def prepare_pending_migration_backup(
    conn: sqlite3.Connection,
) -> (
    EvaluationHandoffBackupReceipt
    | TriageWorkItemBackupReceipt
    | TriageDispositionBackupReceipt
    | TriageExecutionBackupReceipt
    | EventHypothesisBackupReceipt
    | EventHypothesisRelationshipBackupReceipt
    | EventHypothesisLineageBackupReceipt
    | StoryCandidateBackupReceipt
    | EvaluationFeedbackBackupReceipt
    | PlannedAgendaBackupReceipt
    | BoundedSearchBackupReceipt
    | CoverageAuditBackupReceipt
    | LocalWatchBackupReceipt
    | Increment8EvaluationBackupReceipt
    | Increment8OperationalBackupReceipt
    | Increment8RecoveryBackupReceipt
    | GraphitiEvaluationBackupReceipt
    | GraphitiAccountedZeroBackupReceipt
    | None
):
    """Prepare the exact retained backup required by a checked predecessor."""
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version not in {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34}:
        return None
    database_path = next(
        str(row[2])
        for row in conn.execute("PRAGMA database_list").fetchall()
        if row[1] == "main"
    )
    if not database_path:
        raise sqlite3.DatabaseError(
            "existing v16 upgrade requires a file-backed database"
        )
    if version == 16:
        backup_path, _ = evaluation_handoff_backup_paths(database_path)
        return prepare_evaluation_handoff_backup(conn, backup_path)
    if version == 17:
        backup_path, _ = triage_work_item_backup_paths(database_path)
        return prepare_triage_work_item_backup(conn, backup_path)
    if version == 18:
        backup_path, _ = triage_disposition_backup_paths(database_path)
        return prepare_triage_disposition_backup(conn, backup_path)
    if version == 19:
        backup_path, _ = triage_execution_backup_paths(database_path)
        return prepare_triage_execution_backup(conn, backup_path)
    if version == 20:
        backup_path, _ = event_hypothesis_backup_paths(database_path)
        return prepare_event_hypothesis_backup(conn, backup_path)
    if version == 21:
        backup_path, _ = event_hypothesis_relationship_backup_paths(database_path)
        return prepare_event_hypothesis_relationship_backup(conn, backup_path)
    if version == 22:
        backup_path, _ = event_hypothesis_lineage_backup_paths(database_path)
        return prepare_event_hypothesis_lineage_backup(conn, backup_path)
    if version == 23:
        backup_path, _ = story_candidate_backup_paths(database_path)
        return prepare_story_candidate_backup(conn, backup_path)
    if version == 24:
        backup_path, _ = evaluation_feedback_backup_paths(database_path)
        return prepare_evaluation_feedback_backup(conn, backup_path)
    if version == 25:
        backup_path, _ = planned_agenda_backup_paths(database_path)
        return prepare_planned_agenda_backup(conn, backup_path)
    if version == 26:
        backup_path, _ = bounded_search_backup_paths(database_path)
        return prepare_bounded_search_backup(conn, backup_path)
    if version == 27:
        backup_path, _ = coverage_audit_backup_paths(database_path)
        return prepare_coverage_audit_backup(conn, backup_path)
    if version == 28:
        backup_path, _ = local_watch_backup_paths(database_path)
        return prepare_local_watch_backup(conn, backup_path)
    if version == 29:
        backup_path, _ = increment8_evaluation_backup_paths(database_path)
        return prepare_increment8_evaluation_backup(conn, backup_path)
    if version == 30:
        backup_path, _ = increment8_operational_backup_paths(database_path)
        return prepare_increment8_operational_backup(conn, backup_path)
    if version == 31:
        backup_path, _ = increment8_recovery_backup_paths(database_path)
        return prepare_increment8_recovery_backup(conn, backup_path)
    if version == 32:
        backup_path, _ = graphiti_evaluation_backup_paths(database_path)
        return prepare_graphiti_evaluation_backup(conn, backup_path)
    backup_path, _ = graphiti_accounted_zero_backup_paths(database_path)
    return prepare_graphiti_accounted_zero_backup(conn, backup_path)


def apply_migration(
    conn: sqlite3.Connection,
    *,
    applied_at: str,
    statements: Iterable[str] = MIGRATION_STATEMENTS,
) -> None:
    try:
        conn.execute("BEGIN EXCLUSIVE")
        for statement in statements:
            conn.execute(statement)
        conn.execute(
            "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
            "VALUES(?,?,?,?)",
            (BASE_SCHEMA_VERSION, MIGRATION_NAME, MIGRATION_CHECKSUM, applied_at),
        )
        conn.execute(f"PRAGMA user_version={BASE_SCHEMA_VERSION}")
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def apply_pending_migrations(conn: sqlite3.Connection, *, applied_at: str) -> None:
    """Apply every pending checked migration in one exclusive transaction.

    Fresh schema creation is all-or-nothing across every retained authority
    migration. Existing databases upgrade through checked extraction v13,
    entity-resolution v14, editorial-relation v15, isolated Graphiti
    proposal-adapter v16 and evaluation-Handoff authority v17.

    An exact retained v16 database must first call
    ``evaluation_handoff_migrations.prepare_evaluation_handoff_backup`` with a
    durable backup path.
    The v17 transaction revalidates that backup, its digest sidecar, the exact
    predecessor schema and migration history while holding its exclusive lock.
    """

    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    starting_version = current
    if current > SCHEMA_VERSION:
        raise sqlite3.DatabaseError(
            f"database schema {current} is newer than supported {SCHEMA_VERSION}"
        )
    if current == SCHEMA_VERSION:
        return
    try:
        conn.execute("BEGIN EXCLUSIVE")
        if current == 0:
            for statement in MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (BASE_SCHEMA_VERSION, MIGRATION_NAME, MIGRATION_CHECKSUM, applied_at),
            )
            current = BASE_SCHEMA_VERSION
        if current == BASE_SCHEMA_VERSION:
            for statement in OBJECT_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    OBJECT_SCHEMA_VERSION,
                    OBJECT_MIGRATION_NAME,
                    OBJECT_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = OBJECT_SCHEMA_VERSION
        if current == OBJECT_SCHEMA_VERSION:
            for statement in PROJECTION_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    PROJECTION_SCHEMA_VERSION,
                    PROJECTION_MIGRATION_NAME,
                    PROJECTION_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = PROJECTION_SCHEMA_VERSION
        if current == PROJECTION_SCHEMA_VERSION:
            for statement in PROJECTION_PROMOTION_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    PROJECTION_PROMOTION_SCHEMA_VERSION,
                    PROJECTION_PROMOTION_MIGRATION_NAME,
                    PROJECTION_PROMOTION_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = PROJECTION_PROMOTION_SCHEMA_VERSION
        if current == PROJECTION_PROMOTION_SCHEMA_VERSION:
            for statement in INTEGRATED_FOUNDATION_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    INTEGRATED_FOUNDATION_SCHEMA_VERSION,
                    INTEGRATED_FOUNDATION_MIGRATION_NAME,
                    INTEGRATED_FOUNDATION_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = INTEGRATED_FOUNDATION_SCHEMA_VERSION
        if current == INTEGRATED_FOUNDATION_SCHEMA_VERSION:
            for statement in RELATION_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    RELATION_SCHEMA_VERSION,
                    RELATION_MIGRATION_NAME,
                    RELATION_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = RELATION_SCHEMA_VERSION
        if current == RELATION_SCHEMA_VERSION:
            for statement in COMPLETE_PROJECTION_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    COMPLETE_PROJECTION_SCHEMA_VERSION,
                    COMPLETE_PROJECTION_MIGRATION_NAME,
                    COMPLETE_PROJECTION_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = COMPLETE_PROJECTION_SCHEMA_VERSION
        if current == COMPLETE_PROJECTION_SCHEMA_VERSION:
            for statement in HYBRID_RETRIEVAL_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    HYBRID_RETRIEVAL_SCHEMA_VERSION,
                    HYBRID_RETRIEVAL_MIGRATION_NAME,
                    HYBRID_RETRIEVAL_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = HYBRID_RETRIEVAL_SCHEMA_VERSION
        if current == HYBRID_RETRIEVAL_SCHEMA_VERSION:
            for statement in DEVELOPMENT_CANDIDATE_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    DEVELOPMENT_CANDIDATE_SCHEMA_VERSION,
                    DEVELOPMENT_CANDIDATE_MIGRATION_NAME,
                    DEVELOPMENT_CANDIDATE_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = DEVELOPMENT_CANDIDATE_SCHEMA_VERSION
        if current == DEVELOPMENT_CANDIDATE_SCHEMA_VERSION:
            for statement in SOURCE_REGISTRY_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations("
                "version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    SOURCE_REGISTRY_SCHEMA_VERSION,
                    SOURCE_REGISTRY_MIGRATION_NAME,
                    SOURCE_REGISTRY_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = SOURCE_REGISTRY_SCHEMA_VERSION
        if current == SOURCE_REGISTRY_SCHEMA_VERSION:
            for statement in CHECK_AUTHORITY_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations("
                "version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    CHECK_AUTHORITY_SCHEMA_VERSION,
                    CHECK_AUTHORITY_MIGRATION_NAME,
                    CHECK_AUTHORITY_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = CHECK_AUTHORITY_SCHEMA_VERSION
        if current == CHECK_AUTHORITY_SCHEMA_VERSION:
            for statement in DISCOVERY_AUTHORITY_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations("
                "version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    DISCOVERY_AUTHORITY_SCHEMA_VERSION,
                    DISCOVERY_AUTHORITY_MIGRATION_NAME,
                    DISCOVERY_AUTHORITY_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = DISCOVERY_AUTHORITY_SCHEMA_VERSION
        if current == DISCOVERY_AUTHORITY_SCHEMA_VERSION:
            for statement in EXTRACTION_AUTHORITY_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations("
                "version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    EXTRACTION_AUTHORITY_SCHEMA_VERSION,
                    EXTRACTION_AUTHORITY_MIGRATION_NAME,
                    EXTRACTION_AUTHORITY_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = EXTRACTION_AUTHORITY_SCHEMA_VERSION
        if current == EXTRACTION_AUTHORITY_SCHEMA_VERSION:
            for statement in ENTITY_AUTHORITY_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations("
                "version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    ENTITY_AUTHORITY_SCHEMA_VERSION,
                    ENTITY_AUTHORITY_MIGRATION_NAME,
                    ENTITY_AUTHORITY_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = ENTITY_AUTHORITY_SCHEMA_VERSION
        if current == ENTITY_AUTHORITY_SCHEMA_VERSION:
            for statement in EDITORIAL_RELATION_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations("
                "version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    EDITORIAL_RELATION_SCHEMA_VERSION,
                    EDITORIAL_RELATION_MIGRATION_NAME,
                    EDITORIAL_RELATION_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = EDITORIAL_RELATION_SCHEMA_VERSION
        if current == EDITORIAL_RELATION_SCHEMA_VERSION:
            for statement in GRAPHITI_ADAPTER_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations("
                "version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    GRAPHITI_ADAPTER_SCHEMA_VERSION,
                    GRAPHITI_ADAPTER_MIGRATION_NAME,
                    GRAPHITI_ADAPTER_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = GRAPHITI_ADAPTER_SCHEMA_VERSION
        if current == GRAPHITI_ADAPTER_SCHEMA_VERSION:
            if 0 < starting_version < GRAPHITI_ADAPTER_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version={GRAPHITI_ADAPTER_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(
                    str(row[2])
                    for row in conn.execute("PRAGMA database_list").fetchall()
                    if row[1] == "main"
                )
                if not database_path:
                    raise sqlite3.DatabaseError(
                        "existing multihop upgrade requires a file-backed database"
                    )
                backup_path, _ = evaluation_handoff_backup_paths(database_path)
                prepare_evaluation_handoff_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
            if starting_version != 0:
                require_evaluation_handoff_backup(
                    conn,
                    expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS if r.version <= GRAPHITI_ADAPTER_SCHEMA_VERSION),
                )
            for statement in EVALUATION_HANDOFF_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations("
                "version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    EVALUATION_HANDOFF_SCHEMA_VERSION,
                    EVALUATION_HANDOFF_MIGRATION_NAME,
                    EVALUATION_HANDOFF_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = EVALUATION_HANDOFF_SCHEMA_VERSION
        if current == EVALUATION_HANDOFF_SCHEMA_VERSION:
            if 0 < starting_version < EVALUATION_HANDOFF_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version={EVALUATION_HANDOFF_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(
                    str(row[2])
                    for row in conn.execute("PRAGMA database_list").fetchall()
                    if row[1] == "main"
                )
                if not database_path:
                    raise sqlite3.DatabaseError(
                        "existing multihop upgrade requires a file-backed database"
                    )
                backup_path, _ = triage_work_item_backup_paths(database_path)
                prepare_triage_work_item_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
            if starting_version != 0:
                require_triage_work_item_backup(
                    conn,
                    expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS if r.version <= EVALUATION_HANDOFF_SCHEMA_VERSION),
                )
            for statement in TRIAGE_WORK_ITEM_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    TRIAGE_WORK_ITEM_SCHEMA_VERSION,
                    TRIAGE_WORK_ITEM_MIGRATION_NAME,
                    TRIAGE_WORK_ITEM_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = TRIAGE_WORK_ITEM_SCHEMA_VERSION
        if current == TRIAGE_WORK_ITEM_SCHEMA_VERSION:
            if 0 < starting_version < TRIAGE_WORK_ITEM_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version={TRIAGE_WORK_ITEM_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(
                    str(row[2])
                    for row in conn.execute("PRAGMA database_list").fetchall()
                    if row[1] == "main"
                )
                if not database_path:
                    raise sqlite3.DatabaseError(
                        "existing multihop upgrade requires a file-backed database"
                    )
                backup_path, _ = triage_disposition_backup_paths(database_path)
                prepare_triage_disposition_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
            if starting_version != 0:
                require_triage_disposition_backup(
                    conn,
                    expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS if r.version <= TRIAGE_WORK_ITEM_SCHEMA_VERSION),
                )
            for statement in TRIAGE_DISPOSITION_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    TRIAGE_DISPOSITION_SCHEMA_VERSION,
                    TRIAGE_DISPOSITION_MIGRATION_NAME,
                    TRIAGE_DISPOSITION_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = TRIAGE_DISPOSITION_SCHEMA_VERSION
        if current == TRIAGE_DISPOSITION_SCHEMA_VERSION:
            if 0 < starting_version < TRIAGE_DISPOSITION_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version={TRIAGE_DISPOSITION_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(
                    str(row[2])
                    for row in conn.execute("PRAGMA database_list").fetchall()
                    if row[1] == "main"
                )
                if not database_path:
                    raise sqlite3.DatabaseError(
                        "existing multihop upgrade requires a file-backed database"
                    )
                backup_path, _ = triage_execution_backup_paths(database_path)
                prepare_triage_execution_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
            if starting_version != 0:
                require_triage_execution_backup(
                    conn,
                    expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS if r.version <= TRIAGE_DISPOSITION_SCHEMA_VERSION),
                )
            for statement in TRIAGE_EXECUTION_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    TRIAGE_EXECUTION_SCHEMA_VERSION,
                    TRIAGE_EXECUTION_MIGRATION_NAME,
                    TRIAGE_EXECUTION_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = TRIAGE_EXECUTION_SCHEMA_VERSION
        if current == TRIAGE_EXECUTION_SCHEMA_VERSION:
            if 0 < starting_version < TRIAGE_EXECUTION_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version={TRIAGE_EXECUTION_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(
                    str(row[2])
                    for row in conn.execute("PRAGMA database_list")
                    if row[1] == "main"
                )
                if not database_path:
                    raise sqlite3.DatabaseError(
                        "existing multihop upgrade requires a file-backed database"
                    )
                backup_path, _ = event_hypothesis_backup_paths(database_path)
                prepare_event_hypothesis_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
            if starting_version != 0:
                require_event_hypothesis_backup(
                    conn, expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS if r.version <= TRIAGE_EXECUTION_SCHEMA_VERSION)
                )
            for statement in EVENT_HYPOTHESIS_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    EVENT_HYPOTHESIS_SCHEMA_VERSION,
                    EVENT_HYPOTHESIS_MIGRATION_NAME,
                    EVENT_HYPOTHESIS_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = EVENT_HYPOTHESIS_SCHEMA_VERSION
        if current == EVENT_HYPOTHESIS_SCHEMA_VERSION:
            if 0 < starting_version < EVENT_HYPOTHESIS_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version={EVENT_HYPOTHESIS_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(
                    str(row[2])
                    for row in conn.execute("PRAGMA database_list")
                    if row[1] == "main"
                )
                if not database_path:
                    raise sqlite3.DatabaseError(
                        "existing multihop upgrade requires a file-backed database"
                    )
                backup_path, _ = event_hypothesis_relationship_backup_paths(
                    database_path
                )
                prepare_event_hypothesis_relationship_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
            if starting_version != 0:
                require_event_hypothesis_relationship_backup(
                    conn, expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS if r.version <= EVENT_HYPOTHESIS_SCHEMA_VERSION)
                )
            for statement in EVENT_HYPOTHESIS_RELATIONSHIP_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    EVENT_HYPOTHESIS_RELATIONSHIP_SCHEMA_VERSION,
                    EVENT_HYPOTHESIS_RELATIONSHIP_MIGRATION_NAME,
                    EVENT_HYPOTHESIS_RELATIONSHIP_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = EVENT_HYPOTHESIS_RELATIONSHIP_SCHEMA_VERSION
        if current == EVENT_HYPOTHESIS_RELATIONSHIP_SCHEMA_VERSION:
            if 0 < starting_version < EVENT_HYPOTHESIS_RELATIONSHIP_SCHEMA_VERSION:
                conn.execute(
                    f"PRAGMA user_version={EVENT_HYPOTHESIS_RELATIONSHIP_SCHEMA_VERSION}"
                )
                conn.execute("COMMIT")
                database_path = next(
                    str(row[2])
                    for row in conn.execute("PRAGMA database_list")
                    if row[1] == "main"
                )
                if not database_path:
                    raise sqlite3.DatabaseError(
                        "existing multihop upgrade requires a file-backed database"
                    )
                backup_path, _ = event_hypothesis_lineage_backup_paths(database_path)
                prepare_event_hypothesis_lineage_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
            if starting_version != 0:
                predecessor_history = tuple(
                    (record.version, record.name, record.checksum)
                    for record in MIGRATIONS
                    if record.version <= EVENT_HYPOTHESIS_RELATIONSHIP_SCHEMA_VERSION
                )
                require_event_hypothesis_lineage_backup(
                    conn, expected_history=predecessor_history
                )
            for statement in EVENT_HYPOTHESIS_LINEAGE_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                (
                    EVENT_HYPOTHESIS_LINEAGE_SCHEMA_VERSION,
                    EVENT_HYPOTHESIS_LINEAGE_MIGRATION_NAME,
                    EVENT_HYPOTHESIS_LINEAGE_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = EVENT_HYPOTHESIS_LINEAGE_SCHEMA_VERSION
        # fmt: off
        if current == EVENT_HYPOTHESIS_LINEAGE_SCHEMA_VERSION:
            if 0 < starting_version < EVENT_HYPOTHESIS_LINEAGE_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version={EVENT_HYPOTHESIS_LINEAGE_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(str(row[2]) for row in conn.execute("PRAGMA database_list") if row[1] == "main")
                if not database_path:
                    raise sqlite3.DatabaseError("existing multihop upgrade requires a file-backed database")
                backup_path, _ = story_candidate_backup_paths(database_path)
                prepare_story_candidate_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
            if starting_version != 0:
                require_story_candidate_backup(
                    conn, expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS
                                                 if r.version <= EVENT_HYPOTHESIS_LINEAGE_SCHEMA_VERSION),
                )
            for statement in STORY_CANDIDATE_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                (STORY_CANDIDATE_SCHEMA_VERSION, STORY_CANDIDATE_MIGRATION_NAME,
                 STORY_CANDIDATE_MIGRATION_CHECKSUM, applied_at),
            )
            current = STORY_CANDIDATE_SCHEMA_VERSION
        if current == STORY_CANDIDATE_SCHEMA_VERSION:
            if 0 < starting_version < STORY_CANDIDATE_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version={STORY_CANDIDATE_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(str(row[2]) for row in conn.execute("PRAGMA database_list") if row[1] == "main")
                if not database_path:
                    raise sqlite3.DatabaseError("existing multihop upgrade requires a file-backed database")
                backup_path, _ = evaluation_feedback_backup_paths(database_path)
                prepare_evaluation_feedback_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
            if starting_version != 0:
                require_evaluation_feedback_backup(
                    conn, expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS
                                                 if r.version <= STORY_CANDIDATE_SCHEMA_VERSION),
                )
            for statement in EVALUATION_FEEDBACK_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                (EVALUATION_FEEDBACK_SCHEMA_VERSION, EVALUATION_FEEDBACK_MIGRATION_NAME,
                 EVALUATION_FEEDBACK_MIGRATION_CHECKSUM, applied_at),
            )
            current = EVALUATION_FEEDBACK_SCHEMA_VERSION
        if current == EVALUATION_FEEDBACK_SCHEMA_VERSION:
            if 0 < starting_version < EVALUATION_FEEDBACK_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version={EVALUATION_FEEDBACK_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(str(row[2]) for row in conn.execute("PRAGMA database_list") if row[1] == "main")
                if not database_path:
                    raise sqlite3.DatabaseError("existing multihop upgrade requires a file-backed database")
                backup_path, _ = planned_agenda_backup_paths(database_path)
                prepare_planned_agenda_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
            if starting_version != 0:
                require_planned_agenda_backup(
                    conn, expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS
                                                 if r.version <= EVALUATION_FEEDBACK_SCHEMA_VERSION),
                )
            for statement in PLANNED_AGENDA_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                (PLANNED_AGENDA_SCHEMA_VERSION, PLANNED_AGENDA_MIGRATION_NAME,
                 PLANNED_AGENDA_MIGRATION_CHECKSUM, applied_at),
            )
            current = PLANNED_AGENDA_SCHEMA_VERSION
        if current == PLANNED_AGENDA_SCHEMA_VERSION:
            if 0 < starting_version < PLANNED_AGENDA_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version={PLANNED_AGENDA_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(str(row[2]) for row in conn.execute("PRAGMA database_list") if row[1] == "main")
                if not database_path:
                    raise sqlite3.DatabaseError("existing multihop upgrade requires a file-backed database")
                backup_path, _ = bounded_search_backup_paths(database_path)
                prepare_bounded_search_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
            if starting_version != 0:
                require_bounded_search_backup(
                    conn, expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS
                                                 if r.version <= PLANNED_AGENDA_SCHEMA_VERSION),
                )
            for statement in BOUNDED_SEARCH_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                (BOUNDED_SEARCH_SCHEMA_VERSION, BOUNDED_SEARCH_MIGRATION_NAME,
                 BOUNDED_SEARCH_MIGRATION_CHECKSUM, applied_at),
            )
            current = BOUNDED_SEARCH_SCHEMA_VERSION
        if current == BOUNDED_SEARCH_SCHEMA_VERSION:
            if 0 < starting_version < BOUNDED_SEARCH_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version={BOUNDED_SEARCH_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(str(row[2]) for row in conn.execute("PRAGMA database_list") if row[1] == "main")
                if not database_path:
                    raise sqlite3.DatabaseError("existing multihop upgrade requires a file-backed database")
                backup_path, _ = coverage_audit_backup_paths(database_path)
                prepare_coverage_audit_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
            if starting_version != 0:
                require_coverage_audit_backup(
                    conn, expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS
                                                 if r.version <= BOUNDED_SEARCH_SCHEMA_VERSION),
                )
            for statement in COVERAGE_AUDIT_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                (COVERAGE_AUDIT_SCHEMA_VERSION, COVERAGE_AUDIT_MIGRATION_NAME,
                 COVERAGE_AUDIT_MIGRATION_CHECKSUM, applied_at),
            )
            current = COVERAGE_AUDIT_SCHEMA_VERSION
        if current == COVERAGE_AUDIT_SCHEMA_VERSION:
            if 0 < starting_version < COVERAGE_AUDIT_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version={COVERAGE_AUDIT_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(str(row[2]) for row in conn.execute("PRAGMA database_list") if row[1] == "main")
                if not database_path:
                    raise sqlite3.DatabaseError("existing multihop upgrade requires a file-backed database")
                backup_path, _ = local_watch_backup_paths(database_path)
                prepare_local_watch_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
            if starting_version != 0:
                require_local_watch_backup(
                    conn, expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS
                                                 if r.version <= COVERAGE_AUDIT_SCHEMA_VERSION),
                )
            for statement in LOCAL_WATCH_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                (LOCAL_WATCH_SCHEMA_VERSION, LOCAL_WATCH_MIGRATION_NAME,
                 LOCAL_WATCH_MIGRATION_CHECKSUM, applied_at),
            )
            current = LOCAL_WATCH_SCHEMA_VERSION
        if current == LOCAL_WATCH_SCHEMA_VERSION:
            if 0 < starting_version < LOCAL_WATCH_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version={LOCAL_WATCH_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(str(row[2]) for row in conn.execute("PRAGMA database_list") if row[1] == "main")
                if not database_path:
                    raise sqlite3.DatabaseError("existing multihop upgrade requires a file-backed database")
                backup_path, _ = increment8_evaluation_backup_paths(database_path)
                prepare_increment8_evaluation_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
            if starting_version != 0:
                require_increment8_evaluation_backup(
                    conn, expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS
                                                 if r.version <= LOCAL_WATCH_SCHEMA_VERSION),
                )
            for statement in INCREMENT8_EVALUATION_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                (INCREMENT8_EVALUATION_SCHEMA_VERSION, INCREMENT8_EVALUATION_MIGRATION_NAME,
                 INCREMENT8_EVALUATION_MIGRATION_CHECKSUM, applied_at),
            )
            current = INCREMENT8_EVALUATION_SCHEMA_VERSION
        if current == INCREMENT8_EVALUATION_SCHEMA_VERSION:
            if 0 < starting_version < INCREMENT8_EVALUATION_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version={INCREMENT8_EVALUATION_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(str(row[2]) for row in conn.execute("PRAGMA database_list") if row[1] == "main")
                if not database_path:
                    raise sqlite3.DatabaseError("existing multihop upgrade requires a file-backed database")
                backup_path, _ = increment8_operational_backup_paths(database_path)
                prepare_increment8_operational_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
            if starting_version != 0:
                require_increment8_operational_backup(
                    conn, expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS
                                                 if r.version <= INCREMENT8_EVALUATION_SCHEMA_VERSION),
                )
            for statement in INCREMENT8_OPERATIONAL_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                (INCREMENT8_OPERATIONAL_SCHEMA_VERSION, INCREMENT8_OPERATIONAL_MIGRATION_NAME,
                 INCREMENT8_OPERATIONAL_MIGRATION_CHECKSUM, applied_at),
            )
            current = INCREMENT8_OPERATIONAL_SCHEMA_VERSION
        if current == INCREMENT8_OPERATIONAL_SCHEMA_VERSION:
            if 0 < starting_version < INCREMENT8_OPERATIONAL_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version={INCREMENT8_OPERATIONAL_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(str(row[2]) for row in conn.execute("PRAGMA database_list") if row[1] == "main")
                if not database_path:
                    raise sqlite3.DatabaseError("existing multihop upgrade requires a file-backed database")
                backup_path, _ = increment8_recovery_backup_paths(database_path)
                prepare_increment8_recovery_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
            if starting_version != 0:
                require_increment8_recovery_backup(
                    conn, expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS
                                                 if r.version <= INCREMENT8_OPERATIONAL_SCHEMA_VERSION),
                )
            for statement in INCREMENT8_RECOVERY_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                (INCREMENT8_RECOVERY_SCHEMA_VERSION, INCREMENT8_RECOVERY_MIGRATION_NAME,
                 INCREMENT8_RECOVERY_MIGRATION_CHECKSUM, applied_at),
            )
            current = INCREMENT8_RECOVERY_SCHEMA_VERSION
        if current == INCREMENT8_RECOVERY_SCHEMA_VERSION:
            if starting_version != 0:
                conn.execute(f"PRAGMA user_version={INCREMENT8_RECOVERY_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(str(row[2]) for row in conn.execute("PRAGMA database_list") if row[1] == "main")
                if not database_path:
                    raise sqlite3.DatabaseError("existing v32 upgrade requires a file-backed database")
                backup_path, _ = graphiti_evaluation_backup_paths(database_path)
                prepare_graphiti_evaluation_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
                require_graphiti_evaluation_backup(
                    conn, expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS
                                                 if r.version <= INCREMENT8_RECOVERY_SCHEMA_VERSION),
                )
            for statement in GRAPHITI_EVALUATION_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                (GRAPHITI_EVALUATION_SCHEMA_VERSION, GRAPHITI_EVALUATION_MIGRATION_NAME,
                 GRAPHITI_EVALUATION_MIGRATION_CHECKSUM, applied_at),
            )
            current = GRAPHITI_EVALUATION_SCHEMA_VERSION
        if current == GRAPHITI_EVALUATION_SCHEMA_VERSION:
            if starting_version != 0:
                conn.execute(f"PRAGMA user_version={GRAPHITI_EVALUATION_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                database_path = next(str(row[2]) for row in conn.execute("PRAGMA database_list") if row[1] == "main")
                if not database_path:
                    raise sqlite3.DatabaseError("existing v34 upgrade requires a file-backed database")
                backup_path, _ = graphiti_accounted_zero_backup_paths(database_path)
                prepare_graphiti_accounted_zero_backup(conn, backup_path)
                conn.execute("BEGIN EXCLUSIVE")
                require_graphiti_accounted_zero_backup(
                    conn, expected_history=tuple((r.version, r.name, r.checksum) for r in MIGRATIONS
                                                 if r.version <= GRAPHITI_EVALUATION_SCHEMA_VERSION),
                )
            for statement in GRAPHITI_ACCOUNTED_ZERO_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                (GRAPHITI_ACCOUNTED_ZERO_SCHEMA_VERSION, GRAPHITI_ACCOUNTED_ZERO_MIGRATION_NAME,
                 GRAPHITI_ACCOUNTED_ZERO_MIGRATION_CHECKSUM, applied_at),
            )
            current = GRAPHITI_ACCOUNTED_ZERO_SCHEMA_VERSION
        if current == GRAPHITI_ACCOUNTED_ZERO_SCHEMA_VERSION:
            migrate_authorisation_scope_content(
                conn,
                expected_history=tuple(
                    (r.version, r.name, r.checksum) for r in MIGRATIONS
                    if r.version <= GRAPHITI_ACCOUNTED_ZERO_SCHEMA_VERSION
                ),
            )
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                (AUTHORISATION_SCOPE_CONTENT_SCHEMA_VERSION, AUTHORISATION_SCOPE_CONTENT_MIGRATION_NAME,
                 AUTHORISATION_SCOPE_CONTENT_MIGRATION_CHECKSUM, applied_at),
            )
            current = AUTHORISATION_SCOPE_CONTENT_SCHEMA_VERSION
        if current == AUTHORISATION_SCOPE_CONTENT_SCHEMA_VERSION:
            migrate_security_records(
                conn,
                expected_history=tuple(
                    (r.version, r.name, r.checksum) for r in MIGRATIONS
                    if r.version <= AUTHORISATION_SCOPE_CONTENT_SCHEMA_VERSION
                ),
            )
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                (SECURITY_RECORD_SCHEMA_VERSION, SECURITY_RECORD_MIGRATION_NAME,
                 SECURITY_RECORD_MIGRATION_CHECKSUM, applied_at),
            )
            current = SECURITY_RECORD_SCHEMA_VERSION
        if current == SECURITY_RECORD_SCHEMA_VERSION:
            migrate_authorization_request_storage(
                conn,
                expected_history=tuple(
                    (r.version, r.name, r.checksum) for r in MIGRATIONS
                    if r.version <= SECURITY_RECORD_SCHEMA_VERSION
                ),
            )
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                (AUTHORIZATION_REQUEST_STORAGE_SCHEMA_VERSION,
                 AUTHORIZATION_REQUEST_STORAGE_MIGRATION_NAME,
                 AUTHORIZATION_REQUEST_STORAGE_MIGRATION_CHECKSUM, applied_at),
            )
            current = AUTHORIZATION_REQUEST_STORAGE_SCHEMA_VERSION
        if current == AUTHORIZATION_REQUEST_STORAGE_SCHEMA_VERSION:
            for statement in GRAPHITI_RECOVERED_AMBIGUOUS_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    GRAPHITI_RECOVERED_AMBIGUOUS_SCHEMA_VERSION,
                    GRAPHITI_RECOVERED_AMBIGUOUS_MIGRATION_NAME,
                    GRAPHITI_RECOVERED_AMBIGUOUS_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = GRAPHITI_RECOVERED_AMBIGUOUS_SCHEMA_VERSION
        if current == GRAPHITI_RECOVERED_AMBIGUOUS_SCHEMA_VERSION:
            for statement in RELATIONSHIP_OPEN_INDEX_MIGRATION_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (
                    RELATIONSHIP_OPEN_INDEX_SCHEMA_VERSION,
                    RELATIONSHIP_OPEN_INDEX_MIGRATION_NAME,
                    RELATIONSHIP_OPEN_INDEX_MIGRATION_CHECKSUM,
                    applied_at,
                ),
            )
            current = RELATIONSHIP_OPEN_INDEX_SCHEMA_VERSION
        if current == RELATIONSHIP_OPEN_INDEX_SCHEMA_VERSION:
            migrate_command_bound_storage(conn, expected_history=tuple(
                (r.version, r.name, r.checksum) for r in MIGRATIONS
                if r.version <= RELATIONSHIP_OPEN_INDEX_SCHEMA_VERSION
            ))
            conn.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                (COMMAND_BOUND_STORAGE_SCHEMA_VERSION, COMMAND_BOUND_STORAGE_MIGRATION_NAME,
                 COMMAND_BOUND_STORAGE_MIGRATION_CHECKSUM, applied_at),
            )
            current = COMMAND_BOUND_STORAGE_SCHEMA_VERSION
        # fmt: on
        conn.execute(f"PRAGMA user_version={current}")
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


MIGRATIONS: tuple[MigrationRecord | object, ...] = (
    MIGRATION,
    OBJECT_MIGRATION,
    PROJECTION_MIGRATION,
    PROJECTION_PROMOTION_MIGRATION,
    INTEGRATED_FOUNDATION_MIGRATION,
    RELATION_MIGRATION,
    COMPLETE_PROJECTION_MIGRATION,
    HYBRID_RETRIEVAL_MIGRATION,
    DEVELOPMENT_CANDIDATE_MIGRATION,
    SOURCE_REGISTRY_MIGRATION,
    CHECK_AUTHORITY_MIGRATION,
    DISCOVERY_AUTHORITY_MIGRATION,
    EXTRACTION_AUTHORITY_MIGRATION,
    ENTITY_AUTHORITY_MIGRATION,
    EDITORIAL_RELATION_MIGRATION,
    GRAPHITI_ADAPTER_MIGRATION,
    EVALUATION_HANDOFF_MIGRATION,
    TRIAGE_WORK_ITEM_MIGRATION,
    TRIAGE_DISPOSITION_MIGRATION,
    TRIAGE_EXECUTION_MIGRATION,
    EVENT_HYPOTHESIS_MIGRATION,
    EVENT_HYPOTHESIS_RELATIONSHIP_MIGRATION,
    EVENT_HYPOTHESIS_LINEAGE_MIGRATION,
    STORY_CANDIDATE_MIGRATION,
    EVALUATION_FEEDBACK_MIGRATION,
    PLANNED_AGENDA_MIGRATION,
    BOUNDED_SEARCH_MIGRATION,
    COVERAGE_AUDIT_MIGRATION,
    LOCAL_WATCH_MIGRATION,
    INCREMENT8_EVALUATION_MIGRATION,
    INCREMENT8_OPERATIONAL_MIGRATION,
    INCREMENT8_RECOVERY_MIGRATION,
    GRAPHITI_EVALUATION_MIGRATION,
    GRAPHITI_ACCOUNTED_ZERO_MIGRATION,
    AUTHORISATION_SCOPE_CONTENT_MIGRATION,
    SECURITY_RECORD_MIGRATION,
    AUTHORIZATION_REQUEST_STORAGE_MIGRATION,
    GRAPHITI_RECOVERED_AMBIGUOUS_MIGRATION,
    RELATIONSHIP_OPEN_INDEX_MIGRATION,
    COMMAND_BOUND_STORAGE_MIGRATION,
)


def _expected_fingerprint() -> str:
    conn = sqlite3.connect(":memory:", isolation_level=None)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        apply_pending_migrations(conn, applied_at="1970-01-01T00:00:00.000000Z")
        return schema_fingerprint(conn)
    finally:
        conn.close()


EXPECTED_SCHEMA_FINGERPRINT = _expected_fingerprint()
EXPECTED_MIGRATION_HISTORY: tuple[tuple[int, str, str], ...] = (
    (BASE_SCHEMA_VERSION, MIGRATION_NAME, MIGRATION_CHECKSUM),
    (OBJECT_SCHEMA_VERSION, OBJECT_MIGRATION_NAME, OBJECT_MIGRATION_CHECKSUM),
    (
        PROJECTION_SCHEMA_VERSION,
        PROJECTION_MIGRATION_NAME,
        PROJECTION_MIGRATION_CHECKSUM,
    ),
    (
        PROJECTION_PROMOTION_SCHEMA_VERSION,
        PROJECTION_PROMOTION_MIGRATION_NAME,
        PROJECTION_PROMOTION_MIGRATION_CHECKSUM,
    ),
    (
        INTEGRATED_FOUNDATION_SCHEMA_VERSION,
        INTEGRATED_FOUNDATION_MIGRATION_NAME,
        INTEGRATED_FOUNDATION_MIGRATION_CHECKSUM,
    ),
    (
        RELATION_SCHEMA_VERSION,
        RELATION_MIGRATION_NAME,
        RELATION_MIGRATION_CHECKSUM,
    ),
    (
        COMPLETE_PROJECTION_SCHEMA_VERSION,
        COMPLETE_PROJECTION_MIGRATION_NAME,
        COMPLETE_PROJECTION_MIGRATION_CHECKSUM,
    ),
    (
        HYBRID_RETRIEVAL_SCHEMA_VERSION,
        HYBRID_RETRIEVAL_MIGRATION_NAME,
        HYBRID_RETRIEVAL_MIGRATION_CHECKSUM,
    ),
    (
        DEVELOPMENT_CANDIDATE_SCHEMA_VERSION,
        DEVELOPMENT_CANDIDATE_MIGRATION_NAME,
        DEVELOPMENT_CANDIDATE_MIGRATION_CHECKSUM,
    ),
    (
        SOURCE_REGISTRY_SCHEMA_VERSION,
        SOURCE_REGISTRY_MIGRATION_NAME,
        SOURCE_REGISTRY_MIGRATION_CHECKSUM,
    ),
    (
        CHECK_AUTHORITY_SCHEMA_VERSION,
        CHECK_AUTHORITY_MIGRATION_NAME,
        CHECK_AUTHORITY_MIGRATION_CHECKSUM,
    ),
    (
        DISCOVERY_AUTHORITY_SCHEMA_VERSION,
        DISCOVERY_AUTHORITY_MIGRATION_NAME,
        DISCOVERY_AUTHORITY_MIGRATION_CHECKSUM,
    ),
    (
        EXTRACTION_AUTHORITY_SCHEMA_VERSION,
        EXTRACTION_AUTHORITY_MIGRATION_NAME,
        EXTRACTION_AUTHORITY_MIGRATION_CHECKSUM,
    ),
    (
        ENTITY_AUTHORITY_SCHEMA_VERSION,
        ENTITY_AUTHORITY_MIGRATION_NAME,
        ENTITY_AUTHORITY_MIGRATION_CHECKSUM,
    ),
    (
        EDITORIAL_RELATION_SCHEMA_VERSION,
        EDITORIAL_RELATION_MIGRATION_NAME,
        EDITORIAL_RELATION_MIGRATION_CHECKSUM,
    ),
    (
        GRAPHITI_ADAPTER_SCHEMA_VERSION,
        GRAPHITI_ADAPTER_MIGRATION_NAME,
        GRAPHITI_ADAPTER_MIGRATION_CHECKSUM,
    ),
    (
        EVALUATION_HANDOFF_SCHEMA_VERSION,
        EVALUATION_HANDOFF_MIGRATION_NAME,
        EVALUATION_HANDOFF_MIGRATION_CHECKSUM,
    ),
    (
        TRIAGE_WORK_ITEM_SCHEMA_VERSION,
        TRIAGE_WORK_ITEM_MIGRATION_NAME,
        TRIAGE_WORK_ITEM_MIGRATION_CHECKSUM,
    ),
    (
        TRIAGE_DISPOSITION_SCHEMA_VERSION,
        TRIAGE_DISPOSITION_MIGRATION_NAME,
        TRIAGE_DISPOSITION_MIGRATION_CHECKSUM,
    ),
    (
        TRIAGE_EXECUTION_SCHEMA_VERSION,
        TRIAGE_EXECUTION_MIGRATION_NAME,
        TRIAGE_EXECUTION_MIGRATION_CHECKSUM,
    ),
    (
        EVENT_HYPOTHESIS_SCHEMA_VERSION,
        EVENT_HYPOTHESIS_MIGRATION_NAME,
        EVENT_HYPOTHESIS_MIGRATION_CHECKSUM,
    ),
    (
        EVENT_HYPOTHESIS_RELATIONSHIP_SCHEMA_VERSION,
        EVENT_HYPOTHESIS_RELATIONSHIP_MIGRATION_NAME,
        EVENT_HYPOTHESIS_RELATIONSHIP_MIGRATION_CHECKSUM,
    ),
    (
        EVENT_HYPOTHESIS_LINEAGE_SCHEMA_VERSION,
        EVENT_HYPOTHESIS_LINEAGE_MIGRATION_NAME,
        EVENT_HYPOTHESIS_LINEAGE_MIGRATION_CHECKSUM,
    ),
    (
        STORY_CANDIDATE_SCHEMA_VERSION,
        STORY_CANDIDATE_MIGRATION_NAME,
        STORY_CANDIDATE_MIGRATION_CHECKSUM,
    ),
    (
        EVALUATION_FEEDBACK_SCHEMA_VERSION,
        EVALUATION_FEEDBACK_MIGRATION_NAME,
        EVALUATION_FEEDBACK_MIGRATION_CHECKSUM,
    ),
    (
        PLANNED_AGENDA_SCHEMA_VERSION,
        PLANNED_AGENDA_MIGRATION_NAME,
        PLANNED_AGENDA_MIGRATION_CHECKSUM,
    ),
    (
        BOUNDED_SEARCH_SCHEMA_VERSION,
        BOUNDED_SEARCH_MIGRATION_NAME,
        BOUNDED_SEARCH_MIGRATION_CHECKSUM,
    ),
    (
        COVERAGE_AUDIT_SCHEMA_VERSION,
        COVERAGE_AUDIT_MIGRATION_NAME,
        COVERAGE_AUDIT_MIGRATION_CHECKSUM,
    ),
    (
        LOCAL_WATCH_SCHEMA_VERSION,
        LOCAL_WATCH_MIGRATION_NAME,
        LOCAL_WATCH_MIGRATION_CHECKSUM,
    ),
    (
        INCREMENT8_EVALUATION_SCHEMA_VERSION,
        INCREMENT8_EVALUATION_MIGRATION_NAME,
        INCREMENT8_EVALUATION_MIGRATION_CHECKSUM,
    ),
    (
        INCREMENT8_OPERATIONAL_SCHEMA_VERSION,
        INCREMENT8_OPERATIONAL_MIGRATION_NAME,
        INCREMENT8_OPERATIONAL_MIGRATION_CHECKSUM,
    ),
    (
        INCREMENT8_RECOVERY_SCHEMA_VERSION,
        INCREMENT8_RECOVERY_MIGRATION_NAME,
        INCREMENT8_RECOVERY_MIGRATION_CHECKSUM,
    ),
    (
        GRAPHITI_EVALUATION_SCHEMA_VERSION,
        GRAPHITI_EVALUATION_MIGRATION_NAME,
        GRAPHITI_EVALUATION_MIGRATION_CHECKSUM,
    ),
    (
        GRAPHITI_ACCOUNTED_ZERO_SCHEMA_VERSION,
        GRAPHITI_ACCOUNTED_ZERO_MIGRATION_NAME,
        GRAPHITI_ACCOUNTED_ZERO_MIGRATION_CHECKSUM,
    ),
    (
        AUTHORISATION_SCOPE_CONTENT_SCHEMA_VERSION,
        AUTHORISATION_SCOPE_CONTENT_MIGRATION_NAME,
        AUTHORISATION_SCOPE_CONTENT_MIGRATION_CHECKSUM,
    ),
    (SECURITY_RECORD_SCHEMA_VERSION, SECURITY_RECORD_MIGRATION_NAME,
     SECURITY_RECORD_MIGRATION_CHECKSUM),
    (AUTHORIZATION_REQUEST_STORAGE_SCHEMA_VERSION,
     AUTHORIZATION_REQUEST_STORAGE_MIGRATION_NAME,
     AUTHORIZATION_REQUEST_STORAGE_MIGRATION_CHECKSUM),
    (
        GRAPHITI_RECOVERED_AMBIGUOUS_SCHEMA_VERSION,
        GRAPHITI_RECOVERED_AMBIGUOUS_MIGRATION_NAME,
        GRAPHITI_RECOVERED_AMBIGUOUS_MIGRATION_CHECKSUM,
    ),
    (
        RELATIONSHIP_OPEN_INDEX_SCHEMA_VERSION,
        RELATIONSHIP_OPEN_INDEX_MIGRATION_NAME,
        RELATIONSHIP_OPEN_INDEX_MIGRATION_CHECKSUM,
    ),
    (
        COMMAND_BOUND_STORAGE_SCHEMA_VERSION,
        COMMAND_BOUND_STORAGE_MIGRATION_NAME,
        COMMAND_BOUND_STORAGE_MIGRATION_CHECKSUM,
    ),
)
# fmt: on
