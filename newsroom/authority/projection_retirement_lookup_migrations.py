"""v43 persistent lookup indexes for bounded retired-success diagnostic expiry."""
from __future__ import annotations
import sqlite3
from .canonical import digest_canonical
from .authorisation_scope_content_migrations import AuthorisationScopeContentMigrationRecord

RETIREMENT_LOOKUP_SCHEMA_VERSION = 43
RETIREMENT_LOOKUP_MIGRATION_NAME = "native_projection_retirement_lookup_indexes_v43"
RETIREMENT_LOOKUP_PREDECESSOR_FINGERPRINT = "sha256:30bbcbc39e452773ae141b19794601c7ff8d294b09fa47624e561dc9de5ba104"
# Every prefix is an existing FK probe used by protection or expiry. No new
# source/business authority or speculative future-table index is introduced.
_REQUIRED_CHILD_PREFIXES = (
    ('authority_aggregate_versions', 'payload_id'),
    ('authority_audit_events', 'authorization_decision_id'),
    ('authority_audit_events', 'authorization_request_digest'),
    ('authority_commands', 'authentication_context_id'),
    ('authority_commands', 'authorization_decision_id'),
    ('authority_commands', 'authorization_request_digest'),
    ('authority_commands', 'payload_id'),
    ('authorization_decisions', 'authorization_request_digest'),
    ('authorization_requests', 'authentication_context_id'),
    ('blob_lifecycle_versions', 'event_id'),
    ('canonical_entities', 'authority_event_id'),
    ('canonical_entity_versions', 'authority_event_id'),
    ('editorial_relation_workflow_evidence', 'authority_event_id'),
    ('extraction_runs', 'created_by_event_id'),
    ('graphiti_replay_sources', 'approval_event_id'),
    ('hybrid_retrieval_attempts', 'authentication_context_id'),
    ('hybrid_retrieval_attempts', 'authorization_decision_id'),
    ('hybrid_retrieval_attempts', 'authorization_request_digest'),
    ('integrated_retrieval_contexts', 'fixture_event_id'),
    ('ledger_events', 'authorization_decision_id'),
    ('ledger_events', 'authorization_request_digest'),
    ('ledger_events', 'live_command_id'),
    ('ledger_events', 'payload_id'),
    ('object_access_decisions', 'authorization_decision_id'),
    ('object_access_decisions', 'authorization_request_digest'),
    ('object_admission_preflights', 'authorization_decision_id'),
    ('object_admission_preflights', 'authorization_request_digest'),
    ('object_admission_versions', 'event_id'),
    ('object_deletion_versions', 'event_id'),
    ('object_lifecycle_operations', 'authorization_decision_id'),
    ('object_lifecycle_operations', 'authorization_request_digest'),
    ('object_recovery_pin_versions', 'event_id'),
    ('object_rights_decisions', 'authorization_decision_id'),
    ('object_rights_decisions', 'authorization_request_digest'),
    ('projection_dead_letters', 'source_event_id'),
    ('projection_delivery_states', 'last_authority_event_id'),
    ('projection_delivery_states', 'source_event_id'),
    ('projection_gap_versions', 'authority_event_id'),
    ('projection_gaps', 'opened_event_id'),
    ('projection_gaps', 'resolved_event_id'),
    ('projection_generations', 'created_event_id'),
    ('projection_generations', 'updated_event_id'),
)
RETIREMENT_LOOKUP_MIGRATION_STATEMENTS = tuple(
    f'CREATE INDEX "idx_retirement_{table}_{column}" ON "{table}"("{column}")'
    for table, column in _REQUIRED_CHILD_PREFIXES
) + (
    "CREATE INDEX idx_retirement_causation ON ledger_events(causation_kind,causation_identifier,event_id)",
    "CREATE INDEX idx_retirement_validation_watermark ON projection_generation_validations(json_extract(CAST(canonical_bytes AS TEXT),'$.source_watermark_ledger_seq'))",
    "CREATE INDEX idx_retirement_retrieval_watermark ON hybrid_retrieval_attempts(authority_watermark)",
    "CREATE INDEX idx_retirement_last_business_seq ON ledger_events(ledger_seq) WHERE aggregate_type NOT IN ('projection_family','projection_generation')",
)
RETIREMENT_LOOKUP_MIGRATION_CHECKSUM = digest_canonical({"version": RETIREMENT_LOOKUP_SCHEMA_VERSION,
    "name": RETIREMENT_LOOKUP_MIGRATION_NAME, "statements": RETIREMENT_LOOKUP_MIGRATION_STATEMENTS})
RETIREMENT_LOOKUP_MIGRATION = AuthorisationScopeContentMigrationRecord(
    RETIREMENT_LOOKUP_SCHEMA_VERSION, RETIREMENT_LOOKUP_MIGRATION_NAME, RETIREMENT_LOOKUP_MIGRATION_CHECKSUM)


def migrate_retirement_lookup(connection: sqlite3.Connection, *, expected_history):
    from .migrations import schema_fingerprint
    if (not connection.in_transaction
        or connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1
        or schema_fingerprint(connection) != RETIREMENT_LOOKUP_PREDECESSOR_FINGERPRINT
        or tuple(tuple(row) for row in connection.execute(
            "SELECT version,name,checksum FROM authority_migrations ORDER BY version")) != expected_history):
        raise sqlite3.DatabaseError("v43 migration requires exact checked schema v42 and native foreign keys")
    for statement in RETIREMENT_LOOKUP_MIGRATION_STATEMENTS:
        connection.execute(statement)
