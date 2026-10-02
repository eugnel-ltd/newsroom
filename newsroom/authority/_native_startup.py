"""Native boot validates durable heads; exact business proofs remain on use."""
from __future__ import annotations

import sqlite3

from .persistence import AuthorityPersistenceError


def validate_current_heads(connection: sqlite3.Connection) -> None:
    """Check current pointers without decoding any unused historical bodies."""
    checks = (
        ("object admission", "SELECT h.admission_id FROM object_admission_heads h "
         "LEFT JOIN object_admissions a ON a.admission_id=h.admission_id "
         "LEFT JOIN object_admission_versions v ON v.admission_id=h.admission_id "
         "AND v.lifecycle_version=h.current_version "
         "WHERE a.admission_id IS NULL OR v.admission_id IS NULL LIMIT 1"),
        ("blob lifecycle", "SELECT h.blob_digest FROM blob_lifecycle_heads h "
         "LEFT JOIN blob_identities b ON b.blob_digest=h.blob_digest "
         "LEFT JOIN blob_lifecycle_versions v ON v.blob_digest=h.blob_digest "
         "AND v.lifecycle_version=h.current_version "
         "WHERE b.blob_digest IS NULL OR v.blob_digest IS NULL LIMIT 1"),
        ("object deletion", "SELECT h.deletion_id FROM object_deletion_heads h "
         "LEFT JOIN object_deletions d ON d.deletion_id=h.deletion_id "
         "LEFT JOIN object_deletion_versions v ON v.deletion_id=h.deletion_id "
         "AND v.lifecycle_version=h.current_version "
         "WHERE d.deletion_id IS NULL OR v.deletion_id IS NULL LIMIT 1"),
        ("object recovery pin", "SELECT h.pin_id FROM object_recovery_pin_heads h "
         "LEFT JOIN object_recovery_pins p ON p.pin_id=h.pin_id "
         "LEFT JOIN object_recovery_pin_versions v ON v.pin_id=h.pin_id "
         "AND v.lifecycle_version=h.current_version "
         "WHERE p.pin_id IS NULL OR v.pin_id IS NULL LIMIT 1"),
        ("source", "SELECT h.definition_id FROM source_definition_version_heads h "
         "LEFT JOIN source_definitions d ON d.definition_id=h.definition_id "
         "LEFT JOIN source_definition_versions v ON v.version_id=h.current_version_id "
         "AND v.definition_id=h.definition_id AND v.version_number=h.current_version_number "
         "WHERE d.definition_id IS NULL OR v.version_id IS NULL LIMIT 1"),
        ("canonical entity", "SELECT h.entity_id FROM canonical_entity_heads h "
         "LEFT JOIN canonical_entities e ON e.entity_id=h.entity_id "
         "LEFT JOIN canonical_entity_versions v ON v.entity_version_id=h.current_entity_version_id "
         "AND v.entity_id=h.entity_id AND v.version_number=h.current_version_number "
         "AND v.lifecycle=h.lifecycle WHERE e.entity_id IS NULL OR v.entity_version_id IS NULL LIMIT 1"),
        ("Work Item", "SELECT h.work_item_id FROM triage_work_item_heads h "
         "LEFT JOIN triage_work_items i ON i.work_item_id=h.work_item_id "
         "LEFT JOIN triage_work_item_versions v ON v.version_id=h.current_version_id "
         "AND v.work_item_id=h.work_item_id AND v.ordinal=h.current_ordinal "
         "AND v.canonical_digest=h.current_version_digest WHERE i.work_item_id IS NULL OR v.version_id IS NULL LIMIT 1"),
        ("Hypothesis", "SELECT h.hypothesis_id FROM event_hypothesis_heads_v2 h "
         "LEFT JOIN event_hypotheses_v2 i ON i.hypothesis_id=h.hypothesis_id "
         "LEFT JOIN event_hypothesis_versions_v2 v ON v.version_id=h.version_id "
         "AND v.hypothesis_id=h.hypothesis_id AND v.ordinal=h.ordinal "
         "AND v.canonical_digest=h.version_digest WHERE i.hypothesis_id IS NULL OR v.version_id IS NULL LIMIT 1"),
        ("Hypothesis lineage", "SELECT h.hypothesis_id FROM event_hypothesis_lineage_heads h "
         "LEFT JOIN event_hypothesis_versions_v2 v ON v.version_id=h.version_id "
         "AND v.hypothesis_id=h.hypothesis_id AND v.canonical_digest=h.version_digest "
         "LEFT JOIN event_hypothesis_lineage l ON l.lineage_id=h.producing_lineage_id "
         "WHERE v.version_id IS NULL OR l.lineage_id IS NULL LIMIT 1"),
        ("Candidate", "SELECT h.candidate_id FROM story_candidate_heads h "
         "LEFT JOIN story_candidate_admission_receipts_v2 r ON r.admission_digest=h.current_admission_digest "
         "AND r.candidate_id=h.candidate_id AND r.version_id=h.current_version_id "
         "AND r.version_ordinal=h.current_version_ordinal AND r.version_digest=h.current_version_digest "
         "LEFT JOIN story_candidate_collision_bindings b ON b.collision_namespace=h.collision_namespace "
         "AND b.collision_key_digest=h.collision_key_digest "
         "WHERE r.admission_digest IS NULL OR b.candidate_id IS NULL OR b.candidate_id!=h.candidate_id LIMIT 1"),
    )
    for kind, query in checks:
        if connection.execute(query).fetchone() is not None:
            raise AuthorityPersistenceError(f"current {kind} head differs from durable state")
