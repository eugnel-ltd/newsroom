from __future__ import annotations

import json
import sqlite3

import pytest

from newsroom.authority.event_hypothesis_relationship_migrations import (
    EVENT_HYPOTHESIS_RELATIONSHIP_SCHEMA_VERSION,
)
from newsroom.authority.migrations import apply_pending_migrations

from .graphiti_adapter_4d_migration_helpers import (
    drop_empty_v22_relationship_schema,
    drop_empty_v23_lineage_schema,
)

RELATIONSHIP_TRIGGERS = (
    "retained_event_hypothesis_relationship_delete",
    "immutable_event_hypothesis_relationship_update",
    "event_hypothesis_relationship_coherence",
)


def _fresh() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    apply_pending_migrations(connection, applied_at="2042-03-12T10:00:00Z")
    return connection


def _retained_state(connection: sqlite3.Connection) -> tuple[object, ...]:
    return (
        connection.execute("PRAGMA user_version").fetchone()[0],
        connection.execute(
            "SELECT version,name,checksum,applied_at "
            "FROM authority_migrations ORDER BY version"
        ).fetchall(),
        connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
        ).fetchall(),
        connection.execute(
            "SELECT * FROM event_hypothesis_relationship_decisions"
        ).fetchall(),
    )


@pytest.mark.parametrize("missing_trigger", RELATIONSHIP_TRIGGERS)
def test_v22_downgrade_preflight_preserves_state_when_trigger_is_missing(
    missing_trigger: str,
) -> None:
    connection = _fresh()
    connection.execute(f'DROP TRIGGER "{missing_trigger}"')
    before = _retained_state(connection)

    with pytest.raises(sqlite3.DatabaseError, match="exact v22 relationship schema"):
        drop_empty_v22_relationship_schema(connection)

    assert _retained_state(connection) == before


def test_v25_downgrade_preflight_preserves_state_when_trigger_sql_differs() -> None:
    connection = _fresh()
    connection.execute("DROP TRIGGER immutable_evaluation_feedback")
    connection.execute(
        "CREATE TRIGGER immutable_evaluation_feedback BEFORE UPDATE ON evaluation_feedback "
        "WHEN 1=1 BEGIN SELECT RAISE(ABORT,'immutable Evaluation Feedback'); END"
    )
    before = _retained_state(connection)

    with pytest.raises(sqlite3.DatabaseError, match="exact empty v25 Feedback schema"):
        drop_empty_v23_lineage_schema(connection)

    assert _retained_state(connection) == before


def test_v22_downgrade_preflight_preserves_state_when_table_is_not_empty() -> None:
    connection = _fresh()
    assessment_digest = "sha256:" + "1" * 64
    assessment = json.dumps(
        {
            "schema_version": "newsroom.increment6.hypothesis-relationship-decision.v1",
            "subject": {
                "hypothesis_id": "hypothesis:test",
                "version_id": "hypothesis-version:test",
                "version_digest": "sha256:" + "2" * 64,
            },
            "decision": "REL_NO_ADEQUATE_PRIOR_MATCH",
            "comparator": {"version_id": None},
        },
        separators=(",", ":"),
    ).encode()
    # Deliberate orphan injection exercises emptiness preflight, not valid
    # authority adoption. Normal migrations retain native foreign keys.
    connection.execute("PRAGMA foreign_keys=OFF")
    connection.execute(
        "INSERT INTO event_hypothesis_relationship_decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            assessment_digest,
            "aggregate:test",
            "event:test",
            "hypothesis:test",
            "hypothesis-version:test",
            "sha256:" + "2" * 64,
            b"{}",
            "sha256:" + "3" * 64,
            None,
            None,
            None,
            "REL_NO_ADEQUATE_PRIOR_MATCH",
            assessment,
            assessment_digest,
            b"[]",
            "sha256:" + "4" * 64,
            "sha256:" + "5" * 64,
            "2042-03-12T10:00:00Z",
        ),
    )
    connection.execute("PRAGMA foreign_keys=ON")
    before = _retained_state(connection)

    with pytest.raises(sqlite3.DatabaseError, match="must be empty"):
        drop_empty_v22_relationship_schema(connection)

    assert _retained_state(connection) == before


def test_v22_downgrade_preflight_preserves_state_when_history_is_corrupt() -> None:
    connection = _fresh()
    guard = connection.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type='trigger' AND name='immutable_authority_migrations_update'"
    ).fetchone()[0]
    connection.execute("DROP TRIGGER immutable_authority_migrations_update")
    connection.execute(
        "UPDATE authority_migrations SET checksum=? WHERE version=?",
        (
            "sha256:" + "0" * 64,
            EVENT_HYPOTHESIS_RELATIONSHIP_SCHEMA_VERSION,
        ),
    )
    connection.execute(guard)
    before = _retained_state(connection)

    with pytest.raises(sqlite3.DatabaseError, match="exact v22 migration history"):
        drop_empty_v22_relationship_schema(connection)

    assert _retained_state(connection) == before


def test_v22_downgrade_preflight_preserves_state_when_version_differs_from_history() -> (
    None
):
    connection = _fresh()
    connection.execute(
        f"PRAGMA user_version={EVENT_HYPOTHESIS_RELATIONSHIP_SCHEMA_VERSION - 1}"
    )
    before = _retained_state(connection)

    with pytest.raises(sqlite3.DatabaseError, match="version/history mismatch"):
        drop_empty_v22_relationship_schema(connection)

    assert _retained_state(connection) == before
