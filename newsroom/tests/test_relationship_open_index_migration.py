from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from newsroom.authority._event_hypothesis_relationship_system import (
    _verify_relationship_event_coverage,
)
from newsroom.authority.migrations import (
    EXPECTED_SCHEMA_FINGERPRINT,
    SCHEMA_VERSION,
    apply_pending_migrations,
    schema_fingerprint,
)
from newsroom.authority.persistence import AuthoritySchemaError
from newsroom.authority.relationship_open_index_migrations import (
    RELATIONSHIP_OPEN_INDEX_MIGRATION_CHECKSUM,
    RELATIONSHIP_OPEN_INDEX_MIGRATION_NAME,
    RELATIONSHIP_OPEN_INDEX_MIGRATION_STATEMENTS,
    RELATIONSHIP_OPEN_INDEX_PREDECESSOR_SCHEMA_VERSION,
    RELATIONSHIP_OPEN_INDEX_SCHEMA_VERSION,
)
from newsroom.increment6.relationships import (
    RELATIONSHIP_AGGREGATE_TYPE,
    RELATIONSHIP_EVENT_TYPE,
)
from newsroom.tests.authority_migration_compatibility import (
    build_exact_prefix,
    canonical_cell,
    inspect_exact_prefix,
)


def _coverage_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE ledger_events("
        "event_id TEXT PRIMARY KEY,event_type TEXT NOT NULL,"
        "aggregate_type TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE event_hypothesis_relationship_decisions("
        "decision_id TEXT PRIMARY KEY,authority_event_id TEXT NOT NULL UNIQUE)"
    )
    connection.execute(RELATIONSHIP_OPEN_INDEX_MIGRATION_STATEMENTS[0])
    return connection


def test_fresh_and_v39_upgrade_have_exact_partial_index_schema_identity(
    tmp_path: Path,
) -> None:
    fresh = sqlite3.connect(tmp_path / "fresh.sqlite3", isolation_level=None)
    try:
        fresh.execute("PRAGMA foreign_keys=ON")
        apply_pending_migrations(fresh, applied_at="2042-01-01T00:00:00.000000Z")
        assert fresh.execute("PRAGMA user_version").fetchone() == (
            SCHEMA_VERSION,
        )
        assert SCHEMA_VERSION >= RELATIONSHIP_OPEN_INDEX_SCHEMA_VERSION
        assert fresh.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='index' "
            "AND name='idx_ledger_events_relationship_event_type'"
        ).fetchone() == (
            "idx_ledger_events_relationship_event_type",
            RELATIONSHIP_OPEN_INDEX_MIGRATION_STATEMENTS[0],
        )
        plan = fresh.execute(
            "EXPLAIN QUERY PLAN SELECT e.event_id FROM ledger_events e "
            "LEFT JOIN event_hypothesis_relationship_decisions r "
            "ON r.authority_event_id=e.event_id WHERE e.event_type=? "
            "AND r.decision_id IS NULL",
            (RELATIONSHIP_EVENT_TYPE,),
        ).fetchall()
        assert any(
            "idx_ledger_events_relationship_event_type" in str(row)
            for row in plan
        )
        assert schema_fingerprint(fresh) == EXPECTED_SCHEMA_FINGERPRINT
    finally:
        fresh.close()

    upgraded = tmp_path / "upgraded-v39.sqlite3"
    predecessor = build_exact_prefix(
        upgraded, RELATIONSHIP_OPEN_INDEX_PREDECESSOR_SCHEMA_VERSION
    )
    connection = sqlite3.connect(upgraded, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        apply_pending_migrations(
            connection, applied_at="2042-01-01T00:00:01.000000Z"
        )
    finally:
        connection.close()
    assert predecessor.version == RELATIONSHIP_OPEN_INDEX_PREDECESSOR_SCHEMA_VERSION
    assert inspect_exact_prefix(
        upgraded, expected_version=SCHEMA_VERSION
    ) == canonical_cell(SCHEMA_VERSION)


def test_partial_index_migration_checksum_pins_exact_statement() -> None:
    assert RELATIONSHIP_OPEN_INDEX_MIGRATION_NAME == "relationship_open_index_v40"
    assert RELATIONSHIP_OPEN_INDEX_MIGRATION_CHECKSUM.startswith("sha256:")
    assert RELATIONSHIP_OPEN_INDEX_MIGRATION_STATEMENTS == (
        "CREATE INDEX idx_ledger_events_relationship_event_type "
        "ON ledger_events(event_id) "
        "WHERE event_type='event_hypothesis_relationship_decision_retained'",
    )


@pytest.mark.parametrize(
    ("corruption", "aggregate_type"),
    (
        ("orphan_relationship_event", None),
        ("orphan_relationship_event_wrong_aggregate", None),
        ("relationship_decision_wrong_event", None),
        ("relationship_decision_wrong_aggregate", RELATIONSHIP_AGGREGATE_TYPE),
    ),
)
def test_partial_index_keeps_relationship_event_corruption_detection(
    corruption: str, aggregate_type: str | None
) -> None:
    connection = _coverage_connection()
    try:
        if corruption.startswith("orphan_relationship_event"):
            connection.execute(
                "INSERT INTO ledger_events VALUES(?,?,?)",
                ("orphan", RELATIONSHIP_EVENT_TYPE,
                 "other_aggregate" if corruption.endswith("wrong_aggregate")
                 else RELATIONSHIP_AGGREGATE_TYPE),
            )
        elif corruption == "relationship_decision_wrong_event":
            connection.execute(
                "INSERT INTO ledger_events VALUES(?,?,?)",
                ("wrong-event", "other_event", RELATIONSHIP_AGGREGATE_TYPE),
            )
            connection.execute(
                "INSERT INTO event_hypothesis_relationship_decisions VALUES(?,?)",
                ("decision", "wrong-event"),
            )
        else:
            connection.execute(
                "INSERT INTO ledger_events VALUES(?,?,?)",
                ("wrong-aggregate", RELATIONSHIP_EVENT_TYPE, "other_aggregate"),
            )
            connection.execute(
                "INSERT INTO event_hypothesis_relationship_decisions VALUES(?,?)",
                ("decision", "wrong-aggregate"),
            )

        with pytest.raises(AuthoritySchemaError, match="coverage differs"):
            _verify_relationship_event_coverage(
                connection, aggregate_type=aggregate_type
            )
    finally:
        connection.close()
