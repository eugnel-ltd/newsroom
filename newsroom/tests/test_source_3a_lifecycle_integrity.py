from __future__ import annotations

import inspect
import sqlite3

import pytest

from newsroom.authority.migrations import SCHEMA_VERSION
from newsroom.authority.persistence import AuthorityPersistenceError, AuthoritySchemaError
from newsroom.authority.source_registry_migrations import (
    SOURCE_REGISTRY_MIGRATION_CHECKSUM,
    SOURCE_REGISTRY_MIGRATION_NAME,
    SOURCE_REGISTRY_SCHEMA_VERSION,
)

from .source_3a_helpers import (
    DEFINITION_ID,
    definition_request,
    open_source_system,
    proof,
    version_request,
)


def test_source_system_open_retains_page_and_foreign_key_checks(tmp_path) -> None:
    from newsroom.authority._event_store_base import _EventStoreBase

    source = inspect.getsource(_EventStoreBase._validate_retained_database_history)
    assert "_validate_retained_database_history" in source
    assert _EventStoreBase._current_state_only is False
    history = inspect.getsource(_EventStoreBase._validate_retained_database_history)
    assert 'execute("PRAGMA quick_check")' in history
    database = tmp_path / "source.sqlite3"
    with open_source_system(database) as system:
        system.sources.register_definition(definition_request(), proof=proof())
    with sqlite3.connect(database) as connection:
        trigger = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='authority_aggregates_update_guard'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER authority_aggregates_update_guard")
        connection.execute("UPDATE authority_aggregates SET current_version=current_version+1")
        connection.execute(trigger)
        assert connection.execute("PRAGMA foreign_key_check").fetchone() is not None
    with pytest.raises(AuthoritySchemaError, match="foreign-key"):
        open_source_system(database)


def test_increment4_open_retains_row_integrity() -> None:
    from newsroom.authority._event_store_base import _EventStoreBase
    from newsroom.authority._graphiti_increment4_system import (
        _GraphitiIncrement4AuthorityStore,
    )

    assert (
        _GraphitiIncrement4AuthorityStore._should_validate_row_integrity
        is _EventStoreBase._should_validate_row_integrity
    )
    base = inspect.getsource(_EventStoreBase._validate_schema_and_integrity)
    assert "_validate_retained_database_history" in base
    history = inspect.getsource(_EventStoreBase._validate_retained_database_history)
    assert "_validate_immutable_records" in history
    assert "_validate_relational_invariants" in history


def test_checked_source_registry_migration_is_retained_in_v11(
    tmp_path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    system = open_source_system(database)
    system.close()

    conn = sqlite3.connect(database)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == (
            SCHEMA_VERSION
        )
        row = conn.execute(
            "SELECT name,checksum FROM authority_migrations WHERE version=?",
            (SOURCE_REGISTRY_SCHEMA_VERSION,),
        ).fetchone()
        assert row == (
            SOURCE_REGISTRY_MIGRATION_NAME,
            SOURCE_REGISTRY_MIGRATION_CHECKSUM,
        )
    finally:
        conn.close()

    open_source_system(database).close()


def test_immutable_source_rows_reject_update_and_delete(tmp_path) -> None:
    database = tmp_path / "authority.sqlite3"
    system = open_source_system(database)
    system.sources.register_definition(definition_request(), proof=proof())
    system.close()

    conn = sqlite3.connect(database)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE source_definitions SET name='changed' WHERE definition_id=?",
                (str(DEFINITION_ID),),
            )
        with pytest.raises(sqlite3.IntegrityError, match="retained"):
            conn.execute(
                "DELETE FROM source_definitions WHERE definition_id=?",
                (str(DEFINITION_ID),),
            )
    finally:
        conn.close()


def test_startup_detects_source_canonical_tampering(tmp_path) -> None:
    database = tmp_path / "authority.sqlite3"
    system = open_source_system(database)
    system.sources.register_definition(definition_request(), proof=proof())
    system.sources.record_definition_version(
        version_request(), proof=proof()
    )
    system.close()

    conn = sqlite3.connect(database)
    try:
        trigger_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
            ("immutable_source_definition_update",),
        ).fetchone()[0]
        conn.execute("DROP TRIGGER immutable_source_definition_update")
        conn.execute(
            "UPDATE source_definitions SET canonical_digest=? "
            "WHERE definition_id=?",
            ("sha256:" + "0" * 64, str(DEFINITION_ID)),
        )
        conn.execute(trigger_sql)
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(AuthorityPersistenceError, match="canonical digest"):
        open_source_system(database)
