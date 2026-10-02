from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from newsroom.authority.evaluation_handoff_migrations import (
    EVALUATION_HANDOFF_SCHEMA_VERSION,
)
from newsroom.authority.event_hypothesis_lineage_migrations import (
    EVENT_HYPOTHESIS_LINEAGE_MIGRATION_CHECKSUM,
    EVENT_HYPOTHESIS_LINEAGE_MIGRATION_NAME,
    EVENT_HYPOTHESIS_LINEAGE_MIGRATION_STATEMENTS,
    EVENT_HYPOTHESIS_LINEAGE_SCHEMA_VERSION,
)
from newsroom.authority.event_hypothesis_migrations import (
    EVENT_HYPOTHESIS_SCHEMA_VERSION,
)
from newsroom.authority.event_hypothesis_relationship_migrations import (
    EVENT_HYPOTHESIS_RELATIONSHIP_MIGRATION_CHECKSUM,
    EVENT_HYPOTHESIS_RELATIONSHIP_MIGRATION_NAME,
    EVENT_HYPOTHESIS_RELATIONSHIP_SCHEMA_VERSION,
)
from newsroom.authority.graphiti_adapter_migrations import (
    GRAPHITI_ADAPTER_SCHEMA_VERSION,
)
from newsroom.authority.triage_disposition_migrations import (
    TRIAGE_DISPOSITION_SCHEMA_VERSION,
)
from newsroom.authority.triage_execution_migrations import (
    TRIAGE_EXECUTION_SCHEMA_VERSION,
)
from newsroom.authority.triage_work_item_migrations import (
    TRIAGE_WORK_ITEM_SCHEMA_VERSION,
)


def drop_empty_v29_local_watch_schema(connection: sqlite3.Connection) -> None:
    """Remove the exact empty v29 Local Watch suffix atomically."""
    savepoint = "checked_local_watch_downgrade"
    connection.execute(f"SAVEPOINT {savepoint}")
    try:
        _drop_empty_v29_local_watch_schema(connection)
    except Exception:
        connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise
    connection.execute(f"RELEASE SAVEPOINT {savepoint}")


def _drop_empty_v29_local_watch_schema(connection: sqlite3.Connection) -> None:
    _drop_empty_v30_evaluation_schema(connection)
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) == 29:
        from newsroom.authority.local_watch_migrations import (
            LOCAL_WATCH_MIGRATION_CHECKSUM,
            LOCAL_WATCH_MIGRATION_NAME,
            LOCAL_WATCH_MIGRATION_STATEMENTS,
        )

        def normalise_local_watch_sql(value: str) -> str:
            return " ".join(value.split()).replace(" IF NOT EXISTS", "")

        local_watch_tables = (
            "event_scoped_local_watches",
            "event_scoped_local_watch_versions",
            "event_scoped_local_watch_heads",
            "event_scoped_local_watch_closures",
        )
        placeholders = ",".join("?" for _ in local_watch_tables)
        objects = connection.execute(
            f"SELECT type,name,sql FROM sqlite_master WHERE tbl_name IN ({placeholders}) "
            "AND type IN ('table','trigger','index') AND sql IS NOT NULL",
            local_watch_tables,
        ).fetchall()
        if connection.execute(
            "SELECT name,checksum FROM authority_migrations WHERE version=29"
        ).fetchone() != (
            LOCAL_WATCH_MIGRATION_NAME,
            LOCAL_WATCH_MIGRATION_CHECKSUM,
        ) or {normalise_local_watch_sql(str(row[2])) for row in objects} != {
            normalise_local_watch_sql(statement)
            for statement in LOCAL_WATCH_MIGRATION_STATEMENTS
        }:
            raise sqlite3.DatabaseError(
                "downgrade requires exact empty v29 Local Watch schema"
            )
        for table in local_watch_tables:
            if connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone() != (0,):
                raise sqlite3.DatabaseError("v29 Local Watch tables must be empty")
        guard = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' "
            "AND name='immutable_authority_migrations_delete'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
        for object_type, name, _ in objects:
            if object_type == "trigger":
                connection.execute(f'DROP TRIGGER "{name}"')
        for table in reversed(local_watch_tables):
            connection.execute(f'DROP TABLE "{table}"')
        connection.execute("DELETE FROM authority_migrations WHERE version=29")
        connection.execute(guard)
        connection.execute("PRAGMA user_version=28")


def _drop_empty_v30_evaluation_schema(connection: sqlite3.Connection) -> None:
    _drop_empty_v31_operational_schema(connection)
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) != 30:
        return
    from newsroom.authority.increment8_evaluation_migrations import (
        INCREMENT8_EVALUATION_MIGRATION_CHECKSUM,
        INCREMENT8_EVALUATION_MIGRATION_NAME,
        INCREMENT8_EVALUATION_MIGRATION_STATEMENTS,
    )

    def normalise(value: str) -> str:
        return " ".join(value.split()).replace(" IF NOT EXISTS", "")

    tables = (
        "evaluation_plans",
        "evaluation_epochs",
        "evaluation_runs",
        "evaluation_cases",
        "evaluation_labels",
        "evaluation_adjudications",
        "evaluation_release_decisions",
    )
    placeholders = ",".join("?" for _ in tables)
    objects = connection.execute(
        f"SELECT type,name,sql FROM sqlite_master WHERE tbl_name IN ({placeholders}) "
        "AND type IN ('table','trigger','index') AND sql IS NOT NULL",
        tables,
    ).fetchall()
    if connection.execute(
        "SELECT name,checksum FROM authority_migrations WHERE version=30"
    ).fetchone() != (
        INCREMENT8_EVALUATION_MIGRATION_NAME,
        INCREMENT8_EVALUATION_MIGRATION_CHECKSUM,
    ) or {normalise(str(row[2])) for row in objects} != {
        normalise(statement) for statement in INCREMENT8_EVALUATION_MIGRATION_STATEMENTS
    }:
        raise sqlite3.DatabaseError(
            "downgrade requires exact empty v30 evaluation schema"
        )
    for table in tables:
        if connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone() != (0,):
            raise sqlite3.DatabaseError("v30 evaluation tables must be empty")
    guard = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' "
        "AND name='immutable_authority_migrations_delete'"
    ).fetchone()[0]
    connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
    for object_type, name, _ in objects:
        if object_type == "trigger":
            connection.execute(f'DROP TRIGGER "{name}"')
    for table in reversed(tables):
        connection.execute(f'DROP TABLE "{table}"')
    connection.execute("DELETE FROM authority_migrations WHERE version=30")
    connection.execute(guard)
    connection.execute("PRAGMA user_version=29")


def _drop_empty_v31_operational_schema(connection: sqlite3.Connection) -> None:
    _drop_empty_v32_recovery_schema(connection)
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) != 31:
        return
    from newsroom.authority.increment8_operational_migrations import (
        INCREMENT8_OPERATIONAL_MIGRATION_CHECKSUM,
        INCREMENT8_OPERATIONAL_MIGRATION_NAME,
        INCREMENT8_OPERATIONAL_MIGRATION_STATEMENTS,
        INCREMENT8_OPERATIONAL_TABLES,
    )

    def normalise(value: str) -> str:
        return " ".join(value.split()).replace(" IF NOT EXISTS", "")

    tables = INCREMENT8_OPERATIONAL_TABLES
    placeholders = ",".join("?" for _ in tables)
    objects = connection.execute(
        f"SELECT type,name,sql FROM sqlite_master WHERE tbl_name IN ({placeholders}) "
        "AND type IN ('table','trigger','index') AND sql IS NOT NULL",
        tables,
    ).fetchall()
    if connection.execute(
        "SELECT name,checksum FROM authority_migrations WHERE version=31"
    ).fetchone() != (
        INCREMENT8_OPERATIONAL_MIGRATION_NAME,
        INCREMENT8_OPERATIONAL_MIGRATION_CHECKSUM,
    ) or {normalise(str(row[2])) for row in objects} != {
        normalise(statement) for statement in INCREMENT8_OPERATIONAL_MIGRATION_STATEMENTS
    }:
        raise sqlite3.DatabaseError(
            "downgrade requires exact empty v31 operational schema"
        )
    for table in tables:
        if connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone() != (0,):
            raise sqlite3.DatabaseError("v31 operational tables must be empty")
    guard = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' "
        "AND name='immutable_authority_migrations_delete'"
    ).fetchone()[0]
    connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
    for object_type, name, _ in objects:
        if object_type == "trigger":
            connection.execute(f'DROP TRIGGER "{name}"')
    for table in reversed(tables):
        connection.execute(f'DROP TABLE "{table}"')
    connection.execute("DELETE FROM authority_migrations WHERE version=31")
    connection.execute(guard)
    connection.execute("PRAGMA user_version=30")


def _drop_empty_v32_recovery_schema(connection: sqlite3.Connection) -> None:
    _drop_empty_v34_graphiti_evaluation_schema(connection)
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) != 32:
        return
    from newsroom.authority.increment8_recovery_migrations import (
        INCREMENT8_RECOVERY_MIGRATION_CHECKSUM,
        INCREMENT8_RECOVERY_MIGRATION_NAME,
        INCREMENT8_RECOVERY_MIGRATION_STATEMENTS,
        INCREMENT8_RECOVERY_TABLES,
    )

    def normalise(value: str) -> str:
        return " ".join(value.split()).replace(" IF NOT EXISTS", "")

    tables = INCREMENT8_RECOVERY_TABLES
    placeholders = ",".join("?" for _ in tables)
    objects = connection.execute(
        f"SELECT type,name,sql FROM sqlite_master WHERE tbl_name IN ({placeholders}) "
        "AND type IN ('table','trigger','index') AND sql IS NOT NULL",
        tables,
    ).fetchall()
    if connection.execute(
        "SELECT name,checksum FROM authority_migrations WHERE version=32"
    ).fetchone() != (
        INCREMENT8_RECOVERY_MIGRATION_NAME,
        INCREMENT8_RECOVERY_MIGRATION_CHECKSUM,
    ) or {normalise(str(row[2])) for row in objects} != {
        normalise(statement) for statement in INCREMENT8_RECOVERY_MIGRATION_STATEMENTS
    }:
        raise sqlite3.DatabaseError("downgrade requires exact empty v32 recovery schema")
    for table in tables:
        if connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone() != (0,):
            raise sqlite3.DatabaseError("v32 recovery tables must be empty")
    guard = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' "
        "AND name='immutable_authority_migrations_delete'"
    ).fetchone()[0]
    connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
    for object_type, name, _ in objects:
        if object_type == "trigger":
            connection.execute(f'DROP TRIGGER "{name}"')
    for table in reversed(tables):
        connection.execute(f'DROP TABLE "{table}"')
    connection.execute("DELETE FROM authority_migrations WHERE version=32")
    connection.execute(guard)
    connection.execute("PRAGMA user_version=31")


def _drop_v37_security_record_schema(connection: sqlite3.Connection) -> None:
    """Restore exact v36 context bytes without changing any retained identity."""
    _drop_v38_authorization_request_storage(connection)
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) != 37:
        return
    from newsroom.authority._event_store import _EventAuthorityStore
    from newsroom.authority.migrations import MIGRATION_STATEMENTS
    from newsroom.authority.security_record_migrations import (
        SECURITY_RECORD_MIGRATION_CHECKSUM, SECURITY_RECORD_MIGRATION_NAME,
    )
    if connection.execute(
        "SELECT name,checksum FROM authority_migrations WHERE version=37"
    ).fetchone() != (SECURITY_RECORD_MIGRATION_NAME, SECURITY_RECORD_MIGRATION_CHECKSUM):
        raise sqlite3.DatabaseError("downgrade requires exact v37 context authority")
    reader = object.__new__(_EventAuthorityStore)
    cursor = connection.execute("SELECT * FROM authentication_contexts")
    names = tuple(item[0] for item in cursor.description)
    rows = []
    for raw in cursor:
        row = dict(zip(names, raw, strict=True))
        canonical = reader._authentication_record_from_row(row).canonical_bytes
        rows.append(tuple(canonical if name == "storage_context_marker" else row[name] for name in names))
    connection.execute("SAVEPOINT checked_context_downgrade")
    connection.execute("PRAGMA defer_foreign_keys=ON")
    try:
        connection.execute("DROP TABLE authentication_contexts")
        connection.execute(next(sql for sql in MIGRATION_STATEMENTS
                                if sql.startswith("CREATE TABLE authentication_contexts(")))
        for row in rows:
            connection.execute("INSERT INTO authentication_contexts VALUES(?,?,?,?,?,?,?,?,?,?)", row)
        for suffix in ("update", "delete"):
            connection.execute(next(sql for sql in MIGRATION_STATEMENTS
                                    if sql.startswith(f"CREATE TRIGGER immutable_authentication_contexts_{suffix}")))
        guard = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='immutable_authority_migrations_delete'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
        connection.execute("DELETE FROM authority_migrations WHERE version=37")
        connection.execute(guard)
        connection.execute("PRAGMA user_version=36")
    except Exception:
        connection.execute("ROLLBACK TO SAVEPOINT checked_context_downgrade")
        connection.execute("RELEASE SAVEPOINT checked_context_downgrade")
        raise
    connection.execute("RELEASE SAVEPOINT checked_context_downgrade")


def _drop_v38_authorization_request_storage(
    connection: sqlite3.Connection,
) -> None:
    """Restore exact v37 request bytes for retained-prefix fixtures."""

    _drop_v39_recovered_ambiguous_guards(connection)
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) != 38:
        return
    from newsroom.authority._event_store import _EventAuthorityStore
    from newsroom.authority.authorization_request_storage_migrations import (
        AUTHORIZATION_REQUEST_STORAGE_MIGRATION_CHECKSUM,
        AUTHORIZATION_REQUEST_STORAGE_MIGRATION_NAME,
    )
    from newsroom.authority.migrations import MIGRATION_STATEMENTS

    if connection.execute(
        "SELECT name,checksum FROM authority_migrations WHERE version=38"
    ).fetchone() != (
        AUTHORIZATION_REQUEST_STORAGE_MIGRATION_NAME,
        AUTHORIZATION_REQUEST_STORAGE_MIGRATION_CHECKSUM,
    ):
        raise sqlite3.DatabaseError("downgrade requires exact v38 request authority")
    reader = object.__new__(_EventAuthorityStore)
    cursor = connection.execute("SELECT rowid,* FROM authorization_requests")
    names = tuple(item[0] for item in cursor.description)
    rows = tuple(dict(zip(names, row, strict=True)) for row in cursor)
    connection.execute("SAVEPOINT checked_request_downgrade")
    try:
        connection.execute("DROP TRIGGER immutable_authorization_requests_update")
        connection.execute("DROP TRIGGER authorization_request_storage_guard")
        for row in rows:
            connection.execute(
                "UPDATE authorization_requests SET storage_request_residual=? "
                "WHERE rowid=?",
                (reader._request_record_from_row(row).canonical_bytes, row["rowid"]),
            )
        connection.execute(
            "ALTER TABLE authorization_requests DROP COLUMN storage_request_marker"
        )
        connection.execute(
            "ALTER TABLE authorization_requests RENAME COLUMN "
            "storage_request_residual TO canonical_bytes"
        )
        connection.execute(
            next(
                sql
                for sql in MIGRATION_STATEMENTS
                if sql.startswith(
                    "CREATE TRIGGER immutable_authorization_requests_update"
                )
            )
        )
        guard = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE name='immutable_authority_migrations_delete'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
        connection.execute("DELETE FROM authority_migrations WHERE version=38")
        connection.execute(guard)
        connection.execute("PRAGMA user_version=37")
    except Exception:
        connection.execute("ROLLBACK TO SAVEPOINT checked_request_downgrade")
        connection.execute("RELEASE SAVEPOINT checked_request_downgrade")
        raise
    connection.execute("RELEASE SAVEPOINT checked_request_downgrade")


def _drop_v39_recovered_ambiguous_guards(
    connection: sqlite3.Connection,
) -> None:
    """Restore the exact v38 Graphiti progression triggers."""

    _drop_v40_relationship_open_index(connection)
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) != 39:
        return
    from newsroom.authority.graphiti_accounted_zero_migrations import (
        GRAPHITI_ACCOUNTED_ZERO_MIGRATION_STATEMENTS,
    )
    from newsroom.authority.graphiti_adapter_migrations import (
        GRAPHITI_ADAPTER_MIGRATION_STATEMENTS,
    )
    from newsroom.authority.graphiti_recovered_ambiguous_migrations import (
        GRAPHITI_RECOVERED_AMBIGUOUS_MIGRATION_CHECKSUM,
        GRAPHITI_RECOVERED_AMBIGUOUS_MIGRATION_NAME,
    )

    if connection.execute(
        "SELECT name,checksum FROM authority_migrations WHERE version=39"
    ).fetchone() != (
        GRAPHITI_RECOVERED_AMBIGUOUS_MIGRATION_NAME,
        GRAPHITI_RECOVERED_AMBIGUOUS_MIGRATION_CHECKSUM,
    ):
        raise sqlite3.DatabaseError("downgrade requires exact v39 Graphiti guards")
    connection.execute("SAVEPOINT checked_graphiti_guard_downgrade")
    try:
        connection.execute("DROP TRIGGER graphiti_attempt_chain_guard")
        connection.execute("DROP TRIGGER graphiti_attempt_head_update_guard")
        connection.execute(
            next(
                sql
                for sql in GRAPHITI_ACCOUNTED_ZERO_MIGRATION_STATEMENTS
                if sql.startswith("CREATE TRIGGER graphiti_attempt_chain_guard")
            )
        )
        connection.execute(
            next(
                sql
                for sql in GRAPHITI_ADAPTER_MIGRATION_STATEMENTS
                if sql.startswith("CREATE TRIGGER graphiti_attempt_head_update_guard")
            )
        )
        guard = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE name='immutable_authority_migrations_delete'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
        connection.execute("DELETE FROM authority_migrations WHERE version=39")
        connection.execute(guard)
        connection.execute("PRAGMA user_version=38")
    except Exception:
        connection.execute("ROLLBACK TO SAVEPOINT checked_graphiti_guard_downgrade")
        connection.execute("RELEASE SAVEPOINT checked_graphiti_guard_downgrade")
        raise
    connection.execute("RELEASE SAVEPOINT checked_graphiti_guard_downgrade")


def _drop_v40_relationship_open_index(connection: sqlite3.Connection) -> None:
    """Remove the exact v40 partial index from an older-schema fixture."""

    _drop_v41_command_bound_storage(connection)
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) != 40:
        return
    from newsroom.authority.relationship_open_index_migrations import (
        RELATIONSHIP_OPEN_INDEX_MIGRATION_CHECKSUM,
        RELATIONSHIP_OPEN_INDEX_MIGRATION_NAME,
        RELATIONSHIP_OPEN_INDEX_MIGRATION_STATEMENTS,
    )

    if connection.execute(
        "SELECT name,checksum FROM authority_migrations WHERE version=40"
    ).fetchone() != (
        RELATIONSHIP_OPEN_INDEX_MIGRATION_NAME,
        RELATIONSHIP_OPEN_INDEX_MIGRATION_CHECKSUM,
    ) or connection.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type='index' AND name='idx_ledger_events_relationship_event_type'"
    ).fetchone() != (RELATIONSHIP_OPEN_INDEX_MIGRATION_STATEMENTS[0],):
        raise sqlite3.DatabaseError("downgrade requires exact v40 relationship index")
    connection.execute("SAVEPOINT checked_relationship_index_downgrade")
    try:
        guard = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE name='immutable_authority_migrations_delete'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
        connection.execute("DROP INDEX idx_ledger_events_relationship_event_type")
        connection.execute("DELETE FROM authority_migrations WHERE version=40")
        connection.execute(guard)
        connection.execute("PRAGMA user_version=39")
    except Exception:
        connection.execute("ROLLBACK TO SAVEPOINT checked_relationship_index_downgrade")
        connection.execute("RELEASE SAVEPOINT checked_relationship_index_downgrade")
        raise
    connection.execute("RELEASE SAVEPOINT checked_relationship_index_downgrade")


def _drop_v36_shared_scope_schema(connection: sqlite3.Connection) -> None:
    """Losslessly restore the v35 decision representation for downgrade fixtures."""
    _drop_v37_security_record_schema(connection)
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) != 36:
        return
    from newsroom.authority import canonical_json_bytes
    from newsroom.authority.authorisation_scope_content_migrations import (
        AUTHORISATION_SCOPE_CONTENT_MIGRATION_CHECKSUM,
        AUTHORISATION_SCOPE_CONTENT_MIGRATION_NAME,
    )
    if connection.execute(
        "SELECT name,checksum FROM authority_migrations WHERE version=36"
    ).fetchone() != (
        AUTHORISATION_SCOPE_CONTENT_MIGRATION_NAME,
        AUTHORISATION_SCOPE_CONTENT_MIGRATION_CHECKSUM,
    ):
        raise sqlite3.DatabaseError("downgrade requires exact v36 scope authority")
    cursor = connection.execute(
        "SELECT d.*,s.canonical_bytes AS scopes_bytes FROM authorization_decisions d "
        "JOIN authorization_scope_contents s USING(scope_content_digest)"
    )
    names = tuple(item[0] for item in cursor.description)
    rows = tuple(dict(zip(names, row, strict=True)) for row in cursor)
    # An enclosing checked downgrade owns a savepoint. SQLite ignores
    # foreign_keys=OFF inside it; defer the retained NO ACTION references
    # until the same decision rows have been restored instead.
    if connection.in_transaction:
        connection.execute("PRAGMA defer_foreign_keys=ON")
    else:
        connection.execute("PRAGMA foreign_keys=OFF")
    connection.execute("DROP TRIGGER immutable_authorization_decisions_update")
    connection.execute("DROP TRIGGER immutable_authorization_decisions_delete")
    connection.execute("DROP TABLE authorization_decisions")
    from newsroom.authority.migrations import MIGRATION_STATEMENTS
    connection.execute(next(statement for statement in MIGRATION_STATEMENTS
                            if statement.startswith("CREATE TABLE authorization_decisions(")))
    for row in rows:
        scopes = json.loads(bytes(row["scopes_bytes"]))
        value = {
            "authorization_decision_id": row["authorization_decision_id"],
            "authentication_context_id": row["authentication_context_id"],
            "authorization_request_digest": row["authorization_request_digest"],
            "authorization_policy_version": row["authorization_policy_version"],
            "effective_scopes": scopes,
            "effective_scope_digest": row["effective_scope_digest"],
            "allowed": bool(row["allowed"]), "reason_code": row["reason_code"],
            "decided_at": row["decided_at"],
        }
        connection.execute(
            "INSERT INTO authorization_decisions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (row["authorization_decision_id"], row["authentication_context_id"],
             row["authorization_request_digest"], row["authorization_policy_version"],
             row["scopes_bytes"], row["effective_scope_digest"], row["allowed"],
             row["reason_code"], row["decided_at"], canonical_json_bytes(value),
             row["canonical_digest"]),
        )
    connection.execute("DROP TRIGGER immutable_authorization_scope_contents_update")
    connection.execute("DROP TRIGGER immutable_authorization_scope_contents_delete")
    connection.execute("DROP TABLE authorization_scope_contents")
    for prefix in (
        "CREATE INDEX idx_authorization_decisions_context",
        "CREATE TRIGGER immutable_authorization_decisions_update",
        "CREATE TRIGGER immutable_authorization_decisions_delete",
    ):
        connection.execute(next(statement for statement in MIGRATION_STATEMENTS
                                if statement.startswith(prefix)))
    guard = connection.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE name='immutable_authority_migrations_delete'"
    ).fetchone()[0]
    connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
    connection.execute("DELETE FROM authority_migrations WHERE version=36")
    connection.execute(guard)
    connection.execute("PRAGMA user_version=35")
    if connection.execute(
        'PRAGMA foreign_key_check("authorization_decisions")'
    ).fetchone() is not None:
        raise sqlite3.IntegrityError("restored v35 fixture references differ")


def _drop_empty_v35_accounted_zero_schema(connection: sqlite3.Connection) -> None:
    """Restore the exact v34 Graphiti attempt CHECK."""

    _drop_v36_shared_scope_schema(connection)
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) != 35:
        return
    from newsroom.authority.graphiti_accounted_zero_migrations import (
        GRAPHITI_ACCOUNTED_ZERO_MIGRATION_CHECKSUM,
        GRAPHITI_ACCOUNTED_ZERO_MIGRATION_NAME,
    )
    from newsroom.authority.graphiti_adapter_migrations import (
        GRAPHITI_ADAPTER_MIGRATION_STATEMENTS,
    )

    if connection.execute(
        "SELECT name,checksum FROM authority_migrations WHERE version=35"
    ).fetchone() != (
        GRAPHITI_ACCOUNTED_ZERO_MIGRATION_NAME,
        GRAPHITI_ACCOUNTED_ZERO_MIGRATION_CHECKSUM,
    ):
        raise sqlite3.DatabaseError("downgrade requires exact v35 Graphiti authority")
    if connection.execute(
        "SELECT COUNT(*) FROM graphiti_adapter_attempts"
    ).fetchone() != (0,):
        raise sqlite3.DatabaseError("v35 Graphiti adapter attempts must be empty")

    connection.execute("PRAGMA foreign_keys=OFF")
    for trigger in (
        "graphiti_attempt_chain_guard",
        "graphiti_attempt_lineage_guard",
        "graphiti_attempt_output_guard",
        "immutable_graphiti_adapter_attempts_update",
        "immutable_graphiti_adapter_attempts_delete",
    ):
        connection.execute(f'DROP TRIGGER "{trigger}"')
    connection.execute("DROP INDEX idx_graphiti_attempts_run")
    connection.execute("DROP TABLE graphiti_adapter_attempts")

    def adapter_statement(prefix: str) -> str:
        return next(
            item
            for item in GRAPHITI_ADAPTER_MIGRATION_STATEMENTS
            if " ".join(item.split()).startswith(prefix)
        )

    connection.execute(adapter_statement("CREATE TABLE graphiti_adapter_attempts("))
    connection.execute(adapter_statement("CREATE INDEX idx_graphiti_attempts_run "))
    for prefix in (
        "CREATE TRIGGER graphiti_attempt_chain_guard",
        "CREATE TRIGGER graphiti_attempt_lineage_guard",
        "CREATE TRIGGER graphiti_attempt_output_guard",
        "CREATE TRIGGER immutable_graphiti_adapter_attempts_update ",
        "CREATE TRIGGER immutable_graphiti_adapter_attempts_delete ",
    ):
        connection.execute(adapter_statement(prefix))
    guard = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' "
        "AND name='immutable_authority_migrations_delete'"
    ).fetchone()[0]
    connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
    connection.execute("DELETE FROM authority_migrations WHERE version=35")
    connection.execute(guard)
    connection.execute("PRAGMA user_version=34")


def _drop_empty_v34_graphiti_evaluation_schema(
    connection: sqlite3.Connection,
) -> None:
    """Restore the exact v32 central schema; v33 belongs to an isolated store."""

    _drop_empty_v35_accounted_zero_schema(connection)
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) != 34:
        return
    from newsroom.authority.extraction_migrations import (
        EXTRACTION_AUTHORITY_MIGRATION_STATEMENTS,
    )
    from newsroom.authority.graphiti_adapter_migrations import (
        GRAPHITI_ADAPTER_MIGRATION_STATEMENTS,
    )
    from newsroom.authority.graphiti_evaluation_migrations import (
        GRAPHITI_EVALUATION_MIGRATION_CHECKSUM,
        GRAPHITI_EVALUATION_MIGRATION_NAME,
    )

    if connection.execute(
        "SELECT name,checksum FROM authority_migrations WHERE version=34"
    ).fetchone() != (
        GRAPHITI_EVALUATION_MIGRATION_NAME,
        GRAPHITI_EVALUATION_MIGRATION_CHECKSUM,
    ):
        raise sqlite3.DatabaseError("downgrade requires exact v34 Graphiti authority")
    if connection.execute(
        "SELECT COUNT(*) FROM graphiti_attempt_receipts"
    ).fetchone() != (0,):
        raise sqlite3.DatabaseError("v34 Graphiti attempt receipts must be empty")
    if connection.execute(
        "SELECT COUNT(*) FROM graphiti_adapter_configurations "
        "WHERE runtime_mode='REAL_GRAPHITI'"
    ).fetchone() != (0,):
        raise sqlite3.DatabaseError("v34 real Graphiti configuration must be empty")
    if connection.execute(
        "SELECT COUNT(*) FROM extractor_contracts "
        "WHERE producer_kind='GRAPHITI_EVALUATION'"
    ).fetchone() != (0,):
        raise sqlite3.DatabaseError("v34 Graphiti extraction contracts must be empty")

    connection.execute("PRAGMA foreign_keys=OFF")
    for trigger in (
        "immutable_graphiti_attempt_receipt",
        "retained_graphiti_attempt_receipt",
        "immutable_graphiti_workspace_policies_update",
        "immutable_graphiti_workspace_policies_delete",
        "immutable_extractor_contract_update",
        "immutable_extractor_contract_delete",
        "extraction_run_version_chain_guard",
        "extraction_run_version_predecessor_guard",
        "immutable_extraction_run_version_update",
        "immutable_extraction_run_version_delete",
        "extraction_run_version_create_head",
        "extraction_run_version_advance_head",
    ):
        connection.execute(f'DROP TRIGGER "{trigger}"')
    connection.execute("DROP TABLE graphiti_attempt_receipts")
    connection.execute(
        "DELETE FROM graphiti_workspace_policies "
        "WHERE policy_id='00000000-0000-4000-8000-000000004803'"
    )

    def statement(prefix: str) -> str:
        return next(
            item
            for item in EXTRACTION_AUTHORITY_MIGRATION_STATEMENTS
            if " ".join(item.split()).startswith(prefix)
        )

    connection.execute("DROP INDEX idx_extraction_versions_run")
    connection.execute(
        "CREATE TABLE extractor_contracts_v32 AS SELECT * FROM extractor_contracts"
    )
    connection.execute("DROP TABLE extractor_contracts")
    connection.execute(statement("CREATE TABLE extractor_contracts("))
    connection.execute("INSERT INTO extractor_contracts SELECT * FROM extractor_contracts_v32")
    connection.execute("DROP TABLE extractor_contracts_v32")
    connection.execute(
        "CREATE TABLE extraction_run_versions_v32 AS SELECT * FROM extraction_run_versions"
    )
    connection.execute("DROP TABLE extraction_run_versions")
    connection.execute(statement("CREATE TABLE extraction_run_versions("))
    connection.execute(
        "INSERT INTO extraction_run_versions SELECT * FROM extraction_run_versions_v32"
    )
    connection.execute("DROP TABLE extraction_run_versions_v32")
    for prefix in (
        "CREATE INDEX idx_extraction_versions_run ",
        "CREATE TRIGGER immutable_extractor_contract_update ",
        "CREATE TRIGGER immutable_extractor_contract_delete ",
        "CREATE TRIGGER extraction_run_version_chain_guard ",
        "CREATE TRIGGER extraction_run_version_predecessor_guard ",
        "CREATE TRIGGER immutable_extraction_run_version_update ",
        "CREATE TRIGGER immutable_extraction_run_version_delete ",
        "CREATE TRIGGER extraction_run_version_create_head ",
        "CREATE TRIGGER extraction_run_version_advance_head ",
    ):
        connection.execute(statement(prefix))
    for prefix in (
        "CREATE TRIGGER immutable_graphiti_workspace_policies_update ",
        "CREATE TRIGGER immutable_graphiti_workspace_policies_delete ",
    ):
        connection.execute(
            next(
                item
                for item in GRAPHITI_ADAPTER_MIGRATION_STATEMENTS
                if " ".join(item.split()).startswith(prefix)
            )
        )
    guard = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' "
        "AND name='immutable_authority_migrations_delete'"
    ).fetchone()[0]
    connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
    connection.execute("DELETE FROM authority_migrations WHERE version=34")
    connection.execute(guard)
    connection.execute("PRAGMA user_version=32")


def drop_empty_v28_coverage_schema(connection: sqlite3.Connection) -> None:
    """Remove the exact empty v28 Coverage Audit suffix atomically."""
    savepoint = "checked_coverage_audit_downgrade"
    connection.execute(f"SAVEPOINT {savepoint}")
    try:
        _drop_empty_v28_coverage_schema(connection)
    except Exception:
        connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise
    connection.execute(f"RELEASE SAVEPOINT {savepoint}")


def _drop_empty_v28_coverage_schema(connection: sqlite3.Connection) -> None:
    _drop_empty_v29_local_watch_schema(connection)
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) == 28:
        from newsroom.authority.coverage_audit_migrations import (
            COVERAGE_AUDIT_MIGRATION_CHECKSUM,
            COVERAGE_AUDIT_MIGRATION_NAME,
            COVERAGE_AUDIT_MIGRATION_STATEMENTS,
        )

        def normalise_coverage_sql(value: str) -> str:
            return " ".join(value.split()).replace(" IF NOT EXISTS", "")

        coverage_tables = (
            "coverage_audits",
            "coverage_audit_observations",
            "coverage_gaps",
            "coverage_gap_decisions",
        )
        placeholders = ",".join("?" for _ in coverage_tables)
        objects = connection.execute(
            f"SELECT type,name,sql FROM sqlite_master WHERE tbl_name IN ({placeholders}) "
            "AND type IN ('table','trigger','index') AND sql IS NOT NULL",
            coverage_tables,
        ).fetchall()
        if connection.execute(
            "SELECT name,checksum FROM authority_migrations WHERE version=28"
        ).fetchone() != (
            COVERAGE_AUDIT_MIGRATION_NAME,
            COVERAGE_AUDIT_MIGRATION_CHECKSUM,
        ) or {normalise_coverage_sql(str(row[2])) for row in objects} != {
            normalise_coverage_sql(statement)
            for statement in COVERAGE_AUDIT_MIGRATION_STATEMENTS
        }:
            raise sqlite3.DatabaseError(
                "downgrade requires exact empty v28 Coverage Audit schema"
            )
        for table in coverage_tables:
            if connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone() != (0,):
                raise sqlite3.DatabaseError("v28 Coverage Audit tables must be empty")
        guard = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' "
            "AND name='immutable_authority_migrations_delete'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
        for object_type, name, _ in objects:
            if object_type == "trigger":
                connection.execute(f'DROP TRIGGER "{name}"')
        for table in reversed(coverage_tables):
            connection.execute(f'DROP TABLE "{table}"')
        connection.execute("DELETE FROM authority_migrations WHERE version=28")
        connection.execute(guard)
        connection.execute("PRAGMA user_version=27")


def drop_empty_v23_lineage_schema(connection: sqlite3.Connection) -> None:
    """Remove exact empty v24/v23 schemas as one rollback-safe operation."""
    savepoint = "checked_candidate_lineage_downgrade"
    connection.execute(f"SAVEPOINT {savepoint}")
    try:
        _drop_empty_v23_lineage_schema(connection)
    except Exception:
        connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise
    connection.execute(f"RELEASE SAVEPOINT {savepoint}")


def _drop_empty_v23_lineage_schema(connection: sqlite3.Connection) -> None:
    """Remove an exact, empty v23 lineage schema atomically."""
    _drop_empty_v28_coverage_schema(connection)
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) == 27:
        from newsroom.authority.bounded_search_migrations import (
            BOUNDED_SEARCH_MIGRATION_CHECKSUM,
            BOUNDED_SEARCH_MIGRATION_NAME,
            BOUNDED_SEARCH_MIGRATION_STATEMENTS,
        )

        def normalise_search_sql(value: str) -> str:
            return " ".join(value.split()).replace(" IF NOT EXISTS", "")

        search_tables = (
            "search_purposes",
            "search_requests",
            "search_attempts",
            "search_outcomes",
            "search_result_references",
            "search_review_decisions",
            "search_budget_ledger",
        )
        placeholders = ",".join("?" for _ in search_tables)
        objects = connection.execute(
            f"SELECT type,name,sql FROM sqlite_master WHERE tbl_name IN ({placeholders}) "
            "AND type IN ('table','trigger','index') AND sql IS NOT NULL",
            search_tables,
        ).fetchall()
        if connection.execute(
            "SELECT name,checksum FROM authority_migrations WHERE version=27"
        ).fetchone() != (
            BOUNDED_SEARCH_MIGRATION_NAME,
            BOUNDED_SEARCH_MIGRATION_CHECKSUM,
        ) or {normalise_search_sql(str(row[2])) for row in objects} != {
            normalise_search_sql(statement)
            for statement in BOUNDED_SEARCH_MIGRATION_STATEMENTS
        }:
            raise sqlite3.DatabaseError(
                "downgrade requires exact empty v27 bounded Search schema"
            )
        for table in search_tables:
            if connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone() != (0,):
                raise sqlite3.DatabaseError("v27 bounded Search tables must be empty")
        guard = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' "
            "AND name='immutable_authority_migrations_delete'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
        for object_type, name, _ in objects:
            if object_type == "trigger":
                connection.execute(f'DROP TRIGGER "{name}"')
        for table in reversed(search_tables):
            connection.execute(f'DROP TABLE "{table}"')
        connection.execute("DELETE FROM authority_migrations WHERE version=27")
        connection.execute(guard)
        connection.execute("PRAGMA user_version=26")
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) == 26:
        from newsroom.authority.planned_agenda_migrations import (
            PLANNED_AGENDA_MIGRATION_CHECKSUM,
            PLANNED_AGENDA_MIGRATION_NAME,
            PLANNED_AGENDA_MIGRATION_STATEMENTS,
        )

        def normalise_agenda_sql(value: str) -> str:
            return " ".join(value.split()).replace(" IF NOT EXISTS", "")

        agenda_tables = (
            "planned_agenda_items",
            "planned_agenda_versions",
            "planned_agenda_heads",
            "planned_agenda_resolutions",
        )
        objects = connection.execute(
            "SELECT type,name,sql FROM sqlite_master WHERE tbl_name IN (?,?,?,?) "
            "AND type IN ('table','trigger')",
            agenda_tables,
        ).fetchall()
        if connection.execute(
            "SELECT name,checksum FROM authority_migrations WHERE version=26"
        ).fetchone() != (
            PLANNED_AGENDA_MIGRATION_NAME,
            PLANNED_AGENDA_MIGRATION_CHECKSUM,
        ) or {normalise_agenda_sql(str(row[2])) for row in objects} != {
            normalise_agenda_sql(statement)
            for statement in PLANNED_AGENDA_MIGRATION_STATEMENTS
        }:
            raise sqlite3.DatabaseError(
                "downgrade requires exact empty v26 Planned Agenda schema"
            )
        for table in agenda_tables:
            if connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone() != (0,):
                raise sqlite3.DatabaseError("v26 Planned Agenda tables must be empty")
        guard = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' "
            "AND name='immutable_authority_migrations_delete'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
        for object_type, name, _ in objects:
            if object_type == "trigger":
                connection.execute(f'DROP TRIGGER "{name}"')
        for table in reversed(agenda_tables):
            connection.execute(f'DROP TABLE "{table}"')
        connection.execute("DELETE FROM authority_migrations WHERE version=26")
        connection.execute(guard)
        connection.execute("PRAGMA user_version=25")
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) == 25:
        from newsroom.authority.evaluation_feedback_migrations import (
            EVALUATION_FEEDBACK_MIGRATION_CHECKSUM,
            EVALUATION_FEEDBACK_MIGRATION_NAME,
            EVALUATION_FEEDBACK_MIGRATION_STATEMENTS,
        )

        def normalise_sql(value: str) -> str:
            return " ".join(value.split()).replace(" IF NOT EXISTS", "")

        objects = connection.execute(
            "SELECT type,name,sql FROM sqlite_master WHERE "
            "tbl_name IN ('evaluation_feedback','evaluation_reconciliation_obligations',"
            "'evaluation_reconciliation_dispositions') AND type IN ('table','trigger')"
        ).fetchall()
        expected_names = {
            "evaluation_feedback",
            "evaluation_reconciliation_obligations",
            "evaluation_reconciliation_dispositions",
            "immutable_evaluation_feedback",
            "retained_evaluation_feedback",
            "immutable_evaluation_obligation",
            "retained_evaluation_obligation",
            "immutable_evaluation_disposition",
            "retained_evaluation_disposition",
            "evaluation_disposition_predecessor_guard",
        }
        if (
            connection.execute(
                "SELECT name,checksum FROM authority_migrations WHERE version=25"
            ).fetchone()
            != (
                EVALUATION_FEEDBACK_MIGRATION_NAME,
                EVALUATION_FEEDBACK_MIGRATION_CHECKSUM,
            )
            or {str(row[1]) for row in objects} != expected_names
            or {normalise_sql(str(row[2])) for row in objects}
            != {
                normalise_sql(statement)
                for statement in EVALUATION_FEEDBACK_MIGRATION_STATEMENTS
            }
        ):
            raise sqlite3.DatabaseError(
                "downgrade requires exact empty v25 Feedback schema"
            )
        for table in (
            "evaluation_feedback",
            "evaluation_reconciliation_obligations",
            "evaluation_reconciliation_dispositions",
        ):
            if connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone() != (0,):
                raise sqlite3.DatabaseError("v25 Feedback tables must be empty")
        guard = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' "
            "AND name='immutable_authority_migrations_delete'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
        for object_type, name, _ in objects:
            if object_type == "trigger":
                connection.execute(f'DROP TRIGGER "{name}"')
        for table in (
            "evaluation_reconciliation_dispositions",
            "evaluation_reconciliation_obligations",
            "evaluation_feedback",
        ):
            connection.execute(f'DROP TABLE "{table}"')
        connection.execute("DELETE FROM authority_migrations WHERE version=25")
        connection.execute(guard)
        connection.execute("PRAGMA user_version=24")
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) == 24:
        candidate_tables = (
            "story_candidate_heads",
            "story_candidate_collision_bindings",
            "story_candidate_admission_receipts_v2",
        )
        candidate_triggers = (
            "candidate_head_update_guard",
            "candidate_head_insert_guard",
            "retained_candidate_head",
            "retained_candidate_collision",
            "immutable_candidate_collision",
            "retained_candidate_receipt",
            "immutable_candidate_receipt",
        )
        history = connection.execute(
            "SELECT name,checksum FROM authority_migrations WHERE version=24"
        ).fetchone()
        from newsroom.authority.story_candidate_migrations import (
            STORY_CANDIDATE_MIGRATION_CHECKSUM,
            STORY_CANDIDATE_MIGRATION_NAME,
            STORY_CANDIDATE_MIGRATION_STATEMENTS,
        )

        required = {("table", name) for name in candidate_tables} | {
            ("trigger", name) for name in candidate_triggers
        }
        present = set(
            connection.execute(
                "SELECT type,name FROM sqlite_master WHERE name IN ("
                + ",".join("?" for _ in required)
                + ")",
                tuple(name for _, name in required),
            ).fetchall()
        )
        actual_sql = {
            " ".join(str(row[0]).split())
            for row in connection.execute(
                "SELECT sql FROM sqlite_master WHERE name IN ("
                + ",".join("?" for _ in required)
                + ")",
                tuple(name for _, name in required),
            )
        }
        expected_sql = {
            " ".join(statement.split())
            for statement in STORY_CANDIDATE_MIGRATION_STATEMENTS
        }
        if (
            history
            != (STORY_CANDIDATE_MIGRATION_NAME, STORY_CANDIDATE_MIGRATION_CHECKSUM)
            or present != required
            or actual_sql != expected_sql
        ):
            raise sqlite3.DatabaseError(
                "downgrade requires exact empty v24 Candidate schema"
            )
        for table in candidate_tables:
            if connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone() != (0,):
                raise sqlite3.DatabaseError("v24 Candidate tables must be empty")
        savepoint_v24 = "checked_empty_v24_candidate_downgrade"
        connection.execute(f"SAVEPOINT {savepoint_v24}")
        try:
            guard = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='trigger' "
                "AND name='immutable_authority_migrations_delete'"
            ).fetchone()[0]
            connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
            for trigger in candidate_triggers:
                connection.execute(f'DROP TRIGGER "{trigger}"')
            for table in candidate_tables:
                connection.execute(f'DROP TABLE "{table}"')
            connection.execute("DELETE FROM authority_migrations WHERE version=24")
            connection.execute(guard)
            connection.execute("PRAGMA user_version=23")
            connection.execute(f"RELEASE SAVEPOINT {savepoint_v24}")
        except Exception:
            connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint_v24}")
            connection.execute(f"RELEASE SAVEPOINT {savepoint_v24}")
            raise
    savepoint = "checked_empty_v23_lineage_downgrade"
    lineage_tables = (
        "event_hypothesis_lineage",
        "event_hypothesis_lineage_heads",
    )
    lineage_triggers = (
        "event_hypothesis_lineage_head_update_guard",
        "event_hypothesis_lineage_head_insert_guard",
        "retained_event_hypothesis_lineage_delete",
        "immutable_event_hypothesis_lineage_update",
        "event_hypothesis_lineage_coherence",
    )
    history_guard = "immutable_authority_migrations_delete"
    connection.execute(f"SAVEPOINT {savepoint}")
    try:
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        maximum_history_version = connection.execute(
            "SELECT MAX(version) FROM authority_migrations"
        ).fetchone()[0]
        if maximum_history_version != user_version:
            raise sqlite3.DatabaseError("v23 schema version/history mismatch")
        if user_version < EVENT_HYPOTHESIS_LINEAGE_SCHEMA_VERSION:
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            return
        if user_version != EVENT_HYPOTHESIS_LINEAGE_SCHEMA_VERSION:
            raise sqlite3.DatabaseError("downgrade requires exact schema v23")
        history = connection.execute(
            "SELECT name,checksum FROM authority_migrations WHERE version=?",
            (EVENT_HYPOTHESIS_LINEAGE_SCHEMA_VERSION,),
        ).fetchone()
        if history != (
            EVENT_HYPOTHESIS_LINEAGE_MIGRATION_NAME,
            EVENT_HYPOTHESIS_LINEAGE_MIGRATION_CHECKSUM,
        ):
            raise sqlite3.DatabaseError(
                "downgrade requires exact v23 migration history"
            )
        required_objects = {("table", table, table) for table in lineage_tables} | {
            (
                "trigger",
                trigger,
                "event_hypothesis_lineage_heads"
                if "head_" in trigger
                else "event_hypothesis_lineage",
            )
            for trigger in lineage_triggers
        }
        required_objects.add(("trigger", history_guard, "authority_migrations"))
        names = tuple(name for _, name, _ in required_objects)
        lineage_names = (*lineage_tables, *lineage_triggers)
        present_objects = set(
            connection.execute(
                "SELECT type,name,tbl_name FROM sqlite_master "
                f"WHERE name IN ({','.join('?' for _ in names)})",
                names,
            ).fetchall()
        )
        if present_objects != required_objects:
            raise sqlite3.DatabaseError("downgrade requires exact v23 lineage schema")
        lineage_sql = {
            " ".join(str(row[0]).split())
            for row in connection.execute(
                f"SELECT sql FROM sqlite_master WHERE name IN ({','.join('?' for _ in lineage_names)})",
                lineage_names,
            )
        }
        expected_lineage_sql = {
            " ".join(statement.split())
            for statement in EVENT_HYPOTHESIS_LINEAGE_MIGRATION_STATEMENTS
        }
        if lineage_sql != expected_lineage_sql:
            raise sqlite3.DatabaseError("downgrade requires exact v23 lineage SQL")
        if any(
            connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone() != (0,)
            for table in lineage_tables
        ):
            raise sqlite3.DatabaseError("v23 lineage tables must be empty")
        guard = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
            (history_guard,),
        ).fetchone()[0]

        connection.execute(f'DROP TRIGGER "{history_guard}"')
        for trigger in lineage_triggers:
            connection.execute(f'DROP TRIGGER "{trigger}"')
        for table in reversed(lineage_tables):
            connection.execute(f'DROP TABLE "{table}"')
        connection.execute(
            "DELETE FROM authority_migrations WHERE version=?",
            (EVENT_HYPOTHESIS_LINEAGE_SCHEMA_VERSION,),
        )
        connection.execute(guard)
        connection.execute(
            f"PRAGMA user_version={EVENT_HYPOTHESIS_LINEAGE_SCHEMA_VERSION - 1}"
        )
    except Exception:
        connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise
    else:
        connection.execute(f"RELEASE SAVEPOINT {savepoint}")


def drop_empty_v22_relationship_schema(connection: sqlite3.Connection) -> None:
    """Remove an exact, empty v22 relationship schema atomically."""

    savepoint = "checked_empty_v22_relationship_downgrade"
    relationship_table = "event_hypothesis_relationship_decisions"
    relationship_triggers = (
        "retained_event_hypothesis_relationship_delete",
        "immutable_event_hypothesis_relationship_update",
        "event_hypothesis_relationship_coherence",
    )
    connection.execute(f"SAVEPOINT {savepoint}")
    try:
        drop_empty_v23_lineage_schema(connection)
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        maximum_history_version = connection.execute(
            "SELECT MAX(version) FROM authority_migrations"
        ).fetchone()[0]
        if maximum_history_version != user_version:
            raise sqlite3.DatabaseError("v22 schema version/history mismatch")
        if user_version < EVENT_HYPOTHESIS_RELATIONSHIP_SCHEMA_VERSION:
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            return
        if user_version != EVENT_HYPOTHESIS_RELATIONSHIP_SCHEMA_VERSION:
            raise sqlite3.DatabaseError("downgrade requires exact schema v22")

        v22_history = connection.execute(
            "SELECT name,checksum FROM authority_migrations WHERE version=?",
            (EVENT_HYPOTHESIS_RELATIONSHIP_SCHEMA_VERSION,),
        ).fetchone()
        if v22_history != (
            EVENT_HYPOTHESIS_RELATIONSHIP_MIGRATION_NAME,
            EVENT_HYPOTHESIS_RELATIONSHIP_MIGRATION_CHECKSUM,
        ):
            raise sqlite3.DatabaseError(
                "downgrade requires exact v22 migration history"
            )

        required_objects = {
            ("table", relationship_table, relationship_table),
            *(
                ("trigger", trigger, relationship_table)
                for trigger in relationship_triggers
            ),
        }
        present_objects = set(
            connection.execute(
                "SELECT type,name,tbl_name FROM sqlite_master "
                f"WHERE name IN ({','.join('?' for _ in required_objects)})",
                tuple(name for _, name, _ in required_objects),
            ).fetchall()
        )
        if present_objects != required_objects:
            raise sqlite3.DatabaseError(
                "downgrade requires exact v22 relationship schema"
            )
        if connection.execute(
            f'SELECT COUNT(*) FROM "{relationship_table}"'
        ).fetchone() != (0,):
            raise sqlite3.DatabaseError("v22 relationship table must be empty")

        for trigger in relationship_triggers:
            connection.execute(f'DROP TRIGGER "{trigger}"')
        connection.execute(f'DROP TABLE "{relationship_table}"')
    except Exception:
        connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise
    else:
        connection.execute(f"RELEASE SAVEPOINT {savepoint}")


def downgrade_empty_graphiti_adapter_schema_to_v15(database: Path) -> None:
    """Remove only the empty v16 Graphiti-adapter schema from a checked test DB.

    This helper is test-only. It first removes the empty additive v17 successor,
    preserves all v1-v15 authority rows, and restores the migration-history
    delete guard after removing the v16+ history records.
    """

    conn = sqlite3.connect(database, isolation_level=None)
    try:
        current = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if current < GRAPHITI_ADAPTER_SCHEMA_VERSION:
            return
        drop_empty_v22_relationship_schema(conn)
        conn.execute("PRAGMA foreign_keys=OFF")
        delete_trigger = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' "
            "AND name='immutable_authority_migrations_delete'"
        ).fetchone()
        assert delete_trigger is not None and delete_trigger[0]
        conn.execute("DROP TRIGGER immutable_authority_migrations_delete")

        if current >= EVENT_HYPOTHESIS_SCHEMA_VERSION:
            for table in (
                "event_hypothesis_heads_v2",
                "event_hypothesis_versions_v2",
                "event_hypotheses_v2",
            ):
                conn.execute(f'DROP TABLE "{table}"')

        if current >= TRIAGE_EXECUTION_SCHEMA_VERSION:
            for table in (
                "triage_work_item_leases",
                "triage_worker_attempts",
                "triage_execution_batches",
            ):
                conn.execute(f'DROP TABLE "{table}"')

        if current >= TRIAGE_DISPOSITION_SCHEMA_VERSION:
            for table in (
                "triage_proposal_dispositions",
                "triage_proposal_validation_findings",
            ):
                conn.execute(f'DROP TABLE "{table}"')

        if current >= TRIAGE_WORK_ITEM_SCHEMA_VERSION:
            for table in (
                "triage_work_item_heads",
                "triage_work_item_versions",
                "triage_work_items",
            ):
                conn.execute(f'DROP TABLE "{table}"')

        if current >= EVALUATION_HANDOFF_SCHEMA_VERSION:
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND (name LIKE '%evaluation_handoff%' "
                "OR tbl_name LIKE 'evaluation_handoff%') ORDER BY name DESC"
            ).fetchall():
                conn.execute(f'DROP TRIGGER "{row[0]}"')
            for table in (
                "evaluation_handoff_acknowledgements",
                "evaluation_handoff_attempts",
                "evaluation_handoffs",
            ):
                conn.execute(f'DROP TABLE "{table}"')

        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' "
            "AND name LIKE 'graphiti_%' ORDER BY name DESC"
        ).fetchall():
            conn.execute(f'DROP VIEW "{row[0]}"')
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' "
            "AND (name LIKE 'graphiti_%' OR name LIKE 'immutable_graphiti_%' OR tbl_name LIKE 'graphiti_%') "
            "ORDER BY name DESC"
        ).fetchall():
            conn.execute(f'DROP TRIGGER "{row[0]}"')
        for table in (
            "graphiti_adapter_attempt_replays",
            "graphiti_replay_sources",
            "graphiti_adapter_attempt_heads",
            "graphiti_adapter_attempts",
            "graphiti_cleanup_receipts",
            "graphiti_input_manifest_passages",
            "graphiti_input_manifests",
            "graphiti_workspace_lifecycle_events",
            "graphiti_workspaces",
            "graphiti_adapter_configurations",
            "graphiti_workspace_policies",
        ):
            conn.execute(f'DROP TABLE "{table}"')

        conn.execute(
            "DELETE FROM authority_migrations WHERE version>=?",
            (GRAPHITI_ADAPTER_SCHEMA_VERSION,),
        )
        conn.execute(str(delete_trigger[0]))
        conn.execute("PRAGMA user_version=15")
    finally:
        conn.close()


def _drop_v41_command_bound_storage(connection: sqlite3.Connection) -> None:
    """Restore exact v40 bytes/DDL for explicit historical-codec fixtures."""
    _drop_v42_projection_retirement(connection)
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) != 41:
        return
    from newsroom.authority.command_bound_storage import command_request_bytes, command_result_bytes
    from newsroom.authority.authorization_request_storage_migrations import AUTHORIZATION_REQUEST_INDEXED_FIELDS
    from newsroom.authority.canonical import canonical_json_bytes
    from newsroom.authority.command_bound_storage_migrations import (
        COMMAND_BOUND_STORAGE_MIGRATION_NAME, COMMAND_BOUND_STORAGE_MIGRATION_CHECKSUM,
    )
    import json
    if tuple(connection.execute("SELECT name,checksum FROM authority_migrations WHERE version=41").fetchone()) != (
        COMMAND_BOUND_STORAGE_MIGRATION_NAME, COMMAND_BOUND_STORAGE_MIGRATION_CHECKSUM,
    ):
        raise sqlite3.DatabaseError("downgrade requires exact v41 storage authority")
    connection.execute("SAVEPOINT checked_command_storage_downgrade")
    try:
        guards = tuple(connection.execute("SELECT sql FROM sqlite_schema WHERE name IN "
            "('immutable_authorization_requests_update','immutable_authority_commands_update','immutable_authority_migrations_delete') ORDER BY name"))
        for name in ('immutable_authorization_requests_update','immutable_authority_commands_update','immutable_authority_migrations_delete'):
            connection.execute(f"DROP TRIGGER {name}")
        for table in ('authorization_requests','authority_commands'):
            cursor = connection.execute(f"SELECT rowid,* FROM {table}")
            names = tuple(item[0] for item in cursor.description)
            for raw in cursor:
                row = dict(zip(names, raw, strict=True))
                if table == 'authorization_requests':
                    original = json.loads(command_request_bytes(connection, row))
                    residual = canonical_json_bytes({key: value for key, value in original.items() if key not in AUTHORIZATION_REQUEST_INDEXED_FIELDS})
                    connection.execute("UPDATE authorization_requests SET storage_request_residual=?,storage_request_marker=? WHERE rowid=?", (residual,b'v38',row['rowid']))
                else:
                    original = command_result_bytes(connection, bytes(row['result_bytes']), str(row['result_digest']))
                    connection.execute("UPDATE authority_commands SET result_bytes=? WHERE rowid=?", (original,row['rowid']))
        connection.execute("DROP TRIGGER authorization_request_storage_guard")
        connection.execute("ALTER TABLE authorization_requests DROP COLUMN storage_request_marker")
        connection.execute("ALTER TABLE authorization_requests ADD COLUMN storage_request_marker "
            "BLOB NOT NULL DEFAULT X'763338' CHECK(storage_request_marker=X'763338')")
        connection.execute("""CREATE TRIGGER authorization_request_storage_guard
        BEFORE INSERT ON authorization_requests
        WHEN NEW.storage_request_marker != X'763338'
        BEGIN SELECT RAISE(ABORT,'authorization request storage differs'); END""")
        connection.execute("DELETE FROM authority_migrations WHERE version=41")
        for (guard,) in guards:
            connection.execute(guard)
        connection.execute("PRAGMA user_version=40")
    except Exception:
        connection.execute("ROLLBACK TO SAVEPOINT checked_command_storage_downgrade")
        connection.execute("RELEASE SAVEPOINT checked_command_storage_downgrade")
        raise
    connection.execute("RELEASE SAVEPOINT checked_command_storage_downgrade")


def _drop_v42_projection_retirement(connection: sqlite3.Connection) -> None:
    """Remove schema-only v42 from full-history migration fixtures, never live data."""
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) != 42:
        return
    from newsroom.authority.migrations import MIGRATION_STATEMENTS, EXPECTED_MIGRATION_HISTORY, EXPECTED_SCHEMA_FINGERPRINT, schema_fingerprint
    from newsroom.authority.projection_retirement_migrations import PROJECTION_RETIREMENT_PREDECESSOR_FINGERPRINT

    if (schema_fingerprint(connection) != EXPECTED_SCHEMA_FINGERPRINT
            or tuple(tuple(row) for row in connection.execute("SELECT version,name,checksum FROM authority_migrations ORDER BY version")) != EXPECTED_MIGRATION_HISTORY
            or connection.execute("SELECT 1 FROM ledger_events WHERE retired_header_digest IS NOT NULL LIMIT 1").fetchone() is not None):
        raise sqlite3.DatabaseError("historical fixture requires exact unexpired v42 authority")
    connection.execute("SAVEPOINT checked_projection_retirement_downgrade")
    try:
        connection.execute("PRAGMA defer_foreign_keys=ON")
        indices = tuple(row[0] for row in connection.execute("SELECT sql FROM sqlite_schema WHERE type='index' AND tbl_name='ledger_events' AND sql IS NOT NULL AND name<>'idx_retired_command_key'"))
        sequence = connection.execute("SELECT seq FROM sqlite_sequence WHERE name='ledger_events'").fetchone()
        connection.execute("CREATE TEMP TABLE predecessor_ledger AS SELECT * FROM ledger_events")
        connection.execute("DROP TABLE ledger_events")
        connection.execute(next(sql for sql in MIGRATION_STATEMENTS if sql.startswith("CREATE TABLE ledger_events(")))
        names = ",".join(row[1] for row in connection.execute("PRAGMA table_info(ledger_events)"))
        connection.execute(f"INSERT INTO ledger_events({names}) SELECT {names} FROM predecessor_ledger")
        connection.execute("DROP TABLE predecessor_ledger")
        if sequence is not None:
            connection.execute("UPDATE sqlite_sequence SET seq=? WHERE name='ledger_events'", (sequence[0],))
        for sql in MIGRATION_STATEMENTS:
            if sql.startswith("CREATE TRIGGER") and " ON ledger_events" in sql:
                connection.execute(sql)
        for sql in indices:
            connection.execute(sql)
        connection.execute("DROP TRIGGER authority_commands_retired_key_guard")
        connection.execute("DROP TRIGGER projection_diagnostic_expiry_insert_guard")
        connection.execute("DROP TRIGGER projection_diagnostic_expiry_update_guard")
        connection.execute("ALTER TABLE projection_generations DROP COLUMN diagnostic_history_expired")
        guard = connection.execute("SELECT sql FROM sqlite_schema WHERE name='immutable_authority_migrations_delete'").fetchone()[0]
        connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
        connection.execute("DELETE FROM authority_migrations WHERE version=42")
        connection.execute(guard)
        connection.execute("PRAGMA user_version=41")
        if schema_fingerprint(connection) != PROJECTION_RETIREMENT_PREDECESSOR_FINGERPRINT or connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise sqlite3.DatabaseError("v42 fixture downgrade lost exact v41 authority")
    except Exception:
        connection.execute("ROLLBACK TO SAVEPOINT checked_projection_retirement_downgrade")
        connection.execute("RELEASE SAVEPOINT checked_projection_retirement_downgrade")
        raise
    connection.execute("RELEASE SAVEPOINT checked_projection_retirement_downgrade")
