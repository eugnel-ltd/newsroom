"""Canonical retained-schema fixtures for authority migration compatibility tests.

This module deliberately builds old schemas forwards from the production SQL
registry.  It does not infer predecessors by removing objects from a newer
schema and it never addresses migration history relative to the list tail.
"""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from newsroom.authority import migrations as authority_migrations
from newsroom.authority.canonical import digest_canonical

RETAINED_MIN_VERSION = 13
CURRENT_VERSION = authority_migrations.SCHEMA_VERSION
CENTRAL_SCHEMA_VERSIONS = tuple(
    int(record.version) for record in authority_migrations.MIGRATIONS
)
PREDECESSOR_VERSION = CENTRAL_SCHEMA_VERSIONS[-2]
NEWER_VERSION = CURRENT_VERSION + 1
RETAINED_VERSIONS = tuple(
    version for version in CENTRAL_SCHEMA_VERSIONS if version >= RETAINED_MIN_VERSION
)
UPGRADE_PREDECESSOR_VERSIONS = RETAINED_VERSIONS[:-1]
BACKUP_PREDECESSOR_VERSIONS = tuple(
    version
    for version in UPGRADE_PREDECESSOR_VERSIONS
    if (
        authority_migrations.GRAPHITI_ADAPTER_SCHEMA_VERSION
        <= version
        <= authority_migrations.GRAPHITI_EVALUATION_SCHEMA_VERSION
    )
)
FIXTURE_APPLIED_AT = "1970-01-01T00:00:00.000000Z"


class MigrationCompatibilityError(AssertionError):
    """The migration registry or a retained SQLite prefix is not exact."""


class MigrationLike(Protocol):
    version: int
    name: str
    checksum: str


StatementExecutor = Callable[[sqlite3.Connection, int, int, str], None]
ObjectRow = tuple[str, str, str, str | None]
HistoryRow = tuple[int, str, str]

# Independent release pins. A checked migration must append its literal record.
PINNED_MIGRATION_HISTORY: tuple[HistoryRow, ...] = (
    (
        1,
        "authority_event_foundation_v1",
        "sha256:2b50772c37bb426ccac46f84efc254cdb0e8124103b9af0d44dffbccf84b14ee",
    ),
    (
        2,
        "governed_object_authority_v2",
        "sha256:e64a85069c061c03d85889524c92f9707f23a8abe417c31875fbbbadcad465b1",
    ),
    (
        3,
        "projection_authority_v3",
        "sha256:ddcc7020aff2e1ec449bab190b1e22767e6c4c78db27f24ca3ee3638c548b6af",
    ),
    (
        4,
        "projection_generation_promotion_v4",
        "sha256:eddebadf6bfb4a315d0be9525e7af245f75fea6043a88c35b84356d4bd7b8194",
    ),
    (
        5,
        "integrated_foundation_proof_v5",
        "sha256:dd3c0f05beadbcdb0dc98c06549120825a793b6e6083454b9604d8d10c336a35",
    ),
    (
        6,
        "governed_relation_authority_v6",
        "sha256:5417391a9d2d2da2fa2ed860ad64164865d4433c319c96e63359d4dff736c36b",
    ),
    (
        7,
        "complete_projection_authority_v7",
        "sha256:fb0bf5a42912a761ba2c228c0da518baedc42c19aea94b347f87990d4d807615",
    ),
    (
        8,
        "hybrid_retrieval_authority_v8",
        "sha256:8c73b3b0a73b4e61b56b4017b514704c1c3517e5bb93dce7d0d16b50d329643f",
    ),
    (
        9,
        "complete_fixture_candidate_authority_v9",
        "sha256:cfb90b0e8432a8edc2c4edddfc71bd0e9b0b25edc712e23835b6d2bdcdec17b1",
    ),
    (
        10,
        "source_registry_authority_v10",
        "sha256:332856936e220c4c39124dffb6132638be22c6da899ccb46f600aa1f21ddbd17",
    ),
    (
        11,
        "check_transition_authority_v11",
        "sha256:8f0ae515f90058c21d985f51543a2a57578091114c4785aee1aeaca8ed2f2a34",
    ),
    (
        12,
        "discovery_signal_lead_authority_v12",
        "sha256:4cba2fc7309516a9344b3a0fa12bb4f5c3506ce9ae57e6677ec2cbf0119e03c4",
    ),
    (
        13,
        "extraction_run_authority_v13",
        "sha256:c3e5ae627dda1c04bebc50952786413d977bd399e67b7f5b87452794f08f49ab",
    ),
    (
        14,
        "entity_resolution_authority_v14",
        "sha256:546e81c2419ecb895a1eea7f9c9556931a3e8ad85efe61e878b2fcc25ad72ee9",
    ),
    (
        15,
        "editorial_relation_authority_v15",
        "sha256:946a697524cd1ce84546208c21948ec29c59df79410c5eafef196c344f2d8587",
    ),
    (
        16,
        "graphiti_proposal_adapter_v16",
        "sha256:ffd44aa70e65e7a2c69a48b3b652160ccc33285d9282c7eed202d206133ba991",
    ),
    (
        17,
        "evaluation_handoff_authority_v17",
        "sha256:c15b3a3fc90833048938b591291d16f59ee1f36b54a6d72dbd04b63877682e7f",
    ),
    (
        18,
        "triage_work_item_authority_v18",
        "sha256:f815499c103fed95fbff0c25528331b2483b7c01687f8742394faa92a538bb88",
    ),
    (
        19,
        "triage_proposal_disposition_authority_v19",
        "sha256:d5f9702d359836e3b564ba1cadbad27e5fc17ba79e5155e2b34382ec30681177",
    ),
    (
        20,
        "triage_execution_authority_v20",
        "sha256:6eb04f981f650bbb4956f148d11f1656bcd2b7c510117db96602dd9d83ab9bd3",
    ),
    (
        21,
        "event_hypothesis_authority_v21",
        "sha256:42009475669a475af8e3e24bbcd02e6fcd9fbb71a800e18d83624e34e79e5e21",
    ),
    (
        22,
        "event_hypothesis_relationship_authority_v22",
        "sha256:e59eb222a95e2901ccaae29ce1b9e8eded797306e9796718a6d2c4fa505a6636",
    ),
    (
        23,
        "event_hypothesis_lineage_authority_v23",
        "sha256:6c24d402f246f4e82a49a9772d70677d922282aae3b6dde93c62c0ef9b1b7a72",
    ),
    (
        24,
        "story_candidate_authority_v24",
        "sha256:1eea25005483de124e0add0100f4805ed5a537852fc70916f17a209c633e0ca0",
    ),
    (
        25,
        "evaluation_feedback_authority_v25",
        "sha256:59fe3bd40a2e22e874b4e5b02448501deffc23597e11b442e35b18e39ead0496",
    ),
    (
        26,
        "planned_agenda_authority_v26",
        "sha256:55e6e8878140714dc6fc6c8149357e1f15e4683fcb7ee0b31b168a737bfd3d4c",
    ),
    (
        27,
        "bounded_search_authority_v27",
        "sha256:ee5679aba6ceb3e95ba925febbbb7853369f93d055563ac85402d380377672b0",
    ),
    (
        28,
        "coverage_audit_authority_v28",
        "sha256:c923daf18aed10bb9c197bfd588d816223d978668bda56c157438d1a4b7cc487",
    ),
    (
        29,
        "event_scoped_local_watch_authority_v29",
        "sha256:ca57c62c9bfadc2ea0a09a3bf762f95854e413aa71d324a296b4c867c90dec7b",
    ),
    (
        30,
        "increment8_evaluation_authority_v30",
        "sha256:764306cbc8fced0b50657c87c2c8735aa07b6ed6b02b1d7ceec84afd9db7dc15",
    ),
    (
        31,
        "increment8_operational_authority_v31",
        "sha256:b3a9535516836d7a0023cc0c030926edd8036b0fd8b31b9647342a9612152342",
    ),
    (
        32,
        "increment8_recovery_authority_v32",
        "sha256:513d983ce8f21f576c08b6a99337f3164025b73e588867d8dde4d500805f79ee",
    ),
    (
        34,
        "graphiti_evaluation_extraction_authority_v34",
        "sha256:ff4d4c7fdb3d9ebe8354002fe0c71739edd549f46f685782821038ab535f350f",
    ),
    (
        35,
        "graphiti_accounted_zero_proposal_authority_v35",
        "sha256:5618c2a4392aabc196687b1fccbb47deee20bcdccc52cc21759cb46e32277829",
    ),
    (
        36,
        "authorisation_shared_scope_content_v36",
        "sha256:a91546af0a81e4dbc5c1fb2aaa215a455e36b8e28014991238d597c9ef1b15f9",
    ),
    (37, "authentication_context_compaction_v37", "sha256:8ac6c775a376d1787f645ef6526a7aff8e1c70bbb7d22489d06a810d28cfd1be"),
    (38, "authorization_request_residual_storage_v38", "sha256:18c4ef2179dfeea90d8ffec815e89190b1053b87dc4a06410d8b00ee7a789b38"),
    (
        39,
        "graphiti_recovered_ambiguous_progression_v39",
        "sha256:40a2d6a969e6759ad76fa950d122dda1552a81f44f6a65c30c4b1e8389ccc3ed",
    ),
    (
        40,
        "relationship_open_index_v40",
        "sha256:1f126c29773eded31cb36abcf6f6101dd827e0032c9e379adabbbaffd06c1c79",
    ),
    (
        41,
        "command_bound_request_result_storage_v41",
        "sha256:e2beb20eb921e69a8892234f2e4cae2349369700de2f615ffc9d744e8833fd22",
    ),
)


@dataclass(frozen=True, slots=True)
class CompatibilityCell:
    """Deterministic identity and integrity results for one retained schema."""

    version: int
    migration_name: str
    migration_checksum: str
    history: tuple[HistoryRow, ...]
    history_fingerprint: str
    schema_fingerprint: str
    object_inventory: tuple[ObjectRow, ...]
    object_fingerprint: str
    foreign_key_check: tuple[tuple[object, ...], ...]
    quick_check: tuple[str, ...]

    @property
    def object_count(self) -> int:
        return len(self.object_inventory)


def _catalog_text(value: object, *, column: str) -> str:
    if type(value) is not str:
        raise MigrationCompatibilityError(
            f"sqlite_master.{column} must be an exact string"
        )
    return value


def _catalog_sql(value: object) -> str | None:
    if value is None:
        return None
    return _catalog_text(value, column="sql")


def _catalog_row(row: tuple[object, ...] | sqlite3.Row) -> ObjectRow:
    return (
        _catalog_text(row[0], column="type"),
        _catalog_text(row[1], column="name"),
        _catalog_text(row[2], column="tbl_name"),
        _catalog_sql(row[3]),
    )


def _record_tuple(record: MigrationLike) -> HistoryRow:
    return int(record.version), str(record.name), str(record.checksum)


def _require_exact_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise MigrationCompatibilityError(f"{label} must be an exact integer")
    return value


def _discover_statement_bindings() -> dict[str, tuple[str, tuple[str, ...]]]:
    """Bind each named migration record to its authoritative statement tuple."""
    bindings: dict[str, tuple[str, tuple[str, ...]]] = {}
    namespace = vars(authority_migrations)
    for symbol, candidate in namespace.items():
        if symbol == "MIGRATION":
            statement_symbol = "MIGRATION_STATEMENTS"
        elif symbol.endswith("_MIGRATION"):
            statement_symbol = f"{symbol}_STATEMENTS"
        else:
            continue
        statements = namespace.get(statement_symbol)
        if statements is None or not all(
            hasattr(candidate, field) for field in ("version", "name", "checksum")
        ):
            continue
        if not isinstance(statements, tuple) or not all(
            isinstance(statement, str) for statement in statements
        ):
            raise MigrationCompatibilityError(
                f"{statement_symbol} is not an immutable SQL statement tuple"
            )
        migration_name = str(candidate.name)
        binding = (statement_symbol, statements)
        existing = bindings.setdefault(migration_name, binding)
        if existing != binding:
            raise MigrationCompatibilityError(
                f"multiple statement tuples claim migration {migration_name}"
            )
    return bindings


def _checked_registry() -> tuple[MigrationLike, ...]:
    records = tuple(authority_migrations.MIGRATIONS)
    record_history = tuple(_record_tuple(record) for record in records)
    expected_history = tuple(authority_migrations.EXPECTED_MIGRATION_HISTORY)
    if record_history != PINNED_MIGRATION_HISTORY:
        raise MigrationCompatibilityError(
            "MIGRATIONS differ from independent literal release pins"
        )
    if expected_history != PINNED_MIGRATION_HISTORY:
        raise MigrationCompatibilityError(
            "EXPECTED_MIGRATION_HISTORY differs from independent literal release pins"
        )
    versions = tuple(record[0] for record in record_history)
    if versions != (*range(1, 33), 34, 35, 36, 37, 38, 39, 40, 41):
        raise MigrationCompatibilityError(
            "authority migration registry differs from the central release sequence"
        )
    names = tuple(record[1] for record in record_history)
    if len(names) != len(set(names)):
        raise MigrationCompatibilityError("authority migration names are not unique")
    bindings = _discover_statement_bindings()
    statement_names = set(bindings)
    if statement_names != set(names):
        missing = sorted(set(names) - statement_names)
        extra = sorted(statement_names - set(names))
        raise MigrationCompatibilityError(
            f"statement registry differs: missing={missing!r}, extra={extra!r}"
        )
    for version, name, checksum in PINNED_MIGRATION_HISTORY:
        statements = bindings[name][1]
        actual_checksum = digest_canonical(
            {"version": version, "name": name, "statements": list(statements)}
        )
        if actual_checksum != checksum:
            raise MigrationCompatibilityError(
                f"authoritative statements differ from literal release pin for v{version}"
            )
    return records


MIGRATION_REGISTRY = _checked_registry()
STATEMENT_BINDINGS_BY_NAME = _discover_statement_bindings()
STATEMENTS_BY_NAME = {
    name: binding[1] for name, binding in STATEMENT_BINDINGS_BY_NAME.items()
}


def migration_for_version(version: int) -> MigrationLike:
    """Return a migration by its declared version, not by a tail offset."""
    version = _require_exact_int(version, label="migration version")
    matches = [record for record in MIGRATION_REGISTRY if record.version == version]
    if len(matches) != 1:
        raise MigrationCompatibilityError(
            f"expected one named migration for version {version}, found {len(matches)}"
        )
    return matches[0]


def history_through(version: int) -> tuple[HistoryRow, ...]:
    """Return the complete ordered history ending at the named version."""
    migration_for_version(version)
    history = tuple(
        _record_tuple(record)
        for record in MIGRATION_REGISTRY
        if record.version <= version
    )
    if not history or history[-1][0] != version:
        raise MigrationCompatibilityError(f"history does not end at version {version}")
    return history


def statements_for_version(version: int) -> tuple[str, ...]:
    record = migration_for_version(version)
    try:
        return STATEMENTS_BY_NAME[record.name]
    except KeyError as exc:  # pragma: no cover - guarded at module import
        raise MigrationCompatibilityError(
            f"no SQL statements are bound to {record.name}"
        ) from exc


def statement_symbol_for_version(version: int) -> str:
    """Return the authoritative module symbol for a named migration version."""
    record = migration_for_version(version)
    return STATEMENT_BINDINGS_BY_NAME[record.name][0]


def _execute_statement(
    connection: sqlite3.Connection, version: int, index: int, statement: str
) -> None:
    del version, index
    connection.execute(statement)


def build_exact_prefix(
    database: str | Path,
    version: int,
    *,
    statement_executor: StatementExecutor = _execute_statement,
) -> CompatibilityCell:
    """Create a fresh file-backed exact empty schema prefix in one transaction."""
    version = _require_exact_int(version, label="fixture version")
    if version not in RETAINED_VERSIONS:
        raise MigrationCompatibilityError(
            f"retained fixture version must be one of {RETAINED_VERSIONS}"
        )
    path = Path(database)
    if path.exists():
        raise MigrationCompatibilityError(f"fixture database already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN EXCLUSIVE")
        for record in MIGRATION_REGISTRY:
            if record.version > version:
                break
            if record.version == authority_migrations.AUTHORISATION_SCOPE_CONTENT_SCHEMA_VERSION:
                authority_migrations.migrate_authorisation_scope_content(
                    connection,
                    expected_history=tuple(
                        _record_tuple(previous)
                        for previous in MIGRATION_REGISTRY
                        if previous.version < record.version
                    ),
                )
            elif record.version == authority_migrations.SECURITY_RECORD_SCHEMA_VERSION:
                authority_migrations.migrate_security_records(
                    connection,
                    expected_history=tuple(
                        _record_tuple(previous) for previous in MIGRATION_REGISTRY
                        if previous.version < record.version
                    ),
                )
            elif record.version == authority_migrations.AUTHORIZATION_REQUEST_STORAGE_SCHEMA_VERSION:
                authority_migrations.migrate_authorization_request_storage(
                    connection,
                    expected_history=tuple(
                        _record_tuple(previous) for previous in MIGRATION_REGISTRY
                        if previous.version < record.version
                    ),
                )
            elif record.version == authority_migrations.COMMAND_BOUND_STORAGE_SCHEMA_VERSION:
                authority_migrations.migrate_command_bound_storage(
                    connection,
                    expected_history=tuple(
                        _record_tuple(previous) for previous in MIGRATION_REGISTRY
                        if previous.version < record.version
                    ),
                )
            else:
                for index, statement in enumerate(
                    statements_for_version(record.version)
                ):
                    statement_executor(connection, record.version, index, statement)
            connection.execute(
                "INSERT INTO authority_migrations(version,name,checksum,applied_at) "
                "VALUES(?,?,?,?)",
                (*_record_tuple(record), FIXTURE_APPLIED_AT),
            )
        connection.execute(f"PRAGMA user_version={version}")
        connection.execute("COMMIT")
        cell = _snapshot(connection, version)
        if cell.history != history_through(version):
            raise MigrationCompatibilityError(
                f"constructed schema v{version} has incomplete migration history"
            )
        if cell.foreign_key_check or cell.quick_check != ("ok",):
            raise MigrationCompatibilityError(
                f"constructed schema v{version} failed SQLite integrity checks"
            )
        return cell
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def _snapshot(connection: sqlite3.Connection, version: int) -> CompatibilityCell:
    record = migration_for_version(version)
    history = tuple(
        (int(row[0]), str(row[1]), str(row[2]))
        for row in connection.execute(
            "SELECT version,name,checksum FROM authority_migrations ORDER BY version"
        )
    )
    inventory = tuple(
        _catalog_row(row)
        for row in connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name,tbl_name"
        )
    )
    foreign_keys = tuple(
        tuple(row) for row in connection.execute("PRAGMA foreign_key_check")
    )
    quick_check = tuple(str(row[0]) for row in connection.execute("PRAGMA quick_check"))
    return CompatibilityCell(
        version=version,
        migration_name=str(record.name),
        migration_checksum=str(record.checksum),
        history=history,
        history_fingerprint=digest_canonical([list(row) for row in history]),
        schema_fingerprint=authority_migrations.schema_fingerprint(connection),
        object_inventory=inventory,
        object_fingerprint=digest_canonical([list(row) for row in inventory]),
        foreign_key_check=foreign_keys,
        quick_check=quick_check,
    )


@lru_cache(maxsize=len(RETAINED_VERSIONS))
def canonical_cell(version: int) -> CompatibilityCell:
    """Derive the immutable expected cell from a separate fresh file fixture."""
    version = _require_exact_int(version, label="canonical version")
    if version not in RETAINED_VERSIONS:
        raise MigrationCompatibilityError(f"unsupported retained version {version}")
    with tempfile.TemporaryDirectory(prefix=f"authority-v{version}-") as directory:
        path = Path(directory) / "canonical.sqlite3"
        build_exact_prefix(path, version)
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return _snapshot(connection, version)
        finally:
            connection.close()


def inspect_exact_prefix(
    database: str | Path, *, expected_version: int | None = None
) -> CompatibilityCell:
    """Inspect and reject any non-canonical history, object, or integrity result."""
    path = Path(database)
    if not path.is_file():
        raise MigrationCompatibilityError(f"fixture database is not a file: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if expected_version is not None:
            expected_version = _require_exact_int(
                expected_version, label="expected version"
            )
        if expected_version is not None and version != expected_version:
            raise MigrationCompatibilityError(
                f"schema version {version} differs from expected {expected_version}"
            )
        if version not in RETAINED_VERSIONS:
            raise MigrationCompatibilityError(
                f"schema version {version} is outside retained versions"
            )
        actual = _snapshot(connection, version)
    except sqlite3.DatabaseError as exc:
        raise MigrationCompatibilityError(f"database inspection failed: {exc}") from exc
    finally:
        connection.close()

    expected = canonical_cell(version)
    differences: list[str] = []
    if actual.history != expected.history:
        differences.append("migration history")
    if actual.history_fingerprint != expected.history_fingerprint:
        differences.append("migration history fingerprint")
    if actual.schema_fingerprint != expected.schema_fingerprint:
        differences.append("schema fingerprint")
    if actual.object_inventory != expected.object_inventory:
        differences.append("sqlite_master inventory")
    if actual.object_fingerprint != expected.object_fingerprint:
        differences.append("sqlite_master fingerprint")
    if actual.foreign_key_check:
        differences.append("foreign_key_check")
    if actual.quick_check != ("ok",):
        differences.append("quick_check")
    if differences:
        raise MigrationCompatibilityError(
            f"schema v{version} is not exact: {', '.join(differences)}"
        )
    return actual


def prepare_default_connection_backup(connection: sqlite3.Connection) -> object:
    """Prepare a v16-v21 backup without changing default transaction behaviour."""
    if connection.isolation_level is None:
        raise MigrationCompatibilityError(
            "backup proof requires a normal sqlite3.connect default connection"
        )
    if connection.in_transaction:
        raise MigrationCompatibilityError("backup requires no active transaction")
    receipt = authority_migrations.prepare_pending_migration_backup(connection)
    if receipt is None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        raise MigrationCompatibilityError(
            f"schema v{version} has no retained backup boundary"
        )
    if connection.in_transaction:
        raise MigrationCompatibilityError("backup preparation opened a transaction")
    return receipt


def render_compatibility_matrix(
    cells: tuple[CompatibilityCell, ...] | None = None,
) -> str:
    """Render stable, reviewable pins without volatile file-system details."""
    selected = cells or tuple(canonical_cell(version) for version in RETAINED_VERSIONS)
    header = (
        "version | migration | objects | history fingerprint | "
        "schema fingerprint | object fingerprint"
    )
    divider = "--- | --- | ---: | --- | --- | ---"
    rows = [
        f"v{cell.version} | {cell.migration_name} | {cell.object_count} | "
        f"{cell.history_fingerprint} | {cell.schema_fingerprint} | "
        f"{cell.object_fingerprint}"
        for cell in selected
    ]
    return "\n".join((header, divider, *rows)) + "\n"


__all__ = [
    "BACKUP_PREDECESSOR_VERSIONS",
    "CURRENT_VERSION",
    "FIXTURE_APPLIED_AT",
    "MIGRATION_REGISTRY",
    "NEWER_VERSION",
    "PINNED_MIGRATION_HISTORY",
    "PREDECESSOR_VERSION",
    "RETAINED_MIN_VERSION",
    "RETAINED_VERSIONS",
    "UPGRADE_PREDECESSOR_VERSIONS",
    "CompatibilityCell",
    "MigrationCompatibilityError",
    "build_exact_prefix",
    "canonical_cell",
    "history_through",
    "inspect_exact_prefix",
    "migration_for_version",
    "prepare_default_connection_backup",
    "render_compatibility_matrix",
    "statement_symbol_for_version",
    "statements_for_version",
]
