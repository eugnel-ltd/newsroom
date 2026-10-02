from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import sqlite3
import stat
import threading
from time import perf_counter_ns, process_time_ns
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - supported runtime/CI is POSIX
    fcntl = None  # type: ignore[assignment]

from ._capability import _CapabilityIssuer
from ._foreign_keys import has_foreign_key_violation
from .migrations import (
    EXPECTED_MIGRATION_HISTORY,
    EXPECTED_SCHEMA_FINGERPRINT,
    SCHEMA_VERSION,
    apply_pending_migrations,
    prepare_pending_migration_backup,
    schema_fingerprint,
)
from .models import CommittedCommandIdentity
from .persistence import (
    AuthorityPersistenceError,
    AuthoritySchemaError,
    AuthorityWriterBusy,
)
from .policy import CommandRegistry, PayloadSchemaRegistry
from .types import ObjectAdmissionId, PayloadMode, UtcTimestamp, require_token


_OPEN_LOG = logging.getLogger("newsroom.authority.open")


@contextmanager
def _validation_stage(stage: str) -> Iterator[None]:
    """Log inclusive wall and process CPU time; nested phases are not additive."""
    started = perf_counter_ns()
    cpu_started = process_time_ns()
    _OPEN_LOG.info("authority_open stage=%s status=STARTED", stage)
    status = "FAILED"
    try:
        yield
        status = "COMPLETE"
    finally:
        _OPEN_LOG.info(
            "authority_open stage=%s status=%s elapsed_ms=%d cpu_ms=%d",
            stage, status, (perf_counter_ns() - started) // 1_000_000,
            (process_time_ns() - cpu_started) // 1_000_000,
        )


_INCOMPLETE_COMMAND_QUERY = (
    "SELECT c.command_id FROM authority_commands c "
    "WHERE (SELECT COUNT(*) FROM authority_aggregate_versions v "
    "WHERE v.command_id=c.command_id) != 1 "
    "OR (SELECT COUNT(*) FROM ledger_events e "
    "WHERE e.command_id=c.command_id) != 1 "
    "OR (SELECT COUNT(*) FROM authority_audit_events a "
    "WHERE a.command_id=c.command_id) != 1 LIMIT 1"
)


class _EventStoreBase:
    """SQLite lifecycle, migration, validation and writer ownership."""

    _current_state_only = False

    def __init__(
        self,
        path: Path,
        *,
        issuer: _CapabilityIssuer,
        command_registry: CommandRegistry,
        payload_schemas: PayloadSchemaRegistry,
        command_service_version: str,
        busy_timeout_ms: int = 5_000,
        clock: Any = UtcTimestamp.now,
    ) -> None:
        if fcntl is None:
            raise AuthoritySchemaError("POSIX advisory locking is required")
        self.path = Path(path)
        if self.path.is_symlink():
            raise AuthoritySchemaError("authority database path cannot be a symlink")
        if busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be positive")
        require_token(command_service_version, field="command_service_version")
        self._issuer = issuer
        self._command_registry = command_registry
        self._payload_schemas = payload_schemas
        self._command_service_version = command_service_version
        self._busy_timeout_ms = busy_timeout_ms
        self._clock = clock
        self._lock = threading.RLock()
        self._closed = False
        self._lock_fd: int | None = None
        self._conn: sqlite3.Connection | None = None

        self._secure_directory(self.path.parent)
        existed = self.path.exists()
        if existed:
            self._validate_owned_file(self.path)
        try:
            self._acquire_writer_lock()
            self._conn = sqlite3.connect(
                self.path,
                isolation_level=None,
                timeout=busy_timeout_ms / 1000,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._configure_connection()
            if not existed:
                os.chmod(self.path, 0o600)
            self._validate_owned_file(self.path)
            with _validation_stage("validation"):
                self._migrate_or_validate()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _secure_directory(path: Path) -> None:
        if path.exists():
            if path.is_symlink() or not path.is_dir():
                raise AuthoritySchemaError(
                    "authority database parent must be a real directory"
                )
        else:
            path.mkdir(parents=True, mode=0o700)
            os.chmod(path, 0o700)
        info = path.stat()
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise AuthoritySchemaError(
                "authority directory must be owned by the writer"
            )
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise AuthoritySchemaError(
                "authority directory cannot grant group or other permissions"
            )

    @staticmethod
    def _validate_owned_file(path: Path) -> None:
        if path.is_symlink() or not path.is_file():
            raise AuthoritySchemaError(
                "authority database must be a regular file"
            )
        info = path.stat()
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise AuthoritySchemaError(
                "authority database must be owned by the writer"
            )
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise AuthoritySchemaError(
                "authority database cannot grant group or other permissions"
            )

    def _acquire_writer_lock(self) -> None:
        lock_path = self.path.with_name(self.path.name + ".writer.lock")
        if lock_path.is_symlink():
            raise AuthoritySchemaError("writer lock path cannot be a symlink")
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(descriptor)
            raise AuthorityWriterBusy(
                "another authority writer is active"
            ) from exc
        self._lock_fd = descriptor

    @property
    def _connection(self) -> sqlite3.Connection:
        if self._closed or self._conn is None:
            raise AuthorityPersistenceError("authority store is closed")
        return self._conn

    def _configure_connection(self) -> None:
        conn = self._connection
        conn.execute("PRAGMA foreign_keys=ON")
        mode = str(
            conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        ).lower()
        if mode != "wal":
            raise AuthoritySchemaError(
                f"journal_mode WAL unavailable: {mode}"
            )
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute(f"PRAGMA busy_timeout={int(self._busy_timeout_ms)}")
        # Fixed 16 MiB page-cache budget for retained FK lookup locality,
        # independent of database/page size; not a bound on total process RSS.
        conn.execute("PRAGMA cache_size=-16384")

    def _table_names(self) -> set[str]:
        return {
            str(row[0])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }

    def _migrate_or_validate(self) -> None:
        conn = self._connection
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        tables = self._table_names()
        if version > SCHEMA_VERSION:
            raise AuthoritySchemaError(
                f"database schema {version} is newer than supported "
                f"{SCHEMA_VERSION}"
            )
        if version == 0 and tables:
            raise AuthoritySchemaError(
                "refusing to adopt a non-empty unversioned authority database"
            )
        if version < SCHEMA_VERSION:
            prepare_pending_migration_backup(conn)
            apply_pending_migrations(
                conn, applied_at=self._clock().to_text()
            )
        self._validate_schema_and_integrity()

    def _validate_schema_and_integrity(self) -> None:
        conn = self._connection
        with _validation_stage("schema"):
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version != SCHEMA_VERSION:
                raise AuthoritySchemaError(
                    f"database schema {version} does not match {SCHEMA_VERSION}"
                )
            rows = conn.execute(
                "SELECT version,name,checksum FROM authority_migrations "
                "ORDER BY version"
            ).fetchall()
            history = tuple(
                (int(row["version"]), str(row["name"]), str(row["checksum"]))
                for row in rows
            )
            if history != EXPECTED_MIGRATION_HISTORY:
                raise AuthoritySchemaError(
                    f"authority migration history mismatch: {history!r}"
                )
            if schema_fingerprint(conn) != EXPECTED_SCHEMA_FINGERPRINT:
                raise AuthoritySchemaError(
                    "authority schema fingerprint mismatch"
                )
        if not self._current_state_only:
            self._validate_retained_database_history(conn)
        with _validation_stage("connection_settings"):
            if not bool(conn.execute("PRAGMA foreign_keys").fetchone()[0]):
                raise AuthoritySchemaError("SQLite foreign keys are not enabled")
            if str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() != "wal":
                raise AuthoritySchemaError("SQLite WAL mode is not active")
            if int(conn.execute("PRAGMA synchronous").fetchone()[0]) != 2:
                raise AuthoritySchemaError("SQLite synchronous=FULL is not active")
        with _validation_stage("registry_coverage"):
            self._validate_registry_coverage(conn)

    def _validate_retained_database_history(self, conn: sqlite3.Connection) -> None:
        """Historical maintenance, separate from native current-state startup."""
        with _validation_stage("quick_check"):
            quick = [
                str(row[0])
                for row in conn.execute("PRAGMA quick_check").fetchall()
            ]
            if quick != ["ok"]:
                raise AuthoritySchemaError(
                    f"authority quick_check failed: {quick!r}"
                )
        with _validation_stage("foreign_key_check"):
            if has_foreign_key_violation(conn):
                raise AuthoritySchemaError(
                    "authority foreign-key check failed"
                )
        with _validation_stage("relational_invariants"):
            self._validate_relational_invariants(conn)
        with _validation_stage("immutable_records"):
            self._validate_immutable_records(conn)

    def validate_retained_history(self) -> None:
        """Run the complete retained-history checks explicitly under writer ownership."""
        with self._lock:
            current_state_only = self._current_state_only
            self._current_state_only = False
            try:
                self._validate_schema_and_integrity()
            finally:
                self._current_state_only = current_state_only

    def _should_validate_row_integrity(self) -> bool:
        return not self._current_state_only

    def _prove_current_record(self, row: sqlite3.Row) -> None:
        """A selected business row needs its exact event proof, not a boot-time sweep."""
        if self._current_state_only:
            self._validate_retained_event(str(row["authority_event_id"]))

    @staticmethod
    def _validate_relational_invariants(conn: sqlite3.Connection) -> None:
        with _validation_stage("aggregate_heads"):
            missing_head = conn.execute(
                "SELECT a.aggregate_type,a.aggregate_id,a.current_version "
                "FROM authority_aggregates a "
                "LEFT JOIN authority_aggregate_versions v "
                "ON v.aggregate_type=a.aggregate_type "
                "AND v.aggregate_id=a.aggregate_id "
                "AND v.aggregate_version=a.current_version "
                "WHERE v.aggregate_version IS NULL LIMIT 1"
            ).fetchone()
            if missing_head is not None:
                raise AuthoritySchemaError(
                    "aggregate head does not reference an exact version"
                )

        # Correlated indexed counts preserve the exact-one invariant without
        # materialising three COUNT(DISTINCT) temporary B-trees during
        # every authority open.
        with _validation_stage("command_completeness"):
            incomplete = conn.execute(_INCOMPLETE_COMMAND_QUERY).fetchone()
            if incomplete is not None:
                raise AuthoritySchemaError(
                    "each command must own one version, audit and event"
                )

        with _validation_stage("event_routing"):
            mismatch = conn.execute(
                "SELECT e.event_id FROM ledger_events e "
                "JOIN authority_commands c ON c.command_id=e.command_id "
                "JOIN authority_payloads p ON p.payload_id=e.payload_id "
                "WHERE e.producer_version != c.producer_version "
                "OR e.command_definition_version != "
                "c.command_definition_version "
                "OR e.command_definition_digest != "
                "c.command_definition_digest "
                "OR e.payload_id != c.payload_id "
                "OR e.payload_mode != p.mode "
                "OR e.payload_schema_version != p.schema_version "
                "OR e.payload_schema_contract_version != "
                "p.schema_contract_version "
                "OR e.payload_schema_contract_digest != "
                "p.schema_contract_digest "
                "OR e.payload_canonicalizer_version != "
                "p.canonicalizer_implementation_version "
                "OR e.payload_digest != p.payload_digest "
                "OR NOT (e.object_admission_id IS p.object_admission_id) "
                "LIMIT 1"
            ).fetchone()
            if mismatch is not None:
                raise AuthoritySchemaError(
                    "event routing envelope does not match immutable authority"
                )

    def _validate_immutable_records(
        self, conn: sqlite3.Connection
    ) -> None:
        from .command_bound_storage import (
            COMMAND_REQUEST_MARKER, COMMAND_RESULT_PREFIX, COMMAND_RESULT_SELECT, command_request_rows,
            restore_command_result, selected_command_request_backing,
        )
        verified_schemas = {}
        for row in conn.execute("SELECT * FROM payload_schema_contracts"):
            record = self._schema_record_from_row(row)
            verified_schemas[record.contract_digest] = (
                record.canonical_bytes, self._decode_canonical(record.canonical_bytes),
            )
        verified_definitions = {}
        for row in conn.execute("SELECT * FROM command_definitions"):
            record = self._definition_record_from_row(row)
            verified_definitions[record.definition_digest] = (
                record.canonical_bytes, self._decode_canonical(record.canonical_bytes),
            )
        for row in conn.execute("SELECT * FROM authentication_contexts"):
            self._authentication_record_from_row(row)
        # Local verified definitions live for this validation operation only.
        # Each joined row still checks their exact retained bytes and indexes.
        for row in command_request_rows(conn):
            backing = selected_command_request_backing(row) if bytes(row["storage_request_marker"]) == COMMAND_REQUEST_MARKER else None
            self._request_record_from_row(row, connection=conn, selected_backing=backing,
                verified_definitions=verified_definitions, verified_schemas=verified_schemas)
        for row in conn.execute("SELECT * FROM authorization_scope_contents"):
            self._scope_content_from_row(row)
        for row in conn.execute(
            "SELECT d.*,s.canonical_bytes AS selected_scope_canonical_bytes "
            "FROM authorization_decisions d "
            "LEFT JOIN authorization_scope_contents s "
            "ON s.scope_content_digest=d.scope_content_digest"
        ):
            self._decision_record_from_row(
                row,
                selected_scope_bytes=row["selected_scope_canonical_bytes"],
            )
        cursor = conn.execute(COMMAND_RESULT_SELECT)
        for row in cursor:
            data = bytes(row["result_bytes"])
            if data.startswith(COMMAND_RESULT_PREFIX):
                event = {name: row[f"selected_{name}"] for name in (
                    "command_id", "aggregate_type", "aggregate_id", "aggregate_version", "ledger_seq", "event_id",
                )}
                data = restore_command_result(row, data, event_row=event)
            self._decode_result(data, str(row["result_digest"]), replayed=False,
                connection=conn, command_id=str(row["command_id"]))

    def _validate_registry_coverage(
        self, conn: sqlite3.Connection
    ) -> None:
        for row in conn.execute(
            "SELECT command_type,definition_version,definition_digest "
            "FROM command_definitions"
        ).fetchall():
            self._command_registry.resolve_exact(
                str(row["command_type"]),
                str(row["definition_version"]),
                str(row["definition_digest"]),
            )
        for row in conn.execute(
            "SELECT schema_version,payload_mode,contract_version,"
            "contract_digest,canonicalizer_implementation_version "
            "FROM payload_schema_contracts"
        ).fetchall():
            self._payload_schemas.resolve_exact(
                str(row["schema_version"]),
                PayloadMode(str(row["payload_mode"])),
                str(row["contract_version"]),
                str(row["contract_digest"]),
                str(row["canonicalizer_implementation_version"]),
            )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connection
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except sqlite3.OperationalError as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            lowered = str(exc).lower()
            if "locked" in lowered or "busy" in lowered:
                raise AuthorityWriterBusy(
                    "authority writer remained busy"
                ) from exc
            raise AuthorityPersistenceError(str(exc)) from exc
        except sqlite3.DatabaseError as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise AuthorityPersistenceError(str(exc)) from exc
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            if self._lock_fd is not None:
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(self._lock_fd)
                    self._lock_fd = None

    def __enter__(self) -> _EventStoreBase:
        return self

    def __exit__(
        self, exc_type: object, exc: object, tb: object
    ) -> None:
        self.close()

    def find(
        self, *, idempotency_namespace: str, idempotency_key: str
    ) -> CommittedCommandIdentity | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT c.command_id,c.command_type,"
                "c.command_definition_version,"
                "c.command_definition_digest,"
                "c.idempotency_namespace,c.idempotency_key,"
                "c.stable_semantic_request_digest,"
                "p.mode AS payload_mode,p.payload_digest,"
                "p.object_admission_id,"
                "a.principal_id,a.authority_domain "
                "FROM authority_commands c "
                "JOIN authority_payloads p ON p.payload_id=c.payload_id "
                "JOIN authentication_contexts a "
                "ON a.authentication_context_id="
                "c.authentication_context_id "
                "WHERE c.idempotency_namespace=? "
                "AND c.idempotency_key=?",
                (idempotency_namespace, idempotency_key),
            ).fetchone()
            if row is None:
                return None
            return CommittedCommandIdentity(
                command_id=str(row["command_id"]),
                authority_domain=str(row["authority_domain"]),
                principal_id=str(row["principal_id"]),
                command_type=str(row["command_type"]),
                idempotency_namespace=str(
                    row["idempotency_namespace"]
                ),
                idempotency_key=str(row["idempotency_key"]),
                command_definition_version=str(
                    row["command_definition_version"]
                ),
                command_definition_digest=str(
                    row["command_definition_digest"]
                ),
                stable_semantic_request_digest=str(
                    row["stable_semantic_request_digest"]
                ),
                payload_mode=str(row["payload_mode"]),
                payload_digest=str(row["payload_digest"]),
                object_admission_id=(
                    None
                    if row["object_admission_id"] is None
                    else ObjectAdmissionId.parse(
                        str(row["object_admission_id"])
                    )
                ),
            )
