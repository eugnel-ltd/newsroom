"""Private trusted v20 triage execution persistence and composition."""

from __future__ import annotations

import fcntl
import os
import sqlite3
import stat
import uuid
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from threading import Lock, get_ident
from typing import Self

from newsroom.authority.auth import AuthenticationProof, StaticAuthenticator
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority.migrations import (
    EXPECTED_MIGRATION_HISTORY,
    EXPECTED_SCHEMA_FINGERPRINT,
    SCHEMA_VERSION,
    apply_pending_migrations,
    prepare_pending_migration_backup,
    schema_fingerprint,
)
from newsroom.authority.types import UtcTimestamp
from newsroom.increment6.work_items import (
    RetrievalContextAuthority,
    TriageWorkItemStore,
    TriageWorkItemVersion,
)


class TriageExecutionAuthorityError(RuntimeError):
    """The checked v20 execution authority rejected an operation or state."""


def _actor_identity(context: object) -> str:
    try:
        return digest_bytes(
            canonical_json_bytes(
                {
                    "principal_id": context.principal_id,  # type: ignore[attr-defined]
                    "credential_binding_digest": context.credential_binding_digest,  # type: ignore[attr-defined]
                }
            )
        )
    except Exception as exc:
        raise TriageExecutionAuthorityError("authenticated identity differs") from exc


def _owner_profile(context: object) -> str:
    try:
        return digest_bytes(
            canonical_json_bytes(
                {
                    "principal_id": context.principal_id,  # type: ignore[attr-defined]
                    "assurance_class": context.assurance_class,  # type: ignore[attr-defined]
                }
            )
        )
    except Exception as exc:
        raise TriageExecutionAuthorityError("authenticated owner profile differs") from exc


def _execution_time(value: UtcTimestamp) -> str:
    return value.value.isoformat().replace("+00:00", "Z")


def _secure_directory(path: Path) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise TriageExecutionAuthorityError(
                "authority database parent must be a real directory"
            )
    else:
        path.mkdir(parents=True, mode=0o700)
        os.chmod(path, 0o700)
    info = path.stat()
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise TriageExecutionAuthorityError(
            "authority directory must be owned by the writer"
        )
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise TriageExecutionAuthorityError(
            "authority directory cannot grant group or other permissions"
        )


def _validate_owned_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise TriageExecutionAuthorityError(
            "authority database must be a regular file"
        )
    info = path.stat()
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise TriageExecutionAuthorityError(
            "authority database must be owned by the writer"
        )
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise TriageExecutionAuthorityError(
            "authority database cannot grant group or other permissions"
        )


class _TriageExecutionStore:
    def __init__(
        self,
        connection: sqlite3.Connection,
        retrieval_authority: RetrievalContextAuthority,
        authenticator: StaticAuthenticator,
        *,
        clock: Callable[[], UtcTimestamp],
        lease_ttl_seconds: int,
        work_items: TriageWorkItemStore | None = None,
        _validate_on_open: bool = True,
    ) -> None:
        if (
            type(connection) is not sqlite3.Connection
            or connection.in_transaction
            or type(retrieval_authority) is not RetrievalContextAuthority
            or type(authenticator) is not StaticAuthenticator
            or isinstance(lease_ttl_seconds, bool)
            or not isinstance(lease_ttl_seconds, int)
            or lease_ttl_seconds <= 0
            or (
                work_items is not None
                and (
                    type(work_items) is not TriageWorkItemStore
                    or work_items._connection is not connection
                    or work_items._retrieval_authority is not retrieval_authority
                )
            )
        ):
            raise TriageExecutionAuthorityError("execution authority collaborators differ")
        self._connection = connection
        self._retrieval_authority = retrieval_authority
        self._authenticator = authenticator
        self._clock = clock
        self._lease_ttl_seconds = lease_ttl_seconds
        self._transaction_lock = Lock()
        self._transaction_owner: int | None = None
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            retrieval_authority.attach(connection)
            self._work_items = work_items or TriageWorkItemStore(
                connection, retrieval_authority
            )
            self._begin()
            if _validate_on_open:
                self._verify_integrity()
            else:
                self._verify_pending_leases()
            self._commit()
        except BaseException as exc:
            self._rollback()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, TriageExecutionAuthorityError):
                raise
            raise TriageExecutionAuthorityError("execution authority initialisation failed") from exc

    def _begin(self) -> None:
        self._transaction_lock.acquire()
        try:
            if self._connection.in_transaction:
                raise TriageExecutionAuthorityError("connection has an active transaction")
            self._connection.execute("BEGIN IMMEDIATE")
            self._transaction_owner = get_ident()
        except BaseException:
            self._transaction_lock.release()
            raise

    def _commit(self) -> None:
        if self._transaction_owner != get_ident():
            raise TriageExecutionAuthorityError("transaction owner differs")
        if not self._connection.in_transaction:
            raise TriageExecutionAuthorityError("owned transaction is absent")
        self._connection.execute("COMMIT")
        self._transaction_owner = None
        self._transaction_lock.release()

    def _rollback(self) -> None:
        if self._transaction_owner != get_ident():
            return
        try:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
        finally:
            self._transaction_owner = None
            self._transaction_lock.release()

    def _authenticate(self, proof: AuthenticationProof) -> tuple[object, str, str, str]:
        if type(proof) is not AuthenticationProof:
            raise TriageExecutionAuthorityError("authentication proof must be exact")
        now = self._clock()
        try:
            context = self._authenticator.authenticate(proof, now=now)
            context.require_current(now)  # type: ignore[attr-defined]
            return (
                context,
                _actor_identity(context),
                _owner_profile(context),
                str(context.principal_id),  # type: ignore[attr-defined]
            )
        except Exception as exc:
            if isinstance(exc, TriageExecutionAuthorityError):
                raise
            raise TriageExecutionAuthorityError("authentication failed") from exc

    def _recorded_at(self) -> str:
        return self._clock().to_text()

    @staticmethod
    def _event_id(canonical_digest: str, actor: str, recorded_at: str) -> str:
        provenance = digest_bytes(
            canonical_json_bytes(
                {
                    "canonical_digest": canonical_digest,
                    "actor_identity_digest": actor,
                    "recorded_at": recorded_at,
                }
            )
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"triage-execution|{provenance}"))

    @staticmethod
    def _version_matches_member(version: TriageWorkItemVersion, member: object) -> bool:
        retrieval = version.retrieval
        return (
            version.version_id == member.work_item_version_id  # type: ignore[attr-defined]
            and version.canonical_digest == member.work_item_version_digest  # type: ignore[attr-defined]
            and (retrieval.context_id or retrieval.request_id)
            == member.retrieval_context_id  # type: ignore[attr-defined]
            and (retrieval.context_digest or retrieval.request_digest)
            == member.retrieval_context_digest  # type: ignore[attr-defined]
            and version.priority.selection_digest == member.priority_digest  # type: ignore[attr-defined]
        )

    def register_batch(self, batch: object, *, proof: AuthenticationProof) -> object:
        from newsroom.increment6.execution import ExecutionBatch

        if type(batch) is not ExecutionBatch:
            raise TriageExecutionAuthorityError("execution Batch must be exact")
        _, actor, _, _ = self._authenticate(proof)
        raw = batch.canonical_bytes
        try:
            self._begin()
            row = self._connection.execute(
                "SELECT canonical_bytes FROM triage_execution_batches WHERE batch_id=?",
                (batch.batch_id,),
            ).fetchone()
            if row is not None:
                if bytes(row[0]) != raw:
                    raise TriageExecutionAuthorityError("execution Batch replay diverges")
                retained = ExecutionBatch.from_canonical_bytes(bytes(row[0]))
                self._commit()
                return retained
            exact = ExecutionBatch.from_canonical_bytes(raw)
            for member in exact.members:
                current = self._work_items.require_usable_current_in_transaction(
                    member.work_item_id
                )
                if not self._version_matches_member(current, member):
                    raise TriageExecutionAuthorityError(
                        "execution Batch member is not exact current authority"
                    )
            recorded_at = self._recorded_at()
            self._connection.execute(
                "INSERT INTO triage_execution_batches VALUES(?,?,?,?,?,?,?)",
                (
                    exact.batch_id,
                    len(exact.members),
                    raw,
                    exact.canonical_digest,
                    actor,
                    self._event_id(exact.canonical_digest, actor, recorded_at),
                    recorded_at,
                ),
            )
            self._commit()
            return exact
        except BaseException as exc:
            self._rollback()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, TriageExecutionAuthorityError):
                raise
            raise TriageExecutionAuthorityError("execution Batch registration failed") from exc

    def _load_batch(self, batch_id: str) -> object:
        from newsroom.increment6.execution import ExecutionBatch

        row = self._connection.execute(
            "SELECT canonical_bytes,canonical_digest FROM triage_execution_batches WHERE batch_id=?",
            (batch_id,),
        ).fetchone()
        if row is None:
            raise TriageExecutionAuthorityError("retained execution Batch is absent")
        value = ExecutionBatch.from_canonical_bytes(bytes(row[0]))
        if value.canonical_digest != row[1] or value.batch_id != batch_id:
            raise TriageExecutionAuthorityError("retained execution Batch differs")
        return value

    @staticmethod
    def _batch_member(batch: object, attempt: object) -> object:
        members = [
            item
            for item in batch.members  # type: ignore[attr-defined]
            if item.work_item_id == attempt.work_item_id  # type: ignore[attr-defined]
            and item.work_item_version_id == attempt.work_item_version_id  # type: ignore[attr-defined]
        ]
        if len(members) != 1:
            raise TriageExecutionAuthorityError("Worker Attempt is not a Batch member")
        member = members[0]
        if (
            member.work_item_version_digest != attempt.work_item_version_digest  # type: ignore[attr-defined]
            or member.retrieval_context_digest != attempt.retrieval_context_digest  # type: ignore[attr-defined]
            or member.priority_digest != attempt.priority_digest  # type: ignore[attr-defined]
        ):
            raise TriageExecutionAuthorityError("Worker Attempt membership differs")
        return member

    def register_attempt(
        self, batch_id: str, attempt: object, *, proof: AuthenticationProof
    ) -> object:
        from newsroom.increment6.execution import WorkerAttempt

        if type(attempt) is not WorkerAttempt or attempt.ordinal != 1:
            raise TriageExecutionAuthorityError("initial Worker Attempt must be ordinal one")
        _, actor, _, _ = self._authenticate(proof)
        raw = attempt.canonical_bytes
        try:
            self._begin()
            batch = self._load_batch(batch_id)
            row = self._connection.execute(
                "SELECT canonical_bytes,batch_id FROM triage_worker_attempts WHERE attempt_id=?",
                (attempt.attempt_id,),
            ).fetchone()
            if row is not None:
                if str(row[1]) != batch_id:
                    raise TriageExecutionAuthorityError(
                        "Worker Attempt requested Batch differs"
                    )
                if bytes(row[0]) != raw:
                    raise TriageExecutionAuthorityError("Worker Attempt replay diverges")
                retained = WorkerAttempt.from_canonical_bytes(bytes(row[0]))
                self._batch_member(batch, retained)
                self._commit()
                return retained
            conflict = self._connection.execute(
                "SELECT canonical_bytes FROM triage_worker_attempts "
                "WHERE work_item_id=? AND work_item_version_id=? AND ordinal=?",
                (attempt.work_item_id, attempt.work_item_version_id, attempt.ordinal),
            ).fetchone()
            if conflict is not None:
                raise TriageExecutionAuthorityError("Worker Attempt ordinal diverges")
            self._batch_member(batch, attempt)
            current = self._work_items.require_usable_current_in_transaction(
                attempt.work_item_id
            )
            member = self._batch_member(batch, attempt)
            if not self._version_matches_member(current, member):
                raise TriageExecutionAuthorityError("Worker Attempt Work Item is stale")
            self._insert_attempt(batch_id, attempt, actor)
            self._commit()
            return attempt
        except BaseException as exc:
            self._rollback()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, TriageExecutionAuthorityError):
                raise
            raise TriageExecutionAuthorityError("Worker Attempt registration failed") from exc

    def _insert_attempt(self, batch_id: str, attempt: object, actor: str) -> None:
        recorded_at = self._recorded_at()
        self._connection.execute(
            "INSERT INTO triage_worker_attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                attempt.attempt_id,  # type: ignore[attr-defined]
                batch_id,
                attempt.work_item_id,  # type: ignore[attr-defined]
                attempt.work_item_version_id,  # type: ignore[attr-defined]
                attempt.work_item_version_digest,  # type: ignore[attr-defined]
                attempt.retrieval_context_digest,  # type: ignore[attr-defined]
                attempt.priority_digest,  # type: ignore[attr-defined]
                attempt.ordinal,  # type: ignore[attr-defined]
                attempt.previous_attempt_id,  # type: ignore[attr-defined]
                attempt.previous_attempt_digest,  # type: ignore[attr-defined]
                attempt.semantic_request_key,  # type: ignore[attr-defined]
                attempt.canonical_bytes,  # type: ignore[attr-defined]
                attempt.canonical_digest,  # type: ignore[attr-defined]
                actor,
                self._event_id(attempt.canonical_digest, actor, recorded_at),  # type: ignore[attr-defined]
                recorded_at,
            ),
        )

    def _load_attempt_row(self, attempt_id: str) -> tuple[object, str]:
        from newsroom.increment6.execution import WorkerAttempt

        row = self._connection.execute(
            "SELECT canonical_bytes,canonical_digest,batch_id FROM triage_worker_attempts WHERE attempt_id=?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise TriageExecutionAuthorityError("retained Worker Attempt is absent")
        attempt = WorkerAttempt.from_canonical_bytes(bytes(row[0]))
        if attempt.canonical_digest != row[1] or attempt.attempt_id != attempt_id:
            raise TriageExecutionAuthorityError("retained Worker Attempt differs")
        return attempt, str(row[2])

    def _require_current_attempt(self, attempt: object) -> TriageWorkItemVersion:
        current = self._work_items.require_usable_current_in_transaction(
            attempt.work_item_id  # type: ignore[attr-defined]
        )
        retrieval_digest = (
            current.retrieval.context_digest or current.retrieval.request_digest
        )
        if (
            current.version_id != attempt.work_item_version_id  # type: ignore[attr-defined]
            or current.canonical_digest != attempt.work_item_version_digest  # type: ignore[attr-defined]
            or retrieval_digest != attempt.retrieval_context_digest  # type: ignore[attr-defined]
            or current.priority.selection_digest != attempt.priority_digest  # type: ignore[attr-defined]
        ):
            raise TriageExecutionAuthorityError("Worker Attempt Work Item is stale")
        return current

    def claim(self, attempt_id: str, *, proof: AuthenticationProof) -> object:
        from newsroom.increment6.execution import WorkItemLease

        context, actor, owner_profile, owner_id = self._authenticate(proof)
        capability = str(context.credential_binding_digest)  # type: ignore[attr-defined]
        try:
            self._begin()
            row = self._connection.execute(
                "SELECT canonical_bytes FROM triage_work_item_leases WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if row is not None:
                retained = WorkItemLease.from_canonical_bytes(bytes(row[0]))
                self._require_lease_attempt(retained)
                if (
                    retained.owner_id != owner_id
                    or retained.owner_profile_digest != owner_profile
                    or retained.capability_digest != capability
                ):
                    raise TriageExecutionAuthorityError("Lease replay owner differs")
                self._commit()
                return retained
            attempt, batch_id = self._load_attempt_row(attempt_id)
            self._require_current_attempt(attempt)
            claimed = self._connection.execute(
                "SELECT 1 FROM triage_work_item_leases WHERE work_item_id=? AND lifecycle='CLAIMED'",
                (attempt.work_item_id,),
            ).fetchone()
            if claimed is not None:
                raise TriageExecutionAuthorityError("Work Item already has a current Lease")
            fence = int(
                self._connection.execute(
                    "SELECT COALESCE(MAX(fence),0)+1 FROM triage_work_item_leases WHERE work_item_id=?",
                    (attempt.work_item_id,),
                ).fetchone()[0]
            )
            now = self._clock()
            expires = UtcTimestamp(now.value + timedelta(seconds=self._lease_ttl_seconds))
            lease = WorkItemLease.pending(
                attempt=attempt,
                owner_id=owner_id,
                owner_profile_digest=owner_profile,
                capability_digest=capability,
                fence=fence,
            ).claim(
                issued_at=_execution_time(now),
                expires_at=_execution_time(expires),
                actor_identity_digest=actor,
            )
            recorded_at = self._recorded_at()
            self._connection.execute(
                "INSERT INTO triage_work_item_leases VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    lease.lease_id,
                    lease.attempt_id,
                    lease.attempt_digest,
                    batch_id,
                    lease.work_item_id,
                    lease.work_item_version_id,
                    lease.work_item_version_digest,
                    lease.owner_id,
                    lease.owner_profile_digest,
                    lease.capability_digest,
                    lease.fence,
                    lease.lifecycle.value,
                    lease.issued_at,
                    lease.expires_at,
                    lease.canonical_bytes,
                    lease.canonical_digest,
                    actor,
                    self._event_id(lease.canonical_digest, actor, recorded_at),
                    recorded_at,
                    recorded_at,
                ),
            )
            self._commit()
            return lease
        except BaseException as exc:
            self._rollback()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, TriageExecutionAuthorityError):
                raise
            raise TriageExecutionAuthorityError("Work Item Lease claim failed") from exc

    def _load_lease(self, lease_id: str) -> tuple[object, str]:
        from newsroom.increment6.execution import WorkItemLease

        row = self._connection.execute(
            "SELECT canonical_bytes,canonical_digest,batch_id FROM triage_work_item_leases WHERE lease_id=?",
            (lease_id,),
        ).fetchone()
        if row is None:
            raise TriageExecutionAuthorityError("retained Work Item Lease is absent")
        lease = WorkItemLease.from_canonical_bytes(bytes(row[0]))
        if lease.canonical_digest != row[1] or lease.lease_id != lease_id:
            raise TriageExecutionAuthorityError("retained Work Item Lease differs")
        return lease, str(row[2])

    def _require_lease_attempt(self, lease: object) -> tuple[object, str]:
        attempt, batch_id = self._load_attempt_row(lease.attempt_id)  # type: ignore[attr-defined]
        if (
            lease.attempt_digest != attempt.canonical_digest  # type: ignore[attr-defined]
            or lease.work_item_id != attempt.work_item_id  # type: ignore[attr-defined]
            or lease.work_item_version_id != attempt.work_item_version_id  # type: ignore[attr-defined]
            or lease.work_item_version_digest != attempt.work_item_version_digest  # type: ignore[attr-defined]
        ):
            raise TriageExecutionAuthorityError("Lease attempt binding differs")
        return attempt, batch_id

    @staticmethod
    def _require_owner(lease: object, context: object, owner_profile: str) -> None:
        if (
            lease.owner_id != str(context.principal_id)  # type: ignore[attr-defined]
            or lease.owner_profile_digest != owner_profile  # type: ignore[attr-defined]
            or lease.capability_digest != str(context.credential_binding_digest)  # type: ignore[attr-defined]
        ):
            raise TriageExecutionAuthorityError("Lease owner capability differs")

    def _terminal(
        self,
        lease_id: str,
        *,
        proof: AuthenticationProof,
        operation: str,
        evidence_digest: str | None = None,
    ) -> object:
        from newsroom.increment6.execution import (
            LeaseLifecycle,
            LeaseProgress,
            LeaseProgressEvidence,
        )

        context, actor, owner_profile, _ = self._authenticate(proof)
        if evidence_digest is not None:
            if not isinstance(evidence_digest, str) or not evidence_digest.startswith("sha256:"):
                raise TriageExecutionAuthorityError("completion evidence digest differs")
        try:
            self._begin()
            lease, _ = self._load_lease(lease_id)
            self._require_lease_attempt(lease)
            if operation != "expire":
                self._require_owner(lease, context, owner_profile)
            if lease.lifecycle is not LeaseLifecycle.CLAIMED:
                if operation == "complete":
                    progress = tuple(item for receipt in lease.transitions for item in receipt.progress)
                    if (
                        lease.lifecycle is LeaseLifecycle.RELEASED
                        and progress
                        and progress[-1].progress is LeaseProgress.COMPLETED
                        and progress[-1].evidence_digest == evidence_digest
                    ):
                        self._commit()
                        return lease
                elif (
                    operation == "release"
                    and lease.lifecycle is LeaseLifecycle.RELEASED
                    and not lease.transitions[-1].progress
                ):
                    self._commit()
                    return lease
                elif (
                    operation == "expire"
                    and lease.lifecycle is LeaseLifecycle.EXPIRED
                    and not lease.transitions[-1].progress
                ):
                    self._commit()
                    return lease
                raise TriageExecutionAuthorityError("terminal Lease replay diverges")
            now = _execution_time(self._clock())
            if operation == "complete":
                attempt, _ = self._require_lease_attempt(lease)
                self._require_current_attempt(attempt)
                updated = lease.release(
                    observed_at=now,
                    actor_identity_digest=actor,
                    progress=(
                        LeaseProgressEvidence(LeaseProgress.COMPLETED, evidence_digest),
                    ),
                )
            elif operation == "release":
                updated = lease.release(observed_at=now, actor_identity_digest=actor)
            elif operation == "expire":
                if not lease.is_expired_at(now):
                    raise TriageExecutionAuthorityError(
                        "Lease has not reached its expiry boundary"
                    )
                updated = lease.expire(observed_at=now, actor_identity_digest=actor)
            else:
                raise TriageExecutionAuthorityError("unsupported Lease terminal operation")
            updated_at = self._recorded_at()
            changed = self._connection.execute(
                "UPDATE triage_work_item_leases SET lifecycle=?,canonical_bytes=?,canonical_digest=?,"
                "actor_identity_digest=?,authority_event_id=?,updated_at=? "
                "WHERE lease_id=? AND lifecycle='CLAIMED' "
                "AND attempt_digest=? AND owner_id=? AND owner_profile_digest=? "
                "AND capability_digest=? AND fence=?",
                (
                    updated.lifecycle.value,
                    updated.canonical_bytes,
                    updated.canonical_digest,
                    actor,
                    self._event_id(updated.canonical_digest, actor, updated_at),
                    updated_at,
                    lease.lease_id,
                    lease.attempt_digest,
                    lease.owner_id,
                    lease.owner_profile_digest,
                    lease.capability_digest,
                    lease.fence,
                ),
            ).rowcount
            if changed != 1:
                raise TriageExecutionAuthorityError("Lease CAS differs")
            self._commit()
            return updated
        except BaseException as exc:
            self._rollback()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, TriageExecutionAuthorityError):
                raise
            raise TriageExecutionAuthorityError("Lease terminal transition failed") from exc

    def release(self, lease_id: str, *, proof: AuthenticationProof) -> object:
        return self._terminal(lease_id, proof=proof, operation="release")

    def expire(self, lease_id: str, *, proof: AuthenticationProof) -> object:
        return self._terminal(lease_id, proof=proof, operation="expire")

    def complete(
        self, lease_id: str, evidence_digest: str, *, proof: AuthenticationProof
    ) -> object:
        return self._terminal(
            lease_id,
            proof=proof,
            operation="complete",
            evidence_digest=evidence_digest,
        )

    def restart(
        self,
        previous_attempt_id: str,
        *,
        worker_kind: object,
        worker_version: str,
        input_digest: str,
        proof: AuthenticationProof,
    ) -> object:
        from newsroom.increment6.execution import LeaseLifecycle, WorkerAttempt

        _, actor, _, _ = self._authenticate(proof)
        try:
            self._begin()
            previous, batch_id = self._load_attempt_row(previous_attempt_id)
            successor_row = self._connection.execute(
                "SELECT canonical_bytes FROM triage_worker_attempts WHERE previous_attempt_id=?",
                (previous_attempt_id,),
            ).fetchone()
            batch = self._load_batch(batch_id)
            member = self._batch_member(batch, previous)
            expected = WorkerAttempt.create(
                member=member,
                ordinal=previous.ordinal + 1,
                previous_attempt=previous,
                worker_kind=worker_kind,
                worker_version=worker_version,
                input_digest=input_digest,
            )
            if successor_row is not None:
                retained = WorkerAttempt.from_canonical_bytes(bytes(successor_row[0]))
                if retained != expected:
                    raise TriageExecutionAuthorityError("Worker Attempt successor diverges")
                self._commit()
                return retained
            lease_row = self._connection.execute(
                "SELECT lease_id FROM triage_work_item_leases WHERE attempt_id=?",
                (previous_attempt_id,),
            ).fetchone()
            if lease_row is None:
                raise TriageExecutionAuthorityError("restart requires predecessor Lease")
            lease, _ = self._load_lease(str(lease_row[0]))
            retained_attempt, retained_batch_id = self._require_lease_attempt(lease)
            if retained_attempt != previous or retained_batch_id != batch_id:
                raise TriageExecutionAuthorityError("restart predecessor Lease differs")
            if lease.lifecycle is LeaseLifecycle.CLAIMED:
                now = _execution_time(self._clock())
                if not lease.is_expired_at(now):
                    raise TriageExecutionAuthorityError("predecessor Lease is still current")
                expired = lease.expire(observed_at=now, actor_identity_digest=actor)
                updated_at = self._recorded_at()
                changed = self._connection.execute(
                    "UPDATE triage_work_item_leases SET lifecycle=?,canonical_bytes=?,canonical_digest=?,"
                    "actor_identity_digest=?,authority_event_id=?,updated_at=? "
                    "WHERE lease_id=? AND lifecycle='CLAIMED'",
                    (
                        expired.lifecycle.value,
                        expired.canonical_bytes,
                        expired.canonical_digest,
                        actor,
                        self._event_id(expired.canonical_digest, actor, updated_at),
                        updated_at,
                        expired.lease_id,
                    ),
                ).rowcount
                if changed != 1:
                    raise TriageExecutionAuthorityError("Lease reclaim CAS differs")
            elif lease.lifecycle not in {LeaseLifecycle.RELEASED, LeaseLifecycle.EXPIRED}:
                raise TriageExecutionAuthorityError("predecessor Lease is not terminal")
            self._require_current_attempt(previous)
            self._insert_attempt(batch_id, expected, actor)
            self._commit()
            return expected
        except BaseException as exc:
            self._rollback()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, TriageExecutionAuthorityError):
                raise
            raise TriageExecutionAuthorityError("Worker Attempt restart failed") from exc

    def _verify_pending_leases(self) -> None:
        """Verify recoverable lease obligations, not terminal execution diagnostics."""
        from newsroom.increment6.execution import LeaseLifecycle

        for row in self._connection.execute(
            "SELECT lease_id FROM triage_work_item_leases WHERE lifecycle='CLAIMED'"
        ):
            lease, batch_id = self._load_lease(str(row[0]))
            if lease.lifecycle is not LeaseLifecycle.CLAIMED:
                raise TriageExecutionAuthorityError("pending Lease lifecycle differs")
            attempt, retained_batch_id = self._require_lease_attempt(lease)
            if retained_batch_id != batch_id:
                raise TriageExecutionAuthorityError("pending Lease Batch differs")
            batch = self._load_batch(batch_id)
            member = self._batch_member(batch, attempt)
            version = self._work_items.load_version(member.work_item_version_id)
            if not self._version_matches_member(version, member):
                raise TriageExecutionAuthorityError("pending Lease Work Item differs")

    def _verify_integrity(self) -> None:
        from newsroom.increment6.execution import ExecutionBatch, WorkItemLease, WorkerAttempt

        required = {
            "triage_execution_batches",
            "triage_worker_attempts",
            "triage_work_item_leases",
        }
        tables = {
            str(row[0])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not required <= tables:
            raise TriageExecutionAuthorityError("v20 execution schema is absent")
        batches: dict[str, object] = {}
        for row in self._connection.execute(
            "SELECT batch_id,member_count,canonical_bytes,canonical_digest,"
            "actor_identity_digest,authority_event_id,recorded_at "
            "FROM triage_execution_batches"
        ):
            batch = ExecutionBatch.from_canonical_bytes(bytes(row[2]))
            if (
                batch.batch_id != row[0]
                or len(batch.members) != row[1]
                or batch.canonical_digest != row[3]
                or row[5] != self._event_id(str(row[3]), str(row[4]), str(row[6]))
            ):
                raise TriageExecutionAuthorityError("retained execution Batch differs")
            for member in batch.members:
                try:
                    version = self._work_items.load_version(
                        member.work_item_version_id
                    )
                except Exception as exc:
                    raise TriageExecutionAuthorityError(
                        "execution Batch Work Item Version is absent"
                    ) from exc
                if not self._version_matches_member(version, member):
                    raise TriageExecutionAuthorityError(
                        "retained execution Batch member differs"
                    )
            batches[batch.batch_id] = batch
        attempts: dict[str, object] = {}
        attempt_batches: dict[str, str] = {}
        for row in self._connection.execute(
            "SELECT attempt_id,batch_id,work_item_id,work_item_version_id,"
            "work_item_version_digest,retrieval_context_digest,priority_digest,"
            "ordinal,previous_attempt_id,previous_attempt_digest,"
            "semantic_request_key,canonical_bytes,canonical_digest,"
            "actor_identity_digest,authority_event_id,recorded_at "
            "FROM triage_worker_attempts "
            "ORDER BY work_item_id,work_item_version_id,ordinal"
        ):
            attempt = WorkerAttempt.from_canonical_bytes(bytes(row[11]))
            batch = batches.get(str(row[1]))
            if (
                attempt.attempt_id != row[0]
                or attempt.work_item_id != row[2]
                or attempt.work_item_version_id != row[3]
                or attempt.work_item_version_digest != row[4]
                or attempt.retrieval_context_digest != row[5]
                or attempt.priority_digest != row[6]
                or attempt.ordinal != row[7]
                or attempt.previous_attempt_id != row[8]
                or attempt.previous_attempt_digest != row[9]
                or attempt.semantic_request_key != row[10]
                or attempt.canonical_digest != row[12]
                or row[14]
                != self._event_id(str(row[12]), str(row[13]), str(row[15]))
                or batch is None
            ):
                raise TriageExecutionAuthorityError("retained Worker Attempt differs")
            self._batch_member(batch, attempt)
            if attempt.previous_attempt_id is not None:
                predecessor = attempts.get(attempt.previous_attempt_id)
                if (
                    predecessor is None
                    or predecessor.ordinal + 1 != attempt.ordinal
                    or predecessor.work_item_id != attempt.work_item_id
                    or predecessor.work_item_version_id
                    != attempt.work_item_version_id
                    or predecessor.work_item_version_digest
                    != attempt.work_item_version_digest
                    or predecessor.retrieval_context_digest
                    != attempt.retrieval_context_digest
                    or predecessor.priority_digest != attempt.priority_digest
                    or predecessor.canonical_digest
                    != attempt.previous_attempt_digest
                    or attempt_batches.get(attempt.previous_attempt_id) != str(row[1])
                ):
                    raise TriageExecutionAuthorityError("Worker Attempt predecessor chain differs")
            attempts[attempt.attempt_id] = attempt
            attempt_batches[attempt.attempt_id] = str(row[1])
        claimed: set[str] = set()
        attempt_leases: dict[str, object] = {}
        for row in self._connection.execute(
            "SELECT lease_id,attempt_id,attempt_digest,batch_id,work_item_id,"
            "work_item_version_id,work_item_version_digest,owner_id,"
            "owner_profile_digest,capability_digest,fence,lifecycle,issued_at,"
            "expires_at,canonical_bytes,canonical_digest,actor_identity_digest,"
            "authority_event_id,recorded_at,updated_at FROM triage_work_item_leases"
        ):
            lease = WorkItemLease.from_canonical_bytes(bytes(row[14]))
            attempt = attempts.get(str(row[1]))
            if (
                lease.lease_id != row[0]
                or attempt is None
                or lease.attempt_digest != row[2]
                or lease.attempt_digest != attempt.canonical_digest
                or str(row[3]) != attempt_batches[attempt.attempt_id]
                or lease.work_item_id != row[4]
                or lease.work_item_version_id != row[5]
                or lease.work_item_version_digest != row[6]
                or lease.owner_id != row[7]
                or lease.owner_profile_digest != row[8]
                or lease.capability_digest != row[9]
                or lease.fence != row[10]
                or lease.lifecycle.value != row[11]
                or lease.issued_at != row[12]
                or lease.expires_at != row[13]
                or lease.canonical_digest != row[15]
                or row[17]
                != self._event_id(str(row[15]), str(row[16]), str(row[19]))
            ):
                raise TriageExecutionAuthorityError("retained Work Item Lease differs")
            if (
                not lease.transitions
                or row[16] != lease.transitions[-1].actor_identity_digest
            ):
                raise TriageExecutionAuthorityError(
                    "retained Work Item Lease provenance differs"
                )
            if lease.lifecycle.value == "CLAIMED":
                if lease.work_item_id in claimed:
                    raise TriageExecutionAuthorityError(
                        "multiple current Work Item owners"
                    )
                claimed.add(lease.work_item_id)
            attempt_leases[lease.attempt_id] = lease
        for attempt in attempts.values():
            if attempt.previous_attempt_id is None:
                continue
            predecessor_lease = attempt_leases.get(attempt.previous_attempt_id)
            if (
                predecessor_lease is None
                or predecessor_lease.lifecycle.value not in {"RELEASED", "EXPIRED"}
            ):
                raise TriageExecutionAuthorityError(
                    "Worker Attempt predecessor Lease differs"
                )
        # The owning standalone/native opener checks the whole database once.
        # Recheck every execution child reference, including cross-domain Work
        # Item Versions, without rescanning unrelated retained authority rows.
        for table in sorted(required):
            if self._connection.execute(
                f'PRAGMA foreign_key_check("{table}")'
            ).fetchone() is not None:
                raise TriageExecutionAuthorityError("execution authority foreign keys differ")


class TriageExecutionAuthority:
    """Narrow public authority facade; SQLite and capability internals stay private."""

    __slots__ = ("__store", "__close", "__closed")

    def __init__(self, store: _TriageExecutionStore, close: Callable[[], None]) -> None:
        self.__store = store
        self.__close = close
        self.__closed = False

    def register_batch(self, batch: object, *, proof: AuthenticationProof) -> object:
        return self.__store.register_batch(batch, proof=proof)

    def register_attempt(
        self, batch_id: str, attempt: object, *, proof: AuthenticationProof
    ) -> object:
        return self.__store.register_attempt(batch_id, attempt, proof=proof)

    def claim(self, attempt_id: str, *, proof: AuthenticationProof) -> object:
        return self.__store.claim(attempt_id, proof=proof)

    def release(self, lease_id: str, *, proof: AuthenticationProof) -> object:
        return self.__store.release(lease_id, proof=proof)

    def expire(self, lease_id: str, *, proof: AuthenticationProof) -> object:
        return self.__store.expire(lease_id, proof=proof)

    def complete(
        self, lease_id: str, evidence_digest: str, *, proof: AuthenticationProof
    ) -> object:
        return self.__store.complete(lease_id, evidence_digest, proof=proof)

    def restart(
        self,
        previous_attempt_id: str,
        *,
        worker_kind: object,
        worker_version: str,
        input_digest: str,
        proof: AuthenticationProof,
    ) -> object:
        return self.__store.restart(
            previous_attempt_id,
            worker_kind=worker_kind,
            worker_version=worker_version,
            input_digest=input_digest,
            proof=proof,
        )

    def close(self) -> None:
        if not self.__closed:
            self.__closed = True
            self.__close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _open_on_connection(
    connection: sqlite3.Connection,
    *,
    retrieval_authority: RetrievalContextAuthority,
    authenticator: StaticAuthenticator,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
    lease_ttl_seconds: int = 300,
    work_items: TriageWorkItemStore | None = None,
    _validate_on_open: bool = True,
) -> TriageExecutionAuthority:
    store = _TriageExecutionStore(
        connection,
        retrieval_authority,
        authenticator,
        clock=clock,
        lease_ttl_seconds=lease_ttl_seconds,
        work_items=work_items,
        _validate_on_open=_validate_on_open,
    )
    return TriageExecutionAuthority(store, lambda: None)


def open_triage_execution_authority(
    database: str | Path,
    *,
    retrieval_authority: RetrievalContextAuthority,
    authenticator: StaticAuthenticator,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
    lease_ttl_seconds: int = 300,
    busy_timeout_ms: int = 5_000,
) -> TriageExecutionAuthority:
    if (
        isinstance(busy_timeout_ms, bool)
        or not isinstance(busy_timeout_ms, int)
        or busy_timeout_ms <= 0
    ):
        raise TriageExecutionAuthorityError("busy timeout must be positive")
    path = Path(database).expanduser().absolute()
    if path.is_symlink():
        raise TriageExecutionAuthorityError("authority database path cannot be a symlink")
    try:
        _secure_directory(path.parent)
    except Exception as exc:
        raise TriageExecutionAuthorityError("authority directory differs") from exc
    existed = path.exists()
    if existed:
        try:
            _validate_owned_file(path)
        except Exception as exc:
            raise TriageExecutionAuthorityError("authority database file differs") from exc
    lock_path = path.with_name(path.name + ".writer.lock")
    if lock_path.is_symlink():
        raise TriageExecutionAuthorityError("writer lock path cannot be a symlink")
    if lock_path.exists():
        try:
            _validate_owned_file(lock_path)
        except Exception as exc:
            raise TriageExecutionAuthorityError("writer lock file differs") from exc
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    os.fchmod(descriptor, 0o600)
    lock_info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(lock_info.st_mode)
        or (hasattr(os, "getuid") and lock_info.st_uid != os.getuid())
        or stat.S_IMODE(lock_info.st_mode) & 0o077
    ):
        os.close(descriptor)
        raise TriageExecutionAuthorityError("writer lock ownership differs")
    connection: sqlite3.Connection | None = None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise TriageExecutionAuthorityError(
                "another authority writer is active"
            ) from exc
        connection = sqlite3.connect(
            path,
            isolation_level=None,
            timeout=busy_timeout_ms / 1000,
            check_same_thread=False,
        )
        if not existed:
            os.chmod(path, 0o600)
        _validate_owned_file(path)
        connection.execute("PRAGMA foreign_keys=ON")
        mode = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
        if mode != "wal":
            raise TriageExecutionAuthorityError("SQLite WAL mode is unavailable")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        current = int(connection.execute("PRAGMA user_version").fetchone()[0])
        tables = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone()
        if current == 0 and tables is not None:
            raise TriageExecutionAuthorityError(
                "refusing a non-empty unversioned authority database"
            )
        if current < SCHEMA_VERSION:
            prepare_pending_migration_backup(connection)
        apply_pending_migrations(connection, applied_at=clock().to_text())
        history = connection.execute(
            "SELECT version,name,checksum FROM authority_migrations ORDER BY version"
        ).fetchall()
        if (
            int(connection.execute("PRAGMA user_version").fetchone()[0]) != SCHEMA_VERSION
            or history != list(EXPECTED_MIGRATION_HISTORY)
            or schema_fingerprint(connection) != EXPECTED_SCHEMA_FINGERPRINT
            or connection.execute("PRAGMA quick_check").fetchone()[0] != "ok"
            or connection.execute("PRAGMA foreign_key_check").fetchone() is not None
            or connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1
            or connection.execute("PRAGMA synchronous").fetchone()[0] != 2
            or connection.execute("PRAGMA busy_timeout").fetchone()[0]
            != busy_timeout_ms
            or str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            != "wal"
        ):
            raise TriageExecutionAuthorityError("checked authority lifecycle differs")
        store = _TriageExecutionStore(
            connection,
            retrieval_authority,
            authenticator,
            clock=clock,
            lease_ttl_seconds=lease_ttl_seconds,
        )

        def close() -> None:
            assert connection is not None
            connection.close()
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

        return TriageExecutionAuthority(store, close)
    except Exception:
        if connection is not None:
            connection.close()
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        raise


__all__ = [
    "TriageExecutionAuthority",
    "TriageExecutionAuthorityError",
    "open_triage_execution_authority",
]
