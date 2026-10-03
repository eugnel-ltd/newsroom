from __future__ import annotations

from ._event_store_base import _validation_stage

from contextlib import contextmanager
import json
import sqlite3
from typing import Any, Iterator

from ._capability import _AuthorizedCommandGrant
from ._object_capability import _ObjectCapabilityIssuer
from ._object_cas import _GovernedCAS, _PinnedBlob
from .canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
    validate_sha256_digest,
)
from .models import ObjectAdmissionDescriptor
from .object_policy import (
    HydrationPolicyRegistry,
    ObjectAdmissionRegistry,
    RightsPolicyRegistry,
)
from .objects import (
    AdmissionState,
    BlobIdentity,
    BlobIntegrityState,
    BlobLifecycleState,
    BlobLifecycleView,
    DeletionState,
    ObjectAdmissionDenied,
    ObjectAdmissionId,
    ObjectAdmissionView,
    ObjectIntegrityError,
    ObjectLifecycleError,
    ObjectLivenessSnapshot,
    RightsDecisionId,
    RightsDecisionView,
)
from .persistence import AuthorityPersistenceError, UnsupportedPayloadMode
from .types import EventId, PayloadMode, UtcTimestamp


class _ObjectStoreBase:
    """Policy persistence, current-state resolution, CAS hooks, and reconciliation."""

    def _migrate_or_validate(self) -> None:
        """Apply A2a/A2b migrations, retain exact object contracts, then validate."""

        from .migrations import (
            SCHEMA_VERSION,
            apply_pending_migrations,
            prepare_pending_migration_backup,
        )
        from .persistence import AuthoritySchemaError

        conn = self._connection
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        tables = self._table_names()
        if version == 44 and self._current_state_only:
            from .native_current_checkpoint_migrations import require_checkpoint_schema
            require_checkpoint_schema(conn)
            self._native_checkpoint_schema = True
        elif version > SCHEMA_VERSION:
            raise AuthoritySchemaError(
                f"database schema {version} is newer than supported {SCHEMA_VERSION}"
            )
        if version == 0 and tables:
            raise AuthoritySchemaError(
                "refusing to adopt a non-empty unversioned authority database"
            )
        if version < SCHEMA_VERSION:
            prepare_pending_migration_backup(conn)
            apply_pending_migrations(conn, applied_at=self._clock().to_text())
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._persist_object_contracts(
                conn, recorded_at=self._clock().to_text()
            )
            conn.execute("COMMIT")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        self._validate_schema_and_integrity()


    def _configure_object_store(
        self,
        *,
        object_issuer: _ObjectCapabilityIssuer,
        admission_registry: ObjectAdmissionRegistry,
        rights_policies: RightsPolicyRegistry,
        hydration_policies: HydrationPolicyRegistry,
        cas: _GovernedCAS,
    ) -> None:
        self._object_issuer = object_issuer
        self._admission_registry = admission_registry
        self._rights_policies = rights_policies
        self._hydration_policies = hydration_policies
        self._cas = cas
        with _validation_stage("cas_reconciliation"):
            self._reconcile_objects()

    def _persist_object_contracts(
        self, conn: sqlite3.Connection, *, recorded_at: str
    ) -> None:
        # Foreign-key order is rights/hydration contracts before definitions.
        for contract in self._rights_policies.contracts():
            canonical = canonical_json_bytes(contract.canonical_value())
            if digest_bytes(canonical) != contract.contract_digest:
                raise AuthorityPersistenceError(
                    "rights policy contract digest mismatch"
                )
            conn.execute(
                "INSERT OR IGNORE INTO rights_policy_contracts("
                "contract_digest,policy_key,contract_version,implementation_version,"
                "canonical_bytes,registered_at) VALUES(?,?,?,?,?,?)",
                (
                    contract.contract_digest,
                    contract.policy_key,
                    contract.contract_version,
                    contract.implementation_version,
                    canonical,
                    recorded_at,
                ),
            )
            row = conn.execute(
                "SELECT canonical_bytes FROM rights_policy_contracts "
                "WHERE contract_digest=?",
                (contract.contract_digest,),
            ).fetchone()
            if row is None or bytes(row["canonical_bytes"]) != canonical:
                raise AuthorityPersistenceError(
                    "rights policy contract identity conflict"
                )

        for contract in self._hydration_policies.contracts():
            canonical = canonical_json_bytes(contract.canonical_value())
            if digest_bytes(canonical) != contract.contract_digest:
                raise AuthorityPersistenceError(
                    "hydration policy contract digest mismatch"
                )
            conn.execute(
                "INSERT OR IGNORE INTO hydration_policy_contracts("
                "contract_digest,policy_id,contract_version,implementation_version,"
                "purpose,canonical_bytes,registered_at) VALUES(?,?,?,?,?,?,?)",
                (
                    contract.contract_digest,
                    contract.policy_id,
                    contract.contract_version,
                    contract.implementation_version,
                    contract.purpose,
                    canonical,
                    recorded_at,
                ),
            )
            row = conn.execute(
                "SELECT canonical_bytes FROM hydration_policy_contracts "
                "WHERE contract_digest=?",
                (contract.contract_digest,),
            ).fetchone()
            if row is None or bytes(row["canonical_bytes"]) != canonical:
                raise AuthorityPersistenceError(
                    "hydration policy contract identity conflict"
                )

        for definition in self._admission_registry.definitions():
            canonical = canonical_json_bytes(definition.canonical_value())
            if digest_bytes(canonical) != definition.digest:
                raise AuthorityPersistenceError(
                    "object admission definition digest mismatch"
                )
            conn.execute(
                "INSERT OR IGNORE INTO object_admission_definitions("
                "definition_digest,admission_type,definition_version,"
                "object_class,allowed_use,security_scope,retention_scope,"
                "required_write_scope,required_read_scope,required_manage_scope,"
                "rights_policy_contract_digest,hydration_policy_contract_digests,"
                "canonical_bytes,registered_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    definition.digest,
                    definition.admission_type,
                    definition.definition_version,
                    definition.object_class,
                    definition.allowed_use,
                    definition.security_scope,
                    definition.retention_scope,
                    definition.required_write_scope,
                    definition.required_read_scope,
                    definition.required_manage_scope,
                    definition.rights_policy_contract_digest,
                    canonical_json_bytes(
                        sorted(definition.hydration_policy_contract_digests)
                    ),
                    canonical,
                    recorded_at,
                ),
            )
            row = conn.execute(
                "SELECT canonical_bytes FROM object_admission_definitions "
                "WHERE definition_digest=?",
                (definition.digest,),
            ).fetchone()
            if row is None or bytes(row["canonical_bytes"]) != canonical:
                raise AuthorityPersistenceError(
                    "object admission definition identity conflict"
                )

    def persist_object_contracts(self) -> None:
        with self._lock, self._transaction() as conn:
            self._persist_object_contracts(
                conn, recorded_at=self._clock().to_text()
            )

    @staticmethod
    def _decode_canonical_object(data: bytes) -> Any:
        try:
            value = json.loads(data.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise AuthorityPersistenceError("stored object JSON is invalid") from exc
        if canonical_json_bytes(value) != data:
            raise AuthorityPersistenceError("stored object JSON is not canonical")
        return value

    def _require_canonical_record(self, row: sqlite3.Row) -> dict[str, Any]:
        """Decode an immutable canonical record and verify its stored digest."""

        data = bytes(row["canonical_bytes"])
        digest = str(row["canonical_digest"])
        if digest_bytes(data) != digest:
            raise AuthorityPersistenceError(
                "stored object record digest mismatch"
            )
        value = self._decode_canonical_object(data)
        if not isinstance(value, dict):
            raise AuthorityPersistenceError(
                "stored object record is not a canonical object"
            )
        return value

    def _verified_rights_value(self, rights: sqlite3.Row) -> dict[str, Any]:
        """One exact rights record, shared by receipt reads and current effects."""
        value = self._require_canonical_record(rights)
        expected = {
            field: rights[field]
            for field in (
                "rights_decision_id", "authentication_context_id",
                "authorization_request_digest", "authorization_decision_id",
                "rights_request_digest", "policy_contract_digest",
                "admission_definition_digest", "object_class", "allowed_use",
                "security_scope", "retention_scope", "reason_code",
                "decided_at", "valid_from", "valid_until",
            )
        }
        expected.update({
            "blob": {"blob_digest": rights["blob_digest"], "size_bytes": rights["size_bytes"]},
            "allowed": bool(rights["allowed"]),
        })
        if value != expected:
            raise AuthorityPersistenceError("access decision rights indexed fields differ")
        return value

    def _prove_current_admission(self, conn: sqlite3.Connection, row: sqlite3.Row) -> None:
        rights = conn.execute(
            "SELECT * FROM object_rights_decisions WHERE rights_decision_id=?", (row["rights_decision_id"],),
        ).fetchone()
        if rights is None:
            raise AuthorityPersistenceError("current admission rights are absent")
        value = self._verified_rights_value(rights)
        if any(value[rights_field] != row[admission_field] for rights_field, admission_field in (
            ("admission_definition_digest", "definition_digest"), ("object_class", "object_class"),
            ("allowed_use", "allowed_use"), ("security_scope", "security_scope"),
            ("retention_scope", "retention_scope"), ("valid_from", "valid_from"), ("valid_until", "valid_until"),
        )) or value["blob"] != {"blob_digest": row["blob_digest"], "size_bytes": row["size_bytes"]}:
            raise AuthorityPersistenceError("current admission differs from exact rights")
        authentication = conn.execute(
            "SELECT * FROM authentication_contexts WHERE authentication_context_id=?", (rights["authentication_context_id"],),
        ).fetchone()
        request = conn.execute(
            "SELECT * FROM authorization_requests WHERE request_digest=?", (rights["authorization_request_digest"],),
        ).fetchone()
        decision = conn.execute(
            "SELECT * FROM authorization_decisions WHERE authorization_decision_id=?", (rights["authorization_decision_id"],),
        ).fetchone()
        if any(record is None for record in (authentication, request, decision)):
            raise AuthorityPersistenceError("current rights security closure is absent")
        context = self._authentication_record_from_row(authentication)
        requested = self._request_record_from_row(request, connection=conn)
        authorised = self._decision_record_from_row(decision, connection=conn)
        if (
            not authorised.allowed
            or requested.authentication_context_id != context.authentication_context_id
            or authorised.authentication_context_id != context.authentication_context_id
            or authorised.authorization_request_digest != requested.request_digest
        ):
            raise AuthorityPersistenceError("current rights security closure differs")
        if row["event_id"] is None:
            raise AuthorityPersistenceError("current admission activation is absent")
        self._prove_admission_lifecycle(conn, row)

    def _object_lifecycle_event(self, conn: sqlite3.Connection, event_id: str, identity: str):
        self._validate_retained_event(event_id)
        event = conn.execute(
            "SELECT e.*,p.payload_bytes FROM ledger_events e "
            "JOIN authority_payloads p ON p.payload_id=e.payload_id WHERE e.event_id=?",
            (event_id,),
        ).fetchone()
        if event is None or event["aggregate_type"] != "governed_object_lifecycle" or event["aggregate_id"] != identity:
            raise AuthorityPersistenceError("selected object lifecycle event identity differs")
        return event, self._decode_canonical_object(bytes(event["payload_bytes"]))

    def _prove_admission_lifecycle(self, conn: sqlite3.Connection, row: sqlite3.Row) -> None:
        version = conn.execute(
            "SELECT v.*,h.updated_at AS head_updated_at FROM object_admission_versions v "
            "JOIN object_admission_heads h ON h.admission_id=v.admission_id AND h.current_version=v.lifecycle_version "
            "WHERE v.admission_id=? AND v.lifecycle_version="
            "(SELECT MAX(lifecycle_version) FROM object_admission_versions WHERE admission_id=?)",
            (row["admission_id"], row["admission_id"]),
        ).fetchone()
        if version is None or version["lifecycle_version"] != row["current_version"] or version["state"] != row["state"]:
            raise AuthorityPersistenceError("current admission lifecycle head is not the exact latest version")
        event, payload = self._object_lifecycle_event(conn, str(version["event_id"]), str(row["admission_id"]))
        state = {
            "governed_object.admission.activated": AdmissionState.ACTIVE.value,
            "governed_object.admission.revoked": AdmissionState.REVOKED.value,
        }.get(str(event["event_type"]))
        head = conn.execute(
            "SELECT current_version FROM authority_aggregates WHERE aggregate_type=? AND aggregate_id=?",
            (event["aggregate_type"], event["aggregate_id"]),
        ).fetchone()
        if (
            state != version["state"] or head is None or head["current_version"] != event["aggregate_version"]
            or event["aggregate_version"] != version["lifecycle_version"]
            or event["recorded_at"] != version["recorded_at"] or version["head_updated_at"] != version["recorded_at"]
            or payload.get("admission_id") != row["admission_id"] or payload.get("operation_id") != version["operation_id"]
        ):
            raise AuthorityPersistenceError("current admission lifecycle differs from exact authority")
        if state == AdmissionState.ACTIVE.value:
            expected = {name: row[name] for name in (
                "admission_id", "blob_digest", "size_bytes", "definition_digest", "rights_decision_id",
                "object_class", "allowed_use", "security_scope", "retention_scope", "valid_from", "valid_until",
            )}
            expected["operation_id"] = version["operation_id"]
            detail = {
                "admission_id": row["admission_id"], "state": state, "event_id": str(version["event_id"]),
                "rights_decision_id": row["rights_decision_id"],
            }
        else:
            expected = {"admission_id": row["admission_id"], "operation_id": version["operation_id"], "reason_code": version["reason_code"]}
            detail = expected
        if payload != expected or digest_canonical(detail) != version["detail_digest"]:
            raise AuthorityPersistenceError("current admission lifecycle payload differs")

    def _admission_row(
        self, admission_id: str, *, conn: sqlite3.Connection | None = None
    ) -> sqlite3.Row:
        selected = conn or self._connection
        row = selected.execute(
            "SELECT a.*,h.current_version,"
            "h.current_version AS admission_lifecycle_version,"
            "v.state,v.event_id,v.recorded_at,b.size_bytes,"
            "bh.current_version AS blob_lifecycle_version,"
            "r.allowed AS rights_allowed,r.reason_code,r.decided_at,"
            "r.valid_from AS rights_valid_from,"
            "r.valid_until AS rights_valid_until,"
            "r.canonical_digest AS rights_digest,"
            "r.canonical_digest AS rights_decision_digest "
            "FROM object_admissions a "
            "JOIN object_admission_heads h ON h.admission_id=a.admission_id "
            "JOIN object_admission_versions v "
            "ON v.admission_id=h.admission_id "
            "AND v.lifecycle_version=h.current_version "
            "JOIN blob_identities b ON b.blob_digest=a.blob_digest "
            "JOIN blob_lifecycle_heads bh ON bh.blob_digest=a.blob_digest "
            "JOIN object_rights_decisions r "
            "ON r.rights_decision_id=a.rights_decision_id "
            "WHERE a.admission_id=?",
            (admission_id,),
        ).fetchone()
        if row is None:
            raise KeyError(admission_id)
        return row

    def _admission_activation_row(
        self,
        admission_id: str,
        *,
        event_id: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> sqlite3.Row:
        """Return the immutable admission result committed at activation.

        Idempotent command replay returns the original committed result, not a
        projection of later revocation, expiry, deletion, or blob lifecycle
        state.  This row is metadata only and therefore deliberately does not
        require current rights or bytes.
        """

        selected = conn or self._connection
        event_clause = "" if event_id is None else " AND v.event_id=?"
        parameters: tuple[str, ...] = (
            (admission_id,) if event_id is None else (admission_id, event_id)
        )
        row = selected.execute(
            "SELECT a.*,1 AS current_version,"
            "1 AS admission_lifecycle_version,v.state,v.event_id,v.recorded_at,"
            "b.size_bytes,1 AS blob_lifecycle_version,"
            "r.allowed AS rights_allowed,r.reason_code,r.decided_at,"
            "r.valid_from AS rights_valid_from,"
            "r.valid_until AS rights_valid_until,"
            "r.canonical_digest AS rights_digest,"
            "r.canonical_digest AS rights_decision_digest "
            "FROM object_admissions a "
            "JOIN object_admission_versions v "
            "ON v.admission_id=a.admission_id "
            "AND v.lifecycle_version=1 AND v.state='ACTIVE' "
            "JOIN blob_identities b ON b.blob_digest=a.blob_digest "
            "JOIN object_rights_decisions r "
            "ON r.rights_decision_id=a.rights_decision_id "
            "WHERE a.admission_id=?" + event_clause,
            parameters,
        ).fetchone()
        if row is None:
            raise AuthorityPersistenceError(
                "committed admission lacks its immutable activation result"
            )
        return row

    def _blob_lifecycle_row(
        self, blob_digest: str, *, conn: sqlite3.Connection | None = None
    ) -> sqlite3.Row:
        selected = conn or self._connection
        row = selected.execute(
            "SELECT b.size_bytes,h.current_version,h.updated_at AS head_updated_at,v.* FROM blob_identities b "
            "JOIN blob_lifecycle_heads h ON h.blob_digest=b.blob_digest "
            "JOIN blob_lifecycle_versions v "
            "ON v.blob_digest=h.blob_digest "
            "AND v.lifecycle_version=h.current_version "
            "WHERE b.blob_digest=?",
            (blob_digest,),
        ).fetchone()
        if row is None:
            raise KeyError(blob_digest)
        if self._current_state_only:
            latest = selected.execute(
                "SELECT MAX(lifecycle_version) FROM blob_lifecycle_versions WHERE blob_digest=?", (blob_digest,),
            ).fetchone()[0]
            if row["current_version"] != latest or row["head_updated_at"] != row["recorded_at"]:
                raise AuthorityPersistenceError("current blob lifecycle head is not the exact latest version")
            detail = {
                "state": row["state"], "integrity_state": row["integrity_state"],
                "operation_id": row["operation_id"], "event_id": row["event_id"],
            }
            if row["event_id"] is None:
                expected = {"STAGING": "UNVERIFIED", "INSTALLED": "VERIFIED"}
                if expected.get(str(row["state"])) != row["integrity_state"]:
                    raise AuthorityPersistenceError("pending blob lifecycle state differs")
                detail["blob"] = {"blob_digest": blob_digest, "size_bytes": row["size_bytes"]}
            else:
                self._validate_retained_event(str(row["event_id"]))
                event = selected.execute(
                    "SELECT e.*,p.payload_bytes FROM ledger_events e JOIN authority_payloads p "
                    "ON p.payload_id=e.payload_id WHERE e.event_id=?", (row["event_id"],),
                ).fetchone()
                payload = self._decode_canonical_object(bytes(event["payload_bytes"]))
                expected = {
                    "governed_object.admission.activated": ("ACTIVE", "VERIFIED", "admission_id"),
                    "governed_blob.deletion.tombstoned": ("DELETION_PENDING", "VERIFIED", "deletion_id"),
                    "governed_blob.deletion.completed": ("DELETED", "MISSING", "deletion_id"),
                    "governed_blob.orphan.removed": ("DELETED", "MISSING", "operation_id"),
                }.get(str(event["event_type"]))
                if expected is None or (row["state"], row["integrity_state"]) != expected[:2] \
                        or payload.get(expected[2]) != event["aggregate_id"] or payload.get("blob_digest") != blob_digest \
                        or payload.get("operation_id") != row["operation_id"] or event["recorded_at"] != row["recorded_at"]:
                    raise AuthorityPersistenceError("current blob lifecycle differs from exact event")
                if expected[0] == "ACTIVE":
                    detail["blob"] = {"blob_digest": blob_digest, "size_bytes": row["size_bytes"]}
                else:
                    detail["blob_digest"] = blob_digest
            if digest_canonical(detail) != row["detail_digest"]:
                raise AuthorityPersistenceError("current blob lifecycle detail differs")
        return row

    @staticmethod
    def _require_rights_current(row: sqlite3.Row, now: UtcTimestamp) -> None:
        if not bool(row["rights_allowed"]):
            raise ObjectAdmissionDenied(str(row["reason_code"]))
        decided_at = UtcTimestamp.parse(str(row["decided_at"]))
        valid_from = UtcTimestamp.parse(str(row["rights_valid_from"]))
        valid_until = (
            None
            if row["rights_valid_until"] is None
            else UtcTimestamp.parse(str(row["rights_valid_until"]))
        )
        if now.value < decided_at.value or now.value < valid_from.value:
            raise ObjectAdmissionDenied("RIGHTS_NOT_YET_VALID")
        if valid_until is not None and now.value >= valid_until.value:
            raise ObjectAdmissionDenied("RIGHTS_EXPIRED")

    def _current_admission_row(
        self,
        conn: sqlite3.Connection,
        admission_id: str,
        *,
        now: UtcTimestamp,
        require_active: bool = True,
        require_bytes: bool = True,
    ) -> sqlite3.Row:
        row = self._admission_row(admission_id, conn=conn)
        if require_active and str(row["state"]) != AdmissionState.ACTIVE.value:
            raise ObjectAdmissionDenied("object admission is not ACTIVE")
        if self._current_state_only:
            self._prove_current_admission(conn, row)
        valid_from = UtcTimestamp.parse(str(row["valid_from"]))
        valid_until = (
            None
            if row["valid_until"] is None
            else UtcTimestamp.parse(str(row["valid_until"]))
        )
        if now.value < valid_from.value:
            raise ObjectAdmissionDenied("object admission is not yet valid")
        if valid_until is not None and now.value >= valid_until.value:
            raise ObjectAdmissionDenied("object admission has expired")
        self._require_rights_current(row, now)
        deletion = self._active_deletion_for_blob(
            conn, str(row["blob_digest"])
        )
        if deletion is not None:
            state = DeletionState(str(deletion["state"]))
            if state in {
                DeletionState.TOMBSTONED,
                DeletionState.PHYSICALLY_REMOVED,
            }:
                raise ObjectAdmissionDenied(
                    "governed tombstone blocks object authority"
                )
        lifecycle = self._blob_lifecycle_row(
            str(row["blob_digest"]), conn=conn
        )
        if require_bytes and (
            str(lifecycle["state"]) != BlobLifecycleState.ACTIVE.value
            or str(lifecycle["integrity_state"])
            != BlobIntegrityState.VERIFIED.value
        ):
            raise ObjectAdmissionDenied("blob bytes are not ACTIVE and verified")
        return row

    def resolve(self, admission_id: object) -> ObjectAdmissionDescriptor:
        """Private CommandService lookup adapter; returned data is non-authoritative."""

        if not isinstance(admission_id, ObjectAdmissionId):
            raise TypeError("admission lookup requires ObjectAdmissionId")
        return self.resolve_admission(admission_id)

    def resolve_admission(
        self, admission_id: ObjectAdmissionId
    ) -> ObjectAdmissionDescriptor:
        if not isinstance(admission_id, ObjectAdmissionId):
            raise TypeError("admission lookup requires ObjectAdmissionId")
        now = self._clock()
        with self._lock:
            # This lookup happens before the exact command authorization
            # decision.  Resolve only SQLite policy metadata here; do not hash or
            # read governed bytes until the authorized store commit guard pins
            # and revalidates the exact object.
            row = self._current_admission_row(
                self._connection,
                str(admission_id),
                now=now,
                require_active=True,
                require_bytes=False,
            )
            identity = BlobIdentity(
                str(row["blob_digest"]), int(row["size_bytes"])
            )
            return ObjectAdmissionDescriptor(
                admission_id=admission_id,
                blob_digest=identity.blob_digest,
                object_class=str(row["object_class"]),
                allowed_use=str(row["allowed_use"]),
                security_scope=str(row["security_scope"]),
                retention_scope=str(row["retention_scope"]),
                active=True,
            )

    def _object_payload_commit_guard(
        self, conn: sqlite3.Connection, grant: _AuthorizedCommandGrant
    ) -> Iterator[_PinnedBlob]:
        if grant.payload.kind != PayloadMode.OBJECT_ADMISSION.value:
            return super()._object_payload_commit_guard(conn, grant)
        if grant.payload.object_admission_id is None:
            raise UnsupportedPayloadMode(
                "object payload requires admission identity"
            )
        now = self._clock()
        row = self._current_admission_row(
            conn,
            str(grant.payload.object_admission_id),
            now=now,
            require_active=True,
            require_bytes=True,
        )
        expected = (
            str(row["object_class"]) == grant.definition.required_object_class
            and str(row["allowed_use"])
            == grant.definition.required_allowed_use
            and str(row["security_scope"]) == grant.definition.security_scope
            and str(row["retention_scope"])
            == grant.definition.retention_scope
            and str(row["blob_digest"]) == grant.payload.blob_digest
            and str(row["blob_digest"]) == grant.payload.digest
        )
        if not expected:
            raise ObjectAdmissionDenied(
                "object admission semantics do not match command definition"
            )
        identity = BlobIdentity(
            str(row["blob_digest"]), int(row["size_bytes"])
        )
        pinned = self._cas.pin(identity)

        @contextmanager
        def guard() -> Iterator[_PinnedBlob]:
            try:
                yield pinned
            finally:
                pinned.close()

        return guard()

    def _final_object_payload_commit_check(
        self,
        conn: sqlite3.Connection,
        grant: _AuthorizedCommandGrant,
        pinned: _PinnedBlob,
    ) -> None:
        # Re-resolve every mutable authority record inside the command write
        # transaction.  The earlier descriptor is deliberately ignored.
        now = self._clock()
        row = self._current_admission_row(
            conn,
            str(grant.payload.object_admission_id),
            now=now,
            require_active=True,
            require_bytes=True,
        )
        if (
            str(row["blob_digest"]) != pinned.identity.blob_digest
            or int(row["size_bytes"]) != pinned.identity.size_bytes
        ):
            raise ObjectAdmissionDenied(
                "object authority changed while command was committing"
            )
        self._cas.verify_pinned(pinned)
        # Authentication is also time-bounded authority.  A large governed
        # object can take long enough to hash that the context expires between
        # authorization and the final exact-byte check.  Revalidate it after
        # the potentially expensive rehash, inside the same write transaction.
        final_now = self._clock()
        grant.authentication.require_current(final_now)
        # Recheck time-bounded rights after the potentially expensive final
        # rehash.  The admission can expire while bytes are being verified; no
        # event/reference may commit from a pre-expiry snapshot.
        final_row = self._current_admission_row(
            conn,
            str(grant.payload.object_admission_id),
            now=final_now,
            require_active=True,
            require_bytes=True,
        )
        if (
            str(final_row["blob_digest"]) != pinned.identity.blob_digest
            or int(final_row["size_bytes"]) != pinned.identity.size_bytes
            or str(final_row["object_class"])
            != grant.definition.required_object_class
            or str(final_row["allowed_use"])
            != grant.definition.required_allowed_use
            or str(final_row["security_scope"])
            != grant.definition.security_scope
            or str(final_row["retention_scope"])
            != grant.definition.retention_scope
        ):
            raise ObjectAdmissionDenied(
                "object admission changed during final command verification"
            )

    @staticmethod
    def _validate_object_admission_payload_record(
        conn: sqlite3.Connection, row: sqlite3.Row
    ) -> None:
        if row["payload_bytes"] is not None:
            raise AuthorityPersistenceError(
                "object admission payload cannot embed bytes"
            )
        if row["object_admission_id"] is None:
            raise AuthorityPersistenceError(
                "object admission payload lacks admission identity"
            )
        admission_id = ObjectAdmissionId.parse(
            str(row["object_admission_id"])
        )
        validate_sha256_digest(
            str(row["payload_digest"]), field="object_payload_digest"
        )
        admission = conn.execute(
            "SELECT blob_digest FROM object_admissions WHERE admission_id=?",
            (str(admission_id),),
        ).fetchone()
        if admission is None:
            raise AuthorityPersistenceError(
                "object admission payload identity does not resolve"
            )
        if str(admission["blob_digest"]) != str(row["payload_digest"]):
            raise AuthorityPersistenceError(
                "object payload digest differs from admitted blob"
            )
        contract = conn.execute(
            "SELECT schema_version,payload_mode,contract_version,"
            "canonicalizer_implementation_version "
            "FROM payload_schema_contracts WHERE contract_digest=?",
            (str(row["schema_contract_digest"]),),
        ).fetchone()
        if contract is None:
            raise AuthorityPersistenceError(
                "object payload schema contract is missing"
            )
        if (
            str(contract["schema_version"]) != str(row["schema_version"])
            or str(contract["payload_mode"])
            != PayloadMode.OBJECT_ADMISSION.value
            or str(contract["contract_version"])
            != str(row["schema_contract_version"])
            or str(contract["canonicalizer_implementation_version"])
            != str(row["canonicalizer_implementation_version"])
        ):
            raise AuthorityPersistenceError(
                "object payload does not match its immutable schema contract"
            )

    def _active_deletion_for_blob(
        self, conn: sqlite3.Connection, blob_digest: str
    ) -> sqlite3.Row | None:
        row = conn.execute(
            "SELECT d.deletion_id,h.current_version,v.state,v.recorded_at "
            "FROM object_deletions d "
            "JOIN object_deletion_heads h ON h.deletion_id=d.deletion_id "
            "JOIN object_deletion_versions v "
            "ON v.deletion_id=h.deletion_id "
            "AND v.lifecycle_version=h.current_version "
            "WHERE d.blob_digest=? "
            + ("AND v.state!='PHYSICALLY_REMOVED' " if not self._current_state_only else "") +
            "ORDER BY d.created_at DESC LIMIT 1",
            (blob_digest,),
        ).fetchone()
        if row is not None and self._current_state_only:
            self._prove_deletion_lifecycle(conn, row)
        return row

    def _prove_deletion_lifecycle(self, conn: sqlite3.Connection, row: sqlite3.Row) -> None:
        version = conn.execute(
            "SELECT d.blob_digest,d.reason_code,h.updated_at AS head_updated_at,v.* FROM object_deletions d "
            "JOIN object_deletion_heads h ON h.deletion_id=d.deletion_id "
            "JOIN object_deletion_versions v ON v.deletion_id=h.deletion_id AND v.lifecycle_version=h.current_version "
            "WHERE d.deletion_id=? AND v.lifecycle_version="
            "(SELECT MAX(lifecycle_version) FROM object_deletion_versions WHERE deletion_id=d.deletion_id)",
            (row["deletion_id"],),
        ).fetchone()
        if version is None or version["lifecycle_version"] != row["current_version"] or version["state"] != row["state"]:
            raise AuthorityPersistenceError("current deletion head is not the exact latest version")
        event, payload = self._object_lifecycle_event(conn, str(version["event_id"]), str(row["deletion_id"]))
        states = {
            "governed_blob.deletion.requested": "REQUESTED", "governed_blob.deletion.tombstoned": "TOMBSTONED",
            "governed_blob.deletion.completed": "PHYSICALLY_REMOVED", "governed_blob.deletion.failed": "TOMBSTONED",
        }
        expected = {"operation_id": version["operation_id"], "deletion_id": version["deletion_id"], "blob_digest": version["blob_digest"]}
        if event["event_type"] in {"governed_blob.deletion.requested", "governed_blob.deletion.tombstoned"}:
            expected["reason_code"] = version["reason_code"]
        elif event["event_type"] == "governed_blob.deletion.failed":
            expected["error_code"] = version["error_code"]
        if (
            states.get(str(event["event_type"])) != version["state"] or payload != expected
            or event["aggregate_version"] != version["lifecycle_version"]
            or event["recorded_at"] != version["recorded_at"] or version["head_updated_at"] != version["recorded_at"]
            or digest_canonical(payload) != version["detail_digest"]
        ):
            raise AuthorityPersistenceError("current deletion lifecycle differs from exact event")

    def _admission_view_from_row(self, row: sqlite3.Row) -> ObjectAdmissionView:
        activation_event = (
            None
            if row["event_id"] is None
            else EventId.parse(str(row["event_id"]))
        )
        state = AdmissionState(str(row["state"]))
        return ObjectAdmissionView(
            admission_id=ObjectAdmissionId.parse(str(row["admission_id"])),
            admission_type=str(row["admission_type"]),
            definition_version=str(row["definition_version"]),
            definition_digest=str(row["definition_digest"]),
            blob=BlobIdentity(
                str(row["blob_digest"]), int(row["size_bytes"])
            ),
            object_class=str(row["object_class"]),
            allowed_use=str(row["allowed_use"]),
            security_scope=str(row["security_scope"]),
            retention_scope=str(row["retention_scope"]),
            rights_decision_id=RightsDecisionId.parse(
                str(row["rights_decision_id"])
            ),
            rights_decision_digest=str(row["rights_digest"]),
            valid_from=UtcTimestamp.parse(str(row["valid_from"])),
            valid_until=(
                None
                if row["valid_until"] is None
                else UtcTimestamp.parse(str(row["valid_until"]))
            ),
            lifecycle_version=int(row["current_version"]),
            state=state,
            activation_event_id=activation_event
        )

    def admission_view(self, admission_id: ObjectAdmissionId) -> ObjectAdmissionView:
        if not isinstance(admission_id, ObjectAdmissionId):
            raise TypeError("admission view requires ObjectAdmissionId")
        with self._lock:
            return self._admission_view_from_row(
                self._admission_row(str(admission_id))
            )

    def rights_view(
        self, rights_decision_id: RightsDecisionId
    ) -> RightsDecisionView:
        if not isinstance(rights_decision_id, RightsDecisionId):
            raise TypeError("rights view requires RightsDecisionId")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM object_rights_decisions WHERE rights_decision_id=?",
                (str(rights_decision_id),),
            ).fetchone()
            if row is None:
                raise KeyError(str(rights_decision_id))
            canonical = bytes(row["canonical_bytes"])
            digest = str(row["canonical_digest"])
            if digest_bytes(canonical) != digest:
                raise AuthorityPersistenceError(
                    "stored rights decision digest mismatch"
                )
            value = self._decode_canonical_object(canonical)
            if not isinstance(value, dict):
                raise AuthorityPersistenceError(
                    "stored rights decision is not an object"
                )
            return RightsDecisionView(
                rights_decision_id=rights_decision_id,
                authentication_context_id=str(row["authentication_context_id"]),
                authorization_request_digest=str(
                    row["authorization_request_digest"]
                ),
                authorization_decision_id=str(row["authorization_decision_id"]),
                policy_contract_digest=str(row["policy_contract_digest"]),
                admission_definition_digest=str(
                    row["admission_definition_digest"]
                ),
                blob=BlobIdentity(
                    str(row["blob_digest"]), int(row["size_bytes"])
                ),
                object_class=str(row["object_class"]),
                allowed_use=str(row["allowed_use"]),
                security_scope=str(row["security_scope"]),
                retention_scope=str(row["retention_scope"]),
                allowed=bool(row["allowed"]),
                reason_code=str(row["reason_code"]),
                decided_at=UtcTimestamp.parse(str(row["decided_at"])),
                valid_from=UtcTimestamp.parse(str(row["valid_from"])),
                valid_until=(
                    None
                    if row["valid_until"] is None
                    else UtcTimestamp.parse(str(row["valid_until"]))
                ),
                canonical_digest=digest,
            )

    def blob_view(self, blob_digest: str) -> BlobLifecycleView:
        validate_sha256_digest(blob_digest, field="blob_digest")
        with self._lock:
            row = self._blob_lifecycle_row(blob_digest)
            return BlobLifecycleView(
                blob=BlobIdentity(blob_digest, int(row["size_bytes"])),
                lifecycle_version=int(row["current_version"]),
                state=BlobLifecycleState(str(row["state"])),
                integrity_state=BlobIntegrityState(
                    str(row["integrity_state"])
                ),
                recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
                event_id=(
                    None
                    if row["event_id"] is None
                    else EventId.parse(str(row["event_id"]))
                ),
            )

    def object_liveness(
        self, conn: sqlite3.Connection, blob_digest: str
    ) -> ObjectLivenessSnapshot:
        row = conn.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM object_admissions a "
            "JOIN object_admission_heads h ON h.admission_id=a.admission_id "
            "JOIN object_admission_versions v "
            "ON v.admission_id=h.admission_id "
            "AND v.lifecycle_version=h.current_version "
            "WHERE a.blob_digest=? AND v.state='ACTIVE'),"
            "(SELECT COUNT(*) FROM object_admissions a "
            "JOIN object_admission_heads h ON h.admission_id=a.admission_id "
            "JOIN object_admission_versions v "
            "ON v.admission_id=h.admission_id "
            "AND v.lifecycle_version=h.current_version "
            "WHERE a.blob_digest=? AND v.state='PENDING'),"
            "(SELECT COUNT(*) FROM authority_payloads "
            "WHERE payload_digest=? AND object_admission_id IS NOT NULL),"
            "(SELECT COUNT(*) FROM object_recovery_pins p "
            "JOIN object_recovery_pin_heads h ON h.pin_id=p.pin_id "
            "JOIN object_recovery_pin_versions v "
            "ON v.pin_id=h.pin_id AND v.lifecycle_version=h.current_version "
            "WHERE p.blob_digest=? AND v.state='ACTIVE')",
            (blob_digest, blob_digest, blob_digest, blob_digest),
        ).fetchone()
        deletion = self._active_deletion_for_blob(conn, blob_digest)
        return ObjectLivenessSnapshot(
            active_admissions=int(row[0]),
            pending_admissions=int(row[1]),
            authority_references=int(row[2]),
            active_recovery_pins=int(row[3]),
            deletion_state=(
                None
                if deletion is None
                else DeletionState(str(deletion["state"]))
            ),
        )

    def _append_blob_lifecycle(
        self,
        conn: sqlite3.Connection,
        *,
        blob_digest: str,
        state: BlobLifecycleState,
        integrity: BlobIntegrityState,
        operation_id: str,
        event_id: str | None,
        recorded_at: UtcTimestamp,
    ) -> int:
        head = conn.execute(
            "SELECT current_version FROM blob_lifecycle_heads WHERE blob_digest=?",
            (blob_digest,),
        ).fetchone()
        if head is None:
            raise KeyError(blob_digest)
        next_version = int(head["current_version"]) + 1
        detail = {
            "blob_digest": blob_digest,
            "lifecycle_version": next_version,
            "state": state.value,
            "integrity_state": integrity.value,
            "operation_id": operation_id,
            "event_id": event_id,
        }
        conn.execute(
            "INSERT INTO blob_lifecycle_versions("
            "blob_digest,lifecycle_version,state,integrity_state,"
            "operation_id,event_id,recorded_at,detail_digest) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                blob_digest,
                next_version,
                state.value,
                integrity.value,
                operation_id,
                event_id,
                recorded_at.to_text(),
                digest_canonical(detail),
            ),
        )
        conn.execute(
            "UPDATE blob_lifecycle_heads SET current_version=?,updated_at=? "
            "WHERE blob_digest=?",
            (next_version, recorded_at.to_text(), blob_digest),
        )
        return next_version

    def _validate_relational_invariants(self, conn: sqlite3.Connection) -> None:
        super()._validate_relational_invariants(conn)
        missing = conn.execute(
            "SELECT a.admission_id FROM object_admission_heads h "
            "JOIN object_admissions a ON a.admission_id=h.admission_id "
            "LEFT JOIN object_admission_versions v "
            "ON v.admission_id=h.admission_id "
            "AND v.lifecycle_version=h.current_version "
            "WHERE v.admission_id IS NULL LIMIT 1"
        ).fetchone()
        if missing is not None:
            raise AuthorityPersistenceError(
                "object admission head does not resolve exact version"
            )
        active_without_event = conn.execute(
            "SELECT v.admission_id FROM object_admission_versions v "
            "WHERE v.state='ACTIVE' AND v.event_id IS NULL LIMIT 1"
        ).fetchone()
        if active_without_event is not None:
            raise AuthorityPersistenceError(
                "ACTIVE object admission lacks committed activation event"
            )
        blob_missing = conn.execute(
            "SELECT h.blob_digest FROM blob_lifecycle_heads h "
            "LEFT JOIN blob_lifecycle_versions v "
            "ON v.blob_digest=h.blob_digest "
            "AND v.lifecycle_version=h.current_version "
            "WHERE v.blob_digest IS NULL LIMIT 1"
        ).fetchone()
        if blob_missing is not None:
            raise AuthorityPersistenceError(
                "blob lifecycle head does not resolve exact version"
            )

    def _validate_immutable_records(self, conn: sqlite3.Connection) -> None:
        super()._validate_immutable_records(conn)
        for table, bytes_column, digest_column in (
            (
                "rights_policy_contracts",
                "canonical_bytes",
                "contract_digest",
            ),
            (
                "hydration_policy_contracts",
                "canonical_bytes",
                "contract_digest",
            ),
            (
                "object_admission_definitions",
                "canonical_bytes",
                "definition_digest",
            ),
            (
                "object_admission_preflights",
                "canonical_bytes",
                "canonical_digest",
            ),
            (
                "object_rights_decisions",
                "canonical_bytes",
                "canonical_digest",
            ),
            (
                "object_access_decisions",
                "canonical_bytes",
                "canonical_digest",
            ),
        ):
            for row in conn.execute(
                f"SELECT {bytes_column},{digest_column} FROM {table}"
            ):
                data = bytes(row[bytes_column])
                self._decode_canonical_object(data)
                expected = str(row[digest_column])
                # Contract/definition primary identities are digests over the
                # canonical contract bytes. Other records retain an independent
                # canonical_digest column. Decoding already proved these exact
                # bytes equal the canonical serialisation.
                if digest_bytes(data) != expected:
                    raise AuthorityPersistenceError(
                        f"immutable {table} canonical digest mismatch"
                    )

    def _validate_registry_coverage(self, conn: sqlite3.Connection) -> None:
        super()._validate_registry_coverage(conn)
        # Every historical contract retained by SQLite must remain resolvable by
        # the runtime registries for the full idempotency/audit horizon.  Merely
        # persisting canonical bytes is not enough: replay and provenance paths
        # execute the exact versioned policy/definition contract.
        for row in conn.execute(
            "SELECT admission_type,definition_version,definition_digest "
            "FROM object_admission_definitions"
        ).fetchall():
            self._admission_registry.resolve_exact(
                str(row["admission_type"]),
                str(row["definition_version"]),
                str(row["definition_digest"]),
            )
        for row in conn.execute(
            "SELECT policy_key,contract_version,contract_digest "
            "FROM rights_policy_contracts"
        ).fetchall():
            self._rights_policies.resolve_exact(
                str(row["policy_key"]),
                str(row["contract_version"]),
                str(row["contract_digest"]),
            )
        for row in conn.execute(
            "SELECT policy_id,contract_version,contract_digest "
            "FROM hydration_policy_contracts"
        ).fetchall():
            self._hydration_policies.resolve_exact(
                str(row["policy_id"]),
                str(row["contract_version"]),
                str(row["contract_digest"]),
            )

        for definition in self._admission_registry.definitions():
            row = conn.execute(
                "SELECT canonical_bytes FROM object_admission_definitions "
                "WHERE definition_digest=?",
                (definition.digest,),
            ).fetchone()
            if row is None or bytes(row["canonical_bytes"]) != canonical_json_bytes(
                definition.canonical_value()
            ):
                raise AuthorityPersistenceError(
                    "retained admission definition is missing from authority"
                )
        for contract in self._rights_policies.contracts():
            row = conn.execute(
                "SELECT canonical_bytes FROM rights_policy_contracts "
                "WHERE contract_digest=?",
                (contract.contract_digest,),
            ).fetchone()
            if row is None or bytes(row["canonical_bytes"]) != canonical_json_bytes(
                contract.canonical_value()
            ):
                raise AuthorityPersistenceError(
                    "retained rights policy is missing from authority"
                )
        for contract in self._hydration_policies.contracts():
            row = conn.execute(
                "SELECT canonical_bytes FROM hydration_policy_contracts "
                "WHERE contract_digest=?",
                (contract.contract_digest,),
            ).fetchone()
            if row is None or bytes(row["canonical_bytes"]) != canonical_json_bytes(
                contract.canonical_value()
            ):
                raise AuthorityPersistenceError(
                    "retained hydration policy is missing from authority"
                )

    def _known_installed_digests(self) -> frozenset[str]:
        return frozenset(
            str(row["blob_digest"])
            for row in self._connection.execute(
                "SELECT b.blob_digest FROM blob_identities b "
                "JOIN blob_lifecycle_heads h ON h.blob_digest=b.blob_digest "
                "JOIN blob_lifecycle_versions v "
                "ON v.blob_digest=h.blob_digest "
                "AND v.lifecycle_version=h.current_version "
                "WHERE v.state IN ('INSTALLED','ACTIVE','DELETION_PENDING')"
            ).fetchall()
        )

    def _reconcile_objects(self) -> None:
        with self._lock:
            self._persist_object_contracts(
                self._connection, recorded_at=self._clock().to_text()
            )
            self._reconcile_staging_records()
            # An install can complete durably before SQLite commit.  Such a file
            # has no authoritative blob identity and is safe to remove.
            if not self._current_state_only:
                self._cas.cleanup_unreferenced_installed(
                    known_digests=frozenset(
                        str(row["blob_digest"])
                        for row in self._connection.execute(
                            "SELECT blob_digest FROM blob_identities"
                        ).fetchall()
                    )
                )
            self._reconcile_blob_integrity()
            self._reconcile_expired_rights()
            self._reconcile_deletions()

    def _reconcile_staging_records(self) -> None:
        records = self._connection.execute(
            "SELECT stage_id,staged_name,state FROM object_staging_records"
            + (" WHERE state='STAGED'" if self._current_state_only else "")
        ).fetchall()
        for row in records:
            state = str(row["state"])
            path = self._cas.staging_root / str(row["staged_name"])
            if state in {"COMMITTED", "FAILED", "CLEANED"} and path.exists():
                path.unlink()
                self._cas._fsync_directory(self._cas.staging_root)
                if state != "COMMITTED":
                    self._connection.execute(
                        "UPDATE object_staging_records "
                        "SET state='CLEANED',updated_at=? WHERE stage_id=?",
                        (self._clock().to_text(), str(row["stage_id"])),
                    )
            elif state == "STAGED" and not path.exists():
                self._connection.execute(
                    "UPDATE object_staging_records "
                    "SET state='FAILED',failure_code='MISSING_STAGED_BYTES',updated_at=? "
                    "WHERE stage_id=?",
                    (self._clock().to_text(), str(row["stage_id"])),
                )
        keep = frozenset(
            str(row["staged_name"])
            for row in self._connection.execute(
                "SELECT staged_name FROM object_staging_records WHERE state='STAGED'"
            ).fetchall()
        )
        if not self._current_state_only:
            self._cas.cleanup_staging(keep_names=keep)

    def _reconcile_blob_integrity(self, *, _include_historical_active: bool = False) -> None:
        # ACTIVE is retained admission state, not proof that an old input is
        # part of today's work. Rehydrate pins/rehashes it before use. Native
        # boot only reconciles pending install/deletion byte obligations.
        states = "'INSTALLED','DELETION_PENDING'"
        if _include_historical_active:
            states = "'INSTALLED','ACTIVE','DELETION_PENDING'"
        rows = self._connection.execute(
            "SELECT b.blob_digest,b.size_bytes,v.state,v.integrity_state "
            "FROM blob_identities b "
            "JOIN blob_lifecycle_heads h ON h.blob_digest=b.blob_digest "
            "JOIN blob_lifecycle_versions v "
            "ON v.blob_digest=h.blob_digest "
            "AND v.lifecycle_version=h.current_version"
            + (f" WHERE v.state IN ({states})"
               if self._current_state_only else "")
        ).fetchall()
        for row in rows:
            state = BlobLifecycleState(str(row["state"]))
            identity = BlobIdentity(
                str(row["blob_digest"]), int(row["size_bytes"])
            )
            if state in {
                BlobLifecycleState.INSTALLED,
                BlobLifecycleState.ACTIVE,
                BlobLifecycleState.DELETION_PENDING,
            }:
                try:
                    with self._cas.pin(identity):
                        pass
                except ObjectIntegrityError as exc:
                    raise AuthorityPersistenceError(
                        "active authoritative blob bytes are missing or corrupt"
                    ) from exc
            if state is BlobLifecycleState.DELETED and self._cas.object_path(
                identity
            ).exists():
                # SQLite tombstone authority wins.  Bytes can reappear after a
                # crash, restore, or manual filesystem mutation; remove them
                # deterministically instead of allowing content resurrection.
                try:
                    self._cas.unlink(identity)
                except Exception as exc:
                    raise AuthorityPersistenceError(
                        "deleted blob bytes could not be reconciled"
                    ) from exc

    def _reconcile_expired_rights(self) -> None:
        now = self._clock().to_text()
        if not self._current_state_only:
            invalid_active = self._connection.execute(
                "SELECT a.admission_id FROM object_admissions a "
                "JOIN object_admission_heads h ON h.admission_id=a.admission_id "
                "JOIN object_admission_versions v "
                "ON v.admission_id=h.admission_id "
                "AND v.lifecycle_version=h.current_version "
                "JOIN object_rights_decisions r "
                "ON r.rights_decision_id=a.rights_decision_id "
                "WHERE v.state='ACTIVE' AND r.allowed=0 LIMIT 1"
            ).fetchone()
            if invalid_active is not None:
                raise AuthorityPersistenceError(
                    "ACTIVE admission references a denying rights decision"
                )

        pending = self._connection.execute(
            "SELECT a.admission_id FROM object_admissions a "
            "JOIN object_admission_heads h ON h.admission_id=a.admission_id "
            "JOIN object_admission_versions v "
            "ON v.admission_id=h.admission_id "
            "AND v.lifecycle_version=h.current_version "
            "JOIN object_rights_decisions r "
            "ON r.rights_decision_id=a.rights_decision_id "
            "WHERE v.state='PENDING' "
            "AND (r.allowed=0 OR r.valid_until IS NOT NULL AND r.valid_until<=?) "
            "LIMIT 1",
            (now,),
        ).fetchone()
        if pending is not None:
            raise AuthorityPersistenceError(
                "pending admission has denied or expired rights"
            )
        # An admission that was validly activated can naturally pass its
        # rights/admission validity deadline while the process is offline.  The
        # immutable activation remains history, while every current-use path
        # (_current_admission_row) rejects the expired authority.  Refusing to
        # start the whole ledger would turn a normal expiry into an operational
        # outage without adding any safety.

    def _reconcile_deletions(self) -> None:
        rows = self._connection.execute(
            "SELECT d.deletion_id,d.blob_digest,h.current_version,v.state "
            "FROM object_deletions d "
            "JOIN object_deletion_heads h ON h.deletion_id=d.deletion_id "
            "JOIN object_deletion_versions v "
            "ON v.deletion_id=h.deletion_id "
            "AND v.lifecycle_version=h.current_version"
            + (" WHERE v.state!='PHYSICALLY_REMOVED'" if self._current_state_only else "")
        ).fetchall()
        for row in rows:
            state = DeletionState(str(row["state"]))
            digest = str(row["blob_digest"])
            try:
                lifecycle = self._blob_lifecycle_row(digest)
            except KeyError as exc:
                raise AuthorityPersistenceError(
                    "governed deletion references an unknown blob"
                ) from exc
            identity = BlobIdentity(digest, int(lifecycle["size_bytes"]))
            path = self._cas.object_path(identity)
            pin_count = int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM object_recovery_pins p "
                    "JOIN object_recovery_pin_heads h ON h.pin_id=p.pin_id "
                    "JOIN object_recovery_pin_versions v "
                    "ON v.pin_id=h.pin_id AND v.lifecycle_version=h.current_version "
                    "WHERE p.blob_digest=? AND v.state='ACTIVE'",
                    (digest,),
                ).fetchone()[0]
            )

            if state is DeletionState.PHYSICALLY_REMOVED:
                # SQLite tombstone authority wins over stray filesystem bytes.
                # Removing a reappeared file is deterministic reconciliation,
                # not a new editorial or lifecycle decision.
                if path.exists():
                    self._cas.unlink(identity)
                continue

            if state in {DeletionState.REQUESTED, DeletionState.FAILED}:
                try:
                    with self._cas.pin(identity):
                        pass
                except ObjectIntegrityError as exc:
                    raise AuthorityPersistenceError(
                        "pre-tombstone governed deletion has missing or corrupt bytes"
                    ) from exc
                continue

            if state is DeletionState.TOMBSTONED and pin_count:
                try:
                    with self._cas.pin(identity):
                        pass
                except ObjectIntegrityError as exc:
                    raise AuthorityPersistenceError(
                        "recovery-pinned tombstone has missing or corrupt bytes"
                    ) from exc
                continue

            if state is DeletionState.TOMBSTONED:
                # Hydration is already blocked.  Bytes may still await an
                # authenticated completion operation, or an interrupted unlink
                # may already have removed them.  Startup never invents the
                # terminal lifecycle event; it only preserves non-resurrection.
                continue

            raise AuthorityPersistenceError(
                f"unsupported governed deletion state during reconciliation: {state.value}"
            )
