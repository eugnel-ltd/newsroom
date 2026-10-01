from __future__ import annotations

from contextlib import nullcontext
import json
import sqlite3
from typing import Any

from ._capability import _AuthorizedCommandGrant
from .authorization_request_storage_migrations import (
    AUTHORIZATION_REQUEST_INDEXED_FIELDS,
    AUTHORIZATION_REQUEST_STORAGE_MARKER,
)
from .canonical import canonical_json_bytes, digest_bytes, digest_canonical
from .command_bound_storage import (
    COMMAND_REQUEST_MARKER, COMMAND_RESULT_PREFIX, command_request_storage,
    command_result_bytes, compact_command_result, pending_command_request_bytes,
)
from .persistence import (
    AuthorityPersistenceError,
    CommittedCommand,
    ExpectedVersionConflict,
    IdempotencyConflict,
    PayloadId,
    UnknownCausation,
    UnsupportedPayloadMode,
)
from .service import IdempotencyIdentityConflict
from .types import AuditId, CommandId, EventId, PayloadMode


class _EventStoreCommitMixin:
    """Atomic command, payload, provenance, audit and event persistence."""

    def commit(self, grant: _AuthorizedCommandGrant) -> CommittedCommand:
        self._issuer.verify(grant)
        with self._lock, self._transaction() as conn:
            return self._commit_grant_in_transaction(
                conn, grant, recorded_at=self._clock().to_text()
            )

    def _object_payload_commit_guard(
        self, conn: sqlite3.Connection, grant: _AuthorizedCommandGrant
    ) -> Any:
        del conn
        if grant.payload.kind == PayloadMode.OBJECT_ADMISSION.value:
            raise UnsupportedPayloadMode(
                "object-admission payload persistence requires the A2b authority store"
            )
        return nullcontext(None)

    def _final_object_payload_commit_check(
        self,
        conn: sqlite3.Connection,
        grant: _AuthorizedCommandGrant,
        pinned: Any,
    ) -> None:
        del conn, grant, pinned

    def _commit_grant_in_transaction(
        self,
        conn: sqlite3.Connection,
        grant: _AuthorizedCommandGrant,
        *,
        recorded_at: str,
    ) -> CommittedCommand:
        """Commit an authorised command into an already-open authority transaction."""

        self._issuer.verify(grant)
        existing = conn.execute(
            "SELECT command_id,command_definition_version,"
            "command_definition_digest,stable_semantic_request_digest,"
            "result_digest,result_bytes FROM authority_commands "
            "WHERE idempotency_namespace=? AND idempotency_key=?",
            (grant.idempotency_namespace, grant.idempotency_key),
        ).fetchone()
        if existing is not None:
            return self._replay_existing(grant, existing)
        if grant.replay_of_command_id is not None:
            raise IdempotencyConflict(
                "command boundary expected a replay but no row exists"
            )

        try:
            payload_mode = PayloadMode(grant.payload.kind)
        except ValueError as exc:
            raise UnsupportedPayloadMode(
                f"unsupported authority payload mode: {grant.payload.kind}"
            ) from exc

        with self._object_payload_commit_guard(conn, grant) as pinned:
            command_id = str(CommandId.new())
            self._persist_schema_contract(conn, grant, recorded_at=recorded_at)
            self._persist_definition(conn, grant, recorded_at=recorded_at)
            self._persist_security(conn, grant, recorded_at=recorded_at, pending_command_id=command_id)
            self._validate_causation(conn, grant)
            current_version, new_version = self._resolve_version(conn, grant)

            object_admission_id: str | None = None
            payload_bytes = grant.payload.inline_bytes
            if payload_mode is PayloadMode.OBJECT_ADMISSION:
                if (
                    payload_bytes is not None
                    or grant.payload.object_admission_id is None
                    or grant.payload.blob_digest is None
                    or grant.payload.digest != grant.payload.blob_digest
                ):
                    raise AuthorityPersistenceError(
                        "object-admission payload is not a closed retained reference"
                    )
                object_admission_id = str(grant.payload.object_admission_id)
                if pinned is None:
                    raise AuthorityPersistenceError(
                        "object-admission payload lacks a pinned authority guard"
                    )
            else:
                if payload_bytes is None:
                    raise AuthorityPersistenceError(
                        "retained payload bytes are required"
                    )
                if digest_bytes(payload_bytes) != grant.payload.digest:
                    raise AuthorityPersistenceError(
                        "retained payload digest mismatch"
                    )

            event_id = str(EventId.new())
            audit_id = str(AuditId.new())
            payload_id = str(PayloadId.new())
            ledger_seq = int(
                conn.execute(
                    "SELECT COALESCE(MAX(ledger_seq),0)+1 FROM ledger_events"
                ).fetchone()[0]
            )
            result_value = {
                "command_id": command_id,
                "aggregate_type": grant.definition.aggregate_type,
                "aggregate_id": grant.aggregate_id,
                "aggregate_version": new_version,
                "ledger_seq": ledger_seq,
                "event_id": event_id,
            }
            result_bytes = canonical_json_bytes(result_value)
            result_digest = digest_bytes(result_bytes)
            stored_result = compact_command_result(
                {**result_value, "result_bytes": result_bytes, "result_digest": result_digest},
                event_row=result_value,
            )

            conn.execute(
                "INSERT INTO authority_payloads("
                "payload_id,mode,schema_version,schema_contract_version,"
                "schema_contract_digest,canonicalizer_implementation_version,"
                "payload_digest,payload_bytes,object_admission_id,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    payload_id,
                    grant.payload.kind,
                    grant.payload.schema_version,
                    grant.payload.schema_contract_version,
                    grant.payload.schema_contract_digest,
                    grant.payload.canonicalizer_version,
                    grant.payload.digest,
                    payload_bytes,
                    object_admission_id,
                    recorded_at,
                ),
            )
            conn.execute(
                "INSERT INTO authority_commands("
                "command_id,command_type,producer_version,"
                "command_definition_version,command_definition_digest,"
                "aggregate_type,aggregate_id,expected_aggregate_version,"
                "payload_id,idempotency_namespace,idempotency_key,"
                "stable_semantic_request_digest,authentication_context_id,"
                "authorization_request_digest,authorization_decision_id,"
                "result_digest,result_bytes,committed_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    command_id,
                    grant.command_type,
                    self._command_service_version,
                    grant.definition.definition_version,
                    grant.definition.digest,
                    grant.definition.aggregate_type,
                    grant.aggregate_id,
                    grant.expected_aggregate_version,
                    payload_id,
                    grant.idempotency_namespace,
                    grant.idempotency_key,
                    grant.stable_semantic_request_digest,
                    str(grant.authentication.authentication_context_id),
                    grant.authorization_request.request_digest,
                    str(grant.authorization.authorization_decision_id),
                    result_digest,
                    stored_result,
                    recorded_at,
                ),
            )

            if current_version is None:
                conn.execute(
                    "INSERT INTO authority_aggregates("
                    "aggregate_type,aggregate_id,current_version,created_at,updated_at) "
                    "VALUES(?,?,?,?,?)",
                    (
                        grant.definition.aggregate_type,
                        grant.aggregate_id,
                        new_version,
                        recorded_at,
                        recorded_at,
                    ),
                )
            else:
                conn.execute(
                    "UPDATE authority_aggregates SET current_version=?,updated_at=? "
                    "WHERE aggregate_type=? AND aggregate_id=?",
                    (
                        new_version,
                        recorded_at,
                        grant.definition.aggregate_type,
                        grant.aggregate_id,
                    ),
                )
            conn.execute(
                "INSERT INTO authority_aggregate_versions("
                "aggregate_type,aggregate_id,aggregate_version,command_id,"
                "payload_id,trust_scope,recorded_at) VALUES(?,?,?,?,?,?,?)",
                (
                    grant.definition.aggregate_type,
                    grant.aggregate_id,
                    new_version,
                    command_id,
                    payload_id,
                    grant.definition.trust_scope.value,
                    recorded_at,
                ),
            )
            conn.execute(
                "INSERT INTO authority_audit_events("
                "audit_id,command_id,authentication_context_id,"
                "authorization_request_digest,authorization_decision_id,"
                "event_type,detail_digest,recorded_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    audit_id,
                    command_id,
                    str(grant.authentication.authentication_context_id),
                    grant.authorization_request.request_digest,
                    str(grant.authorization.authorization_decision_id),
                    grant.definition.event_type,
                    digest_canonical(grant.unsigned_value()),
                    recorded_at,
                ),
            )
            conn.execute(
                "INSERT INTO ledger_events("
                "ledger_seq,event_id,event_type,event_schema_version,"
                "aggregate_type,aggregate_id,aggregate_version,recorded_at,"
                "command_id,producer_version,command_definition_version,"
                "command_definition_digest,payload_id,payload_mode,"
                "payload_schema_version,payload_schema_contract_version,"
                "payload_schema_contract_digest,payload_canonicalizer_version,"
                "payload_digest,object_admission_id,principal_id,"
                "authentication_context_id,authorization_request_digest,"
                "authorization_decision_id,correlation_id,causation_kind,"
                "causation_identifier,causation_external_system,security_scope,"
                "retention_scope,trust_scope) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ledger_seq,
                    event_id,
                    grant.definition.event_type,
                    grant.definition.event_schema_version,
                    grant.definition.aggregate_type,
                    grant.aggregate_id,
                    new_version,
                    recorded_at,
                    command_id,
                    self._command_service_version,
                    grant.definition.definition_version,
                    grant.definition.digest,
                    payload_id,
                    grant.payload.kind,
                    grant.payload.schema_version,
                    grant.payload.schema_contract_version,
                    grant.payload.schema_contract_digest,
                    grant.payload.canonicalizer_version,
                    grant.payload.digest,
                    object_admission_id,
                    grant.authentication.principal_id,
                    str(grant.authentication.authentication_context_id),
                    grant.authorization_request.request_digest,
                    str(grant.authorization.authorization_decision_id),
                    grant.correlation_id,
                    grant.causation_kind,
                    grant.causation_identifier,
                    grant.causation_external_system,
                    grant.definition.security_scope,
                    grant.definition.retention_scope,
                    grant.definition.trust_scope.value,
                ),
            )
            # Pending compact security is authenticated against real persisted
            # backing only after the command/event FK closure exists.
            request_row = conn.execute(
                "SELECT * FROM authorization_requests WHERE request_digest=?",
                (grant.authorization_request.request_digest,),
            ).fetchone()
            self._request_record_from_row(request_row, connection=conn)
            self._decode_result(stored_result, result_digest, replayed=False, connection=conn, command_id=command_id)
            if payload_mode is PayloadMode.OBJECT_ADMISSION:
                self._final_object_payload_commit_check(
                    conn, grant, pinned
                )
            return CommittedCommand(
                command_id=command_id,
                aggregate_type=grant.definition.aggregate_type,
                aggregate_id=grant.aggregate_id,
                aggregate_version=new_version,
                ledger_seq=ledger_seq,
                event_id=event_id,
                result_digest=result_digest,
                replayed=False,
            )

    def _persist_schema_contract(
        self,
        conn: sqlite3.Connection,
        grant: _AuthorizedCommandGrant,
        *,
        recorded_at: str,
    ) -> None:
        contract = self._payload_schemas.resolve_exact(
            grant.payload.schema_version,
            grant.definition.payload_mode,
            grant.payload.schema_contract_version,
            grant.payload.schema_contract_digest,
            grant.payload.canonicalizer_version,
        )
        canonical = canonical_json_bytes(contract.canonical_value())
        if digest_bytes(canonical) != contract.contract_digest:
            raise AuthorityPersistenceError("payload schema contract digest mismatch")
        conn.execute(
            "INSERT OR IGNORE INTO payload_schema_contracts("
            "contract_digest,schema_version,payload_mode,contract_version,"
            "canonicalizer_implementation_version,canonical_bytes,registered_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                contract.contract_digest,
                contract.schema_version,
                contract.payload_mode.value,
                contract.contract_version,
                contract.canonicalizer_implementation_version,
                canonical,
                recorded_at,
            ),
        )
        row = conn.execute(
            "SELECT * FROM payload_schema_contracts WHERE contract_digest=?",
            (contract.contract_digest,),
        ).fetchone()
        if row is None:
            raise AuthorityPersistenceError("payload schema contract was not persisted")
        self._schema_record_from_row(row)

    def _persist_definition(
        self,
        conn: sqlite3.Connection,
        grant: _AuthorizedCommandGrant,
        *,
        recorded_at: str,
    ) -> None:
        definition = self._command_registry.resolve_exact(
            grant.definition.command_type,
            grant.definition.definition_version,
            grant.definition.digest,
        )
        canonical = canonical_json_bytes(definition.canonical_value())
        if digest_bytes(canonical) != definition.digest:
            raise AuthorityPersistenceError("command definition digest mismatch")
        conn.execute(
            "INSERT OR IGNORE INTO command_definitions("
            "definition_digest,command_type,definition_version,"
            "payload_schema_contract_digest,canonical_bytes,registered_at) "
            "VALUES(?,?,?,?,?,?)",
            (
                definition.digest,
                definition.command_type,
                definition.definition_version,
                definition.payload_schema_contract_digest,
                canonical,
                recorded_at,
            ),
        )
        row = conn.execute(
            "SELECT * FROM command_definitions WHERE definition_digest=?",
            (definition.digest,),
        ).fetchone()
        if row is None:
            raise AuthorityPersistenceError("command definition was not persisted")
        self._definition_record_from_row(row)

    def _persist_security(
        self,
        conn: sqlite3.Connection,
        grant: _AuthorizedCommandGrant,
        *,
        recorded_at: str,
        pending_command_id: str | None = None,
    ) -> None:
        self._persist_security_records(
            conn,
            authentication=grant.authentication,
            request=grant.authorization_request,
            decision=grant.authorization,
            recorded_at=recorded_at,
            pending_command_id=pending_command_id,
        )

    def _persist_security_records(
        self,
        conn: sqlite3.Connection,
        *,
        authentication: Any,
        request: Any,
        decision: Any,
        recorded_at: str,
        pending_command_id: str | None = None,
    ) -> None:
        request_value = request.canonical_value()
        if pending_command_id is None:
            request_residual = canonical_json_bytes({
                key: value for key, value in request_value.items()
                if key not in AUTHORIZATION_REQUEST_INDEXED_FIELDS
            })
            request_marker = AUTHORIZATION_REQUEST_STORAGE_MARKER
        else:
            request_residual = command_request_storage(request_value, pending_command_id)
            request_marker = COMMAND_REQUEST_MARKER
        scopes_bytes = canonical_json_bytes(list(decision.effective_scopes))
        scope_content_digest = digest_bytes(scopes_bytes)

        conn.execute(
            "INSERT OR IGNORE INTO authentication_contexts("
            "authentication_context_id,principal_id,authority_domain,"
            "authentication_method,assurance_class,credential_binding_digest,"
            "authenticated_at,expires_at,storage_context_marker,canonical_digest) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                str(authentication.authentication_context_id),
                authentication.principal_id,
                authentication.authority_domain,
                authentication.authentication_method,
                authentication.assurance_class,
                authentication.credential_binding_digest,
                authentication.authenticated_at.to_text(),
                authentication.expires_at.to_text(),
                b"v37",
                authentication.digest,
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO authorization_requests("
            "request_digest,authentication_context_id,principal_id,authority_domain,"
            "operation_type,required_scope,storage_request_residual,"
            "canonical_record_digest,recorded_at,storage_request_marker) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                request.request_digest,
                str(authentication.authentication_context_id),
                authentication.principal_id,
                authentication.authority_domain,
                request.operation_type,
                request.required_scope,
                request_residual,
                request.digest,
                recorded_at,
                request_marker,
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO authorization_scope_contents("
            "scope_content_digest,canonical_bytes) VALUES(?,?)",
            (scope_content_digest, scopes_bytes),
        )
        scope_row = conn.execute(
            "SELECT canonical_bytes FROM authorization_scope_contents "
            "WHERE scope_content_digest=?", (scope_content_digest,),
        ).fetchone()
        if scope_row is None or bytes(scope_row[0]) != scopes_bytes:
            raise AuthorityPersistenceError("authorization scope content differs")
        conn.execute(
            "INSERT OR IGNORE INTO authorization_decisions("
            "authorization_decision_id,authentication_context_id,"
            "authorization_request_digest,authorization_policy_version,"
            "storage_scope_marker,effective_scope_digest,allowed,reason_code,"
            "decided_at,storage_decision_marker,canonical_digest,scope_content_digest) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                str(decision.authorization_decision_id),
                str(authentication.authentication_context_id),
                request.request_digest,
                decision.authorization_policy_version,
                b"v36",
                decision.effective_scope_digest,
                int(decision.allowed),
                decision.reason_code,
                decision.decided_at.to_text(),
                b"v36",
                decision.digest,
                scope_content_digest,
            ),
        )
        auth_row = conn.execute(
            "SELECT * FROM authentication_contexts WHERE authentication_context_id=?",
            (str(authentication.authentication_context_id),),
        ).fetchone()
        request_row = conn.execute(
            "SELECT * FROM authorization_requests WHERE request_digest=?",
            (request.request_digest,),
        ).fetchone()
        decision_row = conn.execute(
            "SELECT * FROM authorization_decisions WHERE authorization_decision_id=?",
            (str(decision.authorization_decision_id),),
        ).fetchone()
        if auth_row is None or request_row is None or decision_row is None:
            raise AuthorityPersistenceError("security provenance was not persisted")
        self._authentication_record_from_row(auth_row)
        if pending_command_id is not None and bytes(request_row["storage_request_marker"]) == COMMAND_REQUEST_MARKER:
            pending_command_request_bytes(request_row, canonical_json_bytes(request_value), pending_command_id)
        else:
            self._request_record_from_row(request_row, connection=conn)
        self._decision_record_from_row(decision_row, connection=conn)

    def _replay_existing(
        self, grant: _AuthorizedCommandGrant, row: sqlite3.Row
    ) -> CommittedCommand:
        if str(row["stable_semantic_request_digest"]) != grant.stable_semantic_request_digest:
            raise IdempotencyIdentityConflict(
                "idempotency identity belongs to a different semantic command"
            )
        if (
            str(row["command_definition_version"])
            != grant.definition.definition_version
            or str(row["command_definition_digest"]) != grant.definition.digest
        ):
            raise IdempotencyConflict(
                "replay grant does not use the committed command definition"
            )
        if (
            grant.replay_of_command_id is not None
            and str(row["command_id"]) != grant.replay_of_command_id
        ):
            raise IdempotencyConflict("replay command identity mismatch")
        return self._decode_result(
            bytes(row["result_bytes"]), str(row["result_digest"]), replayed=True, command_id=str(row["command_id"])
        )

    @staticmethod
    def _validate_causation(
        conn: sqlite3.Connection, grant: _AuthorizedCommandGrant
    ) -> None:
        if grant.causation_kind == "COMMAND":
            row = conn.execute(
                "SELECT 1 FROM authority_commands WHERE command_id=?",
                (grant.causation_identifier,),
            ).fetchone()
            if row is None:
                raise UnknownCausation("causation command does not resolve")
        elif grant.causation_kind == "EVENT":
            row = conn.execute(
                "SELECT 1 FROM ledger_events WHERE event_id=?",
                (grant.causation_identifier,),
            ).fetchone()
            if row is None:
                raise UnknownCausation("causation event does not resolve")

    @staticmethod
    def _resolve_version(
        conn: sqlite3.Connection, grant: _AuthorizedCommandGrant
    ) -> tuple[int | None, int]:
        row = conn.execute(
            "SELECT current_version FROM authority_aggregates "
            "WHERE aggregate_type=? AND aggregate_id=?",
            (grant.definition.aggregate_type, grant.aggregate_id),
        ).fetchone()
        current = None if row is None else int(row["current_version"])
        if current is None:
            if grant.expected_aggregate_version != 0:
                raise ExpectedVersionConflict(
                    "create command requires expected aggregate version 0"
                )
            return None, 1
        if grant.expected_aggregate_version != current:
            raise ExpectedVersionConflict(
                f"expected aggregate version {grant.expected_aggregate_version}, "
                f"current is {current}"
            )
        return current, current + 1

    @staticmethod
    def _decode_canonical(data: bytes) -> Any:
        try:
            value = json.loads(data.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise AuthorityPersistenceError("stored canonical JSON is invalid") from exc
        if canonical_json_bytes(value) != data:
            raise AuthorityPersistenceError("stored JSON is not canonical")
        return value

    def _logical_result_bytes(
        self, data: bytes, expected_digest: str, *, connection: sqlite3.Connection | None = None,
        command_id: str | None = None,
    ) -> bytes:
        if data.startswith(COMMAND_RESULT_PREFIX):
            return command_result_bytes(connection or self._connection, data, expected_digest, command_id=command_id)
        return data

    def _decode_result(
        self, data: bytes, expected_digest: str, *, replayed: bool,
        connection: sqlite3.Connection | None = None, command_id: str | None = None,
    ) -> CommittedCommand:
        data = self._logical_result_bytes(data, expected_digest, connection=connection, command_id=command_id)
        if digest_bytes(data) != expected_digest:
            raise AuthorityPersistenceError("stored command result digest mismatch")
        value = self._decode_canonical(data)
        required = {
            "command_id",
            "aggregate_type",
            "aggregate_id",
            "aggregate_version",
            "ledger_seq",
            "event_id",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise AuthorityPersistenceError("stored command result shape is invalid")
        if command_id is not None and value["command_id"] != command_id:
            raise AuthorityPersistenceError("stored command result identity mismatch")
        return CommittedCommand(
            command_id=str(value["command_id"]),
            aggregate_type=str(value["aggregate_type"]),
            aggregate_id=str(value["aggregate_id"]),
            aggregate_version=int(value["aggregate_version"]),
            ledger_seq=int(value["ledger_seq"]),
            event_id=str(value["event_id"]),
            result_digest=expected_digest,
            replayed=replayed,
        )
