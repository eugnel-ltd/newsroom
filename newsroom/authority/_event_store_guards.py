from __future__ import annotations

import sqlite3
from contextlib import contextmanager

from ._capability import _AuthorizedCommandGrant
from .canonical import canonical_json_bytes, digest_bytes, digest_canonical
from .command_bound_storage import COMMAND_REQUEST_MARKER, pending_command_request_bytes
from .persistence import (
    AuthorityPersistenceError,
    CommandDefinitionRecord,
    EventProvenanceRecord,
    EventReadPolicy,
    MetadataClass,
)
from .types import AuditId, CausationKind, PayloadMode, TrustScope, UtcTimestamp


class _ExactAuthorityGuards:
    """Cross-record checks that cannot be expressed by IDs or FKs alone."""

    def _definition_record_from_row(
        self, row: sqlite3.Row
    ) -> CommandDefinitionRecord:
        data = bytes(row["canonical_bytes"])
        value = self._decode_canonical(data)  # type: ignore[attr-defined]
        digest = str(row["definition_digest"])
        schema_digest = str(row["payload_schema_contract_digest"])
        if digest_bytes(data) != digest:
            raise AuthorityPersistenceError(
                "stored command definition digest mismatch"
            )
        if (
            not isinstance(value, dict)
            or value.get("command_type") != str(row["command_type"])
            or value.get("definition_version")
            != str(row["definition_version"])
            or value.get("payload_schema_contract_digest")
            != schema_digest
        ):
            raise AuthorityPersistenceError(
                "stored command definition fields mismatch"
            )
        return CommandDefinitionRecord(
            definition_digest=digest,
            command_type=str(row["command_type"]),
            definition_version=str(row["definition_version"]),
            payload_schema_contract_digest=schema_digest,
            canonical_bytes=data,
        )

    def _persist_security_records(
        self,
        conn: sqlite3.Connection,
        *,
        authentication: object,
        request: object,
        decision: object,
        recorded_at: str,
        pending_command_id: str | None = None,
    ) -> None:
        super()._persist_security_records(  # type: ignore[misc]
            conn,
            authentication=authentication,
            request=request,
            decision=decision,
            recorded_at=recorded_at,
            pending_command_id=pending_command_id,
        )
        expected = (
            (
                "authentication_contexts",
                "authentication_context_id",
                str(authentication.authentication_context_id),  # type: ignore[attr-defined]
                canonical_json_bytes(authentication.canonical_value()),  # type: ignore[attr-defined]
                authentication.digest,  # type: ignore[attr-defined]
                "canonical_digest",
            ),
            (
                "authorization_requests",
                "request_digest",
                request.request_digest,  # type: ignore[attr-defined]
                canonical_json_bytes(request.canonical_value()),  # type: ignore[attr-defined]
                request.digest,  # type: ignore[attr-defined]
                "canonical_record_digest",
            ),
        )
        for (
            table,
            identity_column,
            identity,
            expected_bytes,
            expected_digest,
            digest_column,
        ) in expected:
            columns = "*"
            row = conn.execute(
                f"SELECT {columns} FROM {table} "
                f"WHERE {identity_column}=?",
                (identity,),
            ).fetchone()
            retained_bytes = None if row is None else (
                self._authentication_record_from_row(row).canonical_bytes
                if table == "authentication_contexts"
                else pending_command_request_bytes(row, expected_bytes, pending_command_id)
                if pending_command_id is not None and bytes(row["storage_request_marker"]) == COMMAND_REQUEST_MARKER
                else self._request_record_from_row(row, connection=conn).canonical_bytes
            )
            if (
                retained_bytes != expected_bytes
                or str(row[digest_column]) != expected_digest
            ):
                raise AuthorityPersistenceError(
                    f"{table} identity already belongs to different provenance"
                )
        decision_row = conn.execute(
            "SELECT * FROM authorization_decisions WHERE authorization_decision_id=?",
            (str(decision.authorization_decision_id),),  # type: ignore[attr-defined]
        ).fetchone()
        if decision_row is None:
            raise AuthorityPersistenceError("authorization decision is missing")
        retained = self._decision_record_from_row(decision_row, connection=conn)
        if (
            retained.canonical_bytes
            != canonical_json_bytes(decision.canonical_value())  # type: ignore[attr-defined]
            or retained.canonical_digest != decision.digest  # type: ignore[attr-defined]
        ):
            raise AuthorityPersistenceError(
                "authorization_decisions identity already belongs to different provenance"
            )

    @staticmethod
    def _validate_relational_invariants(
        conn: sqlite3.Connection,
    ) -> None:
        # Preserve the base exact-head, one-version/audit/event and envelope checks.
        from ._event_store_base import _EventStoreBase

        _EventStoreBase._validate_relational_invariants(conn)
        mismatch = conn.execute(
            "SELECT e.event_id FROM ledger_events e "
            "JOIN authentication_contexts a "
            "ON a.authentication_context_id=e.authentication_context_id "
            "JOIN authorization_requests r "
            "ON r.request_digest=e.authorization_request_digest "
            "JOIN authorization_decisions z "
            "ON z.authorization_decision_id=e.authorization_decision_id "
            "JOIN command_definitions d "
            "ON d.definition_digest=e.command_definition_digest "
            "WHERE e.principal_id != a.principal_id "
            "OR r.authentication_context_id != e.authentication_context_id "
            "OR r.principal_id != a.principal_id "
            "OR r.authority_domain != a.authority_domain "
            "OR z.authentication_context_id != e.authentication_context_id "
            "OR z.authorization_request_digest != "
            "e.authorization_request_digest "
            "OR z.allowed != 1 "
            "OR d.payload_schema_contract_digest != "
            "e.payload_schema_contract_digest LIMIT 1"
        ).fetchone()
        if mismatch is not None:
            raise AuthorityPersistenceError(
                "event security or definition provenance is inconsistent"
            )

    def event_provenance(
        self, *, event_id: str, policy: EventReadPolicy
    ) -> EventProvenanceRecord:
        provenance = super().event_provenance(  # type: ignore[misc]
            event_id=event_id, policy=policy
        )
        event = provenance.event
        authentication = provenance.authentication
        request = provenance.authorization_request
        decision = provenance.authorization_decision
        definition = provenance.command_definition

        if (
            request.principal_id != authentication.principal_id
            or request.authority_domain != authentication.authority_domain
        ):
            raise AuthorityPersistenceError(
                "authorization request is not bound to authentication provenance"
            )
        expected_scope_digest = digest_canonical(
            {
                "authentication_context_digest": (
                    authentication.canonical_digest
                ),
                "effective_scopes": list(decision.effective_scopes),
            }
        )
        if decision.effective_scope_digest != expected_scope_digest:
            raise AuthorityPersistenceError(
                "authorization scopes are not bound to authentication provenance"
            )
        if not decision.allowed:
            raise AuthorityPersistenceError(
                "a committed event cannot reference a denied decision"
            )
        authenticated_at = UtcTimestamp.parse(
            authentication.authenticated_at
        )
        expires_at = UtcTimestamp.parse(authentication.expires_at)
        decided_at = UtcTimestamp.parse(decision.decided_at)
        if not (
            authenticated_at.value
            <= decided_at.value
            < expires_at.value
        ):
            raise AuthorityPersistenceError(
                "authorization decision is outside authentication validity"
            )
        if (
            definition.payload_schema_contract_digest
            != event.payload_schema_contract_digest
        ):
            raise AuthorityPersistenceError(
                "event schema contract differs from command definition"
            )

        request_value = self._decode_canonical(  # type: ignore[attr-defined]
            request.canonical_bytes
        )
        expected_request_fields = {
            "command_definition_digest": event.command_definition_digest,
            "aggregate_type": event.aggregate_type,
            "aggregate_id": event.aggregate_id,
            "event_type": event.event_type,
            "event_schema_version": event.event_schema_version,
            "payload_mode": event.payload_mode,
            "payload_schema_version": event.payload_schema_version,
            "payload_schema_contract_version": (
                event.payload_schema_contract_version
            ),
            "payload_schema_contract_digest": (
                event.payload_schema_contract_digest
            ),
            "payload_canonicalizer_version": (
                event.payload_canonicalizer_version
            ),
            "trust_scope": event.trust_scope,
            "security_scope": event.security_scope,
            "retention_scope": event.retention_scope,
        }
        if not isinstance(request_value, dict) or any(
            request_value.get(field) != value
            for field, value in expected_request_fields.items()
        ):
            raise AuthorityPersistenceError(
                "event routing metadata differs from exact authorization request"
            )
        return provenance

    @contextmanager
    def _exact_event_read_scope(self, conn: sqlite3.Connection):
        """Reuse exact immutable closure only inside this owned transaction."""
        with self._lock:
            if conn is not self._connection or not conn.in_transaction:
                raise AuthorityPersistenceError("exact event scope requires its current transaction")
            previous = getattr(self, "_exact_event_scope", None)
            scope = {"connection": conn, "stamp": None, "proofs": {}}
            anchor = f"exact_event_read_{id(scope):x}"
            conn.execute(f"SAVEPOINT {anchor}")
            self._exact_event_scope = scope
            failed = False
            try:
                yield
            except BaseException:
                failed = True
                raise
            finally:
                scope["proofs"].clear()
                self._exact_event_scope = previous
                try:
                    conn.execute(f"RELEASE {anchor}")
                except sqlite3.DatabaseError as exc:
                    if not failed:
                        raise AuthorityPersistenceError("exact event scope transaction changed") from exc

    def _validate_retained_event(self, event_id: str) -> None:
        with self._lock:
            conn = self._connection
            scope = getattr(self, "_exact_event_scope", None)
            if scope is not None and scope["connection"] is conn and conn.in_transaction:
                stamp = (conn.total_changes, conn.execute("PRAGMA data_version").fetchone()[0],
                         conn.execute("PRAGMA schema_version").fetchone()[0])
                if scope["stamp"] != stamp:
                    scope["proofs"].clear()
                    scope["stamp"] = stamp
                if event_id in scope["proofs"]:
                    return
            else:
                if scope is not None:
                    scope["proofs"].clear()
                scope = None
            closure = self._validate_retained_event_exact(event_id)
            if scope is not None and closure is not None:
                scope["proofs"][event_id] = closure

    def _validate_retained_event_exact(self, event_id: str) -> tuple | None:
        """Validate the complete generic authority closure for one event."""

        with self._lock:  # type: ignore[attr-defined]
            conn = self._connection  # type: ignore[attr-defined]
            event_row = conn.execute(
                "SELECT * FROM ledger_events WHERE event_id=?", (event_id,)
            ).fetchone()
            if event_row is None:
                raise AuthorityPersistenceError("retained event is missing")
            if event_row["retired_header_digest"] is not None:
                if self._current_state_only and self._native_checkpoint_schema and event_row['native_checkpoint_id'] is not None:
                    from .native_current_checkpoint import checkpoint_member
                    checkpoint_member(self,event_id)
                    return
                self._validate_event_types(event_row)  # type: ignore[attr-defined]
                return
            self._validate_event_types(event_row)  # type: ignore[attr-defined]
            policy = EventReadPolicy(
                policy_id="exact-retained-event-v1",
                purpose="authority.exact-retained-event",
                required_scope="authority.exact-retained-event.read",
                allowed_principal_ids=frozenset({str(event_row["principal_id"])}),
                allowed_security_scopes=frozenset(
                    {str(event_row["security_scope"])}
                ),
                allowed_trust_scopes=frozenset(
                    {TrustScope(str(event_row["trust_scope"]))}
                ),
                metadata_classes=frozenset({MetadataClass.PROVENANCE}),
                minimum_ledger_seq=int(event_row["ledger_seq"]),
                maximum_ledger_seq=int(event_row["ledger_seq"]),
                max_results=1,
            )
            provenance = self.event_provenance(event_id=event_id, policy=policy)
            event = provenance.event

            counts = conn.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM ledger_events WHERE command_id=?) AS events,"
                "(SELECT COUNT(*) FROM authority_aggregate_versions "
                " WHERE command_id=?) AS versions,"
                "(SELECT COUNT(*) FROM authority_audit_events "
                " WHERE command_id=?) AS audits",
                (event.command_id, event.command_id, event.command_id),
            ).fetchone()
            if counts is None or tuple(counts) != (1, 1, 1):
                raise AuthorityPersistenceError(
                    "retained event command closure cardinality differs"
                )
            command = conn.execute(
                "SELECT * FROM authority_commands WHERE command_id=?",
                (event.command_id,),
            ).fetchone()
            version = conn.execute(
                "SELECT * FROM authority_aggregate_versions WHERE command_id=?",
                (event.command_id,),
            ).fetchone()
            audit = conn.execute(
                "SELECT * FROM authority_audit_events WHERE command_id=?",
                (event.command_id,),
            ).fetchone()
            payload = conn.execute(
                "SELECT * FROM authority_payloads WHERE payload_id=?",
                (event.payload_id,),
            ).fetchone()
            aggregate = conn.execute(
                "SELECT * FROM authority_aggregates "
                "WHERE aggregate_type=? AND aggregate_id=?",
                (event.aggregate_type, event.aggregate_id),
            ).fetchone()
            if any(
                row is None
                for row in (command, version, audit, payload, aggregate)
            ):
                raise AuthorityPersistenceError(
                    "retained event command closure is incomplete"
                )

            self._validate_payload_record(conn, payload)  # type: ignore[attr-defined]
            definition = self._command_registry.resolve_exact(  # type: ignore[attr-defined]
                str(command["command_type"]),
                str(command["command_definition_version"]),
                str(command["command_definition_digest"]),
            )
            contract = self._payload_schemas.resolve_exact(  # type: ignore[attr-defined]
                str(payload["schema_version"]),
                PayloadMode(str(payload["mode"])),
                str(payload["schema_contract_version"]),
                str(payload["schema_contract_digest"]),
                str(payload["canonicalizer_implementation_version"]),
            )
            if (
                canonical_json_bytes(definition.canonical_value())
                != provenance.command_definition.canonical_bytes
                or canonical_json_bytes(contract.canonical_value())
                != provenance.payload_schema_contract.canonical_bytes
            ):
                raise AuthorityPersistenceError(
                    "retained event definition or schema differs"
                )

            head_version = int(aggregate["current_version"])
            head = conn.execute(
                "SELECT recorded_at FROM authority_aggregate_versions "
                "WHERE aggregate_type=? AND aggregate_id=? "
                "AND aggregate_version=?",
                (event.aggregate_type, event.aggregate_id, head_version),
            ).fetchone()
            if head is None or not event.aggregate_version <= head_version:
                raise AuthorityPersistenceError(
                    "retained event aggregate head differs"
                )

            request_value = self._decode_canonical(  # type: ignore[attr-defined]
                provenance.authorization_request.canonical_bytes
            )
            if not isinstance(request_value, dict):
                raise AuthorityPersistenceError(
                    "retained authorization request is not an object"
                )
            payload_mode = PayloadMode(str(payload["mode"]))
            payload_value = {
                "kind": payload_mode.value,
                "schema_version": str(payload["schema_version"]),
                "schema_contract_version": str(
                    payload["schema_contract_version"]
                ),
                "schema_contract_digest": str(payload["schema_contract_digest"]),
                "canonicalizer_version": str(
                    payload["canonicalizer_implementation_version"]
                ),
                "digest": str(payload["payload_digest"]),
                "inline_digest": (
                    None
                    if payload_mode is PayloadMode.OBJECT_ADMISSION
                    else str(payload["payload_digest"])
                ),
                "object_admission_id": (
                    None
                    if payload["object_admission_id"] is None
                    else str(payload["object_admission_id"])
                ),
                "blob_digest": (
                    str(payload["payload_digest"])
                    if payload_mode is PayloadMode.OBJECT_ADMISSION
                    else None
                ),
                "object_class": request_value.get("object_class"),
                "allowed_use": request_value.get("allowed_use"),
            }
            expected_semantic_digest = digest_canonical(
                {
                    "command_type": definition.command_type,
                    "command_definition_version": definition.definition_version,
                    "command_definition_digest": definition.digest,
                    "aggregate_type": definition.aggregate_type,
                    "aggregate_id": event.aggregate_id,
                    "expected_aggregate_version": event.aggregate_version - 1,
                    "payload": payload_value,
                }
            )
            expected_namespace = digest_canonical(
                {
                    "authority_domain": provenance.authentication.authority_domain,
                    "principal_id": provenance.authentication.principal_id,
                    "command_type": definition.command_type,
                }
            )
            routing = {
                "command_type": definition.command_type,
                "aggregate_type": definition.aggregate_type,
                "event_type": definition.event_type,
                "event_schema_version": definition.event_schema_version,
                "payload_mode": definition.payload_mode.value,
                "payload_schema_version": definition.payload_schema_version,
                "payload_schema_contract_version": (
                    definition.payload_schema_contract_version
                ),
                "payload_schema_contract_digest": (
                    definition.payload_schema_contract_digest
                ),
                "payload_canonicalizer_version": (
                    definition.payload_canonicalizer_version
                ),
                "trust_scope": definition.trust_scope.value,
                "security_scope": definition.security_scope,
                "retention_scope": definition.retention_scope,
            }
            event_routing = {
                "command_type": str(command["command_type"]),
                "aggregate_type": event.aggregate_type,
                "event_type": event.event_type,
                "event_schema_version": event.event_schema_version,
                "payload_mode": event.payload_mode,
                "payload_schema_version": event.payload_schema_version,
                "payload_schema_contract_version": (
                    event.payload_schema_contract_version
                ),
                "payload_schema_contract_digest": (
                    event.payload_schema_contract_digest
                ),
                "payload_canonicalizer_version": (
                    event.payload_canonicalizer_version
                ),
                "trust_scope": event.trust_scope,
                "security_scope": event.security_scope,
                "retention_scope": event.retention_scope,
            }
            request_routing = {
                "command_definition_digest": definition.digest,
                "aggregate_type": definition.aggregate_type,
                "event_type": definition.event_type,
                "event_schema_version": definition.event_schema_version,
                "payload_mode": definition.payload_mode.value,
                "payload_schema_version": definition.payload_schema_version,
                "payload_schema_contract_version": (
                    definition.payload_schema_contract_version
                ),
                "payload_schema_contract_digest": (
                    definition.payload_schema_contract_digest
                ),
                "payload_canonicalizer_version": (
                    definition.payload_canonicalizer_version
                ),
                "trust_scope": definition.trust_scope.value,
                "security_scope": definition.security_scope,
                "retention_scope": definition.retention_scope,
            }
            if routing != event_routing or any(
                request_value.get(field) != value
                for field, value in {
                    **request_routing,
                    "operation_type": f"command:{definition.command_type}",
                    "required_scope": definition.required_scope,
                    "stable_semantic_request_digest": expected_semantic_digest,
                    "aggregate_id": event.aggregate_id,
                    "object_class": definition.required_object_class,
                    "allowed_use": definition.required_allowed_use,
                }.items()
            ):
                raise AuthorityPersistenceError(
                    "retained event routing definition differs"
                )

            recorded_at = event.recorded_at
            if (
                tuple(
                    command[field]
                    for field in (
                        "producer_version",
                        "command_definition_version",
                        "command_definition_digest",
                        "aggregate_type",
                        "aggregate_id",
                        "payload_id",
                        "authentication_context_id",
                        "authorization_request_digest",
                        "authorization_decision_id",
                    )
                )
                != (
                    event.producer_version,
                    event.command_definition_version,
                    event.command_definition_digest,
                    event.aggregate_type,
                    event.aggregate_id,
                    event.payload_id,
                    event.authentication_context_id,
                    event.authorization_request_digest,
                    event.authorization_decision_id,
                )
                or int(command["expected_aggregate_version"])
                != event.aggregate_version - 1
                or str(command["idempotency_namespace"]) != expected_namespace
                or str(command["stable_semantic_request_digest"])
                != expected_semantic_digest
                or tuple(
                    version[field]
                    for field in (
                        "aggregate_type",
                        "aggregate_id",
                        "aggregate_version",
                        "command_id",
                        "payload_id",
                        "trust_scope",
                    )
                )
                != (
                    event.aggregate_type,
                    event.aggregate_id,
                    event.aggregate_version,
                    event.command_id,
                    event.payload_id,
                    event.trust_scope,
                )
                or event.object_admission_id
                != (
                    None
                    if payload["object_admission_id"] is None
                    else str(payload["object_admission_id"])
                )
                or (
                    event.payload_mode,
                    event.payload_schema_version,
                    event.payload_schema_contract_version,
                    event.payload_schema_contract_digest,
                    event.payload_canonicalizer_version,
                    event.payload_digest,
                )
                != (
                    str(payload["mode"]),
                    str(payload["schema_version"]),
                    str(payload["schema_contract_version"]),
                    str(payload["schema_contract_digest"]),
                    str(payload["canonicalizer_implementation_version"]),
                    str(payload["payload_digest"]),
                )
            ):
                raise AuthorityPersistenceError(
                    "retained event command envelope differs"
                )

            result = self._decode_result(  # type: ignore[attr-defined]
                bytes(command["result_bytes"]),
                str(command["result_digest"]),
                replayed=False,
                command_id=str(command["command_id"]),
            )
            if (
                result.command_id,
                result.aggregate_type,
                result.aggregate_id,
                result.aggregate_version,
                result.ledger_seq,
                result.event_id,
            ) != (
                event.command_id,
                event.aggregate_type,
                event.aggregate_id,
                event.aggregate_version,
                event.ledger_seq,
                event.event_id,
            ):
                raise AuthorityPersistenceError(
                    "retained event command result differs"
                )

            AuditId.parse(str(audit["audit_id"]))
            detail = {
                "operation": "COMMAND_COMMIT",
                "command_type": definition.command_type,
                "aggregate_id": event.aggregate_id,
                "expected_aggregate_version": event.aggregate_version - 1,
                "definition_digest": definition.digest,
                "definition_version": definition.definition_version,
                "payload": payload_value,
                "authentication_context_digest": (
                    provenance.authentication.canonical_digest
                ),
                "authorization_request_record_digest": (
                    provenance.authorization_request.canonical_record_digest
                ),
                "authorization_request_digest": (
                    provenance.authorization_request.request_digest
                ),
                "authorization_decision_digest": (
                    provenance.authorization_decision.canonical_digest
                ),
                "idempotency_namespace": str(command["idempotency_namespace"]),
                "idempotency_key": str(command["idempotency_key"]),
                "stable_semantic_request_digest": expected_semantic_digest,
                "correlation_id": event.correlation_id,
                "causation_kind": event.causation_kind,
                "causation_identifier": event.causation_identifier,
                "causation_external_system": event.causation_external_system,
                "replay_of_command_id": None,
            }
            if (
                tuple(
                    audit[field]
                    for field in (
                        "command_id",
                        "authentication_context_id",
                        "authorization_request_digest",
                        "authorization_decision_id",
                        "event_type",
                    )
                )
                != (
                    event.command_id,
                    event.authentication_context_id,
                    event.authorization_request_digest,
                    event.authorization_decision_id,
                    event.event_type,
                )
                or str(audit["detail_digest"]) != digest_canonical(detail)
            ):
                raise AuthorityPersistenceError(
                    "retained event audit envelope differs"
                )

            timestamps = (
                recorded_at,
                str(command["committed_at"]),
                str(version["recorded_at"]),
                str(audit["recorded_at"]),
                str(payload["created_at"]),
            )
            parsed = tuple(UtcTimestamp.parse(value).value for value in timestamps)
            created = UtcTimestamp.parse(str(aggregate["created_at"])).value
            updated = UtcTimestamp.parse(str(aggregate["updated_at"])).value
            head_recorded = UtcTimestamp.parse(str(head["recorded_at"])).value
            if (
                len(set(parsed)) != 1
                or not created <= parsed[0] <= updated
                or head_recorded != updated
            ):
                raise AuthorityPersistenceError(
                    "retained event timestamps differ"
                )

            if event.causation_kind in {
                CausationKind.COMMAND.value,
                CausationKind.EVENT.value,
            }:
                table, column = (
                    ("authority_commands", "command_id")
                    if event.causation_kind == CausationKind.COMMAND.value
                    else ("ledger_events", "event_id")
                )
                if conn.execute(
                    f"SELECT 1 FROM {table} WHERE {column}=?"
                    + (" AND retired_header_digest IS NULL" if table == "ledger_events" else ""),
                    (event.causation_identifier,),
                ).fetchone() is None:
                    raise AuthorityPersistenceError(
                        "retained event causation target is missing"
                    )
            # Retain the complete verified closure, not a count or ledger watermark.
            return (
                tuple(event_row), tuple(command), tuple(version), tuple(audit),
                tuple(payload), tuple(aggregate), tuple(head), tuple(counts),
                provenance.authentication.canonical_bytes,
                provenance.authorization_request.canonical_bytes,
                provenance.authorization_decision.canonical_bytes,
                provenance.command_definition.canonical_bytes,
                provenance.payload_schema_contract.canonical_bytes,
            )
