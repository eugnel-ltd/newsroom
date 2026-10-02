from __future__ import annotations

import sqlite3
from collections import OrderedDict
from typing import Sequence

from .canonical import canonical_json_bytes, digest_bytes, digest_canonical
from .persistence import (
    AuthenticationContextRecord,
    AuthorityPersistenceError,
    DiagnosticHistoryExpired,
    RetiredLedgerEventRecord,
    AuthorizationDecisionRecord,
    AuthorizationRequestRecord,
    CommandDefinitionRecord,
    CommandResultRecord,
    EventProvenanceRecord,
    EventReadPolicy,
    LedgerEventRecord,
    MetadataClass,
    PayloadSchemaContractRecord,
)


_UNSELECTED_SCOPE_CONTENT = object()


class _EventStoreReadMixin:
    """Policy-filtered metadata reads and exact provenance reconstruction."""

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> LedgerEventRecord | RetiredLedgerEventRecord:
        if "retired_header_digest" in row.keys() and row["retired_header_digest"] is not None:
            from .projection_retirement_migrations import retired_record_digest
            if retired_record_digest(row) != bytes(row["retired_record_digest"]):
                raise AuthorityPersistenceError("retired record digest differs")
            return RetiredLedgerEventRecord(
                **{name: row[name] for name in (
                    "ledger_seq", "event_id", "command_id", "event_type", "aggregate_type",
                    "aggregate_id", "security_scope", "trust_scope",
                )}, original_header_digest="sha256:" + bytes(row["retired_header_digest"]).hex(),
            )
        return LedgerEventRecord(
            ledger_seq=int(row["ledger_seq"]),
            event_id=str(row["event_id"]),
            event_type=str(row["event_type"]),
            event_schema_version=int(row["event_schema_version"]),
            aggregate_type=str(row["aggregate_type"]),
            aggregate_id=str(row["aggregate_id"]),
            aggregate_version=int(row["aggregate_version"]),
            recorded_at=str(row["recorded_at"]),
            command_id=str(row["command_id"]),
            producer_version=str(row["producer_version"]),
            command_definition_version=str(row["command_definition_version"]),
            command_definition_digest=str(row["command_definition_digest"]),
            payload_id=str(row["payload_id"]),
            payload_mode=str(row["payload_mode"]),
            payload_schema_version=str(row["payload_schema_version"]),
            payload_schema_contract_version=str(
                row["payload_schema_contract_version"]
            ),
            payload_schema_contract_digest=str(
                row["payload_schema_contract_digest"]
            ),
            payload_canonicalizer_version=str(
                row["payload_canonicalizer_version"]
            ),
            payload_digest=str(row["payload_digest"]),
            object_admission_id=(
                None
                if row["object_admission_id"] is None
                else str(row["object_admission_id"])
            ),
            principal_id=str(row["principal_id"]),
            authentication_context_id=str(row["authentication_context_id"]),
            authorization_request_digest=str(
                row["authorization_request_digest"]
            ),
            authorization_decision_id=str(row["authorization_decision_id"]),
            correlation_id=(
                None if row["correlation_id"] is None else str(row["correlation_id"])
            ),
            causation_kind=(
                None if row["causation_kind"] is None else str(row["causation_kind"])
            ),
            causation_identifier=(
                None
                if row["causation_identifier"] is None
                else str(row["causation_identifier"])
            ),
            causation_external_system=(
                None
                if row["causation_external_system"] is None
                else str(row["causation_external_system"])
            ),
            security_scope=str(row["security_scope"]),
            retention_scope=str(row["retention_scope"]),
            trust_scope=str(row["trust_scope"]),
        )

    @staticmethod
    def _policy_lists(policy: EventReadPolicy) -> tuple[list[str], list[str]]:
        security = sorted(policy.allowed_security_scopes)
        trust = sorted(value.value for value in policy.allowed_trust_scopes)
        return security, trust

    def events_after(
        self,
        ledger_seq: int,
        *,
        limit: int,
        policy: EventReadPolicy,
    ) -> tuple[LedgerEventRecord | RetiredLedgerEventRecord, ...]:
        policy.require_metadata_class(MetadataClass.ROUTING)
        policy.require_window(after_ledger_seq=ledger_seq, limit=limit)
        security, trust = self._policy_lists(policy)
        security_marks = ",".join("?" for _ in security)
        trust_marks = ",".join("?" for _ in trust)
        conditions = [
            "ledger_seq > ?",
            "ledger_seq >= ?",
            f"security_scope IN ({security_marks})",
            f"trust_scope IN ({trust_marks})",
        ]
        values: list[object] = [
            ledger_seq,
            policy.minimum_ledger_seq,
            *security,
            *trust,
        ]
        if policy.maximum_ledger_seq is not None:
            conditions.append("ledger_seq <= ?")
            values.append(policy.maximum_ledger_seq)
        values.append(limit)
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM ledger_events WHERE "
                + " AND ".join(conditions)
                + " ORDER BY ledger_seq LIMIT ?",
                tuple(values),
            ).fetchall()
            if self._current_state_only:
                for row in rows:
                    self._validate_retained_event(str(row["event_id"]))
            return tuple(self._event_from_row(row) for row in rows)

    def _visible_event_row(
        self,
        *,
        policy: EventReadPolicy,
        event_id: str | None = None,
        command_id: str | None = None,
    ) -> sqlite3.Row:
        if (event_id is None) == (command_id is None):
            raise ValueError("exactly one event or command identity is required")
        security, trust = self._policy_lists(policy)
        security_marks = ",".join("?" for _ in security)
        trust_marks = ",".join("?" for _ in trust)
        column = "event_id" if event_id is not None else "command_id"
        identifier = event_id if event_id is not None else command_id
        values: list[object] = [identifier, *security, *trust]
        sequence_clause = "ledger_seq >= ?"
        values.append(policy.minimum_ledger_seq)
        if policy.maximum_ledger_seq is not None:
            sequence_clause += " AND ledger_seq <= ?"
            values.append(policy.maximum_ledger_seq)
        row = self._connection.execute(
            f"SELECT * FROM ledger_events WHERE {column}=? "
            f"AND security_scope IN ({security_marks}) "
            f"AND trust_scope IN ({trust_marks}) AND {sequence_clause}",
            tuple(values),
        ).fetchone()
        if row is None:
            raise KeyError(identifier)
        return row

    def event_provenance(
        self, *, event_id: str, policy: EventReadPolicy
    ) -> EventProvenanceRecord:
        policy.require_metadata_class(MetadataClass.PROVENANCE)
        with self._lock:
            event_row = self._visible_event_row(policy=policy, event_id=event_id)
            event = self._event_from_row(event_row)
            if isinstance(event, RetiredLedgerEventRecord):
                raise DiagnosticHistoryExpired("event diagnostic provenance expired")
            auth_row = self._connection.execute(
                "SELECT * FROM authentication_contexts "
                "WHERE authentication_context_id=?",
                (event.authentication_context_id,),
            ).fetchone()
            request_row = self._connection.execute(
                "SELECT * FROM authorization_requests WHERE request_digest=?",
                (event.authorization_request_digest,),
            ).fetchone()
            decision_row = self._connection.execute(
                "SELECT * FROM authorization_decisions "
                "WHERE authorization_decision_id=?",
                (event.authorization_decision_id,),
            ).fetchone()
            definition_row = self._connection.execute(
                "SELECT * FROM command_definitions WHERE definition_digest=?",
                (event.command_definition_digest,),
            ).fetchone()
            schema_row = self._connection.execute(
                "SELECT * FROM payload_schema_contracts WHERE contract_digest=?",
                (event.payload_schema_contract_digest,),
            ).fetchone()
            if any(
                row is None
                for row in (
                    auth_row,
                    request_row,
                    decision_row,
                    definition_row,
                    schema_row,
                )
            ):
                raise AuthorityPersistenceError(
                    "event provenance is incomplete"
                )

            authentication = self._authentication_record_from_row(auth_row)
            request = self._request_record_from_row(request_row)
            decision = self._decision_record_from_row(decision_row)
            definition = self._definition_record_from_row(definition_row)
            schema = self._schema_record_from_row(schema_row)
            contexts = {
                event.authentication_context_id,
                request.authentication_context_id,
                decision.authentication_context_id,
            }
            requests = {
                event.authorization_request_digest,
                request.request_digest,
                decision.authorization_request_digest,
            }
            if len(contexts) != 1 or len(requests) != 1:
                raise AuthorityPersistenceError(
                    "event authentication/request/decision binding mismatch"
                )
            if event.principal_id != authentication.principal_id:
                raise AuthorityPersistenceError(
                    "event principal does not match authentication context"
                )
            if (
                event.command_definition_version != definition.definition_version
                or event.command_definition_digest != definition.definition_digest
            ):
                raise AuthorityPersistenceError(
                    "event command-definition provenance mismatch"
                )
            if (
                event.payload_schema_version != schema.schema_version
                or event.payload_mode != schema.payload_mode
                or event.payload_schema_contract_version != schema.contract_version
                or event.payload_schema_contract_digest != schema.contract_digest
                or event.payload_canonicalizer_version
                != schema.canonicalizer_implementation_version
            ):
                raise AuthorityPersistenceError(
                    "event payload-schema provenance mismatch"
                )
            return EventProvenanceRecord(
                event=event,
                command_definition=definition,
                payload_schema_contract=schema,
                authentication=authentication,
                authorization_request=request,
                authorization_decision=decision,
            )

    def command_result(
        self, *, command_id: str, policy: EventReadPolicy
    ) -> CommandResultRecord:
        policy.require_metadata_class(MetadataClass.RESULT)
        with self._lock:
            event = self._visible_event_row(policy=policy, command_id=command_id)
            if event["retired_header_digest"] is not None:
                self._event_from_row(event)
                raise DiagnosticHistoryExpired("command diagnostic result expired")
            if self._current_state_only:
                self._validate_retained_event(str(event["event_id"]))
            row = self._connection.execute(
                "SELECT command_id,result_digest,result_bytes "
                "FROM authority_commands WHERE command_id=?",
                (command_id,),
            ).fetchone()
            if row is None:
                raise KeyError(command_id)
            digest = str(row["result_digest"])
            data = self._logical_result_bytes(bytes(row["result_bytes"]), digest, command_id=str(row["command_id"]))
            self._decode_result(data, digest, replayed=False)
            return CommandResultRecord(
                command_id=str(row["command_id"]),
                result_digest=digest,
                result_bytes=data,
            )

    def _schema_record_from_row(
        self, row: sqlite3.Row
    ) -> PayloadSchemaContractRecord:
        data = bytes(row["canonical_bytes"])
        value = self._decode_canonical(data)
        digest = str(row["contract_digest"])
        if digest_bytes(data) != digest:
            raise AuthorityPersistenceError(
                "stored payload schema contract digest mismatch"
            )
        expected = {
            "schema_version": str(row["schema_version"]),
            "payload_mode": str(row["payload_mode"]),
            "contract_version": str(row["contract_version"]),
            "canonicalizer_implementation_version": str(
                row["canonicalizer_implementation_version"]
            ),
        }
        if not isinstance(value, dict) or any(
            value.get(key) != expected_value
            for key, expected_value in expected.items()
        ):
            raise AuthorityPersistenceError(
                "stored payload schema contract fields mismatch"
            )
        return PayloadSchemaContractRecord(
            contract_digest=digest,
            canonical_bytes=data,
            **expected,
        )

    def _definition_record_from_row(
        self, row: sqlite3.Row
    ) -> CommandDefinitionRecord:
        data = bytes(row["canonical_bytes"])
        value = self._decode_canonical(data)
        digest = str(row["definition_digest"])
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
            != str(row["payload_schema_contract_digest"])
        ):
            raise AuthorityPersistenceError(
                "stored command definition fields mismatch"
            )
        return CommandDefinitionRecord(
            definition_digest=digest,
            command_type=str(row["command_type"]),
            definition_version=str(row["definition_version"]),
            canonical_bytes=data,
        )

    def _authentication_record_from_row(
        self, row: sqlite3.Row
    ) -> AuthenticationContextRecord:
        if bytes(row["storage_context_marker"]) != b"v37":
            raise AuthorityPersistenceError("stored authentication context format differs")
        digest = str(row["canonical_digest"])
        expected = {
            "authentication_context_id": str(row["authentication_context_id"]),
            "principal_id": str(row["principal_id"]),
            "authority_domain": str(row["authority_domain"]),
            "authentication_method": str(row["authentication_method"]),
            "assurance_class": str(row["assurance_class"]),
            "credential_binding_digest": str(row["credential_binding_digest"]),
            "authenticated_at": str(row["authenticated_at"]),
            "expires_at": str(row["expires_at"]),
        }
        data = canonical_json_bytes(expected)
        if digest_bytes(data) != digest:
            raise AuthorityPersistenceError(
                "stored authentication context is not canonical"
            )
        return AuthenticationContextRecord(
            **expected,
            canonical_digest=digest,
            canonical_bytes=data,
        )

    def _request_record_from_row(
        self, row: sqlite3.Row, *, connection: sqlite3.Connection | None = None,
        selected_backing: dict | None = None, verified_definitions: dict | None = None,
        verified_schemas: dict | None = None,
    ) -> AuthorizationRequestRecord:
        from .authorization_request_storage_migrations import (
            authorization_request_value_from_v38_row,
        )

        try:
            if bytes(row["storage_request_marker"]) == b"v41":
                from .command_bound_storage import command_request_value
                value = command_request_value(connection or self._connection, row,
                    selected_backing=selected_backing, verified_definitions=verified_definitions,
                    verified_schemas=verified_schemas)
                data = canonical_json_bytes(value)
            else:
                value = authorization_request_value_from_v38_row(row)
                data = canonical_json_bytes(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise AuthorityPersistenceError(
                "stored authorization request representation differs"
            ) from exc
        record_digest = str(row["canonical_record_digest"])
        request_digest = str(row["request_digest"])
        if digest_bytes(data) != record_digest:
            raise AuthorityPersistenceError(
                "authorization request record digest mismatch"
            )
        if value.get("request_digest") != request_digest:
            raise AuthorityPersistenceError(
                "authorization request identity mismatch"
            )
        unsigned = dict(value)
        unsigned.pop("request_digest", None)
        if digest_canonical(unsigned) != request_digest:
            raise AuthorityPersistenceError(
                "authorization request digest mismatch"
            )
        expected = {
            "authentication_context_id": str(row["authentication_context_id"]),
            "principal_id": str(row["principal_id"]),
            "authority_domain": str(row["authority_domain"]),
            "operation_type": str(row["operation_type"]),
            "required_scope": str(row["required_scope"]),
        }
        if any(value.get(key) != expected_value for key, expected_value in expected.items()):
            raise AuthorityPersistenceError(
                "authorization request fields mismatch"
            )
        return AuthorizationRequestRecord(
            request_digest=request_digest,
            canonical_record_digest=record_digest,
            canonical_bytes=data,
            **expected,
        )

    def _decision_record_from_row(
        self,
        row: sqlite3.Row,
        *,
        connection: sqlite3.Connection | None = None,
        selected_scope_bytes: object = _UNSELECTED_SCOPE_CONTENT,
    ) -> AuthorizationDecisionRecord:
        digest = str(row["canonical_digest"])
        if (
            bytes(row["storage_scope_marker"]) != b"v36"
            or bytes(row["storage_decision_marker"]) != b"v36"
        ):
            raise AuthorityPersistenceError("stored authorization decision format differs")
        scope_digest = str(row["scope_content_digest"])
        if selected_scope_bytes is _UNSELECTED_SCOPE_CONTENT:
            scope_row = (connection or self._connection).execute(
                "SELECT scope_content_digest,canonical_bytes "
                "FROM authorization_scope_contents WHERE scope_content_digest=?",
                (scope_digest,),
            ).fetchone()
            if scope_row is None:
                raise AuthorityPersistenceError("stored effective scopes are missing")
        else:
            if selected_scope_bytes is None:
                raise AuthorityPersistenceError("stored effective scopes are missing")
            scope_row = {
                "scope_content_digest": scope_digest,
                "canonical_bytes": selected_scope_bytes,
            }
        scopes_value = self._scope_content_from_row(scope_row)
        expected = {
            "authorization_decision_id": str(row["authorization_decision_id"]),
            "authentication_context_id": str(row["authentication_context_id"]),
            "authorization_request_digest": str(
                row["authorization_request_digest"]
            ),
            "authorization_policy_version": str(
                row["authorization_policy_version"]
            ),
            "effective_scopes": scopes_value,
            "effective_scope_digest": str(row["effective_scope_digest"]),
            "allowed": bool(row["allowed"]),
            "reason_code": str(row["reason_code"]),
            "decided_at": str(row["decided_at"]),
        }
        data = canonical_json_bytes(expected)
        if digest_bytes(data) != digest:
            raise AuthorityPersistenceError(
                "stored authorization decision is not canonical"
            )
        return AuthorizationDecisionRecord(
            authorization_decision_id=expected["authorization_decision_id"],
            authentication_context_id=expected["authentication_context_id"],
            authorization_request_digest=expected[
                "authorization_request_digest"
            ],
            authorization_policy_version=expected[
                "authorization_policy_version"
            ],
            effective_scopes=tuple(scopes_value),
            effective_scope_digest=expected["effective_scope_digest"],
            allowed=expected["allowed"],
            reason_code=expected["reason_code"],
            decided_at=expected["decided_at"],
            canonical_digest=digest,
            canonical_bytes=data,
        )

    def _scope_content_from_row(self, row: sqlite3.Row) -> list[str]:
        data = bytes(row["canonical_bytes"])
        digest = str(row["scope_content_digest"])
        key = (data, digest)
        cache = getattr(self, "_validated_scope_contents", None)
        if len(data) <= 4096 and cache is not None:
            retained = cache.get(key)
            if retained is not None:
                cache.move_to_end(key)
                return list(retained)
        if digest_bytes(data) != digest:
            raise AuthorityPersistenceError("stored effective scopes digest differs")
        value = self._decode_canonical(data)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise AuthorityPersistenceError("stored effective scopes are invalid")
        # Reuse only fully validated exact content, never a decision or store
        # identity. Bounds also cap retained input bytes; larger scopes stay uncached.
        if len(data) <= 4096:
            if cache is None:
                cache = self._validated_scope_contents = OrderedDict()
            cache[key] = tuple(value)
            if len(cache) > 8:
                cache.popitem(last=False)
        return value

    # Private adversarial test seams; never exported as application API.
    def _execute_test_sql(
        self, sql: str, parameters: Sequence[object] = ()
    ) -> None:
        with self._lock:
            self._connection.execute(sql, tuple(parameters))

    def _validate_test_integrity(self) -> None:
        with self._lock:
            self._validate_schema_and_integrity()
