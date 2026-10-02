from __future__ import annotations

import re
import sqlite3

from .canonical import digest_bytes, validate_sha256_digest
from .persistence import (
    AuthorityPersistenceError,
    PayloadId,
    UnsupportedPayloadMode,
)
from .types import (
    AggregateId,
    AuthenticationContextId,
    AuthorizationDecisionId,
    CausationKind,
    CausationRef,
    CommandId,
    CorrelationId,
    EventId,
    PayloadMode,
    TrustScope,
    UtcTimestamp,
    require_scope,
    require_token,
)


_CANONICAL_UUID4 = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)


class _PayloadAndEnvelopeIntegrity:
    """Rehash retained payloads and validate typed event routing values."""

    def _validate_immutable_records(
        self, conn: sqlite3.Connection
    ) -> None:
        super()._validate_immutable_records(conn)  # type: ignore[misc]
        for row in conn.execute(
            "SELECT p.*,c.contract_digest AS selected_contract_digest,"
            "c.schema_version AS selected_schema_version,"
            "c.payload_mode AS selected_payload_mode,"
            "c.contract_version AS selected_contract_version,"
            "c.canonicalizer_implementation_version AS selected_canonicalizer_version "
            "FROM authority_payloads p LEFT JOIN payload_schema_contracts c "
            "ON c.contract_digest=p.schema_contract_digest"
        ):
            self._validate_payload_record(conn, row, selected_contract_row=row)

        for row in conn.execute(
            "SELECT * FROM ledger_events ORDER BY ledger_seq"
        ):
            self._validate_event_types(row)

    def _validate_payload_record(
        self, conn: sqlite3.Connection, row: sqlite3.Row,
        *, selected_contract_row: sqlite3.Row | None = None,
    ) -> None:
        """Validate one payload, optionally using its same-query joined contract."""

        PayloadId.parse(str(row["payload_id"]))
        mode = PayloadMode(str(row["mode"]))
        if mode is PayloadMode.OBJECT_ADMISSION:
            self._validate_object_admission_payload_record(conn, row)
            return
        if row["payload_bytes"] is None:
            raise AuthorityPersistenceError(
                "A2a retained payload bytes are missing"
            )
        data = bytes(row["payload_bytes"])
        expected_digest = str(row["payload_digest"])
        validate_sha256_digest(expected_digest, field="payload_digest")
        if digest_bytes(data) != expected_digest:
            raise AuthorityPersistenceError(
                "retained payload digest does not match exact bytes"
            )
        if mode is PayloadMode.NO_PAYLOAD and data != b"":
            raise AuthorityPersistenceError(
                "NO_PAYLOAD authority must retain exact empty bytes"
            )
        if mode is PayloadMode.INLINE and not data:
            raise AuthorityPersistenceError(
                "INLINE authority cannot retain an empty payload"
            )
        if selected_contract_row is None:
            contract = conn.execute(
                "SELECT * FROM payload_schema_contracts WHERE contract_digest=?",
                (str(row["schema_contract_digest"]),),
            ).fetchone()
        elif selected_contract_row["selected_contract_digest"] is None:
            contract = None
        elif str(selected_contract_row["selected_contract_digest"]) != str(
            row["schema_contract_digest"]
        ):
            raise AuthorityPersistenceError(
                "payload does not match its immutable schema contract"
            )
        else:
            # The LEFT JOIN keeps missing parents visible; no contract or
            # validation result is retained beyond the current streamed row.
            contract = {
                "schema_version": selected_contract_row["selected_schema_version"],
                "payload_mode": selected_contract_row["selected_payload_mode"],
                "contract_version": selected_contract_row["selected_contract_version"],
                "canonicalizer_implementation_version": selected_contract_row[
                    "selected_canonicalizer_version"
                ],
            }
        if contract is None:
            raise AuthorityPersistenceError("payload schema contract is missing")
        if (
            str(contract["schema_version"]) != str(row["schema_version"])
            or str(contract["payload_mode"]) != mode.value
            or str(contract["contract_version"])
            != str(row["schema_contract_version"])
            or str(contract["canonicalizer_implementation_version"])
            != str(row["canonicalizer_implementation_version"])
        ):
            raise AuthorityPersistenceError(
                "payload does not match its immutable schema contract"
            )


    @staticmethod
    def _validate_object_admission_payload_record(
        conn: sqlite3.Connection, row: sqlite3.Row
    ) -> None:
        del conn, row
        raise UnsupportedPayloadMode(
            "object-admission authority belongs to Increment 1A2b"
        )

    @staticmethod
    def _validate_event_types(row: sqlite3.Row) -> None:
        if "retired_header_digest" in row.keys() and row["retired_header_digest"] is not None:
            from ._event_store_read import _EventStoreReadMixin
            _EventStoreReadMixin._event_from_row(row)
            for name, identifier_type in (("event_id", EventId), ("command_id", CommandId), ("aggregate_id", AggregateId)):
                identifier_type.parse(str(row[name]))
            if int(row["ledger_seq"]) <= 0:
                raise AuthorityPersistenceError("retired ledger sequence differs")
            return
        ledger_seq = int(row["ledger_seq"])
        aggregate_version = int(row["aggregate_version"])
        event_schema_version = int(row["event_schema_version"])
        if ledger_seq <= 0 or aggregate_version <= 0 or event_schema_version <= 0:
            raise AuthorityPersistenceError(
                "event sequence and versions must be positive"
            )
        for field, identifier_type in (
            ("event_id", EventId),
            ("command_id", CommandId),
            ("payload_id", PayloadId),
            ("aggregate_id", AggregateId),
            ("authentication_context_id", AuthenticationContextId),
            ("authorization_decision_id", AuthorizationDecisionId),
        ):
            value = str(row[field])
            # These validated identities are not returned. Avoid constructing
            # discarded UUID/typed-ID objects; retain exact parse failures.
            if _CANONICAL_UUID4.fullmatch(value) is None:
                identifier_type.parse(value)
        UtcTimestamp.parse(str(row["recorded_at"]))
        for field in (
            "event_type",
            "aggregate_type",
            "producer_version",
            "command_definition_version",
            "payload_schema_version",
            "payload_schema_contract_version",
            "payload_canonicalizer_version",
            "principal_id",
        ):
            require_token(str(row[field]), field=field)
        for field in ("security_scope", "retention_scope"):
            require_scope(str(row[field]), field=field)
        TrustScope(str(row["trust_scope"]))
        PayloadMode(str(row["payload_mode"]))
        for field in (
            "command_definition_digest",
            "payload_schema_contract_digest",
            "payload_digest",
            "authorization_request_digest",
        ):
            validate_sha256_digest(str(row[field]), field=field)
        if row["correlation_id"] is not None:
            CorrelationId.parse(str(row["correlation_id"]))
        if row["causation_kind"] is not None:
            CausationRef(
                kind=CausationKind(str(row["causation_kind"])),
                identifier=str(row["causation_identifier"]),
                external_system=(
                    None
                    if row["causation_external_system"] is None
                    else str(row["causation_external_system"])
                ),
            )
