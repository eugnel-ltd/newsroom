from __future__ import annotations

from ._event_store_base import _validation_stage
# ruff: noqa: I001 - preserve legacy import layout within bounded change
# fmt: off - preserve legacy layout and bounded addition within the line cap

import sqlite3
from collections.abc import Callable, Mapping
from threading import get_ident
from typing import Any

from newsroom.authority._capability import _AuthorizedCommandGrant
from newsroom.authority._check_store import _CheckAuthorityStore
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.authority.persistence import AuthorityPersistenceError, AuthoritySchemaError
from newsroom.authority.types import EventId, PayloadMode, TrustScope, UtcTimestamp
from newsroom.discovery import _compose_discovery_governing_producer_read_port
from newsroom.discovery.models import (
    DiscoverySignalRequest,
    GateDecisionRequest,
    LeadDispositionDecisionRequest,
    NewsLeadRequest,
    WatchConditionRequest,
)
from newsroom.discovery.policy import (
    DISCOVERY_GATE_DECIDE_COMMAND,
    DISCOVERY_LEAD_DISPOSITION_RECORD_COMMAND,
    DISCOVERY_LEAD_OPEN_COMMAND,
    DISCOVERY_SIGNAL_ADMIT_COMMAND,
    DISCOVERY_WATCH_CONDITION_RECORD_COMMAND,
)
from newsroom.discovery.read_models import (
    DiscoveryCurrentActionSource,
    DiscoveryCurrentPhase,
    DiscoveryCurrentStatus,
)
from newsroom.discovery.record_models import (
    DiscoverySignal,
    GateDecision,
    LeadDispositionDecision,
    NewsLead,
    WatchCondition,
)
from newsroom.discovery.types import (
    DiscoveryContractError,
    DiscoveryIdentifierReuse,
    DiscoverySemanticCollision,
    DiscoverySignalId,
    DiscoveryStateError,
    DiscoveryVersionConflict,
    GateDecisionId,
    GateOutcome,
    LeadDispositionDecisionId,
    LeadDispositionOutcome,
    NewsLeadId,
    WatchConditionId,
    deterministic_gate_outcome,
    permitted_newness_for_transition,
)
from newsroom.sources import SourceDefinitionId, SourceDefinitionVersionId

from ._discovery_decoding import (
    disposition_request_from_bytes,
    gate_request_from_bytes,
    lead_request_from_bytes,
    signal_request_from_bytes,
    watch_request_from_bytes,
)

_SOURCE_OBSERVATION_COMPATIBILITY_COLUMNS = (
    "adapter_policy_id",
    "adapter_policy_version",
    "extraction_scope_bytes",
    "observation_model",
    "baseline_policy_id",
    "baseline_policy_version",
    "baseline_kind",
    "baseline_freshness_seconds",
    "baseline_reset_requires_decision",
    "baseline_notes",
    "item_identity_policy_id",
    "item_identity_policy_version",
    "revision_policy_id",
    "revision_policy_version",
    "canonicalization_policy_id",
    "canonicalization_policy_version",
)

_DISCOVERY_RECORD_SPECS: dict[str, tuple[str, str, TrustScope]] = {
    DISCOVERY_SIGNAL_ADMIT_COMMAND: (
        "discovery_signal",
        "discovery.signal.admitted",
        TrustScope.ADMITTED,
    ),
    DISCOVERY_GATE_DECIDE_COMMAND: (
        "gate_decision",
        "discovery.gate.decided",
        TrustScope.ADMITTED,
    ),
    DISCOVERY_LEAD_OPEN_COMMAND: (
        "news_lead",
        "discovery.lead.opened",
        TrustScope.ADMITTED,
    ),
    DISCOVERY_WATCH_CONDITION_RECORD_COMMAND: (
        "watch_condition",
        "discovery.watch_condition.recorded",
        TrustScope.ADMITTED,
    ),
    DISCOVERY_LEAD_DISPOSITION_RECORD_COMMAND: (
        "lead_disposition_decision",
        "discovery.lead.disposition.recorded",
        TrustScope.ADMITTED,
    ),
}

class _DiscoveryGoverningProducerReader:
    __slots__ = ("_connection", "_owner", "_object_payload_validator", "_validate_retained_event")

    def __init__(self, connection: sqlite3.Connection, object_payload_validator=None, validate_retained_event=None) -> None:
        self._connection = connection
        self._owner = get_ident()
        self._object_payload_validator = object_payload_validator
        self._validate_retained_event = validate_retained_event

    def _require_transaction(self) -> None:
        if (
            get_ident() != self._owner
            or self._connection.isolation_level is not None
            or self._connection.row_factory is not sqlite3.Row
            or not self._connection.in_transaction
        ):
            raise DiscoveryContractError("Discovery transaction ownership differs")

    def require_current_governing_producers(
        self, lead_ids: tuple[NewsLeadId, ...]
    ) -> tuple[tuple[NewsLead, DiscoverySignal, GateDecision], ...]:
        self._require_transaction()
        if (
            type(lead_ids) is not tuple
            or not lead_ids
            or any(type(value) is not NewsLeadId for value in lead_ids)
            or tuple(str(value) for value in lead_ids)
            != tuple(sorted({str(value) for value in lead_ids}))
        ):
            raise DiscoveryContractError("Lead IDs must be exact ordered unique values")
        connection = self._connection
        if self._validate_retained_event is None:
            _validate_discovery_reads_in_transaction(
                connection, self._object_payload_validator,
            )
        selected_rows: dict[str, dict[str, sqlite3.Row]] = {}

        def retain(table: str, row: sqlite3.Row) -> None:
            selected_rows.setdefault(table, {})[str(row["authority_event_id"])] = row

        result: list[tuple[NewsLead, DiscoverySignal, GateDecision]] = []
        for lead_id in lead_ids:
            lead_row = _DiscoveryAuthorityStore._row(
                connection, "news_leads", "lead_id", str(lead_id)
            )
            lead = _DiscoveryAuthorityStore._lead_from_row(
                connection, lead_row, replayed=False
            )
            signal_row = _DiscoveryAuthorityStore._row(connection, "discovery_signals", "signal_id", str(lead.request.signal_id))
            signal = _DiscoveryAuthorityStore._signal_from_row(
                connection, signal_row, replayed=False
            )
            current_gate_row = connection.execute(
                "SELECT d.* FROM discovery_gate_decision_heads h "
                "JOIN discovery_gate_decisions d "
                "ON d.decision_id=h.current_decision_id "
                "WHERE h.signal_id=?",
                (str(signal.request.signal_id),),
            ).fetchone()
            if current_gate_row is None:
                raise DiscoveryStateError("current Gate Decision is not retained")
            current_gate = _DiscoveryAuthorityStore._gate_from_row(
                connection, current_gate_row, replayed=False
            )
            promoting_gate_row = _DiscoveryAuthorityStore._row(
                connection,
                "discovery_gate_decisions",
                "decision_id",
                str(lead.request.promoting_gate_decision_id),
            )
            promoting_gate = _DiscoveryAuthorityStore._gate_from_row(
                connection, promoting_gate_row, replayed=False
            )
            if self._validate_retained_event is not None:
                retain("news_leads", lead_row)
                retain("discovery_signals", signal_row)
                retain("discovery_gate_decisions", promoting_gate_row)
                retain("discovery_gate_decisions", current_gate_row)
                # A recoverable Lead prefix need not yet have a disposition.
                # Where present, retain its current decision and exact Watch /
                # Gate references, not every superseded historical decision.
                disposition_row = connection.execute(
                    "SELECT d.* FROM lead_disposition_heads h "
                    "JOIN lead_disposition_decisions d ON d.decision_id=h.current_decision_id "
                    "WHERE h.lead_id=?", (str(lead_id),),
                ).fetchone()
                if disposition_row is not None:
                    retain("lead_disposition_decisions", disposition_row)
                    retain("discovery_gate_decisions", _DiscoveryAuthorityStore._row(
                        connection, "discovery_gate_decisions", "decision_id",
                        str(disposition_row["gate_decision_id"]),
                    ))
                    if disposition_row["watch_condition_id"] is not None:
                        watch_row = _DiscoveryAuthorityStore._row(
                            connection, "discovery_watch_conditions", "watch_condition_id",
                            str(disposition_row["watch_condition_id"]),
                        )
                        retain("discovery_watch_conditions", watch_row)
                        retain("discovery_gate_decisions", _DiscoveryAuthorityStore._row(
                            connection, "discovery_gate_decisions", "decision_id",
                            str(watch_row["gate_decision_id"]),
                        ))
            if (
                lead.request.signal_id != signal.request.signal_id
                or promoting_gate.request.signal_id != signal.request.signal_id
                or promoting_gate.request.outcome is not GateOutcome.PROMOTED_TO_LEAD
                or current_gate.request.signal_id != signal.request.signal_id
                or current_gate.request.outcome is not GateOutcome.PROMOTED_TO_LEAD
                or lead.request.definition_id != signal.request.definition_id
                or lead.request.item_id != signal.request.item_id
                or lead.request.revision_id != signal.request.revision_id
                or lead.request.representation_id != signal.request.representation_id
                or lead.request.occurrence_id != signal.request.occurrence_id
                or lead.request.transition_id != signal.request.transition_id
                or lead.request.coverage != promoting_gate.request.coverage
                or lead.request.coverage != current_gate.request.coverage
                or promoting_gate.request.evaluated_definition_version_id
                != signal.request.definition_version_id
                or current_gate.request.evaluated_definition_version_id
                != signal.request.definition_version_id
                or promoting_gate.request.evaluated_definition_version_id
                != lead.request.definition_version_id
                or current_gate.request.evaluated_definition_version_id
                != lead.request.definition_version_id
            ):
                raise DiscoveryVersionConflict("Discovery closure differs from authority")
            # Candidate lineage names the immutable Gate that promoted the Lead;
            # current eligibility is independently established by current_gate.
            result.append((lead, signal, promoting_gate))
        if self._validate_retained_event is not None:
            _validate_discovery_domain_reads(
                connection, self._validate_retained_event,
                selected_rows=selected_rows,
            )
        return tuple(result)


def _validate_discovery_domain_reads(
    connection: sqlite3.Connection,
    validate_retained_event: Callable[[str], None],
    *,
    selected_rows: dict[str, dict[str, sqlite3.Row]] | None = None,
) -> None:
    domain_rows = (
        ("discovery_signals", _DiscoveryAuthorityStore._signal_from_row),
        ("discovery_gate_decisions", _DiscoveryAuthorityStore._gate_from_row),
        ("news_leads", _DiscoveryAuthorityStore._lead_from_row),
        ("discovery_watch_conditions", _DiscoveryAuthorityStore._watch_from_row),
        ("lead_disposition_decisions", _DiscoveryAuthorityStore._disposition_from_row),
    )
    for table, decoder in domain_rows:
        rows = (
            connection.execute(f"SELECT * FROM {table}")
            if selected_rows is None
            else selected_rows.get(table, {}).values()
        )
        for row in rows:
            validate_retained_event(str(row["authority_event_id"]))
            record = decoder(connection, row, replayed=False)
            if table == "discovery_signals":
                _DiscoveryAuthorityStore._require_exact_signal_lineage(
                    connection, record.request
                )
            elif table == "news_leads":
                _DiscoveryAuthorityStore._require_source_contract_matches_lead(
                    connection, record.request
                )
    _DiscoveryAuthorityStore._validate_discovery_heads(connection)
    _DiscoveryAuthorityStore._validate_discovery_event_coverage(
        connection,
        aggregate_types=tuple(spec[0] for spec in _DISCOVERY_RECORD_SPECS.values()),
    )
    for table, _ in domain_rows:
        if connection.execute(
            f'PRAGMA foreign_key_check("{table}")'
        ).fetchone() is not None:
            raise AuthoritySchemaError("Discovery foreign-key integrity differs")
    for table in ("discovery_gate_decision_heads", "lead_disposition_heads"):
        if connection.execute(
            f'PRAGMA foreign_key_check("{table}")'
        ).fetchone() is not None:
            raise AuthoritySchemaError("Discovery foreign-key integrity differs")


def _validate_discovery_reads_in_transaction(
    connection: sqlite3.Connection,
    object_payload_validator=None,
    validate_retained_event: Callable[[str], None] | None = None,
) -> None:
    if validate_retained_event is not None:
        _validate_discovery_domain_reads(connection, validate_retained_event)
        return
    _DiscoveryAuthorityStore._validate_relational_invariants(connection)
    verifier = object.__new__(_DiscoveryAuthorityStore)
    if object_payload_validator is not None:
        verifier._validate_object_admission_payload_record = object_payload_validator
    verifier._validate_immutable_records(connection)
    for row in connection.execute(
        "SELECT c.*,p.mode,p.schema_version,p.schema_contract_version,p.schema_contract_digest,"
        "p.canonicalizer_implementation_version,p.payload_digest,p.payload_bytes,p.object_admission_id,"
        "e.ledger_seq,e.event_id,e.event_type,e.aggregate_version,e.recorded_at,e.trust_scope,e.correlation_id,"
        "e.causation_kind,e.causation_identifier,e.causation_external_system,e.authentication_context_id event_auth,e.authorization_request_digest event_request,e.authorization_decision_id event_decision,a.event_type audit_event_type,"
        "a.detail_digest,a.recorded_at audit_recorded_at,a.authentication_context_id audit_auth,a.authorization_request_digest audit_request,a.authorization_decision_id audit_decision,x.authority_domain,x.principal_id,"
        "x.canonical_digest auth_digest,r.canonical_record_digest request_record_digest,d.canonical_digest decision_digest "
        "FROM authority_commands c JOIN authority_payloads p ON p.payload_id=c.payload_id "
        "JOIN ledger_events e ON e.command_id=c.command_id JOIN authority_audit_events a ON a.command_id=c.command_id "
        "JOIN authentication_contexts x ON x.authentication_context_id=c.authentication_context_id "
        "JOIN authorization_requests r ON r.request_digest=c.authorization_request_digest "
        "JOIN authorization_decisions d ON d.authorization_decision_id=c.authorization_decision_id "
        "WHERE e.event_type IN (?,?,?,?,?)",
        tuple(spec[1] for spec in _DISCOVERY_RECORD_SPECS.values()),
    ):
        def fields(output: str, source: str, captured: sqlite3.Row = row) -> dict[str, Any]:
            return dict(zip(output.split(), (captured[key] for key in source.split())))

        payload = fields(
            "kind schema_version schema_contract_version schema_contract_digest canonicalizer_version digest object_admission_id",
            "mode schema_version schema_contract_version schema_contract_digest canonicalizer_implementation_version payload_digest object_admission_id",
        )
        payload.update(
            inline_digest=digest_bytes(bytes(row["payload_bytes"])),
            blob_digest=None,
            object_class=None,
            allowed_use=None,
        )
        command = fields(
            "command_type command_definition_version command_definition_digest aggregate_type aggregate_id expected_aggregate_version",
            "command_type command_definition_version command_definition_digest aggregate_type aggregate_id expected_aggregate_version",
        )
        command["payload"] = payload
        expected_result = {
            "command_id": str(row["command_id"]),
            "aggregate_type": str(row["aggregate_type"]),
            "aggregate_id": str(row["aggregate_id"]),
            "aggregate_version": int(row["aggregate_version"]),
            "ledger_seq": int(row["ledger_seq"]),
            "event_id": str(row["event_id"]),
        }
        detail = {key: row[key] for key in "command_type aggregate_id expected_aggregate_version authorization_request_digest idempotency_namespace idempotency_key stable_semantic_request_digest correlation_id causation_kind causation_identifier causation_external_system".split()}  # noqa: SIM905
        detail.update(
            operation="COMMAND_COMMIT",
            definition_digest=row["command_definition_digest"],
            definition_version=row["command_definition_version"],
            payload=payload,
            authentication_context_digest=row["auth_digest"],
            authorization_request_record_digest=row["request_record_digest"],
            authorization_decision_digest=row["decision_digest"],
            replay_of_command_id=None,
        )
        if (
            (spec := _DISCOVERY_RECORD_SPECS.get(row["command_type"])) is None
            or (row["aggregate_type"], row["event_type"], row["trust_scope"]) != (spec[0], spec[1], spec[2].value)
            or digest_canonical(command) != row["stable_semantic_request_digest"]
            or _DiscoveryAuthorityStore._decode_canonical(bytes(row["result_bytes"])) != expected_result
            or digest_bytes(bytes(row["result_bytes"])) != row["result_digest"]
            or len({(row["authentication_context_id"], row["authorization_request_digest"], row["authorization_decision_id"]), (row["event_auth"], row["event_request"], row["event_decision"]), (row["audit_auth"], row["audit_request"], row["audit_decision"])}) != 1 or row["committed_at"] != row["recorded_at"]
            or row["audit_recorded_at"] != row["recorded_at"]
            or row["audit_event_type"] != row["event_type"]
            or row["idempotency_namespace"]
            != digest_canonical(
                {
                    "authority_domain": row["authority_domain"],
                    "principal_id": row["principal_id"],
                    "command_type": row["command_type"],
                }
            )
            or row["detail_digest"] != digest_canonical(detail)
        ):
            raise AuthorityPersistenceError(
                "Discovery command or audit authority differs"
            )
    for row in connection.execute("SELECT * FROM discovery_signals"):
        record = _DiscoveryAuthorityStore._signal_from_row(
            connection, row, replayed=False
        )
        _DiscoveryAuthorityStore._require_exact_signal_lineage(
            connection, record.request
        )
    for row in connection.execute("SELECT * FROM discovery_gate_decisions"):
        _DiscoveryAuthorityStore._gate_from_row(connection, row, replayed=False)
    for row in connection.execute("SELECT * FROM news_leads"):
        record = _DiscoveryAuthorityStore._lead_from_row(
            connection, row, replayed=False
        )
        _DiscoveryAuthorityStore._require_source_contract_matches_lead(
            connection, record.request
        )
    _DiscoveryAuthorityStore._validate_discovery_heads(connection)
    _DiscoveryAuthorityStore._validate_discovery_event_coverage(connection)
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise AuthoritySchemaError("Discovery foreign-key integrity differs")


def _create_discovery_governing_producer_read_port(
    connection: sqlite3.Connection,
    *,
    object_admission_payload_validator: Callable[[sqlite3.Connection, sqlite3.Row], None] | None = None,
    validate_retained_event: Callable[[str], None] | None = None,
):
    try:
        if (
            type(connection) is not sqlite3.Connection
            or connection.isolation_level is not None
            or connection.row_factory is not sqlite3.Row
            or not connection.in_transaction
            or connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1
            or str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            != "wal"
            or connection.execute("PRAGMA synchronous").fetchone()[0] != 2
            or (
                validate_retained_event is not None
                and not callable(validate_retained_event)
            )
        ):
            raise DiscoveryContractError(
                "Discovery read-port factory requires an exact active checked connection"
            )
        reader = _DiscoveryGoverningProducerReader(
            connection,
            object_admission_payload_validator,
            validate_retained_event,
        )
        return _compose_discovery_governing_producer_read_port(
            reader.require_current_governing_producers
        )
    except DiscoveryContractError:
        raise
    except Exception as exc:
        raise DiscoveryContractError(
            "Discovery read-port factory failed closed"
        ) from exc

# fmt: on

class _DiscoveryAuthorityStore(_CheckAuthorityStore):
    """Private single-writer Signal, Gate and Lead authority store."""

    # ------------------------------------------------------------------
    # Shared grant, identity, envelope and row helpers
    # ------------------------------------------------------------------
    def _require_discovery_grant(
        self,
        grant: _AuthorizedCommandGrant,
        *,
        command_type: str,
        aggregate_id: str,
        canonical_bytes: bytes,
    ) -> None:
        self._issuer.verify(grant)
        spec = _DISCOVERY_RECORD_SPECS.get(command_type)
        if spec is None:
            raise AuthorityPersistenceError("unknown Signal/Lead authority command")
        aggregate_type, event_type, trust_scope = spec
        definition = grant.definition
        if (
            grant.command_type != command_type
            or grant.aggregate_id != aggregate_id
            or grant.expected_aggregate_version != 0
            or definition.command_type != command_type
            or definition.aggregate_type != aggregate_type
            or definition.event_type != event_type
            or definition.trust_scope is not trust_scope
            or definition.security_scope != "authority.discovery"
            or definition.retention_scope != "authority.audit"
            or definition.payload_mode is not PayloadMode.INLINE
            or grant.payload.kind != PayloadMode.INLINE.value
            or grant.payload.inline_bytes != canonical_bytes
            or grant.payload.digest != digest_bytes(canonical_bytes)
        ):
            raise AuthorityPersistenceError(
                "Signal/Lead command grant differs from the typed record"
            )

    @staticmethod
    def _discovery_identifier_absent(
        conn: sqlite3.Connection,
        *,
        table: str,
        column: str,
        identifier: str,
        identity: str,
    ) -> None:
        if conn.execute(
            f"SELECT 1 FROM {table} WHERE {column}=?", (identifier,)
        ).fetchone() is not None:
            raise DiscoveryIdentifierReuse(
                f"{identity} is already retained under different command identity"
            )

    @staticmethod
    def _discovery_semantic_absent(
        conn: sqlite3.Connection,
        *,
        table: str,
        semantic_digest: str,
        identity: str,
    ) -> None:
        if conn.execute(
            f"SELECT 1 FROM {table} WHERE semantic_digest=?",
            (semantic_digest,),
        ).fetchone() is not None:
            raise DiscoverySemanticCollision(
                f"{identity} already exists under another stable identity"
            )

    @classmethod
    def _validate_record_envelope(
        cls,
        conn: sqlite3.Connection,
        row: Mapping[str, Any],
        *,
        command_type: str,
        aggregate_id: str,
        canonical_bytes: bytes,
        canonical_digest: str,
    ) -> sqlite3.Row:
        spec = _DISCOVERY_RECORD_SPECS.get(command_type)
        if spec is None:
            return super()._validate_record_envelope(
                conn,
                row,
                command_type=command_type,
                aggregate_id=aggregate_id,
                canonical_bytes=canonical_bytes,
                canonical_digest=canonical_digest,
            )
        event = cls._record_context(conn, event_id=str(row["authority_event_id"]))
        aggregate_type, event_type, trust_scope = spec
        if (
            str(event["event_type"]) != event_type
            or int(event["event_schema_version"]) != 1
            or str(event["aggregate_type"]) != aggregate_type
            or str(event["aggregate_id"]) != aggregate_id
            or int(event["aggregate_version"])
            != int(row["authority_aggregate_version"])
            or int(row["authority_aggregate_version"]) != 1
            or str(event["recorded_at"]) != str(row["recorded_at"])
            or str(event["security_scope"]) != "authority.discovery"
            or str(event["retention_scope"]) != "authority.audit"
            or str(event["trust_scope"]) != trust_scope.value
            or str(event["payload_mode"]) != PayloadMode.INLINE.value
            or str(event["payload_digest"]) != canonical_digest
            or event["payload_bytes"] is None
            or bytes(event["payload_bytes"]) != canonical_bytes
            or digest_bytes(canonical_bytes) != canonical_digest
        ):
            raise AuthorityPersistenceError(
                "Signal/Lead record authority envelope is inconsistent"
            )
        return event

    @staticmethod
    def _row(conn: sqlite3.Connection, table: str, column: str, value: str) -> sqlite3.Row:
        row = conn.execute(
            f"SELECT * FROM {table} WHERE {column}=?", (value,)
        ).fetchone()
        if row is None:
            raise DiscoveryStateError(f"{table} record is not retained")
        return row

    # ------------------------------------------------------------------
    # Canonical decoders / normalized row tie-out
    # ------------------------------------------------------------------
    @classmethod
    def _signal_from_row(cls, conn: sqlite3.Connection, row: sqlite3.Row, *, replayed: bool) -> DiscoverySignal:
        data = bytes(row["canonical_bytes"])
        request = signal_request_from_bytes(data)
        expected = {
            "signal_id": str(request.signal_id),
            "definition_id": str(request.definition_id),
            "definition_version_id": str(request.definition_version_id),
            "item_id": str(request.item_id),
            "revision_id": str(request.revision_id),
            "representation_id": str(request.representation_id),
            "check_outcome_id": str(request.check_outcome_id),
            "occurrence_id": str(request.occurrence_id),
            "transition_id": str(request.transition_id),
            "purpose": request.purpose,
            "discriminator": request.discriminator,
            "admission_policy_id": request.admission_policy.policy_id,
            "admission_policy_version": request.admission_policy.policy_version,
            "incomplete": int(request.incomplete),
            "operational_finding_count": len(request.operational_finding_ids),
            "admitted_at": request.admitted_at.to_text(),
            "semantic_digest": request.semantic_digest,
            "canonical_digest": request.digest,
        }
        cls._require_normalized_columns(row, expected, identity="Discovery Signal")
        cls._require_canonical_blob(
            row,
            "operational_finding_ids_bytes",
            [str(value) for value in request.operational_finding_ids],
            identity="Discovery Signal",
        )
        children = tuple(
            str(value["finding_id"])
            for value in conn.execute(
                "SELECT finding_id FROM discovery_signal_findings "
                "WHERE signal_id=? ORDER BY finding_ordinal",
                (str(request.signal_id),),
            ).fetchall()
        )
        if children != tuple(str(value) for value in request.operational_finding_ids):
            raise AuthorityPersistenceError(
                "Discovery Signal Finding index differs from canonical bytes"
            )
        cls._validate_record_envelope(
            conn,
            row,
            command_type=DISCOVERY_SIGNAL_ADMIT_COMMAND,
            aggregate_id=str(request.signal_id),
            canonical_bytes=data,
            canonical_digest=request.digest,
        )
        return DiscoverySignal(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    @classmethod
    def _gate_from_row(cls, conn: sqlite3.Connection, row: sqlite3.Row, *, replayed: bool) -> GateDecision:
        data = bytes(row["canonical_bytes"])
        request = gate_request_from_bytes(data)
        expected = {
            "decision_id": str(request.decision_id),
            "signal_id": str(request.signal_id),
            "decision_ordinal": request.decision_ordinal,
            "previous_decision_id": None if request.previous_decision_id is None else str(request.previous_decision_id),
            "evaluated_definition_version_id": str(request.evaluated_definition_version_id),
            "coverage_obligation_id": request.coverage.obligation_id,
            "coverage_responsibility": request.coverage.responsibility.value,
            "coverage_contribution": request.coverage.contribution.value,
            "coverage_policy_id": request.coverage.coverage_policy.policy_id,
            "coverage_policy_version": request.coverage.coverage_policy.policy_version,
            "rights_decision_id": request.rights_decision_id,
            "rights_policy_version": request.rights_policy_version,
            "signal_admission_policy_id": request.signal_admission_policy.policy_id,
            "signal_admission_policy_version": request.signal_admission_policy.policy_version,
            "gate_policy_id": request.gate_policy.policy_id,
            "gate_policy_version": request.gate_policy.policy_version,
            "duplicate_policy_id": request.duplicate_policy.policy_id,
            "duplicate_policy_version": request.duplicate_policy.policy_version,
            "newness_policy_id": request.newness_policy.policy_id,
            "newness_policy_version": request.newness_policy.policy_version,
            "time_validity_policy_id": request.time_validity_policy.policy_id,
            "time_validity_policy_version": request.time_validity_policy.policy_version,
            "exclusion_policy_id": request.exclusion_policy.policy_id,
            "exclusion_policy_version": request.exclusion_policy.policy_version,
            "identity_integrity": int(request.basis.identity_integrity),
            "duplicate_signal_id": None if request.basis.duplicate_signal_id is None else str(request.basis.duplicate_signal_id),
            "duplicate_rule_id": None if request.basis.duplicate_rule is None else request.basis.duplicate_rule.policy_id,
            "duplicate_rule_version": None if request.basis.duplicate_rule is None else request.basis.duplicate_rule.policy_version,
            "observable_newness": request.basis.observable_newness.value,
            "time_validity": request.basis.time_validity.value,
            "scope_disposition": request.basis.scope_disposition.value,
            "clear_exclusion_rule_id": None if request.basis.clear_exclusion_rule is None else request.basis.clear_exclusion_rule.policy_id,
            "clear_exclusion_rule_version": None if request.basis.clear_exclusion_rule is None else request.basis.clear_exclusion_rule.policy_version,
            "rights_current": int(request.basis.rights_current),
            "policy_current": int(request.basis.policy_current),
            "operationally_executable": int(request.basis.operationally_executable),
            "ambiguity_count": len(request.basis.ambiguities),
            "outcome": request.outcome.value,
            "terminality": request.terminality.value,
            "supporting_reason_count": len(request.supporting_reasons),
            "reason_taxonomy_version": request.reason_taxonomy_version,
            "outcome_taxonomy_version": request.outcome_taxonomy_version,
            "next_action_kind": None if request.next_action is None else request.next_action.kind.value,
            "next_action_code": None if request.next_action is None else request.next_action.action_code,
            "decided_at": request.decided_at.to_text(),
            "semantic_digest": request.semantic_digest,
            "canonical_digest": request.digest,
        }
        cls._require_normalized_columns(row, expected, identity="Gate Decision")
        cls._require_canonical_blob(row, "ambiguities_bytes", list(request.basis.ambiguities), identity="Gate Decision")
        cls._require_canonical_blob(row, "primary_reason_bytes", request.primary_reason.canonical_value(), identity="Gate Decision")
        cls._require_canonical_blob(row, "supporting_reasons_bytes", [value.canonical_value() for value in request.supporting_reasons], identity="Gate Decision")
        if request.next_action is None:
            if row["next_action_bytes"] is not None:
                raise AuthorityPersistenceError("Gate Decision unexpected next-action bytes")
        else:
            cls._require_canonical_blob(row, "next_action_bytes", request.next_action.canonical_value(), identity="Gate Decision")
        cls._validate_record_envelope(
            conn,
            row,
            command_type=DISCOVERY_GATE_DECIDE_COMMAND,
            aggregate_id=str(request.decision_id),
            canonical_bytes=data,
            canonical_digest=request.digest,
        )
        return GateDecision(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    @classmethod
    def _lead_from_row(cls, conn: sqlite3.Connection, row: sqlite3.Row, *, replayed: bool) -> NewsLead:
        data = bytes(row["canonical_bytes"])
        request = lead_request_from_bytes(data)
        expected = {
            "lead_id": str(request.lead_id),
            "signal_id": str(request.signal_id),
            "promoting_gate_decision_id": str(request.promoting_gate_decision_id),
            "definition_id": str(request.definition_id),
            "definition_version_id": str(request.definition_version_id),
            "item_id": str(request.item_id),
            "revision_id": str(request.revision_id),
            "representation_id": str(request.representation_id),
            "occurrence_id": str(request.occurrence_id),
            "transition_id": str(request.transition_id),
            "transition_kind": request.transition_kind.value,
            "coverage_obligation_id": request.coverage.obligation_id,
            "coverage_responsibility": request.coverage.responsibility.value,
            "coverage_contribution": request.coverage.contribution.value,
            "coverage_policy_id": request.coverage.coverage_policy.policy_id,
            "coverage_policy_version": request.coverage.coverage_policy.policy_version,
            "source_role_count": len(request.source_roles),
            "portfolio_function_count": len(request.portfolio_functions),
            "source_dependency_count": len(request.source_dependencies),
            "incompleteness_warning_count": len(request.incompleteness_warnings),
            "urgency_route": request.urgency.route.value,
            "urgency_hard_deadline": None if request.urgency.hard_deadline is None else request.urgency.hard_deadline.to_text(),
            "urgency_planned_window": request.urgency.planned_window,
            "urgency_isolation_required": int(request.urgency.isolation_required),
            "lead_policy_id": request.lead_policy.policy_id,
            "lead_policy_version": request.lead_policy.policy_version,
            "reason_taxonomy_version": request.reason_taxonomy_version,
            "outcome_taxonomy_version": request.outcome_taxonomy_version,
            "created_at": request.created_at.to_text(),
            "semantic_digest": request.semantic_digest,
            "canonical_digest": request.digest,
        }
        cls._require_normalized_columns(row, expected, identity="News Lead")
        cls._require_canonical_blob(row, "source_roles_bytes", [value.canonical_value() for value in request.source_roles], identity="News Lead")
        cls._require_canonical_blob(row, "portfolio_functions_bytes", [value.value for value in request.portfolio_functions], identity="News Lead")
        cls._require_canonical_blob(row, "source_dependencies_bytes", [value.canonical_value() for value in request.source_dependencies], identity="News Lead")
        cls._require_canonical_blob(row, "incompleteness_warnings_bytes", list(request.incompleteness_warnings), identity="News Lead")
        cls._require_canonical_blob(row, "urgency_bytes", request.urgency.canonical_value(), identity="News Lead")
        cls._validate_record_envelope(
            conn,
            row,
            command_type=DISCOVERY_LEAD_OPEN_COMMAND,
            aggregate_id=str(request.lead_id),
            canonical_bytes=data,
            canonical_digest=request.digest,
        )
        return NewsLead(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    @classmethod
    def _watch_from_row(cls, conn: sqlite3.Connection, row: sqlite3.Row, *, replayed: bool) -> WatchCondition:
        data = bytes(row["canonical_bytes"])
        request = watch_request_from_bytes(data)
        expected = {
            "watch_condition_id": str(request.watch_condition_id),
            "lead_id": str(request.lead_id),
            "gate_decision_id": str(request.gate_decision_id),
            "resume_transition_kind_count": len(request.resume_transition_kinds),
            "expected_occurrence": request.expected_occurrence,
            "corroborating_lead_id": None if request.corroborating_lead_id is None else str(request.corroborating_lead_id),
            "review_at": None if request.review_at is None else request.review_at.to_text(),
            "expires_at": None if request.expires_at is None else request.expires_at.to_text(),
            "operator_review_condition": request.operator_review_condition,
            "closure_rule": request.closure_rule,
            "watch_policy_id": request.watch_policy.policy_id,
            "watch_policy_version": request.watch_policy.policy_version,
            "condition_recorded_at": request.recorded_at.to_text(),
            "semantic_digest": request.semantic_digest,
            "canonical_digest": request.digest,
        }
        cls._require_normalized_columns(row, expected, identity="Watch Condition")
        cls._require_canonical_blob(row, "resume_transition_kinds_bytes", [value.value for value in request.resume_transition_kinds], identity="Watch Condition")
        cls._validate_record_envelope(
            conn,
            row,
            command_type=DISCOVERY_WATCH_CONDITION_RECORD_COMMAND,
            aggregate_id=str(request.watch_condition_id),
            canonical_bytes=data,
            canonical_digest=request.digest,
        )
        return WatchCondition(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    @classmethod
    def _disposition_from_row(cls, conn: sqlite3.Connection, row: sqlite3.Row, *, replayed: bool) -> LeadDispositionDecision:
        data = bytes(row["canonical_bytes"])
        request = disposition_request_from_bytes(data)
        expected = {
            "decision_id": str(request.decision_id),
            "lead_id": str(request.lead_id),
            "gate_decision_id": str(request.gate_decision_id),
            "decision_ordinal": request.decision_ordinal,
            "previous_decision_id": None if request.previous_decision_id is None else str(request.previous_decision_id),
            "outcome": request.outcome.value,
            "terminality": request.terminality.value,
            "supporting_reason_count": len(request.supporting_reasons),
            "watch_condition_id": None if request.watch_condition_id is None else str(request.watch_condition_id),
            "next_action_kind": request.next_action.kind.value,
            "next_action_code": request.next_action.action_code,
            "urgency_route": request.urgency_route.route.value,
            "disposition_policy_id": request.disposition_policy.policy_id,
            "disposition_policy_version": request.disposition_policy.policy_version,
            "reason_taxonomy_version": request.reason_taxonomy_version,
            "outcome_taxonomy_version": request.outcome_taxonomy_version,
            "decided_at": request.decided_at.to_text(),
            "semantic_digest": request.semantic_digest,
            "canonical_digest": request.digest,
        }
        cls._require_normalized_columns(row, expected, identity="Lead Disposition")
        cls._require_canonical_blob(row, "primary_reason_bytes", request.primary_reason.canonical_value(), identity="Lead Disposition")
        cls._require_canonical_blob(row, "supporting_reasons_bytes", [value.canonical_value() for value in request.supporting_reasons], identity="Lead Disposition")
        cls._require_canonical_blob(row, "next_action_bytes", request.next_action.canonical_value(), identity="Lead Disposition")
        cls._require_canonical_blob(row, "urgency_bytes", request.urgency_route.canonical_value(), identity="Lead Disposition")
        cls._validate_record_envelope(
            conn,
            row,
            command_type=DISCOVERY_LEAD_DISPOSITION_RECORD_COMMAND,
            aggregate_id=str(request.decision_id),
            canonical_bytes=data,
            canonical_digest=request.digest,
        )
        return LeadDispositionDecision(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    # ------------------------------------------------------------------
    # Public private-store reads
    # ------------------------------------------------------------------
    def discovery_signal(self, signal_id: DiscoverySignalId) -> DiscoverySignal | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM discovery_signals WHERE signal_id=?",
                (str(signal_id),),
            ).fetchone()
            return None if row is None else self._signal_from_row(self._connection, row, replayed=False)

    def gate_decision(self, decision_id: GateDecisionId) -> GateDecision | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM discovery_gate_decisions WHERE decision_id=?",
                (str(decision_id),),
            ).fetchone()
            return None if row is None else self._gate_from_row(self._connection, row, replayed=False)

    def current_gate_decision(self, signal_id: DiscoverySignalId) -> GateDecision | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT d.* FROM discovery_gate_decision_heads h "
                "JOIN discovery_gate_decisions d ON d.decision_id=h.current_decision_id "
                "WHERE h.signal_id=?",
                (str(signal_id),),
            ).fetchone()
            return None if row is None else self._gate_from_row(self._connection, row, replayed=False)

    def gate_decisions_for_signal(self, signal_id: DiscoverySignalId, limit: int) -> tuple[GateDecision, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM discovery_gate_decisions WHERE signal_id=? "
                "ORDER BY decision_ordinal LIMIT ?",
                (str(signal_id), limit),
            ).fetchall()
            return tuple(self._gate_from_row(self._connection, row, replayed=False) for row in rows)

    def news_lead(self, lead_id: NewsLeadId) -> NewsLead | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM news_leads WHERE lead_id=?", (str(lead_id),)
            ).fetchone()
            return None if row is None else self._lead_from_row(self._connection, row, replayed=False)

    def lead_for_signal(self, signal_id: DiscoverySignalId) -> NewsLead | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM news_leads WHERE signal_id=?", (str(signal_id),)
            ).fetchone()
            return None if row is None else self._lead_from_row(self._connection, row, replayed=False)

    def watch_condition(self, watch_id: WatchConditionId) -> WatchCondition | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM discovery_watch_conditions WHERE watch_condition_id=?",
                (str(watch_id),),
            ).fetchone()
            return None if row is None else self._watch_from_row(self._connection, row, replayed=False)

    def lead_disposition(self, decision_id: LeadDispositionDecisionId) -> LeadDispositionDecision | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM lead_disposition_decisions WHERE decision_id=?",
                (str(decision_id),),
            ).fetchone()
            return None if row is None else self._disposition_from_row(self._connection, row, replayed=False)

    def current_lead_disposition(self, lead_id: NewsLeadId) -> LeadDispositionDecision | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT d.* FROM lead_disposition_heads h "
                "JOIN lead_disposition_decisions d ON d.decision_id=h.current_decision_id "
                "JOIN news_leads l ON l.lead_id=h.lead_id "
                "JOIN discovery_gate_decision_heads g "
                "ON g.signal_id=l.signal_id "
                "AND g.current_decision_id=d.gate_decision_id "
                "WHERE h.lead_id=?",
                (str(lead_id),),
            ).fetchone()
            return None if row is None else self._disposition_from_row(self._connection, row, replayed=False)

    def latest_lead_disposition(self, lead_id: NewsLeadId) -> LeadDispositionDecision | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT d.* FROM lead_disposition_heads h "
                "JOIN lead_disposition_decisions d "
                "ON d.decision_id=h.current_decision_id WHERE h.lead_id=?",
                (str(lead_id),),
            ).fetchone()
            return None if row is None else self._disposition_from_row(
                self._connection, row, replayed=False
            )

    def lead_dispositions(self, lead_id: NewsLeadId, limit: int) -> tuple[LeadDispositionDecision, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM lead_disposition_decisions WHERE lead_id=? "
                "ORDER BY decision_ordinal LIMIT ?",
                (str(lead_id), limit),
            ).fetchall()
            return tuple(self._disposition_from_row(self._connection, row, replayed=False) for row in rows)

    def signals_for_revision(self, revision_id: str, limit: int) -> tuple[DiscoverySignal, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM discovery_signals WHERE revision_id=? "
                "ORDER BY admitted_at,signal_id LIMIT ?",
                (revision_id, limit),
            ).fetchall()
            return tuple(self._signal_from_row(self._connection, row, replayed=False) for row in rows)

    def discovery_current_status(self, signal_id: DiscoverySignalId) -> DiscoveryCurrentStatus:
        with self._lock:
            signal_row = self._row(self._connection, "discovery_signals", "signal_id", str(signal_id))
            signal = self._signal_from_row(self._connection, signal_row, replayed=False)
            gate_row = self._connection.execute(
                "SELECT d.* FROM discovery_gate_decision_heads h "
                "JOIN discovery_gate_decisions d ON d.decision_id=h.current_decision_id "
                "WHERE h.signal_id=?",
                (str(signal_id),),
            ).fetchone()
            if gate_row is None:
                raise DiscoveryStateError("Signal has no current Gate Decision")
            gate = self._gate_from_row(self._connection, gate_row, replayed=False)
            # A later Gate Decision is authoritative for current executability.
            # Historical Leads remain inspectable, but a current suppression or
            # operational hold must not expose the old Lead disposition as the
            # current action.
            lead_row = self._connection.execute(
                "SELECT * FROM news_leads WHERE signal_id=?", (str(signal_id),)
            ).fetchone()
            if (
                gate.request.outcome is not GateOutcome.PROMOTED_TO_LEAD
                or lead_row is None
            ):
                if gate.request.outcome is GateOutcome.OPERATIONAL_HOLD:
                    phase = DiscoveryCurrentPhase.SIGNAL_OPERATIONAL_HOLD
                elif gate.request.outcome in {
                    GateOutcome.SUPPRESSED_DUPLICATE,
                    GateOutcome.SUPPRESSED_NON_CHANGE,
                    GateOutcome.REJECTED_CLEAR_EXCLUSION,
                }:
                    phase = DiscoveryCurrentPhase.SIGNAL_SUPPRESSED
                else:
                    phase = DiscoveryCurrentPhase.SIGNAL_ADMITTED
                return DiscoveryCurrentStatus(
                    signal=signal,
                    current_gate=gate,
                    lead=None,
                    current_disposition=None,
                    watch_condition=None,
                    phase=phase,
                    action_source=DiscoveryCurrentActionSource.GATE_DECISION,
                    next_action=gate.request.next_action,
                    urgency_route=None,
                )
            lead = self._lead_from_row(self._connection, lead_row, replayed=False)
            disposition_row = self._connection.execute(
                "SELECT d.* FROM lead_disposition_heads h "
                "JOIN lead_disposition_decisions d ON d.decision_id=h.current_decision_id "
                "WHERE h.lead_id=? AND d.gate_decision_id=?",
                (str(lead.request.lead_id), str(gate.request.decision_id)),
            ).fetchone()
            if disposition_row is None:
                return DiscoveryCurrentStatus(
                    signal=signal,
                    current_gate=gate,
                    lead=lead,
                    current_disposition=None,
                    watch_condition=None,
                    phase=DiscoveryCurrentPhase.LEAD_QUEUED,
                    action_source=DiscoveryCurrentActionSource.GATE_DECISION,
                    next_action=gate.request.next_action,
                    urgency_route=lead.request.urgency.route,
                )
            disposition = self._disposition_from_row(self._connection, disposition_row, replayed=False)
            watch = None
            if disposition.request.watch_condition_id is not None:
                watch_row = self._row(
                    self._connection,
                    "discovery_watch_conditions",
                    "watch_condition_id",
                    str(disposition.request.watch_condition_id),
                )
                watch = self._watch_from_row(self._connection, watch_row, replayed=False)
            phase = {
                LeadDispositionOutcome.QUEUED_FOR_TRIAGE: DiscoveryCurrentPhase.LEAD_QUEUED,
                LeadDispositionOutcome.OPERATIONAL_HOLD: DiscoveryCurrentPhase.LEAD_OPERATIONAL_HOLD,
                LeadDispositionOutcome.WATCH_DEFER: DiscoveryCurrentPhase.LEAD_WATCH_DEFER,
            }[disposition.request.outcome]
            return DiscoveryCurrentStatus(
                signal=signal,
                current_gate=gate,
                lead=lead,
                current_disposition=disposition,
                watch_condition=watch,
                phase=phase,
                action_source=DiscoveryCurrentActionSource.LEAD_DISPOSITION,
                next_action=disposition.request.next_action,
                urgency_route=lead.request.urgency.route,
            )

    # ------------------------------------------------------------------
    # Lineage validators used by commits
    # ------------------------------------------------------------------
    @staticmethod
    def _require_exact_signal_lineage(conn: sqlite3.Connection, request: DiscoverySignalRequest) -> None:
        version = conn.execute(
            "SELECT * FROM source_definition_versions WHERE version_id=? AND definition_id=?",
            (str(request.definition_version_id), str(request.definition_id)),
        ).fetchone()
        item = conn.execute(
            "SELECT * FROM source_items WHERE item_id=? AND definition_id=?",
            (str(request.item_id), str(request.definition_id)),
        ).fetchone()
        revision = conn.execute(
            "SELECT * FROM source_revisions WHERE revision_id=? AND item_id=? AND definition_id=?",
            (str(request.revision_id), str(request.item_id), str(request.definition_id)),
        ).fetchone()
        representation = conn.execute(
            "SELECT * FROM discovery_representations WHERE representation_id=? "
            "AND revision_id=? AND definition_id=? AND definition_version_id=?",
            (
                str(request.representation_id),
                str(request.revision_id),
                str(request.definition_id),
                str(request.definition_version_id),
            ),
        ).fetchone()
        outcome = conn.execute(
            "SELECT * FROM check_outcomes WHERE outcome_id=? AND definition_id=? AND definition_version_id=?",
            (
                str(request.check_outcome_id),
                str(request.definition_id),
                str(request.definition_version_id),
            ),
        ).fetchone()
        occurrence = conn.execute(
            "SELECT * FROM discovery_occurrences WHERE occurrence_id=? "
            "AND check_outcome_id=? AND revision_id=? AND representation_id=? "
            "AND definition_id=? AND definition_version_id=?",
            (
                str(request.occurrence_id),
                str(request.check_outcome_id),
                str(request.revision_id),
                str(request.representation_id),
                str(request.definition_id),
                str(request.definition_version_id),
            ),
        ).fetchone()
        transition = conn.execute(
            "SELECT * FROM observable_transitions WHERE transition_id=? "
            "AND check_outcome_id=? AND item_id=? AND current_revision_id=? "
            "AND representation_id=? AND definition_id=? AND definition_version_id=?",
            (
                str(request.transition_id),
                str(request.check_outcome_id),
                str(request.item_id),
                str(request.revision_id),
                str(request.representation_id),
                str(request.definition_id),
                str(request.definition_version_id),
            ),
        ).fetchone()
        if None in (version, item, revision, representation, outcome, occurrence, transition):
            raise DiscoveryStateError("Discovery Signal source-transition lineage is incomplete")
        if str(outcome["kind"]) not in {
            "SUCCESS_UNCHANGED",
            "SUCCESS_CHANGED",
            "SUCCESS_PARTIAL",
            "SUCCESS_TRUNCATED",
        } or bool(outcome["incomplete"]) != request.incomplete:
            raise DiscoveryVersionConflict("Signal Check Outcome is not an admissible observation")
        if str(version["recorded_at"]) > str(outcome["completed_at"]):
            raise DiscoveryVersionConflict(
                "Signal source version was not retained before its observation"
            )
        if request.admitted_at.to_text() < max(
            str(outcome["completed_at"]), str(transition["observed_at"])
        ):
            raise DiscoveryVersionConflict("Signal admission precedes its exact evidence")
        finding_rows = conn.execute(
            "SELECT DISTINCT f.finding_id FROM operational_findings f "
            "LEFT JOIN operational_finding_occurrences o "
            "ON o.finding_id=f.finding_id AND o.outcome_id=? "
            "WHERE f.opened_by_outcome_id=? OR o.finding_id IS NOT NULL "
            "ORDER BY f.finding_id",
            (str(request.check_outcome_id), str(request.check_outcome_id)),
        ).fetchall()
        exact_findings = tuple(str(row["finding_id"]) for row in finding_rows)
        supplied_findings = tuple(str(value) for value in request.operational_finding_ids)
        if supplied_findings != exact_findings:
            raise DiscoveryVersionConflict(
                "Signal Operational Finding lineage differs from exact Outcome findings"
            )

    @classmethod
    def _require_source_contract_matches_lead(cls, conn: sqlite3.Connection, request: NewsLeadRequest) -> None:
        cls._require_current_version(
            conn,
            definition_id=request.definition_id,
            version_id=request.definition_version_id,
        )
        roles = tuple(
            cls._canonical_child(row, identity="source role")
            for row in conn.execute(
                "SELECT * FROM source_version_roles WHERE version_id=? ORDER BY role",
                (str(request.definition_version_id),),
            ).fetchall()
        )
        functions = tuple(
            str(row["portfolio_function"])
            for row in conn.execute(
                "SELECT portfolio_function FROM source_version_portfolio_functions "
                "WHERE version_id=? ORDER BY portfolio_function",
                (str(request.definition_version_id),),
            ).fetchall()
        )
        dependencies = tuple(
            cls._canonical_child(row, identity="source dependency")
            for row in conn.execute(
                "SELECT * FROM source_version_dependencies WHERE version_id=? ORDER BY dependency_id",
                (str(request.definition_version_id),),
            ).fetchall()
        )
        if roles != tuple(value.canonical_value() for value in request.source_roles):
            raise DiscoveryVersionConflict("Lead source roles differ from exact source version")
        if functions != tuple(value.value for value in request.portfolio_functions):
            raise DiscoveryVersionConflict("Lead portfolio functions differ from exact source version")
        if dependencies != tuple(value.canonical_value() for value in request.source_dependencies):
            raise DiscoveryVersionConflict("Lead dependencies differ from exact source version")

    @staticmethod
    def _source_versions_observation_compatible(
        original: sqlite3.Row,
        evaluated: sqlite3.Row,
    ) -> bool:
        """Return whether later source configuration preserves observed identity.

        Locator, rights, coverage, source-role and portfolio changes are exactly
        what the Gate re-evaluates.  Collection, parsing, item identity,
        Revision equality, canonicalization, observation-model and baseline
        changes require an explicit hold because the retained Representation
        was not produced under the new contract.
        """

        return all(
            original[column] == evaluated[column]
            for column in _SOURCE_OBSERVATION_COMPATIBILITY_COLUMNS
        )

    # ------------------------------------------------------------------
    # Commit methods
    # ------------------------------------------------------------------
    def commit_discovery_signal(self, grant: _AuthorizedCommandGrant, *, request: DiscoverySignalRequest) -> DiscoverySignal:
        if not isinstance(request, DiscoverySignalRequest):
            raise TypeError("Discovery Signal commit requires a typed request")
        self._require_discovery_grant(
            grant,
            command_type=DISCOVERY_SIGNAL_ADMIT_COMMAND,
            aggregate_id=str(request.signal_id),
            canonical_bytes=request.canonical_bytes,
        )
        with self._lock, self._transaction() as conn:
            if grant.replay_of_command_id is not None:
                committed = self._commit_grant_in_transaction(conn, grant, recorded_at=self._clock().to_text())
                row = self._row(conn, "discovery_signals", "authority_event_id", committed.event_id)
                return self._signal_from_row(conn, row, replayed=True)
            self._require_exact_signal_lineage(conn, request)
            self._discovery_identifier_absent(conn, table="discovery_signals", column="signal_id", identifier=str(request.signal_id), identity="Discovery Signal identity")
            self._discovery_semantic_absent(conn, table="discovery_signals", semantic_digest=request.semantic_digest, identity="Discovery Signal semantics")
            recorded_at = self._clock().to_text()
            committed = self._commit_grant_in_transaction(conn, grant, recorded_at=recorded_at)
            if committed.replayed:
                row = self._row(conn, "discovery_signals", "authority_event_id", committed.event_id)
                return self._signal_from_row(conn, row, replayed=True)
            try:
                conn.execute(
                    "INSERT INTO discovery_signals("
                    "signal_id,definition_id,definition_version_id,item_id,revision_id,representation_id,"
                    "check_outcome_id,occurrence_id,transition_id,purpose,discriminator,admission_policy_id,"
                    "admission_policy_version,incomplete,operational_finding_ids_bytes,operational_finding_count,"
                    "admitted_at,semantic_digest,authority_event_id,authority_aggregate_version,canonical_bytes,"
                    "canonical_digest,recorded_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        str(request.signal_id), str(request.definition_id), str(request.definition_version_id),
                        str(request.item_id), str(request.revision_id), str(request.representation_id),
                        str(request.check_outcome_id), str(request.occurrence_id), str(request.transition_id),
                        request.purpose, request.discriminator, request.admission_policy.policy_id,
                        request.admission_policy.policy_version, int(request.incomplete),
                        canonical_json_bytes([str(value) for value in request.operational_finding_ids]),
                        len(request.operational_finding_ids), request.admitted_at.to_text(), request.semantic_digest,
                        committed.event_id, committed.aggregate_version, request.canonical_bytes, request.digest,
                        recorded_at,
                    ),
                )
                for ordinal, finding_id in enumerate(request.operational_finding_ids):
                    conn.execute(
                        "INSERT INTO discovery_signal_findings(signal_id,finding_id,finding_ordinal) VALUES(?,?,?)",
                        (str(request.signal_id), str(finding_id), ordinal),
                    )
            except sqlite3.IntegrityError as exc:
                raise DiscoveryStateError("Discovery Signal persistence rejected inconsistent lineage") from exc
            row = self._row(conn, "discovery_signals", "authority_event_id", committed.event_id)
            return self._signal_from_row(conn, row, replayed=False)

    def commit_gate_decision(self, grant: _AuthorizedCommandGrant, *, request: GateDecisionRequest) -> GateDecision:
        if not isinstance(request, GateDecisionRequest):
            raise TypeError("Gate Decision commit requires a typed request")
        self._require_discovery_grant(
            grant,
            command_type=DISCOVERY_GATE_DECIDE_COMMAND,
            aggregate_id=str(request.decision_id),
            canonical_bytes=request.canonical_bytes,
        )
        with self._lock, self._transaction() as conn:
            if grant.replay_of_command_id is not None:
                committed = self._commit_grant_in_transaction(conn, grant, recorded_at=self._clock().to_text())
                row = self._row(conn, "discovery_gate_decisions", "authority_event_id", committed.event_id)
                return self._gate_from_row(conn, row, replayed=True)
            signal = self._row(conn, "discovery_signals", "signal_id", str(request.signal_id))
            if (
                str(signal["admission_policy_id"]) != request.signal_admission_policy.policy_id
                or str(signal["admission_policy_version"]) != request.signal_admission_policy.policy_version
                or request.decided_at.to_text() < str(signal["admitted_at"])
            ):
                raise DiscoveryVersionConflict("Gate Decision differs from exact Signal authority")
            original_version = self._row(
                conn,
                "source_definition_versions",
                "version_id",
                str(signal["definition_version_id"]),
            )
            version = self._row(
                conn,
                "source_definition_versions",
                "version_id",
                str(request.evaluated_definition_version_id),
            )
            if str(version["definition_id"]) != str(signal["definition_id"]):
                raise DiscoveryVersionConflict("Gate evaluated source version belongs to another source")
            if request.decided_at.to_text() < str(version["recorded_at"]):
                raise DiscoveryVersionConflict(
                    "Gate Decision precedes its evaluated source version"
                )
            if (
                str(version["rights_decision_id"]) != request.rights_decision_id
                or str(version["rights_policy_version"]) != request.rights_policy_version
            ):
                raise DiscoveryVersionConflict("Gate rights basis differs from source version")
            compatible = self._source_versions_observation_compatible(
                original_version,
                version,
            )
            if not compatible and request.basis.identity_integrity:
                raise DiscoveryVersionConflict(
                    "Gate cannot claim identity integrity across an incompatible "
                    "source observation contract"
                )
            if request.basis.policy_current:
                self._require_current_version(
                    conn,
                    definition_id=SourceDefinitionId.parse(str(signal["definition_id"])),
                    version_id=request.evaluated_definition_version_id,
                )
            coverage = conn.execute(
                "SELECT 1 FROM source_version_coverage_mappings WHERE version_id=? "
                "AND obligation_id=? AND responsibility=? AND contribution=?",
                (
                    str(request.evaluated_definition_version_id), request.coverage.obligation_id,
                    request.coverage.responsibility.value, request.coverage.contribution.value,
                ),
            ).fetchone()
            if coverage is None:
                raise DiscoveryVersionConflict("Gate coverage basis differs from source version")
            transition = self._row(conn, "observable_transitions", "transition_id", str(signal["transition_id"]))
            if request.basis.observable_newness not in permitted_newness_for_transition(
                __import__("newsroom.checks", fromlist=["ObservableTransitionKind"]).ObservableTransitionKind(str(transition["kind"]))
            ):
                raise DiscoveryVersionConflict("Gate newness class conflicts with source transition")
            if request.outcome is not deterministic_gate_outcome(request.basis):
                raise DiscoveryVersionConflict("Gate outcome differs from deterministic basis")
            if request.basis.duplicate_signal_id is not None:
                duplicate = self._row(conn, "discovery_signals", "signal_id", str(request.basis.duplicate_signal_id))
                if (
                    str(duplicate["definition_id"]) != str(signal["definition_id"])
                    or str(duplicate["item_id"]) != str(signal["item_id"])
                    or str(duplicate["revision_id"]) != str(signal["revision_id"])
                    or str(duplicate["purpose"]) != str(signal["purpose"])
                    or str(duplicate["discriminator"])
                    != str(signal["discriminator"])
                ):
                    raise DiscoveryVersionConflict(
                        "cross-source, cross-state, or distinct-purpose Signals "
                        "cannot be suppressed as an exact duplicate"
                    )
                duplicate_event = self._record_context(
                    conn,
                    event_id=str(duplicate["authority_event_id"]),
                )
                signal_event = self._record_context(
                    conn,
                    event_id=str(signal["authority_event_id"]),
                )
                if int(duplicate_event["ledger_seq"]) >= int(
                    signal_event["ledger_seq"]
                ):
                    raise DiscoveryVersionConflict(
                        "duplicate suppression requires an earlier retained Signal"
                    )
            head = conn.execute(
                "SELECT current_decision_id,current_decision_ordinal FROM discovery_gate_decision_heads WHERE signal_id=?",
                (str(request.signal_id),),
            ).fetchone()
            expected_ordinal = 1 if head is None else int(head["current_decision_ordinal"]) + 1
            expected_previous = None if head is None else str(head["current_decision_id"])
            actual_previous = None if request.previous_decision_id is None else str(request.previous_decision_id)
            if request.decision_ordinal != expected_ordinal or actual_previous != expected_previous:
                raise DiscoveryVersionConflict("Gate Decision does not extend exact current head")
            self._discovery_identifier_absent(conn, table="discovery_gate_decisions", column="decision_id", identifier=str(request.decision_id), identity="Gate Decision identity")
            self._discovery_semantic_absent(conn, table="discovery_gate_decisions", semantic_digest=request.semantic_digest, identity="Gate Decision semantics")
            recorded_at = self._clock().to_text()
            committed = self._commit_grant_in_transaction(conn, grant, recorded_at=recorded_at)
            if committed.replayed:
                row = self._row(conn, "discovery_gate_decisions", "authority_event_id", committed.event_id)
                return self._gate_from_row(conn, row, replayed=True)
            basis = request.basis
            action = request.next_action
            try:
                conn.execute(
                    "INSERT INTO discovery_gate_decisions("
                    "decision_id,signal_id,decision_ordinal,previous_decision_id,evaluated_definition_version_id,"
                    "coverage_obligation_id,coverage_responsibility,coverage_contribution,coverage_policy_id,coverage_policy_version,"
                    "rights_decision_id,rights_policy_version,signal_admission_policy_id,signal_admission_policy_version,"
                    "gate_policy_id,gate_policy_version,duplicate_policy_id,duplicate_policy_version,newness_policy_id,newness_policy_version,"
                    "time_validity_policy_id,time_validity_policy_version,exclusion_policy_id,exclusion_policy_version,"
                    "identity_integrity,duplicate_signal_id,duplicate_rule_id,duplicate_rule_version,observable_newness,time_validity,"
                    "scope_disposition,clear_exclusion_rule_id,clear_exclusion_rule_version,rights_current,policy_current,"
                    "operationally_executable,ambiguities_bytes,ambiguity_count,outcome,terminality,primary_reason_bytes,"
                    "supporting_reasons_bytes,supporting_reason_count,reason_taxonomy_version,outcome_taxonomy_version,"
                    "next_action_kind,next_action_code,next_action_bytes,decided_at,semantic_digest,authority_event_id,"
                    "authority_aggregate_version,canonical_bytes,canonical_digest,recorded_at) "
                    "VALUES(" + ",".join("?" for _ in range(55)) + ")",
                    (
                        str(request.decision_id), str(request.signal_id), request.decision_ordinal, actual_previous,
                        str(request.evaluated_definition_version_id), request.coverage.obligation_id,
                        request.coverage.responsibility.value, request.coverage.contribution.value,
                        request.coverage.coverage_policy.policy_id, request.coverage.coverage_policy.policy_version,
                        request.rights_decision_id, request.rights_policy_version,
                        request.signal_admission_policy.policy_id, request.signal_admission_policy.policy_version,
                        request.gate_policy.policy_id, request.gate_policy.policy_version,
                        request.duplicate_policy.policy_id, request.duplicate_policy.policy_version,
                        request.newness_policy.policy_id, request.newness_policy.policy_version,
                        request.time_validity_policy.policy_id, request.time_validity_policy.policy_version,
                        request.exclusion_policy.policy_id, request.exclusion_policy.policy_version,
                        int(basis.identity_integrity), None if basis.duplicate_signal_id is None else str(basis.duplicate_signal_id),
                        None if basis.duplicate_rule is None else basis.duplicate_rule.policy_id,
                        None if basis.duplicate_rule is None else basis.duplicate_rule.policy_version,
                        basis.observable_newness.value, basis.time_validity.value, basis.scope_disposition.value,
                        None if basis.clear_exclusion_rule is None else basis.clear_exclusion_rule.policy_id,
                        None if basis.clear_exclusion_rule is None else basis.clear_exclusion_rule.policy_version,
                        int(basis.rights_current), int(basis.policy_current), int(basis.operationally_executable),
                        canonical_json_bytes(list(basis.ambiguities)), len(basis.ambiguities), request.outcome.value,
                        request.terminality.value, canonical_json_bytes(request.primary_reason.canonical_value()),
                        canonical_json_bytes([value.canonical_value() for value in request.supporting_reasons]),
                        len(request.supporting_reasons), request.reason_taxonomy_version, request.outcome_taxonomy_version,
                        None if action is None else action.kind.value, None if action is None else action.action_code,
                        None if action is None else canonical_json_bytes(action.canonical_value()), request.decided_at.to_text(),
                        request.semantic_digest, committed.event_id, committed.aggregate_version, request.canonical_bytes,
                        request.digest, recorded_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DiscoveryStateError("Gate Decision persistence rejected inconsistent authority") from exc
            row = self._row(conn, "discovery_gate_decisions", "authority_event_id", committed.event_id)
            return self._gate_from_row(conn, row, replayed=False)

    def commit_news_lead(self, grant: _AuthorizedCommandGrant, *, request: NewsLeadRequest) -> NewsLead:
        if not isinstance(request, NewsLeadRequest):
            raise TypeError("News Lead commit requires a typed request")
        self._require_discovery_grant(
            grant,
            command_type=DISCOVERY_LEAD_OPEN_COMMAND,
            aggregate_id=str(request.lead_id),
            canonical_bytes=request.canonical_bytes,
        )
        with self._lock, self._transaction() as conn:
            if grant.replay_of_command_id is not None:
                committed = self._commit_grant_in_transaction(conn, grant, recorded_at=self._clock().to_text())
                row = self._row(conn, "news_leads", "authority_event_id", committed.event_id)
                return self._lead_from_row(conn, row, replayed=True)
            signal = self._row(conn, "discovery_signals", "signal_id", str(request.signal_id))
            gate = self._row(conn, "discovery_gate_decisions", "decision_id", str(request.promoting_gate_decision_id))
            head = conn.execute(
                "SELECT current_decision_id FROM discovery_gate_decision_heads WHERE signal_id=?",
                (str(request.signal_id),),
            ).fetchone()
            if head is None or str(head["current_decision_id"]) != str(request.promoting_gate_decision_id):
                raise DiscoveryVersionConflict("News Lead requires the current promoting Gate Decision")
            if str(gate["outcome"]) != GateOutcome.PROMOTED_TO_LEAD.value:
                raise DiscoveryVersionConflict("News Lead Gate Decision is not a promotion")
            exact = {
                "definition_id": str(request.definition_id),
                "item_id": str(request.item_id),
                "revision_id": str(request.revision_id),
                "representation_id": str(request.representation_id),
                "occurrence_id": str(request.occurrence_id),
                "transition_id": str(request.transition_id),
            }
            if any(str(signal[key]) != value for key, value in exact.items()):
                raise DiscoveryVersionConflict("News Lead lineage differs from promoted Signal")
            if str(gate["evaluated_definition_version_id"]) != str(
                request.definition_version_id
            ):
                raise DiscoveryVersionConflict(
                    "News Lead source version differs from the promoting Gate"
                )
            if request.created_at.to_text() < str(gate["decided_at"]):
                raise DiscoveryVersionConflict(
                    "News Lead creation precedes its promoting Gate"
                )
            if (
                str(gate["coverage_obligation_id"]) != request.coverage.obligation_id
                or str(gate["coverage_responsibility"]) != request.coverage.responsibility.value
                or str(gate["coverage_contribution"]) != request.coverage.contribution.value
                or str(gate["coverage_policy_id"]) != request.coverage.coverage_policy.policy_id
                or str(gate["coverage_policy_version"]) != request.coverage.coverage_policy.policy_version
            ):
                raise DiscoveryVersionConflict("News Lead coverage differs from promoting Gate")
            transition = self._row(conn, "observable_transitions", "transition_id", str(request.transition_id))
            if str(transition["kind"]) != request.transition_kind.value:
                raise DiscoveryVersionConflict("News Lead transition kind differs from exact transition")
            self._require_source_contract_matches_lead(conn, request)
            if bool(signal["incomplete"]) and not request.incompleteness_warnings:
                raise DiscoveryVersionConflict("incomplete Signal requires visible Lead warning")
            self._discovery_identifier_absent(conn, table="news_leads", column="lead_id", identifier=str(request.lead_id), identity="News Lead identity")
            if conn.execute("SELECT 1 FROM news_leads WHERE signal_id=?", (str(request.signal_id),)).fetchone() is not None:
                raise DiscoverySemanticCollision("promoted Signal already owns one News Lead")
            self._discovery_semantic_absent(conn, table="news_leads", semantic_digest=request.semantic_digest, identity="News Lead semantics")
            recorded_at = self._clock().to_text()
            committed = self._commit_grant_in_transaction(conn, grant, recorded_at=recorded_at)
            if committed.replayed:
                row = self._row(conn, "news_leads", "authority_event_id", committed.event_id)
                return self._lead_from_row(conn, row, replayed=True)
            try:
                conn.execute(
                    "INSERT INTO news_leads("
                    "lead_id,signal_id,promoting_gate_decision_id,definition_id,definition_version_id,item_id,revision_id,"
                    "representation_id,occurrence_id,transition_id,transition_kind,coverage_obligation_id,coverage_responsibility,"
                    "coverage_contribution,coverage_policy_id,coverage_policy_version,source_roles_bytes,source_role_count,"
                    "portfolio_functions_bytes,portfolio_function_count,source_dependencies_bytes,source_dependency_count,"
                    "incompleteness_warnings_bytes,incompleteness_warning_count,urgency_bytes,urgency_route,urgency_hard_deadline,"
                    "urgency_planned_window,urgency_isolation_required,lead_policy_id,lead_policy_version,reason_taxonomy_version,"
                    "outcome_taxonomy_version,created_at,semantic_digest,authority_event_id,authority_aggregate_version,"
                    "canonical_bytes,canonical_digest,recorded_at) VALUES(" + ",".join("?" for _ in range(40)) + ")",
                    (
                        str(request.lead_id), str(request.signal_id), str(request.promoting_gate_decision_id),
                        str(request.definition_id), str(request.definition_version_id), str(request.item_id),
                        str(request.revision_id), str(request.representation_id), str(request.occurrence_id),
                        str(request.transition_id), request.transition_kind.value, request.coverage.obligation_id,
                        request.coverage.responsibility.value, request.coverage.contribution.value,
                        request.coverage.coverage_policy.policy_id, request.coverage.coverage_policy.policy_version,
                        canonical_json_bytes([value.canonical_value() for value in request.source_roles]), len(request.source_roles),
                        canonical_json_bytes([value.value for value in request.portfolio_functions]), len(request.portfolio_functions),
                        canonical_json_bytes([value.canonical_value() for value in request.source_dependencies]), len(request.source_dependencies),
                        canonical_json_bytes(list(request.incompleteness_warnings)), len(request.incompleteness_warnings),
                        canonical_json_bytes(request.urgency.canonical_value()), request.urgency.route.value,
                        None if request.urgency.hard_deadline is None else request.urgency.hard_deadline.to_text(),
                        request.urgency.planned_window, int(request.urgency.isolation_required), request.lead_policy.policy_id,
                        request.lead_policy.policy_version, request.reason_taxonomy_version, request.outcome_taxonomy_version,
                        request.created_at.to_text(), request.semantic_digest, committed.event_id, committed.aggregate_version,
                        request.canonical_bytes, request.digest, recorded_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DiscoveryStateError("News Lead persistence rejected inconsistent authority") from exc
            row = self._row(conn, "news_leads", "authority_event_id", committed.event_id)
            return self._lead_from_row(conn, row, replayed=False)

    def commit_watch_condition(self, grant: _AuthorizedCommandGrant, *, request: WatchConditionRequest) -> WatchCondition:
        if not isinstance(request, WatchConditionRequest):
            raise TypeError("Watch Condition commit requires a typed request")
        self._require_discovery_grant(
            grant,
            command_type=DISCOVERY_WATCH_CONDITION_RECORD_COMMAND,
            aggregate_id=str(request.watch_condition_id),
            canonical_bytes=request.canonical_bytes,
        )
        with self._lock, self._transaction() as conn:
            if grant.replay_of_command_id is not None:
                committed = self._commit_grant_in_transaction(conn, grant, recorded_at=self._clock().to_text())
                row = self._row(conn, "discovery_watch_conditions", "authority_event_id", committed.event_id)
                return self._watch_from_row(conn, row, replayed=True)
            lead = self._row(conn, "news_leads", "lead_id", str(request.lead_id))
            gate_head = conn.execute(
                "SELECT d.decision_id,d.outcome,d.evaluated_definition_version_id,d.decided_at "
                "FROM discovery_gate_decision_heads h "
                "JOIN discovery_gate_decisions d ON d.decision_id=h.current_decision_id "
                "WHERE h.signal_id=?",
                (str(lead["signal_id"]),),
            ).fetchone()
            if (
                gate_head is None
                or str(gate_head["decision_id"]) != str(request.gate_decision_id)
                or str(gate_head["outcome"])
                != GateOutcome.PROMOTED_TO_LEAD.value
            ):
                raise DiscoveryVersionConflict(
                    "Watch Condition requires the exact current promoting Gate"
                )
            self._require_current_version(
                conn,
                definition_id=SourceDefinitionId.parse(str(lead["definition_id"])),
                version_id=SourceDefinitionVersionId.parse(
                    str(gate_head["evaluated_definition_version_id"])
                ),
            )
            if request.recorded_at.value < UtcTimestamp.parse(
                str(gate_head["decided_at"])
            ).value:
                raise DiscoveryVersionConflict(
                    "Watch Condition predates its exact Gate Decision"
                )
            if request.corroborating_lead_id is not None:
                self._row(conn, "news_leads", "lead_id", str(request.corroborating_lead_id))
            self._discovery_identifier_absent(conn, table="discovery_watch_conditions", column="watch_condition_id", identifier=str(request.watch_condition_id), identity="Watch Condition identity")
            self._discovery_semantic_absent(conn, table="discovery_watch_conditions", semantic_digest=request.semantic_digest, identity="Watch Condition semantics")
            recorded_at = self._clock().to_text()
            committed = self._commit_grant_in_transaction(conn, grant, recorded_at=recorded_at)
            if committed.replayed:
                row = self._row(conn, "discovery_watch_conditions", "authority_event_id", committed.event_id)
                return self._watch_from_row(conn, row, replayed=True)
            try:
                conn.execute(
                    "INSERT INTO discovery_watch_conditions("
                    "watch_condition_id,lead_id,gate_decision_id,resume_transition_kinds_bytes,resume_transition_kind_count,expected_occurrence,"
                    "corroborating_lead_id,review_at,expires_at,operator_review_condition,closure_rule,watch_policy_id,"
                    "watch_policy_version,condition_recorded_at,semantic_digest,authority_event_id,authority_aggregate_version,"
                    "canonical_bytes,canonical_digest,recorded_at) VALUES(" + ",".join("?" for _ in range(20)) + ")",
                    (
                        str(request.watch_condition_id), str(request.lead_id),
                        str(request.gate_decision_id), canonical_json_bytes([value.value for value in request.resume_transition_kinds]), len(request.resume_transition_kinds),
                        request.expected_occurrence, None if request.corroborating_lead_id is None else str(request.corroborating_lead_id),
                        None if request.review_at is None else request.review_at.to_text(),
                        None if request.expires_at is None else request.expires_at.to_text(), request.operator_review_condition,
                        request.closure_rule, request.watch_policy.policy_id, request.watch_policy.policy_version,
                        request.recorded_at.to_text(), request.semantic_digest, committed.event_id, committed.aggregate_version,
                        request.canonical_bytes, request.digest, recorded_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DiscoveryStateError("Watch Condition persistence rejected inconsistent authority") from exc
            row = self._row(conn, "discovery_watch_conditions", "authority_event_id", committed.event_id)
            return self._watch_from_row(conn, row, replayed=False)

    def commit_lead_disposition(self, grant: _AuthorizedCommandGrant, *, request: LeadDispositionDecisionRequest) -> LeadDispositionDecision:
        if not isinstance(request, LeadDispositionDecisionRequest):
            raise TypeError("Lead Disposition commit requires a typed request")
        self._require_discovery_grant(
            grant,
            command_type=DISCOVERY_LEAD_DISPOSITION_RECORD_COMMAND,
            aggregate_id=str(request.decision_id),
            canonical_bytes=request.canonical_bytes,
        )
        with self._lock, self._transaction() as conn:
            if grant.replay_of_command_id is not None:
                committed = self._commit_grant_in_transaction(conn, grant, recorded_at=self._clock().to_text())
                row = self._row(conn, "lead_disposition_decisions", "authority_event_id", committed.event_id)
                return self._disposition_from_row(conn, row, replayed=True)
            lead = self._row(conn, "news_leads", "lead_id", str(request.lead_id))
            gate_head = conn.execute(
                "SELECT d.decision_id,d.outcome,d.evaluated_definition_version_id,d.decided_at "
                "FROM discovery_gate_decision_heads h "
                "JOIN discovery_gate_decisions d ON d.decision_id=h.current_decision_id "
                "WHERE h.signal_id=?",
                (str(lead["signal_id"]),),
            ).fetchone()
            if (
                gate_head is None
                or str(gate_head["decision_id"]) != str(request.gate_decision_id)
                or str(gate_head["outcome"])
                != GateOutcome.PROMOTED_TO_LEAD.value
            ):
                raise DiscoveryVersionConflict(
                    "Lead disposition requires the exact current promoting Gate"
                )
            self._require_current_version(
                conn,
                definition_id=SourceDefinitionId.parse(str(lead["definition_id"])),
                version_id=SourceDefinitionVersionId.parse(
                    str(gate_head["evaluated_definition_version_id"])
                ),
            )
            if request.decided_at.value < UtcTimestamp.parse(
                str(gate_head["decided_at"])
            ).value:
                raise DiscoveryVersionConflict(
                    "Lead disposition predates its exact Gate Decision"
                )
            if bytes(lead["urgency_bytes"]) != canonical_json_bytes(request.urgency_route.canonical_value()):
                raise DiscoveryVersionConflict("Lead disposition urgency differs from immutable Lead")
            if (
                request.decision_ordinal == 1
                and str(request.gate_decision_id)
                != str(lead["promoting_gate_decision_id"])
            ):
                raise DiscoveryVersionConflict(
                    "initial Lead disposition must consume the Lead's promoting Gate"
                )
            head = conn.execute(
                "SELECT current_decision_id,current_decision_ordinal FROM lead_disposition_heads WHERE lead_id=?",
                (str(request.lead_id),),
            ).fetchone()
            expected_ordinal = 1 if head is None else int(head["current_decision_ordinal"]) + 1
            expected_previous = None if head is None else str(head["current_decision_id"])
            actual_previous = None if request.previous_decision_id is None else str(request.previous_decision_id)
            if request.decision_ordinal != expected_ordinal or actual_previous != expected_previous:
                raise DiscoveryVersionConflict("Lead disposition does not extend exact current head")
            if request.watch_condition_id is not None:
                watch = self._row(conn, "discovery_watch_conditions", "watch_condition_id", str(request.watch_condition_id))
                if str(watch["lead_id"]) != str(request.lead_id):
                    raise DiscoveryVersionConflict("Lead disposition Watch Condition belongs to another Lead")
                if str(watch["gate_decision_id"]) != str(request.gate_decision_id):
                    raise DiscoveryVersionConflict(
                        "Lead disposition Watch Condition belongs to another Gate"
                    )
            self._discovery_identifier_absent(conn, table="lead_disposition_decisions", column="decision_id", identifier=str(request.decision_id), identity="Lead disposition identity")
            self._discovery_semantic_absent(conn, table="lead_disposition_decisions", semantic_digest=request.semantic_digest, identity="Lead disposition semantics")
            recorded_at = self._clock().to_text()
            committed = self._commit_grant_in_transaction(conn, grant, recorded_at=recorded_at)
            if committed.replayed:
                row = self._row(conn, "lead_disposition_decisions", "authority_event_id", committed.event_id)
                return self._disposition_from_row(conn, row, replayed=True)
            try:
                conn.execute(
                    "INSERT INTO lead_disposition_decisions("
                    "decision_id,lead_id,gate_decision_id,decision_ordinal,previous_decision_id,outcome,terminality,primary_reason_bytes,"
                    "supporting_reasons_bytes,supporting_reason_count,watch_condition_id,next_action_kind,next_action_code,"
                    "next_action_bytes,urgency_bytes,urgency_route,disposition_policy_id,disposition_policy_version,"
                    "reason_taxonomy_version,outcome_taxonomy_version,decided_at,semantic_digest,authority_event_id,"
                    "authority_aggregate_version,canonical_bytes,canonical_digest,recorded_at) VALUES(" + ",".join("?" for _ in range(27)) + ")",
                    (
                        str(request.decision_id), str(request.lead_id), str(request.gate_decision_id),
                        request.decision_ordinal, actual_previous,
                        request.outcome.value, request.terminality.value,
                        canonical_json_bytes(request.primary_reason.canonical_value()),
                        canonical_json_bytes([value.canonical_value() for value in request.supporting_reasons]),
                        len(request.supporting_reasons), None if request.watch_condition_id is None else str(request.watch_condition_id),
                        request.next_action.kind.value, request.next_action.action_code,
                        canonical_json_bytes(request.next_action.canonical_value()),
                        canonical_json_bytes(request.urgency_route.canonical_value()), request.urgency_route.route.value,
                        request.disposition_policy.policy_id, request.disposition_policy.policy_version,
                        request.reason_taxonomy_version, request.outcome_taxonomy_version, request.decided_at.to_text(),
                        request.semantic_digest, committed.event_id, committed.aggregate_version, request.canonical_bytes,
                        request.digest, recorded_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DiscoveryStateError("Lead disposition persistence rejected inconsistent authority") from exc
            row = self._row(conn, "lead_disposition_decisions", "authority_event_id", committed.event_id)
            return self._disposition_from_row(conn, row, replayed=False)

    # ------------------------------------------------------------------
    # Startup integrity: canonical rehydration, heads and event coverage
    # ------------------------------------------------------------------
    def _validate_schema_and_integrity(self) -> None:
        super()._validate_schema_and_integrity()
        if not self._should_validate_row_integrity():
            return
        with _validation_stage("discovery_integrity"):
            conn = self._connection
            for row in conn.execute("SELECT * FROM discovery_signals ORDER BY recorded_at,signal_id"):
                self._signal_from_row(conn, row, replayed=False)
                request = signal_request_from_bytes(bytes(row["canonical_bytes"]))
                self._require_exact_signal_lineage(conn, request)
            for row in conn.execute("SELECT * FROM discovery_gate_decisions ORDER BY signal_id,decision_ordinal"):
                self._gate_from_row(conn, row, replayed=False)
            for row in conn.execute("SELECT * FROM news_leads ORDER BY recorded_at,lead_id"):
                self._lead_from_row(conn, row, replayed=False)
                self._require_source_contract_matches_lead(conn, lead_request_from_bytes(bytes(row["canonical_bytes"])))
            for row in conn.execute("SELECT * FROM discovery_watch_conditions ORDER BY recorded_at,watch_condition_id"):
                self._watch_from_row(conn, row, replayed=False)
            for row in conn.execute("SELECT * FROM lead_disposition_decisions ORDER BY lead_id,decision_ordinal"):
                self._disposition_from_row(conn, row, replayed=False)
            self._validate_discovery_heads(conn)
            self._validate_discovery_event_coverage(conn)

    @staticmethod
    def _validate_discovery_heads(conn: sqlite3.Connection) -> None:
        gate_bad = conn.execute(
            "SELECT h.signal_id FROM discovery_gate_decision_heads h "
            "LEFT JOIN discovery_gate_decisions d ON d.decision_id=h.current_decision_id "
            "WHERE d.decision_id IS NULL OR d.signal_id!=h.signal_id "
            "OR d.decision_ordinal!=h.current_decision_ordinal OR d.decided_at!=h.updated_at "
            "OR EXISTS(SELECT 1 FROM discovery_gate_decisions n WHERE n.signal_id=h.signal_id "
            "AND n.decision_ordinal>h.current_decision_ordinal) LIMIT 1"
        ).fetchone()
        if gate_bad is not None:
            raise AuthoritySchemaError("Gate Decision head differs from immutable chain")
        disposition_bad = conn.execute(
            "SELECT h.lead_id FROM lead_disposition_heads h "
            "LEFT JOIN lead_disposition_decisions d ON d.decision_id=h.current_decision_id "
            "WHERE d.decision_id IS NULL OR d.lead_id!=h.lead_id "
            "OR d.decision_ordinal!=h.current_decision_ordinal OR d.decided_at!=h.updated_at "
            "OR EXISTS(SELECT 1 FROM lead_disposition_decisions n WHERE n.lead_id=h.lead_id "
            "AND n.decision_ordinal>h.current_decision_ordinal) LIMIT 1"
        ).fetchone()
        if disposition_bad is not None:
            raise AuthoritySchemaError("Lead disposition head differs from immutable chain")
        missing_gate = conn.execute(
            "SELECT s.signal_id FROM discovery_signals s LEFT JOIN discovery_gate_decision_heads h "
            "ON h.signal_id=s.signal_id WHERE h.signal_id IS NULL LIMIT 1"
        ).fetchone()
        # Signal-only and Lead-without-disposition prefixes are recoverable
        # boundaries between independently authorised commands. A retained
        # Lead must still point to its exact promoting Gate; admission replay
        # completes a missing first disposition idempotently.
        orphan_lead = conn.execute(
            "SELECT l.lead_id FROM news_leads l "
            "LEFT JOIN discovery_gate_decisions g ON g.decision_id=l.promoting_gate_decision_id "
            "WHERE g.decision_id IS NULL OR g.signal_id!=l.signal_id "
            "OR g.outcome!='SIGNAL_PROMOTED_TO_LEAD' LIMIT 1"
        ).fetchone()
        if orphan_lead is not None:
            raise AuthoritySchemaError("News Lead lacks exact promotion authority")
        del missing_gate

    @staticmethod
    def _validate_discovery_event_coverage(
        conn: sqlite3.Connection,
        *,
        aggregate_types: tuple[str, ...] | None = None,
    ) -> None:
        specs = (
            ("discovery.signal.admitted", "discovery_signals", "discovery_signal"),
            ("discovery.gate.decided", "discovery_gate_decisions", "gate_decision"),
            ("discovery.lead.opened", "news_leads", "news_lead"),
            ("discovery.watch_condition.recorded", "discovery_watch_conditions", "watch_condition"),
            ("discovery.lead.disposition.recorded", "lead_disposition_decisions", "lead_disposition_decision"),
        )
        if aggregate_types is not None and aggregate_types != tuple(
            spec[2] for spec in specs
        ):
            raise AuthoritySchemaError("Discovery aggregate coverage differs")
        for event_type, table, aggregate_type in specs:
            if aggregate_types is None:
                missing = conn.execute(
                    f"SELECT e.event_id FROM ledger_events e LEFT JOIN {table} r "
                    "ON r.authority_event_id=e.event_id WHERE e.event_type=? "
                    "AND r.authority_event_id IS NULL LIMIT 1",
                    (event_type,),
                ).fetchone()
            else:
                domain_mismatch = conn.execute(
                    f"SELECT r.authority_event_id FROM {table} r LEFT JOIN "
                    "ledger_events e ON e.event_id=r.authority_event_id "
                    "WHERE e.event_id IS NULL OR e.aggregate_type!=? "
                    "OR e.event_type!=? LIMIT 1",
                    (aggregate_type, event_type),
                ).fetchone()
                missing = conn.execute(
                    f"SELECT e.event_id FROM ledger_events e LEFT JOIN {table} r "
                    "ON r.authority_event_id=e.event_id WHERE e.aggregate_type=? "
                    "AND (e.event_type!=? OR r.authority_event_id IS NULL) LIMIT 1",
                    (aggregate_type, event_type),
                ).fetchone()
                if domain_mismatch is not None:
                    missing = domain_mismatch
            if missing is not None:
                raise AuthoritySchemaError(f"{event_type} has no exact domain record")


__all__ = ["_DiscoveryAuthorityStore"]
