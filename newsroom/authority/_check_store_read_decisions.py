from __future__ import annotations

import sqlite3

from newsroom.authority._check_store_decoding import (
    decode_baseline_decision,
    decode_observable_transition,
    decode_operational_finding,
    decode_operational_finding_occurrence,
)
from newsroom.authority._source_registry_decoding import canonical_row_value
from newsroom.authority.canonical import canonical_json_bytes, digest_canonical
from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.authority.types import EventId, UtcTimestamp
from newsroom.checks.policy import (
    CHECK_BASELINE_DECIDE_COMMAND,
    OBSERVABLE_TRANSITION_RECORD_COMMAND,
    OPERATIONAL_FINDING_OCCURRENCE_RECORD_COMMAND,
    OPERATIONAL_FINDING_OPEN_COMMAND,
)
from newsroom.checks.record_models import (
    BaselineDecision,
    ObservableTransition,
    OperationalFinding,
    OperationalFindingOccurrence,
)
from newsroom.checks.types import (
    BaselineDecisionId,
    ObservableTransitionId,
    OperationalFindingId,
    OperationalFindingOccurrenceId,
)
from newsroom.sources import SourceDefinitionId


class _CheckStoreReadDecisionMixin:
    def _baseline_decision_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> BaselineDecision:
        self._prove_current_record(row)
        value = canonical_row_value(row, identity="Baseline Decision")
        event = self._validate_record_envelope(
            conn,
            row,
            command_type=CHECK_BASELINE_DECIDE_COMMAND,
            aggregate_id=str(row["decision_id"]),
            canonical_bytes=bytes(row["canonical_bytes"]),
            canonical_digest=str(row["canonical_digest"]),
        )
        request = decode_baseline_decision(
            value,
            idempotency_key=str(event["idempotency_key"]),
        )
        self._require_normalized_columns(
            row,
            {
                "definition_id": str(request.definition_id),
                "definition_version_id": str(request.definition_version_id),
                "check_request_id": str(request.check_request_id),
                "check_outcome_id": str(request.check_outcome_id),
                "kind": request.kind.value,
                "disposition": request.disposition.value,
                "observation_model": request.observation_model.value,
                "baseline_policy_id": request.baseline_policy.policy_id,
                "baseline_policy_version": (
                    request.baseline_policy.policy_version
                ),
                "previous_decision_id": (
                    None
                    if request.previous_decision_id is None
                    else str(request.previous_decision_id)
                ),
                "source_body_digest": request.source_body_digest,
                "producer_slot_digest": request.producer_slot_digest,
                "representation_digest": request.representation_digest,
                "validator_digest": request.validator_digest,
                "item_keys_digest": request.item_keys_digest,
                "decided_at": request.decided_at.to_text(),
                "semantic_digest": request.semantic_digest,
            },
            identity="Baseline Decision",
        )
        self._require_canonical_blob(
            row,
            "reason_codes_bytes",
            list(request.reason_codes),
            identity="Baseline Decision",
        )
        children = conn.execute(
            "SELECT * FROM baseline_manifest_entries WHERE decision_id=? "
            "ORDER BY item_key",
            (str(request.decision_id),),
        ).fetchall()
        if len(children) != len(request.entries):
            raise AuthorityPersistenceError(
                "Baseline Decision manifest count differs from canonical bytes"
            )
        for child, entry in zip(children, request.entries, strict=True):
            expected = entry.canonical_value()
            expected_bytes = canonical_json_bytes(expected)
            if (
                str(child["item_key"]) != entry.item_key
                or str(child["disposition"]) != entry.disposition.value
                or str(child["reason_code"]) != entry.reason_code
                or child["item_id"]
                != (None if entry.item_id is None else str(entry.item_id))
                or child["revision_id"]
                != (
                    None
                    if entry.revision_id is None
                    else str(entry.revision_id)
                )
                or bytes(child["canonical_bytes"]) != expected_bytes
                or str(child["canonical_digest"])
                != digest_canonical(expected)
            ):
                raise AuthorityPersistenceError(
                    "Baseline Decision manifest differs from canonical bytes"
                )
        evidence_error = self._baseline_evidence_error(conn, request)
        if evidence_error is not None:
            raise AuthorityPersistenceError(evidence_error)
        if str(request.decision_id) != str(row["decision_id"]):
            raise AuthorityPersistenceError(
                "Baseline Decision identity differs from canonical bytes"
            )
        return BaselineDecision(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    def _observable_transition_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> ObservableTransition:
        self._prove_current_record(row)
        value = canonical_row_value(row, identity="Observable Transition")
        event = self._validate_record_envelope(
            conn,
            row,
            command_type=OBSERVABLE_TRANSITION_RECORD_COMMAND,
            aggregate_id=str(row["transition_id"]),
            canonical_bytes=bytes(row["canonical_bytes"]),
            canonical_digest=str(row["canonical_digest"]),
        )
        request = decode_observable_transition(
            value,
            idempotency_key=str(event["idempotency_key"]),
        )
        self._require_normalized_columns(
            row,
            {
                "definition_id": str(request.definition_id),
                "definition_version_id": str(request.definition_version_id),
                "check_outcome_id": str(request.check_outcome_id),
                "item_id": str(request.item_id),
                "kind": request.kind.value,
                "basis": request.basis.value,
                "observation_model": request.observation_model.value,
                "prior_revision_id": (
                    None
                    if request.prior_revision_id is None
                    else str(request.prior_revision_id)
                ),
                "current_revision_id": (
                    None
                    if request.current_revision_id is None
                    else str(request.current_revision_id)
                ),
                "representation_id": (
                    None
                    if request.representation_id is None
                    else str(request.representation_id)
                ),
                "related_item_id": (
                    None
                    if request.related_item_id is None
                    else str(request.related_item_id)
                ),
                "transition_policy_id": request.transition_policy.policy_id,
                "transition_policy_version": (
                    request.transition_policy.policy_version
                ),
                "observed_at": request.observed_at.to_text(),
                "transition_discriminator": (
                    request.transition_discriminator
                ),
                "semantic_digest": request.semantic_digest,
            },
            identity="Observable Transition",
        )
        self._require_canonical_blob(
            row,
            "change_facets_bytes",
            list(request.change_facets),
            identity="Observable Transition",
        )
        for column, guard in (
            ("absence_guard_bytes", request.absence_guard),
            ("agenda_guard_bytes", request.agenda_guard),
        ):
            expected = None if guard is None else canonical_json_bytes(
                guard.canonical_value()
            )
            actual = None if row[column] is None else bytes(row[column])
            if actual != expected:
                raise AuthorityPersistenceError(
                    f"Observable Transition column {column} differs from canonical bytes"
                )
        self._require_canonical_blob(
            row,
            "source_asserted_time_bytes",
            request.source_asserted_time.canonical_value(),
            identity="Observable Transition",
        )
        evidence_error = self._transition_evidence_error(conn, request)
        if evidence_error is not None:
            raise AuthorityPersistenceError(evidence_error)
        if str(request.transition_id) != str(row["transition_id"]):
            raise AuthorityPersistenceError(
                "Observable Transition identity differs from canonical bytes"
            )
        return ObservableTransition(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    def _operational_finding_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> OperationalFinding:
        self._prove_current_record(row)
        value = canonical_row_value(row, identity="Operational Finding")
        event = self._validate_record_envelope(
            conn,
            row,
            command_type=OPERATIONAL_FINDING_OPEN_COMMAND,
            aggregate_id=str(row["finding_id"]),
            canonical_bytes=bytes(row["canonical_bytes"]),
            canonical_digest=str(row["canonical_digest"]),
        )
        request = decode_operational_finding(
            value,
            idempotency_key=str(event["idempotency_key"]),
        )
        self._require_normalized_columns(
            row,
            {
                "scope_kind": request.scope_kind.value,
                "scope_id": request.scope_id,
                "category": request.category.value,
                "severity": request.severity.value,
                "finding_policy_id": request.finding_policy.policy_id,
                "finding_policy_version": (
                    request.finding_policy.policy_version
                ),
                "summary": request.summary,
                "opened_by_request_id": (
                    None
                    if request.opened_by_request_id is None
                    else str(request.opened_by_request_id)
                ),
                "opened_by_attempt_id": (
                    None
                    if request.opened_by_attempt_id is None
                    else str(request.opened_by_attempt_id)
                ),
                "opened_by_outcome_id": (
                    None
                    if request.opened_by_outcome_id is None
                    else str(request.opened_by_outcome_id)
                ),
                "opened_at": request.opened_at.to_text(),
                "semantic_digest": request.semantic_digest,
            },
            identity="Operational Finding",
        )
        if str(request.finding_id) != str(row["finding_id"]):
            raise AuthorityPersistenceError(
                "Operational Finding identity differs from canonical bytes"
            )
        return OperationalFinding(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    def _finding_occurrence_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> OperationalFindingOccurrence:
        self._prove_current_record(row)
        value = canonical_row_value(
            row, identity="Operational Finding occurrence"
        )
        event = self._validate_record_envelope(
            conn,
            row,
            command_type=OPERATIONAL_FINDING_OCCURRENCE_RECORD_COMMAND,
            aggregate_id=str(row["occurrence_id"]),
            canonical_bytes=bytes(row["canonical_bytes"]),
            canonical_digest=str(row["canonical_digest"]),
        )
        request = decode_operational_finding_occurrence(
            value,
            idempotency_key=str(event["idempotency_key"]),
        )
        self._require_normalized_columns(
            row,
            {
                "finding_id": str(request.finding_id),
                "request_id": (
                    None if request.request_id is None else str(request.request_id)
                ),
                "attempt_id": (
                    None if request.attempt_id is None else str(request.attempt_id)
                ),
                "outcome_id": (
                    None if request.outcome_id is None else str(request.outcome_id)
                ),
                "code": request.code,
                "detail_digest": request.detail_digest,
                "observed_at": request.observed_at.to_text(),
                "semantic_digest": request.semantic_digest,
            },
            identity="Operational Finding occurrence",
        )
        if str(request.occurrence_id) != str(row["occurrence_id"]):
            raise AuthorityPersistenceError(
                "Finding occurrence identity differs from canonical bytes"
            )
        return OperationalFindingOccurrence(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    def _baseline_decision_for_event(
        self, conn: sqlite3.Connection, event_id: str, *, replayed: bool
    ) -> BaselineDecision:
        return self._for_event(
            conn,
            event_id,
            table="baseline_decisions",
            identity="Baseline Decision",
            loader=self._baseline_decision_from_row,
            replayed=replayed,
        )

    def _observable_transition_for_event(
        self, conn: sqlite3.Connection, event_id: str, *, replayed: bool
    ) -> ObservableTransition:
        return self._for_event(
            conn,
            event_id,
            table="observable_transitions",
            identity="Observable Transition",
            loader=self._observable_transition_from_row,
            replayed=replayed,
        )

    def _operational_finding_for_event(
        self, conn: sqlite3.Connection, event_id: str, *, replayed: bool
    ) -> OperationalFinding:
        return self._for_event(
            conn,
            event_id,
            table="operational_findings",
            identity="Operational Finding",
            loader=self._operational_finding_from_row,
            replayed=replayed,
        )

    def _finding_occurrence_for_event(
        self, conn: sqlite3.Connection, event_id: str, *, replayed: bool
    ) -> OperationalFindingOccurrence:
        return self._for_event(
            conn,
            event_id,
            table="operational_finding_occurrences",
            identity="Operational Finding occurrence",
            loader=self._finding_occurrence_from_row,
            replayed=replayed,
        )

    def baseline_decision(
        self, decision_id: BaselineDecisionId
    ) -> BaselineDecision | None:
        with self._lock:
            row = self._row_by_id(
                self._connection,
                table="baseline_decisions",
                column="decision_id",
                identifier=str(decision_id),
            )
            return None if row is None else self._baseline_decision_from_row(
                self._connection, row, replayed=False
            )

    def current_baseline_decision(
        self, definition_id: SourceDefinitionId
    ) -> BaselineDecision | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT d.* FROM baseline_decision_heads h "
                "JOIN baseline_decisions d "
                "ON d.decision_id=h.current_decision_id "
                "WHERE h.definition_id=?",
                (str(definition_id),),
            ).fetchone()
            return None if row is None else self._baseline_decision_from_row(
                self._connection, row, replayed=False
            )

    def observable_transition(
        self, transition_id: ObservableTransitionId
    ) -> ObservableTransition | None:
        with self._lock:
            row = self._row_by_id(
                self._connection,
                table="observable_transitions",
                column="transition_id",
                identifier=str(transition_id),
            )
            return None if row is None else self._observable_transition_from_row(
                self._connection, row, replayed=False
            )

    def operational_finding(
        self, finding_id: OperationalFindingId
    ) -> OperationalFinding | None:
        with self._lock:
            row = self._row_by_id(
                self._connection,
                table="operational_findings",
                column="finding_id",
                identifier=str(finding_id),
            )
            return None if row is None else self._operational_finding_from_row(
                self._connection, row, replayed=False
            )

    def finding_occurrences(
        self,
        finding_id: OperationalFindingId,
        *,
        limit: int,
    ) -> tuple[OperationalFindingOccurrence, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM operational_finding_occurrences "
                "WHERE finding_id=? ORDER BY observed_at,recorded_at LIMIT ?",
                (str(finding_id), limit),
            ).fetchall()
            return tuple(
                self._finding_occurrence_from_row(
                    self._connection,
                    row,
                    replayed=False,
                )
                for row in rows
            )


__all__ = ["_CheckStoreReadDecisionMixin"]
