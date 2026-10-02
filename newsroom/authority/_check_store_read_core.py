from __future__ import annotations

import sqlite3

from newsroom.authority._check_store_decoding import (
    decode_check_attempt,
    decode_check_outcome,
    decode_check_request,
)
from newsroom.authority._check_store_support import _observed_item_id
from newsroom.authority._source_registry_decoding import canonical_row_value
from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.authority.types import EventId, UtcTimestamp
from newsroom.checks.policy import (
    CHECK_ATTEMPT_START_COMMAND,
    CHECK_OUTCOME_RECORD_COMMAND,
    CHECK_REQUEST_REGISTER_COMMAND,
)
from newsroom.checks.record_models import (
    CheckAttempt,
    CheckOutcome,
    CheckRequest,
)
from newsroom.checks.types import (
    CheckAttemptId,
    CheckRequestId,
)
from newsroom.sources import CheckOutcomeId


class _CheckStoreReadCoreMixin:
    def _check_request_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> CheckRequest:
        self._prove_current_record(row)
        value = canonical_row_value(row, identity="Check Request")
        event = self._validate_record_envelope(
            conn,
            row,
            command_type=CHECK_REQUEST_REGISTER_COMMAND,
            aggregate_id=str(row["request_id"]),
            canonical_bytes=bytes(row["canonical_bytes"]),
            canonical_digest=str(row["canonical_digest"]),
        )
        request = decode_check_request(
            value,
            idempotency_key=str(event["idempotency_key"]),
        )
        self._require_normalized_columns(
            row,
            {
                "definition_id": str(request.definition_id),
                "definition_version_id": str(request.definition_version_id),
                "trigger_kind": request.trigger.kind.value,
                "trigger_id": request.trigger.trigger_id,
                "trigger_version": request.trigger.trigger_version,
                "expected_window_digest": (
                    request.trigger.expected_window_digest
                ),
                "coverage_obligation_id": request.coverage.obligation_id,
                "coverage_responsibility": (
                    request.coverage.responsibility.value
                ),
                "coverage_contribution": (
                    request.coverage.contribution.value
                ),
                "coverage_policy_id": (
                    request.coverage.coverage_policy.policy_id
                ),
                "coverage_policy_version": (
                    request.coverage.coverage_policy.policy_version
                ),
                "rights_decision_id": request.rights_decision_id,
                "rights_policy_version": request.rights_policy_version,
                "adapter_request_digest": request.adapter_request_digest,
                "producer_slot_digest": request.producer_slot_digest,
                "baseline_policy_id": request.baseline_policy.policy_id,
                "baseline_policy_version": (
                    request.baseline_policy.policy_version
                ),
                "revision_policy_id": request.revision_policy.policy_id,
                "revision_policy_version": (
                    request.revision_policy.policy_version
                ),
                "transition_policy_id": request.transition_policy.policy_id,
                "transition_policy_version": (
                    request.transition_policy.policy_version
                ),
                "validator_policy_id": request.validator_policy.policy_id,
                "validator_policy_version": (
                    request.validator_policy.policy_version
                ),
                "purpose": request.purpose,
                "requested_at": request.requested_at.to_text(),
                "semantic_digest": request.semantic_digest,
            },
            identity="Check Request",
        )
        if str(request.request_id) != str(row["request_id"]):
            raise AuthorityPersistenceError(
                "Check Request identity differs from canonical bytes"
            )
        return CheckRequest(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    def _check_attempt_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> CheckAttempt:
        self._prove_current_record(row)
        value = canonical_row_value(row, identity="Check Attempt")
        event = self._validate_record_envelope(
            conn,
            row,
            command_type=CHECK_ATTEMPT_START_COMMAND,
            aggregate_id=str(row["attempt_id"]),
            canonical_bytes=bytes(row["canonical_bytes"]),
            canonical_digest=str(row["canonical_digest"]),
        )
        request = decode_check_attempt(
            value,
            idempotency_key=str(event["idempotency_key"]),
        )
        self._require_normalized_columns(
            row,
            {
                "request_id": str(request.request_id),
                "attempt_number": request.attempt_number,
                "kind": request.kind.value,
                "prior_attempt_id": (
                    None
                    if request.prior_attempt_id is None
                    else str(request.prior_attempt_id)
                ),
                "adapter_request_id": str(request.adapter_request_id),
                "adapter_request_digest": request.adapter_request_digest,
                "started_at": request.started_at.to_text(),
                "semantic_digest": request.semantic_digest,
            },
            identity="Check Attempt",
        )
        if str(request.attempt_id) != str(row["attempt_id"]):
            raise AuthorityPersistenceError(
                "Check Attempt identity differs from canonical bytes"
            )
        return CheckAttempt(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    def _check_outcome_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> CheckOutcome:
        self._prove_current_record(row)
        value = canonical_row_value(row, identity="Check Outcome")
        event = self._validate_record_envelope(
            conn,
            row,
            command_type=CHECK_OUTCOME_RECORD_COMMAND,
            aggregate_id=str(row["outcome_id"]),
            canonical_bytes=bytes(row["canonical_bytes"]),
            canonical_digest=str(row["canonical_digest"]),
        )
        request = decode_check_outcome(
            value,
            idempotency_key=str(event["idempotency_key"]),
        )
        self._require_normalized_columns(
            row,
            {
                "request_id": str(request.request_id),
                "attempt_id": str(request.attempt_id),
                "proposal_id": str(request.proposal_id),
                "definition_id": str(request.definition_id),
                "definition_version_id": str(request.definition_version_id),
                "kind": request.kind.value,
                "quarantine": request.quarantine.value,
                "incomplete": int(request.incomplete),
                "receipt_digest": request.receipt_digest,
                "capture_digest": request.capture_digest,
                "parser_result_digest": request.parser_result_digest,
                "source_body_digest": request.source_body_digest,
                "producer_slot_digest": request.producer_slot_digest,
                "representation_digest": request.representation_digest,
                "validator_digest": request.validator_digest,
                "candidate_count": len(request.candidate_observations),
                "observed_item_count": len(request.observed_items),
                "completed_at": request.completed_at.to_text(),
                "admission_semantic_digest": (
                    request.admission_semantic_digest
                ),
                "semantic_digest": request.semantic_digest,
            },
            identity="Check Outcome",
        )
        self._require_canonical_blob(
            row,
            "reason_codes_bytes",
            list(request.reason_codes),
            identity="Check Outcome",
        )
        self._require_canonical_blob(
            row,
            "candidate_observations_bytes",
            [item.canonical_value() for item in request.candidate_observations],
            identity="Check Outcome",
        )
        self._require_canonical_blob(
            row,
            "observed_items_bytes",
            [item.canonical_value() for item in request.observed_items],
            identity="Check Outcome",
        )
        expected_observed_items = tuple(
            (
                item.item_key,
                item.item_digest,
                str(
                    _observed_item_id(
                        self,
                        conn,
                        request_id=str(request.request_id),
                        definition_id=str(request.definition_id),
                        item_key=item.item_key,
                    )
                ),
            )
            for item in request.observed_items
        )
        actual_observed_items = tuple(
            (
                str(item["item_key"]),
                str(item["item_digest"]),
                str(item["item_id"]),
            )
            for item in conn.execute(
                "SELECT item_key,item_digest,item_id "
                "FROM check_outcome_observed_items WHERE outcome_id=? "
                "ORDER BY item_key",
                (str(request.outcome_id),),
            ).fetchall()
        )
        if actual_observed_items != expected_observed_items:
            raise AuthorityPersistenceError(
                "Check Outcome observed-item index differs from canonical bytes"
            )
        if str(request.outcome_id) != str(row["outcome_id"]):
            raise AuthorityPersistenceError(
                "Check Outcome identity differs from canonical bytes"
            )
        return CheckOutcome(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    def _check_request_for_event(
        self,
        conn: sqlite3.Connection,
        event_id: str,
        *,
        replayed: bool,
    ) -> CheckRequest:
        return self._for_event(
            conn,
            event_id,
            table="check_requests",
            identity="Check Request",
            loader=self._check_request_from_row,
            replayed=replayed,
        )

    def _check_attempt_for_event(
        self,
        conn: sqlite3.Connection,
        event_id: str,
        *,
        replayed: bool,
    ) -> CheckAttempt:
        return self._for_event(
            conn,
            event_id,
            table="check_attempts",
            identity="Check Attempt",
            loader=self._check_attempt_from_row,
            replayed=replayed,
        )

    def _check_outcome_for_event(
        self,
        conn: sqlite3.Connection,
        event_id: str,
        *,
        replayed: bool,
    ) -> CheckOutcome:
        return self._for_event(
            conn,
            event_id,
            table="check_outcomes",
            identity="Check Outcome",
            loader=self._check_outcome_from_row,
            replayed=replayed,
        )

    def check_request(self, request_id: CheckRequestId) -> CheckRequest | None:
        with self._lock:
            row = self._row_by_id(
                self._connection,
                table="check_requests",
                column="request_id",
                identifier=str(request_id),
            )
            return (
                None
                if row is None
                else self._check_request_from_row(
                    self._connection,
                    row,
                    replayed=False,
                )
            )

    def check_attempt(self, attempt_id: CheckAttemptId) -> CheckAttempt | None:
        with self._lock:
            row = self._row_by_id(
                self._connection,
                table="check_attempts",
                column="attempt_id",
                identifier=str(attempt_id),
            )
            return (
                None
                if row is None
                else self._check_attempt_from_row(
                    self._connection,
                    row,
                    replayed=False,
                )
            )

    def check_outcome(self, outcome_id: CheckOutcomeId) -> CheckOutcome | None:
        with self._lock:
            row = self._row_by_id(
                self._connection,
                table="check_outcomes",
                column="outcome_id",
                identifier=str(outcome_id),
            )
            return (
                None
                if row is None
                else self._check_outcome_from_row(
                    self._connection,
                    row,
                    replayed=False,
                )
            )

    def attempts_for_request(
        self,
        request_id: CheckRequestId,
        *,
        limit: int,
    ) -> tuple[CheckAttempt, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM check_attempts WHERE request_id=? "
                "ORDER BY attempt_number LIMIT ?",
                (str(request_id), limit),
            ).fetchall()
            return tuple(
                self._check_attempt_from_row(
                    self._connection,
                    row,
                    replayed=False,
                )
                for row in rows
            )

    def outcomes_for_request(
        self,
        request_id: CheckRequestId,
        *,
        limit: int,
    ) -> tuple[CheckOutcome, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM check_outcomes WHERE request_id=? "
                "ORDER BY completed_at,recorded_at LIMIT ?",
                (str(request_id), limit),
            ).fetchall()
            return tuple(
                self._check_outcome_from_row(
                    self._connection,
                    row,
                    replayed=False,
                )
                for row in rows
            )


__all__ = ["_CheckStoreReadCoreMixin"]
