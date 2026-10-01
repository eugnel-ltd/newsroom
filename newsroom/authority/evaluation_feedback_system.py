"""Server-owned v25 Evaluation Feedback and reconciliation composition root."""

# fmt: off
# ruff: noqa: I001
from __future__ import annotations
import json
import threading
from collections.abc import Callable
from pathlib import Path
from newsroom.authority._capability import _CapabilityIssuer
from newsroom.authority._event_store import _EventAuthorityStore
from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.authority.increment8_evaluation_migrations import INCREMENT8_EVALUATION_TABLES
from newsroom.authority.models import InlinePayload, SemanticCommand
from newsroom.authority.policy import CommandRegistry, PayloadSchemaRegistry
from newsroom.authority.service import CommandService
from newsroom.authority.story_candidate_system import _create_story_candidate_read_port
from newsroom.authority.types import AggregateId, UtcTimestamp
from newsroom.increment6.candidates import StoryCandidateReadPort, merge_candidate_authority_registries
from newsroom.increment6.feedback import EvaluationFeedback, EvaluationFeedbackAcceptance, EvaluationFeedbackAuthority, FeedbackContractError, FeedbackCorrelationOutcome, HandoffAcceptanceSnapshot, ReconciliationDisposition, ReconciliationObligation, _compose_evaluation_feedback_authority, append_reconciliation_disposition, correlate_evaluation_feedback, create_reconciliation_obligation, evaluation_feedback_command_definition, merge_evaluation_feedback_authority_registries, validate_reconciliation_history
from newsroom.increment6.handoffs import EvaluationHandoffReadPort, _create_evaluation_handoff_read_port
from newsroom.increment6.lineage import merge_lineage_authority_registries
from newsroom.increment6.relationships import merge_relationship_authority_registries
_V25_TABLES = {
    "evaluation_feedback",
    "evaluation_reconciliation_obligations",
    "evaluation_reconciliation_dispositions",
}
class _EvaluationFeedbackAuthorityRoot:
    """Own the only v25 writer and its single checked SQLite connection."""
    def __init__(
        self,
        event_store: _EventAuthorityStore,
        service: CommandService,
        candidate_port: StoryCandidateReadPort,
        handoff_port: EvaluationHandoffReadPort,
        clock: Callable[[], UtcTimestamp],
    ) -> None:
        self._event_store = event_store
        self._connection = event_store._connection
        self._service = service
        self._candidate = candidate_port
        self._handoff = handoff_port
        self._clock = clock
        self._lock = threading.RLock()
        self._closed = False
        self._run(self._verify_local)
    def _require_open(self) -> None:
        if self._closed:
            raise FeedbackContractError("Feedback authority is closed")
    def _begin(self) -> None:
        self._require_open()
        if self._connection.in_transaction:
            raise FeedbackContractError("Feedback transaction ownership differs")
        self._connection.execute("BEGIN IMMEDIATE")
    def _run(self, operation: Callable[[], object], *, commit: bool = True) -> object:
        with self._lock:
            self._begin()
            try:
                result = operation()
                self._candidate.verify_retained_integrity_in_transaction()
                self._handoff.verify_retained_integrity_in_transaction()
                if commit:
                    self._connection.execute("COMMIT")
                return result
            finally:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
    def _recorded_at(self) -> str:
        value = self._clock()
        if type(value) is not UtcTimestamp:
            raise FeedbackContractError("Feedback clock returned a forged timestamp")
        return value.to_text()
    def _verify_schema(self) -> None:
        rows = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'evaluation_%'"
        ).fetchall()
        names = {str(row[0]) for row in rows}
        # v17 Handoff tables are deliberately outside this atom.
        if names - {
            "evaluation_handoffs",
            "evaluation_handoff_attempts",
            "evaluation_handoff_acknowledgements",
        } - set(INCREMENT8_EVALUATION_TABLES) != _V25_TABLES:
            raise FeedbackContractError("Feedback retained table allocation differs")
        required_triggers = {
            "immutable_evaluation_feedback",
            "retained_evaluation_feedback",
            "immutable_evaluation_obligation",
            "retained_evaluation_obligation",
            "immutable_evaluation_disposition",
            "retained_evaluation_disposition",
            "evaluation_disposition_predecessor_guard",
        }
        actual = {
            str(row[0])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND "
                "(tbl_name='evaluation_feedback' OR tbl_name LIKE 'evaluation_reconciliation_%')"
            )
        }
        if actual != required_triggers:
            raise FeedbackContractError("Feedback retained schema objects differ")
    def _feedback_rows(self) -> dict[str, EvaluationFeedbackAcceptance]:
        feedback_rows = self._connection.execute(
            "SELECT * FROM evaluation_feedback ORDER BY feedback_id"
        ).fetchall()
        obligation_rows = self._connection.execute(
            "SELECT * FROM evaluation_reconciliation_obligations ORDER BY obligation_id"
        ).fetchall()
        obligations: dict[str, ReconciliationObligation] = {}
        for row in obligation_rows:
            value = ReconciliationObligation.from_canonical_bytes(bytes(row[1]))
            if (
                tuple(row[0:11])
                != (
                    value.obligation_id,
                    value.canonical_bytes,
                    value.canonical_digest,
                    value.feedback_id,
                    value.feedback_digest,
                    value.candidate_id,
                    value.candidate_version_id,
                    value.request_id,
                    value.actor_identity_digest,
                    value.idempotency_key,
                    row[10],
                )
            ):
                raise FeedbackContractError("retained obligation representation differs")
            if value.feedback_id in obligations:
                raise FeedbackContractError("retained obligation cardinality differs")
            obligations[value.feedback_id] = value
        result: dict[str, EvaluationFeedbackAcceptance] = {}
        for row in feedback_rows:
            feedback = EvaluationFeedback.from_canonical_bytes(bytes(row[1]))
            snapshot = HandoffAcceptanceSnapshot.from_canonical_bytes(bytes(row[15]))
            if (
                tuple(row[0:12])
                != (
                    feedback.feedback_id,
                    feedback.canonical_bytes,
                    feedback.canonical_digest,
                    feedback.source_feedback_id,
                    feedback.handoff_id,
                    feedback.acknowledgement_id,
                    feedback.candidate_id,
                    feedback.candidate_version_id,
                    feedback.candidate_version_digest,
                    feedback.request_id,
                    feedback.actor_identity_digest,
                    feedback.idempotency_key,
                )
                or int(row[14]) != 1
                or bytes(row[15]) != snapshot.canonical_bytes
                or str(row[16]) != snapshot.canonical_digest
                or snapshot.handoff_id != feedback.handoff_id
                or snapshot.candidate_version_id != feedback.candidate_version_id
                or snapshot.governing_manifest_digest != feedback.governing_manifest_digest
                or snapshot.sink_id != feedback.sink_id
                or not any(
                    attempt.attempt_id == feedback.handoff_attempt_id
                    and attempt.attempt_number == feedback.handoff_attempt_number
                    and attempt.sent
                    for attempt in snapshot.attempts
                )
                or not any(
                    acknowledgement.acknowledgement_id == feedback.acknowledgement_id
                    and acknowledgement.attempt_id == feedback.handoff_attempt_id
                    and acknowledgement.response_digest
                    == feedback.acknowledgement_response_digest
                    and acknowledgement.candidate_version_id
                    == feedback.candidate_version_id
                    and acknowledgement.governing_manifest_digest
                    == feedback.governing_manifest_digest
                    and acknowledgement.sink_id == feedback.sink_id
                    for acknowledgement in snapshot.acknowledgements
                )
            ):
                raise FeedbackContractError("retained feedback representation differs")
            obligation = obligations.pop(feedback.feedback_id, None)
            if obligation is None or obligation.feedback_digest != feedback.canonical_digest:
                raise FeedbackContractError("mandatory reconciliation obligation differs")
            obligation_row = next(
                item for item in obligation_rows if str(item[0]) == obligation.obligation_id
            )
            if (
                str(obligation_row[10]) != str(row[12])
                or str(obligation_row[11]) != str(row[13])
                or int(obligation_row[12]) != 1
                or str(obligation_row[13]) != str(row[17])
            ):
                raise FeedbackContractError("obligation authority event binding differs")
            payload = self._ledger_payload(
                str(row[12]), str(row[13]), 1,
                actor=feedback.actor_identity_digest,
                idempotency_key=feedback.idempotency_key,
                recorded_at=str(row[17]),
            )
            if payload != self._payload(
                "accept",
                feedback=feedback,
                obligation=obligation,
                snapshot=snapshot,
                disposition=None,
            ):
                raise FeedbackContractError("feedback authority payload differs")
            result[feedback.feedback_id] = EvaluationFeedbackAcceptance(
                feedback, obligation, snapshot
            )
        if obligations:
            raise FeedbackContractError("orphan reconciliation obligation")
        return result
    def _history(
        self, obligation: ReconciliationObligation
    ) -> tuple[ReconciliationDisposition, ...]:
        rows = self._connection.execute(
            "SELECT * FROM evaluation_reconciliation_dispositions "
            "WHERE obligation_id=? ORDER BY ordinal",
            (obligation.obligation_id,),
        ).fetchall()
        values: list[ReconciliationDisposition] = []
        for row in rows:
            value = ReconciliationDisposition.from_canonical_bytes(bytes(row[1]))
            if tuple(row[0:11]) != (
                value.disposition_id,
                value.canonical_bytes,
                value.canonical_digest,
                value.obligation_id,
                value.ordinal,
                value.previous_disposition_id,
                value.previous_disposition_digest,
                value.outcome.value,
                value.request_id,
                value.actor_identity_digest,
                value.idempotency_key,
            ):
                raise FeedbackContractError("retained disposition representation differs")
            acceptance = self._connection.execute(
                "SELECT authority_aggregate_id FROM evaluation_feedback WHERE feedback_id=?",
                (obligation.feedback_id,),
            ).fetchone()
            if (
                acceptance is None
                or str(row[11]) != str(acceptance[0])
                or int(row[13]) != value.ordinal + 1
                or self._ledger_payload(
                    str(row[11]), str(row[12]), int(row[13]),
                    actor=value.actor_identity_digest,
                    idempotency_key=value.idempotency_key,
                    recorded_at=str(row[14]),
                )
                != self._payload(
                    "append_disposition",
                    feedback=None,
                    obligation=None,
                    snapshot=None,
                    disposition=value,
                )
            ):
                raise FeedbackContractError("disposition authority event binding differs")
            values.append(value)
        return validate_reconciliation_history(obligation, tuple(values))
    def _ledger_payload(
        self,
        aggregate_id: str,
        event_id: str,
        aggregate_version: int,
        *,
        actor: str,
        idempotency_key: str,
        recorded_at: str,
    ) -> dict[str, object]:
        definition = evaluation_feedback_command_definition()
        row = self._connection.execute(
            "SELECT p.payload_bytes,e.aggregate_type,e.aggregate_id,e.aggregate_version,"
            "c.aggregate_type,c.aggregate_id,v.aggregate_version,g.current_version,"
            "c.command_type,c.command_definition_version,c.command_definition_digest,c.expected_aggregate_version,"
            "c.idempotency_key,c.committed_at,c.producer_version,"
            "e.event_type,e.event_schema_version,e.producer_version,e.recorded_at,"
            "e.payload_mode,e.payload_schema_version,e.payload_schema_contract_version,"
            "e.payload_schema_contract_digest,e.payload_canonicalizer_version,"
            "e.security_scope,e.retention_scope,e.trust_scope,"
            "v.recorded_at,p.created_at,au.recorded_at,"
            "a.principal_id,a.credential_binding_digest,"
            "c.authentication_context_id,c.authorization_request_digest,c.authorization_decision_id,"
            "e.authentication_context_id,e.authorization_request_digest,e.authorization_decision_id,"
            "au.authentication_context_id,au.authorization_request_digest,au.authorization_decision_id,"
            "au.event_type,au.detail_digest,r.operation_type,r.required_scope,d.allowed,s.canonical_bytes,"
            "a.authority_domain,c.command_id,c.result_bytes,c.result_digest,e.ledger_seq,"
            "e.correlation_id,e.causation_kind,e.causation_identifier,e.causation_external_system "
            "FROM ledger_events e JOIN authority_commands c ON c.command_id=e.command_id "
            "JOIN authority_payloads p ON p.payload_id=e.payload_id "
            "JOIN authority_aggregate_versions v ON v.command_id=c.command_id "
            "JOIN authority_aggregates g ON g.aggregate_type=v.aggregate_type "
            "AND g.aggregate_id=v.aggregate_id "
            "JOIN authority_audit_events au ON au.command_id=c.command_id "
            "JOIN authentication_contexts a ON a.authentication_context_id=c.authentication_context_id "
            "JOIN authorization_requests r ON r.request_digest=c.authorization_request_digest "
            "JOIN authorization_decisions d ON d.authorization_decision_id=c.authorization_decision_id "
            "JOIN authorization_scope_contents s ON s.scope_content_digest=d.scope_content_digest "
            "WHERE e.event_id=?",
            (event_id,),
        ).fetchone()
        if row is None or tuple(row[1:7]) != (
            "evaluation_reconciliation",
            aggregate_id,
            aggregate_version,
            "evaluation_reconciliation",
            aggregate_id,
            aggregate_version,
        ) or int(row[7]) < aggregate_version:
            raise FeedbackContractError("Feedback generic ledger binding differs")
        contract = self._event_store._payload_schemas.resolve_exact(
            definition.payload_schema_version,
            definition.payload_mode,
            definition.payload_schema_contract_version,
            definition.payload_schema_contract_digest,
            definition.payload_canonicalizer_version,
        )
        route = (
            definition.command_type,
            definition.definition_version,
            definition.digest,
            aggregate_version - 1,
            idempotency_key,
            recorded_at,
            "increment6-feedback-v25",
            definition.event_type,
            definition.event_schema_version,
            "increment6-feedback-v25",
            recorded_at,
            definition.payload_mode.value,
            contract.schema_version,
            contract.contract_version,
            contract.contract_digest,
            contract.canonicalizer_implementation_version,
            definition.security_scope,
            definition.retention_scope,
            definition.trust_scope.value,
            recorded_at,
            recorded_at,
            recorded_at,
        )
        if tuple(row[8:30]) != route or digest_bytes(
            canonical_json_bytes(
                {
                    "principal_id": str(row[30]),
                    "credential_binding_digest": str(row[31]),
                }
            )
        ) != actor:
            raise FeedbackContractError("Feedback authority route binding differs")
        command_triple = tuple(row[32:35])
        if (
            command_triple != tuple(row[35:38])
            or command_triple != tuple(row[38:41])
            or row[41] != definition.event_type
            or row[43] != f"command:{definition.command_type}"
            or row[44] != definition.required_scope
            or row[45] != 1
            or definition.required_scope not in json.loads(bytes(row[46]))
        ):
            raise FeedbackContractError("Feedback authorization envelope differs")
        if type(row[47]) is not str:
            raise FeedbackContractError("Feedback authentication authority differs")
        expected_namespace = digest_canonical({
            "authority_domain": row[47],
            "principal_id": str(row[30]),
            "command_type": definition.command_type,
        })
        try:
            result = self._event_store._decode_result(
                bytes(row[49]), str(row[50]), replayed=False
            )
        except Exception as exc:
            raise FeedbackContractError("Feedback command result differs") from exc
        actual_result = (
            str(result.command_id),
            result.aggregate_type,
            str(result.aggregate_id),
            result.aggregate_version,
            result.ledger_seq,
            str(result.event_id),
        )
        expected_result = (
            str(row[48]),
            definition.aggregate_type,
            aggregate_id,
            aggregate_version,
            int(row[51]),
            event_id,
        )
        causation = tuple(row[53:56])
        if actual_result != expected_result or causation != (None, None, None):
            raise FeedbackContractError("Feedback command result or causation differs")
        try:
            value = json.loads(bytes(row[0]))
        except Exception as exc:
            raise FeedbackContractError("Feedback authority payload is malformed") from exc
        if type(value) is not dict:
            raise FeedbackContractError("Feedback authority payload differs")
        payload_digest = digest_bytes(bytes(row[0]))
        envelope = self._connection.execute(
            "SELECT c.idempotency_namespace,c.stable_semantic_request_digest,a.canonical_digest,"
            "r.canonical_record_digest,d.canonical_digest,e.correlation_id,au.detail_digest,v.trust_scope "
            "FROM ledger_events e JOIN authority_commands c ON c.command_id=e.command_id "
            "JOIN authority_aggregate_versions v ON v.command_id=c.command_id "
            "JOIN authentication_contexts a ON a.authentication_context_id=c.authentication_context_id "
            "JOIN authorization_requests r ON r.request_digest=c.authorization_request_digest "
            "JOIN authorization_decisions d ON d.authorization_decision_id=c.authorization_decision_id "
            "JOIN authority_audit_events au ON au.command_id=c.command_id WHERE e.event_id=?",
            (event_id,),
        ).fetchone()
        payload_identity = {
            "kind": definition.payload_mode.value, "schema_version": contract.schema_version,
            "schema_contract_version": contract.contract_version,
            "schema_contract_digest": contract.contract_digest,
            "canonicalizer_version": contract.canonicalizer_implementation_version,
            "digest": payload_digest, "inline_digest": payload_digest,
            "object_admission_id": None, "blob_digest": None, "object_class": None, "allowed_use": None,
        }
        stable = digest_canonical({"command_type": definition.command_type,
            "command_definition_version": definition.definition_version, "command_definition_digest": definition.digest,
            "aggregate_type": definition.aggregate_type, "aggregate_id": aggregate_id,
            "expected_aggregate_version": aggregate_version - 1, "payload": payload_identity})
        unsigned = {"operation": "COMMAND_COMMIT", "command_type": definition.command_type,
            "aggregate_id": aggregate_id, "expected_aggregate_version": aggregate_version - 1,
            "definition_digest": definition.digest, "definition_version": definition.definition_version,
            "payload": payload_identity, "authentication_context_digest": envelope[2],
            "authorization_request_record_digest": envelope[3], "authorization_request_digest": command_triple[1],
            "authorization_decision_digest": envelope[4], "idempotency_namespace": expected_namespace,
            "idempotency_key": idempotency_key, "stable_semantic_request_digest": stable,
            "correlation_id": row[52], "causation_kind": causation[0],
            "causation_identifier": causation[1], "causation_external_system": causation[2],
            "replay_of_command_id": None}
        if (envelope[0] != expected_namespace or envelope[1] != stable
            or envelope[5] != row[52] or envelope[6] != digest_canonical(unsigned)
            or envelope[7] != definition.trust_scope.value):
            raise FeedbackContractError("Feedback audit envelope differs")
        return value
    def _verify_local(self) -> dict[str, EvaluationFeedbackAcceptance]:
        try:
            self._event_store._validate_schema_and_integrity()
        except AuthorityPersistenceError as exc:
            raise FeedbackContractError("Feedback retained command authority differs") from exc
        self._verify_schema()
        accepted = self._feedback_rows()
        obligations = {item.obligation.obligation_id: item.obligation for item in accepted.values()}
        event_ids: set[str] = set()
        for item in accepted.values():
            history = self._history(item.obligation)
            row = self._connection.execute(
                "SELECT authority_aggregate_id,authority_event_id FROM evaluation_feedback "
                "WHERE feedback_id=?",
                (item.feedback.feedback_id,),
            ).fetchone()
            if row is None:
                raise FeedbackContractError("feedback aggregate binding is absent")
            current = self._connection.execute(
                "SELECT current_version FROM authority_aggregates WHERE "
                "aggregate_type='evaluation_reconciliation' AND aggregate_id=?",
                (row[0],),
            ).fetchone()
            if current is None or tuple(current) != (1 + len(history),):
                raise FeedbackContractError("Feedback aggregate head differs")
            event_ids.add(str(row[1]))
            event_ids.update(
                str(event[0])
                for event in self._connection.execute(
                    "SELECT authority_event_id FROM evaluation_reconciliation_dispositions "
                    "WHERE obligation_id=?",
                    (item.obligation.obligation_id,),
                )
            )
        count = self._connection.execute(
            "SELECT COUNT(*) FROM evaluation_reconciliation_dispositions"
        ).fetchone()[0]
        if count != sum(len(self._history(item)) for item in obligations.values()):
            raise FeedbackContractError("orphan reconciliation disposition")
        generic_event_ids = {
            str(row[0])
            for row in self._connection.execute(
                "SELECT e.event_id FROM ledger_events e JOIN authority_commands c "
                "ON c.command_id=e.command_id WHERE "
                "c.command_type='evaluation-feedback.reconcile'"
            )
        }
        if generic_event_ids != event_ids:
            raise FeedbackContractError("Feedback generic event coverage differs")
        if self._connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise FeedbackContractError("Feedback foreign keys differ")
        return accepted
    @staticmethod
    def _payload(
        operation: str,
        *,
        feedback: EvaluationFeedback | None,
        obligation: ReconciliationObligation | None,
        snapshot: HandoffAcceptanceSnapshot | None,
        disposition: ReconciliationDisposition | None,
    ) -> dict[str, object]:
        return {
            "operation": operation,
            "feedback": None if feedback is None else json.loads(feedback.canonical_bytes),
            "obligation": None if obligation is None else json.loads(obligation.canonical_bytes),
            "handoff_acceptance_snapshot": (
                None if snapshot is None else json.loads(snapshot.canonical_bytes)
            ),
            "disposition": (
                None if disposition is None else json.loads(disposition.canonical_bytes)
            ),
        }
    @staticmethod
    def _actor(grant: object) -> str:
        authentication = grant.authentication
        return digest_bytes(
            canonical_json_bytes(
                {
                    "principal_id": authentication.principal_id,
                    "credential_binding_digest": authentication.credential_binding_digest,
                }
            )
        )
    def _grant(
        self,
        *,
        aggregate_id: str,
        expected_version: int,
        idempotency_key: str,
        payload: dict[str, object],
        proof: object,
    ):
        command = SemanticCommand(
            command_type="evaluation-feedback.reconcile",
            aggregate_id=AggregateId.parse(aggregate_id),
            expected_aggregate_version=expected_version,
            payload=InlinePayload(payload),
            idempotency_key=idempotency_key,
        )
        return self._service._authorize_for_commit(command, proof=proof)
    @staticmethod
    def _find_collision(
        accepted: dict[str, EvaluationFeedbackAcceptance], feedback: EvaluationFeedback
    ) -> EvaluationFeedbackAcceptance | None:
        for item in accepted.values():
            old = item.feedback
            if old.feedback_id == feedback.feedback_id:
                if old.canonical_bytes == feedback.canonical_bytes:
                    return item
                raise FeedbackContractError("feedback identity binding conflict")
            if (
                old.request_id == feedback.request_id
                or (old.actor_identity_digest, old.idempotency_key)
                == (feedback.actor_identity_digest, feedback.idempotency_key)
                or old.handoff_id == feedback.handoff_id
                or old.acknowledgement_id == feedback.acknowledgement_id
            ):
                raise FeedbackContractError("feedback request binding conflict")
        return None
    def _current_correlation(
        self, feedback: EvaluationFeedback, *, candidate_proof: object
    ) -> tuple[object, object, FeedbackCorrelationOutcome]:
        version, handoff, outcome = self._retained_correlation(feedback)
        current = self._candidate.require_current_head_in_transaction(
            feedback.candidate_id, proof=candidate_proof
        )
        if current.version_id != version.version_id:
            raise FeedbackContractError("fresh feedback requires current Candidate head")
        return version, handoff, outcome
    def _retained_correlation(
        self, feedback: EvaluationFeedback
    ) -> tuple[object, object, FeedbackCorrelationOutcome]:
        version = self._candidate.require_retained_version_in_transaction(
            feedback.candidate_version_id
        )
        handoff = self._handoff.require_retained_handoff_in_transaction(feedback.handoff_id)
        outcome = correlate_evaluation_feedback(handoff, version, feedback, ())
        return version, handoff, outcome
    def _accept_active(
        self,
        feedback: EvaluationFeedback,
        obligation: ReconciliationObligation,
        candidate_proof: object,
    ) -> EvaluationFeedbackAcceptance:
        accepted = self._verify_local()
        replay = self._find_collision(accepted, feedback)
        if replay is not None:
            if replay.obligation.canonical_bytes != obligation.canonical_bytes:
                raise FeedbackContractError("obligation replay binding conflict")
            return replay
        expected = create_reconciliation_obligation(
            feedback,
            request_id=obligation.request_id,
            actor_identity_digest=obligation.actor_identity_digest,
            idempotency_key=obligation.idempotency_key,
        )
        if expected.canonical_bytes != obligation.canonical_bytes:
            raise FeedbackContractError("mandatory obligation derivation differs")
        if obligation.actor_identity_digest != feedback.actor_identity_digest:
            raise FeedbackContractError("feedback and obligation actor differ")
        _, handoff, outcome = self._current_correlation(
            feedback, candidate_proof=candidate_proof
        )
        if outcome not in {
            FeedbackCorrelationOutcome.READY,
            FeedbackCorrelationOutcome.DELAYED_READY,
        }:
            raise FeedbackContractError(f"feedback correlation is {outcome.value}")
        snapshot = HandoffAcceptanceSnapshot.observe(handoff)
        recorded_at = self._recorded_at()
        aggregate_id = str(AggregateId.new())
        payload = self._payload(
            "accept",
            feedback=feedback,
            obligation=obligation,
            snapshot=snapshot,
            disposition=None,
        )
        grant = self._grant(
            aggregate_id=aggregate_id,
            expected_version=0,
            idempotency_key=feedback.idempotency_key,
            payload=payload,
            proof=candidate_proof,
        )
        if self._actor(grant) != feedback.actor_identity_digest:
            raise FeedbackContractError("authenticated feedback actor differs")
        committed = self._event_store._commit_grant_in_transaction(
            self._connection, grant, recorded_at=recorded_at
        )
        if committed.replayed or committed.aggregate_version != 1:
            raise FeedbackContractError("fresh feedback command replayed")
        self._connection.execute(
            "INSERT INTO evaluation_feedback VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                feedback.feedback_id, feedback.canonical_bytes, feedback.canonical_digest,
                feedback.source_feedback_id, feedback.handoff_id,
                feedback.acknowledgement_id, feedback.candidate_id,
                feedback.candidate_version_id, feedback.candidate_version_digest,
                feedback.request_id, feedback.actor_identity_digest, feedback.idempotency_key,
                aggregate_id, committed.event_id, committed.aggregate_version,
                snapshot.canonical_bytes, snapshot.canonical_digest, recorded_at,
            ),
        )
        self._connection.execute(
            "INSERT INTO evaluation_reconciliation_obligations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                obligation.obligation_id, obligation.canonical_bytes,
                obligation.canonical_digest, obligation.feedback_id,
                obligation.feedback_digest, obligation.candidate_id,
                obligation.candidate_version_id, obligation.request_id,
                obligation.actor_identity_digest, obligation.idempotency_key,
                aggregate_id, committed.event_id, committed.aggregate_version, recorded_at,
            ),
        )
        result = self._verify_local()[feedback.feedback_id]
        return result
    def accept(
        self, feedback_bytes: bytes, obligation_bytes: bytes, *, candidate_proof: object
    ) -> EvaluationFeedbackAcceptance:
        feedback = EvaluationFeedback.from_canonical_bytes(feedback_bytes)
        obligation = ReconciliationObligation.from_canonical_bytes(obligation_bytes)
        return self._run(
            lambda: self._accept_active(feedback, obligation, candidate_proof)
        )  # type: ignore[return-value]
    def _append_active(
        self, proposed: ReconciliationDisposition, candidate_proof: object
    ) -> ReconciliationDisposition:
        accepted = self._verify_local()
        parent = next(
            (item for item in accepted.values() if item.obligation.obligation_id == proposed.obligation_id),
            None,
        )
        if parent is None:
            raise FeedbackContractError("unknown reconciliation obligation")
        history = self._history(parent.obligation)
        retained_dispositions = tuple(
            ReconciliationDisposition.from_canonical_bytes(bytes(row[0]))
            for row in self._connection.execute(
                "SELECT disposition_bytes FROM evaluation_reconciliation_dispositions "
                "ORDER BY obligation_id,ordinal"
            )
        )
        for old in retained_dispositions:
            if old.disposition_id == proposed.disposition_id or old.request_id == proposed.request_id or (
                old.actor_identity_digest, old.idempotency_key
            ) == (proposed.actor_identity_digest, proposed.idempotency_key):
                if old.canonical_bytes == proposed.canonical_bytes:
                    return old
                raise FeedbackContractError("disposition request binding conflict")
        rebuilt = append_reconciliation_disposition(
            parent.obligation,
            history,
            outcome=proposed.outcome,
            reason=proposed.reason,
            resolution_digest=proposed.resolution_digest,
            request_id=proposed.request_id,
            actor_identity_digest=proposed.actor_identity_digest,
            idempotency_key=proposed.idempotency_key,
            expected_current_disposition_id=proposed.expected_current_disposition_id,
            expected_current_disposition_digest=proposed.expected_current_disposition_digest,
            expected_current_ordinal=proposed.expected_current_ordinal,
            supplemental_reentry=proposed.supplemental_reentry,
        )
        if rebuilt.canonical_bytes != proposed.canonical_bytes:
            raise FeedbackContractError("disposition derivation differs")
        _, _, outcome = self._retained_correlation(parent.feedback)
        self._candidate.require_current_head_in_transaction(
            parent.feedback.candidate_id, proof=candidate_proof
        )
        if outcome not in {
            FeedbackCorrelationOutcome.READY,
            FeedbackCorrelationOutcome.DELAYED_READY,
        }:
            raise FeedbackContractError(f"current feedback correlation is {outcome.value}")
        aggregate_row = self._connection.execute(
            "SELECT authority_aggregate_id FROM evaluation_feedback WHERE feedback_id=?",
            (parent.feedback.feedback_id,),
        ).fetchone()
        if aggregate_row is None:
            raise FeedbackContractError("feedback aggregate binding is absent")
        aggregate_id = str(aggregate_row[0])
        recorded_at = self._recorded_at()
        grant = self._grant(
            aggregate_id=aggregate_id,
            expected_version=proposed.ordinal,
            idempotency_key=proposed.idempotency_key,
            payload=self._payload(
                "append_disposition",
                feedback=None,
                obligation=None,
                snapshot=None,
                disposition=proposed,
            ),
            proof=candidate_proof,
        )
        if self._actor(grant) != proposed.actor_identity_digest:
            raise FeedbackContractError("authenticated disposition actor differs")
        committed = self._event_store._commit_grant_in_transaction(
            self._connection, grant, recorded_at=recorded_at
        )
        if committed.replayed or committed.aggregate_version != proposed.ordinal + 1:
            raise FeedbackContractError("fresh disposition command replayed")
        self._connection.execute(
            "INSERT INTO evaluation_reconciliation_dispositions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                proposed.disposition_id, proposed.canonical_bytes,
                proposed.canonical_digest, proposed.obligation_id, proposed.ordinal,
                proposed.previous_disposition_id, proposed.previous_disposition_digest,
                proposed.outcome.value, proposed.request_id,
                proposed.actor_identity_digest, proposed.idempotency_key,
                aggregate_id, committed.event_id, committed.aggregate_version,
                recorded_at,
            ),
        )
        self._verify_local()
        return proposed
    def append_disposition(
        self, disposition_bytes: bytes, *, candidate_proof: object
    ) -> ReconciliationDisposition:
        proposed = ReconciliationDisposition.from_canonical_bytes(disposition_bytes)
        return self._run(
            lambda: self._append_active(proposed, candidate_proof)
        )  # type: ignore[return-value]
    def _load_active(self, feedback_id: str) -> EvaluationFeedbackAcceptance:
        result = self._verify_local().get(feedback_id)
        if result is None:
            raise FeedbackContractError("unknown Evaluation Feedback")
        return result
    def load(self, feedback_id: str) -> EvaluationFeedbackAcceptance:
        return self._run(lambda: self._load_active(feedback_id))  # type: ignore[return-value]
    def _dispositions_active(
        self, obligation_id: str
    ) -> tuple[ReconciliationDisposition, ...]:
        accepted = self._verify_local()
        obligation = next(
            (item.obligation for item in accepted.values() if item.obligation.obligation_id == obligation_id),
            None,
        )
        if obligation is None:
            raise FeedbackContractError("unknown reconciliation obligation")
        result = self._history(obligation)
        return result
    def dispositions(self, obligation_id: str) -> tuple[ReconciliationDisposition, ...]:
        return self._run(
            lambda: self._dispositions_active(obligation_id)
        )  # type: ignore[return-value]
    def rollback_scope(self, operation: Callable[[object], None]) -> None:
        self._run(lambda: operation(_EvaluationFeedbackTransactionView(self)), commit=False)
    def close(self) -> None:
        with self._lock:
            if not self._closed:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                self._event_store.close()
                self._closed = True
class _EvaluationFeedbackTransactionView:
    def __init__(self, root: _EvaluationFeedbackAuthorityRoot) -> None:
        self._root = root
    def accept(self, feedback_bytes: bytes, obligation_bytes: bytes, *, candidate_proof: object):
        return self._root._accept_active(
            EvaluationFeedback.from_canonical_bytes(feedback_bytes),
            ReconciliationObligation.from_canonical_bytes(obligation_bytes), candidate_proof,
        )
    def append_disposition(self, disposition_bytes: bytes, *, candidate_proof: object):
        return self._root._append_active(
            ReconciliationDisposition.from_canonical_bytes(disposition_bytes), candidate_proof
        )
    def load(self, feedback_id: str):
        return self._root._load_active(feedback_id)
    def dispositions(self, obligation_id: str):
        return self._root._dispositions_active(obligation_id)
class _UnlockedEvaluationFeedbackEventStoreForTest(_EventAuthorityStore):
    def _acquire_writer_lock(self) -> None:
        self._lock_fd = None
def _compose_evaluation_feedback_authority_root(
    database: str | Path,
    *,
    retrieval_authority: object,
    authenticator: object,
    authorizer: object,
    command_registry: CommandRegistry,
    payload_schemas: PayloadSchemaRegistry,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
    busy_timeout_ms: int = 5_000,
    store_class: type[_EventAuthorityStore] = _EventAuthorityStore,
) -> _EvaluationFeedbackAuthorityRoot:
    relationship_commands, relationship_schemas = merge_relationship_authority_registries(
        command_registry, payload_schemas
    )
    lineage_commands, lineage_schemas = merge_lineage_authority_registries(
        relationship_commands, relationship_schemas
    )
    candidate_commands, candidate_schemas = merge_candidate_authority_registries(
        lineage_commands, lineage_schemas
    )
    commands, schemas = merge_evaluation_feedback_authority_registries(
        candidate_commands, candidate_schemas
    )
    issuer = _CapabilityIssuer(command_registry=commands, payload_schemas=schemas)
    event_store = store_class(
        Path(database),
        issuer=issuer,
        command_registry=commands,
        payload_schemas=schemas,
        command_service_version="increment6-feedback-v25",
        busy_timeout_ms=busy_timeout_ms,
        clock=clock,
    )
    connection = event_store._connection
    try:
        candidate = _create_story_candidate_read_port(
            connection,
            retrieval_authority=retrieval_authority,  # type: ignore[arg-type]
            authenticator=authenticator,
            command_registry=commands,
            payload_schemas=schemas,
            clock=clock,
        )
        handoff = _create_evaluation_handoff_read_port(connection)
        service = CommandService(
            registry=commands,
            payload_schemas=schemas,
            authenticator=authenticator,
            authorizer=authorizer,
            committed_lookup=event_store,
            clock=clock,
            _issuer=issuer,
        )
        return _EvaluationFeedbackAuthorityRoot(event_store, service, candidate, handoff, clock)
    except BaseException:
        try: event_store.close()
        except BaseException:  # noqa: BLE001, S110 - preserve composition failure
            pass
        raise
def open_evaluation_feedback_authority_system(
    database: str | Path,
    *,
    retrieval_authority: object,
    authenticator: object,
    authorizer: object,
    command_registry: CommandRegistry,
    payload_schemas: PayloadSchemaRegistry,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
    busy_timeout_ms: int = 5_000,
) -> EvaluationFeedbackAuthority:
    root = _compose_evaluation_feedback_authority_root(
        database, retrieval_authority=retrieval_authority, authenticator=authenticator, authorizer=authorizer,
        command_registry=command_registry, payload_schemas=payload_schemas, clock=clock,
        busy_timeout_ms=busy_timeout_ms, store_class=_EventAuthorityStore,
    )
    try:
        facade = _compose_evaluation_feedback_authority(root)
        if type(facade) is not EvaluationFeedbackAuthority:
            raise FeedbackContractError("Feedback facade composition differs")
        return facade
    except BaseException:
        try: root.close()
        except BaseException:  # noqa: BLE001, S110 - preserve composition failure
            pass
        raise
def _open_unlocked_evaluation_feedback_authority_for_test(*args: object, **kwargs: object):
    return _compose_evaluation_feedback_authority_root(
        *args, store_class=_UnlockedEvaluationFeedbackEventStoreForTest, **kwargs
    )
__all__ = ["open_evaluation_feedback_authority_system"]
# fmt: on
