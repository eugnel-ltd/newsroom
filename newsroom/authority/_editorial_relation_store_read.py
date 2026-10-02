from __future__ import annotations

import sqlite3

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.authority.types import EventId, TrustScope, UtcTimestamp
from newsroom.relations.editorial_models import (
    EDITORIAL_PREDICATE_REGISTRY_V1,
    EditorialRelationAssertion,
    EditorialRelationCurrentView,
    EditorialRelationDecision,
    EditorialRelationProducer,
    EditorialRelationProjectionEvent,
    EditorialRelationProposal,
    EditorialRelationProposalVersion,
    EditorialRelationTemporalScope,
    evidence_canonical_value,
    endpoint_canonical_value,
)
from newsroom.relations.editorial_types import (
    EditorialPredicateCode,
    EditorialRelationAssertionId,
    EditorialRelationAssertionLifecycle,
    EditorialRelationCurrentState,
    EditorialRelationDecisionAction,
    EditorialRelationDecisionId,
    EditorialRelationProducerKind,
    EditorialRelationProjectionAction,
    EditorialRelationProposalId,
    EditorialRelationProposalVersionId,
    EditorialRelationStateError,
)

from ._editorial_relation_decoding import (
    decode_editorial_relation_decision_request,
    decode_editorial_relation_proposal_request,
)


class _EditorialRelationReadMixin:
    @staticmethod
    def _editorial_required_row(
        conn: sqlite3.Connection,
        *,
        table: str,
        column: str,
        identifier: str,
        identity: str,
    ) -> sqlite3.Row:
        row = conn.execute(
            f"SELECT * FROM {table} WHERE {column}=?", (identifier,)
        ).fetchone()
        if row is None:
            raise EditorialRelationStateError(f"{identity} is not retained")
        return row

    def _editorial_proposal_from_row(
        self, conn: sqlite3.Connection, row: sqlite3.Row
    ) -> EditorialRelationProposal:
        subject_row = self._editorial_required_row(
            conn,
            table="editorial_relation_endpoints",
            column="endpoint_digest",
            identifier=str(row["subject_endpoint_digest"]),
            identity="relation subject endpoint",
        )
        object_row = self._editorial_required_row(
            conn,
            table="editorial_relation_endpoints",
            column="endpoint_digest",
            identifier=str(row["object_endpoint_digest"]),
            identity="relation object endpoint",
        )
        subject = self._editorial_endpoint_from_row(subject_row)
        object_ = self._editorial_endpoint_from_row(object_row)
        producer = EditorialRelationProducer(
            kind=EditorialRelationProducerKind(str(row["producer_kind"])),
            producer_id=str(row["producer_id"]),
            producer_version=str(row["producer_version"]),
            contract_digest=str(row["producer_contract_digest"]),
        )
        contract = EDITORIAL_PREDICATE_REGISTRY_V1.contract(
            EditorialPredicateCode(str(row["predicate"]))
        )
        value = {
            "proposal_id": str(row["proposal_id"]),
            "registry_version": str(row["registry_version"]),
            "predicate_registry_digest": str(row["predicate_registry_digest"]),
            "predicate": str(row["predicate"]),
            "predicate_contract_version": str(row["predicate_contract_version"]),
            "predicate_contract_digest": str(row["predicate_contract_digest"]),
            "subject": endpoint_canonical_value(subject),
            "object": endpoint_canonical_value(object_),
            "producer": producer.canonical_value(),
            "semantic_slot_digest": str(row["semantic_slot_digest"]),
            "stable_semantic_digest": str(row["stable_semantic_digest"]),
        }
        data = canonical_json_bytes(value)
        if (
            str(row["registry_version"])
            != EDITORIAL_PREDICATE_REGISTRY_V1.registry_version
            or str(row["predicate_registry_digest"])
            != EDITORIAL_PREDICATE_REGISTRY_V1.digest
            or str(row["predicate_contract_version"])
            != contract.contract_version
            or str(row["predicate_contract_digest"]) != contract.digest
            or bytes(row["canonical_bytes"]) != data
            or str(row["canonical_digest"]) != digest_bytes(data)
        ):
            raise AuthorityPersistenceError(
                "editorial relation proposal base differs from canonical authority"
            )
        return EditorialRelationProposal(
            proposal_id=EditorialRelationProposalId.parse(str(row["proposal_id"])),
            predicate_registry_digest=str(row["predicate_registry_digest"]),
            predicate_contract_digest=str(row["predicate_contract_digest"]),
            predicate=EditorialPredicateCode(str(row["predicate"])),
            subject=subject,
            object=object_,
            producer=producer,
            semantic_slot_digest=str(row["semantic_slot_digest"]),
            stable_semantic_digest=str(row["stable_semantic_digest"]),
            canonical_digest=str(row["canonical_digest"]),
            created_by_event_id=EventId.parse(str(row["created_by_event_id"])),
            created_at=UtcTimestamp.parse(str(row["created_at"])),
        )

    def _validate_editorial_proposal_children(
        self,
        conn: sqlite3.Connection,
        *,
        proposal_version_id: str,
        request,
    ) -> None:
        evidence_rows = conn.execute(
            "SELECT * FROM editorial_relation_evidence_items "
            "WHERE proposal_version_id=? ORDER BY evidence_ordinal",
            (proposal_version_id,),
        ).fetchall()
        if len(evidence_rows) != len(request.evidence):
            raise AuthorityPersistenceError(
                "editorial relation evidence count differs from request"
            )
        for ordinal, (row, item) in enumerate(zip(evidence_rows, request.evidence)):
            data = item.canonical_bytes
            if (
                int(row["evidence_ordinal"]) != ordinal
                or str(row["evidence_kind"]) != item.kind.value
                or bytes(row["canonical_bytes"]) != data
                or str(row["canonical_digest"]) != digest_bytes(data)
            ):
                raise AuthorityPersistenceError(
                    "editorial relation evidence item differs from request"
                )
            if item.kind.value == "EXTRACTION_PROPOSAL":
                child = conn.execute(
                    "SELECT * FROM editorial_relation_extraction_evidence "
                    "WHERE proposal_version_id=? AND evidence_ordinal=?",
                    (proposal_version_id, ordinal),
                ).fetchone()
                expected = item.canonical_value()
                if child is None or any(
                    str(child[column]) != str(expected[key])
                    for column, key in (
                        ("source_proposal_id", "source_proposal_id"),
                        ("source_proposal_digest", "source_proposal_digest"),
                        ("run_id", "run_id"),
                        ("run_version_id", "run_version_id"),
                        ("output_id", "output_id"),
                        ("passage_id", "passage_id"),
                        ("evidence_text_digest", "evidence_text_digest"),
                    )
                ) or int(child["source_evidence_ordinal"]) != (
                    int(expected["source_evidence_ordinal"]) + 1
                ) or any(
                    int(child[column]) != int(expected[key])
                    for column, key in (
                        ("start_byte", "start_byte"),
                        ("end_byte", "end_byte"),
                    )
                ):
                    raise AuthorityPersistenceError(
                        "editorial relation extraction evidence differs"
                    )
            else:
                child = conn.execute(
                    "SELECT * FROM editorial_relation_workflow_evidence "
                    "WHERE proposal_version_id=? AND evidence_ordinal=?",
                    (proposal_version_id, ordinal),
                ).fetchone()
                expected = item.canonical_value()
                if child is None or any(
                    str(child[column]) != str(expected[key])
                    for column, key in (
                        ("authority_event_id", "authority_event_id"),
                        ("aggregate_type", "aggregate_type"),
                        ("aggregate_id", "aggregate_id"),
                        ("event_digest", "event_digest"),
                    )
                ) or int(child["aggregate_version"]) != int(
                    expected["aggregate_version"]
                ):
                    raise AuthorityPersistenceError(
                        "editorial relation workflow evidence differs"
                    )
        dependency_rows = conn.execute(
            "SELECT dependency_id FROM editorial_relation_resolution_dependencies "
            "WHERE proposal_version_id=? ORDER BY dependency_ordinal",
            (proposal_version_id,),
        ).fetchall()
        if tuple(str(row["dependency_id"]) for row in dependency_rows) != tuple(
            str(item) for item in request.resolution_dependency_ids
        ):
            raise AuthorityPersistenceError(
                "editorial relation resolution dependencies differ from request"
            )

    def _editorial_proposal_version_from_row(
        self, conn: sqlite3.Connection, row: sqlite3.Row, *, replayed: bool
    ) -> EditorialRelationProposalVersion:
        self._prove_current_record(row)
        event = self._editorial_record_context(
            conn, event_id=str(row["authority_event_id"])
        )
        payload = bytes(event["payload_bytes"])
        request_value = self._editorial_decode_json_blob(
            payload, identity="editorial relation proposal request"
        )
        request = decode_editorial_relation_proposal_request(
            request_value, idempotency_key=str(event["idempotency_key"])
        )
        self._validate_editorial_relation_record_envelope(
            conn,
            row,
            command_type="editorial.relation.proposal.record",
            aggregate_id=str(request.proposal_version_id),
            payload_bytes=payload,
            payload_digest=request.canonical_digest,
        )
        if (
            str(row["proposal_version_id"]) != str(request.proposal_version_id)
            or str(row["proposal_id"]) != str(request.proposal_id)
            or int(row["version_number"]) != request.version_number
            or (
                None
                if row["previous_proposal_version_id"] is None
                else str(row["previous_proposal_version_id"])
            )
            != (
                None
                if request.expected_previous_version_id is None
                else str(request.expected_previous_version_id)
            )
            or str(row["request_digest"]) != request.canonical_digest
            or bytes(row["request_bytes"]) != request.canonical_bytes
            or bytes(row["canonical_bytes"]) != request.canonical_bytes
            or str(row["canonical_digest"]) != request.canonical_digest
        ):
            raise AuthorityPersistenceError(
                "editorial relation proposal version differs from request"
            )
        self._validate_editorial_proposal_children(
            conn,
            proposal_version_id=str(request.proposal_version_id),
            request=request,
        )
        return EditorialRelationProposalVersion(
            proposal_version_id=request.proposal_version_id,
            proposal_id=request.proposal_id,
            version_number=request.version_number,
            previous_proposal_version_id=request.expected_previous_version_id,
            temporal_scope=request.temporal_scope,
            evidence=request.evidence,
            resolution_dependency_ids=request.resolution_dependency_ids,
            statement=request.statement,
            confidence_basis_points=request.confidence_basis_points,
            uncertainty_codes=request.uncertainty_codes,
            basis_codes=request.basis_codes,
            request_digest=request.canonical_digest,
            canonical_digest=request.canonical_digest,
            authority_event_id=EventId.parse(str(row["authority_event_id"])),
            authority_ledger_seq=int(row["authority_ledger_seq"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            replayed=replayed,
        )

    def _editorial_decision_from_row(
        self, conn: sqlite3.Connection, row: sqlite3.Row, *, replayed: bool
    ) -> EditorialRelationDecision:
        self._prove_current_record(row)
        event = self._editorial_record_context(
            conn, event_id=str(row["authority_event_id"])
        )
        payload = bytes(event["payload_bytes"])
        value = self._editorial_decode_json_blob(
            payload, identity="editorial relation decision request"
        )
        request = decode_editorial_relation_decision_request(
            value, idempotency_key=str(event["idempotency_key"])
        )
        self._validate_editorial_relation_record_envelope(
            conn,
            row,
            command_type="editorial.relation.decision.record",
            aggregate_id=str(request.decision_id),
            payload_bytes=payload,
            payload_digest=request.canonical_digest,
        )
        expected = {
            "decision_id": str(request.decision_id),
            "proposal_id": str(request.proposal_id),
            "proposal_version_id": str(request.proposal_version_id),
            "proposal_version_digest": request.expected_proposal_version_digest,
            "action": request.action.value,
            "reason_code": request.reason_code,
            "decision_policy_version": request.decision_policy_version,
        }
        if any(str(row[key]) != value for key, value in expected.items()):
            raise AuthorityPersistenceError(
                "editorial relation decision columns differ from request"
            )
        optional_pairs = (
            ("previous_decision_id", request.expected_previous_decision_id),
            ("assertion_id", request.assertion_id),
            ("target_assertion_id", request.target_assertion_id),
            ("successor_assertion_id", request.successor_assertion_id),
            ("supersession_id", request.supersession_id),
        )
        if any(
            (None if row[column] is None else str(row[column]))
            != (None if expected_id is None else str(expected_id))
            for column, expected_id in optional_pairs
        ) or int(row["decision_version"]) != (
            request.expected_previous_decision_version + 1
        ):
            raise AuthorityPersistenceError(
                "editorial relation decision sequence differs from request"
            )
        if (
            bytes(row["canonical_bytes"]) != request.canonical_bytes
            or str(row["canonical_digest"]) != request.canonical_digest
        ):
            raise AuthorityPersistenceError(
                "editorial relation decision canonical bytes differ"
            )
        return EditorialRelationDecision(
            decision_id=request.decision_id,
            action=request.action,
            proposal_id=request.proposal_id,
            proposal_version_id=request.proposal_version_id,
            proposal_version_digest=request.expected_proposal_version_digest,
            decision_version=request.expected_previous_decision_version + 1,
            previous_decision_id=request.expected_previous_decision_id,
            assertion_id=request.assertion_id,
            target_assertion_id=request.target_assertion_id,
            successor_assertion_id=request.successor_assertion_id,
            supersession_id=request.supersession_id,
            reason_code=request.reason_code,
            decision_policy_version=request.decision_policy_version,
            authority_event_id=EventId.parse(str(row["authority_event_id"])),
            authority_ledger_seq=int(row["authority_ledger_seq"]),
            canonical_digest=request.canonical_digest,
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            replayed=replayed,
        )

    def _editorial_assertion_from_row(
        self, conn: sqlite3.Connection, row: sqlite3.Row
    ) -> EditorialRelationAssertion:
        proposal_row = self._editorial_required_row(
            conn,
            table="editorial_relation_proposals",
            column="proposal_id",
            identifier=str(row["proposal_id"]),
            identity="editorial relation proposal",
        )
        proposal = self._editorial_proposal_from_row(conn, proposal_row)
        version_row = self._editorial_required_row(
            conn,
            table="editorial_relation_proposal_versions",
            column="proposal_version_id",
            identifier=str(row["proposal_version_id"]),
            identity="editorial relation proposal version",
        )
        version = self._editorial_proposal_version_from_row(
            conn, version_row, replayed=False
        )
        result = EditorialRelationAssertion(
            assertion_id=EditorialRelationAssertionId.parse(str(row["assertion_id"])),
            proposal_id=proposal.proposal_id,
            proposal_version_id=version.proposal_version_id,
            predicate_registry_digest=proposal.predicate_registry_digest,
            predicate_contract_digest=proposal.predicate_contract_digest,
            predicate=proposal.predicate,
            subject=proposal.subject,
            object=proposal.object,
            temporal_scope=version.temporal_scope,
            evidence=version.evidence,
            resolution_dependency_ids=version.resolution_dependency_ids,
            producer=proposal.producer,
            statement=version.statement,
            uncertainty_codes=version.uncertainty_codes,
            trust_scope=TrustScope.ADMITTED,
            admission_decision_id=EditorialRelationDecisionId.parse(
                str(row["admission_decision_id"])
            ),
            admitted_at=UtcTimestamp.parse(str(row["admitted_at"])),
            canonical_digest=str(row["canonical_digest"]),
        )
        value = {
            "assertion_id": str(result.assertion_id),
            "proposal_id": str(result.proposal_id),
            "proposal_version_id": str(result.proposal_version_id),
            "predicate_registry_digest": result.predicate_registry_digest,
            "predicate_contract_digest": result.predicate_contract_digest,
            "predicate": result.predicate.value,
            "subject": endpoint_canonical_value(result.subject),
            "object": endpoint_canonical_value(result.object),
            "temporal_scope": result.temporal_scope.canonical_value(),
            "evidence": [evidence_canonical_value(item) for item in result.evidence],
            "resolution_dependency_ids": [
                str(item) for item in result.resolution_dependency_ids
            ],
            "producer": result.producer.canonical_value(),
            "statement": result.statement,
            "uncertainty_codes": list(result.uncertainty_codes),
            "trust_scope": result.trust_scope.value,
            "admission_decision_id": str(result.admission_decision_id),
            "admitted_at": result.admitted_at.to_text(),
        }
        data = canonical_json_bytes(value)
        if (
            str(row["relation_key"]) != result.relation_key
            or str(row["proposal_version_digest"]) != version.canonical_digest
            or bytes(row["canonical_bytes"]) != data
            or str(row["canonical_digest"]) != digest_bytes(data)
        ):
            raise AuthorityPersistenceError(
                "editorial relation assertion differs from canonical authority"
            )
        return result

    def _editorial_projection_event_from_row(
        self, conn: sqlite3.Connection, row: sqlite3.Row
    ) -> EditorialRelationProjectionEvent:
        action = EditorialRelationProjectionAction(str(row["action"]))
        assertion = None
        if action is EditorialRelationProjectionAction.UPSERT:
            assertion_row = self._editorial_required_row(
                conn,
                table="editorial_relation_assertions",
                column="assertion_id",
                identifier=str(row["assertion_id"]),
                identity="editorial relation assertion",
            )
            assertion = self._editorial_assertion_from_row(conn, assertion_row)
        value = {
            "projection_event_id": str(row["projection_event_id"]),
            "source_event_id": str(row["source_event_id"]),
            "source_ledger_seq": int(row["source_ledger_seq"]),
            "action": action.value,
            "assertion_id": str(row["assertion_id"]),
            "assertion_digest": None if assertion is None else assertion.canonical_digest,
            "lifecycle": str(row["lifecycle"]),
            "recorded_at": str(row["recorded_at"]),
        }
        data = canonical_json_bytes(value)
        if (
            bytes(row["canonical_bytes"]) != data
            or str(row["canonical_digest"]) != digest_bytes(data)
        ):
            raise AuthorityPersistenceError(
                "editorial relation projection event differs from canonical authority"
            )
        return EditorialRelationProjectionEvent(
            projection_event_id=EventId.parse(str(row["projection_event_id"])),
            source_event_id=EventId.parse(str(row["source_event_id"])),
            source_ledger_seq=int(row["source_ledger_seq"]),
            action=action,
            assertion_id=EditorialRelationAssertionId.parse(str(row["assertion_id"])),
            assertion=assertion,
            lifecycle=EditorialRelationAssertionLifecycle(str(row["lifecycle"])),
            canonical_digest=str(row["canonical_digest"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
        )

    def editorial_proposal_current(
        self, proposal_id: EditorialRelationProposalId
    ) -> EditorialRelationProposalVersion:
        with self._lock:
            row = self._connection.execute(
                "SELECT v.* FROM editorial_relation_proposal_heads h "
                "JOIN editorial_relation_proposal_versions v "
                "ON v.proposal_version_id=h.current_proposal_version_id "
                "WHERE h.proposal_id=?",
                (str(proposal_id),),
            ).fetchone()
            if row is None:
                raise EditorialRelationStateError("relation proposal is not retained")
            result = self._editorial_proposal_version_from_row(
                self._connection, row, replayed=False
            )
            self._require_editorial_proposal_version_current(
                self._connection,
                result.proposal_version_id,
                require_dependencies_accepted=False,
            )
            return result

    def editorial_proposal_version(
        self, proposal_version_id: EditorialRelationProposalVersionId
    ) -> EditorialRelationProposalVersion:
        with self._lock:
            row = self._editorial_required_row(
                self._connection,
                table="editorial_relation_proposal_versions",
                column="proposal_version_id",
                identifier=str(proposal_version_id),
                identity="relation proposal version",
            )
            result = self._editorial_proposal_version_from_row(
                self._connection, row, replayed=False
            )
            self._validate_editorial_evidence_current(self._connection, result.evidence)
            return result

    def editorial_decision_current(
        self, proposal_id: EditorialRelationProposalId
    ) -> EditorialRelationDecision | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT d.* FROM editorial_relation_decision_heads h "
                "JOIN editorial_relation_decisions d "
                "ON d.decision_id=h.current_decision_id WHERE h.proposal_id=?",
                (str(proposal_id),),
            ).fetchone()
            if row is None:
                return None
            result = self._editorial_decision_from_row(
                self._connection, row, replayed=False
            )
            self._require_editorial_proposal_version_current(
                self._connection,
                result.proposal_version_id,
                require_dependencies_accepted=result.current_state
                is EditorialRelationCurrentState.ADMITTED,
            )
            return result

    def editorial_assertion(
        self, assertion_id: EditorialRelationAssertionId
    ) -> EditorialRelationAssertion:
        with self._lock:
            return self._require_editorial_assertion_current(
                self._connection, assertion_id
            )

    def editorial_current(
        self, assertion_id: EditorialRelationAssertionId
    ) -> EditorialRelationCurrentView:
        with self._lock:
            assertion = self._require_editorial_assertion_current(
                self._connection, assertion_id
            )
            head = self._editorial_assertion_head_row(
                self._connection, assertion_id
            )
            assert head is not None
            return EditorialRelationCurrentView(
                assertion=assertion,
                lifecycle=EditorialRelationAssertionLifecycle(
                    str(head["lifecycle"])
                ),
                current_decision_id=EditorialRelationDecisionId.parse(
                    str(head["current_decision_id"])
                ),
                current_decision_version=int(head["current_decision_version"]),
                updated_at=UtcTimestamp.parse(str(head["updated_at"])),
            )

    def editorial_current_relations(
        self, *, limit: int
    ) -> tuple[EditorialRelationCurrentView, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT assertion_id FROM editorial_current_admitted_relations "
                "ORDER BY admitted_at,assertion_id LIMIT ?",
                (limit,),
            ).fetchall()
            return tuple(
                self.editorial_current(
                    EditorialRelationAssertionId.parse(str(row["assertion_id"]))
                )
                for row in rows
            )

    def editorial_projection_events_after(
        self, *, after_ledger_seq: int, limit: int
    ) -> tuple[EditorialRelationProjectionEvent, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM editorial_relation_projection_events "
                "WHERE source_ledger_seq>? ORDER BY source_ledger_seq,assertion_id "
                "LIMIT ?",
                (after_ledger_seq, limit),
            ).fetchall()
            results = tuple(
                self._editorial_projection_event_from_row(self._connection, row)
                for row in rows
            )
            for result in results:
                if result.assertion is not None:
                    self._require_editorial_assertion_rights_current(
                        self._connection, result.assertion_id
                    )
            return results


__all__ = ["_EditorialRelationReadMixin"]
