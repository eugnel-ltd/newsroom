from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import Any

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.authority.types import EventId, UtcTimestamp
from newsroom.extraction.models import (
    ExtractionInputBinding,
    ExtractionOutputView,
    ExtractionRawOutput,
    ExtractionRunMetadata,
    ExtractionRunVersion,
    ExtractorContract,
    ProposalDraft,
    ProposalEnvelope,
    ProposalSet,
)
from newsroom.extraction.policy import (
    EXTRACTION_RUN_EXECUTE_COMMAND,
    EXTRACTOR_CONTRACT_REGISTER_COMMAND,
)
from newsroom.extraction.types import (
    EvidenceRange,
    ExtractionFailureCode,
    ExtractionOutcome,
    ExtractionOutputId,
    ExtractionOutputValidation,
    ExtractionProposalKind,
    ExtractionRunId,
    ExtractionRunVersionId,
    ExtractionUsage,
    ExtractorContractId,
    ProposalEnvelopeId,
    ProposalPredicateHint,
    ProposalSetId,
)

from ._extraction_decoding import (
    decode_extraction_run,
    decode_extractor_contract,
)


class _ExtractionReadMixin:
    @staticmethod
    def _row_for_event(
        conn: sqlite3.Connection,
        *,
        table: str,
        event_id: str,
        identity: str,
    ) -> sqlite3.Row:
        row = conn.execute(
            f"SELECT * FROM {table} WHERE authority_event_id=?", (event_id,)
        ).fetchone()
        if row is None:
            raise AuthorityPersistenceError(f"committed {identity} row is missing")
        return row

    @staticmethod
    def _required_row(
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
            raise KeyError(identifier)
        return row

    @staticmethod
    def _require_columns(
        row: Mapping[str, Any], expected: Mapping[str, object], *, identity: str
    ) -> None:
        for column, value in expected.items():
            if row[column] != value:
                raise AuthorityPersistenceError(
                    f"{identity} column {column} differs from canonical authority"
                )

    def _contract_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> ExtractorContract:
        self._prove_current_record(row)
        value = self._canonical_row_value(row, identity="extractor contract")
        event = self._validate_record_envelope(
            conn,
            row,
            command_type=EXTRACTOR_CONTRACT_REGISTER_COMMAND,
            aggregate_id=str(row["contract_id"]),
            canonical_bytes=bytes(row["canonical_bytes"]),
            canonical_digest=str(row["canonical_digest"]),
        )
        request = decode_extractor_contract(
            value, idempotency_key=str(event["idempotency_key"])
        )
        if (
            str(request.contract_id) != str(row["contract_id"])
            or request.semantic_digest != str(row["semantic_digest"])
            or not self._contract_components_match(row, request)
            or request.execution_profile.value != str(row["execution_profile"])
            or request.producer_kind != str(row["producer_kind"])
        ):
            raise AuthorityPersistenceError(
                "extractor contract normalized columns differ"
            )
        return ExtractorContract(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    def _contract_for_event(
        self, conn: sqlite3.Connection, event_id: str, *, replayed: bool
    ) -> ExtractorContract:
        row = self._row_for_event(
            conn,
            table="extractor_contracts",
            event_id=event_id,
            identity="extractor contract",
        )
        return self._contract_from_row(conn, row, replayed=replayed)

    def contract(self, contract_id: ExtractorContractId) -> ExtractorContract:
        if not isinstance(contract_id, ExtractorContractId):
            raise TypeError("extractor contract identity must be typed")
        with self._lock:
            row = self._required_row(
                self._connection,
                table="extractor_contracts",
                column="contract_id",
                identifier=str(contract_id),
                identity="extractor contract",
            )
            return self._contract_from_row(
                self._connection, row, replayed=False
            )

    def _request_for_version_row(
        self, conn: sqlite3.Connection, row: sqlite3.Row
    ):
        request_bytes = bytes(row["request_bytes"])
        if digest_bytes(request_bytes) != str(row["request_digest"]):
            raise AuthorityPersistenceError("run request digest mismatch")
        value = self._decode_json_blob(request_bytes, identity="run request")
        if not isinstance(value, dict):
            raise AuthorityPersistenceError("run request must be an object")
        event = self._validate_record_envelope(
            conn,
            row,
            command_type=EXTRACTION_RUN_EXECUTE_COMMAND,
            aggregate_id=str(row["run_version_id"]),
            canonical_bytes=request_bytes,
            canonical_digest=str(row["request_digest"]),
        )
        request = decode_extraction_run(
            value, idempotency_key=str(event["idempotency_key"])
        )
        stable = self._run_row(conn, str(request.run_id))
        passages = conn.execute(
            "SELECT * FROM extraction_run_passages WHERE run_id=? "
            "ORDER BY passage_id",
            (str(request.run_id),),
        ).fetchall()
        if len(passages) != len(request.input_binding.passages):
            raise AuthorityPersistenceError("run passage count differs")
        for retained, passage in zip(
            passages, request.input_binding.passages, strict=True
        ):
            passage_bytes = canonical_json_bytes(passage.canonical_value())
            self._require_columns(
                retained,
                {
                    "run_id": str(request.run_id),
                    "passage_id": str(passage.passage_id),
                    "admission_id": str(passage.admission_id),
                    "access_decision_id": str(passage.access_decision_id),
                    "hydration_policy_contract_digest": (
                        passage.hydration_policy_contract_digest
                    ),
                    "principal_id": passage.principal_id,
                    "authority_domain": passage.authority_domain,
                    "purpose": passage.purpose,
                    "object_class": passage.object_class,
                    "allowed_use": passage.allowed_use,
                    "security_scope": passage.security_scope,
                    "retention_scope": passage.retention_scope,
                    "byte_offset": passage.byte_offset,
                    "byte_length": passage.byte_length,
                    "blob_digest": passage.blob_digest,
                    "text_digest": passage.text_digest,
                    "language": passage.language,
                },
                identity="extraction run passage",
            )
            if (
                bytes(retained["canonical_bytes"]) != passage_bytes
                or str(retained["canonical_digest"]) != digest_bytes(passage_bytes)
            ):
                raise AuthorityPersistenceError(
                    "retained run passage differs from request"
                )
        stable_value = self._stable_run_value(request)
        stable_bytes = canonical_json_bytes(stable_value)
        self._require_columns(
            stable,
            {
                "contract_id": str(request.contract_id),
                "definition_id": str(request.input_binding.definition_id),
                "definition_version_id": str(
                    request.input_binding.definition_version_id
                ),
                "item_id": str(request.input_binding.item_id),
                "revision_id": str(request.input_binding.revision_id),
                "representation_id": str(
                    request.input_binding.representation_id
                ),
                "input_binding_digest": request.input_binding.digest,
                "budget_digest": request.budget.digest,
                "stable_semantic_digest": request.stable_run_semantic_digest,
                "canonical_digest": digest_bytes(stable_bytes),
            },
            identity="extraction run",
        )
        budget_bytes = canonical_json_bytes(request.budget.canonical_value())
        if (
            bytes(stable["canonical_bytes"]) != stable_bytes
            or bytes(stable["budget_bytes"]) != budget_bytes
            or str(stable["budget_digest"]) != digest_bytes(budget_bytes)
        ):
            raise AuthorityPersistenceError("stable run canonical bytes differ")
        first = conn.execute(
            "SELECT authority_event_id,recorded_at FROM extraction_run_versions "
            "WHERE run_id=? AND version_number=1",
            (str(request.run_id),),
        ).fetchone()
        if (
            first is None
            or str(stable["created_by_event_id"])
            != str(first["authority_event_id"])
            or str(stable["created_at"]) != str(first["recorded_at"])
        ):
            raise AuthorityPersistenceError(
                "stable run creation lineage differs from first version"
            )
        return request

    def _output_for_version(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        schema_contract_digest: str,
    ) -> ExtractionOutputView | None:
        output = conn.execute(
            "SELECT * FROM extraction_outputs WHERE run_version_id=?",
            (str(row["run_version_id"]),),
        ).fetchone()
        outcome = ExtractionOutcome(str(row["outcome"]))
        if output is None:
            if int(row["output_bytes"]) != 0 or outcome.may_retain_output:
                raise AuthorityPersistenceError(
                    "run outcome or usage requires retained output"
                )
            return None
        data = bytes(output["canonical_bytes"])
        validation = ExtractionOutputValidation(str(output["validation_state"]))
        expected_validation = (
            ExtractionOutputValidation.INVALID
            if outcome is ExtractionOutcome.INVALID_OUTPUT
            else ExtractionOutputValidation.VALID
        )
        if (
            len(data) != int(output["byte_length"])
            or digest_bytes(data) != str(output["canonical_digest"])
            or canonical_json_bytes(
                self._decode_json_blob(data, identity="structured output")
            )
            != data
            or str(output["run_id"]) != str(row["run_id"])
            or str(output["run_version_id"]) != str(row["run_version_id"])
            or str(output["schema_contract_digest"])
            != schema_contract_digest
            or validation is not expected_validation
            or str(output["retained_at"]) != str(row["recorded_at"])
        ):
            raise AuthorityPersistenceError(
                "retained structured output lineage is inconsistent"
            )
        if int(row["output_bytes"]) != len(data):
            raise AuthorityPersistenceError("run output usage differs")
        return ExtractionOutputView(
            output_id=ExtractionOutputId.parse(str(output["output_id"])),
            run_id=ExtractionRunId.parse(str(output["run_id"])),
            run_version_id=ExtractionRunVersionId.parse(
                str(output["run_version_id"])
            ),
            validation=validation,
            schema_contract_digest=str(output["schema_contract_digest"]),
            byte_length=int(output["byte_length"]),
            canonical_digest=str(output["canonical_digest"]),
            retained_at=UtcTimestamp.parse(str(output["retained_at"])),
        )

    def _evidence_for_proposal(
        self, conn: sqlite3.Connection, proposal_row: sqlite3.Row
    ) -> tuple[EvidenceRange, ...]:
        rows = conn.execute(
            "SELECT e.*,p.byte_length AS passage_byte_length "
            "FROM extraction_proposal_evidence e "
            "JOIN extraction_run_passages p "
            "ON p.run_id=e.run_id AND p.passage_id=e.passage_id "
            "WHERE e.proposal_id=? ORDER BY e.evidence_ordinal",
            (str(proposal_row["proposal_id"]),),
        ).fetchall()
        evidence: list[EvidenceRange] = []
        for ordinal, row in enumerate(rows, start=1):
            value = {
                "proposal_id": str(proposal_row["proposal_id"]),
                "evidence_ordinal": ordinal,
                "run_id": str(row["run_id"]),
                "passage_id": str(row["passage_id"]),
                "start_byte": int(row["start_byte"]),
                "end_byte": int(row["end_byte"]),
                "evidence_text_digest": str(row["evidence_text_digest"]),
            }
            data = canonical_json_bytes(value)
            if (
                int(row["evidence_ordinal"]) != ordinal
                or bytes(row["canonical_bytes"]) != data
                or str(row["canonical_digest"]) != digest_bytes(data)
                or str(row["run_id"]) != str(proposal_row["run_id"])
                or int(row["end_byte"]) > int(row["passage_byte_length"])
            ):
                raise AuthorityPersistenceError(
                    "proposal evidence normalized columns differ"
                )
            evidence.append(
                EvidenceRange(
                    passage_id=self._passage_id(str(row["passage_id"])),
                    start_byte=int(row["start_byte"]),
                    end_byte=int(row["end_byte"]),
                    evidence_text_digest=str(row["evidence_text_digest"]),
                )
            )
        if not evidence:
            raise AuthorityPersistenceError("retained proposal has no evidence")
        return tuple(evidence)

    @staticmethod
    def _passage_id(value: str):
        from newsroom.extraction.types import ExtractionPassageId

        return ExtractionPassageId.parse(value)

    def _proposal_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        expected_proposal_set_id: str,
        expected_output_id: str,
        expected_run_id: str,
        expected_run_version_id: str,
        expected_contract_digest: str,
        expected_retained_at: str,
    ) -> ProposalEnvelope:
        evidence = self._evidence_for_proposal(conn, row)
        uncertainty_bytes = bytes(row["uncertainty_codes_bytes"])
        rationale_bytes = bytes(row["rationale_codes_bytes"])
        uncertainties = self._decode_json_blob(
            uncertainty_bytes, identity="uncertainty codes"
        )
        rationales = self._decode_json_blob(
            rationale_bytes, identity="rationale codes"
        )
        if not isinstance(uncertainties, list) or not isinstance(rationales, list):
            raise AuthorityPersistenceError("proposal code lists are invalid")
        draft = ProposalDraft(
            local_id=str(row["local_id"]),
            kind=ExtractionProposalKind(str(row["proposal_kind"])),
            subject_placeholder=str(row["subject_placeholder"]),
            object_placeholder=(
                None
                if row["object_placeholder"] is None
                else str(row["object_placeholder"])
            ),
            predicate_hint=(
                None
                if row["predicate_hint"] is None
                else ProposalPredicateHint(str(row["predicate_hint"]))
            ),
            confidence_basis_points=(
                None
                if row["confidence_basis_points"] is None
                else int(row["confidence_basis_points"])
            ),
            uncertainty_codes=tuple(str(item) for item in uncertainties),
            rationale_codes=tuple(str(item) for item in rationales),
            evidence=evidence,
        )
        value = {
            "proposal_id": str(row["proposal_id"]),
            "proposal_set_id": str(row["proposal_set_id"]),
            "output_id": str(row["output_id"]),
            "run_id": str(row["run_id"]),
            "run_version_id": str(row["run_version_id"]),
            "draft": draft.canonical_value(),
            "producer_contract_digest": str(row["producer_contract_digest"]),
        }
        data = canonical_json_bytes(value)
        if (
            str(row["proposal_set_id"]) != expected_proposal_set_id
            or str(row["output_id"]) != expected_output_id
            or str(row["run_id"]) != expected_run_id
            or str(row["run_version_id"]) != expected_run_version_id
            or str(row["producer_contract_digest"])
            != expected_contract_digest
            or str(row["retained_at"]) != expected_retained_at
            or uncertainty_bytes
            != canonical_json_bytes(list(draft.uncertainty_codes))
            or rationale_bytes != canonical_json_bytes(list(draft.rationale_codes))
            or bytes(row["canonical_bytes"]) != data
            or str(row["canonical_digest"]) != digest_bytes(data)
            or str(row["semantic_digest"]) != draft.digest
        ):
            raise AuthorityPersistenceError(
                "proposal normalized columns differ from canonical envelope"
            )
        return ProposalEnvelope(
            proposal_id=ProposalEnvelopeId.parse(str(row["proposal_id"])),
            proposal_set_id=ProposalSetId.parse(str(row["proposal_set_id"])),
            output_id=ExtractionOutputId.parse(str(row["output_id"])),
            run_id=ExtractionRunId.parse(str(row["run_id"])),
            run_version_id=ExtractionRunVersionId.parse(
                str(row["run_version_id"])
            ),
            local_id=draft.local_id,
            kind=draft.kind,
            subject_placeholder=draft.subject_placeholder,
            object_placeholder=draft.object_placeholder,
            predicate_hint=draft.predicate_hint,
            confidence_basis_points=draft.confidence_basis_points,
            uncertainty_codes=draft.uncertainty_codes,
            rationale_codes=draft.rationale_codes,
            evidence=draft.evidence,
            producer_contract_digest=str(row["producer_contract_digest"]),
            canonical_digest=str(row["canonical_digest"]),
            retained_at=UtcTimestamp.parse(str(row["retained_at"])),
        )

    def _proposal_set_for_version(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        output: ExtractionOutputView | None,
        *,
        producer_contract_digest: str,
    ) -> ProposalSet | None:
        retained = conn.execute(
            "SELECT * FROM extraction_proposal_sets WHERE run_version_id=?",
            (str(row["run_version_id"]),),
        ).fetchone()
        if retained is None:
            if int(row["proposal_count"]) != 0:
                raise AuthorityPersistenceError(
                    "run reports proposals without a proposal set"
                )
            return None
        if output is None:
            raise AuthorityPersistenceError("proposal set exists without output")
        expected_set_id = str(retained["proposal_set_id"])
        expected_output_id = str(output.output_id)
        expected_run_id = str(row["run_id"])
        expected_run_version_id = str(row["run_version_id"])
        expected_retained_at = str(row["recorded_at"])
        proposal_rows = conn.execute(
            "SELECT * FROM extraction_proposals WHERE proposal_set_id=? "
            "ORDER BY local_id",
            (expected_set_id,),
        ).fetchall()
        proposals = tuple(
            self._proposal_from_row(
                conn,
                proposal_row,
                expected_proposal_set_id=expected_set_id,
                expected_output_id=expected_output_id,
                expected_run_id=expected_run_id,
                expected_run_version_id=expected_run_version_id,
                expected_contract_digest=producer_contract_digest,
                expected_retained_at=expected_retained_at,
            )
            for proposal_row in proposal_rows
        )
        if (
            len(proposals) != int(retained["proposal_count"])
            or len(proposals) != int(row["proposal_count"])
        ):
            raise AuthorityPersistenceError("proposal-set count differs")
        value = {
            "proposal_set_id": expected_set_id,
            "output_id": str(retained["output_id"]),
            "run_id": str(retained["run_id"]),
            "run_version_id": str(retained["run_version_id"]),
            "producer_contract_digest": str(
                retained["producer_contract_digest"]
            ),
            "proposal_digests": [item.canonical_digest for item in proposals],
        }
        data = canonical_json_bytes(value)
        if (
            str(retained["output_id"]) != expected_output_id
            or str(retained["run_id"]) != expected_run_id
            or str(retained["run_version_id"]) != expected_run_version_id
            or str(retained["producer_contract_digest"])
            != producer_contract_digest
            or str(retained["retained_at"]) != expected_retained_at
            or output.retained_at.to_text() != expected_retained_at
            or bytes(retained["canonical_bytes"]) != data
            or str(retained["canonical_digest"]) != digest_bytes(data)
        ):
            raise AuthorityPersistenceError("proposal-set canonical record differs")
        return ProposalSet(
            proposal_set_id=ProposalSetId.parse(expected_set_id),
            output_id=output.output_id,
            run_id=ExtractionRunId.parse(expected_run_id),
            run_version_id=ExtractionRunVersionId.parse(
                expected_run_version_id
            ),
            proposals=proposals,
            producer_contract_digest=producer_contract_digest,
            canonical_digest=str(retained["canonical_digest"]),
            retained_at=UtcTimestamp.parse(str(retained["retained_at"])),
        )

    def _run_version_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> ExtractionRunVersion:
        self._prove_current_record(row)
        request = self._request_for_version_row(conn, row)
        contract = self._contract_row(conn, str(request.contract_id))
        contract_digest = str(contract["canonical_digest"])
        if str(row["contract_canonical_digest"]) != contract_digest:
            raise AuthorityPersistenceError(
                "run-version extractor contract digest differs"
            )
        usage = ExtractionUsage(
            elapsed_ms=int(row["elapsed_ms"]),
            input_bytes=int(row["input_bytes"]),
            output_bytes=int(row["output_bytes"]),
            proposal_count=int(row["proposal_count"]),
            evidence_range_count=int(row["evidence_range_count"]),
            request_tokens=int(row["request_tokens"]),
            response_tokens=int(row["response_tokens"]),
            cost_microunits=int(row["cost_microunits"]),
        )
        value = self._run_version_value(
            request=request,
            contract_digest=str(row["contract_canonical_digest"]),
            outcome=str(row["outcome"]),
            failure_code=str(row["failure_code"]),
            started_at=str(row["started_at"]),
            ended_at=str(row["ended_at"]),
            usage_value=usage.canonical_value(),
        )
        data = canonical_json_bytes(value)
        if (
            bytes(row["canonical_bytes"]) != data
            or str(row["canonical_digest"]) != digest_bytes(data)
            or str(request.run_version_id) != str(row["run_version_id"])
            or str(request.run_id) != str(row["run_id"])
            or request.version_number != int(row["version_number"])
            or (
                None
                if request.expected_previous_version_id is None
                else str(request.expected_previous_version_id)
            )
            != row["previous_run_version_id"]
        ):
            raise AuthorityPersistenceError(
                "run-version normalized columns differ from canonical authority"
            )
        output = self._output_for_version(
            conn,
            row,
            schema_contract_digest=str(contract["output_schema_digest"]),
        )
        proposal_set = self._proposal_set_for_version(
            conn,
            row,
            output,
            producer_contract_digest=contract_digest,
        )
        return ExtractionRunVersion(
            request=request,
            contract_canonical_digest=contract_digest,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            started_at=UtcTimestamp.parse(str(row["started_at"])),
            ended_at=UtcTimestamp.parse(str(row["ended_at"])),
            outcome=ExtractionOutcome(str(row["outcome"])),
            failure_code=ExtractionFailureCode(str(row["failure_code"])),
            usage=usage,
            output=output,
            proposal_set=proposal_set,
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    def _run_version_for_event(
        self, conn: sqlite3.Connection, event_id: str, *, replayed: bool
    ) -> ExtractionRunVersion:
        row = self._row_for_event(
            conn,
            table="extraction_run_versions",
            event_id=event_id,
            identity="extraction run version",
        )
        return self._run_version_from_row(conn, row, replayed=replayed)

    def _revalidate_result_current(
        self, conn: sqlite3.Connection, result: ExtractionRunVersion
    ) -> None:
        self._revalidate_input_binding_current(conn, result.request.input_binding)

    def _revalidate_input_binding_current(
        self, conn: sqlite3.Connection, binding: ExtractionInputBinding,
    ) -> None:
        self._require_source_binding_current(conn, binding)
        now = self._clock()
        for passage in binding.passages:
            self._current_passage_authority_row(
                conn,
                passage,
                now=now,
                principal_id=passage.principal_id,
            )

    def run_version(
        self, run_version_id: ExtractionRunVersionId
    ) -> ExtractionRunVersion:
        if not isinstance(run_version_id, ExtractionRunVersionId):
            raise TypeError("run version identity must be typed")
        with self._lock:
            row = self._required_row(
                self._connection,
                table="extraction_run_versions",
                column="run_version_id",
                identifier=str(run_version_id),
                identity="extraction run version",
            )
            result = self._run_version_from_row(
                self._connection, row, replayed=False
            )
            self._revalidate_result_current(self._connection, result)
            return result

    def run_history(
        self, run_id: ExtractionRunId, *, limit: int
    ) -> tuple[ExtractionRunMetadata, ...]:
        if not isinstance(run_id, ExtractionRunId):
            raise TypeError("run identity must be typed")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM extraction_run_versions WHERE run_id=? "
                "ORDER BY version_number DESC LIMIT ?",
                (str(run_id), limit),
            ).fetchall()
            results = tuple(
                self._run_version_from_row(
                    self._connection, row, replayed=False
                )
                for row in rows
            )
            for result in results:
                self._revalidate_result_current(self._connection, result)
            return tuple(self._metadata(result) for result in results)

    @staticmethod
    def _metadata(result: ExtractionRunVersion) -> ExtractionRunMetadata:
        return ExtractionRunMetadata(
            run_id=result.request.run_id,
            run_version_id=result.request.run_version_id,
            version_number=result.request.version_number,
            contract_id=result.request.contract_id,
            input_binding_digest=result.request.input_binding.digest,
            outcome=result.outcome,
            failure_code=result.failure_code,
            started_at=result.started_at,
            ended_at=result.ended_at,
            recorded_at=result.recorded_at,
            usage=result.usage,
            output=result.output,
            proposal_count=(
                0
                if result.proposal_set is None
                else len(result.proposal_set.proposals)
            ),
            terminal=result.outcome.terminal,
        )

    def metadata(
        self, run_version_id: ExtractionRunVersionId
    ) -> ExtractionRunMetadata:
        return self._metadata(self.run_version(run_version_id))

    def proposals(
        self, run_version_id: ExtractionRunVersionId
    ) -> tuple[ProposalEnvelope, ...]:
        result = self.run_version(run_version_id)
        return () if result.proposal_set is None else result.proposal_set.proposals

    def raw_output(self, output_id: ExtractionOutputId) -> ExtractionRawOutput:
        if not isinstance(output_id, ExtractionOutputId):
            raise TypeError("output identity must be typed")
        with self._lock:
            row = self._required_row(
                self._connection,
                table="extraction_outputs",
                column="output_id",
                identifier=str(output_id),
                identity="extraction output",
            )
            version_row = self._required_row(
                self._connection,
                table="extraction_run_versions",
                column="run_version_id",
                identifier=str(row["run_version_id"]),
                identity="extraction run version",
            )
            result = self._run_version_from_row(
                self._connection, version_row, replayed=False
            )
            self._revalidate_result_current(self._connection, result)
            assert result.output is not None
            return ExtractionRawOutput(
                view=result.output,
                canonical_bytes=bytes(row["canonical_bytes"]),
            )


__all__ = ["_ExtractionReadMixin"]
