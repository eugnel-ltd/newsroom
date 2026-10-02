from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from newsroom.authority._capability import _AuthorizedCommandGrant
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.authority.objects import ObjectAccessDecisionId
from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.authority.types import (
    EventId,
    ObjectAdmissionId,
    PayloadMode,
    TrustScope,
    UtcTimestamp,
)
from newsroom.extraction.models import ExtractionRunVersion, ExtractionUsage
from newsroom.extraction.types import (
    ExtractionFailureCode,
    ExtractionOutcome,
    ExtractionOutputId,
    ExtractionPassageId,
    ExtractionRunId,
    ExtractionRunVersionId,
    ExtractorContractId,
    FixtureExtractionCase,
    ProposalSetId,
)
from newsroom.graphiti_adapter.contracts import (
    QUALIFICATION_WORKSPACE_POLICY,
    REPLAY_WORKSPACE_POLICY,
    qualification_configuration,
    replay_configuration,
)
from newsroom.graphiti_adapter.models import (
    GraphitiAdapterConfigurationRecord,
    GraphitiAttemptRecord,
    GraphitiAttemptRequest,
    GraphitiCleanupReceipt,
    GraphitiInputManifest,
    GraphitiManifestPassage,
    GraphitiReplayApprovalRequest,
    GraphitiReplaySource,
    GraphitiReplaySourceRecord,
    GraphitiWorkspaceDescriptor,
    GraphitiWorkspacePolicy,
    RecoveredAmbiguousProgressionProof,
)
from newsroom.graphiti_adapter.policy import (
    GRAPHITI_ATTEMPT_EXECUTE_COMMAND,
    GRAPHITI_CONFIGURATION_REGISTER_COMMAND,
    GRAPHITI_REPLAY_APPROVE_COMMAND,
)
from newsroom.graphiti_adapter.temporal_vocabulary import TemporalBasis
from newsroom.graphiti_adapter.types import (
    GraphitiAdapterConfigurationId,
    GraphitiAdapterIdentifierReuse,
    GraphitiAdapterOutcome,
    GraphitiAdapterRightsDenied,
    GraphitiAdapterSemanticCollision,
    GraphitiAdapterStateError,
    GraphitiAttemptId,
    GraphitiCleanupReason,
    GraphitiCleanupReceiptId,
    GraphitiInputManifestId,
    GraphitiReplayEligibility,
    GraphitiReplaySourceId,
    GraphitiRuntimeMode,
    GraphitiWorkspaceId,
    GraphitiWorkspacePolicyId,
    GraphitiWorkspaceState,
)
from newsroom.sources import (
    DiscoveryRepresentationId,
    SourceDefinitionId,
    SourceDefinitionVersionId,
    SourceItemId,
    SourceRevisionId,
)


_RECORD_SPECS: dict[str, tuple[str, str, TrustScope]] = {
    GRAPHITI_CONFIGURATION_REGISTER_COMMAND: (
        "graphiti_adapter_configuration",
        "graphiti.adapter.configuration.registered",
        TrustScope.ADMITTED,
    ),
    GRAPHITI_ATTEMPT_EXECUTE_COMMAND: (
        "graphiti_adapter_attempt",
        "graphiti.adapter.attempt.executed",
        TrustScope.PROPOSED,
    ),
    GRAPHITI_REPLAY_APPROVE_COMMAND: (
        "graphiti_replay_source",
        "graphiti.adapter.replay.approved",
        TrustScope.ADMITTED,
    ),
}


def graphiti_event_digest(row: Mapping[str, Any]) -> str:
    return digest_canonical(
        {
            "event_id": str(row["event_id"]),
            "event_type": str(row["event_type"]),
            "event_schema_version": int(row["event_schema_version"]),
            "aggregate_type": str(row["aggregate_type"]),
            "aggregate_id": str(row["aggregate_id"]),
            "aggregate_version": int(row["aggregate_version"]),
            "recorded_at": str(row["recorded_at"]),
            "command_definition_digest": str(row["command_definition_digest"]),
            "payload_digest": str(row["payload_digest"]),
            "principal_id": str(row["principal_id"]),
            "security_scope": str(row["security_scope"]),
            "retention_scope": str(row["retention_scope"]),
            "trust_scope": str(row["trust_scope"]),
        }
    )


class _GraphitiAdapterStoreSupport:
    _workspace_root: Path

    def _require_graphiti_grant(
        self,
        grant: _AuthorizedCommandGrant,
        *,
        command_type: str,
        aggregate_id: str,
        canonical_bytes: bytes,
    ) -> None:
        self._issuer.verify(grant)
        spec = _RECORD_SPECS.get(command_type)
        if spec is None:
            raise AuthorityPersistenceError("unknown Graphiti adapter command")
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
            or definition.security_scope != "authority.graphiti_adapter"
            or definition.retention_scope != "authority.audit"
            or definition.payload_mode is not PayloadMode.INLINE
            or grant.payload.kind != PayloadMode.INLINE.value
            or grant.payload.inline_bytes != canonical_bytes
            or grant.payload.digest != digest_bytes(canonical_bytes)
        ):
            raise AuthorityPersistenceError(
                "Graphiti adapter grant differs from the typed record"
            )

    @staticmethod
    def _graphiti_ensure_identifier_absent(
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
            raise GraphitiAdapterIdentifierReuse(
                f"{identity} is already retained under another command identity"
            )

    @staticmethod
    def _graphiti_ensure_semantic_absent(
        conn: sqlite3.Connection,
        *,
        table: str,
        column: str,
        digest: str,
        identity: str,
    ) -> None:
        if conn.execute(
            f"SELECT 1 FROM {table} WHERE {column}=?", (digest,)
        ).fetchone() is not None:
            raise GraphitiAdapterSemanticCollision(
                f"{identity} already exists under another stable identity"
            )

    @staticmethod
    def _graphiti_decode_json_blob(
        value: bytes | memoryview, *, identity: str
    ) -> Any:
        data = bytes(value)
        try:
            decoded = json.loads(data.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise AuthorityPersistenceError(
                f"{identity} retained JSON is invalid"
            ) from exc
        if canonical_json_bytes(decoded) != data:
            raise AuthorityPersistenceError(
                f"{identity} retained JSON is not canonical"
            )
        return decoded

    @classmethod
    def _graphiti_canonical_row_value(
        cls, row: Mapping[str, Any], *, identity: str
    ) -> dict[str, Any]:
        data = bytes(row["canonical_bytes"])
        if digest_bytes(data) != str(row["canonical_digest"]):
            raise AuthorityPersistenceError(f"{identity} canonical digest mismatch")
        value = cls._graphiti_decode_json_blob(data, identity=identity)
        if not isinstance(value, dict):
            raise AuthorityPersistenceError(f"{identity} must be a canonical object")
        return value

    @staticmethod
    def _graphiti_record_context(
        conn: sqlite3.Connection, *, event_id: str
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT e.*,c.idempotency_key,p.payload_bytes "
            "FROM ledger_events e "
            "JOIN authority_commands c ON c.command_id=e.command_id "
            "JOIN authority_payloads p ON p.payload_id=e.payload_id "
            "WHERE e.event_id=?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise AuthorityPersistenceError(
                "Graphiti adapter record has no exact authority event"
            )
        return row

    @classmethod
    def _validate_graphiti_record_envelope(
        cls,
        conn: sqlite3.Connection,
        row: Mapping[str, Any],
        *,
        command_type: str,
        aggregate_id: str,
        payload_bytes: bytes,
        event_id_column: str = "authority_event_id",
        recorded_at_column: str = "recorded_at",
    ) -> sqlite3.Row:
        event = cls._graphiti_record_context(
            conn, event_id=str(row[event_id_column])
        )
        aggregate_type, event_type, trust_scope = _RECORD_SPECS[command_type]
        payload_digest = digest_bytes(payload_bytes)
        if (
            str(event["event_type"]) != event_type
            or int(event["event_schema_version"]) != 1
            or str(event["aggregate_type"]) != aggregate_type
            or str(event["aggregate_id"]) != aggregate_id
            or int(event["aggregate_version"]) != 1
            or str(event["recorded_at"])
            != str(row[recorded_at_column])
            or str(event["security_scope"]) != "authority.graphiti_adapter"
            or str(event["retention_scope"]) != "authority.audit"
            or str(event["trust_scope"]) != trust_scope.value
            or str(event["payload_mode"]) != PayloadMode.INLINE.value
            or str(event["payload_digest"]) != payload_digest
            or event["payload_bytes"] is None
            or bytes(event["payload_bytes"]) != payload_bytes
        ):
            raise AuthorityPersistenceError(
                "Graphiti adapter authority envelope is inconsistent"
            )
        return event

    @staticmethod
    def _graphiti_configuration_row(
        conn: sqlite3.Connection, configuration_id: GraphitiAdapterConfigurationId | str
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM graphiti_adapter_configurations WHERE configuration_id=?",
            (str(configuration_id),),
        ).fetchone()
        if row is None:
            raise GraphitiAdapterStateError("adapter configuration is not retained")
        return row

    @staticmethod
    def _graphiti_workspace_policy_row(
        conn: sqlite3.Connection, policy_id: GraphitiWorkspacePolicyId | str
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM graphiti_workspace_policies WHERE policy_id=?",
            (str(policy_id),),
        ).fetchone()
        if row is None:
            raise GraphitiAdapterStateError("adapter workspace policy is not retained")
        return row

    @classmethod
    def _graphiti_workspace_policy_from_row(
        cls, row: Mapping[str, Any]
    ) -> GraphitiWorkspacePolicy:
        from newsroom.graphiti_adapter.evaluation_packet import (
            EVALUATION_WORKSPACE_POLICY,
        )

        value = cls._graphiti_canonical_row_value(
            row, identity="Graphiti workspace policy"
        )
        for policy in (
            QUALIFICATION_WORKSPACE_POLICY,
            REPLAY_WORKSPACE_POLICY,
            EVALUATION_WORKSPACE_POLICY,
        ):
            if value == policy.canonical_value():
                if str(row["canonical_digest"]) != policy.canonical_digest:
                    raise AuthorityPersistenceError(
                        "Graphiti workspace policy digest differs"
                    )
                return policy
        raise AuthorityPersistenceError(
            "Graphiti workspace policy is outside the authorised closed set"
        )

    def _graphiti_configuration_from_row(
        self, conn: sqlite3.Connection, row: Mapping[str, Any], *, replayed: bool
    ) -> GraphitiAdapterConfigurationRecord:
        self._prove_current_record(row)
        value = self._graphiti_canonical_row_value(
            row, identity="Graphiti adapter configuration"
        )
        event = self._graphiti_record_context(
            conn, event_id=str(row["authority_event_id"])
        )
        contract_row = self._contract_row(conn, str(row["extractor_contract_id"]))
        contract = self._contract_from_row(conn, contract_row, replayed=False).request
        mode = GraphitiRuntimeMode(str(row["runtime_mode"]))
        configuration_id = GraphitiAdapterConfigurationId.parse(
            str(row["configuration_id"])
        )
        if mode is GraphitiRuntimeMode.DETERMINISTIC_FAKE:
            configuration = qualification_configuration(
                configuration_id=configuration_id,
                contract=contract,
                fixture_case=FixtureExtractionCase(str(row["fixture_case"])),
                idempotency_key=str(event["idempotency_key"]),
            )
        elif mode is GraphitiRuntimeMode.APPROVED_REPLAY:
            configuration = replay_configuration(
                configuration_id=configuration_id,
                contract=contract,
                idempotency_key=str(event["idempotency_key"]),
            )
        else:
            from newsroom.graphiti_adapter.evaluation_attempt import (
                evaluation_attempt_for,
            )

            configuration = evaluation_attempt_for(
                ("retained Graphiti evaluation configuration",)
            ).configuration
            if (
                configuration.configuration_id != configuration_id
                or configuration.extractor_contract_id != contract.contract_id
                or configuration.extractor_contract_digest != contract.digest
                or configuration.idempotency_key != str(event["idempotency_key"])
            ):
                raise AuthorityPersistenceError(
                    "real Graphiti configuration differs from exact evaluation authority"
                )
        component_columns = {
            "framework": ("framework_id", "framework_version", "framework_digest"),
            "model": ("model_id", "model_version", "model_digest"),
            "embedding": ("embedding_id", "embedding_version", "embedding_digest"),
            "prompt": ("prompt_id", "prompt_version", "prompt_digest"),
            "output_schema": (
                "output_schema_id",
                "output_schema_version",
                "output_schema_digest",
            ),
            "code": ("code_id", "code_version", "code_digest"),
            "normalisation": (
                "normalisation_id",
                "normalisation_version",
                "normalisation_digest",
            ),
            "temporal_policy": (
                "temporal_policy_id",
                "temporal_policy_version",
                "temporal_policy_digest",
            ),
            "adapter_policy": (
                "adapter_policy_id",
                "adapter_policy_version",
                "adapter_policy_digest",
            ),
        }
        component_mismatch = any(
            (
                str(row[id_column]),
                str(row[version_column]),
                str(row[digest_column]),
            )
            != (
                component.component_id,
                component.component_version,
                component.contract_digest,
            )
            for name, (id_column, version_column, digest_column) in component_columns.items()
            for component in (getattr(configuration, name),)
        )
        expected_real_runtime_digest = (
            None
            if configuration.real_runtime_authority is None
            else configuration.real_runtime_authority.authority_decision_digest
        )
        if (
            configuration.canonical_value() != value
            or configuration.canonical_bytes != bytes(row["canonical_bytes"])
            or configuration.canonical_digest != str(row["canonical_digest"])
            or configuration.semantic_digest != str(row["semantic_digest"])
            or str(row["execution_profile"])
            != configuration.execution_profile.value
            or component_mismatch
            or str(row["extractor_contract_id"])
            != str(configuration.extractor_contract_id)
            or str(row["extractor_contract_digest"])
            != configuration.extractor_contract_digest
            or str(row["workspace_policy_id"])
            != str(configuration.workspace_policy.policy_id)
            or str(row["workspace_policy_digest"])
            != configuration.workspace_policy.canonical_digest
            or (
                None if row["fixture_case"] is None else str(row["fixture_case"])
            )
            != (
                None
                if configuration.fixture_case is None
                else configuration.fixture_case.value
            )
            or (
                None
                if row["real_runtime_authority_digest"] is None
                else str(row["real_runtime_authority_digest"])
            )
            != expected_real_runtime_digest
        ):
            raise AuthorityPersistenceError(
                "Graphiti adapter configuration canonical authority differs"
            )
        self._validate_graphiti_record_envelope(
            conn,
            row,
            command_type=GRAPHITI_CONFIGURATION_REGISTER_COMMAND,
            aggregate_id=str(configuration.configuration_id),
            payload_bytes=configuration.canonical_bytes,
        )
        return GraphitiAdapterConfigurationRecord(
            configuration=configuration,
            authority_event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=1,
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            replayed=replayed,
        )

    @staticmethod
    def _graphiti_attempt_row(
        conn: sqlite3.Connection, attempt_id: GraphitiAttemptId | str
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM graphiti_adapter_attempts WHERE attempt_id=?",
            (str(attempt_id),),
        ).fetchone()
        if row is None:
            raise GraphitiAdapterStateError("adapter attempt is not retained")
        return row

    @staticmethod
    def _graphiti_attempt_head_row(
        conn: sqlite3.Connection, run_id: ExtractionRunId | str
    ) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM graphiti_adapter_attempt_heads WHERE run_id=?",
            (str(run_id),),
        ).fetchone()

    @classmethod
    def _graphiti_workspace_from_row(
        cls, row: Mapping[str, Any]
    ) -> GraphitiWorkspaceDescriptor:
        value = cls._graphiti_canonical_row_value(row, identity="Graphiti workspace")
        descriptor = GraphitiWorkspaceDescriptor(
            workspace_id=GraphitiWorkspaceId.parse(str(row["workspace_id"])),
            configuration_id=GraphitiAdapterConfigurationId.parse(
                str(row["configuration_id"])
            ),
            policy_id=GraphitiWorkspacePolicyId.parse(str(row["policy_id"])),
            policy_digest=str(row["policy_digest"]),
            namespace=str(row["namespace"]),
            created_at=UtcTimestamp.parse(str(row["created_at"])),
        )
        if (
            descriptor.canonical_value() != value
            or descriptor.canonical_digest != str(row["canonical_digest"])
        ):
            raise AuthorityPersistenceError("Graphiti workspace canonical data differs")
        return descriptor

    @classmethod
    def _graphiti_manifest_from_row(
        cls, conn: sqlite3.Connection, row: Mapping[str, Any]
    ) -> GraphitiInputManifest:
        value = cls._graphiti_canonical_row_value(
            row, identity="Graphiti input manifest"
        )
        passage_rows = conn.execute(
            "SELECT * FROM graphiti_input_manifest_passages "
            "WHERE manifest_id=? ORDER BY passage_ordinal",
            (str(row["manifest_id"]),),
        ).fetchall()
        passages: list[GraphitiManifestPassage] = []
        for ordinal, passage_row in enumerate(passage_rows, start=1):
            if int(passage_row["passage_ordinal"]) != ordinal:
                raise AuthorityPersistenceError(
                    "Graphiti manifest passage ordinals are not contiguous"
                )
            passage_value = cls._graphiti_canonical_row_value(
                passage_row, identity="Graphiti manifest passage"
            )
            passage = GraphitiManifestPassage(
                passage_id=ExtractionPassageId.parse(
                    str(passage_row["passage_id"])
                ),
                admission_id=ObjectAdmissionId.parse(
                    str(passage_row["admission_id"])
                ),
                access_decision_id=ObjectAccessDecisionId.parse(
                    str(passage_row["access_decision_id"])
                ),
                hydration_policy_contract_digest=str(
                    passage_row["hydration_policy_contract_digest"]
                ),
                principal_id=str(passage_row["principal_id"]),
                authority_domain=str(passage_row["authority_domain"]),
                purpose=str(passage_row["purpose"]),
                object_class=str(passage_row["object_class"]),
                allowed_use=str(passage_row["allowed_use"]),
                security_scope=str(passage_row["security_scope"]),
                retention_scope=str(passage_row["retention_scope"]),
                byte_offset=int(passage_row["byte_offset"]),
                byte_length=int(passage_row["byte_length"]),
                blob_digest=str(passage_row["blob_digest"]),
                text_digest=str(passage_row["text_digest"]),
                language=str(passage_row["language"]),
            )
            if passage.canonical_value() != passage_value:
                raise AuthorityPersistenceError(
                    "Graphiti manifest passage canonical data differs"
                )
            passages.append(passage)
        manifest = GraphitiInputManifest(
            manifest_id=GraphitiInputManifestId.parse(str(row["manifest_id"])),
            configuration_id=GraphitiAdapterConfigurationId.parse(
                str(row["configuration_id"])
            ),
            configuration_digest=str(row["configuration_digest"]),
            extractor_contract_id=ExtractorContractId.parse(
                str(row["extractor_contract_id"])
            ),
            extractor_contract_digest=str(row["extractor_contract_digest"]),
            run_id=ExtractionRunId.parse(str(row["run_id"])),
            requested_run_version_id=ExtractionRunVersionId.parse(
                str(row["requested_run_version_id"])
            ),
            requested_version_number=int(row["requested_version_number"]),
            definition_id=SourceDefinitionId.parse(str(row["definition_id"])),
            definition_version_id=SourceDefinitionVersionId.parse(
                str(row["definition_version_id"])
            ),
            item_id=SourceItemId.parse(str(row["item_id"])),
            revision_id=SourceRevisionId.parse(str(row["revision_id"])),
            representation_id=DiscoveryRepresentationId.parse(
                str(row["representation_id"])
            ),
            input_binding_digest=str(row["input_binding_digest"]),
            passages=tuple(passages),
        )
        if (
            len(passages) != int(row["passage_count"])
            or manifest.canonical_value() != value
            or manifest.canonical_digest != str(row["canonical_digest"])
        ):
            raise AuthorityPersistenceError("Graphiti input manifest canonical data differs")
        return manifest

    @classmethod
    def _graphiti_cleanup_from_row(
        cls, row: Mapping[str, Any]
    ) -> GraphitiCleanupReceipt:
        value = cls._graphiti_canonical_row_value(
            row, identity="Graphiti cleanup receipt"
        )
        receipt = GraphitiCleanupReceipt(
            receipt_id=GraphitiCleanupReceiptId.parse(str(row["receipt_id"])),
            workspace_id=GraphitiWorkspaceId.parse(str(row["workspace_id"])),
            final_state=GraphitiWorkspaceState(str(row["final_state"])),
            reason=GraphitiCleanupReason(str(row["reason"])),
            private_node_count=int(row["private_node_count"]),
            private_relation_count=int(row["private_relation_count"]),
            file_count=int(row["file_count"]),
            byte_count=int(row["byte_count"]),
            workspace_absent=bool(row["workspace_absent"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
        )
        if receipt.canonical_value() != value or receipt.canonical_digest != str(
            row["canonical_digest"]
        ):
            raise AuthorityPersistenceError("Graphiti cleanup canonical data differs")
        return receipt

    def _graphiti_attempt_from_row(
        self, conn: sqlite3.Connection, row: Mapping[str, Any], *, replayed: bool,
        require_current: bool = False,
    ) -> GraphitiAttemptRecord:
        self._prove_current_record(row)
        value = self._graphiti_canonical_row_value(row, identity="Graphiti attempt")
        cleanup_row = conn.execute(
            "SELECT * FROM graphiti_cleanup_receipts WHERE receipt_id=?",
            (str(row["cleanup_receipt_id"]),),
        ).fetchone()
        if cleanup_row is None:
            raise AuthorityPersistenceError("Graphiti attempt cleanup receipt is missing")
        cleanup = self._graphiti_cleanup_from_row(cleanup_row)
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
        receipt_row = conn.execute(
            "SELECT canonical_bytes,canonical_digest FROM graphiti_attempt_receipts "
            "WHERE attempt_id=?",
            (str(row["attempt_id"]),),
        ).fetchone()
        attempt_receipt = None
        if receipt_row is not None:
            receipt_bytes = bytes(receipt_row["canonical_bytes"])
            if str(receipt_row["canonical_digest"]) != digest_bytes(receipt_bytes):
                raise AuthorityPersistenceError("Graphiti attempt receipt digest differs")
            attempt_receipt = json.loads(receipt_bytes)
            if (
                not isinstance(attempt_receipt, dict)
                or canonical_json_bytes(attempt_receipt) != receipt_bytes
            ):
                raise AuthorityPersistenceError("Graphiti attempt receipt is not canonical")
        record = GraphitiAttemptRecord(
            attempt_id=GraphitiAttemptId.parse(str(row["attempt_id"])),
            run_id=ExtractionRunId.parse(str(row["run_id"])),
            run_version_id=ExtractionRunVersionId.parse(str(row["run_version_id"])),
            attempt_number=int(row["attempt_number"]),
            previous_attempt_id=(
                None
                if row["previous_attempt_id"] is None
                else GraphitiAttemptId.parse(str(row["previous_attempt_id"]))
            ),
            configuration_id=GraphitiAdapterConfigurationId.parse(
                str(row["configuration_id"])
            ),
            configuration_digest=str(row["configuration_digest"]),
            workspace_id=GraphitiWorkspaceId.parse(str(row["workspace_id"])),
            manifest_id=GraphitiInputManifestId.parse(str(row["manifest_id"])),
            outcome=GraphitiAdapterOutcome(str(row["outcome"])),
            failure_code=str(row["failure_code"]),
            started_at=UtcTimestamp.parse(str(row["started_at"])),
            ended_at=UtcTimestamp.parse(str(row["ended_at"])),
            usage=usage,
            output_id=(
                None
                if row["extraction_output_id"] is None
                else ExtractionOutputId.parse(str(row["extraction_output_id"]))
            ),
            proposal_set_id=(
                None
                if row["proposal_set_id"] is None
                else ProposalSetId.parse(str(row["proposal_set_id"]))
            ),
            attempt_receipt=attempt_receipt,
            cleanup_receipt=cleanup,
            authority_event_id=EventId.parse(str(row["authority_event_id"])),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            recovered_ambiguous_progression=(
                None
                if "recovered_ambiguous_progression" not in value
                else RecoveredAmbiguousProgressionProof.from_canonical_value(
                    value["recovered_ambiguous_progression"]
                )
            ),
            replayed=replayed,
        )
        expected = record.canonical_value()
        if value != expected:
            raise AuthorityPersistenceError("Graphiti attempt canonical data differs")
        configuration = self._graphiti_configuration_from_row(
            conn,
            self._graphiti_configuration_row(conn, record.configuration_id),
            replayed=False,
        ).configuration
        manifest_row = conn.execute(
            "SELECT * FROM graphiti_input_manifests WHERE manifest_id=?",
            (str(record.manifest_id),),
        ).fetchone()
        if manifest_row is None:
            raise AuthorityPersistenceError("Graphiti attempt manifest is missing")
        manifest = self._graphiti_manifest_from_row(conn, manifest_row)
        version_row = conn.execute(
            "SELECT * FROM extraction_run_versions WHERE run_version_id=?",
            (str(record.run_version_id),),
        ).fetchone()
        if version_row is None:
            raise AuthorityPersistenceError(
                "Graphiti attempt extraction run version is missing"
            )
        version = self._run_version_from_row(conn, version_row, replayed=False)
        contract = self._contract_from_row(
            conn,
            self._contract_row(conn, str(configuration.extractor_contract_id)),
            replayed=False,
        ).request
        if (
            contract.producer_kind == "GRAPHITI_EVALUATION"
            and record.output_id is None
            and record.attempt_receipt is None
        ):
            raise AuthorityPersistenceError(
                "Graphiti evaluation attempt lacks its terminal receipt"
            )
        replay_binding = conn.execute(
            "SELECT * FROM graphiti_adapter_attempt_replays WHERE attempt_id=?",
            (str(record.attempt_id),),
        ).fetchone()
        replay_source = None
        if replay_binding is not None:
            replay_row = conn.execute(
                "SELECT * FROM graphiti_replay_sources WHERE replay_source_id=?",
                (str(replay_binding["replay_source_id"]),),
            ).fetchone()
            if replay_row is None:
                raise AuthorityPersistenceError(
                    "Graphiti attempt replay source is missing"
                )
            replay_source = self._graphiti_replay_source_from_row(
                conn, replay_row, replayed=False
            ).source
        event = self._graphiti_record_context(
            conn, event_id=str(row["authority_event_id"])
        )
        payload = json.loads(bytes(event["payload_bytes"]))
        if not isinstance(payload, dict):
            raise AuthorityPersistenceError(
                "Graphiti attempt command payload is not canonical"
            )
        request = GraphitiAttemptRequest(
            attempt_id=record.attempt_id,
            attempt_number=record.attempt_number,
            expected_previous_attempt_id=record.previous_attempt_id,
            configuration=configuration,
            workspace_id=record.workspace_id,
            cleanup_receipt_id=record.cleanup_receipt.receipt_id,
            manifest=manifest,
            extraction_contract=contract,
            extraction_request=version.request,
            replay_source=replay_source,
            idempotency_key=str(event["idempotency_key"]),
            reference_time=(
                None
                if payload.get("reference_time") is None
                else UtcTimestamp.parse(str(payload["reference_time"]))
            ),
            temporal_basis=TemporalBasis(str(payload["temporal_basis"])),
            episode_uuid=(
                None
                if payload.get("episode_uuid") is None
                else str(payload["episode_uuid"])
            ),
            generation_id=str(payload["generation_id"]),
            predecessor_episode_uuid=(
                None
                if payload.get("predecessor_episode_uuid") is None
                else str(payload["predecessor_episode_uuid"])
            ),
            recovered_ambiguous_progression=(
                None
                if "recovered_ambiguous_progression" not in payload
                else RecoveredAmbiguousProgressionProof.from_canonical_value(
                    payload["recovered_ambiguous_progression"]
                )
            ),
        )
        if (
            request.recovered_ambiguous_progression
            != record.recovered_ambiguous_progression
        ):
            raise AuthorityPersistenceError(
                "Graphiti attempt recovery proof differs from command authority"
            )
        self._validate_graphiti_record_envelope(
            conn,
            row,
            command_type=GRAPHITI_ATTEMPT_EXECUTE_COMMAND,
            aggregate_id=str(record.attempt_id),
            payload_bytes=request.canonical_bytes,
        )
        if require_current:
            self._require_graphiti_attempt_current(
                conn, record, selected_version=version,
            )
        return record

    @classmethod
    def _graphiti_replay_source_from_row(
        cls, conn: sqlite3.Connection, row: Mapping[str, Any], *, replayed: bool
    ) -> GraphitiReplaySourceRecord:
        value = cls._graphiti_canonical_row_value(
            row, identity="Graphiti replay source"
        )
        source = GraphitiReplaySource(
            replay_source_id=GraphitiReplaySourceId.parse(
                str(row["replay_source_id"])
            ),
            source_attempt_id=GraphitiAttemptId.parse(
                str(row["source_attempt_id"])
            ),
            source_run_version_id=ExtractionRunVersionId.parse(
                str(row["source_run_version_id"])
            ),
            source_output_id=ExtractionOutputId.parse(str(row["source_output_id"])),
            source_proposal_set_id=(
                None
                if row["source_proposal_set_id"] is None
                else ProposalSetId.parse(str(row["source_proposal_set_id"]))
            ),
            eligibility=GraphitiReplayEligibility(str(row["eligibility"])),
            output_canonical_digest=str(row["output_canonical_digest"]),
            proposal_set_canonical_digest=(
                None
                if row["proposal_set_canonical_digest"] is None
                else str(row["proposal_set_canonical_digest"])
            ),
            replay_payload_digest=str(row["replay_payload_digest"]),
            approval_event_digest=str(row["approval_event_digest"]),
        )
        if source.canonical_value() != value or source.canonical_digest != str(
            row["canonical_digest"]
        ):
            raise AuthorityPersistenceError("Graphiti replay source canonical data differs")
        event_context = cls._graphiti_record_context(
            conn, event_id=str(row["approval_event_id"])
        )
        approval = GraphitiReplayApprovalRequest(
            replay_source_id=source.replay_source_id,
            source_attempt_id=source.source_attempt_id,
            source_run_version_id=source.source_run_version_id,
            source_output_id=source.source_output_id,
            source_proposal_set_id=source.source_proposal_set_id,
            eligibility=source.eligibility,
            expected_output_canonical_digest=source.output_canonical_digest,
            expected_proposal_set_canonical_digest=(
                source.proposal_set_canonical_digest
            ),
            expected_replay_payload_digest=source.replay_payload_digest,
            idempotency_key=str(event_context["idempotency_key"]),
        )
        event = cls._validate_graphiti_record_envelope(
            conn,
            row,
            command_type=GRAPHITI_REPLAY_APPROVE_COMMAND,
            aggregate_id=str(source.replay_source_id),
            payload_bytes=approval.canonical_bytes,
            event_id_column="approval_event_id",
            recorded_at_column="approved_at",
        )
        if graphiti_event_digest(event) != source.approval_event_digest:
            raise AuthorityPersistenceError(
                "Graphiti replay source approval digest differs"
            )
        return GraphitiReplaySourceRecord(
            source=source,
            authority_event_id=EventId.parse(str(row["approval_event_id"])),
            aggregate_version=1,
            approved_at=UtcTimestamp.parse(str(row["approved_at"])),
            replayed=replayed,
        )

    def _require_graphiti_configuration_current(
        self,
        conn: sqlite3.Connection,
        record: GraphitiAdapterConfigurationRecord,
    ) -> None:
        configuration = record.configuration
        contract_row = self._contract_row(
            conn, str(configuration.extractor_contract_id)
        )
        contract = self._contract_from_row(conn, contract_row, replayed=False)
        if (
            contract.request.digest != configuration.extractor_contract_digest
            or contract.request.contract_id != configuration.extractor_contract_id
        ):
            raise AuthorityPersistenceError(
                "Graphiti configuration extractor contract is no longer exact"
            )
        policy_row = self._graphiti_workspace_policy_row(
            conn, configuration.workspace_policy.policy_id
        )
        policy = self._graphiti_workspace_policy_from_row(policy_row)
        if policy != configuration.workspace_policy:
            raise AuthorityPersistenceError(
                "Graphiti configuration workspace policy is no longer exact"
            )
        configuration.require_execution_authorized()

    def _validate_graphiti_attempt_lineage(
        self, conn: sqlite3.Connection, attempt: GraphitiAttemptRecord,
        *, selected_version: ExtractionRunVersion | None = None,
    ):
        if selected_version is None:
            version_row = conn.execute(
                "SELECT * FROM extraction_run_versions WHERE run_version_id=?",
                (str(attempt.run_version_id),),
            ).fetchone()
            if version_row is None:
                raise AuthorityPersistenceError(
                    "Graphiti attempt extraction run version is missing"
                )
            result = self._run_version_from_row(conn, version_row, replayed=False)
        else:
            # The current-read decoder supplies its exact authenticated version
            # within the same locked operation, never across independent reads.
            result = selected_version
        if (
            type(result) is not ExtractionRunVersion
            or result.request.run_version_id != attempt.run_version_id
        ):
            raise AuthorityPersistenceError(
                "Graphiti attempt differs from retained Extraction Run authority"
            )
        expected_outcome = {
            ExtractionOutcome.SUCCESS: GraphitiAdapterOutcome.COMPLETE,
            ExtractionOutcome.PARTIAL: GraphitiAdapterOutcome.PARTIAL,
            ExtractionOutcome.INVALID_OUTPUT: GraphitiAdapterOutcome.MALFORMED_OUTPUT,
            ExtractionOutcome.RETRYABLE_FAILURE: {
                ExtractionFailureCode.AMBIGUOUS_EFFECT: (
                    GraphitiAdapterOutcome.AMBIGUOUS_EFFECT
                ),
                ExtractionFailureCode.EXECUTION_TIMEOUT: GraphitiAdapterOutcome.TIMEOUT,
            }.get(result.failure_code, GraphitiAdapterOutcome.FAILED),
            ExtractionOutcome.BLOCKING_FAILURE: (
                GraphitiAdapterOutcome.POLICY_BLOCKED
                if result.failure_code is ExtractionFailureCode.POLICY_BLOCKED
                else GraphitiAdapterOutcome.PROVIDER_REJECTED
            ),
        }[result.outcome]
        expected_output_id = None if result.output is None else result.output.output_id
        expected_proposal_set_id = (
            None if result.proposal_set is None else result.proposal_set.proposal_set_id
        )
        if (
            attempt.run_id != result.request.run_id
            or attempt.outcome is not expected_outcome
            or attempt.failure_code != result.failure_code.value
            or attempt.started_at != result.started_at
            or attempt.ended_at != result.ended_at
            or attempt.usage != result.usage
            or attempt.output_id != expected_output_id
            or attempt.proposal_set_id != expected_proposal_set_id
        ):
            raise AuthorityPersistenceError(
                "Graphiti attempt differs from retained Extraction Run authority"
            )
        return result

    def _require_graphiti_attempt_current(
        self, conn: sqlite3.Connection, attempt: GraphitiAttemptRecord,
        *, selected_version: ExtractionRunVersion | None = None,
    ) -> None:
        result = self._validate_graphiti_attempt_lineage(
            conn, attempt, selected_version=selected_version,
        )
        try:
            self._revalidate_result_current(conn, result)
        except PermissionError as exc:
            raise GraphitiAdapterRightsDenied(str(exc)) from exc

    def _require_graphiti_workspace_absent(
        self, workspace: GraphitiWorkspaceDescriptor
    ) -> None:
        if (self._workspace_root / workspace.namespace).exists():
            raise AuthorityPersistenceError(
                "disposable Graphiti workspace still exists after cleanup"
            )


__all__ = [
    "_GraphitiAdapterStoreSupport",
    "graphiti_event_digest",
]
