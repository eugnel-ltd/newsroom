from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from uuid import UUID
from typing import Any

from newsroom.authority._capability import _AuthorizedCommandGrant
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.authority.types import PayloadMode, TrustScope, UtcTimestamp
from newsroom.entities.models import (
    EntityMergeDecisionRequest,
    EntityMention,
    EntityMentionAdmissionRequest,
    EntityReversalDecisionRequest,
    EntityResolutionDecisionRequest,
    EntityResolutionDependency,
    EntityResolutionProposalRequest,
    EntitySplitDecisionRequest,
)
from newsroom.entities.policy import (
    ENTITY_MENTION_ADMIT_COMMAND,
    ENTITY_MERGE_DECIDE_COMMAND,
    ENTITY_RESOLUTION_DECIDE_COMMAND,
    ENTITY_RESOLUTION_DEPENDENCY_BIND_COMMAND,
    ENTITY_RESOLUTION_PROPOSE_COMMAND,
    ENTITY_REVERSAL_DECIDE_COMMAND,
    ENTITY_SPLIT_DECIDE_COMMAND,
)
from newsroom.entities.types import (
    CanonicalEntityId,
    CanonicalEntityVersionId,
    EntityAliasId,
    EntityCreationDecisionKind,
    EntityDecisionConflict,
    EntityIdentifierReuse,
    EntityMentionId,
    EntityMergeDecisionId,
    EntityReversalDecisionId,
    EntityResolutionDecisionId,
    EntityResolutionDependencyId,
    EntityResolutionProposalId,
    EntityResolutionProposalVersionId,
    EntityRightsDenied,
    EntitySemanticCollision,
    EntityStaleDecision,
    EntityStateError,
    EntitySplitDecisionId,
)
from newsroom.extraction.models import ProposalEnvelope
from newsroom.extraction.types import ExtractionProposalKind, ProposalEnvelopeId


_RECORD_SPECS: dict[str, tuple[str, str, TrustScope]] = {
    ENTITY_MENTION_ADMIT_COMMAND: (
        "entity_mention",
        "entity.mention.admitted",
        TrustScope.PROPOSED,
    ),
    ENTITY_RESOLUTION_PROPOSE_COMMAND: (
        "entity_resolution_proposal_version",
        "entity.resolution.proposed",
        TrustScope.PROPOSED,
    ),
    ENTITY_RESOLUTION_DECIDE_COMMAND: (
        "entity_resolution_decision",
        "entity.resolution.decided",
        TrustScope.ADMITTED,
    ),
    ENTITY_RESOLUTION_DEPENDENCY_BIND_COMMAND: (
        "entity_resolution_dependency",
        "entity.resolution.dependency.bound",
        TrustScope.PROPOSED,
    ),
    ENTITY_MERGE_DECIDE_COMMAND: (
        "entity_merge_decision",
        "entity.merge.decided",
        TrustScope.ADMITTED,
    ),
    ENTITY_SPLIT_DECIDE_COMMAND: (
        "entity_split_decision",
        "entity.split.decided",
        TrustScope.ADMITTED,
    ),
    ENTITY_REVERSAL_DECIDE_COMMAND: (
        "entity_reversal_decision",
        "entity.reversal.decided",
        TrustScope.ADMITTED,
    ),
}


def _deterministic_v4(identifier_type: type, domain: bytes, value: str):
    digest = hashlib.sha256(domain + b"\0" + value.encode("ascii")).digest()
    raw = bytearray(digest[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return identifier_type(UUID(bytes=bytes(raw)))


def deterministic_decision_id(request: EntityResolutionDecisionRequest) -> EntityResolutionDecisionId:
    """Derive a stable opaque UUIDv4-shaped identity from exact decision semantics.

    The public decision request deliberately carries optimistic concurrency rather
    than a caller-selected decision identity.  The authority derives the identity
    before authorisation, so exact replay resolves to the same aggregate while a
    changed semantic request cannot silently reuse it.
    """

    return _deterministic_v4(
        EntityResolutionDecisionId,
        b"newsroom.entity-resolution-decision.v1",
        request.digest,
    )


def deterministic_lineage_version_id(
    *, decision_kind: str, decision_id: str, entity_id: CanonicalEntityId, role: str
) -> CanonicalEntityVersionId:
    return _deterministic_v4(
        CanonicalEntityVersionId,
        b"newsroom.entity-lineage-version.v1",
        f"{decision_kind}:{decision_id}:{entity_id}:{role}",
    )


def deterministic_projection_event_id(
    *, source_event_id: str, entity_id: CanonicalEntityId
):
    from newsroom.authority.types import EventId

    return _deterministic_v4(
        EventId,
        b"newsroom.entity-projection-event.v1",
        f"{source_event_id}:{entity_id}",
    )


class _EntityStoreSupport:
    def _require_entity_grant(
        self,
        grant: _AuthorizedCommandGrant,
        *,
        command_type: str,
        aggregate_id: str,
        expected_aggregate_version: int,
        canonical_bytes: bytes,
    ) -> None:
        self._issuer.verify(grant)
        spec = _RECORD_SPECS.get(command_type)
        if spec is None:
            raise AuthorityPersistenceError("unknown entity authority command")
        aggregate_type, event_type, trust_scope = spec
        definition = grant.definition
        if (
            grant.command_type != command_type
            or grant.aggregate_id != aggregate_id
            or grant.expected_aggregate_version != expected_aggregate_version
            or definition.command_type != command_type
            or definition.aggregate_type != aggregate_type
            or definition.event_type != event_type
            or definition.trust_scope is not trust_scope
            or definition.security_scope != "authority.entity"
            or definition.retention_scope != "authority.audit"
            or definition.payload_mode is not PayloadMode.INLINE
            or grant.payload.kind != PayloadMode.INLINE.value
            or grant.payload.inline_bytes != canonical_bytes
            or grant.payload.digest != digest_bytes(canonical_bytes)
        ):
            raise AuthorityPersistenceError("entity grant differs from the typed record")

    @staticmethod
    def _ensure_identifier_absent(
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
            raise EntityIdentifierReuse(
                f"{identity} is already retained under different command identity"
            )

    @staticmethod
    def _ensure_semantic_absent(
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
            raise EntitySemanticCollision(
                f"{identity} already exists under another stable identity"
            )

    @staticmethod
    def _decode_json_blob(value: bytes | memoryview, *, identity: str) -> Any:
        data = bytes(value)
        try:
            decoded = json.loads(data.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise AuthorityPersistenceError(f"{identity} retained JSON is invalid") from exc
        if canonical_json_bytes(decoded) != data:
            raise AuthorityPersistenceError(f"{identity} retained JSON is not canonical")
        return decoded

    @classmethod
    def _canonical_row_value(
        cls, row: Mapping[str, Any], *, identity: str
    ) -> dict[str, Any]:
        data = bytes(row["canonical_bytes"])
        if digest_bytes(data) != str(row["canonical_digest"]):
            raise AuthorityPersistenceError(f"{identity} canonical digest mismatch")
        value = cls._decode_json_blob(data, identity=identity)
        if not isinstance(value, dict):
            raise AuthorityPersistenceError(f"{identity} must be a canonical object")
        return value

    @staticmethod
    def _record_context(conn: sqlite3.Connection, *, event_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT e.*,c.idempotency_key,p.payload_bytes "
            "FROM ledger_events e "
            "JOIN authority_commands c ON c.command_id=e.command_id "
            "JOIN authority_payloads p ON p.payload_id=e.payload_id "
            "WHERE e.event_id=?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise AuthorityPersistenceError("entity record has no exact authority event")
        return row

    @classmethod
    def _validate_entity_record_envelope(
        cls,
        conn: sqlite3.Connection,
        row: Mapping[str, Any],
        *,
        command_type: str,
        aggregate_id: str,
        payload_bytes: bytes,
        payload_digest: str,
    ) -> sqlite3.Row:
        event = cls._record_context(conn, event_id=str(row["authority_event_id"]))
        aggregate_type, event_type, trust_scope = _RECORD_SPECS[command_type]
        if (
            str(event["event_type"]) != event_type
            or int(event["event_schema_version"]) != 1
            or str(event["aggregate_type"]) != aggregate_type
            or str(event["aggregate_id"]) != aggregate_id
            or int(event["aggregate_version"])
            != int(row["authority_aggregate_version"])
            or int(row["authority_aggregate_version"]) != 1
            or str(event["recorded_at"]) != str(row["recorded_at"])
            or str(event["security_scope"]) != "authority.entity"
            or str(event["retention_scope"]) != "authority.audit"
            or str(event["trust_scope"]) != trust_scope.value
            or str(event["payload_mode"]) != PayloadMode.INLINE.value
            or str(event["payload_digest"]) != payload_digest
            or event["payload_bytes"] is None
            or bytes(event["payload_bytes"]) != payload_bytes
            or digest_bytes(payload_bytes) != payload_digest
        ):
            raise AuthorityPersistenceError("entity record authority envelope is inconsistent")
        return event

    def _source_proposal(
        self,
        conn: sqlite3.Connection,
        proposal_id: ProposalEnvelopeId,
    ) -> ProposalEnvelope:
        row = conn.execute(
            "SELECT p.*,s.producer_contract_digest AS set_contract_digest,"
            "s.retained_at AS set_retained_at "
            "FROM extraction_proposals p "
            "JOIN extraction_proposal_sets s ON s.proposal_set_id=p.proposal_set_id "
            "WHERE p.proposal_id=?",
            (str(proposal_id),),
        ).fetchone()
        if row is None:
            raise EntityStateError("source extraction proposal is not retained")
        proposal = self._proposal_from_row(
            conn,
            row,
            expected_proposal_set_id=str(row["proposal_set_id"]),
            expected_output_id=str(row["output_id"]),
            expected_run_id=str(row["run_id"]),
            expected_run_version_id=str(row["run_version_id"]),
            expected_contract_digest=str(row["set_contract_digest"]),
            expected_retained_at=str(row["set_retained_at"]),
        )
        version_row = conn.execute(
            "SELECT * FROM extraction_run_versions WHERE run_version_id=?",
            (str(proposal.run_version_id),),
        ).fetchone()
        if version_row is None:
            raise AuthorityPersistenceError("source proposal run version is missing")
        result = self._run_version_from_row(conn, version_row, replayed=False)
        self._revalidate_result_current(conn, result)
        return proposal

    def _mention_row(self, conn: sqlite3.Connection, mention_id: EntityMentionId) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM entity_mentions WHERE mention_id=?", (str(mention_id),)
        ).fetchone()
        if row is None:
            raise EntityStateError("entity mention is not retained")
        return row

    def _require_mention_current(
        self, conn: sqlite3.Connection, mention: EntityMention
    ) -> None:
        try:
            # Mentions retain an exact 4A Proposal Envelope identity.  Resolve that
            # proposal through the extraction authority so current-use validation
            # covers the complete immutable Run Version, not only the evidence
            # passage copied into the mention.  This prevents a rights-invalid run
            # from being treated as partially usable by later entity authority.
            proposal = self._source_proposal(conn, mention.source_proposal_id)
        except PermissionError as exc:
            raise EntityRightsDenied(str(exc)) from exc
        if (
            proposal.canonical_digest != mention.source_proposal_digest
            or proposal.proposal_set_id != mention.proposal_set_id
            or proposal.output_id != mention.output_id
            or proposal.run_id != mention.run_id
            or proposal.run_version_id != mention.run_version_id
            or not any(
                evidence.passage_id == mention.passage_id
                and evidence.start_byte == mention.start_byte
                and evidence.end_byte == mention.end_byte
                and evidence.evidence_text_digest == mention.evidence_text_digest
                for evidence in proposal.evidence
            )
        ):
            raise AuthorityPersistenceError(
                "entity mention current proposal provenance differs"
            )

    def _require_mention_id_current(
        self, conn: sqlite3.Connection, mention_id: EntityMentionId
    ) -> EntityMention:
        mention = self._mention_from_row(
            conn, self._mention_row(conn, mention_id), replayed=False
        )
        self._require_mention_current(conn, mention)
        return mention

    def _require_resolution_proposal_current(
        self, conn: sqlite3.Connection, proposal_id: EntityResolutionProposalId
    ):
        row = conn.execute(
            "SELECT v.* FROM entity_resolution_proposal_heads h "
            "JOIN entity_resolution_proposal_versions v "
            "ON v.proposal_version_id=h.current_proposal_version_id "
            "WHERE h.resolution_proposal_id=?",
            (str(proposal_id),),
        ).fetchone()
        if row is None:
            raise EntityStateError("entity resolution proposal is not retained")
        proposal = self._proposal_version_from_row(conn, row, replayed=False)
        self._require_mention_id_current(conn, proposal.subject_mention_id)
        if proposal.object_mention_id is not None:
            self._require_mention_id_current(conn, proposal.object_mention_id)
        return proposal

    def _require_dependency_current(
        self,
        conn: sqlite3.Connection,
        dependency: EntityResolutionDependency,
    ) -> None:
        dependent = self._source_proposal(conn, dependency.dependent_proposal_id)
        if (
            dependent.kind is not ExtractionProposalKind.RELATION
            or dependent.canonical_digest != dependency.dependent_proposal_digest
        ):
            raise AuthorityPersistenceError(
                "dependent extraction proposal provenance differs"
            )
        proposal = self._require_resolution_proposal_current(
            conn, dependency.resolution_proposal_id
        )
        if (
            proposal.proposal_version_id != dependency.proposal_version_id
            or proposal.canonical_digest != dependency.proposal_version_digest
        ):
            raise EntityStaleDecision(
                "resolution dependency no longer names the current proposal"
            )

    def _require_entity_current(
        self,
        conn: sqlite3.Connection,
        entity_id: CanonicalEntityId,
    ) -> None:
        row = conn.execute(
            "SELECT * FROM canonical_entities WHERE entity_id=?",
            (str(entity_id),),
        ).fetchone()
        if row is None:
            raise EntityStateError("canonical entity is not retained")
        entity = self._entity_from_row(conn, row)
        if self._current_state_only:
            self._entity_head_row(conn, entity_id)

        if entity.created_by_kind is EntityCreationDecisionKind.RESOLUTION:
            decision = conn.execute(
                "SELECT resolution_proposal_id,accepted_entity_id "
                "FROM entity_resolution_decisions WHERE decision_id=?",
                (entity.created_by_decision_id,),
            ).fetchone()
            if (
                decision is None
                or str(decision["accepted_entity_id"]) != str(entity.entity_id)
            ):
                raise AuthorityPersistenceError(
                    "canonical entity resolution provenance is inconsistent"
                )
            self._require_resolution_proposal_current(
                conn,
                EntityResolutionProposalId.parse(
                    str(decision["resolution_proposal_id"])
                ),
            )
            return

        if entity.created_by_kind is EntityCreationDecisionKind.MERGE:
            decision_row = conn.execute(
                "SELECT * FROM entity_merge_decisions WHERE merge_decision_id=?",
                (entity.created_by_decision_id,),
            ).fetchone()
            if decision_row is None:
                raise AuthorityPersistenceError(
                    "canonical entity merge provenance is missing"
                )
            decision = self._merge_decision_from_row(
                conn, decision_row, replayed=False
            )
            if decision.successor_entity_id != entity.entity_id:
                raise AuthorityPersistenceError(
                    "canonical entity merge provenance differs"
                )
            for proposal_id in decision.basis_resolution_proposal_ids:
                self._require_resolution_proposal_current(conn, proposal_id)
            return

        if entity.created_by_kind is EntityCreationDecisionKind.SPLIT:
            decision_row = conn.execute(
                "SELECT * FROM entity_split_decisions WHERE split_decision_id=?",
                (entity.created_by_decision_id,),
            ).fetchone()
            if decision_row is None:
                raise AuthorityPersistenceError(
                    "canonical entity split provenance is missing"
                )
            decision = self._split_decision_from_row(
                conn, decision_row, replayed=False
            )
            if entity.entity_id not in {item.entity_id for item in decision.successors}:
                raise AuthorityPersistenceError(
                    "canonical entity split provenance differs"
                )
            allocations = tuple(
                item for item in decision.allocations
                if item.successor_entity_id == entity.entity_id
            )
            if not allocations:
                raise AuthorityPersistenceError(
                    "split successor has no mention provenance"
                )
            for allocation in allocations:
                self._require_mention_id_current(conn, allocation.mention_id)
            return

        raise AuthorityPersistenceError("unknown canonical entity creation provenance")

    @staticmethod
    def _proposal_head_row(
        conn: sqlite3.Connection, proposal_id: EntityResolutionProposalId
    ) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM entity_resolution_proposal_heads "
            "WHERE resolution_proposal_id=?",
            (str(proposal_id),),
        ).fetchone()

    @staticmethod
    def _decision_head_row(
        conn: sqlite3.Connection, proposal_id: EntityResolutionProposalId
    ) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM entity_resolution_decision_heads "
            "WHERE resolution_proposal_id=?",
            (str(proposal_id),),
        ).fetchone()

    @staticmethod
    def _entity_head_row(
        conn: sqlite3.Connection, entity_id: CanonicalEntityId
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT h.*,(SELECT MAX(version_number) FROM canonical_entity_versions "
            "WHERE entity_id=h.entity_id) AS selected_max_version FROM canonical_entity_heads h WHERE entity_id=?",
            (str(entity_id),),
        ).fetchone()
        if row is None:
            raise EntityStateError("canonical entity is not retained")
        if row["current_version_number"] != row["selected_max_version"]:
            raise AuthorityPersistenceError("current canonical entity head is not the latest version")
        return row

    @classmethod
    def _require_candidate_current(
        cls,
        conn: sqlite3.Connection,
        *,
        entity_id: CanonicalEntityId,
        version_id: CanonicalEntityVersionId,
    ) -> sqlite3.Row:
        head = cls._entity_head_row(conn, entity_id)
        if str(head["current_entity_version_id"]) != str(version_id):
            raise EntityStaleDecision("candidate entity version is no longer current")
        if str(head["lifecycle"]) != "ACTIVE":
            raise EntityStateError("candidate canonical entity is not active")
        return head

    @staticmethod
    def _require_proposal_open(conn: sqlite3.Connection, proposal_id: EntityResolutionProposalId) -> None:
        head = conn.execute(
            "SELECT current_state,terminal FROM entity_resolution_decision_heads "
            "WHERE resolution_proposal_id=?",
            (str(proposal_id),),
        ).fetchone()
        if head is not None and bool(head["terminal"]):
            raise EntityDecisionConflict(
                f"resolution proposal already has terminal state {head['current_state']}"
            )

    @staticmethod
    def _json_bytes(values: tuple[str, ...]) -> bytes:
        return canonical_json_bytes(list(values))


__all__ = [
    "_EntityStoreSupport",
    "deterministic_decision_id",
    "deterministic_lineage_version_id",
    "deterministic_projection_event_id",
]
