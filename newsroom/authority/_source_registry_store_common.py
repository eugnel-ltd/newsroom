from __future__ import annotations

import sqlite3
from typing import Any, Mapping

from newsroom.authority._capability import _AuthorizedCommandGrant
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.authority.types import PayloadMode, TrustScope, UtcTimestamp
from newsroom.sources.policy import (
    DISCOVERY_OCCURRENCE_RECORD_COMMAND,
    DISCOVERY_REPRESENTATION_RECORD_COMMAND,
    SOURCE_DEFINITION_REGISTER_COMMAND,
    SOURCE_DEFINITION_VERSION_RECORD_COMMAND,
    SOURCE_ITEM_REGISTER_COMMAND,
    SOURCE_LOCATOR_CONTINUITY_DECIDE_COMMAND,
    SOURCE_REVISION_RECORD_COMMAND,
)
from newsroom.sources.types import (
    SourceDefinitionId,
    SourceDefinitionVersionId,
    SourceIdentifierReuse,
    SourceSemanticCollision,
    SourceStateError,
    SourceVersionConflict,
)

_RECORD_SPECS: dict[str, tuple[str, str, TrustScope]] = {
    SOURCE_DEFINITION_REGISTER_COMMAND: (
        "source_definition",
        "source.definition.registered",
        TrustScope.ADMITTED,
    ),
    SOURCE_DEFINITION_VERSION_RECORD_COMMAND: (
        "source_definition_version",
        "source.definition.version.recorded",
        TrustScope.ADMITTED,
    ),
    SOURCE_ITEM_REGISTER_COMMAND: (
        "source_item",
        "source.item.registered",
        TrustScope.OBSERVED,
    ),
    SOURCE_LOCATOR_CONTINUITY_DECIDE_COMMAND: (
        "source_locator_continuity",
        "source.locator.continuity.decided",
        TrustScope.ADMITTED,
    ),
    SOURCE_REVISION_RECORD_COMMAND: (
        "source_revision",
        "source.revision.recorded",
        TrustScope.OBSERVED,
    ),
    DISCOVERY_REPRESENTATION_RECORD_COMMAND: (
        "discovery_representation",
        "discovery.representation.recorded",
        TrustScope.OBSERVED,
    ),
    DISCOVERY_OCCURRENCE_RECORD_COMMAND: (
        "discovery_occurrence",
        "discovery.occurrence.recorded",
        TrustScope.OBSERVED,
    ),
}


class _SourceRegistryStoreSupport:
    def _require_source_grant(
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
            raise AuthorityPersistenceError(
                "unknown source registry command"
            )
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
            or definition.security_scope != "authority.source_registry"
            or definition.retention_scope != "authority.audit"
            or definition.payload_mode is not PayloadMode.INLINE
            or grant.payload.kind != PayloadMode.INLINE.value
            or grant.payload.inline_bytes != canonical_bytes
            or grant.payload.digest != digest_bytes(canonical_bytes)
        ):
            raise AuthorityPersistenceError(
                "source registry grant differs from the typed record"
            )

    @staticmethod
    def _ensure_identifier_absent(
        conn: sqlite3.Connection,
        *,
        table: str,
        column: str,
        identifier: str,
        identity: str,
    ) -> None:
        row = conn.execute(
            f"SELECT 1 FROM {table} WHERE {column}=?", (identifier,)
        ).fetchone()
        if row is not None:
            raise SourceIdentifierReuse(
                f"{identity} is already retained under different command identity"
            )

    @staticmethod
    def _ensure_semantic_absent(
        conn: sqlite3.Connection,
        *,
        table: str,
        predicate: str,
        parameters: tuple[object, ...],
        identity: str,
    ) -> None:
        row = conn.execute(
            f"SELECT 1 FROM {table} WHERE {predicate}", parameters
        ).fetchone()
        if row is not None:
            raise SourceSemanticCollision(
                f"{identity} already exists under a different stable identity"
            )

    @staticmethod
    def _definition_row(
        conn: sqlite3.Connection, definition_id: SourceDefinitionId
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM source_definitions WHERE definition_id=?",
            (str(definition_id),),
        ).fetchone()
        if row is None:
            raise SourceStateError("source definition is not retained")
        return row

    @staticmethod
    def _version_row(
        conn: sqlite3.Connection,
        version_id: SourceDefinitionVersionId,
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM source_definition_versions WHERE version_id=?",
            (str(version_id),),
        ).fetchone()
        if row is None:
            raise SourceStateError(
                "source definition version is not retained"
            )
        return row

    @staticmethod
    def _source_version_was_current_at_event(
        conn: sqlite3.Connection,
        *,
        definition_id: str,
        version_id: str,
        record_event_id: str,
    ) -> bool:
        row = conn.execute(
            "SELECT v.version_number,ve.ledger_seq AS version_seq,"
            "re.ledger_seq AS record_seq "
            "FROM source_definition_versions v "
            "JOIN ledger_events ve ON ve.event_id=v.authority_event_id "
            "JOIN ledger_events re ON re.event_id=? "
            "WHERE v.version_id=? AND v.definition_id=?",
            (record_event_id, version_id, definition_id),
        ).fetchone()
        if row is None or int(row["version_seq"]) > int(row["record_seq"]):
            return False
        next_row = conn.execute(
            "SELECT MIN(e.ledger_seq) AS next_seq "
            "FROM source_definition_versions v "
            "JOIN ledger_events e ON e.event_id=v.authority_event_id "
            "WHERE v.definition_id=? AND v.version_number>?",
            (definition_id, int(row["version_number"])),
        ).fetchone()
        next_seq = None if next_row is None else next_row["next_seq"]
        return next_seq is None or int(row["record_seq"]) < int(next_seq)

    @staticmethod
    def _source_version_is_same_or_later(
        conn: sqlite3.Connection,
        *,
        definition_id: str,
        lineage_version_id: str,
        observation_version_id: str,
    ) -> bool:
        lineage = conn.execute(
            "SELECT definition_id,version_number "
            "FROM source_definition_versions WHERE version_id=?",
            (lineage_version_id,),
        ).fetchone()
        observation = conn.execute(
            "SELECT definition_id,version_number "
            "FROM source_definition_versions WHERE version_id=?",
            (observation_version_id,),
        ).fetchone()
        return bool(
            lineage is not None
            and observation is not None
            and str(lineage["definition_id"]) == definition_id
            and str(observation["definition_id"]) == definition_id
            and int(observation["version_number"])
            >= int(lineage["version_number"])
        )

    @staticmethod
    def _current_version_row(
        conn: sqlite3.Connection,
        definition_id: SourceDefinitionId,
    ) -> sqlite3.Row | None:
        row = conn.execute(
            "SELECT h.current_version_number,h.current_version_id,v.*,"
            "(SELECT MAX(version_number) FROM source_definition_versions WHERE definition_id=h.definition_id) AS selected_max_version "
            "FROM source_definition_version_heads h "
            "JOIN source_definition_versions v "
            "ON v.version_id=h.current_version_id "
            "WHERE h.definition_id=?",
            (str(definition_id),),
        ).fetchone()
        if row is not None and (
            row["definition_id"] != str(definition_id)
            or row["version_number"] != row["current_version_number"]
            or row["version_number"] != row["selected_max_version"]
        ):
            raise AuthorityPersistenceError("current source version head is not the exact latest producer")
        return row

    @staticmethod
    def _require_current_version(
        conn: sqlite3.Connection,
        *,
        definition_id: SourceDefinitionId,
        version_id: SourceDefinitionVersionId,
    ) -> sqlite3.Row:
        row = _SourceRegistryStoreSupport._current_version_row(
            conn, definition_id
        )
        if row is None or str(row["current_version_id"]) != str(version_id):
            raise SourceVersionConflict(
                "source operation is not pinned to the exact current version"
            )
        return row

    @staticmethod
    def _record_context(
        conn: sqlite3.Connection,
        *,
        event_id: str,
    ) -> sqlite3.Row:
        header = None
        if conn.execute('PRAGMA user_version').fetchone()[0] in (44,45):
            header = conn.execute('SELECT native_checkpoint_header(?)',(event_id,)).fetchone()[0]
        if header is not None:
            import json
            envelope = json.loads(header)
            if envelope.get('checkpoint_error') == 'EXPIRED':
                from .persistence import DiagnosticHistoryExpired
                raise DiagnosticHistoryExpired('source RPC provenance expired')
            if 'checkpoint_error' in envelope:
                raise AuthorityPersistenceError('selected checkpoint original proof differs')
            payload = conn.execute('SELECT payload_bytes FROM authority_payloads WHERE payload_id=?',
                (envelope['payload_id'],)).fetchone()
            if payload is None:
                raise AuthorityPersistenceError('checkpoint business payload is missing')
            envelope['payload_bytes'] = payload[0]
            return envelope
        expired = conn.execute('SELECT retired_header_digest FROM ledger_events WHERE event_id=?',(event_id,)).fetchone()
        if expired is not None and expired[0] is not None:
            from .persistence import DiagnosticHistoryExpired
            raise DiagnosticHistoryExpired('source RPC provenance expired')
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
                "source record has no exact authority event"
            )
        return row

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
        event = cls._record_context(
            conn, event_id=str(row["authority_event_id"])
        )
        aggregate_type, event_type, trust_scope = _RECORD_SPECS[
            command_type
        ]
        if (
            str(event["event_type"]) != event_type
            or int(event["event_schema_version"]) != 1
            or str(event["aggregate_type"]) != aggregate_type
            or str(event["aggregate_id"]) != aggregate_id
            or int(event["aggregate_version"])
            != int(row["authority_aggregate_version"])
            or int(row["authority_aggregate_version"]) != 1
            or str(event["recorded_at"]) != str(row["recorded_at"])
            or str(event["security_scope"])
            != "authority.source_registry"
            or str(event["retention_scope"]) != "authority.audit"
            or str(event["trust_scope"]) != trust_scope.value
            or str(event["payload_mode"]) != PayloadMode.INLINE.value
            or str(event["payload_digest"]) != canonical_digest
            or event["payload_bytes"] is None
            or bytes(event["payload_bytes"]) != canonical_bytes
            or digest_bytes(canonical_bytes) != canonical_digest
        ):
            raise AuthorityPersistenceError(
                "source record authority envelope is inconsistent"
            )
        return event

    @staticmethod
    def _json_blob(value: object) -> bytes:
        return canonical_json_bytes(value)

    @staticmethod
    def _digest_value(value: object) -> str:
        return digest_canonical(value)

    @staticmethod
    def _recorded_at(value: str) -> UtcTimestamp:
        return UtcTimestamp.parse(value)


__all__ = ["_SourceRegistryStoreSupport"]
