from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from typing import Any

from newsroom.authority.canonical import canonical_json_bytes, digest_canonical
from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.authority.types import EventId, UtcTimestamp
from newsroom.sources.policy import (
    DISCOVERY_OCCURRENCE_RECORD_COMMAND,
    DISCOVERY_REPRESENTATION_RECORD_COMMAND,
    SOURCE_DEFINITION_REGISTER_COMMAND,
    SOURCE_DEFINITION_VERSION_RECORD_COMMAND,
    SOURCE_ITEM_REGISTER_COMMAND,
    SOURCE_LOCATOR_CONTINUITY_DECIDE_COMMAND,
    SOURCE_REVISION_RECORD_COMMAND,
)
from newsroom.sources.record_models import (
    DiscoveryOccurrence,
    DiscoveryRepresentation,
    LocatorContinuityDecision,
    SourceDefinition,
    SourceDefinitionVersion,
    SourceDefinitionVersionSummary,
    SourceItem,
    SourceRevision,
)
from newsroom.sources.types import (
    DiscoveryRepresentationId,
    ObservationModel,
    PortfolioFunction,
    SourceDefinitionId,
    SourceDefinitionVersionId,
    SourceItemId,
    SourceLifecycleStage,
    SourceRevisionId,
    SourceRole,
)

from ._source_registry_decoding import (
    canonical_row_value,
    decode_locator_continuity,
    decode_occurrence,
    decode_representation,
    decode_source_definition,
    decode_source_definition_version,
    decode_source_item,
    decode_source_revision,
)


class _SourceRegistryReadMixin:
    @staticmethod
    def _require_normalized_columns(
        row: Mapping[str, Any],
        expected: Mapping[str, object],
        *,
        identity: str,
    ) -> None:
        for column, value in expected.items():
            if row[column] != value:
                raise AuthorityPersistenceError(
                    f"{identity} column {column} differs from canonical bytes"
                )

    @staticmethod
    def _require_canonical_blob(
        row: Mapping[str, Any],
        column: str,
        value: object,
        *,
        identity: str,
    ) -> None:
        if row[column] is None or bytes(row[column]) != canonical_json_bytes(value):
            raise AuthorityPersistenceError(
                f"{identity} column {column} differs from canonical bytes"
            )

    @staticmethod
    def _row_by_id(
        conn: sqlite3.Connection,
        *,
        table: str,
        column: str,
        identifier: str,
    ) -> sqlite3.Row | None:
        return conn.execute(
            f"SELECT * FROM {table} WHERE {column}=?", (identifier,)
        ).fetchone()

    @classmethod
    def _required_row_by_id(
        cls,
        conn: sqlite3.Connection,
        *,
        table: str,
        column: str,
        identifier: str,
        identity: str,
    ) -> sqlite3.Row:
        row = cls._row_by_id(
            conn,
            table=table,
            column=column,
            identifier=identifier,
        )
        if row is None:
            raise AuthorityPersistenceError(f"{identity} is not retained")
        return row

    @classmethod
    def _require_historically_current_version(
        cls,
        conn: sqlite3.Connection,
        row: Mapping[str, Any],
        *,
        definition_id: str,
        version_id: str,
        identity: str,
    ) -> None:
        if not cls._source_version_was_current_at_event(
            conn,
            definition_id=definition_id,
            version_id=version_id,
            record_event_id=str(row["authority_event_id"]),
        ):
            raise AuthorityPersistenceError(
                f"{identity} source version was not current when recorded"
            )

    @classmethod
    def _item_row(
        cls, conn: sqlite3.Connection, item_id: str
    ) -> sqlite3.Row:
        return cls._required_row_by_id(
            conn,
            table="source_items",
            column="item_id",
            identifier=item_id,
            identity="source item",
        )

    @classmethod
    def _revision_row(
        cls, conn: sqlite3.Connection, revision_id: str
    ) -> sqlite3.Row:
        return cls._required_row_by_id(
            conn,
            table="source_revisions",
            column="revision_id",
            identifier=revision_id,
            identity="source revision",
        )

    @classmethod
    def _representation_row(
        cls, conn: sqlite3.Connection, representation_id: str
    ) -> sqlite3.Row:
        return cls._required_row_by_id(
            conn,
            table="discovery_representations",
            column="representation_id",
            identifier=representation_id,
            identity="discovery representation",
        )

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
            raise AuthorityPersistenceError(
                f"committed {identity} row is missing"
            )
        return row

    def _source_definition_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> SourceDefinition:
        self._prove_current_record(row)
        value = canonical_row_value(row, identity="source definition")
        event = self._validate_record_envelope(
            conn,
            row,
            command_type=SOURCE_DEFINITION_REGISTER_COMMAND,
            aggregate_id=str(row["definition_id"]),
            canonical_bytes=bytes(row["canonical_bytes"]),
            canonical_digest=str(row["canonical_digest"]),
        )
        request = decode_source_definition(
            value, idempotency_key=str(event["idempotency_key"])
        )
        if (
            str(request.definition_id) != str(row["definition_id"])
            or request.name != str(row["name"])
            or request.editorial_purpose != str(row["editorial_purpose"])
        ):
            raise AuthorityPersistenceError(
                "source definition normalized columns differ from canonical bytes"
            )
        return SourceDefinition(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    def _source_version_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> SourceDefinitionVersion:
        self._prove_current_record(row)
        value = canonical_row_value(row, identity="source definition version")
        event = self._validate_record_envelope(
            conn,
            row,
            command_type=SOURCE_DEFINITION_VERSION_RECORD_COMMAND,
            aggregate_id=str(row["version_id"]),
            canonical_bytes=bytes(row["canonical_bytes"]),
            canonical_digest=str(row["canonical_digest"]),
        )
        request = decode_source_definition_version(
            value, idempotency_key=str(event["idempotency_key"])
        )
        if (
            str(request.version_id) != str(row["version_id"])
            or str(request.definition_id) != str(row["definition_id"])
            or request.version_number != int(row["version_number"])
            or request.locator != str(row["locator"])
            or request.locator_digest != str(row["locator_digest"])
            or request.semantic_digest != str(row["semantic_digest"])
        ):
            raise AuthorityPersistenceError(
                "source version normalized columns differ from canonical bytes"
            )
        self._validate_source_version_children(conn, row=row, request=request)
        return SourceDefinitionVersion(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    def _source_item_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> SourceItem:
        self._prove_current_record(row)
        value = canonical_row_value(row, identity="source item")
        event = self._validate_record_envelope(
            conn,
            row,
            command_type=SOURCE_ITEM_REGISTER_COMMAND,
            aggregate_id=str(row["item_id"]),
            canonical_bytes=bytes(row["canonical_bytes"]),
            canonical_digest=str(row["canonical_digest"]),
        )
        request = decode_source_item(
            value, idempotency_key=str(event["idempotency_key"])
        )
        self._require_normalized_columns(
            row,
            {
                "definition_id": str(request.definition_id),
                "definition_version_id": str(request.definition_version_id),
                "identity_kind": request.identity_kind.value,
                "identity_policy_id": request.identity_policy.policy_id,
                "identity_policy_version": request.identity_policy.policy_version,
                "source_native_id": request.source_native_id,
                "identity_digest": request.identity_digest,
            },
            identity="source item",
        )
        self._require_canonical_blob(
            row,
            "identity_components_bytes",
            [item.canonical_value() for item in request.identity_components],
            identity="source item",
        )
        self._require_canonical_blob(
            row,
            "uncertainties_bytes",
            list(request.uncertainties),
            identity="source item",
        )
        self._require_historically_current_version(
            conn,
            row,
            definition_id=str(request.definition_id),
            version_id=str(request.definition_version_id),
            identity="source item",
        )
        if (
            str(request.item_id) != str(row["item_id"])
            or str(request.definition_id) != str(row["definition_id"])
            or str(request.definition_version_id)
            != str(row["definition_version_id"])
            or request.identity_digest != str(row["identity_digest"])
        ):
            raise AuthorityPersistenceError(
                "source item normalized columns differ from canonical bytes"
            )
        return SourceItem(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    def _locator_decision_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> LocatorContinuityDecision:
        self._prove_current_record(row)
        value = canonical_row_value(row, identity="locator continuity decision")
        event = self._validate_record_envelope(
            conn,
            row,
            command_type=SOURCE_LOCATOR_CONTINUITY_DECIDE_COMMAND,
            aggregate_id=str(row["decision_id"]),
            canonical_bytes=bytes(row["canonical_bytes"]),
            canonical_digest=str(row["canonical_digest"]),
        )
        request = decode_locator_continuity(
            value, idempotency_key=str(event["idempotency_key"])
        )
        prior_locator_digest = digest_canonical(
            {
                "definition_id": str(request.definition_id),
                "locator": request.prior_locator,
            }
        )
        observed_locator_digest = digest_canonical(
            {
                "definition_id": str(request.definition_id),
                "locator": request.observed_locator,
            }
        )
        self._require_normalized_columns(
            row,
            {
                "definition_id": str(request.definition_id),
                "definition_version_id": str(request.definition_version_id),
                "prior_item_id": str(request.prior_item_id),
                "prior_locator": request.prior_locator,
                "prior_locator_digest": prior_locator_digest,
                "observed_locator": request.observed_locator,
                "observed_locator_digest": observed_locator_digest,
                "outcome": request.outcome.value,
                "related_item_id": str(request.related_item_id),
                "rationale": request.rationale,
                "decision_policy_id": request.decision_policy.policy_id,
                "decision_policy_version": request.decision_policy.policy_version,
                "observed_at": request.observed_at.to_text(),
                "semantic_digest": request.semantic_digest,
            },
            identity="locator continuity decision",
        )
        self._require_historically_current_version(
            conn,
            row,
            definition_id=str(request.definition_id),
            version_id=str(request.definition_version_id),
            identity="locator continuity decision",
        )
        if (
            str(request.decision_id) != str(row["decision_id"])
            or request.semantic_digest != str(row["semantic_digest"])
        ):
            raise AuthorityPersistenceError(
                "locator decision columns differ from canonical bytes"
            )
        return LocatorContinuityDecision(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    def _source_revision_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> SourceRevision:
        self._prove_current_record(row)
        value = canonical_row_value(row, identity="source revision")
        event = self._validate_record_envelope(
            conn,
            row,
            command_type=SOURCE_REVISION_RECORD_COMMAND,
            aggregate_id=str(row["revision_id"]),
            canonical_bytes=bytes(row["canonical_bytes"]),
            canonical_digest=str(row["canonical_digest"]),
        )
        request = decode_source_revision(
            value, idempotency_key=str(event["idempotency_key"])
        )
        item = self._item_row(conn, str(request.item_id))
        self._require_normalized_columns(
            row,
            {
                "item_id": str(request.item_id),
                "definition_id": str(item["definition_id"]),
                "definition_version_id": str(request.definition_version_id),
                "prior_revision_id": (
                    None
                    if request.prior_revision_id is None
                    else str(request.prior_revision_id)
                ),
                "source_native_revision_token": request.source_native_revision_token,
                "permitted_state_digest": request.permitted_state_digest,
                "revision_policy_id": request.revision_policy.policy_id,
                "revision_policy_version": request.revision_policy.policy_version,
                "canonicalizer_version": request.canonicalizer_version,
                "observed_at": request.observed_at.to_text(),
                "revision_identity_digest": request.revision_identity_digest,
            },
            identity="source revision",
        )
        self._require_canonical_blob(
            row,
            "source_published_time_bytes",
            request.source_published_time.canonical_value(),
            identity="source revision",
        )
        self._require_canonical_blob(
            row,
            "source_updated_time_bytes",
            request.source_updated_time.canonical_value(),
            identity="source revision",
        )
        self._require_historically_current_version(
            conn,
            row,
            definition_id=str(item["definition_id"]),
            version_id=str(request.definition_version_id),
            identity="source revision",
        )
        if (
            str(request.revision_id) != str(row["revision_id"])
            or str(request.item_id) != str(row["item_id"])
            or request.revision_identity_digest
            != str(row["revision_identity_digest"])
        ):
            raise AuthorityPersistenceError(
                "source revision columns differ from canonical bytes"
            )
        return SourceRevision(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    def _representation_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> DiscoveryRepresentation:
        self._prove_current_record(row)
        value = canonical_row_value(row, identity="discovery representation")
        event = self._validate_record_envelope(
            conn,
            row,
            command_type=DISCOVERY_REPRESENTATION_RECORD_COMMAND,
            aggregate_id=str(row["representation_id"]),
            canonical_bytes=bytes(row["canonical_bytes"]),
            canonical_digest=str(row["canonical_digest"]),
        )
        request = decode_representation(
            value, idempotency_key=str(event["idempotency_key"])
        )
        revision = self._revision_row(conn, str(request.revision_id))
        if not self._source_version_is_same_or_later(
            conn,
            definition_id=str(revision["definition_id"]),
            lineage_version_id=str(revision["definition_version_id"]),
            observation_version_id=str(request.definition_version_id),
        ):
            raise AuthorityPersistenceError(
                "representation source version predates or differs from its revision"
            )
        self._require_historically_current_version(
            conn,
            row,
            definition_id=str(revision["definition_id"]),
            version_id=str(request.definition_version_id),
            identity="discovery representation",
        )
        if request.produced_at.to_text() < str(revision["observed_at"]):
            raise AuthorityPersistenceError(
                "representation precedes its Source Revision observation"
            )
        self._require_normalized_columns(
            row,
            {
                "revision_id": str(request.revision_id),
                "definition_id": str(revision["definition_id"]),
                "definition_version_id": str(request.definition_version_id),
                "adapter_version": request.adapter_version,
                "parser_version": request.parser_version,
                "normalizer_version": request.normalizer_version,
                "extraction_scope_version": request.extraction_scope_version,
                "permitted_fields_digest": request.permitted_fields_digest,
                "representation_digest": request.representation_digest,
                "producer_slot_digest": request.producer_slot_digest,
                "representation_identity_digest": request.representation_identity_digest,
                "produced_at": request.produced_at.to_text(),
            },
            identity="discovery representation",
        )
        if (
            str(request.representation_id) != str(row["representation_id"])
            or request.producer_slot_digest != str(row["producer_slot_digest"])
            or request.representation_identity_digest
            != str(row["representation_identity_digest"])
        ):
            raise AuthorityPersistenceError(
                "representation columns differ from canonical bytes"
            )
        return DiscoveryRepresentation(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    def _occurrence_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> DiscoveryOccurrence:
        self._prove_current_record(row)
        value = canonical_row_value(row, identity="discovery occurrence")
        event = self._validate_record_envelope(
            conn,
            row,
            command_type=DISCOVERY_OCCURRENCE_RECORD_COMMAND,
            aggregate_id=str(row["occurrence_id"]),
            canonical_bytes=bytes(row["canonical_bytes"]),
            canonical_digest=str(row["canonical_digest"]),
        )
        request = decode_occurrence(
            value, idempotency_key=str(event["idempotency_key"])
        )
        revision = self._revision_row(conn, str(request.revision_id))
        if not self._source_version_is_same_or_later(
            conn,
            definition_id=str(revision["definition_id"]),
            lineage_version_id=str(revision["definition_version_id"]),
            observation_version_id=str(request.definition_version_id),
        ):
            raise AuthorityPersistenceError(
                "occurrence source version predates or differs from its revision"
            )
        self._require_historically_current_version(
            conn,
            row,
            definition_id=str(revision["definition_id"]),
            version_id=str(request.definition_version_id),
            identity="discovery occurrence",
        )
        if request.observed_at.to_text() < str(revision["observed_at"]):
            raise AuthorityPersistenceError(
                "occurrence precedes its Source Revision observation"
            )
        if request.representation_id is not None:
            representation = self._representation_row(
                conn,
                str(request.representation_id),
            )
            if str(representation["revision_id"]) != str(request.revision_id):
                raise AuthorityPersistenceError(
                    "occurrence representation belongs to another revision"
                )
            if str(representation["definition_version_id"]) != str(
                request.definition_version_id
            ):
                raise AuthorityPersistenceError(
                    "occurrence source version differs from its representation"
                )
            if request.observed_at.to_text() < str(
                representation["produced_at"]
            ):
                raise AuthorityPersistenceError(
                    "occurrence precedes its Representation"
                )
        self._require_normalized_columns(
            row,
            {
                "check_outcome_id": str(request.check_outcome_id),
                "revision_id": str(request.revision_id),
                "representation_id": (
                    None
                    if request.representation_id is None
                    else str(request.representation_id)
                ),
                "definition_id": str(revision["definition_id"]),
                "definition_version_id": str(request.definition_version_id),
                "occurrence_kind": request.kind.value,
                "observed_at": request.observed_at.to_text(),
                "receipt_digest": request.receipt_digest,
                "semantic_digest": request.semantic_digest,
            },
            identity="discovery occurrence",
        )
        self._require_canonical_blob(
            row,
            "source_asserted_time_bytes",
            request.source_asserted_time.canonical_value(),
            identity="discovery occurrence",
        )
        if (
            str(request.occurrence_id) != str(row["occurrence_id"])
            or request.semantic_digest != str(row["semantic_digest"])
        ):
            raise AuthorityPersistenceError(
                "occurrence columns differ from canonical bytes"
            )
        return DiscoveryOccurrence(
            request=request,
            event_id=EventId.parse(str(row["authority_event_id"])),
            aggregate_version=int(row["authority_aggregate_version"]),
            recorded_at=UtcTimestamp.parse(str(row["recorded_at"])),
            canonical_digest=str(row["canonical_digest"]),
            replayed=replayed,
        )

    def _for_event(
        self,
        conn: sqlite3.Connection,
        event_id: str,
        *,
        table: str,
        identity: str,
        loader: Callable[..., Any],
        replayed: bool,
    ) -> Any:
        row = self._row_for_event(
            conn,
            table=table,
            event_id=event_id,
            identity=identity,
        )
        return loader(conn, row, replayed=replayed)

    def _source_definition_for_event(
        self, conn: sqlite3.Connection, event_id: str, *, replayed: bool
    ) -> SourceDefinition:
        return self._for_event(
            conn,
            event_id,
            table="source_definitions",
            identity="source definition",
            loader=self._source_definition_from_row,
            replayed=replayed,
        )

    def _source_version_for_event(
        self, conn: sqlite3.Connection, event_id: str, *, replayed: bool
    ) -> SourceDefinitionVersion:
        return self._for_event(
            conn,
            event_id,
            table="source_definition_versions",
            identity="source version",
            loader=self._source_version_from_row,
            replayed=replayed,
        )

    def _source_item_for_event(
        self, conn: sqlite3.Connection, event_id: str, *, replayed: bool
    ) -> SourceItem:
        return self._for_event(
            conn,
            event_id,
            table="source_items",
            identity="source item",
            loader=self._source_item_from_row,
            replayed=replayed,
        )

    def _locator_decision_for_event(
        self, conn: sqlite3.Connection, event_id: str, *, replayed: bool
    ) -> LocatorContinuityDecision:
        return self._for_event(
            conn,
            event_id,
            table="source_locator_continuity_decisions",
            identity="locator decision",
            loader=self._locator_decision_from_row,
            replayed=replayed,
        )

    def _source_revision_for_event(
        self, conn: sqlite3.Connection, event_id: str, *, replayed: bool
    ) -> SourceRevision:
        return self._for_event(
            conn,
            event_id,
            table="source_revisions",
            identity="source revision",
            loader=self._source_revision_from_row,
            replayed=replayed,
        )

    def _representation_for_event(
        self, conn: sqlite3.Connection, event_id: str, *, replayed: bool
    ) -> DiscoveryRepresentation:
        return self._for_event(
            conn,
            event_id,
            table="discovery_representations",
            identity="representation",
            loader=self._representation_from_row,
            replayed=replayed,
        )

    def _occurrence_for_event(
        self, conn: sqlite3.Connection, event_id: str, *, replayed: bool
    ) -> DiscoveryOccurrence:
        return self._for_event(
            conn,
            event_id,
            table="discovery_occurrences",
            identity="occurrence",
            loader=self._occurrence_from_row,
            replayed=replayed,
        )

    def source_definition(
        self, definition_id: SourceDefinitionId
    ) -> SourceDefinition | None:
        with self._lock:
            row = self._row_by_id(
                self._connection,
                table="source_definitions",
                column="definition_id",
                identifier=str(definition_id),
            )
            return (
                None
                if row is None
                else self._source_definition_from_row(
                    self._connection, row, replayed=False
                )
            )

    def source_definition_version(
        self, version_id: SourceDefinitionVersionId
    ) -> SourceDefinitionVersion | None:
        with self._lock:
            row = self._row_by_id(
                self._connection,
                table="source_definition_versions",
                column="version_id",
                identifier=str(version_id),
            )
            return (
                None
                if row is None
                else self._source_version_from_row(
                    self._connection, row, replayed=False
                )
            )

    def current_source_definition_version(
        self, definition_id: SourceDefinitionId
    ) -> SourceDefinitionVersion | None:
        with self._lock:
            row = self._current_version_row(self._connection, definition_id)
            return (
                None
                if row is None
                else self._source_version_from_row(
                    self._connection, row, replayed=False
                )
            )

    def current_source_definition_summary(
        self, definition_id: SourceDefinitionId
    ) -> SourceDefinitionVersionSummary | None:
        version = self.current_source_definition_version(definition_id)
        if version is None:
            return None
        request = version.request
        return SourceDefinitionVersionSummary(
            version_id=request.version_id,
            definition_id=request.definition_id,
            version_number=request.version_number,
            lifecycle_stage=request.lifecycle_stage,
            observation_model=request.observation_model,
            roles=tuple(item.role for item in request.roles),
            portfolio_functions=request.portfolio_functions,
            coverage_obligation_ids=tuple(
                sorted({item.obligation_id for item in request.coverage_mappings})
            ),
            explicit_gap_ids=tuple(
                sorted(item.gap_id for item in request.explicit_gaps)
            ),
            locator_digest=request.locator_digest,
            rights_decision_id=request.rights.rights_decision_id,
            execution_authority=request.execution_authority,
            recorded_at=version.recorded_at,
        )

    def source_item(self, item_id: SourceItemId) -> SourceItem | None:
        with self._lock:
            row = self._row_by_id(
                self._connection,
                table="source_items",
                column="item_id",
                identifier=str(item_id),
            )
            return (
                None
                if row is None
                else self._source_item_from_row(
                    self._connection, row, replayed=False
                )
            )

    def source_revision(
        self, revision_id: SourceRevisionId
    ) -> SourceRevision | None:
        with self._lock:
            row = self._row_by_id(
                self._connection,
                table="source_revisions",
                column="revision_id",
                identifier=str(revision_id),
            )
            return (
                None
                if row is None
                else self._source_revision_from_row(
                    self._connection, row, replayed=False
                )
            )

    def latest_source_revision(self, item_id: SourceItemId) -> SourceRevision | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT r.* FROM source_revisions r "
                "JOIN ledger_events e ON e.event_id=r.authority_event_id "
                "WHERE r.item_id=? ORDER BY e.ledger_seq DESC LIMIT 1",
                (str(item_id),),
            ).fetchone()
            return None if row is None else self._source_revision_from_row(
                self._connection, row, replayed=False
            )

    def discovery_representation(
        self, representation_id: DiscoveryRepresentationId
    ) -> DiscoveryRepresentation | None:
        with self._lock:
            row = self._row_by_id(
                self._connection,
                table="discovery_representations",
                column="representation_id",
                identifier=str(representation_id),
            )
            return (
                None
                if row is None
                else self._representation_from_row(
                    self._connection, row, replayed=False
                )
            )
    def occurrences_for_revision(
        self, revision_id: SourceRevisionId, *, limit: int
    ) -> tuple[DiscoveryOccurrence, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("occurrence read limit must be positive")
        with self._lock:
            rows = self._connection.execute(
                "SELECT o.* FROM discovery_occurrences o "
                "JOIN ledger_events e ON e.event_id=o.authority_event_id "
                "WHERE o.revision_id=? ORDER BY e.ledger_seq LIMIT ?",
                (str(revision_id), limit),
            ).fetchall()
            return tuple(
                self._occurrence_from_row(
                    self._connection, row, replayed=False
                )
                for row in rows
            )


__all__ = ["_SourceRegistryReadMixin"]
