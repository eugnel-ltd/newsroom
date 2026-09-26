"""Admitted-only GraphRAG context for governed editorial consumers.

The hydrator reads durable authority and projection receipts, never the private
Graphiti workspace.  Downstream consumers receive a bounded canonical value
with explicit authority, provenance, temporal, currency and trust bindings.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane.graphiti_admission import (
    GraphitiAdmissionConsumerError,
    GraphitiAdmissionRequest,
    GraphitiAdmissionTelemetry,
    GraphitiGovernedDecision,
    GraphitiProjectionReconciliationReceipt,
    GraphitiRightsAuthority,
    graphiti_admission_request_from_value,
    graphiti_admission_telemetry,
    graphiti_decided_cohort_generation_identity,
    graphiti_governed_decision_from_json,
    graphiti_projection_receipt_from_json,
    graphiti_projection_receipt_matches_binding,
    graphiti_projection_reconciliation_from_json,
)
from newsroom.extraction.types import ExtractionProposalKind
from newsroom.graphiti_adapter.admission import GraphitiProposalAdmissionAction
from newsroom.projection.models import ProjectionGenerationId

ADMITTED_CONTEXT_SCHEMA_VERSION = "newsroom.admitted-graphrag-context.v1"
ADMITTED_CONTEXT_TRUST_LABEL = "ADMITTED_GOVERNED_AUTHORITY_CONTEXT"
_CURRENT = "CURRENT"


class GovernedContextStatus(StrEnum):
    READY = "READY"
    EMPTY = "EMPTY"
    HOLD = "HOLD"


@dataclass(frozen=True, slots=True)
class AuthorityContextBinding:
    authority_kind: str
    authority_id: str
    authority_version: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.authority_kind,
                self.authority_id,
                self.authority_version,
            )
        ):
            raise ValueError("governed context authority binding is incomplete")

    def canonical_value(self) -> dict[str, str]:
        return {
            "authority_kind": self.authority_kind,
            "authority_id": self.authority_id,
            "authority_version": self.authority_version,
        }


@dataclass(frozen=True, slots=True)
class GovernedAuthorityContext:
    """Current admitted authority returned by an injected authority read port."""

    bindings: tuple[AuthorityContextBinding, ...]
    admitted_temporal_fields: tuple[tuple[str, str | None], ...]
    currentness_state: str
    admitted_structured_value_json: str

    def __post_init__(self) -> None:
        if not self.bindings or self.bindings != tuple(
            sorted(
                set(self.bindings),
                key=lambda item: (
                    item.authority_kind,
                    item.authority_id,
                    item.authority_version,
                ),
            )
        ):
            raise ValueError("governed context authority bindings must be canonical")
        if self.admitted_temporal_fields != tuple(
            sorted(self.admitted_temporal_fields)
        ) or len({name for name, _value in self.admitted_temporal_fields}) != len(
            self.admitted_temporal_fields
        ):
            raise ValueError("admitted temporal fields must be canonical")
        if not self.currentness_state:
            raise ValueError("governed context currentness is required")
        structured = json.loads(self.admitted_structured_value_json)
        if (
            not isinstance(structured, dict)
            or canonical_json_bytes(structured).decode()
            != self.admitted_structured_value_json
        ):
            raise ValueError("admitted structured context must be a canonical object")

    @property
    def admitted_structured_value(self) -> Mapping[str, object]:
        value = json.loads(self.admitted_structured_value_json)
        if not isinstance(value, dict):  # pragma: no cover - guarded at construction
            raise TypeError("admitted structured context must be an object")
        return value


class GovernedContextAuthority(Protocol):
    def current_context(
        self,
        request: GraphitiAdmissionRequest,
        decision: GraphitiGovernedDecision,
    ) -> GovernedAuthorityContext | None: ...


@dataclass(frozen=True, slots=True)
class EvidencePassageLineage:
    passage_id: str
    admission_id: str
    access_decision_id: str
    byte_offset: int
    byte_length: int
    blob_digest: str
    text_digest: str

    @classmethod
    def from_value(cls, value: Mapping[str, object]) -> EvidencePassageLineage:
        fields = {
            "passage_id",
            "admission_id",
            "access_decision_id",
            "byte_offset",
            "byte_length",
            "blob_digest",
            "text_digest",
        }
        if set(value) != fields:
            raise ValueError("governed context passage lineage fields differ")
        offset = value["byte_offset"]
        length = value["byte_length"]
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
        ):
            raise ValueError("governed context passage range is invalid")
        return cls(
            passage_id=str(value["passage_id"]),
            admission_id=str(value["admission_id"]),
            access_decision_id=str(value["access_decision_id"]),
            byte_offset=offset,
            byte_length=length,
            blob_digest=str(value["blob_digest"]),
            text_digest=str(value["text_digest"]),
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "passage_id": self.passage_id,
            "admission_id": self.admission_id,
            "access_decision_id": self.access_decision_id,
            "byte_offset": self.byte_offset,
            "byte_length": self.byte_length,
            "blob_digest": self.blob_digest,
            "text_digest": self.text_digest,
        }


@dataclass(frozen=True, slots=True)
class GovernedContextItem:
    proposal_key: str
    authority_bindings: tuple[AuthorityContextBinding, ...]
    admission_decision_id: str
    admission_authority_version: int
    source_id: str
    item_key: str
    source_revision_id: str
    source_receipt_digest: str
    evidence_passages: tuple[EvidencePassageLineage, ...]
    proposed_temporal_fields: tuple[tuple[str, str | None], ...]
    admitted_temporal_fields: tuple[tuple[str, str | None], ...]
    admitted_structured_value_json: str
    currentness_state: str
    projection_effect_id: str
    projection_generation_id: str
    projection_authority_watermark: int
    contiguous_projection_watermark: int
    projection_gap_count: int
    oldest_lag_seconds: int
    stale: bool
    degraded: bool
    admission_current: bool
    rights_current: bool
    currency_read_at: str
    trust_label: str = ADMITTED_CONTEXT_TRUST_LABEL

    def __post_init__(self) -> None:
        if not self.evidence_passages:
            raise ValueError("governed context item requires exact evidence lineage")
        if self.currentness_state != _CURRENT:
            raise ValueError("only current admitted authority may enter context")
        if self.trust_label != ADMITTED_CONTEXT_TRUST_LABEL:
            raise ValueError("governed context trust label differs")
        if self.proposed_temporal_fields != tuple(
            sorted(self.proposed_temporal_fields)
        ) or self.admitted_temporal_fields != tuple(
            sorted(self.admitted_temporal_fields)
        ):
            raise ValueError("governed context temporal fields must be canonical")
        structured = json.loads(self.admitted_structured_value_json)
        if (
            not isinstance(structured, dict)
            or canonical_json_bytes(structured).decode()
            != self.admitted_structured_value_json
        ):
            raise ValueError("admitted structured context must be a canonical object")
        if (
            self.admission_authority_version <= 0
            or self.projection_authority_watermark <= 0
        ):
            raise ValueError("governed context authority versions must be positive")
        if (
            self.contiguous_projection_watermark <= 0
            or self.projection_gap_count != 0
            or self.oldest_lag_seconds < 0
            or self.stale
            or self.degraded
            or not self.admission_current
            or not self.rights_current
            or not self.currency_read_at
        ):
            raise ValueError("governed context item is not current and gap-free")

    def canonical_value(self) -> dict[str, object]:
        return {
            "proposal_key": self.proposal_key,
            "authority_bindings": [
                item.canonical_value() for item in self.authority_bindings
            ],
            "admission_decision_id": self.admission_decision_id,
            "admission_authority_version": self.admission_authority_version,
            "source_id": self.source_id,
            "item_key": self.item_key,
            "source_revision_id": self.source_revision_id,
            "source_receipt_digest": self.source_receipt_digest,
            "evidence_passages": [
                item.canonical_value() for item in self.evidence_passages
            ],
            "proposed_temporal_fields": [
                [name, value] for name, value in self.proposed_temporal_fields
            ],
            "admitted_temporal_fields": [
                [name, value] for name, value in self.admitted_temporal_fields
            ],
            "admitted_structured_value": json.loads(
                self.admitted_structured_value_json
            ),
            "currentness_state": self.currentness_state,
            "projection_effect_id": self.projection_effect_id,
            "projection_generation_id": self.projection_generation_id,
            "projection_authority_watermark": self.projection_authority_watermark,
            "contiguous_projection_watermark": self.contiguous_projection_watermark,
            "projection_gap_count": self.projection_gap_count,
            "oldest_lag_seconds": self.oldest_lag_seconds,
            "stale": self.stale,
            "degraded": self.degraded,
            "admission_current": self.admission_current,
            "rights_current": self.rights_current,
            "currency_read_at": self.currency_read_at,
            "trust_label": self.trust_label,
        }


def _measure_context(value: Mapping[str, object]) -> tuple[int, int]:
    """Measure the full canonical envelope with a conservative token ceiling."""

    size = 0
    tokens = 0
    while True:
        measured = {
            **value,
            "context_bytes": size,
            "token_contribution": tokens,
        }
        next_size = len(canonical_json_bytes(measured))
        next_tokens = next_size
        if (next_size, next_tokens) == (size, tokens):
            return size, tokens
        size, tokens = next_size, next_tokens


@dataclass(frozen=True, slots=True)
class GovernedContext:
    status: GovernedContextStatus
    reason_code: str
    projection_generation_id: str | None
    contiguous_projection_watermark: int | None
    projection_gap_count: int
    oldest_lag_seconds: int
    stale: bool
    degraded: bool
    items: tuple[GovernedContextItem, ...]
    context_bytes: int
    token_contribution: int
    read_at: str
    schema_version: str = ADMITTED_CONTEXT_SCHEMA_VERSION
    trust_label: str = ADMITTED_CONTEXT_TRUST_LABEL
    provider_model_calls: int = 0
    authority_effect: str = "NONE"

    def __post_init__(self) -> None:
        if self.status is GovernedContextStatus.READY and not self.items:
            raise ValueError("ready governed context must contain admitted items")
        if self.status is not GovernedContextStatus.READY and self.items:
            raise ValueError("non-ready governed context cannot expose items")
        if self.status is GovernedContextStatus.HOLD and not self.degraded:
            raise ValueError("held governed context must be explicitly degraded")
        if self.status is not GovernedContextStatus.HOLD and (
            self.stale or self.degraded or self.projection_gap_count != 0
        ):
            raise ValueError(
                "ready or empty governed context must be current and gap-free"
            )
        if not self.currency_consistent:
            raise ValueError(
                "ready governed context currency differs from its admitted items"
            )
        measured_bytes, measured_tokens = _measure_context(
            self._canonical_value_without_measurement()
        )
        if (self.context_bytes, self.token_contribution) != (
            measured_bytes,
            measured_tokens,
        ):
            raise ValueError("governed context size measurement differs")
        if self.trust_label != ADMITTED_CONTEXT_TRUST_LABEL:
            raise ValueError("governed context trust label differs")
        if self.provider_model_calls != 0 or self.authority_effect != "NONE":
            raise ValueError(
                "governed context hydration cannot create authority effects"
            )

    @property
    def digest(self) -> str:
        return digest_bytes(canonical_json_bytes(self.canonical_value()))

    @property
    def currency_consistent(self) -> bool:
        if self.status is not GovernedContextStatus.READY:
            return True
        try:
            ProjectionGenerationId.parse(str(self.projection_generation_id))
            for item in self.items:
                ProjectionGenerationId.parse(item.projection_generation_id)
        except (TypeError, ValueError):
            return False
        return bool(
            self.projection_generation_id
            and self.contiguous_projection_watermark is not None
            and self.contiguous_projection_watermark > 0
            and all(
                item.projection_generation_id
                and item.contiguous_projection_watermark
                == self.contiguous_projection_watermark
                and item.projection_gap_count == self.projection_gap_count
                and item.oldest_lag_seconds == self.oldest_lag_seconds
                and item.stale == self.stale
                and item.degraded == self.degraded
                and item.currency_read_at == self.read_at
                and item.admission_authority_version
                <= item.projection_authority_watermark
                and 0
                < item.projection_authority_watermark
                <= self.contiguous_projection_watermark
                for item in self.items
            )
        )

    def _canonical_value_without_measurement(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "projection_generation_id": self.projection_generation_id,
            "contiguous_projection_watermark": self.contiguous_projection_watermark,
            "projection_gap_count": self.projection_gap_count,
            "oldest_lag_seconds": self.oldest_lag_seconds,
            "stale": self.stale,
            "degraded": self.degraded,
            "items": [item.canonical_value() for item in self.items],
            "read_at": self.read_at,
            "trust_label": self.trust_label,
            "provider_model_calls": self.provider_model_calls,
            "authority_effect": self.authority_effect,
        }

    def canonical_value(self) -> dict[str, object]:
        return {
            **self._canonical_value_without_measurement(),
            "context_bytes": self.context_bytes,
            "token_contribution": self.token_contribution,
        }

    def scoped_to(self, source_items: frozenset[tuple[str, str]]) -> GovernedContext:
        if self.status is GovernedContextStatus.HOLD:
            return self
        items = tuple(
            item
            for item in self.items
            if (item.source_id, item.item_key) in source_items
        )
        return GovernedContext.build(
            status=(
                GovernedContextStatus.READY if items else GovernedContextStatus.EMPTY
            ),
            reason_code="ADMITTED_CONTEXT_READY" if items else "ZERO_ADMITTED_CONTEXT",
            projection_generation_id=self.projection_generation_id,
            contiguous_projection_watermark=self.contiguous_projection_watermark,
            projection_gap_count=self.projection_gap_count,
            oldest_lag_seconds=self.oldest_lag_seconds,
            stale=self.stale,
            degraded=self.degraded,
            items=items,
            read_at=self.read_at,
        )

    @classmethod
    def build(
        cls,
        *,
        status: GovernedContextStatus,
        reason_code: str,
        projection_generation_id: str | None,
        contiguous_projection_watermark: int | None,
        projection_gap_count: int,
        oldest_lag_seconds: int,
        stale: bool,
        degraded: bool,
        items: tuple[GovernedContextItem, ...],
        read_at: str,
        schema_version: str = ADMITTED_CONTEXT_SCHEMA_VERSION,
        trust_label: str = ADMITTED_CONTEXT_TRUST_LABEL,
        provider_model_calls: int = 0,
        authority_effect: str = "NONE",
    ) -> GovernedContext:
        unmeasured = {
            "schema_version": schema_version,
            "status": status.value,
            "reason_code": reason_code,
            "projection_generation_id": projection_generation_id,
            "contiguous_projection_watermark": contiguous_projection_watermark,
            "projection_gap_count": projection_gap_count,
            "oldest_lag_seconds": oldest_lag_seconds,
            "stale": stale,
            "degraded": degraded,
            "items": [item.canonical_value() for item in items],
            "read_at": read_at,
            "trust_label": trust_label,
            "provider_model_calls": provider_model_calls,
            "authority_effect": authority_effect,
        }
        size, tokens = _measure_context(unmeasured)
        return cls(
            status=status,
            reason_code=reason_code,
            projection_generation_id=projection_generation_id,
            contiguous_projection_watermark=contiguous_projection_watermark,
            projection_gap_count=projection_gap_count,
            oldest_lag_seconds=oldest_lag_seconds,
            stale=stale,
            degraded=degraded,
            items=items,
            context_bytes=size,
            token_contribution=tokens,
            read_at=read_at,
            schema_version=schema_version,
            trust_label=trust_label,
            provider_model_calls=provider_model_calls,
            authority_effect=authority_effect,
        )


class GovernedContextHydrator:
    """Read admitted authority receipts into a bounded editorial context."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        authority: GovernedContextAuthority,
        rights: GraphitiRightsAuthority,
        clock: Callable[[], datetime] | None = None,
        max_items: int = 32,
        max_context_bytes: int = 64 * 1024,
        max_token_contribution: int = 16_384,
        max_oldest_lag_seconds: int = 3600,
    ) -> None:
        if (
            min(
                max_items,
                max_context_bytes,
                max_token_contribution,
                max_oldest_lag_seconds,
            )
            <= 0
        ):
            raise ValueError("governed context bounds must be positive")
        self._connection = connection
        self._authority = authority
        self._rights = rights
        self._clock = clock or (lambda: datetime.now(tz=UTC))
        self._max_items = max_items
        self._max_context_bytes = max_context_bytes
        self._max_token_contribution = max_token_contribution
        self._max_oldest_lag_seconds = max_oldest_lag_seconds

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("governed context clock must be timezone-aware")
        return value.astimezone(UTC)

    def _result(
        self,
        *,
        status: GovernedContextStatus,
        reason_code: str,
        items: tuple[GovernedContextItem, ...] = (),
        generation_id: str | None = None,
        watermark: int | None = None,
        gap_count: int = 0,
        oldest_lag_seconds: int = 0,
        stale: bool = False,
        read_at: datetime | None = None,
    ) -> GovernedContext:
        return GovernedContext.build(
            status=status,
            reason_code=reason_code,
            projection_generation_id=generation_id,
            contiguous_projection_watermark=watermark,
            projection_gap_count=gap_count,
            oldest_lag_seconds=oldest_lag_seconds,
            stale=stale,
            degraded=status is GovernedContextStatus.HOLD,
            items=items,
            read_at=(read_at or self._now()).isoformat().replace("+00:00", "Z"),
        )

    def _hold(
        self,
        reason_code: str,
        telemetry: GraphitiAdmissionTelemetry,
        *,
        read_at: datetime,
    ) -> GovernedContext:
        return self._result(
            status=GovernedContextStatus.HOLD,
            reason_code=reason_code,
            watermark=telemetry.contiguous_projection_watermark,
            gap_count=telemetry.projection_gap_count,
            oldest_lag_seconds=telemetry.oldest_lag_seconds,
            stale=telemetry.oldest_lag_seconds > self._max_oldest_lag_seconds,
            read_at=read_at,
        )

    def hydrate(
        self,
        source_items: frozenset[tuple[str, str]] | None = None,
    ) -> GovernedContext:
        """Hydrate one scope from a single SQLite read snapshot."""

        self._connection.execute("SAVEPOINT governed_context_hydration")
        try:
            result = self._hydrate_snapshot(source_items)
        except BaseException:
            self._connection.execute("ROLLBACK TO governed_context_hydration")
            self._connection.execute("RELEASE governed_context_hydration")
            raise
        self._connection.execute("RELEASE governed_context_hydration")
        return result

    def _exceeds_size_bound(self, context: GovernedContext) -> bool:
        return (
            context.context_bytes > self._max_context_bytes
            or context.token_contribution > self._max_token_contribution
        )

    def _active_projection_snapshot(self) -> tuple[tuple[str, int, str], ...]:
        snapshot: list[tuple[str, int, str]] = []
        for effect_id, watermark, generation_id in self._connection.execute(
            """
            SELECT projection.effect_id, projection.authority_watermark,
                   projection.generation_id
            FROM unpublished_graphiti_projection_receipts AS projection
            JOIN unpublished_graphiti_admission_queue AS queue
              USING(proposal_key)
            JOIN unpublished_graphiti_admission_decisions AS decision
              USING(proposal_key)
            LEFT JOIN unpublished_graphiti_projection_tombstones AS tombstone
              USING(proposal_key)
            WHERE queue.state='PROJECTED' AND decision.action='ADMIT'
              AND tombstone.proposal_key IS NULL
            ORDER BY projection.effect_id
            """
        ):
            if isinstance(watermark, bool) or not isinstance(watermark, int):
                raise TypeError("projection snapshot watermark must be an integer")
            snapshot.append((str(effect_id), watermark, str(generation_id)))
        return tuple(snapshot)

    def _tombstone_projection_watermark(self, generation_id: str) -> int | None:
        row = self._connection.execute(
            """
            SELECT MAX(tombstone.authority_watermark)
            FROM unpublished_graphiti_projection_tombstones AS tombstone
            JOIN unpublished_graphiti_projection_receipts AS projection
              USING(proposal_key)
            WHERE projection.generation_id=?
            """,
            (generation_id,),
        ).fetchone()
        watermark = None if row is None else row[0]
        if watermark is None:
            return None
        if isinstance(watermark, bool) or not isinstance(watermark, int):
            raise TypeError("projection tombstone watermark must be an integer")
        return watermark

    def _validate_exact_projection_generations(
        self,
        *,
        reconciliation_rows: tuple[tuple[object, ...], ...],
        active_projection_snapshot: tuple[tuple[str, int, str], ...],
        contiguous_watermark: int | None,
    ) -> str:
        """Validate the latest generation and every active receipt's cohort."""

        parsed: list[
            tuple[
                GraphitiProjectionReconciliationReceipt,
                dict[str, object] | None,
            ]
        ] = []
        for row in reconciliation_rows:
            receipt_text = str(row[0])
            receipt, binding = graphiti_projection_reconciliation_from_json(
                receipt_text
            )
            if (
                receipt.receipt_digest != str(row[1])
                or receipt.projector_family_id != str(row[2])
                or receipt.generation_id != str(row[3])
                or receipt.authority_watermark != int(row[4])
            ):
                raise GraphitiAdmissionConsumerError(
                    "projection reconciliation SQL identity differs"
                )
            parsed.append((receipt, binding))

        latest_receipt, latest_binding = parsed[0]
        if latest_binding is None:
            raise GraphitiAdmissionConsumerError(
                "latest projection generation lacks exact cohort authority"
            )
        latest_generation_id = str(latest_receipt.generation_id)
        if latest_receipt.authority_watermark != contiguous_watermark:
            raise GraphitiAdmissionConsumerError(
                "latest projection watermark differs from contiguous authority"
            )

        bound_ingest_ids: set[str] = set()
        reconciled_generation_ids: set[str] = set()
        for reconciliation, binding in parsed:
            if binding is None:
                raise GraphitiAdmissionConsumerError(
                    "active projection lacks exact cohort reconciliation"
                )
            generation_id = reconciliation.generation_id
            reconciled_generation_ids.add(generation_id)
            ingest_ids = tuple(str(item) for item in binding["ingest_ids"])
            cohort_digest, _rebuilt_generation_id = (
                graphiti_decided_cohort_generation_identity(
                    self._connection,
                    ingest_ids=ingest_ids,
                    require_terminal_states=True,
                )
            )
            if (
                cohort_digest != str(binding["cohort_digest"])
                or bound_ingest_ids.intersection(ingest_ids)
            ):
                raise GraphitiAdmissionConsumerError(
                    "exact projection cohort authority differs"
                )
            bound_ingest_ids.update(ingest_ids)

            placeholders = ",".join("?" for _ in ingest_ids)
            projection_rows = self._connection.execute(
                f"""
                SELECT projection.receipt_json, projection.receipt_digest,
                       projection.effect_id, projection.authority_watermark,
                       projection.generation_id, queue.ingest_id
                FROM unpublished_graphiti_projection_receipts AS projection
                JOIN unpublished_graphiti_admission_queue AS queue
                  USING(proposal_key)
                WHERE queue.ingest_id IN ({placeholders})
                ORDER BY projection.effect_id
                """,
                ingest_ids,
            ).fetchall()
            effect_ids: list[str] = []
            for projection_row in projection_rows:
                projection = graphiti_projection_receipt_from_json(
                    str(projection_row[0])
                )
                projection_material = projection.canonical_value()
                projection_material.pop("receipt_digest")
                if (
                    projection.receipt_digest != str(projection_row[1])
                    or digest_bytes(canonical_json_bytes(projection_material))
                    != projection.receipt_digest
                    or projection.effect_id != str(projection_row[2])
                    or projection.authority_watermark != int(projection_row[3])
                    or projection.generation_id != str(projection_row[4])
                    or str(projection_row[5]) not in ingest_ids
                    or projection.generation_id != generation_id
                    or not graphiti_projection_receipt_matches_binding(
                        projection, binding
                    )
                    or projection.authority_watermark
                    != reconciliation.authority_watermark
                ):
                    raise GraphitiAdmissionConsumerError(
                        "exact projection receipt authority differs"
                    )
                effect_ids.append(projection.effect_id)
            exact_effect_ids = tuple(effect_ids)
            if (
                reconciliation.expected_effect_ids != exact_effect_ids
                or reconciliation.actual_effect_ids != exact_effect_ids
            ):
                raise GraphitiAdmissionConsumerError(
                    "exact projection cohort effects differ"
                )

            decision_watermark = self._connection.execute(
                f"""
                SELECT MAX(decision.authority_ledger_seq)
                FROM unpublished_graphiti_admission_queue AS queue
                JOIN unpublished_graphiti_admission_decisions AS decision
                  USING(proposal_key)
                WHERE queue.ingest_id IN ({placeholders})
                """,
                ingest_ids,
            ).fetchone()[0]
            if (
                decision_watermark is None
                or reconciliation.authority_watermark < int(decision_watermark)
            ):
                raise GraphitiAdmissionConsumerError(
                    "exact projection cohort watermark differs"
                )
        required_generation_ids = {
            latest_generation_id,
            *(
                generation_id
                for _effect_id, _watermark, generation_id
                in active_projection_snapshot
            ),
        }
        if not required_generation_ids.issubset(reconciled_generation_ids):
            raise GraphitiAdmissionConsumerError(
                "active projection lacks exact cohort reconciliation"
            )
        return latest_generation_id

    def _empty_or_size_hold(
        self,
        *,
        telemetry: GraphitiAdmissionTelemetry,
        generation_id: str | None,
        read_at: datetime,
    ) -> GovernedContext:
        empty = self._result(
            status=GovernedContextStatus.EMPTY,
            reason_code="ZERO_ADMITTED_CONTEXT",
            generation_id=generation_id,
            watermark=telemetry.contiguous_projection_watermark,
            oldest_lag_seconds=telemetry.oldest_lag_seconds,
            read_at=read_at,
        )
        if self._exceeds_size_bound(empty):
            return self._hold(
                "ADMITTED_CONTEXT_SIZE_BOUND_EXCEEDED",
                telemetry,
                read_at=read_at,
            )
        return empty

    def _hydrate_snapshot(
        self,
        source_items: frozenset[tuple[str, str]] | None,
    ) -> GovernedContext:
        read_at = self._now()
        telemetry = graphiti_admission_telemetry(
            self._connection,
            now=read_at,
        )
        if telemetry.dead_letter_count or telemetry.integrity_hold_receipt_count:
            return self._hold("ADMISSION_INTEGRITY_GAP", telemetry, read_at=read_at)
        if telemetry.oldest_lag_seconds > self._max_oldest_lag_seconds:
            return self._hold("ADMITTED_CONTEXT_STALE", telemetry, read_at=read_at)
        if telemetry.admission_backlog or telemetry.projection_gap_count:
            return self._hold("UNRESOLVED_PROJECTION_GAP", telemetry, read_at=read_at)
        if telemetry.admitted_count and (
            not telemetry.projection_reconciled
            or telemetry.contiguous_projection_watermark is None
        ):
            return self._hold(
                "AMBIGUOUS_PROJECTION_WATERMARK", telemetry, read_at=read_at
            )

        reconciliation_rows = tuple(self._connection.execute(
            "SELECT receipt_json, receipt_digest, projector_family_id, generation_id, "
            "authority_watermark FROM "
            "unpublished_graphiti_projection_reconciliations "
            "ORDER BY reconciled_at DESC, receipt_digest DESC"
        ).fetchall())
        reconciliation_row = reconciliation_rows[0] if reconciliation_rows else None
        generation_id: str | None = None
        exact_generation_receipts = False
        active_projection_snapshot: tuple[tuple[str, int, str], ...] = ()
        if telemetry.admitted_count or reconciliation_row is not None:
            if reconciliation_row is None:
                return self._hold(
                    "AMBIGUOUS_PROJECTION_WATERMARK", telemetry, read_at=read_at
                )
            try:
                reconciliation_text = str(reconciliation_row[0])
                reconciliation_value = json.loads(reconciliation_text)
                reconciliation, _binding = (
                    graphiti_projection_reconciliation_from_json(
                        reconciliation_text
                    )
                )
                generation_id = reconciliation.generation_id
                active_projection_snapshot = self._active_projection_snapshot()
                if _binding is not None:
                    generation_id = self._validate_exact_projection_generations(
                        reconciliation_rows=reconciliation_rows,
                        active_projection_snapshot=active_projection_snapshot,
                        contiguous_watermark=(
                            telemetry.contiguous_projection_watermark
                        ),
                    )
                    exact_generation_receipts = True
                else:
                    active_effect_ids = tuple(
                        str(row[0]) for row in active_projection_snapshot
                    )
                    governed_projection_watermark = (
                        max(row[1] for row in active_projection_snapshot)
                        if active_projection_snapshot
                        else self._tombstone_projection_watermark(generation_id)
                    )
                    if (
                        len(reconciliation_rows) != 1
                        or reconciliation.authority_watermark
                        != governed_projection_watermark
                        or any(
                            str(row[2]) != generation_id
                            for row in active_projection_snapshot
                        )
                        or reconciliation.expected_effect_ids != active_effect_ids
                        or reconciliation.actual_effect_ids != active_effect_ids
                    ):
                        return self._hold(
                            "ADMITTED_CONTEXT_RECEIPT_DRIFT",
                            telemetry,
                            read_at=read_at,
                        )
                if (
                    canonical_json_bytes(reconciliation_value).decode()
                    != reconciliation_text
                    or reconciliation.receipt_digest
                    != str(reconciliation_row[1])
                    or reconciliation.projector_family_id
                    != str(reconciliation_row[2])
                    or generation_id != str(reconciliation_row[3])
                    or reconciliation.authority_watermark
                    != int(reconciliation_row[4])
                ):
                    return self._hold(
                        "ADMITTED_CONTEXT_RECEIPT_DRIFT", telemetry, read_at=read_at
                    )
            except (
                GraphitiAdmissionConsumerError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                return self._hold(
                    "ADMITTED_CONTEXT_RECEIPT_INVALID", telemetry, read_at=read_at
                )

        scope_sql = ""
        parameters: list[str] = []
        if source_items is not None:
            if not source_items:
                return self._empty_or_size_hold(
                    telemetry=telemetry,
                    generation_id=generation_id,
                    read_at=read_at,
                )
            clauses = []
            for source_id, item_key in sorted(source_items):
                clauses.append("(ingest.source_id=? AND ingest.item_key=?)")
                parameters.extend((source_id, item_key))
            scope_sql = " AND (" + " OR ".join(clauses) + ")"

        rows = self._connection.execute(
            """
            SELECT queue.request_json, queue.request_digest,
                   queue.proposal_key, queue.source_revision_id,
                   queue.source_receipt_digest, queue.proposal_digest,
                   queue.proposal_kind, queue.ingest_id,
                   ingest.source_id, ingest.item_key,
                   decision.decision_json, decision.decision_digest,
                   decision.action, decision.decision_id,
                   decision.authority_ledger_seq,
                   decision.authority_receipt_digest,
                   projection.receipt_json, projection.receipt_digest,
                   projection.effect_id, projection.authority_watermark,
                   projection.projector_family_id, projection.generation_id,
                   projection.schema_version, projection.trust_scope
            FROM unpublished_graphiti_admission_queue AS queue
            JOIN unpublished_graphiti_ingest AS ingest
              ON ingest.ingest_id=queue.ingest_id
            JOIN unpublished_graphiti_admission_decisions AS decision
              USING(proposal_key)
            JOIN unpublished_graphiti_projection_receipts AS projection
              USING(proposal_key)
            LEFT JOIN unpublished_graphiti_projection_tombstones AS tombstone
              USING(proposal_key)
            WHERE queue.state='PROJECTED' AND decision.action='ADMIT'
              AND tombstone.proposal_key IS NULL
            """
            + scope_sql
            + """
            ORDER BY queue.queue_seq
            """,
            parameters,
        ).fetchall()
        if not rows:
            return self._empty_or_size_hold(
                telemetry=telemetry,
                generation_id=generation_id,
                read_at=read_at,
            )
        if len(rows) > self._max_items:
            return self._hold(
                "ADMITTED_CONTEXT_ITEM_BOUND_EXCEEDED", telemetry, read_at=read_at
            )

        items: list[GovernedContextItem] = []
        try:
            for row in rows:
                (
                    request_raw,
                    request_digest,
                    proposal_key,
                    source_revision_id,
                    source_receipt_digest,
                    proposal_digest,
                    proposal_kind,
                    ingest_id,
                    ingest_source_id,
                    ingest_item_key,
                    decision_raw,
                    decision_digest,
                    decision_action,
                    decision_id,
                    authority_ledger_seq,
                    authority_receipt_digest,
                    projection_raw,
                    projection_receipt_digest,
                    projection_effect_id,
                    projection_authority_watermark,
                    projector_family_id,
                    projection_generation_id,
                    projection_schema_version,
                    projection_trust_scope,
                ) = row
                request_text = str(request_raw)
                decision_text = str(decision_raw)
                projection_text = str(projection_raw)
                request_value = json.loads(request_text)
                decision_value = json.loads(decision_text)
                projection_value = json.loads(projection_text)
                if (
                    canonical_json_bytes(request_value).decode() != request_text
                    or canonical_json_bytes(decision_value).decode() != decision_text
                    or canonical_json_bytes(projection_value).decode()
                    != projection_text
                    or digest_bytes(request_text.encode()) != str(request_digest)
                    or digest_bytes(decision_text.encode()) != str(decision_digest)
                ):
                    return self._hold(
                        "ADMITTED_CONTEXT_RECEIPT_DRIFT", telemetry, read_at=read_at
                    )
                request = graphiti_admission_request_from_value(request_value)
                decision = graphiti_governed_decision_from_json(decision_text)
                projection = graphiti_projection_receipt_from_json(projection_text)
                if (
                    decision.action is not GraphitiProposalAdmissionAction.ADMIT
                    or request.proposal_key != str(proposal_key)
                    or str(request.source_lineage["revision_id"])
                    != str(source_revision_id)
                    or request.source_receipt_digest != str(source_receipt_digest)
                    or request.proposal.digest != str(proposal_digest)
                    or request.proposal.kind.value != str(proposal_kind)
                    or str(request.source_lineage["ingest_id"]) != str(ingest_id)
                    or str(request.source_lineage["source_id"]) != str(ingest_source_id)
                    or str(request.source_lineage["item_key"]) != str(ingest_item_key)
                    or decision.action.value != str(decision_action)
                    or decision.decision_id != str(decision_id)
                    or decision.authority_ledger_seq != int(authority_ledger_seq)
                    or decision.authority_receipt_digest
                    != str(authority_receipt_digest)
                    or projection.receipt_digest != str(projection_receipt_digest)
                    or projection.effect_id != str(projection_effect_id)
                    or projection.authority_watermark
                    != int(projection_authority_watermark)
                    or projection.projector_family_id != str(projector_family_id)
                    or projection.generation_id != str(projection_generation_id)
                    or projection.schema_version != str(projection_schema_version)
                    or projection.trust_scope != str(projection_trust_scope)
                    or request.proposal_key != decision.proposal_key
                    or projection.proposal_key != request.proposal_key
                    or projection.decision_id != decision.decision_id
                    or (
                        projection.authority_watermark
                        < decision.authority_ledger_seq
                        if exact_generation_receipts
                        else projection.authority_watermark
                        != decision.authority_ledger_seq
                    )
                    or projection.trust_scope != "ADMITTED"
                ):
                    return self._hold(
                        "ADMITTED_CONTEXT_RECEIPT_DRIFT", telemetry, read_at=read_at
                    )
                try:
                    rights_current = self._rights.is_current(request)
                    current = self._authority.current_context(request, decision)
                except Exception:  # noqa: BLE001 - currency read faults fail closed
                    return self._hold(
                        "ADMITTED_CONTEXT_CURRENCY_UNAVAILABLE",
                        telemetry,
                        read_at=read_at,
                    )
                if not rights_current:
                    return self._hold(
                        "ADMITTED_CONTEXT_RIGHTS_LOST", telemetry, read_at=read_at
                    )
                if current is None or current.currentness_state != _CURRENT:
                    return self._hold(
                        "ADMITTED_CONTEXT_AUTHORITY_STALE",
                        telemetry,
                        read_at=read_at,
                    )
                lineage = request.source_lineage
                passage_metadata = {
                    str(value["passage_id"]): EvidencePassageLineage.from_value(
                        {
                            field: value[field]
                            for field in (
                                "passage_id",
                                "admission_id",
                                "access_decision_id",
                                "byte_offset",
                                "byte_length",
                                "blob_digest",
                                "text_digest",
                            )
                        }
                    )
                    for value in request.evidence_passages
                }
                passages_list: list[EvidencePassageLineage] = []
                for evidence in request.proposal.evidence:
                    metadata = passage_metadata[str(evidence.passage_id)]
                    if evidence.end_byte > metadata.byte_length:
                        raise ValueError("proposal evidence exceeds retained passage")
                    passages_list.append(
                        EvidencePassageLineage(
                            passage_id=metadata.passage_id,
                            admission_id=metadata.admission_id,
                            access_decision_id=metadata.access_decision_id,
                            byte_offset=metadata.byte_offset + evidence.start_byte,
                            byte_length=evidence.end_byte - evidence.start_byte,
                            blob_digest=metadata.blob_digest,
                            text_digest=evidence.evidence_text_digest,
                        )
                    )
                passages = tuple(
                    sorted(
                        passages_list,
                        key=lambda item: (
                            item.passage_id,
                            item.byte_offset,
                            item.byte_length,
                            item.text_digest,
                        ),
                    )
                )
                proposed_temporal: dict[str, str | None] = {
                    "reference_time": str(lineage["reference_time"]),
                    "temporal_basis": str(lineage["temporal_basis"]),
                }
                if request.proposal.kind is ExtractionProposalKind.RELATION:
                    proposed_temporal.update(
                        {
                            name: None if value is None else str(value)
                            for name, value in (
                                request.relation_temporal_bounds or {}
                            ).items()
                        }
                    )
                items.append(
                    GovernedContextItem(
                        proposal_key=request.proposal_key,
                        authority_bindings=current.bindings,
                        admission_decision_id=decision.decision_id,
                        admission_authority_version=decision.authority_ledger_seq,
                        source_id=str(lineage["source_id"]),
                        item_key=str(lineage["item_key"]),
                        source_revision_id=str(lineage["revision_id"]),
                        source_receipt_digest=request.source_receipt_digest,
                        evidence_passages=passages,
                        proposed_temporal_fields=tuple(
                            sorted(proposed_temporal.items())
                        ),
                        admitted_temporal_fields=current.admitted_temporal_fields,
                        admitted_structured_value_json=(
                            current.admitted_structured_value_json
                        ),
                        currentness_state=current.currentness_state,
                        projection_effect_id=projection.effect_id,
                        projection_generation_id=projection.generation_id,
                        projection_authority_watermark=projection.authority_watermark,
                        contiguous_projection_watermark=(
                            telemetry.contiguous_projection_watermark
                            if telemetry.contiguous_projection_watermark is not None
                            else 0
                        ),
                        projection_gap_count=telemetry.projection_gap_count,
                        oldest_lag_seconds=telemetry.oldest_lag_seconds,
                        stale=False,
                        degraded=False,
                        admission_current=True,
                        rights_current=True,
                        currency_read_at=read_at.isoformat().replace("+00:00", "Z"),
                    )
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return self._hold(
                "ADMITTED_CONTEXT_RECEIPT_INVALID", telemetry, read_at=read_at
            )

        canonical_items = tuple(sorted(items, key=lambda item: item.proposal_key))
        final_projection_snapshot = self._active_projection_snapshot()
        final_reconciliation_rows = tuple(self._connection.execute(
            "SELECT receipt_json, receipt_digest, projector_family_id, generation_id, "
            "authority_watermark FROM "
            "unpublished_graphiti_projection_reconciliations "
            "ORDER BY reconciled_at DESC, receipt_digest DESC"
        ).fetchall())
        if (
            final_projection_snapshot != active_projection_snapshot
            or final_reconciliation_rows != reconciliation_rows
        ):
            return self._hold(
                "ADMITTED_CONTEXT_RECEIPT_DRIFT", telemetry, read_at=read_at
            )
        ready = self._result(
            status=GovernedContextStatus.READY,
            reason_code="ADMITTED_CONTEXT_READY",
            items=canonical_items,
            generation_id=generation_id,
            watermark=telemetry.contiguous_projection_watermark,
            oldest_lag_seconds=telemetry.oldest_lag_seconds,
            read_at=read_at,
        )
        if self._exceeds_size_bound(ready):
            return self._hold(
                "ADMITTED_CONTEXT_SIZE_BOUND_EXCEEDED", telemetry, read_at=read_at
            )
        return ready


__all__ = [
    "ADMITTED_CONTEXT_SCHEMA_VERSION",
    "ADMITTED_CONTEXT_TRUST_LABEL",
    "AuthorityContextBinding",
    "EvidencePassageLineage",
    "GovernedAuthorityContext",
    "GovernedContext",
    "GovernedContextAuthority",
    "GovernedContextHydrator",
    "GovernedContextItem",
    "GovernedContextStatus",
]
