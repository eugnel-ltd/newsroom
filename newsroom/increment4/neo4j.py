from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from newsroom.authority import AuthenticationProof, UtcTimestamp
from newsroom.authority.canonical import validate_sha256_digest
from newsroom.projection import (
    ProjectionGenerationId,
    ProjectionGenerationPromotionView,
    ProjectionGenerationValidationView,
    ProjectionGenerationView,
)
from newsroom.projection.neo4j import (
    StructuralReadResponse,
    StructuralReconciliationView,
)

from .contracts import INCREMENT4_ADMITTED_FAMILY_ID
from .models import Increment4AdmittedProjectionSnapshot


class Increment4Neo4jProofError(RuntimeError):
    """Base error for the bounded Increment 4 actual-Neo4j proof controller."""


def _require_idempotency_key(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value.encode("utf-8")) > 192
    ):
        raise ValueError("Increment 4 idempotency key must be bounded canonical text")
    return value


def _require_reason(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value.encode("utf-8")) > 128
    ):
        raise ValueError("Increment 4 reason code must be bounded canonical text")
    if any(not (character.isalnum() or character in "._:-") for character in value):
        raise ValueError("Increment 4 reason code contains unsupported characters")
    return value


def _require_non_negative_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class Increment4Neo4jBuildRequest:
    generation_id: ProjectionGenerationId
    snapshot: Increment4AdmittedProjectionSnapshot
    reason_code: str
    idempotency_key: str
    purge_retired_generation: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.generation_id, ProjectionGenerationId):
            raise TypeError("Increment 4 build generation identity must be typed")
        if not isinstance(self.snapshot, Increment4AdmittedProjectionSnapshot):
            raise TypeError("Increment 4 build requires a typed admitted snapshot")
        _require_reason(self.reason_code)
        _require_idempotency_key(self.idempotency_key)
        if not isinstance(self.purge_retired_generation, bool):
            raise TypeError("Increment 4 retired-generation purge flag must be boolean")


@dataclass(frozen=True, slots=True)
class Increment4Neo4jCurrentBuildRequest:
    """Build current authority, optionally extending an unchanged ACTIVE prefix."""

    generation_id: ProjectionGenerationId
    reason_code: str
    idempotency_key: str
    purge_retired_generation: bool = True
    allow_active_extension: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.generation_id, ProjectionGenerationId):
            raise TypeError("Increment 4 build generation identity must be typed")
        _require_reason(self.reason_code)
        _require_idempotency_key(self.idempotency_key)
        if not isinstance(self.purge_retired_generation, bool):
            raise TypeError("Increment 4 retired-generation purge flag must be boolean")

        if not isinstance(self.allow_active_extension, bool):
            raise TypeError("Increment 4 ACTIVE extension flag must be boolean")


@dataclass(frozen=True, slots=True)
class Increment4Neo4jBuildResult:
    family_id: str
    generation: ProjectionGenerationView
    prior_generation: ProjectionGenerationView | None
    validation: ProjectionGenerationValidationView
    promotion: ProjectionGenerationPromotionView
    source_watermark_ledger_seq: int
    checkpoint_ledger_seq: int
    projected_batch_count: int
    ignored_optional_count: int
    deleted_target_graph_record_count: int
    purged_retired_graph_record_count: int
    source_snapshot_digest: str
    projection_state_digest: str
    serving_time: UtcTimestamp

    def __post_init__(self) -> None:
        if self.family_id != INCREMENT4_ADMITTED_FAMILY_ID:
            raise ValueError("Increment 4 build result belongs to another family")
        if not isinstance(self.generation, ProjectionGenerationView):
            raise TypeError("Increment 4 build result requires a typed generation")
        if self.generation.family_id != self.family_id:
            raise ValueError("Increment 4 build result generation family differs")
        if self.prior_generation is not None:
            if not isinstance(self.prior_generation, ProjectionGenerationView):
                raise TypeError("Increment 4 prior generation must be typed")
            if self.prior_generation.family_id != self.family_id:
                raise ValueError("Increment 4 prior generation family differs")
        if not isinstance(self.validation, ProjectionGenerationValidationView):
            raise TypeError("Increment 4 build result requires typed validation")
        if not isinstance(self.promotion, ProjectionGenerationPromotionView):
            raise TypeError("Increment 4 build result requires typed promotion")
        if (
            self.validation.generation_id != self.generation.generation_id
            or self.promotion.generation.generation_id
            != self.generation.generation_id
        ):
            raise ValueError("Increment 4 validation/promotion generation differs")
        for field, value in (
            ("source_watermark_ledger_seq", self.source_watermark_ledger_seq),
            ("checkpoint_ledger_seq", self.checkpoint_ledger_seq),
            ("projected_batch_count", self.projected_batch_count),
            ("ignored_optional_count", self.ignored_optional_count),
            (
                "deleted_target_graph_record_count",
                self.deleted_target_graph_record_count,
            ),
            (
                "purged_retired_graph_record_count",
                self.purged_retired_graph_record_count,
            ),
        ):
            _require_non_negative_int(value, field=field)
        if self.checkpoint_ledger_seq < self.source_watermark_ledger_seq:
            raise ValueError("Increment 4 checkpoint cannot precede source watermark")
        validate_sha256_digest(
            self.source_snapshot_digest, field="source_snapshot_digest"
        )
        validate_sha256_digest(
            self.projection_state_digest, field="projection_state_digest"
        )
        if self.projection_state_digest != self.validation.projection_state_digest:
            raise ValueError("Increment 4 result state digest differs from validation")
        if not isinstance(self.serving_time, UtcTimestamp):
            raise TypeError("Increment 4 build serving time must be typed")


@dataclass(frozen=True, slots=True)
class Increment4Neo4jGenerationStatus:
    generation: ProjectionGenerationView
    contiguous_ledger_seq: int
    open_gap_count: int
    dead_letter_count: int
    source_watermark_ledger_seq: int
    serving_time: UtcTimestamp

    def __post_init__(self) -> None:
        if not isinstance(self.generation, ProjectionGenerationView):
            raise TypeError("Increment 4 generation status requires typed generation")
        if self.generation.family_id != INCREMENT4_ADMITTED_FAMILY_ID:
            raise ValueError("Increment 4 generation status belongs to another family")
        for field, value in (
            ("contiguous_ledger_seq", self.contiguous_ledger_seq),
            ("open_gap_count", self.open_gap_count),
            ("dead_letter_count", self.dead_letter_count),
            ("source_watermark_ledger_seq", self.source_watermark_ledger_seq),
        ):
            _require_non_negative_int(value, field=field)
        if not isinstance(self.serving_time, UtcTimestamp):
            raise TypeError("Increment 4 status serving time must be typed")


@dataclass(frozen=True, slots=True)
class Increment4Neo4jActiveReadRequest:
    canonical_ids: tuple[str, ...]
    query_valid_time: UtcTimestamp
    limit: int = 100

    def __post_init__(self) -> None:
        if (
            not isinstance(self.canonical_ids, tuple)
            or not self.canonical_ids
            or len(self.canonical_ids) > 1000
        ):
            raise ValueError("Increment 4 active read requires a bounded ID tuple")
        if any(
            not isinstance(item, str)
            or not item
            or item != item.strip()
            or len(item.encode("utf-8")) > 512
            for item in self.canonical_ids
        ):
            raise ValueError("Increment 4 active read canonical ID is invalid")
        if len(set(self.canonical_ids)) != len(self.canonical_ids):
            raise ValueError("Increment 4 active read canonical IDs must be unique")
        if not isinstance(self.query_valid_time, UtcTimestamp):
            raise TypeError("Increment 4 active read valid time must be typed")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit <= 0:
            raise ValueError("Increment 4 active read limit must be positive")


class Increment4Neo4jController:
    """Bounded proof controller; it owns no SQLite or Neo4j capability."""

    __slots__ = (
        "__build",
        "__build_current",
        "__status",
        "__read_active",
        "__reconcile_active",
    )

    def __init__(
        self,
        *,
        build: Callable[
            [Increment4Neo4jBuildRequest, AuthenticationProof],
            Increment4Neo4jBuildResult,
        ],
        build_current: Callable[
            [Increment4Neo4jCurrentBuildRequest, AuthenticationProof],
            Increment4Neo4jBuildResult,
        ],
        status: Callable[
            [ProjectionGenerationId, AuthenticationProof],
            Increment4Neo4jGenerationStatus,
        ],
        read_active: Callable[
            [Increment4Neo4jActiveReadRequest, AuthenticationProof],
            StructuralReadResponse,
        ],
        reconcile_active: Callable[
            [AuthenticationProof], StructuralReconciliationView
        ],
    ) -> None:
        self.__build = build
        self.__build_current = build_current
        self.__status = status
        self.__read_active = read_active
        self.__reconcile_active = reconcile_active

    def build_and_promote(
        self,
        request: Increment4Neo4jBuildRequest,
        *,
        proof: AuthenticationProof,
    ) -> Increment4Neo4jBuildResult:
        return self.__build(request, proof)

    def build_current_and_promote(
        self,
        request: Increment4Neo4jCurrentBuildRequest,
        *,
        proof: AuthenticationProof,
    ) -> Increment4Neo4jBuildResult:
        """Build one generation from all current admitted 4B/4C authority."""

        return self.__build_current(request, proof)

    def generation_status(
        self,
        generation_id: ProjectionGenerationId,
        *,
        proof: AuthenticationProof,
    ) -> Increment4Neo4jGenerationStatus:
        return self.__status(generation_id, proof)

    def read_active(
        self,
        request: Increment4Neo4jActiveReadRequest,
        *,
        proof: AuthenticationProof,
    ) -> StructuralReadResponse:
        return self.__read_active(request, proof)

    def reconcile_active(
        self, *, proof: AuthenticationProof
    ) -> StructuralReconciliationView:
        """Authenticate and reconcile the authority-selected ACTIVE generation."""

        return self.__reconcile_active(proof)


__all__ = [
    "Increment4Neo4jActiveReadRequest",
    "Increment4Neo4jBuildRequest",
    "Increment4Neo4jBuildResult",
    "Increment4Neo4jController",
    "Increment4Neo4jCurrentBuildRequest",
    "Increment4Neo4jGenerationStatus",
    "Increment4Neo4jProofError",
]
