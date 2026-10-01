"""Authoritative hydration and immutable Retrieval Contexts for Increment 5D2.

The boundary replays the exact 5D1 composition from retained inputs, validates a
fresh current SQLite authority receipt through the pure 5D1 validator, pins and
re-hashes governed CAS bytes, and emits one bounded read-only context. Ranking
never becomes factual authority and no Candidate or Hypothesis is created.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import threading
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes

from .hybrid_composer import (
    HybridCandidate,
    HybridComposer,
    HybridCompositionInput,
    HybridCompositionOutcome,
    HybridCompositionPurpose,
    HybridCompositionReceipt,
    HybridCompositionRequest,
    HybridManifestEntry,
    HybridMode,
)
from .named_tool_authority_execution import (
    NamedAuthorityExecutionOutcome,
    NamedAuthorityExecutionReceipt,
)
from .named_tool_authority_receipt_validation import (
    NamedAuthorityReceiptValidationError,
    validate_named_authority_receipt,
)
from .named_tool_contracts import (
    NAMED_TOOL_CONTRACT_DIGEST,
    NAMED_TOOL_POLICY_ID,
    NAMED_TOOL_PROFILE_ID,
    CollisionHydrationLookupToolRequest,
    NamedToolId,
    NamedToolPurpose,
    decode_named_tool_json,
)


CONTEXT_LIMIT_BYTES = 262_144
AUTHORITY_PASSAGE_LIMIT = 7
CAS_BLOB_LIMIT_BYTES = 64 * 1024 * 1024
CAS_IO_CHUNK_BYTES = 64 * 1024
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\-]{0,255}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{6})?Z\Z")
_MODE_ORDER = {mode: index for index, mode in enumerate(HybridMode)}
_TOOL_ORDER = {tool: index for index, tool in enumerate(NamedToolId)}


class RetrievalContextError(RuntimeError):
    """The context request, evidence, bytes, or replay journal is invalid."""


class GovernedBytesUnavailable(RuntimeError):
    """The governed CAS object is not safely available."""


class GovernedBytesIntegrityError(RuntimeError):
    """The governed CAS object or admitted range failed integrity checks."""


class RetrievalContextOutcome(StrEnum):
    COMPLETE = "COMPLETE"
    DEGRADED = "DEGRADED"
    INCOMPLETE = "INCOMPLETE"
    STALE = "STALE"
    RIGHTS_BLOCKED = "RIGHTS_BLOCKED"
    BUDGET_BLOCKED = "BUDGET_BLOCKED"
    INTEGRITY_BLOCKED = "INTEGRITY_BLOCKED"
    UNAVAILABLE = "UNAVAILABLE"


class RetrievalContextReason(StrEnum):
    NO_MATCH = "NO_MATCH"
    OPTIONAL_EVIDENCE_NON_COMPLETE = "OPTIONAL_EVIDENCE_NON_COMPLETE"
    COMPOSITION_INCOMPLETE = "COMPOSITION_INCOMPLETE"
    COMPOSITION_STALE = "COMPOSITION_STALE"
    COMPOSITION_RIGHTS_BLOCKED = "COMPOSITION_RIGHTS_BLOCKED"
    COMPOSITION_UNAVAILABLE = "COMPOSITION_UNAVAILABLE"
    COMPOSITION_REPLAY_MISMATCH = "COMPOSITION_REPLAY_MISMATCH"
    MISSING_AUTHORITY_EVIDENCE = "MISSING_AUTHORITY_EVIDENCE"
    AUTHORITY_REQUEST_MISMATCH = "AUTHORITY_REQUEST_MISMATCH"
    AUTHORITY_INCOMPLETE = "AUTHORITY_INCOMPLETE"
    AUTHORITY_STALE = "AUTHORITY_STALE"
    AUTHORITY_RIGHTS_BLOCKED = "AUTHORITY_RIGHTS_BLOCKED"
    AUTHORITY_UNAVAILABLE = "AUTHORITY_UNAVAILABLE"
    AUTHORITY_RECEIPT_INVALID = "AUTHORITY_RECEIPT_INVALID"
    AUTHORITY_WATERMARK_STALE = "AUTHORITY_WATERMARK_STALE"
    NO_AUTHORITATIVE_PASSAGE = "NO_AUTHORITATIVE_PASSAGE"
    COLLISION_CONTRADICTS_NO_MATCH = "COLLISION_CONTRADICTS_NO_MATCH"
    GOVERNED_BYTES_UNAVAILABLE = "GOVERNED_BYTES_UNAVAILABLE"
    GOVERNED_BYTES_INTEGRITY = "GOVERNED_BYTES_INTEGRITY"
    RETAINED_CONTEXT_PURGED = "RETAINED_CONTEXT_PURGED"
    AUTHORITY_RESULT_BOUND = "AUTHORITY_RESULT_BOUND"
    CONTEXT_BYTE_BOUND = "CONTEXT_BYTE_BOUND"


class RetrievalContextExclusionReason(StrEnum):
    AUTHORITY_RESULT_BOUND = "AUTHORITY_RESULT_BOUND"
    CONTEXT_BYTE_BOUND = "CONTEXT_BYTE_BOUND"


RETRIEVAL_CONTEXT_CONTRACT_DIGEST = digest_bytes(
    canonical_json_bytes(
        {
            "schema_version": "newsroom.increment5.retrieval-context-contract.v1",
            "composition": {
                "exact_5d1_replay_required": True,
                "all_manifest_receipts_bound": True,
                "ranking_is_authority": False,
            },
            "authority": {
                "tool": (
                    NamedToolId.CURRENT_COLLISION_AND_AUTHORITY_HYDRATION_LOOKUP.value
                ),
                "pure_receipt_validator_required": True,
                "current_collision_separate_from_rank": True,
                "current_rights_and_lifecycle_required": True,
                "minimum_watermark_not_behind_composition": True,
            },
            "hydration": {
                "governed_cas_only": True,
                "pin_and_full_blob_rehash_before_and_after": True,
                "exact_passage_digest_required": True,
                "utf8_required": True,
                "source_content_instruction_effect": "NONE",
            },
            "bounds": {
                "context_bytes": CONTEXT_LIMIT_BYTES,
                "authority_passages": AUTHORITY_PASSAGE_LIMIT,
                "cas_blob_bytes": CAS_BLOB_LIMIT_BYTES,
            },
            "outcomes": [item.value for item in RetrievalContextOutcome],
            "no_match_only_when_complete": True,
            "authority_effect": "NONE",
            "external_calls": 0,
            "provider_spend_micros": 0,
        }
    )
)
GOVERNED_CAS_HYDRATOR_CONTRACT_DIGEST = digest_bytes(
    canonical_json_bytes(
        {
            "schema_version": "newsroom.increment5.governed-cas-hydrator.v1",
            "path": "objects/<sha256-prefix>/<sha256-hex>",
            "nofollow": True,
            "regular_read_only_file": True,
            "full_blob_rehash_before_and_after": True,
            "exact_range_digest": True,
            "maximum_blob_bytes": CAS_BLOB_LIMIT_BYTES,
            "io_chunk_bytes": CAS_IO_CHUNK_BYTES,
        }
    )
)


def _canonical(value: object) -> bytes:
    try:
        return canonical_json_bytes(value)
    except Exception as exc:
        raise RetrievalContextError("value is not canonical JSON") from exc


def _digest_bytes(raw: bytes) -> str:
    return digest_bytes(raw)


def _require_digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise RetrievalContextError(f"{field} must be a canonical SHA-256 digest")
    return value


def _require_token(value: object, field: str) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise RetrievalContextError(f"{field} must be a bounded canonical token")
    return value


def _require_text(value: object, field: str, maximum_bytes: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value.encode("utf-8")) > maximum_bytes
    ):
        raise RetrievalContextError(f"{field} must be bounded canonical text")
    return value


def _require_uint(value: object, field: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RetrievalContextError(f"{field} must be an integer >= {minimum}")
    return value


def _parse_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise RetrievalContextError(f"{field} must be canonical UTC text")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise RetrievalContextError(f"{field} must be canonical UTC text") from exc
    if parsed.tzinfo != UTC:
        raise RetrievalContextError(f"{field} must use UTC")
    return parsed


def _require_uuid(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise RetrievalContextError(f"{field} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise RetrievalContextError(f"{field} must be a canonical UUID") from exc
    if str(parsed) != value:
        raise RetrievalContextError(f"{field} must be a canonical UUID")
    return value


def _sorted_unique_text(
    values: tuple[str, ...], field: str, *, allow_empty: bool = True
) -> tuple[str, ...]:
    if (
        not isinstance(values, tuple)
        or values != tuple(sorted(set(values)))
        or (not allow_empty and not values)
    ):
        raise RetrievalContextError(f"{field} must be sorted and unique")
    for item in values:
        _require_text(item, field)
    return values


def _sorted_unique_digests(
    values: tuple[str, ...], field: str, *, allow_empty: bool = True
) -> tuple[str, ...]:
    if (
        not isinstance(values, tuple)
        or values != tuple(sorted(set(values)))
        or (not allow_empty and not values)
    ):
        raise RetrievalContextError(f"{field} must be sorted and unique")
    for item in values:
        _require_digest(item, field)
    return values


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RetrievalContextError("retained JSON contains duplicate keys")
        result[key] = value
    return result


def _decode_canonical(raw: bytes, field: str) -> dict[str, object]:
    if not isinstance(raw, bytes):
        raise RetrievalContextError(f"{field} must be bytes")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetrievalContextError(f"{field} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or _canonical(value) != raw:
        raise RetrievalContextError(f"{field} is not canonical JSON")
    return value


def named_request_bytes(request: CollisionHydrationLookupToolRequest) -> bytes:
    """Return the exact deterministic bytes accepted by the named-tool decoder."""

    try:
        value = {
            "schema_version": request.SCHEMA_VERSION,
            "envelope": request.envelope.canonical_value(),
            **request.payload_value(),
        }
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        decoded = decode_named_tool_json(raw)
    except Exception as exc:
        raise RetrievalContextError("authority request cannot be retained") from exc
    if decoded != request:
        raise RetrievalContextError("authority request round trip failed")
    return raw


def context_collision_key_digest(composition: HybridCompositionReceipt) -> str:
    """Bind a request-local collision probe without asserting semantic identity."""

    if not isinstance(composition, HybridCompositionReceipt):
        raise TypeError("collision key requires a typed composition receipt")
    return _digest_bytes(
        _canonical(
            {
                "schema_version": (
                    "newsroom.increment5.retrieval-context-collision-key.v1"
                ),
                "composition_id": composition.composition_id,
                "query_valid_time": composition.query_valid_time,
                "dependency_root_ids": [
                    item.dependency_root_id for item in composition.candidates
                ],
                "ranking_is_authority": False,
            }
        )
    )


@dataclass(frozen=True, slots=True)
class GovernedPassageReference:
    passage_id: str
    admission_id: str
    access_decision_id: str
    blob_digest: str
    text_digest: str
    language: str
    byte_offset: int
    byte_length: int
    object_class: str
    allowed_use: str
    security_scope: str
    retention_scope: str
    rights_decision_id: str
    rights_digest: str
    authority_metadata_digest: str

    def __post_init__(self) -> None:
        for name in (
            "passage_id",
            "admission_id",
            "access_decision_id",
            "language",
            "object_class",
            "allowed_use",
            "security_scope",
            "retention_scope",
            "rights_decision_id",
        ):
            _require_text(getattr(self, name), f"governed_{name}")
        for name in (
            "blob_digest",
            "text_digest",
            "rights_digest",
            "authority_metadata_digest",
        ):
            _require_digest(getattr(self, name), f"governed_{name}")
        _require_uint(self.byte_offset, "governed_byte_offset")
        _require_uint(self.byte_length, "governed_byte_length", positive=True)
        if self.byte_length > CONTEXT_LIMIT_BYTES:
            raise RetrievalContextError("passage exceeds the context byte bound")

    @classmethod
    def from_authority(
        cls, value: Mapping[str, object]
    ) -> "GovernedPassageReference":
        required = {
            "passage_id",
            "run_id",
            "admission_id",
            "access_decision_id",
            "blob_digest",
            "text_digest",
            "language",
            "byte_offset",
            "byte_length",
            "object_class",
            "allowed_use",
            "security_scope",
            "retention_scope",
            "admission_state",
            "blob_state",
            "blob_integrity_state",
            "rights_decision_id",
            "rights_allowed",
            "rights_reason",
            "rights_digest",
            "usable",
            "block_reason",
        }
        if set(value) != required:
            raise RetrievalContextError("authority passage keys are not exact")
        if (
            value["admission_state"] != "ACTIVE"
            or value["blob_state"] != "ACTIVE"
            or value["blob_integrity_state"] != "VERIFIED"
            or value["rights_allowed"] is not True
            or value["usable"] is not True
            or value["block_reason"] is not None
        ):
            raise RetrievalContextError("authority passage is not currently usable")
        return cls(
            passage_id=value["passage_id"],
            admission_id=value["admission_id"],
            access_decision_id=value["access_decision_id"],
            blob_digest=value["blob_digest"],
            text_digest=value["text_digest"],
            language=value["language"],
            byte_offset=value["byte_offset"],
            byte_length=value["byte_length"],
            object_class=value["object_class"],
            allowed_use=value["allowed_use"],
            security_scope=value["security_scope"],
            retention_scope=value["retention_scope"],
            rights_decision_id=value["rights_decision_id"],
            rights_digest=value["rights_digest"],
            authority_metadata_digest=_digest_bytes(_canonical(dict(value))),
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "passage_id": self.passage_id,
            "admission_id": self.admission_id,
            "access_decision_id": self.access_decision_id,
            "blob_digest": self.blob_digest,
            "text_digest": self.text_digest,
            "language": self.language,
            "byte_offset": self.byte_offset,
            "byte_length": self.byte_length,
            "object_class": self.object_class,
            "allowed_use": self.allowed_use,
            "security_scope": self.security_scope,
            "retention_scope": self.retention_scope,
            "rights_decision_id": self.rights_decision_id,
            "rights_digest": self.rights_digest,
            "authority_metadata_digest": self.authority_metadata_digest,
        }


class GovernedPassageHydrator(Protocol):
    @property
    def implementation_digest(self) -> str:
        ...

    def read(self, reference: GovernedPassageReference) -> bytes:
        ...


class GovernedCasPassageHydrator:
    """Read an exact admitted range from a pinned, immutable CAS blob."""

    def __init__(self, root: Path) -> None:
        supplied = Path(root)
        if supplied.is_symlink():
            raise GovernedBytesIntegrityError("CAS root cannot be a symlink")
        try:
            self.root = supplied.resolve(strict=True)
        except FileNotFoundError as exc:
            raise GovernedBytesUnavailable("CAS root is unavailable") from exc
        self.objects_root = self.root / "objects"
        if self.objects_root.is_symlink() or not self.objects_root.is_dir():
            raise GovernedBytesIntegrityError("CAS objects root is invalid")

    @property
    def implementation_digest(self) -> str:
        return GOVERNED_CAS_HYDRATOR_CONTRACT_DIGEST

    @staticmethod
    def _hash_fd(fd: int) -> tuple[str, int]:
        os.lseek(fd, 0, os.SEEK_SET)
        hasher = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(fd, CAS_IO_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > CAS_BLOB_LIMIT_BYTES:
                raise GovernedBytesIntegrityError("CAS blob exceeds fixed bound")
            hasher.update(chunk)
        os.lseek(fd, 0, os.SEEK_SET)
        return f"sha256:{hasher.hexdigest()}", size

    def read(self, reference: GovernedPassageReference) -> bytes:
        if not isinstance(reference, GovernedPassageReference):
            raise TypeError("hydrator requires a typed passage reference")
        digest_hex = reference.blob_digest.removeprefix("sha256:")
        shard = self.objects_root / digest_hex[:2]
        path = shard / digest_hex
        if shard.is_symlink() or path.is_symlink():
            raise GovernedBytesIntegrityError("CAS path contains a symlink")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            fd = os.open(path, flags)
        except FileNotFoundError as exc:
            raise GovernedBytesUnavailable("CAS object is unavailable") from exc
        except OSError as exc:
            raise GovernedBytesIntegrityError("CAS object cannot be opened") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise GovernedBytesIntegrityError("CAS object is not regular")
            if stat.S_IMODE(info.st_mode) & 0o222:
                raise GovernedBytesIntegrityError("CAS object remains writable")
            digest_before, size = self._hash_fd(fd)
            if digest_before != reference.blob_digest:
                raise GovernedBytesIntegrityError("CAS blob digest differs")
            end = reference.byte_offset + reference.byte_length
            if end > size:
                raise GovernedBytesIntegrityError("passage range exceeds blob")
            data = os.pread(fd, reference.byte_length, reference.byte_offset)
            if len(data) != reference.byte_length:
                raise GovernedBytesIntegrityError("passage range ended early")
            if _digest_bytes(data) != reference.text_digest:
                raise GovernedBytesIntegrityError("passage digest differs")
            digest_after, size_after = self._hash_fd(fd)
            if digest_after != digest_before or size_after != size:
                raise GovernedBytesIntegrityError("CAS blob changed during read")
            return data
        finally:
            os.close(fd)


@dataclass(frozen=True, slots=True)
class RetrievalAuthorityEvidence:
    tool_request_digest: str
    named_request_bytes_digest: str
    execution_receipt_digest: str
    raw_receipt_digest: str
    adapter_contract_digest: str
    adapter_config_digest: str
    authority_scope_id: str
    authority_watermark: int
    collision_namespace: str
    collision_key_digest: str
    collision_state: str
    candidate_id: str | None
    requested_object_ids: tuple[str, ...]
    requested_passage_ids: tuple[str, ...]
    query_valid_time: str
    serving_time: str
    outcome: str
    reason: str | None

    def __post_init__(self) -> None:
        for name in (
            "tool_request_digest",
            "named_request_bytes_digest",
            "execution_receipt_digest",
            "raw_receipt_digest",
            "adapter_contract_digest",
            "adapter_config_digest",
            "collision_key_digest",
        ):
            _require_digest(getattr(self, name), f"authority_{name}")
        for name in (
            "authority_scope_id",
            "collision_namespace",
            "collision_state",
            "outcome",
        ):
            _require_token(getattr(self, name), f"authority_{name}")
        if self.candidate_id is not None:
            _require_text(self.candidate_id, "authority_candidate_id")
        _require_uint(self.authority_watermark, "authority_watermark")
        _sorted_unique_text(self.requested_object_ids, "authority_object_id")
        _sorted_unique_text(self.requested_passage_ids, "authority_passage_id")
        if _parse_utc(
            self.query_valid_time, "authority_query_valid_time"
        ) > _parse_utc(self.serving_time, "authority_serving_time"):
            raise RetrievalContextError("authority valid time exceeds serving")
        if self.reason is not None:
            _require_token(self.reason, "authority_reason")

    def canonical_value(self) -> dict[str, object]:
        return {
            "tool_request_digest": self.tool_request_digest,
            "named_request_bytes_digest": self.named_request_bytes_digest,
            "execution_receipt_digest": self.execution_receipt_digest,
            "raw_receipt_digest": self.raw_receipt_digest,
            "adapter_contract_digest": self.adapter_contract_digest,
            "adapter_config_digest": self.adapter_config_digest,
            "authority_scope_id": self.authority_scope_id,
            "authority_watermark": self.authority_watermark,
            "collision_namespace": self.collision_namespace,
            "collision_key_digest": self.collision_key_digest,
            "collision_state": self.collision_state,
            "candidate_id": self.candidate_id,
            "requested_object_ids": list(self.requested_object_ids),
            "requested_passage_ids": list(self.requested_passage_ids),
            "query_valid_time": self.query_valid_time,
            "serving_time": self.serving_time,
            "outcome": self.outcome,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class HydratedContextItem:
    context_rank: int
    composition_rank: int
    dependency_root_id: str
    composition_candidate_digest: str
    precedence: str
    score_numerator: int
    score_denominator: int
    contributing_modes: tuple[HybridMode, ...]
    all_origin_digests: tuple[str, ...]
    passage_origin_digests: tuple[str, ...]
    provenance_digests: tuple[str, ...]
    trust_scopes: tuple[str, ...]
    passage: GovernedPassageReference
    text: str
    text_bytes: int
    query_valid_time: str
    composition_serving_time: str
    authority_serving_time: str
    context_serving_time: str
    source_content_instruction_effect: str = "NONE"

    def __post_init__(self) -> None:
        _require_uint(self.context_rank, "item_context_rank", positive=True)
        _require_uint(self.composition_rank, "item_composition_rank", positive=True)
        _require_text(self.dependency_root_id, "item_dependency_root")
        _require_digest(
            self.composition_candidate_digest, "item_candidate_digest"
        )
        _require_token(self.precedence, "item_precedence")
        _require_uint(self.score_numerator, "item_score_numerator", positive=True)
        _require_uint(self.score_denominator, "item_score_denominator", positive=True)
        if self.score_numerator >= self.score_denominator:
            raise RetrievalContextError("item score must be below one")
        expected_modes = tuple(
            sorted(set(self.contributing_modes), key=_MODE_ORDER.__getitem__)
        )
        if self.contributing_modes != expected_modes:
            raise RetrievalContextError("item modes must be sorted and unique")
        _sorted_unique_digests(
            self.all_origin_digests, "item_origin_digest", allow_empty=False
        )
        _sorted_unique_digests(
            self.passage_origin_digests,
            "item_passage_origin_digest",
            allow_empty=False,
        )
        _sorted_unique_digests(
            self.provenance_digests,
            "item_provenance_digest",
            allow_empty=False,
        )
        _sorted_unique_text(self.trust_scopes, "item_trust_scope")
        if not isinstance(self.passage, GovernedPassageReference):
            raise RetrievalContextError("item passage reference must be typed")
        if not isinstance(self.text, str):
            raise RetrievalContextError("item text must be UTF-8 text")
        encoded = self.text.encode("utf-8")
        _require_uint(self.text_bytes, "item_text_bytes", positive=True)
        if len(encoded) != self.text_bytes:
            raise RetrievalContextError("item text byte count differs")
        if _digest_bytes(encoded) != self.passage.text_digest:
            raise RetrievalContextError("item text digest differs from authority")
        times = tuple(
            _parse_utc(getattr(self, name), f"item_{name}")
            for name in (
                "query_valid_time",
                "composition_serving_time",
                "authority_serving_time",
                "context_serving_time",
            )
        )
        if not times[0] <= times[1] <= times[2] <= times[3]:
            raise RetrievalContextError("item time boundary is not increasing")
        if self.source_content_instruction_effect != "NONE":
            raise RetrievalContextError("source content cannot alter authority")

    def canonical_value(self) -> dict[str, object]:
        return {
            "context_rank": self.context_rank,
            "composition_rank": self.composition_rank,
            "dependency_root_id": self.dependency_root_id,
            "composition_candidate_digest": self.composition_candidate_digest,
            "precedence": self.precedence,
            "score": {
                "numerator": self.score_numerator,
                "denominator": self.score_denominator,
            },
            "contributing_modes": [item.value for item in self.contributing_modes],
            "all_origin_digests": list(self.all_origin_digests),
            "passage_origin_digests": list(self.passage_origin_digests),
            "provenance_digests": list(self.provenance_digests),
            "trust_scopes": list(self.trust_scopes),
            "passage": self.passage.canonical_value(),
            "text": self.text,
            "text_bytes": self.text_bytes,
            "query_valid_time": self.query_valid_time,
            "composition_serving_time": self.composition_serving_time,
            "authority_serving_time": self.authority_serving_time,
            "context_serving_time": self.context_serving_time,
            "source_content_instruction_effect": (
                self.source_content_instruction_effect
            ),
        }


@dataclass(frozen=True, slots=True)
class RetrievalContextExclusion:
    composition_rank: int
    dependency_root_id: str
    reason: RetrievalContextExclusionReason
    candidate_digest: str

    def __post_init__(self) -> None:
        _require_uint(
            self.composition_rank, "exclusion_composition_rank", positive=True
        )
        _require_text(self.dependency_root_id, "exclusion_dependency_root")
        if not isinstance(self.reason, RetrievalContextExclusionReason):
            raise RetrievalContextError("exclusion reason must be typed")
        _require_digest(self.candidate_digest, "exclusion_candidate_digest")

    def canonical_value(self) -> dict[str, object]:
        return {
            "composition_rank": self.composition_rank,
            "dependency_root_id": self.dependency_root_id,
            "reason": self.reason.value,
            "candidate_digest": self.candidate_digest,
        }


@dataclass(frozen=True, slots=True)
class RetrievalContextRequest:
    request_id: str
    idempotency_key: str
    actor_id: str
    authenticated_principal_digest: str
    purpose: HybridCompositionPurpose
    policy_id: str
    policy_digest: str
    named_tool_contract_digest: str
    profile_id: str
    query_valid_time: str
    composition_serving_time: str
    context_serving_time: str
    composition_idempotency_key: str
    composition_receipt_bytes: bytes
    composition_inputs: tuple[HybridCompositionInput, ...]
    authority_request_bytes: bytes | None
    authority_execution_receipt_bytes: bytes | None
    authority_receipt_bytes: bytes | None
    context_limit_bytes: int = CONTEXT_LIMIT_BYTES
    authority_passage_limit: int = AUTHORITY_PASSAGE_LIMIT

    def __post_init__(self) -> None:
        _require_uuid(self.request_id, "context_request_id")
        _require_token(self.idempotency_key, "context_idempotency_key")
        _require_token(self.actor_id, "context_actor_id")
        _require_digest(
            self.authenticated_principal_digest, "context_principal_digest"
        )
        if not isinstance(self.purpose, HybridCompositionPurpose):
            raise RetrievalContextError("context purpose must be typed")
        if self.policy_id != NAMED_TOOL_POLICY_ID:
            raise RetrievalContextError("context policy id is not accepted")
        _require_digest(self.policy_digest, "context_policy_digest")
        if self.named_tool_contract_digest != NAMED_TOOL_CONTRACT_DIGEST:
            raise RetrievalContextError("named-tool contract is not accepted")
        if self.profile_id != NAMED_TOOL_PROFILE_ID:
            raise RetrievalContextError("named-tool profile is not accepted")
        times = (
            _parse_utc(self.query_valid_time, "context_query_valid_time"),
            _parse_utc(
                self.composition_serving_time, "composition_serving_time"
            ),
            _parse_utc(self.context_serving_time, "context_serving_time"),
        )
        if not times[0] <= times[1] <= times[2]:
            raise RetrievalContextError("context request times are not increasing")
        _require_token(
            self.composition_idempotency_key, "composition_idempotency_key"
        )
        if not isinstance(self.composition_receipt_bytes, bytes):
            raise RetrievalContextError("composition receipt must be bytes")
        if (
            not isinstance(self.composition_inputs, tuple)
            or len(self.composition_inputs) > 6
            or not all(
                isinstance(item, HybridCompositionInput)
                for item in self.composition_inputs
            )
        ):
            raise RetrievalContextError("composition inputs exceed inventory")
        authority = (
            self.authority_request_bytes,
            self.authority_execution_receipt_bytes,
            self.authority_receipt_bytes,
        )
        if any(item is None for item in authority) and any(
            item is not None for item in authority
        ):
            raise RetrievalContextError("authority bytes must be retained together")
        if any(
            item is not None and not isinstance(item, bytes)
            for item in authority
        ):
            raise RetrievalContextError("authority evidence must be bytes")
        if type(self.context_limit_bytes) is not int:
            raise RetrievalContextError("context limit must be an exact integer")
        if self.context_limit_bytes != CONTEXT_LIMIT_BYTES:
            raise RetrievalContextError("context limit is fixed")
        if type(self.authority_passage_limit) is not int:
            raise RetrievalContextError("passage limit must be an exact integer")
        if self.authority_passage_limit != AUTHORITY_PASSAGE_LIMIT:
            raise RetrievalContextError("passage limit is fixed")

    @property
    def authority_request(self) -> CollisionHydrationLookupToolRequest | None:
        if self.authority_request_bytes is None:
            return None
        try:
            decoded = decode_named_tool_json(self.authority_request_bytes)
        except Exception as exc:
            raise RetrievalContextError("authority request is invalid") from exc
        if not isinstance(decoded, CollisionHydrationLookupToolRequest):
            raise RetrievalContextError("authority request has wrong type")
        if named_request_bytes(decoded) != self.authority_request_bytes:
            raise RetrievalContextError("authority request bytes are not canonical")
        return decoded

    def canonical_value(self) -> dict[str, object]:
        def identity(raw: bytes | None) -> dict[str, object]:
            return {
                "digest": None if raw is None else _digest_bytes(raw),
                "bytes": 0 if raw is None else len(raw),
            }

        return {
            "schema_version": "newsroom.increment5.retrieval-context-request.v1",
            "request_id": self.request_id,
            "idempotency_key": self.idempotency_key,
            "actor_id": self.actor_id,
            "authenticated_principal_digest": self.authenticated_principal_digest,
            "purpose": self.purpose.value,
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "named_tool_contract_digest": self.named_tool_contract_digest,
            "profile_id": self.profile_id,
            "query_valid_time": self.query_valid_time,
            "composition_serving_time": self.composition_serving_time,
            "context_serving_time": self.context_serving_time,
            "composition_idempotency_key": self.composition_idempotency_key,
            "composition_receipt": identity(self.composition_receipt_bytes),
            "composition_inputs": [
                item.canonical_value() for item in self.composition_inputs
            ],
            "authority_request": identity(self.authority_request_bytes),
            "authority_execution_receipt": identity(
                self.authority_execution_receipt_bytes
            ),
            "authority_receipt": identity(self.authority_receipt_bytes),
            "context_contract_digest": RETRIEVAL_CONTEXT_CONTRACT_DIGEST,
            "context_limit_bytes": self.context_limit_bytes,
            "authority_passage_limit": self.authority_passage_limit,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical(self.canonical_value())

    @property
    def request_digest(self) -> str:
        return _digest_bytes(self.canonical_bytes)


def _evidence_digest(
    projection_evidence: Sequence[HybridManifestEntry],
    authority_evidence: RetrievalAuthorityEvidence | None,
    items: Sequence[HydratedContextItem],
    exclusions: Sequence[RetrievalContextExclusion],
    no_match: bool,
    truncated: bool,
) -> str:
    return _digest_bytes(
        _canonical(
            {
                "projection_evidence": [
                    item.canonical_value() for item in projection_evidence
                ],
                "authority_evidence": (
                    None
                    if authority_evidence is None
                    else authority_evidence.canonical_value()
                ),
                "items": [item.canonical_value() for item in items],
                "exclusions": [item.canonical_value() for item in exclusions],
                "no_match": no_match,
                "truncated": truncated,
            }
        )
    )


@dataclass(frozen=True, slots=True)
class RetrievalContextReceipt:
    context_id: str
    request_digest: str
    request_id: str
    actor_id: str
    authenticated_principal_digest: str
    purpose: HybridCompositionPurpose
    policy_id: str
    policy_digest: str
    named_tool_contract_digest: str
    profile_id: str
    query_valid_time: str
    composition_serving_time: str
    context_serving_time: str
    contract_digest: str
    composition_id: str | None
    composition_request_digest: str | None
    composition_receipt_digest: str
    composition_plan_context_digest: str | None
    composition_outcome: str | None
    composition_truncated: bool
    projection_evidence: tuple[HybridManifestEntry, ...]
    authority_evidence: RetrievalAuthorityEvidence | None
    hydrator_digest: str
    outcome: RetrievalContextOutcome
    reason: RetrievalContextReason | None
    items: tuple[HydratedContextItem, ...]
    exclusions: tuple[RetrievalContextExclusion, ...]
    total_composed_candidates: int
    known_omission_tools: tuple[NamedToolId, ...]
    no_match: bool
    truncated: bool
    context_limit_bytes: int = CONTEXT_LIMIT_BYTES
    authority_passage_limit: int = AUTHORITY_PASSAGE_LIMIT
    external_call_count: int = 0
    provider_call_count: int = 0
    model_call_count: int = 0
    embedding_call_count: int = 0
    provider_spend_micros: int = 0
    authority_effect: str = "NONE"
    candidate_created: bool = False
    hypothesis_created: bool = False
    source_content_instruction_effect: str = "NONE"
    qualification_authority_granted: bool = False
    production_activation_authorized: bool = False

    def __post_init__(self) -> None:
        _require_uuid(self.context_id, "context_id")
        _require_uuid(self.request_id, "context_request_id")
        _require_digest(self.request_digest, "context_request_digest")
        _require_token(self.actor_id, "context_actor_id")
        _require_digest(
            self.authenticated_principal_digest, "context_principal_digest"
        )
        if not isinstance(self.purpose, HybridCompositionPurpose):
            raise RetrievalContextError("context purpose must be typed")
        if self.policy_id != NAMED_TOOL_POLICY_ID:
            raise RetrievalContextError("context policy is not accepted")
        _require_digest(self.policy_digest, "context_policy_digest")
        if self.named_tool_contract_digest != NAMED_TOOL_CONTRACT_DIGEST:
            raise RetrievalContextError("context named-tool contract differs")
        if self.profile_id != NAMED_TOOL_PROFILE_ID:
            raise RetrievalContextError("context profile differs")
        times = (
            _parse_utc(self.query_valid_time, "context_query_valid_time"),
            _parse_utc(
                self.composition_serving_time, "composition_serving_time"
            ),
            _parse_utc(self.context_serving_time, "context_serving_time"),
        )
        if not times[0] <= times[1] <= times[2]:
            raise RetrievalContextError("context times are not increasing")
        if self.contract_digest != RETRIEVAL_CONTEXT_CONTRACT_DIGEST:
            raise RetrievalContextError("context contract differs")
        _require_digest(
            self.composition_receipt_digest, "composition_receipt_digest"
        )
        if self.composition_id is not None:
            _require_uuid(self.composition_id, "composition_id")
        for name in (
            "composition_request_digest",
            "composition_plan_context_digest",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_digest(value, name)
        if self.composition_outcome is not None:
            _require_token(self.composition_outcome, "composition_outcome")
        if type(self.composition_truncated) is not bool:
            raise RetrievalContextError("composition truncation must be boolean")
        if (
            not isinstance(self.projection_evidence, tuple)
            or not all(
                isinstance(item, HybridManifestEntry)
                for item in self.projection_evidence
            )
        ):
            raise RetrievalContextError("projection evidence must be typed")
        if self.projection_evidence and tuple(
            item.tool_id for item in self.projection_evidence
        ) != tuple(NamedToolId):
            raise RetrievalContextError("six-tool manifest is not retained")
        if self.authority_evidence is not None and not isinstance(
            self.authority_evidence, RetrievalAuthorityEvidence
        ):
            raise RetrievalContextError("authority evidence must be typed")
        _require_digest(self.hydrator_digest, "hydrator_digest")
        if not isinstance(self.outcome, RetrievalContextOutcome):
            raise RetrievalContextError("context outcome must be typed")
        if self.reason is not None and not isinstance(
            self.reason, RetrievalContextReason
        ):
            raise RetrievalContextError("context reason must be typed")
        if not isinstance(self.items, tuple) or not all(
            isinstance(item, HydratedContextItem) for item in self.items
        ):
            raise RetrievalContextError("context items must be typed")
        if tuple(item.context_rank for item in self.items) != tuple(
            range(1, len(self.items) + 1)
        ):
            raise RetrievalContextError("context item ranks are not contiguous")
        if not isinstance(self.exclusions, tuple) or not all(
            isinstance(item, RetrievalContextExclusion)
            for item in self.exclusions
        ):
            raise RetrievalContextError("context exclusions must be typed")
        exclusion_ranks = tuple(item.composition_rank for item in self.exclusions)
        if exclusion_ranks != tuple(sorted(set(exclusion_ranks))):
            raise RetrievalContextError("context exclusions are not canonical")
        if {item.dependency_root_id for item in self.items}.intersection(
            item.dependency_root_id for item in self.exclusions
        ):
            raise RetrievalContextError("retained and excluded roots overlap")
        _require_uint(
            self.total_composed_candidates, "total_composed_candidates"
        )
        if self.total_composed_candidates < len(self.items) + len(self.exclusions):
            raise RetrievalContextError("candidate total is inconsistent")
        expected_omissions = tuple(
            sorted(set(self.known_omission_tools), key=_TOOL_ORDER.__getitem__)
        )
        if self.known_omission_tools != expected_omissions:
            raise RetrievalContextError("known omissions are not canonical")
        if type(self.no_match) is not bool or type(self.truncated) is not bool:
            raise RetrievalContextError("context flags must be boolean")
        if self.truncated != (
            self.composition_truncated or bool(self.exclusions)
        ):
            raise RetrievalContextError("context truncation is inconsistent")
        if self.outcome is RetrievalContextOutcome.COMPLETE:
            if self.items:
                if self.no_match or self.reason is not None:
                    raise RetrievalContextError(
                        "positive complete context cannot state no-match"
                    )
            elif (
                not self.no_match
                or self.reason is not RetrievalContextReason.NO_MATCH
                or self.authority_evidence is None
                or self.authority_evidence.outcome != "COMPLETE"
                or self.authority_evidence.collision_state != "UNOCCUPIED"
            ):
                raise RetrievalContextError(
                    "empty complete context lacks current no-match proof"
                )
        elif self.outcome is RetrievalContextOutcome.DEGRADED:
            if (
                not self.items
                or self.no_match
                or self.reason
                is not RetrievalContextReason.OPTIONAL_EVIDENCE_NON_COMPLETE
            ):
                raise RetrievalContextError("degraded context is inconsistent")
        elif self.items or self.no_match or self.reason is None:
            raise RetrievalContextError(
                "non-success context must be empty with a reason"
            )
        if (
            type(self.context_limit_bytes) is not int
            or self.context_limit_bytes != CONTEXT_LIMIT_BYTES
            or type(self.authority_passage_limit) is not int
            or self.authority_passage_limit != AUTHORITY_PASSAGE_LIMIT
        ):
            raise RetrievalContextError("context bounds drifted")
        for name in (
            "external_call_count",
            "provider_call_count",
            "model_call_count",
            "embedding_call_count",
            "provider_spend_micros",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) != 0:
                raise RetrievalContextError("context reports external work")
        if self.authority_effect != "NONE":
            raise RetrievalContextError("context claims an authority effect")
        for name in (
            "candidate_created",
            "hypothesis_created",
            "qualification_authority_granted",
            "production_activation_authorized",
        ):
            if type(getattr(self, name)) is not bool or getattr(self, name):
                raise RetrievalContextError("context claims a forbidden effect")
        if self.source_content_instruction_effect != "NONE":
            raise RetrievalContextError("source content altered authority")
        expected_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "|".join(
                    (
                        self.request_digest,
                        self.outcome.value,
                        "NONE" if self.reason is None else self.reason.value,
                        _evidence_digest(
                            self.projection_evidence,
                            self.authority_evidence,
                            self.items,
                            self.exclusions,
                            self.no_match,
                            self.truncated,
                        ),
                    )
                ),
            )
        )
        if self.context_id != expected_id:
            raise RetrievalContextError("context identity differs from evidence")

    def canonical_value(self) -> dict[str, object]:
        return {
            "schema_version": "newsroom.increment5.retrieval-context-receipt.v1",
            "context_id": self.context_id,
            "request_digest": self.request_digest,
            "request_id": self.request_id,
            "actor_id": self.actor_id,
            "authenticated_principal_digest": self.authenticated_principal_digest,
            "purpose": self.purpose.value,
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "named_tool_contract_digest": self.named_tool_contract_digest,
            "profile_id": self.profile_id,
            "query_valid_time": self.query_valid_time,
            "composition_serving_time": self.composition_serving_time,
            "context_serving_time": self.context_serving_time,
            "contract_digest": self.contract_digest,
            "composition_id": self.composition_id,
            "composition_request_digest": self.composition_request_digest,
            "composition_receipt_digest": self.composition_receipt_digest,
            "composition_plan_context_digest": (
                self.composition_plan_context_digest
            ),
            "composition_outcome": self.composition_outcome,
            "composition_truncated": self.composition_truncated,
            "projection_evidence": [
                item.canonical_value() for item in self.projection_evidence
            ],
            "authority_evidence": (
                None
                if self.authority_evidence is None
                else self.authority_evidence.canonical_value()
            ),
            "hydrator_digest": self.hydrator_digest,
            "outcome": self.outcome.value,
            "reason": None if self.reason is None else self.reason.value,
            "items": [item.canonical_value() for item in self.items],
            "exclusions": [item.canonical_value() for item in self.exclusions],
            "total_composed_candidates": self.total_composed_candidates,
            "known_omission_tools": [
                item.value for item in self.known_omission_tools
            ],
            "no_match": self.no_match,
            "truncated": self.truncated,
            "context_limit_bytes": self.context_limit_bytes,
            "authority_passage_limit": self.authority_passage_limit,
            "external_call_count": self.external_call_count,
            "provider_call_count": self.provider_call_count,
            "model_call_count": self.model_call_count,
            "embedding_call_count": self.embedding_call_count,
            "provider_spend_micros": self.provider_spend_micros,
            "authority_effect": self.authority_effect,
            "candidate_created": self.candidate_created,
            "hypothesis_created": self.hypothesis_created,
            "source_content_instruction_effect": (
                self.source_content_instruction_effect
            ),
            "qualification_authority_granted": (
                self.qualification_authority_granted
            ),
            "production_activation_authorized": (
                self.production_activation_authorized
            ),
        }

    @property
    def canonical_bytes(self) -> bytes:
        raw = _canonical(self.canonical_value())
        if len(raw) > self.context_limit_bytes:
            raise RetrievalContextError("context exceeds fixed response bound")
        return raw

    @property
    def receipt_digest(self) -> str:
        return _digest_bytes(self.canonical_bytes)


_DERIVATIVE_IDENTITY_KEYS = frozenset(
    {"passage_id", "admission_id", "blob_digest", "text_digest"}
)
_DerivativeIdentity = tuple[str, str, str, str]
_PURGE_RECEIPT_SCHEMA_VERSION = "newsroom.increment5.retrieval-context-purge.v2"
_LEGACY_PURGE_TABLE_SHAPE = (
    ("idempotency_key", "TEXT", 0, None, 1),
    ("request_digest", "TEXT", 1, None, 0),
    ("prior_receipt_digest", "TEXT", 1, None, 0),
    ("purge_receipt_digest", "TEXT", 1, None, 0),
    ("purge_receipt_bytes", "BLOB", 1, None, 0),
)
_PURGE_TABLE_SHAPE = (
    ("purge_id", "TEXT", 0, None, 1),
    ("idempotency_key", "TEXT", 1, None, 0),
    ("request_digest", "TEXT", 1, None, 0),
    ("prior_receipt_digest", "TEXT", 1, None, 0),
    ("purge_receipt_digest", "TEXT", 1, None, 0),
    ("purge_receipt_bytes", "BLOB", 1, None, 0),
)


def _derivative_identity_value(
    identity: _DerivativeIdentity,
) -> dict[str, str]:
    return {
        "passage_id": identity[0],
        "admission_id": identity[1],
        "blob_digest": identity[2],
        "text_digest": identity[3],
    }


def _require_derivative_identities(
    value: object,
    field: str,
) -> tuple[_DerivativeIdentity, ...]:
    if not isinstance(value, tuple) or not value:
        raise RetrievalContextError(f"{field} differs")
    identities: list[_DerivativeIdentity] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 4:
            raise RetrievalContextError(f"{field} differs")
        identities.append(
            (
                _require_text(item[0], f"{field}_passage_id"),
                _require_text(item[1], f"{field}_admission_id"),
                _require_digest(item[2], f"{field}_blob_digest"),
                _require_digest(item[3], f"{field}_text_digest"),
            )
        )
    normalised = tuple(sorted(set(identities)))
    if tuple(value) != normalised:
        raise RetrievalContextError(f"{field} differs")
    return normalised


def _decode_derivative_identities(
    value: object,
    field: str,
) -> tuple[_DerivativeIdentity, ...]:
    if not isinstance(value, list) or not value:
        raise RetrievalContextError(f"{field} differs")
    identities: list[_DerivativeIdentity] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != _DERIVATIVE_IDENTITY_KEYS:
            raise RetrievalContextError(f"{field} differs")
        identities.append(
            (
                item["passage_id"],
                item["admission_id"],
                item["blob_digest"],
                item["text_digest"],
            )
        )
    decoded = _require_derivative_identities(tuple(identities), field)
    if value != [_derivative_identity_value(item) for item in decoded]:
        raise RetrievalContextError(f"{field} differs")
    return decoded


def _derivative_identity_columns(
    identities: tuple[_DerivativeIdentity, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(sorted({item[0] for item in identities})),
        tuple(sorted({item[1] for item in identities})),
        tuple(sorted({item[2] for item in identities})),
        tuple(sorted({item[3] for item in identities})),
    )


def _retrieval_context_purge_id(
    *,
    idempotency_key: str,
    context_id: str,
    request_digest: str,
    prior_receipt_digest: str,
    purged_derivative_identities: tuple[_DerivativeIdentity, ...],
    context_derivative_identities: tuple[_DerivativeIdentity, ...],
    reason_code: str,
    raw_context_bytes_deleted_in_event: bool,
) -> str:
    identity_digest = _digest_bytes(
        _canonical(
            {
                "purged_derivative_identities": [
                    _derivative_identity_value(item)
                    for item in purged_derivative_identities
                ],
                "context_derivative_identities": [
                    _derivative_identity_value(item)
                    for item in context_derivative_identities
                ],
                "raw_context_bytes_deleted_in_event": (
                    raw_context_bytes_deleted_in_event
                ),
            }
        )
    )
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            "|".join(
                (
                    idempotency_key,
                    context_id,
                    request_digest,
                    prior_receipt_digest,
                    identity_digest,
                    reason_code,
                )
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class RetrievalContextPurgeReceipt:
    purge_id: str
    idempotency_key: str
    context_id: str
    request_digest: str
    prior_receipt_digest: str
    passage_ids: tuple[str, ...]
    admission_ids: tuple[str, ...]
    blob_digests: tuple[str, ...]
    text_digests: tuple[str, ...]
    purged_derivative_identities: tuple[_DerivativeIdentity, ...]
    context_derivative_identities: tuple[_DerivativeIdentity, ...]
    reason_code: str
    raw_context_bytes_deleted_in_event: bool
    raw_context_bytes_absent: bool = True
    tombstone_retained: bool = True
    external_call_count: int = 0
    candidate_created: bool = False
    hypothesis_created: bool = False

    def __post_init__(self) -> None:
        _require_uuid(self.purge_id, "purge_id")
        _require_token(self.idempotency_key, "purge_idempotency_key")
        _require_uuid(self.context_id, "purged_context_id")
        _require_digest(self.request_digest, "purged_request_digest")
        _require_digest(self.prior_receipt_digest, "prior_receipt_digest")
        _sorted_unique_text(self.passage_ids, "purged_passage_id", allow_empty=False)
        _sorted_unique_text(
            self.admission_ids, "purged_admission_id", allow_empty=False
        )
        _sorted_unique_digests(
            self.blob_digests, "purged_blob_digest", allow_empty=False
        )
        _sorted_unique_digests(
            self.text_digests, "purged_text_digest", allow_empty=False
        )
        purged = _require_derivative_identities(
            self.purged_derivative_identities,
            "purged_derivative_identities",
        )
        context = _require_derivative_identities(
            self.context_derivative_identities,
            "context_derivative_identities",
        )
        if not set(purged) <= set(context) or (
            self.passage_ids,
            self.admission_ids,
            self.blob_digests,
            self.text_digests,
        ) != _derivative_identity_columns(purged):
            raise RetrievalContextError("purged derivative identity binding differs")
        _require_token(self.reason_code, "purge_reason_code")
        if (
            type(self.raw_context_bytes_deleted_in_event) is not bool
            or self.raw_context_bytes_absent is not True
            or self.tombstone_retained is not True
            or type(self.external_call_count) is not int
            or self.external_call_count != 0
            or type(self.candidate_created) is not bool
            or self.candidate_created
            or type(self.hypothesis_created) is not bool
            or self.hypothesis_created
        ):
            raise RetrievalContextError("purge receipt claims an invalid effect")
        expected_id = _retrieval_context_purge_id(
            idempotency_key=self.idempotency_key,
            context_id=self.context_id,
            request_digest=self.request_digest,
            prior_receipt_digest=self.prior_receipt_digest,
            purged_derivative_identities=purged,
            context_derivative_identities=context,
            reason_code=self.reason_code,
            raw_context_bytes_deleted_in_event=(
                self.raw_context_bytes_deleted_in_event
            ),
        )
        if self.purge_id != expected_id:
            raise RetrievalContextError("purge identity differs from evidence")

    def canonical_value(self) -> dict[str, object]:
        return {
            "schema_version": _PURGE_RECEIPT_SCHEMA_VERSION,
            "purge_id": self.purge_id,
            "idempotency_key": self.idempotency_key,
            "context_id": self.context_id,
            "request_digest": self.request_digest,
            "prior_receipt_digest": self.prior_receipt_digest,
            "passage_ids": list(self.passage_ids),
            "admission_ids": list(self.admission_ids),
            "blob_digests": list(self.blob_digests),
            "text_digests": list(self.text_digests),
            "purged_derivative_identities": [
                _derivative_identity_value(item)
                for item in self.purged_derivative_identities
            ],
            "context_derivative_identities": [
                _derivative_identity_value(item)
                for item in self.context_derivative_identities
            ],
            "reason_code": self.reason_code,
            "raw_context_bytes_deleted_in_event": (
                self.raw_context_bytes_deleted_in_event
            ),
            "raw_context_bytes_absent": self.raw_context_bytes_absent,
            "tombstone_retained": self.tombstone_retained,
            "external_call_count": self.external_call_count,
            "candidate_created": self.candidate_created,
            "hypothesis_created": self.hypothesis_created,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical(self.canonical_value())

    @property
    def receipt_digest(self) -> str:
        return _digest_bytes(self.canonical_bytes)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "RetrievalContextPurgeReceipt":
        value = _decode_canonical(raw, "retrieval context purge receipt")
        required = {
            "schema_version",
            "purge_id",
            "idempotency_key",
            "context_id",
            "request_digest",
            "prior_receipt_digest",
            "passage_ids",
            "admission_ids",
            "blob_digests",
            "text_digests",
            "purged_derivative_identities",
            "context_derivative_identities",
            "reason_code",
            "raw_context_bytes_deleted_in_event",
            "raw_context_bytes_absent",
            "tombstone_retained",
            "external_call_count",
            "candidate_created",
            "hypothesis_created",
        }
        if (
            set(value) != required
            or value["schema_version"] != _PURGE_RECEIPT_SCHEMA_VERSION
        ):
            raise RetrievalContextError("purge receipt keys differ")
        try:
            return cls(
                purge_id=value["purge_id"],
                idempotency_key=value["idempotency_key"],
                context_id=value["context_id"],
                request_digest=value["request_digest"],
                prior_receipt_digest=value["prior_receipt_digest"],
                passage_ids=tuple(value["passage_ids"]),
                admission_ids=tuple(value["admission_ids"]),
                blob_digests=tuple(value["blob_digests"]),
                text_digests=tuple(value["text_digests"]),
                purged_derivative_identities=_decode_derivative_identities(
                    value["purged_derivative_identities"],
                    "purged_derivative_identities",
                ),
                context_derivative_identities=_decode_derivative_identities(
                    value["context_derivative_identities"],
                    "context_derivative_identities",
                ),
                reason_code=value["reason_code"],
                raw_context_bytes_deleted_in_event=value[
                    "raw_context_bytes_deleted_in_event"
                ],
                raw_context_bytes_absent=value["raw_context_bytes_absent"],
                tombstone_retained=value["tombstone_retained"],
                external_call_count=value["external_call_count"],
                candidate_created=value["candidate_created"],
                hypothesis_created=value["hypothesis_created"],
            )
        except (KeyError, TypeError) as exc:
            raise RetrievalContextError("purge receipt values differ") from exc


def _purge_derivative_identities(
    raw: bytes,
) -> tuple[str, tuple[tuple[str, str, str, str], ...]]:
    value = _decode_canonical(raw, "retained retrieval context")
    context_id = value.get("context_id")
    items = value.get("items")
    if not isinstance(context_id, str) or not isinstance(items, list):
        raise RetrievalContextError("retained context purge evidence differs")
    identities: set[tuple[str, str, str, str]] = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("passage"), dict):
            raise RetrievalContextError("retained context purge evidence differs")
        passage = item["passage"]
        try:
            identities.add(
                (
                    _require_text(passage["passage_id"], "purged_passage_id"),
                    _require_text(
                        passage["admission_id"], "purged_admission_id"
                    ),
                    _require_digest(
                        passage["blob_digest"], "purged_blob_digest"
                    ),
                    _require_digest(
                        passage["text_digest"], "purged_text_digest"
                    ),
                )
            )
        except KeyError as exc:
            raise RetrievalContextError(
                "retained context purge evidence differs"
            ) from exc
    return _require_uuid(context_id, "purged_context_id"), tuple(sorted(identities))


def _retained_purge_receipt(
    row: tuple[object, ...],
) -> RetrievalContextPurgeReceipt:
    if len(row) != 6 or not isinstance(row[5], bytes):
        raise RetrievalContextError("retained purge receipt metadata differs")
    raw = bytes(row[5])
    if _digest_bytes(raw) != row[4]:
        raise RetrievalContextError("retained purge receipt is corrupt")
    purge = RetrievalContextPurgeReceipt.from_canonical_bytes(raw)
    if (
        purge.purge_id != row[0]
        or purge.idempotency_key != row[1]
        or purge.request_digest != row[2]
        or purge.prior_receipt_digest != row[3]
    ):
        raise RetrievalContextError("retained purge receipt metadata differs")
    return purge


class RetrievalContextJournal:
    """Immutable first-writer-wins journal with deterministic replay."""

    @staticmethod
    def _purge_table_shape(
        connection: sqlite3.Connection,
    ) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                row[1],
                str(row[2]).upper(),
                int(row[3]),
                row[4],
                int(row[5]),
            )
            for row in connection.execute(
                "PRAGMA table_info(increment5d2_retrieval_context_purges)"
            )
        )

    @staticmethod
    def _create_purge_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE increment5d2_retrieval_context_purges (
                purge_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                prior_receipt_digest TEXT NOT NULL,
                purge_receipt_digest TEXT NOT NULL,
                purge_receipt_bytes BLOB NOT NULL
            )
            """
        )

    @classmethod
    def _initialise_schema(cls, connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS increment5d2_retrieval_contexts (
                    idempotency_key TEXT PRIMARY KEY,
                    request_digest TEXT NOT NULL,
                    receipt_digest TEXT NOT NULL,
                    receipt_bytes BLOB NOT NULL
                )
                """
            )
            objects = connection.execute(
                "SELECT type FROM sqlite_master WHERE name=?",
                ("increment5d2_retrieval_context_purges",),
            ).fetchall()
            if not objects:
                cls._create_purge_table(connection)
            elif objects != [("table",)]:
                raise RetrievalContextError(
                    "retrieval context purge journal schema differs"
                )
            else:
                shape = cls._purge_table_shape(connection)
                if shape == _LEGACY_PURGE_TABLE_SHAPE:
                    retained = connection.execute(
                        "SELECT COUNT(*) "
                        "FROM increment5d2_retrieval_context_purges"
                    ).fetchone()
                    if retained != (0,):
                        raise RetrievalContextError(
                            "legacy purge journal lacks sibling identities"
                        )
                    connection.execute(
                        "DROP TABLE increment5d2_retrieval_context_purges"
                    )
                    cls._create_purge_table(connection)
                elif shape != _PURGE_TABLE_SHAPE:
                    raise RetrievalContextError(
                        "retrieval context purge journal schema differs"
                    )

            index_name = "increment5d2_retrieval_context_purges_by_key"
            indexes = {
                row[1]
                for row in connection.execute(
                    "PRAGMA index_list(increment5d2_retrieval_context_purges)"
                )
            }
            if index_name not in indexes:
                conflicting = connection.execute(
                    "SELECT type FROM sqlite_master WHERE name=?",
                    (index_name,),
                ).fetchall()
                if conflicting:
                    raise RetrievalContextError(
                        "retrieval context purge journal index differs"
                    )
                connection.execute(
                    """
                    CREATE INDEX increment5d2_retrieval_context_purges_by_key
                    ON increment5d2_retrieval_context_purges(
                        idempotency_key,purge_id
                    )
                    """
                )
            index_columns = tuple(
                row[2]
                for row in connection.execute(
                    "PRAGMA index_info("
                    "increment5d2_retrieval_context_purges_by_key)"
                )
            )
            if index_columns != ("idempotency_key", "purge_id"):
                raise RetrievalContextError(
                    "retrieval context purge journal index differs"
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with closing(self._connect()) as connection, connection:
            self._initialise_schema(connection)

    @staticmethod
    def _require_purge_safe_journal(connection: sqlite3.Connection) -> None:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
        if (
            journal_mode is None
            or len(journal_mode) != 1
            or str(journal_mode[0]).lower() != "delete"
        ):
            raise RetrievalContextError(
                "purge-safe SQLite journal mode is unavailable"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            self._require_purge_safe_journal(connection)
        except RetrievalContextError:
            connection.close()
            raise
        secure_delete = connection.execute("PRAGMA secure_delete=ON").fetchone()
        if secure_delete != (1,):
            connection.close()
            raise RetrievalContextError("secure context deletion is unavailable")
        return connection

    def execute(self, *, idempotency_key: str, request_digest: str, producer):
        _require_token(idempotency_key, "journal_idempotency_key")
        _require_digest(request_digest, "journal_request_digest")
        if not callable(producer):
            raise TypeError("journal producer must be callable")
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_purge_safe_journal(connection)

            def derivative_was_purged(
                *,
                derivative_identities: tuple[_DerivativeIdentity, ...],
            ) -> bool:
                selected = frozenset(
                    _require_derivative_identities(
                        derivative_identities,
                        "context_derivative_identities",
                    )
                )
                rows = connection.execute(
                    """
                    SELECT purge_id,idempotency_key,request_digest,
                           prior_receipt_digest,
                           purge_receipt_digest,purge_receipt_bytes
                    FROM increment5d2_retrieval_context_purges
                    ORDER BY purge_id
                    """
                )
                for retained_row in rows:
                    retained = _retained_purge_receipt(retained_row)
                    if selected.intersection(
                        retained.purged_derivative_identities
                    ):
                        return True
                return False

            purge_rows = connection.execute(
                """
                SELECT purge_id,idempotency_key,request_digest,
                       prior_receipt_digest,
                       purge_receipt_digest,purge_receipt_bytes
                FROM increment5d2_retrieval_context_purges
                WHERE idempotency_key=?
                ORDER BY purge_id
                """,
                (idempotency_key,),
            ).fetchall()
            if purge_rows:
                purges = tuple(
                    _retained_purge_receipt(item) for item in purge_rows
                )
                if any(
                    purge.idempotency_key != idempotency_key
                    or purge.request_digest != request_digest
                    for purge in purges
                ):
                    raise RetrievalContextError(
                        "purged idempotency key is bound to another request"
                    )
                raise RetrievalContextError("retrieval context was purged")
            row = connection.execute(
                """
                SELECT request_digest,receipt_digest,receipt_bytes
                FROM increment5d2_retrieval_contexts
                WHERE idempotency_key=?
                """,
                (idempotency_key,),
            ).fetchone()
            if row is not None:
                if row[0] != request_digest:
                    raise RetrievalContextError(
                        "idempotency key is bound to another request"
                    )
                raw = bytes(row[2])
                if _digest_bytes(raw) != row[1]:
                    raise RetrievalContextError("retained context is corrupt")
                expected = producer(derivative_was_purged)
                if (
                    not isinstance(expected, RetrievalContextReceipt)
                    or expected.request_digest != request_digest
                    or expected.canonical_bytes != raw
                ):
                    raise RetrievalContextError(
                        "retained context differs from deterministic replay"
                    )
                return expected
            receipt = producer(derivative_was_purged)
            if (
                not isinstance(receipt, RetrievalContextReceipt)
                or receipt.request_digest != request_digest
            ):
                raise RetrievalContextError("context producer binding differs")
            raw = receipt.canonical_bytes
            connection.execute(
                """
                INSERT INTO increment5d2_retrieval_contexts(
                    idempotency_key,request_digest,receipt_digest,receipt_bytes
                ) VALUES(?,?,?,?)
                """,
                (
                    idempotency_key,
                    request_digest,
                    receipt.receipt_digest,
                    raw,
                ),
            )
            connection.commit()
            return receipt

    def purge_affected(
        self,
        *,
        reason_code: str,
        passage_ids: tuple[str, ...] = (),
        admission_ids: tuple[str, ...] = (),
        blob_digests: tuple[str, ...] = (),
        text_digests: tuple[str, ...] = (),
    ) -> tuple[RetrievalContextPurgeReceipt, ...]:
        """Delete matching governed bytes and retain only exact purge tombstones."""

        _require_token(reason_code, "purge_reason_code")
        selected_passages = frozenset(
            _sorted_unique_text(passage_ids, "purge_passage_id")
        )
        selected_admissions = frozenset(
            _sorted_unique_text(admission_ids, "purge_admission_id")
        )
        selected_blobs = frozenset(
            _sorted_unique_digests(blob_digests, "purge_blob_digest")
        )
        selected_texts = frozenset(
            _sorted_unique_digests(text_digests, "purge_text_digest")
        )
        if not any(
            (
                selected_passages,
                selected_admissions,
                selected_blobs,
                selected_texts,
            )
        ):
            raise RetrievalContextError("purge requires an exact derivative identity")

        def selected(identity: _DerivativeIdentity) -> bool:
            return bool(
                identity[0] in selected_passages
                or identity[1] in selected_admissions
                or identity[2] in selected_blobs
                or identity[3] in selected_texts
            )

        retained: list[RetrievalContextPurgeReceipt] = []
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_purge_safe_journal(connection)
            inventories: dict[
                str,
                tuple[
                    str,
                    str,
                    str,
                    tuple[_DerivativeIdentity, ...],
                ],
            ] = {}
            purged_by_context: dict[str, set[_DerivativeIdentity]] = {}
            raw_context_keys: dict[str, str] = {}
            for row in connection.execute(
                """
                SELECT purge_id,idempotency_key,request_digest,
                       prior_receipt_digest,
                       purge_receipt_digest,purge_receipt_bytes
                FROM increment5d2_retrieval_context_purges
                ORDER BY purge_id
                """
            ):
                purge = _retained_purge_receipt(row)
                inventory = (
                    purge.idempotency_key,
                    purge.request_digest,
                    purge.prior_receipt_digest,
                    purge.context_derivative_identities,
                )
                previous = inventories.setdefault(purge.context_id, inventory)
                if previous != inventory:
                    raise RetrievalContextError(
                        "retained purge context inventory differs"
                    )
                purged_by_context.setdefault(purge.context_id, set()).update(
                    purge.purged_derivative_identities
                )
                if purge.reason_code == reason_code and any(
                    selected(item)
                    for item in purge.purged_derivative_identities
                ):
                    retained.append(purge)

            rows = connection.execute(
                """
                SELECT idempotency_key,request_digest,receipt_digest,receipt_bytes
                FROM increment5d2_retrieval_contexts
                ORDER BY idempotency_key
                """
            ).fetchall()
            for row in rows:
                raw = bytes(row[3])
                if _digest_bytes(raw) != row[2]:
                    raise RetrievalContextError("retained context is corrupt")
                context_id, identities = _purge_derivative_identities(raw)
                inventory = (row[0], row[1], row[2], identities)
                previous = inventories.setdefault(context_id, inventory)
                if previous != inventory or context_id in raw_context_keys:
                    raise RetrievalContextError(
                        "retained purge context inventory differs"
                    )
                if context_id in purged_by_context:
                    raise RetrievalContextError(
                        "purged retrieval context bytes remain retained"
                    )
                raw_context_keys[context_id] = row[0]

            for context_id in sorted(inventories):
                (
                    idempotency_key,
                    request_digest,
                    prior_receipt_digest,
                    identities,
                ) = inventories[context_id]
                matched = tuple(item for item in identities if selected(item))
                already_purged = purged_by_context.setdefault(context_id, set())
                newly_purged = tuple(
                    item for item in matched if item not in already_purged
                )
                if not newly_purged:
                    continue
                (
                    matched_passages,
                    matched_admissions,
                    matched_blobs,
                    matched_texts,
                ) = _derivative_identity_columns(newly_purged)
                raw_key = raw_context_keys.get(context_id)
                deleted_in_event = raw_key is not None
                purge_id = _retrieval_context_purge_id(
                    idempotency_key=idempotency_key,
                    context_id=context_id,
                    request_digest=request_digest,
                    prior_receipt_digest=prior_receipt_digest,
                    purged_derivative_identities=newly_purged,
                    context_derivative_identities=identities,
                    reason_code=reason_code,
                    raw_context_bytes_deleted_in_event=deleted_in_event,
                )
                purge = RetrievalContextPurgeReceipt(
                    purge_id=purge_id,
                    idempotency_key=idempotency_key,
                    context_id=context_id,
                    request_digest=request_digest,
                    prior_receipt_digest=prior_receipt_digest,
                    passage_ids=matched_passages,
                    admission_ids=matched_admissions,
                    blob_digests=matched_blobs,
                    text_digests=matched_texts,
                    purged_derivative_identities=newly_purged,
                    context_derivative_identities=identities,
                    reason_code=reason_code,
                    raw_context_bytes_deleted_in_event=deleted_in_event,
                )
                if raw_key is not None:
                    deleted = connection.execute(
                        "DELETE FROM increment5d2_retrieval_contexts "
                        "WHERE idempotency_key=?",
                        (raw_key,),
                    )
                    if deleted.rowcount != 1:
                        raise RetrievalContextError(
                            "retained context purge deletion differs"
                        )
                connection.execute(
                    """
                    INSERT INTO increment5d2_retrieval_context_purges(
                        purge_id,idempotency_key,request_digest,
                        prior_receipt_digest,
                        purge_receipt_digest,purge_receipt_bytes
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        purge.purge_id,
                        purge.idempotency_key,
                        purge.request_digest,
                        purge.prior_receipt_digest,
                        purge.receipt_digest,
                        purge.canonical_bytes,
                    ),
                )
                already_purged.update(newly_purged)
                retained.append(purge)
            connection.commit()
        return tuple(
            sorted(retained, key=lambda item: (item.context_id, item.purge_id))
        )


@dataclass(frozen=True, slots=True)
class _PlannedCandidate:
    candidate: HybridCandidate
    passage_id: str


def _candidate_passage(candidate: HybridCandidate) -> str | None:
    origins = sorted(
        (item for item in candidate.origins if item.passage_id is not None),
        key=lambda item: (
            item.rank,
            _MODE_ORDER[item.mode],
            item.passage_id or "",
            item.origin_digest,
        ),
    )
    return None if not origins else origins[0].passage_id


def _plan_candidates(composition: HybridCompositionReceipt):
    planned: list[_PlannedCandidate] = []
    exclusions: list[RetrievalContextExclusion] = []
    missing: list[str] = []
    passage_roots: dict[str, str] = {}
    for candidate in composition.candidates:
        passage_id = _candidate_passage(candidate)
        if passage_id is None:
            missing.append(candidate.dependency_root_id)
            continue
        prior_root = passage_roots.get(passage_id)
        if prior_root is not None and prior_root != candidate.dependency_root_id:
            raise RetrievalContextError("one passage crosses dependency roots")
        passage_roots[passage_id] = candidate.dependency_root_id
        if len(planned) >= AUTHORITY_PASSAGE_LIMIT:
            exclusions.append(
                RetrievalContextExclusion(
                    candidate.final_rank,
                    candidate.dependency_root_id,
                    RetrievalContextExclusionReason.AUTHORITY_RESULT_BOUND,
                    candidate.candidate_digest,
                )
            )
        else:
            planned.append(_PlannedCandidate(candidate, passage_id))
    return tuple(planned), tuple(exclusions), tuple(sorted(missing))


def _composition_request(
    request: RetrievalContextRequest, receipt: HybridCompositionReceipt
) -> HybridCompositionRequest:
    return HybridCompositionRequest(
        request_id=receipt.request_id,
        idempotency_key=request.composition_idempotency_key,
        actor_id=request.actor_id,
        authenticated_principal_digest=request.authenticated_principal_digest,
        purpose=request.purpose,
        policy_id=request.policy_id,
        policy_digest=request.policy_digest,
        named_tool_contract_digest=request.named_tool_contract_digest,
        profile_id=request.profile_id,
        query_valid_time=request.query_valid_time,
        serving_time=request.composition_serving_time,
        inputs=request.composition_inputs,
        reciprocal_rank_k=receipt.reciprocal_rank_k,
        candidate_limit=receipt.candidate_limit,
        response_limit_bytes=receipt.response_limit_bytes,
    )


def _composition_failure(outcome: HybridCompositionOutcome):
    return {
        HybridCompositionOutcome.POLICY_BLOCKED: (
            RetrievalContextOutcome.RIGHTS_BLOCKED,
            RetrievalContextReason.COMPOSITION_RIGHTS_BLOCKED,
        ),
        HybridCompositionOutcome.STALE: (
            RetrievalContextOutcome.STALE,
            RetrievalContextReason.COMPOSITION_STALE,
        ),
        HybridCompositionOutcome.UNAVAILABLE: (
            RetrievalContextOutcome.UNAVAILABLE,
            RetrievalContextReason.COMPOSITION_UNAVAILABLE,
        ),
    }.get(
        outcome,
        (
            RetrievalContextOutcome.INCOMPLETE,
            RetrievalContextReason.COMPOSITION_INCOMPLETE,
        ),
    )


def _authority_failure(
    execution: NamedAuthorityExecutionReceipt, raw: Mapping[str, object]
):
    if execution.outcome is NamedAuthorityExecutionOutcome.POLICY_BLOCKED:
        return (
            RetrievalContextOutcome.RIGHTS_BLOCKED,
            RetrievalContextReason.AUTHORITY_RIGHTS_BLOCKED,
        )
    if execution.outcome is NamedAuthorityExecutionOutcome.STALE:
        return (
            RetrievalContextOutcome.STALE,
            RetrievalContextReason.AUTHORITY_STALE,
        )
    if execution.outcome is NamedAuthorityExecutionOutcome.UNAVAILABLE:
        return (
            RetrievalContextOutcome.UNAVAILABLE,
            RetrievalContextReason.AUTHORITY_UNAVAILABLE,
        )
    if raw.get("reason") in {
        "RESULT_BOUND_EXCEEDED",
        "RESPONSE_LIMIT_EXCEEDED",
        "QUERY_TIMEOUT",
    }:
        return (
            RetrievalContextOutcome.BUDGET_BLOCKED,
            RetrievalContextReason.AUTHORITY_RESULT_BOUND,
        )
    return (
        RetrievalContextOutcome.INCOMPLETE,
        RetrievalContextReason.AUTHORITY_INCOMPLETE,
    )


class RetrievalContextBuilder:
    """Build one truthful bounded context from replayed composition and authority."""

    def __init__(
        self,
        *,
        composition_replayer: HybridComposer,
        journal: RetrievalContextJournal,
        hydrator: GovernedPassageHydrator,
    ) -> None:
        if not isinstance(composition_replayer, HybridComposer):
            raise TypeError("builder requires a typed 5D1 composer")
        if not isinstance(journal, RetrievalContextJournal):
            raise TypeError("builder requires a typed context journal")
        digest = getattr(hydrator, "implementation_digest", None)
        if not isinstance(digest, str):
            raise TypeError("builder requires an attributable hydrator")
        _require_digest(digest, "hydrator_implementation_digest")
        if not callable(getattr(hydrator, "read", None)):
            raise TypeError("hydrator must provide read()")
        self.composition_replayer = composition_replayer
        self.journal = journal
        self.hydrator = hydrator

    def execute(self, request: RetrievalContextRequest) -> RetrievalContextReceipt:
        if not isinstance(request, RetrievalContextRequest):
            raise TypeError("builder request must be typed")
        return self.journal.execute(
            idempotency_key=request.idempotency_key,
            request_digest=request.request_digest,
            producer=lambda purge_guard: self._produce(request, purge_guard),
        )

    def _produce(
        self, request: RetrievalContextRequest, purge_guard
    ) -> RetrievalContextReceipt:
        composition: HybridCompositionReceipt | None = None
        try:
            composition = HybridCompositionReceipt.from_canonical_bytes(
                request.composition_receipt_bytes
            )
            self._bind_composition(request, composition)
            replayed = self.composition_replayer.execute(
                _composition_request(request, composition)
            )
            if replayed.canonical_bytes != request.composition_receipt_bytes:
                raise RetrievalContextError("composition replay differs")
        except Exception:
            return self._receipt(
                request,
                composition,
                None,
                RetrievalContextOutcome.INTEGRITY_BLOCKED,
                RetrievalContextReason.COMPOSITION_REPLAY_MISMATCH,
            )

        if composition.outcome not in {
            HybridCompositionOutcome.COMPLETE,
            HybridCompositionOutcome.DEGRADED,
        }:
            outcome, reason = _composition_failure(composition.outcome)
            return self._receipt(request, composition, None, outcome, reason)

        try:
            planned, exclusions, missing = _plan_candidates(composition)
        except RetrievalContextError:
            return self._receipt(
                request,
                composition,
                None,
                RetrievalContextOutcome.INTEGRITY_BLOCKED,
                RetrievalContextReason.NO_AUTHORITATIVE_PASSAGE,
            )
        if missing:
            return self._receipt(
                request,
                composition,
                None,
                RetrievalContextOutcome.INCOMPLETE,
                RetrievalContextReason.NO_AUTHORITATIVE_PASSAGE,
            )
        if request.authority_request_bytes is None:
            return self._receipt(
                request,
                composition,
                None,
                RetrievalContextOutcome.INCOMPLETE,
                RetrievalContextReason.MISSING_AUTHORITY_EVIDENCE,
            )

        try:
            (
                execution,
                raw,
                authority,
                references,
            ) = self._validate_authority(request, composition, planned)
        except NamedAuthorityReceiptValidationError:
            return self._receipt(
                request,
                composition,
                None,
                RetrievalContextOutcome.INTEGRITY_BLOCKED,
                RetrievalContextReason.AUTHORITY_RECEIPT_INVALID,
            )
        except RetrievalContextError:
            return self._receipt(
                request,
                composition,
                None,
                RetrievalContextOutcome.INTEGRITY_BLOCKED,
                RetrievalContextReason.AUTHORITY_REQUEST_MISMATCH,
            )
        except Exception:
            return self._receipt(
                request,
                composition,
                None,
                RetrievalContextOutcome.INTEGRITY_BLOCKED,
                RetrievalContextReason.AUTHORITY_RECEIPT_INVALID,
            )

        if execution.outcome is not NamedAuthorityExecutionOutcome.COMPLETE:
            outcome, reason = _authority_failure(execution, raw)
            return self._receipt(
                request, composition, authority, outcome, reason
            )

        required_watermark = max(
            (item.authority_watermark or 0 for item in composition.manifest),
            default=0,
        )
        if authority.authority_watermark < required_watermark:
            return self._receipt(
                request,
                composition,
                authority,
                RetrievalContextOutcome.STALE,
                RetrievalContextReason.AUTHORITY_WATERMARK_STALE,
            )

        if composition.no_match:
            if authority.collision_state != "UNOCCUPIED":
                return self._receipt(
                    request,
                    composition,
                    authority,
                    RetrievalContextOutcome.INCOMPLETE,
                    RetrievalContextReason.COLLISION_CONTRADICTS_NO_MATCH,
                )
            return self._receipt(
                request,
                composition,
                authority,
                RetrievalContextOutcome.COMPLETE,
                RetrievalContextReason.NO_MATCH,
                no_match=True,
            )

        try:
            if purge_guard(
                derivative_identities=tuple(
                    sorted(
                        (
                            passage_id,
                            reference.admission_id,
                            reference.blob_digest,
                            reference.text_digest,
                        )
                        for passage_id, reference in references.items()
                    )
                ),
            ):
                return self._receipt(
                    request,
                    composition,
                    authority,
                    RetrievalContextOutcome.RIGHTS_BLOCKED,
                    RetrievalContextReason.RETAINED_CONTEXT_PURGED,
                )
            items = self._hydrate(request, composition, authority, planned, references)
        except GovernedBytesUnavailable:
            return self._receipt(
                request,
                composition,
                authority,
                RetrievalContextOutcome.UNAVAILABLE,
                RetrievalContextReason.GOVERNED_BYTES_UNAVAILABLE,
            )
        except (
            GovernedBytesIntegrityError,
            UnicodeDecodeError,
            RetrievalContextError,
        ):
            return self._receipt(
                request,
                composition,
                authority,
                RetrievalContextOutcome.INTEGRITY_BLOCKED,
                RetrievalContextReason.GOVERNED_BYTES_INTEGRITY,
            )

        outcome = (
            RetrievalContextOutcome.DEGRADED
            if composition.outcome is HybridCompositionOutcome.DEGRADED
            else RetrievalContextOutcome.COMPLETE
        )
        reason = (
            RetrievalContextReason.OPTIONAL_EVIDENCE_NON_COMPLETE
            if outcome is RetrievalContextOutcome.DEGRADED
            else None
        )
        mutable_items = list(items)
        mutable_exclusions = list(exclusions)
        while mutable_items:
            receipt = self._receipt(
                request,
                composition,
                authority,
                outcome,
                reason,
                items=tuple(mutable_items),
                exclusions=tuple(
                    sorted(mutable_exclusions, key=lambda item: item.composition_rank)
                ),
            )
            try:
                receipt.canonical_bytes
                return receipt
            except RetrievalContextError:
                item = mutable_items.pop()
                mutable_exclusions.append(
                    RetrievalContextExclusion(
                        item.composition_rank,
                        item.dependency_root_id,
                        RetrievalContextExclusionReason.CONTEXT_BYTE_BOUND,
                        item.composition_candidate_digest,
                    )
                )

        return self._receipt(
            request,
            composition,
            authority,
            RetrievalContextOutcome.BUDGET_BLOCKED,
            RetrievalContextReason.CONTEXT_BYTE_BOUND,
            exclusions=tuple(
                sorted(mutable_exclusions, key=lambda item: item.composition_rank)
            ),
        )

    @staticmethod
    def _bind_composition(
        request: RetrievalContextRequest, composition: HybridCompositionReceipt
    ) -> None:
        expected = (
            request.actor_id,
            request.authenticated_principal_digest,
            request.purpose,
            request.policy_id,
            request.policy_digest,
            request.named_tool_contract_digest,
            request.profile_id,
            request.query_valid_time,
            request.composition_serving_time,
        )
        actual = (
            composition.actor_id,
            composition.authenticated_principal_digest,
            composition.purpose,
            composition.policy_id,
            composition.policy_digest,
            composition.named_tool_contract_digest,
            composition.profile_id,
            composition.query_valid_time,
            composition.serving_time,
        )
        if actual != expected:
            raise RetrievalContextError("composition binding differs")

    def _validate_authority(
        self,
        request: RetrievalContextRequest,
        composition: HybridCompositionReceipt,
        planned: Sequence[_PlannedCandidate],
    ):
        authority_request = request.authority_request
        if authority_request is None:
            raise RetrievalContextError("authority request is missing")
        envelope = authority_request.envelope
        if envelope.tool_id is not (
            NamedToolId.CURRENT_COLLISION_AND_AUTHORITY_HYDRATION_LOOKUP
        ):
            raise RetrievalContextError("authority tool differs")
        if envelope.purpose not in {
            NamedToolPurpose.AUTHORITY_HYDRATION,
            NamedToolPurpose.COLLISION_CHECK,
            NamedToolPurpose.REPLAY_AUDIT,
        }:
            raise RetrievalContextError("authority purpose differs")
        if (
            envelope.actor_id != request.actor_id
            or envelope.authenticated_principal_digest
            != request.authenticated_principal_digest
            or envelope.policy_id != request.policy_id
            or envelope.policy_digest != request.policy_digest
            or envelope.contract_digest != request.named_tool_contract_digest
            or envelope.profile_id != request.profile_id
            or envelope.query_valid_time != request.query_valid_time
        ):
            raise RetrievalContextError("authority caller or contract differs")
        authority_time = _parse_utc(
            envelope.serving_time, "authority_serving_time"
        )
        if not (
            _parse_utc(request.composition_serving_time, "composition_serving_time")
            <= authority_time
            <= _parse_utc(request.context_serving_time, "context_serving_time")
        ):
            raise RetrievalContextError("authority time is outside context")
        expected_passages = tuple(sorted(item.passage_id for item in planned))
        if authority_request.passage_ids != expected_passages:
            raise RetrievalContextError("authority passage plan differs")
        if composition.candidates:
            if authority_request.authority_object_ids:
                raise RetrievalContextError("positive plan added authority objects")
        elif not authority_request.authority_object_ids:
            raise RetrievalContextError("empty plan lacks authority object")
        if authority_request.collision_key_digest != context_collision_key_digest(
            composition
        ):
            raise RetrievalContextError("collision key differs")
        if (
            request.authority_execution_receipt_bytes is None
            or request.authority_receipt_bytes is None
            or request.authority_request_bytes is None
        ):
            raise RetrievalContextError("authority evidence is incomplete")
        execution = NamedAuthorityExecutionReceipt.from_canonical_bytes(
            request.authority_execution_receipt_bytes
        )
        validate_named_authority_receipt(
            request=authority_request,
            execution_receipt=execution,
            raw_receipt_bytes=request.authority_receipt_bytes,
        )
        raw = _decode_canonical(request.authority_receipt_bytes, "authority_receipt")
        authority = RetrievalAuthorityEvidence(
            tool_request_digest=authority_request.request_digest,
            named_request_bytes_digest=_digest_bytes(
                request.authority_request_bytes
            ),
            execution_receipt_digest=execution.receipt_digest,
            raw_receipt_digest=_digest_bytes(request.authority_receipt_bytes),
            adapter_contract_digest=raw["adapter_contract_digest"],
            adapter_config_digest=raw["adapter_config_digest"],
            authority_scope_id=raw["authority_scope_id"],
            authority_watermark=raw["authority_watermark"],
            collision_namespace=raw["collision_namespace"],
            collision_key_digest=raw["collision_key_digest"],
            collision_state=raw["collision_state"],
            candidate_id=raw["candidate_id"],
            requested_object_ids=tuple(
                sorted(authority_request.authority_object_ids)
            ),
            requested_passage_ids=tuple(
                sorted(authority_request.passage_ids)
            ),
            query_valid_time=raw["query_valid_time"],
            serving_time=raw["serving_time"],
            outcome=raw["outcome"],
            reason=raw["reason"],
        )
        if (
            authority.tool_request_digest != execution.tool_request_digest
            or authority.query_valid_time != request.query_valid_time
            or authority.serving_time != envelope.serving_time
        ):
            raise RetrievalContextError("authority evidence binding differs")
        raw_passages = raw["passages"]
        if not isinstance(raw_passages, list) or not all(
            isinstance(item, dict) for item in raw_passages
        ):
            raise RetrievalContextError("authority passages are not objects")
        references = {
            item["passage_id"]: GovernedPassageReference.from_authority(item)
            for item in raw_passages
        }
        if len(references) != len(raw_passages):
            raise RetrievalContextError("authority passages duplicate")
        if (
            execution.outcome is NamedAuthorityExecutionOutcome.COMPLETE
            and tuple(sorted(references)) != expected_passages
        ):
            raise RetrievalContextError("authority result differs from plan")
        return execution, raw, authority, references

    def _hydrate(
        self,
        request: RetrievalContextRequest,
        composition: HybridCompositionReceipt,
        authority: RetrievalAuthorityEvidence,
        planned: Sequence[_PlannedCandidate],
        references: Mapping[str, GovernedPassageReference],
    ) -> tuple[HydratedContextItem, ...]:
        items: list[HydratedContextItem] = []
        for context_rank, plan in enumerate(planned, start=1):
            reference = references.get(plan.passage_id)
            if reference is None:
                raise RetrievalContextError("planned passage lacks authority")
            raw = self.hydrator.read(reference)
            text = raw.decode("utf-8", errors="strict")
            origins = tuple(
                item
                for item in plan.candidate.origins
                if item.passage_id == plan.passage_id
            )
            if not origins:
                raise RetrievalContextError("passage lacks composed origin")
            items.append(
                HydratedContextItem(
                    context_rank=context_rank,
                    composition_rank=plan.candidate.final_rank,
                    dependency_root_id=plan.candidate.dependency_root_id,
                    composition_candidate_digest=plan.candidate.candidate_digest,
                    precedence=plan.candidate.precedence.value,
                    score_numerator=plan.candidate.score.numerator,
                    score_denominator=plan.candidate.score.denominator,
                    contributing_modes=plan.candidate.contributing_modes,
                    all_origin_digests=tuple(
                        sorted(
                            item.origin_digest
                            for item in plan.candidate.origins
                        )
                    ),
                    passage_origin_digests=tuple(
                        sorted(item.origin_digest for item in origins)
                    ),
                    provenance_digests=tuple(
                        sorted({item.provenance_digest for item in origins})
                    ),
                    trust_scopes=tuple(
                        sorted(
                            {
                                item.trust_scope
                                for item in origins
                                if item.trust_scope is not None
                            }
                        )
                    ),
                    passage=reference,
                    text=text,
                    text_bytes=len(raw),
                    query_valid_time=request.query_valid_time,
                    composition_serving_time=request.composition_serving_time,
                    authority_serving_time=authority.serving_time,
                    context_serving_time=request.context_serving_time,
                )
            )
        return tuple(items)

    def _receipt(
        self,
        request: RetrievalContextRequest,
        composition: HybridCompositionReceipt | None,
        authority: RetrievalAuthorityEvidence | None,
        outcome: RetrievalContextOutcome,
        reason: RetrievalContextReason | None,
        *,
        items: tuple[HydratedContextItem, ...] = (),
        exclusions: tuple[RetrievalContextExclusion, ...] = (),
        no_match: bool = False,
    ) -> RetrievalContextReceipt:
        projection = () if composition is None else composition.manifest
        truncated = (
            False
            if composition is None
            else composition.truncated or bool(exclusions)
        )
        context_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "|".join(
                    (
                        request.request_digest,
                        outcome.value,
                        "NONE" if reason is None else reason.value,
                        _evidence_digest(
                            projection,
                            authority,
                            items,
                            exclusions,
                            no_match,
                            truncated,
                        ),
                    )
                ),
            )
        )
        return RetrievalContextReceipt(
            context_id=context_id,
            request_digest=request.request_digest,
            request_id=request.request_id,
            actor_id=request.actor_id,
            authenticated_principal_digest=(
                request.authenticated_principal_digest
            ),
            purpose=request.purpose,
            policy_id=request.policy_id,
            policy_digest=request.policy_digest,
            named_tool_contract_digest=request.named_tool_contract_digest,
            profile_id=request.profile_id,
            query_valid_time=request.query_valid_time,
            composition_serving_time=request.composition_serving_time,
            context_serving_time=request.context_serving_time,
            contract_digest=RETRIEVAL_CONTEXT_CONTRACT_DIGEST,
            composition_id=(
                None if composition is None else composition.composition_id
            ),
            composition_request_digest=(
                None if composition is None else composition.request_digest
            ),
            composition_receipt_digest=_digest_bytes(
                request.composition_receipt_bytes
            ),
            composition_plan_context_digest=(
                None
                if composition is None
                else composition.plan_context_digest
            ),
            composition_outcome=(
                None if composition is None else composition.outcome.value
            ),
            composition_truncated=(
                False if composition is None else composition.truncated
            ),
            projection_evidence=projection,
            authority_evidence=authority,
            hydrator_digest=self.hydrator.implementation_digest,
            outcome=outcome,
            reason=reason,
            items=items,
            exclusions=exclusions,
            total_composed_candidates=(
                0 if composition is None else len(composition.candidates)
            ),
            known_omission_tools=(
                ()
                if composition is None
                else composition.known_omission_tools
            ),
            no_match=no_match,
            truncated=truncated,
        )


__all__ = [
    "AUTHORITY_PASSAGE_LIMIT",
    "CONTEXT_LIMIT_BYTES",
    "GOVERNED_CAS_HYDRATOR_CONTRACT_DIGEST",
    "RETRIEVAL_CONTEXT_CONTRACT_DIGEST",
    "GovernedBytesIntegrityError",
    "GovernedBytesUnavailable",
    "GovernedCasPassageHydrator",
    "GovernedPassageHydrator",
    "GovernedPassageReference",
    "HydratedContextItem",
    "RetrievalAuthorityEvidence",
    "RetrievalContextBuilder",
    "RetrievalContextError",
    "RetrievalContextExclusion",
    "RetrievalContextExclusionReason",
    "RetrievalContextJournal",
    "RetrievalContextOutcome",
    "RetrievalContextPurgeReceipt",
    "RetrievalContextReason",
    "RetrievalContextReceipt",
    "RetrievalContextRequest",
    "context_collision_key_digest",
    "named_request_bytes",
]
