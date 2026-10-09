"""Native governed passage full-text and vector retrieval.

This is the non-fixture document seam for Increment 5.  The Neo4j state is a
rebuildable projection: exact passage bytes, embedding bytes and provider
accounting remain governed-object authority and the typed command event binds
the immutable document manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from newsroom.authority import AuthenticationProof, GovernedObjects
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
    validate_sha256_digest,
)
from newsroom.authority.models import ObjectAdmissionPayload, SemanticCommand
from newsroom.authority.persistence import AuthorityCommands, AuthorityEvents
from newsroom.authority.objects import HydrationRequest, ObjectAdmissionRequest
from newsroom.authority.types import AggregateId, ObjectAdmissionId, TrustScope, UtcTimestamp
from newsroom.extraction.models import ExtractionRunRequest
from newsroom.extraction.types import ExtractionOutcome, ExtractionPassageId
from newsroom.authority.extraction_facade import GovernedExtractionRecords
from .fulltext_contracts import (
    FULLTEXT_ACTOR_ID, FULLTEXT_COMPONENT_DIGEST, FULLTEXT_POLICY_ID,
    FULLTEXT_PURPOSE, NORMALIZATION_COMPONENT_DIGEST, FullTextAuthorityView,
    FullTextBranchRequest, FullTextDocumentBinding, FullTextLanguageMode,
    FullTextProjectionSnapshot,
    FullTextCorpusScope,
)
from .branch_contracts import (
    BRANCH_RESULT_LIMIT, BRANCH_TIMEOUT_MS, EXACT_BRANCH_ACTOR_ID,
    EXACT_BRANCH_POLICY_ID, EXACT_BRANCH_PURPOSE, BranchMode, BranchOutcome,
    BranchRequestId, ExactBranchRequest, ExactLookupKind,
)
from .branch_receipts import ExactBranchReceipt
from .fulltext_receipts import FullTextBranchReceipt
from .decision import INCREMENT_5A_CONTRACT_DIGEST
from .fulltext_retriever import FullTextRetriever
from .exact_retriever import SQLiteExactRetriever
from newsroom.increment4.neo4j import Increment4Neo4jActiveReadRequest, Increment4Neo4jController
from newsroom.projection.models import ProjectionGenerationState
from newsroom.projection.neo4j.models import (
    StructuralReadAuthoritySelection, StructuralReadResponse,
)


NATIVE_DOCUMENT_ADMISSION_TYPE = "retrieval.native-document"
NATIVE_VECTOR_ADMISSION_TYPE = "retrieval.native-vector"
NATIVE_EMBEDDING_RECEIPT_ADMISSION_TYPE = "retrieval.native-embedding-receipt"
NATIVE_CONTEXT_ADMISSION_TYPE = "retrieval.native-context"
NATIVE_DOCUMENT_CLASS = "NATIVE_RETRIEVAL_DOCUMENT"
NATIVE_DOCUMENT_USE = "RETRIEVAL_PROJECTION"
NATIVE_VECTOR_CLASS = "NATIVE_RETRIEVAL_EMBEDDING_VECTOR"
NATIVE_VECTOR_USE = "RETRIEVAL_VECTOR"
NATIVE_EMBEDDING_RECEIPT_CLASS = "NATIVE_RETRIEVAL_EMBEDDING_RECEIPT"
NATIVE_EMBEDDING_RECEIPT_USE = "RETRIEVAL_ACCOUNTING"
NATIVE_CONTEXT_CLASS = "NATIVE_RETRIEVAL_CONTEXT"
NATIVE_CONTEXT_USE = "TRIAGE_RETRIEVAL"
NATIVE_DOCUMENT_COMMAND = "retrieval.native_document.admit"
NATIVE_DOCUMENT_EVENT = "retrieval.native_document.admitted"
NATIVE_CONTEXT_COMMAND = "retrieval.native_context.admit"
NATIVE_CONTEXT_EVENT = "retrieval.native_context.admitted"
NATIVE_SECURITY_SCOPE = "authority.retrieval"
NATIVE_RETENTION_SCOPE = "authority.retrieval.retained"
NATIVE_DOCUMENT_SCHEMA = "newsroom.increment5.native-retrieval-document.v1"
NATIVE_EMBEDDING_SCHEMA = "newsroom.increment5.native-embedding-receipt.v1"
NATIVE_VECTOR_DIMENSIONS = 1_024
NATIVE_RESULT_LIMIT = 8
NATIVE_GRAPH_ROOT_LIMIT = 64
NATIVE_CONTEXT_DOCUMENT_LIMIT = NATIVE_RESULT_LIMIT * 2 + 1
NATIVE_VECTOR_PROFILE = "native-governed-vector-v1"
NATIVE_VECTOR_RECEIPT_SCHEMA = "newsroom.increment5.native-vector-branch-receipt.v1"
NATIVE_CONTEXT_SCHEMA = "newsroom.increment5.native-retrieval-context.v1"
NATIVE_CONTEXT_RECEIPT_SCHEMA = "newsroom.increment5.native-retrieval-context-receipt.v1"
NATIVE_GRAPH_RECEIPT_SCHEMA = "newsroom.increment5.native-graph-branch-receipt.v1"
NATIVE_CONTEXT_ADMISSION_TYPE = "retrieval.native-context"
NATIVE_CONTEXT_CLASS = "NATIVE_RETRIEVAL_CONTEXT"
NATIVE_CONTEXT_USE = "TRIAGE_RETRIEVAL"
NATIVE_CONTEXT_COMMAND = "retrieval.native_context.admit"
NATIVE_CONTEXT_EVENT = "retrieval.native_context.admitted"

_INDEX = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}\Z")


class NativeRetrievalError(RuntimeError):
    """Native retrieval input or retained authority is inconsistent."""


class NativeRetrievalHold(NativeRetrievalError):
    """A mandatory native branch has no usable governed input."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _digest(value: str, field: str) -> str:
    try:
        return validate_sha256_digest(value, field=field)
    except (TypeError, ValueError) as exc:
        raise NativeRetrievalError(f"{field} differs") from exc


def _text(value: object, field: str, maximum: int = 512) -> str:
    if type(value) is not str or not value or value != value.strip() or len(value.encode()) > maximum:
        raise NativeRetrievalError(f"{field} differs")
    return value


def _json(raw: bytes, schema: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise NativeRetrievalError("governed retrieval object is malformed") from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw or value.get("schema_identity") != schema:
        raise NativeRetrievalError("governed retrieval object is non-canonical")
    return value


@dataclass(frozen=True, slots=True)
class NativeEmbeddingReceipt:
    input_text_digest: str
    vector_digest: str
    dimensions: int
    provider: str
    model: str
    model_digest: str
    provider_request_id: str
    usage_receipt_digest: str
    recorded_at: str
    outcome: str = "COMPLETE"

    def __post_init__(self) -> None:
        for name in ("input_text_digest", "vector_digest", "model_digest", "usage_receipt_digest"):
            _digest(getattr(self, name), name)
        for name in ("provider", "model", "provider_request_id"):
            _text(getattr(self, name), name)
        if self.dimensions != NATIVE_VECTOR_DIMENSIONS:
            raise NativeRetrievalError("native embedding dimensions differ")
        if self.outcome != "COMPLETE":
            raise NativeRetrievalHold("EMBEDDING_NOT_COMPLETE")
        UtcTimestamp.parse(self.recorded_at)

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes({"schema_identity": NATIVE_EMBEDDING_SCHEMA, **self.canonical_value()})

    def canonical_value(self) -> dict[str, object]:
        return {
            "input_text_digest": self.input_text_digest,
            "vector_digest": self.vector_digest,
            "dimensions": self.dimensions,
            "provider": self.provider,
            "model": self.model,
            "model_digest": self.model_digest,
            "provider_request_id": self.provider_request_id,
            "usage_receipt_digest": self.usage_receipt_digest,
            "recorded_at": self.recorded_at,
            "outcome": self.outcome,
        }

    @classmethod
    def from_bytes(cls, raw: bytes) -> "NativeEmbeddingReceipt":
        value = _json(raw, NATIVE_EMBEDDING_SCHEMA)
        value.pop("schema_identity")
        try:
            return cls(**value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise NativeRetrievalError("embedding receipt fields differ") from exc


@dataclass(frozen=True, slots=True)
class NativeEmbeddingReference:
    vector_admission_id: ObjectAdmissionId
    receipt_admission_id: ObjectAdmissionId

    def __post_init__(self) -> None:
        if type(self.vector_admission_id) is not ObjectAdmissionId or type(self.receipt_admission_id) is not ObjectAdmissionId:
            raise NativeRetrievalError("embedding references must be governed admissions")


@dataclass(frozen=True, slots=True)
class NativePassageDocument:
    generation_id: str
    passage_id: str
    dependency_root_id: str
    source_id: str
    revision_id: str
    representation_id: str
    language: str
    text: str
    text_digest: str
    rights_digest: str
    provenance_digest: str
    vector_digest: str
    vector_admission_id: str
    embedding_receipt_digest: str
    embedding_receipt_admission_id: str
    embedding_model_digest: str

    def __post_init__(self) -> None:
        for name in ("generation_id", "passage_id", "dependency_root_id", "source_id", "revision_id", "representation_id", "language", "vector_admission_id", "embedding_receipt_admission_id"):
            _text(getattr(self, name), name)
        for name in ("text_digest", "rights_digest", "provenance_digest", "vector_digest", "embedding_receipt_digest", "embedding_model_digest"):
            _digest(getattr(self, name), name)
        if digest_bytes(self.text.encode("utf-8")) != self.text_digest:
            raise NativeRetrievalError("native passage text digest differs")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes({"schema_identity": NATIVE_DOCUMENT_SCHEMA, **self.projection_value()})

    def projection_value(self) -> dict[str, object]:
        return {
            "generation_id": self.generation_id,
            "passage_id": self.passage_id,
            "dependency_root_id": self.dependency_root_id,
            "source_id": self.source_id,
            "revision_id": self.revision_id,
            "representation_id": self.representation_id,
            "language": self.language,
            "text": self.text,
            "text_digest": self.text_digest,
            "rights_digest": self.rights_digest,
            "provenance_digest": self.provenance_digest,
            "vector_digest": self.vector_digest,
            "vector_admission_id": self.vector_admission_id,
            "embedding_receipt_digest": self.embedding_receipt_digest,
            "embedding_receipt_admission_id": self.embedding_receipt_admission_id,
            "embedding_model_digest": self.embedding_model_digest,
        }

    @property
    def digest(self) -> str:
        return digest_bytes(self.canonical_bytes)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "NativePassageDocument":
        value = _json(raw, NATIVE_DOCUMENT_SCHEMA)
        value.pop("schema_identity")
        try:
            return cls(**value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise NativeRetrievalError("native document fields differ") from exc


@dataclass(frozen=True, slots=True)
class NativeDocumentRequest:
    extraction_request: ExtractionRunRequest
    passage_id: ExtractionPassageId
    dependency_root_id: str
    generation_id: str
    embedding: NativeEmbeddingReference
    aggregate_id: AggregateId
    expected_aggregate_version: int
    idempotency_key: str

    def __post_init__(self) -> None:
        if type(self.extraction_request) is not ExtractionRunRequest or type(self.passage_id) is not ExtractionPassageId:
            raise NativeRetrievalError("native document requires exact extraction authority input")
        if type(self.embedding) is not NativeEmbeddingReference or type(self.aggregate_id) is not AggregateId:
            raise NativeRetrievalError("native document authority identity differs")
        _text(self.dependency_root_id, "dependency_root_id")
        _text(self.generation_id, "generation_id")
        _text(self.idempotency_key, "idempotency_key", 256)
        if type(self.expected_aggregate_version) is not int or self.expected_aggregate_version < 0:
            raise NativeRetrievalError("expected aggregate version differs")


@dataclass(frozen=True, slots=True)
class NativeDocumentReceipt:
    event_id: str
    command_id: str
    aggregate_id: AggregateId
    aggregate_version: int
    admission_id: ObjectAdmissionId
    document_digest: str
    vector_admission_id: ObjectAdmissionId
    embedding_receipt_admission_id: ObjectAdmissionId

    def __post_init__(self) -> None:
        _text(self.event_id, "native_document_event_id")
        _text(self.command_id, "native_document_command_id")
        if type(self.aggregate_id) is not AggregateId or type(self.admission_id) is not ObjectAdmissionId or type(self.vector_admission_id) is not ObjectAdmissionId or type(self.embedding_receipt_admission_id) is not ObjectAdmissionId:
            raise NativeRetrievalError("native document receipt identity differs")
        if type(self.aggregate_version) is not int or self.aggregate_version <= 0:
            raise NativeRetrievalError("native document aggregate version differs")
        _digest(self.document_digest, "native_document_digest")

    def projection_value(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "command_id": self.command_id,
            "aggregate_id": str(self.aggregate_id),
            "aggregate_version": self.aggregate_version,
            "admission_id": str(self.admission_id),
            "document_digest": self.document_digest,
            "vector_admission_id": str(self.vector_admission_id),
            "embedding_receipt_admission_id": str(
                self.embedding_receipt_admission_id
            ),
        }

    @classmethod
    def from_projection(cls, value: Mapping[str, object]) -> "NativeDocumentReceipt":
        expected = {
            "event_id",
            "command_id",
            "aggregate_id",
            "aggregate_version",
            "admission_id",
            "document_digest",
            "vector_admission_id",
            "embedding_receipt_admission_id",
        }
        if set(value) != expected:
            raise NativeRetrievalError("native projection receipt differs")
        try:
            return cls(
                str(value["event_id"]),
                str(value["command_id"]),
                AggregateId.parse(str(value["aggregate_id"])),
                value["aggregate_version"],  # type: ignore[arg-type]
                ObjectAdmissionId.parse(str(value["admission_id"])),
                str(value["document_digest"]),
                ObjectAdmissionId.parse(str(value["vector_admission_id"])),
                ObjectAdmissionId.parse(str(value["embedding_receipt_admission_id"])),
            )
        except (TypeError, ValueError) as exc:
            raise NativeRetrievalError("native projection receipt differs") from exc


@dataclass(frozen=True, slots=True)
class NativeRetrievalHit:
    passage_id: str
    dependency_root_id: str
    score: float

    def __post_init__(self) -> None:
        _text(self.passage_id, "hit_passage_id")
        _text(self.dependency_root_id, "hit_dependency_root_id")
        if type(self.score) is not float or not math.isfinite(self.score):
            raise NativeRetrievalError("native retrieval score differs")


@dataclass(frozen=True, slots=True)
class NativeRetrievalResult:
    fulltext_hits: tuple[NativeRetrievalHit, ...]
    vector_hits: tuple[NativeRetrievalHit, ...]
    generation_id: str
    outcome: str = "COMPLETE"

    def __post_init__(self) -> None:
        if (
            type(self.fulltext_hits) is not tuple
            or type(self.vector_hits) is not tuple
            or any(type(item) is not NativeRetrievalHit for item in self.fulltext_hits + self.vector_hits)
        ):
            raise NativeRetrievalError("native retrieval hits differ")
        _text(self.generation_id, "native_retrieval_generation_id")
        if self.outcome != "COMPLETE":
            raise NativeRetrievalError("native retrieval outcome differs")

    @property
    def no_match(self) -> bool:
        return not self.fulltext_hits and not self.vector_hits


@dataclass(frozen=True, slots=True)
class NativeVectorRequest:
    request_id: str
    idempotency_key: str
    query_event_id: str
    query_document_digest: str
    generation_id: str
    query_valid_time: str
    serving_time: str

    def __post_init__(self) -> None:
        try:
            if str(uuid.UUID(self.request_id)) != self.request_id:
                raise ValueError
        except (TypeError, ValueError, AttributeError) as exc:
            raise NativeRetrievalError("native vector request identity differs") from exc
        for name in ("idempotency_key", "query_event_id", "generation_id"):
            _text(getattr(self, name), name)
        _digest(self.query_document_digest, "query_document_digest")
        if UtcTimestamp.parse(self.query_valid_time).value > UtcTimestamp.parse(self.serving_time).value:
            raise NativeRetrievalError("native vector request time differs")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes({
            "schema_identity": "newsroom.increment5.native-vector-request.v1",
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
        })

    @property
    def request_digest(self) -> str:
        return digest_bytes(self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class NativeVectorBranchHit:
    rank: int
    passage_id: str
    dependency_root_id: str
    source_revision_id: str
    document_digest: str
    rights_digest: str
    provenance_digest: str
    raw_score_ppm: int

    def __post_init__(self) -> None:
        if type(self.rank) is not int or not 1 <= self.rank <= NATIVE_RESULT_LIMIT:
            raise NativeRetrievalError("native vector rank differs")
        for name in ("passage_id", "dependency_root_id", "source_revision_id"):
            _text(getattr(self, name), name)
        for name in ("document_digest", "rights_digest", "provenance_digest"):
            _digest(getattr(self, name), name)
        if type(self.raw_score_ppm) is not int or not 0 <= self.raw_score_ppm <= 1_000_000:
            raise NativeRetrievalError("native vector score differs")

    def canonical_value(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class NativeVectorBranchReceipt:
    receipt_id: str
    request_digest: str
    mode: BranchMode
    outcome: BranchOutcome
    reason: str | None
    generation_id: str
    generation_digest: str
    profile_id: str
    query_valid_time: str
    serving_time: str
    hits: tuple[NativeVectorBranchHit, ...]
    authority_read_count: int
    external_call_count: int = 0
    provider_call_count: int = 0
    model_call_count: int = 0
    embedding_call_count: int = 0
    provider_spend_micros: int = 0
    read_only: bool = True
    authority_effect: str = "NONE"

    def __post_init__(self) -> None:
        try:
            if str(uuid.UUID(self.receipt_id)) != self.receipt_id:
                raise ValueError
        except (TypeError, ValueError, AttributeError) as exc:
            raise NativeRetrievalError("native vector receipt identity differs") from exc
        _digest(self.request_digest, "native_vector_request_digest")
        if self.mode is not BranchMode.VECTOR or type(self.outcome) is not BranchOutcome:
            raise NativeRetrievalError("native vector receipt mode/outcome differs")
        _text(self.generation_id, "native_vector_generation_id")
        _digest(self.generation_digest, "native_vector_generation_digest")
        if self.profile_id != NATIVE_VECTOR_PROFILE:
            raise NativeRetrievalError("native vector profile differs")
        UtcTimestamp.parse(self.query_valid_time)
        UtcTimestamp.parse(self.serving_time)
        if UtcTimestamp.parse(self.query_valid_time).value > UtcTimestamp.parse(self.serving_time).value:
            raise NativeRetrievalError("native vector receipt time differs")
        if type(self.hits) is not tuple or tuple(hit.rank for hit in self.hits) != tuple(range(1, len(self.hits) + 1)):
            raise NativeRetrievalError("native vector hits differ")
        if self.outcome is BranchOutcome.COMPLETE:
            if (not self.hits) != (self.reason == "NO_MATCH"):
                raise NativeRetrievalError("native vector complete result differs")
        elif self.hits:
            raise NativeRetrievalError("native vector non-complete result has hits")
        if self.authority_read_count < 1 or any(getattr(self, name) != 0 for name in ("external_call_count", "provider_call_count", "model_call_count", "embedding_call_count", "provider_spend_micros")) or not self.read_only or self.authority_effect != "NONE":
            raise NativeRetrievalError("native vector receipt effects differ")

    def canonical_value(self) -> dict[str, object]:
        return {
            "schema_version": NATIVE_VECTOR_RECEIPT_SCHEMA,
            "receipt_id": self.receipt_id,
            "request_digest": self.request_digest,
            "mode": self.mode.value,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "generation_id": self.generation_id,
            "generation_digest": self.generation_digest,
            "profile_id": self.profile_id,
            "query_valid_time": self.query_valid_time,
            "serving_time": self.serving_time,
            "hits": [hit.canonical_value() for hit in self.hits],
            "authority_read_count": self.authority_read_count,
            "external_call_count": self.external_call_count,
            "provider_call_count": self.provider_call_count,
            "model_call_count": self.model_call_count,
            "embedding_call_count": self.embedding_call_count,
            "provider_spend_micros": self.provider_spend_micros,
            "read_only": self.read_only,
            "authority_effect": self.authority_effect,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_value())

    @property
    def receipt_digest(self) -> str:
        return digest_bytes(self.canonical_bytes)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "NativeVectorBranchReceipt":
        try:
            value = json.loads(raw)
            if type(value) is not dict or canonical_json_bytes(value) != raw or value.pop("schema_version", None) != NATIVE_VECTOR_RECEIPT_SCHEMA:
                raise ValueError
            value["mode"] = BranchMode(value["mode"])
            value["outcome"] = BranchOutcome(value["outcome"])
            value["hits"] = tuple(NativeVectorBranchHit(**item) for item in value["hits"])
            receipt = cls(**value)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeRetrievalError("native vector receipt fields differ") from exc
        if receipt.canonical_bytes != raw:
            raise NativeRetrievalError("native vector receipt is non-canonical")
        return receipt


@dataclass(frozen=True, slots=True)
class NativeGraphBranchReceipt:
    """Exact retained view of the authority-selected Increment 4 graph."""

    receipt_id: str
    request_digest: str
    requested_ids: tuple[str, ...]
    family_id: str
    family_definition_version: str
    projector_version: str
    ontology_contract_digest: str
    mapping_contract_digest: str
    generation_id: str
    authority_selection: str
    watermark_seq: int
    query_valid_time: str
    serving_time: str
    nodes: tuple[dict[str, object], ...]
    relations: tuple[dict[str, object], ...]
    outcome: str = "COMPLETE"
    reason: str | None = None

    def __post_init__(self) -> None:
        try:
            if str(uuid.UUID(self.receipt_id)) != self.receipt_id:
                raise ValueError
        except (TypeError, ValueError, AttributeError) as exc:
            raise NativeRetrievalError("native graph receipt identity differs") from exc
        _digest(self.request_digest, "native graph request digest")
        for name in ("family_id", "family_definition_version", "projector_version", "generation_id"):
            _text(getattr(self, name), name)
        for name in ("ontology_contract_digest", "mapping_contract_digest"):
            _digest(getattr(self, name), name)
        if self.authority_selection != StructuralReadAuthoritySelection.AUTHORITY_SELECTED_ACTIVE.value:
            raise NativeRetrievalError("native graph authority selection differs")
        if type(self.watermark_seq) is not int or self.watermark_seq < 0:
            raise NativeRetrievalError("native graph watermark differs")
        if type(self.requested_ids) is not tuple or not self.requested_ids or len(self.requested_ids) > NATIVE_GRAPH_ROOT_LIMIT or len(set(self.requested_ids)) != len(self.requested_ids):
            raise NativeRetrievalError("native graph request inventory differs")
        for item in self.requested_ids:
            _text(item, "native graph requested id")
        if type(self.nodes) is not tuple or type(self.relations) is not tuple or any(type(item) is not dict for item in self.nodes + self.relations):
            raise NativeRetrievalError("native graph evidence differs")
        if UtcTimestamp.parse(self.query_valid_time).value > UtcTimestamp.parse(self.serving_time).value:
            raise NativeRetrievalError("native graph time differs")
        if self.outcome != "COMPLETE" or self.reason not in {None, "NO_MATCH"} or (not self.nodes and not self.relations) != (self.reason == "NO_MATCH"):
            raise NativeRetrievalError("native graph outcome differs")

    @property
    def hits(self) -> tuple[dict[str, object], ...]:
        return self.nodes + self.relations

    def canonical_value(self) -> dict[str, object]:
        return {
            "schema_identity": NATIVE_GRAPH_RECEIPT_SCHEMA,
            **{name: getattr(self, name) for name in (
                "receipt_id", "request_digest", "family_id",
                "family_definition_version", "projector_version",
                "ontology_contract_digest", "mapping_contract_digest",
                "generation_id", "authority_selection", "watermark_seq",
                "query_valid_time", "serving_time", "outcome", "reason",
            )},
            "requested_ids": list(self.requested_ids),
            "nodes": list(self.nodes), "relations": list(self.relations),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_value())

    @property
    def receipt_digest(self) -> str:
        return digest_bytes(self.canonical_bytes)

    @classmethod
    def from_response(cls, request_digest: str, requested_ids: tuple[str, ...], response: StructuralReadResponse) -> "NativeGraphBranchReceipt":
        if type(response) is not StructuralReadResponse:
            raise NativeRetrievalError("native graph response differs")
        metadata = response.metadata
        if metadata.generation_state is not ProjectionGenerationState.ACTIVE or metadata.trust_scope is not TrustScope.ADMITTED or metadata.open_gap_count or metadata.dead_letter_count:
            raise NativeRetrievalHold("NATIVE_GRAPH_NOT_COMPLETE")
        nodes = tuple({
            "canonical_id": item.canonical_id, "node_type": item.node_type.value,
            "identity_source": item.identity_source,
            "identity_reference_digest": item.identity_reference_digest,
            "first_ledger_seq": item.first_ledger_seq,
            "first_source_event_id": item.first_source_event_id,
            "first_source_event_digest": item.first_source_event_digest,
        } for item in response.nodes)
        relations = tuple({
            "relation_key": item.relation_key, "relation_type": item.relation_type.value,
            "source_canonical_id": item.source_canonical_id,
            "target_canonical_id": item.target_canonical_id,
            "ledger_seq": item.ledger_seq, "source_event_id": item.source_event_id,
            "source_event_type": item.source_event_type,
            "source_event_digest": item.source_event_digest,
            "aggregate_type": item.aggregate_type, "aggregate_id": item.aggregate_id,
            "aggregate_version": item.aggregate_version, "payload_id": item.payload_id,
            "payload_digest": item.payload_digest,
            "object_admission_id": item.object_admission_id,
            "principal_id": item.principal_id, "trust_scope": item.trust_scope.value,
            "security_scope": item.security_scope,
            "retention_scope": item.retention_scope,
            "recorded_at": item.recorded_at.to_text(),
        } for item in response.relations)
        semantic = canonical_json_bytes({
            "request_digest": request_digest, "generation": str(metadata.generation_id),
            "nodes": nodes, "relations": relations,
        })
        return cls(
            str(uuid.uuid5(uuid.NAMESPACE_URL, digest_bytes(semantic))),
            request_digest, requested_ids, metadata.family_id,
            metadata.family_definition_version, metadata.projector_version,
            metadata.ontology_contract_digest, metadata.mapping_contract_digest,
            str(metadata.generation_id), metadata.authority_selection.value,
            metadata.contiguous_ledger_seq, metadata.query_valid_time.to_text(),
            metadata.serving_time.to_text(), nodes, relations,
            reason=None if nodes or relations else "NO_MATCH",
        )

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "NativeGraphBranchReceipt":
        value = _json(raw, NATIVE_GRAPH_RECEIPT_SCHEMA)
        value.pop("schema_identity")
        try:
            value["requested_ids"] = tuple(value["requested_ids"])
            value["nodes"] = tuple(value["nodes"])
            value["relations"] = tuple(value["relations"])
            receipt = cls(**value)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeRetrievalError("native graph receipt fields differ") from exc
        if receipt.canonical_bytes != raw:
            raise NativeRetrievalError("native graph receipt is non-canonical")
        return receipt


@dataclass(frozen=True, slots=True)
class NativeRetrievalContextRequest:
    """Stable server request joining the four mandatory native branches."""

    request_id: str
    idempotency_key: str
    aggregate_id: AggregateId
    expected_aggregate_version: int
    lead_id: str
    lead_digest: str
    authority_scope_id: str
    rights_inventory_digest: str
    exact_receipt_bytes: bytes
    fulltext_receipt_bytes: bytes
    vector_receipt_bytes: bytes
    graph_receipt_bytes: bytes
    selected_documents: tuple[NativeDocumentReceipt, ...]

    def __post_init__(self) -> None:
        try:
            if str(uuid.UUID(self.request_id)) != self.request_id:
                raise ValueError
        except (TypeError, ValueError, AttributeError) as exc:
            raise NativeRetrievalError("native context request identity differs") from exc
        _text(self.idempotency_key, "native context idempotency key", 256)
        if type(self.aggregate_id) is not AggregateId or type(self.expected_aggregate_version) is not int or self.expected_aggregate_version < 0:
            raise NativeRetrievalError("native context aggregate differs")
        _text(self.lead_id, "native context lead id")
        _digest(self.lead_digest, "native context lead digest")
        _text(self.authority_scope_id, "native context authority scope")
        _digest(self.rights_inventory_digest, "native context rights inventory")
        self.branch_receipts()
        if type(self.selected_documents) is not tuple or not self.selected_documents or len(self.selected_documents) > NATIVE_CONTEXT_DOCUMENT_LIMIT or any(type(item) is not NativeDocumentReceipt for item in self.selected_documents):
            raise NativeRetrievalError("native context selected documents differ")
        if len({item.event_id for item in self.selected_documents}) != len(self.selected_documents):
            raise NativeRetrievalError("native context selected documents repeat")

    def branch_receipts(self) -> tuple[ExactBranchReceipt, FullTextBranchReceipt, NativeVectorBranchReceipt, NativeGraphBranchReceipt]:
        try:
            values = (
                ExactBranchReceipt.from_canonical_bytes(self.exact_receipt_bytes),
                FullTextBranchReceipt.from_canonical_bytes(self.fulltext_receipt_bytes),
                NativeVectorBranchReceipt.from_canonical_bytes(self.vector_receipt_bytes),
                NativeGraphBranchReceipt.from_canonical_bytes(self.graph_receipt_bytes),
            )
        except Exception as exc:
            raise NativeRetrievalError("native context branch receipt differs") from exc
        if any(receipt.outcome is not BranchOutcome.COMPLETE for receipt in values[:3]) or values[3].outcome != "COMPLETE":
            raise NativeRetrievalHold("MANDATORY_RETRIEVAL_BRANCH_NOT_COMPLETE")
        return values

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes({
            "schema_identity": "newsroom.increment5.native-retrieval-context-request.v1",
            "request_id": self.request_id,
            "idempotency_key": self.idempotency_key,
            "aggregate_id": str(self.aggregate_id),
            "expected_aggregate_version": self.expected_aggregate_version,
            "lead_id": self.lead_id,
            "lead_digest": self.lead_digest,
            "authority_scope_id": self.authority_scope_id,
            "rights_inventory_digest": self.rights_inventory_digest,
            "branch_receipts": {
                "exact": json.loads(self.exact_receipt_bytes),
                "fulltext": json.loads(self.fulltext_receipt_bytes),
                "vector": json.loads(self.vector_receipt_bytes),
                "admitted_graph": json.loads(self.graph_receipt_bytes),
            },
            "selected_documents": [item.projection_value() for item in self.selected_documents],
        })

    @property
    def request_digest(self) -> str:
        return digest_bytes(self.canonical_bytes)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "NativeRetrievalContextRequest":
        value = _json(raw, "newsroom.increment5.native-retrieval-context-request.v1")
        try:
            branches = value["branch_receipts"]
            request = cls(
                request_id=value["request_id"], idempotency_key=value["idempotency_key"],
                aggregate_id=AggregateId.parse(value["aggregate_id"]),
                expected_aggregate_version=value["expected_aggregate_version"],
                lead_id=value["lead_id"], lead_digest=value["lead_digest"],
                authority_scope_id=value["authority_scope_id"],
                rights_inventory_digest=value["rights_inventory_digest"],
                exact_receipt_bytes=canonical_json_bytes(branches["exact"]),
                fulltext_receipt_bytes=canonical_json_bytes(branches["fulltext"]),
                vector_receipt_bytes=canonical_json_bytes(branches["vector"]),
                graph_receipt_bytes=canonical_json_bytes(branches["admitted_graph"]),
                selected_documents=tuple(
                    NativeDocumentReceipt.from_projection(item)
                    for item in value["selected_documents"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeRetrievalError("native context request fields differ") from exc
        if request.canonical_bytes != raw:
            raise NativeRetrievalError("native context request is non-canonical")
        return request


@dataclass(frozen=True, slots=True)
class NativeRetrievalContext:
    context_id: str
    request_id: str
    request_digest: str
    lead_id: str
    lead_digest: str
    authority_scope_id: str
    rights_inventory_digest: str
    generation_id: str
    query_valid_time: str
    serving_time: str
    branch_digests: tuple[str, str, str, str]
    branch_receipts: tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]
    selected_documents: tuple[dict[str, object], ...]
    outcome: str
    no_match: bool

    def __post_init__(self) -> None:
        try:
            if str(uuid.UUID(self.context_id)) != self.context_id or str(uuid.UUID(self.request_id)) != self.request_id:
                raise ValueError
        except (TypeError, ValueError, AttributeError) as exc:
            raise NativeRetrievalError("native context identity differs") from exc
        _digest(self.request_digest, "native context request digest")
        _text(self.lead_id, "native context lead id")
        _digest(self.lead_digest, "native context lead digest")
        _text(self.authority_scope_id, "native context authority scope")
        _digest(self.rights_inventory_digest, "native context rights inventory")
        _text(self.generation_id, "native context generation")
        if UtcTimestamp.parse(self.query_valid_time).value > UtcTimestamp.parse(self.serving_time).value:
            raise NativeRetrievalError("native context time differs")
        if type(self.branch_digests) is not tuple or len(self.branch_digests) != 4:
            raise NativeRetrievalError("native context branch inventory differs")
        for value in self.branch_digests:
            _digest(value, "native context branch digest")
        if type(self.branch_receipts) is not tuple or len(self.branch_receipts) != 4 or any(type(item) is not dict for item in self.branch_receipts):
            raise NativeRetrievalError("native context branch receipts differ")
        raw = tuple(canonical_json_bytes(item) for item in self.branch_receipts)
        if tuple(digest_bytes(item) for item in raw) != self.branch_digests:
            raise NativeRetrievalError("native context branch receipt digest differs")
        try:
            exact = ExactBranchReceipt.from_canonical_bytes(raw[0])
            fulltext = FullTextBranchReceipt.from_canonical_bytes(raw[1])
            vector = NativeVectorBranchReceipt.from_canonical_bytes(raw[2])
            graph = NativeGraphBranchReceipt.from_canonical_bytes(raw[3])
        except Exception as exc:
            raise NativeRetrievalError("native context branch receipt differs") from exc
        if (
            any(item.outcome is not BranchOutcome.COMPLETE for item in (exact, fulltext, vector))
            or graph.outcome != "COMPLETE"
            or fulltext.snapshot is None
            or str(fulltext.snapshot.generation_id) != self.generation_id
            or vector.generation_id != self.generation_id
            or vector.query_valid_time != self.query_valid_time
            or vector.serving_time != self.serving_time
            or UtcTimestamp.parse(graph.query_valid_time).value != UtcTimestamp.parse(self.query_valid_time).value
            or UtcTimestamp.parse(graph.serving_time).value != UtcTimestamp.parse(self.serving_time).value
            or self.no_match != (not (exact.hits or fulltext.hits or vector.hits or graph.hits))
        ):
            raise NativeRetrievalError("native context branch semantics differ")
        if type(self.selected_documents) is not tuple or any(type(item) is not dict for item in self.selected_documents):
            raise NativeRetrievalError("native context selected inventory differs")
        try:
            selected = tuple(NativeDocumentReceipt.from_projection(item) for item in self.selected_documents)
        except (TypeError, ValueError) as exc:
            raise NativeRetrievalError("native context selected inventory differs") from exc
        if len({item.event_id for item in selected}) != len(selected):
            raise NativeRetrievalError("native context selected inventory repeats")
        selected_digests = {item.document_digest for item in selected}
        if not selected or not {hit.document_digest for hit in vector.hits}.issubset(selected_digests):
            raise NativeRetrievalError("native context selected inventory differs")
        if self.outcome != "COMPLETE" or type(self.no_match) is not bool:
            raise NativeRetrievalError("native context outcome differs")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes({"schema_identity": NATIVE_CONTEXT_SCHEMA, **self.canonical_value()})

    @property
    def graph_generation_id(self) -> str:
        """Return the independently governed Increment 4 generation."""
        return NativeGraphBranchReceipt.from_canonical_bytes(
            canonical_json_bytes(self.branch_receipts[3])
        ).generation_id

    def canonical_value(self) -> dict[str, object]:
        return {
            "context_id": self.context_id, "request_id": self.request_id,
            "request_digest": self.request_digest, "lead_id": self.lead_id,
            "lead_digest": self.lead_digest, "branch_digests": list(self.branch_digests),
            "authority_scope_id": self.authority_scope_id,
            "rights_inventory_digest": self.rights_inventory_digest,
            "generation_id": self.generation_id,
            "graph_generation_id": self.graph_generation_id,
            "query_valid_time": self.query_valid_time,
            "serving_time": self.serving_time,
            "branch_receipts": list(self.branch_receipts),
            "selected_documents": list(self.selected_documents), "outcome": self.outcome,
            "no_match": self.no_match,
        }

    @property
    def digest(self) -> str:
        return digest_bytes(self.canonical_bytes)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "NativeRetrievalContext":
        value = _json(raw, NATIVE_CONTEXT_SCHEMA)
        value.pop("schema_identity")
        try:
            graph_generation_id = value.pop("graph_generation_id")
            value["branch_digests"] = tuple(value["branch_digests"])
            value["branch_receipts"] = tuple(value["branch_receipts"])
            value["selected_documents"] = tuple(value["selected_documents"])
            context = cls(**value)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeRetrievalError("native context fields differ") from exc
        if context.graph_generation_id != graph_generation_id:
            raise NativeRetrievalError("native graph generation differs")
        return context


@dataclass(frozen=True, slots=True)
class NativeRetrievalContextReceipt:
    context_id: str
    request_id: str
    request_digest: str
    aggregate_id: AggregateId
    aggregate_version: int
    event_id: str
    command_id: str
    admission_id: ObjectAdmissionId
    context_object_digest: str
    authority_scope_id: str
    rights_inventory_digest: str
    generation_id: str
    query_valid_time: str
    serving_time: str
    exact_receipt_bytes: bytes
    fulltext_receipt_bytes: bytes
    vector_receipt_bytes: bytes
    graph_receipt_bytes: bytes
    controller_principal_id: str
    authority_domain: str
    outcome: str = "COMPLETE"
    reason: None = None
    no_match: bool = False

    def __post_init__(self) -> None:
        try:
            if str(uuid.UUID(self.context_id)) != self.context_id or str(uuid.UUID(self.request_id)) != self.request_id:
                raise ValueError
        except (TypeError, ValueError, AttributeError) as exc:
            raise NativeRetrievalError("native context receipt identity differs") from exc
        _digest(self.request_digest, "native context receipt request digest")
        if type(self.aggregate_id) is not AggregateId or type(self.aggregate_version) is not int or self.aggregate_version <= 0 or type(self.admission_id) is not ObjectAdmissionId:
            raise NativeRetrievalError("native context authority receipt differs")
        for name in ("event_id", "command_id", "controller_principal_id", "authority_domain"):
            _text(getattr(self, name), name)
        _digest(self.context_object_digest, "native context object digest")
        _text(self.authority_scope_id, "native context receipt authority scope")
        _digest(self.rights_inventory_digest, "native context receipt rights inventory")
        _text(self.generation_id, "native context receipt generation")
        if UtcTimestamp.parse(self.query_valid_time).value > UtcTimestamp.parse(self.serving_time).value:
            raise NativeRetrievalError("native context receipt time differs")
        branches = self.branch_receipts()
        if any(item.outcome is not BranchOutcome.COMPLETE for item in branches[:3]) or branches[3].outcome != "COMPLETE":
            raise NativeRetrievalError("native context receipt branch is incomplete")
        if self.reason is not None:
            raise NativeRetrievalError("complete native context has a reason")

    def canonical_value(self) -> dict[str, object]:
        return {
            "schema_identity": NATIVE_CONTEXT_RECEIPT_SCHEMA,
            "context_id": self.context_id, "request_id": self.request_id,
            "request_digest": self.request_digest, "aggregate_id": str(self.aggregate_id),
            "aggregate_version": self.aggregate_version, "event_id": self.event_id,
            "command_id": self.command_id, "admission_id": str(self.admission_id),
            "context_object_digest": self.context_object_digest,
            "authority_scope_id": self.authority_scope_id,
            "rights_inventory_digest": self.rights_inventory_digest,
            "generation_id": self.generation_id,
            "graph_generation_id": self.graph_generation_id,
            "query_valid_time": self.query_valid_time,
            "serving_time": self.serving_time,
            "branch_receipts": {
                "exact": json.loads(self.exact_receipt_bytes),
                "fulltext": json.loads(self.fulltext_receipt_bytes),
                "vector": json.loads(self.vector_receipt_bytes),
                "admitted_graph": json.loads(self.graph_receipt_bytes),
            },
            "controller_principal_id": self.controller_principal_id,
            "authority_domain": self.authority_domain, "outcome": self.outcome,
            "reason": self.reason, "no_match": self.no_match,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_value())

    @property
    def graph_generation_id(self) -> str:
        """Return the independently governed Increment 4 generation."""
        return self.branch_receipts()[3].generation_id

    @property
    def receipt_digest(self) -> str:
        return digest_bytes(self.canonical_bytes)

    def branch_receipts(self) -> tuple[ExactBranchReceipt, FullTextBranchReceipt, NativeVectorBranchReceipt, NativeGraphBranchReceipt]:
        try:
            return (
                ExactBranchReceipt.from_canonical_bytes(self.exact_receipt_bytes),
                FullTextBranchReceipt.from_canonical_bytes(self.fulltext_receipt_bytes),
                NativeVectorBranchReceipt.from_canonical_bytes(self.vector_receipt_bytes),
                NativeGraphBranchReceipt.from_canonical_bytes(self.graph_receipt_bytes),
            )
        except Exception as exc:
            raise NativeRetrievalError("native context receipt branch differs") from exc

    @classmethod
    def from_bytes(cls, raw: bytes) -> "NativeRetrievalContextReceipt":
        value = _json(raw, NATIVE_CONTEXT_RECEIPT_SCHEMA)
        value.pop("schema_identity")
        try:
            branches = value.pop("branch_receipts")
            graph_generation_id = value.pop("graph_generation_id")
            value["exact_receipt_bytes"] = canonical_json_bytes(branches["exact"])
            value["fulltext_receipt_bytes"] = canonical_json_bytes(branches["fulltext"])
            value["vector_receipt_bytes"] = canonical_json_bytes(branches["vector"])
            value["graph_receipt_bytes"] = canonical_json_bytes(branches["admitted_graph"])
            value["aggregate_id"] = AggregateId.parse(value["aggregate_id"])
            value["admission_id"] = ObjectAdmissionId.parse(value["admission_id"])
            receipt = cls(**value)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeRetrievalError("native context receipt fields differ") from exc
        if receipt.canonical_bytes != raw:
            raise NativeRetrievalError("native context receipt is non-canonical")
        if receipt.graph_generation_id != graph_generation_id:
            raise NativeRetrievalError("native graph generation differs")
        return receipt


class NativeDocumentProjection(Protocol):
    def upsert(self, receipt: NativeDocumentReceipt, document: NativePassageDocument, vector: tuple[float, ...]) -> None: ...
    def reconcile_membership(self, receipts: tuple[NativeDocumentReceipt, ...]) -> tuple[NativeDocumentReceipt, ...]: ...
    def retrieve(self, *, query_text: str, query_vector: tuple[float, ...]) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]: ...
    def retrieve_vector(self, *, query_vector: tuple[float, ...]) -> tuple[Mapping[str, object], ...]: ...


_CONTEXT_READ_PORT_TOKEN = object()
_DOCUMENT_INVENTORY_TOKEN = object()


class _AuthenticatedCorpusScope:
    """Complete admitted receipt metadata for one call; no document/vector cache."""

    def __init__(self, owner, receipts, proof, sequences, *, _token):
        if _token is not _DOCUMENT_INVENTORY_TOKEN:
            raise NativeRetrievalError("native corpus scope is factory-owned")
        self.owner, self.receipts, self.proof = owner, receipts, proof
        self.watermark = max(sequences)
        self.latest = receipts[sequences.index(self.watermark)]
        self.digest = digest_canonical(tuple(item.projection_value() for item in receipts))
        self.catalogue = None
        self.by_event = None
        self.used = False

    def require(self, owner, proof=None):
        if owner is not self.owner or proof is not None and (type(proof) is not AuthenticationProof or proof != self.proof):
            raise NativeRetrievalError("native corpus scope binding differs")
        return self.receipts


class _AuthenticatedDocumentInventory:
    """Documents authenticated once for one synchronous retrieval call."""

    __slots__ = ("_consumed", "_documents", "_owner", "_proof", "_receipts", "_watermark")

    def __init__(
        self,
        owner: "NativeRetrievalDocuments",
        values: tuple[tuple[NativeDocumentReceipt, NativePassageDocument, int], ...],
        proof: AuthenticationProof,
        *,
        _token: object,
    ) -> None:
        if _token is not _DOCUMENT_INVENTORY_TOKEN:
            raise NativeRetrievalError("native document inventory is factory-owned")
        self._owner = owner
        self._proof = proof
        self._consumed = False
        self._receipts = tuple(receipt for receipt, _document, _sequence in values)
        self._documents = {
            receipt.event_id: document for receipt, document, _sequence in values
        }
        self._watermark = max(sequence for _receipt, _document, sequence in values)

    def inspect(
        self,
        owner: "NativeRetrievalDocuments",
        receipts: tuple[NativeDocumentReceipt, ...],
    ) -> dict[str, NativePassageDocument]:
        if self._consumed or owner is not self._owner or receipts != self._receipts:
            raise NativeRetrievalError("native document inventory binding differs")
        return dict(self._documents)

    def consume(
        self,
        owner: "NativeRetrievalDocuments",
        receipts: tuple[NativeDocumentReceipt, ...],
        proof: AuthenticationProof,
    ) -> dict[str, NativePassageDocument]:
        documents = self.inspect(owner, receipts)
        if type(proof) is not AuthenticationProof or proof != self._proof:
            raise NativeRetrievalError("native document inventory proof differs")
        self._consumed = True
        return documents

    def watermark(self, owner, receipts, proof) -> int:
        self.inspect(owner, receipts)
        if type(proof) is not AuthenticationProof or proof != self._proof:
            raise NativeRetrievalError("native document inventory proof differs")
        return self._watermark


class NativeRetrievalContextReadPort:
    """Authenticated composed read seam; callers cannot supply raw contexts."""

    __slots__ = ("_read",)

    def __init__(self, read, *, _token: object) -> None:
        if _token is not _CONTEXT_READ_PORT_TOKEN:
            raise NativeRetrievalError("native context read port is factory-owned")
        self._read = read

    def require(self, receipt: NativeRetrievalContextReceipt) -> NativeRetrievalContext:
        if type(receipt) is not NativeRetrievalContextReceipt:
            raise NativeRetrievalError("native context read receipt differs")
        context = self._read(receipt)
        if type(context) is not NativeRetrievalContext:
            raise NativeRetrievalError("native context read result differs")
        return context


class NativeRetrievalDocuments:
    """Trusted facade binding native extraction, embedding and projection state."""

    def __init__(self, *, objects: GovernedObjects, extraction: GovernedExtractionRecords, commands: AuthorityCommands, events: AuthorityEvents, projector: NativeDocumentProjection, reader_principal_id: str, authority_domain: str, controller_principal_id: str, passage_hydration_policy_digest: str, vector_hydration_policy_digest: str, receipt_hydration_policy_digest: str, document_hydration_policy_digest: str, document_admission_definition_digest: str, command_definition_digest: str, context_hydration_policy_digest: str, context_admission_definition_digest: str, context_command_definition_digest: str) -> None:
        if type(objects) is not GovernedObjects or type(extraction) is not GovernedExtractionRecords or type(commands) is not AuthorityCommands or type(events) is not AuthorityEvents:
            raise NativeRetrievalError("native retrieval requires exact authority facades")
        if not callable(getattr(projector, "upsert", None)) or not callable(getattr(projector, "retrieve", None)) or not callable(getattr(projector, "retrieve_vector", None)):
            raise NativeRetrievalError("native retrieval projector differs")
        for value in (reader_principal_id, authority_domain, controller_principal_id): _text(value, "native retrieval identity")
        for value in (passage_hydration_policy_digest, vector_hydration_policy_digest, receipt_hydration_policy_digest, document_hydration_policy_digest, document_admission_definition_digest, command_definition_digest, context_hydration_policy_digest, context_admission_definition_digest, context_command_definition_digest): _digest(value, "native retrieval policy")
        self._objects, self._extraction, self._commands, self._events, self._projector = objects, extraction, commands, events, projector
        self._reader, self._domain, self._controller = reader_principal_id, authority_domain, controller_principal_id
        self._passage_policy, self._vector_policy, self._receipt_policy, self._document_policy = passage_hydration_policy_digest, vector_hydration_policy_digest, receipt_hydration_policy_digest, document_hydration_policy_digest
        self._document_definition, self._command_definition = document_admission_definition_digest, command_definition_digest
        self._context_policy = context_hydration_policy_digest
        self._context_definition = context_admission_definition_digest
        self._context_command_definition = context_command_definition_digest

    @property
    def command_definition_digest(self) -> str:
        return self._command_definition

    @property
    def context_command_definition_digest(self) -> str:
        return self._context_command_definition

    def context_read_port(self, *, proof: AuthenticationProof) -> NativeRetrievalContextReadPort:
        if type(proof) is not AuthenticationProof:
            raise NativeRetrievalError("native context read proof differs")
        return NativeRetrievalContextReadPort(
            lambda receipt: self.read_context(receipt, proof=proof),
            _token=_CONTEXT_READ_PORT_TOKEN,
        )

    def require_document(self, receipt: NativeDocumentReceipt, *, proof: AuthenticationProof) -> NativePassageDocument:
        """Re-read one exact governed document and its embedding authority."""
        return self._read(receipt, proof)[0]

    def authenticated_corpus_scope(self, receipts, *, proof):
        """Authenticate complete metadata membership, not every body/vector."""
        if (type(receipts) is not tuple or not receipts or type(proof) is not AuthenticationProof
                or any(type(item) is not NativeDocumentReceipt for item in receipts)
                or len({item.event_id for item in receipts}) != len(receipts)):
            raise NativeRetrievalError("native corpus scope inventory differs")
        sequences = tuple(self._verify_event(item, proof) for item in receipts)
        return _AuthenticatedCorpusScope(self, receipts, proof, sequences, _token=_DOCUMENT_INVENTORY_TOKEN)

    def corpus_scope_receipts(self, scope, *, proof=None):
        if type(scope) is not _AuthenticatedCorpusScope:
            raise NativeRetrievalError("native corpus scope type differs")
        return scope.require(self, proof)

    def corpus_scope_watermark(self, scope, *, proof):
        self.corpus_scope_receipts(scope, proof=proof)
        self._verify_event(scope.latest, proof)
        return scope.watermark

    def prepare_corpus_catalogue(self, scope, subjects, generation_id, *, proof):
        """Projection headers are search hints, never authoritative content."""
        receipts = self.corpus_scope_receipts(scope, proof=proof)
        expected = {item.document_receipt.event_id: item for item in subjects}
        if tuple(item.document_receipt for item in subjects) != receipts:
            raise NativeRetrievalError("native corpus subject membership differs")
        rows = self._projector.corpus_metadata(receipts)
        catalogue = {}
        seen = set()
        for row in rows:
            values = {key: row[key] for key in receipts[0].projection_value()}
            receipt = NativeDocumentReceipt.from_projection(values)
            subject = expected.get(receipt.event_id)
            if (subject is None or receipt != subject.document_receipt or receipt.event_id in seen
                    or row['revision_id'] != subject.revision_id or row['generation_id'] != generation_id
                    or subject.source_id is not None and row['source_id'] != subject.source_id):
                raise NativeRetrievalError("native corpus projection membership differs")
            _text(row['passage_id'], 'corpus passage')
            _text(row['source_id'], 'corpus source')
            if row['passage_id'] in catalogue:
                raise NativeRetrievalError("native corpus passages repeat")
            catalogue[row['passage_id']] = (receipt, subject, dict(row))
            seen.add(receipt.event_id)
        if len(catalogue) != len(receipts):
            raise NativeRetrievalError("native corpus projection gap")
        scope.catalogue = catalogue
        scope.by_event = {item[0].event_id: item for item in catalogue.values()}

    def fulltext_authority_view_for_scope(self, scope, snapshot, *, proof):
        self.corpus_scope_receipts(scope, proof=proof)
        if (
            scope.catalogue is None
            or snapshot.index_document_count != len(scope.catalogue)
            or any(item[2]["generation_id"] != str(snapshot.generation_id)
                   for item in scope.catalogue.values())
            or snapshot.document_label != getattr(self._projector, "document_label", None)
            or snapshot.index_name != getattr(self._projector, "fulltext_index", None)
        ):
            raise NativeRetrievalError("native corpus snapshot differs")
        def resolve(passages):
            if len(passages) > BRANCH_RESULT_LIMIT:
                raise NativeRetrievalHold("RESULT_LIMIT_EXCEEDED")
            result = []
            for passage in passages:
                receipt, subject, hinted = self.corpus_passage(scope, passage, proof=proof)
                document = self.require_document(receipt, proof=proof)
                self._require_corpus_document(document, subject, hinted, str(snapshot.generation_id))
                result.append(self._fulltext_binding(document))
            return tuple(result)
        metadata = tuple(sorted((passage, item[2]['source_id']) for passage, item in scope.catalogue.items()))
        return FullTextAuthorityView(snapshot, (), (), FullTextCorpusScope(scope.digest, metadata, resolve))

    def corpus_passage(self, scope, passage_id, *, proof):
        self.corpus_scope_receipts(scope, proof=proof)
        if scope.catalogue is None or passage_id not in scope.catalogue:
            raise NativeRetrievalError("native retrieval hit outside current corpus")
        return scope.catalogue[passage_id]

    @staticmethod
    def _require_corpus_document(document, subject, hinted, generation_id):
        if (document.revision_id != subject.revision_id or document.generation_id != generation_id
                or document.passage_id != hinted['passage_id'] or document.source_id != hinted['source_id']):
            raise NativeRetrievalError("native corpus search hint differs from admitted document")

    @staticmethod
    def _fulltext_binding(document):
        return FullTextDocumentBinding(
            passage_id=document.passage_id, dependency_root_id=document.dependency_root_id,
            source_id=document.source_id, source_identity=document.revision_id,
            provenance_digest=document.digest, language=document.language, rights_current=True, lifecycle="ACTIVE",
        )

    def authenticated_document_inventory(
        self,
        receipts: tuple[NativeDocumentReceipt, ...],
        *,
        proof: AuthenticationProof,
    ) -> _AuthenticatedDocumentInventory:
        """Authenticate an exact inventory for reuse only in the current call."""
        if (
            type(receipts) is not tuple
            or not receipts
            or len(receipts) > 4_096
            or any(type(item) is not NativeDocumentReceipt for item in receipts)
            or len({item.event_id for item in receipts}) != len(receipts)
        ):
            raise NativeRetrievalError("native document inventory differs")
        values = []
        for receipt in receipts:
            document, _vector, sequence = self._read_with_sequence(receipt, proof)
            values.append((receipt, document, sequence))
        return _AuthenticatedDocumentInventory(
            self,
            tuple(values),
            proof,
            _token=_DOCUMENT_INVENTORY_TOKEN,
        )

    def require_authenticated_inventory(
        self,
        inventory: _AuthenticatedDocumentInventory,
        receipts: tuple[NativeDocumentReceipt, ...],
    ) -> dict[str, NativePassageDocument]:
        if type(inventory) is not _AuthenticatedDocumentInventory:
            raise NativeRetrievalError("native document inventory type differs")
        return inventory.inspect(self, receipts)

    def consume_authenticated_inventory(
        self,
        inventory: _AuthenticatedDocumentInventory,
        receipts: tuple[NativeDocumentReceipt, ...],
        *,
        proof: AuthenticationProof,
    ) -> dict[str, NativePassageDocument]:
        if type(inventory) is not _AuthenticatedDocumentInventory:
            raise NativeRetrievalError("native document inventory type differs")
        return inventory.consume(self, receipts, proof)

    def authenticated_inventory_watermark(self, inventory, receipts, *, proof: AuthenticationProof) -> int:
        """Use this call's verified sequence evidence, retaining current proof validation."""
        if type(inventory) is not _AuthenticatedDocumentInventory:
            raise NativeRetrievalError("native document inventory type differs")
        watermark = inventory.watermark(self, receipts, proof)
        # The old watermark pass refreshed read authorisation after inventory
        # construction. Keep that boundary once, not once per immutable event.
        self._verify_event(receipts[0], proof)
        return watermark

    def reproject(
        self, receipt: NativeDocumentReceipt, *, proof: AuthenticationProof,
    ) -> NativePassageDocument:
        """Restore one exact retained document to the derived active index."""
        document, vector = self._read(receipt, proof)
        self._projector.upsert(receipt, document, vector)
        return document

    def admit(self, request: NativeDocumentRequest, *, proof: AuthenticationProof) -> tuple[NativeDocumentReceipt, NativePassageDocument]:
        metadata = self._extraction.metadata(request.extraction_request.run_version_id, proof=proof)
        if metadata.input_binding_digest != request.extraction_request.input_binding.digest or metadata.outcome is not ExtractionOutcome.SUCCESS or not metadata.terminal:
            raise NativeRetrievalHold("EXTRACTION_INPUT_NOT_ADMITTED")
        passage = request.extraction_request.input_binding.passage(request.passage_id)
        hydrated_passage = self._objects.hydrate(HydrationRequest(passage.admission_id, passage.purpose), proof=proof)
        self._access(hydrated_passage.decision, self._passage_policy, passage.object_class, passage.allowed_use)
        if digest_bytes(hydrated_passage.data) != passage.blob_digest or len(hydrated_passage.data) != passage.byte_length:
            raise NativeRetrievalError("retained passage bytes differ")
        try:
            text = hydrated_passage.data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NativeRetrievalHold("PASSAGE_NOT_UTF8") from exc
        vector_object = self._objects.rehydrate(HydrationRequest(request.embedding.vector_admission_id, NATIVE_VECTOR_USE), proof=proof)
        self._access(vector_object.decision, self._vector_policy, NATIVE_VECTOR_CLASS, NATIVE_VECTOR_USE)
        vector = _vector(vector_object.data)
        receipt_object = self._objects.rehydrate(HydrationRequest(request.embedding.receipt_admission_id, NATIVE_EMBEDDING_RECEIPT_USE), proof=proof)
        self._access(receipt_object.decision, self._receipt_policy, NATIVE_EMBEDDING_RECEIPT_CLASS, NATIVE_EMBEDDING_RECEIPT_USE)
        embedding = NativeEmbeddingReceipt.from_bytes(receipt_object.data)
        if embedding.input_text_digest != passage.text_digest or embedding.vector_digest != digest_bytes(vector_object.data):
            raise NativeRetrievalError("embedding receipt does not bind passage/vector bytes")
        document = NativePassageDocument(
            generation_id=request.generation_id,
            passage_id=str(passage.passage_id),
            dependency_root_id=request.dependency_root_id,
            source_id=str(request.extraction_request.input_binding.definition_id), revision_id=str(request.extraction_request.input_binding.revision_id), representation_id=str(request.extraction_request.input_binding.representation_id),
            language=passage.language, text=text, text_digest=passage.text_digest,
            rights_digest=hydrated_passage.decision.state_cutoff_digest,
            provenance_digest=digest_bytes(request.extraction_request.canonical_bytes), vector_digest=embedding.vector_digest,
            vector_admission_id=str(request.embedding.vector_admission_id),
            embedding_receipt_digest=digest_bytes(receipt_object.data),
            embedding_receipt_admission_id=str(request.embedding.receipt_admission_id),
            embedding_model_digest=embedding.model_digest,
        )
        admitted = self._objects.admit(ObjectAdmissionRequest(NATIVE_DOCUMENT_ADMISSION_TYPE, request.idempotency_key), document.canonical_bytes, proof=proof).admission
        if admitted.definition_digest != self._document_definition or admitted.object_class != NATIVE_DOCUMENT_CLASS or admitted.allowed_use != NATIVE_DOCUMENT_USE or admitted.blob.blob_digest != document.digest or not admitted.active:
            raise NativeRetrievalError("native document admission differs")
        committed = self._commands.execute(SemanticCommand(NATIVE_DOCUMENT_COMMAND, request.aggregate_id, request.expected_aggregate_version, ObjectAdmissionPayload(admitted.admission_id), request.idempotency_key), proof=proof)
        result = NativeDocumentReceipt(str(committed.event_id), str(committed.command_id), request.aggregate_id, committed.aggregate_version, admitted.admission_id, document.digest, request.embedding.vector_admission_id, request.embedding.receipt_admission_id)
        self._verify_event(result, proof)
        self._projector.upsert(result, document, vector)
        return result, document

    def retrieve(self, *, query_receipt: NativeDocumentReceipt, query_text: str, proof: AuthenticationProof) -> NativeRetrievalResult:
        if type(query_receipt) is not NativeDocumentReceipt:
            raise NativeRetrievalError("native retrieval query receipt differs")
        query_document, query_vector = self._read(query_receipt, proof)
        fulltext, vector = self._projector.retrieve(
            query_text=_text(query_text, "query_text", 16_384),
            query_vector=query_vector,
        )
        return NativeRetrievalResult(
            self._hits(fulltext, query_document.generation_id, proof),
            self._hits(vector, query_document.generation_id, proof),
            query_document.generation_id,
        )

    def retrieve_vector(
        self,
        request: NativeVectorRequest,
        *,
        proof: AuthenticationProof,
        allowed_receipts: tuple[NativeDocumentReceipt, ...] | None = None,
    ) -> NativeVectorBranchReceipt:
        """Execute one attributed production VECTOR branch without embedding work."""
        if type(request) is not NativeVectorRequest:
            raise NativeRetrievalError("native vector request differs")
        query_receipt = self._receipt_for_event(request.query_event_id, proof)
        query_document, query_vector = self._read(query_receipt, proof)
        if (
            query_document.digest != request.query_document_digest
            or query_document.generation_id != request.generation_id
        ):
            raise NativeRetrievalError("native vector query authority differs")
        rows = self._projector.retrieve_vector(query_vector=query_vector)
        documents = self._documents(rows, query_document.generation_id, proof, allowed_receipts=allowed_receipts)
        hits = tuple(
            NativeVectorBranchHit(
                rank=index,
                passage_id=document.passage_id,
                dependency_root_id=document.dependency_root_id,
                source_revision_id=document.revision_id,
                document_digest=document.digest,
                rights_digest=document.rights_digest,
                provenance_digest=document.provenance_digest,
                raw_score_ppm=max(0, min(1_000_000, int(round(score * 1_000_000)))),
            )
            for index, (document, score) in enumerate(documents, 1)
        )
        generation_digest = digest_bytes(canonical_json_bytes({
            "generation_id": query_document.generation_id,
            "document_command_definition": self._command_definition,
            "embedding_model_digest": query_document.embedding_model_digest,
        }))
        semantic = canonical_json_bytes({
            "request_digest": request.request_digest,
            "generation_digest": generation_digest,
            "hits": [hit.canonical_value() for hit in hits],
        })
        return NativeVectorBranchReceipt(
            receipt_id=str(uuid.uuid5(uuid.NAMESPACE_URL, digest_bytes(semantic))),
            request_digest=request.request_digest,
            mode=BranchMode.VECTOR,
            outcome=BranchOutcome.COMPLETE,
            reason=None if hits else "NO_MATCH",
            generation_id=query_document.generation_id,
            generation_digest=generation_digest,
            profile_id=NATIVE_VECTOR_PROFILE,
            query_valid_time=request.query_valid_time,
            serving_time=request.serving_time,
            hits=hits,
            authority_read_count=1 + len(documents),
        )

    def fulltext_authority_view(
        self,
        receipts: tuple[NativeDocumentReceipt, ...],
        snapshot: FullTextProjectionSnapshot,
        *,
        proof: AuthenticationProof,
    ) -> FullTextAuthorityView:
        """Build the existing full-text authority view from reverified documents."""
        if type(receipts) is not tuple or not receipts or len(receipts) > 4_096 or type(snapshot) is not FullTextProjectionSnapshot:
            raise NativeRetrievalError("native full-text authority inventory differs")
        documents = tuple(self._read(receipt, proof)[0] for receipt in receipts)
        return self._fulltext_authority_view(documents, snapshot)

    def fulltext_authority_view_from_inventory(
        self,
        inventory: _AuthenticatedDocumentInventory,
        receipts: tuple[NativeDocumentReceipt, ...],
        snapshot: FullTextProjectionSnapshot,
    ) -> FullTextAuthorityView:
        """Build a view from this call's authenticated immutable inventory."""
        documents = self.require_authenticated_inventory(inventory, receipts)
        return self._fulltext_authority_view(
            tuple(documents[receipt.event_id] for receipt in receipts), snapshot,
        )

    def _fulltext_authority_view(
        self,
        documents: tuple[NativePassageDocument, ...],
        snapshot: FullTextProjectionSnapshot,
    ) -> FullTextAuthorityView:
        if (
            len({document.passage_id for document in documents}) != len(documents)
            or any(document.generation_id != str(snapshot.generation_id) for document in documents)
            or snapshot.index_document_count != len(documents)
            or snapshot.document_label != getattr(self._projector, "document_label", None)
            or snapshot.index_name != getattr(self._projector, "fulltext_index", None)
        ):
            raise NativeRetrievalError("native full-text snapshot differs")
        return FullTextAuthorityView(
            snapshot=snapshot,
            authority_aliases=(),
            document_bindings=tuple(sorted((
                FullTextDocumentBinding(
                    passage_id=document.passage_id,
                    dependency_root_id=document.dependency_root_id,
                    source_id=document.source_id,
                    source_identity=document.revision_id,
                    # The reused full-text row contract calls this the
                    # provenance digest; it is the exact indexed document
                    # digest.  The document itself separately retains source
                    # provenance and is rehydrated before this view is built.
                    provenance_digest=document.digest,
                    language=document.language,
                    rights_current=True,
                    lifecycle="ACTIVE",
                )
                for document in documents
            ), key=lambda item: item.passage_id)),
        )
    def retain_context(
        self,
        request: NativeRetrievalContextRequest,
        *,
        proof: AuthenticationProof,
    ):
        """Commit one typed four-branch context and return its work-item binding."""
        if type(request) is not NativeRetrievalContextRequest:
            raise NativeRetrievalError("native context request differs")
        exact, fulltext, vector, graph = request.branch_receipts()
        if (
            fulltext.snapshot is None
            or str(fulltext.snapshot.generation_id) != vector.generation_id
            or UtcTimestamp.parse(graph.query_valid_time).value != UtcTimestamp.parse(vector.query_valid_time).value
            or UtcTimestamp.parse(graph.serving_time).value != UtcTimestamp.parse(vector.serving_time).value
        ):
            raise NativeRetrievalError("native context generation/time differs")
        documents = tuple(self._read(item, proof)[0] for item in request.selected_documents)
        selected_ids = {item.passage_id for item in documents}
        used_ids = {
            *(str(hit.passage_id) for hit in fulltext.hits if hit.passage_id is not None),
            *(hit.passage_id for hit in vector.hits),
        }
        if not used_ids.issubset(selected_ids):
            raise NativeRetrievalError("native context selected passage authority differs")
        branch_digests = tuple(digest_bytes(raw) for raw in (
            request.exact_receipt_bytes, request.fulltext_receipt_bytes,
            request.vector_receipt_bytes, request.graph_receipt_bytes,
        ))
        no_match = not (exact.hits or fulltext.hits or vector.hits or graph.hits)
        identity = canonical_json_bytes({
            "request_digest": request.request_digest,
            "branches": list(branch_digests),
            "documents": [item.digest for item in documents],
        })
        branch_values = tuple(json.loads(raw) for raw in (
            request.exact_receipt_bytes, request.fulltext_receipt_bytes,
            request.vector_receipt_bytes, request.graph_receipt_bytes,
        ))
        context = NativeRetrievalContext(
            context_id=str(uuid.uuid5(uuid.NAMESPACE_URL, digest_bytes(identity))),
            request_id=request.request_id, request_digest=request.request_digest,
            lead_id=request.lead_id, lead_digest=request.lead_digest,
            authority_scope_id=request.authority_scope_id,
            rights_inventory_digest=request.rights_inventory_digest,
            generation_id=vector.generation_id,
            query_valid_time=vector.query_valid_time,
            serving_time=vector.serving_time,
            branch_digests=branch_digests,  # type: ignore[arg-type]
            branch_receipts=branch_values,  # type: ignore[arg-type]
            selected_documents=tuple(item.projection_value() for item in request.selected_documents),
            outcome="COMPLETE", no_match=no_match,
        )
        admitted = self._objects.admit(
            ObjectAdmissionRequest(NATIVE_CONTEXT_ADMISSION_TYPE, request.idempotency_key),
            context.canonical_bytes, proof=proof,
        ).admission
        if admitted.definition_digest != self._context_definition or admitted.object_class != NATIVE_CONTEXT_CLASS or admitted.allowed_use != NATIVE_CONTEXT_USE or not admitted.active:
            raise NativeRetrievalError("native context admission differs")
        if admitted.blob.blob_digest != context.digest:
            raise NativeRetrievalError("native context admission replay differs")
        committed = self._commands.execute(
            SemanticCommand(NATIVE_CONTEXT_COMMAND, request.aggregate_id,
                            request.expected_aggregate_version,
                            ObjectAdmissionPayload(admitted.admission_id),
                            request.idempotency_key),
            proof=proof,
        )
        receipt = NativeRetrievalContextReceipt(
            context_id=context.context_id, request_id=context.request_id,
            request_digest=context.request_digest, aggregate_id=request.aggregate_id,
            aggregate_version=committed.aggregate_version,
            event_id=str(committed.event_id), command_id=str(committed.command_id),
            admission_id=admitted.admission_id, context_object_digest=context.digest,
            authority_scope_id=context.authority_scope_id,
            rights_inventory_digest=context.rights_inventory_digest,
            generation_id=context.generation_id,
            query_valid_time=context.query_valid_time, serving_time=context.serving_time,
            exact_receipt_bytes=canonical_json_bytes(context.branch_receipts[0]),
            fulltext_receipt_bytes=canonical_json_bytes(context.branch_receipts[1]),
            vector_receipt_bytes=canonical_json_bytes(context.branch_receipts[2]),
            graph_receipt_bytes=canonical_json_bytes(context.branch_receipts[3]),
            controller_principal_id=self._controller, authority_domain=self._domain,
            no_match=context.no_match,
        )
        self.read_context(receipt, proof=proof)
        retained_request = NativeRetrievalContextRequest(
            context.request_id, request.idempotency_key, request.aggregate_id,
            request.expected_aggregate_version, context.lead_id,
            context.lead_digest, context.authority_scope_id,
            context.rights_inventory_digest,
            receipt.exact_receipt_bytes, receipt.fulltext_receipt_bytes,
            receipt.vector_receipt_bytes, receipt.graph_receipt_bytes,
            tuple(
                NativeDocumentReceipt.from_projection(item)
                for item in context.selected_documents
            ),
        )
        if retained_request.request_digest != context.request_digest:
            raise NativeRetrievalError("native context replay request differs")
        from newsroom.increment6.work_items import RetrievalBindingState, RetrievalInputBinding
        return RetrievalInputBinding(
            RetrievalBindingState.RECEIPT, retained_request.request_id,
            retained_request.idempotency_key, retained_request.request_digest,
            retained_request.canonical_bytes,
            context.context_id, receipt.receipt_digest, receipt.outcome,
            receipt.reason, receipt.no_match, receipt.canonical_bytes,
        )

    def read_context(self, receipt: NativeRetrievalContextReceipt, *, proof: AuthenticationProof) -> NativeRetrievalContext:
        if type(receipt) is not NativeRetrievalContextReceipt:
            raise NativeRetrievalError("native context receipt differs")
        provenance = self._events.provenance(receipt.event_id, proof=proof)
        event = provenance.event
        if provenance.command_definition.command_type != NATIVE_CONTEXT_COMMAND or provenance.command_definition.definition_digest != self._context_command_definition or event.command_definition_digest != self._context_command_definition or event.event_type != NATIVE_CONTEXT_EVENT or event.object_admission_id != str(receipt.admission_id) or event.payload_digest != receipt.context_object_digest or event.command_id != receipt.command_id or event.aggregate_id != str(receipt.aggregate_id) or event.aggregate_version != receipt.aggregate_version or event.principal_id != self._controller or provenance.authentication.principal_id != self._controller or provenance.authentication.authority_domain != self._domain or event.trust_scope != TrustScope.ADMITTED.value or event.security_scope != NATIVE_SECURITY_SCOPE or event.retention_scope != NATIVE_RETENTION_SCOPE:
            raise NativeRetrievalError("native context authority event differs")
        hydrated = self._objects.rehydrate(HydrationRequest(receipt.admission_id, NATIVE_CONTEXT_USE), proof=proof)
        self._access(hydrated.decision, self._context_policy, NATIVE_CONTEXT_CLASS, NATIVE_CONTEXT_USE)
        context = NativeRetrievalContext.from_bytes(hydrated.data)
        if digest_bytes(hydrated.data) != receipt.context_object_digest or context.context_id != receipt.context_id or context.request_id != receipt.request_id or context.request_digest != receipt.request_digest or context.authority_scope_id != receipt.authority_scope_id or context.rights_inventory_digest != receipt.rights_inventory_digest or context.generation_id != receipt.generation_id or context.graph_generation_id != receipt.graph_generation_id or context.query_valid_time != receipt.query_valid_time or context.serving_time != receipt.serving_time or context.branch_digests != tuple(digest_bytes(raw) for raw in (receipt.exact_receipt_bytes, receipt.fulltext_receipt_bytes, receipt.vector_receipt_bytes, receipt.graph_receipt_bytes)) or context.outcome != receipt.outcome or context.no_match != receipt.no_match:
            raise NativeRetrievalError("native context retained bytes differ")
        selected_receipts = tuple(
            NativeDocumentReceipt.from_projection(item)
            for item in context.selected_documents
        )
        selected_documents = tuple(self._read(item, proof)[0] for item in selected_receipts)
        _, fulltext, vector, _ = receipt.branch_receipts()
        used_passages = {
            *(str(hit.passage_id) for hit in fulltext.hits if hit.passage_id is not None),
            *(hit.passage_id for hit in vector.hits),
        }
        if not used_passages.issubset({item.passage_id for item in selected_documents}):
            raise NativeRetrievalError("native context selected passage authority differs")
        return context

    def _read(self, receipt: NativeDocumentReceipt, proof: AuthenticationProof) -> tuple[NativePassageDocument, tuple[float, ...]]:
        document, vector, _sequence = self._read_with_sequence(receipt, proof)
        return document, vector

    def _read_with_sequence(self, receipt: NativeDocumentReceipt, proof: AuthenticationProof) -> tuple[NativePassageDocument, tuple[float, ...], int]:
        sequence = self._verify_event(receipt, proof)
        hydrated = self._objects.rehydrate(HydrationRequest(receipt.admission_id, NATIVE_DOCUMENT_USE), proof=proof)
        self._access(hydrated.decision, self._document_policy, NATIVE_DOCUMENT_CLASS, NATIVE_DOCUMENT_USE)
        document = NativePassageDocument.from_bytes(hydrated.data)
        if document.digest != receipt.document_digest or document.vector_admission_id != str(receipt.vector_admission_id) or document.embedding_receipt_admission_id != str(receipt.embedding_receipt_admission_id):
            raise NativeRetrievalError("native document receipt differs")
        vector_object = self._objects.rehydrate(HydrationRequest(receipt.vector_admission_id, NATIVE_VECTOR_USE), proof=proof)
        self._access(vector_object.decision, self._vector_policy, NATIVE_VECTOR_CLASS, NATIVE_VECTOR_USE)
        vector = _vector(vector_object.data)
        if digest_bytes(vector_object.data) != document.vector_digest:
            raise NativeRetrievalError("native document vector differs")
        receipt_object = self._objects.rehydrate(HydrationRequest(receipt.embedding_receipt_admission_id, NATIVE_EMBEDDING_RECEIPT_USE), proof=proof)
        self._access(receipt_object.decision, self._receipt_policy, NATIVE_EMBEDDING_RECEIPT_CLASS, NATIVE_EMBEDDING_RECEIPT_USE)
        embedding = NativeEmbeddingReceipt.from_bytes(receipt_object.data)
        if digest_bytes(receipt_object.data) != document.embedding_receipt_digest or embedding.vector_digest != document.vector_digest or embedding.input_text_digest != document.text_digest:
            raise NativeRetrievalError("native embedding provenance differs")
        return document, vector, sequence

    def _verify_event(self, receipt: NativeDocumentReceipt, proof: AuthenticationProof) -> int:
        provenance = self._events.provenance(receipt.event_id, proof=proof)
        event = provenance.event
        if provenance.command_definition.command_type != NATIVE_DOCUMENT_COMMAND or provenance.command_definition.definition_digest != self._command_definition or event.command_definition_digest != self._command_definition or event.event_type != NATIVE_DOCUMENT_EVENT or event.object_admission_id != str(receipt.admission_id) or event.payload_digest != receipt.document_digest or event.command_id != receipt.command_id or event.aggregate_id != str(receipt.aggregate_id) or event.aggregate_version != receipt.aggregate_version or event.principal_id != self._controller or provenance.authentication.principal_id != self._controller or provenance.authentication.authority_domain != self._domain or event.trust_scope != TrustScope.ADMITTED.value or event.security_scope != NATIVE_SECURITY_SCOPE or event.retention_scope != NATIVE_RETENTION_SCOPE:
            raise NativeRetrievalError("native retrieval authority event differs")
        return event.ledger_seq

    def _receipt_for_event(self, event_id: str, proof: AuthenticationProof) -> NativeDocumentReceipt:
        provenance = self._events.provenance(event_id, proof=proof)
        event = provenance.event
        if event.object_admission_id is None:
            raise NativeRetrievalError("native vector query event lacks an object")
        admission_id = ObjectAdmissionId.parse(event.object_admission_id)
        hydrated = self._objects.rehydrate(
            HydrationRequest(admission_id, NATIVE_DOCUMENT_USE), proof=proof
        )
        self._access(hydrated.decision, self._document_policy, NATIVE_DOCUMENT_CLASS, NATIVE_DOCUMENT_USE)
        document = NativePassageDocument.from_bytes(hydrated.data)
        return NativeDocumentReceipt(
            event.event_id,
            event.command_id,
            AggregateId.parse(event.aggregate_id),
            event.aggregate_version,
            admission_id,
            document.digest,
            ObjectAdmissionId.parse(document.vector_admission_id),
            ObjectAdmissionId.parse(document.embedding_receipt_admission_id),
        )

    def _access(self, decision: Any, policy: str, object_class: str, allowed_use: str) -> None:
        if decision.policy_contract_digest != policy or decision.principal_id != self._reader or decision.authority_domain != self._domain or decision.object_class != object_class or decision.allowed_use != allowed_use:
            raise NativeRetrievalError("native retrieval object access differs")

    def _hits(self, rows: tuple[Mapping[str, object], ...], generation_id: str, proof: AuthenticationProof) -> tuple[NativeRetrievalHit, ...]:
        return tuple(
            NativeRetrievalHit(document.passage_id, document.dependency_root_id, score)
            for document, score in self._documents(rows, generation_id, proof)
        )

    def _documents(self, rows: tuple[Mapping[str, object], ...], generation_id: str, proof: AuthenticationProof,
                   *, allowed_receipts=None) -> tuple[tuple[NativePassageDocument, float], ...]:
        if len(rows) > NATIVE_RESULT_LIMIT:
            raise NativeRetrievalHold("RESULT_LIMIT_EXCEEDED")
        result: list[tuple[NativePassageDocument, float]] = []
        seen: set[str] = set()
        for row in rows:
            values = dict(row)
            score = values.pop("score", None)
            receipt = NativeDocumentReceipt.from_projection(values)
            if allowed_receipts is not None and receipt not in allowed_receipts:
                raise NativeRetrievalError("projection hit outside current corpus")
            document, _ = self._read(receipt, proof)
            if (
                document.generation_id != generation_id
                or document.passage_id in seen
                or type(score) not in (float, int)
                or isinstance(score, bool)
            ):
                raise NativeRetrievalError("projection hit lacks admitted authority")
            seen.add(document.passage_id)
            result.append((document, float(score)))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class NativeRetrievalSubject:
    revision_id: str
    graph_root_id: str
    document_receipt: NativeDocumentReceipt
    query_text: str
    source_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.revision_id, "native retrieval subject revision")
        _text(self.graph_root_id, "native retrieval graph root")
        _text(self.query_text, "native retrieval subject query", 16_384)
        if self.source_id is not None:
            _text(self.source_id, "native retrieval subject source")
        if type(self.document_receipt) is not NativeDocumentReceipt:
            raise NativeRetrievalError("native retrieval subject document differs")


def _stable_uuid4(value: object) -> str:
    raw = bytearray(hashlib.sha256(canonical_json_bytes(value)).digest()[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(raw)))


class NativeRetrievalPort:
    """Concrete four-branch runtime port over current governed documents."""

    def __init__(
        self, *, documents: NativeRetrievalDocuments,
        exact: SQLiteExactRetriever, fulltext: FullTextRetriever,
        increment4: Increment4Neo4jController, fulltext_view: FullTextAuthorityView,
        subjects: tuple[NativeRetrievalSubject, ...],
        document_inventory: _AuthenticatedDocumentInventory | _AuthenticatedCorpusScope,
        authority_scope_id: str, rights_inventory_digest: str,
        minimum_authority_watermark: int,
    ) -> None:
        if type(documents) is not NativeRetrievalDocuments or type(exact) is not SQLiteExactRetriever or type(fulltext) is not FullTextRetriever or type(increment4) is not Increment4Neo4jController:
            raise NativeRetrievalError("native retrieval port requires exact branch facades")
        if type(fulltext_view) is not FullTextAuthorityView:
            raise NativeRetrievalError("native retrieval authority views differ")
        if type(subjects) is not tuple or not subjects or any(type(item) is not NativeRetrievalSubject for item in subjects):
            raise NativeRetrievalError("native retrieval subject inventory differs")
        if len({item.document_receipt.event_id for item in subjects}) != len(subjects):
            raise NativeRetrievalError("native retrieval subject documents repeat")
        receipts = tuple(item.document_receipt for item in subjects)
        corpus = type(document_inventory) is _AuthenticatedCorpusScope
        if corpus:
            if (documents.corpus_scope_receipts(document_inventory) != receipts
                    or document_inventory.used or document_inventory.catalogue is None or fulltext_view.corpus_scope is None
                    or fulltext_view.corpus_scope.inventory_digest != document_inventory.digest
                    or fulltext_view.corpus_scope.eligible_passages != tuple(sorted(
                        (passage, item[2]['source_id']) for passage, item in document_inventory.catalogue.items()))):
                raise NativeRetrievalError("native corpus scope binding differs")
        else:
            document_by_event = documents.require_authenticated_inventory(document_inventory, receipts)
            if any(document_by_event[item.document_receipt.event_id].revision_id != item.revision_id
                   or document_by_event[item.document_receipt.event_id].generation_id != str(fulltext_view.snapshot.generation_id)
                   for item in subjects):
                raise NativeRetrievalError("native retrieval subject authority differs")
        _text(authority_scope_id, "native retrieval authority scope")
        _digest(rights_inventory_digest, "native retrieval rights inventory")
        if type(minimum_authority_watermark) is not int or minimum_authority_watermark < 0:
            raise NativeRetrievalError("native retrieval authority watermark differs")
        self._documents, self._exact, self._fulltext, self._increment4 = documents, exact, fulltext, increment4
        self._fulltext_view = fulltext_view
        self._corpus_scope = document_inventory if corpus else None
        self._document_inventory = None if corpus else document_inventory
        self._subject_receipts = receipts
        grouped: dict[str, list[NativeRetrievalSubject]] = {}
        for item in subjects:
            grouped.setdefault(item.revision_id, []).append(item)
        self._subjects = {
            revision_id: tuple(sorted(items, key=lambda item: item.document_receipt.event_id))
            for revision_id, items in grouped.items()
        }
        self._scope = authority_scope_id
        self._rights_inventory_digest = rights_inventory_digest
        self._minimum = minimum_authority_watermark

    def retrieve(self, lead, *, proof: AuthenticationProof):
        from newsroom.discovery import NewsLead
        if type(lead) is not NewsLead:
            raise NativeRetrievalError("native retrieval Lead differs")
        revision_id = str(lead.request.revision_id)
        subjects = self._subjects.get(revision_id)
        if subjects is None:
            raise NativeRetrievalHold("NATIVE_RETRIEVAL_DOCUMENT_MISSING")
        if len({item.query_text for item in subjects}) != 1:
            raise NativeRetrievalError("native retrieval subject query differs")
        if self._corpus_scope is not None:
            scope = self._corpus_scope
            if scope.used:
                scope = self._documents.authenticated_corpus_scope(self._subject_receipts, proof=proof)
                all_subjects = tuple(item for items in self._subjects.values() for item in items)
                by_event = {item.document_receipt.event_id: item for item in all_subjects}
                self._documents.prepare_corpus_catalogue(scope,
                    tuple(by_event[item.event_id] for item in self._subject_receipts),
                    str(self._fulltext_view.snapshot.generation_id), proof=proof)
                self._fulltext_view = self._documents.fulltext_authority_view_for_scope(
                    scope, self._fulltext_view.snapshot, proof=proof)
                self._corpus_scope = scope
            self._documents.corpus_scope_receipts(scope, proof=proof)
            scope.used = True
            subject = min(subjects, key=lambda item: scope.by_event[item.document_receipt.event_id][2]['passage_id'])
            receipt, _subject, hinted = scope.by_event[subject.document_receipt.event_id]
            document = self._documents.require_document(receipt, proof=proof)
            self._documents._require_corpus_document(document, subject, hinted, str(self._fulltext_view.snapshot.generation_id))
            document_by_event = {receipt.event_id: document}
            retained_subjects = ((subject, document),)
        else:
            inventory = self._document_inventory
            if inventory is None:
                inventory = self._documents.authenticated_document_inventory(self._subject_receipts, proof=proof)
            document_by_event = self._documents.consume_authenticated_inventory(inventory, self._subject_receipts, proof=proof)
            self._document_inventory = None
            retained_subjects = tuple((subject, document_by_event[subject.document_receipt.event_id]) for subject in subjects)
        if any(document.revision_id != revision_id for _, document in retained_subjects):
            raise NativeRetrievalError("native retrieval subject authority differs")
        if len({item.query_text for item, _document in retained_subjects}) != 1:
            raise NativeRetrievalError("native retrieval subject query differs")
        subject, document = min(retained_subjects, key=lambda item: item[1].passage_id)
        ordered_subjects = sorted(
            (
                item
                for items in self._subjects.values()
                for item in items
            ),
            key=lambda item: (
                item.revision_id,
                item.graph_root_id,
                item.document_receipt.event_id,
            ),
        )
        corpus_inventory_digest = digest_canonical(tuple(
            {
                "revision_id": item.revision_id,
                "graph_root_id": item.graph_root_id,
                "document_receipt": item.document_receipt.projection_value(),
                "query_text_digest": digest_bytes(item.query_text.encode()),
            }
            for item in ordered_subjects
        ))
        seed = {
            "lead": str(lead.request.lead_id),
            "lead_digest": lead.canonical_digest,
            "rights_inventory_digest": self._rights_inventory_digest,
            "corpus_inventory_digest": corpus_inventory_digest,
        }
        graph_ids = tuple(sorted({item.graph_root_id for item in subjects}))
        if len(graph_ids) > NATIVE_GRAPH_ROOT_LIMIT:
            raise NativeRetrievalHold("NATIVE_GRAPH_ROOT_LIMIT_EXCEEDED")
        graph_request_digest = digest_bytes(canonical_json_bytes({
            "schema_identity": "newsroom.increment5.native-graph-request.v1",
            "lead_digest": lead.canonical_digest, "canonical_ids": list(graph_ids),
            "query_valid_time": lead.recorded_at.to_text(),
        }))
        graph_response = self._increment4.read_active(
            Increment4Neo4jActiveReadRequest(graph_ids, lead.recorded_at, NATIVE_GRAPH_ROOT_LIMIT),
            proof=proof,
        )
        serving = graph_response.metadata.serving_time
        if serving.value < lead.recorded_at.value:
            raise NativeRetrievalError("native graph serving time precedes Lead")
        graph = NativeGraphBranchReceipt.from_response(
            graph_request_digest, graph_ids, graph_response,
        )
        branch_seed = {**seed, "serving_time": serving.to_text()}
        branch_identity = digest_canonical(branch_seed)
        exact_request = ExactBranchRequest(
            BranchRequestId.parse(_stable_uuid4({**branch_seed, "branch": "exact"})),
            f"native-exact:{lead.request.lead_id}:{branch_identity}", EXACT_BRANCH_ACTOR_ID,
            EXACT_BRANCH_PURPOSE, EXACT_BRANCH_POLICY_ID,
            INCREMENT_5A_CONTRACT_DIGEST, ExactLookupKind.SOURCE_REVISION_ID,
            revision_id, lead.recorded_at, serving,
            minimum_ledger_seq=self._minimum,
        )
        exact = self._exact.retrieve(exact_request).receipt
        snapshot = self._fulltext_view.snapshot
        fulltext_request = FullTextBranchRequest(
            BranchRequestId.parse(_stable_uuid4({**branch_seed, "branch": "fulltext"})),
            f"native-fulltext:{lead.request.lead_id}:{branch_identity}", FULLTEXT_ACTOR_ID,
            FULLTEXT_PURPOSE, FULLTEXT_POLICY_ID, INCREMENT_5A_CONTRACT_DIGEST,
            FULLTEXT_COMPONENT_DIGEST, NORMALIZATION_COMPONENT_DIGEST,
            snapshot.generation_id, snapshot.generation_identity_digest,
            snapshot.rights_manifest_digest, subject.query_text,
            FullTextLanguageMode.MIXED_EN_GB_ZH_HANT_HK, (document.source_id,),
            lead.recorded_at, serving, snapshot.contiguous_ledger_seq,
        )
        fulltext = self._fulltext.retrieve(fulltext_request,
            **({"authority_view": self._fulltext_view} if self._corpus_scope is not None else {})).receipt
        if fulltext.authority_view_digest != self._fulltext_view.view_digest:
            raise NativeRetrievalError("native full-text authority view differs")
        vector_request = NativeVectorRequest(
            _stable_uuid4({**branch_seed, "branch": "vector"}),
            f"native-vector:{lead.request.lead_id}:{branch_identity}", subject.document_receipt.event_id,
            document.digest, document.generation_id,
            lead.recorded_at.to_text(), serving.to_text(),
        )
        vector = self._documents.retrieve_vector(vector_request, proof=proof,
            **({"allowed_receipts": self._subject_receipts} if self._corpus_scope is not None else {}))
        by_passage: dict[str, NativeDocumentReceipt] = {}
        if self._corpus_scope is not None:
            by_passage = {passage: item[0] for passage, item in self._corpus_scope.catalogue.items()}
        else:
            for items in self._subjects.values():
                for item in items:
                    retained = document_by_event[item.document_receipt.event_id]
                    by_passage[retained.passage_id] = item.document_receipt
        used = {
            *(str(hit.passage_id) for hit in fulltext.hits if hit.passage_id is not None),
            *(hit.passage_id for hit in vector.hits),
        }
        if not used.issubset(by_passage):
            raise NativeRetrievalError("native retrieval hit lacks governed document")
        if self._corpus_scope is not None:
            for passage in used:
                receipt, selected_subject, hint = self._documents.corpus_passage(self._corpus_scope, passage, proof=proof)
                selected_document = self._documents.require_document(receipt, proof=proof)
                self._documents._require_corpus_document(selected_document, selected_subject, hint, document.generation_id)
        selected = tuple(dict.fromkeys((
            subject.document_receipt,
            *(by_passage[item] for item in sorted(used)),
        )))
        context_inputs = {
            **seed,
            "branch_digests": [
                digest_bytes(exact.canonical_bytes),
                digest_bytes(fulltext.canonical_bytes),
                digest_bytes(vector.canonical_bytes),
                digest_bytes(graph.canonical_bytes),
            ],
            "selected_documents": [item.projection_value() for item in selected],
        }
        context_identity = digest_canonical(context_inputs)
        request = NativeRetrievalContextRequest(
            _stable_uuid4({**context_inputs, "kind": "native-context"}),
            f"native-context:{context_identity}",
            AggregateId.parse(_stable_uuid4({**context_inputs, "aggregate": "retrieval-context"})),
            0, str(lead.request.lead_id), lead.canonical_digest, self._scope,
            self._rights_inventory_digest,
            exact.canonical_bytes, fulltext.canonical_bytes,
            vector.canonical_bytes, graph.canonical_bytes, selected,
        )
        return self._documents.retain_context(request, proof=proof)


def _vector(raw: bytes) -> tuple[float, ...]:
    if type(raw) is not bytes or len(raw) != 4 * NATIVE_VECTOR_DIMENSIONS:
        raise NativeRetrievalHold("EMBEDDING_VECTOR_MISSING")
    values = struct.unpack(f">{NATIVE_VECTOR_DIMENSIONS}f", raw)
    if not all(math.isfinite(value) for value in values) or not any(value != 0.0 for value in values):
        raise NativeRetrievalHold("EMBEDDING_VECTOR_INVALID")
    return values


__all__ = [name for name in globals() if name.startswith("Native") or name.startswith("NATIVE_")]
