"""Immutable contracts for the independent Increment 5B full-text branch."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from enum import StrEnum
import re
import uuid

from newsroom.authority.canonical import (
    MAX_SAFE_INTEGER,
    canonical_json_bytes,
    digest_bytes,
    validate_sha256_digest,
)
from newsroom.authority.types import TrustScope, UtcTimestamp, require_token
from newsroom.increment5.branch_contracts import (
    BRANCH_RESULT_LIMIT,
    BRANCH_TIMEOUT_MS,
    BranchExclusionReason,
    BranchRequestId,
    Increment5BranchContractError,
)
from newsroom.increment5.decision import INCREMENT_5A_CONTRACT_DIGEST
from newsroom.projection.models import ProjectionGenerationId, ProjectionGenerationState


INCREMENT5_RETRIEVAL_CONTRACT_DIGEST = INCREMENT_5A_CONTRACT_DIGEST
FULLTEXT_COMPONENT_DIGEST = (
    "sha256:ec859d0a25d7684f6c3a693b59dca96337946b07552eae6aa870910eaf24465a"
)
NORMALIZATION_COMPONENT_DIGEST = (
    "sha256:0ed4fa41238d589933905cb3bf55b4dd9fe290c563ff07ee8676d776ad104070"
)
FULLTEXT_POLICY_ID = "increment5-fulltext-branch-v1"
FULLTEXT_ACTOR_ID = "retrieval_worker"
FULLTEXT_PURPOSE = "bounded_fulltext_lookup"
FULLTEXT_QUERY_ID = "increment5.fulltext.v1"
FULLTEXT_PROVIDER = "fulltext-2.0"
FULLTEXT_ANALYZER = "standard-no-stop-words"
FULLTEXT_RESPONSE_BYTE_LIMIT = 262_144
FULLTEXT_MAX_PROJECTION_AGE_SECONDS = 3_600
FULLTEXT_MAX_QUERY_BYTES = 16_384
FULLTEXT_MAX_LUCENE_QUERY_BYTES = 32_768
FULLTEXT_MAX_TERMS = 64
FULLTEXT_INDEXED_FIELDS = (
    "authority_aliases",
    "formal_tokens",
    "han_bigrams",
    "latin_terms",
    "retrieval_text",
)

_NEO4J_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")


class FullTextContractError(Increment5BranchContractError):
    """A full-text request, projection snapshot or binding is malformed."""


class FullTextLanguageMode(StrEnum):
    EN_GB = "EN_GB"
    ZH_HANT_HK = "ZH_HANT_HK"
    MIXED_EN_GB_ZH_HANT_HK = "MIXED_EN_GB_ZH_HANT_HK"


class FullTextIndexState(StrEnum):
    ONLINE = "ONLINE"
    POPULATING = "POPULATING"
    FAILED = "FAILED"
    MISSING = "MISSING"


class FullTextProfile(StrEnum):
    FIXTURE_REPLAY = "FIXTURE_REPLAY"
    PRODUCTION_SHAPED_QUALIFICATION = "PRODUCTION_SHAPED_QUALIFICATION"
    NATIVE_RUNTIME = "NATIVE_RUNTIME"


def _bounded_text(
    value: str,
    *,
    field: str,
    maximum_bytes: int = 4_096,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value.encode("utf-8")) > maximum_bytes
    ):
        raise FullTextContractError(f"{field} must be bounded canonical text")
    return value


def _non_negative(value: int, *, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_SAFE_INTEGER
    ):
        raise FullTextContractError(
            f"{field} must be a canonical non-negative integer"
        )
    return value


def _sorted_unique(
    value: tuple[str, ...],
    *,
    field: str,
    maximum_items: int = FULLTEXT_MAX_TERMS,
    maximum_item_bytes: int = 1_024,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise FullTextContractError(f"{field} must be an immutable tuple")
    if not allow_empty and not value:
        raise FullTextContractError(f"{field} cannot be empty")
    if len(value) > maximum_items:
        raise FullTextContractError(f"{field} exceeds its item bound")
    normalized = tuple(
        _bounded_text(item, field=field, maximum_bytes=maximum_item_bytes)
        for item in value
    )
    if normalized != tuple(sorted(set(normalized))):
        raise FullTextContractError(f"{field} must be sorted and unique")
    return normalized


@dataclass(frozen=True, slots=True)
class AuthorityAliasTerm:
    alias_id: str
    surface_text: str
    normalized_text: str
    valid_from: UtcTimestamp | None
    valid_until: UtcTimestamp | None
    rights_current: bool
    lifecycle: str

    def __post_init__(self) -> None:
        _bounded_text(self.alias_id, field="authority_alias_id", maximum_bytes=128)
        _bounded_text(
            self.surface_text,
            field="authority_alias_surface_text",
            maximum_bytes=1_024,
        )
        _bounded_text(
            self.normalized_text,
            field="authority_alias_normalized_text",
            maximum_bytes=1_024,
        )
        if self.valid_from is not None and not isinstance(
            self.valid_from, UtcTimestamp
        ):
            raise FullTextContractError("authority alias valid_from must be typed")
        if self.valid_until is not None and not isinstance(
            self.valid_until, UtcTimestamp
        ):
            raise FullTextContractError("authority alias valid_until must be typed")
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until.value <= self.valid_from.value
        ):
            raise FullTextContractError("authority alias validity window is invalid")
        if not isinstance(self.rights_current, bool):
            raise FullTextContractError("authority alias rights flag must be boolean")
        require_token(self.lifecycle, field="authority_alias_lifecycle")

    def is_eligible_at(self, query_valid_time: UtcTimestamp) -> bool:
        if not isinstance(query_valid_time, UtcTimestamp):
            raise FullTextContractError("authority alias query-valid time must be typed")
        if not self.rights_current or self.lifecycle != "ACTIVE":
            return False
        if (
            self.valid_from is not None
            and query_valid_time.value < self.valid_from.value
        ):
            return False
        return not (
            self.valid_until is not None
            and query_valid_time.value >= self.valid_until.value
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "alias_id": self.alias_id,
            "surface_text": self.surface_text,
            "normalized_text": self.normalized_text,
            "valid_from": (
                None if self.valid_from is None else self.valid_from.to_text()
            ),
            "valid_until": (
                None if self.valid_until is None else self.valid_until.to_text()
            ),
            "rights_current": self.rights_current,
            "lifecycle": self.lifecycle,
        }


@dataclass(frozen=True, slots=True)
class NormalizedFullTextQuery:
    surface_text: str
    normalized_text: str
    language_mode: FullTextLanguageMode
    latin_terms: tuple[str, ...]
    han_bigrams: tuple[str, ...]
    formal_tokens: tuple[str, ...]
    authority_alias_terms: tuple[str, ...]
    authority_alias_ids: tuple[str, ...]
    lucene_query: str
    implementation_version: str = "bilingual-search-normalizer-v1"
    component_digest: str = NORMALIZATION_COMPONENT_DIGEST

    def __post_init__(self) -> None:
        _bounded_text(
            self.surface_text,
            field="fulltext_query_surface",
            maximum_bytes=FULLTEXT_MAX_QUERY_BYTES,
        )
        _bounded_text(
            self.normalized_text,
            field="fulltext_query_normalized",
            maximum_bytes=FULLTEXT_MAX_QUERY_BYTES,
        )
        if not isinstance(self.language_mode, FullTextLanguageMode):
            raise FullTextContractError("full-text language mode must be typed")
        _sorted_unique(self.latin_terms, field="fulltext_latin_terms")
        _sorted_unique(self.han_bigrams, field="fulltext_han_bigrams")
        _sorted_unique(self.formal_tokens, field="fulltext_formal_tokens")
        _sorted_unique(
            self.authority_alias_terms,
            field="fulltext_authority_alias_terms",
            maximum_items=32,
        )
        _sorted_unique(
            self.authority_alias_ids,
            field="fulltext_authority_alias_ids",
            maximum_items=32,
            maximum_item_bytes=128,
        )
        _bounded_text(
            self.lucene_query,
            field="fulltext_lucene_query",
            maximum_bytes=FULLTEXT_MAX_LUCENE_QUERY_BYTES,
        )
        require_token(
            self.implementation_version,
            field="fulltext_normalizer_implementation_version",
        )
        validate_sha256_digest(
            self.component_digest,
            field="fulltext_normalization_component_digest",
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "surface_text": self.surface_text,
            "normalized_text": self.normalized_text,
            "language_mode": self.language_mode.value,
            "latin_terms": list(self.latin_terms),
            "han_bigrams": list(self.han_bigrams),
            "formal_tokens": list(self.formal_tokens),
            "authority_alias_terms": list(self.authority_alias_terms),
            "authority_alias_ids": list(self.authority_alias_ids),
            "lucene_query": self.lucene_query,
            "implementation_version": self.implementation_version,
            "component_digest": self.component_digest,
        }

    @property
    def query_digest(self) -> str:
        return digest_bytes(canonical_json_bytes(self.canonical_value()))

    @classmethod
    def from_canonical_value(
        cls, value: dict[str, object]
    ) -> "NormalizedFullTextQuery":
        return cls(
            surface_text=str(value["surface_text"]),
            normalized_text=str(value["normalized_text"]),
            language_mode=FullTextLanguageMode(str(value["language_mode"])),
            latin_terms=tuple(str(item) for item in value["latin_terms"]),
            han_bigrams=tuple(str(item) for item in value["han_bigrams"]),
            formal_tokens=tuple(str(item) for item in value["formal_tokens"]),
            authority_alias_terms=tuple(
                str(item) for item in value["authority_alias_terms"]
            ),
            authority_alias_ids=tuple(
                str(item) for item in value["authority_alias_ids"]
            ),
            lucene_query=str(value["lucene_query"]),
            implementation_version=str(value["implementation_version"]),
            component_digest=str(value["component_digest"]),
        )


@dataclass(frozen=True, slots=True)
class FullTextProjectionSnapshot:
    generation_id: ProjectionGenerationId
    generation_state: ProjectionGenerationState
    generation_identity_digest: str
    document_label: str
    index_name: str
    index_state: FullTextIndexState
    fulltext_component_digest: str
    normalization_component_digest: str
    rights_manifest_digest: str
    profile: FullTextProfile
    contiguous_ledger_seq: int
    open_gap_count: int
    dead_letter_count: int
    validation_recorded_at: UtcTimestamp
    freshness_deadline: UtcTimestamp
    index_document_count: int
    provider: str = FULLTEXT_PROVIDER
    analyzer: str = FULLTEXT_ANALYZER
    server_version: str = "2026.06.0"
    driver_version: str = "6.2.0"
    projection_role: str = "non-authoritative-rebuildable-context"

    def __post_init__(self) -> None:
        if not isinstance(self.generation_id, ProjectionGenerationId):
            raise FullTextContractError(
                "full-text projection generation identity must be typed"
            )
        if not isinstance(self.generation_state, ProjectionGenerationState):
            raise FullTextContractError(
                "full-text projection generation state must be typed"
            )
        for field_name in (
            "generation_identity_digest",
            "fulltext_component_digest",
            "normalization_component_digest",
            "rights_manifest_digest",
        ):
            validate_sha256_digest(
                getattr(self, field_name),
                field=field_name,
            )
        for field_name in ("document_label", "index_name"):
            if _NEO4J_NAME.fullmatch(getattr(self, field_name)) is None:
                raise FullTextContractError(
                    f"{field_name} is not a bounded server-derived name"
                )
        if not isinstance(self.index_state, FullTextIndexState):
            raise FullTextContractError("full-text index state must be typed")
        if not isinstance(self.profile, FullTextProfile):
            raise FullTextContractError("full-text profile must be typed")
        for field_name in (
            "contiguous_ledger_seq",
            "open_gap_count",
            "dead_letter_count",
            "index_document_count",
        ):
            _non_negative(getattr(self, field_name), field=field_name)
        if not isinstance(self.validation_recorded_at, UtcTimestamp) or not isinstance(
            self.freshness_deadline, UtcTimestamp
        ):
            raise FullTextContractError("full-text snapshot times must be typed")
        if self.freshness_deadline.value < self.validation_recorded_at.value:
            raise FullTextContractError(
                "full-text freshness deadline precedes validation"
            )
        for field_name in (
            "provider",
            "analyzer",
            "server_version",
            "driver_version",
        ):
            _bounded_text(
                getattr(self, field_name),
                field=f"fulltext_{field_name}",
                maximum_bytes=128,
            )
        if self.projection_role != "non-authoritative-rebuildable-context":
            raise FullTextContractError("full-text projection role is invalid")

    def canonical_value(self) -> dict[str, object]:
        return {
            "generation_id": str(self.generation_id),
            "generation_state": self.generation_state.value,
            "generation_identity_digest": self.generation_identity_digest,
            "document_label": self.document_label,
            "index_name": self.index_name,
            "index_state": self.index_state.value,
            "fulltext_component_digest": self.fulltext_component_digest,
            "normalization_component_digest": self.normalization_component_digest,
            "rights_manifest_digest": self.rights_manifest_digest,
            "profile": self.profile.value,
            "contiguous_ledger_seq": self.contiguous_ledger_seq,
            "open_gap_count": self.open_gap_count,
            "dead_letter_count": self.dead_letter_count,
            "validation_recorded_at": self.validation_recorded_at.to_text(),
            "freshness_deadline": self.freshness_deadline.to_text(),
            "index_document_count": self.index_document_count,
            "provider": self.provider,
            "analyzer": self.analyzer,
            "server_version": self.server_version,
            "driver_version": self.driver_version,
            "projection_role": self.projection_role,
        }

    @property
    def snapshot_digest(self) -> str:
        return digest_bytes(canonical_json_bytes(self.canonical_value()))

    @classmethod
    def from_canonical_value(
        cls, value: dict[str, object]
    ) -> "FullTextProjectionSnapshot":
        return cls(
            generation_id=ProjectionGenerationId.parse(
                str(value["generation_id"])
            ),
            generation_state=ProjectionGenerationState(
                str(value["generation_state"])
            ),
            generation_identity_digest=str(value["generation_identity_digest"]),
            document_label=str(value["document_label"]),
            index_name=str(value["index_name"]),
            index_state=FullTextIndexState(str(value["index_state"])),
            fulltext_component_digest=str(value["fulltext_component_digest"]),
            normalization_component_digest=str(
                value["normalization_component_digest"]
            ),
            rights_manifest_digest=str(value["rights_manifest_digest"]),
            profile=FullTextProfile(str(value["profile"])),
            contiguous_ledger_seq=int(value["contiguous_ledger_seq"]),
            open_gap_count=int(value["open_gap_count"]),
            dead_letter_count=int(value["dead_letter_count"]),
            validation_recorded_at=UtcTimestamp.parse(
                str(value["validation_recorded_at"])
            ),
            freshness_deadline=UtcTimestamp.parse(
                str(value["freshness_deadline"])
            ),
            index_document_count=int(value["index_document_count"]),
            provider=str(value["provider"]),
            analyzer=str(value["analyzer"]),
            server_version=str(value["server_version"]),
            driver_version=str(value["driver_version"]),
            projection_role=str(value["projection_role"]),
        )


@dataclass(frozen=True, slots=True)
class FullTextDocumentBinding:
    passage_id: str
    dependency_root_id: str
    source_id: str
    source_identity: str
    provenance_digest: str
    language: str
    rights_current: bool
    lifecycle: str
    valid_from: UtcTimestamp | None = None
    valid_until: UtcTimestamp | None = None
    trust_scope: TrustScope = TrustScope.OBSERVED

    def __post_init__(self) -> None:
        try:
            require_token(self.passage_id, field="fulltext_passage_id")
        except ValueError:
            try:
                if str(uuid.UUID(self.passage_id)) != self.passage_id:
                    raise ValueError
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(
                    "fulltext_passage_id is not a valid authority token or canonical UUID"
                ) from exc
        for field_name in (
            "dependency_root_id",
            "source_id",
            "source_identity",
        ):
            _bounded_text(
                getattr(self, field_name),
                field=field_name,
                maximum_bytes=256,
            )
        validate_sha256_digest(
            self.provenance_digest,
            field="fulltext_binding_provenance_digest",
        )
        if self.language not in {"en-GB", "zh-HK"}:
            raise FullTextContractError("full-text binding language is invalid")
        if not isinstance(self.rights_current, bool):
            raise FullTextContractError(
                "full-text binding rights flag must be boolean"
            )
        require_token(self.lifecycle, field="fulltext_binding_lifecycle")
        if self.valid_from is not None and not isinstance(
            self.valid_from, UtcTimestamp
        ):
            raise FullTextContractError("full-text binding valid_from must be typed")
        if self.valid_until is not None and not isinstance(
            self.valid_until, UtcTimestamp
        ):
            raise FullTextContractError("full-text binding valid_until must be typed")
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until.value <= self.valid_from.value
        ):
            raise FullTextContractError("full-text binding validity window is invalid")
        if self.trust_scope is not TrustScope.OBSERVED:
            raise FullTextContractError(
                "full-text projection bindings remain OBSERVED"
            )

    def exclusion_at(
        self, query_valid_time: UtcTimestamp
    ) -> BranchExclusionReason | None:
        if not isinstance(query_valid_time, UtcTimestamp):
            raise FullTextContractError(
                "full-text binding query-valid time must be typed"
            )
        if not self.rights_current:
            return BranchExclusionReason.RIGHTS_NOT_CURRENT
        if self.lifecycle in {
            "TOMBSTONED",
            "RETIRED",
            "REJECTED",
            "REVOKED",
            "MERGED",
            "SPLIT",
            "REVERSED",
        }:
            return BranchExclusionReason.TOMBSTONED
        if self.lifecycle != "ACTIVE":
            return BranchExclusionReason.STALE_SOURCE_VERSION
        if (
            self.valid_from is not None
            and query_valid_time.value < self.valid_from.value
        ) or (
            self.valid_until is not None
            and query_valid_time.value >= self.valid_until.value
        ):
            return BranchExclusionReason.OUTSIDE_QUERY_VALID_TIME
        return None

    def canonical_value(self) -> dict[str, object]:
        return {
            "passage_id": self.passage_id,
            "dependency_root_id": self.dependency_root_id,
            "source_id": self.source_id,
            "source_identity": self.source_identity,
            "provenance_digest": self.provenance_digest,
            "language": self.language,
            "rights_current": self.rights_current,
            "lifecycle": self.lifecycle,
            "valid_from": (
                None if self.valid_from is None else self.valid_from.to_text()
            ),
            "valid_until": (
                None if self.valid_until is None else self.valid_until.to_text()
            ),
            "trust_scope": self.trust_scope.value,
        }


@dataclass(frozen=True, slots=True)
class FullTextCorpusScope:
    """Complete eligibility metadata, separate from bounded authenticated hits."""
    inventory_digest: str
    eligible_passages: tuple[tuple[str, str], ...]
    resolve: Callable = field(repr=False, compare=False)

    def __post_init__(self):
        validate_sha256_digest(self.inventory_digest)
        if (type(self.eligible_passages) is not tuple or not callable(self.resolve)
                or any(type(item) is not tuple or len(item) != 2
                       for item in self.eligible_passages)):
            raise FullTextContractError("full-text corpus metadata differs")
        for passage_id, source_id in self.eligible_passages:
            _bounded_text(passage_id, field="fulltext_corpus_passage", maximum_bytes=256)
            _bounded_text(source_id, field="fulltext_corpus_source", maximum_bytes=256)
        passages = tuple(item[0] for item in self.eligible_passages)
        if passages != tuple(sorted(set(passages))):
            raise FullTextContractError("full-text corpus passages must be sorted and unique")

    def canonical_value(self):
        return {"inventory_digest": self.inventory_digest,
                "eligible_passages": [list(item) for item in self.eligible_passages]}


@dataclass(frozen=True, slots=True)
class FullTextAuthorityView:
    snapshot: FullTextProjectionSnapshot
    authority_aliases: tuple[AuthorityAliasTerm, ...]
    document_bindings: tuple[FullTextDocumentBinding, ...]
    corpus_scope: FullTextCorpusScope | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, FullTextProjectionSnapshot):
            raise FullTextContractError("full-text authority snapshot must be typed")
        if (
            not isinstance(self.authority_aliases, tuple)
            or len(self.authority_aliases) > 32
            or any(
                not isinstance(item, AuthorityAliasTerm)
                for item in self.authority_aliases
            )
        ):
            raise FullTextContractError(
                "full-text authority aliases must be a bounded typed tuple"
            )
        alias_ids = tuple(item.alias_id for item in self.authority_aliases)
        if alias_ids != tuple(sorted(set(alias_ids))):
            raise FullTextContractError(
                "full-text authority aliases must be sorted and unique"
            )
        if (
            not isinstance(self.document_bindings, tuple)
            or len(self.document_bindings) > 4_096
            or any(
                not isinstance(item, FullTextDocumentBinding)
                for item in self.document_bindings
            )
        ):
            raise FullTextContractError(
                "full-text document bindings must be a bounded typed tuple"
            )
        passage_ids = tuple(item.passage_id for item in self.document_bindings)
        if passage_ids != tuple(sorted(set(passage_ids))):
            raise FullTextContractError(
                "full-text document bindings must be sorted and unique"
            )
        if self.corpus_scope is not None and (
            type(self.corpus_scope) is not FullTextCorpusScope or self.document_bindings
            or self.snapshot.profile is not FullTextProfile.NATIVE_RUNTIME
            or self.snapshot.index_document_count != len(self.corpus_scope.eligible_passages)
        ):
            raise FullTextContractError("full-text complete corpus scope differs")

    @property
    def binding_by_passage_id(self) -> dict[str, FullTextDocumentBinding]:
        return {item.passage_id: item for item in self.document_bindings}

    def canonical_value(self) -> dict[str, object]:
        value = {
            "snapshot": self.snapshot.canonical_value(),
            "authority_aliases": [
                item.canonical_value() for item in self.authority_aliases
            ],
            "document_bindings": [
                item.canonical_value() for item in self.document_bindings
            ],
        }
        if self.corpus_scope is not None:
            value["corpus_scope"] = self.corpus_scope.canonical_value()
        return value

    @property
    def view_digest(self) -> str:
        return digest_bytes(canonical_json_bytes(self.canonical_value()))


@dataclass(frozen=True, slots=True)
class FullTextBranchRequest:
    request_id: BranchRequestId
    idempotency_key: str
    actor_id: str
    purpose: str
    policy_id: str
    contract_digest: str
    fulltext_component_digest: str
    normalization_component_digest: str
    expected_generation_id: ProjectionGenerationId
    expected_generation_identity_digest: str
    expected_rights_manifest_digest: str
    query_text: str
    language_mode: FullTextLanguageMode
    source_ids: tuple[str, ...]
    query_valid_time: UtcTimestamp
    serving_time: UtcTimestamp
    minimum_watermark: int
    result_limit: int = BRANCH_RESULT_LIMIT
    timeout_ms: int = BRANCH_TIMEOUT_MS
    response_byte_limit: int = FULLTEXT_RESPONSE_BYTE_LIMIT
    max_projection_age_seconds: int = FULLTEXT_MAX_PROJECTION_AGE_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, BranchRequestId):
            raise FullTextContractError(
                "full-text request identity must be typed"
            )
        _bounded_text(
            self.idempotency_key,
            field="fulltext_idempotency_key",
            maximum_bytes=256,
        )
        require_token(self.actor_id, field="fulltext_actor_id")
        require_token(self.purpose, field="fulltext_purpose")
        if self.actor_id != FULLTEXT_ACTOR_ID or self.purpose != FULLTEXT_PURPOSE:
            raise FullTextContractError(
                "full-text actor and purpose must equal the reviewed lane"
            )
        require_token(self.policy_id, field="fulltext_policy_id")
        for field_name in (
            "contract_digest",
            "fulltext_component_digest",
            "normalization_component_digest",
            "expected_generation_identity_digest",
            "expected_rights_manifest_digest",
        ):
            validate_sha256_digest(getattr(self, field_name), field=field_name)
        if not isinstance(
            self.expected_generation_id, ProjectionGenerationId
        ):
            raise FullTextContractError(
                "expected full-text generation identity must be typed"
            )
        _bounded_text(
            self.query_text,
            field="fulltext_query_text",
            maximum_bytes=FULLTEXT_MAX_QUERY_BYTES,
        )
        if not isinstance(self.language_mode, FullTextLanguageMode):
            raise FullTextContractError("full-text language mode must be typed")
        object.__setattr__(
            self,
            "source_ids",
            _sorted_unique(
                self.source_ids,
                field="fulltext_source_ids",
                maximum_items=BRANCH_RESULT_LIMIT,
                maximum_item_bytes=256,
            ),
        )
        if not isinstance(self.query_valid_time, UtcTimestamp) or not isinstance(
            self.serving_time, UtcTimestamp
        ):
            raise FullTextContractError("full-text request times must be typed")
        _non_negative(self.minimum_watermark, field="minimum_watermark")
        if self.result_limit != BRANCH_RESULT_LIMIT:
            raise FullTextContractError(
                "full-text result limit must remain fixed at 8"
            )
        if self.timeout_ms != BRANCH_TIMEOUT_MS:
            raise FullTextContractError(
                "full-text timeout must remain fixed at 5000 ms"
            )
        if self.response_byte_limit != FULLTEXT_RESPONSE_BYTE_LIMIT:
            raise FullTextContractError(
                "full-text response byte limit must remain fixed"
            )
        if (
            self.max_projection_age_seconds
            != FULLTEXT_MAX_PROJECTION_AGE_SECONDS
        ):
            raise FullTextContractError(
                "full-text projection age limit must remain fixed"
            )

    def canonical_value(self) -> dict[str, object]:
        return {
            "schema_version": "newsroom.increment5.fulltext-branch-request.v2",
            "request_id": str(self.request_id),
            "idempotency_key": self.idempotency_key,
            "actor_id": self.actor_id,
            "purpose": self.purpose,
            "policy_id": self.policy_id,
            "contract_digest": self.contract_digest,
            "fulltext_component_digest": self.fulltext_component_digest,
            "normalization_component_digest": (
                self.normalization_component_digest
            ),
            "expected_generation_id": str(self.expected_generation_id),
            "expected_generation_identity_digest": (
                self.expected_generation_identity_digest
            ),
            "expected_rights_manifest_digest": (
                self.expected_rights_manifest_digest
            ),
            "query_text": self.query_text,
            "language_mode": self.language_mode.value,
            "source_ids": list(self.source_ids),
            "query_valid_time": self.query_valid_time.to_text(),
            "serving_time": self.serving_time.to_text(),
            "minimum_watermark": self.minimum_watermark,
            "result_limit": self.result_limit,
            "timeout_ms": self.timeout_ms,
            "response_byte_limit": self.response_byte_limit,
            "max_projection_age_seconds": (
                self.max_projection_age_seconds
            ),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_value())

    @property
    def request_digest(self) -> str:
        return digest_bytes(self.canonical_bytes)


__all__ = [
    "AuthorityAliasTerm",
    "FULLTEXT_ACTOR_ID",
    "FULLTEXT_ANALYZER",
    "FULLTEXT_COMPONENT_DIGEST",
    "FULLTEXT_INDEXED_FIELDS",
    "FULLTEXT_MAX_LUCENE_QUERY_BYTES",
    "FULLTEXT_MAX_PROJECTION_AGE_SECONDS",
    "FULLTEXT_MAX_QUERY_BYTES",
    "FULLTEXT_MAX_TERMS",
    "FULLTEXT_POLICY_ID",
    "FULLTEXT_PROVIDER",
    "FULLTEXT_PURPOSE",
    "FULLTEXT_QUERY_ID",
    "FULLTEXT_RESPONSE_BYTE_LIMIT",
    "FullTextAuthorityView",
    "FullTextBranchRequest",
    "FullTextContractError",
    "FullTextDocumentBinding",
    "FullTextIndexState",
    "FullTextLanguageMode",
    "FullTextProfile",
    "FullTextProjectionSnapshot",
    "INCREMENT5_RETRIEVAL_CONTRACT_DIGEST",
    "NORMALIZATION_COMPONENT_DIGEST",
    "NormalizedFullTextQuery",
]
