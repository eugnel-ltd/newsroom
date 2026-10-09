"""Bounded authority port for Increment 5 full-text Neo4j reads."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import re
from types import MappingProxyType

from newsroom.projection.models import ProjectionGenerationId
from newsroom.authority.canonical import validate_sha256_digest


class Neo4jFullTextReadError(RuntimeError):
    """A bounded full-text graph read is unavailable or malformed."""


class Neo4jFullTextReadTimeout(Neo4jFullTextReadError):
    """A bounded full-text graph-read budget expired."""


class Neo4jFullTextReadPhase(StrEnum):
    COMPONENT = "COMPONENT"
    INDEX = "INDEX"
    QUERY = "QUERY"


_NEO4J_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")

FULLTEXT_SOURCE_SCOPE_CANDIDATE_LIMIT = 65


def _bounded_text(value: str, *, field: str, maximum_bytes: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value.encode("utf-8")) > maximum_bytes
        or any(ord(character) < 0x20 for character in value)
    ):
        raise Neo4jFullTextReadError(f"{field} must be bounded canonical text")
    return value


def _timeout(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= 5_000_000_000
    ):
        raise Neo4jFullTextReadError(
            "full-text graph-read timeout must be within 5,000 ms"
        )
    return value


def _index_name(value: str) -> str:
    _bounded_text(value, field="fulltext_index_name", maximum_bytes=128)
    if _NEO4J_NAME.fullmatch(value) is None:
        raise Neo4jFullTextReadError(
            "full-text index name must be a server-derived Neo4j name"
        )
    return value


def _copy_record(
    value: Mapping[str, object] | None,
    *,
    field: str,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise Neo4jFullTextReadError(f"{field} must be a mapping")
    return MappingProxyType(dict(value))


def _copy_records(
    values: tuple[Mapping[str, object], ...],
    *,
    field: str,
    maximum: int,
) -> tuple[Mapping[str, object], ...]:
    if not isinstance(values, tuple) or len(values) > maximum:
        raise Neo4jFullTextReadError(f"{field} exceeds its fixed bound")
    copied: list[Mapping[str, object]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise Neo4jFullTextReadError(f"{field} contains a malformed record")
        copied.append(MappingProxyType(dict(value)))
    return tuple(copied)


@dataclass(frozen=True, slots=True)
class Neo4jFullTextReadRequest:
    phase: Neo4jFullTextReadPhase
    timeout_ns: int
    index_name: str | None = None
    lucene_expression: str | None = None
    generation_id: ProjectionGenerationId | None = None
    source_ids: tuple[str, ...] = ()
    eligible_passage_ids: tuple[str, ...] | None = None
    corpus_scope_digest: str | None = None
    limit: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.phase, Neo4jFullTextReadPhase):
            raise Neo4jFullTextReadError("full-text read phase must be typed")
        _timeout(self.timeout_ns)
        if self.corpus_scope_digest is not None:
            validate_sha256_digest(self.corpus_scope_digest)
            if self.phase is not Neo4jFullTextReadPhase.QUERY or self.eligible_passage_ids is None:
                raise Neo4jFullTextReadError("complete corpus metadata requires a scoped query")
        if self.phase is Neo4jFullTextReadPhase.COMPONENT:
            if any(
                value is not None
                for value in (
                    self.index_name,
                    self.lucene_expression,
                    self.generation_id,
                )
            ) or self.source_ids or self.eligible_passage_ids is not None or self.limit != 0:
                raise Neo4jFullTextReadError(
                    "component read cannot carry index or query controls"
                )
            return
        if self.index_name is None:
            raise Neo4jFullTextReadError("full-text index read requires an index")
        _index_name(self.index_name)
        if self.phase is Neo4jFullTextReadPhase.INDEX:
            if (
                self.lucene_expression is not None
                or self.generation_id is not None
                or self.source_ids
                or self.eligible_passage_ids is not None
                or self.limit != 0
            ):
                raise Neo4jFullTextReadError(
                    "index read cannot carry query controls"
                )
            return
        if self.lucene_expression is None:
            raise Neo4jFullTextReadError(
                "full-text query read requires a Lucene expression"
            )
        _bounded_text(
            self.lucene_expression,
            field="fulltext_lucene_expression",
            maximum_bytes=32_768,
        )
        if not isinstance(self.generation_id, ProjectionGenerationId):
            raise Neo4jFullTextReadError(
                "full-text generation identity must be typed"
            )
        if not isinstance(self.source_ids, tuple) or len(self.source_ids) > 8:
            raise Neo4jFullTextReadError(
                "full-text source scope exceeds its fixed bound"
            )
        for source_id in self.source_ids:
            _bounded_text(
                source_id,
                field="fulltext_source_id",
                maximum_bytes=256,
            )
        if self.source_ids != tuple(sorted(set(self.source_ids))):
            raise Neo4jFullTextReadError(
                "full-text source scope must be sorted and unique"
            )
        if self.eligible_passage_ids is None:
            if isinstance(self.limit, bool) or self.limit != 9:
                raise Neo4jFullTextReadError(
                    "full-text overflow sentinel limit must equal 9"
                )
        else:
            if (
                not isinstance(self.eligible_passage_ids, tuple)
                or self.corpus_scope_digest is None and len(self.eligible_passage_ids) > 4_096
            ):
                raise Neo4jFullTextReadError(
                    "native full-text eligible passages exceed their fixed bound"
                )
            for passage_id in self.eligible_passage_ids:
                _bounded_text(
                    passage_id,
                    field="native_fulltext_eligible_passage_id",
                    maximum_bytes=256,
                )
            if self.eligible_passage_ids != tuple(
                sorted(set(self.eligible_passage_ids))
            ):
                raise Neo4jFullTextReadError(
                    "native full-text eligible passages must be sorted and unique"
                )
            if isinstance(self.limit, bool) or self.limit != 8:
                raise Neo4jFullTextReadError(
                    "native full-text result limit must equal 8"
                )

    @classmethod
    def component(cls, *, timeout_ns: int) -> "Neo4jFullTextReadRequest":
        return cls(
            phase=Neo4jFullTextReadPhase.COMPONENT,
            timeout_ns=timeout_ns,
        )

    @classmethod
    def index(
        cls,
        *,
        index_name: str,
        timeout_ns: int,
    ) -> "Neo4jFullTextReadRequest":
        return cls(
            phase=Neo4jFullTextReadPhase.INDEX,
            timeout_ns=timeout_ns,
            index_name=index_name,
        )

    @classmethod
    def query(
        cls,
        *,
        index_name: str,
        lucene_expression: str,
        generation_id: ProjectionGenerationId,
        source_ids: tuple[str, ...],
        limit: int,
        timeout_ns: int,
        eligible_passage_ids: tuple[str, ...] | None = None,
        corpus_scope_digest: str | None = None,
    ) -> "Neo4jFullTextReadRequest":
        return cls(
            phase=Neo4jFullTextReadPhase.QUERY,
            timeout_ns=timeout_ns,
            index_name=index_name,
            lucene_expression=lucene_expression,
            generation_id=generation_id,
            source_ids=source_ids,
            eligible_passage_ids=eligible_passage_ids,
            corpus_scope_digest=corpus_scope_digest,
            limit=limit,
        )


@dataclass(frozen=True, slots=True)
class Neo4jFullTextReadResult:
    phase: Neo4jFullTextReadPhase
    driver_version: str
    component: Mapping[str, object] | None = None
    indexes: tuple[Mapping[str, object], ...] = ()
    rows: tuple[Mapping[str, object], ...] = ()
    candidate_overflow: bool = False
    read_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.phase, Neo4jFullTextReadPhase):
            raise Neo4jFullTextReadError("full-text result phase must be typed")
        _bounded_text(
            self.driver_version,
            field="fulltext_driver_version",
            maximum_bytes=64,
        )
        if self.read_count != 1:
            raise Neo4jFullTextReadError(
                "one full-text phase must retain exactly one Neo4j read"
            )
        component = _copy_record(self.component, field="fulltext_component")
        indexes = _copy_records(
            self.indexes,
            field="fulltext_indexes",
            maximum=16,
        )
        rows = _copy_records(
            self.rows,
            field="fulltext_rows",
            maximum=FULLTEXT_SOURCE_SCOPE_CANDIDATE_LIMIT,
        )
        if type(self.candidate_overflow) is not bool:
            raise Neo4jFullTextReadError(
                "full-text candidate overflow flag must be boolean"
            )
        if self.phase is Neo4jFullTextReadPhase.COMPONENT:
            if indexes or rows or self.candidate_overflow:
                raise Neo4jFullTextReadError(
                    "component result cannot carry index or query rows"
                )
        elif self.phase is Neo4jFullTextReadPhase.INDEX:
            if component is not None or rows or self.candidate_overflow:
                raise Neo4jFullTextReadError(
                    "index result cannot carry component or query rows"
                )
        elif component is not None or indexes:
            raise Neo4jFullTextReadError(
                "query result cannot carry component or index rows"
            )
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "indexes", indexes)
        object.__setattr__(self, "rows", rows)


class Neo4jFullTextReader:
    """Capability facade exposing one phased read, never a driver or session."""

    __slots__ = ("__driver_version", "__read", "__close", "__closed")

    def __init__(
        self,
        *,
        driver_version: str,
        read: Callable[[Neo4jFullTextReadRequest], Neo4jFullTextReadResult],
        close: Callable[[], None] | None = None,
    ) -> None:
        _bounded_text(
            driver_version,
            field="fulltext_driver_version",
            maximum_bytes=64,
        )
        if not callable(read) or (close is not None and not callable(close)):
            raise TypeError("full-text authority port requires callable boundaries")
        self.__driver_version = driver_version
        self.__read = read
        self.__close = close or (lambda: None)
        self.__closed = False

    @property
    def driver_version(self) -> str:
        return self.__driver_version

    def read(self, request: Neo4jFullTextReadRequest) -> Neo4jFullTextReadResult:
        if not isinstance(request, Neo4jFullTextReadRequest):
            raise TypeError("full-text graph read requires a typed request")
        if self.__closed:
            raise Neo4jFullTextReadError("full-text authority port is closed")
        result = self.__read(request)
        if not isinstance(result, Neo4jFullTextReadResult):
            raise Neo4jFullTextReadError(
                "full-text authority port returned an untyped result"
            )
        if result.phase is not request.phase:
            raise Neo4jFullTextReadError(
                "full-text authority port returned another read phase"
            )
        if result.driver_version != self.__driver_version:
            raise Neo4jFullTextReadError(
                "full-text authority port driver identity changed"
            )
        if request.eligible_passage_ids is not None and (
            result.candidate_overflow or len(result.rows) > request.limit
        ):
            raise Neo4jFullTextReadError(
                "native full-text result exceeds its fixed bound"
            )
        if request.eligible_passage_ids is not None:
            eligible = set(request.eligible_passage_ids)
            if any(
                type(row.get("passage_id")) is not str
                or row["passage_id"] not in eligible
                for row in result.rows
            ):
                raise Neo4jFullTextReadError(
                    "native full-text result is outside its authority scope"
                )
        return result

    def close(self) -> None:
        if self.__closed:
            return
        self.__closed = True
        self.__close()

    def __enter__(self) -> "Neo4jFullTextReader":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()


__all__ = [
    "FULLTEXT_SOURCE_SCOPE_CANDIDATE_LIMIT",
    "Neo4jFullTextReadError",
    "Neo4jFullTextReadPhase",
    "Neo4jFullTextReadRequest",
    "Neo4jFullTextReadResult",
    "Neo4jFullTextReadTimeout",
    "Neo4jFullTextReader",
]
