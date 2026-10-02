"""Controller-owned identities and call-shape policy for Graphiti leaves."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from newsroom.authority.canonical import digest_canonical, validate_sha256_digest

GRAPHITI_INTERNAL_REQUEST_SCHEMA_VERSION = (
    "newsroom.graphiti-internal-request.v1"
)
GRAPHITI_CALL_SHAPE_SCHEMA_VERSION = "newsroom.graphiti-call-shape-policy.v1"
_POLICY_PATH = Path(__file__).with_name("graphiti_call_shape_policy_v1.json")
_NATIVE_POLICY_PATH = Path(__file__).with_name(
    "native_graphiti_call_shape_policy_v2.json"
)
ALLOWED_GRAPHITI_SEMANTIC_REQUEST_CLASSES = frozenset(
    {
        "ExtractedEntities",
        "NodeResolutions",
        "ExtractedEdges",
        "EdgeTimestamps",
        "EdgeDuplicate",
        "SummarizedEntities",
        "CombinedExtraction",
        "BatchEdgeTimestamps",
        "NewsroomCombinedTemporalExtractionV1",
        "EMBEDDING_VECTOR",
        "UNSTRUCTURED",
    }
)


class GraphitiRequestContractError(ValueError):
    """A Graphiti request identity or checked call-shape policy is invalid."""


class GraphitiLeafClass(StrEnum):
    PRIMARY = "PRIMARY"
    FALLBACK = "FALLBACK"
    EMBEDDING = "EMBEDDING"


def _token(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value.encode("utf-8")) > 512
    ):
        raise GraphitiRequestContractError(f"{field} must be bounded canonical text")
    return value


def _digest(value: str, *, field: str) -> str:
    try:
        retained = validate_sha256_digest(value, field=field)
    except ValueError as exc:
        raise GraphitiRequestContractError(str(exc)) from exc
    if retained != value:
        raise GraphitiRequestContractError(f"{field} must use canonical lowercase text")
    return value


def _positive(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GraphitiRequestContractError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class GraphitiQualifiedRoute:
    leaf_class: GraphitiLeafClass
    provider: str
    route: str
    model: str
    reasoning: str
    config_identity: str
    command_semantic_version: str
    command_flags: tuple[str, ...]
    disabled_capabilities: tuple[str, ...]
    implementation_revision: str
    max_prompt_bytes: int
    max_context_tokens: int
    max_output_tokens: int
    max_total_tokens: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GraphitiQualifiedRoute:
        try:
            leaf_class = GraphitiLeafClass(str(value.get("leaf_class", "")))
        except ValueError as exc:
            raise GraphitiRequestContractError(
                "qualified Graphiti route has an invalid leaf class"
            ) from exc
        route = cls(
            leaf_class=leaf_class,
            provider=str(value.get("provider", "")),
            route=str(value.get("route", "")),
            model=str(value.get("model", "")),
            reasoning=str(value.get("reasoning", "")),
            config_identity=str(value.get("config_identity", "")),
            command_semantic_version=str(value.get("command_semantic_version", "")),
            command_flags=_string_tuple(value.get("command_flags")),
            disabled_capabilities=_string_tuple(value.get("disabled_capabilities")),
            implementation_revision=str(value.get("implementation_revision", "")),
            max_prompt_bytes=value.get("max_prompt_bytes", 0),  # type: ignore[arg-type]
            max_context_tokens=value.get("max_context_tokens", 0),  # type: ignore[arg-type]
            max_output_tokens=value.get("max_output_tokens", 0),  # type: ignore[arg-type]
            max_total_tokens=value.get("max_total_tokens", 0),  # type: ignore[arg-type]
        )
        for name in (
            "provider",
            "route",
            "model",
            "reasoning",
            "config_identity",
            "command_semantic_version",
            "implementation_revision",
        ):
            _token(str(getattr(route, name)), field=f"qualified route {name}")
        if not route.command_flags or not route.disabled_capabilities:
            raise GraphitiRequestContractError(
                "qualified Graphiti route lacks command/capability bindings"
            )
        for item in (*route.command_flags, *route.disabled_capabilities):
            _token(item, field="qualified route command value")
        for name in (
            "max_prompt_bytes",
            "max_context_tokens",
            "max_output_tokens",
            "max_total_tokens",
        ):
            _positive(getattr(route, name), field=f"qualified route {name}")
        return route

    def as_record(self) -> dict[str, object]:
        return {
            "leaf_class": self.leaf_class.value,
            "provider": self.provider,
            "route": self.route,
            "model": self.model,
            "reasoning": self.reasoning,
            "config_identity": self.config_identity,
            "command_semantic_version": self.command_semantic_version,
            "command_flags": list(self.command_flags),
            "disabled_capabilities": list(self.disabled_capabilities),
            "implementation_revision": self.implementation_revision,
            "max_prompt_bytes": self.max_prompt_bytes,
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_total_tokens": self.max_total_tokens,
        }


@dataclass(frozen=True, slots=True)
class GraphitiQualifiedRequestShape:
    """One checked semantic class/schema pair admitted for a leaf class."""

    leaf_class: GraphitiLeafClass
    semantic_request_class: str
    response_schema_identity: str
    response_schema_digest: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> GraphitiQualifiedRequestShape:
        try:
            leaf_class = GraphitiLeafClass(str(value.get("leaf_class", "")))
        except ValueError as exc:
            raise GraphitiRequestContractError(
                "qualified Graphiti request has an invalid leaf class"
            ) from exc
        shape = cls(
            leaf_class=leaf_class,
            semantic_request_class=str(value.get("semantic_request_class", "")),
            response_schema_identity=str(value.get("response_schema_identity", "")),
            response_schema_digest=str(value.get("response_schema_digest", "")),
        )
        _token(shape.semantic_request_class, field="semantic request class")
        _token(shape.response_schema_identity, field="response schema identity")
        _digest(shape.response_schema_digest, field="response schema digest")
        return shape

    def as_record(self) -> dict[str, object]:
        return {
            "leaf_class": self.leaf_class.value,
            "semantic_request_class": self.semantic_request_class,
            "response_schema_identity": self.response_schema_identity,
            "response_schema_digest": self.response_schema_digest,
        }


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise GraphitiRequestContractError(
            "qualified Graphiti route string list is invalid"
        )
    return tuple(value)


@dataclass(frozen=True, slots=True)
class GraphitiCallShapeFixture:
    fixture_id: str
    fixture_class: str
    fixture_source: str
    primary_chat_request_count: int
    fallback_chat_request_count: int
    embedding_request_count: int
    distinct_internal_request_count: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GraphitiCallShapeFixture:
        distinct_count = value.get("distinct_internal_request_count", 0)
        primary_count = value.get("primary_chat_request_count", distinct_count)
        fallback_count = value.get("fallback_chat_request_count", 0)
        embedding_count = value.get("embedding_request_count", 0)
        fixture = cls(
            fixture_id=str(value.get("fixture_id", "")),
            fixture_class=str(value.get("fixture_class", "")),
            fixture_source=str(value.get("fixture_source", "INLINE_TEST_FIXTURE")),
            primary_chat_request_count=primary_count,  # type: ignore[arg-type]
            fallback_chat_request_count=fallback_count,  # type: ignore[arg-type]
            embedding_request_count=embedding_count,  # type: ignore[arg-type]
            distinct_internal_request_count=distinct_count,  # type: ignore[arg-type]
        )
        _token(fixture.fixture_id, field="fixture id")
        _token(fixture.fixture_class, field="fixture class")
        _token(fixture.fixture_source, field="fixture source")
        for name in (
            "primary_chat_request_count",
            "fallback_chat_request_count",
            "embedding_request_count",
        ):
            count = getattr(fixture, name)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise GraphitiRequestContractError(
                    f"{name} must be a non-negative integer"
                )
        _positive(
            fixture.distinct_internal_request_count,
            field="distinct internal request count",
        )
        if fixture.distinct_internal_request_count != (
            fixture.primary_chat_request_count
            + fixture.fallback_chat_request_count
            + fixture.embedding_request_count
        ):
            raise GraphitiRequestContractError(
                "distinct internal request count differs from its leaf derivation"
            )
        return fixture

    def as_record(self) -> dict[str, object]:
        return {
            "fixture_id": self.fixture_id,
            "fixture_class": self.fixture_class,
            "fixture_source": self.fixture_source,
            "primary_chat_request_count": self.primary_chat_request_count,
            "fallback_chat_request_count": self.fallback_chat_request_count,
            "embedding_request_count": self.embedding_request_count,
            "distinct_internal_request_count": self.distinct_internal_request_count,
        }


@dataclass(frozen=True, slots=True)
class GraphitiCallShapePolicy:
    policy_id: str
    version: str
    graphiti_core_release: str
    framework_identity: str
    prompt_identity: str
    ontology_identity: str
    temporal_identity: str
    generation_policy_identity: str
    qualified_routes: tuple[GraphitiQualifiedRoute, ...]
    qualified_request_shapes: tuple[GraphitiQualifiedRequestShape, ...]
    fixtures: tuple[GraphitiCallShapeFixture, ...]
    evidence_digest: str
    maximum_qualified_fixture_count: int
    headroom: int
    max_distinct_internal_requests: int
    canonical_digest: str

    @classmethod
    def create(cls, **values: object) -> GraphitiCallShapePolicy:
        values.pop("maximum_qualified_fixture_count", None)
        values.pop("headroom", None)
        values.pop("max_distinct_internal_requests", None)
        values.pop("canonical_digest", None)
        raw_fixtures = values.get("fixtures")
        if not isinstance(raw_fixtures, tuple) or not raw_fixtures:
            raise GraphitiRequestContractError("call-shape fixtures must be a tuple")
        fixtures = tuple(
            item
            if isinstance(item, GraphitiCallShapeFixture)
            else GraphitiCallShapeFixture.from_mapping(item)
            if isinstance(item, Mapping)
            else (_raise_fixture_type())
            for item in raw_fixtures
        )
        raw_routes = values.get("qualified_routes")
        if not isinstance(raw_routes, tuple) or not raw_routes:
            raise GraphitiRequestContractError("qualified Graphiti routes must be a tuple")
        routes = tuple(
            item
            if isinstance(item, GraphitiQualifiedRoute)
            else GraphitiQualifiedRoute.from_mapping(item)
            if isinstance(item, Mapping)
            else _raise_route_type()
            for item in raw_routes
        )
        raw_request_shapes = values.get("qualified_request_shapes")
        if not isinstance(raw_request_shapes, tuple) or not raw_request_shapes:
            raise GraphitiRequestContractError(
                "qualified Graphiti request shapes must be a tuple"
            )
        request_shapes = tuple(
            item
            if isinstance(item, GraphitiQualifiedRequestShape)
            else GraphitiQualifiedRequestShape.from_mapping(item)
            if isinstance(item, Mapping)
            else _raise_request_shape_type()
            for item in raw_request_shapes
        )
        supplied_evidence_digest = values.pop("evidence_digest", None)
        evidence_digest = digest_canonical(
            {
                "graphiti_core_release": values.get("graphiti_core_release"),
                "fixtures": [item.as_record() for item in fixtures],
                "qualified_routes": [item.as_record() for item in routes],
                "qualified_request_shapes": [
                    item.as_record() for item in request_shapes
                ],
            }
        )
        if (
            supplied_evidence_digest is not None
            and supplied_evidence_digest != evidence_digest
        ):
            raise GraphitiRequestContractError(
                "call-shape evidence digest differs from its fixture derivation"
            )
        values["evidence_digest"] = evidence_digest
        maximum = max(item.distinct_internal_request_count for item in fixtures)
        headroom = max(2, (maximum + 3) // 4)
        maximum_with_headroom = maximum + headroom
        record = {
            "schema_version": GRAPHITI_CALL_SHAPE_SCHEMA_VERSION,
            **{
                key: value
                for key, value in values.items()
                if key
                not in {"fixtures", "qualified_routes", "qualified_request_shapes"}
            },
            "fixtures": [item.as_record() for item in fixtures],
            "qualified_routes": [item.as_record() for item in routes],
            "qualified_request_shapes": [
                item.as_record() for item in request_shapes
            ],
            "maximum_qualified_fixture_count": maximum,
            "headroom": headroom,
            "max_distinct_internal_requests": maximum_with_headroom,
        }
        policy = cls(
            **{
                key: value
                for key, value in values.items()
                if key
                not in {"fixtures", "qualified_routes", "qualified_request_shapes"}
            },  # type: ignore[arg-type]
            qualified_routes=routes,
            qualified_request_shapes=request_shapes,
            fixtures=fixtures,
            maximum_qualified_fixture_count=maximum,
            headroom=headroom,
            max_distinct_internal_requests=maximum_with_headroom,
            canonical_digest=digest_canonical(record),
        )
        policy._validate()
        return policy

    def _validate(self) -> None:
        for name in (
            "policy_id",
            "version",
            "graphiti_core_release",
            "framework_identity",
            "prompt_identity",
            "ontology_identity",
            "temporal_identity",
            "generation_policy_identity",
        ):
            _token(str(getattr(self, name)), field=name)
        _digest(self.evidence_digest, field="call-shape evidence digest")
        expected_evidence_digest = digest_canonical(
            {
                "graphiti_core_release": self.graphiti_core_release,
                "fixtures": [item.as_record() for item in self.fixtures],
                "qualified_routes": [item.as_record() for item in self.qualified_routes],
                "qualified_request_shapes": [
                    item.as_record() for item in self.qualified_request_shapes
                ],
            }
        )
        if self.evidence_digest != expected_evidence_digest:
            raise GraphitiRequestContractError(
                "call-shape evidence digest differs from its fixture derivation"
            )
        if len({item.fixture_id for item in self.fixtures}) != len(self.fixtures):
            raise GraphitiRequestContractError("call-shape fixture identities repeat")
        if {item.leaf_class for item in self.qualified_routes} != set(GraphitiLeafClass):
            raise GraphitiRequestContractError(
                "qualified Graphiti routes must bind every leaf class exactly once"
            )
        request_shape_keys = {
            (
                item.leaf_class,
                item.semantic_request_class,
                item.response_schema_identity,
                item.response_schema_digest,
            )
            for item in self.qualified_request_shapes
        }
        if len(request_shape_keys) != len(self.qualified_request_shapes):
            raise GraphitiRequestContractError(
                "qualified Graphiti request shapes repeat"
            )
        if {item.leaf_class for item in self.qualified_request_shapes} != set(
            GraphitiLeafClass
        ):
            raise GraphitiRequestContractError(
                "qualified Graphiti request shapes must cover every leaf class"
            )

    def route_for(self, leaf_class: GraphitiLeafClass) -> GraphitiQualifiedRoute:
        matches = tuple(
            route for route in self.qualified_routes if route.leaf_class is leaf_class
        )
        if len(matches) != 1:
            raise GraphitiRequestContractError(
                "qualified Graphiti route identity is ambiguous"
            )
        return matches[0]

    def qualify_request(
        self,
        *,
        leaf_class: GraphitiLeafClass,
        semantic_request_class: str,
        response_schema_identity: str,
        response_schema_digest: str,
    ) -> GraphitiQualifiedRequestShape:
        matches = tuple(
            shape
            for shape in self.qualified_request_shapes
            if (
                shape.leaf_class,
                shape.semantic_request_class,
                shape.response_schema_identity,
                shape.response_schema_digest,
            )
            == (
                leaf_class,
                semantic_request_class,
                response_schema_identity,
                response_schema_digest,
            )
        )
        if len(matches) != 1:
            raise GraphitiRequestContractError(
                "Graphiti semantic class/schema is outside the checked call shape"
            )
        return matches[0]

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": GRAPHITI_CALL_SHAPE_SCHEMA_VERSION,
            "canonical_digest": self.canonical_digest,
            "policy_id": self.policy_id,
            "version": self.version,
            "graphiti_core_release": self.graphiti_core_release,
            "framework_identity": self.framework_identity,
            "prompt_identity": self.prompt_identity,
            "ontology_identity": self.ontology_identity,
            "temporal_identity": self.temporal_identity,
            "generation_policy_identity": self.generation_policy_identity,
            "qualified_routes": [item.as_record() for item in self.qualified_routes],
            "qualified_request_shapes": [
                item.as_record() for item in self.qualified_request_shapes
            ],
            "fixtures": [item.as_record() for item in self.fixtures],
            "evidence_digest": self.evidence_digest,
            "maximum_qualified_fixture_count": self.maximum_qualified_fixture_count,
            "headroom": self.headroom,
            "max_distinct_internal_requests": self.max_distinct_internal_requests,
        }


def _raise_fixture_type() -> GraphitiCallShapeFixture:
    raise GraphitiRequestContractError("call-shape fixture must be a mapping")


def _raise_route_type() -> GraphitiQualifiedRoute:
    raise GraphitiRequestContractError("qualified Graphiti route must be a mapping")


def _raise_request_shape_type() -> GraphitiQualifiedRequestShape:
    raise GraphitiRequestContractError(
        "qualified Graphiti request shape must be a mapping"
    )


@dataclass(frozen=True, slots=True)
class GraphitiInternalRequestIdentity:
    effective_revision_digest: str
    ingest_obligation_id: str
    graphiti_attempt_id: str
    provider_attempt_id: str
    internal_ordinal: int
    semantic_request_class: str
    provider: str
    model: str
    reasoning: str
    prompt_bytes: int
    prompt_digest: str
    response_schema_identity: str
    response_schema_digest: str
    requested_max_tokens: int
    framework_identity: str
    prompt_identity: str
    ontology_identity: str
    temporal_identity: str
    generation_policy_identity: str
    context_manifest_digest: str
    leaf_class: GraphitiLeafClass
    retry_state_digest: str
    parent_invocation_id: str | None
    envelope_id: str
    invocation_id: str
    invocation_policy_digest: str
    call_shape_policy_digest: str
    dispatch_authority_digest: str
    dispatch_deadline_at: str | None
    owner_stop_clear: bool
    route_circuit_state: str
    primary_unavailable_event_digest: str | None
    semantic_state_digest: str
    canonical_digest: str

    @classmethod
    def create(cls, **values: object) -> GraphitiInternalRequestIdentity:
        values.pop("semantic_state_digest", None)
        values.pop("canonical_digest", None)
        values.setdefault("primary_unavailable_event_digest", None)
        leaf_class = values.get("leaf_class")
        if not isinstance(leaf_class, GraphitiLeafClass):
            raise GraphitiRequestContractError("Graphiti leaf class must be typed")
        semantic_state_digest = graphiti_semantic_state_digest(
            semantic_request_class=str(values["semantic_request_class"]),
            prompt_digest=str(values["prompt_digest"]),
            response_schema_digest=str(values["response_schema_digest"]),
            requested_max_tokens=values["requested_max_tokens"],  # type: ignore[arg-type]
            leaf_class=leaf_class,
            retry_state_digest=str(values["retry_state_digest"]),
        )
        record = _identity_record(
            values,
            semantic_state_digest=semantic_state_digest,
            canonical_digest="",
        )
        identity = cls(
            **values,  # type: ignore[arg-type]
            semantic_state_digest=semantic_state_digest,
            canonical_digest=digest_canonical(record),
        )
        identity._validate()
        return identity

    def _validate(self) -> None:
        for name in (
            "ingest_obligation_id",
            "graphiti_attempt_id",
            "provider_attempt_id",
            "semantic_request_class",
            "provider",
            "model",
            "reasoning",
            "response_schema_identity",
            "framework_identity",
            "prompt_identity",
            "ontology_identity",
            "temporal_identity",
            "generation_policy_identity",
            "envelope_id",
            "invocation_id",
        ):
            _token(str(getattr(self, name)), field=name)
        for name in (
            "effective_revision_digest",
            "prompt_digest",
            "response_schema_digest",
            "context_manifest_digest",
            "retry_state_digest",
            "invocation_policy_digest",
            "call_shape_policy_digest",
            "dispatch_authority_digest",
            "semantic_state_digest",
            "canonical_digest",
        ):
            _digest(str(getattr(self, name)), field=name)
        _positive(self.internal_ordinal, field="internal ordinal")
        _positive(self.prompt_bytes, field="prompt bytes")
        _positive(self.requested_max_tokens, field="requested max tokens")
        if self.dispatch_deadline_at is not None:
            _token(self.dispatch_deadline_at, field="dispatch deadline")
        if self.owner_stop_clear is not True:
            raise GraphitiRequestContractError("owner stop is not proved clear")
        if self.route_circuit_state != "CLOSED":
            raise GraphitiRequestContractError("route circuit is not proved closed")
        if self.leaf_class is GraphitiLeafClass.FALLBACK:
            if (self.parent_invocation_id is None) == (
                self.primary_unavailable_event_digest is None
            ):
                raise GraphitiRequestContractError(
                    "fallback Graphiti request requires one primary authority"
                )
            if self.primary_unavailable_event_digest is not None:
                _digest(
                    self.primary_unavailable_event_digest,
                    field="primary unavailable event digest",
                )
        elif (
            self.parent_invocation_id is not None
            or self.primary_unavailable_event_digest is not None
        ):
            raise GraphitiRequestContractError(
                "only fallback Graphiti requests may bind a parent invocation"
            )
        if self.semantic_request_class not in ALLOWED_GRAPHITI_SEMANTIC_REQUEST_CLASSES:
            raise GraphitiRequestContractError(
                "semantic request class is outside the qualified Graphiti shape"
            )

    def validate(self) -> None:
        """Revalidate an identity reconstructed outside :meth:`create`."""

        self._validate()

    def as_record(self) -> dict[str, object]:
        return _identity_record(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in {"semantic_state_digest", "canonical_digest"}
            },
            semantic_state_digest=self.semantic_state_digest,
            canonical_digest=self.canonical_digest,
        )


def _identity_record(
    values: Mapping[str, object],
    *,
    semantic_state_digest: str,
    canonical_digest: str,
) -> dict[str, object]:
    leaf_class = values["leaf_class"]
    record = {
        "schema_version": GRAPHITI_INTERNAL_REQUEST_SCHEMA_VERSION,
        "canonical_digest": canonical_digest,
        "effective_revision_digest": values["effective_revision_digest"],
        "ingest_obligation_id": values["ingest_obligation_id"],
        "graphiti_attempt_id": values["graphiti_attempt_id"],
        "provider_attempt_id": values["provider_attempt_id"],
        "internal_ordinal": values["internal_ordinal"],
        "semantic_request_class": values["semantic_request_class"],
        "provider": values["provider"],
        "model": values["model"],
        "reasoning": values["reasoning"],
        "prompt_bytes": values["prompt_bytes"],
        "prompt_digest": values["prompt_digest"],
        "response_schema_identity": values["response_schema_identity"],
        "response_schema_digest": values["response_schema_digest"],
        "requested_max_tokens": values["requested_max_tokens"],
        "framework_identity": values["framework_identity"],
        "prompt_identity": values["prompt_identity"],
        "ontology_identity": values["ontology_identity"],
        "temporal_identity": values["temporal_identity"],
        "generation_policy_identity": values["generation_policy_identity"],
        "context_manifest_digest": values["context_manifest_digest"],
        "leaf_class": (
            leaf_class.value if isinstance(leaf_class, GraphitiLeafClass) else leaf_class
        ),
        "retry_state_digest": values["retry_state_digest"],
        "parent_invocation_id": values.get("parent_invocation_id"),
        "envelope_id": values["envelope_id"],
        "invocation_id": values["invocation_id"],
        "invocation_policy_digest": values["invocation_policy_digest"],
        "call_shape_policy_digest": values["call_shape_policy_digest"],
        "dispatch_authority_digest": values["dispatch_authority_digest"],
        "dispatch_deadline_at": values.get("dispatch_deadline_at"),
        "owner_stop_clear": values["owner_stop_clear"],
        "route_circuit_state": values["route_circuit_state"],
        "semantic_state_digest": semantic_state_digest,
    }
    primary_unavailable_event_digest = values.get(
        "primary_unavailable_event_digest"
    )
    if primary_unavailable_event_digest is not None:
        record["primary_unavailable_event_digest"] = (
            primary_unavailable_event_digest
        )
    return record


def graphiti_semantic_state_digest(
    *,
    semantic_request_class: str,
    prompt_digest: str,
    response_schema_digest: str,
    requested_max_tokens: int,
    leaf_class: GraphitiLeafClass,
    retry_state_digest: str,
) -> str:
    _token(semantic_request_class, field="semantic request class")
    _digest(prompt_digest, field="prompt digest")
    _digest(response_schema_digest, field="response schema digest")
    _positive(requested_max_tokens, field="requested max tokens")
    _digest(retry_state_digest, field="retry state digest")
    return digest_canonical(
        {
            "semantic_request_class": semantic_request_class,
            "prompt_digest": prompt_digest,
            "response_schema_digest": response_schema_digest,
            "requested_max_tokens": requested_max_tokens,
            "leaf_class": leaf_class.value,
            "retry_state_digest": retry_state_digest,
        }
    )


def _load_checked_graphiti_call_shape_policy(
    path: Path,
) -> GraphitiCallShapePolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GraphitiRequestContractError("checked call-shape policy is not an object")
    expected_digest = payload.pop("canonical_digest", None)
    payload.pop("schema_version", None)
    raw_fixtures = payload.get("fixtures")
    if not isinstance(raw_fixtures, list):
        raise GraphitiRequestContractError("checked call-shape fixtures are absent")
    payload["fixtures"] = tuple(raw_fixtures)
    raw_routes = payload.get("qualified_routes")
    if not isinstance(raw_routes, list):
        raise GraphitiRequestContractError("checked qualified routes are absent")
    payload["qualified_routes"] = tuple(raw_routes)
    raw_request_shapes = payload.get("qualified_request_shapes")
    if not isinstance(raw_request_shapes, list):
        raise GraphitiRequestContractError(
            "checked qualified request shapes are absent"
        )
    payload["qualified_request_shapes"] = tuple(raw_request_shapes)
    policy = GraphitiCallShapePolicy.create(**payload)
    if expected_digest is not None and expected_digest != policy.canonical_digest:
        raise GraphitiRequestContractError("checked call-shape policy digest changed")
    return policy


def load_checked_graphiti_call_shape_policy() -> GraphitiCallShapePolicy:
    return _load_checked_graphiti_call_shape_policy(_POLICY_PATH)


def load_checked_native_graphiti_call_shape_policy() -> GraphitiCallShapePolicy:
    return _load_checked_graphiti_call_shape_policy(_NATIVE_POLICY_PATH)


__all__ = [
    "ALLOWED_GRAPHITI_SEMANTIC_REQUEST_CLASSES",
    "GRAPHITI_CALL_SHAPE_SCHEMA_VERSION",
    "GRAPHITI_INTERNAL_REQUEST_SCHEMA_VERSION",
    "GraphitiCallShapeFixture",
    "GraphitiCallShapePolicy",
    "GraphitiInternalRequestIdentity",
    "GraphitiLeafClass",
    "GraphitiQualifiedRequestShape",
    "GraphitiQualifiedRoute",
    "GraphitiRequestContractError",
    "graphiti_semantic_state_digest",
    "load_checked_graphiti_call_shape_policy",
    "load_checked_native_graphiti_call_shape_policy",
]
