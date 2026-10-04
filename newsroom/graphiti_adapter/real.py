"""EVALUATION Graphiti workspace executor.

CLI transport and graphiti-core result mapping live in focused sibling modules.
This module owns only optional runtime loading, deterministic episode execution
and disposable local workspace orchestration.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, ContextManager

from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.authority.types import UtcTimestamp
from newsroom.control_plane.broker import (
    NEO4J_BOLT_HOST,
    NEO4J_BOLT_PORT,
    BrokerError,
    neo4j_community_password,
    openrouter_api_key,
)
from newsroom.control_plane.veto import OperatorDrainRequested, VetoError
from newsroom.extraction.models import ProducedExtraction, ProposalDraft
from newsroom.extraction.types import (
    ExtractionContractError,
    ExtractionFailureCode,
    ExtractionOutcome,
    ExtractionOutputValidation,
    ExtractionUsage,
)
from newsroom.graphiti_adapter.cli_client import build_cli_llm_client
from newsroom.graphiti_adapter.cli_process import (
    timeout_deadline_after,
    timeout_diagnostic,
)
from newsroom.graphiti_adapter.combined_temporal_pipeline import (
    CombinedTemporalPipelineError,
    ExistingGraphitiPipeline,
)
from newsroom.graphiti_adapter.combined_temporal_contract import SourceRevisionInput
from newsroom.graphiti_adapter.combined_temporal_extraction import (
    CombinedTemporalFailureCode,
    CombinedTemporalOutcome,
)
from newsroom.graphiti_adapter.combined_temporal_runtime import (
    CliCombinedTemporalTransport,
    extract_combined_temporal_async,
    resolve_nodes_with_optional_embeddings,
)
from newsroom.graphiti_adapter.deterministic_sidecar import DeterministicSidecarInput
from newsroom.graphiti_adapter.deterministic_summary import AdmittedSummaryAssertion
from newsroom.graphiti_adapter.contracts import GRAPHITI_PROMPT_COMPONENT
from newsroom.graphiti_adapter.donor_store import DonorStore, SqliteDonorStore
from newsroom.graphiti_adapter.embedding_meter import MeteredOpenAIEmbedder
from newsroom.graphiti_adapter.usage_meter import (
    is_exact_predispatch_no_provider_call,
    summarise_graphiti_usage,
)
from newsroom.graphiti_adapter.edge_guard import guard_extracted_edges
from newsroom.graphiti_adapter.evaluation_packet import (
    GRAPHITI_CHAT_FALLBACK,
    GRAPHITI_CHAT_MODEL,
    GRAPHITI_CLEANUP_TIMEOUT_MS,
    GRAPHITI_CORE_RELEASE,
    GRAPHITI_EMBEDDING_MODEL,
    GRAPHITI_GENERATION_ID,
    GRAPHITI_WORKSPACE_GROUP,
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBEDDING_SLUG,
)
from newsroom.graphiti_adapter.result_mapping import (
    entity_proposals,
    entity_receipts,
    episode_body,
    episode_uuid,
    is_source_registry_name,
    private_graph,
    produced_extraction,
    relation_proposals,
    relation_receipts,
)
from newsroom.graphiti_adapter.result_snapshot import restore_validated_snapshot
from newsroom.graphiti_adapter.recovery_vocabulary import (
    GraphitiRecoveryClassification,
)
from newsroom.graphiti_adapter.neo4j_guard import (
    _OWNED_RECOVERY_PHASES,
    _owned_recovery_guard_reason,
    GuardError,
    GuardMarker,
    GuardState,
    Neo4jMutationGuard,
)

from .models import (
    GraphitiAdapterConfiguration,
    GraphitiAdapterExecution,
    GraphitiAttemptRequest,
    GraphitiWorkspaceDescriptor,
    RecoveredAmbiguousProgressionProof,
    adapter_outcome_for,
)
from .types import (
    GRAPHITI_CORE_RELEASE_MISMATCH,
    GRAPHITI_EXTRA_REQUIRED,
    GraphitiAdapterContractError,
    GraphitiCleanupReason,
    GraphitiExecutionProfile,
    GraphitiRuntimeMode,
    graphiti_setup_failure_detail,
)
from .workspace import DisposableProposalWorkspace

_GRAPHITI_CORE_VERSION = "0.29.3"
_LOGGER = logging.getLogger(__name__)
_NEO4J_USER = "neo4j"
_GRAPHITI_SCHEMA_BOOTSTRAPPED = False
_REASON_BY_OUTCOME = {
    "COMPLETE": GraphitiCleanupReason.NORMAL,
    "PARTIAL": GraphitiCleanupReason.PARTIAL,
    "TIMEOUT": GraphitiCleanupReason.TIMEOUT,
    "MALFORMED_OUTPUT": GraphitiCleanupReason.MALFORMED_OUTPUT,
    "PROVIDER_REJECTED": GraphitiCleanupReason.PROVIDER_REJECTED,
    "POLICY_BLOCKED": GraphitiCleanupReason.POLICY_BLOCKED,
    "FAILED": GraphitiCleanupReason.FAILED,
    "AMBIGUOUS_EFFECT": GraphitiCleanupReason.AMBIGUOUS_EFFECT,
}

# Compatibility names retained for callers while implementation lives in focused modules.
_is_source_registry_name = is_source_registry_name


async def _bootstrap_graphiti_schema(driver: Any) -> None:
    """Create journal schema once before this process starts attempts."""

    global _GRAPHITI_SCHEMA_BOOTSTRAPPED
    if _GRAPHITI_SCHEMA_BOOTSTRAPPED:
        return
    await Neo4jMutationGuard.bootstrap_schema(driver)
    _GRAPHITI_SCHEMA_BOOTSTRAPPED = True


def _no_embedding_usage() -> dict[str, object]:
    return {
        "requests": [],
        "request_count": 0,
        "embedding_tokens": 0,
        "cost_usd_microunits": 0,
        "usage_basis": "NO_EMBEDDING_CALL",
    }


def _telemetry_proves_predispatch_refusal(telemetry: _EpisodeTelemetry) -> bool:
    embedding = telemetry.embedding_usage
    return (
        bool(telemetry.chat_invocations)
        and all(
            is_exact_predispatch_no_provider_call(item)
            for item in telemetry.chat_invocations
        )
        and embedding.get("usage_basis") == "NO_EMBEDDING_CALL"
        and embedding.get("request_count") == 0
        and embedding.get("embedding_tokens") == 0
        and embedding.get("cost_usd_microunits") == 0
        and embedding.get("requests") == []
    )


def _require_evaluation_authority(
    configuration: object,
) -> GraphitiAdapterConfiguration:
    if not isinstance(configuration, GraphitiAdapterConfiguration):
        raise GraphitiAdapterContractError(
            "real Graphiti adapter requires a typed configuration"
        )
    if configuration.runtime_mode is not GraphitiRuntimeMode.REAL_GRAPHITI:
        raise GraphitiAdapterContractError(
            "real adapter rejects a non-real configuration"
        )
    if configuration.execution_profile is not GraphitiExecutionProfile.EVALUATION:
        raise GraphitiAdapterContractError(
            "real Graphiti adapter is authorised only under EVALUATION"
        )
    configuration.require_execution_authorized()
    authority = configuration.real_runtime_authority
    if (
        authority is None
        or authority.framework_release != GRAPHITI_CORE_RELEASE
        or authority.model_release != GRAPHITI_CHAT_MODEL
        or authority.embedding_release != GRAPHITI_EMBEDDING_MODEL
        or configuration.workspace_policy.namespace_prefix
        != GRAPHITI_WORKSPACE_GROUP
    ):
        raise GraphitiAdapterContractError(
            "real Graphiti adapter requires the EVALUATION CLI packet pins"
        )
    return configuration


@dataclass(slots=True)
class _EpisodeTelemetry:
    chat_invocations: list[dict[str, object]] = field(default_factory=list)
    embedding_usage: dict[str, object] = field(default_factory=_no_embedding_usage)
    predecessor_episode_uuid: str | None = None
    provider_attempt_number: int | None = None
    recovery_classification: GraphitiRecoveryClassification | None = None
    timeout_diagnostics: list[dict[str, object]] = field(default_factory=list)


ResultValidator = Callable[..., dict[str, object]]
SnapshotRestorer = Callable[[dict[str, object], _EpisodeTelemetry], None]
FailureValidator = Callable[
    [dict[str, object], _EpisodeTelemetry], dict[str, object]
]


def _runtime_metered_embedder_type(
    embedder_base: type[Any],
) -> type[MeteredOpenAIEmbedder]:
    class RuntimeMeteredOpenAIEmbedder(MeteredOpenAIEmbedder, embedder_base):
        """Metering wrapper satisfying Graphiti's nominal runtime contract."""

    return RuntimeMeteredOpenAIEmbedder


class AmbiguousEpisodeEffect(RuntimeError):
    """The deterministic episode exists without a completed ingest marker."""


class GraphitiCleanupTimeout(GraphitiAdapterContractError):
    """A bounded clean-up phase expired with causal retained evidence."""

    def __init__(self, message: str, *, evidence: Mapping[str, object]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


def _utc_deadline_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _episode_input_digest(
    *, episode_id: str, name: str, body: str, reference_time: datetime,
) -> str:
    return digest_canonical({
        "episode_uuid": episode_id, "name": name, "body": body,
        "reference_time": reference_time.astimezone(UTC).isoformat(),
        "group_id": GRAPHITI_WORKSPACE_GROUP,
    })


def _open_compensation_driver() -> Any:
    # Construct outside asyncio.run: the pinned driver would otherwise schedule
    # schema/index writes automatically. Recovery never bootstraps a fresh graph.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise GraphitiAdapterContractError("compensation driver needs a synchronous boundary")
    try:
        from graphiti_core.driver.neo4j_driver import Neo4jDriver
    except ImportError as exc:
        raise GraphitiAdapterContractError(
            "graphiti extra is required for owned Graphiti compensation",
            reason_code=GRAPHITI_EXTRA_REQUIRED,
        ) from exc
    if importlib.metadata.version("graphiti-core") != _GRAPHITI_CORE_VERSION:
        raise GraphitiAdapterContractError(
            "owned Graphiti compensation requires graphiti-core 0.29.3",
            reason_code=GRAPHITI_CORE_RELEASE_MISMATCH,
        )
    return Neo4jDriver(
        f"bolt://{NEO4J_BOLT_HOST}:{NEO4J_BOLT_PORT}",
        _NEO4J_USER, neo4j_community_password(),
    )


def _owned_recovery_failure_class(error: BaseException) -> str:
    for kind, name in (
        (TimeoutError, "TimeoutError"), (GuardError, "GuardError"),
        (VetoError, "VetoError"), (OperatorDrainRequested, "OperatorDrainRequested"),
        (BrokerError, "BrokerError"), (GraphitiAdapterContractError, "GraphitiAdapterContractError"),
        (asyncio.CancelledError, "CancelledError"), (RuntimeError, "RuntimeError"),
    ):
        if isinstance(error, kind):
            return name
    return "OTHER_ERROR"


def recover_owned_pending(
    *,
    authenticated_input: Callable[[str, int], ContextManager[str | None]],
    owner_stop_check: Callable[[], None],
    deadline: datetime,
    clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
) -> GuardMarker | None:
    """Compensate only the expired generation owner, without provider setup."""
    owner_stop_check()
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise GraphitiAdapterContractError("compensation deadline needs an explicit offset")
    remaining = (deadline.astimezone(UTC) - clock().astimezone(UTC)).total_seconds()
    if remaining <= 0:
        return None
    started = time.monotonic()
    phase, status, failure_class, reason_code = "OWNERSHIP", "NO_OWNER", "NONE", "NONE"
    selected = False

    def diagnostic() -> None:
        try:
            _LOGGER.warning(
                "owned_graphiti_recovery status=%s phase=%s failure_class=%s reason_code=%s elapsed_ms=%d deadline_ms=%d",
                status, phase, failure_class, reason_code,
                round((time.monotonic() - started) * 1_000), round(remaining * 1_000),
            )
        except Exception:
            # Observation must not replace a verified result or an owner stop.
            pass

    def observe_phase(value: str) -> None:
        nonlocal phase
        phase = value if value in _OWNED_RECOVERY_PHASES else "UNOBSERVED"

    try:
        driver = _open_compensation_driver()
    except Exception as exc:
        phase, status, failure_class = "SETUP", "FAILED", _owned_recovery_failure_class(exc)
        reason_code = "SETUP_ERROR"
        diagnostic()
        raise

    async def recover() -> GuardMarker | None:
        nonlocal phase, status, selected
        identity = await Neo4jMutationGuard.owned_pending_identity(
            driver, group_id=GRAPHITI_WORKSPACE_GROUP,
        )
        owner_stop_check()
        if identity is None:
            return None
        selected, status, phase = True, "REFUSED", "AUTHENTICATION"
        episode_id, marker_id, attempt_number = identity
        with authenticated_input(episode_id, attempt_number) as input_digest:
            if input_digest is None:
                return None
            owner_stop_check()
            guard = Neo4jMutationGuard(
                driver, group_id=GRAPHITI_WORKSPACE_GROUP,
                episode_uuid=episode_id, marker_episode_uuid=marker_id,
                attempt_number=attempt_number, input_digest=input_digest,
            )
            phase = "OWNERSHIP"
            marker = await guard.recover_owned_pending(
                owner_stop_check=owner_stop_check, phase_observer=observe_phase,
            )
            if marker is not None and (
                marker.state is not GuardState.RECOVERED_AMBIGUOUS
                or marker.attempt_number != attempt_number
                or marker.input_digest != input_digest
            ):
                raise GuardError("owned Graphiti compensation returned a different marker")
            if marker is not None:
                status = "RECOVERED"
            return marker

    async def bounded_recovery() -> GuardMarker | None:
        nonlocal phase, status, failure_class, reason_code
        try:
            return await asyncio.wait_for(recover(), timeout=remaining)
        except TimeoutError:
            # Cancellation retains the owned journal; no completed marker is
            # fabricated and a later native tick may resume after lease expiry.
            status, failure_class = "TIMEOUT", "TimeoutError"
            reason_code = "DEADLINE"
            return None
        except BaseException as exc:
            status = "STOPPED" if isinstance(exc, (VetoError, OperatorDrainRequested, asyncio.CancelledError)) else "FAILED"
            failure_class = _owned_recovery_failure_class(exc)
            reason_code = _owned_recovery_guard_reason(exc) if isinstance(exc, GuardError) else "OWNER_STOP" if status == "STOPPED" else "EXCEPTION"
            raise
        finally:
            try:
                await asyncio.wait_for(
                    driver.close(), timeout=GRAPHITI_CLEANUP_TIMEOUT_MS / 1_000,
                )
            except TimeoutError:
                # Closing a connection changes neither retained graph state nor
                # the verified recovery result; preserve an owner stop as well.
                if failure_class == "NONE":
                    phase, failure_class = "CONNECTION_CLEANUP", "TimeoutError"
                    reason_code = "DEADLINE"
                    status += "_CLEANUP_TIMEOUT"
            except BaseException as exc:
                if failure_class == "NONE":
                    phase, failure_class = "CONNECTION_CLEANUP", _owned_recovery_failure_class(exc)
                    reason_code = _owned_recovery_guard_reason(exc) if isinstance(exc, GuardError) else "EXCEPTION"
                    status += "_CLEANUP_FAILED"
                raise
            finally:
                if selected or failure_class != "NONE":
                    diagnostic()

    return asyncio.run(bounded_recovery())


def _load_graphiti() -> SimpleNamespace:
    try:
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.client import CrossEncoderClient
        from graphiti_core.embedder.client import EmbedderClient
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.errors import NodeNotFoundError
        from graphiti_core.edges import EntityEdge, create_entity_edge_embeddings
        from graphiti_core.nodes import EpisodeType, EpisodicNode
        from graphiti_core.nodes import EntityNode
        from graphiti_core.utils.bulk_utils import resolve_edge_pointers
        from graphiti_core.utils.maintenance.edge_operations import extract_edges
        from graphiti_core.utils.maintenance.node_operations import (
            resolve_extracted_nodes,
        )
    except ImportError as exc:
        raise GraphitiAdapterContractError(
            "graphiti extra (graphiti-core 0.29.3) is required for real Graphiti execution",
            reason_code=GRAPHITI_EXTRA_REQUIRED,
        ) from exc
    if importlib.metadata.version("graphiti-core") != _GRAPHITI_CORE_VERSION:
        raise GraphitiAdapterContractError(
            "real Graphiti requires graphiti-core 0.29.3",
            reason_code=GRAPHITI_CORE_RELEASE_MISMATCH,
        )

    class IdentityCrossEncoder(CrossEncoderClient):
        async def rank(
            self, query: str, passages: list[str]
        ) -> list[tuple[str, float]]:
            del query
            return [(item, 0.0) for item in passages]

    RuntimeMeteredOpenAIEmbedder = _runtime_metered_embedder_type(EmbedderClient)

    class GuardedGraphiti(Graphiti):
        """Pinned runtime with automatic edge invalidation disabled."""

        async def _extract_and_resolve_edges(
            self,
            episode: Any,
            extracted_nodes: list[Any],
            previous_episodes: list[Any],
            edge_type_map: dict[tuple[str, str], list[str]],
            group_id: str,
            edge_types: dict[str, type[Any]] | None,
            nodes: list[Any],
            uuid_map: dict[str, str],
            custom_extraction_instructions: str | None = None,
        ) -> tuple[list[Any], list[Any], list[Any]]:
            del nodes
            extracted = await extract_edges(
                self.clients,
                episode,
                extracted_nodes,
                previous_episodes,
                edge_type_map,
                group_id,
                edge_types,
                custom_extraction_instructions,
            )
            return await guard_extracted_edges(
                extracted_edges=extracted,
                uuid_map=uuid_map,
                embedder=self.clients.embedder,
                resolve_pointers=resolve_edge_pointers,
                create_embeddings=create_entity_edge_embeddings,
            )

    return SimpleNamespace(
        Graphiti=GuardedGraphiti,
        OpenAIEmbedder=OpenAIEmbedder,
        OpenAIEmbedderConfig=OpenAIEmbedderConfig,
        MeteredOpenAIEmbedder=RuntimeMeteredOpenAIEmbedder,
        IdentityCrossEncoder=IdentityCrossEncoder,
        EpisodeType=EpisodeType,
        EpisodicNode=EpisodicNode,
        EntityEdge=EntityEdge,
        EntityNode=EntityNode,
        create_entity_edge_embeddings=create_entity_edge_embeddings,
        resolve_edge_pointers=resolve_edge_pointers,
        resolve_extracted_nodes=resolve_extracted_nodes,
        NodeNotFoundError=NodeNotFoundError,
        MutationGuard=Neo4jMutationGuard,
    )


def combined_temporal_pipeline_for(
    *,
    configuration: GraphitiAdapterConfiguration,
    graphiti: Any,
    guard: Neo4jMutationGuard,
    episode: Any,
    previous_episodes: tuple[Any, ...] = (),
    entity_types: dict[str, type[Any]] | None = None,
    source_id: str = "UNPERMITTED",
    expected_ingest_id: str | None = None,
) -> ExistingGraphitiPipeline:
    """Wire combined-temporal proposals to the pinned existing Graphiti pipeline."""

    configuration = _require_evaluation_authority(configuration)
    runtime = _load_graphiti()
    expected_group_id = str(episode.group_id)
    expected_episode_uuid = str(episode.uuid)
    if (
        guard.driver is not graphiti.driver
        or expected_group_id != configuration.workspace_policy.namespace_prefix
        or guard.group_id != expected_group_id
        or guard.episode_uuid != expected_episode_uuid
    ):
        raise GraphitiAdapterContractError(
            "combined-temporal graph, journal and episode identity differ"
        )
    async def resolve_nodes(nodes: list[Any]) -> tuple[
        list[Any], dict[str, str], list[tuple[Any, Any]]
    ]:
        typed_nodes = [
            runtime.EntityNode(
                uuid=str(node.uuid),
                name=str(node.name),
                group_id=str(node.group_id),
                labels=list(node.labels),
                created_at=node.created_at,
                summary=str(node.summary),
                attributes=dict(node.attributes),
            )
            for node in nodes
        ]
        existing = tuple(
            sorted(
                await runtime.EntityNode.get_by_group_ids(
                    graphiti.driver,
                    [expected_group_id],
                    with_embeddings=True,
                ),
                key=lambda item: str(item.uuid),
            )
        )
        create_batch = getattr(graphiti.clients.embedder, "create_batch", None)
        create = getattr(graphiti.clients.embedder, "create", None)
        return await resolve_nodes_with_optional_embeddings(
            typed_nodes,
            existing,
            source_id=source_id,
            embed_name=(
                create
                if not callable(create_batch) and callable(create)
                else None
            ),
            embed_names=create_batch if callable(create_batch) else None,
        )

    def resolve_pointers(edges: list[Any], uuid_map: dict[str, str]) -> list[Any]:
        typed_edges = [
            runtime.EntityEdge(
                uuid=str(edge.uuid),
                group_id=str(edge.group_id),
                source_node_uuid=str(edge.source_node_uuid),
                target_node_uuid=str(edge.target_node_uuid),
                created_at=edge.created_at,
                name=str(edge.name),
                fact=str(edge.fact),
                fact_embedding=getattr(edge, "fact_embedding", None),
                episodes=list(edge.episodes),
                expired_at=getattr(edge, "expired_at", None),
                valid_at=edge.valid_at,
                invalid_at=edge.invalid_at,
                reference_time=edge.reference_time,
                attributes=dict(edge.attributes),
            )
            for edge in edges
        ]
        return runtime.resolve_edge_pointers(typed_edges, uuid_map)

    async def persist_graph(nodes: list[Any], edges: list[Any]) -> None:
        await _batch_missing_node_name_embeddings(
            graphiti.clients.embedder,
            nodes,
        )
        await graphiti._process_episode_data(
            episode,
            nodes,
            edges,
            datetime.now(tz=UTC),
            str(episode.group_id),
        )

    def chat_receipt() -> list[dict[str, object]]:
        return [
            dict(item)
            for item in getattr(graphiti.clients.llm_client, "invocations", ())
        ]

    def embedding_receipt() -> dict[str, object]:
        receipt = getattr(graphiti.clients.embedder, "receipt", None)
        return dict(receipt()) if callable(receipt) else _no_embedding_usage()

    return ExistingGraphitiPipeline(
        guard=guard,
        resolve_nodes=resolve_nodes,
        resolve_pointers=resolve_pointers,
        create_embeddings=runtime.create_entity_edge_embeddings,
        persist_graph=persist_graph,
        embedder=graphiti.clients.embedder,
        run_async=asyncio.run,
        chat_receipt=chat_receipt,
        embedding_receipt=embedding_receipt,
        expected_group_id=expected_group_id,
        expected_episode_uuid=expected_episode_uuid,
        expected_ingest_id=expected_ingest_id or guard.input_digest,
    )


async def _batch_missing_node_name_embeddings(
    embedder: Any,
    nodes: list[Any],
) -> None:
    missing = [node for node in nodes if node.name_embedding is None]
    if not missing:
        return
    create_batch = getattr(embedder, "create_batch", None)
    if not callable(create_batch):
        raise GraphitiAdapterContractError(
            "node name embedding batch is unavailable"
        )
    embeddings = await create_batch(
        [str(node.name).replace("\n", " ") for node in missing]
    )
    if len(embeddings) != len(missing):
        raise GraphitiAdapterContractError(
            "node name embedding batch cardinality differs"
        )
    for node, embedding in zip(missing, embeddings, strict=True):
        node.name_embedding = embedding


def _same_time(left: object, right: datetime) -> bool:
    return isinstance(left, datetime) and left.astimezone(UTC) == right.astimezone(UTC)


async def _ensure_episode(
    *,
    graphiti: Any,
    runtime: SimpleNamespace,
    episode_id: str,
    name: str,
    body: str,
    reference_time: datetime,
    create: bool = True,
) -> tuple[Any, str] | None:
    """Create the deterministic episode once, or validate the retained identity."""

    try:
        retained = await runtime.EpisodicNode.get_by_uuid(graphiti.driver, episode_id)
    except runtime.NodeNotFoundError:
        if not create:
            return None
        retained = runtime.EpisodicNode(
            uuid=episode_id,
            name=name,
            group_id=GRAPHITI_WORKSPACE_GROUP,
            labels=[],
            source=runtime.EpisodeType.text,
            source_description=GRAPHITI_WORKSPACE_GROUP,
            content=body,
            created_at=datetime.now(tz=UTC),
            valid_at=reference_time,
        )
        await retained.save(graphiti.driver)
        return retained, "CREATED"
    if (
        retained.name != name
        or retained.group_id != GRAPHITI_WORKSPACE_GROUP
        or retained.content != body
        or retained.source != runtime.EpisodeType.text
        or not _same_time(retained.valid_at, reference_time)
    ):
        raise GraphitiAdapterContractError(
            "deterministic Graphiti episode identity was reused for different input"
        )
    return retained, "RETAINED"


def _restore_marker_telemetry(
    telemetry: _EpisodeTelemetry, marker: GuardMarker
) -> None:
    telemetry.chat_invocations = [dict(item) for item in marker.chat_invocations]
    telemetry.embedding_usage = (
        dict(marker.embedding_usage)
        if marker.embedding_usage is not None
        else {
            "requests": [],
            "request_count": 0,
            "embedding_tokens": 0,
            "cost_usd_microunits": None,
            "usage_basis": "UNRECONCILED_PROVIDER_EFFECT",
        }
    )
    telemetry.provider_attempt_number = marker.attempt_number


async def _record_guard_telemetry(
    *, guard: Neo4jMutationGuard, llm_client: Any, embedder: MeteredOpenAIEmbedder,
    telemetry: _EpisodeTelemetry, attempt_number: int
) -> None:
    telemetry.chat_invocations = list(getattr(llm_client, "invocations", ()))
    telemetry.embedding_usage = embedder.receipt()
    telemetry.provider_attempt_number = attempt_number
    await guard.record_pending_telemetry(
        chat_invocations=telemetry.chat_invocations,
        embedding_usage=telemetry.embedding_usage,
    )


def _source_revision_input(
    attempt: GraphitiAttemptRequest,
    *,
    body: str,
    ingested_at: UtcTimestamp,
) -> SourceRevisionInput:
    if attempt.reference_time is None:
        raise GraphitiAdapterContractError("source reference_time is required")
    reference_time = attempt.reference_time.to_text()
    published_at = (
        reference_time
        if attempt.temporal_basis.value == "SOURCE_PUBLISHED"
        else None
    )
    updated_at = (
        reference_time
        if attempt.temporal_basis.value == "SOURCE_UPDATED"
        else None
    )
    return SourceRevisionInput(
        body=body,
        revision_id=str(attempt.manifest.revision_id),
        source_id=str(attempt.manifest.definition_id),
        item_key=str(attempt.manifest.item_id),
        representation_digest=str(attempt.manifest.representation_id),
        published_at=published_at,
        updated_at=updated_at,
        observed_at=reference_time,
        ingested_at=ingested_at.to_text(),
        chunk_ordinal=int(getattr(attempt, "chunk_ordinal", 1)),
        predecessor_revision_id=attempt.predecessor_episode_uuid,
        group_id=GRAPHITI_WORKSPACE_GROUP,
        episode_uuid=attempt.episode_uuid or str(attempt.attempt_id),
    )


async def _add_episode(
    *,
    api_key: str,
    password: str,
    body: str,
    name: str,
    episode_id: str,
    reference_time: datetime,
    telemetry: _EpisodeTelemetry,
    attempt_number: int,
    validate_result: ResultValidator,
    restore_result: SnapshotRestorer,
    configuration: GraphitiAdapterConfiguration | None = None,
    revision: SourceRevisionInput | None = None,
    max_tokens: int = 16_384,
    validate_failure: FailureValidator | None = None,
    sidecar_input: DeterministicSidecarInput | None = None,
    admitted_summary_assertions: tuple[AdmittedSummaryAssertion, ...] = (),
    invocation_observer: Any | None = None,
    donor_store: DonorStore | None = None,
    fallback_permitted: bool = True,
    retry_snapshot_is_failed: Callable[[dict[str, object]], bool] | None = None,
    recovered_ambiguous_progression: (
        RecoveredAmbiguousProgressionProof | None
    ) = None,
) -> Any:
    os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
    runtime = _load_graphiti()
    client_options: dict[str, object] = {}
    if invocation_observer is not None:
        client_options["invocation_observer"] = invocation_observer
    if not fallback_permitted:
        client_options["fallback_permitted"] = False
    llm_client = build_cli_llm_client(**client_options)
    delegate = runtime.OpenAIEmbedder(
        config=runtime.OpenAIEmbedderConfig(
            api_key=api_key,
            embedding_model=OPENROUTER_EMBEDDING_SLUG,
            base_url=OPENROUTER_BASE_URL,
        )
    )
    embedder = (
        runtime.MeteredOpenAIEmbedder(delegate, donor_store=donor_store)
        if invocation_observer is None
        else runtime.MeteredOpenAIEmbedder(
            delegate,
            invocation_observer=invocation_observer,
            donor_store=donor_store,
        )
    )
    graphiti = runtime.Graphiti(
        f"bolt://{NEO4J_BOLT_HOST}:{NEO4J_BOLT_PORT}",
        _NEO4J_USER,
        password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=runtime.IdentityCrossEncoder(),
    )
    input_digest = _episode_input_digest(
        episode_id=episode_id, name=name, body=body, reference_time=reference_time,
    )
    guard = runtime.MutationGuard(
        graphiti.driver,
        group_id=GRAPHITI_WORKSPACE_GROUP,
        episode_uuid=episode_id,
        attempt_number=attempt_number,
        input_digest=input_digest,
    )
    fresh_retry = False
    cancellation_cleanup_active = False
    failure_completed = False
    try:
        if runtime.MutationGuard is Neo4jMutationGuard:
            await _bootstrap_graphiti_schema(graphiti.driver)
            if attempt_number > 1:
                retry_guard = runtime.MutationGuard(
                    graphiti.driver,
                    group_id=GRAPHITI_WORKSPACE_GROUP,
                    episode_uuid=episode_id,
                    marker_episode_uuid=f"{episode_id}:attempt:{attempt_number}",
                    attempt_number=attempt_number,
                    input_digest=input_digest,
                )
                if await retry_guard.marker_exists():
                    guard = retry_guard
                    fresh_retry = True
                else:
                    prior_guard = guard
                    for prior_attempt in range(attempt_number - 1, 1, -1):
                        candidate = runtime.MutationGuard(
                            graphiti.driver,
                            group_id=GRAPHITI_WORKSPACE_GROUP,
                            episode_uuid=episode_id,
                            marker_episode_uuid=(
                                f"{episode_id}:attempt:{prior_attempt}"
                            ),
                            attempt_number=prior_attempt,
                            input_digest=input_digest,
                        )
                        if await candidate.marker_exists():
                            prior_guard = candidate
                            break
                    guard = prior_guard
                    zero_retry_proof = getattr(
                        invocation_observer,
                        "allows_fresh_zero_dispatch_retry",
                        None,
                    )
                    zero_retry_allowed = (
                        callable(zero_retry_proof)
                        and zero_retry_proof(
                            episode_uuid=episode_id,
                            attempt_number=attempt_number,
                        )
                    )
                    settled_retry_proof = getattr(
                        invocation_observer,
                        "allows_fresh_completed_rollback_retry",
                        None,
                    )
                    prior_raw = await prior_guard.completed_raw_or_none()
                    if prior_raw is not None:
                        try:
                            # The provider attempt is leaf-local; retries bind the
                            # authenticated outer Graphiti attempt identity.
                            prior_attempt_number = int(
                                prior_raw["attempt_number"]
                            )
                        except (KeyError, TypeError, ValueError):
                            prior_attempt_number = 0
                        settled_retry_allowed = (
                            callable(settled_retry_proof)
                            and prior_attempt_number > 0
                            and settled_retry_proof(
                                episode_uuid=episode_id,
                                attempt_number=attempt_number,
                                prior_attempt_number=prior_attempt_number,
                            )
                        )
                        if (
                            (zero_retry_allowed or settled_retry_allowed)
                            and retry_snapshot_is_failed is not None
                            and retry_snapshot_is_failed(prior_raw)
                        ):
                            guard = retry_guard
                            fresh_retry = True
                    else:
                        recovered = (
                            await prior_guard.recovered_ambiguous_marker_or_none()
                        )
                        recovered_proof_allowed = False
                        if (
                            recovered is not None
                            and recovered_ambiguous_progression is not None
                        ):
                            proof = recovered_ambiguous_progression
                            fresh_recovery_proof = getattr(
                                invocation_observer,
                                "allows_fresh_recovered_ambiguous_retry",
                                None,
                            )
                            marker_digest = digest_canonical(
                                {
                                    "state": "RECOVERED_AMBIGUOUS",
                                    "attempt_number": recovered.attempt_number,
                                    "workspace_group": GRAPHITI_WORKSPACE_GROUP,
                                    "episode_uuid": episode_id,
                                    "input_digest": recovered.input_digest,
                                }
                            )
                            recovered_proof_allowed = (
                                proof.ingest_id == episode_id
                                and proof.skipped_attempt_number + 1
                                == attempt_number
                                and proof.marker_attempt_number
                                == recovered.attempt_number
                                and proof.marker_input_digest
                                == recovered.input_digest
                                and proof.recovery_marker_digest == marker_digest
                                and callable(fresh_recovery_proof)
                                and fresh_recovery_proof(
                                    episode_uuid=episode_id,
                                    attempt_number=attempt_number,
                                    prior_attempt_number=recovered.attempt_number,
                                    proof=proof,
                                )
                            )
                        if (
                            recovered is not None
                            and (
                                recovered_proof_allowed
                                or (
                                    recovered_ambiguous_progression is None
                                    and callable(settled_retry_proof)
                                    and settled_retry_proof(
                                        episode_uuid=episode_id,
                                        attempt_number=attempt_number,
                                        prior_attempt_number=recovered.attempt_number,
                                    )
                                )
                            )
                        ):
                            guard = retry_guard
                            fresh_retry = True
        if configuration is None or revision is None:
            raise GraphitiAdapterContractError(
                "combined-temporal runtime requires typed attempt authority"
            )
        episode = runtime.EpisodicNode(
            uuid=episode_id,
            name=name,
            group_id=GRAPHITI_WORKSPACE_GROUP,
            labels=[],
            source=runtime.EpisodeType.text,
            source_description=GRAPHITI_WORKSPACE_GROUP,
            content=body,
            created_at=datetime.now(tz=UTC),
            valid_at=reference_time,
        )
        pipeline = combined_temporal_pipeline_for(
            configuration=configuration,
            graphiti=graphiti,
            guard=guard,
            episode=episode,
            source_id=revision.source_id,
            expected_ingest_id=revision.ingest_id,
        )

        def complete_receipt(
            nodes: list[Any],
            edges: list[Any],
            combined_receipt: Mapping[str, object],
        ) -> Mapping[str, object]:
            telemetry.chat_invocations = list(
                getattr(llm_client, "invocations", ())
            )
            telemetry.embedding_usage = embedder.receipt()
            telemetry.provider_attempt_number = attempt_number
            # Seal once after attaching the combined-temporal/projection receipt so
            # Neo4j guard completion and ProducedExtraction share one raw object.
            return validate_result(
                SimpleNamespace(
                    episode=episode,
                    nodes=tuple(nodes),
                    edges=tuple(edges),
                ),
                telemetry,
                combined_receipt,
            )

        pipeline.complete_receipt = complete_receipt
        if validate_failure is not None:
            pipeline.complete_failure_receipt = lambda receipt: validate_failure(
                dict(receipt), telemetry
            )
        try:
            completed = await pipeline._prepare_attempt()
        except CombinedTemporalPipelineError as exc:
            marker = pipeline.recovery_marker
            if isinstance(marker, GuardMarker):
                _restore_marker_telemetry(telemetry, marker)
                telemetry.recovery_classification = (
                    GraphitiRecoveryClassification.RECOVERED_AMBIGUOUS
                    if marker.state is GuardState.RECOVERED_AMBIGUOUS
                    else GraphitiRecoveryClassification.RECOVERED_PENDING_PROCESS_DEATH
                )
            raise AmbiguousEpisodeEffect(
                "prior Graphiti attempt blocks another provider leaf"
            ) from exc
        if completed is not None:
            restore_result(dict(completed), telemetry)
            return SimpleNamespace(episode=None, nodes=(), edges=())

        async def prepare_execution() -> None:
            async with guard.fenced_graph_mutation():
                created = await _ensure_episode(
                    graphiti=graphiti, runtime=runtime, episode_id=episode_id,
                    name=name, body=body, reference_time=reference_time,
                )
            if created is None or created[1] != "CREATED" and not (
                fresh_retry and created[1] == "RETAINED"
            ):
                raise GraphitiAdapterContractError(
                    "deterministic episode predates its durable mutation marker"
                )

        pipeline.prepare_execution = prepare_execution
        # Keep the owner/identity read before provider work, but defer business
        # creation until typed extraction and pipeline context are validated.
        async with guard.fenced_graph_mutation():
            retained = await _ensure_episode(
                graphiti=graphiti,
                runtime=runtime,
                episode_id=episode_id,
                name=name,
                body=body,
                reference_time=reference_time,
                create=False,
            )
        if retained is not None and not fresh_retry:
            raise GraphitiAdapterContractError(
                "deterministic episode predates its durable mutation marker"
            )
        try:
            leaf = await extract_combined_temporal_async(
                revision,
                transport=CliCombinedTemporalTransport(llm_client),
                pipeline=pipeline,
                max_tokens=max_tokens,
                sidecar_input=sidecar_input,
                admitted_summary_assertions=admitted_summary_assertions,
                attempt_prepared=True,
                donor_store=donor_store,
            )
            if leaf.outcome is CombinedTemporalOutcome.TERMINAL_ATTEMPT_FAILURE:
                failure_completed = leaf.journal_skipped is False
                raise ExtractionContractError(
                    f"combined-temporal leaf failed: {leaf.failure_code.value}"
                )
            result = SimpleNamespace(
                episode=episode,
                nodes=leaf.nodes,
                edges=leaf.edges,
            )
        except asyncio.CancelledError:
            cancellation_cleanup_active = True

            async def cleanup_cancelled_attempt() -> None:
                await _record_guard_telemetry(
                    guard=guard,
                    llm_client=llm_client,
                    embedder=embedder,
                    telemetry=telemetry,
                    attempt_number=attempt_number,
                )
                await guard.rollback_pending(
                    chat_invocations=telemetry.chat_invocations,
                    embedding_usage=telemetry.embedding_usage,
                    reason="CANCELLED_OR_TIMED_OUT",
                )

            cleanup_loop = asyncio.get_running_loop()
            cleanup_started = cleanup_loop.time()
            cleanup_deadline_at = timeout_deadline_after(
                GRAPHITI_CLEANUP_TIMEOUT_MS / 1_000
            )
            try:
                await asyncio.wait_for(
                    cleanup_cancelled_attempt(),
                    timeout=GRAPHITI_CLEANUP_TIMEOUT_MS / 1_000,
                )
            except asyncio.TimeoutError:
                telemetry.timeout_diagnostics.append(
                    timeout_diagnostic(
                        boundary="CLEANUP_DEADLINE",
                        phase="ROLLBACK_CLEANUP",
                        cause="CLEANUP_DEADLINE_EXPIRED",
                        configured_timeout_ms=GRAPHITI_CLEANUP_TIMEOUT_MS,
                        elapsed_ms=round(
                            (cleanup_loop.time() - cleanup_started) * 1_000
                        ),
                        deadline_at=cleanup_deadline_at,
                        last_progress="ROLLBACK_INCOMPLETE",
                        termination="TASK_CANCELLED",
                    )
                )
            raise
        except ExtractionContractError:
            if not failure_completed:
                await guard.rollback_pending(
                    chat_invocations=telemetry.chat_invocations,
                    embedding_usage=telemetry.embedding_usage,
                    reason="OUTPUT_VALIDATION_FAILED",
                )
            raise
        except CombinedTemporalPipelineError as exc:
            rollback_completed = exc.rollback_completed
            if not rollback_completed:
                await _record_guard_telemetry(
                    guard=guard,
                    llm_client=llm_client,
                    embedder=embedder,
                    telemetry=telemetry,
                    attempt_number=attempt_number,
                )
                rollback_completed = await guard.rollback_pending(
                    chat_invocations=telemetry.chat_invocations,
                    embedding_usage=telemetry.embedding_usage,
                    reason=type(exc).__name__,
                )
            if rollback_completed:
                telemetry.recovery_classification = (
                    GraphitiRecoveryClassification.ROLLED_BACK_AMBIGUOUS_EFFECT
                )
            raise AmbiguousEpisodeEffect(
                "Graphiti write failed after provider dispatch and was rolled back"
            ) from exc
        except (GuardError, GraphitiAdapterContractError):
            raise
        except Exception as exc:
            await _record_guard_telemetry(
                guard=guard,
                llm_client=llm_client,
                embedder=embedder,
                telemetry=telemetry,
                attempt_number=attempt_number,
            )
            rollback_completed = await guard.rollback_pending(
                chat_invocations=telemetry.chat_invocations,
                embedding_usage=telemetry.embedding_usage,
                reason=type(exc).__name__,
            )
            if rollback_completed:
                telemetry.recovery_classification = (
                    GraphitiRecoveryClassification.ROLLED_BACK_AMBIGUOUS_EFFECT
                )
            raise AmbiguousEpisodeEffect(
                "Graphiti write failed after provider dispatch and was rolled back"
            ) from exc
        return result
    finally:
        if telemetry.provider_attempt_number is None:
            telemetry.chat_invocations = list(getattr(llm_client, "invocations", ()))
            telemetry.embedding_usage = embedder.receipt()
        close_loop = asyncio.get_running_loop()
        close_started = close_loop.time()
        close_deadline_at = timeout_deadline_after(
            GRAPHITI_CLEANUP_TIMEOUT_MS / 1_000
        )
        try:
            # Graphiti closes its driver, not the HTTP client owned by its embedder.
            close_results = await asyncio.wait_for(
                asyncio.gather(
                    delegate.client.close(), graphiti.close(), return_exceptions=True
                ),
                timeout=GRAPHITI_CLEANUP_TIMEOUT_MS / 1_000,
            )
            for close_result in close_results:
                if isinstance(close_result, BaseException):
                    raise close_result
        except asyncio.TimeoutError:
            evidence = timeout_diagnostic(
                boundary="CLEANUP_DEADLINE",
                phase="CONNECTION_CLEANUP",
                cause="CLEANUP_DEADLINE_EXPIRED",
                configured_timeout_ms=GRAPHITI_CLEANUP_TIMEOUT_MS,
                elapsed_ms=round((close_loop.time() - close_started) * 1_000),
                deadline_at=close_deadline_at,
                last_progress="CONNECTION_CLOSE_INCOMPLETE",
                termination="TASK_CANCELLED",
            )
            telemetry.timeout_diagnostics.append(evidence)
            if not cancellation_cleanup_active:
                raise GraphitiCleanupTimeout(
                    "Graphiti connection cleanup timed out",
                    evidence=evidence,
                ) from None


def _raw_receipt(
    attempt: GraphitiAttemptRequest,
    *,
    started_at: UtcTimestamp,
    telemetry: _EpisodeTelemetry,
    result: Any | None,
    proposals: tuple[ProposalDraft, ...],
) -> dict[str, object]:
    reference = attempt.reference_time
    if reference is None:
        raise GraphitiAdapterContractError("source reference_time is required")
    entities = () if result is None else entity_receipts(result)
    relations = () if result is None else relation_receipts(result)
    actual_episode = attempt.episode_uuid or str(attempt.attempt_id)
    if result is not None:
        returned_episode = episode_uuid(result)
        if returned_episode and returned_episode != actual_episode:
            raise GraphitiAdapterContractError(
                "graphiti-core returned a different deterministic episode identity"
            )
    proposal_values = {item.local_id: item.canonical_value() for item in proposals}
    entity_values = tuple(
        {
            **item,
            "episode_uuid": actual_episode,
            "passage_evidence": proposal_values.get(
                str(item.get("local_id")), {}
            ).get("evidence", []),
        }
        for item in entities
    )
    relation_values = tuple(
        {
            **item,
            "episode_uuid": actual_episode,
            "passage_evidence": proposal_values.get(
                str(item.get("local_id")), {}
            ).get("evidence", []),
            "proposal_status": (
                "PROPOSED"
                if str(item.get("local_id")) in proposal_values
                else "HELD_NO_EXACT_EVIDENCE"
            ),
        }
        for item in relations
    )
    raw: dict[str, object] = {
        "workspace_group": GRAPHITI_WORKSPACE_GROUP,
        "generation_id": attempt.generation_id or GRAPHITI_GENERATION_ID,
        "episode_uuid": actual_episode,
        "attempt_number": attempt.attempt_number,
        "provider_attempt_number": (
            telemetry.provider_attempt_number or attempt.attempt_number
        ),
        "predecessor_episode_uuid": attempt.predecessor_episode_uuid,
        "temporal_basis": attempt.temporal_basis,
        "reference_time": reference.to_text(),
        "ingest_started_at": started_at.to_text(),
        "passages": [item.canonical_value() for item in attempt.manifest.passages],
        "proposals": [item.canonical_value() for item in proposals],
        "entities": list(entity_values),
        "relations": list(relation_values),
        "entity_count": len(entity_values),
        "relation_count": len(relation_values),
        "proposal_count": len(proposals),
        "chat_invocations": list(telemetry.chat_invocations),
        "chat_invocation_count": len(telemetry.chat_invocations),
        "chat_subscription_not_debited": True,
        "embedding_usage": telemetry.embedding_usage,
        "token_usage": summarise_graphiti_usage(
            chat_invocations=telemetry.chat_invocations,
            embedding_usage=telemetry.embedding_usage,
        ),
        "usage_basis": str(
            telemetry.embedding_usage.get("usage_basis", "UNREPORTED")
        ),
        "framework": GRAPHITI_CORE_RELEASE,
        "chat": GRAPHITI_CHAT_MODEL,
        "chat_fallback": GRAPHITI_CHAT_FALLBACK,
        "embedding": GRAPHITI_EMBEDDING_MODEL,
        "prompt_version": GRAPHITI_PROMPT_COMPONENT.component_version,
    }
    if telemetry.recovery_classification is not None:
        raw["recovery_classification"] = telemetry.recovery_classification
    if telemetry.timeout_diagnostics:
        raw["timeout_diagnostics"] = [
            dict(item) for item in telemetry.timeout_diagnostics
        ]
    raw["raw_output_digest"] = digest_bytes(canonical_json_bytes(raw))
    return raw


class RealGraphitiAdapter:
    """Repository-owned real Graphiti adapter for EVALUATION only."""

    __slots__ = (
        "_clock",
        "_execution_deadline",
        "_fallback_permitted",
        "_invocation_observer",
        "_monotonic",
    )

    def __init__(
        self,
        *,
        clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
        monotonic: Callable[[], float] = time.monotonic,
        execution_deadline: datetime | None = None,
        invocation_observer: Any | None = None,
        fallback_permitted: bool = True,
    ) -> None:
        if not isinstance(fallback_permitted, bool):
            raise TypeError("Graphiti fallback permission must be boolean")
        self._clock = clock
        self._monotonic = monotonic
        self._execution_deadline = execution_deadline
        self._fallback_permitted = fallback_permitted
        self._invocation_observer = invocation_observer

    def execute(
        self,
        *,
        attempt: GraphitiAttemptRequest,
        workspace_root: object,
    ) -> GraphitiAdapterExecution:
        if not isinstance(attempt, GraphitiAttemptRequest):
            raise GraphitiAdapterContractError("real adapter needs a typed attempt")
        if not isinstance(workspace_root, Path):
            raise GraphitiAdapterContractError(
                "real adapter workspace root must be a pathlib Path"
            )
        configuration = _require_evaluation_authority(attempt.configuration)

        started_at = self._clock()
        execution_started = self._monotonic()
        remaining_timeout_s = attempt.extraction_request.budget.timeout_ms / 1_000
        if self._execution_deadline is not None:
            if (
                self._execution_deadline.tzinfo is None
                or self._execution_deadline.utcoffset() is None
            ):
                raise GraphitiAdapterContractError(
                    "real adapter execution deadline must have an explicit offset"
                )
            remaining_timeout_s = min(
                remaining_timeout_s,
                max(
                    0.0,
                    (
                        self._execution_deadline.astimezone(UTC) - started_at.value
                    ).total_seconds(),
                ),
            )
        monotonic_deadline = execution_started + remaining_timeout_s
        execution_deadline_at = started_at.value + timedelta(
            seconds=remaining_timeout_s
        )
        workspace = GraphitiWorkspaceDescriptor(
            workspace_id=attempt.workspace_id,
            configuration_id=configuration.configuration_id,
            policy_id=configuration.workspace_policy.policy_id,
            policy_digest=configuration.workspace_policy.canonical_digest,
            namespace=(
                f"{configuration.workspace_policy.namespace_prefix}-"
                f"{str(attempt.workspace_id)}"
            ),
            created_at=started_at,
        )
        private = DisposableProposalWorkspace(
            root=workspace_root,
            descriptor=workspace,
            policy=configuration.workspace_policy,
        )
        private.activate()
        try:
            produced = self._produce(
                attempt,
                started_at,
                execution_deadline=monotonic_deadline,
                execution_started=execution_started,
                execution_deadline_at=execution_deadline_at,
                donor_workspace_root=workspace_root,
            )
            outcome = adapter_outcome_for(produced)
            raw = (
                produced.raw_output_value
                if isinstance(produced.raw_output_value, dict)
                else {}
            )
            relation_values = raw.get("relations", ())
            relations = (
                tuple(relation_values)
                if isinstance(relation_values, (list, tuple))
                else ()
            )
            nodes, private_relations = private_graph(produced, relations)
            private.write_private_graph(nodes=nodes, relations=private_relations)
            ended_at = self._clock()
            cleanup = private.cleanup(
                receipt_id=attempt.cleanup_receipt_id,
                reason=_REASON_BY_OUTCOME[outcome.value],
                recorded_at=ended_at,
            )
        except Exception:
            if private.exists:
                private.cleanup(
                    receipt_id=attempt.cleanup_receipt_id,
                    reason=GraphitiCleanupReason.FAILED,
                    recorded_at=self._clock(),
                )
            raise

        return GraphitiAdapterExecution(
            attempt=attempt,
            outcome=outcome,
            failure_code=produced.failure_code.value,
            produced=produced,
            workspace=workspace,
            cleanup_receipt=cleanup,
            started_at=started_at,
            ended_at=ended_at,
        )

    def _produce(
        self,
        attempt: GraphitiAttemptRequest,
        started_at: UtcTimestamp,
        *,
        execution_deadline: float | None = None,
        execution_started: float | None = None,
        execution_deadline_at: datetime | None = None,
        donor_workspace_root: Path | None = None,
    ) -> ProducedExtraction:
        timeout_s = attempt.extraction_request.budget.timeout_ms / 1000
        if execution_started is None:
            execution_started = self._monotonic()
        if execution_deadline is None:
            execution_deadline = execution_started + timeout_s
        configured_timeout_s = max(0.0, execution_deadline - execution_started)
        if execution_deadline_at is None:
            execution_deadline_at = started_at.value + timedelta(
                seconds=configured_timeout_s
            )
        if attempt.reference_time is None:
            raise GraphitiAdapterContractError(
                "source reference_time is required; started_at must not replace it"
            )
        reference = attempt.reference_time
        deterministic_episode_id = attempt.episode_uuid or str(attempt.attempt_id)
        telemetry = _EpisodeTelemetry(
            predecessor_episode_uuid=attempt.predecessor_episode_uuid
        )
        validated: dict[str, ProducedExtraction] = {}
        validation_phase = "ENTITY_PROPOSALS"
        validation_error: ExtractionContractError | None = None
        validation_usage: ExtractionUsage | None = None
        rollback = "UNOBSERVED"

        def validation_diagnostic(rollback: str) -> None:
            if validation_error is None:
                return
            try:
                message = validation_error.args[0] if len(validation_error.args) == 1 and type(validation_error.args[0]) is str else None
                reason = {
                    "extraction usage exceeds the fixed budget": "BUDGET_EXCEEDED",
                    "extraction usage exceeds the fixed timeout": "TIMEOUT_EXCEEDED",
                    "proposal local identities must be unique": "PROPOSAL_ID_DUPLICATE",
                    "produced proposals must be sorted": "PROPOSAL_ORDER",
                    "produced proposals must be typed": "PROPOSAL_TYPE",
                }.get(message, "CONTRACT_INVALID")
                exceeded = []
                if validation_usage is not None:
                    budget = attempt.extraction_request.budget
                    for field, maximum in (
                        ("elapsed_ms", "timeout_ms"), ("input_bytes", "max_input_bytes"),
                        ("output_bytes", "max_output_bytes"), ("proposal_count", "max_proposals"),
                        ("evidence_range_count", "max_evidence_ranges"),
                        ("request_tokens", "max_request_tokens"),
                        ("response_tokens", "max_response_tokens"),
                        ("cost_microunits", "max_cost_microunits"),
                    ):
                        if getattr(validation_usage, field) > getattr(budget, maximum):
                            exceeded.append(field)
                _LOGGER.warning(
                    "graphiti_validation phase=%s reason=%s exceeded=%s rollback=%s",
                    validation_phase, reason, ",".join(exceeded) or "NONE", rollback,
                )
            except Exception:
                # Observation must not replace the original failure or outcome.
                pass

        def timeout_result(*, phase: str, termination: str) -> ProducedExtraction:
            last_progress = (
                str(telemetry.chat_invocations[-1].get("outcome", "UNOBSERVED"))
                if telemetry.chat_invocations
                else "NO_PROVIDER_INVOCATION"
            )
            telemetry.timeout_diagnostics.append(
                timeout_diagnostic(
                    boundary="EXTRACTION_DEADLINE",
                    phase=phase,
                    cause="EXTRACTION_DEADLINE_EXPIRED",
                    configured_timeout_ms=round(configured_timeout_s * 1_000),
                    elapsed_ms=round(
                        max(0.0, self._monotonic() - execution_started) * 1_000
                    ),
                    deadline_at=_utc_deadline_text(execution_deadline_at),
                    last_progress=last_progress,
                    termination=termination,
                )
            )
            raw = _raw_receipt(
                attempt,
                started_at=started_at,
                telemetry=telemetry,
                result=None,
                proposals=(),
            )
            return produced_extraction(
                attempt,
                outcome=ExtractionOutcome.RETRYABLE_FAILURE,
                failure_code=ExtractionFailureCode.EXECUTION_TIMEOUT,
                validation=None,
                raw=None,
                proposals=(),
                embedding_usage=telemetry.embedding_usage,
                attempt_receipt=raw,
            )

        def validate_result_value(
            result: Any,
            current_telemetry: _EpisodeTelemetry,
            combined_receipt: Mapping[str, object] | None = None,
        ) -> dict[str, object]:
            nonlocal validation_phase, validation_usage
            validation_phase = "ENTITY_PROPOSALS"
            entities = entity_proposals(result, attempt)
            validation_phase = "RELATION_PROPOSALS"
            relations = relation_proposals(result, attempt)
            proposals = tuple(sorted((*entities, *relations), key=lambda item: item.local_id))
            validation_phase = "RAW_RECEIPT"
            raw = _raw_receipt(
                attempt,
                started_at=started_at,
                telemetry=current_telemetry,
                result=result,
                proposals=proposals,
            )
            raw.pop("raw_output_digest", None)
            if combined_receipt is not None:
                raw["combined_temporal_receipt"] = dict(combined_receipt)
            raw["raw_output_digest"] = digest_bytes(canonical_json_bytes(raw))
            validation_phase = "PRODUCED_RESULT"
            produced = produced_extraction(
                attempt,
                outcome=ExtractionOutcome.SUCCESS,
                failure_code=ExtractionFailureCode.NONE,
                validation=ExtractionOutputValidation.VALID,
                raw=raw,
                proposals=proposals,
                embedding_usage=current_telemetry.embedding_usage,
            )
            validation_phase = "BUDGET"
            validation_usage = produced.usage
            try:
                produced.usage.require_within(attempt.extraction_request.budget)
            except ExtractionContractError:
                diagnostic = _raw_receipt(
                    attempt,
                    started_at=started_at,
                    telemetry=current_telemetry,
                    result=None,
                    proposals=(),
                )
                diagnostic.pop("raw_output_digest", None)
                diagnostic["budget_status"] = "EXCEEDED"
                diagnostic["raw_output_digest"] = digest_bytes(
                    canonical_json_bytes(diagnostic)
                )
                validated["produced"] = produced_extraction(
                    attempt,
                    outcome=ExtractionOutcome.INVALID_OUTPUT,
                    failure_code=ExtractionFailureCode.OUTPUT_SCHEMA_INVALID,
                    validation=ExtractionOutputValidation.INVALID,
                    raw=diagnostic,
                    proposals=(),
                    embedding_usage=current_telemetry.embedding_usage,
                )
                raise
            validated["produced"] = produced
            return raw

        def validate_result(
            result: Any, current_telemetry: _EpisodeTelemetry,
            combined_receipt: Mapping[str, object] | None = None,
        ) -> dict[str, object]:
            nonlocal validation_error, validation_usage
            validation_error, validation_usage = None, None
            try:
                return validate_result_value(result, current_telemetry, combined_receipt)
            except ExtractionContractError as error:
                validation_error = error
                raise

        def validate_failure(
            combined_receipt: dict[str, object],
            current_telemetry: _EpisodeTelemetry,
        ) -> dict[str, object]:
            pipeline_calls = combined_receipt.get("pipeline_chat_invocations")
            embedding_usage = combined_receipt.get("embedding_usage")
            current_telemetry.chat_invocations = (
                [dict(item) for item in pipeline_calls]
                if isinstance(pipeline_calls, list)
                else []
            )
            current_telemetry.embedding_usage = (
                dict(embedding_usage)
                if isinstance(embedding_usage, dict)
                else _no_embedding_usage()
            )
            current_telemetry.provider_attempt_number = attempt.attempt_number
            raw = _raw_receipt(
                attempt,
                started_at=started_at,
                telemetry=current_telemetry,
                result=None,
                proposals=(),
            )
            raw.pop("raw_output_digest", None)
            failure_code = str(
                combined_receipt.get("failure_code", "PIPELINE_FAILED")
            )
            raw["combined_temporal_failure_code"] = failure_code
            raw["combined_temporal_receipt"] = combined_receipt
            raw["raw_output_digest"] = digest_bytes(canonical_json_bytes(raw))
            pipeline_failed = (
                failure_code == CombinedTemporalFailureCode.PIPELINE_FAILED.value
            )
            validated["produced"] = produced_extraction(
                attempt,
                outcome=(
                    ExtractionOutcome.RETRYABLE_FAILURE
                    if pipeline_failed
                    else ExtractionOutcome.INVALID_OUTPUT
                ),
                failure_code=(
                    ExtractionFailureCode.PRODUCER_INTERNAL_ERROR
                    if pipeline_failed
                    else ExtractionFailureCode.OUTPUT_SCHEMA_INVALID
                ),
                validation=(
                    None if pipeline_failed else ExtractionOutputValidation.INVALID
                ),
                raw=None if pipeline_failed else raw,
                proposals=(),
                embedding_usage=current_telemetry.embedding_usage,
                attempt_receipt=raw if pipeline_failed else None,
            )
            return raw

        def restore_result(
            raw: dict[str, object], current_telemetry: _EpisodeTelemetry
        ) -> None:
            restoration = restore_validated_snapshot(raw=raw, attempt=attempt)
            current_telemetry.chat_invocations = list(
                restoration.chat_invocations
            )
            current_telemetry.embedding_usage = dict(
                restoration.embedding_usage
            )
            current_telemetry.provider_attempt_number = (
                restoration.provider_attempt_number
            )
            current_telemetry.recovery_classification = (
                restoration.recovery_classification
            )
            validated["produced"] = restoration.produced

        def retry_snapshot_is_failed(raw: dict[str, object]) -> bool:
            restoration = restore_validated_snapshot(raw=raw, attempt=attempt)
            return (
                restoration.produced.outcome
                is ExtractionOutcome.RETRYABLE_FAILURE
                and raw.get("combined_temporal_failure_code")
                == CombinedTemporalFailureCode.PIPELINE_FAILED.value
            )

        inherited_for = getattr(self._invocation_observer, "inherited_empty_for", None)
        inherited = inherited_for(attempt) if callable(inherited_for) else None
        if inherited is not None:
            from .combined_temporal_extraction import inherited_empty_receipt
            revision = _source_revision_input(
                attempt, body=episode_body(attempt), ingested_at=started_at,
            )
            current = inherited_empty_receipt(revision, inherited[0], inherited[1])
            if current is not None:
                raw = _raw_receipt(
                    attempt, started_at=started_at, telemetry=telemetry,
                    result=None, proposals=(),
                )
                raw["episode_uuid"] = attempt.episode_uuid or str(attempt.attempt_id)
                raw["execution_source"] = "INHERITED_NO_PROPOSALS"
                raw["combined_temporal_receipt"] = current
                raw.pop("raw_output_digest", None)
                raw["raw_output_digest"] = digest_bytes(canonical_json_bytes(raw))
                produced = produced_extraction(
                    attempt, outcome=ExtractionOutcome.SUCCESS,
                    failure_code=ExtractionFailureCode.NONE,
                    validation=ExtractionOutputValidation.VALID,
                    raw=raw, proposals=(), embedding_usage=telemetry.embedding_usage,
                )
                produced.usage.require_within(attempt.extraction_request.budget)
                return produced
        try:
            _load_graphiti()
            api_key = openrouter_api_key()
            password = neo4j_community_password()
        except (BrokerError, GraphitiAdapterContractError) as exc:
            raw = _raw_receipt(
                attempt,
                started_at=started_at,
                telemetry=telemetry,
                result=None,
                proposals=(),
            )
            raw.pop("raw_output_digest", None)
            raw["dispatch_state"] = "NOT_DISPATCHED"
            raw["setup_failure"] = type(exc).__name__
            detail = graphiti_setup_failure_detail(exc)
            if detail is not None:
                raw["setup_failure_detail"] = detail
            raw["raw_output_digest"] = digest_bytes(canonical_json_bytes(raw))
            return produced_extraction(
                attempt,
                outcome=ExtractionOutcome.RETRYABLE_FAILURE,
                failure_code=ExtractionFailureCode.PRODUCER_INTERNAL_ERROR,
                validation=None,
                raw=None,
                proposals=(),
                embedding_usage=telemetry.embedding_usage,
                attempt_receipt=raw,
            )
        remaining_timeout_s = execution_deadline - self._monotonic()
        if remaining_timeout_s <= 0:
            return timeout_result(
                phase="PREDISPATCH_SETUP",
                termination="NO_PROVIDER_TASK",
            )
        donor_store = (
            None
            if donor_workspace_root is None
            else SqliteDonorStore(
                donor_workspace_root / "donor_identities.sqlite3"
            )
        )
        try:
            result = asyncio.run(
                asyncio.wait_for(
                    _add_episode(
                        api_key=api_key,
                        password=password,
                        body=episode_body(attempt),
                        name=deterministic_episode_id,
                        episode_id=deterministic_episode_id,
                        reference_time=reference.value,
                        telemetry=telemetry,
                        attempt_number=attempt.attempt_number,
                        validate_result=validate_result,
                        restore_result=restore_result,
                        validate_failure=validate_failure,
                        configuration=attempt.configuration,
                        revision=_source_revision_input(
                            attempt,
                            body=episode_body(attempt),
                            ingested_at=started_at,
                        ),
                        max_tokens=(
                            attempt.extraction_request.budget.max_response_tokens
                        ),
                        invocation_observer=self._invocation_observer,
                        donor_store=donor_store,
                        fallback_permitted=self._fallback_permitted,
                        retry_snapshot_is_failed=retry_snapshot_is_failed,
                        recovered_ambiguous_progression=(
                            attempt.recovered_ambiguous_progression
                        ),
                    ),
                    timeout=remaining_timeout_s,
                )
            )
        except asyncio.TimeoutError:
            return timeout_result(
                phase="EXTRACTION",
                termination="TASK_CANCELLED",
            )
        except GraphitiCleanupTimeout as exc:
            raw = _raw_receipt(
                attempt,
                started_at=started_at,
                telemetry=telemetry,
                result=None,
                proposals=(),
            )
            raw.pop("raw_output_digest", None)
            raw["producer_failure"] = type(exc).__name__
            raw["raw_output_digest"] = digest_bytes(canonical_json_bytes(raw))
            return produced_extraction(
                attempt,
                outcome=ExtractionOutcome.RETRYABLE_FAILURE,
                failure_code=ExtractionFailureCode.PRODUCER_INTERNAL_ERROR,
                validation=None,
                raw=None,
                proposals=(),
                embedding_usage=telemetry.embedding_usage,
                attempt_receipt=raw,
            )
        except (BrokerError, GraphitiAdapterContractError):
            raise
        except ExtractionContractError:
            produced = validated.get("produced")
            if produced is None:
                raise
            return produced
        except AmbiguousEpisodeEffect as error:
            pipeline_error = error.__cause__
            rollback = "UNOBSERVED"
            if isinstance(pipeline_error, CombinedTemporalPipelineError) and pipeline_error.__cause__ is validation_error:
                rollback_completed = pipeline_error.rollback_completed or (
                    telemetry.recovery_classification
                    == GraphitiRecoveryClassification.ROLLED_BACK_AMBIGUOUS_EFFECT
                )
                rollback = "COMPLETE" if rollback_completed else "INCOMPLETE"
            produced = validated.get("produced")
            if (
                produced is not None
                and produced.outcome is ExtractionOutcome.SUCCESS
                and not produced.proposals
            ):
                # The validated empty result proves there is no governed ingest
                # effect to classify. A recovery marker remains evidence when
                # present, but is not a precondition for terminal empty success.
                raw = dict(produced.raw_output_value or {})
                raw.pop("raw_output_digest", None)
                if telemetry.recovery_classification is not None:
                    raw["recovery_classification"] = (
                        telemetry.recovery_classification
                    )
                raw["raw_output_digest"] = digest_bytes(canonical_json_bytes(raw))
                return produced_extraction(
                    attempt,
                    outcome=produced.outcome,
                    failure_code=produced.failure_code,
                    validation=produced.validation,
                    raw=raw,
                    proposals=produced.proposals,
                    embedding_usage=telemetry.embedding_usage,
                )
            raw = _raw_receipt(
                attempt,
                started_at=started_at,
                telemetry=telemetry,
                result=None,
                proposals=(),
            )
            return produced_extraction(
                attempt,
                outcome=ExtractionOutcome.RETRYABLE_FAILURE,
                failure_code=ExtractionFailureCode.AMBIGUOUS_EFFECT,
                validation=None,
                raw=None,
                proposals=(),
                embedding_usage=telemetry.embedding_usage,
                attempt_receipt=raw,
            )
        except Exception as exc:
            raw = _raw_receipt(
                attempt,
                started_at=started_at,
                telemetry=telemetry,
                result=None,
                proposals=(),
            )
            raw.pop("raw_output_digest", None)
            if _telemetry_proves_predispatch_refusal(telemetry):
                raw["dispatch_state"] = "NOT_DISPATCHED"
            raw["producer_failure"] = type(exc).__name__
            raw["raw_output_digest"] = digest_bytes(canonical_json_bytes(raw))
            return produced_extraction(
                attempt,
                outcome=ExtractionOutcome.RETRYABLE_FAILURE,
                failure_code=ExtractionFailureCode.PRODUCER_INTERNAL_ERROR,
                validation=None,
                raw=None,
                proposals=(),
                embedding_usage=telemetry.embedding_usage,
                attempt_receipt=raw,
            )

        finally:
            validation_diagnostic(rollback)

        produced = validated.get("produced")
        if produced is None:
            raise GraphitiAdapterContractError(
                "Graphiti result was not validated before completion"
            )
        return produced


__all__ = ["RealGraphitiAdapter", "combined_temporal_pipeline_for"]
