"""Existing Graphiti pipeline adapter for combined-temporal proposals."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from newsroom.authority.canonical import digest_canonical
from newsroom.graphiti_adapter.edge_guard import guard_extracted_edges
from newsroom.graphiti_adapter.neo4j_guard import GuardState, GuardWorkspaceBusy, Neo4jMutationGuard

ResolveNodes = Callable[
    [list[Any]],
    Awaitable[tuple[list[Any], dict[str, str], list[tuple[Any, Any]]]],
]
ResolvePointers = Callable[[list[Any], dict[str, str]], list[Any]]
CreateEmbeddings = Callable[[Any, list[Any]], Awaitable[None]]
PersistGraph = Callable[[list[Any], list[Any]], Awaitable[None]]
CompleteReceipt = Callable[
    [list[Any], list[Any], Mapping[str, object]], Mapping[str, object]
]
CompleteFailureReceipt = Callable[
    [Mapping[str, object]], Mapping[str, object]
]


@dataclass(frozen=True, slots=True)
class CombinedTemporalPipelineResult:
    nodes: tuple[Any, ...]
    edges: tuple[Any, ...]
    guarded_edges: tuple[Any, ...]
    node_resolutions: tuple[str, ...]
    graph_effect_attempted: bool
    embedding_skipped: bool
    journal_skipped: bool
    rollback_skipped: bool
    completed_receipt: Mapping[str, object] | None = None


class CombinedTemporalPipeline(Protocol):
    def prepare_attempt(self) -> Mapping[str, object] | None: ...

    def complete_failure(
        self, receipt: Mapping[str, object]
    ) -> Mapping[str, object]: ...

    def execute(
        self,
        *,
        nodes: tuple[Any, ...],
        edges: tuple[Any, ...],
        receipt: Mapping[str, object],
    ) -> CombinedTemporalPipelineResult: ...


class CombinedTemporalPipelineError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        graph_effect_attempted: bool,
        rollback_completed: bool,
    ) -> None:
        super().__init__(message)
        self.graph_effect_attempted = graph_effect_attempted
        self.rollback_completed = rollback_completed


def _durable_receipt(
    receipt: Mapping[str, object],
    *,
    nodes: list[Any],
    edges: list[Any],
    resolutions: tuple[str, ...],
    chat_invocations: list[dict[str, object]],
    embedding_usage: dict[str, object],
) -> dict[str, object]:
    durable = dict(receipt)
    durable.setdefault("provider_attempt_number", 1)
    durable["pipeline_chat_invocations"] = [dict(item) for item in chat_invocations]
    durable["embedding_usage"] = dict(embedding_usage)
    retained_invocations = int(durable.get("invocation_count", 0))
    durable["invocation_count"] = (
        max(retained_invocations, len(chat_invocations))
        if durable.get("chat_receipts_include_transport") is True
        else retained_invocations + len(chat_invocations)
    )
    proposal_raw = durable.get("proposal_receipt")
    if not isinstance(proposal_raw, Mapping):
        return durable
    # An explicit empty graph effect keeps the original proposal receipt as
    # rejected/held evidence. Do not require mention/edge identity binding.
    if not nodes and not edges:
        return durable
    proposal = dict(proposal_raw)
    mentions_raw = proposal.get("entity_mentions")
    relations_raw = proposal.get("relation_proposals")
    if not isinstance(mentions_raw, list) or len(mentions_raw) != len(nodes):
        raise ValueError("entity mention receipt does not match resolved nodes")
    if not isinstance(relations_raw, list) or len(relations_raw) != len(edges):
        raise ValueError("relation proposal receipt does not match guarded edges")

    mentions: list[dict[str, object]] = []
    for raw, node, resolution in zip(mentions_raw, nodes, resolutions, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("entity mention receipt is malformed")
        item = dict(raw)
        attributes = getattr(node, "attributes", {}) or {}
        # Legacy in-memory callers may still carry the list. New records retain
        # its exact commitment, not an ever-growing graph inventory per mention.
        considered = attributes.get("considered_canonical_entity_ids", ())
        item.update(
            {
                "canonical_identity": (
                    None
                    if resolution == "AMBIGUOUS_HOLD"
                    else str(
                        (getattr(node, "attributes", {}) or {}).get(
                            "canonical_identity"
                        )
                        or node.uuid
                    )
                ),
                "resolution": resolution,
                "entity_resolution_proposal": {
                    "outcome": resolution,
                    "basis": (getattr(node, "attributes", {}) or {}).get(
                        "resolution_basis"
                    ),
                    "considered_canonical_entity_count": attributes.get(
                        "considered_canonical_entity_count", len(considered)
                    ),
                    "considered_canonical_entity_digest": attributes.get(
                        "considered_canonical_entity_digest", digest_canonical(list(considered))
                    ),
                    "provider_leaf_count": 0,
                },
            }
        )
        mentions.append(item)

    relations: list[dict[str, object]] = []
    for raw, edge in zip(relations_raw, edges, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("relation proposal receipt is malformed")
        item = dict(raw)
        item.pop("fact_embedding", None)
        fact_embedding = getattr(edge, "fact_embedding", None)
        item.update(
            {
                "proposal_identity": str(edge.uuid),
                "source_identity": str(edge.source_node_uuid),
                "target_identity": str(edge.target_node_uuid),
                "proposal_status": (
                    "COLLAPSED_EXACT_SIDECAR_DUPLICATE"
                    if (getattr(edge, "attributes", {}) or {}).get(
                        "collapsed_sidecar_duplicate"
                    )
                    else (
                        "AMBIGUOUS_HOLD_ENDPOINT"
                        if fact_embedding is None
                        else "PROPOSED"
                    )
                ),
            }
        )
        relations.append(item)
    proposal["entity_mentions"] = mentions
    proposal["relation_proposals"] = relations
    proposal["node_resolutions"] = list(resolutions)
    durable["proposal_receipt"] = proposal
    return durable


@dataclass(slots=True)
class ExistingGraphitiPipeline:
    """Bind proposals to Graphiti resolution, embedding and durable rollback."""

    guard: Neo4jMutationGuard
    resolve_nodes: ResolveNodes
    resolve_pointers: ResolvePointers
    create_embeddings: CreateEmbeddings
    persist_graph: PersistGraph
    embedder: Any
    run_async: Callable[[Awaitable[Any]], Any]
    chat_receipt: Callable[[], list[dict[str, object]]]
    embedding_receipt: Callable[[], dict[str, object]]
    expected_group_id: str | None = None
    expected_episode_uuid: str | None = None
    expected_ingest_id: str | None = None
    complete_receipt: CompleteReceipt | None = None
    complete_failure_receipt: CompleteFailureReceipt | None = None
    prepare_execution: Callable[[], Awaitable[None]] | None = None
    _attempt_started: bool = field(default=False, init=False)
    recovery_marker: Any | None = field(default=None, init=False)

    def prepare_attempt(self) -> Mapping[str, object] | None:
        return self.run_async(self._prepare_attempt())

    async def _prepare_attempt(self) -> Mapping[str, object] | None:
        try:
            marker = await self.guard.begin()
        except GuardWorkspaceBusy:
            raise
        except Exception as exc:
            raise CombinedTemporalPipelineError(
                "combined-temporal journal could not start",
                graph_effect_attempted=False,
                rollback_completed=False,
            ) from exc
        if marker.state is GuardState.CREATED:
            self._attempt_started = True
            return None
        if marker.state is GuardState.COMPLETE:
            try:
                return await self.guard.completed_raw()
            except Exception as exc:
                raise CombinedTemporalPipelineError(
                    "combined-temporal completed receipt is malformed",
                    graph_effect_attempted=False,
                    rollback_completed=False,
                ) from exc
        rollback_completed = marker.state is GuardState.RECOVERED_AMBIGUOUS
        self.recovery_marker = marker
        if marker.state in {GuardState.PENDING, GuardState.ROLLING_BACK}:
            try:
                rollback_completed = await self.guard.rollback_pending(
                    chat_invocations=[dict(item) for item in marker.chat_invocations],
                    embedding_usage=dict(marker.embedding_usage or {}),
                    reason="RECOVERED_BEFORE_PROVIDER_RETRY",
                )
            except Exception:
                rollback_completed = False
        raise CombinedTemporalPipelineError(
            "combined-temporal journal blocks another provider leaf",
            graph_effect_attempted=True,
            rollback_completed=rollback_completed,
        )

    def complete_failure(
        self, receipt: Mapping[str, object]
    ) -> Mapping[str, object]:
        return self.run_async(self._complete_failure(receipt))

    async def _complete_failure(
        self, receipt: Mapping[str, object]
    ) -> Mapping[str, object]:
        if not self._attempt_started:
            raise CombinedTemporalPipelineError(
                "combined-temporal failed leaf has no pending journal",
                graph_effect_attempted=False,
                rollback_completed=False,
            )
        chat_invocations = self.chat_receipt()
        embedding_usage = self.embedding_receipt()
        durable = _durable_receipt(
            receipt,
            nodes=[],
            edges=[],
            resolutions=(),
            chat_invocations=chat_invocations,
            embedding_usage=embedding_usage,
        )
        completed = (
            durable
            if self.complete_failure_receipt is None
            else dict(self.complete_failure_receipt(durable))
        )
        await self.guard.record_pending_telemetry(
            chat_invocations=chat_invocations,
            embedding_usage=embedding_usage,
        )
        await self.guard.complete(completed)
        self._attempt_started = False
        return completed

    async def _seal_empty_effect(
        self,
        receipt: Mapping[str, object],
        *,
        embedding_skipped: bool,
    ) -> CombinedTemporalPipelineResult:
        """Journal an explicit zero graph effect without mutation."""

        chat_invocations = self.chat_receipt()
        embedding_usage = self.embedding_receipt()
        await self.guard.record_pending_telemetry(
            chat_invocations=chat_invocations,
            embedding_usage=embedding_usage,
        )
        durable_receipt = _durable_receipt(
            receipt,
            nodes=[],
            edges=[],
            resolutions=(),
            chat_invocations=chat_invocations,
            embedding_usage=embedding_usage,
        )
        # Integer 0 counts are also emitted for unmarked result=None receipts.
        # TERMINAL zero-as-success requires this explicit accepted-empty mark.
        durable_receipt["zero_proposal_effect"] = "EXPLICIT"
        if self.complete_receipt is None:
            completed_receipt = durable_receipt
        else:
            sealed = self.complete_receipt([], [], durable_receipt)
            # Prefer the sealed object itself so guard completion and
            # ProducedExtraction share one canonical receipt.
            completed_receipt = (
                sealed if isinstance(sealed, dict) else dict(sealed)
            )
        await self.guard.complete(completed_receipt)
        self._attempt_started = False
        return CombinedTemporalPipelineResult(
            nodes=(),
            edges=(),
            guarded_edges=(),
            node_resolutions=(),
            graph_effect_attempted=False,
            embedding_skipped=embedding_skipped,
            journal_skipped=False,
            rollback_skipped=True,
            completed_receipt=completed_receipt,
        )

    def execute(
        self,
        *,
        nodes: tuple[Any, ...],
        edges: tuple[Any, ...],
        receipt: Mapping[str, object],
    ) -> CombinedTemporalPipelineResult:
        return self.run_async(self._execute(nodes=nodes, edges=edges, receipt=receipt))

    async def _execute(
        self,
        *,
        nodes: tuple[Any, ...],
        edges: tuple[Any, ...],
        receipt: Mapping[str, object],
    ) -> CombinedTemporalPipelineResult:
        self._validate_context(nodes=nodes, edges=edges, receipt=receipt)
        if not self._attempt_started:
            completed = await self._prepare_attempt()
            if completed is not None:
                raise CombinedTemporalPipelineError(
                    "combined-temporal completed result must be replayed "
                    "before execution",
                    graph_effect_attempted=False,
                    rollback_completed=False,
                )
        if self.prepare_execution is None and not nodes and not edges:
            return await self._seal_empty_effect(receipt, embedding_skipped=True)
        mutation_attempted = False
        try:
            if self.prepare_execution is not None:
                mutation_attempted = True
                await self.prepare_execution()
            if not nodes and not edges:
                return await self._seal_empty_effect(
                    receipt, embedding_skipped=True
                )
            resolved_nodes, uuid_map, _duplicates = await self.resolve_nodes(
                list(nodes)
            )
            resolutions = tuple(
                str((getattr(node, "attributes", {}) or {}).get("resolution"))
                if (getattr(node, "attributes", {}) or {}).get("resolution")
                in {
                    "DETERMINISTIC_EXISTING_NODE",
                    "DETERMINISTIC_NEW_NODE",
                    "AMBIGUOUS_HOLD",
                }
                else (
                    "NEW"
                    if str(node.uuid) in {str(item.uuid) for item in nodes}
                    else "RESOLVED_EXISTING"
                )
                for node in resolved_nodes
            )
            persistable_nodes = [
                node
                for node, resolution in zip(
                    resolved_nodes, resolutions, strict=True
                )
                if resolution
                not in {
                    "AMBIGUOUS_HOLD",
                    "DETERMINISTIC_EXISTING_NODE",
                }
            ]
            persistable_edges = [
                edge
                for edge in edges
                if not (getattr(edge, "attributes", {}) or {}).get(
                    "collapsed_sidecar_duplicate"
                )
                and str(edge.source_node_uuid) in uuid_map
                and str(edge.target_node_uuid) in uuid_map
            ]
            if not persistable_edges:
                # Resolution/embeddings may already have run. Held, existing, or
                # leftover NEW nodes without a persistable relation are not a
                # governed graph mutation. Seal explicit zero instead of persisting
                # entity-only state and classifying the empty completion as unmarked.
                return await self._seal_empty_effect(
                    receipt, embedding_skipped=True
                )
            mutation_attempted = True
            guarded, _invalidated, episode_edges = await guard_extracted_edges(
                extracted_edges=persistable_edges,
                uuid_map=uuid_map,
                embedder=self.embedder,
                resolve_pointers=self.resolve_pointers,
                create_embeddings=self.create_embeddings,
            )
            if (
                len(guarded) != len(persistable_edges)
                or len(episode_edges) != len(persistable_edges)
            ):
                raise RuntimeError("edge guard dropped a combined-temporal relation")
            for node, resolution in zip(resolved_nodes, resolutions, strict=True):
                attributes = dict(getattr(node, "attributes", {}) or {})
                attributes["resolution"] = resolution
                node.attributes = attributes
            async with self.guard.fenced_graph_mutation():
                await self.persist_graph(persistable_nodes, guarded)
                await self.guard.restore_preexisting()
            guarded_iterator = iter(guarded)
            persistable_edge_ids = {id(edge) for edge in persistable_edges}
            output_edges = [
                next(guarded_iterator) if id(edge) in persistable_edge_ids else edge
                for edge in edges
            ]
            chat_invocations = self.chat_receipt()
            embedding_usage = self.embedding_receipt()
            durable_receipt = _durable_receipt(
                receipt,
                nodes=resolved_nodes,
                edges=output_edges,
                resolutions=resolutions,
                chat_invocations=chat_invocations,
                embedding_usage=embedding_usage,
            )
            if self.complete_receipt is None:
                completed_receipt = durable_receipt
            else:
                sealed = self.complete_receipt(
                    resolved_nodes, output_edges, durable_receipt
                )
                completed_receipt = (
                    sealed if isinstance(sealed, dict) else dict(sealed)
                )
            await self.guard.record_pending_telemetry(
                chat_invocations=chat_invocations,
                embedding_usage=embedding_usage,
            )
            await self.guard.complete(completed_receipt)
        except Exception as exc:
            rollback_completed = False
            try:
                rollback_completed = await self.guard.rollback_pending(
                    chat_invocations=self.chat_receipt(),
                    embedding_usage=self.embedding_receipt(),
                    reason=type(exc).__name__,
                )
            except Exception:
                rollback_completed = False
            self._attempt_started = False
            raise CombinedTemporalPipelineError(
                "combined-temporal pipeline failed",
                graph_effect_attempted=mutation_attempted,
                rollback_completed=rollback_completed,
            ) from exc

        self._attempt_started = False

        return CombinedTemporalPipelineResult(
            nodes=tuple(resolved_nodes),
            edges=tuple(output_edges),
            guarded_edges=tuple(guarded),
            node_resolutions=resolutions,
            graph_effect_attempted=bool(persistable_nodes or guarded),
            embedding_skipped=not persistable_edges,
            journal_skipped=False,
            rollback_skipped=True,
            completed_receipt=completed_receipt,
        )

    def _validate_context(
        self,
        *,
        nodes: tuple[Any, ...],
        edges: tuple[Any, ...],
        receipt: Mapping[str, object],
    ) -> None:
        if self.expected_group_id is None:
            return
        proposal = receipt.get("proposal_receipt")
        if (
            receipt.get("ingest_id") != self.expected_ingest_id
            or not isinstance(proposal, Mapping)
            or proposal.get("episode_id") != self.expected_episode_uuid
            or any(str(node.group_id) != self.expected_group_id for node in nodes)
            or any(str(edge.group_id) != self.expected_group_id for edge in edges)
            or any(
                list(edge.episodes) != [self.expected_episode_uuid]
                for edge in edges
            )
        ):
            raise CombinedTemporalPipelineError(
                "combined-temporal pipeline identity differs",
                graph_effect_attempted=False,
                rollback_completed=False,
            )


__all__ = [
    "CombinedTemporalPipeline",
    "CombinedTemporalPipelineError",
    "CombinedTemporalPipelineResult",
    "ExistingGraphitiPipeline",
]
