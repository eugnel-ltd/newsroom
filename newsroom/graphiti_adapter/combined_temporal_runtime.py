"""Async EVALUATION runtime for combined-temporal extraction."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from math import isfinite, sqrt
from types import SimpleNamespace
from typing import Any, Protocol
from uuid import UUID

from newsroom.authority.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    digest_canonical,
    validate_sha256_digest,
)
from newsroom.authority.types import UtcTimestamp
from newsroom.graphiti_adapter.combined_temporal_contract import (
    CONTRACT_NAME,
    SCHEMA,
    SourceRevisionInput,
    _candidate_prompt_digest,
    build_compact_prompt,
)
from newsroom.graphiti_adapter.evaluation_packet import GRAPHITI_CORE_RELEASE
from newsroom.graphiti_adapter.combined_temporal_extraction import (
    CombinedTemporalLeaf,
    CombinedTemporalOutcome,
    CombinedTemporalTransportResult,
    _leaf,
    _leaf_from_completed,
    _proposal_receipt,
    _retain_artifact,
    _retain_request_identity,
    _validate_and_expand,
)
from newsroom.graphiti_adapter.combined_temporal_pipeline import (
    CombinedTemporalPipelineError,
    ExistingGraphitiPipeline,
)
from newsroom.graphiti_adapter.combined_temporal_response import raw_digest
from newsroom.graphiti_adapter.combined_temporal_types import (
    CombinedTemporalError,
    CombinedTemporalFailureCode,
)
from newsroom.graphiti_adapter.identity import configuration_digest
from newsroom.graphiti_adapter.deterministic_sidecar import (
    DeterministicSidecarInput,
    RelationTriple,
    SemanticRelationProposal,
    collapse_sidecar_duplicates,
    project_deterministic_sidecar,
)
from newsroom.graphiti_adapter.deterministic_summary import (
    AdmittedSummaryAssertion,
    build_deterministic_summary,
)
from newsroom.graphiti_adapter.donor_store import DonorStore
from newsroom.graphiti_adapter.temporal_vocabulary import TEMPORAL_POLICY_VERSION
from newsroom.graphiti_adapter.local_entity_resolution import (
    CanonicalEntityCandidate,
    EntityMentionInput,
    LocalEntityResolutionBasis,
    LocalEntityResolutionOutcome,
    resolve_entity_locally,
)

_EMBEDDING_RETRY_BASES = frozenset(
    {LocalEntityResolutionBasis.BELOW_NEW_CANONICAL_ENTITY_CEILING}
)

_VERIFIER_FAILURE_REASONS = frozenset({
    "GRAPHITI_JUDGMENT_CALLER_BINDING_INVALID", "GRAPHITI_PROPOSAL_BINDING_INVALID",
    "GRAPHITI_JUDGMENT_INPUT_BOUND", "GRAPHITI_JUDGMENT_ANSWER_INVENTORY_INVALID",
    "GRAPHITI_JUDGMENT_ANSWER_INVALID", "GRAPHITI_RELATION_DIRECTION_UNPROVEN",
    "GRAPHITI_PROPOSAL_UNSUPPORTED",
})


def _observe_runtime_failure(stage: str, error: Exception, identity_digest: object) -> None:
    """Optional scalar code-site evidence, never provider text or work authority."""
    try:
        from newsroom.control_plane.diagnostic_logging import emit_diagnostic

        data = {"stage": stage}
        # One explicit wrapped cause only; never traverse context or a cause chain.
        for prefix, failure in (("", error), ("cause_", error.__cause__)):
            if failure is None:
                continue
            data[prefix + "exception_class"] = type(failure).__name__[:128]
            point = failure.__traceback__
            while point is not None and point.tb_next is not None:
                point = point.tb_next
            if point is not None:
                code = point.tb_frame.f_code
                data.update({
                    prefix + "file": code.co_filename.replace("\\", "/").rsplit("/", 1)[-1][:128],
                    prefix + "function": code.co_name[:128], prefix + "line": point.tb_lineno,
                })
            point = None
        if identity_digest is not None:
            validate_sha256_digest(identity_digest)
            data["request_identity_digest"] = identity_digest
        emit_diagnostic("graphiti_combined_temporal_failure", data)
    except Exception:
        pass


def _verification_failure_evidence(error: Exception) -> dict[str, object] | None:
    """Retain only the fixed error carrier, never exception/provider text."""
    reason = getattr(error, "reason_code", None)
    reference = getattr(error, "reference", None)
    retained = None
    if reference is not None:
        try:
            invocation_id = reference.invocation_id
            validate_sha256_digest(invocation_id)
            raw_id, receipt_id = str(reference.raw_admission_id), str(reference.receipt_admission_id)
            if str(UUID(raw_id)) != raw_id or str(UUID(receipt_id)) != receipt_id:
                raise ValueError("noncanonical admission identity")
            retained = {"invocation_id": invocation_id, "raw_admission_id": raw_id,
                        "receipt_admission_id": receipt_id}
        except (AttributeError, TypeError, ValueError):
            pass
    if type(reason) is str and reason in _VERIFIER_FAILURE_REASONS:
        return {"reason_code": reason, "judgment_reference": retained}
    if retained is not None:
        return {"reason_code": "GRAPHITI_VERIFIER_FAILED", "judgment_reference": retained}
    return None


class AsyncCombinedTemporalTransport(Protocol):
    async def generate_response(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        response_model: str,
        max_tokens: int,
    ) -> CombinedTemporalTransportResult: ...


class NewsroomCombinedTemporalExtractionV1:
    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        return SCHEMA


class CliCombinedTemporalTransport:
    """Forward the compact object through #769's observed CLI client."""

    def __init__(self, llm_client: Any) -> None:
        self._llm_client = llm_client

    async def generate_response(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        response_model: str,
        max_tokens: int,
    ) -> CombinedTemporalTransportResult:
        if response_model != CONTRACT_NAME or dict(schema) != SCHEMA:
            raise ValueError("combined-temporal runtime request identity differs")
        raw = await self._llm_client._generate_response(
            [SimpleNamespace(role="user", content=prompt)],
            response_model=NewsroomCombinedTemporalExtractionV1,
            max_tokens=max_tokens,
        )
        invocations = list(getattr(self._llm_client, "invocations", ()))
        latest = invocations[-1] if invocations else {}
        usage = latest.get("usage")
        return CombinedTemporalTransportResult(
            raw=raw,
            framework_version=GRAPHITI_CORE_RELEASE,
            model_version=(
                str(latest["model"]) if latest.get("model") is not None else None
            ),
            token_usage=(
                dict(usage) if isinstance(usage, Mapping) else {"basis": "UNMEASURED"}
            ),
            provider_cost=(
                usage.get("cost_usd_microunits")
                if isinstance(usage, Mapping)
                else None
            ),
        )


def _cosine_ppm(left: list[float], right: object) -> int | None:
    if not isinstance(right, (list, tuple)) or not right:
        return None
    try:
        pairs = tuple(zip(left, right, strict=True))
    except ValueError:
        return None
    denominator = sqrt(sum(value * value for value in left)) * sqrt(
        sum(value * value for value in right)
    )
    if not denominator:
        return None
    cosine = sum(a * b for a, b in pairs) / denominator
    return max(0, min(1_000_000, round(cosine * 1_000_000)))


def resolve_nodes_locally(
    nodes: list[Any],
    existing_nodes: tuple[Any, ...],
    *,
    source_id: str,
    similarities_ppm: Mapping[tuple[str, str], int] | None = None,
) -> tuple[list[Any], dict[str, str], list[tuple[Any, Any]]]:
    """Apply the #748 common-case policy without a chat dedupe leaf."""

    similarities = similarities_ppm or {}
    resolved_nodes: list[Any] = []
    uuid_map: dict[str, str] = {}
    for node in nodes:
        local_id = str(node.uuid)
        entity_type = str(node.attributes["entity_type_id"])
        candidates = tuple(
            CanonicalEntityCandidate(
                canonical_entity_id=str(candidate.uuid),
                canonical_name=str(candidate.name),
                entity_type=str(candidate.attributes.get("entity_type_id", "")),
                governed_aliases=tuple(
                    candidate.attributes.get("governed_aliases", ())
                ),
                governed_identifiers=tuple(
                    candidate.attributes.get("governed_identifiers", ())
                ),
                embedding_similarity_ppm=similarities.get(
                    (local_id, str(candidate.uuid)), 0
                ),
                permitted_source_ids=tuple(
                    candidate.attributes.get(
                        "permitted_source_ids",
                        (candidate.attributes.get("source_id", "UNPERMITTED"),),
                    )
                ),
            )
            for candidate in existing_nodes
        )
        resolution = resolve_entity_locally(
            EntityMentionInput(
                name=str(node.name),
                entity_type=entity_type,
                source_id=source_id,
                governed_identifiers=tuple(
                    node.attributes.get("governed_identifiers", ())
                ),
            ),
            candidates,
        )
        selected = node
        if (
            resolution.outcome
            is LocalEntityResolutionOutcome.DETERMINISTIC_EXISTING_NODE
        ):
            selected = next(
                item
                for item in existing_nodes
                if str(item.uuid) == resolution.selected_canonical_entity_id
            )
            uuid_map[local_id] = str(selected.uuid)
        elif (
            resolution.outcome
            is LocalEntityResolutionOutcome.DETERMINISTIC_NEW_NODE
        ):
            uuid_map[local_id] = local_id
        attributes = dict(getattr(node, "attributes", {}) or {})
        attributes.update(
            {
                "resolution": resolution.outcome.value,
                "resolution_basis": resolution.basis.value,
                "considered_canonical_entity_count": len(resolution.considered_canonical_entity_ids),
                "considered_canonical_entity_digest": digest_canonical(
                    list(resolution.considered_canonical_entity_ids)
                ),
                "canonical_identity": (
                    str(selected.uuid)
                    if resolution.outcome
                    is LocalEntityResolutionOutcome.DETERMINISTIC_EXISTING_NODE
                    else None
                ),
            }
        )
        node.attributes = attributes
        resolved_nodes.append(node)
    return resolved_nodes, uuid_map, []


async def resolve_nodes_with_optional_embeddings(
    nodes: list[Any],
    existing_nodes: tuple[Any, ...],
    *,
    source_id: str,
    embed_name: Callable[[str], Awaitable[list[float]]] | None = None,
    embed_names: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
) -> tuple[list[Any], dict[str, str], list[tuple[Any, Any]]]:
    """Embed mention names only when exact/alias/normalised resolution is insufficient."""

    resolved, uuid_map, extra = resolve_nodes_locally(
        nodes, existing_nodes, source_id=source_id
    )
    if not existing_nodes or (embed_name is None and embed_names is None):
        return resolved, uuid_map, extra
    retry_ids = {
        str(node.uuid)
        for node in resolved
        if (getattr(node, "attributes", {}) or {}).get("resolution_basis")
        in {item.value for item in _EMBEDDING_RETRY_BASES}
    }
    if not retry_ids:
        return resolved, uuid_map, extra
    retry_nodes = [node for node in nodes if str(node.uuid) in retry_ids]
    candidate_dimensions = {
        len(embedding)
        for candidate in existing_nodes
        if isinstance(
            embedding := getattr(candidate, "name_embedding", None),
            (list, tuple),
        )
        and embedding
    }
    if not candidate_dimensions:
        return resolved, uuid_map, extra
    if len(candidate_dimensions) != 1:
        raise ValueError("canonical name embedding dimensions differ")
    expected_dimension = next(iter(candidate_dimensions))
    if embed_names is not None:
        mention_embeddings = await embed_names(
            [str(node.name).replace("\n", " ") for node in retry_nodes]
        )
    else:
        assert embed_name is not None
        mention_embeddings = [
            await embed_name(str(node.name)) for node in retry_nodes
        ]
    if len(mention_embeddings) != len(retry_nodes):
        raise ValueError("mention embedding batch cardinality differs")
    for embedding in mention_embeddings:
        if (
            not isinstance(embedding, list)
            or len(embedding) != expected_dimension
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                for value in embedding
            )
        ):
            raise ValueError("mention embedding vector differs")
    similarities: dict[tuple[str, str], int] = {}
    for node, mention_embedding in zip(
        retry_nodes, mention_embeddings, strict=True
    ):
        local_id = str(node.uuid)
        for candidate in existing_nodes:
            ppm = _cosine_ppm(
                mention_embedding, getattr(candidate, "name_embedding", None)
            )
            if ppm is None:
                continue
            similarities[(local_id, str(candidate.uuid))] = ppm
    if similarities:
        resolved, uuid_map, extra = resolve_nodes_locally(
            nodes,
            existing_nodes,
            source_id=source_id,
            similarities_ppm=similarities,
        )
    embeddings_by_id = {
        str(node.uuid): embedding
        for node, embedding in zip(retry_nodes, mention_embeddings, strict=True)
    }
    for node in resolved:
        attributes = getattr(node, "attributes", {}) or {}
        embedding = embeddings_by_id.get(str(node.uuid))
        if (
            attributes.get("resolution") == "DETERMINISTIC_NEW_NODE"
            and embedding is not None
            and getattr(node, "name_embedding", None) is None
        ):
            node.name_embedding = embedding
    return resolved, uuid_map, extra


async def _complete_failure(
    pipeline: ExistingGraphitiPipeline,
    prompt: Any,
    receipt: Mapping[str, object],
    *,
    failure_code: CombinedTemporalFailureCode,
) -> CombinedTemporalLeaf:
    terminal = {
        **receipt,
        "terminal_outcome": CombinedTemporalOutcome.TERMINAL_ATTEMPT_FAILURE,
        "failure_code": failure_code,
    }
    completed = await pipeline._complete_failure(terminal)
    return _leaf(
        prompt,
        completed,
        outcome=CombinedTemporalOutcome.TERMINAL_ATTEMPT_FAILURE,
        failure_code=failure_code,
        journal_skipped=False,
    )


async def extract_combined_temporal_async(
    revision: SourceRevisionInput,
    *,
    transport: AsyncCombinedTemporalTransport,
    pipeline: ExistingGraphitiPipeline,
    max_tokens: int = 16_384,
    sidecar_input: DeterministicSidecarInput | None = None,
    admitted_summary_assertions: tuple[AdmittedSummaryAssertion, ...] = (),
    attempt_prepared: bool = False,
    donor_store: DonorStore | None = None,
    typed_proposal_verifier: Callable[..., Mapping[str, object] | None] | None = None,
) -> CombinedTemporalLeaf:
    """Run one combined-temporal leaf without crossing event loops."""

    UtcTimestamp.parse(revision.ingested_at)
    prompt = build_compact_prompt(revision)
    prompt_digest = _candidate_prompt_digest(prompt)
    completed = None if attempt_prepared else await pipeline._prepare_attempt()
    if completed is not None:
        return _leaf_from_completed(
            revision=revision,
            prompt=prompt,
            prompt_digest=prompt_digest,
            completed=completed,
        )
    request_identity = _retain_request_identity(
        donor_store=donor_store,
        revision=revision,
        prompt=prompt,
    )
    receipt: dict[str, object] = {
        "prompt_digest": prompt_digest,
        "ingest_id": revision.ingest_id,
        "source_revision_id": revision.revision_id,
        "predecessor_revision_id": revision.predecessor_revision_id,
        "temporal_basis": revision.temporal_basis,
        "configuration_digest": configuration_digest(),
        "temporal_policy_digest": digest_canonical(TEMPORAL_POLICY_VERSION),
        "invocation_count": 1,
        "chat_receipts_include_transport": True,
        "transport_calls": [],
    }
    if request_identity is not None:
        receipt["request_identity_digest"] = request_identity.identity_digest
    try:
        result = await transport.generate_response(
            prompt=prompt.text,
            schema=SCHEMA,
            response_model=CONTRACT_NAME,
            max_tokens=max_tokens,
        )
        raw = result.raw
        raw_digest_value = raw_digest(raw)
        usage = dict(result.token_usage)
        receipt.update(
            {
                "raw_output_digest": raw_digest_value,
                "framework_version": result.framework_version,
                "model_version": result.model_version,
                "token_usage": usage,
                "provider_cost": result.provider_cost,
                "transport_calls": [
                    {
                        "response_model": CONTRACT_NAME,
                        "prompt_bytes": len(prompt.text.encode("utf-8")),
                        "schema_bytes": len(canonical_json_bytes(SCHEMA)),
                        "raw_output_digest": raw_digest_value,
                        "framework_version": result.framework_version,
                        "model_version": result.model_version,
                        "token_usage": usage,
                        "provider_cost": result.provider_cost,
                    }
                ],
            }
        )
    except Exception as exc:
        _observe_runtime_failure("TRANSPORT", exc, receipt.get("request_identity_digest"))
        return await _complete_failure(
            pipeline,
            prompt,
            receipt,
            failure_code=CombinedTemporalFailureCode.PIPELINE_FAILED,
        )
    try:
        normalised, ranges, nodes, edges, projection_receipt = _validate_and_expand(
            revision=revision,
            prompt=prompt,
            raw=raw,
        )
        receipt["projection_receipt"] = projection_receipt
        receipt["proposal_receipt"] = _proposal_receipt(
            revision=revision,
            payload=normalised,
            ranges=ranges,
        )
        if sidecar_input is not None:
            sidecar = project_deterministic_sidecar(sidecar_input)
            node_names = {str(node.uuid): str(node.name) for node in nodes}
            semantic = tuple(
                SemanticRelationProposal(
                    proposal_id=str(edge.uuid),
                    relation=RelationTriple(
                        node_names[str(edge.source_node_uuid)],
                        str(edge.name),
                        node_names[str(edge.target_node_uuid)],
                    ),
                    evidence_segment_ids=tuple(
                        edge.attributes["evidence_segment_ids"]
                    ),
                )
                for edge in edges
            )
            collapse = collapse_sidecar_duplicates(sidecar, semantic)
            collapsed_ids = {
                item.semantic_proposal_id for item in collapse.collapsed_duplicates
            }
            for edge in edges:
                if str(edge.uuid) in collapsed_ids:
                    edge.attributes = {
                        **edge.attributes,
                        "collapsed_sidecar_duplicate": True,
                    }
            receipt["deterministic_sidecar"] = {
                "relation_proposals": [
                    item.canonical_value() for item in sidecar.relation_proposals
                ],
                "authority": sidecar.authority,
                "proposal_only": sidecar.proposal_only,
                "model_leaf_count": sidecar.model_leaf_count,
                "digest": sidecar.digest,
            }
            receipt["sidecar_collapse"] = {
                "semantic_relation_proposals": [
                    item.canonical_value()
                    for item in collapse.semantic_relation_proposals
                ],
                "collapsed_duplicates": [
                    item.canonical_value()
                    for item in collapse.collapsed_duplicates
                ],
                "model_leaf_count": collapse.model_leaf_count,
                "digest": collapse.digest,
            }
        if admitted_summary_assertions:
            summary = build_deterministic_summary(admitted_summary_assertions)
            receipt["deterministic_summary"] = {
                "outcome": summary.outcome.value,
                "summary": summary.summary,
                "assertion_ids": list(summary.assertion_ids),
                "evidence_links": list(summary.evidence_links),
                "temporal_links": list(summary.temporal_links),
                "admission_decisions": [
                    item.canonical_value() for item in summary.admission_decisions
                ],
                "maximum_bytes": summary.maximum_bytes,
                "provider_leaf_count": summary.provider_leaf_count,
                "requires_separate_policy": summary.requires_separate_policy,
                "digest": summary.digest,
            }
    except CombinedTemporalError as exc:
        return await _complete_failure(
            pipeline,
            prompt,
            receipt,
            failure_code=exc.code,
        )
    except CanonicalizationError:
        return await _complete_failure(
            pipeline,
            prompt,
            receipt,
            failure_code=CombinedTemporalFailureCode.MALFORMED_OBJECT,
        )
    if typed_proposal_verifier is not None:
        try:
            verification = typed_proposal_verifier(
                source_revision=revision,
                proposal_receipt=deepcopy(receipt["proposal_receipt"]),
            )
            if verification is not None:
                if not isinstance(verification, Mapping) or not verification:
                    raise ValueError("Graphiti typed verification evidence is absent")
                retained_verification = deepcopy(dict(verification))
                digest_canonical(retained_verification)
                receipt["typed_proposal_verification"] = retained_verification
        except Exception as exc:
            _observe_runtime_failure("TYPED_VERIFIER", exc, receipt.get("request_identity_digest"))
            failure_evidence = _verification_failure_evidence(exc)
            failure_code = (exc.code if isinstance(exc, CombinedTemporalError)
                            else CombinedTemporalFailureCode.PIPELINE_FAILED)
            if failure_evidence is not None:
                receipt["typed_proposal_verification"] = failure_evidence
                if (not isinstance(exc, CombinedTemporalError)
                        and failure_evidence["judgment_reference"] is not None
                        and failure_evidence["reason_code"] in {
                            "GRAPHITI_RELATION_DIRECTION_UNPROVEN", "GRAPHITI_PROPOSAL_UNSUPPORTED",
                        }):
                    failure_code = CombinedTemporalFailureCode.EVIDENCE_UNRESOLVED
            return await _complete_failure(
                pipeline, prompt, receipt,
                failure_code=failure_code,
            )
    try:
        pipeline_result = await pipeline._execute(
            nodes=nodes,
            edges=edges,
            receipt={**receipt, "provider_attempt_number": 1},
        )
    except CombinedTemporalPipelineError as exc:
        if not exc.graph_effect_attempted and not exc.rollback_completed:
            _observe_runtime_failure("PIPELINE", exc, receipt.get("request_identity_digest"))
            return await _complete_failure(
                pipeline,
                prompt,
                receipt,
                failure_code=CombinedTemporalFailureCode.PIPELINE_FAILED,
            )
        raise
    if pipeline_result.completed_receipt is not None:
        receipt = dict(pipeline_result.completed_receipt)
        if request_identity is not None:
            receipt.setdefault(
                "request_identity_digest",
                request_identity.identity_digest,
            )
    outcome = (
        CombinedTemporalOutcome.TERMINAL_SUCCESS_ZERO_PROPOSALS
        if not normalised["facts"]
        else CombinedTemporalOutcome.TERMINAL_SUCCESS_WITH_PROPOSALS
    )
    _retain_artifact(
        donor_store=donor_store,
        request_identity=request_identity,
        payload=normalised,
        prompt=prompt,
        receipt=receipt,
        outcome=outcome,
    )
    return _leaf(
        prompt,
        receipt,
        outcome=outcome,
        payload=normalised,
        payload_digest=digest_canonical(normalised),
        nodes=pipeline_result.nodes,
        edges=pipeline_result.edges,
        guarded_edges=pipeline_result.guarded_edges,
        evidence_ranges=ranges,
        node_resolutions=pipeline_result.node_resolutions,
        embedding_skipped=pipeline_result.embedding_skipped,
        journal_skipped=pipeline_result.journal_skipped,
        rollback_skipped=pipeline_result.rollback_skipped,
        graph_effect_attempted=pipeline_result.graph_effect_attempted,
    )


__all__ = [
    "AsyncCombinedTemporalTransport",
    "CliCombinedTemporalTransport",
    "NewsroomCombinedTemporalExtractionV1",
    "extract_combined_temporal_async",
    "resolve_nodes_locally",
    "resolve_nodes_with_optional_embeddings",
]
