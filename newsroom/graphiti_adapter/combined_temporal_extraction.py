"""Authority-private Newsroom combined-temporal extraction seam (#747).

Provider-free by default. Does not fork graphiti-core, mutate Neo4j, or
amend GING-010.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from newsroom.authority.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.authority.types import UtcTimestamp
from newsroom.graphiti_adapter.combined_temporal_pipeline import (
    CombinedTemporalPipeline,
    CombinedTemporalPipelineError,
)
from newsroom.graphiti_adapter.combined_temporal_contract import (
    CONTRACT_NAME,
    GROUP_ID,
    SCHEMA,
    SCHEMA_DIGEST,
    CompactPrompt,
    SourceRevisionInput,
    _candidate_prompt_digest,
    build_compact_prompt,
    segment_source,
)
from newsroom.graphiti_adapter.combined_temporal_response import (
    parse_payload,
    raw_digest,
)
from newsroom.graphiti_adapter.combined_temporal_types import (
    CombinedTemporalError,
    CombinedTemporalFailureCode,
    EvidenceSegment,
)
from newsroom.graphiti_adapter.combined_temporal_projection import (
    project_governed_proposals,
)
from newsroom.graphiti_adapter.combined_temporal_validation import (
    GOVERNED_ENTITY_TYPE_IDS,
    normalise,
)
from newsroom.graphiti_adapter.donor_identities import (
    SemanticExtractionRequestIdentityV1,
    build_semantic_request_identity,
    build_validated_artifact,
)
from newsroom.graphiti_adapter.donor_store import DonorStore
from newsroom.graphiti_adapter.evaluation_packet import GRAPHITI_CORE_RELEASE
from newsroom.graphiti_adapter.identity import (
    configuration_digest,
    uuid4_from_digest,
)
from newsroom.graphiti_adapter.temporal_vocabulary import TEMPORAL_POLICY_VERSION

UNMEASURED = "UNMEASURED"
_REPO = Path(__file__).resolve().parents[2]
MEASUREMENTS_PATH = (
    _REPO
    / "docs"
    / "research"
    / "2026-08-22-graphiti-combined-temporal-extraction-measurements.json"
)
LIVE_PACKET_PATH = (
    _REPO
    / "docs"
    / "research"
    / "2026-08-22-graphiti-combined-temporal-extraction-packet.json"
)
CALL_SHAPES_PATH = (
    _REPO
    / "docs"
    / "research"
    / "2026-08-22-graphiti-combined-temporal-call-shapes.json"
)


class CombinedTemporalOutcome(StrEnum):
    TERMINAL_SUCCESS_WITH_PROPOSALS = "TERMINAL_SUCCESS_WITH_PROPOSALS"
    TERMINAL_SUCCESS_ZERO_PROPOSALS = "TERMINAL_SUCCESS_ZERO_PROPOSALS"
    TERMINAL_ATTEMPT_FAILURE = "TERMINAL_ATTEMPT_FAILURE"


@dataclass(frozen=True, slots=True)
class CombinedTemporalTransportResult:
    raw: object
    framework_version: str
    model_version: str | None
    token_usage: Mapping[str, object]
    provider_cost: object | None


@dataclass(frozen=True, slots=True)
class CombinedTemporalLeaf:
    outcome: CombinedTemporalOutcome
    failure_code: CombinedTemporalFailureCode
    prompt: CompactPrompt
    payload: dict[str, Any] | None
    payload_digest: str | None
    nodes: tuple[Any, ...]
    edges: tuple[Any, ...]
    guarded_edges: tuple[Any, ...]
    transport_calls: tuple[dict[str, object], ...]
    graph_effect_attempted: bool
    evidence_ranges: dict[str, tuple[EvidenceSegment, ...]] = field(
        default_factory=dict
    )
    node_resolutions: tuple[str, ...] = ()
    embedding_skipped: bool = True
    journal_skipped: bool = True
    rollback_skipped: bool = True
    raw_output_digest: str | None = None
    framework_version: str = GRAPHITI_CORE_RELEASE
    model_version: str | None = None
    prompt_digest: str | None = None
    invocation_count: int = 1
    token_usage: Mapping[str, object] = field(
        default_factory=lambda: {"basis": UNMEASURED}
    )
    provider_cost: object | None = None
    pipeline_chat_invocations: tuple[dict[str, object], ...] = ()
    embedding_usage: Mapping[str, object] = field(default_factory=dict)
    ingest_id: str | None = None
    temporal_basis: str | None = None
    configuration_digest: str | None = None
    temporal_policy_digest: str | None = None
    deterministic_sidecar: Mapping[str, object] | None = None
    sidecar_collapse: Mapping[str, object] | None = None
    deterministic_summary: Mapping[str, object] | None = None
    request_identity_digest: str | None = None
    donor_artifact_digest: str | None = None


class CombinedTemporalTransport(Protocol):
    def generate_response(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        response_model: str,
    ) -> CombinedTemporalTransportResult: ...


def extract_combined_temporal(
    revision: SourceRevisionInput,
    *,
    transport: CombinedTemporalTransport,
    pipeline: CombinedTemporalPipeline | None = None,
    donor_store: DonorStore | None = None,
) -> CombinedTemporalLeaf:
    _require_pipeline(pipeline)
    ingest_id = revision.ingest_id
    temporal_basis = revision.temporal_basis
    UtcTimestamp.parse(revision.ingested_at)
    prompt = build_compact_prompt(revision)
    prompt_digest = _candidate_prompt_digest(prompt)
    completed = _prepare_attempt(pipeline)
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
        "ingest_id": ingest_id,
        "source_revision_id": revision.revision_id,
        "predecessor_revision_id": revision.predecessor_revision_id,
        "temporal_basis": temporal_basis,
        "configuration_digest": configuration_digest(),
        "temporal_policy_digest": digest_canonical(TEMPORAL_POLICY_VERSION),
        "invocation_count": 1,
        "transport_calls": [],
    }
    if request_identity is not None:
        receipt["request_identity_digest"] = request_identity.identity_digest
    try:
        result = transport.generate_response(
            prompt=prompt.text,
            schema=SCHEMA,
            response_model=CONTRACT_NAME,
        )
        raw = result.raw
        raw_digest_value = raw_digest(raw)
        usage = dict(result.token_usage)
        calls = [
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
        ]
        receipt.update(
            {
                "raw_output_digest": raw_digest_value,
                "framework_version": result.framework_version,
                "model_version": result.model_version,
                "token_usage": usage,
                "provider_cost": result.provider_cost,
                "transport_calls": calls,
            }
        )
    except Exception:
        return _failure_leaf(
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
    except CombinedTemporalError as exc:
        return _failure_leaf(
            pipeline,
            prompt,
            receipt,
            failure_code=exc.code,
        )
    except CanonicalizationError:
        return _failure_leaf(
            pipeline,
            prompt,
            receipt,
            failure_code=CombinedTemporalFailureCode.MALFORMED_OBJECT,
        )
    if pipeline is None and edges:
        return _leaf(
            prompt,
            receipt,
            outcome=CombinedTemporalOutcome.TERMINAL_ATTEMPT_FAILURE,
            failure_code=CombinedTemporalFailureCode.PIPELINE_FAILED,
        )
    if pipeline is None:
        pipeline_result = None
    else:
        try:
            pipeline_result = pipeline.execute(
                nodes=nodes,
                edges=edges,
                receipt={**receipt, "provider_attempt_number": 1},
            )
            if pipeline_result.completed_receipt is not None:
                receipt = dict(pipeline_result.completed_receipt)
                if request_identity is not None:
                    receipt.setdefault(
                        "request_identity_digest",
                        request_identity.identity_digest,
                    )
        except CombinedTemporalPipelineError as exc:
            return _leaf(
                prompt,
                receipt,
                outcome=CombinedTemporalOutcome.TERMINAL_ATTEMPT_FAILURE,
                failure_code=CombinedTemporalFailureCode.PIPELINE_FAILED,
                embedding_skipped=False,
                journal_skipped=False,
                rollback_skipped=not exc.rollback_completed,
                graph_effect_attempted=exc.graph_effect_attempted,
            )
    if pipeline_result is None:
        output_nodes = nodes
        output_edges = edges
        guarded: tuple[Any, ...] = ()
        resolutions: tuple[str, ...] = ()
        graph_effect_attempted = False
        embedding_skipped = True
        journal_skipped = True
        rollback_skipped = True
    else:
        output_nodes = pipeline_result.nodes
        output_edges = pipeline_result.edges
        guarded = pipeline_result.guarded_edges
        resolutions = pipeline_result.node_resolutions
        graph_effect_attempted = pipeline_result.graph_effect_attempted
        embedding_skipped = pipeline_result.embedding_skipped
        journal_skipped = pipeline_result.journal_skipped
        rollback_skipped = pipeline_result.rollback_skipped
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
        nodes=output_nodes,
        edges=output_edges,
        guarded_edges=guarded,
        evidence_ranges=ranges,
        node_resolutions=resolutions,
        embedding_skipped=embedding_skipped,
        journal_skipped=journal_skipped,
        rollback_skipped=rollback_skipped,
        graph_effect_attempted=graph_effect_attempted,
    )


def _retain_request_identity(
    *,
    donor_store: DonorStore | None,
    revision: SourceRevisionInput,
    prompt: CompactPrompt,
) -> SemanticExtractionRequestIdentityV1 | None:
    if donor_store is None:
        return None
    try:
        identity = build_semantic_request_identity(revision, prompt)
    except Exception:
        return None
    try:
        donor_store.count_semantic_opportunity(identity)
    except Exception:
        pass
    try:
        donor_store.retain_extraction_request(identity)
    except Exception:
        return None
    return identity


def _retain_artifact(
    *,
    donor_store: DonorStore | None,
    request_identity: SemanticExtractionRequestIdentityV1 | None,
    payload: Mapping[str, Any],
    prompt: CompactPrompt,
    receipt: dict[str, object],
    outcome: CombinedTemporalOutcome,
) -> None:
    if donor_store is None or request_identity is None:
        return
    try:
        raw_output_digest, framework_version, model_version = (
            _donor_identity_fields(receipt)
        )
        artifact = build_validated_artifact(
            request_identity=request_identity,
            payload=payload,
            payload_digest=digest_canonical(payload),
            raw_output_digest=raw_output_digest,
            prompt=prompt,
            framework_version=framework_version,
            model_version=model_version,
            outcome=outcome.value,
        )
        donor_store.retain_validated_artifact(artifact)
        receipt["donor_artifact_digest"] = artifact.artifact_digest
    except Exception:
        return


def _donor_identity_fields(
    receipt: Mapping[str, object],
) -> tuple[str, str, str | None]:
    nested = receipt.get("combined_temporal_receipt")
    sources: tuple[Mapping[str, object], ...] = (
        (nested, receipt) if isinstance(nested, Mapping) else (receipt,)
    )
    raw_output_digest: object = None
    framework_version: object = None
    model_version: object = None
    for source in sources:
        if raw_output_digest in {None, ""}:
            raw_output_digest = source.get("raw_output_digest")
        if framework_version in {None, ""}:
            framework_version = source.get("framework_version") or source.get(
                "framework"
            )
        if model_version is None:
            model_version = source.get("model_version")
    if not isinstance(raw_output_digest, str) or not raw_output_digest:
        raise KeyError("raw_output_digest")
    if not isinstance(framework_version, str) or not framework_version:
        raise KeyError("framework_version")
    return (
        raw_output_digest,
        framework_version,
        str(model_version) if model_version is not None else None,
    )


def _prepare_attempt(
    pipeline: CombinedTemporalPipeline | None,
) -> Mapping[str, object] | None:
    if pipeline is None:
        return None
    completed = pipeline.prepare_attempt()
    if completed is None:
        return None
    if not isinstance(completed, Mapping):
        raise CombinedTemporalPipelineError(
            "combined-temporal completed receipt is malformed",
            graph_effect_attempted=False,
            rollback_completed=False,
        )
    return completed


def _failure_leaf(
    pipeline: CombinedTemporalPipeline | None,
    prompt: CompactPrompt,
    receipt: Mapping[str, object],
    *,
    failure_code: CombinedTemporalFailureCode,
) -> CombinedTemporalLeaf:
    terminal = {
        **receipt,
        "terminal_outcome": CombinedTemporalOutcome.TERMINAL_ATTEMPT_FAILURE,
        "failure_code": failure_code,
    }
    journal_skipped = True
    if pipeline is not None:
        completed = pipeline.complete_failure(terminal)
        if not isinstance(completed, Mapping):
            raise CombinedTemporalPipelineError(
                "combined-temporal failed receipt is malformed",
                graph_effect_attempted=False,
                rollback_completed=False,
            )
        terminal = dict(completed)
        journal_skipped = False
    return _leaf(
        prompt,
        terminal,
        outcome=CombinedTemporalOutcome.TERMINAL_ATTEMPT_FAILURE,
        failure_code=failure_code,
        journal_skipped=journal_skipped,
    )


def _require_pipeline(pipeline: CombinedTemporalPipeline | None) -> None:
    if pipeline is None:
        return
    missing = [
        name
        for name in ("prepare_attempt", "complete_failure", "execute")
        if not callable(getattr(pipeline, name, None))
    ]
    if missing:
        raise CombinedTemporalPipelineError(
            "combined-temporal pipeline is incomplete: " + ", ".join(missing),
            graph_effect_attempted=False,
            rollback_completed=False,
        )


def inherited_empty_receipt(
    revision: SourceRevisionInput,
    retained: Mapping[str, object],
    source_refs: Mapping[str, str],
) -> dict[str, object] | None:
    """Build current metadata from an authenticated native empty source.

    This pure constructor grants no authority. Its native caller authenticates
    source/output provenance, settled usage and current rights/leaf absence.
    """
    prompt = build_compact_prompt(revision)
    prompt_digest = _candidate_prompt_digest(prompt)
    proposal = retained.get("proposal_receipt")
    if (
        retained.get("prompt_digest") != prompt_digest
        or retained.get("configuration_digest") != configuration_digest()
        or retained.get("zero_proposal_effect") != "EXPLICIT"
        or not isinstance(proposal, Mapping)
        or proposal.get("wire_payload") != {"entities": [], "facts": []}
        or set(source_refs) != {"run_version_id", "output_id", "output_digest", "attempt_id"}
        or any(type(value) is not str or not value for value in source_refs.values())
    ):
        return None
    payload, ranges, nodes, edges, projection = _validate_and_expand(
        revision=revision, prompt=prompt, raw={"entities": [], "facts": []},
    )
    if payload != {"entities": [], "facts": []} or ranges or nodes or edges:
        return None
    return {
        "execution_source": "INHERITED_NO_PROPOSALS",
        "inherited_source": dict(source_refs),
        "prompt_digest": prompt_digest,
        "ingest_id": revision.ingest_id,
        "source_revision_id": revision.revision_id,
        "predecessor_revision_id": revision.predecessor_revision_id,
        "temporal_basis": revision.temporal_basis,
        "configuration_digest": configuration_digest(),
        "temporal_policy_digest": digest_canonical(TEMPORAL_POLICY_VERSION),
        "invocation_count": 0,
        "transport_calls": [],
        "pipeline_chat_invocations": [],
        "proposal_receipt": _proposal_receipt(revision=revision, payload=payload, ranges=ranges),
        "projection_receipt": projection,
        "zero_proposal_effect": "EXPLICIT",
        "graph_effect_attempted": False,
    }


def _proposal_receipt(
    *,
    revision: SourceRevisionInput,
    payload: Mapping[str, Any],
    ranges: Mapping[str, tuple[EvidenceSegment, ...]],
) -> dict[str, object]:
    evidence_passages = [
        {
            "fact": fact,
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "start_byte": segment.start_byte,
                    "end_byte": segment.end_byte,
                    "text": segment.text,
                }
                for segment in segments
            ],
        }
        for fact, segments in sorted(ranges.items())
    ]
    canonical_payload = {
        "entities": [dict(item) for item in payload["entities"]],
        "facts": [dict(item) for item in payload["facts"]],
    }
    return {
        "contract": CONTRACT_NAME,
        "ingest_id": revision.ingest_id,
        "source_id": revision.source_id,
        "source_revision_id": revision.revision_id,
        "predecessor_revision_id": revision.predecessor_revision_id,
        "representation_digest": revision.representation_digest,
        "episode_id": revision.episode_uuid or revision.ingest_id,
        "reference_time": revision.reference_time,
        "temporal_basis": revision.temporal_basis,
        "ingested_at": revision.ingested_at,
        "payload_digest": digest_canonical(canonical_payload),
        "wire_payload": canonical_payload,
        "entity_mentions": [dict(item) for item in payload["entities"]],
        "relation_proposals": [dict(item) for item in payload["facts"]],
        "evidence_passages": evidence_passages,
    }


def _validate_and_expand(
    *,
    revision: SourceRevisionInput,
    prompt: CompactPrompt,
    raw: object,
) -> tuple[
    dict[str, Any],
    dict[str, tuple[EvidenceSegment, ...]],
    tuple[Any, ...],
    tuple[Any, ...],
    dict[str, object],
]:
    payload = parse_payload(raw)
    projected = project_governed_proposals(
        payload,
        prompt.segments,
        UtcTimestamp.parse(revision.reference_time).value,
        raw_provider_digest=raw_digest(raw),
    )
    nodes, edges = _expand(revision, projected.payload)
    return projected.payload, projected.ranges, nodes, edges, projected.receipt


def _leaf_from_completed(
    *,
    revision: SourceRevisionInput,
    prompt: CompactPrompt,
    prompt_digest: str,
    completed: Mapping[str, object],
) -> CombinedTemporalLeaf:
    if (
        completed.get("ingest_id") != revision.ingest_id
        or completed.get("prompt_digest") != prompt_digest
    ):
        raise CombinedTemporalPipelineError(
            "combined-temporal completed receipt identity differs",
            graph_effect_attempted=False,
            rollback_completed=False,
        )
    terminal_outcome = completed.get("terminal_outcome")
    if terminal_outcome is not None:
        try:
            outcome = CombinedTemporalOutcome(str(terminal_outcome))
            failure_code = CombinedTemporalFailureCode(
                str(completed.get("failure_code"))
            )
        except ValueError as exc:
            raise CombinedTemporalPipelineError(
                "combined-temporal completed failure is malformed",
                graph_effect_attempted=False,
                rollback_completed=False,
            ) from exc
        if (
            outcome is not CombinedTemporalOutcome.TERMINAL_ATTEMPT_FAILURE
            or failure_code is CombinedTemporalFailureCode.NONE
        ):
            raise CombinedTemporalPipelineError(
                "combined-temporal completed failure is malformed",
                graph_effect_attempted=False,
                rollback_completed=False,
            )
        return _leaf(
            prompt,
            completed,
            outcome=outcome,
            failure_code=failure_code,
        )
    proposal = completed.get("proposal_receipt")
    if not isinstance(proposal, Mapping):
        raise CombinedTemporalPipelineError(
            "combined-temporal completed receipt has no proposals",
            graph_effect_attempted=False,
            rollback_completed=False,
        )
    if (
        proposal.get("contract") != CONTRACT_NAME
        or proposal.get("ingest_id") != revision.ingest_id
        or proposal.get("source_revision_id") != revision.revision_id
    ):
        raise CombinedTemporalPipelineError(
            "combined-temporal completed receipt identity differs",
            graph_effect_attempted=False,
            rollback_completed=False,
        )
    payload_raw = proposal.get("wire_payload")
    if not isinstance(payload_raw, Mapping):
        raise CombinedTemporalPipelineError(
            "combined-temporal completed payload is malformed",
            graph_effect_attempted=False,
            rollback_completed=False,
        )
    try:
        payload, ranges = normalise(
            parse_payload(payload_raw),
            prompt.segments,
            UtcTimestamp.parse(revision.reference_time).value,
        )
    except (CanonicalizationError, CombinedTemporalError, ValueError) as exc:
        raise CombinedTemporalPipelineError(
            "combined-temporal completed evidence or payload is malformed",
            graph_effect_attempted=False,
            rollback_completed=False,
        ) from exc
    payload_digest = digest_canonical(payload)
    if proposal.get("payload_digest") != payload_digest:
        raise CombinedTemporalPipelineError(
            "combined-temporal completed payload digest differs",
            graph_effect_attempted=False,
            rollback_completed=False,
        )
    expected_passages = _proposal_receipt(
        revision=revision,
        payload=payload,
        ranges=ranges,
    )["evidence_passages"]
    if proposal.get("evidence_passages") != expected_passages:
        raise CombinedTemporalPipelineError(
            "combined-temporal completed evidence differs",
            graph_effect_attempted=False,
            rollback_completed=False,
        )
    mentions = proposal.get("entity_mentions")
    relations = proposal.get("relation_proposals")
    if not isinstance(mentions, list) or not isinstance(relations, list):
        raise CombinedTemporalPipelineError(
            "combined-temporal completed proposals are malformed",
            graph_effect_attempted=False,
            rollback_completed=False,
        )
    nodes, edges = _expand(revision, payload)
    if len(mentions) != len(nodes) or len(relations) != len(edges):
        raise CombinedTemporalPipelineError(
            "combined-temporal completed proposal counts differ",
            graph_effect_attempted=False,
            rollback_completed=False,
        )
    resolutions: list[str] = []
    for mention, node in zip(mentions, nodes, strict=True):
        if not isinstance(mention, Mapping):
            raise CombinedTemporalPipelineError(
                "combined-temporal completed entity mention is malformed",
                graph_effect_attempted=False,
                rollback_completed=False,
            )
        resolution = str(mention["resolution"])
        canonical_identity = mention.get("canonical_identity")
        if resolution != "AMBIGUOUS_HOLD":
            if not isinstance(canonical_identity, str) or not canonical_identity:
                raise CombinedTemporalPipelineError(
                    "combined-temporal completed Canonical Entity identity is malformed",
                    graph_effect_attempted=False,
                    rollback_completed=False,
                )
            node.uuid = canonical_identity
        node.attributes = {**node.attributes, "resolution": resolution}
        resolutions.append(resolution)
    for relation, edge in zip(relations, edges, strict=True):
        if not isinstance(relation, Mapping):
            raise CombinedTemporalPipelineError(
                "combined-temporal completed relation proposal is malformed",
                graph_effect_attempted=False,
                rollback_completed=False,
            )
        edge.uuid = str(relation["proposal_identity"])
        edge.source_node_uuid = str(relation["source_identity"])
        edge.target_node_uuid = str(relation["target_identity"])
        edge.fact_embedding = relation.get("fact_embedding")
    return _leaf(
        prompt,
        completed,
        outcome=(
            CombinedTemporalOutcome.TERMINAL_SUCCESS_ZERO_PROPOSALS
            if not payload["facts"]
            else CombinedTemporalOutcome.TERMINAL_SUCCESS_WITH_PROPOSALS
        ),
        payload=payload,
        payload_digest=payload_digest,
        nodes=nodes,
        edges=edges,
        guarded_edges=edges,
        evidence_ranges=ranges,
        node_resolutions=tuple(resolutions),
        embedding_skipped=True,
        journal_skipped=True,
        rollback_skipped=True,
        graph_effect_attempted=False,
    )


def _leaf(
    prompt: CompactPrompt,
    receipt: Mapping[str, object],
    *,
    outcome: CombinedTemporalOutcome,
    failure_code: CombinedTemporalFailureCode = CombinedTemporalFailureCode.NONE,
    payload: dict[str, Any] | None = None,
    payload_digest: str | None = None,
    nodes: tuple[Any, ...] = (),
    edges: tuple[Any, ...] = (),
    guarded_edges: tuple[Any, ...] = (),
    evidence_ranges: dict[str, tuple[EvidenceSegment, ...]] | None = None,
    node_resolutions: tuple[str, ...] = (),
    embedding_skipped: bool = True,
    journal_skipped: bool = True,
    rollback_skipped: bool = True,
    graph_effect_attempted: bool = False,
) -> CombinedTemporalLeaf:
    calls = receipt.get("transport_calls", ())
    token_usage = receipt.get("token_usage", {"basis": UNMEASURED})
    pipeline_calls = receipt.get("pipeline_chat_invocations", ())
    embedding_usage = receipt.get("embedding_usage", {})
    return CombinedTemporalLeaf(
        outcome=outcome,
        failure_code=failure_code,
        prompt=prompt,
        payload=payload,
        payload_digest=payload_digest,
        nodes=nodes,
        edges=edges,
        guarded_edges=guarded_edges,
        transport_calls=(
            tuple(dict(item) for item in calls if isinstance(item, Mapping))
            if isinstance(calls, (list, tuple))
            else ()
        ),
        graph_effect_attempted=graph_effect_attempted,
        evidence_ranges=evidence_ranges or {},
        node_resolutions=node_resolutions,
        embedding_skipped=embedding_skipped,
        journal_skipped=journal_skipped,
        rollback_skipped=rollback_skipped,
        raw_output_digest=str(receipt.get("raw_output_digest") or "") or None,
        framework_version=str(
            receipt.get("framework_version") or GRAPHITI_CORE_RELEASE
        ),
        model_version=(
            str(receipt["model_version"])
            if receipt.get("model_version") is not None
            else None
        ),
        prompt_digest=str(receipt.get("prompt_digest") or "") or None,
        invocation_count=int(receipt.get("invocation_count", 1)),
        token_usage=(
            dict(token_usage)
            if isinstance(token_usage, Mapping)
            else {"basis": UNMEASURED}
        ),
        provider_cost=receipt.get("provider_cost"),
        pipeline_chat_invocations=(
            tuple(
                dict(item)
                for item in pipeline_calls
                if isinstance(item, Mapping)
            )
            if isinstance(pipeline_calls, (list, tuple))
            else ()
        ),
        embedding_usage=(
            dict(embedding_usage) if isinstance(embedding_usage, Mapping) else {}
        ),
        ingest_id=str(receipt.get("ingest_id") or "") or None,
        temporal_basis=str(receipt.get("temporal_basis") or "") or None,
        configuration_digest=(
            str(receipt.get("configuration_digest") or "") or None
        ),
        temporal_policy_digest=(
            str(receipt.get("temporal_policy_digest") or "") or None
        ),
        deterministic_sidecar=(
            dict(receipt["deterministic_sidecar"])
            if isinstance(receipt.get("deterministic_sidecar"), Mapping)
            else None
        ),
        sidecar_collapse=(
            dict(receipt["sidecar_collapse"])
            if isinstance(receipt.get("sidecar_collapse"), Mapping)
            else None
        ),
        deterministic_summary=(
            dict(receipt["deterministic_summary"])
            if isinstance(receipt.get("deterministic_summary"), Mapping)
            else None
        ),
        request_identity_digest=(
            str(receipt.get("request_identity_digest") or "") or None
        ),
        donor_artifact_digest=(
            str(receipt.get("donor_artifact_digest") or "") or None
        ),
    )


def _expand(
    revision: SourceRevisionInput,
    payload: Mapping[str, Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    created = UtcTimestamp.parse(revision.ingested_at).value
    reference = UtcTimestamp.parse(revision.reference_time).value
    ingest_id = revision.ingest_id
    nodes_by_id: dict[int, Any] = {}
    nodes: list[Any] = []
    for entity in payload["entities"]:
        node_uuid = _uuid(
            "node",
            ingest_id,
            entity["local_id"],
            entity["name"],
            entity["entity_type_id"],
        )
        node = SimpleNamespace(
            uuid=node_uuid,
            name=entity["name"],
            group_id=revision.group_id,
            labels=["Entity"],
            summary="",
            created_at=created,
            attributes={
                "entity_type_id": entity["entity_type_id"],
                "evidence_segment_ids": list(entity["evidence_segment_ids"]),
                "resolution": "UNRESOLVED",
                "ingest_id": ingest_id,
                "source_id": revision.source_id,
            },
        )
        nodes_by_id[entity["local_id"]] = node
        nodes.append(node)
    episode_uuid = revision.episode_uuid or ingest_id
    edges: list[Any] = []
    for fact in payload["facts"]:
        source = nodes_by_id[fact["source_local_id"]]
        target = nodes_by_id[fact["target_local_id"]]
        valid_at = (
            None
            if fact["valid_at"] is None
            else UtcTimestamp.parse(fact["valid_at"]).value
        )
        invalid_at = (
            None
            if fact["invalid_at"] is None
            else UtcTimestamp.parse(fact["invalid_at"]).value
        )
        edges.append(
            SimpleNamespace(
                uuid=_uuid(
                    "edge",
                    ingest_id,
                    fact["source_local_id"],
                    fact["target_local_id"],
                    fact["relation_type"],
                    fact["fact"],
                ),
                group_id=revision.group_id,
                source_node_uuid=source.uuid,
                target_node_uuid=target.uuid,
                created_at=created,
                name=fact["relation_type"],
                fact=fact["fact"],
                fact_embedding=None,
                episodes=[episode_uuid],
                expired_at=None,
                valid_at=valid_at,
                invalid_at=invalid_at,
                reference_time=reference,
                attributes={
                    "evidence_segment_ids": list(fact["evidence_segment_ids"]),
                    "temporal_basis": revision.temporal_basis,
                    "temporal_policy": TEMPORAL_POLICY_VERSION,
                    "ingest_id": ingest_id,
                },
            )
        )
    return tuple(nodes), tuple(edges)


def _uuid(*parts: object) -> str:
    digest = digest_bytes(canonical_json_bytes(list(parts)))
    return str(uuid4_from_digest(bytes.fromhex(digest.removeprefix("sha256:")[:32])))


__all__ = [
    "CALL_SHAPES_PATH",
    "CONTRACT_NAME",
    "CombinedTemporalError",
    "CombinedTemporalFailureCode",
    "CombinedTemporalLeaf",
    "CombinedTemporalOutcome",
    "CombinedTemporalTransport",
    "CombinedTemporalTransportResult",
    "CompactPrompt",
    "EvidenceSegment",
    "GOVERNED_ENTITY_TYPE_IDS",
    "GROUP_ID",
    "LIVE_PACKET_PATH",
    "MEASUREMENTS_PATH",
    "SCHEMA",
    "SCHEMA_DIGEST",
    "SourceRevisionInput",
    "UNMEASURED",
    "build_compact_prompt",
    "extract_combined_temporal",
    "segment_source",
]
