from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest
from types import SimpleNamespace

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.graphiti_adapter.combined_temporal_contract import (
    CONTRACT_NAME,
    SCHEMA,
)
from newsroom.graphiti_adapter.combined_temporal_extraction import (
    CombinedTemporalFailureCode,
    CombinedTemporalOutcome,
    CombinedTemporalTransportResult,
)
from newsroom.graphiti_adapter.combined_temporal_fixtures import fixture
from newsroom.graphiti_adapter.combined_temporal_pipeline import (
    CombinedTemporalPipelineError,
    CombinedTemporalPipelineResult,
    ExistingGraphitiPipeline,
)
from newsroom.graphiti_adapter.combined_temporal_runtime import (
    CliCombinedTemporalTransport,
    extract_combined_temporal_async,
    resolve_nodes_locally,
    resolve_nodes_with_optional_embeddings,
)
from newsroom.graphiti_adapter.evaluation_packet import GRAPHITI_CORE_RELEASE
from newsroom.graphiti_adapter.donor_store import InMemoryDonorStore
from newsroom.graphiti_adapter.neo4j_guard import GuardState
from newsroom.graphiti_adapter.deterministic_sidecar import (
    AuthorityRecordRef,
    DeterministicSidecarInput,
)
from newsroom.graphiti_adapter.deterministic_summary import (
    AdmittedSummaryAssertion,
)


class _Transport:
    def __init__(self, raw: object) -> None:
        self.raw = raw
        self.calls: list[dict[str, object]] = []

    async def generate_response(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        response_model: str,
        max_tokens: int,
    ) -> CombinedTemporalTransportResult:
        self.calls.append(
            {
                "prompt": prompt,
                "schema": schema,
                "response_model": response_model,
                "max_tokens": max_tokens,
            }
        )
        return CombinedTemporalTransportResult(
            raw=self.raw,
            framework_version=GRAPHITI_CORE_RELEASE,
            model_version="composer-2.5",
            token_usage={"basis": "UNMEASURED"},
            provider_cost=None,
        )


class _Pipeline:
    async def _prepare_attempt(self) -> None:
        return None

    async def _complete_failure(
        self, receipt: Mapping[str, object]
    ) -> Mapping[str, object]:
        return receipt

    async def _execute(
        self,
        *,
        nodes: tuple[Any, ...],
        edges: tuple[Any, ...],
        receipt: Mapping[str, object],
    ) -> CombinedTemporalPipelineResult:
        return CombinedTemporalPipelineResult(
            nodes=nodes,
            edges=edges,
            guarded_edges=edges,
            node_resolutions=tuple("DETERMINISTIC_NEW_NODE" for _ in nodes),
            graph_effect_attempted=bool(nodes or edges),
            embedding_skipped=not edges,
            journal_skipped=False,
            rollback_skipped=True,
            completed_receipt=receipt,
        )


def _authority(
    record_id: str,
    *,
    record_kind: str,
    **fields: object,
) -> AuthorityRecordRef:
    encoded = canonical_json_bytes(
        {"record_id": record_id, "record_kind": record_kind, **fields}
    )
    return AuthorityRecordRef(record_id, encoded, digest_bytes(encoded))


def _sidecar_input() -> DeterministicSidecarInput:
    definition = "source-definition:legco"
    item = "source-item:answer-42"
    revision = "source-revision:answer-42:r1"
    representation = "representation:answer-42:r1"
    evidence = "evidence:answer-42:r1"
    rights = "rights:answer-42:r1"
    chunk = "chunk:answer-42:r1:1"
    reference_time = "2026-08-21T00:00:00Z"
    return DeterministicSidecarInput(
        source_definition=_authority(
            definition,
            record_kind="SOURCE_DEFINITION",
        ),
        source_item=_authority(
            item,
            record_kind="SOURCE_ITEM",
            source_definition_id=definition,
        ),
        source_revision=_authority(
            revision,
            record_kind="SOURCE_REVISION",
            source_item_id=item,
            predecessor_revision_id=None,
            representation_id=representation,
            evidence_package_id=evidence,
            rights_decision_id=rights,
            chunk_id=chunk,
            reference_time=reference_time,
        ),
        predecessor_revision=None,
        discovery_representation=_authority(
            representation,
            record_kind="DISCOVERY_REPRESENTATION",
            source_revision_id=revision,
            evidence_package_id=evidence,
        ),
        evidence_package=_authority(
            evidence,
            record_kind="EVIDENCE_PACKAGE",
            source_revision_id=revision,
        ),
        rights_decision=_authority(
            rights,
            record_kind="RIGHTS_DECISION",
            source_revision_id=revision,
        ),
        chunk=_authority(
            chunk,
            record_kind="SOURCE_CHUNK",
            source_revision_id=revision,
            chunk_ordinal=1,
            predecessor_chunk_id=None,
        ),
        predecessor_chunk=None,
        reference_time=reference_time,
    )


@pytest.mark.parametrize("fixture_name", ("zero-result", "pair-current"))
def test_runtime_uses_one_combined_temporal_primary_leaf(
    fixture_name: str,
) -> None:
    case = fixture(fixture_name)
    transport = _Transport(case.gold)

    leaf = asyncio.run(
        extract_combined_temporal_async(
            case.revision,
            transport=transport,
            pipeline=_Pipeline(),
            max_tokens=4_000,
        )
    )

    assert leaf.outcome in {
        CombinedTemporalOutcome.TERMINAL_SUCCESS_ZERO_PROPOSALS,
        CombinedTemporalOutcome.TERMINAL_SUCCESS_WITH_PROPOSALS,
    }
    assert len(transport.calls) == 1
    assert transport.calls[0]["response_model"] == CONTRACT_NAME
    assert transport.calls[0]["schema"] == SCHEMA
    assert transport.calls[0]["max_tokens"] == 4_000
    assert [call["response_model"] for call in transport.calls] == [
        "NewsroomCombinedTemporalExtractionV1"
    ]


def test_valid_runtime_result_has_no_ordinary_graphiti_chat_leaf() -> None:
    case = fixture("pair-current")
    transport = _Transport(case.gold)

    asyncio.run(
        extract_combined_temporal_async(
            case.revision,
            transport=transport,
            pipeline=_Pipeline(),
        )
    )

    ordinary = {
        "ExtractedEntities",
        "ExtractedEdges",
        "EdgeTimestamps",
        "BatchEdgeTimestamps",
        "NodeResolutions",
        "SummarizedEntities",
    }
    assert ordinary.isdisjoint(
        {str(call["response_model"]) for call in transport.calls}
    )


@pytest.mark.parametrize(
    ("fixture_name", "expected_valid_at", "expected_invalid_at"),
    (
        ("pair-current", None, None),
        ("explicit-valid-at", datetime(2026, 8, 20, tzinfo=UTC), None),
    ),
)
def test_runtime_preserves_gold_identity_evidence_and_temporal_values(
    fixture_name: str,
    expected_valid_at: datetime | None,
    expected_invalid_at: datetime | None,
) -> None:
    case = fixture(fixture_name)
    leaf = asyncio.run(
        extract_combined_temporal_async(
            case.revision,
            transport=_Transport(case.gold),
            pipeline=_Pipeline(),
        )
    )

    assert leaf.ingest_id == case.revision.ingest_id
    assert leaf.payload == case.gold
    assert [node.name for node in leaf.nodes] == [
        "Legislative Council",
        "Technology and Living curriculum",
    ]
    assert leaf.edges[0].name == "ASKED_ABOUT"
    assert leaf.edges[0].episodes == [case.revision.ingest_id]
    assert leaf.edges[0].valid_at == expected_valid_at
    assert leaf.edges[0].invalid_at == expected_invalid_at
    assert leaf.evidence_ranges[case.gold["facts"][0]["fact"]][0].segment_id == 0


def test_invalid_compact_output_fails_before_graph_effect_without_redispatch() -> None:
    case = fixture("pair-current")
    transport = _Transport(
        {"entities": [], "facts": [{"unexpected": "shape"}]}
    )

    leaf = asyncio.run(
        extract_combined_temporal_async(
            case.revision,
            transport=transport,
            pipeline=_Pipeline(),
        )
    )

    assert leaf.outcome is CombinedTemporalOutcome.TERMINAL_ATTEMPT_FAILURE
    assert leaf.graph_effect_attempted is False
    assert len(transport.calls) == 1


@pytest.mark.parametrize("stage", ("TRANSPORT", "TYPED_VERIFIER", "PIPELINE"))
@pytest.mark.parametrize("with_identity", (False, True))
@pytest.mark.parametrize("broken_sink", (False, True))
def test_failure_diagnostic_exposes_only_bounded_stage_and_final_code_site(
    monkeypatch, stage, with_identity, broken_sink,
) -> None:
    case = fixture("pair-current")
    secret = "TOKEN_PRIVATE_SOURCE_MODEL_OUTPUT_" + case.revision.body
    error = (CombinedTemporalPipelineError(
        secret, graph_effect_attempted=False, rollback_completed=False,
    ) if stage == "PIPELINE" else RuntimeError(secret))
    events = []
    raised_at = []
    def sink(event, data):
        events.append((event, data))
        if broken_sink:
            raise OSError("diagnostic sink full: TOKEN_PRIVATE")

    monkeypatch.setattr("newsroom.control_plane.diagnostic_logging.emit_diagnostic", sink)

    def fail():
        raised_at.append(sys._getframe().f_lineno + 1)
        raise error

    class FailingTransport(_Transport):
        async def generate_response(self, **kwargs):
            if stage == "TRANSPORT":
                self.calls.append(kwargs)
                fail()
            return await super().generate_response(**kwargs)

    class RecordingPipeline(_Pipeline):
        failed_receipts = []

        async def _execute(self, **kwargs):
            if stage == "PIPELINE":
                fail()
            return await super()._execute(**kwargs)

        async def _complete_failure(self, receipt):
            self.failed_receipts.append(dict(receipt))
            return await super()._complete_failure(receipt)

    transport = FailingTransport(case.gold)
    pipeline = RecordingPipeline()

    def verify(**kwargs):
        if stage == "TYPED_VERIFIER":
            fail()

    leaf = asyncio.run(extract_combined_temporal_async(
        case.revision, transport=transport, pipeline=pipeline,
        donor_store=InMemoryDonorStore() if with_identity else None,
        typed_proposal_verifier=verify,
    ))

    assert leaf.outcome is CombinedTemporalOutcome.TERMINAL_ATTEMPT_FAILURE
    assert leaf.failure_code is CombinedTemporalFailureCode.PIPELINE_FAILED
    assert leaf.graph_effect_attempted is False
    assert leaf.journal_skipped is False
    assert leaf.rollback_skipped is True
    assert leaf.token_usage == {"basis": "UNMEASURED"}
    assert leaf.provider_cost is None
    assert len(leaf.transport_calls) == (0 if stage == "TRANSPORT" else 1)
    assert len(transport.calls) == len(pipeline.failed_receipts) == 1
    expected = {
        "stage": stage,
        "exception_class": "CombinedTemporalPipelineError" if stage == "PIPELINE" else "RuntimeError",
        "file": "test_graphiti_combined_temporal_runtime.py",
        "function": "fail", "line": raised_at[0],
    }
    if with_identity:
        identity = pipeline.failed_receipts[0]["request_identity_digest"]
        assert identity.startswith("sha256:") and len(identity) == 71
        expected["request_identity_digest"] = identity
    assert events == [("graphiti_combined_temporal_failure", expected)]
    assert len(canonical_json_bytes(events)) < 2048
    assert secret not in str(events)
    assert case.revision.body not in str(events)
    assert "/Users/" not in str(events)
    assert not {"stage", "exception_class", "file", "function", "line"}.intersection(
        pipeline.failed_receipts[0]
    )


def test_failure_diagnostic_bounds_names_and_discards_path_directories(monkeypatch):
    case = fixture("pair-current")
    secret = "TOKEN_PRIVATE_SOURCE_MODEL_OUTPUT"
    basename = "module_" + "x" * 300 + ".py"
    function_name = "fail_" + "y" * 300
    class_name = "Failure" + "z" * 300
    error = type(class_name, (RuntimeError,), {})(secret)
    events = []
    monkeypatch.setattr(
        "newsroom.control_plane.diagnostic_logging.emit_diagnostic",
        lambda event, data: events.append((event, data)),
    )

    def fail():
        raise error

    fail.__code__ = fail.__code__.replace(
        co_filename="C:\\private\\" + secret + "\\" + basename, co_name=function_name,
    )

    class FailingTransport(_Transport):
        async def generate_response(self, **kwargs):
            fail()

    leaf = asyncio.run(extract_combined_temporal_async(
        case.revision, transport=FailingTransport(case.gold), pipeline=_Pipeline(),
    ))
    assert leaf.failure_code is CombinedTemporalFailureCode.PIPELINE_FAILED
    event, data = events[0]
    assert event == "graphiti_combined_temporal_failure"
    assert data == {
        "stage": "TRANSPORT", "exception_class": class_name[:128],
        "file": basename[:128], "function": function_name[:128],
        "line": fail.__code__.co_firstlineno + 1,
    }
    assert type(data["line"]) is int and data["line"] > 0
    assert len(canonical_json_bytes(events)) < 2048
    assert secret not in str(events)
    assert "C:" not in str(events) and "private" not in str(events)


@pytest.mark.parametrize("broken_sink", (False, True))
def test_wrapped_pipeline_diagnostic_retains_one_explicit_cause_site_only(
    monkeypatch, broken_sink,
):
    case = fixture("pair-current")
    secret = "TOKEN_PRIVATE_SOURCE_MODEL_OUTPUT_" + case.revision.body
    events, cause_lines, outer_lines, failed_receipts = [], [], [], []

    def sink(event, data):
        events.append((event, data))
        if broken_sink:
            raise OSError("diagnostic sink full: " + secret)

    monkeypatch.setattr("newsroom.control_plane.diagnostic_logging.emit_diagnostic", sink)

    def fail_original():
        cause_lines.append(sys._getframe().f_lineno + 1)
        raise ValueError(secret) from RuntimeError("DEEP_CAUSE_TOKEN_PRIVATE")

    fail_original.__code__ = fail_original.__code__.replace(
        co_filename="C:\\private\\TOKEN_DIRECTORY\\original_pipeline.py",
    )

    class WrappedPipeline(_Pipeline):
        async def _execute(self, **kwargs):
            try:
                fail_original()
            except ValueError as exc:
                outer_lines.append(sys._getframe().f_lineno + 1)
                raise CombinedTemporalPipelineError(
                    secret, graph_effect_attempted=False, rollback_completed=False,
                ) from exc

        async def _complete_failure(self, receipt):
            failed_receipts.append(dict(receipt))
            return await super()._complete_failure(receipt)

    transport = _Transport(case.gold)
    leaf = asyncio.run(extract_combined_temporal_async(
        case.revision, transport=transport, pipeline=WrappedPipeline(),
    ))
    assert leaf.outcome is CombinedTemporalOutcome.TERMINAL_ATTEMPT_FAILURE
    assert leaf.failure_code is CombinedTemporalFailureCode.PIPELINE_FAILED
    assert leaf.graph_effect_attempted is False
    assert leaf.journal_skipped is False and leaf.rollback_skipped is True
    assert leaf.token_usage == {"basis": "UNMEASURED"} and leaf.provider_cost is None
    assert len(transport.calls) == len(failed_receipts) == 1
    assert events == [("graphiti_combined_temporal_failure", {
        "stage": "PIPELINE", "exception_class": "CombinedTemporalPipelineError",
        "file": "test_graphiti_combined_temporal_runtime.py",
        "function": "_execute", "line": outer_lines[0],
        "cause_exception_class": "ValueError", "cause_file": "original_pipeline.py",
        "cause_function": "fail_original", "cause_line": cause_lines[0],
    })]
    assert len(canonical_json_bytes(events)) < 2048
    assert secret not in str(events) and "TOKEN_DIRECTORY" not in str(events)
    assert "DEEP_CAUSE" not in str(events) and "C:" not in str(events)
    assert not any(key.startswith("cause_") for key in failed_receipts[0])


def test_pre_effect_pipeline_failure_completes_the_pending_journal() -> None:
    case = fixture("pair-current")

    class IdentityMismatchPipeline(_Pipeline):
        complete_failure_calls = 0

        async def _complete_failure(
            self, receipt: Mapping[str, object]
        ) -> Mapping[str, object]:
            self.complete_failure_calls += 1
            return await super()._complete_failure(receipt)

        async def _execute(
            self,
            *,
            nodes: tuple[Any, ...],
            edges: tuple[Any, ...],
            receipt: Mapping[str, object],
        ) -> CombinedTemporalPipelineResult:
            raise CombinedTemporalPipelineError(
                "combined-temporal pipeline identity differs",
                graph_effect_attempted=False,
                rollback_completed=False,
            )

    pipeline = IdentityMismatchPipeline()
    leaf = asyncio.run(
        extract_combined_temporal_async(
            case.revision,
            transport=_Transport(case.gold),
            pipeline=pipeline,
        )
    )

    assert leaf.outcome is CombinedTemporalOutcome.TERMINAL_ATTEMPT_FAILURE
    assert leaf.failure_code is CombinedTemporalFailureCode.PIPELINE_FAILED
    assert leaf.graph_effect_attempted is False
    assert leaf.journal_skipped is False
    assert pipeline.complete_failure_calls == 1


def test_rolled_back_pipeline_failure_is_not_returned_as_complete() -> None:
    case = fixture("pair-current")

    class RolledBackPipeline(_Pipeline):
        complete_failure_calls = 0

        async def _complete_failure(
            self, receipt: Mapping[str, object]
        ) -> Mapping[str, object]:
            self.complete_failure_calls += 1
            return await super()._complete_failure(receipt)

        async def _execute(
            self,
            *,
            nodes: tuple[Any, ...],
            edges: tuple[Any, ...],
            receipt: Mapping[str, object],
        ) -> CombinedTemporalPipelineResult:
            raise CombinedTemporalPipelineError(
                "combined-temporal pipeline failed",
                graph_effect_attempted=True,
                rollback_completed=True,
            )

    pipeline = RolledBackPipeline()
    with pytest.raises(CombinedTemporalPipelineError) as failure:
        asyncio.run(
            extract_combined_temporal_async(
                case.revision,
                transport=_Transport(case.gold),
                pipeline=pipeline,
            )
        )

    assert failure.value.rollback_completed is True
    assert pipeline.complete_failure_calls == 0


def test_sidecar_uses_no_extra_provider_leaf_and_exact_collapse_keeps_attribution(
) -> None:
    sidecar = _sidecar_input()
    source_revision_id = sidecar.source_revision.record_id
    source_item_id = sidecar.source_item.record_id
    other_item_id = "source-item:answer-43"
    body = (
        f"{source_revision_id} is a revision of {source_item_id}. "
        f"{source_revision_id} is a revision of {other_item_id}."
    )
    base = fixture("pair-current").revision
    revision = replace(
        base,
        body=body,
        representation_digest=digest_bytes(body.encode("utf-8")),
    )
    entities = [
        {
            "local_id": 0,
            "name": source_revision_id,
            "entity_type_id": 0,
            "evidence_segment_ids": [0, 1],
        },
        {
            "local_id": 1,
            "name": source_item_id,
            "entity_type_id": 0,
            "evidence_segment_ids": [0],
        },
        {
            "local_id": 2,
            "name": other_item_id,
            "entity_type_id": 0,
            "evidence_segment_ids": [1],
        },
    ]
    facts = [
        {
            "source_local_id": 0,
            "target_local_id": target,
            "relation_type": "REVISION_OF",
            "fact": sentence,
            "valid_at": None,
            "invalid_at": None,
            "evidence_segment_ids": [target - 1],
        }
        for target, sentence in (
            (1, f"{source_revision_id} is a revision of {source_item_id}."),
            (2, f"{source_revision_id} is a revision of {other_item_id}."),
        )
    ]
    transport = _Transport({"entities": entities, "facts": facts})

    leaf = asyncio.run(
        extract_combined_temporal_async(
            revision,
            transport=transport,
            pipeline=_Pipeline(),
            sidecar_input=sidecar,
        )
    )

    assert len(transport.calls) == 1
    assert leaf.deterministic_sidecar is not None
    assert leaf.deterministic_sidecar["model_leaf_count"] == 0
    assert leaf.deterministic_sidecar["proposal_only"] is True
    bindings = leaf.deterministic_sidecar["relation_proposals"][0][
        "authority_bindings"
    ]
    assert bindings == [
        {
            "record_id": source_revision_id,
            "canonical_digest": sidecar.source_revision.canonical_digest,
        },
        {
            "record_id": source_item_id,
            "canonical_digest": sidecar.source_item.canonical_digest,
        },
    ]
    assert leaf.sidecar_collapse is not None
    assert len(leaf.sidecar_collapse["collapsed_duplicates"]) == 1
    assert len(leaf.sidecar_collapse["semantic_relation_proposals"]) == 1
    assert leaf.edges[0].attributes["collapsed_sidecar_duplicate"] is True
    assert "collapsed_sidecar_duplicate" not in leaf.edges[1].attributes


@pytest.mark.parametrize(
    ("text", "expected_outcome", "expected_summary"),
    (
        (
            "A short admitted assertion.",
            "DETERMINISTIC_SUMMARY",
            "A short admitted assertion.",
        ),
        ("x" * 1_025, "OVERLONG_HOLD", None),
    ),
)
def test_admitted_summaries_are_deterministic_and_never_add_a_provider_leaf(
    text: str,
    expected_outcome: str,
    expected_summary: str | None,
) -> None:
    decision = _authority(
        "admission:assertion-1",
        record_kind="ADMISSION_DECISION",
        admitted_assertion_id="assertion-1",
    )
    assertion = AdmittedSummaryAssertion(
        assertion_id="assertion-1",
        text=text,
        evidence_links=("evidence:1",),
        temporal_links=("temporal:1",),
        admission_decision=decision,
    )
    case = fixture("zero-result")
    transport = _Transport(case.gold)

    leaf = asyncio.run(
        extract_combined_temporal_async(
            case.revision,
            transport=transport,
            pipeline=_Pipeline(),
            admitted_summary_assertions=(assertion,),
        )
    )

    assert len(transport.calls) == 1
    assert leaf.deterministic_summary is not None
    assert leaf.deterministic_summary["outcome"] == expected_outcome
    assert leaf.deterministic_summary["summary"] == expected_summary
    assert leaf.deterministic_summary["provider_leaf_count"] == 0


@pytest.mark.parametrize(
    ("mention_name", "canonical_name", "aliases", "expected_basis"),
    (
        ("Education Bureau", "Education Bureau", (), "EXACT_NAME_AND_TYPE"),
        ("EDB", "Education Bureau", ("EDB",), "GOVERNED_ALIAS_OR_IDENTIFIER"),
        ("education-bureau", "Education Bureau", (), "NORMALISED_NAME_AND_TYPE"),
    ),
)
def test_common_entity_mentions_resolve_locally_without_chat_leaf(
    mention_name: str,
    canonical_name: str,
    aliases: tuple[str, ...],
    expected_basis: str,
) -> None:
    mention = SimpleNamespace(
        uuid="mention:1",
        name=mention_name,
        attributes={"entity_type_id": 2},
    )
    canonical = SimpleNamespace(
        uuid="canonical:edb",
        name=canonical_name,
        attributes={
            "entity_type_id": 2,
            "governed_aliases": aliases,
            "permitted_source_ids": ("source:legco",),
        },
    )

    original_canonical = dict(canonical.attributes)

    resolved, uuid_map, provider_calls = resolve_nodes_locally(
        [mention],
        (canonical,),
        source_id="source:legco",
    )

    assert resolved == [mention]
    assert uuid_map == {"mention:1": "canonical:edb"}
    assert mention.attributes["resolution"] == "DETERMINISTIC_EXISTING_NODE"
    assert mention.attributes["resolution_basis"] == expected_basis
    assert mention.attributes["canonical_identity"] == "canonical:edb"
    assert canonical.attributes == original_canonical
    assert provider_calls == []


def test_exact_name_resolution_does_not_call_the_embedder() -> None:
    mention = SimpleNamespace(
        uuid="mention:1",
        name="Education Bureau",
        attributes={"entity_type_id": 2},
    )
    canonical = SimpleNamespace(
        uuid="canonical:edb",
        name="Education Bureau",
        name_embedding=[1.0, 0.0],
        attributes={
            "entity_type_id": 2,
            "permitted_source_ids": ("source:legco",),
        },
    )
    embed_calls: list[str] = []

    async def embed_name(name: str) -> list[float]:
        embed_calls.append(name)
        return [1.0, 0.0]

    resolved, uuid_map, _provider = asyncio.run(
        resolve_nodes_with_optional_embeddings(
            [mention],
            (canonical,),
            source_id="source:legco",
            embed_name=embed_name,
        )
    )

    assert embed_calls == []
    assert uuid_map == {"mention:1": "canonical:edb"}
    assert resolved[0].attributes["resolution"] == "DETERMINISTIC_EXISTING_NODE"


def test_unresolved_mentions_use_one_ordered_embedding_batch() -> None:
    from newsroom.graphiti_adapter.embedding_meter import MeteredOpenAIEmbedder

    mentions = [
        SimpleNamespace(
            uuid=f"mention:{index}",
            name=f"Agency\n{index}",
            attributes={"entity_type_id": 2},
        )
        for index in range(7)
    ]
    canonical = SimpleNamespace(
        uuid="canonical:agency",
        name="Existing Agency",
        name_embedding=[1.0, 0.0],
        attributes={
            "entity_type_id": 2,
            "permitted_source_ids": ("source:legco",),
        },
    )
    calls: list[list[str]] = []
    observer_events: list[str] = []

    class Embeddings:
        async def create(self, **values: object) -> object:
            names = list(values["input"])
            calls.append(names)
            observer_events.append("DISPATCH")
            return SimpleNamespace(
                id="mention-batch",
                data=[
                    SimpleNamespace(embedding=[1.0, 0.0])
                    for _name in names
                ],
                usage={
                    "prompt_tokens": len(names),
                    "total_tokens": len(names),
                    "cost": "0",
                },
            )

    class Observer:
        def before_embedding_invocation(self, **_values: object) -> object:
            observer_events.append("ALLOCATE")
            return "mention-batch"

        def after_embedding_invocation(
            self, token: object, *, outcome: str, usage: dict[str, object]
        ) -> None:
            assert token == "mention-batch"
            assert usage["total_tokens"] == 7
            observer_events.append(outcome)

    meter = MeteredOpenAIEmbedder(
        SimpleNamespace(
            client=SimpleNamespace(embeddings=Embeddings()),
            config=SimpleNamespace(
                embedding_model="openai/text-embedding-3-large",
                embedding_dim=2,
            ),
        ),
        invocation_observer=Observer(),
    )

    resolved, uuid_map, _extra = asyncio.run(
        resolve_nodes_with_optional_embeddings(
            mentions,
            (canonical,),
            source_id="source:legco",
            embed_names=meter.create_batch,
        )
    )

    assert calls == [[f"Agency {index}" for index in range(7)]]
    assert observer_events == ["ALLOCATE", "DISPATCH", "COMPLETE"]
    assert meter.receipt()["request_count"] == 1
    assert resolved == mentions
    assert uuid_map == {
        f"mention:{index}": "canonical:agency" for index in range(7)
    }
    assert all(
        mention.attributes["resolution"] == "DETERMINISTIC_EXISTING_NODE"
        for mention in mentions
    )
    assert canonical.name_embedding == [1.0, 0.0]
    assert all(not hasattr(mention, "name_embedding") for mention in mentions)


@pytest.mark.parametrize("canonical_embedding", ([1.0, 0.0], [0.0, 0.0]))
def test_unresolved_new_mention_reuses_validated_name_embedding(
    canonical_embedding: list[float],
) -> None:
    new = SimpleNamespace(
        uuid="mention:new",
        name="New Agency",
        attributes={"entity_type_id": 2},
    )
    known = SimpleNamespace(
        uuid="mention:known",
        name="Existing Agency",
        attributes={"entity_type_id": 2},
    )
    canonical = SimpleNamespace(
        uuid="canonical:agency",
        name="Existing Agency",
        name_embedding=canonical_embedding,
        attributes={
            "entity_type_id": 2,
            "permitted_source_ids": ("source:legco",),
        },
    )
    calls: list[list[str]] = []

    async def embed_names(names: list[str]) -> list[list[float]]:
        calls.append(names)
        return [[0.0, 1.0]]

    resolved, uuid_map, _extra = asyncio.run(
        resolve_nodes_with_optional_embeddings(
            [new, known],
            (canonical,),
            source_id="source:legco",
            embed_names=embed_names,
        )
    )

    assert calls == [["New Agency"]]
    assert resolved == [new, known]
    assert uuid_map == {
        "mention:new": "mention:new",
        "mention:known": "canonical:agency",
    }
    assert new.attributes["resolution"] == "DETERMINISTIC_NEW_NODE"
    assert new.name_embedding == [0.0, 1.0]
    assert known.attributes["resolution"] == "DETERMINISTIC_EXISTING_NODE"
    assert canonical.name_embedding == canonical_embedding
    assert not hasattr(known, "name_embedding")


@pytest.mark.parametrize(
    "embeddings",
    (
        [[1.0, 0.0]],
        [[1.0], [1.0]],
    ),
)
def test_unresolved_mention_embedding_batch_rejects_malformed_output(
    embeddings: list[list[float]],
) -> None:
    mentions = [
        SimpleNamespace(
            uuid=f"mention:{index}",
            name=f"Agency {index}",
            attributes={"entity_type_id": 2},
        )
        for index in range(2)
    ]
    canonical = SimpleNamespace(
        uuid="canonical:agency",
        name="Existing Agency",
        name_embedding=[1.0, 0.0],
        attributes={
            "entity_type_id": 2,
            "permitted_source_ids": ("source:legco",),
        },
    )

    async def embed_names(_names: list[str]) -> list[list[float]]:
        return embeddings

    with pytest.raises(ValueError, match="embedding (batch cardinality|vector) differs"):
        asyncio.run(
            resolve_nodes_with_optional_embeddings(
                mentions,
                (canonical,),
                source_id="source:legco",
                embed_names=embed_names,
            )
        )
    assert all(not hasattr(mention, "name_embedding") for mention in mentions)


def test_similar_distinct_and_low_margin_mentions_are_not_forced_to_merge() -> None:
    distinct = SimpleNamespace(
        uuid="mention:distinct",
        name="Bank of East Asia",
        attributes={"entity_type_id": 2},
    )
    ambiguous = SimpleNamespace(
        uuid="mention:ambiguous",
        name="Lee",
        attributes={"entity_type_id": 2},
    )
    first = SimpleNamespace(
        uuid="canonical:first",
        name="Bank of China",
        attributes={
            "entity_type_id": 2,
            "permitted_source_ids": ("source:legco",),
        },
    )
    second = SimpleNamespace(
        uuid="canonical:second",
        name="Lee Holdings",
        attributes={
            "entity_type_id": 2,
            "permitted_source_ids": ("source:legco",),
        },
    )
    third = SimpleNamespace(
        uuid="canonical:third",
        name="Lee Group",
        attributes={
            "entity_type_id": 2,
            "permitted_source_ids": ("source:legco",),
        },
    )

    resolved, uuid_map, provider_calls = resolve_nodes_locally(
        [distinct, ambiguous],
        (first, second, third),
        source_id="source:legco",
        similarities_ppm={
            ("mention:distinct", "canonical:first"): 720_000,
            ("mention:ambiguous", "canonical:second"): 930_000,
            ("mention:ambiguous", "canonical:third"): 910_000,
        },
    )

    assert resolved == [distinct, ambiguous]
    assert uuid_map == {"mention:distinct": "mention:distinct"}
    assert distinct.attributes["resolution"] == "DETERMINISTIC_NEW_NODE"
    assert ambiguous.attributes["resolution"] == "AMBIGUOUS_HOLD"
    assert provider_calls == []


def test_cli_transport_forwards_compact_schema_identity_and_max_tokens() -> None:
    calls: list[dict[str, object]] = []

    class LlmClient:
        invocations = [
            {
                "model": "composer-2.5",
                "usage": {"usage_basis": "UNMEASURED"},
            }
        ]

        async def _generate_response(
            self,
            messages: list[object],
            *,
            response_model: type[object],
            max_tokens: int,
        ) -> dict[str, object]:
            calls.append(
                {
                    "content": messages[0].content,
                    "response_model": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                    "max_tokens": max_tokens,
                }
            )
            return {"entities": [], "facts": []}

    result = asyncio.run(
        CliCombinedTemporalTransport(LlmClient()).generate_response(
            prompt="compact prompt",
            schema=SCHEMA,
            response_model=CONTRACT_NAME,
            max_tokens=4_000,
        )
    )

    assert result.raw == {"entities": [], "facts": []}
    assert calls == [
        {
            "content": "compact prompt",
            "response_model": "NewsroomCombinedTemporalExtractionV1",
            "schema": SCHEMA,
            "max_tokens": 4_000,
        }
    ]


def test_multi_chunk_revision_retains_every_byte_and_one_revision_denominator() -> None:
    base = fixture("zero-result").revision
    chunks = ("First retained chunk.", "Second retained chunk.")
    revisions = tuple(
        replace(
            base,
            body=body,
            revision_id="revision:shared",
            representation_digest=digest_bytes(body.encode("utf-8")),
            chunk_ordinal=ordinal,
        )
        for ordinal, body in enumerate(chunks, start=1)
    )
    leaves = tuple(
        asyncio.run(
            extract_combined_temporal_async(
                revision,
                transport=_Transport({"entities": [], "facts": []}),
                pipeline=_Pipeline(),
            )
        )
        for revision in revisions
    )

    assert b"".join(
        revision.body.encode("utf-8") for revision in revisions
    ) == b"First retained chunk.Second retained chunk."
    assert [revision.revision_id for revision in revisions] == [
        "revision:shared",
        "revision:shared",
    ]
    assert [revision.chunk_ordinal for revision in revisions] == [1, 2]
    assert len({leaf.ingest_id for leaf in leaves}) == 2
    assert all(
        revision.body in leaf.prompt.text
        for revision, leaf in zip(revisions, leaves, strict=True)
    )


def test_complete_replay_skips_chat_embedding_proposal_and_graph_effects() -> None:
    case = fixture("pair-current")

    class ReplayPipeline(_Pipeline):
        completed: dict[str, object] | None = None
        execute_calls = 0

        async def _prepare_attempt(self) -> Mapping[str, object] | None:
            return self.completed

        async def _execute(
            self,
            *,
            nodes: tuple[Any, ...],
            edges: tuple[Any, ...],
            receipt: Mapping[str, object],
        ) -> CombinedTemporalPipelineResult:
            self.execute_calls += 1
            durable = dict(receipt)
            proposal = dict(durable["proposal_receipt"])
            proposal["entity_mentions"] = [
                {
                    **dict(item),
                    "canonical_identity": str(node.uuid),
                    "resolution": "DETERMINISTIC_NEW_NODE",
                }
                for item, node in zip(
                    proposal["entity_mentions"], nodes, strict=True
                )
            ]
            proposal["relation_proposals"] = [
                {
                    **dict(item),
                    "proposal_identity": str(edge.uuid),
                    "source_identity": str(edge.source_node_uuid),
                    "target_identity": str(edge.target_node_uuid),
                    "fact_embedding": [0.5],
                }
                for item, edge in zip(
                    proposal["relation_proposals"], edges, strict=True
                )
            ]
            durable["proposal_receipt"] = proposal
            self.completed = durable
            return await super()._execute(
                nodes=nodes,
                edges=edges,
                receipt=durable,
            )

    pipeline = ReplayPipeline()
    first_transport = _Transport(case.gold)
    first = asyncio.run(
        extract_combined_temporal_async(
            case.revision,
            transport=first_transport,
            pipeline=pipeline,
        )
    )
    replay_transport = _Transport(RuntimeError("provider must stay unused"))
    replayed = asyncio.run(
        extract_combined_temporal_async(
            case.revision,
            transport=replay_transport,
            pipeline=pipeline,
        )
    )

    assert len(first_transport.calls) == 1
    assert replay_transport.calls == []
    assert pipeline.execute_calls == 1
    assert replayed.payload == first.payload
    assert [node.uuid for node in replayed.nodes] == [
        node.uuid for node in first.nodes
    ]


def test_ambiguous_hold_does_not_mint_or_guess_relation_endpoint() -> None:
    case = fixture("pair-current")
    completed: dict[str, object] = {}
    persisted: list[tuple[list[object], list[object]]] = []
    resolved_mentions: list[object] = []
    resolved_identities: dict[str, str] = {}

    class Guard:
        async def begin(self) -> object:
            return SimpleNamespace(state=GuardState.CREATED)

        async def record_pending_telemetry(self, **_values: object) -> None:
            return None

        @asynccontextmanager
        async def fenced_graph_mutation(self):
            yield

        async def restore_preexisting(self) -> None:
            return None

        async def complete(self, receipt: dict[str, object]) -> None:
            completed.update(receipt)

        async def rollback_pending(self, **_values: object) -> bool:
            return True

    candidates = (
        SimpleNamespace(
            uuid="canonical:legco-a",
            name="Legislative Council A",
            attributes={
                "entity_type_id": 0,
                "permitted_source_ids": (case.revision.source_id,),
            },
        ),
        SimpleNamespace(
            uuid="canonical:legco-b",
            name="Legislative Council B",
            attributes={
                "entity_type_id": 0,
                "permitted_source_ids": (case.revision.source_id,),
            },
        ),
    )

    async def resolve(nodes: list[object]):
        source = next(node for node in nodes if node.name == "Legislative Council")
        resolved, uuid_map, duplicates = resolve_nodes_locally(
            nodes,
            candidates,
            source_id=case.revision.source_id,
            similarities_ppm={
                (str(source.uuid), "canonical:legco-a"): 930_000,
                (str(source.uuid), "canonical:legco-b"): 910_000,
            },
        )
        resolved_mentions.extend(resolved)
        resolved_identities.update(uuid_map)
        return resolved, uuid_map, duplicates

    async def persist(nodes: list[object], edges: list[object]) -> None:
        persisted.append((nodes, edges))

    pipeline = ExistingGraphitiPipeline(
        guard=Guard(),  # type: ignore[arg-type]
        resolve_nodes=resolve,
        resolve_pointers=lambda edges, _uuid_map: edges,
        create_embeddings=lambda _embedder, _edges: asyncio.sleep(0),
        persist_graph=persist,
        embedder=object(),
        run_async=asyncio.run,
        chat_receipt=lambda: [{"model": "composer-2.5"}],
        embedding_receipt=lambda: {"usage_basis": "NO_EMBEDDING_CALL"},
    )

    leaf = asyncio.run(
        extract_combined_temporal_async(
            case.revision,
            transport=_Transport(case.gold),
            pipeline=pipeline,
        )
    )

    held = resolved_mentions[0]
    assert held.attributes["resolution"] == "AMBIGUOUS_HOLD"
    assert held.attributes["resolution_basis"] == "LOW_CONFIDENCE_OR_MARGIN"
    assert held.attributes["considered_canonical_entity_count"] == 2
    assert held.attributes["considered_canonical_entity_digest"] == digest_bytes(canonical_json_bytes([
        "canonical:legco-a", "canonical:legco-b",
    ]))
    assert str(held.uuid) not in resolved_identities
    # The accepted empty-effect contract does not persist a leftover entity
    # when its only relation has an ambiguous endpoint.
    assert leaf.nodes == leaf.edges == leaf.node_resolutions == ()
    assert persisted == []
    assert completed["zero_proposal_effect"] == "EXPLICIT"
    assert completed["invocation_count"] == 1
    proposal = completed["proposal_receipt"]
    assert proposal["entity_mentions"] == proposal["wire_payload"]["entities"]
    assert proposal["relation_proposals"] == proposal["wire_payload"]["facts"]
    assert all("canonical_identity" not in item for item in proposal["entity_mentions"])
    assert all("source_identity" not in item for item in proposal["relation_proposals"])


def test_resolution_diagnostics_do_not_scale_output_with_entire_graph_inventory():
    candidates = tuple(
        SimpleNamespace(uuid=f"canonical-{i:032d}", name=f"Entity {i}", attributes={
            "entity_type_id": "organisation", "permitted_source_ids": ("UK-01",),
        }) for i in range(2000)
    )
    nodes = [SimpleNamespace(uuid=f"mention-{i}", name=f"Entity {i}", attributes={
        "entity_type_id": "organisation",
    }) for i in range(6)]
    resolved, identities, duplicates = resolve_nodes_locally(nodes, candidates, source_id="UK-01")
    metadata = [node.attributes for node in resolved]
    # Six ordinary mentions must not consume the 256 KiB extraction output
    # budget just by repeating the same historic candidate inventory six times.
    assert len(canonical_json_bytes(metadata)) < 4096
    expected = digest_bytes(canonical_json_bytes([candidate.uuid for candidate in candidates]))
    for i, attributes in enumerate(metadata):
        assert attributes["resolution"] == "DETERMINISTIC_EXISTING_NODE"
        assert attributes["canonical_identity"] == candidates[i].uuid
        assert attributes["considered_canonical_entity_count"] == len(candidates)
        assert attributes["considered_canonical_entity_digest"] == expected
        assert "considered_canonical_entity_ids" not in attributes
    assert identities == {node.uuid: candidates[i].uuid for i, node in enumerate(nodes)}
    assert duplicates == []

    from newsroom.graphiti_adapter.combined_temporal_pipeline import _durable_receipt
    edge = SimpleNamespace(uuid="relation", source_node_uuid=nodes[0].uuid,
                           target_node_uuid=nodes[1].uuid, fact_embedding=[0.1], attributes={})
    receipt = _durable_receipt(
        {"proposal_receipt": {"entity_mentions": [{} for _ in nodes], "relation_proposals": [{}]}},
        nodes=resolved, edges=[edge], resolutions=tuple(a["resolution"] for a in metadata),
        chat_invocations=[], embedding_usage={},
    )
    assert len(canonical_json_bytes(receipt)) < 4096
    for mention in receipt["proposal_receipt"]["entity_mentions"]:
        proof = mention["entity_resolution_proposal"]
        assert proof["considered_canonical_entity_count"] == len(candidates)
        assert proof["considered_canonical_entity_digest"] == expected
        assert "considered_canonical_entity_ids" not in proof


def test_legacy_resolution_inventory_is_compacted_when_sealing_new_receipt():
    from newsroom.graphiti_adapter.combined_temporal_pipeline import _durable_receipt
    identifiers = ["existing-a", "existing-b"]
    node = SimpleNamespace(uuid="mention", attributes={
        "considered_canonical_entity_ids": identifiers,
        "resolution_basis": "MULTIPLE_EXACT_OR_GOVERNED_MATCHES",
    })
    receipt = _durable_receipt(
        {"proposal_receipt": {"entity_mentions": [{}], "relation_proposals": []}},
        nodes=[node], edges=[], resolutions=("AMBIGUOUS_HOLD",),
        chat_invocations=[], embedding_usage={},
    )
    mention = receipt["proposal_receipt"]["entity_mentions"][0]
    assert mention["canonical_identity"] is None
    proof = mention["entity_resolution_proposal"]
    assert proof["considered_canonical_entity_count"] == 2
    assert proof["considered_canonical_entity_digest"] == digest_bytes(canonical_json_bytes(identifiers))
    assert "considered_canonical_entity_ids" not in proof
    assert node.attributes["considered_canonical_entity_ids"] == identifiers
