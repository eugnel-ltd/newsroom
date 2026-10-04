"""An invalid first model result leaves no Graphiti business Episode behind."""
from __future__ import annotations

import asyncio
import copy
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from newsroom.extraction.types import ExtractionContractError
from newsroom.authority.types import UtcTimestamp
from newsroom.graphiti_adapter import real
from newsroom.graphiti_adapter.combined_temporal_contract import build_compact_prompt
from newsroom.graphiti_adapter.combined_temporal_fixtures import fixture
from newsroom.graphiti_adapter.evaluation_attempt import evaluation_attempt_for
from newsroom.graphiti_adapter.neo4j_guard import GuardError, GuardState


def _run(monkeypatch, *, case_name="pair-current", invalid=False, empty=False,
         stale=False, fail_resolution=False, retained_body=None):
    case = fixture(case_name)
    revision = replace(case.revision, episode_uuid=case.revision.ingest_id)
    episode_id = revision.episode_uuid
    payload = copy.deepcopy(case.gold)
    if invalid:
        payload = {"invalid": True}
    if empty:
        payload = {"entities": [], "facts": []}
    observed = SimpleNamespace(events=[], business={}, prompts=[], contexts=[],
                               snapshots=[], rollback=0, model_calls=0)
    actual = real._load_graphiti()
    runtime = SimpleNamespace(**vars(actual))
    accepted = runtime.EntityNode(
        uuid="00000000-0000-4000-8000-000000009815", name="Retained context",
        group_id=revision.group_id, labels=["Entity"], summary="previously accepted",
        created_at=UtcTimestamp.parse(revision.ingested_at).value,
        name_embedding=[0.25, 0.75], attributes={"source_id": revision.source_id},
    )
    observed.context_before = accepted.model_dump(mode="json")

    class Guard:
        def __init__(self, driver, **values):
            self.driver = driver
            self.group_id = values["group_id"]
            self.episode_uuid = values["episode_uuid"]
            self.input_digest = values["input_digest"]

        async def begin(self):
            observed.events.append("journal")
            return SimpleNamespace(state=GuardState.CREATED)

        @asynccontextmanager
        async def fenced_graph_mutation(self):
            if stale:
                raise GuardError("lost current owner")
            observed.events.append("fence")
            yield

        async def record_pending_telemetry(self, **_values):
            pass

        async def restore_preexisting(self):
            pass

        async def complete(self, raw):
            observed.snapshots.append(copy.deepcopy(raw))
            observed.events.append("complete")

        async def rollback_pending(self, **_values):
            observed.rollback += 1
            observed.business.clear()
            return True

    class Llm:
        invocations = []

        async def _generate_response(self, messages, **_values):
            observed.model_calls += 1
            observed.events.append("model")
            observed.prompts.append(messages[0].content)
            observed.writes_at_model = len(observed.business)
            return payload

    class Embedder:
        async def create_batch(self, values):
            return [[0.25, 0.75] for _ in values]

        def receipt(self):
            return {"usage_basis": "NO_EMBEDDING_CALL", "request_count": 0}

    embedder = Embedder()
    llm = Llm()

    class Graphiti:
        def __init__(self, *_args, **values):
            self.driver = object()
            self.clients = SimpleNamespace(llm_client=values["llm_client"], embedder=values["embedder"])

        async def _process_episode_data(self, episode, nodes, edges, *_args):
            observed.events.append("persist")
            observed.final_nodes = [(str(n.uuid), str(n.name)) for n in nodes]
            observed.final_edges = [(str(e.uuid), str(e.fact)) for e in edges]
            observed.business[episode_id] = {"uuid": str(episode.uuid), "body": episode.content}

        async def close(self):
            pass

    delegate = SimpleNamespace(client=SimpleNamespace(close=AsyncMock()))
    runtime.Graphiti = Graphiti
    runtime.OpenAIEmbedder = lambda **_values: delegate
    runtime.OpenAIEmbedderConfig = lambda **values: SimpleNamespace(**values)
    runtime.MeteredOpenAIEmbedder = lambda *_args, **_values: embedder
    runtime.MutationGuard = Guard
    monkeypatch.setattr(real, "_load_graphiti", lambda: runtime)
    monkeypatch.setattr(real, "build_cli_llm_client", lambda **_values: llm)

    if retained_body is not None:
        observed.business[episode_id] = runtime.EpisodicNode(
            uuid=episode_id, name=episode_id, group_id=revision.group_id, labels=[],
            source=runtime.EpisodeType.text, source_description=revision.group_id,
            content=revision.body if retained_body == "MATCH" else retained_body,
            created_at=UtcTimestamp.parse(revision.ingested_at).value,
            valid_at=UtcTimestamp.parse(revision.reference_time).value,
        )

    async def get_episode(_driver, uuid):
        if uuid not in observed.business:
            raise runtime.NodeNotFoundError("fixture episode absent")
        return observed.business[uuid]

    async def save_episode(self, _driver):
        observed.events.append("episode")
        observed.business[str(self.uuid)] = self

    async def existing_nodes(_driver, groups, **values):
        # Real factory queries Entity nodes, not the current Episodic node.
        observed.contexts.append((tuple(groups), values, (str(accepted.uuid),)))
        observed.events.append("context")
        return (accepted,)

    async def resolve(nodes, existing, **_values):
        if fail_resolution:
            raise RuntimeError("fixture resolver failure")
        assert existing == (accepted,)
        assert accepted.model_dump(mode="json") == observed.context_before
        return nodes, {str(n.uuid): str(n.uuid) for n in nodes}, []

    async def embeddings(_embedder, edges):
        for edge in edges:
            edge.fact_embedding = [0.25, 0.75]

    monkeypatch.setattr(runtime.EpisodicNode, "get_by_uuid", get_episode)
    monkeypatch.setattr(runtime.EpisodicNode, "save", save_episode)
    monkeypatch.setattr(runtime.EntityNode, "get_by_group_ids", existing_nodes)
    monkeypatch.setattr(real, "resolve_nodes_with_optional_embeddings", resolve)
    runtime.create_entity_edge_embeddings = embeddings

    def seal(result, _telemetry, receipt):
        nodes, edges = result.nodes, result.edges
        return {"provider_attempt_number": 1, "nodes": [(str(n.uuid), str(n.name)) for n in nodes],
                "edges": [(str(e.uuid), str(e.fact)) for e in edges], "combined": dict(receipt)}

    def failure(receipt, _telemetry):
        return {"provider_attempt_number": 1, "combined": dict(receipt)}

    configuration = evaluation_attempt_for((revision.body,)).configuration
    call = real._add_episode(
        api_key="fixture", password="fixture", body=revision.body,
        name=episode_id, episode_id=episode_id,
        reference_time=UtcTimestamp.parse(revision.reference_time).value,
        telemetry=real._EpisodeTelemetry(), attempt_number=1,
        validate_result=seal, restore_result=lambda *_args: None,
        validate_failure=failure, configuration=configuration, revision=revision,
    )
    return call, observed, revision, case


def test_invalid_first_response_leaves_no_business_episode(monkeypatch):
    call, seen, _revision, _case = _run(monkeypatch, invalid=True)
    with pytest.raises(ExtractionContractError, match="MALFORMED_OBJECT"):
        asyncio.run(call)
    assert seen.model_calls == 1
    assert seen.writes_at_model == 0
    assert seen.business == {}
    assert "episode" not in seen.events
    assert seen.rollback == 0
    assert seen.snapshots[0]["combined"]["failure_code"] == "MALFORMED_OBJECT"


@pytest.mark.parametrize("case_name", ["pair-current", "explicit-valid-at", "null-temporal"])
def test_validated_outputs_preserve_prompt_context_uuid_and_fact(monkeypatch, case_name):
    call, seen, revision, case = _run(monkeypatch, case_name=case_name)
    result = asyncio.run(call)
    assert seen.writes_at_model == 0
    assert seen.prompts == [build_compact_prompt(revision).text]
    assert seen.events.index("model") < seen.events.index("episode") < seen.events.index("context")
    assert len(seen.contexts) == 1 and seen.contexts[0][1] == {"with_embeddings": True}
    assert seen.contexts[0][2] == ("00000000-0000-4000-8000-000000009815",)
    assert set(seen.business) == {revision.episode_uuid}
    assert [str(e.fact) for e in result.edges] == [f["fact"] for f in case.gold["facts"]]
    assert [(str(n.uuid), str(n.name)) for n in result.nodes] == seen.final_nodes
    assert [(str(e.uuid), str(e.fact)) for e in result.edges] == seen.final_edges
    assert seen.snapshots[0]["nodes"] == seen.final_nodes
    assert seen.snapshots[0]["edges"] == seen.final_edges
    assert seen.rollback == 0


def test_valid_empty_response_still_retains_episode(monkeypatch):
    call, seen, revision, _case = _run(monkeypatch, empty=True)
    result = asyncio.run(call)
    assert seen.writes_at_model == 0
    assert result.nodes == () and result.edges == ()
    assert set(seen.business) == {revision.episode_uuid}
    assert seen.events.index("model") < seen.events.index("episode") < seen.events.index("complete")
    assert seen.contexts == []
    assert seen.snapshots[0]["combined"]["zero_proposal_effect"] == "EXPLICIT"


def test_stale_owner_still_denies_before_model_or_episode(monkeypatch):
    call, seen, _revision, _case = _run(monkeypatch, stale=True)
    with pytest.raises(GuardError, match="lost current owner"):
        asyncio.run(call)
    assert seen.model_calls == 0 and seen.business == {}


def test_later_resolution_failure_remains_guarded_mutation(monkeypatch):
    call, seen, _revision, _case = _run(monkeypatch, fail_resolution=True)
    with pytest.raises(real.AmbiguousEpisodeEffect, match="rolled back"):
        asyncio.run(call)
    assert "episode" in seen.events and seen.rollback == 1
    assert seen.business == {}


@pytest.mark.parametrize("retained_body", ["MATCH", "unrelated retained input"])
def test_retained_episode_identity_still_denies_before_model(monkeypatch, retained_body):
    call, seen, _revision, _case = _run(monkeypatch, retained_body=retained_body)
    original = copy.deepcopy(seen.business)
    with pytest.raises(real.GraphitiAdapterContractError):
        asyncio.run(call)
    assert seen.model_calls == 0 and "episode" not in seen.events
    assert seen.business == original


def test_no_hook_empty_seal_preserves_original_exception_without_rollback():
    from newsroom.tests.test_graphiti_combined_temporal_pipeline import _Guard, _pipeline

    guard = _Guard()
    pipeline = _pipeline(guard)
    failure = RuntimeError("fixture empty seal failed")

    def fail_seal(*_args):
        raise failure

    pipeline.complete_receipt = fail_seal
    assert pipeline.prepare_execution is None
    with pytest.raises(RuntimeError) as caught:
        pipeline.execute(nodes=(), edges=(), receipt={"provider_attempt_number": 1})
    assert caught.value is failure
    assert guard.calls == ["begin", "telemetry"]
    assert guard.completed_receipt is None
