from __future__ import annotations

import asyncio
import copy
import inspect
import os
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from newsroom.authority._graphiti_adapter_boundary import _GraphitiAdapterBoundary
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.authority.objects import ObjectAccessDecisionId
from newsroom.authority.types import UtcTimestamp
from newsroom.extraction.types import (
    ExtractionFailureCode,
    ExtractionOutcome,
    ExtractionOutputValidation,
    VersionedExtractionComponent,
)
from newsroom.extraction.models import ExtractorContractRequest
from newsroom.graphiti_adapter import (
    DeterministicFakeGraphitiAdapter,
    GraphitiAdapterConfiguration,
    GraphitiAdapterConfigurationId,
    GraphitiAdapterContractError,
    GraphitiAttemptId,
    GraphitiAttemptRequest,
    GraphitiCleanupReceiptId,
    GraphitiCredentialClass,
    GraphitiEgressPolicy,
    GraphitiExecutionProfile,
    GraphitiInputManifest,
    GraphitiInputManifestId,
    GraphitiRuntimeMode,
    GraphitiRuntimeNotAuthorized,
    GraphitiWorkspaceId,
    GraphitiWorkspacePolicy,
    GraphitiWorkspacePolicyId,
    REAL_GRAPHITI_RUNTIME_ENABLED,
    RealGraphitiRuntimeAuthority,
)
from newsroom.graphiti_adapter.contracts import (
    GRAPHITI_ADAPTER_CODE_COMPONENT,
    GRAPHITI_ADAPTER_NORMALISATION_COMPONENT,
    GRAPHITI_ADAPTER_OUTPUT_SCHEMA_COMPONENT,
    GRAPHITI_ADAPTER_POLICY_COMPONENT,
    GRAPHITI_ADAPTER_TEMPORAL_COMPONENT,
    GRAPHITI_PROMPT_COMPONENT,
)
from newsroom.graphiti_adapter.combined_temporal_extraction import (
    CombinedTemporalFailureCode,
)
from newsroom.graphiti_adapter.evaluation_packet import (
    EVALUATION_GRAPHITI_PACKET,
    EVALUATION_WORKSPACE_POLICY,
    GRAPHITI_CHAT_MODEL,
    GRAPHITI_CORE_RELEASE,
    GRAPHITI_EMBEDDING_MODEL,
    GRAPHITI_WORKSPACE_GROUP,
)
from newsroom.graphiti_adapter.evaluation_attempt import evaluation_attempt_for
from newsroom.graphiti_adapter.real import RealGraphitiAdapter
from newsroom.graphiti_adapter.recovery_vocabulary import (
    GraphitiRecoveryClassification,
)
from newsroom.graphiti_adapter.neo4j_guard import GuardMarker, GuardState
from newsroom.graphiti_adapter.temporal_vocabulary import TemporalBasis
from newsroom.graphiti_adapter import cli_process

from .extraction_4a_helpers import contract_request, run_request, seed_extraction_fixture
from .graphiti_adapter_4d_helpers import FAKE_CONFIGURATION_ID

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_CONFIGURATION_ID = GraphitiAdapterConfigurationId.parse(
    "00000000-0000-4000-8000-000000004930"
)
EVALUATION_MANIFEST_ID = GraphitiInputManifestId.parse(
    "00000000-0000-4000-8000-000000004931"
)
EVALUATION_WORKSPACE_ID = GraphitiWorkspaceId.parse(
    "00000000-0000-4000-8000-000000004932"
)
EVALUATION_CLEANUP_ID = GraphitiCleanupReceiptId.parse(
    "00000000-0000-4000-8000-000000004933"
)
EVALUATION_ATTEMPT_ID = GraphitiAttemptId.parse(
    "00000000-0000-4000-8000-000000004934"
)


def _combined_runtime_inputs(
    body: str, episode_id: str
) -> tuple[GraphitiAdapterConfiguration, object]:
    import newsroom.graphiti_adapter.real as real

    attempt = evaluation_attempt_for((body,))
    revision = real._source_revision_input(
        attempt,
        body=body,
        ingested_at=UtcTimestamp.parse("2026-08-25T00:00:00Z"),
    )
    return attempt.configuration, replace(revision, episode_uuid=episode_id)


@asynccontextmanager
async def _uncontended_generation_fence(_guard: object):
    """Uncontended fixture; stale ownership has a separate direct regression."""
    yield


def _provider_free_pipeline(**values: object) -> object:
    import newsroom.graphiti_adapter.real as real

    async def resolve(
        nodes: list[object],
    ) -> tuple[list[object], dict[str, str], list[tuple[object, object]]]:
        return nodes, {str(node.uuid): str(node.uuid) for node in nodes}, []

    def pointers(edges: list[object], _uuid_map: dict[str, str]) -> list[object]:
        return edges

    async def embed(_embedder: object, _edges: list[object]) -> None:
        return None

    async def persist(_nodes: list[object], _edges: list[object]) -> None:
        return None

    graphiti = values["graphiti"]
    return real.ExistingGraphitiPipeline(
        guard=values["guard"],
        resolve_nodes=resolve,
        resolve_pointers=pointers,
        create_embeddings=embed,
        persist_graph=persist,
        embedder=graphiti.clients.embedder,
        run_async=asyncio.run,
        chat_receipt=lambda: list(graphiti.clients.llm_client.invocations),
        embedding_receipt=graphiti.clients.embedder.receipt,
    )


@pytest.mark.parametrize("stale_owner", (False, True))
def test_episode_creation_requires_current_generation_owner_before_provider(
    monkeypatch: pytest.MonkeyPatch, stale_owner: bool,
) -> None:
    import newsroom.graphiti_adapter.real as real
    from newsroom.graphiti_adapter.neo4j_guard import GuardError

    events: list[str] = []
    fenced = False

    class Guard:
        @asynccontextmanager
        async def fenced_graph_mutation(self):
            nonlocal fenced
            if stale_owner:
                raise GuardError("Graphiti generation mutation lost its pending claim")
            fenced = True
            try:
                yield
            finally:
                fenced = False

    class Pipeline:
        recovery_marker = None

        async def _prepare_attempt(self):
            # A paused after begin; recovery and B have replaced its owner token.
            events.append("prepared")
            return None

    pipeline = Pipeline()

    class Graphiti:
        def __init__(self, *_args: object, **_values: object) -> None:
            self.driver = object()

        async def close(self) -> None:
            events.append("close")

    delegate = SimpleNamespace(
        client=SimpleNamespace(embeddings=SimpleNamespace(), close=AsyncMock()),
        config=SimpleNamespace(embedding_model="model", embedding_dim=2),
    )
    runtime = SimpleNamespace(
        Graphiti=Graphiti, OpenAIEmbedder=lambda **_values: delegate,
        OpenAIEmbedderConfig=lambda **values: SimpleNamespace(**values),
        MeteredOpenAIEmbedder=real.MeteredOpenAIEmbedder,
        IdentityCrossEncoder=lambda: object(), EpisodeType=SimpleNamespace(text="text"),
        EpisodicNode=lambda **values: SimpleNamespace(**values),
        MutationGuard=lambda *_args, **_values: Guard(),
    )

    async def save_episode(**_values: object):
        assert fenced, "episode write must hold the current generation fence"
        if _values.get("create") is False:
            return None
        events.append("save")
        return SimpleNamespace(uuid="episode-id"), "CREATED"

    async def provider(*_args: object, **_values: object):
        assert not fenced, "provider work must not hold the database lock"
        events.append("provider")
        await pipeline.prepare_execution()
        return SimpleNamespace(
            outcome=real.CombinedTemporalOutcome.TERMINAL_SUCCESS_ZERO_PROPOSALS,
            nodes=(), edges=(),
        )

    monkeypatch.setattr(real, "_load_graphiti", lambda: runtime)
    monkeypatch.setattr(real, "build_cli_llm_client", lambda: SimpleNamespace(invocations=[]))
    monkeypatch.setattr(real, "combined_temporal_pipeline_for", lambda **_values: pipeline)
    monkeypatch.setattr(real, "_ensure_episode", save_episode)
    monkeypatch.setattr(real, "extract_combined_temporal_async", provider)
    configuration, revision = _combined_runtime_inputs("Body", "episode-id")
    call = real._add_episode(
        api_key="key", password="password", body="Body", name="episode-id",
        episode_id="episode-id", reference_time=datetime(2026, 8, 20, tzinfo=UTC),
        telemetry=real._EpisodeTelemetry(), attempt_number=1,
        validate_result=lambda *_args: {}, restore_result=lambda *_args: None,
        configuration=configuration, revision=revision,
    )
    if stale_owner:
        with pytest.raises(GuardError, match="lost its pending claim"):
            asyncio.run(call)
        assert events == ["prepared", "close"]
    else:
        asyncio.run(call)
        assert events == ["prepared", "provider", "save", "close"]


def _digest(label: str) -> str:
    return digest_canonical({"contract": label})


def _placeholder_authority() -> RealGraphitiRuntimeAuthority:
    return RealGraphitiRuntimeAuthority(
        authority_decision_digest=_digest("owner-decision"),
        framework_release="graphiti-placeholder-release",
        model_release="model-placeholder-release",
        embedding_release="embedding-placeholder-release",
        destination_contract_digest=_digest("destination"),
        data_processing_terms_digest=_digest("terms"),
        prompt_contract_digest=_digest("prompt"),
        output_schema_contract_digest=_digest("output"),
        permitted_expression_digest=_digest("expression"),
        rights_privacy_retention_digest=_digest("rights"),
        workspace_security_digest=_digest("workspace"),
        egress_credential_digest=_digest("egress"),
        budget_digest=_digest("budget"),
        evaluation_plan_digest=_digest("evaluation"),
        rollback_digest=_digest("rollback"),
    )


def _real_configuration(
    contract: ExtractorContractRequest,
    *,
    execution_profile: GraphitiExecutionProfile = GraphitiExecutionProfile.EVALUATION,
    authority: RealGraphitiRuntimeAuthority | None = None,
    workspace_policy: GraphitiWorkspacePolicy | None = None,
    framework_version: str = GRAPHITI_CORE_RELEASE,
    model_version: str = GRAPHITI_CHAT_MODEL,
    embedding_version: str = GRAPHITI_EMBEDDING_MODEL,
) -> GraphitiAdapterConfiguration:
    digest = _digest("evaluation-component")
    return GraphitiAdapterConfiguration(
        configuration_id=EVALUATION_CONFIGURATION_ID,
        runtime_mode=GraphitiRuntimeMode.REAL_GRAPHITI,
        execution_profile=execution_profile,
        framework=VersionedExtractionComponent(
            "graphiti.framework", framework_version, digest
        ),
        model=VersionedExtractionComponent("graphiti.model", model_version, digest),
        embedding=VersionedExtractionComponent(
            "graphiti.embedding", embedding_version, digest
        ),
        prompt=GRAPHITI_PROMPT_COMPONENT,
        output_schema=GRAPHITI_ADAPTER_OUTPUT_SCHEMA_COMPONENT,
        code=GRAPHITI_ADAPTER_CODE_COMPONENT,
        normalisation=GRAPHITI_ADAPTER_NORMALISATION_COMPONENT,
        temporal_policy=GRAPHITI_ADAPTER_TEMPORAL_COMPONENT,
        adapter_policy=GRAPHITI_ADAPTER_POLICY_COMPONENT,
        extractor_contract_id=contract.contract_id,
        extractor_contract_digest=contract.digest,
        workspace_policy=workspace_policy or EVALUATION_WORKSPACE_POLICY,
        fixture_case=None,
        real_runtime_authority=authority or EVALUATION_GRAPHITI_PACKET,
        idempotency_key="evaluation-real-adapter-v1",
    )


def _real_attempt(
    tmp_path: Path,
    *,
    execution_profile: GraphitiExecutionProfile = GraphitiExecutionProfile.EVALUATION,
    authority: RealGraphitiRuntimeAuthority | None = None,
    workspace_policy: GraphitiWorkspacePolicy | None = None,
) -> GraphitiAttemptRequest:
    state = seed_extraction_fixture(tmp_path / "authority")
    contract = contract_request()
    request = run_request(state, contract_id=contract.contract_id)
    configuration = _real_configuration(
        contract,
        execution_profile=execution_profile,
        authority=authority,
        workspace_policy=workspace_policy,
    )
    manifest = GraphitiInputManifest.from_run_request(
        manifest_id=EVALUATION_MANIFEST_ID,
        configuration=configuration,
        contract=contract,
        request=request,
    )
    return GraphitiAttemptRequest(
        attempt_id=EVALUATION_ATTEMPT_ID,
        attempt_number=1,
        expected_previous_attempt_id=None,
        configuration=configuration,
        workspace_id=EVALUATION_WORKSPACE_ID,
        cleanup_receipt_id=EVALUATION_CLEANUP_ID,
        manifest=manifest,
        extraction_contract=contract,
        extraction_request=request,
        replay_source=None,
        idempotency_key="evaluation-real-attempt-v1",
    )


def test_flag_is_true_for_evaluation_and_graphiti_core_is_an_optional_extra() -> None:
    assert REAL_GRAPHITI_RUNTIME_ENABLED is True
    pyproject = (_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "graphiti-core==0.29.3" in pyproject
    assert "[project.optional-dependencies]" in pyproject
    lock = (_REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "graphiti-core"' in lock
    assert 'version = "0.29.3"' in lock


def test_cli_llm_client_is_wired_for_graphiti_chat() -> None:
    from newsroom.graphiti_adapter.evaluation_packet import (
        CURSOR_AGENT_MODEL_ID,
        GRAPHITI_CHAT_FALLBACK,
        GRAPHITI_CHAT_MODEL,
        GROK_CHAT_REASONING,
    )
    from newsroom.graphiti_adapter.real import _add_episode, _ensure_episode

    source = inspect.getsource(_add_episode)
    assert "build_cli_llm_client" in source
    assert "OpenAIGenericClient" not in source
    assert "reference_time or started_at" not in inspect.getsource(
        RealGraphitiAdapter._produce
    )
    assert GRAPHITI_CHAT_MODEL == "cursor-agent-cli:composer-2.5"
    assert GRAPHITI_CHAT_FALLBACK == "grok-build-cli:grok-4.6-medium"
    assert CURSOR_AGENT_MODEL_ID == "composer-2.5"
    assert GROK_CHAT_REASONING == "medium"
    assert "uuid=episode_id" in source
    assert "EpisodicNode.get_by_uuid" in inspect.getsource(_ensure_episode)
    assert "EpisodicNode(" in inspect.getsource(_ensure_episode)
    assert "extract_combined_temporal_async" in source
    assert "graphiti.add_episode" not in source


def test_guarded_graphiti_never_invalidates_or_reuses_existing_edges(
) -> None:
    from newsroom.graphiti_adapter.edge_guard import guard_extracted_edges

    proposed = SimpleNamespace(
        source_node_uuid="source",
        target_node_uuid="target",
        fact="same fact as a pre-existing edge",
    )
    calls: list[str] = []

    def resolve(values: list[object], _uuid_map: dict[str, str]) -> list[object]:
        calls.append("resolve")
        return values

    async def embed(_embedder: object, values: list[object]) -> None:
        assert values == [proposed]
        calls.append("embed")

    new_edges, invalidated, episode_edges = asyncio.run(
        guard_extracted_edges(
            extracted_edges=[proposed],
            uuid_map={},
            embedder=object(),
            resolve_pointers=resolve,
            create_embeddings=embed,
        )
    )
    assert calls == ["resolve", "embed"]
    assert new_edges == [proposed]
    assert invalidated == []
    assert episode_edges == [proposed]


def test_edge_guard_keeps_distinct_relation_types_on_the_same_fact() -> None:
    from newsroom.graphiti_adapter.edge_guard import guard_extracted_edges

    asked = SimpleNamespace(
        source_node_uuid="source",
        target_node_uuid="target",
        name="ASKED_ABOUT",
        fact="same fact",
    )
    about = SimpleNamespace(
        source_node_uuid="source",
        target_node_uuid="target",
        name="ABOUT",
        fact="same fact",
    )

    async def embed(_embedder: object, values: list[object]) -> None:
        del _embedder
        assert values == [asked, about]

    new_edges, invalidated, episode_edges = asyncio.run(
        guard_extracted_edges(
            extracted_edges=[asked, about],
            uuid_map={},
            embedder=object(),
            resolve_pointers=lambda items, _uuid_map: items,
            create_embeddings=embed,
        )
    )
    assert new_edges == [asked, about]
    assert invalidated == []
    assert episode_edges == [asked, about]


def test_cursor_malformed_json_executes_grok_fallback_and_records_both_calls() -> None:
    from newsroom.graphiti_adapter.cli_client import run_cli_chain

    calls: list[str] = []

    def cursor(_prompt: str, *, max_tokens: int) -> str:
        calls.append("cursor")
        return "not-json"

    def grok(_prompt: str, _schema: str | None, *, max_tokens: int) -> str:
        calls.append("grok")
        return '{"value":"fallback"}'

    invocations: list[dict[str, object]] = []
    result = asyncio.run(
        run_cli_chain(
            prompt="prompt",
            schema='{"type":"object"}',
            cursor_runner=cursor,
            grok_runner=grok,
            invocations=invocations,
        )
    )
    assert result == {"value": "fallback"}
    assert calls == ["cursor", "grok"]
    assert [item["provider"] for item in invocations] == [
        "cursor-agent-cli",
        "grok-build-cli",
    ]
    assert [item["model"] for item in invocations] == ["composer-2.5", "grok-4.6"]
    assert [item["outcome"] for item in invocations] == [
        "MALFORMED_OUTPUT",
        "COMPLETE",
    ]
    assert [item["usage"]["usage_basis"] for item in invocations] == [
        "UNREPORTED",
        "UNREPORTED",
    ]


def test_disabled_fallback_fails_before_grok_allocation_or_dispatch() -> None:
    from newsroom.graphiti_adapter.cli_client import CliResponseError, run_cli_chain

    calls: list[str] = []

    def cursor(_prompt: str, *, max_tokens: int) -> str:
        del max_tokens
        calls.append("cursor")
        return "not-json"

    def grok(_prompt: str, _schema: str | None, *, max_tokens: int) -> str:
        del max_tokens
        calls.append("grok")
        raise AssertionError("disabled fallback reached Grok")

    invocations: list[dict[str, object]] = []
    with pytest.raises(CliResponseError, match="disabled before dispatch"):
        asyncio.run(
            run_cli_chain(
                prompt="prompt",
                schema='{"type":"object"}',
                cursor_runner=cursor,
                grok_runner=grok,
                invocations=invocations,
                fallback_permitted=False,
            )
        )

    assert calls == ["cursor"]
    assert [item["provider"] for item in invocations] == ["cursor-agent-cli"]
    assert [item["outcome"] for item in invocations] == ["MALFORMED_OUTPUT"]


def test_both_cli_malformed_json_results_fail_after_recording_both_calls() -> None:
    from newsroom.graphiti_adapter.cli_client import CliResponseError, run_cli_chain

    invocations: list[dict[str, object]] = []
    with pytest.raises(CliResponseError, match="JSON was not an object"):
        asyncio.run(
            run_cli_chain(
                prompt="prompt",
                schema=None,
                cursor_runner=lambda _prompt, *, max_tokens: "[]",
                grok_runner=lambda _prompt, _schema, *, max_tokens: "also malformed",
                invocations=invocations,
            )
        )
    assert [item["outcome"] for item in invocations] == [
        "MALFORMED_OUTPUT",
        "MALFORMED_OUTPUT",
    ]


def test_grok_predispatch_refusal_retains_timeout_qualification() -> None:
    from newsroom.graphiti_adapter.cli_client import (
        CliPredispatchRefusal,
        CliResponseError,
        run_cli_chain,
    )

    diagnostic = {
        "schema_version": "newsroom.graphiti-timeout-diagnostic.v1",
        "boundary": "CONTROLLER_DEADLINE",
        "phase": "PREDISPATCH_HELP",
        "cause": "CONFIGURED_TIMEOUT_EXPIRED",
        "provider_cause": "UNOBSERVED",
        "configured_timeout_ms": 20_000,
        "elapsed_ms": 20_000,
        "deadline_at": "2026-08-26T18:00:20.000000Z",
        "last_progress": "NO_OUTPUT_OBSERVED",
        "termination": "PROCESS_KILLED",
    }

    def grok(_prompt: str, _schema: str | None, *, max_tokens: int) -> str:
        del max_tokens
        raise CliPredispatchRefusal(
            "Graphiti CLI preflight timed out",
            qualification_evidence={"timeout_diagnostic": diagnostic},
        )

    invocations: list[dict[str, object]] = []
    with pytest.raises(CliResponseError, match="fallback CLI executable not found"):
        asyncio.run(
            run_cli_chain(
                prompt="prompt",
                schema=None,
                cursor_runner=lambda _prompt, *, max_tokens: "not-json",
                grok_runner=grok,
                invocations=invocations,
            )
        )

    assert invocations[-1]["outcome"] == "PREDISPATCH_REFUSED"
    assert invocations[-1]["transport_qualification"] == {
        "timeout_diagnostic": diagnostic
    }


def test_successful_qualification_retains_only_fixed_tokens_and_digests() -> None:
    from newsroom.graphiti_adapter.cli_client import CliExecution, run_cli_chain
    from newsroom.graphiti_adapter.cursor_transport import (
        MIN_COMPOSER_FLOOR,
        MIN_SDK_REQUIREMENT,
        MIN_SDK_VERSION,
        cursor_transport_policy_digest,
    )

    qualification = {
        "schema_version": "newsroom.cursor-sdk-qualification.v2",
        "transport": "CURSOR_SDK",
        "sdk_floor": MIN_SDK_VERSION,
        "sdk_version": MIN_SDK_VERSION,
        "lock_identity": MIN_SDK_REQUIREMENT,
        "composer_floor": MIN_COMPOSER_FLOOR,
        "selected_model": "composer-2.5",
        "model": "composer-2.5",
        "unary_timeout_seconds": 160,
        "stream_timeout_seconds": 160,
        "max_retries": 0,
        "transport_policy_digest": cursor_transport_policy_digest(
            selected_model="composer-2.5"
        ),
    }

    async def cursor(_prompt: str, *, max_tokens: int) -> CliExecution:
        del max_tokens
        return CliExecution(
            text="{}",
            usage={"usage_basis": "UNREPORTED"},
            transport_qualification=qualification,
        )

    invocations: list[dict[str, object]] = []
    asyncio.run(
        run_cli_chain(
            prompt="prompt",
            schema=None,
            cursor_runner=cursor,
            grok_runner=lambda *_args, **_values: pytest.fail(
                "complete primary reached fallback"
            ),
            invocations=invocations,
        )
    )

    retained = invocations[0]["transport_qualification"]
    assert isinstance(retained, dict)
    assert retained["schema_version"] == "newsroom.cursor-sdk-qualification.v2"
    assert retained["transport"] == "CURSOR_SDK"
    assert retained["sdk_version"] == MIN_SDK_VERSION
    assert retained["lock_identity"] == MIN_SDK_REQUIREMENT
    assert retained["selected_model"] == "composer-2.5"
    assert "binary" not in retained
    assert "resolved_binary" not in retained
    assert "/secret/" not in repr(retained)
    assert "crsr_" not in repr(retained)


def test_non_utf8_cursor_is_recorded_before_grok_fallback() -> None:
    from newsroom.graphiti_adapter.cli_client import (
        CliResponseError,
        run_cli_async,
        run_cli_chain,
    )

    async def invalid_cursor(_prompt: str, *, max_tokens: int) -> str:
        return await run_cli_async(
            (
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'\\xff')",
            ),
            timeout=5,
        )

    invocations: list[dict[str, object]] = []
    grok_called = False

    def grok(_prompt: str, _schema: str | None, *, max_tokens: int) -> str:
        nonlocal grok_called
        grok_called = True
        return '{"value":"fallback"}'

    with pytest.raises(CliResponseError, match="ineligible for fallback"):
        asyncio.run(
            run_cli_chain(
                prompt="prompt",
                schema=None,
                cursor_runner=invalid_cursor,
                grok_runner=grok,
                invocations=invocations,
            )
        )
    assert grok_called is False
    assert [item["outcome"] for item in invocations] == ["FAILED"]
    assert invocations[0]["failure"] == "CliOutputDecodeError"


def test_non_utf8_grok_is_recorded_before_chain_failure() -> None:
    from newsroom.graphiti_adapter.cli_client import (
        CliResponseError,
        run_cli_async,
        run_cli_chain,
    )

    async def invalid_grok(_prompt: str, _schema: str | None, *, max_tokens: int) -> str:
        return await run_cli_async(
            (
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'\\xff')",
            ),
            timeout=5,
        )

    invocations: list[dict[str, object]] = []
    with pytest.raises(CliResponseError, match="fallback CLI failed"):
        asyncio.run(
            run_cli_chain(
                prompt="prompt",
                schema=None,
                cursor_runner=lambda _prompt, *, max_tokens: "not-json",
                grok_runner=invalid_grok,
                invocations=invocations,
            )
        )
    assert [item["outcome"] for item in invocations] == [
        "MALFORMED_OUTPUT",
        "FAILED",
    ]
    assert invocations[1]["failure"] == "CliOutputDecodeError"


def test_cursor_timeout_is_ineligible_for_fallback() -> None:
    from newsroom.graphiti_adapter.cli_client import CliResponseError, run_cli_chain

    grok_called = False

    def timeout(_prompt: str, *, max_tokens: int) -> str:
        raise TimeoutError("cursor-agent Graphiti LLM timed out")

    def grok(_prompt: str, _schema: str | None, *, max_tokens: int) -> str:
        nonlocal grok_called
        grok_called = True
        return '{"value":"fallback"}'

    invocations: list[dict[str, object]] = []
    with pytest.raises(CliResponseError, match="ineligible for fallback"):
        asyncio.run(
            run_cli_chain(
                prompt="prompt",
                schema=None,
                cursor_runner=timeout,
                grok_runner=grok,
                invocations=invocations,
            )
        )

    assert grok_called is False
    assert [item["outcome"] for item in invocations] == ["TIMEOUT"]
    assert invocations[0]["transport_diagnostic"]["boundary"] == (
        "UNOBSERVED_TIMEOUT_BOUNDARY"
    )
    assert invocations[0]["transport_diagnostic"]["cause"] == (
        "TIMEOUT_ORIGIN_UNOBSERVED"
    )


def test_cursor_timeout_retains_transport_diagnostic_in_invocation() -> None:
    from newsroom.graphiti_adapter.cli_client import CliResponseError, run_cli_chain
    from newsroom.graphiti_adapter.cli_process import CliTransportTimeout

    diagnostic = {
        "schema_version": "newsroom.graphiti-timeout-diagnostic.v1",
        "boundary": "CONTROLLER_DEADLINE",
        "phase": "PRIMARY_TRANSPORT",
        "cause": "CONFIGURED_TIMEOUT_EXPIRED",
        "provider_cause": "UNOBSERVED",
        "process": "CLI_CHILD",
        "configured_timeout_ms": 160_000,
        "elapsed_ms": 160_000,
        "deadline_at": "2026-08-26T17:06:10.000000Z",
        "last_progress": "NO_OUTPUT_OBSERVED",
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "stdout_digest": digest_bytes(b""),
        "stderr_digest": digest_bytes(b""),
        "termination": "PROCESS_KILLED",
    }

    def timeout(_prompt: str, *, max_tokens: int) -> str:
        raise CliTransportTimeout(
            "cursor-agent Graphiti LLM timed out",
            evidence=diagnostic,
        )

    invocations: list[dict[str, object]] = []
    with pytest.raises(CliResponseError, match="ineligible for fallback"):
        asyncio.run(
            run_cli_chain(
                prompt="prompt",
                schema=None,
                cursor_runner=timeout,
                grok_runner=lambda *_args, **_values: pytest.fail(
                    "timeout reached fallback"
                ),
                invocations=invocations,
            )
        )

    assert invocations[0]["transport_diagnostic"] == diagnostic


def test_grok_timeout_is_recorded_as_timeout_not_failed() -> None:
    from newsroom.graphiti_adapter.cli_client import CliResponseError, run_cli_chain

    def malformed(_prompt: str, *, max_tokens: int) -> str:
        return "not-json"

    def timeout(_prompt: str, _schema: str | None, *, max_tokens: int) -> str:
        raise TimeoutError("grok Graphiti LLM timed out")

    invocations: list[dict[str, object]] = []
    with pytest.raises(CliResponseError, match="timed out"):
        asyncio.run(
            run_cli_chain(
                prompt="prompt",
                schema=None,
                cursor_runner=malformed,
                grok_runner=timeout,
                invocations=invocations,
            )
        )

    assert [item["outcome"] for item in invocations] == [
        "MALFORMED_OUTPUT",
        "TIMEOUT",
    ]
    assert invocations[1]["transport_diagnostic"]["boundary"] == (
        "UNOBSERVED_TIMEOUT_BOUNDARY"
    )


def test_sync_cli_rejects_non_utf8_output_with_typed_failure() -> None:
    from newsroom.graphiti_adapter.cli_client import (
        CliOutputDecodeError,
        run_cli,
    )

    with pytest.raises(CliOutputDecodeError, match="malformed UTF-8"):
        run_cli(
            (
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'\\xff')",
            ),
            timeout=5,
        )


@pytest.mark.parametrize("cancelled_provider", ["cursor", "grok"])
def test_cli_deadline_cancellation_is_recorded(cancelled_provider: str) -> None:
    from newsroom.graphiti_adapter.cli_client import run_cli_chain

    async def cancelled_cursor(_prompt: str, *, max_tokens: int) -> str:
        raise asyncio.CancelledError

    async def malformed_cursor(_prompt: str, *, max_tokens: int) -> str:
        return "not-json"

    async def cancelled_grok(_prompt: str, _schema: str | None, *, max_tokens: int) -> str:
        raise asyncio.CancelledError

    invocations: list[dict[str, object]] = []
    cursor = cancelled_cursor if cancelled_provider == "cursor" else malformed_cursor
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_cli_chain(
                prompt="prompt",
                schema=None,
                cursor_runner=cursor,
                grok_runner=cancelled_grok,
                invocations=invocations,
            )
        )

    expected_provider = (
        "cursor-agent-cli" if cancelled_provider == "cursor" else "grok-build-cli"
    )
    assert invocations[-1]["provider"] == expected_provider
    assert invocations[-1]["outcome"] == "CANCELLED"
    assert invocations[-1]["failure"] == "CancelledError"
    diagnostic = invocations[-1]["transport_diagnostic"]
    assert diagnostic["schema_version"] == "newsroom.graphiti-timeout-diagnostic.v1"
    assert diagnostic["boundary"] == "CALLER_CANCELLATION"
    assert diagnostic["phase"] == (
        "PRIMARY_TRANSPORT"
        if cancelled_provider == "cursor"
        else "FALLBACK_TRANSPORT"
    )
    assert diagnostic["cause"] == "CALLER_CANCELLED"
    assert diagnostic["provider_cause"] == "UNOBSERVED"
    assert diagnostic["termination"] == "TASK_CANCELLED"
    expected_outcomes = (
        ["CANCELLED"]
        if cancelled_provider == "cursor"
        else ["MALFORMED_OUTPUT", "CANCELLED"]
    )
    assert [item["outcome"] for item in invocations] == expected_outcomes


def test_deterministic_episode_creation_rejects_unmarked_retained_identity() -> None:
    from newsroom.graphiti_adapter.real import _ensure_episode

    class Missing(Exception):
        pass

    retained: dict[str, object] = {}
    saves: list[str] = []

    class Episode:
        def __init__(self, **values: object) -> None:
            for key, value in values.items():
                setattr(self, key, value)

        @classmethod
        async def get_by_uuid(cls, _driver: object, uuid: str) -> object:
            if uuid not in retained:
                raise Missing(uuid)
            return retained[uuid]

        async def save(self, _driver: object) -> None:
            retained[str(self.uuid)] = self
            saves.append(str(self.uuid))

    runtime = SimpleNamespace(
        EpisodicNode=Episode,
        NodeNotFoundError=Missing,
        EpisodeType=SimpleNamespace(text="text"),
    )
    graphiti = SimpleNamespace(driver=object())
    reference = datetime(2026, 8, 20, tzinfo=UTC)
    arguments = {
        "graphiti": graphiti,
        "runtime": runtime,
        "episode_id": "deterministic-id",
        "name": "deterministic-id",
        "body": "retained body",
        "reference_time": reference,
    }
    _episode, first_state = asyncio.run(_ensure_episode(**arguments))
    _episode, retained_state = asyncio.run(_ensure_episode(**arguments))
    assert first_state == "CREATED"
    assert retained_state == "RETAINED"
    assert saves == ["deterministic-id"]
    assert tuple(retained) == ("deterministic-id",)


def test_durable_guard_marker_restores_original_provider_metering() -> None:
    from newsroom.graphiti_adapter.real import (
        _EpisodeTelemetry,
        _restore_marker_telemetry,
    )
    from newsroom.graphiti_adapter.neo4j_guard import GuardMarker, GuardState

    telemetry = _EpisodeTelemetry()
    marker = GuardMarker(
        state=GuardState.COMPLETE,
        attempt_number=1,
        input_digest="sha256:" + "0" * 64,
        chat_invocations=({"provider": "cursor-agent-cli"},),
        embedding_usage={
            "usage_basis": "PROVIDER_REPORTED",
            "request_count": 1,
            "cost_usd_microunits": 17,
        },
    )
    _restore_marker_telemetry(telemetry, marker)
    assert telemetry.provider_attempt_number == 1
    assert telemetry.embedding_usage["cost_usd_microunits"] == 17
    assert telemetry.chat_invocations == [{"provider": "cursor-agent-cli"}]


def test_missing_node_names_use_one_ordered_normalised_batch() -> None:
    import newsroom.graphiti_adapter.real as real

    supplied: list[list[str]] = []
    nodes = [
        SimpleNamespace(name=f"node\n{index}", name_embedding=None)
        for index in range(7)
    ]

    class Embedder:
        async def create_batch(self, names: list[str]) -> list[list[float]]:
            supplied.append(names)
            return [[float(index)] for index, _name in enumerate(names)]

    asyncio.run(real._batch_missing_node_name_embeddings(Embedder(), nodes))

    assert supplied == [[f"node {index}" for index in range(7)]]
    assert [node.name_embedding for node in nodes] == [
        [float(index)] for index in range(7)
    ]


@pytest.mark.parametrize(
    "nodes",
    ([], [SimpleNamespace(name="kept", name_embedding=[1.0])]),
)
def test_node_name_batch_skips_empty_and_already_embedded_nodes(
    nodes: list[object],
) -> None:
    import newsroom.graphiti_adapter.real as real

    class Embedder:
        async def create_batch(self, _values: list[str]) -> list[list[float]]:
            pytest.fail("node embedding provider was called")

    asyncio.run(real._batch_missing_node_name_embeddings(Embedder(), nodes))


def test_malformed_node_name_batch_fails_before_partial_assignment() -> None:
    import newsroom.graphiti_adapter.real as real

    nodes = [
        SimpleNamespace(name="first", name_embedding=None),
        SimpleNamespace(name="second", name_embedding=None),
    ]

    class Embedder:
        async def create_batch(self, _values: list[str]) -> list[list[float]]:
            return [[1.0]]

    with pytest.raises(
        real.GraphitiAdapterContractError,
        match="batch cardinality differs",
    ):
        asyncio.run(real._batch_missing_node_name_embeddings(Embedder(), nodes))

    assert [node.name_embedding for node in nodes] == [None, None]


def test_episode_uses_default_database_and_validates_before_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real

    class Missing(Exception):
        pass

    retained: dict[str, object] = {}
    saves: list[str] = []
    guard_events: list[str] = []

    class Episode:
        def __init__(self, **values: object) -> None:
            for key, value in values.items():
                setattr(self, key, value)
            self.entity_edges = []

        @classmethod
        async def get_by_uuid(cls, _driver: object, episode_id: str) -> object:
            if episode_id not in retained:
                raise Missing(episode_id)
            return retained[episode_id]

        async def save(self, _driver: object) -> None:
            retained[str(self.uuid)] = self
            saves.append(str(self.uuid))

    class Driver:
        _database = "neo4j"

        def clone(self, **_values: object) -> object:
            raise AssertionError("group_id must not replace the configured database")

    class Graphiti:
        def __init__(self, *_args: object, **values: object) -> None:
            self.driver = Driver()
            self.clients = SimpleNamespace(
                driver=self.driver,
                llm_client=values["llm_client"],
                embedder=values["embedder"],
            )

        async def retrieve_episodes(
            self, *_args: object, **_values: object
        ) -> list[object]:
            raise AssertionError(
                "ambient episodes have no current rights proof and must not be reused"
            )

        async def add_episode(self, **values: object) -> object:
            raise AssertionError("ordinary graphiti-core extraction must stay unused")

        async def close(self) -> None:
            return None

    class Guard:
        fenced_graph_mutation = _uncontended_generation_fence

        async def begin(self) -> object:
            guard_events.append("begin")
            return real.GuardMarker(
                state=real.GuardState.CREATED,
                attempt_number=1,
                input_digest="sha256:" + "0" * 64,
            )

        async def record_pending_telemetry(self, **_values: object) -> None:
            guard_events.append("metered")

        async def restore_preexisting(self) -> None:
            guard_events.append("restored")

        async def complete(self, raw: dict[str, object]) -> None:
            assert raw["provider_attempt_number"] == 1
            assert "combined_temporal_receipt" in raw
            guard_events.append("complete")

    delegate = SimpleNamespace(
        client=SimpleNamespace(embeddings=SimpleNamespace(), close=AsyncMock()),
        config=SimpleNamespace(
            embedding_model="openai/text-embedding-3-large",
            embedding_dim=2,
        ),
    )
    runtime = SimpleNamespace(
        Graphiti=Graphiti,
        OpenAIEmbedder=lambda **_values: delegate,
        OpenAIEmbedderConfig=lambda **values: SimpleNamespace(**values),
        MeteredOpenAIEmbedder=real.MeteredOpenAIEmbedder,
        IdentityCrossEncoder=lambda: object(),
        EpisodeType=SimpleNamespace(text="text"),
        EpisodicNode=Episode,
        NodeNotFoundError=Missing,
        MutationGuard=lambda *_args, **_values: Guard(),
    )
    monkeypatch.setattr(real, "_load_graphiti", lambda: runtime)

    class LlmClient:
        invocations: list[dict[str, object]] = []

        async def _generate_response(self, *_args: object, **_values: object) -> object:
            return {"entities": [], "facts": []}

    monkeypatch.setattr(real, "build_cli_llm_client", LlmClient)
    monkeypatch.setattr(
        real, "combined_temporal_pipeline_for", _provider_free_pipeline
    )
    telemetry = real._EpisodeTelemetry()
    validation_states: list[str] = []

    def validate(
        _result: object, _telemetry: object, combined_receipt=None
    ) -> dict[str, object]:
        validation_states.append(guard_events[-1])
        raw: dict[str, object] = {"provider_attempt_number": 1}
        if combined_receipt is not None:
            raw["combined_temporal_receipt"] = dict(combined_receipt)
        return raw

    configuration, revision = _combined_runtime_inputs("Body", "episode-id")
    result = asyncio.run(
        real._add_episode(
            api_key="key",
            password="password",
            body="Body",
            name="episode-id",
            episode_id="episode-id",
            reference_time=datetime(2026, 8, 20, tzinfo=UTC),
            telemetry=telemetry,
            attempt_number=1,
            validate_result=validate,
            restore_result=lambda _raw, _telemetry: None,
            configuration=configuration,
            revision=revision,
        )
    )
    assert result.episode.uuid == "episode-id"
    assert validation_states == ["metered"]
    assert guard_events == ["begin", "metered", "complete"]
    assert saves == ["episode-id"]
    delegate.client.close.assert_awaited_once()


@pytest.mark.parametrize(
    ("state", "expected_event"),
    [
        ("COMPLETE", "restore_complete"),
        ("PENDING", "rollback_pending"),
    ],
)
def test_process_recovery_uses_durable_guard_before_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    expected_event: str,
) -> None:
    import newsroom.graphiti_adapter.real as real

    events: list[str] = []

    class Graphiti:
        def __init__(self, *_args: object, **values: object) -> None:
            self.driver = object()
            self.clients = SimpleNamespace(
                llm_client=values["llm_client"],
                embedder=values["embedder"],
            )

        async def add_episode(self, **_values: object) -> object:
            raise AssertionError("recovery must happen before provider dispatch")

        async def close(self) -> None:
            events.append("close")

    class Guard:
        async def begin(self) -> object:
            return real.GuardMarker(
                state=real.GuardState(state),
                attempt_number=1,
                input_digest="sha256:" + "0" * 64,
                embedding_usage={
                    "usage_basis": "PROVIDER_REPORTED",
                    "request_count": 1,
                    "cost_usd_microunits": 7,
                },
            )

        async def completed_raw(self) -> dict[str, object]:
            return {"immutable": True}

        async def rollback_pending(self, **_values: object) -> None:
            events.append("rollback_pending")

    delegate = SimpleNamespace(
        client=SimpleNamespace(embeddings=SimpleNamespace(), close=AsyncMock()),
        config=SimpleNamespace(embedding_model="model", embedding_dim=2),
    )
    runtime = SimpleNamespace(
        Graphiti=Graphiti,
        OpenAIEmbedder=lambda **_values: delegate,
        OpenAIEmbedderConfig=lambda **values: SimpleNamespace(**values),
        MeteredOpenAIEmbedder=real.MeteredOpenAIEmbedder,
        IdentityCrossEncoder=lambda: object(),
        EpisodeType=SimpleNamespace(text="text"),
        EpisodicNode=lambda **values: SimpleNamespace(**values),
        MutationGuard=lambda *_args, **_values: Guard(),
    )
    monkeypatch.setattr(real, "_load_graphiti", lambda: runtime)
    monkeypatch.setattr(
        real, "build_cli_llm_client", lambda: SimpleNamespace(invocations=[])
    )
    monkeypatch.setattr(
        real, "combined_temporal_pipeline_for", _provider_free_pipeline
    )

    def restore(raw: dict[str, object], _telemetry: object) -> None:
        assert raw == {"immutable": True}
        events.append("restore_complete")

    configuration, revision = _combined_runtime_inputs("Body", "episode-id")
    call = real._add_episode(
        api_key="key",
        password="password",
        body="Body",
        name="episode-id",
        episode_id="episode-id",
        reference_time=datetime(2026, 8, 20, tzinfo=UTC),
        telemetry=real._EpisodeTelemetry(),
        attempt_number=1,
        validate_result=lambda _result, _telemetry, _combined=None: {},
        restore_result=restore,
        configuration=configuration,
        revision=revision,
    )
    if state == "PENDING":
        with pytest.raises(real.AmbiguousEpisodeEffect, match="blocks another"):
            asyncio.run(call)
    else:
        asyncio.run(call)
    assert events == [expected_event, "close"]


@pytest.mark.parametrize(
    ("pipeline_rollback_completed", "guard_rollback_completed", "expected_events"),
    (
        (True, False, ["close"]),
        (False, False, ["telemetry", "rollback", "close"]),
    ),
)
def test_only_proven_pipeline_rollback_is_classified_complete(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_rollback_completed: bool,
    guard_rollback_completed: bool,
    expected_events: list[str],
) -> None:
    import newsroom.graphiti_adapter.real as real

    events: list[str] = []

    class Graphiti:
        def __init__(self, *_args: object, **values: object) -> None:
            self.driver = object()
            self.clients = SimpleNamespace(
                llm_client=values["llm_client"],
                embedder=values["embedder"],
            )

        async def close(self) -> None:
            events.append("close")

    class Guard:
        fenced_graph_mutation = _uncontended_generation_fence

        async def begin(self) -> GuardMarker:
            return GuardMarker(
                state=GuardState.CREATED,
                attempt_number=1,
                input_digest="sha256:" + "0" * 64,
            )

        async def record_pending_telemetry(self, **_values: object) -> None:
            events.append("telemetry")

        async def rollback_pending(self, **_values: object) -> bool:
            events.append("rollback")
            return guard_rollback_completed

    class Pipeline:
        complete_receipt = None
        complete_failure_receipt = None
        recovery_marker = None

        async def _prepare_attempt(self) -> None:
            return None

    async def created_episode(**_values: object) -> tuple[SimpleNamespace, str]:
        if _values.get("create") is False:
            return None
        return SimpleNamespace(uuid="episode-id"), "CREATED"

    async def rolled_back_extract(*_args: object, **_values: object) -> object:
        await _values["pipeline"].prepare_execution()
        raise real.CombinedTemporalPipelineError(
            "combined-temporal pipeline failed",
            graph_effect_attempted=True,
            rollback_completed=pipeline_rollback_completed,
        )

    delegate = SimpleNamespace(
        client=SimpleNamespace(embeddings=SimpleNamespace(), close=AsyncMock()),
        config=SimpleNamespace(embedding_model="model", embedding_dim=2),
    )
    runtime = SimpleNamespace(
        Graphiti=Graphiti,
        OpenAIEmbedder=lambda **_values: delegate,
        OpenAIEmbedderConfig=lambda **values: SimpleNamespace(**values),
        MeteredOpenAIEmbedder=real.MeteredOpenAIEmbedder,
        IdentityCrossEncoder=lambda: object(),
        EpisodeType=SimpleNamespace(text="text"),
        EpisodicNode=lambda **values: SimpleNamespace(**values),
        MutationGuard=lambda *_args, **_values: Guard(),
    )
    monkeypatch.setattr(real, "_load_graphiti", lambda: runtime)
    monkeypatch.setattr(
        real, "build_cli_llm_client", lambda: SimpleNamespace(invocations=[])
    )
    monkeypatch.setattr(real, "_ensure_episode", created_episode)
    monkeypatch.setattr(
        real, "combined_temporal_pipeline_for", lambda **_values: Pipeline()
    )
    monkeypatch.setattr(real, "extract_combined_temporal_async", rolled_back_extract)

    configuration, revision = _combined_runtime_inputs("Body", "episode-id")
    telemetry = real._EpisodeTelemetry()
    with pytest.raises(real.AmbiguousEpisodeEffect, match="rolled back"):
        asyncio.run(
            real._add_episode(
                api_key="key",
                password="password",
                body="Body",
                name="episode-id",
                episode_id="episode-id",
                reference_time=datetime(2026, 8, 20, tzinfo=UTC),
                telemetry=telemetry,
                attempt_number=1,
                validate_result=lambda _result, _telemetry, _combined=None: {},
                restore_result=lambda _raw, _telemetry: None,
                configuration=configuration,
                revision=revision,
            )
        )

    assert events == expected_events
    assert telemetry.recovery_classification is (
        GraphitiRecoveryClassification.ROLLED_BACK_AMBIGUOUS_EFFECT
        if pipeline_rollback_completed or guard_rollback_completed
        else None
    )


@pytest.mark.parametrize("slow_cleanup", (False, True))
def test_cancelled_episode_cleanup_is_ordered_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
    slow_cleanup: bool,
) -> None:
    import newsroom.graphiti_adapter.real as real

    events: list[str] = []

    class Graphiti:
        def __init__(self, *_args: object, **values: object) -> None:
            self.driver = object()
            self.clients = SimpleNamespace(
                llm_client=values["llm_client"],
                embedder=values["embedder"],
            )

        async def add_episode(self, **_values: object) -> SimpleNamespace:
            raise AssertionError("ordinary graphiti-core extraction must stay unused")

        async def close(self) -> None:
            events.append("close")
            if slow_cleanup:
                await asyncio.Event().wait()

    class Guard:
        fenced_graph_mutation = _uncontended_generation_fence

        async def begin(self) -> GuardMarker:
            return GuardMarker(
                state=GuardState.CREATED,
                attempt_number=1,
                input_digest="sha256:" + "0" * 64,
            )

        async def record_pending_telemetry(self, **_values: object) -> None:
            events.append("telemetry")

        async def rollback_pending(self, **_values: object) -> None:
            events.append("rollback")
            if slow_cleanup:
                await asyncio.Event().wait()

    async def created_episode(**_values: object) -> tuple[SimpleNamespace, str]:
        if _values.get("create") is False:
            return None
        return SimpleNamespace(uuid="episode-id"), "CREATED"

    async def close_embedding() -> None:
        events.append("embedding-close")
        if slow_cleanup:
            await asyncio.Event().wait()

    delegate = SimpleNamespace(
        client=SimpleNamespace(embeddings=SimpleNamespace(), close=close_embedding),
        config=SimpleNamespace(embedding_model="model", embedding_dim=2),
    )
    runtime = SimpleNamespace(
        Graphiti=Graphiti,
        OpenAIEmbedder=lambda **_values: delegate,
        OpenAIEmbedderConfig=lambda **values: SimpleNamespace(**values),
        MeteredOpenAIEmbedder=real.MeteredOpenAIEmbedder,
        IdentityCrossEncoder=lambda: object(),
        EpisodeType=SimpleNamespace(text="text"),
        EpisodicNode=lambda **values: SimpleNamespace(**values),
        MutationGuard=lambda *_args, **_values: Guard(),
    )
    monkeypatch.setattr(real, "_load_graphiti", lambda: runtime)
    monkeypatch.setattr(real, "_ensure_episode", created_episode)

    class LlmClient:
        invocations: list[dict[str, object]] = []

        async def _generate_response(self, *_args: object, **_values: object) -> object:
            events.append("provider-start")
            await asyncio.Event().wait()
            raise AssertionError("cancelled provider unexpectedly resumed")

    monkeypatch.setattr(real, "build_cli_llm_client", LlmClient)
    monkeypatch.setattr(
        real, "combined_temporal_pipeline_for", _provider_free_pipeline
    )
    if slow_cleanup:
        monkeypatch.setattr(real, "GRAPHITI_CLEANUP_TIMEOUT_MS", 10)

    started = time.monotonic()
    configuration, revision = _combined_runtime_inputs("Body", "episode-id")
    telemetry = real._EpisodeTelemetry()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(
            asyncio.wait_for(
                real._add_episode(
                    api_key="key",
                    password="password",
                    body="Body",
                    name="episode-id",
                    episode_id="episode-id",
                    reference_time=datetime(2026, 8, 20, tzinfo=UTC),
                    telemetry=telemetry,
                    attempt_number=1,
                    validate_result=lambda _result, _telemetry, _combined=None: {},
                    restore_result=lambda _raw, _telemetry: None,
                    configuration=configuration,
                    revision=revision,
                ),
                timeout=0.01,
            )
        )
    elapsed = time.monotonic() - started
    assert events == [
        "provider-start", "telemetry", "rollback", "embedding-close", "close"
    ]
    if slow_cleanup:
        assert elapsed < 0.2
        assert [
            item["phase"] for item in telemetry.timeout_diagnostics
        ] == ["ROLLBACK_CLEANUP", "CONNECTION_CLEANUP"]
        assert all(
            item["boundary"] == "CLEANUP_DEADLINE"
            and item["cause"] == "CLEANUP_DEADLINE_EXPIRED"
            and item["provider_cause"] == "UNOBSERVED"
            and item["deadline_at"].endswith("Z")
            for item in telemetry.timeout_diagnostics
        )
    else:
        assert telemetry.timeout_diagnostics == []


def _guard_fixture_session(driver, lock: asyncio.Lock | None = None):
    """Adapt query-shaped fixtures to the explicit generation-fence API."""

    class Result:
        def __init__(self, records: list[dict[str, object]]) -> None:
            self.records = records

        async def single(self, *, strict: bool = False):
            if strict:
                assert len(self.records) == 1
            return self.records[0] if self.records else None

        async def data(self) -> list[dict[str, object]]:
            return self.records

        async def __aiter__(self):
            for record in self.records:
                yield record

    class Transaction:
        held = False

        async def run(self, query: str, **params: object) -> Result:
            if lock is not None and "SET g.lock_tick" in query and not self.held:
                await lock.acquire()
                self.held = True
            execute = driver._execute_query if self.held else driver.execute_query
            records, _, _ = await execute(query, params=params, routing_="w")
            return Result(records)

        async def commit(self) -> None:
            if self.held:
                self.held = False
                lock.release()

        async def rollback(self) -> None:
            await self.commit()

    class Session:
        def __init__(self) -> None:
            self._config = SimpleNamespace(fetch_size=1000)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def begin_transaction(self) -> Transaction:
            return Transaction()

        async def execute_write(self, callback):
            return await callback(Transaction())

    return Session()


def _guard_fixture_owner(marker: dict[str, object]) -> dict[str, object]:
    return {
        "owner_marker_uuid": "episode-id",
        "snapshot_id": marker["snapshot_id"],
        "claim_token": marker["claim_token"],
    }


def _guard_fixture_is_owned(owner, marker, params) -> bool:
    return marker is not None and owner == {
        "owner_marker_uuid": params["episode_uuid"],
        "snapshot_id": params["snapshot_id"],
        "claim_token": params["claim_token"],
    } and all(marker.get(key) == params[key] for key in (
        "group_id", "input_digest", "snapshot_id", "claim_token",
    ))


def test_pending_guard_recovery_uses_retained_attempt_snapshot() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import Neo4jMutationGuard

    snapshot_ids: list[str] = []
    marker = {
        "state": "PENDING",
        "group_id": GRAPHITI_WORKSPACE_GROUP,
        "attempt_number": 1,
        "input_digest": "sha256:" + "0" * 64,
        "snapshot_id": "episode-id:1",
        "chat_invocations_json": "[]",
        "embedding_usage_json": "null",
        "claim_token": "expired",
    }
    owner = _guard_fixture_owner(marker)

    class Driver:
        def session(self):
            return _guard_fixture_session(self)

        async def execute_query(
            self,
            query: str,
            *,
            params: dict[str, object],
            routing_: str,
        ) -> tuple[list[dict[str, object]], None, None]:
            assert routing_ == "w"
            snapshot_id = params.get("snapshot_id")
            if (
                isinstance(snapshot_id, str)
                and "MERGE (m:NewsroomIngestMarker" not in query
            ):
                snapshot_ids.append(snapshot_id)
            if "CREATE CONSTRAINT" in query:
                return ([], None, None)
            if "MERGE (m:NewsroomIngestMarker" in query:
                assert owner == _guard_fixture_owner(marker)
                return (
                    [{"marker": marker, "claimed": False, "active": False}],
                    None,
                    None,
                )
            if "SET m.state = $state" in query:
                assert owner == _guard_fixture_owner(marker)
                assert marker["claim_token"] == params["retained_claim_token"]
                marker["state"] = params["state"]
                marker["claim_token"] = params["claim_token"]
                owner.update(_guard_fixture_owner(marker))
                return ([{"marker": marker}], None, None)
            if "RETURN properties(m) AS marker" in query:
                return ([{"marker": marker}], None, None)
            if "SET g.lock_tick" in query:
                assert _guard_fixture_is_owned(owner, marker, params)
            if "RETURN m.claim_token AS claim_token" in query:
                return ([{"claim_token": marker["claim_token"]}], None, None)
            if "SET m.state = 'ROLLING_BACK'" in query:
                marker["state"] = "ROLLING_BACK"
                return ([{"state": "ROLLING_BACK"}], None, None)
            if "SET m.state = 'RECOVERED_AMBIGUOUS'" in query:
                marker["state"] = "RECOVERED_AMBIGUOUS"
                owner.clear()
                return ([{"state": "RECOVERED_AMBIGUOUS"}], None, None)
            if "NewsroomSnapshot" in query and "DELETE s" in query:
                return ([{"deleted": 0}], None, None)
            if "RETURN count(s) AS snapshot_count" in query:
                return ([{"snapshot_count": 0}], None, None)
            return ([], None, None)

    guard = Neo4jMutationGuard(
        Driver(),
        group_id=GRAPHITI_WORKSPACE_GROUP,
        episode_uuid="episode-id",
        attempt_number=2,
        input_digest="sha256:" + "0" * 64,
    )
    retained = asyncio.run(guard.begin())
    assert retained.attempt_number == 1
    assert asyncio.run(
        guard.rollback_pending(
            chat_invocations=[],
            embedding_usage={"usage_basis": "NO_EMBEDDING_CALL"},
            reason="RECOVERED_PENDING_PROCESS_DEATH",
        )
    )
    assert not owner
    assert snapshot_ids
    assert set(snapshot_ids) == {"episode-id:1"}


@pytest.mark.parametrize("state", ["SNAPSHOTTING", "RECOVERED_AMBIGUOUS"])
def test_guard_retry_resets_snapshot_after_retained_attempt_cleanup(
    state: str,
) -> None:
    from newsroom.graphiti_adapter.neo4j_guard import Neo4jMutationGuard

    marker: dict[str, object] | None = {
        "state": state,
        "group_id": GRAPHITI_WORKSPACE_GROUP,
        "attempt_number": 1,
        "input_digest": "sha256:" + "0" * 64,
        "snapshot_id": "episode-id:1",
        "chat_invocations_json": "[]",
        "embedding_usage_json": "null",
        "claim_token": "expired",
    }
    owner = _guard_fixture_owner(marker) if state == "SNAPSHOTTING" else {}
    deleted_snapshots: list[str] = []
    created_snapshots: list[str] = []

    class Driver:
        def session(self):
            return _guard_fixture_session(self)

        async def execute_query(
            self,
            query: str,
            *,
            params: dict[str, object],
            routing_: str,
        ) -> tuple[list[dict[str, object]], None, None]:
            nonlocal marker
            assert routing_ == "w"
            if "RETURN count(*) AS snapshot_count" in query:
                assert params == {"group_id": GRAPHITI_WORKSPACE_GROUP}
                return ([{"snapshot_count": 0}], None, None)
            if "CREATE CONSTRAINT" in query:
                return ([], None, None)
            if "MERGE (m:NewsroomIngestMarker" in query:
                if marker is None:
                    marker = {
                        "state": "SNAPSHOTTING",
                        "group_id": params["group_id"],
                        "attempt_number": params["attempt_number"],
                        "input_digest": params["input_digest"],
                        "snapshot_id": params["snapshot_id"],
                        "chat_invocations_json": "[]",
                        "embedding_usage_json": "null",
                        "claim_token": params["claim_token"],
                    }
                    assert not owner
                    owner.update(_guard_fixture_owner(marker))
                    created_snapshots.append(str(params["snapshot_id"]))
                    return (
                        [{"marker": marker, "claimed": True, "active": False}],
                        None,
                        None,
                    )
                assert owner == _guard_fixture_owner(marker)
                return (
                    [{"marker": marker, "claimed": False, "active": False}],
                    None,
                    None,
                )
            if "SET m.state = $state" in query:
                assert marker is not None
                assert marker["claim_token"] == params["retained_claim_token"]
                assert owner == _guard_fixture_owner(marker) or (
                    not owner and marker["state"] == "RECOVERED_AMBIGUOUS"
                    and not params["require_expired"]
                )
                marker["state"] = params["state"]
                marker["claim_token"] = params["claim_token"]
                owner.update(_guard_fixture_owner(marker))
                return ([{"marker": marker}], None, None)
            if "SET g.lock_tick" in query:
                assert _guard_fixture_is_owned(owner, marker, params)
            if "RETURN m.claim_token AS claim_token" in query:
                return ([{"claim_token": marker["claim_token"]}], None, None)
            if "DELETE m" in query and "RETURN episode_uuid" in query:
                owner.clear()
                marker = None
                return ([{"episode_uuid": "episode-id"}], None, None)
            if "SET m.state = 'PENDING'" in query:
                assert marker is not None
                marker["state"] = "PENDING"
                return ([{"state": "PENDING"}], None, None)
            if "RETURN properties(m) AS marker" in query:
                return ([] if marker is None else [{"marker": marker}], None, None)
            if "NewsroomSnapshot" in query and "DELETE s" in query:
                deleted_snapshots.append(str(params["snapshot_id"]))
                return ([{"deleted": 0}], None, None)
            if "MATCH (m:NewsroomIngestMarker" in query and "DELETE m" in query:
                marker = None
            return ([], None, None)

    guard = Neo4jMutationGuard(
        Driver(),
        group_id=GRAPHITI_WORKSPACE_GROUP,
        episode_uuid="episode-id",
        attempt_number=2,
        input_digest="sha256:" + "0" * 64,
    )
    created = asyncio.run(guard.begin())
    assert created.attempt_number == 2
    assert created_snapshots == ["episode-id:2"]
    assert deleted_snapshots == ["episode-id:1"] * 2


def test_concurrent_guard_begin_has_one_atomic_marker_claim() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import (
        GuardError,
        GuardState,
        Neo4jMutationGuard,
    )

    marker: dict[str, object] | None = None
    owner: dict[str, object] = {}
    lock = asyncio.Lock()
    claims = 0
    constraints = 0

    class Driver:
        async def _execute_query(
            self,
            query: str,
            *,
            params: dict[str, object],
            routing_: str,
        ) -> tuple[list[dict[str, object]], None, None]:
            nonlocal claims, constraints, marker
            assert routing_ == "w"
            if "RETURN count(*) AS snapshot_count" in query:
                assert params == {"group_id": GRAPHITI_WORKSPACE_GROUP}
                return ([{"snapshot_count": 0}], None, None)
            if "CREATE CONSTRAINT" in query:
                constraints += 1
                assert "REQUIRE m.episode_uuid IS UNIQUE" in query
                return ([], None, None)
            if "MERGE (m:NewsroomIngestMarker" in query:
                if marker is None:
                    claims += 1
                    marker = {
                        "state": "SNAPSHOTTING",
                        "group_id": params["group_id"],
                        "attempt_number": params["attempt_number"],
                        "input_digest": params["input_digest"],
                        "snapshot_id": params["snapshot_id"],
                        "chat_invocations_json": "[]",
                        "embedding_usage_json": "null",
                        "claim_token": params["claim_token"],
                    }
                    assert not owner
                    owner.update(_guard_fixture_owner(marker))
                    return (
                        [{"marker": marker, "claimed": True, "active": False}],
                        None,
                        None,
                    )
                assert owner == _guard_fixture_owner(marker)
                return (
                    [{"marker": marker, "claimed": False, "active": True}],
                    None,
                    None,
                )
            if "RETURN properties(m) AS marker" in query:
                return ([] if marker is None else [{"marker": dict(marker)}], None, None)
            if "SET g.lock_tick" in query:
                if not _guard_fixture_is_owned(owner, marker, params):
                    return ([], None, None)
            if "RETURN m.claim_token AS claim_token" in query:
                return ([{"claim_token": marker["claim_token"]}], None, None)
            if "NewsroomSnapshot" in query and "DELETE s" in query:
                return ([{"deleted": 0}], None, None)
            if "SET m.state = 'PENDING'" in query:
                assert marker is not None
                marker["state"] = "PENDING"
                return ([{"state": "PENDING"}], None, None)
            return ([], None, None)

        async def execute_query(self, query: str, *, params, routing_):
            if "SET g.lock_tick" in query:
                async with lock:
                    if "MERGE (m:NewsroomIngestMarker" in query:
                        await asyncio.sleep(0.01)
                    return await self._execute_query(query, params=params, routing_=routing_)
            return await self._execute_query(query, params=params, routing_=routing_)

        def session(self):
            return _guard_fixture_session(self, lock)

    driver = Driver()
    guards = [
        Neo4jMutationGuard(
            driver,
            group_id=GRAPHITI_WORKSPACE_GROUP,
            episode_uuid="episode-id",
            attempt_number=1,
            input_digest="sha256:" + "0" * 64,
        )
        for _ in range(2)
    ]

    async def begin_both() -> list[object]:
        return list(
            await asyncio.gather(
                *(guard.begin() for guard in guards),
                return_exceptions=True,
            )
        )

    results = asyncio.run(begin_both())

    assert claims == 1
    assert constraints == 0
    assert sum(
        getattr(result, "state", None) is GuardState.CREATED for result in results
    ) == 1
    assert sum(isinstance(result, GuardError) for result in results) == 1


def test_guard_streams_preexisting_validation_in_managed_write_transactions() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import Neo4jMutationGuard

    events: list[str] = []
    node = {
        "snapshot_identity": "snapshot-node-1",
        "snapshot": {
            "uuid": "node-1", "embedding": [1.0, 2.0],
            "_newsroom_source_labels": ["Entity"],
        },
        "current": {"uuid": "node-1", "embedding": [1.0, 2.0]},
        "current_labels": ["Entity"],
    }
    relationship = {
        "snapshot_identity": "snapshot-relationship-1",
        "snapshot": {
            "uuid": "relationship-1", "weight": 1.0,
            "_newsroom_source_uuid": "node-1",
            "_newsroom_target_uuid": "node-2",
            "_newsroom_relationship_type": "RELATES_TO",
        },
        "current": {"uuid": "relationship-1", "weight": 1.0},
        "source_uuid": "node-1", "target_uuid": "node-2",
        "relationship_type": "RELATES_TO",
    }

    class Result:
        def __init__(self, records: list[dict[str, object]]) -> None:
            self.records = records

        def __iter__(self):
            raise AssertionError("guard validation must not eagerly consume results")

        async def single(self, *, strict: bool) -> dict[str, object]:
            assert strict and len(self.records) == 1
            events.append("count")
            return self.records[0]

        async def __aiter__(self):
            for record in self.records:
                events.append("record")
                yield record

    class Transaction:
        async def run(self, query: str, **params: object) -> Result:
            assert params == {"snapshot_id": "episode-id:1"}
            if "RETURN count(s) AS snapshot_count" in query:
                return Result([{"snapshot_count": 1}])
            return Result([node] if "labels(n) AS current_labels" in query else [relationship])

    class Session:
        async def __aenter__(self):
            events.append("session-enter")
            return self

        async def __aexit__(self, *_args: object) -> None:
            events.append("session-exit")

        async def execute_write(self, callback):
            events.append("managed-write")
            await callback(Transaction())

    class Driver:
        def session(self) -> Session:
            return Session()

        async def execute_query(self, *_args: object, **_kwargs: object):
            raise AssertionError("guard validation must use streaming sessions")

    guard = Neo4jMutationGuard(
        Driver(), group_id="group-id", episode_uuid="episode-id",
        attempt_number=1, input_digest="sha256:" + "0" * 64,
    )
    asyncio.run(guard.assert_preexisting_unchanged())
    assert events == [
        "session-enter", "managed-write", "count", "record", "session-exit",
        "session-enter", "managed-write", "count", "record", "session-exit",
    ]


@pytest.mark.parametrize("failure", ["late-corruption", "cancellation"])
def test_guard_streaming_validation_closes_session_after_failure(failure: str) -> None:
    from newsroom.graphiti_adapter.neo4j_guard import GuardError, Neo4jMutationGuard

    exit_error: type[BaseException] | None = None

    class Result:
        async def single(self, *, strict: bool) -> dict[str, object]:
            assert strict
            return {"snapshot_count": 2}

        async def __aiter__(self):
            yield {
                "snapshot_identity": "snapshot-node-1",
                "snapshot": {"uuid": "node-1", "_newsroom_source_labels": []},
                "current": {"uuid": "node-1"}, "current_labels": [],
            }
            if failure == "cancellation":
                raise asyncio.CancelledError
            yield {
                "snapshot_identity": "snapshot-node-2",
                "snapshot": {"uuid": "node-2", "_newsroom_source_labels": []},
                "current": {"uuid": "changed"}, "current_labels": [],
            }

    class Transaction:
        async def run(self, *_args: object, **_kwargs: object) -> Result:
            return Result()

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, error, *_args: object) -> None:
            nonlocal exit_error
            exit_error = error

        async def execute_write(self, callback):
            await callback(Transaction())

    guard = Neo4jMutationGuard(
        SimpleNamespace(session=lambda: Session()), group_id="group-id",
        episode_uuid="episode-id", attempt_number=1,
        input_digest="sha256:" + "0" * 64,
    )
    error = asyncio.CancelledError if failure == "cancellation" else GuardError
    with pytest.raises(error, match=None if failure == "cancellation" else "node changed"):
        asyncio.run(guard.assert_preexisting_unchanged())
    assert exit_error is error


def test_guard_schema_bootstrap_is_explicit_and_separate_from_begin() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import Neo4jMutationGuard

    queries: list[tuple[str, dict[str, object]]] = []

    class Driver:
        async def execute_query(
            self,
            query: str,
            *,
            params: dict[str, object],
            routing_: str,
        ) -> tuple[list[dict[str, object]], None, None]:
            assert routing_ == "w"
            queries.append((query, params))
            return [], None, None

    asyncio.run(Neo4jMutationGuard.bootstrap_schema(Driver()))

    assert len(queries) == 3
    assert "CREATE CONSTRAINT newsroom_ingest_marker_episode" in queries[0][0]
    assert "CREATE INDEX newsroom_snapshot_node_identity" in queries[1][0]
    assert "CREATE INDEX newsroom_snapshot_relationship_identity" in queries[2][0]
    assert all(params == {} for _, params in queries)


def test_real_runtime_bootstraps_guard_schema_once_before_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real

    calls = 0

    class Driver:
        async def execute_query(
            self,
            query: str,
            *,
            params: dict[str, object],
            routing_: str,
        ) -> tuple[list[dict[str, object]], None, None]:
            nonlocal calls
            assert "CREATE CONSTRAINT" in query or "CREATE INDEX" in query
            assert params == {}
            assert routing_ == "w"
            calls += 1
            return [], None, None

    monkeypatch.setattr(real, "_GRAPHITI_SCHEMA_BOOTSTRAPPED", False)

    async def bootstrap_twice() -> None:
        driver = Driver()
        await real._bootstrap_graphiti_schema(driver)
        await real._bootstrap_graphiti_schema(driver)

    asyncio.run(bootstrap_twice())

    assert calls == 3


def test_guard_rejects_telemetry_after_claim_takeover() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import GuardError, Neo4jMutationGuard

    class Driver:
        async def execute_query(
            self,
            _query: str,
            *,
            params: dict[str, object],
            routing_: str,
        ) -> tuple[list[dict[str, object]], None, None]:
            del params
            assert routing_ == "w"
            return [], None, None

    guard = Neo4jMutationGuard(
        Driver(),
        group_id=GRAPHITI_WORKSPACE_GROUP,
        episode_uuid="episode-id",
        attempt_number=1,
        input_digest="sha256:" + "0" * 64,
    )
    guard._claim_token = "stale"  # type: ignore[attr-defined]

    with pytest.raises(GuardError, match="lost its pending claim"):
        asyncio.run(
            guard.record_pending_telemetry(
                chat_invocations=[],
                embedding_usage={"usage_basis": "NO_EMBEDDING_CALL"},
            )
        )


def test_guard_holds_marker_lock_across_external_graph_mutation() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import Neo4jMutationGuard

    lock = asyncio.Lock()
    events: list[str] = []

    class Result:
        async def single(self) -> dict[str, object]:
            return {"claim_token": "owner"}

    class Transaction:
        async def run(self, _query: str, **_params: object) -> Result:
            if not lock.locked():
                await lock.acquire()
                events.append("fenced")
            return Result()

        async def commit(self) -> None:
            events.append("commit")
            lock.release()

        async def rollback(self) -> None:
            if lock.locked():
                lock.release()

    class Session:
        async def __aenter__(self) -> Session:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def begin_transaction(self) -> Transaction:
            return Transaction()

    class Driver:
        def session(self) -> Session:
            return Session()

    guard = Neo4jMutationGuard(
        Driver(),
        group_id=GRAPHITI_WORKSPACE_GROUP,
        episode_uuid="episode-id",
        attempt_number=1,
        input_digest="sha256:" + "0" * 64,
    )
    guard._claim_token = "owner"  # type: ignore[attr-defined]

    async def prove_lock() -> None:
        async with guard.fenced_graph_mutation():
            contender = asyncio.create_task(lock.acquire())
            await asyncio.sleep(0)
            assert not contender.done()
            events.append("persist")
        await contender
        events.append("takeover")
        lock.release()

    asyncio.run(prove_lock())

    assert events == ["fenced", "persist", "commit", "takeover"]


def test_concurrent_expired_marker_takeover_is_fenced() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import (
        GuardError,
        GuardState,
        Neo4jMutationGuard,
    )

    marker: dict[str, object] | None = {
        "state": "SNAPSHOTTING",
        "group_id": GRAPHITI_WORKSPACE_GROUP,
        "attempt_number": 1,
        "input_digest": "sha256:" + "0" * 64,
        "snapshot_id": "episode-id:1",
        "chat_invocations_json": "[]",
        "embedding_usage_json": "null",
        "claim_token": "expired",
        "active": False,
    }
    owner = _guard_fixture_owner(marker)
    lock = asyncio.Lock()
    claims = 0

    class Driver:
        async def _execute_query(
            self,
            query: str,
            *,
            params: dict[str, object],
            routing_: str,
        ) -> tuple[list[dict[str, object]], None, None]:
            nonlocal claims, marker
            assert routing_ == "w"
            if "RETURN count(*) AS snapshot_count" in query:
                assert params == {"group_id": GRAPHITI_WORKSPACE_GROUP}
                return ([{"snapshot_count": 0}], None, None)
            if "CREATE CONSTRAINT" in query:
                return ([], None, None)
            if "MERGE (m:NewsroomIngestMarker" in query:
                if marker is None:
                    claims += 1
                    marker = {
                        "state": "SNAPSHOTTING",
                        "group_id": params["group_id"],
                        "attempt_number": params["attempt_number"],
                        "input_digest": params["input_digest"],
                        "snapshot_id": params["snapshot_id"],
                        "chat_invocations_json": "[]",
                        "embedding_usage_json": "null",
                        "claim_token": params["claim_token"],
                        "active": True,
                    }
                    assert not owner
                    owner.update(_guard_fixture_owner(marker))
                    return (
                        [{"marker": marker, "claimed": True, "active": False}],
                        None,
                        None,
                    )
                assert owner == _guard_fixture_owner(marker)
                return (
                    [
                        {
                            "marker": marker,
                            "claimed": False,
                            "active": marker["active"],
                        }
                    ],
                    None,
                    None,
                )
            if "SET m.state = $state" in query:
                if (
                    marker is None
                    or marker["state"] != params["retained_state"]
                    or marker["claim_token"] != params["retained_claim_token"]
                    or marker["active"]
                ):
                    return ([], None, None)
                assert owner == _guard_fixture_owner(marker)
                marker["state"] = params["state"]
                marker["claim_token"] = params["claim_token"]
                marker["active"] = True
                owner.update(_guard_fixture_owner(marker))
                return ([{"marker": marker}], None, None)
            if "DELETE m" in query and "RETURN episode_uuid" in query:
                if marker is None or marker["claim_token"] != params["claim_token"]:
                    return ([], None, None)
                assert _guard_fixture_is_owned(owner, marker, params)
                owner.clear()
                marker = None
                return ([{"episode_uuid": "episode-id"}], None, None)
            if "RETURN properties(m) AS marker" in query:
                return ([] if marker is None else [{"marker": dict(marker)}], None, None)
            if "SET g.lock_tick" in query:
                if not _guard_fixture_is_owned(owner, marker, params):
                    return ([], None, None)
            if "RETURN m.claim_token AS claim_token" in query:
                return ([{"claim_token": marker["claim_token"]}], None, None)
            if "NewsroomSnapshot" in query and "DELETE s" in query:
                return ([{"deleted": 0}], None, None)
            if "SET m.state = 'PENDING'" in query:
                assert marker is not None
                if marker["claim_token"] != params["claim_token"]:
                    return ([], None, None)
                marker["state"] = "PENDING"
                return ([{"state": "PENDING"}], None, None)
            return ([], None, None)

        async def execute_query(self, query: str, *, params, routing_):
            if "SET g.lock_tick" in query:
                async with lock:
                    return await self._execute_query(query, params=params, routing_=routing_)
            return await self._execute_query(query, params=params, routing_=routing_)

        def session(self):
            return _guard_fixture_session(self, lock)

    driver = Driver()
    guards = [
        Neo4jMutationGuard(
            driver,
            group_id=GRAPHITI_WORKSPACE_GROUP,
            episode_uuid="episode-id",
            attempt_number=2,
            input_digest="sha256:" + "0" * 64,
        )
        for _ in range(2)
    ]

    async def begin_both() -> list[object]:
        return list(
            await asyncio.gather(
                *(guard.begin() for guard in guards), return_exceptions=True
            )
        )

    results = asyncio.run(begin_both())

    assert claims == 1
    assert sum(
        getattr(result, "state", None) is GuardState.CREATED for result in results
    ) == 1
    assert sum(isinstance(result, GuardError) for result in results) == 1


def test_guard_rejects_mismatched_retained_snapshot_identity() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import GuardError, Neo4jMutationGuard

    marker = {
        "state": "PENDING",
        "group_id": GRAPHITI_WORKSPACE_GROUP,
        "attempt_number": 1,
        "input_digest": "sha256:" + "0" * 64,
        "snapshot_id": "episode-id:2",
        "chat_invocations_json": "[]",
        "embedding_usage_json": "null",
        "claim_token": "expired",
    }
    owner = _guard_fixture_owner(marker)

    class Driver:
        async def execute_query(
            self,
            query: str,
            *,
            params: dict[str, object],
            routing_: str,
        ) -> tuple[list[dict[str, object]], None, None]:
            assert routing_ == "w"
            if "CREATE CONSTRAINT" in query:
                return ([], None, None)
            if "RETURN properties(m) AS marker" in query and "MERGE (m:" not in query:
                return ([{"marker": marker}], None, None)
            assert "MERGE (m:NewsroomIngestMarker" in query
            assert owner == _guard_fixture_owner(marker)
            assert params["episode_uuid"] == "episode-id"
            return (
                [{"marker": marker, "claimed": False, "active": False}],
                None,
                None,
            )

    guard = Neo4jMutationGuard(
        Driver(),
        group_id=GRAPHITI_WORKSPACE_GROUP,
        episode_uuid="episode-id",
        attempt_number=2,
        input_digest="sha256:" + "0" * 64,
    )
    with pytest.raises(GuardError, match="snapshot identity"):
        asyncio.run(guard.begin())


def test_complete_guard_recovery_cleans_crash_window_snapshot() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import Neo4jMutationGuard

    deleted_snapshots: list[str] = []
    marker = {
        "state": "COMPLETE",
        "group_id": GRAPHITI_WORKSPACE_GROUP,
        "attempt_number": 1,
        "input_digest": "sha256:" + "0" * 64,
        "snapshot_id": "episode-id:1",
        "chat_invocations_json": "[]",
        "embedding_usage_json": "null",
    }

    class Driver:
        async def execute_query(
            self,
            query: str,
            *,
            params: dict[str, object],
            routing_: str,
        ) -> tuple[list[dict[str, object]], None, None]:
            assert routing_ == "w"
            if "CREATE CONSTRAINT" in query:
                return ([], None, None)
            if "MERGE (m:NewsroomIngestMarker" in query:
                return (
                    [{"marker": marker, "claimed": False, "active": False}],
                    None,
                    None,
                )
            if "RETURN properties(m) AS marker" in query:
                return ([{"marker": marker}], None, None)
            if "NewsroomSnapshot" in query and "DELETE s" in query:
                deleted_snapshots.append(str(params["snapshot_id"]))
                return ([{"deleted": 0}], None, None)
            return ([], None, None)

    guard = Neo4jMutationGuard(
        Driver(),
        group_id=GRAPHITI_WORKSPACE_GROUP,
        episode_uuid="episode-id",
        attempt_number=2,
        input_digest="sha256:" + "0" * 64,
    )
    retained = asyncio.run(guard.begin())
    assert retained.state.value == "COMPLETE"
    assert deleted_snapshots == ["episode-id:1"] * 2


def test_complete_guard_recovery_requires_byte_exact_canonical_snapshot() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import (
        GuardError,
        Neo4jMutationGuard,
    )

    raw = {"provider_attempt_number": 1, "result": "fixed"}
    raw_json = canonical_json_bytes(raw).decode("utf-8")
    marker = {
        "state": "COMPLETE",
        "group_id": GRAPHITI_WORKSPACE_GROUP,
        "input_digest": "sha256:" + "0" * 64,
        "attempt_number": 1,
        "snapshot_id": "episode-id:1",
        "chat_invocations_json": "[]",
        "embedding_usage_json": "null",
        "validated_raw_json": raw_json,
        "validated_raw_digest": digest_bytes(raw_json.encode("utf-8")),
    }

    class Driver:
        async def execute_query(
            self,
            _query: str,
            *,
            params: dict[str, object],
            routing_: str,
        ) -> tuple[list[dict[str, object]], None, None]:
            assert params == {"episode_uuid": "episode-id"}
            assert routing_ == "w"
            return ([{"marker": marker}], None, None)

    guard = Neo4jMutationGuard(
        Driver(),
        group_id=GRAPHITI_WORKSPACE_GROUP,
        episode_uuid="episode-id",
        attempt_number=1,
        input_digest="sha256:" + "0" * 64,
    )
    assert asyncio.run(guard.completed_raw()) == raw
    marker["validated_raw_json"] = '{"provider_attempt_number": 1, "result": "fixed"}'
    with pytest.raises(GuardError, match="digest differs"):
        asyncio.run(guard.completed_raw())


def test_completed_guard_probe_is_read_only_when_marker_is_absent() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import Neo4jMutationGuard

    class Driver:
        async def execute_query(
            self,
            query: str,
            *,
            params: dict[str, object],
            routing_: str,
        ) -> tuple[list[dict[str, object]], None, None]:
            assert "RETURN properties(m) AS marker" in query
            assert params == {"episode_uuid": "episode-id"}
            assert routing_ == "w"
            return ([], None, None)

    guard = Neo4jMutationGuard(
        Driver(),
        group_id=GRAPHITI_WORKSPACE_GROUP,
        episode_uuid="episode-id",
        attempt_number=1,
        input_digest="sha256:" + "0" * 64,
    )
    assert asyncio.run(guard.completed_raw_or_none()) is None


def test_recovered_guard_probe_validates_exact_attempt_without_mutation() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import (
        GuardError,
        GuardState,
        Neo4jMutationGuard,
    )

    marker = {
        "state": "RECOVERED_AMBIGUOUS",
        "group_id": GRAPHITI_WORKSPACE_GROUP,
        "attempt_number": 3,
        "input_digest": "sha256:" + "0" * 64,
        "snapshot_id": "episode-id:3",
        "chat_invocations_json": "[]",
        "embedding_usage_json": "null",
    }
    queries: list[str] = []

    class Driver:
        async def execute_query(
            self,
            query: str,
            *,
            params: dict[str, object],
            routing_: str,
        ) -> tuple[list[dict[str, object]], None, None]:
            queries.append(query)
            assert params == {"episode_uuid": "episode-id:attempt:3"}
            assert routing_ == "w"
            return ([{"marker": marker}], None, None)

    guard = Neo4jMutationGuard(
        Driver(),
        group_id=GRAPHITI_WORKSPACE_GROUP,
        episode_uuid="episode-id",
        marker_episode_uuid="episode-id:attempt:3",
        attempt_number=3,
        input_digest="sha256:" + "0" * 64,
    )
    retained = asyncio.run(guard.recovered_ambiguous_marker_or_none())

    assert retained is not None
    assert retained.state is GuardState.RECOVERED_AMBIGUOUS
    assert len(queries) == 1
    assert "MATCH (m:NewsroomIngestMarker" in queries[0]
    assert "SET " not in queries[0]

    marker["attempt_number"] = 2
    with pytest.raises(GuardError, match="attempt marker identity differs"):
        asyncio.run(guard.recovered_ambiguous_marker_or_none())


def test_guard_completion_checks_the_committed_transition() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import Neo4jMutationGuard

    queries: list[str] = []
    marker = {
        "state": "PENDING", "group_id": GRAPHITI_WORKSPACE_GROUP,
        "input_digest": "sha256:" + "0" * 64,
        "snapshot_id": "episode-id:1", "claim_token": "owner",
    }
    owner = _guard_fixture_owner(marker)

    class Driver:
        def session(self):
            return _guard_fixture_session(self)

        async def execute_query(
            self,
            query: str,
            *,
            params: dict[str, object],
            routing_: str,
        ) -> tuple[list[dict[str, object]], None, None]:
            assert routing_ == "w"
            queries.append(query)
            if "SET g.lock_tick" in query:
                assert _guard_fixture_is_owned(owner, marker, params)
                assert marker["state"] == "PENDING"
            if "RETURN m.claim_token AS claim_token" in query:
                return ([{"claim_token": marker["claim_token"]}], None, None)
            if "SET m.state = 'COMPLETE'" in query:
                marker["state"] = "COMPLETE"
                owner.clear()
                return ([{"state": "COMPLETE"}], None, None)
            if "NewsroomSnapshot" in query and "DELETE s" in query:
                assert marker["state"] == "COMPLETE" and not owner
                return ([{"deleted": 0}], None, None)
            return ([], None, None)

    guard = Neo4jMutationGuard(
        Driver(),
        group_id=GRAPHITI_WORKSPACE_GROUP,
        episode_uuid="episode-id",
        attempt_number=1,
        input_digest="sha256:" + "0" * 64,
    )
    guard._claim_token = "owner"  # type: ignore[attr-defined]
    asyncio.run(guard.complete({"provider_attempt_number": 1}))
    assert any("SET m.state = 'COMPLETE'" in query for query in queries)
    assert any("NewsroomSnapshot" in query and "DELETE s" in query for query in queries)


def test_complete_marker_blocks_cancellation_rollback_deletion() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import Neo4jMutationGuard

    queries: list[str] = []
    marker = {
        "state": "COMPLETE", "group_id": GRAPHITI_WORKSPACE_GROUP,
        "input_digest": "sha256:" + "0" * 64, "attempt_number": 1,
        "snapshot_id": "episode-id:1", "claim_token": "owner",
        "chat_invocations_json": "[]", "embedding_usage_json": "null",
    }

    class Driver:
        async def execute_query(
            self,
            query: str,
            *,
            params: dict[str, object],
            routing_: str,
        ) -> tuple[list[dict[str, object]], None, None]:
            del params, routing_
            queries.append(query)
            if "SET m.state = 'ROLLING_BACK'" in query:
                return [], None, None
            if "RETURN properties(m) AS marker" in query:
                return [{"marker": marker}], None, None
            if "NewsroomSnapshot" in query and "DELETE s" in query:
                return ([{"deleted": 0}], None, None)
            return [], None, None

    guard = Neo4jMutationGuard(
        Driver(),
        group_id=GRAPHITI_WORKSPACE_GROUP,
        episode_uuid="episode-id",
        attempt_number=1,
        input_digest="sha256:" + "0" * 64,
    )
    rolled_back = asyncio.run(
        guard.rollback_pending(
            chat_invocations=[],
            embedding_usage={"usage_basis": "NO_EMBEDDING_CALL"},
            reason="CANCELLED",
        )
    )
    assert rolled_back is False
    assert not any("DELETE r" in query or "DETACH DELETE n" in query for query in queries)


def test_immutable_completion_snapshot_restores_without_graph_rehydration(
    tmp_path: Path,
) -> None:
    from newsroom.graphiti_adapter.combined_temporal_contract import SourceRevisionInput
    from newsroom.graphiti_adapter.combined_temporal_evidence import segment_source
    from newsroom.graphiti_adapter.combined_temporal_extraction import (
        CombinedTemporalTransportResult,
        extract_combined_temporal,
    )
    from newsroom.graphiti_adapter.evaluation_packet import GRAPHITI_CORE_RELEASE
    from newsroom.graphiti_adapter.real import _EpisodeTelemetry, _raw_receipt
    from newsroom.graphiti_adapter.result_snapshot import restore_validated_snapshot
    from newsroom.graphiti_adapter.temporal_vocabulary import TEMPORAL_POLICY_DIGEST_V2

    instant = UtcTimestamp(datetime(2026, 8, 20, tzinfo=UTC))
    attempt = replace(
        _real_attempt(tmp_path),
        reference_time=instant,
        temporal_basis=TemporalBasis.SOURCE_PUBLISHED,
    )
    captured: dict[str, object] = {}

    class Transport:
        def generate_response(self, **kwargs: object) -> CombinedTemporalTransportResult:
            del kwargs
            return CombinedTemporalTransportResult(
                raw={"entities": [], "facts": []},
                framework_version=GRAPHITI_CORE_RELEASE,
                model_version="composer-2.5",
                token_usage={"basis": "PROVIDER_REPORTED", "output_tokens": 1},
                provider_cost=None,
            )

    class Pipeline:
        def prepare_attempt(self) -> None:
            return None

        def complete_failure(self, terminal: dict[str, object]) -> dict[str, object]:
            return dict(terminal)

        def execute(self, **kwargs: object) -> SimpleNamespace:
            receipt = dict(kwargs["receipt"])
            captured["combined"] = receipt
            return SimpleNamespace(
                nodes=(),
                edges=(),
                guarded_edges=(),
                node_resolutions=(),
                graph_effect_attempted=False,
                embedding_skipped=True,
                journal_skipped=True,
                rollback_skipped=True,
                completed_receipt=receipt,
            )

    revision = SourceRevisionInput(
        body="Alice met Bob on 2026-08-20.",
        revision_id="rev",
        source_id="src",
        item_key="item",
        representation_digest="sha256:" + "ab" * 32,
        published_at=instant.to_text(),
        updated_at=None,
        observed_at=instant.to_text(),
        ingested_at=instant.to_text(),
    )
    extract_combined_temporal(revision, transport=Transport(), pipeline=Pipeline())
    combined = dict(captured["combined"])  # type: ignore[arg-type]
    assert combined.get("temporal_policy_digest") == TEMPORAL_POLICY_DIGEST_V2
    raw = _raw_receipt(
        attempt,
        started_at=instant,
        telemetry=_EpisodeTelemetry(provider_attempt_number=1),
        result=None,
        proposals=(),
    )
    raw.pop("raw_output_digest", None)
    raw["combined_temporal_receipt"] = combined
    raw["raw_output_digest"] = digest_bytes(canonical_json_bytes(raw))
    restored = restore_validated_snapshot(raw=raw, attempt=attempt)
    assert restored.produced.raw_output_value == raw
    assert restored.provider_attempt_number == 1
    assert (
        restored.recovery_classification
        is GraphitiRecoveryClassification.RECOVERED_IMMUTABLE_COMPLETE
    )

    corrupted = dict(raw)
    corrupted["framework"] = "graphiti-core==mutated"
    unsigned = dict(corrupted)
    unsigned.pop("raw_output_digest")
    corrupted["raw_output_digest"] = digest_bytes(canonical_json_bytes(unsigned))
    with pytest.raises(GraphitiAdapterContractError, match="immutable attempt"):
        restore_validated_snapshot(raw=corrupted, attempt=attempt)


def test_completed_pipeline_failure_snapshot_restores_as_retryable(
    tmp_path: Path,
) -> None:
    from newsroom.graphiti_adapter.cli_process import process_exit_diagnostic
    from newsroom.graphiti_adapter.real import _EpisodeTelemetry, _raw_receipt
    from newsroom.graphiti_adapter.result_snapshot import restore_validated_snapshot

    instant = UtcTimestamp(datetime(2026, 8, 20, tzinfo=UTC))
    first = _real_attempt(tmp_path)
    attempt = replace(
        first,
        attempt_id=GraphitiAttemptId.parse(
            "00000000-0000-4000-8000-000000004935"
        ),
        attempt_number=2,
        expected_previous_attempt_id=first.attempt_id,
        idempotency_key="evaluation-real-attempt-v2",
        reference_time=instant,
        temporal_basis=TemporalBasis.SOURCE_PUBLISHED,
    )
    diagnostic = process_exit_diagnostic(
        returncode=1,
        cause="UPSTREAM_SERVER",
        stdout="",
        stderr="fixture provider content",
    )
    raw = _raw_receipt(
        attempt,
        started_at=instant,
        telemetry=_EpisodeTelemetry(
            provider_attempt_number=2,
            chat_invocations=[{"process_exit_diagnostic": diagnostic}],
        ),
        result=None,
        proposals=(),
    )
    raw.pop("raw_output_digest")
    raw["combined_temporal_failure_code"] = "PIPELINE_FAILED"
    raw["combined_temporal_receipt"] = {"failure_code": "PIPELINE_FAILED"}
    raw["raw_output_digest"] = digest_bytes(canonical_json_bytes(raw))

    restored = restore_validated_snapshot(raw=raw, attempt=attempt)

    assert restored.produced.outcome is ExtractionOutcome.RETRYABLE_FAILURE
    assert (
        restored.produced.failure_code
        is ExtractionFailureCode.PRODUCER_INTERNAL_ERROR
    )
    assert restored.produced.validation is None
    assert restored.produced.raw_output_value is None
    assert restored.produced.attempt_receipt_value == raw
    assert restored.provider_attempt_number == 2
    assert restored.chat_invocations[0]["process_exit_diagnostic"] == diagnostic

    malformed = copy.deepcopy(raw)
    malformed["chat_invocations"][0]["process_exit_diagnostic"]["stderr"] = (
        "SECRET"
    )
    malformed.pop("raw_output_digest")
    malformed["raw_output_digest"] = digest_bytes(canonical_json_bytes(malformed))
    with pytest.raises(
        GraphitiAdapterContractError,
        match="process-exit diagnostic is malformed",
    ):
        restore_validated_snapshot(raw=malformed, attempt=attempt)


def test_zero_dispatch_failure_opens_attempt_marker_and_replays_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real
    from newsroom.graphiti_adapter.neo4j_guard import GuardMarker, GuardState

    markers: dict[str, dict[str, object]] = {
        "episode-id": {"failure": "attempt-1"}
    }
    provider_calls = 0
    restored: list[dict[str, object]] = []

    async def marker_exists(guard: object) -> bool:
        return guard._marker_episode_uuid in markers

    async def completed_raw_or_none(
        guard: object,
    ) -> dict[str, object] | None:
        retained = markers.get(guard._marker_episode_uuid)
        if retained is not None and retained.get("state") in {
            "PENDING",
            "RECOVERED_AMBIGUOUS",
        }:
            return None
        return retained

    async def recovered_ambiguous_marker_or_none(
        guard: object,
    ) -> GuardMarker | None:
        retained = markers.get(guard._marker_episode_uuid)
        if retained is None or retained.get("state") != "RECOVERED_AMBIGUOUS":
            return None
        return GuardMarker(
            state=GuardState.RECOVERED_AMBIGUOUS,
            attempt_number=guard._attempt_number,
            input_digest=guard._input_digest,
        )

    async def begin(guard: object) -> GuardMarker:
        retained = markers.get(guard._marker_episode_uuid)
        if retained is not None:
            if retained.get("state") == "PENDING":
                return GuardMarker(
                    state=GuardState.PENDING,
                    attempt_number=guard._attempt_number,
                    input_digest=guard._input_digest,
                )
            if retained.get("state") == "RECOVERED_AMBIGUOUS":
                return GuardMarker(
                    state=GuardState.RECOVERED_AMBIGUOUS,
                    attempt_number=guard._attempt_number,
                    input_digest=guard._input_digest,
                )
            return GuardMarker(
                state=GuardState.COMPLETE,
                attempt_number=guard._attempt_number,
                input_digest=guard._input_digest,
            )
        markers[guard._marker_episode_uuid] = {"pending": True}
        return GuardMarker(
            state=GuardState.CREATED,
            attempt_number=guard._attempt_number,
            input_digest=guard._input_digest,
        )

    async def complete(guard: object, raw: dict[str, object]) -> None:
        markers[guard._marker_episode_uuid] = dict(raw)

    monkeypatch.setattr(
        real.Neo4jMutationGuard, "fenced_graph_mutation", _uncontended_generation_fence
    )
    monkeypatch.setattr(real.Neo4jMutationGuard, "marker_exists", marker_exists)
    monkeypatch.setattr(
        real.Neo4jMutationGuard, "completed_raw_or_none", completed_raw_or_none
    )
    monkeypatch.setattr(
        real.Neo4jMutationGuard,
        "recovered_ambiguous_marker_or_none",
        recovered_ambiguous_marker_or_none,
    )
    monkeypatch.setattr(real.Neo4jMutationGuard, "begin", begin)
    monkeypatch.setattr(real.Neo4jMutationGuard, "complete", complete)
    monkeypatch.setattr(real, "_bootstrap_graphiti_schema", lambda _driver: asyncio.sleep(0))

    class Graphiti:
        def __init__(self, *_args: object, **_values: object) -> None:
            self.driver = object()

        async def close(self) -> None:
            return None

    class Embedder:
        def __init__(self, *_args: object, **_values: object) -> None:
            pass

        def receipt(self) -> dict[str, object]:
            return {
                "requests": [],
                "request_count": 0,
                "embedding_tokens": 0,
                "cost_usd_microunits": 0,
                "usage_basis": "NO_EMBEDDING_CALL",
            }

    runtime = SimpleNamespace(
        Graphiti=Graphiti,
        OpenAIEmbedder=lambda **_values: SimpleNamespace(
            client=SimpleNamespace(close=AsyncMock())
        ),
        OpenAIEmbedderConfig=lambda **values: SimpleNamespace(**values),
        MeteredOpenAIEmbedder=Embedder,
        IdentityCrossEncoder=lambda: object(),
        EpisodeType=SimpleNamespace(text="text"),
        EpisodicNode=lambda **values: SimpleNamespace(**values),
        MutationGuard=real.Neo4jMutationGuard,
    )
    monkeypatch.setattr(real, "_load_graphiti", lambda: runtime)
    monkeypatch.setattr(
        real, "build_cli_llm_client", lambda **_values: SimpleNamespace(invocations=[])
    )
    monkeypatch.setattr(
        real,
        "_ensure_episode",
        lambda **_values: asyncio.sleep(
            0, result=(SimpleNamespace(uuid="episode-id"), "RETAINED")
        ),
    )

    class Pipeline:
        def __init__(self, guard: object) -> None:
            self.guard = guard
            self.recovery_marker = None
            self.complete_receipt = None
            self.complete_failure_receipt = None

        async def _prepare_attempt(self) -> dict[str, object] | None:
            marker = await self.guard.begin()
            if marker.state in {
                GuardState.PENDING,
                GuardState.RECOVERED_AMBIGUOUS,
            }:
                self.recovery_marker = marker
                raise real.CombinedTemporalPipelineError(
                    "retained marker is unresolved",
                    graph_effect_attempted=False,
                    rollback_completed=False,
                )
            if marker.state is GuardState.COMPLETE:
                return await self.guard.completed_raw_or_none()
            return None

    monkeypatch.setattr(
        real,
        "combined_temporal_pipeline_for",
        lambda **values: Pipeline(values["guard"]),
    )

    async def extract(_revision: object, **values: object) -> SimpleNamespace:
        nonlocal provider_calls
        provider_calls += 1
        pipeline = values["pipeline"]
        await pipeline.guard.complete({"success": "attempt-3"})
        return SimpleNamespace(
            outcome=real.CombinedTemporalOutcome.TERMINAL_SUCCESS_ZERO_PROPOSALS,
            nodes=(),
            edges=(),
        )

    monkeypatch.setattr(real, "extract_combined_temporal_async", extract)
    configuration, revision = _combined_runtime_inputs("Body", "episode-id")

    class Proof:
        def __init__(
            self, allowed: bool, settled: bool, recovered_allowed: bool
        ) -> None:
            self.allowed = allowed
            self.settled = settled
            self.recovered_allowed = recovered_allowed

        def allows_fresh_zero_dispatch_retry(
            self, *, episode_uuid: str, attempt_number: int
        ) -> bool:
            return self.allowed and episode_uuid == "episode-id" and attempt_number == 3

        def allows_fresh_completed_rollback_retry(
            self,
            *,
            episode_uuid: str,
            attempt_number: int,
            prior_attempt_number: int,
        ) -> bool:
            return (
                self.settled
                and episode_uuid == "episode-id"
                and attempt_number > 2
                and prior_attempt_number == attempt_number - 1
            )

        def allows_fresh_recovered_ambiguous_retry(self, **_values: object) -> bool:
            return self.recovered_allowed

    async def run(
        attempt_number: int,
        *,
        allowed: bool,
        settled: bool = False,
        recovered: object | None = None,
        recovered_allowed: bool = False,
    ) -> None:
        await real._add_episode(
            api_key="key",
            password="password",
            body="Body",
            name="episode-id",
            episode_id="episode-id",
            reference_time=datetime(2026, 8, 20, tzinfo=UTC),
            telemetry=real._EpisodeTelemetry(),
            attempt_number=attempt_number,
            validate_result=lambda *_args, **_values: {},
            restore_result=lambda raw, _telemetry: restored.append(dict(raw)),
            configuration=configuration,
            revision=revision,
            invocation_observer=Proof(allowed, settled, recovered_allowed),
            retry_snapshot_is_failed=lambda raw: "failure" in raw,
            recovered_ambiguous_progression=recovered,
        )

    complete_attempt_2 = {
        "attempt_number": 2,
        "provider_attempt_number": 1,
        "failure": "attempt-2",
    }
    markers["episode-id"] = complete_attempt_2
    asyncio.run(run(3, allowed=False, settled=False))
    assert restored == [complete_attempt_2]
    assert provider_calls == 0
    asyncio.run(run(3, allowed=False, settled=True))
    assert provider_calls == 1
    assert markers["episode-id"] == complete_attempt_2
    assert markers["episode-id:attempt:3"] == {"success": "attempt-3"}

    del markers["episode-id:attempt:3"]
    markers["episode-id"] = {
        "attempt_number": 1,
        "provider_attempt_number": 1,
        "failure": "attempt-1",
    }
    restored.clear()
    provider_calls = 0
    asyncio.run(run(3, allowed=False, settled=True))
    assert provider_calls == 0
    assert restored == [markers["episode-id"]]

    markers["episode-id"] = {"failure": "attempt-1"}
    restored.clear()
    asyncio.run(run(2, allowed=False))
    assert restored == [{"failure": "attempt-1"}]
    asyncio.run(run(3, allowed=True))
    assert provider_calls == 1
    assert markers["episode-id"] == {"failure": "attempt-1"}
    assert markers["episode-id:attempt:3"] == {"success": "attempt-3"}
    asyncio.run(run(3, allowed=False))
    assert provider_calls == 1
    assert restored[-1] == {"success": "attempt-3"}
    asyncio.run(run(4, allowed=False))
    assert provider_calls == 1
    assert restored[-1] == {"success": "attempt-3"}

    for state in ("PENDING", "RECOVERED_AMBIGUOUS"):
        markers["episode-id:attempt:3"] = {"state": state}
        with pytest.raises(real.AmbiguousEpisodeEffect):
            asyncio.run(run(4, allowed=False))
        assert provider_calls == 1

    markers["episode-id:attempt:3"] = {"state": "RECOVERED_AMBIGUOUS"}
    with pytest.raises(real.AmbiguousEpisodeEffect):
        asyncio.run(run(4, allowed=False, settled=False))
    assert provider_calls == 1
    asyncio.run(run(4, allowed=False, settled=True))
    assert provider_calls == 2
    assert markers["episode-id:attempt:4"] == {"success": "attempt-3"}

    markers["episode-id:attempt:4"] = {
        "failure": "attempt-4",
        "attempt_number": 4,
        "provider_attempt_number": 4,
    }
    asyncio.run(run(5, allowed=False, settled=True))
    assert provider_calls == 3
    assert markers["episode-id:attempt:5"] == {"success": "attempt-3"}

    del markers["episode-id:attempt:4"]
    del markers["episode-id:attempt:5"]
    markers["episode-id:attempt:3"] = {"state": "RECOVERED_AMBIGUOUS"}
    marker_input = digest_canonical(
        {
            "episode_uuid": "episode-id",
            "name": "episode-id",
            "body": "Body",
            "reference_time": datetime(2026, 8, 20, tzinfo=UTC).isoformat(),
            "group_id": GRAPHITI_WORKSPACE_GROUP,
        }
    )
    exact_recovery = SimpleNamespace(
        ingest_id="episode-id",
        skipped_attempt_number=4,
        marker_attempt_number=3,
        marker_input_digest=marker_input,
        recovery_marker_digest=digest_canonical(
            {
                "state": "RECOVERED_AMBIGUOUS",
                "attempt_number": 3,
                "workspace_group": GRAPHITI_WORKSPACE_GROUP,
                "episode_uuid": "episode-id",
                "input_digest": marker_input,
            }
        ),
    )
    with pytest.raises(real.AmbiguousEpisodeEffect):
        asyncio.run(
            run(
                5,
                allowed=False,
                recovered=SimpleNamespace(
                    **{
                        **vars(exact_recovery),
                        "recovery_marker_digest": "sha256:" + "0" * 64,
                    }
                ),
                recovered_allowed=True,
            )
        )
        assert provider_calls == 3
    with pytest.raises(real.AmbiguousEpisodeEffect):
        asyncio.run(run(5, allowed=False, recovered=exact_recovery))
    asyncio.run(
        run(
            5,
            allowed=False,
            recovered=exact_recovery,
            recovered_allowed=True,
        )
    )
    assert provider_calls == 4
    assert markers["episode-id:attempt:5"] == {"success": "attempt-3"}


def test_attempt_marker_keeps_the_stable_source_episode_identity() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import GuardError, Neo4jMutationGuard

    queried: list[str] = []

    class Driver:
        async def execute_query(
            self,
            _query: str,
            *,
            params: dict[str, object],
            routing_: str,
        ) -> tuple[list[object], None, None]:
            assert routing_ == "w"
            queried.append(str(params["episode_uuid"]))
            return [], None, None

    guard = Neo4jMutationGuard(
        Driver(),
        group_id=GRAPHITI_WORKSPACE_GROUP,
        episode_uuid="episode-id",
        marker_episode_uuid="episode-id:attempt:3",
        attempt_number=3,
        input_digest="sha256:" + "0" * 64,
    )

    assert guard.episode_uuid == "episode-id"
    assert not asyncio.run(guard.marker_exists())
    assert queried == ["episode-id:attempt:3"]
    with pytest.raises(GuardError, match="attempt marker identity differs"):
        guard._bind_marker(
            {
                "state": "COMPLETE",
                "group_id": GRAPHITI_WORKSPACE_GROUP,
                "attempt_number": 1,
                "input_digest": "sha256:" + "0" * 64,
                "snapshot_id": "episode-id:1",
                "chat_invocations_json": "[]",
                "embedding_usage_json": "null",
            }
        )


def test_immutable_completion_preserves_original_access_after_rights_renewal(
    tmp_path: Path,
) -> None:
    from newsroom.graphiti_adapter.combined_temporal_contract import SourceRevisionInput
    from newsroom.graphiti_adapter.combined_temporal_extraction import (
        CombinedTemporalTransportResult,
        extract_combined_temporal,
    )
    from newsroom.graphiti_adapter.evaluation_packet import GRAPHITI_CORE_RELEASE
    from newsroom.graphiti_adapter.real import _EpisodeTelemetry, _raw_receipt
    from newsroom.graphiti_adapter.result_snapshot import restore_validated_snapshot

    instant = UtcTimestamp(datetime(2026, 8, 20, tzinfo=UTC))
    original = replace(
        _real_attempt(tmp_path),
        reference_time=instant,
        temporal_basis=TemporalBasis.SOURCE_PUBLISHED,
    )
    captured: dict[str, object] = {}

    class Transport:
        def generate_response(self, **kwargs: object) -> CombinedTemporalTransportResult:
            del kwargs
            return CombinedTemporalTransportResult(
                raw={"entities": [], "facts": []},
                framework_version=GRAPHITI_CORE_RELEASE,
                model_version="composer-2.5",
                token_usage={"basis": "PROVIDER_REPORTED", "output_tokens": 1},
                provider_cost=None,
            )

    class Pipeline:
        def prepare_attempt(self) -> None:
            return None

        def complete_failure(self, terminal: dict[str, object]) -> dict[str, object]:
            return dict(terminal)

        def execute(self, **kwargs: object) -> SimpleNamespace:
            receipt = dict(kwargs["receipt"])
            captured["combined"] = receipt
            return SimpleNamespace(
                nodes=(),
                edges=(),
                guarded_edges=(),
                node_resolutions=(),
                graph_effect_attempted=False,
                embedding_skipped=True,
                journal_skipped=True,
                rollback_skipped=True,
                completed_receipt=receipt,
            )

    revision = SourceRevisionInput(
        body="Alice met Bob on 2026-08-20.",
        revision_id="rev",
        source_id="src",
        item_key="item",
        representation_digest="sha256:" + "ab" * 32,
        published_at=instant.to_text(),
        updated_at=None,
        observed_at=instant.to_text(),
        ingested_at=instant.to_text(),
    )
    extract_combined_temporal(revision, transport=Transport(), pipeline=Pipeline())
    raw = _raw_receipt(
        original,
        started_at=instant,
        telemetry=_EpisodeTelemetry(provider_attempt_number=1),
        result=None,
        proposals=(),
    )
    raw.pop("raw_output_digest", None)
    raw["combined_temporal_receipt"] = dict(captured["combined"])  # type: ignore[arg-type]
    raw["raw_output_digest"] = digest_bytes(canonical_json_bytes(raw))
    old_access = raw["passages"][0]["access_decision_id"]
    current_passages = tuple(
        replace(
            passage,
            access_decision_id=ObjectAccessDecisionId.parse(
                f"00000000-0000-4000-8000-{9_900 + index:012d}"
            ),
        )
        for index, passage in enumerate(
            original.extraction_request.input_binding.passages,
            start=1,
        )
    )
    current_binding = replace(
        original.extraction_request.input_binding,
        passages=current_passages,
    )
    current_request = replace(
        original.extraction_request,
        input_binding=current_binding,
    )
    current_manifest = GraphitiInputManifest.from_run_request(
        manifest_id=original.manifest.manifest_id,
        configuration=original.configuration,
        contract=original.extraction_contract,
        request=current_request,
    )
    renewed = replace(
        original,
        manifest=current_manifest,
        extraction_request=current_request,
    )
    restored = restore_validated_snapshot(raw=raw, attempt=renewed)
    assert restored.produced.raw_output_value["passages"][0][
        "access_decision_id"
    ] == old_access


def test_runtime_metered_embedder_satisfies_a_nominal_client_contract() -> None:
    import newsroom.graphiti_adapter.real as real

    class NominalEmbedderClient:
        pass

    meter_type = real._runtime_metered_embedder_type(NominalEmbedderClient)
    meter = meter_type(
        SimpleNamespace(
            client=SimpleNamespace(),
            config=SimpleNamespace(
                embedding_model="openai/text-embedding-3-large",
                embedding_dim=2,
            ),
        )
    )

    assert isinstance(meter, NominalEmbedderClient)
    assert isinstance(meter, real.MeteredOpenAIEmbedder)


def test_embedding_meter_retains_provider_tokens_and_native_usd_cost() -> None:
    from newsroom.graphiti_adapter.embedding_meter import MeteredOpenAIEmbedder

    class Embeddings:
        async def create(self, **_values: object) -> object:
            usage = SimpleNamespace(
                model_dump=lambda: {
                    "prompt_tokens": 19,
                    "total_tokens": 19,
                    "cost": "0.000017",
                }
            )
            return SimpleNamespace(
                id="request-1",
                usage=usage,
                data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
            )

    delegate = SimpleNamespace(
        client=SimpleNamespace(embeddings=Embeddings()),
        config=SimpleNamespace(
            embedding_model="openai/text-embedding-3-large",
            embedding_dim=2,
        ),
    )
    meter = MeteredOpenAIEmbedder(delegate)
    assert asyncio.run(meter.create("retained text")) == [0.1, 0.2]
    assert meter.receipt() == {
        "requests": [
            {
                "provider": "openrouter",
                "model": "openai/text-embedding-3-large",
                "request_id": "request-1",
                "prompt_tokens": 19,
                "total_tokens": 19,
                "cost_usd_microunits": 17,
                "cost_reported": True,
                "outcome": "COMPLETE",
            }
        ],
        "request_count": 1,
        "embedding_tokens": 19,
        "cost_usd_microunits": 17,
        "usage_basis": "PROVIDER_REPORTED",
    }


def test_embedding_meter_retains_ambiguous_failed_provider_request() -> None:
    from newsroom.graphiti_adapter.embedding_meter import MeteredOpenAIEmbedder

    class Embeddings:
        async def create(self, **_values: object) -> object:
            raise RuntimeError("provider response was lost")

    delegate = SimpleNamespace(
        client=SimpleNamespace(embeddings=Embeddings()),
        config=SimpleNamespace(
            embedding_model="openai/text-embedding-3-large",
            embedding_dim=2,
        ),
    )
    meter = MeteredOpenAIEmbedder(delegate)
    with pytest.raises(RuntimeError, match="response was lost"):
        asyncio.run(meter.create("retained text"))
    receipt = meter.receipt()
    assert receipt["usage_basis"] == "PROVIDER_PARTIALLY_UNREPORTED"
    assert receipt["request_count"] == 1
    assert receipt["requests"][0]["outcome"] == "UNOBSERVED"


def test_else_branch_constructs_real_adapter_instead_of_unreachable_assertion() -> None:
    source = inspect.getsource(_GraphitiAdapterBoundary._execute_attempt_locked)
    assert "unreachable real Graphiti execution path" not in source
    assert "require_execution_authorized()" in source
    assert "adapter = RealGraphitiAdapter(" in source
    assert inspect.signature(RealGraphitiAdapter.execute) == inspect.signature(
        DeterministicFakeGraphitiAdapter.execute
    )


def test_placeholder_packet_still_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import newsroom.graphiti_adapter.real as real

    runtime_loads: list[str] = []
    monkeypatch.setattr(
        real,
        "_load_graphiti",
        lambda: runtime_loads.append("loaded"),
    )
    policy = GraphitiWorkspacePolicy(
        policy_id=GraphitiWorkspacePolicyId.parse(
            "00000000-0000-4000-8000-000000004935"
        ),
        policy_version="graphiti-disposable-workspace-v1",
        namespace_prefix="graphiti-real-evaluation",
        max_workspace_bytes=1024 * 1024,
        max_private_nodes=100,
        max_private_relations=100,
        egress_policy=GraphitiEgressPolicy.APPROVED_PROVIDER_ONLY,
        credential_class=GraphitiCredentialClass.PROPOSAL_WORKSPACE_ONLY,
    )
    attempt = _real_attempt(
        tmp_path,
        authority=_placeholder_authority(),
        workspace_policy=policy,
    )
    attempt.configuration.require_execution_authorized()
    with pytest.raises(GraphitiAdapterContractError, match="EVALUATION CLI packet pins"):
        RealGraphitiAdapter().execute(
            attempt=attempt,
            workspace_root=(tmp_path / "workspace").resolve(),
        )
    assert runtime_loads == []


def test_evaluation_packet_is_the_only_authorised_real_profile(tmp_path: Path) -> None:
    assert REAL_GRAPHITI_RUNTIME_ENABLED is True
    production = _real_attempt(
        tmp_path, execution_profile=GraphitiExecutionProfile.PRODUCTION
    )
    with pytest.raises(GraphitiAdapterContractError, match="EVALUATION"):
        RealGraphitiAdapter().execute(
            attempt=production,
            workspace_root=(tmp_path / "workspace").resolve(),
        )
    with pytest.raises(GraphitiRuntimeNotAuthorized, match="EVALUATION"):
        production.configuration.require_execution_authorized()


def test_authorised_evaluation_attempt_does_not_import_graphiti_core() -> None:
    graphiti_was_loaded = "graphiti_core" in sys.modules
    from newsroom.graphiti_adapter.evaluation_attempt import evaluation_attempt_for
    from newsroom.graphiti_adapter.evaluation_packet import GRAPHITI_WORKSPACE_GROUP

    attempt = evaluation_attempt_for(("香港天文台發出強烈季候風信號。",))
    attempt.configuration.require_execution_authorized()
    assert attempt.configuration.workspace_policy.namespace_prefix == GRAPHITI_WORKSPACE_GROUP
    assert ("graphiti_core" in sys.modules) is graphiti_was_loaded
    assert REAL_GRAPHITI_RUNTIME_ENABLED is True


def test_evaluation_attempt_response_budget_matches_call_shape_ceiling() -> None:
    from newsroom.control_plane.graphiti_requests import (
        GraphitiLeafClass,
        load_checked_graphiti_call_shape_policy,
    )
    from newsroom.graphiti_adapter.evaluation_attempt import evaluation_attempt_for

    primary = load_checked_graphiti_call_shape_policy().route_for(
        GraphitiLeafClass.PRIMARY
    )
    attempt = evaluation_attempt_for(("A retained source passage.",))
    assert (
        attempt.extraction_request.budget.max_response_tokens
        == primary.max_output_tokens
        == 16_384
    )


def test_retryable_failure_returns_diagnostic_receipt_without_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real

    async def fail(**values: object) -> object:
        telemetry = values["telemetry"]
        telemetry.chat_invocations.append(
            {
                "provider": "cursor-agent-cli",
                "model": "composer-2.5",
                "outcome": "FAILED",
            }
        )
        telemetry.embedding_usage = {
            "usage_basis": "NO_EMBEDDING_CALL",
            "request_count": 0,
            "embedding_tokens": 0,
            "cost_usd_microunits": 0,
            "requests": [],
        }
        raise RuntimeError("chat failed")

    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "key")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "password")
    monkeypatch.setattr(real, "_add_episode", fail)
    attempt = evaluation_attempt_for(("A retained source passage.",))
    produced = RealGraphitiAdapter()._produce(
        attempt,
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )
    assert produced.outcome is ExtractionOutcome.RETRYABLE_FAILURE
    assert produced.failure_code is ExtractionFailureCode.PRODUCER_INTERNAL_ERROR
    assert produced.raw_output_value is None
    assert produced.attempt_receipt_value is not None
    assert produced.attempt_receipt_value["chat_invocation_count"] == 1
    assert produced.attempt_receipt_value["producer_failure"] == "RuntimeError"
    assert produced.attempt_receipt_value["usage_basis"] == "NO_EMBEDDING_CALL"
    assert produced.attempt_receipt_value["token_usage"]["usage_basis"] == (
        "UNREPORTED"
    )
    assert produced.attempt_receipt_value["token_usage"][
        "unreported_chat_requests"
    ] == 1


def test_completed_pipeline_failure_is_retryable_not_schema_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real

    terminal_digest = digest_bytes(b"attempt-two-terminal")

    async def completed_failure(**values: object) -> object:
        validate_failure = values["validate_failure"]
        assert callable(validate_failure)
        validate_failure(
            {
                "failure_code": CombinedTemporalFailureCode.PIPELINE_FAILED,
                "provider_attempt_number": 1,
                "pipeline_chat_invocations": [
                    {
                        "provider": "grok-build-cli",
                        "model_invocation_terminal_digest": terminal_digest,
                    }
                ],
                "embedding_usage": {
                    "usage_basis": "NO_EMBEDDING_CALL",
                    "request_count": 0,
                    "embedding_tokens": 0,
                    "cost_usd_microunits": 0,
                    "requests": [],
                },
            },
            values["telemetry"],
        )
        raise real.ExtractionContractError("combined-temporal leaf failed")

    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "key")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "password")
    monkeypatch.setattr(real, "_add_episode", completed_failure)

    first = evaluation_attempt_for(("A retained source passage.",))
    attempt = replace(
        first,
        attempt_id=GraphitiAttemptId.parse(
            "00000000-0000-4000-8000-000000004935"
        ),
        attempt_number=2,
        expected_previous_attempt_id=first.attempt_id,
        idempotency_key="evaluation-real-attempt-v2",
    )
    produced = RealGraphitiAdapter()._produce(
        attempt,
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )

    assert produced.outcome is ExtractionOutcome.RETRYABLE_FAILURE
    assert produced.failure_code is ExtractionFailureCode.PRODUCER_INTERNAL_ERROR
    assert produced.validation is None
    assert produced.raw_output_value is None
    assert produced.attempt_receipt_value is not None
    assert produced.attempt_receipt_value["combined_temporal_failure_code"] == (
        "PIPELINE_FAILED"
    )
    assert produced.attempt_receipt_value["provider_attempt_number"] == 2
    assert produced.attempt_receipt_value["combined_temporal_receipt"][
        "provider_attempt_number"
    ] == 1
    assert produced.attempt_receipt_value["chat_invocations"] == [
        {
            "provider": "grok-build-cli",
            "model_invocation_terminal_digest": terminal_digest,
        }
    ]


def test_unmarked_ambiguity_after_empty_success_returns_validated_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real

    async def stash_success_then_roll_back(**values: object) -> object:
        values["telemetry"].embedding_usage = {
            "usage_basis": "NO_EMBEDDING_CALL",
            "request_count": 0,
            "embedding_tokens": 0,
            "cost_usd_microunits": 0,
            "requests": [],
        }
        values["validate_result"](
            SimpleNamespace(episode=None, nodes=(), edges=()),
            values["telemetry"],
        )
        raise real.AmbiguousEpisodeEffect(
            "Graphiti write failed after provider dispatch and was rolled back"
        )

    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "key")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "password")
    monkeypatch.setattr(real, "_add_episode", stash_success_then_roll_back)

    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )

    assert produced.outcome is ExtractionOutcome.SUCCESS
    assert produced.failure_code is ExtractionFailureCode.NONE
    assert produced.validation is ExtractionOutputValidation.VALID
    assert produced.proposals == ()
    assert produced.raw_output_value is not None
    assert "recovery_classification" not in produced.raw_output_value


def test_unmarked_zero_without_validated_result_remains_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real

    async def complete_then_unmarked(**values: object) -> object:
        values["telemetry"].chat_invocations = [
            {
                "provider": "cursor-agent-cli",
                "model": "composer-2.5",
                "outcome": "COMPLETE",
                "usage": {
                    "usage_basis": "PROVIDER_REPORTED",
                    "input_tokens": 4_694,
                    "cached_read_tokens": 448,
                    "output_tokens": 3_124,
                    "total_tokens": 8_266,
                },
            }
        ]
        values["telemetry"].embedding_usage = {
            "usage_basis": "PROVIDER_REPORTED",
            "request_count": 5,
            "embedding_tokens": 43,
            "cost_usd_microunits": 4,
            "requests": [
                {
                    "provider": "openrouter",
                    "model": "openai/text-embedding-3-large",
                    "request_id": f"unmarked-embedding-{index}",
                    "prompt_tokens": 2,
                    "total_tokens": 2,
                    "cost_usd_microunits": 0,
                    "cost_reported": True,
                    "outcome": "COMPLETE",
                }
                for index in range(5)
            ],
        }
        raise real.AmbiguousEpisodeEffect(
            "Graphiti completion became ambiguous after provider dispatch"
        )

    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "key")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "password")
    monkeypatch.setattr(real, "_add_episode", complete_then_unmarked)

    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )

    assert produced.outcome is ExtractionOutcome.RETRYABLE_FAILURE
    assert produced.failure_code is ExtractionFailureCode.AMBIGUOUS_EFFECT
    assert produced.validation is None
    assert produced.raw_output_value is None
    assert produced.proposals == ()
    receipt = produced.attempt_receipt_value
    assert receipt is not None
    # Live 13665 stored integer zeros from result=None. Those counts are not
    # an accepted zero-proposal mark.
    assert receipt["proposal_count"] == 0
    assert receipt["entity_count"] == 0
    assert receipt["relation_count"] == 0
    assert receipt.get("zero_proposal_effect") is None


def test_persistable_empty_after_embeddings_is_explicit_zero_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real
    from newsroom.graphiti_adapter.combined_temporal_pipeline import (
        ExistingGraphitiPipeline,
    )
    from newsroom.graphiti_adapter.neo4j_guard import GuardState

    persist_calls: list[object] = []

    class Guard:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def begin(self) -> object:
            self.calls.append("begin")
            return SimpleNamespace(state=GuardState.CREATED)

        async def record_pending_telemetry(self, **_values: object) -> None:
            self.calls.append("telemetry")

        async def complete(self, _receipt: object) -> None:
            self.calls.append("complete")

        async def rollback_pending(self, **_values: object) -> bool:
            self.calls.append("rollback")
            return True

    async def persistable_empty_after_embeddings(**values: object) -> object:
        telemetry = values["telemetry"]
        telemetry.chat_invocations = [
            {
                "provider": "cursor-agent-cli",
                "model": "composer-2.5",
                "outcome": "COMPLETE",
                "usage": {
                    "usage_basis": "PROVIDER_REPORTED",
                    "input_tokens": 4_694,
                    "cached_read_tokens": 448,
                    "output_tokens": 3_124,
                    "total_tokens": 8_266,
                },
            }
        ]
        telemetry.embedding_usage = {
            "usage_basis": "PROVIDER_REPORTED",
            "request_count": 5,
            "embedding_tokens": 43,
            "cost_usd_microunits": 4,
            "requests": [
                {
                    "provider": "openrouter",
                    "model": "openai/text-embedding-3-large",
                    "request_id": f"live-embedding-{index}",
                    "prompt_tokens": tokens,
                    "total_tokens": tokens,
                    "cost_usd_microunits": cost,
                    "cost_reported": True,
                    "outcome": "COMPLETE",
                }
                for index, (tokens, cost) in enumerate(
                    ((33, 4), (3, 0), (2, 0), (2, 0), (3, 0))
                )
            ],
        }

        async def resolve_nodes(
            nodes: list[object],
        ) -> tuple[list[object], dict[str, str], list[tuple[object, object]]]:
            held = [
                SimpleNamespace(
                    uuid=str(getattr(node, "uuid", "held")),
                    attributes={"resolution": "AMBIGUOUS_HOLD"},
                )
                for node in nodes
            ]
            return (
                held,
                {
                    str(getattr(node, "uuid", "held")): str(
                        getattr(node, "uuid", "held")
                    )
                    for node in nodes
                },
                [],
            )

        async def persist_graph(nodes: list[object], edges: list[object]) -> None:
            persist_calls.append((list(nodes), list(edges)))
            raise RuntimeError("persist must not run for empty persistable effect")

        pipeline = ExistingGraphitiPipeline(
            guard=Guard(),  # type: ignore[arg-type]
            resolve_nodes=resolve_nodes,
            resolve_pointers=lambda edges, _uuid_map: edges,
            create_embeddings=lambda _embedder, _edges: asyncio.sleep(0),
            persist_graph=persist_graph,
            embedder=object(),
            run_async=asyncio.run,
            chat_receipt=lambda: list(telemetry.chat_invocations),
            embedding_receipt=lambda: dict(telemetry.embedding_usage),
            complete_receipt=lambda nodes, edges, receipt: values["validate_result"](
                SimpleNamespace(
                    episode=None,
                    nodes=tuple(nodes),
                    edges=tuple(edges),
                ),
                telemetry,
                receipt,
            ),
        )
        await pipeline._prepare_attempt()
        sealed = await pipeline._execute(
            nodes=(SimpleNamespace(uuid="entity-1", attributes={}),),
            edges=(),
            receipt={"provider_attempt_number": 1},
        )
        assert persist_calls == []
        assert sealed.graph_effect_attempted is False
        assert sealed.nodes == ()
        return SimpleNamespace(episode=None, nodes=(), edges=())

    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "key")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "password")
    monkeypatch.setattr(real, "_add_episode", persistable_empty_after_embeddings)

    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )

    assert persist_calls == []
    assert produced.outcome is ExtractionOutcome.SUCCESS
    assert produced.failure_code is ExtractionFailureCode.NONE
    assert produced.validation is ExtractionOutputValidation.VALID
    assert produced.proposals == ()
    assert produced.raw_output_value is not None
    assert produced.raw_output_value["entity_count"] == 0
    assert produced.raw_output_value["relation_count"] == 0
    assert produced.raw_output_value["proposal_count"] == 0
    combined = produced.raw_output_value.get("combined_temporal_receipt")
    assert isinstance(combined, dict)
    assert combined["zero_proposal_effect"] == "EXPLICIT"
    assert produced.raw_output_value["embedding_usage"]["request_count"] == 5
    assert produced.raw_output_value["chat_invocations"][0]["usage"][
        "total_tokens"
    ] == 8_266


def test_persistable_new_nodes_without_edges_after_embeddings_is_explicit_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real
    from newsroom.graphiti_adapter.combined_temporal_pipeline import (
        ExistingGraphitiPipeline,
    )
    from newsroom.graphiti_adapter.neo4j_guard import GuardState

    persist_calls: list[object] = []

    class Guard:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def begin(self) -> object:
            self.calls.append("begin")
            return SimpleNamespace(state=GuardState.CREATED)

        async def record_pending_telemetry(self, **_values: object) -> None:
            self.calls.append("telemetry")

        async def complete(self, _receipt: object) -> None:
            self.calls.append("complete")

        async def rollback_pending(self, **_values: object) -> bool:
            self.calls.append("rollback")
            return True

    async def leftover_new_nodes_without_edges(**values: object) -> object:
        telemetry = values["telemetry"]
        telemetry.chat_invocations = [
            {
                "provider": "cursor-agent-cli",
                "model": "composer-2.5",
                "outcome": "COMPLETE",
                "usage": {
                    "usage_basis": "PROVIDER_REPORTED",
                    "input_tokens": 5_400,
                    "cached_read_tokens": 500,
                    "output_tokens": 3_046,
                    "total_tokens": 8_446,
                },
            }
        ]
        telemetry.embedding_usage = {
            "usage_basis": "PROVIDER_REPORTED",
            "request_count": 4,
            "embedding_tokens": 44,
            "cost_usd_microunits": 6,
            "requests": [
                {
                    "provider": "openrouter",
                    "model": "openai/text-embedding-3-large",
                    "request_id": f"live-13677-embedding-{index}",
                    "prompt_tokens": tokens,
                    "total_tokens": tokens,
                    "cost_usd_microunits": cost,
                    "cost_reported": True,
                    "outcome": "COMPLETE",
                }
                for index, (tokens, cost) in enumerate(((36, 5), (4, 1), (2, 0), (2, 0)))
            ],
        }

        async def resolve_nodes(
            nodes: list[object],
        ) -> tuple[list[object], dict[str, str], list[tuple[object, object]]]:
            created = [
                SimpleNamespace(
                    uuid=str(getattr(node, "uuid", "new")),
                    attributes={"resolution": "DETERMINISTIC_NEW_NODE"},
                )
                for node in nodes
            ]
            return (
                created,
                {
                    str(getattr(node, "uuid", "new")): str(
                        getattr(node, "uuid", "new")
                    )
                    for node in nodes
                },
                [],
            )

        async def persist_graph(nodes: list[object], edges: list[object]) -> None:
            persist_calls.append((list(nodes), list(edges)))
            raise RuntimeError("persist must not run without persistable edges")

        pipeline = ExistingGraphitiPipeline(
            guard=Guard(),  # type: ignore[arg-type]
            resolve_nodes=resolve_nodes,
            resolve_pointers=lambda edges, _uuid_map: edges,
            create_embeddings=lambda _embedder, _edges: asyncio.sleep(0),
            persist_graph=persist_graph,
            embedder=object(),
            run_async=asyncio.run,
            chat_receipt=lambda: list(telemetry.chat_invocations),
            embedding_receipt=lambda: dict(telemetry.embedding_usage),
            complete_receipt=lambda nodes, edges, receipt: values["validate_result"](
                SimpleNamespace(
                    episode=None,
                    nodes=tuple(nodes),
                    edges=tuple(edges),
                ),
                telemetry,
                receipt,
            ),
        )
        await pipeline._prepare_attempt()
        sealed = await pipeline._execute(
            nodes=(SimpleNamespace(uuid="entity-1", attributes={}),),
            edges=(),
            receipt={"provider_attempt_number": 1},
        )
        assert persist_calls == []
        assert sealed.graph_effect_attempted is False
        return SimpleNamespace(episode=None, nodes=(), edges=())

    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "key")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "password")
    monkeypatch.setattr(real, "_add_episode", leftover_new_nodes_without_edges)

    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )

    assert persist_calls == []
    assert produced.outcome is ExtractionOutcome.SUCCESS
    assert produced.failure_code is ExtractionFailureCode.NONE
    assert produced.validation is ExtractionOutputValidation.VALID
    assert produced.proposals == ()
    assert produced.raw_output_value is not None
    assert produced.raw_output_value["entity_count"] == 0
    assert produced.raw_output_value["relation_count"] == 0
    assert produced.raw_output_value["proposal_count"] == 0
    combined = produced.raw_output_value.get("combined_temporal_receipt")
    assert isinstance(combined, dict)
    assert combined["zero_proposal_effect"] == "EXPLICIT"
    assert produced.raw_output_value["embedding_usage"]["request_count"] == 4
    assert produced.raw_output_value["chat_invocations"][0]["usage"][
        "total_tokens"
    ] == 8_446


@pytest.mark.parametrize("with_recovery_marker", (False, True))
def test_ambiguity_with_proposals_remains_fail_closed_after_validation(
    monkeypatch: pytest.MonkeyPatch, with_recovery_marker: bool
) -> None:
    import newsroom.graphiti_adapter.real as real

    async def stash_proposals_then_roll_back(**values: object) -> object:
        values["telemetry"].embedding_usage = {
            "usage_basis": "PROVIDER_REPORTED",
            "request_count": 1,
            "embedding_tokens": 4,
            "cost_usd_microunits": 1,
            "requests": [
                {
                    "provider": "openrouter",
                    "model": "openai/text-embedding-3-large",
                    "request_id": "proposal-embedding",
                    "prompt_tokens": 4,
                    "total_tokens": 4,
                    "cost_usd_microunits": 1,
                    "cost_reported": True,
                    "outcome": "COMPLETE",
                }
            ],
        }
        values["validate_result"](
            SimpleNamespace(
                episode=None,
                nodes=(
                    SimpleNamespace(
                        uuid="node-1",
                        name="retained source",
                        summary=None,
                    ),
                ),
                edges=(),
            ),
            values["telemetry"],
        )
        if with_recovery_marker:
            values["telemetry"].recovery_classification = (
                GraphitiRecoveryClassification.ROLLED_BACK_AMBIGUOUS_EFFECT
            )
        raise real.AmbiguousEpisodeEffect(
            "Graphiti write failed after provider dispatch and was rolled back"
        )

    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "key")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "password")
    monkeypatch.setattr(real, "_add_episode", stash_proposals_then_roll_back)

    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )

    assert produced.outcome is ExtractionOutcome.RETRYABLE_FAILURE
    assert produced.failure_code is ExtractionFailureCode.AMBIGUOUS_EFFECT
    assert produced.validation is None
    assert produced.raw_output_value is None


def test_pre_dispatch_setup_failure_is_a_proved_no_call_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real

    def missing_runtime() -> object:
        raise GraphitiAdapterContractError("graphiti runtime is absent")

    monkeypatch.setattr(real, "_load_graphiti", missing_runtime)
    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )

    receipt = produced.attempt_receipt_value
    assert produced.outcome is ExtractionOutcome.RETRYABLE_FAILURE
    assert receipt is not None
    assert receipt["dispatch_state"] == "NOT_DISPATCHED"
    assert receipt["setup_failure"] == "GraphitiAdapterContractError"
    assert "setup_failure_detail" not in receipt
    assert receipt["chat_invocation_count"] == 0
    assert receipt["embedding_usage"]["request_count"] == 0
    assert receipt["embedding_usage"]["cost_usd_microunits"] == 0
    assert receipt["usage_basis"] == "NO_EMBEDDING_CALL"
    assert receipt["token_usage"]["usage_basis"] == "NO_PROVIDER_CALL"


def test_adapter_contract_error_rejects_unlisted_reason_code() -> None:
    with pytest.raises(ValueError, match="not allow-listed"):
        GraphitiAdapterContractError(
            "TOKEN=must-not-reach-the-store",
            reason_code="TOKEN=must-not-reach-the-store",
        )


def test_load_graphiti_missing_extra_sets_typed_reason_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    import newsroom.graphiti_adapter.real as real
    from newsroom.graphiti_adapter.types import GRAPHITI_EXTRA_REQUIRED

    real_import = builtins.__import__

    def refuse_graphiti(name: str, *args: object, **kwargs: object) -> object:
        if name == "graphiti_core" or name.startswith("graphiti_core."):
            raise ImportError("No module named 'graphiti_core'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_graphiti)
    with pytest.raises(GraphitiAdapterContractError) as caught:
        real._load_graphiti()
    assert caught.value.reason_code == GRAPHITI_EXTRA_REQUIRED


def test_missing_graphiti_extra_receipt_retains_typed_reason_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins
    import json

    import newsroom.graphiti_adapter.real as real
    from newsroom.graphiti_adapter.types import GRAPHITI_EXTRA_REQUIRED

    real_import = builtins.__import__
    secret = "TOKEN=must-not-reach-the-store"

    def refuse_graphiti(name: str, *args: object, **kwargs: object) -> object:
        if name == "graphiti_core" or name.startswith("graphiti_core."):
            raise ImportError(secret)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_graphiti)
    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )
    receipt = produced.attempt_receipt_value
    dumped = json.dumps(receipt)

    assert produced.outcome is ExtractionOutcome.RETRYABLE_FAILURE
    assert produced.failure_code is ExtractionFailureCode.PRODUCER_INTERNAL_ERROR
    assert receipt is not None
    assert receipt["dispatch_state"] == "NOT_DISPATCHED"
    assert receipt["setup_failure"] == "GraphitiAdapterContractError"
    assert receipt["setup_failure_detail"] == GRAPHITI_EXTRA_REQUIRED
    assert receipt["chat_invocation_count"] == 0
    assert receipt["embedding_usage"]["request_count"] == 0
    assert receipt["usage_basis"] == "NO_EMBEDDING_CALL"
    assert receipt["token_usage"]["usage_basis"] == "NO_PROVIDER_CALL"
    assert receipt["proposal_count"] == 0
    assert secret not in dumped
    assert "graphiti extra" not in dumped


def test_graphiti_core_release_mismatch_receipt_retains_typed_reason_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real
    from newsroom.graphiti_adapter.types import GRAPHITI_CORE_RELEASE_MISMATCH

    def mismatched_runtime() -> object:
        raise GraphitiAdapterContractError(
            "real Graphiti requires graphiti-core 0.29.3",
            reason_code=GRAPHITI_CORE_RELEASE_MISMATCH,
        )

    monkeypatch.setattr(real, "_load_graphiti", mismatched_runtime)
    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )
    receipt = produced.attempt_receipt_value
    assert receipt is not None
    assert receipt["setup_failure"] == "GraphitiAdapterContractError"
    assert receipt["setup_failure_detail"] == GRAPHITI_CORE_RELEASE_MISMATCH
    assert receipt["dispatch_state"] == "NOT_DISPATCHED"


def test_pre_dispatch_setup_failure_without_reason_code_omits_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    import newsroom.graphiti_adapter.real as real

    secret = "TOKEN=must-not-reach-the-store"

    def missing_runtime() -> object:
        raise GraphitiAdapterContractError(secret)

    monkeypatch.setattr(real, "_load_graphiti", missing_runtime)
    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )
    receipt = produced.attempt_receipt_value
    dumped = json.dumps(receipt)
    assert receipt is not None
    assert receipt["setup_failure"] == "GraphitiAdapterContractError"
    assert "setup_failure_detail" not in receipt
    assert secret not in dumped


def test_credential_time_is_deducted_from_absolute_extraction_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real

    class MonotonicClock:
        value = 0.0

        def __call__(self) -> float:
            return self.value

        def advance(self) -> None:
            self.value += 100.0

    monotonic = MonotonicClock()
    provider_calls = 0

    def delayed_api_key() -> str:
        monotonic.advance()
        return "key"

    def delayed_password() -> str:
        monotonic.advance()
        return "password"

    async def must_not_dispatch(**_values: object) -> object:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("expired extraction deadline reached provider")

    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", delayed_api_key)
    monkeypatch.setattr(real, "neo4j_community_password", delayed_password)
    monkeypatch.setattr(real, "_add_episode", must_not_dispatch)
    produced = RealGraphitiAdapter(monotonic=monotonic)._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )

    assert provider_calls == 0
    assert produced.outcome is ExtractionOutcome.RETRYABLE_FAILURE
    assert produced.failure_code is ExtractionFailureCode.EXECUTION_TIMEOUT
    assert produced.attempt_receipt_value is not None
    assert produced.attempt_receipt_value["embedding_usage"]["request_count"] == 0
    diagnostics = produced.attempt_receipt_value["timeout_diagnostics"]
    assert len(diagnostics) == 1
    assert diagnostics[0]["boundary"] == "EXTRACTION_DEADLINE"
    assert diagnostics[0]["phase"] == "PREDISPATCH_SETUP"
    assert diagnostics[0]["cause"] == "EXTRACTION_DEADLINE_EXPIRED"
    assert diagnostics[0]["provider_cause"] == "UNOBSERVED"
    assert diagnostics[0]["last_progress"] == "NO_PROVIDER_INVOCATION"
    assert diagnostics[0]["termination"] == "NO_PROVIDER_TASK"
    assert diagnostics[0]["deadline_at"].endswith("Z")


def test_outer_deadline_retains_causal_leaf_and_attempt_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real
    from newsroom.graphiti_adapter.cli_client import run_cli_chain

    async def cancelled_chain(**values: object) -> object:
        async def pending_cursor(_prompt: str, *, max_tokens: int) -> str:
            await asyncio.Event().wait()
            raise AssertionError("cancelled provider unexpectedly resumed")

        await run_cli_chain(
            prompt="provider-free prompt",
            schema=None,
            cursor_runner=pending_cursor,
            grok_runner=lambda *_args, **_values: pytest.fail(
                "cancelled primary reached fallback"
            ),
            invocations=values["telemetry"].chat_invocations,
        )
        raise AssertionError("cancelled chain unexpectedly completed")

    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "key")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "password")
    monkeypatch.setattr(real, "_add_episode", cancelled_chain)

    started_at = UtcTimestamp.parse("2026-08-20T00:00:00.000000Z")
    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        started_at,
        execution_deadline=time.monotonic() + 0.05,
        execution_deadline_at=datetime(2026, 8, 20, 0, 0, 0, 50_000, tzinfo=UTC),
    )

    receipt = produced.attempt_receipt_value
    assert receipt is not None
    assert produced.failure_code is ExtractionFailureCode.EXECUTION_TIMEOUT
    assert receipt["chat_invocations"][0]["outcome"] == "CANCELLED"
    leaf_diagnostic = receipt["chat_invocations"][0]["transport_diagnostic"]
    assert leaf_diagnostic["boundary"] == "CALLER_CANCELLATION"
    assert leaf_diagnostic["phase"] == "PRIMARY_TRANSPORT"
    attempt_diagnostic = receipt["timeout_diagnostics"][-1]
    assert attempt_diagnostic["boundary"] == "EXTRACTION_DEADLINE"
    assert attempt_diagnostic["phase"] == "EXTRACTION"
    assert attempt_diagnostic["last_progress"] == "CANCELLED"
    assert attempt_diagnostic["termination"] == "TASK_CANCELLED"
    assert attempt_diagnostic["deadline_at"] == "2026-08-20T00:00:00.050000Z"


def test_connection_cleanup_timeout_is_durable_in_attempt_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real
    from newsroom.graphiti_adapter.cli_process import timeout_diagnostic

    diagnostic = timeout_diagnostic(
        boundary="CLEANUP_DEADLINE",
        phase="CONNECTION_CLEANUP",
        cause="CLEANUP_DEADLINE_EXPIRED",
        configured_timeout_ms=10_000,
        elapsed_ms=10_000,
        deadline_at="2026-08-20T00:00:10.000000Z",
        last_progress="CONNECTION_CLOSE_INCOMPLETE",
        termination="TASK_CANCELLED",
    )

    async def cleanup_timeout(**values: object) -> object:
        values["telemetry"].timeout_diagnostics.append(diagnostic)
        raise real.GraphitiCleanupTimeout(
            "Graphiti connection cleanup timed out",
            evidence=diagnostic,
        )

    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "key")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "password")
    monkeypatch.setattr(real, "_add_episode", cleanup_timeout)
    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )

    receipt = produced.attempt_receipt_value
    assert receipt is not None
    assert produced.failure_code is ExtractionFailureCode.PRODUCER_INTERNAL_ERROR
    assert receipt["producer_failure"] == "GraphitiCleanupTimeout"
    assert receipt["timeout_diagnostics"] == [diagnostic]


def test_public_execute_honours_expired_absolute_rights_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import newsroom.graphiti_adapter.real as real

    fixed = UtcTimestamp.parse("2026-08-21T00:03:00.000000Z")
    provider_calls = 0

    async def must_not_dispatch(**_values: object) -> object:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("expired absolute deadline reached provider")

    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "key")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "password")
    monkeypatch.setattr(real, "_add_episode", must_not_dispatch)
    execution = RealGraphitiAdapter(
        clock=lambda: fixed,
        execution_deadline=fixed.value,
    ).execute(
        attempt=evaluation_attempt_for(("A retained source passage.",)),
        workspace_root=tmp_path / "expired-workspace",
    )

    assert provider_calls == 0
    assert execution.outcome.value == "TIMEOUT"
    assert execution.produced.failure_code is ExtractionFailureCode.EXECUTION_TIMEOUT
    assert list((tmp_path / "expired-workspace").iterdir()) == []


def test_relations_without_exact_evidence_are_retained_without_proposals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real

    async def relation_without_evidence(**values: object) -> object:
        values["telemetry"].embedding_usage = {
            "usage_basis": "NO_EMBEDDING_CALL",
            "request_count": 0,
            "embedding_tokens": 0,
            "cost_usd_microunits": 0,
            "requests": [],
        }
        result = SimpleNamespace(
            episode=SimpleNamespace(uuid=values["episode_id"]),
            nodes=(
                SimpleNamespace(uuid="node-a", name="Absent A", summary="A"),
                SimpleNamespace(uuid="node-b", name="Absent B", summary="B"),
            ),
            edges=(
                SimpleNamespace(
                    uuid="edge-1",
                    name="ABOUT_EVENT",
                    fact="This exact fact is absent from the retained passage.",
                    source_node_uuid="node-a",
                    target_node_uuid="node-b",
                    valid_at=None,
                    invalid_at=None,
                    expired_at=None,
                ),
            ),
        )
        values["validate_result"](result, values["telemetry"])
        return result

    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "key")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "password")
    monkeypatch.setattr(real, "_add_episode", relation_without_evidence)
    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )
    assert produced.outcome is ExtractionOutcome.SUCCESS
    assert produced.proposals == ()
    assert produced.raw_output_value is not None
    assert produced.raw_output_value["relations"][0]["proposal_status"] == (
        "HELD_NO_EXACT_EVIDENCE"
    )


def test_true_empty_graphiti_extraction_is_a_valid_zero_proposal_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real

    async def empty_graph(**values: object) -> object:
        values["telemetry"].embedding_usage = {
            "usage_basis": "NO_EMBEDDING_CALL",
            "request_count": 0,
            "embedding_tokens": 0,
            "cost_usd_microunits": 0,
            "requests": [],
        }
        result = SimpleNamespace(
            episode=SimpleNamespace(uuid=values["episode_id"]),
            nodes=(),
            edges=(),
        )
        values["validate_result"](result, values["telemetry"])
        return result

    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "key")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "password")
    monkeypatch.setattr(real, "_add_episode", empty_graph)
    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )
    assert produced.outcome is ExtractionOutcome.SUCCESS
    assert produced.proposals == ()
    assert produced.raw_output_value is not None
    assert produced.raw_output_value["entity_count"] == 0
    assert produced.raw_output_value["relation_count"] == 0


def test_success_over_fixed_provider_budget_is_retained_as_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real

    async def over_budget(**values: object) -> object:
        values["telemetry"].embedding_usage = {
            "usage_basis": "PROVIDER_REPORTED",
            "request_count": 1,
            "embedding_tokens": 1,
            "cost_usd_microunits": 500_001,
            "requests": [],
        }
        result = SimpleNamespace(
            episode=SimpleNamespace(uuid=values["episode_id"]),
            nodes=(),
            edges=(),
        )
        values["validate_result"](result, values["telemetry"])
        return result

    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "key")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "password")
    monkeypatch.setattr(real, "_add_episode", over_budget)
    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )
    assert produced.outcome is ExtractionOutcome.INVALID_OUTPUT
    assert produced.failure_code is ExtractionFailureCode.OUTPUT_SCHEMA_INVALID
    assert produced.proposals == ()
    assert produced.raw_output_value is not None
    assert produced.raw_output_value["budget_status"] == "EXCEEDED"


def test_real_adapter_retains_truthful_predispatch_refusal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real

    async def predispatch_refusal(**values: object) -> object:
        telemetry = values["telemetry"]
        telemetry.chat_invocations = [
            {
                "provider": "cursor-agent-cli",
                "outcome": "PREDISPATCH_REFUSED",
                "usage": {
                    "usage_basis": "NO_PROVIDER_CALL",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_read_tokens": 0,
                    "cached_write_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                },
            }
        ]
        telemetry.embedding_usage = real._no_embedding_usage()
        raise RuntimeError("qualified Cursor configuration refused")

    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "key")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "password")
    monkeypatch.setattr(real, "_add_episode", predispatch_refusal)

    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )

    assert produced.outcome is ExtractionOutcome.RETRYABLE_FAILURE
    assert produced.failure_code is ExtractionFailureCode.PRODUCER_INTERNAL_ERROR
    assert produced.attempt_receipt_value is not None
    assert produced.attempt_receipt_value["dispatch_state"] == "NOT_DISPATCHED"
    assert produced.attempt_receipt_value["producer_failure"] == "RuntimeError"









def test_grok_cli_runs_outside_repository_cwd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from newsroom.graphiti_adapter import cli_client

    observed: dict[str, object] = {}

    def capture_grok(
        command: tuple[str, ...],
        *,
        timeout: int,
        cwd: str | None = None,
        environment: dict[str, str] | None = None,
        max_output_bytes: int = 0,
    ) -> str:
        observed.update(
            command=command,
            timeout=timeout,
            cwd=cwd,
            environment=environment,
            max_output_bytes=max_output_bytes,
            inventory=([] if cwd is None else list(Path(cwd).iterdir())),
            config=(
                None
                if environment is None
                else (
                    Path(environment["HOME"]) / ".grok" / "config.toml"
                ).read_text(encoding="utf-8")
            ),
        )
        return "{}"

    monkeypatch.setattr(cli_client, "run_cli", capture_grok)
    monkeypatch.setattr(cli_client, "_prove_cli_controls", lambda **_values: None)
    monkeypatch.setattr(
        cli_client,
        "_prepare_grok_resume_session",
        lambda _workspace: "fixture-session",
    )
    monkeypatch.setattr(cli_client, "GROK_AUTH_PATH", "/fixture/grok/auth.json")
    grok_execution = cli_client.run_grok_llm("untrusted source", None, max_tokens=512)
    assert grok_execution.text == "{}"
    assert grok_execution.usage["usage_basis"] == "UNREPORTED"
    grok_cwd = observed["cwd"]
    assert isinstance(grok_cwd, str)
    assert Path(grok_cwd) != _REPOSITORY_ROOT
    assert "newsroom-grok-graphiti-" in grok_cwd
    assert observed["timeout"] == cli_client.CLI_CALL_TIMEOUT_SECONDS
    assert "--max-output-tokens" not in observed["command"]
    assert observed["config"] == (
        "[features]\n"
        "title_refresh = false\n"
        "turn_summary = false\n"
        "[models]\n"
        "max_retries = 0\n"
        '[model."grok-4.6"]\n'
        "max_completion_tokens = 512\n"
        "max_retries = 0\n"
    )
    assert observed["environment"]["GROK_AUTH_PATH"] == (
        "/fixture/grok/auth.json"
    )
    assert observed["max_output_bytes"] == cli_client.grok_stdout_limit(512)
    assert observed["inventory"] == []
    assert observed["command"][-2:] == ("--resume", "fixture-session")


def test_subscription_cli_deadline_reserves_only_cleanup_budget() -> None:
    from newsroom.graphiti_adapter import cli_client
    from newsroom.graphiti_adapter.evaluation_packet import (
        GRAPHITI_EXTRACTION_TIMEOUT_MS,
        GRAPHITI_MAX_CLEANUP_TIMEOUT_MS,
    )

    assert cli_client.CLI_CALL_TIMEOUT_SECONDS * 1_000 == (
        GRAPHITI_EXTRACTION_TIMEOUT_MS - GRAPHITI_MAX_CLEANUP_TIMEOUT_MS
    )


def test_async_cli_child_is_terminated_when_attempt_deadline_cancels() -> None:
    from newsroom.graphiti_adapter.cli_client import run_cli_async

    async def cancelled_call() -> str:
        return await asyncio.wait_for(
            run_cli_async(
                (
                    sys.executable,
                    "-c",
                    "import time; time.sleep(10)",
                ),
                timeout=5,
            ),
            timeout=0.05,
        )

    with pytest.raises(TimeoutError):
        asyncio.run(cancelled_call())
































def test_controller_output_bound_terminates_oversized_cursor_output(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        cli_process.CliOutputBoundExceeded, match="output byte limit"
    ):
        cli_process.run_bounded_process(
            (
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'x' * 131072)",
            ),
            timeout=5,
            max_output_bytes=1024,
            cwd=str(tmp_path),
            environment={},
        )


def test_async_controller_output_bound_drains_both_child_streams(
    tmp_path: Path,
) -> None:
    script = (
        "import sys; "
        "sys.stdout.buffer.write(b'x' * 200000); sys.stdout.flush(); "
        "sys.stderr.buffer.write(b'y' * 200000); sys.stderr.flush()"
    )

    async def invoke() -> None:
        await cli_process.run_bounded_process_async(
            (sys.executable, "-c", script),
            timeout=5,
            max_output_bytes=1_024,
            cwd=str(tmp_path),
            environment={},
        )

    with pytest.raises(cli_process.CliOutputBoundExceeded):
        asyncio.run(asyncio.wait_for(invoke(), timeout=2))


def _descendant_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = subprocess.run(
        ("ps", "-o", "stat=", "-p", str(pid)),
        check=False,
        capture_output=True,
        text=True,
    )
    state = status.stdout.strip()
    return status.returncode == 0 and bool(state) and not state.startswith("Z")


@pytest.mark.parametrize("boundary", ("OUTPUT_LIMIT", "TIMEOUT"))
def test_sync_controller_terminates_descendants_that_inherit_pipes(
    tmp_path: Path,
    boundary: str,
) -> None:
    pid_path = tmp_path / f"sync-{boundary.lower()}-descendant.pid"
    output = (
        "sys.stdout.buffer.write(b'x' * 200000); sys.stdout.flush(); "
        if boundary == "OUTPUT_LIMIT"
        else ""
    )
    script = (
        "import pathlib,subprocess,sys,time; "
        "descendant=subprocess.Popen((sys.executable,'-c',"
        "'import time; time.sleep(30)')); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(descendant.pid)); "
        f"{output}"
        "time.sleep(30)"
    )
    expected = (
        cli_process.CliOutputBoundExceeded
        if boundary == "OUTPUT_LIMIT"
        else cli_process.CliTransportTimeout
    )
    descendant_pid: int | None = None
    try:
        with pytest.raises(expected):
            cli_process.run_bounded_process(
                (sys.executable, "-c", script),
                timeout=0.5,
                max_output_bytes=(1_024 if boundary == "OUTPUT_LIMIT" else 1_000_000),
                cwd=str(tmp_path),
                environment={},
            )
        descendant_pid = int(pid_path.read_text())
        deadline = time.monotonic() + 1
        while _descendant_is_running(descendant_pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _descendant_is_running(descendant_pid)
    finally:
        if descendant_pid is not None and _descendant_is_running(descendant_pid):
            os.kill(descendant_pid, signal.SIGKILL)


@pytest.mark.parametrize("boundary", ("OUTPUT_LIMIT", "TIMEOUT", "CANCELLATION"))
def test_async_controller_terminates_descendants_that_inherit_pipes(
    tmp_path: Path,
    boundary: str,
) -> None:
    pid_path = tmp_path / f"{boundary.lower()}-descendant.pid"
    output = (
        "sys.stdout.buffer.write(b'x' * 200000); sys.stdout.flush(); "
        if boundary == "OUTPUT_LIMIT"
        else ""
    )
    script = (
        "import pathlib,subprocess,sys,time; "
        "descendant=subprocess.Popen((sys.executable,'-c',"
        "'import time; time.sleep(30)')); "
        # Publish readiness atomically: cancellation must not observe an empty
        # file between open/truncate and write/close of the descendant PID.
        f"pathlib.Path({str(pid_path.with_suffix('.tmp'))!r}).write_text(str(descendant.pid)); "
        f"pathlib.Path({str(pid_path.with_suffix('.tmp'))!r}).replace({str(pid_path)!r}); "
        f"{output}"
        "time.sleep(30)"
    )

    async def invoke() -> None:
        task = asyncio.create_task(
            cli_process.run_bounded_process_async(
                (sys.executable, "-c", script),
                timeout=0.5 if boundary == "TIMEOUT" else 5,
                max_output_bytes=(1_024 if boundary == "OUTPUT_LIMIT" else 1_000_000),
                cwd=str(tmp_path),
                environment={},
            )
        )
        if boundary == "CANCELLATION":
            deadline = asyncio.get_running_loop().time() + 1
            while not pid_path.exists():
                if asyncio.get_running_loop().time() >= deadline:
                    pytest.fail("descendant did not start before cancellation")
                await asyncio.sleep(0.01)
            task.cancel()
        await task

    expected = {
        "OUTPUT_LIMIT": cli_process.CliOutputBoundExceeded,
        "TIMEOUT": cli_process.CliTransportTimeout,
        "CANCELLATION": asyncio.CancelledError,
    }[boundary]
    descendant_pid: int | None = None
    try:
        with pytest.raises(expected):
            asyncio.run(asyncio.wait_for(invoke(), timeout=2))
        descendant_pid = int(pid_path.read_text())
        deadline = time.monotonic() + 1
        while _descendant_is_running(descendant_pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _descendant_is_running(descendant_pid)
    finally:
        if descendant_pid is not None and _descendant_is_running(descendant_pid):
            os.kill(descendant_pid, signal.SIGKILL)


def test_async_process_cleanup_drain_has_a_hard_deadline() -> None:
    from newsroom.graphiti_adapter.cli_process import stop_process_async

    killed = False

    class Process:
        returncode: int | None = None

        def kill(self) -> None:
            nonlocal killed
            killed = True
            self.returncode = -9

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.Future()
            raise AssertionError("cleanup drain resumed")

    async def invoke() -> str:
        return await stop_process_async(  # type: ignore[arg-type]
            Process(),
            cleanup_timeout=0.01,
        )

    termination = asyncio.run(asyncio.wait_for(invoke(), timeout=0.5))
    assert killed is True
    assert termination == "PROCESS_CLEANUP_TIMEOUT"


@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("failure", ("NONZERO", "MALFORMED_UTF8"))
def test_cli_result_failure_cleans_descendants_that_closed_inherited_pipes(
    tmp_path: Path,
    asynchronous: bool,
    failure: str,
) -> None:
    from newsroom.graphiti_adapter.cli_client import (
        CliOutputDecodeError,
        run_cli,
        run_cli_async,
    )

    pid_path = tmp_path / f"result-{failure.lower()}-{asynchronous}.pid"
    result = (
        "sys.stdout.buffer.write(b'{}'); sys.stdout.flush(); sys.exit(7)"
        if failure == "NONZERO"
        else "sys.stdout.buffer.write(b'\\xff'); sys.stdout.flush()"
    )
    script = (
        "import pathlib,subprocess,sys; "
        "descendant=subprocess.Popen((sys.executable,'-c',"
        "'import os,time; os.close(1); os.close(2); time.sleep(30)')); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(descendant.pid)); "
        f"{result}"
    )
    command = (sys.executable, "-c", script)
    expected = RuntimeError if failure == "NONZERO" else CliOutputDecodeError
    descendant_pid: int | None = None
    try:
        with pytest.raises(expected):
            if asynchronous:
                asyncio.run(run_cli_async(command, timeout=5, cwd=str(tmp_path)))
            else:
                run_cli(command, timeout=5, cwd=str(tmp_path))
        descendant_pid = int(pid_path.read_text())
        deadline = time.monotonic() + 1
        while _descendant_is_running(descendant_pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _descendant_is_running(descendant_pid)
    finally:
        if descendant_pid is not None and _descendant_is_running(descendant_pid):
            os.kill(descendant_pid, signal.SIGKILL)


@pytest.mark.parametrize("boundary", ("TIMEOUT", "CANCELLATION"))
def test_async_controller_drains_high_volume_pipes_on_termination(
    tmp_path: Path,
    boundary: str,
) -> None:
    script = (
        "import sys,time; "
        "sys.stdout.buffer.write(b'x' * 200000); sys.stdout.flush(); "
        "sys.stderr.buffer.write(b'y' * 200000); sys.stderr.flush(); "
        "time.sleep(5)"
    )

    async def invoke() -> None:
        task = asyncio.create_task(
            cli_process.run_bounded_process_async(
                (sys.executable, "-c", script),
                timeout=0.05 if boundary == "TIMEOUT" else 5,
                max_output_bytes=1_000_000,
                cwd=str(tmp_path),
                environment={},
            )
        )
        if boundary == "CANCELLATION":
            await asyncio.sleep(0.05)
            task.cancel()
        await task

    expected = (
        cli_process.CliTransportTimeout
        if boundary == "TIMEOUT"
        else asyncio.CancelledError
    )
    with pytest.raises(expected):
        asyncio.run(asyncio.wait_for(invoke(), timeout=2))


def test_transport_modules_import_in_a_fresh_interpreter() -> None:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "import newsroom.graphiti_adapter.cli_process; "
            "import newsroom.graphiti_adapter.cursor_transport",
        ),
        check=False,
        capture_output=True,
        text=True,
        cwd=repository,
    )

    assert completed.returncode == 0, completed.stderr


def test_controller_timeout_retains_secret_free_transport_diagnostics(
    tmp_path: Path,
) -> None:
    stdout = b"partial stdout\n"
    stderr = b"partial stderr\n"
    script = (
        "import sys,time; "
        f"sys.stdout.buffer.write({stdout!r}); sys.stdout.flush(); "
        f"sys.stderr.buffer.write({stderr!r}); sys.stderr.flush(); "
        "time.sleep(5)"
    )
    with pytest.raises(cli_process.CliTransportTimeout) as caught:
        cli_process.run_bounded_process(
            (sys.executable, "-c", script),
            timeout=0.05,
            max_output_bytes=1_024,
            cwd=str(tmp_path),
            environment={},
            phase="PRIMARY_TRANSPORT",
        )

    evidence = caught.value.evidence
    assert evidence == {
        "schema_version": "newsroom.graphiti-timeout-diagnostic.v1",
        "boundary": "CONTROLLER_DEADLINE",
        "phase": "PRIMARY_TRANSPORT",
        "cause": "CONFIGURED_TIMEOUT_EXPIRED",
        "provider_cause": "UNOBSERVED",
        "process": "CLI_CHILD",
        "configured_timeout_ms": 50,
        "elapsed_ms": evidence["elapsed_ms"],
        "deadline_at": evidence["deadline_at"],
        "last_progress": "OUTPUT_OBSERVED",
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
        "stdout_digest": digest_bytes(stdout),
        "stderr_digest": digest_bytes(stderr),
        "termination": "PROCESS_KILLED",
    }
    assert "partial stdout" not in repr(evidence)
    assert "partial stderr" not in repr(evidence)
    assert evidence["elapsed_ms"] >= 50
    assert evidence["deadline_at"].endswith("Z")


def test_timeout_diagnostic_rejects_secret_in_causal_token() -> None:
    from newsroom.graphiti_adapter.cli_process import timeout_diagnostic

    with pytest.raises(ValueError, match="last_progress"):
        timeout_diagnostic(
            boundary="CONTROLLER_DEADLINE",
            phase="PRIMARY_TRANSPORT",
            cause="CONFIGURED_TIMEOUT_EXPIRED",
            configured_timeout_ms=160_000,
            elapsed_ms=160_000,
            deadline_at="2026-08-26T18:00:20.000000Z",
            last_progress="TOKEN=secret-provider-credential",
            termination="PROCESS_KILLED",
        )


def test_async_controller_timeout_retains_transport_diagnostics(
    tmp_path: Path,
) -> None:
    stdout = b"async stdout\n"
    stderr = b"async stderr\n"
    script = (
        "import sys,time; "
        f"sys.stdout.buffer.write({stdout!r}); sys.stdout.flush(); "
        f"sys.stderr.buffer.write({stderr!r}); sys.stderr.flush(); "
        "time.sleep(5)"
    )

    async def invoke() -> None:
        await cli_process.run_bounded_process_async(
            (sys.executable, "-c", script),
            timeout=0.05,
            max_output_bytes=1_024,
            cwd=str(tmp_path),
            environment={},
            phase="PRIMARY_TRANSPORT",
        )

    with pytest.raises(cli_process.CliTransportTimeout) as caught:
        asyncio.run(invoke())

    evidence = caught.value.evidence
    assert evidence["configured_timeout_ms"] == 50
    assert evidence["stdout_bytes"] == len(stdout)
    assert evidence["stderr_bytes"] == len(stderr)
    assert evidence["stdout_digest"] == digest_bytes(stdout)
    assert evidence["stderr_digest"] == digest_bytes(stderr)


def test_controller_timeout_survives_process_exit_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_kill = os.killpg

    def exits_during_kill(process_group_id: int, sent_signal: int) -> None:
        original_kill(process_group_id, sent_signal)
        raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", exits_during_kill)
    with pytest.raises(cli_process.CliTransportTimeout) as caught:
        cli_process.run_bounded_process(
            (sys.executable, "-c", "import time; time.sleep(5)"),
            timeout=0.05,
            max_output_bytes=1_024,
            cwd=str(tmp_path),
            environment={},
            phase="PRIMARY_TRANSPORT",
        )

    assert caught.value.evidence["termination"] == "PROCESS_EXIT_RACE"




def test_grok_preflight_timeout_retains_causal_qualification_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from newsroom.graphiti_adapter import cli_client
    from newsroom.graphiti_adapter.cli_process import (
        CliTransportTimeout,
        timeout_diagnostic,
    )

    diagnostic = timeout_diagnostic(
        boundary="CONTROLLER_DEADLINE",
        phase="PREDISPATCH_HELP",
        cause="CONFIGURED_TIMEOUT_EXPIRED",
        configured_timeout_ms=20_000,
        elapsed_ms=20_000,
        deadline_at="2026-08-26T18:00:20.000000Z",
        last_progress="NO_OUTPUT_OBSERVED",
        termination="PROCESS_KILLED",
        process="CLI_CHILD",
        stdout=b"",
        stderr=b"",
    )

    def timeout(*_args: object, **_values: object) -> object:
        raise CliTransportTimeout("grok Graphiti LLM timed out", evidence=diagnostic)

    monkeypatch.setattr(cli_client, "run_bounded_process", timeout)
    workspace = cli_client._GraphitiCliWorkspace(
        cwd=str(tmp_path),
        request_dir=str(tmp_path),
        environment={},
    )
    with pytest.raises(cli_client.CliPredispatchRefusal) as caught:
        cli_client._prove_cli_controls(
            binary="grok",
            required_controls=("--max-output-tokens",),
            workspace=workspace,
        )

    assert caught.value.qualification_evidence["timeout_diagnostic"] == diagnostic


def test_async_grok_preflight_timeout_retains_causal_qualification_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from newsroom.graphiti_adapter import cli_client
    from newsroom.graphiti_adapter.cli_process import (
        CliTransportTimeout,
        timeout_diagnostic,
    )

    diagnostic = timeout_diagnostic(
        boundary="CONTROLLER_DEADLINE",
        phase="PREDISPATCH_HELP",
        cause="CONFIGURED_TIMEOUT_EXPIRED",
        configured_timeout_ms=20_000,
        elapsed_ms=20_000,
        deadline_at="2026-08-26T18:00:20.000000Z",
        last_progress="NO_OUTPUT_OBSERVED",
        termination="PROCESS_KILLED",
    )

    async def timeout(*_args: object, **_values: object) -> object:
        raise CliTransportTimeout("grok Graphiti LLM timed out", evidence=diagnostic)

    monkeypatch.setattr(cli_client, "run_bounded_process_async", timeout)
    workspace = cli_client._GraphitiCliWorkspace(
        cwd=str(tmp_path),
        request_dir=str(tmp_path),
        environment={},
    )
    with pytest.raises(cli_client.CliPredispatchRefusal) as caught:
        asyncio.run(
            cli_client._prove_cli_controls_async(
                binary="grok",
                required_controls=("--max-output-tokens",),
                workspace=workspace,
            )
        )

    assert caught.value.qualification_evidence["timeout_diagnostic"] == diagnostic


@pytest.mark.parametrize("asynchronous", (False, True))
def test_fallback_timeout_retains_causal_diagnostics(
    tmp_path: Path, asynchronous: bool
) -> None:
    from newsroom.graphiti_adapter.cli_client import (
        CliTransportTimeout,
        run_cli,
        run_cli_async,
    )

    stdout = b"fallback stdout\n"
    stderr = b"fallback stderr\n"
    script = (
        "import sys,time; "
        f"sys.stdout.buffer.write({stdout!r}); sys.stdout.flush(); "
        f"sys.stderr.buffer.write({stderr!r}); sys.stderr.flush(); "
        "time.sleep(5)"
    )
    command = (sys.executable, "-c", script)
    with pytest.raises(CliTransportTimeout) as caught:
        if asynchronous:
            asyncio.run(run_cli_async(command, timeout=0.05, cwd=str(tmp_path)))
        else:
            run_cli(command, timeout=0.05, cwd=str(tmp_path))

    evidence = caught.value.evidence
    assert evidence["boundary"] == "CONTROLLER_DEADLINE"
    assert evidence["phase"] == "FALLBACK_TRANSPORT"
    assert evidence["cause"] == "CONFIGURED_TIMEOUT_EXPIRED"
    assert evidence["provider_cause"] == "UNOBSERVED"
    assert evidence["last_progress"] == "OUTPUT_OBSERVED"
    assert evidence["stdout_bytes"] == len(stdout)
    assert evidence["stderr_bytes"] == len(stderr)
    assert evidence["stdout_digest"] == digest_bytes(stdout)
    assert evidence["stderr_digest"] == digest_bytes(stderr)


def test_sync_fallback_timeout_survives_process_exit_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from newsroom.graphiti_adapter.cli_client import run_cli
    from newsroom.graphiti_adapter.cli_process import CliTransportTimeout

    original_kill = os.killpg

    def exits_during_kill(process_group_id: int, sent_signal: int) -> None:
        original_kill(process_group_id, sent_signal)
        raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", exits_during_kill)
    with pytest.raises(CliTransportTimeout) as caught:
        run_cli(
            (sys.executable, "-c", "import time; time.sleep(5)"),
            timeout=0.05,
            cwd=str(tmp_path),
        )

    assert caught.value.evidence["termination"] == "PROCESS_EXIT_RACE"


@pytest.mark.parametrize("identity", (None, ("retained-episode", "retained-episode", 1)))
def test_owned_compensation_uses_only_bare_driver_and_exact_retained_guard(
    monkeypatch, identity,
):
    import newsroom.graphiti_adapter.real as real

    events = []
    driver = SimpleNamespace(close=AsyncMock())
    now = datetime(2026, 9, 26, tzinfo=UTC)
    digest = digest_canonical({"retained": "exact original input"})
    marker = GuardMarker(GuardState.RECOVERED_AMBIGUOUS, 1, digest)
    active = False

    class Guard:
        @staticmethod
        async def owned_pending_identity(observed_driver, *, group_id):
            assert observed_driver is driver
            assert group_id == GRAPHITI_WORKSPACE_GROUP
            events.append("identity")
            return identity

        def __init__(self, observed_driver, **values):
            assert active
            assert observed_driver is driver
            assert values == dict(
                group_id=GRAPHITI_WORKSPACE_GROUP,
                episode_uuid="retained-episode", marker_episode_uuid="retained-episode",
                attempt_number=1, input_digest=digest,
            )
            events.append("guard")

        async def recover_owned_pending(self, *, owner_stop_check, phase_observer=None):
            assert active
            owner_stop_check()
            events.append("compensate")
            return marker

    @contextmanager
    def authenticated_input(episode_uuid, attempt_number):
        nonlocal active
        assert (episode_uuid, attempt_number) == ("retained-episode", 1)
        active = True
        events.append("fence")
        try:
            yield digest
        finally:
            active = False
            events.append("unfence")

    monkeypatch.setattr(real, "_open_compensation_driver", lambda: driver, raising=False)
    monkeypatch.setattr(real, "Neo4jMutationGuard", Guard)
    for name in ("_load_graphiti", "openrouter_api_key", "build_cli_llm_client", "_bootstrap_graphiti_schema"):
        monkeypatch.setattr(real, name, lambda *a, **kw: pytest.fail("provider/setup boundary"))
    result = real.recover_owned_pending(
        authenticated_input=authenticated_input,
        owner_stop_check=lambda: events.append("stop"),
        deadline=now + timedelta(seconds=180), clock=lambda: now,
    )
    assert result is (marker if identity is not None else None)
    assert [event for event in events if event != "stop"] == (
        ["identity", "fence", "guard", "compensate", "unfence"]
        if identity is not None else ["identity"]
    )
    driver.close.assert_awaited_once()
    assert not active


@pytest.mark.parametrize("boundary", ("input_missing", "stop", "timeout", "expired"))
def test_owned_compensation_preserves_pending_state_when_boundary_refuses(monkeypatch, boundary):
    import newsroom.graphiti_adapter.real as real
    from newsroom.control_plane.veto import VetoError

    now = datetime(2026, 9, 26, tzinfo=UTC)
    events = []
    driver = SimpleNamespace(close=AsyncMock())

    class Guard:
        @staticmethod
        async def owned_pending_identity(*_args, **_values):
            events.append("identity")
            return "episode", "episode", 1

        def __init__(self, *_args, **_values):
            events.append("guard")

        async def recover_owned_pending(self, *, owner_stop_check, phase_observer=None):
            owner_stop_check()
            events.append("compensate")
            await asyncio.sleep(1)
            pytest.fail("timeout must leave durable pending recovery")

    @contextmanager
    def authenticated_input(*_args):
        events.append("fence")
        try:
            yield None if boundary == "input_missing" else digest_canonical({"input": 1})
        finally:
            events.append("unfence")

    def stop():
        if boundary == "stop" and "identity" in events:
            raise VetoError("owner emergency stop")

    monkeypatch.setattr(real, "_open_compensation_driver", lambda: events.append("driver") or driver, raising=False)
    monkeypatch.setattr(real, "Neo4jMutationGuard", Guard)
    if boundary == "stop":
        with pytest.raises(VetoError):
            real.recover_owned_pending(
                authenticated_input=authenticated_input, owner_stop_check=stop,
                deadline=now + timedelta(seconds=180), clock=lambda: now,
            )
    else:
        result = real.recover_owned_pending(
            authenticated_input=authenticated_input, owner_stop_check=stop,
            deadline=now + timedelta(seconds=0.01 if boundary == "timeout" else -1 if boundary == "expired" else 180),
            clock=lambda: now,
        )
        assert result is None
    if boundary == "expired":
        assert not events
        driver.close.assert_not_awaited()
    else:
        driver.close.assert_awaited_once()
    assert events.count("fence") == events.count("unfence")
    assert events.count("compensate") == int(boundary == "timeout")


def test_owned_compensation_driver_never_schedules_schema_or_provider_setup(monkeypatch):
    import newsroom.graphiti_adapter.real as real

    events = []
    class Driver:
        def __init__(self, uri, user, password):
            with pytest.raises(RuntimeError, match="no running event loop"):
                asyncio.get_running_loop()
            assert (uri, user, password) == (f"bolt://{real.NEO4J_BOLT_HOST}:{real.NEO4J_BOLT_PORT}", "neo4j", "fixture")
            events.append("driver")

    monkeypatch.setitem(sys.modules, "graphiti_core.driver.neo4j_driver", SimpleNamespace(Neo4jDriver=Driver))
    monkeypatch.setattr(real.importlib.metadata, "version", lambda _: "0.29.3")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "fixture")
    monkeypatch.setattr(real, "_load_graphiti", lambda: pytest.fail("provider runtime"))
    assert isinstance(real._open_compensation_driver(), Driver)

    async def forbidden_async_boundary():
        with pytest.raises(GraphitiAdapterContractError, match="synchronous boundary"):
            real._open_compensation_driver()
    asyncio.run(forbidden_async_boundary())
    assert events == ["driver"]


@pytest.mark.parametrize("recovery", ("no_owner", "recovered", "owner_stop"))
def test_owned_compensation_bounds_never_finishing_driver_close_without_changing_result(
    monkeypatch, recovery,
):
    import newsroom.graphiti_adapter.real as real
    from newsroom.control_plane.veto import VetoError

    now = datetime(2026, 9, 26, tzinfo=UTC)
    digest = digest_canonical({"retained": "input"})
    marker = GuardMarker(GuardState.RECOVERED_AMBIGUOUS, 1, digest)
    events = []

    async def never_close():
        events.append("close")
        try:
            await asyncio.Event().wait()
        finally:
            events.append("close_cancelled")

    class Guard:
        @staticmethod
        async def owned_pending_identity(*_args, **_values):
            events.append("identity")
            return None if recovery == "no_owner" else ("episode", "episode", 1)

        def __init__(self, *_args, **_values):
            pass

        async def recover_owned_pending(self, *, owner_stop_check, phase_observer=None):
            owner_stop_check()
            events.append("recovered")
            return marker

    @contextmanager
    def authenticated_input(*_args):
        yield digest

    def stop():
        if recovery == "owner_stop" and "identity" in events:
            raise VetoError("owner emergency stop")

    # A cancellable outer child bound proves the defect without hanging pytest.
    run = asyncio.run
    monkeypatch.setattr(real.asyncio, "run", lambda child: run(asyncio.wait_for(child, timeout=0.25)))
    monkeypatch.setattr(real, "GRAPHITI_CLEANUP_TIMEOUT_MS", 10)
    monkeypatch.setattr(real, "_open_compensation_driver", lambda: SimpleNamespace(close=never_close))
    monkeypatch.setattr(real, "Neo4jMutationGuard", Guard)
    arguments = dict(
        authenticated_input=authenticated_input, owner_stop_check=stop,
        deadline=now + timedelta(seconds=180), clock=lambda: now,
    )
    if recovery == "owner_stop":
        with pytest.raises(VetoError, match="owner emergency stop"):
            real.recover_owned_pending(**arguments)
    else:
        assert real.recover_owned_pending(**arguments) is (marker if recovery == "recovered" else None)
    assert events.count("close") == events.count("close_cancelled") == 1
    assert events.count("recovered") == int(recovery == "recovered")


@pytest.mark.parametrize("outcome,phase,cause,status", (
    ("timeout", "INVENTORY", "TimeoutError", "TIMEOUT"),
    ("guard", "FULL_VERIFY", "GuardError", "FAILED"),
    ("stop", "RESTORE", "VetoError", "STOPPED"),
    ("recovered", "SNAPSHOT_CLEANUP", "NONE", "RECOVERED"),
    ("refused", "OWNERSHIP", "NONE", "REFUSED"),
    ("cleanup_error", "CONNECTION_CLEANUP", "RuntimeError", "RECOVERED_CLEANUP_FAILED"),
    ("inventory_bound", "INVENTORY", "GuardError", "FAILED"),
    ("inventory_coverage", "INVENTORY", "GuardError", "FAILED"),
    ("inventory_identity", "INVENTORY", "GuardError", "FAILED"),
))
def test_owned_recovery_emits_one_coarse_phase_diagnostic_without_private_content(
    monkeypatch, caplog, outcome, phase, cause, status,
):
    import newsroom.graphiti_adapter.real as real
    from newsroom.control_plane.veto import VetoError
    from newsroom.graphiti_adapter.neo4j_guard import GuardError

    now = datetime(2026, 9, 26, tzinfo=UTC)
    digest = digest_canonical({"retained": "private source payload"})
    driver = SimpleNamespace(close=AsyncMock())
    if outcome == "cleanup_error":
        driver.close.side_effect = RuntimeError("TOKEN=private-credential; cleanup detail")
    active_phase = None

    class Guard:
        @staticmethod
        async def owned_pending_identity(*_args, **_values):
            return "private-episode-identity", "private-marker-identity", 1

        def __init__(self, *_args, **_values):
            pass

        async def recover_owned_pending(self, *, owner_stop_check, phase_observer=None):
            nonlocal active_phase
            active_phase = phase
            if phase_observer is not None:
                phase_observer(phase)
            if outcome == "timeout":
                await asyncio.Event().wait()
            elif outcome == "guard":
                raise GuardError("TOKEN=private-credential; raw source payload; query text")
            elif outcome.startswith("inventory_"):
                raise GuardError({
                    "inventory_bound": "Graphiti identity inventory exceeds its byte bound",
                    "inventory_coverage": "Graphiti identity inventory omits a pre-existing target",
                    "inventory_identity": "Graphiti inventory identity is absent",
                }[outcome])
            elif outcome == "stop":
                owner_stop_check()
            elif outcome == "refused":
                return None
            return GuardMarker(GuardState.RECOVERED_AMBIGUOUS, 1, digest)

    @contextmanager
    def authenticated_input(*_args):
        yield digest

    def stop():
        if outcome == "stop" and active_phase is not None:
            raise VetoError("TOKEN=private-credential; raw source payload")

    monkeypatch.setattr(real, "_open_compensation_driver", lambda: driver)
    monkeypatch.setattr(real, "Neo4jMutationGuard", Guard)
    for name in ("_load_graphiti", "openrouter_api_key", "build_cli_llm_client", "_bootstrap_graphiti_schema"):
        monkeypatch.setattr(real, name, lambda *a, **kw: pytest.fail("provider/setup boundary"))
    caplog.set_level("WARNING", logger=real.__name__)
    arguments = dict(
        authenticated_input=authenticated_input, owner_stop_check=stop,
        deadline=now + timedelta(seconds=0.01 if outcome == "timeout" else 180), clock=lambda: now,
    )
    if outcome in {"guard", "stop", "cleanup_error"} or outcome.startswith("inventory_"):
        with pytest.raises(GuardError if cause == "GuardError" else VetoError if outcome == "stop" else RuntimeError):
            real.recover_owned_pending(**arguments)
    else:
        result = real.recover_owned_pending(**arguments)
        assert (result is not None) == (outcome == "recovered")
    records = [record for record in caplog.records if record.name == real.__name__]
    assert len(records) == 1
    message = records[0].getMessage()
    assert f"status={status}" in message
    assert f"phase={phase}" in message
    assert f"failure_class={cause}" in message
    if cause == "GuardError":
        reason = {
            "inventory_bound": "INVENTORY_BYTE_BOUND",
            "inventory_coverage": "SNAPSHOT_COVERAGE",
            "inventory_identity": "IDENTITY_BINDING",
        }.get(outcome, "GUARD_ERROR")
        assert f"reason_code={reason}" in message
    assert "deadline_ms=" in message and "elapsed_ms=" in message
    assert len(message.encode()) <= 200
    assert records[0].exc_info is None
    assert all(private not in message for private in (
        "private-credential", "private source", "private-episode", "private-marker", "query text", digest,
    ))
    driver.close.assert_awaited_once()


def test_owned_recovery_diagnostic_logs_setup_failure_but_not_no_owner_ticks(monkeypatch, caplog):
    import newsroom.graphiti_adapter.real as real
    from newsroom.control_plane.broker import BrokerError

    now = datetime(2026, 9, 26, tzinfo=UTC)
    arguments = dict(
        authenticated_input=lambda *a: pytest.fail("no selected owner"), owner_stop_check=lambda: None,
        deadline=now + timedelta(seconds=180), clock=lambda: now,
    )
    caplog.set_level("WARNING", logger=real.__name__)
    monkeypatch.setattr(real, "_open_compensation_driver", lambda: (_ for _ in ()).throw(BrokerError("TOKEN=private")))
    with pytest.raises(BrokerError):
        real.recover_owned_pending(**arguments)
    messages = [record.getMessage() for record in caplog.records if record.name == real.__name__]
    assert len(messages) == 1
    assert "phase=SETUP" in messages[0] and "failure_class=BrokerError" in messages[0]
    assert "private" not in messages[0]
    caplog.clear()
    driver = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(real, "_open_compensation_driver", lambda: driver)
    monkeypatch.setattr(real, "Neo4jMutationGuard", SimpleNamespace(owned_pending_identity=AsyncMock(return_value=None)))
    for _ in range(2):
        assert real.recover_owned_pending(**arguments) is None
    assert not caplog.records


@pytest.mark.parametrize("boundary", ("setup", "recovery", "cleanup"))
@pytest.mark.parametrize("recognised", (False, True))
def test_owned_recovery_diagnostics_never_emit_custom_or_oversized_exception_class_names(
    monkeypatch, caplog, boundary, recognised,
):
    import newsroom.graphiti_adapter.real as real
    from newsroom.control_plane.broker import BrokerError
    from newsroom.graphiti_adapter.neo4j_guard import GuardError

    base, safe_name = {
        "setup": (BrokerError, "BrokerError"),
        "recovery": (GuardError, "GuardError"),
        "cleanup": (RuntimeError, "RuntimeError"),
    }[boundary]
    custom = type("TOKEN_PRIVATE_" + "X" * 300, (base if recognised else Exception,), {})
    failure = custom("Graphiti identity inventory exceeds its byte bound")
    now = datetime(2026, 9, 26, tzinfo=UTC)
    digest = digest_canonical({"retained": "input"})
    driver = SimpleNamespace(close=AsyncMock(side_effect=failure if boundary == "cleanup" else None))

    class Guard:
        @staticmethod
        async def owned_pending_identity(*_args, **_values):
            return "episode", "episode", 1

        def __init__(self, *_args, **_values):
            pass

        async def recover_owned_pending(self, *, owner_stop_check, phase_observer):
            phase_observer("INVENTORY")
            if boundary == "recovery":
                raise failure
            return GuardMarker(GuardState.RECOVERED_AMBIGUOUS, 1, digest)

    @contextmanager
    def authenticated_input(*_args):
        yield digest

    def open_driver():
        if boundary == "setup":
            raise failure
        return driver

    monkeypatch.setattr(real, "_open_compensation_driver", open_driver)
    monkeypatch.setattr(real, "Neo4jMutationGuard", Guard)
    caplog.set_level("WARNING", logger=real.__name__)
    with pytest.raises(custom) as caught:
        real.recover_owned_pending(
            authenticated_input=authenticated_input, owner_stop_check=lambda: None,
            deadline=now + timedelta(seconds=180), clock=lambda: now,
        )
    assert caught.value is failure
    records = [record for record in caplog.records if record.name == real.__name__]
    assert len(records) == 1
    message = records[0].getMessage()
    assert f"failure_class={safe_name if recognised else 'OTHER_ERROR'}" in message
    assert "TOKEN_PRIVATE_" not in message and custom.__name__ not in message
    assert len(message.encode()) <= 200 and records[0].exc_info is None
    if boundary == "recovery" and recognised:
        assert "reason_code=INVENTORY_BYTE_BOUND" in message


@pytest.mark.parametrize("outcome", ("recovered", "refused", "stop"))
def test_owned_recovery_logger_failure_preserves_exact_result_or_owner_stop(monkeypatch, outcome):
    import newsroom.graphiti_adapter.real as real
    from newsroom.control_plane.veto import VetoError

    now = datetime(2026, 9, 26, tzinfo=UTC)
    digest = digest_canonical({"retained": "input"})
    marker = GuardMarker(GuardState.RECOVERED_AMBIGUOUS, 1, digest)
    stopped = VetoError("owner emergency stop")
    driver = SimpleNamespace(close=AsyncMock())

    class Guard:
        @staticmethod
        async def owned_pending_identity(*_args, **_values):
            return "episode", "episode", 1

        def __init__(self, *_args, **_values):
            pass

        async def recover_owned_pending(self, *, owner_stop_check, phase_observer):
            phase_observer("RESTORE")
            if outcome == "stop":
                raise stopped
            return marker if outcome == "recovered" else None

    @contextmanager
    def authenticated_input(*_args):
        yield digest

    monkeypatch.setattr(real, "_open_compensation_driver", lambda: driver)
    monkeypatch.setattr(real, "Neo4jMutationGuard", Guard)
    monkeypatch.setattr(real._LOGGER, "warning", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fixture logger failure")))
    arguments = dict(
        authenticated_input=authenticated_input, owner_stop_check=lambda: None,
        deadline=now + timedelta(seconds=180), clock=lambda: now,
    )
    if outcome == "stop":
        with pytest.raises(VetoError) as caught:
            real.recover_owned_pending(**arguments)
        assert caught.value is stopped
    else:
        assert real.recover_owned_pending(**arguments) is (marker if outcome == "recovered" else None)
    driver.close.assert_awaited_once()

@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("logger_fault", [False, True])
@pytest.mark.parametrize("late_rollback", [False, True])
def test_validation_budget_diagnostic_survives_wrapping_without_changing_outcome(
    monkeypatch, caplog, wrapped, logger_fault, late_rollback,
):
    import logging
    import newsroom.graphiti_adapter.real as real
    from newsroom.extraction.types import ExtractionContractError

    caplog.set_level(logging.WARNING, logger=real.__name__)
    async def over_budget(**values):
        values["telemetry"].embedding_usage = {
            "usage_basis": "PROVIDER_REPORTED", "request_count": 1,
            "embedding_tokens": 1, "cost_usd_microunits": 500_001, "requests": [],
        }
        result = SimpleNamespace(episode=SimpleNamespace(uuid=values["episode_id"]), nodes=(), edges=())
        try:
            values["validate_result"](result, values["telemetry"])
        except ExtractionContractError as error:
            if not wrapped:
                raise
            pipeline = real.CombinedTemporalPipelineError(
                "fixture wrapped validation", graph_effect_attempted=True, rollback_completed=not late_rollback,
            )
            if late_rollback:
                values["telemetry"].recovery_classification = real.GraphitiRecoveryClassification.ROLLED_BACK_AMBIGUOUS_EFFECT
            pipeline.__cause__ = error
            raise real.AmbiguousEpisodeEffect("fixture ambiguous") from pipeline
        raise AssertionError("Expected budget failure")
    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "fixture")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "fixture")
    monkeypatch.setattr(real, "_add_episode", over_budget)
    if logger_fault:
        def broken(*args, **kwargs):
            raise RuntimeError("fixture logger failure")
        monkeypatch.setattr(real._LOGGER, "warning", broken)
    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)),
        UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )
    assert produced.outcome is (ExtractionOutcome.RETRYABLE_FAILURE if wrapped else ExtractionOutcome.INVALID_OUTPUT)
    if not logger_fault:
        records = [r.getMessage() for r in caplog.records if "graphiti_validation" in r.getMessage()]
        assert len(records) == 1
        assert "phase=BUDGET" in records[0]
        assert "reason=BUDGET_EXCEEDED" in records[0]
        assert "exceeded=cost_microunits" in records[0]
        assert ("rollback=COMPLETE" if wrapped else "rollback=UNOBSERVED") in records[0]
        assert len(records[0].encode()) <= 300

@pytest.mark.parametrize("private_message", ["PRIVATE_SOURCE_SECRET", "PRIVATE_SOURCE_SECRET" * 1000], ids=["private", "oversized"])
def test_validation_contract_diagnostic_never_emits_private_error_text(monkeypatch, caplog, private_message):
    import logging
    import newsroom.graphiti_adapter.real as real
    from newsroom.extraction.types import ExtractionContractError

    caplog.set_level(logging.WARNING, logger=real.__name__)
    class PRIVATE_SOURCE_SECRET(ExtractionContractError):
        pass
    original = PRIVATE_SOURCE_SECRET(private_message)
    def invalid_proposals(*args):
        raise original
    async def rejected(**values):
        result = SimpleNamespace(episode=SimpleNamespace(uuid=values["episode_id"]), nodes=(), edges=())
        try:
            values["validate_result"](result, values["telemetry"])
        except ExtractionContractError as error:
            assert error is original
            pipeline = real.CombinedTemporalPipelineError("fixture", graph_effect_attempted=True, rollback_completed=False)
            pipeline.__cause__ = error
            raise real.AmbiguousEpisodeEffect("fixture") from pipeline
    monkeypatch.setattr(real, "entity_proposals", invalid_proposals)
    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "fixture")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "fixture")
    monkeypatch.setattr(real, "_add_episode", rejected)
    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)), UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )
    assert produced.failure_code is ExtractionFailureCode.AMBIGUOUS_EFFECT
    records = [r.getMessage() for r in caplog.records if "graphiti_validation" in r.getMessage()]
    assert len(records) == 1 and "phase=ENTITY_PROPOSALS" in records[0] and "reason=CONTRACT_INVALID" in records[0]
    assert "rollback=INCOMPLETE" in records[0]
    assert "PRIVATE_SOURCE_SECRET" not in records[0] and len(records[0].encode()) <= 300

@pytest.mark.parametrize("termination", ["guard", "timeout"])
def test_validation_cause_is_observed_when_secondary_rollback_terminates(monkeypatch, caplog, termination):
    import logging
    import newsroom.graphiti_adapter.real as real
    from newsroom.extraction.types import ExtractionContractError
    caplog.set_level(logging.WARNING, logger=real.__name__)
    def invalid_proposals(*args):
        raise ExtractionContractError("PRIVATE_SOURCE_SECRET")
    async def interrupted(**values):
        result = SimpleNamespace(episode=SimpleNamespace(uuid=values["episode_id"]), nodes=(), edges=())
        try:
            values["validate_result"](result, values["telemetry"])
        except ExtractionContractError:
            if termination == "guard":
                raise real.GuardError("fixture secondary rollback failure")
            raise asyncio.TimeoutError
    monkeypatch.setattr(real, "entity_proposals", invalid_proposals)
    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real, "openrouter_api_key", lambda: "fixture")
    monkeypatch.setattr(real, "neo4j_community_password", lambda: "fixture")
    monkeypatch.setattr(real, "_add_episode", interrupted)
    produced = RealGraphitiAdapter()._produce(
        evaluation_attempt_for(("A retained source passage.",)), UtcTimestamp.parse("2026-08-20T00:00:00.000000Z"),
    )
    assert produced.failure_code is (ExtractionFailureCode.PRODUCER_INTERNAL_ERROR if termination == "guard" else ExtractionFailureCode.EXECUTION_TIMEOUT)
    records = [r.getMessage() for r in caplog.records if "graphiti_validation" in r.getMessage()]
    assert len(records) == 1 and "phase=ENTITY_PROPOSALS" in records[0] and "rollback=UNOBSERVED" in records[0]
    assert "PRIVATE_SOURCE_SECRET" not in records[0]
