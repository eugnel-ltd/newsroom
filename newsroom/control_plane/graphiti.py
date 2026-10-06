"""EVALUATION Graphiti runner for corpus ingest, decoupled from CONT writes."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ContextManager, Protocol, runtime_checkable

from newsroom.authority.extraction_facade import GovernedExtractionRecords
from newsroom.authority.graphiti_adapter_facade import (
    GovernedGraphitiProposalAdapter,
)
from newsroom.authority.auth import AuthenticationProof, AuthorizationDenied
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.control_plane.corpus import CorpusIngestUnit
from newsroom.control_plane.graphiti_fallback_policy import (
    FallbackEligibility,
    GraphitiFallbackCircuitPolicy,
    classify_graphiti_fallback,
    load_checked_graphiti_fallback_circuit_policy,
)
from newsroom.control_plane.graphiti_requests import (
    GraphitiCallShapePolicy,
    GraphitiInternalRequestIdentity,
    GraphitiLeafClass,
    graphiti_semantic_state_digest,
    load_checked_graphiti_call_shape_policy,
)
from newsroom.control_plane.model_usage import (
    InvocationAllocation,
    InvocationEfficiencyPolicy,
    InvocationTerminal,
    ModelUsageAdmissionError,
    ModelUsageIntegrityError,
    ModelUsageService,
    UsageComponents,
    UsageStatus,
    WorkEnvelope,
    WorkloadClass,
    native_sdk_reported_token_targets_are_advisory,
    native_graphiti_usage_cycle_id,
)
from newsroom.extraction.types import ExtractionRunId
from newsroom.control_plane.store import GRAPHITI_MAX_FAILURES, LEDGER_GENESIS
from newsroom.control_plane.writer import (
    cont_writer_implementation_identity as _graphiti_implementation_identity,
)
from newsroom.graphiti_adapter.contracts import GRAPHITI_PROMPT_COMPONENT
from newsroom.graphiti_adapter.cursor_transport import composer_model_meets_floor
from newsroom.graphiti_adapter.evaluation_packet import (
    GRAPHITI_CHAT_FALLBACK,
    GRAPHITI_CHAT_MODEL,
    GRAPHITI_CORE_RELEASE,
    GRAPHITI_EMBEDDING_MODEL,
    GRAPHITI_GENERATION_ID,
    GRAPHITI_WORKSPACE_GROUP,
    GROK_CHAT_REASONING,
)
from newsroom.graphiti_adapter.temporal_vocabulary import TemporalBasis
from newsroom.graphiti_adapter.identity import typed_id
from newsroom.graphiti_adapter.types import (
    GraphitiAdapterContractError,
    GraphitiAdapterOutcome,
)


GRAPHITI_RESULT_STAGE_ADAPTER_EXECUTION = "ADAPTER_EXECUTION"
GRAPHITI_RESULT_STAGE_CYCLE_RESULT_CONSTRUCTION = "CYCLE_RESULT_CONSTRUCTION"
GRAPHITI_RESULT_STAGE_CYCLE_RESULT_BINDING = "CYCLE_RESULT_BINDING"
GRAPHITI_RESULT_STAGE_UNCLASSIFIED = "UNCLASSIFIED_RESULT_BOUNDARY"
GRAPHITI_RESULT_FAILURE_STAGES = frozenset(
    {
        GRAPHITI_RESULT_STAGE_ADAPTER_EXECUTION,
        GRAPHITI_RESULT_STAGE_CYCLE_RESULT_CONSTRUCTION,
        GRAPHITI_RESULT_STAGE_CYCLE_RESULT_BINDING,
        GRAPHITI_RESULT_STAGE_UNCLASSIFIED,
    }
)


class GraphitiResultStageError(ValueError):
    """Safe result-boundary classification without provider response content."""

    def __init__(self, stage: str) -> None:
        if stage not in GRAPHITI_RESULT_FAILURE_STAGES:
            raise ValueError("Graphiti result failure stage is not allow-listed")
        self.stage = stage
        super().__init__(f"Graphiti result boundary failed at {stage}")


class GraphitiPreProviderAuthorizationDenied(PermissionError):
    """Typed governed-adapter refusal before Graphiti provider execution."""

    stage = GRAPHITI_RESULT_STAGE_ADAPTER_EXECUTION


@dataclass(frozen=True, slots=True)
class GraphitiCycleResult:
    ingest_id: str
    source_id: str
    item_key: str
    outcome: str
    proposal_count: int
    entity_count: int
    relation_count: int
    failure_code: str
    temporal_basis: TemporalBasis
    reference_time: str
    generation_id: str = GRAPHITI_GENERATION_ID
    receipt_digest: str = ""
    workspace_group: str = GRAPHITI_WORKSPACE_GROUP
    episode_uuid: str = ""
    entities: tuple[dict[str, object], ...] = ()
    relations: tuple[dict[str, object], ...] = ()
    proposals: tuple[dict[str, object], ...] = ()
    passages: tuple[dict[str, object], ...] = ()
    chat_invocations: tuple[dict[str, object], ...] = ()
    embedding_usage: dict[str, object] | None = None
    token_usage: dict[str, object] | None = None
    request_tokens: int = 0
    response_tokens: int = 0
    cost_microunits: int = 0
    usage_basis: str = "UNOBSERVED"
    prompt_version: str = GRAPHITI_PROMPT_COMPONENT.component_version
    framework: str = GRAPHITI_CORE_RELEASE
    chat: str = GRAPHITI_CHAT_MODEL
    chat_fallback: str = GRAPHITI_CHAT_FALLBACK
    embedding: str = GRAPHITI_EMBEDDING_MODEL
    attempt_number: int = 1
    provider_attempt_number: int = 1
    predecessor_episode_uuid: str | None = None
    raw_receipt: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.temporal_basis, TemporalBasis):
            raise TypeError("GraphitiCycleResult.temporal_basis must be typed")


class GraphitiPort(Protocol):
    def ingest(self, unit: CorpusIngestUnit) -> GraphitiCycleResult: ...


@runtime_checkable
class GovernedRealGraphitiPort(GraphitiPort, Protocol):
    requires_canonical_control_plane_stores: bool

    def ingest_until(
        self, unit: CorpusIngestUnit, *, deadline: datetime
    ) -> GraphitiCycleResult: ...

    def ingest_with_usage(
        self,
        unit: CorpusIngestUnit,
        *,
        model_usage: ModelUsageService,
        cycle_id: str,
        dispatch_authority: Mapping[str, object],
        owner_stop_check: Callable[[], None],
        deadline: datetime | None = None,
    ) -> GraphitiCycleResult: ...


GRAPHITI_CONTEXT_IDENTITY = "graphiti-combined-temporal-hermetic-v1"
GRAPHITI_CONTEXT_MANIFEST_SCHEMA_VERSION = (
    "newsroom.graphiti-hermetic-context-manifest.v1"
)
GRAPHITI_CHAT_PRIMARY_ROUTE = "GRAPHITI_CHAT_PRIMARY"
GRAPHITI_CHAT_FALLBACK_ROUTE = "GRAPHITI_CHAT_FALLBACK"
GRAPHITI_EMBEDDING_ROUTE = "GRAPHITI_EMBEDDING"


def graphiti_required_route_holds(
    service: ModelUsageService | None,
    *,
    fallback_permitted: bool = False,
) -> tuple[dict[str, object], ...]:
    """Check required downstream routes before spending on an upstream chat."""
    if service is None:
        return ()
    holds = []
    with service.route_state_snapshot() as route_state:
        embedding = route_state(GRAPHITI_EMBEDDING_ROUTE)
        if embedding["state"] == "OPEN":
            holds.append(embedding)
        primary = route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)
        if primary["state"] == "OPEN":
            fallback_substitutes = False
            if fallback_permitted and isinstance(primary.get("event_digest"), str):
                fallback_substitutes = (
                    route_state(GRAPHITI_CHAT_FALLBACK_ROUTE)["state"]
                    == "CLOSED"
                )
            if not fallback_substitutes:
                holds.append(primary)
    return tuple(holds)

_GRAPHITI_ADAPTER_DIRECTORY = Path(__file__).parent.parent / "graphiti_adapter"
_GRAPHITI_HERMETIC_ENVIRONMENT_KEYS = (
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "TMPDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
)


def _qualified_route_model_admits(*, observed_model: str, policy_model: str) -> bool:
    if policy_model == "composer>=2.5":
        return composer_model_meets_floor(observed_model)
    return observed_model == policy_model


def _graphiti_transport_implementation_revision(
    leaf_class: GraphitiLeafClass,
) -> str:
    if leaf_class is GraphitiLeafClass.EMBEDDING:
        return digest_bytes(
            (_GRAPHITI_ADAPTER_DIRECTORY / "embedding_meter.py").read_bytes()
        )
    sources = (
        _GRAPHITI_ADAPTER_DIRECTORY / "cli_process.py",
        _GRAPHITI_ADAPTER_DIRECTORY / "cli_client.py",
        _GRAPHITI_ADAPTER_DIRECTORY / "cursor_transport.py",
    )
    retained = b"".join(
        path.name.encode("utf-8") + b"\x00" + path.read_bytes() + b"\x00"
        for path in sources
    )
    return digest_bytes(retained)


class GraphitiModelUsageObserver:
    def __init__(
        self,
        *,
        service: ModelUsageService,
        envelope: WorkEnvelope,
        clock: Callable[[], datetime],
        effective_revision_digest: str | None = None,
        ingest_obligation_id: str | None = None,
        provider_attempt_number: int = 1,
        deadline: datetime | None = None,
        dispatch_authority_digest: str | None = None,
        owner_stop_check: Callable[[], None],
        call_shape_policy: GraphitiCallShapePolicy | None = None,
        fallback_policy: GraphitiFallbackCircuitPolicy | None = None,
        recovered_ambiguous_progression: object | None = None,
        inherited_empty_for: Callable | None = None,
        typed_proposal_verifier: Callable | None = None,
    ) -> None:
        self._service = service
        self._envelope = envelope
        self._clock = clock
        _, self._implementation_worktree_clean = _graphiti_implementation_identity()
        if envelope.graphiti_attempt_id is None:
            raise ValueError("Graphiti usage observer lacks an attempt identity")
        self._ordinal = (
            service.next_graphiti_internal_ordinal(
                graphiti_attempt_id=envelope.graphiti_attempt_id
            )
            - 1
        )
        self._allocations: list[InvocationAllocation] = []
        self._dispatch_at: dict[str, datetime] = {}
        self._policies: dict[str, InvocationEfficiencyPolicy] = {}
        self._primary_by_request: dict[str, str] = {}
        self._terminal_outcome: dict[str, str] = {}
        self._fallback_by_primary: set[str] = set()
        self._direct_fallback_by_request: dict[str, str] = {}
        self._shape = call_shape_policy or load_checked_graphiti_call_shape_policy()
        self._fallback_policy = (
            fallback_policy or load_checked_graphiti_fallback_circuit_policy()
        )
        if self._fallback_policy.call_shape_policy_digest != self._shape.canonical_digest:
            raise ValueError("Graphiti fallback policy differs from the call shape")
        self._effective_revision_digest = effective_revision_digest or digest_canonical(
            {"ingest_id": envelope.ingest_id, "basis": "LEGACY_TEST_FIXTURE"}
        )
        self._ingest_obligation_id = ingest_obligation_id or envelope.ingest_id
        if self._ingest_obligation_id is None:
            raise ValueError("Graphiti usage observer lacks an ingest obligation")
        if provider_attempt_number <= 0:
            raise ValueError("Graphiti provider attempt number must be positive")
        self._provider_attempt_number = provider_attempt_number
        self._recovered_ambiguous_progression = recovered_ambiguous_progression
        native_attempt = (
            envelope.workload_class is WorkloadClass.GRAPHITI_CHAT_PRIMARY
            and all(value is None for value in (
                envelope.admission_decision_id, envelope.candidate_id,
                envelope.hypothesis_digest, envelope.evidence_package_digest,
            ))
            and envelope.ingest_id == self._ingest_obligation_id
            and envelope.graphiti_attempt_id == f"{self._ingest_obligation_id}:{provider_attempt_number}"
            and envelope.cycle_id == native_graphiti_usage_cycle_id(
                ingest_id=self._ingest_obligation_id, attempt_number=provider_attempt_number,
            )
        )
        if provider_attempt_number > 1 and native_attempt:
            # The bounded reader authenticates paid semantic verifiers separately;
            # their envelopes are not Graphiti leaves or zero-dispatch credits.
            self._retry_evidence = service.native_graphiti_ingest_retry_evidence_many(
                failed_attempts={self._ingest_obligation_id: provider_attempt_number - 1},
                max_attempts=provider_attempt_number - 1,
            )[self._ingest_obligation_id]
        else:
            retry_evidence = getattr(service, "graphiti_ingest_retry_evidence", None)
            self._retry_evidence = (
                retry_evidence(
                    ingest_id=self._ingest_obligation_id,
                    before_attempt_number=provider_attempt_number,
                )
                if provider_attempt_number > 1 and callable(retry_evidence)
                else None
            )
        self._proved_pre_dispatch_zero_retry = bool(
            self._retry_evidence is not None
            and getattr(self._retry_evidence, "zero_dispatch_attempts", ())
            and not getattr(self._retry_evidence, "settled_provider_attempts", ())
            and not getattr(self._retry_evidence, "unresolved_attempts", ())
        )
        self._deadline = deadline
        self._dispatch_authority_digest = (
            dispatch_authority_digest
            or digest_canonical(
                {
                    "envelope_id": envelope.envelope_id,
                    "basis": "LEGACY_TEST_FIXTURE",
                }
            )
        )
        self._owner_stop_check = owner_stop_check
        self.inherited_empty_for = inherited_empty_for
        self._typed_proposal_verifier = typed_proposal_verifier

    def verify_typed_proposals(self, *, source_revision, proposal_receipt):
        """Verify proposals under the captured authoritative attempt, not SDK IDs."""
        if self._typed_proposal_verifier is None:
            return None
        self._owner_stop_check()
        verified = self._typed_proposal_verifier(
            source_revision=source_revision,
            proposal_receipt=deepcopy(proposal_receipt),
        )
        if not isinstance(verified, Mapping) or not verified:
            raise ValueError("Graphiti typed verification evidence is absent")
        retained = deepcopy(dict(verified))
        digest_canonical(retained)
        return retained

    def allows_fresh_zero_dispatch_retry(
        self, *, episode_uuid: str, attempt_number: int
    ) -> bool:
        """Bind the internal usage proof to this exact ingest attempt."""

        return (
            self._proved_pre_dispatch_zero_retry
            and episode_uuid == self._ingest_obligation_id
            and attempt_number == self._provider_attempt_number
            and self._envelope.graphiti_attempt_id
            == f"{episode_uuid}:{attempt_number}"
        )

    def allows_fresh_completed_rollback_retry(
        self,
        *,
        episode_uuid: str,
        attempt_number: int,
        prior_attempt_number: int,
    ) -> bool:
        """Permit a bounded retry after an exactly settled provider attempt."""

        evidence = self._retry_evidence
        if evidence is None:
            return False
        settled = tuple(getattr(evidence, "settled_provider_attempts", ()))
        unresolved = tuple(getattr(evidence, "unresolved_attempts", ()))
        proof = self._recovered_ambiguous_progression
        if proof is not None and self._fresh_recovered_ambiguous_usage(proof):
            unresolved = tuple(
                number for number in unresolved
                if number != getattr(proof, "skipped_attempt_number", None)
            )
        return (
            episode_uuid == self._ingest_obligation_id
            and attempt_number == self._provider_attempt_number
            and self._envelope.graphiti_attempt_id
            == f"{episode_uuid}:{attempt_number}"
            and not unresolved
            and prior_attempt_number in settled
            and prior_attempt_number
            == getattr(evidence, "latest_settled_provider_attempt", None)
            and len(settled) < GRAPHITI_MAX_FAILURES
        )

    def _fresh_recovered_ambiguous_usage(self, proof: object) -> bool:
        try:
            digest = self._service.native_recovered_ambiguous_usage_evidence_digest(
                ingest_id=self._ingest_obligation_id,
                authoritative_attempt_number=proof.authoritative_attempt_number,
                skipped_attempt_number=proof.skipped_attempt_number,
                skipped_receipt_digest=proof.skipped_receipt_digest,
                skipped_recorded_at=proof.skipped_recorded_at.value,
                latest_allowed_attempt_number=self._provider_attempt_number - 1,
            )
        except (AttributeError, ModelUsageIntegrityError):
            return False
        return digest == proof.settled_usage_evidence_digest

    def allows_fresh_recovered_ambiguous_retry(
        self,
        *,
        episode_uuid: str,
        attempt_number: int,
        prior_attempt_number: int,
        proof: object,
    ) -> bool:
        """Rebind the retained gap proof to current accounting before dispatch."""

        return (
            proof == self._recovered_ambiguous_progression
            and episode_uuid == self._ingest_obligation_id
            and attempt_number == self._provider_attempt_number
            and self._envelope.graphiti_attempt_id
            == f"{episode_uuid}:{attempt_number}"
            and prior_attempt_number == proof.authoritative_attempt_number
            and self._fresh_recovered_ambiguous_usage(proof)
        )

    def _policy_for(
        self,
        *,
        workload: WorkloadClass,
        provider: str,
        route: str,
        model: str,
        reasoning: str,
        output_schema_digest: str,
        semantic_request_class: str,
        response_schema_identity: str,
    ) -> InvocationEfficiencyPolicy:
        leaf_class = (
            GraphitiLeafClass.EMBEDDING
            if workload is WorkloadClass.GRAPHITI_EMBEDDING
            else GraphitiLeafClass.FALLBACK
            if workload is WorkloadClass.GRAPHITI_CHAT_FALLBACK
            else GraphitiLeafClass.PRIMARY
        )
        route_contract = self._shape.route_for(leaf_class)
        implementation_revision = _graphiti_transport_implementation_revision(
            leaf_class
        )
        self._shape.qualify_request(
            leaf_class=leaf_class,
            semantic_request_class=semantic_request_class,
            response_schema_identity=response_schema_identity,
            response_schema_digest=output_schema_digest,
        )
        if (
            provider,
            route,
            reasoning,
        ) != (
            route_contract.provider,
            route_contract.route,
            route_contract.reasoning,
        ) or not _qualified_route_model_admits(
            observed_model=model,
            policy_model=route_contract.model,
        ):
            raise ModelUsageAdmissionError(
                "Graphiti route differs from the checked qualified policy"
            )
        policy = InvocationEfficiencyPolicy.create(
            policy_id=(
                f"graphiti-{workload.value.lower()}-{semantic_request_class.lower()}-"
                f"{output_schema_digest.removeprefix('sha256:')[:12]}"
            ),
            version=self._shape.version,
            workload_class=workload,
            provider=provider,
            route=route,
            model=model,
            reasoning=reasoning,
            one_turn=True,
            exact_input=True,
            skills_enabled=False,
            tools_enabled=False,
            mcp_enabled=False,
            prior_message_count=0,
            command_semantic_version=route_contract.command_semantic_version,
            command_flags=route_contract.command_flags,
            context_manifest_schema_version=GRAPHITI_CONTEXT_MANIFEST_SCHEMA_VERSION,
            disabled_capabilities=route_contract.disabled_capabilities,
            implementation_revision=implementation_revision,
            max_prompt_bytes=route_contract.max_prompt_bytes,
            max_context_tokens=route_contract.max_context_tokens,
            max_output_tokens=route_contract.max_output_tokens,
            max_total_tokens=route_contract.max_total_tokens,
            prompt_contract_version=GRAPHITI_PROMPT_COMPONENT.component_version,
            output_schema_digest=output_schema_digest,
            allowed_context_identities=(GRAPHITI_CONTEXT_IDENTITY,),
            allowed_config_identities=(route_contract.config_identity,),
            hard_estimate_ceiling_tokens=None,
            evidence_digest=digest_canonical(
                {
                    "call_shape_policy_digest": self._shape.canonical_digest,
                    "fallback_circuit_policy_digest": (
                        self._fallback_policy.canonical_digest
                    ),
                    "qualified_route": route_contract.as_record(),
                    "semantic_request_class": semantic_request_class,
                    "output_schema_digest": output_schema_digest,
                    "bounds": {
                        "prompt_bytes": route_contract.max_prompt_bytes,
                        "context_tokens": route_contract.max_context_tokens,
                        "output_tokens": route_contract.max_output_tokens,
                    },
                }
            ),
            qualified=True,
        )
        self._service.register_policy(policy)
        return policy

    def _allocate(
        self,
        *,
        workload: WorkloadClass,
        provider: str,
        route: str,
        model: str,
        reasoning: str,
        prompt_bytes: bytes,
        semantic_request_class: str,
        response_schema_identity: str,
        output_schema_digest: str,
        requested_max_tokens: int,
        parent_invocation_id: str | None = None,
        primary_unavailable_event_digest: str | None = None,
    ) -> InvocationAllocation:
        now = self._clock().astimezone(UTC)
        if self._deadline is not None:
            if self._deadline.tzinfo is None or self._deadline.utcoffset() is None:
                raise ValueError("Graphiti dispatch deadline must be timezone-aware")
            if now >= self._deadline.astimezone(UTC):
                raise ValueError("Graphiti dispatch deadline has expired")
        policy = self._policy_for(
            workload=workload,
            provider=provider,
            route=route,
            model=model,
            reasoning=reasoning,
            output_schema_digest=output_schema_digest,
            semantic_request_class=semantic_request_class,
            response_schema_identity=response_schema_identity,
        )
        if requested_max_tokens <= 0:
            requested_max_tokens = policy.max_output_tokens
        if (
            not policy.one_turn
            or not policy.exact_input
            or policy.skills_enabled
            or policy.tools_enabled
            or policy.mcp_enabled
            or policy.prior_message_count != 0
        ):
            raise ValueError(
                "Graphiti invocation controls do not match the exact one-turn contract"
            )
        self._ordinal += 1
        allocated_at = now
        prompt_digest = digest_bytes(prompt_bytes)
        leaf_class = (
            GraphitiLeafClass.EMBEDDING
            if workload is WorkloadClass.GRAPHITI_EMBEDDING
            else GraphitiLeafClass.FALLBACK
            if workload is WorkloadClass.GRAPHITI_CHAT_FALLBACK
            else GraphitiLeafClass.PRIMARY
        )
        route_contract = self._shape.route_for(leaf_class)
        retry_state_digest = digest_canonical(
            {
                "leaf_class": leaf_class.value,
                "parent_invocation_id": parent_invocation_id,
                "primary_unavailable_event_digest": (
                    primary_unavailable_event_digest
                ),
                "semantic_request_class": semantic_request_class,
            }
        )
        semantic_state_digest = graphiti_semantic_state_digest(
            semantic_request_class=semantic_request_class,
            prompt_digest=prompt_digest,
            response_schema_digest=output_schema_digest,
            requested_max_tokens=requested_max_tokens,
            leaf_class=leaf_class,
            retry_state_digest=retry_state_digest,
        )
        provider_attempt_id = (
            f"{self._envelope.graphiti_attempt_id}:provider-attempt:"
            f"{self._provider_attempt_number}:leaf:{self._ordinal}"
        )
        system_digest = digest_canonical(
            {"system_identity": self._shape.prompt_identity}
        )
        request_digest = digest_canonical(
            {
                "provider": provider,
                "route": route,
                "model": model,
                "reasoning": reasoning,
                "command_semantic_version": route_contract.command_semantic_version,
                "command_flags": list(route_contract.command_flags),
                "implementation_revision": policy.implementation_revision,
                "system_digest": system_digest,
                "prompt_digest": prompt_digest,
                "output_schema_digest": output_schema_digest,
            }
        )
        context_manifest = {
            "schema_version": GRAPHITI_CONTEXT_MANIFEST_SCHEMA_VERSION,
            "provider": provider,
            "route": route,
            "model": model,
            "reasoning": reasoning,
            "command_semantic_version": route_contract.command_semantic_version,
            "command_flags": list(route_contract.command_flags),
            "implementation_revision": policy.implementation_revision,
            "implementation_worktree_clean": self._implementation_worktree_clean,
            "disabled_capabilities": list(route_contract.disabled_capabilities),
            "working_directory_inventory": [],
            "working_directory_inventory_digest": digest_canonical([]),
            "environment_keys": (
                []
                if leaf_class is GraphitiLeafClass.EMBEDDING
                else [
                    *(
                        ("GROK_AUTH_PATH",)
                        if leaf_class is GraphitiLeafClass.FALLBACK
                        else ()
                    ),
                    *_GRAPHITI_HERMETIC_ENVIRONMENT_KEYS,
                ]
            ),
            "config_identity": route_contract.config_identity,
            "context_identity": GRAPHITI_CONTEXT_IDENTITY,
            "system_digest": system_digest,
            "prompt_contract_version": GRAPHITI_PROMPT_COMPONENT.component_version,
            "prompt_bytes": len(prompt_bytes),
            "prompt_digest": prompt_digest,
            "schema_digest": output_schema_digest,
            "output_schema_digest": output_schema_digest,
            "evidence_package_digest": self._effective_revision_digest,
            "evidence_package_bytes": len(prompt_bytes),
            "effective_revision_digest": self._effective_revision_digest,
            "ingest_obligation_id": self._ingest_obligation_id,
            "graphiti_attempt_id": self._envelope.graphiti_attempt_id,
            "provider_attempt_id": provider_attempt_id,
            "dispatch_authority_digest": self._dispatch_authority_digest,
            "dispatch_deadline": (
                None
                if self._deadline is None
                else self._deadline.astimezone(UTC).isoformat()
            ),
            "semantic_state_digest": semantic_state_digest,
            "request_digest": request_digest,
            "call_shape_policy_digest": self._shape.canonical_digest,
            "one_turn": True,
            "exact_input": True,
            "skills_enabled": False,
            "tools_enabled": False,
            "mcp_enabled": False,
            "prior_message_count": 0,
            "skill_count": 0,
            "tool_count": 0,
            "mcp_server_count": 0,
            "mcp_tool_count": 0,
        }
        context_manifest_digest = digest_canonical(context_manifest)
        self._service.retain_context_manifest(
            {
                "context_manifest_digest": context_manifest_digest,
                **context_manifest,
            }
        )
        allocation = InvocationAllocation.create(
            envelope_id=self._envelope.envelope_id,
            cycle_id=self._envelope.cycle_id,
            leaf_ordinal=self._ordinal,
            workload_class=workload,
            invocation_policy_digest=policy.canonical_digest,
            provider=provider,
            route=route,
            model=model,
            reasoning=reasoning,
            prompt_contract_version=GRAPHITI_PROMPT_COMPONENT.component_version,
            prompt_bytes=len(prompt_bytes),
            prompt_digest=prompt_digest,
            request_digest=request_digest,
            output_schema_digest=output_schema_digest,
            max_output_tokens=requested_max_tokens,
            context_manifest_digest=context_manifest_digest,
            context_identity=GRAPHITI_CONTEXT_IDENTITY,
            config_identity=route_contract.config_identity,
            one_turn=True,
            exact_input=True,
            skills_enabled=False,
            tools_enabled=False,
            mcp_enabled=False,
            prior_message_count=0,
            allocated_at=allocated_at,
            recovery_deadline_at=allocated_at + timedelta(minutes=16),
            parent_invocation_id=parent_invocation_id,
        )
        self._owner_stop_check()
        route_circuit_state = str(self._service.route_state(route)["state"])
        if route_circuit_state != "CLOSED":
            raise ModelUsageAdmissionError("affected route circuit is open")
        identity = GraphitiInternalRequestIdentity.create(
            effective_revision_digest=self._effective_revision_digest,
            ingest_obligation_id=self._ingest_obligation_id,
            graphiti_attempt_id=str(self._envelope.graphiti_attempt_id),
            provider_attempt_id=provider_attempt_id,
            internal_ordinal=self._ordinal,
            semantic_request_class=semantic_request_class,
            provider=provider,
            model=model,
            reasoning=reasoning,
            prompt_bytes=len(prompt_bytes),
            prompt_digest=prompt_digest,
            response_schema_identity=response_schema_identity,
            response_schema_digest=output_schema_digest,
            requested_max_tokens=requested_max_tokens,
            framework_identity=self._shape.framework_identity,
            prompt_identity=self._shape.prompt_identity,
            ontology_identity=self._shape.ontology_identity,
            temporal_identity=self._shape.temporal_identity,
            generation_policy_identity=self._shape.generation_policy_identity,
            context_manifest_digest=context_manifest_digest,
            leaf_class=leaf_class,
            retry_state_digest=retry_state_digest,
            parent_invocation_id=parent_invocation_id,
            envelope_id=self._envelope.envelope_id,
            invocation_id=allocation.invocation_id,
            invocation_policy_digest=policy.canonical_digest,
            call_shape_policy_digest=self._shape.canonical_digest,
            dispatch_authority_digest=self._dispatch_authority_digest,
            dispatch_deadline_at=(
                None
                if self._deadline is None
                else self._deadline.astimezone(UTC).isoformat()
            ),
            owner_stop_clear=True,
            route_circuit_state=route_circuit_state,
            primary_unavailable_event_digest=primary_unavailable_event_digest,
        )
        self._service.allocate_graphiti_request(
            allocation,
            identity=identity,
            max_distinct_internal_requests=self._shape.max_distinct_internal_requests,
        )
        self._allocations.append(allocation)
        self._policies[allocation.invocation_id] = policy
        return allocation

    def transport_dispatch_started(self, token: object) -> None:
        """Retain the provider-I/O boundary after local preflight succeeds."""

        if not isinstance(token, InvocationAllocation):
            raise TypeError("Graphiti transport token is not an allocation")
        if token.invocation_id in self._dispatch_at:
            raise ModelUsageAdmissionError("Graphiti transport dispatch repeated")
        dispatch_at = self._clock().astimezone(UTC)
        if self._deadline is not None and dispatch_at >= self._deadline.astimezone(UTC):
            raise ModelUsageAdmissionError(
                "Graphiti dispatch deadline expired during local preflight"
            )
        self._owner_stop_check()
        if self._service.route_state(token.route)["state"] != "CLOSED":
            raise ModelUsageAdmissionError(
                "Graphiti route circuit opened during local preflight"
            )
        self._service.observe_transport(
            invocation_id=token.invocation_id,
            observed_at=dispatch_at,
            state="DISPATCH_STARTED",
            evidence_digest=digest_canonical(
                {
                    "invocation_id": token.invocation_id,
                    "provider": token.provider,
                    "route": token.route,
                    "request_digest": token.request_digest,
                }
            ),
        )
        self._dispatch_at[token.invocation_id] = dispatch_at

    def link_provider_attempts(
        self, *, provider_attempt_number: int, linked_at: datetime
    ) -> None:
        if provider_attempt_number <= 0:
            raise ValueError("Graphiti provider attempt number must be positive")
        del linked_at
        if not self._allocations:
            return
        if provider_attempt_number != self._provider_attempt_number:
            raise ValueError("Graphiti provider attempt identity changed after dispatch")

    def reported_token_targets_are_advisory(self, token: object) -> bool:
        """Bind future response semantics to this observer's exact allocation."""
        if not isinstance(token, InvocationAllocation) or token not in self._allocations:
            raise ModelUsageIntegrityError("Graphiti response contract allocation differs")
        policy = self._policies[token.invocation_id]
        if policy.canonical_digest != token.invocation_policy_digest:
            raise ModelUsageIntegrityError("Graphiti response contract policy differs")
        return native_sdk_reported_token_targets_are_advisory(policy)

    def before_cli_invocation(
        self,
        *,
        provider: str,
        model: str,
        prompt: str,
        schema: str | None,
        semantic_request_class: str = "UNSTRUCTURED",
        max_tokens: int = 0,
    ) -> object:
        prompt_digest = digest_bytes(prompt.encode("utf-8"))
        schema_digest = digest_canonical(
            {"response_schema": schema or "UNSTRUCTURED"}
        )
        request_key = digest_canonical(
            {
                "semantic_request_class": semantic_request_class,
                "prompt_digest": prompt_digest,
                "response_schema_digest": schema_digest,
                "requested_max_tokens": max_tokens,
            }
        )
        primary = provider == "cursor-agent-cli"
        fallback = provider == "grok-build-cli"
        parent_invocation_id = (
            self._primary_by_request.get(request_key) if fallback else None
        )
        parent_outcome = (
            None
            if parent_invocation_id is None
            else self._terminal_outcome.get(parent_invocation_id)
        )
        primary_unavailable_event_digest = (
            self._direct_fallback_by_request.get(request_key)
            if fallback
            else None
        )
        if fallback and (
            primary_unavailable_event_digest is None
            and (
                parent_outcome is None
                or classify_graphiti_fallback(parent_outcome).eligibility
                is not FallbackEligibility.ELIGIBLE
            )
        ):
            raise ModelUsageAdmissionError(
                "Graphiti fallback requires a malformed primary"
            )
        fallback_authority = parent_invocation_id or primary_unavailable_event_digest
        fallback_leaf_key = (
            digest_canonical(
                {"authority": fallback_authority, "request_key": request_key}
            )
            if fallback_authority is not None
            else None
        )
        if fallback and fallback_leaf_key in self._fallback_by_primary:
            raise ModelUsageAdmissionError(
                "Graphiti primary already has its single fallback leaf"
            )
        allocation = self._allocate(
            workload=(
                WorkloadClass.GRAPHITI_CHAT_PRIMARY
                if primary
                else WorkloadClass.GRAPHITI_CHAT_FALLBACK
            ),
            provider=provider,
            route=(
                GRAPHITI_CHAT_PRIMARY_ROUTE
                if primary
                else GRAPHITI_CHAT_FALLBACK_ROUTE
            ),
            model=model,
            reasoning="provider-default" if primary else GROK_CHAT_REASONING,
            prompt_bytes=prompt.encode("utf-8"),
            semantic_request_class=semantic_request_class,
            response_schema_identity=semantic_request_class,
            output_schema_digest=schema_digest,
            requested_max_tokens=max_tokens,
            parent_invocation_id=parent_invocation_id,
            primary_unavailable_event_digest=primary_unavailable_event_digest,
        )
        if primary:
            self._primary_by_request[request_key] = allocation.invocation_id
        elif fallback and fallback_leaf_key is not None:
            self._fallback_by_primary.add(fallback_leaf_key)
        return allocation

    def use_direct_fallback(
        self,
        *,
        prompt: str,
        schema: str | None,
        semantic_request_class: str,
        max_tokens: int,
    ) -> bool:
        """Select Grok only when the retained primary route head is OPEN."""

        primary = self._service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)
        if primary.get("state") != "OPEN":
            return False
        if self._service.route_state(GRAPHITI_CHAT_FALLBACK_ROUTE).get("state") != (
            "CLOSED"
        ):
            return False
        event_digest = primary.get("event_digest")
        if not isinstance(event_digest, str):
            return False
        request_key = digest_canonical(
            {
                "semantic_request_class": semantic_request_class,
                "prompt_digest": digest_bytes(prompt.encode("utf-8")),
                "response_schema_digest": digest_canonical(
                    {"response_schema": schema or "UNSTRUCTURED"}
                ),
                "requested_max_tokens": max_tokens,
            }
        )
        self._direct_fallback_by_request[request_key] = event_digest
        return True

    def after_cli_invocation(
        self,
        token: object,
        *,
        outcome: str,
        usage: dict[str, object],
    ) -> dict[str, str]:
        if not isinstance(token, InvocationAllocation):
            raise TypeError("Graphiti chat usage token is not an allocation")
        binding = self._complete(
            token,
            outcome=outcome,
            usage=usage,
            od_011_reference=None,
            subscription_not_debited=True,
        )
        self._terminal_outcome[token.invocation_id] = outcome
        return binding

    def before_embedding_invocation(
        self, *, provider: str, model: str, input_data: object
    ) -> object:
        prompt_bytes = canonical_json_bytes({"embedding_input": input_data})
        return self._allocate(
            workload=WorkloadClass.GRAPHITI_EMBEDDING,
            provider=provider,
            route=GRAPHITI_EMBEDDING_ROUTE,
            model=model,
            reasoning="none",
            prompt_bytes=prompt_bytes,
            semantic_request_class="EMBEDDING_VECTOR",
            response_schema_identity="embedding-vector",
            output_schema_digest=digest_canonical(
                {"schema": "embedding-vector", "model": model}
            ),
            requested_max_tokens=1,
        )

    def after_embedding_invocation(
        self,
        token: object,
        *,
        outcome: str,
        usage: dict[str, object],
    ) -> dict[str, str]:
        if not isinstance(token, InvocationAllocation):
            raise TypeError("Graphiti embedding usage token is not an allocation")
        return self._complete(
            token,
            outcome=outcome,
            usage=usage,
            od_011_reference="OD-011:EVALUATION_GRAPHITI_EMBEDDING",
            subscription_not_debited=False,
        )

    def _complete(
        self,
        allocation: InvocationAllocation,
        *,
        outcome: str,
        usage: dict[str, object],
        od_011_reference: str | None,
        subscription_not_debited: bool,
    ) -> dict[str, str]:
        reported = usage.get("usage_basis") == "PROVIDER_REPORTED"
        no_provider_call = usage.get("usage_basis") == "NO_PROVIDER_CALL"
        policy = self._policies[allocation.invocation_id]
        components = (
            UsageComponents(
                input_tokens=usage.get("input_tokens"),  # type: ignore[arg-type]
                output_tokens=usage.get("output_tokens"),  # type: ignore[arg-type]
                cached_read_tokens=usage.get("cached_read_tokens"),  # type: ignore[arg-type]
                cached_write_tokens=usage.get("cached_write_tokens"),  # type: ignore[arg-type]
                reasoning_tokens=usage.get("reasoning_tokens"),  # type: ignore[arg-type]
                context_tokens=usage.get("context_tokens"),  # type: ignore[arg-type]
                total_tokens=usage.get("total_tokens"),  # type: ignore[arg-type]
                provenance="PROVIDER_REPORTED",
            )
            if reported
            else (
                UsageComponents(total_tokens=0, provenance="CLI_DERIVED")
                if no_provider_call
                else UsageComponents(
                    total_tokens=policy.hard_estimate_ceiling_tokens,
                    provenance="BOUNDED_ESTIMATE",
                )
                if policy.hard_estimate_ceiling_tokens is not None
                else UsageComponents(
                    provenance="UNAVAILABLE",
                )
            )
        )
        telemetry = usage.get("provider_telemetry", usage)
        telemetry_mapping = telemetry if isinstance(telemetry, dict) else usage
        completed_at = self._clock().astimezone(UTC)
        outcome_class = classify_graphiti_fallback(outcome).outcome_class
        route_circuit_reason = (
            outcome_class.value
            if outcome_class in self._fallback_policy.circuit_open_classes
            else None
        )
        terminal = self._service.complete(
            InvocationTerminal.create(
                invocation_id=allocation.invocation_id,
                outcome=outcome,
                failure_class=(
                    None
                    if reported
                    else (
                        outcome
                        if no_provider_call
                        else "MISSING_PROVIDER_TELEMETRY"
                    )
                ),
                usage_status=(
                    UsageStatus.REPORTED
                    if reported or no_provider_call
                    else UsageStatus.ESTIMATED
                    if policy.hard_estimate_ceiling_tokens is not None
                    else UsageStatus.UNREPORTED
                ),
                components=components,
                dispatch_at=(
                    None
                    if no_provider_call
                    else self._dispatch_at[allocation.invocation_id]
                ),
                completed_at=completed_at,
                observed_at=completed_at,
                provider_telemetry_digest=(
                    digest_canonical(telemetry_mapping) if reported else None
                ),
                raw_telemetry_pointer=(
                    "sqlite-private://model_provider_telemetry/"
                    f"{allocation.invocation_id}"
                    if reported
                    else None
                ),
                od_011_reference=od_011_reference,
                subscription_cli_chat_not_cash_debited=subscription_not_debited,
                pre_dispatch_zero_proved=no_provider_call,
                estimate_policy_digest=(
                    policy.canonical_digest
                    if not reported
                    and not no_provider_call
                    and policy.hard_estimate_ceiling_tokens is not None
                    else None
                ),
                estimate_calculation=(
                    "qualified_policy.hard_estimate_ceiling_tokens="
                    f"{policy.hard_estimate_ceiling_tokens}"
                    if not reported
                    and not no_provider_call
                    and policy.hard_estimate_ceiling_tokens is not None
                    else None
                ),
            ),
            provider_telemetry=telemetry_mapping if reported else None,
        )
        if route_circuit_reason is not None:
            self._service.open_route_circuit(
                route=allocation.route,
                reason=route_circuit_reason,
                invocation_id=allocation.invocation_id,
                recorded_at=completed_at,
            )
        return {
            "model_work_envelope_id": allocation.envelope_id,
            "model_invocation_id": allocation.invocation_id,
            "model_invocation_allocation_digest": allocation.canonical_digest,
            "model_invocation_terminal_digest": terminal.terminal_digest,
        }


class EvaluationGraphitiRunner:
    """Real Graphiti under EVALUATION. Does not write the ledger or admitted labels."""

    requires_canonical_control_plane_stores = True

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
        fallback_permitted: bool = True,
        governed_fallback_permitted: bool = False,
        proposal_adapter: GovernedGraphitiProposalAdapter | None = None,
        extraction_records: GovernedExtractionRecords | None = None,
        proof: AuthenticationProof | None = None,
        call_shape_policy: GraphitiCallShapePolicy | None = None,
        fallback_policy: GraphitiFallbackCircuitPolicy | None = None,
        inherited_empty_for: Callable | None = None,
        typed_proposal_verifier: Callable | None = None,
    ) -> None:
        if not isinstance(fallback_permitted, bool):
            raise TypeError("Graphiti fallback permission must be boolean")
        if not isinstance(governed_fallback_permitted, bool):
            raise TypeError("governed Graphiti fallback permission must be boolean")
        if governed_fallback_permitted and not fallback_permitted:
            raise ValueError("governed Graphiti fallback requires fallback permission")
        self._clock = clock
        self._typed_proposal_verifier = typed_proposal_verifier
        self._fallback_permitted = fallback_permitted
        self._governed_fallback_permitted = governed_fallback_permitted
        governed_dependencies = (proposal_adapter, extraction_records, proof)
        if any(item is not None for item in governed_dependencies) and not all(
            item is not None for item in governed_dependencies
        ):
            raise ValueError(
                "Governed Graphiti execution dependencies must be supplied together"
            )
        if proposal_adapter is not None and not isinstance(
            proposal_adapter, GovernedGraphitiProposalAdapter
        ):
            raise TypeError("Graphiti proposal adapter must be governed")
        if extraction_records is not None and not isinstance(
            extraction_records, GovernedExtractionRecords
        ):
            raise TypeError("Graphiti extraction records must be governed")
        if proof is not None and not isinstance(proof, AuthenticationProof):
            raise TypeError("Graphiti authority proof must be typed")
        self._proposal_adapter = proposal_adapter
        self._extraction_records = extraction_records
        self._proof = proof
        self._call_shape_policy = call_shape_policy
        self._fallback_policy = fallback_policy
        self._inherited_empty_for = inherited_empty_for
        self._pending_usage: dict[
            tuple[str, int], tuple[ModelUsageService, WorkEnvelope]
        ] = {}
        self._recovered_ambiguous_progressions: dict[
            tuple[str, int], object
        ] = {}
        self._authenticated_recovered_gaps: dict[tuple[str, int], object] = {}

    @staticmethod
    def _attempt_for_unit(
        unit: CorpusIngestUnit, *, recovered_progression=None, extraction_previous=None,
    ):
        from newsroom.graphiti_adapter.evaluation_attempt import evaluation_attempt_for_body

        return evaluation_attempt_for_body(
            episode_body=unit.episode_body,
            ingest_id=unit.ingest_id,
            proving_run_id=unit.proving_run_id,
            source_id=unit.source_id,
            item_key=unit.item_key,
            observation_digest=unit.observation_digest,
            published_at=unit.published_at,
            updated_at=unit.updated_at,
            effective_revision=unit.effective_revision,
            canonical_url=unit.canonical_url,
            revision_digest=unit.revision_digest,
            representation_digest=unit.representation_digest,
            authority_ids=(
                None
                if unit.authority is None
                else (
                    unit.authority.admission_id,
                    unit.authority.access_decision_id,
                    unit.authority.definition_id,
                    unit.authority.definition_version_id,
                    unit.authority.item_id,
                    unit.authority.revision_id,
                    unit.authority.representation_id,
                )
            ),
            attempt_number=unit.attempt_number,
            predecessor_episode_uuid=unit.predecessor_ingest_id,
            recovered_ambiguous_progression=recovered_progression,
            extraction_previous_version_number=(
                None
                if extraction_previous is None
                else extraction_previous.version_number
            ),
            extraction_previous_run_version_id=(
                None
                if extraction_previous is None
                else extraction_previous.run_version_id
            ),
        )

    def recover_owned_pending(
        self, *, connection: sqlite3.Connection,
        owner_stop_check: Callable[[], None],
        rights_fence: Callable[[CorpusIngestUnit], ContextManager[object | None]],
        defer_before_unit: Callable[[CorpusIngestUnit], bool],
        deadline: datetime,
    ) -> object | None:
        """Recover the exact retained native owner without retry or accounting."""
        if self._proposal_adapter is None or self._extraction_records is None or self._proof is None:
            return None
        from newsroom.control_plane.model_usage import _native_landed_source_unit
        from newsroom.graphiti_adapter.identity import attempt_ids
        from newsroom.graphiti_adapter.neo4j_guard import GuardError
        from newsroom.graphiti_adapter.real import _episode_input_digest, recover_owned_pending
        from newsroom.graphiti_adapter.types import GraphitiAdapterRightsDenied, GraphitiAdapterStateError
        from newsroom.control_plane.broker import BrokerError

        @contextmanager
        def authenticated_input(episode_id: str, attempt_number: int):
            owner_stop_check()
            attempt_id, workspace_id, _ = attempt_ids(episode_id, attempt_number)
            try:
                attempt = self._proposal_adapter.attempt(attempt_id, proof=self._proof)
                manifest = self._proposal_adapter.manifest_for_attempt(attempt_id, proof=self._proof)
                metadata = self._extraction_records.metadata(attempt.run_version_id, proof=self._proof)
            except (AuthorizationDenied, GraphitiAdapterRightsDenied, GraphitiAdapterStateError):
                yield None
                return
            unit = _native_landed_source_unit(
                connection, ingest_id=episode_id, revision_id_hint=str(manifest.revision_id),
            )
            if unit is None or unit.authority is None or unit.proving_run_id != "native-source:" + unit.observation_digest:
                yield None
                return
            expected = self._attempt_for_unit(replace(unit, attempt_number=attempt_number))
            expected_manifest = replace(expected.manifest,
                requested_run_version_id=metadata.run_version_id,
                requested_version_number=metadata.version_number)
            if (
                attempt.outcome is GraphitiAdapterOutcome.COMPLETE
                or connection.execute(
                    "SELECT 1 FROM unpublished_graphiti_ingest WHERE ingest_id=? AND outcome='COMPLETE'",
                    (episode_id,),
                ).fetchone() is not None
                or attempt.attempt_id != attempt_id or attempt.attempt_number != attempt_number
                or attempt.workspace_id != workspace_id or attempt.run_id != expected.manifest.run_id
                or attempt.run_version_id != metadata.run_version_id
                or attempt.manifest_id != manifest.manifest_id
                or attempt.configuration_id != expected.configuration.configuration_id
                or attempt.configuration_digest != expected.configuration.canonical_digest
                or metadata.run_id != attempt.run_id
                or metadata.input_binding_digest != expected.manifest.input_binding_digest
                or manifest.canonical_digest != expected_manifest.canonical_digest
            ):
                yield None
                return
            owner_stop_check()
            if defer_before_unit(unit):
                yield None
                return
            with rights_fence(unit) as active_authority:
                if active_authority is None:
                    yield None
                    return
                active_authority.owner_stop_check()
                owner_stop_check()
                yield _episode_input_digest(
                    episode_id=episode_id, name=episode_id,
                    body=" ".join(unit.episode_body.split()),
                    reference_time=unit.temporal().reference_time.value,
                )

        try:
            return recover_owned_pending(
                authenticated_input=authenticated_input,
                owner_stop_check=owner_stop_check, deadline=deadline, clock=self._clock,
            )
        except (BrokerError, GuardError, GraphitiAdapterContractError):
            # Missing/mismatched/live ownership remains a group HOLD. Recovery
            # does not fabricate a receipt, settle spend or create a successor.
            return None

    @staticmethod
    def _exact_private_receipt_ledger_row(
        connection: sqlite3.Connection,
        *,
        receipt: Mapping[str, object],
    ) -> tuple[object, ...] | None:
        payload_digest = digest_canonical(receipt)
        rows = connection.execute(
            "SELECT current.seq,current.at,current.prev_digest,current.digest,"
            "current.payload_digest,current.payload_json,prior.digest FROM ledger current "
            "LEFT JOIN ledger prior ON prior.seq=current.seq-1 "
            "WHERE current.kind='GRAPHITI_EVALUATION_ATTEMPT' "
            "AND current.payload_digest=? "
            "ORDER BY current.seq",
            (payload_digest,),
        ).fetchall()
        if len(rows) != 1:
            return None
        row = tuple(rows[0])
        try:
            raw = str(row[5]).encode("utf-8")
            decoded = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        event_digest = digest_canonical(
            {
                "at": str(row[1]),
                "kind": "GRAPHITI_EVALUATION_ATTEMPT",
                "payload_digest": payload_digest,
                "prev": str(row[2]),
            }
        )
        if (
            type(row[0]) is not int
            or decoded != receipt
            or raw != canonical_json_bytes(receipt)
            or digest_bytes(raw) != str(row[4])
            or str(row[4]) != payload_digest
            or str(row[3]) != event_digest
            or row[2] != (
                LEDGER_GENESIS if row[0] == 1 else row[6]
            )
        ):
            return None
        return row

    def prepare_recovered_ambiguous_successor(
        self,
        *,
        unit: CorpusIngestUnit,
        authoritative_attempt: object,
        private_successor_attempt_number: int,
        connection: sqlite3.Connection,
        model_usage: ModelUsageService,
        latest_allowed_attempt_number: int | None = None,
        retained_reentry_attempt_number: int | None = None,
    ) -> bool:
        """Authenticate the one historical terminal-head/private-receipt gap."""

        from newsroom.authority.types import UtcTimestamp
        from newsroom.graphiti_adapter import GraphitiAttemptRecord
        from newsroom.graphiti_adapter.models import (
            RecoveredAmbiguousProgressionProof,
        )

        if not isinstance(authoritative_attempt, GraphitiAttemptRecord):
            return False
        if self._extraction_records is None or self._proof is None:
            return False
        authoritative_number = authoritative_attempt.attempt_number
        skipped_number = authoritative_number + 1
        latest_allowed = (
            skipped_number
            if latest_allowed_attempt_number is None
            else latest_allowed_attempt_number
        )
        if (
            authoritative_number != GRAPHITI_MAX_FAILURES
            or skipped_number != GRAPHITI_MAX_FAILURES + 1
            or private_successor_attempt_number != GRAPHITI_MAX_FAILURES + 2
            or latest_allowed not in {skipped_number, private_successor_attempt_number}
            or authoritative_attempt.outcome
            is not GraphitiAdapterOutcome.AMBIGUOUS_EFFECT
            or authoritative_attempt.failure_code != "AMBIGUOUS_EFFECT"
            or str(authoritative_attempt.run_id)
            != str(typed_id(ExtractionRunId, "run", unit.ingest_id))
        ):
            return False
        receipt_row = connection.execute(
            "SELECT outcome,receipt_digest,receipt_json,at FROM "
            "unpublished_graphiti_attempt_receipts WHERE ingest_id=? "
            "AND attempt_number=?",
            (unit.ingest_id, skipped_number),
        ).fetchone()
        if receipt_row is None or str(receipt_row[0]) != "FAILED":
            return False
        try:
            receipt = json.loads(str(receipt_row[2]))
            unsigned = dict(receipt)
            supplied_receipt_digest = unsigned.pop("receipt_digest")
            skipped_recorded_at = UtcTimestamp.parse(str(receipt_row[3]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        receipt_digest = digest_bytes(canonical_json_bytes(unsigned))
        if (
            not isinstance(receipt, dict)
            or supplied_receipt_digest != receipt_digest != ""
            or str(receipt_row[1]) != receipt_digest
            or receipt.get("ingest_id") != unit.ingest_id
            or receipt.get("attempt_number") != skipped_number
            or receipt.get("outcome") != "FAILED"
            or receipt.get("failure_code") != "PRODUCER_INTERNAL_ERROR"
            or receipt.get("binding_failure") != "RESULT_CONTRACT_REJECTED"
            or receipt.get("binding_failure_type")
            != "GraphitiAdapterVersionConflict"
            or receipt.get("binding_failure_stage")
            != GRAPHITI_RESULT_STAGE_UNCLASSIFIED
        ):
            return False
        ledger = self._exact_private_receipt_ledger_row(
            connection, receipt=receipt,
        )
        if ledger is None:
            return False
        spend = connection.execute(
            "SELECT status,usage_basis,actual_usd_microunits,"
            "actual_gbp_microunits FROM unpublished_graphiti_spend "
            "WHERE ingest_id=? AND attempt_number=?",
            (unit.ingest_id, skipped_number),
        ).fetchone()
        later_effect = connection.execute(
            "SELECT 1 FROM unpublished_graphiti_spend WHERE ingest_id=? "
            "AND attempt_number>? LIMIT 1",
            (unit.ingest_id, latest_allowed),
        ).fetchone()
        if (
            spend is None
            or tuple(spend) != ("UNRECONCILED", "UNREPORTED", None, None)
            or later_effect is not None
        ):
            return False
        try:
            usage_digest = (
                model_usage.native_recovered_ambiguous_usage_evidence_digest(
                    ingest_id=unit.ingest_id,
                    authoritative_attempt_number=authoritative_number,
                    skipped_attempt_number=skipped_number,
                    skipped_receipt_digest=receipt_digest,
                    skipped_recorded_at=skipped_recorded_at.value,
                    latest_allowed_attempt_number=latest_allowed,
                    retained_reentry_attempt_number=retained_reentry_attempt_number,
                )
            )
        except ModelUsageIntegrityError:
            return False
        authoritative_extraction = self._extraction_records.metadata(
            authoritative_attempt.run_version_id,
            proof=self._proof,
        )
        temporal = unit.temporal()
        marker_input_digest = digest_canonical(
            {
                "episode_uuid": unit.ingest_id,
                "name": unit.ingest_id,
                "body": " ".join(unit.episode_body.split()),
                "reference_time": temporal.reference_time.value.isoformat(),
                "group_id": GRAPHITI_WORKSPACE_GROUP,
            }
        )
        marker_digest = digest_canonical(
            {
                "state": "RECOVERED_AMBIGUOUS",
                "attempt_number": authoritative_number,
                "workspace_group": GRAPHITI_WORKSPACE_GROUP,
                "episode_uuid": unit.ingest_id,
                "input_digest": marker_input_digest,
            }
        )
        proof = RecoveredAmbiguousProgressionProof(
            authoritative_attempt_id=authoritative_attempt.attempt_id,
            authoritative_attempt_digest=authoritative_attempt.canonical_digest,
            authoritative_attempt_number=authoritative_number,
            authoritative_run_version_id=authoritative_attempt.run_version_id,
            authoritative_recorded_at=authoritative_attempt.recorded_at,
            input_binding_digest=authoritative_extraction.input_binding_digest,
            skipped_attempt_number=skipped_number,
            skipped_receipt_digest=receipt_digest,
            skipped_ledger_sequence=int(ledger[0]),
            skipped_ledger_digest=str(ledger[3]),
            skipped_recorded_at=skipped_recorded_at,
            settled_usage_evidence_digest=usage_digest,
            recovery_marker_digest=marker_digest,
            marker_attempt_number=authoritative_number,
            marker_workspace_id=authoritative_attempt.workspace_id,
            marker_input_digest=marker_input_digest,
            ingest_id=unit.ingest_id,
        )
        self._recovered_ambiguous_progressions[
            (unit.ingest_id, private_successor_attempt_number)
        ] = proof
        self._authenticated_recovered_gaps[
            (unit.ingest_id, private_successor_attempt_number)
        ] = proof
        return True

    def authenticate_retained_recovered_ambiguous_progression(
        self,
        *,
        unit: CorpusIngestUnit,
        current_attempt: object,
        authoritative_attempt: object,
        next_attempt_number: int,
        connection: sqlite3.Connection,
        model_usage: ModelUsageService,
    ) -> int | None:
        """Authenticate attempt 5's retained gap proof for replay or attempt 6."""

        from newsroom.graphiti_adapter import GraphitiAttemptRecord

        same_attempt_replay = (
            isinstance(current_attempt, GraphitiAttemptRecord)
            and next_attempt_number == current_attempt.attempt_number
        )
        if (
            not isinstance(current_attempt, GraphitiAttemptRecord)
            or current_attempt.attempt_number != GRAPHITI_MAX_FAILURES + 2
            or current_attempt.recovered_ambiguous_progression is None
            or next_attempt_number
            not in {
                current_attempt.attempt_number,
                current_attempt.attempt_number + 1,
            }
            or str(current_attempt.run_id)
            != str(typed_id(ExtractionRunId, "run", unit.ingest_id))
            or (
                not same_attempt_replay
                and current_attempt.outcome.terminal
            )
        ):
            return None
        if not self.prepare_recovered_ambiguous_successor(
            unit=unit,
            authoritative_attempt=authoritative_attempt,
            private_successor_attempt_number=current_attempt.attempt_number,
            connection=connection,
            model_usage=model_usage,
            latest_allowed_attempt_number=current_attempt.attempt_number,
            retained_reentry_attempt_number=(
                current_attempt.attempt_number if same_attempt_replay else None
            ),
        ):
            return None
        key = (unit.ingest_id, current_attempt.attempt_number)
        expected = self._recovered_ambiguous_progressions.pop(key, None)
        self._authenticated_recovered_gaps.pop(key, None)
        if expected != current_attempt.recovered_ambiguous_progression:
            return None
        if next_attempt_number == current_attempt.attempt_number:
            self._recovered_ambiguous_progressions[key] = expected
        self._authenticated_recovered_gaps[
            (unit.ingest_id, next_attempt_number)
        ] = expected
        return int(expected.skipped_attempt_number)

    def ingest(self, unit: CorpusIngestUnit) -> GraphitiCycleResult:
        return self._ingest(unit, deadline=None)

    def ingest_until(
        self, unit: CorpusIngestUnit, *, deadline: datetime
    ) -> GraphitiCycleResult:
        return self._ingest(unit, deadline=deadline)

    def ingest_with_usage(
        self,
        unit: CorpusIngestUnit,
        *,
        model_usage: ModelUsageService,
        cycle_id: str,
        dispatch_authority: Mapping[str, object],
        owner_stop_check: Callable[[], None],
        deadline: datetime | None = None,
    ) -> GraphitiCycleResult:
        envelope = WorkEnvelope.create(
            cycle_id=cycle_id,
            workload_class=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            admitted_at=self._clock().astimezone(UTC),
            admission_decision_id=None,
            candidate_id=None,
            hypothesis_digest=None,
            evidence_package_digest=None,
            ingest_id=unit.ingest_id,
            graphiti_attempt_id=f"{unit.ingest_id}:{unit.attempt_number}",
        )
        envelope = model_usage.resume_or_open_graphiti_envelope(envelope)
        observer = GraphitiModelUsageObserver(
            service=model_usage,
            envelope=envelope,
            clock=self._clock,
            effective_revision_digest=digest_canonical(
                {
                    "source_id": unit.effective_revision.source_id,
                    "item_key": unit.effective_revision.item_key,
                    "revision_digest": unit.effective_revision.revision_digest,
                    "first_observed_at": unit.effective_revision.first_observed_at,
                }
            ),
            ingest_obligation_id=unit.ingest_id,
            provider_attempt_number=unit.attempt_number,
            deadline=deadline,
            dispatch_authority_digest=digest_canonical(dict(dispatch_authority)),
            owner_stop_check=owner_stop_check,
            call_shape_policy=self._call_shape_policy,
            fallback_policy=self._fallback_policy,
            recovered_ambiguous_progression=self._authenticated_recovered_gaps.get(
                (unit.ingest_id, unit.attempt_number)
            ),
            inherited_empty_for=(
                None if self._inherited_empty_for is None else
                lambda attempt: self._inherited_empty_for(unit, attempt)
            ),
            typed_proposal_verifier=(
                None if self._typed_proposal_verifier is None else
                lambda **values: self._typed_proposal_verifier(
                    unit=deepcopy(unit), envelope=envelope, **values
                )
            ),
        )
        self._pending_usage[(unit.ingest_id, unit.attempt_number)] = (
            model_usage,
            envelope,
        )
        result = self._ingest(unit, deadline=deadline, invocation_observer=observer)
        terminal_at = self._clock().astimezone(UTC)
        observer.link_provider_attempts(
            provider_attempt_number=result.provider_attempt_number,
            linked_at=terminal_at,
        )
        return result

    def finalise_usage(
        self,
        unit: CorpusIngestUnit,
        *,
        outcome: str,
        outcome_record_id: str,
        retained_proposal_count: int,
        terminal_at: datetime,
        connection: sqlite3.Connection,
    ) -> None:
        pending = self._pending_usage.pop(
            (unit.ingest_id, unit.attempt_number), None
        )
        if pending is None:
            return
        model_usage, envelope = pending
        model_usage.record_work_outcome(
            envelope_id=envelope.envelope_id,
            outcome=outcome,
            outcome_record_id=outcome_record_id,
            payload_digest=None,
            terminal_at=terminal_at,
            retained_proposal_count=retained_proposal_count,
            connection=connection,
        )

    def _ingest(
        self,
        unit: CorpusIngestUnit,
        *,
        deadline: datetime | None,
        invocation_observer: object | None = None,
    ) -> GraphitiCycleResult:
        from newsroom.graphiti_adapter.real import RealGraphitiAdapter

        temporal = unit.temporal()
        recovered_progression = self._recovered_ambiguous_progressions.get(
            (unit.ingest_id, unit.attempt_number)
        )
        extraction_previous = None
        if (
            self._extraction_records is not None
            and self._proof is not None
            and unit.proving_run_id
            == f"native-source:{unit.observation_digest}"
            and unit.attempt_number > 1
        ):
            if recovered_progression is not None:
                extraction_previous = self._extraction_records.metadata(
                    recovered_progression.authoritative_run_version_id,
                    proof=self._proof,
                )
            else:
                history = self._extraction_records.run_history(
                    typed_id(ExtractionRunId, "run", unit.ingest_id),
                    limit=1,
                    proof=self._proof,
                )
                if len(history) != 1:
                    raise GraphitiResultStageError(
                        GRAPHITI_RESULT_STAGE_ADAPTER_EXECUTION
                    )
                extraction_previous = history[0]
        attempt = self._attempt_for_unit(
            unit, recovered_progression=recovered_progression,
            extraction_previous=extraction_previous,
        )
        if self._proposal_adapter is not None:
            assert self._extraction_records is not None
            assert self._proof is not None
            if (
                (
                    self._fallback_permitted
                    and not self._governed_fallback_permitted
                )
                or deadline is None
                or invocation_observer is None
            ):
                # A governed REAL attempt is safe only when the caller's
                # bounded deadline, fallback policy and usage observer all
                # cross the 4D boundary together.
                raise GraphitiResultStageError(
                    GRAPHITI_RESULT_STAGE_ADAPTER_EXECUTION
                )
            try:
                self._extraction_records.register_contract(
                    attempt.extraction_contract,
                    proof=self._proof,
                )
                self._proposal_adapter.register_configuration(
                    attempt.configuration,
                    proof=self._proof,
                )
            except AuthorizationDenied as exc:
                raise GraphitiPreProviderAuthorizationDenied(
                    "Graphiti adapter authorisation was denied"
                ) from exc
            except GraphitiAdapterContractError as exc:
                raise GraphitiResultStageError(
                    GRAPHITI_RESULT_STAGE_ADAPTER_EXECUTION
                ) from exc
            try:
                retained_attempt = self._proposal_adapter.execute_attempt(
                    attempt,
                    proof=self._proof,
                    execution_deadline=deadline,
                    fallback_permitted=self._fallback_permitted,
                    governed_fallback_permitted=(
                        self._governed_fallback_permitted
                    ),
                    invocation_observer=invocation_observer,
                )
            except AuthorizationDenied as exc:
                raise GraphitiPreProviderAuthorizationDenied(
                    "Graphiti adapter authorisation was denied"
                ) from exc
            except GraphitiAdapterContractError as exc:
                raise GraphitiResultStageError(
                    GRAPHITI_RESULT_STAGE_ADAPTER_EXECUTION
                ) from exc
            try:
                return self._result_from_governed_authority(
                    unit=unit,
                    attempt=attempt,
                    retained_attempt=retained_attempt,
                )
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise GraphitiResultStageError(
                    GRAPHITI_RESULT_STAGE_CYCLE_RESULT_CONSTRUCTION
                ) from exc
        with TemporaryDirectory() as root:
            adapter_options: dict[str, object] = {"execution_deadline": deadline}
            if invocation_observer is not None:
                adapter_options["invocation_observer"] = invocation_observer
            if not self._fallback_permitted:
                adapter_options["fallback_permitted"] = False
            adapter_cls = getattr(type(self), "_adapter_cls", RealGraphitiAdapter)
            adapter = adapter_cls(**adapter_options)
            try:
                execution = adapter.execute(
                    attempt=attempt,
                    workspace_root=Path(root),
                )
            except GraphitiAdapterContractError as exc:
                raise GraphitiResultStageError(
                    GRAPHITI_RESULT_STAGE_ADAPTER_EXECUTION
                ) from exc
        try:
            raw = (
                execution.produced.raw_output_value
                or execution.produced.attempt_receipt_value
            )
            payload = raw if isinstance(raw, dict) else {}
            relations = (
                tuple(payload["relations"])
                if isinstance(payload.get("relations"), list)
                else ()
            )
            entities = (
                tuple(payload["entities"])
                if isinstance(payload.get("entities"), list)
                else ()
            )
            invocations = payload.get("chat_invocations")
            proposal_receipts = payload.get("proposals")
            passage_receipts = payload.get("passages")
            embedding_usage = payload.get("embedding_usage")
            token_usage = payload.get("token_usage")
            usage = execution.produced.usage
            usage_basis = payload.get("usage_basis")
            return GraphitiCycleResult(
                ingest_id=unit.ingest_id,
                source_id=unit.source_id,
                item_key=unit.item_key,
                outcome=execution.outcome.value,
                proposal_count=len(execution.produced.proposals),
                entity_count=len(entities),
                relation_count=len(relations),
                failure_code=execution.failure_code,
                temporal_basis=temporal.basis,
                reference_time=temporal.reference_time.to_text(),
                generation_id=GRAPHITI_GENERATION_ID,
                receipt_digest=str(payload.get("raw_output_digest") or ""),
                episode_uuid=str(payload.get("episode_uuid") or ""),
                entities=entities,
                relations=relations,
                proposals=(
                    tuple(proposal_receipts)
                    if isinstance(proposal_receipts, list)
                    else ()
                ),
                passages=(
                    tuple(passage_receipts)
                    if isinstance(passage_receipts, list)
                    else ()
                ),
                chat_invocations=(
                    tuple(invocations) if isinstance(invocations, list) else ()
                ),
                embedding_usage=(
                    embedding_usage if isinstance(embedding_usage, dict) else None
                ),
                token_usage=token_usage if isinstance(token_usage, dict) else None,
                request_tokens=usage.request_tokens,
                response_tokens=usage.response_tokens,
                cost_microunits=usage.cost_microunits,
                usage_basis=(
                    str(usage_basis)
                    if isinstance(usage_basis, str)
                    else "UNOBSERVED"
                ),
                prompt_version=GRAPHITI_PROMPT_COMPONENT.component_version,
                attempt_number=unit.attempt_number,
                provider_attempt_number=(
                    int(payload["provider_attempt_number"])
                    if isinstance(payload.get("provider_attempt_number"), int)
                    and not isinstance(payload.get("provider_attempt_number"), bool)
                    else unit.attempt_number
                ),
                predecessor_episode_uuid=unit.predecessor_ingest_id,
                raw_receipt=payload,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise GraphitiResultStageError(
                GRAPHITI_RESULT_STAGE_CYCLE_RESULT_CONSTRUCTION
            ) from exc

    def _result_from_governed_authority(
        self,
        *,
        unit: CorpusIngestUnit,
        attempt: object,
        retained_attempt: object,
    ) -> GraphitiCycleResult:
        from newsroom.extraction.models import ProposalDraft
        from newsroom.extraction.types import ExtractionOutcome
        from newsroom.graphiti_adapter import GraphitiAttemptRecord
        from newsroom.graphiti_adapter.models import GraphitiAttemptRequest
        from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome

        if not isinstance(attempt, GraphitiAttemptRequest):
            raise TypeError("governed Graphiti request must be typed")
        if not isinstance(retained_attempt, GraphitiAttemptRecord):
            raise TypeError("governed Graphiti attempt must be typed")
        if (
            retained_attempt.attempt_id != attempt.attempt_id
            or retained_attempt.run_id != attempt.extraction_request.run_id
            or retained_attempt.run_version_id
            != attempt.extraction_request.run_version_id
            or retained_attempt.attempt_number != attempt.attempt_number
            or retained_attempt.previous_attempt_id
            != attempt.expected_previous_attempt_id
            or retained_attempt.configuration_id
            != attempt.configuration.configuration_id
            or retained_attempt.configuration_digest
            != attempt.configuration.canonical_digest
            or retained_attempt.manifest_id != attempt.manifest.manifest_id
        ):
            raise ValueError("governed Graphiti attempt differs from its request")
        assert self._extraction_records is not None
        assert self._proof is not None
        metadata = self._extraction_records.metadata(
            retained_attempt.run_version_id,
            proof=self._proof,
        )
        raw_output = (
            None
            if retained_attempt.output_id is None
            else self._extraction_records.raw_output(
                retained_attempt.output_id,
                proof=self._proof,
            )
        )
        proposals = self._extraction_records.proposals(
            retained_attempt.run_version_id,
            proof=self._proof,
        )
        extraction_outcome = {
            GraphitiAdapterOutcome.COMPLETE: ExtractionOutcome.SUCCESS,
            GraphitiAdapterOutcome.PARTIAL: ExtractionOutcome.PARTIAL,
            GraphitiAdapterOutcome.TIMEOUT: ExtractionOutcome.RETRYABLE_FAILURE,
            GraphitiAdapterOutcome.MALFORMED_OUTPUT: ExtractionOutcome.INVALID_OUTPUT,
            GraphitiAdapterOutcome.PROVIDER_REJECTED: (
                ExtractionOutcome.RETRYABLE_FAILURE
            ),
            GraphitiAdapterOutcome.POLICY_BLOCKED: (
                ExtractionOutcome.BLOCKING_FAILURE
            ),
            GraphitiAdapterOutcome.FAILED: ExtractionOutcome.RETRYABLE_FAILURE,
            GraphitiAdapterOutcome.AMBIGUOUS_EFFECT: (
                ExtractionOutcome.RETRYABLE_FAILURE
            ),
        }[retained_attempt.outcome]
        if (
            metadata.run_id != retained_attempt.run_id
            or metadata.run_version_id != retained_attempt.run_version_id
            or metadata.output != (None if raw_output is None else raw_output.view)
            or metadata.outcome is not extraction_outcome
            or metadata.failure_code.value != retained_attempt.failure_code
            or metadata.usage != retained_attempt.usage
            or metadata.proposal_count != len(proposals)
        ):
            raise ValueError("governed extraction metadata differs from the attempt")
        if any(
            proposal.output_id != retained_attempt.output_id
            or proposal.proposal_set_id != retained_attempt.proposal_set_id
            or proposal.run_id != retained_attempt.run_id
            or proposal.run_version_id != retained_attempt.run_version_id
            for proposal in proposals
        ):
            raise ValueError("governed proposal lineage differs from the attempt")

        if raw_output is None:
            decoded = retained_attempt.attempt_receipt
            if decoded is None:
                raise ValueError("governed Graphiti attempt has no terminal receipt")
        else:
            decoded = json.loads(raw_output.canonical_bytes)
            if not isinstance(decoded, dict):
                raise TypeError("governed Graphiti raw output must be an object")
            if canonical_json_bytes(decoded) != raw_output.canonical_bytes:
                raise ValueError("governed Graphiti raw output is not canonical")
        expected_receipt_binding = {
            "workspace_group": GRAPHITI_WORKSPACE_GROUP,
            "generation_id": attempt.generation_id,
            "episode_uuid": attempt.episode_uuid,
            "attempt_number": attempt.attempt_number,
            "predecessor_episode_uuid": attempt.predecessor_episode_uuid,
            "temporal_basis": attempt.temporal_basis,
            "reference_time": (
                None
                if attempt.reference_time is None
                else attempt.reference_time.to_text()
            ),
            "passages": [item.canonical_value() for item in attempt.manifest.passages],
        }
        if any(
            decoded.get(field) != expected
            for field, expected in expected_receipt_binding.items()
        ):
            raise ValueError("governed Graphiti raw output differs from its manifest")
        unsigned = dict(decoded)
        receipt_digest = unsigned.pop("raw_output_digest", None)
        if not isinstance(receipt_digest, str) or receipt_digest != digest_bytes(
            canonical_json_bytes(unsigned)
        ):
            raise ValueError("governed Graphiti raw receipt digest differs")

        proposal_values = decoded.get("proposals")
        if not isinstance(proposal_values, list) or any(
            not isinstance(item, dict) for item in proposal_values
        ):
            raise TypeError("governed Graphiti raw proposals must be objects")
        retained_drafts = tuple(
            ProposalDraft(
                local_id=item.local_id,
                kind=item.kind,
                subject_placeholder=item.subject_placeholder,
                object_placeholder=item.object_placeholder,
                predicate_hint=item.predicate_hint,
                confidence_basis_points=item.confidence_basis_points,
                uncertainty_codes=item.uncertainty_codes,
                rationale_codes=item.rationale_codes,
                evidence=item.evidence,
            ).canonical_value()
            for item in proposals
        )
        proposal_bearing_outcome = retained_attempt.outcome in {
            GraphitiAdapterOutcome.COMPLETE,
            GraphitiAdapterOutcome.PARTIAL,
        }
        if proposal_bearing_outcome:
            if tuple(proposal_values) != retained_drafts:
                raise ValueError(
                    "governed raw proposals differ from ProposalEnvelope authority"
                )
        elif retained_drafts:
            raise ValueError(
                "non-admissible Graphiti outcome retained ProposalEnvelope authority"
            )

        relations = decoded.get("relations")
        entities = decoded.get("entities")
        invocations = decoded.get("chat_invocations")
        passages = decoded.get("passages")
        embedding_usage = decoded.get("embedding_usage")
        token_usage = decoded.get("token_usage")
        if not isinstance(relations, list) or not isinstance(entities, list):
            raise TypeError(
                "governed Graphiti entity or relation receipts are malformed"
            )
        temporal = unit.temporal()
        usage_basis = decoded.get("usage_basis")
        return GraphitiCycleResult(
            ingest_id=unit.ingest_id,
            source_id=unit.source_id,
            item_key=unit.item_key,
            outcome=retained_attempt.outcome.value,
            proposal_count=len(proposals),
            entity_count=len(entities),
            relation_count=len(relations),
            failure_code=retained_attempt.failure_code,
            temporal_basis=temporal.basis,
            reference_time=temporal.reference_time.to_text(),
            generation_id=attempt.generation_id,
            receipt_digest=receipt_digest,
            workspace_group=GRAPHITI_WORKSPACE_GROUP,
            episode_uuid=str(decoded.get("episode_uuid") or ""),
            entities=tuple(entities),
            relations=tuple(relations),
            proposals=retained_drafts,
            passages=tuple(passages) if isinstance(passages, list) else (),
            chat_invocations=(
                tuple(invocations) if isinstance(invocations, list) else ()
            ),
            embedding_usage=(
                embedding_usage if isinstance(embedding_usage, dict) else None
            ),
            token_usage=token_usage if isinstance(token_usage, dict) else None,
            request_tokens=retained_attempt.usage.request_tokens,
            response_tokens=retained_attempt.usage.response_tokens,
            cost_microunits=retained_attempt.usage.cost_microunits,
            usage_basis=(
                usage_basis if isinstance(usage_basis, str) else "UNOBSERVED"
            ),
            prompt_version=GRAPHITI_PROMPT_COMPONENT.component_version,
            attempt_number=retained_attempt.attempt_number,
            provider_attempt_number=(
                int(decoded["provider_attempt_number"])
                if isinstance(decoded.get("provider_attempt_number"), int)
                and not isinstance(decoded.get("provider_attempt_number"), bool)
                else retained_attempt.attempt_number
            ),
            predecessor_episode_uuid=attempt.predecessor_episode_uuid,
            raw_receipt=decoded,
        )


__all__ = [
    "EvaluationGraphitiRunner",
    "GraphitiCycleResult",
    "GraphitiPreProviderAuthorizationDenied",
    "GraphitiResultStageError",
    "GRAPHITI_RESULT_FAILURE_STAGES",
    "GRAPHITI_RESULT_STAGE_ADAPTER_EXECUTION",
    "GRAPHITI_RESULT_STAGE_CYCLE_RESULT_BINDING",
    "GRAPHITI_RESULT_STAGE_CYCLE_RESULT_CONSTRUCTION",
    "GRAPHITI_RESULT_STAGE_UNCLASSIFIED",
    "GraphitiModelUsageObserver",
    "GraphitiPort",
]
