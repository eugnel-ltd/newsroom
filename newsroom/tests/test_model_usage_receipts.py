from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from pathlib import Path

import pytest

import newsroom.control_plane.issue_790_canary as issue_790_canary_module
import newsroom.control_plane.issue_790_contract as issue_790_contract_module
import newsroom.control_plane.issue_790_disposition as issue_790_operation
import newsroom.control_plane.model_usage as model_usage_module
from newsroom.authority.canonical import digest_bytes, digest_canonical
from newsroom.control_plane.graphiti import (
    GRAPHITI_CHAT_PRIMARY_ROUTE,
    GRAPHITI_CONTEXT_IDENTITY,
    GRAPHITI_EMBEDDING_ROUTE,
    GraphitiModelUsageObserver,
)
from newsroom.control_plane.graphiti_requests import load_checked_graphiti_call_shape_policy
from newsroom.control_plane.graphiti_events import (
    GraphitiEventQueue,
    GraphitiProcessResult,
)
from newsroom.control_plane.issue_790_disposition import (
    ISSUE_790_PLAN_SCHEMA,
    Issue790DispositionError,
    apply_issue_790_plan,
    assert_issue_790_paths_disjoint,
    dry_run_issue_790_plan,
    load_issue_790_plan,
    run_issue_790_canary,
    validate_issue_790_plan,
    write_issue_790_receipt,
)
from newsroom.control_plane.issue_790_canary import (
    Issue790CanaryIntegrityError,
    Issue790CanaryRepository,
)
from newsroom.control_plane.issue_790_prepared_canary import (
    CANDIDATE_EVENT_ID,
    CANDIDATE_LEDGER_SEQ,
    PreparedCanaryError,
    prepare_issue_790_canary,
    prepared_canary_from_record,
    prepared_canary_record,
)
from newsroom.tests.test_issue_790_rehearsal_fixtures import (
    SUCCESSOR_EVENT_ID,
    SUCCESSOR_LEDGER_SEQ,
)
from newsroom.control_plane.model_usage import (
    GraphitiIngestRetryEvidence,
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
    native_graphiti_usage_cycle_id,
)
from newsroom.graphiti_adapter.cli_process import timeout_diagnostic
from newsroom.graphiti_adapter.contracts import GRAPHITI_PROMPT_COMPONENT
from newsroom.graphiti_adapter.evaluation_packet import (
    CURSOR_AGENT_MODEL_ID,
    OPENROUTER_EMBEDDING_SLUG,
)
from newsroom.control_plane.native_progress import NativeRevisionJournal
from newsroom.control_plane.store import (
    append_ledger,
    connect as connect_unpublished_store,
    insert_graphiti_attempt_receipt,
)
from newsroom.tests.test_native_graphiti import _native

@pytest.fixture(autouse=True)
def _graphiti_test_deployment_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    # These fake-provider observers model a deployed tree, not pytest artefacts.
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti._graphiti_implementation_identity",
        lambda: ("a" * 40, True),
    )


T0 = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
FIXTURE_790_PLAN_DIGEST = digest_canonical(
    {"issue": 790, "authority": "fixture-only"}
)
RETRY_FORBIDDEN_EVENTS = [
    {
        "attempt_count": 1,
        "available_at": "2026-08-26T12:25:29.807056Z",
        "event_id": (
            "sha256:bacb9104c81dd86ca3f62a39f6c386cd4d84ab470e9675e31acf8e2feb50443e"
        ),
        "last_failure_code": "PRODUCER_INTERNAL_ERROR",
        "ledger_seq": 1932,
        "provider_dispatched": True,
        "state": "RETRY_HELD",
    },
    {
        "attempt_count": 1,
        "available_at": "2026-08-26T13:52:15.763233Z",
        "event_id": (
            "sha256:de7bb58fde4829f4778936e7c5ebd1dd583a63f8658fb6af2fcb4b6fc873b0d5"
        ),
        "last_failure_code": "PRODUCER_INTERNAL_ERROR",
        "ledger_seq": 1972,
        "provider_dispatched": False,
        "state": "RETRY_HELD",
    },
]


def _digest(value: object) -> str:
    return digest_canonical(value)


def _service(tmp_path: Path) -> ModelUsageService:
    return ModelUsageService(str(tmp_path / "unpublished.sqlite3"))


def _land_native_ingest(tmp_path: Path):
    path = tmp_path / "unpublished.sqlite3"
    connection = connect_unpublished_store(str(path))
    unit = _native()
    NativeRevisionJournal(connection).land((unit,))
    connection.close()
    return unit


def _unique_consumption_columns(path: Path) -> set[tuple[str, ...]]:
    connection = sqlite3.connect(path)
    try:
        return {
            tuple(
                str(column[2])
                for column in connection.execute(f'PRAGMA index_info("{index[1]}")')
            )
            for index in connection.execute(
                "PRAGMA index_list(issue_790_bounded_canary_consumptions)"
            )
            if index[2]
        }
    finally:
        connection.close()


def test_issue_790_migrates_legacy_unique_disposition_constraint(
    tmp_path: Path,
) -> None:
    _service(tmp_path)
    path = tmp_path / "unpublished.sqlite3"
    Issue790CanaryRepository(str(path))
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE UNIQUE INDEX issue_790_legacy_disposition_unique ON "
        "issue_790_bounded_canary_consumptions(disposition_digest)"
    )
    connection.commit()
    connection.close()

    Issue790CanaryRepository(str(path))

    connection = sqlite3.connect(path)
    unique_columns = {
        tuple(
            str(column[2])
            for column in connection.execute(f'PRAGMA index_info("{index[1]}")')
        )
        for index in connection.execute(
            "PRAGMA index_list(issue_790_bounded_canary_consumptions)"
        )
        if index[2]
    }
    foreign_key_failure = connection.execute(
        "PRAGMA foreign_key_check"
    ).fetchone()
    connection.close()
    assert ("disposition_digest",) not in unique_columns
    assert foreign_key_failure is None


def test_issue_790_preserves_legacy_unique_plan_consumption_constraint(
    tmp_path: Path,
) -> None:
    _service(tmp_path)
    path = tmp_path / "unpublished.sqlite3"
    Issue790CanaryRepository(str(path))
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE UNIQUE INDEX issue_790_legacy_plan_consumption_unique ON "
        "issue_790_bounded_canary_consumptions(approved_plan_digest)"
    )
    connection.commit()
    connection.close()

    Issue790CanaryRepository(str(path))

    unique_columns = _unique_consumption_columns(path)
    connection = sqlite3.connect(path)
    foreign_key_failure = connection.execute(
        "PRAGMA foreign_key_check"
    ).fetchone()
    connection.close()
    assert ("approved_plan_digest",) in unique_columns
    assert ("event_id",) in unique_columns
    assert ("ledger_seq",) in unique_columns
    assert ("owner_id",) in unique_columns
    assert ("disposition_digest",) not in unique_columns
    assert foreign_key_failure is None


def test_v1_store_replays_v2_context_manifest_migration_idempotently(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unpublished.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE model_usage_migrations("
        "migration_id TEXT PRIMARY KEY,schema_version TEXT NOT NULL,"
        "applied_at TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO model_usage_migrations VALUES(?,?,?)",
        ("model-usage-v1", "newsroom.model-usage.v1", T0.isoformat()),
    )
    connection.commit()
    connection.close()

    ModelUsageService(str(path))
    ModelUsageService(str(path))
    Issue790CanaryRepository(str(path))

    connection = sqlite3.connect(path)
    migrations = connection.execute(
        "SELECT migration_id,schema_version FROM model_usage_migrations "
        "ORDER BY migration_id"
    ).fetchall()
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    connection.close()
    assert migrations == [
        ("model-usage-v1", "newsroom.model-usage.v1"),
        ("model-usage-v2", "newsroom.model-usage.v2"),
        ("model-usage-v3", "newsroom.model-usage.v3"),
        (
            "model-usage-v4-conservative-disposition",
            "newsroom.model-usage.v4",
        ),
        ("model-usage-v5-reported-output-disposition", "newsroom.model-usage.v5"),
    ]
    assert "model_invocation_context_manifests" in tables
    assert "model_invocation_context_observations" in tables
    assert "graphiti_internal_requests" in tables
    assert "graphiti_internal_request_refusals" in tables
    assert "model_usage_conservative_dispositions" in tables
    assert "model_usage_reported_output_dispositions" in tables
    assert "issue_790_graphiti_retry_exclusions" in tables
    assert "issue_790_bounded_canary_consumptions" in tables
    assert "issue_790_bounded_canary_outcomes" in tables


def _policy(
    *,
    workload: WorkloadClass = WorkloadClass.CONT_WRITER_PRIMARY,
    provider: str = "grok-build-cli",
    route: str = "CONT_PRIMARY",
    model: str = "grok-4.6",
    hard_estimate_ceiling_tokens: int | None = 2_500,
) -> InvocationEfficiencyPolicy:
    return InvocationEfficiencyPolicy.create(
        policy_id=f"policy-{workload.value.lower()}",
        version="v1",
        workload_class=workload,
        provider=provider,
        route=route,
        model=model,
        reasoning="low",
        one_turn=True,
        exact_input=True,
        skills_enabled=False,
        tools_enabled=False,
        mcp_enabled=False,
        prior_message_count=0,
        max_prompt_bytes=4_096,
        max_context_tokens=2_000,
        max_output_tokens=500,
        max_total_tokens=2_500,
        prompt_contract_version="prompt-v1",
        output_schema_digest=_digest({"schema": 1}),
        allowed_context_identities=("context-v1",),
        allowed_config_identities=("config-v1",),
        hard_estimate_ceiling_tokens=hard_estimate_ceiling_tokens,
        evidence_digest=_digest({"qualification": "fixture"}),
        qualified=True,
    )


def _envelope(
    *,
    cycle_id: str = "00000000-0000-4000-8000-000000000001",
    workload: WorkloadClass = WorkloadClass.CONT_WRITER_PRIMARY,
    candidate_id: str | None = "candidate-1",
    ingest_id: str | None = None,
) -> WorkEnvelope:
    return WorkEnvelope.create(
        cycle_id=cycle_id,
        workload_class=workload,
        admitted_at=T0,
        admission_decision_id=("decision-1" if candidate_id else None),
        candidate_id=candidate_id,
        hypothesis_digest=(_digest({"hypothesis": 1}) if candidate_id else None),
        evidence_package_digest=(_digest({"evidence": 1}) if candidate_id else None),
        ingest_id=ingest_id,
        graphiti_attempt_id=(f"{ingest_id}:1" if ingest_id else None),
    )


def _allocation(
    envelope: WorkEnvelope,
    policy: InvocationEfficiencyPolicy,
    *,
    leaf_ordinal: int = 1,
    request: str = "request-1",
    workload: WorkloadClass | None = None,
    parent_invocation_id: str | None = None,
    prompt_bytes: int = 200,
    config_identity: str = "config-v1",
) -> InvocationAllocation:
    return InvocationAllocation.create(
        envelope_id=envelope.envelope_id,
        cycle_id=envelope.cycle_id,
        leaf_ordinal=leaf_ordinal,
        workload_class=workload or policy.workload_class,
        invocation_policy_digest=policy.canonical_digest,
        provider=policy.provider,
        route=policy.route,
        model=policy.model,
        reasoning=policy.reasoning,
        prompt_contract_version=policy.prompt_contract_version,
        prompt_bytes=prompt_bytes,
        prompt_digest=_digest({"prompt": request}),
        request_digest=_digest({"request": request}),
        output_schema_digest=policy.output_schema_digest,
        max_output_tokens=policy.max_output_tokens,
        context_manifest_digest=_digest({"context": "v1"}),
        context_identity="context-v1",
        config_identity=config_identity,
        one_turn=policy.one_turn,
        exact_input=policy.exact_input,
        skills_enabled=policy.skills_enabled,
        tools_enabled=policy.tools_enabled,
        mcp_enabled=policy.mcp_enabled,
        prior_message_count=policy.prior_message_count,
        allocated_at=T0 + timedelta(seconds=leaf_ordinal),
        recovery_deadline_at=T0 + timedelta(seconds=30 + leaf_ordinal),
        parent_invocation_id=parent_invocation_id,
    )


def _reported(
    allocation: InvocationAllocation,
    *,
    total: int = 125,
    outcome: str = "ACCEPTED",
    completed_at: datetime | None = None,
) -> InvocationTerminal:
    return InvocationTerminal.create(
        invocation_id=allocation.invocation_id,
        outcome=outcome,
        failure_class=None,
        usage_status=UsageStatus.REPORTED,
        components=UsageComponents(
            input_tokens=total - 25,
            output_tokens=20,
            cached_read_tokens=5,
            reasoning_tokens=0,
            context_tokens=100,
            total_tokens=total,
            provenance="PROVIDER_REPORTED",
        ),
        dispatch_at=allocation.allocated_at + timedelta(milliseconds=1),
        completed_at=completed_at or allocation.allocated_at + timedelta(seconds=1),
        observed_at=completed_at or allocation.allocated_at + timedelta(seconds=1),
        provider_telemetry_digest=_digest({"invocation": allocation.invocation_id}),
        raw_telemetry_pointer=f"private://telemetry/{allocation.invocation_id}",
        subscription_cli_chat_not_cash_debited=True,
    )


def _open_and_allocate(
    service: ModelUsageService,
    *,
    envelope: WorkEnvelope | None = None,
    policy: InvocationEfficiencyPolicy | None = None,
) -> tuple[WorkEnvelope, InvocationEfficiencyPolicy, InvocationAllocation]:
    envelope = envelope or _envelope()
    policy = policy or _policy(workload=envelope.workload_class)
    service.register_policy(policy)
    service.open_envelope(envelope)
    allocation = _allocation(envelope, policy)
    service.allocate(allocation, owner_emergency_stop=False)
    return envelope, policy, allocation


def _open_unreported_graphiti_subscription_leaf(
    service: ModelUsageService,
    *,
    cycle_id: str = "graphiti-cycle-unreported",
    request: str = "graphiti-unreported-request",
    failure_class: str = "MISSING_PROVIDER_TELEMETRY",
    outcome: str = "FAILED",
    observe_dispatch: bool = True,
    subscription_not_cash_debited: bool = True,
    elapsed_ms: int = 999,
    ingest_id: str | None = None,
) -> tuple[InvocationEfficiencyPolicy, InvocationAllocation, InvocationTerminal]:
    envelope = _envelope(
        cycle_id=cycle_id,
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        candidate_id=None,
        ingest_id=ingest_id or f"ingest-{cycle_id}",
    )
    policy = _policy(
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        provider="cursor-agent-cli",
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        model="composer-2.5",
        hard_estimate_ceiling_tokens=None,
    )
    _envelope_value, _policy_value, allocation = _open_and_allocate(
        service,
        envelope=envelope,
        policy=policy,
    )
    dispatch_at = allocation.allocated_at + timedelta(milliseconds=1)
    if observe_dispatch:
        service.observe_transport(
            invocation_id=allocation.invocation_id,
            observed_at=dispatch_at,
            state="DISPATCH_STARTED",
            evidence_digest=_digest({"dispatch": allocation.invocation_id}),
        )
    terminal = service.complete(
        InvocationTerminal.create(
            invocation_id=allocation.invocation_id,
            outcome=outcome,
            failure_class=failure_class,
            usage_status=UsageStatus.UNREPORTED,
            components=UsageComponents(provenance="UNAVAILABLE"),
            dispatch_at=dispatch_at,
            completed_at=dispatch_at + timedelta(milliseconds=elapsed_ms),
            observed_at=dispatch_at + timedelta(milliseconds=elapsed_ms),
            subscription_cli_chat_not_cash_debited=(
                subscription_not_cash_debited
            ),
        )
    )
    return policy, allocation, terminal


def _conservative_disposition_authority(
    allocation: InvocationAllocation,
    terminal: InvocationTerminal,
    *,
    approved_at: datetime,
    approved_by: str = "github:fol2",
    approval_reference: str = (
        "https://github.com/fol2/newsroom/issues/790#issuecomment-fixture"
    ),
    approved_plan_digest: str = FIXTURE_790_PLAN_DIGEST,
) -> tuple[dict[str, object], str]:
    record = {
        "schema_version": (
            "newsroom.model-usage.conservative-disposition-authority.v2"
        ),
        "approved_plan_digest": approved_plan_digest,
        "approved_by": approved_by,
        "approval_reference": approval_reference,
        "approved_at": approved_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "invocation_id": allocation.invocation_id,
        "terminal_digest": terminal.terminal_digest,
        "allocation_digest": allocation.canonical_digest,
        "scope": "CONSERVATIVE_SUBSCRIPTION_CLI_USAGE_DISPOSITION",
    }
    return record, _digest(record)


def _issue_790_plan(
    policy: InvocationEfficiencyPolicy,
    allocation: InvocationAllocation,
    terminal: InvocationTerminal,
) -> dict[str, object]:
    plan: dict[str, object] = {
        "schema_version": ISSUE_790_PLAN_SCHEMA,
        "issue": 790,
        "approval": {
            "approved_by": "github:fol2",
            "approval_reference": (
                "https://github.com/fol2/newsroom/issues/790#issuecomment-fixture"
            ),
            "approved_at": "2026-08-24T10:00:05.000000Z",
            "scope": "CONSERVATIVE_SUBSCRIPTION_CLI_USAGE_DISPOSITION",
        },
        "target": {
            "invocation_id": allocation.invocation_id,
            "terminal_digest": terminal.terminal_digest,
            "allocation_digest": allocation.canonical_digest,
            "policy_digest": policy.canonical_digest,
            "route": GRAPHITI_CHAT_PRIMARY_ROUTE,
            "provider": "cursor-agent-cli",
            "workload_class": "GRAPHITI_CHAT_PRIMARY",
            "terminal_usage_status": "UNREPORTED",
            "terminal_failure_class": "MISSING_PROVIDER_TELEMETRY",
            "route_open_reason": "SYSTEMIC_TRANSPORT",
            "conservative_total_source": "QUALIFIED_POLICY_MAX_TOTAL_TOKENS",
            "expected_conservative_total_tokens": policy.max_total_tokens,
        },
        "release": {
            "kind": "AUTHORISED_OPERATOR_RESET",
            "evidence": "CONSERVATIVE_DISPOSITION_DIGEST",
        },
        "retry_forbidden_events": RETRY_FORBIDDEN_EVENTS,
        "canary": {
            "authority_consumption": "APPEND_ONLY_SINGLE_USE_BEFORE_PROVIDER_IO",
            "event_binding": "EXPLICIT_QUEUED_ATTEMPT_ZERO_EVENT",
            "fresh_provider_backed_attempt_count": 1,
            "persistent_worker_state_before_canary": "UNLOADED",
            "requires_exact_main_deployment": True,
        },
        "non_effects": [
            "NO_PUBLICATION",
            "NO_PUBLIC_DISPATCH",
            "NO_BACKLOG_DRAIN",
            "NO_BULK_REQUEUE",
            "NO_PRODUCTION_OPERATIONAL_ADMISSION",
            "NO_WIDER_ACTIVATION",
            "NO_PROVIDER_SUBSTITUTION",
            "NO_MODEL_SUBSTITUTION",
            "NO_TOKEN_LIMIT_REMOVAL",
            "NO_UNRELATED_SPEND_DISPOSITION",
        ],
    }
    plan["canonical_digest"] = _digest(plan)
    return plan


def _issue_790_controller_timeout_report_fixture(
    terminal: InvocationTerminal,
    *,
    configured_timeout_ms: int,
) -> dict[str, object]:
    assert terminal.dispatch_at is not None
    diagnostic = timeout_diagnostic(
        boundary="CONTROLLER_DEADLINE",
        phase="PRIMARY_TRANSPORT",
        cause="CONFIGURED_TIMEOUT_EXPIRED",
        configured_timeout_ms=configured_timeout_ms,
        elapsed_ms=configured_timeout_ms,
        deadline_at=(
            terminal.dispatch_at
            + timedelta(milliseconds=configured_timeout_ms + 1)
        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        last_progress="OUTPUT_OBSERVED",
        termination="PROCESS_KILLED",
        process="CLI_CHILD",
        stdout=b"bounded progress",
        stderr=b"",
    )
    causal_report: dict[str, object] = {
        "schema_version": "newsroom.issue-790.causal-report.v1",
        "classification": "CONTROLLER_TIMEOUT",
        "causal_constraint": "CONTROLLER_TIMEOUT_MS",
        "local_cause": "CONFIGURED_TIMEOUT_EXPIRED",
        "provider_cause": "UNOBSERVED",
        "diagnostic_reference": "fixture:issue-790-controller-timeout",
        "diagnostic": diagnostic,
    }
    causal_report["report_digest"] = _digest(causal_report)
    return causal_report


def _issue_790_successor_plan(
    policy: InvocationEfficiencyPolicy,
    allocation: InvocationAllocation,
    terminal: InvocationTerminal,
    *,
    approved_at: datetime,
    predecessor_consumption: dict[str, object],
    predecessor_outcome: dict[str, object],
    sequence_ordinal: int = 1,
    controller_timeout_ms: int = 160_000,
    extraction_timeout_ms: int = 180_000,
    predecessor_controller_timeout_ms: int = 80_000,
    root_plan_digest: str | None = None,
    fixed_constraints_digest: str | None = None,
    call_shape_policy_digest: str = (
        "sha256:7e6bd15613cefda0820a1d339c8790f0185946aade1622dadbb8c468f558bb18"
    ),
    call_shape_policy_version: str = "issue-790-v9",
) -> dict[str, object]:
    plan = _issue_790_plan(policy, allocation, terminal)
    plan.pop("canonical_digest")
    plan["schema_version"] = "newsroom.issue-790.iterative-canary-plan.v2"
    approval = dict(plan["approval"])  # type: ignore[arg-type]
    approval["approved_at"] = approved_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    plan["approval"] = approval
    target = dict(plan["target"])  # type: ignore[arg-type]
    target["terminal_outcome"] = terminal.outcome
    target["route_open_reason"] = "TIMEOUT"
    plan["target"] = target
    canary = dict(plan["canary"])  # type: ignore[arg-type]
    canary["fallback_mode"] = "DISABLED_BEFORE_PROVIDER_DISPATCH"
    plan["canary"] = canary
    if sequence_ordinal == 1:
        causal_report = _issue_790_controller_timeout_report_fixture(
            terminal,
            configured_timeout_ms=predecessor_controller_timeout_ms,
        )
    else:
        causal_report = dict(predecessor_outcome["causal_report"])  # type: ignore[arg-type]
    retained_fixed_digest = fixed_constraints_digest or _digest(
        {"fixture": "fixed-constraints"}
    )
    plan["sequence"] = {
        "sequence_ordinal": sequence_ordinal,
        "stop_condition": "FIRST_TRUTHFUL_PROVIDER_BACKED_SUCCESS",
        "constraint_change": (
            "INITIAL_QUALIFIED_BASELINE"
            if sequence_ordinal == 1
            else "CONTROLLER_TIMEOUT_INCREMENT"
        ),
        "controller_timeout_ms": controller_timeout_ms,
        "extraction_timeout_ms": extraction_timeout_ms,
        "cleanup_reserve_ms": 20_000,
        "timeout_increment_ms": 10_000,
        "call_shape_policy_digest": call_shape_policy_digest,
        "call_shape_policy_version": call_shape_policy_version,
        "fixed_constraints_digest": retained_fixed_digest,
        "root_plan_digest": (
            predecessor_consumption["approved_plan_digest"]
            if root_plan_digest is None
            else root_plan_digest
        ),
        "predecessor": {
            "plan_digest": predecessor_consumption["approved_plan_digest"],
            "consumption_digest": predecessor_consumption["consumption_digest"],
            "outcome_digest": predecessor_outcome["outcome_digest"],
            "event_id": predecessor_consumption["event_id"],
            "ledger_seq": predecessor_consumption["ledger_seq"],
        },
        "predecessor_causal_report": causal_report,
    }
    plan["canonical_digest"] = _digest(plan)
    return plan


def _bind_issue_790_fixture_contract(
    monkeypatch: pytest.MonkeyPatch,
    *,
    allocation: InvocationAllocation,
    terminal: InvocationTerminal,
    plan_digest: str = FIXTURE_790_PLAN_DIGEST,
) -> None:
    monkeypatch.setattr(
        issue_790_contract_module,
        "ISSUE_790_APPROVED_PLAN_DIGEST",
        plan_digest,
    )
    monkeypatch.setattr(
        issue_790_contract_module,
        "ISSUE_790_APPROVED_INVOCATION_ID",
        allocation.invocation_id,
    )
    monkeypatch.setattr(
        issue_790_contract_module,
        "ISSUE_790_APPROVED_TERMINAL_DIGEST",
        terminal.terminal_digest,
    )
    monkeypatch.setattr(
        issue_790_contract_module,
        "ISSUE_790_APPROVED_ALLOCATION_DIGEST",
        allocation.canonical_digest,
    )
    monkeypatch.setattr(
        issue_790_contract_module,
        "ISSUE_790_APPROVED_TERMINAL_OUTCOME",
        terminal.outcome,
    )
    monkeypatch.setattr(
        issue_790_contract_module,
        "ISSUE_790_APPROVED_BY",
        "github:fol2",
    )
    monkeypatch.setattr(
        issue_790_contract_module,
        "ISSUE_790_APPROVAL_REFERENCE",
        "https://github.com/fol2/newsroom/issues/790#issuecomment-fixture",
    )
    monkeypatch.setattr(
        issue_790_contract_module,
        "ISSUE_790_APPROVED_AT",
        "2026-08-24T10:00:05.000000Z",
    )
    monkeypatch.setattr(
        issue_790_operation,
        "ISSUE_790_APPROVED_PLAN_DIGEST",
        plan_digest,
    )


def _issue_790_successor_fixture_contract(
    *,
    allocation: InvocationAllocation,
    terminal: InvocationTerminal,
    plan: Mapping[str, object],
    approved_at: datetime,
) -> issue_790_contract_module.Issue790ApprovedPlanContract:
    sequence = dict(plan["sequence"])  # type: ignore[arg-type]
    predecessor = dict(sequence["predecessor"])  # type: ignore[arg-type]
    causal_report = dict(sequence["predecessor_causal_report"])  # type: ignore[arg-type]
    return issue_790_contract_module.Issue790ApprovedPlanContract(
        schema_version="newsroom.issue-790.iterative-canary-plan.v2",
        plan_digest=str(plan["canonical_digest"]),
        invocation_id=allocation.invocation_id,
        terminal_digest=terminal.terminal_digest,
        allocation_digest=allocation.canonical_digest,
        approved_by="github:fol2",
        approval_reference=(
            "https://github.com/fol2/newsroom/issues/790#issuecomment-fixture"
        ),
        approved_at=approved_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        scope="CONSERVATIVE_SUBSCRIPTION_CLI_USAGE_DISPOSITION",
        terminal_outcome=terminal.outcome,
        route_open_reason="TIMEOUT",
        root_plan_digest=str(sequence["root_plan_digest"]),
        predecessor_plan_digest=str(predecessor["plan_digest"]),
        sequence_ordinal=int(sequence["sequence_ordinal"]),
        controller_timeout_ms=int(sequence["controller_timeout_ms"]),
        extraction_timeout_ms=int(sequence["extraction_timeout_ms"]),
        cleanup_reserve_ms=int(sequence["cleanup_reserve_ms"]),
        fixed_constraints_digest=str(sequence["fixed_constraints_digest"]),
        predecessor_causal_report_digest=str(causal_report["report_digest"]),
        constraint_change=str(sequence["constraint_change"]),
        reviewed_fix_digest=(
            None
            if sequence.get("reviewed_fix") is None
            else str(dict(sequence["reviewed_fix"])["record_digest"])
        ),
    )


def _bind_issue_790_successor_fixture_contract(
    monkeypatch: pytest.MonkeyPatch,
    *,
    allocation: InvocationAllocation,
    terminal: InvocationTerminal,
    plan: Mapping[str, object],
    approved_at: datetime,
) -> issue_790_contract_module.Issue790ApprovedPlanContract:
    contract = _issue_790_successor_fixture_contract(
        allocation=allocation,
        terminal=terminal,
        plan=plan,
        approved_at=approved_at,
    )
    monkeypatch.setattr(
        issue_790_contract_module,
        "_SUCCESS_SEQUENCE_CONTRACTS",
        (contract,),
    )
    return contract


def _seed_issue_790_retry_events(path: Path) -> None:
    connection = connect_unpublished_store(str(path))
    try:
        for item in RETRY_FORBIDDEN_EVENTS:
            manifest = {
                "event_type": "EFFECTIVE_SOURCE_REVISION_LANDED",
                "ledger_seq": item["ledger_seq"],
                "ledger_digest": item["event_id"],
                "landed_ingest_ids": [],
                "landed_payload_digest": _digest(
                    {"ledger_seq": item["ledger_seq"]}
                ),
                "unit_refs": [],
            }
            connection.execute(
                "INSERT INTO unpublished_graphiti_revision_events("
                "event_id,ledger_seq,ledger_digest,source_id,item_key,"
                "revision_digest,published_at,updated_at,landed_at,manifest_json,"
                "manifest_digest,unit_count,projector_version,"
                "projection_generation,state,attempt_count,available_at,"
                "last_failure_code,provider_dispatched) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    item["event_id"],
                    item["ledger_seq"],
                    item["event_id"],
                    f"source-{item['ledger_seq']}",
                    f"item-{item['ledger_seq']}",
                    _digest({"revision": item["ledger_seq"]}),
                    "",
                    "",
                    T0.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                    _digest(manifest),
                    0,
                    "test-projector",
                    "test-projection",
                    item["state"],
                    item["attempt_count"],
                    item["available_at"],
                    item["last_failure_code"],
                    int(bool(item["provider_dispatched"])),
                ),
            )
        connection.commit()
    finally:
        connection.close()


def _fixture_issue_790_unit_ref(ledger_seq: int) -> dict[str, object]:
    return {
        "ingest_id": f"fresh-ingest-{ledger_seq}",
        "revision_id": f"revision-{ledger_seq}",
        "representation_digest": _digest({"representation": ledger_seq}),
        "chunk_digest": _digest({"chunk": ledger_seq}),
        "chunk_ordinal": 1,
        "predecessor_ingest_id": None,
    }


def _seed_fresh_issue_790_event(
    path: Path,
    *,
    ledger_seq: int = 2001,
    retain_unit_refs: bool = True,
    event_id: str | None = None,
) -> tuple[str, str]:
    event_id = event_id or _digest({"fresh-event": ledger_seq})
    ingest_id = f"fresh-ingest-{ledger_seq}"
    unit_ref = _fixture_issue_790_unit_ref(ledger_seq)
    manifest = {
        "event_type": "EFFECTIVE_SOURCE_REVISION_LANDED",
        "ledger_seq": ledger_seq,
        "ledger_digest": event_id,
        "landed_ingest_ids": [ingest_id],
        "landed_payload_digest": _digest({"landed": ledger_seq}),
        "unit_refs": [unit_ref] if retain_unit_refs else [],
    }
    connection = connect_unpublished_store(str(path))
    try:
        connection.execute(
            "INSERT INTO unpublished_graphiti_revision_events("
            "event_id,ledger_seq,ledger_digest,source_id,item_key,revision_digest,"
            "published_at,updated_at,landed_at,manifest_json,manifest_digest,"
            "unit_count,projector_version,projection_generation,state,"
            "attempt_count,available_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                event_id,
                ledger_seq,
                event_id,
                f"source-{ledger_seq}",
                f"item-{ledger_seq}",
                _digest({"revision": ledger_seq}),
                "",
                "",
                T0.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                _digest(manifest),
                int(retain_unit_refs),
                "test-projector",
                "test-projection",
                "QUEUED",
                0,
                T0.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return event_id, ingest_id


def _released_issue_790_canary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object], dict[str, object], Issue790CanaryRepository]:
    service = _service(tmp_path)
    policy, allocation, terminal = _open_unreported_graphiti_subscription_leaf(
        service
    )
    plan = _issue_790_plan(policy, allocation, terminal)
    _bind_issue_790_fixture_contract(
        monkeypatch,
        allocation=allocation,
        terminal=terminal,
        plan_digest=str(plan["canonical_digest"]),
    )
    approved_at = T0 + timedelta(seconds=5)
    authority, authority_digest = _conservative_disposition_authority(
        allocation,
        terminal,
        approved_at=approved_at,
        approved_plan_digest=str(plan["canonical_digest"]),
    )
    disposition = service.disposition_unreported_subscription_usage(
        invocation_id=allocation.invocation_id,
        expected_terminal_digest=terminal.terminal_digest,
        expected_allocation_digest=allocation.canonical_digest,
        approved_by=str(authority["approved_by"]),
        approval_reference=str(authority["approval_reference"]),
        approved_at=approved_at,
        approved_plan_digest=str(plan["canonical_digest"]),
        authority_digest=authority_digest,
        observed_at=T0 + timedelta(seconds=10),
    )
    route = service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)
    service.release_route_circuit(
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        release_kind="AUTHORISED_OPERATOR_RESET",
        bound_failure_reason=str(route["reason"]),
        evidence_digest=str(disposition["disposition_digest"]),
        recorded_at=T0 + timedelta(seconds=11),
    )
    store = tmp_path / "unpublished.sqlite3"
    _seed_issue_790_retry_events(store)
    repository = Issue790CanaryRepository(str(store))
    repository.retain_retry_exclusions(
        approved_plan_digest=str(plan["canonical_digest"]),
        disposition_digest=str(disposition["disposition_digest"]),
        events=RETRY_FORBIDDEN_EVENTS,
        excluded_at=T0 + timedelta(seconds=12),
    )
    return store, plan, disposition, repository


def _consume_plan_event(
    repository: Issue790CanaryRepository,
    store: Path,
    *,
    plan_digest: str,
    disposition_digest: str,
    event_id: str,
    ledger_seq: int,
    owner_id: str,
    consumed_at: datetime,
) -> dict[str, object]:
    return repository.consume(
        approved_plan_digest=plan_digest,
        disposition_digest=disposition_digest,
        event_id=event_id,
        ledger_seq=ledger_seq,
        owner_id=owner_id,
        preflight_evidence=_issue_790_canary_preflight(
            store,
            event_id=event_id,
            ledger_seq=ledger_seq,
            evaluated_at=consumed_at - timedelta(seconds=1),
        ),
        consumed_at=consumed_at,
    )


def _issue_790_canary_preflight(
    path: Path,
    *,
    event_id: str,
    ledger_seq: int,
    evaluated_at: datetime,
    resolved_units: list[dict[str, object]] | None = None,
    approved_plan_digest: str | None = None,
    fixed_constraints_digest: str | None = None,
) -> dict[str, object]:
    connection = sqlite3.connect(path)
    row = connection.execute(
        "SELECT manifest_json,manifest_digest FROM "
        "unpublished_graphiti_revision_events WHERE event_id=? AND ledger_seq=?",
        (event_id, ledger_seq),
    ).fetchone()
    connection.close()
    assert row is not None
    manifest = json.loads(str(row[0]))
    if resolved_units is None:
        resolved_units = manifest["unit_refs"]
    without_digest: dict[str, object] = {
        "schema_version": (
            "newsroom.issue-790.iterative-fresh-event-preflight.v2"
            if approved_plan_digest is not None
            else "newsroom.graphiti-fresh-event-preflight.v1"
        ),
        "event_id": event_id,
        "ledger_seq": ledger_seq,
        "event_state": "QUEUED",
        "event_attempt_count": 0,
        "event_manifest_digest": str(row[1]),
        "resolved_units": resolved_units,
        "rights_decision_digests": [
            _digest({"rights": item["ingest_id"]}) for item in resolved_units
        ],
        "owner_emergency_stop_clear": True,
        "provider_calls": 0,
        "store_mutations": 0,
        "evaluated_at": evaluated_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }
    if approved_plan_digest is not None:
        assert fixed_constraints_digest is not None
        without_digest.update(
            {
                "approved_plan_digest": approved_plan_digest,
                "fallback_mode": "DISABLED_BEFORE_PROVIDER_DISPATCH",
                "fixed_constraints_digest": fixed_constraints_digest,
            }
        )
    return {**without_digest, "evidence_digest": _digest(without_digest)}


def _issue_790_operational_evidence(
    *,
    store: Path,
    observed_at: datetime,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": "newsroom.issue-790.operational-preconditions.v1",
        "repository_root": "/fixture/repository",
        "branch": "main",
        "revision": "a" * 40,
        "tree": "b" * 40,
        "local_main_revision": "a" * 40,
        "origin_main_revision": "a" * 40,
        "github_main_revision": "a" * 40,
        "worktree_clean": True,
        "running_code": [
            {
                "module": module_name,
                "repository_path": relative_path,
                "git_blob": "c" * 40,
                "sha256": "sha256:" + "d" * 64,
            }
            for module_name, relative_path in issue_790_operation._RUNNING_CODE_MODULES
        ],
        "ci_test": {
            "name": "test",
            "status": "completed",
            "conclusion": "success",
            "head_sha": "a" * 40,
            "url": "https://github.com/fol2/newsroom/actions/runs/1",
        },
        "worker": {
            "label": "com.jamesto.newsroom-graphiti-worker",
            "launchctl_loaded": False,
            "process_ids": [],
        },
        "retry_forbidden_events": RETRY_FORBIDDEN_EVENTS,
        "store": str(store.absolute()),
        "store_quick_check": "ok",
        "observed_at": observed_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }
    return {**record, "evidence_digest": _digest(record)}


def _patch_issue_790_live_evidence(
    monkeypatch: pytest.MonkeyPatch,
    *,
    store: Path,
    observed_at: datetime,
) -> None:
    evidence = _issue_790_operational_evidence(
        store=store,
        observed_at=observed_at,
    )
    monkeypatch.setattr(
        issue_790_operation,
        "collect_issue_790_operational_evidence",
        lambda **_values: evidence,
    )
    monkeypatch.setattr(
        issue_790_operation,
        "_worker_state",
        lambda: {
            "label": "com.jamesto.newsroom-graphiti-worker",
            "launchctl_loaded": False,
            "process_ids": [],
        },
    )
    runtime = {
        "schema_version": "newsroom.issue-790.graphiti-runtime-readiness.v1",
        "framework": "graphiti-core",
        "framework_version": "0.29.3",
        "provider_calls": 0,
        "credential_resolution": False,
    }
    monkeypatch.setattr(
        issue_790_operation,
        "_qualify_real_graphiti_runtime",
        lambda: {**runtime, "runtime_digest": _digest(runtime)},
    )
    monkeypatch.setattr(
        issue_790_operation,
        "_other_issue_790_legacy_canary_process_ids",
        lambda **_kwargs: (),
    )


def _persisted_issue_790_prepared_canary(
    *,
    store: Path,
    proving_store: Path,
    plan: Mapping[str, object],
    observed_at: datetime,
    event_id: str,
    ledger_seq: int,
):
    prepared = prepare_issue_790_canary(
        store=store,
        proving_store=proving_store,
        plan=plan,
        observed_at=observed_at,
        exact_head="a" * 40,
        event_id=event_id,
        ledger_seq=ledger_seq,
        role="canary",
    )
    return prepared_canary_from_record(prepared_canary_record(prepared))


def test_hold_or_reject_admission_creates_no_envelope_or_leaf(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.retain_zero_call_admission(
        decision_id="decision-hold",
        decision="HOLD",
        cycle_id="00000000-0000-4000-8000-000000000001",
        recorded_at=T0,
    )
    service.retain_zero_call_admission(
        decision_id="decision-reject",
        decision="REJECT",
        cycle_id="00000000-0000-4000-8000-000000000001",
        recorded_at=T0,
    )

    report = service.report(start=T0, end=T0 + timedelta(minutes=1))

    assert report["envelope_count"] == 0
    assert report["leaf_dispatch_count"] == 0
    assert report["zero_call_admission_counts"] == {"HOLD": 1, "REJECT": 1}


def test_primary_and_fallback_are_distinct_leaves_joined_to_one_outcome(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    envelope, _primary_policy, primary = _open_and_allocate(service)
    fallback_policy = _policy(
        workload=WorkloadClass.CONT_WRITER_FALLBACK,
        provider="cursor-agent-cli",
        route="CONT_FALLBACK",
        model="cursor-pinned",
    )
    service.register_policy(fallback_policy)
    fallback = _allocation(
        envelope,
        fallback_policy,
        leaf_ordinal=2,
        workload=WorkloadClass.CONT_WRITER_FALLBACK,
        request="request-fallback",
        parent_invocation_id=primary.invocation_id,
    )
    service.allocate(fallback, owner_emergency_stop=False)
    service.link_provider_attempt(
        invocation_id=primary.invocation_id,
        provider_attempt_id="provider-primary",
        linked_at=T0 + timedelta(seconds=2),
    )
    service.link_provider_attempt(
        invocation_id=fallback.invocation_id,
        provider_attempt_id="provider-fallback",
        linked_at=T0 + timedelta(seconds=2),
    )
    service.complete(_reported(primary, total=125, outcome="REJECTED_OUTPUT"))
    service.complete(_reported(fallback, total=125, outcome="ACCEPTED_OUTPUT"))
    service.record_work_outcome(
        envelope_id=envelope.envelope_id,
        outcome="ACCEPTED",
        outcome_record_id="draft-outcome-1",
        payload_digest=_digest({"payload": 1}),
        terminal_at=T0 + timedelta(seconds=4),
        accepted_provider_attempt_id="provider-fallback",
    )

    rows = service.query(start=T0, end=T0 + timedelta(minutes=1))["leaves"]

    assert [row["workload_class"] for row in rows] == [
        "CONT_WRITER_PRIMARY",
        "CONT_WRITER_FALLBACK",
    ]
    assert rows[1]["parent_invocation_id"] == primary.invocation_id
    assert {row["work_outcome"] for row in rows} == {"ACCEPTED"}
    assert {row["payload_digest"] for row in rows} == {_digest({"payload": 1})}
    report = service.report(start=T0, end=T0 + timedelta(minutes=1))
    assert report["productive_tokens"] == 125
    assert report["no_result_tokens"] == 125
    assert report["no_result_reasons"] == {"REJECTED_OUTPUT": 125}


def test_rejected_provider_output_is_no_result_usage_not_productivity(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    envelope, _policy_value, allocation = _open_and_allocate(service)
    service.complete(_reported(allocation, outcome="PLANNING_RESIDUE"))
    service.record_work_outcome(
        envelope_id=envelope.envelope_id,
        outcome="REJECT",
        outcome_record_id="draft-outcome-reject",
        payload_digest=None,
        terminal_at=T0 + timedelta(seconds=3),
    )

    report = service.report(start=T0, end=T0 + timedelta(minutes=1))

    assert report["accepted_payload_count"] == 0
    assert report["productive_tokens"] == 0
    assert report["no_result_tokens"] == 125
    assert report["tokens_per_accepted_payload"] is None
    assert report["no_result_reasons"] == {"PLANNING_RESIDUE": 125}


def test_no_result_cycle_joins_durable_backoff_state(tmp_path: Path) -> None:
    service = _service(tmp_path)
    envelope, _policy_value, allocation = _open_and_allocate(service)
    service.complete(_reported(allocation, outcome="REJECTED_OUTPUT"))
    service.record_work_outcome(
        envelope_id=envelope.envelope_id,
        outcome="REJECT",
        outcome_record_id="draft-outcome-reject",
        payload_digest=None,
        terminal_at=T0 + timedelta(seconds=3),
        cycle_outcome="UNPRODUCTIVE_PROVIDER",
        route_circuit_state="CLOSED",
        route_circuit_reason="FIRST_NO_RESULT_BACKOFF",
    )

    row = service.query(start=T0, end=T0 + timedelta(minutes=1))["envelopes"][0]

    assert row["cycle_outcome"] == "UNPRODUCTIVE_PROVIDER"
    assert row["route_circuit_reason"] == "FIRST_NO_RESULT_BACKOFF"


def test_reported_components_validate_without_context_double_counting(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    _envelope_value, _policy_value, allocation = _open_and_allocate(service)
    service.complete(_reported(allocation, total=125))

    report = service.report(start=T0, end=T0 + timedelta(minutes=1))

    assert report["observed_total_tokens"] == 125
    assert report["context_tokens"] == 100
    invalid_envelope = _envelope(
        cycle_id="00000000-0000-4000-8000-000000000099",
        candidate_id="candidate-invalid",
    )
    service.open_envelope(invalid_envelope)
    invalid_allocation = _allocation(invalid_envelope, _policy_value, request="invalid")
    service.allocate(invalid_allocation, owner_emergency_stop=False)
    service.complete(
        InvocationTerminal.create(
            invocation_id=invalid_allocation.invocation_id,
            outcome="ACCEPTED",
            failure_class=None,
            usage_status=UsageStatus.REPORTED,
            components=UsageComponents(
                input_tokens=100,
                output_tokens=20,
                cached_read_tokens=5,
                cached_write_tokens=0,
                reasoning_tokens=0,
                context_tokens=100,
                total_tokens=224,
                provenance="PROVIDER_REPORTED",
            ),
            dispatch_at=T0 + timedelta(seconds=2),
            completed_at=T0 + timedelta(seconds=3),
            observed_at=T0 + timedelta(seconds=3),
            provider_telemetry_digest=_digest({"invalid": 1}),
            raw_telemetry_pointer="private://invalid",
            subscription_cli_chat_not_cash_debited=True,
        )
    )
    invalid_row = service.query(start=T0, end=T0 + timedelta(minutes=1))["leaves"][1]
    assert invalid_row["usage_status"] == "INVALID"
    assert invalid_row["failure_class"] == "REPORTED_COMPONENT_TOTAL_INVALID"
    assert (
        service.report(start=T0, end=T0 + timedelta(minutes=1))["observed_total_tokens"]
        == 125
    )


def test_grok_headless_cache_inclusive_total_is_reported(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    _envelope_value, _policy_value, allocation = _open_and_allocate(service)
    service.complete(
        InvocationTerminal.create(
            invocation_id=allocation.invocation_id,
            outcome="ACCEPTED_OUTPUT",
            failure_class=None,
            usage_status=UsageStatus.REPORTED,
            components=UsageComponents(
                input_tokens=80,
                output_tokens=10,
                cached_read_tokens=5,
                cached_write_tokens=0,
                reasoning_tokens=2,
                context_tokens=85,
                total_tokens=95,
                provenance="PROVIDER_REPORTED",
            ),
            dispatch_at=T0 + timedelta(seconds=2),
            completed_at=T0 + timedelta(seconds=3),
            observed_at=T0 + timedelta(seconds=3),
            provider_telemetry_digest=_digest({"grok": "1.0.10"}),
            raw_telemetry_pointer="private://grok-cache-total",
            subscription_cli_chat_not_cash_debited=True,
        )
    )
    row = service.query(start=T0, end=T0 + timedelta(minutes=1))["leaves"][0]
    assert row["usage_status"] == "REPORTED"
    assert row["failure_class"] is None
    assert row["total_tokens"] == 95
    assert row["context_tokens"] == 85


def test_estimate_is_explicit_and_unbounded_missing_usage_opens_only_route(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    _envelope_value, policy, allocation = _open_and_allocate(service)
    service.complete(
        InvocationTerminal.create(
            invocation_id=allocation.invocation_id,
            outcome="REJECTED_OUTPUT",
            failure_class=None,
            usage_status=UsageStatus.ESTIMATED,
            components=UsageComponents(
                total_tokens=2_500, provenance="BOUNDED_ESTIMATE"
            ),
            dispatch_at=T0 + timedelta(seconds=1),
            completed_at=T0 + timedelta(seconds=2),
            observed_at=T0 + timedelta(seconds=2),
            estimate_policy_digest=policy.canonical_digest,
            estimate_calculation="hard_estimate_ceiling_tokens=2500",
            subscription_cli_chat_not_cash_debited=True,
        )
    )
    second_envelope = _envelope(
        cycle_id="00000000-0000-4000-8000-000000000002",
        candidate_id="candidate-2",
    )
    service.open_envelope(second_envelope)
    second = _allocation(second_envelope, policy, request="request-2")
    service.allocate(second, owner_emergency_stop=False)
    service.complete(
        InvocationTerminal.create(
            invocation_id=second.invocation_id,
            outcome="TRANSPORT_LOST",
            failure_class="MISSING_TELEMETRY",
            usage_status=UsageStatus.UNREPORTED,
            components=UsageComponents(provenance="UNAVAILABLE"),
            dispatch_at=T0 + timedelta(seconds=5),
            completed_at=T0 + timedelta(seconds=6),
            observed_at=T0 + timedelta(seconds=6),
            subscription_cli_chat_not_cash_debited=True,
        )
    )

    report = service.report(start=T0, end=T0 + timedelta(minutes=1))

    assert report["estimated_tokens"] == 2_500
    assert report["unresolved_invocation_count"] == 1
    assert service.route_state("CONT_PRIMARY")["state"] == "OPEN"
    assert service.route_state("GRAPHITI_EMBEDDING")["state"] == "CLOSED"
    with pytest.raises(ModelUsageAdmissionError, match="unresolved usage"):
        service.allocate(
            _allocation(
                second_envelope,
                policy,
                leaf_ordinal=2,
                request="request-3",
            ),
            owner_emergency_stop=False,
        )


def test_proved_pre_dispatch_failure_is_explicit_zero(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _envelope_value, _policy_value, allocation = _open_and_allocate(service)
    service.complete(
        InvocationTerminal.create(
            invocation_id=allocation.invocation_id,
            outcome="EXECUTABLE_NOT_FOUND",
            failure_class="LOCAL_EXECUTABLE",
            usage_status=UsageStatus.REPORTED,
            components=UsageComponents(total_tokens=0, provenance="CLI_DERIVED"),
            dispatch_at=None,
            completed_at=T0 + timedelta(seconds=2),
            observed_at=T0 + timedelta(seconds=2),
            pre_dispatch_zero_proved=True,
            subscription_cli_chat_not_cash_debited=True,
        )
    )

    row = service.query(start=T0, end=T0 + timedelta(minutes=1))["leaves"][0]

    assert row["total_tokens"] == 0
    assert row["pre_dispatch_zero_proved"] is True


def test_telemetry_digest_mismatch_is_retained_invalid_and_opens_route(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    _envelope_value, _policy_value, allocation = _open_and_allocate(service)
    terminal = _reported(allocation)
    service.complete(
        terminal,
        provider_telemetry={"different": "content"},
    )

    row = service.query(start=T0, end=T0 + timedelta(minutes=1))["leaves"][0]

    assert row["usage_status"] == "INVALID"
    assert row["failure_class"] == "TELEMETRY_DIGEST_MISMATCH"
    assert service.route_state("CONT_PRIMARY")["state"] == "OPEN"
    connection = sqlite3.connect(tmp_path / "unpublished.sqlite3")
    telemetry = json.loads(
        connection.execute(
            "SELECT record_json FROM model_provider_telemetry"
        ).fetchone()[0]
    )
    connection.close()
    assert telemetry["provider_telemetry"] == {"different": "content"}
    assert (
        service.report(start=T0, end=T0 + timedelta(minutes=1))["observed_total_tokens"]
        == 0
    )


def test_duplicate_request_digest_is_stopped_before_dispatch(tmp_path: Path) -> None:
    service = _service(tmp_path)
    envelope, policy, first = _open_and_allocate(service)

    with pytest.raises(ModelUsageAdmissionError, match="duplicate request"):
        service.allocate(
            _allocation(
                envelope,
                policy,
                leaf_ordinal=2,
                request="request-1",
            ),
            owner_emergency_stop=False,
        )

    assert (
        service.report(start=T0, end=T0 + timedelta(minutes=1))["allocation_count"]
        == 1
    )
    assert first.invocation_id


def test_policy_preflight_and_post_dispatch_breach_are_route_local(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    envelope = _envelope()
    policy = _policy()
    service.register_policy(policy)
    service.open_envelope(envelope)
    with pytest.raises(ModelUsageAdmissionError, match="prompt bytes"):
        service.allocate(
            _allocation(envelope, policy, prompt_bytes=4_097),
            owner_emergency_stop=False,
        )
    with pytest.raises(ModelUsageAdmissionError, match="config identity"):
        service.allocate(
            _allocation(envelope, policy, config_identity="unqualified-config-v2"),
            owner_emergency_stop=False,
        )

    allocation = _allocation(envelope, policy, prompt_bytes=4_096)
    service.allocate(allocation, owner_emergency_stop=False)
    service.complete(_reported(allocation, total=2_501))

    row = service.query(start=T0, end=T0 + timedelta(minutes=1))["leaves"][0]
    assert row["policy_breach"] == "MAX_TOTAL_TOKENS_EXCEEDED"
    assert service.route_state("CONT_PRIMARY")["state"] == "OPEN"


def test_graphiti_chat_and_embedding_are_distinct_and_terminal_ingests_are_valid(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    envelope = _envelope(
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        candidate_id=None,
        ingest_id="ingest-1",
    )
    service.open_envelope(envelope)
    chat_policy = _policy(
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        provider="cursor-agent-cli",
        route="GRAPHITI_CHAT",
        model="cursor-pinned",
    )
    embedding_policy = _policy(
        workload=WorkloadClass.GRAPHITI_EMBEDDING,
        provider="openrouter",
        route="GRAPHITI_EMBEDDING",
        model="openai/text-embedding-3-small",
    )
    service.register_policy(chat_policy)
    service.register_policy(embedding_policy)
    chat = _allocation(envelope, chat_policy)
    embedding = _allocation(
        envelope,
        embedding_policy,
        leaf_ordinal=2,
        request="embedding-request",
        workload=WorkloadClass.GRAPHITI_EMBEDDING,
        parent_invocation_id=chat.invocation_id,
    )
    service.allocate(chat, owner_emergency_stop=False)
    service.allocate(embedding, owner_emergency_stop=False)
    service.complete(_reported(chat, total=125, outcome="COMPLETE"))
    service.complete(
        InvocationTerminal.create(
            invocation_id=embedding.invocation_id,
            outcome="COMPLETE",
            failure_class=None,
            usage_status=UsageStatus.REPORTED,
            components=UsageComponents(
                input_tokens=40,
                total_tokens=40,
                provenance="PROVIDER_REPORTED",
            ),
            dispatch_at=T0 + timedelta(seconds=2),
            completed_at=T0 + timedelta(seconds=3),
            observed_at=T0 + timedelta(seconds=3),
            provider_telemetry_digest=_digest({"embedding": 1}),
            raw_telemetry_pointer="private://embedding/1",
            od_011_reference="OD-011:EVALUATION",
            subscription_cli_chat_not_cash_debited=False,
        )
    )
    service.record_work_outcome(
        envelope_id=envelope.envelope_id,
        outcome="GRAPHITI_SUCCESS_ZERO_PROPOSALS",
        outcome_record_id="graphiti-attempt-1",
        payload_digest=None,
        terminal_at=T0 + timedelta(seconds=4),
        retained_proposal_count=0,
    )

    report = service.report(start=T0, end=T0 + timedelta(minutes=1))

    assert report["graphiti_valid_ingest_count"] == 1
    assert report["graphiti_tokens_per_valid_ingest"] == {
        "numerator": 165,
        "denominator": 1,
    }
    assert report["graphiti_tokens_per_retained_proposal"] is None
    assert report["workload_totals"]["GRAPHITI_CHAT_PRIMARY"] == 125
    assert report["workload_totals"]["GRAPHITI_EMBEDDING"] == 40
    assert report["graphiti_result_telemetry"]["completed_useful_ingest_count"] == 1
    assert (
        report["graphiti_result_telemetry"]["completed_ingests_zero_proposals"] == 1
    )


def test_graphiti_partial_is_excluded_from_completed_useful_ingests(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    envelope = _envelope(
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        candidate_id=None,
        ingest_id="ingest-1",
    )
    service.open_envelope(envelope)
    chat_policy = _policy(
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        provider="cursor-agent-cli",
        route="GRAPHITI_CHAT",
        model="cursor-pinned",
    )
    embedding_policy = _policy(
        workload=WorkloadClass.GRAPHITI_EMBEDDING,
        provider="openrouter",
        route="GRAPHITI_EMBEDDING",
        model="openai/text-embedding-3-small",
    )
    service.register_policy(chat_policy)
    service.register_policy(embedding_policy)
    chat = _allocation(envelope, chat_policy)
    embedding = _allocation(
        envelope,
        embedding_policy,
        leaf_ordinal=2,
        request="embedding-request",
        workload=WorkloadClass.GRAPHITI_EMBEDDING,
        parent_invocation_id=chat.invocation_id,
    )
    service.allocate(chat, owner_emergency_stop=False)
    service.allocate(embedding, owner_emergency_stop=False)
    service.complete(_reported(chat, total=125, outcome="COMPLETE"))
    service.complete(
        InvocationTerminal.create(
            invocation_id=embedding.invocation_id,
            outcome="COMPLETE",
            failure_class=None,
            usage_status=UsageStatus.REPORTED,
            components=UsageComponents(
                input_tokens=40,
                total_tokens=40,
                provenance="PROVIDER_REPORTED",
            ),
            dispatch_at=T0 + timedelta(seconds=2),
            completed_at=T0 + timedelta(seconds=3),
            observed_at=T0 + timedelta(seconds=3),
            provider_telemetry_digest=_digest({"embedding": 1}),
            raw_telemetry_pointer="private://embedding/1",
            od_011_reference="OD-011:EVALUATION",
            subscription_cli_chat_not_cash_debited=False,
        )
    )
    service.record_work_outcome(
        envelope_id=envelope.envelope_id,
        outcome="GRAPHITI_PARTIAL",
        outcome_record_id="graphiti-attempt-partial",
        payload_digest=None,
        terminal_at=T0 + timedelta(seconds=4),
        retained_proposal_count=2,
    )

    report = service.report(start=T0, end=T0 + timedelta(minutes=1))
    telemetry = report["graphiti_result_telemetry"]

    assert report["graphiti_valid_ingest_count"] == 0
    assert report["graphiti_tokens_per_valid_ingest"] is None
    assert telemetry["completed_useful_ingest_count"] == 0
    assert telemetry["completed_ingests_with_proposals"] == 0
    assert telemetry["completed_ingests_zero_proposals"] == 0
    assert telemetry["failed_or_rolled_back_attempt_tokens"] == 165


def test_parent_and_child_totals_are_not_double_counted(tmp_path: Path) -> None:
    service = _service(tmp_path)
    envelope, _primary_policy, primary = _open_and_allocate(service)
    fallback_policy = _policy(
        workload=WorkloadClass.CONT_WRITER_FALLBACK,
        provider="cursor-agent-cli",
        route="CONT_FALLBACK",
        model="cursor-pinned",
    )
    service.register_policy(fallback_policy)
    fallback = _allocation(
        envelope,
        fallback_policy,
        leaf_ordinal=2,
        request="fallback",
        parent_invocation_id=primary.invocation_id,
    )
    service.allocate(fallback, owner_emergency_stop=False)
    service.complete(_reported(primary, total=125))
    service.complete(_reported(fallback, total=125))

    report = service.report(start=T0, end=T0 + timedelta(minutes=1))

    assert report["observed_total_tokens"] == 250
    assert report["envelope_allocated_tokens"] == 250


def test_fixed_buckets_include_zeros_rolling_views_and_daily_totals(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    _envelope_value, policy, first = _open_and_allocate(service)
    service.complete(
        _reported(first, total=125, completed_at=T0 + timedelta(seconds=10))
    )
    second_envelope = _envelope(
        cycle_id="00000000-0000-4000-8000-000000000002",
        candidate_id="candidate-2",
    )
    service.open_envelope(second_envelope)
    second = _allocation(second_envelope, policy, request="second")
    service.allocate(second, owner_emergency_stop=False)
    service.complete(
        _reported(
            second, total=125, completed_at=T0 + timedelta(minutes=10, seconds=10)
        )
    )

    report = service.report(
        start=T0,
        end=T0 + timedelta(minutes=15),
        bucket_seconds=300,
    )

    assert [bucket["observed_total_tokens"] for bucket in report["fixed_buckets"]] == [
        125,
        0,
        125,
    ]
    assert report["utc_day_totals"] == {"2026-08-24": 250}
    assert report["rolling_300_at_dispatch"] == [
        {"invocation_id": first.invocation_id, "observed_total_tokens": 0},
        {"invocation_id": second.invocation_id, "observed_total_tokens": 0},
    ]


def test_fixed_buckets_are_utc_aligned_for_unaligned_proof_window(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    _envelope_value, _policy_value, allocation = _open_and_allocate(service)
    service.complete(_reported(allocation, completed_at=T0 + timedelta(seconds=2)))

    report = service.report(
        start=T0 + timedelta(milliseconds=500),
        end=T0 + timedelta(minutes=5, milliseconds=500),
        bucket_seconds=300,
    )

    assert [bucket["window_start"] for bucket in report["fixed_buckets"]] == [
        "2026-08-24T10:00:00.000000Z",
        "2026-08-24T10:05:00.000000Z",
    ]
    assert [bucket["observed_total_tokens"] for bucket in report["fixed_buckets"]] == [
        125,
        0,
    ]


def test_more_than_daily_500k_is_alerted_but_not_an_admission_gate(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    baseline = _policy()
    policy = InvocationEfficiencyPolicy.create(
        **{
            field: getattr(baseline, field)
            for field in baseline.__dataclass_fields__
            if field
            not in {
                "canonical_digest",
                "max_total_tokens",
                "max_output_tokens",
                "hard_estimate_ceiling_tokens",
            }
        },
        max_total_tokens=300_000,
        max_output_tokens=20_000,
        hard_estimate_ceiling_tokens=300_000,
    )
    service.register_policy(policy)
    for index in range(2):
        envelope = _envelope(
            cycle_id=f"00000000-0000-4000-8000-{index + 10:012d}",
            candidate_id=f"candidate-{index + 10}",
        )
        service.open_envelope(envelope)
        allocation = _allocation(envelope, policy, request=f"request-{index + 10}")
        service.allocate(allocation, owner_emergency_stop=False)
        service.complete(
            InvocationTerminal.create(
                invocation_id=allocation.invocation_id,
                outcome="ACCEPTED",
                failure_class=None,
                usage_status=UsageStatus.REPORTED,
                components=UsageComponents(
                    input_tokens=260_000,
                    output_tokens=10_000,
                    total_tokens=270_000,
                    provenance="PROVIDER_REPORTED",
                ),
                dispatch_at=T0 + timedelta(minutes=index),
                completed_at=T0 + timedelta(minutes=index, seconds=10),
                observed_at=T0 + timedelta(minutes=index, seconds=10),
                provider_telemetry_digest=_digest({"large": index}),
                raw_telemetry_pointer=f"private://large/{index}",
                subscription_cli_chat_not_cash_debited=True,
            )
        )

    report = service.report(start=T0, end=T0 + timedelta(days=1))

    assert report["observed_total_tokens"] == 540_000
    assert report["daily_500k_alert"] is True
    assert report["normal_daily_hard_cut"] is None
    assert report["leaf_dispatch_count"] == 2


def test_owner_emergency_stop_overrides_otherwise_valid_allocation(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    envelope = _envelope()
    policy = _policy()
    service.register_policy(policy)
    service.open_envelope(envelope)

    with pytest.raises(ModelUsageAdmissionError, match="emergency stop"):
        service.allocate(_allocation(envelope, policy), owner_emergency_stop=True)


def test_restart_recovery_retains_unresolved_leaf_without_duplication(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unpublished.sqlite3"
    first_service = ModelUsageService(str(path))
    _envelope_value, _policy_value, _allocation_value = _open_and_allocate(
        first_service
    )
    restarted = ModelUsageService(str(path))
    restarted.recover_unresolved(observed_at=T0 + timedelta(minutes=1))
    restarted.recover_unresolved(observed_at=T0 + timedelta(minutes=2))
    report = restarted.report(start=T0, end=T0 + timedelta(minutes=3))

    assert report["allocation_count"] == 1
    assert report["actual_provider_dispatch_count"] == 0
    assert report["unresolved_invocation_count"] == 1
    row = restarted.query(start=T0, end=T0 + timedelta(minutes=3))["leaves"][0]
    assert row["usage_status"] == "AMBIGUOUS"
    assert row["total_tokens"] is None
    assert service_row_count(path, "model_invocation_terminals") == 1


def test_restart_recovery_does_not_claim_a_live_invocation(tmp_path: Path) -> None:
    path = tmp_path / "unpublished.sqlite3"
    active = ModelUsageService(str(path))
    _envelope_value, _policy_value, allocation = _open_and_allocate(active)
    active.observe_transport(
        invocation_id=allocation.invocation_id,
        observed_at=T0 + timedelta(seconds=2),
        state="DISPATCH_STARTED",
        evidence_digest=_digest({"live": True}),
    )

    concurrent = ModelUsageService(str(path))
    assert concurrent.recover_unresolved(
        observed_at=T0 + timedelta(seconds=3)
    ) == 0
    active.complete(_reported(allocation, completed_at=T0 + timedelta(seconds=4)))
    assert active.query(start=T0, end=T0 + timedelta(minutes=1))["leaves"][0][
        "usage_status"
    ] == "REPORTED"


def test_concurrent_recovery_is_idempotent_at_the_retained_deadline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unpublished.sqlite3"
    service = ModelUsageService(str(path))
    _envelope_value, _policy_value, allocation = _open_and_allocate(service)
    service.observe_transport(
        invocation_id=allocation.invocation_id,
        observed_at=T0 + timedelta(seconds=2),
        state="DISPATCH_STARTED",
        evidence_digest=_digest({"concurrent": True}),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(
                lambda seconds: ModelUsageService(str(path)).recover_unresolved(
                    observed_at=T0 + timedelta(seconds=seconds)
                ),
                (40, 45),
            )
        )

    assert sum(results) >= 2
    assert service_row_count(path, "model_invocation_terminals") == 1
    assert service_row_count(path, "model_work_outcomes") == 1
    leaf = service.query(start=T0, end=T0 + timedelta(minutes=1))["leaves"][0]
    assert leaf["completed_at"] == "2026-08-24T10:00:31.000000Z"
    assert leaf["work_outcome_terminal_at"] == "2026-08-24T10:00:31.000000Z"


def test_later_provider_telemetry_appends_reconciliation_without_editing_history(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    _envelope_value, _policy_value, allocation = _open_and_allocate(service)
    service.complete(
        InvocationTerminal.create(
            invocation_id=allocation.invocation_id,
            outcome="TRANSPORT_LOST",
            failure_class="MISSING_TELEMETRY",
            usage_status=UsageStatus.UNREPORTED,
            components=UsageComponents(provenance="UNAVAILABLE"),
            dispatch_at=T0 + timedelta(seconds=1),
            completed_at=T0 + timedelta(seconds=2),
            observed_at=T0 + timedelta(seconds=2),
            subscription_cli_chat_not_cash_debited=True,
        )
    )
    service.reconcile(
        invocation_id=allocation.invocation_id,
        components=UsageComponents(
            input_tokens=100,
            output_tokens=20,
            cached_read_tokens=5,
            total_tokens=125,
            provenance="PROVIDER_REPORTED",
        ),
        provider_telemetry={"late": "valid"},
        observed_at=T0 + timedelta(minutes=5),
        raw_telemetry_pointer="private://late/1",
    )

    historical = service.query(start=T0, end=T0 + timedelta(minutes=1))["leaves"][0]
    row = service.query(start=T0, end=T0 + timedelta(minutes=6))["leaves"][0]

    assert historical["usage_status"] == "UNREPORTED"
    assert historical["reconciliation_usage_status"] is None
    assert row["usage_status"] == "REPORTED"
    assert row["total_tokens"] == 125
    assert row["terminal_usage_status"] == "UNREPORTED"
    assert row["reconciliation_usage_status"] == "REPORTED"
    assert row["reconciled_at"] == "2026-08-24T10:05:00.000000Z"
    assert service.route_state("CONT_PRIMARY")["state"] == "CLOSED"
    assert (
        service_row_count(
            tmp_path / "unpublished.sqlite3", "model_invocation_terminals"
        )
        == 1
    )
    assert (
        service_row_count(
            tmp_path / "unpublished.sqlite3", "model_usage_reconciliations"
        )
        == 1
    )
    assert (
        service_row_count(tmp_path / "unpublished.sqlite3", "model_provider_telemetry")
        == 1
    )


def test_graphiti_retry_evidence_classifies_zero_settled_and_active_attempts(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    policy = _policy(
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        provider="cursor-agent-cli",
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        model="composer-2.5",
        hard_estimate_ceiling_tokens=None,
    )
    service.register_policy(policy)

    first = _envelope(
        cycle_id="retry-evidence-1",
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        candidate_id=None,
        ingest_id="retry-evidence-ingest",
    )
    service.open_envelope(first)
    first_allocation = _allocation(first, policy, request="retry-evidence-1")
    service.allocate(first_allocation, owner_emergency_stop=False)
    first_terminal = service.complete(
        InvocationTerminal.create(
            invocation_id=first_allocation.invocation_id,
            outcome="DISPATCH_FENCE_REFUSED",
            failure_class="DISPATCH_FENCE_REFUSED",
            usage_status=UsageStatus.REPORTED,
            components=UsageComponents(total_tokens=0, provenance="CLI_DERIVED"),
            dispatch_at=None,
            completed_at=T0 + timedelta(seconds=2),
            observed_at=T0 + timedelta(seconds=2),
            pre_dispatch_zero_proved=True,
            subscription_cli_chat_not_cash_debited=True,
        )
    )
    service.record_work_outcome(
        envelope_id=first.envelope_id,
        outcome="GRAPHITI_FAILED",
        outcome_record_id="retry-evidence-outcome-1",
        payload_digest=None,
        terminal_at=first_terminal.completed_at,
    )

    second = WorkEnvelope.create(
        cycle_id="retry-evidence-2",
        workload_class=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        admitted_at=T0 + timedelta(seconds=3),
        admission_decision_id=None,
        candidate_id=None,
        hypothesis_digest=None,
        evidence_package_digest=None,
        ingest_id="retry-evidence-ingest",
        graphiti_attempt_id="retry-evidence-ingest:2",
    )
    service.open_envelope(second)
    second_allocation = _allocation(second, policy, request="retry-evidence-2")
    service.allocate(second_allocation, owner_emergency_stop=False)
    service.observe_transport(
        invocation_id=second_allocation.invocation_id,
        observed_at=second_allocation.allocated_at + timedelta(milliseconds=1),
        state="DISPATCH_STARTED",
        evidence_digest=_digest({"dispatch": second_allocation.invocation_id}),
    )
    second_terminal = _reported(second_allocation)
    service.complete(
        second_terminal,
        provider_telemetry={"invocation": second_allocation.invocation_id},
    )
    service.record_work_outcome(
        envelope_id=second.envelope_id,
        outcome="GRAPHITI_FAILED",
        outcome_record_id="retry-evidence-outcome-2",
        payload_digest=None,
        terminal_at=second_terminal.completed_at,
    )

    third = WorkEnvelope.create(
        cycle_id="retry-evidence-3",
        workload_class=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        admitted_at=T0 + timedelta(seconds=4),
        admission_decision_id=None,
        candidate_id=None,
        hypothesis_digest=None,
        evidence_package_digest=None,
        ingest_id="retry-evidence-ingest",
        graphiti_attempt_id="retry-evidence-ingest:3",
    )
    service.open_envelope(third)

    prior = service.graphiti_ingest_retry_evidence(
        ingest_id="retry-evidence-ingest", before_attempt_number=3
    )
    assert prior.attempt_numbers == (1, 2)
    assert prior.zero_dispatch_attempts == (1,)
    assert prior.settled_provider_attempts == (2,)
    assert prior.latest_settled_provider_attempt == 2
    assert prior.unresolved_attempts == ()
    assert service.graphiti_ingest_retry_evidence(
        ingest_id="retry-evidence-ingest"
    ).unresolved_attempts == (3,)


def test_graphiti_retry_evidence_batch_authenticates_history_once(tmp_path, monkeypatch):
    service = _service(tmp_path)
    policy = _policy(
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY, provider="cursor-agent-cli",
        route=GRAPHITI_CHAT_PRIMARY_ROUTE, model="composer-2.5",
        hard_estimate_ceiling_tokens=None,
    )
    service.register_policy(policy)
    for ingest_id in ("selected-ingest", "second-ingest", "unrelated-ingest"):
        envelope = _envelope(
            cycle_id=ingest_id, workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            candidate_id=None, ingest_id=ingest_id,
        )
        service.open_envelope(envelope)
        allocation = _allocation(envelope, policy, request=ingest_id)
        service.allocate(allocation, owner_emergency_stop=False)
        if ingest_id == "second-ingest":
            terminal = service.complete(InvocationTerminal.create(
                invocation_id=allocation.invocation_id,
                outcome="DISPATCH_FENCE_REFUSED", failure_class="DISPATCH_FENCE_REFUSED",
                usage_status=UsageStatus.REPORTED,
                components=UsageComponents(total_tokens=0, provenance="CLI_DERIVED"),
                dispatch_at=None, completed_at=T0 + timedelta(seconds=2),
                observed_at=T0 + timedelta(seconds=2), pre_dispatch_zero_proved=True,
                subscription_cli_chat_not_cash_debited=True,
            ))
            service.record_work_outcome(
                envelope_id=envelope.envelope_id, outcome="GRAPHITI_FAILED",
                outcome_record_id="second-outcome", payload_digest=None,
                terminal_at=terminal.completed_at,
            )
    decoded_envelopes, decoded_allocations = [], []
    envelope_decode = model_usage_module._envelope_from_record
    allocation_decode = model_usage_module._allocation_from_record

    def decode_envelope(record):
        decoded_envelopes.append(record["ingest_id"])
        return envelope_decode(record)

    def decode_allocation(record):
        decoded_allocations.append(record["cycle_id"])
        return allocation_decode(record)

    monkeypatch.setattr(model_usage_module, "_envelope_from_record", decode_envelope)
    monkeypatch.setattr(model_usage_module, "_allocation_from_record", decode_allocation)
    evidence = service.graphiti_ingest_retry_evidence_many(
        ingest_ids=("selected-ingest", "second-ingest", "absent-ingest"),
    )
    assert evidence["selected-ingest"].attempt_numbers == evidence["selected-ingest"].unresolved_attempts == (1,)
    assert evidence["second-ingest"].attempt_numbers == evidence["second-ingest"].zero_dispatch_attempts == (1,)
    assert evidence["second-ingest"].unresolved_attempts == ()
    assert evidence["absent-ingest"].attempt_numbers == ()
    assert sorted(decoded_envelopes) == sorted(decoded_allocations) == [
        "second-ingest", "selected-ingest", "unrelated-ingest",
    ]


def test_recovered_ambiguous_usage_rejects_rehashed_outcome_rebinding(
    tmp_path, monkeypatch,
):
    service = _service(tmp_path)
    ingest_id = "recovered-ambiguous-ingest"

    def envelope(number):
        return WorkEnvelope.create(
            cycle_id=native_graphiti_usage_cycle_id(
                ingest_id=ingest_id, attempt_number=number
            ),
            workload_class=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            admitted_at=T0 + timedelta(seconds=number),
            admission_decision_id=None,
            candidate_id=None,
            hypothesis_digest=None,
            evidence_package_digest=None,
            ingest_id=ingest_id,
            graphiti_attempt_id=f"{ingest_id}:{number}",
        )

    third, fourth = envelope(3), envelope(4)
    service.open_envelope(third)
    service.open_envelope(fourth)
    receipt_digest = _digest({"private": "receipt-4"})
    service.record_work_outcome(
        envelope_id=fourth.envelope_id,
        outcome="GRAPHITI_REJECTED_BINDING",
        outcome_record_id=receipt_digest,
        payload_digest=None,
        terminal_at=T0 + timedelta(seconds=4),
        retained_proposal_count=0,
    )
    evidence = GraphitiIngestRetryEvidence(
        attempt_numbers=(3, 4),
        zero_dispatch_attempts=(),
        settled_provider_attempts=(3,),
        latest_settled_provider_attempt=3,
        unresolved_attempts=(4,),
    )
    monkeypatch.setattr(
        ModelUsageService,
        "native_graphiti_ingest_retry_evidence_many",
        lambda *_args, **_values: {ingest_id: evidence},
    )
    service.native_recovered_ambiguous_usage_evidence_digest(
        ingest_id=ingest_id,
        authoritative_attempt_number=3,
        skipped_attempt_number=4,
        skipped_receipt_digest=receipt_digest,
        skipped_recorded_at=T0 + timedelta(seconds=5),
    )

    with sqlite3.connect(service.path) as connection:
        record = json.loads(connection.execute(
            "SELECT record_json FROM model_work_outcomes WHERE envelope_id=?",
            (fourth.envelope_id,),
        ).fetchone()[0])
        record.pop("outcome_digest")
        record["outcome_record_id"] = _digest({"private": "forged"})
        forged_digest = digest_canonical(record)
        record["outcome_digest"] = forged_digest
        connection.execute(
            "UPDATE model_work_outcomes SET outcome_digest=?,record_json=? "
            "WHERE envelope_id=?",
            (forged_digest, json.dumps(record), fourth.envelope_id),
        )

    with pytest.raises(
        ModelUsageIntegrityError,
        match="skipped usage record differs",
    ):
        service.native_recovered_ambiguous_usage_evidence_digest(
            ingest_id=ingest_id,
            authoritative_attempt_number=3,
            skipped_attempt_number=4,
            skipped_receipt_digest=receipt_digest,
            skipped_recorded_at=T0 + timedelta(seconds=5),
        )

    fifth = envelope(5)
    policy = _policy(workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY)
    service.register_policy(policy)
    service.open_envelope(fifth)
    service.allocate(_allocation(fifth, policy), owner_emergency_stop=False)
    with pytest.raises(
        ModelUsageIntegrityError,
        match="later allocation is retained",
    ):
        service.native_recovered_ambiguous_usage_evidence_digest(
            ingest_id=ingest_id,
            authoritative_attempt_number=3,
            skipped_attempt_number=4,
            skipped_receipt_digest=receipt_digest,
            skipped_recorded_at=T0 + timedelta(seconds=5),
        )


def test_recovered_gap_observer_rebinds_usage_for_attempt_six() -> None:
    digest = _digest({"usage": "attempt-3-and-rejection-4"})
    proof = SimpleNamespace(
        authoritative_attempt_number=3,
        skipped_attempt_number=4,
        skipped_receipt_digest=_digest({"receipt": 4}),
        skipped_recorded_at=SimpleNamespace(value=T0 + timedelta(seconds=4)),
        settled_usage_evidence_digest=digest,
    )
    envelope = WorkEnvelope.create(
        cycle_id=native_graphiti_usage_cycle_id(
            ingest_id="recovered-observer", attempt_number=6
        ),
        workload_class=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        admitted_at=T0,
        admission_decision_id=None,
        candidate_id=None,
        hypothesis_digest=None,
        evidence_package_digest=None,
        ingest_id="recovered-observer",
        graphiti_attempt_id="recovered-observer:6",
    )
    observer = object.__new__(GraphitiModelUsageObserver)
    observer._service = SimpleNamespace(
        native_recovered_ambiguous_usage_evidence_digest=lambda **values: (
            digest if values["latest_allowed_attempt_number"] == 5 else "stale"
        )
    )
    observer._ingest_obligation_id = "recovered-observer"
    observer._provider_attempt_number = 6
    observer._envelope = envelope
    observer._recovered_ambiguous_progression = proof
    observer._retry_evidence = GraphitiIngestRetryEvidence(
        attempt_numbers=(1, 2, 3, 4, 5),
        zero_dispatch_attempts=(1, 2),
        settled_provider_attempts=(3, 5),
        latest_settled_provider_attempt=5,
        unresolved_attempts=(4,),
    )

    assert observer.allows_fresh_completed_rollback_retry(
        episode_uuid="recovered-observer",
        attempt_number=6,
        prior_attempt_number=5,
    )
    observer._service = SimpleNamespace(
        native_recovered_ambiguous_usage_evidence_digest=lambda **_values: (
            _digest({"usage": "later-allocation"})
        )
    )
    assert not observer.allows_fresh_completed_rollback_retry(
        episode_uuid="recovered-observer",
        attempt_number=6,
        prior_attempt_number=5,
    )


@pytest.mark.parametrize("table,field", (
    ("model_work_envelopes", "ingest_id"),
    ("model_work_envelopes", "graphiti_attempt_id"),
    ("model_work_envelopes", "both-identities"),
    ("model_invocation_allocations", "envelope_id"),
    ("model_work_outcomes", "envelope_id"),
))
def test_graphiti_retry_evidence_rejects_selected_identity_rebinding(tmp_path, table, field):
    service = _service(tmp_path)
    selected, unrelated = (
        _envelope(cycle_id=name, workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
                  candidate_id=None, ingest_id=name)
        for name in ("selected-ingest", "unrelated-ingest")
    )
    service.open_envelope(selected)
    service.open_envelope(unrelated)
    if table == "model_invocation_allocations":
        policy = _policy(
            workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY, provider="cursor-agent-cli",
            route=GRAPHITI_CHAT_PRIMARY_ROUTE, model="composer-2.5",
            hard_estimate_ceiling_tokens=None,
        )
        service.register_policy(policy)
        service.allocate(_allocation(selected, policy), owner_emergency_stop=False)
    if table == "model_work_outcomes":
        service.record_work_outcome(
            envelope_id=selected.envelope_id, outcome="GRAPHITI_FAILED",
            outcome_record_id="controller-refusal", payload_digest=None, terminal_at=T0,
        )
    with sqlite3.connect(service.path) as connection:
        if table == "model_work_envelopes":
            record = selected.as_record()
            for key in ("ingest_id", "graphiti_attempt_id") if field == "both-identities" else (field,):
                record[key] = unrelated.as_record()[key]
            connection.execute("UPDATE model_work_envelopes SET record_json=? WHERE envelope_id=?",
                               (json.dumps(record), selected.envelope_id))
        else:
            connection.execute(f"UPDATE {table} SET envelope_id=? WHERE envelope_id=?",
                               (unrelated.envelope_id, selected.envelope_id))
    with pytest.raises(ModelUsageIntegrityError):
        service.graphiti_ingest_retry_evidence(ingest_id="selected-ingest")


def test_native_conservative_disposition_uses_qualified_policy_bound(
    tmp_path: Path,
) -> None:
    unit = _land_native_ingest(tmp_path)
    service = _service(tmp_path)
    policy, allocation, terminal = _open_unreported_graphiti_subscription_leaf(
        service, ingest_id=unit.ingest_id
    )

    disposition = service.disposition_native_unreported_subscription_usage(
        invocation_id=allocation.invocation_id,
        expected_terminal_digest=terminal.terminal_digest,
        expected_allocation_digest=allocation.canonical_digest,
        observed_at=T0 + timedelta(seconds=10),
    )

    assert disposition["authority_scope"] == "NATIVE_AUTONOMOUS_INTERNAL_PIPELINE"
    assert disposition["usage_status"] == "ESTIMATED"
    assert disposition["components"] == UsageComponents(
        total_tokens=policy.max_total_tokens,
        provenance="BOUNDED_ESTIMATE",
    ).as_record()
    assert disposition["exact_usage_remains_unknown"] is True
    assert disposition["provider_dispatch_preserved"] is True
    assert disposition["unknown_spend_released"] is False
    assert service.terminal(allocation.invocation_id) == terminal
    assert service.disposition_native_unreported_subscription_usage(
        invocation_id=allocation.invocation_id,
        expected_terminal_digest=terminal.terminal_digest,
        expected_allocation_digest=allocation.canonical_digest,
        observed_at=T0 + timedelta(seconds=20),
    ) == disposition
    service.record_work_outcome(
        envelope_id=allocation.envelope_id,
        outcome="GRAPHITI_REJECTED_BINDING",
        outcome_record_id="native-conservative-outcome",
        payload_digest=None,
        terminal_at=T0 + timedelta(seconds=20),
    )
    evidence = service.graphiti_ingest_retry_evidence(ingest_id=unit.ingest_id)
    assert evidence.settled_provider_attempts == (1,)
    assert evidence.unresolved_attempts == ()

    current = service.query(start=T0, end=T0 + timedelta(minutes=1))["leaves"][0]
    assert current["usage_status"] == "ESTIMATED"
    assert current["terminal_usage_status"] == "UNREPORTED"
    assert current["total_tokens"] == policy.max_total_tokens
    assert current["provider_telemetry_digest"] is None
    state = service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)
    service.release_route_circuit(
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        release_kind="AUTHORISED_OPERATOR_RESET",
        bound_failure_reason=str(state["reason"]),
        evidence_digest=str(disposition["disposition_digest"]),
        recorded_at=T0 + timedelta(seconds=21),
    )
    assert service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)["state"] == "CLOSED"


def test_native_conservative_disposition_reads_only_its_landed_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "unpublished.sqlite3"
    connection = connect_unpublished_store(str(path))
    selected_first = replace(_native("selected"), chunk_count=2)
    selected = replace(
        selected_first,
        chunk_ordinal=2,
        predecessor_ingest_id=selected_first.ingest_id,
    )
    unrelated = _native("unrelated")
    journal = NativeRevisionJournal(connection)
    journal.land((selected_first, selected))
    journal.land((unrelated,))
    for ordinal in range(100):
        journal.advance(
            unrelated.revision_id,
            stage="UNRELATED",
            facts={"ordinal": ordinal},
        )
    append_ledger(
        connection,
        "NATIVE_REVISION_LANDED",
        {
            "revision_id": selected.revision_id,
            "units": [asdict(selected_first), asdict(selected)],
        },
    )
    connection.commit()
    connection.close()
    service = _service(tmp_path)
    _policy_value, allocation, terminal = (
        _open_unreported_graphiti_subscription_leaf(
            service,
            ingest_id=selected.ingest_id,
        )
    )

    def refuse_full_replay(*_args: object, **_kwargs: object) -> None:
        pytest.fail("native disposition replayed the full journal")

    monkeypatch.setattr(NativeRevisionJournal, "__init__", refuse_full_replay)

    disposition = service.disposition_native_unreported_subscription_usage(
        invocation_id=allocation.invocation_id,
        expected_terminal_digest=terminal.terminal_digest,
        expected_allocation_digest=allocation.canonical_digest,
        observed_at=T0 + timedelta(seconds=10),
    )

    assert disposition["authority_scope"] == "NATIVE_AUTONOMOUS_INTERNAL_PIPELINE"


def test_native_conservative_disposition_rejects_conflicting_landing(
    tmp_path: Path,
) -> None:
    unit = _land_native_ingest(tmp_path)
    service = _service(tmp_path)
    _policy_value, allocation, terminal = (
        _open_unreported_graphiti_subscription_leaf(
            service,
            ingest_id=unit.ingest_id,
        )
    )
    conflicting = replace(unit, headline="Contradictory retained headline")
    connection = connect_unpublished_store(str(tmp_path / "unpublished.sqlite3"))
    append_ledger(
        connection,
        "NATIVE_REVISION_LANDED",
        {
            "revision_id": unit.revision_id,
            "units": [asdict(conflicting)],
        },
    )
    connection.commit()
    connection.close()

    with pytest.raises(
        ModelUsageIntegrityError,
        match="native conservative source landing changed",
    ):
        service.disposition_native_unreported_subscription_usage(
            invocation_id=allocation.invocation_id,
            expected_terminal_digest=terminal.terminal_digest,
            expected_allocation_digest=allocation.canonical_digest,
            observed_at=T0 + timedelta(seconds=10),
        )


@pytest.mark.parametrize("mutation", ("digest", "canonical_bytes"))
def test_native_conservative_landing_requires_exact_canonical_record(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / "unpublished.sqlite3"
    connection = connect_unpublished_store(str(path))
    unit = _native("selected")
    NativeRevisionJournal(connection).land((unit,))
    raw = str(
        connection.execute(
            "SELECT payload_json FROM ledger WHERE kind='NATIVE_REVISION_LANDED'"
        ).fetchone()[0]
    )
    if mutation == "digest":
        connection.execute(
            "UPDATE ledger SET payload_digest=? "
            "WHERE kind='NATIVE_REVISION_LANDED'",
            (_digest({"wrong": "landing"}),),
        )
    else:
        noncanonical = json.dumps(json.loads(raw), indent=2)
        connection.execute(
            "UPDATE ledger SET payload_json=?,payload_digest=? "
            "WHERE kind='NATIVE_REVISION_LANDED'",
            (noncanonical, digest_bytes(noncanonical.encode("utf-8"))),
        )

    with pytest.raises(
        ModelUsageIntegrityError,
        match="native conservative source landing differs",
    ):
        model_usage_module._native_landed_source_unit(
            connection,
            ingest_id=unit.ingest_id,
            effective_revision_digest=_digest(asdict(unit.effective_revision)),
        )
    connection.close()


def test_native_landing_revision_hint_cannot_select_another_unit(
    tmp_path: Path,
) -> None:
    connection = connect_unpublished_store(str(tmp_path / "unpublished.sqlite3"))
    unit = _native("selected")
    NativeRevisionJournal(connection).land((unit,))

    assert model_usage_module._native_landed_source_unit(
        connection,
        ingest_id=unit.ingest_id,
        effective_revision_digest=_digest({"revision": "another"}),
    ) is None
    connection.close()


def test_native_landing_revision_hint_binds_the_selected_chunk(
    tmp_path: Path,
) -> None:
    connection = connect_unpublished_store(str(tmp_path / "unpublished.sqlite3"))
    first = replace(_native("selected"), chunk_count=2)
    second = replace(
        first,
        chunk_ordinal=2,
        predecessor_ingest_id=first.ingest_id,
    )
    altered_first = replace(
        first,
        effective_revision=replace(
            first.effective_revision,
            first_observed_at="2026-09-02T12:00:01.000000Z",
        ),
    )
    NativeRevisionJournal(connection).land((altered_first, second))

    assert model_usage_module._native_landed_source_unit(
        connection,
        ingest_id=altered_first.ingest_id,
        effective_revision_digest=_digest(asdict(second.effective_revision)),
    ) is None
    assert model_usage_module._native_landed_source_unit(
        connection,
        ingest_id=second.ingest_id,
        effective_revision_digest=_digest(asdict(second.effective_revision)),
    ) == second
    connection.close()


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("wrong_terminal", "terminal differs"),
        ("wrong_allocation", "allocation differs"),
        ("missing_dispatch", "lacks committed dispatch"),
        ("not_native", "landed source observation"),
        ("reported", "target is ineligible"),
        ("wrong_provider", "target is ineligible"),
    ),
)
def test_native_conservative_disposition_rejects_unqualified_targets(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    unit = _native() if case == "not_native" else _land_native_ingest(tmp_path)
    service = _service(tmp_path)
    if case == "wrong_provider":
        envelope = _envelope(
            cycle_id="native-wrong-provider",
            workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            candidate_id=None,
            ingest_id=unit.ingest_id,
        )
        policy = _policy(
            workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            provider="another-cli",
            route=GRAPHITI_CHAT_PRIMARY_ROUTE,
            model="composer-2.5",
            hard_estimate_ceiling_tokens=None,
        )
        service.register_policy(policy)
        service.open_envelope(envelope)
        allocation = _allocation(envelope, policy)
        service.allocate(allocation, owner_emergency_stop=False)
        dispatch_at = allocation.allocated_at + timedelta(milliseconds=1)
        service.observe_transport(
            invocation_id=allocation.invocation_id,
            observed_at=dispatch_at,
            state="DISPATCH_STARTED",
            evidence_digest=_digest({"dispatch": allocation.invocation_id}),
        )
        terminal = service.complete(
            InvocationTerminal.create(
                invocation_id=allocation.invocation_id,
                outcome="FAILED",
                failure_class="MISSING_PROVIDER_TELEMETRY",
                usage_status=UsageStatus.UNREPORTED,
                components=UsageComponents(provenance="UNAVAILABLE"),
                dispatch_at=dispatch_at,
                completed_at=T0 + timedelta(seconds=3),
                observed_at=T0 + timedelta(seconds=3),
                subscription_cli_chat_not_cash_debited=True,
            )
        )
    elif case == "reported":
        envelope = _envelope(
            cycle_id="native-reported",
            workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            candidate_id=None,
            ingest_id=unit.ingest_id,
        )
        policy = _policy(
            workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            provider="cursor-agent-cli",
            route=GRAPHITI_CHAT_PRIMARY_ROUTE,
            model="composer-2.5",
            hard_estimate_ceiling_tokens=None,
        )
        service.register_policy(policy)
        service.open_envelope(envelope)
        allocation = _allocation(envelope, policy)
        service.allocate(allocation, owner_emergency_stop=False)
        service.observe_transport(
            invocation_id=allocation.invocation_id,
            observed_at=allocation.allocated_at + timedelta(milliseconds=1),
            state="DISPATCH_STARTED",
            evidence_digest=_digest({"dispatch": allocation.invocation_id}),
        )
        terminal = _reported(allocation)
        service.complete(
            terminal,
            provider_telemetry={"invocation": allocation.invocation_id},
        )
    else:
        _policy_value, allocation, terminal = (
            _open_unreported_graphiti_subscription_leaf(
                service,
                ingest_id=unit.ingest_id,
                observe_dispatch=case != "missing_dispatch",
            )
        )

    with pytest.raises(ModelUsageIntegrityError, match=message):
        service.disposition_native_unreported_subscription_usage(
            invocation_id=allocation.invocation_id,
            expected_terminal_digest=(
                _digest({"wrong": "terminal"})
                if case == "wrong_terminal"
                else terminal.terminal_digest
            ),
            expected_allocation_digest=(
                _digest({"wrong": "allocation"})
                if case == "wrong_allocation"
                else allocation.canonical_digest
            ),
            observed_at=T0 + timedelta(seconds=10),
        )
    assert service_row_count(
        tmp_path / "unpublished.sqlite3",
        "model_usage_conservative_dispositions",
    ) == 0


def test_authorised_conservative_disposition_preserves_unknown_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    policy, allocation, terminal = _open_unreported_graphiti_subscription_leaf(
        service
    )
    _bind_issue_790_fixture_contract(
        monkeypatch,
        allocation=allocation,
        terminal=terminal,
    )
    approved_at = T0 + timedelta(seconds=5)
    authority, authority_digest = _conservative_disposition_authority(
        allocation,
        terminal,
        approved_at=approved_at,
    )

    disposition = service.disposition_unreported_subscription_usage(
        invocation_id=allocation.invocation_id,
        expected_terminal_digest=terminal.terminal_digest,
        expected_allocation_digest=allocation.canonical_digest,
        approved_by=str(authority["approved_by"]),
        approval_reference=str(authority["approval_reference"]),
        approved_at=approved_at,
        approved_plan_digest=FIXTURE_790_PLAN_DIGEST,
        authority_digest=authority_digest,
        observed_at=T0 + timedelta(seconds=10),
    )

    assert disposition["schema_version"] == (
        "newsroom.model-usage.conservative-disposition.v2"
    )
    assert disposition["usage_status"] == "ESTIMATED"
    assert disposition["components"] == UsageComponents(
        total_tokens=policy.max_total_tokens,
        provenance="BOUNDED_ESTIMATE",
    ).as_record()
    assert disposition["exact_usage_remains_unknown"] is True
    assert disposition["provider_dispatch_preserved"] is True
    assert disposition["unknown_spend_released"] is False
    assert disposition["authority_digest"] == authority_digest
    assert service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)["state"] == "OPEN"

    historical = service.query(
        start=T0, end=T0 + timedelta(seconds=9)
    )["leaves"][0]
    current = service.query(
        start=T0, end=T0 + timedelta(seconds=20)
    )["leaves"][0]
    assert historical["usage_status"] == "UNREPORTED"
    assert historical["conservative_disposition_digest"] is None
    assert current["usage_status"] == "ESTIMATED"
    assert current["terminal_usage_status"] == "UNREPORTED"
    assert current["disposition_usage_status"] == "ESTIMATED"
    assert current["total_tokens"] == policy.max_total_tokens
    assert current["exact_usage_remains_unknown"] is True
    assert current["disposition_approved_plan_digest"] == FIXTURE_790_PLAN_DIGEST
    assert current["provider_telemetry_digest"] is None
    assert current["actual_provider_dispatch"] is True

    report = service.report(start=T0, end=T0 + timedelta(seconds=20))
    assert report["estimated_tokens"] == policy.max_total_tokens
    assert report["unresolved_invocation_count"] == 0
    assert report["missing_usage_is_zero"] is False
    assert service_row_count(
        tmp_path / "unpublished.sqlite3",
        "model_usage_conservative_dispositions",
    ) == 1
    assert service_row_count(
        tmp_path / "unpublished.sqlite3", "model_usage_reconciliations"
    ) == 0
    assert service_row_count(
        tmp_path / "unpublished.sqlite3", "model_provider_telemetry"
    ) == 0

    open_state = service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)
    service.release_route_circuit(
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        release_kind="AUTHORISED_OPERATOR_RESET",
        bound_failure_reason=str(open_state["reason"]),
        evidence_digest=str(disposition["disposition_digest"]),
        recorded_at=T0 + timedelta(seconds=11),
    )
    assert service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)["state"] == "CLOSED"


def test_authorised_timeout_disposition_uses_the_exact_reviewed_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    policy, allocation, terminal = _open_unreported_graphiti_subscription_leaf(
        service,
        outcome="TIMEOUT",
    )
    _bind_issue_790_fixture_contract(
        monkeypatch,
        allocation=allocation,
        terminal=terminal,
    )
    approved_at = T0 + timedelta(seconds=5)
    authority, authority_digest = _conservative_disposition_authority(
        allocation,
        terminal,
        approved_at=approved_at,
    )

    disposition = service.disposition_unreported_subscription_usage(
        invocation_id=allocation.invocation_id,
        expected_terminal_digest=terminal.terminal_digest,
        expected_allocation_digest=allocation.canonical_digest,
        approved_by=str(authority["approved_by"]),
        approval_reference=str(authority["approval_reference"]),
        approved_at=approved_at,
        approved_plan_digest=FIXTURE_790_PLAN_DIGEST,
        authority_digest=authority_digest,
        observed_at=T0 + timedelta(seconds=10),
    )

    assert disposition["usage_status"] == "ESTIMATED"
    assert disposition["components"] == UsageComponents(
        total_tokens=policy.max_total_tokens,
        provenance="BOUNDED_ESTIMATE",
    ).as_record()
    assert disposition["exact_usage_remains_unknown"] is True
    assert disposition["provider_dispatch_preserved"] is True
    assert disposition["unknown_spend_released"] is False


def test_conservative_disposition_replay_is_idempotent_and_conflicts_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    _policy_value, allocation, terminal = (
        _open_unreported_graphiti_subscription_leaf(service)
    )
    _bind_issue_790_fixture_contract(
        monkeypatch,
        allocation=allocation,
        terminal=terminal,
    )
    approved_at = T0 + timedelta(seconds=5)
    authority, authority_digest = _conservative_disposition_authority(
        allocation,
        terminal,
        approved_at=approved_at,
    )
    values = {
        "invocation_id": allocation.invocation_id,
        "expected_terminal_digest": terminal.terminal_digest,
        "expected_allocation_digest": allocation.canonical_digest,
        "approved_by": str(authority["approved_by"]),
        "approval_reference": str(authority["approval_reference"]),
        "approved_at": approved_at,
        "approved_plan_digest": FIXTURE_790_PLAN_DIGEST,
        "authority_digest": authority_digest,
    }

    first = service.disposition_unreported_subscription_usage(
        **values,
        observed_at=T0 + timedelta(seconds=10),
    )
    replay = service.disposition_unreported_subscription_usage(
        **values,
        observed_at=T0 + timedelta(seconds=20),
    )
    assert replay == first

    with pytest.raises(
        ModelUsageIntegrityError,
        match="conservative disposition authority differs",
    ):
        service.disposition_unreported_subscription_usage(
            **{**values, "authority_digest": _digest({"wrong": True})},
            observed_at=T0 + timedelta(seconds=30),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("terminal", "approved terminal differs"),
        ("allocation", "approved allocation differs"),
        ("approval", "approval authority differs"),
        ("authority", "authority differs"),
    ),
)
def test_conservative_disposition_requires_exact_bound_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    service = _service(tmp_path)
    _policy_value, allocation, terminal = (
        _open_unreported_graphiti_subscription_leaf(service)
    )
    _bind_issue_790_fixture_contract(
        monkeypatch,
        allocation=allocation,
        terminal=terminal,
    )
    approved_at = T0 + timedelta(seconds=5)
    authority, authority_digest = _conservative_disposition_authority(
        allocation,
        terminal,
        approved_at=approved_at,
    )
    values = {
        "invocation_id": allocation.invocation_id,
        "expected_terminal_digest": terminal.terminal_digest,
        "expected_allocation_digest": allocation.canonical_digest,
        "approved_by": str(authority["approved_by"]),
        "approval_reference": str(authority["approval_reference"]),
        "approved_at": approved_at,
        "approved_plan_digest": FIXTURE_790_PLAN_DIGEST,
        "authority_digest": authority_digest,
        "observed_at": T0 + timedelta(seconds=10),
    }
    if mutation == "terminal":
        values["expected_terminal_digest"] = _digest({"wrong": "terminal"})
    elif mutation == "allocation":
        values["expected_allocation_digest"] = _digest({"wrong": "allocation"})
    elif mutation == "approval":
        values["approved_by"] = "github:other"
    else:
        values["authority_digest"] = _digest({"wrong": "authority"})

    with pytest.raises(ModelUsageIntegrityError, match=message):
        service.disposition_unreported_subscription_usage(
            **values  # type: ignore[arg-type]
        )


def test_conservative_disposition_rejects_missing_committed_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    _policy_value, allocation, terminal = (
        _open_unreported_graphiti_subscription_leaf(
            service,
            observe_dispatch=False,
        )
    )
    _bind_issue_790_fixture_contract(
        monkeypatch,
        allocation=allocation,
        terminal=terminal,
    )
    approved_at = T0 + timedelta(seconds=5)
    authority, authority_digest = _conservative_disposition_authority(
        allocation,
        terminal,
        approved_at=approved_at,
    )

    with pytest.raises(
        ModelUsageIntegrityError,
        match="committed transport dispatch is absent",
    ):
        service.disposition_unreported_subscription_usage(
            invocation_id=allocation.invocation_id,
            expected_terminal_digest=terminal.terminal_digest,
            expected_allocation_digest=allocation.canonical_digest,
            approved_by=str(authority["approved_by"]),
            approval_reference=str(authority["approval_reference"]),
            approved_at=approved_at,
            approved_plan_digest=FIXTURE_790_PLAN_DIGEST,
            authority_digest=authority_digest,
            observed_at=T0 + timedelta(seconds=10),
        )


def test_conservative_disposition_does_not_clear_other_route_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    policy = _policy(
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        provider="cursor-agent-cli",
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        model="composer-2.5",
        hard_estimate_ceiling_tokens=None,
    )
    service.register_policy(policy)
    allocations = []
    for cycle_id, request in (
        ("graphiti-cycle-first", "graphiti-first"),
        ("graphiti-cycle-second", "graphiti-second"),
    ):
        envelope = _envelope(
            cycle_id=cycle_id,
            workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            candidate_id=None,
            ingest_id=f"ingest-{cycle_id}",
        )
        service.open_envelope(envelope)
        allocation = _allocation(envelope, policy, request=request)
        service.allocate(allocation, owner_emergency_stop=False)
        allocations.append(allocation)
    terminals = []
    for allocation in allocations:
        dispatch_at = allocation.allocated_at + timedelta(milliseconds=1)
        service.observe_transport(
            invocation_id=allocation.invocation_id,
            observed_at=dispatch_at,
            state="DISPATCH_STARTED",
            evidence_digest=_digest({"dispatch": allocation.invocation_id}),
        )
        terminals.append(
            service.complete(
                InvocationTerminal.create(
                    invocation_id=allocation.invocation_id,
                    outcome="FAILED",
                    failure_class="MISSING_PROVIDER_TELEMETRY",
                    usage_status=UsageStatus.UNREPORTED,
                    components=UsageComponents(provenance="UNAVAILABLE"),
                    dispatch_at=dispatch_at,
                    completed_at=allocation.allocated_at + timedelta(seconds=1),
                    observed_at=allocation.allocated_at + timedelta(seconds=1),
                    subscription_cli_chat_not_cash_debited=True,
                )
            )
        )
    first, _second = allocations
    first_terminal, _second_terminal = terminals
    _bind_issue_790_fixture_contract(
        monkeypatch,
        allocation=first,
        terminal=first_terminal,
    )
    approved_at = T0 + timedelta(seconds=5)
    authority, authority_digest = _conservative_disposition_authority(
        first,
        first_terminal,
        approved_at=approved_at,
    )
    disposition = service.disposition_unreported_subscription_usage(
        invocation_id=first.invocation_id,
        expected_terminal_digest=first_terminal.terminal_digest,
        expected_allocation_digest=first.canonical_digest,
        approved_by=str(authority["approved_by"]),
        approval_reference=str(authority["approval_reference"]),
        approved_at=approved_at,
        approved_plan_digest=FIXTURE_790_PLAN_DIGEST,
        authority_digest=authority_digest,
        observed_at=T0 + timedelta(seconds=10),
    )

    open_state = service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)
    with pytest.raises(
        ModelUsageAdmissionError,
        match="unresolved usage or a policy breach",
    ):
        service.release_route_circuit(
            route=GRAPHITI_CHAT_PRIMARY_ROUTE,
            release_kind="AUTHORISED_OPERATOR_RESET",
            bound_failure_reason=str(open_state["reason"]),
            evidence_digest=str(disposition["disposition_digest"]),
            recorded_at=T0 + timedelta(seconds=11),
        )


def test_late_provider_telemetry_supersedes_conservative_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    _policy_value, allocation, terminal = (
        _open_unreported_graphiti_subscription_leaf(service)
    )
    _bind_issue_790_fixture_contract(
        monkeypatch,
        allocation=allocation,
        terminal=terminal,
    )
    approved_at = T0 + timedelta(seconds=5)
    authority, authority_digest = _conservative_disposition_authority(
        allocation,
        terminal,
        approved_at=approved_at,
    )
    service.disposition_unreported_subscription_usage(
        invocation_id=allocation.invocation_id,
        expected_terminal_digest=terminal.terminal_digest,
        expected_allocation_digest=allocation.canonical_digest,
        approved_by=str(authority["approved_by"]),
        approval_reference=str(authority["approval_reference"]),
        approved_at=approved_at,
        approved_plan_digest=FIXTURE_790_PLAN_DIGEST,
        authority_digest=authority_digest,
        observed_at=T0 + timedelta(seconds=10),
    )
    service.reconcile(
        invocation_id=allocation.invocation_id,
        components=UsageComponents(
            input_tokens=100,
            output_tokens=25,
            total_tokens=125,
            provenance="PROVIDER_REPORTED",
        ),
        provider_telemetry={"late": "exact"},
        observed_at=T0 + timedelta(seconds=20),
        raw_telemetry_pointer="private://late/exact",
    )

    row = service.query(start=T0, end=T0 + timedelta(seconds=30))["leaves"][0]
    assert row["usage_status"] == "REPORTED"
    assert row["total_tokens"] == 125
    assert row["reconciliation_usage_status"] == "REPORTED"
    assert row["disposition_usage_status"] == "ESTIMATED"
    assert row["exact_usage_remains_unknown"] is False


def test_checked_issue_790_live_plan_retains_exact_approved_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_issue_790_plan(
        root
        / "docs/operations/2026-08-26-issue-790-conservative-disposition.json"
    )

    assert plan["canonical_digest"] == (
        "sha256:ce7ee7fd56c931b147158dad2a74047ada90b805e5a4c545e53db1f4d2ae7383"
    )
    assert plan["target"] == {
        "allocation_digest": (
            "sha256:800dd0c6155a34cfafe91c1c240dac2d44730f558be9417d5fe34b5fb23780b2"
        ),
        "conservative_total_source": "QUALIFIED_POLICY_MAX_TOTAL_TOKENS",
        "expected_conservative_total_tokens": 147456,
        "invocation_id": (
            "sha256:75f14fd50f54c01c852c557291eb7bb92b05a79c937d10d048bb245863b7a196"
        ),
        "policy_digest": (
            "sha256:c3a876540d2d1d2b3cf4864f649340c4357c8619468841bea5640b8d3567db3c"
        ),
        "provider": "cursor-agent-cli",
        "route": "GRAPHITI_CHAT_PRIMARY",
        "route_open_reason": "SYSTEMIC_TRANSPORT",
        "terminal_digest": (
            "sha256:0c73f6a7ad2255f13bfdb617370f0c935464917e0e80c69b2da216ffca60ee0c"
        ),
        "terminal_failure_class": "MISSING_PROVIDER_TELEMETRY",
        "terminal_usage_status": "UNREPORTED",
        "workload_class": "GRAPHITI_CHAT_PRIMARY",
    }


def test_checked_issue_790_success_sequence_plan_binds_initial_160_second_step() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_issue_790_plan(
        root
        / "docs/operations/2026-08-26-issue-790-success-sequence-step-1.json"
    )

    assert plan["canonical_digest"] == (
        "sha256:3347669cc57fcc3740f9e7027cf7c9c6936626dfb1932eeec5ea2018fe6f6308"
    )
    assert plan["schema_version"] == "newsroom.issue-790.iterative-canary-plan.v2"
    assert plan["target"] == {
        "allocation_digest": (
            "sha256:468bc90fb8c9114ca8d4fc780d137f676ce69b453fcfda74bef88e0508a15643"
        ),
        "conservative_total_source": "QUALIFIED_POLICY_MAX_TOTAL_TOKENS",
        "expected_conservative_total_tokens": 147456,
        "invocation_id": (
            "sha256:8e219f498ee1eff71cd21c5d9dd3d958e5aed62db8f938b0a2bfdba6d4e9de7d"
        ),
        "policy_digest": (
            "sha256:e29581d0488a5da9d869a26e5bd0d599ff3180c63204c420ebab999bf737d63c"
        ),
        "provider": "cursor-agent-cli",
        "route": "GRAPHITI_CHAT_PRIMARY",
        "route_open_reason": "TIMEOUT",
        "terminal_digest": (
            "sha256:f5e67d327b215c1eda3a320b07e2cee642151880c5fa275686e8d534646ca9b9"
        ),
        "terminal_failure_class": "MISSING_PROVIDER_TELEMETRY",
        "terminal_outcome": "TIMEOUT",
        "terminal_usage_status": "UNREPORTED",
        "workload_class": "GRAPHITI_CHAT_PRIMARY",
    }
    sequence = dict(plan["sequence"])
    assert {
        key: sequence[key]
        for key in (
            "call_shape_policy_digest",
            "call_shape_policy_version",
            "cleanup_reserve_ms",
            "constraint_change",
            "controller_timeout_ms",
            "extraction_timeout_ms",
            "fixed_constraints_digest",
            "root_plan_digest",
            "sequence_ordinal",
            "stop_condition",
            "timeout_increment_ms",
        )
    } == {
        "call_shape_policy_digest": (
            "sha256:7e6bd15613cefda0820a1d339c8790f0185946aade1622dadbb8c468f558bb18"
        ),
        "call_shape_policy_version": "issue-790-v9",
        "cleanup_reserve_ms": 20_000,
        "constraint_change": "INITIAL_QUALIFIED_BASELINE",
        "controller_timeout_ms": 160_000,
        "extraction_timeout_ms": 180_000,
        "fixed_constraints_digest": (
            "sha256:a3d6a7759c57df52e0a25feae3edcc740ce7ec26064996aae018b276fd36fbb2"
        ),
        "root_plan_digest": (
            "sha256:ce7ee7fd56c931b147158dad2a74047ada90b805e5a4c545e53db1f4d2ae7383"
        ),
        "sequence_ordinal": 1,
        "stop_condition": "FIRST_TRUTHFUL_PROVIDER_BACKED_SUCCESS",
        "timeout_increment_ms": 10_000,
    }
    assert sequence["predecessor"] == {
        "consumption_digest": (
            "sha256:6ec11fc9d1ee75421c6a4d5419e96abc6a960dd4e02b1c53ac396f6433d6b07a"
        ),
        "event_id": (
            "sha256:46663a38929b02c74275e39ea986c416f9893ebeaf9b81fc92d34ac0b9e2efcd"
        ),
        "ledger_seq": 8891,
        "outcome_digest": (
            "sha256:4c4ec972b8569e05c45dec60d549d9a5eac1950045d2454da9d86cd76293d05c"
        ),
        "plan_digest": (
            "sha256:ce7ee7fd56c931b147158dad2a74047ada90b805e5a4c545e53db1f4d2ae7383"
        ),
    }
    causal_report = dict(sequence["predecessor_causal_report"])
    assert causal_report["report_digest"] == (
        "sha256:cb1b72361e6f17d02e5f8ecce30d2ff53a79e9334ba942728f58fcf8d977f7f2"
    )
    assert causal_report["classification"] == "CONTROLLER_TIMEOUT"
    assert causal_report["causal_constraint"] == "CONTROLLER_TIMEOUT_MS"
    assert causal_report["provider_cause"] == "UNOBSERVED"
    assert causal_report["diagnostic"] == {
        "boundary": "CONTROLLER_DEADLINE",
        "cause": "CONFIGURED_TIMEOUT_EXPIRED",
        "configured_timeout_ms": 80_000,
        "deadline_at": "2026-08-26T17:04:50.926743Z",
        "elapsed_ms": 81_217,
        "last_progress": "DISPATCH_STARTED",
        "phase": "PRIMARY_TRANSPORT",
        "process": "CLI_CHILD",
        "provider_cause": "UNOBSERVED",
        "schema_version": "newsroom.graphiti-timeout-diagnostic.v1",
        "termination": "PROCESS_KILLED",
    }


def test_issue_790_success_sequence_fails_closed_on_call_shape_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(
        issue_790_operation,
        "load_checked_graphiti_call_shape_policy",
        lambda: SimpleNamespace(
            canonical_digest=_digest({"drifted": "call-shape"}),
            version="issue-790-drifted",
            qualified_routes=(),
        ),
        raising=False,
    )

    with pytest.raises(Issue790DispositionError, match="call-shape policy differs"):
        load_issue_790_plan(
            root
            / "docs/operations/2026-08-28-issue-790-success-sequence-step-15.json"
        )


def test_checked_issue_790_step_two_binds_reviewed_non_timeout_fix() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_issue_790_plan(
        root
        / "docs/operations/2026-08-26-issue-790-success-sequence-step-2.json"
    )

    assert plan["canonical_digest"] == (
        "sha256:f759403c9838ef431cff38126989d8732ed5ba7e03ce92f554760ba0ef2d2c61"
    )
    sequence = dict(plan["sequence"])
    assert sequence["sequence_ordinal"] == 2
    assert sequence["constraint_change"] == "REVIEWED_NON_TIMEOUT_FIX"
    assert sequence["predecessor_causal_report"]["report_digest"] == (
        "sha256:0f06ffa65fc95a8e3278fccc92eed8dc23cebf5517a722d74bce14c73e2984a8"
    )
    assert sequence["reviewed_fix"]["record_digest"] == (
        "sha256:1bfba70f2f88eec47da9d8329030239c316cdc995b519c929fe074dcb9b14e32"
    )


def test_checked_issue_790_step_three_binds_unpinned_harness_fix() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_issue_790_plan(
        root
        / "docs/operations/2026-08-26-issue-790-success-sequence-step-3.json"
    )

    assert plan["canonical_digest"] == (
        "sha256:598bc32d1e9c662d19188df6b0d038ac4205641d59810356504ee3da805250d4"
    )
    assert plan["target"]["invocation_id"] == (
        "sha256:e6a0ffed3f985874890cecb49fe39ffac4cda35b0500d15974138afb733deb98"
    )
    sequence = dict(plan["sequence"])
    assert sequence["sequence_ordinal"] == 3
    assert sequence["constraint_change"] == "REVIEWED_NON_TIMEOUT_FIX"
    assert sequence["call_shape_policy_version"] == "issue-790-v10"
    assert sequence["fixed_constraints_digest"] == (
        "sha256:84400663bfddfef14935cdf9c6a0942d548adeab08a732b023e19876de2b2fc2"
    )
    assert sequence["reviewed_fix"]["reviewed_fix_revision"] == (
        "c66eaf698a310413da015fdde7d9d67699ffaafc"
    )


def test_checked_issue_790_step_four_binds_process_exit_diagnostic_fix() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_issue_790_plan(
        root
        / "docs/operations/2026-08-27-issue-790-success-sequence-step-4.json"
    )

    assert plan["canonical_digest"] == (
        "sha256:12e2aa639b1d378b48d1a8ae10113720f887e679432b1f8866aaff3576df98fd"
    )
    sequence = dict(plan["sequence"])
    assert sequence["sequence_ordinal"] == 4
    assert sequence["predecessor_causal_report"]["local_cause"] == (
        "CURSOR_NONZERO_EXIT_DIAGNOSTIC_DISCARDED"
    )
    assert sequence["reviewed_fix"]["reviewed_fix_revision"] == (
        "edc230449cfd31ab846bc95bfaffa16d1d8576a6"
    )


def test_checked_issue_790_step_five_binds_call_shape_alignment() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_issue_790_plan(
        root
        / "docs/operations/2026-08-27-issue-790-success-sequence-step-5.json"
    )

    assert plan["canonical_digest"] == (
        "sha256:cca39c56b4c8368fc87b262b501f55b2e754f923eda83f38330c099f1888dacb"
    )
    sequence = dict(plan["sequence"])
    assert sequence["sequence_ordinal"] == 5
    assert sequence["predecessor_causal_report"]["local_cause"] == (
        "CALL_SHAPE_IMPLEMENTATION_REVISION_DRIFT"
    )
    assert sequence["reviewed_fix"]["reviewed_fix_revision"] == (
        "d540de3863e21ec1f0ce3c486d471c373bbee903"
    )


def test_checked_issue_790_step_six_is_withdrawn_before_sdk_successor() -> None:
    root = Path(__file__).resolve().parents[2]
    withdrawal = json.loads(
        (
            root
            / "docs/operations/2026-08-27-issue-790-success-sequence-step-6-withdrawal.json"
        ).read_text(encoding="utf-8")
    )

    assert withdrawal["plan_status"] == "WITHDRAWN"
    assert withdrawal["superseded_by"] == "SUPERSEDED_BY_SDK_ARCHITECTURE"
    assert withdrawal["withdrawn_plan_digest"] == (
        "sha256:be8ccb6cec126cdaffe9801421cfc115d4651b5a305435a7e820290e17099239"
    )
    draft = json.loads(
        (
            root
            / "docs/operations/2026-08-27-issue-790-success-sequence-step-7-draft.json"
        ).read_text(encoding="utf-8")
    )
    assert withdrawal["successor_plan_digest"] == draft["canonical_digest"]
    assert withdrawal["successor_executable"] is False

    step_six = validate_issue_790_plan(
        json.loads(
            (
                root
                / "docs/operations/2026-08-27-issue-790-success-sequence-step-6.json"
            ).read_text(encoding="utf-8")
        )
    )
    with pytest.raises(Issue790DispositionError, match="call-shape policy differs"):
        issue_790_operation._require_iterative_call_shape(step_six)


def test_checked_issue_790_step_seven_remains_non_executable_draft() -> None:
    root = Path(__file__).resolve().parents[2]
    draft = json.loads(
        (
            root
            / "docs/operations/2026-08-27-issue-790-success-sequence-step-7-draft.json"
        ).read_text(encoding="utf-8")
    )
    step_six = json.loads(
        (
            root
            / "docs/operations/2026-08-27-issue-790-success-sequence-step-6.json"
        ).read_text(encoding="utf-8")
    )

    assert draft["plan_status"] == "DRAFT"
    assert draft["executable"] is False
    assert draft["sequence_ordinal"] == 7
    assert draft["retained_predecessor_causal_report_digest"] == (
        step_six["sequence"]["predecessor_causal_report"]["report_digest"]
    )
    assert step_six["sequence"]["predecessor_causal_report"]["provider_cause"] == (
        "NETWORK"
    )
    assert step_six["sequence"]["predecessor_causal_report"]["termination"] == (
        "PROCESS_EXITED_NONZERO"
    )
    assert draft["intended_sequence"]["call_shape_policy_version"] == "issue-807-v1"
    assert draft["satisfied_prerequisites"] == ["PR_810_MERGED"]
    assert "PR_810_MERGED" not in draft["blocked_until"]
    assert draft["provider_free_qualification_path"] == (
        "docs/operations/2026-08-27-issue-790-provider-free-sdk-qualification.json"
    )
    assert draft["intended_reviewed_fix"]["merge_sha"] == (
        "4dbe41c4178d689e41dfaf5497ee157b5b084c1c"
    )

    with pytest.raises(Issue790DispositionError, match="plan fields differ"):
        load_issue_790_plan(
            root
            / "docs/operations/2026-08-27-issue-790-success-sequence-step-7-draft.json"
        )


def test_checked_issue_790_step_eight_remains_non_executable_draft() -> None:
    root = Path(__file__).resolve().parents[2]
    draft = json.loads(
        (
            root
            / "docs/operations/2026-08-27-issue-790-success-sequence-step-8-draft.json"
        ).read_text(encoding="utf-8")
    )

    assert draft["plan_status"] == "DRAFT"
    assert draft["executable"] is False
    assert draft["sequence_ordinal"] == 8
    assert draft["withdrawal_reference"] == (
        "docs/operations/2026-08-27-issue-790-step-7-exact-pin-withdrawal.json"
    )
    assert draft["intended_sequence"]["call_shape_policy_version"] == "issue-816-v2"
    # Historical draft stays pinned to the Step 8 call-shape; tip enforcement is Step 9.
    assert draft["intended_sequence"]["call_shape_policy_digest"] == (
        "sha256:00e094297b2423272ade50b52a3ce8a75392128b6d60c70ea24b07f0ff8e89a2"
    )
    assert draft["intended_sequence"]["sdk_floor"] == "cursor-sdk>=1.0.29"
    assert draft["intended_sequence"]["composer_floor"] == "composer>=2.5"
    assert "OWNER_AUTHENTICATED_SDK_CANARY_APPROVAL" in draft["satisfied_prerequisites"]
    assert draft["blocked_until"] == []
    assert draft["executable_plan_digest"] == (
        "sha256:e45fb670577de1b929a4c7cde114e6cc05c589ff7e010abbab9656445a2edb8c"
    )
    assert draft["provider_free_qualification_path"] == (
        "docs/operations/2026-08-27-issue-790-compatibility-floor-provider-free-qualification.json"
    )

    with pytest.raises(Issue790DispositionError, match="plan fields differ"):
        load_issue_790_plan(
            root
            / "docs/operations/2026-08-27-issue-790-success-sequence-step-8-draft.json"
        )


def test_checked_issue_790_step_eight_executable_plan_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_issue_790_plan(
        root / "docs/operations/2026-08-27-issue-790-success-sequence-step-8.json"
    )
    contract = issue_790_contract_module.issue_790_approved_plan_contract(
        plan["canonical_digest"]
    )

    assert plan["sequence"]["constraint_change"] == "COMPATIBILITY_FLOOR_ARCHITECTURE"
    assert plan["sequence"]["call_shape_policy_version"] == "issue-816-v2"
    assert contract.sequence_ordinal == 8
    assert contract.constraint_change == "COMPATIBILITY_FLOOR_ARCHITECTURE"
    assert (
        contract.predecessor_plan_digest
        == issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_5_PLAN_DIGEST
    )


def test_issue_790_invocation_plan_digests_cover_shared_successor_chain() -> None:
    invocation_id = (
        "sha256:d0712807fd025520d0a94e5a28c532d4cb8684c936387290fe7eeb49d0b2336c"
    )
    digests = issue_790_contract_module.issue_790_invocation_plan_digests(
        invocation_id
    )

    assert (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_4_PLAN_DIGEST
        in digests
    )
    assert (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_5_PLAN_DIGEST
        in digests
    )
    assert (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_8_PLAN_DIGEST
        in digests
    )
    assert (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_9_PLAN_DIGEST
        in digests
    )
    assert (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_10_PLAN_DIGEST
        in digests
    )
    assert (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_11_PLAN_DIGEST
        in digests
    )


def test_checked_issue_790_step_nine_executable_plan_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_issue_790_plan(
        root / "docs/operations/2026-08-27-issue-790-success-sequence-step-9.json"
    )
    contract = issue_790_contract_module.issue_790_approved_plan_contract(
        plan["canonical_digest"]
    )
    qualification = json.loads(
        (
            root
            / "docs/operations/2026-08-27-issue-790-step-9-cursor-api-key-configuration-qualification.json"
        ).read_text(encoding="utf-8")
    )

    assert plan["canonical_digest"] == (
        "sha256:ed9ddb506df0cc6572777bd9a91249b474032eb4dea6458d2c3e9b0466cbce77"
    )
    assert plan["sequence"]["sequence_ordinal"] == 9
    assert plan["sequence"]["constraint_change"] == "REVIEWED_NON_TIMEOUT_FIX"
    assert plan["sequence"]["predecessor"]["plan_digest"] == (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_8_PLAN_DIGEST
    )
    assert plan["sequence"]["reviewed_fix"]["fix_kind"] == "CONFIGURATION"
    assert contract.sequence_ordinal == 9
    assert contract.constraint_change == "REVIEWED_NON_TIMEOUT_FIX"
    assert (
        contract.predecessor_plan_digest
        == issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_8_PLAN_DIGEST
    )
    assert qualification["qualification"] == (
        "CURSOR_API_KEY_CONFIGURATION_RECOVERED"
    )
    assert qualification["provider_calls"] == 0


def test_checked_issue_790_step_ten_executable_plan_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_issue_790_plan(
        root / "docs/operations/2026-08-27-issue-790-success-sequence-step-10.json"
    )
    contract = issue_790_contract_module.issue_790_approved_plan_contract(
        plan["canonical_digest"]
    )
    qualification = json.loads(
        (
            root
            / "docs/operations/2026-08-27-issue-790-step-10-local-sdk-idempotency-qualification.json"
        ).read_text(encoding="utf-8")
    )

    assert plan["canonical_digest"] == (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_10_PLAN_DIGEST
    )
    assert plan["sequence"]["sequence_ordinal"] == 10
    assert plan["sequence"]["constraint_change"] == "REVIEWED_NON_TIMEOUT_FIX"
    assert plan["sequence"]["predecessor"]["plan_digest"] == (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_9_PLAN_DIGEST
    )
    assert plan["sequence"]["reviewed_fix"]["fix_kind"] == "CODE"
    assert plan["sequence"]["predecessor_causal_report"]["local_cause"] == (
        "LOCAL_SDK_IDEMPOTENCY_KEY_UNSUPPORTED"
    )
    assert contract.sequence_ordinal == 10
    assert contract.constraint_change == "REVIEWED_NON_TIMEOUT_FIX"
    assert (
        contract.predecessor_plan_digest
        == issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_9_PLAN_DIGEST
    )
    assert qualification["qualification"] == (
        "LOCAL_SDK_IDEMPOTENCY_KEY_OMIT_PROVIDER_FREE_READY"
    )
    assert qualification["provider_calls"] == 0


def test_checked_issue_790_step_eleven_executable_plan_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_issue_790_plan(
        root / "docs/operations/2026-08-27-issue-790-success-sequence-step-11.json"
    )
    contract = issue_790_contract_module.issue_790_approved_plan_contract(
        plan["canonical_digest"]
    )
    qualification = json.loads(
        (
            root
            / "docs/operations/2026-08-27-issue-790-step-11-wire-constraint-qualification.json"
        ).read_text(encoding="utf-8")
    )

    assert plan["canonical_digest"] == (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_11_PLAN_DIGEST
    )
    assert plan["sequence"]["sequence_ordinal"] == 11
    assert plan["sequence"]["constraint_change"] == "REVIEWED_NON_TIMEOUT_FIX"
    assert plan["sequence"]["predecessor"]["plan_digest"] == (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_10_PLAN_DIGEST
    )
    assert plan["sequence"]["reviewed_fix"]["fix_kind"] == "CODE"
    assert plan["sequence"]["predecessor_causal_report"]["local_cause"] == (
        "IDENTITY_INVALID_ENTITY_TYPE_ID_OUTSIDE_GOVERNED_ONTOLOGY"
    )
    assert contract.sequence_ordinal == 11
    assert contract.constraint_change == "REVIEWED_NON_TIMEOUT_FIX"
    assert (
        contract.predecessor_plan_digest
        == issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_10_PLAN_DIGEST
    )
    assert qualification["qualification"] == (
        "COMBINED_TEMPORAL_WIRE_CONSTRAINT_PROVIDER_FREE_READY"
    )
    assert qualification["provider_calls"] == 0



def test_checked_issue_790_step_twelve_executable_plan_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_issue_790_plan(
        root / "docs/operations/2026-08-28-issue-790-success-sequence-step-12.json"
    )
    contract = issue_790_contract_module.issue_790_approved_plan_contract(
        plan["canonical_digest"]
    )
    qualification = json.loads(
        (
            root
            / "docs/operations/2026-08-28-issue-790-step-12-fact-attribution-qualification.json"
        ).read_text(encoding="utf-8")
    )

    assert plan["canonical_digest"] == (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_12_PLAN_DIGEST
    )
    assert plan["sequence"]["sequence_ordinal"] == 12
    assert plan["sequence"]["constraint_change"] == "REVIEWED_NON_TIMEOUT_FIX"
    assert plan["sequence"]["predecessor"]["plan_digest"] == (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_11_PLAN_DIGEST
    )
    assert plan["sequence"]["reviewed_fix"]["fix_kind"] == "CODE"
    assert plan["sequence"]["predecessor_causal_report"]["local_cause"] == (
        "EVIDENCE_UNRESOLVED_CONTRADICTORY_FACT_ATTRIBUTION"
    )
    assert contract.sequence_ordinal == 12
    assert contract.constraint_change == "REVIEWED_NON_TIMEOUT_FIX"
    assert (
        contract.predecessor_plan_digest
        == issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_11_PLAN_DIGEST
    )
    assert qualification["qualification"] == (
        "COMBINED_TEMPORAL_FACT_ATTRIBUTION_PROVIDER_FREE_READY"
    )
    assert qualification["provider_calls"] == 0


def test_checked_issue_790_step_thirteen_executable_plan_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_issue_790_plan(
        root / "docs/operations/2026-08-28-issue-790-success-sequence-step-13.json"
    )
    contract = issue_790_contract_module.issue_790_approved_plan_contract(
        plan["canonical_digest"]
    )
    qualification = json.loads(
        (
            root
            / "docs/operations/2026-08-28-issue-790-step-13-output-limit-qualification.json"
        ).read_text(encoding="utf-8")
    )

    assert plan["canonical_digest"] == (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_13_PLAN_DIGEST
    )
    assert plan["sequence"]["sequence_ordinal"] == 13
    assert plan["sequence"]["constraint_change"] == "REVIEWED_NON_TIMEOUT_FIX"
    assert plan["sequence"]["predecessor"]["plan_digest"] == (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_12_PLAN_DIGEST
    )
    assert plan["sequence"]["reviewed_fix"]["fix_kind"] == "CODE"
    assert plan["sequence"]["predecessor_causal_report"]["local_cause"] == (
        "EXTRACTION_MAX_RESPONSE_TOKENS_BELOW_CALL_SHAPE"
    )
    assert contract.sequence_ordinal == 13
    assert contract.constraint_change == "REVIEWED_NON_TIMEOUT_FIX"
    assert (
        contract.predecessor_plan_digest
        == issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_12_PLAN_DIGEST
    )
    assert qualification["qualification"] == (
        "EXTRACTION_RESPONSE_BUDGET_CALL_SHAPE_ALIGNED_PROVIDER_FREE_READY"
    )
    assert qualification["provider_calls"] == 0


def test_checked_issue_790_step_fourteen_executable_plan_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_issue_790_plan(
        root / "docs/operations/2026-08-28-issue-790-success-sequence-step-14.json"
    )
    contract = issue_790_contract_module.issue_790_approved_plan_contract(
        plan["canonical_digest"]
    )
    qualification = json.loads(
        (
            root
            / "docs/operations/2026-08-28-issue-790-step-14-endpoint-prefer-empty-qualification.json"
        ).read_text(encoding="utf-8")
    )

    assert plan["canonical_digest"] == (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_14_PLAN_DIGEST
    )
    assert plan["sequence"]["sequence_ordinal"] == 14
    assert plan["sequence"]["constraint_change"] == "REVIEWED_NON_TIMEOUT_FIX"
    assert plan["sequence"]["predecessor"]["plan_digest"] == (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_13_PLAN_DIGEST
    )
    assert plan["sequence"]["reviewed_fix"]["fix_kind"] == "CODE"
    assert plan["sequence"]["predecessor_causal_report"]["local_cause"] == (
        "EVIDENCE_UNRESOLVED_ENDPOINT_NAMES_ABSENT_FROM_FACT"
    )
    assert contract.sequence_ordinal == 14
    assert contract.constraint_change == "REVIEWED_NON_TIMEOUT_FIX"
    assert (
        contract.predecessor_plan_digest
        == issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_13_PLAN_DIGEST
    )
    assert qualification["qualification"] == (
        "ENDPOINT_IN_FACT_PREFER_EMPTY_PROVIDER_FREE_READY"
    )
    assert qualification["provider_calls"] == 0


def test_checked_issue_790_step_fifteen_executable_plan_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_issue_790_plan(
        root / "docs/operations/2026-08-28-issue-790-success-sequence-step-15.json"
    )
    contract = issue_790_contract_module.issue_790_approved_plan_contract(
        plan["canonical_digest"]
    )
    qualification = json.loads(
        (
            root
            / "docs/operations/2026-08-28-issue-790-step-15-temporal-qualification.json"
        ).read_text(encoding="utf-8")
    )

    assert plan["canonical_digest"] == (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_15_PLAN_DIGEST
    )
    assert plan["sequence"]["sequence_ordinal"] == 15
    assert plan["sequence"]["constraint_change"] == "REVIEWED_NON_TIMEOUT_FIX"
    assert plan["sequence"]["predecessor"]["plan_digest"] == (
        issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_14_PLAN_DIGEST
    )
    assert plan["sequence"]["reviewed_fix"]["fix_kind"] == "CODE"
    assert plan["sequence"]["predecessor_causal_report"]["local_cause"] == (
        "TEMPORAL_INVALID_VALID_AT_NOT_GROUNDED_IN_EVIDENCE"
    )
    assert contract.sequence_ordinal == 15
    assert (
        contract.predecessor_plan_digest
        == issue_790_contract_module.ISSUE_790_SUCCESS_SEQUENCE_STEP_14_PLAN_DIGEST
    )
    assert qualification["qualification"] == (
        "TEMPORAL_REFERENCE_TIME_STUFFING_PROVIDER_FREE_READY"
    )
    assert qualification["provider_calls"] == 0


def test_checked_issue_790_compatibility_floor_provider_free_qualification_receipt() -> (
    None
):
    root = Path(__file__).resolve().parents[2]
    qualification = json.loads(
        (
            root
            / "docs/operations/2026-08-27-issue-790-compatibility-floor-provider-free-qualification.json"
        ).read_text(encoding="utf-8")
    )
    draft = json.loads(
        (
            root
            / "docs/operations/2026-08-27-issue-790-success-sequence-step-8-draft.json"
        ).read_text(encoding="utf-8")
    )
    unsigned = {
        key: value
        for key, value in qualification.items()
        if key != "canonical_digest"
    }

    assert qualification["schema_version"] == (
        "newsroom.issue-790.provider-free-sdk-qualification.v2"
    )
    assert qualification["canonical_digest"] == _digest(unsigned)
    assert qualification["provider_calls"] == 0
    assert qualification["public_effects"] == 0
    assert qualification["model_catalogue_queries"] == 0
    assert qualification["live_canary_authorised"] is False
    assert qualification["qualification"] == "COMPATIBILITY_FLOOR_PROVIDER_FREE_READY"
    assert draft["intended_sequence"]["call_shape_policy_digest"] == (
        qualification["provider_free_checks"]["call_shape_policy_digest"]
    )


def test_checked_issue_790_provider_free_sdk_qualification_receipt() -> None:
    root = Path(__file__).resolve().parents[2]
    qualification = json.loads(
        (
            root
            / "docs/operations/2026-08-27-issue-790-provider-free-sdk-qualification.json"
        ).read_text(encoding="utf-8")
    )
    draft = json.loads(
        (
            root
            / "docs/operations/2026-08-27-issue-790-success-sequence-step-7-draft.json"
        ).read_text(encoding="utf-8")
    )
    unsigned = {
        key: value
        for key, value in qualification.items()
        if key != "canonical_digest"
    }

    assert qualification["schema_version"] == (
        "newsroom.issue-790.provider-free-sdk-qualification.v1"
    )
    assert qualification["canonical_digest"] == _digest(unsigned)
    assert qualification["provider_calls"] == 0
    assert qualification["public_effects"] == 0
    assert qualification["live_canary_authorised"] is False
    assert qualification["qualification"] == "EXACT_MAIN_PROVIDER_FREE_SDK_READY"
    assert qualification["prerequisite_merge"]["pull_request"] == 810
    assert qualification["exact_main_revision"] == (
        qualification["prerequisite_merge"]["merge_sha"]
    )
    assert draft["intended_sequence"]["call_shape_policy_digest"] == (
        qualification["provider_free_checks"]["call_shape_policy_digest"]
    )
    assert set(qualification["focused_tests"]) == {
        "newsroom/tests/test_graphiti_cursor_sdk_transport.py",
        "newsroom/tests/test_model_usage_receipts.py",
    }


def test_issue_790_plan_rejects_malformed_sha256_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(
        (
            root
            / "docs/operations/2026-08-26-issue-790-success-sequence-step-3.json"
        ).read_text(encoding="utf-8")
    )
    plan["target"]["invocation_id"] += "a"
    plan["canonical_digest"] = _digest(
        {key: value for key, value in plan.items() if key != "canonical_digest"}
    )

    with pytest.raises(Issue790DispositionError, match="invocation_id differs"):
        validate_issue_790_plan(plan)


def test_issue_790_exact_main_evidence_binds_cursor_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binding = (
        "newsroom.graphiti_adapter.cursor_transport",
        "newsroom/graphiti_adapter/cursor_transport.py",
    )
    assert binding in issue_790_operation._RUNNING_CODE_MODULES
    wrong_transport = tmp_path / "cursor_transport.py"
    wrong_transport.write_text("# stale transport\n", encoding="utf-8")
    monkeypatch.setattr(issue_790_operation, "_RUNNING_CODE_MODULES", (binding,))
    monkeypatch.setitem(
        issue_790_operation._RUNNING_CODE_PATHS,
        binding[0],
        str(wrong_transport),
    )

    with pytest.raises(
        Issue790DispositionError,
        match="executing operation code is outside exact main",
    ):
        issue_790_operation._running_code_evidence(
            root=Path(__file__).resolve().parents[2],
            git="git",
            revision="0" * 40,
        )


def test_issue_790_plan_digest_and_bounds_fail_closed(tmp_path: Path) -> None:
    service = _service(tmp_path)
    policy, allocation, terminal = _open_unreported_graphiti_subscription_leaf(
        service
    )
    plan = _issue_790_plan(policy, allocation, terminal)

    with pytest.raises(Issue790DispositionError, match="plan digest differs"):
        validate_issue_790_plan(
            {
                **plan,
                "target": {
                    **plan["target"],  # type: ignore[dict-item]
                    "expected_conservative_total_tokens": 1,
                },
            }
        )


def test_issue_790_unapproved_structural_plan_is_rejected_before_backup(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    policy, allocation, terminal = _open_unreported_graphiti_subscription_leaf(
        service
    )
    source = tmp_path / "unpublished.sqlite3"
    scratch = tmp_path / "unapproved-dry-run.sqlite3"
    plan = _issue_790_plan(policy, allocation, terminal)

    assert validate_issue_790_plan(plan) == plan
    with pytest.raises(
        Issue790DispositionError,
        match="approved plan identity differs",
    ):
        dry_run_issue_790_plan(
            source_store=source,
            scratch_store=scratch,
            plan=plan,
            observed_at=T0 + timedelta(seconds=10),
        )

    assert scratch.exists() is False


def test_issue_790_evidence_paths_reject_aliases_and_existing_receipt(
    tmp_path: Path,
) -> None:
    store = tmp_path / "unpublished.sqlite3"
    store.write_bytes(b"fixture")
    hardlink = tmp_path / "store-hardlink.sqlite3"
    hardlink.hardlink_to(store)

    with pytest.raises(Issue790DispositionError, match="paths alias"):
        assert_issue_790_paths_disjoint(store, hardlink)

    receipt = tmp_path / "receipt.json"
    receipt.write_text("retained\n", encoding="utf-8")
    with pytest.raises(Issue790DispositionError, match="already exists"):
        write_issue_790_receipt(receipt, {"receipt_digest": _digest({"a": 1})})
    assert receipt.read_text(encoding="utf-8") == "retained\n"


def test_issue_790_dry_run_mutates_only_isolated_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    policy, allocation, terminal = _open_unreported_graphiti_subscription_leaf(
        service
    )
    service.open_route_circuit(
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        reason="SYSTEMIC_TRANSPORT",
        invocation_id=allocation.invocation_id,
        recorded_at=T0 + timedelta(seconds=3),
    )
    source = tmp_path / "unpublished.sqlite3"
    scratch = tmp_path / "dry-run.sqlite3"
    plan = _issue_790_plan(policy, allocation, terminal)
    _bind_issue_790_fixture_contract(
        monkeypatch,
        allocation=allocation,
        terminal=terminal,
        plan_digest=str(plan["canonical_digest"]),
    )
    _seed_issue_790_retry_events(source)

    receipt = dry_run_issue_790_plan(
        source_store=source,
        scratch_store=scratch,
        plan=plan,
        observed_at=T0 + timedelta(seconds=10),
    )

    assert receipt["mode"] == "dry-run"
    assert receipt["source_mutated"] is False
    assert receipt["retry_performed"] is False
    assert receipt["canary_performed"] is False
    assert receipt["receipt_digest"] == _digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    assert service_row_count(
        source, "model_usage_conservative_dispositions"
    ) == 0
    assert service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)["state"] == "OPEN"
    copied = ModelUsageService(str(scratch))
    assert service_row_count(
        scratch, "model_usage_conservative_dispositions"
    ) == 1
    assert copied.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)["state"] == "CLOSED"
    assert service_row_count(scratch, "model_usage_reconciliations") == 0
    assert service_row_count(scratch, "model_provider_telemetry") == 0


def test_issue_790_apply_retains_pre_operation_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    policy, allocation, terminal = _open_unreported_graphiti_subscription_leaf(
        service
    )
    service.open_route_circuit(
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        reason="SYSTEMIC_TRANSPORT",
        invocation_id=allocation.invocation_id,
        recorded_at=T0 + timedelta(seconds=3),
    )
    store = tmp_path / "unpublished.sqlite3"
    backup = tmp_path / "unpublished.pre-790.sqlite3"
    plan = _issue_790_plan(policy, allocation, terminal)
    _bind_issue_790_fixture_contract(
        monkeypatch,
        allocation=allocation,
        terminal=terminal,
        plan_digest=str(plan["canonical_digest"]),
    )
    _seed_issue_790_retry_events(store)
    _patch_issue_790_live_evidence(
        monkeypatch,
        store=store,
        observed_at=T0 + timedelta(seconds=10),
    )

    receipt = apply_issue_790_plan(
        store=store,
        backup_path=backup,
        plan=plan,
        observed_at=T0 + timedelta(seconds=10),
        repository_root=tmp_path,
    )

    assert receipt["mode"] == "apply"
    assert backup.is_file()
    assert receipt["pre_operation_snapshot_digest"] == (
        "sha256:" + hashlib.sha256(backup.read_bytes()).hexdigest()
    )
    assert receipt["pre_operation_snapshot_retained"] is True
    assert service_row_count(
        backup, "model_usage_conservative_dispositions"
    ) == 0
    backup_connection = sqlite3.connect(
        f"{backup.absolute().as_uri()}?mode=ro", uri=True
    )
    backup_state = backup_connection.execute(
        "SELECT state FROM model_usage_route_circuit_events "
        "WHERE route=? ORDER BY recorded_at DESC,rowid DESC LIMIT 1",
        (GRAPHITI_CHAT_PRIMARY_ROUTE,),
    ).fetchone()
    backup_connection.close()
    assert backup_state == ("OPEN",)
    assert service_row_count(store, "model_usage_conservative_dispositions") == 1
    assert service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)["state"] == "CLOSED"
    assert len(receipt["retry_exclusions"]) == 2
    resumed_worker_queue = GraphitiEventQueue(
        str(store),
        clock=lambda: datetime(2026, 8, 27, tzinfo=UTC),
    )
    assert resumed_worker_queue.claim(
        owner_id="resumed-persistent-worker",
        lease_for=timedelta(minutes=1),
    ) is None
    for retained_failure in RETRY_FORBIDDEN_EVENTS:
        with pytest.raises(ValueError, match="retry-excluded"):
            resumed_worker_queue.claim(
                owner_id="resumed-persistent-worker",
                lease_for=timedelta(minutes=1),
                event_id=str(retained_failure["event_id"]),
            )


def test_issue_790_bounded_canary_is_single_use_and_failed_attempt_is_inert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    policy, allocation, terminal = _open_unreported_graphiti_subscription_leaf(
        service
    )
    _bind_issue_790_fixture_contract(
        monkeypatch,
        allocation=allocation,
        terminal=terminal,
    )
    approved_at = T0 + timedelta(seconds=5)
    authority, authority_digest = _conservative_disposition_authority(
        allocation,
        terminal,
        approved_at=approved_at,
    )
    disposition = service.disposition_unreported_subscription_usage(
        invocation_id=allocation.invocation_id,
        expected_terminal_digest=terminal.terminal_digest,
        expected_allocation_digest=allocation.canonical_digest,
        approved_by=str(authority["approved_by"]),
        approval_reference=str(authority["approval_reference"]),
        approved_at=approved_at,
        approved_plan_digest=FIXTURE_790_PLAN_DIGEST,
        authority_digest=authority_digest,
        observed_at=T0 + timedelta(seconds=10),
    )
    route = service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)
    service.release_route_circuit(
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        release_kind="AUTHORISED_OPERATOR_RESET",
        bound_failure_reason=str(route["reason"]),
        evidence_digest=str(disposition["disposition_digest"]),
        recorded_at=T0 + timedelta(seconds=11),
    )
    store = tmp_path / "unpublished.sqlite3"
    event_id, _ingest_id = _seed_fresh_issue_790_event(
        store,
        retain_unit_refs=False,
    )
    owner_id = "issue-790-canary:fixture"
    canary = Issue790CanaryRepository(str(store))
    preflight = _issue_790_canary_preflight(
        store,
        event_id=event_id,
        ledger_seq=2001,
        evaluated_at=T0 + timedelta(seconds=19),
        resolved_units=[_fixture_issue_790_unit_ref(2001)],
    )

    with pytest.raises(Issue790CanaryIntegrityError, match="retained failure"):
        canary.consume(
            approved_plan_digest=FIXTURE_790_PLAN_DIGEST,
            disposition_digest=str(disposition["disposition_digest"]),
            event_id=event_id,
            ledger_seq=1932,
            owner_id=owner_id,
            preflight_evidence=preflight,
            consumed_at=T0 + timedelta(seconds=20),
        )

    consumption = canary.consume(
        approved_plan_digest=FIXTURE_790_PLAN_DIGEST,
        disposition_digest=str(disposition["disposition_digest"]),
        event_id=event_id,
        ledger_seq=2001,
        owner_id=owner_id,
        preflight_evidence=preflight,
        consumed_at=T0 + timedelta(seconds=20),
    )
    connection = sqlite3.connect(store)
    original_record_json = str(
        connection.execute(
            "SELECT record_json FROM issue_790_bounded_canary_consumptions "
            "WHERE consumption_digest=?",
            (str(consumption["consumption_digest"]),),
        ).fetchone()[0]
    )
    forged = dict(consumption)
    forged["owner_id"] = "issue-790-canary:forged-owner"
    forged_without_digest = dict(forged)
    forged_without_digest.pop("consumption_digest")
    forged["consumption_digest"] = _digest(forged_without_digest)
    connection.execute(
        "UPDATE issue_790_bounded_canary_consumptions SET record_json=? "
        "WHERE consumption_digest=?",
        (
            json.dumps(forged, sort_keys=True, separators=(",", ":")),
            str(consumption["consumption_digest"]),
        ),
    )
    connection.commit()
    connection.close()
    with pytest.raises(Issue790CanaryIntegrityError, match="SQL identity differs"):
        canary.preflight_for_consumption(
            consumption_digest=str(consumption["consumption_digest"]),
            event_id=event_id,
            owner_id=owner_id,
        )
    forged_queue = GraphitiEventQueue(
        str(store),
        clock=lambda: T0 + timedelta(seconds=20),
    )
    with pytest.raises(Issue790CanaryIntegrityError, match="SQL identity differs"):
        forged_queue.claim(
            owner_id=owner_id,
            lease_for=timedelta(seconds=30),
            event_id=event_id,
            require_fresh=True,
            canary_consumption_digest=str(consumption["consumption_digest"]),
        )
    connection = sqlite3.connect(store)
    connection.execute(
        "UPDATE issue_790_bounded_canary_consumptions SET record_json=? "
        "WHERE consumption_digest=?",
        (original_record_json, str(consumption["consumption_digest"])),
    )
    connection.commit()
    connection.close()
    with pytest.raises(Issue790CanaryIntegrityError, match="already consumed"):
        canary.consume(
            approved_plan_digest=FIXTURE_790_PLAN_DIGEST,
            disposition_digest=str(disposition["disposition_digest"]),
            event_id=event_id,
            ledger_seq=2001,
            owner_id=owner_id,
            preflight_evidence=preflight,
            consumed_at=T0 + timedelta(seconds=21),
        )

    queue = GraphitiEventQueue(
        str(store),
        clock=lambda: T0 + timedelta(seconds=20),
    )
    assert queue.claim(
        owner_id="generic-worker",
        lease_for=timedelta(seconds=30),
    ) is None
    with pytest.raises(ValueError, match="claim authority is required"):
        queue.claim(
            owner_id=owner_id,
            lease_for=timedelta(seconds=30),
            event_id=event_id,
            require_fresh=True,
        )
    claimed = queue.claim(
        owner_id=owner_id,
        lease_for=timedelta(seconds=30),
        event_id=event_id,
        require_fresh=True,
        canary_consumption_digest=str(consumption["consumption_digest"]),
    )
    assert claimed is not None
    assert claimed.event_id == event_id

    connection = connect_unpublished_store(str(store))
    connection.execute(
        "UPDATE unpublished_graphiti_revision_events SET state='DEAD_LETTER',"
        "attempt_count=1,last_failure_code='PRODUCER_INTERNAL_ERROR',"
        "provider_dispatched=1,claim_owner=NULL,claim_expires_at=NULL "
        "WHERE event_id=?",
        (event_id,),
    )
    connection.commit()
    connection.close()
    outcome = canary.finalise_without_dispatch(
        consumption_digest=str(consumption["consumption_digest"]),
        event_id=event_id,
        ledger_seq=2001,
        owner_id=owner_id,
        completed_at=T0 + timedelta(seconds=30),
    )
    replay = canary.finalise_without_dispatch(
        consumption_digest=str(consumption["consumption_digest"]),
        event_id=event_id,
        ledger_seq=2001,
        owner_id=owner_id,
        completed_at=T0 + timedelta(seconds=31),
    )

    connection = sqlite3.connect(store)
    retained = connection.execute(
        "SELECT state,attempt_count,last_failure_code,provider_dispatched "
        "FROM unpublished_graphiti_revision_events WHERE event_id=?",
        (event_id,),
    ).fetchone()
    connection.close()
    assert replay == outcome
    assert outcome["state_after_seal"] == "CONFIGURATION_HELD"
    assert outcome["completion_mode"] == "ZERO_IO_RECOVERY"
    assert outcome["process_result"]["state"] == "DEAD_LETTER"
    assert outcome["event_provider_dispatched_before_seal"] is True
    assert outcome["provider_dispatched"] is False
    assert outcome["retry_authorised"] is False
    assert retained == (
        "CONFIGURATION_HELD",
        1,
        "BOUNDED_CANARY_AUTHORITY_EXHAUSTED:PRODUCER_INTERNAL_ERROR",
        0,
    )
    assert service_row_count(store, "issue_790_bounded_canary_consumptions") == 1
    assert service_row_count(store, "issue_790_bounded_canary_outcomes") == 1


def test_issue_790_historical_same_plan_rows_are_preserved_and_authority_stays_spent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, plan, disposition, repository = _released_issue_790_canary(
        tmp_path, monkeypatch
    )
    plan_digest = str(plan["canonical_digest"])
    disposition_digest = str(disposition["disposition_digest"])
    _seed_fresh_issue_790_event(
        store,
        ledger_seq=CANDIDATE_LEDGER_SEQ,
        event_id=CANDIDATE_EVENT_ID,
    )
    spent = _consume_plan_event(
        repository,
        store,
        plan_digest=plan_digest,
        disposition_digest=disposition_digest,
        event_id=CANDIDATE_EVENT_ID,
        ledger_seq=CANDIDATE_LEDGER_SEQ,
        owner_id="issue-790-canary:spent-13665",
        consumed_at=T0 + timedelta(seconds=20),
    )
    repository.complete(
        consumption_digest=str(spent["consumption_digest"]),
        event_id=CANDIDATE_EVENT_ID,
        ledger_seq=CANDIDATE_LEDGER_SEQ,
        owner_id="issue-790-canary:spent-13665",
        process_result=None,
        exception_code="AMBIGUOUS_EFFECT",
        completed_at=T0 + timedelta(seconds=30),
    )
    _seed_fresh_issue_790_event(
        store,
        ledger_seq=SUCCESSOR_LEDGER_SEQ,
        event_id=SUCCESSOR_EVENT_ID,
    )

    connection = sqlite3.connect(store)
    connection.execute(
        "CREATE UNIQUE INDEX issue_790_legacy_plan_consumption_unique ON "
        "issue_790_bounded_canary_consumptions(approved_plan_digest)"
    )
    connection.commit()
    connection.close()
    writer = Issue790CanaryRepository.open_existing(str(store))
    assert ("approved_plan_digest",) in _unique_consumption_columns(store)
    with pytest.raises(Issue790CanaryIntegrityError, match="already consumed"):
        _consume_plan_event(
            writer,
            store,
            plan_digest=plan_digest,
            disposition_digest=disposition_digest,
            event_id=SUCCESSOR_EVENT_ID,
            ledger_seq=SUCCESSOR_LEDGER_SEQ,
            owner_id="issue-790-canary:successor-13671",
            consumed_at=T0 + timedelta(seconds=40),
        )

    connection = sqlite3.connect(store)
    connection.execute("DROP INDEX issue_790_legacy_plan_consumption_unique")
    connection.execute(
        "INSERT INTO issue_790_bounded_canary_consumptions VALUES(?,?,?,?,?,?,?,?)",
        (
            "sha256:" + "ab" * 32,
            plan_digest,
            disposition_digest,
            SUCCESSOR_EVENT_ID,
            SUCCESSOR_LEDGER_SEQ,
            "issue-790-canary:historical-successor",
            (T0 + timedelta(seconds=41)).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "{}",
        ),
    )
    connection.commit()
    connection.close()

    reopened = Issue790CanaryRepository.open_existing(str(store))
    assert service_row_count(store, "issue_790_bounded_canary_consumptions") == 2
    future_event = _digest({"future-same-plan-event": 1})
    _seed_fresh_issue_790_event(store, ledger_seq=13702, event_id=future_event)
    with pytest.raises(Issue790CanaryIntegrityError, match="already consumed"):
        _consume_plan_event(
            reopened,
            store,
            plan_digest=plan_digest,
            disposition_digest=disposition_digest,
            event_id=future_event,
            ledger_seq=13702,
            owner_id="issue-790-canary:future-successor",
            consumed_at=T0 + timedelta(seconds=42),
        )
    assert service_row_count(store, "issue_790_bounded_canary_consumptions") == 2

def test_issue_790_successor_plan_consumes_a_new_event_after_failed_predecessor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    _policy_value, first_allocation, first_terminal = (
        _open_unreported_graphiti_subscription_leaf(
            service,
            cycle_id="graphiti-cycle-predecessor",
            request="graphiti-predecessor-request",
        )
    )
    _bind_issue_790_fixture_contract(
        monkeypatch,
        allocation=first_allocation,
        terminal=first_terminal,
    )
    first_approved_at = T0 + timedelta(seconds=5)
    first_authority, first_authority_digest = _conservative_disposition_authority(
        first_allocation,
        first_terminal,
        approved_at=first_approved_at,
    )
    first_disposition = service.disposition_unreported_subscription_usage(
        invocation_id=first_allocation.invocation_id,
        expected_terminal_digest=first_terminal.terminal_digest,
        expected_allocation_digest=first_allocation.canonical_digest,
        approved_by=str(first_authority["approved_by"]),
        approval_reference=str(first_authority["approval_reference"]),
        approved_at=first_approved_at,
        approved_plan_digest=FIXTURE_790_PLAN_DIGEST,
        authority_digest=first_authority_digest,
        observed_at=T0 + timedelta(seconds=10),
    )
    first_route = service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)
    service.release_route_circuit(
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        release_kind="AUTHORISED_OPERATOR_RESET",
        bound_failure_reason=str(first_route["reason"]),
        evidence_digest=str(first_disposition["disposition_digest"]),
        recorded_at=T0 + timedelta(seconds=11),
    )

    store = tmp_path / "unpublished.sqlite3"
    _seed_issue_790_retry_events(store)
    first_event_id, _first_ingest = _seed_fresh_issue_790_event(
        store,
        ledger_seq=2004,
        retain_unit_refs=False,
    )
    repository = Issue790CanaryRepository(str(store))
    repository.retain_retry_exclusions(
        approved_plan_digest=FIXTURE_790_PLAN_DIGEST,
        disposition_digest=str(first_disposition["disposition_digest"]),
        events=RETRY_FORBIDDEN_EVENTS,
        excluded_at=T0 + timedelta(seconds=12),
    )
    first_owner = "issue-790-canary:predecessor"
    first_consumption = repository.consume(
        approved_plan_digest=FIXTURE_790_PLAN_DIGEST,
        disposition_digest=str(first_disposition["disposition_digest"]),
        event_id=first_event_id,
        ledger_seq=2004,
        owner_id=first_owner,
        preflight_evidence=_issue_790_canary_preflight(
            store,
            event_id=first_event_id,
            ledger_seq=2004,
            evaluated_at=T0 + timedelta(seconds=19),
            resolved_units=[_fixture_issue_790_unit_ref(2004)],
        ),
        consumed_at=T0 + timedelta(seconds=20),
    )
    first_outcome = repository.complete(
        consumption_digest=str(first_consumption["consumption_digest"]),
        event_id=first_event_id,
        ledger_seq=2004,
        owner_id=first_owner,
        process_result=None,
        exception_code="CliTransportTimeout",
        completed_at=T0 + timedelta(seconds=30),
    )
    assert first_outcome["state_after_seal"] == "CONFIGURATION_HELD"

    second_policy, second_allocation, second_terminal = (
        _open_unreported_graphiti_subscription_leaf(
            service,
            cycle_id="graphiti-cycle-successor",
            request="graphiti-successor-request",
            outcome="TIMEOUT",
            elapsed_ms=80_001,
        )
    )
    second_approved_at = T0 + timedelta(seconds=90)
    successor_plan = _issue_790_successor_plan(
        second_policy,
        second_allocation,
        second_terminal,
        approved_at=second_approved_at,
        predecessor_consumption=first_consumption,
        predecessor_outcome=first_outcome,
    )
    successor_plan_digest = str(successor_plan["canonical_digest"])
    successor_contract = _bind_issue_790_successor_fixture_contract(
        monkeypatch,
        allocation=second_allocation,
        terminal=second_terminal,
        plan=successor_plan,
        approved_at=second_approved_at,
    )
    monkeypatch.setattr(
        issue_790_operation,
        "load_checked_graphiti_call_shape_policy",
        lambda: SimpleNamespace(
            canonical_digest=(
                "sha256:7e6bd15613cefda0820a1d339c8790f0185946aade1622dadbb8c468f558bb18"
            ),
            version="issue-790-v9",
            qualified_routes=(
                SimpleNamespace(
                    leaf_class=issue_790_operation.GraphitiLeafClass.PRIMARY,
                    provider="cursor-agent-cli",
                    route=GRAPHITI_CHAT_PRIMARY_ROUTE,
                    model="composer-2.5",
                    max_total_tokens=second_policy.max_total_tokens,
                    command_flags=("CONTROLLER_TIMEOUT_MS=160000",),
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        issue_790_operation,
        "_issue_790_fixed_constraints_digest",
        lambda _policy: str(
            dict(successor_plan["sequence"])["fixed_constraints_digest"]
        ),
    )
    service.open_route_circuit(
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        reason="TIMEOUT",
        invocation_id=second_allocation.invocation_id,
        recorded_at=T0 + timedelta(seconds=85),
    )
    second_authority, second_authority_digest = _conservative_disposition_authority(
        second_allocation,
        second_terminal,
        approved_at=second_approved_at,
        approved_plan_digest=successor_plan_digest,
    )
    second_disposition = service.disposition_unreported_subscription_usage(
        invocation_id=second_allocation.invocation_id,
        expected_terminal_digest=second_terminal.terminal_digest,
        expected_allocation_digest=second_allocation.canonical_digest,
        approved_by=str(second_authority["approved_by"]),
        approval_reference=str(second_authority["approval_reference"]),
        approved_at=second_approved_at,
        approved_plan_digest=successor_plan_digest,
        authority_digest=second_authority_digest,
        observed_at=T0 + timedelta(seconds=95),
    )
    second_route = service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)
    service.release_route_circuit(
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        release_kind="AUTHORISED_OPERATOR_RESET",
        bound_failure_reason=str(second_route["reason"]),
        evidence_digest=str(second_disposition["disposition_digest"]),
        recorded_at=T0 + timedelta(seconds=96),
    )

    dry_run_receipt = dry_run_issue_790_plan(
        source_store=store,
        scratch_store=tmp_path / "successor-dry-run.sqlite3",
        plan=successor_plan,
        observed_at=T0 + timedelta(seconds=100),
    )
    assert dry_run_receipt["schema_version"] == (
        "newsroom.issue-790.iterative-disposition-receipt.v2"
    )
    assert dry_run_receipt["plan_digest"] == successor_plan_digest
    assert {
        record["approved_plan_digest"]
        for record in dry_run_receipt["retry_exclusions"]
    } == {FIXTURE_790_PLAN_DIGEST}

    second_event_id, _second_ingest = _seed_fresh_issue_790_event(
        store,
        ledger_seq=2005,
        retain_unit_refs=False,
    )
    iterative_preflight = _issue_790_canary_preflight(
        store,
        event_id=second_event_id,
        ledger_seq=2005,
        evaluated_at=T0 + timedelta(seconds=104),
        resolved_units=[_fixture_issue_790_unit_ref(2005)],
        approved_plan_digest=successor_plan_digest,
        fixed_constraints_digest=str(
            dict(successor_plan["sequence"])["fixed_constraints_digest"]
        ),
    )
    tampered_preflight = dict(iterative_preflight)
    tampered_preflight.pop("evidence_digest")
    tampered_preflight["fallback_mode"] = "ELIGIBLE_AFTER_PRIMARY_FAILURE"
    tampered_preflight["evidence_digest"] = _digest(tampered_preflight)
    with pytest.raises(
        Issue790CanaryIntegrityError,
        match="iterative preflight differs",
    ):
        repository.consume(
            approved_plan_digest=successor_plan_digest,
            disposition_digest=str(second_disposition["disposition_digest"]),
            event_id=second_event_id,
            ledger_seq=2005,
            owner_id="issue-790-canary:tampered-successor",
            preflight_evidence=tampered_preflight,
            consumed_at=T0 + timedelta(seconds=105),
        )
    second_backup = tmp_path / "successor-pre-operation.sqlite3"
    second_backup_digest = issue_790_operation._sqlite_backup(store, second_backup)
    second_consumption = repository.consume(
        approved_plan_digest=successor_plan_digest,
        disposition_digest=str(second_disposition["disposition_digest"]),
        event_id=second_event_id,
        ledger_seq=2005,
        owner_id="issue-790-canary:successor",
        preflight_evidence=iterative_preflight,
        consumed_at=T0 + timedelta(seconds=105),
    )

    assert second_consumption["approved_plan_digest"] == successor_plan_digest
    assert second_consumption["event_id"] == second_event_id
    assert second_event_id != first_event_id
    assert service_row_count(store, "issue_790_bounded_canary_consumptions") == 2
    connection = sqlite3.connect(store)
    excluded = issue_790_canary_module.graphiti_excluded_event_ids(connection)
    connection.close()
    assert {first_event_id, second_event_id}.issubset(excluded)

    connection = sqlite3.connect(store)
    connection.execute(
        "UPDATE unpublished_graphiti_revision_events SET state='RETRY_HELD',"
        "attempt_count=1,last_failure_code='PRODUCER_INTERNAL_ERROR',"
        "provider_dispatched=1 WHERE event_id=? AND ledger_seq=?",
        (second_event_id, 2005),
    )
    connection.commit()
    connection.close()
    third_policy, third_allocation, third_terminal = (
        _open_unreported_graphiti_subscription_leaf(
            service,
            cycle_id=second_event_id,
            request="graphiti-second-successor-request",
            outcome="TIMEOUT",
            elapsed_ms=160_001,
        )
    )
    second_causal_report = _issue_790_controller_timeout_report_fixture(
        third_terminal,
        configured_timeout_ms=160_000,
    )
    service.open_route_circuit(
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        reason="TIMEOUT",
        invocation_id=third_allocation.invocation_id,
        recorded_at=T0 + timedelta(seconds=275),
    )
    connection = connect_unpublished_store(str(store))
    timeout_receipt = {
        "ingest_id": f"ingest-{second_event_id}",
        "attempt_number": 1,
        "outcome": "FAILED",
        "timeout_diagnostics": [second_causal_report["diagnostic"]],
        "chat_invocations": [
            {
                "provider": "cursor-agent-cli",
                "transport_diagnostic": second_causal_report["diagnostic"],
            }
        ],
        "receipt_digest": "",
    }
    insert_graphiti_attempt_receipt(
        connection,
        ingest_id=f"ingest-{second_event_id}",
        attempt_number=1,
        outcome="FAILED",
        receipt=timeout_receipt,
    )
    connection.commit()
    connection.close()
    proving = tmp_path / "successor-proving.sqlite3"
    proving_connection = sqlite3.connect(proving)
    proving_connection.execute("CREATE TABLE fixture(value INTEGER)")
    proving_connection.commit()
    proving_connection.close()
    recovery_observed_at = T0 + timedelta(seconds=276)
    _patch_issue_790_live_evidence(
        monkeypatch,
        store=store,
        observed_at=recovery_observed_at,
    )
    route_before_recovery = service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)
    recovery_receipt = run_issue_790_canary(
        store=store,
        proving_store=proving,
        backup_path=second_backup,
        plan=successor_plan,
        observed_at=recovery_observed_at,
        repository_root=tmp_path,
        event_id=second_event_id,
        ledger_seq=2005,
        disposition_digest=str(second_disposition["disposition_digest"]),
        recover_interrupted=True,
        expected_backup_digest=second_backup_digest,
    )
    second_outcome = recovery_receipt["outcome"]
    retained_second_causal_report = second_outcome["causal_report"]
    assert retained_second_causal_report["diagnostic"] == (
        second_causal_report["diagnostic"]
    )
    second_causal_report = retained_second_causal_report
    assert second_outcome["result_class"] == "CONTROLLER_TIMEOUT_NON_SUCCESS"
    assert recovery_receipt["resumed_zero_io_finalisation"] is True
    assert recovery_receipt["provider_dispatch_attempted_this_run"] is False
    assert recovery_receipt["route_before"] == route_before_recovery
    assert recovery_receipt["route_after"] == route_before_recovery

    step_two_approved_at = T0 + timedelta(seconds=280)
    step_two_call_shape_digest = _digest({"call-shape": "issue-790-step-two"})
    step_two_plan = _issue_790_successor_plan(
        third_policy,
        third_allocation,
        third_terminal,
        approved_at=step_two_approved_at,
        predecessor_consumption=second_consumption,
        predecessor_outcome=second_outcome,
        sequence_ordinal=2,
        controller_timeout_ms=170_000,
        extraction_timeout_ms=190_000,
        predecessor_controller_timeout_ms=160_000,
        root_plan_digest=FIXTURE_790_PLAN_DIGEST,
        fixed_constraints_digest=str(
            dict(successor_plan["sequence"])["fixed_constraints_digest"]
        ),
        call_shape_policy_digest=step_two_call_shape_digest,
        call_shape_policy_version="issue-790-v10",
    )
    step_two_contract = _issue_790_successor_fixture_contract(
        allocation=third_allocation,
        terminal=third_terminal,
        plan=step_two_plan,
        approved_at=step_two_approved_at,
    )
    monkeypatch.setattr(
        issue_790_contract_module,
        "_SUCCESS_SEQUENCE_CONTRACTS",
        (successor_contract, step_two_contract),
    )
    monkeypatch.setattr(
        issue_790_operation,
        "load_checked_graphiti_call_shape_policy",
        lambda: SimpleNamespace(
            canonical_digest=step_two_call_shape_digest,
            version="issue-790-v10",
            qualified_routes=(
                SimpleNamespace(
                    leaf_class=issue_790_operation.GraphitiLeafClass.PRIMARY,
                    provider="cursor-agent-cli",
                    route=GRAPHITI_CHAT_PRIMARY_ROUTE,
                    model="composer-2.5",
                    max_total_tokens=third_policy.max_total_tokens,
                    command_flags=("CONTROLLER_TIMEOUT_MS=170000",),
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        issue_790_operation,
        "GRAPHITI_EXTRACTION_TIMEOUT_MS",
        190_000,
    )
    step_two_dry_run = dry_run_issue_790_plan(
        source_store=store,
        scratch_store=tmp_path / "step-two-dry-run.sqlite3",
        plan=step_two_plan,
        observed_at=T0 + timedelta(seconds=285),
    )
    assert step_two_dry_run["predecessor"]["causal_report"] == (
        second_causal_report
    )
    assert {
        record["approved_plan_digest"]
        for record in step_two_dry_run["retry_exclusions"]
    } == {FIXTURE_790_PLAN_DIGEST}

    connection = sqlite3.connect(store)
    connection.execute("DELETE FROM issue_790_graphiti_retry_exclusions")
    connection.commit()
    connection.close()
    repository.retain_retry_exclusions(
        approved_plan_digest=successor_plan_digest,
        disposition_digest=str(second_disposition["disposition_digest"]),
        events=RETRY_FORBIDDEN_EVENTS,
        excluded_at=T0 + timedelta(seconds=106),
    )
    monkeypatch.setattr(
        issue_790_operation,
        "load_checked_graphiti_call_shape_policy",
        lambda: SimpleNamespace(
            canonical_digest=(
                "sha256:7e6bd15613cefda0820a1d339c8790f0185946aade1622dadbb8c468f558bb18"
            ),
            version="issue-790-v9",
            qualified_routes=(
                SimpleNamespace(
                    leaf_class=issue_790_operation.GraphitiLeafClass.PRIMARY,
                    provider="cursor-agent-cli",
                    route=GRAPHITI_CHAT_PRIMARY_ROUTE,
                    model="composer-2.5",
                    max_total_tokens=second_policy.max_total_tokens,
                    command_flags=("CONTROLLER_TIMEOUT_MS=160000",),
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        issue_790_operation,
        "GRAPHITI_EXTRACTION_TIMEOUT_MS",
        180_000,
    )
    with pytest.raises(
        Issue790DispositionError,
        match="retry exclusions do not bind the immutable root",
    ):
        dry_run_issue_790_plan(
            source_store=store,
            scratch_store=tmp_path / "rebound-exclusions-dry-run.sqlite3",
            plan=successor_plan,
            observed_at=T0 + timedelta(seconds=110),
        )


def test_issue_790_success_sequence_stops_after_truthful_predecessor_success() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(
        (
            root
            / "docs/operations/2026-08-26-issue-790-success-sequence-step-1.json"
        ).read_text(encoding="utf-8")
    )
    sequence = dict(plan["sequence"])
    predecessor = dict(sequence["predecessor"])
    consumption = {
        "consumption_digest": predecessor["consumption_digest"],
        "approved_plan_digest": predecessor["plan_digest"],
        "event_id": predecessor["event_id"],
        "ledger_seq": predecessor["ledger_seq"],
    }
    outcome = {
        "schema_version": "newsroom.issue-790.canary-outcome.v3",
        "outcome_digest": predecessor["outcome_digest"],
        "approved_plan_digest": predecessor["plan_digest"],
        "event_id": predecessor["event_id"],
        "ledger_seq": predecessor["ledger_seq"],
        "retry_authorised": False,
        "result_class": "TRUTHFUL_PROVIDER_SUCCESS",
    }

    class SuccessfulPredecessorRepository:
        @staticmethod
        def existing_consumption(
            *, approved_plan_digest: str, event_id: str, ledger_seq: int
        ) -> dict[str, object]:
            assert approved_plan_digest == predecessor["plan_digest"]
            assert event_id == predecessor["event_id"]
            assert ledger_seq == predecessor["ledger_seq"]
            return consumption

        @staticmethod
        def existing_outcome(*, consumption_digest: str) -> dict[str, object]:
            assert consumption_digest == predecessor["consumption_digest"]
            return outcome

    with pytest.raises(
        Issue790DispositionError,
        match="predecessor already reached truthful success",
    ):
        issue_790_operation._require_sequence_predecessor(
            SuccessfulPredecessorRepository(),  # type: ignore[arg-type]
            plan=plan,
        )


def test_issue_790_sequence_stops_after_misclassified_terminal_success() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(
        (
            root
            / "docs/operations/2026-08-26-issue-790-success-sequence-step-1.json"
        ).read_text(encoding="utf-8")
    )
    sequence = dict(plan["sequence"])
    sequence["sequence_ordinal"] = 2
    plan["sequence"] = sequence
    predecessor = dict(sequence["predecessor"])
    consumption = {
        "consumption_digest": predecessor["consumption_digest"],
        "approved_plan_digest": predecessor["plan_digest"],
        "event_id": predecessor["event_id"],
        "ledger_seq": predecessor["ledger_seq"],
    }
    outcome = {
        "schema_version": "newsroom.issue-790.canary-outcome.v3",
        "outcome_digest": predecessor["outcome_digest"],
        "approved_plan_digest": predecessor["plan_digest"],
        "event_id": predecessor["event_id"],
        "ledger_seq": predecessor["ledger_seq"],
        "retry_authorised": False,
        "result_class": "UNCLASSIFIED_NON_SUCCESS",
        "causal_report": None,
        "state_before_seal": "TERMINAL",
        "attempt_count": 1,
        "provider_dispatched": True,
        "process_result": {
            "state": "TERMINAL",
            "attempt_count": 1,
        },
    }

    class UnclassifiedPredecessorRepository:
        @staticmethod
        def existing_consumption(
            *, approved_plan_digest: str, event_id: str, ledger_seq: int
        ) -> dict[str, object]:
            assert approved_plan_digest == predecessor["plan_digest"]
            assert event_id == predecessor["event_id"]
            assert ledger_seq == predecessor["ledger_seq"]
            return consumption

        @staticmethod
        def existing_outcome(*, consumption_digest: str) -> dict[str, object]:
            assert consumption_digest == predecessor["consumption_digest"]
            return outcome

    with pytest.raises(
        Issue790DispositionError,
        match="predecessor retained a terminal success boundary",
    ):
        issue_790_operation._require_sequence_predecessor(
            UnclassifiedPredecessorRepository(),  # type: ignore[arg-type]
            plan=plan,
        )


def test_issue_790_reviewed_non_timeout_fix_preserves_attempt_budgets() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(
        (
            root
            / "docs/operations/2026-08-26-issue-790-success-sequence-step-1.json"
        ).read_text(encoding="utf-8")
    )
    sequence = dict(plan["sequence"])
    sequence["sequence_ordinal"] = 2
    sequence["constraint_change"] = "REVIEWED_NON_TIMEOUT_FIX"
    predecessor = dict(sequence["predecessor"])
    causal_report: dict[str, object] = {
        "schema_version": "newsroom.issue-790.non-timeout-causal-report.v1",
        "classification": "NON_TIMEOUT_FAILURE",
        "causal_constraint": "REVIEWED_CODE_OR_CONFIGURATION_FIX",
        "local_cause": "RESULT_VALIDATION_FAILED",
        "provider_cause": "MALFORMED_RESPONSE",
        "predecessor_outcome_digest": predecessor["outcome_digest"],
        "event_id": predecessor["event_id"],
        "boundary": "RESPONSE_VALIDATION",
        "configured_controller_timeout_ms": 160_000,
        "configured_extraction_timeout_ms": 180_000,
        "cleanup_reserve_ms": 20_000,
        "deadline_at": None,
        "elapsed_ms": 2_000,
        "last_progress": "PROVIDER_RESPONSE_RECEIVED",
        "termination": "PROCESS_EXITED",
        "diagnostic_reference": "retained:fixture:non-timeout",
    }
    causal_report["report_digest"] = _digest(causal_report)
    reviewed_fix: dict[str, object] = {
        "schema_version": "newsroom.issue-790.reviewed-non-timeout-fix.v1",
        "predecessor_outcome_digest": predecessor["outcome_digest"],
        "causal_report_digest": causal_report["report_digest"],
        "fix_kind": "CODE",
        "pull_request_url": "https://github.com/fol2/newsroom/pull/999",
        "reviewed_fix_revision": "a" * 40,
        "review_receipt_digest": _digest({"review": "PASS"}),
        "provider_free_qualification_digest": _digest(
            {"focused_tests": "PASS"}
        ),
    }
    reviewed_fix["record_digest"] = _digest(reviewed_fix)
    sequence["predecessor_causal_report"] = causal_report
    sequence["reviewed_fix"] = reviewed_fix
    plan["sequence"] = sequence
    plan["canonical_digest"] = _digest(
        {key: value for key, value in plan.items() if key != "canonical_digest"}
    )
    assert validate_issue_790_plan(plan) == plan

    consumption = {
        "consumption_digest": predecessor["consumption_digest"],
        "approved_plan_digest": predecessor["plan_digest"],
        "event_id": predecessor["event_id"],
        "ledger_seq": predecessor["ledger_seq"],
    }
    outcome = {
        "schema_version": "newsroom.issue-790.canary-outcome.v3",
        "outcome_digest": predecessor["outcome_digest"],
        "approved_plan_digest": predecessor["plan_digest"],
        "event_id": predecessor["event_id"],
        "ledger_seq": predecessor["ledger_seq"],
        "retry_authorised": False,
        "result_class": "UNCLASSIFIED_NON_SUCCESS",
        "causal_report": None,
    }

    class ReviewedFixRepository:
        @staticmethod
        def existing_consumption(
            *, approved_plan_digest: str, event_id: str, ledger_seq: int
        ) -> dict[str, object]:
            assert approved_plan_digest == predecessor["plan_digest"]
            assert event_id == predecessor["event_id"]
            assert ledger_seq == predecessor["ledger_seq"]
            return consumption

        @staticmethod
        def existing_outcome(*, consumption_digest: str) -> dict[str, object]:
            assert consumption_digest == predecessor["consumption_digest"]
            return outcome

    retained = issue_790_operation._require_sequence_predecessor(
        ReviewedFixRepository(),  # type: ignore[arg-type]
        plan=plan,
    )
    assert retained is not None
    assert retained["reviewed_fix"] == reviewed_fix
    assert sequence["controller_timeout_ms"] == 160_000
    assert sequence["extraction_timeout_ms"] == 180_000


def test_issue_790_timeout_report_deduplicates_one_retained_diagnostic(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    event_id = _digest({"issue-790-timeout-event": 1})
    _policy_value, _allocation, terminal = (
        _open_unreported_graphiti_subscription_leaf(
            service,
            cycle_id=event_id,
            request="issue-790-timeout-report",
            outcome="TIMEOUT",
            elapsed_ms=160_001,
        )
    )
    diagnostic = dict(
        _issue_790_controller_timeout_report_fixture(
            terminal,
            configured_timeout_ms=160_000,
        )["diagnostic"]  # type: ignore[arg-type]
    )
    store = tmp_path / "unpublished.sqlite3"
    connection = connect_unpublished_store(str(store))
    try:
        receipt = {
            "ingest_id": f"ingest-{event_id}",
            "attempt_number": 1,
            "outcome": "FAILED",
            "timeout_diagnostics": [diagnostic],
            "chat_invocations": [
                {
                    "provider": "cursor-agent-cli",
                    "transport_diagnostic": diagnostic,
                }
            ],
            "receipt_digest": "",
        }
        receipt_digest = insert_graphiti_attempt_receipt(
            connection,
            ingest_id=f"ingest-{event_id}",
            attempt_number=1,
            outcome="FAILED",
            receipt=receipt,
        )
        connection.commit()
    finally:
        connection.close()

    report = issue_790_operation._issue_790_controller_timeout_report(
        store,
        event_id=event_id,
        configured_timeout_ms=160_000,
    )

    assert report is not None
    assert report["diagnostic"] == diagnostic
    assert report["diagnostic_reference"] == (
        "retained:unpublished_graphiti_attempt_receipts:" + receipt_digest
    )


def test_issue_790_timeout_report_ignores_non_timeout_receipt(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    event_id = _digest({"issue-790-non-timeout-event": 1})
    _open_unreported_graphiti_subscription_leaf(
        service,
        cycle_id=event_id,
        request="issue-790-non-timeout-report",
        outcome="FAILED",
        elapsed_ms=85_000,
    )
    store = tmp_path / "unpublished.sqlite3"
    connection = connect_unpublished_store(str(store))
    try:
        receipt = {
            "ingest_id": f"ingest-{event_id}",
            "attempt_number": 1,
            "outcome": "FAILED",
            "chat_invocations": [
                {
                    "provider": "cursor-agent-cli",
                    "outcome": "FAILED",
                    "failure": "RuntimeError",
                }
            ],
            "receipt_digest": "",
        }
        insert_graphiti_attempt_receipt(
            connection,
            ingest_id=f"ingest-{event_id}",
            attempt_number=1,
            outcome="FAILED",
            receipt=receipt,
        )
        connection.commit()
    finally:
        connection.close()

    assert (
        issue_790_operation._issue_790_controller_timeout_report(
            store,
            event_id=event_id,
            configured_timeout_ms=160_000,
        )
        is None
    )


def test_issue_790_crash_after_success_recovers_truthful_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        issue_790_operation,
        "_issue_790_canary_usage_evidence",
        lambda *_args, **_kwargs: {
            "primary_chat_leaf_count": 1,
            "qualified_primary_identity_count": 1,
            "truthful_primary_usage_count": 1,
            "fallback_chat_leaf_count": 0,
            "unresolved_terminal_count": 0,
            "unterminated_leaf_count": 0,
        },
    )
    monkeypatch.setattr(
        issue_790_operation,
        "_issue_790_controller_timeout_report",
        lambda *_args, **_kwargs: pytest.fail(
            "truthful success must stop before timeout classification"
        ),
    )
    recovered = issue_790_operation._issue_790_iterative_result(
        store=tmp_path / "unpublished.sqlite3",
        plan={"sequence": {"controller_timeout_ms": 160_000}},
        event_id=_digest({"crash": "after-success"}),
        process_result={"state": "TERMINAL", "attempt_count": 1},
        exception_present=False,
    )
    assert recovered == {
        "result_class": "TRUTHFUL_PROVIDER_SUCCESS",
        "causal_report": None,
    }


def test_issue_790_crash_after_timeout_recovers_causal_non_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        issue_790_operation,
        "_issue_790_canary_usage_evidence",
        lambda *_args, **_kwargs: {
            "primary_chat_leaf_count": 1,
            "qualified_primary_identity_count": 1,
            "truthful_primary_usage_count": 0,
            "fallback_chat_leaf_count": 0,
            "unresolved_terminal_count": 1,
            "unterminated_leaf_count": 0,
        },
    )
    causal_report = {
        "schema_version": "newsroom.issue-790.causal-report.v1",
        "report_digest": _digest({"crash": "after-timeout"}),
    }
    monkeypatch.setattr(
        issue_790_operation,
        "_issue_790_controller_timeout_report",
        lambda *_args, **values: (
            causal_report
            if values["configured_timeout_ms"] == 160_000
            else None
        ),
    )
    recovered = issue_790_operation._issue_790_iterative_result(
        store=tmp_path / "unpublished.sqlite3",
        plan={"sequence": {"controller_timeout_ms": 160_000}},
        event_id=_digest({"crash": "after-timeout"}),
        process_result={"state": "RETRY_HELD", "attempt_count": 1},
        exception_present=False,
    )
    assert recovered == {
        "result_class": "CONTROLLER_TIMEOUT_NON_SUCCESS",
        "causal_report": causal_report,
    }


def test_issue_790_canary_orchestrator_runs_only_exact_fresh_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    policy, allocation, terminal = _open_unreported_graphiti_subscription_leaf(
        service
    )
    plan = _issue_790_plan(policy, allocation, terminal)
    _bind_issue_790_fixture_contract(
        monkeypatch,
        allocation=allocation,
        terminal=terminal,
        plan_digest=str(plan["canonical_digest"]),
    )
    approved_at = T0 + timedelta(seconds=5)
    authority, authority_digest = _conservative_disposition_authority(
        allocation,
        terminal,
        approved_at=approved_at,
        approved_plan_digest=str(plan["canonical_digest"]),
    )
    disposition = service.disposition_unreported_subscription_usage(
        invocation_id=allocation.invocation_id,
        expected_terminal_digest=terminal.terminal_digest,
        expected_allocation_digest=allocation.canonical_digest,
        approved_by=str(authority["approved_by"]),
        approval_reference=str(authority["approval_reference"]),
        approved_at=approved_at,
        approved_plan_digest=str(plan["canonical_digest"]),
        authority_digest=authority_digest,
        observed_at=T0 + timedelta(seconds=10),
    )
    route = service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)
    service.release_route_circuit(
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        release_kind="AUTHORISED_OPERATOR_RESET",
        bound_failure_reason=str(route["reason"]),
        evidence_digest=str(disposition["disposition_digest"]),
        recorded_at=T0 + timedelta(seconds=11),
    )
    store = tmp_path / "unpublished.sqlite3"
    _seed_issue_790_retry_events(store)
    event_id, ingest_id = _seed_fresh_issue_790_event(store, ledger_seq=2002)
    canary_repository = Issue790CanaryRepository(str(store))
    canary_repository.retain_retry_exclusions(
        approved_plan_digest=str(plan["canonical_digest"]),
        disposition_digest=str(disposition["disposition_digest"]),
        events=RETRY_FORBIDDEN_EVENTS,
        excluded_at=T0 + timedelta(seconds=12),
    )
    proving = tmp_path / "proving.sqlite3"
    proving_connection = sqlite3.connect(proving)
    proving_connection.execute("CREATE TABLE fixture(value INTEGER)")
    proving_connection.commit()
    proving_connection.close()
    observed_at = T0 + timedelta(seconds=20)
    _patch_issue_790_live_evidence(
        monkeypatch,
        store=store,
        observed_at=observed_at,
    )
    preflight = _issue_790_canary_preflight(
        store,
        event_id=event_id,
        ledger_seq=2002,
        evaluated_at=observed_at,
    )
    monkeypatch.setattr(
        issue_790_operation,
        "_qualify_issue_790_event",
        lambda **_values: preflight,
    )
    prepared = _persisted_issue_790_prepared_canary(
        store=store,
        proving_store=proving,
        plan=plan,
        observed_at=observed_at,
        event_id=event_id,
        ledger_seq=2002,
    )
    ready_runtime_check = issue_790_operation._qualify_real_graphiti_runtime
    monkeypatch.setattr(
        issue_790_operation,
        "_qualify_real_graphiti_runtime",
        lambda: (_ for _ in ()).throw(
            Issue790DispositionError("issue #790 real Graphiti runtime is unavailable")
        ),
    )
    missing_runtime_backup = tmp_path / "missing-runtime.sqlite3"
    with pytest.raises(
        Issue790DispositionError,
        match="real Graphiti runtime is unavailable",
    ):
        run_issue_790_canary(
            store=store,
            proving_store=proving,
            backup_path=missing_runtime_backup,
            plan=plan,
            observed_at=observed_at,
            repository_root=tmp_path,
            event_id=event_id,
            ledger_seq=2002,
            disposition_digest=str(disposition["disposition_digest"]),
            prepared=prepared,
        )
    assert missing_runtime_backup.exists() is False
    assert canary_repository.existing_consumption(
        approved_plan_digest=str(plan["canonical_digest"])
    ) is None
    monkeypatch.setattr(
        issue_790_operation,
        "_qualify_real_graphiti_runtime",
        ready_runtime_check,
    )
    runtime_checks: list[str] = []
    runtime_check = issue_790_operation._qualify_real_graphiti_runtime

    def tracked_runtime_check() -> dict[str, object]:
        runtime_checks.append("checked-before-consumption")
        return runtime_check()

    monkeypatch.setattr(
        issue_790_operation,
        "_qualify_real_graphiti_runtime",
        tracked_runtime_check,
    )
    retry_snapshot_calls: list[list[int]] = []
    original_retry_check = issue_790_operation._require_retry_events_unchanged

    def tracked_retry_check(
        checked_store: Path,
        checked_plan: Mapping[str, object],
    ) -> list[dict[str, object]]:
        retry_snapshot_calls.append(
            [
                int(item["ledger_seq"])
                for item in checked_plan["retry_forbidden_events"]  # type: ignore[index]
            ]
        )
        return original_retry_check(checked_store, checked_plan)

    monkeypatch.setattr(
        issue_790_operation,
        "_require_retry_events_unchanged",
        tracked_retry_check,
    )

    dispatch_calls: list[str] = []

    def consume_exact_event(**values: object) -> GraphitiProcessResult:
        dispatch_calls.append(str(values["event_id"]))
        assert values["event_id"] == event_id
        assert values["unpublished_store"] == store
        canary_service = values["model_usage"]
        assert isinstance(canary_service, ModelUsageService)
        canary_envelope = _envelope(
            cycle_id=event_id,
            workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            candidate_id=None,
            ingest_id=ingest_id,
        )
        canary_policy = _policy(
            workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            provider="cursor-agent-cli",
            route=GRAPHITI_CHAT_PRIMARY_ROUTE,
            model="composer-2.5",
            hard_estimate_ceiling_tokens=None,
        )
        canary_service.register_policy(canary_policy)
        canary_service.open_envelope(canary_envelope)
        canary_allocation = _allocation(
            canary_envelope,
            canary_policy,
            request="issue-790-canary",
        )
        canary_service.allocate(canary_allocation, owner_emergency_stop=False)
        canary_service.observe_transport(
            invocation_id=canary_allocation.invocation_id,
            observed_at=T0 + timedelta(seconds=21),
            state="DISPATCH_STARTED",
            evidence_digest=_digest({"dispatch": event_id}),
        )
        canary_service.complete(
            _reported(
                canary_allocation,
                total=125,
                completed_at=T0 + timedelta(seconds=22),
            )
        )
        connection = connect_unpublished_store(str(store))
        connection.execute(
            "UPDATE unpublished_graphiti_revision_events SET state='TERMINAL',"
            "attempt_count=1,provider_dispatched=1,terminal_at=?,proposal_count=1 "
            "WHERE event_id=?",
            (T0.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), event_id),
        )
        connection.commit()
        connection.close()
        return GraphitiProcessResult(event_id, 2002, "TERMINAL", 1)

    monkeypatch.setattr(
        issue_790_operation,
        "_consume_issue_790_event",
        consume_exact_event,
    )
    backup = tmp_path / "pre-canary.sqlite3"
    receipt = run_issue_790_canary(
        store=store,
        proving_store=proving,
        backup_path=backup,
        plan=plan,
        observed_at=observed_at,
        repository_root=tmp_path,
        event_id=event_id,
        ledger_seq=2002,
        disposition_digest=str(disposition["disposition_digest"]),
        prepared=prepared,
    )

    assert receipt["canary_evidence_passed"] is True
    assert receipt["process_result"] == {
        "event_id": event_id,
        "ledger_seq": 2002,
        "state": "TERMINAL",
        "attempt_count": 1,
    }
    assert receipt["usage_evidence"]["provider_backed_terminal_count"] == 1  # type: ignore[index]
    assert receipt["usage_evidence"]["truthful_nonzero_usage_count"] == 1  # type: ignore[index]
    assert receipt["usage_evidence"]["primary_chat_leaf_count"] == 1  # type: ignore[index]
    assert receipt["usage_evidence"]["qualified_primary_identity_count"] == 1  # type: ignore[index]
    assert receipt["usage_evidence"]["truthful_primary_usage_count"] == 1  # type: ignore[index]
    assert receipt["usage_evidence"]["fallback_chat_leaf_count"] == 0  # type: ignore[index]
    assert receipt["retry_forbidden_events_unchanged"] is True
    assert retry_snapshot_calls == [[1932, 1972], [1932, 1972]]
    assert receipt["runtime_readiness"]["framework_version"] == "0.29.3"
    assert runtime_checks == ["checked-before-consumption"]
    assert receipt["worker_remained_unloaded"] is True
    assert dispatch_calls == [event_id]
    assert receipt["receipt_digest"] == _digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    resumed = run_issue_790_canary(
        store=store,
        proving_store=proving,
        backup_path=backup,
        plan=plan,
        observed_at=observed_at,
        repository_root=tmp_path,
        event_id=event_id,
        ledger_seq=2002,
        disposition_digest=str(disposition["disposition_digest"]),
    )
    assert resumed["canary_evidence_passed"] is True
    assert resumed["resumed_zero_io_finalisation"] is True
    assert resumed["provider_dispatch_attempted_this_run"] is False
    assert retry_snapshot_calls == [
        [1932, 1972],
        [1932, 1972],
        [1932, 1972],
        [1932, 1972],
    ]
    assert resumed["runtime_readiness"] is None
    assert runtime_checks == ["checked-before-consumption"]
    assert dispatch_calls == [event_id]


def test_issue_790_production_canary_rejects_successor_for_spent_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, plan, disposition, repository = _released_issue_790_canary(
        tmp_path, monkeypatch
    )
    plan_digest = str(plan["canonical_digest"])
    disposition_digest = str(disposition["disposition_digest"])
    _seed_fresh_issue_790_event(
        store,
        ledger_seq=CANDIDATE_LEDGER_SEQ,
        event_id=CANDIDATE_EVENT_ID,
    )
    spent = _consume_plan_event(
        repository,
        store,
        plan_digest=plan_digest,
        disposition_digest=disposition_digest,
        event_id=CANDIDATE_EVENT_ID,
        ledger_seq=CANDIDATE_LEDGER_SEQ,
        owner_id="issue-790-canary:spent-13665",
        consumed_at=T0 + timedelta(seconds=20),
    )
    repository.complete(
        consumption_digest=str(spent["consumption_digest"]),
        event_id=CANDIDATE_EVENT_ID,
        ledger_seq=CANDIDATE_LEDGER_SEQ,
        owner_id="issue-790-canary:spent-13665",
        process_result=None,
        exception_code="AMBIGUOUS_EFFECT",
        completed_at=T0 + timedelta(seconds=30),
    )
    connection = sqlite3.connect(store)
    connection.execute(
        "CREATE UNIQUE INDEX issue_790_legacy_plan_consumption_unique ON "
        "issue_790_bounded_canary_consumptions(approved_plan_digest)"
    )
    connection.commit()
    connection.close()
    _seed_fresh_issue_790_event(
        store,
        ledger_seq=SUCCESSOR_LEDGER_SEQ,
        event_id=SUCCESSOR_EVENT_ID,
    )
    proving = tmp_path / "proving.sqlite3"
    proving_connection = sqlite3.connect(proving)
    proving_connection.execute("CREATE TABLE fixture(value INTEGER)")
    proving_connection.commit()
    proving_connection.close()
    observed_at = T0 + timedelta(seconds=40)
    _patch_issue_790_live_evidence(monkeypatch, store=store, observed_at=observed_at)

    backup = tmp_path / "successor-13671.sqlite3"
    with pytest.raises(PreparedCanaryError) as caught:
        _persisted_issue_790_prepared_canary(
            store=store,
            proving_store=proving,
            plan=plan,
            observed_at=observed_at,
            event_id=SUCCESSOR_EVENT_ID,
            ledger_seq=SUCCESSOR_LEDGER_SEQ,
        )
    assert caught.value.failure_code == "BOUNDED_CANARY_AUTHORITY_CONSUMED"
    assert backup.exists() is False
    assert ("approved_plan_digest",) in _unique_consumption_columns(store)
    assert service_row_count(store, "issue_790_bounded_canary_consumptions") == 1

def test_issue_790_crash_after_leaf_marker_syncs_dispatch_during_finalisation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    _policy_value, allocation, terminal = (
        _open_unreported_graphiti_subscription_leaf(service)
    )
    _bind_issue_790_fixture_contract(
        monkeypatch,
        allocation=allocation,
        terminal=terminal,
    )
    approved_at = T0 + timedelta(seconds=5)
    authority, authority_digest = _conservative_disposition_authority(
        allocation,
        terminal,
        approved_at=approved_at,
    )
    disposition = service.disposition_unreported_subscription_usage(
        invocation_id=allocation.invocation_id,
        expected_terminal_digest=terminal.terminal_digest,
        expected_allocation_digest=allocation.canonical_digest,
        approved_by=str(authority["approved_by"]),
        approval_reference=str(authority["approval_reference"]),
        approved_at=approved_at,
        approved_plan_digest=FIXTURE_790_PLAN_DIGEST,
        authority_digest=authority_digest,
        observed_at=T0 + timedelta(seconds=10),
    )
    route = service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)
    service.release_route_circuit(
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        release_kind="AUTHORISED_OPERATOR_RESET",
        bound_failure_reason=str(route["reason"]),
        evidence_digest=str(disposition["disposition_digest"]),
        recorded_at=T0 + timedelta(seconds=11),
    )
    store = tmp_path / "unpublished.sqlite3"
    event_id, _ingest_id = _seed_fresh_issue_790_event(
        store,
        ledger_seq=2003,
        retain_unit_refs=False,
    )
    repository = Issue790CanaryRepository(str(store))
    owner_id = "issue-790-canary:crash-before-claim"
    consumption = repository.consume(
        approved_plan_digest=FIXTURE_790_PLAN_DIGEST,
        disposition_digest=str(disposition["disposition_digest"]),
        event_id=event_id,
        ledger_seq=2003,
        owner_id=owner_id,
        preflight_evidence=_issue_790_canary_preflight(
            store,
            event_id=event_id,
            ledger_seq=2003,
            evaluated_at=T0 + timedelta(seconds=19),
            resolved_units=[_fixture_issue_790_unit_ref(2003)],
        ),
        consumed_at=T0 + timedelta(seconds=20),
    )
    canary_envelope = _envelope(
        cycle_id=event_id,
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        candidate_id=None,
        ingest_id=_ingest_id,
    )
    canary_policy = _policy(
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        provider="cursor-agent-cli",
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        model="composer-2.5",
        hard_estimate_ceiling_tokens=None,
    )
    service.register_policy(canary_policy)
    service.open_envelope(canary_envelope)
    canary_allocation = _allocation(
        canary_envelope,
        canary_policy,
        request="issue-790-crash-after-marker",
    )
    service.allocate(canary_allocation, owner_emergency_stop=False)
    service.observe_transport(
        invocation_id=canary_allocation.invocation_id,
        observed_at=T0 + timedelta(seconds=20, milliseconds=1),
        state="DISPATCH_STARTED",
        evidence_digest=_digest({"dispatch": event_id}),
    )
    connection = sqlite3.connect(store)
    connection.execute(
        "UPDATE unpublished_graphiti_revision_events SET state='RUNNING',"
        "attempt_count=1,claim_owner=?,claim_expires_at=?,provider_dispatched=0 "
        "WHERE event_id=?",
        (
            owner_id,
            (T0 + timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            event_id,
        ),
    )
    connection.commit()
    connection.close()

    outcome = repository.finalise_without_dispatch(
        consumption_digest=str(consumption["consumption_digest"]),
        event_id=event_id,
        ledger_seq=2003,
        owner_id=owner_id,
        completed_at=T0 + timedelta(seconds=21),
    )

    connection = sqlite3.connect(store)
    retained = connection.execute(
        "SELECT state,attempt_count,provider_dispatched,last_failure_code "
        "FROM unpublished_graphiti_revision_events WHERE event_id=?",
        (event_id,),
    ).fetchone()
    connection.close()
    assert outcome["completion_mode"] == "ZERO_IO_RECOVERY"
    assert outcome["process_result"]["state"] == "RUNNING"
    assert outcome["event_provider_dispatched_before_seal"] is False
    assert outcome["provider_dispatched"] is True
    assert retained == (
        "CONFIGURATION_HELD",
        1,
        1,
        "BOUNDED_CANARY_AUTHORITY_EXHAUSTED:NO_EVENT_RESULT",
    )


def test_issue_790_route_mismatch_writes_no_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    policy, allocation, terminal = _open_unreported_graphiti_subscription_leaf(
        service
    )
    store = tmp_path / "unpublished.sqlite3"
    plan = _issue_790_plan(policy, allocation, terminal)
    _bind_issue_790_fixture_contract(
        monkeypatch,
        allocation=allocation,
        terminal=terminal,
        plan_digest=str(plan["canonical_digest"]),
    )
    _seed_issue_790_retry_events(store)
    _patch_issue_790_live_evidence(
        monkeypatch,
        store=store,
        observed_at=T0 + timedelta(seconds=10),
    )

    with pytest.raises(
        Issue790DispositionError,
        match="current route failure differs",
    ):
        apply_issue_790_plan(
            store=store,
            backup_path=tmp_path / "unpublished.pre-790.sqlite3",
            plan=plan,
            observed_at=T0 + timedelta(seconds=10),
            repository_root=tmp_path,
        )

    assert service_row_count(
        store, "model_usage_conservative_dispositions"
    ) == 0
    assert service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)["state"] == "OPEN"


def test_reconciled_policy_breach_keeps_the_route_open(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _envelope_value, _policy_value, allocation = _open_and_allocate(service)
    service.complete(
        InvocationTerminal.create(
            invocation_id=allocation.invocation_id,
            outcome="TRANSPORT_LOST",
            failure_class="MISSING_TELEMETRY",
            usage_status=UsageStatus.UNREPORTED,
            components=UsageComponents(provenance="UNAVAILABLE"),
            dispatch_at=T0 + timedelta(seconds=1),
            completed_at=T0 + timedelta(seconds=2),
            observed_at=T0 + timedelta(seconds=2),
            subscription_cli_chat_not_cash_debited=True,
        )
    )
    service.reconcile(
        invocation_id=allocation.invocation_id,
        components=UsageComponents(
            input_tokens=2_900,
            output_tokens=100,
            total_tokens=3_000,
            provenance="PROVIDER_REPORTED",
        ),
        provider_telemetry={"late": "over-limit"},
        observed_at=T0 + timedelta(minutes=5),
        raw_telemetry_pointer="private://late/over-limit",
    )

    row = service.query(start=T0, end=T0 + timedelta(minutes=6))["leaves"][0]
    assert row["policy_breach"] == "MAX_TOTAL_TOKENS_EXCEEDED"
    assert service.route_state("CONT_PRIMARY")["state"] == "OPEN"


def test_reconciliation_closes_only_after_every_route_uncertainty_is_resolved(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    envelope = _envelope(
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        candidate_id=None,
        ingest_id="ingest-reconciliation",
    )
    policy = _policy(
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        provider="cursor-agent-cli",
        route="GRAPHITI_CHAT",
        model="cursor-pinned",
    )
    _envelope_value, _policy_value, first = _open_and_allocate(
        service, envelope=envelope, policy=policy
    )
    second = _allocation(
        envelope,
        policy,
        leaf_ordinal=2,
        request="request-2",
    )
    service.allocate(second, owner_emergency_stop=False)
    for ordinal, allocation in enumerate((second, first), start=1):
        service.complete(
            InvocationTerminal.create(
                invocation_id=allocation.invocation_id,
                outcome="TRANSPORT_LOST",
                failure_class="MISSING_TELEMETRY",
                usage_status=UsageStatus.UNREPORTED,
                components=UsageComponents(provenance="UNAVAILABLE"),
                dispatch_at=T0 + timedelta(seconds=ordinal),
                completed_at=T0 + timedelta(seconds=ordinal + 2),
                observed_at=T0 + timedelta(seconds=ordinal + 2),
                subscription_cli_chat_not_cash_debited=True,
            )
        )

    for ordinal, allocation in enumerate((first, second), start=1):
        service.reconcile(
            invocation_id=allocation.invocation_id,
            components=UsageComponents(
                input_tokens=100,
                output_tokens=25,
                total_tokens=125,
                provenance="PROVIDER_REPORTED",
            ),
            provider_telemetry={"late": ordinal},
            observed_at=T0 + timedelta(minutes=ordinal),
            raw_telemetry_pointer=f"private://late/{ordinal}",
        )
        assert service.route_state("GRAPHITI_CHAT")["state"] == (
            "OPEN" if ordinal == 1 else "CLOSED"
        )


def test_reconciliation_does_not_clear_an_earlier_policy_breach(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    envelope = _envelope(
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        candidate_id=None,
        ingest_id="ingest-policy-breach",
    )
    policy = _policy(
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        provider="cursor-agent-cli",
        route="GRAPHITI_CHAT",
        model="cursor-pinned",
    )
    _envelope_value, _policy_value, breached = _open_and_allocate(
        service, envelope=envelope, policy=policy
    )
    uncertain = _allocation(
        envelope,
        policy,
        leaf_ordinal=2,
        request="request-uncertain",
    )
    service.allocate(uncertain, owner_emergency_stop=False)
    service.complete(_reported(breached, total=2_501))
    service.complete(
        InvocationTerminal.create(
            invocation_id=uncertain.invocation_id,
            outcome="TRANSPORT_LOST",
            failure_class="MISSING_TELEMETRY",
            usage_status=UsageStatus.UNREPORTED,
            components=UsageComponents(provenance="UNAVAILABLE"),
            dispatch_at=T0 + timedelta(seconds=2),
            completed_at=T0 + timedelta(seconds=4),
            observed_at=T0 + timedelta(seconds=4),
            subscription_cli_chat_not_cash_debited=True,
        )
    )
    service.reconcile(
        invocation_id=uncertain.invocation_id,
        components=UsageComponents(
            input_tokens=100,
            output_tokens=25,
            total_tokens=125,
            provenance="PROVIDER_REPORTED",
        ),
        provider_telemetry={"late": "valid"},
        observed_at=T0 + timedelta(minutes=1),
        raw_telemetry_pointer="private://late/valid",
    )

    assert service.route_state("GRAPHITI_CHAT")["state"] == "OPEN"


def test_cont_usage_circuit_uses_the_canonical_governor_route(tmp_path: Path) -> None:
    from newsroom.control_plane.cycle_governor import DurableCycleGovernor

    path = tmp_path / "unpublished.sqlite3"
    DurableCycleGovernor(str(path))
    service = ModelUsageService(str(path))
    _envelope_value, policy, allocation = _open_and_allocate(service)
    service.complete(
        InvocationTerminal.create(
            invocation_id=allocation.invocation_id,
            outcome="TRANSPORT_LOST",
            failure_class="MISSING_TELEMETRY",
            usage_status=UsageStatus.UNREPORTED,
            components=UsageComponents(provenance="UNAVAILABLE"),
            dispatch_at=T0 + timedelta(seconds=1),
            completed_at=T0 + timedelta(seconds=2),
            observed_at=T0 + timedelta(seconds=2),
            subscription_cli_chat_not_cash_debited=True,
        )
    )

    connection = sqlite3.connect(path)
    routes = connection.execute(
        "SELECT route,state FROM unpublished_route_circuits ORDER BY route"
    ).fetchall()
    connection.execute(
        "UPDATE unpublished_route_circuits SET state='CLOSED' WHERE route='CONT'"
    )
    connection.commit()
    connection.close()
    assert routes == [("CONT", "OPEN")]
    assert service.route_state("CONT_PRIMARY")["state"] == "OPEN"

    later_envelope = _envelope(
        cycle_id="00000000-0000-4000-8000-000000000099",
        candidate_id="candidate-after-health",
    )
    service.open_envelope(later_envelope)
    with pytest.raises(ModelUsageAdmissionError, match="unresolved usage"):
        service.allocate(
            _allocation(later_envelope, policy, request="after-health"),
            owner_emergency_stop=False,
        )
    service.reconcile(
        invocation_id=allocation.invocation_id,
        components=UsageComponents(
            input_tokens=100,
            output_tokens=25,
            total_tokens=125,
            provenance="PROVIDER_REPORTED",
        ),
        provider_telemetry={"late": "canonical-route"},
        observed_at=T0 + timedelta(minutes=1),
        raw_telemetry_pointer="private://late/canonical-route",
    )
    service.allocate(
        _allocation(later_envelope, policy, request="after-health"),
        owner_emergency_stop=False,
    )


def test_usage_windows_are_sliced_by_terminal_event_time(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _envelope_value, _policy_value, allocation = _open_and_allocate(service)
    service.complete(
        _reported(
            allocation,
            total=125,
            completed_at=T0 + timedelta(minutes=1, seconds=10),
        )
    )

    first = service.report(start=T0, end=T0 + timedelta(minutes=1))
    second = service.report(
        start=T0 + timedelta(minutes=1),
        end=T0 + timedelta(minutes=2),
    )

    assert first["observed_total_tokens"] == 0
    assert first["fixed_buckets"][0]["observed_total_tokens"] == 0
    assert second["observed_total_tokens"] == 125
    assert second["fixed_buckets"][0]["observed_total_tokens"] == 125
    assert second["utc_day_totals"] == {"2026-08-24": 125}


def test_work_outcome_is_visible_in_its_own_cross_bucket_window(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    envelope, _policy_value, allocation = _open_and_allocate(service)
    service.complete(
        _reported(
            allocation,
            completed_at=T0 + timedelta(minutes=4, seconds=59),
            outcome="ACCEPTED_OUTPUT",
        )
    )
    service.record_work_outcome(
        envelope_id=envelope.envelope_id,
        outcome="ACCEPTED",
        outcome_record_id="draft-cross-window",
        payload_digest=_digest({"payload": "cross-window"}),
        terminal_at=T0 + timedelta(minutes=5, seconds=1),
    )

    second = service.report(
        start=T0 + timedelta(minutes=5),
        end=T0 + timedelta(minutes=10),
    )
    buckets = list(
        csv.DictReader(
            io.StringIO(
                service.export_bucket_csv(
                    start=T0 + timedelta(minutes=5),
                    end=T0 + timedelta(minutes=10),
                )
            )
        )
    )
    assert second["accepted_payload_count"] == 1
    assert buckets[0]["minted_reported"] == "1"
    short_rows = list(
        csv.DictReader(
            io.StringIO(
                service.export_bucket_csv(
                    start=T0,
                    end=T0 + timedelta(minutes=5),
                )
            )
        )
    )
    long_rows = list(
        csv.DictReader(
            io.StringIO(
                service.export_bucket_csv(
                    start=T0,
                    end=T0 + timedelta(minutes=10),
                )
            )
        )
    )
    assert short_rows[0]["productive_tokens"] == long_rows[0]["productive_tokens"]
    assert short_rows[0]["no_result_tokens"] == long_rows[0]["no_result_tokens"]
    assert short_rows[0]["productive_tokens"] == "0"
    assert short_rows[0]["no_result_tokens"] == "125"


def test_deterministic_csv_export_reconciles_leaf_count_and_uncertainty(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    _envelope_value, _policy_value, allocation = _open_and_allocate(service)
    service.complete(_reported(allocation))

    first = service.export_csv(start=T0, end=T0 + timedelta(minutes=1))
    second = service.export_csv(start=T0, end=T0 + timedelta(minutes=1))
    rows = list(csv.DictReader(io.StringIO(first)))

    assert first == second
    assert len(rows) == 1
    assert rows[0]["invocation_id"] == allocation.invocation_id
    assert rows[0]["usage_status"] == "REPORTED"
    assert rows[0]["uncertainty"] == ""
    assert (
        service.report(start=T0, end=T0 + timedelta(minutes=1))[
            "leaf_dispatch_count_reconciles"
        ]
        is True
    )
    bucket_rows = list(
        csv.DictReader(
            io.StringIO(
                service.export_bucket_csv(
                    start=T0,
                    end=T0 + timedelta(minutes=10),
                )
            )
        )
    )
    assert len(bucket_rows) == 2
    assert bucket_rows[0]["grok_model_calls"] == "1"
    assert bucket_rows[0]["grok_total_tokens"] == "125"
    assert bucket_rows[1]["grok_total_tokens"] == "0"


def test_estimate_requires_the_separately_qualified_hard_ceiling(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    policy = _policy(hard_estimate_ceiling_tokens=None)
    _envelope_value, _policy_value, allocation = _open_and_allocate(
        service, policy=policy
    )

    with pytest.raises(ModelUsageIntegrityError, match="bounded estimate evidence"):
        service.complete(
            InvocationTerminal.create(
                invocation_id=allocation.invocation_id,
                outcome="TRANSPORT_LOST",
                failure_class="MISSING_TELEMETRY",
                usage_status=UsageStatus.ESTIMATED,
                components=UsageComponents(
                    total_tokens=policy.max_total_tokens,
                    provenance="BOUNDED_ESTIMATE",
                ),
                dispatch_at=T0 + timedelta(seconds=1),
                completed_at=T0 + timedelta(seconds=2),
                observed_at=T0 + timedelta(seconds=2),
                subscription_cli_chat_not_cash_debited=True,
                estimate_policy_digest=policy.canonical_digest,
                estimate_calculation="unproved policy maximum",
            )
        )


def test_graphiti_provider_observer_persists_chat_and_embedding_leaves(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    envelope = _envelope(
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        candidate_id=None,
        ingest_id="ingest-observed",
    )
    service.open_envelope(envelope)
    for policy in (
        InvocationEfficiencyPolicy.create(
            policy_id="graphiti-chat-primary",
            version="v1",
            workload_class=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            provider="cursor-agent-cli",
            route=GRAPHITI_CHAT_PRIMARY_ROUTE,
            model=CURSOR_AGENT_MODEL_ID,
            reasoning="provider-default",
            one_turn=True,
            exact_input=True,
            skills_enabled=False,
            tools_enabled=False,
            mcp_enabled=False,
            prior_message_count=0,
            max_prompt_bytes=4_096,
            max_context_tokens=2_000,
            max_output_tokens=500,
            max_total_tokens=2_500,
            prompt_contract_version=GRAPHITI_PROMPT_COMPONENT.component_version,
            output_schema_digest=_digest({"response_schema": "UNSTRUCTURED"}),
            allowed_context_identities=(GRAPHITI_CONTEXT_IDENTITY,),
            allowed_config_identities=("graphiti-cli-command-v1",),
            hard_estimate_ceiling_tokens=2_500,
            evidence_digest=_digest({"graphiti": "chat"}),
            qualified=True,
        ),
        InvocationEfficiencyPolicy.create(
            policy_id="graphiti-embedding",
            version="v1",
            workload_class=WorkloadClass.GRAPHITI_EMBEDDING,
            provider="openrouter",
            route=GRAPHITI_EMBEDDING_ROUTE,
            model=OPENROUTER_EMBEDDING_SLUG,
            reasoning="none",
            one_turn=True,
            exact_input=True,
            skills_enabled=False,
            tools_enabled=False,
            mcp_enabled=False,
            prior_message_count=0,
            max_prompt_bytes=4_096,
            max_context_tokens=2_000,
            max_output_tokens=500,
            max_total_tokens=2_500,
            prompt_contract_version=GRAPHITI_PROMPT_COMPONENT.component_version,
            output_schema_digest=_digest(
                {"schema": "embedding-vector", "model": OPENROUTER_EMBEDDING_SLUG}
            ),
            allowed_context_identities=(GRAPHITI_CONTEXT_IDENTITY,),
            allowed_config_identities=("graphiti-embedding-command-v1",),
            hard_estimate_ceiling_tokens=2_500,
            evidence_digest=_digest({"graphiti": "embedding"}),
            qualified=True,
        ),
    ):
        service.register_policy(policy)
    observer = GraphitiModelUsageObserver(
        service=service,
        envelope=envelope,
        clock=lambda: T0 + timedelta(seconds=10),
        owner_stop_check=lambda: None,
    )
    chat = observer.before_cli_invocation(
        provider="cursor-agent-cli",
        model=CURSOR_AGENT_MODEL_ID,
        prompt="chat prompt",
        schema=None,
    )
    observer.transport_dispatch_started(chat)
    observer.after_cli_invocation(
        chat,
        outcome="COMPLETE",
        usage={
            "usage_basis": "PROVIDER_REPORTED",
            "input_tokens": 10,
            "output_tokens": 2,
            "cached_read_tokens": 0,
            "cached_write_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 12,
        },
    )
    embedding = observer.before_embedding_invocation(
        provider="openrouter",
        model=OPENROUTER_EMBEDDING_SLUG,
        input_data=["embedding input"],
    )
    observer.transport_dispatch_started(embedding)
    observer.after_embedding_invocation(
        embedding,
        outcome="COMPLETE",
        usage={
            "usage_basis": "PROVIDER_REPORTED",
            "input_tokens": 4,
            "output_tokens": 0,
            "cached_read_tokens": 0,
            "cached_write_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 4,
            "provider_telemetry": {"request_id": "embedding-1", "total_tokens": 4},
        },
    )

    leaves = service.query(start=T0, end=T0 + timedelta(minutes=1))["leaves"]

    assert [leaf["workload_class"] for leaf in leaves] == [
        "GRAPHITI_CHAT_PRIMARY",
        "GRAPHITI_EMBEDDING",
    ]
    assert [leaf["total_tokens"] for leaf in leaves] == [12, 4]


def test_graphiti_missing_telemetry_without_a_hard_bound_is_unreported(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    envelope = _envelope(
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        candidate_id=None,
        ingest_id="ingest-unbounded",
    )
    service.open_envelope(envelope)
    service.register_policy(
        InvocationEfficiencyPolicy.create(
            policy_id="graphiti-chat-unbounded",
            version="v1",
            workload_class=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            provider="cursor-agent-cli",
            route=GRAPHITI_CHAT_PRIMARY_ROUTE,
            model=CURSOR_AGENT_MODEL_ID,
            reasoning="provider-default",
            one_turn=True,
            exact_input=True,
            skills_enabled=False,
            tools_enabled=False,
            mcp_enabled=False,
            prior_message_count=0,
            max_prompt_bytes=4_096,
            max_context_tokens=2_000,
            max_output_tokens=500,
            max_total_tokens=2_500,
            prompt_contract_version=GRAPHITI_PROMPT_COMPONENT.component_version,
            output_schema_digest=_digest({"response_schema": "UNSTRUCTURED"}),
            allowed_context_identities=(GRAPHITI_CONTEXT_IDENTITY,),
            allowed_config_identities=("graphiti-cli-command-v1",),
            hard_estimate_ceiling_tokens=None,
            evidence_digest=_digest({"graphiti": "unbounded"}),
            qualified=True,
        )
    )
    observer = GraphitiModelUsageObserver(
        service=service,
        envelope=envelope,
        clock=lambda: T0 + timedelta(seconds=10),
        owner_stop_check=lambda: None,
    )
    token = observer.before_cli_invocation(
        provider="cursor-agent-cli",
        model=CURSOR_AGENT_MODEL_ID,
        prompt="chat prompt",
        schema=None,
    )
    observer.transport_dispatch_started(token)
    observer.after_cli_invocation(
        token,
        outcome="TRANSPORT_LOST",
        usage={"usage_basis": "UNAVAILABLE"},
    )

    leaf = service.query(start=T0, end=T0 + timedelta(minutes=1))["leaves"][0]
    assert leaf["usage_status"] == "UNREPORTED"
    assert leaf["total_tokens"] is None
    assert service.route_state(GRAPHITI_CHAT_PRIMARY_ROUTE)["state"] == "OPEN"


def service_row_count(path: Path, table: str) -> int:
    connection = sqlite3.connect(path)
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        connection.close()


def test_hermes_usage_command_exports_shared_receipts_as_json_and_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import hermes_control_plane as hermes

    path = tmp_path / "unpublished.sqlite3"
    service = ModelUsageService(str(path))
    _envelope_value, _policy_value, allocation = _open_and_allocate(service)
    service.complete(_reported(allocation))
    monkeypatch.setattr(hermes, "ensure_control_plane_state_root", lambda: None)
    common = [
        "usage",
        "--unpublished",
        str(path),
        "--usage-start",
        "2026-08-24T10:00:00Z",
        "--usage-end",
        "2026-08-24T10:01:00Z",
    ]

    assert hermes.main(common) == 0
    json_report = json.loads(capsys.readouterr().out)
    assert json_report["schema_version"] == "newsroom.model-usage.v5"
    assert json_report["leaf_dispatch_count"] == 1
    assert json_report["observed_total_tokens"] == 125

    assert hermes.main([*common, "--usage-format", "leaf-csv"]) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [row["invocation_id"] for row in rows] == [allocation.invocation_id]
    assert rows[0]["schema_version"] == "newsroom.model-usage.v5"
    assert rows[0]["allocation_schema_version"] == "newsroom.model-usage.v3"


def test_hermes_usage_command_exports_allocation_free_envelope_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import hermes_control_plane as hermes

    path = tmp_path / "unpublished.sqlite3"
    service = ModelUsageService(str(path))
    envelope = _envelope()
    service.open_envelope(envelope)
    service.record_work_outcome(
        envelope_id=envelope.envelope_id,
        outcome="HOLD",
        outcome_record_id="owner-stop-outcome",
        payload_digest=None,
        terminal_at=T0 + timedelta(seconds=1),
        stable_reason_codes=("OWNER_EMERGENCY_STOP",),
    )
    monkeypatch.setattr(hermes, "ensure_control_plane_state_root", lambda: None)
    common = [
        "usage",
        "--unpublished",
        str(path),
        "--usage-start",
        "2026-08-24T10:00:00Z",
        "--usage-end",
        "2026-08-24T10:01:00Z",
    ]

    assert hermes.main(common) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == "newsroom.model-usage.v5"
    assert report["leaf_dispatch_count"] == 0
    assert report["envelope_outcome_counts"] == {"HOLD": 1}
    assert report["envelopes"][0]["outcome"] == "HOLD"
    assert report["envelopes"][0]["stable_reason_codes"] == [
        "OWNER_EMERGENCY_STOP"
    ]

    assert hermes.main([*common, "--usage-format", "envelope-csv"]) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(rows) == 1
    assert rows[0]["schema_version"] == "newsroom.model-usage.v3"
    assert rows[0]["envelope_id"] == envelope.envelope_id
    assert rows[0]["outcome"] == "HOLD"
    assert rows[0]["work_outcome_terminal_at"] == "2026-08-24T10:00:01.000000Z"
    assert rows[0]["stable_reason_codes"] == '["OWNER_EMERGENCY_STOP"]'


def test_committed_graphiti_transport_marker_is_cycle_dispatch_truth(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    envelope = _envelope(
        cycle_id="00000000-0000-4000-8000-000000000790",
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        candidate_id=None,
        ingest_id="ingest-790",
    )
    policy = _policy(
        workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        provider="cursor-agent-cli",
        route=GRAPHITI_CHAT_PRIMARY_ROUTE,
        model=CURSOR_AGENT_MODEL_ID,
    )
    _envelope_value, _policy_value, allocation = _open_and_allocate(
        service, envelope=envelope, policy=policy
    )

    assert service.has_committed_provider_dispatch(cycle_id=envelope.cycle_id) is False
    service.observe_transport(
        invocation_id=allocation.invocation_id,
        observed_at=T0 + timedelta(seconds=2),
        state="DISPATCH_STARTED",
        evidence_digest=_digest({"dispatch": allocation.invocation_id}),
    )
    assert service.has_committed_provider_dispatch(cycle_id=envelope.cycle_id) is True
    assert service.has_committed_provider_dispatch(cycle_id="different-cycle") is False


@pytest.mark.parametrize("hint", [False, True])
def test_native_landing_shared_body_decodes_on_both_accounting_read_paths(tmp_path, hint):
    connection = connect_unpublished_store(str(tmp_path / "private.sqlite3"))
    first = replace(_native("selected"), chunk_count=2)
    second = replace(first, chunk_ordinal=2, predecessor_ingest_id=first.ingest_id)
    NativeRevisionJournal(connection).land((first, second))
    digest = _digest(asdict(second.effective_revision)) if hint else None
    assert model_usage_module._native_landed_source_unit(
        connection, ingest_id=second.ingest_id, effective_revision_digest=digest,
    ) == second
    connection.close()


def test_native_accounting_rejects_mixed_shared_body_encoding(tmp_path):
    connection = connect_unpublished_store(str(tmp_path / "private.sqlite3"))
    first = replace(_native("selected"), chunk_count=2)
    second = replace(first, chunk_ordinal=2, predecessor_ingest_id=first.ingest_id)
    # The encoding is malformed even though its ledger hash was correctly made.
    append_ledger(connection, "NATIVE_REVISION_LANDED", {
        "revision_id": first.revision_id, "shared_body": first.body,
        "units": [asdict(first), asdict(second)],
    })
    connection.commit()
    with pytest.raises(ModelUsageIntegrityError, match="source landing differs"):
        model_usage_module._native_landed_source_unit(
            connection, ingest_id=second.ingest_id,
            effective_revision_digest=_digest(asdict(second.effective_revision)),
        )
    connection.close()
