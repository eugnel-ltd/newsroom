"""Controller-owned, append-only model invocation usage accounting.

The service owns the durable relationship between one qualified unit of work and
every provider leaf dispatched for it.  Aggregate token views are telemetry, not
normal production quotas; missing usage is always represented explicitly.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, TypeGuard

from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
    validate_sha256_digest,
)
from newsroom.control_plane.cycle_governor import CONT_WRITER_ROUTE
from newsroom.control_plane.graphiti_requests import (
    GraphitiInternalRequestIdentity,
    GraphitiLeafClass,
    load_checked_graphiti_call_shape_policy,
)
from newsroom.control_plane.issue_790_step16_activation import (
    effective_issue_790_plan_contract,
)
from newsroom.control_plane.sqlite_profile import apply_control_plane_sqlite_profile
from newsroom.control_plane.store import GRAPHITI_MAX_FAILURES
from newsroom.control_plane.veto import assert_private_store
from newsroom.control_plane import model_usage_current

if TYPE_CHECKING:
    from newsroom.control_plane.corpus import CorpusIngestUnit

MODEL_USAGE_SCHEMA_VERSION = "newsroom.model-usage.v3"
MODEL_USAGE_INTERFACE_SCHEMA_VERSION = "newsroom.model-usage.v5"
MODEL_USAGE_MIGRATION_ID = "model-usage-v5-reported-output-disposition"
REPORTED_OUTPUT_DISPOSITION_SCHEMA = "newsroom.model-usage.reported-output-disposition.v1"
REPORTED_SUBSCRIPTION_OVERRUN_SCHEMA = "newsroom.model-usage.reported-output-disposition.v2"
_REPORTED_OUTPUT_DISPOSITION_SCOPES = {
    REPORTED_OUTPUT_DISPOSITION_SCHEMA: "NATIVE_REPORTED_OUTPUT_ONLY_CANDIDATE_FAILURE",
    REPORTED_SUBSCRIPTION_OVERRUN_SCHEMA: "NATIVE_REPORTED_SUBSCRIPTION_BUDGET_CANDIDATE_FAILURE",
}
CONSERVATIVE_DISPOSITION_SCHEMA_VERSION = (
    "newsroom.model-usage.conservative-disposition.v2"
)
CONSERVATIVE_DISPOSITION_AUTHORITY_SCHEMA_VERSION = (
    "newsroom.model-usage.conservative-disposition-authority.v2"
)
NATIVE_CONSERVATIVE_DISPOSITION_AUTHORITY_SCHEMA_VERSION = (
    "newsroom.model-usage.native-conservative-disposition-authority.v1"
)
NATIVE_AUTONOMOUS_USAGE_SCOPE = "NATIVE_AUTONOMOUS_INTERNAL_PIPELINE"
NATIVE_EMBEDDING_TIMEOUT_DISPOSITION_AUTHORITY_SCHEMA_VERSION = (
    "newsroom.model-usage.native-embedding-timeout-disposition-authority.v1"
)
NATIVE_EMBEDDING_TIMEOUT_USAGE_SCOPE = (
    "NATIVE_AUTONOMOUS_OPENROUTER_EMBEDDING_TIMEOUT"
)
NATIVE_GRAPHITI_FALLBACK_CANCELLATION_USAGE_SCOPE = (
    "NATIVE_AUTONOMOUS_GRAPHITI_FALLBACK_CANCELLATION"
)
NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE = (
    "NATIVE_AUTONOMOUS_GRAPHITI_EMBEDDING_CANCELLATION"
)
_MODEL_USAGE_MIGRATIONS = (
    ("model-usage-v1", "newsroom.model-usage.v1"),
    ("model-usage-v2", "newsroom.model-usage.v2"),
    ("model-usage-v3", "newsroom.model-usage.v3"),
    ("model-usage-v4-conservative-disposition", "newsroom.model-usage.v4"),
    (MODEL_USAGE_MIGRATION_ID, MODEL_USAGE_INTERFACE_SCHEMA_VERSION),
)
_HERMETIC_CONT_CONFIG_IDENTITIES = frozenset(
    {
        "cont-writer-grok-hermetic-command-v2",
        "cont-writer-grok-hermetic-command-v3",
        "cont-writer-cursor-hermetic-command-v2",
        "native-evidence-assessor-grok-hermetic-command-v1",
    }
)
DAILY_USAGE_ALERT_TOKENS = 500_000


class ModelUsageIntegrityError(ValueError):
    """Retained model-usage evidence is malformed or contradictory."""


class ModelUsageAdmissionError(RuntimeError):
    """A leaf failed deterministic pre-dispatch admission."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "MODEL_USAGE_ADMISSION_HELD",
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class WorkloadClass(StrEnum):
    TYPESAFE_JUDGMENT = "TYPESAFE_JUDGMENT"
    NATIVE_EVIDENCE_ASSESSOR = "NATIVE_EVIDENCE_ASSESSOR"
    NATIVE_STORY_WRITER = "NATIVE_STORY_WRITER"
    NATIVE_RETRIEVAL_EMBEDDING = "NATIVE_RETRIEVAL_EMBEDDING"
    CONT_WRITER_PRIMARY = "CONT_WRITER_PRIMARY"
    CONT_WRITER_FALLBACK = "CONT_WRITER_FALLBACK"
    CONT_ROUTE_HEALTH_PROBE = "CONT_ROUTE_HEALTH_PROBE"
    GRAPHITI_CHAT_PRIMARY = "GRAPHITI_CHAT_PRIMARY"
    GRAPHITI_CHAT_FALLBACK = "GRAPHITI_CHAT_FALLBACK"
    GRAPHITI_EMBEDDING = "GRAPHITI_EMBEDDING"


def _nullable_native_output(
    workload: WorkloadClass, provider: str, route: str,
) -> bool:
    return (
        provider == "grok-build-cli"
        and (
            (workload is WorkloadClass.NATIVE_EVIDENCE_ASSESSOR
             and route in {"NATIVE_EVIDENCE_ASSESSOR", "NATIVE_CLAIM_LOCALISATION",
                           "NATIVE_SOURCE_QUALIFICATION"})
            or (workload is WorkloadClass.NATIVE_STORY_WRITER
                and route in {"NATIVE_STORY_DRAFT", "NATIVE_STORY_REVIEW"})
        )
    )


class UsageStatus(StrEnum):
    REPORTED = "REPORTED"
    ESTIMATED = "ESTIMATED"
    UNREPORTED = "UNREPORTED"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"


def native_graphiti_usage_cycle_id(*, ingest_id: str, attempt_number: int) -> str:
    """Stable native attempt identity, shared by admission and retained proof."""
    return digest_canonical({
        "namespace": "native-graphiti-model-usage-attempt-v1",
        "ingest_id": ingest_id,
        "attempt_number": attempt_number,
    })


@dataclass(frozen=True, slots=True)
class GraphitiIngestRetryEvidence:
    """Settled model-use evidence grouped by durable Graphiti attempt."""

    attempt_numbers: tuple[int, ...]
    zero_dispatch_attempts: tuple[int, ...]
    settled_provider_attempts: tuple[int, ...]
    latest_settled_provider_attempt: int | None
    unresolved_attempts: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = (
            self.attempt_numbers,
            self.zero_dispatch_attempts,
            self.settled_provider_attempts,
            self.unresolved_attempts,
        )
        if any(
            values != tuple(sorted(set(values)))
            or any(type(value) is not int or value <= 0 for value in values)
            for values in groups
        ):
            raise ModelUsageIntegrityError("Graphiti retry attempt evidence differs")
        attempts = set(self.attempt_numbers)
        classified = (
            set(self.zero_dispatch_attempts)
            | set(self.settled_provider_attempts)
            | set(self.unresolved_attempts)
        )
        zero = set(self.zero_dispatch_attempts)
        settled = set(self.settled_provider_attempts)
        unresolved = set(self.unresolved_attempts)
        if (
            not classified <= attempts
            or zero & settled
            or zero & unresolved
            or settled & unresolved
        ):
            raise ModelUsageIntegrityError("Graphiti retry classification differs")
        expected_latest = (
            self.settled_provider_attempts[-1]
            if self.settled_provider_attempts
            else None
        )
        if self.latest_settled_provider_attempt != expected_latest:
            raise ModelUsageIntegrityError("Graphiti retry latest attempt differs")


_GRAPHITI_COMPLETED_USEFUL_OUTCOMES = frozenset(
    {"GRAPHITI_SUCCESS", "GRAPHITI_SUCCESS_ZERO_PROPOSALS"}
)
_UNRESOLVED_USAGE_STATUSES = frozenset(
    {"UNREPORTED", "AMBIGUOUS", "INVALID"}
)
_GRAPHITI_CIRCUIT_ROUTES = (
    WorkloadClass.GRAPHITI_CHAT_PRIMARY.value,
    WorkloadClass.GRAPHITI_CHAT_FALLBACK.value,
    WorkloadClass.GRAPHITI_EMBEDDING.value,
)


_PROVENANCE = frozenset(
    {"PROVIDER_REPORTED", "CLI_DERIVED", "BOUNDED_ESTIMATE", "UNAVAILABLE"}
)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_usage_migrations(
    migration_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_invocation_policies(
    canonical_digest TEXT PRIMARY KEY,
    policy_id TEXT NOT NULL,
    version TEXT NOT NULL,
    workload_class TEXT NOT NULL,
    provider TEXT NOT NULL,
    route TEXT NOT NULL,
    model TEXT NOT NULL,
    qualified INTEGER NOT NULL CHECK(qualified IN (0,1)),
    record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_work_envelopes(
    envelope_id TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL,
    workload_class TEXT NOT NULL,
    admitted_at TEXT NOT NULL,
    canonical_digest TEXT NOT NULL UNIQUE,
    record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_zero_call_admissions(
    decision_id TEXT PRIMARY KEY,
    decision TEXT NOT NULL CHECK(decision IN ('HOLD','REJECT')),
    cycle_id TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_invocation_context_manifests(
    context_manifest_digest TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    route TEXT NOT NULL,
    evidence_package_digest TEXT NOT NULL,
    record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_invocation_context_observations(
    observation_digest TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL UNIQUE,
    context_manifest_digest TEXT NOT NULL,
    provider_context_tokens INTEGER,
    record_json TEXT NOT NULL,
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY(context_manifest_digest)
        REFERENCES model_invocation_context_manifests(context_manifest_digest)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_invocation_allocations(
    invocation_id TEXT PRIMARY KEY,
    envelope_id TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    leaf_ordinal INTEGER NOT NULL CHECK(leaf_ordinal > 0),
    workload_class TEXT NOT NULL,
    policy_digest TEXT NOT NULL,
    provider TEXT NOT NULL,
    route TEXT NOT NULL,
    model TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    parent_invocation_id TEXT,
    allocated_at TEXT NOT NULL,
    canonical_digest TEXT NOT NULL UNIQUE,
    record_json TEXT NOT NULL,
    UNIQUE(envelope_id, leaf_ordinal),
    UNIQUE(envelope_id, request_digest),
    FOREIGN KEY(envelope_id) REFERENCES model_work_envelopes(envelope_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY(policy_digest) REFERENCES model_invocation_policies(canonical_digest)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY(parent_invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_transport_observations(
    observation_digest TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    state TEXT NOT NULL,
    evidence_digest TEXT NOT NULL,
    record_json TEXT NOT NULL,
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_invocation_provider_attempt_links(
    link_digest TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL UNIQUE,
    provider_attempt_id TEXT NOT NULL,
    linked_at TEXT NOT NULL,
    record_json TEXT NOT NULL,
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_invocation_terminals(
    terminal_digest TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL UNIQUE,
    usage_status TEXT NOT NULL,
    outcome TEXT NOT NULL,
    failure_class TEXT,
    completed_at TEXT NOT NULL,
    record_json TEXT NOT NULL,
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_provider_telemetry(
    telemetry_record_digest TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL,
    provider_telemetry_digest TEXT NOT NULL,
    record_json TEXT NOT NULL,
    UNIQUE(invocation_id, provider_telemetry_digest),
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_work_outcomes(
    outcome_digest TEXT PRIMARY KEY,
    envelope_id TEXT NOT NULL UNIQUE,
    outcome TEXT NOT NULL,
    terminal_at TEXT NOT NULL,
    record_json TEXT NOT NULL,
    FOREIGN KEY(envelope_id) REFERENCES model_work_envelopes(envelope_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_usage_cycle_outcomes(
    cycle_digest TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL UNIQUE,
    outcome_class TEXT NOT NULL,
    terminal_at TEXT NOT NULL,
    record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_usage_reconciliations(
    reconciliation_digest TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    record_json TEXT NOT NULL,
    UNIQUE(invocation_id, reconciliation_digest),
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_usage_conservative_dispositions(
    disposition_digest TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL UNIQUE,
    terminal_digest TEXT NOT NULL UNIQUE,
    allocation_digest TEXT NOT NULL,
    policy_digest TEXT NOT NULL,
    approved_plan_digest TEXT NOT NULL UNIQUE,
    authority_digest TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    approval_reference TEXT NOT NULL,
    approved_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    usage_status TEXT NOT NULL CHECK(usage_status='ESTIMATED'),
    record_json TEXT NOT NULL,
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY(terminal_digest) REFERENCES model_invocation_terminals(terminal_digest)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY(allocation_digest)
        REFERENCES model_invocation_allocations(canonical_digest)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY(policy_digest) REFERENCES model_invocation_policies(canonical_digest)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_usage_reported_output_dispositions(
    invocation_id TEXT PRIMARY KEY,
    disposition_digest TEXT NOT NULL UNIQUE,
    record_json TEXT NOT NULL,
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_usage_route_circuit_events(
    event_digest TEXT PRIMARY KEY,
    route TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('OPEN','CLOSED')),
    reason TEXT NOT NULL,
    invocation_id TEXT,
    recorded_at TEXT NOT NULL,
    record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS graphiti_internal_requests(
    canonical_digest TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL UNIQUE,
    envelope_id TEXT NOT NULL,
    graphiti_attempt_id TEXT NOT NULL,
    internal_ordinal INTEGER NOT NULL CHECK(internal_ordinal > 0),
    semantic_state_digest TEXT NOT NULL,
    provider_attempt_id TEXT NOT NULL,
    call_shape_policy_digest TEXT NOT NULL,
    record_json TEXT NOT NULL,
    UNIQUE(graphiti_attempt_id, semantic_state_digest),
    UNIQUE(graphiti_attempt_id, internal_ordinal),
    UNIQUE(provider_attempt_id),
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY(envelope_id) REFERENCES model_work_envelopes(envelope_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS graphiti_internal_request_refusals(
    refusal_digest TEXT PRIMARY KEY,
    envelope_id TEXT NOT NULL,
    attempted_ordinal INTEGER NOT NULL CHECK(attempted_ordinal > 0),
    route TEXT NOT NULL,
    semantic_state_digest TEXT NOT NULL,
    call_shape_policy_digest TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    refused_at TEXT NOT NULL,
    record_json TEXT NOT NULL,
    FOREIGN KEY(envelope_id) REFERENCES model_work_envelopes(envelope_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS model_usage_allocation_cycle
ON model_invocation_allocations(cycle_id);
CREATE INDEX IF NOT EXISTS model_usage_allocation_record_envelope
ON model_invocation_allocations(json_extract(record_json, '$.envelope_id'));
CREATE INDEX IF NOT EXISTS model_usage_allocation_record_cycle
ON model_invocation_allocations(json_extract(record_json, '$.cycle_id'));
CREATE INDEX IF NOT EXISTS graphiti_request_envelope
ON graphiti_internal_requests(envelope_id);
CREATE INDEX IF NOT EXISTS graphiti_refusal_envelope
ON graphiti_internal_request_refusals(envelope_id);
CREATE INDEX IF NOT EXISTS model_usage_allocated_at
ON model_invocation_allocations(allocated_at, invocation_id);
CREATE INDEX IF NOT EXISTS model_usage_completed_at
ON model_invocation_terminals(completed_at, invocation_id);
CREATE INDEX IF NOT EXISTS model_usage_native_graphiti_ingest
ON model_work_envelopes(json_extract(record_json, '$.ingest_id'), envelope_id)
WHERE workload_class='GRAPHITI_CHAT_PRIMARY';
CREATE INDEX IF NOT EXISTS model_usage_transport_invocation
ON model_transport_observations(invocation_id, observed_at, observation_digest);
CREATE INDEX IF NOT EXISTS model_usage_route_state
ON model_usage_route_circuit_events(route, recorded_at);
CREATE INDEX IF NOT EXISTS graphiti_request_effective_revision
ON graphiti_internal_requests(json_extract(record_json, '$.effective_revision_digest'), invocation_id);
"""


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ModelUsageIntegrityError("timestamp must be timezone-aware")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ModelUsageIntegrityError("retained timestamp lacks timezone")
    return parsed.astimezone(UTC)


def _token(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value.encode("utf-8")) > 512
    ):
        raise ModelUsageIntegrityError(f"{field} must be bounded canonical text")
    return value


def _canonical_circuit_route(route: str) -> str:
    return (
        CONT_WRITER_ROUTE
        if route.startswith("CONT_") and route != "CONT_HEALTH_PROBE"
        else route
    )


def _usage_blocking_routes(connection: sqlite3.Connection) -> set[str]:
    """Read only current active/unsettled facts, never re-prove settled history."""
    try:
        return model_usage_current.blocking_routes(connection)
    except model_usage_current.CurrentUsageIntegrityError as exc:
        raise ModelUsageIntegrityError(str(exc)) from exc


def _refresh_current_usage(connection: sqlite3.Connection, invocation_id: str) -> None:
    try:
        model_usage_current.refresh(connection, invocation_id)
    except model_usage_current.CurrentUsageIntegrityError as exc:
        raise ModelUsageIntegrityError(str(exc)) from exc


def _current_has_active(connection: sqlite3.Connection, route: str) -> bool:
    try:
        return model_usage_current.has_active(connection, route)
    except model_usage_current.CurrentUsageIntegrityError as exc:
        raise ModelUsageIntegrityError(str(exc)) from exc


def _policy_for_allocation(
    connection: sqlite3.Connection,
    allocation: InvocationAllocation,
) -> InvocationEfficiencyPolicy:
    row = connection.execute(
        "SELECT canonical_digest,policy_id,version,workload_class,provider,route,"
        "model,qualified,record_json FROM model_invocation_policies "
        "WHERE canonical_digest=?",
        (allocation.invocation_policy_digest,),
    ).fetchone()
    if row is None:
        raise ModelUsageIntegrityError("retained Graphiti policy is absent")
    raw_record = str(row[8])
    retained_record = _object(raw_record)
    try:
        decoded = _policy_from_record(retained_record)
        values = asdict(decoded)
        values.pop("canonical_digest")
        policy = InvocationEfficiencyPolicy.create(**values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelUsageIntegrityError(
            "retained Graphiti policy binding differs"
        ) from exc
    expected_record = policy.as_record()
    if tuple(row[index] for index in range(8)) != (
        policy.canonical_digest,
        policy.policy_id,
        policy.version,
        policy.workload_class.value,
        policy.provider,
        policy.route,
        policy.model,
        int(policy.qualified),
    ) or (
        retained_record != expected_record
        or raw_record != _json(expected_record)
    ) or (
        policy.canonical_digest != allocation.invocation_policy_digest
        or policy.workload_class is not allocation.workload_class
        or policy.provider != allocation.provider
        or policy.route != allocation.route
        or policy.model != allocation.model
    ):
        raise ModelUsageIntegrityError("retained Graphiti policy binding differs")
    return policy


def _retained_terminal_allocation(
    connection: sqlite3.Connection, invocation_id: str
) -> tuple[InvocationAllocation, InvocationTerminal]:
    row = connection.execute(
        "SELECT t.terminal_digest,t.invocation_id,t.usage_status,t.outcome,"
        "t.failure_class,t.completed_at,t.record_json,a.invocation_id,"
        "a.envelope_id,a.cycle_id,a.leaf_ordinal,a.workload_class,a.policy_digest,"
        "a.provider,a.route,a.model,a.request_digest,a.parent_invocation_id,"
        "a.allocated_at,a.canonical_digest,a.record_json "
        "FROM model_invocation_terminals t JOIN model_invocation_allocations a "
        "ON a.invocation_id=t.invocation_id WHERE t.invocation_id=?",
        (invocation_id,),
    ).fetchone()
    if row is None:
        raise ModelUsageIntegrityError("retained invocation terminal is absent")
    terminal = _terminal_from_record(_object(row[6]))
    allocation = _allocation_from_record(_object(row[20]))
    if tuple(row[:6]) != (
        terminal.terminal_digest,
        terminal.invocation_id,
        terminal.usage_status.value,
        terminal.outcome,
        terminal.failure_class,
        _utc_text(terminal.completed_at),
    ) or tuple(row[7:20]) != (
        allocation.invocation_id,
        allocation.envelope_id,
        allocation.cycle_id,
        allocation.leaf_ordinal,
        allocation.workload_class.value,
        allocation.invocation_policy_digest,
        allocation.provider,
        allocation.route,
        allocation.model,
        allocation.request_digest,
        allocation.parent_invocation_id,
        _utc_text(allocation.allocated_at),
        allocation.canonical_digest,
    ):
        raise ModelUsageIntegrityError("retained invocation binding differs")
    return allocation, terminal


def native_sdk_reported_token_targets_are_advisory(policy: InvocationEfficiencyPolicy) -> bool:
    """Only the qualified native v13 SDK protocol separates consumption targets."""
    from newsroom.graphiti_adapter.cursor_transport import composer_model_meets_floor
    return (
        policy.qualified and not policy.calibration_only
        and policy.version == "issue-981-native-sdk-advisory-v1"
        and policy.workload_class is WorkloadClass.GRAPHITI_CHAT_PRIMARY
        and policy.provider == "cursor-agent-cli" and policy.route == "GRAPHITI_CHAT_PRIMARY"
        and policy.command_semantic_version == "newsroom.graphiti-provider-dispatch.v13"
        and composer_model_meets_floor(policy.model)
        and policy.allowed_context_identities == ("graphiti-combined-temporal-hermetic-v1",)
        and policy.allowed_config_identities == ("cursor-sdk-api-key-composer-floor-v2",)
        and policy.one_turn and policy.exact_input and policy.prior_message_count == 0
        and not (policy.skills_enabled or policy.tools_enabled or policy.mcp_enabled)
        and {"TRANSPORT=CURSOR_SDK", "fresh_run=TRUE", "resume=FALSE",
             "CONTROLLER_OUTPUT_CONTRACT=cursor-sdk-controller-output-v1:65536+64*REQUEST_MAX_TOKENS",
             "REPORTED_OUTPUT_TOKENS=ADVISORY_INCLUDES_REASONING_V1",
             "REPORTED_TOTAL_TOKENS=ADVISORY_CUMULATIVE_CONSUMPTION_V1"}.issubset(policy.command_flags)
    )


def _is_exact_pre_dispatch_zero(terminal: InvocationTerminal) -> bool:
    components = terminal.components
    return bool(
        terminal.usage_status is UsageStatus.REPORTED
        and terminal.pre_dispatch_zero_proved is True
        and terminal.dispatch_at is None
        and terminal.provider_telemetry_digest is None
        and terminal.raw_telemetry_pointer is None
        and terminal.policy_breach is None
        and components.provenance == "CLI_DERIVED"
        and components.total_tokens == 0
        and all(
            value in {None, 0}
            for value in (
                components.input_tokens,
                components.output_tokens,
                components.cached_read_tokens,
                components.cached_write_tokens,
                components.reasoning_tokens,
                components.context_tokens,
            )
        )
    )


def _has_exact_dispatch(
    connection: sqlite3.Connection, terminal: InvocationTerminal
) -> bool:
    rows = connection.execute(
        "SELECT observation_digest,observed_at,state,evidence_digest,record_json "
        "FROM model_transport_observations WHERE invocation_id=? "
        "ORDER BY observed_at,observation_digest",
        (terminal.invocation_id,),
    ).fetchall()
    matched = False
    for row in rows:
        record = _object(row[4])
        unsigned = dict(record)
        retained_digest = unsigned.pop("observation_digest", None)
        if (
            retained_digest != row[0]
            or digest_canonical(unsigned) != retained_digest
            or (
                record.get("invocation_id"),
                record.get("observed_at"),
                record.get("state"),
                record.get("evidence_digest"),
            )
            != (terminal.invocation_id, row[1], row[2], row[3])
        ):
            raise ModelUsageIntegrityError("retained transport observation differs")
        matched |= row[2] == "DISPATCH_STARTED"
    return matched


def _has_exact_native_embedding_dispatch(
    connection: sqlite3.Connection,
    terminal: InvocationTerminal,
    allocation: InvocationAllocation,
) -> bool:
    if not _has_exact_dispatch(connection, terminal):
        return False
    rows = connection.execute(
        "SELECT observed_at,evidence_digest FROM model_transport_observations "
        "WHERE invocation_id=? AND state='DISPATCH_STARTED'",
        (terminal.invocation_id,),
    ).fetchall()
    return len(rows) == 1 and tuple(rows[0]) == (
        _utc_text(terminal.dispatch_at),
        allocation.request_digest,
    )


def _require_reported_telemetry(
    connection: sqlite3.Connection, terminal: InvocationTerminal
) -> None:
    if terminal.components.provenance == "CLI_DERIVED":
        return
    rows = connection.execute(
        "SELECT telemetry_record_digest,invocation_id,provider_telemetry_digest,"
        "record_json FROM model_provider_telemetry WHERE invocation_id=?",
        (terminal.invocation_id,),
    ).fetchall()
    if len(rows) != 1:
        raise ModelUsageIntegrityError("reported provider telemetry differs")
    row = rows[0]
    record = _object(row[3])
    if (
        digest_canonical(record) != row[0]
        or (row[1], row[2])
        != (record.get("invocation_id"), record.get("provider_telemetry_digest"))
        or record.get("invocation_id") != terminal.invocation_id
        or record.get("provider_telemetry_digest")
        != terminal.provider_telemetry_digest
        or digest_canonical(record.get("provider_telemetry")) != row[2]
        or terminal.raw_telemetry_pointer is None
    ):
        raise ModelUsageIntegrityError("reported provider telemetry differs")


def _valid_native_disposition(
    connection: sqlite3.Connection,
    *,
    allocation: InvocationAllocation,
    terminal: InvocationTerminal,
    validated_native_envelope: WorkEnvelope | None = None,
) -> dict[str, object] | None:
    # Only bounded native retry reads supply an already authenticated envelope
    # from their proved native unit. Creation and generic reads still prove LAND.
    if validated_native_envelope is not None and (
        validated_native_envelope.envelope_id != allocation.envelope_id
        or validated_native_envelope.cycle_id != allocation.cycle_id
        or validated_native_envelope.workload_class is not WorkloadClass.GRAPHITI_CHAT_PRIMARY
    ):
        raise ModelUsageIntegrityError("native conservative envelope binding differs")
    retained_allocation, retained_terminal = _retained_terminal_allocation(
        connection, allocation.invocation_id
    )
    if (retained_allocation, retained_terminal) != (allocation, terminal):
        raise ModelUsageIntegrityError("retained invocation binding differs")
    row = connection.execute(
        "SELECT disposition_digest,terminal_digest,allocation_digest,policy_digest,"
        "approved_plan_digest,authority_digest,approved_by,approval_reference,"
        "approved_at,observed_at,usage_status,record_json "
        "FROM model_usage_conservative_dispositions WHERE invocation_id=?",
        (allocation.invocation_id,),
    ).fetchone()
    if row is None:
        return None
    record = _object(row[11])
    unsigned = dict(record)
    retained_digest = unsigned.pop("disposition_digest", None)
    authority_scope = record.get("authority_scope")
    if authority_scope not in {
        NATIVE_AUTONOMOUS_USAGE_SCOPE,
        NATIVE_EMBEDDING_TIMEOUT_USAGE_SCOPE,
        NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE,
        NATIVE_GRAPHITI_FALLBACK_CANCELLATION_USAGE_SCOPE,
    }:
        if row[6] in {
            NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE,
            NATIVE_GRAPHITI_FALLBACK_CANCELLATION_USAGE_SCOPE,
        }:
            raise ModelUsageIntegrityError("native cancellation disposition scope differs")
        return None
    policy = _policy_for_allocation(connection, allocation)
    if authority_scope == NATIVE_AUTONOMOUS_USAGE_SCOPE:
        envelope = validated_native_envelope or _native_envelope(
            connection, allocation
        )
        leaf_class = _native_conservative_subscription_leaf(allocation)
        if leaf_class is None:
            raise ModelUsageIntegrityError(
                "native conservative disposition target is ineligible"
            )
        if leaf_class is GraphitiLeafClass.FALLBACK:
            identity = _retained_graphiti_request_identity(connection, allocation)
            if (
                identity is None
                or identity.leaf_class is not GraphitiLeafClass.FALLBACK
                or identity.primary_unavailable_event_digest is None
            ):
                raise ModelUsageIntegrityError(
                    "native fallback request authority differs"
                )
            _require_native_failed_attempt_receipt(
                connection,
                allocation=allocation,
                terminal=terminal,
                envelope=envelope,
            )
        expected_scope = _native_disposition_authority(
            allocation=allocation,
            terminal=terminal,
            policy=policy,
            envelope=envelope,
        )
        conservative_total = policy.max_total_tokens
    elif authority_scope == NATIVE_GRAPHITI_FALLBACK_CANCELLATION_USAGE_SCOPE:
        if row[11] != _json(record):
            raise ModelUsageIntegrityError("native fallback cancellation is not canonical")
        expected_scope = _native_graphiti_fallback_cancellation_authority(
            connection, allocation=allocation, terminal=terminal, policy=policy,
        )
        conservative_total = policy.max_total_tokens
    elif authority_scope == NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE:
        if row[11] != _json(record):
            raise ModelUsageIntegrityError("native cancellation disposition is not canonical")
        expected_scope = _native_graphiti_embedding_cancellation_authority(
            connection, allocation=allocation, terminal=terminal, policy=policy,
        )
        conservative_total = max(policy.max_total_tokens, allocation.prompt_bytes)
    else:
        if (
            allocation.workload_class
            is not WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING
            or allocation.provider != "openrouter"
            or allocation.route != "NATIVE_RETRIEVAL_EMBEDDING"
            or allocation.model != "openai/text-embedding-3-large"
            or not policy.qualified
            or terminal.outcome != "NATIVE_EMBEDDING_FAILED"
            or terminal.failure_class != "TimeoutError"
            or terminal.usage_status is not UsageStatus.UNREPORTED
            or terminal.dispatch_at is None
            or terminal.policy_breach is not None
            or terminal.provider_telemetry_digest is not None
            or terminal.raw_telemetry_pointer is not None
            or terminal.pre_dispatch_zero_proved
            or terminal.subscription_cli_chat_not_cash_debited
            or terminal.od_011_reference
            != "OD-011:NATIVE_RETRIEVAL_EMBEDDING"
            or not _has_exact_native_embedding_dispatch(
                connection, terminal, allocation
            )
            or connection.execute(
                "SELECT 1 FROM model_provider_telemetry WHERE invocation_id=? "
                "UNION ALL SELECT 1 FROM model_usage_reconciliations "
                "WHERE invocation_id=? LIMIT 1",
                (allocation.invocation_id, allocation.invocation_id),
            ).fetchone()
            is not None
        ):
            raise ModelUsageIntegrityError(
                "native embedding conservative disposition is ineligible"
            )
        expected_scope = _native_embedding_timeout_disposition_authority(
            connection,
            allocation=allocation,
            terminal=terminal,
            policy=policy,
            retained_progress=record,
        )
        conservative_total = max(
            policy.max_total_tokens, allocation.prompt_bytes
        )
    scope_digest = digest_canonical(expected_scope)
    if authority_scope in {
        NATIVE_EMBEDDING_TIMEOUT_USAGE_SCOPE,
        NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE,
        NATIVE_GRAPHITI_FALLBACK_CANCELLATION_USAGE_SCOPE,
    } and any(
        record.get(key) != value for key, value in expected_scope.items()
    ):
        raise ModelUsageIntegrityError("native conservative disposition differs")
    if (
        retained_digest != row[0]
        or digest_canonical(unsigned) != retained_digest
        or tuple(row[index] for index in range(1, 11))
        != (
            terminal.terminal_digest,
            allocation.canonical_digest,
            policy.canonical_digest,
            scope_digest,
            scope_digest,
            authority_scope,
            authority_scope,
            record.get("observed_at"),
            record.get("observed_at"),
            UsageStatus.ESTIMATED.value,
        )
        or record.get("schema_version")
        != CONSERVATIVE_DISPOSITION_SCHEMA_VERSION
        or record.get("invocation_id") != allocation.invocation_id
        or record.get("terminal_digest") != terminal.terminal_digest
        or record.get("allocation_digest") != allocation.canonical_digest
        or record.get("policy_digest") != policy.canonical_digest
        or record.get("native_scope_digest") != scope_digest
        or record.get("authority_digest") != scope_digest
        or _instant(str(record.get("observed_at"))) < terminal.observed_at
        or record.get("usage_status") != UsageStatus.ESTIMATED.value
        or record.get("components")
        != UsageComponents(
            total_tokens=conservative_total,
            provenance="BOUNDED_ESTIMATE",
        ).as_record()
        or record.get("estimate_policy_digest") != policy.canonical_digest
        or record.get("estimate_calculation")
        != (
            "QUALIFIED_POLICY_MAX_TOTAL_TOKENS_CONSERVATIVE_UPPER_BOUND"
            if authority_scope in {NATIVE_AUTONOMOUS_USAGE_SCOPE, NATIVE_GRAPHITI_FALLBACK_CANCELLATION_USAGE_SCOPE}
            else "MAX_QUALIFIED_POLICY_TOTAL_OR_EXACT_REQUEST_UTF8_BYTES"
        )
        or record.get("exact_usage_remains_unknown") is not True
        or record.get("provider_dispatch_preserved") is not True
        or record.get("unknown_spend_released") is not False
    ):
        raise ModelUsageIntegrityError("native conservative disposition differs")
    return record


def _valid_native_embedding_timeout_disposition_record(
    connection: sqlite3.Connection,
    *,
    allocation_record: Mapping[str, object],
    terminal_record: Mapping[str, object],
    disposition_record: Mapping[str, object],
) -> bool:
    """Re-prove the exact native embedding disposition for qualification."""

    if (
        disposition_record.get("authority_scope")
        != NATIVE_EMBEDDING_TIMEOUT_USAGE_SCOPE
    ):
        return False
    try:
        allocation = _allocation_from_record(allocation_record)
        terminal = _terminal_from_record(terminal_record)
        retained = _valid_native_disposition(
            connection, allocation=allocation, terminal=terminal
        )
    except ModelUsageIntegrityError:
        return False
    return retained == dict(disposition_record)


def _valid_native_graphiti_embedding_cancellation_disposition_record(
    connection: sqlite3.Connection,
    *,
    allocation_record: Mapping[str, object],
    terminal_record: Mapping[str, object],
    disposition_record: Mapping[str, object],
) -> bool:
    if disposition_record.get("authority_scope") != NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE:
        return False
    try:
        retained = _valid_native_disposition(
            connection, allocation=_allocation_from_record(allocation_record),
            terminal=_terminal_from_record(terminal_record),
        )
    except ModelUsageIntegrityError:
        return False
    return retained == dict(disposition_record)


def _valid_native_graphiti_fallback_cancellation_disposition_record(
    connection: sqlite3.Connection,
    *,
    allocation_record: Mapping[str, object],
    terminal_record: Mapping[str, object],
    disposition_record: Mapping[str, object],
) -> bool:
    if disposition_record.get("authority_scope") != NATIVE_GRAPHITI_FALLBACK_CANCELLATION_USAGE_SCOPE:
        return False
    try:
        retained = _valid_native_disposition(
            connection, allocation=_allocation_from_record(allocation_record),
            terminal=_terminal_from_record(terminal_record),
        )
    except ModelUsageIntegrityError:
        return False
    return retained == dict(disposition_record)


def _reported_output_disposition_authority(
    connection: sqlite3.Connection, *, invocation_id: str, revision_id: str,
    schema_version: str = REPORTED_OUTPUT_DISPOSITION_SCHEMA,
) -> dict[str, object]:
    """Prove a settled native SDK output rejection, never reclassify its usage."""
    if type(schema_version) is not str or schema_version not in _REPORTED_OUTPUT_DISPOSITION_SCOPES:
        raise ModelUsageIntegrityError("reported output disposition version differs")
    subscription_overrun = schema_version == REPORTED_SUBSCRIPTION_OVERRUN_SCHEMA
    allocation, terminal = _retained_terminal_allocation(connection, invocation_id)
    policy = _policy_for_allocation(connection, allocation)
    components = terminal.components
    counters = ("input_tokens", "output_tokens", "cached_read_tokens", "cached_write_tokens", "total_tokens")
    if (
        _native_conservative_subscription_leaf(allocation) is not GraphitiLeafClass.PRIMARY
        or allocation.config_identity != "cursor-sdk-api-key-composer-floor-v2"
        or allocation.parent_invocation_id is not None
        or not policy.qualified or policy.calibration_only
        or terminal.outcome != "OUTPUT_LIMIT_EXCEEDED"
        or terminal.failure_class is not None
        or terminal.usage_status is not UsageStatus.REPORTED
        or terminal.policy_breach not in ({
            "REQUESTED_MAX_OUTPUT_TOKENS_EXCEEDED", "MAX_OUTPUT_TOKENS_EXCEEDED",
            "MAX_TOTAL_TOKENS_EXCEEDED",
        } if subscription_overrun else {"REQUESTED_MAX_OUTPUT_TOKENS_EXCEEDED"})
        or components.provenance != "PROVIDER_REPORTED"
        or any(type(getattr(components, name)) is not int for name in counters)
        or _invalid_reported_components(terminal) is not None
        or terminal.dispatch_at is None or terminal.pre_dispatch_zero_proved
        or terminal.subscription_cli_chat_not_cash_debited is not True
        or allocation.prompt_bytes > policy.max_prompt_bytes
        or allocation.max_output_tokens is None or policy.max_output_tokens is None
        or allocation.max_output_tokens > policy.max_output_tokens
        or (components.output_tokens <= allocation.max_output_tokens
            and not (subscription_overrun and components.total_tokens > policy.max_total_tokens))
        or components.total_tokens != (components.input_tokens + components.output_tokens
                                        + components.cached_read_tokens + components.cached_write_tokens)
        or (not subscription_overrun and components.total_tokens > policy.max_total_tokens)
        # Without a context measurement, require the entire reported run to
        # fit the context bound; missing context is not represented as zero.
        or (not subscription_overrun and components.context_tokens is None
            and components.total_tokens > policy.max_context_tokens)
        or (components.context_tokens is not None and components.context_tokens > policy.max_context_tokens)
        or (components.reasoning_tokens is not None and components.reasoning_tokens > components.output_tokens)
        or allocation.context_identity not in policy.allowed_context_identities
        or allocation.config_identity not in policy.allowed_config_identities
        or any(getattr(allocation, key) != getattr(policy, key) for key in (
            "reasoning", "prompt_contract_version", "output_schema_digest", "one_turn",
            "exact_input", "skills_enabled", "tools_enabled", "mcp_enabled", "prior_message_count",
        ))
    ):
        raise ModelUsageAdmissionError("reported output-only disposition is ineligible")
    # SDK usage is cumulative consumption, not a measured context window.
    # v1 remains frozen; v2 isolates an accounted failed subscription unit,
    # without pretending its prompt targets were enforced by the provider.
    if subscription_overrun and ModelUsageService._validate_terminal(
        terminal, allocation.workload_class, policy,
        requested_max_output_tokens=allocation.max_output_tokens,
    ) != terminal.policy_breach:
        raise ModelUsageIntegrityError("reported subscription overrun policy differs")
    if not _has_exact_dispatch(connection, terminal):
        raise ModelUsageIntegrityError("reported output disposition lacks exact dispatch")
    dispatches = connection.execute(
        "SELECT observed_at,evidence_digest FROM model_transport_observations "
        "WHERE invocation_id=? AND state='DISPATCH_STARTED'", (invocation_id,),
    ).fetchall()
    if len(dispatches) != 1 or tuple(dispatches[0]) != (_utc_text(terminal.dispatch_at), digest_canonical({
        "invocation_id": invocation_id, "provider": allocation.provider,
        "route": allocation.route, "request_digest": allocation.request_digest,
    })):
        raise ModelUsageIntegrityError("reported output dispatch request differs")
    _require_reported_telemetry(connection, terminal)
    telemetry = _object(connection.execute(
        "SELECT record_json FROM model_provider_telemetry WHERE invocation_id=?", (invocation_id,),
    ).fetchone()[0])["provider_telemetry"]
    if (not isinstance(telemetry, dict) or telemetry.get("usage_basis") != "PROVIDER_REPORTED"
            or any(telemetry.get(name) != getattr(components, name)
                   for name in (*counters, "reasoning_tokens", "context_tokens"))):
        raise ModelUsageIntegrityError("reported output telemetry components differ")
    identity = _retained_graphiti_request_identity(connection, allocation)
    if identity is None or identity.leaf_class is not GraphitiLeafClass.PRIMARY:
        raise ModelUsageIntegrityError("reported output request identity differs")
    envelope = _native_envelope(connection, allocation, revision_id_hint=revision_id)
    attempt = int(str(envelope.graphiti_attempt_id).rpartition(":")[2])
    if envelope.cycle_id != native_graphiti_usage_cycle_id(ingest_id=envelope.ingest_id, attempt_number=attempt):
        raise ModelUsageIntegrityError("reported output is outside native work")
    unit = _native_landed_source_unit(
        connection, ingest_id=envelope.ingest_id,
        effective_revision_digest=identity.effective_revision_digest, revision_id_hint=revision_id,
    )
    if unit is None or unit.proving_run_id != "native-source:" + unit.observation_digest:
        raise ModelUsageIntegrityError("reported output source binding differs")
    outcome_digest, receipt_digest = _require_native_failed_attempt_receipt(
        connection, allocation=allocation, terminal=terminal, envelope=envelope,
    )
    outcome = _object(connection.execute(
        "SELECT record_json FROM model_work_outcomes WHERE envelope_id=?", (envelope.envelope_id,),
    ).fetchone()[0])
    receipt = _object(connection.execute(
        "SELECT receipt_json FROM unpublished_graphiti_attempt_receipts WHERE ingest_id=? AND attempt_number=?",
        (envelope.ingest_id, attempt),
    ).fetchone()[0])
    bound = [leaf for leaf in receipt.get("chat_invocations", [])
             if isinstance(leaf, dict) and leaf.get("model_invocation_id") == invocation_id]
    spends = connection.execute(
        "SELECT spend_id,status,actual_usd_microunits,actual_gbp_microunits FROM unpublished_graphiti_spend "
        "WHERE ingest_id=? AND attempt_number=?", (envelope.ingest_id, attempt),
    ).fetchall()
    accounting = receipt.get("accounting", {})
    if isinstance(accounting, dict) and accounting.get("status") != "RECONCILED":
        raise ModelUsageAdmissionError("reported output attempt accounting is unsettled")
    sdk = bound[0].get("sdk_terminal") if len(bound) == 1 else None
    if not isinstance(sdk, dict):
        raise ModelUsageAdmissionError("reported output lacks a finished SDK receipt")
    leaf_usage = bound[0].get("usage")
    if (not isinstance(leaf_usage, dict)
            or leaf_usage.get("usage_basis") != "PROVIDER_REPORTED"
            or digest_canonical(leaf_usage.get("provider_telemetry", leaf_usage)) != terminal.provider_telemetry_digest
            or any(name in leaf_usage and leaf_usage[name] != getattr(components, name)
                   for name in (*counters, "reasoning_tokens", "context_tokens"))):
        raise ModelUsageIntegrityError("reported output receipt usage differs")
    from newsroom.graphiti_adapter.cursor_transport import _diagnostic_digest
    sdk_digest = sdk.get("diagnostic_digest")
    if sdk_digest != _diagnostic_digest(**{key: sdk.get(key) for key in (
        "status", "error_class", "error_code", "tool_call_count", "cancelled", "duration_ms",
    )}):
        raise ModelUsageIntegrityError("reported output SDK receipt digest differs")
    if (sdk.get("schema_version") != "newsroom.cursor-sdk-terminal.v1"
            or sdk.get("status") != "finished" or sdk.get("cancelled") is not False
            or sdk.get("error_class") != "NONE" or sdk.get("error_code") != "NONE"
            or type(sdk.get("tool_call_count")) is not int or sdk.get("tool_call_count") != 0
            or sdk.get("resolved_model") != allocation.model
            or any(type(sdk.get(key)) is not str or sdk.get(key) in {"", "UNOBSERVED"}
                   for key in ("agent_id", "run_id"))):
        raise ModelUsageAdmissionError("reported output SDK completion is ineligible")
    if (
        outcome.get("outcome") != "GRAPHITI_FAILED"
        or receipt.get("failure_code") != "PRODUCER_INTERNAL_ERROR"
        or len(bound) != 1
        or bound[0].get("outcome") != "OUTPUT_LIMIT_EXCEEDED"
        or bound[0].get("provider") != allocation.provider
        or bound[0].get("model") != allocation.model
        or type(bound[0].get("requested_max_tokens")) is not int
        or bound[0].get("requested_max_tokens") != allocation.max_output_tokens
        or bound[0].get("sdk_run_id") != sdk.get("run_id")
        or bound[0].get("sdk_agent_id") != sdk.get("agent_id")
        or bound[0].get("model_work_envelope_id") != envelope.envelope_id
        or bound[0].get("model_invocation_allocation_digest") != allocation.canonical_digest
        or bound[0].get("model_invocation_terminal_digest") != terminal.terminal_digest
        or not isinstance(accounting, dict) or len(spends) != 1
        or accounting.get("unused_reservation_released") is not True
        or any(type(accounting.get(key)) is not int for key in ("actual_usd_microunits", "actual_gbp_microunits"))
        or tuple(spends[0]) != (accounting.get("spend_id"), "RECONCILED",
                                accounting.get("actual_usd_microunits"), accounting.get("actual_gbp_microunits"))
        or accounting.get("status") != "RECONCILED"
    ):
        raise ModelUsageIntegrityError("reported output failed-attempt closure differs")
    return {
        "schema_version": schema_version,
        "authority_scope": _REPORTED_OUTPUT_DISPOSITION_SCOPES[schema_version],
        "invocation_id": invocation_id, "route": allocation.route,
        "terminal_outcome": terminal.outcome, "allocation_digest": allocation.canonical_digest,
        "terminal_digest": terminal.terminal_digest, "policy_digest": policy.canonical_digest,
        "envelope_digest": envelope.canonical_digest, "internal_request_digest": identity.canonical_digest,
        "provider_telemetry_digest": terminal.provider_telemetry_digest,
        "sdk_terminal_digest": sdk_digest,
        "revision_id": unit.revision_id, "ingest_id": envelope.ingest_id,
        "landed_unit_digest": digest_canonical(asdict(unit)),
        "work_outcome_digest": outcome_digest, "attempt_receipt_digest": receipt_digest,
        "failure_settled_at": outcome["terminal_at"],
        "policy_breach": terminal.policy_breach, "usage_status": terminal.usage_status.value,
        "components": components.as_record(), "requested_max_output_tokens": allocation.max_output_tokens,
        "maximum_context_tokens": policy.max_context_tokens, "maximum_total_tokens": policy.max_total_tokens,
        "failed_attempt_preserved": True, "retry_authorised": False, "unknown_spend_released": False,
    }


_ASSESSOR_REQUALIFICATION_KIND = "NATIVE_ASSESSOR_INPUT_REQUALIFICATION"
_ASSESSOR_OUTPUT_REQUALIFICATION_KIND = "NATIVE_ASSESSOR_OUTPUT_GUARD_REQUALIFICATION"


def _assessor_requalification_authority(
    connection, invocation_id, qualified_policy_digest, *, output_guard=False,
):
    """Authenticate one exact assessor correction without changing old evidence."""
    from .native_assessor import (
        _V15_PRODUCER_VERSION, _V15_SYSTEM, _V15_SCHEMA_DIGEST,
        _V19_PRODUCER_VERSION, _V19_SYSTEM, _V19_PROVIDER_SCHEMA_DIGEST,
        _ASSESSMENT_RESULT_SCHEMA_VERSION, native_assessment_input_bound,
        _MAX_RETAINED_RESULT_BYTES,
    )

    allocation, terminal = _retained_terminal_allocation(connection, invocation_id)
    old = _policy_for_allocation(connection, allocation)
    new = _policy_for_allocation(connection, replace(
        allocation, invocation_policy_digest=qualified_policy_digest,
        model="grok-4.7" if output_guard else allocation.model,
    ))
    # v20 deliberately retains the frozen v19 wire instructions and schema.
    bound = native_assessment_input_bound(
        replace(new, prompt_contract_version=_V19_PRODUCER_VERSION)
        if output_guard else new
    )
    components = terminal.components
    unchanged = (
        "workload_class", "provider", "route", "one_turn", "exact_input",
        "skills_enabled", "tools_enabled", "mcp_enabled", "prior_message_count",
        "disabled_capabilities", "max_context_tokens", "max_total_tokens",
        "allowed_context_identities", "allowed_config_identities",
        "hard_estimate_ceiling_tokens",
    )
    if (
        allocation.workload_class is not WorkloadClass.NATIVE_EVIDENCE_ASSESSOR
        or allocation.provider != "grok-build-cli"
        or allocation.route != "NATIVE_EVIDENCE_ASSESSOR"
        or allocation.config_identity != "native-evidence-assessor-grok-hermetic-command-v1"
        or allocation.parent_invocation_id is not None or allocation.leaf_ordinal != 1
        or not old.qualified or not new.qualified or old.calibration_only or new.calibration_only
        or any(getattr(old, key) != getattr(new, key) for key in unchanged)
        or terminal.outcome != "ASSESSOR_VALIDATION_FAILED"
        or terminal.failure_class != "ASSESSMENT_VALIDATION_FAILED"
        or terminal.usage_status is not UsageStatus.REPORTED
        or components.provenance != "PROVIDER_REPORTED"
        or _invalid_reported_components(terminal) is not None
        or terminal.dispatch_at is None or terminal.pre_dispatch_zero_proved
        or terminal.subscription_cli_chat_not_cash_debited is not True
        or allocation.max_output_tokens is None or old.max_output_tokens is None
    ):
        raise ModelUsageAdmissionError("native assessor requalification is ineligible")
    if output_guard:
        from .writer import _grok_command_flags

        counters = (
            "input_tokens", "output_tokens", "cached_read_tokens", "cached_write_tokens",
            "reasoning_tokens", "context_tokens", "total_tokens",
        )
        if (
            (old.model, old.reasoning) != ("grok-4.6", "medium")
            or (new.model, new.reasoning) != ("grok-4.7", "high")
            or old.prompt_contract_version != _V19_PRODUCER_VERSION
            or new.prompt_contract_version != "newsroom.native-evidence-assessor.v20"
            or old.output_schema_digest != _V19_PROVIDER_SCHEMA_DIGEST
            or new.output_schema_digest != _V19_PROVIDER_SCHEMA_DIGEST
            or old.context_manifest_schema_version != "newsroom.native-evidence-assessor.context-manifest.v3"
            or old.context_manifest_schema_version != new.context_manifest_schema_version
            or old.command_flags != _grok_command_flags("medium", model="grok-4.6")
            or new.command_flags != _grok_command_flags("high", model="grok-4.7")
            or old.command_semantic_version != new.command_semantic_version
            or old.max_output_tokens != 10_000 or allocation.max_output_tokens != 10_000
            or new.max_output_tokens is not None
            or old.max_context_tokens != 100_000 or old.max_total_tokens != 100_000
            or not 0 < new.max_prompt_bytes <= old.max_prompt_bytes
            or new.max_prompt_bytes != 56_464
            or bound["max_request_bytes"] != new.max_prompt_bytes
            or allocation.prompt_bytes > native_assessment_input_bound(old)["max_request_bytes"]
            or terminal.policy_breach not in {
                "REQUESTED_MAX_OUTPUT_TOKENS_EXCEEDED", "MAX_OUTPUT_TOKENS_EXCEEDED",
            }
            or any(type(getattr(components, key)) is not int for key in counters)
            or components.output_tokens <= old.max_output_tokens
            or components.context_tokens > old.max_context_tokens
            or components.total_tokens > old.max_total_tokens
            or components.reasoning_tokens > components.output_tokens
            or components.total_tokens != (components.input_tokens + components.output_tokens
                                           + components.cached_read_tokens + components.cached_write_tokens)
            or allocation.context_identity not in old.allowed_context_identities
            or allocation.config_identity not in old.allowed_config_identities
            or any(getattr(allocation, key) != getattr(old, key) for key in (
                "reasoning", "prompt_contract_version", "output_schema_digest", "one_turn",
                "exact_input", "skills_enabled", "tools_enabled", "mcp_enabled", "prior_message_count",
            ))
            or not old.one_turn or not old.exact_input
            or old.skills_enabled or old.tools_enabled or old.mcp_enabled or old.prior_message_count != 0
        ):
            raise ModelUsageAdmissionError("native assessor output-guard requalification is ineligible")
    elif (
        old.context_manifest_schema_version != "newsroom.native-evidence-assessor.context-manifest.v1"
        or new.context_manifest_schema_version != "newsroom.native-evidence-assessor.context-manifest.v2"
        or any(getattr(old, key) != getattr(new, key) for key in (
            "model", "reasoning", "command_flags", "max_output_tokens", "prompt_contract_version", "output_schema_digest",
        ))
        or old.prompt_contract_version != _V15_PRODUCER_VERSION or old.output_schema_digest != _V15_SCHEMA_DIGEST
        or not 0 < new.max_prompt_bytes < old.max_prompt_bytes
        or new.max_prompt_bytes != bound["max_request_bytes"]
        or allocation.prompt_bytes <= bound["max_request_bytes"]
        or allocation.prompt_bytes > old.max_prompt_bytes
        or terminal.policy_breach != "MAX_TOTAL_TOKENS_EXCEEDED"
        or type(components.total_tokens) is not int or components.total_tokens <= old.max_total_tokens
        or type(components.context_tokens) is not int or components.context_tokens <= old.max_context_tokens
        or type(components.output_tokens) is not int
        or not components.output_tokens <= allocation.max_output_tokens <= old.max_output_tokens
    ):
        raise ModelUsageAdmissionError("native assessor input requalification is ineligible")
    _require_reported_telemetry(connection, terminal)
    telemetry = _object(connection.execute(
        "SELECT record_json FROM model_provider_telemetry WHERE invocation_id=?", (invocation_id,),
    ).fetchone()[0])["provider_telemetry"]
    if (not isinstance(telemetry, dict) or telemetry.get("usage_basis") != "PROVIDER_REPORTED"
            or any(telemetry.get(key) != getattr(components, key) for key in (
                "input_tokens", "output_tokens", "cached_read_tokens", "cached_write_tokens",
                "reasoning_tokens", "context_tokens", "total_tokens",
            ))):
        raise ModelUsageIntegrityError("assessor requalification telemetry differs")
    if not _has_exact_dispatch(connection, terminal):
        raise ModelUsageIntegrityError("assessor requalification dispatch is absent")
    dispatches = connection.execute(
        "SELECT observed_at,evidence_digest FROM model_transport_observations "
        "WHERE invocation_id=? AND state='DISPATCH_STARTED'", (invocation_id,),
    ).fetchall()
    if len(dispatches) != 1 or tuple(dispatches[0]) != (_utc_text(terminal.dispatch_at), allocation.request_digest):
        raise ModelUsageIntegrityError("assessor requalification dispatch differs")
    row = connection.execute(
        "SELECT envelope_id,cycle_id,workload_class,admitted_at,canonical_digest,record_json "
        "FROM model_work_envelopes WHERE envelope_id=?", (allocation.envelope_id,),
    ).fetchone()
    if row is None:
        raise ModelUsageIntegrityError("assessor requalification candidate envelope is absent")
    envelope = _envelope_from_record(_object(row[5]))
    if (tuple(row[:5]) != (envelope.envelope_id, envelope.cycle_id, envelope.workload_class.value,
                          _utc_text(envelope.admitted_at), envelope.canonical_digest)
            or envelope.envelope_id != allocation.envelope_id
            or row[5] != _json(envelope.as_record()) or envelope.cycle_id != allocation.cycle_id
            or envelope.workload_class is not allocation.workload_class
            or not envelope.candidate_id or not envelope.hypothesis_digest or not envelope.evidence_package_digest
            or connection.execute("SELECT count(*) FROM model_invocation_allocations WHERE envelope_id=?", (allocation.envelope_id,)).fetchone()[0] != 1):
        raise ModelUsageIntegrityError("assessor requalification candidate envelope differs")
    row = connection.execute(
        "SELECT context_manifest_digest,provider,route,evidence_package_digest,record_json "
        "FROM model_invocation_context_manifests WHERE context_manifest_digest=?",
        (allocation.context_manifest_digest,),
    ).fetchone()
    if row is None:
        raise ModelUsageIntegrityError("assessor requalification context is absent")
    manifest = _object(row[4])
    unsigned = dict(manifest); manifest_digest = unsigned.pop("context_manifest_digest", None)
    if (tuple(row[:4]) != (manifest_digest, allocation.provider, allocation.route, envelope.evidence_package_digest)
            or row[4] != _json(manifest) or digest_canonical(unsigned) != manifest_digest
            or manifest_digest != allocation.context_manifest_digest
            or manifest.get("schema_version") != old.context_manifest_schema_version
            or manifest.get("system_digest") != digest_bytes((_V19_SYSTEM if output_guard else _V15_SYSTEM).encode())
            or manifest.get("evidence_package_digest") != envelope.evidence_package_digest
            or any(manifest.get(key) != getattr(allocation, key) for key in (
                "provider", "route", "model", "reasoning", "prompt_bytes", "prompt_digest",
                "request_digest", "output_schema_digest", "prompt_contract_version",
            ))):
        raise ModelUsageIntegrityError("assessor requalification context differs")
    if output_guard and (
        manifest.get("input_bound") != native_assessment_input_bound(old)
        or manifest.get("schema_digest") != old.output_schema_digest
        or manifest.get("implementation_worktree_clean") is not True
        or manifest.get("implementation_revision") != old.implementation_revision
        or manifest.get("command_flags") != list(old.command_flags)
        or manifest.get("disabled_capabilities") != list(old.disabled_capabilities)
        or any(manifest.get(key) != getattr(allocation, key) for key in (
            "context_identity", "config_identity", "one_turn", "exact_input", "skills_enabled",
            "tools_enabled", "mcp_enabled", "prior_message_count",
        ))
        or any(type(manifest.get(key)) is not int or manifest[key] != 0 for key in (
            "skill_count", "tool_count", "mcp_server_count", "mcp_tool_count",
        ))
        or digest_canonical({key: manifest.get(key) for key in (
            "provider", "route", "model", "reasoning", "command_semantic_version",
            "command_flags", "implementation_revision", "system_digest", "prompt_digest", "output_schema_digest",
        )}) != allocation.request_digest
        or connection.execute(
            "SELECT 1 FROM ledger WHERE kind='NATIVE_ASSESSMENT_MATERIALISATION' "
            "AND json_extract(payload_json,'$.invocation_id')=?", (invocation_id,),
        ).fetchone()
    ):
        raise ModelUsageIntegrityError("assessor output requalification contract differs")
    results = connection.execute(
        "SELECT payload_digest,payload_json FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT' "
        "AND json_extract(payload_json,'$.invocation_id')=?", (invocation_id,),
    ).fetchall()
    if len(results) != 1:
        raise ModelUsageIntegrityError("assessor requalification failed result is absent")
    result = _object(results[0][1]); text = result.get("result_text")
    if (results[0][1] != _json(result) or digest_bytes(results[0][1].encode()) != results[0][0]
            or result.get("schema_version") != _ASSESSMENT_RESULT_SCHEMA_VERSION
            or result.get("invocation_id") != invocation_id
            or result.get("allocation_digest") != allocation.canonical_digest
            or result.get("invocation_policy_digest") != old.canonical_digest
            or result.get("request_digest") != allocation.request_digest
            or result.get("retention_outcome") != "RETAINED" or type(text) is not str
            or len(text.encode()) > _MAX_RETAINED_RESULT_BYTES
            or result.get("result_bytes") != len(text.encode())
            or result.get("result_digest") != digest_bytes(text.encode())
            or _instant(str(result.get("dispatch_at"))) != terminal.dispatch_at
            or not terminal.dispatch_at <= _instant(str(result.get("observed_at"))) <= terminal.completed_at):
        raise ModelUsageIntegrityError("assessor requalification failed result differs")
    return {
        "schema_version": (
            "newsroom.native-assessor-output-guard-requalification.v1" if output_guard
            else "newsroom.native-assessor-input-requalification.v1"
        ),
        "invocation_id": invocation_id, "allocation_digest": allocation.canonical_digest,
        "terminal_digest": terminal.terminal_digest, "telemetry_digest": terminal.provider_telemetry_digest,
        "context_manifest_digest": allocation.context_manifest_digest,
        "envelope_digest": envelope.canonical_digest, "candidate_id": envelope.candidate_id,
        "failed_result_digest": results[0][0], "original_policy_digest": old.canonical_digest,
        "qualified_policy_digest": new.canonical_digest, "input_bound_digest": bound["bound_digest"],
        "implementation_revision": new.implementation_revision, "qualification_evidence_digest": new.evidence_digest,
        "failure_settled_at": _utc_text(terminal.completed_at), "failure_reason": terminal.policy_breach,
        "original_candidate_retry": False,
    }


def reported_output_rejected_ingests(
    connection: sqlite3.Connection, *, ingest_ids: tuple[str, ...] | None = None,
) -> frozenset[str]:
    """Read durable selected no-retry obligations without historical LAND replay."""
    if ingest_ids is not None and not ingest_ids:
        return frozenset()
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_usage_reported_output_dispositions'"
    ).fetchone() is None:
        return frozenset()
    selected = None if ingest_ids is None else frozenset(ingest_ids)
    result = set()
    # These are current no-retry reservations, not settled diagnostic history.
    # Authenticate compact selectors before filtering: damaged JSON must never
    # make an existing reservation disappear from its original ingest.
    for row in connection.execute(
        "SELECT d.invocation_id,d.disposition_digest,d.record_json,"
        "a.envelope_id,a.canonical_digest,a.record_json,"
        "e.envelope_id,e.cycle_id,e.workload_class,e.admitted_at,e.canonical_digest,e.record_json "
        "FROM model_usage_reported_output_dispositions d "
        "LEFT JOIN model_invocation_allocations a ON a.invocation_id=d.invocation_id "
        "LEFT JOIN model_work_envelopes e ON e.envelope_id=a.envelope_id"
    ):
        invocation_id, digest, raw = row[:3]
        record = _object(raw)
        unsigned = dict(record)
        retained_digest = unsigned.pop("disposition_digest", None)
        if (retained_digest != digest or digest_canonical(unsigned) != digest
                or record.get("invocation_id") != invocation_id
                or type(record.get("schema_version")) is not str
                or record.get("schema_version") not in _REPORTED_OUTPUT_DISPOSITION_SCOPES
                or record.get("authority_scope") != _REPORTED_OUTPUT_DISPOSITION_SCOPES.get(record.get("schema_version"))
                or record.get("retry_authorised") is not False or raw != _json(record)
                or any(value is None for value in row[3:])):
            raise ModelUsageIntegrityError("current no-retry disposition differs")
        allocation_record = _object(row[5])
        allocation = _allocation_from_record(allocation_record)
        envelope_record = _object(row[11])
        envelope = _envelope_from_record(envelope_record)
        if (
            (allocation.invocation_id, allocation.envelope_id, allocation.canonical_digest)
            != (invocation_id, row[3], row[4])
            or allocation.canonical_digest != record.get("allocation_digest")
            or _json(allocation_record) != row[5]
            or tuple(row[6:11]) != (envelope.envelope_id, envelope.cycle_id,
                envelope.workload_class.value, _utc_text(envelope.admitted_at), envelope.canonical_digest)
            or _json(envelope_record) != row[11]
            or envelope.canonical_digest != record.get("envelope_digest")
            or envelope.workload_class is not WorkloadClass.GRAPHITI_CHAT_PRIMARY
            or envelope.ingest_id != record.get("ingest_id")
        ):
            raise ModelUsageIntegrityError("current no-retry selection binding differs")
        ingest_id = _token(record.get("ingest_id"), field="current no-retry ingest")
        if selected is None or ingest_id in selected:
            result.add(ingest_id)
    return frozenset(result)


def _native_envelope(
    connection: sqlite3.Connection, allocation: InvocationAllocation,
    *, revision_id_hint: str | None = None,
) -> WorkEnvelope:
    row = connection.execute(
        "SELECT envelope_id,cycle_id,workload_class,admitted_at,canonical_digest,"
        "record_json FROM model_work_envelopes WHERE envelope_id=?",
        (allocation.envelope_id,),
    ).fetchone()
    if row is None:
        raise ModelUsageIntegrityError("native conservative envelope is absent")
    envelope = _envelope_from_record(_object(row[5]))
    if tuple(row[index] for index in range(5)) != (
        envelope.envelope_id,
        envelope.cycle_id,
        envelope.workload_class.value,
        _utc_text(envelope.admitted_at),
        envelope.canonical_digest,
    ) or (
        envelope.envelope_id != allocation.envelope_id
        or envelope.cycle_id != allocation.cycle_id
        or envelope.workload_class is not WorkloadClass.GRAPHITI_CHAT_PRIMARY
        or envelope.ingest_id is None
    ):
        raise ModelUsageIntegrityError("native conservative envelope binding differs")
    prefix, separator, suffix = str(envelope.graphiti_attempt_id or "").rpartition(":")
    if (
        separator != ":"
        or prefix != envelope.ingest_id
        or not suffix.isdigit()
        or int(suffix) <= 0
    ):
        raise ModelUsageIntegrityError("native conservative attempt binding differs")

    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ledger'"
    ).fetchone() is None:
        raise ModelUsageIntegrityError(
            "native conservative disposition lacks a landed source observation"
        )
    identity = _retained_graphiti_request_identity(connection, allocation)
    if identity is not None and (
        identity.envelope_id != envelope.envelope_id
        or identity.graphiti_attempt_id != envelope.graphiti_attempt_id
        or identity.ingest_obligation_id != envelope.ingest_id
    ):
        raise ModelUsageIntegrityError(
            "native conservative request binding differs"
        )
    if _native_landed_source_unit(
        connection,
        ingest_id=envelope.ingest_id, revision_id_hint=revision_id_hint,
        effective_revision_digest=(
            identity.effective_revision_digest if identity is not None else None
        ),
    ) is None:
        raise ModelUsageIntegrityError(
            "native conservative disposition lacks a landed source observation"
        )
    return envelope


def _native_conservative_subscription_leaf(
    allocation: InvocationAllocation,
) -> GraphitiLeafClass | None:
    return {
        (
            WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            WorkloadClass.GRAPHITI_CHAT_PRIMARY.value,
            "cursor-agent-cli",
        ): GraphitiLeafClass.PRIMARY,
        (
            WorkloadClass.GRAPHITI_CHAT_FALLBACK,
            WorkloadClass.GRAPHITI_CHAT_FALLBACK.value,
            "grok-build-cli",
        ): GraphitiLeafClass.FALLBACK,
    }.get((allocation.workload_class, allocation.route, allocation.provider))


def _require_native_failed_attempt_receipt(
    connection: sqlite3.Connection,
    *,
    allocation: InvocationAllocation,
    terminal: InvocationTerminal,
    envelope: WorkEnvelope,
    cancelled: bool = False,
) -> tuple[str, str]:
    outcome_row = connection.execute(
        "SELECT outcome_digest,envelope_id,outcome,terminal_at,record_json "
        "FROM model_work_outcomes WHERE envelope_id=?",
        (envelope.envelope_id,),
    ).fetchone()
    if outcome_row is None:
        raise ModelUsageIntegrityError("native fallback work outcome is absent")
    outcome = _object(outcome_row[4])
    unsigned_outcome = dict(outcome)
    outcome_digest = unsigned_outcome.pop("outcome_digest", None)
    attempt = int(str(envelope.graphiti_attempt_id).rpartition(":")[2])
    receipt_row = connection.execute(
        "SELECT ingest_id,attempt_number,outcome,receipt_digest,receipt_json FROM "
        "unpublished_graphiti_attempt_receipts "
        "WHERE ingest_id=? AND attempt_number=?",
        (envelope.ingest_id, attempt),
    ).fetchone()
    if receipt_row is None:
        raise ModelUsageIntegrityError("native fallback attempt receipt is absent")
    receipt = _object(receipt_row[4])
    unsigned_receipt = dict(receipt)
    receipt_digest = unsigned_receipt.pop("receipt_digest", None)
    invocations = receipt.get("chat_invocations")
    expected_outcomes = {"GRAPHITI_TIMEOUT"} if cancelled else {"GRAPHITI_FAILED", "GRAPHITI_REJECTED_BINDING"}
    expected_receipt = "TIMEOUT" if cancelled else "FAILED"
    if (
        outcome_digest != outcome_row[0]
        or digest_canonical(unsigned_outcome) != outcome_digest
        or tuple(outcome_row[1:4])
        != (
            outcome.get("envelope_id"),
            outcome.get("outcome"),
            outcome.get("terminal_at"),
        )
        or outcome.get("envelope_id") != envelope.envelope_id
        or outcome.get("outcome") not in expected_outcomes
        or _instant(str(outcome.get("terminal_at"))) < terminal.observed_at
        or outcome.get("outcome_record_id") != receipt_digest
        or tuple(receipt_row[:3]) != (envelope.ingest_id, attempt, expected_receipt)
        or (receipt.get("ingest_id"), receipt.get("attempt_number"), receipt.get("outcome"))
        != (envelope.ingest_id, attempt, expected_receipt)
        or receipt_digest != receipt_row[3]
        or digest_bytes(canonical_json_bytes(unsigned_receipt)) != receipt_digest
        or not isinstance(invocations, list)
        # The native result boundary can fail after retaining the request and
        # terminal but before returning leaf pointers. Their independent exact
        # bindings remain authoritative; missing outer telemetry is not zero.
        or (invocations and not any(
            isinstance(item, dict)
            and item.get("model_invocation_id") == allocation.invocation_id
            and item.get("model_invocation_terminal_digest") == terminal.terminal_digest
            for item in invocations
        ))
    ):
        raise ModelUsageIntegrityError("native fallback failure receipt differs")

    if cancelled:
        bound = [item for item in invocations if isinstance(item, dict)
                 and item.get("model_invocation_id") == allocation.invocation_id]
        if (
            outcome_row[4] != _json(outcome)
            or len(bound) != 1
            or bound[0].get("outcome") != "CANCELLED"
            or bound[0].get("model_work_envelope_id") != envelope.envelope_id
            or bound[0].get("model_invocation_allocation_digest") != allocation.canonical_digest
            or bound[0].get("model_invocation_terminal_digest") != terminal.terminal_digest
        ):
            raise ModelUsageIntegrityError("native fallback cancellation receipt differs")
    return str(outcome_digest), str(receipt_digest)


def _retained_graphiti_request_identity(
    connection: sqlite3.Connection,
    allocation: InvocationAllocation,
) -> GraphitiInternalRequestIdentity | None:
    row = connection.execute(
        "SELECT canonical_digest,invocation_id,envelope_id,graphiti_attempt_id,"
        "internal_ordinal,semantic_state_digest,provider_attempt_id,"
        "call_shape_policy_digest,record_json FROM graphiti_internal_requests "
        "WHERE invocation_id=?",
        (allocation.invocation_id,),
    ).fetchone()
    if row is None:
        return None
    record = _object(row[8])
    try:
        values = dict(record)
        values.pop("schema_version", None)
        values["leaf_class"] = GraphitiLeafClass(values["leaf_class"])
        identity = GraphitiInternalRequestIdentity.create(**values)
        ModelUsageService._validate_graphiti_identity(allocation, identity)
    except (KeyError, TypeError, ValueError, ModelUsageAdmissionError) as exc:
        raise ModelUsageIntegrityError(
            "native Graphiti request identity differs"
        ) from exc
    if row[8] != _json(identity.as_record()) or tuple(row[:8]) != (
        identity.canonical_digest,
        identity.invocation_id,
        identity.envelope_id,
        identity.graphiti_attempt_id,
        identity.internal_ordinal,
        identity.semantic_state_digest,
        identity.provider_attempt_id,
        identity.call_shape_policy_digest,
    ):
        raise ModelUsageIntegrityError("native Graphiti request identity differs")
    if identity.primary_unavailable_event_digest is not None:
        try:
            _require_direct_fallback_authority(connection, identity)
        except ModelUsageAdmissionError as exc:
            raise ModelUsageIntegrityError(str(exc)) from exc
    manifest_row = connection.execute(
        "SELECT context_manifest_digest,provider,route,evidence_package_digest,"
        "record_json FROM model_invocation_context_manifests "
        "WHERE context_manifest_digest=?",
        (allocation.context_manifest_digest,),
    ).fetchone()
    if manifest_row is None:
        raise ModelUsageIntegrityError("native Graphiti context manifest is absent")
    manifest = _object(manifest_row[4])
    unsigned = dict(manifest)
    retained_digest = unsigned.pop("context_manifest_digest", None)
    if (
        manifest_row[4] != _json(manifest)
        or retained_digest != allocation.context_manifest_digest
        or digest_canonical(unsigned) != retained_digest
        or tuple(manifest_row[:4]) != (
            retained_digest,
            allocation.provider,
            allocation.route,
            identity.effective_revision_digest,
        )
        or any(
            manifest.get(key) != getattr(allocation, key)
            for key in (
                "provider",
                "route",
                "model",
                "reasoning",
                "prompt_bytes",
                "prompt_digest",
                "request_digest",
                "output_schema_digest",
                "context_identity",
                "config_identity",
                "one_turn",
                "exact_input",
                "skills_enabled",
                "tools_enabled",
                "mcp_enabled",
                "prior_message_count",
            )
        )
        or any(
            manifest.get(key) != getattr(identity, key)
            for key in (
                "effective_revision_digest",
                "ingest_obligation_id",
                "graphiti_attempt_id",
                "provider_attempt_id",
                "semantic_state_digest",
                "call_shape_policy_digest",
                "dispatch_authority_digest",
            )
        )
    ):
        raise ModelUsageIntegrityError("native Graphiti context manifest differs")
    return identity


def _require_open_route_event(
    connection: sqlite3.Connection,
    event_digest: str,
    *, route: str = "GRAPHITI_CHAT_PRIMARY",
) -> dict[str, object]:
    row = connection.execute(
        "SELECT route,state,reason,invocation_id,recorded_at,record_json "
        "FROM model_usage_route_circuit_events WHERE event_digest=?",
        (event_digest,),
    ).fetchone()
    if row is None:
        raise ModelUsageAdmissionError(
            "Graphiti direct fallback primary authority is absent"
        )
    record = _object(row[5])
    unsigned = dict(record)
    retained_digest = unsigned.pop("event_digest", None)
    if (
        row[5] != _json(record)
        or retained_digest != event_digest
        or digest_canonical(unsigned) != event_digest
        or record.get("schema_version") != MODEL_USAGE_SCHEMA_VERSION
        or tuple(row[:5])
        != (
            record.get("route"),
            record.get("state"),
            record.get("reason"),
            record.get("invocation_id"),
            record.get("recorded_at"),
        )
        or record.get("route") != route
        or record.get("state") != "OPEN"
    ):
        raise ModelUsageAdmissionError(
            "Graphiti direct fallback primary authority differs"
        )
    return record


def _require_direct_fallback_authority(
    connection: sqlite3.Connection,
    identity: GraphitiInternalRequestIdentity,
) -> None:
    event_digest = identity.primary_unavailable_event_digest
    if event_digest is None:
        return
    _require_open_route_event(connection, event_digest)
    if (
        identity.leaf_class is not GraphitiLeafClass.FALLBACK
        or identity.parent_invocation_id is not None
    ):
        raise ModelUsageAdmissionError(
            "Graphiti direct fallback primary authority differs"
        )


def _native_landed_source_unit(
    connection: sqlite3.Connection,
    *,
    ingest_id: str,
    effective_revision_digest: str | None = None,
    revision_id_hint: str | None = None,
) -> CorpusIngestUnit | None:
    """Prove one governed unit without replaying unrelated native progress."""

    from newsroom.control_plane.native_progress import (
        LAND,
        NativeRevisionJournal,
        _landed_units,
    )

    def decode_landing(
        row: tuple[object, object],
    ) -> tuple[str, tuple[CorpusIngestUnit, ...]]:
        payload_digest, raw_value = row
        raw = str(raw_value)
        payload = _object(raw)
        try:
            units = _landed_units(payload)
            NativeRevisionJournal._validate_units(units)
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelUsageIntegrityError(
                "native conservative source landing differs"
            ) from exc
        if (
            raw != canonical_json_bytes(payload).decode("utf-8")
            or payload_digest != digest_bytes(raw.encode("utf-8"))
            or payload.get("revision_id") != units[0].revision_id
        ):
            raise ModelUsageIntegrityError(
                "native conservative source landing differs"
            )
        return units[0].revision_id, units

    if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='native_current_meta'").fetchone():
        from .native_progress_state import selected_landing_rows
        try:
            rows = selected_landing_rows(connection, ingest_id=ingest_id,
                revision_id_hint=revision_id_hint, effective_revision_digest=effective_revision_digest)
            matches = [unit for selected in rows for unit in decode_landing(selected)[1]
                       if unit.ingest_id == ingest_id and (effective_revision_digest is None
                           or digest_canonical(asdict(unit.effective_revision)) == effective_revision_digest)]
        except (KeyError, TypeError, ValueError) as exc:
            message = ('native conservative source landing changed' if 'source landing changed' in str(exc)
                       else 'native conservative source landing differs')
            raise ModelUsageIntegrityError(message) from exc
        return matches[0] if len(matches) == 1 else None

    candidate_revisions: set[str] = set()
    if revision_id_hint is not None:
        candidate_revisions.add(_token(revision_id_hint, field="native revision hint"))
    elif effective_revision_digest is not None:
        for revision_id, raw_revision in connection.execute(
            "SELECT json_extract(payload_json,'$.revision_id'),"
            "json_extract(unit.value,'$.effective_revision') FROM ledger "
            "JOIN json_each(payload_json,'$.units') AS unit WHERE kind=?",
            (LAND,),
        ):
            try:
                revision = json.loads(str(raw_revision))
            except (TypeError, ValueError):
                continue
            if (
                isinstance(revision, dict)
                and digest_canonical(revision) == effective_revision_digest
            ):
                candidate_revisions.add(str(revision_id))
    else:
        # Retained pre-request fixtures have no revision hint. Preserve their
        # exact validation path without making it the normal native route cost.
        for (raw_value,) in connection.execute(
            "SELECT payload_json FROM ledger WHERE kind=?", (LAND,)
        ):
            try:
                payload = json.loads(str(raw_value))
                units = _landed_units(payload)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            for unit in units:
                if unit.ingest_id == ingest_id:
                    candidate_revisions.add(str(payload.get("revision_id")))
    if not candidate_revisions:
        return None

    matches: list[CorpusIngestUnit] = []
    for revision_id in candidate_revisions:
        retained_units: tuple[CorpusIngestUnit, ...] | None = None
        for row in connection.execute(
            "SELECT payload_digest,payload_json FROM ledger WHERE kind=? "
            "AND json_extract(payload_json,'$.revision_id')=?",
            (LAND, revision_id),
        ):
            landed_revision_id, units = decode_landing(row)
            if landed_revision_id != revision_id:
                raise ModelUsageIntegrityError(
                    "native conservative source landing differs"
                )
            if retained_units is not None and retained_units != units:
                raise ModelUsageIntegrityError(
                    "native conservative source landing changed"
                )
            retained_units = units
        matches.extend(
            unit
            for unit in retained_units or ()
            if unit.ingest_id == ingest_id
            and (
                effective_revision_digest is None
                or digest_canonical(asdict(unit.effective_revision))
                == effective_revision_digest
            )
        )
    if len(matches) != 1:
        return None
    return matches[0]



def _native_graphiti_fallback_cancellation_authority(
    connection: sqlite3.Connection,
    *,
    allocation: InvocationAllocation,
    terminal: InvocationTerminal,
    policy: InvocationEfficiencyPolicy,
) -> dict[str, object]:
    """Prove one native subscription cancellation without changing circuit authority."""
    if (
        _native_conservative_subscription_leaf(allocation) is not GraphitiLeafClass.FALLBACK
        or not policy.qualified or policy.calibration_only
        or terminal.outcome != "CANCELLED"
        or terminal.failure_class != "MISSING_PROVIDER_TELEMETRY"
        or terminal.usage_status is not UsageStatus.UNREPORTED
        or terminal.components.total_tokens is not None
        or terminal.dispatch_at is None or terminal.policy_breach is not None
        or terminal.provider_telemetry_digest is not None
        or terminal.raw_telemetry_pointer is not None
        or terminal.pre_dispatch_zero_proved
        or terminal.subscription_cli_chat_not_cash_debited is not True
        or allocation.prompt_bytes > policy.max_prompt_bytes
        or allocation.max_output_tokens is None or policy.max_output_tokens is None
        or allocation.max_output_tokens > policy.max_output_tokens
        or allocation.context_identity not in policy.allowed_context_identities
        or allocation.config_identity not in policy.allowed_config_identities
        or any(getattr(allocation, key) != getattr(policy, key) for key in (
            "reasoning", "prompt_contract_version", "output_schema_digest",
            "one_turn", "exact_input", "skills_enabled", "tools_enabled",
            "mcp_enabled", "prior_message_count",
        ))
    ):
        raise ModelUsageIntegrityError("native fallback cancellation is ineligible")
    if not _has_exact_dispatch(connection, terminal):
        raise ModelUsageIntegrityError("native fallback cancellation lacks exact dispatch")
    dispatches = connection.execute(
        "SELECT observed_at,evidence_digest FROM model_transport_observations "
        "WHERE invocation_id=? AND state='DISPATCH_STARTED'",
        (allocation.invocation_id,),
    ).fetchall()
    if len(dispatches) != 1 or tuple(dispatches[0]) != (_utc_text(terminal.dispatch_at), digest_canonical({
        "invocation_id": allocation.invocation_id, "provider": allocation.provider,
        "route": allocation.route, "request_digest": allocation.request_digest,
    })):
        raise ModelUsageIntegrityError("native fallback cancellation dispatch binding differs")
    envelope = _native_envelope(connection, allocation)
    identity = _retained_graphiti_request_identity(connection, allocation)
    if (
        identity is None or identity.leaf_class is not GraphitiLeafClass.FALLBACK
        or identity.primary_unavailable_event_digest is None
    ):
        raise ModelUsageIntegrityError("native fallback cancellation request authority differs")
    unit = _native_landed_source_unit(
        connection, ingest_id=envelope.ingest_id,
        effective_revision_digest=identity.effective_revision_digest,
    )
    if unit is None or unit.proving_run_id != "native-source:" + unit.observation_digest:
        raise ModelUsageIntegrityError("native fallback cancellation lacks native source landing")
    outcome_digest, receipt_digest = _require_native_failed_attempt_receipt(
        connection, allocation=allocation, terminal=terminal, envelope=envelope, cancelled=True,
    )
    return {
        **{key: value for key, value in _native_disposition_authority(
            allocation=allocation, terminal=terminal, policy=policy, envelope=envelope,
        ).items() if key != "schema_version"},
        "authority_schema_version": NATIVE_CONSERVATIVE_DISPOSITION_AUTHORITY_SCHEMA_VERSION,
        "authority_scope": NATIVE_GRAPHITI_FALLBACK_CANCELLATION_USAGE_SCOPE,
        "envelope_digest": envelope.canonical_digest,
        "landed_unit_digest": digest_canonical(asdict(unit)),
        "internal_request_digest": identity.canonical_digest,
        "primary_unavailable_event_digest": identity.primary_unavailable_event_digest,
        "work_outcome_digest": outcome_digest,
        "attempt_receipt_digest": receipt_digest,
    }


def _native_graphiti_embedding_cancellation_authority(
    connection: sqlite3.Connection,
    *,
    allocation: InvocationAllocation,
    terminal: InvocationTerminal,
    policy: InvocationEfficiencyPolicy,
) -> dict[str, object]:
    """Re-prove a native cancelled leaf; no exact usage or cash is invented."""
    if (
        allocation.workload_class is not WorkloadClass.GRAPHITI_EMBEDDING
        or allocation.provider != "openrouter"
        or allocation.route != "GRAPHITI_EMBEDDING"
        or allocation.model != "openai/text-embedding-3-large"
        or not policy.qualified or policy.calibration_only
        or terminal.outcome != "CANCELLED"
        or terminal.failure_class != "MISSING_PROVIDER_TELEMETRY"
        or terminal.usage_status is not UsageStatus.UNREPORTED
        or terminal.components.total_tokens is not None
        or terminal.dispatch_at is None or terminal.policy_breach is not None
        or terminal.provider_telemetry_digest is not None
        or terminal.raw_telemetry_pointer is not None
        or terminal.pre_dispatch_zero_proved
        or terminal.subscription_cli_chat_not_cash_debited
        or terminal.od_011_reference != "OD-011:EVALUATION_GRAPHITI_EMBEDDING"
        or allocation.prompt_bytes > policy.max_prompt_bytes
        or allocation.max_output_tokens is None or policy.max_output_tokens is None
        or allocation.max_output_tokens > policy.max_output_tokens
        or allocation.context_identity not in policy.allowed_context_identities
        or allocation.config_identity not in policy.allowed_config_identities
        or allocation.reasoning != "none"
        or not allocation.one_turn or not allocation.exact_input
        or allocation.skills_enabled or allocation.tools_enabled or allocation.mcp_enabled
        or allocation.prior_message_count != 0
        or any(getattr(allocation, key) != getattr(policy, key) for key in (
            "reasoning", "prompt_contract_version", "output_schema_digest",
            "one_turn", "exact_input", "skills_enabled", "tools_enabled",
            "mcp_enabled", "prior_message_count",
        ))
    ):
        raise ModelUsageIntegrityError("native Graphiti embedding cancellation is ineligible")
    if not _has_exact_dispatch(connection, terminal):
        raise ModelUsageIntegrityError("native Graphiti cancellation lacks exact dispatch")
    dispatches = connection.execute(
        "SELECT observed_at,evidence_digest FROM model_transport_observations "
        "WHERE invocation_id=? AND state='DISPATCH_STARTED'",
        (allocation.invocation_id,),
    ).fetchall()
    if len(dispatches) != 1 or tuple(dispatches[0]) != (_utc_text(terminal.dispatch_at), digest_canonical({
        "invocation_id": allocation.invocation_id, "provider": allocation.provider,
        "route": allocation.route, "request_digest": allocation.request_digest,
    })):
        raise ModelUsageIntegrityError("native Graphiti cancellation dispatch binding differs")
    # The parent envelope is CHAT_PRIMARY, but the leaf is EMBEDDING. Do not
    # widen the existing subscription-disposition entry point to cash workloads.
    row = connection.execute(
        "SELECT envelope_id,cycle_id,workload_class,admitted_at,canonical_digest,"
        "record_json FROM model_work_envelopes WHERE envelope_id=?",
        (allocation.envelope_id,),
    ).fetchone()
    if row is None:
        raise ModelUsageIntegrityError("native Graphiti cancellation envelope is absent")
    envelope = _envelope_from_record(_object(row[5]))
    if tuple(row[:5]) != (
        envelope.envelope_id, envelope.cycle_id, envelope.workload_class.value,
        _utc_text(envelope.admitted_at), envelope.canonical_digest,
    ) or (
        row[5] != _json(envelope.as_record())
        or envelope.envelope_id != allocation.envelope_id
        or envelope.cycle_id != allocation.cycle_id
        or envelope.workload_class is not WorkloadClass.GRAPHITI_CHAT_PRIMARY
        or envelope.ingest_id is None
    ):
        raise ModelUsageIntegrityError("native Graphiti cancellation envelope differs")
    prefix, separator, attempt = str(envelope.graphiti_attempt_id or "").rpartition(":")
    if separator != ":" or prefix != envelope.ingest_id or not attempt.isdigit() or int(attempt) <= 0:
        raise ModelUsageIntegrityError("native Graphiti cancellation attempt differs")
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ledger'"
    ).fetchone() is None:
        raise ModelUsageIntegrityError("native Graphiti cancellation lacks source landing")
    identity = _retained_graphiti_request_identity(connection, allocation)
    if identity is None:
        raise ModelUsageIntegrityError("native Graphiti cancellation request is absent")
    unit = _native_landed_source_unit(
        connection,
        ingest_id=envelope.ingest_id,
        effective_revision_digest=identity.effective_revision_digest,
    )
    if unit is None or unit.proving_run_id != "native-source:" + unit.observation_digest:
        raise ModelUsageIntegrityError("native Graphiti cancellation lacks source landing")

    revision = unit.effective_revision
    if (
        identity.leaf_class is not GraphitiLeafClass.EMBEDDING
        or identity.semantic_request_class != "EMBEDDING_VECTOR"
        or identity.response_schema_identity != "embedding-vector"
        or identity.response_schema_digest != digest_canonical({
            "schema": "embedding-vector", "model": allocation.model,
        })
        or identity.ingest_obligation_id != envelope.ingest_id
        or identity.graphiti_attempt_id != envelope.graphiti_attempt_id
        or identity.effective_revision_digest != digest_canonical({
            "source_id": revision.source_id, "item_key": revision.item_key,
            "revision_digest": revision.revision_digest,
            "first_observed_at": revision.first_observed_at,
        })
    ):
        raise ModelUsageIntegrityError("native Graphiti cancellation request binding differs")
    conservative_total = max(policy.max_total_tokens, allocation.prompt_bytes)
    return {
        "authority_schema_version": NATIVE_CONSERVATIVE_DISPOSITION_AUTHORITY_SCHEMA_VERSION,
        "authority_scope": NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE,
        "invocation_id": allocation.invocation_id,
        "terminal_digest": terminal.terminal_digest,
        "allocation_digest": allocation.canonical_digest,
        "policy_digest": policy.canonical_digest,
        "envelope_digest": envelope.canonical_digest,
        "ingest_id": envelope.ingest_id,
        "graphiti_attempt_id": envelope.graphiti_attempt_id,
        "landed_unit_digest": digest_canonical(asdict(unit)),
        "internal_request_digest": identity.canonical_digest,
        "context_manifest_digest": allocation.context_manifest_digest,
        "request_digest": allocation.request_digest,
        "request_bytes": allocation.prompt_bytes,
        "qualified_policy_maximum_total_tokens": policy.max_total_tokens,
        "conservative_total_tokens": conservative_total,
        "estimated_policy_ceiling_exceeded": conservative_total > policy.max_total_tokens,
        "exact_policy_compliance_unknown": True,
        "cash_spend_known": False,
    }


def _native_disposition_authority(
    *,
    allocation: InvocationAllocation,
    terminal: InvocationTerminal,
    policy: InvocationEfficiencyPolicy,
    envelope: WorkEnvelope,
) -> dict[str, object]:
    return {
        "schema_version": (
            NATIVE_CONSERVATIVE_DISPOSITION_AUTHORITY_SCHEMA_VERSION
        ),
        "authority_scope": NATIVE_AUTONOMOUS_USAGE_SCOPE,
        "ingest_id": envelope.ingest_id,
        "graphiti_attempt_id": envelope.graphiti_attempt_id,
        "invocation_id": allocation.invocation_id,
        "terminal_digest": terminal.terminal_digest,
        "allocation_digest": allocation.canonical_digest,
        "policy_digest": policy.canonical_digest,
        "maximum_total_tokens": policy.max_total_tokens,
    }


def _native_embedding_progress_binding(
    connection: sqlite3.Connection,
    *,
    envelope: WorkEnvelope,
    retained_progress: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ledger'"
    ).fetchone() is None:
        raise ModelUsageIntegrityError(
            "native embedding disposition lacks retained progress"
        )
    matches: list[dict[str, object]] = []
    if retained_progress is None:
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='native_current_meta'").fetchone():
            from .native_progress_state import require_ready
            try:
                require_ready(connection, verify_inventory=False)
            except ValueError as exc:
                raise ModelUsageIntegrityError('native embedding disposition CURRENT state differs') from exc
            rows = connection.execute(
                "SELECT pin.progress_seq,pin.progress_digest,record.payload_json "
                "FROM native_embedding_progress_pins AS pin JOIN ledger AS record "
                "ON record.seq=pin.progress_seq AND record.payload_digest=pin.progress_digest "
                "WHERE pin.passage_id=? AND pin.cycle_id=? "
                "AND record.kind='NATIVE_REVISION_PROGRESS' ORDER BY pin.progress_seq",
                (envelope.ingest_id, envelope.cycle_id),
            )
        else:
            rows = connection.execute(
                "SELECT seq,payload_digest,payload_json FROM ledger "
                "WHERE kind='NATIVE_REVISION_PROGRESS' "
                "AND json_extract(payload_json,'$.stage')='EMBEDDING_STARTED' "
                "AND payload_json LIKE ? AND payload_json LIKE ? ORDER BY seq",
                (f'%"{envelope.ingest_id}"%', f'%"{envelope.cycle_id}"%'),
            )
    else:
        rows = connection.execute(
            "SELECT seq,payload_digest,payload_json FROM ledger WHERE seq=? "
            "AND kind='NATIVE_REVISION_PROGRESS'",
            (retained_progress.get("progress_seq"),),
        )
    for seq, payload_digest, raw in rows:
        payload = _object(raw)
        facts = payload.get("facts")
        embeddings = (
            facts.get("retrieval_embeddings", {})
            if isinstance(facts, dict)
            else {}
        )
        if not isinstance(embeddings, dict):
            continue
        for unit_ingest_id, retained in embeddings.items():
            if not isinstance(retained, dict) or (
                retained.get("state"),
                retained.get("passage_id"),
                retained.get("cycle_id"),
            ) != ("STARTED", envelope.ingest_id, envelope.cycle_id):
                continue
            attempt = retained.get("attempt_number")
            base_cycle = f"native-passage:{unit_ingest_id}"
            expected_cycle = (
                base_cycle
                if attempt == 1
                else f"{base_cycle}:retry:{attempt}"
            )
            if (
                payload.get("stage") != "EMBEDDING_STARTED"
                or type(attempt) is not int
                or attempt < 1
                or envelope.cycle_id != expected_cycle
                or raw != canonical_json_bytes(payload).decode("utf-8")
                or payload_digest != digest_bytes(raw.encode("utf-8"))
            ):
                raise ModelUsageIntegrityError(
                    "native embedding disposition progress differs"
                )
            matches.append(
                {
                    "revision_id": _token(
                        str(payload.get("revision_id")), field="revision id"
                    ),
                    "unit_ingest_id": _token(
                        str(unit_ingest_id), field="unit ingest id"
                    ),
                    "passage_id": envelope.ingest_id,
                    "embedding_cycle_id": envelope.cycle_id,
                    "embedding_attempt_number": attempt,
                    "progress_seq": int(seq),
                    "progress_payload_digest": str(payload_digest),
                }
            )
    if len(matches) != 1:
        raise ModelUsageIntegrityError(
            "native embedding disposition progress is absent or ambiguous"
        )
    result = matches[0]
    from newsroom.control_plane.native_progress import (
        LAND,
        NativeRevisionJournal,
        _landed_units,
    )

    if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='native_current_meta'").fetchone():
        unit = _native_landed_source_unit(connection, ingest_id=result['unit_ingest_id'],
                                          revision_id_hint=result['revision_id'])
        landed = [] if unit is None else [unit]
    else:
        landed = []
        for payload_digest, raw in connection.execute(
            "SELECT payload_digest,payload_json FROM ledger WHERE kind=? "
            "AND json_extract(payload_json,'$.revision_id')=?",
            (LAND, result["revision_id"]),
        ):
            payload = _object(raw)
            units = _landed_units(payload)
            NativeRevisionJournal._validate_units(units)
            if (
                raw != canonical_json_bytes(payload).decode("utf-8")
                or payload_digest != digest_bytes(raw.encode("utf-8"))
                or payload.get("revision_id") != result["revision_id"]
            ):
                raise ModelUsageIntegrityError(
                    "native embedding disposition source landing differs"
                )
            landed.extend(
                unit
                for unit in units
                if unit.ingest_id == result["unit_ingest_id"]
                and unit.proving_run_id.startswith("native-source:")
                and unit.authority is not None
            )
    if len(landed) != 1:
        raise ModelUsageIntegrityError(
            "native embedding disposition lacks its landed source unit"
        )
    unit = landed[0]
    result.update(
        {
            "landed_unit_digest": digest_canonical(asdict(unit)),
            "source_observation_digest": unit.observation_digest,
            "source_admission_id": unit.authority.admission_id,
            "source_access_decision_id": unit.authority.access_decision_id,
            "source_revision_id": unit.authority.revision_id,
            "source_representation_id": unit.authority.representation_id,
        }
    )
    if retained_progress is not None and any(
        retained_progress.get(key) != value for key, value in result.items()
    ):
        raise ModelUsageIntegrityError(
            "native embedding disposition progress differs"
        )
    return result


def _native_embedding_timeout_disposition_authority(
    connection: sqlite3.Connection,
    *,
    allocation: InvocationAllocation,
    terminal: InvocationTerminal,
    policy: InvocationEfficiencyPolicy,
    retained_progress: Mapping[str, object] | None = None,
) -> dict[str, object]:
    row = connection.execute(
        "SELECT envelope_id,cycle_id,workload_class,admitted_at,canonical_digest,"
        "record_json FROM model_work_envelopes WHERE envelope_id=?",
        (allocation.envelope_id,),
    ).fetchone()
    if row is None:
        raise ModelUsageIntegrityError(
            "native embedding disposition envelope is absent"
        )
    envelope = _envelope_from_record(_object(row[5]))
    if tuple(row[index] for index in range(5)) != (
        envelope.envelope_id,
        envelope.cycle_id,
        envelope.workload_class.value,
        _utc_text(envelope.admitted_at),
        envelope.canonical_digest,
    ) or (
        envelope.envelope_id != allocation.envelope_id
        or envelope.cycle_id != allocation.cycle_id
        or envelope.workload_class
        is not WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING
        or envelope.ingest_id is None
        or envelope.graphiti_attempt_id is not None
    ):
        raise ModelUsageIntegrityError(
            "native embedding disposition envelope binding differs"
        )
    progress = _native_embedding_progress_binding(
        connection, envelope=envelope, retained_progress=retained_progress
    )
    conservative_total = max(policy.max_total_tokens, allocation.prompt_bytes)
    return {
        "authority_schema_version": (
            NATIVE_EMBEDDING_TIMEOUT_DISPOSITION_AUTHORITY_SCHEMA_VERSION
        ),
        "authority_scope": NATIVE_EMBEDDING_TIMEOUT_USAGE_SCOPE,
        **progress,
        "invocation_id": allocation.invocation_id,
        "terminal_digest": terminal.terminal_digest,
        "allocation_digest": allocation.canonical_digest,
        "policy_digest": policy.canonical_digest,
        "request_digest": allocation.request_digest,
        "request_bytes": allocation.prompt_bytes,
        "qualified_policy_maximum_total_tokens": policy.max_total_tokens,
        "conservative_total_tokens": conservative_total,
        "estimated_policy_ceiling_exceeded": (
            conservative_total > policy.max_total_tokens
        ),
        "exact_policy_compliance_unknown": True,
        "cash_spend_known": False,
    }


def _non_negative(value: int | None, *, field: str) -> int | None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ModelUsageIntegrityError(f"{field} must be a non-negative integer")
    return value


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class UsageComponents:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_read_tokens: int | None = None
    cached_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    context_tokens: int | None = None
    total_tokens: int | None = None
    provenance: str = "UNAVAILABLE"

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "cached_read_tokens",
            "cached_write_tokens",
            "reasoning_tokens",
            "context_tokens",
            "total_tokens",
        ):
            _non_negative(getattr(self, name), field=name)
        if self.provenance not in _PROVENANCE:
            raise ModelUsageIntegrityError("component provenance is invalid")

    def as_record(self) -> dict[str, object]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_read_tokens": self.cached_read_tokens,
            "cached_write_tokens": self.cached_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "context_tokens": self.context_tokens,
            "total_tokens": self.total_tokens,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class InvocationEfficiencyPolicy:
    policy_id: str
    version: str
    workload_class: WorkloadClass
    provider: str
    route: str
    model: str
    reasoning: str
    one_turn: bool
    exact_input: bool
    skills_enabled: bool
    tools_enabled: bool
    mcp_enabled: bool
    prior_message_count: int
    command_semantic_version: str
    command_flags: tuple[str, ...]
    context_manifest_schema_version: str
    disabled_capabilities: tuple[str, ...]
    implementation_revision: str
    calibration_only: bool
    allowed_candidate_ids: tuple[str, ...]
    max_prompt_bytes: int
    max_context_tokens: int
    max_output_tokens: int | None
    max_total_tokens: int
    prompt_contract_version: str
    output_schema_digest: str
    allowed_context_identities: tuple[str, ...]
    allowed_config_identities: tuple[str, ...]
    hard_estimate_ceiling_tokens: int | None
    evidence_digest: str
    qualified: bool
    canonical_digest: str

    @classmethod
    def create(cls, **values: object) -> InvocationEfficiencyPolicy:
        values.pop("canonical_digest", None)
        values.setdefault("command_semantic_version", "UNSPECIFIED")
        values.setdefault("command_flags", ())
        values.setdefault("context_manifest_schema_version", "UNSPECIFIED")
        values.setdefault("disabled_capabilities", ())
        values.setdefault("implementation_revision", "UNSPECIFIED")
        values.setdefault("calibration_only", False)
        values.setdefault("allowed_candidate_ids", ())
        workload = values.get("workload_class")
        if not isinstance(workload, WorkloadClass):
            raise ModelUsageIntegrityError("policy workload class must be typed")
        record = {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            **{
                name: (
                    value.value
                    if isinstance(value, StrEnum)
                    else list(value)
                    if isinstance(value, tuple)
                    else value
                )
                for name, value in values.items()
            },
        }
        policy = cls(**values, canonical_digest=digest_canonical(record))  # type: ignore[arg-type]
        policy._validate()
        return policy

    def _validate(self) -> None:
        for name in (
            "policy_id",
            "version",
            "provider",
            "route",
            "model",
            "reasoning",
            "prompt_contract_version",
            "output_schema_digest",
            "evidence_digest",
            "command_semantic_version",
            "implementation_revision",
            "context_manifest_schema_version",
        ):
            _token(str(getattr(self, name)), field=name)
        for name in (
            "max_prompt_bytes",
            "max_context_tokens",
            "max_total_tokens",
        ):
            value = getattr(self, name)
            if not _is_int(value) or value <= 0:
                raise ModelUsageIntegrityError(f"{name} must be positive")
        if self.max_output_tokens is None:
            if not _nullable_native_output(self.workload_class, self.provider, self.route):
                raise ModelUsageIntegrityError("unbounded output is outside scoped native routes")
        elif not _is_int(self.max_output_tokens) or self.max_output_tokens <= 0:
            raise ModelUsageIntegrityError("max_output_tokens must be positive or scoped native null")
        if not isinstance(self.one_turn, bool) or not isinstance(
            self.exact_input, bool
        ):
            raise ModelUsageIntegrityError("policy turn/input controls must be boolean")
        for name in ("skills_enabled", "tools_enabled", "mcp_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ModelUsageIntegrityError(f"policy {name} must be boolean")
        if not isinstance(self.calibration_only, bool):
            raise ModelUsageIntegrityError("policy calibration_only must be boolean")
        if self.calibration_only and not self.allowed_candidate_ids:
            raise ModelUsageIntegrityError(
                "calibration policy must bind at least one candidate"
            )
        if not self.calibration_only and self.allowed_candidate_ids:
            raise ModelUsageIntegrityError(
                "non-calibration policy cannot bind calibration candidates"
            )
        for candidate_id in self.allowed_candidate_ids:
            _token(candidate_id, field="allowed_candidate_id")
        if len(set(self.allowed_candidate_ids)) != len(self.allowed_candidate_ids):
            raise ModelUsageIntegrityError("allowed calibration candidates repeat")
        if not isinstance(self.command_flags, tuple) or not all(
            isinstance(value, str) for value in self.command_flags
        ):
            raise ModelUsageIntegrityError("policy command flags are invalid")
        if len(set(self.disabled_capabilities)) != len(self.disabled_capabilities):
            raise ModelUsageIntegrityError("disabled capabilities repeat")
        for capability in self.disabled_capabilities:
            _token(capability, field="disabled_capability")
        if _is_hermetic_cont_policy(self) and (
            self.command_semantic_version == "UNSPECIFIED"
            or not self.command_flags
            or self.context_manifest_schema_version == "UNSPECIFIED"
            or not self.disabled_capabilities
            or not re.fullmatch(r"[0-9a-f]{40}", self.implementation_revision)
        ):
            raise ModelUsageIntegrityError(
                "hermetic CONT policy lacks an exact command/manifest binding"
            )
        if not _is_int(self.prior_message_count) or self.prior_message_count < 0:
            raise ModelUsageIntegrityError(
                "policy prior message count must be non-negative"
            )
        if self.qualified and (not self.one_turn or not self.exact_input):
            raise ModelUsageIntegrityError(
                "qualified invocation policy must be one-turn exact-input"
            )
        if self.max_output_tokens is not None and self.max_total_tokens < self.max_output_tokens:
            raise ModelUsageIntegrityError("policy total is below output maximum")
        if not self.allowed_context_identities or len(
            set(self.allowed_context_identities)
        ) != len(self.allowed_context_identities):
            raise ModelUsageIntegrityError("allowed context identities are invalid")
        for identity in self.allowed_context_identities:
            _token(identity, field="allowed_context_identity")
        if not self.allowed_config_identities or len(
            set(self.allowed_config_identities)
        ) != len(self.allowed_config_identities):
            raise ModelUsageIntegrityError("allowed config identities are invalid")
        for identity in self.allowed_config_identities:
            _token(identity, field="allowed_config_identity")
        if self.hard_estimate_ceiling_tokens is not None and (
            not _is_int(self.hard_estimate_ceiling_tokens)
            or self.hard_estimate_ceiling_tokens < self.max_total_tokens
        ):
            raise ModelUsageIntegrityError(
                "hard estimate ceiling must cover the policy total"
            )
        if not isinstance(self.one_turn, bool) or not isinstance(self.qualified, bool):
            raise ModelUsageIntegrityError("policy booleans must be typed")

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "canonical_digest": self.canonical_digest,
            "policy_id": self.policy_id,
            "version": self.version,
            "workload_class": self.workload_class.value,
            "provider": self.provider,
            "route": self.route,
            "model": self.model,
            "reasoning": self.reasoning,
            "one_turn": self.one_turn,
            "exact_input": self.exact_input,
            "skills_enabled": self.skills_enabled,
            "tools_enabled": self.tools_enabled,
            "mcp_enabled": self.mcp_enabled,
            "prior_message_count": self.prior_message_count,
            "command_semantic_version": self.command_semantic_version,
            "command_flags": list(self.command_flags),
            "context_manifest_schema_version": self.context_manifest_schema_version,
            "disabled_capabilities": list(self.disabled_capabilities),
            "implementation_revision": self.implementation_revision,
            "calibration_only": self.calibration_only,
            "allowed_candidate_ids": list(self.allowed_candidate_ids),
            "max_prompt_bytes": self.max_prompt_bytes,
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_total_tokens": self.max_total_tokens,
            "prompt_contract_version": self.prompt_contract_version,
            "output_schema_digest": self.output_schema_digest,
            "allowed_context_identities": list(self.allowed_context_identities),
            "allowed_config_identities": list(self.allowed_config_identities),
            "hard_estimate_ceiling_tokens": self.hard_estimate_ceiling_tokens,
            "evidence_digest": self.evidence_digest,
            "qualified": self.qualified,
        }


@dataclass(frozen=True, slots=True)
class WorkEnvelope:
    envelope_id: str
    cycle_id: str
    workload_class: WorkloadClass
    admitted_at: datetime
    admission_decision_id: str | None
    candidate_id: str | None
    hypothesis_digest: str | None
    evidence_package_digest: str | None
    ingest_id: str | None
    graphiti_attempt_id: str | None
    canonical_digest: str

    @classmethod
    def create(cls, **values: object) -> WorkEnvelope:
        values.pop("envelope_id", None)
        values.pop("canonical_digest", None)
        workload = values.get("workload_class")
        admitted_at = values.get("admitted_at")
        if not isinstance(workload, WorkloadClass):
            raise ModelUsageIntegrityError("envelope workload class must be typed")
        if not isinstance(admitted_at, datetime):
            raise ModelUsageIntegrityError("envelope admitted_at must be datetime")
        identity = {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "cycle_id": values["cycle_id"],
            "workload_class": workload.value,
            "admission_decision_id": values.get("admission_decision_id"),
            "candidate_id": values.get("candidate_id"),
            "hypothesis_digest": values.get("hypothesis_digest"),
            "evidence_package_digest": values.get("evidence_package_digest"),
            "ingest_id": values.get("ingest_id"),
            "graphiti_attempt_id": values.get("graphiti_attempt_id"),
        }
        envelope_id = digest_canonical(identity)
        envelope = cls(
            **values,  # type: ignore[arg-type]
            envelope_id=envelope_id,
            canonical_digest=digest_canonical(
                {
                    **identity,
                    "envelope_id": envelope_id,
                    "admitted_at": _utc_text(admitted_at),
                }
            ),
        )
        envelope._validate()
        return envelope

    def _validate(self) -> None:
        _token(self.cycle_id, field="cycle_id")
        if self.workload_class is WorkloadClass.TYPESAFE_JUDGMENT:
            if not self.evidence_package_digest:
                raise ModelUsageIntegrityError("Typesafe envelope lacks its source snapshot")
            if self.candidate_id:
                if not self.hypothesis_digest or self.graphiti_attempt_id is not None:
                    raise ModelUsageIntegrityError("Typesafe candidate identity differs")
            else:
                prefix, separator, number = str(self.graphiti_attempt_id or "").rpartition(":")
                if (not self.ingest_id or prefix != self.ingest_id or not separator
                        or not number.isdigit() or int(number) <= 0):
                    raise ModelUsageIntegrityError("Typesafe graphiti identity differs")
        native_assessor = (
            self.workload_class is WorkloadClass.NATIVE_EVIDENCE_ASSESSOR
        )
        native_embedding = (
            self.workload_class is WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING
        )
        native_story = self.workload_class is WorkloadClass.NATIVE_STORY_WRITER
        cont = self.workload_class in {
            WorkloadClass.CONT_WRITER_PRIMARY,
            WorkloadClass.CONT_WRITER_FALLBACK,
            WorkloadClass.CONT_ROUTE_HEALTH_PROBE,
        }
        graphiti = self.workload_class in {
            WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            WorkloadClass.GRAPHITI_CHAT_FALLBACK,
            WorkloadClass.GRAPHITI_EMBEDDING,
        }
        if (
            native_assessor
            and (
                not all(
                    (
                        self.candidate_id,
                        self.hypothesis_digest,
                        self.evidence_package_digest,
                    )
                )
                or any(
                    value is not None
                    for value in (
                        self.admission_decision_id,
                        self.ingest_id,
                        self.graphiti_attempt_id,
                    )
                )
            )
        ):
            raise ModelUsageIntegrityError(
                "native assessor envelope lacks acquired evidence identities"
            )
        if (
            native_story
            and (
                not all((self.admission_decision_id, self.candidate_id,
                         self.hypothesis_digest, self.evidence_package_digest))
                or self.ingest_id is not None or self.graphiti_attempt_id is not None
            )
        ):
            raise ModelUsageIntegrityError("native story envelope lacks admitted editorial identities")
        if (
            cont
            and self.workload_class is not WorkloadClass.CONT_ROUTE_HEALTH_PROBE
            and not all(
                (
                    self.admission_decision_id,
                    self.candidate_id,
                    self.hypothesis_digest,
                    self.evidence_package_digest,
                )
            )
        ):
            raise ModelUsageIntegrityError("CONT envelope lacks editorial identities")
        if graphiti and not self.ingest_id:
            raise ModelUsageIntegrityError("Graphiti envelope lacks ingest identity")
        if native_embedding and (
            not self.ingest_id or self.graphiti_attempt_id is not None
        ):
            raise ModelUsageIntegrityError(
                "native embedding envelope lacks passage identity"
            )
        _utc_text(self.admitted_at)

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "canonical_digest": self.canonical_digest,
            "envelope_id": self.envelope_id,
            "cycle_id": self.cycle_id,
            "workload_class": self.workload_class.value,
            "admitted_at": _utc_text(self.admitted_at),
            "admission_decision_id": self.admission_decision_id,
            "candidate_id": self.candidate_id,
            "hypothesis_digest": self.hypothesis_digest,
            "evidence_package_digest": self.evidence_package_digest,
            "ingest_id": self.ingest_id,
            "graphiti_attempt_id": self.graphiti_attempt_id,
        }


@dataclass(frozen=True, slots=True)
class InvocationAllocation:
    invocation_id: str
    envelope_id: str
    cycle_id: str
    leaf_ordinal: int
    workload_class: WorkloadClass
    invocation_policy_digest: str
    provider: str
    route: str
    model: str
    reasoning: str
    prompt_contract_version: str
    prompt_bytes: int
    prompt_digest: str
    request_digest: str
    output_schema_digest: str
    max_output_tokens: int | None
    context_manifest_digest: str
    context_identity: str
    config_identity: str
    one_turn: bool
    exact_input: bool
    skills_enabled: bool
    tools_enabled: bool
    mcp_enabled: bool
    prior_message_count: int
    allocated_at: datetime
    recovery_deadline_at: datetime
    parent_invocation_id: str | None
    canonical_digest: str

    @classmethod
    def create(cls, **values: object) -> InvocationAllocation:
        values.pop("invocation_id", None)
        values.pop("canonical_digest", None)
        workload = values.get("workload_class")
        allocated_at = values.get("allocated_at")
        if not isinstance(workload, WorkloadClass):
            raise ModelUsageIntegrityError("allocation workload class must be typed")
        if not isinstance(allocated_at, datetime):
            raise ModelUsageIntegrityError("allocation timestamp must be datetime")
        recovery_deadline_at = values.get("recovery_deadline_at")
        if not isinstance(recovery_deadline_at, datetime):
            raise ModelUsageIntegrityError(
                "allocation recovery deadline must be datetime"
            )
        identity = {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "envelope_id": values["envelope_id"],
            "cycle_id": values["cycle_id"],
            "leaf_ordinal": values["leaf_ordinal"],
            "workload_class": workload.value,
            "request_digest": values["request_digest"],
            "route": values["route"],
            "parent_invocation_id": values.get("parent_invocation_id"),
        }
        invocation_id = digest_canonical(identity)
        allocation = cls(
            **values,  # type: ignore[arg-type]
            invocation_id=invocation_id,
            canonical_digest=digest_canonical(
                {
                    **identity,
                    "invocation_id": invocation_id,
                    "invocation_policy_digest": values["invocation_policy_digest"],
                    "provider": values["provider"],
                    "model": values["model"],
                    "reasoning": values["reasoning"],
                    "prompt_contract_version": values["prompt_contract_version"],
                    "prompt_bytes": values["prompt_bytes"],
                    "prompt_digest": values["prompt_digest"],
                    "output_schema_digest": values["output_schema_digest"],
                    "max_output_tokens": values["max_output_tokens"],
                    "context_manifest_digest": values["context_manifest_digest"],
                    "context_identity": values["context_identity"],
                    "config_identity": values["config_identity"],
                    "one_turn": values["one_turn"],
                    "exact_input": values["exact_input"],
                    "skills_enabled": values["skills_enabled"],
                    "tools_enabled": values["tools_enabled"],
                    "mcp_enabled": values["mcp_enabled"],
                    "prior_message_count": values["prior_message_count"],
                    "allocated_at": _utc_text(allocated_at),
                    "recovery_deadline_at": _utc_text(recovery_deadline_at),
                }
            ),
        )
        allocation._validate()
        return allocation

    def _validate(self) -> None:
        for name in (
            "envelope_id",
            "cycle_id",
            "invocation_policy_digest",
            "provider",
            "route",
            "model",
            "reasoning",
            "prompt_contract_version",
            "prompt_digest",
            "request_digest",
            "output_schema_digest",
            "context_manifest_digest",
            "context_identity",
            "config_identity",
        ):
            _token(str(getattr(self, name)), field=name)
        if not _is_int(self.leaf_ordinal) or self.leaf_ordinal <= 0:
            raise ModelUsageIntegrityError("leaf ordinal must be positive")
        if not _is_int(self.prompt_bytes) or self.prompt_bytes < 0:
            raise ModelUsageIntegrityError("prompt bytes must be non-negative")
        if self.max_output_tokens is None:
            if not _nullable_native_output(self.workload_class, self.provider, self.route):
                raise ModelUsageIntegrityError("unbounded allocation output is outside scoped native routes")
        elif not _is_int(self.max_output_tokens) or self.max_output_tokens <= 0:
            raise ModelUsageIntegrityError("max output tokens must be positive")
        for name in (
            "one_turn",
            "exact_input",
            "skills_enabled",
            "tools_enabled",
            "mcp_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ModelUsageIntegrityError(
                    f"allocation {name} must be boolean"
                )
        if not _is_int(self.prior_message_count) or self.prior_message_count < 0:
            raise ModelUsageIntegrityError(
                "allocation prior message count must be non-negative"
            )
        if self.recovery_deadline_at <= self.allocated_at:
            raise ModelUsageIntegrityError(
                "allocation recovery deadline must follow allocation"
            )
        _utc_text(self.allocated_at)
        _utc_text(self.recovery_deadline_at)

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "canonical_digest": self.canonical_digest,
            "invocation_id": self.invocation_id,
            "envelope_id": self.envelope_id,
            "cycle_id": self.cycle_id,
            "leaf_ordinal": self.leaf_ordinal,
            "workload_class": self.workload_class.value,
            "invocation_policy_digest": self.invocation_policy_digest,
            "provider": self.provider,
            "route": self.route,
            "model": self.model,
            "reasoning": self.reasoning,
            "prompt_contract_version": self.prompt_contract_version,
            "prompt_bytes": self.prompt_bytes,
            "prompt_digest": self.prompt_digest,
            "request_digest": self.request_digest,
            "output_schema_digest": self.output_schema_digest,
            "max_output_tokens": self.max_output_tokens,
            "context_manifest_digest": self.context_manifest_digest,
            "context_identity": self.context_identity,
            "config_identity": self.config_identity,
            "one_turn": self.one_turn,
            "exact_input": self.exact_input,
            "skills_enabled": self.skills_enabled,
            "tools_enabled": self.tools_enabled,
            "mcp_enabled": self.mcp_enabled,
            "prior_message_count": self.prior_message_count,
            "allocated_at": _utc_text(self.allocated_at),
            "recovery_deadline_at": _utc_text(self.recovery_deadline_at),
            "parent_invocation_id": self.parent_invocation_id,
        }


@dataclass(frozen=True, slots=True)
class InvocationTerminal:
    terminal_digest: str
    invocation_id: str
    outcome: str
    failure_class: str | None
    usage_status: UsageStatus
    components: UsageComponents
    dispatch_at: datetime | None
    completed_at: datetime
    observed_at: datetime
    provider_telemetry_digest: str | None
    raw_telemetry_pointer: str | None
    estimate_policy_digest: str | None
    estimate_calculation: str | None
    pre_dispatch_zero_proved: bool
    od_011_reference: str | None
    subscription_cli_chat_not_cash_debited: bool
    policy_breach: str | None

    @classmethod
    def create(cls, **values: object) -> InvocationTerminal:
        values.pop("terminal_digest", None)
        values.setdefault("provider_telemetry_digest", None)
        values.setdefault("raw_telemetry_pointer", None)
        values.setdefault("estimate_policy_digest", None)
        values.setdefault("estimate_calculation", None)
        values.setdefault("pre_dispatch_zero_proved", False)
        values.setdefault("od_011_reference", None)
        values.setdefault("policy_breach", None)
        status = values.get("usage_status")
        components = values.get("components")
        if not isinstance(status, UsageStatus):
            raise ModelUsageIntegrityError("usage status must be typed")
        if not isinstance(components, UsageComponents):
            raise ModelUsageIntegrityError("usage components must be typed")
        record = _terminal_record(values, digest="")
        terminal = cls(
            **values,  # type: ignore[arg-type]
            terminal_digest=digest_canonical(record),
        )
        terminal._validate_shape()
        return terminal

    def _validate_shape(self) -> None:
        _token(self.invocation_id, field="invocation_id")
        _token(self.outcome, field="outcome")
        if self.failure_class is not None:
            _token(self.failure_class, field="failure_class")
        if self.dispatch_at is not None:
            _utc_text(self.dispatch_at)
        _utc_text(self.completed_at)
        _utc_text(self.observed_at)
        if self.dispatch_at is not None and self.completed_at < self.dispatch_at:
            raise ModelUsageIntegrityError("completion precedes dispatch")
        if self.observed_at < self.completed_at:
            raise ModelUsageIntegrityError("observation precedes completion")
        if not isinstance(self.pre_dispatch_zero_proved, bool):
            raise ModelUsageIntegrityError("pre-dispatch proof flag must be boolean")

    def as_record(self) -> dict[str, object]:
        return _terminal_record(self.__dict_values(), digest=self.terminal_digest)

    def __dict_values(self) -> dict[str, object]:
        return {
            "invocation_id": self.invocation_id,
            "outcome": self.outcome,
            "failure_class": self.failure_class,
            "usage_status": self.usage_status,
            "components": self.components,
            "dispatch_at": self.dispatch_at,
            "completed_at": self.completed_at,
            "observed_at": self.observed_at,
            "provider_telemetry_digest": self.provider_telemetry_digest,
            "raw_telemetry_pointer": self.raw_telemetry_pointer,
            "estimate_policy_digest": self.estimate_policy_digest,
            "estimate_calculation": self.estimate_calculation,
            "pre_dispatch_zero_proved": self.pre_dispatch_zero_proved,
            "od_011_reference": self.od_011_reference,
            "subscription_cli_chat_not_cash_debited": (
                self.subscription_cli_chat_not_cash_debited
            ),
            "policy_breach": self.policy_breach,
        }


def _terminal_record(values: Mapping[str, object], *, digest: str) -> dict[str, object]:
    status = values["usage_status"]
    components = values["components"]
    dispatch_at = values.get("dispatch_at")
    return {
        "schema_version": MODEL_USAGE_SCHEMA_VERSION,
        "terminal_digest": digest,
        "invocation_id": values["invocation_id"],
        "outcome": values["outcome"],
        "failure_class": values.get("failure_class"),
        "usage_status": status.value if isinstance(status, UsageStatus) else status,
        "components": (
            components.as_record()
            if isinstance(components, UsageComponents)
            else components
        ),
        "dispatch_at": (
            _utc_text(dispatch_at) if isinstance(dispatch_at, datetime) else None
        ),
        "completed_at": _utc_text(values["completed_at"]),  # type: ignore[arg-type]
        "observed_at": _utc_text(values["observed_at"]),  # type: ignore[arg-type]
        "provider_telemetry_digest": values.get("provider_telemetry_digest"),
        "raw_telemetry_pointer": values.get("raw_telemetry_pointer"),
        "estimate_policy_digest": values.get("estimate_policy_digest"),
        "estimate_calculation": values.get("estimate_calculation"),
        "pre_dispatch_zero_proved": values.get("pre_dispatch_zero_proved", False),
        "od_011_reference": values.get("od_011_reference"),
        "subscription_cli_chat_not_cash_debited": values[
            "subscription_cli_chat_not_cash_debited"
        ],
        "policy_breach": values.get("policy_breach"),
    }


def _native_sdk_reported_components_error(components: UsageComponents) -> str | None:
    """SDK buckets are complete and disjoint; reasoning is nested in output."""
    counts = (components.input_tokens, components.output_tokens,
              components.cached_read_tokens, components.cached_write_tokens)
    if components.provenance != "PROVIDER_REPORTED":
        return "REPORTED_PROVENANCE_INVALID"
    if any(type(value) is not int or value < 0 for value in (*counts, components.total_tokens)):
        return "REPORTED_SDK_COMPONENTS_MISSING"
    if components.total_tokens != sum(counts):
        return "REPORTED_COMPONENT_TOTAL_INVALID"
    if components.reasoning_tokens is not None and (
        type(components.reasoning_tokens) is not int
        or not 0 <= components.reasoning_tokens <= components.output_tokens
    ):
        return "REPORTED_SDK_REASONING_INVALID"
    if components.context_tokens is not None and (
        type(components.context_tokens) is not int or components.context_tokens < 0
    ):
        return "REPORTED_SDK_CONTEXT_INVALID"
    return None


def _invalid_reported_components(terminal: InvocationTerminal, *, native_sdk: bool = False) -> str | None:
    if terminal.usage_status is not UsageStatus.REPORTED:
        return None
    components = terminal.components
    if native_sdk and not terminal.pre_dispatch_zero_proved:
        return _native_sdk_reported_components_error(components)
    if components.total_tokens is None:
        return "REPORTED_TOTAL_MISSING"
    if components.provenance not in {"PROVIDER_REPORTED", "CLI_DERIVED"}:
        return "REPORTED_PROVENANCE_INVALID"
    input_tokens = components.input_tokens
    output_tokens = components.output_tokens
    known = tuple(
        value
        for value in (
            input_tokens,
            output_tokens,
            components.cached_read_tokens,
            components.cached_write_tokens,
            components.reasoning_tokens,
        )
        if value is not None
    )
    if not known:
        return None
    possible_totals = {sum(known)}
    if input_tokens is not None and output_tokens is not None:
        possible_totals.add(input_tokens + output_tokens)
        cache_read = components.cached_read_tokens or 0
        cache_write = components.cached_write_tokens or 0
        # Grok 1.0.10 headless: total = uncached input + output + cache
        # buckets; reasoning is nested in output, not additive.
        possible_totals.add(
            input_tokens + output_tokens + cache_read + cache_write
        )
    if components.total_tokens not in possible_totals:
        return "REPORTED_COMPONENT_TOTAL_INVALID"
    return None


def _retain_provider_telemetry(
    connection: sqlite3.Connection,
    *,
    invocation_id: str,
    provider_telemetry: Mapping[str, object],
) -> str:
    telemetry_value = dict(provider_telemetry)
    provider_telemetry_digest = digest_canonical(telemetry_value)
    telemetry_record = {
        "schema_version": MODEL_USAGE_SCHEMA_VERSION,
        "invocation_id": invocation_id,
        "provider_telemetry_digest": provider_telemetry_digest,
        "provider_telemetry": telemetry_value,
    }
    telemetry_record_digest = digest_canonical(telemetry_record)
    connection.execute(
        "INSERT OR IGNORE INTO model_provider_telemetry("
        "telemetry_record_digest,invocation_id,provider_telemetry_digest,record_json) "
        "VALUES(?,?,?,?)",
        (
            telemetry_record_digest,
            invocation_id,
            provider_telemetry_digest,
            _json(telemetry_record),
        ),
    )
    retained = connection.execute(
        "SELECT record_json FROM model_provider_telemetry "
        "WHERE telemetry_record_digest=?",
        (telemetry_record_digest,),
    ).fetchone()
    if retained is None or _object(retained[0]) != telemetry_record:
        raise ModelUsageIntegrityError("conflicting provider telemetry replay")
    return provider_telemetry_digest


def _native_workspace_busy_zero(
    connection: sqlite3.Connection, *, envelope: WorkEnvelope, attempt_number: int,
) -> bool:
    """Authenticate one forward controller refusal, never an absent paid receipt."""
    ingest = envelope.ingest_id
    if (not ingest or envelope.cycle_id != native_graphiti_usage_cycle_id(
            ingest_id=ingest, attempt_number=attempt_number)
        or envelope.graphiti_attempt_id != f"{ingest}:{attempt_number}"
        or envelope.workload_class is not WorkloadClass.GRAPHITI_CHAT_PRIMARY
        or any(value is not None for value in (envelope.admission_decision_id, envelope.candidate_id,
            envelope.hypothesis_digest, envelope.evidence_package_digest))
        or connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ledger'").fetchone() is None):
        return False
    row = connection.execute(
        "SELECT outcome,receipt_digest,receipt_json FROM unpublished_graphiti_attempt_receipts "
        "WHERE ingest_id=? AND attempt_number=?", (ingest, attempt_number),
    ).fetchone() if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='unpublished_graphiti_attempt_receipts'"
    ).fetchone() else None
    if row is None:
        return False
    record = _object(row[2]); unsigned = dict(record); digest = unsigned.pop("receipt_digest", None)
    embedding = record.get("embedding_usage"); accounting = record.get("accounting")
    if (row[0] != record.get("outcome") or row[0] != "FAILED"
        or row[1] != digest or digest_bytes(canonical_json_bytes(unsigned)) != digest
        or record.get("ingest_id") != ingest or record.get("attempt_number") != attempt_number
        or record.get("provider_attempt_number") != attempt_number
        or record.get("episode_uuid") != ingest or record.get("profile") != "EVALUATION"
        or record.get("setup_failure") != "GuardWorkspaceBusy"
        or record.get("dispatch_state") != "NOT_DISPATCHED"
        or record.get("failure_code") != "PRODUCER_INTERNAL_ERROR"
        or any(record.get(key) != [] for key in ("chat_invocations", "entities", "relations", "proposals"))
        or any(type(record.get(key)) is not int or record[key] != 0 for key in
               ("proposal_count", "entity_count", "relation_count", "request_tokens", "response_tokens", "cost_microunits"))
        or type(embedding) is not dict or embedding.get("usage_basis") != "NO_EMBEDDING_CALL"
        or embedding.get("requests") != [] or any(type(embedding.get(key)) is not int or embedding[key] != 0
               for key in ("request_count", "embedding_tokens", "cost_usd_microunits"))
        or type(accounting) is not dict or accounting.get("status") != "RECONCILED"
        or accounting.get("spend_id") != f"{ingest}:{attempt_number}"
        or accounting.get("usage_basis") != "NO_EMBEDDING_CALL"
        or accounting.get("unused_reservation_released") is not True
        or accounting.get("reused_unresolved_reservation") is True
        or any(type(accounting.get(key)) is not int or accounting[key] != 0
               for key in ("actual_usd_microunits", "actual_gbp_microunits"))):
        return False
    outcome = connection.execute("SELECT record_json FROM model_work_outcomes WHERE envelope_id=?",
                                 (envelope.envelope_id,)).fetchone()
    if outcome is None:
        return False
    work = _object(outcome[0])
    if (work.get("schema_version") != MODEL_USAGE_SCHEMA_VERSION or work.get("outcome") != "GRAPHITI_FAILED"
        or work.get("envelope_id") != envelope.envelope_id or work.get("outcome_record_id") != digest
        or work.get("payload_digest") is not None or type(work.get("retained_proposal_count")) is not int
        or work.get("retained_proposal_count") != 0 or work.get("accepted_provider_attempt_id") is not None):
        return False
    if connection.execute(
        "SELECT 1 FROM model_invocation_allocations WHERE envelope_id=? OR cycle_id=? "
        "OR json_extract(record_json,'$.envelope_id')=? OR json_extract(record_json,'$.cycle_id')=? "
        "UNION ALL SELECT 1 FROM graphiti_internal_requests WHERE envelope_id=? OR graphiti_attempt_id=? "
        "UNION ALL SELECT 1 FROM graphiti_internal_request_refusals WHERE envelope_id=? LIMIT 1",
        (envelope.envelope_id, envelope.cycle_id, envelope.envelope_id, envelope.cycle_id,
         envelope.envelope_id, envelope.graphiti_attempt_id, envelope.envelope_id),
    ).fetchone():
        return False
    spend = connection.execute(
        "SELECT ingest_id,attempt_number,status,usage_basis,actual_usd_microunits,actual_gbp_microunits "
        "FROM unpublished_graphiti_spend WHERE spend_id=?", (accounting["spend_id"],),
    ).fetchone()
    if spend is None or tuple(spend) != (ingest, attempt_number, "RECONCILED", "NO_EMBEDDING_CALL", 0, 0):
        return False
    payload = canonical_json_bytes(record)
    ledger = connection.execute(
        "SELECT at,payload_json,prev_digest,digest FROM ledger "
        "WHERE kind='GRAPHITI_EVALUATION_ATTEMPT' AND payload_digest=? LIMIT 2", (digest_bytes(payload),),
    ).fetchall()
    if (len(ledger) != 1 or ledger[0][1] != payload.decode()
        or ledger[0][3] != digest_canonical({"at": ledger[0][0], "kind": "GRAPHITI_EVALUATION_ATTEMPT",
            "payload_digest": digest_bytes(payload), "prev": ledger[0][2]})):
        return False
    unit = _native_landed_source_unit(connection, ingest_id=ingest, revision_id_hint=record.get("revision_id"))
    if unit is None or unit.authority is None:
        return False
    passages = record.get("passages"); body = " ".join(unit.episode_body.split()).encode()
    if (type(passages) is not list or len(passages) != 1 or type(passages[0]) is not dict
        or passages[0].get("byte_offset") != 0 or passages[0].get("byte_length") != len(body)
        or passages[0].get("text_digest") != digest_bytes(body) or passages[0].get("blob_digest") != digest_bytes(body)
        or passages[0].get("admission_id") != unit.authority.admission_id
        or passages[0].get("access_decision_id") != unit.authority.access_decision_id
        or record.get("authority_record_ids") != [str(item["record_id"]) for item in unit.authority.records]):
        return False
    return all(record.get(key) == getattr(unit, key) for key in (
        "source_id", "item_key", "proving_run_id", "observation_digest", "revision_id",
        "published_at", "updated_at", "observed_at", "chunk_ordinal", "chunk_count", "predecessor_ingest_id"))


def _native_immutable_replay_proof(
    connection: sqlite3.Connection,
    *,
    ingest_id: str,
    attempt_number: int,
    evidence: GraphitiIngestRetryEvidence,
) -> tuple[bool, dict[str, object] | None]:
    """Authenticate one provider-free receipt replay before ambiguity recovery."""

    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='unpublished_graphiti_attempt_receipts'"
    ).fetchone() is None:
        return False, None
    row = connection.execute(
        "SELECT outcome,receipt_digest,receipt_json FROM "
        "unpublished_graphiti_attempt_receipts WHERE ingest_id=? "
        "AND attempt_number=?",
        (ingest_id, attempt_number),
    ).fetchone()
    if row is None:
        return False, None
    try:
        receipt = _object(row[2])
    except ModelUsageIntegrityError:
        return False, None
    accounting = receipt.get("accounting")
    claimed = bool(
        isinstance(accounting, Mapping)
        and accounting.get("recovery_classification")
        == "RECOVERED_IMMUTABLE_COMPLETE"
    )
    if not claimed:
        return False, None

    unsigned = dict(receipt)
    supplied_digest = unsigned.pop("receipt_digest", None)
    receipt_digest = digest_bytes(canonical_json_bytes(unsigned))
    provider_attempt_number = receipt.get("provider_attempt_number")
    if (
        supplied_digest != receipt_digest
        or row[1] != receipt_digest
        or receipt.get("ingest_id") != ingest_id
        or receipt.get("attempt_number") != attempt_number
        or receipt.get("outcome") != row[0]
        or row[0] != "FAILED"
        or receipt.get("failure_code") != "PRODUCER_INTERNAL_ERROR"
        or receipt.get("combined_temporal_failure_code") != "PIPELINE_FAILED"
        or receipt.get("chat_subscription_not_debited") is not True
        or type(provider_attempt_number) is not int
        or not 0 < provider_attempt_number < attempt_number
        or provider_attempt_number in evidence.unresolved_attempts
        or provider_attempt_number
        not in {
            *evidence.zero_dispatch_attempts,
            *evidence.settled_provider_attempts,
        }
    ):
        return True, None

    provider_row = connection.execute(
        "SELECT outcome,receipt_digest,receipt_json FROM "
        "unpublished_graphiti_attempt_receipts WHERE ingest_id=? "
        "AND attempt_number=?",
        (ingest_id, provider_attempt_number),
    ).fetchone()
    if provider_row is None:
        return True, None
    try:
        provider_receipt = _object(provider_row[2])
    except ModelUsageIntegrityError:
        return True, None
    provider_unsigned = dict(provider_receipt)
    provider_supplied_digest = provider_unsigned.pop("receipt_digest", None)
    provider_receipt_digest = digest_bytes(canonical_json_bytes(provider_unsigned))
    replay_invocations = receipt.get("chat_invocations")
    provider_invocations = provider_receipt.get("chat_invocations")
    if (
        provider_supplied_digest != provider_receipt_digest
        or provider_row[1] != provider_receipt_digest
        or provider_receipt.get("ingest_id") != ingest_id
        or provider_receipt.get("attempt_number") != provider_attempt_number
        or provider_receipt.get("outcome") != provider_row[0]
        or provider_receipt.get("failure_code") != "PRODUCER_INTERNAL_ERROR"
        or not isinstance(replay_invocations, list)
        or not replay_invocations
        or replay_invocations != provider_invocations
        or any(not isinstance(item, Mapping) for item in replay_invocations)
    ):
        return True, None

    # Snapshot recovery copies every provider result/effect field. It changes
    # only the attempt identity and recomputed raw digest; each attempt also
    # carries its own freshly retained rights evidence and spend accounting.
    stable_rights = ("policy_digest", "scope", "source_id", "source_url")
    member_fields = {"observation_source_id", "observation_member_digest"}
    fresh_rights = (
        "assessment_admission_id", "assessment_blob_digest",
        "observation_admission_id", "observation_blob_digest",
        "packet_digest", "rights_decision_id",
    )
    replay_rights = receipt.get("dispatch_rights")
    provider_rights = provider_receipt.get("dispatch_rights")
    replay_raw_digest = receipt.get("raw_output_digest")
    provider_raw_digest = provider_receipt.get("raw_output_digest")
    try:
        validate_sha256_digest(str(replay_raw_digest), field="replay raw output")
        validate_sha256_digest(str(provider_raw_digest), field="provider raw output")
        for rights in (replay_rights, provider_rights):
            if not isinstance(rights, Mapping):
                raise ValueError("replay rights evidence is absent")
            from .native_source_rights import validate_rights_observation_selector
            validate_rights_observation_selector(rights)
            for field in fresh_rights:
                value = rights.get(field)
                if type(value) is not str or not value:
                    raise ValueError("replay rights evidence differs")
                if field.endswith("digest") or field == "rights_decision_id":
                    validate_sha256_digest(value, field=field)
    except ValueError:
        return True, None
    if (
        set(replay_rights) not in (set(stable_rights) | set(fresh_rights),
                                  set(stable_rights) | set(fresh_rights) | member_fields)
        or set(provider_rights) not in (set(stable_rights) | set(fresh_rights),
                                        set(stable_rights) | set(fresh_rights) | member_fields)
        or any(replay_rights[field] != provider_rights[field] for field in stable_rights)
        or replay_raw_digest == provider_raw_digest
    ):
        return True, None
    normalised_replay = dict(unsigned)
    normalised_provider = dict(provider_unsigned)
    for value in (normalised_replay, normalised_provider):
        value.pop("accounting", None)
        value.pop("attempt_number", None)
        value.pop("raw_output_digest", None)
        value["dispatch_rights"] = {
            field: value["dispatch_rights"][field] for field in stable_rights
        }
    if canonical_json_bytes(normalised_replay) != canonical_json_bytes(
        normalised_provider
    ):
        return True, None

    provider_envelope = WorkEnvelope.create(
        cycle_id=native_graphiti_usage_cycle_id(
            ingest_id=ingest_id, attempt_number=provider_attempt_number
        ),
        workload_class=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        admitted_at=datetime(1970, 1, 1, tzinfo=UTC),
        admission_decision_id=None,
        candidate_id=None,
        hypothesis_digest=None,
        evidence_package_digest=None,
        ingest_id=ingest_id,
        graphiti_attempt_id=f"{ingest_id}:{provider_attempt_number}",
    )
    invocation_proof = []
    for item in replay_invocations:
        invocation_id = item.get("model_invocation_id")
        if type(invocation_id) is not str:
            return True, None
        try:
            allocation, terminal = _retained_terminal_allocation(
                connection, invocation_id
            )
        except ModelUsageIntegrityError:
            return True, None
        if (
            allocation.envelope_id != provider_envelope.envelope_id
            or item.get("model_work_envelope_id") != allocation.envelope_id
            or item.get("model_invocation_allocation_digest")
            != allocation.canonical_digest
            or item.get("model_invocation_terminal_digest")
            != terminal.terminal_digest
            or item.get("outcome") != terminal.outcome
        ):
            return True, None
        invocation_proof.append(
            {
                "invocation_id": invocation_id,
                "allocation_digest": allocation.canonical_digest,
                "terminal_digest": terminal.terminal_digest,
            }
        )

    replay_envelope = WorkEnvelope.create(
        cycle_id=native_graphiti_usage_cycle_id(
            ingest_id=ingest_id, attempt_number=attempt_number
        ),
        workload_class=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        admitted_at=datetime(1970, 1, 1, tzinfo=UTC),
        admission_decision_id=None,
        candidate_id=None,
        hypothesis_digest=None,
        evidence_package_digest=None,
        ingest_id=ingest_id,
        graphiti_attempt_id=f"{ingest_id}:{attempt_number}",
    )
    envelope_row = connection.execute(
        "SELECT envelope_id,cycle_id,workload_class,canonical_digest,record_json "
        "FROM model_work_envelopes WHERE envelope_id=?",
        (replay_envelope.envelope_id,),
    ).fetchone()
    outcome_row = connection.execute(
        "SELECT outcome_digest,envelope_id,outcome,terminal_at,record_json "
        "FROM model_work_outcomes "
        "WHERE envelope_id=?",
        (replay_envelope.envelope_id,),
    ).fetchone()
    if envelope_row is None or outcome_row is None:
        return True, None
    try:
        retained_envelope = _envelope_from_record(_object(envelope_row[4]))
        outcome = _object(outcome_row[4])
    except ModelUsageIntegrityError:
        return True, None
    unsigned_outcome = dict(outcome)
    outcome_digest = unsigned_outcome.pop("outcome_digest", None)
    leaf_count = connection.execute(
        "SELECT COUNT(*) FROM model_invocation_allocations WHERE envelope_id=?",
        (replay_envelope.envelope_id,),
    ).fetchone()[0]
    request_count = connection.execute(
        "SELECT COUNT(*) FROM graphiti_internal_requests WHERE envelope_id=?",
        (replay_envelope.envelope_id,),
    ).fetchone()[0]
    current_attempt = accounting.get("current_attempt")
    provider_attempt = accounting.get("provider_attempt")
    current_spend = connection.execute(
        "SELECT status,usage_basis,actual_usd_microunits,"
        "actual_gbp_microunits FROM unpublished_graphiti_spend "
        "WHERE ingest_id=? AND attempt_number=?",
        (ingest_id, attempt_number),
    ).fetchone()
    provider_spend = connection.execute(
        "SELECT status,usage_basis,actual_usd_microunits,"
        "actual_gbp_microunits FROM unpublished_graphiti_spend "
        "WHERE ingest_id=? AND attempt_number=?",
        (ingest_id, provider_attempt_number),
    ).fetchone()
    provider_accounting = provider_receipt.get("accounting")
    embedding = receipt.get("embedding_usage")
    if (
        tuple(envelope_row[:4])
        != (
            retained_envelope.envelope_id,
            retained_envelope.cycle_id,
            retained_envelope.workload_class.value,
            retained_envelope.canonical_digest,
        )
        or retained_envelope.envelope_id != replay_envelope.envelope_id
        or retained_envelope.cycle_id != replay_envelope.cycle_id
        or retained_envelope.workload_class is not replay_envelope.workload_class
        or retained_envelope.ingest_id != ingest_id
        or retained_envelope.graphiti_attempt_id
        != f"{ingest_id}:{attempt_number}"
        or outcome_digest != outcome_row[0]
        or digest_canonical(unsigned_outcome) != outcome_digest
        or tuple(outcome_row[1:4])
        != (
            outcome.get("envelope_id"),
            outcome.get("outcome"),
            outcome.get("terminal_at"),
        )
        or outcome.get("envelope_id") != replay_envelope.envelope_id
        or outcome_row[2] != "GRAPHITI_FAILED"
        or outcome.get("outcome") != "GRAPHITI_FAILED"
        or outcome.get("outcome_record_id") != receipt_digest
        or outcome.get("payload_digest") is not None
        or outcome.get("retained_proposal_count") != 0
        or outcome.get("accepted_provider_attempt_id") is not None
        or any((leaf_count, request_count))
        or not isinstance(current_attempt, Mapping)
        or not isinstance(provider_attempt, Mapping)
        or current_spend is None
        or tuple(current_spend) != ("RECONCILED", "NO_EMBEDDING_CALL", 0, 0)
        or provider_spend is None
        or not isinstance(provider_accounting, Mapping)
        or provider_accounting.get("spend_id")
        != f"{ingest_id}:{provider_attempt_number}"
        or provider_accounting.get("status") != provider_spend[0]
        or provider_accounting.get("usage_basis") != provider_spend[1]
        or provider_accounting.get("actual_usd_microunits") != provider_spend[2]
        or provider_accounting.get("actual_gbp_microunits") != provider_spend[3]
        or provider_attempt.get("spend_id")
        != f"{ingest_id}:{provider_attempt_number}"
        or provider_attempt.get("status") != provider_spend[0]
        or provider_attempt.get("retained_attempt_receipt") is not True
        or provider_attempt.get("reconciled_again") is not False
        or provider_attempt.get("accounting") is not None
        or current_attempt.get("spend_id") != f"{ingest_id}:{attempt_number}"
        or current_attempt.get("status") != "RECONCILED"
        or current_attempt.get("usage_basis") != "NO_EMBEDDING_CALL"
        or current_attempt.get("actual_usd_microunits") != 0
        or current_attempt.get("actual_gbp_microunits") != 0
        or current_attempt.get("unused_reservation_released") is not True
        or (
            "provider_dispatch_state" in receipt
            and receipt.get("provider_dispatch_state") != "NOT_DISPATCHED"
        )
        or not isinstance(embedding, Mapping)
        or embedding.get("request_count") != 0
        or embedding.get("requests") != []
        or embedding.get("embedding_tokens") != 0
        or embedding.get("cost_usd_microunits") != 0
        or embedding.get("usage_basis") != "NO_EMBEDDING_CALL"
    ):
        return True, None
    return True, {
        "attempt_number": attempt_number,
        "receipt_digest": receipt_digest,
        "provider_attempt_number": provider_attempt_number,
        "provider_receipt_digest": provider_receipt_digest,
        "replay_envelope_id": replay_envelope.envelope_id,
        "replay_outcome_digest": outcome_digest,
        "invocations": invocation_proof,
    }


class ModelUsageService:
    """Single SQLite authority for model usage, outcomes and exports."""

    def __init__(self, path: str) -> None:
        assert_private_store(path)
        self.path = path
        connection = self._connection()
        try:
            connection.executescript(_SCHEMA)
            connection.executescript(model_usage_current.SCHEMA)
            model_usage_current.initialise_empty(connection)
            if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ledger'").fetchone():
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS model_usage_graphiti_attempt_payload "
                    "ON ledger(payload_digest) WHERE kind='GRAPHITI_EVALUATION_ATTEMPT'"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS model_usage_native_landed_revision "
                    "ON ledger(kind,json_extract(payload_json,'$.revision_id')) "
                    "WHERE kind='NATIVE_REVISION_LANDED'"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS model_usage_assessor_requalification "
                    "ON ledger(kind,json_extract(payload_json,'$.invocation_id')) "
                    "WHERE kind='NATIVE_ASSESSOR_INPUT_REQUALIFICATION'"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS model_usage_assessor_output_requalification "
                    "ON ledger(kind,json_extract(payload_json,'$.invocation_id')) "
                    "WHERE kind='NATIVE_ASSESSOR_OUTPUT_GUARD_REQUALIFICATION'"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS model_usage_assessor_result "
                    "ON ledger(kind,json_extract(payload_json,'$.invocation_id')) "
                    "WHERE kind='NATIVE_ASSESSMENT_RESULT'"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS model_usage_assessor_materialisation "
                    "ON ledger(kind,json_extract(payload_json,'$.invocation_id')) "
                    "WHERE kind='NATIVE_ASSESSMENT_MATERIALISATION'"
                )
            applied_at = _utc_text(datetime.now(tz=UTC))
            connection.executemany(
                "INSERT OR IGNORE INTO model_usage_migrations("
                "migration_id,schema_version,applied_at) VALUES(?,?,?)",
                (
                    (migration_id, schema_version, applied_at)
                    for migration_id, schema_version in _MODEL_USAGE_MIGRATIONS
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        apply_control_plane_sqlite_profile(connection)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def import_current_state(self) -> int:
        """Explicit quiescent legacy cutover; ordinary boot never imports history."""
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            count = model_usage_current.import_legacy(connection)
            connection.commit()
            return count
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def register_policy(self, policy: InvocationEfficiencyPolicy) -> None:
        policy._validate()
        record = policy.as_record()
        self._insert_exact(
            table="model_invocation_policies",
            identity_column="canonical_digest",
            identity=policy.canonical_digest,
            record=record,
            sql="INSERT INTO model_invocation_policies("
            "canonical_digest,policy_id,version,workload_class,provider,route,model,"
            "qualified,record_json) VALUES(?,?,?,?,?,?,?,?,?)",
            values=(
                policy.canonical_digest,
                policy.policy_id,
                policy.version,
                policy.workload_class.value,
                policy.provider,
                policy.route,
                policy.model,
                int(policy.qualified),
                _json(record),
            ),
        )

    def qualified_policy(
        self,
        *,
        workload_class: WorkloadClass,
        provider: str,
        route: str,
        model: str,
        reasoning: str,
        candidate_id: str | None = None,
        implementation_revision: str | None = None,
        config_identity: str | None = None,
        output_schema_digest: str | None = None,
    ) -> InvocationEfficiencyPolicy:
        """Resolve a qualified route/contract policy; software versions are audit facts."""

        connection = self._connection()
        try:
            rows = connection.execute(
                "SELECT record_json FROM model_invocation_policies "
                "WHERE workload_class=? AND provider=? AND route=? AND model=? "
                "AND qualified=1 ORDER BY rowid DESC",
                (workload_class.value, provider, route, model),
            ).fetchall()
        finally:
            connection.close()
        policy_records = [_object(row[0]) for row in rows]
        policies = [
            _policy_from_record(record)
            for record in policy_records
            if str(record.get("reasoning")) == reasoning
        ]
        if config_identity is not None:
            policies = [
                policy
                for policy in policies
                if config_identity in policy.allowed_config_identities
            ]
        if output_schema_digest is not None:
            policies = [
                policy
                for policy in policies
                if policy.output_schema_digest == output_schema_digest
            ]
        policies = [
            policy
            for policy in policies
            if not policy.calibration_only
            or candidate_id in policy.allowed_candidate_ids
        ]
        general_policies = [policy for policy in policies if not policy.calibration_only]
        if general_policies:
            compatible_semantic_reader = False
            if (output_schema_digest is not None and
                    (workload_class, provider, route, model, reasoning) in {
                        (WorkloadClass.TYPESAFE_JUDGMENT, "typesafe", "TYPESAFE_JUDGMENT", "jev-latest", "none"),
                        (WorkloadClass.NATIVE_EVIDENCE_ASSESSOR, "grok-build-cli", "NATIVE_CLAIM_LOCALISATION", "grok-4.7", "high"),
                    }):
                # These retained readers authenticate the original policy. An
                # audited implementation-only upgrade keeps the invocation grant.
                contracts = [{key: value for key, value in asdict(policy).items()
                              if key not in {"canonical_digest", "implementation_revision", "evidence_digest"}}
                             for policy in general_policies]
                compatible_semantic_reader = all(contract == contracts[0] for contract in contracts)
            policies = (
                general_policies[:1]
                if all(_is_hermetic_cont_policy(policy) for policy in general_policies)
                or compatible_semantic_reader
                or (
                    workload_class is WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING
                    and output_schema_digest is not None
                )
                else general_policies
            )
        elif policies and all(_is_hermetic_cont_policy(policy) for policy in policies):
            # A later compatible policy supersedes the older bootstrap for the
            # same bounded candidate without mutating retained policy history.
            policies = policies[:1]
        if len(policies) != 1:
            raise ModelUsageAdmissionError(
                "exact qualified invocation policy is absent or ambiguous",
                reason_code="INVOCATION_POLICY_UNAVAILABLE",
            )
        return policies[0]

    def open_envelope(self, envelope: WorkEnvelope) -> None:
        envelope._validate()
        record = envelope.as_record()
        self._insert_exact(
            table="model_work_envelopes",
            identity_column="envelope_id",
            identity=envelope.envelope_id,
            record=record,
            sql="INSERT INTO model_work_envelopes("
            "envelope_id,cycle_id,workload_class,admitted_at,canonical_digest,record_json) "
            "VALUES(?,?,?,?,?,?)",
            values=(
                envelope.envelope_id,
                envelope.cycle_id,
                envelope.workload_class.value,
                _utc_text(envelope.admitted_at),
                envelope.canonical_digest,
                _json(record),
            ),
        )

    def resume_or_open_native_assessor_envelope(self, envelope: WorkEnvelope) -> WorkEnvelope:
        """Reuse exact pre-dispatch intent, never an allocated assessor attempt."""
        return self._resume_or_open_native_envelope(
            envelope, workload=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR, label="assessor",
        )

    def resume_or_open_native_story_envelope(self, envelope: WorkEnvelope) -> WorkEnvelope:
        """Reuse exact admitted intent, never an allocated draft or review leaf."""
        return self._resume_or_open_native_envelope(
            envelope, workload=WorkloadClass.NATIVE_STORY_WRITER, label="story writer",
        )

    def resume_or_open_typesafe_envelope(self, envelope: WorkEnvelope) -> WorkEnvelope:
        """Reuse exact pre-allocation semantic intent without authorising a retry."""
        return self._resume_or_open_native_envelope(
            envelope, workload=WorkloadClass.TYPESAFE_JUDGMENT, label="Typesafe judgment",
        )

    def _resume_or_open_native_envelope(
        self, envelope: WorkEnvelope, *, workload: WorkloadClass, label: str,
    ) -> WorkEnvelope:
        envelope._validate()
        if envelope.workload_class is not workload:
            raise ModelUsageIntegrityError(f"{label} envelope targets another workload")
        connection = self._connection()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT envelope_id,cycle_id,workload_class,admitted_at,canonical_digest,record_json "
                "FROM model_work_envelopes WHERE envelope_id=?",
                (envelope.envelope_id,),
            ).fetchone()
            if row is not None:
                record = _object(row[5])
                retained = _envelope_from_record(record)
                expected = envelope.as_record()
                for key in ("admitted_at", "canonical_digest"):
                    expected[key] = record.get(key)
                if (tuple(row[:5]) != (retained.envelope_id, retained.cycle_id, retained.workload_class.value,
                                      _utc_text(retained.admitted_at), retained.canonical_digest)
                        or record != expected or retained.as_record() != record or row[5] != _json(record)):
                    raise ModelUsageIntegrityError(f"retained {label} envelope differs")
                if connection.execute(
                    "SELECT 1 FROM model_invocation_allocations WHERE envelope_id=?",
                    (envelope.envelope_id,),
                ).fetchone():
                    raise ModelUsageAdmissionError(f"{label} envelope already has an allocation")
                return retained
        finally:
            connection.close()
        self.open_envelope(envelope)
        return envelope

    def resume_or_open_graphiti_envelope(
        self, envelope: WorkEnvelope
    ) -> WorkEnvelope:
        """Open a Graphiti envelope or return its exact durable restart value."""

        envelope._validate()
        if envelope.graphiti_attempt_id is None:
            raise ModelUsageIntegrityError(
                "Graphiti envelope resume lacks an attempt identity"
            )
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT record_json FROM model_work_envelopes WHERE envelope_id=?",
                (envelope.envelope_id,),
            ).fetchone()
            if row is None:
                self._insert_exact(
                    table="model_work_envelopes",
                    identity_column="envelope_id",
                    identity=envelope.envelope_id,
                    record=envelope.as_record(),
                    sql="INSERT INTO model_work_envelopes("
                    "envelope_id,cycle_id,workload_class,admitted_at,"
                    "canonical_digest,record_json) VALUES(?,?,?,?,?,?)",
                    values=(
                        envelope.envelope_id,
                        envelope.cycle_id,
                        envelope.workload_class.value,
                        _utc_text(envelope.admitted_at),
                        envelope.canonical_digest,
                        _json(envelope.as_record()),
                    ),
                    connection=connection,
                )
                connection.commit()
                return envelope
            record = _object(row[0])
            resumed = WorkEnvelope.create(
                cycle_id=str(record["cycle_id"]),
                workload_class=WorkloadClass(str(record["workload_class"])),
                admitted_at=_instant(str(record["admitted_at"])),
                admission_decision_id=record.get("admission_decision_id"),
                candidate_id=record.get("candidate_id"),
                hypothesis_digest=record.get("hypothesis_digest"),
                evidence_package_digest=record.get("evidence_package_digest"),
                ingest_id=record.get("ingest_id"),
                graphiti_attempt_id=record.get("graphiti_attempt_id"),
            )
            if (
                resumed.envelope_id != envelope.envelope_id
                or resumed.canonical_digest != record.get("canonical_digest")
            ):
                raise ModelUsageIntegrityError(
                    "retained Graphiti envelope identity is invalid"
                )
            connection.commit()
            return resumed
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def retain_context_manifest(self, record: Mapping[str, object]) -> None:
        """Retain the non-secret context proof before any provider dispatch."""

        retained = dict(record)
        digest = str(retained.pop("context_manifest_digest", ""))
        if not digest or digest_canonical(retained) != digest:
            raise ModelUsageAdmissionError("context manifest digest is invalid")
        required_text = (
            "provider",
            "route",
            "model",
            "reasoning",
            "command_semantic_version",
            "implementation_revision",
            "schema_version",
            "prompt_digest",
            "schema_digest",
            "system_digest",
            "evidence_package_digest",
        )
        if any(not isinstance(retained.get(field), str) for field in required_text):
            raise ModelUsageAdmissionError("context manifest identity is incomplete")
        evidence_package_bytes = retained.get("evidence_package_bytes")
        if not _is_int(evidence_package_bytes) or evidence_package_bytes <= 0:
            raise ModelUsageAdmissionError(
                "context manifest Evidence Package size is invalid"
            )
        zero_counts = (
            "prior_message_count",
            "skill_count",
            "tool_count",
            "mcp_server_count",
            "mcp_tool_count",
        )
        if any(retained.get(field) != 0 for field in zero_counts) or any(
            retained.get(field) is not False
            for field in ("skills_enabled", "tools_enabled", "mcp_enabled")
        ):
            raise ModelUsageAdmissionError(
                "context manifest contains an ambient capability"
            )
        forbidden = {
            "prompt",
            "system_prompt",
            "schema",
            "passages",
            "source_expression",
            "secret",
        }
        if forbidden.intersection(retained):
            raise ModelUsageAdmissionError("context manifest contains secret input")
        canonical_record = {"context_manifest_digest": digest, **retained}
        self._insert_exact(
            table="model_invocation_context_manifests",
            identity_column="context_manifest_digest",
            identity=digest,
            record=canonical_record,
            sql="INSERT INTO model_invocation_context_manifests("
            "context_manifest_digest,provider,route,evidence_package_digest,record_json) "
            "VALUES(?,?,?,?,?)",
            values=(
                digest,
                retained["provider"],
                retained["route"],
                retained["evidence_package_digest"],
                _json(canonical_record),
            ),
        )

    def retain_zero_call_admission(
        self, *, decision_id: str, decision: str, cycle_id: str, recorded_at: datetime
    ) -> None:
        if decision not in {"HOLD", "REJECT"}:
            raise ModelUsageIntegrityError("zero-call admission must be HOLD or REJECT")
        connection = self._connection()
        try:
            connection.execute(
                "INSERT OR IGNORE INTO model_zero_call_admissions("
                "decision_id,decision,cycle_id,recorded_at) VALUES(?,?,?,?)",
                (decision_id, decision, cycle_id, _utc_text(recorded_at)),
            )
            row = connection.execute(
                "SELECT decision,cycle_id,recorded_at FROM model_zero_call_admissions "
                "WHERE decision_id=?",
                (decision_id,),
            ).fetchone()
            if row is None or tuple(row) != (
                decision,
                cycle_id,
                _utc_text(recorded_at),
            ):
                raise ModelUsageIntegrityError("conflicting zero-call admission replay")
            connection.commit()
        finally:
            connection.close()

    def allocate(
        self,
        allocation: InvocationAllocation,
        *,
        owner_emergency_stop: bool,
    ) -> None:
        allocation._validate()
        if not isinstance(owner_emergency_stop, bool):
            raise ModelUsageAdmissionError(
                "owner emergency stop authority must be an explicit boolean"
            )
        if owner_emergency_stop:
            raise ModelUsageAdmissionError("owner emergency stop is active")
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            envelope = connection.execute(
                "SELECT cycle_id,record_json FROM model_work_envelopes WHERE envelope_id=?",
                (allocation.envelope_id,),
            ).fetchone()
            if envelope is None or str(envelope[0]) != allocation.cycle_id:
                raise ModelUsageAdmissionError("work envelope is absent or mismatched")
            policy_row = connection.execute(
                "SELECT record_json FROM model_invocation_policies WHERE canonical_digest=?",
                (allocation.invocation_policy_digest,),
            ).fetchone()
            if policy_row is None:
                raise ModelUsageAdmissionError("invocation policy is not registered")
            policy = _policy_from_record(_object(policy_row[0]))
            if not policy.qualified:
                raise ModelUsageAdmissionError("invocation policy is not qualified")
            self._validate_preflight(connection, allocation, policy)
            self._insert_allocation(connection, allocation)
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def allocate_graphiti_request(
        self,
        allocation: InvocationAllocation,
        *,
        identity: GraphitiInternalRequestIdentity,
        max_distinct_internal_requests: int,
    ) -> None:
        """Atomically retain one #728 leaf and its Graphiti request identity."""

        allocation._validate()
        if (
            isinstance(max_distinct_internal_requests, bool)
            or not isinstance(max_distinct_internal_requests, int)
            or max_distinct_internal_requests <= 0
        ):
            raise ModelUsageAdmissionError("Graphiti call-shape bound is invalid")
        identity.validate()
        record = identity.as_record()
        self._validate_graphiti_identity(allocation, identity)
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            envelope = connection.execute(
                "SELECT cycle_id FROM model_work_envelopes WHERE envelope_id=?",
                (allocation.envelope_id,),
            ).fetchone()
            if envelope is None or str(envelope[0]) != allocation.cycle_id:
                raise ModelUsageAdmissionError("work envelope is absent or mismatched")
            policy_row = connection.execute(
                "SELECT record_json FROM model_invocation_policies WHERE canonical_digest=?",
                (allocation.invocation_policy_digest,),
            ).fetchone()
            if policy_row is None:
                raise ModelUsageAdmissionError("invocation policy is not registered")
            policy = _policy_from_record(_object(policy_row[0]))
            if not policy.qualified:
                raise ModelUsageAdmissionError("invocation policy is not qualified")
            manifest_row = connection.execute(
                "SELECT record_json FROM model_invocation_context_manifests "
                "WHERE context_manifest_digest=?",
                (identity.context_manifest_digest,),
            ).fetchone()
            if manifest_row is None:
                raise ModelUsageAdmissionError("Graphiti context manifest is absent")
            manifest = _object(manifest_row[0])
            if (
                manifest.get("effective_revision_digest")
                != identity.effective_revision_digest
                or manifest.get("ingest_obligation_id")
                != identity.ingest_obligation_id
                or manifest.get("graphiti_attempt_id")
                != identity.graphiti_attempt_id
                or manifest.get("provider_attempt_id")
                != identity.provider_attempt_id
                or manifest.get("semantic_state_digest")
                != identity.semantic_state_digest
            ):
                raise ModelUsageAdmissionError(
                    "Graphiti context manifest differs from its request identity"
                )
            expected_environment_keys = (
                []
                if allocation.workload_class is WorkloadClass.GRAPHITI_EMBEDDING
                else [
                    *(
                        ["GROK_AUTH_PATH"]
                        if allocation.workload_class
                        is WorkloadClass.GRAPHITI_CHAT_FALLBACK
                        else []
                    ),
                    "HOME",
                    "LANG",
                    "LC_ALL",
                    "PATH",
                    "TMPDIR",
                    "XDG_CACHE_HOME",
                    "XDG_CONFIG_HOME",
                    "XDG_DATA_HOME",
                    "XDG_STATE_HOME",
                ]
            )
            if (
                manifest.get("working_directory_inventory") != []
                or manifest.get("working_directory_inventory_digest")
                != digest_canonical([])
                or manifest.get("environment_keys") != expected_environment_keys
            ):
                raise ModelUsageAdmissionError(
                    "Graphiti hermetic workspace proof differs from policy"
                )

            unavailable_event_digest = identity.primary_unavailable_event_digest
            if unavailable_event_digest is not None:
                _require_direct_fallback_authority(
                    connection, identity,
                )
                current_primary = self._route_state(
                    connection, "GRAPHITI_CHAT_PRIMARY",
                )
                if (
                    current_primary.get("state") != "OPEN"
                    or current_primary.get("event_digest")
                    != unavailable_event_digest
                ):
                    raise ModelUsageAdmissionError(
                        "Graphiti direct fallback primary authority differs"
                    )

            semantic_state_digest = str(record["semantic_state_digest"])
            reused_attempt_identity = connection.execute(
                "SELECT 1 FROM graphiti_internal_requests "
                "WHERE graphiti_attempt_id=? AND "
                "(internal_ordinal=? OR provider_attempt_id=?)",
                (
                    identity.graphiti_attempt_id,
                    identity.internal_ordinal,
                    identity.provider_attempt_id,
                ),
            ).fetchone()
            duplicate = connection.execute(
                "SELECT 1 FROM graphiti_internal_requests "
                "WHERE graphiti_attempt_id=? AND semantic_state_digest=?",
                (identity.graphiti_attempt_id, semantic_state_digest),
            ).fetchone()
            retained_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM graphiti_internal_requests "
                    "WHERE graphiti_attempt_id=?",
                    (identity.graphiti_attempt_id,),
                ).fetchone()[0]
            )
            reason_code = (
                "DUPLICATE_INTERNAL_REQUEST"
                if duplicate is not None
                else "GRAPHITI_ATTEMPT_IDENTITY_REUSE"
                if reused_attempt_identity is not None
                else "CALL_SHAPE_DRIFT"
                if retained_count >= max_distinct_internal_requests
                else None
            )
            if reason_code is not None:
                self._retain_graphiti_refusal(
                    connection,
                    allocation=allocation,
                    identity=record,
                    reason_code=reason_code,
                )
                if reason_code == "CALL_SHAPE_DRIFT":
                    self._append_route_state(
                        connection,
                        route=allocation.route,
                        state="OPEN",
                        reason=reason_code,
                        invocation_id=None,
                        recorded_at=allocation.allocated_at,
                    )
                connection.commit()
                raise ModelUsageAdmissionError(
                    "Graphiti internal request was refused before provider I/O",
                    reason_code=reason_code,
                )

            self._require_independent_graphiti_work(connection, allocation, identity)
            self._validate_preflight(
                connection, allocation, policy, graphiti_identity=identity,
            )
            self._insert_allocation(connection, allocation)
            connection.execute(
                "INSERT INTO graphiti_internal_requests("
                "canonical_digest,invocation_id,envelope_id,graphiti_attempt_id,"
                "internal_ordinal,"
                "semantic_state_digest,provider_attempt_id,call_shape_policy_digest,"
                "record_json) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    record["canonical_digest"],
                    allocation.invocation_id,
                    allocation.envelope_id,
                    identity.graphiti_attempt_id,
                    allocation.leaf_ordinal,
                    semantic_state_digest,
                    record["provider_attempt_id"],
                    record["call_shape_policy_digest"],
                    _json(record),
                ),
            )
            self.link_provider_attempt(
                invocation_id=allocation.invocation_id,
                provider_attempt_id=str(record["provider_attempt_id"]),
                linked_at=allocation.allocated_at,
                connection=connection,
            )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _insert_allocation(
        self, connection: sqlite3.Connection, allocation: InvocationAllocation
    ) -> None:
        record = allocation.as_record()
        try:
            connection.execute(
                "INSERT INTO model_invocation_allocations("
                "invocation_id,envelope_id,cycle_id,leaf_ordinal,workload_class,"
                "policy_digest,provider,route,model,request_digest,parent_invocation_id,"
                "allocated_at,canonical_digest,record_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    allocation.invocation_id,
                    allocation.envelope_id,
                    allocation.cycle_id,
                    allocation.leaf_ordinal,
                    allocation.workload_class.value,
                    allocation.invocation_policy_digest,
                    allocation.provider,
                    allocation.route,
                    allocation.model,
                    allocation.request_digest,
                    allocation.parent_invocation_id,
                    _utc_text(allocation.allocated_at),
                    allocation.canonical_digest,
                    _json(record),
                ),
            )
        except sqlite3.IntegrityError as exc:
            if "request_digest" in str(exc):
                raise ModelUsageAdmissionError(
                    "duplicate request digest in work envelope"
                ) from exc
            raise
        _refresh_current_usage(connection, allocation.invocation_id)

    @staticmethod
    def _validate_graphiti_identity(
        allocation: InvocationAllocation,
        identity: GraphitiInternalRequestIdentity,
    ) -> None:
        expected = {
            "invocation_id": allocation.invocation_id,
            "envelope_id": allocation.envelope_id,
            "internal_ordinal": allocation.leaf_ordinal,
            "provider": allocation.provider,
            "model": allocation.model,
            "reasoning": allocation.reasoning,
            "prompt_bytes": allocation.prompt_bytes,
            "prompt_digest": allocation.prompt_digest,
            "response_schema_digest": allocation.output_schema_digest,
            "requested_max_tokens": allocation.max_output_tokens,
            "context_manifest_digest": allocation.context_manifest_digest,
            "parent_invocation_id": allocation.parent_invocation_id,
            "invocation_policy_digest": allocation.invocation_policy_digest,
        }
        if any(getattr(identity, field) != value for field, value in expected.items()):
            raise ModelUsageAdmissionError(
                "Graphiti request identity differs from its #728 allocation"
            )

    def _retain_graphiti_refusal(
        self,
        connection: sqlite3.Connection,
        *,
        allocation: InvocationAllocation,
        identity: Mapping[str, object],
        reason_code: str,
    ) -> None:
        record = {
            "schema_version": "newsroom.graphiti-internal-request-refusal.v1",
            "envelope_id": allocation.envelope_id,
            "attempted_ordinal": allocation.leaf_ordinal,
            "route": allocation.route,
            "semantic_state_digest": identity["semantic_state_digest"],
            "call_shape_policy_digest": identity["call_shape_policy_digest"],
            "reason_code": reason_code,
            "refused_at": _utc_text(allocation.allocated_at),
        }
        refusal_digest = digest_canonical(record)
        connection.execute(
            "INSERT OR IGNORE INTO graphiti_internal_request_refusals("
            "refusal_digest,envelope_id,attempted_ordinal,route,"
            "semantic_state_digest,call_shape_policy_digest,reason_code,refused_at,"
            "record_json) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                refusal_digest,
                allocation.envelope_id,
                allocation.leaf_ordinal,
                allocation.route,
                identity["semantic_state_digest"],
                identity["call_shape_policy_digest"],
                reason_code,
                record["refused_at"],
                _json({**record, "refusal_digest": refusal_digest}),
            ),
        )

    def graphiti_request_records(
        self, *, envelope_id: str
    ) -> dict[str, list[dict[str, object]]]:
        connection = self._connection()
        try:
            requests = [
                _object(row[0])
                for row in connection.execute(
                    "SELECT record_json FROM graphiti_internal_requests "
                    "WHERE envelope_id=? ORDER BY internal_ordinal,canonical_digest",
                    (envelope_id,),
                )
            ]
            refusals = [
                _object(row[0])
                for row in connection.execute(
                    "SELECT record_json FROM graphiti_internal_request_refusals "
                    "WHERE envelope_id=? ORDER BY refused_at,refusal_digest",
                    (envelope_id,),
                )
            ]
        finally:
            connection.close()
        return {"requests": requests, "refusals": refusals}

    def graphiti_ingest_pre_dispatch_zero(self, *, ingest_id: str) -> bool:
        """Prove every retained model allocation for one ingest stopped locally."""

        evidence, allocation_count = self._graphiti_ingest_retry_evidence(
            ingest_id=ingest_id
        )
        return bool(
            allocation_count
            and not evidence.unresolved_attempts
            and not evidence.settled_provider_attempts
            and evidence.zero_dispatch_attempts
        )

    def graphiti_ingest_retry_evidence(
        self, *, ingest_id: str, before_attempt_number: int | None = None
    ) -> GraphitiIngestRetryEvidence:
        """Return exact settled/zero/unresolved evidence for retained attempts."""

        evidence, _allocation_count = self._graphiti_ingest_retry_evidence(
            ingest_id=ingest_id, before_attempt_number=before_attempt_number
        )
        return evidence

    def graphiti_ingest_allocation_count(self, *, ingest_id: str) -> int:
        """Authenticate allocated leaves, independent of an open controller envelope."""
        _evidence, allocation_count = self._graphiti_ingest_retry_evidence(
            ingest_id=ingest_id,
        )
        return allocation_count

    def graphiti_ingest_retry_evidence_many(
        self, *, ingest_ids: tuple[str, ...],
    ) -> dict[str, GraphitiIngestRetryEvidence]:
        """Authenticate one queue's retry history once in one read snapshot."""
        return {
            ingest_id: evidence
            for ingest_id, (evidence, _) in self._graphiti_ingest_retry_evidence_batch(
                ingest_ids=ingest_ids,
            ).items()
        }

    def native_graphiti_ingest_retry_evidence_many(
        self, *, failed_attempts: Mapping[str, int], max_attempts: int,
        _allow_missing_work_outcome_attempts: Mapping[str, int] | None = None,
    ) -> dict[str, GraphitiIngestRetryEvidence]:
        """Read only the finite native attempt allowance, never unrelated history.

        The caller proves native unit identity and supplies its raw failure count.
        Deterministic envelope keys and independent internal-request bindings keep
        missing or retargeted evidence from becoming optimistic retry credits.
        """
        if type(max_attempts) is not int or max_attempts <= 0 or any(
            type(count) is not int or count < 0 for count in failed_attempts.values()
        ):
            raise ModelUsageIntegrityError("native Graphiti retry allowance differs")
        native_attempts = {}
        for ingest_id in failed_attempts:
            _token(ingest_id, field="Graphiti ingest id")
            for number in range(1, max_attempts + 1):
                envelope = WorkEnvelope.create(
                    cycle_id=native_graphiti_usage_cycle_id(
                        ingest_id=ingest_id, attempt_number=number,
                    ),
                    workload_class=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
                    # Admission time is not part of the envelope's identity.
                    admitted_at=datetime(1970, 1, 1, tzinfo=UTC),
                    admission_decision_id=None, candidate_id=None,
                    hypothesis_digest=None, evidence_package_digest=None,
                    ingest_id=ingest_id, graphiti_attempt_id=f"{ingest_id}:{number}",
                )
                native_attempts[envelope.envelope_id] = (ingest_id, number)
        result = {
            ingest_id: evidence
            for ingest_id, (evidence, _) in self._graphiti_ingest_retry_evidence_batch(
                ingest_ids=tuple(failed_attempts), native_attempts=native_attempts,
                native_failed_attempts=failed_attempts,
                allow_missing_work_outcome_envelopes=frozenset(
                    WorkEnvelope.create(
                        cycle_id=native_graphiti_usage_cycle_id(
                            ingest_id=ingest_id, attempt_number=number,
                        ),
                        workload_class=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
                        admitted_at=datetime(1970, 1, 1, tzinfo=UTC),
                        admission_decision_id=None, candidate_id=None,
                        hypothesis_digest=None, evidence_package_digest=None,
                        ingest_id=ingest_id,
                        graphiti_attempt_id=f"{ingest_id}:{number}",
                    ).envelope_id
                    for ingest_id, number in (
                        _allow_missing_work_outcome_attempts or {}
                    ).items()
                ),
            ).items()
        }
        # A separately accounted semantic verifier is not one of the original
        # primary envelopes. Its paid dispatch must never become zero-call credit.
        selected = [
            {"ingest": ingest, "attempt": number,
             "cycle": native_graphiti_usage_cycle_id(ingest_id=ingest, attempt_number=number)}
            for ingest in failed_attempts for number in range(1, max_attempts + 1)
        ]
        connection = self._connection()
        try:
            connection.execute("BEGIN")
            rows = connection.execute(
                "SELECT json_extract(w.value,'$.ingest'),json_extract(w.value,'$.attempt'),"
                "a.invocation_id,e.envelope_id,e.cycle_id,e.workload_class,e.admitted_at,"
                "e.canonical_digest,e.record_json,m.context_manifest_digest,m.provider,"
                "m.route,m.evidence_package_digest,m.record_json "
                "FROM json_each(?) w JOIN model_invocation_allocations a "
                "ON a.cycle_id=json_extract(w.value,'$.cycle') "
                "LEFT JOIN model_work_envelopes e USING(envelope_id) "
                "LEFT JOIN model_invocation_context_manifests m "
                "ON m.context_manifest_digest=json_extract(a.record_json,'$.context_manifest_digest') "
                "WHERE a.workload_class=?",
                (json.dumps(selected), WorkloadClass.TYPESAFE_JUDGMENT.value),
            ).fetchall()
            reported_verifiers: dict[tuple[str, int], bool] = {}
            for row in rows:
                ingest, number, invocation_id = row[:3]
                expected_attempt = f"{ingest}:{number}"
                envelope_record = _object(row[8]) if row[8] is not None else {}
                manifest = _object(row[13]) if row[13] is not None else {}
                # The independent caller/envelope bindings select corrupt records
                # too: an edited role must not hide a paid Graphiti leaf.
                if (envelope_record.get("graphiti_attempt_id") != expected_attempt
                        and manifest.get("caller_graphiti_attempt_id") != expected_attempt
                        and manifest.get("caller_identity") != "GRAPHITI_VERIFIER"):
                    continue
                proved_reported = False
                if connection.execute(
                    "SELECT 1 FROM model_invocation_terminals WHERE invocation_id=?",
                    (invocation_id,),
                ).fetchone() is not None:
                    allocation, terminal = _retained_terminal_allocation(connection, invocation_id)
                    policy = _policy_for_allocation(connection, allocation)
                    self._validate_terminal(terminal, allocation.workload_class, policy,
                        requested_max_output_tokens=allocation.max_output_tokens)
                    envelope = _envelope_from_record(envelope_record)
                    unsigned = dict(manifest)
                    manifest_digest = unsigned.pop("context_manifest_digest", None)
                    if (tuple(row[3:8]) != (envelope.envelope_id, envelope.cycle_id,
                            envelope.workload_class.value, _utc_text(envelope.admitted_at),
                            envelope.canonical_digest)
                            or row[8] != _json(envelope.as_record())
                            or envelope.envelope_id != allocation.envelope_id
                            or envelope.cycle_id != allocation.cycle_id
                            or envelope.workload_class is not WorkloadClass.TYPESAFE_JUDGMENT
                            or envelope.ingest_id != ingest
                            or envelope.graphiti_attempt_id != expected_attempt
                            or envelope.candidate_id is not None
                            or tuple(row[9:13]) != (allocation.context_manifest_digest,
                                allocation.provider, allocation.route, envelope.evidence_package_digest)
                            or row[13] != _json(manifest)
                            or manifest_digest != allocation.context_manifest_digest
                            or digest_canonical(unsigned) != manifest_digest
                            or manifest.get("caller_identity") != "GRAPHITI_VERIFIER"
                            or manifest.get("caller_ingest_id") != ingest
                            or manifest.get("caller_graphiti_attempt_id") != expected_attempt
                            or any(manifest.get(key) != getattr(allocation, key) for key in (
                                "provider", "route", "model", "reasoning", "prompt_bytes",
                                "prompt_digest", "request_digest", "output_schema_digest",
                                "prompt_contract_version", "context_identity", "config_identity"))):
                        raise ModelUsageIntegrityError("retained Graphiti verifier binding differs")
                    if _is_exact_pre_dispatch_zero(terminal):
                        proved_reported = not _has_exact_dispatch(connection, terminal)
                    elif terminal.usage_status is UsageStatus.REPORTED and terminal.dispatch_at is not None:
                        _require_reported_telemetry(connection, terminal)
                        proved_reported = _has_exact_dispatch(connection, terminal)
                key = (ingest, number)
                reported_verifiers[key] = reported_verifiers.get(key, True) and proved_reported
        finally:
            connection.close()
        for (ingest, number), proved_reported in reported_verifiers.items():
            evidence = result[ingest]
            if proved_reported and number in evidence.settled_provider_attempts:
                # Known paid usage preserves a whole attempt already settled by
                # the original Graphiti protocol; it never grants graph settlement.
                continue
            result[ingest] = GraphitiIngestRetryEvidence(
                attempt_numbers=tuple(sorted(set(evidence.attempt_numbers) | {number})),
                zero_dispatch_attempts=tuple(n for n in evidence.zero_dispatch_attempts if n != number),
                settled_provider_attempts=tuple(n for n in evidence.settled_provider_attempts if n != number),
                latest_settled_provider_attempt=max((n for n in evidence.settled_provider_attempts if n != number), default=None),
                unresolved_attempts=tuple(sorted(set(evidence.unresolved_attempts) | {number})),
            )
        return result

    def native_recovered_ambiguous_usage_evidence_digest(
        self,
        *,
        ingest_id: str,
        authoritative_attempt_number: int,
        skipped_attempt_number: int,
        skipped_receipt_digest: str,
        skipped_recorded_at: datetime,
        latest_allowed_attempt_number: int | None = None,
        retained_reentry_attempt_number: int | None = None,
    ) -> str:
        """Authenticate settled use followed only by one local binding refusal."""

        latest_allowed = (
            skipped_attempt_number
            if latest_allowed_attempt_number is None
            else latest_allowed_attempt_number
        )
        if (
            type(authoritative_attempt_number) is not int
            or authoritative_attempt_number <= 0
            or skipped_attempt_number != authoritative_attempt_number + 1
            or type(latest_allowed) is not int
            or latest_allowed < skipped_attempt_number
            or skipped_recorded_at.tzinfo is None
            or skipped_recorded_at.utcoffset() is None
            or (
                retained_reentry_attempt_number is not None
                and retained_reentry_attempt_number != latest_allowed
            )
        ):
            raise ModelUsageIntegrityError(
                "recovered ambiguous usage attempt sequence differs"
            )
        try:
            validate_sha256_digest(
                skipped_receipt_digest, field="skipped receipt digest"
            )
        except ValueError as exc:
            raise ModelUsageIntegrityError(str(exc)) from exc
        evidence = self.native_graphiti_ingest_retry_evidence_many(
            failed_attempts={ingest_id: latest_allowed},
            max_attempts=latest_allowed,
            _allow_missing_work_outcome_attempts=(
                None
                if retained_reentry_attempt_number is None
                else {ingest_id: retained_reentry_attempt_number}
            ),
        )[ingest_id]
        if (
            authoritative_attempt_number not in evidence.settled_provider_attempts
            or len(evidence.settled_provider_attempts) >= GRAPHITI_MAX_FAILURES
            or any(number > latest_allowed for number in evidence.attempt_numbers)
        ):
            raise ModelUsageIntegrityError(
                "recovered ambiguous usage is not exactly settled"
            )

        cycle_id = native_graphiti_usage_cycle_id(
            ingest_id=ingest_id, attempt_number=skipped_attempt_number
        )
        envelope = WorkEnvelope.create(
            cycle_id=cycle_id,
            workload_class=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            admitted_at=datetime(1970, 1, 1, tzinfo=UTC),
            admission_decision_id=None,
            candidate_id=None,
            hypothesis_digest=None,
            evidence_package_digest=None,
            ingest_id=ingest_id,
            graphiti_attempt_id=f"{ingest_id}:{skipped_attempt_number}",
        )
        connection = self._connection()
        try:
            connection.execute("BEGIN")
            replay_proofs = []
            invalid_replay = False
            for number in evidence.attempt_numbers:
                if number >= authoritative_attempt_number:
                    continue
                claimed, replay_proof = _native_immutable_replay_proof(
                    connection,
                    ingest_id=ingest_id,
                    attempt_number=number,
                    evidence=evidence,
                )
                if claimed and replay_proof is None:
                    invalid_replay = True
                elif replay_proof is not None:
                    replay_proofs.append(replay_proof)
            authenticated_replays = {
                int(proof["attempt_number"]) for proof in replay_proofs
            }
            unresolved_attempts = tuple(
                number
                for number in evidence.unresolved_attempts
                if number not in authenticated_replays
            )
            if invalid_replay or unresolved_attempts != (skipped_attempt_number,):
                raise ModelUsageIntegrityError(
                    "recovered ambiguous usage is not exactly settled"
                )
            retained_envelope = connection.execute(
                "SELECT record_json FROM model_work_envelopes WHERE envelope_id=?",
                (envelope.envelope_id,),
            ).fetchone()
            outcome = connection.execute(
                "SELECT outcome_digest,envelope_id,outcome,terminal_at,record_json "
                "FROM model_work_outcomes WHERE envelope_id=?",
                (envelope.envelope_id,),
            ).fetchone()
            leaf_count = connection.execute(
                "SELECT COUNT(*) FROM model_invocation_allocations WHERE envelope_id=?",
                (envelope.envelope_id,),
            ).fetchone()[0]
            request_count = connection.execute(
                "SELECT COUNT(*) FROM graphiti_internal_requests WHERE envelope_id=?",
                (envelope.envelope_id,),
            ).fetchone()[0]
            refusal_count = connection.execute(
                "SELECT COUNT(*) FROM graphiti_internal_request_refusals "
                "WHERE envelope_id=?",
                (envelope.envelope_id,),
            ).fetchone()[0]
            later_allocations = connection.execute(
                "SELECT e.record_json FROM model_work_envelopes e JOIN "
                "model_invocation_allocations a USING(envelope_id) "
                "WHERE json_extract(e.record_json,'$.ingest_id')=?",
                (ingest_id,),
            ).fetchall()
            for row in later_allocations:
                later = _envelope_from_record(_object(row[0]))
                prefix, separator, suffix = str(
                    later.graphiti_attempt_id or ""
                ).rpartition(":")
                if (
                    separator != ":"
                    or prefix != ingest_id
                    or not suffix.isdigit()
                    or int(suffix) > latest_allowed
                ):
                    raise ModelUsageIntegrityError(
                        "recovered ambiguous later allocation is retained"
                    )
            if retained_envelope is None or outcome is None:
                raise ModelUsageIntegrityError(
                    "recovered ambiguous skipped usage record is absent"
                )
            envelope_record = _object(retained_envelope[0])
            outcome_record = _object(outcome[4])
            unsigned_outcome = dict(outcome_record)
            outcome_digest = unsigned_outcome.pop("outcome_digest", None)
            if (
                _envelope_from_record(envelope_record).envelope_id
                != envelope.envelope_id
                or outcome_digest != outcome[0]
                or digest_canonical(unsigned_outcome) != outcome_digest
                or tuple(outcome[1:4])
                != (
                    outcome_record.get("envelope_id"),
                    outcome_record.get("outcome"),
                    outcome_record.get("terminal_at"),
                )
                or outcome_record.get("schema_version")
                != MODEL_USAGE_SCHEMA_VERSION
                or outcome_record.get("envelope_id") != envelope.envelope_id
                or outcome_record.get("outcome") != "GRAPHITI_REJECTED_BINDING"
                or outcome_record.get("outcome_record_id")
                != skipped_receipt_digest
                or outcome_record.get("payload_digest") is not None
                or outcome_record.get("cycle_outcome") is not None
                or outcome_record.get("route_circuit_state") is not None
                or outcome_record.get("route_circuit_reason") is not None
                or outcome_record.get("retained_proposal_count") != 0
                or outcome_record.get("accepted_provider_attempt_id") is not None
                or outcome_record.get("stable_reason_codes") != []
                or any((leaf_count, request_count, refusal_count))
            ):
                raise ModelUsageIntegrityError(
                    "recovered ambiguous skipped usage record differs"
                )

            settled_records: list[dict[str, object]] = []
            authoritative_cycle = native_graphiti_usage_cycle_id(
                ingest_id=ingest_id,
                attempt_number=authoritative_attempt_number,
            )
            for row in connection.execute(
                "SELECT record_json FROM model_work_envelopes WHERE cycle_id=? "
                "UNION ALL SELECT record_json FROM model_work_outcomes "
                "WHERE envelope_id IN (SELECT envelope_id FROM model_work_envelopes "
                "WHERE cycle_id=?) UNION ALL SELECT record_json FROM "
                "model_invocation_allocations WHERE cycle_id=? UNION ALL SELECT "
                "t.record_json FROM model_invocation_terminals t JOIN "
                "model_invocation_allocations a USING(invocation_id) WHERE a.cycle_id=? "
                "UNION ALL SELECT r.record_json FROM graphiti_internal_requests r "
                "JOIN model_invocation_allocations a USING(invocation_id) "
                "WHERE a.cycle_id=? UNION ALL SELECT p.record_json FROM "
                "model_provider_telemetry p JOIN model_invocation_allocations a "
                "USING(invocation_id) WHERE a.cycle_id=? UNION ALL SELECT "
                "o.record_json FROM model_transport_observations o JOIN "
                "model_invocation_allocations a USING(invocation_id) "
                "WHERE a.cycle_id=? UNION ALL SELECT r.record_json FROM "
                "model_usage_reconciliations r JOIN model_invocation_allocations a "
                "USING(invocation_id) WHERE a.cycle_id=? UNION ALL SELECT "
                "d.record_json FROM model_usage_conservative_dispositions d JOIN "
                "model_invocation_allocations a USING(invocation_id) "
                "WHERE a.cycle_id=?",
                (authoritative_cycle,) * 9,
            ):
                settled_records.append(_object(row[0]))
            if not settled_records:
                raise ModelUsageIntegrityError(
                    "recovered ambiguous settled usage records are absent"
                )
            digest_evidence: dict[str, object] = {
                "ingest_id": ingest_id,
                "authoritative_attempt_number": authoritative_attempt_number,
                "skipped_attempt_number": skipped_attempt_number,
                "authoritative_attempt_settled": True,
                "skipped_attempt_unresolved": True,
                "settled_records": sorted(settled_records, key=digest_canonical),
                "skipped_envelope": envelope_record,
                "skipped_outcome": outcome_record,
            }
            if replay_proofs:
                digest_evidence["immutable_replay_proofs"] = replay_proofs
            return digest_canonical(digest_evidence)
        finally:
            connection.close()

    def _graphiti_ingest_retry_evidence(
        self, *, ingest_id: str, before_attempt_number: int | None = None,
    ) -> tuple[GraphitiIngestRetryEvidence, int]:
        return self._graphiti_ingest_retry_evidence_batch(
            ingest_ids=(ingest_id,), before_attempt_number=before_attempt_number,
        )[ingest_id]

    def _graphiti_ingest_retry_evidence_batch(
        self, *, ingest_ids: tuple[str, ...], before_attempt_number: int | None = None,
        native_attempts: Mapping[str, tuple[str, int]] | None = None,
        native_failed_attempts: Mapping[str, int] | None = None,
        allow_missing_work_outcome_envelopes: frozenset[str] = frozenset(),
    ) -> dict[str, tuple[GraphitiIngestRetryEvidence, int]]:
        ingest_ids = tuple(dict.fromkeys(ingest_ids))
        for ingest_id in ingest_ids:
            _token(ingest_id, field="Graphiti ingest id")
        if not ingest_ids:
            return {}
        if before_attempt_number is not None and (
            type(before_attempt_number) is not int or before_attempt_number <= 0
        ):
            raise ModelUsageIntegrityError("Graphiti retry boundary differs")
        connection = self._connection()
        try:
            connection.execute("BEGIN")
            envelope_filter = outcome_filter = allocation_filter = ""
            envelope_parameters: tuple[str, ...] = ()
            allocation_parameters: tuple[str, ...] = ()
            request_rows = ()
            if native_attempts is not None:
                envelope_parameters = tuple(native_attempts)
                keys = ",".join("?" for _ in envelope_parameters)
                attempt_parameters = tuple(
                    f"{ingest_id}:{number}" for ingest_id, number in native_attempts.values()
                )
                envelope_filter = outcome_filter = f"WHERE envelope_id IN ({keys}) "
                # Both independently retained identities are indexed. Either side
                # still selects the leaf when the other side has been retargeted.
                allocation_filter = (
                    "WHERE a.invocation_id IN (SELECT invocation_id "
                    f"FROM model_invocation_allocations WHERE envelope_id IN ({keys}) "
                    "UNION SELECT invocation_id FROM graphiti_internal_requests "
                    f"WHERE graphiti_attempt_id IN ({keys})) "
                )
                allocation_parameters = envelope_parameters + attempt_parameters
                request_rows = connection.execute(
                    "SELECT canonical_digest,invocation_id,envelope_id,graphiti_attempt_id,"
                    "internal_ordinal,semantic_state_digest,provider_attempt_id,"
                    "call_shape_policy_digest,record_json FROM graphiti_internal_requests "
                    f"WHERE graphiti_attempt_id IN ({keys}) OR invocation_id IN "
                    "(SELECT invocation_id FROM model_invocation_allocations "
                    f"WHERE envelope_id IN ({keys}))",
                    attempt_parameters + envelope_parameters,
                )
            envelope_rows = connection.execute(
                "SELECT envelope_id,cycle_id,workload_class,admitted_at,"
                "canonical_digest,record_json FROM model_work_envelopes "
                + envelope_filter + "ORDER BY envelope_id", envelope_parameters,
            )
            envelopes: dict[str, WorkEnvelope] = {}
            attempts: dict[str, int] = {}
            for row in envelope_rows:
                envelope = _envelope_from_record(_object(row[5]))
                if tuple(row[index] for index in range(5)) != (
                    envelope.envelope_id,
                    envelope.cycle_id,
                    envelope.workload_class.value,
                    _utc_text(envelope.admitted_at),
                    envelope.canonical_digest,
                ):
                    raise ModelUsageIntegrityError(
                        "retained Graphiti envelope binding differs"
                    )
                envelopes[envelope.envelope_id] = envelope
                if envelope.ingest_id not in ingest_ids:
                    continue
                prefix, separator, suffix = str(
                    envelope.graphiti_attempt_id or ""
                ).rpartition(":")
                if (
                    envelope.workload_class
                    not in {
                        WorkloadClass.GRAPHITI_CHAT_PRIMARY,
                        WorkloadClass.GRAPHITI_CHAT_FALLBACK,
                        WorkloadClass.GRAPHITI_EMBEDDING,
                    }
                    or separator != ":"
                    or prefix != envelope.ingest_id
                    or not suffix.isdigit()
                    or int(suffix) <= 0
                ):
                    raise ModelUsageIntegrityError(
                        "retained Graphiti envelope binding differs"
                    )
                attempt = int(suffix)
                if any(
                    number == attempt and envelopes[key].ingest_id == envelope.ingest_id
                    for key, number in attempts.items()
                ):
                    raise ModelUsageIntegrityError(
                        "retained Graphiti attempt identity is duplicated"
                    )
                attempts[envelope.envelope_id] = attempt

            work_outcomes: set[str] = set()
            outcome_attempts = attempts if native_attempts is None else native_attempts
            for row in connection.execute(
                "SELECT outcome_digest,envelope_id,outcome,terminal_at,record_json "
                "FROM model_work_outcomes " + outcome_filter + "ORDER BY envelope_id",
                envelope_parameters,
            ):
                record = _object(row[4])
                if (str(row[1]) not in outcome_attempts
                        and record.get("envelope_id") not in outcome_attempts):
                    continue
                unsigned = dict(record)
                retained_digest = unsigned.pop("outcome_digest", None)
                if (
                    retained_digest != row[0]
                    or digest_canonical(unsigned) != retained_digest
                    or (row[1], row[2], row[3])
                    != (
                        record.get("envelope_id"),
                        record.get("outcome"),
                        record.get("terminal_at"),
                    )
                ):
                    raise ModelUsageIntegrityError(
                        "retained Graphiti work outcome binding differs"
                    )
                work_outcomes.add(str(row[1]))

            allocation_rows = connection.execute(
                "SELECT a.invocation_id,a.envelope_id,a.cycle_id,a.leaf_ordinal,"
                "a.workload_class,a.policy_digest,a.provider,a.route,a.model,"
                "a.request_digest,a.parent_invocation_id,a.allocated_at,"
                "a.canonical_digest,a.record_json,t.invocation_id,t.usage_status,"
                "t.outcome,t.failure_class,t.completed_at,t.terminal_digest,"
                "t.record_json FROM model_invocation_allocations a "
                "LEFT JOIN model_invocation_terminals t "
                "ON t.invocation_id=a.invocation_id "
                + allocation_filter + "ORDER BY a.envelope_id,a.leaf_ordinal",
                allocation_parameters,
            )
            requests = {}
            for row in request_rows:
                record = _object(row[8])
                try:
                    values = dict(record)
                    values.pop("schema_version", None)
                    values["leaf_class"] = GraphitiLeafClass(values["leaf_class"])
                    identity = GraphitiInternalRequestIdentity.create(**values)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ModelUsageIntegrityError(
                        "retained Graphiti request identity differs"
                    ) from exc
                expected_attempt = (native_attempts or {}).get(identity.envelope_id)
                if identity.as_record() != record or tuple(row[:8]) != (
                    identity.canonical_digest, identity.invocation_id, identity.envelope_id,
                    identity.graphiti_attempt_id, identity.internal_ordinal,
                    identity.semantic_state_digest, identity.provider_attempt_id,
                    identity.call_shape_policy_digest,
                ) or expected_attempt is None or (
                    identity.ingest_obligation_id != expected_attempt[0]
                    or identity.graphiti_attempt_id != f"{expected_attempt[0]}:{expected_attempt[1]}"
                ):
                    raise ModelUsageIntegrityError("retained Graphiti request binding differs")
                if identity.primary_unavailable_event_digest is not None:
                    try:
                        _require_direct_fallback_authority(connection, identity)
                    except ModelUsageAdmissionError as exc:
                        raise ModelUsageIntegrityError(str(exc)) from exc
                requests[identity.invocation_id] = identity
            incomplete_envelopes: set[str] = set()
            by_attempt: dict[
                str, list[tuple[InvocationAllocation, InvocationTerminal | None]]
            ] = {envelope_id: [] for envelope_id in attempts}
            for row in allocation_rows:
                allocation = _allocation_from_record(_object(row[13]))
                if tuple(row[index] for index in range(13)) != (
                    allocation.invocation_id,
                    allocation.envelope_id,
                    allocation.cycle_id,
                    allocation.leaf_ordinal,
                    allocation.workload_class.value,
                    allocation.invocation_policy_digest,
                    allocation.provider,
                    allocation.route,
                    allocation.model,
                    allocation.request_digest,
                    allocation.parent_invocation_id,
                    _utc_text(allocation.allocated_at),
                    allocation.canonical_digest,
                ):
                    raise ModelUsageIntegrityError(
                        "retained Graphiti allocation binding differs"
                    )
                envelope = envelopes.get(allocation.envelope_id)
                if envelope is None or allocation.cycle_id != envelope.cycle_id:
                    raise ModelUsageIntegrityError(
                        "retained Graphiti allocation envelope differs"
                    )
                attempt = attempts.get(allocation.envelope_id)
                if attempt is None:
                    continue
                if allocation.workload_class not in {
                    WorkloadClass.GRAPHITI_CHAT_PRIMARY,
                    WorkloadClass.GRAPHITI_CHAT_FALLBACK,
                    WorkloadClass.GRAPHITI_EMBEDDING,
                }:
                    raise ModelUsageIntegrityError(
                        "retained Graphiti allocation workload differs"
                    )
                terminal = None
                if row[14] is not None and row[20] is not None:
                    terminal = _terminal_from_record(_object(row[20]))
                    if tuple(row[index] for index in range(14, 20)) != (
                        terminal.invocation_id,
                        terminal.usage_status.value,
                        terminal.outcome,
                        terminal.failure_class,
                        _utc_text(terminal.completed_at),
                        terminal.terminal_digest,
                    ) or terminal.invocation_id != allocation.invocation_id:
                        raise ModelUsageIntegrityError(
                            "retained Graphiti terminal binding differs"
                        )
                if native_attempts is not None:
                    identity = requests.pop(allocation.invocation_id, None)
                    if identity is None:
                        incomplete_envelopes.add(allocation.envelope_id)
                    else:
                        try:
                            self._validate_graphiti_identity(allocation, identity)
                        except ModelUsageAdmissionError as exc:
                            raise ModelUsageIntegrityError(str(exc)) from exc
                by_attempt[allocation.envelope_id].append((allocation, terminal))
            incomplete_envelopes.update(identity.envelope_id for identity in requests.values())

            result = {}
            for ingest_id in ingest_ids:
                selected_attempts = {
                    envelope_id: attempt
                    for envelope_id, attempt in attempts.items()
                    if envelopes[envelope_id].ingest_id == ingest_id
                    and (before_attempt_number is None or attempt < before_attempt_number)
                }
                zero: list[int] = []
                settled: list[int] = []
                unresolved: list[int] = []
                for envelope_id, attempt in sorted(
                    selected_attempts.items(), key=lambda item: item[1]
                ):
                    leaves = by_attempt[envelope_id]
                    if (
                        envelope_id not in work_outcomes
                        and envelope_id not in allow_missing_work_outcome_envelopes
                    ) or envelope_id in incomplete_envelopes:
                        unresolved.append(attempt)
                        continue
                    if not leaves:
                        # A terminal controller refusal has no provider allocation.
                        # Only the new authenticated active-other-owner refusal
                        # supplies structural zero; generic absence stays UNKNOWN.
                        if native_attempts is not None:
                            if _native_workspace_busy_zero(connection, envelope=envelopes[envelope_id], attempt_number=attempt):
                                zero.append(attempt)
                            else:
                                unresolved.append(attempt)
                        continue
                    attempt_zero = True
                    attempt_dispatched = False
                    attempt_unresolved = False
                    for allocation, terminal in leaves:
                        if terminal is None:
                            attempt_unresolved = True
                            continue
                        policy = _policy_for_allocation(connection, allocation)
                        self._validate_terminal(
                            terminal,
                            allocation.workload_class,
                            policy,
                            requested_max_output_tokens=allocation.max_output_tokens,
                        )
                        if _is_exact_pre_dispatch_zero(terminal):
                            if _has_exact_dispatch(connection, terminal):
                                attempt_unresolved = True
                            continue
                        attempt_zero = False
                        if terminal.dispatch_at is None or not _has_exact_dispatch(
                            connection, terminal
                        ):
                            attempt_unresolved = True
                            continue
                        if terminal.usage_status is UsageStatus.REPORTED:
                            _require_reported_telemetry(connection, terminal)
                            attempt_dispatched = True
                        elif _valid_native_disposition(
                            connection,
                            allocation=allocation,
                            terminal=terminal,
                            validated_native_envelope=(
                                envelopes[allocation.envelope_id]
                                if native_attempts is not None else None
                            ),
                        ) is not None:
                            attempt_dispatched = True
                        else:
                            attempt_unresolved = True
                    if attempt_unresolved:
                        unresolved.append(attempt)
                    elif attempt_zero:
                        zero.append(attempt)
                    elif attempt_dispatched:
                        settled.append(attempt)
                    else:
                        unresolved.append(attempt)

                missing = {
                    number for envelope_id, (selected_ingest, number) in
                    (native_attempts or {}).items()
                    if selected_ingest == ingest_id and envelope_id not in attempts
                    and (
                        number <= (native_failed_attempts or {})[ingest_id]
                        or envelope_id in work_outcomes
                        or envelope_id in incomplete_envelopes
                    )
                }
                attempt_numbers = tuple(sorted(set(selected_attempts.values()) | missing))
                unresolved = sorted(set(unresolved) | missing)
                settled_attempts = tuple(settled)
                retry = GraphitiIngestRetryEvidence(
                    attempt_numbers=attempt_numbers,
                    zero_dispatch_attempts=tuple(zero),
                    settled_provider_attempts=settled_attempts,
                    latest_settled_provider_attempt=(
                        settled_attempts[-1] if settled_attempts else None
                    ),
                    unresolved_attempts=tuple(unresolved),
                )
                if native_attempts is not None:
                    # All consumers, including the queue, must see the same
                    # authenticated provider-free replay classification.
                    for number in retry.unresolved_attempts:
                        _claimed, replay = _native_immutable_replay_proof(
                            connection, ingest_id=ingest_id,
                            attempt_number=number, evidence=retry,
                        )
                        if replay is not None:
                            retry = replace(
                                retry,
                                zero_dispatch_attempts=tuple(sorted((*retry.zero_dispatch_attempts, number))),
                                unresolved_attempts=tuple(n for n in retry.unresolved_attempts if n != number),
                            )
                result[ingest_id] = (
                    retry,
                    sum(len(leaves) for key, leaves in by_attempt.items()
                        if envelopes[key].ingest_id == ingest_id),
                )
            return result
        finally:
            connection.close()

    def next_graphiti_internal_ordinal(self, *, graphiti_attempt_id: str) -> int:
        """Return the next durable leaf ordinal for one Graphiti attempt."""

        _token(graphiti_attempt_id, field="Graphiti attempt id")
        connection = self._connection()
        try:
            row = connection.execute(
                "SELECT COALESCE(MAX(internal_ordinal),0) "
                "FROM graphiti_internal_requests WHERE graphiti_attempt_id=?",
                (graphiti_attempt_id,),
            ).fetchone()
        finally:
            connection.close()
        return int(row[0]) + 1

    @staticmethod
    def _independent_typesafe_transport_work(connection, allocation, envelope, route_state):
        """Quarantine exact unknown work, never settle cash or reopen the route.

        No Source-wide clearance is inferred: the retained caller IDs are the
        narrowest available boundary. Same candidate/ingest remains held even
        when its cycle, questions or attempt are changed. Unknown HTTP transport
        failures additionally quarantine exact evidence/request identity and
        allow independent work only after five minutes, one active leaf at a time.
        Proven authentication/payment/rate-limit failures do not use this gate.
        """
        failures = {"TimeoutError", "HTTPError", "HTTPError:400", "HTTPError:422",
                    "HTTPError:500", "HTTPError:502", "HTTPError:503", "HTTPError:504"}
        if (allocation.workload_class is not WorkloadClass.TYPESAFE_JUDGMENT
                or allocation.provider != "typesafe" or allocation.route != "TYPESAFE_JUDGMENT"
                or route_state.get("reason") not in failures):
            return False
        event = connection.execute(
            "SELECT event_digest,route,state,reason,invocation_id,recorded_at,record_json "
            "FROM model_usage_route_circuit_events WHERE route='TYPESAFE_JUDGMENT' "
            "ORDER BY recorded_at DESC,rowid DESC LIMIT 1"
        ).fetchone()
        if event is None:
            return False
        record = _object(event[6]); unsigned = dict(record); digest = unsigned.pop("event_digest", None)
        if (digest != event[0] or digest_canonical(unsigned) != digest
                or _json(record) != event[6] or tuple(event[1:6]) != tuple(record.get(key)
                    for key in ("route", "state", "reason", "invocation_id", "recorded_at"))
                or record.get("state") != "OPEN" or record.get("reason") != route_state.get("reason")
                or record.get("invocation_id") != route_state.get("invocation_id")):
            return False
        current = connection.execute(
            "SELECT invocation_id,active,unresolved,policy_breach FROM model_usage_current "
            "WHERE route='TYPESAFE_JUDGMENT'"
        ).fetchall()
        blocked = set()
        for invocation_id, active, unresolved, breach in current:
            if active or not unresolved or breach:
                return False
            prior, terminal = _retained_terminal_allocation(connection, invocation_id)
            if (prior.workload_class is not WorkloadClass.TYPESAFE_JUDGMENT
                    or prior.provider != "typesafe" or prior.route != "TYPESAFE_JUDGMENT"
                    or terminal is None or terminal.outcome != "TYPESAFE_FAILED"
                    or terminal.usage_status is not UsageStatus.UNREPORTED
                    or terminal.failure_class not in failures or terminal.policy_breach is not None
                    or terminal.dispatch_at is None or terminal.pre_dispatch_zero_proved):
                return False
            if terminal.failure_class not in {"TimeoutError", "HTTPError:400", "HTTPError:422"} and (
                    allocation.allocated_at < terminal.observed_at + timedelta(minutes=5)):
                return False
            old_policy = _policy_for_allocation(connection, prior)
            if not old_policy.qualified or not _has_exact_dispatch(connection, terminal):
                return False
            ModelUsageService._validate_terminal(terminal, WorkloadClass.TYPESAFE_JUDGMENT, old_policy,
                                                requested_max_output_tokens=prior.max_output_tokens)
            dispatches = connection.execute(
                "SELECT observed_at,evidence_digest FROM model_transport_observations "
                "WHERE invocation_id=? AND state='DISPATCH_STARTED'", (invocation_id,),
            ).fetchall()
            if len(dispatches) != 1 or tuple(dispatches[0]) != (_utc_text(terminal.dispatch_at), prior.request_digest):
                return False
            row = connection.execute(
                "SELECT envelope_id,cycle_id,workload_class,admitted_at,canonical_digest,record_json "
                "FROM model_work_envelopes WHERE envelope_id=?", (prior.envelope_id,),
            ).fetchone()
            if row is None:
                return False
            retained = _envelope_from_record(_object(row[5]))
            if (tuple(row[:5]) != (retained.envelope_id, retained.cycle_id, retained.workload_class.value,
                                  _utc_text(retained.admitted_at), retained.canonical_digest)
                    or _json(retained.as_record()) != row[5] or retained.envelope_id != prior.envelope_id
                    or retained.cycle_id != prior.cycle_id
                    or retained.workload_class is not WorkloadClass.TYPESAFE_JUDGMENT):
                return False
            from .native_assessor import _retained_context
            context = _retained_context(connection, prior)
            headers = connection.execute(
                "SELECT provider,route,evidence_package_digest FROM model_invocation_context_manifests "
                "WHERE context_manifest_digest=?", (prior.context_manifest_digest,),
            ).fetchone()
            if (headers is None or tuple(headers) != (context.get("provider"), context.get("route"),
                    context.get("evidence_package_digest"))
                    or context.get("evidence_package_digest") != retained.evidence_package_digest):
                return False
            role = context.get("caller_identity")
            prefix, separator, number = str(retained.graphiti_attempt_id or "").rpartition(":")
            if (role not in {"NATIVE_ASSESSOR", "GRAPHITI_VERIFIER"}
                    or context.get("caller_ingest_id") != retained.ingest_id
                    or context.get("caller_graphiti_attempt_id") != retained.graphiti_attempt_id
                    or role == "NATIVE_ASSESSOR" and (not retained.candidate_id or retained.graphiti_attempt_id is not None)
                    or role == "GRAPHITI_VERIFIER" and (retained.candidate_id is not None or not retained.ingest_id
                        or prefix != retained.ingest_id or separator != ":" or not number.isdigit() or int(number) <= 0)):

                return False
            scope = retained.as_record()
            if terminal.failure_class != "TimeoutError" and (
                    prior.prompt_digest == allocation.prompt_digest
                    or retained.evidence_package_digest == envelope.get("evidence_package_digest")):
                return False
            if not (scope.get("candidate_id") or scope.get("ingest_id")):
                return False
            if any(scope.get(key) is not None and scope.get(key) == envelope.get(key)
                   for key in ("candidate_id", "ingest_id")):
                return False
            blocked.add(invocation_id)
            if invocation_id == route_state.get("invocation_id") and (
                    record.get("recorded_at") != _utc_text(terminal.observed_at)
                    or record.get("reason") != terminal.failure_class):
                return False
        return bool(blocked) and route_state.get("invocation_id") in blocked

    def _validate_preflight(
        self,
        connection: sqlite3.Connection,
        allocation: InvocationAllocation,
        policy: InvocationEfficiencyPolicy,
        *,
        graphiti_identity: GraphitiInternalRequestIdentity | None = None,
    ) -> None:
        manifest: dict[str, object] = {}
        if policy.command_semantic_version != "UNSPECIFIED":
            manifest_row = connection.execute(
                "SELECT record_json FROM model_invocation_context_manifests "
                "WHERE context_manifest_digest=?",
                (allocation.context_manifest_digest,),
            ).fetchone()
            if manifest_row is None:
                raise ModelUsageAdmissionError("context manifest is absent")
            manifest = _object(manifest_row[0])
        envelope_row = connection.execute(
            "SELECT record_json FROM model_work_envelopes WHERE envelope_id=?",
            (allocation.envelope_id,),
        ).fetchone()
        if envelope_row is None:
            raise ModelUsageAdmissionError("work envelope is absent")
        envelope = _object(envelope_row[0])
        if allocation.workload_class is WorkloadClass.TYPESAFE_JUDGMENT:
            role = manifest.get("caller_identity")
            if (role == "NATIVE_ASSESSOR" and (not envelope.get("candidate_id")
                    or envelope.get("graphiti_attempt_id") is not None)
                    or role == "GRAPHITI_VERIFIER" and (envelope.get("candidate_id") is not None
                        or not envelope.get("graphiti_attempt_id"))
                    or role not in {"NATIVE_ASSESSOR", "GRAPHITI_VERIFIER"}
                    or manifest.get("caller_ingest_id") != envelope.get("ingest_id")
                    or manifest.get("caller_graphiti_attempt_id") != envelope.get("graphiti_attempt_id")):
                raise ModelUsageAdmissionError("Typesafe caller context differs from envelope")
        if (
            allocation.workload_class != policy.workload_class
            or allocation.provider != policy.provider
            or allocation.route != policy.route
            or allocation.model != policy.model
            or allocation.reasoning != policy.reasoning
            or allocation.prompt_contract_version != policy.prompt_contract_version
            or allocation.output_schema_digest != policy.output_schema_digest
            or (allocation.max_output_tokens is None) != (policy.max_output_tokens is None)
            or (allocation.max_output_tokens is not None and policy.max_output_tokens is not None
                and allocation.max_output_tokens > policy.max_output_tokens)
            or allocation.one_turn != policy.one_turn
            or allocation.exact_input != policy.exact_input
            or allocation.skills_enabled != policy.skills_enabled
            or allocation.tools_enabled != policy.tools_enabled
            or allocation.mcp_enabled != policy.mcp_enabled
            or allocation.prior_message_count != policy.prior_message_count
        ):
            raise ModelUsageAdmissionError("allocation differs from invocation policy")
        if policy.command_semantic_version != "UNSPECIFIED" and (
            manifest.get("schema_version") != policy.context_manifest_schema_version
            or _record_string_tuple(manifest, "command_flags")
            != policy.command_flags
            or _record_string_tuple(manifest, "disabled_capabilities")
            != policy.disabled_capabilities
            or manifest.get("implementation_worktree_clean") is not True
        ):
            raise ModelUsageAdmissionError(
                "context manifest command contract differs from invocation policy"
            )
        if (
            policy.command_semantic_version != "UNSPECIFIED"
            and not allocation.workload_class.value.startswith("GRAPHITI_")
            and (
                manifest.get("evidence_package_digest")
                != envelope.get("evidence_package_digest")
            )
        ):
            raise ModelUsageAdmissionError(
                "context manifest Evidence Package differs from work envelope"
            )
        expected_request_digest = digest_canonical(
            {
                "provider": manifest.get("provider"),
                "route": manifest.get("route"),
                "model": manifest.get("model"),
                "reasoning": manifest.get("reasoning"),
                "command_semantic_version": manifest.get(
                    "command_semantic_version"
                ),
                "command_flags": manifest.get("command_flags"),
                "implementation_revision": manifest.get(
                    "implementation_revision"
                ),
                "system_digest": manifest.get("system_digest"),
                "prompt_digest": manifest.get("prompt_digest"),
                "output_schema_digest": manifest.get("output_schema_digest"),
            }
        )
        if policy.command_semantic_version != "UNSPECIFIED" and (
            manifest.get("provider") != allocation.provider
            or manifest.get("route") != allocation.route
            or manifest.get("model") != allocation.model
            or manifest.get("reasoning") != allocation.reasoning
            or manifest.get("prompt_contract_version")
            != allocation.prompt_contract_version
            or manifest.get("prompt_bytes") != allocation.prompt_bytes
            or manifest.get("prompt_digest") != allocation.prompt_digest
            or manifest.get("output_schema_digest")
            != allocation.output_schema_digest
            or manifest.get("context_identity") != allocation.context_identity
            or manifest.get("config_identity") != allocation.config_identity
            or manifest.get("one_turn") != allocation.one_turn
            or manifest.get("exact_input") != allocation.exact_input
            or manifest.get("skills_enabled") != allocation.skills_enabled
            or manifest.get("tools_enabled") != allocation.tools_enabled
            or manifest.get("mcp_enabled") != allocation.mcp_enabled
            or manifest.get("prior_message_count")
            != allocation.prior_message_count
            or manifest.get("request_digest") != expected_request_digest
            or manifest.get("request_digest") != allocation.request_digest
        ):
            raise ModelUsageAdmissionError(
                "context manifest invocation identity differs from allocation"
            )
        if policy.calibration_only and envelope.get("candidate_id") not in (
            policy.allowed_candidate_ids
        ):
            raise ModelUsageAdmissionError(
                "candidate is outside the bounded calibration policy"
            )
        if allocation.prompt_bytes > policy.max_prompt_bytes:
            raise ModelUsageAdmissionError(
                "prompt bytes exceed qualified policy",
                reason_code="EXACT_INPUT_EXCEEDS_QUALIFIED_BOUND",
            )
        if (allocation.workload_class is WorkloadClass.NATIVE_EVIDENCE_ASSESSOR
                and allocation.config_identity == "native-evidence-assessor-grok-hermetic-command-v1"):
            from .native_assessor import native_assessment_input_bound

            bound = native_assessment_input_bound(policy)
            if (manifest.get("input_bound") != bound
                    or manifest.get("system_digest") != bound["system_digest"]
                    or manifest.get("schema_digest") != bound["schema_digest"]
                    or manifest.get("output_schema_digest") != bound["schema_digest"]
                    or allocation.prompt_bytes > bound["max_request_bytes"]):
                raise ModelUsageAdmissionError(
                    "native assessor complete input exceeds qualified bound",
                    reason_code="EXACT_INPUT_EXCEEDS_QUALIFIED_BOUND",
                )
        if allocation.context_identity not in policy.allowed_context_identities:
            raise ModelUsageAdmissionError(
                "context identity is outside qualified policy"
            )
        if allocation.config_identity not in policy.allowed_config_identities:
            raise ModelUsageAdmissionError(
                "config identity is outside qualified policy"
            )
        blocking_routes = _usage_blocking_routes(connection)
        route_reader = self._graphiti_work_route_state if graphiti_identity is not None else self._route_state
        route_state = route_reader(connection, allocation.route, blocking_routes=blocking_routes)
        independent_graphiti_work = graphiti_identity is not None and route_state["state"] == "CLOSED"
        independent_timeout_work = self._independent_typesafe_transport_work(
            connection, allocation, envelope, route_state,
        )
        if _canonical_circuit_route(allocation.route) in blocking_routes and not (independent_timeout_work or independent_graphiti_work):
            raise ModelUsageAdmissionError(
                "affected route has unresolved usage or a policy breach"
            )
        if route_state["state"] == "OPEN" and not independent_timeout_work:
            raise ModelUsageAdmissionError("affected route circuit is open")
        duplicate = connection.execute(
            "SELECT 1 FROM model_invocation_allocations "
            "WHERE envelope_id=? AND request_digest=?",
            (allocation.envelope_id, allocation.request_digest),
        ).fetchone()
        if duplicate is not None:
            raise ModelUsageAdmissionError("duplicate request digest in work envelope")
        if allocation.parent_invocation_id is not None:
            parent = connection.execute(
                "SELECT envelope_id FROM model_invocation_allocations WHERE invocation_id=?",
                (allocation.parent_invocation_id,),
            ).fetchone()
            if parent is None or str(parent[0]) != allocation.envelope_id:
                raise ModelUsageAdmissionError(
                    "parent invocation is outside work envelope"
                )

    def observe_transport(
        self,
        *,
        invocation_id: str,
        observed_at: datetime,
        state: str,
        evidence_digest: str,
    ) -> None:
        record = {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "invocation_id": invocation_id,
            "observed_at": _utc_text(observed_at),
            "state": _token(state, field="transport state"),
            "evidence_digest": _token(
                evidence_digest, field="transport evidence digest"
            ),
        }
        observation_digest = digest_canonical(record)
        connection = self._connection()
        try:
            connection.execute(
                "INSERT OR IGNORE INTO model_transport_observations("
                "observation_digest,invocation_id,observed_at,state,evidence_digest,record_json) "
                "VALUES(?,?,?,?,?,?)",
                (
                    observation_digest,
                    invocation_id,
                    record["observed_at"],
                    state,
                    evidence_digest,
                    _json({**record, "observation_digest": observation_digest}),
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def has_committed_provider_dispatch(
        self, *, cycle_id: str, ingest_id: str | None = None,
        graphiti_attempt_id: str | None = None,
    ) -> bool:
        """Return event dispatch truth from a committed provider-leaf marker."""

        cycle_id = _token(cycle_id, field="cycle id")
        connection = self._connection()
        try:
            row = connection.execute(
                "SELECT EXISTS("
                "SELECT 1 FROM model_invocation_allocations AS allocation "
                "JOIN model_transport_observations AS observation "
                "ON observation.invocation_id=allocation.invocation_id "
                "WHERE allocation.cycle_id=? "
                "AND allocation.workload_class IN (?,?,?) "
                "AND observation.state='DISPATCH_STARTED')",
                (
                    cycle_id,
                    WorkloadClass.GRAPHITI_CHAT_PRIMARY.value,
                    WorkloadClass.GRAPHITI_CHAT_FALLBACK.value,
                    WorkloadClass.GRAPHITI_EMBEDDING.value,
                ),
            ).fetchone()
            if row and row[0]:
                return True
            if ingest_id is None or graphiti_attempt_id is None:
                return False
            prefix, separator, number = graphiti_attempt_id.rpartition(":")
            if prefix != ingest_id or not separator or not number.isdigit() or int(number) <= 0:
                raise ModelUsageIntegrityError("Typesafe graphiti trace identity differs")
            row = connection.execute(
                "SELECT EXISTS(SELECT 1 FROM model_invocation_allocations a "
                "JOIN model_transport_observations o USING(invocation_id) "
                "JOIN model_work_envelopes e USING(envelope_id) "
                "JOIN model_invocation_context_manifests m "
                "ON m.context_manifest_digest=json_extract(a.record_json,'$.context_manifest_digest') "
                "WHERE a.cycle_id=? AND a.workload_class=? AND o.state='DISPATCH_STARTED' "
                "AND json_extract(e.record_json,'$.ingest_id')=? "
                "AND json_extract(e.record_json,'$.graphiti_attempt_id')=? "
                "AND json_extract(m.record_json,'$.caller_identity')='GRAPHITI_VERIFIER' "
                "AND json_extract(m.record_json,'$.caller_ingest_id')=? "
                "AND json_extract(m.record_json,'$.caller_graphiti_attempt_id')=?)",
                (cycle_id, WorkloadClass.TYPESAFE_JUDGMENT.value, ingest_id,
                 graphiti_attempt_id, ingest_id, graphiti_attempt_id),
            ).fetchone()
            return bool(row and row[0])
        finally:
            connection.close()

    def link_provider_attempt(
        self,
        *,
        invocation_id: str,
        provider_attempt_id: str,
        linked_at: datetime,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        record = {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "invocation_id": _token(invocation_id, field="invocation id"),
            "provider_attempt_id": _token(
                provider_attempt_id, field="provider attempt id"
            ),
            "linked_at": _utc_text(linked_at),
        }
        digest = digest_canonical(record)
        self._insert_exact(
            table="model_invocation_provider_attempt_links",
            identity_column="invocation_id",
            identity=invocation_id,
            record={**record, "link_digest": digest},
            sql="INSERT INTO model_invocation_provider_attempt_links("
            "link_digest,invocation_id,provider_attempt_id,linked_at,record_json) "
            "VALUES(?,?,?,?,?)",
            values=(
                digest,
                invocation_id,
                provider_attempt_id,
                record["linked_at"],
                _json({**record, "link_digest": digest}),
            ),
            connection=connection,
        )

    def complete(
        self,
        terminal: InvocationTerminal,
        *,
        provider_telemetry: Mapping[str, object] | None = None,
    ) -> InvocationTerminal:
        terminal._validate_shape()
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT a.route,a.workload_class,p.record_json,a.record_json "
                "FROM model_invocation_allocations a "
                "JOIN model_invocation_policies p ON p.canonical_digest=a.policy_digest "
                "WHERE a.invocation_id=?",
                (terminal.invocation_id,),
            ).fetchone()
            if row is None:
                raise ModelUsageIntegrityError("invocation allocation is absent")
            if terminal.pre_dispatch_zero_proved:
                if (
                    terminal.provider_telemetry_digest is not None
                    or terminal.raw_telemetry_pointer is not None
                    or provider_telemetry is not None
                ):
                    raise ModelUsageIntegrityError("pre-dispatch zero contradicts provider telemetry")
                if _has_exact_dispatch(connection, terminal):
                    raise ModelUsageIntegrityError("pre-dispatch zero contradicts committed dispatch")
            route, workload = str(row[0]), WorkloadClass(str(row[1]))
            allocation = _allocation_from_record(_object(row[3]))
            context_manifest_digest = allocation.context_manifest_digest
            requested_max_output_tokens = allocation.max_output_tokens
            policy = _policy_from_record(_object(row[2]))
            retained = terminal
            if provider_telemetry is not None:
                telemetry_digest = _retain_provider_telemetry(
                    connection,
                    invocation_id=terminal.invocation_id,
                    provider_telemetry=provider_telemetry,
                )
                if retained.provider_telemetry_digest != telemetry_digest:
                    retained = replace(
                        retained,
                        usage_status=UsageStatus.INVALID,
                        failure_class="TELEMETRY_DIGEST_MISMATCH",
                        terminal_digest="",
                    )
            invalid_report = _invalid_reported_components(retained,
                native_sdk=native_sdk_reported_token_targets_are_advisory(policy))
            if invalid_report is not None:
                retained = replace(
                    retained,
                    usage_status=UsageStatus.INVALID,
                    failure_class=invalid_report,
                    terminal_digest="",
                )
            policy_breach = self._validate_terminal(
                retained,
                workload,
                policy,
                requested_max_output_tokens=requested_max_output_tokens,
            )
            if policy_breach is not None:
                retained = replace(
                    retained,
                    policy_breach=policy_breach,
                    terminal_digest="",
                )
            record_without_digest = retained.as_record()
            record_without_digest["terminal_digest"] = ""
            retained = replace(
                retained,
                terminal_digest=digest_canonical(record_without_digest),
            )
            record = retained.as_record()
            try:
                connection.execute(
                    "INSERT INTO model_invocation_terminals("
                    "terminal_digest,invocation_id,usage_status,outcome,failure_class,"
                    "completed_at,record_json) VALUES(?,?,?,?,?,?,?)",
                    (
                        retained.terminal_digest,
                        retained.invocation_id,
                        retained.usage_status.value,
                        retained.outcome,
                        retained.failure_class,
                        _utc_text(retained.completed_at),
                        _json(record),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                prior = connection.execute(
                    "SELECT record_json FROM model_invocation_terminals WHERE invocation_id=?",
                    (retained.invocation_id,),
                ).fetchone()
                if prior is None or _object(prior[0]) != record:
                    raise ModelUsageIntegrityError(
                        "conflicting invocation terminal replay"
                    ) from exc
            context_manifest = connection.execute(
                "SELECT 1 FROM model_invocation_context_manifests "
                "WHERE context_manifest_digest=?",
                (context_manifest_digest,),
            ).fetchone()
            if context_manifest is not None:
                context_observation = {
                    "schema_version": MODEL_USAGE_SCHEMA_VERSION,
                    "invocation_id": retained.invocation_id,
                    "context_manifest_digest": context_manifest_digest,
                    "usage_status": retained.usage_status.value,
                    "provider_context_tokens": retained.components.context_tokens,
                    "observed_at": _utc_text(retained.observed_at),
                }
                observation_digest = digest_canonical(context_observation)
                observation_record = {
                    **context_observation,
                    "observation_digest": observation_digest,
                }
                connection.execute(
                    "INSERT OR IGNORE INTO model_invocation_context_observations("
                    "observation_digest,invocation_id,context_manifest_digest,"
                    "provider_context_tokens,record_json) VALUES(?,?,?,?,?)",
                    (
                        observation_digest,
                        retained.invocation_id,
                        context_manifest_digest,
                        retained.components.context_tokens,
                        _json(observation_record),
                    ),
                )
            _refresh_current_usage(connection, retained.invocation_id)
            current_blocker = connection.execute(
                "SELECT 1 FROM model_usage_current WHERE invocation_id=? "
                "AND (unresolved=1 OR policy_breach=1)", (retained.invocation_id,),
            ).fetchone() is not None
            if current_blocker and retained.usage_status in {
                UsageStatus.UNREPORTED,
                UsageStatus.AMBIGUOUS,
                UsageStatus.INVALID,
            }:
                self._append_route_state(
                    connection,
                    route=route,
                    state="OPEN",
                    reason=(retained.failure_class or retained.usage_status.value),
                    invocation_id=retained.invocation_id,
                    recorded_at=retained.observed_at,
                )
            elif current_blocker and retained.policy_breach:
                self._append_route_state(
                    connection,
                    route=route,
                    state="OPEN",
                    reason=retained.policy_breach,
                    invocation_id=retained.invocation_id,
                    recorded_at=retained.observed_at,
                )
            connection.commit()
            return retained
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def terminal(self, invocation_id: str) -> InvocationTerminal | None:
        """Read one exact retained terminal without weakening its validation."""
        invocation_id = _token(invocation_id, field="invocation id")
        connection = self._connection()
        try:
            row = connection.execute(
                "SELECT terminal_digest,record_json FROM model_invocation_terminals "
                "WHERE invocation_id=?", (invocation_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        try:
            record = _object(row[1])
            values = dict(record)
            schema = values.pop("schema_version")
            retained_digest = values.pop("terminal_digest")
            values["usage_status"] = UsageStatus(values["usage_status"])
            values["components"] = UsageComponents(**values["components"])
            values["dispatch_at"] = (
                None if values["dispatch_at"] is None
                else _instant(values["dispatch_at"])
            )
            values["completed_at"] = _instant(values["completed_at"])
            values["observed_at"] = _instant(values["observed_at"])
            terminal = InvocationTerminal.create(**values)
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelUsageIntegrityError("invocation terminal record differs") from exc
        if (
            schema != MODEL_USAGE_SCHEMA_VERSION
            or terminal.invocation_id != invocation_id
            or retained_digest != row[0]
            or terminal.terminal_digest != row[0]
            or terminal.as_record() != record
        ):
            raise ModelUsageIntegrityError("invocation terminal record differs")
        return terminal

    @staticmethod
    def _validate_terminal(
        terminal: InvocationTerminal,
        workload: WorkloadClass,
        policy: InvocationEfficiencyPolicy,
        *,
        requested_max_output_tokens: int | None,
    ) -> str | None:
        policy._validate()
        if (
            (requested_max_output_tokens is None) != (policy.max_output_tokens is None)
            or (requested_max_output_tokens is not None and (
                not _is_int(requested_max_output_tokens) or requested_max_output_tokens <= 0
            ))
        ):
            raise ModelUsageIntegrityError("terminal output guard differs from policy")
        components = terminal.components
        total = components.total_tokens
        if terminal.pre_dispatch_zero_proved:
            if terminal.dispatch_at is not None or total != 0:
                raise ModelUsageIntegrityError(
                    "pre-dispatch zero requires no dispatch and exact zero"
                )
        elif terminal.dispatch_at is None:
            raise ModelUsageIntegrityError(
                "possible provider usage lacks a dispatch observation"
            )
        if terminal.usage_status is UsageStatus.REPORTED:
            if _invalid_reported_components(terminal,
                    native_sdk=native_sdk_reported_token_targets_are_advisory(policy)) is not None:
                raise ModelUsageIntegrityError(
                    "invalid reported usage was not classified"
                )
        elif terminal.usage_status is UsageStatus.ESTIMATED:
            hard_ceiling = policy.hard_estimate_ceiling_tokens
            if (
                hard_ceiling is None
                or total != hard_ceiling
                or components.provenance != "BOUNDED_ESTIMATE"
                or terminal.estimate_policy_digest != policy.canonical_digest
                or not terminal.estimate_calculation
            ):
                raise ModelUsageIntegrityError(
                    "bounded estimate evidence is incomplete"
                )
        elif (
            terminal.usage_status in {UsageStatus.UNREPORTED, UsageStatus.AMBIGUOUS}
            and total is not None
        ):
            raise ModelUsageIntegrityError("unresolved usage must not invent a total")
        if workload is WorkloadClass.TYPESAFE_JUDGMENT:
            if (policy.provider != "typesafe" or policy.route != "TYPESAFE_JUDGMENT"
                    or terminal.od_011_reference != "OD-011:TYPESAFE_JUDGMENT"
                    or terminal.subscription_cli_chat_not_cash_debited):
                raise ModelUsageIntegrityError("Typesafe paid usage linkage differs")
        elif workload in {
            WorkloadClass.GRAPHITI_EMBEDDING,
            WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING,
        }:
            if not terminal.od_011_reference:
                raise ModelUsageIntegrityError("embedding usage lacks OD-011 linkage")
        elif not terminal.subscription_cli_chat_not_cash_debited:
            raise ModelUsageIntegrityError(
                "subscription CLI chat cash-debit confirmation is absent"
            )
        if terminal.usage_status is not UsageStatus.REPORTED:
            return terminal.policy_breach
        advisory = native_sdk_reported_token_targets_are_advisory(policy)
        if not advisory and total is not None and total > policy.max_total_tokens:
            return "MAX_TOTAL_TOKENS_EXCEEDED"
        context = components.context_tokens
        if context is not None and context > policy.max_context_tokens:
            return "MAX_CONTEXT_TOKENS_EXCEEDED"
        output = components.output_tokens
        if not advisory and requested_max_output_tokens is not None and output is not None and output > requested_max_output_tokens:
            return "REQUESTED_MAX_OUTPUT_TOKENS_EXCEEDED"
        if not advisory and policy.max_output_tokens is not None and output is not None and output > policy.max_output_tokens:
            return "MAX_OUTPUT_TOKENS_EXCEEDED"
        return terminal.policy_breach

    def record_work_outcome(
        self,
        *,
        envelope_id: str,
        outcome: str,
        outcome_record_id: str,
        payload_digest: str | None,
        terminal_at: datetime,
        cycle_outcome: str | None = None,
        route_circuit_state: str | None = None,
        route_circuit_reason: str | None = None,
        retained_proposal_count: int | None = None,
        accepted_provider_attempt_id: str | None = None,
        stable_reason_codes: tuple[str, ...] = (),
        connection: sqlite3.Connection | None = None,
    ) -> None:
        record = {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "envelope_id": envelope_id,
            "outcome": _token(outcome, field="work outcome"),
            "outcome_record_id": _token(outcome_record_id, field="outcome record id"),
            "payload_digest": payload_digest,
            "terminal_at": _utc_text(terminal_at),
            "cycle_outcome": cycle_outcome,
            "route_circuit_state": route_circuit_state,
            "route_circuit_reason": route_circuit_reason,
            "retained_proposal_count": retained_proposal_count,
            "accepted_provider_attempt_id": accepted_provider_attempt_id,
            "stable_reason_codes": list(stable_reason_codes),
        }
        if retained_proposal_count is not None:
            _non_negative(retained_proposal_count, field="retained proposal count")
        digest = digest_canonical(record)
        owns_connection = connection is None
        current = self._connection() if connection is None else connection
        try:
            if owns_connection:
                current.execute("BEGIN IMMEDIATE")
            envelope_row = current.execute(
                "SELECT record_json FROM model_work_envelopes WHERE envelope_id=?",
                (envelope_id,),
            ).fetchone()
            graphiti_attempt_id = (
                None
                if envelope_row is None
                else _object(envelope_row[0]).get("graphiti_attempt_id")
            )
            unresolved_graphiti_leaf = current.execute(
                "SELECT 1 FROM graphiti_internal_requests g "
                "LEFT JOIN model_invocation_terminals t "
                "ON t.invocation_id=g.invocation_id "
                "WHERE "
                + (
                    "g.graphiti_attempt_id=? "
                    if isinstance(graphiti_attempt_id, str)
                    else "g.envelope_id=? "
                )
                + "AND t.invocation_id IS NULL LIMIT 1",
                (
                    graphiti_attempt_id
                    if isinstance(graphiti_attempt_id, str)
                    else envelope_id,
                ),
            ).fetchone()
            if unresolved_graphiti_leaf is not None:
                raise ModelUsageIntegrityError(
                    "Graphiti work outcome lacks a terminal receipt for every leaf"
                )
            self._insert_exact(
                table="model_work_outcomes",
                identity_column="envelope_id",
                identity=envelope_id,
                record={**record, "outcome_digest": digest},
                sql="INSERT INTO model_work_outcomes("
                "outcome_digest,envelope_id,outcome,terminal_at,record_json) "
                "VALUES(?,?,?,?,?)",
                values=(
                    digest,
                    envelope_id,
                    outcome,
                    record["terminal_at"],
                    _json({**record, "outcome_digest": digest}),
                ),
                connection=current,
            )
            if owns_connection:
                current.commit()
        except Exception:
            if owns_connection and current.in_transaction:
                current.rollback()
            raise
        finally:
            if owns_connection:
                current.close()

    def record_cycle_outcome(
        self,
        *,
        cycle_id: str,
        outcome_class: str,
        terminal_at: datetime,
        writer_unproductive_streak_before: int,
        writer_unproductive_streak_after: int,
        writer_circuit_state: str,
        writer_circuit_open_reason: str,
    ) -> None:
        record = {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "cycle_id": _token(cycle_id, field="cycle id"),
            "outcome_class": _token(outcome_class, field="cycle outcome class"),
            "terminal_at": _utc_text(terminal_at),
            "writer_unproductive_streak_before": _non_negative(
                writer_unproductive_streak_before,
                field="writer unproductive streak before",
            ),
            "writer_unproductive_streak_after": _non_negative(
                writer_unproductive_streak_after,
                field="writer unproductive streak after",
            ),
            "writer_circuit_state": _token(
                writer_circuit_state, field="writer circuit state"
            ),
            "writer_circuit_open_reason": writer_circuit_open_reason,
        }
        digest = digest_canonical(record)
        self._insert_exact(
            table="model_usage_cycle_outcomes",
            identity_column="cycle_id",
            identity=cycle_id,
            record={**record, "cycle_digest": digest},
            sql="INSERT INTO model_usage_cycle_outcomes("
            "cycle_digest,cycle_id,outcome_class,terminal_at,record_json) "
            "VALUES(?,?,?,?,?)",
            values=(
                digest,
                cycle_id,
                outcome_class,
                record["terminal_at"],
                _json({**record, "cycle_digest": digest}),
            ),
        )

    def recover_unresolved(self, *, observed_at: datetime) -> int:
        connection = self._connection()
        recovered = 0
        try:
            rows = connection.execute(
                "SELECT a.invocation_id,a.allocated_at,a.workload_class,"
                "(SELECT MIN(o.observed_at) FROM model_transport_observations o "
                "WHERE o.invocation_id=a.invocation_id "
                "AND o.state='DISPATCH_STARTED'),"
                "json_extract(a.record_json,'$.recovery_deadline_at'),"
                "EXISTS(SELECT 1 FROM graphiti_internal_requests g "
                "WHERE g.invocation_id=a.invocation_id) "
                "FROM model_invocation_allocations a "
                "LEFT JOIN model_invocation_terminals t ON t.invocation_id=a.invocation_id "
                "WHERE t.invocation_id IS NULL "
                "AND json_extract(a.record_json,'$.recovery_deadline_at') IS NOT NULL "
                "AND json_extract(a.record_json,'$.recovery_deadline_at')<=? "
                "ORDER BY a.invocation_id",
                (_utc_text(observed_at),),
            ).fetchall()
        finally:
            connection.close()
        for row in rows:
            workload = WorkloadClass(str(row[2]))
            embedding = workload in {
                WorkloadClass.GRAPHITI_EMBEDDING,
                WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING,
            }
            graphiti_pre_dispatch = bool(row[5]) and row[3] is None
            dispatch_at = (
                None
                if graphiti_pre_dispatch
                else _instant(str(row[3] or row[1]))
            )
            recovery_terminal_at = _instant(str(row[4]))
            terminal = InvocationTerminal.create(
                invocation_id=str(row[0]),
                outcome=(
                    "RECOVERED_PRE_DISPATCH"
                    if graphiti_pre_dispatch
                    else "RECOVERED_UNRESOLVED"
                ),
                failure_class=(
                    "PROCESS_LOST_BEFORE_DISPATCH"
                    if graphiti_pre_dispatch
                    else "PROCESS_LOST_AFTER_ALLOCATION"
                ),
                usage_status=(
                    UsageStatus.REPORTED
                    if graphiti_pre_dispatch
                    else UsageStatus.AMBIGUOUS
                ),
                components=(
                    UsageComponents(total_tokens=0, provenance="CLI_DERIVED")
                    if graphiti_pre_dispatch
                    else UsageComponents(provenance="UNAVAILABLE")
                ),
                dispatch_at=dispatch_at,
                completed_at=recovery_terminal_at,
                observed_at=recovery_terminal_at,
                od_011_reference=(
                    "OD-011:EVALUATION_GRAPHITI_EMBEDDING" if embedding else None
                ),
                subscription_cli_chat_not_cash_debited=not embedding,
                pre_dispatch_zero_proved=graphiti_pre_dispatch,
            )
            try:
                self.complete(terminal)
            except ModelUsageIntegrityError as exc:
                if not self._terminal_exists(str(row[0])):
                    raise
                if "conflicting invocation terminal replay" not in str(exc):
                    raise
                continue
            recovered += 1
        connection = self._connection()
        try:
            envelope_rows = connection.execute(
                "SELECT e.envelope_id,MAX(json_extract("
                "a_deadline.record_json,'$.recovery_deadline_at')) "
                "FROM model_work_envelopes e "
                "JOIN model_invocation_allocations a_deadline "
                "ON a_deadline.envelope_id=e.envelope_id "
                "WHERE NOT EXISTS (SELECT 1 FROM model_work_outcomes w "
                "WHERE w.envelope_id=e.envelope_id) "
                "AND EXISTS (SELECT 1 FROM model_invocation_allocations a "
                "WHERE a.envelope_id=e.envelope_id) "
                "AND NOT EXISTS (SELECT 1 FROM model_invocation_allocations a "
                "LEFT JOIN model_invocation_terminals t "
                "ON t.invocation_id=a.invocation_id "
                "WHERE a.envelope_id=e.envelope_id AND t.invocation_id IS NULL) "
                "AND NOT EXISTS (SELECT 1 FROM model_invocation_allocations a "
                "WHERE a.envelope_id=e.envelope_id AND ("
                "json_extract(a.record_json,'$.recovery_deadline_at') IS NULL OR "
                "json_extract(a.record_json,'$.recovery_deadline_at')>?)) "
                "GROUP BY e.envelope_id ORDER BY e.envelope_id",
                (_utc_text(observed_at),),
            ).fetchall()
        finally:
            connection.close()
        for envelope_id, raw_terminal_at in envelope_rows:
            recovery_terminal_at = _instant(str(raw_terminal_at))
            try:
                self.record_work_outcome(
                    envelope_id=str(envelope_id),
                    outcome="AMBIGUOUS_PROCESS_LOST_BEFORE_WORK_OUTCOME",
                    outcome_record_id=digest_canonical(
                        {
                            "envelope_id": str(envelope_id),
                            "recovered_at": _utc_text(recovery_terminal_at),
                        }
                    ),
                    payload_digest=None,
                    terminal_at=recovery_terminal_at,
                    stable_reason_codes=("PROCESS_LOST_BEFORE_WORK_OUTCOME",),
                )
            except ModelUsageIntegrityError as exc:
                if not self._work_outcome_exists(str(envelope_id)):
                    raise
                if "conflicting model_work_outcomes replay" not in str(exc):
                    raise
                continue
            recovered += 1
        return recovered

    def _terminal_exists(self, invocation_id: str) -> bool:
        connection = self._connection()
        try:
            return (
                connection.execute(
                    "SELECT 1 FROM model_invocation_terminals WHERE invocation_id=?",
                    (invocation_id,),
                ).fetchone()
                is not None
            )
        finally:
            connection.close()

    def _work_outcome_exists(self, envelope_id: str) -> bool:
        connection = self._connection()
        try:
            return (
                connection.execute(
                    "SELECT 1 FROM model_work_outcomes WHERE envelope_id=?",
                    (envelope_id,),
                ).fetchone()
                is not None
            )
        finally:
            connection.close()

    def disposition_native_unreported_subscription_usage(
        self, *, invocation_id: str, expected_terminal_digest: str,
        expected_allocation_digest: str, observed_at: datetime,
    ) -> dict[str, object]:
        """Retain the native failed-call qualified-policy upper-bound estimate."""
        return self._disposition_native_subscription_usage(
            invocation_id=invocation_id, expected_terminal_digest=expected_terminal_digest,
            expected_allocation_digest=expected_allocation_digest, observed_at=observed_at,
        )

    def disposition_native_graphiti_fallback_cancellation(
        self, *, invocation_id: str, expected_terminal_digest: str,
        expected_allocation_digest: str, observed_at: datetime,
    ) -> dict[str, object]:
        """Settle a proved cancelled fallback; route release remains separately authorised."""
        return self._disposition_native_subscription_usage(
            invocation_id=invocation_id, expected_terminal_digest=expected_terminal_digest,
            expected_allocation_digest=expected_allocation_digest, observed_at=observed_at,
            cancelled_fallback=True,
        )

    def _disposition_native_subscription_usage(
        self,
        *,
        invocation_id: str,
        expected_terminal_digest: str,
        expected_allocation_digest: str,
        observed_at: datetime,
        cancelled_fallback: bool = False,
    ) -> dict[str, object]:
        """Retain the native pipeline's qualified-policy upper-bound estimate."""

        invocation_id = _token(invocation_id, field="invocation id")
        expected_terminal_digest = _token(
            expected_terminal_digest, field="expected terminal digest"
        )
        expected_allocation_digest = _token(
            expected_allocation_digest, field="expected allocation digest"
        )
        observed_at_text = _utc_text(observed_at)
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            retained = connection.execute(
                "SELECT t.terminal_digest,t.record_json,a.canonical_digest,"
                "a.record_json FROM model_invocation_terminals t "
                "JOIN model_invocation_allocations a "
                "ON a.invocation_id=t.invocation_id WHERE t.invocation_id=?",
                (invocation_id,),
            ).fetchone()
            if retained is None:
                raise ModelUsageIntegrityError(
                    "native conservative disposition terminal is absent"
                )
            terminal = _terminal_from_record(_object(retained[1]))
            allocation = _allocation_from_record(_object(retained[3]))
            if (
                retained[0] != terminal.terminal_digest
                or retained[2] != allocation.canonical_digest
                or terminal.invocation_id != invocation_id
                or allocation.invocation_id != invocation_id
            ):
                raise ModelUsageIntegrityError(
                    "native conservative disposition binding differs"
                )
            if terminal.terminal_digest != expected_terminal_digest:
                raise ModelUsageIntegrityError(
                    "native conservative disposition terminal differs"
                )
            if allocation.canonical_digest != expected_allocation_digest:
                raise ModelUsageIntegrityError(
                    "native conservative disposition allocation differs"
                )

            prior = _valid_native_disposition(
                connection, allocation=allocation, terminal=terminal
            )
            scope = (NATIVE_GRAPHITI_FALLBACK_CANCELLATION_USAGE_SCOPE
                     if cancelled_fallback else NATIVE_AUTONOMOUS_USAGE_SCOPE)
            if prior is not None:
                if prior.get("authority_scope") != scope:
                    raise ModelUsageIntegrityError("native subscription disposition scope differs")
                connection.rollback()
                return prior

            policy = _policy_for_allocation(connection, allocation)
            if _native_conservative_subscription_leaf(allocation) is GraphitiLeafClass.FALLBACK:
                # These estimates are qualified for direct route failover, not
                # a child of a malformed primary. Leave unsupported work UNKNOWN.
                _retained_terminal_allocation(connection, invocation_id)
                identity = _retained_graphiti_request_identity(connection, allocation)
                if identity is not None and identity.parent_invocation_id is not None:
                    raise ModelUsageAdmissionError(
                        "parented fallback is outside direct-failover estimate scope",
                        reason_code="NATIVE_DISPOSITION_NOT_APPLICABLE",
                    )
            if cancelled_fallback:
                authority = _native_graphiti_fallback_cancellation_authority(
                    connection, allocation=allocation, terminal=terminal, policy=policy,
                )
            else:
                envelope = _native_envelope(connection, allocation)
                leaf_class = _native_conservative_subscription_leaf(allocation)
                if (
                    leaf_class is None
                    or not policy.qualified
                    or terminal.usage_status is not UsageStatus.UNREPORTED
                    or terminal.outcome not in {"FAILED", "TIMEOUT"}
                    or terminal.failure_class != "MISSING_PROVIDER_TELEMETRY"
                    or terminal.subscription_cli_chat_not_cash_debited is not True
                    or terminal.policy_breach is not None
                    or terminal.provider_telemetry_digest is not None
                    or terminal.raw_telemetry_pointer is not None
                    or terminal.pre_dispatch_zero_proved
                ):
                    raise ModelUsageIntegrityError(
                        "native conservative disposition target is ineligible"
                    )
                if leaf_class is GraphitiLeafClass.FALLBACK:
                    identity = _retained_graphiti_request_identity(
                        connection, allocation
                    )
                    if (
                        identity is None
                        or identity.leaf_class is not GraphitiLeafClass.FALLBACK
                        or identity.primary_unavailable_event_digest is None
                    ):
                        raise ModelUsageIntegrityError(
                            "native fallback request authority differs"
                        )
                    _require_native_failed_attempt_receipt(
                        connection,
                        allocation=allocation,
                        terminal=terminal,
                        envelope=envelope,
                    )
            if observed_at < terminal.observed_at:
                raise ModelUsageIntegrityError(
                    "native conservative disposition precedes terminal"
                )
            if not _has_exact_dispatch(connection, terminal):
                raise ModelUsageIntegrityError(
                    "native conservative disposition lacks committed dispatch"
                )
            if connection.execute(
                "SELECT 1 FROM model_provider_telemetry WHERE invocation_id=? "
                "UNION ALL SELECT 1 FROM model_usage_reconciliations "
                "WHERE invocation_id=? LIMIT 1",
                (invocation_id, invocation_id),
            ).fetchone() is not None:
                raise ModelUsageIntegrityError(
                    "native conservative disposition exact telemetry already exists"
                )

            if not cancelled_fallback:
                authority = _native_disposition_authority(
                    allocation=allocation,
                    terminal=terminal,
                    policy=policy,
                    envelope=envelope,
                )
            scope_digest = digest_canonical(authority)
            record_without_digest: dict[str, object] = {
                **(authority if cancelled_fallback else {}),
                "schema_version": CONSERVATIVE_DISPOSITION_SCHEMA_VERSION,
                "authority_scope": scope,
                "native_scope_digest": scope_digest,
                "invocation_id": invocation_id,
                "terminal_digest": terminal.terminal_digest,
                "allocation_digest": allocation.canonical_digest,
                "policy_digest": policy.canonical_digest,
                "usage_status": UsageStatus.ESTIMATED.value,
                "components": UsageComponents(
                    total_tokens=policy.max_total_tokens,
                    provenance="BOUNDED_ESTIMATE",
                ).as_record(),
                "estimate_policy_digest": policy.canonical_digest,
                "estimate_calculation": (
                    "QUALIFIED_POLICY_MAX_TOTAL_TOKENS_CONSERVATIVE_UPPER_BOUND"
                ),
                "exact_usage_remains_unknown": True,
                "provider_dispatch_preserved": True,
                "unknown_spend_released": False,
                "authority_digest": scope_digest,
                "observed_at": observed_at_text,
            }
            disposition_digest = digest_canonical(record_without_digest)
            record = {
                **record_without_digest,
                "disposition_digest": disposition_digest,
            }
            connection.execute(
                "INSERT INTO model_usage_conservative_dispositions("
                "disposition_digest,invocation_id,terminal_digest,"
                "allocation_digest,policy_digest,approved_plan_digest,"
                "authority_digest,approved_by,approval_reference,approved_at,"
                "observed_at,usage_status,record_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    disposition_digest,
                    invocation_id,
                    terminal.terminal_digest,
                    allocation.canonical_digest,
                    policy.canonical_digest,
                    scope_digest,
                    scope_digest,
                    scope,
                    scope,
                    observed_at_text,
                    observed_at_text,
                    UsageStatus.ESTIMATED.value,
                    _json(record),
                ),
            )
            _refresh_current_usage(connection, invocation_id)
            connection.commit()
            return record
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def disposition_native_graphiti_embedding_cancellation(
        self, *, invocation_id: str, observed_at: datetime,
    ) -> dict[str, object]:
        """Apply standing native authority without releasing unknown cash spend."""
        invocation_id = _token(invocation_id, field="invocation id")
        observed_at_text = _utc_text(observed_at)
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            allocation, terminal = _retained_terminal_allocation(connection, invocation_id)
            policy = _policy_for_allocation(connection, allocation)
            authority = _native_graphiti_embedding_cancellation_authority(
                connection, allocation=allocation, terminal=terminal, policy=policy,
            )
            if observed_at < terminal.observed_at:
                raise ModelUsageIntegrityError("native cancellation disposition precedes terminal")
            record = _valid_native_disposition(
                connection, allocation=allocation, terminal=terminal,
            )
            if record is None:
                # Refuse a new estimate when exact evidence already exists.
                # Later reconciliation may supersede a retained estimate without
                # invalidating the original cancellation or its historical bound.
                if connection.execute(
                    "SELECT 1 FROM model_provider_telemetry WHERE invocation_id=? "
                    "UNION ALL SELECT 1 FROM model_usage_reconciliations "
                    "WHERE invocation_id=? LIMIT 1",
                    (invocation_id, invocation_id),
                ).fetchone() is not None:
                    raise ModelUsageIntegrityError("native Graphiti cancellation already has telemetry")
                scope_digest = digest_canonical(authority)
                unsigned = {
                    **authority,
                    "schema_version": CONSERVATIVE_DISPOSITION_SCHEMA_VERSION,
                    "native_scope_digest": scope_digest,
                    "authority_digest": scope_digest,
                    "usage_status": UsageStatus.ESTIMATED.value,
                    "components": UsageComponents(
                        total_tokens=max(policy.max_total_tokens, allocation.prompt_bytes),
                        provenance="BOUNDED_ESTIMATE",
                    ).as_record(),
                    "estimate_policy_digest": policy.canonical_digest,
                    "estimate_calculation": "MAX_QUALIFIED_POLICY_TOTAL_OR_EXACT_REQUEST_UTF8_BYTES",
                    "exact_usage_remains_unknown": True,
                    "provider_dispatch_preserved": True,
                    "unknown_spend_released": False,
                    "observed_at": observed_at_text,
                }
                record = {**unsigned, "disposition_digest": digest_canonical(unsigned)}
                connection.execute(
                    "INSERT INTO model_usage_conservative_dispositions("
                    "disposition_digest,invocation_id,terminal_digest,allocation_digest,"
                    "policy_digest,approved_plan_digest,authority_digest,approved_by,"
                    "approval_reference,approved_at,observed_at,usage_status,record_json) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (record["disposition_digest"], invocation_id, terminal.terminal_digest,
                     allocation.canonical_digest, policy.canonical_digest, scope_digest,
                     scope_digest, NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE,
                     NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE,
                     observed_at_text, observed_at_text, UsageStatus.ESTIMATED.value,
                     _json(record)),
                )
            elif record.get("authority_scope") != NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE:
                raise ModelUsageIntegrityError("native cancellation disposition scope differs")

            _refresh_current_usage(connection, invocation_id)

            # complete() retained this exact missing-telemetry cause. A valid
            # estimate may close it, but never another cause or live/unknown leaf.
            latest = connection.execute(
                "SELECT state,reason,invocation_id FROM model_usage_route_circuit_events "
                "WHERE route=? ORDER BY recorded_at DESC,rowid DESC LIMIT 1",
                (allocation.route,),
            ).fetchone()
            if (
                latest is not None
                and tuple(latest) == ("OPEN", "MISSING_PROVIDER_TELEMETRY", invocation_id)
                and not _current_has_active(connection, allocation.route)
                and _canonical_circuit_route(allocation.route) not in _usage_blocking_routes(connection)
            ):
                self._append_route_state(
                    connection, route=allocation.route, state="CLOSED",
                    reason="NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_DISPOSITION:"
                    + str(record["disposition_digest"]),
                    invocation_id=invocation_id, recorded_at=observed_at,
                )
            connection.commit()
            return record
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def disposition_native_embedding_timeout(
        self,
        *,
        invocation_id: str,
        expected_terminal_digest: str,
        expected_allocation_digest: str,
        expected_request_digest: str,
        expected_passage_id: str,
        expected_cycle_id: str,
        observed_at: datetime,
    ) -> dict[str, object]:
        """Retain one exact post-dispatch embedding timeout upper bound."""

        invocation_id = _token(invocation_id, field="invocation id")
        expected_terminal_digest = _token(
            expected_terminal_digest, field="expected terminal digest"
        )
        expected_allocation_digest = _token(
            expected_allocation_digest, field="expected allocation digest"
        )
        expected_request_digest = _token(
            expected_request_digest, field="expected request digest"
        )
        expected_passage_id = _token(
            expected_passage_id, field="expected passage id"
        )
        expected_cycle_id = _token(
            expected_cycle_id, field="expected cycle id"
        )
        observed_at_text = _utc_text(observed_at)
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            allocation, terminal = _retained_terminal_allocation(
                connection, invocation_id
            )
            if (
                terminal.invocation_id != invocation_id
                or allocation.invocation_id != invocation_id
                or terminal.terminal_digest != expected_terminal_digest
                or allocation.canonical_digest != expected_allocation_digest
                or allocation.request_digest != expected_request_digest
                or allocation.cycle_id != expected_cycle_id
            ):
                raise ModelUsageIntegrityError(
                    "native embedding disposition binding differs"
                )

            policy = _policy_for_allocation(connection, allocation)
            if self._validate_terminal(
                terminal,
                allocation.workload_class,
                policy,
                requested_max_output_tokens=allocation.max_output_tokens,
            ) is not None:
                raise ModelUsageIntegrityError(
                    "native embedding disposition has a policy breach"
                )
            authority = _native_embedding_timeout_disposition_authority(
                connection,
                allocation=allocation,
                terminal=terminal,
                policy=policy,
            )
            if (
                allocation.workload_class
                is not WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING
                or allocation.provider != "openrouter"
                or allocation.route != "NATIVE_RETRIEVAL_EMBEDDING"
                or allocation.model != "openai/text-embedding-3-large"
                or authority["passage_id"] != expected_passage_id
                or not policy.qualified
                or terminal.outcome != "NATIVE_EMBEDDING_FAILED"
                or terminal.failure_class != "TimeoutError"
                or terminal.usage_status is not UsageStatus.UNREPORTED
                or terminal.dispatch_at is None
                or terminal.policy_breach is not None
                or terminal.provider_telemetry_digest is not None
                or terminal.raw_telemetry_pointer is not None
                or terminal.pre_dispatch_zero_proved
                or terminal.subscription_cli_chat_not_cash_debited
                or terminal.od_011_reference
                != "OD-011:NATIVE_RETRIEVAL_EMBEDDING"
            ):
                raise ModelUsageIntegrityError(
                    "native embedding disposition target is ineligible"
                )
            if observed_at < terminal.observed_at:
                raise ModelUsageIntegrityError(
                    "native embedding disposition precedes terminal"
                )
            if not _has_exact_native_embedding_dispatch(
                connection, terminal, allocation
            ):
                raise ModelUsageIntegrityError(
                    "native embedding disposition lacks committed dispatch"
                )
            if connection.execute(
                "SELECT 1 FROM model_provider_telemetry WHERE invocation_id=? "
                "UNION ALL SELECT 1 FROM model_usage_reconciliations "
                "WHERE invocation_id=? LIMIT 1",
                (invocation_id, invocation_id),
            ).fetchone() is not None:
                raise ModelUsageIntegrityError(
                    "native embedding disposition exact telemetry already exists"
                )

            def close_exact_timeout_route(disposition_digest: str) -> None:
                _refresh_current_usage(connection, invocation_id)
                latest_route = connection.execute(
                    "SELECT state,reason,invocation_id FROM "
                    "model_usage_route_circuit_events WHERE route=? "
                    "ORDER BY recorded_at DESC,rowid DESC LIMIT 1",
                    (allocation.route,),
                ).fetchone()
                if (
                    latest_route is not None
                    and tuple(latest_route)
                    == ("OPEN", "TimeoutError", allocation.invocation_id)
                    and not _current_has_active(connection, allocation.route)
                    and _canonical_circuit_route(allocation.route)
                    not in _usage_blocking_routes(connection)
                ):
                    self._append_route_state(
                        connection,
                        route=allocation.route,
                        state="CLOSED",
                        reason=(
                            "NATIVE_EMBEDDING_TIMEOUT_CONSERVATIVE_DISPOSITION:"
                            f"{disposition_digest}"
                        ),
                        invocation_id=allocation.invocation_id,
                        recorded_at=observed_at,
                    )

            prior = _valid_native_disposition(
                connection, allocation=allocation, terminal=terminal
            )
            if prior is not None:
                close_exact_timeout_route(str(prior["disposition_digest"]))
                connection.commit()
                return prior

            scope_digest = digest_canonical(authority)
            conservative_total = max(
                policy.max_total_tokens, allocation.prompt_bytes
            )
            record_without_digest: dict[str, object] = {
                "schema_version": CONSERVATIVE_DISPOSITION_SCHEMA_VERSION,
                "authority_scope": NATIVE_EMBEDDING_TIMEOUT_USAGE_SCOPE,
                "native_scope_digest": scope_digest,
                "invocation_id": invocation_id,
                "terminal_digest": terminal.terminal_digest,
                "allocation_digest": allocation.canonical_digest,
                "policy_digest": policy.canonical_digest,
                "usage_status": UsageStatus.ESTIMATED.value,
                "components": UsageComponents(
                    total_tokens=conservative_total,
                    provenance="BOUNDED_ESTIMATE",
                ).as_record(),
                "estimate_policy_digest": policy.canonical_digest,
                "estimate_calculation": (
                    "MAX_QUALIFIED_POLICY_TOTAL_OR_EXACT_REQUEST_UTF8_BYTES"
                ),
                "exact_usage_remains_unknown": True,
                "provider_dispatch_preserved": True,
                "unknown_spend_released": False,
                "authority_digest": scope_digest,
                "observed_at": observed_at_text,
                **{
                    key: authority[key]
                    for key in (
                        "revision_id",
                        "authority_schema_version",
                        "unit_ingest_id",
                        "passage_id",
                        "embedding_cycle_id",
                        "embedding_attempt_number",
                        "progress_seq",
                        "progress_payload_digest",
                        "landed_unit_digest",
                        "source_observation_digest",
                        "source_admission_id",
                        "source_access_decision_id",
                        "source_revision_id",
                        "source_representation_id",
                        "request_digest",
                        "request_bytes",
                        "qualified_policy_maximum_total_tokens",
                        "conservative_total_tokens",
                        "estimated_policy_ceiling_exceeded",
                        "exact_policy_compliance_unknown",
                        "cash_spend_known",
                    )
                },
            }
            disposition_digest = digest_canonical(record_without_digest)
            record = {
                **record_without_digest,
                "disposition_digest": disposition_digest,
            }
            connection.execute(
                "INSERT INTO model_usage_conservative_dispositions("
                "disposition_digest,invocation_id,terminal_digest,"
                "allocation_digest,policy_digest,approved_plan_digest,"
                "authority_digest,approved_by,approval_reference,approved_at,"
                "observed_at,usage_status,record_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    disposition_digest,
                    invocation_id,
                    terminal.terminal_digest,
                    allocation.canonical_digest,
                    policy.canonical_digest,
                    scope_digest,
                    scope_digest,
                    NATIVE_EMBEDDING_TIMEOUT_USAGE_SCOPE,
                    NATIVE_EMBEDDING_TIMEOUT_USAGE_SCOPE,
                    observed_at_text,
                    observed_at_text,
                    UsageStatus.ESTIMATED.value,
                    _json(record),
                ),
            )
            close_exact_timeout_route(disposition_digest)
            connection.commit()
            return record
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def disposition_unreported_subscription_usage(
        self,
        *,
        invocation_id: str,
        expected_terminal_digest: str,
        expected_allocation_digest: str,
        approved_by: str,
        approval_reference: str,
        approved_at: datetime,
        approved_plan_digest: str,
        authority_digest: str,
        observed_at: datetime,
    ) -> dict[str, object]:
        """Retain one authorised upper-bound estimate without rewriting history.

        This deliberately narrow path exists for a dispatched Cursor subscription
        CLI Graphiti primary leaf whose exact provider telemetry is permanently
        absent.  It preserves the unresolved terminal and provider dispatch while
        using the retained qualified policy maximum as the conservative total.
        """

        invocation_id = _token(invocation_id, field="invocation id")
        expected_terminal_digest = _token(
            expected_terminal_digest,
            field="expected terminal digest",
        )
        expected_allocation_digest = _token(
            expected_allocation_digest,
            field="expected allocation digest",
        )
        approved_by = _token(approved_by, field="approved by")
        approval_reference = _token(
            approval_reference,
            field="approval reference",
        )
        authority_digest = _token(authority_digest, field="authority digest")
        approved_plan_digest = _token(
            approved_plan_digest,
            field="approved plan digest",
        )
        try:
            preview = sqlite3.connect(self.path)
            try:
                approved_contract = effective_issue_790_plan_contract(
                    approved_plan_digest,
                    connection=preview,
                )
            finally:
                preview.close()
        except KeyError as exc:
            raise ModelUsageIntegrityError(
                "conservative disposition approved plan differs"
            ) from exc
        if invocation_id != approved_contract.invocation_id:
            raise ModelUsageIntegrityError(
                "conservative disposition approved invocation differs"
            )
        if expected_terminal_digest != approved_contract.terminal_digest:
            raise ModelUsageIntegrityError(
                "conservative disposition approved terminal differs"
            )
        if expected_allocation_digest != approved_contract.allocation_digest:
            raise ModelUsageIntegrityError(
                "conservative disposition approved allocation differs"
            )
        approved_at_text = _utc_text(approved_at)
        observed_at_text = _utc_text(observed_at)
        if (
            approved_by != approved_contract.approved_by
            or approval_reference != approved_contract.approval_reference
            or approved_at_text != approved_contract.approved_at
        ):
            raise ModelUsageIntegrityError(
                "conservative disposition approval authority differs"
            )
        if observed_at < approved_at:
            raise ModelUsageIntegrityError(
                "conservative disposition observation precedes approval"
            )

        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            prior_row = connection.execute(
                "SELECT authority_digest,record_json "
                "FROM model_usage_conservative_dispositions "
                "WHERE invocation_id=?",
                (invocation_id,),
            ).fetchone()
            if prior_row is not None:
                prior = _object(prior_row[1])
                if str(prior_row[0]) != authority_digest:
                    raise ModelUsageIntegrityError(
                        "conservative disposition authority differs"
                    )
                if (
                    prior.get("terminal_digest") != expected_terminal_digest
                    or prior.get("allocation_digest")
                    != expected_allocation_digest
                    or prior.get("approved_by") != approved_by
                    or prior.get("approval_reference") != approval_reference
                    or prior.get("approved_at") != approved_at_text
                    or prior.get("approved_plan_digest")
                    != approved_plan_digest
                ):
                    raise ModelUsageIntegrityError(
                        "conflicting conservative disposition replay"
                    )
                connection.rollback()
                return prior

            retained_row = connection.execute(
                "SELECT t.terminal_digest,t.record_json,a.canonical_digest,"
                "a.record_json,p.canonical_digest,p.record_json "
                "FROM model_invocation_terminals t "
                "JOIN model_invocation_allocations a "
                "ON a.invocation_id=t.invocation_id "
                "JOIN model_invocation_policies p "
                "ON p.canonical_digest=a.policy_digest "
                "WHERE t.invocation_id=?",
                (invocation_id,),
            ).fetchone()
            if retained_row is None:
                raise ModelUsageIntegrityError(
                    "conservative disposition terminal is absent"
                )
            terminal_digest = str(retained_row[0])
            terminal = _object(retained_row[1])
            allocation_digest = str(retained_row[2])
            allocation = _object(retained_row[3])
            policy_digest = str(retained_row[4])
            policy = _policy_from_record(_object(retained_row[5]))

            if terminal_digest != expected_terminal_digest:
                raise ModelUsageIntegrityError(
                    "conservative disposition terminal identity differs"
                )
            if allocation_digest != expected_allocation_digest:
                raise ModelUsageIntegrityError(
                    "conservative disposition allocation identity differs"
                )

            authority = {
                "schema_version": (
                    CONSERVATIVE_DISPOSITION_AUTHORITY_SCHEMA_VERSION
                ),
                "approved_plan_digest": approved_plan_digest,
                "approved_by": approved_by,
                "approval_reference": approval_reference,
                "approved_at": approved_at_text,
                "invocation_id": invocation_id,
                "terminal_digest": terminal_digest,
                "allocation_digest": allocation_digest,
                "scope": approved_contract.scope,
            }
            if digest_canonical(authority) != authority_digest:
                raise ModelUsageIntegrityError(
                    "conservative disposition authority differs"
                )
            terminal_observed_at = _instant(str(terminal["observed_at"]))
            if approved_at < terminal_observed_at:
                raise ModelUsageIntegrityError(
                    "conservative disposition approval precedes terminal"
                )
            if (
                allocation.get("workload_class")
                != WorkloadClass.GRAPHITI_CHAT_PRIMARY.value
                or allocation.get("provider") != "cursor-agent-cli"
                or allocation.get("route")
                != WorkloadClass.GRAPHITI_CHAT_PRIMARY.value
            ):
                raise ModelUsageIntegrityError(
                    "conservative disposition target is outside the approved route"
                )
            if (
                terminal.get("usage_status") != UsageStatus.UNREPORTED.value
                or terminal.get("outcome") != approved_contract.terminal_outcome
                or terminal.get("failure_class")
                != "MISSING_PROVIDER_TELEMETRY"
                or terminal.get("subscription_cli_chat_not_cash_debited")
                is not True
                or terminal.get("policy_breach") is not None
            ):
                raise ModelUsageIntegrityError(
                    "conservative disposition terminal is ineligible"
                )
            if (
                terminal.get("provider_telemetry_digest") is not None
                or terminal.get("raw_telemetry_pointer") is not None
                or connection.execute(
                    "SELECT 1 FROM model_provider_telemetry "
                    "WHERE invocation_id=? LIMIT 1",
                    (invocation_id,),
                ).fetchone()
                is not None
                or connection.execute(
                    "SELECT 1 FROM model_usage_reconciliations "
                    "WHERE invocation_id=? LIMIT 1",
                    (invocation_id,),
                ).fetchone()
                is not None
            ):
                raise ModelUsageIntegrityError(
                    "conservative disposition exact telemetry already exists"
                )
            if connection.execute(
                "SELECT 1 FROM model_transport_observations "
                "WHERE invocation_id=? AND state='DISPATCH_STARTED' LIMIT 1",
                (invocation_id,),
            ).fetchone() is None:
                raise ModelUsageIntegrityError(
                    "conservative disposition committed transport dispatch is absent"
                )
            if not policy.qualified or policy.canonical_digest != policy_digest:
                raise ModelUsageIntegrityError(
                    "conservative disposition policy is not qualified"
                )

            components = UsageComponents(
                total_tokens=policy.max_total_tokens,
                provenance="BOUNDED_ESTIMATE",
            )
            record_without_digest: dict[str, object] = {
                "schema_version": CONSERVATIVE_DISPOSITION_SCHEMA_VERSION,
                "invocation_id": invocation_id,
                "terminal_digest": terminal_digest,
                "allocation_digest": allocation_digest,
                "policy_digest": policy_digest,
                "approved_plan_digest": approved_plan_digest,
                "usage_status": UsageStatus.ESTIMATED.value,
                "components": components.as_record(),
                "estimate_policy_digest": policy_digest,
                "estimate_calculation": (
                    "QUALIFIED_POLICY_MAX_TOTAL_TOKENS_CONSERVATIVE_UPPER_BOUND"
                ),
                "exact_usage_remains_unknown": True,
                "provider_dispatch_preserved": True,
                "unknown_spend_released": False,
                "authority_digest": authority_digest,
                "approved_by": approved_by,
                "approval_reference": approval_reference,
                "approved_at": approved_at_text,
                "observed_at": observed_at_text,
            }
            disposition_digest = digest_canonical(record_without_digest)
            record = {
                **record_without_digest,
                "disposition_digest": disposition_digest,
            }
            connection.execute(
                "INSERT INTO model_usage_conservative_dispositions("
                "disposition_digest,invocation_id,terminal_digest,"
                "allocation_digest,policy_digest,approved_plan_digest,"
                "authority_digest,approved_by,approval_reference,approved_at,"
                "observed_at,usage_status,record_json"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    disposition_digest,
                    invocation_id,
                    terminal_digest,
                    allocation_digest,
                    policy_digest,
                    approved_plan_digest,
                    authority_digest,
                    approved_by,
                    approval_reference,
                    approved_at_text,
                    observed_at_text,
                    UsageStatus.ESTIMATED.value,
                    _json(record),
                ),
            )
            _refresh_current_usage(connection, invocation_id)
            connection.commit()
            return record
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def reconcile(
        self,
        *,
        invocation_id: str,
        components: UsageComponents,
        provider_telemetry: Mapping[str, object],
        observed_at: datetime,
        raw_telemetry_pointer: str,
    ) -> None:
        connection = self._connection()
        try:
            terminal_row = connection.execute(
                "SELECT t.record_json,a.route,p.record_json "
                "FROM model_invocation_terminals t "
                "JOIN model_invocation_allocations a "
                "ON a.invocation_id=t.invocation_id "
                "JOIN model_invocation_policies p "
                "ON p.canonical_digest=a.policy_digest "
                "WHERE t.invocation_id=?",
                (invocation_id,),
            ).fetchone()
            if terminal_row is None:
                raise ModelUsageIntegrityError("terminal usage state is absent")
            terminal = _object(terminal_row[0])
            route = str(terminal_row[1])
            policy = _policy_from_record(_object(terminal_row[2]))
            if terminal.get("usage_status") not in {
                "UNREPORTED",
                "AMBIGUOUS",
                "INVALID",
            }:
                raise ModelUsageIntegrityError("terminal usage is already exact")
            if (
                components.total_tokens is None
                or components.provenance != "PROVIDER_REPORTED"
            ):
                raise ModelUsageIntegrityError(
                    "reconciliation usage is not provider-reported"
                )
            known = [
                value
                for value in (
                    components.input_tokens,
                    components.output_tokens,
                    components.cached_read_tokens,
                    components.cached_write_tokens,
                    components.reasoning_tokens,
                )
                if value is not None
            ]
            direct = sum(
                int(value)
                for value in (components.input_tokens, components.output_tokens)
                if value is not None
            )
            expanded = sum(int(value) for value in known)
            advisory = native_sdk_reported_token_targets_are_advisory(policy)
            if advisory:
                invalid = _native_sdk_reported_components_error(components)
                if invalid is not None:
                    raise ModelUsageIntegrityError(invalid)
            elif known and components.total_tokens not in {direct, expanded}:
                raise ModelUsageIntegrityError(
                    "reconciled component total is impossible"
                )
            provider_telemetry_digest = _retain_provider_telemetry(
                connection,
                invocation_id=invocation_id,
                provider_telemetry=provider_telemetry,
            )
            record = {
                "schema_version": MODEL_USAGE_SCHEMA_VERSION,
                "invocation_id": invocation_id,
                "usage_status": UsageStatus.REPORTED.value,
                "components": components.as_record(),
                "provider_telemetry_digest": provider_telemetry_digest,
                "raw_telemetry_pointer": raw_telemetry_pointer,
                "observed_at": _utc_text(observed_at),
                "policy_breach": (
                    "MAX_TOTAL_TOKENS_EXCEEDED"
                    if not advisory and components.total_tokens > policy.max_total_tokens
                    else "MAX_CONTEXT_TOKENS_EXCEEDED"
                    if components.context_tokens is not None
                    and components.context_tokens > policy.max_context_tokens
                    else "MAX_OUTPUT_TOKENS_EXCEEDED"
                    if not advisory and policy.max_output_tokens is not None
                    and components.output_tokens is not None
                    and components.output_tokens > policy.max_output_tokens
                    else None
                ),
            }
            digest = digest_canonical(record)
            connection.execute(
                "INSERT OR IGNORE INTO model_usage_reconciliations("
                "reconciliation_digest,invocation_id,observed_at,record_json) "
                "VALUES(?,?,?,?)",
                (
                    digest,
                    invocation_id,
                    record["observed_at"],
                    _json({**record, "reconciliation_digest": digest}),
                ),
            )
            _refresh_current_usage(connection, invocation_id)
            canonical_route = _canonical_circuit_route(route)
            blocking_cause_on_canonical_route = (
                canonical_route in _usage_blocking_routes(connection)
            )
            prior_route_state = self._route_state(connection, route)
            if record["policy_breach"] is not None:
                self._append_route_state(
                    connection,
                    route=route,
                    state="OPEN",
                    reason=str(record["policy_breach"]),
                    invocation_id=invocation_id,
                    recorded_at=observed_at,
                )
            elif (
                not blocking_cause_on_canonical_route
                and prior_route_state["state"] == "OPEN"
            ):
                self._append_route_state(
                    connection,
                    route=route,
                    state="CLOSED",
                    reason="VALID_PROVIDER_TELEMETRY_RECONCILED",
                    invocation_id=invocation_id,
                    recorded_at=observed_at,
                )
            connection.commit()
        finally:
            connection.close()

    def graphiti_work_route_state(self, route: str) -> dict[str, object]:
        """Admission for independent work; historical usage/circuit truth is retained."""
        connection = self._connection()
        try:
            connection.execute("BEGIN")
            return self._graphiti_work_route_state(connection, route)
        finally:
            connection.close()

    def _graphiti_work_route_state(
        self, connection: sqlite3.Connection, route: str,
        *, blocking_routes: set[str] | None = None,
    ) -> dict[str, object]:
        state = self._route_state(connection, route, blocking_routes=blocking_routes)
        if route not in {"GRAPHITI_CHAT_PRIMARY", "GRAPHITI_CHAT_FALLBACK", "GRAPHITI_EMBEDDING"} or state["state"] != "OPEN":
            return state
        request_failures = {
            "MISSING_PROVIDER_TELEMETRY", "UNREPORTED", "AMBIGUOUS", "TIMEOUT",
            "TimeoutError", "CliTimeoutError", "CANCELLED", "CANCELLATION",
            "FAILED", "AMBIGUOUS_DISPATCH", "SYSTEMIC_TRANSPORT",
        }
        if state.get("reason") not in request_failures or not state.get("event_digest"):
            return state
        event = _require_open_route_event(connection, str(state["event_digest"]), route=route)
        head_id = event.get("invocation_id")
        if not head_id:
            return state
        rows = connection.execute(
            "SELECT invocation_id,active,unresolved,policy_breach FROM model_usage_current WHERE route=?",
            (route,),
        ).fetchall()
        # A conservative disposition may remove CURRENT debt without changing
        # the original terminal or the historical OPEN event. Check both.
        isolated = {head_id}
        for invocation_id, active, unresolved, breach in rows:
            if breach:
                return state
            if not active:
                if not unresolved:
                    return state
                isolated.add(invocation_id)
        for invocation_id in isolated:
            prior, terminal = _retained_terminal_allocation(connection, invocation_id)
            if (prior.route != route or terminal.policy_breach
                    or terminal.usage_status not in {UsageStatus.UNREPORTED, UsageStatus.AMBIGUOUS}
                    or terminal.failure_class not in request_failures):
                return state
        # CLOSED describes admission, not provider health or settled old usage.
        return {**state, "state": "CLOSED", "historical_circuit_state": "OPEN",
                "reason": "REQUEST_FAILURE_ISOLATED", "availability": "UNOBSERVED"}

    def _require_independent_graphiti_work(
        self, connection: sqlite3.Connection, allocation: InvocationAllocation,
        identity: GraphitiInternalRequestIdentity,
    ) -> None:
        """An unknown old work identity cannot acquire a new attempt or backend."""
        rows = connection.execute(
            "SELECT invocation_id,active,unresolved,policy_breach FROM model_usage_current "
            "WHERE route IN ('GRAPHITI_CHAT_PRIMARY','GRAPHITI_CHAT_FALLBACK','GRAPHITI_EMBEDDING')"
        ).fetchall()
        pending = {row[0]: bool(row[1]) for row in rows}
        # Lookup this work only. A new route head or a conservative estimate
        # must not hide an older dispatched UNKNOWN, including a renamed ingest.
        unknown = connection.execute(
            "SELECT r.invocation_id FROM graphiti_internal_requests r "
            "JOIN model_invocation_terminals t USING(invocation_id) "
            "WHERE json_extract(r.record_json,'$.effective_revision_digest')=? "
            "AND t.usage_status IN ('UNREPORTED','AMBIGUOUS') UNION "
            "SELECT a.invocation_id FROM model_work_envelopes e "
            "JOIN model_invocation_allocations a USING(envelope_id) "
            "JOIN model_invocation_terminals t USING(invocation_id) "
            "WHERE e.workload_class='GRAPHITI_CHAT_PRIMARY' "
            "AND json_extract(e.record_json,'$.ingest_id')=? "
            "AND t.usage_status IN ('UNREPORTED','AMBIGUOUS')",
            (identity.effective_revision_digest, identity.ingest_obligation_id),
        ).fetchall()
        for (invocation_id,) in unknown:
            pending.setdefault(invocation_id, False)
        for invocation_id, active in pending.items():
            if invocation_id == allocation.invocation_id:
                continue
            prior = (
                model_usage_current._active_allocation(connection, invocation_id) if active
                else _retained_terminal_allocation(connection, invocation_id)[0]
            )
            previous = _retained_graphiti_request_identity(connection, prior)
            if previous is None:
                raise ModelUsageAdmissionError("Graphiti unresolved work identity is absent")
            same_work = (previous.ingest_obligation_id == identity.ingest_obligation_id
                         or previous.effective_revision_digest == identity.effective_revision_digest)
            if same_work and (not active or previous.graphiti_attempt_id != identity.graphiti_attempt_id):
                raise ModelUsageAdmissionError(
                    "Graphiti work has an active or unresolved prior attempt",
                    reason_code="GRAPHITI_PRIOR_WORK_UNRESOLVED",
                )

    def require_graphiti_dispatch_available(self, allocation: InvocationAllocation) -> None:
        connection = self._connection()
        try:
            connection.execute("BEGIN")
            retained = model_usage_current._active_allocation(connection, allocation.invocation_id)
            terminal = connection.execute(
                "SELECT 1 FROM model_invocation_terminals WHERE invocation_id=?", (allocation.invocation_id,),
            ).fetchone()
            if retained != allocation or terminal is not None:
                raise ModelUsageAdmissionError("Graphiti dispatch allocation is not active")
            identity = _retained_graphiti_request_identity(connection, retained)
            if identity is None:
                raise ModelUsageAdmissionError("Graphiti dispatch identity is absent")
            self._require_independent_graphiti_work(connection, allocation, identity)
            if self._graphiti_work_route_state(connection, allocation.route)["state"] != "CLOSED":
                raise ModelUsageAdmissionError("Graphiti route unavailable before dispatch")
        finally:
            connection.close()

    def route_state(self, route: str) -> dict[str, object]:
        connection = self._connection()
        try:
            connection.execute("BEGIN")
            return self._route_state(connection, route)
        finally:
            connection.close()

    @contextmanager
    def route_state_snapshot(
        self, *, graphiti_work: bool = False,
    ) -> Iterator[Callable[[str], dict[str, object]]]:
        """Authenticate global blockers once for one consistent route decision."""
        connection = self._connection()
        try:
            connection.execute("BEGIN")
            blocking_routes = _usage_blocking_routes(connection)
            reader = self._graphiti_work_route_state if graphiti_work else self._route_state
            yield lambda route: reader(connection, route, blocking_routes=blocking_routes)
        finally:
            connection.close()

    def open_route_circuit(
        self,
        *,
        route: str,
        reason: str,
        invocation_id: str | None,
        recorded_at: datetime,
    ) -> None:
        """Open one affected Graphiti route after a typed systemic outcome."""

        if not route.startswith("GRAPHITI_"):
            raise ModelUsageAdmissionError(
                "Graphiti route circuit operation targeted another workload"
            )
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._append_route_state(
                connection,
                route=route,
                state="OPEN",
                reason=_token(reason, field="route circuit reason"),
                invocation_id=invocation_id,
                recorded_at=recorded_at,
            )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def disposition_native_reported_output_rejection(
        self, *, invocation_id: str, revision_id: str,
        expected_terminal_digest: str, expected_allocation_digest: str,
        observed_at: datetime,
    ) -> dict[str, object]:
        """Isolate one accounted FAILED candidate; neither accept nor retry it.

        SDK output/total targets are not provider-enforced limits. A fully
        reported subscription overrun fails only that unit; all usage and the
        old OPEN/FAILED records stay intact. v1 records retain their old proof.
        """
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            prior = connection.execute(
                "SELECT disposition_digest,record_json FROM model_usage_reported_output_dispositions WHERE invocation_id=?",
                (invocation_id,),
            ).fetchone()
            authority = _reported_output_disposition_authority(
                connection, invocation_id=invocation_id, revision_id=revision_id,
                schema_version=(_object(prior[1]).get("schema_version") if prior
                                else REPORTED_SUBSCRIPTION_OVERRUN_SCHEMA),
            )
            if (authority["terminal_digest"] != expected_terminal_digest
                    or authority["allocation_digest"] != expected_allocation_digest
                    or observed_at < _instant(str(authority["failure_settled_at"]))):
                raise ModelUsageIntegrityError("reported output disposition binding differs")
            if prior is None:
                record = {**authority, "observed_at": _utc_text(observed_at)}
                record["disposition_digest"] = digest_canonical(record)
                connection.execute(
                    "INSERT INTO model_usage_reported_output_dispositions VALUES(?,?,?)",
                    (invocation_id, record["disposition_digest"], _json(record)),
                )
            else:
                record = _object(prior[1])
                expected = {**authority, "observed_at": record.get("observed_at")}
                if _instant(str(expected["observed_at"])) < _instant(str(authority["failure_settled_at"])):
                    raise ModelUsageIntegrityError("reported output disposition precedes failure")
                expected["disposition_digest"] = digest_canonical(expected)
                if record != expected or prior[0] != record["disposition_digest"] or prior[1] != _json(record):
                    raise ModelUsageIntegrityError("reported output disposition differs")
            _refresh_current_usage(connection, invocation_id)
            latest = connection.execute(
                "SELECT state,reason,invocation_id,recorded_at FROM model_usage_route_circuit_events "
                "WHERE route='GRAPHITI_CHAT_PRIMARY' ORDER BY recorded_at DESC,rowid DESC LIMIT 1"
            ).fetchone()
            if (
                latest is not None and latest[0] == "OPEN" and latest[2] == invocation_id
                and observed_at >= _instant(str(latest[3]))
                and latest[1] in {"REQUESTED_MAX_OUTPUT_TOKENS_EXCEEDED", "CONTEXT_OUTPUT_BREACH"}
                and not _current_has_active(connection, "GRAPHITI_CHAT_PRIMARY")
                and "GRAPHITI_CHAT_PRIMARY" not in _usage_blocking_routes(connection)
            ):
                self._append_route_state(
                    connection, route="GRAPHITI_CHAT_PRIMARY", state="CLOSED",
                    reason="AUTHORISED_OPERATOR_RESET:" + str(record["disposition_digest"]),
                    invocation_id=None, recorded_at=observed_at,
                )
            connection.commit()
            return record
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def requalify_native_assessor_input_bound(
        self, *, invocation_id: str, qualified_policy_digest: str, recorded_at: datetime,
    ) -> str:
        """Release only a proved, corrected assessor input-bound failure."""
        return self._requalify_native_assessor(
            invocation_id, qualified_policy_digest, recorded_at, _ASSESSOR_REQUALIFICATION_KIND,
        )

    def requalify_native_assessor_output_guard(
        self, *, invocation_id: str, qualified_policy_digest: str, recorded_at: datetime,
    ) -> str:
        """Exempt one fully accounted v19 output rejection for future v20 work."""
        return self._requalify_native_assessor(
            invocation_id, qualified_policy_digest, recorded_at, _ASSESSOR_OUTPUT_REQUALIFICATION_KIND,
        )

    def _requalify_native_assessor(
        self, invocation_id: str, qualified_policy_digest: str, recorded_at: datetime, kind: str,
    ) -> str:
        from .store import append_ledger

        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            authority = _assessor_requalification_authority(
                connection, invocation_id, qualified_policy_digest,
                output_guard=kind == _ASSESSOR_OUTPUT_REQUALIFICATION_KIND,
            )
            prior_rows = connection.execute(
                    "SELECT payload_json FROM ledger WHERE kind=? AND json_extract(payload_json,'$.invocation_id')=?",
                    (kind, invocation_id),
                ).fetchall()
            if len(prior_rows) > 1:
                raise ModelUsageIntegrityError("assessor requalification replay is duplicated")
            prior = prior_rows[0] if prior_rows else None
            if prior is not None:
                record = _object(prior[0])
                if record["qualified_policy_digest"] != qualified_policy_digest:
                    raise ModelUsageIntegrityError("assessor requalification replay policy differs")
                at = _instant(str(record.get("recorded_at")))
                expected = {**authority, "recorded_at": _utc_text(at)}
                expected["requalification_digest"] = digest_canonical(expected)
                if (at < _instant(authority["failure_settled_at"])
                        or expected != record or prior[0] != _json(record)):
                    raise ModelUsageIntegrityError("assessor requalification replay differs")
                return str(record["requalification_digest"])
            route = "NATIVE_EVIDENCE_ASSESSOR"
            state = self._route_state(connection, route)
            if (state["state"] != "OPEN" or state["invocation_id"] != invocation_id
                    or state["reason"] != authority["failure_reason"]
                    or recorded_at < _instant(str(state["recorded_at"]))
                    or recorded_at < _instant(authority["failure_settled_at"])):
                raise ModelUsageAdmissionError("assessor requalification is not bound to the current failure")
            if _current_has_active(connection, route):
                raise ModelUsageAdmissionError("assessor requalification has an active invocation")
            authority["recorded_at"] = _utc_text(recorded_at)
            digest = digest_canonical(authority)
            authority["requalification_digest"] = digest
            append_ledger(connection, kind, authority)
            _refresh_current_usage(connection, invocation_id)
            if route in _usage_blocking_routes(connection):
                raise ModelUsageAdmissionError("assessor requalification leaves another usage blocker")
            self._append_route_state(
                connection, route=route, state="CLOSED",
                reason=f"DETERMINISTIC_HEALTH_PROBE:{digest}", invocation_id=None,
                recorded_at=recorded_at,
            )
            connection.commit()
            return digest
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def release_route_circuit(
        self,
        *,
        route: str,
        release_kind: str,
        bound_failure_reason: str,
        evidence_digest: str,
        recorded_at: datetime,
    ) -> None:
        """Release a Graphiti route using the checked #729 evidence vocabulary."""

        if release_kind not in {
            "DETERMINISTIC_HEALTH_PROBE",
            "AUTHORISED_OPERATOR_RESET",
        }:
            raise ModelUsageAdmissionError("Graphiti circuit release kind is invalid")
        if not route.startswith("GRAPHITI_"):
            raise ModelUsageAdmissionError(
                "Graphiti route circuit operation targeted another workload"
            )
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            state = self._route_state(connection, route)
            if state["state"] != "OPEN":
                raise ModelUsageAdmissionError("Graphiti route circuit is not open")
            if state["reason"] != bound_failure_reason:
                raise ModelUsageAdmissionError(
                    "Graphiti circuit release is not bound to the current failure"
                )
            if _canonical_circuit_route(route) in _usage_blocking_routes(connection):
                raise ModelUsageAdmissionError(
                    "Graphiti circuit has unresolved usage or a policy breach"
                )
            self._append_route_state(
                connection,
                route=route,
                state="CLOSED",
                reason=f"{release_kind}:{_token(evidence_digest, field='evidence digest')}",
                invocation_id=None,
                recorded_at=recorded_at,
            )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _route_state(
        self, connection: sqlite3.Connection, route: str,
        *, blocking_routes: set[str] | None = None,
    ) -> dict[str, object]:
        canonical_route = _canonical_circuit_route(route)
        usage_blocking = canonical_route in (
            _usage_blocking_routes(connection)
            if blocking_routes is None else blocking_routes
        )
        has_canonical = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='unpublished_route_circuits'"
        ).fetchone()
        if canonical_route == CONT_WRITER_ROUTE and has_canonical is not None:
            canonical = connection.execute(
                "SELECT state,open_reason,opened_at FROM unpublished_route_circuits "
                "WHERE route=?",
                (canonical_route,),
            ).fetchone()
            if canonical is not None and (
                str(canonical[0]) == "OPEN" or not usage_blocking
            ):
                return {
                    "route": canonical_route,
                    "state": str(canonical[0]),
                    "reason": str(canonical[1]),
                    "invocation_id": None,
                    "recorded_at": canonical[2],
                    "authority": "UNPUBLISHED_ROUTE_CIRCUIT",
                }
        row = connection.execute(
            "SELECT state,reason,invocation_id,recorded_at,event_digest "
            "FROM model_usage_route_circuit_events WHERE route=? "
            "ORDER BY recorded_at DESC,rowid DESC LIMIT 1",
            (canonical_route,),
        ).fetchone()
        if row is None:
            if usage_blocking:
                return {
                    "route": canonical_route,
                    "state": "OPEN",
                    "reason": "UNRESOLVED_USAGE_OR_POLICY_BREACH",
                    "invocation_id": None,
                    "authority": "MODEL_USAGE_RECEIPT",
                }
            return {
                "route": route,
                "state": "CLOSED",
                "reason": "",
                "invocation_id": None,
            }
        state = "OPEN" if usage_blocking else str(row[0])
        return {
            "route": route,
            "state": state,
            "reason": (
                "UNRESOLVED_USAGE_OR_POLICY_BREACH"
                if usage_blocking and str(row[0]) != "OPEN"
                else str(row[1])
            ),
            "invocation_id": row[2],
            "recorded_at": str(row[3]),
            "event_digest": str(row[4]),
        }

    def _append_route_state(
        self,
        connection: sqlite3.Connection,
        *,
        route: str,
        state: str,
        reason: str,
        invocation_id: str | None,
        recorded_at: datetime,
    ) -> None:
        canonical_route = _canonical_circuit_route(route)
        has_canonical = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='unpublished_route_circuits'"
        ).fetchone()
        if (
            state == "OPEN"
            and canonical_route == CONT_WRITER_ROUTE
            and has_canonical is not None
        ):
            connection.execute(
                "INSERT INTO unpublished_route_circuits("
                "route,state,open_reason,opened_at,release_evidence_json,"
                "release_evidence_digest,last_probe_at) VALUES(?,?,?,?,NULL,NULL,NULL) "
                "ON CONFLICT(route) DO UPDATE SET state='OPEN',open_reason=excluded.open_reason,"
                "opened_at=COALESCE(unpublished_route_circuits.opened_at,excluded.opened_at),"
                "release_evidence_json=NULL,release_evidence_digest=NULL",
                (
                    canonical_route,
                    "OPEN",
                    reason,
                    _utc_text(recorded_at),
                ),
            )
        record = {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "route": canonical_route,
            "state": state,
            "reason": reason,
            "invocation_id": invocation_id,
            "recorded_at": _utc_text(recorded_at),
        }
        digest = digest_canonical(record)
        connection.execute(
            "INSERT OR IGNORE INTO model_usage_route_circuit_events("
            "event_digest,route,state,reason,invocation_id,recorded_at,record_json) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                digest,
                canonical_route,
                state,
                reason,
                invocation_id,
                record["recorded_at"],
                _json({**record, "event_digest": digest}),
            ),
        )

    def query(self, *, start: datetime, end: datetime) -> dict[str, object]:
        if end <= start:
            raise ValueError("usage query end must follow start")
        connection = self._connection()
        try:
            envelope_rows = connection.execute(
                "SELECT record_json FROM model_work_envelopes "
                "WHERE admitted_at>=? AND admitted_at<? ORDER BY admitted_at,envelope_id",
                (_utc_text(start), _utc_text(end)),
            ).fetchall()
            leaf_rows = connection.execute(
                "SELECT a.record_json,t.record_json,o.record_json,e.record_json "
                "FROM model_invocation_allocations a "
                "JOIN model_work_envelopes e ON e.envelope_id=a.envelope_id "
                "LEFT JOIN model_invocation_terminals t ON t.invocation_id=a.invocation_id "
                "LEFT JOIN model_work_outcomes o ON o.envelope_id=a.envelope_id "
                "WHERE (a.allocated_at>=? AND a.allocated_at<?) "
                "OR (t.completed_at>=? AND t.completed_at<?) "
                "OR EXISTS (SELECT 1 FROM model_usage_reconciliations r "
                "WHERE r.invocation_id=a.invocation_id "
                "AND r.observed_at>=? AND r.observed_at<?) "
                "OR EXISTS (SELECT 1 "
                "FROM model_usage_conservative_dispositions d "
                "WHERE d.invocation_id=a.invocation_id "
                "AND d.observed_at>=? AND d.observed_at<?) "
                "OR EXISTS (SELECT 1 FROM model_transport_observations x "
                "WHERE x.invocation_id=a.invocation_id "
                "AND x.state='DISPATCH_STARTED' "
                "AND x.observed_at>=? AND x.observed_at<?) "
                "OR (o.terminal_at>=? AND o.terminal_at<?) "
                "ORDER BY a.allocated_at,a.cycle_id,a.envelope_id,a.leaf_ordinal",
                (
                    _utc_text(start),
                    _utc_text(end),
                    _utc_text(start),
                    _utc_text(end),
                    _utc_text(start),
                    _utc_text(end),
                    _utc_text(start),
                    _utc_text(end),
                    _utc_text(start),
                    _utc_text(end),
                    _utc_text(start),
                    _utc_text(end),
                ),
            ).fetchall()
            outcome_rows = connection.execute(
                "SELECT o.record_json,e.record_json "
                "FROM model_work_outcomes o "
                "JOIN model_work_envelopes e ON e.envelope_id=o.envelope_id "
                "WHERE o.terminal_at<? AND ("
                "(e.admitted_at>=? AND e.admitted_at<?) OR "
                "(o.terminal_at>=? AND o.terminal_at<?)) "
                "ORDER BY o.terminal_at,o.envelope_id",
                (
                    _utc_text(end),
                    _utc_text(start),
                    _utc_text(end),
                    _utc_text(start),
                    _utc_text(end),
                ),
            ).fetchall()
            reconciliations = {
                str(row[0]): _object(row[1])
                for row in connection.execute(
                    "SELECT invocation_id,record_json FROM model_usage_reconciliations "
                    "WHERE rowid IN (SELECT MAX(rowid) FROM model_usage_reconciliations "
                    "WHERE observed_at<? GROUP BY invocation_id)",
                    (_utc_text(end),),
                )
            }
            conservative_dispositions = {
                str(row[0]): _object(row[1])
                for row in connection.execute(
                    "SELECT invocation_id,record_json "
                    "FROM model_usage_conservative_dispositions "
                    "WHERE observed_at<?",
                    (_utc_text(end),),
                )
            }
            provider_attempts = {
                str(row[0]): _object(row[1])
                for row in connection.execute(
                    "SELECT invocation_id,record_json "
                    "FROM model_invocation_provider_attempt_links"
                )
            }
            graphiti_internal_requests = {
                str(row[0]): _object(row[1])
                for row in connection.execute(
                    "SELECT invocation_id,record_json "
                    "FROM graphiti_internal_requests"
                )
            }
            context_manifests = {
                str(row[0]): _object(row[1])
                for row in connection.execute(
                    "SELECT context_manifest_digest,record_json "
                    "FROM model_invocation_context_manifests"
                )
            }
            context_observations = {
                str(row[0]): _object(row[1])
                for row in connection.execute(
                    "SELECT invocation_id,record_json "
                    "FROM model_invocation_context_observations"
                )
            }
            dispatch_observations = {
                str(row[0]): str(row[1])
                for row in connection.execute(
                    "SELECT invocation_id,MIN(observed_at) "
                    "FROM model_transport_observations "
                    "WHERE state='DISPATCH_STARTED' AND observed_at<? "
                    "GROUP BY invocation_id",
                    (_utc_text(end),),
                )
            }
            projected_cycle_outcomes = {
                str(row[0]): _object(row[1])
                for row in connection.execute(
                    "SELECT cycle_id,record_json FROM model_usage_cycle_outcomes "
                    "WHERE terminal_at>=? AND terminal_at<? ORDER BY terminal_at,cycle_id",
                    (_utc_text(start), _utc_text(end)),
                )
            }
            has_governor = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='unpublished_governed_cycles'"
            ).fetchone()
            canonical_cycle_outcomes: dict[str, dict[str, object]] = {}
            if has_governor is not None:
                for row in connection.execute(
                    "SELECT cycle_id,outcome_class,terminal_at,"
                    "writer_unproductive_streak_before,"
                    "writer_unproductive_streak_after,writer_circuit_state,"
                    "writer_circuit_open_reason,cooldown_seconds,"
                    "cooldown_policy_version,next_cycle_eligible_at "
                    "FROM unpublished_governed_cycles "
                    "WHERE lease_state IN ('TERMINAL','RECOVERED') "
                    "AND terminal_at>=? AND terminal_at<? "
                    "ORDER BY terminal_at,cycle_id",
                    (_utc_text(start), _utc_text(end)),
                ):
                    canonical_cycle_outcomes[str(row[0])] = {
                        "schema_version": MODEL_USAGE_SCHEMA_VERSION,
                        "cycle_id": str(row[0]),
                        "outcome_class": str(row[1]),
                        "terminal_at": str(row[2]),
                        "writer_unproductive_streak_before": int(row[3]),
                        "writer_unproductive_streak_after": int(row[4]),
                        "writer_circuit_state": str(row[5]),
                        "writer_circuit_open_reason": str(row[6]),
                        "cooldown_seconds": (
                            None if row[7] is None else int(row[7])
                        ),
                        "cooldown_policy_version": (
                            None if row[8] is None else str(row[8])
                        ),
                        "next_cycle_eligible_at": (
                            None if row[9] is None else str(row[9])
                        ),
                        "authority": "UNPUBLISHED_GOVERNED_CYCLE_TERMINAL",
                    }
            cycle_outcomes = {
                **projected_cycle_outcomes,
                **canonical_cycle_outcomes,
            }
            zero_call_admissions = [
                {
                    "decision_id": str(row[0]),
                    "decision": str(row[1]),
                    "cycle_id": str(row[2]),
                    "recorded_at": str(row[3]),
                }
                for row in connection.execute(
                    "SELECT decision_id,decision,cycle_id,recorded_at "
                    "FROM model_zero_call_admissions "
                    "WHERE recorded_at>=? AND recorded_at<? "
                    "ORDER BY recorded_at,decision_id",
                    (_utc_text(start), _utc_text(end)),
                )
            ]
        finally:
            connection.close()
        outcomes: dict[str, dict[str, object]] = {}
        envelopes = []
        envelope_ids: set[str] = set()
        for row in envelope_rows:
            record = _object(row[0])
            envelopes.append(record)
            envelope_ids.add(str(record["envelope_id"]))
        for outcome_json, envelope_json in outcome_rows:
            outcome = _object(outcome_json)
            envelope = _object(envelope_json)
            envelope_id = str(envelope["envelope_id"])
            outcomes[envelope_id] = outcome
            if envelope_id not in envelope_ids:
                envelopes.append(envelope)
                envelope_ids.add(envelope_id)
        leaves: list[dict[str, object]] = []
        for allocation_json, terminal_json, outcome_json, envelope_json in leaf_rows:
            allocation = _object(allocation_json)
            envelope = _object(envelope_json)
            terminal = None if terminal_json is None else _object(terminal_json)
            if terminal is not None and _instant(str(terminal["completed_at"])) >= end:
                terminal = None
            outcome = None if outcome_json is None else _object(outcome_json)
            if outcome is not None and _instant(str(outcome["terminal_at"])) >= end:
                outcome = None
            effective = dict(terminal or {})
            invocation_id = str(allocation["invocation_id"])
            disposition = conservative_dispositions.get(invocation_id)
            reconciliation = reconciliations.get(invocation_id)
            provider_attempt = provider_attempts.get(invocation_id)
            if disposition is not None:
                effective.update(
                    {
                        "usage_status": disposition["usage_status"],
                        "components": disposition["components"],
                        "estimate_policy_digest": disposition[
                            "estimate_policy_digest"
                        ],
                        "estimate_calculation": disposition[
                            "estimate_calculation"
                        ],
                        "completed_at": disposition["observed_at"],
                        "observed_at": disposition["observed_at"],
                    }
                )
            if reconciliation is not None:
                effective.update(
                    {
                        "usage_status": reconciliation["usage_status"],
                        "components": reconciliation["components"],
                        "provider_telemetry_digest": reconciliation[
                            "provider_telemetry_digest"
                        ],
                        "raw_telemetry_pointer": reconciliation[
                            "raw_telemetry_pointer"
                        ],
                        "reconciled_at": reconciliation["observed_at"],
                        "completed_at": reconciliation["observed_at"],
                        "policy_breach": reconciliation.get("policy_breach"),
                    }
                )
            components = effective.get("components")
            if not isinstance(components, dict):
                components = UsageComponents().as_record()
            row = {
                **allocation,
                "schema_version": MODEL_USAGE_INTERFACE_SCHEMA_VERSION,
                "allocation_schema_version": allocation.get("schema_version"),
                "admission_decision_id": envelope.get("admission_decision_id"),
                "candidate_id": envelope.get("candidate_id"),
                "hypothesis_digest": envelope.get("hypothesis_digest"),
                "evidence_package_digest": envelope.get("evidence_package_digest"),
                "ingest_id": envelope.get("ingest_id"),
                "graphiti_attempt_id": envelope.get("graphiti_attempt_id"),
                "provider_attempt_id": (
                    None
                    if provider_attempt is None
                    else provider_attempt.get("provider_attempt_id")
                ),
                "graphiti_internal_request": graphiti_internal_requests.get(
                    str(allocation["invocation_id"])
                ),
                "context_manifest": context_manifests.get(
                    str(allocation["context_manifest_digest"])
                ),
                "context_manifest_observation": context_observations.get(
                    str(allocation["invocation_id"])
                ),
                "usage_status": effective.get("usage_status"),
                "terminal_digest": (
                    None if terminal is None else terminal.get("terminal_digest")
                ),
                "terminal_usage_status": (
                    None if terminal is None else terminal.get("usage_status")
                ),
                "terminal_components": (
                    None if terminal is None else terminal.get("components")
                ),
                "terminal_completed_at": (
                    None if terminal is None else terminal.get("completed_at")
                ),
                "reconciliation_usage_status": (
                    None
                    if reconciliation is None
                    else reconciliation.get("usage_status")
                ),
                "reconciliation_components": (
                    None
                    if reconciliation is None
                    else reconciliation.get("components")
                ),
                "reconciled_at": effective.get("reconciled_at"),
                "conservative_disposition_digest": (
                    None
                    if disposition is None
                    else disposition.get("disposition_digest")
                ),
                "disposition_usage_status": (
                    None if disposition is None else disposition.get("usage_status")
                ),
                "disposition_components": (
                    None if disposition is None else disposition.get("components")
                ),
                "disposed_at": (
                    None if disposition is None else disposition.get("observed_at")
                ),
                "disposition_authority_digest": (
                    None
                    if disposition is None
                    else disposition.get("authority_digest")
                ),
                "disposition_approval_reference": (
                    None
                    if disposition is None
                    else disposition.get("approval_reference")
                ),
                "disposition_approved_plan_digest": (
                    None
                    if disposition is None
                    else disposition.get("approved_plan_digest")
                ),
                "exact_usage_remains_unknown": (
                    None
                    if disposition is None
                    else reconciliation is None
                ),
                "provider_dispatch_preserved": (
                    None
                    if disposition is None
                    else disposition.get("provider_dispatch_preserved")
                ),
                "unknown_spend_released": (
                    None
                    if disposition is None
                    else disposition.get("unknown_spend_released")
                ),
                "invocation_outcome": effective.get("outcome"),
                "failure_class": effective.get("failure_class"),
                **components,
                "dispatch_at": (
                    None
                    if effective.get("pre_dispatch_zero_proved")
                    else dispatch_observations.get(
                        str(allocation["invocation_id"]),
                        effective.get("dispatch_at"),
                    )
                ),
                "transport_dispatch_observed": str(allocation["invocation_id"])
                in dispatch_observations,
                "actual_provider_dispatch": bool(
                    not effective.get("pre_dispatch_zero_proved")
                    and (
                        str(allocation["invocation_id"]) in dispatch_observations
                        or (
                            terminal is not None
                            and terminal.get("dispatch_at") is not None
                            and terminal.get("outcome") != "RECOVERED_UNRESOLVED"
                        )
                    )
                ),
                "completed_at": effective.get("completed_at"),
                "observed_at": effective.get("observed_at"),
                "provider_telemetry_digest": effective.get("provider_telemetry_digest"),
                "raw_telemetry_pointer": effective.get("raw_telemetry_pointer"),
                "estimate_policy_digest": effective.get("estimate_policy_digest"),
                "estimate_calculation": effective.get("estimate_calculation"),
                "od_011_reference": effective.get("od_011_reference"),
                "subscription_cli_chat_not_cash_debited": effective.get(
                    "subscription_cli_chat_not_cash_debited"
                ),
                "pre_dispatch_zero_proved": effective.get(
                    "pre_dispatch_zero_proved", False
                ),
                "policy_breach": effective.get("policy_breach"),
                "uncertainty": (
                    effective.get("usage_status")
                    if effective.get("usage_status")
                    in {"ESTIMATED", "UNREPORTED", "AMBIGUOUS", "INVALID"}
                    else ""
                ),
                "work_outcome": None if outcome is None else outcome.get("outcome"),
                "work_outcome_record_id": (
                    None if outcome is None else outcome.get("outcome_record_id")
                ),
                "work_outcome_terminal_at": (
                    None if outcome is None else outcome.get("terminal_at")
                ),
                "payload_digest": None
                if outcome is None
                else outcome.get("payload_digest"),
                "cycle_outcome": None
                if outcome is None
                else outcome.get("cycle_outcome"),
                "route_circuit_state": None
                if outcome is None
                else outcome.get("route_circuit_state"),
                "route_circuit_reason": None
                if outcome is None
                else outcome.get("route_circuit_reason"),
                "retained_proposal_count": None
                if outcome is None
                else outcome.get("retained_proposal_count"),
                "accepted_provider_attempt_id": None
                if outcome is None
                else outcome.get("accepted_provider_attempt_id"),
                "stable_reason_codes": []
                if outcome is None
                else outcome.get("stable_reason_codes", []),
            }
            leaves.append(row)
            if str(envelope["envelope_id"]) not in envelope_ids:
                envelopes.append(envelope)
                envelope_ids.add(str(envelope["envelope_id"]))
            if outcome is not None:
                outcomes[str(outcome["envelope_id"])] = outcome
        for envelope in envelopes:
            work_outcome = outcomes.get(str(envelope["envelope_id"]))
            if work_outcome is not None:
                envelope.update(work_outcome)
                envelope["work_outcome_terminal_at"] = work_outcome["terminal_at"]
            envelope.update(cycle_outcomes.get(str(envelope["cycle_id"]), {}))
        return {
            "envelopes": envelopes,
            "leaves": leaves,
            "cycle_outcomes": [
                cycle_outcomes[key]
                for key in sorted(
                    cycle_outcomes,
                    key=lambda item: (str(cycle_outcomes[item]["terminal_at"]), item),
                )
            ],
            "zero_call_admissions": zero_call_admissions,
        }

    def report(
        self, *, start: datetime, end: datetime, bucket_seconds: int = 300
    ) -> dict[str, object]:
        if bucket_seconds <= 0:
            raise ValueError("usage bucket must be positive")
        data = self.query(start=start, end=end)
        leaves = data["leaves"]
        envelopes = data["envelopes"]
        assert isinstance(leaves, list)
        assert isinstance(envelopes, list)
        allocated_leaves = [
            row
            for row in leaves
            if start <= _instant(str(row["allocated_at"])) < end
        ]
        terminal_leaves = [
            row
            for row in leaves
            if row["usage_status"] is not None
            and isinstance(row.get("completed_at"), str)
            and start <= _instant(str(row["completed_at"])) < end
        ]
        allocation_terminals = [
            row for row in allocated_leaves if row["usage_status"] is not None
        ]
        accounted_leaves = [
            row
            for row in terminal_leaves
            if row["usage_status"] in {"REPORTED", "ESTIMATED"}
        ]
        totals = [
            int(row["total_tokens"])
            for row in accounted_leaves
            if _is_int(row.get("total_tokens"))
        ]
        observed_total = sum(totals)
        reported_total = sum(
            int(row["total_tokens"])
            for row in accounted_leaves
            if row["usage_status"] == "REPORTED" and _is_int(row.get("total_tokens"))
        )
        estimated_total = sum(
            int(row["total_tokens"])
            for row in terminal_leaves
            if row["usage_status"] == "ESTIMATED" and _is_int(row.get("total_tokens"))
        )
        unresolved = sum(
            row["usage_status"] in {"UNREPORTED", "AMBIGUOUS", "INVALID"}
            for row in terminal_leaves
        )
        accepted_envelopes = {
            str(row["envelope_id"])
            for row in leaves
            if row["work_outcome"] == "ACCEPTED" and row["payload_digest"]
        }
        graphiti_envelopes = {
            str(row["envelope_id"])
            for row in leaves
            if row["work_outcome"] in _GRAPHITI_COMPLETED_USEFUL_OUTCOMES
        }

        def productive_leaf(row: Mapping[str, object]) -> bool:
            workload = str(row["workload_class"])
            if workload.startswith("CONT_WRITER_"):
                accepted_attempt = row.get("accepted_provider_attempt_id")
                return bool(
                    accepted_attempt
                    and row.get("provider_attempt_id") == accepted_attempt
                    and row.get("invocation_outcome") == "ACCEPTED_OUTPUT"
                )
            if workload.startswith("GRAPHITI_"):
                return (
                    str(row["envelope_id"]) in graphiti_envelopes
                    and row.get("invocation_outcome") == "COMPLETE"
                )
            return str(row["envelope_id"]) in accepted_envelopes

        productive = sum(
            int(row["total_tokens"])
            for row in accounted_leaves
            if productive_leaf(row) and _is_int(row.get("total_tokens"))
        )
        no_result = observed_total - productive
        no_result_reasons: Counter[str] = Counter()
        for row in accounted_leaves:
            if productive_leaf(row):
                continue
            if _is_int(row.get("total_tokens")):
                stable_reasons = row.get("stable_reason_codes")
                stable_reason = (
                    str(stable_reasons[0])
                    if isinstance(stable_reasons, list) and stable_reasons
                    else None
                )
                no_result_reasons[
                    str(
                        row["failure_class"]
                        or stable_reason
                        or row["invocation_outcome"]
                        or "UNKNOWN"
                    )
                ] += int(row["total_tokens"])
        workload_totals: Counter[str] = Counter()
        provider_totals: Counter[str] = Counter()
        model_totals: Counter[str] = Counter()
        status_totals: Counter[str] = Counter()
        outcome_totals: Counter[str] = Counter()
        context_tokens = 0
        provider_input_tokens = 0
        for row in accounted_leaves:
            total = row.get("total_tokens")
            if _is_int(total):
                workload_totals[str(row["workload_class"])] += total
                provider_totals[str(row["provider"])] += total
                model_totals[str(row["model"])] += total
                status_totals[str(row["usage_status"])] += total
                outcome_totals[
                    str(
                        row["work_outcome"] or row["invocation_outcome"] or "UNRESOLVED"
                    )
                ] += total
            context = row.get("context_tokens")
            if _is_int(context):
                context_tokens += context
            input_tokens = row.get("input_tokens")
            if _is_int(input_tokens) and str(row["workload_class"]).startswith("CONT_"):
                provider_input_tokens += input_tokens
        graphiti_tokens = sum(
            int(row["total_tokens"])
            for row in accounted_leaves
            if str(row["envelope_id"]) in graphiti_envelopes
            and _is_int(row.get("total_tokens"))
        )
        proposals = sum(
            int(row["retained_proposal_count"])
            for row in data["envelopes"]  # type: ignore[union-attr]
            if str(row.get("envelope_id")) in graphiti_envelopes
            and _is_int(row.get("retained_proposal_count"))
        )
        fixed_buckets = self._fixed_buckets(
            leaves=accounted_leaves,
            start=start,
            end=end,
            bucket_seconds=bucket_seconds,
        )
        daily: Counter[str] = Counter()
        for row in accounted_leaves:
            total = row.get("total_tokens")
            completed = row.get("completed_at")
            if _is_int(total) and isinstance(completed, str):
                daily[_instant(completed).date().isoformat()] += total
        zero_calls = self._zero_call_counts(start=start, end=end)
        rolling_data = self.query(
            start=start - timedelta(seconds=300), end=end
        )["leaves"]
        assert isinstance(rolling_data, list)
        rolling_accounted = [
            row
            for row in rolling_data
            if row["usage_status"] in {"REPORTED", "ESTIMATED"}
        ]
        rolling = self._rolling_dispatch_usage(
            rolling_accounted, start=start, end=end
        )
        cycle_rows = data["cycle_outcomes"]
        assert isinstance(cycle_rows, list)
        cycle_counts = Counter(str(row["outcome_class"]) for row in cycle_rows)
        latest_cycle = cycle_rows[-1] if cycle_rows else None
        writer_leaves = [
            row
            for row in terminal_leaves
            if row["workload_class"] in {"CONT_WRITER_PRIMARY", "CONT_WRITER_FALLBACK"}
        ]
        fallback_leaves = [
            row
            for row in writer_leaves
            if row["workload_class"] == "CONT_WRITER_FALLBACK"
        ]
        recovered_fallback_envelopes = {
            str(row["envelope_id"])
            for row in fallback_leaves
            if str(row["envelope_id"]) in accepted_envelopes
        }
        fallback_tokens = sum(
            int(row["total_tokens"])
            for row in fallback_leaves
            if row["usage_status"] in {"REPORTED", "ESTIMATED"}
            and _is_int(row.get("total_tokens"))
        )
        fallback_no_result_tokens = sum(
            int(row["total_tokens"])
            for row in fallback_leaves
            if str(row["envelope_id"]) not in accepted_envelopes
            and row["usage_status"] in {"REPORTED", "ESTIMATED"}
            and _is_int(row.get("total_tokens"))
        )
        accepted_by_cycle: Counter[str] = Counter(
            str(row["cycle_id"])
            for row in data["envelopes"]  # type: ignore[union-attr]
            if row.get("outcome") == "ACCEPTED" and row.get("payload_digest")
        )
        for row in leaves:
            if str(row["workload_class"]).startswith("CONT_WRITER_"):
                accepted_by_cycle.setdefault(str(row["cycle_id"]), 0)
        dispatches = [
            row
            for row in leaves
            if isinstance(row.get("dispatch_at"), str)
            and row.get("actual_provider_dispatch") is True
            and start <= _instant(str(row["dispatch_at"])) < end
        ]
        outstanding = len(allocated_leaves) - len(allocation_terminals)
        envelope_outcome_counts = Counter(
            str(row["outcome"]) for row in envelopes if row.get("outcome")
        )
        return {
            "schema_version": MODEL_USAGE_INTERFACE_SCHEMA_VERSION,
            "start": _utc_text(start),
            "end": _utc_text(end),
            "bucket_seconds": bucket_seconds,
            "envelope_count": len(data["envelopes"]),  # type: ignore[arg-type]
            "envelopes": envelopes,
            "envelope_outcome_counts": dict(sorted(envelope_outcome_counts.items())),
            "allocation_count": len(allocated_leaves),
            "actual_provider_dispatch_count": len(dispatches),
            "terminal_count": len(allocation_terminals),
            "outstanding_count": outstanding,
            "allocation_reconciliation": {
                "allocation_count": len(allocated_leaves),
                "terminal_count": len(allocation_terminals),
                "outstanding_count": outstanding,
                "reconciles": len(allocated_leaves)
                == len(allocation_terminals) + outstanding,
            },
            "leaf_dispatch_count": len(dispatches),
            "terminal_leaf_count": len(terminal_leaves),
            "leaf_dispatch_count_reconciles": len(allocated_leaves)
            == len(allocation_terminals) + outstanding,
            "reported_tokens": reported_total,
            "estimated_tokens": estimated_total,
            "observed_total_tokens": observed_total,
            "envelope_allocated_tokens": observed_total,
            "context_tokens": context_tokens,
            "unresolved_invocation_count": unresolved,
            "accepted_payload_count": len(accepted_envelopes),
            "productive_tokens": productive,
            "no_result_tokens": no_result,
            "tokens_per_accepted_payload": (
                {"numerator": productive, "denominator": len(accepted_envelopes)}
                if accepted_envelopes
                else None
            ),
            "writer_leaf_calls_per_accepted_payload": (
                {"numerator": len(writer_leaves), "denominator": len(accepted_envelopes)}
                if accepted_envelopes
                else None
            ),
            "accepted_unpublished_payloads_by_cycle": dict(
                sorted(accepted_by_cycle.items())
            ),
            "no_result_reasons": dict(sorted(no_result_reasons.items())),
            "workload_totals": dict(sorted(workload_totals.items())),
            "provider_totals": dict(sorted(provider_totals.items())),
            "model_totals": dict(sorted(model_totals.items())),
            "usage_status_totals": dict(sorted(status_totals.items())),
            "outcome_totals": dict(sorted(outcome_totals.items())),
            "provider_context_to_input_ratio": {
                "numerator": context_tokens,
                "denominator": provider_input_tokens,
            },
            "context_to_newsroom_input_ratio": None,
            "context_to_newsroom_input_ratio_reason": (
                "AWAITING_730_EXACT_NEWSROOM_INPUT_TOKEN_MEASURE"
            ),
            "fallback_leaf_count": len(fallback_leaves),
            "fallback_tokens": fallback_tokens,
            "fallback_no_result_tokens": fallback_no_result_tokens,
            "fallback_recovery_rate": {
                "numerator": len(recovered_fallback_envelopes),
                "denominator": len(fallback_leaves),
            },
            "fallback_no_result_rate": {
                "numerator": sum(
                    str(row["envelope_id"]) not in accepted_envelopes
                    for row in fallback_leaves
                ),
                "denominator": len(fallback_leaves),
            },
            "graphiti_valid_ingest_count": len(graphiti_envelopes),
            "graphiti_tokens_per_valid_ingest": (
                {"numerator": graphiti_tokens, "denominator": len(graphiti_envelopes)}
                if graphiti_envelopes
                else None
            ),
            "graphiti_tokens_per_retained_proposal": (
                {"numerator": graphiti_tokens, "denominator": proposals}
                if proposals
                else None
            ),
            "zero_call_admission_counts": zero_calls,
            "cycle_outcome_counts": dict(sorted(cycle_counts.items())),
            "writer_unproductive_streak": (
                None
                if latest_cycle is None
                else latest_cycle["writer_unproductive_streak_after"]
            ),
            "fixed_buckets": fixed_buckets,
            "rolling_300_at_dispatch": rolling,
            "utc_day_totals": dict(sorted(daily.items())),
            "daily_500k_alert": any(
                value > DAILY_USAGE_ALERT_TOKENS for value in daily.values()
            ),
            "normal_daily_hard_cut": None,
            "missing_usage_is_zero": False,
            "graphiti_result_telemetry": self._graphiti_result_telemetry(
                leaves=leaves,
                allocated_leaves=allocated_leaves,
                terminal_leaves=terminal_leaves,
                accounted_leaves=accounted_leaves,
                envelopes=envelopes,
                start=start,
                end=end,
            ),
        }

    def _graphiti_result_telemetry(
        self,
        *,
        leaves: list[dict[str, object]],
        allocated_leaves: list[dict[str, object]],
        terminal_leaves: list[dict[str, object]],
        accounted_leaves: list[dict[str, object]],
        envelopes: list[dict[str, object]],
        start: datetime,
        end: datetime,
    ) -> dict[str, object]:
        def graphiti_leaf(row: Mapping[str, object]) -> bool:
            return str(row.get("workload_class") or "").startswith("GRAPHITI_")

        def token_sum(
            rows: list[dict[str, object]],
            *,
            status: str | None = None,
            workload: str | None = None,
        ) -> int:
            total = 0
            for row in rows:
                if status is not None and row.get("usage_status") != status:
                    continue
                if workload is not None and row.get("workload_class") != workload:
                    continue
                value = row.get("total_tokens")
                if _is_int(value):
                    total += value
            return total

        def ratio(numerator: int, denominator: int) -> dict[str, int] | None:
            if denominator == 0:
                return None
            return {"numerator": numerator, "denominator": denominator}

        with_proposals = {
            str(row["envelope_id"])
            for row in leaves
            if row.get("work_outcome") == "GRAPHITI_SUCCESS"
        }
        zero_proposals = {
            str(row["envelope_id"])
            for row in leaves
            if row.get("work_outcome") == "GRAPHITI_SUCCESS_ZERO_PROPOSALS"
        }
        completed = with_proposals | zero_proposals
        request_leaves = [
            row
            for row in leaves
            if isinstance(row.get("graphiti_internal_request"), dict)
        ]
        distinct_internal_requests = len(request_leaves)
        connection = self._connection()
        try:
            refusal_rows = connection.execute(
                "SELECT reason_code FROM graphiti_internal_request_refusals "
                "WHERE refused_at>=? AND refused_at<?",
                (_utc_text(start), _utc_text(end)),
            ).fetchall()
        finally:
            connection.close()
        graphiti_terminals = [row for row in terminal_leaves if graphiti_leaf(row)]
        graphiti_accounted = [row for row in accounted_leaves if graphiti_leaf(row)]
        failed_envelopes = {
            str(row["envelope_id"])
            for row in leaves
            if graphiti_leaf(row)
            and row.get("work_outcome")
            and str(row["envelope_id"]) not in completed
        }
        failed_tokens = sum(
            int(row["total_tokens"])
            for row in graphiti_accounted
            if str(row["envelope_id"]) in failed_envelopes
            and _is_int(row.get("total_tokens"))
        )
        success_tokens = sum(
            int(row["total_tokens"])
            for row in graphiti_accounted
            if str(row["envelope_id"]) in completed
            and _is_int(row.get("total_tokens"))
        )
        proposals = sum(
            int(row["retained_proposal_count"])
            for row in envelopes
            if str(row.get("envelope_id")) in completed
            and _is_int(row.get("retained_proposal_count"))
        )
        context_values = [
            int(row["context_tokens"])
            for row in request_leaves
            if _is_int(row.get("context_tokens"))
        ]
        context_overhead = (
            None
            if len(context_values) != distinct_internal_requests
            else ratio(sum(context_values), distinct_internal_requests)
        )
        call_shape = load_checked_graphiti_call_shape_policy()
        return {
            "completed_ingests_with_proposals": len(with_proposals),
            "completed_ingests_zero_proposals": len(zero_proposals),
            "completed_useful_ingest_count": len(completed),
            "distinct_internal_requests": distinct_internal_requests,
            "distinct_internal_requests_per_completed_ingest": ratio(
                distinct_internal_requests, len(completed)
            ),
            "call_shape_max_distinct_internal_requests": (
                call_shape.max_distinct_internal_requests
            ),
            "call_shape_headroom": call_shape.headroom,
            "duplicate_request_refusals": sum(
                reason == "DUPLICATE_INTERNAL_REQUEST" for reason, in refusal_rows
            ),
            "call_shape_drift_refusals": sum(
                reason == "CALL_SHAPE_DRIFT" for reason, in refusal_rows
            ),
            "primary_leaf_count": sum(
                row["workload_class"] == WorkloadClass.GRAPHITI_CHAT_PRIMARY.value
                for row in allocated_leaves
            ),
            "fallback_leaf_count": sum(
                row["workload_class"] == WorkloadClass.GRAPHITI_CHAT_FALLBACK.value
                for row in allocated_leaves
            ),
            "embedding_leaf_count": sum(
                row["workload_class"] == WorkloadClass.GRAPHITI_EMBEDDING.value
                for row in allocated_leaves
            ),
            "fallback_recovery_count": len(
                {
                    str(row["envelope_id"])
                    for row in allocated_leaves
                    if row["workload_class"]
                    == WorkloadClass.GRAPHITI_CHAT_FALLBACK.value
                    and str(row["envelope_id"]) in completed
                }
            ),
            "reported_tokens": token_sum(graphiti_terminals, status="REPORTED"),
            "estimated_tokens": token_sum(graphiti_terminals, status="ESTIMATED"),
            "unresolved_invocation_count": sum(
                row.get("usage_status") in _UNRESOLVED_USAGE_STATUSES
                for row in graphiti_terminals
            ),
            "failed_or_rolled_back_attempt_tokens": failed_tokens,
            "tokens_per_proposal": ratio(success_tokens, proposals),
            "context_overhead_per_internal_request": context_overhead,
            "route_circuit_states": {
                route: str(self.route_state(route)["state"])
                for route in _GRAPHITI_CIRCUIT_ROUTES
            },
            "embedding_tokens": token_sum(
                graphiti_accounted,
                workload=WorkloadClass.GRAPHITI_EMBEDDING.value,
            ),
            "embedding_od_011_references": sorted(
                {
                    str(row["od_011_reference"])
                    for row in graphiti_terminals
                    if row["workload_class"]
                    == WorkloadClass.GRAPHITI_EMBEDDING.value
                    and isinstance(row.get("od_011_reference"), str)
                    and row["od_011_reference"]
                }
            ),
            "cli_chat_cash_debited": any(
                row.get("subscription_cli_chat_not_cash_debited") is False
                for row in graphiti_terminals
                if row["workload_class"]
                in {
                    WorkloadClass.GRAPHITI_CHAT_PRIMARY.value,
                    WorkloadClass.GRAPHITI_CHAT_FALLBACK.value,
                }
            ),
            "normal_daily_hard_cut": None,
            "missing_usage_is_zero": False,
        }

    def _fixed_buckets(
        self,
        *,
        leaves: list[dict[str, object]],
        start: datetime,
        end: datetime,
        bucket_seconds: int,
    ) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        start_utc = start.astimezone(UTC)
        end_utc = end.astimezone(UTC)
        epoch_seconds = int(start_utc.timestamp())
        cursor = datetime.fromtimestamp(
            epoch_seconds - (epoch_seconds % bucket_seconds), tz=UTC
        )
        while cursor < end:
            boundary = cursor + timedelta(seconds=bucket_seconds)
            proof_start = max(cursor, start_utc)
            proof_end = min(boundary, end_utc)
            total = 0
            for row in leaves:
                completed = row.get("completed_at")
                tokens = row.get("total_tokens")
                if (
                    isinstance(completed, str)
                    and _is_int(tokens)
                    and proof_start <= _instant(completed) < proof_end
                ):
                    total += tokens
            result.append(
                {
                    "window_start": _utc_text(cursor),
                    "window_end": _utc_text(boundary),
                    "observed_total_tokens": total,
                }
            )
            cursor = boundary
        return result

    def _rolling_dispatch_usage(
        self,
        leaves: list[dict[str, object]],
        *,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for row in leaves:
            dispatched = row.get("dispatch_at")
            if (
                not isinstance(dispatched, str)
                or row.get("actual_provider_dispatch") is not True
            ):
                continue
            at = _instant(dispatched)
            if not start <= at < end:
                continue
            total = 0
            for other in leaves:
                completed = other.get("completed_at")
                tokens = other.get("total_tokens")
                if (
                    isinstance(completed, str)
                    and _is_int(tokens)
                    and at - timedelta(seconds=300) < _instant(completed) <= at
                ):
                    total += tokens
            result.append(
                {
                    "invocation_id": row["invocation_id"],
                    "observed_total_tokens": total,
                }
            )
        return result

    def _zero_call_counts(self, *, start: datetime, end: datetime) -> dict[str, int]:
        connection = self._connection()
        try:
            rows = connection.execute(
                "SELECT decision,COUNT(*) FROM model_zero_call_admissions "
                "WHERE recorded_at>=? AND recorded_at<? GROUP BY decision",
                (_utc_text(start), _utc_text(end)),
            ).fetchall()
        finally:
            connection.close()
        counts = {"HOLD": 0, "REJECT": 0}
        for decision, count in rows:
            counts[str(decision)] = int(count)
        return counts

    def export_csv(self, *, start: datetime, end: datetime) -> str:
        leaves = self.query(start=start, end=end)["leaves"]
        assert isinstance(leaves, list)
        fields = (
            "schema_version",
            "envelope_id",
            "invocation_id",
            "allocation_schema_version",
            "cycle_id",
            "leaf_ordinal",
            "workload_class",
            "admission_decision_id",
            "candidate_id",
            "hypothesis_digest",
            "evidence_package_digest",
            "ingest_id",
            "graphiti_attempt_id",
            "provider_attempt_id",
            "work_outcome_record_id",
            "provider",
            "route",
            "model",
            "reasoning",
            "parent_invocation_id",
            "invocation_policy_digest",
            "prompt_contract_version",
            "prompt_bytes",
            "prompt_digest",
            "request_digest",
            "output_schema_digest",
            "max_output_tokens",
            "context_manifest_digest",
            "allocated_at",
            "dispatch_at",
            "completed_at",
            "work_outcome",
            "invocation_outcome",
            "failure_class",
            "usage_status",
            "terminal_usage_status",
            "reconciliation_usage_status",
            "reconciled_at",
            "conservative_disposition_digest",
            "disposition_usage_status",
            "disposed_at",
            "disposition_authority_digest",
            "disposition_approval_reference",
            "disposition_approved_plan_digest",
            "exact_usage_remains_unknown",
            "provider_dispatch_preserved",
            "unknown_spend_released",
            "input_tokens",
            "output_tokens",
            "cached_read_tokens",
            "cached_write_tokens",
            "reasoning_tokens",
            "context_tokens",
            "total_tokens",
            "provenance",
            "uncertainty",
            "estimate_policy_digest",
            "estimate_calculation",
            "payload_digest",
            "provider_telemetry_digest",
            "raw_telemetry_pointer",
            "od_011_reference",
            "subscription_cli_chat_not_cash_debited",
        )
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in leaves:
            writer.writerow({field: row.get(field) for field in fields})
        return output.getvalue()

    def export_envelope_csv(self, *, start: datetime, end: datetime) -> str:
        envelopes = self.query(start=start, end=end)["envelopes"]
        assert isinstance(envelopes, list)
        fields = (
            "schema_version",
            "envelope_id",
            "cycle_id",
            "workload_class",
            "admitted_at",
            "admission_decision_id",
            "candidate_id",
            "hypothesis_digest",
            "evidence_package_digest",
            "ingest_id",
            "graphiti_attempt_id",
            "canonical_digest",
            "outcome",
            "outcome_record_id",
            "outcome_digest",
            "payload_digest",
            "work_outcome_terminal_at",
            "terminal_at",
            "cycle_outcome",
            "route_circuit_state",
            "route_circuit_reason",
            "retained_proposal_count",
            "accepted_provider_attempt_id",
            "stable_reason_codes",
        )
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in envelopes:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "stable_reason_codes": json.dumps(
                        row.get("stable_reason_codes", []),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
        return output.getvalue()

    def export_bucket_csv(
        self, *, start: datetime, end: datetime, bucket_seconds: int = 300
    ) -> str:
        """Export the canonical fixed-bucket shape used by the 300s incident CSV."""

        data = self.query(start=start, end=end)
        leaves = data["leaves"]
        cycles = data["cycle_outcomes"]
        admissions = data["zero_call_admissions"]
        assert isinstance(leaves, list)
        assert isinstance(cycles, list)
        assert isinstance(admissions, list)
        fields = (
            "bucket_start_utc",
            "bucket_end_utc",
            "cycle_results",
            "minted_reported",
            "graphiti_successes_reported",
            "grok_writer_sessions",
            "grok_completed_sessions",
            "grok_model_calls",
            "grok_input_tokens",
            "grok_output_tokens",
            "grok_total_tokens",
            "grok_cached_read_tokens",
            "grok_reasoning_tokens",
            "cursor_fallback_sessions",
            "stored_outputs",
            "stored_grok_outputs",
            "stored_cursor_outputs",
            "stored_other_outputs",
            "reported_tokens",
            "estimated_tokens",
            "unresolved_invocations",
            "productive_tokens",
            "no_result_tokens",
            "unreported_invocations",
            "ambiguous_invocations",
            "invalid_invocations",
            "admission_only_hold",
            "admission_only_reject",
            "idle_qualified_zero_cycles",
            "productive_cycles",
            "unproductive_provider_cycles",
            "systemic_provider_failure_cycles",
            "cont_reported_tokens",
            "graphiti_reported_tokens",
            "cycle_ids",
            "cycle_outcome_classes",
            "cooldown_seconds_values",
            "next_cycle_eligible_at_values",
        )
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        start_utc = start.astimezone(UTC)
        end_utc = end.astimezone(UTC)
        epoch_seconds = int(start_utc.timestamp())
        cursor = datetime.fromtimestamp(
            epoch_seconds - (epoch_seconds % bucket_seconds), tz=UTC
        )
        while cursor < end_utc:
            boundary = cursor + timedelta(seconds=bucket_seconds)

            def in_bucket(
                value: object,
                bucket_start: datetime = cursor,
                bucket_end: datetime = boundary,
            ) -> bool:
                return (
                    isinstance(value, str)
                    and bucket_start <= _instant(value) < bucket_end
                )

            terminal = [row for row in leaves if in_bucket(row.get("completed_at"))]
            grok = [row for row in terminal if row.get("provider") == "grok-build-cli"]
            cursor_rows = [
                row for row in terminal if row.get("provider") == "cursor-agent-cli"
            ]
            dispatched_grok = [
                row
                for row in leaves
                if row.get("provider") == "grok-build-cli"
                and row.get("actual_provider_dispatch") is True
                and in_bucket(row.get("dispatch_at"))
            ]
            outcomes: dict[str, dict[str, object]] = {}
            for row in leaves:
                if in_bucket(row.get("work_outcome_terminal_at")):
                    outcomes[str(row["envelope_id"])] = row
            accepted = [
                row
                for row in outcomes.values()
                if row.get("work_outcome") == "ACCEPTED"
                and row.get("payload_digest")
            ]
            graphiti_successes = [
                row
                for row in outcomes.values()
                if row.get("work_outcome") in _GRAPHITI_COMPLETED_USEFUL_OUTCOMES
            ]
            accounted = [
                row
                for row in terminal
                if row.get("usage_status") in {"REPORTED", "ESTIMATED"}
            ]
            accepted_providers = Counter(
                str(row.get("provider") or "other")
                for accepted_row in accepted
                for row in leaves
                if row.get("envelope_id") == accepted_row.get("envelope_id")
                and row.get("provider_attempt_id")
                == accepted_row.get("accepted_provider_attempt_id")
            )
            bucket_cycles = [
                row for row in cycles if in_bucket(row.get("terminal_at"))
            ]
            reported = [
                row for row in terminal if row.get("usage_status") == "REPORTED"
            ]

            def token_sum(rows: list[dict[str, object]], field: str) -> int:
                return sum(
                    int(value)
                    for row in rows
                    if _is_int(value := row.get(field))
                    and row.get("usage_status") in {"REPORTED", "ESTIMATED"}
                )

            def json_column(values: list[object]) -> str:
                return json.dumps(
                    values, ensure_ascii=False, separators=(",", ":")
                )

            def productive_as_of_bucket(
                row: dict[str, object], *, bucket_boundary: datetime = boundary
            ) -> bool:
                outcome_at = row.get("work_outcome_terminal_at")
                if (
                    not isinstance(outcome_at, str)
                    or _instant(outcome_at) >= bucket_boundary
                ):
                    return False
                return bool(
                    bool(row.get("accepted_provider_attempt_id"))
                    and row.get("provider_attempt_id")
                    == row.get("accepted_provider_attempt_id")
                    or row.get("work_outcome")
                    in _GRAPHITI_COMPLETED_USEFUL_OUTCOMES
                )

            writer.writerow(
                {
                    "bucket_start_utc": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "bucket_end_utc": boundary.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "cycle_results": sum(
                        in_bucket(row.get("terminal_at")) for row in cycles
                    ),
                    "minted_reported": len(accepted),
                    "graphiti_successes_reported": len(graphiti_successes),
                    "grok_writer_sessions": sum(
                        str(row.get("workload_class", "")).startswith(
                            "CONT_WRITER_"
                        )
                        for row in grok
                    ),
                    "grok_completed_sessions": len(grok),
                    "grok_model_calls": len(dispatched_grok),
                    "grok_input_tokens": token_sum(grok, "input_tokens"),
                    "grok_output_tokens": token_sum(grok, "output_tokens"),
                    "grok_total_tokens": token_sum(grok, "total_tokens"),
                    "grok_cached_read_tokens": token_sum(
                        grok, "cached_read_tokens"
                    ),
                    "grok_reasoning_tokens": token_sum(grok, "reasoning_tokens"),
                    "cursor_fallback_sessions": len(cursor_rows),
                    "stored_outputs": len(accepted),
                    "stored_grok_outputs": accepted_providers["grok-build-cli"],
                    "stored_cursor_outputs": accepted_providers["cursor-agent-cli"],
                    "stored_other_outputs": len(accepted)
                    - accepted_providers["grok-build-cli"]
                    - accepted_providers["cursor-agent-cli"],
                    "reported_tokens": token_sum(
                        [
                            row
                            for row in terminal
                            if row.get("usage_status") == "REPORTED"
                        ],
                        "total_tokens",
                    ),
                    "estimated_tokens": token_sum(
                        [
                            row
                            for row in terminal
                            if row.get("usage_status") == "ESTIMATED"
                        ],
                        "total_tokens",
                    ),
                    "unresolved_invocations": sum(
                        row.get("usage_status")
                        in {"UNREPORTED", "AMBIGUOUS", "INVALID"}
                        for row in terminal
                    ),
                    "productive_tokens": sum(
                        int(row["total_tokens"])
                        for row in accounted
                        if _is_int(row.get("total_tokens"))
                        and productive_as_of_bucket(row)
                    ),
                    "no_result_tokens": sum(
                        int(row["total_tokens"])
                        for row in accounted
                        if _is_int(row.get("total_tokens"))
                        and not productive_as_of_bucket(row)
                    ),
                    "unreported_invocations": sum(
                        row.get("usage_status") == "UNREPORTED" for row in terminal
                    ),
                    "ambiguous_invocations": sum(
                        row.get("usage_status") == "AMBIGUOUS" for row in terminal
                    ),
                    "invalid_invocations": sum(
                        row.get("usage_status") == "INVALID" for row in terminal
                    ),
                    "admission_only_hold": sum(
                        row.get("decision") == "HOLD"
                        and in_bucket(row.get("recorded_at"))
                        for row in admissions
                    ),
                    "admission_only_reject": sum(
                        row.get("decision") == "REJECT"
                        and in_bucket(row.get("recorded_at"))
                        for row in admissions
                    ),
                    "idle_qualified_zero_cycles": sum(
                        row.get("outcome_class") == "IDLE_QUALIFIED_ZERO"
                        for row in bucket_cycles
                    ),
                    "productive_cycles": sum(
                        row.get("outcome_class") == "PRODUCTIVE"
                        for row in bucket_cycles
                    ),
                    "unproductive_provider_cycles": sum(
                        row.get("outcome_class") == "UNPRODUCTIVE_PROVIDER"
                        for row in bucket_cycles
                    ),
                    "systemic_provider_failure_cycles": sum(
                        row.get("outcome_class") == "SYSTEMIC_PROVIDER_FAILURE"
                        for row in bucket_cycles
                    ),
                    "cont_reported_tokens": token_sum(
                        [
                            row
                            for row in reported
                            if str(row.get("workload_class") or "").startswith(
                                "CONT_"
                            )
                        ],
                        "total_tokens",
                    ),
                    "graphiti_reported_tokens": token_sum(
                        [
                            row
                            for row in reported
                            if str(row.get("workload_class") or "").startswith(
                                "GRAPHITI_"
                            )
                        ],
                        "total_tokens",
                    ),
                    "cycle_ids": json_column(
                        [row.get("cycle_id") for row in bucket_cycles]
                    ),
                    "cycle_outcome_classes": json_column(
                        [row.get("outcome_class") for row in bucket_cycles]
                    ),
                    "cooldown_seconds_values": json_column(
                        [row.get("cooldown_seconds") for row in bucket_cycles]
                    ),
                    "next_cycle_eligible_at_values": json_column(
                        [
                            row.get("next_cycle_eligible_at")
                            for row in bucket_cycles
                        ]
                    ),
                }
            )
            cursor = boundary
        return output.getvalue()

    def _insert_exact(
        self,
        *,
        table: str,
        identity_column: str,
        identity: str,
        record: Mapping[str, object],
        sql: str,
        values: tuple[object, ...],
        connection: sqlite3.Connection | None = None,
    ) -> None:
        owns_connection = connection is None
        current = self._connection() if connection is None else connection
        try:
            try:
                current.execute(sql, values)
            except sqlite3.IntegrityError as exc:
                row = current.execute(
                    f"SELECT record_json FROM {table} WHERE {identity_column}=?",
                    (identity,),
                ).fetchone()
                if row is None or _object(row[0]) != dict(record):
                    raise ModelUsageIntegrityError(
                        f"conflicting {table} replay"
                    ) from exc
            if owns_connection:
                current.commit()
        finally:
            if owns_connection:
                current.close()


def _json(value: Mapping[str, object]) -> str:
    return canonical_json_bytes(dict(value)).decode("utf-8")


def _object(value: object) -> dict[str, object]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ModelUsageIntegrityError(
            "retained model usage JSON is malformed"
        ) from exc
    if not isinstance(parsed, dict):
        raise ModelUsageIntegrityError("retained model usage JSON is not an object")
    return parsed


def _policy_from_record(record: Mapping[str, object]) -> InvocationEfficiencyPolicy:
    policy = InvocationEfficiencyPolicy(
        policy_id=str(record["policy_id"]),
        version=str(record["version"]),
        workload_class=WorkloadClass(str(record["workload_class"])),
        provider=str(record["provider"]),
        route=str(record["route"]),
        model=str(record["model"]),
        reasoning=str(record["reasoning"]),
        one_turn=bool(record["one_turn"]),
        exact_input=bool(record["exact_input"]),
        skills_enabled=bool(record["skills_enabled"]),
        tools_enabled=bool(record["tools_enabled"]),
        mcp_enabled=bool(record["mcp_enabled"]),
        prior_message_count=_record_int(record, "prior_message_count"),
        command_semantic_version=str(
            record.get("command_semantic_version", "UNSPECIFIED")
        ),
        command_flags=_record_string_tuple(record, "command_flags", default=()),
        context_manifest_schema_version=str(
            record.get("context_manifest_schema_version", "UNSPECIFIED")
        ),
        disabled_capabilities=_record_string_tuple(
            record, "disabled_capabilities", default=()
        ),
        implementation_revision=str(
            record.get("implementation_revision", "UNSPECIFIED")
        ),
        calibration_only=bool(record.get("calibration_only", False)),
        allowed_candidate_ids=_record_string_tuple(
            record, "allowed_candidate_ids", default=()
        ),
        max_prompt_bytes=_record_int(record, "max_prompt_bytes"),
        max_context_tokens=_record_int(record, "max_context_tokens"),
        max_output_tokens=(None if "max_output_tokens" in record
                           and record["max_output_tokens"] is None
                           else _record_int(record, "max_output_tokens")),
        max_total_tokens=_record_int(record, "max_total_tokens"),
        prompt_contract_version=str(record["prompt_contract_version"]),
        output_schema_digest=str(record["output_schema_digest"]),
        allowed_context_identities=tuple(
            str(value)
            for value in record["allowed_context_identities"]  # type: ignore[union-attr]
        ),
        allowed_config_identities=tuple(
            str(value)
            for value in record["allowed_config_identities"]  # type: ignore[union-attr]
        ),
        hard_estimate_ceiling_tokens=(
            None
            if record.get("hard_estimate_ceiling_tokens") is None
            else _record_int(record, "hard_estimate_ceiling_tokens")
        ),
        evidence_digest=str(record["evidence_digest"]),
        qualified=bool(record["qualified"]),
        canonical_digest=str(record["canonical_digest"]),
    )
    policy._validate()
    return policy


def _envelope_from_record(record: Mapping[str, object]) -> WorkEnvelope:
    """Decode and re-derive one retained work-envelope identity."""

    envelope = WorkEnvelope.create(
        cycle_id=str(record["cycle_id"]),
        workload_class=WorkloadClass(str(record["workload_class"])),
        admitted_at=_instant(str(record["admitted_at"])),
        admission_decision_id=record.get("admission_decision_id"),
        candidate_id=record.get("candidate_id"),
        hypothesis_digest=record.get("hypothesis_digest"),
        evidence_package_digest=record.get("evidence_package_digest"),
        ingest_id=record.get("ingest_id"),
        graphiti_attempt_id=record.get("graphiti_attempt_id"),
    )
    if (
        record.get("envelope_id") != envelope.envelope_id
        or record.get("canonical_digest") != envelope.canonical_digest
        or dict(record) != envelope.as_record()
    ):
        raise ModelUsageIntegrityError("retained work envelope identity differs")
    return envelope


def _allocation_from_record(record: Mapping[str, object]) -> InvocationAllocation:
    """Decode and re-derive one retained invocation-allocation identity."""

    allocation = InvocationAllocation.create(
        envelope_id=str(record["envelope_id"]),
        cycle_id=str(record["cycle_id"]),
        leaf_ordinal=_record_int(record, "leaf_ordinal"),
        workload_class=WorkloadClass(str(record["workload_class"])),
        invocation_policy_digest=str(record["invocation_policy_digest"]),
        provider=str(record["provider"]),
        route=str(record["route"]),
        model=str(record["model"]),
        reasoning=str(record["reasoning"]),
        prompt_contract_version=str(record["prompt_contract_version"]),
        prompt_bytes=_record_int(record, "prompt_bytes"),
        prompt_digest=str(record["prompt_digest"]),
        request_digest=str(record["request_digest"]),
        output_schema_digest=str(record["output_schema_digest"]),
        max_output_tokens=(None if "max_output_tokens" in record
                           and record["max_output_tokens"] is None
                           else _record_int(record, "max_output_tokens")),
        context_manifest_digest=str(record["context_manifest_digest"]),
        context_identity=str(record["context_identity"]),
        config_identity=str(record["config_identity"]),
        one_turn=record["one_turn"],
        exact_input=record["exact_input"],
        skills_enabled=record["skills_enabled"],
        tools_enabled=record["tools_enabled"],
        mcp_enabled=record["mcp_enabled"],
        prior_message_count=_record_int(record, "prior_message_count"),
        allocated_at=_instant(str(record["allocated_at"])),
        recovery_deadline_at=_instant(str(record["recovery_deadline_at"])),
        parent_invocation_id=record.get("parent_invocation_id"),
    )
    if (
        record.get("invocation_id") != allocation.invocation_id
        or record.get("canonical_digest") != allocation.canonical_digest
        or dict(record) != allocation.as_record()
    ):
        raise ModelUsageIntegrityError("retained invocation allocation identity differs")
    return allocation


def _terminal_from_record(record: Mapping[str, object]) -> InvocationTerminal:
    """Decode and re-derive one retained invocation-terminal identity."""

    raw_components = record.get("components")
    if not isinstance(raw_components, Mapping):
        raise ModelUsageIntegrityError("retained invocation terminal is malformed")
    terminal = InvocationTerminal.create(
        invocation_id=str(record["invocation_id"]),
        outcome=str(record["outcome"]),
        failure_class=record.get("failure_class"),
        usage_status=UsageStatus(str(record["usage_status"])),
        components=UsageComponents(
            input_tokens=raw_components.get("input_tokens"),
            output_tokens=raw_components.get("output_tokens"),
            cached_read_tokens=raw_components.get("cached_read_tokens"),
            cached_write_tokens=raw_components.get("cached_write_tokens"),
            reasoning_tokens=raw_components.get("reasoning_tokens"),
            context_tokens=raw_components.get("context_tokens"),
            total_tokens=raw_components.get("total_tokens"),
            provenance=str(raw_components["provenance"]),
        ),
        dispatch_at=(
            None
            if record.get("dispatch_at") is None
            else _instant(str(record["dispatch_at"]))
        ),
        completed_at=_instant(str(record["completed_at"])),
        observed_at=_instant(str(record["observed_at"])),
        provider_telemetry_digest=record.get("provider_telemetry_digest"),
        raw_telemetry_pointer=record.get("raw_telemetry_pointer"),
        estimate_policy_digest=record.get("estimate_policy_digest"),
        estimate_calculation=record.get("estimate_calculation"),
        pre_dispatch_zero_proved=record.get("pre_dispatch_zero_proved"),
        od_011_reference=record.get("od_011_reference"),
        subscription_cli_chat_not_cash_debited=record[
            "subscription_cli_chat_not_cash_debited"
        ],
        policy_breach=record.get("policy_breach"),
    )
    if (
        record.get("terminal_digest") != terminal.terminal_digest
        or dict(record) != terminal.as_record()
    ):
        raise ModelUsageIntegrityError("retained invocation terminal identity differs")
    return terminal


def _record_int(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if not _is_int(value):
        raise ModelUsageIntegrityError(f"retained policy {field} is invalid")
    return value


def _is_hermetic_cont_policy(policy: InvocationEfficiencyPolicy) -> bool:
    return bool(
        _HERMETIC_CONT_CONFIG_IDENTITIES.intersection(
            policy.allowed_config_identities
        )
    )


def _record_string_tuple(
    record: Mapping[str, object],
    field: str,
    *,
    default: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    value = record.get(field, default)
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise ModelUsageIntegrityError(f"retained policy {field} is invalid")
    return tuple(value)


__all__ = [
    "MODEL_USAGE_INTERFACE_SCHEMA_VERSION",
    "MODEL_USAGE_SCHEMA_VERSION",
    "InvocationAllocation",
    "InvocationEfficiencyPolicy",
    "InvocationTerminal",
    "ModelUsageAdmissionError",
    "ModelUsageIntegrityError",
    "ModelUsageService",
    "UsageComponents",
    "UsageStatus",
    "WorkEnvelope",
    "WorkloadClass",
]
