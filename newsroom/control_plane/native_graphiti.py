"""Native revisions through the existing accounted Graphiti/admission path.

EVALUATION here is the isolated untrusted proposal workspace. Serving authority
still comes only from the existing entity/relation decisions and Increment 4
projection. This worker neither resumes a historical campaign nor mints READY.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Callable, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ContextManager

from newsroom.authority import AuthenticationProof
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical, validate_sha256_digest
from newsroom.authority.hermes_native_system import HermesNativeAuthoritySystem
from newsroom.extraction.types import ExtractionRunId
from newsroom.graphiti_adapter.identity import typed_id
from newsroom.graphiti_adapter.evaluation_packet import GRAPHITI_EXTRACTION_TIMEOUT_MS
from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome, GraphitiAdapterRightsDenied
from newsroom.increment4.contracts import INCREMENT4_ADMITTED_FAMILY_ID
from newsroom.increment4.neo4j import Increment4Neo4jCurrentBuildRequest
from newsroom.projection.models import ProjectionGenerationId, ProjectionGenerationState

from .corpus import CorpusIngestUnit
from .cycle import _DispatchAuthority, _ingest
from .graphiti import EvaluationGraphitiRunner, graphiti_required_route_holds
from .graphiti_fallback_policy import (
    load_checked_native_graphiti_fallback_circuit_policy,
)
from .graphiti_requests import load_checked_native_graphiti_call_shape_policy
from .graphiti_admission import GraphitiAdmissionConsumerError
from .graphiti_admission_integration import compose_existing_graphiti_admission_consumer
from .model_usage import ModelUsageAdmissionError, ModelUsageService, reported_output_rejected_ingests
from .native_cycle import _uuid4_for
from .store import append_ledger, graphiti_failure_state
from .veto import OperatorDrainRequested, VetoError


@dataclass(frozen=True, slots=True)
class NativeGraphitiOutcome:
    ingest_id: str
    state: str
    receipt_digest: str | None
    reason: str | None


class NativeGraphitiProcessor:
    """One bounded native cohort; durable retries/accounting use existing stores."""

    def __init__(
        self, *, system: HermesNativeAuthoritySystem,
        connection: sqlite3.Connection, usage: ModelUsageService,
        proof: AuthenticationProof,
        rights_for: Callable[[CorpusIngestUnit], Mapping[str, object] | None],
        stop_check: Callable[[], None],
        dispatch_fence: Callable[[], ContextManager[None]],
        operator_drain_requested: Callable[[], bool] = lambda: False,
        clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
    ) -> None:
        self._system, self._connection, self._usage = system, connection, usage
        self._proof, self._rights_for = proof, rights_for
        self._stop_check, self._dispatch_fence, self._clock = stop_check, dispatch_fence, clock
        self._operator_drain_requested = operator_drain_requested
        self._cohorts: dict[str, tuple[str, ...]] = {}
        self._completed: set[str] = set()
        for raw, digest in connection.execute(
            "SELECT payload_json,payload_digest FROM ledger WHERE kind='NATIVE_GRAPHITI_COHORT' ORDER BY seq"
        ):
            if digest_bytes(raw.encode()) != digest:
                raise ValueError("native Graphiti cohort ledger differs")
            value = json.loads(raw)
            exact = tuple(value["ingest_ids"])
            if tuple(sorted(set(exact))) != exact or value["cohort_id"] != digest_canonical(exact):
                raise ValueError("native Graphiti cohort identity differs")
            self._cohorts[value["cohort_id"]] = exact
            if value["state"] == "COMPLETE":
                self._completed.add(value["cohort_id"])
            elif value["state"] != "STARTED":
                raise ValueError("native Graphiti cohort state differs")
        self._runner = EvaluationGraphitiRunner(
            clock=clock, fallback_permitted=True,
            governed_fallback_permitted=True,
            proposal_adapter=system.graphiti, extraction_records=system.extraction,
            proof=proof,
            call_shape_policy=load_checked_native_graphiti_call_shape_policy(),
            fallback_policy=load_checked_native_graphiti_fallback_circuit_policy(),
        )
        self._admission = compose_existing_graphiti_admission_consumer(
            connection, adapter=system.graphiti, extraction=system.extraction,
            objects=system.objects, entities=system.entities, relations=system.relations,
            increment4=system.increment4, proof=proof,
        )

    def _rights(self, unit: CorpusIngestUnit) -> dict[str, object] | None:
        self._stop_check()
        value = self._rights_for(unit)
        return None if value is None else dict(value)

    @contextmanager
    def _fence(self, unit: CorpusIngestUnit):
        with self._dispatch_fence():
            rights = self._rights(unit)
            active = threading.Event()
            fence_pid = os.getpid()
            active.set()

            def require_active_fence() -> None:
                if os.getpid() != fence_pid:
                    raise VetoError("owner emergency stop fence belongs to another process")
                if not active.is_set():
                    raise VetoError("owner emergency stop fence has expired")

            try:
                yield None if rights is None else _DispatchAuthority(
                    rights, self._clock().astimezone(UTC) + timedelta(minutes=15),
                    require_active_fence,
                )
            finally:
                active.clear()

    def advance(
        self, units: tuple[CorpusIngestUnit, ...], *, cycle_id: str,
        defer_before_unit: Callable[[CorpusIngestUnit], bool] = lambda _: False,
    ) -> tuple[NativeGraphitiOutcome, ...]:
        if not units:
            return ()
        # Deriving an ingest identity hashes the full source body. Keep the
        # exact keys for this operation only, including unchanged terminal holds.
        units_by_ingest = {unit.ingest_id: unit for unit in units}
        if len(units_by_ingest) != len(units):
            raise ValueError("native Graphiti cohort repeats an ingest")
        for unit in units:
            if type(unit) is not CorpusIngestUnit or unit.authority is None:
                raise ValueError("native Graphiti requires retained source authority")
            if not unit.proving_run_id.startswith("native-source:"):
                raise ValueError("historical campaign input is outside native dispatch")
            validate_sha256_digest(unit.proving_run_id.removeprefix("native-source:"))
            if unit.proving_run_id != "native-source:" + unit.observation_digest:
                raise ValueError("native Graphiti observation provenance differs")
        revisions: dict[str, list[str]] = {}
        for ingest_id, unit in units_by_ingest.items():
            revisions.setdefault(unit.revision_id, []).append(ingest_id)
        for members in revisions.values():
            members.sort(key=lambda ingest_id: units_by_ingest[ingest_id].chunk_ordinal)
            chunk_count = units_by_ingest[members[0]].chunk_count
            if (tuple(units_by_ingest[ingest_id].chunk_ordinal for ingest_id in members) != tuple(range(1, chunk_count + 1))
                    or any(units_by_ingest[ingest_id].chunk_count != chunk_count for ingest_id in members)):
                raise ValueError("native Graphiti revision chunk coverage differs")
        self._stop_check()
        recover_owned = getattr(self._runner, "recover_owned_pending", None)
        if callable(recover_owned):
            recover_owned(
                connection=self._connection,
                owner_stop_check=self._stop_check,
                rights_fence=self._fence,
                defer_before_unit=defer_before_unit,
                deadline=self._clock() + timedelta(milliseconds=GRAPHITI_EXTRACTION_TIMEOUT_MS),
            )
        # Resolve retained accounting before considering another provider call.
        self._settle_missing_subscription_usage(units_by_ingest)
        terminal_holds = {
            ingest_id: "REPORTED_OUTPUT_REJECTION_NO_RETRY"
            for ingest_id in reported_output_rejected_ingests(self._connection, ingest_ids=tuple(units_by_ingest))
            if ingest_id in units_by_ingest
        }
        recovered_ambiguous_attempts: dict[str, int] = {}
        authenticated_rejected_attempts: dict[str, tuple[int, ...]] = {}
        authenticated_reentry_attempts: dict[str, int] = {}
        for ingest_id in units_by_ingest:
            if ingest_id in terminal_holds:
                continue
            # The authority commits before the private receipt/failure journal.
            # Inspect it even if a crash left no local failure row.
            if self._connection.execute(
                "SELECT 1 FROM unpublished_graphiti_ingest WHERE ingest_id=? AND outcome='COMPLETE'",
                (ingest_id,),
            ).fetchone() is not None:
                continue
            try:
                history = self._system.graphiti.attempt_history(
                    typed_id(ExtractionRunId, "run", ingest_id),
                    limit=2, proof=self._proof,
                )
            except GraphitiAdapterRightsDenied:
                terminal_holds[ingest_id] = "CURRENT_SOURCE_RIGHTS_HOLD"
                continue
            if history:
                head = history[0]
                next_attempt = self._connection.execute(
                    "SELECT COALESCE(MAX(attempt_number),0)+1 FROM "
                    "unpublished_graphiti_attempt_receipts WHERE ingest_id=?",
                    (ingest_id,),
                ).fetchone()[0]
                authenticate_retained = getattr(
                    self._runner,
                    "authenticate_retained_recovered_ambiguous_progression",
                    None,
                )
                skipped = (
                    authenticate_retained(
                        unit=units_by_ingest[ingest_id],
                        current_attempt=head,
                        authoritative_attempt=history[1],
                        next_attempt_number=int(next_attempt),
                        connection=self._connection,
                        model_usage=self._usage,
                    )
                    if len(history) == 2 and callable(authenticate_retained)
                    else None
                )
                if skipped is not None:
                    authenticated_rejected_attempts[ingest_id] = (int(skipped),)
                    if int(next_attempt) == head.attempt_number:
                        authenticated_reentry_attempts[ingest_id] = int(next_attempt)
                    continue
                if head.outcome.terminal:
                    prepare_recovery = getattr(
                        self._runner,
                        "prepare_recovered_ambiguous_successor",
                        None,
                    )
                    if (
                        head.outcome is GraphitiAdapterOutcome.AMBIGUOUS_EFFECT
                        and callable(prepare_recovery)
                        and prepare_recovery(
                            unit=units_by_ingest[ingest_id],
                            authoritative_attempt=head,
                            private_successor_attempt_number=int(next_attempt),
                            connection=self._connection,
                            model_usage=self._usage,
                        )
                    ):
                        recovered_ambiguous_attempts[ingest_id] = int(next_attempt)
                        authenticated_rejected_attempts[ingest_id] = (
                            head.attempt_number + 1,
                        )
                        continue
                    # A settled terminal adapter result is not a new retryable
                    # provider failure. Keep the original cause and its accounting.
                    terminal_holds[ingest_id] = (
                        "RETAINED_COMPLETE_RECONCILIATION_REQUIRED"
                        if head.outcome is GraphitiAdapterOutcome.COMPLETE else
                        f"{head.outcome.value}:{head.failure_code}"
                    )
                    continue
        deferred = set()
        considered = set()

        def defer(unit: CorpusIngestUnit) -> bool:
            considered.add(unit.ingest_id)
            self._stop_check()
            if defer_before_unit(unit):
                deferred.add(unit.ingest_id)
                return True
            return False

        remaining = {
            ingest_id: unit for ingest_id, unit in units_by_ingest.items()
            if ingest_id not in terminal_holds
        }
        # Refresh only after settled progress. Each queued identity is offered
        # at most once per advance; failed/deferred work is not retried here.
        # The existing defer callback retains the pipeline's wall-time quantum.
        while remaining:
            considered.clear()
            _ingest(
                self._connection, graphiti=self._runner,
                units=tuple(remaining.values()),
                max_graphiti=len(remaining), rights_check=self._rights,
                rights_fence=self._fence, clock=self._clock,
                model_usage=self._usage, cycle_id=cycle_id,
                operator_drain_requested=self._operator_drain_requested,
                defer_before_unit=defer,
                preserve_unit_order=True,
                fallback_permitted=True,
                recovered_ambiguous_attempts=recovered_ambiguous_attempts,
                authenticated_rejected_attempts=authenticated_rejected_attempts,
                authenticated_reentry_attempts=authenticated_reentry_attempts,
            )
            for ingest_id in considered:
                remaining.pop(ingest_id, None)
            if (
                not considered or self._operator_drain_requested()
                or graphiti_required_route_holds(self._usage, fallback_permitted=True)
                or not any(
                    self._connection.execute(
                        "SELECT 1 FROM unpublished_graphiti_ingest WHERE ingest_id=? AND outcome='COMPLETE'",
                        (ingest_id,),
                    ).fetchone() is not None
                    for ingest_id in considered
                )
            ):
                break
        self._settle_missing_subscription_usage(units)
        terminal_holds.update({
            ingest_id: "REPORTED_OUTPUT_REJECTION_NO_RETRY"
            for ingest_id in reported_output_rejected_ingests(self._connection, ingest_ids=tuple(units_by_ingest))
            if ingest_id in units_by_ingest
        })
        if self._operator_drain_requested():
            raise OperatorDrainRequested
        route_held = bool(
            graphiti_required_route_holds(
                self._usage, fallback_permitted=True
            )
        )
        outcomes = []
        complete = []
        for ingest_id in units_by_ingest:
            row = self._connection.execute(
                "SELECT outcome,receipt_digest FROM unpublished_graphiti_ingest "
                "WHERE ingest_id=?", (ingest_id,),
            ).fetchone()
            if row is not None and row[0] == "COMPLETE":
                complete.append(ingest_id)
                outcomes.append(NativeGraphitiOutcome(ingest_id, "EXTRACTION_COMPLETE", str(row[1]), None))
            elif ingest_id in deferred:
                outcomes.append(NativeGraphitiOutcome(
                    ingest_id, "GRAPHITI_DEFERRED", None,
                    "WORK_QUANTUM_EXHAUSTED",
                ))
            else:
                failures, dead = graphiti_failure_state(self._connection, ingest_id)
                reason = (
                    terminal_holds[ingest_id] if ingest_id in terminal_holds else
                    "REQUIRED_MODEL_ROUTE_CIRCUIT_OPEN" if route_held else
                    "PARTIAL_EXTRACTION" if row is not None else
                    "DEAD_LETTER" if dead else "RETRY_PENDING" if failures else
                    "RIGHTS_OR_PREDECESSOR_HOLD"
                )
                outcomes.append(NativeGraphitiOutcome(ingest_id, "GRAPHITI_HOLD", None, reason))
        # Never project a prefix of a multi-chunk revision. An independently
        # complete revision may still advance while another revision is held.
        complete_set = set(complete)
        eligible = set()
        for members in revisions.values():
            if all(ingest_id in complete_set for ingest_id in members):
                eligible.update(members)
        statuses = {item.ingest_id: item for item in outcomes}
        assigned = {ingest for exact in self._cohorts.values() for ingest in exact}
        for members in revisions.values():
            exact = tuple(sorted(members))
            if set(exact) <= eligible and set(exact).isdisjoint(assigned):
                # Durable admission units follow revision atomicity. Projection
                # still batches every ready unit once below.
                self._cohort_state(exact, "STARTED")
                assigned.update(exact)
        pending = []
        for cohort_id, exact in tuple(self._cohorts.items()):
            selected = set(exact) & eligible
            if not selected:
                continue
            if cohort_id in self._completed:
                for ingest in selected:
                    statuses[ingest] = NativeGraphitiOutcome(ingest, "GRAPHITI_COMPLETE", statuses[ingest].receipt_digest, None)
                continue
            if set(exact) != selected:
                for ingest in selected:
                    statuses[ingest] = NativeGraphitiOutcome(ingest, "ADMISSION_HOLD", statuses[ingest].receipt_digest, "COHORT_CONTINUATION_PENDING")
                continue
            pending.append((cohort_id, exact))
        if pending:
            if self._operator_drain_requested():
                raise OperatorDrainRequested
            self._stop_check()
            admission_ready = []
            queued = {}
            for cohort_id, exact in pending:
                try:
                    self._admission.enqueue_complete_receipts(
                        ingest_ids=exact
                    )
                except GraphitiAdmissionConsumerError as exc:
                    for ingest in exact:
                        statuses[ingest] = NativeGraphitiOutcome(
                            ingest,
                            "ADMISSION_HOLD",
                            statuses[ingest].receipt_digest,
                            str(exc),
                        )
                else:
                    admission_ready.append((cohort_id, exact))
                    queued[cohort_id] = self._connection.execute(
                        "SELECT count(*) FROM unpublished_graphiti_admission_queue "
                        "WHERE ingest_id IN ("
                        + ",".join("?" for _ in exact)
                        + ")",
                        exact,
                    ).fetchone()[0]
            combined = tuple(
                sorted(
                    ingest
                    for _, exact in admission_ready
                    for ingest in exact
                )
            )
            if not combined:
                return tuple(statuses[ingest_id] for ingest_id in units_by_ingest)
            self._admission.drain(
                worker_id=cycle_id,
                limit=max(1, sum(queued.values())),
                ingest_ids=combined,
            )
            if self._operator_drain_requested():
                raise OperatorDrainRequested
            ready = []
            with self._dispatch_fence():
                for cohort_id, exact in admission_ready:
                    if all(
                        self._rights(units_by_ingest[ingest]) is not None
                        for ingest in exact
                    ):
                        ready.append((cohort_id, exact))
                    else:
                        for ingest in exact:
                            statuses[ingest] = NativeGraphitiOutcome(
                                ingest,
                                "ADMISSION_HOLD",
                                statuses[ingest].receipt_digest,
                                "CURRENT_RIGHTS_HOLD",
                            )
                verified = []
                for cohort_id, exact in ready:
                    try:
                        self._admission.preflight_decided_cohort(
                            ingest_ids=exact
                        )
                    except GraphitiAdmissionConsumerError as exc:
                        for ingest in exact:
                            statuses[ingest] = NativeGraphitiOutcome(
                                ingest,
                                "ADMISSION_HOLD",
                                statuses[ingest].receipt_digest,
                                str(exc),
                            )
                    else:
                        verified.append((cohort_id, exact))
                ready = verified
                if ready:
                    final_ids = tuple(
                        sorted(ingest for _, exact in ready for ingest in exact)
                    )
                    try:
                        self._admission.finalise_decided_cohort(
                            ingest_ids=final_ids
                        )
                        if (
                            not sum(queued[cohort_id] for cohort_id, _ in ready)
                            and not self._has_active_generation()
                        ):
                            # Zero proposals need no new entity/relation graph,
                            # but a fresh native pipeline needs its first graph
                            # for downstream retrieval. Retain the real build.
                            frontier = digest_canonical(final_ids)
                            generation_id = ProjectionGenerationId.parse(
                                _uuid4_for({"native_zero_proposal_cohort": frontier})
                            )
                            built = self._system.increment4.build_current_and_promote(
                                Increment4Neo4jCurrentBuildRequest(
                                    generation_id,
                                    "NATIVE_ZERO_PROPOSAL_COHORT",
                                    f"native-empty-cohort:{frontier}",
                                ),
                                proof=self._proof,
                            )
                            if (
                                built.generation.state
                                is not ProjectionGenerationState.ACTIVE
                            ):
                                raise GraphitiAdmissionConsumerError(
                                    "native empty-cohort graph is not active"
                                )
                    except GraphitiAdmissionConsumerError as exc:
                        for _, exact in ready:
                            for ingest in exact:
                                statuses[ingest] = NativeGraphitiOutcome(
                                    ingest,
                                    "ADMISSION_HOLD",
                                    statuses[ingest].receipt_digest,
                                    str(exc),
                                )
                        ready = []
            for _, exact in ready:
                self._cohort_state(exact, "COMPLETE")
                for ingest in exact:
                    statuses[ingest] = NativeGraphitiOutcome(
                        ingest,
                        "GRAPHITI_COMPLETE",
                        statuses[ingest].receipt_digest,
                        None,
                    )
            if self._operator_drain_requested():
                raise OperatorDrainRequested
        return tuple(statuses[ingest_id] for ingest_id in units_by_ingest)

    def _has_active_generation(self) -> bool:
        # Select metadata only to decide whether first bootstrap is needed.
        # This is not graph proof: subsequent retrieval still validates its read.
        uri = Path(self._system.authority_store_path).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.execute("PRAGMA query_only=ON")
            rows = connection.execute(
                "SELECT generation_id FROM projection_generations "
                "WHERE family_id=? AND state='ACTIVE' ORDER BY generation_id LIMIT 2",
                (INCREMENT4_ADMITTED_FAMILY_ID,),
            ).fetchall()
        if not rows:
            return False
        if len(rows) != 1:
            raise GraphitiAdmissionConsumerError("native active graph metadata differs")
        generation_id = ProjectionGenerationId.parse(str(rows[0][0]))
        status = self._system.increment4.generation_status(
            generation_id, proof=self._proof,
        )
        if (
            status.generation.generation_id != generation_id
            or status.generation.state is not ProjectionGenerationState.ACTIVE
        ):
            raise GraphitiAdmissionConsumerError("native active graph metadata differs")
        return True

    def _settle_missing_subscription_usage(
        self, units: tuple[CorpusIngestUnit, ...] | Mapping[str, CorpusIngestUnit],
    ) -> None:
        if self._usage is None:
            return
        # Reuse exact keys only before dispatch. After the external-call boundary,
        # advance supplies the units again to derive current identities afresh.
        units_by_ingest = (
            units if isinstance(units, Mapping) else
            {unit.ingest_id: unit for unit in units}
        )
        for ingest_id in sorted(units_by_ingest):
            output_rejections = self._connection.execute(
                "SELECT a.invocation_id,a.canonical_digest,t.terminal_digest "
                "FROM model_work_envelopes e INDEXED BY model_usage_native_graphiti_ingest "
                "JOIN model_invocation_allocations a ON a.envelope_id=e.envelope_id "
                "JOIN model_invocation_terminals t ON t.invocation_id=a.invocation_id "
                "WHERE e.workload_class='GRAPHITI_CHAT_PRIMARY' "
                "AND json_extract(e.record_json,'$.ingest_id')=? "
                "AND a.route='GRAPHITI_CHAT_PRIMARY' AND a.provider='cursor-agent-cli' "
                "AND t.outcome='OUTPUT_LIMIT_EXCEEDED' AND t.usage_status='REPORTED' "
                "AND t.failure_class IS NULL "
                "AND EXISTS (SELECT 1 FROM model_work_outcomes o WHERE o.envelope_id=e.envelope_id "
                "AND o.outcome='GRAPHITI_FAILED') "
                "AND json_extract(t.record_json,'$.policy_breach')='REQUESTED_MAX_OUTPUT_TOKENS_EXCEEDED' "
                "AND (NOT EXISTS (SELECT 1 FROM model_usage_reported_output_dispositions d "
                "WHERE d.invocation_id=a.invocation_id) OR a.invocation_id=("
                "SELECT CASE WHEN state='OPEN' THEN invocation_id END "
                "FROM model_usage_route_circuit_events WHERE route=a.route "
                "ORDER BY recorded_at DESC,rowid DESC LIMIT 1))",
                (ingest_id,),
            ).fetchall()
            for invocation_id, allocation_digest, terminal_digest in output_rejections:
                self._stop_check()
                try:
                    self._usage.disposition_native_reported_output_rejection(
                        invocation_id=invocation_id, revision_id=units_by_ingest[ingest_id].revision_id,
                        expected_terminal_digest=terminal_digest, expected_allocation_digest=allocation_digest,
                        observed_at=self._clock(),
                    )
                except ModelUsageAdmissionError:
                    # Ineligible/partially settled work retains its global HOLD.
                    # Contradictory retained proof still raises IntegrityError.
                    continue
            rows = self._connection.execute(
                "SELECT a.invocation_id,a.canonical_digest,t.terminal_digest,e.record_json "
                "FROM model_work_envelopes e INDEXED BY model_usage_native_graphiti_ingest "
                "JOIN model_invocation_allocations a ON a.envelope_id=e.envelope_id "
                "JOIN model_invocation_terminals t ON t.invocation_id=a.invocation_id "
                "WHERE e.workload_class='GRAPHITI_CHAT_PRIMARY' "
                "AND json_extract(e.record_json, '$.ingest_id')=? "
                "AND ((a.workload_class='GRAPHITI_CHAT_PRIMARY' "
                "AND a.provider='cursor-agent-cli' "
                "AND a.route='GRAPHITI_CHAT_PRIMARY') OR "
                "(a.workload_class='GRAPHITI_CHAT_FALLBACK' "
                "AND a.provider='grok-build-cli' "
                "AND a.route='GRAPHITI_CHAT_FALLBACK')) "
                "AND t.usage_status='UNREPORTED' "
                "AND t.failure_class='MISSING_PROVIDER_TELEMETRY' "
                "AND t.outcome IN ('FAILED','TIMEOUT') "
                "AND NOT EXISTS (SELECT 1 FROM model_usage_conservative_dispositions d "
                "WHERE d.invocation_id=a.invocation_id)",
                (ingest_id,),
            ).fetchall()
            for invocation_id, allocation_digest, terminal_digest, envelope_raw in rows:
                if json.loads(envelope_raw).get("ingest_id") != ingest_id:
                    raise ValueError("model work envelope ingest binding mismatch")
                # The usage service independently validates the exact native source,
                # qualified policy and dispatch. This retains ESTIMATED accounting,
                # never fabricated telemetry, and does not release a route circuit.
                self._usage.disposition_native_unreported_subscription_usage(
                    invocation_id=invocation_id,
                    expected_terminal_digest=terminal_digest,
                    expected_allocation_digest=allocation_digest,
                    observed_at=self._clock(),
                )
            cancelled_fallbacks = self._connection.execute(
                "SELECT a.invocation_id,a.canonical_digest,t.terminal_digest "
                "FROM model_work_envelopes e INDEXED BY model_usage_native_graphiti_ingest "
                "JOIN model_invocation_allocations a ON a.envelope_id=e.envelope_id "
                "JOIN model_invocation_terminals t ON t.invocation_id=a.invocation_id "
                "WHERE e.workload_class='GRAPHITI_CHAT_PRIMARY' "
                "AND json_extract(e.record_json,'$.ingest_id')=? "
                "AND a.workload_class='GRAPHITI_CHAT_FALLBACK' "
                "AND a.provider='grok-build-cli' AND a.route='GRAPHITI_CHAT_FALLBACK' "
                "AND t.outcome='CANCELLED' AND t.usage_status='UNREPORTED' "
                "AND t.failure_class='MISSING_PROVIDER_TELEMETRY' "
                "AND NOT EXISTS (SELECT 1 FROM model_usage_conservative_dispositions d "
                "WHERE d.invocation_id=a.invocation_id)",
                (ingest_id,),
            ).fetchall()
            for invocation_id, allocation_digest, terminal_digest in cancelled_fallbacks:
                self._stop_check()
                self._usage.disposition_native_graphiti_fallback_cancellation(
                    invocation_id=invocation_id,
                    expected_terminal_digest=terminal_digest,
                    expected_allocation_digest=allocation_digest,
                    observed_at=self._clock(),
                )
            cancelled = self._connection.execute(
                "SELECT a.invocation_id FROM model_work_envelopes e "
                "INDEXED BY model_usage_native_graphiti_ingest "
                "JOIN model_invocation_allocations a ON a.envelope_id=e.envelope_id "
                "JOIN model_invocation_terminals t ON t.invocation_id=a.invocation_id "
                "WHERE e.workload_class='GRAPHITI_CHAT_PRIMARY' "
                "AND json_extract(e.record_json,'$.ingest_id')=? "
                "AND a.workload_class='GRAPHITI_EMBEDDING' AND a.provider='openrouter' "
                "AND t.outcome='CANCELLED' AND t.usage_status='UNREPORTED' "
                "AND t.failure_class='MISSING_PROVIDER_TELEMETRY' "
                "AND (NOT EXISTS (SELECT 1 FROM model_usage_conservative_dispositions d "
                "WHERE d.invocation_id=a.invocation_id) OR a.invocation_id=("
                "SELECT CASE WHEN state='OPEN' AND reason='MISSING_PROVIDER_TELEMETRY' "
                "THEN invocation_id END FROM model_usage_route_circuit_events "
                "WHERE route=a.route ORDER BY recorded_at DESC,rowid DESC LIMIT 1))",
                (ingest_id,),
            ).fetchall()
            for (invocation_id,) in cancelled:
                self._stop_check()
                self._usage.disposition_native_graphiti_embedding_cancellation(
                    invocation_id=invocation_id, observed_at=self._clock(),
                )

    def _cohort_state(self, exact: tuple[str, ...], state: str) -> None:
        cohort_id = digest_canonical(exact)
        append_ledger(self._connection, "NATIVE_GRAPHITI_COHORT", {
            "cohort_id": cohort_id, "ingest_ids": list(exact), "state": state,
        })
        self._connection.commit()
        self._cohorts[cohort_id] = exact
        if state == "COMPLETE":
            self._completed.add(cohort_id)
