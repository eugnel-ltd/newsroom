"""One autonomous private Hermes iteration; no per-story owner gate."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
import math
import time
from typing import ContextManager

from newsroom.authority import UtcTimestamp

from .native_cycle import advance_native_cycle
from .native_assessor import assessor_admission_recovery_due, assessment_revalidation_due, same_assessment_producer
from .native_evidence import NativeEvidenceHold

from .native_progress import NativeRevisionJournal
from .veto import OperatorDrainRequested, VetoError


def _source_update_time(item: tuple) -> tuple:
    unit = item[1][0]
    for value in (unit.updated_at, unit.published_at):
        try:
            return True, UtcTimestamp.parse(value).value
        except ValueError:
            continue
    return False, None


@dataclass(frozen=True, slots=True)
class NativePipelineReport:
    sources: tuple[dict, ...]
    revision_states: dict[str, int]
    unclassified_revisions: int


class NativePipeline:
    """Reuse retained stages; isolate a held revision while other work advances.

    The daemon supplies the concrete retrieval builder and publication
    continuation. Neither a heartbeat nor this report is an acceptance PASS.
    """

    def __init__(
        self, *, runtime, journal: NativeRevisionJournal, source_intake,
        graphiti, discovery, retrieval_for: Callable, collision, publish,
        actor_identity_digest: str, stop_check: Callable[[], None],
        stop_fence: Callable[[], ContextManager[None]],
        refresh_rights: Callable[[], None] = lambda: None,
        operator_drain_requested: Callable[[], bool] = lambda: False,
        assessment_contract_version: str | None = None,
        reassessment_quantum_seconds: float = 300,
        monotonic_clock: Callable[[], float] = time.monotonic,
        clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
    ) -> None:
        if assessment_contract_version is not None and (
            type(assessment_contract_version) is not str or not assessment_contract_version
        ):
            raise ValueError("native assessment contract version differs")
        if (
            not callable(monotonic_clock)
            or not math.isfinite(reassessment_quantum_seconds)
            or reassessment_quantum_seconds <= 0
        ):
            raise ValueError("native reassessment quantum differs")
        self._runtime, self._journal = runtime, journal
        self._intake, self._graphiti, self._discovery = source_intake, graphiti, discovery
        self._retrieval_for, self._collision, self._publish = retrieval_for, collision, publish
        self._actor, self._check, self._fence, self._clock = actor_identity_digest, stop_check, stop_fence, clock
        self._refresh_rights = refresh_rights
        self._operator_drain_requested = operator_drain_requested
        self._assessment_contract_version = assessment_contract_version
        self._reassessment_quantum = reassessment_quantum_seconds
        self._monotonic_clock = monotonic_clock
        self._spill_archive_turn = False
        self.runtime_identity_digest: str | None = None

    def _drain_between_work(self) -> None:
        if self._operator_drain_requested():
            raise OperatorDrainRequested

    def terminal_report(self, report: NativePipelineReport) -> dict:
        # The service/CLI still returns the full logical report. Only its
        # durable diagnostic refers to the already committed source inventory.
        return {
            "source_portfolio_ref": self._journal.portfolio_reference(report.sources),
            "revision_states": dict(report.revision_states),
            "unclassified_revisions": report.unclassified_revisions,
        }

    def tick(self, *, cycle_id: str) -> NativePipelineReport:
        self._check()
        self._drain_between_work()
        self._refresh_rights()
        self._check()
        self._drain_between_work()
        dispositions = self._intake.poll()
        self._journal.sources(dispositions)
        grouped = defaultdict(list)
        for disposition in dispositions:
            for unit in disposition.units:
                grouped[unit.revision_id].append(unit)
        for revision_id, units in grouped.items():
            self._journal.land(tuple(units))
        self._drain_between_work()

        # Fixed disjoint cohorts attempt each revision at most once per tick.
        # Retained downstream work must not wait behind fresh model requests.
        ordinary, reassessments, pending_revisions = [], [], []
        for revision_id, units in self._journal.units.items():
            previous = self._journal.summary(revision_id)
            facts = previous.get("facts", {})
            if not facts.get("graphiti_receipts"):
                cohort = pending_revisions
            elif previous.get("stage") == "EVIDENCE_HOLD" and assessment_revalidation_due(
                facts, self._assessment_contract_version,
            ):
                # Exact cached consumer repairs need no model turn and should
                # not wait behind fresh extraction/provider backlog.
                cohort = ordinary if same_assessment_producer(
                    facts.get("assessment_contract_version"), self._assessment_contract_version,
                ) else reassessments
            else:
                cohort = ordinary
            cohort.append((revision_id, units))
        # Use the same current/archive turn for already-admitted downstream
        # work; recent source updates must not wait behind old recovery backlog.
        if not self._spill_archive_turn:
            ordinary.sort(key=_source_update_time, reverse=True)
        # Interrupted/unknown effects still settle before ordinary work. The
        # stable sort preserves source recency, or LAND order on archive turns.
        # Each turn has the existing quantum; an atomic revision may overrun it.
        ordinary.sort(key=lambda item: self._journal.summary(item[0]).get("stage")
                      not in {"ASSESSMENT_INTERRUPTED", "ASSESSMENT_STARTED", "PUBLICATION_STARTED", "COPY_CORRECTION_PREPARED"})
        ordinary_deadline = self._monotonic_clock() + self._reassessment_quantum
        ordinary_before = {revision: self._journal.progress_ordinal(revision) for revision, _ in ordinary}
        recover = getattr(self._publish, "recover_pre_dispatch", None)
        if callable(recover):
            def before_recovery() -> bool:
                self._drain_between_work()
                self._check()
                return self._monotonic_clock() < ordinary_deadline

            attempted = set(recover(
                tuple(revision_id for revision_id, _ in ordinary),
                before_revision=before_recovery,
            ))
            # Reclassification is this revision's only turn in the tick, not
            # permission for a fresh source/model retry using the batch proof.
            ordinary = [item for item in ordinary if item[0] not in attempted]
        deadline_deferred_ready = self._advance_revisions(
            tuple(ordinary),
            work_deadline=ordinary_deadline,
        )
        ordinary_turn_taken = any(
            self._journal.progress_ordinal(revision) != previous
            for revision, previous in ordinary_before.items()
        )
        self._drain_between_work()
        # Reuse one current/archive preference for pending and ready work.
        # Archive turns keep LAND order so later arrivals cannot starve history.
        if not self._spill_archive_turn:
            pending_revisions.sort(key=_source_update_time, reverse=True)
        pending_revisions = tuple(pending_revisions)
        fresh_deadline = self._monotonic_clock() + self._reassessment_quantum
        pending_turn_taken = False

        def defer_pending(unit) -> bool:
            nonlocal pending_turn_taken
            deferred = self._monotonic_clock() >= fresh_deadline
            pending_turn_taken |= not deferred
            return deferred

        # Extraction stays per ingest; projection remains one complete cohort.
        pending = tuple(unit for _, units in pending_revisions for unit in units)
        if pending:
            self._drain_between_work()
            self._check()
            try:
                results = self._graphiti.advance(
                    pending, cycle_id=cycle_id,
                    defer_before_unit=defer_pending,
                )
                if len(results) != len(pending) or {item.ingest_id for item in results} != {unit.ingest_id for unit in pending}:
                    raise ValueError("native Graphiti continuation partition differs")
                by_ingest = {item.ingest_id: item for item in results}
                for revision_id in dict.fromkeys(unit.revision_id for unit in pending):
                    outcomes = tuple(by_ingest[unit.ingest_id] for unit in self._journal.units[revision_id])
                    deferred = tuple(item for item in outcomes if item.state == "GRAPHITI_DEFERRED")
                    if deferred:
                        if any(item.reason != "WORK_QUANTUM_EXHAUSTED" or item.receipt_digest is not None
                               for item in deferred):
                            raise ValueError("native Graphiti deferral reason differs")
                        # A scheduling decision is not a durable failure. Keep
                        # the exact previous stage/facts until a later tick.
                        continue
                    facts = dict(self._journal.current(revision_id).get("facts", {}))
                    complete = all(item.state == "GRAPHITI_COMPLETE" for item in outcomes)
                    if complete:
                        facts.pop("graphiti_outcomes", None)
                        facts.pop("reason", None)
                        facts["graphiti_receipts"] = [asdict(item) for item in outcomes]
                    else:
                        held = tuple(
                            item for item in outcomes
                            if item.state in {"GRAPHITI_HOLD", "ADMISSION_HOLD"}
                        )
                        if not held:
                            raise ValueError("native Graphiti incomplete revision lacks a hold")
                        reasons = {
                            item.reason
                            for item in held
                        }
                        if any(type(reason) is not str or not reason for reason in reasons):
                            raise ValueError("native Graphiti hold reason differs")
                        facts.pop("graphiti_receipts", None)
                        facts["graphiti_outcomes"] = [asdict(item) for item in outcomes]
                        facts["reason"] = (
                            next(iter(reasons))
                            if len(reasons) == 1
                            else "MULTIPLE_GRAPHITI_HOLDS"
                        )
                    self._journal.advance(revision_id, stage="GRAPHITI_COMPLETE" if complete else "GRAPHITI_HOLD", facts=facts)
                    pending_turn_taken |= complete
                self._drain_between_work()
            except OperatorDrainRequested:
                raise
            except VetoError:
                raise
            except Exception as exc:
                for revision_id in dict.fromkeys(unit.revision_id for unit in pending):
                    facts = self._journal.current(revision_id).get("facts", {})
                    self._journal.advance(revision_id, stage="GRAPHITI_HOLD", facts={
                        **facts, "reason": type(exc).__name__,
                    })

        self._advance_revisions(pending_revisions, work_deadline=fresh_deadline)
        self._drain_between_work()
        # Changed-contract reassessment has its own quantum after fresh work;
        # stale model requests cannot delay a newly landed revision's first turn.
        # This finite old-contract cohort drains once: repair recent evidence
        # before older failures, without reordering fresh work or unknown effects.
        reassessments.sort(
            key=lambda item: max(UtcTimestamp.parse(unit.observed_at).value for unit in item[1]),
            reverse=True,
        )
        ready_spill = deadline_deferred_ready
        if not self._spill_archive_turn:
            ready_spill = tuple(sorted(ready_spill, key=_source_update_time, reverse=True))
        deferred = self._advance_revisions(
            # Unattempted canonical work must not starve behind an unresolved
            # predecessor. Reuse this existing assessment budget, once per unit.
            ready_spill + tuple(reassessments),
            work_deadline=self._monotonic_clock() + self._reassessment_quantum,
        )
        if ordinary_turn_taken or pending_turn_taken or len(deferred) < len(ready_spill):
            # Consume the shared preference at most once after eligible pending
            # work, ordinary progress or an actual ready-spill turn.
            # Restart resets this preference, never retained work.
            self._spill_archive_turn = not self._spill_archive_turn
        self._drain_between_work()
        states = Counter(
            self._journal.summary(revision_id).get("stage", "QUEUED")
            for revision_id in self._journal.units
        )
        return NativePipelineReport(
            self._journal.portfolio, dict(states), states.get("QUEUED", 0),
        )

    def _advance_revisions(
        self, revisions: tuple, *, work_deadline: float,
    ) -> tuple:
        # Each revision remains in the journal even when it disappears from the
        # next feed page. This is work continuation, not a fresh provider retry.
        deadline_deferred_ready = []
        for revision_id, units in revisions:
            self._drain_between_work()
            self._check()
            previous = self._journal.summary(revision_id)
            if self._monotonic_clock() >= work_deadline:
                if (previous.get("stage") == "GRAPHITI_COMPLETE"
                        and previous.get("facts", {}).get("graphiti_receipts")
                        and not previous.get("facts", {}).get("candidate_version_id")):
                    deadline_deferred_ready.append((revision_id, units))
                continue
            facts = dict(previous.get("facts", {}))
            stage = "PUBLICATION"
            try:
                if previous.get("stage") in {"ASSESSMENT_INTERRUPTED", "COPY_CORRECTION_PREPARED"} or (
                    previous.get("stage") == "EVIDENCE_HOLD"
                    and assessor_admission_recovery_due(previous.get("facts", {}))
                ):
                    candidate_version_id = previous.get("facts", {}).get(
                        "candidate_version_id"
                    )
                    if type(candidate_version_id) is str and candidate_version_id:
                        self._publish.advance(
                            revision_id=revision_id,
                            candidate_version_id=candidate_version_id,
                        )
                    continue
                if previous.get("stage") == "ACKNOWLEDGED":
                    due = getattr(self._publish, "copy_correction_due", None)
                    facts = previous.get("facts", {})
                    if callable(due) and due(facts):
                        self._publish.advance(revision_id=revision_id, candidate_version_id=facts["candidate_version_id"])
                    continue
                if previous.get("stage") == "SAME_STATE_ASSOCIATED":
                    continue
                if (
                    previous.get("stage") == "EVIDENCE_HOLD"
                    and previous.get("facts", {}).get("acquisition_retryable") is not True
                    and previous.get("facts", {}).get("reason") not in {
                        "CURRENT_RIGHTS_HOLD", "GOVUK_LICENCE_BINDING_HOLD",
                        "GOVUK_LICENCE_REVIEW_HOLD", "NATIVE_SOURCE_RIGHTS_HOLD",
                        "PUBLICATION_RIGHTS_HOLD",
                    }
                    and not assessment_revalidation_due(
                        previous.get("facts", {}), self._assessment_contract_version,
                    )
                ):
                    continue
                stage = "GRAPHITI"
                if not facts.get("graphiti_receipts"):
                    continue
                candidate_version_id = facts.get("candidate_version_id")
                if candidate_version_id is None:
                    stage = "DISCOVERY"
                    now = self._clock()
                    delivered = self._discovery.deliver(
                        units[0], now=now, proof=self._runtime.proof,
                    )
                    status = self._discovery.admit_lead(
                        delivered, now=now, proof=self._runtime.proof,
                    )
                    if status.lead is None:
                        facts = self._journal.current(revision_id).get("facts", {})
                        self._journal.advance(revision_id, stage="DISCOVERY_HOLD", facts={
                            **facts, "reason": status.phase.value,
                        })
                        continue
                    stage = "RETRIEVAL"
                    retrieval = self._retrieval_for(units)
                    outcomes = advance_native_cycle(
                        system=self._runtime.authority, statuses=(status,), retrieval=retrieval,
                        collision_requests=self._collision,
                        actor_identity_digest=self._actor, proof=self._runtime.proof,
                        owner_stop_check=self._check, owner_stop_fence=self._fence,
                    )
                    facts = dict(self._journal.current(revision_id).get("facts", {}))
                    if len(outcomes) != 1 or outcomes[0].revision_id != revision_id:
                        raise ValueError("native Candidate continuation partition differs")
                    outcome = outcomes[0]
                    if outcome.state != "CANDIDATE_ADMITTED":
                        self._journal.advance(revision_id, stage=outcome.state, facts={
                            **facts, "reason": outcome.reason,
                        })
                        continue
                    candidate_version_id = outcome.triage.candidate.version_id
                    facts["candidate_version_id"] = candidate_version_id
                    self._journal.advance(revision_id, stage="CANDIDATE_ADMITTED", facts=facts)
                stage = "PUBLICATION"
                self._drain_between_work()
                self._publish.advance(
                    revision_id=revision_id, candidate_version_id=candidate_version_id,
                )
            except OperatorDrainRequested:
                raise
            except VetoError:
                raise
            except Exception as exc:
                # Do not overwrite a more precise durable provider-dispatch or
                # publication intent marker with a generic outer-loop failure.
                retained = self._journal.summary(revision_id)
                if retained.get("stage") in {
                    "ASSESSMENT_STARTED", "PUBLICATION_STARTED", "ASSESSMENT_INTERRUPTED",
                    "COPY_CORRECTION_PREPARED", "ACKNOWLEDGED",
                } or (
                    retained.get("stage") == "EVIDENCE_HOLD"
                    and assessor_admission_recovery_due(retained.get("facts", {}))
                ):
                    if isinstance(exc, NativeEvidenceHold):
                        self._journal.advance(revision_id, stage=retained["stage"], facts={
                            **self._journal.current(revision_id).get("facts", {}),
                            "last_continuation_hold": {
                                "reason": exc.reason_code, "source_id": exc.source_id,
                            },
                        })
                    continue
                self._journal.advance(revision_id, stage=f"{stage}_HOLD", facts={
                    **self._journal.current(revision_id).get("facts", {}),
                    "reason": getattr(exc, "reason", getattr(exc, "reason_code", type(exc).__name__)),
                })
        self._drain_between_work()
        return tuple(deadline_deferred_ready)
