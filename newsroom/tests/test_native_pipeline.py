from contextlib import nullcontext
from dataclasses import replace
import threading
from types import SimpleNamespace as NS

import pytest

from newsroom.authority import UtcTimestamp
from newsroom.control_plane import native_pipeline as n
from newsroom.control_plane.native_graphiti import NativeGraphitiOutcome
from newsroom.control_plane.native_progress import NativeRevisionJournal
from newsroom.control_plane.store import connect
from newsroom.control_plane.veto import OperatorDrainRequested, VetoError
from newsroom.tests.test_native_graphiti import _native


def test_pending_shares_only_equal_bodies_with_exact_units_and_fresh_authority(tmp_path, monkeypatch):
    from dataclasses import asdict
    from newsroom.authority.canonical import canonical_json_bytes
    from newsroom.tests.test_graphiti_operational_readiness import _next_revision
    pipeline, journal, connection, units, _, dispositions = _open(tmp_path, monkeypatch)
    units = (*units, _next_revision(units[0]))
    pipeline._spill_archive_turn = True
    for unit in units:
        journal.land((unit,))
    dispositions[0] = ()
    seen = []

    def graphiti(selected, **kwargs):
        assert canonical_json_bytes([asdict(unit) for unit in selected]) == canonical_json_bytes([asdict(unit) for unit in units])
        assert selected[0].body is selected[1].body
        assert selected[2].body != selected[0].body
        assert selected[0].authority is not selected[1].authority
        assert selected[0].authority.records[0] is not selected[1].authority.records[0]
        original = selected[0].authority.records[0]["record_id"]
        selected[0].authority.records[0]["record_id"] = "detached-mutation"
        assert selected[1].authority.records[0]["record_id"] != "detached-mutation"
        selected[0].authority.records[0]["record_id"] = original
        assert journal.units[units[0].revision_id] == (units[0],)
        assert not seen or selected[0].body is not seen[0]
        seen.append(selected[0].body)
        return tuple(NativeGraphitiOutcome(unit.ingest_id, "GRAPHITI_DEFERRED", None, "WORK_QUANTUM_EXHAUSTED") for unit in selected)

    pipeline._graphiti = NS(advance=graphiti)
    try:
        pipeline.tick(cycle_id="equal-pending-first")
        pipeline._journal = NativeRevisionJournal(connection)
        pipeline.tick(cycle_id="equal-pending-reopen")
        assert len(seen) == 2 and pipeline._journal._bodies == {}
        assert {revision for revision in journal.units} == {unit.revision_id for unit in units}
    finally:
        connection.close()


@pytest.mark.parametrize("error", [OperatorDrainRequested(), VetoError("isolated veto"), KeyboardInterrupt()])
def test_pending_exception_traceback_keeps_no_pipeline_cohort(error, tmp_path, monkeypatch):
    pipeline, journal, connection, units, _, dispositions = _open(tmp_path, monkeypatch)
    for unit in units:
        journal.land((unit,))
    dispositions[0] = ()

    def graphiti(selected, **kwargs):
        raise error

    pipeline._graphiti = NS(advance=graphiti)
    try:
        with pytest.raises(type(error)) as caught:
            pipeline.tick(cycle_id="pending-cancel")
        assert caught.value is error
        traceback = caught.value.__traceback__
        while traceback.tb_frame.f_code.co_name != "tick":
            traceback = traceback.tb_next
        assert traceback.tb_frame.f_locals["pending"] == ()
        assert traceback.tb_frame.f_locals["results"] == ()
        assert traceback.tb_frame.f_locals["by_ingest"] == {}
        assert all(journal.summary(unit.revision_id) == {} for unit in units)
    finally:
        error.__traceback__ = None
        connection.close()


@pytest.mark.parametrize("failure", [False, True])
def test_pending_units_are_released_before_downstream_even_after_graphiti_failure(tmp_path, monkeypatch, failure):
    import gc
    import weakref
    from dataclasses import fields
    from newsroom.control_plane.corpus import CorpusIngestUnit
    from newsroom.control_plane.native_progress import _CurrentUnits

    class TrackedUnit(CorpusIngestUnit):
        __slots__ = ("__weakref__",)

    pipeline, journal, connection, units, _, dispositions = _open(tmp_path, monkeypatch)
    for unit in units:
        journal.land((unit,))
    dispositions[0] = ()
    original = _CurrentUnits.__getitem__
    monkeypatch.setattr(_CurrentUnits, "__getitem__", lambda self, revision: tuple(
        TrackedUnit(*(getattr(unit, field.name) for field in fields(CorpusIngestUnit)))
        for unit in original(self, revision)))
    retained = []

    def graphiti(selected, **kwargs):
        retained.extend(weakref.ref(unit) for unit in selected)
        if failure:
            raise ValueError("isolated Graphiti failure")
        return tuple(NativeGraphitiOutcome(unit.ingest_id, "GRAPHITI_COMPLETE", unit.digest, None) for unit in selected)

    def downstream(revisions, **kwargs):
        if revisions:
            gc.collect()
            assert retained and all(reference() is None for reference in retained)
        return ()

    pipeline._graphiti = NS(advance=graphiti)
    pipeline._advance_revisions = downstream
    try:
        pipeline.tick(cycle_id="pending-lifetime")
        assert {journal.summary(unit.revision_id)["stage"] for unit in units} == {"GRAPHITI_HOLD" if failure else "GRAPHITI_COMPLETE"}
    finally:
        connection.close()


def _open(tmp_path, monkeypatch):
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    units = (_native("one"), _native("two"))
    calls = []
    class Graphiti:
        def advance(self, selected, *, cycle_id, **kwargs):
            calls.append(("graphiti", selected[0].item_key))
            return tuple(NativeGraphitiOutcome(unit.ingest_id, "GRAPHITI_COMPLETE", unit.digest, None) for unit in selected)
    class Discovery:
        def deliver(self, unit, **kw):
            calls.append(("discovery", unit.item_key))
            return unit
        def admit_lead(self, unit, **kw):
            return NS(lead=unit, phase=NS(value="LEAD"))
    class Publisher:
        def advance(self, *, revision_id, candidate_version_id):
            calls.append(("publish", revision_id))
            journal.advance(revision_id, stage="ACKNOWLEDGED", facts={
                **journal.current(revision_id)["facts"], "ack_receipt": "isolated-test-only",
            })
    def advance(**kw):
        unit = kw["statuses"][0].lead
        return (NS(revision_id=unit.revision_id, state="CANDIDATE_ADMITTED",
                   triage=NS(candidate=NS(version_id="candidate:" + unit.item_key))),)
    monkeypatch.setattr(n, "advance_native_cycle", advance)
    dispositions = [tuple(NS(source_id=unit.source_id, status="READY", reason_code="RETAINED",
                             units=(unit,)) for unit in units)]
    pipeline = n.NativePipeline(
        runtime=NS(authority=object(), proof=object()), journal=journal,
        source_intake=NS(poll=lambda: dispositions[0]), graphiti=Graphiti(),
        discovery=Discovery(), retrieval_for=lambda units: object(), collision=object(),
        publish=Publisher(), actor_identity_digest="sha256:" + "a" * 64,
        stop_check=lambda: None, stop_fence=nullcontext,
        refresh_rights=lambda: calls.append(("rights", "current")),
        clock=lambda: UtcTimestamp.parse("2026-09-08T12:00:00Z"),
    )
    return pipeline, journal, connection, units, calls, dispositions


def test_native_pipeline_continues_multiple_revisions_and_skips_acknowledged(tmp_path, monkeypatch):
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    try:
        report = pipeline.tick(cycle_id="first")
        assert report.revision_states == {"ACKNOWLEDGED": 2}
        assert report.unclassified_revisions == 0
        assert len([item for item in calls if item[0] == "graphiti"]) == 1
        first_calls = tuple(calls)
        dispositions[0] = ()
        pipeline.tick(cycle_id="second")
        assert tuple(calls) == first_calls + (("rights", "current"),)
        assert len(journal.units) == 2
    finally:
        connection.close()


def test_current_output_restoration_runs_once_after_canonical_landing_before_provider_work(tmp_path,monkeypatch):
    pipeline,journal,connection,units,calls,dispositions = _open(tmp_path,monkeypatch)
    def restore():
        assert set(journal.units)=={unit.revision_id for unit in units}
        calls.append(('restore','current'))
    pipeline._publish.restore_current_output = restore
    try:
        pipeline.tick(cycle_id='current-output-restoration')
        assert calls.count(('restore','current'))==1
        assert calls.index(('rights','current'))<calls.index(('restore','current'))
        assert calls.index(('restore','current'))<next(index for index,item in enumerate(calls)if item[0]=='graphiti')
    finally:
        connection.close()


@pytest.mark.parametrize("ministerial", (False, True))
def test_archival_nil_return_waits_for_graphiti_then_skips_optional_model_work(tmp_path, monkeypatch, ministerial):
    from newsroom.tests.test_native_source_disposition import _fixture, _ministerial_hospitality_fixture, NOW
    pipeline, journal, connection, _, calls, dispositions = _open(tmp_path, monkeypatch)
    unit, original = _ministerial_hospitality_fixture() if ministerial else _fixture()
    dispositions[0] = (NS(source_id=unit.source_id, status="READY", reason_code="RETAINED", units=(unit,)),)
    source_reads = []
    def revision(identity, **kwargs):
        source_reads.append(str(identity))
        return NS(request=original)
    pipeline._runtime.authority = NS(sources=NS(revision=revision))
    pipeline._clock = lambda: NOW
    pipeline._retrieval_for = lambda _: pytest.fail("nil disclosure reached embedding/retrieval")
    pipeline._publish.advance = lambda **_: pytest.fail("nil disclosure reached assessor/publication")
    graphiti = pipeline._graphiti.advance
    pipeline._graphiti.advance = lambda selected, **_: tuple(
        NativeGraphitiOutcome(item.ingest_id, "GRAPHITI_HOLD", None, "RETAINED_GRAPH_HOLD") for item in selected)
    try:
        pipeline.tick(cycle_id="graph-held")
        assert journal.current(unit.revision_id)["stage"] == "GRAPHITI_HOLD"
        assert source_reads == []
        pipeline._graphiti.advance = graphiti
        report = pipeline.tick(cycle_id="graph-complete")
        assert report.revision_states == {"EVIDENCE_HOLD": 1}
        facts = journal.current(unit.revision_id)["facts"]
        assert facts["reason"] == "NO_QUALIFYING_NEW_INFORMATION"
        assert facts["source_disposition"]["zero_call"] is True
        assert "assessment_contract_version" not in facts and "candidate_version_id" not in facts
        assert facts["graphiti_receipts"][0]["ingest_id"] == unit.ingest_id
        assert facts["graphiti_receipts"][0]["state"] == "GRAPHITI_COMPLETE"
        assert source_reads == [unit.revision_id]
        assert not any(call[0] in {"discovery", "publish"} for call in calls)
        pipeline._journal = NativeRevisionJournal(connection)
        assert pipeline._journal.units[unit.revision_id] == (unit,)
        pipeline.tick(cycle_id="reopened")
        assert source_reads == [unit.revision_id]
        assert pipeline._journal.current(unit.revision_id)["facts"] == facts
    finally:
        connection.close()


def test_archival_disposition_never_blocks_changed_source_revision(tmp_path, monkeypatch):
    from newsroom.tests.test_native_source_disposition import _fixture, NOW
    from newsroom.sources import SourceRevisionId, SourceTime
    pipeline, journal, connection, _, calls, dispositions = _open(tmp_path, monkeypatch)
    unit, original = _fixture()
    originals = {unit.revision_id: original}
    pipeline._runtime.authority = NS(sources=NS(revision=lambda identity, **__: NS(request=originals[str(identity)])))
    pipeline._clock = lambda: NOW
    dispositions[0] = (NS(source_id=unit.source_id, status="READY", reason_code="RETAINED", units=(unit,)),)
    retrieval_calls = []
    pipeline._retrieval_for = lambda selected: retrieval_calls.append(selected) or object()
    try:
        pipeline.tick(cycle_id="archival-original")
        assert journal.current(unit.revision_id)["stage"] == "EVIDENCE_HOLD" and retrieval_calls == []
        old_revision = unit.revision_id
        revision_id = SourceRevisionId.new()
        # The same item's new native revision/date and populated field remain eligible.
        unit = replace(unit, authority=replace(unit.authority, revision_id=str(revision_id)),
            body=unit.body.replace('B="Nil Return "', 'B="2 August 2019"'),
            updated_at=NOW.to_text(), effective_pull_first_observed_at=NOW.to_text())
        originals[unit.revision_id] = replace(original, revision_id=revision_id,
            prior_revision_id=original.revision_id, permitted_state_digest=unit.revision_digest,
            source_updated_time=SourceTime.exact(NOW), source_native_revision_token=NOW.to_text(), observed_at=NOW)
        dispositions[0] = (NS(source_id=unit.source_id, status="READY", reason_code="RETAINED", units=(unit,)),)
        report = pipeline.tick(cycle_id="material-observation")
        assert report.revision_states == {"ACKNOWLEDGED": 1, "EVIDENCE_HOLD": 1}
        assert journal.current(old_revision)["facts"]["source_disposition"]["zero_call"] is True
        assert retrieval_calls == [(unit,)]
        assert ("discovery", unit.item_key) in calls and ("publish", unit.revision_id) in calls
        assert "source_disposition" not in journal.current(unit.revision_id)["facts"]
    finally:
        connection.close()


def test_ordinary_phase_timing_keeps_pipeline_decisions_and_ledger_when_dropped(tmp_path, monkeypatch):
    from newsroom.control_plane import native_graphiti
    events = []
    monkeypatch.setattr(native_graphiti, "emit_diagnostic", lambda event, value: events.append((event, value)))
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    pipeline._publish.recover_pre_dispatch = lambda *_args, **_kwargs: ()
    try:
        first = pipeline.tick(cycle_id="timed")
        assert first.revision_states == {"ACKNOWLEDGED": 2}
        assert {value["phase"] for _, value in events} == {"CLASSIFY", "ORDINARY_RECOVERY", "ORDINARY_ADVANCE"}
        assert all(value["status"] == "COMPLETE" and value["cpu_scope"] == "PROCESS"
                   and type(value["elapsed_ms"]) is int for _, value in events)
        before_calls, before_changes = tuple(calls), connection.total_changes
        monkeypatch.setattr(native_graphiti, "emit_diagnostic", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("dropped")))
        second = pipeline.tick(cycle_id="dropped")
        assert second.revision_states == first.revision_states
        assert tuple(calls) == before_calls + (("rights", "current"),)
        assert connection.total_changes == before_changes
    finally:
        connection.close()


@pytest.mark.parametrize("phase", ["CLASSIFY", "ORDINARY_RECOVERY", "ORDINARY_ADVANCE"])
def test_ordinary_phase_drop_preserves_original_exception(tmp_path, monkeypatch, phase):
    from newsroom.control_plane import native_graphiti
    monkeypatch.setattr(native_graphiti, "emit_diagnostic", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("dropped")))
    pipeline, journal, connection, _units, _calls, _dispositions = _open(tmp_path, monkeypatch)

    def failure(*_args, **_kwargs):
        raise RuntimeError("original ordinary failure")

    if phase == "CLASSIFY":
        monkeypatch.setattr(journal, "summary", failure)
    elif phase == "ORDINARY_RECOVERY":
        pipeline._publish.recover_pre_dispatch = failure
    else:
        monkeypatch.setattr(pipeline, "_advance_revisions", failure)
    try:
        with pytest.raises(RuntimeError, match="original ordinary failure"):
            pipeline.tick(cycle_id="phase-failure")
    finally:
        connection.close()


@pytest.mark.parametrize("same_producer", (True, False))
def test_exact_consumer_revalidation_precedes_new_provider_work(tmp_path, monkeypatch, same_producer):
    pipeline, journal, connection, units, calls, _ = _open(tmp_path, monkeypatch)
    pipeline._assessment_contract_version = "assessor.v15+new-consumer"
    journal.land((units[0],))
    journal.advance(units[0].revision_id, stage="EVIDENCE_HOLD", facts={
        "graphiti_receipts": [{"retained": True}],
        "candidate_version_id": "candidate:one",
        "reason": "ASSESSOR_RENDERING_CONTRACT_HOLD",
        "assessment_contract_version": (
            "assessor.v15+old-consumer" if same_producer else "assessor.v14+old-consumer"
        ),
    })
    try:
        pipeline.tick(cycle_id="first")
        publication = calls.index(("publish", units[0].revision_id))
        provider = calls.index(("graphiti", units[1].item_key))
        assert (publication < provider) is same_producer
        pipeline.tick(cycle_id="unchanged")
        assert calls.count(("publish", units[0].revision_id)) == 1
    finally:
        connection.close()


def test_first_empty_source_poll_retains_one_reopenable_terminal_reference(tmp_path, monkeypatch):
    pipeline, journal, connection, _units, _calls, dispositions = _open(tmp_path, monkeypatch)
    dispositions[0] = ()
    try:
        report = pipeline.tick(cycle_id="first-empty")
        assert report == n.NativePipelineReport((), {}, 0)
        terminal = pipeline.terminal_report(report)
        reference = terminal["source_portfolio_ref"]
        assert NativeRevisionJournal(connection).portfolio_reference(()) == reference
        assert pipeline.terminal_report(pipeline.tick(cycle_id="still-empty")) == terminal
        assert connection.execute("SELECT count(*) FROM ledger").fetchone()[0] == 1
    finally:
        connection.close()


def test_native_pipeline_retains_same_state_association_without_retry(
    tmp_path, monkeypatch,
):
    pipeline, journal, connection, units, calls, dispositions = _open(
        tmp_path, monkeypatch,
    )
    triage_calls = []

    def associate(**kwargs):
        unit = kwargs["statuses"][0].lead
        triage_calls.append(unit.revision_id)
        return (
            NS(
                revision_id=unit.revision_id,
                state="SAME_STATE_ASSOCIATED",
                triage=NS(candidate=None),
                reason=None,
            ),
        )

    monkeypatch.setattr(n, "advance_native_cycle", associate)
    try:
        first = pipeline.tick(cycle_id="same-state")
        assert first.revision_states == {"SAME_STATE_ASSOCIATED": 2}
        assert triage_calls == [item.revision_id for item in units]
        assert not any(call[0] == "publish" for call in calls)

        dispositions[0] = ()
        second = pipeline.tick(cycle_id="same-state-replay")
        assert second.revision_states == {"SAME_STATE_ASSOCIATED": 2}
        assert triage_calls == [item.revision_id for item in units]
        assert all(
            journal.current(item.revision_id)["stage"]
            == "SAME_STATE_ASSOCIATED"
            for item in units
        )
    finally:
        connection.close()


def test_native_pipeline_drains_between_revisions_and_restart_reuses_settled_work(
    tmp_path, monkeypatch,
):
    pipeline, journal, connection, units, calls, dispositions = _open(
        tmp_path, monkeypatch,
    )
    service_event = threading.Event()
    original = pipeline._publish

    class DrainAfterFirstPublication:
        def advance(self, *, revision_id, candidate_version_id):
            original.advance(
                revision_id=revision_id,
                candidate_version_id=candidate_version_id,
            )
            service_event.set()

    pipeline._publish = DrainAfterFirstPublication()
    pipeline._operator_drain_requested = service_event.is_set
    try:
        with pytest.raises(OperatorDrainRequested):
            pipeline.tick(cycle_id="draining")
        assert len([call for call in calls if call[0] == "graphiti"]) == 1
        assert [
            journal.current(unit.revision_id)["stage"] for unit in units
        ] == ["ACKNOWLEDGED", "GRAPHITI_COMPLETE"]
        assert len([call for call in calls if call[0] == "publish"]) == 1

        service_event.clear()
        pipeline._publish = original
        dispositions[0] = ()
        report = pipeline.tick(cycle_id="restart")
        assert report.revision_states == {"ACKNOWLEDGED": 2}
        assert len([call for call in calls if call[0] == "graphiti"]) == 1
        assert len([call for call in calls if call[0] == "publish"]) == 2
    finally:
        connection.close()


def test_native_pipeline_lands_polled_work_before_operator_drain(
    tmp_path, monkeypatch,
):
    pipeline, journal, connection, units, calls, _, = _open(tmp_path, monkeypatch)
    service_event = threading.Event()
    original_poll = pipeline._intake.poll

    def poll_then_drain():
        result = original_poll()
        service_event.set()
        return result

    pipeline._intake = NS(poll=poll_then_drain)
    pipeline._operator_drain_requested = service_event.is_set
    try:
        with pytest.raises(OperatorDrainRequested):
            pipeline.tick(cycle_id="drain-after-poll")
        assert set(journal.units) == {unit.revision_id for unit in units}
        assert not any(call[0] == "graphiti" for call in calls)
    finally:
        connection.close()


def test_native_pipeline_checkpoints_graphiti_results_before_operator_drain(
    tmp_path, monkeypatch,
):
    pipeline, journal, connection, units, calls, _ = _open(tmp_path, monkeypatch)
    service_event = threading.Event()
    original = pipeline._graphiti

    class GraphitiThenDrain:
        def advance(self, selected, *, cycle_id, **kwargs):
            result = original.advance(selected, cycle_id=cycle_id, **kwargs)
            service_event.set()
            return result

    pipeline._graphiti = GraphitiThenDrain()
    pipeline._operator_drain_requested = service_event.is_set
    try:
        with pytest.raises(OperatorDrainRequested):
            pipeline.tick(cycle_id="drain-after-graphiti")
        assert all(
            journal.current(unit.revision_id)["stage"] == "GRAPHITI_COMPLETE"
            for unit in units
        )
        assert not any(call[0] in {"discovery", "publish"} for call in calls)
    finally:
        connection.close()


def test_native_pipeline_checkpoints_candidate_before_operator_drain(
    tmp_path, monkeypatch,
):
    pipeline, journal, connection, units, calls, dispositions = _open(
        tmp_path, monkeypatch,
    )
    service_event = threading.Event()
    dispositions[0] = ()
    unit = units[0]
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="GRAPHITI_COMPLETE", facts={
        "graphiti_receipts": [{}],
    })

    def advance_then_drain(**kw):
        service_event.set()
        lead = kw["statuses"][0].lead
        return (NS(
            revision_id=lead.revision_id, state="CANDIDATE_ADMITTED",
            triage=NS(candidate=NS(version_id="candidate:" + lead.item_key)),
        ),)

    monkeypatch.setattr(n, "advance_native_cycle", advance_then_drain)
    pipeline._operator_drain_requested = service_event.is_set
    try:
        with pytest.raises(OperatorDrainRequested):
            pipeline.tick(cycle_id="drain-after-candidate")
        progress = journal.current(unit.revision_id)
        assert progress["stage"] == "CANDIDATE_ADMITTED"
        assert progress["facts"]["candidate_version_id"] == "candidate:one"
        assert not any(call[0] == "publish" for call in calls)
    finally:
        connection.close()


@pytest.mark.parametrize(("initial_stage", "failure_class"), (("ASSESSMENT_INTERRUPTED", "EvidencePackageError"), ("EVIDENCE_HOLD", "ModelUsageAdmissionError")))
def test_native_pipeline_only_reclassifies_retained_assessment_interruption(
    tmp_path, monkeypatch, initial_stage, failure_class,
):
    pipeline, journal, connection, units, calls, dispositions = _open(
        tmp_path, monkeypatch,
    )
    dispositions[0] = ()
    journal.land((units[0],))
    journal.advance(units[0].revision_id, stage=initial_stage, facts={
        "candidate_version_id": "candidate:one",
        "graphiti_receipts": [{}],
        "failure_class": failure_class,
        "reason": "ACQUISITION_RESULT_NOT_RETAINED",
    })

    class Recovery:
        def advance(self, *, revision_id, candidate_version_id):
            calls.append(("recover", revision_id, candidate_version_id))
            journal.advance(revision_id, stage="EVIDENCE_HOLD", facts={
                **journal.current(revision_id)["facts"],
                "reason": "ASSESSOR_OUTPUT_CONTRACT_HOLD",
                "acquisition_retryable": False,
            })

    pipeline._publish = Recovery()
    calls.clear()
    try:
        first = pipeline.tick(cycle_id="recovery")
        assert first.revision_states == {"EVIDENCE_HOLD": 1}
        assert calls == [
            ("rights", "current"),
            ("recover", units[0].revision_id, "candidate:one"),
        ]
        calls.clear()
        pipeline.tick(cycle_id="replay")
        assert calls == [("rights", "current")]
    finally:
        connection.close()


def test_native_pipeline_does_not_restart_unproved_assessment_interruptions(
    tmp_path, monkeypatch,
):
    pipeline, journal, connection, units, calls, dispositions = _open(
        tmp_path, monkeypatch,
    )
    dispositions[0] = ()
    retained_ordinals = {}
    for unit, failure_class in zip(
        units, ("EvidencePackageError", "OSError"), strict=True
    ):
        journal.land((unit,))
        journal.advance(unit.revision_id, stage="ASSESSMENT_INTERRUPTED", facts={
            "candidate_version_id": "candidate:" + unit.item_key,
            "graphiti_receipts": [{}],
            "failure_class": failure_class,
            "reason": "ACQUISITION_RESULT_NOT_RETAINED",
        })
        retained_ordinals[unit.revision_id] = journal.current(unit.revision_id)[
            "ordinal"
        ]

    class UnprovedRecovery:
        def advance(self, *, revision_id, candidate_version_id):
            calls.append(("proof-only", revision_id, candidate_version_id))
            return NS(state="ASSESSMENT_INTERRUPTED")

    pipeline._publish = UnprovedRecovery()
    calls.clear()
    try:
        report = pipeline.tick(cycle_id="unproved-recovery")
        assert report.revision_states == {"ASSESSMENT_INTERRUPTED": 2}
        assert calls == [
            ("rights", "current"),
            ("proof-only", units[0].revision_id, "candidate:one"),
            ("proof-only", units[1].revision_id, "candidate:two"),
        ]
        assert {
            revision_id: journal.current(revision_id)["ordinal"]
            for revision_id in retained_ordinals
        } == retained_ordinals
    finally:
        connection.close()


def test_native_pipeline_retains_hold_reason_then_clears_it_on_continuation(
    tmp_path, monkeypatch,
):
    pipeline, journal, connection, units, calls, dispositions = _open(
        tmp_path, monkeypatch,
    )
    prefix = replace(units[1], chunk_count=2)
    predecessor_held = replace(
        prefix, chunk_ordinal=2, predecessor_ingest_id=prefix.ingest_id,
    )
    dispositions[0] = (
        NS(source_id=units[0].source_id, status="READY", reason_code="RETAINED", units=(units[0],)),
        NS(source_id=prefix.source_id, status="READY", reason_code="RETAINED", units=(prefix, predecessor_held)),
    )
    attempts = 0

    class Graphiti:
        def advance(self, selected, *, cycle_id, **kwargs):
            nonlocal attempts
            attempts += 1
            outcomes = []
            for unit in selected:
                state = "GRAPHITI_COMPLETE"
                reason = None
                if attempts == 1 and unit.revision_id == prefix.revision_id:
                    state = (
                        "EXTRACTION_COMPLETE"
                        if unit.chunk_ordinal == 1
                        else "GRAPHITI_HOLD"
                    )
                    reason = (
                        None
                        if unit.chunk_ordinal == 1
                        else "RIGHTS_OR_PREDECESSOR_HOLD"
                    )
                outcomes.append(NativeGraphitiOutcome(
                    unit.ingest_id, state,
                    None if state == "GRAPHITI_HOLD" else "sha256:" + "a" * 64,
                    reason,
                ))
            return tuple(outcomes)

    pipeline._graphiti = Graphiti()
    try:
        first = pipeline.tick(cycle_id="first-frontier")
        assert first.revision_states == {"ACKNOWLEDGED": 1, "GRAPHITI_HOLD": 1}
        held = journal.current(prefix.revision_id)
        assert held["facts"]["reason"] == "RIGHTS_OR_PREDECESSOR_HOLD"
        assert [item["state"] for item in held["facts"]["graphiti_outcomes"]] == [
            "EXTRACTION_COMPLETE", "GRAPHITI_HOLD",
        ]
        first_publish = [call for call in calls if call[0] == "publish"]

        dispositions[0] = ()
        second = pipeline.tick(cycle_id="second-frontier")
        assert second.revision_states == {"ACKNOWLEDGED": 2}
        assert [call for call in calls if call[0] == "publish"] == first_publish + [
            ("publish", prefix.revision_id),
        ]
        completed = journal.current(prefix.revision_id)["facts"]
        assert "reason" not in completed
        assert "graphiti_outcomes" not in completed
        assert completed["graphiti_receipts"][0]["state"] == "GRAPHITI_COMPLETE"
    finally:
        connection.close()


def test_native_pipeline_rolls_up_multiple_holds_and_rejects_a_missing_reason(
    tmp_path, monkeypatch,
):
    pipeline, journal, connection, _, _, dispositions = _open(tmp_path, monkeypatch)
    first = replace(_native("multi"), chunk_count=2)
    second = replace(
        first, chunk_ordinal=2, predecessor_ingest_id=first.ingest_id,
    )
    dispositions[0] = (
        NS(source_id=first.source_id, status="READY", reason_code="RETAINED", units=(first, second)),
    )

    class Graphiti:
        missing = False

        def advance(self, selected, *, cycle_id, **kwargs):
            return (
                NativeGraphitiOutcome(
                    selected[0].ingest_id, "GRAPHITI_HOLD", None,
                    None if self.missing else "RETRY_PENDING",
                ),
                NativeGraphitiOutcome(
                    selected[1].ingest_id, "GRAPHITI_HOLD", None,
                    "RIGHTS_OR_PREDECESSOR_HOLD",
                ),
            )

    graphiti = Graphiti()
    pipeline._graphiti = graphiti
    try:
        pipeline.tick(cycle_id="multiple-holds")
        retained = journal.current(first.revision_id)
        assert retained["facts"]["reason"] == "MULTIPLE_GRAPHITI_HOLDS"
        assert {item["reason"] for item in retained["facts"]["graphiti_outcomes"]} == {
            "RETRY_PENDING", "RIGHTS_OR_PREDECESSOR_HOLD",
        }

        dispositions[0] = ()
        graphiti.missing = True
        pipeline.tick(cycle_id="missing-reason")
        assert journal.current(first.revision_id)["facts"]["reason"] == "ValueError"
        assert "graphiti_receipts" not in journal.current(first.revision_id)["facts"]
    finally:
        connection.close()


def test_native_pipeline_isolates_retrieval_failure_and_keeps_disappeared_work(tmp_path, monkeypatch):
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    try:
        def retrieval(selected):
            if selected[0].item_key == "one": raise RuntimeError("isolated branch fault")
            return object()
        pipeline._retrieval_for = retrieval
        report = pipeline.tick(cycle_id="first")
        assert report.revision_states == {"RETRIEVAL_HOLD": 1, "ACKNOWLEDGED": 1}
        graphiti_calls = [item for item in calls if item[0] == "graphiti"]
        pipeline._retrieval_for = lambda _: object()
        dispositions[0] = ()
        assert pipeline.tick(cycle_id="second").revision_states == {"ACKNOWLEDGED": 2}
        assert [item for item in calls if item[0] == "graphiti"] == graphiti_calls
    finally:
        connection.close()


def test_native_pipeline_preserves_ambiguous_assessment_marker(tmp_path, monkeypatch):
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    def publish(*, revision_id, candidate_version_id):
        journal.advance(revision_id, stage="ASSESSMENT_STARTED", facts=journal.current(revision_id)["facts"])
        raise OSError("interrupted after possible provider dispatch")
    pipeline._publish = NS(advance=publish)
    try:
        report = pipeline.tick(cycle_id="first")
        assert report.revision_states == {"ASSESSMENT_STARTED": 2}
    finally:
        connection.close()


def test_native_pipeline_retries_only_a_rights_evidence_hold_after_refresh(tmp_path, monkeypatch):
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    try:
        journal.land((units[0],))
        journal.advance(units[0].revision_id, stage="EVIDENCE_HOLD", facts={
            "graphiti_receipts": [{"retained": True}],
            "candidate_version_id": "candidate:one",
            "reason": "PUBLICATION_RIGHTS_HOLD",
        })
        dispositions[0] = ()
        report = pipeline.tick(cycle_id="rights-restored")
        assert report.revision_states == {"ACKNOWLEDGED": 1}
        assert calls[:2] == [("rights", "current"), ("publish", units[0].revision_id)]
    finally:
        connection.close()


def test_native_pipeline_retries_a_bounded_acquisition_hold_next_cycle(tmp_path, monkeypatch):
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    try:
        journal.land((units[0],))
        journal.advance(units[0].revision_id, stage="EVIDENCE_HOLD", facts={
            "graphiti_receipts": [{"retained": True}],
            "candidate_version_id": "candidate:one",
            "reason": "ACQUISITION_TRANSPORT_RETRY",
            "acquisition_attempt_count": 1,
            "acquisition_retryable": True,
        })
        dispositions[0] = ()
        report = pipeline.tick(cycle_id="next-cycle")
        assert report.revision_states == {"ACKNOWLEDGED": 1}
        assert calls[:2] == [("rights", "current"), ("publish", units[0].revision_id)]
    finally:
        connection.close()


@pytest.mark.parametrize("reason", (
    "ASSESSOR_CLAIM_BINDING_HOLD", "ASSESSOR_NAMED_ENTITY_CONTRACT_HOLD",
    "ASSESSOR_LOCALISATION_CONTRACT_HOLD",
    "QUALIFICATION_EVIDENCE_NOT_EXACT",
    "NO_QUALIFYING_NEW_INFORMATION",
    "EDITORIAL_ADMISSION_HOLD",
))
def test_native_pipeline_revalidates_only_repairable_holds_once_per_contract(tmp_path, monkeypatch, reason):
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    pipeline._assessment_contract_version = "new-contract"
    journal.land((units[0],))
    journal.advance(units[0].revision_id, stage="EVIDENCE_HOLD", facts={
        "graphiti_receipts": [{"retained": True}],
        "candidate_version_id": "candidate:one", "reason": reason,
        "editorial_hold_reason_codes": (
            ["INVALID_GOVERNED_CLAIM_EVIDENCE", "UNQUALIFIED_HEADLINE_CLAIM"]
            if reason == "EDITORIAL_ADMISSION_HOLD" else []
        ),
    })
    dispositions[0] = ()

    def publish(*, revision_id, candidate_version_id):
        calls.append(("revalidate", revision_id))
        journal.advance(revision_id, stage="EVIDENCE_HOLD", facts={
            **journal.current(revision_id)["facts"],
            "assessment_contract_version": pipeline._assessment_contract_version,
        })

    pipeline._publish = NS(advance=publish)
    try:
        for cycle in ("first", "unchanged"):
            pipeline.tick(cycle_id=cycle)
        expected = int(reason != "NO_QUALIFYING_NEW_INFORMATION")
        assert len([call for call in calls if call[0] == "revalidate"]) == expected
        pipeline._assessment_contract_version = "next-contract"
        pipeline.tick(cycle_id="changed-contract")
        assert len([call for call in calls if call[0] == "revalidate"]) == 2 * expected
    finally:
        connection.close()


@pytest.mark.parametrize("recent_first", (False, True))
def test_native_pipeline_time_slices_changed_contract_reassessment_without_starving_work(
    tmp_path, monkeypatch, recent_first,
):
    pipeline, journal, connection, _, calls, dispositions = _open(
        tmp_path, monkeypatch,
    )
    due = tuple(_native(f"due-{index}") for index in range(3))
    if recent_first:
        due = tuple(replace(unit, observed_at=f"2026-09-{day:02d}T16:00:00.000000Z")
                    for unit, day in zip(due, (2, 7, 5), strict=True))
    ordinary = _native("ordinary")
    fresh = _native("fresh")
    now = [0.0]
    polls = []
    pipeline._assessment_contract_version = "v8"
    pipeline._reassessment_quantum = 300
    pipeline._monotonic_clock = lambda: now[0]
    dispositions[0] = ()
    pipeline._intake = NS(poll=lambda: (polls.append(now[0]) or dispositions[0]))
    for unit in due:
        journal.land((unit,))
        journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts={
            "graphiti_receipts": [{"retained": True}],
            "candidate_version_id": "candidate:" + unit.item_key,
            "reason": "ASSESSOR_CLAIM_BINDING_HOLD",
            "assessment_contract_version": "v7",
        })
    journal.land((ordinary,))
    journal.advance(ordinary.revision_id, stage="GRAPHITI_COMPLETE", facts={
        "graphiti_receipts": [{"retained": True}],
        "candidate_version_id": "candidate:" + ordinary.item_key,
    })
    dispositions[0] = (NS(
        source_id=fresh.source_id, status="READY", reason_code="RETAINED",
        units=(fresh,),
    ),)
    original_publish = pipeline._publish

    class Publisher:
        def advance(self, *, revision_id, candidate_version_id):
            if revision_id in {unit.revision_id for unit in due}:
                if journal.current(revision_id)["stage"] == "ASSESSMENT_STARTED":
                    calls.append(("resume", revision_id))
                    journal.advance(revision_id, stage="EVIDENCE_HOLD", facts={
                        **journal.current(revision_id)["facts"],
                        "assessment_contract_version": "v8",
                    })
                    return
                calls.append(("revalidate", revision_id))
                now[0] += 301
                if revision_id == due[0].revision_id:
                    journal.advance(
                        revision_id, stage="ASSESSMENT_STARTED",
                        facts=journal.current(revision_id)["facts"],
                    )
                    return
                journal.advance(revision_id, stage="EVIDENCE_HOLD", facts={
                    **journal.current(revision_id)["facts"],
                    "assessment_contract_version": "v8",
                })
                return
            original_publish.advance(
                revision_id=revision_id,
                candidate_version_id=candidate_version_id,
            )

    pipeline._publish = Publisher()
    try:
        for cycle in ("first", "second", "third", "unchanged"):
            pipeline.tick(cycle_id=cycle)
            dispositions[0] = ()
        expected = (due[1], due[2], due[0]) if recent_first else due
        assert [revision_id for kind, revision_id in calls if kind == "revalidate"] == [
            unit.revision_id for unit in expected
        ]
        assert [revision_id for kind, revision_id in calls if kind == "resume"] == [
            due[0].revision_id
        ]
        assert len(polls) == 4
        assert journal.current(ordinary.revision_id)["stage"] == "ACKNOWLEDGED"
        assert journal.current(fresh.revision_id)["stage"] == "ACKNOWLEDGED"
    finally:
        connection.close()


@pytest.mark.parametrize("reason", ["ASSESSOR_RENDERING_CONTRACT_HOLD", "ASSESSOR_LOCALISATION_CONTRACT_HOLD", "QUALIFICATION_EVIDENCE_NOT_EXACT"])
def test_fresh_graphiti_crosses_real_queue_before_old_contract_reassessment(
    tmp_path, monkeypatch, reason,
):
    from newsroom.control_plane import cycle
    from newsroom.tests.test_graphiti_corpus_ingest import _complete
    from newsroom.tests.test_native_graphiti import _open as open_graphiti

    pipeline, journal, connection, _, calls, dispositions = _open(
        tmp_path, monkeypatch,
    )
    due = _native("due-reassessment")
    fresh = _native("fresh-graphiti")
    pipeline._assessment_contract_version = "v10"
    journal.land((due,))
    journal.advance(due.revision_id, stage="EVIDENCE_HOLD", facts={
        "graphiti_receipts": [{"retained": True}],
        "candidate_version_id": "candidate:" + due.item_key,
        "reason": reason,
        "assessment_contract_version": "v9",
    })
    dispositions[0] = (NS(
        source_id=fresh.source_id, status="READY", reason_code="RETAINED",
        units=(fresh,),
    ),)
    processor, graph_connection, _ = open_graphiti(
        tmp_path, monkeypatch, ingest=cycle._ingest,
    )
    processor._runner = NS(ingest=lambda unit: (
        calls.append(("extract", unit.revision_id))
        or _complete(unit, proposal_count=0, entity_count=0)
    ))
    pipeline._graphiti = processor
    original_publish = pipeline._publish

    class Publisher:
        def advance(self, *, revision_id, candidate_version_id):
            if revision_id == due.revision_id:
                calls.append(("revalidate", revision_id))
                journal.advance(revision_id, stage="EVIDENCE_HOLD", facts={
                    **journal.current(revision_id)["facts"],
                    "assessment_contract_version": "v10",
                })
                return
            original_publish.advance(
                revision_id=revision_id,
                candidate_version_id=candidate_version_id,
            )

    pipeline._publish = Publisher()
    try:
        pipeline.tick(cycle_id="fresh-before-reassessment")
        relevant = [call for call in calls if call[0] in {"extract", "revalidate"}]
        assert relevant == [
            ("extract", fresh.revision_id),
            ("revalidate", due.revision_id),
        ]
    finally:
        graph_connection.close()
        connection.close()


def test_native_pipeline_honours_global_stop_before_source_poll(tmp_path, monkeypatch):
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    def stop(): raise VetoError("owner stop")
    pipeline._check = stop
    try:
        with pytest.raises(VetoError): pipeline.tick(cycle_id="stopped")
        assert not journal.units and not calls
    finally:
        connection.close()


@pytest.mark.parametrize("stop_after_retained", [False, True])
def test_native_pipeline_advances_retained_work_before_new_graphiti_once(
    tmp_path, monkeypatch, stop_after_retained,
):
    pipeline, journal, connection, units, calls, _ = _open(tmp_path, monkeypatch)
    retained, pending = units
    journal.land((retained,))
    journal.advance(retained.revision_id, stage="GRAPHITI_COMPLETE", facts={
        "graphiti_receipts": [{}],
    })
    stopped = False
    original_graphiti = pipeline._graphiti

    def check():
        if stopped:
            raise VetoError("owner stop before fresh dispatch")

    def retrieval(selected):
        nonlocal stopped
        calls.append(("retrieval", selected[0].item_key))
        if selected[0].revision_id == retained.revision_id:
            stopped = stop_after_retained
            raise RuntimeError("retained downstream failure")
        return object()

    def graphiti(selected, *, cycle_id, **kwargs):
        assert selected == (pending,)
        # The failure is committed before new provider work, not only in memory.
        reopened = NativeRevisionJournal(connection)
        assert reopened.current(retained.revision_id)["stage"] == "RETRIEVAL_HOLD"
        return original_graphiti.advance(selected, cycle_id=cycle_id, **kwargs)

    pipeline._check = check
    pipeline._retrieval_for = retrieval
    pipeline._graphiti = NS(advance=graphiti)
    try:
        if stop_after_retained:
            with pytest.raises(VetoError, match="before fresh dispatch"):
                pipeline.tick(cycle_id="retained-first-stop")
            assert not any(call[0] == "graphiti" for call in calls)
        else:
            report = pipeline.tick(cycle_id="retained-first")
            assert report.revision_states == {"RETRIEVAL_HOLD": 1, "ACKNOWLEDGED": 1}
            assert [call for call in calls if call[0] in {"retrieval", "graphiti"}] == [
                ("retrieval", retained.item_key),
                ("graphiti", pending.item_key),
                ("retrieval", pending.item_key),
            ]
        assert calls.count(("retrieval", retained.item_key)) == 1
    finally:
        connection.close()


def test_ordinary_downstream_quantum_preserves_next_revision_until_next_poll(tmp_path, monkeypatch):
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    now = [0.0]
    pipeline._monotonic_clock = lambda: now[0]
    dispositions[0] = ()
    pipeline._intake = NS(poll=lambda: calls.append(("poll", "current")) or ())
    for unit in units:
        journal.land((unit,))
        journal.advance(unit.revision_id, stage="GRAPHITI_COMPLETE", facts={
            "graphiti_receipts": [{}], "candidate_version_id": "candidate:" + unit.item_key,
        })
    previous = dict(journal.current(units[1].revision_id))
    original = pipeline._publish

    def publish(**kwargs):
        original.advance(**kwargs)
        now[0] += 301

    pipeline._publish = NS(advance=publish)
    try:
        first = pipeline.tick(cycle_id="first")
        assert first.revision_states == {"ACKNOWLEDGED": 1, "GRAPHITI_COMPLETE": 1}
        assert journal.current(units[1].revision_id) == previous
        pipeline.tick(cycle_id="second")
        assert [call for call in calls if call[0] in {"poll", "publish"}] == [
            ("poll", "current"), ("publish", units[0].revision_id),
            ("poll", "current"), ("publish", units[1].revision_id),
        ]
    finally:
        connection.close()


def test_three_disjoint_turns_progress_with_revalidation_and_sustained_fresh_work(tmp_path, monkeypatch):
    pipeline, journal, connection, _, calls, _ = _open(tmp_path, monkeypatch)
    ordinary = _native("ordinary")
    due = tuple(_native(f"due-{index}") for index in range(2))
    fresh = tuple(_native(f"fresh-{index}") for index in range(3))
    now = [0.0]
    incoming = [fresh[:2]]
    pipeline._assessment_contract_version = "v9"
    pipeline._monotonic_clock = lambda: now[0]

    def poll():
        calls.append(("poll", "current"))
        return tuple(NS(source_id=unit.source_id, status="READY", reason_code="RETAINED", units=(unit,)) for unit in incoming[0])

    pipeline._intake = NS(poll=poll)
    for unit in (ordinary, *due):
        journal.land((unit,))
        journal.advance(unit.revision_id, stage="EVIDENCE_HOLD" if unit in due else "GRAPHITI_COMPLETE", facts={
            "graphiti_receipts": [{}], "candidate_version_id": "candidate:" + unit.item_key,
            "reason": "ASSESSOR_CLAIM_BINDING_HOLD", "assessment_contract_version": "v8",
        })
    original = pipeline._publish

    def publish(**kwargs):
        revision_id = kwargs["revision_id"]
        if revision_id in {unit.revision_id for unit in due}:
            calls.append(("revalidate", revision_id))
            journal.advance(revision_id, stage="EVIDENCE_HOLD", facts={
                **journal.current(revision_id)["facts"], "assessment_contract_version": "v9",
            })
        else:
            original.advance(**kwargs)
        now[0] += 301

    def graphiti(selected, *, cycle_id, defer_before_unit):
        results = []
        for unit in selected:
            if defer_before_unit(unit):
                results.append(NativeGraphitiOutcome(unit.ingest_id, "GRAPHITI_DEFERRED", None, "WORK_QUANTUM_EXHAUSTED"))
            else:
                calls.append(("extract", unit.revision_id))
                now[0] += 301
                results.append(NativeGraphitiOutcome(unit.ingest_id, "GRAPHITI_COMPLETE", unit.digest, None))
        return tuple(results)

    pipeline._publish = NS(advance=publish)
    pipeline._graphiti = NS(advance=graphiti)
    try:
        first = pipeline.tick(cycle_id="first")
        assert first.revision_states == {"ACKNOWLEDGED": 2, "EVIDENCE_HOLD": 2, "QUEUED": 1}
        assert journal.current(fresh[0].revision_id)["stage"] == "ACKNOWLEDGED"
        assert fresh[1].revision_id not in {revision: journal.current(revision) for revision, _ in journal.iter_summaries()}
        incoming[0] = (fresh[2],)
        pipeline.tick(cycle_id="second")
        incoming[0] = ()
        pipeline.tick(cycle_id="third")
        pipeline.tick(cycle_id="fourth")
        assert all(journal.current(unit.revision_id)["stage"] == "ACKNOWLEDGED" for unit in fresh)
        assert all(journal.current(unit.revision_id)["facts"]["assessment_contract_version"] == "v9" for unit in due)
        assert [call for call in calls if call[0] == "revalidate"] == [
            ("revalidate", due[0].revision_id), ("revalidate", due[1].revision_id),
        ]
        for kind in ("publish", "extract", "revalidate"):
            revisions = [revision for event, revision in calls if event == kind]
            assert len(revisions) == len(set(revisions))
    finally:
        connection.close()


def test_pending_land_order_resumes_route_hold_before_recurring_fresh_work_without_partial_progress(
    tmp_path, monkeypatch,
):
    pipeline, journal, connection, _, _calls, dispositions = _open(
        tmp_path, monkeypatch,
    )
    held = _native("old-route-held")
    fresh = _native("fresh")
    first_chunk = replace(_native("fresh-chunks"), chunk_count=2)
    second_chunk = replace(
        first_chunk, chunk_ordinal=2, predecessor_ingest_id=first_chunk.ingest_id,
    )
    recurring = tuple(_native(f"recurring-fresh-{index}") for index in range(3))
    now = [0.0]
    route_open = [True]
    completed = set()
    extracted = []
    pipeline._monotonic_clock = lambda: now[0]
    pipeline._reassessment_quantum = 300
    journal.land((held,))
    dispositions[0] = ()

    def graphiti(selected, *, cycle_id, defer_before_unit):
        results = []
        for unit in selected:
            if unit.ingest_id in completed:
                results.append(NativeGraphitiOutcome(
                    unit.ingest_id, "GRAPHITI_COMPLETE", unit.observation_digest, None,
                ))
            elif route_open[0]:
                results.append(NativeGraphitiOutcome(
                    unit.ingest_id, "GRAPHITI_HOLD", None,
                    "REQUIRED_MODEL_ROUTE_CIRCUIT_OPEN",
                ))
            elif defer_before_unit(unit):
                results.append(NativeGraphitiOutcome(
                    unit.ingest_id, "GRAPHITI_DEFERRED", None,
                    "WORK_QUANTUM_EXHAUSTED",
                ))
            else:
                extracted.append((unit.item_key, unit.chunk_ordinal))
                completed.add(unit.ingest_id)
                now[0] += 301
                results.append(NativeGraphitiOutcome(
                    unit.ingest_id, "GRAPHITI_COMPLETE", unit.observation_digest, None,
                ))
        return tuple(results)

    def fresh_poll(units):
        dispositions[0] = tuple(
            NS(source_id=unit.source_id, status="READY", reason_code="RETAINED", units=(unit,))
            for unit in units
        )

    pipeline._graphiti = NS(advance=graphiti)
    try:
        pipeline.tick(cycle_id="route-open")
        assert extracted == [] and completed == set()
        held_facts = journal.current(held.revision_id)["facts"]
        assert held_facts["reason"] == "REQUIRED_MODEL_ROUTE_CIRCUIT_OPEN"
        assert not held_facts.get("graphiti_receipts")

        route_open[0] = False
        fresh_poll((fresh, first_chunk, second_chunk))
        first = pipeline.tick(cycle_id="route-closed-oldest-first")
        assert extracted == [(held.item_key, 1)]
        assert first.unclassified_revisions == 2
        assert first.revision_states == {"ACKNOWLEDGED": 1, "QUEUED": 2}
        assert journal.current(held.revision_id)["stage"] == "ACKNOWLEDGED"
        assert fresh.revision_id not in {revision: journal.current(revision) for revision, _ in journal.iter_summaries()}
        assert first_chunk.revision_id not in {revision: journal.current(revision) for revision, _ in journal.iter_summaries()}

        fresh_poll((recurring[0],))
        pipeline.tick(cycle_id="next-landed-revision")
        assert extracted == [(held.item_key, 1), (fresh.item_key, 1)]
        assert first_chunk.revision_id not in {revision: journal.current(revision) for revision, _ in journal.iter_summaries()}
        assert recurring[0].revision_id not in {revision: journal.current(revision) for revision, _ in journal.iter_summaries()}

        fresh_poll((recurring[1],))
        pipeline.tick(cycle_id="first-chunk")
        assert extracted == [
            (held.item_key, 1), (fresh.item_key, 1), (first_chunk.item_key, 1),
        ]
        # A completed prefix plus a quantum-deferred suffix is not durable
        # revision completion (or a failure/hold invented by scheduling).
        assert first_chunk.revision_id not in {revision: journal.current(revision) for revision, _ in journal.iter_summaries()}

        fresh_poll((recurring[2],))
        pipeline.tick(cycle_id="second-chunk")
        assert extracted == [
            (held.item_key, 1), (fresh.item_key, 1),
            (first_chunk.item_key, 1), (second_chunk.item_key, 2),
        ]
        assert journal.current(first_chunk.revision_id)["stage"] == "ACKNOWLEDGED"
        assert all(unit.revision_id not in {revision: journal.current(revision) for revision, _ in journal.iter_summaries()} for unit in recurring)
    finally:
        connection.close()


@pytest.mark.parametrize("stage", ["ASSESSMENT_INTERRUPTED", "ASSESSMENT_STARTED", "PUBLICATION_STARTED"])
def test_unknown_settlement_keeps_priority_but_defers_next_unit_after_quantum(tmp_path, monkeypatch, stage):
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    ordinary = _native("ordinary")
    now = [0.0]
    pipeline._monotonic_clock = lambda: now[0]
    dispositions[0] = ()
    pipeline._intake = NS(poll=lambda: calls.append(("poll", "current")) or ())
    for unit in (ordinary, *units):
        journal.land((unit,))
        journal.advance(unit.revision_id, stage="GRAPHITI_COMPLETE" if unit == ordinary else stage, facts={
            "graphiti_receipts": [{}], "candidate_version_id": "candidate:" + unit.item_key,
        })
    previous = dict({revision: journal.current(revision) for revision, _ in journal.iter_summaries()})

    def publish(**kwargs):
        calls.append(("settle", kwargs["revision_id"]))
        now[0] += 301
        if (kwargs["revision_id"] == units[0].revision_id
                and calls.count(("settle", units[0].revision_id)) == 2):
            # A later retained settlement result, not permission to redispatch.
            journal.advance(units[0].revision_id, stage="EVIDENCE_HOLD", facts={
                **journal.current(units[0].revision_id)["facts"], "reason": "RETAINED_HOLD",
            })
        elif stage != "ASSESSMENT_INTERRUPTED":
            raise OSError("still unresolved; retain exact marker")

    pipeline._publish = NS(advance=publish)
    try:
        pipeline.tick(cycle_id="settle-first")
        assert [call for call in calls if call[0] == "settle"] == [("settle", units[0].revision_id)]
        assert now[0] == 301
        assert {revision: journal.current(revision) for revision, _ in journal.iter_summaries()} == previous
        pipeline.tick(cycle_id="settle-continuation")
        assert now[0] == 602
        assert journal.current(units[1].revision_id) == previous[units[1].revision_id]
        pipeline.tick(cycle_id="next-continuation")
        assert now[0] == 903
        assert [call for call in calls if call[0] in {"poll", "settle"}] == [
            ("poll", "current"), ("settle", units[0].revision_id),
            ("poll", "current"), ("settle", units[0].revision_id),
            ("poll", "current"), ("settle", units[1].revision_id),
        ]
        assert journal.current(ordinary.revision_id) == previous[ordinary.revision_id]
        assert journal.current(units[1].revision_id) == previous[units[1].revision_id]
        assert not any(call[0] in {"graphiti", "discovery"} for call in calls)
    finally:
        connection.close()


@pytest.mark.parametrize("recovery_seconds", [0, 300, 301])
def test_ready_canonical_revisions_progress_across_repeated_interruption_overruns(
    tmp_path, monkeypatch, recovery_seconds,
):
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    interrupted = _native("old-interrupted")
    now = [0.0]
    pipeline._monotonic_clock = lambda: now[0]
    dispositions[0] = ()
    for unit in units:
        journal.land((unit,))
        journal.advance(unit.revision_id, stage="GRAPHITI_COMPLETE", facts={
            "graphiti_receipts": [{"ingest_id": unit.ingest_id, "retained": True}],
        })
    journal.land((interrupted,))
    journal.advance(interrupted.revision_id, stage="ASSESSMENT_INTERRUPTED", facts={
        "graphiti_receipts": [{"retained": True}],
        "candidate_version_id": "candidate:old-interrupted",
    })
    retained = dict(journal.current(interrupted.revision_id))
    receipts = {unit.revision_id: journal.current(unit.revision_id)["facts"]["graphiti_receipts"] for unit in units}
    original = pipeline._publish

    def publish(**kwargs):
        if kwargs["revision_id"] == interrupted.revision_id:
            calls.append(("recover", interrupted.revision_id))
            now[0] += recovery_seconds
        else:
            original.advance(**kwargs)
            now[0] += 301

    pipeline._publish = NS(advance=publish)
    try:
        first = pipeline.tick(cycle_id="ready-first")
        if recovery_seconds == 0:
            # The first ready revision exhausts ordinary work; the second uses
            # the existing final quantum, without repeating the first advance.
            assert first.revision_states == {"ACKNOWLEDGED": 2, "ASSESSMENT_INTERRUPTED": 1}
        else:
            assert first.revision_states == {"ACKNOWLEDGED": 1, "GRAPHITI_COMPLETE": 1, "ASSESSMENT_INTERRUPTED": 1}
            assert journal.current(units[1].revision_id)["facts"].get("candidate_version_id") is None
        pipeline.tick(cycle_id="ready-next")
        assert journal.current(interrupted.revision_id) == retained
        assert all(journal.current(unit.revision_id)["stage"] == "ACKNOWLEDGED" for unit in units)
        assert all(journal.current(unit.revision_id)["facts"]["graphiti_receipts"] == receipts[unit.revision_id] for unit in units)
        first_ready = [("discovery", units[0].item_key), ("publish", units[0].revision_id)]
        next_ready = [("discovery", units[1].item_key), ("publish", units[1].revision_id)]
        recover = [("recover", interrupted.revision_id)]
        expected = recover + first_ready + next_ready + recover if recovery_seconds == 0 else recover + first_ready + recover + next_ready
        assert [call for call in calls if call[0] in {"recover", "discovery", "publish"}] == expected
        assert not any(call[0] == "graphiti" for call in calls)
    finally:
        connection.close()


def test_deferred_ready_revision_precedes_changed_producer_reassessment(tmp_path, monkeypatch):
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    ready = replace(units[0], observed_at="2026-09-01T12:00:00Z")
    interrupted, due = _native("interrupted"), replace(_native("due"), observed_at="2026-09-29T12:00:00Z")
    now = [0.0]
    pipeline._monotonic_clock = lambda: now[0]
    pipeline._assessment_contract_version = "v9"
    dispositions[0] = ()
    for unit, stage, facts in (
        (ready, "GRAPHITI_COMPLETE", {}),
        (interrupted, "ASSESSMENT_INTERRUPTED", {"candidate_version_id": "candidate:interrupted"}),
        (due, "EVIDENCE_HOLD", {"candidate_version_id": "candidate:due", "assessment_contract_version": "v8",
                                "reason": "ASSESSOR_CLAIM_BINDING_HOLD"}),
    ):
        journal.land((unit,))
        journal.advance(unit.revision_id, stage=stage, facts={"graphiti_receipts": [{"retained": True}], **facts})
    original = pipeline._publish

    def publish(**kwargs):
        revision_id = kwargs["revision_id"]
        if revision_id == interrupted.revision_id:
            calls.append(("recover", revision_id))
        elif revision_id == due.revision_id:
            calls.append(("revalidate", revision_id))
            journal.advance(revision_id, stage="EVIDENCE_HOLD", facts={
                **journal.current(revision_id)["facts"], "assessment_contract_version": "v9",
            })
        else:
            original.advance(**kwargs)
        now[0] += 301

    pipeline._publish = NS(advance=publish)
    try:
        pipeline.tick(cycle_id="deferred-ready-first")
        assert journal.current(ready.revision_id)["stage"] == "ACKNOWLEDGED"
        assert journal.current(due.revision_id)["facts"]["assessment_contract_version"] == "v8"
        pipeline.tick(cycle_id="reassessment-next")
        assert [call for call in calls if call[0] in {"recover", "publish", "revalidate"}] == [
            ("recover", interrupted.revision_id), ("publish", ready.revision_id),
            ("recover", interrupted.revision_id), ("revalidate", due.revision_id),
        ]
        assert not any(call[0] == "graphiti" for call in calls)
    finally:
        connection.close()


@pytest.mark.parametrize("graphiti_complete", [False, True])
def test_deferred_ready_work_and_missing_receipts_keep_disjoint_graphiti_turns(
    tmp_path, monkeypatch, graphiti_complete,
):
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    ready, missing = units
    interrupted = _native("interrupted")
    now = [0.0]
    pipeline._monotonic_clock = lambda: now[0]
    dispositions[0] = ()
    for unit in (ready, missing, interrupted):
        journal.land((unit,))
    journal.advance(ready.revision_id, stage="GRAPHITI_COMPLETE", facts={"graphiti_receipts": [{"retained": True}]})
    journal.advance(missing.revision_id, stage="GRAPHITI_COMPLETE", facts={})
    journal.advance(interrupted.revision_id, stage="ASSESSMENT_INTERRUPTED", facts={
        "graphiti_receipts": [{"retained": True}], "candidate_version_id": "candidate:interrupted",
    })
    original = pipeline._publish

    def publish(**kwargs):
        if kwargs["revision_id"] == interrupted.revision_id:
            calls.append(("recover", interrupted.revision_id))
            now[0] += 301
        else:
            original.advance(**kwargs)

    def graphiti(selected, *, cycle_id, **kwargs):
        assert selected == (missing,)
        calls.append(("graphiti", missing.item_key))
        return (NativeGraphitiOutcome(
            missing.ingest_id, "GRAPHITI_COMPLETE" if graphiti_complete else "GRAPHITI_HOLD",
            missing.digest if graphiti_complete else None,
            None if graphiti_complete else "REQUIRED_MODEL_ROUTE_CIRCUIT_OPEN",
        ),)

    pipeline._publish = NS(advance=publish)
    pipeline._graphiti = NS(advance=graphiti)
    try:
        pipeline.tick(cycle_id="disjoint-ready-and-pending")
        assert journal.current(ready.revision_id)["stage"] == "ACKNOWLEDGED"
        assert calls.count(("graphiti", missing.item_key)) == 1
        assert calls.count(("publish", ready.revision_id)) == 1
        assert calls.count(("discovery", ready.item_key)) == 1
        assert calls.count(("publish", missing.revision_id)) == int(graphiti_complete)
        assert calls.count(("discovery", missing.item_key)) == int(graphiti_complete)
        if graphiti_complete:
            assert journal.current(missing.revision_id)["stage"] == "ACKNOWLEDGED"
            assert calls.index(("publish", missing.revision_id)) < calls.index(("publish", ready.revision_id))
        else:
            assert journal.current(missing.revision_id)["stage"] == "GRAPHITI_HOLD"
            assert not journal.current(missing.revision_id)["facts"].get("graphiti_receipts")
            assert journal.current(missing.revision_id)["facts"].get("candidate_version_id") is None
    finally:
        connection.close()


@pytest.mark.parametrize("signal", ["stop", "drain"])
def test_deferred_ready_work_honours_stop_and_drain_before_continuation(tmp_path, monkeypatch, signal):
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    ready, interrupted = units
    now, signalled = [0.0], [False]
    pipeline._monotonic_clock = lambda: now[0]
    dispositions[0] = ()
    for unit, stage, facts in (
        (ready, "GRAPHITI_COMPLETE", {}),
        (interrupted, "ASSESSMENT_INTERRUPTED", {"candidate_version_id": "candidate:two"}),
    ):
        journal.land((unit,))
        journal.advance(unit.revision_id, stage=stage, facts={"graphiti_receipts": [{"retained": True}], **facts})
    retained = dict(journal.current(ready.revision_id))

    def recover(**kwargs):
        calls.append(("recover", kwargs["revision_id"]))
        now[0] += 301
        signalled[0] = True

    def check():
        if signalled[0] and signal == "stop":
            raise VetoError("owner stop after atomic recovery")

    pipeline._publish = NS(advance=recover)
    pipeline._check = check
    pipeline._operator_drain_requested = lambda: signalled[0] and signal == "drain"
    try:
        with pytest.raises(VetoError if signal == "stop" else OperatorDrainRequested):
            pipeline.tick(cycle_id="stopped-before-deferred-ready")
        assert journal.current(ready.revision_id) == retained
        assert [call for call in calls if call[0] in {"recover", "discovery", "publish", "graphiti"}] == [
            ("recover", interrupted.revision_id),
        ]
    finally:
        connection.close()


def _overrunning_ready_spill(tmp_path, monkeypatch):
    pipeline, journal, connection, _, calls, dispositions = _open(tmp_path, monkeypatch)
    now = [0.0]
    pipeline._monotonic_clock = lambda: now[0]
    dispositions[0] = ()
    interrupted = _native("permanent-interrupted")
    journal.land((interrupted,))
    journal.advance(interrupted.revision_id, stage="ASSESSMENT_INTERRUPTED", facts={
        "graphiti_receipts": [{"retained": True}], "candidate_version_id": "candidate:interrupted",
    })
    original = pipeline._publish

    def publish(**kwargs):
        if kwargs["revision_id"] == interrupted.revision_id:
            calls.append(("recover", interrupted.revision_id))
        else:
            original.advance(**kwargs)
        now[0] += 301

    pipeline._publish = NS(advance=publish)
    return pipeline, journal, connection, calls, now


@pytest.mark.parametrize("fresh_seconds", [0, 301])
def test_ready_spill_alternates_current_and_archive_despite_fresh_turns(tmp_path, monkeypatch, fresh_seconds):
    pipeline, journal, connection, calls, now = _overrunning_ready_spill(tmp_path, monkeypatch)
    archives = tuple(replace(_native(f"archive-{index}"), published_at=f"201{index}-01-01T00:00:00Z") for index in range(3))
    weekly = replace(_native("weekly"), published_at="2024-05-10T00:00:00Z", updated_at="2026-09-29T13:28:01Z")
    for unit in (*archives, weekly):
        journal.land((unit,))
        journal.advance(unit.revision_id, stage="GRAPHITI_COMPLETE", facts={"graphiti_receipts": [{"retained": True}]})
    retained = {key: dict(value) for key, value in journal.iter_summaries()}
    incoming = [None]
    pipeline._intake = NS(poll=lambda: (NS(source_id=incoming[0].source_id, status="READY", reason_code="RETAINED", units=(incoming[0],)),))

    def graphiti(selected, **kwargs):
        now[0] += fresh_seconds
        return tuple(NativeGraphitiOutcome(unit.ingest_id, "GRAPHITI_COMPLETE", unit.digest, None) for unit in selected)

    pipeline._graphiti = NS(advance=graphiti)
    fresh = tuple(replace(_native(f"incoming-{index}"), updated_at=f"2026-10-0{index + 1}T00:00:00Z") for index in range(6))
    spill_turns = []
    try:
        for index, unit in enumerate(fresh):
            incoming[0] = unit
            start = len(calls)
            pipeline.tick(cycle_id=f"fair-spill-{index}")
            publications = [revision for kind, revision in calls[start:] if kind == "publish"]
            assert len(publications) == len(set(publications))
            spill_turns.append(publications[-1])
        # Freshly completed work now joins this same already-budgeted spill.
        expected_current = fresh[0] if fresh_seconds else weekly
        assert spill_turns[:2] == [expected_current.revision_id, archives[0].revision_id]
        assert all(journal.current(unit.revision_id)["stage"] == "ACKNOWLEDGED" for unit in archives)
        if fresh_seconds:
            assert spill_turns[2::2] == [fresh[2].revision_id, fresh[4].revision_id]
            assert journal.current(weekly.revision_id)['stage'] == 'GRAPHITI_COMPLETE'
        for unit in (*archives, weekly):
            assert journal.current(unit.revision_id)["facts"]["graphiti_receipts"] == retained[unit.revision_id]["facts"]["graphiti_receipts"]
        interrupted = next(key for key, value in retained.items() if value["stage"] == "ASSESSMENT_INTERRUPTED")
        assert journal.current(interrupted) == retained[interrupted]
    finally:
        connection.close()


def test_ready_spill_empty_quantum_does_not_consume_current_turn(tmp_path, monkeypatch):
    pipeline, journal, connection, calls, now = _overrunning_ready_spill(tmp_path, monkeypatch)
    units = tuple(replace(_native(f"ready-{index}"), updated_at=f"202{index}-01-01T00:00:00Z") for index in range(2))
    for unit in units:
        journal.land((unit,))
        journal.advance(unit.revision_id, stage="GRAPHITI_COMPLETE", facts={"graphiti_receipts": [{}]})
    original, turns = pipeline._advance_revisions, [0]

    def advance(revisions, **kwargs):
        turns[0] += 1
        if turns[0] == 3:
            now[0] += 301
        return original(revisions, **kwargs)

    pipeline._advance_revisions = advance
    try:
        pipeline.tick(cycle_id="expired-before-spill")
        assert not any(kind == "publish" for kind, _ in calls)
        assert all(journal.current(unit.revision_id)["stage"] == "GRAPHITI_COMPLETE" for unit in units)
        pipeline.tick(cycle_id="first-real-spill")
        pipeline.tick(cycle_id="archive-spill")
        assert [revision for kind, revision in calls if kind == "publish"] == [units[1].revision_id, units[0].revision_id]
    finally:
        connection.close()


@pytest.mark.parametrize("dates,first", [
    (((None, "2021-01-01T00:00:00Z"), ("2026-01-01T00:00:00Z", "2020-01-01T00:00:00Z")), 1),
    (((None, "2021-01-01T00:00:00Z"), ("malformed", "2026-01-01T00:00:00Z")), 1),
    ((("2024-01-01T00:00:00Z", None), ("2024-01-01T00:00:00Z", None)), 0),
    (((None, None), ("malformed", None)), 0),
])
def test_ready_spill_source_dates_fall_back_without_using_observation_time(tmp_path, monkeypatch, dates, first):
    pipeline, journal, connection, calls, _ = _overrunning_ready_spill(tmp_path, monkeypatch)
    units = tuple(replace(_native(f"dated-{index}"), updated_at=updated, published_at=published,
                          observed_at=f"2026-09-{index + 1:02}T00:00:00Z") for index, (updated, published) in enumerate(dates))
    for unit in units:
        journal.land((unit,))
        journal.advance(unit.revision_id, stage="GRAPHITI_COMPLETE", facts={"graphiti_receipts": [{}]})
    try:
        pipeline.tick(cycle_id="source-dated-spill")
        assert [revision for kind, revision in calls if kind == "publish"] == [units[first].revision_id]
    finally:
        connection.close()


def _quantum_pending_graphiti(pipeline, journal, now):
    completed, extracted = set(), []

    def graphiti(selected, *, cycle_id, defer_before_unit):
        expected = tuple(
            unit for revision_id, units in journal.units.items()
            if not journal.current(revision_id).get("facts", {}).get("graphiti_receipts")
            for unit in units
        )
        assert {unit.ingest_id for unit in selected} == {unit.ingest_id for unit in expected}
        for revision_id in dict.fromkeys(unit.revision_id for unit in selected):
            members = tuple(unit for unit in selected if unit.revision_id == revision_id)
            assert tuple(unit.chunk_ordinal for unit in members) == tuple(range(1, members[0].chunk_count + 1))
        results = []
        for unit in selected:
            if unit.ingest_id in completed:
                results.append(NativeGraphitiOutcome(unit.ingest_id, "GRAPHITI_COMPLETE", unit.observation_digest, None))
            elif defer_before_unit(unit):
                results.append(NativeGraphitiOutcome(unit.ingest_id, "GRAPHITI_DEFERRED", None, "WORK_QUANTUM_EXHAUSTED"))
            else:
                extracted.append((unit.item_key, unit.chunk_ordinal))
                completed.add(unit.ingest_id)
                now[0] += 301
                results.append(NativeGraphitiOutcome(unit.ingest_id, "GRAPHITI_COMPLETE", unit.observation_digest, None))
        return tuple(results)

    pipeline._graphiti = NS(advance=graphiti)
    return completed, extracted


def test_pending_alternates_source_recency_and_land_order_without_dropping_chunks(tmp_path, monkeypatch):
    pipeline, journal, connection, _, _, dispositions = _open(tmp_path, monkeypatch)
    now = [0.0]
    pipeline._monotonic_clock = lambda: now[0]
    first_chunk = replace(_native("archive-chunks"), chunk_count=2,
                          published_at="2010-01-01T00:00:00Z", updated_at=None,
                          observed_at="2026-10-02T00:00:00Z")
    second_chunk = replace(first_chunk, chunk_ordinal=2, predecessor_ingest_id=first_chunk.ingest_id)
    archive = replace(_native("archive"), published_at="2011-01-01T00:00:00Z", updated_at=None)
    weekly = replace(_native("weekly"), published_at="2024-05-10T00:00:00Z",
                     updated_at="2026-10-01T00:00:00Z", observed_at="2026-09-01T00:00:00Z")
    recent = replace(_native("recent"), published_at="2026-09-30T00:00:00Z", updated_at=None)
    journal.land((second_chunk, first_chunk))
    for unit in (archive, weekly, recent):
        journal.land((unit,))
    journal.advance(archive.revision_id, stage="GRAPHITI_HOLD", facts={"reason": "RETRY_PENDING"})
    previous_hold = dict(journal.current(archive.revision_id))
    dispositions[0] = ()
    _, extracted = _quantum_pending_graphiti(pipeline, journal, now)
    incoming = tuple(replace(_native(f"incoming-{index}"), updated_at=f"2026-10-{index + 2:02}T00:00:00Z")
                     for index in range(5))
    try:
        for index in range(6):
            if index:
                unit = incoming[index - 1]
                dispositions[0] = (NS(source_id=unit.source_id, status="READY", reason_code="RETAINED", units=(unit,)),)
            pipeline.tick(cycle_id=f"fair-pending-{index}")
            if index < 3:
                assert first_chunk.revision_id not in {revision: journal.current(revision) for revision, _ in journal.iter_summaries()}
            if index < 5:
                assert journal.current(archive.revision_id) == previous_hold
        assert extracted == [
            (weekly.item_key, 1), (first_chunk.item_key, 1), (incoming[1].item_key, 1),
            (second_chunk.item_key, 2), (incoming[3].item_key, 1), (archive.item_key, 1),
        ]
        assert journal.current(first_chunk.revision_id)["facts"]["graphiti_receipts"] == [
            {"ingest_id": unit.ingest_id, "state": "GRAPHITI_COMPLETE", "receipt_digest": unit.observation_digest, "reason": None}
            for unit in (first_chunk, second_chunk)
        ]
        assert set(journal.units) == {unit.revision_id for unit in (first_chunk, archive, weekly, recent, *incoming)}
        assert all(unit.revision_id not in {revision: journal.current(revision) for revision, _ in journal.iter_summaries()} for unit in (recent, incoming[0], incoming[2], incoming[4]))
    finally:
        connection.close()


@pytest.mark.parametrize("cached_prefix", [False, True])
def test_pending_empty_quantum_and_cached_prefix_do_not_consume_current_turn(tmp_path, monkeypatch, cached_prefix):
    pipeline, journal, connection, _, _, dispositions = _open(tmp_path, monkeypatch)
    now = [0.0]
    pipeline._monotonic_clock = lambda: now[0]
    dispositions[0] = ()
    first_chunk = replace(_native("old-chunks"), chunk_count=2, updated_at="2020-01-01T00:00:00Z")
    second_chunk = replace(first_chunk, chunk_ordinal=2, predecessor_ingest_id=first_chunk.ingest_id)
    current = replace(_native("current"), updated_at="2026-10-01T00:00:00Z")
    journal.land((first_chunk, second_chunk))
    journal.land((current,))
    completed, extracted = _quantum_pending_graphiti(pipeline, journal, now)
    if cached_prefix:
        completed.add(first_chunk.ingest_id)
    original, calls = pipeline._graphiti.advance, [0]

    def expire_first_quantum(selected, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            now[0] += 301
        return original(selected, **kwargs)

    pipeline._graphiti = NS(advance=expire_first_quantum)
    try:
        pipeline.tick(cycle_id="expired-before-pending")
        assert extracted == [] and {revision: journal.current(revision) for revision, _ in journal.iter_summaries()} == {}
        assert pipeline._spill_archive_turn is False
        pipeline.tick(cycle_id="first-real-pending")
        pipeline.tick(cycle_id="archive-pending")
        assert extracted[:2] == [(current.item_key, 1), (first_chunk.item_key, 2 if cached_prefix else 1)]
        if not cached_prefix:
            assert first_chunk.revision_id not in {revision: journal.current(revision) for revision, _ in journal.iter_summaries()}
    finally:
        connection.close()


def test_pending_and_ready_spill_share_preference_and_toggle_only_once_per_tick(tmp_path, monkeypatch):
    pipeline, journal, connection, calls, now = _overrunning_ready_spill(tmp_path, monkeypatch)
    spill = tuple(replace(_native(f"spill-{index}"), updated_at=date) for index, date in enumerate(
        ("2020-01-01T00:00:00Z", "2026-10-04T00:00:00Z"),
    ))
    pending = tuple(replace(_native(f"pending-{index}"), updated_at=date) for index, date in enumerate(
        ("2010-01-01T00:00:00Z", "2026-10-01T00:00:00Z", "2026-10-02T00:00:00Z"),
    ))
    for unit in spill:
        journal.land((unit,))
        journal.advance(unit.revision_id, stage="GRAPHITI_COMPLETE", facts={"graphiti_receipts": [{"retained": True}]})
    for unit in pending:
        journal.land((unit,))
    _, extracted = _quantum_pending_graphiti(pipeline, journal, now)
    try:
        pipeline.tick(cycle_id="current-pending-and-spill")
        assert extracted == [(pending[2].item_key, 1)]
        assert [revision for kind, revision in calls if kind == "publish"] == [spill[1].revision_id]
        assert pipeline._spill_archive_turn is True
        pipeline.tick(cycle_id="archive-pending-and-spill")
        assert extracted == [(pending[2].item_key, 1), (pending[0].item_key, 1)]
        assert [revision for kind, revision in calls if kind == "publish"] == [spill[1].revision_id, spill[0].revision_id]
        assert pipeline._spill_archive_turn is False
        assert pending[1].revision_id not in {revision: journal.current(revision) for revision, _ in journal.iter_summaries()}
    finally:
        connection.close()


def test_ordinary_current_turn_precedes_archive_and_next_archive_turn_preserves_history(tmp_path, monkeypatch):
    pipeline, journal, connection, _, calls, dispositions = _open(tmp_path, monkeypatch)
    now = [0.0]
    pipeline._monotonic_clock = lambda: now[0]
    dispositions[0] = ()
    archive = replace(_native("ordinary-archive"), published_at="2021-01-01T00:00:00Z", updated_at=None,
                      observed_at="2026-10-02T00:00:00Z")
    current = replace(_native("ordinary-current"), published_at="2026-10-01T00:00:00Z", updated_at=None,
                      observed_at="2026-09-01T00:00:00Z")
    pending = _native("ordinary-pending")
    for unit in (archive, current):
        journal.land((unit,))
        journal.advance(unit.revision_id, stage="CANDIDATE_ADMITTED", facts={
            "candidate_version_id": "candidate:" + unit.item_key,
            "graphiti_receipts": [{"retained": True}],
        })
    journal.land((pending,))
    original = pipeline._publish.advance
    def publish(**kwargs):
        result = original(**kwargs)
        now[0] += 301
        return result
    pipeline._publish = NS(advance=publish)
    def graphiti(units, *, defer_before_unit, **kwargs):
        for unit in units:
            defer_before_unit(unit)
        return tuple(NativeGraphitiOutcome(unit.ingest_id, "GRAPHITI_DEFERRED", None, "ROUTE_HOLD") for unit in units)
    pipeline._graphiti = NS(advance=graphiti)
    try:
        pipeline.tick(cycle_id="ordinary-current")
        assert [revision for kind, revision in calls if kind == "publish"] == [current.revision_id]
        assert journal.current(archive.revision_id)["stage"] == "CANDIDATE_ADMITTED"
        assert pipeline._spill_archive_turn is True
        pipeline.tick(cycle_id="ordinary-archive")
        assert [revision for kind, revision in calls if kind == "publish"] == [current.revision_id, archive.revision_id]
        assert set(journal.units) == {archive.revision_id, current.revision_id, pending.revision_id}
    finally:
        connection.close()


def test_ordinary_current_turn_settles_unknown_before_fresh_ready_work(tmp_path, monkeypatch):
    pipeline, journal, connection, _, calls, dispositions = _open(tmp_path, monkeypatch)
    dispositions[0] = ()
    unknown = replace(_native("ordinary-unknown"), published_at="2021-01-01T00:00:00Z", updated_at=None)
    current = replace(_native("ordinary-fresh"), published_at="2026-10-01T00:00:00Z", updated_at=None)
    for unit, stage in ((current, "CANDIDATE_ADMITTED"), (unknown, "ASSESSMENT_INTERRUPTED")):
        journal.land((unit,))
        journal.advance(unit.revision_id, stage=stage, facts={
            "candidate_version_id": "candidate:" + unit.item_key,
            "graphiti_receipts": [{"retained": True}],
        })
    try:
        pipeline.tick(cycle_id="ordinary-unknown-first")
        assert [revision for kind, revision in calls if kind == "publish"] == [unknown.revision_id, current.revision_id]
    finally:
        connection.close()


def test_ordinary_only_progress_alternates_without_a_pending_cohort(tmp_path, monkeypatch):
    pipeline, journal, connection, _, calls, dispositions = _open(tmp_path, monkeypatch)
    now = [0.0]
    pipeline._monotonic_clock = lambda: now[0]
    dispositions[0] = ()
    archive = replace(_native("ordinary-only-archive"), updated_at="2021-01-01T00:00:00Z")
    current = replace(_native("ordinary-only-current"), updated_at="2026-10-01T00:00:00Z")
    for unit in (archive, current):
        journal.land((unit,))
        journal.advance(unit.revision_id, stage="CANDIDATE_ADMITTED", facts={
            "candidate_version_id": "candidate:" + unit.item_key,
            "graphiti_receipts": [{"retained": True}],
        })
    original = pipeline._publish.advance
    def publish(**kwargs):
        result = original(**kwargs)
        now[0] += 301
        return result
    pipeline._publish = NS(advance=publish)
    try:
        pipeline.tick(cycle_id="ordinary-only-current")
        assert pipeline._spill_archive_turn is True
        pipeline.tick(cycle_id="ordinary-only-archive")
        assert [revision for kind, revision in calls if kind == "publish"] == [current.revision_id, archive.revision_id]
        assert pipeline._spill_archive_turn is False
    finally:
        connection.close()


def test_ordinary_unchanged_hold_does_not_consume_current_preference(tmp_path, monkeypatch):
    pipeline, journal, connection, units, _, dispositions = _open(tmp_path, monkeypatch)
    dispositions[0] = ()
    unit = units[0]
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="CANDIDATE_ADMITTED", facts={
        "candidate_version_id": "candidate:" + unit.item_key,
        "graphiti_receipts": [{"retained": True}],
    })
    before = journal.current(unit.revision_id)
    pipeline._publish = NS(advance=lambda **kwargs: None)
    try:
        pipeline.tick(cycle_id="ordinary-unchanged")
        assert journal.current(unit.revision_id) == before
        assert pipeline._spill_archive_turn is False
    finally:
        connection.close()


@pytest.mark.parametrize("mutate_public_ordinal", (False, True))
def test_ordinary_deduplicated_journal_hold_does_not_consume_turn(tmp_path, monkeypatch, mutate_public_ordinal):
    pipeline, journal, connection, units, _, dispositions = _open(tmp_path, monkeypatch)
    dispositions[0] = ()
    unit = units[0]
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="DISCOVERY_HOLD", facts={
        "graphiti_receipts": [{"retained": True}], "reason": "SOURCE_HOLD",
    })
    before_ordinal = journal._records[unit.revision_id].ordinal
    if mutate_public_ordinal:
        journal.current(unit.revision_id)["ordinal"] = 999
    pipeline._discovery = NS(
        deliver=lambda unit, **kwargs: unit,
        admit_lead=lambda *args, **kwargs: NS(lead=None, phase=NS(value="SOURCE_HOLD")),
    )
    try:
        pipeline.tick(cycle_id="ordinary-deduplicated")
        assert journal._records[unit.revision_id].ordinal == before_ordinal
        assert pipeline._spill_archive_turn is False
    finally:
        connection.close()


def test_ordinary_recovery_progress_consumes_turn_before_checked_partition_removal(tmp_path, monkeypatch):
    pipeline, journal, connection, units, _, dispositions = _open(tmp_path, monkeypatch)
    dispositions[0] = ()
    unit = units[0]
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="ASSESSMENT_INTERRUPTED", facts={
        "candidate_version_id": "candidate:" + unit.item_key,
        "graphiti_receipts": [{"retained": True}], "failure_class": "NativeEvidenceError",
    })
    def recover(revision_ids, *, before_revision):
        assert before_revision()
        journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts={
            **journal.current(unit.revision_id)["facts"], "reason": "ASSESSOR_PRE_DISPATCH_HOLD",
        })
        return (unit.revision_id,)
    pipeline._publish = NS(recover_pre_dispatch=recover, advance=lambda **kwargs: pytest.fail("duplicate ordinary turn"))
    try:
        pipeline.tick(cycle_id="ordinary-recovery-turn")
        assert journal._records[unit.revision_id].ordinal == 2
        assert pipeline._spill_archive_turn is True
    finally:
        connection.close()


class _ColdPairJournal:
    """Explicit journal API double: a global caller must never ask for cold pairs."""
    def __init__(self, states, units):
        import copy
        self._states = copy.deepcopy(states)
        self.units = units
        self.portfolio = ()
        self.current_reads = []
        self.writes = []
        self._read_before_write = set()

    def summary(self, revision_id):
        import copy
        value = copy.deepcopy(self._states.get(revision_id, {}))
        for key in ("retrieval_binding", "retrieval_rights_inventory"):
            value.get("facts", {}).pop(key, None)
        return value

    def iter_summaries(self):
        for revision_id in tuple(self._states):
            yield revision_id, self.summary(revision_id)

    def current(self, revision_id):
        import copy
        self.current_reads.append(revision_id)
        self._read_before_write.add(revision_id)
        return copy.deepcopy(self._states.get(revision_id, {}))

    def progress_ordinal(self, revision_id):
        return self._states.get(revision_id, {}).get("ordinal")

    def sources(self, dispositions):
        assert not dispositions

    def advance(self, revision_id, *, stage, facts):
        import copy
        assert revision_id in self._read_before_write, "STATE write did not select full facts"
        self._read_before_write.remove(revision_id)
        previous = self._states.get(revision_id, {}).get("facts", {})
        for key in ("retrieval_binding", "retrieval_rights_inventory", "unknown_inline"):
            if key in previous:
                assert facts[key] == previous[key], "thin facts lost a retained pair or unknown fact"
        value = dict(revision_id=revision_id, ordinal=len(self.writes) + 2, stage=stage, facts=copy.deepcopy(facts))
        self._states[revision_id] = value
        self.writes.append(value)
        return copy.deepcopy(value)


def test_global_terminal_routing_uses_inline_summaries_without_cold_pair_reads():
    states = {
        "held": {"revision_id": "held", "ordinal": 1, "stage": "EVIDENCE_HOLD", "facts": {
            "reason": "NO_QUALIFYING_NEW_INFORMATION", "candidate_version_id": "held-version",
            "graphiti_receipts": [{}], "retrieval_binding": {"cold": [1, 2]},
            "retrieval_rights_inventory": {"cold": [3, 4]},
        }},
        "ack": {"revision_id": "ack", "ordinal": 1, "stage": "ACKNOWLEDGED", "facts": {
            "writer_id": "newsroom.offline-exact-copy.v3", "graphiti_receipts": [{}],
            "retrieval_binding": {"cold": [5]}, "retrieval_rights_inventory": {"cold": [6]},
        }},
    }
    units = {key: (replace(_native(key), updated_at="2026-09-08T12:00:00Z", published_at="2026-09-08T12:00:00Z"),) for key in states}
    journal = _ColdPairJournal(states, units)
    pipeline = n.NativePipeline(
        runtime=NS(authority=object(), proof=object()), journal=journal,
        source_intake=NS(poll=lambda: ()), graphiti=object(), discovery=object(),
        retrieval_for=lambda _: pytest.fail("terminal routing entered retrieval"), collision=object(),
        publish=NS(copy_correction_due=lambda facts: False), actor_identity_digest="sha256:" + "a" * 64,
        stop_check=lambda: None, stop_fence=nullcontext,
    )
    assert pipeline.tick(cycle_id="summary-only").revision_states == {"EVIDENCE_HOLD": 1, "ACKNOWLEDGED": 1}
    assert journal.current_reads == [] and journal.writes == []


def test_selected_candidate_write_carries_full_pair_and_unknown_facts(monkeypatch):
    unit = replace(_native("selected"), updated_at="2026-09-08T12:00:00Z", published_at="2026-09-08T12:00:00Z")
    facts = {"graphiti_receipts": [{}], "retrieval_binding": {"documents": [1, 2]},
             "retrieval_rights_inventory": {"rights": [3]}, "unknown_inline": {"future": True}}
    journal = _ColdPairJournal({"selected": {"revision_id": "selected", "ordinal": 1, "stage": "GRAPHITI_COMPLETE", "facts": facts}}, {"selected": (unit,)})
    monkeypatch.setattr(n, "advance_native_cycle", lambda **_: (
        NS(revision_id="selected", state="CANDIDATE_ADMITTED", triage=NS(candidate=NS(version_id="candidate-selected"))),
    ))
    def publish(**_):
        current = journal.current("selected")
        journal.advance("selected", stage="ACKNOWLEDGED", facts=current["facts"])
    pipeline = n.NativePipeline(
        runtime=NS(authority=object(), proof=object()), journal=journal,
        source_intake=NS(poll=lambda: ()), graphiti=object(),
        discovery=NS(deliver=lambda *_args, **_kw: unit, admit_lead=lambda *_args, **_kw: NS(lead=unit)),
        retrieval_for=lambda _: object(), collision=object(), publish=NS(advance=publish),
        actor_identity_digest="sha256:" + "a" * 64, stop_check=lambda: None, stop_fence=nullcontext,
    )
    assert pipeline.tick(cycle_id="selected-full").revision_states == {"ACKNOWLEDGED": 1}
    assert len(journal.writes) == 2
    assert journal.current("selected")["facts"]["retrieval_binding"] == facts["retrieval_binding"]


def test_retained_inventory_hold_gets_one_current_consumer_turn(tmp_path, monkeypatch):
    from newsroom.control_plane.admission import WRITE_ADMISSION_POLICY_VERSION
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    unit = units[0]
    dispositions[0] = ()
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts={
        "candidate_version_id": "candidate:" + unit.item_key, "graphiti_receipts": [{}],
        "reason": "INVALID_SUBSTANTIVE_CLAIM_INVENTORY", "package_admission_id": "retained-package",
        "editorial_decision": {"decision_id": "retained-decision"}})
    attempted = []
    def advance(**request):
        attempted.append(request)
        facts = journal.current(unit.revision_id)["facts"]
        journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts={**facts,
            "write_admission_policy_version": WRITE_ADMISSION_POLICY_VERSION})
    pipeline._publish = NS(advance=advance)
    try:
        pipeline.tick(cycle_id="current-consumer")
        assert len(attempted) == 1
        pipeline._journal = NativeRevisionJournal(connection)
        pipeline.tick(cycle_id="same-consumer-reopened")
        assert len(attempted) == 1
        assert journal.current(unit.revision_id)["facts"]["candidate_version_id"] == "candidate:" + unit.item_key
    finally:
        connection.close()


@pytest.mark.parametrize("sink_fails", (False, True))
def test_unknown_continuation_failure_reports_only_site_without_changing_intent(tmp_path, monkeypatch, sink_fails):
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    unit = units[0]
    dispositions[0] = ()
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="PUBLICATION_STARTED", facts={
        "candidate_version_id": "candidate:" + unit.item_key, "graphiti_receipts": [{}],
        "publication_started_at": "2026-10-03T07:00:00Z"})
    before = journal.current(unit.revision_id)
    def failed(**_request):
        raise RuntimeError("PRIVATE_SOURCE_PROVIDER_TEXT_MUST_NOT_BE_LOGGED")
    pipeline._publish = NS(advance=failed)
    events = []
    def diagnostic(name, value):
        if sink_fails:
            raise OSError("optional sink failed")
        if name == "native_continuation_failure": events.append(value)
    monkeypatch.setattr("newsroom.control_plane.diagnostic_logging.emit_diagnostic", diagnostic)
    try:
        report = pipeline.tick(cycle_id="unknown-publication-failure")
        assert report.revision_states == {"PUBLICATION_STARTED": 1}
        assert journal.current(unit.revision_id) == before
        if not sink_fails:
            assert len(events) == 1
            assert events[0]["failure_class"] == "RuntimeError"
            assert events[0]["function"] == "failed" and events[0]["file"] == "test_native_pipeline.py"
            assert events[0]["line"] > 0
            assert "PRIVATE_SOURCE_PROVIDER_TEXT" not in repr(events)
        else:
            assert events == []
    finally:
        connection.close()


def test_reopened_quiescent_ticks_never_select_cold_source_bodies(tmp_path, monkeypatch):
    pipeline, journal, connection, units, calls, dispositions = _open(tmp_path, monkeypatch)
    try:
        pipeline.tick(cycle_id='first-complete')
        pipeline._journal = NativeRevisionJournal(connection)
        dispositions[0] = ()
        statements = []
        connection.set_trace_callback(statements.append)
        original_calls = tuple(calls)
        for cycle_id in ('reopened-one', 'reopened-two'):
            assert pipeline.tick(cycle_id=cycle_id).revision_states == {'ACKNOWLEDGED': 2}
        assert tuple(calls) == original_calls + (('rights', 'current'),) * 2
        assert not any('content_json' in sql for sql in statements)
        assert pipeline._journal._bodies == {}
    finally:
        connection.close()
