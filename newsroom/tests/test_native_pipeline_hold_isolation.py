"""Provider-free regression coverage for revision-local continuation failures."""

from types import SimpleNamespace as NS

import pytest

from newsroom.control_plane.native_evidence import NativeEvidenceHold
from newsroom.control_plane.native_progress import NativeRevisionJournal
from newsroom.control_plane.native_publication import NativePublicationContinuation
from newsroom.control_plane.veto import OperatorDrainRequested, VetoError
from newsroom.tests.test_native_pipeline import _open


@pytest.fixture
def continuation(tmp_path, monkeypatch):
    pipeline, journal, connection, units, calls, dispositions = _open(
        tmp_path, monkeypatch,
    )
    held, fresh = units
    journal.land((held,))
    facts = {
        "graphiti_receipts": [{"ingest_id": held.ingest_id, "retained": True}],
        "candidate_version_id": "candidate:one",
    }
    dispositions[0] = (NS(
        source_id=fresh.source_id, status="READY", reason_code="RETAINED",
        units=(fresh,),
    ),)
    original = pipeline._publish

    def fail_continuation(exc):
        def publish(*, revision_id, candidate_version_id):
            if revision_id == held.revision_id:
                calls.append(("resume", revision_id))
                raise exc
            original.advance(
                revision_id=revision_id, candidate_version_id=candidate_version_id,
            )

        pipeline._publish = NS(
            advance=publish,
            copy_correction_due=lambda current: (
                current.get("candidate_version_id") == "candidate:one"
                and NativePublicationContinuation.copy_correction_due(current)
            ),
        )

    try:
        yield NS(
            pipeline=pipeline, journal=journal, connection=connection,
            held=held, fresh=fresh, calls=calls, facts=facts,
            fail_continuation=fail_continuation,
        )
    finally:
        connection.close()


def _assert_fresh_progress(context):
    assert [call for call in context.calls if call[0] in {"resume", "graphiti", "discovery", "publish"}] == [
        ("resume", context.held.revision_id),
        ("graphiti", context.fresh.item_key),
        ("discovery", context.fresh.item_key),
        ("publish", context.fresh.revision_id),
    ]
    assert NativeRevisionJournal(context.connection).current(context.fresh.revision_id)["stage"] == "ACKNOWLEDGED"


def _assert_ledger_prefix(connection, before):
    assert tuple(connection.execute(
        "SELECT * FROM ledger WHERE seq <= ? ORDER BY seq", (before[-1][0],),
    )) == before


def test_admission_recovery_source_hold_does_not_starve_fresh_graphiti(continuation):
    context = continuation
    facts = {
        **context.facts, "failure_class": "ModelUsageAdmissionError",
        "reason": "ACQUISITION_RESULT_NOT_RETAINED",
    }
    context.journal.advance(context.held.revision_id, stage="EVIDENCE_HOLD", facts=facts)
    before = tuple(context.connection.execute("SELECT * FROM ledger ORDER BY seq"))
    context.fail_continuation(NativeEvidenceHold(
        "NATIVE_SOURCE_RAW_OBSERVATION_HOLD", context.held.source_id,
    ))

    report = context.pipeline.tick(cycle_id="held-recovery-with-fresh-work")

    assert report.revision_states == {"EVIDENCE_HOLD": 1, "ACKNOWLEDGED": 1}
    assert report.unclassified_revisions == 0
    _assert_fresh_progress(context)
    retained = NativeRevisionJournal(context.connection)
    held = retained.current(context.held.revision_id)
    assert held["stage"] == "EVIDENCE_HOLD"
    assert held["facts"] == {
        **facts, "last_continuation_hold": {
            "reason": "NATIVE_SOURCE_RAW_OBSERVATION_HOLD",
            "source_id": context.held.source_id,
        },
    }
    assert retained.units[context.held.revision_id] == (context.held,)
    _assert_ledger_prefix(context.connection, before)

    settled = tuple(context.connection.execute("SELECT * FROM ledger ORDER BY seq"))
    context.pipeline.tick(cycle_id="unchanged-source-hold")
    assert context.journal.current(context.held.revision_id) == held
    assert tuple(context.connection.execute("SELECT * FROM ledger ORDER BY seq")) == settled
    assert context.calls.count(("resume", context.held.revision_id)) == 2
    assert context.calls.count(("graphiti", context.fresh.item_key)) == 1
    assert context.calls.count(("publish", context.fresh.revision_id)) == 1


@pytest.mark.parametrize("stage", (
    "ASSESSMENT_INTERRUPTED", "COPY_CORRECTION_PREPARED", "ACKNOWLEDGED",
))
@pytest.mark.parametrize("failure", ("source-hold", "unknown-effect"))
def test_retained_resume_failure_preserves_precise_history_and_advances_peer(
    continuation, stage, failure,
):
    context = continuation
    facts = {
        **context.facts, "reason": "ACQUISITION_RESULT_NOT_RETAINED",
        "failure_class": "OSError",
        "writer_id": "newsroom.offline-exact-copy.v2",
        "story_event_id": "original-story-event",
        "publication_event_id": "original-publication-event",
        "delivery_attempt_event_id": "original-delivery-attempt-event",
        "delivery_evidence_event_id": "original-delivery-evidence-event",
    }
    if stage == "COPY_CORRECTION_PREPARED":
        facts["copy_correction_of"] = {
            "story_event_id": "original-story-event", "progress_ordinal": 1,
        }
        facts["expected_story_version"] = 1
        facts["publication_started_at"] = "2026-09-08T12:00:00Z"
    original = context.journal.advance(context.held.revision_id, stage=stage, facts=facts)
    before = tuple(context.connection.execute("SELECT * FROM ledger ORDER BY seq"))
    exc = (
        NativeEvidenceHold("NATIVE_SOURCE_RAW_OBSERVATION_HOLD", context.held.source_id)
        if failure == "source-hold" else OSError("retained effect is still unknown")
    )
    context.fail_continuation(exc)

    report = context.pipeline.tick(cycle_id="retained-resume-failure")

    expected_states = {stage: 1, "ACKNOWLEDGED": 1} if stage != "ACKNOWLEDGED" else {"ACKNOWLEDGED": 2}
    assert report.revision_states == expected_states
    _assert_fresh_progress(context)
    held = NativeRevisionJournal(context.connection).current(context.held.revision_id)
    if failure == "source-hold":
        assert held == {
            **original, "ordinal": original["ordinal"] + 1,
            "facts": {**facts, "last_continuation_hold": {
                "reason": "NATIVE_SOURCE_RAW_OBSERVATION_HOLD",
                "source_id": context.held.source_id,
            }},
        }
    else:
        assert held == original
    _assert_ledger_prefix(context.connection, before)

    settled = tuple(context.connection.execute("SELECT * FROM ledger ORDER BY seq"))
    context.pipeline.tick(cycle_id="unchanged-retained-resume-failure")
    assert context.journal.current(context.held.revision_id) == held
    assert tuple(context.connection.execute("SELECT * FROM ledger ORDER BY seq")) == settled
    assert context.calls.count(("resume", context.held.revision_id)) == 2
    assert context.calls.count(("graphiti", context.fresh.item_key)) == 1
    assert context.calls.count(("publish", context.fresh.revision_id)) == 1


def test_candidate_source_hold_becomes_exact_publication_hold(continuation):
    context = continuation
    context.journal.advance(
        context.held.revision_id, stage="CANDIDATE_ADMITTED", facts=context.facts,
    )
    context.fail_continuation(NativeEvidenceHold(
        "NATIVE_SOURCE_RAW_OBSERVATION_HOLD", context.held.source_id,
    ))

    report = context.pipeline.tick(cycle_id="candidate-source-hold")

    assert report.revision_states == {"PUBLICATION_HOLD": 1, "ACKNOWLEDGED": 1}
    _assert_fresh_progress(context)
    held = NativeRevisionJournal(context.connection).current(context.held.revision_id)
    assert held["stage"] == "PUBLICATION_HOLD"
    assert held["facts"] == {
        **context.facts, "reason": "NATIVE_SOURCE_RAW_OBSERVATION_HOLD",
    }


@pytest.mark.parametrize("stage", ("ASSESSMENT_STARTED", "PUBLICATION_STARTED"))
def test_unknown_dispatch_failure_keeps_existing_intent_marker(continuation, stage):
    context = continuation
    original = context.journal.advance(
        context.held.revision_id, stage=stage, facts={
            **context.facts, "reason": "DISPATCH_RESULT_UNKNOWN",
            "assessment_started_at": "2026-09-08T12:00:00Z",
        },
    )
    before = tuple(context.connection.execute("SELECT * FROM ledger ORDER BY seq"))
    context.fail_continuation(OSError("possible dispatch has no exact terminal"))

    assert context.pipeline.tick(cycle_id="unknown-dispatch").revision_states == {
        stage: 1, "ACKNOWLEDGED": 1,
    }
    _assert_fresh_progress(context)
    assert NativeRevisionJournal(context.connection).current(context.held.revision_id) == original
    _assert_ledger_prefix(context.connection, before)


@pytest.mark.parametrize("signal", (VetoError, OperatorDrainRequested))
def test_owner_stop_and_drain_escape_retained_resume_without_hold_conversion(
    continuation, signal,
):
    context = continuation
    original = context.journal.advance(
        context.held.revision_id, stage="EVIDENCE_HOLD", facts={
            **context.facts, "failure_class": "ModelUsageAdmissionError",
            "reason": "ACQUISITION_RESULT_NOT_RETAINED",
        },
    )
    context.fail_continuation(signal("stop at retained continuation"))

    with pytest.raises(signal):
        context.pipeline.tick(cycle_id="stopped-retained-resume")

    assert NativeRevisionJournal(context.connection).current(context.held.revision_id) == original
    assert not any(call[0] in {"graphiti", "discovery", "publish"} for call in context.calls)
    assert context.calls.count(("resume", context.held.revision_id)) == 1
    # Retained work now runs before intake; a stop must prevent new LAND too.
    assert context.fresh.revision_id not in context.journal.units
    assert context.fresh.revision_id not in {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()}
