"""Keep qualification readiness aligned with retained producer checkpoints."""

import json
import sqlite3
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace as NS

import pytest

from newsroom.control_plane.native_evidence import NativeEvidenceHold
from newsroom.control_plane.native_pipeline import NativePipeline
from newsroom.control_plane.native_progress import NativeRevisionJournal
from newsroom.control_plane.native_qualification import (
    NativeQualificationError,
    qualification_report_ready,
    record_qualification,
)
from newsroom.tests.test_native_graphiti import _native
from newsroom.tests.test_native_qualification import IDENTITY, _cycle, _open
from newsroom.tests.test_native_service import _pipeline, _service


def test_observed_mixed_revision_inventory_defers_qualification():
    # Exact inventory of the completed cycle that previously stopped the daemon.
    states = {
        "ACKNOWLEDGED": 3,
        "ASSESSMENT_INTERRUPTED": 15,
        "CANDIDATE_ADMITTED": 1,
        "DISCOVERY_HOLD": 49,
        "EVIDENCE_HOLD": 532,
        "GRAPHITI_COMPLETE": 2,
        "GRAPHITI_HOLD": 364,
        "PUBLICATION_HOLD": 3,
        "QUEUED": 147,
        "SAME_STATE_ASSOCIATED": 15,
    }

    assert qualification_report_ready(states, 147) is False


def test_publication_hold_is_ready_but_still_requires_retained_evidence(tmp_path):
    assert qualification_report_ready({"PUBLICATION_HOLD": 1}, 0) is True
    connection = _open(tmp_path / "publication-hold.sqlite3")
    try:
        _cycle(connection, revision_state="PUBLICATION_HOLD", reason="RuntimeError")

        with pytest.raises(NativeQualificationError, match="hold is not evidenced"):
            record_qualification(connection, IDENTITY)

        assert connection.execute(
            "SELECT count(*) FROM ledger WHERE kind='NATIVE_SERVICE_QUALIFICATION'"
        ).fetchone()[0] == 0
    finally:
        connection.close()


def test_copy_correction_prepared_remains_pending_and_never_qualifies(tmp_path):
    assert qualification_report_ready({"COPY_CORRECTION_PREPARED": 1}, 0) is False
    connection = _open(tmp_path / "copy-correction.sqlite3")
    try:
        _cycle(connection, revision_state="COPY_CORRECTION_PREPARED")

        with pytest.raises(NativeQualificationError, match="terminal inventory differs"):
            record_qualification(connection, IDENTITY)

        assert connection.execute(
            "SELECT count(*) FROM ledger WHERE kind='NATIVE_SERVICE_QUALIFICATION'"
        ).fetchone()[0] == 0
    finally:
        connection.close()


def test_unknown_revision_state_still_fails_closed():
    with pytest.raises(NativeQualificationError, match="terminal inventory differs"):
        qualification_report_ready({"UNKNOWN": 1, "PUBLICATION_HOLD": 1}, 0)


@pytest.mark.parametrize("count", [True, 0, "3"])
def test_malformed_publication_hold_count_still_fails_closed(count):
    with pytest.raises(NativeQualificationError, match="terminal inventory differs"):
        qualification_report_ready({"PUBLICATION_HOLD": count}, 0)


def test_service_continues_two_real_report_cycles_without_qualifying_pending_copy(
    tmp_path, monkeypatch,
):
    real_tick = NativePipeline.tick
    resumed, waits = [], []
    units = (_native("publication-hold"), _native("copy-correction"))

    def publish(*, revision_id, candidate_version_id):
        resumed.append(revision_id)
        raise NativeEvidenceHold("SOURCE_LOCAL_EVIDENCE_HOLD", units[0].source_id)

    def tick(cycle_id):
        return real_tick(producer, cycle_id=cycle_id)

    factory, opened = _pipeline(tmp_path, monkeypatch, tick)

    @contextmanager
    def bound():
        nonlocal producer
        with factory() as pipeline:
            journal = pipeline._journal
            for unit, stage in zip(units, ("PUBLICATION_HOLD", "COPY_CORRECTION_PREPARED")):
                journal.land((unit,))
                journal.advance(unit.revision_id, stage=stage, facts={
                    "graphiti_receipts": [{"ingest_id": unit.ingest_id, "retained": True}],
                    "candidate_version_id": "candidate:" + unit.item_key,
                    "reason": "SOURCE_LOCAL_EVIDENCE_HOLD",
                })
            producer = NativePipeline(
                runtime=NS(authority=object(), proof=object()), journal=journal,
                source_intake=NS(poll=lambda: ()), graphiti=object(), discovery=object(),
                retrieval_for=lambda _: pytest.fail("retained work requested retrieval"),
                collision=object(), publish=NS(advance=publish),
                actor_identity_digest=IDENTITY, stop_check=lambda: None,
                stop_fence=nullcontext,
            )
            pipeline.runtime_identity_digest = IDENTITY
            yield pipeline

    producer = None
    report = _service(
        tmp_path, bound,
        qualify_once=lambda *_: pytest.fail("pending copy correction was qualified"),
        wait=lambda _: waits.append(True) or len(waits) == 2,
    ).run()

    states = {"PUBLICATION_HOLD": 1, "COPY_CORRECTION_PREPARED": 1}
    assert report.outcome == "COMPLETE"
    assert report.pipeline.revision_states == states
    assert opened == ["open", "close"]
    assert len(waits) == 2
    assert resumed.count(units[0].revision_id) == resumed.count(units[1].revision_id) == 2
    with sqlite3.connect(tmp_path / "unpublished.sqlite3") as connection:
        assert connection.execute(
            "SELECT kind FROM ledger WHERE kind LIKE 'NATIVE_SERVICE_%' ORDER BY seq"
        ).fetchall() == [
            ("NATIVE_SERVICE_CYCLE_STARTED",), ("NATIVE_SERVICE_CYCLE_TERMINAL",),
            ("NATIVE_SERVICE_CYCLE_STARTED",), ("NATIVE_SERVICE_CYCLE_TERMINAL",),
        ]
        terminal = connection.execute(
            "SELECT payload_json FROM ledger WHERE kind='NATIVE_SERVICE_CYCLE_TERMINAL'"
        ).fetchall()
        assert all(json.loads(raw)["pipeline"]["revision_states"] == states for raw, in terminal)
        retained = NativeRevisionJournal(connection)
        assert retained.progress[units[1].revision_id]["stage"] == "COPY_CORRECTION_PREPARED"
