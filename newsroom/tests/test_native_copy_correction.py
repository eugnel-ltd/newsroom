"""Copy correction reuses existing authority versions and survives interruption."""
import json
from types import SimpleNamespace as NS

import pytest

from newsroom.authority import ObjectAdmissionId, UtcTimestamp
from newsroom.control_plane.native_evidence import NativeEvidenceController
from newsroom.control_plane.native_progress import NativeRevisionJournal
from newsroom.control_plane.native_publication import NativePublicationContinuation, NativePublicationError
from newsroom.control_plane.store import connect
from newsroom.tests.authority_helpers import proof
from newsroom.tests.test_native_graphiti import _native
from newsroom.tests.test_native_publication_continuation import _Authority, _decision, _source


@pytest.mark.parametrize("fault", (None, "interrupted", "predecessor", "changed-package"))
def test_copy_correction_retains_predecessor_and_replays_exact_intent(tmp_path, fault):
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    unit = _native("copy-correction")
    package_id = ObjectAdmissionId.new()
    decision = _decision(package_id)
    journal.land((unit,))
    facts = dict(candidate_id="candidate", candidate_version_id="candidate-version",
                 graphiti_receipts=[{}], package_admission_id=str(package_id),
                 editorial_decision=json.loads(decision.canonical_bytes()),
                 story_event_id="old-story", publication_event_id="old-publication",
                 delivery_attempt_event_id="old-attempt", delivery_evidence_event_id="old-evidence")
    journal.advance(unit.revision_id, stage="ACKNOWLEDGED", facts=facts)
    original_rows = connection.execute("SELECT seq,payload_json FROM ledger").fetchall()
    original_ordinal = journal.current(unit.revision_id)["ordinal"]
    prior = NS(story_receipt=NS(aggregate_version=1), attempt_receipt=NS(aggregate_version=2))
    story = NS(candidate_version_id="candidate-version", package_admission_id=package_id,
               policy_decision_id=decision.decision_id)
    if fault == "changed-package":
        story.package_admission_id = ObjectAdmissionId.new()
    calls = []

    class Publication:
        def retained_writer_id(self, event_id, **kwargs):
            assert event_id == "old-story"
            return "newsroom.offline-exact-copy.v2"

        def read_acknowledged(self, references, **kwargs):
            assert references["story_event_id"] == "old-story"
            if fault == "predecessor":
                raise NativePublicationError("invalid prior ACK")
            return prior, story

        def advance(self, admitted, policy, **kwargs):
            assert admitted == package_id and policy == decision
            assert kwargs["correction_of"] is prior
            assert journal.current(unit.revision_id)["stage"] == "COPY_CORRECTION_PREPARED" or len(calls) >= 1
            calls.append(kwargs)
            if fault == "interrupted" and len(calls) == 1:
                raise RuntimeError("interrupted after a versioned operation")
            receipt = lambda name: NS(event_id=name)
            return NS(story_receipt=receipt("new-story"), publication_receipt=receipt("new-publication"),
                      attempt_receipt=receipt("new-attempt"), evidence_receipt=receipt("new-evidence"),
                      writer_id="newsroom.offline-exact-copy.v3")

    times = iter((UtcTimestamp.parse("2026-09-08T12:04:00Z"), UtcTimestamp.parse("2026-09-08T12:05:00Z")))
    runtime = NS(authority=_Authority(), ingress=object(), publication=Publication(), proof=proof(), policies=object())

    def continuation():
        return NativePublicationContinuation(
            journal=journal, runtime=runtime, evidence_controller=object.__new__(NativeEvidenceController),
            sources={unit.revision_id: (_source(unit),)}, clock=lambda: next(times),
        )

    try:
        first = continuation().advance(revision_id=unit.revision_id, candidate_version_id="candidate-version")
        if fault in {"predecessor", "changed-package"}:
            assert first.state == "ACKNOWLEDGED" and first.reason.startswith("COPY_CORRECTION_HOLD")
            assert calls == []
            assert journal.current(unit.revision_id)["facts"]["story_event_id"] == "old-story"
        else:
            if fault == "interrupted":
                assert first.state == "COPY_CORRECTION_PREPARED"
                journal = NativeRevisionJournal(connection)
                first = continuation().advance(revision_id=unit.revision_id, candidate_version_id="candidate-version")
            assert first.state == "ACKNOWLEDGED"
            retained = journal.current(unit.revision_id)["facts"]
            assert retained["copy_correction_of"]["progress_ordinal"] == original_ordinal
            assert retained["copy_correction_of"]["story_event_id"] == "old-story"
            assert retained["story_event_id"] == "new-story"
            assert not continuation().copy_correction_due(retained)
            continuation().advance(revision_id=unit.revision_id, candidate_version_id="candidate-version")
            assert all((c["expected_story_version"], c["expected_publication_version"], c["expected_delivery_evidence_version"]) == (1, 2, 0) for c in calls)
            assert retained["publication_started_at"] == "2026-09-08T12:04:00.000000Z"
            assert all("applied_at" not in call and "observed_at" not in call for call in calls)
        for seq, payload in original_rows:
            assert connection.execute("SELECT payload_json FROM ledger WHERE seq=?", (seq,)).fetchone()[0] == payload
    finally:
        connection.close()
