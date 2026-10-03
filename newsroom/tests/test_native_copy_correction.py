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


@pytest.mark.parametrize('fault,prepared_legacy', (
    (None, False), (None, True), ('interrupted', False), ('interrupted', True),
    ('current-ack', False), ('current-ack', True), ('changed-package', False), ('changed-package', True),
    ('pending-slot', True), ('missing-slot', True),
))
def test_second_writer_upgrade_uses_latest_ack_without_reusing_old_story_slot(tmp_path, fault, prepared_legacy):
    connection = connect(str(tmp_path / 'second-upgrade.sqlite3'))
    journal = NativeRevisionJournal(connection)
    unit = _native('second-writer-upgrade')
    package_id = ObjectAdmissionId.new()
    decision = _decision(package_id)
    original_refs = {key: 'original-' + key for key in (
        'story_event_id', 'publication_event_id', 'delivery_attempt_event_id', 'delivery_evidence_event_id')}
    current_refs = {key: 'current-' + key for key in original_refs}
    journal.land((unit,))
    facts = dict(candidate_id='candidate', candidate_version_id='candidate-version',
                 graphiti_receipts=[{}], package_admission_id=str(package_id),
                 editorial_decision=json.loads(decision.canonical_bytes()),
                 copy_correction_of={**original_refs, 'progress_ordinal': 1},
                 copy_correction_result='CORRECTED',
                 copy_correction_checked_version='newsroom.offline-exact-copy.v3',
                 writer_id='newsroom.offline-exact-copy.v3', **current_refs)
    if prepared_legacy:
        facts.update(expected_story_version=1, expected_publication_version=2)
    if fault in {'pending-slot', 'missing-slot'}:
        if fault == 'pending-slot':
            facts['expected_story_version'] = 2
        else:
            facts.pop('expected_story_version')
    journal.advance(unit.revision_id, stage='COPY_CORRECTION_PREPARED' if prepared_legacy else 'ACKNOWLEDGED', facts=facts)
    original_rows = connection.execute('SELECT seq,payload_json FROM ledger').fetchall()
    prior = NS(story_receipt=NS(aggregate_version=2), attempt_receipt=NS(aggregate_version=3))
    story = NS(candidate_version_id='candidate-version', package_admission_id=package_id,
               policy_decision_id=decision.decision_id)
    calls = []

    class Publication:
        writer_contract_version = 'newsroom.native-story-writer.v1'

        def retained_writer_id(self, event_id, **kwargs):
            assert event_id == current_refs['story_event_id']
            return 'newsroom.offline-exact-copy.v3'

        def read_acknowledged(self, references, **kwargs):
            if references['story_event_id'] != current_refs['story_event_id'] or fault == 'current-ack':
                raise NativePublicationError('latest acknowledged predecessor is required')
            return prior, NS(**{**vars(story), 'package_admission_id': ObjectAdmissionId.new()}) if fault == 'changed-package' else story

        def advance(self, admitted, policy, **kwargs):
            assert admitted == package_id and policy == decision
            assert kwargs['correction_of'] is prior
            assert (kwargs['expected_story_version'], kwargs['expected_publication_version']) == (2, 3)
            assert journal.current(unit.revision_id)['stage'] == 'COPY_CORRECTION_PREPARED'
            calls.append(kwargs)
            if fault == 'interrupted' and len(calls) == 1:
                raise RuntimeError('interrupted after preparing the next version')
            receipt = lambda name: NS(event_id=name)
            return NS(story_receipt=receipt('latest-story'), publication_receipt=receipt('latest-publication'),
                      attempt_receipt=receipt('latest-attempt'), evidence_receipt=receipt('latest-evidence'),
                      writer_id=self.writer_contract_version)

    runtime = NS(authority=_Authority(), ingress=object(), publication=Publication(), proof=proof(), policies=object())
    def continuation():
        return NativePublicationContinuation(journal=journal, runtime=runtime,
            evidence_controller=object.__new__(NativeEvidenceController), sources={unit.revision_id: (_source(unit),)},
            clock=lambda: UtcTimestamp.parse('2026-09-08T12:06:00Z'))
    try:
        result = continuation().advance(revision_id=unit.revision_id, candidate_version_id='candidate-version')
        if fault in {'current-ack', 'changed-package', 'pending-slot', 'missing-slot'}:
            assert result.state == ('COPY_CORRECTION_PREPARED' if prepared_legacy else 'ACKNOWLEDGED') and result.reason.startswith('COPY_CORRECTION_HOLD')
            assert not calls
            assert journal.current(unit.revision_id)['facts']['story_event_id'] == current_refs['story_event_id']
        else:
            if fault == 'interrupted':
                assert result.state == 'COPY_CORRECTION_PREPARED'
                journal = NativeRevisionJournal(connection)
                result = continuation().advance(revision_id=unit.revision_id, candidate_version_id='candidate-version')
            assert result.state == 'ACKNOWLEDGED' and result.reason is None
            retained = journal.current(unit.revision_id)['facts']
            assert retained['story_event_id'] == 'latest-story'
            assert retained['copy_correction_of']['story_event_id'] == current_refs['story_event_id']
            assert retained['copy_correction_origin']['story_event_id'] == original_refs['story_event_id']
            assert not continuation().copy_correction_due(retained, Publication.writer_contract_version)
            assert all((call['expected_story_version'], call['expected_publication_version']) == (2, 3) for call in calls)
        assert all(connection.execute('SELECT payload_json FROM ledger WHERE seq=?', (seq,)).fetchone()[0] == raw
                   for seq, raw in original_rows)
    finally:
        connection.close()
