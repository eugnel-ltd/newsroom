"""Provider-free proof-only recovery before ordinary continuation effects."""

from collections import Counter
import sqlite3
from types import SimpleNamespace as NS

import pytest

from newsroom.authority.canonical import digest_canonical
from newsroom.control_plane.native_evidence import NativeEvidenceController
from newsroom.control_plane.native_publication import NativePublicationContinuation
from newsroom.control_plane.veto import OperatorDrainRequested, VetoError
from newsroom.tests.test_native_assessor import _usage
from newsroom.tests.test_native_graphiti import _native
from newsroom.tests.test_native_pipeline import _open
from newsroom.tests.test_native_predispatch_batch import _allocated, _envelope, _target


def _prefix(tmp_path, monkeypatch, *, count=15, admission=False, allocated=False):
    pipeline, journal, connection, _units, calls, dispositions = _open(tmp_path, monkeypatch)
    if allocated:
        service, usage, allocated_candidate, _allocation = _allocated(tmp_path, monkeypatch)
    else:
        service, usage = _usage(tmp_path, monkeypatch)
        allocated_candidate = None
    dispositions[0] = ()
    units = tuple(_native(f"interrupted-{index}") for index in range(count))
    versions = {
        unit.revision_id: allocated_candidate if allocated and index == 0 else _target(index)
        for index, unit in enumerate(units)
    }
    candidates = {version.version_id: version for version in versions.values()}
    lookups, batches, ordinary = [], [], []
    for index, unit in enumerate(units):
        version = versions[unit.revision_id]
        journal.land((unit,))
        journal.advance(unit.revision_id, stage="EVIDENCE_HOLD" if admission else "ASSESSMENT_INTERRUPTED", facts={
            "graphiti_receipts": [{}],
            "candidate_id": version.candidate_id,
            "candidate_version_id": version.version_id,
            "failure_class": "ModelUsageAdmissionError" if admission else "NativeEvidenceError",
            "reason": "ACQUISITION_RESULT_NOT_RETAINED",
            "acquisition_attempt_count": index % 4,
            "last_continuation_hold": {"reason": "NATIVE_SOURCE_RAW_OBSERVATION_HOLD", "source_id": unit.source_id},
        })

    def candidate_version(version_id):
        lookups.append(version_id)
        return candidates[version_id]

    def batch(selected):
        batches.append(selected)
        return usage.retained_pre_dispatch_failure_many(selected)

    continuation = NativePublicationContinuation(
        journal=journal,
        runtime=NS(authority=NS(candidate_version=candidate_version), ingress=object(),
                   publication=object(), policies=object(), proof=object()),
        evidence_controller=object.__new__(NativeEvidenceController), sources={},
        assessment_pre_dispatch_failure=usage.retained_pre_dispatch_failure,
    )
    original = pipeline._publish

    class Publication:
        def recover_pre_dispatch(self, revision_ids, *, before_revision):
            return continuation.recover_pre_dispatch(
                revision_ids, failure_many=batch, before_revision=before_revision,
            )

        def advance(self, *, revision_id, candidate_version_id):
            if revision_id in versions:
                ordinary.append(revision_id)
                if journal.progress[revision_id]["stage"] == "ASSESSMENT_INTERRUPTED":
                    return continuation.advance(revision_id=revision_id, candidate_version_id=candidate_version_id)
                raise OSError("possible effect remains unknown")
            return original.advance(revision_id=revision_id, candidate_version_id=candidate_version_id)

    pipeline._publish = Publication()
    return NS(pipeline=pipeline, journal=journal, connection=connection, service=service,
              usage=usage, units=units, versions=versions, candidates=candidates, calls=calls,
              dispositions=dispositions, lookups=lookups, batches=batches, ordinary=ordinary,
              continuation=continuation)


@pytest.mark.parametrize("admission", (False, True))
def test_fifteen_interrupted_candidates_share_one_global_walk_and_exact_transition(
    tmp_path, monkeypatch, admission,
):
    context = _prefix(tmp_path, monkeypatch, admission=admission)
    statements = []
    original_connect = sqlite3.connect

    def connect(*args, **kwargs):
        result = original_connect(*args, **kwargs)
        result.set_trace_callback(statements.append)
        return result

    monkeypatch.setattr(sqlite3, "connect", connect)
    before = dict(context.journal.progress)
    ledger = tuple(context.connection.execute("SELECT * FROM ledger ORDER BY seq"))
    try:
        report = context.pipeline.tick(cycle_id="finite-recovery")
        assert report.revision_states == {"EVIDENCE_HOLD": 15}
        assert len(context.batches) == 1 and len(context.batches[0]) == 15
        assert context.lookups == [version.version_id for version in context.versions.values()]
        assert not context.ordinary
        counts = Counter(" ".join(statement.split()).upper() for statement in statements)
        assert counts["PRAGMA FOREIGN_KEY_CHECK"] == counts["BEGIN"] == 1
        for table in ("model_work_envelopes", "model_invocation_allocations", "model_transport_observations"):
            assert sum(count for sql, count in counts.items() if sql.startswith("SELECT ") and f"FROM {table.upper()} ORDER BY" in sql) == 1
        for index, unit in enumerate(context.units):
            version = context.versions[unit.revision_id]
            retained = context.journal.progress[unit.revision_id]
            assert retained["ordinal"] == before[unit.revision_id]["ordinal"] + 1
            assert retained["facts"] == {
                **before[unit.revision_id]["facts"],
                "reason": "ASSESSOR_PRE_DISPATCH_HOLD", "acquisition_retryable": index % 4 < 3,
                "assessment_pre_dispatch_candidate_id": version.candidate_id,
                "assessment_pre_dispatch_candidate_version_id": version.version_id,
                "assessment_pre_dispatch_manifest_digest": version.governing_manifest.canonical_digest,
                "assessment_pre_dispatch_inventory_digest": digest_canonical(()),
            }
        assert tuple(context.connection.execute("SELECT * FROM ledger WHERE seq <= ? ORDER BY seq", (ledger[-1][0],))) == ledger
        assert not any(call[0] in {"graphiti", "discovery", "publish"} for call in context.calls)
    finally:
        context.connection.close()


def test_allocated_unknown_and_protected_candidates_remain_ordinary_before_fresh_work(tmp_path, monkeypatch):
    context = _prefix(tmp_path, monkeypatch, count=4, allocated=True)
    allocated, unknown, started, publishing = context.units
    context.candidates.pop(context.versions[unknown.revision_id].version_id)
    for unit, stage in ((started, "ASSESSMENT_STARTED"), (publishing, "PUBLICATION_STARTED")):
        context.journal.advance(unit.revision_id, stage=stage, facts={**context.journal.progress[unit.revision_id]["facts"], "assessment_started_at": "2026-09-08T12:00:00Z"})
    fresh = _native("fresh")
    context.dispositions[0] = (NS(source_id=fresh.source_id, status="READY", reason_code="RETAINED", units=(fresh,)),)
    before = dict(context.journal.progress)
    try:
        context.pipeline.tick(cycle_id="unknown-before-fresh")
        assert context.ordinary == [unit.revision_id for unit in context.units]
        assert len(context.batches) == 1 and len(context.batches[0]) == 1
        assert context.batches[0][0] == context.versions[allocated.revision_id]
        assert context.journal.progress[allocated.revision_id] == before[allocated.revision_id]
        assert context.journal.progress[unknown.revision_id] == before[unknown.revision_id]
        assert context.journal.progress[publishing.revision_id] == before[publishing.revision_id]
        assert context.journal.progress[started.revision_id] == before[started.revision_id]
        assert context.journal.progress[fresh.revision_id]["stage"] == "ACKNOWLEDGED"
    finally:
        context.connection.close()


def test_unrelated_global_corruption_denies_all_prefix_transitions(tmp_path, monkeypatch):
    context = _prefix(tmp_path, monkeypatch, count=3)
    _envelope(context.service, 0)
    with sqlite3.connect(context.service.path) as connection:
        connection.execute("UPDATE model_work_envelopes SET record_json='{}'")
    before = dict(context.journal.progress)
    try:
        report = context.pipeline.tick(cycle_id="global-integrity-denial")
        assert report.revision_states == {"ASSESSMENT_INTERRUPTED": 3}
        assert context.journal.progress == before
        assert context.ordinary == [unit.revision_id for unit in context.units]
        assert len(context.batches) == 1
    finally:
        context.connection.close()


@pytest.mark.parametrize("signal", ("quantum", "drain", "stop"))
def test_prefix_bounds_preserve_unattempted_facts_and_reprove_next_tick(tmp_path, monkeypatch, signal):
    context = _prefix(tmp_path, monkeypatch, count=3)
    now, stopped = [0.0], [False]
    context.pipeline._monotonic_clock = lambda: now[0]
    context.pipeline._operator_drain_requested = lambda: stopped[0] and signal == "drain"
    original_check = context.pipeline._check

    def check():
        original_check()
        if stopped[0] and signal == "stop":
            raise VetoError("stop in proof-only prefix")

    context.pipeline._check = check
    advance = context.journal.advance

    def retain(revision_id, *, stage, facts):
        result = advance(revision_id, stage=stage, facts=facts)
        if facts.get("reason") == "ASSESSOR_PRE_DISPATCH_HOLD":
            now[0] += 301
            stopped[0] = True
        return result

    monkeypatch.setattr(context.journal, "advance", retain)
    before = dict(context.journal.progress)
    try:
        if signal == "quantum":
            context.pipeline.tick(cycle_id="bounded-prefix")
        else:
            with pytest.raises(OperatorDrainRequested if signal == "drain" else VetoError):
                context.pipeline.tick(cycle_id="bounded-prefix")
        assert context.journal.progress[context.units[0].revision_id]["stage"] == "EVIDENCE_HOLD"
        assert all(context.journal.progress[unit.revision_id] == before[unit.revision_id] for unit in context.units[1:])
        assert not context.ordinary
        assert len(context.batches) == 1
        stopped[0] = False
        # An unrelated mutation is observed by a new snapshot, not the old proof.
        _envelope(context.service, 0)
        with sqlite3.connect(context.service.path) as connection:
            connection.execute("UPDATE model_work_envelopes SET canonical_digest='changed'")
        context.pipeline.tick(cycle_id="fresh-proof-after-mutation")
        assert len(context.batches) == 2
        assert all(context.journal.progress[unit.revision_id] == before[unit.revision_id] for unit in context.units[1:])
        assert context.ordinary.count(context.units[1].revision_id) == 1
        assert context.ordinary.count(context.units[2].revision_id) == 1
    finally:
        context.connection.close()


@pytest.mark.parametrize("boundary", ("candidate-read", "proof-read"))
def test_predispatch_atomic_read_overrun_makes_progress_before_next_tick(
    tmp_path, monkeypatch, boundary,
):
    context = _prefix(tmp_path, monkeypatch, count=3)
    now = [0.0]
    context.pipeline._monotonic_clock = lambda: now[0]
    # Settled revisions are non-retryable so later ticks exercise only the
    # unconsumed proof prefix, never a fresh acquisition effect.
    for unit in context.units:
        context.journal.advance(unit.revision_id, stage="ASSESSMENT_INTERRUPTED", facts={
            **context.journal.progress[unit.revision_id]["facts"], "acquisition_attempt_count": 3,
        })
    before = dict(context.journal.progress)
    candidate_version = context.continuation._runtime.authority.candidate_version
    batch = context.usage.retained_pre_dispatch_failure_many

    def slow_candidate(version_id):
        version = candidate_version(version_id)
        now[0] += 101
        return version

    def slow_proof(candidates):
        result = batch(candidates)
        now[0] += 301
        return result

    if boundary == "candidate-read":
        context.continuation._runtime.authority.candidate_version = slow_candidate
    else:
        monkeypatch.setattr(context.usage, "retained_pre_dispatch_failure_many", slow_proof)
    try:
        first = context.pipeline.tick(cycle_id="atomic-read-overrun")
        assert first.revision_states == {"EVIDENCE_HOLD": 1, "ASSESSMENT_INTERRUPTED": 2}
        assert now[0] == (303 if boundary == "candidate-read" else 301)
        assert len(context.batches) == 1
        assert context.journal.progress[context.units[0].revision_id]["facts"]["acquisition_attempt_count"] == 3
        assert all(context.journal.progress[unit.revision_id] == before[unit.revision_id] for unit in context.units[1:])
        assert not context.ordinary
        for tick in range(2):
            settled_before = sum(progress["stage"] == "EVIDENCE_HOLD" for progress in context.journal.progress.values())
            if settled_before == 3:
                break
            context.pipeline.tick(cycle_id=f"atomic-read-overrun-{tick}")
            assert sum(progress["stage"] == "EVIDENCE_HOLD" for progress in context.journal.progress.values()) > settled_before
        assert all(progress["stage"] == "EVIDENCE_HOLD" for progress in context.journal.progress.values())
        assert len(context.batches) == (2 if boundary == "candidate-read" else 3)
        assert not context.ordinary
    finally:
        context.connection.close()


@pytest.mark.parametrize("boundary", ("candidate-read", "proof-read"))
@pytest.mark.parametrize("signal", ("stop", "drain"))
def test_predispatch_read_overrun_still_obeys_stop_and_drain_before_commit(
    tmp_path, monkeypatch, boundary, signal,
):
    context = _prefix(tmp_path, monkeypatch, count=3)
    now, stopped = [0.0], [False]
    context.pipeline._monotonic_clock = lambda: now[0]
    context.pipeline._operator_drain_requested = lambda: stopped[0] and signal == "drain"

    def check():
        if stopped[0] and signal == "stop":
            raise VetoError("stop after atomic read overrun")

    context.pipeline._check = check
    candidate_version = context.continuation._runtime.authority.candidate_version
    batch = context.usage.retained_pre_dispatch_failure_many

    def slow_candidate(version_id):
        version = candidate_version(version_id)
        now[0], stopped[0] = 301, True
        return version

    def slow_proof(candidates):
        result = batch(candidates)
        now[0], stopped[0] = 301, True
        return result

    if boundary == "candidate-read":
        context.continuation._runtime.authority.candidate_version = slow_candidate
    else:
        monkeypatch.setattr(context.usage, "retained_pre_dispatch_failure_many", slow_proof)
    before = dict(context.journal.progress)
    try:
        with pytest.raises(VetoError if signal == "stop" else OperatorDrainRequested):
            context.pipeline.tick(cycle_id="stopped-atomic-read-overrun")
        assert context.journal.progress == before
        assert len(context.batches) == (0 if boundary == "candidate-read" else 1)
        assert not context.ordinary
    finally:
        context.connection.close()


@pytest.mark.parametrize("unproved", ("allocated", "unknown", "global-corruption"))
def test_predispatch_expired_proof_never_retires_unproved_targets(
    tmp_path, monkeypatch, unproved,
):
    context = _prefix(tmp_path, monkeypatch, count=3, allocated=unproved == "allocated")
    now = [0.0]
    context.pipeline._monotonic_clock = lambda: now[0]
    if unproved == "unknown":
        context.candidates.pop(context.versions[context.units[0].revision_id].version_id)
    elif unproved == "global-corruption":
        _envelope(context.service, 0)
        with sqlite3.connect(context.service.path) as connection:
            connection.execute("UPDATE model_work_envelopes SET record_json='{}'")
    batch = context.usage.retained_pre_dispatch_failure_many

    def slow_proof(candidates):
        result = batch(candidates)
        now[0] += 301
        return result

    monkeypatch.setattr(context.usage, "retained_pre_dispatch_failure_many", slow_proof)
    before = dict(context.journal.progress)
    try:
        context.pipeline.tick(cycle_id="expired-unproved-target")
        assert context.journal.progress[context.units[0].revision_id] == before[context.units[0].revision_id]
        assert context.journal.progress[context.units[2].revision_id] == before[context.units[2].revision_id]
        if unproved == "global-corruption":
            assert context.journal.progress == before
        else:
            assert context.journal.progress[context.units[1].revision_id]["stage"] == "EVIDENCE_HOLD"
        assert len(context.batches) == 1
        assert not context.ordinary
    finally:
        context.connection.close()
