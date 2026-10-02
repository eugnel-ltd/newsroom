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
    lookups, batches, ordinary, version_batches = [], [], [], []
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

    def candidate_versions(version_ids):
        version_batches.append(version_ids)
        lookups.extend(version_ids)
        return tuple(candidates.get(version_id) for version_id in version_ids)

    def batch(selected):
        batches.append(selected)
        return usage.retained_pre_dispatch_failure_many(selected)

    continuation = NativePublicationContinuation(
        journal=journal,
        runtime=NS(authority=NS(candidate_version=candidate_version, candidate_versions=candidate_versions), ingress=object(),
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
                if journal.current(revision_id)["stage"] == "ASSESSMENT_INTERRUPTED":
                    return continuation.advance(revision_id=revision_id, candidate_version_id=candidate_version_id)
                raise OSError("possible effect remains unknown")
            return original.advance(revision_id=revision_id, candidate_version_id=candidate_version_id)

    pipeline._publish = Publication()
    return NS(pipeline=pipeline, journal=journal, connection=connection, service=service,
              usage=usage, units=units, versions=versions, candidates=candidates, calls=calls,
              dispositions=dispositions, lookups=lookups, batches=batches, ordinary=ordinary,
              continuation=continuation, version_batches=version_batches)


def test_eighty_five_recovery_candidates_use_one_authority_snapshot(tmp_path, monkeypatch):
    context = _prefix(tmp_path, monkeypatch, count=85, admission=True)
    try:
        report = context.pipeline.tick(cycle_id="finite-authority-recovery")
        expected = tuple(version.version_id for version in context.versions.values())
        assert context.version_batches == [expected]
        assert context.lookups == list(expected)
        assert report.revision_states == {"EVIDENCE_HOLD": 85}
        assert not context.ordinary
    finally:
        context.connection.close()


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
    before = {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()}
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
            retained = context.journal.current(unit.revision_id)
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


def test_checked_allocated_denial_skips_duplicate_while_unknown_and_protected_remain_ordinary(tmp_path, monkeypatch):
    context = _prefix(tmp_path, monkeypatch, count=4, allocated=True)
    allocated, unknown, started, publishing = context.units
    context.candidates.pop(context.versions[unknown.revision_id].version_id)
    for unit, stage in ((started, "ASSESSMENT_STARTED"), (publishing, "PUBLICATION_STARTED")):
        context.journal.advance(unit.revision_id, stage=stage, facts={**context.journal.current(unit.revision_id)["facts"], "assessment_started_at": "2026-09-08T12:00:00Z"})
    fresh = _native("fresh")
    context.dispositions[0] = (NS(source_id=fresh.source_id, status="READY", reason_code="RETAINED", units=(fresh,)),)
    before = {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()}
    try:
        context.pipeline.tick(cycle_id="unknown-before-fresh")
        assert context.ordinary == [unit.revision_id for unit in (unknown, started, publishing)]
        assert len(context.batches) == 1 and len(context.batches[0]) == 1
        assert context.batches[0][0] == context.versions[allocated.revision_id]
        assert context.journal.current(allocated.revision_id) == before[allocated.revision_id]
        assert context.journal.current(unknown.revision_id) == before[unknown.revision_id]
        assert context.journal.current(publishing.revision_id) == before[publishing.revision_id]
        assert context.journal.current(started.revision_id) == before[started.revision_id]
        assert context.journal.current(fresh.revision_id)["stage"] == "ACKNOWLEDGED"
    finally:
        context.connection.close()


def test_unrelated_global_corruption_denies_all_prefix_transitions(tmp_path, monkeypatch):
    context = _prefix(tmp_path, monkeypatch, count=3)
    _envelope(context.service, 0)
    with sqlite3.connect(context.service.path) as connection:
        connection.execute("UPDATE model_work_envelopes SET record_json='{}'")
    before = {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()}
    try:
        report = context.pipeline.tick(cycle_id="global-integrity-denial")
        assert report.revision_states == {"ASSESSMENT_INTERRUPTED": 3}
        assert {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()} == before
        assert not context.ordinary
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
    before = {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()}
    try:
        if signal == "quantum":
            context.pipeline.tick(cycle_id="bounded-prefix")
        else:
            with pytest.raises(OperatorDrainRequested if signal == "drain" else VetoError):
                context.pipeline.tick(cycle_id="bounded-prefix")
        assert context.journal.current(context.units[0].revision_id)["stage"] == "EVIDENCE_HOLD"
        assert all(context.journal.current(unit.revision_id) == before[unit.revision_id] for unit in context.units[1:])
        assert not context.ordinary
        assert len(context.batches) == 1
        stopped[0] = False
        # An unrelated mutation is observed by a new snapshot, not the old proof.
        _envelope(context.service, 0)
        with sqlite3.connect(context.service.path) as connection:
            connection.execute("UPDATE model_work_envelopes SET canonical_digest='changed'")
        context.pipeline.tick(cycle_id="fresh-proof-after-mutation")
        assert len(context.batches) == 2
        assert all(context.journal.current(unit.revision_id) == before[unit.revision_id] for unit in context.units[1:])
        assert context.ordinary.count(context.units[1].revision_id) == 0
        assert context.ordinary.count(context.units[2].revision_id) == 0
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
            **context.journal.current(unit.revision_id)["facts"], "acquisition_attempt_count": 3,
        })
    before = {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()}
    candidate_versions = context.continuation._runtime.authority.candidate_versions
    batch = context.usage.retained_pre_dispatch_failure_many

    def slow_candidate(version_ids):
        versions = candidate_versions(version_ids)
        now[0] += 301
        return versions

    def slow_proof(candidates):
        result = batch(candidates)
        now[0] += 301
        return result

    if boundary == "candidate-read":
        context.continuation._runtime.authority.candidate_versions = slow_candidate
    else:
        monkeypatch.setattr(context.usage, "retained_pre_dispatch_failure_many", slow_proof)
    try:
        first = context.pipeline.tick(cycle_id="atomic-read-overrun")
        assert first.revision_states == {"EVIDENCE_HOLD": 1, "ASSESSMENT_INTERRUPTED": 2}
        assert now[0] == 301
        assert len(context.batches) == 1
        assert context.journal.current(context.units[0].revision_id)["facts"]["acquisition_attempt_count"] == 3
        assert all(context.journal.current(unit.revision_id) == before[unit.revision_id] for unit in context.units[1:])
        assert not context.ordinary
        for tick in range(2):
            settled_before = sum(progress["stage"] == "EVIDENCE_HOLD" for progress in (value for _, value in context.journal.iter_summaries()))
            if settled_before == 3:
                break
            context.pipeline.tick(cycle_id=f"atomic-read-overrun-{tick}")
            assert sum(progress["stage"] == "EVIDENCE_HOLD" for progress in (value for _, value in context.journal.iter_summaries())) > settled_before
        assert all(progress["stage"] == "EVIDENCE_HOLD" for progress in (value for _, value in context.journal.iter_summaries()))
        assert len(context.batches) == 3
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
    candidate_versions = context.continuation._runtime.authority.candidate_versions
    batch = context.usage.retained_pre_dispatch_failure_many

    def slow_candidate(version_ids):
        versions = candidate_versions(version_ids)
        now[0], stopped[0] = 301, True
        return versions

    def slow_proof(candidates):
        result = batch(candidates)
        now[0], stopped[0] = 301, True
        return result

    if boundary == "candidate-read":
        context.continuation._runtime.authority.candidate_versions = slow_candidate
    else:
        monkeypatch.setattr(context.usage, "retained_pre_dispatch_failure_many", slow_proof)
    before = {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()}
    try:
        with pytest.raises(VetoError if signal == "stop" else OperatorDrainRequested):
            context.pipeline.tick(cycle_id="stopped-atomic-read-overrun")
        assert {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()} == before
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
    before = {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()}
    try:
        context.pipeline.tick(cycle_id="expired-unproved-target")
        assert context.journal.current(context.units[0].revision_id) == before[context.units[0].revision_id]
        assert context.journal.current(context.units[2].revision_id) == before[context.units[2].revision_id]
        if unproved == "global-corruption":
            assert {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()} == before
        else:
            assert context.journal.current(context.units[1].revision_id)["stage"] == "EVIDENCE_HOLD"
        assert len(context.batches) == 1
        assert not context.ordinary
    finally:
        context.connection.close()


@pytest.mark.parametrize("partition", ([], (), (None,), (object(), object(), object())))
def test_recovery_candidate_partition_never_grants_forged_recovery(tmp_path, monkeypatch, partition):
    context = _prefix(tmp_path, monkeypatch, count=3)
    context.continuation._runtime.authority.candidate_versions = lambda _: partition
    before = {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()}
    try:
        if type(partition) is tuple and len(partition) == 3:
            assert context.continuation.recover_pre_dispatch(
                tuple(context.versions), failure_many=lambda _: pytest.fail("no proved Candidate"),
                before_revision=lambda: True,
            ) == ()
        else:
            from newsroom.control_plane.native_publication import NativePublicationError
            with pytest.raises(NativePublicationError, match="version partition"):
                context.continuation.recover_pre_dispatch(
                    tuple(context.versions), failure_many=lambda _: pytest.fail("no proved Candidate"),
                    before_revision=lambda: True,
                )
        assert {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()} == before
    finally:
        context.connection.close()


def test_none_recovery_partition_is_one_checked_turn_without_changes_and_reproves_next_tick(
    tmp_path, monkeypatch,
):
    context = _prefix(tmp_path, monkeypatch, count=85, admission=True)
    original = context.usage.retained_pre_dispatch_failure_many
    monkeypatch.setattr(context.usage, "retained_pre_dispatch_failure_many", lambda candidates: (None,) * len(candidates))
    before = {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()}
    try:
        context.pipeline.tick(cycle_id="checked-no-proof")
        assert {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()} == before
        assert len(context.batches) == len(context.version_batches) == 1
        assert not context.ordinary
        # A later tick sees fresh model history; no denial survives this call.
        monkeypatch.setattr(context.usage, "retained_pre_dispatch_failure_many", original)
        context.pipeline.tick(cycle_id="fresh-proof-after-denial")
        assert len(context.batches) == len(context.version_batches) == 2
        assert not context.ordinary
        assert all(value["facts"]["reason"] == "ASSESSOR_PRE_DISPATCH_HOLD" for value in (value for _, value in context.journal.iter_summaries()))
    finally:
        context.connection.close()


def test_none_recovery_partition_keeps_contract_revalidation_ordinary(tmp_path, monkeypatch):
    context = _prefix(tmp_path, monkeypatch, count=1, admission=True)
    unit = context.units[0]
    context.continuation._assessment_contract_version = "newsroom.native-evidence-assessor.v20"
    context.journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts={
        **context.journal.current(unit.revision_id)["facts"],
        "assessment_contract_version": "newsroom.native-evidence-assessor.v19",
        "editorial_hold_reason_codes": ["EVIDENCE_VALIDATION_HOLD"],
    })
    before = {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()}
    try:
        assert context.continuation.recover_pre_dispatch(
            (unit.revision_id,), failure_many=lambda candidates: (None,) * len(candidates),
            before_revision=lambda: True,
        ) == ()
        assert {revision: context.journal.current(revision) for revision, _ in context.journal.iter_summaries()} == before
    finally:
        context.connection.close()


@pytest.mark.parametrize("proved", (False, True))
def test_recovery_expands_only_proved_writes_and_preserves_full_pairs(tmp_path, monkeypatch, proved):
    from newsroom.tests.test_native_progress import _retrieval_facts

    context = _prefix(tmp_path, monkeypatch, count=2)
    current = context.journal.current
    originals = {}
    for unit in context.units:
        value = current(unit.revision_id)
        facts = {**_retrieval_facts(), **value["facts"], "unknown_inline": {"future": [True]}}
        context.journal.advance(unit.revision_id, stage=value["stage"], facts=facts)
        originals[unit.revision_id] = current(unit.revision_id)
    reads = []

    def selected(revision_id):
        reads.append(revision_id)
        return current(revision_id)

    monkeypatch.setattr(context.journal, "current", selected)
    try:
        checked = context.continuation.recover_pre_dispatch(
            tuple(originals), before_revision=lambda: True,
            failure_many=(context.usage.retained_pre_dispatch_failure_many if proved
                          else lambda versions: (None,) * len(versions)),
        )
        assert checked == tuple(originals)
        assert reads == (list(originals) if proved else [])
        for revision_id, original in originals.items():
            retained = current(revision_id)
            if proved:
                assert retained["stage"] == "EVIDENCE_HOLD"
                for key in ("retrieval_binding", "retrieval_rights_inventory", "unknown_inline"):
                    assert retained["facts"][key] == original["facts"][key]
            else:
                assert retained == original
    finally:
        context.connection.close()


def test_recovery_denies_pair_only_state_change_after_thin_snapshot(tmp_path, monkeypatch):
    from newsroom.tests.test_native_progress import _retrieval_facts

    context = _prefix(tmp_path, monkeypatch, count=1)
    unit = context.units[0]
    current = context.journal.current
    original = current(unit.revision_id)
    context.journal.advance(unit.revision_id, stage=original["stage"], facts={
        **_retrieval_facts(), **original["facts"],
    })
    calls = []

    def failure_many(versions):
        failures = context.usage.retained_pre_dispatch_failure_many(versions)
        fresh = current(unit.revision_id)
        fresh["facts"]["retrieval_binding"]["request"]["nodes"].append("new retained node")
        context.journal.advance(unit.revision_id, stage=fresh["stage"], facts=fresh["facts"])
        return failures

    monkeypatch.setattr(context.journal, "current", lambda revision: (calls.append(revision), current(revision))[1])
    try:
        checked = context.continuation.recover_pre_dispatch(
            (unit.revision_id,), before_revision=lambda: True, failure_many=failure_many,
        )
        assert checked == ()
        assert calls == []
        assert current(unit.revision_id)["stage"] == "ASSESSMENT_INTERRUPTED"
        assert current(unit.revision_id)["facts"]["retrieval_binding"]["request"]["nodes"][-1] == "new retained node"
    finally:
        context.connection.close()


@pytest.mark.parametrize("field", ("candidate_id", "candidate_version_id"))
def test_proved_recovery_write_denies_rebound_selected_candidate(tmp_path, monkeypatch, field):
    from newsroom.control_plane.native_publication import NativePublicationError

    context = _prefix(tmp_path, monkeypatch, count=1)
    unit = context.units[0]
    version = context.versions[unit.revision_id]
    failure = context.usage.retained_pre_dispatch_failure_many((version,))[0]
    retained = context.journal.current(unit.revision_id)
    context.journal.advance(unit.revision_id, stage=retained["stage"], facts={
        **retained["facts"], field: "rebound-candidate",
    })
    before = context.journal.current(unit.revision_id)
    try:
        with pytest.raises(NativePublicationError, match="Candidate differs"):
            context.continuation._retain_pre_dispatch_hold(
                unit.revision_id, version.version_id, version, failure,
            )
        assert context.journal.current(unit.revision_id) == before
    finally:
        context.connection.close()
