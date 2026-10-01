"""Finite, fresh and globally authenticated assessor recovery batches."""

import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from newsroom.authority.canonical import digest_canonical
from newsroom.control_plane import native_assessor
from newsroom.control_plane.model_usage import WorkEnvelope, WorkloadClass
from newsroom.control_plane.native_assessor import NativeAssessmentExecution
from newsroom.tests.assessor_fixture_support import candidate_value
from newsroom.tests.test_native_assessor import _base_package, _ready_package, _usage


_DIGEST = "sha256:" + "a" * 64


def _target(index=0):
    return SimpleNamespace(
        candidate_id=f"candidate-{index}",
        version_id=f"version-{index}",
        governing_manifest=SimpleNamespace(canonical_digest=_DIGEST),
    )


def _envelope(service, index, *, candidate=None, cycle=None, manifest=None):
    envelope = WorkEnvelope.create(
        cycle_id=cycle or (
            native_assessor._assessment_cycle_id(
                candidate.version_id, _DIGEST, native_assessor.VERSION,
            ) if candidate is not None else f"embedding-{index}"
        ),
        workload_class=(
            WorkloadClass.NATIVE_EVIDENCE_ASSESSOR
            if candidate is not None else WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING
        ),
        admitted_at=datetime(2026, 9, 8, tzinfo=UTC),
        admission_decision_id=None,
        candidate_id=None if candidate is None else candidate.candidate_id,
        hypothesis_digest=None if candidate is None else manifest or _DIGEST,
        evidence_package_digest=None if candidate is None else _DIGEST,
        ingest_id=f"passage-{index}" if candidate is None else None,
        graphiti_attempt_id=None,
    )
    service.open_envelope(envelope)
    return envelope


def _allocated(tmp_path, monkeypatch):
    service, usage = _usage(tmp_path, monkeypatch)
    candidate = candidate_value()
    allocation = usage.begin(
        candidate, _base_package(_ready_package(candidate)[1]), "retained prompt",
    )
    return service, usage, candidate, allocation


def test_predispatch_batch_authenticates_global_history_once_and_closes_read(
    tmp_path, monkeypatch,
):
    service, usage = _usage(tmp_path, monkeypatch)
    for index in range(100):
        _envelope(service, index)
    statements, connections = [], []
    original = sqlite3.connect

    def connect(*args, **kwargs):
        connection = original(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    candidates = tuple(_target(index) for index in range(15))
    results = usage.retained_pre_dispatch_failure_many(candidates)
    assert len(results) == len(candidates)
    assert all(result is not None for result in results)
    assert tuple(result.candidate_version_id for result in results) == tuple(
        candidate.version_id for candidate in candidates
    )
    assert {result.envelope_inventory_digest for result in results} == {
        digest_canonical(())
    }
    counts = Counter(" ".join(statement.split()).upper() for statement in statements)
    assert counts["PRAGMA FOREIGN_KEY_CHECK"] == 1
    assert counts["BEGIN"] == 1
    for table in (
        "model_work_envelopes", "model_invocation_allocations",
        "model_transport_observations",
    ):
        assert sum(
            count for statement, count in counts.items()
            if statement.startswith("SELECT ")
            and f"FROM {table.upper()} ORDER BY" in statement
        ) == 1
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")
    # A caller may commit its journal only after the batch has returned.
    with original(service.path, timeout=0) as writable:
        writable.execute("BEGIN IMMEDIATE")
        writable.execute("UPDATE model_work_envelopes SET admitted_at=admitted_at")


def test_single_predispatch_reader_delegates_to_a_fresh_one_member_batch(
    tmp_path, monkeypatch,
):
    _service, usage = _usage(tmp_path, monkeypatch)
    candidate, calls = _target(), []

    def batch(candidates):
        calls.append(candidates)
        return (None,)

    monkeypatch.setattr(usage, "retained_pre_dispatch_failure_many", batch)
    assert usage.retained_pre_dispatch_failure(candidate) is None
    assert usage.retained_pre_dispatch_failure(candidate) is None
    assert calls == [(candidate,), (candidate,)]


def test_predispatch_batch_preserves_target_allocation_and_inventory_digest(
    tmp_path, monkeypatch,
):
    service, usage, candidate, allocation = _allocated(tmp_path, monkeypatch)
    unrelated = _target()
    dispatch_at = usage.mark_dispatch(allocation)
    results = usage.retained_pre_dispatch_failure_many(
        (candidate, unrelated, None, unrelated)
    )
    assert results[0] is None and results[2] is None
    assert results[1] == results[3] == usage.retained_pre_dispatch_failure(unrelated)
    with sqlite3.connect(service.path) as connection:
        inventory = tuple(connection.execute(
            "SELECT canonical_digest FROM model_work_envelopes "
            "WHERE workload_class=? ORDER BY envelope_id",
            (WorkloadClass.NATIVE_EVIDENCE_ASSESSOR.value,),
        ).fetchone()) + (allocation.canonical_digest,) + tuple(connection.execute(
            "SELECT observation_digest FROM model_transport_observations "
            "WHERE invocation_id=? ORDER BY observation_digest",
            (allocation.invocation_id,),
        ).fetchone())
    assert results[1].envelope_inventory_digest == digest_canonical(inventory)
    assert dispatch_at is not None


@pytest.mark.parametrize("defect", (
    "envelope-json", "envelope-canonical", "allocation-orphan",
    "allocation-json", "transport-orphan", "transport-json", "terminal-orphan",
))
def test_predispatch_batch_denies_every_candidate_on_global_corruption(
    tmp_path, monkeypatch, defect,
):
    service, usage, candidate, allocation = _allocated(tmp_path, monkeypatch)
    dispatch_at = usage.mark_dispatch(allocation)
    if defect == "terminal-orphan":
        usage.complete(
            allocation, outcome="ASSESSOR_PROVIDER_FAILED",
            execution=NativeAssessmentExecution("response", {
                "usage_basis": "PROVIDER_REPORTED", "input_tokens": 1,
                "output_tokens": 1, "cached_read_tokens": 0,
                "cached_write_tokens": 0, "reasoning_tokens": 0,
                "context_tokens": 1, "total_tokens": 2,
            }),
            provider_dispatched=True, dispatch_at=dispatch_at, failure_class="SYSTEMIC",
        )
    with sqlite3.connect(service.path) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        if defect == "envelope-json":
            connection.execute("UPDATE model_work_envelopes SET record_json='{}'")
        elif defect == "envelope-canonical":
            connection.execute("UPDATE model_work_envelopes SET canonical_digest='changed'")
        elif defect == "allocation-orphan":
            connection.execute("UPDATE model_invocation_allocations SET envelope_id='orphan'")
        elif defect == "allocation-json":
            connection.execute("UPDATE model_invocation_allocations SET record_json='[]'")
        elif defect == "transport-orphan":
            connection.execute("UPDATE model_transport_observations SET invocation_id='orphan'")
        elif defect == "transport-json":
            connection.execute("UPDATE model_transport_observations SET record_json='{}'")
        else:
            connection.execute("DELETE FROM model_transport_observations")
            connection.execute("DELETE FROM model_invocation_allocations")
            connection.execute("DELETE FROM model_work_envelopes")
    assert usage.retained_pre_dispatch_failure_many(
        (candidate, _target(), _target(1))
    ) == (None, None, None)


@pytest.mark.parametrize("defect", ("exact", "manifest", "cycle", "route"))
def test_predispatch_batch_preserves_empty_envelope_binding_and_route_holds(
    tmp_path, monkeypatch, defect,
):
    service, usage = _usage(tmp_path, monkeypatch)
    candidate, unrelated = _target(), _target(1)
    _envelope(service, 0, candidate=candidate,
              manifest="sha256:" + "b" * 64 if defect == "manifest" else None,
              cycle="wrong-cycle" if defect == "cycle" else None)
    if defect == "route":
        with service._connection() as connection:
            service._append_route_state(
                connection, route=native_assessor.ROUTE, state="OPEN", reason="QUOTA",
                invocation_id=None, recorded_at=datetime(2026, 9, 8, tzinfo=UTC),
            )
    results = usage.retained_pre_dispatch_failure_many((candidate, unrelated))
    assert (results[0] is not None) is (defect == "exact")
    assert (results[1] is not None) is (defect != "route")
    assert results == tuple(usage.retained_pre_dispatch_failure(item)
                            for item in (candidate, unrelated))


def test_predispatch_batch_is_fresh_after_a_prior_proof(tmp_path, monkeypatch):
    service, usage = _usage(tmp_path, monkeypatch)
    candidates = (_target(), _target(1))
    envelope = _envelope(service, 0)
    assert all(usage.retained_pre_dispatch_failure_many(candidates))
    with sqlite3.connect(service.path) as connection:
        record = envelope.as_record()
        record["cycle_id"] = "changed"
        connection.execute(
            "UPDATE model_work_envelopes SET record_json=?", (json.dumps(record),)
        )
    assert usage.retained_pre_dispatch_failure_many(candidates) == (None, None)
    assert usage.retained_pre_dispatch_failure(candidates[0]) is None


def test_predispatch_batch_keeps_same_candidate_versions_independent(
    tmp_path, monkeypatch,
):
    service, usage = _usage(tmp_path, monkeypatch)
    candidate = _target()
    _envelope(service, 0, candidate=candidate)
    different_version = SimpleNamespace(**{
        **vars(candidate), "version_id": "another-version",
    })
    different_manifest = SimpleNamespace(**{
        **vars(candidate),
        "governing_manifest": SimpleNamespace(canonical_digest="sha256:" + "b" * 64),
    })
    results = usage.retained_pre_dispatch_failure_many(
        (different_version, candidate, different_manifest, candidate)
    )
    assert results[0] is results[2] is None
    assert results[1] is not None and results[1] == results[3]


def test_predispatch_empty_and_malformed_batch_does_not_open_a_snapshot(
    tmp_path, monkeypatch,
):
    _service, usage = _usage(tmp_path, monkeypatch)
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: pytest.fail("unexpected read"))
    assert usage.retained_pre_dispatch_failure_many(()) == ()
    assert usage.retained_pre_dispatch_failure_many((None, object())) == (None, None)
