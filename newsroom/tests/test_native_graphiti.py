from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from newsroom.control_plane import native_graphiti as n
from newsroom.control_plane import cycle
from newsroom.control_plane.store import connect, insert_graphiti_ingest
from newsroom.control_plane.veto import OperatorDrainRequested, VetoError
from newsroom.increment4.contracts import INCREMENT4_ADMITTED_FAMILY_ID
from newsroom.projection.models import ProjectionGenerationState
from newsroom.tests.test_graphiti_operational_readiness import _unit


def _open(tmp_path, monkeypatch, *, ingest, rights=lambda _: {"current": True}, active_generation=True):
    connection = connect(str(tmp_path / "private.sqlite3"))
    authority_path = tmp_path / "authority.sqlite3"
    metadata = sqlite3.connect(authority_path)
    metadata.execute("CREATE TABLE projection_generations(generation_id TEXT PRIMARY KEY, family_id TEXT, state TEXT)")
    if active_generation:
        metadata.execute(
            "INSERT INTO projection_generations VALUES(?,?,?)",
            ("00000000-0000-4000-8000-000000008201", INCREMENT4_ADMITTED_FAMILY_ID, "ACTIVE"),
        )
    metadata.commit()
    metadata.close()
    calls = []
    admission = SimpleNamespace(
        enqueue_complete_receipts=lambda **kw: calls.append(("enqueue", kw)),
        drain=lambda **kw: calls.append(("drain", kw)),
        preflight_decided_cohort=lambda **kw: calls.append(("preflight", kw)),
        finalise_decided_cohort=lambda **kw: calls.append(("finalise", kw)),
    )
    monkeypatch.setattr(n, "EvaluationGraphitiRunner", lambda **kw: object())
    monkeypatch.setattr(n, "compose_existing_graphiti_admission_consumer", lambda *a, **kw: admission)
    monkeypatch.setattr(n, "_ingest", ingest)
    @contextmanager
    def fence():
        calls.append(("fence", {}))
        yield
    system = SimpleNamespace(**dict.fromkeys(("graphiti", "extraction", "objects", "entities", "relations", "increment4")))
    system.authority_store_path = authority_path
    def build_current(request, **kw):
        calls.append(("empty-cohort-build", {"request": request}))
        metadata = sqlite3.connect(authority_path)
        metadata.execute(
            "INSERT INTO projection_generations VALUES(?,?,?)",
            (str(request.generation_id), INCREMENT4_ADMITTED_FAMILY_ID, "ACTIVE"),
        )
        metadata.commit()
        metadata.close()
        return SimpleNamespace(generation=SimpleNamespace(
            generation_id=request.generation_id, state=ProjectionGenerationState.ACTIVE,
        ))
    def status(generation_id, **kw):
        calls.append(("generation-status", {"generation_id": generation_id}))
        return SimpleNamespace(generation=SimpleNamespace(
            generation_id=generation_id, state=ProjectionGenerationState.ACTIVE,
        ))
    system.increment4 = SimpleNamespace(build_current_and_promote=build_current, generation_status=status)
    system.graphiti = SimpleNamespace(attempt_history=lambda *args, **kwargs: ())
    processor = n.NativeGraphitiProcessor(
        system=system, connection=connection, usage=None, proof=None,
        rights_for=rights, stop_check=lambda: None, dispatch_fence=fence,
        clock=lambda: datetime(2026, 9, 8, tzinfo=UTC),
    )
    return processor, connection, calls


def _native(item="one"):
    unit = _unit(item_key=item)
    return replace(unit, proving_run_id="native-source:" + unit.observation_digest)


def _complete(connection, **kw):
    for unit in kw["units"]:
        insert_graphiti_ingest(
            connection, ingest_id=unit.ingest_id, source_id=unit.source_id,
            item_key=unit.item_key, outcome="COMPLETE", proposal_count=0,
            entity_count=0, relation_count=0, failure_code="",
            temporal_basis=unit.temporal().basis.value,
            reference_time=unit.temporal().reference_time.to_text(),
            generation_id="test-isolated-generation", receipt_digest=unit.digest,
        )
    connection.commit()


def test_quantum_defers_only_queued_units_before_rights_spend_or_claim(tmp_path):
    connection = connect(str(tmp_path / "deferred.sqlite3"))
    first = replace(_native("chunks"), chunk_count=2)
    second = replace(first, chunk_ordinal=2, predecessor_ingest_id=first.ingest_id)
    units = (first, second, _native("independent"))
    deferred = []
    before = connection.total_changes
    try:
        attempted = cycle._ingest(
            connection, graphiti=SimpleNamespace(ingest=lambda _: pytest.fail("provider")),
            units=units, max_graphiti=len(units),
            rights_check=lambda _: pytest.fail("rights"),
            rights_fence=lambda _: pytest.fail("fence"),
            clock=lambda: datetime(2026, 9, 12, tzinfo=UTC),
            defer_before_unit=lambda unit: deferred.append(unit.ingest_id) or True,
        )
        assert attempted == 0
        assert deferred == [entry[-1].ingest_id for entry in cycle._queue(connection, units)]
        assert set(deferred) == {first.ingest_id, units[2].ingest_id}
        assert connection.total_changes == before
    finally:
        connection.close()


def test_native_cohort_finalises_once_and_replays_without_new_ingests(tmp_path, monkeypatch):
    processor, connection, calls = _open(tmp_path, monkeypatch, ingest=_complete)
    units = (_native("one"), _native("two"))
    try:
        outcomes = processor.advance(units, cycle_id="native-cycle:1")
        assert [item.state for item in outcomes] == ["GRAPHITI_COMPLETE"] * 2
        assert [entry for entry in calls if entry[0] == "finalise"] == [
            ("finalise", {"ingest_ids": tuple(sorted(unit.ingest_id for unit in units))})
        ]
        assert len([entry for entry in calls if entry[0] == "empty-cohort-build"]) == 0
        processor.advance(units, cycle_id="native-cycle:2")
        assert len([entry for entry in calls if entry[0] == "empty-cohort-build"]) == 0
        assert connection.execute("SELECT count(*) FROM unpublished_graphiti_ingest").fetchone()[0] == 2
    finally:
        connection.close()


def test_native_ingest_permits_the_governed_fallback_route(tmp_path, monkeypatch):
    observed = []

    def ingest(*_args, **kwargs):
        observed.append(kwargs["fallback_permitted"])

    processor, connection, _calls = _open(tmp_path, monkeypatch, ingest=ingest)
    try:
        processor.advance((_native(),), cycle_id="native-fallback-route")
        assert observed == [True]
    finally:
        connection.close()


@pytest.mark.parametrize("generation_id", (None, "00000000-0000-4000-8000-000000008201"))
def test_successive_zero_proposal_cohorts_bootstrap_only_without_active_generation(
    tmp_path, monkeypatch, generation_id
):
    from newsroom.tests.test_graphiti_admission_consumer import (
        _Authority, _Projector, _Rights, _consumer, _seed_receipt,
    )

    def complete(connection, **kwargs):
        for unit in kwargs["units"]:
            _seed_receipt(connection, ingest_id=unit.ingest_id)

    processor, connection, calls = _open(
        tmp_path, monkeypatch, ingest=complete, active_generation=generation_id is not None,
    )
    projector = _Projector()
    authority = _Authority({})
    processor._admission = _consumer(
        connection, authority, projector, _Rights(),
        projection_generation_id=generation_id,
    )
    processor._system.increment4.reconcile_active = lambda *a, **kw: pytest.fail(
        "zero proposals do not change the active graph"
    )
    units = (_native("first-zero"), _native("later-independent-zero"))
    try:
        for number, unit in enumerate(units, 1):
            outcome, = processor.advance((unit,), cycle_id=f"zero:{number}")
            assert outcome.state == "GRAPHITI_COMPLETE"
            assert outcome.receipt_digest is not None
            assert outcome.reason is None
        assert len(processor._completed) == 2
        assert not authority.calls
        assert not projector.generation_calls
        assert not projector.generation_effects
        assert sum(name == "empty-cohort-build" for name, _ in calls) == (
            1 if generation_id is None else 0
        )
        assert sum(name == "generation-status" for name, _ in calls) == (
            1 if generation_id is None else 2
        )
        # Zero proposals add no admitted entity/relation effects. A fresh native
        # pipeline nevertheless needs its first real generation for retrieval.
        assert connection.execute(
            "SELECT count(*) FROM unpublished_graphiti_admission_queue"
        ).fetchone()[0] == 0
    finally:
        connection.close()


@pytest.mark.parametrize("mismatch", ("duplicate", "identity", "state", "first_build"))
def test_zero_proposal_bootstrap_keeps_generation_metadata_fail_closed(
    tmp_path, monkeypatch, mismatch
):
    processor, connection, calls = _open(
        tmp_path, monkeypatch, ingest=_complete,
        active_generation=mismatch != "first_build",
    )
    if mismatch == "duplicate":
        metadata = sqlite3.connect(processor._system.authority_store_path)
        metadata.execute(
            "INSERT INTO projection_generations VALUES(?,?,?)",
            ("00000000-0000-4000-8000-000000008202", INCREMENT4_ADMITTED_FAMILY_ID, "ACTIVE"),
        )
        metadata.commit()
        metadata.close()
    elif mismatch == "first_build":
        processor._system.increment4.build_current_and_promote = (
            lambda *a, **kw: SimpleNamespace(generation=SimpleNamespace(state=None))
        )
    else:
        processor._system.increment4.generation_status = (
            lambda generation_id, **kw: SimpleNamespace(generation=SimpleNamespace(
                generation_id=None if mismatch == "identity" else generation_id,
                state=None if mismatch == "state" else ProjectionGenerationState.ACTIVE,
            ))
        )
    try:
        outcome, = processor.advance((_native(),), cycle_id="bad-generation")
        assert outcome.state == "ADMISSION_HOLD"
        assert outcome.reason == (
            "native empty-cohort graph is not active" if mismatch == "first_build"
            else "native active graph metadata differs"
        )
        assert not processor._completed
        assert not any(name == "empty-cohort-build" for name, _ in calls)
    finally:
        connection.close()


def test_native_graphiti_propagates_operator_drain_after_settled_ingest(
    tmp_path, monkeypatch,
):
    drain = threading.Event()

    def settle_then_drain(connection, **kwargs):
        assert kwargs["operator_drain_requested"]() is False
        _complete(connection, **kwargs)
        drain.set()

    processor, connection, calls = _open(
        tmp_path, monkeypatch, ingest=settle_then_drain,
    )
    processor._operator_drain_requested = drain.is_set
    unit = _native()
    try:
        with pytest.raises(OperatorDrainRequested):
            processor.advance((unit,), cycle_id="native-cycle:drain")
        assert connection.execute(
            "SELECT outcome FROM unpublished_graphiti_ingest WHERE ingest_id=?",
            (unit.ingest_id,),
        ).fetchone() == ("COMPLETE",)
        assert not any(name in {"enqueue", "drain", "finalise"} for name, _ in calls)
    finally:
        connection.close()


def test_native_graphiti_records_completed_cohort_before_post_projection_drain(
    tmp_path, monkeypatch,
):
    drain = threading.Event()
    processor, connection, calls = _open(tmp_path, monkeypatch, ingest=_complete)

    def finalise(**kwargs):
        calls.append(("finalise", kwargs))
        drain.set()

    processor._admission.finalise_decided_cohort = finalise
    processor._operator_drain_requested = drain.is_set
    unit = _native()
    try:
        with pytest.raises(OperatorDrainRequested):
            processor.advance((unit,), cycle_id="native-cycle:projection-drain")
        retained = [
            json.loads(row[0]) for row in connection.execute(
                "SELECT payload_json FROM ledger "
                "WHERE kind='NATIVE_GRAPHITI_COHORT' ORDER BY seq"
            )
        ]
        assert [item["state"] for item in retained] == ["STARTED", "COMPLETE"]

        drain.clear()
        result = processor.advance((unit,), cycle_id="native-cycle:restart")
        assert result[0].state == "GRAPHITI_COMPLETE"
        assert len([name for name, _ in calls if name == "finalise"]) == 1
    finally:
        connection.close()


def test_native_worker_rejects_historical_campaign_before_dispatch(tmp_path, monkeypatch):
    processor, connection, calls = _open(tmp_path, monkeypatch, ingest=lambda *a, **kw: pytest.fail("dispatch"))
    try:
        with pytest.raises(ValueError, match="historical campaign"):
            processor.advance((_unit(),), cycle_id="native-cycle:1")
        assert calls == []
    finally:
        connection.close()


def test_native_missing_or_partial_extraction_never_becomes_complete(tmp_path, monkeypatch):
    processor, connection, calls = _open(tmp_path, monkeypatch, ingest=lambda *a, **kw: None)
    try:
        result = processor.advance((_native(),), cycle_id="native-cycle:1")
        assert result[0].state == "GRAPHITI_HOLD"
        assert not any(name == "finalise" for name, _ in calls)
    finally:
        connection.close()


def test_native_rights_revocation_blocks_projection_after_extraction(tmp_path, monkeypatch):
    processor, connection, calls = _open(tmp_path, monkeypatch, ingest=_complete, rights=lambda _: None)
    try:
        result = processor.advance((_native(),), cycle_id="native-cycle:1")
        assert result[0].state == "ADMISSION_HOLD"
        assert not any(name == "finalise" for name, _ in calls)
    finally:
        connection.close()


def test_rights_hold_isolates_complete_revision_and_resumes_without_reprojection(
    tmp_path, monkeypatch
):
    held = set()
    revoked = [False]

    def complete_then_revoke(connection, **kwargs):
        assert all(kwargs["rights_check"](unit) for unit in kwargs["units"])
        _complete(connection, units=kwargs["units"])
        if not revoked[0]:
            held.add("two")
            revoked[0] = True

    processor, connection, calls = _open(
        tmp_path,
        monkeypatch,
        ingest=complete_then_revoke,
        rights=lambda unit: None if unit.item_key in held else {"current": True},
    )
    units = (_native("one"), _native("two"))
    try:
        first = processor.advance(units, cycle_id="one")
        assert [item.state for item in first] == [
            "GRAPHITI_COMPLETE",
            "ADMISSION_HOLD",
        ]
        assert [
            kwargs["ingest_ids"]
            for name, kwargs in calls
            if name == "finalise"
        ] == [(units[0].ingest_id,)]

        held.clear()
        second = processor.advance(units, cycle_id="two")
        assert [item.state for item in second] == ["GRAPHITI_COMPLETE"] * 2
        assert [
            kwargs["ingest_ids"]
            for name, kwargs in calls
            if name == "finalise"
        ] == [(units[0].ingest_id,), (units[1].ingest_id,)]
    finally:
        connection.close()


def test_admission_preflight_isolates_bad_revision_then_projects_only_verified(
    tmp_path, monkeypatch
):
    from newsroom.control_plane.graphiti_admission import (
        GraphitiAdmissionConsumerError,
    )

    processor, connection, calls = _open(
        tmp_path, monkeypatch, ingest=_complete
    )
    units = (_native("one"), _native("two"))
    held = {units[1].ingest_id}

    def preflight(*, ingest_ids):
        calls.append(("preflight", {"ingest_ids": ingest_ids}))
        if set(ingest_ids) & held:
            raise GraphitiAdmissionConsumerError(
                "exact Graphiti cohort contains non-terminal work"
            )

    processor._admission.preflight_decided_cohort = preflight
    try:
        first = processor.advance(units, cycle_id="one")
        assert [item.state for item in first] == [
            "GRAPHITI_COMPLETE",
            "ADMISSION_HOLD",
        ]
        assert first[1].reason == (
            "exact Graphiti cohort contains non-terminal work"
        )
        assert [
            kwargs["ingest_ids"]
            for name, kwargs in calls
            if name == "finalise"
        ] == [(units[0].ingest_id,)]
        assert len([
            entry for entry in calls if entry[0] == "empty-cohort-build"
        ]) == 0

        held.clear()
        second = processor.advance(units, cycle_id="two")
        assert [item.state for item in second] == ["GRAPHITI_COMPLETE"] * 2
        assert [
            kwargs["ingest_ids"]
            for name, kwargs in calls
            if name == "finalise"
        ] == [(units[0].ingest_id,), (units[1].ingest_id,)]
        assert len([
            entry for entry in calls if entry[0] == "empty-cohort-build"
        ]) == 0
    finally:
        connection.close()


def test_actual_consumer_preflight_has_no_projection_and_reuses_finaliser_checks(
    tmp_path
):
    from newsroom.control_plane.graphiti_admission import (
        GraphitiAdmissionConsumerError,
        GraphitiProposalAdmissionAction,
    )
    from newsroom.extraction.types import ExtractionProposalKind
    from newsroom.tests.test_graphiti_admission_consumer import (
        _Authority,
        _Projector,
        _Rights,
        _consumer,
        _draft,
        _seed_receipt,
    )

    connection = connect(str(tmp_path / "actual-preflight.sqlite3"))
    first_id = "00000000-0000-4000-8000-000000008101"
    second_id = "00000000-0000-4000-8000-000000008102"
    first = _draft("entity.8101", ExtractionProposalKind.ENTITY_MENTION)
    second = _draft("entity.8102", ExtractionProposalKind.ENTITY_MENTION)
    _seed_receipt(connection, first, ingest_id=first_id)
    _seed_receipt(connection, second, ingest_id=second_id)
    projector = _Projector()
    consumer = _consumer(
        connection,
        _Authority({
            first.local_id: GraphitiProposalAdmissionAction.ADMIT,
            second.local_id: GraphitiProposalAdmissionAction.ADMIT,
        }),
        projector,
        _Rights(),
    )
    try:
        assert consumer.enqueue_complete_receipts(
            ingest_ids=(first_id, second_id)
        ) == 2
        assert consumer.drain(
            worker_id="first", limit=1, ingest_ids=(first_id,)
        ).decided == 1

        consumer.preflight_decided_cohort(ingest_ids=(first_id,))
        with pytest.raises(
            GraphitiAdmissionConsumerError, match="not completely decided"
        ):
            consumer.preflight_decided_cohort(ingest_ids=(second_id,))
        assert projector.generation_calls == []

        assert consumer.drain(
            worker_id="second", limit=1, ingest_ids=(second_id,)
        ).decided == 1
        consumer.preflight_decided_cohort(ingest_ids=(second_id,))
        result = consumer.finalise_decided_cohort(
            ingest_ids=(first_id, second_id)
        )
        assert result.projected == 2
        assert len(projector.generation_calls) == 1
        consumer.preflight_decided_cohort(ingest_ids=(first_id,))
        assert len(projector.generation_calls) == 1
    finally:
        connection.close()


def test_native_fence_passes_bounded_deadline_and_current_rights(tmp_path, monkeypatch):
    processor, connection, calls = _open(tmp_path, monkeypatch, ingest=lambda *a, **kw: None)
    try:
        with processor._fence(_native()) as authority:
            assert authority.rights == {"current": True}
            assert authority.deadline == datetime(2026, 9, 8, 0, 15, tzinfo=UTC)
        assert calls == [("fence", {})]
    finally:
        connection.close()


def test_native_fence_authorises_its_provider_callback_thread_only_while_held(
    tmp_path, monkeypatch,
):
    proving = tmp_path / "proving.sqlite3"
    with sqlite3.connect(proving) as database:
        database.executescript(
            """
            CREATE TABLE proving_runs(run_id TEXT PRIMARY KEY);
            CREATE TABLE proving_gates(
                run_id TEXT NOT NULL, gate_id TEXT NOT NULL, status TEXT NOT NULL
            );
            INSERT INTO proving_runs VALUES('run-1');
            INSERT INTO proving_gates VALUES(
                'run-1', 'NO_ACTIVE_HUMAN_EMERGENCY_STOP', 'PASS'
            );
            """
        )
    monkeypatch.setattr(cycle, "_PROVING_FENCE_TIMEOUT_SECONDS", 0.05)
    processor, connection, _ = _open(tmp_path, monkeypatch, ingest=lambda *a, **kw: None)
    processor._dispatch_fence = lambda: cycle.owner_emergency_stop_fence(str(proving))
    processor._stop_check = lambda: cycle.assert_no_owner_emergency_stop(str(proving))
    errors = []
    try:
        with processor._fence(_native()) as authority:
            worker = threading.Thread(
                target=lambda: _capture_error(authority.owner_stop_check, errors)
            )
            worker.start()
            worker.join(timeout=0.5)
            assert not worker.is_alive()
            assert errors == []
            parent_pid = n.os.getpid()
            with monkeypatch.context() as child:
                child.setattr(n.os, "getpid", lambda: parent_pid + 1)
                with pytest.raises(VetoError, match="belongs to another process"):
                    authority.owner_stop_check()
        with pytest.raises(VetoError, match="fence has expired"):
            authority.owner_stop_check()
    finally:
        connection.close()


def _capture_error(operation, errors):
    try:
        operation()
    except Exception as exc:
        errors.append(exc)


def test_incomplete_chunk_prefix_does_not_poison_the_completed_revision(tmp_path, monkeypatch):
    from newsroom.control_plane.corpus import MAX_EPISODE_BYTES
    base = replace(_native(), body="Retained source text. " * (MAX_EPISODE_BYTES // 22 + 1))
    units = tuple(replace(base, chunk_ordinal=ordinal, chunk_count=2) for ordinal in (1, 2))
    ready = [units[:1]]
    def ingest(connection, **kw):
        _complete(connection, units=ready[0])
    processor, connection, calls = _open(tmp_path, monkeypatch, ingest=ingest)
    try:
        first = processor.advance(units, cycle_id="one")
        assert not any(item.state == "GRAPHITI_COMPLETE" for item in first)
        assert not any(name == "finalise" for name, _ in calls)
        ready[0] = units
        second = processor.advance(units, cycle_id="two")
        assert all(item.state == "GRAPHITI_COMPLETE" for item in second)
        assert [kw["ingest_ids"] for name, kw in calls if name == "finalise"] == [tuple(sorted(unit.ingest_id for unit in units))]
    finally:
        connection.close()


def test_completed_cohort_continues_a_subset_without_rebuilding(tmp_path, monkeypatch):
    processor, connection, calls = _open(tmp_path, monkeypatch, ingest=_complete)
    units = (_native("one"), _native("two"))
    try:
        processor.advance(units, cycle_id="one")
        before = len(calls)
        result = processor.advance(units[:1], cycle_id="resume-after-first-revision-journal-write")
        assert result[0].state == "GRAPHITI_COMPLETE"
        assert not any(name == "finalise" for name, _ in calls[before:])
    finally:
        connection.close()


def test_native_worker_rejects_another_observation_before_dispatch(tmp_path, monkeypatch):
    processor, connection, calls = _open(tmp_path, monkeypatch, ingest=lambda *a, **kw: pytest.fail("dispatch"))
    try:
        with pytest.raises(ValueError, match="observation provenance"):
            processor.advance((replace(_native(), observation_digest="sha256:" + "f" * 64),), cycle_id="one")
        assert not calls
    finally:
        connection.close()


def test_native_required_route_hold_is_not_reported_as_a_rights_failure(tmp_path, monkeypatch):
    from newsroom.control_plane.model_usage import ModelUsageService

    processor, connection, calls = _open(tmp_path, monkeypatch, ingest=lambda *a, **kw: None)
    usage = ModelUsageService(str(tmp_path / "private.sqlite3"))
    usage.open_route_circuit(
        route="GRAPHITI_EMBEDDING", reason="CALL_SHAPE_DRIFT",
        invocation_id=None, recorded_at=datetime(2026, 9, 9, tzinfo=UTC),
    )
    processor._usage = usage
    try:
        outcome, = processor.advance((_native(),), cycle_id="native-route-held")
        assert outcome.state == "GRAPHITI_HOLD"
        assert outcome.reason == "REQUIRED_MODEL_ROUTE_CIRCUIT_OPEN"
        assert not any(name == "finalise" for name, _ in calls)
    finally:
        connection.close()


@pytest.mark.parametrize("retained", [True, False])
@pytest.mark.parametrize("subscription_outcome", ["FAILED", "TIMEOUT"])
def test_native_advance_settles_subscription_usage_before_or_after_dispatch(
    tmp_path, monkeypatch, retained, subscription_outcome,
):
    from newsroom.control_plane.model_usage import ModelUsageService

    unit = _native()
    order = []
    processor, connection, _ = _open(
        tmp_path, monkeypatch, ingest=lambda *a, **kw: order.append("ingest"),
    )
    usage = ModelUsageService(str(tmp_path / "private.sqlite3"))
    settled = []
    from newsroom.tests.test_model_usage_receipts import _policy
    policy = _policy()
    usage.register_policy(policy)
    # These rows test selection/wiring only. The usage-service tests separately
    # prove authority, canonical bindings, dispatch and policy-derived estimates.
    for number, ingest_id, workload, provider, status, failure, outcome in (
        (1, unit.ingest_id, "GRAPHITI_CHAT_PRIMARY", "cursor-agent-cli", "UNREPORTED", "MISSING_PROVIDER_TELEMETRY", subscription_outcome),
        (2, "other-ingest", "GRAPHITI_CHAT_PRIMARY", "cursor-agent-cli", "UNREPORTED", "MISSING_PROVIDER_TELEMETRY", "FAILED"),
        (3, unit.ingest_id, "GRAPHITI_EMBEDDING", "openrouter", "UNREPORTED", "MISSING_PROVIDER_TELEMETRY", "FAILED"),
        (4, unit.ingest_id, "GRAPHITI_CHAT_PRIMARY", "cursor-agent-cli", "REPORTED", "NONE", "FAILED"),
        (5, unit.ingest_id, "GRAPHITI_CHAT_FALLBACK", "grok-build-cli", "UNREPORTED", "MISSING_PROVIDER_TELEMETRY", "FAILED"),
        # Unknown/ambiguous terminals are not native failed-call estimates.
        (56, unit.ingest_id, "GRAPHITI_CHAT_PRIMARY", "cursor-agent-cli", "UNREPORTED", "MISSING_PROVIDER_TELEMETRY", "AMBIGUOUS_DISPATCH"),
        (57, unit.ingest_id, "GRAPHITI_CHAT_FALLBACK", "grok-build-cli", "UNREPORTED", "MISSING_PROVIDER_TELEMETRY", "CANCELLED"),
        (58, unit.ingest_id, "GRAPHITI_CHAT_PRIMARY", "cursor-agent-cli", "UNREPORTED", "MISSING_PROVIDER_TELEMETRY", "COMPLETED"),
    ):
        identity = str(number)
        connection.execute(
            "INSERT INTO model_work_envelopes VALUES(?,?,?,?,?,?)",
            (identity, "cycle", (
                "GRAPHITI_CHAT_PRIMARY" if workload == "GRAPHITI_CHAT_FALLBACK" else workload
            ), "now", identity,
             json.dumps({"ingest_id": ingest_id})),
        )
        connection.execute(
            "INSERT INTO model_invocation_allocations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (identity, identity, "cycle", 1, workload, policy.canonical_digest, provider,
             workload, "model", identity, None, "now", identity, "{}"),
        )
        connection.execute(
            "INSERT INTO model_invocation_terminals VALUES(?,?,?,?,?,?,?)",
            ("terminal-" + identity, identity, status, outcome, failure, "now", "{}"),
        )
    # Settled and unrelated history must not enter the current-ingest selector.
    for number in range(6, 56):
        identity = str(number)
        connection.execute(
            "INSERT INTO model_work_envelopes VALUES(?,?,?,?,?,?)",
            (identity, "old-cycle", "GRAPHITI_CHAT_PRIMARY", "now", identity,
             json.dumps({"ingest_id": f"old-ingest-{number}"})),
        )
        connection.execute(
            "INSERT INTO model_invocation_allocations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (identity, identity, "old-cycle", 1, "GRAPHITI_CHAT_PRIMARY",
             policy.canonical_digest, "cursor-agent-cli", "route", "model", identity,
             None, "now", identity, "{}"),
        )
        connection.execute(
            "INSERT INTO model_invocation_terminals VALUES(?,?,?,?,?,?,?)",
            ("terminal-" + identity, identity, "UNREPORTED", "FAILED",
             "MISSING_PROVIDER_TELEMETRY", "now", "{}"),
        )
        connection.execute(
            "INSERT INTO model_usage_conservative_dispositions "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("disposition-" + identity, identity, "terminal-" + identity,
             identity, policy.canonical_digest, "plan-" + identity,
             "authority", "owner", "reference", "now", "now", "ESTIMATED", "{}"),
        )
    connection.commit()
    if not retained:
        terminal = connection.execute(
            "SELECT * FROM model_invocation_terminals WHERE invocation_id='1'"
        ).fetchone()
        connection.execute("DELETE FROM model_invocation_terminals WHERE invocation_id='1'")
        connection.commit()

        def ingest(*args, **kwargs):
            order.append("ingest")
            connection.execute("INSERT INTO model_invocation_terminals VALUES(?,?,?,?,?,?,?)", terminal)
            connection.commit()

        monkeypatch.setattr(n, "_ingest", ingest)

    def settle(**kwargs):
        order.append("settle")
        settled.append(kwargs)
        # Match the public method's retained disposition so the post-dispatch
        # reconciliation does not rediscover this already settled invocation.
        connection.execute(
            "INSERT INTO model_usage_conservative_dispositions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("disposition-" + kwargs["invocation_id"], kwargs["invocation_id"],
             kwargs["expected_terminal_digest"], kwargs["expected_allocation_digest"], policy.canonical_digest,
             "plan-" + kwargs["invocation_id"],
             "authority-" + kwargs["invocation_id"],
             "owner", "reference", "now", "now", "ESTIMATED", "{}"),
        )
        connection.commit()

    monkeypatch.setattr(usage, "disposition_native_unreported_subscription_usage", settle, raising=False)
    cancelled_fallbacks = []
    monkeypatch.setattr(usage, "disposition_native_graphiti_fallback_cancellation",
                        lambda **kwargs: cancelled_fallbacks.append(kwargs["invocation_id"]))
    monkeypatch.setattr(
        n, "graphiti_required_route_holds", lambda _, **_values: ()
    )
    processor._usage = usage
    outcomes = processor.advance((unit,), cycle_id="native-settlement")
    assert order == (
        ["settle", "settle", "ingest"]
        if retained
        else ["settle", "ingest", "settle"]
    )
    assert [item["invocation_id"] for item in settled] == (
        ["1", "5"] if retained else ["5", "1"]
    )
    assert set(cancelled_fallbacks) == {"57"}
    assert outcomes[0].state == "GRAPHITI_HOLD"
    assert connection.execute(
        "SELECT usage_status FROM model_invocation_terminals WHERE invocation_id='1'"
    ).fetchone()[0] == "UNREPORTED"
    assert connection.execute(
        "SELECT count(*) FROM model_usage_conservative_dispositions "
        "WHERE invocation_id IN ('56','57','58')"
    ).fetchone()[0] == 0
    connection.close()


def test_missing_subscription_usage_selector_scans_only_relevant_liabilities(
    tmp_path,
):
    from newsroom.control_plane.model_usage import ModelUsageService

    path = str(tmp_path / "private.sqlite3")
    ModelUsageService(path)
    connection = connect(path)
    try:
        plan = connection.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT a.invocation_id,a.canonical_digest,t.terminal_digest,e.record_json "
            "FROM model_work_envelopes e INDEXED BY model_usage_native_graphiti_ingest "
            "JOIN model_invocation_allocations a ON a.envelope_id=e.envelope_id "
            "JOIN model_invocation_terminals t ON t.invocation_id=a.invocation_id "
            "WHERE e.workload_class='GRAPHITI_CHAT_PRIMARY' "
            "AND json_extract(e.record_json, '$.ingest_id')=? "
            "AND a.workload_class='GRAPHITI_CHAT_PRIMARY' "
            "AND a.provider='cursor-agent-cli' AND t.usage_status='UNREPORTED' "
            "AND t.failure_class='MISSING_PROVIDER_TELEMETRY' "
            "AND t.outcome IN ('FAILED','TIMEOUT') "
            "AND NOT EXISTS (SELECT 1 FROM model_usage_conservative_dispositions d "
            "WHERE d.invocation_id=a.invocation_id)",
            ("selected-ingest",),
        ).fetchall()
    finally:
        connection.close()
    details = tuple(str(row[3]) for row in plan)
    assert any("SEARCH e USING INDEX model_usage_native_graphiti_ingest" in detail
               for detail in details), details
    assert not any("SCAN" in detail or "TEMP" in detail for detail in details), details


@pytest.mark.parametrize('outcome', ('MALFORMED_OUTPUT', 'AMBIGUOUS_EFFECT', 'COMPLETE'))
@pytest.mark.parametrize('local_failure_recorded', (False, True))
def test_terminal_authority_outcome_is_not_retried_as_an_internal_error(
    tmp_path, monkeypatch, outcome, local_failure_recorded,
):
    from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome
    from newsroom.control_plane.store import record_graphiti_failure

    queued = []
    processor, connection, _ = _open(
        tmp_path, monkeypatch,
        ingest=lambda _connection, **kwargs: queued.extend(kwargs['units']),
    )
    unit = _native()
    processor._system.graphiti = SimpleNamespace(attempt_history=lambda *args, **kwargs: (
        SimpleNamespace(outcome=GraphitiAdapterOutcome(outcome), failure_code='ORIGINAL_FAILURE'),
    ))
    if local_failure_recorded:
        record_graphiti_failure(connection, ingest_id=unit.ingest_id,
                                source_id=unit.source_id, item_key=unit.item_key,
                                outcome=outcome, failure_code='ORIGINAL_FAILURE')
        connection.commit()
    try:
        before = connection.execute('SELECT count(*) FROM ledger').fetchone()[0]
        result, = processor.advance((unit,), cycle_id='native-terminal')
        assert not queued
        assert result.state == 'GRAPHITI_HOLD'
        assert result.reason == (
            'RETAINED_COMPLETE_RECONCILIATION_REQUIRED' if outcome == 'COMPLETE'
            else outcome + ':ORIGINAL_FAILURE'
        )
        assert connection.execute('SELECT count(*) FROM ledger').fetchone()[0] == before
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("settled_provider_attempts", "expected_state"),
    [((3,), "GRAPHITI_COMPLETE"), ((1, 2, 3), "GRAPHITI_HOLD")],
)
def test_native_recovered_ambiguous_attempt_crosses_private_attempt_gap_once(
    tmp_path, monkeypatch, settled_provider_attempts, expected_state,
):
    """Exact regression: authority attempt 3, rejected private 4, successor 5."""
    from newsroom.control_plane.store import (
        insert_graphiti_attempt_receipt,
        record_graphiti_failure,
        reserve_graphiti_spend,
    )
    from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome
    from newsroom.control_plane.model_usage import GraphitiIngestRetryEvidence
    from newsroom.authority.canonical import digest_canonical
    from newsroom.authority.types import UtcTimestamp
    from newsroom.graphiti_adapter import RecoveredAmbiguousProgressionProof
    from newsroom.graphiti_adapter.evaluation_attempt import (
        evaluation_attempt_for_body,
    )
    from newsroom.tests.test_graphiti_corpus_ingest import (
        _complete as completed_result,
    )

    prepared = []
    attempts = []

    class Runner:
        requires_canonical_control_plane_stores = True

        def prepare_recovered_ambiguous_successor(self, **values):
            prepared.append(values)
            current = values["unit"]
            third = self._attempt(current, number=3)
            instant = UtcTimestamp.parse("2026-09-09T10:41:27.849076Z")
            self.recovery = RecoveredAmbiguousProgressionProof(
                authoritative_attempt_id=third.attempt_id,
                authoritative_attempt_digest=digest_canonical({"attempt": 3}),
                authoritative_attempt_number=3,
                authoritative_run_version_id=third.extraction_request.run_version_id,
                authoritative_recorded_at=instant,
                skipped_attempt_number=4,
                skipped_receipt_digest=digest_canonical({"receipt": 4}),
                skipped_ledger_sequence=4,
                skipped_ledger_digest=digest_canonical({"ledger": 4}),
                skipped_recorded_at=instant,
                settled_usage_evidence_digest=digest_canonical({"usage": 3}),
                recovery_marker_digest=digest_canonical({"marker": 3}),
                marker_attempt_number=3,
                marker_workspace_id=third.workspace_id,
                marker_input_digest=digest_canonical({"marker-input": 3}),
                input_binding_digest=third.extraction_request.input_binding.digest,
                ingest_id=current.ingest_id,
            )
            return True

        @staticmethod
        def _attempt(current, *, number, recovery=None):
            authority = current.authority
            assert authority is not None
            return evaluation_attempt_for_body(
                episode_body=current.episode_body,
                ingest_id=current.ingest_id,
                proving_run_id=current.proving_run_id,
                source_id=current.source_id,
                item_key=current.item_key,
                observation_digest=current.observation_digest,
                published_at=current.published_at,
                updated_at=current.updated_at,
                effective_revision=current.effective_revision,
                canonical_url=current.canonical_url,
                revision_digest=current.revision_digest,
                representation_digest=current.representation_digest,
                authority_ids=(
                    authority.admission_id,
                    authority.access_decision_id,
                    authority.definition_id,
                    authority.definition_version_id,
                    authority.item_id,
                    authority.revision_id,
                    authority.representation_id,
                ),
                attempt_number=number,
                recovered_ambiguous_progression=recovery,
                extraction_previous_version_number=3 if recovery else None,
                extraction_previous_run_version_id=(
                    recovery.authoritative_run_version_id if recovery else None
                ),
            )

        def ingest_with_usage(self, current, **_values):
            attempt = self._attempt(current, number=5, recovery=self.recovery)
            attempts.append(attempt)
            return completed_result(
                current, proposal_count=0, entity_count=0, relation_count=0
            )

        def ingest(self, current):
            return self.ingest_with_usage(current)

        def ingest_until(self, current, **_values):
            return self.ingest_with_usage(current)

        def finalise_usage(self, *_args, **_values):
            return None

    monkeypatch.setattr(n, "EvaluationGraphitiRunner", lambda **_values: Runner())

    processor, connection, _ = _open(tmp_path, monkeypatch, ingest=cycle._ingest)
    unit = _native("recovered-gap")
    prior = SimpleNamespace(
        attempt_number=3,
        outcome=GraphitiAdapterOutcome.AMBIGUOUS_EFFECT,
        failure_code="AMBIGUOUS_EFFECT",
        canonical_digest="sha256:" + "3" * 64,
    )
    processor._system.graphiti = SimpleNamespace(
        attempt_history=lambda *_args, **_values: (prior,),
    )
    processor._runner = Runner()
    processor._usage = SimpleNamespace(
        native_graphiti_ingest_retry_evidence_many=lambda **_values: {
            unit.ingest_id: GraphitiIngestRetryEvidence(
                attempt_numbers=(1, 2, 3, 4),
                zero_dispatch_attempts=(
                    (1, 2) if settled_provider_attempts == (3,) else ()
                ),
                settled_provider_attempts=settled_provider_attempts,
                latest_settled_provider_attempt=settled_provider_attempts[-1],
                unresolved_attempts=(4,),
            )
        }
    )
    processor._settle_missing_subscription_usage = lambda _units: None
    monkeypatch.setattr(
        n, "graphiti_required_route_holds", lambda _usage, **_values: ()
    )
    monkeypatch.setattr(
        cycle, "graphiti_required_route_holds", lambda _usage, **_values: ()
    )
    try:
        for number in range(1, 5):
            reserve_graphiti_spend(
                connection,
                spend_id=f"{unit.ingest_id}:{number}",
                ingest_id=unit.ingest_id,
                attempt_number=number,
                proving_run_id=unit.proving_run_id,
                generation_id=cycle.GRAPHITI_GENERATION_ID,
                reserved_gbp_microunits=500_000,
                ceiling_gbp_microunits=None,
            )
            record_graphiti_failure(
                connection,
                ingest_id=unit.ingest_id,
                source_id=unit.source_id,
                item_key=unit.item_key,
                outcome=("AMBIGUOUS_EFFECT" if number == 3 else "FAILED"),
                failure_code=(
                    "AMBIGUOUS_EFFECT" if number == 3 else "PRODUCER_INTERNAL_ERROR"
                ),
            )
            insert_graphiti_attempt_receipt(
                connection,
                ingest_id=unit.ingest_id,
                attempt_number=number,
                outcome=("AMBIGUOUS_EFFECT" if number == 3 else "FAILED"),
                receipt={
                    "attempt_number": number,
                    "ingest_id": unit.ingest_id,
                    "outcome": "AMBIGUOUS_EFFECT" if number == 3 else "FAILED",
                    **(
                        {
                            "binding_failure": "RESULT_CONTRACT_REJECTED",
                            "binding_failure_stage": "UNCLASSIFIED_RESULT_BOUNDARY",
                            "binding_failure_type": "GraphitiAdapterVersionConflict",
                            "chat_invocation_count": 0,
                        }
                        if number == 4 else {}
                    ),
                },
            )
        connection.commit()

        result, = processor.advance((unit,), cycle_id="recovered-gap")
        assert result.state == expected_state
        assert len(prepared) == 1
        assert prepared[0]["unit"].ingest_id == unit.ingest_id
        assert prepared[0]["authoritative_attempt"] is prior
        assert prepared[0]["private_successor_attempt_number"] == 5
        assert len(attempts) == int(expected_state == "GRAPHITI_COMPLETE")
        if not attempts:
            return
        assert attempts[0].attempt_number == 5
        assert attempts[0].expected_previous_attempt_id == (
            attempts[0].recovered_ambiguous_progression.authoritative_attempt_id
        )
        assert attempts[0].extraction_request.version_number == 4
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("authenticated", "settled", "unresolved", "expected_state"),
    [
        (True, (3, 5), (4,), "GRAPHITI_COMPLETE"),
        (False, (3, 5), (4,), "GRAPHITI_HOLD"),
        (True, (3,), (4, 5), "GRAPHITI_HOLD"),
    ],
)
def test_native_recovered_gap_allows_only_accounted_attempt_six(
    tmp_path, monkeypatch, authenticated, settled, unresolved, expected_state,
):
    """Attempt 6 retains rejection 4 without hiding a second unresolved gap."""
    from newsroom.control_plane.model_usage import GraphitiIngestRetryEvidence
    from newsroom.control_plane.store import (
        insert_graphiti_attempt_receipt,
        record_graphiti_failure,
        reserve_graphiti_spend,
    )
    from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome
    from newsroom.tests.test_graphiti_corpus_ingest import _complete

    calls = []

    class Runner:
        requires_canonical_control_plane_stores = True

        def authenticate_retained_recovered_ambiguous_progression(self, **values):
            calls.append(("authenticate", values["next_attempt_number"]))
            return 4 if authenticated else None

        def ingest_with_usage(self, unit, **_values):
            calls.append(("ingest", unit.attempt_number))
            return _complete(unit, proposal_count=0, entity_count=0, relation_count=0)

        ingest = ingest_with_usage
        ingest_until = ingest_with_usage

        def finalise_usage(self, *_args, **_values):
            return None

    processor, connection, _ = _open(tmp_path, monkeypatch, ingest=cycle._ingest)
    unit = _native("recovered-gap-six")
    recovery = object()
    head = SimpleNamespace(
        attempt_number=5,
        outcome=GraphitiAdapterOutcome.FAILED,
        recovered_ambiguous_progression=recovery,
    )
    third = SimpleNamespace(
        attempt_number=3,
        outcome=GraphitiAdapterOutcome.AMBIGUOUS_EFFECT,
    )
    processor._system.graphiti = SimpleNamespace(
        attempt_history=lambda *_args, **_values: (head, third),
    )
    processor._runner = Runner()
    processor._usage = SimpleNamespace(
        native_graphiti_ingest_retry_evidence_many=lambda **_values: {
            unit.ingest_id: GraphitiIngestRetryEvidence(
                attempt_numbers=(1, 2, 3, 4, 5),
                zero_dispatch_attempts=(1, 2),
                settled_provider_attempts=settled,
                latest_settled_provider_attempt=settled[-1],
                unresolved_attempts=unresolved,
            )
        }
    )
    processor._settle_missing_subscription_usage = lambda _units: None
    monkeypatch.setattr(n, "graphiti_required_route_holds", lambda *_args, **_values: ())
    monkeypatch.setattr(cycle, "graphiti_required_route_holds", lambda *_args, **_values: ())
    try:
        for number in range(1, 6):
            reserve_graphiti_spend(
                connection,
                spend_id=f"{unit.ingest_id}:{number}",
                ingest_id=unit.ingest_id,
                attempt_number=number,
                proving_run_id=unit.proving_run_id,
                generation_id="test-generation",
                reserved_gbp_microunits=500_000,
                ceiling_gbp_microunits=None,
            )
            record_graphiti_failure(
                connection,
                ingest_id=unit.ingest_id,
                source_id=unit.source_id,
                item_key=unit.item_key,
                outcome="FAILED",
                failure_code="PRODUCER_INTERNAL_ERROR",
            )
            insert_graphiti_attempt_receipt(
                connection,
                ingest_id=unit.ingest_id,
                attempt_number=number,
                outcome="FAILED",
                receipt={
                    "attempt_number": number,
                    "ingest_id": unit.ingest_id,
                    "outcome": "FAILED",
                },
            )
        connection.commit()

        result, = processor.advance((unit,), cycle_id="recovered-gap-six")
        assert result.state == expected_state
        assert calls[0] == ("authenticate", 6)
        if expected_state == "GRAPHITI_COMPLETE":
            assert ("ingest", 6) in calls
        else:
            assert len(calls) == 1
    finally:
        connection.close()


@pytest.mark.parametrize(
    "retained_outcome",
    (
        pytest.param("COMPLETE", id="terminal-complete"),
        pytest.param("MALFORMED_OUTPUT", id="terminal-non-complete"),
        pytest.param("TIMEOUT", id="non-terminal"),
    ),
)
def test_native_reenters_retained_recovered_attempt_before_a_new_successor(
    tmp_path, monkeypatch, retained_outcome,
):
    """A crash after authority attempt 5 leaves private attempt 5 reserved."""
    from newsroom.control_plane.model_usage import GraphitiIngestRetryEvidence
    from newsroom.control_plane.store import (
        insert_graphiti_attempt_receipt,
        record_graphiti_failure,
        reserve_graphiti_spend,
    )
    from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome
    from newsroom.tests.test_graphiti_corpus_ingest import _complete

    calls = []

    class Runner:
        requires_canonical_control_plane_stores = True

        def authenticate_retained_recovered_ambiguous_progression(self, **values):
            calls.append(("authenticate", values["next_attempt_number"]))
            return 4

        def ingest_with_usage(self, unit, **_values):
            calls.append(("ingest", unit.attempt_number))
            return _complete(
                unit, proposal_count=0, entity_count=0, relation_count=0
            )

        ingest = ingest_with_usage
        ingest_until = ingest_with_usage

        def finalise_usage(self, *_args, **_values):
            return None

    processor, connection, _ = _open(tmp_path, monkeypatch, ingest=cycle._ingest)
    unit = _native("recovered-gap-reentry")
    recovery = object()
    head = SimpleNamespace(
        attempt_number=5,
        outcome=GraphitiAdapterOutcome(retained_outcome),
        recovered_ambiguous_progression=recovery,
    )
    third = SimpleNamespace(
        attempt_number=3,
        outcome=GraphitiAdapterOutcome.AMBIGUOUS_EFFECT,
    )
    processor._system.graphiti = SimpleNamespace(
        attempt_history=lambda *_args, **_values: (head, third),
    )
    processor._runner = Runner()
    evidence_requests = []

    def retry_evidence(**values):
        requested = values["failed_attempts"]
        evidence_requests.append(requested)
        selected = requested[unit.ingest_id]
        return {
            unit.ingest_id: GraphitiIngestRetryEvidence(
                attempt_numbers=tuple(range(1, selected + 1)),
                zero_dispatch_attempts=(1, 2),
                settled_provider_attempts=(3,),
                latest_settled_provider_attempt=3,
                unresolved_attempts=tuple(range(4, selected + 1)),
            )
        }

    processor._usage = SimpleNamespace(
        native_graphiti_ingest_retry_evidence_many=retry_evidence,
    )
    processor._settle_missing_subscription_usage = lambda _units: None
    monkeypatch.setattr(
        n, "graphiti_required_route_holds", lambda *_args, **_values: ()
    )
    monkeypatch.setattr(
        cycle, "graphiti_required_route_holds", lambda *_args, **_values: ()
    )
    try:
        for number in range(1, 5):
            reserve_graphiti_spend(
                connection,
                spend_id=f"{unit.ingest_id}:{number}",
                ingest_id=unit.ingest_id,
                attempt_number=number,
                proving_run_id=unit.proving_run_id,
                generation_id=cycle.GRAPHITI_GENERATION_ID,
                reserved_gbp_microunits=500_000,
                ceiling_gbp_microunits=None,
            )
            record_graphiti_failure(
                connection,
                ingest_id=unit.ingest_id,
                source_id=unit.source_id,
                item_key=unit.item_key,
                outcome="FAILED",
                failure_code="PRODUCER_INTERNAL_ERROR",
            )
            insert_graphiti_attempt_receipt(
                connection,
                ingest_id=unit.ingest_id,
                attempt_number=number,
                outcome="FAILED",
                receipt={
                    "attempt_number": number,
                    "ingest_id": unit.ingest_id,
                    "outcome": "FAILED",
                },
            )
        reserve_graphiti_spend(
            connection,
            spend_id=f"{unit.ingest_id}:5",
            ingest_id=unit.ingest_id,
            attempt_number=5,
            proving_run_id=unit.proving_run_id,
            generation_id=cycle.GRAPHITI_GENERATION_ID,
            reserved_gbp_microunits=500_000,
            ceiling_gbp_microunits=None,
        )
        connection.commit()
        assert not cycle._queue(
            connection,
            (unit,),
            model_usage=processor._usage,
            recovered_ambiguous_attempts={},
            authenticated_rejected_attempts={unit.ingest_id: (4,)},
            authenticated_reentry_attempts={unit.ingest_id: 4},
        )
        assert cycle._queue(
            connection,
            (unit,),
            model_usage=processor._usage,
            recovered_ambiguous_attempts={},
            authenticated_rejected_attempts={unit.ingest_id: (4,)},
            authenticated_reentry_attempts={unit.ingest_id: 5},
        )
        assert evidence_requests[-1] == {unit.ingest_id: 5}

        result, = processor.advance((unit,), cycle_id="recovered-gap-reentry")
        assert evidence_requests[-1] == {unit.ingest_id: 5}
        assert (result.state, result.reason, calls) == (
            "GRAPHITI_COMPLETE", None, [("authenticate", 5), ("ingest", 5)]
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM unpublished_graphiti_attempt_receipts "
            "WHERE ingest_id=? AND attempt_number=5",
            (unit.ingest_id,),
        ).fetchone() == (1,)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "retained_outcome",
    ("COMPLETE", "MALFORMED_OUTPUT", "TIMEOUT"),
)
def test_governed_reentry_reconstructs_retained_attempt_five_from_attempt_three(
    monkeypatch, retained_outcome,
):
    """Private receipt loss must not shift the retained authority lineage."""
    from newsroom.authority._extraction_facade import GovernedExtractionRecords
    from newsroom.authority._graphiti_adapter_facade import (
        GovernedGraphitiProposalAdapter,
    )
    from newsroom.authority.auth import AuthenticationProof
    from newsroom.authority.canonical import digest_canonical
    from newsroom.authority.types import UtcTimestamp
    from newsroom.control_plane.graphiti import EvaluationGraphitiRunner
    from newsroom.graphiti_adapter import (
        GraphitiAdapterOutcome,
        RecoveredAmbiguousProgressionProof,
    )
    from newsroom.graphiti_adapter.evaluation_attempt import (
        evaluation_attempt_for_body,
    )

    unit = _native("recovered-gap-governed-reentry")
    authority = unit.authority
    assert authority is not None

    def attempt(number, *, recovery=None, previous_number=None, previous_id=None):
        return evaluation_attempt_for_body(
            episode_body=unit.episode_body,
            ingest_id=unit.ingest_id,
            proving_run_id=unit.proving_run_id,
            source_id=unit.source_id,
            item_key=unit.item_key,
            observation_digest=unit.observation_digest,
            published_at=unit.published_at,
            updated_at=unit.updated_at,
            effective_revision=unit.effective_revision,
            canonical_url=unit.canonical_url,
            revision_digest=unit.revision_digest,
            representation_digest=unit.representation_digest,
            authority_ids=(
                authority.admission_id,
                authority.access_decision_id,
                authority.definition_id,
                authority.definition_version_id,
                authority.item_id,
                authority.revision_id,
                authority.representation_id,
            ),
            attempt_number=number,
            recovered_ambiguous_progression=recovery,
            extraction_previous_version_number=previous_number,
            extraction_previous_run_version_id=previous_id,
        )

    third = attempt(3)
    instant = UtcTimestamp.parse("2026-09-13T12:00:00.000000Z")
    recovery = RecoveredAmbiguousProgressionProof(
        authoritative_attempt_id=third.attempt_id,
        authoritative_attempt_digest=digest_canonical({"attempt": 3}),
        authoritative_attempt_number=3,
        authoritative_run_version_id=third.extraction_request.run_version_id,
        authoritative_recorded_at=instant,
        skipped_attempt_number=4,
        skipped_receipt_digest=digest_canonical({"receipt": 4}),
        skipped_ledger_sequence=4,
        skipped_ledger_digest=digest_canonical({"ledger": 4}),
        skipped_recorded_at=instant,
        settled_usage_evidence_digest=digest_canonical({"usage": 3}),
        recovery_marker_digest=digest_canonical({"marker": 3}),
        marker_attempt_number=3,
        marker_workspace_id=third.workspace_id,
        marker_input_digest=digest_canonical({"marker-input": 3}),
        input_binding_digest=third.extraction_request.input_binding.digest,
        ingest_id=unit.ingest_id,
    )
    fifth = attempt(
        5,
        recovery=recovery,
        previous_number=3,
        previous_id=third.extraction_request.run_version_id,
    )
    from newsroom.tests.test_graphiti_governed_runner import (
        _governed_dependencies,
    )

    retained_adapter, _, retained_proof, _, _ = _governed_dependencies(
        late_timeout=retained_outcome == "TIMEOUT",
    )
    current = retained_adapter.execute_attempt(fifth, proof=retained_proof)
    current = replace(
        current,
        outcome=GraphitiAdapterOutcome(retained_outcome),
        failure_code=(
            "OUTPUT_SCHEMA_INVALID"
            if retained_outcome == "MALFORMED_OUTPUT"
            else current.failure_code
        ),
        proposal_set_id=(
            None
            if retained_outcome == "MALFORMED_OUTPUT"
            else current.proposal_set_id
        ),
        recovered_ambiguous_progression=recovery,
    )
    authenticator = EvaluationGraphitiRunner()

    def prepare(**values):
        assert values["retained_reentry_attempt_number"] == 5
        authenticator._recovered_ambiguous_progressions[(unit.ingest_id, 5)] = recovery
        authenticator._authenticated_recovered_gaps[(unit.ingest_id, 5)] = recovery
        return True

    monkeypatch.setattr(
        authenticator, "prepare_recovered_ambiguous_successor", prepare,
    )
    assert authenticator.authenticate_retained_recovered_ambiguous_progression(
        unit=unit,
        current_attempt=current,
        authoritative_attempt=object(),
        next_attempt_number=5,
        connection=object(),
        model_usage=object(),
    ) == 4
    calls = []

    class Extraction:
        def metadata(self, run_version_id, *_args, **_values):
            calls.append(("metadata", run_version_id))
            return SimpleNamespace(
                version_number=3,
                run_version_id=third.extraction_request.run_version_id,
            )

        def run_history(self, *_args, **_values):
            raise AssertionError("reentry must not use the latest extraction")

        def register_contract(self, *_args, **_values):
            calls.append(("register-contract",))

    class Adapter:
        def register_configuration(self, *_args, **_values):
            calls.append(("register-configuration",))

        def execute_attempt(self, request, *_args, **_values):
            calls.append(("execute", request))
            return object()

    expected = object()
    adapter = Adapter()
    extraction = Extraction()
    runner = EvaluationGraphitiRunner(
        proposal_adapter=GovernedGraphitiProposalAdapter(
            register_configuration=adapter.register_configuration,
            execute_attempt=adapter.execute_attempt,
            approve_replay=lambda *_args, **_values: None,
            configuration=lambda *_args, **_values: None,
            attempt=lambda *_args, **_values: None,
            attempt_history=lambda *_args, **_values: (),
            manifest_for_attempt=lambda *_args, **_values: None,
            replay_source=lambda *_args, **_values: None,
        ),
        extraction_records=GovernedExtractionRecords(
            register_contract=extraction.register_contract,
            execute=lambda *_args, **_values: None,
            contract=lambda *_args, **_values: None,
            metadata=extraction.metadata,
            run_history=extraction.run_history,
            proposals=lambda *_args, **_values: (),
            raw_output=lambda *_args, **_values: None,
        ),
        proof=AuthenticationProof(method="STATIC_TOKEN", credential="fixture"),
        fallback_permitted=False,
    )
    runner._recovered_ambiguous_progressions[(unit.ingest_id, 5)] = recovery
    monkeypatch.setattr(
        runner,
        "_result_from_governed_authority",
        lambda **_values: expected,
    )

    result = runner._ingest(
        replace(unit, attempt_number=5),
        deadline=datetime(2026, 9, 13, 12, 1, tzinfo=UTC),
        invocation_observer=object(),
    )

    assert result is expected
    request = next(entry[1] for entry in calls if entry[0] == "execute")
    assert request.attempt_number == 5
    assert request.expected_previous_attempt_id == third.attempt_id
    assert request.extraction_request.version_number == 4
    assert request.extraction_request.expected_previous_version_id == (
        third.extraction_request.run_version_id
    )
    assert calls[0] == (
        "metadata",
        third.extraction_request.run_version_id,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        '{"attempt":4,"provider_dispatched":0}',
        '{ "attempt": 4, "provider_dispatched": false }',
        "orphaned-predecessor",
    ),
)
def test_recovered_gap_requires_byte_exact_private_ledger_payload(mutation):
    from newsroom.authority.canonical import (
        canonical_json_bytes,
        digest_bytes,
        digest_canonical,
    )
    from newsroom.control_plane.graphiti import EvaluationGraphitiRunner

    receipt = {"attempt": 4, "provider_dispatched": False}
    payload_digest = digest_bytes(canonical_json_bytes(receipt))
    at = "2026-09-13T12:00:00.000000Z"
    previous = "sha256:" + "0" * 64
    event_digest = digest_canonical(
        {
            "at": at,
            "kind": "GRAPHITI_EVALUATION_ATTEMPT",
            "payload_digest": payload_digest,
            "prev": previous,
        }
    )
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(
            "CREATE TABLE ledger(seq INTEGER,at TEXT,kind TEXT,prev_digest TEXT,"
            "digest TEXT,payload_digest TEXT,payload_json TEXT)"
        )
        connection.execute(
            "INSERT INTO ledger VALUES(1,?,?,?,?,?,?)",
            (
                at,
                "GRAPHITI_EVALUATION_ATTEMPT",
                previous,
                event_digest,
                payload_digest,
                canonical_json_bytes(receipt).decode("utf-8"),
            ),
        )
        assert EvaluationGraphitiRunner._exact_private_receipt_ledger_row(
            connection, receipt=receipt,
        ) is not None
        if mutation == "orphaned-predecessor":
            forged_previous = digest_canonical({"unretained": "predecessor"})
            forged_event = digest_canonical(
                {
                    "at": at,
                    "kind": "GRAPHITI_EVALUATION_ATTEMPT",
                    "payload_digest": payload_digest,
                    "prev": forged_previous,
                }
            )
            connection.execute(
                "UPDATE ledger SET prev_digest=?,digest=?",
                (forged_previous, forged_event),
            )
        else:
            connection.execute("UPDATE ledger SET payload_json=?", (mutation,))
        assert EvaluationGraphitiRunner._exact_private_receipt_ledger_row(
            connection, receipt=receipt,
        ) is None
    finally:
        connection.close()


def test_native_quantum_reports_exact_deferred_ids_and_never_projects_chunk_prefix(tmp_path, monkeypatch):
    from newsroom.tests.test_graphiti_corpus_ingest import _complete as completed_result
    from newsroom.control_plane.corpus import MAX_EPISODE_BYTES

    processor, connection, calls = _open(tmp_path, monkeypatch, ingest=cycle._ingest)
    first = replace(_native("chunks"), body="x" * MAX_EPISODE_BYTES, chunk_count=2)
    second = replace(first, chunk_ordinal=2, predecessor_ingest_id=first.ingest_id)
    independent = _native("independent")
    units = (first, second, independent)
    dispatched = []
    processor._runner = SimpleNamespace(ingest=lambda unit: (
        dispatched.append(unit.ingest_id) or completed_result(unit, proposal_count=0, entity_count=0)
    ))
    try:
        before = connection.total_changes
        deferred = processor.advance(units, cycle_id="expired", defer_before_unit=lambda _: True)
        assert [(item.ingest_id, item.state, item.reason) for item in deferred] == [
            (first.ingest_id, "GRAPHITI_DEFERRED", "WORK_QUANTUM_EXHAUSTED"),
            (second.ingest_id, "GRAPHITI_HOLD", "RIGHTS_OR_PREDECESSOR_HOLD"),
            (independent.ingest_id, "GRAPHITI_DEFERRED", "WORK_QUANTUM_EXHAUSTED"),
        ]
        assert not dispatched and not calls
        assert connection.total_changes == before
        prefix = processor.advance(units, cycle_id="prefix", defer_before_unit=lambda unit: unit.ingest_id == second.ingest_id)
        assert [item.state for item in prefix] == ["EXTRACTION_COMPLETE", "GRAPHITI_DEFERRED", "GRAPHITI_COMPLETE"]
        assert len([call for call in calls if call[0] == "empty-cohort-build"]) == 0
        complete = processor.advance(units, cycle_id="remainder")
        assert all(item.state == "GRAPHITI_COMPLETE" for item in complete)
        assert set(dispatched) == {unit.ingest_id for unit in units}
        assert len(dispatched) == 3
        assert len([call for call in calls if call[0] == "empty-cohort-build"]) == 0
        processor.advance(units, cycle_id="unchanged")
        assert len(dispatched) == 3
        assert len([call for call in calls if call[0] == "empty-cohort-build"]) == 0
    finally:
        connection.close()


@pytest.mark.parametrize("held_retries", (0, 1))
def test_native_processor_preserves_fresh_priority_through_ingest_queue(
    tmp_path, monkeypatch, held_retries,
):
    from newsroom.control_plane.store import record_graphiti_failure
    from newsroom.tests.test_graphiti_corpus_ingest import _complete as completed_result

    processor, connection, _ = _open(tmp_path, monkeypatch, ingest=cycle._ingest)
    fresh = replace(_native("fresh"), observed_at="2026-09-12T12:00:00.000000Z")
    held = replace(_native("held"), observed_at="2026-09-11T12:00:00.000000Z")
    for _ in range(held_retries):
        record_graphiti_failure(
            connection, ingest_id=held.ingest_id, source_id=held.source_id,
            item_key=held.item_key, outcome="FAILED", failure_code="TEST_FAILURE",
        )
    connection.commit()
    dispatched = []
    processor._runner = SimpleNamespace(ingest=lambda unit: (
        dispatched.append(unit.ingest_id)
        or completed_result(unit, proposal_count=0, entity_count=0)
    ))
    try:
        outcomes = processor.advance(
            (fresh, held), cycle_id=f"fresh-before-held-{held_retries}",
            defer_before_unit=lambda _: bool(dispatched),
        )
        assert dispatched == [fresh.ingest_id]
        assert [(item.ingest_id, item.state) for item in outcomes] == [
            (fresh.ingest_id, "GRAPHITI_COMPLETE"),
            (held.ingest_id, "GRAPHITI_DEFERRED"),
        ]
    finally:
        connection.close()


@pytest.mark.parametrize("stop", ["veto", "drain"])
def test_native_stop_outranks_quantum_deferral(tmp_path, monkeypatch, stop):
    processor, connection, calls = _open(tmp_path, monkeypatch, ingest=cycle._ingest)
    checks = []

    def check():
        checks.append(True)
        if stop == "veto" and len(checks) == 2:
            raise VetoError("owner stop before deferral")

    processor._stop_check = check
    processor._operator_drain_requested = lambda: stop == "drain"
    try:
        before = connection.total_changes
        with pytest.raises(VetoError if stop == "veto" else OperatorDrainRequested):
            processor.advance((_native(),), cycle_id="stopped", defer_before_unit=lambda _: pytest.fail("deferral before stop"))
        assert connection.total_changes == before and not calls
    finally:
        connection.close()


@pytest.mark.parametrize("chunks", (2, 4, 8))
def test_terminal_hold_identity_work_is_bounded_per_independent_advance(
    tmp_path, monkeypatch, chunks,
):
    from newsroom.control_plane import corpus
    from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome

    queued, history = [], []
    processor, connection, _ = _open(
        tmp_path, monkeypatch,
        ingest=lambda _connection, **kw: queued.extend(kw["units"]),
    )
    base = replace(
        _native("terminal-chunks"), body="x" * (chunks * 8192 - 200),
        chunk_count=chunks,
    )
    members = []
    for ordinal in range(1, chunks + 1):
        members.append(replace(
            base, chunk_ordinal=ordinal,
            predecessor_ingest_id=None if not members else members[-1].ingest_id,
        ))
    # Admission validates chunk order independently of the supplied output order.
    units = tuple(reversed(members))
    expected_ids = tuple(unit.ingest_id for unit in units)
    processor._system.graphiti = SimpleNamespace(attempt_history=lambda *args, **kw: (
        history.append(args[0]) or SimpleNamespace(
            outcome=GraphitiAdapterOutcome.AMBIGUOUS_EFFECT,
            failure_code="AMBIGUOUS_EFFECT",
        ),
    ))
    digest = corpus.content_digest
    identity = corpus.CorpusIngestUnit.ingest_id.fget
    identity_reads, body_characters = [], []

    def counted_identity(unit):
        identity_reads.append(unit.chunk_ordinal)
        return identity(unit)

    def counted_digest(**kwargs):
        body_characters.append(len(kwargs["body"]))
        return digest(**kwargs)

    monkeypatch.setattr(corpus.CorpusIngestUnit, "ingest_id", property(counted_identity))
    monkeypatch.setattr(corpus, "content_digest", counted_digest)
    try:
        previous = None
        for number in (1, 2):
            identity_reads.clear()
            body_characters.clear()
            results = processor.advance(units, cycle_id=f"unchanged-terminal:{number}")
            assert tuple(item.ingest_id for item in results) == expected_ids
            assert all(
                item.state == "GRAPHITI_HOLD" and item.receipt_digest is None
                and item.reason == "AMBIGUOUS_EFFECT:AMBIGUOUS_EFFECT"
                for item in results
            )
            assert previous is None or results == previous
            previous = results
            assert len(history) == number * chunks
            assert not queued
            assert connection.execute("SELECT count(*) FROM ledger").fetchone()[0] == 0
            assert (len(identity_reads), len(body_characters), sum(body_characters)) == (
                chunks, chunks, chunks * len(base.body),
            )
    finally:
        connection.close()


@pytest.mark.parametrize("invalid", ("duplicate", "wrong_type", "missing_authority"))
def test_operation_local_identity_selection_preserves_input_rejection(
    tmp_path, monkeypatch, invalid,
):
    processor, connection, calls = _open(
        tmp_path, monkeypatch, ingest=lambda *args, **kw: pytest.fail("dispatch"),
    )
    unit = _native()
    if invalid == "duplicate":
        units = (unit, replace(unit, attempt_number=2))
        message = "native Graphiti cohort repeats an ingest"
    elif invalid == "wrong_type":
        units = (SimpleNamespace(ingest_id=unit.ingest_id),)
        message = "native Graphiti requires retained source authority"
    else:
        units = (replace(unit, authority=None),)
        message = "native Graphiti requires retained source authority"
    try:
        before = connection.total_changes
        with pytest.raises(ValueError, match=message):
            processor.advance(units, cycle_id="bad-operation-input")
        assert connection.total_changes == before
        assert not calls
    finally:
        connection.close()


def test_operation_local_keys_rederive_replacement_units_and_recheck_rights(
    tmp_path, monkeypatch,
):
    from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome, GraphitiAdapterRightsDenied

    processor, connection, _ = _open(
        tmp_path, monkeypatch,
        ingest=lambda _connection, **kw: pytest.fail("dispatch") if kw["units"] else None,
    )
    first = _native("changed-body")
    second = replace(first, body=first.body + " Changed source text.")
    expected_ids = (first.ingest_id, second.ingest_id)
    assert expected_ids[0] != expected_ids[1]
    denied, reads = [], []

    def current_history(*args, **kwargs):
        reads.append(args[0])
        if denied:
            raise GraphitiAdapterRightsDenied("current rights revoked")
        return (SimpleNamespace(
            outcome=GraphitiAdapterOutcome.AMBIGUOUS_EFFECT,
            failure_code="AMBIGUOUS_EFFECT",
        ),)

    processor._system.graphiti = SimpleNamespace(attempt_history=current_history)
    try:
        first_result, = processor.advance((first,), cycle_id="old-body")
        denied.append(True)
        second_result, = processor.advance((second,), cycle_id="new-body")
        assert (first_result.ingest_id, second_result.ingest_id) == expected_ids
        assert first_result.reason == "AMBIGUOUS_EFFECT:AMBIGUOUS_EFFECT"
        assert second_result.reason == "CURRENT_SOURCE_RIGHTS_HOLD"
        assert len(reads) == 2 and reads[0] != reads[1]
        assert first.body != second.body
        assert connection.execute("SELECT count(*) FROM ledger").fetchone()[0] == 0
    finally:
        connection.close()


def _linked_native_units(item, count):
    from newsroom.control_plane.corpus import MAX_EPISODE_BYTES
    base = replace(_native(item), body='x' * MAX_EPISODE_BYTES * (count - 1), chunk_count=count)
    units, predecessor = [], None
    for ordinal in range(1, count + 1):
        unit = replace(base, chunk_ordinal=ordinal, predecessor_ingest_id=predecessor)
        units.append(unit)
        predecessor = unit.ingest_id
    return tuple(units)


def test_native_refreshes_completed_predecessors_within_one_quantum(tmp_path, monkeypatch):
    from newsroom.tests.test_graphiti_corpus_ingest import _complete as result
    from newsroom.control_plane.store import has_graphiti_ingest
    processor, connection, _calls = _open(tmp_path, monkeypatch, ingest=cycle._ingest)
    chain = _linked_native_units('chain', 3)
    independent = _native('independent')
    dispatched = []
    def ingest(unit):
        assert not unit.predecessor_ingest_id or has_graphiti_ingest(connection, unit.predecessor_ingest_id)
        dispatched.append(unit.ingest_id)
        return result(unit, proposal_count=0, entity_count=0)
    processor._runner = SimpleNamespace(ingest=ingest)
    try:
        outcomes = processor.advance((*chain, independent), cycle_id='drain')
        assert all(x.state == 'GRAPHITI_COMPLETE' for x in outcomes)
        # Ready independent work keeps its place before refreshed successors.
        assert dispatched == [chain[0].ingest_id, independent.ingest_id, chain[1].ingest_id, chain[2].ingest_id]
        processor.advance((*chain, independent), cycle_id='replay')
        assert len(dispatched) == 4
    finally:
        connection.close()


def test_native_refreshed_successors_obey_existing_quantum_and_resume(tmp_path, monkeypatch):
    from newsroom.tests.test_graphiti_corpus_ingest import _complete as result
    processor, connection, calls = _open(tmp_path, monkeypatch, ingest=cycle._ingest)
    chain = _linked_native_units('chain', 3)
    dispatched = []
    processor._runner = SimpleNamespace(ingest=lambda unit: (
        dispatched.append(unit.ingest_id) or result(unit, proposal_count=0, entity_count=0)
    ))
    try:
        outcomes = processor.advance(chain, cycle_id='one', defer_before_unit=lambda _: len(dispatched) >= 1)
        assert [x.state for x in outcomes] == ['EXTRACTION_COMPLETE', 'GRAPHITI_DEFERRED', 'GRAPHITI_HOLD']
        assert not any(name == 'finalise' for name, _ in calls)
        assert all(x.state == 'GRAPHITI_COMPLETE' for x in processor.advance(chain, cycle_id='rest'))
        assert dispatched == [u.ingest_id for u in chain]
    finally:
        connection.close()


def test_native_frontier_refresh_does_not_retry_failure_while_peers_progress(tmp_path, monkeypatch):
    from newsroom.tests.test_graphiti_corpus_ingest import _complete as result
    from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
    from newsroom.control_plane.store import graphiti_failure_state
    processor, connection, _ = _open(tmp_path, monkeypatch, ingest=cycle._ingest)
    failed_chain = _linked_native_units('failed', 2)
    good_chain = _linked_native_units('good', 3)
    dispatched = []
    def ingest(unit):
        dispatched.append(unit.ingest_id)
        completed = result(unit, proposal_count=0, entity_count=0)
        if unit.ingest_id != failed_chain[0].ingest_id:
            return completed
        raw = dict(completed.raw_receipt)
        raw.pop('raw_output_digest')
        raw.update(outcome='FAILED', failure_code='PRODUCER_INTERNAL_ERROR')
        raw['raw_output_digest'] = digest_bytes(canonical_json_bytes(raw))
        return replace(completed, outcome='FAILED', failure_code='PRODUCER_INTERNAL_ERROR',
                       raw_receipt=raw, receipt_digest=raw['raw_output_digest'])
    processor._runner = SimpleNamespace(ingest=ingest)
    try:
        outcomes = processor.advance((*failed_chain, *good_chain), cycle_id='mixed')
        assert [x.state for x in outcomes[:2]] == ['GRAPHITI_HOLD', 'GRAPHITI_HOLD']
        assert all(x.state == 'GRAPHITI_COMPLETE' for x in outcomes[2:])
        assert dispatched == [failed_chain[0].ingest_id, *(x.ingest_id for x in good_chain)]
        assert graphiti_failure_state(connection, failed_chain[0].ingest_id) == (1, False)
    finally:
        connection.close()


def test_native_refreshed_successor_rechecks_rights_before_dispatch(tmp_path, monkeypatch):
    from newsroom.tests.test_graphiti_corpus_ingest import _complete as result
    chain = _linked_native_units('rights', 2)
    seen, dispatched = [], []
    def rights(unit):
        seen.append(unit.ingest_id)
        return {'current': True} if unit.chunk_ordinal == 1 else None
    processor, connection, _ = _open(tmp_path, monkeypatch, ingest=cycle._ingest, rights=rights)
    processor._runner = SimpleNamespace(ingest=lambda unit: (
        dispatched.append(unit.ingest_id) or result(unit, proposal_count=0, entity_count=0)
    ))
    try:
        outcomes = processor.advance(chain, cycle_id='rights')
        assert dispatched == [chain[0].ingest_id]
        assert seen == [chain[0].ingest_id, chain[0].ingest_id, chain[1].ingest_id]
        assert [x.state for x in outcomes] == ['EXTRACTION_COMPLETE', 'GRAPHITI_HOLD']
    finally:
        connection.close()


def test_native_route_hold_stops_refresh_without_starting_new_successor(tmp_path, monkeypatch):
    from newsroom.tests.test_graphiti_corpus_ingest import _complete as result
    processor, connection, _ = _open(tmp_path, monkeypatch, ingest=cycle._ingest)
    chain = _linked_native_units('route', 2)
    dispatched = []
    processor._runner = SimpleNamespace(ingest=lambda unit: (
        dispatched.append(unit.ingest_id) or result(unit, proposal_count=0, entity_count=0)
    ))
    monkeypatch.setattr(n, 'graphiti_required_route_holds', lambda *_a, **_k: ('UNKNOWN_USAGE',) if dispatched else ())
    try:
        outcomes = processor.advance(chain, cycle_id='held-route')
        assert dispatched == [chain[0].ingest_id]
        assert outcomes[1].reason == 'REQUIRED_MODEL_ROUTE_CIRCUIT_OPEN'
    finally:
        connection.close()


def test_owned_compensation_precedes_terminal_filter_without_reusing_spend(tmp_path, monkeypatch):
    from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome
    from newsroom.control_plane.store import reserve_graphiti_spend

    events, queued = [], []
    processor, connection, _calls = _open(
        tmp_path, monkeypatch,
        ingest=lambda _connection, **kwargs: queued.extend(kwargs["units"]),
    )
    unit, distinct = _native("owned-pending"), _native("distinct-fresh")
    head = SimpleNamespace(outcome=GraphitiAdapterOutcome.AMBIGUOUS_EFFECT, failure_code="AMBIGUOUS_EFFECT")
    def history(run_id, **_kwargs):
        events.append("terminal-filter")
        return (head,) if run_id == n.typed_id(n.ExtractionRunId, "run", unit.ingest_id) else ()
    processor._system.graphiti.attempt_history = history
    processor._runner = SimpleNamespace(
        recover_owned_pending=lambda **kwargs: events.append("compensate")
    )
    reserve_graphiti_spend(
        connection, spend_id=f"{unit.ingest_id}:1", ingest_id=unit.ingest_id,
        attempt_number=1, proving_run_id=unit.proving_run_id,
        generation_id=cycle.GRAPHITI_GENERATION_ID, reserved_gbp_microunits=500_000,
        ceiling_gbp_microunits=None,
    )
    connection.execute("UPDATE unpublished_graphiti_spend SET status='RECONCILED' WHERE ingest_id=?", (unit.ingest_id,))
    connection.commit()
    before = tuple(connection.execute("SELECT * FROM unpublished_graphiti_spend"))
    try:
        outcomes = processor.advance((unit, distinct), cycle_id="owned-compensation")
        assert events[0] == "compensate"
        assert queued == [distinct]
        assert outcomes[0].reason == "AMBIGUOUS_EFFECT:AMBIGUOUS_EFFECT"
        assert tuple(connection.execute("SELECT * FROM unpublished_graphiti_spend")) == before
        assert connection.execute("SELECT count(*) FROM unpublished_graphiti_attempt_receipts").fetchone()[0] == 0
    finally:
        connection.close()


@pytest.mark.parametrize("boundary", (
    "exact_old_owner", "missing_attempt", "missing_landing", "workspace", "attempt_number",
    "manifest", "extraction_input", "landing_corrupt", "current_rights", "quantum", "owner_stop",
    "authoritative_complete", "accepted_complete",
))
def test_owned_compensation_authenticates_immutable_original_input_and_existing_fence(
    tmp_path, monkeypatch, boundary,
):
    import newsroom.control_plane.graphiti as graphiti
    import newsroom.graphiti_adapter.real as real
    from newsroom.control_plane.native_progress import NativeRevisionJournal
    from newsroom.graphiti_adapter.evaluation_attempt import evaluation_attempt_for_body
    from newsroom.graphiti_adapter.neo4j_guard import GuardMarker, GuardState
    from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome, GraphitiAdapterStateError, GraphitiWorkspaceId
    from newsroom.authority.canonical import digest_canonical

    connection = connect(str(tmp_path / "compensation-auth.sqlite3"))
    unit = _native("retained-owner-no-longer-latest")
    authority = unit.authority
    attempt = evaluation_attempt_for_body(
        episode_body=unit.episode_body, ingest_id=unit.ingest_id, proving_run_id=unit.proving_run_id,
        source_id=unit.source_id, item_key=unit.item_key, observation_digest=unit.observation_digest,
        published_at=unit.published_at, updated_at=unit.updated_at, effective_revision=unit.effective_revision,
        canonical_url=unit.canonical_url, revision_digest=unit.revision_digest,
        representation_digest=unit.representation_digest,
        authority_ids=(authority.admission_id, authority.access_decision_id, authority.definition_id,
            authority.definition_version_id, authority.item_id, authority.revision_id, authority.representation_id),
    )
    retained = SimpleNamespace(
        attempt_id=attempt.attempt_id, attempt_number=1, workspace_id=attempt.workspace_id,
        run_id=attempt.extraction_request.run_id, run_version_id=attempt.extraction_request.run_version_id,
        manifest_id=attempt.manifest.manifest_id, configuration_id=attempt.configuration.configuration_id,
        configuration_digest=attempt.configuration.canonical_digest,
        outcome=GraphitiAdapterOutcome.TIMEOUT,
    )
    manifest = attempt.manifest
    metadata = SimpleNamespace(
        run_id=retained.run_id, run_version_id=retained.run_version_id, version_number=1,
        input_binding_digest=manifest.input_binding_digest,
    )
    if boundary != "missing_landing":
        NativeRevisionJournal(connection).land((unit,))
    if boundary == "landing_corrupt":
        connection.execute("UPDATE ledger SET payload_digest=? WHERE kind='NATIVE_REVISION_LANDED'", ("sha256:" + "0" * 64,))
        connection.commit()
    elif boundary == "workspace":
        retained.workspace_id = GraphitiWorkspaceId.parse("00000000-0000-4000-8000-000000004990")
    elif boundary == "attempt_number":
        retained.attempt_number = 2
    elif boundary == "manifest":
        manifest = replace(manifest, input_binding_digest=digest_canonical({"different": "input"}))
    elif boundary == "extraction_input":
        metadata.input_binding_digest = digest_canonical({"different": "input"})
    elif boundary == "authoritative_complete":
        retained.outcome = GraphitiAdapterOutcome.COMPLETE
    elif boundary == "accepted_complete":
        _complete(connection, units=(unit,))
    events, active = [], False

    def retained_attempt(attempt_id, **_kwargs):
        events.append("retained-attempt")
        assert attempt_id == attempt.attempt_id
        if boundary == "missing_attempt":
            raise GraphitiAdapterStateError("attempt is absent")
        return retained

    @contextmanager
    def fence(current):
        nonlocal active
        assert current == unit
        events.append("fence")
        active = True
        try:
            yield None if boundary == "current_rights" else SimpleNamespace(owner_stop_check=lambda: events.append("active-fence"))
        finally:
            active = False
            events.append("unfence")

    def compensate(**kwargs):
        # Neo4j identity names the old attempt, not the newer terminal head.
        with kwargs["authenticated_input"](unit.ingest_id, 1) as input_digest:
            if input_digest is None:
                return None
            assert active
            assert input_digest == digest_canonical({
                "episode_uuid": unit.ingest_id, "name": unit.ingest_id,
                "body": " ".join(unit.episode_body.split()),
                "reference_time": unit.temporal().reference_time.value.isoformat(),
                "group_id": cycle.GRAPHITI_WORKSPACE_GROUP,
            })
            events.append("compensate")
            return GuardMarker(GuardState.RECOVERED_AMBIGUOUS, 1, input_digest)

    runner = graphiti.EvaluationGraphitiRunner()
    runner._proof = object()
    runner._proposal_adapter = SimpleNamespace(
        attempt=retained_attempt,
        manifest_for_attempt=lambda *a, **kw: manifest,
        attempt_history=lambda *a, **kw: pytest.fail("latest-head lookup loses the original owner"),
        execute_attempt=lambda *a, **kw: pytest.fail("provider attempt"),
    )
    runner._extraction_records = SimpleNamespace(metadata=lambda *a, **kw: metadata)
    monkeypatch.setattr(real, "recover_owned_pending", compensate)
    before = connection.total_changes

    def stop():
        if boundary == "owner_stop":
            raise VetoError("owner emergency stop")

    try:
        arguments = dict(
            connection=connection, owner_stop_check=stop, rights_fence=fence,
            defer_before_unit=lambda _: boundary == "quantum",
            deadline=datetime(2026, 9, 26, tzinfo=UTC),
        )
        if boundary == "owner_stop":
            with pytest.raises(VetoError):
                runner.recover_owned_pending(**arguments)
        elif boundary == "landing_corrupt":
            with pytest.raises(graphiti.ModelUsageIntegrityError):
                runner.recover_owned_pending(**arguments)
        else:
            marker = runner.recover_owned_pending(**arguments)
            assert (marker is not None) == (boundary == "exact_old_owner")
        assert events.count("compensate") == int(boundary == "exact_old_owner")
        assert events.count("fence") == events.count("unfence")
        assert not active
        assert connection.total_changes == before
    finally:
        connection.close()
