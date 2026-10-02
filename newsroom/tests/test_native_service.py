import json
import logging
import sqlite3
import threading
from contextlib import contextmanager

import pytest

from newsroom.control_plane.native_pipeline import NativePipeline, NativePipelineReport
from newsroom.control_plane.native_service import (
    NativeService, NativeServiceAlreadyRunning, NativeServiceReport,
    _instance_lock,
)
from newsroom.control_plane.veto import OperatorDrainRequested, VetoError
from scripts.hermes_native import main
from newsroom.authority.canonical import digest_canonical


def _diagnostics(tmp_path):
    path = tmp_path / "diagnostics/native.log"
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return [(record["event"], record["data"]) for record in records
            if record.get("event", "").startswith("NATIVE_SERVICE_CYCLE_")]


@pytest.mark.parametrize("pending", [
    "QUEUED", "GRAPHITI_COMPLETE", "ASSESSMENT_INTERRUPTED",
    "ASSESSMENT_STARTED", "PUBLICATION_STARTED",
])
def test_pending_service_reports_continue_same_open_and_qualify_only_when_terminal(
    tmp_path, monkeypatch, pending,
):
    reports = iter((
        NativePipelineReport((), {pending: 1}, int(pending == "QUEUED")),
        NativePipelineReport((), {pending: 1}, int(pending == "QUEUED")),
        NativePipelineReport((), {"ACKNOWLEDGED": 1}, 0),
        NativePipelineReport((), {"ACKNOWLEDGED": 1}, 0),
    ))
    order = []
    factory, opened = _pipeline(
        tmp_path, monkeypatch, lambda cycle: order.append(cycle) or next(reports),
    )
    identity = digest_canonical({"runtime": "cooperative-continuation"})

    @contextmanager
    def bound():
        with factory() as pipeline:
            pipeline.runtime_identity_digest = identity
            yield pipeline

    waits = []
    result = _service(
        tmp_path, bound,
        cycle_id_factory=iter(("pending", "continuing", "ack", "unchanged")).__next__,
        qualify_once=lambda *_: order.append("qualified"),
        wait=lambda _: waits.append(True) or len(waits) == 4,
    ).run()
    assert result.pipeline.revision_states == {"ACKNOWLEDGED": 1}
    assert order == ["pending", "continuing", "ack", "qualified", "unchanged"]
    assert opened == ["open", "close"]


def _pipeline(tmp_path, monkeypatch, tick, *, ledger_name="unpublished.sqlite3"):
    pipeline = object.__new__(NativePipeline)
    def retained_tick(cycle_id):
        report = tick(cycle_id)
        if type(report) is NativePipelineReport:
            # The real pipeline retains its portfolio before returning a report.
            # Keep that producer contract even where a test isolates the loop.
            if pipeline._journal._portfolio_record is None or pipeline._journal.portfolio != report.sources:
                pipeline._journal._retain("NATIVE_SOURCE_PORTFOLIO", {"sources": list(report.sources)})
        return report

    pipeline._test_tick = retained_tick
    monkeypatch.setattr(
        NativePipeline, "tick",
        lambda self, *, cycle_id: self._test_tick(cycle_id),
    )
    opened = []

    @contextmanager
    def factory():
        from newsroom.control_plane.native_progress import NativeRevisionJournal
        from newsroom.control_plane.store import connect

        opened.append("open")
        connection = connect(str(tmp_path / ledger_name))
        pipeline._journal = NativeRevisionJournal(connection)
        try:
            yield pipeline
        finally:
            connection.close()
            opened.append("close")

    return factory, opened


def _service(tmp_path, factory, **values):
    return NativeService(
        pipeline_factory=factory,
        ledger_path=str(tmp_path / "unpublished.sqlite3"),
        lock_path=tmp_path / "hermes-native.lock",
        stop_check=values.pop("stop_check", lambda: None),
        interval_seconds=values.pop("interval_seconds", 2),
        failure_backoff_seconds=values.pop("failure_backoff_seconds", 7),
        cycle_id_factory=values.pop("cycle_id_factory", iter(("cycle-1", "cycle-2")).__next__),
        **values,
    )


def test_native_service_runs_two_ticks_without_story_cap_and_closes(tmp_path, monkeypatch):
    ticks, waits = [], []
    factory, opened = _pipeline(
        tmp_path, monkeypatch,
        lambda cycle_id: (
            ticks.append(cycle_id)
            or NativePipelineReport((), {"ACKNOWLEDGED": 4}, 0)
        ),
    )

    def wait(seconds):
        waits.append(seconds)
        return len(waits) == 2

    report = _service(tmp_path, factory, wait=wait).run()
    assert ticks == ["cycle-1", "cycle-2"]
    assert len(waits) == 2 and all(0 <= seconds <= 2 for seconds in waits)
    assert opened == ["open", "close"]
    assert report.outcome == "COMPLETE"
    assert report.pipeline.revision_states == {"ACKNOWLEDGED": 4}
    with sqlite3.connect(tmp_path / "unpublished.sqlite3") as connection:
        assert connection.execute(
            "SELECT kind FROM ledger WHERE kind LIKE 'NATIVE_SERVICE_%' ORDER BY seq"
        ).fetchall() == []
    assert [kind for kind, _ in _diagnostics(tmp_path)] == [
        "NATIVE_SERVICE_CYCLE_STARTED", "NATIVE_SERVICE_CYCLE_TERMINAL",
        "NATIVE_SERVICE_CYCLE_STARTED", "NATIVE_SERVICE_CYCLE_TERMINAL",
    ]


def test_native_service_interval_is_measured_from_cycle_start(tmp_path, monkeypatch):
    now = [10.0]
    waits = []

    def tick(cycle_id):
        now[0] += 1.25 if cycle_id == "cycle-1" else 3
        return NativePipelineReport((), {}, 0)

    factory, _ = _pipeline(tmp_path, monkeypatch, tick)

    def wait(seconds):
        waits.append(seconds)
        return len(waits) == 2

    _service(
        tmp_path, factory, interval_seconds=2, wait=wait,
        monotonic_clock=lambda: now[0],
    ).run()
    assert waits == [0.75, 0]


def test_native_service_failure_keeps_its_full_backoff(tmp_path, monkeypatch):
    now = [10.0]
    outcomes = iter((RuntimeError("failed"), NativePipelineReport((), {}, 0)))
    waits = []

    def tick(_cycle_id):
        now[0] += 3
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    factory, _ = _pipeline(tmp_path, monkeypatch, tick)

    def wait(seconds):
        waits.append(seconds)
        return len(waits) == 2

    _service(
        tmp_path, factory, interval_seconds=2, failure_backoff_seconds=7,
        wait=wait, monotonic_clock=lambda: now[0],
    ).run()
    assert waits == [7, 0]


def test_continuous_service_qualifies_first_complete_cycle_before_second_tick(
    tmp_path, monkeypatch,
):
    order = []
    factory, opened = _pipeline(
        tmp_path, monkeypatch,
        lambda cycle_id: (
            order.append(f"tick:{cycle_id}")
            or NativePipelineReport((), {"ACKNOWLEDGED": 1}, 0)
        ),
    )
    identity = digest_canonical({"runtime": "single-open"})

    @contextmanager
    def bound():
        with factory() as pipeline:
            pipeline.runtime_identity_digest = identity
            yield pipeline

    def qualify(connection, actual_identity):
        assert actual_identity == identity
        assert connection.execute("SELECT count(*) FROM ledger WHERE kind LIKE 'NATIVE_SERVICE_CYCLE_%'").fetchone()[0] == 0
        order.append("qualified")

    waits = []

    def wait(_seconds):
        waits.append(True)
        return len(waits) == 2

    report = _service(
        tmp_path, bound, wait=wait, qualify_once=qualify,
    ).run()

    assert report is not None and report.outcome == "COMPLETE"
    assert order == ["tick:cycle-1", "qualified", "tick:cycle-2"]
    assert opened == ["open", "close"]


def test_continuous_qualification_failure_closes_without_second_cycle(
    tmp_path, monkeypatch,
):
    ticks = []
    factory, opened = _pipeline(
        tmp_path, monkeypatch,
        lambda cycle_id: (
            ticks.append(cycle_id) or NativePipelineReport((), {}, 0)
        ),
    )
    identity = digest_canonical({"runtime": "qualification-rejected"})

    @contextmanager
    def bound():
        with factory() as pipeline:
            pipeline.runtime_identity_digest = identity
            yield pipeline

    with pytest.raises(ValueError, match="qualification rejected"):
        _service(
            tmp_path,
            bound,
            qualify_once=lambda *_: (_ for _ in ()).throw(
                ValueError("qualification rejected")
            ),
            wait=lambda _: pytest.fail("qualification failure entered wait"),
        ).run()

    assert ticks == ["cycle-1"]
    assert opened == ["open", "close"]
    assert _diagnostics(tmp_path)[-1][1]["outcome"] == "COMPLETE"


@pytest.mark.parametrize("settle_after_first_tick,once,corrupt_total", [
    (False, False, False), (True, False, False), (False, True, False),
    (False, False, True), (False, True, True),
])
def test_unreported_usage_defers_qualification_without_reopening_pipeline(
    tmp_path, monkeypatch, settle_after_first_tick, once, corrupt_total,
):
    from newsroom.control_plane.native_qualification import (
        NativeQualificationError, NativeQualificationPending, record_qualification,
    )
    from newsroom.tests.test_native_qualification import (
        IDENTITY, _allocation, _conservative_disposition, _cycle, _open,
    )

    connection = _open(tmp_path / "unpublished.sqlite3")
    journal = _cycle(connection)
    invocation = _allocation(connection, usage_status="UNREPORTED")
    if corrupt_total:
        from newsroom.authority.canonical import canonical_json_bytes
        record = json.loads(connection.execute(
            "SELECT record_json FROM model_invocation_terminals WHERE invocation_id=?",
            (invocation,),
        ).fetchone()[0])
        record["components"]["total_tokens"] = 1
        record["terminal_digest"] = ""
        record["terminal_digest"] = digest_canonical(record)
        connection.execute(
            "UPDATE model_invocation_terminals SET terminal_digest=?,record_json=? "
            "WHERE invocation_id=?",
            (record["terminal_digest"], canonical_json_bytes(record).decode(), invocation),
        )
        connection.commit()
    original_terminal = connection.execute(
        "SELECT record_json FROM model_invocation_terminals WHERE invocation_id=?",
        (invocation,),
    ).fetchone()
    report = NativePipelineReport(journal.portfolio, {"EVIDENCE_HOLD": 1}, 0)
    factory, opened = _pipeline(tmp_path, monkeypatch, lambda _: report)

    @contextmanager
    def bound():
        with factory() as pipeline:
            pipeline.runtime_identity_digest = IDENTITY
            yield pipeline

    waits = []

    def wait(_seconds):
        waits.append(True)
        if len(waits) == 1:
            assert connection.execute(
                "SELECT count(*) FROM ledger WHERE kind='NATIVE_SERVICE_QUALIFICATION'"
            ).fetchone() == (0,)
            if settle_after_first_tick:
                _conservative_disposition(connection, invocation)
        return len(waits) == 2

    try:
        service = _service(
            tmp_path, bound, qualify_once=record_qualification, wait=wait,
        )
        if corrupt_total:
            with pytest.raises(NativeQualificationError, match="semantics") as failure:
                service.run(once=once)
            assert not isinstance(failure.value, NativeQualificationPending)
            assert waits == []
        elif once:
            with pytest.raises(NativeQualificationPending, match="unresolved"):
                service.run(once=True)
            assert waits == []
        else:
            result = service.run()
            assert result.outcome == "COMPLETE" and len(waits) == 2
        assert opened == ["open", "close"]
        assert connection.execute(
            "SELECT count(*) FROM ledger WHERE kind='NATIVE_SERVICE_QUALIFICATION'"
        ).fetchone() == (int(settle_after_first_tick),)
        assert connection.execute(
            "SELECT record_json FROM model_invocation_terminals WHERE invocation_id=?",
            (invocation,),
        ).fetchone() == original_terminal
    finally:
        connection.close()


def test_native_service_failure_is_terminal_then_restart_continues(tmp_path, monkeypatch):
    factory, opened = _pipeline(
        tmp_path, monkeypatch, lambda _cycle_id: (_ for _ in ()).throw(RuntimeError("secret")),
    )
    failed = _service(
        tmp_path, factory, cycle_id_factory=lambda: "failed-cycle",
    ).run(once=True)
    assert failed.failure_class == "RuntimeError"
    assert opened == ["open", "close"]

    factory, _ = _pipeline(
        tmp_path, monkeypatch,
        lambda _cycle_id: NativePipelineReport((), {"QUEUED": 2}, 2),
    )
    complete = _service(
        tmp_path, factory, cycle_id_factory=lambda: "restart-cycle",
    ).run(once=True)
    assert complete.outcome == "COMPLETE"
    terminals = [payload for kind, payload in _diagnostics(tmp_path)
                 if kind == "NATIVE_SERVICE_CYCLE_TERMINAL"]
    assert [(item["cycle_id"], item["outcome"]) for item in terminals] == [
        ("failed-cycle", "FAILED"), ("restart-cycle", "COMPLETE"),
    ]
    assert "secret" not in json.dumps(terminals)


def test_native_service_preserves_veto_and_singleton_lock(tmp_path, monkeypatch):
    factory, opened = _pipeline(
        tmp_path, monkeypatch, lambda _cycle_id: (_ for _ in ()).throw(VetoError("signed stop")),
    )
    service = _service(tmp_path, factory, cycle_id_factory=lambda: "stopped-cycle")
    with pytest.raises(VetoError, match="signed stop"):
        service.run(once=True)
    assert opened == ["open", "close"]
    records = _diagnostics(tmp_path)
    assert [kind for kind, _ in records] == [
        "NATIVE_SERVICE_CYCLE_STARTED", "NATIVE_SERVICE_CYCLE_TERMINAL",
    ]
    assert all(payload["cycle_id"] == "stopped-cycle" for _, payload in records)
    assert records[-1][1] == {
        "cycle_id": "stopped-cycle", "outcome": "STOPPED",
        "failure_class": "VetoError", "pipeline": None,
    }
    assert "signed stop" not in json.dumps(records)

    with _instance_lock(tmp_path / "hermes-native.lock"):
        with pytest.raises(NativeServiceAlreadyRunning):
            service.run(once=True)


def test_native_service_operator_drain_is_terminal_without_qualification(
    tmp_path, monkeypatch,
):
    service_event = threading.Event()
    qualified = []

    def tick(_cycle_id):
        service_event.set()
        return NativePipelineReport((), {"QUEUED": 1}, 1)

    factory, opened = _pipeline(tmp_path, monkeypatch, tick)
    report = _service(
        tmp_path, factory, service_event=service_event,
        cycle_id_factory=lambda: "drained-cycle",
        qualify_once=lambda *_: qualified.append(True),
    ).run(once=True)
    assert report == NativeServiceReport("drained-cycle", "DRAINED", None, None)
    assert qualified == []
    assert opened == ["open", "close"]
    terminal = _diagnostics(tmp_path)[-1][1]
    assert terminal == {
        "cycle_id": "drained-cycle", "outcome": "DRAINED",
        "failure_class": None, "pipeline": None,
    }


def test_native_service_preflight_precedes_lock_ledger_and_pipeline(tmp_path, monkeypatch):
    factory, opened = _pipeline(
        tmp_path, monkeypatch, lambda _: pytest.fail("pipeline tick after failed preflight"),
    )
    ledger = tmp_path / "unpublished.sqlite3"
    lock = tmp_path / "locks" / "hermes-native.lock"

    def reject():
        raise ValueError("deployment path differs")

    service = NativeService(
        pipeline_factory=factory, ledger_path=str(ledger), lock_path=lock,
        stop_check=lambda: None, preflight=reject,
    )
    with pytest.raises(ValueError, match="deployment path differs"):
        service.run(once=True)
    assert opened == []
    assert not ledger.exists()
    assert not lock.parent.exists()


def test_native_service_binds_both_cycle_records_to_runtime_identity(tmp_path, monkeypatch):
    factory, _ = _pipeline(tmp_path, monkeypatch, lambda _: NativePipelineReport((), {}, 0))
    identity = digest_canonical({"actual_test_deployment": str(tmp_path)})

    @contextmanager
    def bound():
        with factory() as pipeline:
            pipeline.runtime_identity_digest = identity
            yield pipeline

    qualified = []

    def qualify(connection, digest):
        assert connection.execute("SELECT count(*) FROM ledger WHERE kind LIKE 'NATIVE_SERVICE_CYCLE_%'").fetchone()[0] == 0
        qualified.append(digest)

    _service(tmp_path, bound, qualify_once=qualify).run(once=True)
    assert qualified == [identity]
    records = tuple(payload for _, payload in _diagnostics(tmp_path))
    assert len(records) == 2
    assert all(record["runtime_identity_digest"] == identity for record in records)


def test_hermes_native_once_cli_reports_exact_terminal(tmp_path, monkeypatch, capsys, caplog):
    caplog.set_level(logging.WARNING)
    caplog.set_level(logging.WARNING, logger="newsroom.authority.open")
    caplog.handler.setLevel(logging.INFO)
    factory, _ = _pipeline(
        tmp_path, monkeypatch, lambda _cycle_id: NativePipelineReport((), {}, 0), ledger_name="cli.sqlite3",
    )

    def service_factory(args):
        assert args.once and args.interval == 3 and args.failure_backoff == 9
        assert logging.getLogger().level == logging.WARNING
        assert not logging.getLogger("noisy_dependency").isEnabledFor(logging.INFO)
        logging.getLogger("newsroom.authority.open").info("authority timing enabled")
        return NativeService(
            pipeline_factory=factory, ledger_path=args.ledger,
            lock_path=tmp_path / "cli.lock", stop_check=lambda: None,
            interval_seconds=args.interval,
            failure_backoff_seconds=args.failure_backoff,
            cycle_id_factory=lambda: "cli-cycle",
        )

    assert main(service_factory, [
        "--once", "--ledger", str(tmp_path / "cli.sqlite3"),
        "--lock", str(tmp_path / "ignored-by-interim-factory.lock"),
        "--interval", "3", "--failure-backoff", "9",
    ]) == 0
    assert "authority timing enabled" in caplog.messages
    assert json.loads(capsys.readouterr().out) == {
        "public_effect": False,
        "service": {
            "cycle_id": "cli-cycle", "failure_class": None,
            "outcome": "COMPLETE",
            "pipeline": {"revision_states": {}, "sources": [], "unclassified_revisions": 0},
        },
    }


def test_hermes_native_operator_drain_is_clean_and_not_an_owner_stop(
    tmp_path, monkeypatch, capsys,
):
    service_event = threading.Event()

    def tick(_cycle_id):
        service_event.set()
        raise OperatorDrainRequested

    factory, _ = _pipeline(tmp_path, monkeypatch, tick)

    def service_factory(args):
        return NativeService(
            pipeline_factory=factory, ledger_path=args.ledger,
            lock_path=tmp_path / "drain-cli.lock", stop_check=lambda: None,
            cycle_id_factory=lambda: "drain-cli-cycle",
            service_event=service_event,
        )

    assert main(service_factory, [
        "--once", "--ledger", str(tmp_path / "drain-cli.sqlite3"),
        "--lock", str(tmp_path / "ignored.lock"),
    ]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "public_effect": False,
        "service": {
            "cycle_id": "drain-cli-cycle", "failure_class": None,
            "outcome": "DRAINED", "pipeline": None,
        },
    }


def test_hermes_native_owner_stop_is_not_a_supervisor_crash(tmp_path, monkeypatch, capsys):
    factory, opened = _pipeline(tmp_path, monkeypatch, lambda _: None)

    def stopped():
        raise VetoError("private stop details")

    assert main(lambda args: _service(tmp_path, factory, stop_check=stopped), [
        "--once", "--ledger", str(tmp_path / "unpublished.sqlite3"),
        "--lock", str(tmp_path / "hermes-native.lock"),
    ]) == 0
    assert not opened
    assert not (tmp_path / "unpublished.sqlite3").exists()
    assert json.loads(capsys.readouterr().out) == {
        "service": None, "owner_stop": True, "public_effect": False,
    }


@pytest.mark.parametrize("states,unclassified", [({"UNKNOWN": 1, "QUEUED": 1}, 1), ({"QUEUED": 1}, 0)])
def test_malformed_pending_report_does_not_silently_defer_qualification(tmp_path, monkeypatch, states, unclassified):
    from newsroom.control_plane.native_qualification import NativeQualificationError

    factory, opened = _pipeline(tmp_path, monkeypatch, lambda _: NativePipelineReport((), states, unclassified))
    with pytest.raises(NativeQualificationError, match="terminal inventory differs"):
        _service(tmp_path, factory, qualify_once=lambda *_: pytest.fail("invalid qualification"),
                 wait=lambda _: pytest.fail("invalid report was silently deferred")).run()
    assert opened == ["open", "close"]


def test_unknown_assessment_continues_same_open_without_redispatch(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from newsroom.control_plane.native_evidence import NativeEvidenceController
    from newsroom.control_plane.native_progress import NativeRevisionJournal
    from newsroom.control_plane.native_publication import NativePublicationContinuation
    from newsroom.control_plane.store import connect
    from newsroom.tests.authority_helpers import proof
    from newsroom.tests.test_native_graphiti import _native
    from newsroom.tests.test_native_publication_continuation import _Authority, _Publication, _source

    connection = connect(str(tmp_path / "unpublished.sqlite3"))
    unit = _native()
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="ASSESSMENT_STARTED", facts={
        "candidate_version_id": "candidate-version", "intake_receipt_id": "intake-receipt",
        "assessment_started_at": "2026-09-08T12:01:00Z", "acquisition_attempt_count": 1,
    })
    monkeypatch.setattr(NativeEvidenceController, "acquire_and_retain",
                        lambda *_args, **_kwargs: pytest.fail("unknown effect was retried"))
    runtime = SimpleNamespace(
        authority=_Authority(), ingress=object(), publication=_Publication(),
        proof=proof(), policies=SimpleNamespace(publication=object()),
    )
    continuation = NativePublicationContinuation(
        journal=journal, runtime=runtime,
        evidence_controller=object.__new__(NativeEvidenceController),
        sources={unit.revision_id: (_source(unit),)},
    )
    order = []

    def tick(cycle):
        order.append(cycle)
        if cycle in {"pending", "continuing"}:
            result = continuation.advance(revision_id=unit.revision_id, candidate_version_id="candidate-version")
            assert result.state == "ASSESSMENT_INTERRUPTED"
        elif cycle == "settled":
            # Fixture receipt of a later terminal disposition, not a retry or
            # permission to manufacture one in the service readiness predicate.
            journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts={
                **journal.current(unit.revision_id)["facts"], "reason": "SOURCE_LOCAL_EVIDENCE_HOLD",
            })
        return NativePipelineReport((), {journal.current(unit.revision_id)["stage"]: 1}, 0)

    factory, opened = _pipeline(tmp_path, monkeypatch, tick)
    identity = digest_canonical({"runtime": "unknown-assessment-continuation"})

    @contextmanager
    def bound():
        with factory() as pipeline:
            pipeline.runtime_identity_digest = identity
            yield pipeline

    waits = []
    try:
        result = _service(
            tmp_path, bound,
            cycle_id_factory=iter(("pending", "continuing", "settled", "unchanged")).__next__,
            qualify_once=lambda *_: order.append("qualified"),
            wait=lambda _: waits.append(True) or len(waits) == 4,
        ).run()
        assert result.pipeline.revision_states == {"EVIDENCE_HOLD": 1}
        assert order == ["pending", "continuing", "settled", "qualified", "unchanged"]
        assert opened == ["open", "close"]
        assert journal.current(unit.revision_id)["facts"]["acquisition_attempt_count"] == 1
        assert runtime.authority.receives == 0 and runtime.publication.calls == 0
    finally:
        connection.close()


def test_service_qualifies_exact_repeated_metadata_hold_without_restart(tmp_path, monkeypatch):
    from newsroom.control_plane.native_qualification import record_qualification, validate_qualification
    from newsroom.tests.test_native_qualification import IDENTITY, _content_hold, _cycle, _open

    connection = _open(tmp_path / "unpublished.sqlite3")
    held = _content_hold("SOURCE_ITEM_METADATA_HOLD")
    held["observations"] *= 2
    held["item_holds"] *= 2
    journal = _cycle(connection, source_override=held)
    report = NativePipelineReport(journal.portfolio, {"EVIDENCE_HOLD": 1}, 0)
    factory, opened = _pipeline(tmp_path, monkeypatch, lambda _: report)

    @contextmanager
    def bound():
        with factory() as pipeline:
            pipeline.runtime_identity_digest = IDENTITY
            yield pipeline

    qualified, waits = [], []

    def qualify(ledger, identity):
        qualified.append(record_qualification(ledger, identity))

    try:
        result = _service(tmp_path, bound, qualify_once=qualify,
                          wait=lambda _: waits.append(True) or len(waits) == 2).run()
        assert result.outcome == "COMPLETE"
        assert opened == ["open", "close"] and len(qualified) == 1
        assert validate_qualification(connection, IDENTITY) == qualified[0]
        source = result.pipeline.sources[0]
        assert source["status"] == "HOLD"
        assert source["item_holds"] == [list(value) for value in held["item_holds"]]
        assert source["observations"] == [list(value) for value in held["observations"]]
    finally:
        connection.close()


@pytest.mark.parametrize("observation_bytes", (100, 10000))
def test_terminal_references_retained_portfolio_without_changing_logical_report(
    tmp_path, monkeypatch, observation_bytes,
):
    from dataclasses import asdict
    from newsroom.control_plane.native_qualification import record_qualification, validate_qualification
    from newsroom.tests.test_native_qualification import IDENTITY, _cycle, _open

    connection = _open(tmp_path / "unpublished.sqlite3")
    journal = _cycle(connection, source_override={"observations": ((
        "https://fixture.example/" + "x" * observation_bytes,
        "sha256:" + "b" * 64,
        "00000000-0000-4000-8000-000000000001",
        "00000000-0000-4000-8000-000000000002",
    ),)})
    expected = NativePipelineReport(journal.portfolio, {"EVIDENCE_HOLD": 1}, 0)
    factory, opened = _pipeline(tmp_path, monkeypatch, lambda _: expected)

    @contextmanager
    def bound():
        with factory() as pipeline:
            pipeline.runtime_identity_digest = IDENTITY
            yield pipeline

    qualified, waits = [], []
    try:
        result = _service(
            tmp_path, bound,
            qualify_once=lambda ledger, identity: qualified.append(record_qualification(ledger, identity)),
            wait=lambda _: waits.append(True) or len(waits) == 2,
        ).run()
        assert opened == ["open", "close"] and len(qualified) == 1
        assert asdict(result.pipeline) == asdict(expected)
        assert validate_qualification(connection, IDENTITY) == qualified[0]
        portfolio = connection.execute(
            "SELECT seq,payload_digest FROM ledger WHERE kind='NATIVE_SOURCE_PORTFOLIO'"
        ).fetchall()
        assert len(portfolio) == 1
        terminals = [payload for kind, payload in _diagnostics(tmp_path)
                     if kind == "NATIVE_SERVICE_CYCLE_TERMINAL"]
        assert len(terminals) == 2
        for terminal in terminals:
            raw = json.dumps(terminal)
            pipeline = terminal["pipeline"]
            assert pipeline == {
                "source_portfolio_ref": {"seq": portfolio[0][0], "payload_digest": portfolio[0][1]},
                "revision_states": {"EVIDENCE_HOLD": 1}, "unclassified_revisions": 0,
            }
            assert len(raw.encode()) < 700
    finally:
        connection.close()
