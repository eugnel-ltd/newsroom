"""Long-running shell for the autonomous private native pipeline."""

from __future__ import annotations

import fcntl
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path

from newsroom.control_plane.native_pipeline import NativePipeline, NativePipelineReport
from newsroom.control_plane.native_qualification import (
    NativeQualificationPending, qualification_report_ready,
)
from newsroom.authority.canonical import validate_sha256_digest
from newsroom.control_plane.store import append_ledger, connect
from newsroom.control_plane.diagnostic_logging import emit_diagnostic, start_diagnostic_logging
from newsroom.control_plane.veto import OperatorDrainRequested, VetoError

LOCK_IDENTITY = "newsroom-hermes-native-service-v1\n"


class NativeServiceAlreadyRunning(RuntimeError):
    """Raised before any pipeline effect when the singleton lock is held."""


@dataclass(frozen=True, slots=True)
class NativeServiceReport:
    cycle_id: str
    outcome: str
    failure_class: str | None
    pipeline: NativePipelineReport | None


@contextmanager
def _instance_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise NativeServiceAlreadyRunning("native service lock is held") from exc
        handle.seek(0)
        identity = handle.read()
        if not identity:
            handle.write(LOCK_IDENTITY)
            handle.flush()
            os.fsync(handle.fileno())
        elif identity != LOCK_IDENTITY:
            raise RuntimeError("native service lock identity differs")
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class NativeService:
    """Run every native revision to its retained outcome; never publish publicly."""

    def __init__(
        self, *,
        pipeline_factory: Callable[[], AbstractContextManager[NativePipeline]],
        ledger_path: str,
        lock_path: Path,
        stop_check: Callable[[], None],
        interval_seconds: float = 300,
        failure_backoff_seconds: float = 60,
        wait: Callable[[float], bool] | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        cycle_id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
        qualify_once: Callable[[sqlite3.Connection, str], object] | None = None,
        preflight: Callable[[], None] = lambda: None,
        service_event: threading.Event | None = None,
    ) -> None:
        if (
            not callable(pipeline_factory) or not callable(stop_check)
            or not callable(preflight)
            or not callable(monotonic_clock)
        ):
            raise TypeError(
                "native service pipeline, stop check and preflight are required"
            )
        if qualify_once is not None and not callable(qualify_once):
            raise TypeError("native qualification callback must be callable")
        if (
            not math.isfinite(interval_seconds)
            or not math.isfinite(failure_backoff_seconds)
            or interval_seconds <= 0
            or failure_backoff_seconds <= 0
        ):
            raise ValueError("native service waits must be positive")
        self._pipeline_factory = pipeline_factory
        self._ledger_path, self._lock_path = ledger_path, lock_path
        self._stop_check = stop_check
        self._interval, self._backoff = interval_seconds, failure_backoff_seconds
        self._shutdown = service_event or threading.Event()
        self._wait = wait or self._shutdown.wait
        self._monotonic_clock = monotonic_clock
        self._cycle_id = cycle_id_factory
        self._qualify_once = qualify_once
        self._preflight = preflight

    def request_shutdown(self) -> None:
        self._shutdown.set()

    def run(self, *, once: bool = False) -> NativeServiceReport | None:
        last = None
        # This must precede lock-directory creation and ledger connect. The
        # deployed preflight checks immutable path identity without opening I/O.
        self._preflight()
        with _instance_lock(self._lock_path):
            self._stop_check()
            ledger = connect(self._ledger_path)
            stop_diagnostics = start_diagnostic_logging(Path(self._ledger_path).parent / 'diagnostics')
            try:
                with self._pipeline_factory() as pipeline:
                    identity = getattr(pipeline, "runtime_identity_digest", None)
                    if identity is not None:
                        validate_sha256_digest(identity)
                    binding = {} if identity is None else {"runtime_identity_digest": identity}
                    qualified = False
                    while not self._shutdown.is_set():
                        cycle_started = self._monotonic_clock()
                        self._stop_check()
                        cycle_id = self._cycle_id()
                        if type(cycle_id) is not str or not cycle_id:
                            raise ValueError("native service cycle identity differs")
                        self._append(ledger, "NATIVE_SERVICE_CYCLE_STARTED", {
                            "cycle_id": cycle_id,
                            **binding,
                        })
                        try:
                            report = pipeline.tick(cycle_id=cycle_id)
                            if type(report) is not NativePipelineReport:
                                raise TypeError("native pipeline report differs")
                        except VetoError:
                            self._append(ledger, "NATIVE_SERVICE_CYCLE_TERMINAL", {
                                **binding,
                                "cycle_id": cycle_id,
                                "outcome": "STOPPED",
                                "failure_class": "VetoError",
                                "pipeline": None,
                            })
                            raise
                        except OperatorDrainRequested:
                            last = NativeServiceReport(
                                cycle_id, "DRAINED", None, None,
                            )
                        except Exception as exc:
                            last = NativeServiceReport(
                                cycle_id, "FAILED", type(exc).__name__, None,
                            )
                        else:
                            drained = self._shutdown.is_set()
                            last = NativeServiceReport(
                                cycle_id,
                                "DRAINED" if drained else "COMPLETE",
                                None,
                                None if drained else report,
                            )
                        self._append(ledger, "NATIVE_SERVICE_CYCLE_TERMINAL", {
                            **binding,
                            "cycle_id": last.cycle_id,
                            "outcome": last.outcome,
                            "failure_class": last.failure_class,
                            "pipeline": (
                                None if last.pipeline is None else pipeline.terminal_report(last.pipeline)
                            ),
                        })
                        if (
                            not qualified
                            and last.outcome == "COMPLETE"
                            and not self._shutdown.is_set()
                            and self._qualify_once is not None
                            and qualification_report_ready(
                                last.pipeline.revision_states,
                                last.pipeline.unclassified_revisions,
                            )
                        ):
                            if identity is None:
                                raise ValueError("native qualification requires a runtime identity")
                            try:
                                self._qualify_once(ledger, identity)
                            except NativeQualificationPending:
                                if once:
                                    raise
                                logging.getLogger(__name__).warning(
                                    "Native qualification pending: model usage remains unresolved"
                                )
                            else:
                                qualified = True
                        if last.outcome == "DRAINED" or once:
                            break
                        if last.outcome == "COMPLETE":
                            elapsed = self._monotonic_clock() - cycle_started
                            if not math.isfinite(elapsed) or elapsed < 0:
                                raise ValueError("native service monotonic clock differs")
                            wait_seconds = max(0, self._interval - elapsed)
                        else:
                            wait_seconds = self._backoff
                        if self._wait(wait_seconds):
                            break
            finally:
                ledger.close()
                stop_diagnostics()
        return last

    @staticmethod
    def _append(connection: sqlite3.Connection, kind: str, payload: dict) -> None:
        if kind in {"NATIVE_SERVICE_CYCLE_STARTED", "NATIVE_SERVICE_CYCLE_TERMINAL"}:
            emit_diagnostic(kind, payload)
            return
        append_ledger(connection, kind, payload)
        connection.commit()


__all__ = [
    "NativeService", "NativeServiceAlreadyRunning", "NativeServiceReport",
]
