"""Process-owned Cursor SDK transport for Graphiti chat."""

from __future__ import annotations

import atexit
import importlib
import importlib.metadata
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from newsroom.authority.canonical import digest_canonical
from newsroom.graphiti_adapter.evaluation_packet import CURSOR_AGENT_MODEL_ID
from newsroom.graphiti_adapter.usage_meter import (
    cursor_sdk_usage,
    unreported_cli_usage,
)

MIN_SDK_VERSION = "1.0.29"
MIN_SDK_REQUIREMENT = "cursor-sdk>=1.0.29"
MIN_COMPOSER_FLOOR = "2.5"
MIN_SDK_VERSION_TUPLE = (1, 0, 29)
MIN_COMPOSER_VERSION_TUPLE = (2, 5, 0)
_COMPOSER_ID_PATTERN = re.compile(r"^composer-(\d+)\.(\d+)(?:\.(\d+))?$")
# Backward-compatible floor-default aliases; must not enforce exact admission.
PINNED_SDK_VERSION = MIN_SDK_VERSION
PINNED_SDK_LOCK_IDENTITY = MIN_SDK_REQUIREMENT
PINNED_MODEL = CURSOR_AGENT_MODEL_ID
CURSOR_SDK_TRANSPORT = "CURSOR_SDK"
CURSOR_SDK_QUALIFICATION_SCHEMA_VERSION = "newsroom.cursor-sdk-qualification.v2"
CURSOR_SDK_QUALIFICATION_SCHEMA_VERSION_V1 = "newsroom.cursor-sdk-qualification.v1"
CURSOR_SDK_TERMINAL_SCHEMA_VERSION = "newsroom.cursor-sdk-terminal.v1"
CURSOR_SDK_AUTH_SOURCE = "CURSOR_API_KEY"
CURSOR_SDK_DISALLOWED_TOOLS = ("shell", "mcp", "task")
CURSOR_OUTPUT_BASE_BYTES = 64 * 1024
CURSOR_OUTPUT_BYTES_PER_TOKEN = 64
CURSOR_OUTPUT_LIMIT_FORMULA = "65536+64*REQUEST_MAX_TOKENS"
CURSOR_OUTPUT_LIMIT_IDENTITY = (
    "cursor-sdk-controller-output-v1:" + CURSOR_OUTPUT_LIMIT_FORMULA
)
CURSOR_STDOUT_LIMIT_IDENTITY = CURSOR_OUTPUT_LIMIT_IDENTITY
_PROCESS_CLIENT_UNARY_TIMEOUT_SECONDS = 160.0
_PROCESS_CLIENT_STREAM_TIMEOUT_SECONDS = 160.0
_PROCESS_CLIENT_MAX_RETRIES = 0
_bound_runtime: "CursorSdkRuntime | None" = None
_official_runtime: "OfficialCursorSdkRuntime | None" = None
_official_lock = threading.Lock()


class CliPredispatchRefusal(RuntimeError):
    """Configuration, authentication or catalogue refused before send()."""

    def __init__(
        self,
        message: str,
        *,
        qualification_evidence: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.qualification_evidence = dict(qualification_evidence or {})


class CursorSdkError(RuntimeError):
    """A typed SDK, network or model failure after or during send()."""

    def __init__(
        self,
        message: str,
        *,
        error_class: str,
        status: str = "error",
        error_code: str = "UNOBSERVED",
        usage: Mapping[str, object] | None = None,
        execution: "CursorSdkExecution | None" = None,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.status = status
        self.error_code = error_code
        self.usage = dict(usage or unreported_cli_usage())
        self.execution = execution


class CursorSdkAmbiguousDispatch(CursorSdkError):
    """send() started but no durable run identity was retained."""

    def __init__(
        self,
        message: str,
        *,
        error_class: str,
        error_code: str = "UNOBSERVED",
        request_id: str = "UNOBSERVED",
    ) -> None:
        super().__init__(
            message,
            error_class=error_class,
            error_code=error_code,
            usage=unreported_cli_usage(),
        )
        self.request_id = request_id


class CursorToolCallViolation(CursorSdkError):
    """A tool-call event violated the no-tool Graphiti contract."""

    def __init__(self, execution: "CursorSdkExecution") -> None:
        super().__init__(
            "Cursor SDK Graphiti run emitted a tool call",
            error_class="TOOL_CALL",
            status=execution.status,
            error_code="TOOL_CALL",
            usage=execution.usage,
            execution=execution,
        )


class CursorSdkBoundedFailure(CursorSdkError):
    """Timeout or output-bound cancellation with any retained usage."""

    def __init__(
        self,
        message: str,
        *,
        error_class: str,
        execution: "CursorSdkExecution",
    ) -> None:
        super().__init__(
            message,
            error_class=error_class,
            status=execution.status,
            error_code=error_class,
            usage=execution.usage,
            execution=execution,
        )
        self.outcome = (
            "OUTPUT_LIMIT_EXCEEDED"
            if error_class == "OUTPUT_BOUND"
            else "TIMEOUT"
        )


class CursorSdkCleanupError(CursorSdkError):
    """Agent or workspace cleanup failed after the terminal was retained."""

    def __init__(self, execution: "CursorSdkExecution") -> None:
        super().__init__(
            "Cursor SDK Graphiti cleanup failed",
            error_class="CLEANUP",
            status=execution.status,
            error_code="CLEANUP",
            usage=execution.usage,
            execution=execution,
        )


@dataclass(frozen=True, slots=True)
class CursorSdkQualification:
    sdk_version: str
    sdk_floor: str
    lock_identity: str
    composer_floor: str
    selected_model: str
    model: str
    unary_timeout_seconds: int
    stream_timeout_seconds: int
    max_retries: int
    transport_policy_digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": CURSOR_SDK_QUALIFICATION_SCHEMA_VERSION,
            "transport": CURSOR_SDK_TRANSPORT,
            "sdk_floor": self.sdk_floor,
            "sdk_version": self.sdk_version,
            "lock_identity": self.lock_identity,
            "composer_floor": self.composer_floor,
            "selected_model": self.selected_model,
            "model": self.selected_model,
            "unary_timeout_seconds": self.unary_timeout_seconds,
            "stream_timeout_seconds": self.stream_timeout_seconds,
            "max_retries": self.max_retries,
            "transport_policy_digest": self.transport_policy_digest,
        }


@dataclass(frozen=True, slots=True)
class CursorSdkExecution:
    text: str
    usage: dict[str, object]
    qualification: CursorSdkQualification
    agent_id: str
    run_id: str
    request_id: str
    resolved_model: str
    status: str
    error_class: str
    error_code: str
    tool_call_count: int
    cancelled: bool
    duration_ms: int | None
    stream_message_classes: tuple[str, ...]
    diagnostic_digest: str

    def terminal_record(self) -> dict[str, object]:
        return {
            "schema_version": CURSOR_SDK_TERMINAL_SCHEMA_VERSION,
            "status": self.status,
            "error_class": self.error_class,
            "error_code": self.error_code,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "resolved_model": self.resolved_model,
            "tool_call_count": self.tool_call_count,
            "cancelled": self.cancelled,
            "duration_ms": self.duration_ms,
            "stream_message_classes": list(dict.fromkeys(self.stream_message_classes)),
            "diagnostic_digest": self.diagnostic_digest,
        }


@dataclass(frozen=True, slots=True)
class CursorSdkRunRequest:
    prompt: str
    api_key: str
    cwd: str
    store: str
    timeout: int
    max_output_bytes: int
    model: str
    idempotency_key: str | None = None


class CursorSdkRun(Protocol):
    id: str
    agent_id: str
    status: str
    model: object | None
    usage: object | None
    duration_ms: int | None

    def stream(self) -> Iterator[object]: ...

    def wait(self) -> object: ...

    def cancel(self) -> None: ...

    def close(self) -> None: ...


class CursorSdkRuntime(Protocol):
    sdk_version: str

    def list_model_ids(self, *, api_key: str) -> tuple[str, ...]: ...

    def start_run(
        self,
        request: CursorSdkRunRequest,
        *,
        dispatch_started: Callable[[], None] | None = None,
    ) -> CursorSdkRun: ...


def cursor_output_limit(max_tokens: int) -> int:
    if isinstance(max_tokens, bool) or max_tokens <= 0:
        raise ValueError("Graphiti requested max_tokens must be positive")
    return CURSOR_OUTPUT_BASE_BYTES + CURSOR_OUTPUT_BYTES_PER_TOKEN * max_tokens


def cursor_transport_policy(*, selected_model: str | None = None) -> dict[str, object]:
    return {
        "sdk": MIN_SDK_REQUIREMENT,
        "auth": CURSOR_SDK_AUTH_SOURCE,
        "composer_floor": MIN_COMPOSER_FLOOR,
        "model": selected_model or PINNED_MODEL,
        "tools": [],
        "disallowed_tools": list(CURSOR_SDK_DISALLOWED_TOOLS),
        "mcp_servers": {},
        "custom_tools": {},
        "subagents": {},
        "setting_sources": "OMITTED",
        "cwd": "EMPTY_NON_GIT",
        "store": "EPHEMERAL_LOCAL_ISOLATED",
        "fresh_run": True,
        "resume": False,
        "max_retries": _PROCESS_CLIENT_MAX_RETRIES,
        "unary_timeout_seconds": int(_PROCESS_CLIENT_UNARY_TIMEOUT_SECONDS),
        "stream_timeout_seconds": int(_PROCESS_CLIENT_STREAM_TIMEOUT_SECONDS),
    }


def cursor_transport_policy_digest(*, selected_model: str | None = None) -> str:
    return digest_canonical(cursor_transport_policy(selected_model=selected_model))


def _parse_stable_version(version: str) -> tuple[int, int, int]:
    if not isinstance(version, str) or not version.strip():
        raise ValueError("malformed SDK version")
    lowered = version.strip().lower()
    if any(token in lowered for token in ("a", "b", "rc", "alpha", "beta", "dev")):
        raise ValueError("prerelease SDK version")
    if "-" in version or "+" in version:
        raise ValueError("prerelease SDK version")
    parts = version.strip().split(".")
    if len(parts) not in {2, 3}:
        raise ValueError("malformed SDK version")
    try:
        major, minor = int(parts[0]), int(parts[1])
        patch = int(parts[2]) if len(parts) == 3 else 0
    except ValueError as exc:
        raise ValueError("malformed SDK version") from exc
    if major < 0 or minor < 0 or patch < 0:
        raise ValueError("malformed SDK version")
    return (major, minor, patch)


def composer_model_meets_floor(model_id: str) -> bool:
    version = _parse_composer_version(model_id)
    return version is not None and version >= MIN_COMPOSER_VERSION_TUPLE


def sdk_version_meets_floor(version: str) -> bool:
    return _sdk_version_meets_floor(version)


def _sdk_version_meets_floor(version: str) -> bool:
    try:
        return _parse_stable_version(version) >= MIN_SDK_VERSION_TUPLE
    except ValueError:
        return False


def _parse_composer_version(model_id: str) -> tuple[int, int, int] | None:
    if not isinstance(model_id, str):
        return None
    match = _COMPOSER_ID_PATTERN.match(model_id.strip())
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


def select_composer_model(model_ids: Sequence[str]) -> str:
    candidates: list[tuple[tuple[int, int, int], str]] = []
    for model_id in model_ids:
        version = _parse_composer_version(model_id)
        if version is None or version < MIN_COMPOSER_VERSION_TUPLE:
            continue
        candidates.append((version, model_id))
    if not candidates:
        raise CliPredispatchRefusal(
            "no compatible canonical composer model at or above composer-2.5"
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _validate_resolved_model(*, selected_model: str, resolved_model: str) -> None:
    if resolved_model == "UNREPORTED":
        raise CursorSdkError(
            "Cursor SDK Graphiti run did not report a resolved composer model",
            error_class="MODEL_DRIFT",
            error_code="RESOLVED_MODEL_UNREPORTED",
        )
    selected_version = _parse_composer_version(selected_model)
    resolved_version = _parse_composer_version(resolved_model)
    if selected_version is None or resolved_version is None:
        raise CursorSdkError(
            "Cursor SDK Graphiti run resolved a non-canonical composer model",
            error_class="MODEL_DRIFT",
            error_code="NON_CANONICAL_RESOLVED_MODEL",
        )
    if resolved_version < MIN_COMPOSER_VERSION_TUPLE:
        raise CursorSdkError(
            "Cursor SDK Graphiti run resolved a composer model below the floor",
            error_class="MODEL_DRIFT",
            error_code="RESOLVED_MODEL_BELOW_FLOOR",
        )
    if resolved_version != selected_version:
        raise CursorSdkError(
            "Cursor SDK Graphiti run resolved a different composer model than selected",
            error_class="MODEL_DRIFT",
            error_code="RESOLVED_MODEL_DRIFT",
        )


def bind_cursor_sdk_runtime(runtime: CursorSdkRuntime | None) -> CursorSdkRuntime | None:
    """Replace the process runtime. Tests inject a narrow fake here."""

    global _bound_runtime
    previous = _bound_runtime
    _bound_runtime = runtime
    return previous


def process_cursor_sdk_runtime() -> CursorSdkRuntime:
    if _bound_runtime is not None:
        return _bound_runtime
    global _official_runtime
    with _official_lock:
        if _official_runtime is None:
            _official_runtime = OfficialCursorSdkRuntime()
        return _official_runtime


def qualify_cursor_sdk(
    *,
    runtime: CursorSdkRuntime,
    api_key: str | None,
) -> CursorSdkQualification:
    if not _sdk_version_meets_floor(runtime.sdk_version):
        raise CliPredispatchRefusal(
            "Cursor SDK version is below the compatibility floor",
            qualification_evidence={"sdk_version": runtime.sdk_version},
        )
    if not api_key or not api_key.strip():
        raise CliPredispatchRefusal("purpose-provisioned CURSOR_API_KEY is required")
    selected_model = select_composer_model(runtime.list_model_ids(api_key=api_key))
    return CursorSdkQualification(
        sdk_version=runtime.sdk_version,
        sdk_floor=MIN_SDK_VERSION,
        lock_identity=MIN_SDK_REQUIREMENT,
        composer_floor=MIN_COMPOSER_FLOOR,
        selected_model=selected_model,
        model=selected_model,
        unary_timeout_seconds=int(_PROCESS_CLIENT_UNARY_TIMEOUT_SECONDS),
        stream_timeout_seconds=int(_PROCESS_CLIENT_STREAM_TIMEOUT_SECONDS),
        max_retries=_PROCESS_CLIENT_MAX_RETRIES,
        transport_policy_digest=cursor_transport_policy_digest(
            selected_model=selected_model
        ),
    )


def run_cursor_transport(
    *,
    prompt: str,
    max_tokens: int,
    timeout: int,
    dispatch_started: Callable[[], None] | None = None,
    idempotency_key: str | None = None,
) -> CursorSdkExecution:
    """Qualify, send one fresh no-context run, and retain the typed terminal."""

    if isinstance(max_tokens, bool) or max_tokens <= 0:
        raise ValueError("Graphiti requested max_tokens must be positive")
    if isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("Graphiti Cursor SDK timeout must be positive")
    if idempotency_key is None or not idempotency_key.strip():
        raise ValueError("Graphiti Cursor SDK idempotency_key is required")
    api_key = os.environ.get(CURSOR_SDK_AUTH_SOURCE)
    runtime = process_cursor_sdk_runtime()
    qualification = qualify_cursor_sdk(runtime=runtime, api_key=api_key)
    assert api_key is not None
    workspace = tempfile.TemporaryDirectory(prefix="newsroom-cursor-sdk-")
    execution: CursorSdkExecution | None = None
    workspace_ok = True
    try:
        root = Path(workspace.name)
        cwd = root / "cwd"
        store = root / "store"
        cwd.mkdir(mode=0o700)
        store.mkdir(mode=0o700)
        if any(cwd.iterdir()) or (cwd / ".git").exists():
            raise CliPredispatchRefusal("Cursor SDK working directory is not empty")
        request = CursorSdkRunRequest(
            prompt=prompt,
            api_key=api_key,
            cwd=str(cwd),
            store=str(store),
            timeout=timeout,
            max_output_bytes=cursor_output_limit(max_tokens),
            model=qualification.selected_model,
            idempotency_key=idempotency_key,
        )
        dispatch_marker_written = False

        def mark_dispatch_started() -> None:
            nonlocal dispatch_marker_written
            if dispatch_marker_written:
                raise RuntimeError("Cursor SDK dispatch marker repeated")
            if dispatch_started is not None:
                dispatch_started()
            dispatch_marker_written = True

        try:
            run = runtime.start_run(
                request,
                dispatch_started=mark_dispatch_started,
            )
        except CliPredispatchRefusal:
            raise
        except CursorSdkError:
            raise
        except Exception as exc:
            from newsroom.graphiti_adapter.cli_client import CliDispatchMarkerError

            if isinstance(exc, CliDispatchMarkerError):
                raise
            raise _map_sdk_exception(
                exc,
                dispatch_confirmed=dispatch_marker_written,
                send_attempted=dispatch_marker_written,
            ) from exc
        if not run.id:
            run.close()
            raise CursorSdkAmbiguousDispatch(
                "Cursor SDK run did not expose a durable run id after send()",
                error_class="MISSING_RUN_ID",
                error_code="MISSING_RUN_ID",
            )
        if not run.agent_id:
            run.close()
            raise CursorSdkAmbiguousDispatch(
                "Cursor SDK run did not expose a durable agent id after send()",
                error_class="MISSING_AGENT_ID",
                error_code="MISSING_AGENT_ID",
            )
        consume_error: BaseException | None = None
        try:
            execution = _consume_run(
                run,
                qualification=qualification,
                timeout=timeout,
                max_output_bytes=request.max_output_bytes,
            )
        except BaseException as exc:
            consume_error = exc
            observed = getattr(exc, "execution", None)
            if isinstance(observed, CursorSdkExecution):
                execution = observed
        try:
            run.close()
        except Exception as exc:
            if consume_error is None:
                raise CursorSdkCleanupError(
                    execution
                    or _cleanup_execution(
                        qualification,
                        agent_id=run.agent_id,
                        run_id=run.id,
                    )
                ) from exc
        if consume_error is not None:
            raise consume_error
        assert execution is not None
        return execution
    except BaseException:
        workspace_ok = False
        raise
    finally:
        try:
            workspace.cleanup()
        except Exception as exc:
            if workspace_ok:
                raise CursorSdkCleanupError(
                    execution or _cleanup_execution(qualification)
                ) from exc


async def run_cursor_transport_async(
    *,
    prompt: str,
    max_tokens: int,
    timeout: int,
    dispatch_started: Callable[[], None] | None = None,
    idempotency_key: str | None = None,
) -> CursorSdkExecution:
    import asyncio

    return await asyncio.to_thread(
        run_cursor_transport,
        prompt=prompt,
        max_tokens=max_tokens,
        timeout=timeout,
        dispatch_started=dispatch_started,
        idempotency_key=idempotency_key,
    )


def _consume_run(
    run: CursorSdkRun,
    *,
    qualification: CursorSdkQualification,
    timeout: int,
    max_output_bytes: int,
) -> CursorSdkExecution:
    started = time.monotonic()
    chunks: list[str] = []
    streamed_bytes = 0
    streamed_usage: list[object] = []
    classes: list[str] = []
    request_id = "UNOBSERVED"
    tool_call_count = 0
    cancel_class = ""
    seen_ids: set[str] = set()

    try:
        for message in run.stream():
            kind = _message_type(message)
            if kind:
                classes.append(kind)
            if kind == "request":
                observed = _field(message, "request_id")
                if isinstance(observed, str) and observed:
                    request_id = observed
            elif kind == "usage":
                streamed_usage.append(_field(message, "usage"))
            elif kind == "assistant":
                text = _assistant_text(message)
                chunks.append(text)
                streamed_bytes += len(text.encode("utf-8"))
            elif kind == "tool_call":
                call_id = _field(message, "call_id")
                identity = str(call_id) if call_id else f"anon:{len(seen_ids)}"
                if identity not in seen_ids:
                    seen_ids.add(identity)
                    tool_call_count += 1
                cancel_class = "TOOL_CALL"
                _cancel_once(run)
                break
            if streamed_bytes > max_output_bytes:
                cancel_class = "OUTPUT_BOUND"
                _cancel_once(run)
                break
            if time.monotonic() - started >= timeout:
                cancel_class = "TIMEOUT"
                _cancel_once(run)
                break
        terminal = run.wait()
    except CursorSdkError:
        raise
    except Exception as exc:
        mapped = _map_sdk_exception(
            exc,
            dispatch_confirmed=True,
            send_attempted=True,
            durable_run_identity=True,
        )
        mapped.execution = _execution(
            run,
            qualification=qualification,
            text="".join(chunks),
            streamed_usage=streamed_usage,
            terminal=None,
            request_id=request_id,
            tool_call_count=tool_call_count,
            cancelled=bool(cancel_class),
            classes=classes,
            error_class=mapped.error_class,
            error_code=mapped.error_code,
            status=mapped.status,
        )
        mapped.usage = mapped.execution.usage
        raise mapped from exc

    # SDK assistant deltas can span several reasoning/answer steps; wait()
    # supplies the final answer, not the concatenation of intermediate steps.
    terminal_text = _terminal_text(terminal) if _terminal_status(terminal) == "finished" else ""
    final_text = terminal_text or "".join(chunks)
    # A runtime may report text only from wait(). It has the same actual UTF-8
    # ceiling as streamed assistant text, independently of token consumption.
    post_wait_output_bound = not cancel_class and len(final_text.encode("utf-8")) > max_output_bytes
    if post_wait_output_bound:
        cancel_class = "OUTPUT_BOUND"
    execution = _execution(
        run,
        qualification=qualification,
        text=final_text,
        streamed_usage=streamed_usage,
        terminal=terminal,
        request_id=request_id,
        tool_call_count=tool_call_count,
        cancelled=(bool(cancel_class) and not post_wait_output_bound) or _terminal_status(terminal) == "cancelled",
        classes=classes,
        error_class=cancel_class or _terminal_error_class(terminal),
        error_code=cancel_class or _terminal_error_code(terminal),
        status=_terminal_status(terminal) or run.status or "error",
    )
    _validate_resolved_model(
        selected_model=qualification.selected_model,
        resolved_model=execution.resolved_model,
    )
    if cancel_class == "TOOL_CALL" or tool_call_count:
        raise CursorToolCallViolation(execution)
    if cancel_class == "OUTPUT_BOUND":
        raise CursorSdkBoundedFailure(
            "Cursor SDK Graphiti run exceeded the output bound",
            error_class="OUTPUT_BOUND",
            execution=execution,
        )
    if cancel_class == "TIMEOUT" or execution.status in {"expired"}:
        raise CursorSdkBoundedFailure(
            "Cursor SDK Graphiti run reached the controller deadline",
            error_class="TIMEOUT",
            execution=execution,
        )
    if execution.status in {"error", "expired", "cancelled"}:
        raise CursorSdkError(
            "Cursor SDK Graphiti run ended in a typed error",
            error_class=execution.error_class or "SDK_ERROR",
            status=execution.status,
            error_code=execution.error_code,
            usage=execution.usage,
            execution=execution,
        )
    if not execution.text.strip():
        raise CursorSdkError(
            "Cursor SDK Graphiti run returned empty assistant text",
            error_class="EMPTY_OUTPUT",
            status=execution.status,
            error_code="EMPTY_OUTPUT",
            usage=execution.usage,
            execution=execution,
        )
    return execution


def _execution(
    run: CursorSdkRun,
    *,
    qualification: CursorSdkQualification,
    text: str,
    streamed_usage: Sequence[object],
    terminal: object | None,
    request_id: str,
    tool_call_count: int,
    cancelled: bool,
    classes: Sequence[str],
    error_class: str,
    error_code: str,
    status: str,
) -> CursorSdkExecution:
    terminal_usage = _field(terminal, "usage") if terminal is not None else run.usage
    resolved = _resolved_model(terminal if terminal is not None else run)
    duration = _optional_int(
        _field(terminal, "duration_ms") if terminal is not None else run.duration_ms
    )
    usage = _reconcile_usage(streamed=streamed_usage, terminal=terminal_usage)
    return CursorSdkExecution(
        text=text,
        usage=usage,
        qualification=qualification,
        agent_id=run.agent_id,
        run_id=run.id,
        request_id=request_id,
        resolved_model=resolved,
        status=status,
        error_class=error_class or "NONE",
        error_code=error_code or "NONE",
        tool_call_count=tool_call_count,
        cancelled=cancelled,
        duration_ms=duration,
        stream_message_classes=tuple(classes),
        diagnostic_digest=_diagnostic_digest(
            status=status,
            error_class=error_class or "NONE",
            error_code=error_code or "NONE",
            tool_call_count=tool_call_count,
            cancelled=cancelled,
            duration_ms=duration,
        ),
    )


def _reconcile_usage(
    *, streamed: Sequence[object], terminal: object
) -> dict[str, object]:
    terminal_usage = cursor_sdk_usage(terminal)
    if terminal_usage["usage_basis"] == "PROVIDER_REPORTED":
        return terminal_usage
    reported = [
        item
        for item in (cursor_sdk_usage(value) for value in streamed)
        if item["usage_basis"] == "PROVIDER_REPORTED"
    ]
    if not reported:
        return unreported_cli_usage()
    total = {
        "usage_basis": "PROVIDER_REPORTED",
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_read_tokens": 0,
        "cached_write_tokens": 0,
        "reasoning_tokens": None,
        "total_tokens": 0,
    }
    reasoning = 0
    have_reasoning = False
    for item in reported:
        for field in (
            "input_tokens",
            "output_tokens",
            "cached_read_tokens",
            "cached_write_tokens",
            "total_tokens",
        ):
            total[field] = int(total[field]) + int(item[field])  # type: ignore[arg-type]
        value = item.get("reasoning_tokens")
        if isinstance(value, int) and not isinstance(value, bool):
            reasoning += value
            have_reasoning = True
    if have_reasoning:
        total["reasoning_tokens"] = reasoning
    return total


def _cleanup_execution(
    qualification: CursorSdkQualification,
    *,
    agent_id: str = "UNOBSERVED",
    run_id: str = "UNOBSERVED",
) -> CursorSdkExecution:
    return CursorSdkExecution(
        text="",
        usage=unreported_cli_usage(),
        qualification=qualification,
        agent_id=agent_id,
        run_id=run_id,
        request_id="UNOBSERVED",
        resolved_model=qualification.selected_model,
        status="error",
        error_class="CLEANUP",
        error_code="CLEANUP",
        tool_call_count=0,
        cancelled=False,
        duration_ms=None,
        stream_message_classes=(),
        diagnostic_digest=_diagnostic_digest(
            status="error",
            error_class="CLEANUP",
            error_code="CLEANUP",
            tool_call_count=0,
            cancelled=False,
            duration_ms=None,
        ),
    )


def _cancel_once(run: CursorSdkRun) -> None:
    if str(getattr(run, "status", "")) == "running":
        run.cancel()


def _message_type(message: object) -> str:
    value = _field(message, "type")
    return str(value) if isinstance(value, str) else ""


def _assistant_text(message: object) -> str:
    payload = _field(message, "message")
    content = _field(payload, "content") if payload is not None else None
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for block in content:
            if _field(block, "type") == "text":
                text = _field(block, "text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    text = _field(message, "text")
    return text if isinstance(text, str) else ""


def _field(value: object, name: str) -> object:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _optional_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _resolved_model(value: object) -> str:
    model = _field(value, "model")
    identity = _field(model, "id") if model is not None else model
    if isinstance(identity, str) and identity.strip():
        return identity
    return "UNREPORTED"


def _terminal_text(terminal: object) -> str:
    text = _field(terminal, "result")
    return text if isinstance(text, str) else ""


def _terminal_status(terminal: object) -> str:
    status = _field(terminal, "status")
    return status if isinstance(status, str) else ""


def _terminal_error_class(terminal: object) -> str:
    status = _terminal_status(terminal)
    if status == "cancelled":
        return "CANCELLED"
    if status == "error":
        return "SDK_ERROR"
    if status == "expired":
        return "TIMEOUT"
    return "NONE"


def _terminal_error_code(terminal: object) -> str:
    code = _field(terminal, "error")
    if isinstance(code, Mapping):
        value = code.get("code") or code.get("type")
        return str(value) if value else _terminal_error_class(terminal)
    if code is None:
        return _terminal_error_class(terminal)
    return str(getattr(code, "code", None) or code)


def _diagnostic_digest(
    *,
    status: str,
    error_class: str,
    error_code: str,
    tool_call_count: int,
    cancelled: bool,
    duration_ms: int | None,
) -> str:
    return digest_canonical(
        {
            "status": status,
            "error_class": error_class,
            "error_code": error_code,
            "tool_call_count": tool_call_count,
            "cancelled": cancelled,
            "duration_ms": duration_ms,
        }
    )


def _sdk_request_id(exc: Exception) -> str:
    observed = getattr(exc, "request_id", None)
    if isinstance(observed, str) and observed.strip():
        return observed
    return "UNOBSERVED"


def _map_sdk_exception(
    exc: Exception,
    *,
    dispatch_confirmed: bool,
    send_attempted: bool,
    durable_run_identity: bool = False,
) -> Exception:
    name = type(exc).__name__
    text = str(exc).lower()
    code = str(getattr(exc, "code", "") or getattr(exc, "type", "") or name)
    request_id = _sdk_request_id(exc)
    if name in {"RateLimitError"} or "rate limit" in text or "resource_exhausted" in text:
        error_class = "RATE_LIMIT"
    elif name in {"AuthenticationError", "PermissionDeniedError"} or (
        "unauthor" in text or "unauthenticated" in text or "api key" in text
    ):
        error_class = "AUTHENTICATION"
        if not send_attempted:
            return CliPredispatchRefusal("Cursor SDK authentication was refused")
    elif "model" in text and (
        "not found" in text or "unavailable" in text or "unsupported" in text
    ):
        error_class = "MODEL_UNAVAILABLE"
        if not send_attempted:
            return CliPredispatchRefusal(
                "no compatible canonical composer model at or above composer-2.5"
            )
    elif "timeout" in text or "deadline" in text or name in {"APITimeoutError"}:
        error_class = "TIMEOUT"
    elif "network" in text or name in {
        "APIConnectionError",
        "ConnectError",
        "NetworkError",
    } or "connect" in text:
        error_class = "NETWORK"
    else:
        error_class = "SDK_ERROR"
    if send_attempted and not durable_run_identity:
        return CursorSdkAmbiguousDispatch(
            "Cursor SDK Graphiti send did not retain a durable run identity",
            error_class=error_class,
            error_code=code,
            request_id=request_id,
        )
    if error_class == "TIMEOUT":
        return CursorSdkBoundedFailure(
            "Cursor SDK Graphiti run reached the controller deadline",
            error_class="TIMEOUT",
            execution=CursorSdkExecution(
                text="",
                usage=unreported_cli_usage(),
                qualification=CursorSdkQualification(
                    sdk_version=MIN_SDK_VERSION,
                    sdk_floor=MIN_SDK_VERSION,
                    lock_identity=MIN_SDK_REQUIREMENT,
                    composer_floor=MIN_COMPOSER_FLOOR,
                    selected_model=PINNED_MODEL,
                    model=PINNED_MODEL,
                    unary_timeout_seconds=int(_PROCESS_CLIENT_UNARY_TIMEOUT_SECONDS),
                    stream_timeout_seconds=int(_PROCESS_CLIENT_STREAM_TIMEOUT_SECONDS),
                    max_retries=_PROCESS_CLIENT_MAX_RETRIES,
                    transport_policy_digest=cursor_transport_policy_digest(
                        selected_model=PINNED_MODEL
                    ),
                ),
                agent_id="UNOBSERVED",
                run_id="UNOBSERVED",
                request_id="UNOBSERVED",
                resolved_model=PINNED_MODEL,
                status="error",
                error_class="TIMEOUT",
                error_code=code,
                tool_call_count=0,
                cancelled=True,
                duration_ms=None,
                stream_message_classes=(),
                diagnostic_digest=_diagnostic_digest(
                    status="error",
                    error_class="TIMEOUT",
                    error_code=code,
                    tool_call_count=0,
                    cancelled=True,
                    duration_ms=None,
                ),
            ),
        )
    return CursorSdkError(
        "Cursor SDK Graphiti run failed",
        error_class=error_class,
        error_code=code,
    )


class OfficialCursorSdkRuntime:
    """One process-owned official SDK client. Fresh agent/run per request."""

    def __init__(self) -> None:
        try:
            sdk = importlib.import_module("cursor_sdk")
        except ImportError as exc:
            raise CliPredispatchRefusal("cursor-sdk is not installed") from exc
        version = importlib.metadata.version("cursor-sdk")
        if not _sdk_version_meets_floor(version):
            raise CliPredispatchRefusal(
                "Cursor SDK version is below the compatibility floor",
                qualification_evidence={"sdk_version": version},
            )
        self.sdk_version = version
        self._sdk = sdk
        self._root = Path(tempfile.mkdtemp(prefix="newsroom-cursor-sdk-client-"))
        workspace = self._root / "workspace"
        store = self._root / "store"
        state = self._root / "bridge-state"
        workspace.mkdir(mode=0o700)
        store.mkdir(mode=0o700)
        state.mkdir(mode=0o700)
        client_cls = getattr(sdk, "CursorClient", None) or getattr(sdk, "Client")
        owned = client_cls.launch_bridge(
            workspace=str(workspace),
            state_root=str(state),
            local=sdk.LocalAgentOptions(
                cwd=str(workspace),
                custom_tools={},
                store=sdk.LocalAgentStoreConfig(type="jsonl", root_dir=str(store)),
            ),
            max_retries=_PROCESS_CLIENT_MAX_RETRIES,
            allow_api_key_env_fallback=False,
            client_timeout=_PROCESS_CLIENT_UNARY_TIMEOUT_SECONDS,
        )
        self._owned_client = owned
        self._client = owned.with_options(
            unary_timeout=_PROCESS_CLIENT_UNARY_TIMEOUT_SECONDS,
            stream_timeout=_PROCESS_CLIENT_STREAM_TIMEOUT_SECONDS,
            max_retries=_PROCESS_CLIENT_MAX_RETRIES,
        )
        atexit.register(self.close)

    def close(self) -> None:
        closer = getattr(self._owned_client, "close", None)
        if callable(closer):
            closer()

    def list_model_ids(self, *, api_key: str) -> tuple[str, ...]:
        models = self._client.models.list(api_key=api_key)
        identities: list[str] = []
        for model in models:
            identity = getattr(model, "id", None)
            if isinstance(identity, str):
                identities.append(identity)
        return tuple(identities)

    def start_run(
        self,
        request: CursorSdkRunRequest,
        *,
        dispatch_started: Callable[[], None] | None = None,
    ) -> CursorSdkRun:
        local = self._sdk.LocalAgentOptions(
            cwd=request.cwd,
            custom_tools={},
            store=self._sdk.LocalAgentStoreConfig(
                type="jsonl",
                root_dir=request.store,
            ),
        )
        try:
            agent = self._client.agents.create(
                self._sdk.AgentOptions(
                    model=request.model,
                    api_key=request.api_key,
                    tools=[],
                    disallowed_tools=list(CURSOR_SDK_DISALLOWED_TOOLS),
                    mcp_servers={},
                    agents={},
                    local=local,
                )
            )
        except Exception as exc:
            raise _map_sdk_exception(
                exc,
                dispatch_confirmed=False,
                send_attempted=False,
            ) from exc
        if dispatch_started is not None:
            try:
                dispatch_started()
            except BaseException:
                closer = getattr(agent, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:
                        pass
                raise
        try:
            # Local Send in cursor-sdk v1 rejects Idempotency-Key (cloud-only).
            # Retain request.idempotency_key for Newsroom correlation only.
            run = agent.send(request.prompt)
        except Exception as exc:
            closer = getattr(agent, "close", None)
            if callable(closer):
                closer()
            message = str(exc).lower()
            local_idempotency_unsupported = (
                "idempotency-key" in message
                and "only supported for cloud" in message
            )
            raise _map_sdk_exception(
                exc,
                dispatch_confirmed=True,
                # Local Idempotency-Key refusal never accepted a durable Send.
                send_attempted=not local_idempotency_unsupported,
            ) from exc
        return OfficialCursorSdkRun(agent=agent, run=run)


class OfficialCursorSdkRun:
    def __init__(self, *, agent: object, run: object) -> None:
        self._agent = agent
        self._run = run

    @property
    def id(self) -> str:
        return str(getattr(self._run, "id", "") or "")

    @property
    def agent_id(self) -> str:
        return str(
            getattr(self._run, "agent_id", None)
            or getattr(self._agent, "agent_id", "")
            or ""
        )

    @property
    def status(self) -> str:
        return str(getattr(self._run, "status", "") or "")

    @property
    def model(self) -> object | None:
        return getattr(self._run, "model", None)

    @property
    def usage(self) -> object | None:
        return getattr(self._run, "usage", None)

    @property
    def duration_ms(self) -> int | None:
        return _optional_int(getattr(self._run, "duration_ms", None))

    def stream(self) -> Iterator[object]:
        stream = getattr(self._run, "stream", None) or getattr(self._run, "messages")
        return stream()

    def wait(self) -> object:
        return self._run.wait()

    def cancel(self) -> None:
        if self.status == "running":
            self._run.cancel()

    def close(self) -> None:
        closer = getattr(self._agent, "close", None)
        if callable(closer):
            closer()


# Historical alias used by Grok preflight and existing observer imports.
cursor_stdout_limit = cursor_output_limit

__all__ = [
    "CURSOR_OUTPUT_LIMIT_IDENTITY",
    "CURSOR_STDOUT_LIMIT_IDENTITY",
    "CURSOR_SDK_AUTH_SOURCE",
    "CURSOR_SDK_DISALLOWED_TOOLS",
    "CURSOR_SDK_QUALIFICATION_SCHEMA_VERSION",
    "CURSOR_SDK_QUALIFICATION_SCHEMA_VERSION_V1",
    "CURSOR_SDK_TERMINAL_SCHEMA_VERSION",
    "CURSOR_SDK_TRANSPORT",
    "MIN_COMPOSER_FLOOR",
    "MIN_SDK_REQUIREMENT",
    "MIN_SDK_VERSION",
    "PINNED_MODEL",
    "PINNED_SDK_LOCK_IDENTITY",
    "PINNED_SDK_VERSION",
    "CliPredispatchRefusal",
    "CursorSdkAmbiguousDispatch",
    "CursorSdkBoundedFailure",
    "CursorSdkCleanupError",
    "CursorSdkError",
    "CursorSdkExecution",
    "CursorSdkQualification",
    "CursorSdkRunRequest",
    "CursorToolCallViolation",
    "OfficialCursorSdkRuntime",
    "bind_cursor_sdk_runtime",
    "cursor_output_limit",
    "cursor_stdout_limit",
    "cursor_transport_policy",
    "cursor_transport_policy_digest",
    "process_cursor_sdk_runtime",
    "qualify_cursor_sdk",
    "run_cursor_transport",
    "run_cursor_transport_async",
    "select_composer_model",
    "composer_model_meets_floor",
    "sdk_version_meets_floor",
]
