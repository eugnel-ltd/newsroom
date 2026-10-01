"""Assessor-only reasoning profile; writer defaults and old profiles stay exact."""

import json
import sqlite3
import subprocess
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from newsroom.tests.assessor_fixture_support import candidate_fixture

from newsroom.control_plane import native_assessor, writer
from newsroom.control_plane.model_usage import InvocationEfficiencyPolicy, ModelUsageService
from newsroom.control_plane.native_assessor import (
    NativeAssessmentUsage, NativeEvidenceError,
)
from newsroom.control_plane.writer import (
    CONT_PRIMARY_COMMAND_FLAGS, CONT_PRIMARY_MODEL, CONT_PRIMARY_REASONING,
    _grok_command_flags, _grok_json_command,
)
from newsroom.tests.test_increment10_editorial import _ready_package
from newsroom.tests.test_increment10_ingress import _candidate
from newsroom.increment10.evidence import _base_package
from newsroom.tests.test_native_assessor import _usage


def _changed_policy(policy, **changes):
    values = asdict(policy)
    values.pop("canonical_digest")
    return InvocationEfficiencyPolicy.create(**(values | changes))


def test_assessor_flags_change_only_model_and_reasoning_values():
    low = _grok_command_flags()
    high = _grok_command_flags(native_assessor.REASONING, model=native_assessor.MODEL)
    assert CONT_PRIMARY_MODEL == "grok-4.6"
    assert CONT_PRIMARY_REASONING == "low"
    assert native_assessor.MODEL == "grok-4.7"
    assert native_assessor.REASONING == "high"
    assert low == CONT_PRIMARY_COMMAND_FLAGS
    assert high == native_assessor.COMMAND_FLAGS
    assert len(low) == len(high)
    changed = [index for index, pair in enumerate(zip(low, high, strict=True))
               if pair[0] != pair[1]]
    assert changed == [low.index("-m") + 1, low.index("--reasoning-effort") + 1]
    assert [low[index] for index in changed] == ["grok-4.6", "low"]
    assert [high[index] for index in changed] == ["grok-4.7", "high"]
    default_command = _grok_json_command("REQUEST", "SCHEMA", "SYSTEM")
    high_command = _grok_json_command(
        "REQUEST", "SCHEMA", "SYSTEM", reasoning_effort="high", model="grok-4.7",
    )
    assert [index for index, pair in enumerate(zip(
        default_command, high_command, strict=True,
    )) if pair[0] != pair[1]] == [
        default_command.index("-m") + 1,
        default_command.index("--reasoning-effort") + 1
    ]


def test_assessor_dispatch_passes_model_and_high_reasoning_overrides(monkeypatch):
    captured = []

    def run(prompt, **kwargs):
        captured.append((prompt, kwargs))
        return SimpleNamespace(text='{"package":{}}', usage={})

    monkeypatch.setattr(native_assessor, "_run_grok_json", run)
    result = native_assessor._dispatch_grok("exact prompt")
    assert result.text == '{"package":{}}'
    assert len(captured) == 1
    prompt, kwargs = captured[0]
    assert prompt == "exact prompt"
    assert kwargs["model"] == "grok-4.7"
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["schema"] is native_assessor.PROVIDER_SCHEMA
    assert kwargs["system_instruction"] == native_assessor.SYSTEM


@pytest.mark.parametrize("change", (
    {"model": "grok-4.6"},
    {"reasoning": "medium"},
    {"reasoning": CONT_PRIMARY_REASONING, "command_flags": CONT_PRIMARY_COMMAND_FLAGS},
    {"max_output_tokens": 10_000},
))
def test_policy_allocation_and_manifest_bind_current_profile_and_reject_stale(
    tmp_path, monkeypatch, change,
):
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)
    try:
        policy = usage._policy
        assert policy.model == native_assessor.MODEL == "grok-4.7"
        assert policy.reasoning == native_assessor.REASONING == "high"
        assert policy.max_output_tokens is None
        assert policy.command_flags == native_assessor.COMMAND_FLAGS
        allocation = usage.begin(candidate, base, "exact request")
        assert allocation.model == "grok-4.7"
        assert allocation.reasoning == "high"
        assert allocation.max_output_tokens is None
        with service._connection() as retained:
            manifest = json.loads(retained.execute(
                "SELECT record_json FROM model_invocation_context_manifests "
                "WHERE context_manifest_digest=?",
                (allocation.context_manifest_digest,),
            ).fetchone()[0])
        assert manifest["model"] == "grok-4.7"
        assert manifest["reasoning"] == "high"
        assert manifest["command_flags"] == list(native_assessor.COMMAND_FLAGS)
        assert manifest["input_bound"]["output_limit_enforced"] is False
        stale = _changed_policy(policy, **change)
        with pytest.raises(NativeEvidenceError, match="qualified native assessment"):
            NativeAssessmentUsage(service, stale)
        assert ModelUsageService(service.path).route_state(
            native_assessor.ROUTE
        )["state"] == "CLOSED"
    finally:
        connection.close()


@pytest.mark.parametrize("assessor", (True, False))
def test_actual_dispatch_argv_selects_assessor_profile_or_unchanged_writer_defaults(
    monkeypatch, assessor,
):
    commands = []
    monkeypatch.setattr(writer, "_minimal_grok_auth_bytes", lambda: b"{}")
    monkeypatch.setattr(writer, "_prove_grok_hermetic_capabilities", lambda _auth: None)

    def run(command, **kwargs):
        commands.append(command)
        assert Path(command[command.index("--prompt-file") + 1]).read_text() == "exact prompt"
        assert kwargs["cwd"] != str(Path.cwd())
        assert "--max-output-tokens" not in command
        assert "--max-tokens" not in command
        return '{"structured_output":{"package":{}}}'

    monkeypatch.setattr(writer, "_run", run)
    execute = native_assessor._dispatch_grok if assessor else writer.run_grok_cli
    assert json.loads(execute("exact prompt").text) == {"package": {}}
    command, = commands
    assert command[command.index("-m") + 1] == ("grok-4.7" if assessor else "grok-4.6")
    assert command[command.index("--reasoning-effort") + 1] == ("high" if assessor else "low")


@pytest.mark.parametrize(
    ("failure", "expected_type"),
    (("timeout", "CliTimeoutError"), ("empty", "CliEmptyOutputError")),
)
def test_cli_failure_type_survives_assessor_without_changing_unknown_accounting(
    tmp_path, monkeypatch, failure, expected_type,
):
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(writer, "_minimal_grok_auth_bytes", lambda: b"{}")
    monkeypatch.setattr(writer, "_prove_grok_hermetic_capabilities", lambda _auth: None)

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 300
        assert command[command.index("-m") + 1] == "grok-4.7"
        assert command[command.index("--reasoning-effort") + 1] == "high"
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                command, 300, output=b"partial-secret-output", stderr=b"private-diagnostic",
            )
        return SimpleNamespace(returncode=0, stdout=" \n", stderr="private-diagnostic")

    monkeypatch.setattr(writer.subprocess, "run", run)
    try:
        with pytest.raises(RuntimeError) as caught:
            native_assessor.AutonomousNativeEvidenceAssessor(
                native_assessor._dispatch_grok, usage=usage, dispatch_fence=nullcontext,
            )(candidate, base, (), ())
        assert type(caught.value).__name__ == expected_type
        assert str(caught.value).endswith(
            "writer timed out" if failure == "timeout" else "writer returned empty stdout"
        )
        assert len(calls) == 1
        with sqlite3.connect(service.path) as retained:
            terminal = json.loads(retained.execute(
                "SELECT record_json FROM model_invocation_terminals"
            ).fetchone()[0])
            assert terminal["outcome"] == "ASSESSOR_PROVIDER_FAILED"
            assert terminal["failure_class"] == "UNKNOWN_PROVIDER_FAILURE"
            assert terminal["usage_status"] == "ESTIMATED"
            assert terminal["pre_dispatch_zero_proved"] is False
            assert terminal["dispatch_at"] is not None
            assert retained.execute(
                "SELECT count(*) FROM ledger WHERE kind IN "
                "('NATIVE_ASSESSMENT_RESULT','NATIVE_ASSESSMENT_MATERIALISATION')"
            ).fetchone() == (0,)
            assert not any(
                "partial-secret-output" in (raw or "") or "private-diagnostic" in (raw or "")
                for raw, in retained.execute("SELECT payload_json FROM ledger")
            )
    finally:
        connection.close()
