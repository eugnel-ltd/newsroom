"""Assessor-only reasoning profile; writer defaults and old profiles stay exact."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from newsroom.control_plane import native_assessor
from newsroom.control_plane.model_usage import InvocationEfficiencyPolicy, ModelUsageService
from newsroom.control_plane.native_assessor import (
    NativeAssessmentUsage, NativeEvidenceError,
)
from newsroom.control_plane.writer import (
    CONT_PRIMARY_COMMAND_FLAGS, CONT_PRIMARY_REASONING,
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


def test_assessor_medium_flags_change_only_the_reasoning_value():
    low = _grok_command_flags()
    medium = _grok_command_flags(native_assessor.REASONING)
    assert CONT_PRIMARY_REASONING == "low"
    assert native_assessor.REASONING == "medium"
    assert low == CONT_PRIMARY_COMMAND_FLAGS
    assert medium == native_assessor.COMMAND_FLAGS
    assert len(low) == len(medium)
    changed = [index for index, pair in enumerate(zip(low, medium, strict=True))
               if pair[0] != pair[1]]
    assert changed == [low.index("--reasoning-effort") + 1]
    assert low[changed[0]] == "low"
    assert medium[changed[0]] == "medium"
    default_command = _grok_json_command("REQUEST", "SCHEMA", "SYSTEM")
    medium_command = _grok_json_command(
        "REQUEST", "SCHEMA", "SYSTEM", reasoning_effort="medium",
    )
    assert [index for index, pair in enumerate(zip(
        default_command, medium_command, strict=True,
    )) if pair[0] != pair[1]] == [
        default_command.index("--reasoning-effort") + 1
    ]


def test_assessor_dispatch_passes_only_the_medium_reasoning_override(monkeypatch):
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
    assert kwargs["reasoning_effort"] == "medium"
    assert kwargs["schema"] is native_assessor.PROVIDER_SCHEMA
    assert kwargs["system_instruction"] == native_assessor.SYSTEM


def test_policy_allocation_and_manifest_bind_medium_and_reject_low(
    tmp_path, monkeypatch,
):
    connection, _port, candidate = _candidate(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)
    try:
        policy = usage._policy
        assert policy.reasoning == native_assessor.REASONING == "medium"
        assert policy.command_flags == native_assessor.COMMAND_FLAGS
        allocation = usage.begin(candidate, base, "exact request")
        assert allocation.reasoning == "medium"
        with service._connection() as retained:
            manifest = json.loads(retained.execute(
                "SELECT record_json FROM model_invocation_context_manifests "
                "WHERE context_manifest_digest=?",
                (allocation.context_manifest_digest,),
            ).fetchone()[0])
        assert manifest["reasoning"] == "medium"
        assert manifest["command_flags"] == list(native_assessor.COMMAND_FLAGS)
        stale = _changed_policy(
            policy,
            reasoning=CONT_PRIMARY_REASONING,
            command_flags=CONT_PRIMARY_COMMAND_FLAGS,
        )
        with pytest.raises(NativeEvidenceError, match="qualified native assessment"):
            NativeAssessmentUsage(service, stale)
        assert ModelUsageService(service.path).route_state(
            native_assessor.ROUTE
        )["state"] == "CLOSED"
    finally:
        connection.close()
