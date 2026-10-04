"""Secret-free parse provenance does not change CLI acceptance or replay."""
from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority.types import UtcTimestamp
from newsroom.graphiti_adapter import cli_client
from newsroom.graphiti_adapter.result_snapshot import restore_validated_snapshot
from newsroom.graphiti_adapter.types import GraphitiAdapterContractError
from newsroom.tests.test_graphiti_adapter_real_executor import _real_attempt
from newsroom.tests.test_graphiti_cursor_sdk_transport import FakeUsage
from newsroom.tests.test_native_sdk_response_quality import _case


@pytest.mark.parametrize(
    "text, expected, diagnostic, decodes",
    [
        ("", None, {"parse_class": "EMPTY"}, 0),
        (" \n", None, {"parse_class": "EMPTY"}, 0),
        ("[]", None, {"parse_class": "NON_OBJECT", "json_type": "ARRAY"}, 1),
        ("false", None, {"parse_class": "NON_OBJECT", "json_type": "BOOLEAN"}, 1),
        ("null", None, {"parse_class": "NON_OBJECT", "json_type": "NULL"}, 1),
        ("42", None, {"parse_class": "NON_OBJECT", "json_type": "NUMBER"}, 1),
        ('"literal"', None, {"parse_class": "NON_OBJECT", "json_type": "STRING"}, 1),
        ('前置 {"é":}', None, {"parse_class": "JSON_SYNTAX", "json_error_position": 8}, 1),
        ('{"ok":true}{"bad":2}', None, {"parse_class": "JSON_SYNTAX", "json_error_position": 11}, 1),
        ('prefix {"ok":true} suffix', {"ok": True}, {}, 1),
        ('[{"ok":true}]', {"ok": True}, {}, 1),
        ('{"ok":true}', {"ok": True}, {}, 1),
    ],
)
def test_parse_metadata_preserves_existing_extraction_and_decodes_once(
    monkeypatch, text, expected, diagnostic, decodes
):
    assert cli_client._parsed_object(text) == expected
    original = json.loads
    decoded = []

    def counted(value, *args, **kwargs):
        decoded.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(cli_client.json, "loads", counted)
    details = {}
    assert cli_client._parsed_object(text, diagnostic=details) == expected
    assert details == diagnostic
    assert len(decoded) == decodes
    assert len(canonical_json_bytes(details)) < 128


def test_advisory_parse_metadata_is_retained_bound_and_legacy_bytes_unchanged(
    tmp_path, monkeypatch
):
    from newsroom.graphiti_adapter.real import _EpisodeTelemetry, _raw_receipt

    (tmp_path / "current").mkdir()
    text = '前置 {"é":}'
    service, _observer, runtime, calls, run = _case(
        tmp_path / "current", monkeypatch, text=text, usage=FakeUsage(2, 1)
    )
    with pytest.raises(cli_client.CliResponseError):
        run()
    call = calls[0]
    quality = call["response_quality"]
    assert quality == {
        "final_utf8_bytes": 14,
        "final_text_digest": digest_bytes(text.encode("utf-8")),
        "json_parse": "NOT_OBJECT",
        "schema_status": "NOT_CHECKED",
        "parse_class": "JSON_SYNTAX",
        "json_error_position": 8,  # Unicode character offset in the original output, not bytes.
    }
    assert set(quality) <= {
        "final_utf8_bytes", "final_text_digest", "json_parse", "schema_status",
        "parse_class", "json_error_position", "json_type",
    }
    assert text not in canonical_json_bytes(call).decode()
    assert "é" not in canonical_json_bytes(call).decode()
    terminal = service.terminal(call["model_invocation_id"])
    assert terminal.outcome == "MALFORMED_OUTPUT"
    assert terminal.usage_status.value == "REPORTED"
    assert len(runtime.requests) == 1

    instant = UtcTimestamp(datetime(2026, 8, 20, tzinfo=UTC))
    attempt = replace(_real_attempt(tmp_path / "snapshot"), reference_time=instant)
    raw = _raw_receipt(
        attempt, started_at=instant,
        telemetry=_EpisodeTelemetry(provider_attempt_number=1, chat_invocations=[call]),
        result=None, proposals=(),
    )
    raw.pop("raw_output_digest")
    raw["combined_temporal_failure_code"] = "PIPELINE_FAILED"
    raw["raw_output_digest"] = digest_bytes(canonical_json_bytes(raw))
    restored = restore_validated_snapshot(raw=raw, attempt=attempt)
    assert restored.chat_invocations[0]["response_quality"] == quality
    changed = copy.deepcopy(raw)
    changed["chat_invocations"][0]["response_quality"]["json_error_position"] = 9
    with pytest.raises(GraphitiAdapterContractError, match="inner digest differs"):
        restore_validated_snapshot(raw=changed, attempt=attempt)

    current_parser = cli_client._parsed_object

    def legacy_parser(raw, *, diagnostic=None):
        assert diagnostic is None
        try:
            payload = json.loads(cli_client.extract_json(raw))
        except (RuntimeError, json.JSONDecodeError, TypeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def without_advisory(parser):
        monkeypatch.setattr(cli_client, "_parsed_object", parser)
        calls = []
        with pytest.raises(cli_client.CliResponseError):
            asyncio.run(cli_client.run_cli_chain(
                prompt="fixture", schema=None,
                cursor_runner=lambda *_args, **_kwargs: text,
                grok_runner=lambda *_args, **_kwargs: pytest.fail("fallback dispatched"),
                invocations=calls, fallback_permitted=False,
            ))
        assert "response_quality" not in calls[0]
        return canonical_json_bytes(calls[0])

    assert without_advisory(current_parser) == without_advisory(legacy_parser)
