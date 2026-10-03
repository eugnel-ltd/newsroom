import json
from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from newsroom.authority.canonical import digest_canonical
from newsroom.control_plane import native_story_model as module
from newsroom.control_plane.model_usage import ModelUsageAdmissionError, ModelUsageIntegrityError
from newsroom.tests.test_native_assessor import _usage

NOW = datetime(2026, 10, 3, tzinfo=UTC)
SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


def _model(tmp_path, monkeypatch, *, invoke=None):
    service, assessor = _usage(tmp_path, monkeypatch)
    policies = {phase: module.story_model_policy(assessor._policy, phase=phase, schema=SCHEMA,
        revision="1" * 40, evidence_digest="sha256:" + "a" * 64) for phase in module.ROUTES}
    for policy in policies.values():
        service.register_policy(policy)
    monkeypatch.setattr(module, "cont_writer_implementation_identity", lambda: ("1" * 40, True))
    monkeypatch.setattr(module, "read_grok_command_semantic_version", lambda: "1.0.8")
    calls = []
    def dispatch(prompt, **kwargs):
        calls.append((prompt, kwargs))
        if invoke:
            return invoke(prompt, **kwargs)
        return SimpleNamespace(text='{"text":"supported copy"}', usage={
            "usage_basis": "PROVIDER_REPORTED", "input_tokens": 9,
            "output_tokens": 3, "total_tokens": 12})
    model = module.NativeStoryModel(service, policies, fence=nullcontext,
        stop_check=lambda: None, clock=lambda: NOW, invoke=dispatch)
    return model, service, calls


def _call(model, phase="DRAFT", candidate_id="candidate-1", **changes):
    request = {"source_package_digest": "sha256:" + "b" * 64, "facts": "approved facts"}
    request.update(changes)
    return model.call(request, phase=phase, schema=SCHEMA, system="source-bound instruction",
        candidate_id=candidate_id, hypothesis_digest="sha256:" + "c" * 64,
        admission_decision_id="decision-1")


def test_accounted_draft_and_review_replay_without_second_dispatch(tmp_path, monkeypatch):
    model, service, calls = _model(tmp_path, monkeypatch)
    assert _call(model) == {"text": "supported copy"}
    assert _call(model) == {"text": "supported copy"}
    assert _call(model, "REVIEW") == {"text": "supported copy"}
    assert len(calls) == 2
    with service._connection() as connection:
        assert connection.execute("SELECT count(*) FROM model_invocation_terminals WHERE outcome='COMPLETE' AND usage_status='REPORTED'").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM model_usage_current WHERE active=1").fetchone()[0] == 0


def test_changed_input_or_cached_response_never_dispatches_again(tmp_path, monkeypatch):
    model, service, calls = _model(tmp_path, monkeypatch)
    _call(model)
    with pytest.raises(ModelUsageAdmissionError, match="differs"):
        _call(model, facts="different facts")
    with pytest.raises(ModelUsageAdmissionError, match="identity differs"):
        _call(model, candidate_id="candidate-other")
    with service._connection() as connection:
        connection.execute("UPDATE native_story_model_results SET response_text='{}'")
    with pytest.raises(ModelUsageAdmissionError, match="differs"):
        _call(model)
    assert len(calls) == 1


def test_malformed_settled_response_is_retained_and_not_retried(tmp_path, monkeypatch):
    model, _service, calls = _model(tmp_path, monkeypatch, invoke=lambda *a, **k:
        SimpleNamespace(text="not JSON", usage={"usage_basis": "PROVIDER_REPORTED",
                                              "input_tokens": 9, "output_tokens": 3, "total_tokens": 12}))
    for _ in range(2):
        with pytest.raises(json.JSONDecodeError):
            _call(model)
    assert len(calls) == 1


def test_corrupt_retained_envelope_cannot_authorise_cached_copy(tmp_path, monkeypatch):
    model, service, calls = _model(tmp_path, monkeypatch)
    _call(model)
    with service._connection() as connection:
        identity, raw = connection.execute("SELECT envelope_id,record_json FROM model_work_envelopes").fetchone()
        record = json.loads(raw)
        record["canonical_digest"] = "sha256:" + "0" * 64
        connection.execute("UPDATE model_work_envelopes SET record_json=? WHERE envelope_id=?",
                           (json.dumps(record), identity))
    with pytest.raises(ModelUsageIntegrityError, match="envelope identity"):
        _call(model)
    assert len(calls) == 1


def test_unknown_usage_stays_held_and_existing_allocation_is_not_retried(tmp_path, monkeypatch):
    model, service, calls = _model(tmp_path, monkeypatch, invoke=lambda *a, **k:
        SimpleNamespace(text='{"text":"copy"}', usage={"usage_basis": "UNREPORTED"}))
    with pytest.raises(ModelUsageAdmissionError):
        _call(model)
    with pytest.raises(ModelUsageAdmissionError):
        _call(model)
    assert len(calls) == 1
    with service._connection() as connection:
        assert connection.execute("SELECT response_text FROM native_story_model_results").fetchone()[0] == '{"text":"copy"}'


def test_stop_fence_before_transport_settles_actual_zero_and_does_not_call_provider(tmp_path, monkeypatch):
    model, service, calls = _model(tmp_path, monkeypatch)
    def blocked():
        raise RuntimeError("owner stop")
    model.fence = blocked
    with pytest.raises(RuntimeError, match="owner stop"):
        _call(model)
    assert not calls
    with service._connection() as connection:
        raw = connection.execute("SELECT record_json FROM model_invocation_terminals").fetchone()[0]
    assert json.loads(raw)["pre_dispatch_zero_proved"] is True
