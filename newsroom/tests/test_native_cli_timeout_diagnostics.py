"""Native timeouts retain diagnostics, never accept partial provider output."""

from contextlib import nullcontext
from pathlib import Path
import json
import sqlite3
import subprocess
import sys

import pytest

from newsroom.control_plane import writer, native_assessor

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.graphiti_adapter.cli_process import validated_timeout_diagnostics
from newsroom.increment10.evidence import _base_package
from newsroom.tests.assessor_fixture_support import candidate_fixture
from newsroom.tests.test_increment10_editorial import _ready_package
from newsroom.tests.test_native_assessor import _usage

_FINAL = json.dumps({'type': 'turn_completed', 'structured_output': {'package': {}}}) + '\n'
_SECRET = 'private-token request-body /private/credential/path'


@pytest.mark.parametrize(('stdout', 'stderr', 'progress'), (
    (None, None, 'NO_OUTPUT_OBSERVED'),
    (b'', b'', 'NO_OUTPUT_OBSERVED'),
    (_FINAL.encode(), b'', 'OUTPUT_OBSERVED'),
    (b'{malformed\xff', _SECRET.encode(), 'OUTPUT_OBSERVED'),
    ('partial 中文', _SECRET, 'OUTPUT_OBSERVED'),
    (42, b'', 'UNOBSERVED'),
))
def test_wrapper_preserves_type_deadline_and_strict_secret_free_evidence(monkeypatch, stdout, stderr, progress):
    calls = []
    def timeout(command, **arguments):
        calls.append(arguments)
        raise subprocess.TimeoutExpired(command, 300, output=stdout, stderr=stderr)
    monkeypatch.setattr(writer.subprocess, 'run', timeout)
    with pytest.raises(writer.CliTimeoutError, match='grok writer timed out') as caught:
        writer._run(('grok', '--prompt-file', '/private/request'), timeout=300)
    evidence = caught.value.evidence
    assert validated_timeout_diagnostics([evidence]) == [evidence]
    assert evidence['phase'] == 'CLI_TRANSPORT'
    assert evidence['cause'] == 'CONFIGURED_TIMEOUT_EXPIRED'
    assert evidence['configured_timeout_ms'] == 300000
    assert evidence['provider_cause'] == 'UNOBSERVED'
    assert evidence['termination'] == 'UNOBSERVED'
    assert evidence['last_progress'] == progress
    assert calls[0]['timeout'] == 300
    assert calls[0]['capture_output'] is True
    assert caught.value.diagnostic_reference is None
    if progress != 'UNOBSERVED':
        def encoded(value):
            return b'' if value is None else value if isinstance(value, bytes) else value.encode()
        assert evidence['stdout_bytes'] == len(encoded(stdout))
        assert evidence['stderr_bytes'] == len(encoded(stderr))
        assert evidence['stdout_digest'] == digest_bytes(encoded(stdout))
        assert evidence['stderr_digest'] == digest_bytes(encoded(stderr))
    assert _SECRET not in json.dumps(evidence)
    assert '/private/request' not in json.dumps(evidence)
    assert 'structured_output' not in json.dumps(evidence)
    assert len(canonical_json_bytes(evidence)) < 2048


def _assessor_fixture(tmp_path, monkeypatch, *, stdout, stderr):
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(writer, '_minimal_grok_auth_bytes', lambda: b'{}')
    monkeypatch.setattr(writer, '_prove_grok_hermetic_capabilities', lambda _: None)
    def timeout(command, **arguments):
        calls.append(command)
        assert arguments['timeout'] == 300
        raise subprocess.TimeoutExpired(command, 300, output=stdout, stderr=stderr)
    monkeypatch.setattr(writer.subprocess, 'run', timeout)
    assessor = native_assessor.AutonomousNativeEvidenceAssessor(
        native_assessor._dispatch_grok, usage=usage, dispatch_fence=nullcontext,
    )
    return connection, candidate, base, service, usage, calls, assessor


@pytest.mark.parametrize(('stdout', 'stderr'), (
    (_FINAL.encode(), b''),
    (None, None),
    (b'{malformed\xff', _SECRET.encode()),
    ('partial 中文', _SECRET),
))
def test_actual_assessor_failure_handler_retains_exact_invocation_and_reader(tmp_path, monkeypatch, stdout, stderr):
    connection, candidate, base, service, usage, calls, assessor = _assessor_fixture(
        tmp_path, monkeypatch, stdout=stdout, stderr=stderr,
    )
    read_calls = []
    original_read = usage.read_transport_diagnostic
    def read_back(allocation, reference):
        read_calls.append(dict(reference))
        return original_read(allocation, reference)
    monkeypatch.setattr(usage, 'read_transport_diagnostic', read_back)
    try:
        with pytest.raises(writer.CliTimeoutError) as caught:
            assessor(candidate, base, (), ())
        reference = caught.value.diagnostic_reference
        assert read_calls == [reference]  # Production failure handler authenticates its read-back.
        assert set(reference) == {'seq', 'payload_digest'}
        with sqlite3.connect(service.path) as retained:
            row = retained.execute('SELECT record_json FROM model_invocation_allocations').fetchone()
            allocation = native_assessor._allocation_from_record(json.loads(row[0]))
            terminal = json.loads(retained.execute('SELECT record_json FROM model_invocation_terminals').fetchone()[0])
            assert terminal['outcome'] == 'ASSESSOR_PROVIDER_FAILED'
            assert terminal['failure_class'] == 'UNKNOWN_PROVIDER_FAILURE'
            assert terminal['usage_status'] == 'ESTIMATED'
            assert terminal['pre_dispatch_zero_proved'] is False
            assert terminal['dispatch_at'] is not None
            assert retained.execute("SELECT count(*) FROM ledger WHERE kind='NATIVE_ASSESSOR_TRANSPORT_DIAGNOSTIC'").fetchone()[0] == 1
            assert retained.execute("SELECT count(*) FROM ledger WHERE kind IN ('NATIVE_ASSESSMENT_RESULT','NATIVE_ASSESSMENT_MATERIALISATION')").fetchone()[0] == 0
            raw = retained.execute('SELECT payload_json FROM ledger WHERE seq=?', (reference['seq'],)).fetchone()[0]
            assert len(raw.encode()) < 2048
            assert _SECRET not in raw and 'structured_output' not in raw
        record = usage.read_transport_diagnostic(allocation, reference)
        assert record['invocation_id'] == allocation.invocation_id
        assert record['allocation_digest'] == allocation.canonical_digest
        assert record['request_digest'] == allocation.request_digest
        assert record['diagnostic'] == caught.value.evidence
        assert record['diagnostic']['provider_cause'] == 'UNOBSERVED'
        assert record['diagnostic']['last_progress'] != 'COMPLETE'
        # A fresh usage object proves retained bytes; no in-memory diagnostic cache.
        reopened = native_assessor.NativeAssessmentUsage(service, usage._policy)
        assert reopened.read_transport_diagnostic(allocation, reference) == record
        assert len(calls) == 1
        with pytest.raises(native_assessor.NativeEvidenceHold):
            assessor(candidate, base, (), ())
        assert len(calls) == 1  # Unknown/estimated timeout never earns a retry.
        with pytest.raises(native_assessor.NativeEvidenceError):
            usage.read_transport_diagnostic(allocation, {**reference, 'payload_digest': 'sha256:' + 'f' * 64})
    finally:
        connection.close()


def test_emitted_terminal_but_process_no_exit_is_capture_only(monkeypatch):
    assert json.loads(writer._parse_grok_writer_output(_FINAL).text) == {'package': {}}
    original = writer.subprocess.run
    def local_only(command, **arguments):
        assert command[0] == sys.executable
        return original(command, **arguments)
    monkeypatch.setattr(writer.subprocess, 'run', local_only)
    program = 'import sys,time;sys.stdout.write(' + repr(_FINAL) + ');sys.stdout.flush();time.sleep(2)'
    with pytest.raises(writer.CliTimeoutError) as caught:
        writer._run((sys.executable, '-c', program), timeout=0.25)
    evidence = caught.value.evidence
    assert evidence['stdout_bytes'] == len(_FINAL.encode())
    assert evidence['stdout_digest'] == digest_bytes(_FINAL.encode())
    assert evidence['last_progress'] == 'OUTPUT_OBSERVED'
    assert evidence['elapsed_ms'] >= 250
    assert evidence['provider_cause'] == 'UNOBSERVED'


def test_invalid_diagnostic_is_rejected_and_optional_legacy_constructor_survives():
    assert writer.CliTimeoutError('legacy timeout').evidence is None
    with pytest.raises(ValueError):
        writer.CliTimeoutError('timeout', evidence={'raw_output': _SECRET})


def test_diagnostic_storage_failure_does_not_mask_timeout_or_change_accounting(tmp_path, monkeypatch, caplog):
    connection, candidate, base, service, usage, calls, assessor = _assessor_fixture(
        tmp_path, monkeypatch, stdout=None, stderr=None,
    )
    def fail(*_):
        raise sqlite3.OperationalError(_SECRET)
    monkeypatch.setattr(usage, 'retain_transport_diagnostic', fail)
    try:
        with pytest.raises(writer.CliTimeoutError) as caught:
            assessor(candidate, base, (), ())
        assert caught.value.diagnostic_reference is None
        assert _SECRET not in caplog.text
        with sqlite3.connect(service.path) as retained:
            assert retained.execute('SELECT outcome,usage_status FROM model_invocation_terminals').fetchone() == ('ASSESSOR_PROVIDER_FAILED', 'ESTIMATED')
        assert len(calls) == 1
    finally:
        connection.close()


@pytest.mark.parametrize('fault', ('invocation', 'request', 'allocation', 'raw_field', 'provider_cause', 'naive_time', 'wrong_seq', 'bool_seq'))
def test_retained_reader_rejects_rebound_or_unsafe_diagnostics(tmp_path, monkeypatch, fault):
    connection, candidate, base, service, usage, _calls, assessor = _assessor_fixture(
        tmp_path, monkeypatch, stdout=None, stderr=None,
    )
    try:
        with pytest.raises(writer.CliTimeoutError) as caught:
            assessor(candidate, base, (), ())
        reference = dict(caught.value.diagnostic_reference)
        with sqlite3.connect(service.path) as retained:
            allocation = native_assessor._allocation_from_record(json.loads(retained.execute(
                'SELECT record_json FROM model_invocation_allocations').fetchone()[0]))
            record = json.loads(retained.execute('SELECT payload_json FROM ledger WHERE seq=?', (reference['seq'],)).fetchone()[0])
            if fault == 'wrong_seq':
                reference['seq'] += 100
            elif fault == 'bool_seq':
                reference['seq'] = True
            else:
                if fault in ('invocation', 'request', 'allocation'):
                    field = {'invocation': 'invocation_id', 'request': 'request_digest', 'allocation': 'allocation_digest'}[fault]
                    record[field] = 'sha256:' + 'f' * 64
                elif fault == 'raw_field':
                    record['diagnostic']['raw_output'] = _SECRET
                elif fault == 'provider_cause':
                    record['diagnostic']['provider_cause'] = 'PROVIDER_GENERATION_FAILED'
                else:
                    record['observed_at'] = '2026-01-01T12:00:00'
                raw = canonical_json_bytes(record)
                reference['payload_digest'] = digest_bytes(raw)
                retained.execute('UPDATE ledger SET payload_json=?,payload_digest=? WHERE seq=?',
                                 (raw.decode(), reference['payload_digest'], reference['seq']))
        with pytest.raises(native_assessor.NativeEvidenceError):
            usage.read_transport_diagnostic(allocation, reference)
    finally:
        connection.close()


@pytest.mark.parametrize('fault', ('rebound_body', 'terminal_digest', 'outcome', 'usage_status', 'failure_class', 'completed_at'))
def test_timeout_diagnostic_subject_rejects_terminal_body_or_index_rebinding(tmp_path, monkeypatch, fault):
    from dataclasses import fields
    connection, candidate, base, service, usage, _calls, assessor = _assessor_fixture(
        tmp_path, monkeypatch, stdout=None, stderr=None,
    )
    try:
        with pytest.raises(writer.CliTimeoutError) as caught:
            assessor(candidate, base, (), ())
        reference = dict(caught.value.diagnostic_reference)
        with sqlite3.connect(service.path) as retained:
            allocation = native_assessor._allocation_from_record(json.loads(retained.execute(
                'SELECT record_json FROM model_invocation_allocations').fetchone()[0]))
            body = json.loads(retained.execute('SELECT record_json FROM model_invocation_terminals').fetchone()[0])
            if fault == 'rebound_body':
                original = native_assessor._terminal_from_record(body)
                values = {field.name: getattr(original, field.name) for field in fields(original)}
                values['invocation_id'] = 'sha256:' + 'f' * 64
                foreign = type(original).create(**values)
                retained.execute('UPDATE model_invocation_terminals SET record_json=? WHERE invocation_id=?',
                                 (canonical_json_bytes(foreign.as_record()).decode(), allocation.invocation_id))
            else:
                value = {
                    'terminal_digest': 'sha256:' + 'f' * 64,
                    'outcome': 'ASSESSOR_ACCEPTED', 'usage_status': 'REPORTED',
                    'failure_class': 'CHANGED_FAILURE', 'completed_at': '2026-01-01T12:00:00Z',
                }[fault]
                retained.execute(f'UPDATE model_invocation_terminals SET {fault}=? WHERE invocation_id=?',
                                 (value, allocation.invocation_id))
        # Pre-fix, retention may accept the rebound subject; its fresh read-back
        # must still deny it. Post-fix, both write and read deny the same binding.
        try:
            reference = usage.retain_transport_diagnostic(allocation, caught.value.evidence)
        except native_assessor.NativeEvidenceError:
            pass
        with pytest.raises(native_assessor.NativeEvidenceError):
            usage.read_transport_diagnostic(allocation, reference)
        with pytest.raises(native_assessor.NativeEvidenceError):
            usage.retain_transport_diagnostic(allocation, caught.value.evidence)
    finally:
        connection.close()


def test_terminal_binding_failure_in_capture_preserves_original_cli_timeout(tmp_path, monkeypatch, caplog):
    connection, candidate, base, service, usage, calls, assessor = _assessor_fixture(
        tmp_path, monkeypatch, stdout=None, stderr=None,
    )
    original_complete = usage.complete
    def rebound_after_completion(*arguments, **keywords):
        original_complete(*arguments, **keywords)
        with sqlite3.connect(service.path) as retained:
            retained.execute("UPDATE model_invocation_terminals SET outcome='ASSESSOR_ACCEPTED'")
    monkeypatch.setattr(usage, 'complete', rebound_after_completion)
    try:
        with pytest.raises(writer.CliTimeoutError, match='grok writer timed out') as caught:
            assessor(candidate, base, (), ())
        assert caught.value.diagnostic_reference is None
        assert len(calls) == 1
        assert 'native timeout diagnostic was not retained' in caplog.text
        with sqlite3.connect(service.path) as retained:
            assert retained.execute("SELECT count(*) FROM ledger WHERE kind='NATIVE_ASSESSOR_TRANSPORT_DIAGNOSTIC'").fetchone()[0] == 0
    finally:
        connection.close()


def test_timeout_emits_bounded_framing_observation_without_output_or_authority(monkeypatch, caplog):
    import logging
    output = '\n'.join(json.dumps(value) for value in (
        {'type': 'thought', 'data': 'private 中文'},
        {'params': {'update': {'sessionUpdate': 'agent_thought_chunk', 'content': {'text': 'more'}}}},
        {'type': 'text', 'data': _SECRET},
        {'update': {'sessionUpdate': 'agent_message_chunk', 'content': {'text': '字'}}},
        {'type': 'usage', 'usage': {'total_tokens': 900}},
        {'type': 'end', 'stopReason': 'end_turn'},
        {'type': _SECRET, 'data': _SECRET},
    )).encode()
    def timeout(command, **_arguments):
        raise subprocess.TimeoutExpired(command, 300, output=output, stderr=b'')
    monkeypatch.setattr(writer.subprocess, 'run', timeout)
    with caplog.at_level(logging.INFO, logger='newsroom.diagnostic'):
        with pytest.raises(writer.CliTimeoutError) as caught:
            writer._run(('grok',), timeout=300)
    records = [record for record in caplog.records if getattr(record, 'diagnostic_event', None) == 'CLI_TIMEOUT_STREAM_OBSERVATION']
    assert len(records) == 1
    data = records[0].diagnostic_data
    assert data['thought'] == [2, len('private 中文more'.encode())]
    assert data['text'] == [2, len((_SECRET + '字').encode())]
    assert data['usage'] == data['end'] == 1
    assert data['end_seen'] is True and data['stop'] == 'end_turn'
    assert data['other'] == 1 and data['invalid'] == 0 and data['truncated'] is False
    assert len(json.dumps(data, separators=(',', ':')).encode()) <= 256
    assert _SECRET not in json.dumps(data) and 'private 中文' not in json.dumps(data)
    assert validated_timeout_diagnostics([caught.value.evidence]) == [caught.value.evidence]
    assert 'end_seen' not in caught.value.evidence
    assert caught.value.diagnostic_reference is None


@pytest.mark.parametrize('output,invalid,truncated,end_seen,stop', (
    (b'{"type":"text","data":"ok"}\n{"type":"thought","data":"\xff"}\n{"type":"end"', 2, False, False, 'UNOBSERVED'),
    (b'{"type":"thought","data":"\\ud800"}\n', 1, False, False, 'UNOBSERVED'),
    (b'{"type":"end","stopReason":"private-token"}\n', 0, False, True, 'UNOBSERVED'),
    ((b'{"type":"text","data":"x"}\n' * 6000) + _FINAL.encode(), 0, True, False, 'UNOBSERVED'),
    (b'{"type":"thought","data":"' + b'x' * 100_000 + b'"}\n', 0, True, False, 'UNOBSERVED'),
    (b'{"type":"turn_completed","stopReason":"cancelled"}\n', 0, False, True, 'cancelled'),
))
def test_timeout_framing_partial_invalid_and_scan_limits_never_imply_complete_output(
        monkeypatch, output, invalid, truncated, end_seen, stop):
    observed = []
    monkeypatch.setattr(writer, 'emit_diagnostic', lambda event, data: observed.append((event, data)))
    def timeout(command, **_arguments):
        raise subprocess.TimeoutExpired(command, 300, output=output, stderr=b'')
    monkeypatch.setattr(writer.subprocess, 'run', timeout)
    with pytest.raises(writer.CliTimeoutError) as caught:
        writer._run(('grok',), timeout=300)
    assert len(observed) == 1
    data = observed[0][1]
    assert data['invalid'] == invalid and data['truncated'] is truncated
    assert data['end_seen'] is end_seen and data['stop'] == stop
    assert len(json.dumps(data, separators=(',', ':')).encode()) <= 256
    assert _SECRET not in json.dumps(data) and 'private-token' not in json.dumps(data)
    assert caught.value.evidence['provider_cause'] == caught.value.evidence['termination'] == 'UNOBSERVED'
    assert caught.value.diagnostic_reference is None


@pytest.mark.parametrize('fault', ['sink', 'parser'])
def test_optional_framing_observation_failure_preserves_original_timeout(monkeypatch, fault):
    def fail(*_args, **_kwargs): raise RuntimeError(_SECRET)
    monkeypatch.setattr(writer, 'emit_diagnostic' if fault == 'sink' else '_grok_writer_update', fail)
    def timeout(command, **arguments):
        assert arguments['timeout'] == 300
        raise subprocess.TimeoutExpired(command, 300, output=_FINAL.encode(), stderr=b'')
    monkeypatch.setattr(writer.subprocess, 'run', timeout)
    with pytest.raises(writer.CliTimeoutError, match='grok writer timed out') as caught:
        writer._run(('grok',), timeout=300)
    assert _SECRET not in str(caught.value)
    assert caught.value.evidence['cause'] == 'CONFIGURED_TIMEOUT_EXPIRED'
    assert validated_timeout_diagnostics([caught.value.evidence]) == [caught.value.evidence]
