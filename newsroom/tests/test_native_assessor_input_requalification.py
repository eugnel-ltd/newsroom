"""A corrected input ceiling is not an accounting waiver or provider retry."""
import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from newsroom.tests.assessor_fixture_support import candidate_fixture

from newsroom.authority.canonical import digest_bytes, digest_canonical
from newsroom.control_plane.model_usage import (
    InvocationAllocation, InvocationEfficiencyPolicy, ModelUsageAdmissionError,
    ModelUsageIntegrityError, ModelUsageService,
)
from newsroom.control_plane.native_assessor import (
    NativeAssessmentExecution, NativeAssessmentUsage, native_assessment_input_bound,
)
from newsroom.tests.test_native_assessor import _usage, _candidate, _ready_package, _base_package


def _changed_policy(policy, **changes):
    values = asdict(policy)
    values.pop('canonical_digest')
    return InvocationEfficiencyPolicy.create(**(values | changes))


def _fixture(tmp_path, monkeypatch):
    # These are immutable v15 historical records, even after producer upgrades.
    import newsroom.control_plane.native_assessor as native
    monkeypatch.setattr(native, 'VERSION', native._V15_PRODUCER_VERSION)
    monkeypatch.setattr(native, 'SYSTEM', native._V15_SYSTEM)
    monkeypatch.setattr(native, 'PROVIDER_SCHEMA_DIGEST', native._V15_SCHEMA_DIGEST)
    monkeypatch.setattr(
        native, 'CONTEXT_MANIFEST_SCHEMA_VERSION',
        'newsroom.native-evidence-assessor.context-manifest.v1',
    )
    monkeypatch.setattr(native, 'REASONING', native.CONT_PRIMARY_REASONING)
    monkeypatch.setattr(native, 'COMMAND_FLAGS', native.CONT_PRIMARY_COMMAND_FLAGS)
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)
    assert usage._policy.model == 'grok-4.6'
    assert usage._policy.reasoning == 'low'
    assert usage._policy.max_output_tokens == 10_000
    assert native_assessment_input_bound(usage._policy)['version'].endswith('.v1')
    allocation = usage.begin(candidate, base, 'small request')
    old = _changed_policy(usage._policy, context_manifest_schema_version='newsroom.native-evidence-assessor.context-manifest.v1')
    service.register_policy(old)
    # Build a predecessor v1 allocation exactly as the old, permissive gate did.
    with sqlite3.connect(service.path) as c:
        manifest = json.loads(c.execute('SELECT record_json FROM model_invocation_context_manifests WHERE context_manifest_digest=?', (allocation.context_manifest_digest,)).fetchone()[0])
        manifest.pop('context_manifest_digest')
        manifest.pop('input_bound')
        manifest.update(schema_version=old.context_manifest_schema_version, prompt_bytes=397_776, prompt_digest=digest_bytes(b'x' * 397_776))
        manifest['request_digest'] = digest_canonical({key: manifest[key] for key in ('provider','route','model','reasoning','command_semantic_version','command_flags','implementation_revision','system_digest','prompt_digest','output_schema_digest')})
        manifest['context_manifest_digest'] = digest_canonical(manifest)
        c.execute('DELETE FROM model_usage_current WHERE invocation_id=?', (allocation.invocation_id,))
        from newsroom.control_plane.model_usage_current import _set_inventory
        _set_inventory(c, 0)
        c.execute('DELETE FROM model_invocation_allocations WHERE invocation_id=?', (allocation.invocation_id,))
    service.retain_context_manifest(manifest)
    values = asdict(allocation)
    for key in ('invocation_id','canonical_digest'):
        values.pop(key)
    values.update(invocation_policy_digest=old.canonical_digest, prompt_bytes=397_776, prompt_digest=manifest['prompt_digest'], request_digest=manifest['request_digest'], context_manifest_digest=manifest['context_manifest_digest'])
    allocation = InvocationAllocation.create(**values)
    with service._connection() as c:
        service._insert_allocation(c, allocation)
    usage._policy = old
    dispatch_at = usage.mark_dispatch(allocation)
    execution = NativeAssessmentExecution('{"package":{}}', {
        'usage_basis':'PROVIDER_REPORTED','input_tokens':229_208,'output_tokens':3_379,
        'cached_read_tokens':128,'cached_write_tokens':0,'reasoning_tokens':1_192,
        'context_tokens':229_336,'total_tokens':232_715,
    })
    usage.retain_result(allocation, execution, dispatch_at=dispatch_at)
    usage.complete(allocation, outcome='ASSESSOR_VALIDATION_FAILED', execution=execution, provider_dispatched=True, dispatch_at=dispatch_at, failure_class='ASSESSMENT_VALIDATION_FAILED')
    new = _changed_policy(
        old,
        version='bounded-input-v1',
        context_manifest_schema_version=(
            'newsroom.native-evidence-assessor.context-manifest.v2'
        ),
        max_prompt_bytes=native_assessment_input_bound(old)['max_request_bytes'],
    )
    service.register_policy(new)
    # #1053's corrected v15 writer used manifest v2; the failed predecessor
    # above remains an authenticated v1 allocation.
    monkeypatch.setattr(
        native, 'CONTEXT_MANIFEST_SCHEMA_VERSION',
        'newsroom.native-evidence-assessor.context-manifest.v2',
    )
    current = NativeAssessmentUsage(service, new, clock=lambda: datetime(2026,9,8,tzinfo=UTC)+timedelta(minutes=1))
    connection.close()
    return service, current, candidate, base, allocation, new


def _recover(service, allocation, policy):
    return service.requalify_native_assessor_input_bound(invocation_id=allocation.invocation_id, qualified_policy_digest=policy.canonical_digest, recorded_at=datetime(2026,9,8,tzinfo=UTC)+timedelta(seconds=30))


def test_corrected_input_bound_releases_only_future_work_and_preserves_history(tmp_path, monkeypatch):
    service, usage, candidate, base, failed, policy = _fixture(tmp_path, monkeypatch)
    with sqlite3.connect(service.path) as c:
        original = c.execute('SELECT record_json FROM model_invocation_terminals').fetchall()
        assert service._route_state(c, policy.route)['state'] == 'OPEN'
    digest = _recover(service, failed, policy)
    assert _recover(service, failed, policy) == digest
    reopened = ModelUsageService(service.path)
    with sqlite3.connect(service.path) as c:
        assert c.execute('SELECT record_json FROM model_invocation_terminals').fetchall() == original
        assert reopened._route_state(c, policy.route)['state'] == 'CLOSED'
        assert c.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0] == 1
    # The failed candidate remains unresolved/held, never offered a fresh call.
    assert usage.retained_assessments(candidate, base) is None
    assert usage.retained_pre_dispatch_failure(candidate) is None
    fresh = SimpleNamespace(candidate_id='fresh-candidate',version_id='fresh-version',governing_manifest=candidate.governing_manifest)
    assert usage.begin(fresh, base, 'x' * 26_051).invocation_id != failed.invocation_id


@pytest.mark.parametrize('change', ({'max_total_tokens':232_716, 'hard_estimate_ceiling_tokens':232_716}, {'max_context_tokens':230_000}, {'max_prompt_bytes':1_000_000}))
def test_requalification_rejects_lifted_or_uncorrected_bounds(tmp_path, monkeypatch, change):
    service, _usage_, _candidate_, _base_, allocation, policy = _fixture(tmp_path, monkeypatch)
    invalid = _changed_policy(policy, version='invalid', **change)
    service.register_policy(invalid)
    with pytest.raises(ModelUsageAdmissionError):
        _recover(service, allocation, invalid)


def test_settled_requalification_history_stays_off_route_admission_but_replay_rechecks(tmp_path, monkeypatch):
    service, _usage_, _candidate_, _base_, allocation, policy = _fixture(tmp_path, monkeypatch)
    _recover(service, allocation, policy)
    with sqlite3.connect(service.path) as c:
        row = c.execute("SELECT seq,payload_json FROM ledger WHERE kind='NATIVE_ASSESSOR_INPUT_REQUALIFICATION'").fetchone()
        record = json.loads(row[1]);record['candidate_id'] = 'different'
        raw = json.dumps(record,sort_keys=True,separators=(',',':'))
        c.execute('UPDATE ledger SET payload_json=?,payload_digest=? WHERE seq=?',(raw,digest_bytes(raw.encode()),row[0]))
    with sqlite3.connect(service.path) as c:
        assert service._route_state(c, policy.route)['state'] == 'CLOSED'
    with pytest.raises(ModelUsageIntegrityError):
        _recover(service, allocation, policy)


@pytest.mark.parametrize('unreported', (False, True))
def test_another_active_or_unknown_leaf_prevents_requalification(tmp_path, monkeypatch, unreported):
    from newsroom.control_plane.model_usage import WorkEnvelope, WorkloadClass, InvocationTerminal, UsageStatus, UsageComponents
    service, _usage_, candidate, base, allocation, policy = _fixture(tmp_path, monkeypatch)
    at = allocation.allocated_at
    envelope = WorkEnvelope.create(cycle_id=digest_bytes(b'other-cycle'), workload_class=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR, admitted_at=at, admission_decision_id=None, candidate_id='other', hypothesis_digest=candidate.governing_manifest.canonical_digest, evidence_package_digest=base.digest, ingest_id=None, graphiti_attempt_id=None)
    service.open_envelope(envelope)
    values = asdict(allocation)
    for key in ('invocation_id', 'canonical_digest'):
        values.pop(key)
    values.update(envelope_id=envelope.envelope_id, cycle_id=envelope.cycle_id)
    other = InvocationAllocation.create(**values)
    with service._connection() as c:
        service._insert_allocation(c, other)
    if unreported:
        service.complete(InvocationTerminal.create(invocation_id=other.invocation_id, outcome='ASSESSOR_PROVIDER_FAILED', failure_class='TIMEOUT', usage_status=UsageStatus.UNREPORTED, components=UsageComponents(), dispatch_at=at, completed_at=at, observed_at=at, subscription_cli_chat_not_cash_debited=True))
        # Even if the last OPEN event is the corrected failure, older unknown
        # usage remains an independent blocker and the transaction rolls back.
        with service._connection() as c:
            service._append_route_state(c, route=policy.route, state='OPEN', reason='MAX_TOTAL_TOKENS_EXCEEDED', invocation_id=allocation.invocation_id, recorded_at=at+timedelta(seconds=1))
    with pytest.raises(ModelUsageAdmissionError):
        _recover(service, allocation, policy)
    with sqlite3.connect(service.path) as c:
        assert service._route_state(c, policy.route)['state'] == 'OPEN'
        assert c.execute("SELECT count(*) FROM ledger WHERE kind='NATIVE_ASSESSOR_INPUT_REQUALIFICATION'").fetchone()[0] == 0


def test_requalification_authenticates_retained_reported_components(tmp_path, monkeypatch):
    service, _usage_, _candidate_, _base_, allocation, policy = _fixture(tmp_path, monkeypatch)
    with sqlite3.connect(service.path) as c:
        c.execute("UPDATE model_provider_telemetry SET record_json=json_set(record_json,'$.provider_telemetry.total_tokens',1) WHERE invocation_id=?", (allocation.invocation_id,))
    with pytest.raises(ModelUsageIntegrityError):
        _recover(service, allocation, policy)


def test_assessor_requalification_reads_do_not_scan_unrelated_ledger(tmp_path):
    from newsroom.control_plane.store import connect, append_ledger
    path = str(tmp_path / 'indexed.sqlite3')
    connect(path).close()
    service = ModelUsageService(path)
    with sqlite3.connect(path) as c:
        queries = (
            ('NATIVE_ASSESSOR_INPUT_REQUALIFICATION', 'SELECT payload_digest,payload_json FROM ledger WHERE kind=?', ('NATIVE_ASSESSOR_INPUT_REQUALIFICATION',)),
            ('NATIVE_ASSESSMENT_RESULT', "SELECT payload_digest,payload_json FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT' AND json_extract(payload_json,'$.invocation_id')=?", ('invocation',)),
        )
        before = []
        for _kind, query, values in queries:
            plan = ' '.join(str(row[3]) for row in c.execute('EXPLAIN QUERY PLAN '+query, values))
            assert 'SEARCH ledger USING INDEX model_usage_' in plan
            steps = [0]
            def count():
                steps[0] += 1
                return 0
            c.set_progress_handler(count, 1)
            assert c.execute(query, values).fetchall() == []
            c.set_progress_handler(None, 0)
            before.append(steps[0])
        for i in range(400):
            append_ledger(c, 'UNRELATED_TEST_EVENT', {'i':i})
        for index, (_kind, query, values) in enumerate(queries):
            steps = [0]
            c.set_progress_handler(count, 1)
            assert c.execute(query, values).fetchall() == []
            c.set_progress_handler(None, 0)
            assert steps[0] <= before[index]


@pytest.mark.parametrize('substitute_candidate', (False, True))
def test_requalification_binds_the_selected_envelope_row(tmp_path, monkeypatch, substitute_candidate):
    from newsroom.control_plane.model_usage import WorkEnvelope
    service, _usage_, _candidate_, _base_, allocation, policy = _fixture(tmp_path, monkeypatch)
    with sqlite3.connect(service.path) as c:
        if substitute_candidate:
            record = json.loads(c.execute('SELECT record_json FROM model_work_envelopes WHERE envelope_id=?', (allocation.envelope_id,)).fetchone()[0])
            from newsroom.control_plane.model_usage import _envelope_from_record
            values = asdict(_envelope_from_record(record))
            values.pop('envelope_id');values.pop('canonical_digest');values['candidate_id'] = 'unrelated-candidate'
            changed = WorkEnvelope.create(**values)
            from newsroom.authority.canonical import canonical_json_bytes
            c.execute('UPDATE model_work_envelopes SET record_json=? WHERE envelope_id=?', (canonical_json_bytes(changed.as_record()).decode(),allocation.envelope_id))
        else:
            c.execute('UPDATE model_work_envelopes SET canonical_digest=? WHERE envelope_id=?', (digest_bytes(b'wrong'), allocation.envelope_id))
    with pytest.raises(ModelUsageIntegrityError):
        _recover(service, allocation, policy)


def test_requalification_binds_context_scalar_columns(tmp_path, monkeypatch):
    service, _usage_, _candidate_, _base_, allocation, policy = _fixture(tmp_path, monkeypatch)
    with sqlite3.connect(service.path) as c:
        c.execute("UPDATE model_invocation_context_manifests SET provider='other-provider' WHERE context_manifest_digest=?", (allocation.context_manifest_digest,))
    with pytest.raises(ModelUsageIntegrityError):
        _recover(service, allocation, policy)


@pytest.mark.parametrize(('field', 'future'), (('VERSION', 'newsroom.native-evidence-assessor.v16'), ('SYSTEM', 'Changed future producer instructions'), ('SCHEMA_DIGEST', 'sha256:'+'f'*64)))
def test_historical_requalification_survives_current_producer_change(tmp_path, monkeypatch, field, future):
    import newsroom.control_plane.native_assessor as native
    service, _usage_, _candidate_, _base_, allocation, policy = _fixture(tmp_path, monkeypatch)
    _recover(service, allocation, policy)
    monkeypatch.setattr(native, field, future)
    with sqlite3.connect(service.path) as c:
        assert service._route_state(c, policy.route)['state'] == 'CLOSED'


@pytest.mark.parametrize('field', ('system_digest', 'schema_digest'))
def test_native_admission_binds_actual_context_to_input_bound(tmp_path, monkeypatch, field):
    service, usage = _usage(tmp_path, monkeypatch)
    connection, _port, candidate = candidate_fixture(tmp_path)
    try:
        base = _base_package(_ready_package(candidate)[1])
        allocation = usage.begin(candidate, base, 'initial request')
        with sqlite3.connect(service.path) as c:
            manifest = json.loads(c.execute('SELECT record_json FROM model_invocation_context_manifests WHERE context_manifest_digest=?',(allocation.context_manifest_digest,)).fetchone()[0])
        manifest.pop('context_manifest_digest')
        manifest[field] = digest_bytes(b'different contract content')
        manifest['prompt_digest'] = digest_bytes(b'different input')
        manifest['request_digest'] = digest_canonical({key:manifest[key] for key in ('provider','route','model','reasoning','command_semantic_version','command_flags','implementation_revision','system_digest','prompt_digest','output_schema_digest')})
        manifest['context_manifest_digest'] = digest_canonical(manifest)
        service.retain_context_manifest(manifest)
        values = asdict(allocation)
        values.pop('canonical_digest');values.pop('invocation_id')
        values.update(prompt_digest=manifest['prompt_digest'],request_digest=manifest['request_digest'],context_manifest_digest=manifest['context_manifest_digest'])
        changed = InvocationAllocation.create(**values)
        with sqlite3.connect(service.path) as c, pytest.raises(ModelUsageAdmissionError,match='complete input exceeds qualified bound'):
            service._validate_preflight(c,changed,usage._policy)
    finally:
        connection.close()
