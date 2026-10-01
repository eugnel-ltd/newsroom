"""One fresh global authority read per atomic allocation preflight."""
from datetime import timedelta

import pytest

from newsroom.control_plane import model_usage
from newsroom.control_plane.model_usage import ModelUsageAdmissionError, ModelUsageIntegrityError, ModelUsageService, WorkloadClass
from newsroom.tests.test_model_usage_receipts import T0, _allocation, _envelope, _policy


def _prepared(tmp_path):
    service = ModelUsageService(str(tmp_path / 'usage.sqlite3'))
    workload = WorkloadClass.GRAPHITI_CHAT_PRIMARY
    policy = _policy(workload=workload, route='GRAPHITI_CHAT_PRIMARY')
    envelope = _envelope(workload=workload, candidate_id=None, ingest_id='ingest-fixture')
    service.register_policy(policy)
    service.open_envelope(envelope)
    return service, envelope, policy


def test_atomic_preflight_reads_global_blockers_once_and_refreshes_next_allocation(tmp_path, monkeypatch):
    service, envelope, policy = _prepared(tmp_path)
    original = model_usage._usage_blocking_routes
    calls = []

    def observe(connection):
        assert connection.in_transaction
        calls.append(connection)
        return original(connection)

    monkeypatch.setattr(model_usage, '_usage_blocking_routes', observe)
    service.allocate(_allocation(envelope, policy), owner_emergency_stop=False)
    assert len(calls) == 1
    service.allocate(_allocation(envelope, policy, leaf_ordinal=2, request='request-2'), owner_emergency_stop=False)
    assert len(calls) == 2
    assert calls[0] is not calls[1]


def test_next_allocation_observes_new_open_route(tmp_path, monkeypatch):
    service, envelope, policy = _prepared(tmp_path)
    service.allocate(_allocation(envelope, policy), owner_emergency_stop=False)
    service.open_route_circuit(route=policy.route, reason='SYSTEMIC_TRANSPORT', invocation_id=None,
                               recorded_at=T0+timedelta(seconds=2))
    original = model_usage._usage_blocking_routes
    calls = []

    def observe(connection):
        calls.append(connection)
        return original(connection)

    monkeypatch.setattr(model_usage, '_usage_blocking_routes', observe)
    with pytest.raises(ModelUsageAdmissionError, match='route circuit is open'):
        service.allocate(_allocation(envelope, policy, leaf_ordinal=2, request='request-2'), owner_emergency_stop=False)
    assert len(calls) == 1


@pytest.mark.parametrize('fault', ('unresolved', 'corrupt_global'))
def test_global_unresolved_and_corrupt_evidence_still_deny_before_allocation(tmp_path, monkeypatch, fault):
    service, envelope, policy = _prepared(tmp_path)

    def denied(connection):
        assert connection.in_transaction
        if fault == 'corrupt_global':
            raise ModelUsageIntegrityError('retained unrelated usage differs')
        return {policy.route}

    monkeypatch.setattr(model_usage, '_usage_blocking_routes', denied)
    with pytest.raises((ModelUsageAdmissionError, ModelUsageIntegrityError)):
        service.allocate(_allocation(envelope, policy), owner_emergency_stop=False)
    connection = service._connection()
    try:
        assert connection.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0] == 0
    finally:
        connection.close()
