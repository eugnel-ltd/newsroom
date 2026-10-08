"""Paid semantic leaves retain precise caller scope and old route semantics."""
from dataclasses import asdict, replace
import pytest
from newsroom.authority.canonical import digest_bytes
from newsroom.control_plane.model_usage import InvocationEfficiencyPolicy, ModelUsageIntegrityError, ModelUsageService, WorkloadClass
from newsroom.tests.test_typesafe_judgment import _case
from newsroom.control_plane.typesafe_judgment import judgment_policy


def test_graphiti_trace_requires_exact_role_ingest_and_attempt(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch)as(engine,inputs,usage,calls,_):
        inputs.update(caller_identity='GRAPHITI_VERIFIER',candidate_id=None,hypothesis_digest=None,
                      ingest_id='ingest-1',graphiti_attempt_id='ingest-1:1')
        engine.evaluate(**inputs)
        assert not usage.has_committed_provider_dispatch(cycle_id='cycle-1')
        assert usage.has_committed_provider_dispatch(cycle_id='cycle-1',ingest_id='ingest-1',graphiti_attempt_id='ingest-1:1')
        assert not usage.has_committed_provider_dispatch(cycle_id='cycle-1',ingest_id='ingest-1',graphiti_attempt_id='ingest-1:2')
        assert len(calls)==1


def test_native_candidate_with_source_ingest_is_not_graphiti_dispatch(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch)as(engine,inputs,usage,_calls,_):
        inputs['ingest_id']='ingest-1'
        engine.evaluate(**inputs)
        assert not usage.has_committed_provider_dispatch(cycle_id='cycle-1',ingest_id='ingest-1',graphiti_attempt_id='ingest-1:1')


def test_paid_semantic_terminal_never_impersonates_subscription_cli(tmp_path,monkeypatch):
    with _case(tmp_path,monkeypatch)as(engine,inputs,usage,_calls,_):
        ref=engine.evaluate(**inputs);terminal=usage.terminal(ref.invocation_id)
        for wrong in (replace(terminal,subscription_cli_chat_not_cash_debited=True),replace(terminal,od_011_reference=None)):
            with pytest.raises(ModelUsageIntegrityError,match='paid usage linkage'):
                ModelUsageService._validate_terminal(wrong,WorkloadClass.TYPESAFE_JUDGMENT,engine.policy,requested_max_output_tokens=engine.policy.max_output_tokens)


@pytest.mark.parametrize('provider,route,allowed',[
    ('grok-build-cli','NATIVE_CLAIM_LOCALISATION',True),
    ('typesafe','NATIVE_CLAIM_LOCALISATION',False),
    ('grok-build-cli','NATIVE_CLAIM_LOCALISATION_OTHER',False),
])
def test_localisation_nullable_output_is_exact_route_and_backend(provider,route,allowed):
    values=asdict(judgment_policy(evidence_digest=digest_bytes(b'fixture'),qualified=True))
    values.update(workload_class=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,provider=provider,route=route,model='grok-4.7',max_output_tokens=None)
    if allowed:
        assert InvocationEfficiencyPolicy.create(**values).max_output_tokens is None
    else:
        with pytest.raises(ModelUsageIntegrityError,match='unbounded output'):
            InvocationEfficiencyPolicy.create(**values)


def test_paid_verifier_never_creates_native_zero_dispatch_credit(tmp_path,monkeypatch):
    from newsroom.control_plane.model_usage import native_graphiti_usage_cycle_id, GraphitiIngestRetryEvidence
    with _case(tmp_path,monkeypatch)as(engine,inputs,usage,_calls,_):
        inputs.update(caller_identity='GRAPHITI_VERIFIER',candidate_id=None,hypothesis_digest=None,
            ingest_id='ingest-1',graphiti_attempt_id='ingest-1:1',
            cycle_id=native_graphiti_usage_cycle_id(ingest_id='ingest-1',attempt_number=1))
        engine.evaluate(**inputs)
        zero = GraphitiIngestRetryEvidence((1,), (1,), (), None, ())
        monkeypatch.setattr(usage, '_graphiti_ingest_retry_evidence_batch',
                            lambda **_kw: {'ingest-1': (zero, 1)})
        evidence=usage.native_graphiti_ingest_retry_evidence_many(failed_attempts={'ingest-1':1},max_attempts=1)['ingest-1']
        assert evidence.zero_dispatch_attempts==()
        assert evidence.unresolved_attempts==(1,)


def test_generic_retry_batch_empty_and_normal_preserve_existing_contract(tmp_path):
    from newsroom.control_plane.model_usage import GraphitiIngestRetryEvidence
    usage = ModelUsageService(str(tmp_path/'usage.sqlite3'))
    assert usage.graphiti_ingest_retry_evidence_many(ingest_ids=()) == {}
    assert usage.graphiti_ingest_retry_evidence_many(ingest_ids=('existing-ingest',)) == {
        'existing-ingest': GraphitiIngestRetryEvidence((), (), (), None, ())
    }


def _settled_native_graphiti(usage, monkeypatch):
    import json
    from datetime import timedelta
    from newsroom.control_plane.graphiti import GraphitiModelUsageObserver
    from newsroom.control_plane.model_usage import WorkEnvelope, native_graphiti_usage_cycle_id
    from newsroom.graphiti_adapter.combined_temporal_contract import CONTRACT_NAME, SCHEMA
    from newsroom.tests.test_typesafe_judgment import NOW
    monkeypatch.setattr('newsroom.control_plane.graphiti._graphiti_implementation_identity', lambda: ('a'*40, True))
    cycle = native_graphiti_usage_cycle_id(ingest_id='ingest-1', attempt_number=1)
    envelope = WorkEnvelope.create(cycle_id=cycle, workload_class=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        admitted_at=NOW, admission_decision_id=None, candidate_id=None, hypothesis_digest=None,
        evidence_package_digest=None, ingest_id='ingest-1', graphiti_attempt_id='ingest-1:1')
    usage.open_envelope(envelope)
    observer = GraphitiModelUsageObserver(service=usage, envelope=envelope,
        clock=lambda: NOW+timedelta(seconds=1), owner_stop_check=lambda: None)
    token = observer.before_cli_invocation(provider='cursor-agent-cli', model='composer-2.5',
        prompt='fixture authenticated full source', schema=json.dumps(SCHEMA),
        semantic_request_class=CONTRACT_NAME, max_tokens=1000)
    observer.transport_dispatch_started(token)
    observer.after_cli_invocation(token, outcome='COMPLETE', usage={
        'usage_basis':'PROVIDER_REPORTED', 'input_tokens':10, 'output_tokens':2,
        'cached_read_tokens':0, 'cached_write_tokens':0, 'reasoning_tokens':0, 'total_tokens':12})
    usage.record_work_outcome(envelope_id=envelope.envelope_id, outcome='GRAPHITI_FAILED',
        outcome_record_id='fixture-pre-mutation-failure', payload_digest=None,
        terminal_at=NOW+timedelta(seconds=2))
    return cycle



def _retry_observer(usage, *, native=True, ingest='ingest-1', attempt=2, extra_bindings=None):
    from datetime import timedelta
    from newsroom.control_plane.graphiti import GraphitiModelUsageObserver
    from newsroom.control_plane.model_usage import WorkEnvelope, native_graphiti_usage_cycle_id
    from newsroom.tests.test_typesafe_judgment import NOW
    bindings = dict(admission_decision_id=None, candidate_id=None,
        hypothesis_digest=None, evidence_package_digest=None)
    bindings.update(extra_bindings or {})
    envelope = WorkEnvelope.create(
        cycle_id=native_graphiti_usage_cycle_id(ingest_id=ingest, attempt_number=attempt) if native else 'generic-retry-fixture',
        workload_class=WorkloadClass.GRAPHITI_CHAT_PRIMARY, admitted_at=NOW,
        ingest_id=ingest, graphiti_attempt_id=f'{ingest}:{attempt}', **bindings)
    envelope = usage.resume_or_open_graphiti_envelope(envelope)
    return GraphitiModelUsageObserver(service=usage, envelope=envelope,
        clock=lambda: NOW+timedelta(seconds=3), owner_stop_check=lambda: None,
        ingest_obligation_id=ingest, provider_attempt_number=attempt)

def test_reported_verifier_preserves_already_settled_whole_graphiti_attempt(tmp_path, monkeypatch):
    with _case(tmp_path, monkeypatch) as (engine, inputs, usage, calls, _):
        cycle = _settled_native_graphiti(usage, monkeypatch)
        before = usage.native_graphiti_ingest_retry_evidence_many(failed_attempts={'ingest-1':1}, max_attempts=1)['ingest-1']
        assert before.settled_provider_attempts == (1,) and before.unresolved_attempts == ()
        inputs.update(caller_identity='GRAPHITI_VERIFIER', candidate_id=None, hypothesis_digest=None,
            ingest_id='ingest-1', graphiti_attempt_id='ingest-1:1', cycle_id=cycle)
        reference = engine.evaluate(**inputs)
        assert engine.read(reference, **inputs)['answers']['support']['choice'] == 'yes'
        after = usage.native_graphiti_ingest_retry_evidence_many(failed_attempts={'ingest-1':1}, max_attempts=1)['ingest-1']
        assert after == before
        assert after.zero_dispatch_attempts == () and len(calls) == 1
        observer = _retry_observer(usage)
        assert observer.allows_fresh_completed_rollback_retry(
            episode_uuid='ingest-1', attempt_number=2, prior_attempt_number=1)
        assert not observer.allows_fresh_zero_dispatch_retry(episode_uuid='ingest-1', attempt_number=2)
        assert len(calls) == 1  # Constructor reads retained proof; no new provider work.


@pytest.mark.parametrize('corruption', ['unreported', 'missing-terminal', 'terminal-header', 'telemetry', 'context-role', 'context-route', 'missing-manifest'])
def test_unproved_paid_verifier_never_preserves_settlement(tmp_path, monkeypatch, corruption):
    import sqlite3
    import json
    from newsroom.control_plane.typesafe_judgment import TypesafeJudgmentError
    with _case(tmp_path, monkeypatch, transport_error=(TimeoutError('fixture') if corruption == 'unreported' else None)) as (engine, inputs, usage, calls, _):
        cycle = _settled_native_graphiti(usage, monkeypatch)
        inputs.update(caller_identity='GRAPHITI_VERIFIER', candidate_id=None, hypothesis_digest=None,
            ingest_id='ingest-1', graphiti_attempt_id='ingest-1:1', cycle_id=cycle)
        if corruption == 'unreported':
            with pytest.raises(TypesafeJudgmentError):
                engine.evaluate(**inputs)
        else:
            reference = engine.evaluate(**inputs)
            with sqlite3.connect(usage.path) as db:
                if corruption == 'missing-terminal':
                    db.execute('DELETE FROM model_invocation_terminals WHERE invocation_id=?', (reference.invocation_id,))
                elif corruption == 'terminal-header':
                    db.execute("UPDATE model_invocation_terminals SET outcome='FOREIGN' WHERE invocation_id=?", (reference.invocation_id,))
                elif corruption == 'telemetry':
                    db.execute('DELETE FROM model_provider_telemetry WHERE invocation_id=?', (reference.invocation_id,))
                elif corruption == 'missing-manifest':
                    db.execute("DELETE FROM model_invocation_context_manifests WHERE context_manifest_digest=(SELECT json_extract(record_json,'$.context_manifest_digest') FROM model_invocation_allocations WHERE invocation_id=?)", (reference.invocation_id,))
                else:
                    digest = db.execute('SELECT json_extract(record_json,\'$.context_manifest_digest\') FROM model_invocation_allocations WHERE invocation_id=?', (reference.invocation_id,)).fetchone()[0]
                    if corruption == 'context-route':
                        db.execute("UPDATE model_invocation_context_manifests SET route='FOREIGN' WHERE context_manifest_digest=?", (digest,))
                    else:
                        value = json.loads(db.execute('SELECT record_json FROM model_invocation_context_manifests WHERE context_manifest_digest=?', (digest,)).fetchone()[0])
                        value['caller_identity'] = 'NATIVE_ASSESSOR'
                        db.execute('UPDATE model_invocation_context_manifests SET record_json=? WHERE context_manifest_digest=?', (json.dumps(value), digest))
        if corruption in {'unreported', 'missing-terminal'}:
            evidence = usage.native_graphiti_ingest_retry_evidence_many(failed_attempts={'ingest-1':1}, max_attempts=1)['ingest-1']
            assert evidence.settled_provider_attempts == () and evidence.unresolved_attempts == (1,)
            assert evidence.zero_dispatch_attempts == ()
            observer = _retry_observer(usage)
            assert not observer.allows_fresh_completed_rollback_retry(
                episode_uuid='ingest-1', attempt_number=2, prior_attempt_number=1)
            assert not observer.allows_fresh_zero_dispatch_retry(episode_uuid='ingest-1', attempt_number=2)
        else:
            with pytest.raises(ModelUsageIntegrityError):
                _retry_observer(usage)
        assert len(calls) == 1


def test_verified_predispatch_zero_preserves_only_already_settled_graphiti(tmp_path, monkeypatch):
    from newsroom.control_plane.typesafe_judgment import TypesafeJudgmentError
    from newsroom.control_plane.model_usage import _is_exact_pre_dispatch_zero
    with _case(tmp_path, monkeypatch) as (engine, inputs, usage, calls, _):
        cycle = _settled_native_graphiti(usage, monkeypatch)
        before = usage.native_graphiti_ingest_retry_evidence_many(failed_attempts={'ingest-1':1}, max_attempts=1)['ingest-1']
        assert before.settled_provider_attempts == (1,)
        inputs.update(caller_identity='GRAPHITI_VERIFIER', candidate_id=None, hypothesis_digest=None,
            ingest_id='ingest-1', graphiti_attempt_id='ingest-1:1', cycle_id=cycle)
        engine.key = lambda: ''
        with pytest.raises(TypesafeJudgmentError):
            engine.evaluate(**inputs)
        import sqlite3
        with sqlite3.connect(usage.path) as db:
            invocation = db.execute('SELECT invocation_id FROM model_invocation_allocations WHERE workload_class=?', (WorkloadClass.TYPESAFE_JUDGMENT.value,)).fetchone()[0]
        assert _is_exact_pre_dispatch_zero(usage.terminal(invocation))
        after = usage.native_graphiti_ingest_retry_evidence_many(failed_attempts={'ingest-1':1}, max_attempts=1)['ingest-1']
        assert after == before and after.zero_dispatch_attempts == ()
        assert not calls


def _typesafe_profile_pair(tmp_path, **changes):
    service = ModelUsageService(str(tmp_path/'profiles.sqlite3'))
    old = judgment_policy(evidence_digest=digest_bytes(b'original qualification'), qualified=True)
    values = asdict(old)
    values.update(implementation_revision=digest_bytes(b'new qualified implementation'),
        evidence_digest=digest_bytes(b'new qualification'), **changes)
    new = InvocationEfficiencyPolicy.create(**values)
    service.register_policy(old);service.register_policy(new)
    return service,old,new


def _selected_typesafe(service, *, schema=True):
    from newsroom.control_plane.typesafe_judgment import SCHEMA_DIGEST
    return service.qualified_policy(workload_class=WorkloadClass.TYPESAFE_JUDGMENT,
        provider='typesafe',route='TYPESAFE_JUDGMENT',model='jev-latest',reasoning='none',
        output_schema_digest=SCHEMA_DIGEST if schema else None)


def test_latest_typesafe_profile_only_supersedes_compatible_software_audit_facts(tmp_path):
    import sqlite3
    service, old, new = _typesafe_profile_pair(tmp_path)
    assert _selected_typesafe(service) == new
    with sqlite3.connect(service.path) as db:
        assert db.execute('SELECT count(*) FROM model_invocation_policies').fetchone()[0] == 2
        original = db.execute('SELECT record_json FROM model_invocation_policies WHERE canonical_digest=?', (old.canonical_digest,)).fetchone()[0]
    import json
    assert json.loads(original) == old.as_record()


@pytest.mark.parametrize('changes', [
    {'max_prompt_bytes': 131071},
    {'command_flags': ('POST=/v1/systemone','RETRIES=1')},
    {'prompt_contract_version': 'different-purpose'},
    {'allowed_config_identities': ('different-config',)},
])
def test_incompatible_typesafe_profiles_remain_ambiguous(tmp_path, changes):
    from newsroom.control_plane.model_usage import ModelUsageAdmissionError
    service, _old, _new = _typesafe_profile_pair(tmp_path, **changes)
    with pytest.raises(ModelUsageAdmissionError, match='absent or ambiguous'):
        _selected_typesafe(service)


def test_typesafe_latest_selection_requires_explicit_schema(tmp_path):
    from newsroom.control_plane.model_usage import ModelUsageAdmissionError
    service, _old, _new = _typesafe_profile_pair(tmp_path)
    with pytest.raises(ModelUsageAdmissionError, match='absent or ambiguous'):
        _selected_typesafe(service, schema=False)


def test_unqualified_newest_never_qualifies_stale_current_implementation(tmp_path):
    from newsroom.control_plane.typesafe_judgment import TypesafeJudgment, TypesafeJudgmentError
    service, old, _new = _typesafe_profile_pair(tmp_path, qualified=False)
    assert _selected_typesafe(service) == old
    stale = InvocationEfficiencyPolicy.create(**{**asdict(old), 'implementation_revision':digest_bytes(b'stale software')})
    service.register_policy(stale)
    assert _selected_typesafe(service) == stale
    calls = []
    with pytest.raises(TypesafeJudgmentError, match='TYPESAFE_IMPLEMENTATION_HOLD'):
        TypesafeJudgment(usage=service,objects=None,policy=stale,
            api_key=lambda:calls.append('key'),source_fence=lambda *_:None,
            implementation_worktree_clean=True)
    assert not calls


@pytest.mark.parametrize('workload,provider,route,allowed',[
    (WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,'grok-build-cli','NATIVE_SOURCE_QUALIFICATION',True),
    (WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,'typesafe','NATIVE_SOURCE_QUALIFICATION',False),
    (WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,'grok-build-cli','NATIVE_SOURCE_QUALIFICATION_OTHER',False),
    (WorkloadClass.NATIVE_STORY_WRITER,'grok-build-cli','NATIVE_SOURCE_QUALIFICATION',False),
])
def test_source_qualification_nullable_output_is_exact_purpose(workload,provider,route,allowed):
    values=asdict(judgment_policy(evidence_digest=digest_bytes(b'purpose fixture'),qualified=True))
    values.update(workload_class=workload,provider=provider,route=route,
        model='grok-4.7',reasoning='high',max_output_tokens=None)
    if allowed:
        assert InvocationEfficiencyPolicy.create(**values).max_output_tokens is None
    else:
        with pytest.raises(ModelUsageIntegrityError,match='unbounded output'):
            InvocationEfficiencyPolicy.create(**values)


@pytest.mark.parametrize('corruption', ['work-outcome', 'internal-request', 'telemetry', 'terminal-header'])
def test_native_retry_observer_keeps_primary_accounting_required(tmp_path, monkeypatch, corruption):
    import sqlite3
    with _case(tmp_path, monkeypatch) as (engine, inputs, usage, calls, _):
        cycle = _settled_native_graphiti(usage, monkeypatch)
        inputs.update(caller_identity='GRAPHITI_VERIFIER', candidate_id=None, hypothesis_digest=None,
            ingest_id='ingest-1', graphiti_attempt_id='ingest-1:1', cycle_id=cycle)
        engine.evaluate(**inputs)
        with sqlite3.connect(usage.path) as db:
            invocation, envelope_id = db.execute('SELECT invocation_id,envelope_id FROM model_invocation_allocations '
                'WHERE workload_class=?', (WorkloadClass.GRAPHITI_CHAT_PRIMARY.value,)).fetchone()
            if corruption == 'work-outcome':
                db.execute('DELETE FROM model_work_outcomes WHERE envelope_id=?', (envelope_id,))
            elif corruption == 'internal-request':
                db.execute('DELETE FROM graphiti_internal_requests WHERE invocation_id=?', (invocation,))
            elif corruption == 'telemetry':
                db.execute('DELETE FROM model_provider_telemetry WHERE invocation_id=?', (invocation,))
            else:
                db.execute("UPDATE model_invocation_terminals SET outcome='FOREIGN' WHERE invocation_id=?", (invocation,))
        if corruption in {'work-outcome', 'internal-request'}:
            observer = _retry_observer(usage)
            assert not observer.allows_fresh_completed_rollback_retry(
                episode_uuid='ingest-1', attempt_number=2, prior_attempt_number=1)
            assert not observer.allows_fresh_zero_dispatch_retry(episode_uuid='ingest-1', attempt_number=2)
        else:
            with pytest.raises(ModelUsageIntegrityError):
                _retry_observer(usage)
        assert len(calls) == 1


def test_native_retry_observer_rejects_foreign_verifier_on_selected_cycle(tmp_path, monkeypatch):
    with _case(tmp_path, monkeypatch) as (engine, inputs, usage, calls, _):
        cycle = _settled_native_graphiti(usage, monkeypatch)
        inputs.update(caller_identity='GRAPHITI_VERIFIER', candidate_id=None, hypothesis_digest=None,
            ingest_id='foreign-ingest', graphiti_attempt_id='foreign-ingest:1', cycle_id=cycle)
        engine.evaluate(**inputs)
        with pytest.raises(ModelUsageIntegrityError, match='verifier binding'):
            _retry_observer(usage)
        assert len(calls) == 1


def test_non_native_observer_keeps_the_generic_reader_contract(tmp_path, monkeypatch):
    with _case(tmp_path, monkeypatch) as (engine, inputs, usage, calls, _):
        cycle = _settled_native_graphiti(usage, monkeypatch)
        inputs.update(caller_identity='GRAPHITI_VERIFIER', candidate_id=None, hypothesis_digest=None,
            ingest_id='ingest-1', graphiti_attempt_id='ingest-1:1', cycle_id=cycle)
        engine.evaluate(**inputs)
        with pytest.raises(ModelUsageIntegrityError, match='envelope binding'):
            _retry_observer(usage, native=False)
        assert len(calls) == 1


@pytest.mark.parametrize('field', ['admission_decision_id', 'candidate_id',
    'hypothesis_digest', 'evidence_package_digest'])
@pytest.mark.parametrize('value', ['foreign-binding', ''])
def test_extra_bound_native_cycle_keeps_generic_observer_contract(tmp_path, monkeypatch, field, value):
    usage = ModelUsageService(str(tmp_path/'usage.sqlite3'))
    calls = []
    def generic_reader(**inputs):
        calls.append(inputs)
        raise ModelUsageIntegrityError('generic extra-binding fixture')
    monkeypatch.setattr(usage, 'graphiti_ingest_retry_evidence', generic_reader)
    monkeypatch.setattr(usage, 'native_graphiti_ingest_retry_evidence_many',
        lambda **_inputs: pytest.fail('extra-bound envelope is not canonical native work'))
    with pytest.raises(ModelUsageIntegrityError, match='generic extra-binding fixture'):
        _retry_observer(usage, extra_bindings={field: value})
    assert calls == [{'ingest_id': 'ingest-1', 'before_attempt_number': 2}]


@pytest.mark.parametrize('changed_contract', [None, 'max_total_tokens', 'command_flags', 'allowed_context_identities'])
def test_localisation_software_upgrade_selects_latest_only_for_identical_contract(tmp_path, changed_contract):
    from newsroom.control_plane.native_claim_localisation import localisation_policy, ROUTE, MODEL, SCHEMA_DIGEST
    from newsroom.control_plane.model_usage import ModelUsageAdmissionError
    usage = ModelUsageService(str(tmp_path / 'localisation-policies.sqlite3'))
    old = localisation_policy(evidence_digest=digest_bytes(b'old proof'), qualified=True)
    values = {**asdict(old), 'implementation_revision': digest_bytes(b'new implementation'),
              'evidence_digest': digest_bytes(b'new proof')}
    if changed_contract == 'max_total_tokens':
        values[changed_contract] -= 1
    elif changed_contract == 'command_flags':
        values[changed_contract] = (*values[changed_contract], 'CHANGED_CONTRACT')
    elif changed_contract == 'allowed_context_identities':
        values[changed_contract] = (*values[changed_contract], 'CHANGED_CONTEXT')
    new = InvocationEfficiencyPolicy.create(**values)
    usage.register_policy(old)
    usage.register_policy(new)
    query = dict(workload_class=old.workload_class, provider='grok-build-cli', route=ROUTE,
                 model=MODEL, reasoning='high', output_schema_digest=SCHEMA_DIGEST)
    if changed_contract is None:
        assert usage.qualified_policy(**query) == new
    else:
        with pytest.raises(ModelUsageAdmissionError, match='absent or ambiguous'):
            usage.qualified_policy(**query)
