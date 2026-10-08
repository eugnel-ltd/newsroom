"""Future typed rendering retains original paid work identities and source state."""
from copy import deepcopy
import sqlite3
import pytest

from newsroom.authority.canonical import digest_bytes
from newsroom.control_plane import native_claim_localisation as m
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.tests.test_native_claim_localisation import case, NOW


def test_typed_policy_does_not_change_v3_producer_bytes():
    policy = m.localisation_policy(evidence_digest=digest_bytes(b'typed fixture'), qualified=True)
    assert policy.prompt_contract_version == m.TYPED_VERSION
    assert policy.output_schema_digest == m.TYPED_SCHEMA_DIGEST
    assert m._contract(m.ALIGNED_VERSION) == (m.ALIGNED_SCHEMA, m.ALIGNED_SYSTEM)
    assert digest_bytes(m.ALIGNED_SYSTEM.encode()) == 'sha256:172e3151ed3f1e9b1302742cf4216566e8152c2a80960a25827a532fefce237d'
    assert m.ALIGNED_SCHEMA_DIGEST == 'sha256:c27751696c2852389921e9aae30c1e16b225056cca001a9c531236b785fda73f'


def test_v4_input_projects_prompt_without_rewriting_original_snapshot(tmp_path, monkeypatch):
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch)
    original = deepcopy(state)
    projected = deepcopy(state)
    projected['claims']['S1L1']['projection_marker'] = 'fixture literal projection'
    monkeypatch.setattr(m, '_project_state', lambda value, version: projected if version == m.TYPED_VERSION else value)
    with open_native_runtime(**args) as runtime:
        localiser = m.NativeClaimLocaliser(usage=usage, objects=runtime.authority.objects,
            policy=m.localisation_policy(evidence_digest=digest_bytes(b'v4'), qualified=True),
            source_fence=fence, runner=runner, implementation_worktree_clean=True, clock=lambda: NOW)
        scope = dict(candidate_id='candidate-1', hypothesis_digest=digest_bytes(b'hypothesis'),
                     evidence_package_digest=digest_bytes(b'package'), proof=runtime.proof)
        ref = localiser.localise(state, **scope)
        read = localiser.read_localisation(ref, state, **scope)
        assert state == original
        assert 'fixture literal projection' in calls[0]
        assert read['original_state'] == original
        assert read['projected_state'] == projected
        assert read['original_claims'] == original['claims']
        assert localiser.localise(state, **scope) == ref
        assert len(calls) == 1


def test_v3_reported_content_failure_is_not_v4_retry_credit(tmp_path, monkeypatch):
    bad = {'renderings': [{'span_id': 'S1L1',
        'rendered_assertion_zh_hant_hk_fragments': ['Untranslated text.'],
        'factual_localisations': [], 'quotation_source_keys': []}]}
    args, usage, state, runner, fence, calls = case(tmp_path, monkeypatch, payload=bad)
    with open_native_runtime(**args) as runtime:
        scope = dict(candidate_id='candidate-1', hypothesis_digest=digest_bytes(b'hypothesis'),
                     evidence_package_digest=digest_bytes(b'package'), proof=runtime.proof)
        def localiser(version):
            return m.NativeClaimLocaliser(usage=usage, objects=runtime.authority.objects,
                policy=m.localisation_policy(evidence_digest=digest_bytes(version.encode()), qualified=True, version=version),
                source_fence=fence, runner=runner, implementation_worktree_clean=True, clock=lambda: NOW)
        with pytest.raises(m.LocalisationHold):
            localiser(m.ALIGNED_VERSION).localise(state, **scope)
        with sqlite3.connect(usage.path) as connection:
            before = connection.execute('SELECT * FROM model_invocation_terminals').fetchall()
        with pytest.raises(m.LocalisationHold):
            localiser(m.TYPED_VERSION).localise(state, **scope)
        with sqlite3.connect(usage.path) as connection:
            assert connection.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0] == 1
            assert connection.execute('SELECT * FROM model_invocation_terminals').fetchall() == before
        assert len(calls) == 1
