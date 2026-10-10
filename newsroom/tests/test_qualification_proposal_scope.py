"""A recorded verdict binds its proposal, not every future corrected proposal."""
import sqlite3

import pytest

from newsroom.control_plane.native_assessor import AutonomousNativeEvidenceAssessor
from newsroom.tests.test_qualification_semantic_witness import _selected_qualification_case


@pytest.mark.parametrize('next_outcome', (None, 'NO', 'timeout'))
def test_settled_fixed_proposal_no_allows_one_distinct_corrected_proposal(tmp_path, monkeypatch, next_outcome):
    with _selected_qualification_case(tmp_path, monkeypatch, fault='corrected',
            resolve_disagreements=True, resolution_fault=('NO', next_outcome)) as (
            post, witness, original, candidate, base, source, acquired, scope, proof, usage, qa, jev, render):
        with pytest.raises(ValueError, match='NOT_AFFIRMATIVE'):
            post.compose_selected(original, candidate, base, (source,), (acquired,), proof=proof)
        assert len(qa) == len(jev) == 2
        with sqlite3.connect(usage.path) as connection:
            before = dict(connection.execute('SELECT invocation_id,record_json FROM model_invocation_terminals'))
        old_record = original.decision_record
        post.resolution_contract = 'newsroom.qualification-semantic-resolution.v2'
        if next_outcome is None:
            selected = post.compose_selected(original, candidate, base, (source,), (acquired,), proof=proof)
            checked = AutonomousNativeEvidenceAssessor._validated_execution(selected.execution, candidate,
                base, (source,), (acquired,), semantic_witnesses=selected.semantic_witnesses,
                source_renderings=selected.source_renderings, semantic_witness_reader=witness.read)
            assert checked.governed_claims[0].claim.startswith('The authority has launched a public consultation')
            assert dict(checked.qualification_evidence[0].semantic_witness_ref)['contract'] == post.resolution_contract
        else:
            with pytest.raises((ValueError, TimeoutError)):
                post.compose_selected(original, candidate, base, (source,), (acquired,), proof=proof)
        assert len(qa) == 3 and len(jev) == 2 and not render
        assert original.decision_record == old_record
        for _ in range(2):
            if next_outcome is None:
                assert post.compose_selected(original, candidate, base, (source,), (acquired,), proof=proof) == selected
            else:
                with pytest.raises(ValueError):
                    post.compose_selected(original, candidate, base, (source,), (acquired,), proof=proof)
        with sqlite3.connect(usage.path) as connection:
            after = dict(connection.execute('SELECT invocation_id,record_json FROM model_invocation_terminals'))
        assert all(after[key] == value for key, value in before.items())
        assert len(qa) == 3 and len(jev) == 2 and not render


@pytest.mark.parametrize('mutation', ('none', 'already_checked', 'same_contract', 'unknown', 'admitted', 'missing_contract'))
def test_corrected_proposal_scheduler_requires_a_new_consumer_epoch(mutation):
    from types import SimpleNamespace
    from newsroom.control_plane.native_composition import ASSESSMENT_CONTRACT_VERSION
    from newsroom.control_plane.native_publication import NativePublicationContinuation
    from newsroom.control_plane.native_source_qualification_consumer import PROPOSAL_SCOPE_CONSUMER_VERSION

    old = ASSESSMENT_CONTRACT_VERSION.replace('+' + PROPOSAL_SCOPE_CONSUMER_VERSION, '')
    facts = {'reason': 'QUALIFICATION_RESOLUTION_NOT_AFFIRMATIVE_HOLD', 'failure_class': None,
        'assessment_contract_version': old, 'candidate_id': 'candidate', 'candidate_version_id': 'version',
        'graphiti_receipts': ['receipt'], 'intake_receipt_id': 'intake'}
    current = ASSESSMENT_CONTRACT_VERSION
    if mutation == 'already_checked': facts['retained_qualification_checked_contract'] = current
    elif mutation == 'same_contract': facts['assessment_contract_version'] = current
    elif mutation == 'unknown': facts['failure_class'] = 'TimeoutError'
    elif mutation == 'admitted': facts['package_admission_id'] = 'package'
    elif mutation == 'missing_contract': current = old
    caller = SimpleNamespace(_assessment_contract_version=current)
    assert NativePublicationContinuation.qualification_resolution_due(caller, facts) is (mutation == 'none')


def test_composed_retained_reader_enters_distinct_proposal_not_legacy_dispatch(tmp_path, monkeypatch):
    import ast
    from contextlib import nullcontext
    from pathlib import Path
    from newsroom.control_plane import native_composition
    from newsroom.tests.test_native_composition import _composed_cached_origin_adapter

    with _selected_qualification_case(tmp_path, monkeypatch, fault='corrected',
            resolve_disagreements=True, resolution_fault=('NO', None)) as (
            post, witness, original, candidate, base, source, acquired, scope, proof, usage, qa, jev, render):
        with pytest.raises(ValueError, match='NOT_AFFIRMATIVE'):
            post.compose_selected(original, candidate, base, (source,), (acquired,), proof=proof)
        post.resolution_contract = 'newsroom.qualification-semantic-resolution.v2'
        tree = ast.parse(Path(native_composition.__file__).read_text())
        reader = next(node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == 'read_qualified_source')
        namespace = {**vars(native_composition), 'qualifier': post.qualifier,
            'qualification_consumer': post, 'proof': proof, 'judgment_scope': lambda *_: scope}
        exec(compile(ast.Module(body=[reader], type_ignores=[]), '<composed-qualified-reader>', 'exec'), namespace)

        def bounded(c, b, sources, results, **mode):
            assert mode['cached_only'] and mode['qualification_cached_only']
            selected = namespace['read_qualified_source'](c, b, sources, results)
            return AutonomousNativeEvidenceAssessor._validated_execution(selected.execution, c, b,
                sources, results, semantic_witnesses=selected.semantic_witnesses,
                source_renderings=selected.source_renderings, semantic_witness_reader=witness.read)

        bounded.assess_with_boundary = bounded
        adapter = _composed_cached_origin_adapter(bounded, qualifier=post.qualifier, proof=proof,
            stop_fence=nullcontext, stop_check=lambda: None, journal=object(), _judgment_scope=lambda *_: scope)
        for _ in range(2):
            result = adapter.assess(candidate, base, (source,), (acquired,), cached_only=True)
            assert result.governed_claims[0].claim.startswith('The authority has launched')
        assert len(qa) == 3 and len(jev) == 2 and not render
