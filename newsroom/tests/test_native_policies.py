from dataclasses import replace

import pytest

from newsroom.authority.policy import PayloadSchemaValidationError
from newsroom.control_plane.native_policies import native_policy_components


def test_private_policy_bindings_are_derived_and_have_no_source_or_public_grant(tmp_path):
    target = tmp_path / "serving.sqlite3"
    args = dict(principal_id="newsroom.hermes", authority_domain="newsroom.authority", target_path=target, target_id="hermes-private-serving")
    policies = native_policy_components(**args)
    same = native_policy_components(**args)
    assert policies.publication == same.publication
    assert not target.exists()
    assert len(policies.registry.definitions()) == 7
    assert len(policies.admission_registry.definitions()) == 15
    context = policies.registry.resolve("retrieval.native_context.admit")
    assert context.aggregate_type == "native_retrieval_context"
    assert context.required_scope == "authority.retrieval.context"
    assert not any("fixture" in definition.definition_version for definition in policies.registry.definitions())
    assert policies.required_scopes.isdisjoint({"authority.objects.manage", "authority.sources.manage", "authority.graphiti.execute"})
    hyd = {contract.purpose: contract for contract in policies.hydration_policies.contracts()}
    assert hyd["evidence.source"].allowed_uses == frozenset({"publication_evidence"})
    assert hyd["evidence.source"].allowed_principal_ids == frozenset({"newsroom.hermes"})
    assert hyd["native-source-intake-read"].allowed_object_classes == frozenset({"source.native-observation"})
    assert hyd["native-source-intake-read"].allowed_uses == frozenset({"native-source-parsing"})
    assert "proposal.extraction" not in hyd["evidence.source"].allowed_uses
    moved = native_policy_components(**{**args, "target_path": tmp_path / "other.sqlite3"})
    assert moved.publication.target_context_digest != policies.publication.target_context_digest
    assert moved.publication.target_policy_digest == policies.publication.target_policy_digest
    renamed = native_policy_components(**{**args, "target_id": "other-private-serving"})
    assert renamed.publication.target_policy_digest != policies.publication.target_policy_digest
    contract = policies.payload_schemas.contracts()[0]
    with pytest.raises(PayloadSchemaValidationError):
        contract.canonicalize({"status": "PASS"})


def test_native_factory_keeps_exact_v15_reader_pair_not_an_unknown_policy(tmp_path):
    policies = native_policy_components(principal_id='newsroom.control-plane',
        authority_domain='newsroom.authority', target_path=tmp_path / 'serving.sqlite3',
        target_id='hermes-private-serving')
    old_editorial, old_authorisation = policies.publication.retained_policy_pairs[0]
    assert old_editorial == 'sha256:e576722159e5a126d60a95e230d667c9da3c83405818c7a3d4584eaba9bef6d5'
    assert old_authorisation == 'sha256:4d7b98fab375ea20763a907e49a4bbe037f56b5020697e2391ffd3661bdcdd79'
    assert old_editorial != policies.publication.editorial_policy_bundle_digest
    assert old_authorisation != policies.publication.publication_authorisation_policy_digest
