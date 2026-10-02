from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from newsroom.authority import ObjectLimits, StaticAuthenticator, StaticAuthorizer, StaticPrincipal
from newsroom.authority._hermes_native_system import open_hermes_native_authority_system
from newsroom.authority._event_hypothesis_relationship_system import (
    _create_event_hypothesis_relationship_read_port,
)
from newsroom.authority.persistence import EventReadPolicy, MetadataClass
from newsroom.authority.persistence import AuthorityWriterBusy
from newsroom.authority.types import TrustScope
from newsroom.increment4 import increment4_admitted_contract_registry
from newsroom.increment6.collision import CurrentCollisionEffectEnforcer, TrustedCurrentCollisionAuthorityBoundary
from newsroom.increment6.candidates import CandidateContractError
from newsroom.increment6.relationships import RelationshipContractError
from newsroom.increment6.work_items import (
    DecisionLeadBinding,
    RetrievalContextAuthority,
    RetrievalInputBinding,
    TriageWorkItem,
    WorkItemContractError,
    TriageWorkItemStore,
)
from newsroom.tests.authority_a2b_helpers import _policy_registries
from newsroom.tests.authority_event_helpers import payload_schemas, registry_v1
from newsroom.tests.authority_helpers import FIXED_NOW, proof
from newsroom.tests.check_3c_authority_helpers import check_read_policy, source_read_policy
from newsroom.tests.discovery_3d_authority_helpers import (
    discovery_read_policy,
    exact_admission_request,
    scopes as discovery_scopes,
    seed_check_lineage,
)
from newsroom.tests.editorial_relation_4c_helpers import relation_read_policy
from newsroom.tests.entity_4b_helpers import entity_read_policy
from newsroom.tests.extraction_4a_helpers import extraction_read_policy
from newsroom.tests.graphiti_adapter_4d_authority_helpers import graphiti_read_policy
from newsroom.tests.increment4e_helpers import increment4_projection_read_policy
from newsroom.tests.increment5b2_helpers import config
from newsroom.tests.projection_b2_helpers import MemoryNeo4jAdapter
from newsroom.tests.test_increment6a2_work_items import _version
from newsroom.tests import test_increment5d1_hybrid_composer as composer_helpers
from newsroom.tests import test_increment5d2_retrieval_context as retrieval_helpers


def _retrieval_authority(path: Path):
    path.mkdir()
    inputs = composer_helpers.branch_inputs.__wrapped__(path)
    builder, _, _, _, _, request, receipt, _ = retrieval_helpers._retained_complete_context(
        path, inputs, name="hermes-native"
    )
    return (
        RetrievalContextAuthority(
            builder.journal.path, {request.request_digest: (request, receipt)}
        ),
        RetrievalInputBinding.from_receipt(request, receipt),
    )


def test_hermes_composition_opens_one_writer_and_all_native_facades(
    tmp_path: Path, monkeypatch,
) -> None:
    adapter = MemoryNeo4jAdapter()
    monkeypatch.setattr(
        "newsroom.authority._graphiti_increment4_system._open_structural_graph_adapter",
        lambda _: adapter,
    )
    rights, hydration, admissions = _policy_registries()
    authenticator = StaticAuthenticator(
        credentials={"token-1": StaticPrincipal("principal.alpha")},
        authority_domain="newsroom.authority",
    )
    authorizer = StaticAuthorizer(
        policy_version="hermes-native-test-v1",
        grants_by_principal={
            "principal.alpha": discovery_scopes() | frozenset({"authority.fixture.events.read"})
        },
    )
    collision = CurrentCollisionEffectEnforcer(
        current_authority_provider=lambda _: None,
        trusted_boundary=TrustedCurrentCollisionAuthorityBoundary(
            "fixture-scope", "fixture-profile", "sha256:" + "a" * 64,
            "sha256:" + "b" * 64, "fixture-port",
        ),
    )
    event_policy = EventReadPolicy(
        policy_id="hermes-native-events-v1", purpose="hermes.native",
        required_scope="authority.fixture.events.read",
        allowed_principal_ids=frozenset({"principal.alpha"}),
        allowed_security_scopes=frozenset({"authority.source_registry"}),
        allowed_trust_scopes=frozenset({TrustScope.ADMITTED}),
        metadata_classes=frozenset({MetadataClass.ROUTING}),
    )
    retrieval, retrieval_binding = _retrieval_authority(tmp_path / "retrieval")
    work_item_constructions: list[sqlite3.Connection] = []
    context_authentications: list[str] = []
    original_work_item_init = TriageWorkItemStore.__init__
    original_context_require = RetrievalContextAuthority.verify_retained_integrity

    def counted_work_item_init(self, connection, retrieval_authority=None, **kwargs):
        work_item_constructions.append(connection)
        original_work_item_init(self, connection, retrieval_authority, **kwargs)

    def counted_context_require(self, connection, binding):
        context_authentications.append(binding.context_id)
        return original_context_require(self, connection, binding)

    monkeypatch.setattr(TriageWorkItemStore, "__init__", counted_work_item_init)
    monkeypatch.setattr(
        RetrievalContextAuthority, "verify_retained_integrity", counted_context_require
    )
    kwargs = dict(
        path=tmp_path / "authority.sqlite3", object_root=tmp_path / "objects",
        workspace_root=tmp_path.resolve(), registry=registry_v1(),
        payload_schemas=payload_schemas(), admission_registry=admissions,
        rights_policies=rights, hydration_policies=hydration,
        contracts=increment4_admitted_contract_registry(), authenticator=authenticator,
        authorizer=authorizer, event_read_policy=event_policy,
        source_read_policy=source_read_policy(), check_read_policy=check_read_policy(),
        discovery_read_policy=discovery_read_policy(),
        extraction_read_policy=extraction_read_policy(), entity_read_policy=entity_read_policy(),
        relation_read_policy=relation_read_policy(), graphiti_read_policy=graphiti_read_policy(),
        projection_read_policy=increment4_projection_read_policy(),
        object_limits=ObjectLimits(
            global_max_bytes=1024 * 1024, class_max_bytes={"source_capture": 1024 * 1024},
            max_read_bytes=1024 * 1024, min_free_bytes=0, io_chunk_bytes=64,
            max_staging_bytes=1024 * 1024, max_range_bytes=1024 * 1024,
        ),
        neo4j_config=config(), retrieval_authority=retrieval,
        collision_enforcer=collision, clock=lambda: FIXED_NOW,
    )
    system = open_hermes_native_authority_system(**kwargs)
    try:
        seed_check_lineage(system)
        admitted = system.discovery.admit_signal_to_lead(
            exact_admission_request(), proof=proof()
        )
        decision = DecisionLeadBinding.from_authority(
            admitted.lead, admitted.initial_disposition
        )
        item = TriageWorkItem.create((decision,))
        system.work_items.create_or_replay(
            item, replace(_version(item), retrieval=retrieval_binding)
        )
        assert system.authority_store_path == str((tmp_path / "authority.sqlite3").resolve())
        assert system.sources and system.checks and system.discovery
        assert system.work_items and system.executions and system.dispositions
        assert system.hypotheses and system.relationships and system.lineage
        assert system.candidates and system.candidate_read_port
        assert system.commands and system.events and system.collision is collision
        missing = "00000000-0000-4000-8000-000000000001"
        assert system.candidate_versions((missing, missing)) == (None, None)
        with pytest.raises(CandidateContractError, match="unknown Candidate Version"):
            system.candidate_read_port.require_retained_version(
                "00000000-0000-4000-8000-000000000001"
            )
        assert system.work_items.current_version(item.work_item_id) is not None
        hypothesis_store = (
            system.hypotheses._EventHypothesisAuthority__authority._authority
        )
        other_authenticator = StaticAuthenticator(
            credentials={"token-2": StaticPrincipal("principal.beta")},
            authority_domain="newsroom.authority",
        )
        with pytest.raises(
            RelationshipContractError,
            match="read-port factory collaborators differ",
        ):
            _create_event_hypothesis_relationship_read_port(
                hypothesis_store._connection,
                retrieval_authority=retrieval,
                authenticator=other_authenticator,
                command_registry=registry_v1(),
                payload_schemas=payload_schemas(),
                hypotheses=hypothesis_store,
            )
        with sqlite3.connect(
            tmp_path / "authority.sqlite3", isolation_level=None
        ) as other_connection:
            other_connection.row_factory = sqlite3.Row
            other_connection.execute("PRAGMA foreign_keys=ON")
            other_connection.execute("PRAGMA journal_mode=WAL")
            other_connection.execute("PRAGMA synchronous=FULL")
            with pytest.raises(
                RelationshipContractError,
                match="read-port factory collaborators differ",
            ):
                _create_event_hypothesis_relationship_read_port(
                    other_connection,
                    retrieval_authority=retrieval,
                    authenticator=authenticator,
                    command_registry=registry_v1(),
                    payload_schemas=payload_schemas(),
                    hypotheses=hypothesis_store,
                )
        monkeypatch.setattr(
            "newsroom.authority._graphiti_increment4_system._open_structural_graph_adapter",
            lambda _: MemoryNeo4jAdapter(),
        )
        with pytest.raises(AuthorityWriterBusy):
            open_hermes_native_authority_system(**kwargs)
    finally:
        system.close()

    work_item_constructions.clear()
    context_authentications.clear()
    monkeypatch.setattr(
        "newsroom.authority._graphiti_increment4_system._open_structural_graph_adapter",
        lambda _: MemoryNeo4jAdapter(),
    )
    reopened = open_hermes_native_authority_system(**kwargs)
    try:
        assert len(work_item_constructions) == 1
        # Native construction does not authenticate unused Work Item history.
        # The current read proves its selected immutable lineage below.
        assert context_authentications == []
        with sqlite3.connect(tmp_path / "authority.sqlite3") as connection:
            trigger_name = "immutable_triage_work_item_versions_update"
            trigger_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name=?", (trigger_name,)
            ).fetchone()[0]
            connection.execute(f"DROP TRIGGER {trigger_name}")
            connection.execute(
                "UPDATE triage_work_item_versions SET canonical_bytes=?",
                (b'{"tampered":true}',),
            )
            connection.execute(trigger_sql)
        with pytest.raises(WorkItemContractError, match="fields are not exact"):
            reopened.work_items.current_version(item.work_item_id)
    finally:
        reopened.close()

    monkeypatch.setattr(
        "newsroom.authority._graphiti_increment4_system._open_structural_graph_adapter",
        lambda _: MemoryNeo4jAdapter(),
    )
    with open_hermes_native_authority_system(**kwargs) as current:
        with pytest.raises(WorkItemContractError, match="fields are not exact"):
            current.work_items.current_version(item.work_item_id)
