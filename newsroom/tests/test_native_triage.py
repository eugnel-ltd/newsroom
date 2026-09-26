from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from dataclasses import replace

import pytest

from newsroom.authority import (
    ObjectLimits,
    StaticAuthenticator,
    StaticAuthorizer,
    StaticPrincipal,
)
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority._hermes_native_system import open_hermes_native_authority_system
from newsroom.authority.persistence import (
    AuthoritySchemaError,
    EventReadPolicy,
    MetadataClass,
)
from newsroom.authority.types import TrustScope
from newsroom.discovery import (
    DecisionTerminality,
    GateDecisionId,
    GateOutcome,
    LeadDispositionDecisionId,
    NextAction,
    NextActionKind,
    TimeValidity,
)
from newsroom.control_plane.native_triage import (
    _admit_native_triage_candidate,
    advance_native_triage,
    build_native_triage_work,
    plan_native_schedule,
)
from newsroom.increment6.outcomes import PriorityLane, PrioritySelection
from newsroom.increment4 import increment4_admitted_contract_registry
from newsroom.increment6.collision import (
    CandidateUseCollisionBinding,
    CandidateUseOperation,
    CurrentCollisionAuthoritySnapshot,
    CurrentCollisionEffectEnforcer,
    CurrentCollisionEligibilityRequest,
    CurrentCollisionReceiptEvidence,
    TrustedCurrentCollisionAuthorityBoundary,
)
from newsroom.increment6.candidates import (
    CandidateAdmissionRequest,
    candidate_command_definition,
)
from newsroom.increment6.relationships import relationship_command_definition
from newsroom.increment6.work_items import RetrievalContextAuthority, RetrievalInputBinding
from newsroom.tests import test_increment6a2_work_items as work_item_helpers
from newsroom.tests import test_increment5d1_hybrid_composer as composer_helpers
from newsroom.tests import test_increment5d2_retrieval_context as retrieval_helpers
from newsroom.tests import test_increment6e1_collision as collision_helpers
from newsroom.tests.authority_a2b_helpers import _policy_registries
from newsroom.tests.authority_event_helpers import payload_schemas, registry_v1
from newsroom.tests.authority_helpers import FIXED_NOW
from newsroom.tests.discovery_3d_authority_helpers import (
    discovery_read_policy,
    exact_admission_request,
    exact_gate_request,
    open_discovery_system,
    proof,
    scopes as discovery_scopes,
    seed_check_lineage,
)
from newsroom.tests.discovery_3d_helpers import disposition_request, reason
from newsroom.tests.check_3c_authority_helpers import check_read_policy, source_read_policy
from newsroom.tests.editorial_relation_4c_helpers import relation_read_policy
from newsroom.tests.entity_4b_helpers import entity_read_policy
from newsroom.tests.extraction_4a_helpers import extraction_read_policy
from newsroom.tests.graphiti_adapter_4d_authority_helpers import graphiti_read_policy
from newsroom.tests.increment4e_helpers import increment4_projection_read_policy
from newsroom.tests.increment5b2_helpers import config
from newsroom.tests.projection_b2_helpers import MemoryNeo4jAdapter
from newsroom.tests.test_increment5c2_named_tool_authority_execution import (
    QUERY_VALID,
    SERVING,
    authority_database,
    authorize,
    collision_request,
    executor,
)


def _admitted(tmp_path):
    with open_discovery_system(tmp_path / "authority.sqlite3") as system:
        seed_check_lineage(system)
        result = system.discovery.admit_signal_to_lead(
            exact_admission_request(), proof=proof()
        )
        assert result.lead is not None
        assert result.initial_disposition is not None
        return result.lead, result.initial_disposition


def _no_match_retrieval(tmp_path):
    branch_inputs = composer_helpers.branch_inputs.__wrapped__(tmp_path)
    inputs = retrieval_helpers._no_match_inputs(branch_inputs)
    composer, composition_request, composition = retrieval_helpers._compose(
        tmp_path,
        inputs,
        key="native-triage-no-match",
        request_id="00000000-0000-4000-8000-000000006001",
    )
    database, _ = retrieval_helpers._authority_database(
        tmp_path,
        name="native-triage-no-match",
        admission_id="object:native-triage-no-match",
        passage_id=None,
        content=None,
    )
    cas_root = retrieval_helpers._cas_root(tmp_path, name="native-triage-no-match")
    authority_request, authority_result = retrieval_helpers._authority_execution(
        tmp_path,
        name="native-triage-no-match",
        database=database,
        composition=composition,
        object_ids=("object:native-triage-no-match",),
    )
    request = retrieval_helpers._context_request(
        key="native-triage-no-match",
        composition_request=composition_request,
        composition=composition,
        inputs=inputs,
        authority_request=authority_request,
        authority_result=authority_result,
    )
    builder = retrieval_helpers._builder(
        tmp_path,
        name="native-triage-no-match",
        composer=composer,
        cas_root=cas_root,
    )
    receipt = builder.execute(request)
    assert receipt.no_match
    return (
        RetrievalContextAuthority(
            builder.journal.path, {request.request_digest: (request, receipt)}
        ),
        RetrievalInputBinding.from_receipt(request, receipt),
    )


def _authenticator():
    return StaticAuthenticator(
        credentials={"token-1": StaticPrincipal("principal.alpha")},
        authority_domain="newsroom.authority",
    )


def _shared_system(
    tmp_path, monkeypatch, retrieval, *, collision=None, candidate_citations=None
):
    adapter = MemoryNeo4jAdapter()
    monkeypatch.setattr(
        "newsroom.authority._graphiti_increment4_system._open_structural_graph_adapter",
        lambda _: adapter,
    )
    rights, hydration, admissions = _policy_registries()
    authenticator = _authenticator()
    scopes = discovery_scopes() | frozenset(
        {
            "authority.fixture.events.read",
            relationship_command_definition().required_scope,
            candidate_command_definition().required_scope,
        }
    )
    authorizer = StaticAuthorizer(
        policy_version="native-triage-test-v1",
        grants_by_principal={"principal.alpha": scopes},
    )
    if collision is None:
        collision = CurrentCollisionEffectEnforcer(
            current_authority_provider=lambda _: None,
            trusted_boundary=TrustedCurrentCollisionAuthorityBoundary(
                "fixture-scope",
                "fixture-profile",
                "sha256:" + "a" * 64,
                "sha256:" + "b" * 64,
                "fixture-port",
            ),
        )
    event_policy = EventReadPolicy(
        policy_id="native-triage-events-v1",
        purpose="native.triage",
        required_scope="authority.fixture.events.read",
        allowed_principal_ids=frozenset({"principal.alpha"}),
        allowed_security_scopes=frozenset({"authority.source_registry"}),
        allowed_trust_scopes=frozenset({TrustScope.ADMITTED}),
        metadata_classes=frozenset({MetadataClass.ROUTING}),
    )
    dependencies = None
    if candidate_citations is not None:
        dependencies = lambda **_: (
            retrieval,
            collision,
            candidate_citations,
        )
    return open_hermes_native_authority_system(
        path=tmp_path / "native.sqlite3",
        object_root=tmp_path / "objects",
        workspace_root=tmp_path.resolve(),
        registry=registry_v1(),
        payload_schemas=payload_schemas(),
        admission_registry=admissions,
        rights_policies=rights,
        hydration_policies=hydration,
        contracts=increment4_admitted_contract_registry(),
        authenticator=authenticator,
        authorizer=authorizer,
        event_read_policy=event_policy,
        source_read_policy=source_read_policy(),
        check_read_policy=check_read_policy(),
        discovery_read_policy=discovery_read_policy(),
        extraction_read_policy=extraction_read_policy(),
        entity_read_policy=entity_read_policy(),
        relation_read_policy=relation_read_policy(),
        graphiti_read_policy=graphiti_read_policy(),
        projection_read_policy=increment4_projection_read_policy(),
        object_limits=ObjectLimits(
            global_max_bytes=1024 * 1024,
            class_max_bytes={"source_capture": 1024 * 1024},
            max_read_bytes=1024 * 1024,
            min_free_bytes=0,
            io_chunk_bytes=64,
            max_staging_bytes=1024 * 1024,
            max_range_bytes=1024 * 1024,
        ),
        neo4j_config=config(),
        retrieval_authority=None if dependencies is not None else retrieval,
        collision_enforcer=None if dependencies is not None else collision,
        native_dependency_factory=dependencies,
        clock=lambda: FIXED_NOW,
    )


def _candidate_collision(tmp_path, hypothesis):
    binding = CandidateUseCollisionBinding(
        hypothesis.hypothesis_id,
        hypothesis.version_id,
        hypothesis.canonical_digest,
        CandidateUseOperation.ADMIT_NEW_CANDIDATE,
        None,
        "candidate-development",
        collision_helpers.COLLISION_DIGEST,
        "retrieval-generation-v2",
        QUERY_VALID,
        SERVING,
        42,
    )
    named = replace(
        collision_request(
            idempotency_key=binding.idempotency_key,
            generation_id=binding.generation_id,
        ),
        collision_key_digest=binding.collision_key_digest,
    )
    root = tmp_path / "collision"
    root.mkdir()
    database = authority_database(root)
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM development_candidates_v2")
    executed = executor(root, database).execute(named, authorize(root, named))
    assert executed.authority_receipt_bytes is not None
    evidence = CurrentCollisionReceiptEvidence(
        named,
        executed.receipt.canonical_bytes,
        executed.authority_receipt_bytes,
    )
    request = CurrentCollisionEligibilityRequest(binding, named.request_digest)
    context = collision_helpers._trusted_context(evidence)
    decision = collision_helpers._decide(
        request=request,
        evidence=evidence,
        trusted_context=context,
    )
    snapshot = CurrentCollisionAuthoritySnapshot(evidence, context)
    enforcer = CurrentCollisionEffectEnforcer(
        current_authority_provider=lambda _: snapshot,
        trusted_boundary=TrustedCurrentCollisionAuthorityBoundary(
            context.authority_scope_id,
            context.authority_profile_id,
            context.adapter_config_digest,
            context.port_registry_digest,
            context.port_id,
        ),
    )
    return request, decision, enforcer


def _actor_digest():
    authentication = _authenticator().authenticate(proof(), now=FIXED_NOW)
    return digest_bytes(
        canonical_json_bytes(
            {
                "principal_id": authentication.principal_id,
                "credential_binding_digest": authentication.credential_binding_digest,
            }
        )
    )


def test_committed_lead_builds_exact_native_work_and_pending_is_visible(tmp_path) -> None:
    lead, disposition = _admitted(tmp_path)
    work = build_native_triage_work(
        admitted_leads=((lead, disposition),),
        retrieval=work_item_helpers._pending(),
    )

    assert work.version.decision_leads[0].lead_digest == lead.canonical_digest
    assert (
        PrioritySelection.from_canonical_bytes(work.version.priority.selection_bytes).lane
        is PriorityLane.ROUTINE
    )

    class WorkItems:
        @staticmethod
        def create_or_replay(item, version):
            assert item == work.item
            return version

    class System:
        work_items = WorkItems()

    result = advance_native_triage(
        System(), work=work, scheduling_decision=None, proof=proof()
    )
    assert result.state == "RETRIEVAL_PENDING"
    assert result.batch is None
    assert not result.candidate_ready


def test_shared_writer_advances_no_match_through_hypothesis_relationship(
    tmp_path, monkeypatch
) -> None:
    retrieval_authority, retrieval = _no_match_retrieval(tmp_path)
    with _shared_system(tmp_path, monkeypatch, retrieval_authority) as system:
        seed_check_lineage(system)
        admitted = system.discovery.admit_signal_to_lead(
            exact_admission_request(), proof=proof()
        )
        assert admitted.lead is not None
        assert admitted.initial_disposition is not None
        hold_id = GateDecisionId.parse(work_item_helpers._id(9000))
        promoted_id = GateDecisionId.parse(work_item_helpers._id(9001))
        system.discovery.decide_gate(
            replace(
                exact_gate_request(),
                decision_id=hold_id,
                decision_ordinal=2,
                previous_decision_id=exact_gate_request().decision_id,
                basis=replace(
                    exact_gate_request().basis,
                    policy_current=False,
                    operationally_executable=False,
                    time_validity=TimeValidity.CURRENT,
                ),
                outcome=GateOutcome.OPERATIONAL_HOLD,
                terminality=DecisionTerminality.PENDING_CONDITION,
                primary_reason=reason("OPS.POLICY_STALE"),
                next_action=NextAction(
                    NextActionKind.REVIEW,
                    "REVIEW_STALE_POLICY",
                    owner="discovery-operator",
                    instructions="Revalidate the deterministic Gate policy.",
                ),
                idempotency_key="native-triage-gate-hold",
            ),
            proof=proof(),
        )
        system.discovery.decide_gate(
            replace(
                exact_gate_request(),
                decision_id=promoted_id,
                decision_ordinal=3,
                previous_decision_id=hold_id,
                idempotency_key="native-triage-gate-repromotion",
            ),
            proof=proof(),
        )
        replacement = system.discovery.record_lead_disposition(
            replace(
                disposition_request(),
                decision_id=LeadDispositionDecisionId.parse(
                    work_item_helpers._id(9002)
                ),
                gate_decision_id=promoted_id,
                decision_ordinal=2,
                previous_decision_id=admitted.initial_disposition.request.decision_id,
                idempotency_key="native-triage-replacement-queue",
            ),
            proof=proof(),
        )
        work = build_native_triage_work(
            admitted_leads=((admitted.lead, replacement),),
            retrieval=retrieval,
        )
        assert work.version.decision_leads[0].disposition_ordinal == 2
        schedule = plan_native_schedule(work)
        assert schedule.state == "SCHEDULED"
        assert schedule.decision is not None
        result = advance_native_triage(
            system,
            work=work,
            scheduling_decision=schedule.decision,
            proof=proof(),
        )

        assert result.state == "CANDIDATE_READY"
        assert result.candidate_ready
        assert result.hypothesis is not None
        assert result.relationship is not None
        assert result.relationship.decision.value == "REL_NO_ADEQUATE_PRIOR_MATCH"
        assert system.hypotheses.load_version(result.hypothesis.version_id) == (
            result.hypothesis
        )
        assert system.relationships.load(result.relationship.canonical_digest) == (
            result.relationship
        )

        replay = advance_native_triage(
            system,
            work=work,
            scheduling_decision=schedule.decision,
            proof=proof(),
        )
        assert replay == result

    collision_request_value, collision_decision, enforcer = _candidate_collision(
        tmp_path, result.hypothesis
    )
    with _shared_system(
        tmp_path,
        monkeypatch,
        retrieval_authority,
        collision=enforcer,
    ) as reopened:
        reopened_result = advance_native_triage(
            reopened,
            work=work,
            scheduling_decision=schedule.decision,
            proof=proof(),
        )
        assert reopened_result == result
        manifest = reopened.build_candidate_manifest(
            result.hypothesis.version_id,
            result.relationship.canonical_digest,
            collision_decision,
            proof=proof(),
        )
        candidate_request = CandidateAdmissionRequest(
            str(uuid.uuid4()),
            _actor_digest(),
            f"native-candidate:{result.hypothesis.version_id}",
            None,
            None,
            0,
            manifest.semantic_scope_digest,
            collision_request_value.request_digest,
            manifest.governing_state_binding.canonical_digest,
            None,
        )
        stale = _admit_native_triage_candidate(
            reopened,
            triage=reopened_result,
            collision_request=collision_request_value,
            collision_decision=collision_decision,
            candidate_request=replace(
                candidate_request,
                expected_governing_state_digest="sha256:" + "0" * 64,
            ),
            current_candidate_version=None,
            proof=proof(),
        )
        assert stale.state == "CANDIDATE_HOLD"
        assert stale.candidate is None
        admitted = _admit_native_triage_candidate(
            reopened,
            triage=reopened_result,
            collision_request=collision_request_value,
            collision_decision=collision_decision,
            candidate_request=candidate_request,
            current_candidate_version=None,
            proof=proof(),
        )
        assert admitted.state == "CANDIDATE_ADMITTED"
        assert admitted.admission is not None
        assert admitted.candidate is not None
        assert reopened.candidates.load_version(admitted.candidate.version_id) == (
            admitted.candidate
        )
        assert (
            advance_native_triage(
                reopened,
                work=work,
                scheduling_decision=schedule.decision,
                proof=proof(),
                collision_request=collision_request_value,
                collision_decision=collision_decision,
                candidate_request=candidate_request,
            ).candidate
            == admitted.candidate
        )

    from newsroom.authority._event_hypothesis_system import _HypothesisStore
    from newsroom.increment6.dispositions import ProposalDispositionStore

    verification_calls = {"hypothesis": 0, "disposition": 0}
    original_hypothesis_verify = _HypothesisStore._verify
    original_disposition_verify = ProposalDispositionStore._verify_integrity

    def counted_hypothesis_verify(store, **kwargs):
        verification_calls["hypothesis"] += 1
        return original_hypothesis_verify(store, **kwargs)

    def counted_disposition_verify(store):
        verification_calls["disposition"] += 1
        return original_disposition_verify(store)

    monkeypatch.setattr(_HypothesisStore, "_verify", counted_hypothesis_verify)
    monkeypatch.setattr(
        ProposalDispositionStore, "_verify_integrity", counted_disposition_verify
    )

    with _shared_system(
        tmp_path,
        monkeypatch,
        retrieval_authority,
        collision=enforcer,
    ) as restarted:
        # Constructor validation remains complete; the final native transaction
        # verifies relationship/lineage/Candidate upstream history only once.
        assert verification_calls == {"hypothesis": 1, "disposition": 1}
        restarted_admission = advance_native_triage(
            restarted,
            work=work,
            scheduling_decision=schedule.decision,
            proof=proof(),
            collision_request=collision_request_value,
            collision_decision=collision_decision,
            candidate_request=candidate_request,
        )
        assert restarted_admission.candidate == admitted.candidate

        # Opening's local values are not retained by subsequent independent reads.
        for _ in range(2):
            before = dict(verification_calls)
            assert restarted.candidates.load_version(admitted.candidate.version_id) == (
                admitted.candidate
            )
            assert verification_calls == {
                name: count + 1 for name, count in before.items()
            }

        candidate_store = (
            restarted.candidates._StoryCandidateAuthority__authority._authority
        )
        with candidate_store._transaction():
            with pytest.raises(ValueError, match="Candidate relationship is absent"):
                candidate_store._verify(relationship_receipts={})

        def rewrite_relationship(raw):
            with closing(sqlite3.connect(tmp_path / "native.sqlite3")) as connection:
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    trigger_name = "immutable_event_hypothesis_relationship_update"
                    trigger_sql = connection.execute(
                        "SELECT sql FROM sqlite_schema WHERE name=?", (trigger_name,)
                    ).fetchone()[0]
                    connection.execute(f"DROP TRIGGER {trigger_name}")
                    connection.execute(
                        "UPDATE event_hypothesis_relationship_decisions "
                        "SET assessment_bytes=?", (raw,),
                    )
                    connection.execute(trigger_sql)

        # Same-count mutation after OPEN must be observed, not served from the
        # earlier transaction's relationship receipts.
        rewrite_relationship(b"{}")
        try:
            with pytest.raises(ValueError, match="relationship|Candidate"):
                restarted.candidates.load_version(admitted.candidate.version_id)
        finally:
            rewrite_relationship(result.relationship.canonical_bytes)
        assert restarted.candidates.load_version(admitted.candidate.version_id) == (
            admitted.candidate
        )

    # The composed OPEN also rechecks the altered row and releases its writer
    # on failure; restoring the exact bytes permits a fully checked reopen.
    rewrite_relationship(b"{}")
    try:
        with pytest.raises(ValueError, match="relationship"):
            _shared_system(tmp_path, monkeypatch, retrieval_authority, collision=enforcer)
    finally:
        rewrite_relationship(result.relationship.canonical_bytes)
    with _shared_system(
        tmp_path, monkeypatch, retrieval_authority, collision=enforcer
    ) as corrected:
        assert corrected.candidates.load_version(admitted.candidate.version_id) == (
            admitted.candidate
        )

    from newsroom.increment6.dispositions import (
        _require_verification_snapshot, _verification_snapshot,
    )
    snapshot_path = tmp_path / "semantic-snapshot.sqlite3"
    with closing(sqlite3.connect(snapshot_path, isolation_level=None)) as reader, closing(
        sqlite3.connect(snapshot_path, isolation_level=None)
    ) as external_writer:
        reader.execute("PRAGMA journal_mode=WAL")
        external_writer.execute("PRAGMA journal_mode=WAL")
        expected_snapshot = _verification_snapshot(reader)
        reader.execute("BEGIN")
        external_writer.execute("CREATE TABLE external_change(value INTEGER)")
        with pytest.raises(
            RuntimeError, match="native semantic verification snapshot changed"
        ):
            _require_verification_snapshot(reader, expected_snapshot)
        reader.execute("ROLLBACK")

    # The retained authority event and Candidate still refer to the now missing
    # relationship. Reuse must not convert that missing upstream into success.
    with closing(sqlite3.connect(tmp_path / "native.sqlite3")) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            trigger_name = "retained_event_hypothesis_relationship_delete"
            trigger_sql = connection.execute(
                "SELECT sql FROM sqlite_schema WHERE name=?", (trigger_name,)
            ).fetchone()[0]
            connection.execute(f"DROP TRIGGER {trigger_name}")
            connection.execute("DELETE FROM event_hypothesis_relationship_decisions")
            connection.execute(trigger_sql)
    with pytest.raises((AuthoritySchemaError, ValueError), match="relationship|foreign"):
        _shared_system(tmp_path, monkeypatch, retrieval_authority, collision=enforcer)
