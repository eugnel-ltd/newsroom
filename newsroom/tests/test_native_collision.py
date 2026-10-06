from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from newsroom.authority import AuthorityEvents, UtcTimestamp
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority.types import AggregateId, ObjectAdmissionId
from newsroom.control_plane.native_collision import (
    NativeCollisionAuthority,
    NativeCollisionHold,
    NativeCollisionIdentity,
)
from newsroom.control_plane.native_cycle import _revision_successor, advance_native_cycle
from newsroom.control_plane.corpus import CorpusAuthorityBinding
from newsroom.control_plane.graphiti_operational_readiness import _source_requests
from newsroom.control_plane.native_discovery import NativeDiscovery
from newsroom.effective_revision import EffectiveRevisionIdentity
from newsroom.graphiti_adapter.identity import observation_authority_ids
from newsroom.increment5.branch_contracts import BranchMode, BranchOutcome
from newsroom.increment5.native_retrieval import (
    NATIVE_VECTOR_PROFILE,
    NativeDocumentReceipt,
    NativeGraphBranchReceipt,
    NativeRetrievalContext,
    NativeRetrievalContextReadPort,
    NativeRetrievalContextReceipt,
    NativeRetrievalContextRequest,
    NativeVectorBranchReceipt,
)
from newsroom.increment6.collision import (
    CandidateUseOperation,
    CurrentCollisionEligibilityBlocked,
)
from newsroom.increment6.candidates import CandidateContractError
from newsroom.increment6.hypotheses import EventHypothesis
from newsroom.increment6.dispositions import CurrentCandidateCitation
from newsroom.increment6.work_items import RetrievalBindingState, RetrievalInputBinding
from newsroom.sources import SourceRevisionId
from newsroom.tests.discovery_3d_authority_helpers import (
    exact_admission_request,
    proof,
    seed_check_lineage,
)
from newsroom.tests.test_increment5d1_hybrid_composer import (
    _coherent_system,
    _parse_receipt,
    authorize,
)
from newsroom.tests.test_native_triage import (
    _actor_digest,
    _no_match_retrieval,
    _shared_system,
)
from newsroom.tests.test_native_discovery import _controller, _current_rights
from newsroom.tests.test_graphiti_operational_readiness import (
    _next_revision,
    _rights,
    _unit,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _native_binding(tmp_path, lead):
    root = tmp_path / "branches"
    root.mkdir(parents=True)
    dispatcher, requests = _coherent_system(root)
    raw = []
    for index in (0, 1, 3):
        result = dispatcher.execute(
            requests[index], authorize(root, requests[index], name=f"native-{index}")
        )
        assert result.upstream_raw_receipt_bytes is not None
        raw.append(
            _parse_receipt(
                result.receipt.tool_id, result.upstream_raw_receipt_bytes
            )
        )
    exact, fulltext, graph = raw
    assert fulltext.snapshot is not None
    vector = NativeVectorBranchReceipt(
        str(uuid.uuid4()),
        _digest("1"),
        BranchMode.VECTOR,
        BranchOutcome.COMPLETE,
        "NO_MATCH",
        str(fulltext.snapshot.generation_id),
        fulltext.snapshot.generation_identity_digest,
        NATIVE_VECTOR_PROFILE,
        graph.query_valid_time,
        graph.serving_time,
        (),
        1,
    )
    native_graph = NativeGraphBranchReceipt(
        str(uuid.uuid4()),
        _digest("3"),
        (str(lead.request.lead_id),),
        "graph.increment4.admitted",
        "increment4-admitted-v1",
        "increment4-neo4j-v1",
        _digest("4"),
        _digest("5"),
        vector.generation_id,
        "authority-selected-active",
        10,
        graph.query_valid_time,
        graph.serving_time,
        (),
        (),
        reason="NO_MATCH",
    )
    document = NativeDocumentReceipt(
        "native-document-event",
        "native-document-command",
        AggregateId.new(),
        1,
        ObjectAdmissionId.new(),
        _digest("2"),
        ObjectAdmissionId.new(),
        ObjectAdmissionId.new(),
    )
    request = NativeRetrievalContextRequest(
        str(uuid.uuid4()),
        "native-context",
        AggregateId.new(),
        0,
        str(lead.request.lead_id),
        lead.canonical_digest,
        "source-a",
        _digest("9"),
        exact.canonical_bytes,
        fulltext.canonical_bytes,
        vector.canonical_bytes,
        native_graph.canonical_bytes,
        (document,),
    )
    branch_bytes = (
        request.exact_receipt_bytes,
        request.fulltext_receipt_bytes,
        request.vector_receipt_bytes,
        request.graph_receipt_bytes,
    )
    context = NativeRetrievalContext(
        str(uuid.uuid4()),
        request.request_id,
        request.request_digest,
        request.lead_id,
        request.lead_digest,
        request.authority_scope_id,
        request.rights_inventory_digest,
        vector.generation_id,
        vector.query_valid_time,
        vector.serving_time,
        tuple(digest_bytes(item) for item in branch_bytes),
        tuple(json.loads(item) for item in branch_bytes),
        (document.projection_value(),),
        "COMPLETE",
        False,
    )
    receipt = NativeRetrievalContextReceipt(
        context.context_id,
        request.request_id,
        request.request_digest,
        request.aggregate_id,
        1,
        "native-context-event",
        "native-context-command",
        ObjectAdmissionId.new(),
        context.digest,
        request.authority_scope_id,
        request.rights_inventory_digest,
        context.generation_id,
        context.query_valid_time,
        context.serving_time,
        *branch_bytes,
        "principal.alpha",
        "newsroom.authority",
        no_match=False,
    )
    binding = RetrievalInputBinding(
        RetrievalBindingState.RECEIPT,
        request.request_id,
        request.idempotency_key,
        request.request_digest,
        request.canonical_bytes,
        context.context_id,
        receipt.receipt_digest,
        receipt.outcome,
        receipt.reason,
        receipt.no_match,
        receipt.canonical_bytes,
    )
    return binding, receipt, context


def _read_port(contexts):
    # Production obtains this exact type from NativeRetrievalDocuments.
    port = object.__new__(NativeRetrievalContextReadPort)
    object.__setattr__(port, "_read", lambda receipt: contexts[receipt.context_id])
    return port


def _same_state_revision(
    unit,
    *,
    observed_at="2026-09-02T12:15:00.000000Z",
    updated_at="2026-09-02T12:10:00.000000Z",
):
    base = replace(
        unit,
        body=unit.body,
        updated_at=updated_at,
        effective_revision=EffectiveRevisionIdentity(
            source_id=unit.source_id,
            item_key=unit.item_key,
            revision_digest=unit.revision_digest,
            first_observed_at=observed_at,
        ),
        effective_pull_first_observed_at=observed_at,
        authority=None,
    )
    admission_id, access_id, definition_id, item_id, revision_id, representation_id = (
        observation_authority_ids(
            source_id=base.source_id,
            item_key=base.item_key,
            revision_digest=base.revision_digest,
            representation_digest=base.representation_digest,
            rights_authority_run_id="rights-run-1",
            rights_gate_id="RIGHTS_UK-01",
            rights_gate_reason="retained PASS",
            published_at=base.published_at,
            updated_at=base.updated_at,
        )
    )
    records = (
        {"record_type": "SOURCE_DEFINITION", "record_id": str(definition_id)},
        {
            "record_type": "SOURCE_DEFINITION_VERSION",
            "record_id": unit.authority.definition_version_id,
        },
        {"record_type": "SOURCE_ITEM", "record_id": str(item_id)},
        {"record_type": "SOURCE_REVISION", "record_id": str(revision_id)},
        {
            "record_type": "DISCOVERY_REPRESENTATION",
            "record_id": str(representation_id),
        },
        {"record_type": "OBJECT_ADMISSION", "record_id": str(admission_id)},
        {"record_type": "OBJECT_ACCESS_DECISION", "record_id": str(access_id)},
    )
    return replace(
        base,
        authority=CorpusAuthorityBinding(
            admission_id=str(admission_id),
            access_decision_id=str(access_id),
            definition_id=str(definition_id),
            definition_version_id=unit.authority.definition_version_id,
            item_id=str(item_id),
            revision_id=str(revision_id),
            representation_id=str(representation_id),
            records=records,
        ),
    )


def _retain_source_revision(system, unit, *, prior_revision_id=None):
    requests = _source_requests(
        unit,
        _rights(),
        prior_revision_id=(
            None
            if prior_revision_id is None
            else SourceRevisionId.parse(prior_revision_id)
        ),
    )
    if prior_revision_id is not None:
        requests = (
            *requests[:3],
            replace(
                requests[3],
                source_native_revision_token=unit.updated_at,
            ),
            requests[4],
        )
    methods = (
        system.sources.register_definition,
        system.sources.record_definition_version,
        system.sources.register_item,
        system.sources.record_revision,
        system.sources.record_representation,
    )
    for method, request in zip(methods, requests, strict=True):
        method(request, proof=proof())


def _native_collision(tmp_path, contexts, receipts, *, authorized=True) -> NativeCollisionAuthority:
    def provenance(event_id, _proof):
        receipt = receipts[event_id]
        authentication_context_id = "native-context-authentication"
        decision_id = "00000000-0000-4000-8000-000000006201"
        decision_bytes = canonical_json_bytes({
            "authorization_decision_id": decision_id,
            "allowed": authorized,
            "event_id": event_id,
        })
        return SimpleNamespace(
            event=SimpleNamespace(
                event_id=receipt.event_id,
                command_id=receipt.command_id,
                aggregate_id=str(receipt.aggregate_id),
                aggregate_version=receipt.aggregate_version,
                object_admission_id=str(receipt.admission_id),
                payload_digest=receipt.context_object_digest,
                principal_id="principal.alpha",
                authentication_context_id=authentication_context_id,
                authorization_request_digest=_digest("a"),
                authorization_decision_id=decision_id,
            ),
            authentication=SimpleNamespace(
                authentication_context_id=authentication_context_id,
                principal_id="principal.alpha",
                authority_domain="newsroom.authority",
            ),
            authorization_request=SimpleNamespace(
                authentication_context_id=authentication_context_id,
                principal_id="principal.alpha",
                authority_domain="newsroom.authority",
                request_digest=_digest("a"),
                operation_type="command:retrieval.native_context.admit",
                required_scope="authority.retrieval.context",
            ),
            authorization_decision=SimpleNamespace(
                authentication_context_id=authentication_context_id,
                authorization_request_digest=_digest("a"),
                authorization_decision_id=decision_id,
                allowed=authorized,
                canonical_bytes=decision_bytes,
                canonical_digest=digest_bytes(decision_bytes),
            ),
            command_definition=SimpleNamespace(
                command_type="retrieval.native_context.admit",
            ),
        )

    return NativeCollisionAuthority(
        authority_path=tmp_path / "native.sqlite3",
        journal_path=tmp_path / "native-collision.sqlite3",
        identity=NativeCollisionIdentity(
            "source-a",
            "principal.alpha",
            "newsroom.authority",
        ),
        context_reader=lambda receipt, *, proof: contexts[receipt.context_id],
        events=AuthorityEvents(
            policy_id="native-collision-test-events",
            read=lambda *_: (),
            provenance=provenance,
            result=lambda *_: pytest.fail("unexpected command-result read"),
        ),
    )


def test_native_collision_reads_current_candidate_and_replays_after_restart(
    tmp_path, monkeypatch
) -> None:
    retrieval_authority, _ = _no_match_retrieval(tmp_path)
    contexts = {}
    receipts = {}
    retrieval_authority._native_context_read_port = _read_port(contexts)
    collision = _native_collision(tmp_path, contexts, receipts)
    with _shared_system(
        tmp_path, monkeypatch, retrieval_authority, collision=collision.enforcer
    ) as system:
        seed_check_lineage(system)
        admitted = system.discovery.admit_signal_to_lead(
            exact_admission_request(), proof=proof()
        )
        status = system.discovery.current_status(
            admitted.lead.request.signal_id, proof=proof()
        )
        retrieval, receipt, context = _native_binding(tmp_path, status.lead)
        assert not retrieval.no_match
        contexts[context.context_id] = context
        receipts[receipt.event_id] = receipt
        outcome = advance_native_cycle(
            system,
            (status,),
            retrieval=type(
                "Retrieval",
                (),
                {"retrieve": lambda self, lead, *, proof: retrieval},
            )(),
            collision_requests=collision,
            actor_identity_digest=_actor_digest(),
            proof=proof(),
            owner_stop_check=lambda: None,
            owner_stop_fence=nullcontext,
        )[0]
        assert outcome.state == "CANDIDATE_ADMITTED"
        triage = outcome.triage
        candidate = outcome.triage.candidate
        assert candidate is not None

    reopened_collision = _native_collision(tmp_path, contexts, receipts)
    with _shared_system(
        tmp_path,
        monkeypatch,
        retrieval_authority,
        collision=reopened_collision.enforcer,
    ) as reopened:
        replay = advance_native_cycle(
            reopened,
            (status,),
            retrieval=type(
                "Retrieval",
                (),
                {"retrieve": lambda self, lead, *, proof: retrieval},
            )(),
            collision_requests=reopened_collision,
            actor_identity_digest=_actor_digest(),
            proof=proof(),
            owner_stop_check=lambda: None,
            owner_stop_fence=nullcontext,
        )[0]
        assert replay.state == "CANDIDATE_ADMITTED"
        assert replay.triage.candidate == candidate

        citation = reopened_collision.current_candidate_citation(
            status.lead, retrieval, proof=proof()
        )
        assert citation is not None
        stale_values = {
            field: getattr(citation, field)
            for field in citation.__dataclass_fields__
            if field != "citation_id"
        }
        stale_values["candidate_version_digest"] = "sha256:" + "0" * 64
        monkeypatch.setattr(
            type(reopened.candidates),
            "versions",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("successor used unbounded Candidate history")
            ),
        )
        monkeypatch.setattr(
            type(reopened.hypotheses),
            "current",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("successor used unbounded Hypothesis history")
            ),
        )
        with pytest.raises(NativeCollisionHold, match="CURRENT_CANDIDATE_CITATION_STALE"):
            _revision_successor(
                reopened,
                status.lead,
                CurrentCandidateCitation.create(**stale_values),
                proof=proof(),
            )

        proposal_id = str(uuid.uuid4())
        hypothesis_id = EventHypothesis.allocate(
            proposal_id, triage.hypothesis.proposal_local_id
        ).hypothesis_id
        other_hypothesis = replace(
            triage.hypothesis,
            hypothesis_id=hypothesis_id,
            version_id=str(uuid.uuid5(uuid.UUID(hypothesis_id), "version:1")),
            proposal_id=proposal_id,
        )
        occupied = reopened_collision.request(
            replace(triage, hypothesis=other_hypothesis), retrieval, proof=proof()
        )
        assert occupied.binding.operation is CandidateUseOperation.USE_CURRENT_CANDIDATE
        assert occupied.binding.expected_candidate_id == candidate.candidate_id

        with sqlite3.connect(tmp_path / "native.sqlite3") as connection:
            trigger = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name='candidate_head_update_guard'"
            ).fetchone()[0]
            connection.execute("DROP TRIGGER candidate_head_update_guard")
            connection.execute(
                "UPDATE story_candidate_heads SET current_version_ordinal="
                "current_version_ordinal+1 WHERE candidate_id=?",
                (candidate.candidate_id,),
            )
            connection.execute(trigger)
        with pytest.raises(CandidateContractError, match="Candidate head"):
            _revision_successor(
                reopened, status.lead, citation, proof=proof()
            )

    with sqlite3.connect(tmp_path / "native-collision.sqlite3") as journal:
        states = {
            row[0]
            for row in journal.execute(
                "SELECT collision_state FROM native_collision_receipts"
            )
        }
        assert journal.execute(
            "SELECT count(*) FROM native_collision_requests"
        ).fetchone()[0] == 3
    assert states == {"UNOCCUPIED", "OCCUPIED"}
    assert NativeRetrievalContextReceipt.from_bytes(receipt.canonical_bytes) == receipt


def test_same_state_association_replays_then_allows_development(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "newsroom.control_plane.cycle._dispatch_rights_decision",
        lambda *args, **kwargs: _current_rights(),
    )
    retrieval_authority, _ = _no_match_retrieval(tmp_path)
    contexts: dict[str, NativeRetrievalContext] = {}
    receipts: dict[str, NativeRetrievalContextReceipt] = {}
    retrieval_authority._native_context_read_port = _read_port(contexts)
    collision = _native_collision(tmp_path, contexts, receipts)
    proving = sqlite3.connect(":memory:")
    try:
        with _shared_system(
            tmp_path,
            monkeypatch,
            retrieval_authority,
            collision=collision.enforcer,
            candidate_citations=collision.candidate_citation_read_port(),
        ) as system:
            controller = _controller(system, proving)
            first_unit = _unit()
            _retain_source_revision(system, first_unit)
            first_delivery = controller.deliver(
                first_unit,
                now=UtcTimestamp.parse(first_unit.effective_revision.first_observed_at),
                proof=proof(),
            )
            first = controller.admit_lead(
                first_delivery,
                now=UtcTimestamp.parse(first_unit.effective_revision.first_observed_at),
                proof=proof(),
            )
            first_binding, first_receipt, first_context = _native_binding(
                tmp_path / "first-retrieval", first.lead
            )
            contexts[first_context.context_id] = first_context
            receipts[first_receipt.event_id] = first_receipt
            admitted = advance_native_cycle(
                system,
                (first,),
                retrieval=SimpleNamespace(
                    retrieve=lambda lead, *, proof: first_binding
                ),
                collision_requests=collision,
                actor_identity_digest=_actor_digest(),
                proof=proof(),
                owner_stop_check=lambda: None,
                owner_stop_fence=nullcontext,
            )[0]
            assert admitted.state == "CANDIDATE_ADMITTED"
            candidate_v1 = admitted.triage.candidate

            same_unit = _same_state_revision(first_unit)
            _retain_source_revision(
                system,
                same_unit,
                prior_revision_id=first_unit.authority.revision_id,
            )
            same_delivery = controller.deliver(
                same_unit,
                now=UtcTimestamp.parse(same_unit.effective_revision.first_observed_at),
                proof=proof(),
            )
            same = controller.admit_lead(
                same_delivery,
                now=UtcTimestamp.parse(same_unit.effective_revision.first_observed_at),
                proof=proof(),
            )
            same_binding, same_receipt, same_context = _native_binding(
                tmp_path / "same-retrieval", same.lead
            )
            contexts[same_context.context_id] = same_context
            receipts[same_receipt.event_id] = same_receipt
            associated = advance_native_cycle(
                system,
                (same,),
                retrieval=SimpleNamespace(
                    retrieve=lambda lead, *, proof: same_binding
                ),
                collision_requests=collision,
                actor_identity_digest=_actor_digest(),
                proof=proof(),
                owner_stop_check=lambda: None,
                owner_stop_fence=nullcontext,
            )[0]
            assert associated.state == "SAME_STATE_ASSOCIATED", (
                associated.triage.hypothesis.proposed_relationship,
                first_unit.revision_digest,
                same_unit.revision_digest,
                first.lead.request.item_id,
                same.lead.request.item_id,
            )
            assert associated.triage.hypothesis.ordinal == 2
            hypothesis_count = len(system.hypotheses.versions(
                associated.triage.hypothesis.hypothesis_id
            ))
            replay = advance_native_cycle(
                system,
                (same,),
                retrieval=SimpleNamespace(
                    retrieve=lambda lead, *, proof: same_binding
                ),
                collision_requests=collision,
                actor_identity_digest=_actor_digest(),
                proof=proof(),
                owner_stop_check=lambda: None,
                owner_stop_fence=nullcontext,
            )[0]
            assert replay.state == "SAME_STATE_ASSOCIATED"
            assert replay.triage is None
            assert len(system.hypotheses.versions(
                associated.triage.hypothesis.hypothesis_id
            )) == hypothesis_count

            second_same_unit = _same_state_revision(
                same_unit,
                observed_at="2026-09-02T12:20:00.000000Z",
                updated_at="2026-09-02T12:15:00.000000Z",
            )
            _retain_source_revision(
                system,
                second_same_unit,
                prior_revision_id=same_unit.authority.revision_id,
            )
            second_delivery = controller.deliver(
                second_same_unit,
                now=UtcTimestamp.parse(
                    second_same_unit.effective_revision.first_observed_at
                ),
                proof=proof(),
            )
            second_same = controller.admit_lead(
                second_delivery,
                now=UtcTimestamp.parse(
                    second_same_unit.effective_revision.first_observed_at
                ),
                proof=proof(),
            )
            second_binding, second_receipt, second_context = _native_binding(
                tmp_path / "second-same-retrieval", second_same.lead
            )
            contexts[second_context.context_id] = second_context
            receipts[second_receipt.event_id] = second_receipt
            second_associated = advance_native_cycle(
                system,
                (second_same,),
                retrieval=SimpleNamespace(
                    retrieve=lambda lead, *, proof: second_binding
                ),
                collision_requests=collision,
                actor_identity_digest=_actor_digest(),
                proof=proof(),
                owner_stop_check=lambda: None,
                owner_stop_fence=nullcontext,
            )[0]
            assert second_associated.state == "SAME_STATE_ASSOCIATED"
            assert second_associated.triage.hypothesis.ordinal == 3
            raw_historical_citation = collision.current_candidate_citation(
                second_same.lead, second_binding, proof=proof()
            )
            associated_candidate, associated_hypothesis = (
                system.candidates._exact_associated_current_producers(
                    candidate_v1.candidate_id, proof=proof()
                )
            )
            non_hypothesis_forgery = {
                "candidate_id": str(uuid.uuid4()),
                "candidate_version_id": str(uuid.uuid4()),
                "candidate_version_digest": _digest("0"),
                "collision_namespace": "forged-collision-namespace",
                "collision_key_digest": _digest("1"),
                "source_definition_id": str(uuid.uuid4()),
                "source_item_id": str(uuid.uuid4()),
                "retrieval_context_digest": _digest("2"),
                "authorization_receipt_digest": _digest("3"),
                "authorization_decision_id": str(uuid.uuid4()),
            }
            forged_citation_ids = []
            for field, forged_value in non_hypothesis_forgery.items():
                forged_values = {
                    name: getattr(raw_historical_citation, name)
                    for name in raw_historical_citation.__dataclass_fields__
                    if name != "citation_id"
                }
                forged_values[field] = forged_value
                forged = CurrentCandidateCitation.create(**forged_values)
                forged_citation_ids.append(forged.citation_id)
                with pytest.raises(
                    NativeCollisionHold,
                    match="CURRENT_CANDIDATE_ASSOCIATION_DIFFERS",
                ):
                    collision.retain_associated_candidate_citation(
                        forged,
                        associated_candidate,
                        associated_hypothesis,
                        lead=second_same.lead,
                        retrieval=second_binding,
                        proof=proof(),
                    )
            with sqlite3.connect(tmp_path / "native-collision.sqlite3") as journal:
                assert all(
                    journal.execute(
                        "SELECT 1 FROM native_current_candidate_citations "
                        "WHERE citation_id=?",
                        (citation_id,),
                    ).fetchone()
                    is None
                    for citation_id in forged_citation_ids
                )
            with pytest.raises(
                NativeCollisionHold,
                match="CURRENT_CANDIDATE_ASSOCIATION_DIFFERS",
            ):
                wrong_association_values = {
                    field: getattr(raw_historical_citation, field)
                    for field in raw_historical_citation.__dataclass_fields__
                    if field != "citation_id"
                }
                wrong_association_values["hypothesis_id"] = str(uuid.uuid4())
                collision.retain_associated_candidate_citation(
                    CurrentCandidateCitation.create(**wrong_association_values),
                    associated_candidate,
                    associated_hypothesis,
                    lead=second_same.lead,
                    retrieval=second_binding,
                    proof=proof(),
                )
            historical_citation = collision.retain_associated_candidate_citation(
                raw_historical_citation,
                associated_candidate,
                associated_hypothesis,
                lead=second_same.lead,
                retrieval=second_binding,
                proof=proof(),
            )
            with sqlite3.connect(tmp_path / "native.sqlite3") as connection:
                trigger = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE name="
                    "'immutable_event_hypothesis_relationship_update'"
                ).fetchone()[0]
                connection.execute(
                    "DROP TRIGGER immutable_event_hypothesis_relationship_update"
                )
                connection.execute(
                    "UPDATE event_hypothesis_relationship_decisions "
                    "SET decision='REL_DEVELOPMENT_OF' WHERE subject_version_id=?",
                    (associated_hypothesis.version_id,),
                )
            try:
                with pytest.raises(
                    NativeCollisionHold,
                    match="CURRENT_CANDIDATE_ASSOCIATION_DIFFERS",
                ):
                    collision.candidate_citation_read_port().require(
                        historical_citation.citation_id,
                        historical_citation.canonical_digest,
                    )
            finally:
                with sqlite3.connect(tmp_path / "native.sqlite3") as connection:
                    connection.execute(
                        "UPDATE event_hypothesis_relationship_decisions "
                        "SET decision='REL_SAME_STATE' WHERE subject_version_id=?",
                        (associated_hypothesis.version_id,),
                    )
                    connection.execute(trigger)

            # Live HK-02 can retain several source revisions while its current
            # Candidate remains on the last editorially processed revision.
            skipped = second_same_unit
            for observed_at, updated_at in (
                ("2026-09-02T12:21:00.000000Z", "2026-09-02T12:16:00.000000Z"),
                ("2026-09-02T12:22:00.000000Z", "2026-09-02T12:17:00.000000Z"),
            ):
                retained = _same_state_revision(
                    skipped, observed_at=observed_at, updated_at=updated_at
                )
                _retain_source_revision(
                    system,
                    retained,
                    prior_revision_id=skipped.authority.revision_id,
                )
                delivery = controller.deliver(
                    retained,
                    now=UtcTimestamp.parse(observed_at),
                    proof=proof(),
                )
                controller.admit_lead(
                    delivery, now=UtcTimestamp.parse(observed_at), proof=proof()
                )
                skipped = retained
            changed_unit = _next_revision(skipped)
            _retain_source_revision(
                system,
                changed_unit,
                prior_revision_id=skipped.authority.revision_id,
            )
            changed_delivery = controller.deliver(
                changed_unit,
                now=UtcTimestamp.parse(changed_unit.effective_revision.first_observed_at),
                proof=proof(),
            )
            changed = controller.admit_lead(
                changed_delivery,
                now=UtcTimestamp.parse(changed_unit.effective_revision.first_observed_at),
                proof=proof(),
            )
            wrong_source_values = {
                field: getattr(historical_citation, field)
                for field in historical_citation.__dataclass_fields__
                if field != "citation_id"
            }
            wrong_source_values["source_item_id"] = str(uuid.uuid4())
            with pytest.raises(
                NativeCollisionHold,
                match="SOURCE_REVISION_RELATIONSHIP_AMBIGUOUS",
            ):
                _revision_successor(
                    system,
                    changed.lead,
                    CurrentCandidateCitation.create(**wrong_source_values),
                    proof=proof(),
                    current_producers=(
                        associated_candidate,
                        associated_hypothesis,
                    ),
                )
            changed_binding, changed_receipt, changed_context = _native_binding(
                tmp_path / "changed-retrieval", changed.lead
            )
            contexts[changed_context.context_id] = changed_context
            receipts[changed_receipt.event_id] = changed_receipt
            developed = advance_native_cycle(
                system,
                (changed,),
                retrieval=SimpleNamespace(
                    retrieve=lambda lead, *, proof: changed_binding
                ),
                collision_requests=collision,
                actor_identity_digest=_actor_digest(),
                proof=proof(),
                owner_stop_check=lambda: None,
                owner_stop_fence=nullcontext,
            )[0]
            assert developed.state == "CANDIDATE_ADMITTED", developed.reason
            assert developed.triage.candidate.candidate_id == candidate_v1.candidate_id
            assert developed.triage.candidate.ordinal == 2
    finally:
        proving.close()

    reopened_collision = _native_collision(tmp_path, contexts, receipts)
    with _shared_system(
        tmp_path,
        monkeypatch,
        retrieval_authority,
        collision=reopened_collision.enforcer,
        candidate_citations=reopened_collision.candidate_citation_read_port(),
    ) as reopened:
        current_candidate, current_hypothesis = (
            reopened.candidates._exact_associated_current_producers(
                candidate_v1.candidate_id, proof=proof()
            )
        )
        assert current_candidate.ordinal == 2
        assert current_hypothesis.version_id == current_candidate.governing_manifest.hypothesis_version_id
        assert (
            reopened_collision.candidate_citation_read_port().require(
                historical_citation.citation_id,
                historical_citation.canonical_digest,
            )
            == historical_citation
        )


def test_native_collision_rechecks_watermark_and_rejects_cross_work_binding(
    tmp_path, monkeypatch
) -> None:
    retrieval_authority, legacy_retrieval = _no_match_retrieval(tmp_path)
    contexts = {}
    receipts = {}
    retrieval_authority._native_context_read_port = _read_port(contexts)
    collision = _native_collision(tmp_path, contexts, receipts)
    with _shared_system(
        tmp_path, monkeypatch, retrieval_authority, collision=collision.enforcer
    ) as system:
        seed_check_lineage(system)
        admitted = system.discovery.admit_signal_to_lead(
            exact_admission_request(), proof=proof()
        )
        status = system.discovery.current_status(
            admitted.lead.request.signal_id, proof=proof()
        )
        retrieval, receipt, context = _native_binding(tmp_path, status.lead)
        contexts[context.context_id] = context
        receipts[receipt.event_id] = receipt
        held = advance_native_cycle(
            system,
            (status,),
            retrieval=type(
                "Retrieval",
                (),
                {"retrieve": lambda self, lead, *, proof: retrieval},
            )(),
            collision_requests=type(
                "Hold",
                (),
                {"request": lambda self, triage, retrieval, *, proof: None},
            )(),
            actor_identity_digest=_actor_digest(),
            proof=proof(),
            owner_stop_check=lambda: None,
            owner_stop_fence=nullcontext,
        )[0]
        request = collision.request(held.triage, retrieval, proof=proof())
        with pytest.raises(NativeCollisionHold, match="AUTHORITY_DIFFERS"):
            collision.request(
                replace(
                    held.triage,
                    work=replace(
                        held.triage.work,
                        version=replace(
                            held.triage.work.version,
                            retrieval=legacy_retrieval,
                        ),
                    ),
                ),
                retrieval,
                proof=proof(),
            )
        read = collision._read
        monkeypatch.setattr(
            collision,
            "_read",
            lambda namespace=None, key=None: (
                read(namespace, key)[0] + 1,
                *read(namespace, key)[1:],
            ),
        )
        with pytest.raises(CurrentCollisionEligibilityBlocked) as exc_info:
            collision.enforcer.enforce(request=request, effect=lambda _: None)
        assert exc_info.value.decision.outcome.value == "BINDING_MISMATCH"

        denied = _native_collision(
            tmp_path, contexts, receipts, authorized=False
        )
        with pytest.raises(
            NativeCollisionHold,
            match="RETRIEVAL_COLLISION_AUTHORIZATION_DIFFERS",
        ):
            denied.request(held.triage, retrieval, proof=proof())


@pytest.mark.parametrize("text,microsecond", [
    ("2026-09-08T15:00:00Z", 0),
    ("2026-09-08T15:00:00.000000Z", 0),
    ("2026-09-08T15:00:00.123456Z", 123456),
])
def test_native_collision_preserves_authority_timestamp_precision(text, microsecond):
    from newsroom.increment6.collision import _parse_utc
    assert _parse_utc(text, field="serving_time").microsecond == microsecond


@pytest.mark.parametrize("text", [
    "2026-09-08T15:00:00.1Z", "2026-09-08T15:00:00.1234567Z",
    "2026-09-08T15:00:00+00:00", "2026-09-08 15:00:00Z",
])
def test_native_collision_rejects_noncanonical_timestamp(text):
    from newsroom.increment6.collision import _parse_utc, CollisionEligibilityContractError
    with pytest.raises(CollisionEligibilityContractError):
        _parse_utc(text, field="serving_time")


@pytest.mark.parametrize('missing_original', [False, True])
def test_current_rebuild_preserves_same_state_successor_relationship(tmp_path, monkeypatch, missing_original):
    """A real retained association must survive selected-root closure, not be invented."""
    from newsroom.authority.native_current_rebuild import copy_selected_native_store
    from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store
    import shutil
    from newsroom.control_plane.native_triage import (
        advance_native_triage, build_native_triage_work, plan_native_schedule,
    )

    monkeypatch.setattr('newsroom.control_plane.cycle._dispatch_rights_decision',
                        lambda *args, **kwargs: _current_rights())
    retrieval_authority, _ = _no_match_retrieval(tmp_path)
    contexts, receipts = {}, {}
    retrieval_authority._native_context_read_port = _read_port(contexts)
    collision = _native_collision(tmp_path, contexts, receipts)
    with sqlite3.connect(':memory:') as proving, _shared_system(
        tmp_path, monkeypatch, retrieval_authority, collision=collision.enforcer,
        candidate_citations=collision.candidate_citation_read_port(),
    ) as system:
        controller = _controller(system, proving)
        first_unit = _unit()
        same_unit = _same_state_revision(first_unit)
        candidate = associated = None
        for index, unit in enumerate((first_unit, same_unit)):
            _retain_source_revision(system, unit,
                prior_revision_id=None if index == 0 else first_unit.authority.revision_id)
            now = UtcTimestamp.parse(unit.effective_revision.first_observed_at)
            status = controller.admit_lead(controller.deliver(unit, now=now, proof=proof()),
                                           now=now, proof=proof())
            binding, receipt, context = _native_binding(tmp_path / f'retrieval-{index}', status.lead)
            contexts[context.context_id] = context
            receipts[receipt.event_id] = receipt
            arguments = dict(retrieval=SimpleNamespace(retrieve=lambda lead, *, proof: binding),
                collision_requests=collision, actor_identity_digest=_actor_digest(),
                proof=proof(), owner_stop_check=lambda: None, owner_stop_fence=nullcontext)
            if index == 1 and missing_original:
                def interrupted(*args, **kwargs):
                    raise RuntimeError('local fixture stopped before relationship commit')
                with monkeypatch.context() as crash:
                    crash.setattr(type(system.relationships), 'retain', interrupted)
                    with pytest.raises(RuntimeError, match='before relationship commit'):
                        advance_native_cycle(system, (status,), **arguments)
                continue
            outcome = advance_native_cycle(system, (status,), **arguments)[0]
            if index == 0:
                assert outcome.state == 'CANDIDATE_ADMITTED'
                candidate = outcome.triage.candidate
            else:
                assert outcome.state == 'SAME_STATE_ASSOCIATED'
                associated = outcome.triage
        store = system.candidates._StoryCandidateAuthority__authority._authority
        successor = system.hypotheses.current(candidate.governing_manifest.hypothesis_id, proof=proof())
        if missing_original:
            decision = None
            with pytest.raises(CandidateContractError, match='SAME_STATE relationship'):
                system.candidates._exact_associated_current_producers(candidate.candidate_id, proof=proof())
            assert store._connection.execute('SELECT decision_id FROM event_hypothesis_relationship_decisions '
                'WHERE subject_version_id=?', (successor.version_id,)).fetchone() is None
        else:
            current_candidate, current_hypothesis = system.candidates._exact_associated_current_producers(
                candidate.candidate_id, proof=proof())
            assert current_candidate == candidate and current_hypothesis == associated.hypothesis
            decision = associated.relationship.canonical_digest
            assert store._connection.execute('SELECT decision_id FROM event_hypothesis_relationship_decisions '
                'WHERE subject_version_id=?', (successor.version_id,)).fetchone()[0] == decision
        unrelated_unit = _unit(item_key='unrelated-current-rebuild-item')
        _retain_source_revision(system, unrelated_unit)
        now = UtcTimestamp.parse(unrelated_unit.effective_revision.first_observed_at)
        unrelated_status = controller.admit_lead(
            controller.deliver(unrelated_unit, now=now, proof=proof()), now=now, proof=proof())
        unrelated_binding, unrelated_receipt, unrelated_context = _native_binding(
            tmp_path / 'unrelated-retrieval', unrelated_status.lead)
        contexts[unrelated_context.context_id] = unrelated_context
        receipts[unrelated_receipt.event_id] = unrelated_receipt
        work = build_native_triage_work(admitted_leads=((unrelated_status.lead,
            unrelated_status.current_disposition),), retrieval=unrelated_binding)
        unrelated = advance_native_triage(system, work=work,
            scheduling_decision=plan_native_schedule(work).decision, proof=proof())
        assert unrelated.relationship is not None and unrelated.hypothesis.hypothesis_id != successor.hypothesis_id
        roots = {
            'story_candidate_admission_receipts_v2': tuple(tuple(row) for row in store._connection.execute(
                'SELECT admission_digest FROM story_candidate_admission_receipts_v2')),
            'story_candidate_heads': ((candidate.candidate_id,),),
            'story_candidate_collision_bindings': tuple(tuple(row) for row in store._connection.execute(
                'SELECT collision_namespace,collision_key_digest FROM story_candidate_collision_bindings')),
        }
        selected_root = tmp_path / 'selected'
        selected_root.mkdir(mode=0o700)
        selected_path = selected_root / 'native.sqlite3'
        before_decisions = tuple(tuple(row) for row in store._connection.execute(
            'SELECT * FROM event_hypothesis_relationship_decisions ORDER BY decision_id'))
        with sqlite3.connect(selected_path, isolation_level=None) as destination:
            initialise_empty_checkpoint_store(destination)
            copy_selected_native_store(store, destination, roots=roots, dev_rebuild=True)
            assert destination.execute('PRAGMA foreign_key_check').fetchall() == []
            assert destination.execute('SELECT version_id FROM event_hypothesis_heads_v2 '
                'WHERE hypothesis_id=?', (successor.hypothesis_id,)).fetchone()[0] == successor.version_id
            expected = None if missing_original else (decision,)
            assert destination.execute('SELECT decision_id FROM event_hypothesis_relationship_decisions '
                'WHERE subject_version_id=?', (successor.version_id,)).fetchone() == expected
            assert destination.execute('SELECT 1 FROM event_hypothesis_relationship_decisions '
                'WHERE decision_id=?', (unrelated.relationship.canonical_digest,)).fetchone() is None
            selected_decisions = destination.execute(
                'SELECT * FROM event_hypothesis_relationship_decisions ORDER BY decision_id').fetchall()
        with sqlite3.connect(tmp_path / 'selected-repeat.sqlite3', isolation_level=None) as repeated:
            initialise_empty_checkpoint_store(repeated)
            copy_selected_native_store(store, repeated, roots=roots, dev_rebuild=True)
            assert repeated.execute('SELECT * FROM event_hypothesis_relationship_decisions '
                'ORDER BY decision_id').fetchall() == selected_decisions
        assert tuple(tuple(row) for row in store._connection.execute(
            'SELECT * FROM event_hypothesis_relationship_decisions ORDER BY decision_id')) == before_decisions
        shutil.copytree(tmp_path / 'objects', selected_root / 'objects')
    selected_path.chmod(0o600)
    with _shared_system(selected_root, monkeypatch, retrieval_authority,
        collision=collision.enforcer, candidate_citations=collision.candidate_citation_read_port()) as rebuilt:
        if missing_original:
            with pytest.raises(CandidateContractError, match='SAME_STATE relationship'):
                rebuilt.candidates._exact_associated_current_producers(candidate.candidate_id, proof=proof())
        else:
            assert rebuilt.candidates._exact_associated_current_producers(candidate.candidate_id, proof=proof()) == (
                candidate, successor)
