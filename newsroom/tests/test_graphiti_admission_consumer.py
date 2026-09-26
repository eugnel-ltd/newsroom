from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.authority.types import UtcTimestamp
from newsroom.control_plane.graphiti_admission import (
    GRAPHITI_ADMISSION_EXTENSION_RECONCILIATION_SCHEMA_VERSION,
    GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION,
    GraphitiAdmissionConsumer,
    GraphitiAdmissionConsumerError,
    GraphitiAdmissionDrainReport,
    GraphitiGovernedDecision,
    GraphitiProposalAuthorityBinding,
    GraphitiProjectionGenerationResult,
    GraphitiProjectionReconciliationReceipt,
    GraphitiProjectionReceipt,
    GraphitiProposalAdmissionAction,
)
from newsroom.control_plane.store import (
    connect,
    insert_graphiti_attempt_receipt,
    insert_graphiti_ingest,
    retain_graphiti_authority_records,
)
from newsroom.extraction.models import ProposalDraft, ProposalEnvelope
from newsroom.graphiti_adapter.evaluation_packet import GRAPHITI_WORKSPACE_GROUP
from newsroom.extraction.types import (
    EvidenceRange,
    ExtractionPassageId,
    ExtractionProposalKind,
    ExtractionOutputId,
    ExtractionRunId,
    ExtractionRunVersionId,
    ProposalEnvelopeId,
    ProposalPredicateHint,
    ProposalSetId,
)
from newsroom.graphiti_adapter.identity import typed_id


NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
DIGEST_A = "sha256:" + ("a1" * 32)
DIGEST_B = "sha256:" + ("b2" * 32)
PROJECTION_GENERATION_ID = "00000000-0000-4000-8000-000000007589"
SECOND_PROJECTION_GENERATION_ID = "00000000-0000-4000-8000-000000007590"
SUBJECT_RESOLUTION_ID = "00000000-0000-4000-8000-000000007591"
OBJECT_RESOLUTION_ID = "00000000-0000-4000-8000-000000007592"


def _draft(
    local_id: str,
    kind: ExtractionProposalKind,
    *,
    subject: str = "Alice",
    object_: str = "Example Council",
) -> ProposalDraft:
    relation = kind is ExtractionProposalKind.RELATION
    return ProposalDraft(
        local_id=local_id,
        kind=kind,
        subject_placeholder=subject,
        object_placeholder=object_ if relation else None,
        predicate_hint=ProposalPredicateHint.SAME_PROCESS_AS if relation else None,
        confidence_basis_points=None,
        uncertainty_codes=("REQUIRES_RELATION_ADMISSION",) if relation else (),
        rationale_codes=("GRAPHITI_EVALUATION_SPAN",),
        evidence=(
            EvidenceRange(
                passage_id=ExtractionPassageId.parse(
                    "00000000-0000-4000-8000-000000007581"
                ),
                start_byte=0,
                end_byte=5,
                evidence_text_digest=DIGEST_A,
            ),
        ),
    )


def _binding(
    draft: ProposalDraft,
    *,
    cohort_seed: str | None = None,
) -> GraphitiProposalAuthorityBinding:
    shared = cohort_seed or draft.digest
    proposal_id = typed_id(ProposalEnvelopeId, "proposal", shared, draft.digest)
    proposal_set_id = typed_id(ProposalSetId, "proposal-set", shared)
    output_id = typed_id(ExtractionOutputId, "output", shared)
    run_id = typed_id(ExtractionRunId, "run", shared)
    run_version_id = typed_id(
        ExtractionRunVersionId, "run-version", shared
    )
    producer_digest = digest_canonical({"producer": "fixture"})
    canonical_digest = digest_canonical(
        {
            "proposal_id": str(proposal_id),
            "proposal_set_id": str(proposal_set_id),
            "output_id": str(output_id),
            "run_id": str(run_id),
            "run_version_id": str(run_version_id),
            "draft": draft.canonical_value(),
            "producer_contract_digest": producer_digest,
        }
    )
    return GraphitiProposalAuthorityBinding(
        graphiti_attempt_id=str(
            typed_id(ProposalEnvelopeId, "attempt-shaped", shared)
        ),
        graphiti_attempt_authority_event_id=str(
            typed_id(ProposalEnvelopeId, "event-shaped", shared)
        ),
        proposal_envelope=ProposalEnvelope(
            proposal_id=proposal_id,
            proposal_set_id=proposal_set_id,
            output_id=output_id,
            run_id=run_id,
            run_version_id=run_version_id,
            local_id=draft.local_id,
            kind=draft.kind,
            subject_placeholder=draft.subject_placeholder,
            object_placeholder=draft.object_placeholder,
            predicate_hint=draft.predicate_hint,
            confidence_basis_points=draft.confidence_basis_points,
            uncertainty_codes=draft.uncertainty_codes,
            rationale_codes=draft.rationale_codes,
            evidence=draft.evidence,
            producer_contract_digest=producer_digest,
            canonical_digest=canonical_digest,
            retained_at=UtcTimestamp.parse("2026-08-24T00:00:00Z"),
        ),
    )


class _ProposalAuthority:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available

    def bind_proposal(self, *, ingest_id, terminal_receipt, proposal):
        del terminal_receipt
        return (
            _binding(proposal, cohort_seed=ingest_id) if self.available else None
        )


def _relation_cohort(
    relation_local_id: str = "relation.0001",
) -> tuple[ProposalDraft, ProposalDraft, ProposalDraft]:
    return (
        _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION),
        _draft(
            "entity.0002",
            ExtractionProposalKind.ENTITY_MENTION,
            subject="Example Council",
        ),
        _draft(relation_local_id, ExtractionProposalKind.RELATION),
    )


def _seed_receipt(
    connection,
    *drafts: ProposalDraft,
    ingest_id: str = "sha256:" + ("75" * 32),
    missing_relation_temporal_field: str | None = None,
    passage_byte_length: int = 128,
    passage_text_digest: str = DIGEST_A,
) -> dict[str, object]:
    revision_id = "00000000-0000-4000-8000-000000007580"
    passage = {
        "passage_id": "00000000-0000-4000-8000-000000007581",
        "admission_id": "00000000-0000-4000-8000-000000007582",
        "access_decision_id": "00000000-0000-4000-8000-000000007583",
        "byte_offset": 0,
        "byte_length": passage_byte_length,
        "blob_digest": DIGEST_A,
        "text_digest": passage_text_digest,
    }
    records = (
        {
            "record_type": "SOURCE_REVISION",
            "record_id": revision_id,
            "source_id": "UK-01",
            "item_key": "item-758",
        },
        {
            "record_type": "OBJECT_ADMISSION",
            "record_id": passage["admission_id"],
            "revision_id": revision_id,
            "decision": "ADMIT",
        },
        {
            "record_type": "OBJECT_ACCESS_DECISION",
            "record_id": passage["access_decision_id"],
            "revision_id": revision_id,
            "decision": "ALLOW",
        },
    )
    retain_graphiti_authority_records(connection, records)
    entity_uuid_by_name = {
        item.subject_placeholder: f"private-{item.local_id}"
        for item in drafts
        if item.kind is ExtractionProposalKind.ENTITY_MENTION
    }
    relations = [
        {
            "local_id": item.local_id,
            "uuid": f"private-{item.local_id}",
            "name": "ABOUT_EVENT",
            "fact": "Alice briefed Example Council",
            "source_node_uuid": entity_uuid_by_name.get(
                item.subject_placeholder, "private-missing-source"
            ),
            "target_node_uuid": entity_uuid_by_name.get(
                str(item.object_placeholder), "private-missing-target"
            ),
            "valid_at": "2026-08-20T00:00:00Z",
            "invalid_at": None,
            "expired_at": None,
            "proposal_status": "PROPOSED",
        }
        for item in drafts
        if item.kind is ExtractionProposalKind.RELATION
    ]
    if missing_relation_temporal_field is not None:
        for relation in relations:
            relation.pop(missing_relation_temporal_field, None)
    entities = [
        {
            "local_id": item.local_id,
            "uuid": f"private-{item.local_id}",
            "name": item.subject_placeholder,
            "summary": "private Graphiti summary",
            "source_registry_id": False,
        }
        for item in drafts
        if item.kind is not ExtractionProposalKind.RELATION
    ]
    receipt: dict[str, object] = {
        "ingest_id": ingest_id,
        "source_id": "UK-01",
        "item_key": "item-758",
        "revision_id": revision_id,
        "outcome": "COMPLETE",
        "proposal_count": len(drafts),
        "entity_count": len(entities),
        "relation_count": len(relations),
        "failure_code": "NONE",
        "generation_id": "newsroom-eval-generation-758",
        "workspace_group": GRAPHITI_WORKSPACE_GROUP,
        "episode_uuid": ingest_id,
        "reference_time": "2026-08-20T00:00:00Z",
        "temporal_basis": "SOURCE_PUBLISHED",
        "authority_record_ids": [item["record_id"] for item in records],
        "proposals": [item.canonical_value() for item in drafts],
        "passages": [passage],
        "entities": entities,
        "relations": relations,
        "profile": "EVALUATION",
        "receipt_digest": "",
    }
    retained_digest = insert_graphiti_attempt_receipt(
        connection,
        ingest_id=ingest_id,
        attempt_number=1,
        outcome="COMPLETE",
        receipt=receipt,
    )
    receipt["receipt_digest"] = retained_digest
    assert insert_graphiti_ingest(
        connection,
        ingest_id=ingest_id,
        source_id="UK-01",
        item_key="item-758",
        outcome="COMPLETE",
        proposal_count=len(drafts),
        entity_count=int(receipt["entity_count"]),
        relation_count=len(relations),
        failure_code="NONE",
        temporal_basis="SOURCE_PUBLISHED",
        reference_time="2026-08-20T00:00:00Z",
        generation_id="newsroom-eval-generation-758",
        receipt_digest=retained_digest,
        receipt=receipt,
    )
    connection.commit()
    return receipt


class _Authority:
    def __init__(self, actions: dict[str, GraphitiProposalAdmissionAction]) -> None:
        self.actions = actions
        self.calls: list[tuple[object, object, str]] = []
        self.effects: dict[str, GraphitiGovernedDecision] = {}
        self.endpoints_current = True

    def _decide(self, request, *, required_action, idempotency_key):
        self.calls.append((request, required_action, idempotency_key))
        action = required_action or self.actions[request.proposal.local_id]
        decision = GraphitiGovernedDecision(
            proposal_key=request.proposal_key,
            proposal_digest=request.proposal.digest,
            proposal_kind=request.proposal.kind,
            proposal_local_id=request.proposal.local_id,
            action=action,
            decision_id=f"decision:{request.proposal_key}",
            authority_ledger_seq=request.queue_seq + 100,
            reason_code=(
                "RIGHTS_REVOKED" if required_action is not None else "FIXTURE_POLICY"
            ),
            authority_receipt_digest=DIGEST_A,
            admitted_authority_id=(
                str(
                    typed_id(
                        ProposalEnvelopeId,
                        "admitted-authority",
                        request.proposal_key,
                    )
                )
                if action is GraphitiProposalAdmissionAction.ADMIT
                else None
            ),
            endpoint_resolution_decision_ids=(
                (SUBJECT_RESOLUTION_ID, OBJECT_RESOLUTION_ID)
                if request.proposal.kind is ExtractionProposalKind.RELATION
                and action is GraphitiProposalAdmissionAction.ADMIT
                else ()
            ),
            resolved_endpoint_names=(
                request.proposed_endpoints
                if request.proposal.kind is ExtractionProposalKind.RELATION
                and action is GraphitiProposalAdmissionAction.ADMIT
                else ()
            ),
        )
        return self.effects.setdefault(idempotency_key, decision)

    def decide_entity_resolution(
        self, request, *, required_action, idempotency_key
    ):
        assert request.proposal.kind is not ExtractionProposalKind.RELATION
        return self._decide(
            request,
            required_action=required_action,
            idempotency_key=idempotency_key,
        )

    def decide_relation_admission(
        self, request, *, required_action, idempotency_key
    ):
        assert request.proposal.kind is ExtractionProposalKind.RELATION
        return self._decide(
            request,
            required_action=required_action,
            idempotency_key=idempotency_key,
        )

    def relation_endpoint_resolutions_current(self, request, decision):
        return (
            self.endpoints_current
            and decision.endpoint_resolution_decision_ids
            == (SUBJECT_RESOLUTION_ID, OBJECT_RESOLUTION_ID)
            and decision.resolved_endpoint_names == request.proposed_endpoints
        )


class _Projector:
    def __init__(self) -> None:
        self.deliveries: dict[str, GraphitiProjectionReceipt] = {}
        self.tombstones: dict[str, GraphitiProjectionReceipt] = {}
        self.raise_after_first_effect = False
        self.generation_calls: list[tuple[object, ...]] = []
        self.generation_effects: dict[str, GraphitiProjectionGenerationResult] = {}
        self.raise_after_first_generation = False

    def build_and_promote_increment4_cohort(
        self,
        requests,
        *,
        cohort_digest,
        generation_id,
        idempotency_key,
    ):
        self.generation_calls.append(tuple(requests))
        effect_ids = tuple(
            sorted(
                str(request.decision.admitted_authority_id)
                for request in requests
                if request.decision.action
                is GraphitiProposalAdmissionAction.ADMIT
            )
        )
        result = self.generation_effects.setdefault(
            idempotency_key,
            GraphitiProjectionGenerationResult(
                cohort_digest=cohort_digest,
                generation_id=generation_id,
                source_snapshot_digest=DIGEST_A,
                authority_watermark=max(
                    request.decision.authority_ledger_seq for request in requests
                ),
                validation_digest=DIGEST_A,
                promotion_digest=DIGEST_B,
                reconciliation_digest=digest_canonical(
                    {"generation_id": generation_id, "effects": effect_ids}
                ),
                admitted_authority_ids=effect_ids,
            ),
        )
        if self.raise_after_first_generation:
            self.raise_after_first_generation = False
            raise RuntimeError("crash after idempotent full generation promotion")
        return result

    def recover_increment4_admitted_receipt(self, *, idempotency_key):
        return self.deliveries.get(idempotency_key)

    def deliver_increment4_admitted(self, request, *, idempotency_key):
        receipt = self.deliveries.setdefault(
            idempotency_key,
            GraphitiProjectionReceipt(
                proposal_key=request.request.proposal_key,
                decision_id=request.decision.decision_id,
                effect_id=f"effect:{request.request.proposal_key}",
                authority_watermark=request.decision.authority_ledger_seq,
                receipt_digest=DIGEST_B,
                generation_id=PROJECTION_GENERATION_ID,
            ),
        )
        if self.raise_after_first_effect:
            self.raise_after_first_effect = False
            raise RuntimeError("crash after idempotent projection effect")
        return receipt

    def tombstone_increment4_admitted(self, request, *, idempotency_key):
        return self.tombstones.setdefault(
            idempotency_key,
            GraphitiProjectionReceipt(
                proposal_key=request.request.proposal_key,
                decision_id=request.decision.decision_id,
                effect_id=f"tombstone:{request.request.proposal_key}",
                authority_watermark=request.decision.authority_ledger_seq,
                receipt_digest=DIGEST_A,
                generation_id=PROJECTION_GENERATION_ID,
            ),
        )

    def reconcile_increment4_admitted(self, expected, *, generation_id):
        effect_ids = tuple(sorted(receipt.effect_id for receipt in expected))
        retained = (*expected, *self.tombstones.values())
        return GraphitiProjectionReconciliationReceipt(
            generation_id=generation_id,
            expected_effect_ids=effect_ids,
            actual_effect_ids=effect_ids,
            authority_watermark=max(receipt.authority_watermark for receipt in retained),
            receipt_digest=DIGEST_A,
        )


class _Rights:
    def __init__(self) -> None:
        self.allowed = True

    def is_current(self, request) -> bool:
        return self.allowed


def _consumer(
    connection,
    authority,
    projector,
    rights,
    *,
    proposal_authority=None,
    max_attempts=3,
    projection_generation_id=PROJECTION_GENERATION_ID,
):
    return GraphitiAdmissionConsumer(
        connection,
        proposal_authority=proposal_authority or _ProposalAuthority(),
        authority=authority,
        projector=projector,
        rights=rights,
        clock=lambda: NOW,
        max_attempts=max_attempts,
        projection_generation_id=projection_generation_id,
    )


def test_complete_receipts_map_exactly_and_only_admit_projects(tmp_path) -> None:
    connection = connect(str(tmp_path / "admission.sqlite3"))
    subject, object_, relation = _relation_cohort()
    source = _seed_receipt(connection, subject, object_, relation)
    authority = _Authority(
        {
            subject.local_id: GraphitiProposalAdmissionAction.ADMIT,
            object_.local_id: GraphitiProposalAdmissionAction.ADMIT,
            relation.local_id: GraphitiProposalAdmissionAction.HOLD,
        }
    )
    projector = _Projector()
    rights = _Rights()
    consumer = _consumer(connection, authority, projector, rights)

    assert consumer.enqueue_complete_receipts() == 3
    report = consumer.drain(worker_id="fixture-worker", limit=10)

    assert report.decided == 3
    assert report.projected == 0
    requests = [call[0] for call in authority.calls]
    relation_request = next(
        item for item in requests if item.proposal.kind is ExtractionProposalKind.RELATION
    )
    assert relation_request.proposal_payload == source["proposals"][2]
    assert relation_request.evidence_passages == tuple(source["passages"])
    assert relation_request.proposed_endpoints == ("Alice", "Example Council")
    assert relation_request.relation_statement == "Alice briefed Example Council"
    assert relation_request.relation_temporal_bounds == {
        "valid_at": "2026-08-20T00:00:00Z",
        "invalid_at": None,
        "expired_at": None,
    }
    canonical_request = json.dumps(relation_request.canonical_value())
    assert "private-entity.0001" not in canonical_request
    assert tuple(
        item.proposal_envelope.local_id
        for item in relation_request.relation_endpoint_bindings
    ) == (subject.local_id, object_.local_id)
    retained_request = json.loads(
        connection.execute(
            "SELECT request_json FROM unpublished_graphiti_admission_queue "
            "WHERE proposal_kind='RELATION'"
        ).fetchone()[0]
    )
    assert retained_request["private_graph_receipt"]["source_node_uuid"] == (
        "private-entity.0001"
    )
    assert relation_request.source_lineage["revision_id"] == source["revision_id"]
    projection = consumer.finalise_decided_cohort(
        ingest_ids=(str(source["ingest_id"]),)
    )
    assert projection.projected == 2
    assert len(projector.generation_calls) == 1
    assert len(projector.generation_calls[0]) == 3
    assert projector.deliveries == {}
    telemetry = consumer.telemetry()
    assert telemetry.proposal_denominator == 3
    assert telemetry.admitted_count == 2
    assert telemetry.held_count == 1
    assert telemetry.rejected_count == 0
    assert telemetry.admission_backlog == 0
    assert telemetry.contiguous_projection_watermark == 103
    assert consumer.telemetry().projection_reconciled is True


def test_local_entity_and_relation_evidence_spans_bind_and_admit(tmp_path) -> None:
    connection = connect(str(tmp_path / "local-evidence-spans.sqlite3"))
    passage = b"Before Alice briefed Example Council after the meeting."
    passage_id = ExtractionPassageId.parse(
        "00000000-0000-4000-8000-000000007581"
    )

    def with_span(
        draft: ProposalDraft,
        evidence: bytes,
    ) -> ProposalDraft:
        start = passage.index(evidence)
        return replace(
            draft,
            evidence=(
                EvidenceRange(
                    passage_id=passage_id,
                    start_byte=start,
                    end_byte=start + len(evidence),
                    evidence_text_digest=digest_bytes(evidence),
                ),
            ),
        )

    subject = with_span(
        _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION),
        b"Alice",
    )
    object_ = with_span(
        _draft(
            "entity.0002",
            ExtractionProposalKind.ENTITY_MENTION,
            subject="Example Council",
        ),
        b"Example Council",
    )
    relation = with_span(
        _draft("relation.0001", ExtractionProposalKind.RELATION),
        b"Alice briefed Example Council",
    )
    source = _seed_receipt(
        connection,
        subject,
        object_,
        relation,
        passage_byte_length=len(passage),
        passage_text_digest=digest_bytes(passage),
    )
    assert all(
        draft.evidence[0].evidence_text_digest != digest_bytes(passage)
        for draft in (subject, object_, relation)
    )
    authority = _Authority(
        {
            subject.local_id: GraphitiProposalAdmissionAction.ADMIT,
            object_.local_id: GraphitiProposalAdmissionAction.ADMIT,
            relation.local_id: GraphitiProposalAdmissionAction.ADMIT,
        }
    )
    consumer = _consumer(connection, authority, _Projector(), _Rights())

    assert consumer.enqueue_complete_receipts(
        ingest_ids=(str(source["ingest_id"]),)
    ) == 3
    report = consumer.drain(
        worker_id="fixture-worker",
        limit=3,
        ingest_ids=(str(source["ingest_id"]),),
    )

    assert report.decided == 3
    assert [call[0].proposal.local_id for call in authority.calls] == [
        subject.local_id,
        object_.local_id,
        relation.local_id,
    ]
    connection.close()


def test_non_empty_all_hold_cohort_still_promotes_one_full_snapshot(tmp_path) -> None:
    from newsroom.control_plane.graphiti_admission import (
        GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION,
        graphiti_decided_cohort_generation_identity,
    )
    from newsroom.control_plane.graphiti_steady_state import (
        _exact_admission_reconciliation,
    )

    connection = connect(str(tmp_path / "all-hold-generation.sqlite3"))
    ingest_id = "00000000-0000-4000-8000-0000000075d0"
    draft = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    _seed_receipt(connection, draft, ingest_id=ingest_id)
    authority = _Authority(
        {draft.local_id: GraphitiProposalAdmissionAction.HOLD}
    )
    projector = _Projector()
    consumer = _consumer(connection, authority, projector, _Rights())

    assert consumer.enqueue_complete_receipts(ingest_ids=(ingest_id,)) == 1
    assert consumer.drain(
        worker_id="fixture-worker",
        limit=1,
        ingest_ids=(ingest_id,),
    ).decided == 1

    projection = consumer.finalise_decided_cohort(ingest_ids=(ingest_id,))

    assert projection.projected == 0
    assert len(projector.generation_calls) == 1
    assert len(projector.generation_calls[0]) == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_graphiti_projection_receipts"
    ).fetchone() == (0,)
    reconciliation = connection.execute(
        "SELECT generation_id,receipt_json FROM "
        "unpublished_graphiti_projection_reconciliations"
    ).fetchone()
    assert reconciliation is not None
    cohort_digest, generation_id = graphiti_decided_cohort_generation_identity(
        connection,
        ingest_ids=(ingest_id,),
    )
    assert generation_id == reconciliation[0]
    reconciliation_binding = json.loads(reconciliation[1])
    assert reconciliation_binding["schema_version"] == (
        GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION
    )
    assert reconciliation_binding["cohort_digest"] == cohort_digest
    assert reconciliation_binding["ingest_ids"] == [ingest_id]
    assert reconciliation_binding["raw_receipt"]["generation_id"] == generation_id
    assert reconciliation_binding["raw_receipt"]["expected_effect_ids"] == []
    assert reconciliation_binding["raw_receipt"]["actual_effect_ids"] == []
    exact_reconciliation = _exact_admission_reconciliation(connection)
    assert exact_reconciliation["total"] is True
    assert exact_reconciliation["disjoint"] is True
    assert exact_reconciliation["cohort_count"] == 1
    assert exact_reconciliation["cohorts"][0]["cohort_digest"] == cohort_digest
    assert cohort_digest.startswith("sha256:")
    assert consumer.finalise_decided_cohort(
        ingest_ids=(ingest_id,)
    ) == GraphitiAdmissionDrainReport()
    assert len(projector.generation_calls) == 2
    connection.execute(
        "UPDATE unpublished_graphiti_admission_queue SET state='READY'"
    )
    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="decision integrity differs",
    ):
        graphiti_decided_cohort_generation_identity(
            connection,
            ingest_ids=(ingest_id,),
            require_terminal_states=True,
        )
    connection.close()


def test_missing_relation_temporal_field_is_not_rewritten_as_null(tmp_path) -> None:
    connection = connect(str(tmp_path / "partial-temporal.sqlite3"))
    subject, object_, relation = _relation_cohort()
    source = _seed_receipt(
        connection,
        subject,
        object_,
        relation,
        missing_relation_temporal_field="invalid_at",
    )
    authority = _Authority(
        {
            subject.local_id: GraphitiProposalAdmissionAction.ADMIT,
            object_.local_id: GraphitiProposalAdmissionAction.ADMIT,
            relation.local_id: GraphitiProposalAdmissionAction.HOLD,
        }
    )
    consumer = _consumer(connection, authority, _Projector(), _Rights())

    assert consumer.enqueue_complete_receipts(
        ingest_ids=(str(source["ingest_id"]),)
    ) == 3
    report = consumer.drain(
        worker_id="fixture-worker",
        limit=10,
        ingest_ids=(str(source["ingest_id"]),),
    )

    assert report.decided == 3
    assert report.projected == 0
    relation_request = next(
        call[0]
        for call in authority.calls
        if call[0].proposal.kind is ExtractionProposalKind.RELATION
    )
    assert relation_request.relation_temporal_bounds == {
        "valid_at": "2026-08-20T00:00:00Z",
        "expired_at": None,
    }
    assert connection.execute(
        "SELECT state FROM unpublished_graphiti_admission_queue "
        "WHERE proposal_kind='RELATION'"
    ).fetchone() == ("TERMINAL",)
    connection.close()


def test_exact_mentions_are_decided_before_equivalence_hold(tmp_path) -> None:
    connection = connect(str(tmp_path / "equivalence-order.sqlite3"))
    first = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    second = replace(
        _draft(
            "entity.0002",
            ExtractionProposalKind.ENTITY_MENTION,
            subject="Example Council",
        ),
        evidence=(
            EvidenceRange(
                passage_id=ExtractionPassageId.parse(
                    "00000000-0000-4000-8000-000000007581"
                ),
                start_byte=6,
                end_byte=21,
                evidence_text_digest=DIGEST_A,
            ),
        ),
    )
    equivalence = ProposalDraft(
        local_id="equivalence.0001",
        kind=ExtractionProposalKind.ENTITY_EQUIVALENCE,
        subject_placeholder="Alice",
        object_placeholder="Example Council",
        predicate_hint=None,
        confidence_basis_points=None,
        uncertainty_codes=("AMBIGUOUS_IDENTITY",),
        rationale_codes=("GRAPHITI_EVALUATION_SPAN",),
        evidence=(*first.evidence, *second.evidence),
    )
    source = _seed_receipt(connection, equivalence, second, first)
    authority = _Authority(
        {
            first.local_id: GraphitiProposalAdmissionAction.ADMIT,
            second.local_id: GraphitiProposalAdmissionAction.ADMIT,
            equivalence.local_id: GraphitiProposalAdmissionAction.HOLD,
        }
    )
    consumer = _consumer(connection, authority, _Projector(), _Rights())

    assert consumer.enqueue_complete_receipts(
        ingest_ids=(str(source["ingest_id"]),)
    ) == 3
    report = consumer.drain(
        worker_id="fixture-worker",
        limit=10,
        ingest_ids=(str(source["ingest_id"]),),
    )

    assert report.decided == 3
    assert [call[0].proposal.kind for call in authority.calls] == [
        ExtractionProposalKind.ENTITY_MENTION,
        ExtractionProposalKind.ENTITY_MENTION,
        ExtractionProposalKind.ENTITY_EQUIVALENCE,
    ]
    equivalence_request = authority.calls[-1][0]
    assert tuple(
        item.proposal_envelope.local_id
        for item in equivalence_request.relation_endpoint_bindings
    ) == (first.local_id, second.local_id)
    connection.close()


def test_exact_ingest_cohort_never_enqueues_or_drains_other_receipts(tmp_path) -> None:
    connection = connect(str(tmp_path / "admission.sqlite3"))
    first_id = "00000000-0000-4000-8000-0000000075a1"
    second_id = "00000000-0000-4000-8000-0000000075a2"
    first = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    second = _draft("entity.0002", ExtractionProposalKind.ENTITY_MENTION)
    _seed_receipt(connection, first, ingest_id=first_id)
    _seed_receipt(connection, second, ingest_id=second_id)
    authority = _Authority(
        {
            first.local_id: GraphitiProposalAdmissionAction.REJECT,
            second.local_id: GraphitiProposalAdmissionAction.REJECT,
        }
    )
    consumer = _consumer(connection, authority, _Projector(), _Rights())

    assert consumer.enqueue_complete_receipts(ingest_ids=(second_id,)) == 1
    assert connection.execute(
        "SELECT ingest_id FROM unpublished_graphiti_admission_queue"
    ).fetchall() == [(second_id,)]
    assert consumer.enqueue_complete_receipts(ingest_ids=(first_id,)) == 1

    report = consumer.drain(
        worker_id="fixture-worker",
        limit=10,
        ingest_ids=(second_id,),
    )

    assert report.claimed == 1
    assert report.decided == 1
    assert connection.execute(
        "SELECT ingest_id,state FROM unpublished_graphiti_admission_queue "
        "ORDER BY ingest_id"
    ).fetchall() == [(first_id, "READY"), (second_id, "TERMINAL")]
    assert [call[0].source_lineage["ingest_id"] for call in authority.calls] == [
        second_id
    ]


@pytest.mark.parametrize(
    "ingest_ids",
    ((), ("b", "a"), ("a", "a"), ("",)),
)
def test_exact_ingest_cohort_must_be_nonempty_sorted_and_unique(
    tmp_path,
    ingest_ids,
) -> None:
    connection = connect(str(tmp_path / "admission.sqlite3"))
    consumer = _consumer(connection, _Authority({}), _Projector(), _Rights())

    with pytest.raises(ValueError, match="exact Graphiti admission ingest"):
        consumer.enqueue_complete_receipts(ingest_ids=ingest_ids)
    with pytest.raises(ValueError, match="exact Graphiti admission ingest"):
        consumer.drain(worker_id="fixture-worker", ingest_ids=ingest_ids)
    assert consumer.telemetry().projection_gap_count == 0
    connection.close()


def test_exact_ingest_cohort_distinguishes_zero_proposals_from_missing(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "admission.sqlite3"))
    zero_id = "00000000-0000-4000-8000-0000000075b1"
    _seed_receipt(connection, ingest_id=zero_id)
    consumer = _consumer(connection, _Authority({}), _Projector(), _Rights())

    assert consumer.enqueue_complete_receipts(ingest_ids=(zero_id,)) == 0
    with pytest.raises(GraphitiAdmissionConsumerError, match="missing or non-terminal"):
        consumer.enqueue_complete_receipts(
            ingest_ids=("00000000-0000-4000-8000-0000000075b2",)
        )

    connection.close()


def test_integrity_invalid_terminal_receipt_is_never_claimable_or_silent(tmp_path) -> None:
    connection = connect(str(tmp_path / "invalid.sqlite3"))
    draft = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, draft)
    tampered = dict(receipt)
    tampered["proposals"] = [
        {**draft.canonical_value(), "subject_placeholder": "Mallory"}
    ]
    connection.execute(
        "UPDATE unpublished_graphiti_receipts SET receipt_json=? WHERE ingest_id=?",
        (json.dumps(tampered, sort_keys=True), receipt["ingest_id"]),
    )
    connection.commit()
    consumer = _consumer(connection, _Authority({}), _Projector(), _Rights())

    assert consumer.enqueue_complete_receipts() == 0
    assert consumer.drain(worker_id="fixture-worker", limit=10).claimed == 0
    telemetry = consumer.telemetry()
    assert telemetry.proposal_denominator == 1
    assert telemetry.admission_backlog == 1
    assert telemetry.integrity_hold_receipt_count == 1
    connection.close()


def test_exact_ingest_cohort_rejects_invalid_receipt_after_recording_failure(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "invalid-exact.sqlite3"))
    ingest_id = "00000000-0000-4000-8000-0000000075b3"
    draft = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, draft, ingest_id=ingest_id)
    tampered = dict(receipt)
    tampered["proposals"] = [
        {**draft.canonical_value(), "subject_placeholder": "Mallory"}
    ]
    connection.execute(
        "UPDATE unpublished_graphiti_receipts SET receipt_json=? WHERE ingest_id=?",
        (json.dumps(tampered, sort_keys=True), ingest_id),
    )
    connection.commit()
    consumer = _consumer(connection, _Authority({}), _Projector(), _Rights())

    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="exact Graphiti admission receipt is invalid",
    ):
        consumer.enqueue_complete_receipts(ingest_ids=(ingest_id,))

    failure = connection.execute(
        "SELECT ingest_id,receipt_digest FROM "
        "unpublished_graphiti_admission_receipt_failures"
    ).fetchone()
    assert failure == (ingest_id, receipt["receipt_digest"])
    connection.close()


def test_exact_ingest_cohort_validates_every_receipt_before_enqueuing(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "invalid-exact-batch.sqlite3"))
    valid_id = "00000000-0000-4000-8000-0000000075b4"
    invalid_id = "00000000-0000-4000-8000-0000000075b5"
    valid = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    invalid = _draft("entity.0002", ExtractionProposalKind.ENTITY_MENTION)
    _seed_receipt(connection, valid, ingest_id=valid_id)
    receipt = _seed_receipt(connection, invalid, ingest_id=invalid_id)
    tampered = dict(receipt)
    tampered["proposals"] = [
        {**invalid.canonical_value(), "subject_placeholder": "Mallory"}
    ]
    connection.execute(
        "UPDATE unpublished_graphiti_receipts SET receipt_json=? WHERE ingest_id=?",
        (json.dumps(tampered, sort_keys=True), invalid_id),
    )
    connection.commit()
    consumer = _consumer(connection, _Authority({}), _Projector(), _Rights())

    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="exact Graphiti admission receipt is invalid",
    ):
        consumer.enqueue_complete_receipts(ingest_ids=(valid_id, invalid_id))

    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_graphiti_admission_queue"
    ).fetchone() == (0,)
    assert connection.execute(
        "SELECT ingest_id FROM unpublished_graphiti_admission_receipt_failures"
    ).fetchall() == [(invalid_id,)]
    connection.close()


def test_restart_reuses_projection_idempotency_key_without_duplicate_effect(tmp_path) -> None:
    connection = connect(str(tmp_path / "restart.sqlite3"))
    draft = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, draft)
    authority = _Authority(
        {draft.local_id: GraphitiProposalAdmissionAction.ADMIT}
    )
    projector = _Projector()
    projector.raise_after_first_generation = True
    consumer = _consumer(connection, authority, projector, _Rights())
    consumer.enqueue_complete_receipts()

    decision = consumer.drain(worker_id="worker-a", limit=1)
    with pytest.raises(RuntimeError, match="idempotent full generation"):
        consumer.finalise_decided_cohort(
            ingest_ids=(str(receipt["ingest_id"]),)
        )
    second = consumer.finalise_decided_cohort(
        ingest_ids=(str(receipt["ingest_id"]),)
    )
    third = consumer.finalise_decided_cohort(
        ingest_ids=(str(receipt["ingest_id"]),)
    )

    assert decision.decided == 1
    assert second.projected == 1
    assert third.projected == 0
    assert len(projector.generation_calls) == 3
    assert len(projector.generation_effects) == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_graphiti_projection_receipts"
    ).fetchone() == (1,)
    assert consumer.telemetry().admission_backlog == 0
    connection.close()


def test_retained_generation_rechecks_active_graph_before_returning(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "retained-generation-recheck.sqlite3"))
    ingest_id = "00000000-0000-4000-8000-0000000075d4"
    draft = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    _seed_receipt(connection, draft, ingest_id=ingest_id)
    authority = _Authority(
        {draft.local_id: GraphitiProposalAdmissionAction.ADMIT}
    )
    first_projector = _Projector()
    first = _consumer(connection, authority, first_projector, _Rights())
    assert first.enqueue_complete_receipts(ingest_ids=(ingest_id,)) == 1
    assert first.drain(
        worker_id="worker-a",
        limit=1,
        ingest_ids=(ingest_id,),
    ).decided == 1
    assert first.finalise_decided_cohort(
        ingest_ids=(ingest_id,)
    ).projected == 1
    retained = connection.execute(
        "SELECT receipt_json,receipt_digest FROM "
        "unpublished_graphiti_projection_receipts"
    ).fetchall()

    class MissingActiveGraph(_Projector):
        def build_and_promote_increment4_cohort(self, *args, **kwargs):
            raise RuntimeError("active graph generation is unavailable")

    restarted = _consumer(
        connection,
        authority,
        MissingActiveGraph(),
        _Rights(),
    )
    with pytest.raises(RuntimeError, match="active graph generation"):
        restarted.finalise_decided_cohort(ingest_ids=(ingest_id,))

    assert connection.execute(
        "SELECT receipt_json,receipt_digest FROM "
        "unpublished_graphiti_projection_receipts"
    ).fetchall() == retained
    connection.close()


def test_rights_revocation_rejects_unadmitted_and_blocks_generation(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "rights.sqlite3"))
    admitted = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, admitted)
    authority = _Authority(
        {admitted.local_id: GraphitiProposalAdmissionAction.ADMIT}
    )
    projector = _Projector()
    rights = _Rights()
    consumer = _consumer(connection, authority, projector, rights)
    consumer.enqueue_complete_receipts()
    assert consumer.drain(worker_id="worker-a", limit=1).decided == 1

    rights.allowed = False
    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="lost current rights",
    ):
        consumer.finalise_decided_cohort(
            ingest_ids=(str(receipt["ingest_id"]),)
        )

    assert projector.generation_calls == []
    assert consumer.telemetry().revoked_count == 1
    assert consumer.telemetry().projection_reconciled is False

    second = replace(admitted, local_id="entity.0002")
    second_connection = connect(str(tmp_path / "rights-before.sqlite3"))
    second_receipt = _seed_receipt(second_connection, second)
    second_authority = _Authority(
        {second.local_id: GraphitiProposalAdmissionAction.ADMIT}
    )
    second_projector = _Projector()
    second_consumer = _consumer(
        second_connection, second_authority, second_projector, rights
    )
    second_consumer.enqueue_complete_receipts()
    report = second_consumer.drain(worker_id="worker-b", limit=1)
    assert report.decided == 1
    assert report.projected == 0
    assert second_consumer.finalise_decided_cohort(
        ingest_ids=(str(second_receipt["ingest_id"]),)
    ).projected == 0
    assert second_authority.calls[0][1] is GraphitiProposalAdmissionAction.REJECT
    assert len(second_projector.generation_calls) == 1
    assert len(second_projector.generation_calls[0]) == 1
    second_connection.close()
    connection.close()


def test_repeated_authority_failure_dead_letters_without_blocking_enqueue(tmp_path) -> None:
    connection = connect(str(tmp_path / "dead-letter.sqlite3"))
    first = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    _seed_receipt(connection, first)

    class BrokenAuthority(_Authority):
        def _decide(self, request, *, required_action, idempotency_key):
            raise RuntimeError("governed authority unavailable")

    consumer = _consumer(
        connection, BrokenAuthority({}), _Projector(), _Rights(), max_attempts=2
    )
    assert consumer.enqueue_complete_receipts() == 1
    assert consumer.drain(worker_id="worker-a", limit=1).failed == 1
    assert consumer.drain(worker_id="worker-b", limit=1).dead_lettered == 1
    telemetry = consumer.telemetry()
    assert telemetry.dead_letter_count == 1
    assert telemetry.admission_backlog == 1
    connection.close()


def test_exact_campaign_stops_after_first_authority_failure(tmp_path) -> None:
    connection = connect(str(tmp_path / "stop-on-failure.sqlite3"))
    first = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    second = _draft("entity.0002", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, first, second)

    class BrokenAuthority(_Authority):
        def _decide(self, request, *, required_action, idempotency_key):
            raise RuntimeError("governed authority unavailable")

    authority = BrokenAuthority({})
    consumer = _consumer(
        connection, authority, _Projector(), _Rights(), max_attempts=3
    )
    assert consumer.enqueue_complete_receipts() == 2

    report = consumer.drain(
        worker_id="campaign-worker",
        limit=2,
        ingest_ids=(str(receipt["ingest_id"]),),
        stop_on_failure=True,
    )

    assert report.claimed == 1
    assert report.failed == 1
    assert report.dead_lettered == 0
    assert connection.execute(
        "SELECT state,attempt_count FROM unpublished_graphiti_admission_queue "
        "ORDER BY queue_seq"
    ).fetchall() == [("READY", 1), ("READY", 0)]
    connection.close()


def test_exact_campaign_rights_drift_stops_before_canonical_decision(tmp_path) -> None:
    connection = connect(str(tmp_path / "rights-stop.sqlite3"))
    first = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    second = _draft("entity.0002", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, first, second)
    authority = _Authority(
        {
            first.local_id: GraphitiProposalAdmissionAction.ADMIT,
            second.local_id: GraphitiProposalAdmissionAction.ADMIT,
        }
    )
    rights = _Rights()
    rights.allowed = False
    consumer = _consumer(connection, authority, _Projector(), rights)
    assert consumer.enqueue_complete_receipts() == 2

    report = consumer.drain(
        worker_id="campaign-worker",
        limit=2,
        ingest_ids=(str(receipt["ingest_id"]),),
        stop_on_failure=True,
    )

    assert report.claimed == 1
    assert report.failed == 1
    assert report.decided == 0
    assert authority.calls == []
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_graphiti_admission_decisions"
    ).fetchone() == (0,)
    assert connection.execute(
        "SELECT state,attempt_count FROM unpublished_graphiti_admission_queue "
        "ORDER BY queue_seq"
    ).fetchall() == [("READY", 1), ("READY", 0)]
    connection.close()


def test_malformed_claim_is_dead_lettered_without_blocking_later_work(tmp_path) -> None:
    connection = connect(str(tmp_path / "malformed-claim.sqlite3"))
    first = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    second = _draft("entity.0002", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, first, second)
    authority = _Authority(
        {
            first.local_id: GraphitiProposalAdmissionAction.ADMIT,
            second.local_id: GraphitiProposalAdmissionAction.ADMIT,
        }
    )
    consumer = _consumer(connection, authority, _Projector(), _Rights())
    assert consumer.enqueue_complete_receipts() == 2
    connection.execute(
        "UPDATE unpublished_graphiti_admission_queue SET request_json='{}' "
        "WHERE queue_seq=(SELECT MIN(queue_seq) "
        "FROM unpublished_graphiti_admission_queue)",
    )
    connection.commit()

    report = consumer.drain(worker_id="worker-a", limit=1)

    assert report.claimed == 1
    assert report.decided == 1
    assert report.projected == 0
    assert authority.calls[0][0].proposal.local_id == second.local_id
    assert consumer.telemetry().dead_letter_count == 1
    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="retained integrity failures",
    ):
        consumer.finalise_decided_cohort(
            ingest_ids=(str(receipt["ingest_id"]),)
        )
    connection.close()


def test_cross_proposal_decision_identity_collision_never_marks_projected(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "decision-collision.sqlite3"))
    first = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    second = _draft("entity.0002", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, first, second)

    class CollidingAuthority(_Authority):
        def _decide(self, request, *, required_action, idempotency_key):
            decision = super()._decide(
                request,
                required_action=required_action,
                idempotency_key=idempotency_key,
            )
            return replace(decision, decision_id="decision:collision")

    consumer = _consumer(
        connection,
        CollidingAuthority(
            {
                first.local_id: GraphitiProposalAdmissionAction.ADMIT,
                second.local_id: GraphitiProposalAdmissionAction.ADMIT,
            }
        ),
        _Projector(),
        _Rights(),
    )
    consumer.enqueue_complete_receipts()

    report = consumer.drain(worker_id="worker-a", limit=2)

    assert report.decided == 1
    assert report.projected == 0
    assert report.failed == 1
    states = dict(
        connection.execute(
            "SELECT proposal_key, state FROM unpublished_graphiti_admission_queue"
        )
    )
    assert sum(state == "DECIDED" for state in states.values()) == 1
    assert sum(state == "READY" for state in states.values()) == 1
    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="not completely decided",
    ):
        consumer.finalise_decided_cohort(
            ingest_ids=(str(receipt["ingest_id"]),)
        )
    connection.close()


def test_cross_proposal_canonical_authority_identity_collision_never_projects(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "projection-collision.sqlite3"))
    first = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    second = _draft("entity.0002", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, first, second)

    class CollidingAuthority(_Authority):
        def _decide(self, request, *, required_action, idempotency_key):
            decision = super()._decide(
                request,
                required_action=required_action,
                idempotency_key=idempotency_key,
            )
            return replace(
                decision,
                admitted_authority_id=(
                    "00000000-0000-4000-8000-000000007599"
                ),
            )

    projector = _Projector()
    consumer = _consumer(
        connection,
        CollidingAuthority(
            {
                first.local_id: GraphitiProposalAdmissionAction.ADMIT,
                second.local_id: GraphitiProposalAdmissionAction.ADMIT,
            }
        ),
        projector,
        _Rights(),
    )
    consumer.enqueue_complete_receipts()

    report = consumer.drain(worker_id="worker-a", limit=2)

    assert report.decided == 2
    assert report.projected == 0
    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="reuses canonical admitted authority identity",
    ):
        consumer.finalise_decided_cohort(
            ingest_ids=(str(receipt["ingest_id"]),)
        )
    states = dict(
        connection.execute(
            "SELECT proposal_key, state FROM unpublished_graphiti_admission_queue"
        )
    )
    assert set(states.values()) == {"DECIDED"}
    assert projector.generation_calls == []
    connection.close()


def test_receipt_count_mismatch_is_an_explicit_integrity_hold(tmp_path) -> None:
    connection = connect(str(tmp_path / "count-mismatch.sqlite3"))
    draft = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, draft)
    connection.execute(
        "UPDATE unpublished_graphiti_ingest SET entity_count=99 WHERE ingest_id=?",
        (receipt["ingest_id"],),
    )
    connection.commit()
    consumer = _consumer(connection, _Authority({}), _Projector(), _Rights())

    assert consumer.enqueue_complete_receipts() == 0
    assert consumer.telemetry().integrity_hold_receipt_count == 1
    connection.close()


def test_generation_failure_retains_no_partial_admitted_effect(tmp_path) -> None:
    connection = connect(str(tmp_path / "projection-gap.sqlite3"))
    first = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    second = _draft("entity.0002", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, first, second)

    class FailingGenerationProjector(_Projector):
        def build_and_promote_increment4_cohort(self, *args, **kwargs):
            raise RuntimeError("fixture full generation failure")

    consumer = _consumer(
        connection,
        _Authority(
            {
                first.local_id: GraphitiProposalAdmissionAction.ADMIT,
                second.local_id: GraphitiProposalAdmissionAction.ADMIT,
            }
        ),
        FailingGenerationProjector(),
        _Rights(),
    )
    consumer.enqueue_complete_receipts()
    assert consumer.drain(worker_id="worker-a", limit=2).decided == 2

    with pytest.raises(RuntimeError, match="full generation failure"):
        consumer.finalise_decided_cohort(
            ingest_ids=(str(receipt["ingest_id"]),)
        )
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_graphiti_projection_receipts"
    ).fetchone() == (0,)
    assert connection.execute(
        "SELECT DISTINCT state FROM unpublished_graphiti_admission_queue"
    ).fetchall() == [("DECIDED",)]
    assert consumer.telemetry().projection_gap_count == 2
    assert consumer.telemetry().projection_reconciled is False
    connection.close()


def test_generation_bound_receipts_never_use_legacy_tombstone_path(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "generation-rights.sqlite3"))
    first = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    second = _draft("entity.0002", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, first, second)

    projector = _Projector()
    rights = _Rights()
    consumer = _consumer(
        connection,
        _Authority(
            {
                first.local_id: GraphitiProposalAdmissionAction.ADMIT,
                second.local_id: GraphitiProposalAdmissionAction.ADMIT,
            }
        ),
        projector,
        rights,
    )
    consumer.enqueue_complete_receipts()
    assert consumer.drain(worker_id="worker-a", limit=2).decided == 2
    assert consumer.finalise_decided_cohort(
        ingest_ids=(str(receipt["ingest_id"]),)
    ).projected == 2
    rights.allowed = False

    assert consumer.reconcile_rights(limit=2) == 0
    assert projector.tombstones == {}
    assert connection.execute(
        "SELECT DISTINCT state FROM unpublished_graphiti_admission_queue"
    ).fetchall() == [("PROJECTED",)]
    connection.close()


def test_passage_rights_ids_must_bind_exact_retained_authority(tmp_path) -> None:
    connection = connect(str(tmp_path / "passage-rights.sqlite3"))
    draft = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, draft)
    tampered = dict(receipt)
    tampered["passages"] = [
        {**receipt["passages"][0], "admission_id": "bogus-admission"}
    ]
    unsigned = dict(tampered)
    unsigned.pop("receipt_digest")
    tampered_digest = digest_bytes(canonical_json_bytes(unsigned))
    tampered["receipt_digest"] = tampered_digest
    connection.execute(
        "UPDATE unpublished_graphiti_ingest SET receipt_digest=? WHERE ingest_id=?",
        (tampered_digest, receipt["ingest_id"]),
    )
    connection.execute(
        "UPDATE unpublished_graphiti_receipts SET receipt_json=? WHERE ingest_id=?",
        (json.dumps(tampered, sort_keys=True), receipt["ingest_id"]),
    )
    connection.commit()
    consumer = _consumer(connection, _Authority({}), _Projector(), _Rights())

    assert consumer.enqueue_complete_receipts() == 0
    assert consumer.telemetry().integrity_hold_receipt_count == 1
    connection.close()


def test_generation_cannot_precede_the_cohort_authority_watermark(tmp_path) -> None:
    connection = connect(str(tmp_path / "watermark.sqlite3"))
    draft = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, draft)

    class StaleSnapshotProjector(_Projector):
        def build_and_promote_increment4_cohort(self, *args, **kwargs):
            result = super().build_and_promote_increment4_cohort(
                *args, **kwargs
            )
            return replace(result, authority_watermark=1)

    consumer = _consumer(
        connection,
        _Authority({draft.local_id: GraphitiProposalAdmissionAction.ADMIT}),
        StaleSnapshotProjector(),
        _Rights(),
    )
    consumer.enqueue_complete_receipts()

    assert consumer.drain(worker_id="worker-a", limit=1).decided == 1

    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="differs from exact admission authority",
    ):
        consumer.finalise_decided_cohort(
            ingest_ids=(str(receipt["ingest_id"]),)
        )
    assert consumer.telemetry().contiguous_projection_watermark is None
    connection.close()


def test_relation_admit_requires_current_endpoint_authority(tmp_path) -> None:
    connection = connect(str(tmp_path / "endpoint-authority.sqlite3"))
    subject, object_, relation = _relation_cohort()
    _seed_receipt(connection, subject, object_, relation)

    class StaleEndpointAuthority(_Authority):
        def relation_endpoint_resolutions_current(self, request, decision):
            return False

    consumer = _consumer(
        connection,
        StaleEndpointAuthority(
            {
                subject.local_id: GraphitiProposalAdmissionAction.ADMIT,
                object_.local_id: GraphitiProposalAdmissionAction.ADMIT,
                relation.local_id: GraphitiProposalAdmissionAction.ADMIT,
            }
        ),
        _Projector(),
        _Rights(),
    )
    consumer.enqueue_complete_receipts()

    report = consumer.drain(worker_id="worker-a", limit=3)

    assert report.failed == 1
    assert report.projected == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_graphiti_admission_decisions AS decision "
        "JOIN unpublished_graphiti_admission_queue AS queue USING(proposal_key) "
        "WHERE queue.proposal_kind='RELATION'"
    ).fetchone() == (0,)
    connection.close()


def test_relation_projection_retry_rechecks_endpoint_authority(tmp_path) -> None:
    connection = connect(str(tmp_path / "endpoint-retry.sqlite3"))
    subject, object_, relation = _relation_cohort()
    receipt = _seed_receipt(connection, subject, object_, relation)
    authority = _Authority(
        {
            subject.local_id: GraphitiProposalAdmissionAction.ADMIT,
            object_.local_id: GraphitiProposalAdmissionAction.ADMIT,
            relation.local_id: GraphitiProposalAdmissionAction.ADMIT,
        }
    )
    projector = _Projector()
    projector.raise_after_first_generation = True
    consumer = _consumer(connection, authority, projector, _Rights())
    consumer.enqueue_complete_receipts()
    assert consumer.drain(worker_id="worker-a", limit=10).decided == 3
    with pytest.raises(RuntimeError, match="idempotent full generation"):
        consumer.finalise_decided_cohort(
            ingest_ids=(str(receipt["ingest_id"]),)
        )
    authority.endpoints_current = False

    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="lost current rights or endpoint authority",
    ):
        consumer.finalise_decided_cohort(
            ingest_ids=(str(receipt["ingest_id"]),)
        )

    assert consumer.telemetry().projected_count == 0
    assert consumer.telemetry().revoked_count == 1
    assert len(projector.generation_calls) == 1
    assert projector.tombstones == {}
    connection.close()


def test_finalisation_rejects_cross_generation_result(tmp_path) -> None:
    connection = connect(str(tmp_path / "generation-drift.sqlite3"))
    draft = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, draft)
    authority = _Authority(
        {draft.local_id: GraphitiProposalAdmissionAction.ADMIT}
    )
    class DriftingGenerationProjector(_Projector):
        def build_and_promote_increment4_cohort(self, *args, **kwargs):
            result = super().build_and_promote_increment4_cohort(
                *args, **kwargs
            )
            return replace(
                result,
                generation_id=SECOND_PROJECTION_GENERATION_ID,
            )

    projector = DriftingGenerationProjector()
    consumer = _consumer(connection, authority, projector, _Rights())
    consumer.enqueue_complete_receipts()
    assert consumer.drain(worker_id="worker-a", limit=1).decided == 1

    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="differs from exact admission authority",
    ):
        consumer.finalise_decided_cohort(
            ingest_ids=(str(receipt["ingest_id"]),)
        )
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_graphiti_projection_receipts"
    ).fetchone() == (0,)
    connection.close()


def test_active_generation_extension_retains_each_exact_cohort_and_replays(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "active-extension.sqlite3"))
    first_id = "00000000-0000-4000-8000-0000000075e1"
    second_id = "00000000-0000-4000-8000-0000000075e2"
    first = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    second = _draft(
        "entity.0002",
        ExtractionProposalKind.ENTITY_MENTION,
        subject="Bob",
    )
    _seed_receipt(connection, first, ingest_id=first_id)
    _seed_receipt(connection, second, ingest_id=second_id)

    class StableGenerationProjector(_Projector):
        def build_and_promote_increment4_cohort(
            self, requests, *, cohort_digest, generation_id, idempotency_key
        ):
            self.generation_calls.append(tuple(requests))
            effect_ids = tuple(
                sorted(
                    str(item.decision.admitted_authority_id)
                    for item in requests
                    if item.decision.action
                    is GraphitiProposalAdmissionAction.ADMIT
                )
            )
            return self.generation_effects.setdefault(
                idempotency_key,
                GraphitiProjectionGenerationResult(
                    cohort_digest=cohort_digest,
                    generation_id=PROJECTION_GENERATION_ID,
                    source_snapshot_digest=digest_canonical(
                        {"snapshot": cohort_digest}
                    ),
                    authority_watermark=max(
                        item.decision.authority_ledger_seq for item in requests
                    ),
                    validation_digest=digest_canonical(
                        {"validation": cohort_digest}
                    ),
                    promotion_digest=DIGEST_B,
                    reconciliation_digest=digest_canonical(
                        {"active": PROJECTION_GENERATION_ID, "cohort": cohort_digest}
                    ),
                    admitted_authority_ids=effect_ids,
                    schema_version=GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION,
                    source_request_digest=digest_canonical(
                        {
                            "generation_id": generation_id,
                            "reason_code": "GRAPHITI_ADMISSION_COHORT",
                            "idempotency_key": idempotency_key,
                            "purge_retired_generation": True,
                            "allow_active_extension": True,
                        }
                    ),
                ),
            )

    projector = StableGenerationProjector()
    consumer = _consumer(
        connection,
        _Authority(
            {
                first.local_id: GraphitiProposalAdmissionAction.ADMIT,
                second.local_id: GraphitiProposalAdmissionAction.ADMIT,
            }
        ),
        projector,
        _Rights(),
    )
    for ingest_id in (first_id, second_id):
        assert consumer.enqueue_complete_receipts(ingest_ids=(ingest_id,)) == 1
        assert consumer.drain(
            worker_id="worker-a", limit=1, ingest_ids=(ingest_id,)
        ).decided == 1
        assert consumer.finalise_decided_cohort(
            ingest_ids=(ingest_id,)
        ).projected == 1

    rows = connection.execute(
        "SELECT generation_id,receipt_json FROM "
        "unpublished_graphiti_projection_reconciliations ORDER BY reconciled_at"
    ).fetchall()
    assert [row[0] for row in rows] == [
        PROJECTION_GENERATION_ID,
        PROJECTION_GENERATION_ID,
    ]
    assert {
        json.loads(row[1])["schema_version"] for row in rows
    } == {GRAPHITI_ADMISSION_EXTENSION_RECONCILIATION_SCHEMA_VERSION}
    assert connection.execute(
        "SELECT DISTINCT schema_version FROM "
        "unpublished_graphiti_projection_receipts"
    ).fetchall() == [(GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION,)]
    assert consumer.telemetry().projection_reconciled is True

    retained = tuple(rows)
    assert consumer.finalise_decided_cohort(
        ingest_ids=(first_id,)
    ) == GraphitiAdmissionDrainReport()
    assert connection.execute(
        "SELECT generation_id,receipt_json FROM "
        "unpublished_graphiti_projection_reconciliations ORDER BY reconciled_at"
    ).fetchall() == list(retained)
    first_key = next(iter(projector.generation_effects))
    projector.generation_effects[first_key] = replace(
        projector.generation_effects[first_key],
        source_request_digest=DIGEST_A,
    )
    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="differs from exact admission authority",
    ):
        consumer.finalise_decided_cohort(ingest_ids=(first_id,))
    assert connection.execute(
        "SELECT generation_id,receipt_json FROM "
        "unpublished_graphiti_projection_reconciliations ORDER BY reconciled_at"
    ).fetchall() == list(retained)
    connection.close()


def test_invalid_generation_result_retains_no_receipts_and_fails_closed(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "invalid-generation-result.sqlite3"))
    draft = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, draft)

    class MismatchedMembershipProjector(_Projector):
        def build_and_promote_increment4_cohort(self, *args, **kwargs):
            result = super().build_and_promote_increment4_cohort(
                *args, **kwargs
            )
            return replace(
                result,
                admitted_authority_ids=(
                    "00000000-0000-4000-8000-0000000075ff",
                ),
            )

    consumer = _consumer(
        connection,
        _Authority({draft.local_id: GraphitiProposalAdmissionAction.ADMIT}),
        MismatchedMembershipProjector(),
        _Rights(),
    )
    consumer.enqueue_complete_receipts()
    assert consumer.drain(worker_id="worker-a", limit=1).decided == 1

    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="differs from exact admission authority",
    ):
        consumer.finalise_decided_cohort(
            ingest_ids=(str(receipt["ingest_id"]),)
        )
    assert connection.execute(
        "SELECT state FROM unpublished_graphiti_admission_queue"
    ).fetchone() == ("DECIDED",)
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_graphiti_projection_receipts"
    ).fetchone() == (0,)
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_graphiti_projection_reconciliations"
    ).fetchone() == (0,)
    connection.close()


def test_active_extension_watermark_before_decision_retains_nothing(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "extension-watermark.sqlite3"))
    draft = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, draft)

    class StaleExtensionProjector(_Projector):
        def build_and_promote_increment4_cohort(self, *args, **kwargs):
            result = super().build_and_promote_increment4_cohort(*args, **kwargs)
            return replace(
                result,
                generation_id=SECOND_PROJECTION_GENERATION_ID,
                authority_watermark=result.authority_watermark - 1,
                schema_version=GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION,
                source_request_digest=digest_canonical(
                    {
                        "generation_id": kwargs["generation_id"],
                        "reason_code": "GRAPHITI_ADMISSION_COHORT",
                        "idempotency_key": kwargs["idempotency_key"],
                        "purge_retired_generation": True,
                        "allow_active_extension": True,
                    }
                ),
            )

    consumer = _consumer(
        connection,
        _Authority({draft.local_id: GraphitiProposalAdmissionAction.ADMIT}),
        StaleExtensionProjector(),
        _Rights(),
    )
    consumer.enqueue_complete_receipts()
    assert consumer.drain(worker_id="worker-a", limit=1).decided == 1

    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="differs from exact admission authority",
    ):
        consumer.finalise_decided_cohort(
            ingest_ids=(str(receipt["ingest_id"]),)
        )
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_graphiti_projection_receipts"
    ).fetchone() == (0,)
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_graphiti_projection_reconciliations"
    ).fetchone() == (0,)
    connection.close()


def test_partial_projection_retention_blocks_generation_replay(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "partial-generation-retention.sqlite3"))
    subject, object_, relation = _relation_cohort()
    source = _seed_receipt(connection, subject, object_, relation)
    authority = _Authority(
        {
            subject.local_id: GraphitiProposalAdmissionAction.ADMIT,
            object_.local_id: GraphitiProposalAdmissionAction.ADMIT,
            relation.local_id: GraphitiProposalAdmissionAction.ADMIT,
        }
    )
    projector = _Projector()
    consumer = _consumer(connection, authority, projector, _Rights())
    consumer.enqueue_complete_receipts()
    assert consumer.drain(worker_id="worker-a", limit=10).decided == 3
    proposal_key, decision_id, authority_watermark = connection.execute(
        """
        SELECT queue.proposal_key, decision.decision_id,
               decision.authority_ledger_seq
        FROM unpublished_graphiti_admission_queue AS queue
        JOIN unpublished_graphiti_admission_decisions AS decision
          USING(proposal_key)
        """
    ).fetchone()
    stale_receipt = GraphitiProjectionReceipt(
        proposal_key=str(proposal_key),
        decision_id=str(decision_id),
        effect_id="stale-independent-effect",
        authority_watermark=int(authority_watermark),
        receipt_digest=DIGEST_A,
        generation_id=PROJECTION_GENERATION_ID,
    )
    connection.execute(
        """
        INSERT INTO unpublished_graphiti_projection_receipts(
            proposal_key, effect_id, authority_watermark,
            projector_family_id, generation_id, schema_version,
            trust_scope, receipt_json, receipt_digest, projected_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?)
        """,
        (
            proposal_key,
            stale_receipt.effect_id,
            stale_receipt.authority_watermark,
            stale_receipt.projector_family_id,
            stale_receipt.generation_id,
            stale_receipt.schema_version,
            stale_receipt.trust_scope,
            canonical_json_bytes(stale_receipt.canonical_value()).decode("utf-8"),
            stale_receipt.receipt_digest,
            "2026-08-24T12:00:00Z",
        ),
    )
    connection.commit()

    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="generation retention is partial",
    ):
        consumer.finalise_decided_cohort(
            ingest_ids=(str(source["ingest_id"]),)
        )
    assert projector.generation_calls == []
    connection.close()


def test_raw_only_historical_receipt_stays_outside_governed_admission(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "historical.sqlite3"))
    ingest_id = "00000000-0000-4000-8000-0000000075d1"
    draft = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    _seed_receipt(connection, draft, ingest_id=ingest_id)
    consumer = _consumer(
        connection,
        _Authority({}),
        _Projector(),
        _Rights(),
        proposal_authority=_ProposalAuthority(available=False),
    )

    assert consumer.enqueue_complete_receipts() == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_graphiti_admission_queue"
    ).fetchone() == (0,)
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_graphiti_admission_receipt_failures"
    ).fetchone() == (0,)
    with pytest.raises(
        GraphitiAdmissionConsumerError,
        match="durable ProposalEnvelope authority",
    ):
        consumer.enqueue_complete_receipts(ingest_ids=(ingest_id,))
    connection.close()


@pytest.mark.parametrize("state", ("READY", "CLAIMED"))
def test_legacy_queued_request_without_envelope_remains_immutable(
    tmp_path,
    state: str,
) -> None:
    connection = connect(str(tmp_path / f"legacy-queue-{state}.sqlite3"))
    ingest_id = "00000000-0000-4000-8000-0000000075d3"
    draft = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    _seed_receipt(connection, draft, ingest_id=ingest_id)
    consumer = _consumer(
        connection,
        _Authority({draft.local_id: GraphitiProposalAdmissionAction.ADMIT}),
        _Projector(),
        _Rights(),
    )
    assert consumer.enqueue_complete_receipts(ingest_ids=(ingest_id,)) == 1
    proposal_key, retained = connection.execute(
        "SELECT proposal_key,request_json "
        "FROM unpublished_graphiti_admission_queue"
    ).fetchone()
    legacy = json.loads(retained)
    legacy.pop("proposal_authority_binding")
    legacy_bytes = canonical_json_bytes(legacy)
    connection.execute(
        "UPDATE unpublished_graphiti_admission_queue "
        "SET request_json=?,request_digest=?,state=?,attempt_count=1,"
        "last_error='retained historical state',claim_owner=?,claim_until=? "
        "WHERE proposal_key=?",
        (
            legacy_bytes.decode("utf-8"),
            digest_bytes(legacy_bytes),
            state,
            "old-worker" if state == "CLAIMED" else None,
            "2026-08-24T11:00:00Z" if state == "CLAIMED" else None,
            proposal_key,
        ),
    )
    connection.commit()
    before = connection.execute(
        "SELECT state,attempt_count,last_error,claim_owner,claim_until,"
        "request_json,request_digest "
        "FROM unpublished_graphiti_admission_queue WHERE proposal_key=?",
        (proposal_key,),
    ).fetchone()
    failure_count = connection.execute(
        "SELECT COUNT(*) FROM unpublished_graphiti_admission_receipt_failures"
    ).fetchone()

    assert consumer.drain(
        worker_id="fixture-worker",
        limit=1,
        ingest_ids=(ingest_id,),
    ) == GraphitiAdmissionDrainReport()

    assert connection.execute(
        "SELECT state,attempt_count,last_error,claim_owner,claim_until,"
        "request_json,request_digest "
        "FROM unpublished_graphiti_admission_queue WHERE proposal_key=?",
        (proposal_key,),
    ).fetchone() == before
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_graphiti_admission_receipt_failures"
    ).fetchone() == failure_count
    connection.close()


def test_exact_cohort_decides_entities_before_one_generation_promotion(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "cohort-generation.sqlite3"))
    ingest_id = "00000000-0000-4000-8000-0000000075d2"
    subject, object_, relation = _relation_cohort()
    _seed_receipt(connection, relation, object_, subject, ingest_id=ingest_id)
    authority = _Authority(
        {
            relation.local_id: GraphitiProposalAdmissionAction.ADMIT,
            subject.local_id: GraphitiProposalAdmissionAction.ADMIT,
            object_.local_id: GraphitiProposalAdmissionAction.ADMIT,
        }
    )
    projector = _Projector()
    consumer = _consumer(connection, authority, projector, _Rights())

    assert consumer.enqueue_complete_receipts(ingest_ids=(ingest_id,)) == 3
    decisions = consumer.drain(
        worker_id="fixture-worker",
        limit=10,
        ingest_ids=(ingest_id,),
    )

    assert decisions.decided == 3
    assert decisions.projected == 0
    assert [call[0].proposal.kind for call in authority.calls] == [
        ExtractionProposalKind.ENTITY_MENTION,
        ExtractionProposalKind.ENTITY_MENTION,
        ExtractionProposalKind.RELATION,
    ]

    projection = consumer.finalise_decided_cohort(ingest_ids=(ingest_id,))

    assert projection.projected == 3
    assert len(projector.generation_calls) == 1
    retained = connection.execute(
        "SELECT generation_id,receipt_json "
        "FROM unpublished_graphiti_projection_receipts ORDER BY proposal_key"
    ).fetchall()
    assert len(retained) == 3
    assert len({row[0] for row in retained}) == 1
    payloads = [json.loads(row[1]) for row in retained]
    assert len({item["cohort_digest"] for item in payloads}) == 1
    assert all(
        item["schema_version"]
        == "newsroom.increment4.admitted-generation-binding.v2"
        for item in payloads
    )
    connection.close()
