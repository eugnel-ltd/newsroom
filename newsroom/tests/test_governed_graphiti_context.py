from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.authority.types import UtcTimestamp
from newsroom.control_plane.cycle import _dispatch_writer
from newsroom.control_plane.editorial import GroupedObservation, form_candidates
from newsroom.control_plane.evidence import package_for
from newsroom.control_plane.governed_context import (
    ADMITTED_CONTEXT_TRUST_LABEL,
    AuthorityContextBinding,
    GovernedAuthorityContext,
    GovernedContext,
    GovernedContextHydrator,
    GovernedContextStatus,
)
from newsroom.control_plane.graphiti_admission import (
    GRAPHITI_ADMISSION_EXTENSION_RECONCILIATION_SCHEMA_VERSION,
    GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION,
    GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION,
    GraphitiAdmissionRequest,
    GraphitiGovernedDecision,
    GraphitiProposalAuthorityBinding,
    GraphitiProjectionReceipt,
    graphiti_admission_generation_identity,
)
from newsroom.control_plane.items import SourceItem
from newsroom.control_plane.store import connect
from newsroom.control_plane.writer import CliChainWriter, WriterDispatchError
from newsroom.extraction.models import ProposalDraft, ProposalEnvelope
from newsroom.extraction.types import (
    EvidenceRange,
    ExtractionPassageId,
    ExtractionProposalKind,
    ExtractionOutputId,
    ExtractionRunId,
    ExtractionRunVersionId,
    ProposalEnvelopeId,
    ProposalSetId,
)
from newsroom.graphiti_adapter.admission import GraphitiProposalAdmissionAction
from newsroom.graphiti_adapter.identity import typed_id

from .test_graphiti_admission_consumer import (
    _Authority as _AdmissionAuthority,
    _Projector as _AdmissionProjector,
    _Rights as _AdmissionRights,
    _consumer as _admission_consumer,
    _draft as _admission_draft,
    _seed_receipt as _seed_admission_receipt,
)

NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
DIGEST_A = "sha256:" + ("a1" * 32)
DIGEST_B = "sha256:" + ("b2" * 32)
DIGEST_C = "sha256:" + ("c3" * 32)
GENERATION_ID = "00000000-0000-4000-8000-000000007599"


def _binding(draft: ProposalDraft) -> GraphitiProposalAuthorityBinding:
    values = {
        "proposal_id": typed_id(ProposalEnvelopeId, "proposal", draft.digest),
        "proposal_set_id": typed_id(ProposalSetId, "set", draft.digest),
        "output_id": typed_id(ExtractionOutputId, "output", draft.digest),
        "run_id": typed_id(ExtractionRunId, "run", draft.digest),
        "run_version_id": typed_id(ExtractionRunVersionId, "version", draft.digest),
    }
    producer = digest_canonical({"producer": "fixture"})
    envelope_digest = digest_canonical({
        **{key: str(value) for key, value in values.items()},
        "draft": draft.canonical_value(),
        "producer_contract_digest": producer,
    })
    return GraphitiProposalAuthorityBinding(
        graphiti_attempt_id=str(typed_id(ProposalEnvelopeId, "attempt", draft.digest)),
        graphiti_attempt_authority_event_id=str(typed_id(ProposalEnvelopeId, "event", draft.digest)),
        proposal_envelope=ProposalEnvelope(
            **values, local_id=draft.local_id, kind=draft.kind,
            subject_placeholder=draft.subject_placeholder,
            object_placeholder=draft.object_placeholder,
            predicate_hint=draft.predicate_hint,
            confidence_basis_points=draft.confidence_basis_points,
            uncertainty_codes=draft.uncertainty_codes,
            rationale_codes=draft.rationale_codes, evidence=draft.evidence,
            producer_contract_digest=producer, canonical_digest=envelope_digest,
            retained_at=UtcTimestamp.parse("2026-08-24T00:00:00Z"),
        ),
    )


class _CurrentAuthority:
    def current_context(self, request, decision):
        assert decision.action is GraphitiProposalAdmissionAction.ADMIT
        return GovernedAuthorityContext(
            bindings=(
                AuthorityContextBinding(
                    authority_kind="CANONICAL_ENTITY",
                    authority_id="00000000-0000-4000-8000-000000007501",
                    authority_version="00000000-0000-4000-8000-000000007502",
                ),
                AuthorityContextBinding(
                    authority_kind="ENTITY_RESOLUTION_DECISION",
                    authority_id=decision.decision_id,
                    authority_version="1",
                ),
            ),
            admitted_temporal_fields=(("observed_at", "2026-08-20T00:00:00Z"),),
            currentness_state="CURRENT",
            admitted_structured_value_json=canonical_json_bytes(
                {
                    "authority_kind": "CANONICAL_ENTITY",
                    "entity": {"display_name": "Alice Example"},
                }
            ).decode(),
        )


class _Rights:
    current = True

    def is_current(self, request):
        return self.current


def _seed_entity(
    connection,
    *,
    action: GraphitiProposalAdmissionAction = GraphitiProposalAdmissionAction.ADMIT,
    reconcile: bool = True,
    exact_binding: bool = False,
    active_extension: bool = False,
) -> str:
    if active_extension and not exact_binding:
        raise ValueError("active extension fixture requires exact binding")
    generation_id = GENERATION_ID
    terminal_receipt = {
        "ingest_id": DIGEST_A,
        "outcome": "COMPLETE",
        "proposal_count": 1,
    }
    terminal_receipt_digest = digest_canonical(terminal_receipt)
    terminal_receipt["receipt_digest"] = terminal_receipt_digest
    source_receipt_digest = terminal_receipt_digest if exact_binding else DIGEST_A
    proposal = ProposalDraft(
        local_id="entity.0001",
        kind=ExtractionProposalKind.ENTITY_MENTION,
        subject_placeholder="Alice",
        object_placeholder=None,
        predicate_hint=None,
        confidence_basis_points=None,
        uncertainty_codes=(),
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
    request = GraphitiAdmissionRequest(
        queue_seq=1,
        proposal_key=DIGEST_B,
        source_receipt_digest=source_receipt_digest,
        proposal_authority_binding=_binding(proposal),
        proposal=proposal,
        proposal_payload=proposal.canonical_value(),
        evidence_passages=(
            {
                "passage_id": "00000000-0000-4000-8000-000000007581",
                "admission_id": "00000000-0000-4000-8000-000000007582",
                "access_decision_id": "00000000-0000-4000-8000-000000007583",
                "byte_offset": 0,
                "byte_length": 128,
                "blob_digest": DIGEST_B,
                "text_digest": DIGEST_B,
            },
        ),
        proposed_endpoints=None,
        relation_statement=None,
        relation_temporal_bounds=None,
        source_lineage={
            "ingest_id": DIGEST_A,
            "source_id": "UK-01",
            "item_key": "item-759",
            "revision_id": "00000000-0000-4000-8000-000000007580",
            "authority_record_ids": [
                "00000000-0000-4000-8000-000000007580",
                "00000000-0000-4000-8000-000000007582",
                "00000000-0000-4000-8000-000000007583",
            ],
            "generation_id": "newsroom-eval-generation-759",
            "episode_uuid": DIGEST_A,
            "reference_time": "2026-08-20T00:00:00Z",
            "temporal_basis": "SOURCE_PUBLISHED",
        },
    )
    decision = GraphitiGovernedDecision(
        proposal_key=request.proposal_key,
        proposal_digest=proposal.digest,
        proposal_kind=proposal.kind,
        proposal_local_id=proposal.local_id,
        action=action,
        decision_id="decision:entity.0001",
        authority_ledger_seq=101,
        reason_code="FIXTURE_POLICY",
        authority_receipt_digest=DIGEST_A,
        admitted_authority_id=(
            "00000000-0000-4000-8000-000000007501"
            if action is GraphitiProposalAdmissionAction.ADMIT
            else None
        ),
    )
    request_json = canonical_json_bytes(request.canonical_value()).decode()
    decision_json = canonical_json_bytes(decision.canonical_value()).decode()
    connection.execute(
        "INSERT INTO unpublished_graphiti_ingest VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            DIGEST_A,
            "UK-01",
            "item-759",
            "COMPLETE",
            1,
            1,
            0,
            "NONE",
            "SOURCE_PUBLISHED",
            "2026-08-20T00:00:00Z",
            "newsroom-eval-generation-759",
            source_receipt_digest,
            "2026-08-24T11:59:00Z",
        ),
    )
    if exact_binding:
        connection.execute(
            "INSERT INTO unpublished_graphiti_receipts VALUES(?,?)",
            (DIGEST_A, canonical_json_bytes(terminal_receipt).decode()),
        )
    connection.execute(
        """
        INSERT INTO unpublished_graphiti_admission_queue(
            queue_seq, proposal_key, ingest_id, source_revision_id,
            source_receipt_digest, proposal_digest, proposal_kind, request_json,
            request_digest, state, created_at, updated_at
        ) VALUES(1,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            request.proposal_key,
            DIGEST_A,
            request.source_lineage["revision_id"],
            source_receipt_digest,
            proposal.digest,
            proposal.kind.value,
            request_json,
            digest_bytes(request_json.encode()),
            (
                "PROJECTED"
                if action is GraphitiProposalAdmissionAction.ADMIT
                else "TERMINAL"
            ),
            "2026-08-24T11:59:00Z",
            "2026-08-24T11:59:00Z",
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_admission_decisions VALUES(?,?,?,?,?,?,?,?,?)",
        (
            request.proposal_key,
            decision.action.value,
            decision.decision_id,
            decision.authority_ledger_seq,
            decision.reason_code,
            decision.authority_receipt_digest,
            decision_json,
            digest_bytes(decision_json.encode()),
            "2026-08-24T11:59:00Z",
        ),
    )
    if action is GraphitiProposalAdmissionAction.ADMIT:
        cohort_digest = None
        if exact_binding:
            cohort_digest, derived_generation_id = graphiti_admission_generation_identity(
                ingest_ids=(DIGEST_A,),
                source_receipts=(
                    {
                        "ingest_id": DIGEST_A,
                        "receipt_digest": source_receipt_digest,
                        "proposal_count": 1,
                    },
                ),
                members=(
                    {
                        "ingest_id": DIGEST_A,
                        "proposal_key": request.proposal_key,
                        "proposal_envelope_id": str(
                            request.proposal_authority_binding.proposal_envelope.proposal_id
                        ),
                        "decision_digest": digest_bytes(decision_json.encode()),
                        "decision": decision.canonical_value(),
                    },
                ),
            )
            generation_id = (
                GENERATION_ID if active_extension else derived_generation_id
            )
        generation_result_digest = (
            digest_canonical(
                {"generation_id": generation_id, "cohort_digest": cohort_digest}
            )
            if active_extension
            else DIGEST_A
        )
        projection = GraphitiProjectionReceipt(
            proposal_key=request.proposal_key,
            decision_id=decision.decision_id,
            effect_id="effect:entity.0001",
            authority_watermark=101,
            receipt_digest=DIGEST_B,
            generation_id=generation_id,
            schema_version=(
                GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION
                if active_extension
                else "newsroom.increment4.admitted-generation-binding.v2"
                if exact_binding
                else "newsroom.increment4.admitted-projection.v1"
            ),
            cohort_digest=cohort_digest,
            source_snapshot_digest=DIGEST_A if exact_binding else None,
            validation_digest=DIGEST_B if exact_binding else None,
            promotion_digest=DIGEST_C if exact_binding else None,
            generation_result_digest=(
                generation_result_digest if exact_binding else None
            ),
            source_request_digest=(
                digest_canonical({"request": cohort_digest})
                if active_extension
                else None
            ),
        )
        if exact_binding:
            projection_material = projection.canonical_value()
            projection_material.pop("receipt_digest")
            projection = replace(
                projection,
                receipt_digest=digest_canonical(projection_material),
            )
        projection_json = canonical_json_bytes(projection.canonical_value()).decode()
        connection.execute(
            "INSERT INTO unpublished_graphiti_projection_receipts "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                request.proposal_key,
                projection.effect_id,
                projection.authority_watermark,
                projection.projector_family_id,
                projection.generation_id,
                projection.schema_version,
                projection.trust_scope,
                projection_json,
                projection.receipt_digest,
                "2026-08-24T11:59:00Z",
            ),
        )
        reconciliation = {
            "generation_id": generation_id,
            "expected_effect_ids": [projection.effect_id],
            "actual_effect_ids": [projection.effect_id],
            "authority_watermark": 101,
            "receipt_digest": generation_result_digest,
            "projector_family_id": projection.projector_family_id,
            "provider_model_calls": 0,
        }
        if reconcile:
            reconciliation_value = (
                {
                    "schema_version": (
                        GRAPHITI_ADMISSION_EXTENSION_RECONCILIATION_SCHEMA_VERSION
                        if active_extension
                        else GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION
                    ),
                    "cohort_digest": cohort_digest,
                    "ingest_ids": [DIGEST_A],
                    "raw_receipt": reconciliation,
                }
                if exact_binding
                else reconciliation
            )
            if active_extension:
                reconciliation_value.update(
                    {
                        "source_snapshot_digest": DIGEST_A,
                        "source_watermark_ledger_seq": 101,
                        "source_request_digest": projection.source_request_digest,
                        "validation_digest": DIGEST_B,
                        "promotion_digest": DIGEST_C,
                        "generation_result_digest": generation_result_digest,
                        "projection_state_digest": digest_canonical(
                            {"projection": cohort_digest}
                        ),
                    }
                )
            connection.execute(
                "INSERT INTO unpublished_graphiti_projection_reconciliations "
                "VALUES(?,?,?,?,?,?)",
                (
                    reconciliation["receipt_digest"],
                    projection.projector_family_id,
                    generation_id,
                    101,
                    canonical_json_bytes(reconciliation_value).decode(),
                    "2026-08-24T11:59:00Z",
                ),
            )
    connection.commit()
    return generation_id


def _seed_exact_all_hold_cohort(
    connection, *, active_generation_id: str | None = None
) -> str:
    ingest_id = "sha256:" + ("e5" * 32)
    proposal_key = "sha256:" + ("d4" * 32)
    terminal_receipt = {
        "ingest_id": ingest_id,
        "outcome": "COMPLETE",
        "proposal_count": 1,
    }
    terminal_receipt_digest = digest_canonical(terminal_receipt)
    terminal_receipt["receipt_digest"] = terminal_receipt_digest
    proposal = ProposalDraft(
        local_id="entity.0002",
        kind=ExtractionProposalKind.ENTITY_MENTION,
        subject_placeholder="Bob",
        object_placeholder=None,
        predicate_hint=None,
        confidence_basis_points=None,
        uncertainty_codes=(),
        rationale_codes=("GRAPHITI_EVALUATION_SPAN",),
        evidence=(
            EvidenceRange(
                passage_id=ExtractionPassageId.parse(
                    "00000000-0000-4000-8000-000000007591"
                ),
                start_byte=0,
                end_byte=3,
                evidence_text_digest=DIGEST_C,
            ),
        ),
    )
    request = GraphitiAdmissionRequest(
        queue_seq=2,
        proposal_key=proposal_key,
        source_receipt_digest=terminal_receipt_digest,
        proposal_authority_binding=_binding(proposal),
        proposal=proposal,
        proposal_payload=proposal.canonical_value(),
        evidence_passages=(
            {
                "passage_id": "00000000-0000-4000-8000-000000007591",
                "admission_id": "00000000-0000-4000-8000-000000007592",
                "access_decision_id": "00000000-0000-4000-8000-000000007593",
                "byte_offset": 0,
                "byte_length": 128,
                "blob_digest": DIGEST_C,
                "text_digest": DIGEST_C,
            },
        ),
        proposed_endpoints=None,
        relation_statement=None,
        relation_temporal_bounds=None,
        source_lineage={
            "ingest_id": ingest_id,
            "source_id": "UK-01",
            "item_key": "item-760",
            "revision_id": "00000000-0000-4000-8000-000000007590",
            "authority_record_ids": [
                "00000000-0000-4000-8000-000000007590",
                "00000000-0000-4000-8000-000000007592",
                "00000000-0000-4000-8000-000000007593",
            ],
            "generation_id": "newsroom-eval-generation-760",
            "episode_uuid": DIGEST_C,
            "reference_time": "2026-08-21T00:00:00Z",
            "temporal_basis": "SOURCE_PUBLISHED",
        },
    )
    decision = GraphitiGovernedDecision(
        proposal_key=proposal_key,
        proposal_digest=proposal.digest,
        proposal_kind=proposal.kind,
        proposal_local_id=proposal.local_id,
        action=GraphitiProposalAdmissionAction.HOLD,
        decision_id="decision:entity.0002",
        authority_ledger_seq=102,
        reason_code="AMBIGUOUS_ENTITY_IDENTITY",
        authority_receipt_digest=DIGEST_C,
    )
    request_json = canonical_json_bytes(request.canonical_value()).decode()
    decision_json = canonical_json_bytes(decision.canonical_value()).decode()
    connection.execute(
        "INSERT INTO unpublished_graphiti_ingest VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            ingest_id,
            "UK-01",
            "item-760",
            "COMPLETE",
            1,
            0,
            0,
            "NONE",
            "SOURCE_PUBLISHED",
            "2026-08-21T00:00:00Z",
            "newsroom-eval-generation-760",
            terminal_receipt_digest,
            "2026-08-24T11:59:30Z",
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_receipts VALUES(?,?)",
        (ingest_id, canonical_json_bytes(terminal_receipt).decode()),
    )
    connection.execute(
        """
        INSERT INTO unpublished_graphiti_admission_queue(
            queue_seq, proposal_key, ingest_id, source_revision_id,
            source_receipt_digest, proposal_digest, proposal_kind, request_json,
            request_digest, state, created_at, updated_at
        ) VALUES(2,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            proposal_key,
            ingest_id,
            request.source_lineage["revision_id"],
            terminal_receipt_digest,
            proposal.digest,
            proposal.kind.value,
            request_json,
            digest_bytes(request_json.encode()),
            "TERMINAL",
            "2026-08-24T11:59:30Z",
            "2026-08-24T11:59:30Z",
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_admission_decisions VALUES(?,?,?,?,?,?,?,?,?)",
        (
            proposal_key,
            decision.action.value,
            decision.decision_id,
            decision.authority_ledger_seq,
            decision.reason_code,
            decision.authority_receipt_digest,
            decision_json,
            digest_bytes(decision_json.encode()),
            "2026-08-24T11:59:30Z",
        ),
    )
    cohort_digest, derived_generation_id = graphiti_admission_generation_identity(
        ingest_ids=(ingest_id,),
        source_receipts=(
            {
                "ingest_id": ingest_id,
                "receipt_digest": terminal_receipt_digest,
                "proposal_count": 1,
            },
        ),
        members=(
            {
                "ingest_id": ingest_id,
                "proposal_key": proposal_key,
                "proposal_envelope_id": str(
                    request.proposal_authority_binding.proposal_envelope.proposal_id
                ),
                "decision_digest": digest_bytes(decision_json.encode()),
                "decision": decision.canonical_value(),
            },
        ),
    )
    generation_id = active_generation_id or derived_generation_id
    generation_result_digest = digest_canonical(
        {"generation_id": generation_id, "cohort_digest": cohort_digest}
    )
    reconciliation = {
        "generation_id": generation_id,
        "expected_effect_ids": [],
        "actual_effect_ids": [],
        "authority_watermark": 102,
        "receipt_digest": (
            DIGEST_C if active_generation_id is None else generation_result_digest
        ),
        "projector_family_id": "graph.increment4.admitted",
        "provider_model_calls": 0,
    }
    envelope: dict[str, object] = {
        "schema_version": (
            GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION
            if active_generation_id is None
            else GRAPHITI_ADMISSION_EXTENSION_RECONCILIATION_SCHEMA_VERSION
        ),
        "cohort_digest": cohort_digest,
        "ingest_ids": [ingest_id],
        "raw_receipt": reconciliation,
    }
    if active_generation_id is not None:
        envelope.update(
            {
                "source_snapshot_digest": DIGEST_A,
                "source_watermark_ledger_seq": 102,
                "source_request_digest": digest_canonical(
                    {"request": cohort_digest}
                ),
                "validation_digest": DIGEST_B,
                "promotion_digest": DIGEST_C,
                "generation_result_digest": generation_result_digest,
                "projection_state_digest": digest_canonical(
                    {"projection": cohort_digest}
                ),
            }
        )
    connection.execute(
        "INSERT INTO unpublished_graphiti_projection_reconciliations "
        "VALUES(?,?,?,?,?,?)",
        (
            reconciliation["receipt_digest"],
            "graph.increment4.admitted",
            generation_id,
            102,
            canonical_json_bytes(envelope).decode(),
            "2026-08-24T11:59:30Z",
        ),
    )
    connection.commit()
    return generation_id


def _seed_four_member_exact_generation(connection) -> None:
    drafts = tuple(
        _admission_draft(
            f"entity.{number:04d}",
            ExtractionProposalKind.ENTITY_MENTION,
            subject=f"Entity {number}",
        )
        for number in range(1, 5)
    )
    receipt = _seed_admission_receipt(connection, *drafts)
    consumer = _admission_consumer(
        connection,
        _AdmissionAuthority(
            {
                draft.local_id: GraphitiProposalAdmissionAction.ADMIT
                for draft in drafts
            }
        ),
        _AdmissionProjector(),
        _AdmissionRights(),
    )
    ingest_id = str(receipt["ingest_id"])
    assert consumer.enqueue_complete_receipts(ingest_ids=(ingest_id,)) == 4
    for proposal_key, request_json in connection.execute(
        "SELECT proposal_key,request_json "
        "FROM unpublished_graphiti_admission_queue"
    ).fetchall():
        request = json.loads(str(request_json))
        request["evidence_passages"][0].update(
            {
                "hydration_policy_contract_digest": DIGEST_A,
                "principal_id": "newsroom.hermes",
                "authority_domain": "newsroom.control-plane",
                "purpose": "fixture",
                "object_class": "article",
                "allowed_use": "extraction",
                "security_scope": "fixture",
                "retention_scope": "fixture",
                "language": "en",
                "text": None,
            }
        )
        encoded = canonical_json_bytes(request).decode()
        connection.execute(
            "UPDATE unpublished_graphiti_admission_queue "
            "SET request_json=?,request_digest=? WHERE proposal_key=?",
            (encoded, digest_bytes(encoded.encode()), proposal_key),
        )
    connection.commit()
    assert consumer.drain(
        worker_id="fixture-worker", limit=4, ingest_ids=(ingest_id,)
    ).decided == 4
    assert consumer.finalise_decided_cohort(
        ingest_ids=(ingest_id,)
    ).projected == 4


def test_full_generation_watermark_covers_all_member_decisions(tmp_path) -> None:
    connection = connect(str(tmp_path / "full-generation.sqlite3"))
    _seed_four_member_exact_generation(connection)

    context = GovernedContextHydrator(
        connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
    ).hydrate()

    assert context.status is GovernedContextStatus.READY
    assert len(context.items) == 4
    assert {item.admission_authority_version for item in context.items} == {
        101,
        102,
        103,
        104,
    }
    assert {item.projection_authority_watermark for item in context.items} == {104}
    assert len({item.projection_generation_id for item in context.items}) == 1
    future_decision = replace(context.items[0], admission_authority_version=105)
    with pytest.raises(ValueError, match="currency differs"):
        replace(context, items=(future_decision, *context.items[1:]))


@pytest.mark.parametrize(
    "tamper",
    ("underflow", "future", "wrong_generation", "reconciliation"),
)
def test_full_generation_watermark_tampering_fails_closed(
    tmp_path, tamper: str
) -> None:
    connection = connect(str(tmp_path / f"full-generation-{tamper}.sqlite3"))
    _seed_four_member_exact_generation(connection)
    if tamper == "reconciliation":
        connection.execute(
            "DROP TRIGGER immutable_graphiti_projection_reconciliations_update"
        )
        connection.execute(
            "UPDATE unpublished_graphiti_projection_reconciliations "
            "SET authority_watermark=105"
        )
    else:
        connection.execute(
            "DROP TRIGGER immutable_graphiti_projection_receipts_update"
        )
        field, value = (
            ("authority_watermark", 100)
            if tamper == "underflow"
            else ("authority_watermark", 105)
            if tamper == "future"
            else ("generation_id", "00000000-0000-4000-8000-0000000075ff")
        )
        connection.execute(
            f"UPDATE unpublished_graphiti_projection_receipts SET {field}=?",
            (value,),
        )
    connection.commit()

    context = GovernedContextHydrator(
        connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
    ).hydrate()

    assert context.status is GovernedContextStatus.HOLD
    assert context.reason_code in {
        "AMBIGUOUS_PROJECTION_WATERMARK",
        "ADMITTED_CONTEXT_RECEIPT_DRIFT",
        "ADMITTED_CONTEXT_RECEIPT_INVALID",
    }
    assert context.items == ()


def test_admitted_context_survives_hypothesis_candidate_evidence_and_cont(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "unpublished.sqlite3"))
    _seed_entity(connection)
    context = GovernedContextHydrator(
        connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
        max_items=4,
        max_context_bytes=16_384,
        max_token_contribution=4_096,
    ).hydrate()

    assert context.status is GovernedContextStatus.READY
    assert context.trust_label == ADMITTED_CONTEXT_TRUST_LABEL
    assert context.contiguous_projection_watermark == 101
    assert context.items[0].source_revision_id == (
        "00000000-0000-4000-8000-000000007580"
    )
    assert context.items[0].evidence_passages[0].byte_offset == 0
    assert context.items[0].evidence_passages[0].byte_length == 5
    assert context.items[0].evidence_passages[0].text_digest == DIGEST_A
    assert context.items[0].admitted_structured_value_json.endswith(
        '"display_name":"Alice Example"}}'
    )
    assert context.items[0].admission_current is True
    assert context.items[0].rights_current is True
    assert context.items[0].projection_gap_count == 0
    assert context.items[0].currency_read_at == "2026-08-24T12:00:00Z"
    assert context.context_bytes > 0
    assert context.context_bytes == len(canonical_json_bytes(context.canonical_value()))
    assert context.token_contribution == context.context_bytes

    source_item = SourceItem(
        "UK-01",
        "item-759",
        "Example headline",
        "Example retained body",
        "https://example.test/item-759",
    )
    candidate = form_candidates(
        (
            GroupedObservation(
                source_id="UK-01",
                observation_digest=DIGEST_A,
                item=source_item,
                observed_at="2026-08-24T11:59:00Z",
            ),
        ),
        governed_context=context,
    )[0]
    package = package_for(candidate)
    prompts: list[str] = []

    def write(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({"title": "【未出版】測試", "body": "測試內容"})

    copy = CliChainWriter(primary=write).dispatch(
        candidate,
        package,
        route="PRIMARY",
    )

    assert candidate.governed_context is not None
    assert package.admitted_context is candidate.governed_context
    assert copy.evidence_package_digest == package.digest
    assert ADMITTED_CONTEXT_TRUST_LABEL in prompts[0]
    assert "Alice Example" in prompts[0]
    assert "graphiti_workspace" not in prompts[0]


    assert candidate.governed_context is not None
    object.__setattr__(
        candidate.governed_context,
        "projection_generation_id",
        "wrong-generation",
    )
    with pytest.raises(WriterDispatchError, match="currency differs") as drift:
        CliChainWriter(primary=write).dispatch(
            candidate,
            package,
            route="PRIMARY",
        )

    assert drift.value.reason_code == "GOVERNED_CONTEXT_CURRENCY_DRIFT"
    assert len(prompts) == 1

    object.__setattr__(
        candidate.governed_context,
        "projection_generation_id",
        candidate.governed_context.items[0].projection_generation_id,
    )
    object.__setattr__(
        candidate.governed_context.items[0],
        "currency_read_at",
        "2026-08-24T12:00:01Z",
    )
    with pytest.raises(WriterDispatchError, match="currency differs"):
        CliChainWriter(primary=write).dispatch(
            candidate,
            package,
            route="PRIMARY",
        )

    assert len(prompts) == 1

    object.__setattr__(
        candidate.governed_context.items[0],
        "currency_read_at",
        candidate.governed_context.read_at,
    )
    object.__setattr__(
        candidate.governed_context.items[0],
        "projection_authority_watermark",
        999,
    )
    with pytest.raises(WriterDispatchError, match="currency differs"):
        CliChainWriter(primary=write).dispatch(
            candidate,
            package,
            route="PRIMARY",
        )

    assert len(prompts) == 1


def test_latest_all_hold_generation_preserves_prior_exact_admitted_context(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "multi-cohort.sqlite3"))
    admitted_generation_id = _seed_entity(connection, exact_binding=True)
    latest_generation_id = _seed_exact_all_hold_cohort(connection)

    context = GovernedContextHydrator(
        connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
    ).hydrate()

    assert context.status is GovernedContextStatus.READY
    assert context.projection_generation_id == latest_generation_id
    assert context.contiguous_projection_watermark == 102
    assert len(context.items) == 1
    assert context.items[0].projection_generation_id == admitted_generation_id
    assert context.items[0].projection_authority_watermark == 101


def test_active_generation_extension_keeps_cohorts_separate_in_context(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "active-extension-context.sqlite3"))
    generation_id = _seed_entity(connection, exact_binding=True)
    assert _seed_exact_all_hold_cohort(
        connection, active_generation_id=generation_id
    ) == generation_id

    context = GovernedContextHydrator(
        connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
    ).hydrate()

    assert context.status is GovernedContextStatus.READY
    assert context.projection_generation_id == generation_id
    assert context.contiguous_projection_watermark == 102
    assert len(context.items) == 1
    assert context.items[0].projection_generation_id == generation_id


def test_source_bound_extension_receipt_hydrates_without_cohort_generation_alias(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "source-bound-extension.sqlite3"))
    assert _seed_entity(
        connection,
        exact_binding=True,
        active_extension=True,
    ) == GENERATION_ID

    context = GovernedContextHydrator(
        connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
    ).hydrate()

    assert context.status is GovernedContextStatus.READY
    assert context.projection_generation_id == GENERATION_ID
    assert len(context.items) == 1
    assert context.items[0].projection_generation_id == GENERATION_ID


def test_exact_all_hold_generation_is_the_empty_context_generation(tmp_path) -> None:
    connection = connect(str(tmp_path / "all-hold-cohort.sqlite3"))
    generation_id = _seed_exact_all_hold_cohort(connection)

    context = GovernedContextHydrator(
        connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
    ).hydrate()

    assert context.status is GovernedContextStatus.EMPTY
    assert context.reason_code == "ZERO_ADMITTED_CONTEXT"
    assert context.projection_generation_id == generation_id
    assert context.contiguous_projection_watermark == 102
    assert context.items == ()


def test_multi_cohort_context_holds_without_every_exact_binding(tmp_path) -> None:
    connection = connect(str(tmp_path / "multi-cohort-legacy.sqlite3"))
    _seed_entity(connection)
    _seed_exact_all_hold_cohort(connection)

    context = GovernedContextHydrator(
        connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
    ).hydrate()

    assert context.status is GovernedContextStatus.HOLD
    assert context.reason_code == "ADMITTED_CONTEXT_RECEIPT_INVALID"
    assert context.items == ()


def test_held_and_rejected_proposals_produce_zero_context(tmp_path) -> None:
    for action in (
        GraphitiProposalAdmissionAction.HOLD,
        GraphitiProposalAdmissionAction.REJECT,
    ):
        connection = connect(str(tmp_path / f"{action.value}.sqlite3"))
        _seed_entity(connection, action=action)

        context = GovernedContextHydrator(
            connection,
            authority=_CurrentAuthority(),
            rights=_Rights(),
            clock=lambda: NOW,
        ).hydrate()

        assert context.status is GovernedContextStatus.EMPTY
        assert context.reason_code == "ZERO_ADMITTED_CONTEXT"
        assert context.items == ()


def test_rights_loss_and_stale_projection_fail_closed_without_items(tmp_path) -> None:
    rights_connection = connect(str(tmp_path / "rights.sqlite3"))
    _seed_entity(rights_connection)
    rights = _Rights()
    rights.current = False

    rights_context = GovernedContextHydrator(
        rights_connection,
        authority=_CurrentAuthority(),
        rights=rights,
        clock=lambda: NOW,
    ).hydrate()

    assert rights_context.status is GovernedContextStatus.HOLD
    assert rights_context.reason_code == "ADMITTED_CONTEXT_RIGHTS_LOST"
    assert rights_context.items == ()

    stale_connection = connect(str(tmp_path / "stale.sqlite3"))
    _seed_entity(stale_connection)
    stale_connection.execute(
        "UPDATE unpublished_graphiti_admission_queue "
        "SET state='READY', created_at='2026-08-20T00:00:00Z'"
    )
    stale_connection.commit()

    stale_context = GovernedContextHydrator(
        stale_connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
        max_oldest_lag_seconds=60,
    ).hydrate()

    assert stale_context.status is GovernedContextStatus.HOLD
    assert stale_context.reason_code == "ADMITTED_CONTEXT_STALE"
    assert stale_context.stale is True
    assert stale_context.items == ()

    candidate = form_candidates(
        (
            GroupedObservation(
                source_id="UK-01",
                observation_digest=DIGEST_A,
                item=SourceItem(
                    "UK-01",
                    "item-759",
                    "Example headline",
                    "Example retained body",
                    "https://example.test/item-759",
                ),
                observed_at="2026-08-24T11:59:00Z",
            ),
        ),
        governed_context=stale_context,
    )[0]
    dispatched: list[str] = []
    with pytest.raises(WriterDispatchError, match="context is held") as held:
        CliChainWriter(primary=dispatched.append).dispatch(
            candidate,
            package_for(candidate),
            route="PRIMARY",
        )

    assert held.value.reason_code == "GOVERNED_CONTEXT_HELD"
    assert dispatched == []

    class CustomWriter:
        def dispatch(
            self,
            candidate: object,
            package: object,
            *,
            route: str,
        ) -> None:
            del candidate, package
            dispatched.append(route)
            raise AssertionError("held context reached custom writer")

    with pytest.raises(WriterDispatchError, match="context is held"):
        _dispatch_writer(
            CustomWriter(),  # type: ignore[arg-type]
            candidate,
            package_for(candidate),
            route="PRIMARY",
        )

    assert dispatched == []

    assert candidate.governed_context is not None
    object.__setattr__(
        candidate.governed_context,
        "status",
        GovernedContextStatus.EMPTY,
    )
    object.__setattr__(candidate.governed_context, "projection_gap_count", 7)
    object.__setattr__(candidate.governed_context, "stale", True)
    object.__setattr__(candidate.governed_context, "degraded", True)
    with pytest.raises(WriterDispatchError, match="not current and gap-free"):
        _dispatch_writer(
            CustomWriter(),  # type: ignore[arg-type]
            candidate,
            package_for(candidate),
            route="PRIMARY",
        )

    assert dispatched == []


def test_context_size_is_bounded_and_replay_stable(tmp_path) -> None:
    connection = connect(str(tmp_path / "bounded.sqlite3"))
    _seed_entity(connection)
    hydrator = GovernedContextHydrator(
        connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
        max_context_bytes=16_384,
    )

    first = hydrator.hydrate()
    second = hydrator.hydrate()
    bounded = GovernedContextHydrator(
        connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
        max_context_bytes=32,
    ).hydrate()

    assert first.digest == second.digest
    assert first.canonical_value() == second.canonical_value()
    with pytest.raises(ValueError, match="current and gap-free"):
        replace(first, stale=True)
    with pytest.raises(ValueError, match="currency differs"):
        replace(first, projection_generation_id="wrong-generation")
    item_with_other_read_time = replace(
        first.items[0],
        currency_read_at="2026-08-24T12:00:01Z",
    )
    with pytest.raises(ValueError, match="currency differs"):
        replace(first, items=(item_with_other_read_time,))
    item_with_other_projection_watermark = replace(
        first.items[0],
        projection_authority_watermark=999,
    )
    with pytest.raises(ValueError, match="currency differs"):
        replace(first, items=(item_with_other_projection_watermark,))
    assert bounded.status is GovernedContextStatus.HOLD
    assert bounded.reason_code == "ADMITTED_CONTEXT_SIZE_BOUND_EXCEEDED"
    assert bounded.items == ()


def test_retained_receipt_drift_fails_closed(tmp_path) -> None:
    connection = connect(str(tmp_path / "tampered.sqlite3"))
    _seed_entity(connection)
    row = connection.execute(
        "SELECT request_json FROM unpublished_graphiti_admission_queue"
    ).fetchone()
    assert row is not None
    value = json.loads(str(row[0]))
    value["source_lineage"]["revision_id"] = "00000000-0000-4000-8000-0000000075ff"
    connection.execute(
        "UPDATE unpublished_graphiti_admission_queue SET request_json=?",
        (canonical_json_bytes(value).decode(),),
    )
    connection.commit()

    context = GovernedContextHydrator(
        connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
    ).hydrate()

    assert context.status is GovernedContextStatus.HOLD
    assert context.reason_code == "ADMITTED_CONTEXT_RECEIPT_DRIFT"
    assert context.items == ()


def test_empty_context_still_enforces_envelope_bounds(tmp_path) -> None:
    connection = connect(str(tmp_path / "empty-bound.sqlite3"))
    _seed_entity(connection, action=GraphitiProposalAdmissionAction.REJECT)

    context = GovernedContextHydrator(
        connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
        max_context_bytes=32,
        max_token_contribution=1,
    ).hydrate()

    assert context.status is GovernedContextStatus.HOLD
    assert context.reason_code == "ADMITTED_CONTEXT_SIZE_BOUND_EXCEEDED"
    assert context.items == ()


def test_fully_tombstoned_reconciled_generation_returns_empty_context(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "revoked-empty.sqlite3"))
    _seed_entity(connection, reconcile=False)
    tombstone = GraphitiProjectionReceipt(
        proposal_key=DIGEST_B,
        decision_id="decision:entity.0001",
        effect_id="tombstone:entity.0001",
        authority_watermark=101,
        receipt_digest=DIGEST_C,
        generation_id=GENERATION_ID,
    )
    tombstone_json = canonical_json_bytes(tombstone.canonical_value()).decode()
    connection.execute(
        "INSERT INTO unpublished_graphiti_projection_tombstones VALUES(?,?,?,?,?,?)",
        (
            tombstone.proposal_key,
            tombstone.effect_id,
            tombstone.authority_watermark,
            tombstone_json,
            tombstone.receipt_digest,
            "2026-08-24T11:59:30Z",
        ),
    )
    connection.execute(
        "UPDATE unpublished_graphiti_admission_queue SET state='REVOKED'"
    )
    reconciliation = {
        "generation_id": GENERATION_ID,
        "expected_effect_ids": [],
        "actual_effect_ids": [],
        "authority_watermark": 101,
        "receipt_digest": DIGEST_C,
        "projector_family_id": "graph.increment4.admitted",
        "provider_model_calls": 0,
    }
    connection.execute(
        "INSERT INTO unpublished_graphiti_projection_reconciliations "
        "VALUES(?,?,?,?,?,?)",
        (
            DIGEST_C,
            "graph.increment4.admitted",
            GENERATION_ID,
            101,
            canonical_json_bytes(reconciliation).decode(),
            "2026-08-24T11:59:30Z",
        ),
    )
    connection.commit()

    context = GovernedContextHydrator(
        connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
    ).hydrate()

    assert context.status is GovernedContextStatus.EMPTY
    assert context.reason_code == "ZERO_ADMITTED_CONTEXT"
    assert context.contiguous_projection_watermark == 101
    assert context.items == ()


@pytest.mark.parametrize(
    ("state", "reconcile", "expected_status", "expected_reason"),
    (
        ("READY", False, GovernedContextStatus.EMPTY, "ZERO_ADMITTED_CONTEXT"),
        (
            "DEAD_LETTER",
            False,
            GovernedContextStatus.HOLD,
            "ADMISSION_INTEGRITY_GAP",
        ),
        (
            "PROJECTED",
            False,
            GovernedContextStatus.HOLD,
            "AMBIGUOUS_PROJECTION_WATERMARK",
        ),
    ),
)
def test_raw_dead_lettered_and_ambiguous_context_cannot_enter_context(
    tmp_path,
    state: str,
    reconcile: bool,
    expected_status: GovernedContextStatus,
    expected_reason: str,
) -> None:
    connection = connect(str(tmp_path / f"{state}-{reconcile}.sqlite3"))
    if state == "PROJECTED":
        _seed_entity(connection, reconcile=reconcile)
    else:
        _seed_entity(connection, action=GraphitiProposalAdmissionAction.HOLD)
        connection.execute(
            "UPDATE unpublished_graphiti_admission_queue SET state=?, "
            "created_at='2026-08-24T11:59:00Z'",
            (state,),
        )
    connection.commit()

    context = GovernedContextHydrator(
        connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
    ).hydrate()

    assert context.status is expected_status
    assert context.reason_code == expected_reason
    assert context.items == ()


def test_candidate_context_builder_receives_one_preformed_hypothesis_scope(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "scoped.sqlite3"))
    _seed_entity(connection)
    hydrator = GovernedContextHydrator(
        connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
    )
    scopes: list[frozenset[tuple[str, str]]] = []

    def build(rows: tuple[GroupedObservation, ...]) -> GovernedContext:
        scope = frozenset((row.source_id, row.item.item_key) for row in rows)
        scopes.append(scope)
        return hydrator.hydrate(scope)

    observations = (
        GroupedObservation(
            source_id="UK-01",
            observation_digest=DIGEST_A,
            item=SourceItem(
                "UK-01",
                "item-759",
                "First event",
                "First retained body",
                "https://example.test/item-759",
            ),
            observed_at="2026-08-24T11:59:00Z",
        ),
        GroupedObservation(
            source_id="UK-02",
            observation_digest=DIGEST_B,
            item=SourceItem(
                "UK-02",
                "unrelated",
                "Different event",
                "Different retained body",
                "https://example.test/unrelated",
            ),
            observed_at="2026-08-24T11:59:00Z",
        ),
    )

    candidates = form_candidates(observations, governed_context_builder=build)

    assert scopes == [
        frozenset({("UK-01", "item-759")}),
        frozenset({("UK-02", "unrelated")}),
    ]
    statuses = {
        candidate.headline: candidate.governed_context.status
        for candidate in candidates
        if candidate.governed_context is not None
    }
    assert statuses == {
        "Different event": GovernedContextStatus.EMPTY,
        "First event": GovernedContextStatus.READY,
    }


def test_later_rejected_decision_advances_contiguous_not_projection_watermark(
    tmp_path,
) -> None:
    connection = connect(str(tmp_path / "rejected-watermark.sqlite3"))
    _seed_entity(connection)
    proposal = ProposalDraft(
        local_id="entity.0002",
        kind=ExtractionProposalKind.ENTITY_MENTION,
        subject_placeholder="Rejected",
        object_placeholder=None,
        predicate_hint=None,
        confidence_basis_points=None,
        uncertainty_codes=(),
        rationale_codes=("GRAPHITI_EVALUATION_SPAN",),
        evidence=(
            EvidenceRange(
                passage_id=ExtractionPassageId.parse(
                    "00000000-0000-4000-8000-0000000075c1"
                ),
                start_byte=0,
                end_byte=5,
                evidence_text_digest=DIGEST_C,
            ),
        ),
    )
    request = GraphitiAdmissionRequest(
        queue_seq=2,
        proposal_key=DIGEST_C,
        source_receipt_digest=DIGEST_C,
        proposal_authority_binding=_binding(proposal),
        proposal=proposal,
        proposal_payload=proposal.canonical_value(),
        evidence_passages=(
            {
                "passage_id": "00000000-0000-4000-8000-0000000075c1",
                "admission_id": "00000000-0000-4000-8000-0000000075c2",
                "access_decision_id": "00000000-0000-4000-8000-0000000075c3",
                "byte_offset": 0,
                "byte_length": 5,
                "blob_digest": DIGEST_C,
                "text_digest": DIGEST_C,
            },
        ),
        proposed_endpoints=None,
        relation_statement=None,
        relation_temporal_bounds=None,
        source_lineage={
            "ingest_id": DIGEST_C,
            "source_id": "UK-02",
            "item_key": "item-rejected",
            "revision_id": "00000000-0000-4000-8000-0000000075c4",
            "authority_record_ids": [
                "00000000-0000-4000-8000-0000000075c2",
                "00000000-0000-4000-8000-0000000075c3",
                "00000000-0000-4000-8000-0000000075c4",
            ],
            "generation_id": "newsroom-eval-generation-759",
            "episode_uuid": DIGEST_C,
            "reference_time": "2026-08-20T00:00:00Z",
            "temporal_basis": "SOURCE_PUBLISHED",
        },
    )
    decision = GraphitiGovernedDecision(
        proposal_key=request.proposal_key,
        proposal_digest=proposal.digest,
        proposal_kind=proposal.kind,
        proposal_local_id=proposal.local_id,
        action=GraphitiProposalAdmissionAction.REJECT,
        decision_id="decision:entity.0002",
        authority_ledger_seq=102,
        reason_code="FIXTURE_REJECT",
        authority_receipt_digest=DIGEST_C,
    )
    request_json = canonical_json_bytes(request.canonical_value()).decode()
    decision_json = canonical_json_bytes(decision.canonical_value()).decode()
    connection.execute(
        "INSERT INTO unpublished_graphiti_ingest VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            DIGEST_C,
            "UK-02",
            "item-rejected",
            "COMPLETE",
            1,
            1,
            0,
            "NONE",
            "SOURCE_PUBLISHED",
            "2026-08-20T00:00:00Z",
            "newsroom-eval-generation-759",
            DIGEST_C,
            "2026-08-24T11:59:00Z",
        ),
    )
    connection.execute(
        """
        INSERT INTO unpublished_graphiti_admission_queue(
            queue_seq, proposal_key, ingest_id, source_revision_id,
            source_receipt_digest, proposal_digest, proposal_kind, request_json,
            request_digest, state, created_at, updated_at
        ) VALUES(2,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            request.proposal_key,
            DIGEST_C,
            request.source_lineage["revision_id"],
            DIGEST_C,
            proposal.digest,
            proposal.kind.value,
            request_json,
            digest_bytes(request_json.encode()),
            "TERMINAL",
            "2026-08-24T11:59:00Z",
            "2026-08-24T11:59:00Z",
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_admission_decisions VALUES(?,?,?,?,?,?,?,?,?)",
        (
            request.proposal_key,
            decision.action.value,
            decision.decision_id,
            decision.authority_ledger_seq,
            decision.reason_code,
            decision.authority_receipt_digest,
            decision_json,
            digest_bytes(decision_json.encode()),
            "2026-08-24T11:59:00Z",
        ),
    )
    connection.commit()

    context = GovernedContextHydrator(
        connection,
        authority=_CurrentAuthority(),
        rights=_Rights(),
        clock=lambda: NOW,
    ).hydrate()

    assert context.status is GovernedContextStatus.READY
    assert context.contiguous_projection_watermark == 102
    assert context.items[0].projection_authority_watermark == 101
