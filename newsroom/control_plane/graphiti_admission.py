"""Durably drain retained Graphiti proposals through governed admission.

The consumer owns queueing, leases, retry state and receipts only.  Governed
entity/relation authority and the admitted projector remain behind injected
ports, so Graphiti itself receives neither ledger nor projector capability.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
    validate_sha256_digest,
)
from newsroom.authority.types import EventId, UtcTimestamp
from newsroom.entities.types import (
    EntityResolutionDecisionId,
    EntityResolutionDependencyId,
)
from newsroom.extraction.models import ProposalDraft, ProposalEnvelope
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
from newsroom.graphiti_adapter.admission import GraphitiProposalAdmissionAction
from newsroom.graphiti_adapter.evaluation_packet import GRAPHITI_WORKSPACE_GROUP
from newsroom.graphiti_adapter.identity import typed_id
from newsroom.graphiti_adapter.types import GraphitiAttemptId
from newsroom.projection.models import ProjectionGenerationId

_TERMINAL_INGEST_OUTCOMES = frozenset({"COMPLETE"})
_PROJECTABLE_KINDS = frozenset(
    {
        ExtractionProposalKind.ENTITY_MENTION,
        ExtractionProposalKind.ENTITY_EQUIVALENCE,
        ExtractionProposalKind.RELATION,
    }
)
_LINEAGE_FIELDS = (
    "ingest_id",
    "source_id",
    "item_key",
    "revision_id",
    "authority_record_ids",
    "generation_id",
    "episode_uuid",
    "reference_time",
    "temporal_basis",
)
_MAX_ADMISSION_REQUEST_BYTES = 256 * 1024
GRAPHITI_ADMISSION_COHORT_SCHEMA_VERSION = (
    "newsroom.graphiti-admission.exact-decided-cohort.v2"
)
GRAPHITI_ADMISSION_GENERATION_IDENTITY_VERSION = (
    "graphiti-admission-generation-v2"
)
GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION = (
    "newsroom.graphiti-admission.exact-generation-reconciliation.v1"
)
GRAPHITI_ADMISSION_EXTENSION_RECONCILIATION_SCHEMA_VERSION = (
    "newsroom.graphiti-admission.exact-active-extension-reconciliation.v2"
)
GRAPHITI_PROJECTION_GENERATION_SCHEMA_VERSION = (
    "newsroom.increment4.admitted-generation-binding.v2"
)
GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION = (
    "newsroom.increment4.admitted-active-extension-binding.v3"
)


class GraphitiAdmissionConsumerError(RuntimeError):
    """A durable receipt or governed admission response is invalid."""


class _HistoricalRawProposalHold(GraphitiAdmissionConsumerError):
    """A raw receipt has no exact 4D/4A authority and stays outside admission."""


def _exact_ingest_ids(
    ingest_ids: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    if ingest_ids is None:
        return None
    if (
        not isinstance(ingest_ids, tuple)
        or not ingest_ids
        or any(
            not isinstance(item, str)
            or not item
            or len(item.encode("utf-8")) > 256
            for item in ingest_ids
        )
        or ingest_ids != tuple(sorted(set(ingest_ids)))
    ):
        raise ValueError("exact Graphiti admission ingest identities are invalid")
    return ingest_ids


def graphiti_admission_generation_identity(
    *,
    ingest_ids: tuple[str, ...],
    source_receipts: tuple[Mapping[str, object], ...],
    members: tuple[Mapping[str, object], ...],
) -> tuple[str, str]:
    """Derive the one deterministic generation identity for an exact cohort."""

    exact = _exact_ingest_ids(ingest_ids)
    assert exact is not None
    receipt_values = tuple(dict(item) for item in source_receipts)
    member_values = tuple(dict(item) for item in members)
    if tuple(str(item.get("ingest_id") or "") for item in receipt_values) != exact:
        raise GraphitiAdmissionConsumerError(
            "generation source receipts differ from the exact cohort"
        )
    proposal_count = 0
    for item in receipt_values:
        if set(item) != {"ingest_id", "receipt_digest", "proposal_count"}:
            raise GraphitiAdmissionConsumerError(
                "generation source receipt fields differ"
            )
        proposal_count += _exact_integer(
            item["proposal_count"], field="generation proposal denominator"
        )
        validate_sha256_digest(
            str(item["receipt_digest"]),
            field="generation source receipt digest",
        )
    proposal_keys: list[str] = []
    for item in member_values:
        if set(item) != {
            "ingest_id",
            "proposal_key",
            "proposal_envelope_id",
            "decision_digest",
            "decision",
        }:
            raise GraphitiAdmissionConsumerError("generation member fields differ")
        if str(item["ingest_id"]) not in exact:
            raise GraphitiAdmissionConsumerError(
                "generation member is outside the exact cohort"
            )
        proposal_key = str(item["proposal_key"])
        if not proposal_key or not str(item["proposal_envelope_id"]):
            raise GraphitiAdmissionConsumerError(
                "generation member identity is incomplete"
            )
        validate_sha256_digest(
            str(item["decision_digest"]),
            field="generation decision digest",
        )
        if not isinstance(item["decision"], Mapping):
            raise GraphitiAdmissionConsumerError(
                "generation decision value is invalid"
            )
        proposal_keys.append(proposal_key)
    if (
        len(member_values) != proposal_count
        or len(proposal_keys) != len(set(proposal_keys))
    ):
        raise GraphitiAdmissionConsumerError(
            "generation members differ from the proposal denominator"
        )
    cohort_digest = digest_canonical(
        {
            "schema_version": GRAPHITI_ADMISSION_COHORT_SCHEMA_VERSION,
            "ingest_ids": list(exact),
            "source_receipts": list(receipt_values),
            "members": sorted(
                member_values,
                key=lambda item: str(item["proposal_key"]),
            ),
        }
    )
    generation_id = str(
        typed_id(
            ProjectionGenerationId,
            GRAPHITI_ADMISSION_GENERATION_IDENTITY_VERSION,
            cohort_digest,
        )
    )
    return cohort_digest, generation_id


def _proposal_envelope_value(envelope: ProposalEnvelope) -> dict[str, object]:
    draft = ProposalDraft(
        local_id=envelope.local_id,
        kind=envelope.kind,
        subject_placeholder=envelope.subject_placeholder,
        object_placeholder=envelope.object_placeholder,
        predicate_hint=envelope.predicate_hint,
        confidence_basis_points=envelope.confidence_basis_points,
        uncertainty_codes=envelope.uncertainty_codes,
        rationale_codes=envelope.rationale_codes,
        evidence=envelope.evidence,
    )
    return {
        "proposal_id": str(envelope.proposal_id),
        "proposal_set_id": str(envelope.proposal_set_id),
        "output_id": str(envelope.output_id),
        "run_id": str(envelope.run_id),
        "run_version_id": str(envelope.run_version_id),
        "draft": draft.canonical_value(),
        "producer_contract_digest": envelope.producer_contract_digest,
        "canonical_digest": envelope.canonical_digest,
        "retained_at": envelope.retained_at.to_text(),
    }


def _proposal_envelope_from_value(value: Mapping[str, object]) -> ProposalEnvelope:
    raw = dict(value)
    expected = {
        "proposal_id",
        "proposal_set_id",
        "output_id",
        "run_id",
        "run_version_id",
        "draft",
        "producer_contract_digest",
        "canonical_digest",
        "retained_at",
    }
    if set(raw) != expected:
        raise GraphitiAdmissionConsumerError(
            "proposal envelope fields differ from the 4A contract"
        )
    draft = _parse_proposal(_mapping(raw["draft"], field="proposal envelope draft"))
    try:
        return ProposalEnvelope(
            proposal_id=ProposalEnvelopeId.parse(str(raw["proposal_id"])),
            proposal_set_id=ProposalSetId.parse(str(raw["proposal_set_id"])),
            output_id=ExtractionOutputId.parse(str(raw["output_id"])),
            run_id=ExtractionRunId.parse(str(raw["run_id"])),
            run_version_id=ExtractionRunVersionId.parse(
                str(raw["run_version_id"])
            ),
            local_id=draft.local_id,
            kind=draft.kind,
            subject_placeholder=draft.subject_placeholder,
            object_placeholder=draft.object_placeholder,
            predicate_hint=draft.predicate_hint,
            confidence_basis_points=draft.confidence_basis_points,
            uncertainty_codes=draft.uncertainty_codes,
            rationale_codes=draft.rationale_codes,
            evidence=draft.evidence,
            producer_contract_digest=str(raw["producer_contract_digest"]),
            canonical_digest=str(raw["canonical_digest"]),
            retained_at=UtcTimestamp.parse(str(raw["retained_at"])),
        )
    except (TypeError, ValueError) as exc:
        raise GraphitiAdmissionConsumerError(
            "proposal envelope is not exact typed 4A authority"
        ) from exc


@dataclass(frozen=True, slots=True)
class GraphitiProposalAuthorityBinding:
    """Exact 4D attempt and 4A ProposalEnvelope authority for one raw proposal."""

    graphiti_attempt_id: str
    graphiti_attempt_authority_event_id: str
    proposal_envelope: ProposalEnvelope

    def __post_init__(self) -> None:
        try:
            GraphitiAttemptId.parse(self.graphiti_attempt_id)
            EventId.parse(self.graphiti_attempt_authority_event_id)
        except (TypeError, ValueError) as exc:
            raise GraphitiAdmissionConsumerError(
                "proposal authority binding has invalid 4D identities"
            ) from exc
        if not isinstance(self.proposal_envelope, ProposalEnvelope):
            raise GraphitiAdmissionConsumerError(
                "proposal authority binding needs a typed 4A envelope"
            )

    def canonical_value(self) -> dict[str, object]:
        return {
            "graphiti_attempt_id": self.graphiti_attempt_id,
            "graphiti_attempt_authority_event_id": (
                self.graphiti_attempt_authority_event_id
            ),
            "proposal_envelope": _proposal_envelope_value(
                self.proposal_envelope
            ),
        }


def _proposal_authority_binding_from_value(
    value: Mapping[str, object],
) -> GraphitiProposalAuthorityBinding:
    raw = dict(value)
    if set(raw) != {
        "graphiti_attempt_id",
        "graphiti_attempt_authority_event_id",
        "proposal_envelope",
    }:
        raise GraphitiAdmissionConsumerError(
            "proposal authority binding fields differ"
        )
    return GraphitiProposalAuthorityBinding(
        graphiti_attempt_id=str(raw["graphiti_attempt_id"]),
        graphiti_attempt_authority_event_id=str(
            raw["graphiti_attempt_authority_event_id"]
        ),
        proposal_envelope=_proposal_envelope_from_value(
            _mapping(raw["proposal_envelope"], field="proposal envelope")
        ),
    )


@dataclass(frozen=True, slots=True)
class GraphitiAdmissionRequest:
    queue_seq: int
    proposal_key: str
    source_receipt_digest: str
    proposal_authority_binding: GraphitiProposalAuthorityBinding
    proposal: ProposalDraft
    proposal_payload: Mapping[str, object]
    evidence_passages: tuple[Mapping[str, object], ...]
    proposed_endpoints: tuple[str, str] | None
    relation_statement: str | None
    relation_temporal_bounds: Mapping[str, object] | None
    source_lineage: Mapping[str, object]
    relation_endpoint_bindings: tuple[GraphitiProposalAuthorityBinding, ...] = ()

    def __post_init__(self) -> None:
        envelope = self.proposal_authority_binding.proposal_envelope
        envelope_draft = ProposalDraft(
            local_id=envelope.local_id,
            kind=envelope.kind,
            subject_placeholder=envelope.subject_placeholder,
            object_placeholder=envelope.object_placeholder,
            predicate_hint=envelope.predicate_hint,
            confidence_basis_points=envelope.confidence_basis_points,
            uncertainty_codes=envelope.uncertainty_codes,
            rationale_codes=envelope.rationale_codes,
            evidence=envelope.evidence,
        )
        if envelope_draft != self.proposal:
            raise GraphitiAdmissionConsumerError(
                "admission request differs from its exact ProposalEnvelope"
            )
        endpoint_bound_kind = self.proposal.kind in {
            ExtractionProposalKind.ENTITY_EQUIVALENCE,
            ExtractionProposalKind.RELATION,
        }
        if endpoint_bound_kind:
            expected = (
                self.proposal.subject_placeholder,
                self.proposal.object_placeholder,
            )
            if self.proposed_endpoints != expected:
                raise GraphitiAdmissionConsumerError(
                    "proposal admission mapping lacks its exact endpoints"
                )
            if self.proposal.kind is ExtractionProposalKind.RELATION:
                temporal_fields = set(self.relation_temporal_bounds or {})
                if (
                    not self.relation_statement
                    or self.relation_temporal_bounds is None
                    or not temporal_fields.issubset(
                        {"valid_at", "invalid_at", "expired_at"}
                    )
                ):
                    raise GraphitiAdmissionConsumerError(
                        "relation admission mapping lacks statement or exact temporal evidence"
                    )
            elif (
                self.relation_statement is not None
                or self.relation_temporal_bounds is not None
            ):
                raise GraphitiAdmissionConsumerError(
                    "entity equivalence cannot carry relation evidence"
                )
            endpoint_envelopes = tuple(
                item.proposal_envelope for item in self.relation_endpoint_bindings
            )
            parent_envelope = self.proposal_authority_binding.proposal_envelope
            if (
                len(endpoint_envelopes) != 2
                or len({item.proposal_id for item in endpoint_envelopes}) != 2
                or len({item.local_id for item in endpoint_envelopes}) != 2
                or any(
                    item.kind is not ExtractionProposalKind.ENTITY_MENTION
                    for item in endpoint_envelopes
                )
                or tuple(item.subject_placeholder for item in endpoint_envelopes)
                != self.proposed_endpoints
                or any(
                    binding.graphiti_attempt_id
                    != self.proposal_authority_binding.graphiti_attempt_id
                    or binding.graphiti_attempt_authority_event_id
                    != self.proposal_authority_binding.graphiti_attempt_authority_event_id
                    or binding.proposal_envelope.proposal_set_id
                    != parent_envelope.proposal_set_id
                    or binding.proposal_envelope.output_id != parent_envelope.output_id
                    or binding.proposal_envelope.run_id != parent_envelope.run_id
                    or binding.proposal_envelope.run_version_id
                    != parent_envelope.run_version_id
                    for binding in self.relation_endpoint_bindings
                )
            ):
                raise GraphitiAdmissionConsumerError(
                    "proposal admission lacks two exact same-attempt entity ProposalEnvelopes"
                )
            if self.proposal.kind is ExtractionProposalKind.ENTITY_EQUIVALENCE:
                parent_evidence = {
                    canonical_json_bytes(item.canonical_value())
                    for item in parent_envelope.evidence
                }
                endpoint_evidence = {
                    canonical_json_bytes(item.canonical_value())
                    for endpoint in endpoint_envelopes
                    for item in endpoint.evidence
                }
                if (
                    len(parent_envelope.evidence) != 2
                    or any(len(endpoint.evidence) != 1 for endpoint in endpoint_envelopes)
                    or parent_evidence != endpoint_evidence
                ):
                    raise GraphitiAdmissionConsumerError(
                        "entity equivalence evidence differs from its exact mentions"
                    )
        elif any(
            value is not None
            for value in (
                self.proposed_endpoints,
                self.relation_statement,
                self.relation_temporal_bounds,
            )
        ) or self.relation_endpoint_bindings:
            raise GraphitiAdmissionConsumerError(
                "entity admission mapping cannot carry relation evidence"
            )
        if not self.evidence_passages:
            raise GraphitiAdmissionConsumerError(
                "admission request needs referenced passage metadata"
            )

    def canonical_value(self) -> dict[str, object]:
        return {
            "queue_seq": self.queue_seq,
            "proposal_key": self.proposal_key,
            "source_receipt_digest": self.source_receipt_digest,
            "proposal_authority_binding": (
                self.proposal_authority_binding.canonical_value()
            ),
            "proposal": dict(self.proposal_payload),
            "evidence_passages": [dict(item) for item in self.evidence_passages],
            "proposed_endpoints": (
                None if self.proposed_endpoints is None else list(self.proposed_endpoints)
            ),
            "relation_endpoint_bindings": [
                item.canonical_value() for item in self.relation_endpoint_bindings
            ],
            "relation_statement": self.relation_statement,
            "relation_temporal_bounds": (
                None
                if self.relation_temporal_bounds is None
                else dict(self.relation_temporal_bounds)
            ),
            "source_lineage": dict(self.source_lineage),
        }

    @property
    def digest(self) -> str:
        return digest_canonical(self.canonical_value())


@dataclass(frozen=True, slots=True)
class GraphitiRelationHoldBasis:
    """Existing 4B dependency receipts supporting a non-authoritative HOLD."""

    dependency_id: str
    authority_event_id: str
    authority_ledger_seq: int
    authority_receipt_digest: str

    def __post_init__(self) -> None:
        try:
            EntityResolutionDependencyId.parse(self.dependency_id)
            EventId.parse(self.authority_event_id)
        except (TypeError, ValueError) as exc:
            raise GraphitiAdmissionConsumerError(
                "relation HOLD basis has invalid 4B identities"
            ) from exc
        if (
            isinstance(self.authority_ledger_seq, bool)
            or not isinstance(self.authority_ledger_seq, int)
            or self.authority_ledger_seq <= 0
        ):
            raise GraphitiAdmissionConsumerError(
                "relation HOLD basis needs a positive 4B ledger sequence"
            )
        validate_sha256_digest(
            self.authority_receipt_digest,
            field="relation HOLD 4B dependency receipt digest",
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "dependency_id": self.dependency_id,
            "authority_event_id": self.authority_event_id,
            "authority_ledger_seq": self.authority_ledger_seq,
            "authority_receipt_digest": self.authority_receipt_digest,
        }


@dataclass(frozen=True, slots=True)
class GraphitiGovernedDecision:
    proposal_key: str
    proposal_digest: str
    proposal_kind: ExtractionProposalKind
    proposal_local_id: str
    action: GraphitiProposalAdmissionAction
    decision_id: str
    authority_ledger_seq: int
    reason_code: str
    authority_receipt_digest: str
    admitted_authority_id: str | None = None
    endpoint_resolution_decision_ids: tuple[str, ...] = ()
    resolved_endpoint_names: tuple[str, ...] = ()
    relation_hold_basis: tuple[GraphitiRelationHoldBasis, ...] = ()
    provider_model_calls: int = 0

    def __post_init__(self) -> None:
        if not self.proposal_key or not self.decision_id or not self.reason_code:
            raise GraphitiAdmissionConsumerError(
                "governed admission decision identity is incomplete"
            )
        validate_sha256_digest(
            self.proposal_digest,
            field="Graphiti governed admission proposal digest",
        )
        if self.proposal_kind not in _PROJECTABLE_KINDS:
            raise GraphitiAdmissionConsumerError(
                "governed admission decision proposal kind is invalid"
            )
        if not self.proposal_local_id:
            raise GraphitiAdmissionConsumerError(
                "governed admission decision proposal local identity is incomplete"
            )
        if not isinstance(self.action, GraphitiProposalAdmissionAction):
            raise GraphitiAdmissionConsumerError(
                "governed admission decision action must be typed"
            )
        if (
            isinstance(self.authority_ledger_seq, bool)
            or not isinstance(self.authority_ledger_seq, int)
            or self.authority_ledger_seq <= 0
        ):
            raise GraphitiAdmissionConsumerError(
                "governed admission decision needs a positive ledger sequence"
            )
        validate_sha256_digest(
            self.authority_receipt_digest,
            field="Graphiti governed admission receipt digest",
        )
        if self.action is GraphitiProposalAdmissionAction.ADMIT:
            if not self.admitted_authority_id:
                raise GraphitiAdmissionConsumerError(
                    "admitted decision needs its canonical authority identity"
                )
        elif self.admitted_authority_id is not None:
            raise GraphitiAdmissionConsumerError(
                "non-admitted decision cannot name canonical authority"
            )
        if self.proposal_kind is ExtractionProposalKind.RELATION:
            if (
                self.action is GraphitiProposalAdmissionAction.ADMIT
                and (
                    len(self.endpoint_resolution_decision_ids) != 2
                    or len(set(self.endpoint_resolution_decision_ids)) != 2
                    or any(not item for item in self.endpoint_resolution_decision_ids)
                    or len(self.resolved_endpoint_names) != 2
                )
            ):
                raise GraphitiAdmissionConsumerError(
                    "relation admission requires two effective endpoint resolutions"
                )
            for decision_id in self.endpoint_resolution_decision_ids:
                try:
                    EntityResolutionDecisionId.parse(decision_id)
                except (TypeError, ValueError) as exc:
                    raise GraphitiAdmissionConsumerError(
                        "relation endpoint resolution identity must be UUIDv4"
                    ) from exc
        elif self.endpoint_resolution_decision_ids or self.resolved_endpoint_names:
            raise GraphitiAdmissionConsumerError(
                "entity resolution decision cannot name relation endpoints"
            )
        if self.relation_hold_basis:
            if (
                self.proposal_kind is not ExtractionProposalKind.RELATION
                or self.action is GraphitiProposalAdmissionAction.ADMIT
                or len(self.relation_hold_basis) != 2
                or len({item.dependency_id for item in self.relation_hold_basis}) != 2
                or tuple(
                    sorted(self.relation_hold_basis, key=lambda item: item.dependency_id)
                )
                != self.relation_hold_basis
            ):
                raise GraphitiAdmissionConsumerError(
                    "operational relation HOLD needs two sorted 4B dependency bases"
                )
        if self.provider_model_calls != 0:
            raise GraphitiAdmissionConsumerError(
                "admission authority must make zero provider/model calls"
            )

    def canonical_value(self) -> dict[str, object]:
        return {
            "proposal_key": self.proposal_key,
            "proposal_digest": self.proposal_digest,
            "proposal_kind": self.proposal_kind.value,
            "proposal_local_id": self.proposal_local_id,
            "action": self.action.value,
            "decision_id": self.decision_id,
            "authority_ledger_seq": self.authority_ledger_seq,
            "reason_code": self.reason_code,
            "authority_receipt_digest": self.authority_receipt_digest,
            "admitted_authority_id": self.admitted_authority_id,
            "endpoint_resolution_decision_ids": list(
                self.endpoint_resolution_decision_ids
            ),
            "resolved_endpoint_names": list(self.resolved_endpoint_names),
            "relation_hold_basis": [
                item.canonical_value() for item in self.relation_hold_basis
            ],
            "provider_model_calls": self.provider_model_calls,
        }


@dataclass(frozen=True, slots=True)
class GraphitiProjectionRequest:
    request: GraphitiAdmissionRequest
    decision: GraphitiGovernedDecision


@dataclass(frozen=True, slots=True)
class GraphitiProjectionReceipt:
    proposal_key: str
    decision_id: str
    effect_id: str
    authority_watermark: int
    receipt_digest: str
    projector_family_id: str = "graph.increment4.admitted"
    generation_id: str = ""
    schema_version: str = "newsroom.increment4.admitted-projection.v1"
    trust_scope: str = "ADMITTED"
    cohort_digest: str | None = None
    source_snapshot_digest: str | None = None
    validation_digest: str | None = None
    promotion_digest: str | None = None
    generation_result_digest: str | None = None
    source_request_digest: str | None = None
    provider_model_calls: int = 0

    def __post_init__(self) -> None:
        if not self.proposal_key or not self.decision_id or not self.effect_id:
            raise GraphitiAdmissionConsumerError(
                "governed projection receipt identity is incomplete"
            )
        if (
            isinstance(self.authority_watermark, bool)
            or not isinstance(self.authority_watermark, int)
            or self.authority_watermark <= 0
        ):
            raise GraphitiAdmissionConsumerError(
                "governed projection receipt needs a positive authority watermark"
            )
        validate_sha256_digest(
            self.receipt_digest,
            field="Graphiti governed projection receipt digest",
        )
        if self.projector_family_id != "graph.increment4.admitted":
            raise GraphitiAdmissionConsumerError(
                "projection receipt is outside the Increment 4 admitted family"
            )
        if not self.generation_id:
            raise GraphitiAdmissionConsumerError(
                "projection receipt needs a governed generation identity"
            )
        try:
            ProjectionGenerationId.parse(self.generation_id)
        except (TypeError, ValueError) as exc:
            raise GraphitiAdmissionConsumerError(
                "projection receipt generation identity must be UUIDv4"
            ) from exc
        legacy_schema = "newsroom.increment4.admitted-projection.v1"
        generation_fields = (
            self.cohort_digest,
            self.source_snapshot_digest,
            self.validation_digest,
            self.promotion_digest,
            self.generation_result_digest,
        )
        if self.schema_version == legacy_schema:
            if any(item is not None for item in generation_fields) or (
                self.source_request_digest is not None
            ):
                raise GraphitiAdmissionConsumerError(
                    "legacy projection receipt cannot claim generation binding"
                )
        elif self.schema_version == GRAPHITI_PROJECTION_GENERATION_SCHEMA_VERSION:
            if any(item is None for item in generation_fields):
                raise GraphitiAdmissionConsumerError(
                    "generation binding receipt is incomplete"
                )
            for field, value in zip(
                (
                    "cohort_digest",
                    "source_snapshot_digest",
                    "validation_digest",
                    "promotion_digest",
                    "generation_result_digest",
                ),
                generation_fields,
                strict=True,
            ):
                validate_sha256_digest(str(value), field=field)
            if self.source_request_digest is not None:
                raise GraphitiAdmissionConsumerError(
                    "generation binding receipt cannot claim active extension"
                )
        elif self.schema_version == GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION:
            if any(item is None for item in generation_fields) or (
                self.source_request_digest is None
            ):
                raise GraphitiAdmissionConsumerError(
                    "active extension receipt is incomplete"
                )
            for field, value in zip(
                (
                    "cohort_digest",
                    "source_snapshot_digest",
                    "validation_digest",
                    "promotion_digest",
                    "generation_result_digest",
                ),
                generation_fields,
                strict=True,
            ):
                validate_sha256_digest(str(value), field=field)
            validate_sha256_digest(
                self.source_request_digest, field="source_request_digest"
            )
        else:
            raise GraphitiAdmissionConsumerError(
                "projection receipt schema differs"
            )
        if self.trust_scope != "ADMITTED":
            raise GraphitiAdmissionConsumerError(
                "projection receipt schema or trust scope differs"
            )
        if self.provider_model_calls != 0:
            raise GraphitiAdmissionConsumerError(
                "admission and projection must make zero provider/model calls"
            )

    def canonical_value(self) -> dict[str, object]:
        value: dict[str, object] = {
            "proposal_key": self.proposal_key,
            "decision_id": self.decision_id,
            "effect_id": self.effect_id,
            "authority_watermark": self.authority_watermark,
            "receipt_digest": self.receipt_digest,
            "projector_family_id": self.projector_family_id,
            "generation_id": self.generation_id,
            "schema_version": self.schema_version,
            "trust_scope": self.trust_scope,
            "cohort_digest": self.cohort_digest,
            "source_snapshot_digest": self.source_snapshot_digest,
            "validation_digest": self.validation_digest,
            "promotion_digest": self.promotion_digest,
            "generation_result_digest": self.generation_result_digest,
            "provider_model_calls": self.provider_model_calls,
        }
        if self.schema_version == GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION:
            value["source_request_digest"] = self.source_request_digest
        return value


@dataclass(frozen=True, slots=True)
class GraphitiProjectionGenerationResult:
    """One complete reconciled Increment 4 generation for a decided cohort."""

    cohort_digest: str
    generation_id: str
    source_snapshot_digest: str
    authority_watermark: int
    validation_digest: str
    promotion_digest: str
    reconciliation_digest: str
    admitted_authority_ids: tuple[str, ...]
    schema_version: str = GRAPHITI_PROJECTION_GENERATION_SCHEMA_VERSION
    source_request_digest: str | None = None
    provider_model_calls: int = 0

    def __post_init__(self) -> None:
        for field, value in (
            ("cohort_digest", self.cohort_digest),
            ("source_snapshot_digest", self.source_snapshot_digest),
            ("validation_digest", self.validation_digest),
            ("promotion_digest", self.promotion_digest),
            ("reconciliation_digest", self.reconciliation_digest),
        ):
            validate_sha256_digest(value, field=field)
        try:
            ProjectionGenerationId.parse(self.generation_id)
        except (TypeError, ValueError) as exc:
            raise GraphitiAdmissionConsumerError(
                "generation result identity must be UUIDv4"
            ) from exc
        if (
            isinstance(self.authority_watermark, bool)
            or not isinstance(self.authority_watermark, int)
            or self.authority_watermark <= 0
        ):
            raise GraphitiAdmissionConsumerError(
                "generation result needs a positive authority watermark"
            )
        if self.admitted_authority_ids != tuple(
            sorted(set(self.admitted_authority_ids))
        ) or any(not item for item in self.admitted_authority_ids):
            raise GraphitiAdmissionConsumerError(
                "generation admitted authority identities must be sorted and unique"
            )
        if self.provider_model_calls != 0:
            raise GraphitiAdmissionConsumerError(
                "generation projection must make zero provider/model calls"
            )
        if self.schema_version == GRAPHITI_PROJECTION_GENERATION_SCHEMA_VERSION:
            if self.source_request_digest is not None:
                raise GraphitiAdmissionConsumerError(
                    "generation result cannot claim active extension"
                )
        elif self.schema_version == GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION:
            if self.source_request_digest is None:
                raise GraphitiAdmissionConsumerError(
                    "active extension result lacks source request authority"
                )
            validate_sha256_digest(
                self.source_request_digest, field="source_request_digest"
            )
        else:
            raise GraphitiAdmissionConsumerError(
                "generation result schema differs"
            )

    def canonical_value(self) -> dict[str, object]:
        value: dict[str, object] = {
            "cohort_digest": self.cohort_digest,
            "generation_id": self.generation_id,
            "source_snapshot_digest": self.source_snapshot_digest,
            "authority_watermark": self.authority_watermark,
            "validation_digest": self.validation_digest,
            "promotion_digest": self.promotion_digest,
            "reconciliation_digest": self.reconciliation_digest,
            "admitted_authority_ids": list(self.admitted_authority_ids),
            "provider_model_calls": self.provider_model_calls,
        }
        if self.schema_version == GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION:
            value["schema_version"] = self.schema_version
            value["source_request_digest"] = self.source_request_digest
        return value

    @property
    def digest(self) -> str:
        return digest_canonical(self.canonical_value())


@dataclass(frozen=True, slots=True)
class GraphitiProjectionReconciliationReceipt:
    generation_id: str
    expected_effect_ids: tuple[str, ...]
    actual_effect_ids: tuple[str, ...]
    authority_watermark: int
    receipt_digest: str
    projector_family_id: str = "graph.increment4.admitted"
    provider_model_calls: int = 0

    def __post_init__(self) -> None:
        if not self.generation_id:
            raise GraphitiAdmissionConsumerError(
                "projection reconciliation needs a generation identity"
            )
        try:
            ProjectionGenerationId.parse(self.generation_id)
        except (TypeError, ValueError) as exc:
            raise GraphitiAdmissionConsumerError(
                "projection reconciliation generation identity must be UUIDv4"
            ) from exc
        if (
            self.expected_effect_ids
            != tuple(sorted(set(self.expected_effect_ids)))
            or self.actual_effect_ids != tuple(sorted(set(self.actual_effect_ids)))
        ):
            raise GraphitiAdmissionConsumerError(
                "projection reconciliation effect identities must be sorted and unique"
            )
        if self.expected_effect_ids != self.actual_effect_ids:
            raise GraphitiAdmissionConsumerError(
                "admitted projection does not reconcile to governed SQLite authority"
            )
        if self.projector_family_id != "graph.increment4.admitted":
            raise GraphitiAdmissionConsumerError(
                "projection reconciliation family differs from Increment 4"
            )
        if (
            isinstance(self.authority_watermark, bool)
            or not isinstance(self.authority_watermark, int)
            or self.authority_watermark <= 0
        ):
            raise GraphitiAdmissionConsumerError(
                "projection reconciliation needs a positive authority watermark"
            )
        validate_sha256_digest(
            self.receipt_digest,
            field="Graphiti projection reconciliation digest",
        )
        if self.provider_model_calls != 0:
            raise GraphitiAdmissionConsumerError(
                "projection reconciliation must make zero provider/model calls"
            )

    def canonical_value(self) -> dict[str, object]:
        return {
            "generation_id": self.generation_id,
            "expected_effect_ids": list(self.expected_effect_ids),
            "actual_effect_ids": list(self.actual_effect_ids),
            "authority_watermark": self.authority_watermark,
            "receipt_digest": self.receipt_digest,
            "projector_family_id": self.projector_family_id,
            "provider_model_calls": self.provider_model_calls,
        }


class GovernedGraphitiAdmissionAuthority(Protocol):
    def decide_entity_resolution(
        self,
        request: GraphitiAdmissionRequest,
        *,
        required_action: GraphitiProposalAdmissionAction | None,
        idempotency_key: str,
    ) -> GraphitiGovernedDecision: ...

    def decide_relation_admission(
        self,
        request: GraphitiAdmissionRequest,
        *,
        required_action: GraphitiProposalAdmissionAction | None,
        idempotency_key: str,
    ) -> GraphitiGovernedDecision: ...

    def relation_endpoint_resolutions_current(
        self,
        request: GraphitiAdmissionRequest,
        decision: GraphitiGovernedDecision,
    ) -> bool:
        """Re-read both effective Entity Resolution Decisions from authority."""
        ...


class GovernedGraphitiProjector(Protocol):
    def build_and_promote_increment4_cohort(
        self,
        requests: tuple[GraphitiProjectionRequest, ...],
        *,
        cohort_digest: str,
        generation_id: str,
        idempotency_key: str,
    ) -> GraphitiProjectionGenerationResult: ...

    def recover_increment4_admitted_receipt(
        self, *, idempotency_key: str
    ) -> GraphitiProjectionReceipt | None: ...

    def deliver_increment4_admitted(
        self,
        request: GraphitiProjectionRequest,
        *,
        idempotency_key: str,
    ) -> GraphitiProjectionReceipt: ...

    def tombstone_increment4_admitted(
        self,
        request: GraphitiProjectionRequest,
        *,
        idempotency_key: str,
    ) -> GraphitiProjectionReceipt: ...

    def reconcile_increment4_admitted(
        self,
        expected: tuple[GraphitiProjectionReceipt, ...],
        *,
        generation_id: str,
    ) -> GraphitiProjectionReconciliationReceipt: ...


class GraphitiProposalAuthority(Protocol):
    """Resolve a raw receipt proposal to exact retained 4D/4A authority."""

    def bind_proposal(
        self,
        *,
        ingest_id: str,
        terminal_receipt: Mapping[str, object],
        proposal: ProposalDraft,
    ) -> GraphitiProposalAuthorityBinding | None: ...


class GraphitiRightsAuthority(Protocol):
    def is_current(self, request: GraphitiAdmissionRequest) -> bool: ...


@dataclass(frozen=True, slots=True)
class GraphitiAdmissionDrainReport:
    claimed: int = 0
    decided: int = 0
    projected: int = 0
    failed: int = 0
    dead_lettered: int = 0


@dataclass(frozen=True, slots=True)
class GraphitiAdmissionTelemetry:
    proposal_denominator: int
    admitted_count: int
    rejected_count: int
    held_count: int
    dead_letter_count: int
    revoked_count: int
    projected_count: int
    admission_backlog: int
    integrity_hold_receipt_count: int
    oldest_lag_seconds: int
    contiguous_projection_watermark: int | None
    projection_gap_count: int
    projection_reconciled: bool
    provider_model_calls: int = 0

    def canonical_value(self) -> dict[str, object]:
        return {
            "proposal_denominator": self.proposal_denominator,
            "admitted_count": self.admitted_count,
            "rejected_count": self.rejected_count,
            "held_count": self.held_count,
            "dead_letter_count": self.dead_letter_count,
            "revoked_count": self.revoked_count,
            "projected_count": self.projected_count,
            "admission_backlog": self.admission_backlog,
            "integrity_hold_receipt_count": self.integrity_hold_receipt_count,
            "oldest_lag_seconds": self.oldest_lag_seconds,
            "contiguous_projection_watermark": (
                self.contiguous_projection_watermark
            ),
            "projection_gap_count": self.projection_gap_count,
            "projection_reconciled": self.projection_reconciled,
            "provider_model_calls": self.provider_model_calls,
        }


@dataclass(frozen=True, slots=True)
class _MappedProposal:
    proposal_key: str
    ingest_id: str
    source_revision_id: str
    source_receipt_digest: str
    proposal_authority_binding: GraphitiProposalAuthorityBinding
    proposal: ProposalDraft
    proposal_payload: dict[str, object]
    evidence_passages: tuple[dict[str, object], ...]
    proposed_endpoints: tuple[str, str] | None
    relation_endpoint_bindings: tuple[GraphitiProposalAuthorityBinding, ...]
    relation_statement: str | None
    relation_temporal_bounds: dict[str, object] | None
    private_graph_receipt: dict[str, object] | None
    source_lineage: dict[str, object]


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise GraphitiAdmissionConsumerError(f"{field} must be an object")
    return dict(value)


def _mapping_tuple(value: object, *, field: str) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        raise GraphitiAdmissionConsumerError(f"{field} must be a list")
    return tuple(_mapping(item, field=f"{field} item") for item in value)


def _exact_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GraphitiAdmissionConsumerError(f"{field} must be an exact integer")
    return value


def _parse_proposal(value: Mapping[str, object]) -> ProposalDraft:
    raw = dict(value)
    expected_keys = {
        "local_id",
        "kind",
        "subject_placeholder",
        "object_placeholder",
        "predicate_hint",
        "confidence_basis_points",
        "uncertainty_codes",
        "rationale_codes",
        "evidence",
    }
    if set(raw) != expected_keys:
        raise GraphitiAdmissionConsumerError(
            "Graphiti proposal payload fields differ from the typed contract"
        )
    evidence_raw = raw["evidence"]
    if not isinstance(evidence_raw, list):
        raise GraphitiAdmissionConsumerError("Graphiti proposal evidence must be a list")
    evidence: list[EvidenceRange] = []
    for item in evidence_raw:
        evidence_value = _mapping(item, field="Graphiti proposal evidence")
        if set(evidence_value) != {
            "passage_id",
            "start_byte",
            "end_byte",
            "evidence_text_digest",
        }:
            raise GraphitiAdmissionConsumerError(
                "Graphiti proposal evidence fields differ from the typed contract"
            )
        evidence.append(
            EvidenceRange(
                passage_id=ExtractionPassageId.parse(
                    str(evidence_value["passage_id"])
                ),
                start_byte=evidence_value["start_byte"],  # type: ignore[arg-type]
                end_byte=evidence_value["end_byte"],  # type: ignore[arg-type]
                evidence_text_digest=str(evidence_value["evidence_text_digest"]),
            )
        )
    try:
        kind = ExtractionProposalKind(str(raw["kind"]))
        predicate = (
            None
            if raw["predicate_hint"] is None
            else ProposalPredicateHint(str(raw["predicate_hint"]))
        )
        draft = ProposalDraft(
            local_id=str(raw["local_id"]),
            kind=kind,
            subject_placeholder=str(raw["subject_placeholder"]),
            object_placeholder=(
                None
                if raw["object_placeholder"] is None
                else str(raw["object_placeholder"])
            ),
            predicate_hint=predicate,
            confidence_basis_points=raw["confidence_basis_points"],  # type: ignore[arg-type]
            uncertainty_codes=tuple(raw["uncertainty_codes"]),  # type: ignore[arg-type]
            rationale_codes=tuple(raw["rationale_codes"]),  # type: ignore[arg-type]
            evidence=tuple(evidence),
        )
    except (TypeError, ValueError) as exc:
        raise GraphitiAdmissionConsumerError(
            "Graphiti proposal payload is not a typed proposal"
        ) from exc
    if draft.kind not in _PROJECTABLE_KINDS or draft.canonical_value() != raw:
        raise GraphitiAdmissionConsumerError(
            "Graphiti proposal payload differs after typed mapping"
        )
    return draft


def graphiti_admission_request_from_value(
    value: Mapping[str, object],
) -> GraphitiAdmissionRequest:
    raw = dict(value)
    proposal_payload = _mapping(raw["proposal"], field="proposal")
    endpoints_raw = raw.get("proposed_endpoints")
    temporal_raw = raw.get("relation_temporal_bounds")
    return GraphitiAdmissionRequest(
        queue_seq=_exact_integer(raw["queue_seq"], field="queue sequence"),
        proposal_key=str(raw["proposal_key"]),
        source_receipt_digest=str(raw["source_receipt_digest"]),
        proposal_authority_binding=_proposal_authority_binding_from_value(
            _mapping(
                raw["proposal_authority_binding"],
                field="proposal authority binding",
            )
        ),
        proposal=_parse_proposal(proposal_payload),
        proposal_payload=proposal_payload,
        evidence_passages=tuple(
            _mapping(item, field="evidence passage")
            for item in raw["evidence_passages"]  # type: ignore[union-attr]
        ),
        proposed_endpoints=(
            None
            if endpoints_raw is None
            else (str(endpoints_raw[0]), str(endpoints_raw[1]))  # type: ignore[index]
        ),
        relation_endpoint_bindings=tuple(
            _proposal_authority_binding_from_value(
                _mapping(item, field="relation endpoint authority binding")
            )
            for item in raw.get("relation_endpoint_bindings", [])  # type: ignore[union-attr]
        ),
        relation_statement=(
            None
            if raw.get("relation_statement") is None
            else str(raw["relation_statement"])
        ),
        relation_temporal_bounds=(
            None
            if temporal_raw is None
            else _mapping(temporal_raw, field="relation temporal bounds")
        ),
        source_lineage=_mapping(raw["source_lineage"], field="source lineage"),
    )


def graphiti_governed_decision_from_json(value: str) -> GraphitiGovernedDecision:
    raw = _mapping(json.loads(value), field="retained admission decision")
    hold_basis = tuple(
        GraphitiRelationHoldBasis(
            dependency_id=str(item["dependency_id"]),
            authority_event_id=str(item["authority_event_id"]),
            authority_ledger_seq=_exact_integer(
                item["authority_ledger_seq"],
                field="relation HOLD basis ledger sequence",
            ),
            authority_receipt_digest=str(item["authority_receipt_digest"]),
        )
        for item in (
            _mapping(value, field="relation HOLD basis")
            for value in raw.get("relation_hold_basis", [])  # type: ignore[union-attr]
        )
    )
    return GraphitiGovernedDecision(
        proposal_key=str(raw["proposal_key"]),
        proposal_digest=str(raw["proposal_digest"]),
        proposal_kind=ExtractionProposalKind(str(raw["proposal_kind"])),
        proposal_local_id=str(raw["proposal_local_id"]),
        action=GraphitiProposalAdmissionAction(str(raw["action"])),
        decision_id=str(raw["decision_id"]),
        authority_ledger_seq=_exact_integer(
            raw["authority_ledger_seq"], field="authority ledger sequence"
        ),
        reason_code=str(raw["reason_code"]),
        authority_receipt_digest=str(raw["authority_receipt_digest"]),
        admitted_authority_id=(
            None
            if raw.get("admitted_authority_id") is None
            else str(raw["admitted_authority_id"])
        ),
        endpoint_resolution_decision_ids=tuple(
            str(item) for item in raw["endpoint_resolution_decision_ids"]  # type: ignore[union-attr]
        ),
        resolved_endpoint_names=tuple(
            str(item) for item in raw["resolved_endpoint_names"]  # type: ignore[union-attr]
        ),
        relation_hold_basis=hold_basis,
        provider_model_calls=_exact_integer(
            raw["provider_model_calls"], field="provider model calls"
        ),
    )


def graphiti_projection_receipt_from_json(value: str) -> GraphitiProjectionReceipt:
    raw = _mapping(json.loads(value), field="retained projection receipt")
    return GraphitiProjectionReceipt(
        proposal_key=str(raw["proposal_key"]),
        decision_id=str(raw["decision_id"]),
        effect_id=str(raw["effect_id"]),
        authority_watermark=_exact_integer(
            raw["authority_watermark"], field="projection authority watermark"
        ),
        receipt_digest=str(raw["receipt_digest"]),
        projector_family_id=str(raw["projector_family_id"]),
        generation_id=str(raw["generation_id"]),
        schema_version=str(raw["schema_version"]),
        trust_scope=str(raw["trust_scope"]),
        cohort_digest=(
            None
            if raw.get("cohort_digest") is None
            else str(raw["cohort_digest"])
        ),
        source_snapshot_digest=(
            None
            if raw.get("source_snapshot_digest") is None
            else str(raw["source_snapshot_digest"])
        ),
        validation_digest=(
            None
            if raw.get("validation_digest") is None
            else str(raw["validation_digest"])
        ),
        promotion_digest=(
            None
            if raw.get("promotion_digest") is None
            else str(raw["promotion_digest"])
        ),
        generation_result_digest=(
            None
            if raw.get("generation_result_digest") is None
            else str(raw["generation_result_digest"])
        ),
        source_request_digest=(
            None
            if raw.get("source_request_digest") is None
            else str(raw["source_request_digest"])
        ),
        provider_model_calls=_exact_integer(
            raw["provider_model_calls"], field="provider model calls"
        ),
    )


def graphiti_projection_reconciliation_from_json(
    value: str,
) -> tuple[GraphitiProjectionReconciliationReceipt, dict[str, object] | None]:
    """Parse a raw reconciliation or its exact-cohort binding envelope."""

    record = _mapping(
        json.loads(value),
        field="retained projection reconciliation",
    )
    binding: dict[str, object] | None = None
    raw = record
    if record.get("schema_version") is not None:
        schema_version = record.get("schema_version")
        legacy_fields = {
            "schema_version", "cohort_digest", "ingest_ids", "raw_receipt",
        }
        extension_fields = {
            *legacy_fields,
            "source_snapshot_digest",
            "source_watermark_ledger_seq",
            "source_request_digest",
            "validation_digest",
            "promotion_digest",
            "generation_result_digest",
            "projection_state_digest",
        }
        if (
            schema_version
            not in {
                GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION,
                GRAPHITI_ADMISSION_EXTENSION_RECONCILIATION_SCHEMA_VERSION,
            }
            or set(record)
            != (
                legacy_fields
                if schema_version
                == GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION
                else extension_fields
            )
            or canonical_json_bytes(record).decode("utf-8") != value
        ):
            raise GraphitiAdmissionConsumerError(
                "exact reconciliation binding envelope differs"
            )
        cohort_digest = str(record.get("cohort_digest") or "")
        validate_sha256_digest(
            cohort_digest,
            field="exact reconciliation cohort digest",
        )
        ingest_values = record.get("ingest_ids")
        if not isinstance(ingest_values, list) or any(
            not isinstance(item, str) for item in ingest_values
        ):
            raise GraphitiAdmissionConsumerError(
                "exact reconciliation ingest identities differ"
            )
        ingest_ids = _exact_ingest_ids(tuple(ingest_values))
        assert ingest_ids is not None
        raw = _mapping(
            record.get("raw_receipt"),
            field="exact reconciliation raw receipt",
        )
        binding = {
            "schema_version": schema_version,
            "cohort_digest": cohort_digest,
            "ingest_ids": list(ingest_ids),
        }
        if schema_version == GRAPHITI_ADMISSION_EXTENSION_RECONCILIATION_SCHEMA_VERSION:
            for field in (
                "source_snapshot_digest",
                "source_request_digest",
                "validation_digest",
                "promotion_digest",
                "generation_result_digest",
                "projection_state_digest",
            ):
                value_digest = str(record.get(field) or "")
                validate_sha256_digest(value_digest, field=field)
                binding[field] = value_digest
            source_watermark = _exact_integer(
                record.get("source_watermark_ledger_seq"),
                field="source watermark ledger sequence",
            )
            if source_watermark <= 0:
                raise GraphitiAdmissionConsumerError(
                    "source watermark ledger sequence must be positive"
                )
            binding["source_watermark_ledger_seq"] = source_watermark

    if set(raw) != {
        "generation_id",
        "expected_effect_ids",
        "actual_effect_ids",
        "authority_watermark",
        "receipt_digest",
        "projector_family_id",
        "provider_model_calls",
    }:
        raise GraphitiAdmissionConsumerError(
            "projection reconciliation receipt fields differ"
        )
    expected = raw.get("expected_effect_ids")
    actual = raw.get("actual_effect_ids")
    if (
        not isinstance(expected, list)
        or not isinstance(actual, list)
        or any(not isinstance(item, str) for item in (*expected, *actual))
    ):
        raise GraphitiAdmissionConsumerError(
            "projection reconciliation effect identities differ"
        )
    receipt = GraphitiProjectionReconciliationReceipt(
        generation_id=str(raw["generation_id"]),
        expected_effect_ids=tuple(expected),
        actual_effect_ids=tuple(actual),
        authority_watermark=_exact_integer(
            raw["authority_watermark"],
            field="projection reconciliation authority watermark",
        ),
        receipt_digest=str(raw["receipt_digest"]),
        projector_family_id=str(raw["projector_family_id"]),
        provider_model_calls=_exact_integer(
            raw["provider_model_calls"],
            field="projection reconciliation provider calls",
        ),
    )
    if receipt.canonical_value() != raw:
        raise GraphitiAdmissionConsumerError(
            "projection reconciliation receipt value differs"
        )
    if (
        binding is not None
        and binding["schema_version"]
        == GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION
    ):
        expected_generation_id = str(
            typed_id(
                ProjectionGenerationId,
                GRAPHITI_ADMISSION_GENERATION_IDENTITY_VERSION,
                str(binding["cohort_digest"]),
            )
        )
        if receipt.generation_id != expected_generation_id:
            raise GraphitiAdmissionConsumerError(
                "exact reconciliation generation identity differs"
            )
    elif binding is not None:
        if (
            receipt.authority_watermark
            != binding["source_watermark_ledger_seq"]
            or receipt.receipt_digest != binding["generation_result_digest"]
        ):
            raise GraphitiAdmissionConsumerError(
                "active extension reconciliation source binding differs"
            )
    return receipt, binding


def graphiti_projection_receipt_matches_binding(
    receipt: GraphitiProjectionReceipt,
    binding: Mapping[str, object],
) -> bool:
    """Match one immutable projection receipt to its exact cohort envelope."""

    if receipt.cohort_digest != binding.get("cohort_digest"):
        return False
    schema_version = binding.get("schema_version")
    if schema_version == GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION:
        return (
            receipt.schema_version
            == GRAPHITI_PROJECTION_GENERATION_SCHEMA_VERSION
            and receipt.source_request_digest is None
        )
    if schema_version != GRAPHITI_ADMISSION_EXTENSION_RECONCILIATION_SCHEMA_VERSION:
        return False
    return (
        receipt.schema_version == GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION
        and receipt.source_snapshot_digest
        == binding.get("source_snapshot_digest")
        and receipt.authority_watermark
        == binding.get("source_watermark_ledger_seq")
        and receipt.source_request_digest == binding.get("source_request_digest")
        and receipt.validation_digest == binding.get("validation_digest")
        and receipt.promotion_digest == binding.get("promotion_digest")
        and receipt.generation_result_digest
        == binding.get("generation_result_digest")
    )


def _cohort_reconciliation_rows(
    connection: sqlite3.Connection,
    *,
    cohort_digest: str,
    ingest_ids: tuple[str, ...],
) -> tuple[tuple[object, ...], ...]:
    matches: list[tuple[object, ...]] = []
    for row in connection.execute(
        "SELECT receipt_json,receipt_digest,projector_family_id,generation_id,"
        "authority_watermark FROM "
        "unpublished_graphiti_projection_reconciliations "
        "ORDER BY reconciled_at,receipt_digest"
    ).fetchall():
        _receipt, binding = graphiti_projection_reconciliation_from_json(
            str(row[0])
        )
        if binding is not None and (
            binding.get("cohort_digest") == cohort_digest
            or tuple(binding.get("ingest_ids", ())) == ingest_ids
        ):
            if (
                binding.get("cohort_digest") != cohort_digest
                or tuple(binding.get("ingest_ids", ())) != ingest_ids
            ):
                raise GraphitiAdmissionConsumerError(
                    "exact Graphiti reconciliation cohort identity differs"
                )
            matches.append(tuple(row))
    return tuple(matches)


def graphiti_decided_cohort_generation_identity(
    connection: sqlite3.Connection,
    *,
    ingest_ids: tuple[str, ...],
    require_terminal_states: bool = False,
) -> tuple[str, str]:
    """Rebuild one exact generation identity from durable cohort evidence."""

    exact = _exact_ingest_ids(ingest_ids)
    assert exact is not None
    placeholders = ",".join("?" for _ in exact)
    ingest_rows = connection.execute(
        f"""
        SELECT ingest.ingest_id, ingest.outcome, ingest.proposal_count,
               ingest.receipt_digest, receipt.receipt_json
        FROM unpublished_graphiti_ingest AS ingest
        JOIN unpublished_graphiti_receipts AS receipt USING(ingest_id)
        WHERE ingest.ingest_id IN ({placeholders})
          AND ingest.outcome='COMPLETE'
        ORDER BY ingest.ingest_id
        """,
        exact,
    ).fetchall()
    if tuple(str(row[0]) for row in ingest_rows) != exact:
        raise GraphitiAdmissionConsumerError(
            "exact generation receipts are missing or non-terminal"
        )

    source_receipts: list[dict[str, object]] = []
    proposal_denominator = 0
    for ingest_id, outcome, proposal_count, receipt_digest, receipt_json in ingest_rows:
        receipt = _mapping(
            json.loads(str(receipt_json)),
            field="exact generation terminal receipt",
        )
        unsigned = dict(receipt)
        supplied_digest = unsigned.pop("receipt_digest", None)
        actual_digest = digest_bytes(canonical_json_bytes(unsigned))
        count = _exact_integer(
            proposal_count,
            field="exact generation proposal denominator",
        )
        if (
            count < 0
            or receipt.get("ingest_id") != str(ingest_id)
            or receipt.get("outcome") != str(outcome)
            or receipt.get("proposal_count") != count
            or supplied_digest != str(receipt_digest)
            or actual_digest != str(receipt_digest)
        ):
            raise GraphitiAdmissionConsumerError(
                "exact generation terminal receipt integrity differs"
            )
        proposal_denominator += count
        source_receipts.append(
            {
                "ingest_id": str(ingest_id),
                "receipt_digest": str(receipt_digest),
                "proposal_count": count,
            }
        )

    integrity_failures = int(
        connection.execute(
            "SELECT COUNT(*) FROM "
            "unpublished_graphiti_admission_receipt_failures "
            f"WHERE ingest_id IN ({placeholders})",
            exact,
        ).fetchone()[0]
    )
    if integrity_failures:
        raise GraphitiAdmissionConsumerError(
            "exact generation has retained admission integrity failures"
        )
    queue_rows = connection.execute(
        f"""
        SELECT queue.queue_seq, queue.proposal_key, queue.ingest_id,
               queue.source_receipt_digest, queue.proposal_digest,
               queue.proposal_kind, queue.request_json, queue.request_digest,
               queue.state,
               decision.action, decision.decision_id,
               decision.authority_ledger_seq, decision.reason_code,
               decision.authority_receipt_digest, decision.decision_json,
               decision.decision_digest
        FROM unpublished_graphiti_admission_queue AS queue
        JOIN unpublished_graphiti_admission_decisions AS decision
          USING(proposal_key)
        WHERE queue.ingest_id IN ({placeholders})
        ORDER BY queue.queue_seq
        """,
        exact,
    ).fetchall()
    if len(queue_rows) != proposal_denominator or proposal_denominator == 0:
        raise GraphitiAdmissionConsumerError(
            "exact generation is not bound to a complete decided cohort"
        )
    receipt_by_ingest = {
        str(item["ingest_id"]): str(item["receipt_digest"])
        for item in source_receipts
    }
    members: list[dict[str, object]] = []
    for row in queue_rows:
        (
            queue_seq,
            proposal_key,
            ingest_id,
            source_receipt_digest,
            proposal_digest,
            proposal_kind,
            request_json,
            request_digest,
            queue_state,
            decision_action,
            decision_id,
            decision_authority_ledger_seq,
            decision_reason_code,
            decision_authority_receipt_digest,
            decision_json,
            decision_digest,
        ) = row
        retained_request = _mapping(
            json.loads(str(request_json)),
            field="exact generation admission request",
        )
        request = graphiti_admission_request_from_value(retained_request)
        decision = graphiti_governed_decision_from_json(str(decision_json))
        expected_states = (
            {"PROJECTED", "REVOKED"}
            if require_terminal_states
            and decision.action is GraphitiProposalAdmissionAction.ADMIT
            else {"DECIDED", "PROJECTED", "REVOKED"}
            if decision.action is GraphitiProposalAdmissionAction.ADMIT
            else {"TERMINAL"}
        )
        if (
            digest_bytes(canonical_json_bytes(retained_request))
            != str(request_digest)
            or digest_bytes(str(decision_json).encode("utf-8"))
            != str(decision_digest)
            or request.queue_seq != int(queue_seq)
            or request.proposal_key != str(proposal_key)
            or request.source_receipt_digest != str(source_receipt_digest)
            or receipt_by_ingest.get(str(ingest_id))
            != str(source_receipt_digest)
            or request.proposal.digest != str(proposal_digest)
            or request.proposal.kind.value != str(proposal_kind)
            or decision.proposal_key != request.proposal_key
            or decision.proposal_digest != request.proposal.digest
            or decision.proposal_kind is not request.proposal.kind
            or decision.proposal_local_id != request.proposal.local_id
            or decision.action.value != str(decision_action)
            or decision.decision_id != str(decision_id)
            or decision.authority_ledger_seq != int(decision_authority_ledger_seq)
            or decision.reason_code != str(decision_reason_code)
            or decision.authority_receipt_digest
            != str(decision_authority_receipt_digest)
            or str(queue_state) not in expected_states
        ):
            raise GraphitiAdmissionConsumerError(
                "exact generation decision integrity differs"
            )
        members.append(
            {
                "ingest_id": str(ingest_id),
                "proposal_key": request.proposal_key,
                "proposal_envelope_id": str(
                    request.proposal_authority_binding.proposal_envelope.proposal_id
                ),
                "decision_digest": str(decision_digest),
                "decision": decision.canonical_value(),
            }
        )
    return graphiti_admission_generation_identity(
        ingest_ids=exact,
        source_receipts=tuple(source_receipts),
        members=tuple(members),
    )


@contextmanager
def _transaction(connection: sqlite3.Connection):
    if connection.in_transaction:
        name = "graphiti_admission_savepoint"
        connection.execute(f"SAVEPOINT {name}")
        try:
            yield
            connection.execute(f"RELEASE SAVEPOINT {name}")
        except Exception:
            connection.execute(f"ROLLBACK TO SAVEPOINT {name}")
            connection.execute(f"RELEASE SAVEPOINT {name}")
            raise
        return
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
        connection.commit()
    except Exception:
        connection.rollback()
        raise


class GraphitiAdmissionConsumer:
    """Provider-free durable consumer for retained proposal receipts."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        proposal_authority: GraphitiProposalAuthority,
        authority: GovernedGraphitiAdmissionAuthority,
        projector: GovernedGraphitiProjector,
        rights: GraphitiRightsAuthority,
        clock: Callable[[], datetime] | None = None,
        max_attempts: int = 3,
        lease_seconds: int = 60,
        projection_generation_id: str | None = None,
    ) -> None:
        if max_attempts <= 0 or lease_seconds <= 0:
            raise ValueError("admission retry and lease bounds must be positive")
        self._connection = connection
        self._proposal_authority = proposal_authority
        self._authority = authority
        self._projector = projector
        self._rights = rights
        self._clock = clock or (lambda: datetime.now(tz=UTC))
        self._max_attempts = max_attempts
        self._lease_seconds = lease_seconds
        self._projection_generation_id = (
            None
            if projection_generation_id is None
            else str(ProjectionGenerationId.parse(projection_generation_id))
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise GraphitiAdmissionConsumerError("admission clock must be timezone-aware")
        return value.astimezone(UTC)

    @staticmethod
    def _time_text(value: datetime) -> str:
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    def enqueue_complete_receipts(
        self, *, ingest_ids: tuple[str, ...] | None = None
    ) -> int:
        exact = _exact_ingest_ids(ingest_ids)
        statement = """
            SELECT ingest.ingest_id, ingest.outcome, ingest.proposal_count,
                   ingest.entity_count, ingest.relation_count,
                   ingest.receipt_digest, receipt.receipt_json
            FROM unpublished_graphiti_ingest AS ingest
            JOIN unpublished_graphiti_receipts AS receipt USING(ingest_id)
            WHERE ingest.outcome='COMPLETE'
            """
        parameters: tuple[object, ...] = ()
        if exact is not None:
            statement += " AND ingest.ingest_id IN (" + ",".join(
                "?" for _ in exact
            ) + ")"
            parameters = exact
        statement += " ORDER BY ingest.at, ingest.ingest_id"
        rows = self._connection.execute(statement, parameters).fetchall()
        if exact is not None and tuple(sorted(str(row[0]) for row in rows)) != exact:
            raise GraphitiAdmissionConsumerError(
                "exact Graphiti admission receipts are missing or non-terminal"
            )
        mapped_receipts: list[tuple[str, tuple[_MappedProposal, ...]]] = []
        for row in rows:
            ingest_id = str(row[0])
            receipt_digest = str(row[5])
            try:
                requests = self._map_receipt(
                    ingest_id=ingest_id,
                    outcome=str(row[1]),
                    proposal_count=int(row[2]),
                    entity_count=int(row[3]),
                    relation_count=int(row[4]),
                    receipt_digest=receipt_digest,
                    receipt_json=str(row[6]),
                )
            except _HistoricalRawProposalHold as exc:
                if exact is not None:
                    raise GraphitiAdmissionConsumerError(
                        "exact Graphiti admission receipt lacks durable "
                        f"ProposalEnvelope authority: {ingest_id}"
                    ) from exc
                continue
            except (
                GraphitiAdmissionConsumerError,
                IndexError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                self._record_receipt_failure(
                    ingest_id=ingest_id,
                    receipt_digest=receipt_digest,
                    detail=str(exc),
                )
                if exact is not None:
                    raise GraphitiAdmissionConsumerError(
                        "exact Graphiti admission receipt is invalid: "
                        f"{ingest_id}"
                    ) from exc
                continue
            mapped_receipts.append((ingest_id, requests))
        enqueued = 0
        with _transaction(self._connection):
            for ingest_id, requests in mapped_receipts:
                for mapped in requests:
                    enqueued += self._insert_request(mapped)
                self._connection.execute(
                    "DELETE FROM unpublished_graphiti_admission_receipt_failures "
                    "WHERE ingest_id=?",
                    (ingest_id,),
                )
        return enqueued

    def _map_receipt(
        self,
        *,
        ingest_id: str,
        outcome: str,
        proposal_count: int,
        entity_count: int,
        relation_count: int,
        receipt_digest: str,
        receipt_json: str,
    ) -> tuple[_MappedProposal, ...]:
        if outcome not in _TERMINAL_INGEST_OUTCOMES:
            raise GraphitiAdmissionConsumerError(
                "only terminal proposal receipts may enter admission"
            )
        receipt = _mapping(json.loads(receipt_json), field="Graphiti receipt")
        unsigned = dict(receipt)
        supplied_digest = unsigned.pop("receipt_digest", None)
        actual_digest = digest_bytes(canonical_json_bytes(unsigned))
        if supplied_digest != receipt_digest or actual_digest != receipt_digest:
            raise GraphitiAdmissionConsumerError(
                "Graphiti retained receipt integrity differs"
            )
        if (
            receipt.get("ingest_id") != ingest_id
            or receipt.get("outcome") != outcome
            or receipt.get("workspace_group") != GRAPHITI_WORKSPACE_GROUP
            or receipt.get("profile") != "EVALUATION"
        ):
            raise GraphitiAdmissionConsumerError(
                "Graphiti terminal receipt authority binding differs"
            )
        proposals = _mapping_tuple(receipt.get("proposals"), field="proposals")
        passages = _mapping_tuple(receipt.get("passages"), field="passages")
        entities = _mapping_tuple(receipt.get("entities"), field="entities")
        relations = _mapping_tuple(receipt.get("relations"), field="relations")
        if proposal_count != len(proposals) or receipt.get("proposal_count") != len(
            proposals
        ):
            raise GraphitiAdmissionConsumerError(
                "Graphiti terminal proposal denominator differs"
            )
        if (
            entity_count != len(entities)
            or relation_count != len(relations)
            or receipt.get("entity_count") != entity_count
            or receipt.get("relation_count") != relation_count
        ):
            raise GraphitiAdmissionConsumerError(
                "Graphiti terminal entity or relation denominator differs"
            )
        lineage = {field: receipt.get(field) for field in _LINEAGE_FIELDS}
        if any(value is None for value in lineage.values()):
            raise GraphitiAdmissionConsumerError(
                "Graphiti source lineage is incomplete"
            )
        passages_by_id = {
            str(item.get("passage_id")): item for item in passages
        }
        if len(passages_by_id) != len(passages):
            raise GraphitiAdmissionConsumerError(
                "Graphiti evidence passage identities must be unique"
            )
        self._validate_authority_records(receipt, lineage, passages)
        entities_by_id = {
            str(item.get("local_id")): item
            for item in entities
            if item.get("local_id") is not None
        }
        relations_by_id = {
            str(item.get("local_id")): item
            for item in relations
            if item.get("local_id") is not None
        }
        if len(entities_by_id) != len(entities) or len(relations_by_id) != len(
            relations
        ):
            raise GraphitiAdmissionConsumerError(
                "Graphiti entity and relation receipt identities must be unique"
            )
        parsed_proposals: list[
            tuple[dict[str, object], ProposalDraft, GraphitiProposalAuthorityBinding]
        ] = []
        proposal_by_local_id: dict[str, ProposalDraft] = {}
        binding_by_local_id: dict[str, GraphitiProposalAuthorityBinding] = {}
        for proposal_payload in proposals:
            proposal = _parse_proposal(proposal_payload)
            if proposal.local_id in proposal_by_local_id:
                raise GraphitiAdmissionConsumerError(
                    "Graphiti proposal local identities must be unique"
                )
            proposal_by_local_id[proposal.local_id] = proposal
            for evidence in proposal.evidence:
                passage = passages_by_id.get(str(evidence.passage_id))
                if passage is None:
                    raise GraphitiAdmissionConsumerError(
                        "Graphiti proposal evidence passage is missing"
                    )
                byte_length = passage.get("byte_length")
                if (
                    isinstance(byte_length, bool)
                    or not isinstance(byte_length, int)
                    or evidence.end_byte > byte_length
                ):
                    raise GraphitiAdmissionConsumerError(
                        "Graphiti proposal evidence differs from retained passage bytes"
                    )
            authority_binding = self._proposal_authority.bind_proposal(
                ingest_id=ingest_id,
                terminal_receipt=receipt,
                proposal=proposal,
            )
            if authority_binding is None:
                raise _HistoricalRawProposalHold(
                    "Graphiti raw receipt lacks durable ProposalEnvelope authority"
                )
            if authority_binding.proposal_envelope.local_id != proposal.local_id:
                raise GraphitiAdmissionConsumerError(
                    "Graphiti ProposalEnvelope names another raw proposal"
                )
            binding_by_local_id[proposal.local_id] = authority_binding
            parsed_proposals.append((proposal_payload, proposal, authority_binding))

        entity_local_id_by_uuid: dict[str, str] = {}
        for local_id, payload in entities_by_id.items():
            if payload.get("source_registry_id") is True:
                continue
            node_uuid = payload.get("uuid")
            if not isinstance(node_uuid, str) or not node_uuid:
                if local_id in proposal_by_local_id:
                    raise GraphitiAdmissionConsumerError(
                        "Graphiti entity proposal lacks its exact private node link"
                    )
                continue
            if node_uuid in entity_local_id_by_uuid:
                raise GraphitiAdmissionConsumerError(
                    "Graphiti private node identities must be unique within a receipt"
                )
            entity_local_id_by_uuid[node_uuid] = local_id

        result = []
        for proposal_payload, proposal, authority_binding in parsed_proposals:
            relation_payload = relations_by_id.get(proposal.local_id)
            entity_payload = entities_by_id.get(proposal.local_id)
            relation_statement: str | None = None
            relation_temporal_bounds: dict[str, object] | None = None
            proposed_endpoints: tuple[str, str] | None = None
            relation_endpoint_bindings: tuple[
                GraphitiProposalAuthorityBinding, ...
            ] = ()
            if proposal.kind is ExtractionProposalKind.RELATION:
                if relation_payload is None:
                    raise GraphitiAdmissionConsumerError(
                        "Graphiti relation proposal lacks endpoint and temporal receipt"
                    )
                for field in ("source_node_uuid", "target_node_uuid"):
                    if field not in relation_payload:
                        raise GraphitiAdmissionConsumerError(
                            "Graphiti relation endpoint receipt is incomplete"
                        )
                relation_statement = str(relation_payload.get("fact") or "")
                if not relation_statement:
                    raise GraphitiAdmissionConsumerError(
                        "Graphiti relation proposal lacks its retained statement"
                    )
                relation_temporal_bounds = {
                    field: relation_payload[field]
                    for field in ("valid_at", "invalid_at", "expired_at")
                    if field in relation_payload
                }
                proposed_endpoints = (
                    proposal.subject_placeholder,
                    str(proposal.object_placeholder),
                )
                private_endpoint_ids = tuple(
                    relation_payload[field]
                    for field in ("source_node_uuid", "target_node_uuid")
                )
                if (
                    any(
                        not isinstance(item, str) or not item
                        for item in private_endpoint_ids
                    )
                    or private_endpoint_ids[0] == private_endpoint_ids[1]
                ):
                    raise GraphitiAdmissionConsumerError(
                        "Graphiti relation private endpoint links are invalid"
                    )
                endpoint_local_ids = tuple(
                    entity_local_id_by_uuid.get(str(item))
                    for item in private_endpoint_ids
                )
                if any(item is None for item in endpoint_local_ids):
                    raise GraphitiAdmissionConsumerError(
                        "Graphiti relation lacks exact same-receipt entity endpoints"
                    )
                endpoint_proposals = tuple(
                    proposal_by_local_id.get(str(item)) for item in endpoint_local_ids
                )
                if (
                    any(item is None for item in endpoint_proposals)
                    or any(
                        item.kind is not ExtractionProposalKind.ENTITY_MENTION
                        for item in endpoint_proposals
                        if item is not None
                    )
                    or tuple(
                        item.subject_placeholder
                        for item in endpoint_proposals
                        if item is not None
                    )
                    != proposed_endpoints
                ):
                    raise GraphitiAdmissionConsumerError(
                        "Graphiti relation endpoints do not bind exact entity proposals"
                    )
                relation_endpoint_bindings = tuple(
                    binding_by_local_id[str(item)] for item in endpoint_local_ids
                )
            elif proposal.kind is ExtractionProposalKind.ENTITY_EQUIVALENCE:
                proposed_endpoints = (
                    proposal.subject_placeholder,
                    str(proposal.object_placeholder),
                )
                equivalence_evidence = {
                    canonical_json_bytes(item.canonical_value())
                    for item in proposal.evidence
                }
                candidates = tuple(
                    candidate
                    for candidate in proposal_by_local_id.values()
                    if candidate.kind is ExtractionProposalKind.ENTITY_MENTION
                    and candidate.subject_placeholder in proposed_endpoints
                    and len(candidate.evidence) == 1
                    and canonical_json_bytes(
                        candidate.evidence[0].canonical_value()
                    )
                    in equivalence_evidence
                )
                if (
                    len(candidates) != 2
                    or len({item.local_id for item in candidates}) != 2
                    or sorted(item.subject_placeholder for item in candidates)
                    != sorted(proposed_endpoints)
                    or {
                        canonical_json_bytes(item.evidence[0].canonical_value())
                        for item in candidates
                    }
                    != equivalence_evidence
                ):
                    raise GraphitiAdmissionConsumerError(
                        "Graphiti entity equivalence lacks two exact entity mentions"
                    )
                ordered_candidates = tuple(
                    sorted(
                        candidates,
                        key=lambda item: (
                            proposed_endpoints.index(item.subject_placeholder),
                            item.local_id,
                        ),
                    )
                )
                relation_endpoint_bindings = tuple(
                    binding_by_local_id[item.local_id]
                    for item in ordered_candidates
                )
            elif (
                entity_payload is None
                or entity_payload.get("name") != proposal.subject_placeholder
                or entity_payload.get("source_registry_id") is True
            ):
                raise GraphitiAdmissionConsumerError(
                    "Graphiti entity proposal lacks its retained mention receipt"
                )
            referenced_passages = tuple(
                passages_by_id[str(evidence.passage_id)]
                for evidence in proposal.evidence
            )
            proposal_key = digest_canonical(
                {
                    "source_receipt_digest": receipt_digest,
                    "graphiti_attempt_id": (
                        authority_binding.graphiti_attempt_id
                    ),
                    "proposal_envelope_id": str(
                        authority_binding.proposal_envelope.proposal_id
                    ),
                    "proposal_envelope_digest": (
                        authority_binding.proposal_envelope.canonical_digest
                    ),
                }
            )
            result.append(
                _MappedProposal(
                    proposal_key=proposal_key,
                    ingest_id=ingest_id,
                    source_revision_id=str(lineage["revision_id"]),
                    source_receipt_digest=receipt_digest,
                    proposal_authority_binding=authority_binding,
                    proposal=proposal,
                    proposal_payload=proposal_payload,
                    evidence_passages=referenced_passages,
                    proposed_endpoints=proposed_endpoints,
                    relation_endpoint_bindings=relation_endpoint_bindings,
                    relation_statement=relation_statement,
                    relation_temporal_bounds=relation_temporal_bounds,
                    private_graph_receipt=(
                        None if relation_payload is None else relation_payload
                    ),
                    source_lineage=lineage,
                )
            )
        return tuple(result)

    def _validate_authority_records(
        self,
        receipt: Mapping[str, object],
        lineage: Mapping[str, object],
        passages: tuple[dict[str, object], ...],
    ) -> None:
        record_ids = receipt.get("authority_record_ids")
        if not isinstance(record_ids, list) or not record_ids:
            raise GraphitiAdmissionConsumerError(
                "Graphiti receipt lacks retained source authority"
            )
        if len({str(item) for item in record_ids}) != len(record_ids):
            raise GraphitiAdmissionConsumerError(
                "Graphiti source authority record identities must be unique"
            )
        records: dict[str, dict[str, object]] = {}
        for record_id in record_ids:
            row = self._connection.execute(
                "SELECT record_digest, record_json "
                "FROM unpublished_graphiti_authority_records WHERE record_id=?",
                (str(record_id),),
            ).fetchone()
            if row is None:
                raise GraphitiAdmissionConsumerError(
                    "Graphiti receipt source authority record is missing"
                )
            raw = str(row[1])
            if digest_bytes(raw.encode("utf-8")) != str(row[0]):
                raise GraphitiAdmissionConsumerError(
                    "Graphiti receipt source authority integrity differs"
                )
            record = _mapping(json.loads(raw), field="source authority record")
            if record.get("record_id") != str(record_id):
                raise GraphitiAdmissionConsumerError(
                    "Graphiti source authority record identity differs"
                )
            records[str(record_id)] = record
        revision_id = str(lineage["revision_id"])
        revision = records.get(revision_id)
        if (
            revision is None
            or revision.get("record_type") != "SOURCE_REVISION"
            or revision.get("source_id") != lineage["source_id"]
            or revision.get("item_key") != lineage["item_key"]
        ):
            raise GraphitiAdmissionConsumerError(
                "Graphiti source revision authority binding differs"
            )
        for passage in passages:
            admission = records.get(str(passage.get("admission_id") or ""))
            access = records.get(str(passage.get("access_decision_id") or ""))
            if (
                admission is None
                or admission.get("record_type") != "OBJECT_ADMISSION"
                or admission.get("revision_id") != revision_id
                or admission.get("decision") != "ADMIT"
                or access is None
                or access.get("record_type") != "OBJECT_ACCESS_DECISION"
                or access.get("revision_id") != revision_id
                or access.get("decision") != "ALLOW"
            ):
                raise GraphitiAdmissionConsumerError(
                    "Graphiti passage rights do not bind retained source authority"
                )
        if not passages:
            raise GraphitiAdmissionConsumerError(
                "Graphiti proposal source has no retained passages"
            )

    def _insert_request(self, mapped: _MappedProposal) -> int:
        retained = self._connection.execute(
            "SELECT proposal_digest FROM unpublished_graphiti_admission_queue "
            "WHERE proposal_key=?",
            (mapped.proposal_key,),
        ).fetchone()
        if retained is not None:
            if str(retained[0]) != mapped.proposal.digest:
                raise GraphitiAdmissionConsumerError(
                    "Graphiti admission proposal identity was reused"
                )
            return 0
        now = self._time_text(self._now())
        cursor = self._connection.execute(
            """
            INSERT INTO unpublished_graphiti_admission_queue(
                proposal_key, ingest_id, source_revision_id,
                source_receipt_digest, proposal_digest, proposal_kind,
                request_json, request_digest, state, created_at, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,'READY',?,?)
            """,
            (
                mapped.proposal_key,
                mapped.ingest_id,
                mapped.source_revision_id,
                mapped.source_receipt_digest,
                mapped.proposal.digest,
                mapped.proposal.kind.value,
                "{}",
                mapped.proposal_key,
                now,
                now,
            ),
        )
        if cursor.lastrowid is None:
            raise GraphitiAdmissionConsumerError(
                "Graphiti admission queue did not return an identity"
            )
        queue_seq = cursor.lastrowid
        request = GraphitiAdmissionRequest(
            queue_seq=queue_seq,
            proposal_key=mapped.proposal_key,
            source_receipt_digest=mapped.source_receipt_digest,
            proposal_authority_binding=mapped.proposal_authority_binding,
            proposal=mapped.proposal,
            proposal_payload=mapped.proposal_payload,
            evidence_passages=mapped.evidence_passages,
            proposed_endpoints=mapped.proposed_endpoints,
            relation_endpoint_bindings=mapped.relation_endpoint_bindings,
            relation_statement=mapped.relation_statement,
            relation_temporal_bounds=mapped.relation_temporal_bounds,
            source_lineage=mapped.source_lineage,
        )
        retained_request = request.canonical_value()
        retained_request["private_graph_receipt"] = mapped.private_graph_receipt
        request_bytes = canonical_json_bytes(retained_request)
        if len(request_bytes) > _MAX_ADMISSION_REQUEST_BYTES:
            raise GraphitiAdmissionConsumerError(
                "Graphiti admission request exceeds its SQLite manifest bound"
            )
        self._connection.execute(
            "UPDATE unpublished_graphiti_admission_queue "
            "SET source_receipt_digest=?, request_json=?, request_digest=? "
            "WHERE proposal_key=?",
            (
                mapped.source_receipt_digest,
                request_bytes.decode("utf-8"),
                digest_bytes(request_bytes),
                mapped.proposal_key,
            ),
        )
        return 1

    def _record_receipt_failure(
        self, *, ingest_id: str, receipt_digest: str, detail: str
    ) -> None:
        now = self._time_text(self._now())
        with _transaction(self._connection):
            self._connection.execute(
                """
                INSERT INTO unpublished_graphiti_admission_receipt_failures(
                    ingest_id, receipt_digest, failure_code, detail,
                    occurrence_count, first_seen_at, last_seen_at
                ) VALUES(?,?,'INTEGRITY_INVALID',?,1,?,?)
                ON CONFLICT(ingest_id) DO UPDATE SET
                    receipt_digest=excluded.receipt_digest,
                    failure_code=excluded.failure_code,
                    detail=excluded.detail,
                    occurrence_count=occurrence_count+1,
                    last_seen_at=excluded.last_seen_at
                """,
                (ingest_id, receipt_digest, detail[:4096], now, now),
            )

    def _claim_next(
        self,
        worker_id: str,
        *,
        ingest_ids: tuple[str, ...] | None = None,
    ) -> tuple[str, GraphitiAdmissionRequest, str] | None:
        if not worker_id or len(worker_id.encode("utf-8")) > 256:
            raise ValueError("admission worker identity is invalid")
        exact = _exact_ingest_ids(ingest_ids)
        now = self._now()
        now_text = self._time_text(now)
        until = self._time_text(now + timedelta(seconds=self._lease_seconds))
        with _transaction(self._connection):
            while True:
                statement = """
                    SELECT proposal_key, ingest_id, request_json, request_digest,
                           state, source_receipt_digest, proposal_digest,
                           proposal_kind
                    FROM unpublished_graphiti_admission_queue
                    WHERE (state='READY'
                       OR (state='CLAIMED' AND claim_until<=?))
                      AND CASE
                            WHEN json_valid(request_json)
                             AND json_type(
                                request_json, '$.proposal_authority_binding'
                             ) IS NULL
                             AND json_type(request_json, '$.proposal')='object'
                             AND json_type(
                                request_json, '$.evidence_passages'
                             )='array'
                             AND json_type(
                                request_json, '$.source_lineage'
                             )='object'
                             AND json_type(
                                request_json, '$.proposal_key'
                             )='text'
                             AND json_type(
                                request_json, '$.source_receipt_digest'
                             )='text'
                            THEN 0
                            ELSE 1
                          END
                    """
                parameters: tuple[object, ...] = (now_text,)
                if exact is not None:
                    statement += " AND ingest_id IN (" + ",".join(
                        "?" for _ in exact
                    ) + ")"
                    parameters = (*parameters, *exact)
                statement += (
                    " ORDER BY CASE proposal_kind "
                    "WHEN 'ENTITY_MENTION' THEN 0 "
                    "WHEN 'ENTITY_EQUIVALENCE' THEN 1 ELSE 2 END, "
                    "queue_seq LIMIT 1"
                )
                row = self._connection.execute(statement, parameters).fetchone()
                if row is None:
                    return None
                (
                    proposal_key,
                    ingest_id,
                    request_json,
                    request_digest,
                    state,
                    source_receipt_digest,
                    proposal_digest,
                    proposal_kind,
                ) = map(str, row)
                if proposal_kind == ExtractionProposalKind.RELATION.value:
                    unresolved = """
                        SELECT COUNT(*)
                        FROM unpublished_graphiti_admission_queue
                        WHERE proposal_kind!='RELATION'
                          AND state NOT IN ('DECIDED','TERMINAL','PROJECTED','REVOKED')
                        """
                    unresolved_parameters: tuple[object, ...] = ()
                    if exact is not None:
                        unresolved += " AND ingest_id IN (" + ",".join(
                            "?" for _ in exact
                        ) + ")"
                        unresolved_parameters = exact
                    if int(
                        self._connection.execute(
                            unresolved, unresolved_parameters
                        ).fetchone()[0]
                    ):
                        return None
                try:
                    retained_request = _mapping(
                        json.loads(request_json),
                        field="queued admission request",
                    )
                    request = graphiti_admission_request_from_value(retained_request)
                    if (
                        digest_bytes(canonical_json_bytes(retained_request))
                        != request_digest
                        or request.proposal_key != proposal_key
                        or request.source_receipt_digest != source_receipt_digest
                        or request.proposal.digest != proposal_digest
                        or request.proposal.kind.value != proposal_kind
                    ):
                        raise GraphitiAdmissionConsumerError(
                            "queued Graphiti admission request integrity differs"
                        )
                except (
                    GraphitiAdmissionConsumerError,
                    IndexError,
                    KeyError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as exc:
                    detail = f"{type(exc).__name__}: {exc}"[:4096]
                    self._connection.execute(
                        """
                        UPDATE unpublished_graphiti_admission_queue
                        SET state='DEAD_LETTER', attempt_count=?, last_error=?,
                            claim_owner=NULL, claim_until=NULL, updated_at=?
                        WHERE proposal_key=?
                        """,
                        (self._max_attempts, detail, now_text, proposal_key),
                    )
                    self._connection.execute(
                        """
                        INSERT INTO unpublished_graphiti_admission_receipt_failures(
                            ingest_id, receipt_digest, failure_code, detail,
                            occurrence_count, first_seen_at, last_seen_at
                        ) VALUES(?,?,'QUEUE_INTEGRITY_INVALID',?,1,?,?)
                        ON CONFLICT(ingest_id) DO UPDATE SET
                            failure_code=excluded.failure_code,
                            detail=excluded.detail,
                            occurrence_count=occurrence_count+1,
                            last_seen_at=excluded.last_seen_at
                        """,
                        (
                            ingest_id,
                            source_receipt_digest,
                            detail,
                            now_text,
                            now_text,
                        ),
                    )
                    continue
                updated = self._connection.execute(
                    """
                    UPDATE unpublished_graphiti_admission_queue
                    SET state='CLAIMED',
                        claim_owner=?, claim_until=?, updated_at=?
                    WHERE proposal_key=?
                      AND (state='READY'
                        OR (state='CLAIMED' AND claim_until<=?))
                    """,
                    (worker_id, until, now_text, proposal_key, now_text),
                ).rowcount
                if updated != 1:
                    continue
                break
        return proposal_key, request, state

    def drain(
        self,
        *,
        worker_id: str,
        limit: int = 100,
        ingest_ids: tuple[str, ...] | None = None,
        stop_on_failure: bool = False,
    ) -> GraphitiAdmissionDrainReport:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("admission drain limit must be positive")
        if not isinstance(stop_on_failure, bool):
            raise TypeError("admission stop-on-failure control must be boolean")
        exact = _exact_ingest_ids(ingest_ids)
        claimed = decided = failed = dead_lettered = 0
        for _ in range(limit):
            claim = self._claim_next(worker_id, ingest_ids=exact)
            if claim is None:
                break
            proposal_key, request, _previous_state = claim
            claimed += 1
            try:
                rights_current = self._rights.is_current(request)
                if stop_on_failure and not rights_current:
                    raise GraphitiAdmissionConsumerError(
                        "exact Graphiti campaign rights drifted before decision"
                    )
                required_action = (
                    None
                    if rights_current
                    else GraphitiProposalAdmissionAction.REJECT
                )
                decide = (
                    self._authority.decide_relation_admission
                    if request.proposal.kind is ExtractionProposalKind.RELATION
                    else self._authority.decide_entity_resolution
                )
                decision = decide(
                    request,
                    required_action=required_action,
                    idempotency_key=f"graphiti-admit:{proposal_key}",
                )
                if (
                    required_action is not None
                    and decision.action is not required_action
                ):
                    raise GraphitiAdmissionConsumerError(
                        "governed authority did not honour the rights rejection"
                    )
                if (
                    request.proposal.kind is ExtractionProposalKind.RELATION
                    and decision.action is GraphitiProposalAdmissionAction.ADMIT
                    and not self._authority.relation_endpoint_resolutions_current(
                        request, decision
                    )
                ):
                    raise GraphitiAdmissionConsumerError(
                        "relation endpoints lack current governed resolutions"
                    )
                self._retain_decision(request, decision)
                decided += 1
            except Exception as exc:
                failed += 1
                if self._record_work_failure(proposal_key, exc):
                    dead_lettered += 1
                if stop_on_failure:
                    break
        return GraphitiAdmissionDrainReport(
            claimed=claimed,
            decided=decided,
            projected=0,
            failed=failed,
            dead_lettered=dead_lettered,
        )

    def preflight_decided_cohort(
        self,
        *,
        ingest_ids: tuple[str, ...],
    ) -> GraphitiAdmissionDrainReport:
        """Validate one revision-local cohort without projecting it."""

        return self._finalise_decided_cohort(
            ingest_ids=ingest_ids, preflight_only=True
        )

    def finalise_decided_cohort(
        self,
        *,
        ingest_ids: tuple[str, ...],
    ) -> GraphitiAdmissionDrainReport:
        """Promote one exact, completely decided cohort as one full generation.

        Admission decisions remain individual immutable 4B/4C authority
        receipts. Projection is deliberately different: every admitted member
        of the exact cohort binds the same complete Increment 4 snapshot
        generation and reconciliation receipt.
        """

        return self._finalise_decided_cohort(
            ingest_ids=ingest_ids, preflight_only=False
        )

    def _finalise_decided_cohort(
        self,
        *,
        ingest_ids: tuple[str, ...],
        preflight_only: bool,
    ) -> GraphitiAdmissionDrainReport:

        exact = _exact_ingest_ids(ingest_ids)
        assert exact is not None
        placeholders = ",".join("?" for _ in exact)
        ingest_rows = self._connection.execute(
            f"""
            SELECT ingest.ingest_id, ingest.outcome, ingest.proposal_count,
                   ingest.receipt_digest, receipt.receipt_json
            FROM unpublished_graphiti_ingest AS ingest
            JOIN unpublished_graphiti_receipts AS receipt USING(ingest_id)
            WHERE ingest.ingest_id IN ({placeholders})
              AND ingest.outcome='COMPLETE'
            ORDER BY ingest.ingest_id
            """,
            exact,
        ).fetchall()
        if tuple(str(row[0]) for row in ingest_rows) != exact:
            raise GraphitiAdmissionConsumerError(
                "exact Graphiti admission receipts are missing or non-terminal"
            )

        source_receipts: list[dict[str, object]] = []
        proposal_denominator = 0
        for ingest_id, outcome, proposal_count, receipt_digest, receipt_json in (
            ingest_rows
        ):
            receipt = _mapping(
                json.loads(str(receipt_json)),
                field="exact Graphiti terminal receipt",
            )
            unsigned = dict(receipt)
            supplied_digest = unsigned.pop("receipt_digest", None)
            actual_digest = digest_bytes(canonical_json_bytes(unsigned))
            count = _exact_integer(
                proposal_count,
                field="exact Graphiti proposal denominator",
            )
            if (
                receipt.get("ingest_id") != str(ingest_id)
                or receipt.get("outcome") != str(outcome)
                or receipt.get("proposal_count") != count
                or supplied_digest != str(receipt_digest)
                or actual_digest != str(receipt_digest)
            ):
                raise GraphitiAdmissionConsumerError(
                    "exact Graphiti terminal receipt integrity differs"
                )
            proposal_denominator += count
            source_receipts.append(
                {
                    "ingest_id": str(ingest_id),
                    "receipt_digest": str(receipt_digest),
                    "proposal_count": count,
                }
            )

        failure_count = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM "
                "unpublished_graphiti_admission_receipt_failures "
                f"WHERE ingest_id IN ({placeholders})",
                exact,
            ).fetchone()[0]
        )
        if failure_count:
            raise GraphitiAdmissionConsumerError(
                "exact Graphiti admission cohort has retained integrity failures"
            )

        queue_rows = self._connection.execute(
            f"""
            SELECT queue.queue_seq, queue.proposal_key, queue.ingest_id,
                   queue.source_receipt_digest, queue.proposal_digest,
                   queue.proposal_kind, queue.request_json,
                   queue.request_digest, queue.state,
                   decision.decision_json, decision.decision_digest
            FROM unpublished_graphiti_admission_queue AS queue
            LEFT JOIN unpublished_graphiti_admission_decisions AS decision
              USING(proposal_key)
            WHERE queue.ingest_id IN ({placeholders})
            ORDER BY CASE queue.proposal_kind
                         WHEN 'ENTITY_MENTION' THEN 0
                         WHEN 'ENTITY_EQUIVALENCE' THEN 1
                         ELSE 2
                     END,
                     queue.queue_seq
            """,
            exact,
        ).fetchall()
        if len(queue_rows) != proposal_denominator:
            raise GraphitiAdmissionConsumerError(
                "exact Graphiti cohort is not fully bound to durable ProposalEnvelope authority"
            )
        if proposal_denominator == 0:
            return GraphitiAdmissionDrainReport()

        receipt_by_ingest = {
            str(item["ingest_id"]): str(item["receipt_digest"])
            for item in source_receipts
        }
        decided_items: list[
            tuple[GraphitiAdmissionRequest, GraphitiGovernedDecision]
        ] = []
        cohort_members: list[dict[str, object]] = []
        stale_proposals: list[str] = []
        for row in queue_rows:
            (
                queue_seq,
                proposal_key,
                ingest_id,
                source_receipt_digest,
                proposal_digest,
                proposal_kind,
                request_json,
                request_digest,
                state,
                decision_json,
                decision_digest,
            ) = row
            if decision_json is None or decision_digest is None:
                raise GraphitiAdmissionConsumerError(
                    "exact Graphiti cohort is not completely decided"
                )
            retained_request = _mapping(
                json.loads(str(request_json)),
                field="queued admission request",
            )
            request = graphiti_admission_request_from_value(retained_request)
            decision = graphiti_governed_decision_from_json(str(decision_json))
            if (
                digest_bytes(canonical_json_bytes(retained_request))
                != str(request_digest)
                or digest_bytes(str(decision_json).encode("utf-8"))
                != str(decision_digest)
                or request.queue_seq != int(queue_seq)
                or request.proposal_key != str(proposal_key)
                or request.source_receipt_digest != str(source_receipt_digest)
                or receipt_by_ingest.get(str(ingest_id))
                != str(source_receipt_digest)
                or request.proposal.digest != str(proposal_digest)
                or request.proposal.kind.value != str(proposal_kind)
                or decision.proposal_key != request.proposal_key
                or decision.proposal_digest != request.proposal.digest
                or decision.proposal_kind is not request.proposal.kind
                or decision.proposal_local_id != request.proposal.local_id
            ):
                raise GraphitiAdmissionConsumerError(
                    "exact Graphiti cohort decision integrity differs"
                )
            expected_states = (
                {"DECIDED", "PROJECTED"}
                if decision.action is GraphitiProposalAdmissionAction.ADMIT
                else {"TERMINAL"}
            )
            if str(state) not in expected_states:
                raise GraphitiAdmissionConsumerError(
                    "exact Graphiti cohort contains non-terminal work"
                )
            if decision.action is GraphitiProposalAdmissionAction.ADMIT:
                current = self._rights.is_current(request)
                if request.proposal.kind is ExtractionProposalKind.RELATION:
                    current = current and (
                        decision.resolved_endpoint_names
                        == request.proposed_endpoints
                        and self._authority.relation_endpoint_resolutions_current(
                            request, decision
                        )
                    )
                if not current:
                    stale_proposals.append(request.proposal_key)
            decided_items.append((request, decision))
            cohort_members.append(
                {
                    "ingest_id": str(ingest_id),
                    "proposal_key": request.proposal_key,
                    "proposal_envelope_id": str(
                        request.proposal_authority_binding.proposal_envelope.proposal_id
                    ),
                    "decision_digest": str(decision_digest),
                    "decision": decision.canonical_value(),
                }
            )

        if stale_proposals:
            now = self._time_text(self._now())
            with _transaction(self._connection):
                self._connection.executemany(
                    """
                    UPDATE unpublished_graphiti_admission_queue
                    SET state='REVOKED', claim_owner=NULL, claim_until=NULL,
                        last_error='RIGHTS_OR_ENDPOINT_AUTHORITY_REVOKED_BEFORE_GENERATION',
                        updated_at=?
                    WHERE proposal_key=? AND state='DECIDED'
                    """,
                    ((now, item) for item in stale_proposals),
                )
            raise GraphitiAdmissionConsumerError(
                "exact Graphiti cohort lost current rights or endpoint authority"
            )

        # Per-revision native fault isolation stops here. Shared generation
        # receipts and projector/store integrity remain finalisation-wide and
        # therefore fail the complete verified subset together.
        if preflight_only:
            return GraphitiAdmissionDrainReport()

        cohort_digest, generation_id = graphiti_admission_generation_identity(
            ingest_ids=exact,
            source_receipts=tuple(source_receipts),
            members=tuple(cohort_members),
        )
        projection_requests = tuple(
            sorted(
                (
                    GraphitiProjectionRequest(request=request, decision=decision)
                    for request, decision in decided_items
                ),
                key=lambda item: item.request.proposal_key,
            )
        )
        admitted = tuple(
            item
            for item in projection_requests
            if item.decision.action is GraphitiProposalAdmissionAction.ADMIT
        )
        admitted_authority_ids = tuple(
            sorted(str(item.decision.admitted_authority_id) for item in admitted)
        )
        if len(admitted_authority_ids) != len(set(admitted_authority_ids)):
            raise GraphitiAdmissionConsumerError(
                "exact Graphiti cohort reuses canonical admitted authority identity"
            )

        existing_rows = self._connection.execute(
            f"""
            SELECT projection.proposal_key, projection.receipt_json
            FROM unpublished_graphiti_projection_receipts AS projection
            JOIN unpublished_graphiti_admission_queue AS queue USING(proposal_key)
            WHERE queue.ingest_id IN ({placeholders})
            ORDER BY projection.proposal_key
            """,
            exact,
        ).fetchall()
        reconciliation_rows = _cohort_reconciliation_rows(
            self._connection,
            cohort_digest=cohort_digest,
            ingest_ids=exact,
        )
        retained_receipts: list[GraphitiProjectionReceipt] = []
        retained_reconciliation: GraphitiProjectionReconciliationReceipt | None = None
        if existing_rows or reconciliation_rows:
            if (
                len(existing_rows) != len(admitted)
                or len(reconciliation_rows) != 1
            ):
                raise GraphitiAdmissionConsumerError(
                    "exact Graphiti generation retention is partial"
                )
            expected_by_key = {
                item.request.proposal_key: item for item in admitted
            }
            retained_binding: dict[str, object] | None = None
            for proposal_key, receipt_json in existing_rows:
                receipt = graphiti_projection_receipt_from_json(str(receipt_json))
                item = expected_by_key.get(str(proposal_key))
                if (
                    item is None
                    or receipt.cohort_digest != cohort_digest
                    or receipt.decision_id != item.decision.decision_id
                    or receipt.effect_id != item.decision.admitted_authority_id
                ):
                    raise GraphitiAdmissionConsumerError(
                        "exact Graphiti retained generation identity differs"
                    )
                retained_receipts.append(receipt)
            retained_reconciliation, retained_binding = (
                graphiti_projection_reconciliation_from_json(
                    str(reconciliation_rows[0][0])
                )
            )
            if (
                retained_reconciliation.receipt_digest
                != str(reconciliation_rows[0][1])
                or retained_reconciliation.projector_family_id
                != str(reconciliation_rows[0][2])
                or retained_reconciliation.generation_id
                != str(reconciliation_rows[0][3])
                or retained_reconciliation.authority_watermark
                != int(reconciliation_rows[0][4])
                or retained_binding is None
                or retained_binding.get("cohort_digest") != cohort_digest
                or retained_binding.get("ingest_ids") != list(exact)
                or retained_reconciliation.expected_effect_ids
                != admitted_authority_ids
                or retained_reconciliation.actual_effect_ids
                != admitted_authority_ids
            ):
                raise GraphitiAdmissionConsumerError(
                    "exact Graphiti retained reconciliation differs"
                )

        generation_idempotency_key = f"graphiti-generation:{cohort_digest}"
        result = self._projector.build_and_promote_increment4_cohort(
            projection_requests,
            cohort_digest=cohort_digest,
            generation_id=generation_id,
            idempotency_key=generation_idempotency_key,
        )
        required_watermark = max(
            decision.authority_ledger_seq for _request, decision in decided_items
        )
        if (
            result.cohort_digest != cohort_digest
            or result.authority_watermark < required_watermark
            or result.admitted_authority_ids != admitted_authority_ids
            or (
                result.schema_version
                == GRAPHITI_PROJECTION_GENERATION_SCHEMA_VERSION
                and result.generation_id != generation_id
            )
            or (
                result.schema_version
                == GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION
                and result.source_request_digest
                != digest_canonical(
                    {
                        "generation_id": generation_id,
                        "reason_code": "GRAPHITI_ADMISSION_COHORT",
                        "idempotency_key": generation_idempotency_key,
                        "purge_retired_generation": True,
                        "allow_active_extension": True,
                    }
                )
            )
        ):
            raise GraphitiAdmissionConsumerError(
                "Increment 4 generation result differs from exact admission authority"
            )

        if retained_reconciliation is not None:
            assert retained_binding is not None
            if (
                retained_reconciliation.authority_watermark
                != result.authority_watermark
                or retained_reconciliation.generation_id != result.generation_id
                or retained_binding.get("schema_version")
                != (
                    GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION
                    if result.schema_version
                    == GRAPHITI_PROJECTION_GENERATION_SCHEMA_VERSION
                    else GRAPHITI_ADMISSION_EXTENSION_RECONCILIATION_SCHEMA_VERSION
                )
                or (
                    result.schema_version
                    == GRAPHITI_PROJECTION_GENERATION_SCHEMA_VERSION
                    and retained_reconciliation.receipt_digest
                    != result.reconciliation_digest
                )
                or (
                    result.schema_version
                    == GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION
                    and (
                        retained_reconciliation.receipt_digest != result.digest
                        or retained_binding.get("source_snapshot_digest")
                        != result.source_snapshot_digest
                        or retained_binding.get("source_watermark_ledger_seq")
                        != result.authority_watermark
                        or retained_binding.get("source_request_digest")
                        != result.source_request_digest
                        or retained_binding.get("validation_digest")
                        != result.validation_digest
                        or retained_binding.get("promotion_digest")
                        != result.promotion_digest
                        or retained_binding.get("generation_result_digest")
                        != result.digest
                        or retained_binding.get("projection_state_digest")
                        != result.reconciliation_digest
                    )
                )
                or any(
                    receipt.authority_watermark != result.authority_watermark
                    or receipt.generation_id != result.generation_id
                    or receipt.schema_version != result.schema_version
                    or receipt.source_snapshot_digest
                    != result.source_snapshot_digest
                    or receipt.source_request_digest
                    != result.source_request_digest
                    or receipt.validation_digest != result.validation_digest
                    or receipt.promotion_digest != result.promotion_digest
                    or receipt.generation_result_digest != result.digest
                    for receipt in retained_receipts
                )
                or any(
                    not graphiti_projection_receipt_matches_binding(
                        receipt, retained_binding
                    )
                    for receipt in retained_receipts
                )
            ):
                raise GraphitiAdmissionConsumerError(
                    "Increment 4 active generation differs from retained reconciliation"
                )
            return GraphitiAdmissionDrainReport()

        generation_result_digest = result.digest
        receipts: list[GraphitiProjectionReceipt] = []
        for item in admitted:
            material: dict[str, object] = {
                "proposal_key": item.request.proposal_key,
                "decision_id": item.decision.decision_id,
                "effect_id": item.decision.admitted_authority_id,
                "authority_watermark": result.authority_watermark,
                "projector_family_id": "graph.increment4.admitted",
                "generation_id": result.generation_id,
                "schema_version": (
                    result.schema_version
                ),
                "trust_scope": "ADMITTED",
                "cohort_digest": cohort_digest,
                "source_snapshot_digest": result.source_snapshot_digest,
                "validation_digest": result.validation_digest,
                "promotion_digest": result.promotion_digest,
                "generation_result_digest": generation_result_digest,
                "provider_model_calls": 0,
            }
            if result.schema_version == GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION:
                material["source_request_digest"] = result.source_request_digest
            receipts.append(
                GraphitiProjectionReceipt(
                    **material,
                    receipt_digest=digest_canonical(material),
                )
            )
        reconciliation = GraphitiProjectionReconciliationReceipt(
            generation_id=result.generation_id,
            expected_effect_ids=admitted_authority_ids,
            actual_effect_ids=admitted_authority_ids,
            authority_watermark=result.authority_watermark,
            receipt_digest=(
                result.reconciliation_digest
                if result.schema_version
                == GRAPHITI_PROJECTION_GENERATION_SCHEMA_VERSION
                else result.digest
            ),
        )
        now = self._time_text(self._now())
        with _transaction(self._connection):
            retained_count = int(
                self._connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM unpublished_graphiti_projection_receipts AS projection
                    JOIN unpublished_graphiti_admission_queue AS queue
                      USING(proposal_key)
                    WHERE queue.ingest_id IN ({placeholders})
                    """,
                    exact,
                ).fetchone()[0]
            )
            retained_reconciliation_count = len(
                _cohort_reconciliation_rows(
                    self._connection,
                    cohort_digest=cohort_digest,
                    ingest_ids=exact,
                )
            )
            if retained_count or retained_reconciliation_count:
                raise GraphitiAdmissionConsumerError(
                    "exact Graphiti generation retention raced another writer"
                )
            for item, receipt in zip(admitted, receipts, strict=True):
                encoded = canonical_json_bytes(
                    receipt.canonical_value()
                ).decode("utf-8")
                self._connection.execute(
                    """
                    INSERT INTO unpublished_graphiti_projection_receipts(
                        proposal_key, effect_id, authority_watermark,
                        projector_family_id, generation_id, schema_version,
                        trust_scope, receipt_json, receipt_digest, projected_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        item.request.proposal_key,
                        receipt.effect_id,
                        receipt.authority_watermark,
                        receipt.projector_family_id,
                        receipt.generation_id,
                        receipt.schema_version,
                        receipt.trust_scope,
                        encoded,
                        receipt.receipt_digest,
                        now,
                    ),
                )
                updated = self._connection.execute(
                    """
                    UPDATE unpublished_graphiti_admission_queue
                    SET state='PROJECTED', claim_owner=NULL, claim_until=NULL,
                        attempt_count=0, last_error=NULL, updated_at=?
                    WHERE proposal_key=? AND state='DECIDED'
                    """,
                    (now, item.request.proposal_key),
                ).rowcount
                if updated != 1:
                    raise GraphitiAdmissionConsumerError(
                        "exact Graphiti admitted cohort state changed before retention"
                    )
            reconciliation_value: dict[str, object] = {
                "schema_version": (
                    GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION
                    if result.schema_version
                    == GRAPHITI_PROJECTION_GENERATION_SCHEMA_VERSION
                    else GRAPHITI_ADMISSION_EXTENSION_RECONCILIATION_SCHEMA_VERSION
                ),
                "cohort_digest": cohort_digest,
                "ingest_ids": list(exact),
                "raw_receipt": reconciliation.canonical_value(),
            }
            if result.schema_version == GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION:
                reconciliation_value.update(
                    {
                        "source_snapshot_digest": result.source_snapshot_digest,
                        "source_watermark_ledger_seq": result.authority_watermark,
                        "source_request_digest": result.source_request_digest,
                        "validation_digest": result.validation_digest,
                        "promotion_digest": result.promotion_digest,
                        "generation_result_digest": result.digest,
                        "projection_state_digest": result.reconciliation_digest,
                    }
                )
            reconciliation_json = canonical_json_bytes(
                reconciliation_value
            ).decode("utf-8")
            self._connection.execute(
                """
                INSERT INTO unpublished_graphiti_projection_reconciliations(
                    receipt_digest, projector_family_id, generation_id,
                    authority_watermark, receipt_json, reconciled_at
                ) VALUES(?,?,?,?,?,?)
                """,
                (
                    reconciliation.receipt_digest,
                    reconciliation.projector_family_id,
                    reconciliation.generation_id,
                    reconciliation.authority_watermark,
                    reconciliation_json,
                    now,
                ),
            )
        return GraphitiAdmissionDrainReport(projected=len(receipts))

    def _mark_revoked_before_projection(
        self, request: GraphitiAdmissionRequest
    ) -> None:
        now = self._time_text(self._now())
        with _transaction(self._connection):
            self._connection.execute(
                """
                UPDATE unpublished_graphiti_admission_queue
                SET state='REVOKED', claim_owner=NULL, claim_until=NULL,
                    last_error='RIGHTS_REVOKED_BEFORE_PROJECTION', updated_at=?
                WHERE proposal_key=?
                """,
                (now, request.proposal_key),
            )

    def _retained_decision(self, proposal_key: str) -> GraphitiGovernedDecision | None:
        row = self._connection.execute(
            "SELECT decision_json FROM unpublished_graphiti_admission_decisions "
            "WHERE proposal_key=?",
            (proposal_key,),
        ).fetchone()
        return None if row is None else graphiti_governed_decision_from_json(str(row[0]))

    def _retain_decision(
        self,
        request: GraphitiAdmissionRequest,
        decision: GraphitiGovernedDecision,
    ) -> None:
        if decision.proposal_key != request.proposal_key:
            raise GraphitiAdmissionConsumerError(
                "governed admission decision names another proposal"
            )
        if (
            decision.proposal_digest != request.proposal.digest
            or decision.proposal_kind is not request.proposal.kind
            or decision.proposal_local_id != request.proposal.local_id
        ):
            raise GraphitiAdmissionConsumerError(
                "governed admission decision does not bind the exact proposal"
            )
        if (
            decision.action is GraphitiProposalAdmissionAction.ADMIT
            and request.proposal.kind is ExtractionProposalKind.RELATION
            and decision.resolved_endpoint_names != request.proposed_endpoints
        ):
            raise GraphitiAdmissionConsumerError(
                "relation authority did not resolve the exact proposed endpoints"
            )
        value = decision.canonical_value()
        encoded = canonical_json_bytes(value).decode("utf-8")
        retained_digest = digest_bytes(encoded.encode("utf-8"))
        now = self._time_text(self._now())
        state = (
            "DECIDED"
            if decision.action is GraphitiProposalAdmissionAction.ADMIT
            else "TERMINAL"
        )
        with _transaction(self._connection):
            existing = self._connection.execute(
                "SELECT decision_json FROM unpublished_graphiti_admission_decisions "
                "WHERE proposal_key=?",
                (request.proposal_key,),
            ).fetchone()
            if existing is not None and str(existing[0]) != encoded:
                raise GraphitiAdmissionConsumerError(
                    "Graphiti proposal already has another governed decision"
                )
            if existing is None:
                self._connection.execute(
                    """
                    INSERT INTO unpublished_graphiti_admission_decisions(
                        proposal_key, action, decision_id, authority_ledger_seq,
                        reason_code, authority_receipt_digest, decision_json,
                        decision_digest, decided_at
                    ) VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        request.proposal_key,
                        decision.action.value,
                        decision.decision_id,
                        decision.authority_ledger_seq,
                        decision.reason_code,
                        decision.authority_receipt_digest,
                        encoded,
                        retained_digest,
                        now,
                    ),
                )
            self._connection.execute(
                """
                UPDATE unpublished_graphiti_admission_queue
                SET state=?, claim_owner=NULL, claim_until=NULL,
                    attempt_count=0, last_error=NULL, updated_at=?
                WHERE proposal_key=?
                """,
                (state, now, request.proposal_key),
            )

    def _retain_projection(
        self,
        request: GraphitiAdmissionRequest,
        decision: GraphitiGovernedDecision,
        receipt: GraphitiProjectionReceipt,
    ) -> None:
        if (
            receipt.proposal_key != request.proposal_key
            or receipt.decision_id != decision.decision_id
            or receipt.authority_watermark != decision.authority_ledger_seq
            or receipt.generation_id != self._projection_generation_id
        ):
            raise GraphitiAdmissionConsumerError(
                "governed projection receipt differs from admission authority"
            )
        value = receipt.canonical_value()
        encoded = canonical_json_bytes(value).decode("utf-8")
        now = self._time_text(self._now())
        with _transaction(self._connection):
            existing = self._connection.execute(
                "SELECT receipt_json FROM unpublished_graphiti_projection_receipts "
                "WHERE proposal_key=?",
                (request.proposal_key,),
            ).fetchone()
            if existing is not None and str(existing[0]) != encoded:
                raise GraphitiAdmissionConsumerError(
                    "Graphiti admitted projection effect identity changed"
                )
            if existing is None:
                self._connection.execute(
                    """
                    INSERT INTO unpublished_graphiti_projection_receipts(
                        proposal_key, effect_id, authority_watermark,
                        projector_family_id, generation_id, schema_version,
                        trust_scope, receipt_json, receipt_digest, projected_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        request.proposal_key,
                        receipt.effect_id,
                        receipt.authority_watermark,
                        receipt.projector_family_id,
                        receipt.generation_id,
                        receipt.schema_version,
                        receipt.trust_scope,
                        encoded,
                        receipt.receipt_digest,
                        now,
                    ),
                )
            self._connection.execute(
                """
                UPDATE unpublished_graphiti_admission_queue
                SET state='PROJECTED', claim_owner=NULL, claim_until=NULL,
                    attempt_count=0, last_error=NULL, updated_at=?
                WHERE proposal_key=?
                """,
                (now, request.proposal_key),
            )

    def _record_work_failure(self, proposal_key: str, exc: Exception) -> bool:
        now = self._time_text(self._now())
        with _transaction(self._connection):
            row = self._connection.execute(
                "SELECT state, attempt_count FROM unpublished_graphiti_admission_queue "
                "WHERE proposal_key=?",
                (proposal_key,),
            ).fetchone()
            if row is None:
                raise GraphitiAdmissionConsumerError(
                    "failed admission work item disappeared"
                )
            state = str(row[0])
            attempts = int(row[1]) + 1
            dead = attempts >= self._max_attempts
            retry_state = "DECIDED" if state == "DECIDED" else "READY"
            self._connection.execute(
                """
                UPDATE unpublished_graphiti_admission_queue
                SET state=?, attempt_count=?, last_error=?, claim_owner=NULL,
                    claim_until=NULL, updated_at=?
                WHERE proposal_key=?
                """,
                (
                    "DEAD_LETTER" if dead else retry_state,
                    attempts,
                    f"{type(exc).__name__}: {exc}"[:4096],
                    now,
                    proposal_key,
                ),
            )
        return dead

    def reconcile_rights(
        self,
        *,
        limit: int = 100,
        ingest_ids: tuple[str, ...] | None = None,
    ) -> int:
        """Reconcile legacy v1 per-proposal effects only.

        Generation-bound v2 receipts are replaced only by another complete
        Increment 4 snapshot; they must never enter this historical tombstone
        path.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("rights reconciliation limit must be positive")
        exact = _exact_ingest_ids(ingest_ids)
        statement = """
            SELECT queue.proposal_key, queue.request_json, decision.decision_json
            FROM unpublished_graphiti_admission_queue AS queue
            JOIN unpublished_graphiti_admission_decisions AS decision
              USING(proposal_key)
            JOIN unpublished_graphiti_projection_receipts AS projection
              USING(proposal_key)
            LEFT JOIN unpublished_graphiti_projection_tombstones AS tombstone
              USING(proposal_key)
            WHERE decision.action='ADMIT' AND tombstone.proposal_key IS NULL
              AND queue.state='PROJECTED'
              AND projection.schema_version=
                  'newsroom.increment4.admitted-projection.v1'
            """
        parameters: tuple[object, ...] = ()
        if exact is not None:
            statement += " AND queue.ingest_id IN (" + ",".join(
                "?" for _ in exact
            ) + ")"
            parameters = exact
        statement += " ORDER BY queue.queue_seq"
        rows = self._connection.execute(statement, parameters).fetchall()
        revoked = 0
        for proposal_key, request_json, decision_json in rows:
            if revoked >= limit:
                break
            try:
                request = graphiti_admission_request_from_value(
                    _mapping(
                        json.loads(str(request_json)),
                        field="queued admission request",
                    )
                )
                if self._rights.is_current(request):
                    continue
                decision = graphiti_governed_decision_from_json(str(decision_json))
                self._tombstone_projection(
                    proposal_key=str(proposal_key),
                    request=request,
                    decision=decision,
                )
            except Exception as exc:
                self._record_rights_reconciliation_failure(
                    str(proposal_key), exc
                )
                continue
            revoked += 1
        return revoked

    def _record_rights_reconciliation_failure(
        self, proposal_key: str, exc: Exception
    ) -> bool:
        now = self._time_text(self._now())
        with _transaction(self._connection):
            self._connection.execute(
                """
                UPDATE unpublished_graphiti_admission_queue
                SET attempt_count=attempt_count+1,
                    state=CASE WHEN attempt_count+1>=? THEN 'DEAD_LETTER'
                               ELSE state END,
                    last_error=?, updated_at=?
                WHERE proposal_key=? AND state='PROJECTED'
                """,
                (
                    self._max_attempts,
                    f"RIGHTS_RECONCILIATION: {type(exc).__name__}: {exc}"[:4096],
                    now,
                    proposal_key,
                ),
            )
            state = self._connection.execute(
                "SELECT state FROM unpublished_graphiti_admission_queue "
                "WHERE proposal_key=?",
                (proposal_key,),
            ).fetchone()
        return state is not None and str(state[0]) == "DEAD_LETTER"

    def _tombstone_projection(
        self,
        *,
        proposal_key: str,
        request: GraphitiAdmissionRequest,
        decision: GraphitiGovernedDecision,
    ) -> None:
        delivery = GraphitiProjectionRequest(request, decision)
        receipt = self._projector.tombstone_increment4_admitted(
            delivery,
            idempotency_key=f"graphiti-tombstone:{proposal_key}:{decision.decision_id}",
        )
        if (
            receipt.proposal_key != proposal_key
            or receipt.decision_id != decision.decision_id
            or receipt.generation_id != self._projection_generation_id
        ):
            raise GraphitiAdmissionConsumerError(
                "governed tombstone receipt names another proposal"
            )
        original = self._connection.execute(
            "SELECT generation_id, authority_watermark "
            "FROM unpublished_graphiti_projection_receipts WHERE proposal_key=?",
            (proposal_key,),
        ).fetchone()
        if (
            original is None
            or str(original[0]) != receipt.generation_id
            or receipt.authority_watermark < int(original[1])
        ):
            raise GraphitiAdmissionConsumerError(
                "governed tombstone watermark precedes admission authority"
            )
        encoded = canonical_json_bytes(receipt.canonical_value()).decode("utf-8")
        now = self._time_text(self._now())
        with _transaction(self._connection):
            existing = self._connection.execute(
                "SELECT receipt_json "
                "FROM unpublished_graphiti_projection_tombstones "
                "WHERE proposal_key=?",
                (proposal_key,),
            ).fetchone()
            if existing is not None and str(existing[0]) != encoded:
                raise GraphitiAdmissionConsumerError(
                    "Graphiti projection tombstone identity changed"
                )
            if existing is None:
                self._connection.execute(
                    """
                    INSERT INTO unpublished_graphiti_projection_tombstones(
                        proposal_key, effect_id, authority_watermark,
                        receipt_json, receipt_digest, tombstoned_at
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        proposal_key,
                        receipt.effect_id,
                        receipt.authority_watermark,
                        encoded,
                        receipt.receipt_digest,
                        now,
                    ),
                )
            self._connection.execute(
                """
                UPDATE unpublished_graphiti_admission_queue
                SET state='REVOKED', attempt_count=0, last_error=NULL, updated_at=?
                WHERE proposal_key=?
                """,
                (now, proposal_key),
            )

    def reconcile_projection(
        self, *, generation_id: str
    ) -> GraphitiProjectionReconciliationReceipt:
        if not generation_id:
            raise ValueError("projection reconciliation generation is required")
        if generation_id != self._projection_generation_id:
            raise GraphitiAdmissionConsumerError(
                "projection reconciliation generation differs from consumer authority"
            )
        rights_failures = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM unpublished_graphiti_admission_queue "
                "WHERE last_error LIKE 'RIGHTS_RECONCILIATION:%'"
            ).fetchone()[0]
        )
        if rights_failures:
            raise GraphitiAdmissionConsumerError(
                "projection reconciliation is blocked by rights tombstone failures"
            )
        rows = self._connection.execute(
            """
            SELECT queue.proposal_key, queue.request_json, projection.receipt_json
            FROM unpublished_graphiti_admission_decisions AS decision
            JOIN unpublished_graphiti_admission_queue AS queue USING(proposal_key)
            LEFT JOIN unpublished_graphiti_projection_receipts AS projection
              USING(proposal_key)
            LEFT JOIN unpublished_graphiti_projection_tombstones AS tombstone
              USING(proposal_key)
            WHERE decision.action='ADMIT' AND queue.state!='REVOKED'
              AND tombstone.proposal_key IS NULL
            ORDER BY queue.queue_seq
            """,
        ).fetchall()
        expected_items: list[GraphitiProjectionReceipt] = []
        missing: list[str] = []
        for proposal_key, request_json, projection_json in rows:
            graphiti_admission_request_from_value(
                _mapping(
                    json.loads(str(request_json)),
                    field="queued admission request",
                )
            )
            if projection_json is None:
                missing.append(str(proposal_key))
                continue
            projection = graphiti_projection_receipt_from_json(str(projection_json))
            if projection.generation_id != generation_id:
                raise GraphitiAdmissionConsumerError(
                    "admitted projection receipt belongs to another generation"
                )
            expected_items.append(projection)
        if missing:
            raise GraphitiAdmissionConsumerError(
                "admitted projection has missing governed effects: "
                + ",".join(missing[:10])
            )
        expected = tuple(sorted(expected_items, key=lambda item: item.effect_id))
        if expected:
            required_watermark = max(
                item.authority_watermark for item in expected
            )
        else:
            tombstone_watermark = self._connection.execute(
                """
                SELECT MAX(tombstone.authority_watermark)
                FROM unpublished_graphiti_projection_tombstones AS tombstone
                JOIN unpublished_graphiti_projection_receipts AS projection
                  USING(proposal_key)
                WHERE projection.generation_id=?
                """,
                (generation_id,),
            ).fetchone()[0]
            if tombstone_watermark is None:
                raise GraphitiAdmissionConsumerError(
                    "projection reconciliation has no governed generation evidence"
                )
            required_watermark = int(tombstone_watermark)
        receipt = self._projector.reconcile_increment4_admitted(
            expected,
            generation_id=generation_id,
        )
        expected_ids = tuple(sorted(item.effect_id for item in expected))
        if (
            receipt.generation_id != generation_id
            or receipt.expected_effect_ids != expected_ids
            or receipt.authority_watermark != required_watermark
        ):
            raise GraphitiAdmissionConsumerError(
                "projection reconciliation receipt differs from governed authority"
            )
        encoded = canonical_json_bytes(receipt.canonical_value()).decode("utf-8")
        now = self._time_text(self._now())
        with _transaction(self._connection):
            existing = self._connection.execute(
                "SELECT receipt_json "
                "FROM unpublished_graphiti_projection_reconciliations "
                "WHERE receipt_digest=?",
                (receipt.receipt_digest,),
            ).fetchone()
            if existing is not None and str(existing[0]) != encoded:
                raise GraphitiAdmissionConsumerError(
                    "projection reconciliation digest was reused"
                )
            if existing is None:
                self._connection.execute(
                    """
                    INSERT INTO unpublished_graphiti_projection_reconciliations(
                        receipt_digest, projector_family_id, generation_id,
                        authority_watermark, receipt_json, reconciled_at
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        receipt.receipt_digest,
                        receipt.projector_family_id,
                        receipt.generation_id,
                        receipt.authority_watermark,
                        encoded,
                        now,
                    ),
                )
        return receipt

    def telemetry(self) -> GraphitiAdmissionTelemetry:
        return graphiti_admission_telemetry(
            self._connection,
            now=self._now(),
        )


def graphiti_admission_telemetry(
    connection: sqlite3.Connection,
    *,
    now: datetime | None = None,
) -> GraphitiAdmissionTelemetry:
    queued_denominator = int(
        connection.execute(
            "SELECT COUNT(*) FROM unpublished_graphiti_admission_queue"
        ).fetchone()[0]
    )
    integrity_hold_denominator = int(
        connection.execute(
            """
            SELECT COALESCE(SUM(ingest.proposal_count), 0)
            FROM unpublished_graphiti_admission_receipt_failures AS failure
            JOIN unpublished_graphiti_ingest AS ingest USING(ingest_id)
            """
        ).fetchone()[0]
    )
    denominator = queued_denominator + integrity_hold_denominator
    action_counts = {
        str(row[0]): int(row[1])
        for row in connection.execute(
            "SELECT action, COUNT(*) FROM unpublished_graphiti_admission_decisions "
            "GROUP BY action"
        )
    }
    decision_count = sum(action_counts.values())
    dead_letters = int(
        connection.execute(
            "SELECT COUNT(*) FROM unpublished_graphiti_admission_queue "
            "WHERE state='DEAD_LETTER'"
        ).fetchone()[0]
    )
    revoked = int(
        connection.execute(
            "SELECT COUNT(*) FROM unpublished_graphiti_admission_queue "
            "WHERE state='REVOKED'"
        ).fetchone()[0]
    )
    projected = int(
        connection.execute(
            "SELECT COUNT(*) FROM unpublished_graphiti_projection_receipts"
        ).fetchone()[0]
    )
    active_admitted = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM unpublished_graphiti_admission_decisions AS decision
            JOIN unpublished_graphiti_admission_queue AS queue USING(proposal_key)
            WHERE decision.action='ADMIT' AND queue.state!='REVOKED'
            """
        ).fetchone()[0]
    )
    active_effect_ids = tuple(
        str(row[0])
        for row in connection.execute(
            """
            SELECT projection.effect_id
            FROM unpublished_graphiti_projection_receipts AS projection
            LEFT JOIN unpublished_graphiti_projection_tombstones AS tombstone
              USING(proposal_key)
            WHERE tombstone.proposal_key IS NULL
            ORDER BY projection.effect_id
            """
        )
    )
    reconciliation_row = connection.execute(
        "SELECT generation_id,receipt_json FROM unpublished_graphiti_projection_reconciliations "
        "ORDER BY reconciled_at DESC,receipt_digest DESC LIMIT 1"
    ).fetchone()
    reconciled = False
    rights_tombstone_failures = int(
        connection.execute(
            "SELECT COUNT(*) FROM unpublished_graphiti_admission_queue "
            "WHERE last_error LIKE 'RIGHTS_RECONCILIATION:%'"
        ).fetchone()[0]
    )
    if reconciliation_row is not None:
        try:
            generation_id = str(reconciliation_row[0])
            reconciliation_rows = connection.execute(
                "SELECT receipt_json FROM "
                "unpublished_graphiti_projection_reconciliations "
                "WHERE generation_id=? ORDER BY reconciled_at,receipt_digest",
                (generation_id,),
            ).fetchall()
            parsed = tuple(
                graphiti_projection_reconciliation_from_json(str(row[0]))
                for row in reconciliation_rows
            )
        except (GraphitiAdmissionConsumerError, json.JSONDecodeError, ValueError):
            pass
        else:
            generation_effect_ids = tuple(
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT projection.effect_id
                    FROM unpublished_graphiti_projection_receipts AS projection
                    LEFT JOIN unpublished_graphiti_projection_tombstones AS tombstone
                      USING(proposal_key)
                    WHERE projection.generation_id=?
                      AND tombstone.proposal_key IS NULL
                    ORDER BY projection.effect_id
                    """,
                    (generation_id,),
                )
            )
            bindings = tuple(binding for _receipt, binding in parsed)
            if len(parsed) == 1 and bindings == (None,):
                reconciled_effect_ids = parsed[0][0].actual_effect_ids
            elif all(binding is not None for binding in bindings):
                cohort_digests = tuple(
                    str(binding["cohort_digest"])
                    for binding in bindings
                    if binding is not None
                )
                effects = tuple(
                    effect_id
                    for receipt, _binding in parsed
                    for effect_id in receipt.actual_effect_ids
                )
                reconciled_effect_ids = (
                    tuple(sorted(effects))
                    if len(cohort_digests) == len(set(cohort_digests))
                    and len(effects) == len(set(effects))
                    else ()
                )
            else:
                reconciled_effect_ids = ()
            reconciled = (
                rights_tombstone_failures == 0
                and reconciled_effect_ids == generation_effect_ids
                and active_admitted == len(active_effect_ids)
            )
    integrity_holds = int(
        connection.execute(
            "SELECT COUNT(*) FROM unpublished_graphiti_admission_receipt_failures"
        ).fetchone()[0]
    )
    failure_order_row = connection.execute(
        """
        SELECT ingest.at, failure.ingest_id
        FROM unpublished_graphiti_admission_receipt_failures AS failure
        JOIN unpublished_graphiti_ingest AS ingest USING(ingest_id)
        ORDER BY ingest.at, failure.ingest_id
        LIMIT 1
        """
    ).fetchone()
    failure_order = (
        None
        if failure_order_row is None
        else (str(failure_order_row[0]), str(failure_order_row[1]))
    )
    watermark: int | None = None
    for (
        queue_seq,
        ingest_at,
        ingest_id,
        state,
        action,
        decision_watermark,
        projection_watermark,
    ) in connection.execute(
        """
        SELECT queue.queue_seq, ingest.at, queue.ingest_id, queue.state,
               decision.action, decision.authority_ledger_seq,
               projection.authority_watermark
        FROM unpublished_graphiti_admission_queue AS queue
        JOIN unpublished_graphiti_ingest AS ingest USING(ingest_id)
        LEFT JOIN unpublished_graphiti_admission_decisions AS decision
          USING(proposal_key)
        LEFT JOIN unpublished_graphiti_projection_receipts AS projection
          USING(proposal_key)
        ORDER BY queue.queue_seq
        """
    ):
        del queue_seq
        if failure_order is not None and failure_order <= (
            str(ingest_at),
            str(ingest_id),
        ):
            break
        candidate = (
            projection_watermark
            if action == "ADMIT" and projection_watermark is not None
            else decision_watermark
        )
        terminal = state == "REVOKED" or action in {"REJECT", "HOLD"} or (
            action == "ADMIT" and projection_watermark is not None
        )
        if not terminal or candidate is None:
            break
        candidate_value = int(candidate)
        if watermark is not None and candidate_value < watermark:
            break
        watermark = candidate_value
    current = (now or datetime.now(tz=UTC)).astimezone(UTC)
    lag_rows = connection.execute(
        """
        SELECT created_at FROM unpublished_graphiti_admission_queue AS queue
        LEFT JOIN unpublished_graphiti_admission_decisions AS decision
          USING(proposal_key)
        WHERE queue.state NOT IN ('TERMINAL','PROJECTED','REVOKED')
        UNION ALL
        SELECT first_seen_at FROM unpublished_graphiti_admission_receipt_failures
        """
    ).fetchall()
    lags = []
    for row in lag_rows:
        try:
            instant = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
        except ValueError:
            continue
        lags.append(max(int((current - instant.astimezone(UTC)).total_seconds()), 0))
    return GraphitiAdmissionTelemetry(
        proposal_denominator=denominator,
        admitted_count=action_counts.get("ADMIT", 0),
        rejected_count=action_counts.get("REJECT", 0),
        held_count=action_counts.get("HOLD", 0),
        dead_letter_count=dead_letters,
        revoked_count=revoked,
        projected_count=projected,
        admission_backlog=max(denominator - decision_count, 0),
        integrity_hold_receipt_count=integrity_holds,
        oldest_lag_seconds=max(lags, default=0),
        contiguous_projection_watermark=watermark,
        projection_gap_count=(
            max(active_admitted - len(active_effect_ids), 0)
            + rights_tombstone_failures
        ),
        projection_reconciled=reconciled,
    )


__all__ = [
    "GovernedGraphitiAdmissionAuthority",
    "GovernedGraphitiProjector",
    "GraphitiAdmissionConsumer",
    "GraphitiAdmissionConsumerError",
    "GraphitiAdmissionDrainReport",
    "GraphitiAdmissionRequest",
    "GraphitiAdmissionTelemetry",
    "GraphitiGovernedDecision",
    "GraphitiProjectionGenerationResult",
    "GraphitiProjectionReceipt",
    "GraphitiProjectionReconciliationReceipt",
    "GraphitiProjectionRequest",
    "GraphitiProposalAuthority",
    "GraphitiProposalAuthorityBinding",
    "GraphitiProposalAdmissionAction",
    "GraphitiRelationHoldBasis",
    "GraphitiRightsAuthority",
    "GRAPHITI_ADMISSION_EXTENSION_RECONCILIATION_SCHEMA_VERSION",
    "GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION",
    "GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION",
    "GRAPHITI_PROJECTION_GENERATION_SCHEMA_VERSION",
    "graphiti_admission_request_from_value",
    "graphiti_admission_telemetry",
    "graphiti_decided_cohort_generation_identity",
    "graphiti_governed_decision_from_json",
    "graphiti_projection_receipt_from_json",
    "graphiti_projection_receipt_matches_binding",
    "graphiti_projection_reconciliation_from_json",
]
