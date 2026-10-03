"""Disposable private-serving delivery with authoritative attempt evidence."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from uuid import UUID

from newsroom.authority import (
    AggregateId,
    AuthenticationProof,
    AuthorityCommands,
    AuthorityEvents,
    CommandId,
    EventId,
    GovernedObjects,
    HydrationRequest,
    ObjectAdmissionId,
    ObjectAdmissionPayload,
    ObjectAdmissionRequest,
    SemanticCommand,
    UtcTimestamp,
)
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    validate_sha256_digest,
)
from newsroom.authority.types import TrustScope
from newsroom.increment6.candidates import StoryCandidateReadPort

from .editorial import StoryVersionReceipt
from .publication import (
    SURFACE_PURPOSE,
    OfflinePublication,
    PublicationReceipt,
)

ATTEMPT_SCHEMA = "newsroom.increment10.private-serving-attempt.v1"
EVIDENCE_SCHEMA = "newsroom.increment10.private-serving-evidence.v1"
ATTEMPT_ADMISSION_TYPE = "private-serving.attempt"
ATTEMPT_CLASS = "private_serving_attempt"
ATTEMPT_USE = "private_serving_dispatch"
ATTEMPT_PURPOSE = "private-serving.attempt"
ATTEMPT_COMMAND = "private-serving.attempt.begin"
ATTEMPT_EVENT = "private-serving.attempt.begun"
EVIDENCE_ADMISSION_TYPE = "private-serving.evidence"
EVIDENCE_CLASS = "private_serving_evidence"
EVIDENCE_USE = "private_serving_observation"
EVIDENCE_PURPOSE = "private-serving.evidence"
EVIDENCE_COMMAND = "private-serving.evidence.record"
EVIDENCE_EVENT = "private-serving.evidence.recorded"
SERVING_SECURITY_SCOPE = "authority.internal"
SERVING_RETENTION_SCOPE = "publication.audit"

_KINDS = ("ARTICLE", "FEED_CARD")
_TOKEN = object()
_READ_PROOF_TOKEN = object()


class PrivateServingError(ValueError):
    """Raised when private-serving authority or projection bindings differ."""


@dataclass(frozen=True, slots=True)
class ServingAttempt:
    attempt_id: str
    operation_id: str
    operation_key: str
    target_id: str
    bundle_id: str
    surface_kind: str
    surface_payload_id: str
    surface_admission_id: ObjectAdmissionId
    surface_digest: str
    adapter_principal_id: str
    target_context_digest: str
    projection_identity_digest: str

    def __post_init__(self) -> None:
        _texts(
            self.operation_id,
            self.operation_key,
            self.target_id,
            self.bundle_id,
            self.surface_kind,
            self.surface_payload_id,
            self.adapter_principal_id,
        )
        if type(self.surface_admission_id) is not ObjectAdmissionId:
            raise PrivateServingError("serving attempt admission differs")
        _digests(
            self.operation_id,
            self.bundle_id,
            self.surface_payload_id,
            self.surface_digest,
            self.target_context_digest,
            self.projection_identity_digest,
        )
        if self.surface_kind not in _KINDS:
            raise PrivateServingError("serving attempt surface differs")
        if self.attempt_id != digest_bytes(
            canonical_json_bytes(self.value(include_identity=False))
        ):
            raise PrivateServingError("serving attempt identity differs")

    def value(self, *, include_identity: bool = True) -> dict[str, object]:
        value = {
            "operation_id": self.operation_id,
            "operation_key": self.operation_key,
            "target_id": self.target_id,
            "bundle_id": self.bundle_id,
            "surface_kind": self.surface_kind,
            "surface_payload_id": self.surface_payload_id,
            "surface_admission_id": str(self.surface_admission_id),
            "surface_digest": self.surface_digest,
            "adapter_principal_id": self.adapter_principal_id,
            "target_context_digest": self.target_context_digest,
            "projection_identity_digest": self.projection_identity_digest,
        }
        if include_identity:
            value["attempt_id"] = self.attempt_id
        return value


@dataclass(frozen=True, slots=True)
class AttemptBatch:
    batch_id: str
    publication_id: AggregateId
    publication_event_id: str
    publication_transaction_id: str
    publication_transaction_digest: str
    publication_aggregate_version: int
    attempts: tuple[ServingAttempt, ...]

    def __post_init__(self) -> None:
        if type(self.publication_id) is not AggregateId:
            raise PrivateServingError("attempt publication aggregate differs")
        EventId.parse(self.publication_event_id)
        _digests(self.publication_transaction_id, self.publication_transaction_digest)
        if (
            type(self.publication_aggregate_version) is not int
            or self.publication_aggregate_version <= 0
            or type(self.attempts) is not tuple
            or tuple(item.surface_kind for item in self.attempts) != _KINDS
            or len({item.operation_key for item in self.attempts}) != 2
        ):
            raise PrivateServingError("attempt batch operations differ")
        if self.batch_id != digest_bytes(
            canonical_json_bytes(self.value(include_identity=False))
        ):
            raise PrivateServingError("attempt batch identity differs")

    @property
    def evidence_aggregate_id(self) -> AggregateId:
        return _aggregate_for(self.batch_id)

    def value(self, *, include_identity: bool = True) -> dict[str, object]:
        value = {
            "schema_identity": ATTEMPT_SCHEMA,
            "publication_id": str(self.publication_id),
            "publication_event_id": self.publication_event_id,
            "publication_transaction_id": self.publication_transaction_id,
            "publication_transaction_digest": self.publication_transaction_digest,
            "publication_aggregate_version": self.publication_aggregate_version,
            "attempts": [item.value() for item in self.attempts],
        }
        if include_identity:
            value["batch_id"] = self.batch_id
        return value

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.value())


@dataclass(frozen=True, slots=True)
class AttemptReceipt:
    command_id: str
    event_id: str
    publication_id: AggregateId
    aggregate_version: int
    admission_id: ObjectAdmissionId
    batch_id: str
    batch_digest: str

    def __post_init__(self) -> None:
        CommandId.parse(self.command_id)
        EventId.parse(self.event_id)
        if type(self.publication_id) is not AggregateId:
            raise PrivateServingError("attempt receipt aggregate differs")
        if type(self.admission_id) is not ObjectAdmissionId:
            raise PrivateServingError("attempt receipt admission differs")
        _digests(self.batch_id, self.batch_digest)


@dataclass(frozen=True, slots=True)
class ProjectionRow:
    operation_key: str
    operation_id: str
    attempt_id: str
    surface_kind: str
    payload_id: str
    payload_digest: str
    payload_bytes: bytes
    applied_at: str


@dataclass(frozen=True, slots=True)
class OperationObservation:
    operation_key: str
    attempt_id: str
    result: Literal["MATCHING", "MISSING", "CONFLICT"]
    observed_payload_digest: str | None
    target_native_id: str | None

    def __post_init__(self) -> None:
        _texts(self.operation_key)
        _digests(self.attempt_id)
        if self.result not in {"MATCHING", "MISSING", "CONFLICT"}:
            raise PrivateServingError("serving observation result differs")
        if (self.result == "MISSING") != (
            self.observed_payload_digest is None and self.target_native_id is None
        ):
            raise PrivateServingError("serving observation evidence differs")
        if self.observed_payload_digest is not None:
            _digests(self.observed_payload_digest)


@dataclass(frozen=True, slots=True)
class DeliveryAcknowledgement:
    acknowledgement_id: str
    batch_id: str
    target_id: str
    target_acknowledged_at: str
    first_private_effect_at: str
    operation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _digests(self.batch_id)
        _texts(
            self.target_id,
            self.target_acknowledged_at,
            self.first_private_effect_at,
        )
        for value in (
            self.target_acknowledged_at,
            self.first_private_effect_at,
        ):
            UtcTimestamp.parse(value)
        if UtcTimestamp.parse(self.target_acknowledged_at).value < UtcTimestamp.parse(
            self.first_private_effect_at
        ).value:
            raise PrivateServingError("serving acknowledgement time differs")
        if self.acknowledgement_id != digest_bytes(
            canonical_json_bytes(self.value(include_identity=False))
        ):
            raise PrivateServingError("serving acknowledgement identity differs")

    def value(self, *, include_identity: bool = True) -> dict[str, object]:
        value = {
            "batch_id": self.batch_id,
            "target_id": self.target_id,
            "target_acknowledged_at": self.target_acknowledged_at,
            "first_private_effect_at": self.first_private_effect_at,
            "operation_ids": list(self.operation_ids),
        }
        if include_identity:
            value["acknowledgement_id"] = self.acknowledgement_id
        return value


@dataclass(frozen=True, slots=True)
class DeliveryEvidence:
    evidence_id: str
    batch_id: str
    target_id: str
    outcome: Literal["ACKNOWLEDGED", "MISSING", "AMBIGUOUS", "FAILED"]
    observed_at: str
    observation_method: str
    observer_principal_id: str
    raw_observation_digest: str
    observation_rows: tuple[ProjectionRow | None, ...]
    observations: tuple[OperationObservation, ...]
    failure_code: str | None
    acknowledgement: DeliveryAcknowledgement | None

    def __post_init__(self) -> None:
        _digests(self.batch_id, self.raw_observation_digest)
        _texts(
            self.target_id,
            self.observed_at,
            self.observation_method,
            self.observer_principal_id,
        )
        UtcTimestamp.parse(self.observed_at)
        if self.outcome not in {"ACKNOWLEDGED", "MISSING", "AMBIGUOUS", "FAILED"}:
            raise PrivateServingError("serving evidence outcome differs")
        if type(self.observations) is not tuple or any(
            type(item) is not OperationObservation for item in self.observations
        ):
            raise PrivateServingError("serving observations differ")
        if type(self.observation_rows) is not tuple or any(
            item is not None and type(item) is not ProjectionRow
            for item in self.observation_rows
        ):
            raise PrivateServingError("serving observation rows differ")
        if (self.outcome == "ACKNOWLEDGED") != (self.acknowledgement is not None):
            raise PrivateServingError("serving acknowledgement outcome differs")
        if self.acknowledgement is not None and (
            self.acknowledgement.batch_id != self.batch_id
            or self.acknowledgement.target_id != self.target_id
            or self.acknowledgement.target_acknowledged_at != self.observed_at
        ):
            raise PrivateServingError("serving acknowledgement binding differs")
        if self.outcome == "FAILED" and not self.failure_code:
            raise PrivateServingError("serving failure evidence differs")
        if self.outcome != "FAILED" and self.failure_code is not None:
            raise PrivateServingError("serving failure code differs")
        if self.evidence_id != digest_bytes(
            canonical_json_bytes(self.value(include_identity=False))
        ):
            raise PrivateServingError("serving evidence identity differs")

    def value(self, *, include_identity: bool = True) -> dict[str, object]:
        value = {
            "schema_identity": EVIDENCE_SCHEMA,
            "batch_id": self.batch_id,
            "target_id": self.target_id,
            "outcome": self.outcome,
            "observed_at": self.observed_at,
            "observation_method": self.observation_method,
            "observer_principal_id": self.observer_principal_id,
            "raw_observation_digest": self.raw_observation_digest,
            "observation_rows": [
                None if item is None else _row_value(item)
                for item in self.observation_rows
            ],
            "observations": [
                {
                    "operation_key": item.operation_key,
                    "attempt_id": item.attempt_id,
                    "result": item.result,
                    "observed_payload_digest": item.observed_payload_digest,
                    "target_native_id": item.target_native_id,
                }
                for item in self.observations
            ],
            "failure_code": self.failure_code,
            "acknowledgement": (
                None if self.acknowledgement is None else self.acknowledgement.value()
            ),
        }
        if include_identity:
            value["evidence_id"] = self.evidence_id
        return value

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.value())


@dataclass(frozen=True, slots=True)
class EvidenceReceipt:
    command_id: str
    event_id: str
    evidence_aggregate_id: AggregateId
    aggregate_version: int
    admission_id: ObjectAdmissionId
    evidence_id: str
    evidence_digest: str

    def __post_init__(self) -> None:
        CommandId.parse(self.command_id)
        EventId.parse(self.event_id)
        if type(self.evidence_aggregate_id) is not AggregateId:
            raise PrivateServingError("evidence receipt aggregate differs")
        if type(self.admission_id) is not ObjectAdmissionId:
            raise PrivateServingError("evidence receipt admission differs")
        if type(self.aggregate_version) is not int or self.aggregate_version <= 0:
            raise PrivateServingError("evidence receipt version differs")
        _digests(self.evidence_id, self.evidence_digest)


@dataclass(frozen=True, slots=True)
class AcknowledgedServing:
    acknowledgement: DeliveryAcknowledgement
    primary_feed_published_at: str
    rows: tuple[ProjectionRow, ...]

    def __post_init__(self) -> None:
        published = UtcTimestamp.parse(self.primary_feed_published_at).value
        if tuple(row.surface_kind for row in self.rows) != _KINDS:
            raise PrivateServingError("acknowledged serving rows differ")
        if published < max(
            UtcTimestamp.parse(self.acknowledgement.target_acknowledged_at).value,
            UtcTimestamp.parse(self.acknowledgement.first_private_effect_at).value,
        ):
            raise PrivateServingError("primary feed acknowledgement time differs")


@dataclass(frozen=True, slots=True, init=False)
class PrivateServingReadProof:
    """Opaque proof that retained authority acknowledged exact private rows."""

    evidence_receipt: EvidenceReceipt
    attempt_receipt: AttemptReceipt
    serving: AcknowledgedServing

    def __init__(
        self,
        token: object,
        evidence_receipt: EvidenceReceipt,
        attempt_receipt: AttemptReceipt,
        serving: AcknowledgedServing,
    ) -> None:
        if token is not _READ_PROOF_TOKEN or not all(
            type(value) is expected
            for value, expected in (
                (evidence_receipt, EvidenceReceipt),
                (attempt_receipt, AttemptReceipt),
                (serving, AcknowledgedServing),
            )
        ):
            raise PrivateServingError("private-serving read proof is forged")
        if (
            evidence_receipt.evidence_aggregate_id
            != _aggregate_for(attempt_receipt.batch_id)
            or serving.acknowledgement.batch_id != attempt_receipt.batch_id
            or tuple(row.operation_id for row in serving.rows)
            != serving.acknowledgement.operation_ids
            or any(
                digest_bytes(row.payload_bytes) != row.payload_digest
                for row in serving.rows
            )
        ):
            raise PrivateServingError("private-serving read proof differs")
        object.__setattr__(self, "evidence_receipt", evidence_receipt)
        object.__setattr__(self, "attempt_receipt", attempt_receipt)
        object.__setattr__(self, "serving", serving)


class PrivateServingReadPort:
    """ACK-only consumer over an existing query-only private projection."""

    __slots__ = ("_connection", "_proof")

    def __init__(
        self,
        token: object,
        *,
        connection: sqlite3.Connection,
        proof: PrivateServingReadProof | None,
    ) -> None:
        if (
            token is not _TOKEN
            or type(connection) is not sqlite3.Connection
            or (proof is not None and type(proof) is not PrivateServingReadProof)
        ):
            raise PrivateServingError("private-serving reader construction differs")
        self._connection = connection
        self._proof = proof

    def close(self) -> None:
        self._connection.close()

    def acknowledged_rows(self) -> AcknowledgedServing | None:
        """Return only rows covered by the retained authoritative ACK proof."""

        if self._proof is None:
            return None
        expected = self._proof.serving
        rows = tuple(
            self._query(row.operation_key)
            for row in expected.rows
        )
        actual_rows = tuple(row for row in rows if row is not None)
        if actual_rows != expected.rows:
            raise PrivateServingError("acknowledged private payload differs")
        return AcknowledgedServing(
            expected.acknowledgement,
            expected.primary_feed_published_at,
            actual_rows,
        )

    def _query(self, operation_key: str) -> ProjectionRow | None:
        row = self._connection.execute(
            "SELECT operation_key,operation_id,attempt_id,surface_kind,payload_id,"
            "payload_digest,payload_bytes,applied_at FROM private_serving_payloads "
            "WHERE operation_key=?",
            (operation_key,),
        ).fetchone()
        return None if row is None else ProjectionRow(*tuple(row))


class PrivateServingDelivery:
    """Trusted server-only controller for a local private projection."""

    def __init__(
        self,
        token: object,
        *,
        connection: sqlite3.Connection,
        objects: GovernedObjects,
        commands: AuthorityCommands,
        events: AuthorityEvents,
        publication: OfflinePublication,
        adapter_principal_id: str,
        authority_domain: str,
        target_id: str,
        target_context_digest: str,
        projection_identity_digest: str,
        attempt_hydration_policy_digest: str,
        evidence_hydration_policy_digest: str,
        attempt_admission_definition_digest: str,
        evidence_admission_definition_digest: str,
        attempt_command_definition_digest: str,
        evidence_command_definition_digest: str,
    ) -> None:
        if token is not _TOKEN:
            raise PrivateServingError("private-serving construction requires factory")
        if not all(
            type(value) is expected
            for value, expected in (
                (connection, sqlite3.Connection),
                (objects, GovernedObjects),
                (commands, AuthorityCommands),
                (events, AuthorityEvents),
                (publication, OfflinePublication),
            )
        ):
            raise PrivateServingError("exact serving authorities are required")
        _texts(adapter_principal_id, authority_domain, target_id)
        _digests(
            target_context_digest,
            projection_identity_digest,
            attempt_hydration_policy_digest,
            evidence_hydration_policy_digest,
            attempt_admission_definition_digest,
            evidence_admission_definition_digest,
            attempt_command_definition_digest,
            evidence_command_definition_digest,
        )
        self._connection = connection
        self._objects = objects
        self._commands = commands
        self._events = events
        self._publication = publication
        self._adapter = adapter_principal_id
        self._domain = authority_domain
        self._target = target_id
        self._target_context = target_context_digest
        self._projection_identity = projection_identity_digest
        self._attempt_policy = attempt_hydration_policy_digest
        self._evidence_policy = evidence_hydration_policy_digest
        self._attempt_definition = attempt_admission_definition_digest
        self._evidence_definition = evidence_admission_definition_digest
        self._attempt_command = attempt_command_definition_digest
        self._evidence_command = evidence_command_definition_digest

    def close(self) -> None:
        self._connection.close()

    def begin(
        self,
        publication_receipt: PublicationReceipt,
        *,
        story_receipt: StoryVersionReceipt,
        candidate_port: StoryCandidateReadPort,
        proof: AuthenticationProof,
    ) -> tuple[AttemptReceipt, AttemptBatch]:
        transaction = self._publication.read(
            publication_receipt,
            story_receipt=story_receipt,
            candidate_port=candidate_port,
            proof=proof,
        )
        batch = self._build_batch(publication_receipt, transaction)
        raw = batch.canonical_bytes()
        admission = self._objects.admit(
            ObjectAdmissionRequest(
                ATTEMPT_ADMISSION_TYPE, f"private-serving:{batch.batch_id}"
            ),
            raw,
            proof=proof,
        ).admission
        if (
            admission.definition_digest != self._attempt_definition
            or admission.blob.blob_digest != digest_bytes(raw)
        ):
            raise PrivateServingError("serving attempt admission differs")
        try:
            committed = self._commands.execute(
                SemanticCommand(
                    ATTEMPT_COMMAND,
                    publication_receipt.publication_id,
                    publication_receipt.aggregate_version,
                    ObjectAdmissionPayload(admission.admission_id),
                    f"private-serving:{batch.batch_id}",
                ),
                proof=proof,
            )
        except Exception as exc:
            raise PrivateServingError("serving attempt fence rejected") from exc
        receipt = AttemptReceipt(
            committed.command_id,
            committed.event_id,
            publication_receipt.publication_id,
            committed.aggregate_version,
            admission.admission_id,
            batch.batch_id,
            digest_bytes(raw),
        )
        self._read_attempt(
            receipt,
            publication_receipt=publication_receipt,
            story_receipt=story_receipt,
            candidate_port=candidate_port,
            proof=proof,
        )
        return receipt, batch

    def apply(
        self,
        receipt: AttemptReceipt,
        *,
        publication_receipt: PublicationReceipt,
        story_receipt: StoryVersionReceipt,
        candidate_port: StoryCandidateReadPort,
        applied_at: str,
        proof: AuthenticationProof,
        before_new_effect: Callable[[], None] | None = None,
    ) -> tuple[ProjectionRow, ...]:
        if before_new_effect is not None and not callable(before_new_effect):
            raise TypeError("private serving source fence must be callable")
        UtcTimestamp.parse(applied_at)
        batch = self._read_attempt(
            receipt,
            publication_receipt=publication_receipt,
            story_receipt=story_receipt,
            candidate_port=candidate_port,
            proof=proof,
        )
        rows = tuple(
            self._projection_row(item, applied_at=applied_at, proof=proof)
            for item in batch.attempts
        )
        retained_rows: list[ProjectionRow] = []
        checked_new_effect = False
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            for row in rows:
                existing = self._query(row.operation_key)
                if existing is None:
                    if not checked_new_effect and before_new_effect is not None:
                        before_new_effect()
                        checked_new_effect = True
                    self._connection.execute(
                        "INSERT INTO private_serving_payloads VALUES(?,?,?,?,?,?,?,?)",
                        (
                            row.operation_key,
                            row.operation_id,
                            row.attempt_id,
                            row.surface_kind,
                            row.payload_id,
                            row.payload_digest,
                            row.payload_bytes,
                            row.applied_at,
                        ),
                    )
                    retained_rows.append(row)
                elif not _same_effect(existing, row):
                    raise PrivateServingError("private projection operation conflicts")
                else:
                    retained_rows.append(existing)
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        return tuple(retained_rows)

    def query(self, operation_key: str) -> ProjectionRow | None:
        _texts(operation_key)
        return self._query(operation_key)

    def observe(
        self,
        receipt: AttemptReceipt,
        *,
        publication_receipt: PublicationReceipt,
        story_receipt: StoryVersionReceipt,
        candidate_port: StoryCandidateReadPort,
        observed_at: str,
        proof: AuthenticationProof,
    ) -> DeliveryEvidence:
        batch = self._read_attempt(
            receipt,
            publication_receipt=publication_receipt,
            story_receipt=story_receipt,
            candidate_port=candidate_port,
            proof=proof,
        )
        return self._observe_batch(batch, observed_at=observed_at, proof=proof)

    def failed(
        self,
        receipt: AttemptReceipt,
        *,
        publication_receipt: PublicationReceipt,
        story_receipt: StoryVersionReceipt,
        candidate_port: StoryCandidateReadPort,
        observed_at: str,
        failure_code: str,
        raw_response_digest: str,
        proof: AuthenticationProof,
    ) -> DeliveryEvidence:
        UtcTimestamp.parse(observed_at)
        _texts(failure_code)
        _digests(raw_response_digest)
        batch = self._read_attempt(
            receipt,
            publication_receipt=publication_receipt,
            story_receipt=story_receipt,
            candidate_port=candidate_port,
            proof=proof,
        )
        return self._evidence(
            batch,
            outcome="FAILED",
            observed_at=observed_at,
            observations=(),
            observation_rows=(),
            raw=raw_response_digest,
            failure_code=failure_code,
            acknowledgement=None,
        )

    def record(
        self,
        evidence: DeliveryEvidence,
        receipt: AttemptReceipt,
        *,
        expected_version: int,
        proof: AuthenticationProof,
    ) -> EvidenceReceipt:
        batch = self._read_attempt_object(receipt, proof=proof)
        self._validate_evidence(evidence, batch, proof=proof, current=True)
        raw = evidence.canonical_bytes()
        first_ack = evidence.outcome == "ACKNOWLEDGED"
        stable_key = (
            f"private-serving-ack:{batch.batch_id}"
            if first_ack
            else f"private-serving-evidence:{evidence.evidence_id}"
        )
        admission = self._objects.admit(
            ObjectAdmissionRequest(
                EVIDENCE_ADMISSION_TYPE,
                stable_key,
            ),
            raw,
            proof=proof,
        ).admission
        if admission.definition_digest != self._evidence_definition:
            raise PrivateServingError("serving evidence admission differs")
        if admission.blob.blob_digest != digest_bytes(raw):
            if not first_ack:
                raise PrivateServingError("serving evidence admission differs")
            material = self._objects.hydrate(
                HydrationRequest(admission.admission_id, EVIDENCE_PURPOSE),
                proof=proof,
            )
            self._access(
                material.decision,
                self._evidence_policy,
                EVIDENCE_CLASS,
                EVIDENCE_USE,
            )
            evidence = _evidence_from_bytes(material.data)
            if evidence.outcome != "ACKNOWLEDGED" or evidence.batch_id != batch.batch_id:
                raise PrivateServingError("first serving acknowledgement differs")
            self._validate_evidence(evidence, batch, proof=proof, current=False)
            raw = material.data
        try:
            committed = self._commands.execute(
                SemanticCommand(
                    EVIDENCE_COMMAND,
                    batch.evidence_aggregate_id,
                    expected_version,
                    ObjectAdmissionPayload(admission.admission_id),
                    stable_key,
                ),
                proof=proof,
            )
        except Exception as exc:
            raise PrivateServingError("serving evidence command rejected") from exc
        result = EvidenceReceipt(
            committed.command_id,
            committed.event_id,
            batch.evidence_aggregate_id,
            committed.aggregate_version,
            admission.admission_id,
            evidence.evidence_id,
            digest_bytes(raw),
        )
        self.read_evidence(result, receipt, proof=proof)
        return result

    def read_evidence(
        self,
        evidence_receipt: EvidenceReceipt,
        attempt_receipt: AttemptReceipt,
        *,
        proof: AuthenticationProof,
    ) -> DeliveryEvidence:
        if type(evidence_receipt) is not EvidenceReceipt:
            raise PrivateServingError("exact EvidenceReceipt is required")
        self._verify_event(
            evidence_receipt.event_id,
            command_type=EVIDENCE_COMMAND,
            event_type=EVIDENCE_EVENT,
            definition_digest=self._evidence_command,
            admission_id=evidence_receipt.admission_id,
            payload_digest=evidence_receipt.evidence_digest,
            command_id=evidence_receipt.command_id,
            aggregate_id=evidence_receipt.evidence_aggregate_id,
            aggregate_version=evidence_receipt.aggregate_version,
            proof=proof,
        )
        material = self._objects.hydrate(
            HydrationRequest(evidence_receipt.admission_id, EVIDENCE_PURPOSE),
            proof=proof,
        )
        self._access(material.decision, self._evidence_policy, EVIDENCE_CLASS, EVIDENCE_USE)
        evidence = _evidence_from_bytes(material.data)
        batch = self._read_attempt_object(attempt_receipt, proof=proof)
        if (
            evidence.evidence_id != evidence_receipt.evidence_id
            or evidence.batch_id != batch.batch_id
            or evidence_receipt.evidence_aggregate_id != batch.evidence_aggregate_id
            or digest_bytes(material.data) != evidence_receipt.evidence_digest
        ):
            raise PrivateServingError("serving evidence receipt differs")
        self._validate_evidence(evidence, batch, proof=proof, current=False)
        return evidence

    def acknowledged_rows(
        self,
        evidence_receipt: EvidenceReceipt,
        attempt_receipt: AttemptReceipt,
        *,
        publication_receipt: PublicationReceipt,
        story_receipt: StoryVersionReceipt,
        candidate_port: StoryCandidateReadPort,
        proof: AuthenticationProof,
    ) -> AcknowledgedServing | None:
        """Return private rows only after exact authoritative acknowledgement."""

        evidence = self.read_evidence(
            evidence_receipt, attempt_receipt, proof=proof
        )
        if evidence.acknowledgement is None:
            return None
        batch = self._read_attempt(
            attempt_receipt,
            publication_receipt=publication_receipt,
            story_receipt=story_receipt,
            candidate_port=candidate_port,
            proof=proof,
        )
        rows: list[ProjectionRow] = []
        for attempt in batch.attempts:
            row = self._query(attempt.operation_key)
            if row is None or not self._row_matches(row, attempt, proof=proof):
                raise PrivateServingError("acknowledged private payload differs")
            rows.append(row)
        if tuple(item.operation_id for item in batch.attempts) != (
            evidence.acknowledgement.operation_ids
        ):
            raise PrivateServingError("acknowledged operations differ")
        recorded_at = self._events.provenance(
            evidence_receipt.event_id, proof=proof
        ).event.recorded_at
        return AcknowledgedServing(
            evidence.acknowledgement,
            recorded_at,
            tuple(rows),
        )

    def acknowledged_read_proof(
        self,
        evidence_receipt: EvidenceReceipt,
        attempt_receipt: AttemptReceipt,
        *,
        publication_receipt: PublicationReceipt,
        story_receipt: StoryVersionReceipt,
        candidate_port: StoryCandidateReadPort,
        proof: AuthenticationProof,
    ) -> PrivateServingReadProof | None:
        """Narrow a verified retained ACK into a query-only consumer proof."""

        serving = self.acknowledged_rows(
            evidence_receipt,
            attempt_receipt,
            publication_receipt=publication_receipt,
            story_receipt=story_receipt,
            candidate_port=candidate_port,
            proof=proof,
        )
        if serving is None:
            return None
        return PrivateServingReadProof(
            _READ_PROOF_TOKEN,
            evidence_receipt,
            attempt_receipt,
            serving,
        )

    def _build_batch(self, receipt, transaction) -> AttemptBatch:
        if (
            transaction.bundle is None
            or transaction.decision.outcome != "AUTO_PUBLISH"
            or transaction.bundle.target_id != self._target
            or tuple(item.surface_kind for item in transaction.operations) != _KINDS
        ):
            raise PrivateServingError("publication has no private target operations")
        attempts = tuple(
            _attempt(
                item,
                adapter=self._adapter,
                context=self._target_context,
                projection=self._projection_identity,
            )
            for item in transaction.operations
        )
        values = {
            "publication_id": receipt.publication_id,
            "publication_event_id": receipt.event_id,
            "publication_transaction_id": transaction.transaction_id,
            "publication_transaction_digest": receipt.transaction_digest,
            "publication_aggregate_version": receipt.aggregate_version,
            "attempts": attempts,
        }
        identity = {
            "schema_identity": ATTEMPT_SCHEMA,
            "publication_id": str(receipt.publication_id),
            "publication_event_id": receipt.event_id,
            "publication_transaction_id": transaction.transaction_id,
            "publication_transaction_digest": receipt.transaction_digest,
            "publication_aggregate_version": receipt.aggregate_version,
            "attempts": [item.value() for item in attempts],
        }
        return AttemptBatch(digest_bytes(canonical_json_bytes(identity)), **values)

    def _read_attempt(self, receipt, *, publication_receipt, story_receipt, candidate_port, proof):
        batch = self._read_attempt_object(receipt, proof=proof)
        transaction = self._publication.read(
            publication_receipt,
            story_receipt=story_receipt,
            candidate_port=candidate_port,
            proof=proof,
        )
        rebuilt = self._build_batch(publication_receipt, transaction)
        if rebuilt != batch:
            raise PrivateServingError("serving attempt publication differs")
        return batch

    def _read_attempt_object(self, receipt, *, proof):
        if type(receipt) is not AttemptReceipt:
            raise PrivateServingError("exact AttemptReceipt is required")
        self._verify_event(
            receipt.event_id,
            command_type=ATTEMPT_COMMAND,
            event_type=ATTEMPT_EVENT,
            definition_digest=self._attempt_command,
            admission_id=receipt.admission_id,
            payload_digest=receipt.batch_digest,
            command_id=receipt.command_id,
            aggregate_id=receipt.publication_id,
            aggregate_version=receipt.aggregate_version,
            proof=proof,
        )
        material = self._objects.hydrate(
            HydrationRequest(receipt.admission_id, ATTEMPT_PURPOSE), proof=proof
        )
        self._access(material.decision, self._attempt_policy, ATTEMPT_CLASS, ATTEMPT_USE)
        batch = _batch_from_bytes(material.data)
        if (
            batch.batch_id != receipt.batch_id
            or batch.publication_id != receipt.publication_id
            or batch.publication_aggregate_version + 1 != receipt.aggregate_version
            or digest_bytes(material.data) != receipt.batch_digest
        ):
            raise PrivateServingError("serving attempt receipt differs")
        return batch

    def _projection_row(self, attempt, *, applied_at, proof):
        material = self._objects.hydrate(
            HydrationRequest(attempt.surface_admission_id, SURFACE_PURPOSE), proof=proof
        )
        if digest_bytes(material.data) != attempt.surface_digest:
            raise PrivateServingError("serving payload digest differs")
        try:
            value = json.loads(material.data)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PrivateServingError("serving payload is malformed") from exc
        if (
            type(value) is not dict
            or value.get("payload_id") != attempt.surface_payload_id
            or value.get("kind") != attempt.surface_kind
            or canonical_json_bytes(value) != material.data
        ):
            raise PrivateServingError("serving payload binding differs")
        return ProjectionRow(
            attempt.operation_key,
            attempt.operation_id,
            attempt.attempt_id,
            attempt.surface_kind,
            attempt.surface_payload_id,
            attempt.surface_digest,
            material.data,
            applied_at,
        )

    def _row_matches(self, row, attempt, *, proof):
        expected = self._projection_row(attempt, applied_at=row.applied_at, proof=proof)
        return _same_effect(row, expected)

    def _query(self, operation_key):
        row = self._connection.execute(
            "SELECT operation_key,operation_id,attempt_id,surface_kind,payload_id,"
            "payload_digest,payload_bytes,applied_at FROM private_serving_payloads "
            "WHERE operation_key=?",
            (operation_key,),
        ).fetchone()
        return None if row is None else ProjectionRow(*tuple(row))

    def _evidence(self, batch, *, outcome, observed_at, observations, observation_rows, raw, failure_code, acknowledgement):
        values = {
            "batch_id": batch.batch_id,
            "target_id": self._target,
            "outcome": outcome,
            "observed_at": observed_at,
            "observation_method": "PRIVATE_SQLITE_QUERY_BY_KEY",
            "observer_principal_id": self._adapter,
            "raw_observation_digest": raw,
            "observation_rows": observation_rows,
            "observations": observations,
            "failure_code": failure_code,
            "acknowledgement": acknowledgement,
        }
        identity = _evidence_value(values)
        return DeliveryEvidence(digest_bytes(canonical_json_bytes(identity)), **values)

    def _observe_batch(self, batch, *, observed_at, proof):
        rows = tuple(self._query(item.operation_key) for item in batch.attempts)
        return self._evidence_from_rows(
            batch, rows=rows, observed_at=observed_at, proof=proof
        )

    def _evidence_from_rows(self, batch, *, rows, observed_at, proof):
        observed_instant = UtcTimestamp.parse(observed_at).value
        observations: list[OperationObservation] = []
        matched_rows: list[ProjectionRow] = []
        raw_rows: list[dict[str, object] | None] = []
        for attempt, row in zip(batch.attempts, rows, strict=True):
            raw_rows.append(None if row is None else _row_value(row))
            if row is None:
                result = "MISSING"
            elif self._row_matches(row, attempt, proof=proof):
                result = "MATCHING"
                matched_rows.append(row)
            else:
                result = "CONFLICT"
            observations.append(
                OperationObservation(
                    attempt.operation_key,
                    attempt.attempt_id,
                    result,
                    None if row is None else row.payload_digest,
                    None if row is None else row.operation_key,
                )
            )
        results = {item.result for item in observations}
        outcome = (
            "ACKNOWLEDGED"
            if results == {"MATCHING"}
            else "MISSING"
            if results == {"MISSING"}
            else "AMBIGUOUS"
        )
        acknowledgement = None
        if outcome == "ACKNOWLEDGED":
            applied = tuple(
                UtcTimestamp.parse(row.applied_at).value for row in matched_rows
            )
            if any(value > observed_instant for value in applied):
                raise PrivateServingError("serving observation precedes effect")
            values = {
                "batch_id": batch.batch_id,
                "target_id": self._target,
                "target_acknowledged_at": observed_at,
                "first_private_effect_at": matched_rows[applied.index(min(applied))].applied_at,
                "operation_ids": tuple(item.operation_id for item in batch.attempts),
            }
            acknowledgement = DeliveryAcknowledgement(
                digest_bytes(canonical_json_bytes(values)), **values
            )
        return self._evidence(
            batch,
            outcome=outcome,
            observed_at=observed_at,
            observations=tuple(observations),
            observation_rows=rows,
            raw=digest_bytes(canonical_json_bytes(raw_rows)),
            failure_code=None,
            acknowledgement=acknowledgement,
        )

    def _validate_evidence(self, evidence, batch, *, proof, current):
        if type(evidence) is not DeliveryEvidence or evidence.batch_id != batch.batch_id:
            raise PrivateServingError("serving evidence attempt differs")
        if (
            evidence.target_id != self._target
            or evidence.observer_principal_id != self._adapter
            or evidence.observation_method != "PRIVATE_SQLITE_QUERY_BY_KEY"
        ):
            raise PrivateServingError("serving evidence context differs")
        if evidence.outcome == "FAILED":
            if (
                evidence.observations
                or evidence.observation_rows
                or evidence.acknowledgement is not None
            ):
                raise PrivateServingError("serving failure evidence differs")
            return
        rows = (
            tuple(self._query(item.operation_key) for item in batch.attempts)
            if current
            else evidence.observation_rows
        )
        expected = self._evidence_from_rows(
            batch, rows=rows, observed_at=evidence.observed_at, proof=proof
        )
        if expected != evidence:
            raise PrivateServingError("serving observation evidence differs")

    def _verify_event(self, event_id, *, command_type, event_type, definition_digest, admission_id, payload_digest, command_id, aggregate_id, aggregate_version, proof):
        provenance = self._events.provenance(event_id, proof=proof)
        event = provenance.event
        if (
            provenance.command_definition.command_type != command_type
            or provenance.command_definition.definition_digest != definition_digest
            or event.command_definition_digest != definition_digest
            or event.event_type != event_type
            or event.object_admission_id != str(admission_id)
            or event.payload_digest != payload_digest
            or event.command_id != command_id
            or event.aggregate_id != str(aggregate_id)
            or event.aggregate_version != aggregate_version
            or event.principal_id != self._adapter
            or provenance.authentication.principal_id != self._adapter
            or provenance.authentication.authority_domain != self._domain
            or event.trust_scope != TrustScope.ADMITTED.value
            or event.security_scope != SERVING_SECURITY_SCOPE
            or event.retention_scope != SERVING_RETENTION_SCOPE
        ):
            raise PrivateServingError("serving authority event differs")

    def _access(self, decision, policy, object_class, allowed_use):
        if (
            decision.policy_contract_digest != policy
            or decision.principal_id != self._adapter
            or decision.authority_domain != self._domain
            or decision.object_class != object_class
            or decision.allowed_use != allowed_use
        ):
            raise PrivateServingError("serving object access differs")


def open_private_serving_delivery(
    path: str | Path,
    *,
    objects: GovernedObjects,
    commands: AuthorityCommands,
    events: AuthorityEvents,
    publication: OfflinePublication,
    adapter_principal_id: str,
    authority_domain: str,
    target_id: str,
    target_context_digest: str,
    attempt_hydration_policy_digest: str,
    evidence_hydration_policy_digest: str,
    attempt_admission_definition_digest: str,
    evidence_admission_definition_digest: str,
    attempt_command_definition_digest: str,
    evidence_command_definition_digest: str,
) -> PrivateServingDelivery:
    raw_path = str(path)
    if raw_path == ":memory:" or raw_path.startswith("file:"):
        raise PrivateServingError("private-serving requires a file target")
    _texts(adapter_principal_id, authority_domain, target_id)
    _digests(
        target_context_digest,
        attempt_hydration_policy_digest,
        evidence_hydration_policy_digest,
        attempt_admission_definition_digest,
        evidence_admission_definition_digest,
        attempt_command_definition_digest,
        evidence_command_definition_digest,
    )
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    connection = sqlite3.connect(path, isolation_level=None)
    if not existed:
        path.chmod(0o600)
    connection.row_factory = sqlite3.Row
    try:
        projection_identity = _projection_identity(
            path, target_id, target_context_digest
        )
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        if (
            connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1
            or str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            != "wal"
            or connection.execute("PRAGMA synchronous").fetchone()[0] != 2
            or not connection.execute("PRAGMA database_list").fetchone()[2]
        ):
            raise PrivateServingError("private-serving durability differs")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS private_serving_metadata(
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                schema_version TEXT NOT NULL,
                target_id TEXT NOT NULL,
                target_context_digest TEXT NOT NULL,
                projection_identity_digest TEXT NOT NULL,
                store_path TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS private_serving_payloads(
                operation_key TEXT PRIMARY KEY,
                operation_id TEXT NOT NULL UNIQUE,
                attempt_id TEXT NOT NULL,
                surface_kind TEXT NOT NULL,
                payload_id TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                payload_bytes BLOB NOT NULL,
                applied_at TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )
        connection.execute(
            "INSERT OR IGNORE INTO private_serving_metadata VALUES(1,?,?,?,?,?)",
            (
                "private-serving-projection-v1",
                target_id,
                target_context_digest,
                projection_identity,
                str(path),
            ),
        )
        metadata = connection.execute(
            "SELECT schema_version,target_id,target_context_digest,"
            "projection_identity_digest,store_path FROM private_serving_metadata "
            "WHERE singleton=1"
        ).fetchone()
        if tuple(metadata or ()) != (
            "private-serving-projection-v1",
            target_id,
            target_context_digest,
            projection_identity,
            str(path),
        ):
            raise PrivateServingError("private-serving target binding differs")
        return PrivateServingDelivery(
            _TOKEN,
            connection=connection,
            objects=objects,
            commands=commands,
            events=events,
            publication=publication,
            adapter_principal_id=adapter_principal_id,
            authority_domain=authority_domain,
            target_id=target_id,
            target_context_digest=target_context_digest,
            projection_identity_digest=projection_identity,
            attempt_hydration_policy_digest=attempt_hydration_policy_digest,
            evidence_hydration_policy_digest=evidence_hydration_policy_digest,
            attempt_admission_definition_digest=attempt_admission_definition_digest,
            evidence_admission_definition_digest=evidence_admission_definition_digest,
            attempt_command_definition_digest=attempt_command_definition_digest,
            evidence_command_definition_digest=evidence_command_definition_digest,
        )
    except Exception:
        connection.close()
        raise


def open_private_serving_read_port(
    path: str | Path,
    *,
    target_id: str,
    target_context_digest: str,
    proof: PrivateServingReadProof | None,
) -> PrivateServingReadPort:
    """Open an existing projection without acquiring any write authority."""

    raw_path = str(path)
    if raw_path == ":memory:" or raw_path.startswith("file:"):
        raise PrivateServingError("private-serving reader requires a file target")
    _texts(target_id)
    _digests(target_context_digest)
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise PrivateServingError("private-serving target is missing")
    connection = sqlite3.connect(
        f"{target.as_uri()}?mode=ro",
        uri=True,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        metadata = connection.execute(
            "SELECT schema_version,target_id,target_context_digest,"
            "projection_identity_digest,store_path FROM private_serving_metadata "
            "WHERE singleton=1"
        ).fetchone()
        if (
            connection.execute("PRAGMA query_only").fetchone()[0] != 1
            or tuple(metadata or ())
            != (
                "private-serving-projection-v1",
                target_id,
                target_context_digest,
                _projection_identity(target, target_id, target_context_digest),
                str(target),
            )
            or (
                proof is not None
                and proof.serving.acknowledgement.target_id != target_id
            )
        ):
            raise PrivateServingError("private-serving reader binding differs")
        return PrivateServingReadPort(
            _TOKEN,
            connection=connection,
            proof=proof,
        )
    except Exception:
        connection.close()
        raise


def _projection_identity(path: Path, target_id: str, context: str) -> str:
    return digest_bytes(
        canonical_json_bytes(
            {
                "path": str(path),
                "target_id": target_id,
                "target_context_digest": context,
            }
        )
    )


def _attempt(operation, *, adapter, context, projection):
    values = {
        "operation_id": operation.operation_id,
        "operation_key": operation.semantic_idempotency_key,
        "target_id": operation.target_id,
        "bundle_id": operation.bundle_id,
        "surface_kind": operation.surface_kind,
        "surface_payload_id": operation.surface_payload_id,
        "surface_admission_id": operation.surface_admission_id,
        "surface_digest": operation.surface_digest,
        "adapter_principal_id": adapter,
        "target_context_digest": context,
        "projection_identity_digest": projection,
    }
    identity = {**values, "surface_admission_id": str(operation.surface_admission_id)}
    return ServingAttempt(digest_bytes(canonical_json_bytes(identity)), **values)


def _batch_from_bytes(raw):
    value = _document(raw)
    try:
        attempts = tuple(
            ServingAttempt(
                **{
                    **item,
                    "surface_admission_id": ObjectAdmissionId.parse(
                        item["surface_admission_id"]
                    ),
                }
            )
            for item in value["attempts"]
        )
        batch = AttemptBatch(
            batch_id=value["batch_id"],
            publication_id=AggregateId.parse(value["publication_id"]),
            publication_event_id=value["publication_event_id"],
            publication_transaction_id=value["publication_transaction_id"],
            publication_transaction_digest=value["publication_transaction_digest"],
            publication_aggregate_version=value["publication_aggregate_version"],
            attempts=attempts,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PrivateServingError("serving attempt fields differ") from exc
    if batch.canonical_bytes() != raw:
        raise PrivateServingError("serving attempt is non-canonical")
    return batch


def _evidence_from_bytes(raw):
    value = _document(raw)
    try:
        ack_value = value["acknowledgement"]
        acknowledgement = (
            None
            if ack_value is None
            else DeliveryAcknowledgement(
                **{
                    **ack_value,
                    "operation_ids": tuple(ack_value["operation_ids"]),
                }
            )
        )
        evidence = DeliveryEvidence(
            evidence_id=value["evidence_id"],
            batch_id=value["batch_id"],
            target_id=value["target_id"],
            outcome=value["outcome"],
            observed_at=value["observed_at"],
            observation_method=value["observation_method"],
            observer_principal_id=value["observer_principal_id"],
            raw_observation_digest=value["raw_observation_digest"],
            observation_rows=tuple(
                None if item is None else _row_from_value(item)
                for item in value["observation_rows"]
            ),
            observations=tuple(OperationObservation(**item) for item in value["observations"]),
            failure_code=value["failure_code"],
            acknowledgement=acknowledgement,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PrivateServingError("serving evidence fields differ") from exc
    if evidence.canonical_bytes() != raw:
        raise PrivateServingError("serving evidence is non-canonical")
    return evidence


def _evidence_value(values):
    return {
        "schema_identity": EVIDENCE_SCHEMA,
        "batch_id": values["batch_id"],
        "target_id": values["target_id"],
        "outcome": values["outcome"],
        "observed_at": values["observed_at"],
        "observation_method": values["observation_method"],
        "observer_principal_id": values["observer_principal_id"],
        "raw_observation_digest": values["raw_observation_digest"],
        "observation_rows": [
            None if item is None else _row_value(item)
            for item in values["observation_rows"]
        ],
        "observations": [
            {
                "operation_key": item.operation_key,
                "attempt_id": item.attempt_id,
                "result": item.result,
                "observed_payload_digest": item.observed_payload_digest,
                "target_native_id": item.target_native_id,
            }
            for item in values["observations"]
        ],
        "failure_code": values["failure_code"],
        "acknowledgement": (
            None
            if values["acknowledgement"] is None
            else values["acknowledgement"].value()
        ),
    }


def _aggregate_for(digest):
    raw = bytearray.fromhex(digest.removeprefix("sha256:")[:32])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return AggregateId.parse(str(UUID(bytes=bytes(raw))))


def _row_value(row):
    return {
        "operation_key": row.operation_key,
        "operation_id": row.operation_id,
        "attempt_id": row.attempt_id,
        "surface_kind": row.surface_kind,
        "payload_id": row.payload_id,
        "payload_digest": row.payload_digest,
        "payload_bytes_digest": digest_bytes(row.payload_bytes),
        "payload_bytes": row.payload_bytes.decode("utf-8"),
        "applied_at": row.applied_at,
    }


def _row_from_value(value):
    try:
        row = ProjectionRow(
            operation_key=value["operation_key"],
            operation_id=value["operation_id"],
            attempt_id=value["attempt_id"],
            surface_kind=value["surface_kind"],
            payload_id=value["payload_id"],
            payload_digest=value["payload_digest"],
            payload_bytes=value["payload_bytes"].encode("utf-8"),
            applied_at=value["applied_at"],
        )
    except (AttributeError, KeyError, TypeError) as exc:
        raise PrivateServingError("serving observation row differs") from exc
    if value.get("payload_bytes_digest") != digest_bytes(row.payload_bytes):
        raise PrivateServingError("serving observation row digest differs")
    return row


def _same_effect(left: ProjectionRow, right: ProjectionRow) -> bool:
    """Attempt identity and observation time do not change one semantic effect."""

    return (
        left.operation_key,
        left.operation_id,
        left.surface_kind,
        left.payload_id,
        left.payload_digest,
        left.payload_bytes,
    ) == (
        right.operation_key,
        right.operation_id,
        right.surface_kind,
        right.payload_id,
        right.payload_digest,
        right.payload_bytes,
    )


def _document(raw):
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PrivateServingError("serving authority object is malformed") from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise PrivateServingError("serving authority object is non-canonical")
    return value


def _texts(*values):
    if any(type(value) is not str or not value for value in values):
        raise PrivateServingError("serving text binding differs")


def _digests(*values):
    try:
        for value in values:
            validate_sha256_digest(value)
    except (TypeError, ValueError) as exc:
        raise PrivateServingError("serving digest binding differs") from exc
