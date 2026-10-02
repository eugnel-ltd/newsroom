"""Stable Triage Work Item identities and immutable version authority.

This is the sole Increment 6A2 public module.  It records bounded manifests;
it grants no Proposal, Candidate, publication, evidence, egress, or operational
authority.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority.types import UtcTimestamp
from newsroom.checks import CheckOutcome, CheckRequest, TriggerRef
from newsroom.discovery import (
    DiscoverySignal,
    GateDecision,
    LeadDispositionDecision,
    LeadDispositionOutcome,
    NewsLead,
    WatchCondition,
)
from newsroom.increment5._retrieval_context_core import (
    RetrievalContextError,
    RetrievalContextPurgeReceipt,
)
from newsroom.increment5.retrieval_context import (
    RetrievalContextOutcome,
    RetrievalContextReason,
    RetrievalContextReceipt,
    RetrievalContextRequest,
)
from newsroom.increment6.outcomes import (
    PriorityLane,
    PrioritySelection,
)

TRIAGE_WORK_ITEM = "newsroom.increment6.triage-work-item.v1"
TRIAGE_WORK_ITEM_VERSION = "newsroom.increment6.triage-work-item-version.v1"
WORK_ITEM_CURRENT_STALE_RULES = "USE_TIME_EXACT_HEAD_AND_UPSTREAM_CURRENTNESS"
LEAD_DISPOSITION_WORK_ITEM_BINDING = "EXACT_IMMUTABLE_LEAD_AND_DISPOSITION"
WATCH_CONDITION_WORK_ITEM_BINDING = "EXACT_IMMUTABLE_WATCH_PROVENANCE"
SUPPLEMENTAL_DISCOVERY_REENTRY = "NEW_GOVERNED_DISCOVERY_LINEAGE_ONLY"

MAX_DECISION_LEADS = 32
MAX_CONTEXT_LEADS = 32
MAX_WORK_ITEM_BYTES = 32 * 1_024
MAX_VERSION_BYTES = 384 * 1_024
MAX_PRIORITY_INPUT_BYTES = 64 * 1_024
MAX_PRIORITY_INPUT_REFERENCES = 256
MAX_CANONICAL_NODES = 131_072
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NATIVE_RETRIEVAL_REQUEST_SCHEMA = (
    "newsroom.increment5.native-retrieval-context-request.v1"
)


class WorkItemContractError(ValueError):
    """A Work Item manifest, replay, or authority claim is invalid."""


class WorkItemStaleError(WorkItemContractError):
    """The current Work Item Version is stale or unusable."""


class RetrievalBindingState(StrEnum):
    REQUEST_PENDING = "REQUEST_PENDING"
    RECEIPT = "RECEIPT"


class ReentryKind(StrEnum):
    DEADLINE = "DEADLINE"
    REVIEW = "REVIEW"
    EXPIRY = "EXPIRY"
    OPERATOR_CONDITION = "OPERATOR_CONDITION"


def _digest(value: str, field: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise WorkItemContractError(f"{field} must be a canonical SHA-256 digest")
    return value


def _uuid(value: str, field: str) -> str:
    if type(value) is not str:
        raise WorkItemContractError(f"{field} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise WorkItemContractError(f"{field} must be a canonical UUID") from exc
    if str(parsed) != value:
        raise WorkItemContractError(f"{field} must be a canonical UUID")
    return value


def _exact_int(
    value: object,
    field: str,
    *,
    minimum: int = 1,
    maximum: int = 2**63 - 1,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise WorkItemContractError(f"{field} must be an exact positive integer")
    return value


def _text(value: object, field: str) -> str:
    if type(value) is not str:
        raise WorkItemContractError(f"{field} must be exact text")
    return value


def _decode(
    raw: bytes,
    *,
    maximum: int = MAX_VERSION_BYTES,
    maximum_nodes: int = MAX_CANONICAL_NODES,
) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        raise WorkItemContractError("canonical input is not bounded immutable bytes")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise WorkItemContractError(f"duplicate object name: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    except WorkItemContractError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
        MemoryError,
    ) as exc:
        raise WorkItemContractError("canonical input is invalid UTF-8 JSON") from exc
    pending: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if depth > 32 or nodes > maximum_nodes:
            raise WorkItemContractError("canonical input exceeds structural bounds")
        if isinstance(current, float):
            raise WorkItemContractError("canonical input contains a float")
        if isinstance(current, dict):
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)
    try:
        canonical = canonical_json_bytes(value)
    except (ValueError, TypeError, RecursionError, MemoryError) as exc:
        raise WorkItemContractError("canonical input cannot be normalised") from exc
    if not isinstance(value, dict) or canonical != raw:
        raise WorkItemContractError("input is not exact canonical JSON")
    return value


def _exact(value: object, fields: set[str], name: str) -> Mapping[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise WorkItemContractError(f"{name} fields are not exact")
    return value


def _tuple(value: object, name: str) -> tuple[object, ...]:
    if type(value) not in (tuple, list):
        raise WorkItemContractError(f"{name} must be a bounded sequence")
    return tuple(value)


def _canonical(value: object, name: str) -> bytes:
    try:
        return canonical_json_bytes(value)
    except WorkItemContractError:
        raise
    except Exception as exc:
        raise WorkItemContractError(f"{name} cannot be canonicalised") from exc


def _retained_digest(
    value: object, expected: str, maximum: int, name: str
) -> str:
    if type(value) is not bytes or not value or len(value) > maximum:
        raise WorkItemContractError(f"{name} must be bounded immutable bytes")
    if digest_bytes(value) != expected:
        raise WorkItemContractError(f"{name} digest differs")
    return expected


def _bounded_utf8(value: object, maximum: int, name: str) -> str:
    if type(value) is not str or not value:
        raise WorkItemContractError(f"{name} is invalid")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise WorkItemContractError(f"{name} is invalid") from exc
    if size > maximum:
        raise WorkItemContractError(f"{name} is invalid")
    return value


def _authority_record(value: object, expected: type[object], name: str) -> None:
    if type(value) is not expected:
        raise WorkItemContractError(f"{name} requires an exact authority record")
    try:
        request = value.request  # type: ignore[attr-defined]
        raw = request.canonical_bytes
        retained_digest = value.canonical_digest  # type: ignore[attr-defined]
    except Exception as exc:
        raise WorkItemContractError(f"{name} authority record is invalid") from exc
    if type(raw) is not bytes or digest_bytes(raw) != retained_digest:
        raise WorkItemContractError(f"{name} authority record differs")


def _retained_request(raw: object, digest: object, name: str) -> dict[str, object]:
    if type(raw) is not bytes or type(digest) is not str:
        raise WorkItemContractError(f"{name} retained row is malformed")
    if digest_bytes(raw) != digest:
        raise WorkItemContractError(f"{name} retained digest differs")
    try:
        value = json.loads(raw.decode("utf-8"))
        if type(value) is not dict or canonical_json_bytes(value) != raw:
            raise WorkItemContractError(f"{name} retained bytes are not canonical")
        return value
    except Exception as exc:
        raise WorkItemContractError(f"{name} retained bytes are invalid") from exc


@dataclass(frozen=True, slots=True)
class DecisionLeadBinding:
    lead_id: str
    lead_digest: str
    lead_event_id: str
    lead_aggregate_version: int
    gate_decision_id: str
    definition_id: str
    definition_version_id: str
    disposition_id: str
    disposition_digest: str
    disposition_event_id: str
    disposition_aggregate_version: int
    disposition_ordinal: int
    previous_disposition_id: str | None
    disposition_outcome: str

    @classmethod
    def from_authority(
        cls, lead: NewsLead, disposition: LeadDispositionDecision
    ) -> Self:
        _authority_record(lead, NewsLead, "decision Lead binding")
        _authority_record(
            disposition, LeadDispositionDecision, "decision Lead disposition binding"
        )
        if disposition.request.lead_id != lead.request.lead_id:
            raise WorkItemContractError("Lead disposition belongs to another Lead")
        if disposition.request.outcome is not LeadDispositionOutcome.QUEUED_FOR_TRIAGE:
            raise WorkItemContractError("decision Lead must be queued for triage")
        return cls(
            str(lead.request.lead_id),
            lead.canonical_digest,
            str(lead.event_id),
            lead.aggregate_version,
            str(disposition.request.gate_decision_id),
            str(lead.request.definition_id),
            str(lead.request.definition_version_id),
            str(disposition.request.decision_id),
            disposition.canonical_digest,
            str(disposition.event_id),
            disposition.aggregate_version,
            disposition.request.decision_ordinal,
            None
            if disposition.request.previous_decision_id is None
            else str(disposition.request.previous_decision_id),
            disposition.request.outcome.value,
        )

    def __post_init__(self) -> None:
        for value, field in (
            (self.lead_id, "lead_id"),
            (self.gate_decision_id, "gate_decision_id"),
            (self.definition_id, "definition_id"),
            (self.definition_version_id, "definition_version_id"),
            (self.disposition_id, "disposition_id"),
        ):
            _uuid(value, field)
        _uuid(self.lead_event_id, "lead_event_id")
        _uuid(self.disposition_event_id, "disposition_event_id")
        _exact_int(self.lead_aggregate_version, "lead_aggregate_version")
        _exact_int(
            self.disposition_aggregate_version,
            "disposition_aggregate_version",
        )
        _exact_int(self.disposition_ordinal, "disposition_ordinal")
        if self.previous_disposition_id is not None:
            _uuid(self.previous_disposition_id, "previous_disposition_id")
        _digest(self.lead_digest, "lead_digest")
        _digest(self.disposition_digest, "disposition_digest")
        if self.disposition_outcome != LeadDispositionOutcome.QUEUED_FOR_TRIAGE.value:
            raise WorkItemContractError("decision Lead disposition is not queued")
        if self.disposition_ordinal < 1 or (self.disposition_ordinal == 1) != (
            self.previous_disposition_id is None
        ):
            raise WorkItemContractError("Lead disposition predecessor differs")

    def stable_lead_value(self) -> tuple[object, ...]:
        return (
            self.lead_id,
            self.lead_digest,
            self.lead_event_id,
            self.lead_aggregate_version,
            self.definition_id,
            self.definition_version_id,
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "lead_id": self.lead_id,
            "lead_digest": self.lead_digest,
            "lead_event_id": self.lead_event_id,
            "lead_aggregate_version": self.lead_aggregate_version,
            "gate_decision_id": self.gate_decision_id,
            "definition_id": self.definition_id,
            "definition_version_id": self.definition_version_id,
            "disposition_id": self.disposition_id,
            "disposition_digest": self.disposition_digest,
            "disposition_event_id": self.disposition_event_id,
            "disposition_aggregate_version": self.disposition_aggregate_version,
            "disposition_ordinal": self.disposition_ordinal,
            "previous_disposition_id": self.previous_disposition_id,
            "disposition_outcome": self.disposition_outcome,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        fields = {
            "lead_id",
            "lead_digest",
            "lead_event_id",
            "lead_aggregate_version",
            "gate_decision_id",
            "definition_id",
            "definition_version_id",
            "disposition_id",
            "disposition_digest",
            "disposition_event_id",
            "disposition_aggregate_version",
            "disposition_ordinal",
            "previous_disposition_id",
            "disposition_outcome",
        }
        item = _exact(value, fields, "decision Lead binding")
        return cls(
            _text(item["lead_id"], "lead_id"),
            _text(item["lead_digest"], "lead_digest"),
            _text(item["lead_event_id"], "lead_event_id"),
            _exact_int(item["lead_aggregate_version"], "lead_aggregate_version"),
            _text(item["gate_decision_id"], "gate_decision_id"),
            _text(item["definition_id"], "definition_id"),
            _text(item["definition_version_id"], "definition_version_id"),
            _text(item["disposition_id"], "disposition_id"),
            _text(item["disposition_digest"], "disposition_digest"),
            _text(item["disposition_event_id"], "disposition_event_id"),
            _exact_int(
                item["disposition_aggregate_version"],
                "disposition_aggregate_version",
            ),
            _exact_int(item["disposition_ordinal"], "disposition_ordinal"),
            None
            if item["previous_disposition_id"] is None
            else _text(item["previous_disposition_id"], "previous_disposition_id"),
            _text(item["disposition_outcome"], "disposition_outcome"),
        )


@dataclass(frozen=True, slots=True)
class ContextLeadBinding:
    lead_id: str
    lead_digest: str
    lead_event_id: str
    lead_aggregate_version: int
    gate_decision_id: str
    definition_id: str
    definition_version_id: str

    @classmethod
    def from_authority(cls, lead: NewsLead) -> Self:
        _authority_record(lead, NewsLead, "context Lead binding")
        return cls(
            str(lead.request.lead_id),
            lead.canonical_digest,
            str(lead.event_id),
            lead.aggregate_version,
            str(lead.request.promoting_gate_decision_id),
            str(lead.request.definition_id),
            str(lead.request.definition_version_id),
        )

    def __post_init__(self) -> None:
        _uuid(self.lead_id, "context lead_id")
        _digest(self.lead_digest, "context lead_digest")
        _uuid(self.lead_event_id, "context Lead authority event")
        _exact_int(self.lead_aggregate_version, "context Lead aggregate version")
        _uuid(self.gate_decision_id, "context Gate Decision")
        _uuid(self.definition_id, "context Source Definition")
        _uuid(self.definition_version_id, "context Source Definition Version")

    def canonical_value(self) -> dict[str, object]:
        return {
            "lead_id": self.lead_id,
            "lead_digest": self.lead_digest,
            "lead_event_id": self.lead_event_id,
            "lead_aggregate_version": self.lead_aggregate_version,
            "gate_decision_id": self.gate_decision_id,
            "definition_id": self.definition_id,
            "definition_version_id": self.definition_version_id,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        item = _exact(
            value,
            {
                "lead_id",
                "lead_digest",
                "lead_event_id",
                "lead_aggregate_version",
                "gate_decision_id",
                "definition_id",
                "definition_version_id",
            },
            "context Lead binding",
        )
        return cls(
            _text(item["lead_id"], "lead_id"),
            _text(item["lead_digest"], "lead_digest"),
            _text(item["lead_event_id"], "lead_event_id"),
            _exact_int(
                item["lead_aggregate_version"], "context Lead aggregate version"
            ),
            _text(item["gate_decision_id"], "gate_decision_id"),
            _text(item["definition_id"], "definition_id"),
            _text(item["definition_version_id"], "definition_version_id"),
        )


@dataclass(frozen=True, slots=True)
class RetrievalInputBinding:
    state: RetrievalBindingState
    request_id: str
    idempotency_key: str
    request_digest: str
    request_bytes: bytes
    context_id: str | None = None
    context_digest: str | None = None
    outcome: str | None = None
    reason: str | None = None
    no_match: bool = False
    receipt_bytes: bytes | None = None

    @classmethod
    def request_pending(cls, request: RetrievalContextRequest) -> Self:
        if type(request) is not RetrievalContextRequest:
            raise WorkItemContractError("retrieval request must be typed")
        try:
            return cls(
                RetrievalBindingState.REQUEST_PENDING,
                str(request.request_id),
                request.idempotency_key,
                request.request_digest,
                request.canonical_bytes,
            )
        except WorkItemContractError:
            raise
        except Exception as exc:
            raise WorkItemContractError("retrieval request is invalid") from exc

    @classmethod
    def from_receipt(
        cls, request: RetrievalContextRequest, receipt: RetrievalContextReceipt
    ) -> Self:
        if type(request) is not RetrievalContextRequest or type(
            receipt
        ) is not RetrievalContextReceipt:
            raise WorkItemContractError("retrieval request and receipt must be typed")
        try:
            if (
                request.request_digest != receipt.request_digest
                or str(request.request_id) != receipt.request_id
            ):
                raise WorkItemContractError(
                    "retrieval receipt belongs to another request"
                )
            return cls(
                RetrievalBindingState.RECEIPT,
                receipt.request_id,
                request.idempotency_key,
                receipt.request_digest,
                request.canonical_bytes,
                receipt.context_id,
                receipt.receipt_digest,
                receipt.outcome.value,
                None if receipt.reason is None else receipt.reason.value,
                receipt.no_match,
                receipt.canonical_bytes,
            )
        except WorkItemContractError:
            raise
        except Exception as exc:
            raise WorkItemContractError("retrieval receipt is invalid") from exc

    def __post_init__(self) -> None:
        if type(self.state) is not RetrievalBindingState:
            raise WorkItemContractError("retrieval binding state must be typed")
        if type(self.no_match) is not bool:
            raise WorkItemContractError("retrieval no_match must be boolean")
        _uuid(self.request_id, "retrieval request_id")
        _bounded_utf8(self.idempotency_key, 512, "retrieval idempotency key")
        _digest(self.request_digest, "retrieval request_digest")
        _retained_digest(
            self.request_bytes,
            self.request_digest,
            MAX_VERSION_BYTES,
            "retrieval request bytes",
        )
        request = _decode(self.request_bytes, maximum=MAX_VERSION_BYTES)
        if request.get("schema_identity") == _NATIVE_RETRIEVAL_REQUEST_SCHEMA:
            try:
                from newsroom.increment5.native_retrieval import (
                    NativeRetrievalContextRequest,
                )

                NativeRetrievalContextRequest.from_bytes(self.request_bytes)
            except Exception as exc:
                raise WorkItemContractError(
                    "native retrieval request bytes differ"
                ) from exc
        else:
            _retained_digest(
                self.request_bytes,
                self.request_digest,
                MAX_WORK_ITEM_BYTES,
                "retrieval request bytes",
            )
        if (
            request.get("request_id") != self.request_id
            or request.get("idempotency_key") != self.idempotency_key
        ):
            raise WorkItemContractError("retrieval request identity differs from bytes")
        if self.state is RetrievalBindingState.REQUEST_PENDING:
            if (
                any(
                    value is not None
                    for value in (
                        self.context_id,
                        self.context_digest,
                        self.outcome,
                        self.reason,
                        self.receipt_bytes,
                    )
                )
                or self.no_match
            ):
                raise WorkItemContractError("pending retrieval contains a receipt")
        elif (
            self.receipt_bytes is None
            or self.context_id is None
            or self.context_digest is None
            or self.outcome is None
        ):
            raise WorkItemContractError("retrieval receipt binding is incomplete")
        else:
            try:
                RetrievalContextOutcome(self.outcome)
                if self.reason is not None:
                    RetrievalContextReason(self.reason)
            except (TypeError, ValueError) as exc:
                raise WorkItemContractError(
                    "retrieval outcome or reason is unsupported"
                ) from exc
            _uuid(self.context_id, "retrieval context_id")
            _digest(self.context_digest, "retrieval context_digest")
            _retained_digest(
                self.receipt_bytes,
                self.context_digest,
                256 * 1_024,
                "retrieval receipt bytes",
            )
            receipt = _decode(self.receipt_bytes, maximum=256 * 1_024)
            if (
                receipt.get("context_id") != self.context_id
                or receipt.get("request_id") != self.request_id
                or receipt.get("request_digest") != self.request_digest
                or receipt.get("outcome") != self.outcome
                or receipt.get("reason") != self.reason
                or receipt.get("no_match") is not self.no_match
            ):
                raise WorkItemContractError(
                    "retrieval receipt scalars differ from bytes"
                )

    @property
    def usable(self) -> bool:
        return (
            self.state is RetrievalBindingState.RECEIPT
            and self.outcome == RetrievalContextOutcome.COMPLETE.value
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "request_id": self.request_id,
            "idempotency_key": self.idempotency_key,
            "request_digest": self.request_digest,
            "request": _decode(self.request_bytes),
            "context_id": self.context_id,
            "context_digest": self.context_digest,
            "outcome": self.outcome,
            "reason": self.reason,
            "no_match": self.no_match,
            "receipt": None
            if self.receipt_bytes is None
            else _decode(self.receipt_bytes),
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        item = _exact(
            value,
            {
                "state",
                "request_id",
                "idempotency_key",
                "request_digest",
                "request",
                "context_id",
                "context_digest",
                "outcome",
                "reason",
                "no_match",
                "receipt",
            },
            "retrieval binding",
        )
        try:
            state = RetrievalBindingState(item["state"])
        except (TypeError, ValueError) as exc:
            raise WorkItemContractError(
                "retrieval binding state is unsupported"
            ) from exc
        if type(item["no_match"]) is not bool:
            raise WorkItemContractError("retrieval no_match must be boolean")
        return cls(
            state,
            _text(item["request_id"], "request_id"),
            _text(item["idempotency_key"], "idempotency_key"),
            _text(item["request_digest"], "request_digest"),
            _canonical(item["request"], "retrieval request"),
            None if item["context_id"] is None else _text(item["context_id"], "context_id"),
            None if item["context_digest"] is None else _text(item["context_digest"], "context_digest"),
            None if item["outcome"] is None else _text(item["outcome"], "outcome"),
            None if item["reason"] is None else _text(item["reason"], "reason"),
            item["no_match"],
            None if item["receipt"] is None else _canonical(item["receipt"], "retrieval receipt"),
        )


class RetrievalContextAuthority:
    """Read-only exact adapter over the separate Increment 5 journal."""

    def __init__(
        self,
        journal_path: Path,
        records: Mapping[
            str, tuple[RetrievalContextRequest, RetrievalContextReceipt | None]
        ],
        *,
        native_context_read_port: object | None = None,
    ) -> None:
        try:
            self._path = Path(journal_path)
        except (TypeError, ValueError, OSError) as exc:
            raise WorkItemContractError("retrieval authority path is invalid") from exc
        if not self._path.is_file():
            raise WorkItemContractError("retrieval authority journal is absent")
        if type(records) is not dict:
            raise WorkItemContractError("retrieval authority records differ")
        try:
            valid_records = not any(
                type(digest) is not str
                or type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not RetrievalContextRequest
                or (
                    pair[1] is not None
                    and type(pair[1]) is not RetrievalContextReceipt
                )
                or digest != pair[0].request_digest
                for digest, pair in records.items()
            )
            copied = dict(records) if valid_records else {}
        except Exception as exc:
            raise WorkItemContractError("retrieval authority records differ") from exc
        if not valid_records:
            raise WorkItemContractError("retrieval authority records differ")
        self._records = copied
        if native_context_read_port is not None:
            from newsroom.increment5.native_retrieval import NativeRetrievalContextReadPort
            if type(native_context_read_port) is not NativeRetrievalContextReadPort:
                raise WorkItemContractError("native retrieval authority port differs")
        self._native_context_read_port = native_context_read_port

    def attach(self, connection: sqlite3.Connection) -> None:
        attached = {
            str(row[1]): str(row[2])
            for row in connection.execute("PRAGMA database_list")
        }
        if "retrieval_authority" not in attached:
            connection.execute(
                "ATTACH DATABASE ? AS retrieval_authority", (str(self._path),)
            )
        elif Path(attached["retrieval_authority"]).resolve() != self._path.resolve():
            raise WorkItemContractError("another retrieval authority is attached")
        contexts = tuple(
            str(row[1])
            for row in connection.execute(
                "PRAGMA retrieval_authority.table_info(increment5d2_retrieval_contexts)"
            )
        )
        purges = tuple(
            str(row[1])
            for row in connection.execute(
                "PRAGMA retrieval_authority.table_info(increment5d2_retrieval_context_purges)"
            )
        )
        if contexts != (
            "idempotency_key",
            "request_digest",
            "receipt_digest",
            "receipt_bytes",
        ) or purges != (
            "purge_id",
            "idempotency_key",
            "request_digest",
            "prior_receipt_digest",
            "purge_receipt_digest",
            "purge_receipt_bytes",
        ):
            raise WorkItemContractError("retrieval authority journal schema differs")
        context_indexes = {
            str(row[1]): str(row[3])
            for row in connection.execute(
                "PRAGMA retrieval_authority.index_list(increment5d2_retrieval_contexts)"
            )
        }
        purge_indexes = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA retrieval_authority.index_list("
                "increment5d2_retrieval_context_purges)"
            )
        }
        if (
            not any(origin == "pk" for origin in context_indexes.values())
            or "increment5d2_retrieval_context_purges_by_key" not in purge_indexes
        ):
            raise WorkItemContractError("retrieval authority indexes differ")

    def verify(
        self, connection: sqlite3.Connection, binding: RetrievalInputBinding
    ) -> None:
        if self._verify_native_binding(binding):
            return
        self._verify_typed_binding(binding, historical=False)
        try:
            purges = connection.execute(
                "SELECT purge_id,idempotency_key,request_digest,"
                "prior_receipt_digest,purge_receipt_digest,purge_receipt_bytes FROM "
                "retrieval_authority.increment5d2_retrieval_context_purges "
                "WHERE idempotency_key=? ORDER BY purge_id",
                (binding.idempotency_key,),
            ).fetchall()
            row = connection.execute(
                "SELECT request_digest,receipt_digest,receipt_bytes "
                "FROM retrieval_authority.increment5d2_retrieval_contexts "
                "WHERE idempotency_key=?",
                (binding.idempotency_key,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise WorkItemContractError("retrieval authority journal differs") from exc
        for purge in purges:
            self._validate_purge(purge, binding)
        if purges:
            raise WorkItemContractError("retrieval authority was purged")
        if binding.state is RetrievalBindingState.REQUEST_PENDING:
            if row is not None:
                raise WorkItemContractError("pending retrieval already has a receipt")
        elif row is None:
            raise WorkItemContractError("retrieval authority retained bytes differ")
        try:
            request, receipt = self._records[binding.request_digest]
        except Exception as exc:
            raise WorkItemContractError(
                "retrieval authority resolution failed"
            ) from exc
        if (
            type(request) is not RetrievalContextRequest
            or request.request_digest != binding.request_digest
            or request.request_id != binding.request_id
            or request.idempotency_key != binding.idempotency_key
            or request.canonical_bytes != binding.request_bytes
        ):
            raise WorkItemContractError("retrieval authority request differs")
        if (
            binding.state is RetrievalBindingState.REQUEST_PENDING
            and receipt is not None
        ):
            raise WorkItemContractError("pending retrieval already has a receipt")
        if binding.state is RetrievalBindingState.RECEIPT and (
            type(receipt) is not RetrievalContextReceipt
            or receipt.request_digest != request.request_digest
            or receipt.request_id != request.request_id
            or receipt.context_id != binding.context_id
            or receipt.receipt_digest != binding.context_digest
            or receipt.canonical_bytes != binding.receipt_bytes
            or receipt.outcome.value != binding.outcome
            or (None if receipt.reason is None else receipt.reason.value)
            != binding.reason
            or receipt.no_match != binding.no_match
        ):
            raise WorkItemContractError("retrieval authority receipt differs")
        if binding.state is RetrievalBindingState.REQUEST_PENDING:
            return
        if (
            row is None
            or row[0] != binding.request_digest
            or row[1] != binding.context_digest
            or bytes(row[2]) != binding.receipt_bytes
            or digest_bytes(bytes(row[2])) != row[1]
        ):
            raise WorkItemContractError("retrieval authority retained bytes differ")

    def _verify_native_binding(self, binding: RetrievalInputBinding) -> bool:
        try:
            value = _decode(binding.request_bytes, maximum=MAX_VERSION_BYTES)
        except WorkItemContractError:
            return False
        if value.get("schema_identity") != _NATIVE_RETRIEVAL_REQUEST_SCHEMA:
            return False
        if self._native_context_read_port is None or binding.state is not RetrievalBindingState.RECEIPT or binding.receipt_bytes is None:
            raise WorkItemContractError("native retrieval authority is unavailable")
        try:
            from newsroom.increment5.native_retrieval import (
                NativeRetrievalContextReceipt,
                NativeRetrievalContextRequest,
            )
            request = NativeRetrievalContextRequest.from_bytes(binding.request_bytes)
            receipt = NativeRetrievalContextReceipt.from_bytes(binding.receipt_bytes)
            context = self._native_context_read_port.require(receipt)
        except Exception as exc:
            raise WorkItemContractError("native retrieval authority differs") from exc
        if (
            request.request_digest != binding.request_digest
            or request.request_id != binding.request_id
            or request.idempotency_key != binding.idempotency_key
            or receipt.request_digest != request.request_digest
            or receipt.request_id != request.request_id
            or receipt.context_id != binding.context_id
            or receipt.receipt_digest != binding.context_digest
            or receipt.outcome != binding.outcome
            or receipt.reason != binding.reason
            or receipt.no_match != binding.no_match
            or context.context_id != receipt.context_id
            or context.request_digest != receipt.request_digest
        ):
            raise WorkItemContractError("native retrieval authority binding differs")
        return True

    def _verify_typed_binding(
        self, binding: RetrievalInputBinding, *, historical: bool
    ) -> None:
        if self._verify_native_binding(binding):
            return
        try:
            request, receipt = self._records[binding.request_digest]
        except Exception as exc:
            raise WorkItemContractError(
                "retrieval authority resolution failed"
            ) from exc
        if (
            type(request) is not RetrievalContextRequest
            or request.request_digest != binding.request_digest
            or str(request.request_id) != binding.request_id
            or request.idempotency_key != binding.idempotency_key
            or request.canonical_bytes != binding.request_bytes
        ):
            raise WorkItemContractError("retrieval authority request differs")
        if binding.state is RetrievalBindingState.REQUEST_PENDING:
            if not historical and receipt is not None:
                raise WorkItemContractError("pending retrieval already has a receipt")
            return
        if (
            type(receipt) is not RetrievalContextReceipt
            or receipt.request_digest != request.request_digest
            or receipt.request_id != str(request.request_id)
            or receipt.context_id != binding.context_id
            or receipt.receipt_digest != binding.context_digest
            or receipt.canonical_bytes != binding.receipt_bytes
            or receipt.outcome.value != binding.outcome
            or (None if receipt.reason is None else receipt.reason.value)
            != binding.reason
            or receipt.no_match != binding.no_match
        ):
            raise WorkItemContractError("retrieval authority receipt differs")

    @staticmethod
    def _validate_purge(
        row: tuple[object, ...], binding: RetrievalInputBinding
    ) -> RetrievalContextPurgeReceipt:
        try:
            raw = bytes(row[5])
            purge = RetrievalContextPurgeReceipt.from_canonical_bytes(raw)
        except (
            RetrievalContextError,
            TypeError,
            ValueError,
            RecursionError,
            MemoryError,
        ) as exc:
            raise WorkItemContractError("retrieval purge tombstone differs") from exc
        if (
            digest_bytes(raw) != row[4]
            or purge.purge_id != row[0]
            or purge.idempotency_key != row[1]
            or purge.request_digest != row[2]
            or purge.prior_receipt_digest != row[3]
            or purge.idempotency_key != binding.idempotency_key
            or purge.request_digest != binding.request_digest
        ):
            raise WorkItemContractError("retrieval purge tombstone differs")
        return purge

    def verify_retained_integrity(
        self, connection: sqlite3.Connection, binding: RetrievalInputBinding
    ) -> None:
        if self._verify_native_binding(binding):
            return
        self._verify_typed_binding(binding, historical=True)
        try:
            purges = connection.execute(
                "SELECT purge_id,idempotency_key,request_digest,"
                "prior_receipt_digest,purge_receipt_digest,purge_receipt_bytes FROM "
                "retrieval_authority.increment5d2_retrieval_context_purges "
                "WHERE idempotency_key=? ORDER BY purge_id",
                (binding.idempotency_key,),
            ).fetchall()
            live = connection.execute(
                "SELECT request_digest,receipt_digest,receipt_bytes "
                "FROM retrieval_authority.increment5d2_retrieval_contexts "
                "WHERE idempotency_key=?",
                (binding.idempotency_key,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise WorkItemContractError("retrieval authority journal differs") from exc
        receipts = tuple(self._validate_purge(row, binding) for row in purges)
        if live is not None and receipts:
            raise WorkItemContractError("retrieval live receipt contradicts purge")
        if receipts:
            context_ids = {receipt.context_id for receipt in receipts}
            inventories = {
                receipt.context_derivative_identities for receipt in receipts
            }
            if len(context_ids) != 1 or len(inventories) != 1:
                raise WorkItemContractError(
                    "retrieval purge tombstone history differs"
                )
            seen_derivatives: set[tuple[str, str, str, str]] = set()
            for receipt in receipts:
                purged = set(receipt.purged_derivative_identities)
                if seen_derivatives.intersection(purged):
                    raise WorkItemContractError(
                        "retrieval purge tombstone history overlaps"
                    )
                seen_derivatives.update(purged)
            if binding.state is RetrievalBindingState.RECEIPT and (
                context_ids != {binding.context_id}
                or any(
                    receipt.prior_receipt_digest != binding.context_digest
                    for receipt in receipts
                )
            ):
                raise WorkItemContractError(
                    "retrieval purge tombstone history differs"
                )
        if binding.state is RetrievalBindingState.REQUEST_PENDING:
            return
        if receipts:
            return
        if (
            live is None
            or live[0] != binding.request_digest
            or live[1] != binding.context_digest
            or bytes(live[2]) != binding.receipt_bytes
            or digest_bytes(bytes(live[2])) != live[1]
        ):
            raise WorkItemContractError("retrieval authority retained bytes differ")

    def verify_retained_currentness(
        self, connection: sqlite3.Connection, binding: RetrievalInputBinding
    ) -> bool:
        """Require intact retention, then report current usability."""
        # Both native checks use the same current governed context read. Perform
        # it once here; retain no result beyond this operation.
        if self._verify_native_binding(binding):
            return True
        self.verify_retained_integrity(connection, binding)
        try:
            self.verify(connection, binding)
        except WorkItemContractError:
            return False
        return True


@dataclass(frozen=True, slots=True)
class WatchConditionWorkItemBinding:
    watch_condition_id: str
    watch_condition_digest: str
    lead_id: str
    watch_event_id: str
    watch_aggregate_version: int
    source_disposition_id: str
    source_disposition_digest: str
    source_disposition_ordinal: int
    source_previous_disposition_id: str | None
    source_disposition_event_id: str
    source_disposition_aggregate_version: int
    allowed_reentry_kinds: tuple[str, ...]
    observable_transition: bool

    @classmethod
    def from_authority(
        cls, watch: WatchCondition, source_disposition: LeadDispositionDecision
    ) -> Self:
        _authority_record(watch, WatchCondition, "Watch binding")
        _authority_record(
            source_disposition,
            LeadDispositionDecision,
            "Watch source disposition binding",
        )
        request = source_disposition.request
        if (
            request.outcome is not LeadDispositionOutcome.WATCH_DEFER
            or request.watch_condition_id != watch.request.watch_condition_id
            or request.lead_id != watch.request.lead_id
            or request.next_action.kind.value != "RESUME_ON_WATCH"
        ):
            raise WorkItemContractError("Watch source disposition differs")
        allowed: list[str] = []
        if watch.request.expected_occurrence and watch.request.review_at is not None:
            allowed.append(ReentryKind.DEADLINE.value)
        if watch.request.review_at is not None:
            allowed.append(ReentryKind.REVIEW.value)
        if watch.request.expires_at is not None:
            allowed.append(ReentryKind.EXPIRY.value)
        if watch.request.operator_review_condition:
            allowed.append(ReentryKind.OPERATOR_CONDITION.value)
        return cls(
            str(watch.request.watch_condition_id),
            watch.canonical_digest,
            str(watch.request.lead_id),
            str(watch.event_id),
            watch.aggregate_version,
            str(request.decision_id),
            source_disposition.canonical_digest,
            request.decision_ordinal,
            None
            if request.previous_decision_id is None
            else str(request.previous_decision_id),
            str(source_disposition.event_id),
            source_disposition.aggregate_version,
            tuple(sorted(set(allowed))),
            bool(
                watch.request.corroborating_lead_id
                or watch.request.resume_transition_kinds
            ),
        )

    def __post_init__(self) -> None:
        _uuid(self.watch_condition_id, "watch_condition_id")
        _uuid(self.lead_id, "watch lead_id")
        _digest(self.watch_condition_digest, "watch_condition_digest")
        _uuid(self.watch_event_id, "Watch authority event")
        _exact_int(self.watch_aggregate_version, "Watch aggregate version")
        _uuid(self.source_disposition_id, "Watch source disposition")
        _digest(self.source_disposition_digest, "Watch source disposition digest")
        _exact_int(self.source_disposition_ordinal, "Watch source disposition ordinal")
        _uuid(self.source_disposition_event_id, "Watch disposition authority event")
        _exact_int(
            self.source_disposition_aggregate_version,
            "Watch disposition aggregate version",
        )
        if (
            type(self.allowed_reentry_kinds) is not tuple
            or not self.allowed_reentry_kinds
            or tuple(sorted(set(self.allowed_reentry_kinds)))
            != self.allowed_reentry_kinds
            or any(type(value) is not str for value in self.allowed_reentry_kinds)
        ):
            raise WorkItemContractError("Watch re-entry kinds are invalid")
        if type(self.observable_transition) is not bool:
            raise WorkItemContractError("Watch transition flag must be boolean")
        if self.source_previous_disposition_id is not None:
            _uuid(self.source_previous_disposition_id, "Watch source predecessor")

    def canonical_value(self) -> dict[str, object]:
        return {
            "watch_condition_id": self.watch_condition_id,
            "watch_condition_digest": self.watch_condition_digest,
            "lead_id": self.lead_id,
            "watch_event_id": self.watch_event_id,
            "watch_aggregate_version": self.watch_aggregate_version,
            "source_disposition_id": self.source_disposition_id,
            "source_disposition_digest": self.source_disposition_digest,
            "source_disposition_ordinal": self.source_disposition_ordinal,
            "source_previous_disposition_id": self.source_previous_disposition_id,
            "source_disposition_event_id": self.source_disposition_event_id,
            "source_disposition_aggregate_version": self.source_disposition_aggregate_version,
            "allowed_reentry_kinds": list(self.allowed_reentry_kinds),
            "observable_transition": self.observable_transition,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        item = _exact(
            value,
            {
                "watch_condition_id",
                "watch_condition_digest",
                "lead_id",
                "watch_event_id",
                "watch_aggregate_version",
                "source_disposition_id",
                "source_disposition_digest",
                "source_disposition_ordinal",
                "source_previous_disposition_id",
                "source_disposition_event_id",
                "source_disposition_aggregate_version",
                "allowed_reentry_kinds",
                "observable_transition",
            },
            "Watch binding",
        )
        return cls(
            _text(item["watch_condition_id"], "watch_condition_id"),
            _text(item["watch_condition_digest"], "watch_condition_digest"),
            _text(item["lead_id"], "lead_id"),
            _text(item["watch_event_id"], "watch_event_id"),
            _exact_int(item["watch_aggregate_version"], "Watch aggregate version"),
            _text(item["source_disposition_id"], "source_disposition_id"),
            _text(item["source_disposition_digest"], "source_disposition_digest"),
            _exact_int(
                item["source_disposition_ordinal"], "Watch source disposition ordinal"
            ),
            None
            if item["source_previous_disposition_id"] is None
            else _text(item["source_previous_disposition_id"], "source_previous_disposition_id"),
            _text(item["source_disposition_event_id"], "source_disposition_event_id"),
            _exact_int(
                item["source_disposition_aggregate_version"],
                "Watch disposition aggregate version",
            ),
            tuple(
                _text(value, "Watch re-entry kind")
                for value in _tuple(
                    item["allowed_reentry_kinds"], "Watch re-entry kinds"
                )
            ),
            item["observable_transition"],
        )


@dataclass(frozen=True, slots=True)
class SupplementalDiscoveryReentry:
    source_work_item_id: str
    source_version_id: str
    source_version_digest: str
    source_lead_disposition_id: str
    source_lead_disposition_digest: str
    source_lead_disposition_event_id: str
    source_lead_disposition_aggregate_version: int
    source_lead_disposition_outcome: str
    source_approval_route: str
    trigger_id: str
    check_request_id: str
    check_outcome_id: str
    signal_id: str
    gate_decision_id: str
    lead_id: str
    queued_disposition_id: str
    target_work_item_id: str
    target_version_id: str
    lineage_bindings: tuple[SupplementalLineageBinding, ...] = ()

    def __post_init__(self) -> None:
        if type(self.lineage_bindings) is not tuple or any(
            type(value) is not SupplementalLineageBinding
            for value in self.lineage_bindings
        ):
            raise WorkItemContractError(
                "supplemental lineage must be an exact typed tuple"
            )
        for field in (
            "source_work_item_id",
            "source_version_id",
            "source_lead_disposition_id",
            "check_request_id",
            "check_outcome_id",
            "signal_id",
            "gate_decision_id",
            "lead_id",
            "queued_disposition_id",
            "target_work_item_id",
            "target_version_id",
        ):
            _uuid(getattr(self, field), field)
        _bounded_utf8(self.trigger_id, 256, "trigger_id")
        _digest(self.source_version_digest, "source_version_digest")
        _digest(
            self.source_lead_disposition_digest,
            "source Lead disposition digest",
        )
        _uuid(
            self.source_lead_disposition_event_id,
            "source Lead disposition event",
        )
        _exact_int(
            self.source_lead_disposition_aggregate_version,
            "source Lead disposition aggregate version",
        )
        if self.source_lead_disposition_outcome != "LEAD_SUPPLEMENTAL_DISCOVERY":
            raise WorkItemContractError(
                "supplemental source approval outcome differs"
            )
        if self.source_approval_route != "REQUEST_SUPPLEMENTAL_DISCOVERY":
            raise WorkItemContractError("supplemental source approval route differs")
        if (
            self.source_work_item_id == self.target_work_item_id
            or self.source_version_id == self.target_version_id
        ):
            raise WorkItemContractError(
                "supplemental discovery must create a new Work Item lineage"
            )
        try:
            kinds = tuple(item.kind for item in self.lineage_bindings)
        except Exception as exc:
            raise WorkItemContractError("supplemental lineage is invalid") from exc
        expected = (
            "TRIGGER",
            "CHECK_REQUEST",
            "CHECK_OUTCOME",
            "SIGNAL",
            "GATE",
            "LEAD",
            "QUEUED_DISPOSITION",
        )
        if kinds != expected:
            raise WorkItemContractError(
                "supplemental discovery requires the exact governed lineage"
            )
        ids = (
            self.trigger_id,
            self.check_request_id,
            self.check_outcome_id,
            self.signal_id,
            self.gate_decision_id,
            self.lead_id,
            self.queued_disposition_id,
        )
        for index, (binding, identifier) in enumerate(
            zip(self.lineage_bindings, ids, strict=True)
        ):
            if binding.identifier != identifier or (
                index and binding.parent_id != ids[index - 1]
            ):
                raise WorkItemContractError(
                    "supplemental discovery lineage binding differs"
                )

    def canonical_value(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in (
                "source_work_item_id",
                "source_version_id",
                "source_version_digest",
                "source_lead_disposition_id",
                "source_lead_disposition_digest",
                "source_lead_disposition_event_id",
                "source_lead_disposition_aggregate_version",
                "source_lead_disposition_outcome",
                "source_approval_route",
                "trigger_id",
                "check_request_id",
                "check_outcome_id",
                "signal_id",
                "gate_decision_id",
                "lead_id",
                "queued_disposition_id",
                "target_work_item_id",
                "target_version_id",
            )
        } | {
            "lineage_bindings": [
                item.canonical_value() for item in self.lineage_bindings
            ],
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        item = _exact(
            value,
            {
                "source_work_item_id",
                "source_version_id",
                "source_version_digest",
                "source_lead_disposition_id",
                "source_lead_disposition_digest",
                "source_lead_disposition_event_id",
                "source_lead_disposition_aggregate_version",
                "source_lead_disposition_outcome",
                "source_approval_route",
                "trigger_id",
                "check_request_id",
                "check_outcome_id",
                "signal_id",
                "gate_decision_id",
                "lead_id",
                "queued_disposition_id",
                "target_work_item_id",
                "target_version_id",
                "lineage_bindings",
            },
            "supplemental discovery re-entry",
        )
        if type(item["lineage_bindings"]) is not list:
            raise WorkItemContractError("supplemental lineage must be an array")
        return cls(
            _text(item["source_work_item_id"], "source_work_item_id"),
            _text(item["source_version_id"], "source_version_id"),
            _text(item["source_version_digest"], "source_version_digest"),
            _text(item["source_lead_disposition_id"], "source_lead_disposition_id"),
            _text(item["source_lead_disposition_digest"], "source_lead_disposition_digest"),
            _text(item["source_lead_disposition_event_id"], "source_lead_disposition_event_id"),
            _exact_int(
                item["source_lead_disposition_aggregate_version"],
                "source Lead disposition aggregate version",
            ),
            _text(item["source_lead_disposition_outcome"], "source_lead_disposition_outcome"),
            _text(item["source_approval_route"], "source_approval_route"),
            _text(item["trigger_id"], "trigger_id"),
            _text(item["check_request_id"], "check_request_id"),
            _text(item["check_outcome_id"], "check_outcome_id"),
            _text(item["signal_id"], "signal_id"),
            _text(item["gate_decision_id"], "gate_decision_id"),
            _text(item["lead_id"], "lead_id"),
            _text(item["queued_disposition_id"], "queued_disposition_id"),
            _text(item["target_work_item_id"], "target_work_item_id"),
            _text(item["target_version_id"], "target_version_id"),
            tuple(
                SupplementalLineageBinding.from_value(v)
                for v in item["lineage_bindings"]
            ),
        )


@dataclass(frozen=True, slots=True)
class SupplementalLineageBinding:
    kind: str
    identifier: str
    digest: str
    parent_id: str | None
    authority_event_id: str | None = None
    authority_aggregate_version: int | None = None

    @classmethod
    def from_authority(
        cls,
        record: TriggerRef
        | CheckRequest
        | CheckOutcome
        | DiscoverySignal
        | GateDecision
        | NewsLead
        | LeadDispositionDecision,
    ) -> Self:
        if type(record) is TriggerRef:
            try:
                raw = _canonical(record.canonical_value(), "Trigger")
            except Exception as exc:
                raise WorkItemContractError("Trigger authority record is invalid") from exc
            return cls("TRIGGER", record.trigger_id, digest_bytes(raw), None)
        kinds: tuple[tuple[type[object], str, str, str | None], ...] = (
            (CheckRequest, "CHECK_REQUEST", "request_id", None),
            (CheckOutcome, "CHECK_OUTCOME", "outcome_id", "request_id"),
            (DiscoverySignal, "SIGNAL", "signal_id", "check_outcome_id"),
            (GateDecision, "GATE", "decision_id", "signal_id"),
            (NewsLead, "LEAD", "lead_id", "promoting_gate_decision_id"),
            (
                LeadDispositionDecision,
                "QUEUED_DISPOSITION",
                "decision_id",
                "lead_id",
            ),
        )
        for record_type, kind, identity_field, parent_field in kinds:
            if type(record) is record_type:
                _authority_record(record, record_type, f"{kind} lineage binding")
                request = record.request
                parent = (
                    str(request.trigger.trigger_id)
                    if kind == "CHECK_REQUEST"
                    else str(getattr(request, parent_field))
                )
                return cls(
                    kind,
                    str(getattr(request, identity_field)),
                    record.canonical_digest,
                    parent,
                    str(record.event_id),
                    record.aggregate_version,
                )
        raise WorkItemContractError("supplemental authority record is unsupported")

    def __post_init__(self) -> None:
        if self.kind == "TRIGGER":
            if type(self.identifier) is not str or not self.identifier:
                raise WorkItemContractError("supplemental Trigger identity differs")
        else:
            _uuid(self.identifier, "supplemental lineage identity")
        if self.kind == "TRIGGER":
            if (
                self.authority_event_id is not None
                or self.authority_aggregate_version is not None
            ):
                raise WorkItemContractError("TriggerRef has no event authority")
        else:
            _uuid(self.authority_event_id, "supplemental authority event")
            _exact_int(
                self.authority_aggregate_version,
                "supplemental authority aggregate version",
            )
        if self.parent_id is not None:
            if self.kind == "CHECK_REQUEST":
                if not self.parent_id:
                    raise WorkItemContractError("supplemental Trigger parent differs")
            else:
                _uuid(self.parent_id, "supplemental lineage parent")
        _digest(self.digest, "supplemental lineage digest")

    def canonical_value(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "identifier": self.identifier,
            "digest": self.digest,
            "parent_id": self.parent_id,
            "authority_event_id": self.authority_event_id,
            "authority_aggregate_version": self.authority_aggregate_version,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        item = _exact(
            value,
            {
                "kind",
                "identifier",
                "digest",
                "parent_id",
                "authority_event_id",
                "authority_aggregate_version",
            },
            "supplemental lineage binding",
        )
        return cls(
            _text(item["kind"], "kind"),
            _text(item["identifier"], "identifier"),
            _text(item["digest"], "digest"),
            None if item["parent_id"] is None else _text(item["parent_id"], "parent_id"),
            None
            if item["authority_event_id"] is None
            else _text(item["authority_event_id"], "authority_event_id"),
            None
            if item["authority_aggregate_version"] is None
            else _exact_int(
                item["authority_aggregate_version"],
                "supplemental authority aggregate version",
            ),
        )


def _sorted_decision(
    value: tuple[DecisionLeadBinding, ...],
) -> tuple[DecisionLeadBinding, ...]:
    if type(value) is not tuple or not 1 <= len(value) <= MAX_DECISION_LEADS or any(
        type(v) is not DecisionLeadBinding for v in value
    ):
        raise WorkItemContractError("decision Leads must be a bounded typed tuple")
    try:
        result = tuple(sorted(value, key=lambda v: v.lead_id))
        if len({v.lead_id for v in result}) != len(result):
            raise WorkItemContractError("decision Leads duplicate")
        return result
    except WorkItemContractError:
        raise
    except Exception as exc:
        raise WorkItemContractError("decision Leads are invalid") from exc


@dataclass(frozen=True, slots=True)
class WorkItemPriorityBinding:
    work_identity: str
    work_version: str
    lane: str
    basis_digest: str
    basis_count: int
    selection_digest: str
    selection_bytes: bytes

    @classmethod
    def from_selection(cls, value: PrioritySelection) -> Self:
        if type(value) is not PrioritySelection:
            raise WorkItemContractError("Priority Selection must be typed")
        try:
            raw = value.canonical_bytes
            if (
                len(raw) > MAX_PRIORITY_INPUT_BYTES
                or len(value.basis_references) > MAX_PRIORITY_INPUT_REFERENCES
            ):
                raise WorkItemContractError("Priority Selection exceeds Work Item bound")
            replay = PrioritySelection.from_canonical_bytes(raw)
            basis = canonical_json_bytes(
                [reference.canonical_value() for reference in value.basis_references]
            )
        except WorkItemContractError:
            raise
        except Exception as exc:
            raise WorkItemContractError("Priority Selection is invalid") from exc
        if replay != value:
            raise WorkItemContractError("Priority Selection is not exact")
        return cls(
            value.work_identity,
            value.work_version,
            value.lane.value,
            digest_bytes(basis),
            len(value.basis_references),
            digest_bytes(raw),
            raw,
        )

    def __post_init__(self) -> None:
        _uuid(self.work_identity, "Priority work identity")
        _uuid(self.work_version, "Priority work version")
        _bounded_utf8(self.lane, 64, "Priority lane")
        try:
            PriorityLane(self.lane)
        except (TypeError, ValueError) as exc:
            raise WorkItemContractError("Priority lane is unsupported") from exc
        _digest(self.basis_digest, "Priority basis digest")
        _exact_int(
            self.basis_count,
            "Priority basis count",
            maximum=MAX_PRIORITY_INPUT_REFERENCES,
        )
        _digest(self.selection_digest, "Priority Selection digest")
        _retained_digest(
            self.selection_bytes,
            self.selection_digest,
            MAX_PRIORITY_INPUT_BYTES,
            "Priority Selection bytes",
        )
        try:
            selection = PrioritySelection.from_canonical_bytes(self.selection_bytes)
            basis = canonical_json_bytes(
                [
                    reference.canonical_value()
                    for reference in selection.basis_references
                ]
            )
        except Exception as exc:
            raise WorkItemContractError("Priority Selection bytes are invalid") from exc
        if (
            selection.work_identity != self.work_identity
            or selection.work_version != self.work_version
            or selection.lane.value != self.lane
            or len(selection.basis_references) != self.basis_count
            or len(selection.basis_references) > MAX_PRIORITY_INPUT_REFERENCES
            or digest_bytes(basis) != self.basis_digest
        ):
            raise WorkItemContractError("Priority Selection compact binding differs")

    def canonical_value(self) -> dict[str, object]:
        return {
            "work_identity": self.work_identity,
            "work_version": self.work_version,
            "lane": self.lane,
            "basis_digest": self.basis_digest,
            "basis_count": self.basis_count,
            "selection_digest": self.selection_digest,
            "selection": _decode(
                self.selection_bytes,
                maximum=MAX_PRIORITY_INPUT_BYTES,
                maximum_nodes=16_384,
            ),
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        item = _exact(
            value,
            {
                "work_identity",
                "work_version",
                "lane",
                "basis_digest",
                "basis_count",
                "selection_digest",
                "selection",
            },
            "Priority binding",
        )
        return cls(
            _text(item["work_identity"], "work_identity"),
            _text(item["work_version"], "work_version"),
            _text(item["lane"], "lane"),
            _text(item["basis_digest"], "basis_digest"),
            _exact_int(item["basis_count"], "Priority basis count"),
            _text(item["selection_digest"], "selection_digest"),
            _canonical(item["selection"], "Priority Selection"),
        )


@dataclass(frozen=True, slots=True)
class TriageWorkItem:
    work_item_id: str
    decision_leads: tuple[DecisionLeadBinding, ...]
    decision_scope_digest: str
    schema_identity: str = TRIAGE_WORK_ITEM

    @classmethod
    def create(cls, decision_leads: tuple[DecisionLeadBinding, ...]) -> Self:
        try:
            leads = _sorted_decision(decision_leads)
            identity = [
                {
                    "lead_id": v.lead_id,
                    "lead_digest": v.lead_digest,
                    "initial_disposition_id": v.disposition_id,
                    "initial_disposition_digest": v.disposition_digest,
                }
                for v in leads
            ]
            scope_digest = digest_bytes(canonical_json_bytes(identity))
            return cls(
                str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL, f"{TRIAGE_WORK_ITEM}|{scope_digest}"
                    )
                ),
                leads,
                scope_digest,
            )
        except WorkItemContractError:
            raise
        except Exception as exc:
            raise WorkItemContractError("Work Item inputs are invalid") from exc

    def __post_init__(self) -> None:
        _uuid(self.work_item_id, "work_item_id")
        _digest(self.decision_scope_digest, "decision_scope_digest")
        leads = _sorted_decision(self.decision_leads)
        identity = [
            {
                "lead_id": v.lead_id,
                "lead_digest": v.lead_digest,
                "initial_disposition_id": v.disposition_id,
                "initial_disposition_digest": v.disposition_digest,
            }
            for v in leads
        ]
        expected_scope = digest_bytes(canonical_json_bytes(identity))
        expected_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{TRIAGE_WORK_ITEM}|{expected_scope}")
        )
        if (
            self.schema_identity != TRIAGE_WORK_ITEM
            or self.decision_leads != leads
            or self.decision_scope_digest != expected_scope
            or self.work_item_id != expected_id
        ):
            raise WorkItemContractError(
                "Work Item identity differs from exact initial decision scope"
            )

    def canonical_value(self) -> dict[str, object]:
        return {
            "schema_identity": self.schema_identity,
            "work_item_id": self.work_item_id,
            "decision_scope_digest": self.decision_scope_digest,
            "decision_leads": [v.canonical_value() for v in self.decision_leads],
        }

    @property
    def canonical_bytes(self) -> bytes:
        try:
            canonical_value = self.canonical_value()
        except WorkItemContractError:
            raise
        except Exception as exc:
            raise WorkItemContractError("Work Item fields are invalid") from exc
        value = _canonical(canonical_value, "Work Item")
        if len(value) > MAX_WORK_ITEM_BYTES:
            raise WorkItemContractError("Work Item exceeds fixed bound")
        return value

    @property
    def canonical_digest(self) -> str:
        return digest_bytes(self.canonical_bytes)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> Self:
        item = _exact(
            _decode(raw, maximum=MAX_WORK_ITEM_BYTES, maximum_nodes=8_192),
            {
                "schema_identity",
                "work_item_id",
                "decision_scope_digest",
                "decision_leads",
            },
            "Work Item",
        )
        decision_values = _tuple(item["decision_leads"], "decision Leads")
        value = cls(
            _text(item["work_item_id"], "work_item_id"),
            tuple(DecisionLeadBinding.from_value(v) for v in decision_values),
            _text(item["decision_scope_digest"], "decision_scope_digest"),
            _text(item["schema_identity"], "schema_identity"),
        )  # type: ignore[arg-type]
        if value.canonical_bytes != raw:
            raise WorkItemContractError("Work Item bytes differ")
        return value


@dataclass(frozen=True, slots=True)
class TriageWorkItemVersion:
    version_id: str
    work_item_id: str
    ordinal: int
    previous_version_id: str | None
    decision_leads: tuple[DecisionLeadBinding, ...]
    context_leads: tuple[ContextLeadBinding, ...]
    retrieval: RetrievalInputBinding
    priority: WorkItemPriorityBinding
    watch: WatchConditionWorkItemBinding | None = None
    reentry_kind: ReentryKind | None = None
    supplemental_reentry: SupplementalDiscoveryReentry | None = None
    schema_identity: str = TRIAGE_WORK_ITEM_VERSION

    def __post_init__(self) -> None:
        _uuid(self.version_id, "version_id")
        _uuid(self.work_item_id, "work_item_id")
        _exact_int(self.ordinal, "Version ordinal")
        if not 1 <= self.ordinal <= 1_000_000 or (self.ordinal == 1) != (
            self.previous_version_id is None
        ):
            raise WorkItemContractError("Version ordinal and predecessor differ")
        expected = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{self.work_item_id}|{self.ordinal}")
        )
        if self.version_id != expected:
            raise WorkItemContractError(
                "Version identity differs from Work Item and ordinal"
            )
        expected_previous = (
            None
            if self.ordinal == 1
            else str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL, f"{self.work_item_id}|{self.ordinal - 1}"
                )
            )
        )
        if self.previous_version_id != expected_previous:
            raise WorkItemContractError("Version predecessor identity differs")
        decisions = _sorted_decision(self.decision_leads)
        if decisions != self.decision_leads:
            raise WorkItemContractError("decision Leads are not canonical")
        if type(self.context_leads) is not tuple or any(
            type(value) is not ContextLeadBinding for value in self.context_leads
        ):
            raise WorkItemContractError("context Leads must be a bounded typed tuple")
        try:
            contexts = tuple(sorted(self.context_leads, key=lambda v: v.lead_id))
        except Exception as exc:
            raise WorkItemContractError("context Leads are invalid") from exc
        if (
            contexts != self.context_leads
            or len(contexts) > MAX_CONTEXT_LEADS
            or len({v.lead_id for v in contexts}) != len(contexts)
        ):
            raise WorkItemContractError(
                "context Leads are not bounded canonical unique values"
            )
        if {v.lead_id for v in decisions}.intersection(v.lead_id for v in contexts):
            raise WorkItemContractError("decision and context Lead scopes overlap")
        if type(self.retrieval) is not RetrievalInputBinding:
            raise WorkItemContractError("retrieval binding must be typed")
        if type(self.priority) is not WorkItemPriorityBinding:
            raise WorkItemContractError("Priority binding must be typed")
        if self.watch is not None and type(self.watch) is not WatchConditionWorkItemBinding:
            raise WorkItemContractError("Watch binding must be typed")
        if (
            self.supplemental_reentry is not None
            and type(self.supplemental_reentry) is not SupplementalDiscoveryReentry
        ):
            raise WorkItemContractError("supplemental re-entry must be typed")
        try:
            self.priority.canonical_value()
            if self.watch is not None:
                self.watch.canonical_value()
            if self.supplemental_reentry is not None:
                self.supplemental_reentry.canonical_value()
        except Exception as exc:
            raise WorkItemContractError("Version nested bindings are invalid") from exc
        if (
            self.priority.work_identity != self.work_item_id
            or self.priority.work_version != self.version_id
        ):
            raise WorkItemContractError("Priority Selection back-reference differs")
        if self.reentry_kind is not None and (self.ordinal == 1 or self.watch is None):
            raise WorkItemContractError(
                "Watch condition re-entry requires a new successor Version"
            )
        if self.watch is not None:
            if type(self.reentry_kind) is not ReentryKind:
                raise WorkItemContractError("Watch re-entry kind must be typed")
            matching = [
                lead
                for lead in self.decision_leads
                if lead.lead_id == self.watch.lead_id
            ]
            if len(matching) != 1:
                raise WorkItemContractError("Watch does not belong to a decision Lead")
            target = matching[0]
            if (
                target.disposition_ordinal != self.watch.source_disposition_ordinal + 1
                or target.previous_disposition_id != self.watch.source_disposition_id
            ):
                raise WorkItemContractError(
                    "Watch re-entry queue is not the immediate successor"
                )
            if self.reentry_kind.value not in self.watch.allowed_reentry_kinds:
                raise WorkItemContractError(
                    "Watch re-entry kind differs from its condition"
                )
            if self.watch.observable_transition:
                raise WorkItemContractError(
                    "observable Watch transition requires a new Lead lineage"
                )
        if self.supplemental_reentry is not None:
            proof = self.supplemental_reentry
            if (
                self.ordinal != 1
                or self.previous_version_id is not None
                or proof.target_work_item_id != self.work_item_id
                or proof.target_version_id != self.version_id
                or proof.lead_id not in {lead.lead_id for lead in self.decision_leads}
                or proof.queued_disposition_id
                not in {lead.disposition_id for lead in self.decision_leads}
            ):
                raise WorkItemContractError(
                    "supplemental target must be a new ordinal-one Version"
                )
        if self.schema_identity != TRIAGE_WORK_ITEM_VERSION:
            raise WorkItemContractError("Version schema identity differs")
        try:
            raw = _canonical(self.canonical_value(), "Version")
        except WorkItemContractError:
            raise
        except Exception as exc:
            raise WorkItemContractError("Version fields are invalid") from exc
        if len(raw) > MAX_VERSION_BYTES:
            raise WorkItemContractError("Version exceeds fixed bound")
        _decode(raw)

    @classmethod
    def create(
        cls,
        *,
        work_item_id: str,
        ordinal: int,
        previous_version_id: str | None,
        decision_leads: tuple[DecisionLeadBinding, ...],
        context_leads: tuple[ContextLeadBinding, ...],
        retrieval: RetrievalInputBinding,
        priority: PrioritySelection,
        watch: WatchConditionWorkItemBinding | None = None,
        reentry_kind: ReentryKind | None = None,
        supplemental_reentry: SupplementalDiscoveryReentry | None = None,
    ) -> Self:
        if type(decision_leads) is not tuple or type(context_leads) is not tuple:
            raise WorkItemContractError("Lead bindings must be typed tuples")
        try:
            version_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{work_item_id}|{ordinal}")
            )
            if type(priority) is not PrioritySelection:
                raise WorkItemContractError("Priority Selection must be typed")
            priority_binding = WorkItemPriorityBinding.from_selection(priority)
            return cls(
                version_id,
                work_item_id,
                ordinal,
                previous_version_id,
                tuple(sorted(decision_leads, key=lambda v: v.lead_id)),
                tuple(sorted(context_leads, key=lambda v: v.lead_id)),
                retrieval,
                priority_binding,
                watch,
                reentry_kind,
                supplemental_reentry,
            )
        except WorkItemContractError:
            raise
        except Exception as exc:
            raise WorkItemContractError("Version inputs are invalid") from exc

    def canonical_value(self) -> dict[str, object]:
        return {
            "schema_identity": self.schema_identity,
            "version_id": self.version_id,
            "work_item_id": self.work_item_id,
            "ordinal": self.ordinal,
            "previous_version_id": self.previous_version_id,
            "decision_leads": [v.canonical_value() for v in self.decision_leads],
            "context_leads": [v.canonical_value() for v in self.context_leads],
            "retrieval": self.retrieval.canonical_value(),
            "priority": self.priority.canonical_value(),
            "watch": None if self.watch is None else self.watch.canonical_value(),
            "reentry_kind": None
            if self.reentry_kind is None
            else self.reentry_kind.value,
            "supplemental_reentry": None
            if self.supplemental_reentry is None
            else self.supplemental_reentry.canonical_value(),
        }

    @property
    def canonical_bytes(self) -> bytes:
        try:
            canonical_value = self.canonical_value()
        except WorkItemContractError:
            raise
        except Exception as exc:
            raise WorkItemContractError("Version fields are invalid") from exc
        value = _canonical(canonical_value, "Version")
        if len(value) > MAX_VERSION_BYTES:
            raise WorkItemContractError("Version exceeds fixed bound")
        return value

    @property
    def canonical_digest(self) -> str:
        return digest_bytes(self.canonical_bytes)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> Self:
        item = _exact(
            _decode(raw),
            {
                "schema_identity",
                "version_id",
                "work_item_id",
                "ordinal",
                "previous_version_id",
                "decision_leads",
                "context_leads",
                "retrieval",
                "priority",
                "watch",
                "reentry_kind",
                "supplemental_reentry",
            },
            "Work Item Version",
        )
        decision_values = _tuple(item["decision_leads"], "decision Leads")
        context_values = _tuple(item["context_leads"], "context Leads")
        if type(item["retrieval"]) is not dict:
            raise WorkItemContractError("retrieval binding must be an object")
        if type(item["priority"]) is not dict:
            raise WorkItemContractError("Priority Selection must be an object")
        reentry: ReentryKind | None = None
        if item["reentry_kind"] is not None:
            try:
                reentry = ReentryKind(item["reentry_kind"])
            except (TypeError, ValueError) as exc:
                raise WorkItemContractError("re-entry kind is unsupported") from exc
        value = cls(
            _text(item["version_id"], "version_id"),
            _text(item["work_item_id"], "work_item_id"),
            _exact_int(item["ordinal"], "version ordinal"),
            None
            if item["previous_version_id"] is None
            else _text(item["previous_version_id"], "previous_version_id"),
            tuple(DecisionLeadBinding.from_value(v) for v in decision_values),
            tuple(ContextLeadBinding.from_value(v) for v in context_values),
            RetrievalInputBinding.from_value(item["retrieval"]),
            WorkItemPriorityBinding.from_value(item["priority"]),
            None
            if item["watch"] is None
            else WatchConditionWorkItemBinding.from_value(item["watch"]),
            reentry,
            None
            if item["supplemental_reentry"] is None
            else SupplementalDiscoveryReentry.from_value(item["supplemental_reentry"]),
            _text(item["schema_identity"], "schema_identity"),
        )  # type: ignore[arg-type]
        if value.canonical_bytes != raw:
            raise WorkItemContractError("Version bytes differ")
        return value


@dataclass(frozen=True, slots=True)
class WorkItemCurrentAssessment:
    work_item_id: str
    version_id: str
    version_digest: str
    current: bool
    usable: bool
    stale_reasons: tuple[str, ...]


class TriageWorkItemStore:
    """Transactional append-only Work Item store over the authority database."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        retrieval_authority: RetrievalContextAuthority | None = None,
        *,
        _validate_on_open: bool = True,
        _validate_retained_event: Callable[[str], None] | None = None,
    ) -> None:
        if type(connection) is not sqlite3.Connection or connection.in_transaction:
            raise WorkItemContractError("store requires an idle sqlite3 connection")
        connection.execute("PRAGMA foreign_keys=ON")
        if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise WorkItemContractError("foreign keys must be enabled")
        self._connection = connection
        self._retrieval_authority = retrieval_authority
        self._current_state_only = not _validate_on_open
        self._validate_retained_event = _validate_retained_event
        if retrieval_authority is not None:
            retrieval_authority.attach(connection)
        connection.execute("BEGIN IMMEDIATE")
        try:
            if _validate_on_open:
                self._verify_integrity()
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def _begin(self) -> None:
        if self._connection.in_transaction:
            raise WorkItemContractError("connection has an active transaction")
        self._connection.execute("BEGIN IMMEDIATE")

    def _rollback(self) -> None:
        if self._connection.in_transaction:
            self._connection.execute("ROLLBACK")

    def create_or_replay(
        self, item: TriageWorkItem, version: TriageWorkItemVersion
    ) -> TriageWorkItemVersion:
        if type(item) is not TriageWorkItem or type(version) is not TriageWorkItemVersion:
            raise WorkItemContractError("create requires typed Work Item records")
        try:
            fresh_item = TriageWorkItem.from_canonical_bytes(item.canonical_bytes)
            fresh_version = TriageWorkItemVersion.from_canonical_bytes(
                version.canonical_bytes
            )
        except WorkItemContractError:
            raise
        except Exception as exc:
            raise WorkItemContractError("create Work Item records are invalid") from exc
        if fresh_item != item or fresh_version != version:
            raise WorkItemContractError("create requires exact base Work Item records")
        if (
            version.work_item_id != item.work_item_id
            or version.ordinal != 1
            or version.decision_leads != item.decision_leads
        ):
            raise WorkItemContractError("initial Version differs from Work Item")
        self._begin()
        try:
            existing_item = self._connection.execute(
                "SELECT canonical_bytes FROM triage_work_items WHERE work_item_id=?",
                (item.work_item_id,),
            ).fetchone()
            existing_version = self._connection.execute(
                "SELECT canonical_bytes FROM triage_work_item_versions WHERE version_id=?",
                (version.version_id,),
            ).fetchone()
            if existing_item is not None or existing_version is not None:
                if (
                    existing_item is None
                    or existing_version is None
                    or bytes(existing_item[0]) != item.canonical_bytes
                    or bytes(existing_version[0]) != version.canonical_bytes
                ):
                    raise WorkItemContractError("Work Item replay diverges")
                self._verify_item_chain(item.work_item_id)
                self._connection.execute("COMMIT")
                return version
            self._require_upstream(version)
            self._reject_overlap(item, version)
            self._reject_reused_causality(version)
            self._connection.execute(
                "INSERT OR IGNORE INTO triage_work_items VALUES(?,?,?,?,?,?,?)",
                (
                    item.work_item_id,
                    TRIAGE_WORK_ITEM,
                    item.decision_scope_digest,
                    len(item.decision_leads),
                    item.canonical_bytes,
                    item.canonical_digest,
                    self._recorded_at(),
                ),
            )
            row = self._connection.execute(
                "SELECT canonical_bytes FROM triage_work_items WHERE work_item_id=?",
                (item.work_item_id,),
            ).fetchone()
            if row is None or bytes(row[0]) != item.canonical_bytes:
                raise WorkItemContractError("Work Item replay diverges")
            self._insert_version(version)
            self._connection.execute(
                "INSERT OR IGNORE INTO triage_work_item_heads VALUES(?,?,?,?,?)",
                (
                    item.work_item_id,
                    version.version_id,
                    1,
                    version.canonical_digest,
                    self._recorded_at(),
                ),
            )
            head = self._head(item.work_item_id)
            if head != (version.version_id, 1, version.canonical_digest):
                raise WorkItemContractError("Work Item create replay diverges")
            self._connection.execute("COMMIT")
            return version
        except Exception:
            self._rollback()
            raise

    def append_version(
        self,
        expected_head_id: str,
        expected_head_digest: str,
        version: TriageWorkItemVersion,
    ) -> TriageWorkItemVersion:
        if type(version) is not TriageWorkItemVersion:
            raise WorkItemContractError("append requires a typed Work Item Version")
        try:
            fresh_version = TriageWorkItemVersion.from_canonical_bytes(
                version.canonical_bytes
            )
        except WorkItemContractError:
            raise
        except Exception as exc:
            raise WorkItemContractError("append Work Item Version is invalid") from exc
        if fresh_version != version:
            raise WorkItemContractError("append requires an exact base Work Item Version")
        _uuid(expected_head_id, "expected head id")
        _digest(expected_head_digest, "expected head digest")
        self._begin()
        try:
            existing = self._connection.execute(
                "SELECT canonical_bytes FROM triage_work_item_versions WHERE version_id=?",
                (version.version_id,),
            ).fetchone()
            if existing is not None:
                if (
                    bytes(existing[0]) != version.canonical_bytes
                    or version.previous_version_id != expected_head_id
                    or self.load_version(expected_head_id).canonical_digest
                    != expected_head_digest
                ):
                    raise WorkItemContractError("Version replay diverges")
                self._verify_item_chain(version.work_item_id)
                self._connection.execute("COMMIT")
                return version
            head = self._head(version.work_item_id)
            if head[0] != expected_head_id or head[2] != expected_head_digest:
                raise WorkItemContractError("stale expected Work Item head")
            if version.ordinal != head[1] + 1 or version.previous_version_id != head[0]:
                raise WorkItemContractError("Version is not the immediate successor")
            predecessor = self.load_version(head[0])
            item = self._load_item(version.work_item_id)
            if tuple(v.stable_lead_value() for v in version.decision_leads) != tuple(
                v.stable_lead_value() for v in item.decision_leads
            ):
                raise WorkItemContractError("stable decision Lead identity changed")
            self._require_disposition_transition(predecessor, version)
            self._require_upstream(version)
            self._reject_overlap(item, version)
            self._reject_reused_causality(version)
            self._insert_version(version)
            self._connection.execute(
                "UPDATE triage_work_item_heads SET current_version_id=?,current_ordinal=?,current_version_digest=?,updated_at=? WHERE work_item_id=?",
                (
                    version.version_id,
                    version.ordinal,
                    version.canonical_digest,
                    self._recorded_at(),
                    version.work_item_id,
                ),
            )
            self._connection.execute("COMMIT")
            return version
        except Exception:
            self._rollback()
            raise

    def _require_disposition_transition(
        self,
        predecessor: TriageWorkItemVersion,
        proposed: TriageWorkItemVersion,
    ) -> None:
        changed = tuple(
            (before, after)
            for before, after in zip(
                predecessor.decision_leads,
                proposed.decision_leads,
                strict=True,
            )
            if before != after
        )
        if not changed:
            return
        if len(changed) != 1:
            raise WorkItemContractError(
                "disposition change requires exactly one Lead and typed Watch proof"
            )
        before, after = changed[0]
        if before.gate_decision_id != after.gate_decision_id:
            if (
                proposed.watch is not None
                or after.lead_id != before.lead_id
                or after.disposition_outcome
                != LeadDispositionOutcome.QUEUED_FOR_TRIAGE.value
                or after.disposition_ordinal != before.disposition_ordinal + 1
                or after.previous_disposition_id != before.disposition_id
            ):
                raise WorkItemContractError(
                    "Gate re-promotion requires an exact immediate queued disposition"
                )
            return
        if proposed.watch is None:
            raise WorkItemContractError(
                "disposition change requires exactly one Lead and typed Watch proof"
            )
        watch = proposed.watch
        if (
            watch.lead_id != before.lead_id
            or after.lead_id != before.lead_id
            or watch.source_disposition_ordinal
            != before.disposition_ordinal + 1
            or watch.source_previous_disposition_id != before.disposition_id
            or after.disposition_outcome
            != LeadDispositionOutcome.QUEUED_FOR_TRIAGE.value
            or after.disposition_ordinal != watch.source_disposition_ordinal + 1
            or after.previous_disposition_id != watch.source_disposition_id
            or not self._watch_retained(watch)
        ):
            raise WorkItemContractError(
                "disposition change requires typed Watch proof; "
                "watch is not the exact immediate successor"
            )

    def load_version(self, version_id: str) -> TriageWorkItemVersion:
        _uuid(version_id, "version_id")
        row = self._connection.execute(
            "SELECT canonical_bytes,canonical_digest FROM triage_work_item_versions WHERE version_id=?",
            (version_id,),
        ).fetchone()
        if row is None:
            raise WorkItemContractError("unknown Work Item Version")
        value = TriageWorkItemVersion.from_canonical_bytes(bytes(row[0]))
        if value.canonical_digest != row[1]:
            raise WorkItemContractError("Version digest differs")
        return value

    def current_version(self, work_item_id: str) -> TriageWorkItemVersion:
        _uuid(work_item_id, "work_item_id")
        if self._current_state_only:
            self._verify_item_chain(work_item_id)
        row = self._connection.execute(
            "SELECT v.canonical_bytes,v.canonical_digest,h.current_version_digest "
            "FROM triage_work_item_heads h JOIN triage_work_item_versions v "
            "ON v.version_id=h.current_version_id AND v.work_item_id=h.work_item_id "
            "WHERE h.work_item_id=?",
            (work_item_id,),
        ).fetchone()
        if row is None:
            raise WorkItemContractError("unknown Work Item")
        version = TriageWorkItemVersion.from_canonical_bytes(bytes(row[0]))
        if version.canonical_digest != row[1] or row[1] != row[2]:
            raise WorkItemContractError("Work Item head digest differs")
        return version

    def assess_current(self, work_item_id: str) -> WorkItemCurrentAssessment:
        self._begin()
        try:
            head = self._head(work_item_id)
            version = self.load_version(head[0])
            reasons = self._upstream_reasons(version)
            current = not reasons
            if not version.retrieval.usable:
                reasons.append("retrieval_not_complete")
            result = WorkItemCurrentAssessment(
                work_item_id,
                version.version_id,
                version.canonical_digest,
                current,
                current and version.retrieval.usable,
                tuple(sorted(set(reasons))),
            )
            self._connection.execute("COMMIT")
            return result
        except Exception:
            self._rollback()
            raise

    def require_usable_current(self, work_item_id: str) -> TriageWorkItemVersion:
        self._begin()
        try:
            version = self.require_usable_current_in_transaction(work_item_id)
            self._connection.execute("COMMIT")
            return version
        except Exception:
            self._rollback()
            raise

    def require_usable_current_in_transaction(
        self, work_item_id: str
    ) -> TriageWorkItemVersion:
        """Recheck for a trusted composition root holding ``BEGIN IMMEDIATE``."""
        if not self._connection.in_transaction:
            raise WorkItemContractError(
                "transaction-aware Work Item use requires an active transaction"
            )
        if self._current_state_only:
            self._verify_item_chain(work_item_id)
        head = self._head(work_item_id)
        version = self.load_version(head[0])
        if version.canonical_digest != head[2]:
            raise WorkItemContractError("Work Item head digest differs")
        reasons = self._upstream_reasons(version)
        if not version.retrieval.usable:
            reasons.append("retrieval_not_complete")
        if reasons:
            raise WorkItemStaleError(";".join(sorted(set(reasons))))
        return version

    def _insert_version(self, v: TriageWorkItemVersion) -> None:
        item = self._load_item(v.work_item_id)
        watch_condition_id, source_lead_disposition_id = self._causal_ids(v)
        self._connection.execute(
            "INSERT OR IGNORE INTO triage_work_item_versions("
            "version_id,schema_identity,work_item_id,ordinal,previous_version_id,"
            "decision_scope_digest,retrieval_outcome,watch_condition_id,"
            "source_lead_disposition_id,canonical_bytes,canonical_digest,recorded_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                v.version_id,
                TRIAGE_WORK_ITEM_VERSION,
                v.work_item_id,
                v.ordinal,
                v.previous_version_id,
                item.decision_scope_digest,
                v.retrieval.outcome or v.retrieval.state.value,
                watch_condition_id,
                source_lead_disposition_id,
                v.canonical_bytes,
                v.canonical_digest,
                self._recorded_at(),
            ),
        )
        row = self._connection.execute(
            "SELECT canonical_bytes FROM triage_work_item_versions WHERE version_id=?",
            (v.version_id,),
        ).fetchone()
        if row is None or bytes(row[0]) != v.canonical_bytes:
            raise WorkItemContractError("Version replay diverges")

    def _head(self, work_item_id: str) -> tuple[str, int, str]:
        _uuid(work_item_id, "work_item_id")
        row = self._connection.execute(
            "SELECT current_version_id,current_ordinal,current_version_digest FROM triage_work_item_heads WHERE work_item_id=?",
            (work_item_id,),
        ).fetchone()
        if row is None:
            raise WorkItemContractError("unknown Work Item")
        return str(row[0]), int(row[1]), str(row[2])

    def _recorded_at(self) -> str:
        return str(
            self._connection.execute(
                "SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now')"
            ).fetchone()[0]
        )

    def _load_item(self, work_item_id: str) -> TriageWorkItem:
        row = self._connection.execute(
            "SELECT canonical_bytes,canonical_digest FROM triage_work_items WHERE work_item_id=?",
            (work_item_id,),
        ).fetchone()
        if row is None:
            raise WorkItemContractError("unknown Work Item")
        value = TriageWorkItem.from_canonical_bytes(bytes(row[0]))
        if value.canonical_digest != row[1]:
            raise WorkItemContractError("Work Item digest differs")
        return value

    def _lead_retained(
        self, lead: DecisionLeadBinding | ContextLeadBinding
    ) -> bool:
        row = self._connection.execute(
            "SELECT canonical_digest,canonical_bytes,authority_event_id,"
            "authority_aggregate_version FROM news_leads WHERE lead_id=?",
            (lead.lead_id,),
        ).fetchone()
        if row is None:
            return False
        try:
            request = _retained_request(
                bytes(row[1]), row[0], "News Lead"
            )
        except WorkItemContractError:
            return False
        return (
            row[0] == lead.lead_digest
            and row[2] == lead.lead_event_id
            and row[3] == lead.lead_aggregate_version
            and request.get("lead_id") == lead.lead_id
            and request.get("definition_id") == lead.definition_id
            and request.get("definition_version_id") == lead.definition_version_id
            and (
                type(lead) is DecisionLeadBinding
                or request.get("promoting_gate_decision_id") == lead.gate_decision_id
            )
        )

    def _disposition_retained(self, lead: DecisionLeadBinding) -> bool:
        row = self._connection.execute(
            "SELECT canonical_digest,outcome,canonical_bytes,authority_event_id,"
            "authority_aggregate_version,decision_ordinal,previous_decision_id,lead_id "
            "FROM lead_disposition_decisions WHERE decision_id=?",
            (lead.disposition_id,),
        ).fetchone()
        if row is None:
            return False
        try:
            request = _retained_request(
                bytes(row[2]),
                row[0],
                "Lead disposition",
            )
        except WorkItemContractError:
            return False
        return (
            row[0] == lead.disposition_digest
            and row[1] == lead.disposition_outcome
            and row[3] == lead.disposition_event_id
            and row[4] == lead.disposition_aggregate_version
            and row[5] == lead.disposition_ordinal
            and row[6] == lead.previous_disposition_id
            and row[7] == lead.lead_id
            and request.get("decision_id") == lead.disposition_id
            and request.get("lead_id") == lead.lead_id
            and request.get("gate_decision_id") == lead.gate_decision_id
            and request.get("decision_ordinal") == lead.disposition_ordinal
            and request.get("previous_decision_id")
            == lead.previous_disposition_id
            and request.get("outcome") == lead.disposition_outcome
        )

    def _watch_retained(self, binding: WatchConditionWorkItemBinding) -> bool:
        watch = self._connection.execute(
            "SELECT canonical_digest,canonical_bytes,lead_id,authority_event_id,"
            "authority_aggregate_version FROM discovery_watch_conditions "
            "WHERE watch_condition_id=?",
            (binding.watch_condition_id,),
        ).fetchone()
        source = self._connection.execute(
            "SELECT canonical_digest,canonical_bytes,lead_id,decision_ordinal,"
            "previous_decision_id,outcome,watch_condition_id,authority_event_id,"
            "authority_aggregate_version FROM lead_disposition_decisions "
            "WHERE decision_id=?",
            (binding.source_disposition_id,),
        ).fetchone()
        if watch is None or source is None:
            return False
        try:
            watch_request = _retained_request(
                bytes(watch[1]), watch[0], "Watch Condition"
            )
            source_request = _retained_request(
                bytes(source[1]),
                source[0],
                "Watch source disposition",
            )
        except WorkItemContractError:
            return False
        try:
            resume_kinds = watch_request["resume_transition_kinds"]
            expected_occurrence = watch_request["expected_occurrence"]
            corroborating_lead = watch_request["corroborating_lead_id"]
            review_at = watch_request["review_at"]
            expires_at = watch_request["expires_at"]
            operator_condition = watch_request["operator_review_condition"]
            if (
                type(resume_kinds) is not list
                or any(type(value) is not str for value in resume_kinds)
                or expected_occurrence is not None
                and type(expected_occurrence) is not str
                or corroborating_lead is not None
                and type(corroborating_lead) is not str
                or review_at is not None
                and type(review_at) is not str
                or expires_at is not None
                and type(expires_at) is not str
                or operator_condition is not None
                and type(operator_condition) is not str
            ):
                return False
            derived_allowed: list[str] = []
            if expected_occurrence and review_at is not None:
                derived_allowed.append(ReentryKind.DEADLINE.value)
            if review_at is not None:
                derived_allowed.append(ReentryKind.REVIEW.value)
            if expires_at is not None:
                derived_allowed.append(ReentryKind.EXPIRY.value)
            if operator_condition:
                derived_allowed.append(ReentryKind.OPERATOR_CONDITION.value)
            allowed_reentry_kinds = tuple(sorted(set(derived_allowed)))
            observable_transition = bool(resume_kinds or corroborating_lead)
        except (KeyError, TypeError, ValueError):
            return False
        return (
            watch[0] == binding.watch_condition_digest
            and watch[2] == binding.lead_id
            and watch[3] == binding.watch_event_id
            and watch[4] == binding.watch_aggregate_version
            and watch_request.get("watch_condition_id")
            == binding.watch_condition_id
            and watch_request.get("lead_id") == binding.lead_id
            and allowed_reentry_kinds == binding.allowed_reentry_kinds
            and observable_transition == binding.observable_transition
            and source[0] == binding.source_disposition_digest
            and source[2] == binding.lead_id
            and source[3] == binding.source_disposition_ordinal
            and source[4] == binding.source_previous_disposition_id
            and source[5] == LeadDispositionOutcome.WATCH_DEFER.value
            and source[6] == binding.watch_condition_id
            and source[7] == binding.source_disposition_event_id
            and source[8] == binding.source_disposition_aggregate_version
            and source_request.get("decision_id") == binding.source_disposition_id
            and source_request.get("lead_id") == binding.lead_id
            and source_request.get("decision_ordinal")
            == binding.source_disposition_ordinal
            and source_request.get("previous_decision_id")
            == binding.source_previous_disposition_id
            and source_request.get("outcome")
            == LeadDispositionOutcome.WATCH_DEFER.value
            and source_request.get("watch_condition_id")
            == binding.watch_condition_id
            and type(source_request.get("next_action")) is dict
            and source_request["next_action"].get("kind") == "RESUME_ON_WATCH"
        )

    def _watch_occurrence_reached(
        self,
        binding: WatchConditionWorkItemBinding,
        reentry_kind: ReentryKind | None,
    ) -> bool:
        if reentry_kind is ReentryKind.OPERATOR_CONDITION or reentry_kind is None:
            return False
        row = self._connection.execute(
            "SELECT canonical_digest,canonical_bytes "
            "FROM discovery_watch_conditions WHERE watch_condition_id=?",
            (binding.watch_condition_id,),
        ).fetchone()
        if row is None or row[0] != binding.watch_condition_digest:
            return False
        try:
            request = _retained_request(
                bytes(row[1]), row[0], "Watch Condition occurrence"
            )
            field = (
                "expires_at"
                if reentry_kind is ReentryKind.EXPIRY
                else "review_at"
            )
            timestamp = request[field]
            if type(timestamp) is not str:
                return False
            parsed = UtcTimestamp.parse(timestamp)
            if parsed.to_text() != timestamp:
                return False
        except (KeyError, TypeError, ValueError):
            return False
        reached = self._connection.execute(
            "SELECT julianday(?) <= julianday('now')",
            (timestamp,),
        ).fetchone()
        return reached is not None and reached[0] == 1

    def _upstream_reasons(self, v: TriageWorkItemVersion) -> list[str]:
        reasons: list[str] = []
        for lead in v.decision_leads:
            if not self._lead_retained(lead):
                reasons.append(f"lead:{lead.lead_id}")
            gate = self._connection.execute(
                "SELECT current_decision_id FROM discovery_gate_decision_heads WHERE signal_id=(SELECT signal_id FROM news_leads WHERE lead_id=?)",
                (lead.lead_id,),
            ).fetchone()
            if gate is None or gate[0] != lead.gate_decision_id:
                reasons.append(f"gate:{lead.lead_id}")
            source = self._connection.execute(
                "SELECT current_version_id FROM source_definition_version_heads WHERE definition_id=?",
                (lead.definition_id,),
            ).fetchone()
            if source is None or source[0] != lead.definition_version_id:
                reasons.append(f"source:{lead.lead_id}")
            disp = self._connection.execute(
                "SELECT h.current_decision_id,d.canonical_digest,d.outcome,"
                "d.canonical_bytes,d.authority_event_id,d.authority_aggregate_version,"
                "d.decision_ordinal,d.previous_decision_id,d.lead_id "
                "FROM lead_disposition_heads h JOIN lead_disposition_decisions d "
                "ON d.decision_id=h.current_decision_id WHERE h.lead_id=?",
                (lead.lead_id,),
            ).fetchone()
            retained_disposition = self._disposition_retained(lead)
            if (
                disp is None
                or disp[0] != lead.disposition_id
                or not retained_disposition
            ):
                reasons.append(f"disposition:{lead.lead_id}")
        for lead in v.context_leads:
            if not self._lead_retained(lead):
                reasons.append(f"context:{lead.lead_id}")
            gate = self._connection.execute(
                "SELECT current_decision_id FROM discovery_gate_decision_heads "
                "WHERE signal_id=(SELECT signal_id FROM news_leads WHERE lead_id=?)",
                (lead.lead_id,),
            ).fetchone()
            if gate is None or gate[0] != lead.gate_decision_id:
                reasons.append(f"gate:{lead.lead_id}")
            source = self._connection.execute(
                "SELECT current_version_id FROM source_definition_version_heads "
                "WHERE definition_id=?",
                (lead.definition_id,),
            ).fetchone()
            if source is None or source[0] != lead.definition_version_id:
                reasons.append(f"source:{lead.lead_id}")
        if v.watch is not None:
            if not self._watch_retained(v.watch):
                reasons.append("watch")
            elif not self._watch_occurrence_reached(v.watch, v.reentry_kind):
                reasons.append("watch occurrence")
        if self._retrieval_authority is not None:
            if not self._retrieval_authority.verify_retained_currentness(
                self._connection, v.retrieval
            ):
                reasons.append("retrieval_authority_differs")
        elif v.retrieval.state is RetrievalBindingState.RECEIPT:
            reasons.append("retrieval_authority_unavailable")
        if v.supplemental_reentry is not None:
            reasons.extend(self._supplemental_reasons(v.supplemental_reentry))
        return reasons

    def _supplemental_reasons(self, proof: SupplementalDiscoveryReentry) -> list[str]:
        reasons: list[str] = []
        try:
            source = self.load_version(proof.source_version_id)
        except WorkItemContractError:
            return ["supplemental_source_version"]
        if (
            source.work_item_id != proof.source_work_item_id
            or source.canonical_digest != proof.source_version_digest
        ):
            reasons.append("supplemental_source_version")
        reasons.append("supplemental_authority_unavailable_v18")
        table_map = {
            "CHECK_REQUEST": ("check_requests", "request_id", "trigger_id"),
            "CHECK_OUTCOME": ("check_outcomes", "outcome_id", "request_id"),
            "SIGNAL": ("discovery_signals", "signal_id", "check_outcome_id"),
            "GATE": ("discovery_gate_decisions", "decision_id", "signal_id"),
            "LEAD": ("news_leads", "lead_id", "promoting_gate_decision_id"),
            "QUEUED_DISPOSITION": (
                "lead_disposition_decisions",
                "decision_id",
                "lead_id",
            ),
        }
        for binding in proof.lineage_bindings:
            if binding.kind == "TRIGGER":
                continue
            table, column, parent_column = table_map[binding.kind]
            row = self._connection.execute(
                f"SELECT canonical_digest,canonical_bytes,authority_event_id,"
                f"authority_aggregate_version,{parent_column} FROM {table} "
                f"WHERE {column}=?",
                (binding.identifier,),
            ).fetchone()
            if (
                row is None
                or row[0] != binding.digest
                or digest_bytes(bytes(row[1])) != binding.digest
                or row[2] != binding.authority_event_id
                or row[3] != binding.authority_aggregate_version
                or row[4] != binding.parent_id
            ):
                reasons.append(f"supplemental_{binding.kind.lower()}")
            if binding.kind == "QUEUED_DISPOSITION":
                outcome = self._connection.execute(
                    "SELECT outcome FROM lead_disposition_decisions "
                    "WHERE decision_id=?",
                    (binding.identifier,),
                ).fetchone()
                if outcome != (LeadDispositionOutcome.QUEUED_FOR_TRIAGE.value,):
                    reasons.append("supplemental_queued_disposition")
        return reasons

    def _require_upstream(self, v: TriageWorkItemVersion) -> None:
        reasons = self._upstream_reasons(v)
        if reasons:
            raise WorkItemContractError(
                "upstream authority differs: " + ",".join(reasons)
            )

    def _is_active_usable(self, version: TriageWorkItemVersion) -> bool:
        return version.retrieval.usable and not self._upstream_reasons(version)

    def _reject_overlap(
        self, item: TriageWorkItem, candidate: TriageWorkItemVersion
    ) -> None:
        if not self._is_active_usable(candidate):
            return
        scope = tuple(v.lead_id for v in item.decision_leads)
        placeholders = ",".join("?" for _ in scope)
        rows = self._connection.execute(
            "SELECT DISTINCT h.work_item_id,v.canonical_bytes "
            "FROM triage_work_item_heads h "
            "JOIN triage_work_item_versions v ON v.version_id=h.current_version_id "
            "JOIN triage_work_items i ON i.work_item_id=h.work_item_id "
            "JOIN json_each(CAST(i.canonical_bytes AS TEXT),'$.decision_leads') j "
            f"WHERE h.work_item_id!=? AND json_extract(j.value,'$.lead_id') IN ({placeholders})",
            (item.work_item_id, *scope),
        ).fetchall()
        for _work_item_id, raw_version in rows:
            other_version = TriageWorkItemVersion.from_canonical_bytes(
                bytes(raw_version)
            )
            if self._is_active_usable(other_version):
                raise WorkItemContractError("active decision Lead scopes overlap")

    @staticmethod
    def _causal_ids(
        version: TriageWorkItemVersion,
    ) -> tuple[str | None, str | None]:
        return (
            None if version.watch is None else version.watch.watch_condition_id,
            None
            if version.supplemental_reentry is None
            else version.supplemental_reentry.source_lead_disposition_id,
        )

    def _reject_reused_causality(self, version: TriageWorkItemVersion) -> None:
        watch, supplemental = self._causal_ids(version)
        if watch is not None and self._connection.execute(
            "SELECT 1 FROM triage_work_item_versions WHERE watch_condition_id=?",
            (watch,),
        ).fetchone():
            raise WorkItemContractError("Watch causality was already claimed")
        if supplemental is not None and self._connection.execute(
            "SELECT 1 FROM triage_work_item_versions "
            "WHERE source_lead_disposition_id=?",
            (supplemental,),
        ).fetchone():
            raise WorkItemContractError("supplemental causality was already claimed")

    def _verify_item_chain(self, work_item_id: str) -> None:
        item_row = self._connection.execute(
            "SELECT decision_scope_digest,decision_lead_count,canonical_bytes,"
            "canonical_digest FROM triage_work_items WHERE work_item_id=?",
            (work_item_id,),
        ).fetchone()
        if item_row is None:
            raise WorkItemContractError("Work Item chain or head is absent")
        item = TriageWorkItem.from_canonical_bytes(bytes(item_row[2]))
        if (
            item.work_item_id != work_item_id
            or item.decision_scope_digest != item_row[0]
            or len(item.decision_leads) != item_row[1]
            or item.canonical_digest != item_row[3]
        ):
            raise WorkItemContractError("Work Item retained bytes differ")
        rows = self._connection.execute(
            "SELECT version_id,ordinal,previous_version_id,decision_scope_digest,"
            "retrieval_outcome,watch_condition_id,source_lead_disposition_id,"
            "canonical_bytes,canonical_digest "
            "FROM triage_work_item_versions WHERE work_item_id=? ORDER BY ordinal",
            (work_item_id,),
        ).fetchall()
        head = self._connection.execute(
            "SELECT current_version_id,current_ordinal,current_version_digest "
            "FROM triage_work_item_heads WHERE work_item_id=?",
            (work_item_id,),
        ).fetchone()
        if not rows or head is None:
            raise WorkItemContractError("Work Item chain or head is absent")
        previous: str | None = None
        for expected_ordinal, row in enumerate(rows, 1):
            version = TriageWorkItemVersion.from_canonical_bytes(bytes(row[7]))
            if (
                version.version_id != row[0]
                or version.work_item_id != work_item_id
                or version.ordinal != row[1]
                or version.previous_version_id != row[2]
                or item.decision_scope_digest != row[3]
                or (version.retrieval.outcome or version.retrieval.state.value)
                != row[4]
                or self._causal_ids(version) != (row[5], row[6])
                or version.canonical_digest != row[8]
                or tuple(
                    lead.stable_lead_value() for lead in version.decision_leads
                )
                != tuple(lead.stable_lead_value() for lead in item.decision_leads)
                or expected_ordinal == 1
                and version.decision_leads != item.decision_leads
                or row[1] != expected_ordinal
                or row[2] != previous
            ):
                raise WorkItemContractError("Work Item Version chain differs")
            missing = self._immutable_lineage_reasons(version)
            if missing:
                raise WorkItemContractError(
                    "Version immutable lineage differs: " + ",".join(missing)
                )
            previous = str(row[0])
        latest = rows[-1]
        if tuple(head) != (latest[0], latest[1], latest[8]):
            raise WorkItemContractError("Work Item head is not the chain maximum")

    def _verify_integrity(self) -> None:
        tables = {
            r[0]
            for r in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "triage_work_items" not in tables:
            return
        items: dict[str, TriageWorkItem] = {}
        for row in self._connection.execute(
            "SELECT work_item_id,decision_scope_digest,decision_lead_count,"
            "canonical_bytes,canonical_digest FROM triage_work_items"
        ):
            item = TriageWorkItem.from_canonical_bytes(bytes(row[3]))
            if (
                item.work_item_id != row[0]
                or item.decision_scope_digest != row[1]
                or len(item.decision_leads) != row[2]
                or item.canonical_digest != row[4]
            ):
                raise WorkItemContractError("Work Item retained bytes differ")
            items[item.work_item_id] = item
        item_ids = set(items)
        version_item_ids = {
            str(row[0])
            for row in self._connection.execute(
                "SELECT DISTINCT work_item_id FROM triage_work_item_versions"
            )
        }
        retained_heads = tuple(self._connection.execute(
            "SELECT work_item_id,current_version_id,current_ordinal,current_version_digest "
            "FROM triage_work_item_heads"
        ))
        head_ids = {str(row[0]) for row in retained_heads}
        head_version_ids = {str(row[1]) for row in retained_heads}
        if item_ids != version_item_ids or item_ids != head_ids:
            raise WorkItemContractError("Work Item chain coverage differs")
        previous_by_item: dict[str, tuple[int, str]] = {}
        versions: dict[str, TriageWorkItemVersion] = {}
        maximum_by_item: dict[str, int] = {}
        current_reasons: dict[str, list[str]] = {}
        for row in self._connection.execute(
            "SELECT version_id,work_item_id,ordinal,previous_version_id,"
            "decision_scope_digest,retrieval_outcome,watch_condition_id,"
            "source_lead_disposition_id,canonical_bytes,canonical_digest "
            "FROM triage_work_item_versions ORDER BY work_item_id,ordinal"
        ):
            version = TriageWorkItemVersion.from_canonical_bytes(bytes(row[8]))
            item = items.get(version.work_item_id)
            if item is None:
                raise WorkItemContractError("Version retained bytes differ")
            prior = previous_by_item.get(version.work_item_id)
            version_stable_scope = tuple(
                lead.stable_lead_value() for lead in version.decision_leads
            )
            item_stable_scope = tuple(
                lead.stable_lead_value() for lead in item.decision_leads
            )
            if (
                version.version_id != row[0]
                or version.work_item_id != row[1]
                or version.ordinal != row[2]
                or version.previous_version_id != row[3]
                or item.decision_scope_digest != row[4]
                or version_stable_scope != item_stable_scope
                or (
                    version.ordinal == 1
                    and version.decision_leads != item.decision_leads
                )
                or (version.retrieval.outcome or version.retrieval.state.value)
                != row[5]
                or self._causal_ids(version) != (row[6], row[7])
                or version.canonical_digest != row[9]
                or (prior is None and version.ordinal != 1)
                or (
                    prior is not None
                    and (
                        version.ordinal != prior[0] + 1
                        or version.previous_version_id != prior[1]
                    )
                )
            ):
                raise WorkItemContractError("Version retained bytes differ")
            previous_by_item[version.work_item_id] = (
                version.ordinal,
                version.version_id,
            )
            versions[version.version_id] = version
            maximum_by_item[version.work_item_id] = version.ordinal
            current_usable = (
                version.version_id in head_version_ids and version.retrieval.usable
                and self._retrieval_authority is not None
            )
            # Currentness already verifies retained retrieval integrity. Keep
            # source corruption fatal even when a current head is merely stale.
            missing = (
                self._immutable_source_reasons(version) if current_usable
                else self._immutable_lineage_reasons(version)
            )
            if missing:
                raise WorkItemContractError(
                    "Version immutable lineage differs: " + ",".join(missing)
                )
            if current_usable:
                current_reasons[version.version_id] = self._upstream_reasons(version)
        heads: dict[str, TriageWorkItemVersion] = {}
        for row in retained_heads:
            version = versions.get(str(row[1]))
            if (
                version is None
                or version.work_item_id != row[0]
                or version.ordinal != row[2]
                or version.canonical_digest != row[3]
                or maximum_by_item.get(str(row[0])) != row[2]
            ):
                raise WorkItemContractError("Work Item head is not rebuildable")
            heads[str(row[0])] = version
        active_leads: dict[str, str] = {}
        for work_item_id, item in sorted(items.items()):
            version = heads[work_item_id]
            if not version.retrieval.usable or current_reasons[version.version_id]:
                continue
            for lead in item.decision_leads:
                owner = active_leads.setdefault(lead.lead_id, item.work_item_id)
                if owner != item.work_item_id:
                    raise WorkItemContractError(
                        "active decision Lead scopes overlap"
                    )

    def _immutable_lineage_reasons(self, version: TriageWorkItemVersion) -> list[str]:
        reasons = self._immutable_source_reasons(version)
        if self._retrieval_authority is not None:
            try:
                self._retrieval_authority.verify_retained_integrity(
                    self._connection, version.retrieval
                )
            except WorkItemContractError:
                reasons.append("retrieval")
        elif version.retrieval.state is RetrievalBindingState.RECEIPT:
            reasons.append("retrieval")
        return reasons

    def _immutable_source_reasons(self, version: TriageWorkItemVersion) -> list[str]:
        if self._validate_retained_event is not None:
            events = {
                lead.lead_event_id
                for lead in (*version.decision_leads, *version.context_leads)
            }
            events.update(lead.disposition_event_id for lead in version.decision_leads)
            for lead in (*version.decision_leads, *version.context_leads):
                for table, column, identifier in (
                    ("discovery_gate_decisions", "decision_id", lead.gate_decision_id),
                    ("source_definition_versions", "version_id", lead.definition_version_id),
                    ("source_definitions", "definition_id", lead.definition_id),
                ):
                    row = self._connection.execute(
                        f"SELECT authority_event_id FROM {table} WHERE {column}=?", (identifier,),
                    ).fetchone()
                    if row is None:
                        raise WorkItemContractError("selected Work Item source authority is absent")
                    events.add(str(row[0]))
            if version.watch is not None:
                events.update((version.watch.watch_event_id, version.watch.source_disposition_event_id))
            if version.supplemental_reentry is not None:
                events.add(version.supplemental_reentry.source_lead_disposition_event_id)
            for event_id in events:
                self._validate_retained_event(event_id)
        reasons: list[str] = []
        for lead in version.decision_leads:
            if not self._lead_retained(lead):
                reasons.append(f"lead:{lead.lead_id}")
            if not self._disposition_retained(lead):
                reasons.append(f"disposition:{lead.lead_id}")
        for lead in version.context_leads:
            if not self._lead_retained(lead):
                reasons.append(f"context:{lead.lead_id}")
        if version.watch is not None:
            if not self._watch_retained(version.watch):
                reasons.append("watch")
        if version.supplemental_reentry is not None:
            reasons.extend(self._supplemental_reasons(version.supplemental_reentry))
        return reasons


__all__ = [
    "LEAD_DISPOSITION_WORK_ITEM_BINDING",
    "SUPPLEMENTAL_DISCOVERY_REENTRY",
    "TRIAGE_WORK_ITEM",
    "TRIAGE_WORK_ITEM_VERSION",
    "WATCH_CONDITION_WORK_ITEM_BINDING",
    "WORK_ITEM_CURRENT_STALE_RULES",
    "ContextLeadBinding",
    "DecisionLeadBinding",
    "ReentryKind",
    "RetrievalBindingState",
    "RetrievalContextAuthority",
    "RetrievalInputBinding",
    "SupplementalDiscoveryReentry",
    "SupplementalLineageBinding",
    "TriageWorkItem",
    "TriageWorkItemStore",
    "TriageWorkItemVersion",
    "WatchConditionWorkItemBinding",
    "WorkItemContractError",
    "WorkItemCurrentAssessment",
    "WorkItemStaleError",
    "WorkItemPriorityBinding",
]
