from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

from newsroom.authority.canonical import (
    MAX_SAFE_INTEGER,
    canonical_json_bytes,
    digest_bytes,
)
from newsroom.authority.models import CommandDefinition
from newsroom.authority.policy import (
    CommandRegistry,
    PayloadGoldenVector,
    PayloadSchemaContract,
    PayloadSchemaRegistry,
)
from newsroom.authority.types import PayloadMode, TrustScope, UtcTimestamp
from newsroom.discovery.record_models import DiscoverySignal, GateDecision, NewsLead
from newsroom.discovery.types import GateOutcome
from newsroom.increment6.collision import (
    CandidateUseOperation,
    CollisionEligibilityOutcome,
    CollisionState,
    CurrentCollisionEligibilityDecision,
    CurrentCollisionEligibilityRequest,
)
from newsroom.increment6.dispositions import ProposalDisposition
from newsroom.increment6.hypotheses import EventHypothesisVersion
from newsroom.increment6.lineage import (
    HypothesisLineageHead,
    HypothesisLineageReceipt,
    HypothesisLineageRelationshipProof,
    replay_hypothesis_lineage,
)
from newsroom.increment6.outcomes import CanonicalOutcome
from newsroom.increment6.proposals import CandidateManifestKind
from newsroom.increment6.relationships import (
    AssessmentStatus,
    RelationshipAssessment,
    RetainedRelationshipDecisionReceipt,
)

_json, _hash = canonical_json_bytes, digest_bytes


STORY_CANDIDATE = "newsroom.increment6.story-candidate.v1"
STORY_CANDIDATE_VERSION = "newsroom.increment6.story-candidate-version.v1"
CANDIDATE_ADMISSION = "newsroom.increment6.candidate-admission.v1"
CANDIDATE_CURRENT_VERSION = "EXACT_RETAINED_MAX_ORDINAL_HEAD"
CANDIDATE_COMMAND_TYPE = "increment6.story-candidate.admit"
CANDIDATE_COMMAND_SCHEMA = "newsroom.increment6.candidate-admission-command.v1"
MAX_CANDIDATE_CANONICAL_BYTES = 16_777_216
MAX_CANDIDATE_COMMAND_BYTES = MAX_CANDIDATE_CANONICAL_BYTES
_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\-]{0,255}\Z")
_MANIFEST_DIGESTS = "semantic_scope_digest hypothesis_version_digest relationship_assessment_digest collision_request_digest collision_decision_digest collision_key_digest"
_MANIFEST_TUPLES = "lineage_history_digests disposition_digests candidate_manifest_digests likely_new_information reader_utility_bases uncertainties evidence_objectives governing_versions missing_context retrieval_incompleteness lead_incompleteness_warnings signal_operational_finding_ids"
_OPTIONAL_TUPLES = "lineage_history_digests missing_context retrieval_incompleteness lead_incompleteness_warnings signal_operational_finding_ids"
_DISPOSITION_FIELDS = "proposal_id proposal_content_identity proposal_canonical_digest work_item_id work_item_version_id work_item_version_digest retrieval_context_id retrieval_context_digest"
_SOURCE_FIELDS = "definition_id definition_version_id item_id revision_id representation_id occurrence_id transition_id"
_POLICIES = "signal_admission_policy gate_policy duplicate_policy newness_policy time_validity_policy exclusion_policy"
_CONTEXT = "collision_namespace generation_id query_valid_time serving_time authority_watermark"
_NO_EFFECTS = "authorises_authority authorises_persistence authorises_external_effect authorises_publication authorises_evidence authorises_egress production_activation_authorised candidate_effect_performed creates_candidate creates_version mutates_current_version"
_COMMAND_FIELDS = "schema_version request hypothesis_version_id relationship_assessment_digest disposition_ids collision_request comparator_collision_request admission collision_decision comparator_collision_decision effect_identity"
_EFFECT_FIELDS = (
    "candidate_id committed_admission_decision_id version_id version_ordinal"
)
_PUBLIC = "STORY_CANDIDATE,STORY_CANDIDATE_VERSION,CANDIDATE_ADMISSION,CANDIDATE_CURRENT_VERSION,CANDIDATE_COMMAND_TYPE,CandidateContractError,CandidateAdmissionOutcome,CandidateAdmissionReason,CandidateLeadSignalBinding,CandidateGoverningStateStatus,CandidateGoverningStateBinding,CandidateGoverningState,CandidateGoverningManifest,StoryCandidate,StoryCandidateVersion,CandidateDistinctScopeProof,CandidateAdmissionRequest,CandidateAdmission,StoryCandidateAuthority,build_candidate_governing_manifest,build_candidate_distinct_scope_proof,evaluate_candidate_admission,validate_candidate_first_version,validate_candidate_version_successor,candidate_command_definition,merge_candidate_authority_registries,open_story_candidate_authority"
_CANDIDATE_ROUTES = {
    "NEW_EVENT_CANDIDATE": CandidateManifestKind.NEW_EVENT,
    "DEVELOPMENT_CANDIDATE": CandidateManifestKind.DEVELOPMENT,
    "CORRECTION_CANDIDATE": CandidateManifestKind.CORRECTION,
}
_ADMISSIBLE_RELATIONSHIP = {
    "NEW_EVENT": CanonicalOutcome.REL_NO_ADEQUATE_PRIOR_MATCH,
    "DEVELOPMENT": CanonicalOutcome.REL_DEVELOPMENT_OF,
    "CORRECTION": CanonicalOutcome.REL_CORRECTION_REVERSAL_OF,
}


class CandidateContractError(ValueError):
    pass


_Error = CandidateContractError


def _string_enum(name: str, values: str):
    return StrEnum(name, {value: value for value in values.split()})


CandidateAdmissionOutcome = _string_enum(
    "CandidateAdmissionOutcome",
    "ADMISSIBLE DUPLICATE_EQUIVALENT DISTINCT INCOMPLETE STALE BLOCKED",
)
CandidateAdmissionReason = _string_enum(
    "CandidateAdmissionReason",
    "NEW_CANDIDATE_PRE_EFFECT SUCCESSOR_VERSION_PRE_EFFECT EXACT_MANIFEST_REPLAY RELATED_DISTINCT_PRE_EFFECT GOVERNING_EVIDENCE_INCOMPLETE COLLISION_AUTHORITY_UNAVAILABLE COLLISION_AUTHORITY_STALE CANDIDATE_CAS_STALE COLLISION_OPERATION_CAS_DIFFERS COLLISION_AUTHORITY_BLOCKED GOVERNING_STATE_STALE GOVERNING_STATE_BLOCKED",
)


_REASON_OUTCOME = {
    CandidateAdmissionReason.NEW_CANDIDATE_PRE_EFFECT: CandidateAdmissionOutcome.ADMISSIBLE,
    CandidateAdmissionReason.SUCCESSOR_VERSION_PRE_EFFECT: CandidateAdmissionOutcome.ADMISSIBLE,
    CandidateAdmissionReason.EXACT_MANIFEST_REPLAY: CandidateAdmissionOutcome.DUPLICATE_EQUIVALENT,
    CandidateAdmissionReason.RELATED_DISTINCT_PRE_EFFECT: CandidateAdmissionOutcome.DISTINCT,
    CandidateAdmissionReason.GOVERNING_EVIDENCE_INCOMPLETE: CandidateAdmissionOutcome.INCOMPLETE,
    CandidateAdmissionReason.COLLISION_AUTHORITY_UNAVAILABLE: CandidateAdmissionOutcome.INCOMPLETE,
    CandidateAdmissionReason.COLLISION_AUTHORITY_STALE: CandidateAdmissionOutcome.STALE,
    CandidateAdmissionReason.CANDIDATE_CAS_STALE: CandidateAdmissionOutcome.STALE,
    CandidateAdmissionReason.GOVERNING_STATE_STALE: CandidateAdmissionOutcome.STALE,
    CandidateAdmissionReason.COLLISION_OPERATION_CAS_DIFFERS: CandidateAdmissionOutcome.BLOCKED,
    CandidateAdmissionReason.COLLISION_AUTHORITY_BLOCKED: CandidateAdmissionOutcome.BLOCKED,
    CandidateAdmissionReason.GOVERNING_STATE_BLOCKED: CandidateAdmissionOutcome.BLOCKED,
}


def _reason_matches(outcome, reason):
    return _REASON_OUTCOME.get(reason) is outcome


class _NoEffect:
    authority = authority_effect = candidate_effect = "NONE"
    for _name in _NO_EFFECTS.split():
        locals()[_name] = False


class _CanonicalFields:
    def canonical_value(self) -> dict[str, object]:
        return _normalise(
            lambda: {name: getattr(self, name) for name in self.__dataclass_fields__},
            "contract cannot be canonicalised",
        )

    @property
    def canonical_digest(self) -> str:
        return _hash(_json(self.canonical_value()))

    @classmethod
    def from_value(cls, value: object) -> Self:
        return _normalise(
            lambda: cls(**_exact(value, set(cls.__dataclass_fields__), cls.__name__)),  # type: ignore[arg-type]
            f"{cls.__name__} replay failed",
        )


def _normalise[T](operation: object, message: str) -> T:
    try:
        return operation()  # type: ignore[operator,no-any-return]
    except CandidateContractError:
        raise
    except Exception as exc:
        raise _Error(message) from exc


def _cap(raw: bytes, field: str) -> bytes:
    _require(
        len(raw) <= MAX_CANDIDATE_CANONICAL_BYTES,
        f"{field} exceeds canonical byte bound",
    )
    _decode(raw)
    return raw


def _total(message: str):
    return lambda function: (
        lambda *args, **kwargs: _normalise(lambda: function(*args, **kwargs), message)
    )


def _require(condition: object, message: str) -> None:
    if not condition:
        raise _Error(message)


def _valid(condition):
    _require(condition, "invalid Candidate contract")


def _same_fields(left: object, right: object, fields: tuple[str, ...]) -> bool:
    return all(getattr(left, name) == getattr(right, name) for name in fields)


def _attrs(value, fields):
    fields = fields.split() if type(fields) is str else fields
    return tuple(getattr(value, field) for field in fields)


def _exact_tuple(value: object, kind: type, field: str, *, required=False):
    _require(
        type(value) is tuple
        and (not required or bool(value))
        and all(type(item) is kind for item in value),
        f"{field} producers must be exact",
    )
    return value


def _all_or_none(values: tuple[object, ...], field: str) -> bool:
    _require(
        all(item is None for item in values)
        or all(item is not None for item in values),
        f"{field} binding is partial",
    )
    return values[0] is not None


def _canonical_bytes(value: object, field: str) -> bytes:
    return _cap(
        _normalise(lambda: _json(value), f"{field} cannot be canonicalised"),
        field,
    )


def _decode_document(raw: bytes, schema: str, fields: set[str], field: str):
    value = _exact(_decode(raw), {"schema_version", *fields}, field)
    _require(value.pop("schema_version") == schema, f"{field} schema is unsupported")
    return value


def _document_bytes(schema: str, value: dict, field: str) -> bytes:
    return _canonical_bytes({"schema_version": schema, **value}, field)


def _pattern(value: object, field: str, pattern, label: str) -> str:
    _require(
        type(value) is str and pattern.fullmatch(value) is not None,
        f"{field} must be {label}",
    )
    return value


def _uuid(value: object, field: str) -> str:
    return _pattern(value, field, _UUID, "a canonical UUID")


def _digest(value: object, field: str) -> str:
    return _pattern(value, field, _DIGEST, "a canonical SHA-256 digest")


def _token(value: object, field: str) -> str:
    return _pattern(value, field, _TOKEN, "bounded canonical text")


def _enum(kind, value: object, field: str):
    return _normalise(lambda: kind(value), f"{field} is unsupported")


def _ordinal(value: object, field: str = "ordinal") -> int:
    if type(value) is not int or not 1 <= value <= MAX_SAFE_INTEGER:
        raise _Error(f"{field} must be an exact positive integer")
    return value


def _exact(value: object, fields: set[str], field: str) -> dict[str, object]:
    if type(value) is not dict:
        raise _Error(f"{field} fields are not exact")
    try:
        if set(value) != fields:
            raise _Error(f"{field} fields are not exact")
    except CandidateContractError:
        raise
    except Exception as exc:
        raise _Error(f"{field} fields are not exact") from exc
    return value


def _decode(raw: bytes) -> dict[str, object]:
    _valid(
        type(raw) is bytes and bool(raw) and (len(raw) <= MAX_CANDIDATE_CANONICAL_BYTES)
    )

    def unique(pairs):
        value = {}
        for key, child in pairs:
            _require(key not in value, f"duplicate object name: {key}")
            value[key] = child
        return value

    def integer(text):
        _valid(len(text.lstrip("-")) <= 16)
        value = int(text)
        _valid(-MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER)
        return value

    def unsupported(_):
        raise _Error("unsupported number")

    try:
        value = json.loads(
            raw.decode(),
            object_pairs_hook=unique,
            parse_int=integer,
            parse_float=unsupported,
            parse_constant=unsupported,
        )
    except CandidateContractError:
        raise
    except Exception as exc:
        raise _Error("canonical input is invalid UTF-8 JSON") from exc
    pending, nodes = [(value, 1)], 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _require(
            depth <= 24 and nodes <= 32768, "canonical input exceeds structural bounds"
        )
        if type(item) in (dict, list):
            children = item.values() if type(item) is dict else item
            pending.extend((child, depth + 1) for child in children)
        else:
            _valid(type(item) in (str, int, bool, type(None)))
    _valid(type(value) is dict)
    _valid(
        _normalise(lambda: _json(value), "canonical input cannot be normalised") == raw
    )
    return value


@dataclass(frozen=True, slots=True)
class CandidateLeadSignalBinding(_NoEffect, _CanonicalFields):
    lead_id: str
    lead_digest: str
    signal_id: str
    signal_digest: str
    lead_event_id: str
    lead_aggregate_version: int
    signal_event_id: str
    signal_aggregate_version: int
    coverage_basis: str
    incomplete: bool

    @_total("invalid Lead/Signal binding")
    def __post_init__(self) -> None:
        _valid(type(self) is CandidateLeadSignalBinding)
        for name in ("lead_id", "signal_id", "lead_event_id", "signal_event_id"):
            _uuid(getattr(self, name), name)
        for name in ("lead_digest", "signal_digest"):
            _digest(getattr(self, name), name)
        for name in ("lead_aggregate_version", "signal_aggregate_version"):
            _ordinal(getattr(self, name), name)
        _valid(type(self.coverage_basis) is str and bool(self.coverage_basis))
        raw = _normalise(
            lambda: self.coverage_basis.encode(),
            "invalid Candidate contract",
        )
        _valid(_json(_decode(raw)) == raw)
        _valid(type(self.incomplete) is bool)


CandidateGoverningStateStatus = _string_enum(
    "CandidateGoverningStateStatus", "COMPLETE INCOMPLETE UNAVAILABLE BLOCKED"
)


@dataclass(frozen=True, slots=True)
class CandidateGoverningStateBinding(_NoEffect, _CanonicalFields):
    coverage: str
    policy: str
    rights: str
    source: str
    triage: str
    retrieval: str
    event_lineage: str
    admission_ruleset: str
    declared_versions: str

    @_total("invalid governing-state binding")
    def __post_init__(self) -> None:
        if type(self) is not CandidateGoverningStateBinding:
            raise _Error("governing-state binding must be exact")
        for name in self.__dataclass_fields__:
            _digest(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class CandidateGoverningState(_NoEffect):
    status: CandidateGoverningStateStatus
    binding: CandidateGoverningStateBinding | None

    @_total("invalid governing state")
    def __post_init__(self) -> None:
        if type(self.status) is not CandidateGoverningStateStatus or (
            self.status is CandidateGoverningStateStatus.COMPLETE
        ) != (type(self.binding) is CandidateGoverningStateBinding):
            raise _Error("governing-state status and binding differ")


@dataclass(frozen=True, slots=True)
class CandidateGoverningManifest(_NoEffect):
    semantic_scope_digest: str
    candidate_kind: CandidateManifestKind
    hypothesis_id: str
    hypothesis_version_id: str
    hypothesis_version_digest: str
    proposed_summary: str
    hypothesis_status: str
    relationship_status: AssessmentStatus
    relationship_outcome: CanonicalOutcome | None
    relationship_assessment_digest: str
    relationship_comparator_hypothesis_id: str | None
    relationship_comparator_version_id: str | None
    relationship_comparator_version_digest: str | None
    lineage_generation: int
    lineage_history_digests: tuple[str, ...]
    disposition_digests: tuple[str, ...]
    candidate_manifest_digests: tuple[str, ...]
    lead_signal_bindings: tuple[CandidateLeadSignalBinding, ...]
    proposed_geography: str
    proposed_category: str
    urgency: str
    likely_new_information: tuple[str, ...]
    reader_utility_bases: tuple[str, ...]
    uncertainties: tuple[str, ...]
    evidence_objectives: tuple[str, ...]
    governing_versions: tuple[str, ...]
    missing_context: tuple[str, ...]
    retrieval_incompleteness: tuple[str, ...]
    lead_incompleteness_warnings: tuple[str, ...]
    signal_operational_finding_ids: tuple[str, ...]
    collision_request_digest: str
    collision_decision_digest: str
    collision_namespace: str
    collision_key_digest: str
    governing_state_binding: CandidateGoverningStateBinding
    incomplete: bool

    @_total("invalid governing manifest")
    def __post_init__(self) -> None:
        _valid(type(self) is CandidateGoverningManifest)
        digests = _MANIFEST_DIGESTS.split()
        for name in digests:
            _digest(getattr(self, name), name)
        _uuid(self.hypothesis_id, "hypothesis_id")
        _uuid(self.hypothesis_version_id, "hypothesis_version_id")
        _valid(
            type(self.proposed_summary) is str
            and self.proposed_summary == self.proposed_summary.strip()
            and bool(self.proposed_summary)
        )
        _token(self.hypothesis_status, "hypothesis_status")
        _valid(self.hypothesis_status == "UNVERIFIED_DISCOVERY_HYPOTHESIS")
        _valid(
            type(self.candidate_kind) is CandidateManifestKind
            and type(self.relationship_status) is AssessmentStatus
        )
        _valid(
            self.relationship_outcome is None
            or type(self.relationship_outcome) is CanonicalOutcome
        )
        _valid(
            (self.relationship_status is AssessmentStatus.COMPLETE)
            == (self.relationship_outcome is not None)
        )
        comparator = (
            self.relationship_comparator_hypothesis_id,
            self.relationship_comparator_version_id,
            self.relationship_comparator_version_digest,
        )
        _valid(
            all(item is None for item in comparator)
            or all(item is not None for item in comparator)
        )
        if comparator[0] is not None:
            _uuid(comparator[0], "relationship_comparator_hypothesis_id")
            _uuid(comparator[1], "relationship_comparator_version_id")
            _digest(comparator[2], "relationship_comparator_version_digest")
        no_comparator = all(item is None for item in comparator)
        _valid(
            no_comparator
            if self.relationship_status is not AssessmentStatus.COMPLETE
            or self.relationship_outcome is CanonicalOutcome.REL_NO_ADEQUATE_PRIOR_MATCH
            else not no_comparator
        )
        _valid(type(self.lineage_generation) is int and self.lineage_generation >= 0)
        _token(self.collision_namespace, "collision_namespace")
        _valid(type(self.governing_state_binding) is CandidateGoverningStateBinding)
        tuple_fields = _MANIFEST_TUPLES.split()
        optional = set(_OPTIONAL_TUPLES.split())
        for name in tuple_fields:
            values = getattr(self, name)
            exact = type(values) is tuple and all(
                type(value) is str and value for value in values
            )
            ordered = (
                (
                    len(set(values)) == len(values)
                    if name == "lineage_history_digests"
                    else values == tuple(sorted(set(values)))
                )
                if exact
                else False
            )
            _require(
                exact and (bool(values) or name in optional) and ordered,
                f"invalid {name}",
            )
        for name in digests[1:3] + tuple_fields[:3]:
            for value in (
                getattr(self, name, ())
                if name in tuple_fields
                else (getattr(self, name),)
            ):
                _digest(value, name)
        for value in self.signal_operational_finding_ids:
            _uuid(value, "signal_operational_finding_id")
        for name in ("proposed_geography", "proposed_category", "urgency"):
            _token(getattr(self, name), name)
        _valid(
            type(self.lead_signal_bindings) is tuple
            and all(
                type(item) is CandidateLeadSignalBinding
                for item in self.lead_signal_bindings
            )
        )
        lead_ids = _normalise(
            lambda: tuple(item.lead_id for item in self.lead_signal_bindings),
            "Lead/Signal bindings cannot be validated",
        )
        _valid(bool(lead_ids) and lead_ids == tuple(sorted(set(lead_ids))))
        _require(
            type(self.incomplete) is bool
            and self.incomplete
            == (
                self.relationship_status is not AssessmentStatus.COMPLETE
                or any(item.incomplete for item in self.lead_signal_bindings)
            ),
            "manifest incompleteness differs from Lead/Signal lineage",
        )
        expected_scope = _project(
            {
                "collision_namespace": self.collision_namespace,
                "collision_key_digest": self.collision_key_digest,
                "hypothesis_id": self.hypothesis_id,
            }
        )
        _valid(self.semantic_scope_digest == expected_scope)
        _ = self.canonical_bytes

    def canonical_value(self) -> dict[str, object]:
        def render():
            value = {name: getattr(self, name) for name in self.__dataclass_fields__}
            for name in (
                "candidate_kind",
                "relationship_status",
                "relationship_outcome",
            ):
                item = value[name]
                value[name] = None if item is None else item.value
            value["lead_signal_bindings"] = [
                item.canonical_value() for item in self.lead_signal_bindings
            ]
            value["governing_state_binding"] = (
                self.governing_state_binding.canonical_value()
            )
            value.update(
                (name, list(item))
                for name, item in tuple(value.items())
                if type(item) is tuple
            )
            return value

        return _normalise(render, "governing manifest cannot be canonicalised")

    def version_material_value(self) -> dict[str, object]:
        value = self.canonical_value()
        for name in ("collision_request_digest", "collision_decision_digest"):
            value.pop(name)
        return value

    @property
    def version_material_digest(self) -> str:
        return _project(
            {
                "schema_version": "newsroom.increment6.candidate-version-material.v1",
                "manifest": self.version_material_value(),
            }
        )

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.canonical_value(), "governing manifest")

    @property
    def canonical_digest(self) -> str:
        return _hash(self.canonical_bytes)

    @classmethod
    def from_value(cls, value: object) -> Self:
        item = _exact(value, set(cls.__dataclass_fields__), "governing manifest")
        copied = dict(item)
        for name, kind in (
            ("candidate_kind", CandidateManifestKind),
            ("relationship_status", AssessmentStatus),
        ):
            copied[name] = _enum(kind, item[name], name)
        copied["relationship_outcome"] = (
            None
            if item["relationship_outcome"] is None
            else _enum(
                CanonicalOutcome, item["relationship_outcome"], "relationship_outcome"
            )
        )
        copied["lead_signal_bindings"] = tuple(
            CandidateLeadSignalBinding.from_value(child)
            for child in _tuple_input(
                item["lead_signal_bindings"], "lead_signal_bindings"
            )
        )
        copied["governing_state_binding"] = CandidateGoverningStateBinding.from_value(
            item["governing_state_binding"]
        )
        arrays = _MANIFEST_TUPLES.split()
        for name in arrays:
            copied[name] = _tuple_input(item[name], name)
        return cls(**copied)  # type: ignore[arg-type]


def _tuple_input(value: object, field: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise _Error(f"{field} must be a canonical array")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class StoryCandidate(_NoEffect, _CanonicalFields):
    candidate_id: str
    committed_admission_decision_id: str
    authority_event_id: str
    semantic_scope_digest: str

    @_total("invalid Story Candidate")
    def __post_init__(self) -> None:
        _valid(type(self) is StoryCandidate)
        for name in (
            "candidate_id",
            "committed_admission_decision_id",
            "authority_event_id",
        ):
            _uuid(getattr(self, name), name)
        _valid(
            len(
                {
                    self.candidate_id,
                    self.committed_admission_decision_id,
                    self.authority_event_id,
                }
            )
            == 3
        )
        _digest(self.semantic_scope_digest, "semantic_scope_digest")

    @property
    def canonical_bytes(self) -> bytes:
        return _document_bytes(
            STORY_CANDIDATE, self.canonical_value(), "Story Candidate"
        )

    @property
    def canonical_digest(self) -> str:
        return _hash(self.canonical_bytes)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> Self:
        root = _decode_document(
            raw, STORY_CANDIDATE, set(cls.__dataclass_fields__), "Story Candidate"
        )
        return _normalise(lambda: cls(**root), "Story Candidate replay failed")  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class StoryCandidateVersion(_NoEffect, _CanonicalFields):
    version_id: str
    candidate_id: str
    ordinal: int
    previous_version_id: str | None
    previous_version_digest: str | None
    committed_admission_decision_id: str
    governing_manifest: CandidateGoverningManifest

    @_total("invalid Candidate Version")
    def __post_init__(self) -> None:
        _valid(type(self) is StoryCandidateVersion)
        for name in ("candidate_id", "version_id", "committed_admission_decision_id"):
            _uuid(getattr(self, name), name)
        _valid(
            len(
                {
                    self.candidate_id,
                    self.version_id,
                    self.committed_admission_decision_id,
                }
            )
            == 3
        )
        ordinal = _ordinal(self.ordinal)
        predecessor = _all_or_none(
            (self.previous_version_id, self.previous_version_digest),
            "Candidate Version predecessor",
        )
        _valid((ordinal == 1) != predecessor)
        if predecessor:
            _uuid(self.previous_version_id, "previous_version_id")
            _digest(self.previous_version_digest, "previous_version_digest")
        if type(self.governing_manifest) is not CandidateGoverningManifest:
            raise _Error("Candidate Version manifest must be exact")
        _ = self.governing_manifest.canonical_digest
        _ = self.canonical_bytes

    @property
    def canonical_value(self) -> dict[str, object]:
        value = _CanonicalFields.canonical_value(self)
        value["governing_manifest"] = self.governing_manifest.canonical_value()
        return value

    @property
    def canonical_bytes(self) -> bytes:
        return _document_bytes(
            STORY_CANDIDATE_VERSION,
            {"version": self.canonical_value},
            "Candidate Version",
        )

    @property
    def canonical_digest(self) -> str:
        return _hash(self.canonical_bytes)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> Self:
        root = _decode_document(
            raw, STORY_CANDIDATE_VERSION, {"version"}, "Candidate Version"
        )
        item = _exact(
            root["version"], set(cls.__dataclass_fields__), "Candidate Version"
        )
        item["governing_manifest"] = CandidateGoverningManifest.from_value(
            item["governing_manifest"]
        )
        return _normalise(lambda: cls(**item), "Candidate Version replay failed")  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class CandidateDistinctScopeProof(_NoEffect, _CanonicalFields):
    proposed_collision_digest: str
    comparator_collision_digest: str
    comparator_candidate_id: str
    comparator_version_id: str
    comparator_version_digest: str
    comparator_semantic_scope_digest: str
    context_digest: str

    @_total("invalid distinct-scope proof")
    def __post_init__(self) -> None:
        if type(self) is not CandidateDistinctScopeProof:
            raise _Error("distinct-scope proof must be exact")
        for name in self.__dataclass_fields__:
            validator = _uuid if name.endswith("_id") else _digest
            validator(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class CandidateAdmissionRequest(_NoEffect, _CanonicalFields):
    request_id: str
    actor_identity_digest: str
    idempotency_key: str
    expected_current_version_id: str | None
    expected_current_version_digest: str | None
    expected_current_ordinal: int
    semantic_scope_digest: str
    collision_request_digest: str
    expected_governing_state_digest: str
    distinct_scope_proof_digest: str | None

    @_total("invalid Candidate Admission request")
    def __post_init__(self) -> None:
        _valid(type(self) is CandidateAdmissionRequest)
        _uuid(self.request_id, "request_id")
        _token(self.idempotency_key, "idempotency_key")
        for name in (
            "actor_identity_digest",
            "semantic_scope_digest",
            "collision_request_digest",
            "expected_governing_state_digest",
        ):
            _digest(getattr(self, name), name)
        if self.distinct_scope_proof_digest is not None:
            _digest(self.distinct_scope_proof_digest, "distinct_scope_proof_digest")
        _valid(
            type(self.expected_current_ordinal) is int
            and self.expected_current_ordinal >= 0
        )
        has_current = _all_or_none(
            (self.expected_current_version_id, self.expected_current_version_digest),
            "admission CAS",
        )
        _valid((self.expected_current_ordinal == 0) != has_current)
        if has_current:
            _uuid(self.expected_current_version_id, "expected_current_version_id")
            _digest(
                self.expected_current_version_digest, "expected_current_version_digest"
            )
        _ = self.canonical_bytes

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.canonical_value(), "admission request")

    @property
    def canonical_digest(self) -> str:
        return _hash(self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class CandidateAdmission(_NoEffect):
    request: CandidateAdmissionRequest
    governing_manifest: CandidateGoverningManifest
    outcome: CandidateAdmissionOutcome
    reason: CandidateAdmissionReason
    current_candidate_id: str | None
    current_candidate_version_id: str | None
    current_candidate_version_digest: str | None
    distinct_scope_proof: CandidateDistinctScopeProof | None = None

    @_total("Candidate Admission cannot be validated")
    def __post_init__(self) -> None:
        q, m, outcome, reason = (
            self.request,
            self.governing_manifest,
            self.outcome,
            self.reason,
        )
        R = CandidateAdmissionReason
        _valid(
            type(self) is CandidateAdmission and type(q) is CandidateAdmissionRequest
        )
        _valid(
            type(m) is CandidateGoverningManifest
            and type(outcome) is CandidateAdmissionOutcome
        )
        _ = m.canonical_digest
        _require(
            type(reason) is CandidateAdmissionReason
            and _reason_matches(outcome, reason),
            "admission outcome and reason differ",
        )
        proof = self.distinct_scope_proof
        _require(
            (type(proof) is CandidateDistinctScopeProof)
            == (reason is R.RELATED_DISTINCT_PRE_EFFECT)
            and (
                proof is None or q.distinct_scope_proof_digest == proof.canonical_digest
            ),
            "distinct result and proof differ",
        )
        current = (
            self.current_candidate_id,
            self.current_candidate_version_id,
            self.current_candidate_version_digest,
        )
        has_current = _all_or_none(current, "current Candidate")
        if has_current:
            _uuid(current[0], "current_candidate_id")
            _uuid(current[1], "current_candidate_version_id")
            _digest(current[2], "current_candidate_version_digest")
        new = reason in {R.NEW_CANDIDATE_PRE_EFFECT, R.RELATED_DISTINCT_PRE_EFFECT}
        retained = reason in {R.SUCCESSOR_VERSION_PRE_EFFECT, R.EXACT_MANIFEST_REPLAY}
        _valid(not new or (q.expected_current_ordinal == 0 and (not has_current)))
        _require(
            not retained or has_current,
            "admission result requires an exact current Candidate binding",
        )
        _valid(
            not retained
            or (q.expected_current_version_id, q.expected_current_version_digest)
            == current[1:]
        )
        _valid(
            not (
                reason is R.CANDIDATE_CAS_STALE
                and (not has_current)
                and (q.expected_current_ordinal == 0)
            )
        )
        _valid(
            q.semantic_scope_digest == m.semantic_scope_digest
            and q.collision_request_digest == m.collision_request_digest
            and (
                reason is R.GOVERNING_STATE_STALE
                or q.expected_governing_state_digest
                == m.governing_state_binding.canonical_digest
            )
        )
        _ = self.canonical_bytes

    @property
    def canonical_bytes(self) -> bytes:
        value = {name: getattr(self, name) for name in self.__dataclass_fields__}
        for name in ("request", "governing_manifest", "distinct_scope_proof"):
            item = value[name]
            value[name] = None if item is None else item.canonical_value()
        value.update(
            schema_version=CANDIDATE_ADMISSION,
            authority="NONE",
            candidate_effect="NONE",
        )
        value["outcome"], value["reason"] = self.outcome.value, self.reason.value
        return _canonical_bytes(value, "Candidate Admission")

    @property
    def canonical_digest(self) -> str:
        return _hash(self.canonical_bytes)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> Self:
        root = _exact(
            _decode(raw),
            {
                "schema_version",
                "authority",
                "candidate_effect",
                *cls.__dataclass_fields__,
            },
            "Candidate Admission",
        )
        if (
            root.pop("schema_version") != CANDIDATE_ADMISSION
            or root.pop("authority") != "NONE"
            or root.pop("candidate_effect") != "NONE"
        ):
            raise _Error("invalid Candidate contract")
        root["request"] = CandidateAdmissionRequest.from_value(root["request"])
        root["governing_manifest"] = CandidateGoverningManifest.from_value(
            root["governing_manifest"]
        )
        root["outcome"] = _enum(
            CandidateAdmissionOutcome, root["outcome"], "admission outcome"
        )
        root["reason"] = _enum(
            CandidateAdmissionReason, root["reason"], "admission reason"
        )
        root["distinct_scope_proof"] = (
            None
            if root["distinct_scope_proof"] is None
            else CandidateDistinctScopeProof.from_value(root["distinct_scope_proof"])
        )
        return _normalise(lambda: cls(**root), "Candidate Admission replay failed")  # type: ignore[arg-type]


def build_candidate_governing_manifest(
    *,
    hypothesis_version: EventHypothesisVersion,
    lineage_receipts: tuple[HypothesisLineageReceipt, ...],
    lineage_initial_heads: tuple[HypothesisLineageHead, ...],
    lineage_versions: tuple[EventHypothesisVersion, ...],
    lineage_relationship_proofs: tuple[HypothesisLineageRelationshipProof, ...],
    dispositions: tuple[ProposalDisposition, ...],
    leads: tuple[NewsLead, ...],
    signals: tuple[DiscoverySignal, ...],
    gates: tuple[GateDecision, ...],
    relationship: RelationshipAssessment | RetainedRelationshipDecisionReceipt,
    collision: CurrentCollisionEligibilityDecision,
) -> CandidateGoverningManifest:
    def build() -> CandidateGoverningManifest:
        version = hypothesis_version
        if type(version) is not EventHypothesisVersion:
            raise _Error("Hypothesis producer must be exact")
        lineage = replay_hypothesis_lineage(
            lineage_receipts,
            initial_heads=lineage_initial_heads,
            versions=lineage_versions,
            relationship_proofs=lineage_relationship_proofs,
        )
        _exact_tuple(dispositions, ProposalDisposition, "disposition", required=True)
        _exact_tuple(leads, NewsLead, "Lead")
        _exact_tuple(signals, DiscoverySignal, "Signal")
        _exact_tuple(gates, GateDecision, "Gate")
        if type(collision) is not CurrentCollisionEligibilityDecision:
            raise _Error("collision producer must be exact")
        if type(relationship) is RetainedRelationshipDecisionReceipt:
            assessment = relationship.assessment
        elif type(relationship) is RelationshipAssessment:
            assessment = relationship
            if assessment.status is AssessmentStatus.COMPLETE:
                raise _Error("invalid Candidate contract")
        else:
            raise _Error("relationship producer must be exact")
        if (
            assessment.subject.hypothesis_id != version.hypothesis_id
            or assessment.subject.version_id != version.version_id
            or assessment.subject.version_digest != version.canonical_digest
        ):
            raise _Error("invalid Candidate contract")
        heads = [
            head
            for head in lineage.active_heads
            if head.node.version_id == version.version_id
            and head.node.version_digest == version.canonical_digest
            and head.node.hypothesis_id == version.hypothesis_id
        ]
        if len(heads) != 1:
            raise _Error("Hypothesis Version is not one exact current lineage head")
        if heads[0].generation > 0 and (
            not any(
                binding.assessment_digest == assessment.canonical_digest
                for receipt in lineage.history
                for binding in receipt.relationships
            )
        ):
            raise _Error("invalid Candidate contract")
        if len({item.disposition_id for item in version.source_bindings}) != len(
            version.source_bindings
        ):
            raise _Error("duplicate disposition IDs")
        source_by_disposition = {
            item.disposition_id: item for item in version.source_bindings
        }
        if set(source_by_disposition) != {item.disposition_id for item in dispositions}:
            raise _Error("invalid Candidate contract")
        lead_by_id = {str(item.request.lead_id): item for item in leads}
        signal_by_id = {str(item.request.signal_id): item for item in signals}
        gate_by_id = {str(item.request.decision_id): item for item in gates}
        if (
            len(lead_by_id) != len(leads)
            or len(signal_by_id) != len(signals)
            or len(gate_by_id) != len(gates)
        ):
            raise _Error("Lead or Signal producers are duplicated")
        bindings: list[CandidateLeadSignalBinding] = []
        manifests = []
        for disposition in dispositions:
            route = disposition.route.value
            manifest = disposition.route_binding.candidate_manifest
            if (
                route not in _CANDIDATE_ROUTES
                or manifest is None
                or manifest.manifest_kind is not _CANDIDATE_ROUTES[route]
            ):
                raise _Error("invalid Candidate contract")
            source = source_by_disposition[disposition.disposition_id]
            source_projection = (
                _hash(disposition.canonical_bytes),
                disposition.finding_set_digest,
                disposition.route_binding_digest,
                disposition.lead_head.decision_lead_id,
                disposition.lead_head.decision_lead_digest,
                disposition.lead_head.current_disposition_head_id,
                disposition.lead_head.current_disposition_head_digest,
            )
            if tuple(source.canonical_value().values()) != (
                disposition.disposition_id,
                *source_projection,
            ):
                raise _Error("invalid Candidate contract")
            proposal_hypothesis = disposition.route_binding.hypothesis
            route_fields = (
                ("proposal_local_id", "proposal_local_id"),
                ("summary", "proposed_summary"),
                ("relationship_kind", "proposed_relationship"),
                ("target_hypothesis_id", "proposed_target_hypothesis_id"),
            )
            if (
                proposal_hypothesis is None
                or not _same_fields(
                    disposition, version, tuple(_DISPOSITION_FIELDS.split())
                )
                or (
                    not all(
                        (
                            getattr(proposal_hypothesis, left)
                            == getattr(version, right)
                            for left, right in route_fields
                        )
                    )
                )
            ):
                raise _Error("invalid Candidate contract")
            manifests.append(manifest)
        kinds = {manifest.manifest_kind for manifest in manifests}
        if len(kinds) != 1:
            raise _Error("invalid Candidate contract")
        candidate_kind = kinds.pop()
        _validate_relationship_route(version, assessment)
        contributing = {
            lead for manifest in manifests for lead in manifest.contributing_lead_ids
        }
        if contributing != set(lead_by_id):
            raise _Error("invalid Candidate contract")
        _validate_contributing_dispositions(contributing, dispositions)
        if set(signal_by_id) != {str(lead.request.signal_id) for lead in leads}:
            raise _Error("Signals differ from exact contributing Lead lineage")
        if set(gate_by_id) != {
            str(lead.request.promoting_gate_decision_id) for lead in leads
        }:
            raise _Error("Gates differ from exact promoting lineage")
        for lead_id in sorted(contributing):
            lead = lead_by_id[lead_id]
            source = next(
                item
                for item in version.source_bindings
                if item.decision_lead_id == lead_id
            )
            if source.decision_lead_digest != lead.canonical_digest:
                raise _Error(
                    "disposition Lead digest differs from exact supplied News Lead"
                )
            signal_id = str(lead.request.signal_id)
            signal = signal_by_id.get(signal_id)
            if signal is None:
                raise _Error("invalid Candidate contract")
            gate = gate_by_id[str(lead.request.promoting_gate_decision_id)]
            if (
                str(gate.request.signal_id) != signal_id
                or gate.request.coverage != lead.request.coverage
                or gate.request.outcome is not GateOutcome.PROMOTED_TO_LEAD
                or gate.request.evaluated_definition_version_id
                != signal.request.definition_version_id
                or gate.request.evaluated_definition_version_id
                != lead.request.definition_version_id
            ):
                raise _Error("promoting Gate differs from Lead lineage")
            # Shared source lineage is exact only when the Lead retains the Signal's source identities.
            for field in _SOURCE_FIELDS.split():
                if getattr(lead.request, field) != getattr(signal.request, field):
                    raise _Error("Lead and Signal source lineage differ")
            bindings.append(
                CandidateLeadSignalBinding(
                    lead_id,
                    lead.canonical_digest,
                    signal_id,
                    signal.canonical_digest,
                    str(lead.event_id),
                    lead.aggregate_version,
                    str(signal.event_id),
                    signal.aggregate_version,
                    _json(lead.request.coverage.canonical_value()).decode("utf-8"),
                    bool(lead.request.incompleteness_warnings)
                    or signal.request.incomplete,
                )
            )
        binding = collision.request.binding
        if (
            binding.subject_id != version.hypothesis_id
            or binding.subject_version_id != version.version_id
            or binding.subject_version_digest != version.canonical_digest
        ):
            raise _Error("invalid Candidate contract")
        state_binding = _derive_governing_state(
            gates, leads, signals, dispositions, version, assessment, lineage
        )
        comparator = assessment.comparator
        comparator_values = (
            (None, None, None)
            if comparator is None
            else (
                comparator.hypothesis_id,
                comparator.version_id,
                comparator.version_digest,
            )
        )
        return CandidateGoverningManifest(
            _project(
                {
                    "collision_namespace": binding.collision_namespace,
                    "collision_key_digest": binding.collision_key_digest,
                    "hypothesis_id": version.hypothesis_id,
                }
            ),
            candidate_kind,
            version.hypothesis_id,
            version.version_id,
            version.canonical_digest,
            version.proposed_summary,
            "UNVERIFIED_DISCOVERY_HYPOTHESIS",
            assessment.status,
            assessment.decision,
            assessment.canonical_digest,
            *comparator_values,
            heads[0].generation,
            tuple(item.canonical_digest for item in lineage.history),
            tuple(sorted(_hash(item.canonical_bytes) for item in dispositions)),
            tuple(sorted({_hash(_json(item.canonical_value())) for item in manifests})),
            tuple(bindings),
            *(
                _one(manifests, field)
                for field in ("proposed_geography", "proposed_category", "urgency")
            ),
            tuple(sorted({item.likely_new_information for item in manifests})),
            tuple(sorted({item.reader_utility_basis for item in manifests})),
            *(
                _union(manifests, lambda item, field=field: getattr(item, field))
                for field in (
                    "uncertainties",
                    "evidence_objectives",
                    "governing_versions",
                )
            ),
            *(
                _union(
                    dispositions,
                    lambda item, field=field: getattr(item.route_binding, field),
                )
                for field in ("missing_context", "retrieval_incompleteness")
            ),
            _union(leads, lambda item: item.request.incompleteness_warnings),
            _union(
                signals,
                lambda item: tuple(
                    str(value) for value in item.request.operational_finding_ids
                ),
            ),
            collision.request.request_digest,
            _hash(collision.canonical_bytes),
            binding.collision_namespace,
            binding.collision_key_digest,
            state_binding,
            assessment.status is not AssessmentStatus.COMPLETE
            or any(item.incomplete for item in bindings),
        )

    return _normalise(build, "invalid Candidate contract")


@_total("Candidate Version succession failed closed")
def validate_candidate_version_successor(
    previous: StoryCandidateVersion,
    proposed: StoryCandidateVersion,
) -> StoryCandidateVersion:
    p, q = previous, proposed
    _valid(type(p) is type(q) is StoryCandidateVersion)
    actual = (
        q.candidate_id,
        q.governing_manifest.semantic_scope_digest,
        *_attrs(q, "ordinal previous_version_id previous_version_digest"),
    )
    expected = (
        p.candidate_id,
        p.governing_manifest.semantic_scope_digest,
        p.ordinal + 1,
        p.version_id,
        p.canonical_digest,
    )
    _require(
        actual == expected, "Candidate Version is not the exact contiguous successor"
    )
    _valid(q.version_id not in {p.version_id, q.previous_version_id})
    _require(
        q.committed_admission_decision_id != p.committed_admission_decision_id,
        "Candidate Version successor requires a new committed admission decision",
    )
    _require(
        q.governing_manifest.version_material_digest
        != p.governing_manifest.version_material_digest,
        "equivalent manifest replay cannot create a successor",
    )
    return q


@_total("invalid Candidate contract")
def validate_candidate_first_version(
    candidate: StoryCandidate,
    version: StoryCandidateVersion,
) -> StoryCandidateVersion:
    c, v = candidate, version
    _valid(type(c) is StoryCandidate and type(v) is StoryCandidateVersion)
    actual = (
        *_attrs(v, "candidate_id committed_admission_decision_id"),
        v.governing_manifest.semantic_scope_digest,
        *_attrs(v, "ordinal previous_version_id previous_version_digest"),
    )
    expected = (
        c.candidate_id,
        c.committed_admission_decision_id,
        c.semantic_scope_digest,
        1,
        None,
        None,
    )
    _require(
        actual == expected,
        "first Candidate Version differs from committed Candidate admission",
    )
    return v


def _one(items: list[object], field: str) -> str:
    values = {getattr(item, field) for item in items}
    if len(values) != 1:
        raise _Error(f"Candidate manifests disagree on {field}")
    return values.pop()


def _union(items: object, values: object) -> tuple[str, ...]:
    return tuple(sorted({value for item in items for value in values(item)}))  # type: ignore[operator,union-attr]


def _validate_contributing_dispositions(
    contributing: set[str], dispositions: tuple[ProposalDisposition, ...]
) -> None:
    decision_leads = {
        disposition.lead_head.decision_lead_id for disposition in dispositions
    }
    if decision_leads != contributing or len(decision_leads) != len(dispositions):
        raise _Error("every contributing Lead requires one exact disposition source")


def _project(value: object) -> str:
    return _hash(_json(value))


def _derive_governing_state(
    gates, leads, signals, dispositions, version, assessment, lineage
):
    gates = tuple(sorted(gates, key=lambda item: str(item.request.decision_id)))
    leads = tuple(sorted(leads, key=lambda item: str(item.request.lead_id)))
    signals = tuple(sorted(signals, key=lambda item: str(item.request.signal_id)))
    dispositions = tuple(sorted(dispositions, key=lambda item: item.disposition_id))
    values = tuple(
        [item.request.canonical_value() for item in group]
        for group in (gates, leads, signals)
    )
    gate_values, lead_values, signal_values = values
    policies = _POLICIES.split()
    projections = (
        [
            [item["coverage"] for item in gate_values],
            [item["coverage"] for item in lead_values],
        ],
        [
            [[item[key] for key in policies] for item in gate_values],
            [item["lead_policy"] for item in lead_values],
            [item["admission_policy"] for item in signal_values],
        ],
        [
            [
                item["rights_decision_id"],
                item["rights_policy_version"],
                item["basis"]["rights_current"],
            ]
            for item in gate_values
        ],
        [
            [item.canonical_digest for item in group]
            for group in (signals, leads, gates)
        ],
        [_hash(item.canonical_bytes) for item in dispositions],
        [
            [item.retrieval_context_id, item.retrieval_context_digest]
            for item in dispositions
        ],
        [
            version.canonical_digest,
            assessment.canonical_digest,
            [item.canonical_digest for item in lineage.history],
        ],
        [CANDIDATE_ADMISSION, CANDIDATE_CURRENT_VERSION],
        sorted(
            {
                value
                for item in dispositions
                for value in item.route_binding.candidate_manifest.governing_versions
            }
        ),
    )
    return CandidateGoverningStateBinding(*map(_project, projections))


def _validate_relationship_route(version, assessment) -> None:
    expected = _enum(
        CanonicalOutcome,
        f"REL_{version.proposed_relationship.value}",
        "D1 relationship",
    )
    if assessment.status is not AssessmentStatus.COMPLETE:
        if assessment.comparator is not None:
            raise _Error("invalid Candidate contract")
        return
    if assessment.decision is not expected:
        raise _Error("invalid Candidate contract")
    comparator = (
        None
        if assessment.comparator is None
        else (
            assessment.comparator.hypothesis_id,
            assessment.comparator.version_id,
            assessment.comparator.version_digest,
        )
    )
    target = (
        version.proposed_target_hypothesis_id,
        version.target_version_id,
        version.target_version_digest,
    )
    if (expected is CanonicalOutcome.REL_NO_ADEQUATE_PRIOR_MATCH) != (
        comparator is None
    ):
        raise _Error("invalid Candidate contract")
    if comparator != (None if target == (None, None, None) else target):
        raise _Error("invalid Candidate contract")


def build_candidate_distinct_scope_proof(
    *,
    proposed_manifest: CandidateGoverningManifest,
    proposed_collision: CurrentCollisionEligibilityDecision,
    comparator_collision: CurrentCollisionEligibilityDecision,
    comparator_version: StoryCandidateVersion,
) -> CandidateDistinctScopeProof:
    def build() -> CandidateDistinctScopeProof:
        p, c, v = proposed_collision, comparator_collision, comparator_version
        _valid(
            all(type(item) is CurrentCollisionEligibilityDecision for item in (p, c))
            and type(v) is StoryCandidateVersion
        )
        pb, cb = p.request.binding, c.request.binding
        context = lambda item: _attrs(item, _CONTEXT.split())

        def historical_collision_matches(manifest, decision):
            return _attrs(
                manifest,
                "collision_request_digest collision_decision_digest collision_namespace collision_key_digest hypothesis_id hypothesis_version_id hypothesis_version_digest",
            ) == (
                decision.request.request_digest,
                _hash(decision.canonical_bytes),
                *_attrs(
                    decision.request.binding,
                    "collision_namespace collision_key_digest subject_id subject_version_id subject_version_digest",
                ),
            )

        _valid(
            not (
                not historical_collision_matches(proposed_manifest, p)
                or _attrs(
                    v.governing_manifest, "collision_namespace collision_key_digest"
                )
                != _attrs(cb, "collision_namespace collision_key_digest")
                or proposed_manifest.relationship_outcome
                is not CanonicalOutcome.REL_NO_ADEQUATE_PRIOR_MATCH
                or p.outcome is not CollisionEligibilityOutcome.ELIGIBLE
                or pb.operation is not CandidateUseOperation.ADMIT_NEW_CANDIDATE
                or (p.collision_state is not CollisionState.UNOCCUPIED)
                or (c.outcome is not CollisionEligibilityOutcome.ELIGIBLE)
                or (cb.operation is not CandidateUseOperation.USE_CURRENT_CANDIDATE)
                or (c.collision_state is not CollisionState.OCCUPIED)
                or (cb.expected_candidate_id != v.candidate_id)
                or (c.observed_candidate_id != v.candidate_id)
                or (
                    (cb.subject_id, cb.subject_version_id, cb.subject_version_digest)
                    != (
                        v.governing_manifest.hypothesis_id,
                        v.governing_manifest.hypothesis_version_id,
                        v.governing_manifest.hypothesis_version_digest,
                    )
                )
                or (pb.collision_key_digest == cb.collision_key_digest)
                or (pb.subject_id == cb.subject_id)
                or (
                    proposed_manifest.semantic_scope_digest
                    == v.governing_manifest.semantic_scope_digest
                )
                or (context(pb) != context(cb))
                or p.trusted_context.context_digest != c.trusted_context.context_digest
            )
        )
        return CandidateDistinctScopeProof(
            _hash(p.canonical_bytes),
            _hash(c.canonical_bytes),
            v.candidate_id,
            v.version_id,
            v.canonical_digest,
            v.governing_manifest.semantic_scope_digest,
            p.trusted_context.context_digest,
        )

    return _normalise(build, "distinct-scope proof failed closed")


@_total("invalid Candidate contract")
def evaluate_candidate_admission(
    *,
    request: CandidateAdmissionRequest,
    manifest: CandidateGoverningManifest,
    collision: CurrentCollisionEligibilityDecision,
    current_version: StoryCandidateVersion | None,
    governing_state: CandidateGoverningState,
    comparator_collision: CurrentCollisionEligibilityDecision | None = None,
    comparator_version: StoryCandidateVersion | None = None,
) -> CandidateAdmission:
    r, m, c, v, state = request, manifest, collision, current_version, governing_state
    _valid(
        type(r) is CandidateAdmissionRequest and type(m) is CandidateGoverningManifest
    )
    _valid(type(c) is CurrentCollisionEligibilityDecision)
    _valid(v is None or type(v) is StoryCandidateVersion)
    _valid(type(state) is CandidateGoverningState)
    binding = c.request.binding
    _valid(
        not (
            r.semantic_scope_digest != m.semantic_scope_digest
            or r.collision_request_digest != c.request.request_digest
            or m.collision_request_digest != c.request.request_digest
            or (m.collision_decision_digest != _hash(c.canonical_bytes))
            or (binding.subject_id != m.hypothesis_id)
            or (binding.subject_version_id != m.hypothesis_version_id)
            or (binding.subject_version_digest != m.hypothesis_version_digest)
            or (binding.collision_namespace != m.collision_namespace)
            or (binding.collision_key_digest != m.collision_key_digest)
        )
    )
    current = () if v is None else (v.candidate_id, v.version_id, v.canonical_digest)

    def result(outcome, reason, proof=None):
        return CandidateAdmission(
            r, m, outcome, reason, *(current or (None, None, None)), proof
        )

    O, R = CandidateAdmissionOutcome, CandidateAdmissionReason
    if v is None and r.expected_current_ordinal != 0:
        return result(O.STALE, R.CANDIDATE_CAS_STALE)
    if v is not None and (
        r.expected_current_ordinal != v.ordinal
        or r.expected_current_version_id != v.version_id
        or r.expected_current_version_digest != v.canonical_digest
    ):
        return result(O.STALE, R.CANDIDATE_CAS_STALE)
    if state.status in {
        CandidateGoverningStateStatus.UNAVAILABLE,
        CandidateGoverningStateStatus.INCOMPLETE,
    }:
        return result(O.INCOMPLETE, R.GOVERNING_EVIDENCE_INCOMPLETE)
    if state.status is CandidateGoverningStateStatus.BLOCKED:
        return result(O.BLOCKED, R.GOVERNING_STATE_BLOCKED)
    if (
        state.binding != m.governing_state_binding
        or r.expected_governing_state_digest
        != m.governing_state_binding.canonical_digest
    ):
        return result(O.STALE, R.GOVERNING_STATE_STALE)
    if c.outcome is CollisionEligibilityOutcome.UNAVAILABLE:
        return result(O.INCOMPLETE, R.COLLISION_AUTHORITY_UNAVAILABLE)
    if m.incomplete or c.outcome is CollisionEligibilityOutcome.INCOMPLETE:
        return result(O.INCOMPLETE, R.GOVERNING_EVIDENCE_INCOMPLETE)
    if c.outcome is CollisionEligibilityOutcome.STALE:
        return result(O.STALE, R.COLLISION_AUTHORITY_STALE)
    if (
        c.outcome is not CollisionEligibilityOutcome.ELIGIBLE
        or m.relationship_outcome is not _ADMISSIBLE_RELATIONSHIP[m.candidate_kind]
    ):
        return result(O.BLOCKED, R.COLLISION_AUTHORITY_BLOCKED)
    if binding.operation is CandidateUseOperation.ADMIT_NEW_CANDIDATE and v is None:
        if r.distinct_scope_proof_digest is None:
            return result(O.ADMISSIBLE, R.NEW_CANDIDATE_PRE_EFFECT)
        proof = build_candidate_distinct_scope_proof(
            proposed_manifest=m,
            proposed_collision=c,
            comparator_collision=comparator_collision,
            comparator_version=comparator_version,
        )
        _valid(proof.canonical_digest == r.distinct_scope_proof_digest)
        return result(O.DISTINCT, R.RELATED_DISTINCT_PRE_EFFECT, proof)
    if (
        binding.operation is not CandidateUseOperation.USE_CURRENT_CANDIDATE
        or v is None
    ):
        return result(O.BLOCKED, R.COLLISION_OPERATION_CAS_DIFFERS)
    if (
        v.governing_manifest.semantic_scope_digest != m.semantic_scope_digest
        or binding.expected_candidate_id != v.candidate_id
        or c.observed_candidate_id != v.candidate_id
    ):
        return result(O.BLOCKED, R.COLLISION_OPERATION_CAS_DIFFERS)
    if v.governing_manifest.version_material_digest == m.version_material_digest:
        return result(O.DUPLICATE_EQUIVALENT, R.EXACT_MANIFEST_REPLAY)
    return result(O.ADMISSIBLE, R.SUCCESSOR_VERSION_PRE_EFFECT)


# fmt: off
def _candidate_command_canonicalizer(value: object) -> bytes:
    raw = _normalise(lambda: _json(value), "Candidate command cannot be canonicalised")
    document = _exact(_decode(raw), set(_COMMAND_FIELDS.split()), "Candidate command")
    _require(document["schema_version"] == CANDIDATE_COMMAND_SCHEMA, "Candidate command schema differs")
    request = CandidateAdmissionRequest.from_value(document["request"])
    collision = CurrentCollisionEligibilityRequest.from_mapping(document["collision_request"])
    comparator_request = (None if document["comparator_collision_request"] is None
        else CurrentCollisionEligibilityRequest.from_mapping(document["comparator_collision_request"]))
    nested = document["admission"], document["collision_decision"], document["effect_identity"]
    if all(item is None for item in nested):
        _require(comparator_request is None and document["comparator_collision_decision"] is None
            and request.collision_request_digest == collision.request_digest, "Candidate pre-effect payload differs")
        return raw
    _require(all(item is not None for item in nested), "Candidate effect payload is partial")
    admission = CandidateAdmission.from_canonical_bytes(_json(document["admission"]))
    decision = CurrentCollisionEligibilityDecision.from_canonical_bytes(_json(document["collision_decision"]))
    comparator = (None if document["comparator_collision_decision"] is None
        else CurrentCollisionEligibilityDecision.from_canonical_bytes(_json(document["comparator_collision_decision"])))
    effect = _exact(document["effect_identity"], set(_EFFECT_FIELDS.split()), "Candidate effect identity")
    for name in ("candidate_id", "committed_admission_decision_id", "version_id"): _uuid(effect[name], name)
    _ordinal(effect["version_ordinal"], "version_ordinal"); dispositions = document["disposition_ids"]
    _valid(type(dispositions) is list and all(type(item) is str for item in dispositions)
        and dispositions == sorted(set(dispositions)))
    manifest = admission.governing_manifest
    _valid(admission.request == request and decision.request == collision
        and (comparator is None) == (comparator_request is None)
        and (comparator is None or comparator.request == comparator_request)
        and request.collision_request_digest == collision.request_digest
        and document["hypothesis_version_id"] == manifest.hypothesis_version_id
        and document["relationship_assessment_digest"] == manifest.relationship_assessment_digest)
    return raw


_CANDIDATE_COMMAND_VECTOR = b'{"admission":null,"collision_decision":null,"collision_request":{"binding":{"authority_watermark":0,"collision_key_digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000","collision_namespace":"candidate-vector","expected_candidate_id":null,"generation_id":"generation-vector","operation":"ADMIT_NEW_CANDIDATE","query_valid_time":"2042-01-01T00:00:00Z","serving_time":"2042-01-01T00:00:00Z","subject_id":"00000000-0000-4000-8000-000000000001","subject_version_digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000","subject_version_id":"00000000-0000-4000-8000-000000000001"},"named_request_digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000"},"comparator_collision_decision":null,"comparator_collision_request":null,"disposition_ids":[],"effect_identity":null,"hypothesis_version_id":"00000000-0000-4000-8000-000000000001","relationship_assessment_digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000","request":{"actor_identity_digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000","collision_request_digest":"sha256:a9617eb9457d3d5f75ce67227dce71b530831bad1adf1325791bee22ccc7e053","distinct_scope_proof_digest":null,"expected_current_ordinal":0,"expected_current_version_digest":null,"expected_current_version_id":null,"expected_governing_state_digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000","idempotency_key":"candidate:vector","request_id":"00000000-0000-4000-8000-000000000001","semantic_scope_digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000"},"schema_version":"newsroom.increment6.candidate-admission-command.v1"}'


def _candidate_payload_contract() -> PayloadSchemaContract:
    value = json.loads(_CANDIDATE_COMMAND_VECTOR)
    vector = PayloadGoldenVector("candidate-command", "candidate:command", value, _CANDIDATE_COMMAND_VECTOR)
    return PayloadSchemaContract(CANDIDATE_COMMAND_SCHEMA, PayloadMode.INLINE, "candidate-command-schema-v1",
        "candidate-command-json-v1", _candidate_command_canonicalizer, (vector,))


def candidate_command_definition() -> CommandDefinition:
    contract = _candidate_payload_contract()
    return CommandDefinition(command_type=CANDIDATE_COMMAND_TYPE, definition_version="candidate-command-v1",
        aggregate_type="story_candidate_admission", event_type="story_candidate_admitted", event_schema_version=1,
        payload_mode=PayloadMode.INLINE, payload_schema_version=contract.schema_version,
        payload_schema_contract_version=contract.contract_version, payload_schema_contract_digest=contract.contract_digest,
        payload_canonicalizer_version=contract.canonicalizer_implementation_version, trust_scope=TrustScope.ADMITTED,
        security_scope="authority.story-candidate", retention_scope="authority.audit",
        required_scope="authority.story-candidate.admit", max_inline_bytes=MAX_CANDIDATE_COMMAND_BYTES)


def merge_candidate_authority_registries(commands: CommandRegistry, schemas: PayloadSchemaRegistry) -> tuple[CommandRegistry, PayloadSchemaRegistry]:
    definition, contract = candidate_command_definition(), _candidate_payload_contract()
    definitions, contracts = tuple(commands.definitions()), tuple(schemas.contracts())
    command_matches = tuple(item for item in definitions if item.command_type == CANDIDATE_COMMAND_TYPE)
    schema_matches = tuple(item for item in contracts if (item.schema_version, item.payload_mode) == (CANDIDATE_COMMAND_SCHEMA, PayloadMode.INLINE))
    if command_matches not in ((), (definition,)) or schema_matches not in ((), (contract,)):
        raise CandidateContractError("Candidate authority registry conflicts")
    definitions += () if command_matches else (definition,); contracts += () if schema_matches else (contract,)
    return (CommandRegistry(definitions, current_versions={item.command_type: (definition.definition_version if item.command_type == CANDIDATE_COMMAND_TYPE else commands.resolve(item.command_type).definition_version) for item in definitions}),
        PayloadSchemaRegistry(contracts, current_versions={(item.schema_version, item.payload_mode): (contract.contract_version if (item.schema_version, item.payload_mode) == (CANDIDATE_COMMAND_SCHEMA, PayloadMode.INLINE) else schemas.resolve(item.schema_version, item.payload_mode).contract_version) for item in contracts}))


_FACADE_TOKEN = object()
_READ_PORT_TOKEN = object()


class StoryCandidateReadPort:
    """Narrow transaction-bound read seam for retained Candidate authority."""

    __slots__ = ("__authority", "__bounded_version")

    def __init__(
        self,
        token: object,
        authority: object,
        bounded_version: Callable[[str], StoryCandidateVersion] | None = None,
    ) -> None:
        if token is not _READ_PORT_TOKEN:
            raise CandidateContractError(
                "Candidate read port construction is authority-private"
            )
        object.__setattr__(self, "_StoryCandidateReadPort__authority", authority)
        if bounded_version is not None and not callable(bounded_version):
            raise CandidateContractError("Candidate bounded reader differs")
        object.__setattr__(
            self, "_StoryCandidateReadPort__bounded_version", bounded_version
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("StoryCandidateReadPort is immutable")

    def _call(self, name: str, identity: str | None, expected: type):
        message = f"Candidate {name} returned a forged result"
        value = _normalise(
            lambda: getattr(self.__authority, name)(identity)
            if identity is not None
            else getattr(self.__authority, name)(),
            message,
        )
        _require(type(value) is expected, message)
        return value

    def verify_retained_integrity_in_transaction(self) -> None:
        self._call("verify_retained_integrity_in_transaction", None, type(None))

    def require_retained_candidate_in_transaction(
        self, candidate_id: str
    ) -> StoryCandidate:
        return self._call(
            "require_retained_candidate_in_transaction",
            candidate_id,
            StoryCandidate,
        )

    def require_retained_version_in_transaction(
        self, version_id: str
    ) -> StoryCandidateVersion:
        return self._call(
            "require_retained_version_in_transaction",
            version_id,
            StoryCandidateVersion,
        )

    def require_retained_versions_in_transaction(
        self, version_ids: tuple[str, ...],
    ) -> tuple[StoryCandidateVersion | None, ...]:
        message = "Candidate retained version partition differs"
        _require(type(version_ids) is tuple and all(
            type(value) is str and value.strip() for value in version_ids
        ), message)
        values = _normalise(
            lambda: self.__authority.require_retained_versions_in_transaction(version_ids),
            message,
        )
        _require(type(values) is tuple and len(values) == len(version_ids), message)
        _require(all(value is None or (
            type(value) is StoryCandidateVersion and value.version_id == identity
        ) for identity, value in zip(version_ids, values, strict=True)), message)
        return values

    def require_retained_version(self, version_id: str) -> StoryCandidateVersion:
        if self.__bounded_version is None:
            return self.require_retained_version_in_transaction(version_id)
        message = "Candidate require_retained_version returned a forged result"
        value = _normalise(lambda: self.__bounded_version(version_id), message)
        _require(type(value) is StoryCandidateVersion, message)
        return value

    def _with_bounded_version(
        self, reader: Callable[[str], StoryCandidateVersion]
    ) -> StoryCandidateReadPort:
        return StoryCandidateReadPort(
            _READ_PORT_TOKEN, self.__authority, bounded_version=reader
        )

    def require_current_head_in_transaction(
        self, candidate_id: str, *, proof: object
    ) -> StoryCandidateVersion:
        message = "Candidate require_current_head_in_transaction returned a forged result"
        value = _normalise(
            lambda: self.__authority.require_current_head_in_transaction(
                candidate_id, proof=proof
            ),
            message,
        )
        _require(type(value) is StoryCandidateVersion, message)
        return value


def _compose_story_candidate_read_port(
    authority: object,
    *,
    bounded_version: Callable[[str], StoryCandidateVersion] | None = None,
) -> StoryCandidateReadPort:
    """Private constructor used only by the checked Candidate authority."""

    return StoryCandidateReadPort(
        _READ_PORT_TOKEN, authority, bounded_version=bounded_version
    )


class StoryCandidateAuthority:
    __slots__ = ("__authority",)

    def __init__(self, token: object, authority: object) -> None:
        if token is not _FACADE_TOKEN: raise CandidateContractError("Candidate authority construction is private")
        self.__authority = authority

    def _call(self, name: str, *args: object, expected: type, **kwargs: object):
        message = f"Candidate {name} returned a forged result"
        value = _normalise(lambda: getattr(self.__authority, name)(*args, **kwargs), message)
        _require(type(value) is expected, message); return value

    def admit(self, admission_bytes: bytes, *, collision_request: CurrentCollisionEligibilityRequest,
        proof: object, comparator_collision_request: CurrentCollisionEligibilityRequest | None = None) -> StoryCandidateVersion:
        return self._call("admit", admission_bytes, collision_request=collision_request, proof=proof,
            comparator_collision_request=comparator_collision_request, expected=StoryCandidateVersion)

    def load_version(self, version_id: str) -> StoryCandidateVersion:
        return self._call("load_version", version_id, expected=StoryCandidateVersion)

    def load_candidate(self, candidate_id: str) -> StoryCandidate:
        return self._call("load_candidate", candidate_id, expected=StoryCandidate)

    def versions(self, candidate_id: str) -> tuple[StoryCandidateVersion, ...]:
        value = _normalise(lambda: self.__authority.versions(candidate_id), "Candidate history failed")
        _require(type(value) is tuple and all(type(item) is StoryCandidateVersion for item in value), "Candidate history is forged")
        return value

    def _exact_current_producers(
        self, candidate_id: str, *, proof: object
    ) -> tuple[StoryCandidateVersion, EventHypothesisVersion]:
        value = _normalise(
            lambda: self.__authority.exact_current_producers(
                candidate_id, proof=proof
            ),
            "Candidate exact current producer read failed",
        )
        _require(
            type(value) is tuple
            and len(value) == 2
            and type(value[0]) is StoryCandidateVersion
            and type(value[1]) is EventHypothesisVersion,
            "Candidate exact current producer read is forged",
        )
        return value

    def _exact_associated_current_producers(
        self, candidate_id: str, *, proof: object
    ) -> tuple[StoryCandidateVersion, EventHypothesisVersion]:
        value = _normalise(
            lambda: self.__authority.exact_associated_current_producers(
                candidate_id, proof=proof
            ),
            "Candidate associated current producer read failed",
        )
        _require(
            type(value) is tuple
            and len(value) == 2
            and type(value[0]) is StoryCandidateVersion
            and type(value[1]) is EventHypothesisVersion
            and value[0].governing_manifest.hypothesis_id
            == value[1].hypothesis_id,
            "Candidate associated current producer read is forged",
        )
        return value

    def _exact_current_candidate(self, candidate_id: str) -> StoryCandidateVersion:
        return self._call(
            "exact_current_candidate", candidate_id, expected=StoryCandidateVersion
        )

    def current(self, candidate_id: str, *, collision_request: CurrentCollisionEligibilityRequest, proof: object) -> StoryCandidateVersion:
        return self._call("current", candidate_id, collision_request=collision_request, proof=proof, expected=StoryCandidateVersion)

    require_current = current

    def close(self) -> None: _normalise(self.__authority.close, "Candidate authority close failed")
    def __enter__(self) -> Self: return self
    def __exit__(self, *_: object) -> None: self.close()


def _compose_story_candidate_authority(authority: object) -> StoryCandidateAuthority:
    return StoryCandidateAuthority(_FACADE_TOKEN, authority)


def open_story_candidate_authority(database: str | Path, *, retrieval_authority: object, authenticator: object,
    authorizer: object, command_registry: CommandRegistry, payload_schemas: PayloadSchemaRegistry,
    collision_enforcer: object, clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
    busy_timeout_ms: int = 5000) -> StoryCandidateAuthority:
    from newsroom.authority.story_candidate_system import (
        open_story_candidate_authority_system,
    )
    try: result = open_story_candidate_authority_system(database, retrieval_authority=retrieval_authority,
        authenticator=authenticator, authorizer=authorizer, command_registry=command_registry,
        payload_schemas=payload_schemas, collision_enforcer=collision_enforcer, clock=clock, busy_timeout_ms=busy_timeout_ms)
    except CandidateContractError: raise
    except Exception as exc: raise CandidateContractError("Candidate authority open failed") from exc
    if type(result) is StoryCandidateAuthority: return result
    close = getattr(result, "close", None)
    if callable(close):
        try: close()
        except BaseException: pass  # noqa: BLE001, S110 - preserve forged result failure
    raise CandidateContractError("Candidate authority opener returned forged facade")
# fmt: on


__all__ = tuple(_PUBLIC.split(",")) + ("StoryCandidateReadPort",)
