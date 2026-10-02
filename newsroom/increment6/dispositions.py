"""Proposal validation and the v19 disposition-authority persistence seam."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, Self

from newsroom.authority.auth import AuthenticationProof, StaticAuthenticator
from newsroom.authority.types import UtcTimestamp

from newsroom.authority.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    digest_bytes,
    validate_sha256_digest,
)
from newsroom.increment6.outcomes import (
    CanonicalNextAction,
    CanonicalOutcome,
    DecisionTerminality,
    OutcomeSelection,
    ReasonCode,
    SUPPLEMENTAL_ACTION_MAPPING,
    WATCH_CONDITION_MAPPING,
)
from newsroom.increment6.proposals import (
    CitationSourceKind,
    LeadRecommendation,
    ProposalRoute,
    TriageProposal,
)
from newsroom.increment6.work_items import (
    DecisionLeadBinding,
    RetrievalBindingState,
    RetrievalContextAuthority,
    TriageWorkItemStore,
    TriageWorkItemVersion,
)


PROPOSAL_VALIDATION_FINDING_SCHEMA_VERSION = (
    "newsroom.increment6.triage-proposal-finding.v1"
)
PROPOSAL_DISPOSITION_SCHEMA_VERSION = (
    "newsroom.increment6.triage-proposal-disposition.v1"
)
PROPOSAL_VALIDATION_FINDING = PROPOSAL_VALIDATION_FINDING_SCHEMA_VERSION
PROPOSAL_DISPOSITION = PROPOSAL_DISPOSITION_SCHEMA_VERSION
VALIDATED_PROPOSAL_LEAD_DISPOSITION_BINDING = (
    "EXACT_PROPOSAL_FINDING_SET_AND_CURRENT_LEAD_HEAD"
)


def _verification_snapshot(connection: sqlite3.Connection) -> tuple[int, int, int]:
    return (
        connection.total_changes,
        int(connection.execute("PRAGMA data_version").fetchone()[0]),
        int(connection.execute("PRAGMA schema_version").fetchone()[0]),
    )


def _require_verification_snapshot(
    connection: sqlite3.Connection, expected: tuple[int, int, int]
) -> None:
    if _verification_snapshot(connection) != expected:
        raise RuntimeError("native semantic verification snapshot changed")


_FINDING_SET_SCHEMA_VERSION = "newsroom.increment6.triage-proposal-finding-set.v1"
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\-]{0,255}\Z")
# #356 has no public raw-byte constant. Its closed field bounds permit up to
# 32 recommendations, each with 64 citations and bounded 32-item text lists.
# Canonical JSON can expand each legal UTF-8 control byte to a six-byte escape,
# so the complete maximum producer envelope is materially larger than the
# ordinary-text equivalent. Sixty-four MiB covers that escaped 32-Lead envelope
# plus this module's finding and disposition wrappers; consumers must not impose
# the former private 1 MiB limit, which was smaller than a valid eight-Lead
# Proposal.
MAX_DISPOSITION_CANONICAL_BYTES = 67_108_864
_MAX_JSON_DEPTH = 64
_MAX_FINDINGS = 64
_VALIDATOR_ID = "triage-proposal-validator"
_VALIDATOR_VERSION = "1"
_RULESET_ID = "triage-proposal-disposition"
_RULESET_VERSION = "1"
_RULESET_DIGEST = digest_bytes(canonical_json_bytes({
    "ruleset_id": _RULESET_ID,
    "ruleset_version": _RULESET_VERSION,
    "binding": VALIDATED_PROPOSAL_LEAD_DISPOSITION_BINDING,
}))

# Bind trusted implementation functions once.  Public instance/class method
# substitution must not turn an untrusted duck type into an authority provider.
_AUTHENTICATE = StaticAuthenticator.authenticate
_RETRIEVAL_ATTACH = RetrievalContextAuthority.attach
_RETRIEVAL_VERIFY = RetrievalContextAuthority.verify
_RETRIEVAL_VERIFY_RETAINED = RetrievalContextAuthority.verify_retained_integrity
_WORK_ITEM_REQUIRE_CURRENT = TriageWorkItemStore.require_usable_current_in_transaction


class DispositionContractError(ValueError):
    """Proposal validation or a pending disposition failed closed."""


@dataclass(frozen=True, slots=True)
class CurrentCandidateCitation:
    """Immutable authenticated observation of one retained Candidate head."""

    citation_id: str
    candidate_id: str
    candidate_version_id: str
    candidate_version_digest: str
    hypothesis_id: str
    hypothesis_version_id: str
    hypothesis_version_digest: str
    collision_namespace: str
    collision_key_digest: str
    source_definition_id: str
    source_item_id: str
    retrieval_context_digest: str
    authorization_receipt_digest: str
    authorization_decision_id: str

    def __post_init__(self) -> None:
        if type(self) is not CurrentCandidateCitation:
            raise DispositionContractError("Candidate citation must be exact typed")
        for field in (
            "citation_id", "candidate_id", "candidate_version_id", "hypothesis_id",
            "hypothesis_version_id", "collision_namespace", "source_definition_id",
            "source_item_id", "authorization_decision_id",
        ):
            value = getattr(self, field)
            if type(value) is not str or not value:
                raise DispositionContractError(f"Candidate citation {field} is required")
        for field in (
            "candidate_version_digest", "hypothesis_version_digest",
            "collision_key_digest", "retrieval_context_digest",
            "authorization_receipt_digest",
        ):
            try:
                validate_sha256_digest(getattr(self, field), field=field)
            except (TypeError, ValueError) as exc:
                raise DispositionContractError(
                    f"Candidate citation {field} differs"
                ) from exc
        if self.citation_id != f"current-candidate:{self.canonical_digest}":
            raise DispositionContractError("Candidate citation identity differs")

    def canonical_value(self) -> dict[str, object]:
        return {
            "schema_version": "newsroom.increment6.current-candidate-citation.v1",
            **{
                field: getattr(self, field)
                for field in self.__dataclass_fields__
                if field != "citation_id"
            },
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_value())

    @property
    def canonical_digest(self) -> str:
        return digest_bytes(self.canonical_bytes)

    @classmethod
    def create(cls, **values: str) -> "CurrentCandidateCitation":
        raw = canonical_json_bytes({
            "schema_version": "newsroom.increment6.current-candidate-citation.v1",
            **values,
        })
        return cls(f"current-candidate:{digest_bytes(raw)}", **values)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "CurrentCandidateCitation":
        value = _decode(raw, field="current Candidate citation")
        fields = set(cls.__dataclass_fields__) - {"citation_id"}
        if (
            set(value) != {"schema_version", *fields}
            or value.pop("schema_version")
            != "newsroom.increment6.current-candidate-citation.v1"
        ):
            raise DispositionContractError("current Candidate citation fields differ")
        return cls.create(**value)  # type: ignore[arg-type]


_CURRENT_CANDIDATE_CITATION_PORT_TOKEN = object()


class CurrentCandidateCitationReadPort:
    """Read exact retained historical Candidate observations without recursion."""

    __slots__ = ("__require",)

    def __init__(
        self,
        token: object,
        require: Callable[[str, str], CurrentCandidateCitation],
    ) -> None:
        if token is not _CURRENT_CANDIDATE_CITATION_PORT_TOKEN or not callable(require):
            raise DispositionContractError("Candidate citation port is private")
        self.__require = require

    def require(self, citation_id: str, citation_digest: str) -> CurrentCandidateCitation:
        value = self.__require(citation_id, citation_digest)
        if (
            type(value) is not CurrentCandidateCitation
            or value.citation_id != citation_id
            or value.canonical_digest != citation_digest
        ):
            raise DispositionContractError("current Candidate citation differs")
        return value


def _create_current_candidate_citation_read_port(
    require: Callable[[str, str], CurrentCandidateCitation],
) -> CurrentCandidateCitationReadPort:
    return CurrentCandidateCitationReadPort(
        _CURRENT_CANDIDATE_CITATION_PORT_TOKEN, require
    )


class DispositionAuthority(StrEnum):
    """Phase-one authority states; neither state permits an effect."""

    NONE = "NONE"
    PENDING = "PENDING"


DISPOSITION_AUTHORITY = DispositionAuthority


class FindingSeverity(StrEnum):
    INFO = "INFO"
    ERROR = "ERROR"


class FindingCode(StrEnum):
    ROUTE_VALIDATED = "ROUTE_VALIDATED"
    OPERATIONAL_ACTION_AMBIGUOUS = "OPERATIONAL_ACTION_AMBIGUOUS"
    OPERATIONAL_ACTION_UNSUPPORTED = "OPERATIONAL_ACTION_UNSUPPORTED"


class DispositionJudgement(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    HOLD = "HOLD"
    ESCALATE = "ESCALATE"


class _NoEffect:
    @property
    def authorises_persistence(self) -> bool:
        return False

    @property
    def authorises_external_effect(self) -> bool:
        return False

    @property
    def authorises_publication(self) -> bool:
        return False

    @property
    def creates_hypothesis(self) -> bool:
        return False

    @property
    def creates_candidate(self) -> bool:
        return False


def _token(value: object, field: str) -> str:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise DispositionContractError(f"{field} must be a bounded canonical token")
    return value


def _digest(value: object, field: str) -> str:
    if type(value) is not str:
        raise DispositionContractError(f"{field} must be a canonical SHA-256 digest")
    try:
        return validate_sha256_digest(value, field=field)
    except CanonicalizationError as exc:
        raise DispositionContractError(str(exc)) from exc


def _exact(value: object, keys: set[str], field: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise DispositionContractError(f"{field} keys are not exact")
    try:
        actual_keys = set(value)
    except DispositionContractError:
        raise
    except Exception as exc:
        raise DispositionContractError(f"{field} keys are not exact") from exc
    if actual_keys != keys:
        raise DispositionContractError(f"{field} keys are not exact")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DispositionContractError(
                f"JSON integrity contains duplicate key: {key}"
            )
        result[key] = value
    return result


def _decode(raw: bytes, *, field: str) -> dict[str, object]:
    if type(raw) is not bytes:
        raise DispositionContractError(f"{field} integrity requires bounded bytes")
    if not raw or len(raw) > MAX_DISPOSITION_CANONICAL_BYTES:
        raise DispositionContractError(f"{field} integrity requires bounded bytes")
    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
        elif byte == ord('"'):
            in_string = True
        elif byte in (ord("{"), ord("[")):
            depth += 1
            if depth > _MAX_JSON_DEPTH:
                raise DispositionContractError(f"{field} integrity exceeds depth bound")
        elif byte in (ord("}"), ord("]")):
            depth -= 1
            if depth < 0:
                raise DispositionContractError(f"{field} integrity has invalid depth")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                DispositionContractError(f"unsupported JSON constant: {item}")
            ),
        )
    except DispositionContractError:
        raise
    except Exception as exc:
        raise DispositionContractError(f"{field} integrity is not valid JSON") from exc
    if type(value) is not dict:
        raise DispositionContractError(f"{field} integrity requires an object")
    try:
        if canonical_json_bytes(value) != raw:
            raise DispositionContractError(f"{field} integrity is not canonical JSON")
    except DispositionContractError:
        raise
    except Exception as exc:
        raise DispositionContractError(f"{field} integrity is outside canonical JSON") from exc
    return value


def _bounded_canonical(value: object, *, field: str) -> bytes:
    """Canonicalise untrusted values, totalising ordinary traversal failures."""

    try:
        raw = canonical_json_bytes(value)
    except DispositionContractError:
        raise
    except Exception as exc:
        raise DispositionContractError(f"{field} is outside canonical JSON") from exc
    if len(raw) > MAX_DISPOSITION_CANONICAL_BYTES:
        raise DispositionContractError(f"{field} exceeds the canonical byte bound")
    return raw


def _constructed_value(
    factory: Callable[[], object], *, field: str
) -> object:
    """Totalise ordinary factory failures while leaving BaseException visible."""
    try:
        return factory()
    except DispositionContractError:
        raise
    except Exception as exc:
        raise DispositionContractError(f"{field} is outside canonical JSON") from exc


def _bounded_canonical_from(
    factory: Callable[[], object], *, field: str
) -> bytes:
    """Canonicalise a lazily constructed value under the public error contract."""

    try:
        return _bounded_canonical(
            _constructed_value(factory, field=field),
            field=field,
        )
    except DispositionContractError:
        raise
    except Exception as exc:
        raise DispositionContractError(f"{field} is outside canonical JSON") from exc


@dataclass(frozen=True, slots=True)
class ValidatorInputBinding(_NoEffect):
    """An exact validator input claim, pending provider authentication."""

    validator_id: str
    validator_version: str
    authenticated_context_identity: str
    retrieval_request_id: str
    retrieval_request_digest: str
    retrieval_receipt_id: str
    retrieval_receipt_digest: str
    ruleset_id: str
    ruleset_version: str
    ruleset_digest: str
    input_digest: str
    authority: DispositionAuthority = DispositionAuthority.PENDING

    def __post_init__(self) -> None:
        if type(self) is not ValidatorInputBinding:
            raise DispositionContractError(
                "validator input constructor requires the exact contract type"
            )
        for field in (
            "validator_id", "validator_version", "retrieval_request_id",
            "retrieval_receipt_id", "ruleset_id", "ruleset_version",
        ):
            _token(getattr(self, field), field)
        for field in (
            "authenticated_context_identity", "retrieval_request_digest",
            "retrieval_receipt_digest", "ruleset_digest", "input_digest",
        ):
            _digest(getattr(self, field), field)
        if self.authority is not DispositionAuthority.PENDING:
            raise DispositionContractError("validator input authority must remain PENDING")

    def canonical_value(self) -> dict[str, object]:
        return {
            "validator_id": self.validator_id,
            "validator_version": self.validator_version,
            "authenticated_context_identity": self.authenticated_context_identity,
            "retrieval_request_id": self.retrieval_request_id,
            "retrieval_request_digest": self.retrieval_request_digest,
            "retrieval_receipt_id": self.retrieval_receipt_id,
            "retrieval_receipt_digest": self.retrieval_receipt_digest,
            "ruleset_id": self.ruleset_id,
            "ruleset_version": self.ruleset_version,
            "ruleset_digest": self.ruleset_digest,
            "input_digest": self.input_digest,
            "authority": self.authority.value,
        }

    def expected_input_digest(self, proposal_canonical_digest: str) -> str:
        """Derive the exact deterministic validation-envelope identity."""

        exact = _exact_validator_input(self, field="validator input binding")
        return exact._expected_input_digest(proposal_canonical_digest)

    def _expected_input_digest(self, proposal_canonical_digest: str) -> str:
        """Derive from an exact base-class binding without virtual dispatch."""

        _digest(proposal_canonical_digest, "proposal_canonical_digest")
        return digest_bytes(_bounded_canonical({
            "proposal_canonical_digest": proposal_canonical_digest,
            "validator_id": self.validator_id,
            "validator_version": self.validator_version,
            "authenticated_context_identity": self.authenticated_context_identity,
            "retrieval_request_id": self.retrieval_request_id,
            "retrieval_request_digest": self.retrieval_request_digest,
            "retrieval_receipt_id": self.retrieval_receipt_id,
            "retrieval_receipt_digest": self.retrieval_receipt_digest,
            "ruleset_id": self.ruleset_id,
            "ruleset_version": self.ruleset_version,
            "ruleset_digest": self.ruleset_digest,
        }, field="validator input envelope"))

    @classmethod
    def for_proposal(cls, *, proposal_bytes: bytes, **values: object) -> Self:
        """Create a pending binding whose input digest is derived, not asserted."""

        if cls is not ValidatorInputBinding:
            raise DispositionContractError(
                "validator input factory requires the exact contract type"
            )
        proposal_digest = digest_bytes(_bounded_canonical(
            _decode(proposal_bytes, field="proposal"), field="proposal"
        ))
        provisional = ValidatorInputBinding(
            input_digest="sha256:" + "0" * 64, **values  # type: ignore[arg-type]
        )
        return ValidatorInputBinding(
            input_digest=provisional._expected_input_digest(proposal_digest),
            **values,  # type: ignore[arg-type]
        )  # type: ignore[return-value]

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        if cls is not ValidatorInputBinding:
            raise DispositionContractError(
                "validator input parser requires the exact contract type"
            )
        item = _exact(value, {
            "validator_id", "validator_version", "authenticated_context_identity",
            "retrieval_request_id", "retrieval_request_digest", "retrieval_receipt_id",
            "retrieval_receipt_digest", "ruleset_id", "ruleset_version",
            "ruleset_digest", "input_digest", "authority",
        }, "validator input binding")
        authority = _constructed_value(
            lambda: item["authority"], field="validator input binding"
        )
        if authority != DispositionAuthority.PENDING.value:
            raise DispositionContractError("validator input authority must remain PENDING")
        values = _constructed_value(
            lambda: {
                key: item[key] for key in item if key != "authority"
            },
            field="validator input binding",
        )
        return _constructed_value(
            lambda: ValidatorInputBinding(**values),  # type: ignore[arg-type]
            field="validator input binding",
        )  # type: ignore[return-value]


def _exact_validator_input(
    value: ValidatorInputBinding, *, field: str
) -> ValidatorInputBinding:
    if type(value) is not ValidatorInputBinding:
        raise DispositionContractError(
            f"{field} is outside canonical JSON: exact contract type required"
        )
    canonical = _constructed_value(
        lambda: value.canonical_value(), field=field
    )
    _bounded_canonical(canonical, field=field)
    return _constructed_value(
        lambda: ValidatorInputBinding.from_mapping(canonical), field=field
    )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ProposalValidationFinding(_NoEffect):
    finding_id: str
    proposal_id: str
    proposal_content_identity: str
    proposal_canonical_digest: str
    code: FindingCode
    severity: FindingSeverity
    path: str
    evidence_reference_type: str
    evidence_reference_id: str
    evidence_reference_digest: str
    validator_input: ValidatorInputBinding
    authority: DispositionAuthority = DispositionAuthority.NONE

    def __post_init__(self) -> None:
        if type(self) is not ProposalValidationFinding:
            raise DispositionContractError(
                "finding constructor requires the exact contract type"
            )
        _digest(self.finding_id, "finding_id")
        _token(self.proposal_id, "proposal_id")
        _digest(self.proposal_content_identity, "proposal_content_identity")
        _digest(self.proposal_canonical_digest, "proposal_canonical_digest")
        if not isinstance(self.code, FindingCode) or not isinstance(self.severity, FindingSeverity):
            raise DispositionContractError("finding code and severity must be typed")
        _token(self.path, "finding path")
        _token(self.evidence_reference_type, "evidence reference type")
        _token(self.evidence_reference_id, "evidence reference id")
        _digest(self.evidence_reference_digest, "evidence reference digest")
        if not isinstance(self.validator_input, ValidatorInputBinding):
            raise DispositionContractError("finding validator input must be typed")
        exact_validator = _exact_validator_input(
            self.validator_input, field="validator input"
        )
        object.__setattr__(self, "validator_input", exact_validator)
        if exact_validator.input_digest != exact_validator._expected_input_digest(
            self.proposal_canonical_digest
        ):
            raise DispositionContractError(
                "finding validator input differs from the exact validation envelope"
            )
        if self.authority is not DispositionAuthority.NONE:
            raise DispositionContractError("a pure finding has no authority")
        if self.finding_id != digest_bytes(
            _bounded_canonical_from(
                lambda: self._identity_value(), field="finding identity"
            )
        ):
            raise DispositionContractError("finding identity differs")
        _bounded_canonical_from(lambda: self.canonical_value(), field="finding")

    def _identity_value(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "proposal_content_identity": self.proposal_content_identity,
            "proposal_canonical_digest": self.proposal_canonical_digest,
            "code": self.code.value,
            "severity": self.severity.value,
            "path": self.path,
            "evidence_reference_type": self.evidence_reference_type,
            "evidence_reference_id": self.evidence_reference_id,
            "evidence_reference_digest": self.evidence_reference_digest,
            "validator_input_binding": self.validator_input.canonical_value(),
            "authority": self.authority.value,
        }

    def canonical_value(self) -> dict[str, object]:
        finding = {"finding_id": self.finding_id, **self._identity_value()}
        return {"schema_version": PROPOSAL_VALIDATION_FINDING_SCHEMA_VERSION, "finding": finding}

    @property
    def canonical_bytes(self) -> bytes:
        if type(self) is not ProposalValidationFinding:
            raise DispositionContractError(
                "finding property requires the exact contract type"
            )
        raw = _bounded_canonical_from(
            lambda: self.canonical_value(), field="finding"
        )
        exact = ProposalValidationFinding.from_canonical_bytes(raw)
        return _bounded_canonical_from(
            lambda: exact.canonical_value(), field="finding"
        )

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> Self:
        if cls is not ProposalValidationFinding:
            raise DispositionContractError(
                "finding parser requires the exact contract type"
            )
        root = _exact(_decode(raw, field="finding"), {"schema_version", "finding"}, "finding document")
        if root["schema_version"] != PROPOSAL_VALIDATION_FINDING_SCHEMA_VERSION:
            raise DispositionContractError("finding schema version is unsupported")
        item = _exact(root["finding"], {
            "finding_id", "proposal_id", "proposal_content_identity", "proposal_canonical_digest",
            "code", "severity", "path", "evidence_reference_type", "evidence_reference_id",
            "evidence_reference_digest", "validator_input_binding", "authority",
        }, "finding")
        if item["authority"] != DispositionAuthority.NONE.value:
            raise DispositionContractError("a pure finding has no authority")
        try:
            result = ProposalValidationFinding(
                finding_id=item["finding_id"], proposal_id=item["proposal_id"],
                proposal_content_identity=item["proposal_content_identity"],
                proposal_canonical_digest=item["proposal_canonical_digest"],
                code=FindingCode(item["code"]), severity=FindingSeverity(item["severity"]),
                path=item["path"], evidence_reference_type=item["evidence_reference_type"],
                evidence_reference_id=item["evidence_reference_id"],
                evidence_reference_digest=item["evidence_reference_digest"],
                validator_input=ValidatorInputBinding.from_mapping(item["validator_input_binding"]),
            )
        except DispositionContractError:
            raise
        except Exception as exc:
            raise DispositionContractError("finding is malformed") from exc
        if _bounded_canonical_from(
            lambda: result.canonical_value(), field="finding"
        ) != raw:
            raise DispositionContractError("finding typed replay differs")
        return result  # type: ignore[return-value]


def _exact_proposal(value: TriageProposal, *, field: str) -> TriageProposal:
    if type(value) is not TriageProposal:
        raise DispositionContractError(
            f"{field} is outside canonical JSON: exact contract type required"
        )
    raw = _bounded_canonical_from(
        lambda: value.canonical_value(), field=field
    )
    return _constructed_value(
        lambda: TriageProposal.from_canonical_bytes(raw), field=field
    )  # type: ignore[return-value]


def _exact_finding(
    value: ProposalValidationFinding, *, field: str
) -> ProposalValidationFinding:
    if type(value) is not ProposalValidationFinding:
        raise DispositionContractError(
            f"{field} is outside canonical JSON: exact contract type required"
        )
    raw = _bounded_canonical_from(
        lambda: value.canonical_value(), field=field
    )
    return _constructed_value(
        lambda: ProposalValidationFinding.from_canonical_bytes(raw), field=field
    )  # type: ignore[return-value]


def _exact_findings(
    value: tuple[ProposalValidationFinding, ...], *, field: str
) -> tuple[ProposalValidationFinding, ...]:
    if type(value) is not tuple:
        raise DispositionContractError(
            f"{field} is outside canonical JSON: exact tuple required"
        )
    return tuple(_exact_finding(item, field="finding") for item in value)


@dataclass(frozen=True, slots=True)
class ProposalValidationResult(_NoEffect):
    proposal: TriageProposal
    validator_input: ValidatorInputBinding
    findings: tuple[ProposalValidationFinding, ...]
    finding_set_digest: str
    authority: DispositionAuthority = DispositionAuthority.NONE

    def __post_init__(self) -> None:
        if type(self) is not ProposalValidationResult:
            raise DispositionContractError(
                "validation result constructor requires the exact contract type"
            )
        finding_set_digest = _digest(
            self.finding_set_digest, "finding_set_digest"
        )
        object.__setattr__(self, "finding_set_digest", finding_set_digest)
        if not isinstance(self.proposal, TriageProposal):
            raise DispositionContractError("validation result proposal must be typed")
        if not isinstance(self.validator_input, ValidatorInputBinding):
            raise DispositionContractError(
                "validation result validator input must be typed"
            )
        if (
            type(self.findings) is not tuple
            or not 1 <= len(self.findings) <= _MAX_FINDINGS
        ):
            raise DispositionContractError("finding set must be complete and bounded")
        exact_proposal = _exact_proposal(self.proposal, field="proposal")
        exact_validator = _exact_validator_input(
            self.validator_input, field="validator input binding"
        )
        exact_findings = _exact_findings(self.findings, field="finding set")
        object.__setattr__(self, "proposal", exact_proposal)
        object.__setattr__(self, "validator_input", exact_validator)
        object.__setattr__(self, "findings", exact_findings)
        proposal_bytes = _bounded_canonical_from(
            lambda: exact_proposal.canonical_value(), field="proposal"
        )
        proposal_digest = digest_bytes(proposal_bytes)
        if any(
            item.proposal_id != self.proposal.proposal_id
            or item.proposal_content_identity != self.proposal.content_identity
            or item.proposal_canonical_digest != proposal_digest
            or item.validator_input != self.validator_input
            for item in self.findings
        ):
            raise DispositionContractError(
                "finding set differs from its exact Proposal or validator input"
            )
        if {
            item.evidence_reference_id for item in self.findings
        } != set(self.proposal.decision_lead_ids):
            raise DispositionContractError(
                "finding set does not cover every decision Lead exactly"
            )
        ids = tuple(item.finding_id for item in self.findings)
        if ids != tuple(sorted(set(ids))):
            raise DispositionContractError("findings must be sorted and deduplicated")
        if finding_set_digest != digest_bytes(self.canonical_bytes):
            raise DispositionContractError("finding set digest differs")
        if self.authority is not DispositionAuthority.NONE:
            raise DispositionContractError("pure validation has no authority")
        if len(self.canonical_bytes) > MAX_DISPOSITION_CANONICAL_BYTES:
            raise DispositionContractError("finding set exceeds the canonical byte bound")

    @property
    def canonical_bytes(self) -> bytes:
        if type(self) is not ProposalValidationResult:
            raise DispositionContractError(
                "validation result property requires the exact contract type"
            )

        def canonical_value() -> dict[str, object]:
            if self.authority is not DispositionAuthority.NONE:
                raise DispositionContractError("pure validation has no authority")
            proposal = _exact_proposal(self.proposal, field="proposal")
            validator = _exact_validator_input(
                self.validator_input, field="validator input binding"
            )
            findings = _exact_findings(self.findings, field="finding set")
            return {
                "schema_version": _FINDING_SET_SCHEMA_VERSION,
                "proposal_content_identity": proposal.content_identity,
                "validator_input_binding": validator.canonical_value(),
                "finding_ids": [item.finding_id for item in findings],
                "authority": self.authority.value,
            }

        return _bounded_canonical_from(canonical_value, field="finding set")


def _exact_validation_result(
    value: ProposalValidationResult, *, field: str
) -> ProposalValidationResult:
    if type(value) is not ProposalValidationResult:
        raise DispositionContractError(
            f"{field} is outside canonical JSON: exact contract type required"
        )
    return _constructed_value(
        lambda: ProposalValidationResult(
            value.proposal,
            value.validator_input,
            value.findings,
            value.finding_set_digest,
            value.authority,
        ),
        field=field,
    )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class LeadDispositionHeadBinding:
    decision_lead_id: str
    decision_lead_digest: str
    current_disposition_head_id: str
    current_disposition_head_digest: str

    def __post_init__(self) -> None:
        if type(self) is not LeadDispositionHeadBinding:
            raise DispositionContractError(
                "Lead head constructor requires the exact contract type"
            )
        _token(self.decision_lead_id, "decision_lead_id")
        _digest(self.decision_lead_digest, "decision_lead_digest")
        _token(self.current_disposition_head_id, "current_disposition_head_id")
        _digest(self.current_disposition_head_digest, "current_disposition_head_digest")

    def canonical_value(self) -> dict[str, object]:
        return {
            "decision_lead_id": self.decision_lead_id,
            "decision_lead_digest": self.decision_lead_digest,
            "current_disposition_head_id": self.current_disposition_head_id,
            "current_disposition_head_digest": self.current_disposition_head_digest,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        if cls is not LeadDispositionHeadBinding:
            raise DispositionContractError(
                "Lead head parser requires the exact contract type"
            )
        item = _exact(value, {"decision_lead_id", "decision_lead_digest", "current_disposition_head_id", "current_disposition_head_digest"}, "Lead head binding")
        return _constructed_value(
            lambda: LeadDispositionHeadBinding(**item),  # type: ignore[arg-type]
            field="Lead head binding",
        )  # type: ignore[return-value]


def _exact_lead_head(
    value: LeadDispositionHeadBinding, *, field: str
) -> LeadDispositionHeadBinding:
    if type(value) is not LeadDispositionHeadBinding:
        raise DispositionContractError(
            f"{field} is outside canonical JSON: exact contract type required"
        )
    canonical = _constructed_value(
        lambda: value.canonical_value(), field=field
    )
    _bounded_canonical(canonical, field=field)
    return _constructed_value(
        lambda: LeadDispositionHeadBinding.from_mapping(canonical), field=field
    )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class _RouteRule:
    judgement: DispositionJudgement
    outcome: CanonicalOutcome
    reasons: frozenset[ReasonCode]
    actions: frozenset[tuple[DecisionTerminality, CanonicalNextAction]]


_ROUTE_RULES: Mapping[ProposalRoute, _RouteRule] = MappingProxyType({
    ProposalRoute.EDITORIAL_REJECT: _RouteRule(DispositionJudgement.REJECT, CanonicalOutcome.LEAD_EDITORIAL_REJECT, frozenset({ReasonCode.NOVELTY_EXACT_DUPLICATE, ReasonCode.NOVELTY_SAME_STATE_REPEAT, ReasonCode.SCOPE_EXCLUDED, ReasonCode.UTILITY_EDITORIAL_MATERIALITY}), frozenset({(DecisionTerminality.TERMINAL_EXACT_VERSION, CanonicalNextAction.CLOSE_DECISION)})),
    ProposalRoute.WATCH_DEFER: _RouteRule(DispositionJudgement.HOLD, CanonicalOutcome.LEAD_WATCH_DEFER, frozenset({ReasonCode.TIME_WATCH_REVIEW, ReasonCode.NOVELTY_INSUFFICIENT_INFORMATION}), frozenset({(next(iter(WATCH_CONDITION_MAPPING.values())).terminality, next(iter(WATCH_CONDITION_MAPPING.values())).next_action)})),
    ProposalRoute.ASSOCIATE_WITHOUT_CANDIDATE: _RouteRule(DispositionJudgement.ACCEPT, CanonicalOutcome.LEAD_ASSOCIATE_WITHOUT_CANDIDATE, frozenset({ReasonCode.REL_SAME_STATE, ReasonCode.REL_RELATED_DISTINCT, ReasonCode.REL_UNCERTAIN}), frozenset({(DecisionTerminality.TERMINAL_EXACT_VERSION, CanonicalNextAction.CLOSE_DECISION)})),
    ProposalRoute.SUPPLEMENTAL_DISCOVERY: _RouteRule(DispositionJudgement.ESCALATE, CanonicalOutcome.LEAD_SUPPLEMENTAL_DISCOVERY, frozenset({ReasonCode.SEARCH_ZERO_RESULTS, ReasonCode.SEARCH_PARTIAL_RESULTS, ReasonCode.SEARCH_PROVIDER_FAILURE, ReasonCode.NOVELTY_INSUFFICIENT_INFORMATION}), frozenset({(next(iter(SUPPLEMENTAL_ACTION_MAPPING.values())).terminality, next(iter(SUPPLEMENTAL_ACTION_MAPPING.values())).next_action)})),
    ProposalRoute.OPERATIONAL_HOLD: _RouteRule(DispositionJudgement.HOLD, CanonicalOutcome.LEAD_OPERATIONAL_HOLD, frozenset({item for item in ReasonCode if item.value.startswith(("OPS.", "CAPACITY."))}), frozenset({(DecisionTerminality.PENDING_CONDITION, CanonicalNextAction.RETRY_SAME_REQUEST), (DecisionTerminality.PENDING_CONDITION, CanonicalNextAction.REQUEST_REVIEW), (DecisionTerminality.PENDING_CONDITION, CanonicalNextAction.WAIT_FOR_DEPENDENCY), (DecisionTerminality.RETRYABLE_SAME_REQUEST, CanonicalNextAction.RETRY_SAME_REQUEST)})),
    ProposalRoute.NEW_EVENT_CANDIDATE: _RouteRule(DispositionJudgement.ACCEPT, CanonicalOutcome.LEAD_ADMIT_NEW_CANDIDATE, frozenset({ReasonCode.NOVELTY_LIKELY_NEW_EVENT, ReasonCode.REL_NO_ADEQUATE_PRIOR_MATCH}), frozenset({(DecisionTerminality.TERMINAL_EXACT_VERSION, CanonicalNextAction.HANDOFF_FOR_EVALUATION)})),
    ProposalRoute.DEVELOPMENT_CANDIDATE: _RouteRule(DispositionJudgement.ACCEPT, CanonicalOutcome.LEAD_ADMIT_DEVELOPMENT_CANDIDATE, frozenset({ReasonCode.NOVELTY_LIKELY_DEVELOPMENT, ReasonCode.REL_DEVELOPMENT}), frozenset({(DecisionTerminality.TERMINAL_EXACT_VERSION, CanonicalNextAction.HANDOFF_FOR_EVALUATION)})),
    ProposalRoute.CORRECTION_CANDIDATE: _RouteRule(DispositionJudgement.ACCEPT, CanonicalOutcome.LEAD_ADMIT_CORRECTION_CANDIDATE, frozenset({ReasonCode.REL_CORRECTION_REVERSAL, ReasonCode.CHANGE_REVISION, ReasonCode.CHANGE_WITHDRAWAL}), frozenset({(DecisionTerminality.TERMINAL_EXACT_VERSION, CanonicalNextAction.HANDOFF_FOR_EVALUATION)})),
})

_OPERATIONAL_REASONS: Mapping[CanonicalNextAction, frozenset[ReasonCode]] = (
    MappingProxyType({
        CanonicalNextAction.RETRY_SAME_REQUEST: frozenset({
            ReasonCode.OPS_TRANSPORT,
            ReasonCode.OPS_PARSER,
            ReasonCode.OPS_RETRIEVAL,
            ReasonCode.OPS_COLLISION,
            ReasonCode.OPS_MODEL,
            ReasonCode.OPS_QUEUE,
            ReasonCode.OPS_HANDOFF,
        }),
        CanonicalNextAction.REQUEST_REVIEW: frozenset({
            ReasonCode.OPS_PARTIAL,
            ReasonCode.OPS_COLLISION,
            ReasonCode.OPS_QUARANTINE,
            ReasonCode.OPS_STALE_STATE,
        }),
        CanonicalNextAction.WAIT_FOR_DEPENDENCY: frozenset({
            ReasonCode.OPS_RETRIEVAL,
            ReasonCode.OPS_QUEUE,
            ReasonCode.OPS_HANDOFF,
            ReasonCode.CAPACITY_SEARCH,
            ReasonCode.CAPACITY_MODEL,
            ReasonCode.CAPACITY_QUEUE,
            ReasonCode.CAPACITY_URGENT_RESERVE,
            ReasonCode.CAPACITY_REVIEWER,
        }),
    })
)

_OPERATIONAL_ACTION_KINDS: Mapping[str, CanonicalNextAction] = MappingProxyType({
    "RETRY_RETRIEVAL": CanonicalNextAction.RETRY_SAME_REQUEST,
    "REQUEST_REVIEW": CanonicalNextAction.REQUEST_REVIEW,
    "WAIT_FOR_DEPENDENCY": CanonicalNextAction.WAIT_FOR_DEPENDENCY,
})


def _exact_recommendation(
    value: LeadRecommendation, *, field: str
) -> tuple[LeadRecommendation, object]:
    if type(value) is not LeadRecommendation:
        raise DispositionContractError(
            f"{field} is outside canonical JSON: exact contract type required"
        )
    canonical = _constructed_value(
        lambda: value.canonical_value(), field=field
    )
    _bounded_canonical(canonical, field=field)
    exact = _constructed_value(
        lambda: LeadRecommendation.from_value(canonical), field=field
    )
    return exact, canonical  # type: ignore[return-value]


def _exact_selection(
    value: OutcomeSelection, *, field: str
) -> tuple[OutcomeSelection, object]:
    if type(value) is not OutcomeSelection:
        raise DispositionContractError(
            f"{field} is outside canonical JSON: exact contract type required"
        )
    canonical = _constructed_value(
        lambda: value.canonical_value(), field=field
    )
    _bounded_canonical(canonical, field=field)
    exact = _constructed_value(
        lambda: OutcomeSelection.from_mapping(canonical), field=field
    )
    return exact, canonical  # type: ignore[return-value]


def _operational_expected_action(
    recommendation: LeadRecommendation,
) -> CanonicalNextAction:
    operational = recommendation.operational_action
    assert operational is not None
    expected = _OPERATIONAL_ACTION_KINDS.get(operational.action_kind)
    if expected is None:
        raise DispositionContractError("operational action kind is unsupported")
    selected_seam = {
        CanonicalNextAction.RETRY_SAME_REQUEST: operational.retry_condition,
        CanonicalNextAction.REQUEST_REVIEW: (
            operational.review_condition
            or operational.expiry_condition
            or operational.owner_id
        ),
        CanonicalNextAction.WAIT_FOR_DEPENDENCY: operational.dependency,
    }[expected]
    if selected_seam is None:
        raise DispositionContractError(
            "operational action kind lacks its exact deterministic seam"
        )
    competing_present = {
        CanonicalNextAction.RETRY_SAME_REQUEST: any((
            operational.owner_id,
            operational.review_condition,
            operational.expiry_condition,
            operational.dependency,
        )),
        CanonicalNextAction.REQUEST_REVIEW: any((
            operational.retry_condition,
            operational.dependency,
        )),
        CanonicalNextAction.WAIT_FOR_DEPENDENCY: any((
            operational.owner_id,
            operational.retry_condition,
            operational.review_condition,
            operational.expiry_condition,
        )),
    }[expected]
    if competing_present:
        raise DispositionContractError(
            "operational action contains a competing action seam"
        )
    return expected


def _validate_exact_route_seam(
    recommendation: LeadRecommendation,
    selection: OutcomeSelection,
    route_binding_digest: str,
) -> None:
    """Bind conditional actions one-to-one to the Proposal seam they consume."""

    ProposalDisposition.validate_route_selection(recommendation.route, selection)
    action = selection.next_action
    assert action is not None
    conditional_routes = {
        ProposalRoute.WATCH_DEFER,
        ProposalRoute.SUPPLEMENTAL_DISCOVERY,
        ProposalRoute.OPERATIONAL_HOLD,
    }
    if recommendation.route in conditional_routes:
        if action.condition_reference != route_binding_digest:
            raise DispositionContractError(
                "next action does not bind the exact route-specific Proposal seam"
            )
    if recommendation.route is not ProposalRoute.OPERATIONAL_HOLD:
        return
    expected = _operational_expected_action(recommendation)
    if (
        action.action_code is not expected
        or selection.primary_reason.code not in _OPERATIONAL_REASONS[expected]
    ):
        raise DispositionContractError(
            "operational selection differs from its exact action and reason seam"
        )


@dataclass(frozen=True, slots=True)
class ProposalDisposition(_NoEffect):
    SCHEMA_VERSION: ClassVar[str] = PROPOSAL_DISPOSITION_SCHEMA_VERSION

    disposition_id: str
    judgement: DispositionJudgement
    proposal_id: str
    proposal_content_identity: str
    proposal_canonical_digest: str
    work_item_id: str
    work_item_version_id: str
    work_item_version_digest: str
    retrieval_context_id: str
    retrieval_context_digest: str
    lead_head: LeadDispositionHeadBinding
    validator_input: ValidatorInputBinding
    finding_set_digest: str
    route: ProposalRoute
    route_binding: LeadRecommendation
    route_binding_digest: str
    selection: OutcomeSelection
    authority: DispositionAuthority = DispositionAuthority.NONE

    def __post_init__(self) -> None:
        if type(self) is not ProposalDisposition:
            raise DispositionContractError(
                "disposition constructor requires the exact contract type"
            )
        _digest(self.disposition_id, "disposition_id")
        if not isinstance(self.judgement, DispositionJudgement) or not isinstance(self.route, ProposalRoute):
            raise DispositionContractError("disposition judgement and route must be typed")
        for field in ("proposal_id", "work_item_id", "work_item_version_id", "retrieval_context_id"):
            _token(getattr(self, field), field)
        for field in ("proposal_content_identity", "proposal_canonical_digest", "work_item_version_digest", "retrieval_context_digest", "finding_set_digest", "route_binding_digest"):
            _digest(getattr(self, field), field)
        if not isinstance(self.lead_head, LeadDispositionHeadBinding) or not isinstance(self.validator_input, ValidatorInputBinding):
            raise DispositionContractError("disposition bindings must be typed")
        exact_head = _exact_lead_head(self.lead_head, field="Lead head binding")
        exact_validator = _exact_validator_input(
            self.validator_input, field="validator input"
        )
        object.__setattr__(self, "lead_head", exact_head)
        object.__setattr__(self, "validator_input", exact_validator)
        if exact_validator.input_digest != exact_validator._expected_input_digest(
            self.proposal_canonical_digest
        ):
            raise DispositionContractError(
                "disposition validator input differs from the exact validation envelope"
            )
        if not isinstance(self.route_binding, LeadRecommendation):
            raise DispositionContractError("route binding must be typed")
        if not isinstance(self.selection, OutcomeSelection):
            raise DispositionContractError("outcome selection must be typed")
        exact_route, route_value = _exact_recommendation(
            self.route_binding, field="route binding"
        )
        exact_selection, selection_value = _exact_selection(
            self.selection, field="selection"
        )
        object.__setattr__(self, "route_binding", exact_route)
        object.__setattr__(self, "selection", exact_selection)
        if (
            exact_route.route is not self.route
            or exact_route.decision_lead_id != exact_head.decision_lead_id
        ):
            raise DispositionContractError("route binding differs from the exact Lead route")
        expected_route_digest = digest_bytes(
            _bounded_canonical(route_value, field="route binding")
        )
        if self.route_binding_digest != expected_route_digest:
            raise DispositionContractError("route binding digest differs")
        citation_digests = {
            citation.source_digest
            for citation in exact_route.input_citations
            if citation.source_kind.value == "DECISION_LEAD"
            and citation.source_id == exact_head.decision_lead_id
        }
        if citation_digests != {exact_head.decision_lead_digest}:
            raise DispositionContractError(
                "Lead head digest differs from the exact Proposal citation"
            )
        _validate_exact_route_seam(
            exact_route, exact_selection, self.route_binding_digest
        )
        if self.judgement is not _ROUTE_RULES[self.route].judgement:
            raise DispositionContractError("judgement differs from route matrix")
        if self.authority is not DispositionAuthority.NONE:
            raise DispositionContractError("phase-one disposition has no authority")
        if self.disposition_id != digest_bytes(
            _bounded_canonical_from(
                lambda: self._identity_value(), field="disposition identity"
            )
        ):
            raise DispositionContractError("disposition identity differs")
        _bounded_canonical_from(
            lambda: self.canonical_value(), field="disposition"
        )

    @staticmethod
    def validate_route_selection(route: ProposalRoute, selection: OutcomeSelection) -> None:
        if not isinstance(route, ProposalRoute) or not isinstance(selection, OutcomeSelection):
            raise DispositionContractError("route selection must be typed")
        selection, _ = _exact_selection(selection, field="selection")
        rule = _ROUTE_RULES[route]
        action = selection.next_action
        reason_codes = {
            selection.primary_reason.code,
            *(reason.code for reason in selection.supporting_reasons),
        }
        if (
            selection.outcome is not rule.outcome
            or not reason_codes <= rule.reasons
            or action is None
            or (selection.terminality, action.action_code) not in rule.actions
        ):
            raise DispositionContractError("selection is outside the exact route matrix")

    def _identity_value(self) -> dict[str, object]:
        return {
            "judgement": self.judgement.value, "proposal_id": self.proposal_id,
            "proposal_content_identity": self.proposal_content_identity,
            "proposal_canonical_digest": self.proposal_canonical_digest,
            "work_item_id": self.work_item_id, "work_item_version_id": self.work_item_version_id,
            "work_item_version_digest": self.work_item_version_digest,
            "retrieval_context_id": self.retrieval_context_id,
            "retrieval_context_digest": self.retrieval_context_digest,
            "lead_head_binding": self.lead_head.canonical_value(),
            "validator_input_binding": self.validator_input.canonical_value(),
            "finding_set_digest": self.finding_set_digest, "route": self.route.value,
            "route_binding": self.route_binding.canonical_value(),
            "route_binding_digest": self.route_binding_digest,
            "selection": self.selection.canonical_value(), "authority": self.authority.value,
        }

    def canonical_value(self) -> dict[str, object]:
        return {"schema_version": self.SCHEMA_VERSION, "disposition": {"disposition_id": self.disposition_id, **self._identity_value()}}

    @property
    def canonical_bytes(self) -> bytes:
        if type(self) is not ProposalDisposition:
            raise DispositionContractError(
                "disposition property requires the exact contract type"
            )
        raw = _bounded_canonical_from(
            lambda: self.canonical_value(), field="disposition"
        )
        exact = ProposalDisposition.from_canonical_bytes(raw)
        return _bounded_canonical_from(
            lambda: exact.canonical_value(), field="disposition"
        )

    @property
    def decision_lead_id(self) -> str:
        if type(self) is not ProposalDisposition:
            raise DispositionContractError(
                "disposition property requires the exact contract type"
            )
        return _constructed_value(
            lambda: _exact_lead_head(
                self.lead_head, field="Lead head binding"
            ).decision_lead_id,
            field="disposition",
        )  # type: ignore[return-value]

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> Self:
        if cls is not ProposalDisposition:
            raise DispositionContractError(
                "disposition parser requires the exact contract type"
            )
        root = _exact(_decode(raw, field="disposition"), {"schema_version", "disposition"}, "disposition document")
        if root["schema_version"] != ProposalDisposition.SCHEMA_VERSION:
            raise DispositionContractError("disposition schema version is unsupported")
        keys = {"disposition_id", "judgement", "proposal_id", "proposal_content_identity", "proposal_canonical_digest", "work_item_id", "work_item_version_id", "work_item_version_digest", "retrieval_context_id", "retrieval_context_digest", "lead_head_binding", "validator_input_binding", "finding_set_digest", "route", "route_binding", "route_binding_digest", "selection", "authority"}
        item = _exact(root["disposition"], keys, "disposition")
        if item["authority"] != DispositionAuthority.NONE.value:
            raise DispositionContractError("phase-one disposition has no authority")
        try:
            result = ProposalDisposition(
                disposition_id=item["disposition_id"], judgement=DispositionJudgement(item["judgement"]),
                proposal_id=item["proposal_id"], proposal_content_identity=item["proposal_content_identity"],
                proposal_canonical_digest=item["proposal_canonical_digest"], work_item_id=item["work_item_id"],
                work_item_version_id=item["work_item_version_id"], work_item_version_digest=item["work_item_version_digest"],
                retrieval_context_id=item["retrieval_context_id"], retrieval_context_digest=item["retrieval_context_digest"],
                lead_head=LeadDispositionHeadBinding.from_mapping(item["lead_head_binding"]),
                validator_input=ValidatorInputBinding.from_mapping(item["validator_input_binding"]),
                finding_set_digest=item["finding_set_digest"], route=ProposalRoute(item["route"]),
                route_binding=LeadRecommendation.from_value(item["route_binding"]),
                route_binding_digest=item["route_binding_digest"], selection=OutcomeSelection.from_mapping(item["selection"]),
            )
        except DispositionContractError:
            raise
        except Exception as exc:
            raise DispositionContractError("disposition is malformed") from exc
        if _bounded_canonical_from(
            lambda: result.canonical_value(), field="disposition"
        ) != raw:
            raise DispositionContractError("disposition typed replay differs")
        return result  # type: ignore[return-value]


def validate_proposal(raw: bytes, validator_input: ValidatorInputBinding) -> ProposalValidationResult:
    """Parse and deterministically validate one exact, untrusted Proposal."""

    if not isinstance(validator_input, ValidatorInputBinding):
        raise DispositionContractError("validator input binding must be typed")
    validator_input = _exact_validator_input(
        validator_input, field="validator input binding"
    )
    _decode(raw, field="proposal")
    try:
        proposal = TriageProposal.from_canonical_bytes(raw)
    except DispositionContractError:
        raise
    except Exception as exc:
        raise DispositionContractError("proposal integrity validation failed") from exc
    proposal_digest = digest_bytes(raw)
    if validator_input.input_digest != validator_input._expected_input_digest(
        proposal_digest
    ):
        raise DispositionContractError(
            "validator input integrity differs from the exact validation envelope"
        )
    findings: list[ProposalValidationFinding] = []
    for recommendation in proposal.recommendations:
        finding_code = FindingCode.ROUTE_VALIDATED
        finding_severity = FindingSeverity.INFO
        finding_path = f"recommendations:{recommendation.decision_lead_id}"
        if recommendation.route is ProposalRoute.OPERATIONAL_HOLD:
            try:
                _operational_expected_action(recommendation)
            except DispositionContractError as exc:
                finding_code = (
                    FindingCode.OPERATIONAL_ACTION_UNSUPPORTED
                    if "unsupported" in str(exc)
                    else FindingCode.OPERATIONAL_ACTION_AMBIGUOUS
                )
                finding_severity = FindingSeverity.ERROR
                finding_path += ":operational_action"
        seed = {
            "proposal_id": proposal.proposal_id,
            "proposal_content_identity": proposal.content_identity,
            "proposal_canonical_digest": proposal_digest,
            "code": finding_code.value,
            "severity": finding_severity.value,
            "path": finding_path,
            "evidence_reference_type": "PROPOSAL_RECOMMENDATION",
            "evidence_reference_id": recommendation.decision_lead_id,
            "evidence_reference_digest": digest_bytes(
                _bounded_canonical_from(
                    lambda: recommendation.canonical_value(),
                    field="recommendation",
                )
            ),
            "validator_input_binding": validator_input.canonical_value(),
            "authority": DispositionAuthority.NONE.value,
        }
        findings.append(ProposalValidationFinding(finding_id=digest_bytes(_bounded_canonical(seed, field="finding identity")), proposal_id=proposal.proposal_id, proposal_content_identity=proposal.content_identity, proposal_canonical_digest=proposal_digest, code=finding_code, severity=finding_severity, path=finding_path, evidence_reference_type="PROPOSAL_RECOMMENDATION", evidence_reference_id=recommendation.decision_lead_id, evidence_reference_digest=seed["evidence_reference_digest"], validator_input=validator_input))  # type: ignore[arg-type]
    ordered = tuple(sorted(findings, key=lambda item: item.finding_id))
    canonical = _bounded_canonical({"schema_version": _FINDING_SET_SCHEMA_VERSION, "proposal_content_identity": proposal.content_identity, "validator_input_binding": validator_input.canonical_value(), "finding_ids": [item.finding_id for item in ordered], "authority": DispositionAuthority.NONE.value}, field="finding set")
    return ProposalValidationResult(proposal, validator_input, ordered, digest_bytes(canonical))


def build_pending_dispositions(
    validation: ProposalValidationResult,
    lead_heads: Mapping[str, LeadDispositionHeadBinding],
    selections: Mapping[str, OutcomeSelection],
) -> tuple[ProposalDisposition, ...]:
    """Build a complete no-authority per-Lead set; persistence remains v19 work."""

    exact_validation = _exact_validation_result(
        validation, field="validation result"
    )
    if not isinstance(lead_heads, Mapping) or not isinstance(selections, Mapping):
        raise DispositionContractError("per-Lead disposition inputs must be mappings")
    proposal = exact_validation.proposal  # type: ignore[union-attr]
    validator_input = exact_validation.validator_input  # type: ignore[union-attr]
    findings = exact_validation.findings  # type: ignore[union-attr]
    proposal_bytes = _bounded_canonical_from(
        lambda: proposal.canonical_value(), field="proposal"
    )
    proposal_digest = digest_bytes(proposal_bytes)
    if any(
        finding.severity is FindingSeverity.ERROR
        for finding in findings
    ):
        raise DispositionContractError(
            "a disposition requires a complete finding set without validation errors"
        )
    expected = set(proposal.decision_lead_ids)
    head_keys = _constructed_value(lambda: set(lead_heads), field="Lead heads")
    selection_keys = _constructed_value(
        lambda: set(selections), field="outcome selections"
    )
    if head_keys != expected or selection_keys != expected:
        raise DispositionContractError("per-Lead disposition set must be complete")
    findings_by_lead = {
        item.evidence_reference_id for item in findings
    }
    if findings_by_lead != expected:
        raise DispositionContractError("finding manifest must be complete")
    recommendations = {
        item.decision_lead_id: item for item in proposal.recommendations
    }
    validator_value = _constructed_value(
        lambda: validator_input.canonical_value(), field="validator input binding"
    )
    _bounded_canonical(validator_value, field="validator input binding")
    result: list[ProposalDisposition] = []
    for lead_id in proposal.decision_lead_ids:
        head = _constructed_value(
            lambda: lead_heads[lead_id], field="Lead head binding"
        )
        selection = _constructed_value(
            lambda: selections[lead_id], field="outcome selection"
        )
        if not isinstance(head, LeadDispositionHeadBinding):
            raise DispositionContractError("Lead head binding must be typed")
        if not isinstance(selection, OutcomeSelection):
            raise DispositionContractError("outcome selection must be typed")
        head = _exact_lead_head(head, field="Lead head binding")
        head_value = _constructed_value(
            lambda: head.canonical_value(), field="Lead head binding"
        )
        if head.decision_lead_id != lead_id:
            raise DispositionContractError("Lead head binding differs from manifest")
        recommendation = recommendations[lead_id]
        if not isinstance(recommendation, LeadRecommendation):
            raise DispositionContractError("Proposal recommendation must be typed")
        recommendation, route_value = _exact_recommendation(
            recommendation, field="recommendation"
        )
        route_digest = digest_bytes(
            _bounded_canonical(route_value, field="recommendation")
        )
        rule = _ROUTE_RULES[recommendation.route]
        selection, selection_value = _exact_selection(
            selection, field="selection"
        )
        citation_digests = {
            citation.source_digest
            for citation in recommendation.input_citations
            if citation.source_kind.value == "DECISION_LEAD"
            and citation.source_id == lead_id
        }
        if citation_digests != {head.decision_lead_digest}:
            raise DispositionContractError(
                "Lead head digest differs from the exact Proposal citation"
            )
        _validate_exact_route_seam(recommendation, selection, route_digest)
        kwargs = dict(
            judgement=rule.judgement, proposal_id=proposal.proposal_id,
            proposal_content_identity=proposal.content_identity,
            proposal_canonical_digest=proposal_digest,
            work_item_id=proposal.work_item.work_item_id,
            work_item_version_id=proposal.work_item.work_item_version_id,
            work_item_version_digest=proposal.work_item.work_item_version_digest,
            retrieval_context_id=proposal.retrieval_context.context_id,
            retrieval_context_digest=proposal.retrieval_context.context_digest,
            lead_head=head, validator_input=validator_input,
            finding_set_digest=exact_validation.finding_set_digest,
            route=recommendation.route, route_binding=recommendation,
            route_binding_digest=route_digest, selection=selection,
        )
        identity = {
            "judgement": rule.judgement.value,
            "proposal_id": proposal.proposal_id,
            "proposal_content_identity": proposal.content_identity,
            "proposal_canonical_digest": proposal_digest,
            "work_item_id": proposal.work_item.work_item_id,
            "work_item_version_id": proposal.work_item.work_item_version_id,
            "work_item_version_digest": proposal.work_item.work_item_version_digest,
            "retrieval_context_id": proposal.retrieval_context.context_id,
            "retrieval_context_digest": proposal.retrieval_context.context_digest,
            "lead_head_binding": head_value,
            "validator_input_binding": validator_value,
            "finding_set_digest": exact_validation.finding_set_digest,
            "route": recommendation.route.value,
            "route_binding": route_value,
            "route_binding_digest": route_digest,
            "selection": selection_value,
            "authority": DispositionAuthority.NONE.value,
        }
        disposition_id = digest_bytes(
            _bounded_canonical(identity, field="disposition identity")
        )
        result.append(ProposalDisposition(disposition_id=disposition_id, **kwargs))  # type: ignore[arg-type]
    return tuple(result)


class ProposalDispositionStore:
    """Trusted v19 facade that persists one complete Proposal decision."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        retrieval_authority: RetrievalContextAuthority,
        authenticator: StaticAuthenticator,
        current_candidate_citations: "CurrentCandidateCitationReadPort | None" = None,
        *,
        work_items: TriageWorkItemStore | None = None,
        _open_verification: list[
            tuple[dict[str, ProposalDisposition], tuple[int, int, int]]
        ] | None = None,
        _validate_on_open: bool = True,
    ) -> None:
        if (
            type(connection) is not sqlite3.Connection
            or connection.in_transaction
            or type(retrieval_authority) is not RetrievalContextAuthority
            or type(authenticator) is not StaticAuthenticator
            or (
                current_candidate_citations is not None
                and type(current_candidate_citations)
                is not CurrentCandidateCitationReadPort
            )
            or (
                work_items is not None
                and (
                    type(work_items) is not TriageWorkItemStore
                    or work_items._connection is not connection
                    or work_items._retrieval_authority is not retrieval_authority
                )
            )
            or (
                _open_verification is not None
                and (type(_open_verification) is not list or _open_verification)
            )
        ):
            raise DispositionContractError(
                "disposition store requires exact trusted collaborators"
            )
        self._connection = connection
        self._retrieval_authority = retrieval_authority
        self._authenticator = authenticator
        self._current_candidate_citations = current_candidate_citations
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
                raise DispositionContractError("foreign keys must be enabled")
            _RETRIEVAL_ATTACH(retrieval_authority, connection)
            self._work_items = work_items or TriageWorkItemStore(
                connection, retrieval_authority
            )
            self._begin()
            if _validate_on_open:
                verified = self._verify_integrity()
                snapshot = _verification_snapshot(connection)
            connection.execute("COMMIT")
            if _open_verification is not None and _validate_on_open:
                _open_verification.append((verified, snapshot))
        except BaseException as exc:
            self._rollback()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, DispositionContractError):
                raise
            raise DispositionContractError(
                "disposition authority initialisation failed"
            ) from exc

    def _begin(self) -> None:
        if self._connection.in_transaction:
            raise DispositionContractError("connection has an active transaction")
        self._connection.execute("BEGIN IMMEDIATE")

    def _rollback(self) -> None:
        if self._connection.in_transaction:
            self._connection.execute("ROLLBACK")

    @staticmethod
    def _authenticated_identity(context: object) -> str:
        try:
            return digest_bytes(canonical_json_bytes({
                "principal_id": context.principal_id,  # type: ignore[attr-defined]
                "credential_binding_digest": context.credential_binding_digest,  # type: ignore[attr-defined]
            }))
        except Exception as exc:
            raise DispositionContractError("authenticated context differs") from exc

    def _authenticate(self, proof: AuthenticationProof) -> tuple[object, str]:
        if type(proof) is not AuthenticationProof:
            raise DispositionContractError("authentication proof must be exact")
        try:
            context = _AUTHENTICATE(
                self._authenticator, proof, now=UtcTimestamp.now()
            )
        except Exception as exc:
            raise DispositionContractError("authentication failed") from exc
        return context, self._authenticated_identity(context)

    @staticmethod
    def _validator(
        proposal_bytes: bytes,
        version: TriageWorkItemVersion,
        authenticated_identity: str,
    ) -> ValidatorInputBinding:
        retrieval = version.retrieval
        if (
            retrieval.state is not RetrievalBindingState.RECEIPT
            or not retrieval.usable
            or retrieval.context_id is None
            or retrieval.context_digest is None
            or retrieval.receipt_bytes is None
        ):
            raise DispositionContractError("retrieval receipt is not COMPLETE")
        return ValidatorInputBinding.for_proposal(
            proposal_bytes=proposal_bytes,
            validator_id=_VALIDATOR_ID,
            validator_version=_VALIDATOR_VERSION,
            authenticated_context_identity=authenticated_identity,
            retrieval_request_id=retrieval.request_id,
            retrieval_request_digest=retrieval.request_digest,
            retrieval_receipt_id=retrieval.context_id,
            retrieval_receipt_digest=retrieval.context_digest,
            ruleset_id=_RULESET_ID,
            ruleset_version=_RULESET_VERSION,
            ruleset_digest=_RULESET_DIGEST,
        )

    def _current_version(
        self, proposal: TriageProposal
    ) -> TriageWorkItemVersion:
        version = _WORK_ITEM_REQUIRE_CURRENT(
            self._work_items, proposal.work_item.work_item_id
        )
        _RETRIEVAL_VERIFY(
            self._retrieval_authority, self._connection, version.retrieval
        )
        if (
            version.version_id != proposal.work_item.work_item_version_id
            or version.canonical_digest
            != proposal.work_item.work_item_version_digest
            or version.retrieval.context_id != proposal.retrieval_context.context_id
            or version.retrieval.context_digest
            != proposal.retrieval_context.context_digest
            or version.retrieval.outcome != "COMPLETE"
            or tuple(proposal.decision_lead_ids)
            != tuple(lead.lead_id for lead in version.decision_leads)
            or tuple(proposal.context_lead_ids)
            != tuple(lead.lead_id for lead in version.context_leads)
        ):
            raise DispositionContractError(
                "Proposal differs from the exact current Work Item"
            )
        return version

    @staticmethod
    def _lead_heads(
        version: TriageWorkItemVersion,
    ) -> dict[str, LeadDispositionHeadBinding]:
        result: dict[str, LeadDispositionHeadBinding] = {}
        for lead in version.decision_leads:
            if type(lead) is not DecisionLeadBinding:
                raise DispositionContractError("decision Lead binding is not exact")
            result[lead.lead_id] = LeadDispositionHeadBinding(
                lead.lead_id,
                lead.lead_digest,
                lead.disposition_id,
                lead.disposition_digest,
            )
        return result

    def _reconcile_citations(
        self,
        version: TriageWorkItemVersion,
        recommendations: tuple[LeadRecommendation, ...],
    ) -> None:
        """Bind every Proposal citation to the exact retained v18 inputs."""

        decision_leads = {
            lead.lead_id: lead.lead_digest for lead in version.decision_leads
        }
        context_leads = {
            lead.lead_id: lead.lead_digest for lead in version.context_leads
        }
        if version.retrieval.receipt_bytes is None:
            raise DispositionContractError("retrieval citation authority differs")
        receipt = _decode(
            version.retrieval.receipt_bytes, field="retrieval citation receipt"
        )
        retrieval_items: dict[str, tuple[str, bytes]] = {}
        if (
            receipt.get("schema_identity")
            == "newsroom.increment5.native-retrieval-context-receipt.v1"
        ):
            try:
                from newsroom.increment5.native_retrieval import (
                    NativeRetrievalContextReceipt,
                )

                NativeRetrievalContextReceipt.from_bytes(
                    version.retrieval.receipt_bytes
                )
            except Exception as exc:
                raise DispositionContractError(
                    "retrieval citation authority differs"
                ) from exc
        else:
            raw_items = receipt.get("items")
            if type(raw_items) is not list:
                raise DispositionContractError("retrieval citation authority differs")
            try:
                for item in raw_items:
                    if type(item) is not dict or type(item.get("passage")) is not dict:
                        raise DispositionContractError(
                            "retrieval citation authority differs"
                        )
                    passage = item["passage"]
                    passage_id = passage["passage_id"]
                    text_digest = passage["text_digest"]
                    text = item["text"].encode("utf-8")
                    if passage_id in retrieval_items:
                        raise DispositionContractError(
                            "retrieval citation authority differs"
                        )
                    retrieval_items[passage_id] = (text_digest, text)
            except DispositionContractError:
                raise
            except Exception as exc:
                raise DispositionContractError(
                    "retrieval citation authority differs"
                ) from exc

        for recommendation in recommendations:
            for citation in recommendation.input_citations:
                if citation.source_kind is CitationSourceKind.DECISION_LEAD:
                    if decision_leads.get(citation.source_id) != citation.source_digest:
                        raise DispositionContractError(
                            "decision Lead citation differs from the exact Version"
                        )
                    continue
                if citation.source_kind is CitationSourceKind.CONTEXT_LEAD:
                    if context_leads.get(citation.source_id) != citation.source_digest:
                        raise DispositionContractError(
                            "context Lead citation differs from the exact Version"
                        )
                    continue
                if citation.source_kind is CitationSourceKind.CURRENT_CANDIDATE:
                    if self._current_candidate_citations is None:
                        raise DispositionContractError(
                            "current Candidate citation authority is unavailable"
                        )
                    observed = self._current_candidate_citations.require(
                        citation.source_id, citation.source_digest
                    )
                    if (
                        citation.target_hypothesis_id != observed.hypothesis_id
                        or citation.field_path != "$"
                        or citation.byte_start != 0
                        or citation.byte_end != len(observed.canonical_bytes)
                        or citation.quote_digest != observed.canonical_digest
                    ):
                        raise DispositionContractError(
                            "current Candidate citation differs from retained authority"
                        )
                    continue
                item = retrieval_items.get(citation.source_id)
                if (
                    item is None
                    or citation.field_path != "passage.text"
                    or citation.source_digest != item[0]
                ):
                    raise DispositionContractError(
                        "retrieval citation differs from the exact receipt item"
                    )
                text = item[1]
                if (
                    citation.byte_end > len(text)
                    or digest_bytes(text[citation.byte_start:citation.byte_end])
                    != citation.quote_digest
                ):
                    raise DispositionContractError(
                        "retrieval citation range differs from the exact receipt item"
                    )

    def persist(
        self,
        proposal_bytes: bytes,
        selections: Mapping[str, OutcomeSelection],
        *,
        proof: AuthenticationProof,
    ) -> tuple[ProposalDisposition, ...]:
        """Validate and atomically replay or persist the complete disposition set."""
        try:
            self._begin()
            proposal = TriageProposal.from_canonical_bytes(proposal_bytes)
            _, authenticated_identity = self._authenticate(proof)
            version = self._current_version(proposal)
            validator = self._validator(
                proposal_bytes, version, authenticated_identity
            )
            validation = validate_proposal(proposal_bytes, validator)
            findings = validation.findings
            self._reconcile_citations(version, validation.proposal.recommendations)
            self._persist_findings(version, validation)
            if any(item.severity is FindingSeverity.ERROR for item in findings):
                raise DispositionContractError("Proposal validation contains ERROR")
            dispositions = build_pending_dispositions(
                validation, self._lead_heads(version), selections
            )
            self._persist_dispositions(version, findings, dispositions)
            stored = self._load_complete_set(
                version.version_id, proposal.proposal_id
            )
            if tuple(item.canonical_bytes for item in stored) != tuple(
                item.canonical_bytes for item in dispositions
            ):
                raise DispositionContractError("disposition replay diverges")
            self._connection.execute("COMMIT")
            return stored
        except BaseException as exc:
            self._rollback()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, DispositionContractError):
                raise
            raise DispositionContractError("disposition persistence failed") from exc

    # A concise alias makes the facade legible at composition sites.
    decide = persist

    def _persist_findings(
        self,
        version: TriageWorkItemVersion,
        validation: ProposalValidationResult,
    ) -> None:
        for finding in validation.findings:
            raw = finding.canonical_bytes
            self._connection.execute(
                "INSERT OR IGNORE INTO triage_proposal_validation_findings("
                "finding_id,work_item_id,work_item_version_id,work_item_version_digest,"
                "proposal_id,proposal_content_identity,proposal_canonical_digest,"
                "decision_lead_id,validator_input_digest,finding_set_digest,severity,"
                "canonical_bytes,canonical_digest,recorded_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    finding.finding_id, version.work_item_id, version.version_id,
                    version.canonical_digest, finding.proposal_id,
                    finding.proposal_content_identity,
                    finding.proposal_canonical_digest,
                    finding.evidence_reference_id,
                    finding.validator_input.input_digest,
                    validation.finding_set_digest, finding.severity.value, raw,
                    digest_bytes(raw), self._recorded_at(),
                ),
            )
            row = self._connection.execute(
                "SELECT canonical_bytes,finding_set_digest FROM triage_proposal_validation_findings "
                "WHERE work_item_version_id=? AND decision_lead_id=?",
                (version.version_id, finding.evidence_reference_id),
            ).fetchone()
            if row is None or bytes(row[0]) != raw or row[1] != validation.finding_set_digest:
                raise DispositionContractError("finding replay diverges")
        rows = self._connection.execute(
            "SELECT decision_lead_id FROM triage_proposal_validation_findings "
            "WHERE work_item_version_id=? AND proposal_id=? ORDER BY decision_lead_id",
            (version.version_id, validation.proposal.proposal_id),
        ).fetchall()
        if tuple(str(row[0]) for row in rows) != tuple(
            sorted(validation.proposal.decision_lead_ids)
        ):
            raise DispositionContractError("finding set replay is partial or extra")

    def _persist_dispositions(
        self,
        version: TriageWorkItemVersion,
        findings: tuple[ProposalValidationFinding, ...],
        dispositions: tuple[ProposalDisposition, ...],
    ) -> None:
        finding_by_lead = {item.evidence_reference_id: item for item in findings}
        for disposition in dispositions:
            raw = disposition.canonical_bytes
            selection_digest = digest_bytes(canonical_json_bytes(
                disposition.selection.canonical_value()
            ))
            self._connection.execute(
                "INSERT OR IGNORE INTO triage_proposal_dispositions("
                "disposition_id,work_item_id,work_item_version_id,work_item_version_digest,"
                "proposal_id,proposal_content_identity,proposal_canonical_digest,"
                "decision_lead_id,lead_head_id,lead_head_digest,validator_input_digest,"
                "finding_set_digest,finding_id,selection_digest,canonical_bytes,"
                "canonical_digest,recorded_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    disposition.disposition_id, version.work_item_id,
                    version.version_id, version.canonical_digest,
                    disposition.proposal_id, disposition.proposal_content_identity,
                    disposition.proposal_canonical_digest,
                    disposition.decision_lead_id,
                    disposition.lead_head.current_disposition_head_id,
                    disposition.lead_head.current_disposition_head_digest,
                    disposition.validator_input.input_digest,
                    disposition.finding_set_digest,
                    finding_by_lead[disposition.decision_lead_id].finding_id,
                    selection_digest, raw, digest_bytes(raw), self._recorded_at(),
                ),
            )
            row = self._connection.execute(
                "SELECT canonical_bytes FROM triage_proposal_dispositions "
                "WHERE work_item_version_id=? AND decision_lead_id=?",
                (version.version_id, disposition.decision_lead_id),
            ).fetchone()
            if row is None or bytes(row[0]) != raw:
                raise DispositionContractError("disposition replay diverges")

    def _load_complete_set(
        self, version_id: str, proposal_id: str
    ) -> tuple[ProposalDisposition, ...]:
        rows = self._connection.execute(
            "SELECT canonical_bytes FROM triage_proposal_dispositions "
            "WHERE work_item_version_id=? AND proposal_id=? ORDER BY decision_lead_id",
            (version_id, proposal_id),
        ).fetchall()
        result = tuple(
            ProposalDisposition.from_canonical_bytes(bytes(row[0])) for row in rows
        )
        if not result:
            raise DispositionContractError("complete disposition set is absent")
        return result

    def require_current(
        self, disposition_id: str, *, proof: AuthenticationProof
    ) -> ProposalDisposition:
        """Load a retained disposition and recheck every use-time authority seam."""
        _digest(disposition_id, "disposition_id")
        try:
            self._begin()
            disposition = self.require_current_in_transaction(
                disposition_id, proof=proof
            )
            self._connection.execute("COMMIT")
            return disposition
        except BaseException as exc:
            self._rollback()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, DispositionContractError):
                raise
            raise DispositionContractError("disposition currentness failed") from exc

    def require_current_in_transaction(
        self, disposition_id: str, *, proof: AuthenticationProof
    ) -> ProposalDisposition:
        """Recheck a retained disposition inside the caller's write transaction."""
        _digest(disposition_id, "disposition_id")
        if not self._connection.in_transaction:
            raise DispositionContractError(
                "transaction-aware disposition use requires an active transaction"
            )
        disposition = self._verify_requested_disposition_in_transaction(
            disposition_id
        )
        return self._require_current_value_after_integrity_in_transaction(
            disposition, proof=proof
        )

    def _verify_requested_disposition_in_transaction(
        self, disposition_id: str
    ) -> ProposalDisposition:
        """Verify one complete proposal group in the caller's transaction."""
        _digest(disposition_id, "disposition_id")
        if not self._connection.in_transaction:
            raise DispositionContractError(
                "transaction-aware disposition use requires an active transaction"
            )
        requested = self._connection.execute(
            "SELECT work_item_version_id,proposal_id "
            "FROM triage_proposal_dispositions WHERE disposition_id=?",
            (disposition_id,),
        ).fetchone()
        if requested is None:
            raise DispositionContractError("unknown disposition")
        group = (str(requested[0]), str(requested[1]))
        dispositions = self._verify_retained_disposition_rows(
            self._connection.execute(
                "SELECT finding_id,work_item_id,work_item_version_id,"
                "work_item_version_digest,proposal_id,proposal_content_identity,"
                "proposal_canonical_digest,decision_lead_id,validator_input_digest,"
                "finding_set_digest,severity,canonical_bytes,canonical_digest "
                "FROM triage_proposal_validation_findings "
                "WHERE work_item_version_id=? AND proposal_id=?",
                group,
            ),
            self._connection.execute(
                "SELECT disposition_id,work_item_id,work_item_version_id,"
                "work_item_version_digest,proposal_id,proposal_content_identity,"
                "proposal_canonical_digest,decision_lead_id,lead_head_id,"
                "lead_head_digest,validator_input_digest,finding_set_digest,"
                "finding_id,selection_digest,canonical_bytes,canonical_digest "
                "FROM triage_proposal_dispositions "
                "WHERE work_item_version_id=? AND proposal_id=?",
                group,
            ),
        )
        disposition = dispositions.get(disposition_id)
        if disposition is None:
            raise DispositionContractError("unknown disposition")
        return disposition

    def _require_current_after_integrity_in_transaction(
        self, disposition_id: str, *, proof: AuthenticationProof
    ) -> ProposalDisposition:
        """Recheck currentness after this transaction verified retained rows."""
        _digest(disposition_id, "disposition_id")
        if not self._connection.in_transaction:
            raise DispositionContractError(
                "transaction-aware disposition use requires an active transaction"
            )
        row = self._connection.execute(
            "SELECT canonical_bytes FROM triage_proposal_dispositions WHERE disposition_id=?",
            (disposition_id,),
        ).fetchone()
        if row is None:
            raise DispositionContractError("unknown disposition")
        disposition = ProposalDisposition.from_canonical_bytes(bytes(row[0]))
        return self._require_current_value_after_integrity_in_transaction(
            disposition, proof=proof
        )

    def _require_current_value_after_integrity_in_transaction(
        self,
        disposition: ProposalDisposition,
        *,
        proof: AuthenticationProof,
    ) -> ProposalDisposition:
        """Recheck current authority for an integrity-verified disposition."""
        if not self._connection.in_transaction:
            raise DispositionContractError(
                "transaction-aware disposition use requires an active transaction"
            )
        _, authenticated_identity = self._authenticate(proof)
        version = _WORK_ITEM_REQUIRE_CURRENT(self._work_items, disposition.work_item_id)
        _RETRIEVAL_VERIFY(
            self._retrieval_authority, self._connection, version.retrieval
        )
        lead = next(
            (
                item
                for item in version.decision_leads
                if item.lead_id == disposition.decision_lead_id
            ),
            None,
        )
        if (
            version.version_id != disposition.work_item_version_id
            or version.canonical_digest != disposition.work_item_version_digest
            or version.retrieval.context_id != disposition.retrieval_context_id
            or version.retrieval.context_digest != disposition.retrieval_context_digest
            or disposition.validator_input.authenticated_context_identity
            != authenticated_identity
            or disposition.validator_input.validator_id != _VALIDATOR_ID
            or disposition.validator_input.validator_version != _VALIDATOR_VERSION
            or disposition.validator_input.ruleset_id != _RULESET_ID
            or disposition.validator_input.ruleset_version != _RULESET_VERSION
            or disposition.validator_input.ruleset_digest != _RULESET_DIGEST
            or disposition.validator_input.retrieval_request_id
            != version.retrieval.request_id
            or disposition.validator_input.retrieval_request_digest
            != version.retrieval.request_digest
            or disposition.validator_input.retrieval_receipt_id
            != version.retrieval.context_id
            or disposition.validator_input.retrieval_receipt_digest
            != version.retrieval.context_digest
            or lead is None
            or lead.lead_digest != disposition.lead_head.decision_lead_digest
            or lead.disposition_id != disposition.lead_head.current_disposition_head_id
            or lead.disposition_digest
            != disposition.lead_head.current_disposition_head_digest
        ):
            raise DispositionContractError("disposition is no longer current")
        return disposition

    def verify_retained_integrity_in_transaction(self) -> None:
        """Revalidate retained disposition/finding authority without currentness."""
        self._verified_dispositions_in_transaction()

    def _verified_dispositions_in_transaction(self) -> dict[str, ProposalDisposition]:
        """Return fully checked dispositions from the caller's current snapshot."""
        if not self._connection.in_transaction:
            raise DispositionContractError(
                "transaction-aware disposition integrity requires an active transaction"
            )
        return self._verify_integrity()

    def _verify_integrity(self) -> dict[str, ProposalDisposition]:
        tables = {
            str(row[0])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {
            "triage_proposal_validation_findings",
            "triage_proposal_dispositions",
        }
        if not required <= tables:
            raise DispositionContractError("v19 disposition schema is absent")
        return self._verify_retained_disposition_rows(
            self._connection.execute(
                "SELECT finding_id,work_item_id,work_item_version_id,"
                "work_item_version_digest,proposal_id,proposal_content_identity,"
                "proposal_canonical_digest,decision_lead_id,validator_input_digest,"
                "finding_set_digest,severity,canonical_bytes,canonical_digest "
                "FROM triage_proposal_validation_findings"
            ),
            self._connection.execute(
                "SELECT disposition_id,work_item_id,work_item_version_id,"
                "work_item_version_digest,proposal_id,proposal_content_identity,"
                "proposal_canonical_digest,decision_lead_id,lead_head_id,"
                "lead_head_digest,validator_input_digest,finding_set_digest,"
                "finding_id,selection_digest,canonical_bytes,canonical_digest "
                "FROM triage_proposal_dispositions"
            ),
        )

    def _verify_retained_disposition_rows(
        self,
        finding_rows: Iterable[sqlite3.Row],
        disposition_rows: Iterable[sqlite3.Row],
    ) -> dict[str, ProposalDisposition]:
        """Verify canonical finding/disposition groups supplied by one snapshot."""
        findings_by_group: dict[tuple[str, str], dict[str, ProposalValidationFinding]] = {}
        finding_sets: dict[tuple[str, str], str] = {}
        finding_meta: dict[tuple[str, str, str], tuple[str, str]] = {}
        for row in finding_rows:
            finding = ProposalValidationFinding.from_canonical_bytes(bytes(row[11]))
            group = (str(row[2]), str(row[4]))
            if (
                finding.finding_id != row[0]
                or finding.proposal_id != row[4]
                or finding.proposal_content_identity != row[5]
                or finding.proposal_canonical_digest != row[6]
                or finding.evidence_reference_id != row[7]
                or finding.validator_input.input_digest != row[8]
                or finding.severity.value != row[10]
                or digest_bytes(bytes(row[11])) != row[12]
            ):
                raise DispositionContractError("retained finding bytes differ")
            findings_by_group.setdefault(group, {})[finding.evidence_reference_id] = finding
            finding_meta[(group[0], group[1], finding.evidence_reference_id)] = (
                str(row[1]), str(row[3])
            )
            previous = finding_sets.setdefault(group, str(row[9]))
            if previous != row[9]:
                raise DispositionContractError("retained finding set differs")
        dispositions_by_group: dict[tuple[str, str], dict[str, ProposalDisposition]] = {}
        for row in disposition_rows:
            disposition = ProposalDisposition.from_canonical_bytes(bytes(row[14]))
            group = (str(row[2]), str(row[4]))
            findings = findings_by_group.get(group, {})
            finding = findings.get(disposition.decision_lead_id)
            if (
                disposition.disposition_id != row[0]
                or disposition.work_item_id != row[1]
                or disposition.work_item_version_id != row[2]
                or disposition.work_item_version_digest != row[3]
                or disposition.proposal_id != row[4]
                or disposition.proposal_content_identity != row[5]
                or disposition.proposal_canonical_digest != row[6]
                or disposition.decision_lead_id != row[7]
                or disposition.lead_head.current_disposition_head_id != row[8]
                or disposition.lead_head.current_disposition_head_digest != row[9]
                or disposition.validator_input.input_digest != row[10]
                or disposition.finding_set_digest != row[11]
                or finding is None or finding.finding_id != row[12]
                or digest_bytes(canonical_json_bytes(disposition.selection.canonical_value())) != row[13]
                or digest_bytes(bytes(row[14])) != row[15]
            ):
                raise DispositionContractError("retained disposition bytes differ")
            dispositions_by_group.setdefault(group, {})[disposition.decision_lead_id] = disposition
        if set(findings_by_group) != set(dispositions_by_group):
            raise DispositionContractError("retained disposition group coverage differs")
        for group, findings in findings_by_group.items():
            dispositions = dispositions_by_group[group]
            if set(findings) != set(dispositions):
                raise DispositionContractError("retained disposition set is partial or extra")
            version_row = self._connection.execute(
                "SELECT work_item_id,retrieval_outcome,canonical_bytes,canonical_digest "
                "FROM triage_work_item_versions WHERE version_id=?",
                (group[0],),
            ).fetchone()
            if version_row is None:
                raise DispositionContractError("retained Work Item Version is absent")
            version = TriageWorkItemVersion.from_canonical_bytes(bytes(version_row[2]))
            if (
                version.version_id != group[0]
                or version.work_item_id != version_row[0]
                or version.canonical_digest != version_row[3]
                or version.retrieval.outcome != "COMPLETE"
                or version_row[1] != "COMPLETE"
                or version.retrieval.context_id is None
                or version.retrieval.context_digest is None
            ):
                raise DispositionContractError("retained Work Item Version differs")
            _RETRIEVAL_VERIFY_RETAINED(
                self._retrieval_authority, self._connection, version.retrieval
            )
            self._reconcile_citations(
                version,
                tuple(
                    disposition.route_binding
                    for disposition in dispositions.values()
                ),
            )
            leads = {lead.lead_id: lead for lead in version.decision_leads}
            if set(leads) != set(findings):
                raise DispositionContractError("retained decision Lead manifest differs")
            first = next(iter(findings.values()))
            if any(
                item.proposal_content_identity != first.proposal_content_identity
                or item.proposal_canonical_digest != first.proposal_canonical_digest
                or item.validator_input != first.validator_input
                or item.severity is FindingSeverity.ERROR
                for item in findings.values()
            ):
                raise DispositionContractError("retained finding set differs")
            for lead_id, disposition in dispositions.items():
                finding = findings[lead_id]
                lead = leads[lead_id]
                if (
                    finding_meta[(group[0], group[1], lead_id)]
                    != (version.work_item_id, version.canonical_digest)
                    or disposition.work_item_id != version.work_item_id
                    or disposition.work_item_version_digest != version.canonical_digest
                    or disposition.retrieval_context_id != version.retrieval.context_id
                    or disposition.retrieval_context_digest != version.retrieval.context_digest
                    or disposition.lead_head.decision_lead_id != lead.lead_id
                    or disposition.lead_head.decision_lead_digest != lead.lead_digest
                    or disposition.lead_head.current_disposition_head_id != lead.disposition_id
                    or disposition.lead_head.current_disposition_head_digest != lead.disposition_digest
                    or finding.proposal_id != disposition.proposal_id
                    or finding.proposal_content_identity != disposition.proposal_content_identity
                    or finding.proposal_canonical_digest != disposition.proposal_canonical_digest
                    or finding.validator_input != disposition.validator_input
                    or finding.evidence_reference_digest != disposition.route_binding_digest
                    or finding.evidence_reference_id != disposition.decision_lead_id
                    or disposition.route_binding.decision_lead_id != lead_id
                    or disposition.validator_input.retrieval_request_id != version.retrieval.request_id
                    or disposition.validator_input.retrieval_request_digest != version.retrieval.request_digest
                    or disposition.validator_input.retrieval_receipt_id != version.retrieval.context_id
                    or disposition.validator_input.retrieval_receipt_digest != version.retrieval.context_digest
                ):
                    raise DispositionContractError("retained finding linkage differs")
            ordered_ids = sorted(item.finding_id for item in findings.values())
            expected_set = digest_bytes(canonical_json_bytes({
                "schema_version": _FINDING_SET_SCHEMA_VERSION,
                "proposal_content_identity": first.proposal_content_identity,
                "validator_input_binding": first.validator_input.canonical_value(),
                "finding_ids": ordered_ids,
                "authority": DispositionAuthority.NONE.value,
            }))
            if expected_set != finding_sets[group] or any(
                item.finding_set_digest != expected_set
                for item in dispositions.values()
            ):
                raise DispositionContractError("retained finding-set digest differs")

        return {
            item.disposition_id: item
            for dispositions in dispositions_by_group.values()
            for item in dispositions.values()
        }

    def _recorded_at(self) -> str:
        return str(self._connection.execute(
            "SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now')"
        ).fetchone()[0])


__all__ = [
    "DISPOSITION_AUTHORITY", "MAX_DISPOSITION_CANONICAL_BYTES",
    "PROPOSAL_DISPOSITION", "PROPOSAL_DISPOSITION_SCHEMA_VERSION",
    "PROPOSAL_VALIDATION_FINDING", "PROPOSAL_VALIDATION_FINDING_SCHEMA_VERSION",
    "VALIDATED_PROPOSAL_LEAD_DISPOSITION_BINDING", "CurrentCandidateCitation",
    "CurrentCandidateCitationReadPort", "DispositionAuthority",
    "CurrentCandidateCitation", "CurrentCandidateCitationReadPort",
    "DispositionContractError", "DispositionJudgement", "FindingCode", "FindingSeverity",
    "LeadDispositionHeadBinding", "ProposalDisposition", "ProposalValidationFinding",
    "ProposalDispositionStore", "ProposalValidationResult", "ValidatorInputBinding", "build_pending_dispositions",
    "validate_proposal",
]
