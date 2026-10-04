"""Governed-object retention for native, non-public Evidence Packages.

Source inventory comes from authenticated governed record objects.  This slice
does not establish live Source Registry currentness or drafting admission.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from newsroom.authority import (
    AuthenticationProof,
    GovernedObjects,
    HydrationRequest,
    ObjectAdmissionId,
    ObjectAdmissionRequest,
)
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane.evidence import (
    ClaimAuthorityClass,
    EvidenceGateEvidence,
    EvidencePackage,
    Evid012QualificationTest,
    GovernedClaimEvidence,
    GovernedClaimStatus,
    QualificationEvidence,
    evidence_package_value,
    validate_governed_evidence_records,
)
from newsroom.increment6.candidates import StoryCandidateReadPort

from .ingress import EvidenceIntakeIngress, IntakeAcknowledgement

_SCHEMA = "newsroom.increment10.governed-evidence-package.v1"
_SOURCE_PURPOSE = "evidence.source"
_RECORD_PURPOSE = "evidence.record"
_PACKAGE_PURPOSE = "evidence.package.retained"
_SOURCE_CLASS = "evidence_source"
_RECORD_CLASS = "evidence_record"
_PACKAGE_CLASS = "evidence_package"
_PUBLICATION_EVIDENCE = "publication_evidence"
_PACKAGE_RETENTION = "evidence_package_retention"


class EvidencePackageError(ValueError):
    """Raised when governed material cannot establish one native package."""


@dataclass(frozen=True, slots=True)
class GovernedEvidencePackage:
    """Typed result; a generic object admission alone is not package authority."""

    package_admission_id: ObjectAdmissionId
    package: EvidencePackage
    receipt_id: str
    candidate_version_id: str
    candidate_version_digest: str
    governing_manifest_digest: str
    source_admission_ids: tuple[ObjectAdmissionId, ...]
    record_admission_ids: tuple[ObjectAdmissionId, ...]
    source_inventory: tuple[tuple[str, str], ...]

    @property
    def drafting_authority(self) -> bool:
        return False

    @property
    def publication_authority(self) -> bool:
        return False


class GovernedEvidencePackages:
    """Validate exact governed material before retaining or reading a package."""

    def __init__(
        self,
        *,
        objects: GovernedObjects,
        ingress: EvidenceIntakeIngress,
        reader_principal_id: str,
        reader_authority_domain: str,
        source_hydration_policy_digest: str,
        record_hydration_policy_digest: str,
        package_hydration_policy_digest: str,
        package_admission_definition_digest: str,
    ) -> None:
        if (
            type(objects) is not GovernedObjects
            or type(ingress) is not EvidenceIntakeIngress
        ):
            raise EvidencePackageError("exact governed authorities are required")
        for value in (reader_principal_id, reader_authority_domain):
            if type(value) is not str or not value:
                raise EvidencePackageError("evidence reader identity is required")
        digests = (
            source_hydration_policy_digest,
            record_hydration_policy_digest,
            package_hydration_policy_digest,
            package_admission_definition_digest,
        )
        if any(
            type(value) is not str or not value.startswith("sha256:")
            for value in digests
        ):
            raise EvidencePackageError("evidence policy digest is required")
        self._objects = objects
        self._ingress = ingress
        self._reader_principal_id = reader_principal_id
        self._reader_authority_domain = reader_authority_domain
        self._source_policy_digest = source_hydration_policy_digest
        self._record_policy_digest = record_hydration_policy_digest
        self._package_policy_digest = package_hydration_policy_digest
        self._package_definition_digest = package_admission_definition_digest

    def retain(
        self,
        package: EvidencePackage,
        *,
        receipt_id: str,
        candidate_port: StoryCandidateReadPort,
        source_admission_ids: tuple[ObjectAdmissionId, ...],
        record_admission_ids: tuple[ObjectAdmissionId, ...],
        proof: AuthenticationProof,
    ) -> GovernedEvidencePackage:
        acknowledgement = self._ingress.receipt(receipt_id)
        resolved, envelope, source_inventory = self._validated_envelope(
            package,
            acknowledgement=acknowledgement,
            candidate_port=candidate_port,
            source_admission_ids=source_admission_ids,
            record_admission_ids=record_admission_ids,
            proof=proof,
        )
        result = self._objects.admit(
            ObjectAdmissionRequest(
                "evidence.package",
                f"package:{acknowledgement.candidate_version_id}:{resolved.digest}",
            ),
            envelope,
            proof=proof,
        )
        admission = result.admission
        if (
            admission.definition_digest != self._package_definition_digest
            or admission.object_class != _PACKAGE_CLASS
            or admission.allowed_use != _PACKAGE_RETENTION
            or admission.blob.blob_digest != digest_bytes(envelope)
            or not admission.active
        ):
            raise EvidencePackageError("package object admission policy differs")
        return self._result(
            admission.admission_id,
            resolved,
            acknowledgement,
            source_admission_ids,
            record_admission_ids,
            source_inventory,
        )

    def read(
        self,
        package_admission_id: ObjectAdmissionId,
        *,
        candidate_port: StoryCandidateReadPort,
        proof: AuthenticationProof,
    ) -> GovernedEvidencePackage:
        hydrated = self._objects.rehydrate(
            HydrationRequest(package_admission_id, _PACKAGE_PURPOSE), proof=proof
        )
        self._require_decision(
            hydrated.decision,
            policy_digest=self._package_policy_digest,
            object_class=_PACKAGE_CLASS,
            allowed_use=_PACKAGE_RETENTION,
        )
        value = _document(hydrated.data)
        if value.get("schema_identity") != _SCHEMA:
            raise EvidencePackageError("package object schema differs")
        package = _package_from_value(value.get("package"))
        try:
            source_ids = tuple(
                ObjectAdmissionId.parse(item) for item in value["source_admission_ids"]
            )
            record_ids = tuple(
                ObjectAdmissionId.parse(item) for item in value["record_admission_ids"]
            )
            acknowledgement = self._ingress.receipt(str(value["receipt_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidencePackageError("package object bindings differ") from exc
        resolved, rebuilt, source_inventory = self._validated_envelope(
            package,
            acknowledgement=acknowledgement,
            candidate_port=candidate_port,
            source_admission_ids=source_ids,
            record_admission_ids=record_ids,
            proof=proof,
            reuse_retained=True,
        )
        if rebuilt != hydrated.data:
            raise EvidencePackageError("retained package object differs")
        return self._result(
            package_admission_id,
            resolved,
            acknowledgement,
            source_ids,
            record_ids,
            source_inventory,
        )

    def _validated_envelope(
        self,
        package: EvidencePackage,
        *,
        acknowledgement: IntakeAcknowledgement,
        candidate_port: StoryCandidateReadPort,
        source_admission_ids: tuple[ObjectAdmissionId, ...],
        record_admission_ids: tuple[ObjectAdmissionId, ...],
        proof: AuthenticationProof,
        reuse_retained: bool = False,
    ) -> tuple[EvidencePackage, bytes, tuple[tuple[str, str], ...]]:
        if type(package) is not EvidencePackage or package.admitted_context is not None:
            raise EvidencePackageError("native package must use exact evidence values")
        package = _package_from_value(evidence_package_value(package))
        if type(candidate_port) is not StoryCandidateReadPort:
            raise EvidencePackageError("authenticated Candidate read port required")
        try:
            version = candidate_port.require_retained_version(
                acknowledgement.candidate_version_id
            )
        except Exception as exc:
            raise EvidencePackageError(
                "Candidate authority verification failed"
            ) from exc
        manifest = version.governing_manifest
        bindings = manifest.lead_signal_bindings
        if (
            version.canonical_digest != acknowledgement.candidate_version_digest
            or manifest.canonical_digest != acknowledgement.governing_manifest_digest
            or package.candidate_id != version.candidate_id
            or package.hypothesis_id != manifest.hypothesis_id
            or package.lead_ids != tuple(item.lead_id for item in bindings)
            or package.signal_ids != tuple(item.signal_id for item in bindings)
        ):
            raise EvidencePackageError("package Candidate binding differs")
        if (
            type(source_admission_ids) is not tuple
            or len(source_admission_ids) != len(package.passages)
            or len(source_admission_ids) != len(package.source_ids)
            or len(source_admission_ids) != len(package.observation_digests)
            or any(type(item) is not ObjectAdmissionId for item in source_admission_ids)
            or len({str(item) for item in source_admission_ids})
            != len(source_admission_ids)
            or type(record_admission_ids) is not tuple
            or not record_admission_ids
            or any(type(item) is not ObjectAdmissionId for item in record_admission_ids)
            or len({str(item) for item in record_admission_ids})
            != len(record_admission_ids)
        ):
            raise EvidencePackageError("governed material inventory differs")

        hydrate = self._objects.rehydrate if reuse_retained else self._objects.hydrate
        source_digests: list[str] = []
        for index, admission_id in enumerate(source_admission_ids):
            material = hydrate(
                HydrationRequest(admission_id, _SOURCE_PURPOSE), proof=proof
            )
            self._require_decision(
                material.decision,
                policy_digest=self._source_policy_digest,
                object_class=_SOURCE_CLASS,
                allowed_use=_PUBLICATION_EVIDENCE,
            )
            try:
                passage = material.data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise EvidencePackageError("evidence source is not UTF-8") from exc
            digest = digest_bytes(material.data)
            if (
                passage != package.passages[index]
                or digest != package.observation_digests[index]
            ):
                raise EvidencePackageError(
                    "evidence passage differs from governed bytes"
                )
            source_digests.append(digest)

        rows: list[tuple[object, object, object, object]] = []
        for admission_id in record_admission_ids:
            material = hydrate(
                HydrationRequest(admission_id, _RECORD_PURPOSE), proof=proof
            )
            self._require_decision(
                material.decision,
                policy_digest=self._record_policy_digest,
                object_class=_RECORD_CLASS,
                allowed_use=_PUBLICATION_EVIDENCE,
            )
            record = _document(material.data)
            record_id, record_type = record.get("record_id"), record.get("record_type")
            if type(record_id) is not str or type(record_type) is not str:
                raise EvidencePackageError("governed evidence record identity differs")
            rows.append(
                (
                    record_id,
                    record_type,
                    material.data.decode(),
                    digest_bytes(material.data),
                )
            )

        decoded_records = [json.loads(str(row[2])) for row in rows]
        source_record_values = tuple(
            record
            for record in decoded_records
            if record.get("record_type") == "SOURCE_RECORD"
        )
        try:
            source_records = {
                str(record["source_id"]): record for record in source_record_values
            }
        except KeyError as exc:
            raise EvidencePackageError("package source inventory differs") from exc
        if (
            len(source_records) != len(source_record_values)
            or set(source_records) != set(package.source_ids)
        ):
            raise EvidencePackageError("package source inventory differs")
        source_inventory = tuple(
            (source_id, str(source_records[source_id]["canonical_url"]))
            for source_id in package.source_ids
        )
        base = _base_package(package)
        resolved_records = validate_governed_evidence_records(
            candidate_id=package.candidate_id,
            source_inventory=source_inventory,
            base_package_digest=base.digest,
            package=package,
            retained_records=tuple(rows),
        )
        if resolved_records is None:
            raise EvidencePackageError("governed evidence records differ")
        resolved = replace(package, resolved_evidence_records=resolved_records)
        envelope = canonical_json_bytes(
            {
                "schema_identity": _SCHEMA,
                "receipt_id": acknowledgement.receipt_id,
                "handoff_id": acknowledgement.handoff_id,
                "candidate_version_id": version.version_id,
                "candidate_version_digest": version.canonical_digest,
                "governing_manifest_digest": manifest.canonical_digest,
                "source_admission_ids": [str(item) for item in source_admission_ids],
                "source_digests": source_digests,
                "record_admission_ids": [str(item) for item in record_admission_ids],
                "record_digests": [str(item[3]) for item in rows],
                "package": evidence_package_value(resolved),
                "package_digest": resolved.digest,
            }
        )
        return resolved, envelope, source_inventory

    def _require_decision(
        self,
        decision,
        *,
        policy_digest: str,
        object_class: str,
        allowed_use: str,
    ) -> None:
        if (
            decision.policy_contract_digest != policy_digest
            or decision.principal_id != self._reader_principal_id
            or decision.authority_domain != self._reader_authority_domain
            or decision.object_class != object_class
            or decision.allowed_use != allowed_use
        ):
            raise EvidencePackageError("governed evidence access policy differs")

    @staticmethod
    def _result(
        admission_id: ObjectAdmissionId,
        package: EvidencePackage,
        acknowledgement: IntakeAcknowledgement,
        source_ids: tuple[ObjectAdmissionId, ...],
        record_ids: tuple[ObjectAdmissionId, ...],
        source_inventory: tuple[tuple[str, str], ...],
    ) -> GovernedEvidencePackage:
        return GovernedEvidencePackage(
            admission_id,
            package,
            acknowledgement.receipt_id,
            acknowledgement.candidate_version_id,
            acknowledgement.candidate_version_digest,
            acknowledgement.governing_manifest_digest,
            source_ids,
            record_ids,
            source_inventory,
        )


def _base_package(package: EvidencePackage) -> EvidencePackage:
    return replace(
        package,
        substantive_new_information=(),
        governed_claims=(),
        qualification_evidence=(),
        selection_rationale="",
        geography=(),
        categories=(),
        evidence_gate_results=(),
        evidence_gate_evidence=(),
        freshness_result="MISSING",
        integrity_result="MISSING",
        explicit_exclusions=(),
        resolved_evidence_records=(),
    )


def _document(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidencePackageError("governed evidence object is malformed") from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise EvidencePackageError("governed evidence object is non-canonical")
    return value


def _tuple(value: object) -> tuple[Any, ...]:
    if type(value) is not list:
        raise EvidencePackageError("package sequence differs")
    return tuple(value)


def _package_from_value(raw: object) -> EvidencePackage:
    if type(raw) is not dict:
        raise EvidencePackageError("package value differs")
    try:
        value = dict(raw)
        claims = tuple(
            _claim_from_value(item) for item in _tuple(value["governed_claims"])
        )
        qualifications = tuple(
            QualificationEvidence(
                test=Evid012QualificationTest(item["test"]),
                governed_claim_id=item["governed_claim_id"],
                qualification_record_id=item["qualification_record_id"],
                test_evidence=tuple(map(tuple, item["test_evidence"])),
                policy_version=item["policy_version"],
            )
            for item in _tuple(value["qualification_evidence"])
        )
        gates = tuple(
            EvidenceGateEvidence(
                gate=item["gate"],
                result=item["result"],
                governed_claim_ids=_tuple(item["governed_claim_ids"]),
                policy_version=item["policy_version"],
            )
            for item in _tuple(value["evidence_gate_evidence"])
        )
        package = EvidencePackage(
            candidate_id=value["candidate_id"],
            hypothesis_id=value["hypothesis_id"],
            signal_ids=_tuple(value["signal_ids"]),
            lead_ids=_tuple(value["lead_ids"]),
            source_ids=_tuple(value["source_ids"]),
            observation_digests=_tuple(value["observation_digests"]),
            passages=_tuple(value["passages"]),
            substantive_new_information=_tuple(value["substantive_new_information"]),
            governed_claims=claims,
            qualification_evidence=qualifications,
            selection_rationale=value["selection_rationale"],
            geography=_tuple(value["geography"]),
            categories=_tuple(value["categories"]),
            evidence_gate_results=tuple(map(tuple, value["evidence_gate_results"])),
            evidence_gate_evidence=gates,
            freshness_result=value["freshness_result"],
            integrity_result=value["integrity_result"],
            explicit_exclusions=_tuple(value["explicit_exclusions"]),
            resolved_evidence_records=tuple(
                map(tuple, value["resolved_evidence_records"])
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidencePackageError("package value is invalid") from exc
    if evidence_package_value(package) != raw:
        raise EvidencePackageError("package value is non-canonical")
    return package


def _claim_from_value(raw: object) -> GovernedClaimEvidence:
    if type(raw) is not dict:
        raise EvidencePackageError("governed claim value differs")
    item = dict(raw)
    return GovernedClaimEvidence(
        **{
            **item,
            "source_ids": _tuple(item["source_ids"]),
            "source_record_ids": _tuple(item["source_record_ids"]),
            "source_authority_decision_ids": _tuple(
                item["source_authority_decision_ids"]
            ),
            "rights_decision_ids": _tuple(item["rights_decision_ids"]),
            "dependency_evidence_ids": _tuple(item["dependency_evidence_ids"]),
            "evidential_origin_ids": _tuple(item["evidential_origin_ids"]),
            "authority_class": ClaimAuthorityClass(item["authority_class"]),
            "status": GovernedClaimStatus(item["status"]),
            "localised_factual_expressions": tuple(
                map(tuple, item["localised_factual_expressions"])
            ),
            "named_entity_evidence": tuple(map(tuple, item["named_entity_evidence"])),
            "named_entities": _tuple(item["named_entities"]),
            "rendered_named_entities": _tuple(item["rendered_named_entities"]),
            "quotations": _tuple(item["quotations"]),
        }
    )


__all__ = [
    "EvidencePackageError",
    "GovernedEvidencePackage",
    "GovernedEvidencePackages",
]
