"""Typed initial Source Registry requests for the four missing native sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from newsroom.authority import AuthenticationProof
from newsroom.authority.types import UUIDv4Id
from newsroom.checks import deterministic_uuid4
from newsroom.increment9.proving import SOURCE_URLS
from newsroom.sources import (
    BaselinePolicy,
    BaselinePolicyKind,
    CoverageContribution,
    CoverageMapping,
    CoverageResponsibility,
    ObservationModel,
    PortfolioFunction,
    RightsReference,
    SourceDefinitionId,
    SourceDefinitionRequest,
    SourceDefinitionVersionId,
    SourceDefinitionVersionRequest,
    SourceDependency,
    SourceDependencyKind,
    SourceLifecycleStage,
    SourceRole,
    SourceRoleAssignment,
    VersionedPolicyRef,
)

from .native_evidence import PublicationRightsAssessment
from .govuk_rights import POLICY_VERSION as GOVUK_RIGHTS_POLICY_VERSION

VERSION = "hermes-native-source-definition-v1"
MISSING_SOURCE_IDS = ("UK-02", "UK-03", "UK-05", "HK-02")


@dataclass(frozen=True, slots=True)
class NativeSourceDefinitionRequests:
    source_id: str
    definition: SourceDefinitionRequest
    version: SourceDefinitionVersionRequest
    rights: PublicationRightsAssessment


@dataclass(frozen=True, slots=True)
class _Contract:
    name: str
    purpose: str
    role: SourceRole
    contribution: CoverageContribution
    geography: str
    language: str
    limitation: str
    observation: ObservationModel
    baseline: BaselinePolicyKind
    adapter: str
    dependency_id: str
    dependency: str
    obligation: str


_CONTRACTS = {
    "UK-02": _Contract(
        "UK-02 British National (Overseas) visa guide",
        "Observe complete revisions of the core British National (Overseas) visa guide.",
        SourceRole.ORIGINATING_AUTHORITY,
        CoverageContribution.REVISION_VISIBILITY,
        "UK", "en-GB",
        "Related guidance pages require their own Source Definition.",
        ObservationModel.MUTABLE_ITEM, BaselinePolicyKind.MAINTAINED_DOCUMENT,
        "govuk-content-api-complete-guide", "uk-02-complete-guide",
        "Every part declared by the maintained guide must be retained together.", "COV-021",
    ),
    "UK-03": _Contract(
        "UK-03 Immigration Rules manual",
        "Observe complete revisions of the Immigration Rules manual and all listed sections.",
        SourceRole.ORIGINATING_AUTHORITY,
        CoverageContribution.REVISION_VISIBILITY,
        "UK", "en-GB",
        "A root-index change is incomplete until every listed manual section is inspected.",
        ObservationModel.MUTABLE_ITEM, BaselinePolicyKind.MAINTAINED_DOCUMENT,
        "govuk-content-api-complete-manual", "uk-03-manual-sections",
        "Every section listed by the current manual index must be independently retained.", "COV-021",
    ),
    "UK-05": _Contract(
        "UK-05 Department for Education and Ofqual updates",
        "Observe education, examination and family-relevant policy revisions.",
        SourceRole.ORIGINATING_AUTHORITY,
        CoverageContribution.DETECTION_PATH,
        "UK", "en-GB",
        "Coverage is England-centred and excludes individual councils and schools.",
        ObservationModel.ROLLING_LIST, BaselinePolicyKind.BOUNDED_BACKFILL,
        "govuk-atom-complete-content-page", "uk-05-maintained-page",
        "Feed entries require inspection of the linked maintained page.", "COV-020",
    ),
    "HK-02": _Contract(
        "HK-02 Hong Kong Observatory warning summary",
        "Observe starts, changes and cancellations of Hong Kong weather warnings.",
        SourceRole.ORIGINATING_AUTHORITY,
        CoverageContribution.URGENT_FAST_PATH,
        "Hong Kong", "zh-HK",
        "The payload is current state; first observation does not establish activation time.",
        ObservationModel.COMPLETE_CURRENT_STATE,
        BaselinePolicyKind.COMPLETE_STATE_FIRST_OBSERVED_ACTIVE,
        "hko-warning-summary-json", "hk-02-current-warning-state",
        "Warning transitions depend on comparison with the retained preceding complete state.",
        "COV-023",
    ),
}

_RIGHTS_POLICY_VERSIONS = {
    "UK-02": GOVUK_RIGHTS_POLICY_VERSION,
    "UK-03": GOVUK_RIGHTS_POLICY_VERSION,
    "UK-05": GOVUK_RIGHTS_POLICY_VERSION,
    "HK-02": "hermes-observed-portfolio-rights-v1",
}


def native_source_definition_requests(
    *, source_id: str, rights: PublicationRightsAssessment,
) -> NativeSourceDefinitionRequests:
    """Build one initial definition/version after exact current rights permit use."""

    try:
        contract = _CONTRACTS[source_id]
    except KeyError:
        raise ValueError("native source does not need an initial definition") from None
    if (
        type(rights) is not PublicationRightsAssessment
        or rights.decision != "PERMITTED"
        or rights.permitted_use != "PUBLICATION_EVIDENCE"
    ):
        raise ValueError("native source definition requires current permitted rights")
    definition_id = deterministic_uuid4(
        SourceDefinitionId,
        namespace=f"{VERSION}:definition",
        semantic_value=[source_id, SOURCE_URLS[source_id]],
    )
    version_id = deterministic_uuid4(
        SourceDefinitionVersionId,
        namespace=f"{VERSION}:version",
        semantic_value=[source_id, SOURCE_URLS[source_id], rights.record_id,
                        rights.policy_digest, rights.evidence_digest],
    )
    rights_decision_id = deterministic_uuid4(
        UUIDv4Id,
        namespace=f"{VERSION}:rights-binding",
        semantic_value=[source_id, rights.record_id, rights.policy_digest,
                        rights.evidence_digest],
    )
    definition = SourceDefinitionRequest(
        definition_id, contract.name, contract.purpose,
        f"native-source-definition:{source_id}",
    )
    version = SourceDefinitionVersionRequest(
        version_id=version_id,
        definition_id=definition_id,
        version_number=1,
        expected_previous_version_id=None,
        locator=SOURCE_URLS[source_id],
        adapter_contract=VersionedPolicyRef(contract.adapter, "v1"),
        extraction_scope=(
            "body", "canonical_url", "headline", "published_at", "updated_at",
        ),
        rights=RightsReference(
            rights_decision_id=str(rights_decision_id),
            rights_policy_version=_RIGHTS_POLICY_VERSIONS[source_id],
            allowed_use="publication.evidence",
            retention_scope="authority.audit",
        ),
        roles=(SourceRoleAssignment(
            contract.role, contract.purpose, (contract.limitation,),
        ),),
        portfolio_functions=(PortfolioFunction.ANCHOR,),
        coverage_mappings=(CoverageMapping(
            contract.obligation, CoverageResponsibility.ACTIVE,
            contract.contribution, (contract.geography,), (contract.language,),
            (contract.limitation,),
        ),),
        dependencies=(SourceDependency(
            contract.dependency_id, SourceDependencyKind.ORIGINATING_MATERIAL,
            contract.dependency,
        ),),
        explicit_gaps=(),
        observation_model=contract.observation,
        baseline_policy=BaselinePolicy(
            VersionedPolicyRef("native-source-baseline", "v1"), contract.baseline,
            freshness_window_seconds=(
                7 * 24 * 60 * 60
                if contract.baseline is BaselinePolicyKind.BOUNDED_BACKFILL else None
            ),
            notes="Initial observation is retained without inventing earlier history.",
        ),
        item_identity_policy=VersionedPolicyRef("native-source-item-identity", "v1"),
        revision_policy=VersionedPolicyRef("native-effective-revision", "v1"),
        canonicalization_policy=VersionedPolicyRef("native-source-canonicalization", "v1"),
        lifecycle_stage=SourceLifecycleStage.SHADOW_SHORTLISTED,
        change_reason="Initial approved native source contract.",
        idempotency_key=f"native-source-definition-version:{source_id}:{rights.record_id}",
    )
    return NativeSourceDefinitionRequests(source_id, definition, version, rights)


def register_missing_native_source_definitions(
    *, sources, proof: AuthenticationProof,
    rights_by_source: Mapping[str, PublicationRightsAssessment],
) -> dict[str, SourceDefinitionId]:
    """Retain an approved subset without making one source block another."""

    if set(rights_by_source) - set(MISSING_SOURCE_IDS):
        raise ValueError("native source rights inventory exceeds missing sources")
    requests = tuple(
        native_source_definition_requests(
            source_id=source_id, rights=rights_by_source[source_id],
        )
        for source_id in MISSING_SOURCE_IDS if source_id in rights_by_source
    )
    for item in requests:
        try:
            current = sources.current_summary(
                item.definition.definition_id, proof=proof,
            )
        except LookupError:
            current = None
        if current is not None:
            retained = sources.version_details(current.version_id, proof=proof)
            if retained.request.locator != item.version.locator:
                raise ValueError("retained native source definition differs")
            continue
        sources.register_definition(item.definition, proof=proof)
        sources.record_definition_version(item.version, proof=proof)
    return {item.source_id: item.definition.definition_id for item in requests}


__all__ = [
    "MISSING_SOURCE_IDS", "NativeSourceDefinitionRequests",
    "native_source_definition_requests", "register_missing_native_source_definitions",
]
