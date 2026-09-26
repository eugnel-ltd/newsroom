"""Canonical provider-free Increment 4 readiness for the current Graphiti cohort.

This module composes existing Source Registry, Object, Graphiti-admission and
Increment 4 projection contracts.  It does not select or dispatch provider
work.  Callers must hold stable proving and unpublished-store snapshots while
planning and applying a bootstrap.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from newsroom.authority import (
    AuthenticationProof,
    CommandRegistry,
    HydrationPolicyRegistry,
    HydrationRequest,
    IdempotencyIdentityConflict,
    ObjectAdmissionDefinition,
    ObjectAdmissionRegistry,
    ObjectAdmissionRequest,
    ObjectLimits,
    PayloadSchemaRegistry,
    RightsPolicyContract,
    RightsPolicyRegistry,
    StaticAuthenticator,
    StaticAuthorizer,
    StaticPrincipal,
    UtcTimestamp,
)
from newsroom.authority.canonical import digest_bytes, digest_canonical
from newsroom.authority.graphiti_increment4_system import (
    GovernedGraphitiIncrement4AuthoritySystem,
    open_governed_graphiti_increment4_authority_system,
)
from newsroom.authority.types import UUIDv4Id
from newsroom.control_plane.broker import neo4j_projector_config
from newsroom.control_plane.corpus import CorpusAuthorityBinding, CorpusIngestUnit
from newsroom.control_plane.cycle import (
    _dispatch_rights_decision,
    load_graphiti_units_from_connection,
)
from newsroom.control_plane.graphiti_steady_state import (
    CAMPAIGN_PER_EVENT_SPEND_GBP_MICROUNITS,
    CAMPAIGN_RAMP_ADVANCE_CONDITIONS,
    CAMPAIGN_RAMP_ENTRY_CONDITIONS,
    CAMPAIGN_REQUIRED_STOP_CONDITIONS,
    CAMPAIGN_SCHEMA_VERSION,
    CAMPAIGN_SUCCESS_OBJECTIVE_BASE,
    GraphitiCampaignRuntime,
    campaign_event_limits,
    graphiti_graph_destination_identity,
    graphiti_graph_destination_readback,
    graphiti_operational_partition_snapshot,
    graphiti_store_snapshot_digests,
    validate_graphiti_campaign_packet,
)
from newsroom.control_plane.paths import (
    CANONICAL_GRAPHITI_WORKSPACE_ROOT,
    CANONICAL_INCREMENT4_AUTHORITY_STORE,
    CANONICAL_OBJECT_CAS_ROOT,
    CANONICAL_PROVING_STORE,
    CANONICAL_UNPUBLISHED_STORE,
    ensure_increment4_state_paths,
)
from newsroom.control_plane.read_only_snapshot import read_only_snapshot
from newsroom.entities.policy import (
    ENTITY_MENTION_ADMIT_COMMAND,
    ENTITY_RESOLUTION_DECIDE_COMMAND,
    ENTITY_RESOLUTION_DEPENDENCY_BIND_COMMAND,
    ENTITY_RESOLUTION_PROPOSE_COMMAND,
    entity_command_definitions,
)
from newsroom.entities.types import EntityReadPolicy
from newsroom.extraction.types import ExtractionReadPolicy
from newsroom.graphiti_adapter import (
    GraphitiAdapterReadPolicy,
    RecoveredAmbiguousProgressionProof,
)
from newsroom.graphiti_adapter.cursor_transport import CURSOR_SDK_TRANSPORT
from newsroom.graphiti_adapter.evaluation_attempt import (
    GRAPHITI_EVALUATION_HYDRATION_POLICY,
    evaluation_attempt_for_body,
)
from newsroom.graphiti_adapter.evaluation_packet import (
    CURSOR_AGENT_MODEL_ID,
    GRAPHITI_EMBEDDING_MODEL,
    GRAPHITI_EXTRACTION_TIMEOUT_MS,
    OPENROUTER_EMBEDDING_SLUG,
)
from newsroom.graphiti_adapter.identity import MAX_EPISODE_BYTES, typed_id
from newsroom.graphiti_adapter.models import GraphitiAttemptRequest
from newsroom.graphiti_adapter.policy import graphiti_adapter_command_definitions
from newsroom.increment4 import increment4_admitted_contract_registry
from newsroom.increment4.contracts import INCREMENT4_ADMITTED_FAMILY_ID
from newsroom.increment4.neo4j import Increment4Neo4jCurrentBuildRequest
from newsroom.projection import (
    ProjectionFamilyKind,
    ProjectionGenerationId,
    ProjectionGenerationState,
)
from newsroom.projection.models import ProjectionReadPolicy
from newsroom.projection.neo4j import StructuralReconciliationView
from newsroom.relations.editorial_models import EditorialRelationReadPolicy
from newsroom.sources.definition_models import (
    SourceDefinitionRequest,
    SourceDefinitionVersionRequest,
)
from newsroom.sources.item_models import SourceItemRequest
from newsroom.sources.observation_models import (
    DiscoveryRepresentationRequest,
    SourceRevisionRequest,
)
from newsroom.sources.policy import (
    source_registry_command_definitions,
    source_registry_payload_contracts,
)
from newsroom.sources.types import (
    BaselinePolicy,
    BaselinePolicyKind,
    CoverageContribution,
    CoverageMapping,
    CoverageResponsibility,
    DiscoveryRepresentationId,
    IdentityComponent,
    ObservationModel,
    PortfolioFunction,
    RightsReference,
    SourceDefinitionId,
    SourceDefinitionVersionId,
    SourceDependency,
    SourceDependencyKind,
    SourceItemId,
    SourceItemIdentityKind,
    SourceLifecycleStage,
    SourceRegistryReadPolicy,
    SourceRevisionId,
    SourceRole,
    SourceRoleAssignment,
    SourceTime,
    VersionedPolicyRef,
)

OPERATOR_PRINCIPAL_ID = "newsroom.control-plane"
OPERATOR_AUTHORITY_DOMAIN = "newsroom.evaluation"
OPERATIONAL_ADMISSION_TYPE = "graphiti.evaluation.passage"
OPERATIONAL_SELECTION_POLICY_ID = "graphiti-operational-current-cohort"
OPERATIONAL_SELECTION_POLICY_VERSION = "v1"

# Source Registry requires immutable semantic version labels.  These v1 labels
# describe the existing retained Control Plane transformation used here; they
# grant no source execution authority and are not aliases for an Increment 9
# authority store.
_SOURCE_ADAPTER_POLICY = VersionedPolicyRef(
    "control-plane-retained-source-adapter", "v1"
)
_SOURCE_BASELINE_POLICY = VersionedPolicyRef(
    "operational-current-cohort-baseline", "v1"
)
_SOURCE_ITEM_IDENTITY_POLICY = VersionedPolicyRef(
    "source-id-item-key-composite", "v1"
)
_SOURCE_REVISION_POLICY = VersionedPolicyRef(
    "effective-revision-identity", "v1"
)
_SOURCE_CANONICALIZATION_POLICY = VersionedPolicyRef(
    "graphiti-corpus-canonicalization", "v1"
)
_SOURCE_BASELINE_FRESHNESS_SECONDS = 7 * 24 * 60 * 60
_SOURCE_ADAPTER_VERSION = "control-plane-retained-source-adapter-v1"
_SOURCE_PARSER_VERSION = "control-plane-parse-observation-v1"
_SOURCE_NORMALIZER_VERSION = "graphiti-corpus-canonicalization-v1"
_SOURCE_EXTRACTION_SCOPE_VERSION = "operational-graphiti-passage-fields-v1"

_RIGHTS_POLICY = RightsPolicyContract(
    policy_key="graphiti-current-proving-rights",
    contract_version="v1",
    implementation_version="control-plane-dispatch-rights-v1",
    preflight_allowed=True,
    reason_code="CURRENT_PROVING_RIGHTS_VERIFIED",
    preflight_ttl_seconds=60,
)

_SOURCE_RECORD_TYPES = frozenset(
    {
        "SOURCE_DEFINITION",
        "SOURCE_DEFINITION_VERSION",
        "SOURCE_ITEM",
        "SOURCE_REVISION",
        "DISCOVERY_REPRESENTATION",
    }
)

class GraphitiOperationalReadinessError(RuntimeError):
    """The exact provider-free operational readiness contract failed closed."""


@dataclass(frozen=True, slots=True)
class _AcceptedSourceContract:
    name: str
    role: SourceRole
    function: PortfolioFunction
    purpose: str
    limitations: tuple[str, ...]
    coverage_obligation_id: str
    coverage_responsibility: CoverageResponsibility
    coverage_contribution: CoverageContribution
    coverage_geographies: tuple[str, ...]
    coverage_languages: tuple[str, ...]
    dependencies: tuple[SourceDependency, ...]
    observation_model: ObservationModel
    baseline_kind: BaselinePolicyKind
    baseline_notes: str
    lifecycle_stage: SourceLifecycleStage


_ACCEPTED_SOURCE_CONTRACTS = {
    "UK-01": _AcceptedSourceContract(
        name="UK-01 Home Office + UKVI Atom",
        role=SourceRole.ORIGINATING_AUTHORITY,
        function=PortfolioFunction.ANCHOR,
        purpose=(
            "Observe Home Office and UKVI immigration, status and guidance updates."
        ),
        limitations=("Entry metadata still needs maintained-page inspection.",),
        coverage_obligation_id="COV-020",
        coverage_responsibility=CoverageResponsibility.ACTIVE,
        coverage_contribution=CoverageContribution.DETECTION_PATH,
        coverage_geographies=("UK",),
        coverage_languages=("en-GB",),
        dependencies=(
            SourceDependency(
                dependency_id="uk-01-maintained-page",
                kind=SourceDependencyKind.ORIGINATING_MATERIAL,
                description=(
                    "Feed entries depend on inspection of the linked maintained page "
                    "before a document revision is established."
                ),
            ),
        ),
        observation_model=ObservationModel.ROLLING_LIST,
        baseline_kind=BaselinePolicyKind.BOUNDED_BACKFILL,
        baseline_notes=(
            "Bootstrap only feed entries inside the bounded freshness window."
        ),
        lifecycle_stage=SourceLifecycleStage.SHADOW_SHORTLISTED,
    ),
    "UK-10": _AcceptedSourceContract(
        name="UK-10 Met Office warnings",
        role=SourceRole.ORIGINATING_AUTHORITY,
        function=PortfolioFunction.ANCHOR,
        purpose="Observe current Met Office severe-weather warnings.",
        limitations=("Regional and transition semantics remain Topic 5 work.",),
        coverage_obligation_id="COV-023",
        coverage_responsibility=CoverageResponsibility.ACTIVE,
        coverage_contribution=CoverageContribution.URGENT_FAST_PATH,
        coverage_geographies=("UK",),
        coverage_languages=("en-GB",),
        dependencies=(),
        observation_model=ObservationModel.COMPLETE_CURRENT_STATE,
        baseline_kind=BaselinePolicyKind.COMPLETE_STATE_FIRST_OBSERVED_ACTIVE,
        baseline_notes=(
            "Existing warnings are first-observed active; baseline time does not "
            "establish their activation time."
        ),
        lifecycle_stage=SourceLifecycleStage.SHADOW_SHORTLISTED,
    ),
    "HK-01": _AcceptedSourceContract(
        name="HK-01 news.gov.hk top stories",
        role=SourceRole.SPECIALIST_OR_LOCAL_RADAR,
        function=PortfolioFunction.ANCHOR,
        purpose=(
            "Observe selected major Hong Kong policy, service and breaking-news items."
        ),
        limitations=("Curated, not the full government-release universe.",),
        coverage_obligation_id="COV-012",
        coverage_responsibility=CoverageResponsibility.ACTIVE,
        coverage_contribution=CoverageContribution.DETECTION_PATH,
        coverage_geographies=("Hong Kong",),
        coverage_languages=("zh-HK",),
        dependencies=(
            SourceDependency(
                dependency_id="hk-01-editorial-selection",
                kind=SourceDependencyKind.EDITORIAL_SELECTION,
                description=(
                    "Feed inclusion depends on news.gov.hk editorial selection."
                ),
            ),
        ),
        observation_model=ObservationModel.ROLLING_LIST,
        baseline_kind=BaselinePolicyKind.BOUNDED_BACKFILL,
        baseline_notes=(
            "Bootstrap only feed entries inside the bounded freshness window."
        ),
        lifecycle_stage=SourceLifecycleStage.SHADOW_SHORTLISTED,
    ),
    "HK-04": _AcceptedSourceContract(
        name="HK-04 Education Bureau latest news",
        role=SourceRole.ORIGINATING_AUTHORITY,
        function=PortfolioFunction.ANCHOR,
        purpose="Observe Hong Kong education and bureau-service updates.",
        limitations=("No school-level completeness.",),
        coverage_obligation_id="COV-012",
        coverage_responsibility=CoverageResponsibility.ACTIVE,
        coverage_contribution=CoverageContribution.DETECTION_PATH,
        coverage_geographies=("Hong Kong",),
        coverage_languages=("zh-HK",),
        dependencies=(),
        observation_model=ObservationModel.ROLLING_LIST,
        baseline_kind=BaselinePolicyKind.BOUNDED_BACKFILL,
        baseline_notes=(
            "Bootstrap only feed entries inside the bounded freshness window."
        ),
        lifecycle_stage=SourceLifecycleStage.SHADOW_SHORTLISTED,
    ),
    "RAD-01": _AcceptedSourceContract(
        name="RAD-01 RTHK local news",
        role=SourceRole.ESTABLISHED_MEDIA_RADAR,
        function=PortfolioFunction.COMPARATOR,
        purpose=(
            "Observe Hong Kong unscheduled events, public affairs and lived-impact "
            "leads."
        ),
        limitations=(
            "Lead-only discovery role.",
            "No publisher-body republication or image-model submission.",
        ),
        coverage_obligation_id="COV-012",
        coverage_responsibility=CoverageResponsibility.EVALUATION,
        coverage_contribution=CoverageContribution.COMPARATOR,
        coverage_geographies=("Hong Kong",),
        coverage_languages=("zh-HK",),
        dependencies=(
            SourceDependency(
                dependency_id="rad-01-editorial-selection",
                kind=SourceDependencyKind.EDITORIAL_SELECTION,
                description="Feed inclusion depends on RTHK editorial selection.",
            ),
        ),
        observation_model=ObservationModel.ROLLING_LIST,
        baseline_kind=BaselinePolicyKind.BOUNDED_BACKFILL,
        baseline_notes=(
            "Bootstrap only feed entries inside the bounded freshness window."
        ),
        lifecycle_stage=SourceLifecycleStage.COMPARATOR_ONLY,
    ),
    "RAD-02": _AcceptedSourceContract(
        name="RAD-02 BBC UK news",
        role=SourceRole.ESTABLISHED_MEDIA_RADAR,
        function=PortfolioFunction.COMPARATOR,
        purpose="Observe UK major incidents and official-list blind-spot leads.",
        limitations=(
            "Broad, duplicate-prone and weak on local completeness.",
            "No publisher-body republication or image-model submission.",
        ),
        coverage_obligation_id="COV-010",
        coverage_responsibility=CoverageResponsibility.EVALUATION,
        coverage_contribution=CoverageContribution.COMPARATOR,
        coverage_geographies=("UK",),
        coverage_languages=("en-GB",),
        dependencies=(
            SourceDependency(
                dependency_id="rad-02-editorial-selection",
                kind=SourceDependencyKind.EDITORIAL_SELECTION,
                description="Feed inclusion depends on BBC editorial selection.",
            ),
        ),
        observation_model=ObservationModel.ROLLING_LIST,
        baseline_kind=BaselinePolicyKind.BOUNDED_BACKFILL,
        baseline_notes=(
            "Bootstrap only feed entries inside the bounded freshness window."
        ),
        lifecycle_stage=SourceLifecycleStage.COMPARATOR_ONLY,
    ),
}


@dataclass(frozen=True, slots=True)
class OperationalAuthorityBootstrapPlan:
    observed_at: UtcTimestamp
    partition_snapshot: Mapping[str, object]
    candidate_events: tuple[Mapping[str, object], ...]
    units: tuple[CorpusIngestUnit, ...]
    rights_by_source: tuple[tuple[str, Mapping[str, object]], ...]
    revision_predecessors: tuple[tuple[str, str | None], ...]
    cohort_digest: str
    plan_digest: str

    def rights_for(self, source_id: str) -> Mapping[str, object]:
        try:
            return dict(self.rights_by_source)[source_id]
        except KeyError as exc:
            raise GraphitiOperationalReadinessError(
                f"current rights are unavailable for {source_id}"
            ) from exc

    def prior_revision_for(
        self, unit: CorpusIngestUnit
    ) -> SourceRevisionId | None:
        authority = unit.authority
        if authority is None:
            raise GraphitiOperationalReadinessError(
                "current corpus unit lacks its retained source identity"
            )
        predecessors = dict(self.revision_predecessors)
        if authority.revision_id not in predecessors:
            raise GraphitiOperationalReadinessError(
                "current source revision lacks an exact predecessor binding"
            )
        prior = predecessors[authority.revision_id]
        return None if prior is None else SourceRevisionId.parse(prior)


def _operational_plan_semantic_digest(
    *,
    candidate_events: tuple[Mapping[str, object], ...],
    rights_by_source: tuple[tuple[str, Mapping[str, object]], ...],
    revision_predecessors: tuple[tuple[str, str | None], ...],
) -> str:
    """Bind the durable plan semantics while excluding observation metadata."""

    events = sorted(
        (
            {
                "ledger_seq": event["ledger_seq"],
                "event_id": event["event_id"],
                "manifest_digest": event["manifest_digest"],
                "ingest_ids": list(event["ingest_ids"]),
            }
            for event in candidate_events
        ),
        key=lambda event: (int(event["ledger_seq"]), str(event["event_id"])),
    )
    return digest_canonical(
        {
            "schema_version": "newsroom.graphiti-operational-plan-semantics.v1",
            "candidate_events": events,
            "rights": [
                {
                    "source_id": source_id,
                    **{
                        key: value
                        for key, value in rights.items()
                        if key != "evaluated_at"
                    },
                }
                for source_id, rights in sorted(rights_by_source)
            ],
            "revision_predecessors": [
                {
                    "revision_id": revision_id,
                    "prior_revision_id": prior_revision_id,
                }
                for revision_id, prior_revision_id in sorted(revision_predecessors)
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class OperationalAuthorityBootstrapResult:
    observed_at: UtcTimestamp
    plan_digest: str
    cohort_digest: str
    plan_semantic_digest: str
    candidate_event_count: int
    source_count: int
    unit_count: int
    bound_units: tuple[CorpusIngestUnit, ...]
    provider_calls: int = 0

    def canonical_value(self) -> dict[str, object]:
        return {
            "schema_version": "newsroom.graphiti-operational-bootstrap.v1",
            "observed_at": self.observed_at.to_text(),
            "plan_digest": self.plan_digest,
            "cohort_digest": self.cohort_digest,
            "plan_semantic_digest": self.plan_semantic_digest,
            "candidate_event_count": self.candidate_event_count,
            "source_count": self.source_count,
            "unit_count": self.unit_count,
            "bound_ingest_ids": [item.ingest_id for item in self.bound_units],
            "provider_calls": self.provider_calls,
        }


def _evaluation_attempt_for_unit(
    unit: CorpusIngestUnit,
    *,
    recovered_ambiguous_progression: RecoveredAmbiguousProgressionProof | None = None,
) -> GraphitiAttemptRequest:
    authority = unit.authority
    if authority is None:
        raise GraphitiOperationalReadinessError(
            "current corpus unit lacks its retained source identity"
        )
    return evaluation_attempt_for_body(
        episode_body=unit.episode_body,
        ingest_id=unit.ingest_id,
        proving_run_id=unit.proving_run_id,
        source_id=unit.source_id,
        item_key=unit.item_key,
        observation_digest=unit.observation_digest,
        published_at=unit.published_at,
        updated_at=unit.updated_at,
        effective_revision=unit.effective_revision,
        canonical_url=unit.canonical_url,
        revision_digest=unit.revision_digest,
        representation_digest=unit.representation_digest,
        authority_ids=(
            authority.admission_id,
            authority.access_decision_id,
            authority.definition_id,
            authority.definition_version_id,
            authority.item_id,
            authority.revision_id,
            authority.representation_id,
        ),
        attempt_number=unit.attempt_number,
        predecessor_episode_uuid=unit.predecessor_ingest_id,
        recovered_ambiguous_progression=recovered_ambiguous_progression,
    )


def _operational_generation_identity(
    *,
    graph_destination_id: str,
    plan_semantic_digest: str,
    bootstrap: OperationalAuthorityBootstrapResult,
    configuration: Mapping[str, object],
) -> tuple[ProjectionGenerationId, str]:
    """Derive one resumable generation from durable semantic inputs only."""

    bound_units: list[dict[str, object]] = []
    for unit in sorted(bootstrap.bound_units, key=lambda item: item.ingest_id):
        authority = unit.authority
        if authority is None:
            raise GraphitiOperationalReadinessError(
                "current corpus unit lacks its retained source identity"
            )
        bound_units.append(
            {
                "ingest_id": unit.ingest_id,
                "authority_ids": [
                    authority.admission_id,
                    authority.access_decision_id,
                    authority.definition_id,
                    authority.definition_version_id,
                    authority.item_id,
                    authority.revision_id,
                    authority.representation_id,
                ],
            }
        )
    semantic_digest = digest_canonical(
        {
            "schema_version": "newsroom.graphiti-operational-generation-identity.v1",
            "plan_semantic_digest": plan_semantic_digest,
            "bound_units": bound_units,
            "configuration": {
                "configuration_id": configuration["configuration_id"],
                "canonical_digest": digest_canonical(dict(configuration)),
            },
        }
    )
    generation_id = typed_id(
        ProjectionGenerationId,
        "canonical-operational-increment4-v2",
        str(CANONICAL_INCREMENT4_AUTHORITY_STORE),
        INCREMENT4_ADMITTED_FAMILY_ID,
        graph_destination_id,
        semantic_digest,
    )
    return generation_id, semantic_digest


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _source_time(value: str | None) -> SourceTime:
    if not value:
        return SourceTime.unknown()
    try:
        return SourceTime.exact(UtcTimestamp.parse(value))
    except ValueError:
        return SourceTime.unknown()


def _source_contract_shape(
    source_id: str,
) -> tuple[SourceRole, PortfolioFunction]:
    contract = _accepted_source_contract(source_id)
    return contract.role, contract.function


def _accepted_source_contract(source_id: str) -> _AcceptedSourceContract:
    try:
        return _ACCEPTED_SOURCE_CONTRACTS[source_id]
    except KeyError as exc:
        raise GraphitiOperationalReadinessError(
            f"source {source_id!r} is outside the accepted operational cohort"
        ) from exc


def _source_version_rights_decision_id(
    authority_store: sqlite3.Connection | None,
    unit: CorpusIngestUnit,
    rights: Mapping[str, object],
) -> str:
    """Keep the immutable source-version rights identity when the packet is unchanged.

    proving_run_id is dispatch/access identity. The version command is keyed by
    source and packet digest, so a later poll of the same packet must replay the
    retained rights_decision_id instead of minting a new one.
    """

    packet_digest = str(rights.get("packet_digest") or "")
    run_id = str(rights.get("rights_authority_run_id") or "")
    gate_id = str(rights.get("gate_id") or "")
    current_id = str(
        typed_id(
            UUIDv4Id,
            "retained-rights-packet",
            run_id,
            gate_id,
            packet_digest,
        )
    )
    if authority_store is None or unit.authority is None:
        return current_id
    tables = {
        str(name)
        for (name,) in authority_store.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ("
            "'source_definition_versions','ledger_events','authority_commands')"
        )
    }
    if tables != {
        "source_definition_versions",
        "ledger_events",
        "authority_commands",
    }:
        return current_id
    version_id = unit.authority.definition_version_id
    if (
        authority_store.execute(
            "SELECT 1 FROM source_definition_versions WHERE version_id=?",
            (version_id,),
        ).fetchone()
        is None
    ):
        return current_id
    retained = authority_store.execute(
        "SELECT v.rights_decision_id,c.idempotency_key "
        "FROM source_definition_versions v "
        "JOIN ledger_events e ON e.event_id=v.authority_event_id "
        "JOIN authority_commands c ON c.command_id=e.command_id "
        "WHERE v.version_id=?",
        (version_id,),
    ).fetchone()
    if retained is None:
        raise GraphitiOperationalReadinessError(
            "retained source version lacks its exact command identity"
        )
    if str(retained[1]) != (
        f"issue895-source-version:{unit.source_id}:{packet_digest}"
    ):
        raise GraphitiOperationalReadinessError(
            "retained source version requires an explicit rights-packet change"
        )
    return str(retained[0])


def _source_requests(
    unit: CorpusIngestUnit,
    rights: Mapping[str, object],
    *,
    prior_revision_id: SourceRevisionId | None = None,
    authority_store: sqlite3.Connection | None = None,
) -> tuple[
    SourceDefinitionRequest,
    SourceDefinitionVersionRequest,
    SourceItemRequest,
    SourceRevisionRequest,
    DiscoveryRepresentationRequest,
]:
    authority = unit.authority
    if authority is None:
        raise GraphitiOperationalReadinessError(
            "current corpus unit lacks its retained source identity"
        )
    contract = _accepted_source_contract(unit.source_id)
    packet_digest = str(rights.get("packet_digest") or "")
    run_id = str(rights.get("rights_authority_run_id") or "")
    gate_id = str(rights.get("gate_id") or "")
    if not packet_digest or not run_id or not gate_id:
        raise GraphitiOperationalReadinessError(
            "current rights evidence lacks its exact retained identity"
        )
    definition_id = SourceDefinitionId.parse(authority.definition_id)
    version_id = SourceDefinitionVersionId.parse(authority.definition_version_id)
    item_id = SourceItemId.parse(authority.item_id)
    revision_id = SourceRevisionId.parse(authority.revision_id)
    representation_id = DiscoveryRepresentationId.parse(authority.representation_id)
    definition = SourceDefinitionRequest(
        definition_id=definition_id,
        name=contract.name,
        editorial_purpose=contract.purpose,
        idempotency_key=f"issue895-source-definition:{unit.source_id}",
    )
    version = SourceDefinitionVersionRequest(
        version_id=version_id,
        definition_id=definition_id,
        version_number=1,
        expected_previous_version_id=None,
        locator=unit.source_definition_url,
        adapter_contract=_SOURCE_ADAPTER_POLICY,
        extraction_scope=(
            "body",
            "canonical_url",
            "headline",
            "published_at",
            "updated_at",
        ),
        rights=RightsReference(
            rights_decision_id=_source_version_rights_decision_id(
                authority_store, unit, rights
            ),
            rights_policy_version=_RIGHTS_POLICY.implementation_version,
            allowed_use="proposal.extraction",
            retention_scope="disposable-workspace",
        ),
        roles=(
            SourceRoleAssignment(
                role=contract.role,
                purpose=contract.purpose,
                limitations=contract.limitations,
            ),
        ),
        portfolio_functions=(contract.function,),
        coverage_mappings=(
            CoverageMapping(
                obligation_id=contract.coverage_obligation_id,
                responsibility=contract.coverage_responsibility,
                contribution=contract.coverage_contribution,
                geographies=contract.coverage_geographies,
                languages=contract.coverage_languages,
                limitations=contract.limitations,
            ),
        ),
        dependencies=contract.dependencies,
        # Portfolio-wide unresolved work has no source-specific governed gap IDs.
        # Its absence here must not be rewritten as synthetic Source Registry gaps.
        explicit_gaps=(),
        observation_model=contract.observation_model,
        baseline_policy=BaselinePolicy(
            reference=_SOURCE_BASELINE_POLICY,
            kind=contract.baseline_kind,
            freshness_window_seconds=(
                _SOURCE_BASELINE_FRESHNESS_SECONDS
                if contract.baseline_kind is BaselinePolicyKind.BOUNDED_BACKFILL
                else None
            ),
            notes=contract.baseline_notes,
        ),
        item_identity_policy=_SOURCE_ITEM_IDENTITY_POLICY,
        revision_policy=_SOURCE_REVISION_POLICY,
        canonicalization_policy=_SOURCE_CANONICALIZATION_POLICY,
        lifecycle_stage=contract.lifecycle_stage,
        change_reason="Initial canonical provider-free Graphiti readiness baseline.",
        idempotency_key=f"issue895-source-version:{unit.source_id}:{packet_digest}",
    )
    item = SourceItemRequest(
        item_id=item_id,
        definition_id=definition_id,
        definition_version_id=version_id,
        identity_kind=SourceItemIdentityKind.COMPOSITE,
        identity_policy=version.item_identity_policy,
        source_native_id=None,
        identity_components=(
            IdentityComponent("item_key", unit.item_key),
            IdentityComponent("source_id", unit.source_id),
        ),
        uncertainties=(),
        idempotency_key=f"issue895-source-item:{item_id}",
    )
    revision = SourceRevisionRequest(
        revision_id=revision_id,
        item_id=item_id,
        definition_version_id=version_id,
        prior_revision_id=prior_revision_id,
        source_native_revision_token=None,
        permitted_state_digest=unit.revision_digest,
        revision_policy=version.revision_policy,
        canonicalizer_version=_SOURCE_NORMALIZER_VERSION,
        source_published_time=_source_time(unit.published_at),
        source_updated_time=_source_time(unit.updated_at),
        observed_at=UtcTimestamp.parse(unit.coverage_first_observed_at),
        idempotency_key=f"issue895-source-revision:{revision_id}",
    )
    representation = DiscoveryRepresentationRequest(
        representation_id=representation_id,
        revision_id=revision_id,
        definition_version_id=version_id,
        adapter_version=_SOURCE_ADAPTER_VERSION,
        parser_version=_SOURCE_PARSER_VERSION,
        normalizer_version=_SOURCE_NORMALIZER_VERSION,
        extraction_scope_version=_SOURCE_EXTRACTION_SCOPE_VERSION,
        permitted_fields_digest=digest_canonical(
            {
                "headline": unit.headline,
                "body": unit.body,
                "canonical_url": unit.canonical_url,
                "published_at": unit.published_at,
                "updated_at": unit.updated_at,
            }
        ),
        representation_digest=unit.representation_digest,
        produced_at=UtcTimestamp.parse(unit.coverage_first_observed_at),
        idempotency_key=f"issue895-representation:{representation_id}",
    )
    return definition, version, item, revision, representation


def _preflight_source_revision_semantics(
    authority: sqlite3.Connection,
    requests: tuple[SourceRevisionRequest, ...],
) -> None:
    """Reject SourceRevision identity conflicts before any authority write."""

    planned: dict[tuple[str, str], str] = {}
    for request in requests:
        semantic_key = (
            str(request.item_id),
            request.revision_identity_digest,
        )
        prior_id = planned.setdefault(semantic_key, str(request.revision_id))
        if prior_id != str(request.revision_id):
            raise GraphitiOperationalReadinessError(
                "current cohort allocates multiple SourceRevision identities "
                "to one permitted source state"
            )

    table = authority.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_revisions'"
    ).fetchone()
    if table is None:
        return
    retained_rows = authority.execute(
        "SELECT revision_id,item_id,revision_identity_digest,canonical_digest "
        "FROM source_revisions"
    ).fetchall()
    retained_by_id = {str(row[0]): row for row in retained_rows}
    retained_by_semantics: dict[tuple[str, str], object] = {}
    for row in retained_rows:
        semantic_key = (str(row[1]), str(row[2]))
        prior = retained_by_semantics.setdefault(semantic_key, row)
        if str(prior[0]) != str(row[0]):
            raise GraphitiOperationalReadinessError(
                "canonical authority retains conflicting SourceRevision semantics"
            )

    for request in requests:
        revision_id = str(request.revision_id)
        semantic_key = (
            str(request.item_id),
            request.revision_identity_digest,
        )
        retained = retained_by_id.get(revision_id)
        if retained is not None:
            if (
                str(retained[1]) != semantic_key[0]
                or str(retained[2]) != semantic_key[1]
                or str(retained[3]) != request.digest
            ):
                raise GraphitiOperationalReadinessError(
                    "retained SourceRevision identity differs from the exact "
                    "bootstrap request"
                )
            continue
        retained = retained_by_semantics.get(semantic_key)
        if retained is not None:
            raise GraphitiOperationalReadinessError(
                "SourceRevision semantics already belong to another retained identity"
            )


def _revision_predecessor_bindings(
    units: tuple[CorpusIngestUnit, ...],
) -> tuple[tuple[str, str | None], ...]:
    """Order each exact eligible item's retained revisions by first observation."""

    by_item: dict[str, dict[str, CorpusIngestUnit]] = {}
    for unit in units:
        authority = unit.authority
        if authority is None:
            raise GraphitiOperationalReadinessError(
                "current corpus unit lacks its retained source identity"
            )
        revisions = by_item.setdefault(authority.item_id, {})
        retained = revisions.setdefault(authority.revision_id, unit)
        if (
            retained.source_id,
            retained.item_key,
            retained.revision_digest,
            retained.published_at,
            retained.updated_at,
            retained.coverage_first_observed_at,
        ) != (
            unit.source_id,
            unit.item_key,
            unit.revision_digest,
            unit.published_at,
            unit.updated_at,
            unit.coverage_first_observed_at,
        ):
            raise GraphitiOperationalReadinessError(
                "one retained revision identity has conflicting source semantics"
            )
    bindings: dict[str, str | None] = {}
    for revisions in by_item.values():
        ordered = sorted(
            revisions.values(),
            key=lambda unit: (
                UtcTimestamp.parse(unit.coverage_first_observed_at).to_text(),
                unit.updated_at or "",
                unit.published_at or "",
                unit.authority.revision_id if unit.authority is not None else "",
            ),
        )
        prior: str | None = None
        for unit in ordered:
            assert unit.authority is not None
            bindings[unit.authority.revision_id] = prior
            prior = unit.authority.revision_id
    return tuple(sorted(bindings.items()))


def plan_operational_authority_bootstrap(
    proving: sqlite3.Connection,
    unpublished: sqlite3.Connection,
    authority: sqlite3.Connection,
    *,
    observed_at: datetime,
) -> OperationalAuthorityBootstrapPlan:
    """Plan the one current-cohort bootstrap without provider or store effects."""

    if observed_at.tzinfo is None:
        raise GraphitiOperationalReadinessError(
            "operational bootstrap observation must be timezone-aware"
        )
    partition = graphiti_operational_partition_snapshot(
        proving,
        unpublished,
        authority=authority,
        observed_at=observed_at,
    )
    actionable = partition.get("actionable")
    if not isinstance(actionable, list) or not actionable:
        raise GraphitiOperationalReadinessError(
            "the exact operational cohort has no fresh candidate events"
        )
    if any(
        not isinstance(item, Mapping) or item.get("kind") != "FRESH_EVENT"
        for item in actionable
    ):
        raise GraphitiOperationalReadinessError(
            "event gaps must be reconciled before authority bootstrap"
        )
    candidate_events = tuple(dict(item) for item in actionable)
    candidate_ingest_ids = tuple(
        str(ingest_id)
        for event in candidate_events
        for ingest_id in event.get("ingest_ids", ())
    )
    if not candidate_ingest_ids or len(candidate_ingest_ids) != len(
        set(candidate_ingest_ids)
    ):
        raise GraphitiOperationalReadinessError(
            "operational candidate ingest membership is empty or duplicated"
        )
    all_units = load_graphiti_units_from_connection(
        proving,
        evaluated_at=observed_at,
    )
    by_ingest = {unit.ingest_id: unit for unit in all_units}
    if len(by_ingest) != len(all_units) or set(candidate_ingest_ids) - set(by_ingest):
        raise GraphitiOperationalReadinessError(
            "operational candidate units differ from current retained input"
        )
    units = tuple(by_ingest[ingest_id] for ingest_id in candidate_ingest_ids)
    revision_predecessors = _revision_predecessor_bindings(units)
    predecessor_map = dict(revision_predecessors)
    rights_by_source: list[tuple[str, Mapping[str, object]]] = []
    for source_id in sorted({unit.source_id for unit in units}):
        source_urls = {
            unit.source_definition_url for unit in units if unit.source_id == source_id
        }
        if len(source_urls) != 1:
            raise GraphitiOperationalReadinessError(
                f"{source_id} has ambiguous current source identity"
            )
        decision = _dispatch_rights_decision(
            proving,
            source_id=source_id,
            source_url=next(iter(source_urls)),
            evaluated_at=_utc_text(observed_at),
        )
        if decision is None:
            raise GraphitiOperationalReadinessError(
                f"{source_id} lacks current exact Graphiti dispatch rights"
            )
        rights_by_source.append((source_id, decision))
    rights_map = dict(rights_by_source)
    request_digests: list[dict[str, object]] = []
    revision_requests: list[SourceRevisionRequest] = []
    for unit in units:
        assert unit.authority is not None
        prior = predecessor_map[unit.authority.revision_id]
        requests = _source_requests(
            unit,
            rights_map[unit.source_id],
            prior_revision_id=(
                None if prior is None else SourceRevisionId.parse(prior)
            ),
            authority_store=authority,
        )
        revision_requests.append(requests[3])
        request_digests.append(
            {
                "ingest_id": unit.ingest_id,
                "episode_digest": digest_bytes(
                    " ".join(unit.episode_body.split()).encode("utf-8")
                ),
                "source_request_digests": [item.digest for item in requests],
                "rights_packet_digest": rights_map[unit.source_id]["packet_digest"],
            }
        )
    _preflight_source_revision_semantics(authority, tuple(revision_requests))
    cohort_value = {
        "partition_snapshot_digest": partition["snapshot_digest"],
        "candidate_events": [dict(item) for item in candidate_events],
        "units": request_digests,
        "rights": [
            {"source_id": source_id, **dict(rights)}
            for source_id, rights in rights_by_source
        ],
    }
    cohort_digest = digest_canonical(cohort_value)
    plan_value = {
        "schema_version": "newsroom.graphiti-operational-bootstrap-plan.v1",
        "observed_at": _utc_text(observed_at),
        "cohort_digest": cohort_digest,
        "provider_calls": 0,
        "historical_graphiti_imports": 0,
    }
    return OperationalAuthorityBootstrapPlan(
        observed_at=UtcTimestamp.parse(_utc_text(observed_at)),
        partition_snapshot=dict(partition),
        candidate_events=candidate_events,
        units=units,
        rights_by_source=tuple(rights_by_source),
        revision_predecessors=revision_predecessors,
        cohort_digest=cohort_digest,
        plan_digest=digest_canonical(plan_value),
    )


class OperationalCorpusAuthorityBinder:
    """Bind prepared units to real current Source/Object authority records."""

    def __init__(
        self,
        *,
        authority_system: GovernedGraphitiIncrement4AuthoritySystem,
        proof: AuthenticationProof,
        plan: OperationalAuthorityBootstrapPlan,
    ) -> None:
        self._system = authority_system
        self._proof = proof
        self._plan = plan
        self._expected = {item.ingest_id: item for item in plan.units}
        self._bound: dict[str, CorpusIngestUnit] = {}

    def _require_expected(self, unit: CorpusIngestUnit) -> CorpusIngestUnit:
        expected = self._expected.get(unit.ingest_id)
        if expected is None or replace(
            unit,
            observation_digest=expected.observation_digest,
            observed_at=expected.observed_at,
            proving_run_id=expected.proving_run_id,
        ) != expected:
            raise GraphitiOperationalReadinessError(
                "runtime unit differs from the bootstrapped exact cohort"
            )
        return expected

    def commit_sources(self) -> None:
        """Commit every unique current source-lineage record idempotently."""

        definitions: dict[str, SourceDefinitionRequest] = {}
        versions: dict[str, SourceDefinitionVersionRequest] = {}
        items: dict[str, SourceItemRequest] = {}
        revisions: dict[str, SourceRevisionRequest] = {}
        representations: dict[str, DiscoveryRepresentationRequest] = {}
        with sqlite3.connect(self._system.authority_store_path) as authority_store:
            for unit in self._plan.units:
                definition, version, item, revision, representation = _source_requests(
                    unit,
                    self._plan.rights_for(unit.source_id),
                    prior_revision_id=self._plan.prior_revision_for(unit),
                    authority_store=authority_store,
                )
                for target, identity, request in (
                    (definitions, str(definition.definition_id), definition),
                    (versions, str(version.version_id), version),
                    (items, str(item.item_id), item),
                    (revisions, str(revision.revision_id), revision),
                    (
                        representations,
                        str(representation.representation_id),
                        representation,
                    ),
                ):
                    retained = target.setdefault(identity, request)
                    if retained.digest != request.digest:
                        raise GraphitiOperationalReadinessError(
                            "one source authority identity has conflicting current semantics"
                        )
        sources = self._system.sources
        for request in definitions.values():
            sources.register_definition(request, proof=self._proof)
        for request in versions.values():
            sources.record_definition_version(request, proof=self._proof)
        for request in items.values():
            sources.register_item(request, proof=self._proof)
        pending_revisions = dict(revisions)
        committed_revision_ids: set[str] = set()
        ordered_revisions: list[SourceRevisionRequest] = []
        while pending_revisions:
            ready = sorted(
                (
                    request
                    for request in pending_revisions.values()
                    if request.prior_revision_id is None
                    or str(request.prior_revision_id) in committed_revision_ids
                ),
                key=lambda value: (str(value.item_id), str(value.revision_id)),
            )
            if not ready:
                raise GraphitiOperationalReadinessError(
                    "current source revision predecessor chain is incomplete"
                )
            for request in ready:
                revision_id = str(request.revision_id)
                ordered_revisions.append(request)
                committed_revision_ids.add(revision_id)
                pending_revisions.pop(revision_id)
        for request in ordered_revisions:
            sources.record_revision(request, proof=self._proof)
        for request in representations.values():
            sources.record_representation(request, proof=self._proof)

    def __call__(self, unit: CorpusIngestUnit) -> CorpusIngestUnit:
        expected = self._require_expected(unit)
        cached = self._bound.get(unit.ingest_id)
        if cached is not None:
            return replace(unit, authority=cached.authority)
        text = " ".join(expected.episode_body.split())
        data = text.encode("utf-8")
        rights = self._plan.rights_for(expected.source_id)
        admission = self._system.objects.admit(
            ObjectAdmissionRequest(
                admission_type=OPERATIONAL_ADMISSION_TYPE,
                idempotency_key=(
                    f"issue895-passage:{expected.ingest_id}:"
                    f"{str(rights['packet_digest']).removeprefix('sha256:')}"
                ),
            ),
            data,
            proof=self._proof,
        ).admission
        try:
            access = self._system.objects.latest_access_decision(
                admission.admission_id,
                purpose=GRAPHITI_EVALUATION_HYDRATION_POLICY.purpose,
                proof=self._proof,
            )
            hydrated_data = data
        except KeyError:
            hydrated = self._system.objects.hydrate(
                HydrationRequest(
                    admission_id=admission.admission_id,
                    purpose=GRAPHITI_EVALUATION_HYDRATION_POLICY.purpose,
                    offset=0,
                    length=len(data),
                ),
                proof=self._proof,
            )
            access = hydrated.decision
            hydrated_data = hydrated.data
        if (
            hydrated_data != data
            or admission.blob.blob_digest != digest_bytes(data)
            or access.admission_id != admission.admission_id
            or access.policy_contract_digest
            != GRAPHITI_EVALUATION_HYDRATION_POLICY.contract_digest
            or access.principal_id != OPERATOR_PRINCIPAL_ID
            or access.authority_domain != OPERATOR_AUTHORITY_DOMAIN
            or access.allowed_bytes != len(data)
        ):
            raise GraphitiOperationalReadinessError(
                "current Object CAS access differs from the exact passage"
            )
        original = expected.authority
        assert original is not None
        source_records = tuple(
            dict(record)
            for record in original.records
            if record.get("record_type") in _SOURCE_RECORD_TYPES
        )
        source_record_ids = {str(item.get("record_id")) for item in source_records}
        if source_record_ids != {
            original.definition_id,
            original.definition_version_id,
            original.item_id,
            original.revision_id,
            original.representation_id,
        }:
            raise GraphitiOperationalReadinessError(
                "retained source receipt identities differ from Source Registry input"
            )
        records = (
            *source_records,
            {
                "record_type": "OBJECT_ADMISSION",
                "record_id": str(admission.admission_id),
                "revision_id": original.revision_id,
                "decision": "ADMIT",
                "scope": "EVALUATION_CORPUS_INGEST",
                "blob_digest": admission.blob.blob_digest,
                "definition_digest": admission.definition_digest,
                "rights_decision_digest": admission.rights_decision_digest,
            },
            {
                "record_type": "OBJECT_ACCESS_DECISION",
                "record_id": str(access.access_decision_id),
                "revision_id": original.revision_id,
                "decision": "ALLOW",
                "principal_id": access.principal_id,
                "authority_domain": access.authority_domain,
                "purpose": access.purpose,
                "admission_id": str(access.admission_id),
                "hydration_policy_contract_digest": access.policy_contract_digest,
                "state_cutoff_digest": access.state_cutoff_digest,
                "rights_authority_run_id": rights["rights_authority_run_id"],
                "rights_gate_id": rights["gate_id"],
                "rights_gate_status": rights["status"],
                "rights_gate_reason": "CURRENT_PROVING_RIGHTS_VERIFIED",
                "rights_packet_digest": rights["packet_digest"],
            },
        )
        binding = CorpusAuthorityBinding(
            admission_id=str(admission.admission_id),
            access_decision_id=str(access.access_decision_id),
            definition_id=original.definition_id,
            definition_version_id=original.definition_version_id,
            item_id=original.item_id,
            revision_id=original.revision_id,
            representation_id=original.representation_id,
            records=records,
        )
        bound = replace(expected, authority=binding)
        if bound.ingest_id != expected.ingest_id:
            raise GraphitiOperationalReadinessError(
                "canonical authority binding changed the ingest identity"
            )
        self._bound[unit.ingest_id] = bound
        return replace(unit, authority=binding)


def bootstrap_operational_authority(
    authority_system: GovernedGraphitiIncrement4AuthoritySystem,
    *,
    proof: AuthenticationProof,
    plan: OperationalAuthorityBootstrapPlan,
) -> tuple[OperationalAuthorityBootstrapResult, OperationalCorpusAuthorityBinder]:
    """Apply only the planned current Source/Object authority idempotently."""

    binder = OperationalCorpusAuthorityBinder(
        authority_system=authority_system,
        proof=proof,
        plan=plan,
    )
    try:
        binder.commit_sources()
        bound_units = tuple(binder(unit) for unit in plan.units)
    except IdempotencyIdentityConflict as exc:
        raise GraphitiOperationalReadinessError(
            "retained authority identity differs from the exact bootstrap request"
        ) from exc
    attempts = tuple(_evaluation_attempt_for_unit(unit) for unit in bound_units)
    contract = attempts[0].extraction_contract
    configuration = attempts[0].configuration
    if any(
        attempt.extraction_contract != contract
        or attempt.configuration != configuration
        for attempt in attempts[1:]
    ):
        raise GraphitiOperationalReadinessError(
            "operational cohort does not share exact Graphiti evaluation authority"
        )
    authority_system.extraction.register_contract(contract, proof=proof)
    authority_system.graphiti.register_configuration(configuration, proof=proof)
    return (
        OperationalAuthorityBootstrapResult(
            observed_at=plan.observed_at,
            plan_digest=plan.plan_digest,
            cohort_digest=plan.cohort_digest,
            plan_semantic_digest=_operational_plan_semantic_digest(
                candidate_events=plan.candidate_events,
                rights_by_source=plan.rights_by_source,
                revision_predecessors=plan.revision_predecessors,
            ),
            candidate_event_count=len(plan.candidate_events),
            source_count=len(plan.rights_by_source),
            unit_count=len(plan.units),
            bound_units=bound_units,
        ),
        binder,
    )


def build_and_reconcile_operational_generation(
    authority_system: GovernedGraphitiIncrement4AuthoritySystem,
    *,
    proof: AuthenticationProof,
    plan: OperationalAuthorityBootstrapPlan,
    bootstrap: OperationalAuthorityBootstrapResult,
) -> StructuralReconciliationView:
    """Build or idempotently recheck one complete current Increment 4 view."""

    candidate_ingest_ids = tuple(
        str(ingest_id)
        for event in plan.candidate_events
        for ingest_id in event.get("ingest_ids", ())
    )
    bound_ingest_ids = tuple(unit.ingest_id for unit in bootstrap.bound_units)
    if (
        not candidate_ingest_ids
        or len(candidate_ingest_ids) != len(set(candidate_ingest_ids))
        or candidate_ingest_ids != bound_ingest_ids
    ):
        raise GraphitiOperationalReadinessError(
            "operational build plan differs from its exact bound cohort"
        )
    plan_semantic_digest = _operational_plan_semantic_digest(
        candidate_events=plan.candidate_events,
        rights_by_source=plan.rights_by_source,
        revision_predecessors=plan.revision_predecessors,
    )
    if plan_semantic_digest != bootstrap.plan_semantic_digest:
        raise GraphitiOperationalReadinessError(
            "operational build plan semantics differ from its bootstrap"
        )
    attempts = tuple(
        _evaluation_attempt_for_unit(unit) for unit in bootstrap.bound_units
    )
    if not attempts:
        raise GraphitiOperationalReadinessError(
            "operational generation requires at least one bound unit"
        )
    configuration = attempts[0].configuration.canonical_value()
    if any(
        attempt.configuration.canonical_value() != configuration
        for attempt in attempts[1:]
    ):
        raise GraphitiOperationalReadinessError(
            "operational cohort does not share exact Graphiti evaluation authority"
        )
    generation_id, semantic_digest = _operational_generation_identity(
        graph_destination_id=authority_system.graph_destination_id,
        plan_semantic_digest=plan_semantic_digest,
        bootstrap=bootstrap,
        configuration=configuration,
    )
    result = authority_system.increment4.build_current_and_promote(
        Increment4Neo4jCurrentBuildRequest(
            generation_id=generation_id,
            reason_code="ISSUE_895_CURRENT_ADMITTED",
            idempotency_key=f"issue895-current-admitted:{semantic_digest}",
            purge_retired_generation=False,
        ),
        proof=proof,
    )
    status = authority_system.increment4.generation_status(
        generation_id,
        proof=proof,
    )
    reconciliation = authority_system.increment4.reconcile_active(proof=proof)
    if (
        result.generation.generation_id != generation_id
        or result.generation.state is not ProjectionGenerationState.ACTIVE
        or status.generation.generation_id != generation_id
        or status.generation.state is not ProjectionGenerationState.ACTIVE
        or status.open_gap_count != 0
        or status.dead_letter_count != 0
        or status.source_watermark_ledger_seq != result.source_watermark_ledger_seq
        or reconciliation.family_id != INCREMENT4_ADMITTED_FAMILY_ID
        or reconciliation.generation_id != generation_id
        or reconciliation.checkpoint_ledger_seq != result.checkpoint_ledger_seq
        or reconciliation.projection_state_digest != result.projection_state_digest
    ):
        raise GraphitiOperationalReadinessError(
            "ACTIVE Increment 4 generation differs from exact authority readback"
        )
    return reconciliation


def operational_policy_components() -> dict[str, object]:
    """Return the single retained policy set used by the operational opener."""

    hydration = HydrationPolicyRegistry((GRAPHITI_EVALUATION_HYDRATION_POLICY,))
    rights = RightsPolicyRegistry((_RIGHTS_POLICY,))
    admission = ObjectAdmissionDefinition(
        admission_type=OPERATIONAL_ADMISSION_TYPE,
        definition_version="v1",
        object_class="source.expression",
        allowed_use="proposal.extraction",
        security_scope="evaluation",
        retention_scope="disposable-workspace",
        required_write_scope="authority.objects.admit",
        required_read_scope="authority.objects.read",
        required_manage_scope="authority.objects.manage",
        rights_policy_contract_digest=_RIGHTS_POLICY.contract_digest,
        hydration_policy_contract_digests=frozenset(
            {GRAPHITI_EVALUATION_HYDRATION_POLICY.contract_digest}
        ),
    )
    admissions = ObjectAdmissionRegistry(
        (admission,),
        rights_policies=rights,
        hydration_policies=hydration,
    )
    return {
        "rights_policies": rights,
        "hydration_policies": hydration,
        "admission_registry": admissions,
    }


def _operational_graphiti_write_scopes() -> frozenset[str]:
    return frozenset(
        definition.required_scope
        for definition in graphiti_adapter_command_definitions()
    )


def _operational_entity_write_scopes() -> frozenset[str]:
    supported_commands = frozenset(
        {
            ENTITY_MENTION_ADMIT_COMMAND,
            ENTITY_RESOLUTION_PROPOSE_COMMAND,
            ENTITY_RESOLUTION_DECIDE_COMMAND,
            ENTITY_RESOLUTION_DEPENDENCY_BIND_COMMAND,
        }
    )
    definitions = {
        definition.command_type: definition
        for definition in entity_command_definitions()
    }
    return frozenset(
        definitions[command_type].required_scope
        for command_type in supported_commands
    )


def open_operational_graphiti_authority_system(
    *,
    credential: str,
) -> tuple[GovernedGraphitiIncrement4AuthoritySystem, AuthenticationProof]:
    """Open the one canonical local Increment 4 system with projector identity."""

    if not credential:
        raise ValueError("operational authority credential must be non-empty")
    projector_config = neo4j_projector_config()
    ensure_increment4_state_paths()
    principal = frozenset({OPERATOR_PRINCIPAL_ID})
    source_read = SourceRegistryReadPolicy(
        policy_id="graphiti-operational-source-read-v1",
        purpose="graphiti.operational.bootstrap",
        metadata_required_scope="authority.sources.metadata.read",
        sensitive_required_scope="authority.sources.sensitive.read",
        allowed_principal_ids=principal,
    )
    extraction_read = ExtractionReadPolicy(
        policy_id="graphiti-operational-extraction-read-v1",
        purpose="graphiti.operational.admission",
        metadata_required_scope="authority.extraction.metadata.read",
        proposal_required_scope="authority.extraction.proposal.read",
        raw_output_required_scope="authority.extraction.raw.read",
        allowed_principal_ids=principal,
    )
    entity_read = EntityReadPolicy(
        policy_id="graphiti-operational-entity-read-v1",
        purpose="graphiti.operational.admission",
        proposal_required_scope="authority.entity.proposal.read",
        admitted_required_scope="authority.entity.admitted.read",
        projection_required_scope="authority.entity.projection.read",
        allowed_principal_ids=principal,
    )
    relation_read = EditorialRelationReadPolicy(
        policy_id="graphiti-operational-relation-read-v1",
        purpose="graphiti.operational.admission",
        proposal_required_scope="authority.relation.proposal.read",
        admitted_required_scope="authority.relation.admitted.read",
        projection_required_scope="authority.relation.projection.read",
        allowed_principal_ids=principal,
    )
    graphiti_read = GraphitiAdapterReadPolicy(
        policy_id="graphiti-operational-adapter-read-v1",
        purpose="graphiti.operational.admission",
        attempt_required_scope="authority.graphiti.attempt.read",
        configuration_required_scope="authority.graphiti.configuration.read",
        replay_required_scope="authority.graphiti.replay.read",
        allowed_principal_ids=principal,
    )
    projection_read = ProjectionReadPolicy(
        policy_id="graphiti-operational-projection-read-v1",
        purpose="graphiti.operational.reconciliation",
        required_scope="authority.projection.read",
        allowed_principal_ids=principal,
        allowed_family_ids=frozenset({INCREMENT4_ADMITTED_FAMILY_ID}),
        allowed_family_kinds=frozenset({ProjectionFamilyKind.GRAPH}),
    )
    registry = CommandRegistry(source_registry_command_definitions())
    schemas = PayloadSchemaRegistry(source_registry_payload_contracts())
    policies = operational_policy_components()
    command_scopes = {
        definition.required_scope for definition in registry.definitions()
    }
    graphiti_write_scopes = _operational_graphiti_write_scopes()
    entity_write_scopes = _operational_entity_write_scopes()
    scopes = frozenset(
        {
            *command_scopes,
            *graphiti_write_scopes,
            *entity_write_scopes,
            source_read.metadata_required_scope,
            source_read.sensitive_required_scope,
            extraction_read.metadata_required_scope,
            extraction_read.proposal_required_scope,
            extraction_read.raw_output_required_scope,
            entity_read.proposal_required_scope,
            entity_read.admitted_required_scope,
            entity_read.projection_required_scope,
            relation_read.proposal_required_scope,
            relation_read.admitted_required_scope,
            relation_read.projection_required_scope,
            graphiti_read.attempt_required_scope,
            graphiti_read.configuration_required_scope,
            graphiti_read.replay_required_scope,
            projection_read.required_scope,
            "authority.objects.admit",
            "authority.objects.read",
            "authority.objects.manage",
            "authority.objects.lifecycle.write",
            "authority.observed.write",
            "authority.admitted.write",
            "authority.extraction.execute",
            "authority.extraction.manage",
            "authority.relation.propose",
            "authority.relation.admit",
            "authority.projection.manage",
            "authority.projection.write",
        }
    )
    authenticator = StaticAuthenticator(
        credentials={
            credential: StaticPrincipal(
                principal_id=OPERATOR_PRINCIPAL_ID,
                assurance_class="LOCAL_OPERATOR_BOUND",
                credential_binding_id="graphiti-operational-authority-v1",
            )
        },
        authority_domain=OPERATOR_AUTHORITY_DOMAIN,
        ttl_seconds=300,
    )
    authorizer = StaticAuthorizer(
        policy_version="graphiti-operational-authorisation-v1",
        grants_by_principal={OPERATOR_PRINCIPAL_ID: scopes},
    )
    system = open_governed_graphiti_increment4_authority_system(
        path=CANONICAL_INCREMENT4_AUTHORITY_STORE,
        object_root=CANONICAL_OBJECT_CAS_ROOT,
        workspace_root=CANONICAL_GRAPHITI_WORKSPACE_ROOT,
        registry=registry,
        payload_schemas=schemas,
        admission_registry=policies["admission_registry"],
        rights_policies=policies["rights_policies"],
        hydration_policies=policies["hydration_policies"],
        contracts=increment4_admitted_contract_registry(),
        authenticator=authenticator,
        authorizer=authorizer,
        source_read_policy=source_read,
        extraction_read_policy=extraction_read,
        entity_read_policy=entity_read,
        relation_read_policy=relation_read,
        graphiti_read_policy=graphiti_read,
        projection_read_policy=projection_read,
        object_limits=ObjectLimits(
            global_max_bytes=MAX_EPISODE_BYTES,
            class_max_bytes={"source.expression": MAX_EPISODE_BYTES},
            max_read_bytes=MAX_EPISODE_BYTES,
            min_free_bytes=100 * 1024 * 1024,
            max_staging_bytes=MAX_EPISODE_BYTES,
            max_range_bytes=MAX_EPISODE_BYTES,
        ),
        neo4j_config=projector_config,
    )
    return system, AuthenticationProof(method="STATIC_TOKEN", credential=credential)


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GraphitiOperationalReadinessError(f"{field} is missing")
    return value


@contextmanager
def reopen_operational_campaign_runtime(
    *,
    packet: Mapping[str, object],
    credential: str,
) -> Iterator[GraphitiCampaignRuntime]:
    """Reopen and rebind one sealed campaign before any future F4 dispatch."""

    try:
        campaign = validate_graphiti_campaign_packet(packet)
    except (TypeError, ValueError) as exc:
        raise GraphitiOperationalReadinessError(
            "sealed operational campaign packet is invalid"
        ) from exc
    operational = _require_mapping(
        packet.get("operational_reconciliation"),
        field="operational reconciliation",
    )
    bootstrap_evidence = _require_mapping(
        operational.get("bootstrap"),
        field="operational bootstrap evidence",
    )
    required_steps = {
        "BACKUP",
        "CANONICAL_AUTHORITY_OPEN",
        "CURRENT_COHORT_PLAN",
        "SOURCE_AND_OBJECT_BOOTSTRAP",
        "ACTIVE_GENERATION_RECONCILIATION",
        "STORE_IDENTITY_SNAPSHOT",
        "DORMANT_RUNTIME_COMPOSITION",
        "AUTHENTICATED_GRAPH_READBACK",
        "DORMANT_CAMPAIGN_INPUT",
    }
    completed_steps = operational.get("completed_steps")
    if (
        operational.get("schema_version")
        != "newsroom.graphiti-operational-reconciliation.v1"
        or operational.get("status") != "COMPLETE"
        or not isinstance(completed_steps, list)
        or not all(isinstance(step, str) for step in completed_steps)
        or not required_steps.issubset(set(completed_steps))
        or operational.get("campaign_authorised") is not False
        or any(
            operational.get(field) != 0
            for field in (
                "provider_calls",
                "graphiti_dispatches",
                "service_loads",
                "publication_effects",
                "production_admission_effects",
            )
        )
        or bootstrap_evidence.get("schema_version")
        != "newsroom.graphiti-operational-bootstrap.v1"
        or bootstrap_evidence.get("provider_calls") != 0
    ):
        raise GraphitiOperationalReadinessError(
            "operational reconciliation evidence is incomplete"
        )
    try:
        observed_at = UtcTimestamp.parse(str(bootstrap_evidence["observed_at"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise GraphitiOperationalReadinessError(
            "operational bootstrap observation identity is invalid"
        ) from exc

    snapshot_digests = _require_mapping(
        campaign.get("source_snapshot_digests"),
        field="campaign source snapshot digests",
    )
    expected_snapshots = {
        str(name): str(digest) for name, digest in snapshot_digests.items()
    }
    if (
        set(expected_snapshots) != {"proving", "unpublished", "authority"}
        or operational.get("store_snapshot_digests") != expected_snapshots
    ):
        raise GraphitiOperationalReadinessError(
            "operational store snapshot identities differ"
        )
    stores = _require_mapping(packet.get("store_snapshots"), field="store snapshots")
    canonical_paths = {
        "proving": CANONICAL_PROVING_STORE,
        "unpublished": CANONICAL_UNPUBLISHED_STORE,
        "authority": CANONICAL_INCREMENT4_AUTHORITY_STORE,
    }
    for name, canonical_path in canonical_paths.items():
        descriptor = _require_mapping(
            stores.get(name), field=f"{name} store descriptor"
        )
        if (
            Path(str(descriptor.get("source_path") or "")).expanduser().resolve()
            != canonical_path.expanduser().resolve()
            or descriptor.get("descriptor_digest") != expected_snapshots[name]
        ):
            raise GraphitiOperationalReadinessError(
                f"{name} store is not the canonical sealed input"
            )
    actual_snapshots = graphiti_store_snapshot_digests(
        proving_store=CANONICAL_PROVING_STORE,
        unpublished_store=CANONICAL_UNPUBLISHED_STORE,
        authority_store=CANONICAL_INCREMENT4_AUTHORITY_STORE,
    )
    if actual_snapshots != expected_snapshots:
        raise GraphitiOperationalReadinessError(
            "canonical operational stores drifted after packet sealing"
        )

    system: GovernedGraphitiIncrement4AuthoritySystem | None = None
    try:
        system, proof = open_operational_graphiti_authority_system(
            credential=credential
        )
        with ExitStack() as stack:
            proving = stack.enter_context(
                read_only_snapshot(CANONICAL_PROVING_STORE)
            ).connection
            unpublished = stack.enter_context(
                read_only_snapshot(CANONICAL_UNPUBLISHED_STORE)
            ).connection
            authority = stack.enter_context(
                read_only_snapshot(CANONICAL_INCREMENT4_AUTHORITY_STORE)
            ).connection
            plan = plan_operational_authority_bootstrap(
                proving,
                unpublished,
                authority,
                observed_at=observed_at.value,
            )
        bootstrap, binder = bootstrap_operational_authority(
            system,
            proof=proof,
            plan=plan,
        )
        if bootstrap.canonical_value() != dict(bootstrap_evidence):
            raise GraphitiOperationalReadinessError(
                "reopened operational cohort differs from sealed bootstrap"
            )
        reconciliation = system.increment4.reconcile_active(proof=proof)
        graph_readback = graphiti_graph_destination_readback(
            destination_id=system.graph_destination_id,
            reconciliation=reconciliation,
        )
        graph_identity = graphiti_graph_destination_identity(graph_readback)
        if graph_identity != graphiti_graph_destination_identity(
            _require_mapping(
                operational.get("graph_readback"),
                field="operational graph readback",
            )
        ) or graph_identity != graphiti_graph_destination_identity(
            _require_mapping(
                campaign.get("graph_destination_readback"),
                field="campaign graph readback",
            )
        ):
            raise GraphitiOperationalReadinessError(
                "reopened ACTIVE graph differs from sealed readback"
            )
        if (
            graphiti_store_snapshot_digests(
                proving_store=CANONICAL_PROVING_STORE,
                unpublished_store=CANONICAL_UNPUBLISHED_STORE,
                authority_store=CANONICAL_INCREMENT4_AUTHORITY_STORE,
            )
            != expected_snapshots
        ):
            raise GraphitiOperationalReadinessError(
                "operational authority replay was not idempotent"
            )
        from scripts.hermes_graphiti_worker import (
            compose_governed_graphiti_worker_runtime,
        )

        yield compose_governed_graphiti_worker_runtime(
            authority_system=system,
            authority_store_descriptor_digest=expected_snapshots["authority"],
            proof=proof,
            bind_unit_authority=binder,
            expected_authority_store_path=str(CANONICAL_INCREMENT4_AUTHORITY_STORE),
        )
    finally:
        if system is not None:
            system.close()


def build_operational_campaign_input(
    *,
    head_sha: str,
    tree_sha: str,
    focus_manifest_digest: str,
    graph_destination_id: str,
    candidate_event_count: int,
    recovery_identity: str,
) -> dict[str, object]:
    """Machine-generate the dormant exact-cohort live campaign input."""

    if candidate_event_count <= 0:
        raise ValueError("campaign candidate event count must be positive")
    phases = [
        {
            "phase_id": f"phase-{index}",
            "event_limit": limit,
            "entry_conditions": sorted(CAMPAIGN_RAMP_ENTRY_CONDITIONS),
            "advance_conditions": sorted(CAMPAIGN_RAMP_ADVANCE_CONDITIONS),
        }
        for index, limit in enumerate(
            campaign_event_limits(candidate_event_count), start=1
        )
    ]
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "focus_gate": {
            "head_sha": head_sha,
            "tree_sha": tree_sha,
            "conclusion": "SUCCESS",
            "manifest_digest": focus_manifest_digest,
        },
        "selection_policy": {
            "policy_id": OPERATIONAL_SELECTION_POLICY_ID,
            "policy_version": OPERATIONAL_SELECTION_POLICY_VERSION,
        },
        "provider": {
            "provider_id": "cursor-agent-cli",
            "transport_id": CURSOR_SDK_TRANSPORT,
            "model_id": CURSOR_AGENT_MODEL_ID,
            "embedding_provider_id": GRAPHITI_EMBEDDING_MODEL.split(":", 1)[0],
            "embedding_model_id": OPENROUTER_EMBEDDING_SLUG,
        },
        "graph": {"destination_id": graph_destination_id},
        "caps": {
            "per_event": {
                "proposals": 100,
                "entity_admits": 100,
                "relation_admits": 100,
                "effects": 200,
                "retries": 0,
                "fallbacks": 0,
            },
            "total": {
                "events": candidate_event_count,
                "proposals": candidate_event_count * 100,
                "entity_admits": candidate_event_count * 100,
                "relation_admits": candidate_event_count * 100,
                "effects": candidate_event_count * 200,
                "retries": 0,
                "fallbacks": 0,
                "wall_time_seconds": max(
                    600,
                    candidate_event_count * (GRAPHITI_EXTRACTION_TIMEOUT_MS // 1000),
                ),
                "spend_gbp_microunits": (
                    candidate_event_count * CAMPAIGN_PER_EVENT_SPEND_GBP_MICROUNITS
                ),
            },
            "rate": {"events_per_minute": 1},
        },
        "ramp": {"phases": phases},
        "recovery": {
            "backup_identity": recovery_identity,
            "rollback_procedure_id": (
                "increment4-active-generation-preserved-on-build-failure-v1"
            ),
            "reconciliation_procedure_id": "structural.reconcile_active-v1",
        },
        "immediate_stop_conditions": sorted(CAMPAIGN_REQUIRED_STOP_CONDITIONS),
        "success_objectives": {
            **CAMPAIGN_SUCCESS_OBJECTIVE_BASE,
            "lag": {"max_oldest_eligible_seconds": 300},
        },
        "campaign_authorised": False,
    }


__all__ = [
    "GraphitiOperationalReadinessError",
    "OPERATIONAL_ADMISSION_TYPE",
    "OperationalAuthorityBootstrapPlan",
    "OperationalAuthorityBootstrapResult",
    "OperationalCorpusAuthorityBinder",
    "bootstrap_operational_authority",
    "build_and_reconcile_operational_generation",
    "build_operational_campaign_input",
    "open_operational_graphiti_authority_system",
    "operational_policy_components",
    "plan_operational_authority_bootstrap",
    "reopen_operational_campaign_runtime",
]
