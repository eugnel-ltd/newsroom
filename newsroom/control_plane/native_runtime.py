"""Concrete private Hermes composition; one native authority writer, no dispatch.

Opening this composition does not fetch a source or call a model. The daemon
supplies its authenticated credential, qualified retrieval/collision authorities
and actual storage/projector identities. No fixture certificates are minted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from newsroom.authority import (
    AuthenticationProof, CommandRegistry, ObjectLimits, PayloadSchemaRegistry,
    StaticAuthenticator, StaticAuthorizer, StaticPrincipal, UtcTimestamp,
)
from newsroom.authority.hermes_native_system import (
    HermesNativeAuthoritySystem, NativeDependencyFactory,
    open_hermes_native_authority_system,
)
from newsroom.authority.canonical import digest_canonical
from newsroom.authority.persistence import EventReadPolicy, MetadataClass
from newsroom.authority.types import TrustScope
from newsroom.checks.policy import discovery_check_command_definitions
from newsroom.checks.read_policy import DiscoveryCheckReadPolicy
from newsroom.discovery.policy import discovery_signal_lead_command_definitions
from newsroom.discovery.types import DiscoveryReadPolicy
from newsroom.entities.types import EntityReadPolicy
from newsroom.extraction.types import ExtractionReadPolicy
from newsroom.graphiti_adapter import GraphitiAdapterReadPolicy
from newsroom.increment4 import increment4_admitted_contract_registry
from newsroom.increment4.contracts import INCREMENT4_ADMITTED_FAMILY_ID
from newsroom.increment6.candidates import candidate_command_definition
from newsroom.increment6.collision import CurrentCollisionEffectEnforcer
from newsroom.increment6.lineage import lineage_command_definition
from newsroom.increment6.relationships import relationship_command_definition
from newsroom.increment6.work_items import RetrievalContextAuthority
from newsroom.increment10.evidence import GovernedEvidencePackages
from newsroom.increment10.ingress import EvidenceIntakeIngress, open_evidence_intake_ingress
from newsroom.projection.models import ProjectionFamilyKind, ProjectionReadPolicy
from newsroom.projection.neo4j.models import Neo4jProjectorConfig
from newsroom.relations.editorial_models import EditorialRelationReadPolicy
from newsroom.sources.policy import (
    source_registry_command_definitions, source_registry_payload_contracts,
)
from newsroom.sources.types import SourceRegistryReadPolicy

from .native_policies import (MAX_OBJECT_BYTES, MAX_PDF_OBJECT_BYTES,
    NATIVE_PDF_OBSERVATION_CLASS, VERSION, NativePolicies, native_policy_components)
from .native_publication import NativePublicationController
from .graphiti_operational_readiness import (
    _operational_entity_write_scopes, _operational_graphiti_write_scopes,
)


@dataclass(slots=True)
class NativeRuntime:
    authority: HermesNativeAuthoritySystem
    proof: AuthenticationProof
    policies: NativePolicies
    ingress: EvidenceIntakeIngress
    evidence: GovernedEvidencePackages
    publication: NativePublicationController
    actor_identity_digest: str

    def close(self) -> None:
        try:
            self.publication.close()
        finally:
            try:
                self.ingress.close()
            finally:
                self.authority.close()

    def __enter__(self) -> NativeRuntime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def open_native_runtime(
    *, authority_path: Path, object_root: Path, workspace_root: Path,
    intake_path: Path, target_path: Path, target_id: str,
    credential: str, principal_id: str, authority_domain: str,
    neo4j_config: Neo4jProjectorConfig,
    retrieval_authority: RetrievalContextAuthority | None = None,
    collision_enforcer: CurrentCollisionEffectEnforcer | None = None,
    native_dependency_factory: NativeDependencyFactory | None = None,
    story_writer=None,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
) -> NativeRuntime:
    """Bind the existing native boundaries to actual private runtime identities."""
    if not credential:
        raise ValueError("Hermes runtime credential must be non-empty")
    paths = (authority_path, intake_path, target_path)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("authority, intake and serving stores must be distinct")
    policies = native_policy_components(
        principal_id=principal_id, authority_domain=authority_domain,
        target_path=target_path, target_id=target_id,
    )
    principals = frozenset({principal_id})
    source_read = SourceRegistryReadPolicy(
        policy_id="hermes-native-source-read-v1", purpose="hermes.native.source",
        metadata_required_scope="authority.sources.metadata.read",
        sensitive_required_scope="authority.sources.sensitive.read",
        allowed_principal_ids=principals,
    )
    check_read = DiscoveryCheckReadPolicy(
        policy_id="hermes-native-check-read-v1", purpose="hermes.native.check",
        metadata_required_scope="authority.checks.metadata.read",
        sensitive_required_scope="authority.checks.sensitive.read",
        allowed_principal_ids=principals,
    )
    discovery_read = DiscoveryReadPolicy(
        policy_id="hermes-native-discovery-read-v1", purpose="hermes.native.discovery",
        metadata_required_scope="authority.discovery.read",
        sensitive_required_scope="authority.discovery.read_sensitive",
        allowed_principal_ids=principals,
    )
    extraction_read = ExtractionReadPolicy(
        policy_id="hermes-native-extraction-read-v1", purpose="hermes.native.extraction",
        metadata_required_scope="authority.extraction.metadata.read",
        proposal_required_scope="authority.extraction.proposal.read",
        raw_output_required_scope="authority.extraction.raw.read",
        allowed_principal_ids=principals,
    )
    entity_read = EntityReadPolicy(
        policy_id="hermes-native-entity-read-v1", purpose="hermes.native.context",
        proposal_required_scope="authority.entity.proposal.read",
        admitted_required_scope="authority.entity.admitted.read",
        projection_required_scope="authority.entity.projection.read",
        allowed_principal_ids=principals,
    )
    relation_read = EditorialRelationReadPolicy(
        policy_id="hermes-native-relation-read-v1", purpose="hermes.native.context",
        proposal_required_scope="authority.relation.proposal.read",
        admitted_required_scope="authority.relation.admitted.read",
        projection_required_scope="authority.relation.projection.read",
        allowed_principal_ids=principals,
    )
    graphiti_read = GraphitiAdapterReadPolicy(
        policy_id="hermes-native-graphiti-read-v1", purpose="hermes.native.graphiti",
        attempt_required_scope="authority.graphiti.attempt.read",
        configuration_required_scope="authority.graphiti.configuration.read",
        replay_required_scope="authority.graphiti.replay.read",
        allowed_principal_ids=principals,
    )
    projection_read = ProjectionReadPolicy(
        policy_id="hermes-native-projection-read-v1", purpose="hermes.native.context",
        required_scope="authority.projection.read", allowed_principal_ids=principals,
        allowed_family_ids=frozenset({INCREMENT4_ADMITTED_FAMILY_ID}),
        allowed_family_kinds=frozenset({ProjectionFamilyKind.GRAPH}),
    )
    native_definitions = (
        *source_registry_command_definitions(),
        *discovery_check_command_definitions(),
        *discovery_signal_lead_command_definitions(),
        relationship_command_definition(), lineage_command_definition(),
        candidate_command_definition(), *policies.registry.definitions(),
    )
    # These are the daemon's internal capabilities, not provider or source I/O
    # permission. Each actual dispatch still crosses current rights, accounting
    # and stop controls. No per-revision human grant is introduced.
    reads = (source_read, check_read, discovery_read, extraction_read,
             entity_read, relation_read, graphiti_read, projection_read)
    scopes = policies.required_scopes | {
        "authority.objects.admit", "authority.objects.manage",
        "authority.objects.lifecycle.write", "authority.observed.write",
        "authority.admitted.write", "authority.extraction.execute",
        "authority.extraction.manage", "authority.relation.propose",
        "authority.relation.admit", "authority.projection.manage",
        "authority.projection.write",
    } | _operational_graphiti_write_scopes() | _operational_entity_write_scopes() | frozenset(
        definition.required_scope for definition in native_definitions
    ) | frozenset(
        getattr(policy, name)
        for policy in reads for name in policy.__dataclass_fields__
        if name.endswith("required_scope") or name == "required_scope"
    )
    event_read = EventReadPolicy(
        policy_id="hermes-native-event-read-v1", purpose="hermes.native.reconcile",
        required_scope="authority.events.read", allowed_principal_ids=principals,
        allowed_security_scopes=frozenset(
            definition.security_scope for definition in native_definitions
        ),
        allowed_trust_scopes=frozenset({TrustScope.ADMITTED}),
        metadata_classes=frozenset(MetadataClass),
    )
    authenticator = StaticAuthenticator(
        credentials={credential: StaticPrincipal(
            principal_id=principal_id, assurance_class="LOCAL_OPERATOR_BOUND",
            credential_binding_id="hermes-native-runtime-v1",
        )}, authority_domain=authority_domain,
    )
    authority = open_hermes_native_authority_system(
        path=authority_path, object_root=object_root, workspace_root=workspace_root,
        registry=CommandRegistry((
            *source_registry_command_definitions(), *policies.registry.definitions(),
        )),
        payload_schemas=PayloadSchemaRegistry((
            *source_registry_payload_contracts(), *policies.payload_schemas.contracts(),
        )),
        admission_registry=policies.admission_registry,
        rights_policies=policies.rights_policies,
        hydration_policies=policies.hydration_policies,
        contracts=increment4_admitted_contract_registry(), authenticator=authenticator,
        authorizer=StaticAuthorizer(
            policy_version=VERSION, grants_by_principal={principal_id: scopes},
        ),
        event_read_policy=event_read, source_read_policy=source_read,
        check_read_policy=check_read, discovery_read_policy=discovery_read,
        extraction_read_policy=extraction_read, entity_read_policy=entity_read,
        relation_read_policy=relation_read, graphiti_read_policy=graphiti_read,
        projection_read_policy=projection_read,
        object_limits=ObjectLimits(
            global_max_bytes=MAX_PDF_OBJECT_BYTES,
            class_max_bytes={definition.object_class:
                             MAX_PDF_OBJECT_BYTES if definition.object_class == NATIVE_PDF_OBSERVATION_CLASS else MAX_OBJECT_BYTES
                             for definition in policies.admission_registry.definitions()},
            max_read_bytes=MAX_PDF_OBJECT_BYTES, max_staging_bytes=MAX_PDF_OBJECT_BYTES,
            max_range_bytes=MAX_PDF_OBJECT_BYTES, min_free_bytes=100 * 1024 * 1024,
        ),
        neo4j_config=neo4j_config, retrieval_authority=retrieval_authority,
        collision_enforcer=collision_enforcer,
        native_dependency_factory=native_dependency_factory, clock=clock,
    )
    ingress = None
    try:
        ingress = open_evidence_intake_ingress(intake_path)
        evidence = GovernedEvidencePackages(
            objects=authority.objects, ingress=ingress,
            reader_principal_id=principal_id, reader_authority_domain=authority_domain,
            source_hydration_policy_digest=policies.evidence_hydration[0],
            record_hydration_policy_digest=policies.evidence_hydration[1],
            package_hydration_policy_digest=policies.evidence_hydration[2],
            package_admission_definition_digest=policies.evidence_package_definition,
        )
        publication = NativePublicationController(
            objects=authority.objects, commands=authority.commands,
            events=authority.events, candidate_port=authority.candidate_read_port,
            evidence_packages=evidence, bindings=policies.publication, clock=clock,
            story_writer=story_writer,
        )
        proof = AuthenticationProof(
            method="STATIC_TOKEN", credential=credential,
        )
        authentication = authenticator.authenticate(proof, now=clock())
        actor_identity_digest = digest_canonical({
            "principal_id": authentication.principal_id,
            "credential_binding_digest": authentication.credential_binding_digest,
        })
        return NativeRuntime(authority, proof, policies, ingress, evidence,
                             publication, actor_identity_digest)
    except BaseException:
        if ingress is not None:
            ingress.close()
        authority.close()
        raise
