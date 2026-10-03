"""Concrete server-side policies for the owner-approved private Hermes route.

These policies admit internal retention only. Source acquisition permissions,
editorial facts/currentness and Graphiti admission are evaluated separately;
none is manufactured from this object policy. No public destination is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from newsroom.authority import (
    CommandDefinition, CommandRegistry, HydrationPolicyContract,
    HydrationPolicyRegistry, ObjectAdmissionDefinition, ObjectAdmissionId,
    ObjectAdmissionRegistry, PayloadMode, RightsPolicyContract,
    RightsPolicyRegistry, TrustScope,
)
from newsroom.authority.canonical import canonical_json_bytes, digest_canonical
from newsroom.authority.models import ObjectAdmissionDescriptor
from newsroom.authority.policy import (
    PayloadGoldenVector, PayloadSchemaContract, PayloadSchemaRegistry,
    PayloadSchemaValidationError,
)
from newsroom.checks import deterministic_uuid4
from newsroom.increment10 import editorial as e, publication as p, private_serving as s

from .graphiti_operational_readiness import operational_policy_components
from .native_publication import NativePublicationBindings
from . import evidence as evidence_policy
from .govuk_pdf import MAX_RAW_BYTES as MAX_PDF_OBJECT_BYTES
from newsroom.increment5 import native_retrieval as r

VERSION = "hermes-private-native-v1"
MAX_OBJECT_BYTES = 1_048_576
NATIVE_SOURCE_OBSERVATION_ADMISSION_TYPE = "source.native-observation"
NATIVE_SOURCE_OBSERVATION_CLASS = "source.native-observation"
NATIVE_SOURCE_OBSERVATION_USE = "native-source-parsing"
NATIVE_SOURCE_OBSERVATION_PURPOSE = "native-source-intake-read"
NATIVE_PDF_OBSERVATION_ADMISSION_TYPE = "source.native-pdf-observation"
NATIVE_PDF_OBSERVATION_CLASS = "source.native-pdf-observation"
NATIVE_PDF_OBSERVATION_PURPOSE = "native-source-pdf-intake-read"

# class, allowed use, hydration purpose, security, retention, write scope
_OBJECTS = {
    NATIVE_PDF_OBSERVATION_ADMISSION_TYPE: (
        NATIVE_PDF_OBSERVATION_CLASS, NATIVE_SOURCE_OBSERVATION_USE,
        NATIVE_PDF_OBSERVATION_PURPOSE, "authority.protected",
        "source.observation.retained", "authority.source.observe",
    ),
    NATIVE_SOURCE_OBSERVATION_ADMISSION_TYPE: (
        NATIVE_SOURCE_OBSERVATION_CLASS, NATIVE_SOURCE_OBSERVATION_USE,
        NATIVE_SOURCE_OBSERVATION_PURPOSE, "authority.protected",
        "source.observation.retained", "authority.source.observe",
    ),
    r.NATIVE_DOCUMENT_ADMISSION_TYPE: (r.NATIVE_DOCUMENT_CLASS, r.NATIVE_DOCUMENT_USE, r.NATIVE_DOCUMENT_USE, r.NATIVE_SECURITY_SCOPE, r.NATIVE_RETENTION_SCOPE, "authority.retrieval.project"),
    r.NATIVE_CONTEXT_ADMISSION_TYPE: (r.NATIVE_CONTEXT_CLASS, r.NATIVE_CONTEXT_USE, r.NATIVE_CONTEXT_USE, r.NATIVE_SECURITY_SCOPE, r.NATIVE_RETENTION_SCOPE, "authority.retrieval.context"),
    "retrieval.native-vector": (r.NATIVE_VECTOR_CLASS, r.NATIVE_VECTOR_USE, r.NATIVE_VECTOR_USE, r.NATIVE_SECURITY_SCOPE, r.NATIVE_RETENTION_SCOPE, "authority.retrieval.project"),
    "retrieval.native-embedding-receipt": (r.NATIVE_EMBEDDING_RECEIPT_CLASS, r.NATIVE_EMBEDDING_RECEIPT_USE, r.NATIVE_EMBEDDING_RECEIPT_USE, r.NATIVE_SECURITY_SCOPE, r.NATIVE_RETENTION_SCOPE, "authority.retrieval.project"),
    "evidence.source": ("evidence_source", "publication_evidence", "evidence.source", "authority.protected", "evidence.retained", "authority.evidence.admit"),
    "evidence.record": ("evidence_record", "publication_evidence", "evidence.record", "authority.protected", "evidence.retained", "authority.evidence.admit"),
    "evidence.package": ("evidence_package", "evidence_package_retention", "evidence.package.retained", "authority.protected", "evidence.retained", "authority.evidence.admit"),
    e.DECISION_ADMISSION_TYPE: (e.DECISION_CLASS, e.DECISION_USE, e.DECISION_PURPOSE, e.EDITORIAL_SECURITY_SCOPE, e.EDITORIAL_RETENTION_SCOPE, "authority.editorial.decide"),
    e.STORY_ADMISSION_TYPE: (e.STORY_CLASS, e.STORY_USE, e.STORY_PURPOSE, e.EDITORIAL_SECURITY_SCOPE, e.EDITORIAL_RETENTION_SCOPE, "authority.editorial.story.write"),
    p.SURFACE_ADMISSION_TYPE: (p.SURFACE_CLASS, p.SURFACE_USE, p.SURFACE_PURPOSE, p.PUBLICATION_SECURITY_SCOPE, p.PUBLICATION_RETENTION_SCOPE, "authority.publication.surface.write"),
    p.TRANSACTION_ADMISSION_TYPE: (p.TRANSACTION_CLASS, p.TRANSACTION_USE, p.TRANSACTION_PURPOSE, p.PUBLICATION_SECURITY_SCOPE, p.PUBLICATION_RETENTION_SCOPE, "authority.publication.decide"),
    s.ATTEMPT_ADMISSION_TYPE: (s.ATTEMPT_CLASS, s.ATTEMPT_USE, s.ATTEMPT_PURPOSE, s.SERVING_SECURITY_SCOPE, s.SERVING_RETENTION_SCOPE, "authority.private-serving.attempt"),
    s.EVIDENCE_ADMISSION_TYPE: (s.EVIDENCE_CLASS, s.EVIDENCE_USE, s.EVIDENCE_PURPOSE, s.SERVING_SECURITY_SCOPE, s.SERVING_RETENTION_SCOPE, "authority.private-serving.evidence"),
}
_COMMANDS = (
    (r.NATIVE_DOCUMENT_COMMAND, r.NATIVE_DOCUMENT_EVENT, "native_retrieval_document", r.NATIVE_DOCUMENT_ADMISSION_TYPE, "authority.retrieval.project"),
    (r.NATIVE_CONTEXT_COMMAND, r.NATIVE_CONTEXT_EVENT, "native_retrieval_context", r.NATIVE_CONTEXT_ADMISSION_TYPE, "authority.retrieval.context"),
    (e.DECISION_COMMAND, e.DECISION_EVENT, "editorial_package_decision", e.DECISION_ADMISSION_TYPE, "authority.editorial.decide"),
    (e.STORY_COMMAND, e.STORY_EVENT, "story", e.STORY_ADMISSION_TYPE, "authority.editorial.story.admit"),
    (p.PUBLICATION_COMMAND, p.PUBLICATION_EVENT, "publication", p.TRANSACTION_ADMISSION_TYPE, "authority.publication.decide"),
    (s.ATTEMPT_COMMAND, s.ATTEMPT_EVENT, "publication", s.ATTEMPT_ADMISSION_TYPE, "authority.private-serving.attempt"),
    (s.EVIDENCE_COMMAND, s.EVIDENCE_EVENT, "private_serving_attempt", s.EVIDENCE_ADMISSION_TYPE, "authority.private-serving.evidence"),
)


def _reference(value: object) -> bytes:
    if type(value) is not ObjectAdmissionDescriptor:
        raise PayloadSchemaValidationError("exact governed object descriptor required")
    return canonical_json_bytes({
        "admission_id": str(value.admission_id), "blob_digest": value.blob_digest,
        "object_class": value.object_class, "allowed_use": value.allowed_use,
        "security_scope": value.security_scope, "retention_scope": value.retention_scope,
    })


@dataclass(frozen=True, slots=True)
class NativePolicies:
    rights_policies: RightsPolicyRegistry
    hydration_policies: HydrationPolicyRegistry
    admission_registry: ObjectAdmissionRegistry
    registry: CommandRegistry
    payload_schemas: PayloadSchemaRegistry
    publication: NativePublicationBindings
    evidence_hydration: tuple[str, str, str]
    evidence_package_definition: str
    required_scopes: frozenset[str]
    retrieval_hydration: tuple[str, str, str]
    retrieval_document_definition: str
    retrieval_command_definition: str
    retrieval_context_hydration: str
    retrieval_context_definition: str
    retrieval_context_command_definition: str


def native_policy_components(
    *, principal_id: str, authority_domain: str, target_path: Path, target_id: str,
) -> NativePolicies:
    """Bind existing native boundaries without test helpers or invented digests."""
    # Golden values test the canonicaliser, not source/runtime/owner evidence.
    golden = ObjectAdmissionDescriptor(
        deterministic_uuid4(ObjectAdmissionId, namespace=VERSION, semantic_value="object-reference-golden"),
        digest_canonical({"schema_golden": VERSION}), e.DECISION_CLASS, e.DECISION_USE,
        e.EDITORIAL_SECURITY_SCOPE, e.EDITORIAL_RETENTION_SCOPE, True,
    )
    contract = PayloadSchemaContract(
        schema_version="hermes_private_object_reference_v1",
        payload_mode=PayloadMode.OBJECT_ADMISSION, contract_version=VERSION,
        canonicalizer_implementation_version=VERSION, canonicalizer=_reference,
        golden_vectors=(PayloadGoldenVector("object-reference", "hermes-object-reference-v1", golden, _reference(golden)),),
    )
    base = operational_policy_components()
    internal_rights = RightsPolicyContract(
        policy_key="hermes-internal-retention", contract_version=VERSION,
        implementation_version=VERSION, preflight_allowed=True,
        reason_code="CONTROLLER_INTERNAL_RETENTION_ONLY",
    )
    rights = RightsPolicyRegistry((*base["rights_policies"].contracts(), internal_rights))
    hydration_by_type = {
        admission_type: HydrationPolicyContract(
            policy_id=f"{admission_type}-hermes-read", contract_version=VERSION,
            implementation_version=VERSION, purpose=values[2],
            required_scope="authority.objects.read",
            allowed_principal_ids=frozenset({principal_id}),
            allowed_authority_domains=frozenset({authority_domain}),
            allowed_object_classes=frozenset({values[0]}), allowed_uses=frozenset({values[1]}),
            allowed_security_scopes=frozenset({values[3]}),
            allowed_retention_scopes=frozenset({values[4]}),
            max_bytes=MAX_PDF_OBJECT_BYTES if values[0] == NATIVE_PDF_OBSERVATION_CLASS else MAX_OBJECT_BYTES,
        ) for admission_type, values in _OBJECTS.items()
    }
    hydration = HydrationPolicyRegistry((*base["hydration_policies"].contracts(), *hydration_by_type.values()))
    definitions = {
        admission_type: ObjectAdmissionDefinition(
            admission_type=admission_type, definition_version=VERSION,
            object_class=values[0], allowed_use=values[1], security_scope=values[3],
            retention_scope=values[4], required_write_scope=values[5],
            required_read_scope="authority.objects.read", required_manage_scope="authority.objects.manage",
            rights_policy_contract_digest=internal_rights.contract_digest,
            hydration_policy_contract_digests=frozenset({hydration_by_type[admission_type].contract_digest}),
        ) for admission_type, values in _OBJECTS.items()
    }
    admissions = ObjectAdmissionRegistry(
        (*base["admission_registry"].definitions(), *definitions.values()),
        rights_policies=rights, hydration_policies=hydration,
    )
    commands = {
        command: CommandDefinition(
            command_type=command, definition_version=VERSION, aggregate_type=aggregate,
            event_type=event, event_schema_version=1, payload_mode=PayloadMode.OBJECT_ADMISSION,
            payload_schema_version=contract.schema_version,
            payload_schema_contract_version=contract.contract_version,
            payload_schema_contract_digest=contract.contract_digest,
            payload_canonicalizer_version=contract.canonicalizer_implementation_version,
            trust_scope=TrustScope.ADMITTED, security_scope=_OBJECTS[admission_type][3],
            retention_scope=_OBJECTS[admission_type][4], required_scope=scope,
            required_object_class=_OBJECTS[admission_type][0],
            required_allowed_use=_OBJECTS[admission_type][1],
        ) for command, event, aggregate, admission_type, scope in _COMMANDS
    }
    editorial_policy_value = {
        "version": VERSION,
        "decision_schema": e.DECISION_SCHEMA, "story_schema": e.STORY_VERSION_SCHEMA,
        "evidence_policies": {
            name: getattr(evidence_policy, name) for name in (
                "EVID_012_POLICY_VERSION", "GOVERNED_CLAIM_POLICY_VERSION",
                "EVIDENCE_GATE_POLICY_VERSION", "GOVERNED_INPUT_SCHEMA_VERSION",
                "EVIDENCE_APPROVAL_POLICY_VERSION", "ORIGINALITY_POLICY_VERSION",
                "NAMED_ENTITY_POLICY_VERSION",
            )
        },
        "decision_definition": commands[e.DECISION_COMMAND].digest,
        "story_definition": commands[e.STORY_COMMAND].digest,
        "publication_rights_from_proposal": False,
    }
    editorial_policy = digest_canonical(editorial_policy_value)
    # The policy is tied to the reviewed source terms. It is
    # publication metadata, never an editorial claim or permission exception.
    from .govuk_rights import ATTRIBUTION, LICENCE_URL, POLICY_DIGEST as GOVUK_RIGHTS_DIGEST
    from .native_source_rights import SOURCE_LICENCE_POLICY, POLICY_DIGEST as PORTFOLIO_RIGHTS_DIGEST
    source_licence_policy = (("www.gov.uk", ATTRIBUTION, LICENCE_URL), *SOURCE_LICENCE_POLICY)
    target_policy = digest_canonical({
        "version": VERSION, "target_id": target_id,
        "surfaces": ["ARTICLE", "FEED_CARD"], "ack_required": True,
        "public_exposure": False,
        "source_licence_policy": source_licence_policy,
        "source_licence_policy_digests": (GOVUK_RIGHTS_DIGEST, PORTFOLIO_RIGHTS_DIGEST),
    })
    target_context = digest_canonical({
        "target_policy": target_policy, "path": str(target_path.resolve()),
        "adapter": "private-serving-projection-v1",
    })
    retained_policy_pairs = ()
    if evidence_policy.NAMED_ENTITY_POLICY_VERSION == "newsroom.named-entity.v16":
        prior = {**editorial_policy_value, "evidence_policies": {
            **editorial_policy_value["evidence_policies"],
            "NAMED_ENTITY_POLICY_VERSION": "newsroom.named-entity.v15",
        }}
        prior_editorial = digest_canonical(prior)
        retained_policy_pairs = ((prior_editorial, digest_canonical({
            "version": VERSION, "owner_scope": "issue-151-private-autonomous",
            "editorial_policy": prior_editorial, "target_policy": target_policy,
        })),)
    def hd(kind): return hydration_by_type[kind].contract_digest
    def ad(kind): return definitions[kind].digest
    publication = NativePublicationBindings(
        target_path=target_path.resolve(), reader_principal_id=principal_id,
        authority_domain=authority_domain, editorial_controller_principal_id=principal_id,
        story_principal_id=principal_id, publication_controller_principal_id=principal_id,
        serving_adapter_principal_id=principal_id, editorial_policy_bundle_digest=editorial_policy,
        editorial_decision_hydration_policy_digest=hd(e.DECISION_ADMISSION_TYPE),
        editorial_story_hydration_policy_digest=hd(e.STORY_ADMISSION_TYPE),
        editorial_decision_admission_definition_digest=ad(e.DECISION_ADMISSION_TYPE),
        editorial_decision_command_definition_digest=commands[e.DECISION_COMMAND].digest,
        editorial_story_command_definition_digest=commands[e.STORY_COMMAND].digest,
        editorial_story_admission_definition_digest=ad(e.STORY_ADMISSION_TYPE),
        publication_authorisation_policy_digest=digest_canonical({"version": VERSION, "owner_scope": "issue-151-private-autonomous", "editorial_policy": editorial_policy, "target_policy": target_policy}),
        target_id=target_id, target_policy_digest=target_policy,
        publication_surface_hydration_policy_digest=hd(p.SURFACE_ADMISSION_TYPE),
        publication_transaction_hydration_policy_digest=hd(p.TRANSACTION_ADMISSION_TYPE),
        publication_surface_admission_definition_digest=ad(p.SURFACE_ADMISSION_TYPE),
        publication_transaction_admission_definition_digest=ad(p.TRANSACTION_ADMISSION_TYPE),
        publication_command_definition_digest=commands[p.PUBLICATION_COMMAND].digest,
        target_context_digest=target_context,
        serving_attempt_hydration_policy_digest=hd(s.ATTEMPT_ADMISSION_TYPE),
        serving_evidence_hydration_policy_digest=hd(s.EVIDENCE_ADMISSION_TYPE),
        serving_attempt_admission_definition_digest=ad(s.ATTEMPT_ADMISSION_TYPE),
        serving_evidence_admission_definition_digest=ad(s.EVIDENCE_ADMISSION_TYPE),
        serving_attempt_command_definition_digest=commands[s.ATTEMPT_COMMAND].digest,
        serving_evidence_command_definition_digest=commands[s.EVIDENCE_COMMAND].digest,
        source_licence_policy=source_licence_policy,
        retained_policy_pairs=retained_policy_pairs,
    )
    scopes = frozenset({
        "authority.objects.read", "authority.events.read",
        *[value.required_write_scope for value in definitions.values()],
        *[value.required_scope for value in commands.values()],
    })
    return NativePolicies(rights, hydration, admissions, CommandRegistry(commands.values()),
        PayloadSchemaRegistry((contract,)), publication,
        tuple(hd(key) for key in ("evidence.source", "evidence.record", "evidence.package")),
        ad("evidence.package"), scopes,
        tuple(hd(key) for key in (r.NATIVE_DOCUMENT_ADMISSION_TYPE, "retrieval.native-vector", "retrieval.native-embedding-receipt")),
        ad(r.NATIVE_DOCUMENT_ADMISSION_TYPE), commands[r.NATIVE_DOCUMENT_COMMAND].digest,
        hd(r.NATIVE_CONTEXT_ADMISSION_TYPE), ad(r.NATIVE_CONTEXT_ADMISSION_TYPE),
        commands[r.NATIVE_CONTEXT_COMMAND].digest)
