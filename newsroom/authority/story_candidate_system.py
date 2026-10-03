"""Server-owned checked v24 Story Candidate authority composition root."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Callable
from pathlib import Path

from newsroom.authority._capability import _CapabilityIssuer
from newsroom.authority._discovery_store import (
    _create_discovery_governing_producer_read_port,
)
from newsroom.authority._event_hypothesis_lineage_system import (
    _create_event_hypothesis_lineage_read_port,
)
from newsroom.authority._event_store import _EventAuthorityStore
from newsroom.authority.auth import AuthenticationProof, StaticAuthenticator
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.authority.models import InlinePayload, SemanticCommand
from newsroom.authority.persistence import EventReadPolicy, MetadataClass
from newsroom.authority.policy import CommandRegistry, PayloadSchemaRegistry
from newsroom.authority.service import CommandService
from newsroom.authority.types import AggregateId, UtcTimestamp
from newsroom.discovery import NewsLeadId
from newsroom.increment6.candidates import (
    CANDIDATE_COMMAND_SCHEMA,
    CANDIDATE_COMMAND_TYPE,
    CandidateAdmission,
    CandidateAdmissionReason,
    CandidateAdmissionRequest,
    CandidateContractError,
    CandidateGoverningState,
    CandidateGoverningStateStatus,
    StoryCandidate,
    StoryCandidateAuthority,
    StoryCandidateReadPort,
    StoryCandidateVersion,
    _compose_story_candidate_authority,
    _compose_story_candidate_read_port,
    build_candidate_distinct_scope_proof,
    build_candidate_governing_manifest,
    candidate_command_definition,
    evaluate_candidate_admission,
    merge_candidate_authority_registries,
    validate_candidate_first_version,
    validate_candidate_version_successor,
)
from newsroom.increment6.collision import (
    CandidateUseOperation,
    CurrentCollisionEffectEnforcer,
    CurrentCollisionEligibilityDecision,
    CurrentCollisionEligibilityRequest,
)
from newsroom.increment6.dispositions import (
    CurrentCandidateCitationReadPort,
    ProposalDispositionStore,
)
from newsroom.increment6.lineage import merge_lineage_authority_registries
from newsroom.increment6.hypotheses import EventHypothesisVersion
from newsroom.increment6.proposals import HypothesisRelationship
from newsroom.increment6.relationships import (
    CanonicalOutcome,
    merge_relationship_authority_registries,
)
from newsroom.increment6.work_items import RetrievalContextAuthority

from ._event_hypothesis_system import _HypothesisStore

_TOKEN = object()
_READ_AUTHORITY_TOKEN = object()


def _uuid() -> str:
    return str(uuid.uuid4())


def _actor(authentication: object) -> str:
    return digest_bytes(
        canonical_json_bytes(
            {
                "principal_id": authentication.principal_id,
                "credential_binding_digest": authentication.credential_binding_digest,
            }
        )
    )  # type: ignore[attr-defined]


# fmt: off
def _collision_position(value):
    return value.subject_id, value.subject_version_id, value.subject_version_digest, value.collision_namespace, value.collision_key_digest


def _manifest_position(value):
    return value.hypothesis_id, value.hypothesis_version_id, value.hypothesis_version_digest, value.collision_namespace, value.collision_key_digest


def _collision_context(value):
    return value.generation_id, value.query_valid_time, value.serving_time, value.authority_watermark
# fmt: on


def _payload(
    request,
    hypothesis_version_id,
    relationship_digest,
    disposition_ids,
    collision_request,
    comparator_collision_request=None,
    admission=None,
    collision_decision=None,
    comparator_collision_decision=None,
    effect_identity=None,
):
    return {
        "schema_version": CANDIDATE_COMMAND_SCHEMA,
        "request": request.canonical_value(),
        "hypothesis_version_id": hypothesis_version_id,
        "relationship_assessment_digest": relationship_digest,
        "disposition_ids": list(disposition_ids),
        "collision_request": collision_request.canonical_value(),
        "comparator_collision_request": None
        if comparator_collision_request is None
        else comparator_collision_request.canonical_value(),
        "admission": None
        if admission is None
        else json.loads(admission.canonical_bytes),
        "collision_decision": None
        if collision_decision is None
        else json.loads(collision_decision.canonical_bytes),
        "comparator_collision_decision": None
        if comparator_collision_decision is None
        else json.loads(comparator_collision_decision.canonical_bytes),
        "effect_identity": effect_identity,
    }


# fmt: off
class _CandidateStore(_EventAuthorityStore):
    def __init__(
        self,
        token: object,
        path: Path,
        *,
        retrieval_authority: RetrievalContextAuthority,
        authenticator: StaticAuthenticator,
        authorizer: object,
        command_registry: CommandRegistry,
        payload_schemas: PayloadSchemaRegistry,
        collision_enforcer: CurrentCollisionEffectEnforcer,
        clock: Callable[[], UtcTimestamp],
        busy_timeout_ms: int,
    ):
        if (
            token is not _TOKEN
            or type(retrieval_authority) is not RetrievalContextAuthority
            or type(authenticator) is not StaticAuthenticator
            or type(collision_enforcer) is not CurrentCollisionEffectEnforcer
        ):
            raise CandidateContractError("Candidate composition collaborators differ")
        relationship_commands, relationship_schemas = (
            merge_relationship_authority_registries(command_registry, payload_schemas)
        )
        lineage_commands, lineage_schemas = merge_lineage_authority_registries(
            relationship_commands, relationship_schemas
        )
        commands, schemas = merge_candidate_authority_registries(
            lineage_commands, lineage_schemas
        )
        issuer = _CapabilityIssuer(command_registry=commands, payload_schemas=schemas)
        super().__init__(
            path,
            issuer=issuer,
            command_registry=commands,
            payload_schemas=schemas,
            command_service_version="increment6-candidate-v1",
            busy_timeout_ms=busy_timeout_ms,
            clock=clock,
        )
        try:
            self._retrieval = retrieval_authority
            self._authenticator = authenticator
            self._collision = collision_enforcer
            self._lineage = _create_event_hypothesis_lineage_read_port(
                self._connection,
                retrieval_authority=retrieval_authority,
                authenticator=authenticator,
                command_registry=commands,
                payload_schemas=schemas,
                clock=clock,
            )
            self._dispositions = ProposalDispositionStore(
                self._connection, retrieval_authority, authenticator
            )
            self._service = CommandService(
                registry=commands,
                payload_schemas=schemas,
                authenticator=authenticator,
                authorizer=authorizer,
                committed_lookup=self,
                clock=clock,
                _issuer=issuer,
            )
            with self._lock, self._transaction():
                self._verify_global_event_coverage()
                self._verify()
        except BaseException:
            try:
                self.close()
            except BaseException:  # noqa: BLE001, S110 - preserve opening failure
                pass
            raise

    def _command(
        self, payload: dict, request: CandidateAdmissionRequest, aggregate_id: str
    ):
        return SemanticCommand(
            command_type=CANDIDATE_COMMAND_TYPE,
            aggregate_id=AggregateId.parse(aggregate_id),
            expected_aggregate_version=request.expected_current_ordinal,
            payload=InlinePayload(payload),
            idempotency_key=request.idempotency_key,
        )

    def _receipt(self, row):
        candidate_bytes = row["candidate_bytes"]
        if candidate_bytes is None:
            candidate_bytes = self._connection.execute(
                "SELECT candidate_bytes FROM story_candidate_heads WHERE candidate_id=?",
                (row["candidate_id"],),
            ).fetchone()
            if candidate_bytes is None:
                raise CandidateContractError("Candidate identity is absent")
            candidate_bytes = candidate_bytes[0]
        comparator = row["comparator_collision_decision_bytes"]
        return (
            CandidateAdmission.from_canonical_bytes(bytes(row["admission_bytes"])),
            StoryCandidate.from_canonical_bytes(bytes(candidate_bytes)),
            StoryCandidateVersion.from_canonical_bytes(bytes(row["version_bytes"])),
            CurrentCollisionEligibilityDecision.from_canonical_bytes(
                bytes(row["collision_decision_bytes"])
            ),
            None
            if comparator is None
            else CurrentCollisionEligibilityDecision.from_canonical_bytes(
                bytes(comparator)
            ),
            tuple(json.loads(bytes(row["disposition_ids_bytes"]))),
        )

    def _row(self, digest: str):
        return self._connection.execute(
            "SELECT r.*,c.command_type,c.aggregate_type command_aggregate_type,"
            "c.aggregate_id command_aggregate_id,c.expected_aggregate_version,"
            "c.idempotency_namespace,c.idempotency_key command_idempotency_key,"
            "c.stable_semantic_request_digest,c.authentication_context_id command_auth,"
            "c.authorization_request_digest command_request,c.authorization_decision_id command_decision,"
            "c.result_bytes,c.result_digest,c.committed_at,v.aggregate_version retained_aggregate_version,"
            "v.trust_scope version_trust_scope,v.recorded_at version_recorded_at,"
            "g.current_version,p.payload_bytes,au.event_type audit_event_type,"
            "au.detail_digest,au.recorded_at audit_recorded_at,"
            "au.authentication_context_id audit_auth,au.authorization_request_digest audit_request,"
            "au.authorization_decision_id audit_decision FROM story_candidate_admission_receipts_v2 r "
            "JOIN ledger_events e ON e.event_id=r.authority_event_id "
            "JOIN authority_commands c ON c.command_id=e.command_id "
            "JOIN authority_aggregate_versions v ON v.command_id=c.command_id "
            "JOIN authority_aggregates g ON g.aggregate_type=v.aggregate_type AND g.aggregate_id=v.aggregate_id "
            "JOIN authority_payloads p ON p.payload_id=e.payload_id "
            "JOIN authority_audit_events au ON au.command_id=c.command_id "
            "WHERE r.admission_digest=?",
            (digest,),
        ).fetchone()

    @staticmethod
    def _columns(row, names):
        return tuple(row[name] for name in names.split())

    def _verify_row(self, digest: str):
        row = self._row(digest)
        if row is None:
            raise CandidateContractError("Candidate receipt coverage differs")
        self._validate_retained_event(str(row["authority_event_id"]))
        admission, candidate, version, collision, comparator, disposition_ids = (
            self._receipt(row)
        )
        request, manifest = admission.request, admission.governing_manifest
        definition = candidate_command_definition()
        policy = EventReadPolicy("candidate-retained-v1", "candidate.retained", definition.required_scope,
                                 frozenset({str(row["actor_identity_digest"])}), frozenset({definition.security_scope}),
                                 frozenset({definition.trust_scope}), frozenset({MetadataClass.PROVENANCE}), max_results=1)
        provenance = self.event_provenance(event_id=str(row["authority_event_id"]), policy=policy)
        event, authentication = provenance.event, provenance.authentication
        authorisation, decision = provenance.authorization_request, provenance.authorization_decision
        contract = provenance.payload_schema_contract
        payload_meta = {
            "kind": event.payload_mode, "schema_version": contract.schema_version,
            "schema_contract_version": contract.contract_version, "schema_contract_digest": contract.contract_digest,
            "canonicalizer_version": contract.canonicalizer_implementation_version,
            "digest": event.payload_digest, "inline_digest": event.payload_digest,
            "object_admission_id": None, "blob_digest": None, "object_class": None, "allowed_use": None,
        }
        aggregate_version = request.expected_current_ordinal + 1
        stable = digest_canonical({"command_type": definition.command_type, "command_definition_version": definition.definition_version,
            "command_definition_digest": definition.digest, "aggregate_type": definition.aggregate_type,
            "aggregate_id": row["authority_aggregate_id"], "expected_aggregate_version": request.expected_current_ordinal,
            "payload": payload_meta})
        result = self._decode_result(bytes(row["result_bytes"]), str(row["result_digest"]), replayed=False)
        detail = {
            "operation": "COMMAND_COMMIT", "command_type": definition.command_type, "aggregate_id": row["authority_aggregate_id"],
            "expected_aggregate_version": request.expected_current_ordinal, "definition_digest": definition.digest,
            "definition_version": definition.definition_version, "payload": payload_meta,
            "authentication_context_digest": authentication.canonical_digest,
            "authorization_request_record_digest": authorisation.canonical_record_digest,
            "authorization_request_digest": authorisation.request_digest, "authorization_decision_digest": decision.canonical_digest,
            "idempotency_namespace": row["idempotency_namespace"], "idempotency_key": row["command_idempotency_key"],
            "stable_semantic_request_digest": stable, "correlation_id": event.correlation_id,
            "causation_kind": event.causation_kind, "causation_identifier": event.causation_identifier,
            "causation_external_system": event.causation_external_system, "replay_of_command_id": None,
        }
        effect = {"candidate_id": row["candidate_id"], "committed_admission_decision_id": row["committed_admission_decision_id"],
                  "version_id": row["version_id"], "version_ordinal": row["version_ordinal"]}
        expected_payload = _payload(
            request, row["hypothesis_version_id"], row["relationship_assessment_digest"], disposition_ids,
            collision.request, None if comparator is None else comparator.request,
            admission, collision, comparator, effect,
        )
        expected_reason = (CandidateAdmissionReason.RELATED_DISTINCT_PRE_EFFECT if admission.distinct_scope_proof is not None
                           else CandidateAdmissionReason.NEW_CANDIDATE_PRE_EFFECT if version.ordinal == 1
                           else CandidateAdmissionReason.SUCCESSOR_VERSION_PRE_EFFECT)
        optional = self._columns(
            row,
            "comparator_collision_request_bytes comparator_collision_request_digest "
            "comparator_collision_decision_bytes comparator_collision_decision_digest",
        )
        route = (definition.command_type, definition.definition_version, definition.digest, definition.aggregate_type,
                 definition.event_type, definition.event_schema_version, definition.payload_schema_version,
                 definition.payload_schema_contract_version, definition.payload_schema_contract_digest,
                 definition.payload_canonicalizer_version)
        actual_route = (provenance.command_definition.command_type, provenance.command_definition.definition_version,
                        provenance.command_definition.definition_digest, event.aggregate_type, event.event_type,
                        event.event_schema_version, event.payload_schema_version, event.payload_schema_contract_version,
                        event.payload_schema_contract_digest, event.payload_canonicalizer_version)
        actor = digest_bytes(canonical_json_bytes({"principal_id": authentication.principal_id,
                                                   "credential_binding_digest": authentication.credential_binding_digest}))
        failures = (
            digest_bytes(admission.canonical_bytes) != row["admission_digest"],
            self._columns(row, "request_id request_digest actor_identity_digest idempotency_key")
            != (request.request_id, request.canonical_digest, request.actor_identity_digest, request.idempotency_key),
            request.collision_request_digest != row["collision_request_digest"],
            route != actual_route,
            event.producer_version != self._command_service_version,
            (event.retention_scope, event.trust_scope, row["version_trust_scope"])
            != (definition.retention_scope, definition.trust_scope.value, definition.trust_scope.value),
            (event.aggregate_id, event.aggregate_version)
            != (row["authority_aggregate_id"], aggregate_version),
            self._columns(row, "command_type command_aggregate_type command_aggregate_id expected_aggregate_version")
            != (definition.command_type, definition.aggregate_type, row["authority_aggregate_id"], request.expected_current_ordinal),
            row["retained_aggregate_version"] != aggregate_version,
            self._columns(row, "command_idempotency_key idempotency_key stable_semantic_request_digest")
            != (request.idempotency_key, request.idempotency_key, stable),
            row["idempotency_namespace"]
            != digest_canonical({"authority_domain": authentication.authority_domain, "principal_id": authentication.principal_id, "command_type": definition.command_type}),
            (result.command_id, result.aggregate_type, result.aggregate_id, result.aggregate_version, result.ledger_seq, result.event_id)
            != (event.command_id, definition.aggregate_type, row["authority_aggregate_id"], aggregate_version, event.ledger_seq, event.event_id),
            len({event.recorded_at, row["committed_at"], row["version_recorded_at"], row["audit_recorded_at"], row["recorded_at"]}) != 1,
            bytes(row["payload_bytes"]) != canonical_json_bytes(expected_payload),
            actor != row["actor_identity_digest"],
            len({self._columns(row, "command_auth command_request command_decision"), (event.authentication_context_id, event.authorization_request_digest, event.authorization_decision_id), self._columns(row, "audit_auth audit_request audit_decision")}) != 1,
            row["audit_event_type"] != definition.event_type,
            row["detail_digest"] != digest_canonical(detail),
            not decision.allowed or definition.required_scope not in decision.effective_scopes,
            (authorisation.operation_type, authorisation.required_scope)
            != (f"command:{definition.command_type}", definition.required_scope),
            collision.request.request_digest != row["collision_request_digest"],
            collision.request.canonical_value() != json.loads(bytes(row["collision_request_bytes"])),
            collision.canonical_bytes != bytes(row["collision_decision_bytes"]),
            collision.decision_digest != row["collision_decision_digest"],
            manifest.collision_decision_digest != collision.decision_digest,
            _collision_position(collision.request.binding) != _manifest_position(manifest),
            manifest.version_material_digest != row["manifest_material_digest"],
            version.committed_admission_decision_id != row["committed_admission_decision_id"],
            version.ordinal != aggregate_version,
            (comparator is None) != (admission.distinct_scope_proof is None),
            not (all(item is None for item in optional) or all(item is not None for item in optional)),
            admission.reason is not expected_reason,
            (version.candidate_id, version.version_id, version.ordinal, version.canonical_digest)
            != self._columns(row, "candidate_id version_id version_ordinal version_digest"),
            version.canonical_digest != digest_bytes(bytes(row["version_bytes"])),
            (version.previous_version_id, version.previous_version_digest)
            != self._columns(row, "previous_version_id previous_version_digest"),
            (admission.current_candidate_id, admission.current_candidate_version_id, admission.current_candidate_version_digest)
            != ((None, None, None) if version.ordinal == 1 else (version.candidate_id, version.previous_version_id, version.previous_version_digest)),
            row["authority_aggregate_id"] != row["candidate_id"],
            (candidate.candidate_id, candidate.semantic_scope_digest)
            != self._columns(row, "candidate_id semantic_scope_digest"),
            (version.ordinal == 1) != (row["candidate_bytes"] is not None),
            version.ordinal == 1 and candidate.canonical_bytes != bytes(row["candidate_bytes"]),
            version.governing_manifest != manifest,
            (manifest.semantic_scope_digest, manifest.hypothesis_version_id, manifest.relationship_assessment_digest)
            != self._columns(row, "semantic_scope_digest hypothesis_version_id relationship_assessment_digest"),
            version.ordinal == 1 and (candidate.committed_admission_decision_id, candidate.authority_event_id)
            != (row["committed_admission_decision_id"], row["authority_event_id"]),
        )
        if any(failures):
            raise CandidateContractError("Candidate retained authority graph differs")
        if comparator is not None and (
            comparator.request.request_digest != optional[1]
            or canonical_json_bytes(comparator.request.canonical_value()) != bytes(optional[0])
            or comparator.canonical_bytes != bytes(optional[2])
            or comparator.decision_digest != optional[3]
        ):
            raise CandidateContractError("Candidate comparator collision differs")
        UtcTimestamp.parse(str(row["recorded_at"]))
        return admission, candidate, version, collision, comparator, disposition_ids, row

    def _all_receipts(self, *, candidate_ids=None):
        where = "" if candidate_ids is None else " WHERE candidate_id IN (SELECT value FROM json_each(?))"
        parameters = () if candidate_ids is None else (json.dumps(sorted(candidate_ids)),)
        return {
            str(row["admission_digest"]): self._verify_row(str(row["admission_digest"]))
            for row in self._connection.execute(
                "SELECT admission_digest FROM story_candidate_admission_receipts_v2"
                + where + " ORDER BY recorded_at,admission_digest", parameters,
            )
        }

    def _verify_local(self, *, candidate_ids=None):
        # Opening and explicit inventory retain the whole domain. Runtime reads
        # retain every version and exact event closure of selected and related histories.
        if candidate_ids is not None and not candidate_ids:
            return {}
        verified = self._all_receipts(candidate_ids=candidate_ids)
        if candidate_ids is not None:
            selected = set(candidate_ids)
            while related := {
                item[0].distinct_scope_proof.comparator_candidate_id
                for item in verified.values() if item[0].distinct_scope_proof is not None
            } - selected:
                selected.update(related)
                verified.update(self._all_receipts(candidate_ids=related))
            candidate_ids = frozenset(selected)
        where = "" if candidate_ids is None else " WHERE candidate_id IN (SELECT value FROM json_each(?))"
        parameters = () if candidate_ids is None else (json.dumps(sorted(candidate_ids)),)
        event_count = self._connection.execute(
            "SELECT COUNT(*) FROM ledger_events WHERE aggregate_type=?"
            + ("" if candidate_ids is None else " AND aggregate_id IN (SELECT value FROM json_each(?))"),
            (candidate_command_definition().aggregate_type, *parameters),
        ).fetchone()[0]
        if event_count != len(verified):
            raise CandidateContractError("Candidate event coverage differs")
        by_version = {item[2].version_id: item for item in verified.values()}
        for admission, _, _, collision, comparator, _, _ in verified.values():
            proof = admission.distinct_scope_proof
            if proof is not None:
                retained = by_version.get(proof.comparator_version_id)
                if retained is None or comparator is None or build_candidate_distinct_scope_proof(
                    proposed_manifest=admission.governing_manifest,
                    proposed_collision=collision,
                    comparator_collision=comparator,
                    comparator_version=retained[2],
                ) != proof:
                    raise CandidateContractError("Candidate distinct proof differs")
        heads = {row["candidate_id"]: row for row in self._connection.execute(
            "SELECT h.*,g.current_version generic_current FROM story_candidate_heads h "
            "LEFT JOIN authority_aggregates g ON g.aggregate_type='story_candidate_admission' AND g.aggregate_id=h.candidate_id"
            + where.replace("candidate_id", "h.candidate_id"), parameters,
        )}
        bindings = {row["candidate_id"]: row for row in self._connection.execute(
            "SELECT * FROM story_candidate_collision_bindings" + where, parameters,
        )}
        groups = {}
        for item in verified.values():
            groups.setdefault(item[2].candidate_id, []).append(item)
        if set(groups) != set(heads) or set(groups) != set(bindings):
            raise CandidateContractError("Candidate coverage differs")
        for candidate_id, values in groups.items():
            values.sort(key=lambda item: item[2].ordinal)
            head, binding = heads[candidate_id], bindings[candidate_id]
            first, last = values[0], values[-1]
            if [item[2].ordinal for item in values] != list(range(1, len(values) + 1)):
                raise CandidateContractError("Candidate Version chain differs")
            decision = CurrentCollisionEligibilityDecision.from_canonical_bytes(
                bytes(binding["initial_decision_bytes"])
            )
            if (
                (binding["collision_namespace"], binding["collision_key_digest"], binding["semantic_scope_digest"], binding["admission_digest"], binding["initial_request_digest"], binding["initial_decision_digest"], binding["initial_decision_bytes"])
                != (first[3].request.binding.collision_namespace, first[3].request.binding.collision_key_digest, first[1].semantic_scope_digest, first[0].canonical_digest, first[3].request.request_digest, first[3].decision_digest, first[3].canonical_bytes)
                or decision != first[3]
                or (head["candidate_bytes"], head["semantic_scope_digest"], head["current_version_id"], head["current_version_ordinal"], head["current_version_digest"], head["current_admission_digest"], head["collision_namespace"], head["collision_key_digest"], head["updated_at"], head["generic_current"])
                != (first[1].canonical_bytes, first[1].semantic_scope_digest, last[2].version_id, last[2].ordinal, last[2].canonical_digest, last[0].canonical_digest, binding["collision_namespace"], binding["collision_key_digest"], last[6]["recorded_at"], last[2].ordinal)
            ):
                raise CandidateContractError("Candidate head or collision binding differs")
            validate_candidate_first_version(first[1], first[2])
            for previous, successor in zip(values, values[1:], strict=False):  # noqa: RUF007
                validate_candidate_version_successor(previous[2], successor[2])
        for table in (
            "story_candidate_admission_receipts_v2",
            "story_candidate_collision_bindings", "story_candidate_heads",
        ):
            if candidate_ids is None:
                violation = self._connection.execute(f'PRAGMA foreign_key_check("{table}")').fetchone()
            else:
                violation = self._connection.execute(
                    f'SELECT 1 FROM pragma_foreign_key_check(?) bad JOIN "{table}" child ON child.rowid=bad.rowid '
                    'WHERE child.candidate_id IN (SELECT value FROM json_each(?)) LIMIT 1',
                    (table, *parameters),
                ).fetchone()
            if violation is not None:
                raise CandidateContractError("Candidate foreign keys differ")
        return verified

    def _verify_global_event_coverage(self):
        # Opening still detects orphan events even with a forged aggregate tag.
        counts = self._connection.execute(
            "SELECT (SELECT count(*) FROM ledger_events WHERE event_type=?),"
            "(SELECT count(*) FROM story_candidate_admission_receipts_v2)",
            (candidate_command_definition().event_type,),
        ).fetchone()
        if counts[0] != counts[1]:
            raise CandidateContractError("Candidate event coverage differs")

    def _verify(self, *, relationship_receipts=None):
        verified = self._verify_local()
        self._verify_upstream(verified, relationship_receipts=relationship_receipts)
        return verified

    def _verify_upstream(self, verified, *, exact: bool = False, relationship_receipts=None):
        if exact:
            relationships = {}
            for admission, *_ in verified.values():
                manifest = admission.governing_manifest
                receipt = self._lineage._require_exact_candidate_manifest_in_transaction(
                    manifest.hypothesis_version_id,
                    manifest.relationship_assessment_digest,
                    manifest.lineage_history_digests,
                    manifest.lineage_generation,
                )
                relationships[manifest.relationship_assessment_digest] = (
                    receipt.assessment
                )
        elif relationship_receipts is not None:
            try:
                relationships = {
                    admission.governing_manifest.relationship_assessment_digest:
                    relationship_receipts[
                        admission.governing_manifest.relationship_assessment_digest
                    ].assessment
                    for admission, *_ in verified.values()
                }
            except KeyError as exc:
                raise CandidateContractError("Candidate relationship is absent") from exc
        else:
            digests = {
                admission.governing_manifest.relationship_assessment_digest
                for admission, *_ in verified.values()
            }
            ordered_digests = tuple(sorted(digests))
            # Opening verifies the complete retained relationship authority.
            relationships = {
                digest: receipt.assessment
                for digest, receipt in zip(
                    ordered_digests,
                    self._lineage.require_retained_relationships_in_transaction(
                        ordered_digests
                    ),
                    strict=True,
                )
            }
        for admission, *_ in verified.values():
            manifest = admission.governing_manifest
            assessment = relationships[manifest.relationship_assessment_digest]
            comparator = assessment.comparator
            if (
                (assessment.subject.hypothesis_id, assessment.subject.version_id,
                    assessment.subject.version_digest, assessment.status, assessment.decision,
                    None if comparator is None else comparator.hypothesis_id,
                    None if comparator is None else comparator.version_id,
                    None if comparator is None else comparator.version_digest)
                != (manifest.hypothesis_id, manifest.hypothesis_version_id,
                    manifest.hypothesis_version_digest, manifest.relationship_status,
                    manifest.relationship_outcome, manifest.relationship_comparator_hypothesis_id,
                    manifest.relationship_comparator_version_id,
                    manifest.relationship_comparator_version_digest)):
                raise CandidateContractError("Candidate relationship differs")

    def _existing(self, request, actor):
        rows = self._connection.execute(
            "SELECT * FROM story_candidate_admission_receipts_v2 WHERE "
            "request_id=? OR request_digest=? OR "
            "(actor_identity_digest=? AND idempotency_key=?)",
            (
                request.request_id,
                request.canonical_digest,
                actor,
                request.idempotency_key,
            ),
        ).fetchall()
        if not rows:
            return None
        row = rows[0]
        if (
            len(rows) != 1
            or
            row["request_digest"] != request.canonical_digest
            or row["request_id"] != request.request_id
        ):
            raise CandidateContractError("Candidate replay diverges")
        return self._verify_local()[str(row["admission_digest"])]

    def _current_scope(self, scope):
        row = self._connection.execute(
            "SELECT r.version_bytes FROM story_candidate_heads h JOIN story_candidate_admission_receipts_v2 r ON r.admission_digest=h.current_admission_digest WHERE h.semantic_scope_digest=?",
            (scope,),
        ).fetchone()
        return (
            None
            if row is None
            else StoryCandidateVersion.from_canonical_bytes(bytes(row[0]))
        )

    def _producers_from_refs(
        self, hypothesis_version_id, relationship_assessment_digest, proof
    ):
        snapshot, relationship, dispositions = (
            self._lineage._require_candidate_inputs_in_transaction(
                hypothesis_version_id,
                relationship_assessment_digest,
                proof=proof,
            )
        )
        disposition_ids = tuple(
            sorted(
                item.disposition_id
                for item in snapshot.subject.source_bindings
            )
        )
        # The lineage batch has already replayed the full disposition authority
        # in this same transaction, including the empty-digest relationship case.
        if tuple(item.disposition_id for item in dispositions) != disposition_ids:
            raise CandidateContractError("Candidate current dispositions differ")
        lead_ids = tuple(
            sorted(
                {NewsLeadId.parse(item.decision_lead_id) for item in dispositions},
                key=str,
            )
        )
        discovery = _create_discovery_governing_producer_read_port(
            self._connection,
            object_admission_payload_validator=self._validate_object_admission_payload_record,
            validate_retained_event=self._validate_retained_event,
        ).require_current_governing_producers(lead_ids)
        return (
            snapshot,
            relationship,
            disposition_ids,
            dispositions,
            tuple(zip(*discovery, strict=True)),
        )

    def _producers(self, manifest, proof):
        return self._producers_from_refs(
            manifest.hypothesis_version_id,
            manifest.relationship_assessment_digest,
            proof,
        )

    def build_manifest(
        self,
        hypothesis_version_id: str,
        relationship_assessment_digest: str,
        collision: CurrentCollisionEligibilityDecision,
        *,
        proof: AuthenticationProof,
    ):
        if type(collision) is not CurrentCollisionEligibilityDecision:
            raise CandidateContractError("Candidate collision decision differs")
        if not self._connection.in_transaction:
            with self._lock, self._transaction():
                return self.build_manifest(
                    hypothesis_version_id,
                    relationship_assessment_digest,
                    collision,
                    proof=proof,
                )
        producers = self._producers_from_refs(
            hypothesis_version_id, relationship_assessment_digest, proof
        )
        return self._manifest(producers, collision)

    @staticmethod
    def _manifest(producers, collision):
        snapshot, relationship, _, dispositions, discovery = producers
        return build_candidate_governing_manifest(
            hypothesis_version=snapshot.subject,
            lineage_receipts=snapshot.receipts,
            lineage_initial_heads=snapshot.initial_heads,
            lineage_versions=snapshot.versions,
            lineage_relationship_proofs=snapshot.relationship_proofs,
            dispositions=dispositions,
            leads=discovery[0],
            signals=discovery[1],
            gates=discovery[2],
            relationship=relationship,
            collision=collision,
        )

    def admit(
        self,
        admission_bytes: bytes,
        *,
        collision_request: CurrentCollisionEligibilityRequest,
        proof: AuthenticationProof,
        comparator_collision_request: CurrentCollisionEligibilityRequest | None = None,
    ):
        supplied = CandidateAdmission.from_canonical_bytes(admission_bytes)
        request = supplied.request
        if type(collision_request) is not CurrentCollisionEligibilityRequest:
            raise CandidateContractError("Candidate collision request differs")
        authentication = self._authenticator.authenticate(proof, now=self._clock())
        actor = _actor(authentication)
        if actor != request.actor_identity_digest:
            raise CandidateContractError("Candidate actor binding differs")
        if not self._connection.in_transaction:
            with self._lock, self._transaction():
                return self.admit(
                    admission_bytes,
                    collision_request=collision_request,
                    proof=proof,
                    comparator_collision_request=comparator_collision_request,
                )
        if type(comparator_collision_request) not in (
            type(None),
            CurrentCollisionEligibilityRequest,
        ):
            raise CandidateContractError("Candidate comparator request differs")
        if (comparator_collision_request is None) != (
            supplied.distinct_scope_proof is None
        ):
            raise CandidateContractError("Candidate distinct comparator differs")
        replay = self._existing(request, actor)
        if replay is not None:
            historical_admission, _, historical_version, *_, row = replay
            historical_comparator_digest = row[
                "comparator_collision_request_digest"
            ]
            if (
                historical_admission.canonical_bytes != admission_bytes
                or collision_request.request_digest != row["collision_request_digest"]
                or (comparator_collision_request is None)
                != (historical_comparator_digest is None)
                or (
                    comparator_collision_request is not None
                    and comparator_collision_request.request_digest
                    != historical_comparator_digest
                )
            ):
                raise CandidateContractError("Candidate exact replay differs")
            payload = json.loads(bytes(row["payload_bytes"]))
            grant = self._service._authorize_for_commit(
                self._command(payload, request, str(row["authority_aggregate_id"])),
                proof=proof,
            )
            committed = self._commit_grant_in_transaction(
                self._connection, grant, recorded_at=str(row["recorded_at"])
            )
            if (
                not committed.replayed
                or committed.event_id != row["authority_event_id"]
            ):
                raise CandidateContractError("Candidate generic command replay differs")
            return historical_version
        self._verify_local()
        manifest = supplied.governing_manifest
        binding = collision_request.binding
        if (
            collision_request.request_digest != request.collision_request_digest
            or _collision_position(binding) != _manifest_position(manifest)
        ):
            raise CandidateContractError("Candidate collision input differs")
        slot = self._connection.execute(
            "SELECT candidate_id,semantic_scope_digest FROM story_candidate_collision_bindings "
            "WHERE collision_namespace=? AND collision_key_digest=?",
            (binding.collision_namespace, binding.collision_key_digest),
        ).fetchone()
        current = self._current_scope(request.semantic_scope_digest)
        if (slot is None) != (current is None) or (
            slot is not None
            and tuple(slot) != (current.candidate_id, request.semantic_scope_digest)
        ):
            raise CandidateContractError("Candidate local collision binding differs")
        expected = (request.expected_current_version_id, request.expected_current_version_digest, request.expected_current_ordinal)
        actual = (None, None, 0) if current is None else (current.version_id, current.canonical_digest, current.ordinal)
        expected_operation = CandidateUseOperation.ADMIT_NEW_CANDIDATE if current is None else CandidateUseOperation.USE_CURRENT_CANDIDATE
        if (expected != actual or binding.operation is not expected_operation
            or binding.expected_candidate_id != (None if current is None else current.candidate_id)):
            raise CandidateContractError("Candidate local CAS differs")
        comparator_version = None
        if comparator_collision_request is not None:
            comparator_binding = comparator_collision_request.binding
            comparator_id = comparator_binding.expected_candidate_id
            comparator_version = None if comparator_id is None else self._current_candidate(comparator_id)
            proof_value = supplied.distinct_scope_proof
            comparator_manifest = None if comparator_version is None else comparator_version.governing_manifest
            if (
                comparator_version is None
                or proof_value is None
                or binding.operation is not CandidateUseOperation.ADMIT_NEW_CANDIDATE
                or comparator_binding.operation is not CandidateUseOperation.USE_CURRENT_CANDIDATE
                or comparator_id != proof_value.comparator_candidate_id
                or comparator_version.version_id != proof_value.comparator_version_id
                or comparator_version.canonical_digest != proof_value.comparator_version_digest
                or comparator_manifest.semantic_scope_digest != proof_value.comparator_semantic_scope_digest
                or _collision_position(comparator_binding) != _manifest_position(comparator_manifest)
                or _collision_context(binding) != _collision_context(comparator_binding)
                or binding.subject_id == comparator_binding.subject_id
                or (
                    binding.collision_namespace,
                    binding.collision_key_digest,
                )
                == (
                    comparator_binding.collision_namespace,
                    comparator_binding.collision_key_digest,
                )
            ):
                raise CandidateContractError("Candidate distinct comparator is stale")
        producers = self._producers(manifest, proof)
        disposition_ids = producers[2]

        def effect(collision: CurrentCollisionEligibilityDecision, comparator_collision=None):
            rebuilt = self._manifest(producers, collision)
            state = CandidateGoverningState(CandidateGoverningStateStatus.COMPLETE, rebuilt.governing_state_binding)
            admission = evaluate_candidate_admission(
                request=request, manifest=rebuilt, collision=collision,
                current_version=current, governing_state=state,
                comparator_collision=comparator_collision,
                comparator_version=comparator_version,
            )
            if admission.canonical_bytes != admission_bytes:
                raise CandidateContractError("Candidate Admission differs from current producer replay")
            if admission.reason is CandidateAdmissionReason.EXACT_MANIFEST_REPLAY:
                if current is None:
                    raise CandidateContractError("Candidate equivalent head is absent")
                return current
            if admission.reason not in {
                CandidateAdmissionReason.NEW_CANDIDATE_PRE_EFFECT,
                CandidateAdmissionReason.RELATED_DISTINCT_PRE_EFFECT,
                CandidateAdmissionReason.SUCCESSOR_VERSION_PRE_EFFECT,
            }:
                raise CandidateContractError("Candidate admission is not effect eligible")
            candidate_id = _uuid() if current is None else current.candidate_id
            ordinal = 1 if current is None else current.ordinal + 1
            decision_id, version_id = _uuid(), _uuid()
            effect_identity = {"candidate_id": candidate_id, "committed_admission_decision_id": decision_id,
                               "version_id": version_id, "version_ordinal": ordinal}
            payload = _payload(
                request, manifest.hypothesis_version_id, manifest.relationship_assessment_digest,
                disposition_ids, collision_request, comparator_collision_request,
                admission, collision, comparator_collision, effect_identity,
            )
            grant = self._service._authorize_for_commit(self._command(payload, request, candidate_id), proof=proof)
            recorded = self._clock().to_text()
            committed = self._commit_grant_in_transaction(self._connection, grant, recorded_at=recorded)
            if committed.replayed:
                raise CandidateContractError("fresh Candidate command replayed")
            candidate = (StoryCandidate(candidate_id, decision_id, committed.event_id, rebuilt.semantic_scope_digest)
                         if current is None else self.load_candidate_in_transaction(current.candidate_id))
            version = StoryCandidateVersion(
                version_id, candidate_id, ordinal, None if current is None else current.version_id,
                None if current is None else current.canonical_digest, decision_id, rebuilt,
            )
            (validate_candidate_first_version(candidate, version) if current is None
             else validate_candidate_version_successor(current, version))
            self._persist(
                admission, candidate, version, committed, actor, recorded,
                disposition_ids, collision, payload, comparator_collision,
            )
            self._verify_local()
            return version

        if comparator_collision_request is None:
            return self._collision.enforce(request=collision_request, effect=effect)
        return self._collision.enforce(request=collision_request, effect=lambda collision:
            self._collision.enforce(request=comparator_collision_request,
                                    effect=lambda comparator: effect(collision, comparator)))

    def _persist(self, a, c, v, committed, actor, recorded, disposition_ids, collision, payload, comparator):
        b, ad = collision.request.binding, a.canonical_digest
        optional = (None, None, None, None) if comparator is None else (
            canonical_json_bytes(comparator.request.canonical_value()),
            comparator.request.request_digest,
            comparator.canonical_bytes,
            comparator.decision_digest,
        )
        values = (
            ad, a.request.request_id, a.request.canonical_digest, actor,
            a.request.idempotency_key, str(committed.aggregate_id), committed.event_id,
            v.committed_admission_decision_id, a.canonical_bytes, c.candidate_id,
            c.canonical_bytes if v.ordinal == 1 else None, v.version_id, v.ordinal,
            v.canonical_bytes, v.canonical_digest, v.previous_version_id,
            v.previous_version_digest, a.governing_manifest.version_material_digest,
            a.governing_manifest.semantic_scope_digest,
            a.governing_manifest.hypothesis_version_id,
            a.governing_manifest.relationship_assessment_digest,
            canonical_json_bytes(list(disposition_ids)),
            canonical_json_bytes(payload["collision_request"]),
            collision.request.request_digest, collision.canonical_bytes,
            collision.decision_digest, *optional, recorded,
        )
        self._connection.execute(
            "INSERT INTO story_candidate_admission_receipts_v2 VALUES("
            + ",".join("?" for _ in values) + ")", values
        )
        if v.ordinal == 1:
            self._connection.execute(
                "INSERT INTO story_candidate_collision_bindings VALUES(?,?,?,?,?,?,?,?)",
                (b.collision_namespace, b.collision_key_digest, c.candidate_id,
                 c.semantic_scope_digest, ad, collision.request.request_digest,
                 collision.decision_digest, collision.canonical_bytes),
            )
            self._connection.execute(
                "INSERT INTO story_candidate_heads VALUES(?,?,?,?,?,?,?,?,?,?)",
                (c.candidate_id, c.canonical_bytes, c.semantic_scope_digest,
                 v.version_id, v.ordinal, v.canonical_digest, ad,
                 b.collision_namespace, b.collision_key_digest, recorded),
            )
            return
        binding = self._connection.execute(
            "SELECT candidate_id,semantic_scope_digest FROM story_candidate_collision_bindings "
            "WHERE collision_namespace=? AND collision_key_digest=?",
            (b.collision_namespace, b.collision_key_digest),
        ).fetchone()
        if binding is None or tuple(binding) != (c.candidate_id, c.semantic_scope_digest):
            raise CandidateContractError("Candidate collision binding differs")
        changed = self._connection.execute(
            "UPDATE story_candidate_heads SET current_version_id=?,current_version_ordinal=?,"
            "current_version_digest=?,current_admission_digest=?,updated_at=? WHERE candidate_id=? "
            "AND current_version_id=? AND current_version_digest=?",
            (v.version_id, v.ordinal, v.canonical_digest, ad, recorded, c.candidate_id,
             v.previous_version_id, v.previous_version_digest),
        ).rowcount
        if changed != 1:
            raise CandidateContractError("Candidate head CAS differs")

    def load_candidate_in_transaction(self, candidate_id: str):
        row = self._connection.execute(
            "SELECT candidate_bytes FROM story_candidate_heads WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise CandidateContractError("unknown Candidate")
        return StoryCandidate.from_canonical_bytes(bytes(row[0]))

    def _current_candidate(self, candidate_id: str):
        row = self._connection.execute(
            "SELECT r.version_bytes FROM story_candidate_heads h JOIN story_candidate_admission_receipts_v2 r ON r.admission_digest=h.current_admission_digest WHERE h.candidate_id=?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise CandidateContractError("unknown Candidate")
        return StoryCandidateVersion.from_canonical_bytes(bytes(row[0]))

    def load_candidate(self, candidate_id: str):
        with self._lock, self._transaction():
            self._verify()
            return self.load_candidate_in_transaction(candidate_id)

    def exact_current_producers(self, candidate_id: str, *, proof: AuthenticationProof):
        if not self._connection.in_transaction:
            with self._lock, self._transaction():
                return self.exact_current_producers(candidate_id, proof=proof)
        admission, _, version, collision, _, disposition_ids, _ = (
            self._exact_current_receipt(candidate_id)
        )

        producers = self._producers(admission.governing_manifest, proof)
        if (
            producers[2] != disposition_ids
            or self._manifest(producers, collision) != version.governing_manifest
        ):
            raise CandidateContractError("Candidate current governing material differs")
        return version, producers[0].subject

    def exact_associated_current_producers(
        self, candidate_id: str, *, proof: AuthenticationProof
    ):
        """Read a Candidate and its current Hypothesis through SAME_STATE only."""
        if not self._connection.in_transaction:
            with self._lock, self._transaction():
                return self.exact_associated_current_producers(
                    candidate_id, proof=proof
                )
        retained = self._exact_current_receipt(candidate_id)
        admission, _, candidate, *_ = retained
        self._verify_upstream(
            {admission.canonical_digest: retained}, exact=True
        )
        manifest = candidate.governing_manifest
        head = self._connection.execute(
            "SELECT version_id,ordinal,version_digest FROM "
            "event_hypothesis_heads_v2 WHERE hypothesis_id=?",
            (manifest.hypothesis_id,),
        ).fetchone()
        if head is None:
            raise CandidateContractError("Candidate associated Hypothesis is absent")
        head_id = str(head["version_id"])
        if head_id == manifest.hypothesis_version_id:
            producers = self._producers(manifest, proof)
            if self._manifest(producers, retained[3]) != manifest:
                raise CandidateContractError(
                    "Candidate current governing material differs"
                )
            return candidate, producers[0].subject

        base = self._connection.execute(
            "SELECT ordinal FROM event_hypothesis_versions_v2 WHERE version_id=?",
            (manifest.hypothesis_version_id,),
        ).fetchone()
        if base is None:
            raise CandidateContractError("Candidate associated Hypothesis is absent")
        previous_id = manifest.hypothesis_version_id
        previous_digest = manifest.hypothesis_version_digest
        expected_ordinal = int(base[0])
        relationship_digests: list[str] = []
        while previous_id != head_id:
            rows = self._connection.execute(
                "SELECT version_id,ordinal,previous_version_id,"
                "previous_version_digest,proposed_relationship,canonical_bytes,"
                "canonical_digest FROM event_hypothesis_versions_v2 "
                "WHERE hypothesis_id=? AND previous_version_id=? LIMIT 2",
                (manifest.hypothesis_id, previous_id),
            ).fetchall()
            if len(rows) != 1:
                raise CandidateContractError(
                    "Candidate SAME_STATE association chain differs"
                )
            row = rows[0]
            successor = EventHypothesisVersion.from_canonical_bytes(
                bytes(row["canonical_bytes"])
            )
            expected_ordinal += 1
            if (
                successor.version_id != str(row["version_id"])
                or successor.ordinal != int(row["ordinal"])
                or successor.ordinal != expected_ordinal
                or successor.previous_version_id != previous_id
                or successor.previous_version_digest != previous_digest
                or successor.proposed_relationship is not HypothesisRelationship.SAME_STATE
                or successor.canonical_digest != str(row["canonical_digest"])
            ):
                raise CandidateContractError(
                    "Candidate SAME_STATE association chain differs"
                )
            relationship_rows = self._connection.execute(
                "SELECT decision_id FROM event_hypothesis_relationship_decisions "
                "WHERE subject_version_id=? LIMIT 2",
                (successor.version_id,),
            ).fetchall()
            if len(relationship_rows) != 1:
                raise CandidateContractError(
                    "Candidate SAME_STATE relationship differs"
                )
            relationship_digests.append(str(relationship_rows[0][0]))
            previous_id = successor.version_id
            previous_digest = successor.canonical_digest
        if (
            int(head["ordinal"]) != expected_ordinal
            or str(head["version_digest"]) != previous_digest
        ):
            raise CandidateContractError("Candidate associated Hypothesis head differs")

        receipts = (
            self._lineage._require_exact_retained_relationships_in_transaction(
                tuple(relationship_digests)
            )
        )
        previous_id = manifest.hypothesis_version_id
        previous_digest = manifest.hypothesis_version_digest
        for receipt in receipts:
            relationship = receipt.assessment
            selected = relationship.comparator
            if (
                relationship.decision is not CanonicalOutcome.REL_SAME_STATE
                or selected is None
                or selected.version_id != previous_id
                or selected.version_digest != previous_digest
            ):
                raise CandidateContractError(
                    "Candidate SAME_STATE relationship differs"
                )
            previous_id = relationship.subject.version_id
            previous_digest = relationship.subject.version_digest
        snapshot, _, dispositions = (
            self._lineage._require_candidate_inputs_in_transaction(
                head_id, relationship_digests[-1], proof=proof
            )
        )
        if (
            snapshot.subject.version_id != head_id
            or snapshot.subject.canonical_digest != str(head["version_digest"])
        ):
            raise CandidateContractError("Candidate associated Hypothesis head differs")
        lead_ids = tuple(
            sorted(
                {NewsLeadId.parse(item.decision_lead_id) for item in dispositions},
                key=str,
            )
        )
        _create_discovery_governing_producer_read_port(
            self._connection,
            object_admission_payload_validator=self._validate_object_admission_payload_record,
            validate_retained_event=self._validate_retained_event,
        ).require_current_governing_producers(lead_ids)
        return candidate, snapshot.subject

    def _exact_current_receipt(self, candidate_id: str):
        verified = self._verify_local()
        row = self._connection.execute(
            "SELECT current_admission_digest FROM "
            "story_candidate_heads WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise CandidateContractError("unknown Candidate")
        try:
            current = verified[str(row[0])]
        except KeyError as exc:
            raise CandidateContractError("Candidate head differs") from exc
        if current[2].candidate_id != candidate_id:
            raise CandidateContractError("Candidate head differs")
        return current

    def exact_current_candidate(self, candidate_id: str):
        if not self._connection.in_transaction:
            with self._lock, self._transaction():
                return self.exact_current_candidate(candidate_id)
        return self._exact_current_receipt(candidate_id)[2]

    def versions(self, candidate_id: str):
        with self._lock, self._transaction():
            self._verify()
            rows = self._connection.execute(
                "SELECT version_bytes FROM story_candidate_admission_receipts_v2 WHERE candidate_id=? ORDER BY version_ordinal",
                (candidate_id,),
            ).fetchall()
            return tuple(
                StoryCandidateVersion.from_canonical_bytes(bytes(row[0]))
                for row in rows
            )

    def load_version(self, version_id: str):
        with self._lock, self._transaction():
            verified = self._verify()
            row = self._connection.execute(
                "SELECT admission_digest FROM story_candidate_admission_receipts_v2 "
                "WHERE version_id=?",
                (version_id,),
            ).fetchone()
            if row is None:
                raise CandidateContractError("unknown Candidate Version")
            return verified[str(row[0])][2]

    def current(
        self,
        candidate_id: str,
        *,
        collision_request: CurrentCollisionEligibilityRequest,
        proof: AuthenticationProof,
    ):
        if type(collision_request) is not CurrentCollisionEligibilityRequest:
            raise CandidateContractError("Candidate current collision request differs")
        self._authenticator.authenticate(proof, now=self._clock())
        with self._lock, self._transaction():
            verified = self._verify_local()
            row = self._connection.execute(
                "SELECT r.admission_digest,r.disposition_ids_bytes "
                "FROM story_candidate_heads h JOIN "
                "story_candidate_admission_receipts_v2 r "
                "ON r.admission_digest=h.current_admission_digest "
                "WHERE h.candidate_id=?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise CandidateContractError("unknown Candidate")
            _, _, v, *_ = verified[str(row["admission_digest"])]
            binding = collision_request.binding
            if (binding.operation is not CandidateUseOperation.USE_CURRENT_CANDIDATE
                or binding.expected_candidate_id != candidate_id
                or _collision_position(binding) != _manifest_position(v.governing_manifest)):
                raise CandidateContractError(
                    "Candidate current collision binding differs"
                )
            manifest = v.governing_manifest
            disposition_ids = tuple(json.loads(bytes(row["disposition_ids_bytes"])))
            producers = self._producers(manifest, proof)
            if producers[2] != disposition_ids:
                raise CandidateContractError("Candidate dispositions differ")

            def use(collision):
                rebuilt = self._manifest(producers, collision)
                if (
                    rebuilt.version_material_digest != manifest.version_material_digest
                    or rebuilt.semantic_scope_digest != manifest.semantic_scope_digest
                ):
                    raise CandidateContractError(
                        "Candidate current governing material differs"
                    )
                return v

            return self._collision.enforce(request=collision_request, effect=use)

    def rollback_scope(self, operation: Callable[[object], None]) -> None:
        """Test-only inspection of real uncommitted Candidate authority rows."""
        if not callable(operation):
            raise CandidateContractError("Candidate rollback operation differs")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                operation(self)
                self._verify()
            finally:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")


# fmt: on


class _Authority:
    def __init__(self, store):
        self._store = store

    def __getattr__(self, name):
        return getattr(self._store, name)

    def close(self):
        self._store.close()


class _StoryCandidateReadAuthority:
    """Private same-connection adapter for complete v24 retained reads."""

    __slots__ = ("__connection", "__verifier")

    def __init__(
        self, token: object, connection: sqlite3.Connection, verifier: _CandidateStore
    ) -> None:
        if token is not _READ_AUTHORITY_TOKEN or verifier._connection is not connection:
            raise CandidateContractError(
                "Candidate read authority construction is private"
            )
        self.__connection = connection
        self.__verifier = verifier

    def __verified(self):
        _require_candidate_read_connection(self.__connection, active=True)
        return self.__verifier._verify()

    def __verified_receipts(self, *, version_id=None, candidate_id=None, version_ids=None):
        _require_candidate_read_connection(self.__connection, active=True)
        candidate_ids = None
        if candidate_id is not None:
            candidate_ids = frozenset({candidate_id})
        elif version_id is not None or version_ids is not None:
            requested = version_ids if version_ids is not None else (version_id,)
            candidate_ids = frozenset(row[0] for row in self.__connection.execute(
                "SELECT DISTINCT candidate_id FROM story_candidate_admission_receipts_v2 "
                "WHERE version_id IN (SELECT value FROM json_each(?))", (json.dumps(sorted(requested)),),
            ))
        verified = self.__verifier._verify_local(candidate_ids=candidate_ids)
        matches = tuple(
            item
            for item in verified.values()
            if (version_id is None or item[2].version_id == version_id)
            and (candidate_id is None or item[2].candidate_id == candidate_id)
            and (version_ids is None or item[2].version_id in version_ids)
        )
        if not matches and version_ids is None:
            raise CandidateContractError(
                "unknown Candidate" if candidate_id is not None
                else "unknown Candidate Version"
            )
        if version_id is not None and len(matches) != 1:
            raise CandidateContractError("unknown Candidate Version")
        self.__verifier._verify_upstream(
            {item[0].canonical_digest: item for item in matches}, exact=True
        )
        return matches

    def verify_retained_integrity_in_transaction(self) -> None:
        self.__verified()

    def require_retained_candidate_in_transaction(
        self, candidate_id: str
    ) -> StoryCandidate:
        matches = self.__verified_receipts(candidate_id=candidate_id)
        candidate = matches[0][1]
        if any(item[1] != candidate for item in matches):
            raise CandidateContractError("Candidate identity differs")
        return candidate

    def require_retained_version_in_transaction(
        self, version_id: str
    ) -> StoryCandidateVersion:
        return self.__verified_receipts(version_id=version_id)[0][2]

    def require_retained_versions_in_transaction(
        self, version_ids: tuple[str, ...],
    ) -> tuple[StoryCandidateVersion | None, ...]:
        matches = self.__verified_receipts(version_ids=frozenset(version_ids))
        by_version = {item[2].version_id: item[2] for item in matches}
        return tuple(by_version.get(version_id) for version_id in version_ids)

    def require_current_head_in_transaction(
        self, candidate_id: str, *, proof: AuthenticationProof
    ) -> StoryCandidateVersion:
        _require_candidate_read_connection(self.__connection, active=True)
        version, _ = self.__verifier.exact_current_producers(
            candidate_id, proof=proof
        )
        return version


def _require_candidate_read_connection(
    connection: sqlite3.Connection, *, active: bool
) -> None:
    state = "active" if active else "idle"
    if (
        type(connection) is not sqlite3.Connection
        or connection.isolation_level is not None
        or connection.row_factory is not sqlite3.Row
        or connection.in_transaction is not active
    ):
        raise CandidateContractError(
            f"Candidate read requires an exact {state} checked connection"
        )
    try:
        settings = (
            connection.execute("PRAGMA foreign_keys").fetchone()[0],
            str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
            connection.execute("PRAGMA synchronous").fetchone()[0],
        )
    except Exception as exc:
        raise CandidateContractError(
            "Candidate checked connection cannot be inspected"
        ) from exc
    if settings != (1, "wal", 2):
        raise CandidateContractError("Candidate checked connection settings differ")


def _create_story_candidate_read_port(
    connection: sqlite3.Connection,
    *,
    retrieval_authority: RetrievalContextAuthority,
    authenticator: object,
    command_registry: CommandRegistry,
    payload_schemas: PayloadSchemaRegistry,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
    command_service_version: str = "increment6-candidate-v1",
    bounded_version: Callable[[str], StoryCandidateVersion] | None = None,
    object_admission_payload_validator: Callable[
        [sqlite3.Connection, sqlite3.Row], None
    ]
    | None = None,
    current_candidate_citations: CurrentCandidateCitationReadPort | None = None,
    hypotheses: _HypothesisStore | None = None,
    dispositions: ProposalDispositionStore | None = None,
) -> StoryCandidateReadPort:
    """Bind complete Candidate reads to one caller-owned transaction."""

    try:
        _require_candidate_read_connection(connection, active=False)
        if (
            type(retrieval_authority) is not RetrievalContextAuthority
            or type(command_registry) is not CommandRegistry
            or type(payload_schemas) is not PayloadSchemaRegistry
            or not callable(clock)
            or (
                dispositions is not None
                and (
                    type(dispositions) is not ProposalDispositionStore
                    or dispositions._connection is not connection
                    or dispositions._retrieval_authority is not retrieval_authority
                    or dispositions._authenticator is not authenticator
                    or dispositions._current_candidate_citations
                    is not current_candidate_citations
                )
            )
        ):
            raise CandidateContractError(
                "Candidate read-port factory collaborators differ"
            )
        relationship_commands, relationship_schemas = (
            merge_relationship_authority_registries(command_registry, payload_schemas)
        )
        lineage_commands, lineage_schemas = merge_lineage_authority_registries(
            relationship_commands, relationship_schemas
        )
        commands, schemas = merge_candidate_authority_registries(
            lineage_commands, lineage_schemas
        )
        lineage = _create_event_hypothesis_lineage_read_port(
            connection,
            retrieval_authority=retrieval_authority,
            authenticator=authenticator,
            command_registry=commands,
            payload_schemas=schemas,
            clock=clock,
            object_admission_payload_validator=object_admission_payload_validator,
            current_candidate_citations=current_candidate_citations,
            hypotheses=hypotheses,
        )
        verifier = object.__new__(_CandidateStore)
        verifier._conn = connection
        verifier._closed = False
        verifier._lock = threading.RLock()
        verifier._command_registry = commands
        verifier._payload_schemas = schemas
        verifier._command_service_version = command_service_version
        verifier._lineage = lineage
        if object_admission_payload_validator is not None:
            verifier._validate_object_admission_payload_record = (
                object_admission_payload_validator
            )
        verifier._dispositions = dispositions or ProposalDispositionStore(
            connection, retrieval_authority, authenticator,
            current_candidate_citations,
        )
        private = _StoryCandidateReadAuthority(
            _READ_AUTHORITY_TOKEN, connection, verifier
        )
        port = _compose_story_candidate_read_port(
            private, bounded_version=bounded_version
        )
        if type(port) is not StoryCandidateReadPort:
            raise CandidateContractError(
                "Candidate read-port factory returned a forged port"
            )
        _require_candidate_read_connection(connection, active=False)
        return port
    except CandidateContractError:
        raise
    except Exception as exc:
        raise CandidateContractError(
            "Candidate read-port factory failed closed"
        ) from exc


def open_story_candidate_authority_system(database: str | Path, **kwargs):
    raw = _Authority(_CandidateStore(_TOKEN, Path(database), **kwargs))
    try:
        facade = _compose_story_candidate_authority(raw)
        if type(facade) is not StoryCandidateAuthority:
            raise CandidateContractError("Candidate facade differs")
        return facade
    except BaseException:
        try:
            raw.close()
        except BaseException:  # noqa: BLE001, S110 - preserve facade failure
            pass
        raise


class _UnlockedCandidateStore(_CandidateStore):
    def _acquire_writer_lock(self) -> None:
        self._lock_fd = None


def _open_unlocked_story_candidate_authority_for_test(database: str | Path, **kwargs):
    return _Authority(_UnlockedCandidateStore(_TOKEN, Path(database), **kwargs))


__all__ = ["open_story_candidate_authority_system"]
