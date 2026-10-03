"""One governed native Story-to-private-serving publication transaction."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from newsroom.authority import (
    AggregateId,
    AuthenticationProof,
    AuthorityCommands,
    AuthorityEvents,
    GovernedObjects,
    HydrationRequest,
    ObjectAdmissionId,
    ObjectAdmissionPayload,
    ObjectAdmissionRequest,
    SemanticCommand,
)
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    validate_sha256_digest,
)
from newsroom.authority.types import UtcTimestamp
from newsroom.control_plane.admission import (
    WRITE_ADMISSION_POLICY_VERSION, write_admission_revalidation_due,
)
from newsroom.control_plane.native_evidence import (
    NativeEvidenceController,
    NativeEvidenceHold,
    NativeEvidenceSource,
)
from newsroom.control_plane.native_assessor import (
    assessor_admission_recovery_due,
    assessment_revalidation_due,
    same_assessment_producer,
    RetainedAssessorContractFailure,
    RetainedAssessorResult,
    VERSION as ASSESSOR_PRODUCER_VERSION,
    RetainedAssessorPreDispatchFailure,
)
from newsroom.control_plane.native_progress import NativeRevisionJournal
from newsroom.control_plane.veto import OperatorDrainRequested, VetoError
from newsroom.increment6.candidates import StoryCandidateReadPort
from newsroom.increment10.editorial import (
    DECISION_ADMISSION_TYPE,
    DECISION_CLASS,
    DECISION_COMMAND,
    DECISION_USE,
    DecisionReference,
    EditorialError,
    EditorialHold,
    EditorialPolicyDecision,
    NativeEditorial,
    STORY_COMMAND,
    STORY_ADMISSION_TYPE,
    STORY_EVENT,
    STORY_PURPOSE,
    STORY_CLASS,
    STORY_USE,
    StoryVersionReceipt,
    StoryVersion,
    StoryVersionRequest,
)
from newsroom.increment10.evidence import GovernedEvidencePackages
from newsroom.increment10.private_serving import (
    ATTEMPT_COMMAND,
    ATTEMPT_EVENT,
    ATTEMPT_PURPOSE,
    EVIDENCE_PURPOSE,
    AttemptReceipt,
    EvidenceReceipt,
    PrivateServingDelivery,
    PrivateServingReadProof,
    open_private_serving_read_port,
    open_private_serving_delivery,
)
from newsroom.increment10.publication import (
    LAUNCH_CAPABILITIES,
    OfflinePublication,
    PublicationReceipt,
    PublicationRequest,
    TRANSACTION_PURPOSE,
)
from newsroom.increment10.ingress import NON_PUBLIC_EVIDENCE_INTAKE_BOUNDARY


class NativePublicationError(ValueError):
    """Raised when the connected native publication transaction cannot advance."""


_MAX_ACQUISITION_ATTEMPTS = 3
_RETRYABLE_ACQUISITION_HOLDS = frozenset({
    "GOVUK_ACQUISITION_UNAVAILABLE",
    "WEATHER_ACQUISITION_UNAVAILABLE",
})
_REFRESHABLE_EVIDENCE_HOLDS = frozenset({
    "CURRENT_RIGHTS_HOLD",
    "GOVUK_LICENCE_BINDING_HOLD",
    "GOVUK_LICENCE_REVIEW_HOLD",
    "NATIVE_SOURCE_RIGHTS_HOLD",
    "PUBLICATION_RIGHTS_HOLD",
})


@dataclass(frozen=True, slots=True)
class NativePublicationContinuationResult:
    state: str
    reason: str | None
    publication: NativePublicationResult | None


@dataclass(frozen=True, slots=True)
class NativePublicationBindings:
    target_path: Path
    reader_principal_id: str
    authority_domain: str
    editorial_controller_principal_id: str
    story_principal_id: str
    publication_controller_principal_id: str
    serving_adapter_principal_id: str
    editorial_policy_bundle_digest: str
    editorial_decision_hydration_policy_digest: str
    editorial_story_hydration_policy_digest: str
    editorial_decision_admission_definition_digest: str
    editorial_decision_command_definition_digest: str
    editorial_story_command_definition_digest: str
    editorial_story_admission_definition_digest: str
    publication_authorisation_policy_digest: str
    target_id: str
    target_policy_digest: str
    publication_surface_hydration_policy_digest: str
    publication_transaction_hydration_policy_digest: str
    publication_surface_admission_definition_digest: str
    publication_transaction_admission_definition_digest: str
    publication_command_definition_digest: str
    target_context_digest: str
    serving_attempt_hydration_policy_digest: str
    serving_evidence_hydration_policy_digest: str
    serving_attempt_admission_definition_digest: str
    serving_evidence_admission_definition_digest: str
    serving_attempt_command_definition_digest: str
    serving_evidence_command_definition_digest: str
    source_licence_policy: tuple[tuple[str, str, str], ...] = ()
    retained_policy_pairs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.retained_policy_pairs) is not tuple or any(
            type(pair) is not tuple or len(pair) != 2 for pair in self.retained_policy_pairs
        ):
            raise NativePublicationError("retained publication policy pairs differ")
        for pair in self.retained_policy_pairs:
            for digest in pair:
                validate_sha256_digest(digest)
        if not isinstance(self.target_path, Path):
            raise NativePublicationError("native publication target path is required")
        for name in (
            "reader_principal_id",
            "authority_domain",
            "editorial_controller_principal_id",
            "story_principal_id",
            "publication_controller_principal_id",
            "serving_adapter_principal_id",
            "target_id",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise NativePublicationError(
                    f"native publication {name} is required"
                )
        for name in self.__dataclass_fields__:
            if name.endswith("_digest"):
                try:
                    validate_sha256_digest(getattr(self, name))
                except Exception as exc:
                    raise NativePublicationError(
                        f"native publication {name} differs"
                    ) from exc


@dataclass(frozen=True, slots=True)
class NativePublicationResult:
    story_receipt: StoryVersionReceipt
    publication_receipt: PublicationReceipt
    attempt_receipt: AttemptReceipt
    evidence_receipt: EvidenceReceipt
    read_proof: PrivateServingReadProof
    writer_id: str = ""


class NativePublicationController:
    """Compose existing native authorities into one retry-stable transaction."""

    def __init__(
        self,
        *,
        objects: GovernedObjects,
        commands: AuthorityCommands,
        events: AuthorityEvents,
        candidate_port: StoryCandidateReadPort,
        evidence_packages: GovernedEvidencePackages,
        bindings: NativePublicationBindings,
        clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
        story_writer=None,
        source_currentness_fence: Callable | None = None,
    ) -> None:
        if not callable(clock) or not all(
            type(value) is expected
            for value, expected in (
                (objects, GovernedObjects),
                (commands, AuthorityCommands),
                (events, AuthorityEvents),
                (candidate_port, StoryCandidateReadPort),
                (evidence_packages, GovernedEvidencePackages),
                (bindings, NativePublicationBindings),
            )
        ):
            raise NativePublicationError(
                "exact native publication authorities required"
            )
        if source_currentness_fence is not None and not callable(source_currentness_fence):
            raise NativePublicationError("native source currentness fence must be callable")
        self._source_currentness_fence = source_currentness_fence
        self._objects = objects
        self._events = events
        self._commands = commands
        self._candidate_port = candidate_port
        self._evidence = evidence_packages
        self._bindings = bindings
        self._clock = clock
        self._editorial = NativeEditorial(
            objects=objects,
            commands=commands,
            events=events,
            evidence=evidence_packages,
            reader_principal_id=bindings.reader_principal_id,
            reader_authority_domain=bindings.authority_domain,
            controller_principal_id=bindings.editorial_controller_principal_id,
            story_principal_id=bindings.story_principal_id,
            policy_bundle_digest=bindings.editorial_policy_bundle_digest,
            decision_hydration_policy_digest=(
                bindings.editorial_decision_hydration_policy_digest
            ),
            story_hydration_policy_digest=(
                bindings.editorial_story_hydration_policy_digest
            ),
            decision_command_definition_digest=(
                bindings.editorial_decision_command_definition_digest
            ),
            story_command_definition_digest=(
                bindings.editorial_story_command_definition_digest
            ),
            story_admission_definition_digest=(
                bindings.editorial_story_admission_definition_digest
            ),
            story_writer=story_writer, clock=clock,
            retained_policy_bundles=tuple(pair[0] for pair in bindings.retained_policy_pairs),
        )
        self.writer_contract_version = ("newsroom.native-story-writer.v1" if story_writer is not None
                                        else "newsroom.offline-exact-copy.v3")
        self._publication = OfflinePublication(
            objects=objects,
            commands=commands,
            events=events,
            editorial=self._editorial,
            evidence=evidence_packages,
            reader_principal_id=bindings.reader_principal_id,
            authority_domain=bindings.authority_domain,
            controller_principal_id=bindings.publication_controller_principal_id,
            authorisation_policy_digest=(
                bindings.publication_authorisation_policy_digest
            ),
            target_id=bindings.target_id,
            target_policy_digest=bindings.target_policy_digest,
            target_capabilities=LAUNCH_CAPABILITIES,
            surface_hydration_policy_digest=(
                bindings.publication_surface_hydration_policy_digest
            ),
            transaction_hydration_policy_digest=(
                bindings.publication_transaction_hydration_policy_digest
            ),
            surface_admission_definition_digest=(
                bindings.publication_surface_admission_definition_digest
            ),
            transaction_admission_definition_digest=(
                bindings.publication_transaction_admission_definition_digest
            ),
            command_definition_digest=bindings.publication_command_definition_digest,
            source_licence_policy=bindings.source_licence_policy,
            retained_policy_pairs=bindings.retained_policy_pairs,
        )
        self._delivery = open_private_serving_delivery(
            bindings.target_path,
            objects=objects,
            commands=commands,
            events=events,
            publication=self._publication,
            adapter_principal_id=bindings.serving_adapter_principal_id,
            authority_domain=bindings.authority_domain,
            target_id=bindings.target_id,
            target_context_digest=bindings.target_context_digest,
            attempt_hydration_policy_digest=(
                bindings.serving_attempt_hydration_policy_digest
            ),
            evidence_hydration_policy_digest=(
                bindings.serving_evidence_hydration_policy_digest
            ),
            attempt_admission_definition_digest=(
                bindings.serving_attempt_admission_definition_digest
            ),
            evidence_admission_definition_digest=(
                bindings.serving_evidence_admission_definition_digest
            ),
            attempt_command_definition_digest=(
                bindings.serving_attempt_command_definition_digest
            ),
            evidence_command_definition_digest=(
                bindings.serving_evidence_command_definition_digest
            ),
        )

    def close(self) -> None:
        self._delivery.close()

    def restore_current_publisher_output(self, journal: NativeRevisionJournal, *, proof: AuthenticationProof) -> None:
        """Append one factual correction from already acknowledged current copy.

        Original ACK references remain immutable facts. A tiny retained intent
        makes interrupted correction resume its exact slot without a model call.
        """
        if self._source_currentness_fence is None:
            return
        references = ('story_event_id','publication_event_id','delivery_attempt_event_id','delivery_evidence_event_id')
        groups = {}
        for revision, progress in journal.iter_summaries():
            facts = progress.get('facts', {})
            if (progress.get('stage') != 'ACKNOWLEDGED' or not facts.get('candidate_id')
                    or not all(facts.get(key) for key in references)):
                continue
            units = journal.units.get(revision, ())
            if units and units[0].source_id == 'HK-02':
                groups.setdefault(facts['candidate_id'], []).append((revision,facts))
        for members in groups.values():
            try:
                self._restore_publisher_group(journal,members,references,proof)
            except (OperatorDrainRequested,VetoError):
                raise
            except Exception as exc:
                # One unproved or conflicting original chain cannot block peers.
                # No source/exception text or successful correction is invented.
                for revision, _ in members:
                    facts = dict(journal.current(revision)['facts'])
                    facts['factual_correction_hold'] = type(exc).__name__
                    journal.advance(revision,stage='ACKNOWLEDGED',facts=facts)
                continue

    def _restore_publisher_group(self,journal,members,references,proof):
        if len(members) < 2:
            return
        selected = []
        for revision, facts in members:
            result, story = self.read_acknowledged({key:facts[key]for key in references}, proof=proof)
            if story.copy.writer_id != 'newsroom.native-story-writer.v1':
                continue
            package = self._evidence.read(story.package_admission_id, candidate_port=self._candidate_port, proof=proof)
            policy = self._editorial._read_policy_decision(
                DecisionReference(story.policy_decision_event_id,story.policy_decision_admission_id),
                retained=package,proof=proof)
            selected.append((revision,facts,result,story,package,policy))
            correction = facts.get('factual_correction_result')
            if correction is not None:
                derived, derived_story = self.read_acknowledged(correction,proof=proof)
                if (derived_story.story_id != story.story_id
                        or derived.attempt_receipt.publication_id != result.attempt_receipt.publication_id
                        or derived.story_receipt.aggregate_version <= result.story_receipt.aggregate_version):
                    raise NativePublicationError('derived factual correction belongs to another acknowledged chain')
                derived_package = self._evidence.read(derived_story.package_admission_id,
                    candidate_port=self._candidate_port,proof=proof)
                derived_policy = self._editorial._read_policy_decision(
                    DecisionReference(derived_story.policy_decision_event_id,derived_story.policy_decision_admission_id),
                    retained=derived_package,proof=proof)
                selected.append((revision,{**facts,**correction},derived,derived_story,derived_package,derived_policy))
        if len(selected) < 2:
            return
        latest = max(selected, key=lambda item:item[2].story_receipt.aggregate_version)
        prior_facts = latest[1]
        intent = prior_facts.get('factual_correction_intent') if 'factual_correction_result' not in prior_facts else None
        if intent is None:
            try:
                self._source_currentness_fence(latest[4].package,latest[5].currentness)
            except EditorialHold as exc:
                if str(exc) != 'NATIVE_STORY_SOURCE_SUPERSEDED':
                    return
            else:
                return
            current = {}
            for item in selected:
                if item is latest:
                    continue
                try:self._source_currentness_fence(item[4].package,item[5].currentness)
                except EditorialHold:continue
                identity = str(item[3].package_admission_id)
                previous = current.get(identity)
                if previous is None or item[2].story_receipt.aggregate_version > previous[2].story_receipt.aggregate_version:
                    current[identity] = item
            if len(current) != 1:
                return
            chosen = next(iter(current.values()))
            intent = {'superseded':{key:prior_facts[key]for key in references},
                'reviewed':{key:chosen[1][key]for key in references},
                'package_admission_id':str(chosen[3].package_admission_id),
                'editorial_decision':json.loads(chosen[5].canonical_bytes()),
                'expected_story_version':latest[2].story_receipt.aggregate_version,
                'expected_publication_version':latest[2].attempt_receipt.aggregate_version}
            facts = dict(journal.current(latest[0])['facts'])
            facts['factual_correction_intent'] = intent
            journal.advance(latest[0],stage='ACKNOWLEDGED',facts=facts)
        old, _ = self.read_acknowledged(intent['superseded'],proof=proof)
        reviewed, _ = self.read_acknowledged(intent['reviewed'],proof=proof)
        try:
            corrected = self.advance(ObjectAdmissionId.parse(intent['package_admission_id']),
                EditorialPolicyDecision.from_bytes(canonical_json_bytes(intent['editorial_decision'])),
                expected_story_version=intent['expected_story_version'],
                expected_publication_version=intent['expected_publication_version'],expected_delivery_evidence_version=0,
                proof=proof,factual_correction_of=old,reviewed_copy_from=reviewed)
        except EditorialHold as exc:
            facts = dict(journal.current(latest[0])['facts'])
            facts['factual_correction_hold'] = str(exc)
            journal.advance(latest[0],stage='ACKNOWLEDGED',facts=facts)
            return
        facts = dict(journal.current(latest[0])['facts'])
        facts.pop('factual_correction_hold',None)
        facts['factual_correction_result'] = dict(zip(references,(
            corrected.story_receipt.event_id,corrected.publication_receipt.event_id,
            corrected.attempt_receipt.event_id,corrected.evidence_receipt.event_id)))
        journal.advance(latest[0],stage='ACKNOWLEDGED',facts=facts)

    def advance(
        self,
        package_admission_id: ObjectAdmissionId,
        editorial_decision: EditorialPolicyDecision,
        *,
        expected_story_version: int,
        expected_publication_version: int,
        expected_delivery_evidence_version: int,
        proof: AuthenticationProof,
        correction_of: NativePublicationResult | None = None,
        factual_correction_of: NativePublicationResult | None = None,
        reviewed_copy_from: NativePublicationResult | None = None,
        reconciled_predecessor: dict | None = None,
    ) -> NativePublicationResult:
        if (
            type(package_admission_id) is not ObjectAdmissionId
            or type(editorial_decision) is not EditorialPolicyDecision
            or (correction_of is not None and type(correction_of) is not NativePublicationResult)
            or any(
                type(value) is not int or value < 0
                for value in (
                    expected_story_version,
                    expected_publication_version,
                    expected_delivery_evidence_version,
                )
            )
        ):
            raise NativePublicationError("native publication request differs")
        retained = self._evidence.read(
            package_admission_id,
            candidate_port=self._candidate_port,
            proof=proof,
        )
        if factual_correction_of is not None or reviewed_copy_from is not None:
            if (type(factual_correction_of) is not NativePublicationResult
                    or type(reviewed_copy_from) is not NativePublicationResult
                    or correction_of is not None or reconciled_predecessor is not None
                    or self._source_currentness_fence is None):
                raise NativePublicationError("factual correction requires exact source and ACK authorities")
            def acknowledged(result):
                return self.read_acknowledged({
                    "story_event_id": result.story_receipt.event_id,
                    "publication_event_id": result.publication_receipt.event_id,
                    "delivery_attempt_event_id": result.attempt_receipt.event_id,
                    "delivery_evidence_event_id": result.evidence_receipt.event_id,
                }, proof=proof)
            old, old_story = acknowledged(factual_correction_of)
            reviewed, reviewed_story = acknowledged(reviewed_copy_from)
            current_decision = self._editorial.current_copy_decision(editorial_decision)
            reviewed_policy = self._editorial._read_policy_decision(
                DecisionReference(reviewed_story.policy_decision_event_id,reviewed_story.policy_decision_admission_id),
                retained=retained,proof=proof)
            expected_decision = self._editorial.current_copy_decision(reviewed_policy)
            if (old != factual_correction_of or reviewed != reviewed_copy_from
                    or old_story.story_id != _aggregate("story", retained.package.candidate_id)
                    or reviewed_story.story_id != old_story.story_id
                    or reviewed_story.package_admission_id != package_admission_id
                    or old_story.package_admission_id == package_admission_id
                    or old.story_receipt.aggregate_version != expected_story_version
                    or old.attempt_receipt.aggregate_version != expected_publication_version
                    or expected_delivery_evidence_version != 0):
                raise NativePublicationError("factual correction predecessor binding differs")
            if (current_decision.package_digest != retained.package.digest
                    or current_decision.canonical_bytes() != expected_decision.canonical_bytes()):
                raise NativePublicationError("factual correction current package differs")
            editorial_decision = current_decision
        if reconciled_predecessor is not None:
            prior, prior_story = self.read_acknowledged(reconciled_predecessor, proof=proof)
            prior_package = self._evidence.read(
                prior_story.package_admission_id, candidate_port=self._candidate_port, proof=proof,
            )
            if (
                correction_of is not None
                or prior_story.package_admission_id == package_admission_id
                or prior_package.package.candidate_id != retained.package.candidate_id
                or prior.story_receipt.story_id != _aggregate("story", retained.package.candidate_id)
                or prior.attempt_receipt.publication_id != _aggregate("publication", retained.package.candidate_id)
                or prior.story_receipt.aggregate_version != expected_story_version
                or prior.attempt_receipt.aggregate_version != expected_publication_version
                or expected_delivery_evidence_version != 0
            ):
                raise NativePublicationError("stale publication predecessor binding differs")
            # The original decision remains retained; only the fresh Story uses
            # the canonical current member of this exact known policy pair.
            editorial_decision = self._editorial.current_copy_decision(editorial_decision)
        if correction_of is not None:
            old, old_story = self.read_acknowledged({
                "story_event_id": correction_of.story_receipt.event_id,
                "publication_event_id": correction_of.publication_receipt.event_id,
                "delivery_attempt_event_id": correction_of.attempt_receipt.event_id,
                "delivery_evidence_event_id": correction_of.evidence_receipt.event_id,
            }, proof=proof)
            prior_decision = self._editorial._read_policy_decision(
                DecisionReference(old_story.policy_decision_event_id, old_story.policy_decision_admission_id),
                retained=retained, proof=proof,
            )
            try:
                current_decision = self._editorial.current_copy_decision(editorial_decision)
                expected_decision = self._editorial.current_copy_decision(prior_decision)
            except EditorialError as exc:
                raise NativePublicationError("copy correction predecessor binding differs") from exc
            if (
                old != correction_of
                or old_story.copy.writer_id not in (
                    {"newsroom.offline-exact-copy.v1", "newsroom.offline-exact-copy.v2", "newsroom.offline-exact-copy.v3"}
                    if self.writer_contract_version == "newsroom.native-story-writer.v1"
                    else {"newsroom.offline-exact-copy.v2"})
                or old_story.package_admission_id != package_admission_id
                or old_story.policy_decision_id != prior_decision.decision_id
                or current_decision.canonical_bytes() != expected_decision.canonical_bytes()
                or old_story.candidate_version_id != retained.candidate_version_id
                or old.story_receipt.aggregate_version != expected_story_version
                or old.attempt_receipt.aggregate_version != expected_publication_version
                or expected_delivery_evidence_version != 0
            ):
                raise NativePublicationError("copy correction predecessor binding differs")
            editorial_decision = current_decision
        identity = retained.package.candidate_id
        story_id = _aggregate("story", identity)
        publication_id = _aggregate("publication", identity)
        if self._source_currentness_fence is not None:
            own_story = self._objects.committed_admission(ObjectAdmissionRequest(
                STORY_ADMISSION_TYPE, f"story-version:{story_id}:{expected_story_version + 1}",
            ), proof=proof)
            if own_story is None:
                # Do not spend on a fresh draft for a superseded publisher state.
                # An admitted Story alone does not waive the fresh effect fence below.
                self._source_currentness_fence(retained.package, editorial_decision.currentness)
        decision_reference = self._record_decision(editorial_decision, proof=proof)
        story_receipt, _story = self._editorial.admit_story_version(
            StoryVersionRequest(
                story_id,
                expected_story_version,
                f"native-story:{identity}:{expected_story_version + 1}",
            ),
            package_admission_id=package_admission_id,
            decision_reference=decision_reference,
            candidate_port=self._candidate_port,
            proof=proof,
            reviewed_copy_from=(None if reviewed_copy_from is None else reviewed_copy_from.story_receipt),
        )
        if correction_of is not None and _story.copy.writer_id != self.writer_contract_version:
            raise NativePublicationError("copy correction writer differs")
        publication_receipt, _transaction = self._publication.decide(
            PublicationRequest(
                publication_id,
                expected_publication_version,
                f"native-publication:{identity}:{expected_publication_version + 1}",
                "AUTO_PUBLISH",
                (("NATIVE_FACTUAL_CORRECTION", "NATIVE_STORY_WRITE_READY") if factual_correction_of is not None else
                 ("NATIVE_COPY_CORRECTION", "NATIVE_STORY_WRITE_READY") if correction_of is not None else
                 ("NATIVE_STORY_WRITE_READY",)),
                editorial_decision.evaluated_at,
            ),
            story_receipt=story_receipt,
            candidate_port=self._candidate_port,
            proof=proof,
        )
        attempt_receipt, _batch = self._delivery.begin(
            publication_receipt,
            story_receipt=story_receipt,
            candidate_port=self._candidate_port,
            proof=proof,
        )
        self._delivery.apply(
            attempt_receipt,
            publication_receipt=publication_receipt,
            story_receipt=story_receipt,
            candidate_port=self._candidate_port,
            # Intent can predate a crash or a blocked writer by hours. Sample
            # the effect boundary; apply preserves any already committed time.
            applied_at=self._clock().to_text(),
            proof=proof,
            before_new_effect=(None if self._source_currentness_fence is None else
                lambda: self._source_currentness_fence(retained.package, editorial_decision.currentness)),
        )
        evidence = self._delivery.observe(
            attempt_receipt,
            publication_receipt=publication_receipt,
            story_receipt=story_receipt,
            candidate_port=self._candidate_port,
            observed_at=self._clock().to_text(),
            proof=proof,
        )
        evidence_receipt = self._delivery.record(
            evidence,
            attempt_receipt,
            expected_version=expected_delivery_evidence_version,
            proof=proof,
        )
        read_proof = self._delivery.acknowledged_read_proof(
            evidence_receipt,
            attempt_receipt,
            publication_receipt=publication_receipt,
            story_receipt=story_receipt,
            candidate_port=self._candidate_port,
            proof=proof,
        )
        if read_proof is None:
            raise NativePublicationError("private delivery is not acknowledged")
        return NativePublicationResult(
            story_receipt,
            publication_receipt,
            attempt_receipt,
            evidence_receipt,
            read_proof,
            _story.copy.writer_id,
        )

    def reconcile_stale_intent(
        self, package_admission_id: ObjectAdmissionId, *, expected_story_version: int,
        expected_publication_version: int, acknowledged: tuple[dict, ...],
        proof: AuthenticationProof,
    ) -> tuple[NativePublicationResult, dict] | None:
        """Move a blocked intent only beyond its exact, fully acknowledged sibling."""
        retained = self._evidence.read(
            package_admission_id, candidate_port=self._candidate_port, proof=proof,
        )
        story_id = _aggregate("story", retained.package.candidate_id)
        occupied = self._objects.committed_admission(ObjectAdmissionRequest(
            STORY_ADMISSION_TYPE, f"story-version:{story_id}:{expected_story_version + 1}",
        ), proof=proof)
        if occupied is None:
            return None
        admission = occupied.admission
        material = self._objects.hydrate(HydrationRequest(admission.admission_id, STORY_PURPOSE), proof=proof)
        self._editorial._verify_access(
            material.decision, policy=self._bindings.editorial_story_hydration_policy_digest,
            object_class=STORY_CLASS, allowed_use=STORY_USE,
        )
        story = StoryVersion.from_bytes(material.data)
        if (
            admission.definition_digest != self._bindings.editorial_story_admission_definition_digest
            or not admission.active or admission.blob.blob_digest != story.digest
            or story.story_id != story_id or story.aggregate_version != expected_story_version + 1
        ):
            raise NativePublicationError("occupied Story authority differs")
        if story.package_admission_id == package_admission_id:
            return None  # Own partial intent must replay its exact original slot.
        keys = ("story_event_id", "publication_event_id", "delivery_attempt_event_id", "delivery_evidence_event_id")
        for facts in acknowledged:
            if not all(key in facts for key in keys):
                continue
            event = self._events.provenance(facts["story_event_id"], proof=proof).event
            if event.object_admission_id != str(admission.admission_id):
                continue
            prior, original = self.read_acknowledged(facts, proof=proof)
            if (
                original.canonical_bytes() != story.canonical_bytes()
                or prior.story_receipt.story_id != story_id
                or prior.attempt_receipt.publication_id != _aggregate("publication", retained.package.candidate_id)
                or prior.attempt_receipt.aggregate_version <= expected_publication_version
            ):
                raise NativePublicationError("stale publication predecessor binding differs")
            return prior, {key: facts[key] for key in keys}
        return None

    def read_acknowledged(
        self, facts: dict, *, proof: AuthenticationProof,
    ) -> tuple[NativePublicationResult, StoryVersion]:
        """Reconstruct receipts, then delegate all bindings to existing readers."""
        def event_for(key):
            return self._events.provenance(facts[key], proof=proof).event

        def metadata(key, purpose):
            event = event_for(key)
            material = self._objects.hydrate(
                HydrationRequest(ObjectAdmissionId.parse(event.object_admission_id), purpose),
                proof=proof,
            )
            if digest_bytes(material.data) != event.payload_digest:
                raise NativePublicationError("acknowledged object digest differs")
            return event, json.loads(material.data)

        try:
            event = event_for("story_event_id")
            story_receipt = StoryVersionReceipt(
                event.command_id, event.event_id, AggregateId.parse(event.aggregate_id),
                event.aggregate_version, ObjectAdmissionId.parse(event.object_admission_id),
                event.payload_digest,
            )
            story = self._editorial.read_story_version(
                story_receipt, candidate_port=self._candidate_port, proof=proof,
            )
            event, value = metadata("publication_event_id", TRANSACTION_PURPOSE)
            publication = PublicationReceipt(
                event.command_id, event.event_id, AggregateId.parse(event.aggregate_id),
                event.aggregate_version, ObjectAdmissionId.parse(event.object_admission_id),
                event.payload_digest, value["transaction_id"], value["decision"]["decision_id"],
                value["bundle"]["bundle_id"], tuple(item["operation_id"] for item in value["operations"]),
            )
            event, value = metadata("delivery_attempt_event_id", ATTEMPT_PURPOSE)
            attempt = AttemptReceipt(
                event.command_id, event.event_id, AggregateId.parse(event.aggregate_id),
                event.aggregate_version, ObjectAdmissionId.parse(event.object_admission_id),
                value["batch_id"], event.payload_digest,
            )
            event, value = metadata("delivery_evidence_event_id", EVIDENCE_PURPOSE)
            evidence = EvidenceReceipt(
                event.command_id, event.event_id, AggregateId.parse(event.aggregate_id),
                event.aggregate_version, ObjectAdmissionId.parse(event.object_admission_id),
                value["evidence_id"], event.payload_digest,
            )
            read_proof = self._delivery.acknowledged_read_proof(
                evidence, attempt, publication_receipt=publication,
                story_receipt=story_receipt, candidate_port=self._candidate_port, proof=proof,
            )
            if read_proof is None:
                raise NativePublicationError("copy correction lacks an acknowledged predecessor")
            reader = open_private_serving_read_port(
                self._bindings.target_path, target_id=self._bindings.target_id,
                target_context_digest=self._bindings.target_context_digest, proof=read_proof,
            )
            try:
                if reader.acknowledged_rows() is None:
                    raise NativePublicationError("copy correction predecessor readback differs")
            finally:
                reader.close()
            return NativePublicationResult(
                story_receipt, publication, attempt, evidence, read_proof, story.copy.writer_id,
            ), story
        except (OperatorDrainRequested, VetoError):
            raise
        except Exception as exc:
            raise NativePublicationError("copy correction predecessor validation failed") from exc

    def retained_writer_id(self, event_id: str, *, proof: AuthenticationProof) -> str:
        """Classify a retained copy without re-admitting its old evidence policy."""
        event = self._events.provenance(event_id, proof=proof).event
        receipt = StoryVersionReceipt(
            event.command_id, event.event_id, AggregateId.parse(event.aggregate_id),
            event.aggregate_version, ObjectAdmissionId.parse(event.object_admission_id), event.payload_digest,
        )
        self._editorial._verify_story_event(receipt, proof=proof)
        material = self._objects.hydrate(
            HydrationRequest(receipt.admission_id, STORY_PURPOSE), proof=proof,
        )
        self._editorial._verify_access(
            material.decision, policy=self._bindings.editorial_story_hydration_policy_digest,
            object_class=STORY_CLASS, allowed_use=STORY_USE,
        )
        story = StoryVersion.from_bytes(material.data)
        if (story.digest, story.story_id, story.aggregate_version) != (
            receipt.story_version_digest, receipt.story_id, receipt.aggregate_version,
        ):
            raise NativePublicationError("retained copy identity differs")
        return story.copy.writer_id

    def _record_decision(
        self,
        decision: EditorialPolicyDecision,
        *,
        proof: AuthenticationProof,
    ) -> DecisionReference:
        raw = decision.canonical_bytes()
        admission = self._objects.admit(
            ObjectAdmissionRequest(DECISION_ADMISSION_TYPE, decision.decision_id),
            raw,
            proof=proof,
        ).admission
        if (
            admission.definition_digest
            != self._bindings.editorial_decision_admission_definition_digest
            or admission.blob.blob_digest != digest_bytes(raw)
            or admission.object_class != DECISION_CLASS
            or admission.allowed_use != DECISION_USE
            or not admission.active
        ):
            raise NativePublicationError("editorial decision admission differs")
        committed = self._commands.execute(
            SemanticCommand(
                DECISION_COMMAND,
                _aggregate("editorial-decision", decision.decision_id),
                0,
                ObjectAdmissionPayload(admission.admission_id),
                f"native-editorial-decision:{decision.decision_id}",
            ),
            proof=proof,
        )
        return DecisionReference(committed.event_id, admission.admission_id)


class NativePublicationContinuation:
    """Resume one Candidate through private evidence and ACK without redispatch."""

    def __init__(
        self,
        *,
        journal: NativeRevisionJournal,
        runtime: object,
        evidence_controller: NativeEvidenceController,
        sources: Mapping[str, tuple[NativeEvidenceSource, ...]],
        assessment_contract_failure: (
            Callable[[object], RetainedAssessorContractFailure | None] | None
        ) = None,
        assessment_pre_dispatch_failure: (
            Callable[[object], RetainedAssessorPreDispatchFailure | None] | None
        ) = None,
        assessment_old_provider_failure: (
            Callable[[object], RetainedAssessorResult | None] | None
        ) = None,
        evidence_sources_for: Callable[[str], tuple[NativeEvidenceSource, ...]] | None = None,
        assessment_contract_version: str | None = None,
        clock=UtcTimestamp.now,
    ) -> None:
        if (
            type(journal) is not NativeRevisionJournal
            or type(evidence_controller) is not NativeEvidenceController
            or not callable(clock)
            or (
                assessment_contract_version is not None
                and (type(assessment_contract_version) is not str or not assessment_contract_version)
            )
            or (
                assessment_contract_failure is not None
                and not callable(assessment_contract_failure)
            )
            or (
                assessment_pre_dispatch_failure is not None
                and not callable(assessment_pre_dispatch_failure)
            )
            or (assessment_old_provider_failure is not None and not callable(assessment_old_provider_failure))
            or (evidence_sources_for is not None and not callable(evidence_sources_for))
            or not isinstance(sources, Mapping)
            or not all(
                type(key) is str
                and type(value) is tuple
                and value
                and all(type(item) is NativeEvidenceSource for item in value)
                for key, value in sources.items()
            )
        ):
            raise NativePublicationError("native continuation composition differs")
        for name in ("authority", "ingress", "publication", "policies", "proof"):
            if not hasattr(runtime, name):
                raise NativePublicationError("native continuation runtime differs")
        self._journal = journal
        self._runtime = runtime
        self._evidence = evidence_controller
        self._sources = dict(sources)
        self._assessment_contract_failure = assessment_contract_failure
        self._assessment_pre_dispatch_failure = assessment_pre_dispatch_failure
        self._assessment_old_provider_failure = assessment_old_provider_failure
        self._evidence_sources_for = evidence_sources_for
        self._assessment_contract_version = assessment_contract_version
        self._clock = clock

    @staticmethod
    def copy_correction_due(facts: dict, writer_contract_version="newsroom.offline-exact-copy.v3") -> bool:
        return (
            facts.get("writer_id") != writer_contract_version
            and facts.get("copy_correction_checked_version") != writer_contract_version
        )

    @staticmethod
    def source_binding_recovery_due(facts: dict, contract_version: str | None) -> bool:
        """Schedule an exact pre-assessment source-binding repair, never authorise it."""
        prior = facts.get("assessment_superseded")
        old = prior.get("provider_failure") if type(prior) is dict else None
        attempts = facts.get("acquisition_attempt_count")
        return (
            facts.get("reason") == "ACQUISITION_RESULT_NOT_RETAINED"
            and facts.get("failure_class") == "KeyError"
            and facts.get("assessment_started_at") is None
            and facts.get("assessment_contract_version") == contract_version
            and same_assessment_producer(contract_version, ASSESSOR_PRODUCER_VERSION)
            and type(attempts) is int and 0 < attempts < _MAX_ACQUISITION_ATTEMPTS
            and type(old) is dict and old.get("outcome") == "ASSESSOR_PROVIDER_FAILED"
            and set(old) == {"outcome", "envelope_id", "invocation_id", "allocation_digest",
                            "terminal_digest", "context_manifest_digest"}
            and all(type(value) is str and value for value in old.values())
            and type(prior.get("contract_version")) is str
            and not same_assessment_producer(prior["contract_version"], contract_version)
        )

    def recover_pre_dispatch(
        self, revision_ids: tuple[str, ...], *,
        failure_many: Callable[[tuple[object, ...]], tuple],
        before_revision: Callable[[], bool],
        denial_many: Callable[..., tuple[bool, ...]] | None = None,
    ) -> tuple[str, ...]:
        """Consume one finite proof-only snapshot before ordinary effects."""
        selected = []
        for revision_id in revision_ids:
            progress = self._journal.summary(revision_id)
            facts = progress.get("facts", {})
            if not (
                (progress.get("stage") == "ASSESSMENT_INTERRUPTED"
                 and facts.get("failure_class") == "NativeEvidenceError")
                or (progress.get("stage") == "EVIDENCE_HOLD"
                    and assessor_admission_recovery_due(facts))
            ):
                continue
            version_id = facts.get("candidate_version_id")
            if type(version_id) is not str or not version_id:
                continue
            if not before_revision():
                break
            selected.append((revision_id, progress, version_id))
        if not selected or not before_revision():
            return ()
        checked = []
        source_order = {revision: index for index, (revision, _, _) in enumerate(selected)}
        if denial_many is not None:
            try:
                denials = denial_many(
                    tuple(version_id for _, _, version_id in selected),
                    expected_candidate_ids=tuple(progress.get("facts", {}).get("candidate_id") for _, progress, _ in selected),
                )
            except (OperatorDrainRequested, VetoError):
                raise
            except Exception:
                # An unavailable denial shortcut leaves the original full read.
                denials = (False,) * len(selected)
            if type(denials) is not tuple or len(denials) != len(selected) or any(type(v) is not bool for v in denials):
                raise NativePublicationError("native allocation-denial partition differs")
            if not before_revision():
                # A denial query is not an atomically started recovery effect.
                # Expired work must not enter survivor reconstruction.
                return ()
            remaining = []
            for (revision_id, progress, version_id), denied in zip(selected, denials, strict=True):
                if not denied:
                    remaining.append((revision_id, progress, version_id))
                elif self._journal.summary(revision_id) == progress and not assessment_revalidation_due(
                    progress.get("facts", {}), self._assessment_contract_version,
                ):
                    # Allocation existence can only deny pre-dispatch recovery.
                    # Contract revalidation still belongs to ordinary advance.
                    checked.append(revision_id)
            selected = remaining
            if not selected:
                return tuple(sorted(checked, key=source_order.__getitem__))
        try:
            versions = self._runtime.authority.candidate_versions(
                tuple(version_id for _, _, version_id in selected)
            )
        except (OperatorDrainRequested, VetoError):
            raise
        except Exception:
            # Global or upstream corruption grants no recovery proof.
            return tuple(sorted(checked, key=source_order.__getitem__))
        if type(versions) is not tuple or len(versions) != len(selected):
            raise NativePublicationError("native Candidate version partition differs")
        retained = []
        for (revision_id, progress, version_id), version in zip(selected, versions, strict=True):
            facts = progress.get("facts", {})
            try:
                if (
                    version is None
                    or version.version_id != version_id
                    or type(version.candidate_id) is not str
                    or not version.candidate_id.strip()
                    or facts.get("candidate_id") not in (None, version.candidate_id)
                ):
                    continue
            except (OperatorDrainRequested, VetoError):
                raise
            except Exception:
                # A failed authoritative read remains ordinary unknown work.
                continue
            retained.append((revision_id, progress, version))
        selected = retained
        if not selected:
            return tuple(sorted(checked, key=source_order.__getitem__))
        # An authoritative read started within the quantum may finish one
        # proved recovery atomically, even if reading the prefix overruns it.
        # The stop/drain check still applies before the proof and every write.
        before_revision()
        failures = failure_many(tuple(version for _, _, version in selected))
        if type(failures) is not tuple or len(failures) != len(selected):
            raise NativePublicationError("native pre-dispatch proof partition differs")
        attempted = []
        # The batch reader has closed its transaction before any journal write.
        # These proofs never enter ordinary advance or survive this call.
        for (revision_id, progress, version), failure in zip(selected, failures, strict=True):
            if not before_revision() and attempted:
                break
            # The thin snapshot includes the ordinal: even a pair-only STATE
            # change invalidates this proof without expanding either pair.
            if self._journal.summary(revision_id) != progress:
                continue
            if failure is None and not assessment_revalidation_due(
                progress.get("facts", {}), self._assessment_contract_version,
            ):
                # This exact proof-only turn was checked and denied. Ordinary
                # advance would only repeat it; no state, proof or effect is minted.
                checked.append(revision_id)
                continue
            if type(failure) is not RetainedAssessorPreDispatchFailure:
                continue
            try:
                result = self._retain_pre_dispatch_hold(
                    revision_id, version.version_id, version, failure,
                )
            except (OperatorDrainRequested, VetoError):
                raise
            except Exception:
                # An attempted proved revision must not enter an effectful
                # second turn in this tick, even when its journal write fails.
                attempted.append(revision_id)
                checked.append(revision_id)
                continue
            if result is not None:
                attempted.append(revision_id)
                checked.append(revision_id)
        return tuple(sorted(checked, key=source_order.__getitem__))

    def _current_candidate_facts(self, revision_id, candidate_version_id, candidate_id) -> dict:
        facts = dict(self._journal.current(revision_id).get("facts", {}))
        if facts.get("candidate_version_id") not in (None, candidate_version_id):
            raise NativePublicationError("native continuation Candidate differs")
        if facts.get("candidate_id") not in (None, candidate_id):
            raise NativePublicationError("native continuation stable Candidate differs")
        facts.update(candidate_version_id=candidate_version_id, candidate_id=candidate_id)
        return facts

    def _retain_pre_dispatch_hold(
        self, revision_id, candidate_version_id, version, pre_dispatch,
    ) -> NativePublicationContinuationResult | None:
        candidate_id = version.candidate_id
        if not (
            type(pre_dispatch) is RetainedAssessorPreDispatchFailure
            and pre_dispatch.candidate_id == candidate_id
            and pre_dispatch.candidate_version_id == candidate_version_id
            and pre_dispatch.governing_manifest_digest
            == version.governing_manifest.canonical_digest
        ):
            return None
        facts = self._current_candidate_facts(revision_id, candidate_version_id, candidate_id)
        attempt_count = facts.get("acquisition_attempt_count", 0)
        if type(attempt_count) is not int or attempt_count < 0:
            raise NativePublicationError("native acquisition attempt differs")
        facts.update(
            candidate_id=candidate_id,
            candidate_version_id=candidate_version_id,
            reason="ASSESSOR_PRE_DISPATCH_HOLD",
            acquisition_retryable=attempt_count < _MAX_ACQUISITION_ATTEMPTS,
            assessment_pre_dispatch_candidate_id=pre_dispatch.candidate_id,
            assessment_pre_dispatch_candidate_version_id=pre_dispatch.candidate_version_id,
            assessment_pre_dispatch_manifest_digest=pre_dispatch.governing_manifest_digest,
            assessment_pre_dispatch_inventory_digest=pre_dispatch.envelope_inventory_digest,
        )
        self._journal.advance(revision_id, stage="EVIDENCE_HOLD", facts=facts)
        return NativePublicationContinuationResult("EVIDENCE_HOLD", facts["reason"], None)

    def advance(
        self, *, revision_id: str, candidate_version_id: str
    ) -> NativePublicationContinuationResult:
        if revision_id not in self._journal.units:
            raise NativePublicationError("native continuation revision differs")
        progress = self._journal.current(revision_id)
        facts = dict(progress.get("facts", {}))
        admission_recovery = (
            progress.get("stage") == "EVIDENCE_HOLD"
            and assessor_admission_recovery_due(facts)
        )
        source_binding_recovery = self.source_binding_recovery_due(facts, self._assessment_contract_version)
        if (
            progress.get("stage") not in {"ASSESSMENT_INTERRUPTED", "ACKNOWLEDGED", "COPY_CORRECTION_PREPARED"}
            and not admission_recovery
            and not (source_binding_recovery and self._evidence_sources_for is not None)
            and revision_id not in self._sources
        ):
            raise NativePublicationError("native continuation revision differs")
        if facts.get("candidate_version_id") not in (None, candidate_version_id):
            raise NativePublicationError("native continuation Candidate differs")
        facts["candidate_version_id"] = candidate_version_id
        version = self._runtime.authority.candidate_version(candidate_version_id)
        candidate_id = getattr(version, "candidate_id", None)
        if type(candidate_id) is not str or not candidate_id.strip():
            raise NativePublicationError("native continuation Candidate identity differs")
        if facts.get("candidate_id") not in (None, candidate_id):
            raise NativePublicationError("native continuation stable Candidate differs")
        facts["candidate_id"] = candidate_id

        def current_facts() -> dict:
            return self._current_candidate_facts(revision_id, candidate_version_id, candidate_id)

        if progress.get("stage") == "COPY_CORRECTION_PREPARED" or (
            progress.get("stage") == "ACKNOWLEDGED"
            and (self.copy_correction_due(facts, getattr(self._runtime.publication, "writer_contract_version", "newsroom.offline-exact-copy.v3")) or facts.get("copy_correction_of"))
        ):
            return self._advance_copy_correction(revision_id, candidate_version_id, facts, progress, current_facts)

        old_provider_failure = None
        if ((progress.get("stage") == "ASSESSMENT_INTERRUPTED" or source_binding_recovery)
                and self._assessment_old_provider_failure is not None
                and same_assessment_producer(self._assessment_contract_version, ASSESSOR_PRODUCER_VERSION)):
            try:
                retained = self._assessment_old_provider_failure(version)
            except (OperatorDrainRequested, VetoError):
                raise
            except Exception:
                retained = None
            if (type(retained) is RetainedAssessorResult
                    and retained.outcome == "ASSESSOR_PROVIDER_FAILED" and retained.execution is None
                    and same_assessment_producer(
                        facts.get("assessment_superseded", {}).get("contract_version")
                        if source_binding_recovery else facts.get("assessment_contract_version"), retained.contract_version,
                    )
                    and not same_assessment_producer(self._assessment_contract_version, retained.contract_version)):
                old = retained.proof
                if not source_binding_recovery or facts["assessment_superseded"]["provider_failure"] == {
                    "outcome": retained.outcome, "envelope_id": old.envelope_id, "invocation_id": old.invocation_id,
                    "allocation_digest": old.allocation_digest, "terminal_digest": old.terminal_digest,
                    "context_manifest_digest": old.context_manifest_digest,
                }:
                    old_provider_failure = retained
        if old_provider_failure is not None or (
            progress.get("stage") == "EVIDENCE_HOLD"
            and assessment_revalidation_due(facts, self._assessment_contract_version)
        ):
            facts = current_facts()
            # Retain the superseded references before clearing continuation-only
            # fields. Intake identity and all original ledger/accounting remain.
            facts["assessment_superseded"] = facts["assessment_superseded"] if source_binding_recovery else {
                "contract_version": facts.get("assessment_contract_version"),
                "reason": facts.get("reason"),
                "package_admission_id": facts.get("package_admission_id"),
                "editorial_decision_id": facts.get("editorial_decision", {}).get("decision_id"),
                "acquisition_attempt_count": facts.get("acquisition_attempt_count", 0),
            }
            if old_provider_failure is not None and not source_binding_recovery:
                old = old_provider_failure.proof
                facts["assessment_superseded"].update(
                    failure_class=facts.get("failure_class"),
                    provider_failure={
                        "outcome": old_provider_failure.outcome, "envelope_id": old.envelope_id,
                        "invocation_id": old.invocation_id, "allocation_digest": old.allocation_digest,
                        "terminal_digest": old.terminal_digest, "context_manifest_digest": old.context_manifest_digest,
                    },
                )
            for key in (
                "package_admission_id", "editorial_decision", "acquisition_receipt_digests",
                "expected_story_version", "expected_publication_version",
                "expected_delivery_evidence_version", "publication_applied_at",
                "publication_observed_at", "publication_started_at",
                "acquisition_retryable", "reason",
                "assessment_started_at", "acquisition_started_at", "failure_class",
                "editorial_hold_reason_codes",
            ):
                facts.pop(key, None)
            facts.update(
                assessment_contract_version=self._assessment_contract_version,
                acquisition_attempt_count=facts.get("acquisition_attempt_count", 0) if source_binding_recovery else 0,
            )
            progress = self._journal.advance(
                revision_id, stage="ASSESSMENT_CONTRACT_REVALIDATION", facts=facts
            )

        if progress.get("stage") == "ASSESSMENT_INTERRUPTED" or admission_recovery:
            retained_failure = None
            if (
                facts.get("failure_class") == "EvidencePackageError"
                and self._assessment_contract_failure is not None
            ):
                retained_failure = self._assessment_contract_failure(version)
            if type(retained_failure) is RetainedAssessorContractFailure:
                facts = current_facts()
                facts.update(
                    reason="ASSESSOR_OUTPUT_CONTRACT_HOLD",
                    acquisition_retryable=False,
                    assessment_failure_envelope_id=retained_failure.envelope_id,
                    assessment_failure_invocation_id=retained_failure.invocation_id,
                    assessment_failure_allocation_digest=(
                        retained_failure.allocation_digest
                    ),
                    assessment_failure_terminal_digest=(
                        retained_failure.terminal_digest
                    ),
                    assessment_failure_context_manifest_digest=(
                        retained_failure.context_manifest_digest
                    ),
                )
                self._journal.advance(
                    revision_id, stage="EVIDENCE_HOLD", facts=facts
                )
                return NativePublicationContinuationResult(
                    "EVIDENCE_HOLD", facts["reason"], None
                )
            pre_dispatch = None
            if (
                (facts.get("failure_class") == "NativeEvidenceError" or admission_recovery)
                and self._assessment_pre_dispatch_failure is not None
            ):
                pre_dispatch = self._assessment_pre_dispatch_failure(version)
            retained = self._retain_pre_dispatch_hold(
                revision_id, candidate_version_id, version, pre_dispatch,
            )
            if retained is not None:
                return retained
            return NativePublicationContinuationResult(
                "EVIDENCE_HOLD" if admission_recovery else "ASSESSMENT_INTERRUPTED",
                str(facts.get("reason", "ACQUISITION_RESULT_NOT_RETAINED")),
                None,
            )

        if "intake_receipt_id" not in facts:
            request_id = facts.get("intake_request_id")
            received = facts.get("intake_received_epoch_seconds")
            if request_id is None or received is None:
                received = int(self._clock().value.timestamp())
                request_id = f"native-intake:{candidate_version_id}"
                facts = current_facts()
                facts.update(
                    intake_request_id=request_id,
                    intake_received_epoch_seconds=received,
                )
                self._journal.advance(
                    revision_id, stage="INTAKE_REQUESTED", facts=facts
                )
            acknowledgement = self._runtime.authority.receive_evidence_intake(
                self._runtime.ingress,
                candidate_version_id=candidate_version_id,
                expected_governing_manifest_digest=(
                    version.governing_manifest.canonical_digest
                ),
                boundary_id=NON_PUBLIC_EVIDENCE_INTAKE_BOUNDARY,
                request_id=str(request_id),
                received_epoch_seconds=int(received),
            )
            facts = current_facts()
            facts["intake_receipt_id"] = acknowledgement.receipt_id
            self._journal.advance(
                revision_id, stage="INTAKE_ACKNOWLEDGED", facts=facts
            )

        decision_value = facts.get("editorial_decision")
        package_id = facts.get("package_admission_id")
        if (
            decision_value is not None
            and package_id is not None
            and progress.get("stage") == "EVIDENCE_HOLD"
            and facts.get("reason") not in _REFRESHABLE_EVIDENCE_HOLDS
            and not write_admission_revalidation_due(facts)
        ):
            return NativePublicationContinuationResult(
                "EVIDENCE_HOLD", str(facts.get("reason")), None
            )
        if decision_value is None or package_id is None:
            if progress.get("stage") in {
                "ASSESSMENT_STARTED",
                "ASSESSMENT_INTERRUPTED",
            }:
                facts = current_facts()
                facts["reason"] = "ACQUISITION_RESULT_NOT_RETAINED"
                self._journal.advance(
                    revision_id, stage="ASSESSMENT_INTERRUPTED", facts=facts
                )
                return NativePublicationContinuationResult(
                    "ASSESSMENT_INTERRUPTED", facts["reason"], None
                )
            if progress.get("stage") == "EVIDENCE_HOLD":
                reason = facts.get("reason")
                retryable_acquisition = (
                    facts.get("acquisition_retryable") is True
                    and facts.get("acquisition_attempt_count", 0)
                    < _MAX_ACQUISITION_ATTEMPTS
                )
                if not retryable_acquisition and reason not in _REFRESHABLE_EVIDENCE_HOLDS:
                    return NativePublicationContinuationResult(
                        "EVIDENCE_HOLD", str(reason), None
                    )
            attempt_count = facts.get("acquisition_attempt_count", 0)
            if type(attempt_count) is not int or attempt_count < 0:
                raise NativePublicationError("native acquisition attempt differs")
            attempt_count += 1
            acquisition_started_at = self._clock().to_text()
            facts = current_facts()
            facts.update(
                acquisition_attempt_count=attempt_count,
                acquisition_started_at=acquisition_started_at,
            )
            facts.pop("acquisition_retryable", None)
            self._journal.advance(
                revision_id, stage="ACQUISITION_STARTED", facts=facts
            )
            assessment_started = False

            def before_assessment() -> None:
                nonlocal assessment_started, facts
                facts = current_facts()
                assessment_started = True
                facts["assessment_started_at"] = acquisition_started_at
                if self._assessment_contract_version is not None:
                    facts["assessment_contract_version"] = self._assessment_contract_version
                self._journal.advance(
                    revision_id, stage="ASSESSMENT_STARTED", facts=facts
                )

            def retain_acquisition_failure(
                failure_class: str,
            ) -> NativePublicationContinuationResult:
                nonlocal facts
                facts = current_facts()
                retryable = attempt_count < _MAX_ACQUISITION_ATTEMPTS
                facts.update(
                    reason=(
                        "ACQUISITION_TRANSPORT_RETRY"
                        if retryable
                        else "ACQUISITION_TRANSPORT_RETRY_EXHAUSTED"
                    ),
                    failure_class=failure_class,
                    acquisition_retryable=retryable,
                )
                self._journal.advance(
                    revision_id, stage="EVIDENCE_HOLD", facts=facts
                )
                return NativePublicationContinuationResult(
                    "EVIDENCE_HOLD", str(facts["reason"]), None
                )

            try:
                superseded = facts.get("assessment_superseded", {})
                prior_contract = (
                    superseded.get("contract_version")
                    if isinstance(superseded, dict)
                    else None
                )
                consumer_only_revalidation = same_assessment_producer(
                    prior_contract, self._assessment_contract_version,
                )
                if revision_id not in self._sources and self._evidence_sources_for is not None:
                    selected_sources = self._evidence_sources_for(revision_id)
                    if (type(selected_sources) is not tuple or not selected_sources
                            or not all(type(item) is NativeEvidenceSource for item in selected_sources)):
                        raise NativePublicationError("native selected evidence sources differ")
                    self._sources[revision_id] = selected_sources
                evidence = self._evidence.acquire_and_retain(
                    candidate_version_id=candidate_version_id,
                    intake_receipt_id=str(facts["intake_receipt_id"]),
                    sources=self._sources[revision_id],
                    before_assessment=before_assessment,
                    assessment_cached_only=consumer_only_revalidation,
                    proof=self._runtime.proof,
                )
            except VetoError:
                raise
            except NativeEvidenceHold as exc:
                if (
                    not assessment_started
                    and exc.reason_code in _RETRYABLE_ACQUISITION_HOLDS
                ):
                    return retain_acquisition_failure(type(exc).__name__)
                facts = current_facts()
                facts["reason"] = exc.reason_code
                facts["acquisition_retryable"] = False
                self._journal.advance(
                    revision_id, stage="EVIDENCE_HOLD", facts=facts
                )
                return NativePublicationContinuationResult(
                    "EVIDENCE_HOLD", exc.reason_code, None
                )
            except OSError as exc:
                if assessment_started:
                    facts = current_facts()
                    facts["reason"] = "ACQUISITION_RESULT_NOT_RETAINED"
                    facts["failure_class"] = type(exc).__name__
                    self._journal.advance(
                        revision_id, stage="ASSESSMENT_INTERRUPTED", facts=facts
                    )
                    return NativePublicationContinuationResult(
                        "ASSESSMENT_INTERRUPTED", facts["reason"], None
                    )
                return retain_acquisition_failure(type(exc).__name__)
            except Exception as exc:
                facts = current_facts()
                facts["reason"] = "ACQUISITION_RESULT_NOT_RETAINED"
                facts["failure_class"] = type(exc).__name__
                self._journal.advance(
                    revision_id,
                    stage=(
                        "ASSESSMENT_INTERRUPTED"
                        if assessment_started
                        else "EVIDENCE_HOLD"
                    ),
                    facts=facts,
                )
                return NativePublicationContinuationResult(
                    (
                        "ASSESSMENT_INTERRUPTED"
                        if assessment_started
                        else "EVIDENCE_HOLD"
                    ),
                    facts["reason"],
                    None,
                )
            decision = evidence.editorial_decision
            package_id = str(evidence.retained.package_admission_id)
            facts = current_facts()
            facts.update(
                package_admission_id=package_id,
                editorial_decision=json.loads(decision.canonical_bytes()),
                acquisition_receipt_digests=list(
                    evidence.acquisition_receipt_digests
                ),
            )
            self._journal.advance(
                revision_id, stage="EVIDENCE_RETAINED", facts=facts
            )
        else:
            decision = EditorialPolicyDecision.from_bytes(
                canonical_json_bytes(decision_value)
            )

        expected_keys = (
            "expected_story_version",
            "expected_publication_version",
            "expected_delivery_evidence_version",
        )
        retained_expected = tuple(key in facts for key in expected_keys)
        if any(retained_expected) and not all(retained_expected):
            raise NativePublicationError("native publication expected versions differ")
        if all(retained_expected) and (
            any(type(facts[key]) is not int or facts[key] < 0 for key in expected_keys)
            or facts["expected_delivery_evidence_version"] != 0
        ):
            raise NativePublicationError("native publication expected versions differ")
        if not all(retained_expected):
            story_version, publication_version = self._prior_acknowledged_versions(
                revision_id=revision_id, candidate_id=candidate_id
            )
            facts = current_facts()
            facts.update(
                expected_story_version=story_version,
                expected_publication_version=publication_version,
                # Each serving attempt owns a distinct evidence aggregate.
                expected_delivery_evidence_version=0,
            )
            self._journal.advance(
                revision_id, stage="PUBLICATION_PREPARED", facts=facts
            )
        else:
            siblings = tuple(
                dict(summary.get("facts", {}))
                for other_revision_id, summary in self._journal.iter_summaries()
                if other_revision_id != revision_id
                and summary.get("stage") == "ACKNOWLEDGED"
                and summary.get("facts", {}).get("candidate_id") == candidate_id
            )
            if siblings:
                reconciled = self._runtime.publication.reconcile_stale_intent(
                    ObjectAdmissionId.parse(str(package_id)),
                    expected_story_version=facts["expected_story_version"],
                    expected_publication_version=facts["expected_publication_version"],
                    acknowledged=siblings, proof=self._runtime.proof,
                )
                if reconciled is not None:
                    prior, predecessor = reconciled
                    facts = current_facts()
                    facts.update(
                        expected_story_version=prior.story_receipt.aggregate_version,
                        expected_publication_version=prior.attempt_receipt.aggregate_version,
                        expected_delivery_evidence_version=0,
                        publication_predecessor=predecessor,
                    )
                    self._journal.advance(revision_id, stage="PUBLICATION_PREPARED", facts=facts)

        # Legacy progress named its intent times applied/observed. Keep those
        # historical facts readable, but never use them as effect timestamps.
        if not any(key in facts for key in ("publication_started_at", "publication_applied_at")):
            facts = current_facts()
            facts["publication_started_at"] = self._clock().to_text()
            self._journal.advance(
                revision_id, stage="PUBLICATION_STARTED", facts=facts
            )
        # Resume the exact retained package/decision; this is not reassessment.
        # Stamp the consumer even on another HOLD so unchanged failures do not loop.
        if write_admission_revalidation_due(facts):
            facts = current_facts()
            facts["write_admission_policy_version"] = WRITE_ADMISSION_POLICY_VERSION
            self._journal.advance(revision_id, stage="PUBLICATION_STARTED", facts=facts)
        try:
            published = self._runtime.publication.advance(
                ObjectAdmissionId.parse(str(package_id)),
                decision,
                expected_story_version=int(facts["expected_story_version"]),
                expected_publication_version=int(facts["expected_publication_version"]),
                expected_delivery_evidence_version=int(
                    facts["expected_delivery_evidence_version"]
                ),
                proof=self._runtime.proof,
                **({"reconciled_predecessor": facts["publication_predecessor"]}
                   if "publication_predecessor" in facts else {}),
            )
        except EditorialHold as exc:
            reason_codes = (
                tuple(exc.decision.stable_reason_codes)
                if exc.decision is not None
                else (str(exc),)
            )
            if (
                not reason_codes
                or any(type(reason) is not str or not reason for reason in reason_codes)
            ):
                raise NativePublicationError("editorial HOLD reasons differ") from exc
            facts = current_facts()
            facts.update(
                reason=(
                    reason_codes[0]
                    if len(reason_codes) == 1
                    else "EDITORIAL_ADMISSION_HOLD"
                ),
                editorial_hold_reason_codes=list(reason_codes),
                write_admission_policy_version=WRITE_ADMISSION_POLICY_VERSION,
                acquisition_retryable=False,
            )
            self._journal.advance(
                revision_id, stage="EVIDENCE_HOLD", facts=facts
            )
            return NativePublicationContinuationResult(
                "EVIDENCE_HOLD", str(facts["reason"]), None
            )
        bindings = self._runtime.policies.publication
        reader = open_private_serving_read_port(
            bindings.target_path,
            target_id=bindings.target_id,
            target_context_digest=bindings.target_context_digest,
            proof=published.read_proof,
        )
        try:
            acknowledged = reader.acknowledged_rows()
            if acknowledged is None or tuple(
                row.surface_kind for row in acknowledged.rows
            ) != ("ARTICLE", "FEED_CARD"):
                raise NativePublicationError("private delivery ACK is absent")
        finally:
            reader.close()
        facts = current_facts()
        facts.update(
            story_event_id=published.story_receipt.event_id,
            publication_event_id=published.publication_receipt.event_id,
            delivery_attempt_event_id=published.attempt_receipt.event_id,
            delivery_evidence_event_id=published.evidence_receipt.event_id,
        )
        if getattr(published, "writer_id", None):
            facts["writer_id"] = published.writer_id
        self._journal.advance(revision_id, stage="ACKNOWLEDGED", facts=facts)
        return NativePublicationContinuationResult("ACKNOWLEDGED", None, published)

    def _advance_copy_correction(self, revision_id, candidate_version_id, facts, progress, current_facts):
        """Append one authenticated v2-to-v3 correction; never replace an ACK."""
        predecessor = facts.get("copy_correction_of")
        prepared = (progress.get("stage") == "COPY_CORRECTION_PREPARED"
                    or facts.get("copy_correction_result") == "CORRECTED")
        target_writer = getattr(self._runtime.publication, "writer_contract_version", "newsroom.offline-exact-copy.v3")
        try:
            if predecessor is None:
                writer_id = self._runtime.publication.retained_writer_id(
                    facts["story_event_id"], proof=self._runtime.proof,
                )
                if writer_id == target_writer or (target_writer == "newsroom.offline-exact-copy.v3"
                    and writer_id == "newsroom.offline-exact-copy.v1"):
                    facts = current_facts()
                    facts.update(writer_id=writer_id, copy_correction_checked_version=target_writer)
                    self._journal.advance(revision_id, stage="ACKNOWLEDGED", facts=facts)
                    return NativePublicationContinuationResult("ACKNOWLEDGED", None, None)
                if writer_id not in ({"newsroom.offline-exact-copy.v1", "newsroom.offline-exact-copy.v2", "newsroom.offline-exact-copy.v3"}
                    if target_writer == "newsroom.native-story-writer.v1" else {"newsroom.offline-exact-copy.v2"}):
                    raise NativePublicationError("copy correction predecessor writer differs")
                predecessor = {key: facts[key] for key in (
                    "story_event_id", "publication_event_id", "delivery_attempt_event_id", "delivery_evidence_event_id",
                )}
                predecessor["progress_ordinal"] = progress["ordinal"]
            if revision_id not in self._sources:
                raise NativePublicationError("copy correction current source is unavailable")
            prior, story = self._runtime.publication.read_acknowledged(predecessor, proof=self._runtime.proof)
            package_id = ObjectAdmissionId.parse(facts["package_admission_id"])
            decision = EditorialPolicyDecision.from_bytes(canonical_json_bytes(facts["editorial_decision"]))
            if (story.candidate_version_id != candidate_version_id
                or story.package_admission_id != package_id
                or story.policy_decision_id != decision.decision_id):
                raise NativePublicationError("copy correction immutable evidence differs")
            expected = (prior.story_receipt.aggregate_version, prior.attempt_receipt.aggregate_version, 0)
            if not prepared:
                facts = current_facts()
                facts.update(
                    copy_correction_of=predecessor,
                    expected_story_version=expected[0], expected_publication_version=expected[1],
                    expected_delivery_evidence_version=0,
                    publication_started_at=self._clock().to_text(),
                )
                self._journal.advance(revision_id, stage="COPY_CORRECTION_PREPARED", facts=facts)
                prepared = True
            elif tuple(facts[key] for key in (
                "expected_story_version", "expected_publication_version", "expected_delivery_evidence_version",
            )) != expected:
                raise NativePublicationError("copy correction expected versions differ")
            published = self._runtime.publication.advance(
                package_id, decision, expected_story_version=expected[0],
                expected_publication_version=expected[1], expected_delivery_evidence_version=0,
                proof=self._runtime.proof, correction_of=prior,
            )
        except (OperatorDrainRequested, VetoError):
            raise
        except Exception as exc:
            facts = current_facts()
            facts["copy_correction_hold_reason"] = f"COPY_CORRECTION_HOLD:{type(exc).__name__}"
            facts["copy_correction_failure_detail"] = str(exc)[:240]
            stage = "COPY_CORRECTION_PREPARED" if prepared else "ACKNOWLEDGED"
            self._journal.advance(revision_id, stage=stage, facts=facts)
            return NativePublicationContinuationResult(stage, facts["copy_correction_hold_reason"], None)
        facts = current_facts()
        facts.pop("copy_correction_hold_reason", None)
        facts.pop("copy_correction_failure_detail", None)
        facts.update(
            story_event_id=published.story_receipt.event_id,
            publication_event_id=published.publication_receipt.event_id,
            delivery_attempt_event_id=published.attempt_receipt.event_id,
            delivery_evidence_event_id=published.evidence_receipt.event_id,
            writer_id=published.writer_id, copy_correction_result="CORRECTED",
            copy_correction_checked_version=target_writer,
        )
        self._journal.advance(revision_id, stage="ACKNOWLEDGED", facts=facts)
        return NativePublicationContinuationResult("ACKNOWLEDGED", None, published)

    def _prior_acknowledged_versions(
        self, *, revision_id: str, candidate_id: str
    ) -> tuple[int, int]:
        prior: list[tuple[int, int]] = []
        for other_revision_id, progress in self._journal.iter_summaries():
            if other_revision_id == revision_id or progress.get("stage") != "ACKNOWLEDGED":
                continue
            retained = progress.get("facts", {})
            if retained.get("candidate_id") != candidate_id:
                continue
            story = self._prior_event(
                retained.get("story_event_id"),
                command=STORY_COMMAND,
                event_type=STORY_EVENT,
                aggregate_type="story",
                aggregate_id=str(_aggregate("story", candidate_id)),
                definition_digest=(
                    self._runtime.policies.publication
                    .editorial_story_command_definition_digest
                ),
                reason="prior Story authority differs",
            )
            attempt = self._prior_event(
                retained.get("delivery_attempt_event_id"),
                command=ATTEMPT_COMMAND,
                event_type=ATTEMPT_EVENT,
                aggregate_type="publication",
                aggregate_id=str(_aggregate("publication", candidate_id)),
                definition_digest=(
                    self._runtime.policies.publication
                    .serving_attempt_command_definition_digest
                ),
                reason="prior publication authority differs",
            )
            prior.append((story.aggregate_version, attempt.aggregate_version))
        if not prior:
            return 0, 0
        return max(item[0] for item in prior), max(item[1] for item in prior)

    def _prior_event(
        self,
        event_id: object,
        *,
        command: str,
        event_type: str,
        aggregate_type: str,
        aggregate_id: str,
        definition_digest: str,
        reason: str,
    ):
        if type(event_id) is not str or not event_id:
            raise NativePublicationError(reason)
        try:
            provenance = self._runtime.authority.events.provenance(
                event_id, proof=self._runtime.proof
            )
        except Exception as exc:
            raise NativePublicationError(reason) from exc
        event = provenance.event
        if (
            provenance.command_definition.command_type != command
            or provenance.command_definition.definition_digest != definition_digest
            or event.command_definition_digest != definition_digest
            or event.event_type != event_type
            or event.aggregate_type != aggregate_type
            or event.aggregate_id != aggregate_id
            or type(event.aggregate_version) is not int
            or event.aggregate_version <= 0
            or (command == ATTEMPT_COMMAND and event.aggregate_version < 2)
        ):
            raise NativePublicationError(reason)
        return event


def _aggregate(kind: str, identity: str) -> AggregateId:
    digest = digest_bytes(canonical_json_bytes([kind, identity]))
    raw = bytearray.fromhex(
        digest.removeprefix("sha256:")[:32]
    )
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return AggregateId(UUID(bytes=bytes(raw)))
