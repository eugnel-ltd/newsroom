"""Durable preparation of real native passages before four-branch retrieval.

The Control Plane ledger stores continuation references, not retrieval truth.
Every reused document/context is checked again by the native authority facade.
An interrupted embedding dispatch is held, never silently billed a second time.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import replace

from newsroom.authority import AggregateId, AuthenticationProof, ObjectAdmissionId
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.entities.types import EntityResolutionProposalId
from newsroom.extraction.types import ExtractionProposalKind, ExtractionRunId
from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome
from newsroom.graphiti_adapter.identity import typed_id
from newsroom.increment5.fulltext_contracts import (
    FullTextContractError,
    FullTextLanguageMode,
)
from newsroom.increment5.fulltext_normalizer import BilingualSearchNormalizer
from newsroom.increment5.native_retrieval import (
    NativeDocumentReceipt, NativeDocumentRequest, NativeEmbeddingReference,
    NativeRetrievalContextReceipt, NativeRetrievalContextRequest,
    NativeRetrievalDocuments, NativeRetrievalHold, NativeRetrievalSubject,
)
from newsroom.increment6.work_items import RetrievalInputBinding
from newsroom.projection.mapping import canonical_governed_node_id
from newsroom.projection.ontology import ProjectionNodeType
from newsroom.sources import SourceRevisionId

from .corpus import CorpusIngestUnit
from .graphiti_operational_readiness import (
    GRAPHITI_EVALUATION_HYDRATION_POLICY, _evaluation_attempt_for_unit,
)
from .native_cycle import _uuid4_for
from .native_progress import NativeRevisionJournal

MAX_EMBEDDING_ATTEMPTS = 3


def compose_native_documents(*, objects, extraction, commands, events, projector,
                             policies, principal_id: str, authority_domain: str):
    """Bind the existing admitted policy definitions without fixture helpers."""
    document, vector, receipt = policies.retrieval_hydration
    return NativeRetrievalDocuments(
        objects=objects, extraction=extraction, commands=commands, events=events,
        projector=projector, reader_principal_id=principal_id,
        authority_domain=authority_domain, controller_principal_id=principal_id,
        passage_hydration_policy_digest=GRAPHITI_EVALUATION_HYDRATION_POLICY.contract_digest,
        vector_hydration_policy_digest=vector, receipt_hydration_policy_digest=receipt,
        document_hydration_policy_digest=document,
        document_admission_definition_digest=policies.retrieval_document_definition,
        command_definition_digest=policies.retrieval_command_definition,
        context_hydration_policy_digest=policies.retrieval_context_hydration,
        context_admission_definition_digest=policies.retrieval_context_definition,
        context_command_definition_digest=policies.retrieval_context_command_definition,
    )


class NativeRetrievalContinuation:
    """Prepare only new passages; retain the exact first completed context."""

    def __init__(
        self, *, system, documents: NativeRetrievalDocuments,
        journal: NativeRevisionJournal, connection: sqlite3.Connection,
        embedder, generation_id: str, port_for: Callable,
        rights_check: Callable[[CorpusIngestUnit], str | None],
    ) -> None:
        self._system, self._documents, self._journal = system, documents, journal
        self._connection, self._embedder = connection, embedder
        self._generation, self._port_for, self._rights = generation_id, port_for, rights_check

    def _facts(self, revision_id: str) -> dict:
        return dict(self._journal.progress.get(revision_id, {}).get("facts", {}))

    def _save(self, revision_id: str, stage: str, **updates) -> None:
        self._journal.advance(revision_id, stage=stage, facts={**self._facts(revision_id), **updates})

    def retrieve(self, lead, *, proof: AuthenticationProof) -> RetrievalInputBinding:
        revision_id = str(lead.request.revision_id)
        units = self._journal.units[revision_id]
        # The requested Lead itself is mandatory; only unrelated historical
        # subjects may be source-locally excluded below.
        for unit in units:
            self._rights(unit)
        headlines = {unit.headline for unit in units}
        if len(headlines) != 1:
            raise NativeRetrievalHold("NATIVE_FULLTEXT_QUERY_AMBIGUOUS")
        try:
            BilingualSearchNormalizer().normalize(
                surface_text=next(iter(headlines)),
                language_mode=FullTextLanguageMode.MIXED_EN_GB_ZH_HANT_HK,
                query_valid_time=lead.recorded_at,
            )
        except FullTextContractError as exc:
            raise NativeRetrievalHold("NATIVE_FULLTEXT_QUERY_BOUND_HOLD") from exc
        retained = self._facts(revision_id).get("retrieval_binding")
        if retained is None:
            self._prepare(units, proof=proof)
        subjects, rights_inventory = self._current_subjects()
        rights_inventory_digest = digest_canonical(rights_inventory)
        if retained is not None:
            binding = RetrievalInputBinding.from_value(retained)
            receipt = NativeRetrievalContextReceipt.from_bytes(binding.receipt_bytes)
            context = self._documents.read_context(receipt, proof=proof)
            if context.lead_id != str(lead.request.lead_id) or context.lead_digest != lead.canonical_digest:
                raise ValueError("retained native context belongs to another Lead")
            if context.generation_id != self._generation:
                raise ValueError("retained native context identity changed")
            if (
                context.rights_inventory_digest == rights_inventory_digest
                and self._facts(revision_id).get("retrieval_rights_inventory")
                == rights_inventory
            ):
                return binding

        document_inventory = self._authenticated_inventory(
            subjects, proof=proof,
        )
        port = self._port_for(
            subjects, document_inventory, rights_inventory_digest
        )
        binding = port.retrieve(lead, proof=proof)
        request = NativeRetrievalContextRequest.from_bytes(binding.request_bytes)
        receipt = NativeRetrievalContextReceipt.from_bytes(binding.receipt_bytes)
        context = self._documents.read_context(receipt, proof=proof)
        permitted_events = {
            item.document_receipt.event_id for item in subjects
        }
        selected_events = {
            NativeDocumentReceipt.from_projection(item).event_id
            for item in context.selected_documents
        }
        if (
            request.request_digest != binding.request_digest
            or receipt.request_digest != request.request_digest
            or context.request_digest != request.request_digest
            or request.rights_inventory_digest != rights_inventory_digest
            or receipt.rights_inventory_digest != rights_inventory_digest
            or context.rights_inventory_digest != rights_inventory_digest
            or context.lead_id != str(lead.request.lead_id)
            or context.lead_digest != lead.canonical_digest
            or not selected_events.issubset(permitted_events)
        ):
            raise ValueError("native retrieval result differs from current request")
        self._save(
            revision_id, "RETRIEVAL_COMPLETE",
            retrieval_binding=binding.canonical_value(),
            retrieval_rights_inventory=rights_inventory,
        )
        return binding

    def _current_subjects(self):
        subjects = []
        inventory = []
        # Historical documents are re-authorised independently. A held source
        # is removed from all real projection branches without blocking a new,
        # currently authorised revision from another source.
        for source_revision, source_units in self._journal.units.items():
            records = self._facts(source_revision).get("retrieval_documents", {})
            if not records:
                continue
            exclusions = dict(self._facts(source_revision).get("retrieval_exclusions", {}))
            for unit in source_units:
                record = records.get(unit.ingest_id)
                if record is None:
                    continue
                receipt = NativeDocumentReceipt.from_projection(record["receipt"])
                try:
                    current_rights_digest = self._rights(unit)
                except NativeRetrievalHold as exc:
                    exclusion = {
                        "state": "EXCLUDED", "reason": exc.reason,
                        "document_digest": receipt.document_digest,
                    }
                    exclusions[unit.ingest_id] = exclusion
                    inventory.append({
                        "revision_id": source_revision, "ingest_id": unit.ingest_id,
                        **exclusion,
                    })
                    continue
                exclusions.pop(unit.ingest_id, None)
                subjects.append(NativeRetrievalSubject(
                    source_revision, record["graph_root_id"], receipt,
                    unit.headline,
                ))
                inventory.append({
                    "revision_id": source_revision, "ingest_id": unit.ingest_id,
                    "state": "INCLUDED", "reason": None,
                    "document_digest": receipt.document_digest,
                    "current_rights_digest": current_rights_digest,
                })
            facts = self._facts(source_revision)
            prior = facts.get("retrieval_exclusions", {})
            if exclusions != prior:
                self._journal.advance(
                    source_revision,
                    stage=self._journal.progress[source_revision]["stage"],
                    facts={**facts, "retrieval_exclusions": exclusions},
                )
        return tuple(subjects), sorted(
            inventory, key=lambda item: (item["revision_id"], item["ingest_id"]),
        )

    def _authenticated_inventory(
        self,
        subjects: tuple[NativeRetrievalSubject, ...],
        *,
        proof: AuthenticationProof,
    ):
        document_inventory = self._documents.authenticated_document_inventory(
            tuple(item.document_receipt for item in subjects), proof=proof,
        )
        retained_by_event = self._documents.require_authenticated_inventory(
            document_inventory,
            tuple(item.document_receipt for item in subjects),
        )
        if any(
            retained_by_event[item.document_receipt.event_id].revision_id
            != item.revision_id
            or retained_by_event[item.document_receipt.event_id].generation_id
            != self._generation
            for item in subjects
        ):
            raise ValueError("native passage continuation identity changed")
        return document_inventory

    def _prepare(self, units: tuple[CorpusIngestUnit, ...], *, proof: AuthenticationProof) -> None:
        revision_id = units[0].revision_id
        for unit in units:
            facts = self._facts(revision_id)
            documents = dict(facts.get("retrieval_documents", {}))
            if unit.ingest_id in documents:
                self._documents.require_document(
                    NativeDocumentReceipt.from_projection(documents[unit.ingest_id]["receipt"]),
                    proof=proof,
                )
                continue
            self._rights(unit)
            row = self._connection.execute(
                "SELECT i.outcome,i.receipt_digest,r.receipt_json "
                "FROM unpublished_graphiti_ingest i JOIN unpublished_graphiti_receipts r "
                "ON r.ingest_id=i.ingest_id WHERE i.ingest_id=?", (unit.ingest_id,),
            ).fetchone()
            if row is None or row[0] != "COMPLETE":
                raise NativeRetrievalHold("NATIVE_EXTRACTION_RECEIPT_MISSING")
            terminal = json.loads(row[2])
            unsigned = dict(terminal)
            receipt_digest = unsigned.pop("receipt_digest", None)
            if (receipt_digest != row[1] or receipt_digest != digest_bytes(canonical_json_bytes(unsigned))
                    or terminal.get("ingest_id") != unit.ingest_id):
                raise NativeRetrievalHold("NATIVE_EXTRACTION_RECEIPT_DIFFERS")
            attempt_number = terminal.get("attempt_number")
            if type(attempt_number) is not int or attempt_number < 1:
                raise ValueError("native extraction attempt number differs")
            retained_attempts = self._system.graphiti.attempt_history(
                typed_id(ExtractionRunId, "run", unit.ingest_id),
                limit=1,
                proof=proof,
            )
            if len(retained_attempts) != 1:
                raise NativeRetrievalHold("NATIVE_EXTRACTION_RECEIPT_DIFFERS")
            retained_attempt = retained_attempts[0]
            attempt = _evaluation_attempt_for_unit(
                replace(unit, attempt_number=attempt_number),
                recovered_ambiguous_progression=(
                    retained_attempt.recovered_ambiguous_progression
                ),
            )
            request = attempt.extraction_request
            if (
                retained_attempt.attempt_number != attempt_number
                or retained_attempt.run_id != request.run_id
                or retained_attempt.run_version_id != request.run_version_id
                or retained_attempt.outcome is not GraphitiAdapterOutcome.COMPLETE
            ):
                raise NativeRetrievalHold("NATIVE_EXTRACTION_RECEIPT_DIFFERS")
            if len(request.input_binding.passages) != 1:
                raise NativeRetrievalHold("NATIVE_PASSAGE_PARTITION_DIFFERS")
            passage = request.input_binding.passages[0]
            text = " ".join(unit.episode_body.split())
            if digest_bytes(text.encode()) != passage.text_digest:
                raise ValueError("native embedding passage bytes differ")
            embeddings = dict(facts.get("retrieval_embeddings", {}))
            embedded = embeddings.get(unit.ingest_id)
            dispatch = embedded is None
            base_cycle = f"native-passage:{unit.ingest_id}"
            if embedded is not None and embedded.get("state") != "RETAINED":
                prior_cycle = embedded.get("cycle_id", base_cycle)
                attempt = embedded.get("attempt_number", 1)
                if (
                    type(prior_cycle) is not str or type(attempt) is not int
                    or attempt < 1 or attempt >= MAX_EMBEDDING_ATTEMPTS
                    or not self._embedder.retryable_settled_attempt(
                        text=text, passage_id=str(passage.passage_id),
                        cycle_id=prior_cycle,
                    )
                ):
                    raise NativeRetrievalHold("NATIVE_EMBEDDING_INTERRUPTED")
                attempt += 1
                embedded = {
                    "state": "STARTED", "passage_id": str(passage.passage_id),
                    "attempt_number": attempt,
                    "cycle_id": f"{base_cycle}:retry:{attempt}",
                }
                embeddings[unit.ingest_id] = embedded
                self._save(revision_id, "EMBEDDING_STARTED", retrieval_embeddings=embeddings)
                dispatch = True
            if embedded is None:
                embedded = {
                    "state": "STARTED", "passage_id": str(passage.passage_id),
                    "attempt_number": 1, "cycle_id": base_cycle,
                }
                embeddings[unit.ingest_id] = embedded
                self._save(revision_id, "EMBEDDING_STARTED", retrieval_embeddings=embeddings)
            if dispatch:
                # Exactly the admitted extraction passage, not a feed summary
                # or separately re-tokenised full document.
                reference = self._embedder.retain(
                    text=text, passage_id=str(passage.passage_id),
                    cycle_id=embedded["cycle_id"], proof=proof,
                )
                embedded = {"state": "RETAINED", "passage_id": str(passage.passage_id),
                            "vector_admission_id": str(reference.vector_admission_id),
                            "receipt_admission_id": str(reference.receipt_admission_id)}
                embeddings[unit.ingest_id] = embedded
                self._save(revision_id, "EMBEDDING_RETAINED", retrieval_embeddings=embeddings)
            elif embedded["passage_id"] != str(passage.passage_id):
                raise ValueError("native embedding continuation passage changed")
            reference = NativeEmbeddingReference(
                ObjectAdmissionId.parse(embedded["vector_admission_id"]),
                ObjectAdmissionId.parse(embedded["receipt_admission_id"]),
            )
            key = f"native-document:{self._generation}:{passage.passage_id}"
            receipt, _document = self._documents.admit(NativeDocumentRequest(
                request, passage.passage_id, unit.authority.item_id, self._generation,
                reference, AggregateId.parse(_uuid4_for(key)), 0, key,
            ), proof=proof)
            graph_root = self._graph_root(request, unit, proof=proof)
            documents[unit.ingest_id] = {"receipt": receipt.projection_value(), "graph_root_id": graph_root}
            self._save(revision_id, "DOCUMENT_RETAINED", retrieval_documents=documents)

    def _graph_root(self, request, unit, *, proof: AuthenticationProof) -> str:
        # Use actual retained entity-resolution authority for this extraction.
        # No entity name or vector hit is promoted into an admitted graph ID.
        for proposal in self._system.extraction.proposals(request.run_version_id, proof=proof):
            if proposal.kind not in {ExtractionProposalKind.ENTITY_MENTION, ExtractionProposalKind.ENTITY_EQUIVALENCE}:
                continue
            proposal_id = typed_id(EntityResolutionProposalId, "graphiti-v1-resolution", proposal.canonical_digest)
            try:
                resolved = self._system.entities.decision(proposal_id, proof=proof)
            except LookupError:
                continue
            if resolved is not None and resolved.accepted_entity_id is not None:
                return canonical_governed_node_id(
                    ProjectionNodeType.AUTHORITY_AGGREGATE, "canonical_entity_id",
                    str(resolved.accepted_entity_id),
                )
        # A valid zero-proposal extraction may have no graph node. Query the
        # genuine SourceRevision event identity and retain the real NO_MATCH;
        # never invent a positive node or skip the actual graph branch.
        revision = self._system.sources.revision(SourceRevisionId.parse(unit.revision_id), proof=proof)
        return canonical_governed_node_id(
            ProjectionNodeType.LEDGER_EVENT, "authority_event_id", str(revision.event_id),
        )
