from __future__ import annotations

import json
import uuid
from dataclasses import replace
from types import SimpleNamespace

import pytest

from newsroom.authority import AggregateId, ObjectAdmissionId, UtcTimestamp
from newsroom.authority.canonical import digest_bytes, digest_canonical
from newsroom.control_plane.cycle import _receipt
from newsroom.control_plane.corpus import MAX_EPISODE_BYTES
from newsroom.control_plane.graphiti import GraphitiCycleResult
from newsroom.control_plane.graphiti_operational_readiness import (
    _evaluation_attempt_for_unit,
)
from newsroom.control_plane.native_progress import NativeRevisionJournal
from newsroom.control_plane.native_retrieval import NativeRetrievalContinuation
from newsroom.control_plane.store import (
    connect,
    insert_graphiti_attempt_receipt,
    insert_graphiti_ingest,
)
from newsroom.increment5.native_retrieval import (
    NativeDocumentReceipt,
    NativeEmbeddingReference,
    NativeRetrievalContextReceipt,
    NativeRetrievalContextRequest,
    NativeRetrievalHold,
)
from newsroom.increment6.work_items import RetrievalInputBinding
from newsroom.graphiti_adapter import RecoveredAmbiguousProgressionProof
from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome
from newsroom.tests.discovery_3d_authority_helpers import proof
from newsroom.tests.test_native_collision import _native_binding
from newsroom.tests.test_native_graphiti import _native


GENERATION = "native-generation-v1"


def _lead(unit):
    return SimpleNamespace(
        request=SimpleNamespace(
            revision_id=unit.revision_id,
            lead_id=str(uuid.uuid4()),
        ),
        recorded_at=UtcTimestamp.parse(unit.observed_at),
        canonical_digest=digest_bytes(f"lead:{unit.revision_id}".encode()),
    )


def _retain_complete(connection, unit, *, attempt_number=1):
    result = GraphitiCycleResult(
        ingest_id=unit.ingest_id,
        source_id=unit.source_id,
        item_key=unit.item_key,
        outcome="COMPLETE",
        proposal_count=0,
        entity_count=0,
        relation_count=0,
        failure_code="NONE",
        temporal_basis=unit.temporal().basis,
        reference_time=unit.temporal().reference_time.to_text(),
        receipt_digest=digest_bytes(f"raw:{unit.ingest_id}".encode()),
        attempt_number=attempt_number,
        provider_attempt_number=attempt_number,
    )
    terminal = _receipt(
        replace(unit, attempt_number=attempt_number), result, accounting={}
    )
    terminal["dispatch_rights"] = {"scope": "retained-test-rights"}
    final_digest = insert_graphiti_attempt_receipt(
        connection,
        ingest_id=unit.ingest_id,
        attempt_number=attempt_number,
        outcome="COMPLETE",
        receipt=terminal,
    )
    terminal["receipt_digest"] = final_digest
    insert_graphiti_ingest(
        connection,
        ingest_id=unit.ingest_id,
        source_id=unit.source_id,
        item_key=unit.item_key,
        outcome="COMPLETE",
        proposal_count=0,
        entity_count=0,
        relation_count=0,
        failure_code="NONE",
        temporal_basis=unit.temporal().basis.value,
        reference_time=unit.temporal().reference_time.to_text(),
        generation_id=result.generation_id,
        receipt_digest=final_digest,
        receipt=terminal,
    )
    connection.commit()
    return final_digest


class _Embedder:
    def __init__(self, *, retryable=False):
        self.calls = []
        self.retryable = retryable

    def retryable_settled_attempt(self, **_arguments):
        return self.retryable

    def retain(self, **arguments):
        self.calls.append(arguments)
        return NativeEmbeddingReference(
            ObjectAdmissionId.new(), ObjectAdmissionId.new()
        )


class _Documents:
    def __init__(self):
        self.documents = {}
        self.document_identities = {}
        self.contexts = {}
        self.admit_calls = []
        self.require_calls = []
        self.context_reads = []
        self.expected_extraction_request = None

    def admit(self, request, *, proof):
        if (
            self.expected_extraction_request is not None
            and request.extraction_request != self.expected_extraction_request
        ):
            raise KeyError(str(request.extraction_request.run_version_id))
        self.admit_calls.append(request)
        receipt = NativeDocumentReceipt(
            f"document-event-{len(self.admit_calls)}",
            f"document-command-{len(self.admit_calls)}",
            request.aggregate_id,
            1,
            ObjectAdmissionId.new(),
            digest_bytes(f"document:{request.passage_id}".encode()),
            request.embedding.vector_admission_id,
            request.embedding.receipt_admission_id,
        )
        document = SimpleNamespace(
            revision_id=str(request.extraction_request.input_binding.revision_id),
            generation_id=request.generation_id,
        )
        self.documents[str(receipt.aggregate_id)] = document
        self.document_identities[str(receipt.aggregate_id)] = (
            document.revision_id,
            document.generation_id,
        )
        return receipt, document

    def require_document(self, receipt, *, proof):
        self.require_calls.append(receipt)
        return self.documents[str(receipt.aggregate_id)]

    def authenticated_document_inventory(self, receipts, *, proof):
        return tuple(
            (receipt, self.require_document(receipt, proof=proof))
            for receipt in receipts
        )

    def require_authenticated_inventory(self, inventory, receipts):
        assert tuple(receipt for receipt, _document in inventory) == receipts
        return {
            receipt.event_id: document for receipt, document in inventory
        }

    def read_context(self, receipt, *, proof):
        self.context_reads.append(receipt)
        context = self.contexts[receipt.context_id]
        for value in context.selected_documents:
            selected = NativeDocumentReceipt.from_projection(value)
            identity = str(selected.aggregate_id)
            document = self.documents.get(identity)
            if document is None:
                continue
            if (
                document.revision_id,
                document.generation_id,
            ) != self.document_identities[identity]:
                raise ValueError("native document receipt differs")
        return context


def _continuation(
    connection,
    journal,
    documents,
    embedder,
    binding,
    context,
    subjects,
    *,
    interrupt_before_binding=False,
    rights_check=lambda _unit: None,
    rights_inventory_digests=None,
    stale_result=False,
    retained_attempts=None,
):
    class _Port:
        def retrieve(self, lead, *, proof):
            subjects.append(self.subjects)
            if interrupt_before_binding:
                raise RuntimeError("context retention interrupted")
            if stale_result:
                documents.contexts[binding.context_id] = context
                return binding
            request = replace(
                NativeRetrievalContextRequest.from_bytes(binding.request_bytes),
                rights_inventory_digest=self.rights_inventory_digest,
                selected_documents=tuple(
                    item.document_receipt for item in self.subjects
                ),
            )
            retained_context = replace(
                context,
                context_id=str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{context.context_id}:{self.rights_inventory_digest}",
                )),
                request_digest=request.request_digest,
                rights_inventory_digest=self.rights_inventory_digest,
                selected_documents=tuple(
                    item.document_receipt.projection_value()
                    for item in self.subjects
                ),
            )
            receipt = replace(
                NativeRetrievalContextReceipt.from_bytes(binding.receipt_bytes),
                context_id=retained_context.context_id,
                request_digest=request.request_digest,
                context_object_digest=retained_context.digest,
                rights_inventory_digest=self.rights_inventory_digest,
            )
            retained_binding = RetrievalInputBinding(
                binding.state,
                request.request_id,
                request.idempotency_key,
                request.request_digest,
                request.canonical_bytes,
                retained_context.context_id,
                receipt.receipt_digest,
                receipt.outcome,
                receipt.reason,
                receipt.no_match,
                receipt.canonical_bytes,
            )
            documents.contexts[retained_context.context_id] = retained_context
            return retained_binding

    def port_for(items, document_inventory, rights_inventory_digest):
        if rights_inventory_digests is not None:
            rights_inventory_digests.append(rights_inventory_digest)
        assert tuple(receipt for receipt, _document in document_inventory) == tuple(
            item.document_receipt for item in items
        )
        port = _Port()
        port.subjects = items
        port.rights_inventory_digest = rights_inventory_digest
        return port

    def attempt_history(run_id, *, limit, proof):
        retained = () if retained_attempts is None else retained_attempts.get(
            str(run_id), ()
        )
        if retained:
            return retained[:limit]
        for exact_units in journal.units.values():
            for unit in exact_units:
                candidate = _evaluation_attempt_for_unit(unit)
                if candidate.extraction_request.run_id == run_id:
                    row = connection.execute(
                        "SELECT receipt_json FROM unpublished_graphiti_receipts "
                        "WHERE ingest_id=?",
                        (unit.ingest_id,),
                    ).fetchone()
                    number = json.loads(row[0])["attempt_number"]
                    candidate = _evaluation_attempt_for_unit(
                        replace(unit, attempt_number=number)
                    )
                    return (SimpleNamespace(
                        attempt_number=number,
                        run_id=candidate.extraction_request.run_id,
                        run_version_id=candidate.extraction_request.run_version_id,
                        outcome=GraphitiAdapterOutcome.COMPLETE,
                        recovered_ambiguous_progression=None,
                    ),)
        return ()

    system = SimpleNamespace(
        graphiti=SimpleNamespace(attempt_history=attempt_history),
        extraction=SimpleNamespace(proposals=lambda *_args, **_kwargs: ()),
        entities=SimpleNamespace(),
        sources=SimpleNamespace(
            revision=lambda *_args, **_kwargs: SimpleNamespace(
                event_id="retained-source-revision-event"
            )
        ),
    )
    return NativeRetrievalContinuation(
        system=system,
        documents=documents,
        journal=journal,
        connection=connection,
        embedder=embedder,
        generation_id=context.generation_id,
        port_for=port_for,
        rights_check=rights_check,
    )


@pytest.mark.parametrize("mismatched_run_version", (False, True))
def test_recovered_attempt_uses_admitted_extraction_identity_without_redispatch(
    tmp_path, mismatched_run_version,
):
    unit = _native("recovered-retrieval")
    connection = connect(str(tmp_path / "private.sqlite3"))
    try:
        journal = NativeRevisionJournal(connection)
        journal.land((unit,))
        journal.advance(
            unit.revision_id,
            stage="GRAPHITI_COMPLETE",
            facts={"graphiti_receipts": [unit.ingest_id]},
        )
        _retain_complete(connection, unit, attempt_number=5)
        third = _evaluation_attempt_for_unit(replace(unit, attempt_number=3))
        instant = UtcTimestamp.parse("2026-09-20T12:00:00.000000Z")
        recovery = RecoveredAmbiguousProgressionProof(
            authoritative_attempt_id=third.attempt_id,
            authoritative_attempt_digest=digest_canonical({"attempt": 3}),
            authoritative_attempt_number=3,
            authoritative_run_version_id=third.extraction_request.run_version_id,
            authoritative_recorded_at=instant,
            skipped_attempt_number=4,
            skipped_receipt_digest=digest_canonical({"receipt": 4}),
            skipped_ledger_sequence=4,
            skipped_ledger_digest=digest_canonical({"ledger": 4}),
            skipped_recorded_at=instant,
            settled_usage_evidence_digest=digest_canonical({"usage": 3}),
            recovery_marker_digest=digest_canonical({"marker": 3}),
            marker_attempt_number=3,
            marker_workspace_id=third.workspace_id,
            marker_input_digest=digest_canonical({"marker-input": 3}),
            input_binding_digest=third.extraction_request.input_binding.digest,
            ingest_id=unit.ingest_id,
        )
        fifth = _evaluation_attempt_for_unit(
            replace(unit, attempt_number=5),
            recovered_ambiguous_progression=recovery,
        )
        passage = fifth.extraction_request.input_binding.passages[0]
        journal.advance(
            unit.revision_id,
            stage="EMBEDDING_RETAINED",
            facts={
                **journal.progress[unit.revision_id]["facts"],
                "retrieval_embeddings": {
                    unit.ingest_id: {
                        "state": "RETAINED",
                        "passage_id": str(passage.passage_id),
                        "vector_admission_id": str(ObjectAdmissionId.new()),
                        "receipt_admission_id": str(ObjectAdmissionId.new()),
                    }
                },
            },
        )
        retained = SimpleNamespace(
            attempt_number=5,
            run_id=fifth.extraction_request.run_id,
            run_version_id=(
                third.extraction_request.run_version_id
                if mismatched_run_version
                else fifth.extraction_request.run_version_id
            ),
            outcome=GraphitiAdapterOutcome.COMPLETE,
            recovered_ambiguous_progression=recovery,
        )
        lead = _lead(unit)
        binding, _receipt_value, context = _native_binding(tmp_path, lead)
        documents, embedder = _Documents(), _Embedder()
        documents.expected_extraction_request = fifth.extraction_request
        continuation = _continuation(
            connection,
            journal,
            documents,
            embedder,
            binding,
            context,
            [],
            retained_attempts={str(fifth.extraction_request.run_id): (retained,)},
        )

        if mismatched_run_version:
            with pytest.raises(
                NativeRetrievalHold,
                match="NATIVE_EXTRACTION_RECEIPT_DIFFERS",
            ):
                continuation.retrieve(lead, proof=proof())
            assert embedder.calls == []
            assert documents.admit_calls == []
            return

        assert continuation.retrieve(lead, proof=proof()).usable
        assert embedder.calls == []
        assert len(documents.admit_calls) == 1
        assert (
            documents.admit_calls[0].extraction_request
            == fifth.extraction_request
        )
        assert fifth.extraction_request.version_number == 4
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("query_case", "reason"),
    (
        ("bounded", "NATIVE_FULLTEXT_QUERY_BOUND_HOLD"),
        ("ambiguous", "NATIVE_FULLTEXT_QUERY_AMBIGUOUS"),
    ),
)
def test_fulltext_query_is_checked_before_embedding(
    tmp_path, query_case, reason,
):
    unit = _native("query-boundary")
    if query_case == "bounded":
        units = (replace(
            unit,
            headline=" ".join(f"term{index}" for index in range(65)),
        ),)
    else:
        units = (unit, replace(unit, headline="A different retained headline"))
    journal = SimpleNamespace(
        units={unit.revision_id: units},
        progress={},
    )
    embedder = _Embedder()
    continuation = NativeRetrievalContinuation(
        system=object(), documents=object(), journal=journal,
        connection=object(), embedder=embedder, generation_id=GENERATION,
        port_for=lambda *_arguments: pytest.fail("query hold opened retrieval"),
        rights_check=lambda _unit: None,
    )

    with pytest.raises(NativeRetrievalHold, match=reason):
        continuation.retrieve(_lead(unit), proof=proof())
    assert embedder.calls == []


def test_multi_chunk_embeddings_and_context_are_reused_across_restart(tmp_path):
    base = replace(
        _native(), body="Retained native passage. " * (MAX_EPISODE_BYTES // 25 + 1)
    )
    units = tuple(
        replace(base, chunk_ordinal=ordinal, chunk_count=2)
        for ordinal in (1, 2)
    )
    database = tmp_path / "private.sqlite3"
    connection = connect(str(database))
    journal = NativeRevisionJournal(connection)
    journal.land(units)
    journal.advance(
        base.revision_id,
        stage="GRAPHITI_COMPLETE",
        facts={"graphiti_receipts": [unit.ingest_id for unit in units]},
    )
    terminal_digests = {
        unit.ingest_id: _retain_complete(connection, unit, attempt_number=ordinal + 1)
        for ordinal, unit in enumerate(units)
    }
    lead = _lead(base)
    binding, _receipt_value, context = _native_binding(tmp_path, lead)
    documents, embedder, subjects = _Documents(), _Embedder(), []
    continuation = _continuation(
        connection,
        journal,
        documents,
        embedder,
        binding,
        context,
        subjects,
        interrupt_before_binding=True,
    )

    with pytest.raises(RuntimeError, match="context retention interrupted"):
        continuation.retrieve(lead, proof=proof())
    assert len(embedder.calls) == len(documents.admit_calls) == 2
    assert len(documents.require_calls) == 2
    assert [call["cycle_id"] for call in embedder.calls] == [
        f"native-passage:{unit.ingest_id}" for unit in units
    ]
    assert [call["passage_id"] for call in embedder.calls] == [
        str(
            _evaluation_attempt_for_unit(
                replace(unit, attempt_number=ordinal + 1)
            ).extraction_request.input_binding.passages[0].passage_id
        )
        for ordinal, unit in enumerate(units)
    ]
    assert len(subjects) == 1 and len(subjects[0]) == 2
    facts = journal.progress[base.revision_id]["facts"]
    assert facts["graphiti_receipts"] == [unit.ingest_id for unit in units]
    assert set(facts["retrieval_documents"]) == {
        unit.ingest_id for unit in units
    }
    assert {
        unit.ingest_id: facts["retrieval_embeddings"][unit.ingest_id]["state"]
        for unit in units
    } == {unit.ingest_id: "RETAINED" for unit in units}
    assert "retrieval_binding" not in facts
    for unit in units:
        row = connection.execute(
            "SELECT receipt_digest FROM unpublished_graphiti_ingest WHERE ingest_id=?",
            (unit.ingest_id,),
        ).fetchone()
        assert row[0] == terminal_digests[unit.ingest_id]

    connection.close()

    reopened = connect(str(database))
    try:
        replay = NativeRevisionJournal(reopened)
        replay_subjects = []
        replay_continuation = _continuation(
            reopened,
            replay,
            documents,
            embedder,
            binding,
            context,
            replay_subjects,
        )
        replayed_binding = replay_continuation.retrieve(lead, proof=proof())
        assert replayed_binding.usable
        assert len(embedder.calls) == len(documents.admit_calls) == 2
        assert len(replay_subjects) == 1 and len(replay_subjects[0]) == 2
        assert len(documents.context_reads) == 1
        replay_facts = replay.progress[base.revision_id]["facts"]
        replay.advance(
            base.revision_id,
            stage="CANDIDATE_ADMITTED",
            facts={**replay_facts, "candidate_version_id": "candidate-version-1"},
        )
    finally:
        reopened.close()

    final_connection = connect(str(database))
    try:
        final_journal = NativeRevisionJournal(final_connection)
        fresh_requests = []
        final_continuation = _continuation(
            final_connection,
            final_journal,
            documents,
            embedder,
            binding,
            context,
            fresh_requests,
        )
        full_inventory_reads = len(documents.require_calls)
        assert final_continuation.retrieve(lead, proof=proof()) == replayed_binding
        assert fresh_requests == []
        assert len(documents.require_calls) == full_inventory_reads
        assert len(documents.context_reads) == 2
        replay_facts = final_journal.progress[base.revision_id]["facts"]
        assert replay_facts["candidate_version_id"] == "candidate-version-1"
        assert replay_facts["graphiti_receipts"] == [
            unit.ingest_id for unit in units
        ]
        assert replay_facts["retrieval_binding"] == replayed_binding.canonical_value()
        retained_document = next(iter(documents.documents.values()))
        original_revision = retained_document.revision_id
        retained_document.revision_id = "different-retained-revision"
        with pytest.raises(ValueError, match="native document receipt differs"):
            final_continuation.retrieve(lead, proof=proof())
        retained_document.revision_id = original_revision
        retained_document.generation_id = "different-native-generation"
        with pytest.raises(ValueError, match="native document receipt differs"):
            final_continuation.retrieve(lead, proof=proof())
    finally:
        final_connection.close()


@pytest.mark.parametrize(("attempt_number", "retryable"), ((1, False), (3, True)))
def test_started_embedding_holds_without_redispatch(
    tmp_path, attempt_number, retryable,
):
    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    try:
        journal = NativeRevisionJournal(connection)
        journal.land((unit,))
        _retain_complete(connection, unit, attempt_number=3)
        passage = _evaluation_attempt_for_unit(
            replace(unit, attempt_number=3)
        ).extraction_request.input_binding.passages[0]
        journal.advance(
            unit.revision_id,
            stage="EMBEDDING_STARTED",
            facts={
                "retrieval_embeddings": {
                    unit.ingest_id: {
                        "state": "STARTED",
                        "passage_id": str(passage.passage_id),
                        "attempt_number": attempt_number,
                        "cycle_id": f"native-passage:{unit.ingest_id}",
                    }
                }
            },
        )
        lead = _lead(unit)
        binding, _receipt_value, context = _native_binding(tmp_path, lead)
        documents, embedder = _Documents(), _Embedder(retryable=retryable)
        documents.contexts[binding.context_id] = context
        continuation = _continuation(
            connection, journal, documents, embedder, binding, context, []
        )

        with pytest.raises(
            NativeRetrievalHold, match="NATIVE_EMBEDDING_INTERRUPTED"
        ):
            continuation.retrieve(lead, proof=proof())
        assert embedder.calls == []
        assert documents.admit_calls == []
        assert journal.progress[unit.revision_id]["stage"] == "EMBEDDING_STARTED"
    finally:
        connection.close()


def test_exact_zero_dispatch_embedding_terminal_allows_one_new_attempt(tmp_path):
    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    try:
        journal = NativeRevisionJournal(connection)
        journal.land((unit,))
        _retain_complete(connection, unit, attempt_number=3)
        passage = _evaluation_attempt_for_unit(
            replace(unit, attempt_number=3)
        ).extraction_request.input_binding.passages[0]
        journal.advance(
            unit.revision_id, stage="EMBEDDING_STARTED", facts={
                "retrieval_embeddings": {unit.ingest_id: {
                    "state": "STARTED", "passage_id": str(passage.passage_id),
                    "attempt_number": 1,
                    "cycle_id": f"native-passage:{unit.ingest_id}",
                }},
            },
        )
        lead = _lead(unit)
        fixture = tmp_path / "binding"
        fixture.mkdir()
        binding, _receipt_value, context = _native_binding(fixture, lead)
        documents, embedder = _Documents(), _Embedder(retryable=False)
        continuation = _continuation(
            connection, journal, documents, embedder, binding, context, [],
        )

        with pytest.raises(
            NativeRetrievalHold, match="NATIVE_EMBEDDING_INTERRUPTED"
        ):
            continuation.retrieve(lead, proof=proof())
        assert embedder.calls == []
        assert journal.progress[unit.revision_id]["facts"][
            "retrieval_embeddings"
        ][unit.ingest_id]["attempt_number"] == 1

        embedder.retryable = True
        assert continuation.retrieve(lead, proof=proof()).usable
        assert [call["cycle_id"] for call in embedder.calls] == [
            f"native-passage:{unit.ingest_id}:retry:2"
        ]
        assert journal.progress[unit.revision_id]["facts"][
            "retrieval_embeddings"
        ][unit.ingest_id]["state"] == "RETAINED"
    finally:
        connection.close()


def test_historical_rights_hold_is_excluded_and_invalidates_context_replay(tmp_path):
    first, current = _native("historical-a"), _native("current-b")
    connection = connect(str(tmp_path / "private.sqlite3"))
    try:
        journal = NativeRevisionJournal(connection)
        documents, embedder = _Documents(), _Embedder()
        for unit in (first, current):
            journal.land((unit,))
            journal.advance(
                unit.revision_id, stage="GRAPHITI_COMPLETE",
                facts={"graphiti_receipts": [unit.ingest_id]},
            )
            _retain_complete(connection, unit)
        first_root, current_root = tmp_path / "first", tmp_path / "current"
        first_root.mkdir()
        current_root.mkdir()
        first_lead, current_lead = _lead(first), _lead(current)
        first_binding, _, first_context = _native_binding(first_root, first_lead)
        current_binding, _, current_context = _native_binding(current_root, current_lead)
        _continuation(
            connection, journal, documents, embedder,
            first_binding, first_context, [],
        ).retrieve(first_lead, proof=proof())

        held = {first.ingest_id}
        rights_digests = {
            first.ingest_id: "rights-a-v1", current.ingest_id: "rights-b-v1",
        }

        def rights_check(unit):
            if unit.ingest_id in held:
                raise NativeRetrievalHold("CURRENT_RIGHTS_HOLD")
            return rights_digests[unit.ingest_id]

        with pytest.raises(
            ValueError, match="result differs from current request"
        ):
            _continuation(
                connection, journal, documents, embedder,
                current_binding, current_context, [],
                rights_check=rights_check,
                stale_result=True,
            ).retrieve(current_lead, proof=proof())

        subjects = []
        inventory_digests = []
        continuation = _continuation(
            connection, journal, documents, embedder,
            current_binding, current_context, subjects,
            rights_check=rights_check,
            rights_inventory_digests=inventory_digests,
        )
        assert continuation.retrieve(current_lead, proof=proof()).usable
        assert [[item.revision_id for item in group] for group in subjects] == [
            [current.revision_id]
        ]
        assert len(inventory_digests) == 1
        exclusion = journal.progress[first.revision_id]["facts"][
            "retrieval_exclusions"
        ][first.ingest_id]
        first_receipt = next(iter(journal.progress[first.revision_id]["facts"][
            "retrieval_documents"
        ].values()))["receipt"]
        assert exclusion == {
            "state": "EXCLUDED", "reason": "CURRENT_RIGHTS_HOLD",
            "document_digest": first_receipt["document_digest"],
        }

        subjects.clear()
        full_inventory_reads = len(documents.require_calls)
        assert continuation.retrieve(current_lead, proof=proof()).usable
        assert subjects == []
        assert len(documents.require_calls) == full_inventory_reads

        held.clear()
        subjects.clear()
        assert continuation.retrieve(current_lead, proof=proof()).usable
        assert [[item.revision_id for item in group] for group in subjects] == [[
            first.revision_id, current.revision_id,
        ]]
        assert len(documents.require_calls) == full_inventory_reads + 2
        assert len(set(inventory_digests)) == 2
        assert journal.progress[first.revision_id]["facts"]["retrieval_exclusions"] == {}

        rights_digests[current.ingest_id] = "rights-b-v2"
        subjects.clear()
        continuation.retrieve(current_lead, proof=proof())
        assert [[item.revision_id for item in group] for group in subjects] == [[
            first.revision_id, current.revision_id,
        ]]
        assert len(set(inventory_digests)) == 3

        held.add(first.ingest_id)
        subjects.clear()
        continuation.retrieve(current_lead, proof=proof())
        assert [[item.revision_id for item in group] for group in subjects] == [
            [current.revision_id]
        ]
        assert len(set(inventory_digests)) == 4
        held.add(current.ingest_id)
        subjects.clear()
        with pytest.raises(NativeRetrievalHold, match="CURRENT_RIGHTS_HOLD"):
            continuation.retrieve(current_lead, proof=proof())
        assert subjects == []
    finally:
        connection.close()


def test_terminal_receipt_tamper_holds_before_embedding(tmp_path):
    unit = _native()
    connection = connect(str(tmp_path / "private.sqlite3"))
    try:
        journal = NativeRevisionJournal(connection)
        journal.land((unit,))
        retained_digest = _retain_complete(connection, unit, attempt_number=4)
        raw = connection.execute(
            "SELECT receipt_json FROM unpublished_graphiti_receipts WHERE ingest_id=?",
            (unit.ingest_id,),
        ).fetchone()[0]
        terminal = json.loads(raw)
        assert terminal["receipt_digest"] == retained_digest
        terminal["attempt_number"] = 5
        connection.execute(
            "UPDATE unpublished_graphiti_receipts SET receipt_json=? WHERE ingest_id=?",
            (json.dumps(terminal, ensure_ascii=False, sort_keys=True), unit.ingest_id),
        )
        connection.commit()
        lead = _lead(unit)
        binding, _receipt_value, context = _native_binding(tmp_path, lead)
        documents, embedder = _Documents(), _Embedder()
        continuation = _continuation(
            connection, journal, documents, embedder, binding, context, []
        )

        with pytest.raises(
            NativeRetrievalHold, match="NATIVE_EXTRACTION_RECEIPT_DIFFERS"
        ):
            continuation.retrieve(lead, proof=proof())
        assert embedder.calls == []
        assert documents.admit_calls == []
    finally:
        connection.close()
