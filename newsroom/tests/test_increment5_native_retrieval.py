from __future__ import annotations

import struct
import uuid
from dataclasses import replace

import pytest

from newsroom.authority.canonical import digest_bytes
from newsroom.authority.neo4j_projection_system import (
    open_native_retrieval_neo4j_resources,
)
from newsroom.authority.types import AggregateId, ObjectAdmissionId
from newsroom.authority.types import TrustScope, UtcTimestamp
from newsroom.projection.models import ProjectionGenerationId, ProjectionGenerationState
from newsroom.projection.neo4j.models import (
    NEO4J_B2_DRIVER_VERSION,
    StructuralGraphNodeView, StructuralReadAuthoritySelection,
    StructuralReadMetadata, StructuralReadResponse,
)
from newsroom.projection.ontology import ProjectionNodeType
from newsroom.increment5.native_retrieval import (
    NATIVE_RESULT_LIMIT,
    NATIVE_VECTOR_DIMENSIONS,
    NativeEmbeddingReceipt,
    NativeDocumentReceipt,
    NativePassageDocument,
    NativeRetrievalDocuments,
    NativeRetrievalContext,
    NativeRetrievalContextReceipt,
    NativeRetrievalContextRequest,
    NativeGraphBranchReceipt,
    NativeRetrievalError,
    NativeRetrievalHold,
    NativeVectorRequest,
    NativeVectorBranchHit,
    NativeVectorBranchReceipt,
    _vector,
)
from newsroom.increment5.branch_contracts import BranchMode, BranchOutcome
from newsroom.increment5.neo4j_native_retrieval import Neo4jNativeRetrievalProjection


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _document() -> NativePassageDocument:
    text = "Hong Kong authority retained this exact passage."
    return NativePassageDocument(
        generation_id="native-generation-1",
        passage_id="00000000-0000-4000-8000-000000000001",
        dependency_root_id="event:one",
        source_id="source.one",
        revision_id="revision.one",
        representation_id="representation.one",
        language="en-GB",
        text=text,
        text_digest=digest_bytes(text.encode()),
        rights_digest=_digest("a"),
        provenance_digest=_digest("b"),
        vector_digest=_digest("c"),
        vector_admission_id="00000000-0000-4000-8000-000000000002",
        embedding_receipt_digest=_digest("d"),
        embedding_receipt_admission_id="00000000-0000-4000-8000-000000000003",
        embedding_model_digest=_digest("e"),
    )


def _receipt() -> NativeDocumentReceipt:
    return NativeDocumentReceipt(
        "event-one",
        "command-one",
        AggregateId.new(),
        1,
        ObjectAdmissionId.new(),
        _document().digest,
        ObjectAdmissionId.new(),
        ObjectAdmissionId.new(),
    )


class _Result(tuple):
    def consume(self):
        return None
    def single(self):
        return self[0] if len(self) == 1 else None


class _Transaction:
    def __init__(self, calls):
        self.calls = calls

    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        if query.startswith("MATCH") and "n.event_id AS event_id" in query:
            return _Result(self.receipt if isinstance(self.receipt, tuple) else (self.receipt,))
        if "RETURN properties(n) AS properties" in query:
            return _Result(({"properties": parameters},))
        if "fulltext.queryNodes" in query:
            return _Result(({**self.receipt, "score": 2.0},))
        if "vector.queryNodes" in query:
            receipts = self.receipt if isinstance(self.receipt, tuple) else (self.receipt,)
            return _Result(tuple({**item, "score": 0.75} for item in receipts))
        return _Result()


class _Session:
    def __init__(self, calls, receipt): self.calls, self.receipt = calls, receipt
    def __enter__(self): return self
    def __exit__(self, *_): return None
    def execute_write(self, work):
        transaction = _Transaction(self.calls)
        transaction.receipt = self.receipt
        return work(transaction)
    def execute_read(self, work):
        transaction = _Transaction(self.calls)
        transaction.receipt = self.receipt
        return work(transaction)


class _Driver:
    def __init__(self, receipt): self.calls = []; self.sessions = []; self.receipt = receipt
    def session(self, **config):
        self.sessions.append(config)
        return _Session(self.calls, self.receipt)


class _MetadataTransaction:
    def __init__(self, label): self.label = label
    def run(self, query, **parameters):
        if "dbms.components" in query:
            return _Result(({"version": "2026.06.0", "edition": "community"},))
        if query.startswith("SHOW INDEXES"):
            return _Result(({"state": "ONLINE", "type": "FULLTEXT", "entityType": "NODE",
                             "labelsOrTypes": [self.label],
                             "properties": ["authority_aliases", "formal_tokens", "han_bigrams", "latin_terms", "retrieval_text"],
                             "indexProvider": "fulltext-2.0",
                             "options": {"indexConfig": {"fulltext.analyzer": "standard-no-stop-words", "fulltext.eventually_consistent": False}}},))
        return _Result(({"count": 1},))


class _MetadataSession:
    def __init__(self, driver): self.driver = driver
    def __enter__(self): return self
    def __exit__(self, *_): return None
    def execute_read(self, work): return work(_MetadataTransaction(self.driver.label))


class _MetadataDriver:
    label = ""
    def session(self, **_config): return _MetadataSession(self)


class _ResourceAdapter:
    driver_version = NEO4J_B2_DRIVER_VERSION

    def __init__(self) -> None:
        self.closed = 0
        self.arguments = None
        self.projector = object()

    def open_native_retrieval_projection(self, **arguments):
        self.arguments = arguments
        return self.projector

    def close(self) -> None:
        self.closed += 1


def test_native_projection_executes_real_fulltext_and_vector_queries() -> None:
    receipt = _receipt()
    driver = _Driver(receipt.projection_value())
    projection = Neo4jNativeRetrievalProjection(
        driver,
        database="neo4j",
        generation_id="native-generation-1",
        fulltext_index="native_fulltext_1",
        vector_index="native_vector_1",
        driver_version=NEO4J_B2_DRIVER_VERSION,
    )
    vector = (1.0,) + (0.0,) * (NATIVE_VECTOR_DIMENSIONS - 1)
    document = _document()

    projection.bootstrap()
    projection.upsert(receipt, document, vector)
    assert projection.reconcile_membership((receipt,)) == ()
    missing = _receipt()
    assert projection.reconcile_membership((missing,)) == (missing,)
    fulltext, vector_hits = projection.retrieve(
        query_text="Hong Kong",
        query_vector=vector,
    )
    vector_only = projection.retrieve_vector(query_vector=vector)

    assert fulltext == ({**receipt.projection_value(), "score": 2.0},)
    assert vector_hits == ({**receipt.projection_value(), "score": 0.75},)
    assert vector_only == vector_hits
    queries = "\n".join(item[0] for item in driver.calls)
    assert "CREATE FULLTEXT INDEX" in queries
    assert "n.authority_aliases,n.formal_tokens,n.han_bigrams,n.latin_terms,n.retrieval_text" in queries
    assert "CREATE VECTOR INDEX" in queries
    assert "db.index.fulltext.queryNodes" in queries
    assert "db.index.vector.queryNodes" in queries
    assert all(
        parameters["limit"] == NATIVE_RESULT_LIMIT
        for query, parameters in driver.calls if "vector.queryNodes" in query
    )
    assert "WHERE NOT n.aggregate_id IN $aggregate_ids DELETE n" in queries

    corrupt = dict(receipt.projection_value())
    corrupt["document_digest"] = _digest("f")
    with pytest.raises(NativeRetrievalError, match="retained document differs"):
        Neo4jNativeRetrievalProjection(
            _Driver(corrupt), database="neo4j",
            generation_id="native-generation-1",
            fulltext_index="native_fulltext_1", vector_index="native_vector_1",
            driver_version=NEO4J_B2_DRIVER_VERSION,
        ).reconcile_membership((receipt,))
    assert "NewsroomNativeRetrievalDocument_" in queries
    assert any(parameters.get("generation_id") == "native-generation-1" for _, parameters in driver.calls)
    assert {item["default_access_mode"] for item in driver.sessions} == {"READ", "WRITE"}


def test_native_projection_indexes_complete_passage_vocabulary() -> None:
    original = _document()
    text = " ".join(f"governedterm{index}" for index in range(65))
    document = replace(original, text=text, text_digest=digest_bytes(text.encode()))
    receipt = replace(_receipt(), document_digest=document.digest)
    driver = _Driver(receipt.projection_value())
    projection = Neo4jNativeRetrievalProjection(
        driver, database="neo4j", generation_id="native-generation-1",
        fulltext_index="native_fulltext_1", vector_index="native_vector_1",
        driver_version=NEO4J_B2_DRIVER_VERSION,
    )

    projection.upsert(
        receipt, document, (1.0,) + (0.0,) * (NATIVE_VECTOR_DIMENSIONS - 1)
    )

    parameters = next(values for query, values in driver.calls if "MERGE (n:" in query)
    assert len(parameters["latin_terms"]) == 65


def test_native_vector_projection_uses_exact_top_eight_and_rejects_overrun() -> None:
    receipts = tuple(
        replace(_receipt(), aggregate_id=AggregateId.new()).projection_value()
        for _ in range(8)
    )
    driver = _Driver(receipts)
    projection = Neo4jNativeRetrievalProjection(
        driver, database="neo4j", generation_id="native-generation-1",
        fulltext_index="native_fulltext_1", vector_index="native_vector_1",
        driver_version=NEO4J_B2_DRIVER_VERSION,
    )
    vector = (1.0,) + (0.0,) * (NATIVE_VECTOR_DIMENSIONS - 1)

    assert len(projection.retrieve_vector(query_vector=vector)) == 8
    query = next(values for query, values in driver.calls if "vector.queryNodes" in query)
    assert query["limit"] == 8
    with pytest.raises(NativeRetrievalHold, match="RESULT_LIMIT_EXCEEDED"):
        NativeRetrievalDocuments._documents(
            object.__new__(NativeRetrievalDocuments),
            tuple({**item, "score": 0.75} for item in (*receipts, receipts[0])),
            "native-generation-1",
            object(),
        )


def test_native_projection_snapshot_is_actual_native_metadata() -> None:
    driver = _MetadataDriver()
    generation = "00000000-0000-4000-8000-000000000010"
    projection = Neo4jNativeRetrievalProjection(
        driver, database="neo4j", generation_id=generation,
        fulltext_index="native_fulltext_10", vector_index="native_vector_10",
        driver_version=NEO4J_B2_DRIVER_VERSION,
    )
    driver.label = projection.document_label
    now = UtcTimestamp.parse("2026-09-08T12:00:00.000000Z")
    snapshot = projection.snapshot(
        generation_identity_digest=_digest("1"),
        rights_manifest_digest=_digest("2"), contiguous_ledger_seq=7,
        expected_document_count=1, clock=lambda: now,
    )

    assert snapshot.profile.value == "NATIVE_RUNTIME"
    assert snapshot.document_label == projection.document_label
    assert snapshot.index_name == projection.fulltext_index
    assert snapshot.index_document_count == 1


def test_native_projection_public_opener_shares_and_closes_private_adapter(
    monkeypatch,
) -> None:
    adapter = _ResourceAdapter()
    monkeypatch.setattr(
        "newsroom.authority._neo4j_projection_system._open_structural_graph_adapter",
        lambda _config: adapter,
    )

    resources = open_native_retrieval_neo4j_resources(
        config=object(),
        generation_id="native-generation-1",
        fulltext_index="native_fulltext_1",
        vector_index="native_vector_1",
    )

    assert resources.projector is adapter.projector
    assert resources.fulltext.driver_version == NEO4J_B2_DRIVER_VERSION
    assert adapter.arguments == {
        "generation_id": "native-generation-1",
        "fulltext_index": "native_fulltext_1",
        "vector_index": "native_vector_1",
    }
    resources.close()
    resources.close()
    assert adapter.closed == 1


def test_embedding_receipt_and_vector_are_exact_non_fixture_inputs() -> None:
    raw = struct.pack(f">{NATIVE_VECTOR_DIMENSIONS}f", 1.0, *((0.0,) * (NATIVE_VECTOR_DIMENSIONS - 1)))
    receipt = NativeEmbeddingReceipt(
        input_text_digest=_digest("1"),
        vector_digest=digest_bytes(raw),
        dimensions=NATIVE_VECTOR_DIMENSIONS,
        provider="openai",
        model="text-embedding-3-large",
        model_digest=_digest("2"),
        provider_request_id="provider-request-1",
        usage_receipt_digest=_digest("3"),
        recorded_at="2026-09-08T12:00:00.000000Z",
    )

    assert NativeEmbeddingReceipt.from_bytes(receipt.canonical_bytes) == receipt
    assert _vector(raw)[0] == 1.0
    with pytest.raises(NativeRetrievalHold, match="EMBEDDING_VECTOR_MISSING"):
        _vector(b"fixture-id-is-not-a-vector")


def test_projection_hits_without_admitted_document_are_rejected() -> None:
    with pytest.raises(NativeRetrievalError, match="projection receipt differs"):
        NativeDocumentReceipt.from_projection(
            {"passage_id": "private-workspace-node"}
        )


def test_native_vector_request_and_receipt_are_distinct_from_fixture_contract() -> None:
    event_id = "native-event-1"
    request = NativeVectorRequest(
        request_id=str(uuid.uuid4()),
        idempotency_key="native-vector-one",
        query_event_id=event_id,
        query_document_digest=_digest("3"),
        generation_id="native-generation-1",
        query_valid_time="2026-09-08T12:00:00Z",
        serving_time="2026-09-08T12:00:01Z",
    )
    hit = NativeVectorBranchHit(
        1, "passage-one", "root-one", "revision-one", _digest("4"),
        _digest("5"), _digest("6"), 750_000,
    )
    receipt = NativeVectorBranchReceipt(
        str(uuid.uuid4()), request.request_digest, BranchMode.VECTOR,
        BranchOutcome.COMPLETE, None, "native-generation-1", _digest("7"),
        "native-governed-vector-v1", request.query_valid_time,
        request.serving_time, (hit,), 2,
    )

    assert request.request_digest == digest_bytes(request.canonical_bytes)
    assert NativeVectorBranchReceipt.from_canonical_bytes(receipt.canonical_bytes) == receipt


def test_native_context_retains_four_real_branch_receipts_and_round_trips(tmp_path) -> None:
    from newsroom.tests.increment5b1_helpers import _request as exact_request, _system as exact_system
    from newsroom.tests.increment5b2_helpers import GENERATION_ID, default_scenario, request as fulltext_request, system as fulltext_system

    exact = exact_system(tmp_path)[2].retrieve(exact_request()).receipt
    fulltext = fulltext_system(tmp_path, scenario=default_scenario(rows=[]))[2].retrieve(fulltext_request()).receipt
    graph_request_digest = _digest("9")
    graph = NativeGraphBranchReceipt.from_response(
        graph_request_digest, ("npid:v1:source-revision:one",),
        StructuralReadResponse(
            StructuralReadMetadata(
                "increment4-admitted", "v1", "projector-v1", _digest("4"),
                _digest("5"), ProjectionGenerationId.new(),
                ProjectionGenerationState.ACTIVE,
                StructuralReadAuthoritySelection.AUTHORITY_SELECTED_ACTIVE,
                42, 0, 0, TrustScope.ADMITTED,
                UtcTimestamp.parse("2026-08-06T08:59:00Z"),
                UtcTimestamp.parse("2026-08-06T09:00:00Z"),
            ),
            (StructuralGraphNodeView(
                "npid:v1:source-revision:one", ProjectionNodeType.SOURCE_REVISION,
                "revision-one", _digest("6"), 1, "event-one", _digest("7"),
            ),), (),
        ),
    )
    vector_request = NativeVectorRequest(
        str(uuid.uuid4()), "vector-context", "event", _digest("1"),
        str(GENERATION_ID), "2026-08-06T08:59:00Z", "2026-08-06T09:00:00Z",
    )
    vector = NativeVectorBranchReceipt(
        str(uuid.uuid4()), vector_request.request_digest, BranchMode.VECTOR,
        BranchOutcome.COMPLETE, "NO_MATCH", str(GENERATION_ID), _digest("2"),
        "native-governed-vector-v1", vector_request.query_valid_time,
        vector_request.serving_time, (), 1,
    )
    request = NativeRetrievalContextRequest(
        str(uuid.uuid4()), "context-one", AggregateId.new(), 0, "lead-one",
        _digest("3"), "native-scope-one", _digest("4"), exact.canonical_bytes,
        fulltext.canonical_bytes, vector.canonical_bytes, graph.canonical_bytes,
        (_receipt(),),
    )
    assert NativeRetrievalContextRequest.from_bytes(request.canonical_bytes) == request
    raws = (request.exact_receipt_bytes, request.fulltext_receipt_bytes,
            request.vector_receipt_bytes, request.graph_receipt_bytes)
    context = NativeRetrievalContext(
        str(uuid.uuid4()), request.request_id, request.request_digest,
        request.lead_id, request.lead_digest, request.authority_scope_id,
        request.rights_inventory_digest,
        str(GENERATION_ID), vector.query_valid_time, vector.serving_time,
        tuple(digest_bytes(raw) for raw in raws),
        tuple(__import__("json").loads(raw) for raw in raws),
        (_receipt().projection_value(),), "COMPLETE", False,
    )
    assert NativeRetrievalContext.from_bytes(context.canonical_bytes) == context
    receipt = NativeRetrievalContextReceipt(
        context.context_id, request.request_id, request.request_digest,
        request.aggregate_id, 1, "event", "command", ObjectAdmissionId.new(),
        context.digest, request.authority_scope_id,
        request.rights_inventory_digest, str(GENERATION_ID),
        vector.query_valid_time, vector.serving_time, *raws,
        "newsroom.hermes", "newsroom.authority", no_match=False,
    )
    assert NativeRetrievalContextReceipt.from_bytes(receipt.canonical_bytes) == receipt


def test_native_projection_compact_corpus_has_no_4096_cliff_or_body_projection():
    receipts = tuple(replace(_receipt(), event_id=f"event-{n}") for n in range(4_097))
    rows = tuple({**receipt.projection_value(), "passage_id": f"passage-{n}",
                  "source_id": "source.one", "revision_id": f"revision-{n}",
                  "generation_id": "native-generation-1"}
                 for n, receipt in enumerate(receipts))
    driver = _Driver(tuple(reversed(rows)))
    projection = Neo4jNativeRetrievalProjection(driver, database="neo4j",
        generation_id="native-generation-1", fulltext_index="native_fulltext_1",
        vector_index="native_vector_1", driver_version=NEO4J_B2_DRIVER_VERSION)
    assert projection.corpus_metadata(receipts) == rows
    assert len(driver.calls) == 1
    query, parameters = driver.calls[0]
    assert parameters["generation_id"] == "native-generation-1"
    assert len(parameters["aggregate_ids"]) == 4_097
    assert "properties(" not in query and ".text" not in query and ".vector" not in query.replace(".vector_admission_id", "")
    assert driver.sessions == [{"default_access_mode": "READ", "database": "neo4j"}]


@pytest.mark.parametrize("defect", ["missing", "extra", "duplicate", "receipt", "generation", "passage", "source", "aggregate-type"])
def test_native_projection_compact_corpus_rejects_inexact_membership(defect):
    receipt = _receipt()
    row = {**receipt.projection_value(), "passage_id": "passage-one", "source_id": "source.one",
           "revision_id": "revision.one", "generation_id": "native-generation-1"}
    rows = [row]
    if defect == "missing": rows = []
    elif defect in {"extra", "duplicate"}: rows.append(dict(row))
    elif defect == "receipt": row["document_digest"] = _digest("f")
    elif defect == "generation": row["generation_id"] = "other-generation"
    elif defect == "passage": row["passage_id"] = ""
    elif defect == "source": row["source_id"] = None
    elif defect == "aggregate-type": row["aggregate_id"] = []
    projection = Neo4jNativeRetrievalProjection(_Driver(tuple(rows)), database="neo4j",
        generation_id="native-generation-1", fulltext_index="native_fulltext_1",
        vector_index="native_vector_1", driver_version=NEO4J_B2_DRIVER_VERSION)
    with pytest.raises(NativeRetrievalError): projection.corpus_metadata((receipt,))


def test_native_projection_compact_corpus_empty_and_repeated_receipts_need_no_io():
    driver = _Driver(())
    projection = Neo4jNativeRetrievalProjection(driver, database="neo4j",
        generation_id="native-generation-1", fulltext_index="native_fulltext_1",
        vector_index="native_vector_1", driver_version=NEO4J_B2_DRIVER_VERSION)
    assert projection.corpus_metadata(()) == ()
    receipt = _receipt()
    for invalid in ([receipt], (receipt, receipt), (object(),)):
        with pytest.raises(NativeRetrievalError): projection.corpus_metadata(invalid)
    assert not driver.calls and not driver.sessions
