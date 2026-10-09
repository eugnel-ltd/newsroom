from __future__ import annotations

import hashlib
import os
import uuid
from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator

import pytest

from newsroom.authority.canonical import digest_bytes
from newsroom.authority.neo4j_admitted_graph_reader import Neo4jAdmittedGraphReadPort
from newsroom.authority.types import AggregateId, ObjectAdmissionId
from newsroom.increment5.admitted_graph_retriever import canonical_node_digest
from newsroom.increment5.native_retrieval import (
    NATIVE_VECTOR_DIMENSIONS,
    NativeDocumentReceipt,
    NativePassageDocument,
    NativeRetrievalError,
)
from newsroom.increment5.neo4j_native_retrieval import (
    Neo4jNativeRetrievalProjection,
)


neo4j = pytest.importorskip("neo4j")


def _setting(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


@contextmanager
def actual_driver() -> Iterator[tuple[object, str | None]]:
    uri = _setting("NEWSROOM_NEO4J_URI", "NEO4J_URI")
    password = _setting("NEWSROOM_NEO4J_PASSWORD", "NEO4J_PASSWORD")
    if uri is None or password is None:
        pytest.skip("authenticated Neo4j service settings are not available")
    user = _setting("NEWSROOM_NEO4J_USER", "NEO4J_USER", default="neo4j")
    database = _setting("NEWSROOM_NEO4J_DATABASE", "NEO4J_DATABASE")
    driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        yield driver, database
    finally:
        driver.close()


def _session(driver: object, database: str | None):
    config: dict[str, object] = {}
    if database is not None:
        config["database"] = database
    return driver.session(**config)


def _cleanup(driver: object, database: str | None, generations: list[str]) -> None:
    with _session(driver, database) as session:
        session.run(
            "MATCH (n) WHERE n.generation_id IN $generations DETACH DELETE n",
            generations=generations,
        ).consume()


def _native_document(
    *, generation: str, marker: str,
) -> tuple[NativeDocumentReceipt, NativePassageDocument]:
    text = f"Native projection common evidence {marker}."
    document = NativePassageDocument(
        generation_id=generation,
        passage_id=str(uuid.uuid4()),
        dependency_root_id=f"event:{marker}",
        source_id=f"source.{marker}",
        revision_id=f"revision.{marker}",
        representation_id=f"representation.{marker}",
        language="en-GB",
        text=text,
        text_digest=digest_bytes(text.encode()),
        rights_digest=digest_bytes(f"rights:{marker}".encode()),
        provenance_digest=digest_bytes(f"provenance:{marker}".encode()),
        vector_digest=digest_bytes(f"vector:{marker}".encode()),
        vector_admission_id=str(ObjectAdmissionId.new()),
        embedding_receipt_digest=digest_bytes(f"receipt:{marker}".encode()),
        embedding_receipt_admission_id=str(ObjectAdmissionId.new()),
        embedding_model_digest=digest_bytes(b"native-model"),
    )
    receipt = NativeDocumentReceipt(
        f"event-{marker}",
        f"command-{marker}",
        AggregateId.new(),
        1,
        ObjectAdmissionId.new(),
        document.digest,
        ObjectAdmissionId.parse(document.vector_admission_id),
        ObjectAdmissionId.parse(document.embedding_receipt_admission_id),
    )
    return receipt, document


def _receipt_ids(rows) -> set[str]:
    return {str(row["aggregate_id"]) for row in rows}


def test_increment5b4_fixed_port_reads_only_exact_generation_and_allowed_state() -> None:
    with actual_driver() as (driver, database):
        generation = f"i5b4-{uuid.uuid4()}"
        other_generation = f"i5b4-other-{uuid.uuid4()}"
        generations = [generation, other_generation]
        try:
            with _session(driver, database) as session:
                session.run(
                    """
                    CREATE (root:Source {
                      generation_id: $generation,
                      canonical_id: 'source:root',
                      identity_digest: $root_digest
                    })
                    CREATE (revision:Revision {
                      generation_id: $generation,
                      canonical_id: 'revision:one',
                      identity_digest: $revision_digest
                    })
                    CREATE (candidate:Candidate {
                      generation_id: $generation,
                      canonical_id: 'candidate:one',
                      identity_digest: $candidate_digest
                    })
                    CREATE (blocked:Lead {
                      generation_id: $generation,
                      canonical_id: 'lead:disallowed-relation',
                      identity_digest: $blocked_digest
                    })
                    CREATE (old:Signal {
                      generation_id: $generation,
                      canonical_id: 'signal:old',
                      identity_digest: $old_digest
                    })
                    CREATE (foreign:Revision {
                      generation_id: $other_generation,
                      canonical_id: 'revision:foreign',
                      identity_digest: $foreign_digest
                    })
                    CREATE (root)-[:DEVELOPMENT_OF {
                      generation_id: $generation,
                      relation_id: 'relation:root-revision',
                      valid_from: '2020-01-01T00:00:00Z',
                      valid_to: '2035-01-01T00:00:00Z',
                      observed_at: '2026-08-01T00:00:00Z'
                    }]->(revision)
                    CREATE (revision)-[:ABOUT_EVENT {
                      generation_id: $generation,
                      relation_id: 'relation:revision-candidate',
                      valid_from: '2020-01-01T00:00:00Z',
                      valid_to: '2035-01-01T00:00:00Z',
                      observed_at: '2026-08-01T00:00:00Z'
                    }]->(candidate)
                    CREATE (root)-[:NOT_ADMITTED {
                      generation_id: $generation,
                      relation_id: 'relation:disallowed',
                      valid_from: '2020-01-01T00:00:00Z',
                      valid_to: '2035-01-01T00:00:00Z',
                      observed_at: '2026-08-01T00:00:00Z'
                    }]->(blocked)
                    CREATE (root)-[:SUPPORTS {
                      generation_id: $generation,
                      relation_id: 'relation:old',
                      valid_from: '2020-01-01T00:00:00Z',
                      valid_to: '2035-01-01T00:00:00Z',
                      observed_at: '2026-06-01T00:00:00Z'
                    }]->(old)
                    CREATE (root)-[:DEVELOPMENT_OF {
                      generation_id: $other_generation,
                      relation_id: 'relation:cross-generation',
                      valid_from: '2020-01-01T00:00:00Z',
                      valid_to: '2035-01-01T00:00:00Z',
                      observed_at: '2026-08-01T00:00:00Z'
                    }]->(foreign)
                    """,
                    generation=generation,
                    other_generation=other_generation,
                    root_digest=canonical_node_digest("source:root"),
                    revision_digest=canonical_node_digest("revision:one"),
                    candidate_digest=canonical_node_digest("candidate:one"),
                    blocked_digest=canonical_node_digest("lead:disallowed-relation"),
                    old_digest=canonical_node_digest("signal:old"),
                    foreign_digest=canonical_node_digest("revision:foreign"),
                ).consume()

            port = Neo4jAdmittedGraphReadPort(driver, database=database)
            root = port.read_root(
                generation_id=generation,
                canonical_id="source:root",
                timeout_ms=5_000,
            )
            assert root is not None
            assert root.generation_id == generation
            assert root.canonical_id == "source:root"
            assert root.identity_digest == canonical_node_digest("source:root")
            assert root.labels == ("Source",)

            first = port.expand_frontier(
                generation_id=generation,
                frontier_ids=("source:root",),
                query_valid_time="2026-08-06T08:59:00Z",
                temporal_lower_bound="2026-07-06T08:59:00Z",
                timeout_ms=5_000,
            )
            assert [edge.relation_id for edge in first] == [
                "relation:root-revision"
            ]
            assert first[0].source_id == "source:root"
            assert first[0].target_id == "revision:one"
            assert first[0].predicate == "DEVELOPMENT_OF"

            second = port.expand_frontier(
                generation_id=generation,
                frontier_ids=("revision:one",),
                query_valid_time="2026-08-06T08:59:00Z",
                temporal_lower_bound="2026-07-06T08:59:00Z",
                timeout_ms=5_000,
            )
            assert [edge.relation_id for edge in second] == [
                "relation:revision-candidate",
                "relation:root-revision",
            ]
            assert all(edge.generation_id == generation for edge in second)
            assert all(edge.target_id != "revision:foreign" for edge in first + second)
            assert all(edge.relation_id != "relation:disallowed" for edge in first + second)
            assert all(edge.relation_id != "relation:old" for edge in first + second)
        finally:
            _cleanup(driver, database, generations)


def test_increment5b4_fixed_port_excludes_future_observations() -> None:
    with actual_driver() as (driver, database):
        generation = f"i5b4-future-{uuid.uuid4()}"
        try:
            with _session(driver, database) as session:
                session.run(
                    """
                    CREATE (root:Source {
                      generation_id: $generation,
                      canonical_id: 'source:root',
                      identity_digest: $root_digest
                    })
                    CREATE (future:Signal {
                      generation_id: $generation,
                      canonical_id: 'signal:future',
                      identity_digest: $future_digest
                    })
                    CREATE (root)-[:ABOUT_EVENT {
                      generation_id: $generation,
                      relation_id: 'relation:future-observation',
                      valid_from: '2020-01-01T00:00:00Z',
                      valid_to: '2035-01-01T00:00:00Z',
                      observed_at: '2026-08-07T00:00:00Z'
                    }]->(future)
                    """,
                    generation=generation,
                    root_digest=canonical_node_digest("source:root"),
                    future_digest=canonical_node_digest("signal:future"),
                ).consume()
            port = Neo4jAdmittedGraphReadPort(driver, database=database)
            edges = port.expand_frontier(
                generation_id=generation,
                frontier_ids=("source:root",),
                query_valid_time="2026-08-06T08:59:00Z",
                temporal_lower_bound="2026-07-06T08:59:00Z",
                timeout_ms=5_000,
            )
            assert edges == ()
        finally:
            _cleanup(driver, database, [generation])


def test_native_projection_reconciles_actual_fulltext_and_vector_membership() -> None:
    with actual_driver() as (driver, database):
        generation = str(uuid.uuid4())
        other_generation = str(uuid.uuid4())
        suffix = uuid.uuid4().hex
        fulltext_index = f"native_fulltext_{suffix}"
        vector_index = f"native_vector_{suffix}"
        constraint = (
            "native_retrieval_passage_"
            f"{hashlib.sha256(generation.encode()).hexdigest()[:16]}"
        )
        projection = Neo4jNativeRetrievalProjection(
            driver,
            database=database,
            generation_id=generation,
            fulltext_index=fulltext_index,
            vector_index=vector_index,
            driver_version=neo4j.__version__,
        )
        receipt_a, document_a = _native_document(
            generation=generation, marker="alpha",
        )
        text = "common " + " ".join(
            f"governedterm{index}" for index in range(65)
        )
        document_a = replace(
            document_a, text=text, text_digest=digest_bytes(text.encode())
        )
        receipt_a = replace(receipt_a, document_digest=document_a.digest)
        receipt_b, document_b = _native_document(
            generation=generation, marker="beta",
        )
        vector = (1.0,) + (0.0,) * (NATIVE_VECTOR_DIMENSIONS - 1)
        expected = {str(receipt_a.aggregate_id), str(receipt_b.aggregate_id)}
        try:
            projection.bootstrap()
            with _session(driver, database) as session:
                session.run("CALL db.awaitIndexes(30)").consume()
                session.run(
                    f"CREATE (n:`{projection.document_label}` "
                    "{generation_id:$generation_id,passage_id:$passage_id})",
                    generation_id=other_generation,
                    passage_id=str(uuid.uuid4()),
                ).consume()
            projection.upsert(receipt_a, document_a, vector)
            projection.upsert(receipt_b, document_b, vector)
            metadata = projection.corpus_metadata((receipt_b, receipt_a))
            assert [row["passage_id"] for row in metadata] == [document_b.passage_id, document_a.passage_id]
            assert all(set(row) == set(receipt_a.projection_value()) | {
                "passage_id", "source_id", "revision_id", "generation_id"} for row in metadata)

            fulltext, vector_hits = projection.retrieve(
                query_text="common", query_vector=vector,
            )
            assert _receipt_ids(fulltext) == expected
            assert _receipt_ids(vector_hits) == expected
            with _session(driver, database) as session:
                assert session.run(
                    f"MATCH (n:`{projection.document_label}` "
                    "{passage_id:$passage_id}) RETURN size(n.latin_terms) AS count",
                    passage_id=document_a.passage_id,
                ).single()["count"] == 66

            corrupt_b = replace(
                receipt_b, document_digest=digest_bytes(b"corrupt-document"),
            )
            with pytest.raises(
                NativeRetrievalError, match="retained document differs",
            ):
                projection.reconcile_membership((corrupt_b,))
            with _session(driver, database) as session:
                assert session.run(
                    f"MATCH (n:`{projection.document_label}` "
                    "{generation_id:$generation_id}) RETURN count(n) AS count",
                    generation_id=generation,
                ).single()["count"] == 2

            assert projection.reconcile_membership((receipt_b,)) == ()
            fulltext, vector_hits = projection.retrieve(
                query_text="common", query_vector=vector,
            )
            assert _receipt_ids(fulltext) == {str(receipt_b.aggregate_id)}
            assert _receipt_ids(vector_hits) == {str(receipt_b.aggregate_id)}

            assert projection.reconcile_membership(
                (receipt_a, receipt_b),
            ) == (receipt_a,)
            projection.upsert(receipt_a, document_a, vector)
            fulltext, vector_hits = projection.retrieve(
                query_text="common", query_vector=vector,
            )
            assert _receipt_ids(fulltext) == expected
            assert _receipt_ids(vector_hits) == expected
            with _session(driver, database) as session:
                assert session.run(
                    f"MATCH (n:`{projection.document_label}` "
                    "{generation_id:$generation_id}) RETURN count(n) AS count",
                    generation_id=other_generation,
                ).single()["count"] == 1
            for index in range(7):
                extra_receipt, extra_document = _native_document(
                    generation=generation, marker=f"extra-{index}",
                )
                projection.upsert(extra_receipt, extra_document, vector)
            assert len(projection.retrieve_vector(query_vector=vector)) == 8
        finally:
            _cleanup(driver, database, [generation, other_generation])
            with _session(driver, database) as session:
                session.run(f"DROP INDEX `{fulltext_index}` IF EXISTS").consume()
                session.run(f"DROP INDEX `{vector_index}` IF EXISTS").consume()
                session.run(f"DROP CONSTRAINT `{constraint}` IF EXISTS").consume()
