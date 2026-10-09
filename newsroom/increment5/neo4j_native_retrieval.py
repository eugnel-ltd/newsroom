"""Fixed Neo4j projection/read adapter for native Increment 5 documents."""

from __future__ import annotations

import hashlib
import re
from datetime import timedelta
from typing import Any, Mapping

from newsroom.authority.types import UtcTimestamp
from newsroom.projection.models import ProjectionGenerationId, ProjectionGenerationState
from .native_retrieval import (
    NATIVE_RESULT_LIMIT,
    NATIVE_VECTOR_DIMENSIONS,
    NativeDocumentReceipt,
    NativePassageDocument,
    NativeRetrievalError,
)
from .fulltext_contracts import (
    FULLTEXT_ANALYZER,
    FULLTEXT_COMPONENT_DIGEST,
    FULLTEXT_INDEXED_FIELDS,
    FULLTEXT_PROVIDER,
    NORMALIZATION_COMPONENT_DIGEST,
    FullTextIndexState,
    FullTextProfile,
    FullTextProjectionSnapshot,
)
from .fulltext_normalizer import _normalization_inventory

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}\Z")
_RECEIPT_FIELDS = (
    "event_id", "command_id", "aggregate_id", "aggregate_version",
    "admission_id", "document_digest", "vector_admission_id",
    "embedding_receipt_admission_id",
)


def _receipt_projection(alias: str) -> str:
    return ",".join(f"{alias}.{field} AS {field}" for field in _RECEIPT_FIELDS)


class Neo4jNativeRetrievalProjection:
    """Own all Cypher for one configured native retrieval generation."""

    def __init__(
        self,
        driver: Any,
        *,
        database: str | None,
        generation_id: str,
        fulltext_index: str,
        vector_index: str,
        driver_version: str,
    ) -> None:
        if driver is None or not callable(getattr(driver, "session", None)):
            raise TypeError("native retrieval requires a Neo4j driver")
        if any(_NAME.fullmatch(item) is None for item in (fulltext_index, vector_index)) or fulltext_index == vector_index:
            raise NativeRetrievalError("native retrieval index names differ")
        if type(generation_id) is not str or not generation_id or generation_id != generation_id.strip():
            raise NativeRetrievalError("native retrieval generation differs")
        suffix = hashlib.sha256(generation_id.encode()).hexdigest()[:16]
        self._driver = driver
        self._driver_version = driver_version
        self._database = database
        self._generation = generation_id
        self._label = f"NewsroomNativeRetrievalDocument_{suffix}"
        self._constraint = f"native_retrieval_passage_{suffix}"
        self._fulltext = fulltext_index
        self._vector = vector_index

    @property
    def document_label(self) -> str:
        return self._label

    @property
    def fulltext_index(self) -> str:
        return self._fulltext

    def snapshot(
        self, *, generation_identity_digest: str, rights_manifest_digest: str,
        contiguous_ledger_seq: int, expected_document_count: int, clock: Any,
    ) -> FullTextProjectionSnapshot:
        """Read and bind the actual native full-text index state."""
        if not callable(clock):
            raise TypeError("native retrieval snapshot clock must be callable")
        recorded_at = clock()
        if type(recorded_at) is not UtcTimestamp:
            raise NativeRetrievalError("native retrieval snapshot clock differs")
        index_query = """
SHOW INDEXES YIELD name,type,state,entityType,labelsOrTypes,properties,indexProvider,options
WHERE name=$index_name
RETURN name,type,state,entityType,labelsOrTypes,properties,indexProvider,options
""".strip()
        count_query = f"MATCH (n:`{self._label}` {{generation_id:$generation_id}}) RETURN count(n) AS count"
        component_query = "CALL dbms.components() YIELD name,versions,edition WHERE name='Neo4j Kernel' RETURN versions[0] AS version,toLower(edition) AS edition"
        with self._session("READ") as session:
            component, index, count = session.execute_read(lambda transaction: (
                transaction.run(component_query).single(),
                transaction.run(index_query, index_name=self._fulltext).single(),
                transaction.run(count_query, generation_id=self._generation).single(),
            ))
        try:
            document_count = int(count["count"])
            index_state = FullTextIndexState(str(index["state"]))
            valid = (
                component["version"] == "2026.06.0"
                and str(component["edition"]).lower() == "community"
                and index["type"] == "FULLTEXT"
                and index["entityType"] == "NODE"
                and tuple(index["labelsOrTypes"]) == (self._label,)
                and tuple(index["properties"]) == FULLTEXT_INDEXED_FIELDS
                and index["indexProvider"] == FULLTEXT_PROVIDER
                and index["options"]["indexConfig"]["fulltext.analyzer"] == FULLTEXT_ANALYZER
                and index["options"]["indexConfig"]["fulltext.eventually_consistent"] is False
                and document_count == expected_document_count
            )
        except Exception as exc:
            raise NativeRetrievalError("native full-text metadata differs") from exc
        if not valid:
            raise NativeRetrievalError("native full-text metadata differs")
        return FullTextProjectionSnapshot(
            generation_id=ProjectionGenerationId.parse(self._generation),
            generation_state=ProjectionGenerationState.ACTIVE,
            generation_identity_digest=generation_identity_digest,
            document_label=self._label,
            index_name=self._fulltext,
            index_state=index_state,
            fulltext_component_digest=FULLTEXT_COMPONENT_DIGEST,
            normalization_component_digest=NORMALIZATION_COMPONENT_DIGEST,
            rights_manifest_digest=rights_manifest_digest,
            profile=FullTextProfile.NATIVE_RUNTIME,
            contiguous_ledger_seq=contiguous_ledger_seq,
            open_gap_count=0,
            dead_letter_count=0,
            validation_recorded_at=recorded_at,
            freshness_deadline=UtcTimestamp(recorded_at.value + timedelta(hours=1)),
            index_document_count=document_count,
            server_version=str(component["version"]),
            driver_version=self._driver_version,
        )

    def bootstrap(self) -> None:
        statements = (
            f"CREATE CONSTRAINT `{self._constraint}` IF NOT EXISTS FOR (n:`{self._label}`) REQUIRE n.passage_id IS UNIQUE",
            f"CREATE FULLTEXT INDEX `{self._fulltext}` IF NOT EXISTS FOR (n:`{self._label}`) ON EACH [{','.join(f'n.{field}' for field in FULLTEXT_INDEXED_FIELDS)}] OPTIONS {{indexConfig: {{`fulltext.analyzer`: 'standard-no-stop-words', `fulltext.eventually_consistent`: false}}}}",
            f"CREATE VECTOR INDEX `{self._vector}` IF NOT EXISTS FOR (n:`{self._label}`) ON n.embedding OPTIONS {{indexConfig: {{`vector.dimensions`: {NATIVE_VECTOR_DIMENSIONS}, `vector.similarity_function`: 'cosine', `vector.quantization.type`: 'none'}}}}",
        )
        with self._session("WRITE") as session:
            for statement in statements:
                session.execute_write(lambda transaction, query=statement: transaction.run(query).consume())

    def upsert(self, receipt: NativeDocumentReceipt, document: NativePassageDocument, vector: tuple[float, ...]) -> None:
        if type(receipt) is not NativeDocumentReceipt or type(document) is not NativePassageDocument or len(vector) != NATIVE_VECTOR_DIMENSIONS or document.generation_id != self._generation:
            raise NativeRetrievalError("native projection input differs")
        query = f"""
MERGE (n:`{self._label}` {{passage_id:$passage_id}})
ON CREATE SET n.dependency_root_id=$dependency_root_id, n.source_id=$source_id,
 n.generation_id=$generation_id,
 n.revision_id=$revision_id, n.representation_id=$representation_id,
 n.language=$language, n.retrieval_text=$retrieval_text,
 n.authority_aliases=$authority_aliases, n.formal_tokens=$formal_tokens,
 n.han_bigrams=$han_bigrams, n.latin_terms=$latin_terms,
 n.text_digest=$text_digest, n.rights_digest=$rights_digest,
 n.provenance_digest=$provenance_digest, n.vector_digest=$vector_digest,
 n.vector_admission_id=$vector_admission_id,
 n.embedding_receipt_digest=$embedding_receipt_digest,
 n.embedding_receipt_admission_id=$embedding_receipt_admission_id,
 n.embedding_model_digest=$embedding_model_digest, n.embedding=$embedding,
 n.event_id=$event_id, n.command_id=$command_id, n.aggregate_id=$aggregate_id,
 n.aggregate_version=$aggregate_version, n.admission_id=$admission_id,
 n.document_digest=$document_digest
ON MATCH SET n.passage_id=n.passage_id
RETURN properties(n) AS properties
""".strip()
        _, _, latin_terms, han_bigrams, formal_tokens = _normalization_inventory(
            document.text
        )
        parameters = {
            **{key: value for key, value in document.projection_value().items() if key != "text"},
            **receipt.projection_value(),
            "retrieval_text": document.text,
            "authority_aliases": [],
            "formal_tokens": list(formal_tokens),
            "han_bigrams": list(han_bigrams),
            "latin_terms": list(latin_terms),
            "embedding": list(vector),
        }
        with self._session("WRITE") as session:
            rows = session.execute_write(
                lambda transaction: tuple(transaction.run(query, **parameters))
            )
        if len(rows) != 1:
            raise NativeRetrievalError("native projection acknowledgement differs")
        try:
            retained = dict(rows[0]["properties"])
        except Exception as exc:
            raise NativeRetrievalError(
                "native projection acknowledgement differs"
            ) from exc
        if retained != parameters:
            raise NativeRetrievalError("native projection acknowledgement differs")

    def reconcile_membership(
        self, receipts: tuple[NativeDocumentReceipt, ...],
    ) -> tuple[NativeDocumentReceipt, ...]:
        """Remove revoked derived nodes and identify exact retained restores."""
        if type(receipts) is not tuple or any(
            type(receipt) is not NativeDocumentReceipt for receipt in receipts
        ):
            raise NativeRetrievalError("native projection membership differs")
        expected = {str(item.aggregate_id): item for item in receipts}
        if len(expected) != len(receipts):
            raise NativeRetrievalError("native projection membership repeats")
        read = (
            f"MATCH (n:`{self._label}` {{generation_id:$generation_id}}) "
            f"RETURN {_receipt_projection('n')}"
        )
        remove = (
            f"MATCH (n:`{self._label}` {{generation_id:$generation_id}}) "
            "WHERE NOT n.aggregate_id IN $aggregate_ids DELETE n"
        )

        def reconcile(transaction):
            rows = tuple(transaction.run(read, generation_id=self._generation))
            present = set()
            for row in rows:
                try:
                    properties = dict(row)
                    aggregate_id = str(properties["aggregate_id"])
                except Exception as exc:
                    raise NativeRetrievalError(
                        "native projection membership differs"
                    ) from exc
                retained = expected.get(aggregate_id)
                if retained is None:
                    continue
                if aggregate_id in present:
                    raise NativeRetrievalError(
                        "native projection membership repeats"
                    )
                binding = retained.projection_value()
                if any(properties.get(name) != value for name, value in binding.items()):
                    raise NativeRetrievalError(
                        "native projection retained document differs"
                    )
                present.add(aggregate_id)
            transaction.run(
                remove, generation_id=self._generation,
                aggregate_ids=sorted(expected),
            ).consume()
            return tuple(
                receipt for receipt in receipts
                if str(receipt.aggregate_id) not in present
            )

        with self._session("WRITE") as session:
            return session.execute_write(reconcile)

    def corpus_metadata(
        self, receipts: tuple[NativeDocumentReceipt, ...],
    ) -> tuple[Mapping[str, object], ...]:
        """Read the complete compact catalogue, not document text or vectors.

        Projection headers are search hints. The caller still authenticates
        selected documents against their governed receipts before using them.
        """
        if type(receipts) is not tuple or any(type(item) is not NativeDocumentReceipt for item in receipts):
            raise NativeRetrievalError("native corpus receipts differ")
        expected = {str(item.aggregate_id): item for item in receipts}
        if len(expected) != len(receipts) or len({item.event_id for item in receipts}) != len(receipts):
            raise NativeRetrievalError("native corpus receipts repeat")
        if not receipts:
            return ()
        query = (
            f"MATCH (n:`{self._label}` {{generation_id:$generation_id}}) "
            "WHERE n.aggregate_id IN $aggregate_ids "
            f"RETURN {_receipt_projection('n')},n.passage_id AS passage_id,"
            "n.source_id AS source_id,n.revision_id AS revision_id,n.generation_id AS generation_id"
        )
        with self._session("READ") as session:
            rows = session.execute_read(lambda transaction: tuple(dict(row) for row in transaction.run(
                query, generation_id=self._generation, aggregate_ids=list(expected),
            )))
        headers = {"passage_id", "source_id", "revision_id", "generation_id"}
        by_aggregate = {}
        passages = set()
        for row in rows:
            if set(row) != set(_RECEIPT_FIELDS) | headers or type(row["aggregate_id"]) is not str:
                raise NativeRetrievalError("native corpus metadata differs")
            retained = expected.get(row["aggregate_id"])
            if (retained is None or row["aggregate_id"] in by_aggregate
                    or any(type(row[key]) is not type(value) or row[key] != value
                           for key, value in retained.projection_value().items())
                    or any(type(row[key]) is not str or not row[key] or row[key] != row[key].strip()
                           for key in headers)
                    or row["generation_id"] != self._generation
                    or row["passage_id"] in passages):
                raise NativeRetrievalError("native corpus metadata differs")
            by_aggregate[row["aggregate_id"]] = row
            passages.add(row["passage_id"])
        if set(by_aggregate) != set(expected):
            raise NativeRetrievalError("native corpus metadata is incomplete")
        return tuple(by_aggregate[str(item.aggregate_id)] for item in receipts)

    def retrieve(self, *, query_text: str, query_vector: tuple[float, ...]) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
        if type(query_text) is not str or not query_text or len(query_vector) != NATIVE_VECTOR_DIMENSIONS:
            raise NativeRetrievalError("native retrieval query differs")
        limit = NATIVE_RESULT_LIMIT + 1
        receipt = _receipt_projection("node")
        fulltext = f"""
CALL db.index.fulltext.queryNodes($index_name,$query_text,{{limit:$limit}}) YIELD node,score
RETURN {receipt},score ORDER BY score DESC,node.passage_id LIMIT $limit
""".strip()
        vector = f"""
CALL db.index.vector.queryNodes($index_name,$limit,$vector) YIELD node,score
RETURN {receipt},score ORDER BY score DESC,node.passage_id LIMIT $limit
""".strip()
        common = {"limit": limit}
        with self._session("READ") as session:
            fulltext_rows = tuple(session.execute_read(lambda transaction: tuple(transaction.run(fulltext, **common, index_name=self._fulltext, query_text=query_text))))
            vector_rows = tuple(session.execute_read(lambda transaction: tuple(transaction.run(
                vector, limit=NATIVE_RESULT_LIMIT, index_name=self._vector,
                vector=list(query_vector),
            ))))
        return fulltext_rows, vector_rows

    def retrieve_vector(
        self, *, query_vector: tuple[float, ...]
    ) -> tuple[Mapping[str, object], ...]:
        """Run only the native vector index; no Lucene text is evaluated."""
        if len(query_vector) != NATIVE_VECTOR_DIMENSIONS:
            raise NativeRetrievalError("native vector query differs")
        limit = NATIVE_RESULT_LIMIT
        receipt = _receipt_projection("node")
        query = f"""
CALL db.index.vector.queryNodes($index_name,$limit,$vector) YIELD node,score
RETURN {receipt},score ORDER BY score DESC,node.passage_id LIMIT $limit
""".strip()
        with self._session("READ") as session:
            return tuple(session.execute_read(
                lambda transaction: tuple(transaction.run(
                    query, index_name=self._vector, limit=limit,
                    vector=list(query_vector),
                ))
            ))

    def _session(self, mode: str):
        values: dict[str, object] = {"default_access_mode": mode}
        if self._database is not None:
            values["database"] = self._database
        return self._driver.session(**values)


__all__ = ["Neo4jNativeRetrievalProjection"]
