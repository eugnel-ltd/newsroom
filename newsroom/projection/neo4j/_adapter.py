from __future__ import annotations

from collections.abc import Callable, Mapping
import time
from threading import RLock
from typing import Any

from newsroom.authority.neo4j_fulltext_reader import (
    FULLTEXT_SOURCE_SCOPE_CANDIDATE_LIMIT,
)
from newsroom.authority.types import TrustScope, UtcTimestamp
from newsroom.projection.ontology import ProjectionNodeType, ProjectionRelationType

from .models import (
    NEO4J_B2_DRIVER_VERSION,
    NEO4J_B2_SERVER_VERSION,
    Neo4jApplyOutcome,
    Neo4jApplyResult,
    Neo4jCompatibility,
    Neo4jCompatibilityError,
    Neo4jConnectionError,
    Neo4jIdentityConflict,
    Neo4jProjectorConfig,
    Neo4jReadError,
    Neo4jStructuralRead,
    Neo4jWriteError,
    StructuralBatch,
    StructuralGraphNodeView,
    StructuralGraphRelationView,
)
from ._state import (
    _DELIVERY_PROPERTY_KEYS,
    _NODE_PROPERTY_KEYS,
    _RELATION_IDENTITY_PROPERTY_KEYS,
    _RELATION_PROPERTY_KEYS,
    _actual_projection_state_digest,
    _delivery_properties,
    _expected_projection_state_digest,
    _node_properties,
    _relation_identity_properties,
    _relation_properties,
)

_COMPONENT_QUERY = """
CALL dbms.components() YIELD name, versions, edition
WHERE name = 'Neo4j Kernel'
RETURN versions[0] AS version, toLower(edition) AS edition
"""

_FULLTEXT_INDEX_INVENTORY_QUERY = """
SHOW INDEXES
YIELD name, type, state, entityType, labelsOrTypes, properties, indexProvider, options
WHERE name = $index_name
RETURN name, type, state, entityType, labelsOrTypes, properties, indexProvider, options
ORDER BY name
"""

_FULLTEXT_READ_QUERY = """
CALL db.index.fulltext.queryNodes(
  $index_name,
  $query,
  {limit: $candidate_limit}
)
YIELD node, score
WITH node, score
ORDER BY score DESC, node.passage_id
WITH collect({node: node, score: score}) AS candidates
WITH candidates,
     size(candidates) = $candidate_limit AS candidate_overflow
WITH [candidate IN candidates
      WHERE candidate.node.generation_id = $generation_id
      | {
          generation_id: candidate.node.generation_id,
          passage_id: candidate.node.passage_id,
          document_digest: candidate.node.document_digest,
          language: candidate.node.language,
          score: candidate.score
        }][0..$candidate_limit] AS rows,
     candidate_overflow
RETURN candidate_overflow, rows
"""

_NATIVE_FULLTEXT_READ_QUERY = """
CALL db.index.fulltext.queryNodes($index_name, $query)
YIELD node, score
WHERE node.generation_id = $generation_id
  AND node.passage_id IN $eligible_passage_ids
WITH node, score
ORDER BY score DESC, node.passage_id
LIMIT $limit
WITH collect({
  generation_id: node.generation_id,
  passage_id: node.passage_id,
  document_digest: node.document_digest,
  language: node.language,
  score: score
}) AS rows
RETURN false AS candidate_overflow, rows
"""


_SCHEMA_QUERIES = (
    """
    CREATE CONSTRAINT newsroom_projection_node_identity IF NOT EXISTS
    FOR (n:NewsroomProjectionNode)
    REQUIRE (n.generation_id, n.canonical_id) IS UNIQUE
    """,
    """
    CREATE CONSTRAINT newsroom_projection_delivery_identity IF NOT EXISTS
    FOR (d:NewsroomProjectionDelivery)
    REQUIRE (d.generation_id, d.ledger_seq) IS UNIQUE
    """,
    """
    CREATE CONSTRAINT newsroom_projection_relation_identity IF NOT EXISTS
    FOR (r:NewsroomProjectionRelationIdentity)
    REQUIRE (r.generation_id, r.relation_key) IS UNIQUE
    """,
    """
    CREATE INDEX newsroom_projection_node_first_sequence IF NOT EXISTS
    FOR (n:NewsroomProjectionNode)
    ON (n.generation_id, n.first_ledger_seq)
    """,
)

_FIND_DELIVERY_QUERY = """
MATCH (d:NewsroomProjectionDelivery {
  generation_id: $generation_id,
  ledger_seq: $ledger_seq
})
RETURN properties(d) AS properties
"""

_CREATE_DELIVERY_QUERY = """
CREATE (d:NewsroomProjectionDelivery)
SET d = $properties
RETURN properties(d) AS properties
"""

_MERGE_NODE_QUERY = """
MERGE (n:NewsroomProjectionNode {
  generation_id: $generation_id,
  canonical_id: $canonical_id
})
ON CREATE SET n = $properties
RETURN properties(n) AS properties
"""

_UPDATE_NODE_FIRST_PROVENANCE_QUERY = """
MATCH (n:NewsroomProjectionNode {
  generation_id: $generation_id,
  canonical_id: $canonical_id
})
SET n.first_ledger_seq = $first_ledger_seq,
    n.first_source_event_id = $first_source_event_id,
    n.first_source_event_digest = $first_source_event_digest
RETURN properties(n) AS properties
"""

_MERGE_RELATION_IDENTITY_QUERY = """
MERGE (i:NewsroomProjectionRelationIdentity {
  generation_id: $generation_id,
  relation_key: $relation_key
})
ON CREATE SET i = $properties
RETURN properties(i) AS properties
"""

_RELATION_QUERY_TEMPLATE = """
MATCH (source:NewsroomProjectionNode {
  generation_id: $generation_id,
  canonical_id: $source_canonical_id
})
MATCH (target:NewsroomProjectionNode {
  generation_id: $generation_id,
  canonical_id: $target_canonical_id
})
MERGE (source)-[r:%s {
  generation_id: $generation_id,
  relation_key: $relation_key
}]->(target)
ON CREATE SET r = $properties
RETURN properties(r) AS properties
"""

_RELATION_QUERIES = {
    relation_type: _RELATION_QUERY_TEMPLATE % relation_type.value
    for relation_type in ProjectionRelationType
}

_READ_NODES_QUERY = """
MATCH (n:NewsroomProjectionNode {generation_id: $generation_id})
WHERE n.canonical_id IN $canonical_ids
  AND n.first_ledger_seq <= $maximum_ledger_seq
RETURN properties(n) AS properties
ORDER BY n.canonical_id
LIMIT $limit
"""

_READ_RELATIONS_QUERY = """
MATCH (source:NewsroomProjectionNode {generation_id: $generation_id})
      -[r]->
      (target:NewsroomProjectionNode {generation_id: $generation_id})
WHERE (source.canonical_id IN $canonical_ids OR target.canonical_id IN $canonical_ids)
  AND r.ledger_seq <= $maximum_ledger_seq
  AND source.first_ledger_seq <= $maximum_ledger_seq
  AND target.first_ledger_seq <= $maximum_ledger_seq
RETURN properties(source) AS source_properties,
       type(r) AS relation_type,
       properties(r) AS relation_properties,
       properties(target) AS target_properties
ORDER BY r.ledger_seq, r.relation_key
LIMIT $limit
"""


_STATE_NODES_QUERY = """
MATCH (value)
WHERE value.generation_id = $generation_id
  AND (value:NewsroomProjectionNode
       OR value:NewsroomProjectionDelivery
       OR value:NewsroomProjectionRelationIdentity)
RETURN labels(value) AS labels, properties(value) AS properties
"""

_STATE_RELATIONSHIPS_QUERY = """
MATCH (source:NewsroomProjectionNode)-[relation]->(target:NewsroomProjectionNode)
WHERE relation.generation_id = $generation_id
  AND type(relation) IN $relation_types
RETURN labels(source) AS source_labels,
       properties(source) AS source_properties,
       type(relation) AS relation_type,
       properties(relation) AS relation_properties,
       labels(target) AS target_labels,
       properties(target) AS target_properties
"""

_CLEANUP_GENERATION_QUERY = """
MATCH (value)
WHERE value.generation_id = $generation_id
  AND (value:NewsroomProjectionNode
       OR value:NewsroomProjectionDelivery
       OR value:NewsroomProjectionRelationIdentity)
WITH value LIMIT $batch_size
DETACH DELETE value
RETURN count(value) AS deleted_count
"""


_TOMBSTONE_RELATIONS_QUERY = """
MATCH (source:NewsroomProjectionNode {generation_id: $generation_id})
      -[relation]->
      (target:NewsroomProjectionNode {generation_id: $generation_id})
WHERE relation.generation_id = $generation_id
  AND type(relation) IN $relation_types
  AND relation.object_admission_id IN $object_admission_ids
WITH collect(DISTINCT relation.relation_key) AS relation_keys,
     collect(DISTINCT source.canonical_id)
       + collect(DISTINCT target.canonical_id) AS canonical_ids,
     collect(relation) AS relations
FOREACH (item IN relations | DELETE item)
RETURN relation_keys, canonical_ids, size(relations) AS deleted_count
"""
_TOMBSTONE_RELATION_IDENTITIES_QUERY = """
MATCH (identity:NewsroomProjectionRelationIdentity {generation_id: $generation_id})
WHERE identity.relation_key IN $relation_keys
WITH collect(identity) AS identities
FOREACH (item IN identities | DELETE item)
RETURN size(identities) AS deleted_count
"""
_TOMBSTONE_ORPHAN_NODES_QUERY = """
MATCH (node:NewsroomProjectionNode {generation_id: $generation_id})
WHERE node.canonical_id IN $canonical_ids AND NOT (node)--()
WITH collect(node) AS nodes
FOREACH (item IN nodes | DELETE item)
RETURN size(nodes) AS deleted_count
"""




class _Neo4jAdapter:
    """Private fixed-query adapter. It is never returned by a public facade."""

    __slots__ = (
        "_driver",
        "_config",
        "_driver_version",
        "_closed",
        "_lock",
        "_monotonic_ns",
        "_unit_of_work",
    )

    def __init__(
        self,
        *,
        driver: Any,
        config: Neo4jProjectorConfig,
        driver_version: str,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        unit_of_work_factory: Callable[..., Callable] | None = None,
    ) -> None:
        if not callable(monotonic_ns):
            raise TypeError("Neo4j monotonic clock must be callable")
        if unit_of_work_factory is None:
            unit_of_work_factory = _neo4j_unit_of_work_factory()
        if not callable(unit_of_work_factory):
            raise TypeError("Neo4j unit-of-work factory must be callable")
        self._driver = driver
        self._config = config
        self._driver_version = driver_version
        self._closed = False
        self._lock = RLock()
        self._monotonic_ns = monotonic_ns
        self._unit_of_work = unit_of_work_factory

    def verify_compatibility(self) -> Neo4jCompatibility:
        self._require_open()
        try:
            self._driver.verify_connectivity()
            with self._driver.session(database=self._config.database) as session:
                record = session.execute_read(
                    lambda transaction: transaction.run(_COMPONENT_QUERY).single()
                )
        except Exception as exc:
            raise Neo4jConnectionError(
                "Neo4j authenticated compatibility check failed"
            ) from None
        if record is None:
            raise Neo4jCompatibilityError("Neo4j service did not identify its component")
        try:
            server_version = str(record["version"])
            edition = str(record["edition"]).lower()
        except Exception:
            raise Neo4jCompatibilityError("Neo4j service returned malformed compatibility metadata") from None
        compatibility = Neo4jCompatibility(
            server_version=server_version,
            edition=edition,
            driver_version=self._driver_version,
        )
        if compatibility.server_version != NEO4J_B2_SERVER_VERSION:
            raise Neo4jCompatibilityError(
                "Neo4j server is not the exact B2 qualification target"
            )
        if compatibility.edition != "community":
            raise Neo4jCompatibilityError(
                "Neo4j edition is not the exact Community qualification target"
            )
        if compatibility.driver_version != NEO4J_B2_DRIVER_VERSION:
            raise Neo4jCompatibilityError(
                "Neo4j driver is not the exact B2 qualification target"
            )
        return compatibility

    def bootstrap_schema(self) -> None:
        self._require_open()
        try:
            with self._driver.session(database=self._config.database) as session:
                for query in _SCHEMA_QUERIES:
                    session.execute_write(
                        lambda transaction, statement=query: transaction.run(statement).consume()
                    )
        except Exception:
            raise Neo4jWriteError("Neo4j structural schema bootstrap failed") from None

    def apply(self, batch: StructuralBatch) -> Neo4jApplyResult:
        self._require_open()
        if not isinstance(batch, StructuralBatch):
            raise TypeError("Neo4j structural apply requires a typed batch")
        with self._lock:
            try:
                with self._driver.session(database=self._config.database) as session:
                    outcome = session.execute_write(self._apply_transaction, batch)
            except Neo4jIdentityConflict:
                raise
            except Exception:
                raise Neo4jWriteError("Neo4j structural transaction failed") from None
        return Neo4jApplyResult(
            outcome=outcome,
            generation_id=batch.generation_id,
            ledger_seq=batch.ledger_seq,
            source_event_id=batch.source_event_id,
            source_event_digest=batch.source_event_digest,
            batch_digest=batch.batch_digest,
        )

    @property
    def driver_version(self) -> str:
        return self._driver_version

    def open_native_retrieval_projection(
        self,
        *,
        generation_id: str,
        fulltext_index: str,
        vector_index: str,
    ) -> Any:
        """Create the fixed native retrieval projection on this owned driver."""

        self._require_open()
        from newsroom.increment5.neo4j_native_retrieval import (
            Neo4jNativeRetrievalProjection,
        )

        return Neo4jNativeRetrievalProjection(
            self._driver,
            database=self._config.database,
            generation_id=generation_id,
            fulltext_index=fulltext_index,
            vector_index=vector_index,
            driver_version=self._driver_version,
        )

    def read_increment5_fulltext(
        self,
        *,
        phase: str,
        index_name: str | None,
        lucene_expression: str | None,
        generation_id: str | None,
        source_ids: tuple[str, ...],
        limit: int,
        timeout_ns: int,
        eligible_passage_ids: tuple[str, ...] | None = None,
    ) -> Any:
        """Execute one fixed phase of the Increment 5 full-text read port."""

        self._require_open()
        if phase not in {"COMPONENT", "INDEX", "QUERY"}:
            raise Neo4jReadError(
                "Neo4j Increment 5 full-text phase is invalid"
            )
        if (
            isinstance(timeout_ns, bool)
            or not isinstance(timeout_ns, int)
            or not 0 < timeout_ns <= 5_000_000_000
        ):
            raise Neo4jReadError(
                "Neo4j Increment 5 full-text timeout is invalid"
            )
        if phase == "COMPONENT":
            if any(
                value is not None
                for value in (index_name, lucene_expression, generation_id)
            ) or limit != 0:
                raise Neo4jReadError(
                    "Neo4j component phase carries forbidden controls"
                )
            callback = lambda transaction: transaction.run(
                _COMPONENT_QUERY,
                {},
            ).single()
            operation = "increment5.fulltext.component"
        else:
            if (
                not isinstance(index_name, str)
                or not index_name
                or len(index_name.encode("utf-8")) > 128
                or not index_name.isascii()
                or not index_name[0].isalpha()
                or any(
                    not (character.isalnum() or character == "_")
                    for character in index_name
                )
            ):
                raise Neo4jReadError(
                    "Neo4j Increment 5 full-text index identity is invalid"
                )
            if phase == "INDEX":
                if (
                    lucene_expression is not None
                    or generation_id is not None
                    or limit != 0
                ):
                    raise Neo4jReadError(
                        "Neo4j index phase carries forbidden query controls"
                    )
                callback = lambda transaction: tuple(
                    transaction.run(
                        _FULLTEXT_INDEX_INVENTORY_QUERY,
                        {"index_name": index_name},
                    )
                )
                operation = "increment5.fulltext.index"
            else:
                if (
                    not isinstance(lucene_expression, str)
                    or not lucene_expression
                    or lucene_expression != lucene_expression.strip()
                    or len(lucene_expression.encode("utf-8")) > 32_768
                    or any(
                        ord(character) < 0x20
                        for character in lucene_expression
                    )
                ):
                    raise Neo4jReadError(
                        "Neo4j Increment 5 full-text expression is invalid"
                    )
                if (
                    not isinstance(generation_id, str)
                    or not generation_id
                    or generation_id != generation_id.strip()
                    or len(generation_id.encode("utf-8")) > 128
                    or any(ord(character) < 0x20 for character in generation_id)
                ):
                    raise Neo4jReadError(
                        "Neo4j Increment 5 generation identity is invalid"
                    )
                if not isinstance(source_ids, tuple) or len(source_ids) > 8:
                    raise Neo4jReadError(
                        "Neo4j Increment 5 full-text source scope is invalid"
                    )
                for source_id in source_ids:
                    if (
                        not isinstance(source_id, str)
                        or not source_id
                        or source_id != source_id.strip()
                        or len(source_id.encode("utf-8")) > 256
                        or any(ord(character) < 0x20 for character in source_id)
                    ):
                        raise Neo4jReadError(
                            "Neo4j Increment 5 full-text source scope is invalid"
                        )
                if source_ids != tuple(sorted(set(source_ids))):
                    raise Neo4jReadError(
                        "Neo4j Increment 5 full-text source scope is invalid"
                    )
                if eligible_passage_ids is None:
                    if isinstance(limit, bool) or limit != 9:
                        raise Neo4jReadError(
                            "Neo4j Increment 5 full-text overflow limit must equal nine"
                        )
                    candidate_limit = (
                        FULLTEXT_SOURCE_SCOPE_CANDIDATE_LIMIT
                        if source_ids
                        else limit
                    )
                    callback = lambda transaction: transaction.run(
                        _FULLTEXT_READ_QUERY,
                        {
                            "index_name": index_name,
                            "query": lucene_expression,
                            "generation_id": generation_id,
                            "candidate_limit": candidate_limit,
                            "limit": limit,
                        },
                    ).single()
                else:
                    if (
                        not isinstance(eligible_passage_ids, tuple)
                        or len(eligible_passage_ids) > 4_096
                        or eligible_passage_ids
                        != tuple(sorted(set(eligible_passage_ids)))
                        or any(
                            not isinstance(item, str)
                            or not item
                            or item != item.strip()
                            or len(item.encode("utf-8")) > 256
                            or any(ord(character) < 0x20 for character in item)
                            for item in eligible_passage_ids
                        )
                        or isinstance(limit, bool)
                        or limit != 8
                    ):
                        raise Neo4jReadError(
                            "Neo4j native full-text query controls are invalid"
                        )
                    callback = lambda transaction: transaction.run(
                        _NATIVE_FULLTEXT_READ_QUERY,
                        {
                            "index_name": index_name,
                            "query": lucene_expression,
                            "generation_id": generation_id,
                            "eligible_passage_ids": list(eligible_passage_ids),
                            "limit": limit,
                        },
                    ).single()
                operation = "increment5.fulltext.query"

        started_ns = self._monotonic_ns()
        if isinstance(started_ns, bool) or not isinstance(started_ns, int):
            raise Neo4jReadError(
                "Neo4j Increment 5 monotonic clock is invalid"
            )
        with self._lock:
            current_ns = self._monotonic_ns()
            if (
                isinstance(current_ns, bool)
                or not isinstance(current_ns, int)
                or current_ns < started_ns
            ):
                raise Neo4jReadError(
                    "Neo4j Increment 5 monotonic clock moved backwards"
                )
            remaining_ns = timeout_ns - (current_ns - started_ns)
            if remaining_ns <= 0:
                raise Neo4jReadError(
                    "Neo4j Increment 5 full-text read timed out"
                )
            managed = self._unit_of_work(
                timeout=remaining_ns / 1_000_000_000,
                metadata={"newsroom_operation": operation},
            )(callback)
            try:
                with self._driver.session(database=self._config.database) as session:
                    result = session.execute_read(managed)
            except Exception as exc:
                completed_ns = self._monotonic_ns()
                if (
                    isinstance(completed_ns, bool)
                    or not isinstance(completed_ns, int)
                    or completed_ns < started_ns
                ):
                    raise Neo4jReadError(
                        "Neo4j Increment 5 monotonic clock moved backwards"
                    ) from None
                if (
                    completed_ns - started_ns > timeout_ns
                    or self._increment5_fulltext_timeout(exc)
                ):
                    raise Neo4jReadError(
                        "Neo4j Increment 5 full-text read timed out"
                    ) from None
                raise Neo4jReadError(
                    "Neo4j Increment 5 full-text read failed"
                ) from None
            completed_ns = self._monotonic_ns()
            if (
                isinstance(completed_ns, bool)
                or not isinstance(completed_ns, int)
                or completed_ns < started_ns
            ):
                raise Neo4jReadError(
                    "Neo4j Increment 5 monotonic clock moved backwards"
                )
            if completed_ns - started_ns > timeout_ns:
                raise Neo4jReadError(
                    "Neo4j Increment 5 full-text read timed out"
                )
            return result

    @staticmethod
    def _increment5_fulltext_timeout(error: Exception) -> bool:
        identity = f"{type(error).__name__} {error}".casefold()
        return any(
            token in identity
            for token in (
                "deadline",
                "terminated",
                "timeout",
                "timed out",
                "transactiontimedout",
            )
        )

    def read(
        self,
        *,
        generation_id: str,
        canonical_ids: tuple[str, ...],
        maximum_ledger_seq: int,
        limit: int,
    ) -> Neo4jStructuralRead:
        self._require_open()
        parameters = {
            "generation_id": generation_id,
            "canonical_ids": list(canonical_ids),
            "maximum_ledger_seq": maximum_ledger_seq,
            "limit": limit,
        }
        try:
            with self._driver.session(database=self._config.database) as session:
                node_rows = session.execute_read(
                    lambda transaction: list(
                        transaction.run(_READ_NODES_QUERY, parameters)
                    )
                )
                relation_rows = session.execute_read(
                    lambda transaction: list(
                        transaction.run(_READ_RELATIONS_QUERY, parameters)
                    )
                )
            nodes: dict[str, StructuralGraphNodeView] = {}
            for row in node_rows:
                node = _node_view(_record_mapping(row, "properties"))
                _require_node_within_watermark(node, maximum_ledger_seq)
                nodes[node.canonical_id] = node
            relations: list[StructuralGraphRelationView] = []
            for row in relation_rows:
                source = _node_view(_record_mapping(row, "source_properties"))
                target = _node_view(_record_mapping(row, "target_properties"))
                _require_node_within_watermark(source, maximum_ledger_seq)
                _require_node_within_watermark(target, maximum_ledger_seq)
                nodes[source.canonical_id] = source
                nodes[target.canonical_id] = target
                relation_type = ProjectionRelationType(str(row["relation_type"]))
                relation = _relation_view(
                    _record_mapping(row, "relation_properties"),
                    relation_type=relation_type,
                )
                relations.append(relation)
        except (Neo4jReadError, ValueError):
            raise Neo4jReadError("Neo4j returned malformed structural projection data") from None
        except Exception:
            raise Neo4jReadError("Neo4j structural read failed") from None
        return Neo4jStructuralRead(
            nodes=tuple(nodes[key] for key in sorted(nodes)),
            relations=tuple(relations),
        )

    def reconcile_generation(
        self,
        *,
        generation_id: str,
        expected_batches: tuple[StructuralBatch, ...],
    ) -> str:
        """Compare one exact generation snapshot with retained SQLite authority."""

        self._require_open()
        expected_digest = _expected_projection_state_digest(
            generation_id, expected_batches
        )
        with self._lock:
            try:
                with self._driver.session(database=self._config.database) as session:
                    actual_digest = session.execute_read(
                        self._state_transaction,
                        generation_id,
                    )
            except Neo4jIdentityConflict:
                raise
            except Exception:
                raise Neo4jReadError(
                    "Neo4j generation reconciliation failed"
                ) from None
        if actual_digest != expected_digest:
            raise Neo4jIdentityConflict(
                "Neo4j generation state differs from retained authority"
            )
        return actual_digest

    @staticmethod
    def _state_transaction(
        transaction: Any,
        generation_id: str,
    ) -> str:
        def node_records():
            for row in transaction.run(
                _STATE_NODES_QUERY, {"generation_id": generation_id},
            ):
                yield (
                    _record_labels(row, "labels"),
                    _record_mapping(row, "properties"),
                )

        def relationship_records():
            for row in transaction.run(
                _STATE_RELATIONSHIPS_QUERY,
                {
                    "generation_id": generation_id,
                    "relation_types": [item.value for item in ProjectionRelationType],
                },
            ):
                yield (
                    _record_labels(row, "source_labels"),
                    _record_mapping(row, "source_properties"),
                    str(row["relation_type"]),
                    _record_mapping(row, "relation_properties"),
                    _record_labels(row, "target_labels"),
                    _record_mapping(row, "target_properties"),
                )

        # Consume and validate nodes before starting the next query, which would
        # otherwise buffer an unfinished driver result. Only the digest escapes.
        return _actual_projection_state_digest(
            generation_id,
            node_records=node_records(),
            relationship_records=relationship_records(),
        )

    def cleanup_generation(self, generation_id: str) -> int:
        """Reclaim one retired/disposable namespace in resumable transactions."""

        self._require_open()
        with self._lock:
            try:
                with self._driver.session(database=self._config.database) as session:
                    deleted = 0
                    while True:
                        # Promotion/retirement is already authoritative. Keep
                        # reclamation outside one generation-sized transaction;
                        # an interruption leaves only this namespace to resume.
                        record = session.execute_write(
                            lambda transaction: transaction.run(
                                _CLEANUP_GENERATION_QUERY,
                                {"generation_id": generation_id, "batch_size": 1_000},
                            ).single()
                        )
                        count = 0 if record is None else int(record["deleted_count"])
                        deleted += count
                        if count < 1_000:
                            return deleted
            except Exception:
                raise Neo4jWriteError("Neo4j generation cleanup failed") from None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._driver.close()

    def _require_open(self) -> None:
        if self._closed:
            raise Neo4jConnectionError("Neo4j projector adapter is closed")

    @staticmethod
    def _apply_transaction(transaction: Any, batch: StructuralBatch) -> Neo4jApplyOutcome:
        delivery_properties = _delivery_properties(batch)
        existing_delivery = transaction.run(
            _FIND_DELIVERY_QUERY,
            {
                "generation_id": str(batch.generation_id),
                "ledger_seq": batch.ledger_seq,
            },
        ).single()
        if existing_delivery is not None:
            _require_exact_properties(
                _record_mapping(existing_delivery, "properties"),
                delivery_properties,
                allowed_keys=_DELIVERY_PROPERTY_KEYS,
                identity="Neo4j delivery marker",
            )
            _apply_tombstone_cleanup(transaction, batch)
            return Neo4jApplyOutcome.DUPLICATE

        _apply_tombstone_cleanup(transaction, batch)
        node_by_id = {item.canonical_id: item for item in batch.nodes}
        for canonical_id in sorted(node_by_id):
            node = node_by_id[canonical_id]
            expected = _node_properties(batch, node)
            record = transaction.run(
                _MERGE_NODE_QUERY,
                {
                    "generation_id": str(batch.generation_id),
                    "canonical_id": node.canonical_id,
                    "properties": expected,
                },
            ).single()
            if record is None:
                raise Neo4jIdentityConflict("Neo4j node upsert returned no exact state")
            current = _record_mapping(record, "properties")
            _require_node_identity(current, expected)
            _require_node_properties(current)
            current_first = int(current["first_ledger_seq"])
            if node.first_ledger_seq == current_first:
                _require_same_sequence_node_provenance(current, expected)
            elif node.first_ledger_seq < current_first:
                updated = transaction.run(
                    _UPDATE_NODE_FIRST_PROVENANCE_QUERY,
                    {
                        "generation_id": str(batch.generation_id),
                        "canonical_id": node.canonical_id,
                        "first_ledger_seq": node.first_ledger_seq,
                        "first_source_event_id": node.first_source_event_id,
                        "first_source_event_digest": node.first_source_event_digest,
                    },
                ).single()
                if updated is None:
                    raise Neo4jIdentityConflict("Neo4j node provenance update returned no exact state")
                current = _record_mapping(updated, "properties")
                _require_node_properties(current)
                _require_same_sequence_node_provenance(current, expected)

        for relation in sorted(batch.relations, key=lambda value: value.relation_key):
            identity_properties = _relation_identity_properties(batch, relation)
            identity_record = transaction.run(
                _MERGE_RELATION_IDENTITY_QUERY,
                {
                    "generation_id": str(batch.generation_id),
                    "relation_key": relation.relation_key,
                    "properties": identity_properties,
                },
            ).single()
            if identity_record is None:
                raise Neo4jIdentityConflict("Neo4j relation identity returned no exact state")
            _require_exact_properties(
                _record_mapping(identity_record, "properties"),
                identity_properties,
                allowed_keys=_RELATION_IDENTITY_PROPERTY_KEYS,
                identity="Neo4j relation identity",
            )
            relation_properties = _relation_properties(batch, relation)
            relation_record = transaction.run(
                _RELATION_QUERIES[relation.relation_type],
                {
                    "generation_id": str(batch.generation_id),
                    "source_canonical_id": relation.source_canonical_id,
                    "target_canonical_id": relation.target_canonical_id,
                    "relation_key": relation.relation_key,
                    "properties": relation_properties,
                },
            ).single()
            if relation_record is None:
                raise Neo4jIdentityConflict("Neo4j relation endpoints are absent")
            _require_exact_properties(
                _record_mapping(relation_record, "properties"),
                relation_properties,
                allowed_keys=_RELATION_PROPERTY_KEYS,
                identity="Neo4j structural relation",
            )

        delivery_record = transaction.run(
            _CREATE_DELIVERY_QUERY,
            {"properties": delivery_properties},
        ).single()
        if delivery_record is None:
            raise Neo4jIdentityConflict("Neo4j delivery marker was not created")
        _require_exact_properties(
            _record_mapping(delivery_record, "properties"),
            delivery_properties,
            allowed_keys=_DELIVERY_PROPERTY_KEYS,
            identity="Neo4j delivery marker",
        )
        return Neo4jApplyOutcome.APPLIED


def _apply_tombstone_cleanup(transaction: Any, batch: StructuralBatch) -> None:
    if not batch.tombstoned_object_admission_ids:
        return
    record = transaction.run(
        _TOMBSTONE_RELATIONS_QUERY,
        {
            "generation_id": str(batch.generation_id),
            "relation_types": [item.value for item in ProjectionRelationType],
            "object_admission_ids": list(
                batch.tombstoned_object_admission_ids
            ),
        },
    ).single()
    if record is None:
        raise Neo4jIdentityConflict(
            "Neo4j tombstone cleanup returned no exact relation state"
        )
    try:
        relation_keys = tuple(
            sorted({str(item) for item in (record["relation_keys"] or [])})
        )
        canonical_ids = tuple(
            sorted({str(item) for item in (record["canonical_ids"] or [])})
        )
        int(record["deleted_count"])
    except Exception:
        raise Neo4jIdentityConflict(
            "Neo4j tombstone cleanup returned malformed relation state"
        ) from None
    if relation_keys:
        identities = transaction.run(
            _TOMBSTONE_RELATION_IDENTITIES_QUERY,
            {
                "generation_id": str(batch.generation_id),
                "relation_keys": list(relation_keys),
            },
        ).single()
        if identities is None or int(identities["deleted_count"]) != len(
            relation_keys
        ):
            raise Neo4jIdentityConflict(
                "Neo4j tombstone relation identity state is incomplete"
            )
    if canonical_ids:
        orphans = transaction.run(
            _TOMBSTONE_ORPHAN_NODES_QUERY,
            {
                "generation_id": str(batch.generation_id),
                "canonical_ids": list(canonical_ids),
            },
        ).single()
        if orphans is None:
            raise Neo4jIdentityConflict(
                "Neo4j tombstone orphan cleanup returned no exact state"
            )


def _open_neo4j_driver(config: Neo4jProjectorConfig) -> tuple[Any, str]:
    """Open the official driver only inside this one private module."""

    try:
        import neo4j
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password),
        )
    except Exception:
        raise Neo4jConnectionError("Neo4j projector driver creation failed") from None
    return driver, str(neo4j.__version__)


def _open_neo4j_adapter(config: Neo4jProjectorConfig) -> _Neo4jAdapter:
    driver, driver_version = _open_neo4j_driver(config)
    return _Neo4jAdapter(
        driver=driver,
        config=config,
        driver_version=driver_version,
    )


def _neo4j_unit_of_work_factory() -> Any:
    """Return the official managed-transaction decorator from the one driver seam."""

    try:
        from neo4j import unit_of_work
    except Exception:
        raise Neo4jConnectionError(
            "Neo4j managed-transaction support is unavailable"
        ) from None
    return unit_of_work



def _record_labels(record: Any, key: str) -> tuple[str, ...]:
    try:
        value = record[key]
    except Exception:
        raise Neo4jReadError(
            "Neo4j record is missing fixed projection labels"
        ) from None
    if isinstance(value, (str, bytes)):
        raise Neo4jReadError("Neo4j projection labels are malformed")
    try:
        return tuple(str(item) for item in value)
    except Exception:
        raise Neo4jReadError("Neo4j projection labels are malformed") from None


def _record_mapping(record: Any, key: str) -> Mapping[str, object]:
    try:
        value = record[key]
    except Exception:
        raise Neo4jReadError("Neo4j record is missing fixed projection properties") from None
    if not isinstance(value, Mapping):
        try:
            value = dict(value)
        except Exception:
            raise Neo4jReadError("Neo4j projection properties are malformed") from None
    return dict(value)


def _require_exact_properties(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    allowed_keys: frozenset[str],
    identity: str,
) -> None:
    if set(actual) != set(allowed_keys) or dict(actual) != dict(expected):
        raise Neo4jIdentityConflict(f"{identity} belongs to another exact projection state")


def _require_node_identity(actual: Mapping[str, object], expected: Mapping[str, object]) -> None:
    if set(actual) != set(_NODE_PROPERTY_KEYS):
        raise Neo4jIdentityConflict("Neo4j node contains properties outside the fixed contract")
    stable_keys = _NODE_PROPERTY_KEYS - {
        "first_ledger_seq",
        "first_source_event_id",
        "first_source_event_digest",
    }
    if any(actual.get(key) != expected.get(key) for key in stable_keys):
        raise Neo4jIdentityConflict("Neo4j canonical node belongs to another exact identity")


def _require_same_sequence_node_provenance(
    actual: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    provenance_keys = {
        "first_ledger_seq",
        "first_source_event_id",
        "first_source_event_digest",
    }
    if any(actual.get(key) != expected.get(key) for key in provenance_keys):
        raise Neo4jIdentityConflict(
            "Neo4j canonical node has conflicting provenance at the same sequence"
        )


def _require_node_properties(actual: Mapping[str, object]) -> None:
    if set(actual) != set(_NODE_PROPERTY_KEYS):
        raise Neo4jIdentityConflict("Neo4j node contains properties outside the fixed contract")
    try:
        ProjectionNodeType(str(actual["entity_type"]))
        int(actual["first_ledger_seq"])
    except Exception:
        raise Neo4jIdentityConflict("Neo4j node contains malformed fixed properties") from None


def _node_view(properties: Mapping[str, object]) -> StructuralGraphNodeView:
    if set(properties) != set(_NODE_PROPERTY_KEYS):
        raise Neo4jReadError("Neo4j node properties do not match the fixed contract")
    return StructuralGraphNodeView(
        canonical_id=str(properties["canonical_id"]),
        node_type=ProjectionNodeType(str(properties["entity_type"])),
        identity_source=str(properties["identity_source"]),
        identity_reference_digest=str(properties["identity_reference_digest"]),
        first_ledger_seq=int(properties["first_ledger_seq"]),
        first_source_event_id=str(properties["first_source_event_id"]),
        first_source_event_digest=str(properties["first_source_event_digest"]),
    )


def _require_node_within_watermark(
    node: StructuralGraphNodeView,
    maximum_ledger_seq: int,
) -> None:
    if node.first_ledger_seq > maximum_ledger_seq:
        raise Neo4jReadError(
            "Neo4j node provenance exceeds the authoritative watermark"
        )


def _relation_view(
    properties: Mapping[str, object],
    *,
    relation_type: ProjectionRelationType,
) -> StructuralGraphRelationView:
    if set(properties) != set(_RELATION_PROPERTY_KEYS):
        raise Neo4jReadError("Neo4j relation properties do not match the fixed contract")
    if str(properties["relation_type"]) != relation_type.value:
        raise Neo4jReadError("Neo4j relation type conflicts with retained properties")
    return StructuralGraphRelationView(
        relation_key=str(properties["relation_key"]),
        relation_type=relation_type,
        source_canonical_id=str(properties["source_canonical_id"]),
        target_canonical_id=str(properties["target_canonical_id"]),
        ledger_seq=int(properties["ledger_seq"]),
        source_event_id=str(properties["source_event_id"]),
        source_event_type=str(properties["source_event_type"]),
        source_event_digest=str(properties["source_event_digest"]),
        aggregate_type=str(properties["aggregate_type"]),
        aggregate_id=str(properties["aggregate_id"]),
        aggregate_version=int(properties["aggregate_version"]),
        payload_id=str(properties["payload_id"]),
        payload_digest=str(properties["payload_digest"]),
        object_admission_id=(
            None
            if properties["object_admission_id"] == ""
            else str(properties["object_admission_id"])
        ),
        principal_id=str(properties["principal_id"]),
        trust_scope=TrustScope(str(properties["trust_scope"])),
        security_scope=str(properties["security_scope"]),
        retention_scope=str(properties["retention_scope"]),
        recorded_at=UtcTimestamp.parse(str(properties["recorded_at"])),
    )


__all__ = [
    "_Neo4jAdapter",
    "_expected_projection_state_digest",
    "_neo4j_unit_of_work_factory",
    "_node_properties",
    "_open_neo4j_adapter",
    "_relation_properties",
]
