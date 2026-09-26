from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from newsroom.projection import ProjectionNodeType
from newsroom.projection.neo4j import (
    Neo4jIdentityConflict,
    Neo4jReadError,
    Neo4jWriteError,
    StructuralGraphNodeView,
)
from newsroom.projection.neo4j._adapter import (
    _CLEANUP_GENERATION_QUERY,
    _FIND_DELIVERY_QUERY,
    _MERGE_NODE_QUERY,
    _READ_RELATIONS_QUERY,
    _Neo4jAdapter,
    _node_properties,
    _require_node_within_watermark,
    _relation_properties,
    _relation_view,
)

from .projection_b2_helpers import structural_batch


class _SingleResult:
    def __init__(self, record: dict[str, object] | None) -> None:
        self._record = record

    def single(self) -> dict[str, object] | None:
        return self._record


class _SameSequenceTamperTransaction:
    def run(
        self,
        statement: str,
        parameters: dict[str, object] | None = None,
    ) -> _SingleResult:
        values = parameters or {}
        if statement == _FIND_DELIVERY_QUERY:
            return _SingleResult(None)
        if statement == _MERGE_NODE_QUERY:
            properties = dict(values["properties"])
            properties["first_source_event_id"] = "tampered-event"
            return _SingleResult({"properties": properties})
        raise AssertionError("same-sequence conflict must fail before later writes")


def test_none_object_admission_uses_explicit_storage_sentinel() -> None:
    batch = structural_batch(object_admission_id=None)
    relation = batch.relations[0]
    properties = _relation_properties(batch, relation)

    assert properties["object_admission_id"] == ""
    assert all(value is not None for value in properties.values())

    view = _relation_view(properties, relation_type=relation.relation_type)
    assert view.object_admission_id is None


def test_present_object_admission_round_trips_without_rewriting() -> None:
    batch = structural_batch(object_admission_id="admission-b2-fixture")
    relation = batch.relations[0]
    properties = _relation_properties(batch, relation)

    assert properties["object_admission_id"] == "admission-b2-fixture"
    view = _relation_view(properties, relation_type=relation.relation_type)
    assert view.object_admission_id == "admission-b2-fixture"


def test_same_sequence_node_provenance_conflict_fails_before_relation_write() -> None:
    batch = structural_batch()
    with pytest.raises(Neo4jIdentityConflict, match="same sequence"):
        _Neo4jAdapter._apply_transaction(
            _SameSequenceTamperTransaction(),
            batch,
        )


def test_cleanup_is_limited_to_repository_owned_projection_labels() -> None:
    assert "value:NewsroomProjectionNode" in _CLEANUP_GENERATION_QUERY
    assert "value:NewsroomProjectionDelivery" in _CLEANUP_GENERATION_QUERY
    assert "value:NewsroomProjectionRelationIdentity" in _CLEANUP_GENERATION_QUERY
    assert "AND (value:" in _CLEANUP_GENERATION_QUERY
    assert "MATCH (value)\nWHERE value.generation_id" in _CLEANUP_GENERATION_QUERY


@pytest.mark.parametrize("interrupt", [False, True])
def test_cleanup_commits_bounded_batches_and_resumes_after_interruption(interrupt) -> None:
    # A retired generation is already outside the active read namespace. Its
    # cleanup must not retain the entire generation in one Neo4j transaction.
    remaining = 2_501
    committed = []
    fail_once = interrupt

    def run(statement, parameters):
        nonlocal remaining, fail_once
        assert parameters["generation_id"] == "retired-generation"
        assert "WITH value LIMIT $batch_size" in statement
        assert statement.index("LIMIT $batch_size") < statement.index("DETACH DELETE")
        assert 0 < parameters["batch_size"] <= 1_000
        if committed and fail_once:
            fail_once = False
            raise RuntimeError("interrupted between committed batches")
        deleted = min(remaining, parameters["batch_size"])
        remaining -= deleted
        committed.append(deleted)
        return _SingleResult({"deleted_count": deleted})

    session = SimpleNamespace(execute_write=lambda callback: callback(SimpleNamespace(run=run)))
    adapter = _Neo4jAdapter(
        driver=SimpleNamespace(session=lambda **kwargs: nullcontext(session)),
        config=SimpleNamespace(database="disposable"),
        driver_version="test",
    )
    if interrupt:
        with pytest.raises(Neo4jWriteError, match="cleanup failed"):
            adapter.cleanup_generation("retired-generation")
        assert remaining == 1_501
    expected_remaining = remaining
    assert adapter.cleanup_generation("retired-generation") == expected_remaining
    assert remaining == 0
    assert sum(committed) == 2_501
    assert max(committed) == 1_000
    assert adapter.cleanup_generation("retired-generation") == 0


def test_node_properties_never_include_driver_internal_identity() -> None:
    batch = structural_batch()
    properties = _node_properties(batch, batch.nodes[0])
    assert "id" not in properties
    assert "element_id" not in properties
    assert "neo4j_id" not in properties


def test_relation_reads_cannot_return_future_endpoint_nodes() -> None:
    assert "source.first_ledger_seq <= $maximum_ledger_seq" in _READ_RELATIONS_QUERY
    assert "target.first_ledger_seq <= $maximum_ledger_seq" in _READ_RELATIONS_QUERY
    node = StructuralGraphNodeView(
        canonical_id="npid:v1:fixture:future",
        node_type=ProjectionNodeType.AUTHORITY_AGGREGATE,
        identity_source="AUTHORITY_AGGREGATE",
        identity_reference_digest="sha256:" + "a" * 64,
        first_ledger_seq=11,
        first_source_event_id="event-future",
        first_source_event_digest="sha256:" + "b" * 64,
    )
    with pytest.raises(Neo4jReadError, match="watermark"):
        _require_node_within_watermark(node, 10)


def _state_rows(batch):
    from newsroom.projection.neo4j._state import (
        _delivery_properties, _relation_identity_properties,
    )

    nodes = [
        {"labels": ["NewsroomProjectionNode"], "properties": _node_properties(batch, node)}
        for node in batch.nodes
    ]
    nodes.append({"labels": ["NewsroomProjectionDelivery"], "properties": _delivery_properties(batch)})
    by_id = {row["properties"]["canonical_id"]: row["properties"] for row in nodes[:-1]}
    relationships = []
    for relation in batch.relations:
        nodes.append({
            "labels": ["NewsroomProjectionRelationIdentity"],
            "properties": _relation_identity_properties(batch, relation),
        })
        relationships.append({
            "source_labels": ["NewsroomProjectionNode"],
            "source_properties": by_id[relation.source_canonical_id],
            "relation_type": relation.relation_type.value,
            "relation_properties": _relation_properties(batch, relation),
            "target_labels": ["NewsroomProjectionNode"],
            "target_properties": by_id[relation.target_canonical_id],
        })
    return nodes, relationships


class _RetryStateRead(Exception):
    pass


class _StreamingStateTransaction:
    """Driver records must be decoded before advancing or leaving the callback."""

    def __init__(self, nodes, relationships, *, fail_after_relationship=False):
        self.nodes, self.relationships = nodes, relationships
        self.fail_after_relationship = fail_after_relationship
        self.active = False
        self.result_open = False
        self.queries = []
        self.decoded_rows = 0

    def run(self, statement, parameters):
        from newsroom.projection.neo4j._adapter import (
            _STATE_NODES_QUERY, _STATE_RELATIONSHIPS_QUERY,
        )

        assert self.active
        assert not self.result_open, "a second query would buffer the unfinished result"
        assert statement in (_STATE_NODES_QUERY, _STATE_RELATIONSHIPS_QUERY)
        self.queries.append(statement)
        self.result_open = True
        owner = self
        values = self.nodes if statement == _STATE_NODES_QUERY else self.relationships

        class Record(dict):
            def __init__(self, value):
                super().__init__(value)
                self.read_keys = set()

            def __getitem__(self, key):
                assert owner.active, "raw record escaped the managed transaction"
                self.read_keys.add(key)
                return super().__getitem__(key)

        class Result:
            def __len__(self):
                raise AssertionError("raw results must not be materialised")

            def __iter__(self):
                for value in values:
                    record = Record(value)
                    yield record
                    assert record.read_keys == set(record), "decode each row before advancing"
                    owner.decoded_rows += 1
                if statement == _STATE_RELATIONSHIPS_QUERY and owner.fail_after_relationship:
                    raise _RetryStateRead("late transport failure")
                owner.result_open = False

        return Result()


class _StateDriver:
    def __init__(self, *transactions):
        self.transactions = transactions
        self.returned = []

    def session(self, *, database):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute_read(self, callback, *args):
        for index, transaction in enumerate(self.transactions):
            transaction.active = True
            try:
                value = callback(transaction, *args)
                self.returned.append(value)
                return value
            except _RetryStateRead:
                if index == len(self.transactions) - 1:
                    raise
            finally:
                transaction.active = False


def _state_adapter(driver):
    from newsroom.projection.neo4j import Neo4jProjectorConfig

    return _Neo4jAdapter(
        driver=driver,
        config=Neo4jProjectorConfig(
            uri="bolt://localhost:7687", database="neo4j",
            username="fixture", password="fixture",
        ),
        driver_version="fixture", unit_of_work_factory=lambda **_: lambda callback: callback,
    )


@pytest.mark.parametrize("empty", (False, True))
def test_generation_state_streams_raw_rows_inside_one_managed_transaction(empty):
    from newsroom.projection.neo4j._state import _expected_projection_state_digest
    from newsroom.projection.neo4j._adapter import _STATE_NODES_QUERY, _STATE_RELATIONSHIPS_QUERY

    batch = structural_batch()
    nodes, relationships = ([], []) if empty else _state_rows(batch)
    transaction = _StreamingStateTransaction(list(reversed(nodes)), relationships)
    driver = _StateDriver(transaction)
    batches = () if empty else (batch,)
    expected = _expected_projection_state_digest(str(batch.generation_id), batches)
    assert _state_adapter(driver).reconcile_generation(
        generation_id=str(batch.generation_id), expected_batches=batches,
    ) == expected
    assert driver.returned == [expected]
    assert transaction.decoded_rows == len(nodes) + len(relationships)
    assert transaction.queries == [_STATE_NODES_QUERY, _STATE_RELATIONSHIPS_QUERY]
    assert not transaction.active and not transaction.result_open


@pytest.mark.parametrize("tamper, error, match", [
    ("late_labels", Neo4jReadError, "reconciliation failed"),
    ("late_properties", Neo4jReadError, "reconciliation failed"),
    ("duplicate_node", Neo4jIdentityConflict, "duplicate structural identity"),
    ("duplicate_delivery", Neo4jIdentityConflict, "duplicate structural identity"),
    ("duplicate_relation_identity", Neo4jIdentityConflict, "duplicate structural identity"),
    ("duplicate_relationship", Neo4jIdentityConflict, "duplicate relationship identity"),
    ("wrong_generation", Neo4jIdentityConflict, "another generation"),
    ("wrong_endpoint", Neo4jIdentityConflict, "endpoints differ"),
    ("endpoint_state", Neo4jIdentityConflict, "endpoint state differs"),
    ("digest_mismatch", Neo4jIdentityConflict, "state differs from retained authority"),
])
def test_streamed_generation_state_rejects_late_corruption(tamper, error, match):
    from copy import deepcopy

    batch = structural_batch()
    nodes, relationships = deepcopy(_state_rows(batch))
    if tamper == "late_labels":
        nodes.append({"labels": "not-labels", "properties": {}})
    elif tamper == "late_properties":
        relationships.append({**relationships[0], "target_properties": None})
    elif tamper.startswith("duplicate_"):
        if tamper == "duplicate_relationship":
            relationships.append(deepcopy(relationships[0]))
        else:
            index = {"duplicate_node": 0, "duplicate_delivery": 2, "duplicate_relation_identity": 3}[tamper]
            nodes.append(deepcopy(nodes[index]))
    elif tamper == "wrong_generation":
        nodes[-1]["properties"]["generation_id"] = "another-generation"
    elif tamper == "wrong_endpoint":
        relationships[0]["relation_properties"]["target_canonical_id"] = "another-endpoint"
    elif tamper == "endpoint_state":
        relationships[0]["target_properties"] = {
            **relationships[0]["target_properties"], "identity_source": "different",
        }
    else:
        relationships[0]["relation_properties"]["principal_id"] = "different"
    driver = _StateDriver(_StreamingStateTransaction(nodes, relationships))
    with pytest.raises(error, match=match):
        _state_adapter(driver).reconcile_generation(
            generation_id=str(batch.generation_id), expected_batches=(batch,),
        )


def test_streamed_generation_retry_discards_partially_decoded_attempt():
    from newsroom.projection.neo4j._state import _expected_projection_state_digest

    batch = structural_batch()
    nodes, relationships = _state_rows(batch)
    first = _StreamingStateTransaction(nodes, relationships, fail_after_relationship=True)
    second = _StreamingStateTransaction(list(reversed(nodes)), relationships)
    driver = _StateDriver(first, second)
    expected = _expected_projection_state_digest(str(batch.generation_id), (batch,))
    assert _state_adapter(driver).reconcile_generation(
        generation_id=str(batch.generation_id), expected_batches=(batch,),
    ) == expected
    assert first.decoded_rows == second.decoded_rows == len(nodes) + len(relationships)
    assert driver.returned == [expected]
    assert not first.active and not second.active
