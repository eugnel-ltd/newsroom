from __future__ import annotations

import atexit
import asyncio
from functools import lru_cache
import os
from pathlib import Path
import sqlite3
import uuid
from unittest.mock import Mock

import pytest
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession, EagerResult

from newsroom.increment4 import (
    Increment4Neo4jActiveReadRequest,
    Increment4Neo4jBuildRequest,
    build_increment4_admitted_batches,
    increment4_admitted_contract_registry,
    sorted_snapshot,
)
from newsroom.projection import ProjectionGenerationId, ProjectionGenerationState
from newsroom.projection.neo4j import Neo4jIdentityConflict, Neo4jProjectorConfig
from newsroom.projection.neo4j._adapter import _open_neo4j_adapter

from .editorial_relation_4c_helpers import (
    RELATION_PROPOSAL_ID,
    RELATION_PROPOSAL_V1_ID,
)
from .authority_a2b_helpers import open_object_system
from .authority_helpers import proof as object_proof
from .extraction_4a_helpers import extraction_proof
from .increment4e_governed_path_helpers import (
    admit_increment4_graphiti_path,
    graphiti_path_registries,
    open_graphiti_path_increment4_neo4j_system,
    seed_increment4_graphiti_path,
)
from .increment4e_helpers import _ledger_events
from .source_3a_helpers import SOURCE_NOW


_REQUIRED_FLAG = "NEWSROOM_NEO4J_SERVICE_REQUIRED"
_PROJECTOR_DRIVER = None


class _DatabaseBoundAsyncDriver:
    def __init__(self, driver: AsyncDriver, *, database: str) -> None:
        self._database = database
        self._driver = driver

    async def execute_query(
        self,
        cypher: str,
        *,
        params: dict[str, object],
        routing_: str,
    ) -> EagerResult:
        return await self._driver.execute_query(
            cypher,
            parameters_=params,
            routing_=routing_,
            database_=self._database,
        )

    def session(self) -> AsyncSession:
        return self._driver.session(database=self._database)

    async def close(self) -> None:
        await self._driver.close()


def test_database_bound_driver_streaming_session_uses_configured_database() -> None:
    driver = Mock(spec=AsyncDriver)
    bound = _DatabaseBoundAsyncDriver(driver, database="guard-test-database")

    assert bound.session() is driver.session.return_value
    driver.session.assert_called_once_with(database="guard-test-database")


def test_actual_guard_restores_multiple_pages_and_duplicate_targets() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import Neo4jMutationGuard

    async def exercise() -> None:
        config = _service_config()
        suffix = str(uuid.uuid4())
        group = f"newsroom-paged-guard-{suffix}"
        identifiers = [f"paged-{suffix}-{index}" for index in range(257)]
        driver = _DatabaseBoundAsyncDriver(
            AsyncGraphDatabase.driver(config.uri, auth=(config.username, config.password)),
            database=config.database,
        )

        async def query(cypher: str, **parameters: object) -> EagerResult:
            return await driver.execute_query(cypher, params=parameters, routing_="w")

        try:
            await query(
                "UNWIND $ids AS id CREATE (:Entity {uuid:id,group_id:$group,name:'original',"
                "vector:[0.125,0.25,0.5],counter:7})",
                ids=identifiers, group=group,
            )
            await query(
                "UNWIND range(0,size($ids)-2) AS i "
                "MATCH (a {uuid:$ids[i]}),(b {uuid:$ids[i+1]}) "
                "CREATE (a)-[:REL {uuid:$ids[i]+'-edge',fact:'original',counter:7}]->(b)",
                ids=identifiers,
            )
            guard = Neo4jMutationGuard(
                driver, group_id=group, episode_uuid=f"episode-{suffix}",
                attempt_number=1, input_digest="sha256:" + "1" * 64,
            )
            await guard.begin()
            async with guard.fenced_graph_mutation():
                await query(
                    "MATCH (n:Entity {group_id:$group}) "
                    "SET n.name='changed',n.extra='remove',n.group_id='changed-'+$group "
                    "REMOVE n:Entity SET n:Changed",
                    group=group,
                )
                await query(
                    "MATCH (a {uuid:$id})-[r:REL]->() SET r.fact='changed',r.extra='remove' "
                    "WITH count(r) AS changed UNWIND range(1,65) AS i "
                    "CREATE (:Changed {uuid:$id,name:'duplicate',group_id:'foreign-'+$group})",
                    id=identifiers[0], group=group,
                )
                await query(
                    "CREATE (:Entity {uuid:$new,group_id:$group,name:'retained-new'})",
                    new=f"new-{suffix}", group=group,
                )
                await guard.restore_preexisting()
                nodes, _, _ = await query(
                    "MATCH (n) WHERE n.uuid IN $ids AND NOT n:NewsroomSnapshotNode "
                    "AND NOT n:NewsroomSnapshotRelationship AND NOT n:NewsroomIngestMarker "
                    "RETURN properties(n) AS props,labels(n) AS labels",
                    ids=identifiers,
                )
                assert len(nodes) == 322
                for row in nodes:
                    assert row["props"] == {
                        "uuid": row["props"]["uuid"], "group_id": group,
                        "name": "original", "vector": [0.125, 0.25, 0.5], "counter": 7,
                    }
                    assert row["labels"] == ["Entity"]
                relationships, _, _ = await query(
                    "MATCH (a)-[r:REL]->(b) WHERE r.uuid IN $edges "
                    "RETURN properties(r) AS props,a.uuid AS source,b.uuid AS target",
                    edges=[identifier + "-edge" for identifier in identifiers[:-1]],
                )
                assert len(relationships) == 256
                for row in relationships:
                    assert row["props"] == {"uuid": row["source"] + "-edge", "fact": "original", "counter": 7}
                new, _, _ = await query(
                    "MATCH (n {uuid:$id}) RETURN n.name AS name", id=f"new-{suffix}",
                )
                assert new[0]["name"] == "retained-new"
            await guard.complete({"provider_attempt_number": 1})
        finally:
            await query(
                "MATCH (n) WHERE n.group_id=$group OR n.uuid IN $ids OR n.uuid=$new "
                "OR n.episode_uuid=$episode OR n._newsroom_snapshot_id=$snapshot DETACH DELETE n",
                group=group, ids=identifiers, new=f"new-{suffix}",
                episode=f"episode-{suffix}", snapshot=f"episode-{suffix}:1",
            )
            await driver.close()

    asyncio.run(exercise())


def test_actual_guard_expiry_keeps_generation_owned_until_recovery() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import GuardError, GuardState, Neo4jMutationGuard

    async def exercise() -> None:
        config = _service_config()
        suffix = str(uuid.uuid4())
        group, episode = f"newsroom-exclusive-guard-{suffix}", f"episode-{suffix}"
        driver = _DatabaseBoundAsyncDriver(
            AsyncGraphDatabase.driver(config.uri, auth=(config.username, config.password)),
            database=config.database,
        )

        async def query(cypher: str, **parameters: object) -> EagerResult:
            return await driver.execute_query(cypher, params=parameters, routing_="w")

        def guard(selected_episode: str) -> Neo4jMutationGuard:
            return Neo4jMutationGuard(
                driver, group_id=group, episode_uuid=selected_episode,
                attempt_number=1, input_digest="sha256:" + "1" * 64,
            )

        try:
            await query("CREATE (:Entity {uuid:$id,group_id:$group,name:'original'})", id=suffix, group=group)
            first = guard(episode)
            await first.begin()
            await query("MATCH (n:Entity {uuid:$id}) SET n.name='partial'", id=suffix)
            await query(
                "MATCH (m:NewsroomIngestMarker {episode_uuid:$episode}) "
                "SET m.claim_expires_at=datetime()-duration('PT16M')",
                episode=episode,
            )
            competitor = guard(f"competing-{suffix}")
            with pytest.raises(GuardError):
                await competitor.begin()
            recovery = guard(episode)
            retained = await recovery.begin()
            assert retained.state in {GuardState.PENDING, GuardState.ROLLING_BACK}
            assert await recovery.rollback_pending(
                chat_invocations=[],
                embedding_usage={"usage_basis": "NO_EMBEDDING_CALL", "request_count": 0},
                reason="ACTUAL_EXPIRED_OWNER_RECOVERY",
            )
            rows, _, _ = await query("MATCH (n:Entity {uuid:$id}) RETURN n.name AS name", id=suffix)
            assert rows[0]["name"] == "original"
            assert (await competitor.begin()).state is GuardState.CREATED
            await competitor.rollback_pending(
                chat_invocations=[],
                embedding_usage={"usage_basis": "NO_EMBEDDING_CALL", "request_count": 0},
                reason="ACTUAL_COMPETITOR_CLEANUP",
            )
            legacy_episode = f"legacy-{suffix}"
            legacy_snapshot = legacy_episode + ":1"
            await query(
                "CREATE (:NewsroomIngestMarker {episode_uuid:$episode,group_id:$group,"
                "attempt_number:1,input_digest:$digest,snapshot_id:$snapshot,state:'PENDING',"
                "claim_token:'legacy',claim_expires_at:datetime()-duration('PT16M')}) "
                "CREATE (:NewsroomSnapshotNode {uuid:$id,group_id:$group,name:'legacy-snapshot',"
                "_newsroom_snapshot_id:$snapshot,_newsroom_source_uuid:$id,"
                "_newsroom_source_labels:['Entity']})",
                episode=legacy_episode, group="foreign-" + group, id=suffix,
                snapshot=legacy_snapshot, digest="sha256:" + "1" * 64,
            )
            fresh = guard(f"fresh-{suffix}")
            assert (await fresh.begin()).state is GuardState.CREATED
            await fresh.restore_preexisting()
            await fresh.complete({"provider_attempt_number": 1})
            with pytest.raises(GuardError):
                await guard(legacy_episode).begin()
            alias = Neo4jMutationGuard(
                driver, group_id=group, episode_uuid=legacy_episode, attempt_number=1,
                marker_episode_uuid=legacy_episode + ":attempt:1",
                input_digest="sha256:" + "1" * 64,
            )
            with pytest.raises(GuardError):
                await alias.begin()
            legacy, _, _ = await query(
                "MATCH (s:NewsroomSnapshotNode {_newsroom_snapshot_id:$snapshot}) "
                "RETURN s.name AS name", snapshot=legacy_snapshot,
            )
            assert [row["name"] for row in legacy] == ["legacy-snapshot"]
        finally:
            await query(
                "MATCH (n) WHERE n.group_id=$group OR n.uuid=$id "
                "OR n.episode_uuid=$legacy "
                "OR n._newsroom_snapshot_id IN $snapshots DETACH DELETE n",
                group=group, id=suffix,
                legacy=f"legacy-{suffix}",
                snapshots=[episode + ":1", f"competing-{suffix}:1"],
            )
            await driver.close()

    asyncio.run(exercise())


@lru_cache(maxsize=1)
def _service_config() -> Neo4jProjectorConfig:
    if os.environ.get(_REQUIRED_FLAG) != "1":
        pytest.skip("actual Neo4j service is required only by the permanent graph gate")
    return Neo4jProjectorConfig.from_environment()


def _projector_driver():
    global _PROJECTOR_DRIVER
    if _PROJECTOR_DRIVER is None:
        from neo4j import GraphDatabase

        config = _service_config()
        _PROJECTOR_DRIVER = GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password),
        )
        _PROJECTOR_DRIVER.verify_connectivity()
    return _PROJECTOR_DRIVER


def _close_projector_driver() -> None:
    global _PROJECTOR_DRIVER
    if _PROJECTOR_DRIVER is not None:
        _PROJECTOR_DRIVER.close()
        _PROJECTOR_DRIVER = None


atexit.register(_close_projector_driver)


def _request(generation_id, snapshot, *, key: str, purge: bool = True):
    return Increment4Neo4jBuildRequest(
        generation_id=generation_id,
        snapshot=snapshot,
        reason_code="INCREMENT4E_ACTUAL_NEO4J_PROOF",
        idempotency_key=key,
        purge_retired_generation=purge,
    )


def _expected(snapshot, generation_id):
    family = increment4_admitted_contract_registry().family(
        "graph.increment4.admitted"
    )
    batches = build_increment4_admitted_batches(
        snapshot,
        generation_id=generation_id,
        family=family,
    )
    node_ids = {
        node.canonical_id
        for batch in batches
        for node in batch.nodes
    }
    # Structural reads are explicit-ID bounded and expand only one relationship hop.
    # Bind the exact admitted generation inventory when asserting complete state.
    canonical_ids = tuple(sorted(node_ids))
    relation_keys = {
        relation.relation_key
        for batch in batches
        for relation in batch.relations
    }
    return batches, canonical_ids, node_ids, relation_keys


def _scalar(statement: str, **parameters) -> int:
    with _projector_driver().session(database=_service_config().database) as session:
        record = session.run(statement, parameters).single(strict=True)
        return int(record[0])


def _values(statement: str, **parameters) -> tuple[str, ...]:
    with _projector_driver().session(database=_service_config().database) as session:
        return tuple(str(record[0]) for record in session.run(statement, parameters))


def _cleanup(
    config: Neo4jProjectorConfig,
    *generation_ids: ProjectionGenerationId,
) -> None:
    adapter = _open_neo4j_adapter(config)
    try:
        adapter.verify_compatibility()
        for generation_id in generation_ids:
            adapter.cleanup_generation(str(generation_id))
    finally:
        adapter.close()


def _event_count(path: Path) -> int:
    with sqlite3.connect(path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM ledger_events").fetchone()[0])


def test_actual_service_guard_rollback_does_not_promote_snapshot_copies() -> None:
    from newsroom.graphiti_adapter.neo4j_guard import Neo4jMutationGuard

    async def exercise() -> None:
        config = _service_config()
        suffix = str(uuid.uuid4())
        group_id = f"newsroom-guard-service-{suffix}"
        episode_uuid = f"episode-{suffix}"
        snapshot_id = f"{episode_uuid}:1"
        driver = _DatabaseBoundAsyncDriver(
            AsyncGraphDatabase.driver(
                config.uri,
                auth=(config.username, config.password),
            ),
            database=config.database,
        )

        async def query(cypher: str, **parameters: object) -> EagerResult:
            return await driver.execute_query(
                cypher, params=parameters, routing_="w"
            )

        try:
            await query(
                """
                CREATE (a:Entity {uuid:$a, group_id:$group_id, name:'before'}),
                       (b:Entity {uuid:$b, group_id:$group_id, name:'other'}),
                       (a)-[:REL {uuid:$relationship, fact:'before'}]->(b)
                """,
                a=f"a-{suffix}",
                b=f"b-{suffix}",
                relationship=f"relationship-{suffix}",
                group_id=group_id,
            )
            guard = Neo4jMutationGuard(
                driver,
                group_id=group_id,
                episode_uuid=episode_uuid,
                attempt_number=1,
                input_digest="sha256:" + "1" * 64,
            )
            await guard.begin()
            await query(
                """
                MATCH (a {uuid:$a})-[r]->()
                SET a.name='after', r.fact='after'
                CREATE (:Entity {uuid:$new, group_id:$group_id, name:'new'})
                """,
                a=f"a-{suffix}",
                new=f"new-{suffix}",
                group_id=group_id,
            )
            await guard.rollback_pending(
                chat_invocations=[],
                embedding_usage={
                    "usage_basis": "NO_EMBEDDING_CALL",
                    "request_count": 0,
                },
                reason="SERVICE_TEST",
            )
            result = await query(
                """
                MATCH (n {group_id:$group_id})
                WHERE n.uuid IS NOT NULL
                OPTIONAL MATCH (n)-[r]->()
                RETURN collect(DISTINCT [n.uuid,n.name]) AS nodes,
                       collect(DISTINCT [r.uuid,r.fact]) AS relationships
                """,
                group_id=group_id,
            )
            snapshots = await query(
                """
                MATCH (s)
                WHERE s._newsroom_snapshot_id=$snapshot_id
                   OR s:NewsroomSnapshotNode
                      AND s.group_id=$group_id
                   OR s:NewsroomSnapshotRelationship
                      AND s.uuid=$relationship
                RETURN count(s) AS count
                """,
                snapshot_id=snapshot_id,
                group_id=group_id,
                relationship=f"relationship-{suffix}",
            )
            record = result.records[0]
            assert sorted(record["nodes"]) == sorted(
                [[f"a-{suffix}", "before"], [f"b-{suffix}", "other"]]
            )
            assert [
                item for item in record["relationships"] if item[0] is not None
            ] == [[f"relationship-{suffix}", "before"]]
            assert int(snapshots.records[0]["count"]) == 0
        finally:
            try:
                await query(
                    """
                    MATCH (n)
                    WHERE n.group_id=$group_id
                       OR n.episode_uuid=$episode_uuid
                       OR n._newsroom_snapshot_id=$snapshot_id
                    DETACH DELETE n
                    """,
                    group_id=group_id,
                    episode_uuid=episode_uuid,
                    snapshot_id=snapshot_id,
                )
            finally:
                await driver.close()

    asyncio.run(exercise())



@pytest.mark.parametrize("case", [
    "empty", "unchanged", "duplicate-node", "duplicate-relationship",
    "missing-node-duplicate-survivor", "missing-relationship-duplicate-survivor",
    "corrupt-duplicate-node", "corrupt-duplicate-relationship", "wrong-endpoint", "wrong-type",
])
def test_actual_service_guard_matches_preserve_exact_snapshot_coverage(case: str) -> None:
    from newsroom.graphiti_adapter.neo4j_guard import GuardError, Neo4jMutationGuard

    async def exercise() -> None:
        config = _service_config()
        suffix = str(uuid.uuid4())
        group_id = f"newsroom-guard-coverage-{suffix}"
        episode_uuid = f"episode-{suffix}"
        snapshot_id = f"{episode_uuid}:1"
        driver = _DatabaseBoundAsyncDriver(
            AsyncGraphDatabase.driver(config.uri, auth=(config.username, config.password)),
            database=config.database,
        )

        async def query(cypher: str, **parameters: object) -> EagerResult:
            return await driver.execute_query(cypher, params=parameters, routing_="w")

        identities = {
            "a": f"a-{suffix}", "b": f"b-{suffix}", "c": f"c-{suffix}",
            "first": f"first-{suffix}", "second": f"second-{suffix}",
            "group_id": group_id,
        }
        try:
            if case != "empty":
                await query(
                    """
                    CREATE (a:Entity {uuid:$a, group_id:$group_id, name:'a'}),
                           (b:Entity {uuid:$b, group_id:$group_id, name:'b'}),
                           (c:Entity {uuid:$c, group_id:$group_id, name:'c'}),
                           (a)-[:REL {uuid:$first, fact:'first'}]->(b),
                           (b)-[:REL {uuid:$second, fact:'second'}]->(c)
                    """, **identities,
                )
            guard = Neo4jMutationGuard(
                driver, group_id=group_id, episode_uuid=episode_uuid,
                attempt_number=1, input_digest="sha256:" + "1" * 64,
            )
            await guard.begin()
            if "duplicate" in case:
                if "relationship" in case:
                    await query(
                        """
                        MATCH (b:Entity {uuid:$b})-[r:REL {uuid:$second}]->(c:Entity {uuid:$c})
                        CREATE (b)-[duplicate:REL]->(c)
                        SET duplicate=properties(r)
                        """ + (" SET duplicate.fact='changed'" if case.startswith("corrupt") else ""),
                        **identities,
                    )
                else:
                    await query(
                        """
                        MATCH (b:Entity {uuid:$b})
                        CREATE (duplicate:Entity) SET duplicate=properties(b)
                        """ + (" SET duplicate.name='changed'" if case.startswith("corrupt") else ""),
                        **identities,
                    )
            if case == "missing-node-duplicate-survivor":
                await query("MATCH (a:Entity {uuid:$a}) DETACH DELETE a", **identities)
            elif case == "missing-relationship-duplicate-survivor":
                await query("MATCH ()-[r:REL {uuid:$first}]->() DELETE r", **identities)
            elif case in {"wrong-endpoint", "wrong-type"}:
                endpoint = "(b)-[replacement:REL]->(c)" if case == "wrong-endpoint" else "(a)-[replacement:OTHER]->(b)"
                await query(
                    """
                    MATCH (a:Entity {uuid:$a})-[r:REL {uuid:$first}]->(b:Entity {uuid:$b}),
                          (c:Entity {uuid:$c})
                    """ + f"CREATE {endpoint} SET replacement=properties(r) DELETE r",
                    **identities,
                )
            if case.startswith(("missing", "corrupt", "wrong")):
                with pytest.raises(GuardError, match="pre-existing Graphiti"):
                    await guard.assert_preexisting_unchanged()
            else:
                await guard.assert_preexisting_unchanged()
            marker = await query(
                "MATCH (m:NewsroomIngestMarker {episode_uuid:$episode_uuid}) RETURN m.state AS state",
                episode_uuid=episode_uuid,
            )
            assert marker.records[0]["state"] == "PENDING"
        finally:
            try:
                await query(
                    """
                    MATCH (n) WHERE n.group_id=$group_id OR n.episode_uuid=$episode_uuid
                                 OR n._newsroom_snapshot_id=$snapshot_id
                    DETACH DELETE n
                    """, group_id=group_id, episode_uuid=episode_uuid, snapshot_id=snapshot_id,
                )
            finally:
                await driver.close()

    asyncio.run(exercise())

def test_actual_service_increment4_admitted_state_projects_exactly_and_replays(
    tmp_path: Path,
) -> None:
    config = _service_config()
    state = seed_increment4_graphiti_path(tmp_path / "authority")
    admitted = admit_increment4_graphiti_path(state)
    snapshot = admitted.snapshot
    generation_id = ProjectionGenerationId.new()
    batches, canonical_ids, node_ids, relation_keys = _expected(
        snapshot, generation_id
    )
    request = _request(
        generation_id,
        snapshot,
        key="increment4e-actual-service-build-v1",
    )
    before_events = _event_count(state.extraction.database)
    adapter = _open_neo4j_adapter(config)
    system = open_graphiti_path_increment4_neo4j_system(
        state.relation,
        adapter,
    )
    try:
        built = system.increment4.build_and_promote(
            request,
            proof=extraction_proof(),
        )
        response = system.increment4.read_active(
            Increment4Neo4jActiveReadRequest(
                canonical_ids=canonical_ids,
                query_valid_time=SOURCE_NOW,
                limit=100,
            ),
            proof=extraction_proof(),
        )
        after_first = _event_count(state.extraction.database)
        replay = system.increment4.build_and_promote(
            request,
            proof=extraction_proof(),
        )
        after_replay = _event_count(state.extraction.database)
    finally:
        system.close()

    try:
        assert built.generation.state is ProjectionGenerationState.ACTIVE
        assert replay.promotion.promotion_digest == built.promotion.promotion_digest
        assert replay.validation.validation_digest == built.validation.validation_digest
        assert after_first > before_events
        assert after_replay == after_first
        assert response.metadata.generation_id == generation_id
        assert {item.canonical_id for item in response.nodes} == node_ids
        assert {item.relation_key for item in response.relations} == relation_keys
        assert all(item.trust_scope.value == "ADMITTED" for item in response.relations)
        assert _scalar(
            "MATCH (n:NewsroomProjectionNode {generation_id:$generation_id}) "
            "RETURN count(n)",
            generation_id=str(generation_id),
        ) == len(node_ids)
        assert _scalar(
            "MATCH (:NewsroomProjectionNode {generation_id:$generation_id})"
            "-[r]->(:NewsroomProjectionNode {generation_id:$generation_id}) "
            "WHERE r.generation_id=$generation_id RETURN count(r)",
            generation_id=str(generation_id),
        ) == len(relation_keys)
        assert _scalar(
            "MATCH (d:NewsroomProjectionDelivery {generation_id:$generation_id}) "
            "RETURN count(d)",
            generation_id=str(generation_id),
        ) == len(batches)

        projected_ids = _values(
            "MATCH (n:NewsroomProjectionNode {generation_id:$generation_id}) "
            "RETURN n.canonical_id ORDER BY n.canonical_id",
            generation_id=str(generation_id),
        )
        prohibited = {
            str(RELATION_PROPOSAL_ID),
            str(RELATION_PROPOSAL_V1_ID),
        }
        assert prohibited.isdisjoint(projected_ids)
        assert not _values(
            "MATCH (n {generation_id:$generation_id}) "
            "UNWIND labels(n) AS label "
            "WITH DISTINCT label WHERE label CONTAINS 'Proposal' "
            "OR label CONTAINS 'Graphiti' RETURN label ORDER BY label",
            generation_id=str(generation_id),
        )
        assert not state.workspace_root.exists() or not any(
            state.workspace_root.iterdir()
        )
        with sqlite3.connect(state.extraction.database) as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM graphiti_adapter_attempts"
            ).fetchone()[0] == 2
            assert conn.execute(
                "SELECT COUNT(*) FROM extraction_proposals"
            ).fetchone()[0] >= 4
            assert conn.execute(
                "SELECT COUNT(*) FROM canonical_entities"
            ).fetchone()[0] == 2
            assert conn.execute(
                "SELECT COUNT(*) FROM editorial_relation_assertions"
            ).fetchone()[0] == 1
    finally:
        _cleanup(config, generation_id)


def test_actual_service_increment4_graph_loss_requires_isolated_replacement(
    tmp_path: Path,
) -> None:
    config = _service_config()
    state = seed_increment4_graphiti_path(tmp_path / "authority")
    admitted = admit_increment4_graphiti_path(state)
    snapshot = admitted.snapshot
    first_id = ProjectionGenerationId.new()
    replacement_id = ProjectionGenerationId.new()
    _first_batches, first_canonical_ids, _node_ids, _relation_keys = _expected(
        snapshot, first_id
    )
    _replacement_batches, replacement_canonical_ids, _node_ids2, _relation_keys2 = (
        _expected(snapshot, replacement_id)
    )
    first_request = _request(
        first_id,
        snapshot,
        key="increment4e-actual-service-loss-v1",
    )
    database = state.extraction.database

    system = open_graphiti_path_increment4_neo4j_system(
        state.relation,
        _open_neo4j_adapter(config),
    )
    try:
        first = system.increment4.build_and_promote(
            first_request,
            proof=extraction_proof(),
        )
        initial = system.increment4.read_active(
            Increment4Neo4jActiveReadRequest(
                canonical_ids=first_canonical_ids,
                query_valid_time=SOURCE_NOW,
                limit=100,
            ),
            proof=extraction_proof(),
        )
        before_events = _event_count(database)
    finally:
        system.close()

    try:
        _cleanup(config, first_id)
        restarted = open_graphiti_path_increment4_neo4j_system(
            state.relation,
            _open_neo4j_adapter(config),
        )
        try:
            missing = restarted.increment4.read_active(
                Increment4Neo4jActiveReadRequest(
                    canonical_ids=first_canonical_ids,
                    query_valid_time=SOURCE_NOW,
                    limit=100,
                ),
                proof=extraction_proof(),
            )
            assert missing.nodes == ()
            assert missing.relations == ()
            with pytest.raises(Neo4jIdentityConflict):
                restarted.increment4.build_and_promote(
                    first_request,
                    proof=extraction_proof(),
                )
            still_active = restarted.increment4.generation_status(
                first_id,
                proof=extraction_proof(),
            )
            replacement = restarted.increment4.build_and_promote(
                _request(
                    replacement_id,
                    snapshot,
                    key="increment4e-actual-service-loss-replacement-v1",
                ),
                proof=extraction_proof(),
            )
            retired = restarted.increment4.generation_status(
                first_id,
                proof=extraction_proof(),
            )
            after = restarted.increment4.read_active(
                Increment4Neo4jActiveReadRequest(
                    canonical_ids=replacement_canonical_ids,
                    query_valid_time=SOURCE_NOW,
                    limit=100,
                ),
                proof=extraction_proof(),
            )
        finally:
            restarted.close()

        assert first.generation.state is ProjectionGenerationState.ACTIVE
        assert still_active.generation.state is ProjectionGenerationState.ACTIVE
        assert replacement.generation.state is ProjectionGenerationState.ACTIVE
        assert replacement.prior_generation is not None
        assert replacement.prior_generation.generation_id == first_id
        assert retired.generation.state is ProjectionGenerationState.RETIRED
        assert after.metadata.generation_id == replacement_id
        assert {item.canonical_id for item in after.nodes} == {
            item.canonical_id for item in initial.nodes
        }
        assert {
            (item.relation_type, item.source_canonical_id, item.target_canonical_id)
            for item in after.relations
        } == {
            (item.relation_type, item.source_canonical_id, item.target_canonical_id)
            for item in initial.relations
        }
        assert _event_count(database) > before_events
    finally:
        _cleanup(config, first_id, replacement_id)



def test_actual_service_increment4_replacement_generation_is_only_serving_state(
    tmp_path: Path,
) -> None:
    config = _service_config()
    state = seed_increment4_graphiti_path(tmp_path / "authority")
    admitted = admit_increment4_graphiti_path(state)
    snapshot = admitted.snapshot
    first_id = ProjectionGenerationId.new()
    second_id = ProjectionGenerationId.new()
    _first_batches, first_canonical_ids, _nodes, _relations = _expected(
        snapshot, first_id
    )
    _second_batches, second_canonical_ids, _nodes2, _relations2 = _expected(
        snapshot, second_id
    )

    system = open_graphiti_path_increment4_neo4j_system(
        state.relation,
        _open_neo4j_adapter(config),
    )
    try:
        first = system.increment4.build_and_promote(
            _request(first_id, snapshot, key="increment4e-service-first-v1"),
            proof=extraction_proof(),
        )
        second = system.increment4.build_and_promote(
            _request(second_id, snapshot, key="increment4e-service-second-v1"),
            proof=extraction_proof(),
        )
        first_status = system.increment4.generation_status(
            first_id,
            proof=extraction_proof(),
        )
        active = system.increment4.read_active(
            Increment4Neo4jActiveReadRequest(
                canonical_ids=second_canonical_ids,
                query_valid_time=SOURCE_NOW,
                limit=100,
            ),
            proof=extraction_proof(),
        )
    finally:
        system.close()

    try:
        assert first.generation.state is ProjectionGenerationState.ACTIVE
        assert second.generation.state is ProjectionGenerationState.ACTIVE
        assert second.prior_generation is not None
        assert second.prior_generation.generation_id == first_id
        assert first_status.generation.state is ProjectionGenerationState.RETIRED
        assert active.metadata.generation_id == second_id
        assert active.nodes and active.relations
        assert _scalar(
            "MATCH (n {generation_id:$generation_id}) RETURN count(n)",
            generation_id=str(first_id),
        ) == 0
        assert _scalar(
            "MATCH (n {generation_id:$generation_id}) RETURN count(n)",
            generation_id=str(second_id),
        ) > 0
        assert first_canonical_ids == second_canonical_ids
    finally:
        _cleanup(config, first_id, second_id)


def test_actual_service_increment4_tombstone_purges_and_never_resurrects(
    tmp_path: Path,
) -> None:
    config = _service_config()
    state = seed_increment4_graphiti_path(tmp_path / "authority")
    admitted = admit_increment4_graphiti_path(state)
    first_id = ProjectionGenerationId.new()
    purge_id = ProjectionGenerationId.new()
    _batches, canonical_ids, _node_ids, _relation_keys = _expected(
        admitted.snapshot, first_id
    )

    system = open_graphiti_path_increment4_neo4j_system(
        state.relation,
        _open_neo4j_adapter(config),
    )
    try:
        first = system.increment4.build_and_promote(
            _request(
                first_id,
                admitted.snapshot,
                key="increment4e-service-tombstone-base-v1",
            ),
            proof=extraction_proof(),
        )
    finally:
        system.close()

    commands, schemas = graphiti_path_registries(state.extraction)
    passage = state.extraction.input_binding.passages[0]
    with open_object_system(
        state.extraction.database,
        object_root=state.extraction.object_root,
        clock=lambda: SOURCE_NOW,
        command_registry=commands,
        payload_schema_registry=schemas,
    ) as objects:
        deletion = objects.objects.request_deletion(
            passage.blob_digest,
            reason_code="INCREMENT4E_ACTUAL_SOURCE_DELETE",
            idempotency_key="increment4e-actual-source-delete-v1",
            proof=object_proof(),
        )
        objects.objects.tombstone(
            deletion.deletion_id,
            reason_code="INCREMENT4E_ACTUAL_SOURCE_TOMBSTONE",
            idempotency_key="increment4e-actual-source-tombstone-v1",
            proof=object_proof(),
        )

    events = _ledger_events(state.extraction.database)
    empty = sorted_snapshot(
        entities=(),
        relations=(),
        events=events,
        through_ledger_seq=events[-1].ledger_seq,
    )
    purge_request = _request(
        purge_id,
        empty,
        key="increment4e-actual-source-purge-v1",
    )
    restarted = open_graphiti_path_increment4_neo4j_system(
        state.relation,
        _open_neo4j_adapter(config),
    )
    try:
        purged = restarted.increment4.build_and_promote(
            purge_request,
            proof=extraction_proof(),
        )
        read = restarted.increment4.read_active(
            Increment4Neo4jActiveReadRequest(
                canonical_ids=canonical_ids,
                query_valid_time=SOURCE_NOW,
                limit=100,
            ),
            proof=extraction_proof(),
        )
        replay = restarted.increment4.build_and_promote(
            purge_request,
            proof=extraction_proof(),
        )
    finally:
        restarted.close()

    try:
        assert first.generation.state is ProjectionGenerationState.ACTIVE
        assert purged.generation.state is ProjectionGenerationState.ACTIVE
        assert purged.prior_generation is not None
        assert purged.prior_generation.generation_id == first_id
        assert purged.projected_batch_count == 0
        assert purged.purged_retired_graph_record_count > 0
        assert read.nodes == ()
        assert read.relations == ()
        assert replay.validation.validation_digest == purged.validation.validation_digest
        assert replay.promotion.promotion_digest == purged.promotion.promotion_digest
        assert _scalar(
            "MATCH (n {generation_id:$generation_id}) RETURN count(n)",
            generation_id=str(first_id),
        ) == 0
        assert _scalar(
            "MATCH (n {generation_id:$generation_id}) RETURN count(n)",
            generation_id=str(purge_id),
        ) == 0
        with sqlite3.connect(state.extraction.database) as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM graphiti_adapter_attempts"
            ).fetchone()[0] == 2
            assert conn.execute(
                "SELECT COUNT(*) FROM editorial_relation_assertions"
            ).fetchone()[0] == 1
            assert conn.execute(
                "SELECT COUNT(*) FROM object_deletion_heads h "
                "JOIN object_deletion_versions v "
                "ON v.deletion_id=h.deletion_id "
                "AND v.lifecycle_version=h.current_version "
                "WHERE h.deletion_id=? AND v.state='TOMBSTONED'",
                (str(deletion.deletion_id),),
            ).fetchone()[0] == 1
    finally:
        _cleanup(config, first_id, purge_id)


def test_actual_service_active_suffix_reopen_and_replay(tmp_path, monkeypatch):
    from newsroom.projection.neo4j._adapter import _Neo4jAdapter
    from .test_increment4_active_extension import add_suffix, request

    config = _service_config()
    state = seed_increment4_graphiti_path(tmp_path / 'authority')
    admit_increment4_graphiti_path(state)
    first_id, fallback_id = ProjectionGenerationId.new(), ProjectionGenerationId.new()
    calls = []
    apply = _Neo4jAdapter.apply

    def measured_apply(self, batch):
        calls.append(batch.ledger_seq)
        return apply(self, batch)

    monkeypatch.setattr(_Neo4jAdapter, 'apply', measured_apply)
    try:
        with open_graphiti_path_increment4_neo4j_system(
            state.relation, _open_neo4j_adapter(config),
        ) as system:
            first = system.increment4.build_current_and_promote(
                request(first_id, 'service-initial'), proof=extraction_proof(),
            )
        add_suffix(state)
        calls.clear()
        before = _event_count(state.extraction.database)
        with open_graphiti_path_increment4_neo4j_system(
            state.relation, _open_neo4j_adapter(config),
        ) as system:
            second = system.increment4.build_current_and_promote(
                request(fallback_id, 'service-suffix'), proof=extraction_proof(),
            )
            assert second.generation.generation_id == first_id
            assert second.promotion == first.promotion
            assert len(calls) == 1
            assert _event_count(state.extraction.database) - before == 2
            system.increment4.reconcile_active(proof=extraction_proof())
        calls.clear()
        before = _event_count(state.extraction.database)
        with open_graphiti_path_increment4_neo4j_system(
            state.relation, _open_neo4j_adapter(config),
        ) as system:
            replay = system.increment4.build_current_and_promote(
                request(fallback_id, 'service-suffix'), proof=extraction_proof(),
            )
            assert replay.validation == second.validation
            assert replay.source_snapshot_digest == second.source_snapshot_digest
            assert calls == []
            assert _event_count(state.extraction.database) == before
        assert _scalar(
            'MATCH (d:NewsroomProjectionDelivery {generation_id:$generation_id}) RETURN count(d)',
            generation_id=str(first_id),
        ) == first.projected_batch_count + 1
        assert _scalar(
            'MATCH (d {generation_id:$generation_id}) RETURN count(d)',
            generation_id=str(fallback_id),
        ) == 0
    finally:
        _cleanup(config, first_id, fallback_id)
