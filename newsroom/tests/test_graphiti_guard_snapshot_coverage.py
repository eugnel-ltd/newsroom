from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import pytest

from newsroom.graphiti_adapter.neo4j_guard import GuardError, Neo4jMutationGuard


_NODE = {
    "snapshot_identity": "snapshot-node-1",
    "snapshot": {
        "uuid": "node-1", "embedding": [1.0, 2.0],
        "_newsroom_source_uuid": "node-1", "_newsroom_source_labels": ["Entity"],
    },
    "current": {"uuid": "node-1", "embedding": [1.0, 2.0]},
    "current_labels": ["Entity"],
}
_RELATIONSHIP = {
    "snapshot_identity": "snapshot-relationship-1",
    "snapshot": {
        "uuid": "relationship-1", "weight": 1.0,
        "_newsroom_source_uuid": "node-1", "_newsroom_target_uuid": "node-2",
        "_newsroom_relationship_type": "RELATES_TO",
    },
    "current": {"uuid": "relationship-1", "weight": 1.0},
    "source_uuid": "node-1", "target_uuid": "node-2", "relationship_type": "RELATES_TO",
}


class _Result:
    def __init__(self, records):
        self.records = records

    def __iter__(self):
        raise AssertionError("snapshot property records must remain streamed")

    async def single(self, *, strict=False):
        assert strict and len(self.records) == 1
        return self.records[0]

    async def __aiter__(self):
        for record in self.records:
            if isinstance(record, BaseException):
                raise record
            yield record


class _RetryTransaction(Exception):
    pass


def _guard(
    *, nodes=(), relationships=(), expected_nodes=0, expected_relationships=0,
    node_attempts=None,
):
    queries = []
    phases = [
        node_attempts or [(expected_nodes, nodes)],
        [(expected_relationships, relationships)],
    ]

    class Transaction:
        def __init__(self, count, records):
            self.count = count
            self.records = records

        async def run(self, query, **params):
            assert params == {"snapshot_id": "episode-id:1"}
            assert "OPTIONAL MATCH" not in query, "per-snapshot full scans must be set-based"
            queries.append(query)
            if "RETURN count(s) AS snapshot_count" in query:
                return _Result([{"snapshot_count": self.count}])
            assert "elementId(s) AS snapshot_identity" in query
            return _Result(self.records)

    class Session:
        def __init__(self):
            self.attempts = phases.pop(0)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def execute_write(self, callback):
            for index, (count, records) in enumerate(self.attempts):
                try:
                    await callback(Transaction(count, records))
                    return
                except _RetryTransaction:
                    assert index < len(self.attempts) - 1
            raise AssertionError("no transaction attempt completed")

    guard = Neo4jMutationGuard(
        SimpleNamespace(session=Session), group_id="group-id", episode_uuid="episode-id",
        attempt_number=1, input_digest="sha256:" + "0" * 64,
    )
    return guard, queries


def test_guard_streams_set_based_matches_with_exact_snapshot_coverage() -> None:
    guard, queries = _guard(
        nodes=[_NODE], relationships=[_RELATIONSHIP],
        expected_nodes=1, expected_relationships=1,
    )
    asyncio.run(guard.assert_preexisting_unchanged())
    assert len(queries) == 4


@pytest.mark.parametrize("kind", ["node", "relationship"])
@pytest.mark.parametrize("duplicate_survivor", [False, True])
def test_guard_rejects_missing_snapshot_even_if_another_original_matches_twice(
    kind: str, duplicate_survivor: bool,
) -> None:
    record = _NODE if kind == "node" else _RELATIONSHIP
    records = [record, copy.deepcopy(record)] if duplicate_survivor else []
    kwargs = (
        {"nodes": records, "expected_nodes": 2 if duplicate_survivor else 1}
        if kind == "node" else
        {"relationships": records, "expected_relationships": 2 if duplicate_survivor else 1}
    )
    guard, _ = _guard(**kwargs)
    with pytest.raises(GuardError, match=f"pre-existing Graphiti {kind} is missing"):
        asyncio.run(guard.assert_preexisting_unchanged())


@pytest.mark.parametrize("kind", ["node", "relationship"])
def test_guard_validates_all_duplicate_originals_without_changing_legacy_acceptance(kind: str) -> None:
    record = _NODE if kind == "node" else _RELATIONSHIP
    records = [record, copy.deepcopy(record)]
    kwargs = (
        {"nodes": records, "expected_nodes": 1} if kind == "node" else
        {"relationships": records, "expected_relationships": 1}
    )
    guard, _ = _guard(**kwargs)
    asyncio.run(guard.assert_preexisting_unchanged())


@pytest.mark.parametrize(("kind", "path", "value"), [
    ("node", ("snapshot",), None),
    ("node", ("current",), None),
    ("node", ("current", "uuid"), "different"),
    ("node", ("current", "embedding"), [2.0, 1.0]),
    ("node", ("current", "extra"), True),
    ("node", ("current_labels",), ["Other"]),
    ("relationship", ("snapshot",), None),
    ("relationship", ("current",), None),
    ("relationship", ("current", "uuid"), "different"),
    ("relationship", ("current", "weight"), 2.0),
    ("relationship", ("current", "extra"), True),
    ("relationship", ("source_uuid",), "other-source"),
    ("relationship", ("target_uuid",), "other-target"),
    ("relationship", ("relationship_type",), "OTHER"),
])
def test_guard_rejects_each_corruption_surface_even_after_a_valid_duplicate(
    kind: str, path: tuple[str, ...], value: object,
) -> None:
    record = _NODE if kind == "node" else _RELATIONSHIP
    changed = copy.deepcopy(record)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    kwargs = (
        {"nodes": [record, changed], "expected_nodes": 1} if kind == "node" else
        {"relationships": [record, changed], "expected_relationships": 1}
    )
    guard, _ = _guard(**kwargs)
    with pytest.raises(GuardError, match="pre-existing Graphiti"):
        asyncio.run(guard.assert_preexisting_unchanged())


def test_guard_accepts_empty_snapshot_sets() -> None:
    guard, queries = _guard()
    asyncio.run(guard.assert_preexisting_unchanged())
    assert len(queries) == 4


@pytest.mark.parametrize("kind", ["node", "relationship"])
def test_guard_partial_cancellation_never_finishes_coverage(kind: str) -> None:
    record = _NODE if kind == "node" else _RELATIONSHIP
    records = [record, asyncio.CancelledError()]
    kwargs = (
        {"nodes": records, "expected_nodes": 1} if kind == "node" else
        {"relationships": records, "expected_relationships": 1}
    )
    guard, queries = _guard(**kwargs)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(guard.assert_preexisting_unchanged())
    assert len(queries) == (2 if kind == "node" else 4)


@pytest.mark.parametrize("second_count", [1, 2])
def test_guard_transaction_retry_recounts_and_does_not_reuse_partial_coverage(second_count: int) -> None:
    second = copy.deepcopy(_NODE)
    second["snapshot_identity"] = "snapshot-node-2"
    guard, queries = _guard(node_attempts=[
        (3, [_NODE, _RetryTransaction()]),
        (second_count, [second]),
    ])
    if second_count == 1:
        asyncio.run(guard.assert_preexisting_unchanged())
        assert len(queries) == 6
    else:
        with pytest.raises(GuardError, match="node is missing"):
            asyncio.run(guard.assert_preexisting_unchanged())
        assert len(queries) == 4


@pytest.mark.parametrize("identity", [None, "", 12])
def test_guard_rejects_absent_snapshot_coverage_identity(identity: object) -> None:
    record = {**_NODE, "snapshot_identity": identity}
    guard, _ = _guard(nodes=[record], expected_nodes=1)
    with pytest.raises(GuardError, match="coverage identity is absent"):
        asyncio.run(guard.assert_preexisting_unchanged())


@pytest.mark.parametrize("count", [None, -1, True, "1"])
def test_guard_rejects_invalid_snapshot_coverage_count(count: object) -> None:
    guard, _ = _guard(nodes=[_NODE], expected_nodes=count)
    with pytest.raises(GuardError, match="coverage count is invalid"):
        asyncio.run(guard.assert_preexisting_unchanged())


class _JournalResult:
    def __init__(self, records):
        self.records = records

    async def single(self, **_kwargs):
        return self.records[0] if self.records else None


class _JournalDriver:
    """Model committed marker state and the generation-only transaction lock."""

    def __init__(self):
        self.markers = {}
        self.owner = {}
        self.lock = asyncio.Lock()
        self.queries = []
        self.events = []

    def apply(self, query, params):
        self.queries.append(query)
        marker = self.markers.get(params.get("episode_uuid"))
        if "AS marker_episode_uuid" in query:
            retained = self.markers.get(self.owner.get("owner_marker_uuid"))
            if (retained is None or retained.get("active")
                or retained.get("state") not in {"PENDING", "ROLLING_BACK"}
                or retained.get("snapshot_id") != self.owner.get("snapshot_id")
                or retained.get("claim_token") != self.owner.get("claim_token")):
                return []
            return [{"marker_episode_uuid": self.owner["owner_marker_uuid"],
                     "snapshot_id": retained["snapshot_id"],
                     "attempt_number": retained["attempt_number"]}]
        if "SET g.lock_tick" in query:
            assert query.index("SET g.lock_tick") < min(query.index(term) for term in ("MATCH (m:", "MERGE (m:") if term in query)
            if "MERGE (m:" in query:
                if self.owner.get("owner_marker_uuid") not in (None, params["episode_uuid"]):
                    return []
                if not self.owner.get("owner_marker_uuid") and any(
                    item["state"] in {"SNAPSHOTTING", "PENDING", "ROLLING_BACK", "RECOVERING"}
                    and ("(unresolved:NewsroomIngestMarker {group_id: $group_id})" not in query
                         or item["group_id"] == params["group_id"])
                    and ("unresolved.snapshot_id = $snapshot_id" not in query
                         or item["snapshot_id"] == params["snapshot_id"])
                    for item in self.markers.values()
                ):
                    return []
                claimed = marker is None
                if claimed:
                    marker = {key: params[key] for key in (
                        "group_id", "attempt_number", "input_digest", "snapshot_id", "claim_token",
                    )}
                    marker.update(state="SNAPSHOTTING", active=True)
                    self.markers[params["episode_uuid"]] = marker
                    self.owner = {"owner_marker_uuid": params["episode_uuid"],
                                  "snapshot_id": marker["snapshot_id"], "claim_token": marker["claim_token"]}
                elif (self.owner.get("snapshot_id"), self.owner.get("claim_token")) != (
                    marker["snapshot_id"], marker["claim_token"],
                ):
                    return []
                return [{"marker": dict(marker), "claimed": claimed, "active": marker.get("active", False)}]
            if "SET m.state = $state" in query:
                if marker is None or marker.get("active") or self.owner.get("owner_marker_uuid") != params["episode_uuid"]:
                    return []
                if (marker["state"], marker["snapshot_id"], marker["claim_token"]) != (
                    params["retained_state"], params["snapshot_id"], params["retained_claim_token"],
                ) or (self.owner.get("snapshot_id"), self.owner.get("claim_token")) != (
                    marker["snapshot_id"], marker["claim_token"],
                ):
                    return []
                marker.update(state=params["state"], claim_token=params["claim_token"], active=True)
                self.owner["claim_token"] = params["claim_token"]
                return [{"marker": dict(marker)}]
            if (marker is None or self.owner.get("owner_marker_uuid") != params["episode_uuid"]
                or self.owner.get("snapshot_id") != params["snapshot_id"]
                or self.owner.get("claim_token") != params["claim_token"]
                or marker["claim_token"] != params["claim_token"]):
                return []
            if "SET m.state = 'PENDING'" in query:
                marker["state"] = "PENDING"
                return [{"state": "PENDING"}]
            if "SET m.state = 'ROLLING_BACK'" in query:
                if marker["state"] != "PENDING":
                    return []
                marker["state"] = "ROLLING_BACK"
                self.events.append("rollback-committed")
                return [{"state": "ROLLING_BACK"}]
            for terminal in ("COMPLETE", "RECOVERED_AMBIGUOUS"):
                if f"SET m.state = '{terminal}'" in query:
                    marker["state"] = terminal
                    self.owner = {}
                    self.events.append("terminal")
                    return [{"state": terminal}]
            return [{"claim_token": marker["claim_token"]}]
        if "RETURN properties(m) AS marker" in query:
            return [] if marker is None else [{"marker": dict(marker)}]
        # A legacy implementation is intentionally modelled faithfully: it has
        # no generation owner, and lets another episode claim after lease expiry.
        if "MERGE (m:" in query:
            claimed = marker is None
            if claimed:
                marker = {key: params[key] for key in (
                    "group_id", "attempt_number", "input_digest", "snapshot_id", "claim_token",
                )}
                marker.update(state="SNAPSHOTTING", active=True)
                self.markers[params["episode_uuid"]] = marker
            return [{"marker": dict(marker), "claimed": claimed, "active": marker.get("active", False)}]
        if "RETURN m.state AS state" in query:
            for state in ("PENDING", "ROLLING_BACK", "COMPLETE", "RECOVERED_AMBIGUOUS"):
                if f"SET m.state = '{state}'" in query and marker is not None:
                    marker["state"] = state
                    return [{"state": state}]
        return []

    async def execute_query(self, query, *, params, routing_):
        assert routing_ == "w"
        if "SET g.lock_tick" in query:
            async with self.lock:
                return self.apply(query, params), None, None
        return self.apply(query, params), None, None

    def session(self):
        driver = self

        class Transaction:
            held = False

            async def run(self, query, **params):
                assert "SET m.claim_expires_at" not in query, "the long fence must not lock the episode marker"
                if "SET g.lock_tick" in query and not self.held:
                    await driver.lock.acquire()
                    self.held = True
                    driver.events.append("fence")
                return _JournalResult(driver.apply(query, params))

            async def commit(self):
                driver.events.append("commit")
                if self.held:
                    driver.lock.release()
                    self.held = False

            async def rollback(self):
                if self.held:
                    driver.lock.release()
                    self.held = False

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def begin_transaction(self):
                return Transaction()

        return Session()


class _JournalGuard(Neo4jMutationGuard):
    async def _snapshot(self):
        pass

    async def _restoration_inventories(self):
        return [], []

    async def _snapshot_uuids(self, inventory, *, snapshot_label):
        return []

    async def _restore_properties(self, **_values):
        self.driver.events.append("properties")

    async def assert_preexisting_unchanged(self):
        self.driver.events.append("verified")

    async def _delete_snapshot(self):
        self.driver.events.append("snapshot-deleted")


def _journal_guard(driver, episode="episode-a"):
    return _JournalGuard(driver, group_id="group-id", episode_uuid=episode,
                         attempt_number=1, input_digest="sha256:" + "0" * 64)


def test_generation_owner_survives_crash_and_expiry_and_blocks_another_episode():
    async def exercise():
        driver = _JournalDriver()
        guard = _journal_guard(driver)
        await guard.begin()
        driver.markers["episode-a"].update(state="ROLLING_BACK", active=False)
        with pytest.raises(GuardError, match="generation"):
            await _journal_guard(driver, "episode-b").begin()
        assert "episode-b" not in driver.markers
        recovery = _journal_guard(driver)
        retained = await recovery.begin()
        assert retained.state.value == "ROLLING_BACK"
        assert driver.owner["claim_token"] == recovery._claim_token
        await recovery.rollback_pending(chat_invocations=[], embedding_usage={}, reason="CRASH")
        assert driver.markers["episode-a"]["state"] == "RECOVERED_AMBIGUOUS"
        assert not driver.owner
        assert driver.events.index("verified") < driver.events.index("terminal") < driver.events.index("snapshot-deleted")
        assert (await _journal_guard(driver, "episode-b").begin()).state.value == "CREATED"

    asyncio.run(exercise())


@pytest.mark.parametrize("state", ["SNAPSHOTTING", "PENDING", "ROLLING_BACK", "RECOVERING"])
def test_generation_refuses_legacy_unowned_unresolved_marker(state):
    async def exercise():
        driver = _JournalDriver()
        driver.markers["episode-a"] = {
            "group_id": "group-id", "input_digest": "sha256:" + "0" * 64,
            "attempt_number": 1, "snapshot_id": "episode-a:1", "state": state,
            "claim_token": "expired", "active": False,
        }
        with pytest.raises(GuardError, match="generation"):
            await _journal_guard(driver).begin()
        before = dict(driver.markers["episode-a"])
        assert (await _journal_guard(driver, "episode-b").begin()).state.value == "CREATED"
        assert driver.markers["episode-a"] == before
        assert "snapshot-deleted" not in driver.events
        with pytest.raises(GuardError, match="generation"):
            await _journal_guard(driver).begin()
        assert driver.markers["episode-a"] == before

    asyncio.run(exercise())


@pytest.mark.parametrize("legacy_group", ("group-id", "foreign-group"))
def test_fresh_internal_marker_cannot_reuse_unresolved_legacy_snapshot(legacy_group):
    async def exercise():
        driver = _JournalDriver()
        driver.markers["episode-a"] = {
            "group_id": legacy_group, "input_digest": "sha256:" + "0" * 64,
            "attempt_number": 2, "snapshot_id": "episode-a:2", "state": "PENDING",
            "claim_token": "expired", "active": False,
        }
        fresh = _JournalGuard(driver, group_id="group-id", episode_uuid="episode-a",
                              attempt_number=2, marker_episode_uuid="episode-a:attempt:2",
                              input_digest="sha256:" + "0" * 64)
        with pytest.raises(GuardError, match="generation"):
            await fresh.begin()
        assert "episode-a:attempt:2" not in driver.markers
        assert "snapshot-deleted" not in driver.events
    asyncio.run(exercise())


def test_exact_owned_compensation_releases_generation_without_a_fresh_snapshot():
    async def exercise():
        driver = _JournalDriver()
        await _journal_guard(driver).begin()
        driver.markers["episode-a"].update(state="ROLLING_BACK", active=False)
        driver.events.clear()
        identity = await Neo4jMutationGuard.owned_pending_identity(driver, group_id="group-id")
        assert identity == ("episode-a", "episode-a", 1)
        recovery = _journal_guard(driver)
        marker = await recovery.recover_owned_pending()
        assert marker.state.value == "RECOVERED_AMBIGUOUS"
        assert not driver.owner
        assert driver.events.index("verified") < driver.events.index("terminal") < driver.events.index("snapshot-deleted")
        assert not any("MERGE (m:" in query for query in driver.queries[3:])
    asyncio.run(exercise())


@pytest.mark.parametrize("defect", ["unowned", "active", "snapshot", "input", "stopped"])
def test_owned_compensation_refuses_without_mutating_retained_history(defect):
    async def exercise():
        driver = _JournalDriver()
        await _journal_guard(driver).begin()
        driver.markers["episode-a"].update(state="ROLLING_BACK", active=False)
        if defect == "unowned":
            driver.owner = {}
        elif defect == "active":
            driver.markers["episode-a"]["active"] = True
        elif defect == "snapshot":
            driver.owner["snapshot_id"] = "other-snapshot"
        recovery = _journal_guard(driver)
        if defect == "input":
            recovery = _JournalGuard(driver, group_id="group-id", episode_uuid="episode-a",
                                     attempt_number=1, input_digest="sha256:" + "1" * 64)
        before = copy.deepcopy((driver.markers, driver.owner))
        driver.events.clear()
        def stopped():
            raise asyncio.CancelledError()
        if defect in {"input", "stopped"}:
            with pytest.raises(GuardError if defect == "input" else asyncio.CancelledError):
                await recovery.recover_owned_pending(owner_stop_check=stopped if defect == "stopped" else None)
        else:
            assert await recovery.recover_owned_pending() is None
        assert (driver.markers, driver.owner) == before
        assert "properties" not in driver.events and "terminal" not in driver.events
    asyncio.run(exercise())


def test_generation_takeover_binds_snapshot_and_token_and_has_one_claimant():
    async def exercise():
        driver = _JournalDriver()
        await _journal_guard(driver).begin()
        driver.markers["episode-a"]["active"] = False
        driver.owner["snapshot_id"] = "another-snapshot"
        with pytest.raises(GuardError, match="generation"):
            await _journal_guard(driver).begin()
        driver.owner["snapshot_id"] = "episode-a:1"
        results = await asyncio.gather(*(_journal_guard(driver).begin() for _ in range(2)), return_exceptions=True)
        assert sum(isinstance(result, GuardError) for result in results) == 1
        assert sum(getattr(result, "state", None) == "PENDING" for result in results) == 1

    asyncio.run(exercise())


def test_generation_fence_terminal_uses_same_transaction_without_deadlock():
    async def exercise():
        driver = _JournalDriver()
        guard = _journal_guard(driver)
        await guard.begin()
        driver.events.clear()
        async with guard.fenced_graph_mutation():
            assert driver.lock.locked()
            await guard.complete({"provider_attempt_number": 1})
            assert driver.lock.locked()
            assert "snapshot-deleted" not in driver.events
        assert driver.events == ["fence", "terminal", "commit", "snapshot-deleted"]

    asyncio.run(asyncio.wait_for(exercise(), 1))


class _PageGuard(Neo4jMutationGuard):
    def __init__(self, rows):
        super().__init__(None, group_id="group-id", episode_uuid="episode-id",
                         attempt_number=1, input_digest="sha256:" + "0" * 64)
        self.rows = rows
        self.writes = []
        self.reads = []
        self.label_writes = []
        self.inventory_reads = []

    async def _identity_inventory(self, query, *, snapshot_label):
        self.inventory_reads.append(query)
        rows = self.rows if snapshot_label == "NewsroomSnapshotNode" else []
        return [(row["source_identity"], row["target_identity"]) for row in rows]

    async def _query(self, query, **params):
        if "unsafe_nodes" in query or "unsafe_relationships" in query:
            return []
        if "UNWIND $page AS row" in query:
            assert "LIMIT" not in query, "only preselected bounded actual identities may reach SET"
            assert 0 < len(params["page"]) <= 64
            self.writes.append(params["page"])
            return [{"written": len(params["page"])}]
        if "REMOVE n:`" in query or "SET n:`" in query:
            assert "elementId(n) = $target_identity" in query
            self.label_writes.append(params["target_identity"])
            return [{"written": 1}]
        if "identities" in params:
            self.reads.append(params)
            selected = {(row["source_identity"], row["target_identity"]) for row in params["identities"]}
            return [row for row in self.rows if (row["source_identity"], row["target_identity"]) in selected]
        assert "ORDER BY source_identity" in query and "LIMIT $limit" in query
        assert "elementId(" in query and "uuid >" not in query
        self.reads.append(params)
        rows = [] if "MATCH (s:NewsroomSnapshotRelationship" in query or "MATCH (a)-[r]->(b)" in query else self.rows
        cursor = (params["cursor_source"], params["cursor_target"])
        return [row for row in rows if (row["source_identity"], row["target_identity"]) > cursor][:params["limit"]]


def _page_row(source, target, *, size=1, expected=(), actual=()):
    return {"source_identity": source, "target_identity": target,
            "source_properties": {"uuid": "duplicate", "vector": [1.0] * size},
            "target_properties": {"uuid": "duplicate", "extra": "remove"},
            "expected": list(expected), "actual": list(actual)}


def test_property_pages_cover_all_actual_duplicate_targets_and_page_boundary():
    rows = [_page_row("snapshot-1", f"target-{index:04}") for index in range(65)]
    guard = _PageGuard(rows)
    asyncio.run(guard._restore_properties())
    assert [len(page) for page in guard.writes] == [64, 1]
    assert [item["target_identity"] for page in guard.writes for item in page] == [row["target_identity"] for row in rows]
    assert len(guard.inventory_reads) == 2
    assert guard.reads[1]["identities"] == [{"source_identity": "snapshot-1", "target_identity": "target-0064"}]


def test_property_page_full_bytes_bound_splits_before_write_and_rejects_single_oversize(monkeypatch):
    import newsroom.graphiti_adapter.neo4j_guard as module
    monkeypatch.setattr(module, "_PAGE_PROPERTY_BYTES", 1024 * 1024)
    rows = [_page_row("snapshot-1", f"target-{index}", size=20000) for index in range(8)]
    guard = _PageGuard(rows)
    asyncio.run(guard._restore_properties())
    assert len(guard.writes) > 1
    assert sum(map(len, guard.writes)) == len(rows)
    oversized = _PageGuard([_page_row("snapshot-1", "target-1", size=200000)])
    with pytest.raises(GuardError, match="property byte bound"):
        asyncio.run(oversized._restore_properties())
    assert not oversized.writes


def test_native_vector_fixture_uses_full_target_pages_under_conservative_byte_bound():
    rows = [_page_row("snapshot-1", f"target-{index:04}", size=1536) for index in range(65)]
    for row in rows:
        row["target_properties"]["vector"] = [1.0] * 1536
    guard = _PageGuard(rows)
    asyncio.run(guard._restore_properties())
    assert [len(page) for page in guard.writes] == [64, 1]


def test_restore_inventory_matches_identities_once_without_property_values():
    guard = _PageGuard([_page_row("snapshot-1", "target-1")])
    asyncio.run(guard._restore_properties())
    assert len(guard.inventory_reads) == 2
    assert all("properties(" not in query for query in guard.inventory_reads)
    assert "uuid: s._newsroom_source_uuid" in guard.inventory_reads[0]
    assert "type(r) = s._newsroom_relationship_type" in guard.inventory_reads[1]


def test_label_pages_repair_each_actual_identity_including_unlabelled_duplicates():
    guard = _PageGuard([
        _page_row("snapshot-1", "target-a", expected=["Entity"], actual=["Changed"]),
        _page_row("snapshot-1", "target-b", expected=["Entity"], actual=[]),
    ])
    asyncio.run(guard._restore_properties())
    assert guard.label_writes == ["target-a", "target-a", "target-b"]
    assert len(guard.reads) == 1, "label repair must reuse property-page identities"


def test_snapshot_pages_commit_stable_original_element_identities():
    rows = [_page_row(f"original-{index:04}", "") for index in range(65)]
    guard = _PageGuard(rows)
    asyncio.run(guard._snapshot())
    assert [len(page) for page in guard.writes] == [64, 1]
    assert guard.reads[1]["cursor_source"] == "original-0063"


def test_committed_rollback_page_crash_retains_durable_owner_until_exact_recovery():
    class InterruptedGuard(_JournalGuard):
        async def _restore_properties(self, **_values):
            self.driver.events.append("property-page-committed")
            raise asyncio.CancelledError()

    async def exercise():
        driver = _JournalDriver()
        guard = InterruptedGuard(driver, group_id="group-id", episode_uuid="episode-a",
                                 attempt_number=1, input_digest="sha256:" + "0" * 64)
        await guard.begin()
        driver.events.clear()
        with pytest.raises(asyncio.CancelledError):
            await guard.rollback_pending(chat_invocations=[], embedding_usage={}, reason="CRASH")
        assert driver.markers["episode-a"]["state"] == "ROLLING_BACK"
        assert driver.owner["owner_marker_uuid"] == "episode-a"
        assert driver.events.index("rollback-committed") < driver.events.index("property-page-committed")
        assert "terminal" not in driver.events and "snapshot-deleted" not in driver.events
        driver.markers["episode-a"]["active"] = False
        with pytest.raises(GuardError, match="generation"):
            await _journal_guard(driver, "episode-b").begin()
        recovery = _journal_guard(driver)
        await recovery.begin()
        await recovery.rollback_pending(chat_invocations=[], embedding_usage={}, reason="RECOVERY")
        assert not driver.owner
        assert driver.events.index("verified") < driver.events.index("terminal") < driver.events.index("snapshot-deleted")

    asyncio.run(exercise())


def test_terminal_marker_readers_remain_available_during_another_generation_owner():
    async def exercise():
        driver = _JournalDriver()
        guard = _journal_guard(driver)
        await guard.begin()
        await guard.complete({"provider_attempt_number": 1})
        await _journal_guard(driver, "episode-b").begin()
        assert (await _journal_guard(driver).begin()).state.value == "COMPLETE"
        assert driver.owner["owner_marker_uuid"] == "episode-b"

    asyncio.run(exercise())


def test_property_bound_measures_native_vectors_without_truncated_display():
    from neo4j.vector import Vector
    from newsroom.graphiti_adapter.neo4j_guard import _property_bytes

    vector = Vector.from_bytes(b"\0" * (1024 * 1024), "f64")
    assert _property_bytes(vector) == 64 + 1024 * 1024


def test_snapshot_refuses_relationship_properties_that_collide_with_guard_metadata():
    class ReservedRelationshipGuard(_PageGuard):
        async def _query(self, query, **params):
            if "unsafe_relationships" in query:
                unsafe = "keys(r)" in query and params.get("reserved_prefix") == "_newsroom_"
                return [{"unsafe_relationships": int(unsafe)}]
            return await super()._query(query, **params)

    guard = ReservedRelationshipGuard([])
    with pytest.raises(GuardError, match="reserved guard properties"):
        asyncio.run(guard._snapshot())
    assert not guard.writes


class _InventoryDriver:
    def __init__(self, rows, count):
        self.rows, self.count = rows, count
        self.queries = []

    def session(self):
        driver = self
        class Result:
            def __init__(self, rows):
                self.rows = rows
            async def single(self, **_kwargs):
                return self.rows[0]
            def __aiter__(self):
                async def iterate():
                    for row in self.rows:
                        yield row
                return iterate()
        class Transaction:
            async def run(self, query, **_parameters):
                driver.queries.append(query)
                return Result([{"snapshot_count": driver.count}] if "count(s)" in query else driver.rows)
        class Session:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_args):
                pass
            async def execute_write(self, callback):
                return await callback(Transaction())
        return Session()


def test_streamed_inventory_preserves_all_duplicate_targets_without_property_values():
    rows = [{"source_identity": "snapshot-1", "target_identity": f"target-{i:04}"} for i in range(65)]
    driver = _InventoryDriver(list(reversed(rows)), 1)
    guard = Neo4jMutationGuard(driver, group_id="group-id", episode_uuid="episode-a",
                               attempt_number=1, input_digest="sha256:" + "0" * 64)
    inventory = asyncio.run(guard._identity_inventory("RETURN identities", snapshot_label="NewsroomSnapshotNode"))
    assert inventory == [(row["source_identity"], row["target_identity"]) for row in rows]
    assert len(driver.queries) == 2


@pytest.mark.parametrize("defect", ["coverage", "duplicate", "absent", "bytes", "count"])
def test_inventory_refuses_missing_duplicate_or_oversized_identities(defect, monkeypatch):
    import newsroom.graphiti_adapter.neo4j_guard as module
    row = {"source_identity": "snapshot-1", "target_identity": "target-1"}
    rows = [row, row] if defect == "duplicate" else [row]
    if defect == "absent":
        rows = [{"source_identity": "snapshot-1", "target_identity": None}]
    if defect == "bytes":
        monkeypatch.setattr(module, "_INVENTORY_BYTES", 300)
    count = 2 if defect == "coverage" else (True if defect == "count" else 1)
    guard = Neo4jMutationGuard(_InventoryDriver(rows, count), group_id="group-id",
                               episode_uuid="episode-a", attempt_number=1,
                               input_digest="sha256:" + "0" * 64)
    with pytest.raises(GuardError):
        asyncio.run(guard._identity_inventory("RETURN identities", snapshot_label="NewsroomSnapshotNode"))


def test_owned_recovery_observes_only_coarse_phases_and_clears_callback_afterwards():
    async def exercise():
        driver = _JournalDriver()
        initial = _journal_guard(driver)
        phases = []
        await initial.begin()
        driver.markers["episode-a"].update(state="ROLLING_BACK", active=False)
        recovery = _journal_guard(driver)
        marker = await recovery.recover_owned_pending(phase_observer=phases.append)
        assert marker.state.value == "RECOVERED_AMBIGUOUS"
        assert phases == [
            "OWNERSHIP", "INVENTORY", "DELETE_NEW_RELATIONSHIPS", "DELETE_NEW_NODES",
            "RESTORE", "FULL_VERIFY", "TERMINAL", "SNAPSHOT_CLEANUP", "TERMINAL",
        ]
        assert recovery._owned_recovery_phase_observer is None
        assert not driver.owner
        assert driver.events.index("verified") < driver.events.index("terminal")
        phases.clear()
        await _journal_guard(driver, "episode-b").begin()
        assert not phases
    asyncio.run(exercise())


@pytest.mark.parametrize("failed_phase", ("INVENTORY", "DELETE_NEW_RELATIONSHIPS", "RESTORE", "FULL_VERIFY"))
def test_owned_recovery_phase_callback_is_reset_when_guard_or_stop_interrupts(failed_phase):
    async def exercise():
        driver = _JournalDriver()
        await _journal_guard(driver).begin()
        driver.markers["episode-a"].update(state="ROLLING_BACK", active=False)
        recovery = _journal_guard(driver)
        phases = []
        def observe(phase):
            phases.append(phase)
            if phase == failed_phase:
                raise GuardError("fixture boundary failure")
        with pytest.raises(GuardError, match="fixture boundary failure"):
            await recovery.recover_owned_pending(phase_observer=observe)
        assert phases[-1] == failed_phase
        assert recovery._owned_recovery_phase_observer is None
        assert driver.markers["episode-a"]["state"] == "ROLLING_BACK"
        assert driver.owner
        assert "terminal" not in driver.events
    asyncio.run(exercise())


def test_rollback_deletion_reads_bounded_uuid_inventory_once_before_each_delete():
    class Guard(_JournalGuard):
        async def _snapshot_uuids(self, inventory, *, snapshot_label):
            self.driver.events.append(snapshot_label)
            return ["original", True, 1, [1, 2]]

    async def exercise():
        driver = _JournalDriver()
        guard = Guard(driver, group_id="group-id", episode_uuid="episode-a", attempt_number=1,
                      input_digest="sha256:" + "0" * 64)
        await guard.begin()
        driver.queries.clear()
        await guard.rollback_pending(chat_invocations=[], embedding_usage={}, reason="FIXTURE")
        deletes = [query for query in driver.queries if "DELETE r" in query or "DETACH DELETE n" in query]
        assert len(deletes) == 2
        assert all("NOT EXISTS" not in query and "MATCH (s:" not in query for query in deletes)
        assert "r.uuid IS NULL OR NOT r.uuid IN $retained_uuids" in deletes[0]
        assert "a.group_id = $group_id OR b.group_id = $group_id" in deletes[0]
        assert "n.uuid IS NULL OR NOT n.uuid IN $retained_uuids" in deletes[1]
        assert all(f"NOT n:{label}" in deletes[1] for label in (
            "NewsroomSnapshotNode", "NewsroomSnapshotRelationship", "NewsroomIngestMarker",
        ))
        assert driver.events.index("NewsroomSnapshotNode") < driver.events.index("properties")
        assert driver.events.index("NewsroomSnapshotRelationship") < driver.events.index("properties")
    asyncio.run(exercise())


@pytest.mark.parametrize("defect", (None, "bytes", "coverage", "duplicate", "null", "fetch"))
def test_retained_uuid_stream_is_fetch_one_bounded_and_preserves_native_values(defect, monkeypatch):
    import newsroom.graphiti_adapter.neo4j_guard as module

    values = ["original", "original", True, 1, [1, 2]]
    if defect == "bytes":
        values = ["x" * 4096] * 200
        monkeypatch.setattr(module, "_INVENTORY_BYTES", 1024)
    rows = [{"source_identity": f"snapshot-{index}", "retained_uuid": value}
            for index, value in enumerate(values)]
    inventory = [(row["source_identity"], f"actual-{index}") for index, row in enumerate(rows)]
    if defect == "coverage":
        inventory.append(("snapshot-missing", "actual-missing"))
    elif defect == "duplicate":
        rows.append(rows[0])
    elif defect == "null":
        rows[0]["retained_uuid"] = None
    yielded, configurations = [], []
    class Result:
        async def __aiter__(self):
            for row in rows:
                yielded.append(row)
                yield row
    class Transaction:
        async def run(self, query, **parameters):
            assert configurations[-1].fetch_size == 1
            assert parameters == {"snapshot_id": "episode-a:1"}
            assert query.count("MATCH (s:NewsroomSnapshotNode") == 1
            assert "s._newsroom_source_uuid AS retained_uuid" in query
            assert "properties(s)" not in query and "uuid:" not in query
            return Result()
    class Session:
        def __init__(self):
            if defect != "fetch":
                self._config = SimpleNamespace(fetch_size=1000)
                configurations.append(self._config)
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_args):
            pass
        async def execute_write(self, callback):
            return await callback(Transaction())
    guard = Neo4jMutationGuard(SimpleNamespace(session=Session), group_id="group-id",
                               episode_uuid="episode-a", attempt_number=1,
                               input_digest="sha256:" + "0" * 64)
    if defect is not None:
        with pytest.raises(GuardError):
            asyncio.run(guard._snapshot_uuids(inventory, snapshot_label="NewsroomSnapshotNode"))
        if defect == "bytes":
            assert len(yielded) == 1
        elif defect == "fetch":
            assert not yielded
    else:
        retained = asyncio.run(guard._snapshot_uuids(inventory, snapshot_label="NewsroomSnapshotNode"))
        assert all(actual is expected for actual, expected in zip(retained, values, strict=True))
        assert len(retained) == len(values)
        assert type(retained[2]) is bool and type(retained[3]) is int


class _SnapshotCleanupDriver(_JournalDriver):
    def __init__(self):
        super().__init__()
        self.snapshots = {}
        self.deleted = []
        self.cleanup_queries = []
        self.cancel_cleanup_at = None

    def apply(self, query, params):
        if 'DELETE s RETURN count(*) AS deleted' not in query:
            return super().apply(query, params)
        assert 'MATCH (s)\n' not in query and ' OR ' not in query
        label, = [label for label in ('NewsroomSnapshotNode', 'NewsroomSnapshotRelationship')
                  if f'MATCH (s:{label} {{_newsroom_snapshot_id: $snapshot_id}})' in query]
        assert params == {'snapshot_id': 'episode-a:1', 'limit': 64}
        self.cleanup_queries.append((label, params.copy()))
        if len(self.cleanup_queries) == self.cancel_cleanup_at:
            raise asyncio.CancelledError()
        selected = [key for key, (labels, snapshot) in self.snapshots.items()
                    if label in labels and snapshot == params['snapshot_id']][:params['limit']]
        for key in selected:
            del self.snapshots[key]
        self.deleted.extend(selected)
        self.events.append('snapshot-deleted')
        return [{'deleted': len(selected)}]


class _ActualCleanupJournalGuard(_JournalGuard):
    _delete_snapshot = Neo4jMutationGuard._delete_snapshot


def _cleanup_fixture():
    driver = _SnapshotCleanupDriver()
    for label, count in [('NewsroomSnapshotNode', 129), ('NewsroomSnapshotRelationship', 65)]:
        for index in range(count):
            driver.snapshots[f'{label}-{index}'] = ({label}, 'episode-a:1')
    driver.snapshots['dual-labelled'] = ({'NewsroomSnapshotNode', 'NewsroomSnapshotRelationship'}, 'episode-a:1')
    driver.snapshots['other-generation'] = ({'NewsroomSnapshotNode'}, 'episode-b:1')
    driver.snapshots['ordinary-node'] = ({'Entity'}, 'episode-a:1')
    guard = _ActualCleanupJournalGuard(driver, group_id='group-id', episode_uuid='episode-a',
                                      attempt_number=1, input_digest='sha256:' + '0' * 64)
    return driver, guard


def test_snapshot_cleanup_uses_exact_labelled_bounded_pages_and_protects_other_snapshots():
    driver, guard = _cleanup_fixture()
    asyncio.run(guard._delete_snapshot())
    assert len(driver.deleted) == len(set(driver.deleted)) == 195
    assert driver.snapshots == {'other-generation': ({'NewsroomSnapshotNode'}, 'episode-b:1'),
                                'ordinary-node': ({'Entity'}, 'episode-a:1')}
    assert [label for label, _ in driver.cleanup_queries] == ['NewsroomSnapshotNode'] * 3 + ['NewsroomSnapshotRelationship'] * 2
    asyncio.run(guard._delete_snapshot())
    assert len(driver.deleted) == 195, 'repeat cleanup must be idempotent'


@pytest.mark.parametrize('count', [None, True, -1, 65, 1.0, '1'])
def test_snapshot_cleanup_rejects_invalid_per_query_count(count):
    driver, guard = _cleanup_fixture()
    async def invalid(_query, **params):
        assert params['limit'] == 64
        return [{'deleted': count}]
    guard._query = invalid
    with pytest.raises(GuardError, match='snapshot deletion count is invalid'):
        asyncio.run(guard._delete_snapshot())
    assert len(driver.snapshots) == 197


@pytest.mark.parametrize('terminal', ['COMPLETE', 'RECOVERED_AMBIGUOUS'])
def test_snapshot_cleanup_preserves_terminal_replay_and_exact_owned_recovery(terminal):
    async def exercise():
        driver, guard = _cleanup_fixture()
        await guard.begin()
        if terminal == 'COMPLETE':
            await guard.complete({'provider_attempt_number': 1})
        else:
            driver.markers['episode-a']['active'] = False
            marker = await guard.recover_owned_pending()
            assert marker.state.value == terminal
        assert driver.markers['episode-a']['state'] == terminal
        assert not driver.owner
        assert len(driver.deleted) == 195
        replay = _ActualCleanupJournalGuard(driver, group_id='group-id', episode_uuid='episode-a',
                                            attempt_number=1, input_digest='sha256:' + '0' * 64)
        assert (await replay.begin()).state.value == terminal
        assert len(driver.deleted) == 195
    asyncio.run(exercise())


def test_snapshot_cleanup_cancellation_keeps_completed_marker_and_replay_finishes_remainder():
    async def exercise():
        driver, guard = _cleanup_fixture()
        await guard.begin()
        driver.cancel_cleanup_at = 2
        with pytest.raises(asyncio.CancelledError):
            await guard.complete({'provider_attempt_number': 1})
        assert driver.markers['episode-a']['state'] == 'COMPLETE' and not driver.owner
        assert len(driver.deleted) == 64
        driver.cancel_cleanup_at = None
        assert (await guard.begin()).state.value == 'COMPLETE'
        assert len(driver.deleted) == len(set(driver.deleted)) == 195
        assert set(driver.snapshots) == {'other-generation', 'ordinary-node'}
    asyncio.run(exercise())


def test_snapshot_cleanup_bootstrap_has_two_idempotent_labelled_snapshot_indexes():
    async def exercise():
        queries = []
        async def query(cypher, **_values):
            queries.append(cypher)
            return [], None, None
        await Neo4jMutationGuard.bootstrap_schema(SimpleNamespace(execute_query=query))
        indexes = [query for query in queries if 'CREATE INDEX' in query]
        assert len(indexes) == 2
        for label in ('NewsroomSnapshotNode', 'NewsroomSnapshotRelationship'):
            query, = [query for query in indexes if f'FOR (s:{label})' in query]
            assert 'IF NOT EXISTS' in query and 'ON (s._newsroom_snapshot_id)' in query
    asyncio.run(exercise())


def test_passive_guard_begin_and_cleanup_timers_do_not_change_marker_or_delete_outcomes(monkeypatch):
    import newsroom.graphiti_adapter.neo4j_guard as module
    events = []
    monkeypatch.setattr(module._LOGGER, 'info', lambda event, *, extra: events.append((event, extra['diagnostic_data'])))
    async def exercise():
        driver, guard = _cleanup_fixture()
        assert (await guard.begin()).state.value == 'CREATED'
        await guard.complete({'provider_attempt_number': 1})
        assert driver.markers['episode-a']['state'] == 'COMPLETE' and len(driver.deleted) == 195
    asyncio.run(exercise())
    assert [data['phase'] for _, data in events] == ['BEGIN', 'SNAPSHOT_CLEANUP']
    assert all(event == 'graphiti_guard_phase' and data['episode_id'] == 'episode-a'
               and data['attempt_number'] == 1 and data['status'] == 'COMPLETE'
               and type(data['elapsed_ms']) is int and type(data['cpu_ms']) is int
               and data['cpu_scope'] == 'PROCESS' and data['nested_spans_not_additive']
               for event, data in events)


@pytest.mark.parametrize('diagnostic_defect', ['drop', 'clock'])
def test_passive_guard_cleanup_timer_failure_preserves_original_cancellation(monkeypatch, diagnostic_defect):
    import newsroom.graphiti_adapter.neo4j_guard as module
    def fail(*_args, **_values):
        raise OSError('optional diagnostic failure')
    monkeypatch.setattr(module._LOGGER, 'info', fail)
    if diagnostic_defect == 'clock':
        monkeypatch.setattr(module, 'perf_counter_ns', fail, raising=False)
    driver, guard = _cleanup_fixture()
    driver.cancel_cleanup_at = 1
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(guard._delete_snapshot())
    assert not driver.deleted


@pytest.mark.parametrize('failure', [None, RuntimeError('original begin failure'), asyncio.CancelledError()])
def test_passive_guard_begin_timer_records_exact_clock_and_preserves_exception(monkeypatch, failure):
    import newsroom.graphiti_adapter.neo4j_guard as module
    events = []
    wall, cpu = iter([10_000_000, 17_000_000]), iter([20_000_000, 23_000_000])
    monkeypatch.setattr(module, 'perf_counter_ns', lambda: next(wall))
    monkeypatch.setattr(module, 'process_time_ns', lambda: next(cpu))
    monkeypatch.setattr(module._LOGGER, 'info', lambda event, *, extra: events.append(extra['diagnostic_data']))
    class SelectedGuard(_JournalGuard):
        async def _snapshot(self):
            if failure is not None:
                raise failure
    driver = _JournalDriver()
    guard = SelectedGuard(driver, group_id='group-id', episode_uuid='episode-a',
                          attempt_number=1, input_digest='sha256:' + '0' * 64)
    if failure is None:
        assert asyncio.run(guard.begin()).state.value == 'CREATED'
    else:
        with pytest.raises(type(failure)) as error:
            asyncio.run(guard.begin())
        assert error.value is failure
    assert events == [{
        'phase': 'BEGIN', 'episode_id': 'episode-a', 'attempt_number': 1,
        'status': 'COMPLETE' if failure is None else 'FAILED',
        'failure_class': 'NONE' if failure is None else type(failure).__name__,
        'elapsed_ms': 7, 'cpu_ms': 3, 'cpu_scope': 'PROCESS',
        'nested_spans_not_additive': True,
    }]
