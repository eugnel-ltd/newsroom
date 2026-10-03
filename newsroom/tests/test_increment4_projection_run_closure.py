"""A projection read shares immutable extraction proof, never live rights."""

from collections import Counter

import pytest

from newsroom.authority._extraction_store_read import _ExtractionReadMixin
from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.extraction.types import ProposalEnvelopeId
from newsroom.extraction.models import ExtractionInputBinding
from newsroom.tests.increment4e_governed_path_helpers import (
    seed_increment4_graphiti_path, admit_increment4_graphiti_path,
    open_graphiti_path_increment4_neo4j_system,
)
from newsroom.tests.projection_b2_helpers import MemoryNeo4jAdapter


@pytest.fixture
def projection_store(tmp_path):
    state = seed_increment4_graphiti_path(tmp_path)
    admit_increment4_graphiti_path(state)
    with open_graphiti_path_increment4_neo4j_system(state.relation, MemoryNeo4jAdapter()) as system:
        store = system.increment4._Increment4Neo4jController__build_current.__self__._store
        store._current_state_only = True
        yield store


def test_current_read_decodes_complete_run_once_and_next_read_revalidates(
    projection_store, monkeypatch,
):
    store = projection_store
    reference = store._increment4_admitted_states()
    original = _ExtractionReadMixin._run_version_from_row
    calls = Counter()

    def counted(self, conn, row, *, replayed):
        calls[str(row["run_version_id"])] += 1
        return original(self, conn, row, replayed=replayed)

    monkeypatch.setattr(_ExtractionReadMixin, "_run_version_from_row", counted)
    for _ in range(2):
        calls.clear()
        with store._increment4_projection_read() as (_, entities, relations, watermark):
            assert (entities, relations, watermark) == reference
            assert len(entities) == 2 and len(relations) == 1
        assert list(calls.values()) == [1]
        assert getattr(store, "_increment4_run_closure", None) is None


def _proposal_ids(store):
    return tuple(ProposalEnvelopeId.parse(row[0]) for row in store._connection.execute(
        "SELECT DISTINCT source_proposal_id FROM entity_mentions ORDER BY source_proposal_id",
    ))


def _damage(store, kind, selected):
    connection = store._connection
    table = "extraction_proposals" if kind == "unused_proposal" else (
        "extraction_outputs" if kind == "output" else "extraction_run_versions"
    )
    triggers = connection.execute(
        "SELECT name FROM sqlite_schema WHERE type='trigger' AND tbl_name=? AND sql LIKE '%BEFORE UPDATE%'",
        (table,),
    ).fetchall()
    for row in triggers:
        connection.execute('DROP TRIGGER "' + row[0] + '"')
    if kind == "unused_proposal":
        other = connection.execute(
            "SELECT proposal_id FROM extraction_proposals WHERE run_version_id=("
            "SELECT run_version_id FROM extraction_proposals WHERE proposal_id=?) "
            "AND proposal_id<>? ORDER BY proposal_id LIMIT 1", (str(selected), str(selected)),
        ).fetchone()[0]
        connection.execute("UPDATE extraction_proposals SET canonical_bytes=? WHERE proposal_id=?",
                           (b"{}", other))
    elif kind == "normalized":
        connection.execute("UPDATE extraction_run_versions SET input_bytes=input_bytes+1")
    elif kind == "output":
        connection.execute("UPDATE extraction_outputs SET canonical_bytes=?,byte_length=2", (b"{}",))
    else:
        connection.execute('UPDATE "' + table + '" SET canonical_bytes=?', (b"{}",))


@pytest.mark.parametrize("kind", ("canonical", "normalized", "output", "unused_proposal"))
@pytest.mark.parametrize("inside", (False, True))
def test_same_count_run_damage_is_rejected_before_or_within_scope(
    projection_store, kind, inside,
):
    store = projection_store
    selected = _proposal_ids(store)[0]
    if inside:
        with store._increment4_projection_read():
            assert store._increment4_run_closure["bindings"]
            _damage(store, kind, selected)
            with pytest.raises(AuthorityPersistenceError):
                store._source_proposal(store._connection, selected)
    else:
        _damage(store, kind, selected)
        with pytest.raises(AuthorityPersistenceError):
            with store._increment4_projection_read():
                pass
    assert getattr(store, "_increment4_run_closure", None) is None


def test_reused_binding_remains_small_and_rights_are_checked_on_every_use(
    projection_store, monkeypatch,
):
    store = projection_store
    original = _ExtractionReadMixin._revalidate_input_binding_current
    checked = []
    deny = [False]

    def current(self, conn, binding):
        checked.append(binding)
        if deny[0]:
            raise PermissionError("current fixture rights revoked")
        return original(self, conn, binding)

    monkeypatch.setattr(_ExtractionReadMixin, "_revalidate_input_binding_current", current)
    selected = _proposal_ids(store)[0]
    with store._increment4_projection_read():
        scope = store._increment4_run_closure
        assert len(scope["bindings"]) == 1
        binding = next(iter(scope["bindings"].values()))
        assert type(binding) is ExtractionInputBinding
        assert not hasattr(binding, "output") and not hasattr(binding, "proposal_set")
        before = len(checked)
        first = store._source_proposal(store._connection, selected)
        assert store._source_proposal(store._connection, selected) == first
        assert len(checked) == before + 2
        assert checked[-1] is binding
        deny[0] = True
        with pytest.raises(PermissionError, match="rights revoked"):
            store._source_proposal(store._connection, selected)
    assert scope["bindings"] == {}
    assert getattr(store, "_increment4_run_closure", None) is None


def test_nested_scopes_restore_parent_and_exception_discards_own_bindings(
    projection_store,
):
    store = projection_store
    with store._increment4_projection_read():
        parent = store._increment4_run_closure
        with pytest.raises(RuntimeError, match="fixture interruption"):
            with store._increment4_projection_read():
                inner = store._increment4_run_closure
                assert inner is not parent and inner["bindings"]
                raise RuntimeError("fixture interruption")
        assert inner["bindings"] == {}
        assert store._increment4_run_closure is parent
        assert parent["bindings"]
        assert store._connection.in_transaction
    assert parent["bindings"] == {}
    assert not store._connection.in_transaction


def test_same_connection_valid_mutation_and_nested_rollback_force_revalidation(
    projection_store, monkeypatch,
):
    store = projection_store
    original = _ExtractionReadMixin._run_version_from_row
    calls = []

    def counted(self, conn, row, *, replayed):
        calls.append(row["run_version_id"])
        return original(self, conn, row, replayed=replayed)

    monkeypatch.setattr(_ExtractionReadMixin, "_run_version_from_row", counted)
    selected = _proposal_ids(store)[0]
    store._connection.execute("CREATE TEMP TABLE fixture_work(value TEXT)")
    with store._increment4_projection_read():
        assert len(calls) == 1
        with store._increment4_projection_read():
            assert len(calls) == 2
            store._connection.execute("INSERT INTO fixture_work VALUES ('discarded')")
        assert store._connection.execute("SELECT count(*) FROM fixture_work").fetchone()[0] == 0
        store._source_proposal(store._connection, selected)
        assert len(calls) == 3
    store._source_proposal(store._connection, selected)
    assert len(calls) == 4  # Public/outside reads never borrow an expired scope.
