"""Actual selected v43→v44 copy must continue normal current projection."""
from dataclasses import replace
import sqlite3

import pytest

from newsroom.authority._increment4_projection_store import _Increment4ProjectionAuthorityStore
from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store
from newsroom.authority.native_current_rebuild import copy_selected_native_store
from newsroom.increment4 import Increment4Neo4jCurrentBuildRequest
from newsroom.projection import ProjectionGenerationId, ProjectionGenerationState
from newsroom.tests.authority_helpers import command
from newsroom.tests.authority_event_helpers import open_test_system
from newsroom.tests.extraction_4a_helpers import extraction_proof
from newsroom.tests.increment4e_governed_path_helpers import (
    seed_increment4_graphiti_path, admit_increment4_graphiti_path,
    open_graphiti_path_increment4_neo4j_system,
)
from newsroom.tests.projection_b2_helpers import MemoryNeo4jAdapter


@pytest.fixture
def sparse_current(tmp_path, monkeypatch):
    import newsroom.tests.graphiti_adapter_4d_authority_helpers as helpers

    original = helpers.seed_extraction_fixture
    missing = []
    def seed_with_unused(root):
        state = original(root)
        with open_test_system(state.database, registry=state.commands, payload_schema_registry=state.schemas) as system:
            missing.append(system.commands.execute(command(key="expired-before-current-entity"), proof=extraction_proof()))
        return state
    monkeypatch.setattr(helpers, "seed_extraction_fixture", seed_with_unused)
    # This is the existing native CURRENT store mode, not a new fixture engine.
    monkeypatch.setattr(_Increment4ProjectionAuthorityStore, "_current_state_only", True)
    captured = []
    initialize = _Increment4ProjectionAuthorityStore.__init__
    def capture(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        captured.append(self)
    monkeypatch.setattr(_Increment4ProjectionAuthorityStore, "__init__", capture)
    state = seed_increment4_graphiti_path(tmp_path / "original")
    admit_increment4_graphiti_path(state)
    destination = tmp_path / "selected.sqlite3"
    adapter = MemoryNeo4jAdapter()
    with open_graphiti_path_increment4_neo4j_system(state.relation, adapter):
        root = captured[-1]
        snapshot = root.increment4_admitted_snapshot()
        roots = {
            "canonical_entities": tuple((str(item.entity.entity_id),) for item in snapshot.entities),
            "canonical_entity_heads": tuple((str(item.entity.entity_id),) for item in snapshot.entities),
            "entity_projection_events": tuple((str(item.projection_event.projection_event_id),) for item in snapshot.entities),
            "editorial_relation_assertions": tuple((str(item.current.assertion.assertion_id),) for item in snapshot.relations),
            "editorial_relation_assertion_heads": tuple((str(item.current.assertion.assertion_id),) for item in snapshot.relations),
            "editorial_relation_projection_events": tuple((str(item.projection_event.projection_event_id),) for item in snapshot.relations),
        }
        roots["entity_resolution_proposal_heads"] = tuple(
            (row[0],) for row in root._connection.execute(
                "SELECT resolution_proposal_id FROM entity_resolution_decisions WHERE decision_id IN ("
                + ",".join("?" for _ in snapshot.entities) + ")",
                tuple(str(item.entity.created_by_decision_id) for item in snapshot.entities),
            )
        )
        roots["entity_resolution_decision_heads"] = tuple(
            (row[0],) for row in root._connection.execute("SELECT resolution_proposal_id FROM entity_resolution_decision_heads")
        )
        with sqlite3.connect(destination, isolation_level=None) as connection:
            initialise_empty_checkpoint_store(connection)
            copy_selected_native_store(root, connection, roots=roots, dev_rebuild=True)
            assert connection.execute("SELECT 1 FROM ledger_events WHERE ledger_seq=?", (missing[0].ledger_seq,)).fetchone() is None
            assert connection.execute("SELECT event_id FROM native_expired_command_keys WHERE ledger_seq=?", (missing[0].ledger_seq,)).fetchone()[0] == str(missing[0].event_id)
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    destination.chmod(0o600)
    extraction = replace(state.extraction, database=destination)
    relation = replace(state.relation, entity=replace(state.relation.entity, extraction=extraction))
    yield relation, adapter, destination, missing[0], captured


def test_actual_selected_copy_continues_current_build_and_receipt(sparse_current):
    relation, adapter, database, expired, captured = sparse_current
    request = Increment4Neo4jCurrentBuildRequest(ProjectionGenerationId.new(), "CURRENT_AFTER_SELECTED_COPY", "sparse-current", allow_active_extension=True)
    with open_graphiti_path_increment4_neo4j_system(relation, adapter) as system:
        result = system.increment4.build_current_and_promote(request, proof=extraction_proof())
        assert result.generation.state is ProjectionGenerationState.ACTIVE
        assert result.checkpoint_ledger_seq >= result.source_watermark_ledger_seq
        assert result.promotion.generation.generation_id == result.generation.generation_id
        assert result.validation.projection_state_digest == result.projection_state_digest
        assert adapter.apply_count > 0
        before = captured[-1]._connection.total_changes
        replay = system.increment4.build_current_and_promote(request, proof=extraction_proof())
        assert replay.validation == result.validation
        assert captured[-1]._connection.total_changes == before
    with open_graphiti_path_increment4_neo4j_system(relation, adapter) as system:
        replay = system.increment4.build_current_and_promote(request, proof=extraction_proof())
        assert replay.validation == result.validation


@pytest.mark.parametrize("fault", ("unknown_sequence", "live_sequence_alias"))
def test_unknown_or_unbound_reservation_never_becomes_accepted_checkpoint(sparse_current, fault):
    from newsroom.authority import AuthorityPersistenceError
    from newsroom.projection import ProjectionStateError
    from newsroom.projection.neo4j import Neo4jAuthorityCommitPending

    relation, adapter, database, expired, captured = sparse_current
    with sqlite3.connect(database) as connection:
        if fault == "unknown_sequence":
            guard = connection.execute("SELECT sql FROM sqlite_schema WHERE name='immutable_native_expired_command_keys_delete'").fetchone()[0]
            connection.execute("DROP TRIGGER immutable_native_expired_command_keys_delete")
            connection.execute("DELETE FROM native_expired_command_keys WHERE ledger_seq=?", (expired.ledger_seq,))
        else:
            guard = connection.execute("SELECT sql FROM sqlite_schema WHERE name='immutable_native_expired_command_keys_update'").fetchone()[0]
            connection.execute("DROP TRIGGER immutable_native_expired_command_keys_update")
            live = connection.execute("SELECT ledger_seq FROM ledger_events ORDER BY ledger_seq DESC LIMIT 1").fetchone()[0]
            connection.execute("UPDATE native_expired_command_keys SET ledger_seq=? WHERE ledger_seq=?", (live, expired.ledger_seq))
        connection.execute(guard)
    request = Increment4Neo4jCurrentBuildRequest(ProjectionGenerationId.new(), "SPARSE_DENIED", "sparse-denied")
    with open_graphiti_path_increment4_neo4j_system(relation, adapter) as system:
        with pytest.raises((AuthorityPersistenceError, ProjectionStateError, Neo4jAuthorityCommitPending)):
            system.increment4.build_current_and_promote(request, proof=extraction_proof())
        root = captured[-1]
        assert root._connection.execute("SELECT 1 FROM projection_generation_promotions WHERE generation_id=?", (str(request.generation_id),)).fetchone() is None
        assert root._connection.execute("SELECT 1 FROM projection_generations WHERE generation_id=? AND state='ACTIVE'", (str(request.generation_id),)).fetchone() is None


def test_reserved_coverage_keeps_actual_mapped_inputs_and_exact_end_bound(sparse_current):
    from newsroom.increment4.contracts import INCREMENT4_ADMITTED_FAMILY_ID

    relation, adapter, database, expired, captured = sparse_current
    generation = ProjectionGenerationId.new()
    with open_graphiti_path_increment4_neo4j_system(relation, adapter) as system:
        boundary = system.increment4._Increment4Neo4jController__build_current.__self__
        boundary._register_family(extraction_proof())
        root = captured[-1]
        family = root._registered_family_definition(root._connection, INCREMENT4_ADMITTED_FAMILY_ID)
        inputs = root._increment4_current_build_inputs(generation_id=generation, family=family)
        boundary._create_generation(
            request=Increment4Neo4jCurrentBuildRequest(generation, "SPARSE_LIMIT", "sparse-limit"),
            snapshot_digest=inputs.snapshot_digest, proof=extraction_proof(),
        )
        family = root._registered_family_definition(root._connection, INCREMENT4_ADMITTED_FAMILY_ID)
        mappings = root._projection_contracts.mappings.resolve_digest(family.mapping_contract_digest)
        assert inputs.batches
        limit = min(expired.ledger_seq - 1, inputs.batches[0].ledger_seq - 1)
        bounded = root._skippable_checkpoint_candidate(root._connection, generation_id=str(generation), current=0, maximum_ledger_seq=limit)
        assert bounded <= limit
        candidate = root._skippable_checkpoint_candidate(root._connection, generation_id=str(generation), current=0, maximum_ledger_seq=None)
        assert candidate >= expired.ledger_seq
        assert root._connection.execute("SELECT count(*) FROM projection_delivery_states WHERE generation_id=?", (str(generation),)).fetchone()[0] == 0
