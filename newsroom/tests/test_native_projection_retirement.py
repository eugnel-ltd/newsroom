"""Native replacement expiry is one SQLite promotion transaction, not best effort."""
import sqlite3
from dataclasses import replace

import pytest

from newsroom.authority import projection_retirement as retirement
from newsroom.increment4 import Increment4Neo4jCurrentBuildRequest
from newsroom.projection import ProjectionGenerationId
from .increment4e_governed_path_helpers import (seed_increment4_graphiti_path,
    admit_increment4_graphiti_path, open_graphiti_path_increment4_neo4j_system)
from .projection_b2_helpers import MemoryNeo4jAdapter
from .extraction_4a_helpers import extraction_proof
from .test_increment4_active_extension import G1, G2


def _request(generation, key):
    return Increment4Neo4jCurrentBuildRequest(generation, 'NATIVE_RETIREMENT_TEST', key,
                                            allow_active_extension=False)


def _seed(tmp_path):
    state = seed_increment4_graphiti_path(tmp_path)
    admit_increment4_graphiti_path(state)
    adapter = MemoryNeo4jAdapter()
    with open_graphiti_path_increment4_neo4j_system(state.relation, adapter) as system:
        first = system.increment4.build_current_and_promote(_request(G1, 'first'), proof=extraction_proof())
    return state, adapter, first


def test_native_replacement_expires_only_predecessor_and_replay_is_stable(tmp_path):
    state, adapter, first = _seed(tmp_path)
    with open_graphiti_path_increment4_neo4j_system(state.relation, adapter) as system:
        second = system.increment4.build_current_and_promote(_request(G2, 'second'), proof=extraction_proof())
        assert second.promotion.prior_generation.generation_id == G1
        conn = system.increment4._Increment4Neo4jController__build_current.__self__._store._connection
        rows = conn.execute('SELECT aggregate_id,count(*) FROM ledger_events WHERE retired_header_digest IS NOT NULL GROUP BY aggregate_id').fetchall()
        assert rows and {row[0] for row in rows} == {str(G1)}
        assert conn.execute('SELECT count(*) FROM projection_delivery_states WHERE generation_id=?', (str(G2),)).fetchone()[0] > 0
        assert conn.execute('SELECT count(*) FROM projection_checkpoint_versions WHERE generation_id=?', (str(G1),)).fetchone()[0] == 1
        assert conn.execute("SELECT name FROM sqlite_temp_schema WHERE name LIKE '_retirement_%'").fetchall() == []
        assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
        before = conn.execute('SELECT count(*) FROM ledger_events WHERE retired_header_digest IS NOT NULL').fetchone()[0]
        replay = system.increment4.build_current_and_promote(_request(G2, 'second'), proof=extraction_proof())
        assert replay.promotion.promotion_digest == second.promotion.promotion_digest
        assert conn.execute('SELECT count(*) FROM ledger_events WHERE retired_header_digest IS NOT NULL').fetchone()[0] == before


@pytest.mark.parametrize('failure', [RuntimeError('expiry failed'), KeyboardInterrupt('expiry interrupted')])
def test_expiry_failure_rolls_back_promotion_and_all_diagnostic_deletes(tmp_path, monkeypatch, failure):
    state, adapter, _ = _seed(tmp_path)
    original = retirement.retire_predecessor_diagnostics
    observed = []
    def fail(conn, predecessor):
        assert conn.in_transaction
        assert conn.execute('SELECT state FROM projection_generations WHERE generation_id=?', (predecessor,)).fetchone()[0] == 'RETIRED'
        original(conn, predecessor)
        observed.append(conn.execute('SELECT count(*) FROM ledger_events WHERE retired_header_digest IS NOT NULL').fetchone()[0])
        raise failure
    monkeypatch.setattr(retirement, 'retire_predecessor_diagnostics', fail)
    with open_graphiti_path_increment4_neo4j_system(state.relation, adapter) as system:
        with pytest.raises(type(failure)):
            system.increment4.build_current_and_promote(_request(G2, 'failed-second'), proof=extraction_proof())
        conn = system.increment4._Increment4Neo4jController__build_current.__self__._store._connection
        assert observed and observed[0] > 0
        assert not conn.in_transaction
        assert conn.execute('SELECT state FROM projection_generations WHERE generation_id=?', (str(G1),)).fetchone()[0] == 'ACTIVE'
        assert conn.execute('SELECT state FROM projection_generations WHERE generation_id=?', (str(G2),)).fetchone()[0] == 'VALIDATING'
        assert conn.execute('SELECT count(*) FROM ledger_events WHERE retired_header_digest IS NOT NULL').fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM projection_generation_promotions WHERE generation_id=?', (str(G2),)).fetchone()[0] == 0
        assert conn.execute("SELECT name FROM sqlite_temp_schema WHERE name LIKE '_retirement_%'").fetchall() == []
        assert conn.execute('PRAGMA foreign_key_check').fetchall() == []


@pytest.mark.parametrize('size', [8, 64])
def test_whole_scoped_expiry_cost_does_not_follow_unrelated_authority_history(tmp_path, size, record_property):
    from .test_retired_projection_audit import _seed as seed_delivery
    from .projection_b1_helpers import open_projection_system, proof
    from newsroom.authority import AggregateId, InlinePayload, SemanticCommand
    costs = []
    for history in (0, 512):
        path = tmp_path / f'cohort-{size}-history-{history}.sqlite3'
        request, _ = seed_delivery(path, delivery_count=size)
        with open_projection_system(path) as system:
            for number in range(history):
                system.commands.execute(SemanticCommand('candidate.fixture.write', AggregateId.new(), 0,
                    InlinePayload({'headline':'unrelated','count':number}), f'unrelated-{number}'), proof=proof())
        with sqlite3.connect(path) as conn:
            conn.execute('PRAGMA foreign_keys=ON')
            conn.execute('BEGIN')
            steps = [0]
            conn.set_progress_handler(lambda: steps.__setitem__(0, steps[0]+100) or 0, 100)
            deleted = retirement.retire_predecessor_diagnostics(conn, str(request.generation_id))
            conn.set_progress_handler(None, 0)
            assert deleted['projection_delivery_states'] == size
            conn.commit()
            assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
            assert conn.execute("SELECT name FROM sqlite_temp_schema WHERE name LIKE '_retirement_%'").fetchall() == []
            costs.append(steps[0])
    assert abs(costs[1] - costs[0]) <= 100, costs
    record_property('whole_scoped_expiry_VM_steps', str(costs))
