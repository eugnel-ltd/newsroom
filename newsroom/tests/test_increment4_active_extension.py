from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from newsroom.authority.persistence import DiagnosticHistoryExpired
from newsroom.increment4 import Increment4Neo4jCurrentBuildRequest, Increment4Neo4jActiveReadRequest
from newsroom.projection import ProjectionDeliveryOutcome, ProjectionGenerationId, ProjectionStateError
from newsroom.projection.neo4j import Neo4jAuthorityCommitPending, Neo4jIdentityConflict
from newsroom.relations import EditorialRelationDecisionAction, EditorialRelationTemporalScope

from .editorial_relation_4c_helpers import (
    RELATION_SECOND_ACCEPT_DECISION_ID, RELATION_SECOND_ASSERTION_ID,
    RELATION_ASSERTION_ID, RELATION_SECOND_DECISION_ID,
    RELATION_SECOND_PROPOSAL_ID, RELATION_SECOND_PROPOSAL_V1_ID,
    relation_decision_request, relation_proposal_request,
)
from .extraction_4a_helpers import extraction_proof
from .increment4e_governed_path_helpers import (
    admit_increment4_graphiti_path, open_graphiti_path_increment4_neo4j_system,
    open_graphiti_path_relation_system, seed_increment4_graphiti_path,
)
from .projection_b2_helpers import MemoryNeo4jAdapter
from .source_3a_helpers import SOURCE_NOW

G1 = ProjectionGenerationId.parse('00000000-0000-4000-8000-000000008101')
G2 = ProjectionGenerationId.parse('00000000-0000-4000-8000-000000008102')
G3 = ProjectionGenerationId.parse('00000000-0000-4000-8000-000000008103')


def request(generation=G1, key='initial'):
    return Increment4Neo4jCurrentBuildRequest(
        generation, 'GRAPHITI_ADMISSION_COHORT', key, allow_active_extension=True,
    )


def build(state, adapter, req):
    with open_graphiti_path_increment4_neo4j_system(state.relation, adapter) as system:
        return system.increment4.build_current_and_promote(req, proof=extraction_proof())


def add_suffix(state):
    with open_graphiti_path_relation_system(state.relation) as system:
        proposal = system.relations.propose(replace(
            relation_proposal_request(state.relation),
            proposal_id=RELATION_SECOND_PROPOSAL_ID,
            proposal_version_id=RELATION_SECOND_PROPOSAL_V1_ID,
            temporal_scope=EditorialRelationTemporalScope(
                valid_from=SOURCE_NOW, valid_until=None, observed_at=SOURCE_NOW,
            ),
            statement='A later admitted relation extends the existing projection.',
            idempotency_key='active-suffix-proposal',
        ), proof=extraction_proof())
        system.relations.decide(relation_decision_request(
            proposal, action=EditorialRelationDecisionAction.ACCEPT,
            decision_id=RELATION_SECOND_ACCEPT_DECISION_ID,
            assertion_id=RELATION_SECOND_ASSERTION_ID, key='active-suffix-accept',
        ), proof=extraction_proof())


@pytest.fixture
def initial(tmp_path):
    state = seed_increment4_graphiti_path(tmp_path)
    admit_increment4_graphiti_path(state)
    adapter = MemoryNeo4jAdapter()
    first = build(state, adapter, request())
    return state, adapter, first


def commands(state):
    with sqlite3.connect(state.extraction.database) as conn:
        return conn.execute('SELECT count(*) FROM authority_commands').fetchone()[0]


def test_suffix_only_and_exact_historical_replay_after_later_validation(initial):
    state, adapter, first = initial
    add_suffix(state)
    before = commands(state)
    applies, cleanups = adapter.apply_count, adapter.cleanup_count
    second = build(state, adapter, request(G2, 'suffix'))
    assert second.generation.generation_id == G1
    assert second.validation.validation_version == first.validation.validation_version + 1
    assert second.validation.source_snapshot_digest == second.source_snapshot_digest
    assert second.validation.source_watermark_ledger_seq == second.source_watermark_ledger_seq
    assert second.promotion.promotion_digest == first.promotion.promotion_digest
    assert adapter.apply_count - applies == 1
    assert adapter.cleanup_count == cleanups
    assert commands(state) - before == 3  # One optional-prefix delivery, APPLIED and validation.
    third = build(state, adapter, request(G3, 'no-new-graph'))
    assert third.generation.generation_id == G1
    assert adapter.apply_count - applies == 1
    before_replay = commands(state)
    replay = build(state, adapter, request(G2, 'suffix'))
    assert replay.validation == second.validation
    assert replay.source_snapshot_digest == second.source_snapshot_digest
    assert replay.source_watermark_ledger_seq == second.source_watermark_ledger_seq
    assert replay.projection_state_digest == second.projection_state_digest
    assert commands(state) == before_replay
    initial_replay = build(state, adapter, request())
    assert initial_replay.projected_batch_count == first.projected_batch_count
    assert initial_replay.ignored_optional_count == first.ignored_optional_count
    with pytest.raises(ProjectionStateError, match='request'):
        build(state, adapter, replace(request(G2, 'suffix'), reason_code='ALTERED'))


def test_unexplained_prefix_drift_rejected_without_write_or_rebuild(initial):
    state, adapter, first = initial
    add_suffix(state)
    sequence = min(seq for gen, seq in adapter.deliveries if gen == str(G1))
    adapter.corrupt_delivery_digest(G1, sequence)
    before = commands(state), adapter.apply_count, adapter.cleanup_count
    with pytest.raises(Neo4jIdentityConflict):
        build(state, adapter, request(G2, 'corrupt-prefix'))
    assert (commands(state), adapter.apply_count, adapter.cleanup_count) == before


def test_graph_commit_before_sqlite_delivery_recovers_one_exact_pending_suffix(initial, monkeypatch):
    state, adapter, first = initial
    add_suffix(state)
    from newsroom.authority._projection_system import _ProjectionBoundary
    original = _ProjectionBoundary._commit_delivery

    def interrupted(self, grant, delivery):
        if delivery.outcome is ProjectionDeliveryOutcome.APPLIED:
            raise RuntimeError('injected SQLite delivery commit failure')
        return original(self, grant, delivery)

    monkeypatch.setattr(_ProjectionBoundary, '_commit_delivery', interrupted)
    with pytest.raises(Neo4jAuthorityCommitPending):
        build(state, adapter, request(G2, 'pending'))
    monkeypatch.setattr(_ProjectionBoundary, '_commit_delivery', original)
    result = build(state, adapter, request(G2, 'pending'))
    assert result.generation.generation_id == G1
    assert result.projected_batch_count == first.projected_batch_count + 1
    assert result.validation.validation_version == first.validation.validation_version + 1
    assert len(adapter.deliveries) == result.projected_batch_count


def test_active_delivery_is_not_served_until_validation_recovers(initial, monkeypatch):
    from newsroom.authority._projection_system import _ProjectionBoundary

    state, adapter, first = initial
    add_suffix(state)
    original = _ProjectionBoundary.validate_generation

    def interrupted(*args, **kwargs):
        raise RuntimeError('injected before validation')

    monkeypatch.setattr(_ProjectionBoundary, 'validate_generation', interrupted)
    with pytest.raises(RuntimeError, match='before validation'):
        build(state, adapter, request(G2, 'validate-pending'))
    with open_graphiti_path_increment4_neo4j_system(state.relation, adapter) as system:
        ids = tuple(sorted({n.canonical_id for b in adapter.deliveries.values() for n in b.nodes}))
        with pytest.raises(ProjectionStateError, match='awaiting complete validation'):
            system.increment4.read_active(Increment4Neo4jActiveReadRequest(
                canonical_ids=ids, query_valid_time=SOURCE_NOW, limit=100,
            ), proof=extraction_proof())
    monkeypatch.setattr(_ProjectionBoundary, 'validate_generation', original)
    applies = adapter.apply_count
    recovered = build(state, adapter, request(G2, 'validate-pending'))
    assert recovered.generation.generation_id == G1
    assert adapter.apply_count == applies
    with open_graphiti_path_increment4_neo4j_system(state.relation, adapter) as system:
        system.increment4.read_active(Increment4Neo4jActiveReadRequest(
            canonical_ids=ids, query_valid_time=SOURCE_NOW, limit=100,
        ), proof=extraction_proof())


def test_historical_revocation_uses_replacement_not_suffix(tmp_path):
    state = seed_increment4_graphiti_path(tmp_path)
    admitted = admit_increment4_graphiti_path(state)
    adapter = MemoryNeo4jAdapter()
    build(state, adapter, request())
    with open_graphiti_path_relation_system(state.relation) as system:
        system.relations.decide(relation_decision_request(
            admitted.proposal, action=EditorialRelationDecisionAction.REVOKE,
            decision_id=RELATION_SECOND_DECISION_ID,
            expected_previous_version=admitted.decision.decision_version,
            previous_decision_id=admitted.decision.decision_id,
            target_assertion_id=RELATION_ASSERTION_ID, key='extension-revocation',
        ), proof=extraction_proof())
    second = build(state, adapter, request(G2, 'changed-prefix'))
    assert second.generation.generation_id == G2
    assert second.prior_generation.generation_id == G1
    assert all(gen == str(G2) for gen, _ in adapter.deliveries)
    # Native replacement deliberately expires this retired generation's diagnostics.
    # Exact replay remains denied, with no new write or adapter mutation.
    before = commands(state), adapter.apply_count, adapter.cleanup_count
    with pytest.raises(DiagnosticHistoryExpired, match='checkpoint history expired'):
        build(state, adapter, request())
    assert (commands(state), adapter.apply_count, adapter.cleanup_count) == before


def test_source_watermark_race_stays_unvalidated(initial, monkeypatch):
    from newsroom.authority import AggregateId
    from newsroom.authority._projection_system import _ProjectionBoundary
    from .authority_helpers import command

    state, adapter, first = initial
    add_suffix(state)
    original = _ProjectionBoundary.validate_generation
    with open_graphiti_path_increment4_neo4j_system(state.relation, adapter) as system:
        def raced(self, *args, **kwargs):
            system.commands.execute(command(
                key='active-extension-source-race', aggregate_id=AggregateId.parse(
                    '00000000-0000-4000-8000-000000008109',
                ),
            ), proof=extraction_proof())
            return original(self, *args, **kwargs)
        monkeypatch.setattr(_ProjectionBoundary, 'validate_generation', raced)
        with pytest.raises(ProjectionStateError, match='source watermark changed'):
            system.increment4.build_current_and_promote(
                request(G2, 'source-race'), proof=extraction_proof(),
            )
        status = system.increment4.generation_status(G1, proof=extraction_proof())
        assert status.generation.validated_through_ledger_seq == first.checkpoint_ledger_seq


@pytest.mark.parametrize('field', ['open_gap_count', 'dead_letter_count'])
def test_active_gap_or_dead_letter_is_not_hidden_by_rebuild(initial, monkeypatch, field):
    from newsroom.authority._increment4_projection_store import _Increment4ProjectionAuthorityStore

    state, adapter, first = initial
    add_suffix(state)
    original = _Increment4ProjectionAuthorityStore.projection_active_generation_metadata
    def failed_metadata(*args, **kwargs):
        return replace(original(*args, **kwargs), **{field: 1})
    monkeypatch.setattr(_Increment4ProjectionAuthorityStore, 'projection_active_generation_metadata', failed_metadata)
    before = commands(state), adapter.apply_count, adapter.cleanup_count
    with pytest.raises(ProjectionStateError, match='gaps or dead letters'):
        build(state, adapter, request(G2, 'active-failure'))
    assert (commands(state), adapter.apply_count, adapter.cleanup_count) == before


def test_initial_validation_commit_before_promotion_resumes_exact_request(tmp_path, monkeypatch):
    from newsroom.authority._increment4_neo4j_boundary import _Increment4Neo4jBoundary

    state = seed_increment4_graphiti_path(tmp_path)
    admit_increment4_graphiti_path(state)
    adapter = MemoryNeo4jAdapter()
    original = _Increment4Neo4jBoundary._promote
    def interrupted(*args, **kwargs):
        raise RuntimeError('injected before promotion')
    monkeypatch.setattr(_Increment4Neo4jBoundary, '_promote', interrupted)
    with pytest.raises(RuntimeError, match='before promotion'):
        build(state, adapter, request())
    monkeypatch.setattr(_Increment4Neo4jBoundary, '_promote', original)
    result = build(state, adapter, request())
    assert result.generation.state.value == 'ACTIVE'
    assert result.validation.validation_version == 1
    assert result.validation.source_snapshot_digest == result.source_snapshot_digest


def test_optional_prefix_over_retained_gap_cap_is_recorded_before_graph_suffix(tmp_path, monkeypatch):
    from newsroom.authority import AggregateId
    from newsroom.authority._projection_system import _ProjectionBoundary
    from newsroom.increment4 import contracts
    from .authority_helpers import command

    original_family = contracts.increment4_admitted_family_v1
    monkeypatch.setattr(contracts, 'increment4_admitted_family_v1',
                        lambda *args: replace(original_family(*args), max_gap_span=2))
    state = seed_increment4_graphiti_path(tmp_path)
    admit_increment4_graphiti_path(state)
    adapter = MemoryNeo4jAdapter()
    first = build(state, adapter, request())
    with open_graphiti_path_increment4_neo4j_system(state.relation, adapter) as system:
        for ordinal in range(3):
            system.commands.execute(command(
                key=f'optional-prefix-{ordinal}',
                aggregate_id=AggregateId.parse('00000000-0000-4000-8000-000000008109'),
                expected_version=ordinal,
            ), proof=extraction_proof())
    add_suffix(state)
    observed = []
    original_commit = _ProjectionBoundary._commit_delivery
    original_apply = adapter.apply
    def committed(self, grant, delivery):
        observed.append(('delivery', delivery.ledger_seq, delivery.outcome, delivery.expected_authority_version))
        result = original_commit(self, grant, delivery)
        if delivery.outcome is ProjectionDeliveryOutcome.IGNORED_OPTIONAL:
            metadata = self._store.projection_generation_metadata(delivery.generation_id)
            observed[-1] += (metadata.contiguous_ledger_seq,)
        return result
    def applied(batch):
        observed.append(('graph', batch.ledger_seq))
        return original_apply(batch)
    monkeypatch.setattr(_ProjectionBoundary, '_commit_delivery', committed)
    monkeypatch.setattr(adapter, 'apply', applied)
    result = build(state, adapter, request(G2, 'long-optional-prefix'))
    assert result.generation.generation_id == G1
    assert result.projected_batch_count == first.projected_batch_count + 1
    assert observed[0][0] == 'delivery' and observed[0][2] is ProjectionDeliveryOutcome.IGNORED_OPTIONAL
    assert observed[1][0] == 'graph'
    assert observed[2][0] == 'delivery' and observed[2][2] is ProjectionDeliveryOutcome.APPLIED
    assert observed[1][1] - observed[0][1] > 2
    assert observed[0][4] >= observed[1][1]  # Optional SKIP does not replace APPLIED authority.
    assert observed[2][3] == observed[0][3] + 1
    assert len([row for row in observed if row[0] == 'graph']) == 1
    before = commands(state), adapter.apply_count
    assert build(state, adapter, request(G2, 'long-optional-prefix')).validation == result.validation
    assert (commands(state), adapter.apply_count) == before


def test_optional_prefix_authority_refusal_precedes_graph_apply(initial, monkeypatch):
    from newsroom.authority._projection_system import _ProjectionBoundary

    state, adapter, first = initial
    add_suffix(state)
    original = _ProjectionBoundary._commit_delivery
    def refused(self, grant, delivery):
        if delivery.outcome is ProjectionDeliveryOutcome.IGNORED_OPTIONAL:
            raise ProjectionStateError('fixture optional-prefix authority refused')
        return original(self, grant, delivery)
    monkeypatch.setattr(_ProjectionBoundary, '_commit_delivery', refused)
    before = commands(state), adapter.apply_count, adapter.cleanup_count
    with pytest.raises(ProjectionStateError, match='optional-prefix authority refused'):
        build(state, adapter, request(G2, 'prefix-refused'))
    assert (commands(state), adapter.apply_count, adapter.cleanup_count) == before
