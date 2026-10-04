"""Selected copies preserve promotion's exact endpoint result versions."""
from dataclasses import replace
import sqlite3
import json
from newsroom.projection import ProjectionGenerationPromotionRequest
from newsroom.authority.native_current_rebuild import copy_selected_native_store
from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store
from newsroom.authority._increment4_projection_store import _Increment4ProjectionAuthorityStore
from newsroom.tests.test_increment4_active_extension import G1,G2,G3,request,add_suffix
from newsroom.tests.increment4e_governed_path_helpers import seed_increment4_graphiti_path,admit_increment4_graphiti_path,open_graphiti_path_increment4_neo4j_system,open_graphiti_path_relation_system
from newsroom.tests.editorial_relation_4c_helpers import relation_decision_request,RELATION_SECOND_DECISION_ID,RELATION_ASSERTION_ID
from newsroom.relations import EditorialRelationDecisionAction
from newsroom.tests.projection_b2_helpers import MemoryNeo4jAdapter
from newsroom.tests.extraction_4a_helpers import extraction_proof


def test_normal_promotion_extension_copy_reads_exact_target_and_prior_results(tmp_path,monkeypatch):
    state=seed_increment4_graphiti_path(tmp_path);admitted=admit_increment4_graphiti_path(state)
    adapter=MemoryNeo4jAdapter();target=tmp_path/'promotion-selected.sqlite3'
    with open_graphiti_path_increment4_neo4j_system(state.relation,adapter) as system:
        first=system.increment4.build_current_and_promote(replace(request(G1,'prior'),allow_active_extension=False),proof=extraction_proof())
    with open_graphiti_path_relation_system(state.relation) as system:
        system.relations.decide(relation_decision_request(admitted.proposal,action=EditorialRelationDecisionAction.REVOKE,
            decision_id=RELATION_SECOND_DECISION_ID,expected_previous_version=admitted.decision.decision_version,
            previous_decision_id=admitted.decision.decision_id,target_assertion_id=RELATION_ASSERTION_ID,key='promotion-copy-revoke'),proof=extraction_proof())
    monkeypatch.setattr(_Increment4ProjectionAuthorityStore,'_current_state_only',True)
    with open_graphiti_path_increment4_neo4j_system(state.relation,adapter) as system:
        promoted=system.increment4.build_current_and_promote(request(G2,'replace'),proof=extraction_proof())
        assert promoted.promotion.prior_generation.generation_id==G1
    add_suffix(state)
    with open_graphiti_path_increment4_neo4j_system(state.relation,adapter) as system:
        extended=system.increment4.build_current_and_promote(request(G3,'extension'),proof=extraction_proof())
        assert extended.generation.generation_id==G2
        root=system.increment4._Increment4Neo4jController__reconcile_active.__self__._store
        root._current_state_only=True
        command=root._connection.execute('SELECT c.*,p.payload_bytes FROM ledger_events e JOIN authority_commands c USING(command_id) JOIN authority_payloads p ON p.payload_id=c.payload_id WHERE e.event_id=?',(str(promoted.promotion.target_authority_event_id),)).fetchone()
        payload=json.loads(command['payload_bytes'])
        replay_request=ProjectionGenerationPromotionRequest(G2,int(command['expected_aggregate_version']),payload['checkpoint_ledger_seq'],payload['validation_digest'],payload['reason_code'],command['idempotency_key'],G1,promoted.promotion.prior_generation.authority_aggregate_version-1)
        original_versions={str(event):tuple(root._connection.execute('SELECT * FROM projection_generation_versions WHERE authority_event_id=?',(str(event),)).fetchone()) for event in (promoted.promotion.target_authority_event_id,promoted.promotion.prior_authority_event_id)}
        with sqlite3.connect(target,isolation_level=None) as c:
            initialise_empty_checkpoint_store(c)
            copy_selected_native_store(root,c,roots={'projection_generation_promotions':((promoted.promotion.promotion_digest,),)},dev_rebuild=True)
            assert c.execute('PRAGMA foreign_key_check').fetchall()==[]
            for event,version in original_versions.items():
                row=c.execute('SELECT * FROM projection_generation_versions WHERE authority_event_id=?',(event,)).fetchone()
                assert row is not None and tuple(row)==version
            assert c.execute('SELECT count(*) FROM projection_generation_versions WHERE generation_id=?',(str(G2),)).fetchone()[0]<root._connection.execute('SELECT count(*) FROM projection_generation_versions WHERE generation_id=?',(str(G2),)).fetchone()[0]
    target.chmod(0o600)
    # The existing actual store/reader, not hand-decoded promotion fields.
    # Fixture uses the same factory arguments and immutable CAS, but a NEW path.
    selected_relation=replace(state.relation,entity=replace(state.relation.entity,
        extraction=replace(state.extraction,database=target)))
    with open_graphiti_path_increment4_neo4j_system(selected_relation,adapter) as system:
        root=system.increment4._Increment4Neo4jController__reconcile_active.__self__._store
        assert root._promotion_for_authority_event(root._connection,str(promoted.promotion.target_authority_event_id))==promoted.promotion
        assert root.projection_promotions('graph.increment4.admitted',10)[0]==promoted.promotion
        before=root._connection.execute('SELECT count(*) FROM authority_commands').fetchone()[0]
        controller=system.increment4._Increment4Neo4jController__reconcile_active.__self__
        assert controller._projection_boundary.promote_generation(replay_request,extraction_proof())==promoted.promotion
        assert root._connection.execute('SELECT count(*) FROM authority_commands').fetchone()[0]==before
