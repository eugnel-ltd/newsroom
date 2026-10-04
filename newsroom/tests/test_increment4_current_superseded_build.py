"""Current management supersedes only unvalidated non-serving build intent."""
from dataclasses import replace
import pytest
from newsroom.authority import AggregateId
from newsroom.authority.persistence import ExpectedVersionConflict
from newsroom.projection import ProjectionGenerationState,ProjectionDeliveryOutcome
from newsroom.projection.neo4j import Neo4jAuthorityCommitPending
from newsroom.tests.test_increment4e_neo4j_controller import GENERATION_1,GENERATION_2,_current_request
from newsroom.tests.increment4e_helpers import admitted_increment4_fixture,open_increment4_neo4j_system
from newsroom.tests.projection_b2_helpers import MemoryNeo4jAdapter
from newsroom.tests.extraction_4a_helpers import extraction_proof
from newsroom.tests.authority_helpers import command


def test_current_source_drift_after_graph_apply_supersedes_without_rebinding(tmp_path,monkeypatch):
    state,_=admitted_increment4_fixture(tmp_path);adapter=MemoryNeo4jAdapter()
    request=replace(_current_request(GENERATION_1,key='current-pending-source'),allow_active_extension=True)
    with open_increment4_neo4j_system(state,adapter) as system:
        boundary=system.increment4._Increment4Neo4jController__build_current.__self__
        commit=boundary._projection_boundary._commit_delivery
        def missing(grant,delivery):
            if delivery.outcome is ProjectionDeliveryOutcome.APPLIED:raise RuntimeError('apply committed but authority absent')
            return commit(grant,delivery)
        monkeypatch.setattr(boundary._projection_boundary,'_commit_delivery',missing)
        with pytest.raises(Neo4jAuthorityCommitPending):system.increment4.build_current_and_promote(request,proof=extraction_proof())
        old=system.increment4.generation_status(GENERATION_1,proof=extraction_proof()).generation
        assert old.state is ProjectionGenerationState.BUILDING and adapter.deliveries
        monkeypatch.setattr(boundary._projection_boundary,'_commit_delivery',commit)
        system.commands.execute(command(key='changed-current-source',aggregate_id=AggregateId.new()),proof=extraction_proof())
        result=system.increment4.build_current_and_promote(request,proof=extraction_proof())
        assert result.generation.state is ProjectionGenerationState.ACTIVE
        assert result.generation.generation_id!=GENERATION_1
        assert system.increment4.generation_status(GENERATION_1,proof=extraction_proof()).generation.state is ProjectionGenerationState.FAILED
        assert result.validation.source_request_digest==boundary._source_request_digest(request)
    with open_increment4_neo4j_system(state,adapter) as system:
        replay=system.increment4.build_current_and_promote(request,proof=extraction_proof())
        assert replay.generation.generation_id==result.generation.generation_id
        assert replay.promotion==result.promotion


@pytest.mark.parametrize('stage',['fail','create','validation'])
def test_supersession_interrupt_reopens_the_exact_management_chain(tmp_path,monkeypatch,stage):
    state,_=admitted_increment4_fixture(tmp_path);adapter=MemoryNeo4jAdapter()
    request=replace(_current_request(GENERATION_1,key='current-interrupt-'+stage),allow_active_extension=True)
    with open_increment4_neo4j_system(state,adapter) as system:
        boundary=system.increment4._Increment4Neo4jController__build_current.__self__
        commit=boundary._projection_boundary._commit_delivery
        def pending(grant,delivery):
            if delivery.outcome is ProjectionDeliveryOutcome.APPLIED:raise RuntimeError('pending authority')
            return commit(grant,delivery)
        monkeypatch.setattr(boundary._projection_boundary,'_commit_delivery',pending)
        with pytest.raises(Neo4jAuthorityCommitPending):system.increment4.build_current_and_promote(request,proof=extraction_proof())
        monkeypatch.setattr(boundary._projection_boundary,'_commit_delivery',commit)
        system.commands.execute(command(key='source-drift-'+stage,aggregate_id=AggregateId.new()),proof=extraction_proof())
        if stage=='fail':
            original=boundary._projection_boundary.transition_generation
            def interrupted(request,proof):
                value=original(request,proof)
                if request.target_state is ProjectionGenerationState.FAILED:raise KeyboardInterrupt('after fail')
                return value
            monkeypatch.setattr(boundary._projection_boundary,'transition_generation',interrupted)
        elif stage=='create':
            original=boundary._create_generation
            def interrupted(**kwargs):
                value=original(**kwargs)
                if kwargs['request'].generation_id!=GENERATION_1:raise KeyboardInterrupt('after replacement create')
                return value
            monkeypatch.setattr(boundary,'_create_generation',interrupted)
        else:
            original=boundary._projection_boundary.validate_generation
            def interrupted(*args,**kwargs):
                original(*args,**kwargs);raise KeyboardInterrupt('after validation')
            monkeypatch.setattr(boundary._projection_boundary,'validate_generation',interrupted)
        with pytest.raises(KeyboardInterrupt):system.increment4.build_current_and_promote(request,proof=extraction_proof())
    with open_increment4_neo4j_system(state,adapter) as system:
        result=system.increment4.build_current_and_promote(request,proof=extraction_proof())
        assert result.generation.state is ProjectionGenerationState.ACTIVE and result.generation.generation_id!=GENERATION_1
        assert system.increment4.generation_status(GENERATION_1,proof=extraction_proof()).generation.state is ProjectionGenerationState.FAILED
        assert system.increment4.build_current_and_promote(request,proof=extraction_proof()).promotion==result.promotion


@pytest.mark.parametrize('changed_source',[False,True])
def test_unchanged_intent_resumes_but_legacy_source_drift_stays_strict(tmp_path,monkeypatch,changed_source):
    state,_=admitted_increment4_fixture(tmp_path);adapter=MemoryNeo4jAdapter()
    request=replace(_current_request(GENERATION_1,key='legacy-or-current'),allow_active_extension=not changed_source)
    with open_increment4_neo4j_system(state,adapter) as system:
        boundary=system.increment4._Increment4Neo4jController__build_current.__self__
        commit=boundary._projection_boundary._commit_delivery
        def pending(grant,delivery):
            if delivery.outcome is ProjectionDeliveryOutcome.APPLIED:raise RuntimeError('pending')
            return commit(grant,delivery)
        monkeypatch.setattr(boundary._projection_boundary,'_commit_delivery',pending)
        with pytest.raises(Neo4jAuthorityCommitPending):system.increment4.build_current_and_promote(request,proof=extraction_proof())
        monkeypatch.setattr(boundary._projection_boundary,'_commit_delivery',commit)
        if changed_source:
            system.commands.execute(command(key='legacy-drift',aggregate_id=AggregateId.new()),proof=extraction_proof())
            with pytest.raises(ExpectedVersionConflict):system.increment4.build_current_and_promote(request,proof=extraction_proof())
            assert system.increment4.generation_status(GENERATION_1,proof=extraction_proof()).generation.state is ProjectionGenerationState.BUILDING
        else:
            result=system.increment4.build_current_and_promote(request,proof=extraction_proof())
            assert result.generation.generation_id==GENERATION_1


def test_failed_replacement_keeps_prior_active_and_unknown_old_namespace(tmp_path,monkeypatch):
    state,_=admitted_increment4_fixture(tmp_path);adapter=MemoryNeo4jAdapter()
    request=replace(_current_request(GENERATION_1,key='prior-active-pending'),allow_active_extension=True)
    with open_increment4_neo4j_system(state,adapter) as system:
        first=system.increment4.build_current_and_promote(_current_request(GENERATION_2,key='serving-prior'),proof=extraction_proof())
        boundary=system.increment4._Increment4Neo4jController__build_current.__self__
        inputs=boundary._store._increment4_current_build_inputs(generation_id=GENERATION_1,family=boundary._store.projection_family_definition('graph.increment4.admitted'))
        boundary._create_generation(request=request,snapshot_digest=inputs.snapshot_digest,proof=extraction_proof())
        commit=boundary._projection_boundary._commit_delivery
        def pending(grant,delivery):
            if delivery.outcome is ProjectionDeliveryOutcome.APPLIED:raise RuntimeError('pending')
            return commit(grant,delivery)
        monkeypatch.setattr(boundary._projection_boundary,'_commit_delivery',pending)
        with pytest.raises(Neo4jAuthorityCommitPending):system.increment4.build_current_and_promote(request,proof=extraction_proof())
        monkeypatch.setattr(boundary._projection_boundary,'_commit_delivery',commit)
        prior={key:value for key,value in adapter.deliveries.items() if key[0]==str(GENERATION_2)}
        unknown={key:value for key,value in adapter.deliveries.items() if key[0]==str(GENERATION_1)}
        system.commands.execute(command(key='source-with-prior',aggregate_id=AggregateId.new()),proof=extraction_proof())
        def reject(**_):raise RuntimeError('new validation failure')
        monkeypatch.setattr(boundary,'_validate',reject)
        with pytest.raises(RuntimeError,match='new validation failure'):system.increment4.build_current_and_promote(request,proof=extraction_proof())
        assert system.increment4.generation_status(GENERATION_2,proof=extraction_proof()).generation.state is ProjectionGenerationState.ACTIVE
        assert {k:v for k,v in adapter.deliveries.items() if k[0]==str(GENERATION_2)}==prior
        assert {k:v for k,v in adapter.deliveries.items() if k[0]==str(GENERATION_1)}==unknown
        assert first.generation.generation_id==GENERATION_2


def test_current_management_permission_denial_precedes_supersession_and_graph(tmp_path,monkeypatch):
    from newsroom.tests.increment4e_helpers import INCREMENT4_PROJECTION_SCOPES
    state,_=admitted_increment4_fixture(tmp_path);adapter=MemoryNeo4jAdapter()
    request=replace(_current_request(GENERATION_1,key='denied-current'),allow_active_extension=True)
    with open_increment4_neo4j_system(state,adapter) as system:
        boundary=system.increment4._Increment4Neo4jController__build_current.__self__
        commit=boundary._projection_boundary._commit_delivery
        def pending(grant,delivery):
            if delivery.outcome is ProjectionDeliveryOutcome.APPLIED:raise RuntimeError('pending')
            return commit(grant,delivery)
        monkeypatch.setattr(boundary._projection_boundary,'_commit_delivery',pending)
        with pytest.raises(Neo4jAuthorityCommitPending):system.increment4.build_current_and_promote(request,proof=extraction_proof())
        system.commands.execute(command(key='denied-source-drift',aggregate_id=AggregateId.new()),proof=extraction_proof())
    counts=adapter.apply_count,adapter.cleanup_count
    with open_increment4_neo4j_system(state,adapter,scopes=INCREMENT4_PROJECTION_SCOPES-{'authority.projection.manage'}) as system:
        with pytest.raises(PermissionError):system.increment4.build_current_and_promote(request,proof=extraction_proof())
        assert system.increment4.generation_status(GENERATION_1,proof=extraction_proof()).generation.state is ProjectionGenerationState.BUILDING
    assert (adapter.apply_count,adapter.cleanup_count)==counts
