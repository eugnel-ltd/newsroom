from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from newsroom.authority import (
    AuthorityPersistenceError, ObjectLimits, ObjectAdmissionRequest, StaticAuthenticator,
    StaticAuthorizer, StaticPrincipal, HydrationRequest, ObjectIntegrityError,
)
from newsroom.authority._hermes_native_system import open_hermes_native_authority_system
from newsroom.increment4 import increment4_admitted_contract_registry
from newsroom.increment6.collision import (
    CurrentCollisionEffectEnforcer, TrustedCurrentCollisionAuthorityBoundary,
)
from newsroom.tests.authority_a2b_helpers import _policy_registries
from newsroom.tests.authority_event_helpers import fixture_read_policy, payload_schemas, registry_v1
from newsroom.tests.authority_helpers import FIXED_NOW, command, proof
from newsroom.tests.check_3c_authority_helpers import check_read_policy, source_read_policy
from newsroom.tests.discovery_3d_authority_helpers import (
    discovery_read_policy, exact_admission_request, scopes, seed_check_lineage,
)
from newsroom.tests.editorial_relation_4c_helpers import relation_read_policy
from newsroom.tests.entity_4b_helpers import entity_read_policy
from newsroom.tests.extraction_4a_helpers import extraction_read_policy
from newsroom.tests.graphiti_adapter_4d_authority_helpers import graphiti_read_policy
from newsroom.tests.increment4e_helpers import INCREMENT4_PROJECTION_SCOPES, increment4_projection_read_policy
from newsroom.tests.increment5b2_helpers import config
from newsroom.tests.projection_b2_helpers import MemoryNeo4jAdapter
from newsroom.tests.test_hermes_native_authority_system import _retrieval_authority
from newsroom.authority._object_cas import _GovernedCAS
from newsroom.increment6.execution import LeaseLifecycle, WorkerAttempt
from newsroom.increment6.proposals import FixtureWorkerKind
from newsroom.increment6.work_items import DecisionLeadBinding, TriageWorkItem
from newsroom.tests.test_increment6a2_work_items import _version
from newsroom.tests.test_increment6b1_execution import _batch, _digest


@pytest.fixture
def native_open(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "newsroom.authority._graphiti_increment4_system._open_structural_graph_adapter",
        lambda _: MemoryNeo4jAdapter(),
    )
    rights, hydration, admissions = _policy_registries()
    retrieval, binding = _retrieval_authority(tmp_path / "retrieval")
    kwargs = dict(
        path=tmp_path / "authority.sqlite3", object_root=tmp_path / "objects",
        workspace_root=tmp_path.resolve(), registry=registry_v1(),
        payload_schemas=payload_schemas(), admission_registry=admissions,
        rights_policies=rights, hydration_policies=hydration,
        contracts=increment4_admitted_contract_registry(),
        authenticator=StaticAuthenticator(
            credentials={"token-1": StaticPrincipal("principal.alpha")},
            authority_domain="newsroom.authority",
        ),
        authorizer=StaticAuthorizer(
            policy_version="native-current-state-test-v1",
            grants_by_principal={"principal.alpha": scopes() | INCREMENT4_PROJECTION_SCOPES | frozenset({
                "authority.fixture.events.read", "authority.observed.write",
                "authority.objects.admit", "authority.objects.read", "authority.objects.manage",
                "authority.objects.lifecycle.write",
            })},
        ),
        event_read_policy=fixture_read_policy(),
        source_read_policy=source_read_policy(), check_read_policy=check_read_policy(),
        discovery_read_policy=discovery_read_policy(),
        extraction_read_policy=extraction_read_policy(), entity_read_policy=entity_read_policy(),
        relation_read_policy=relation_read_policy(), graphiti_read_policy=graphiti_read_policy(),
        projection_read_policy=increment4_projection_read_policy(),
        object_limits=ObjectLimits(
            global_max_bytes=1024 * 1024, class_max_bytes={"source_capture": 1024 * 1024},
            max_read_bytes=1024 * 1024, min_free_bytes=0, io_chunk_bytes=64,
            max_staging_bytes=1024 * 1024, max_range_bytes=1024 * 1024,
        ),
        neo4j_config=config(), retrieval_authority=retrieval,
        collision_enforcer=CurrentCollisionEffectEnforcer(
            current_authority_provider=lambda _: None,
            trusted_boundary=TrustedCurrentCollisionAuthorityBoundary(
                "fixture-scope", "fixture-profile", "sha256:" + "a" * 64,
                "sha256:" + "b" * 64, "fixture-port",
            ),
        ),
        clock=lambda: FIXED_NOW,
    )
    def opener(**overrides):
        return open_hermes_native_authority_system(**(kwargs | overrides))
    opener.retrieval_binding = binding
    return opener


def _change(path: Path, table: str, sql: str, parameters=()) -> None:
    with sqlite3.connect(path) as connection:
        trigger = f"immutable_{table}_update"
        definition = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE name=?", (trigger,),
        ).fetchone()[0]
        connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute(sql, parameters)
        connection.execute(definition)


def test_native_boot_does_not_require_unused_history_integrity(
    tmp_path: Path, native_open,
) -> None:
    request = command(key="unused-history")
    with native_open() as system:
        retained = system.commands.execute(request, proof=proof())
    _change(
        tmp_path / "authority.sqlite3", "authority_payloads",
        "UPDATE authority_payloads SET payload_bytes=? WHERE payload_id="
        "(SELECT payload_id FROM authority_commands WHERE command_id=?)",
        (b'{"headline":"mutated","count":2}', retained.command_id),
    )
    # Unused history is no longer a native boot precondition. Its exact read
    # and any replay still fail closed before an operational consumer uses it.
    with native_open() as reopened:
        with pytest.raises(AuthorityPersistenceError):
            reopened.events.after(0, limit=1, proof=proof())
        with pytest.raises(AuthorityPersistenceError):
            reopened.commands.execute(request, proof=proof())
        with pytest.raises(AuthorityPersistenceError):
            reopened.events.command_result(retained.command_id, proof=proof())
        with pytest.raises(AuthorityPersistenceError):
            reopened.validate_retained_history()
        # Failed maintenance does not leak full-history mode into ordinary use.
        fresh = reopened.commands.execute(command(key="fresh"), proof=proof())
        assert reopened.events.after(retained.ledger_seq, limit=1, proof=proof())[0].event_id == fresh.event_id


def test_native_boot_does_not_issue_historical_sweeps(
    tmp_path: Path, native_open, monkeypatch,
) -> None:
    with native_open():
        pass
    statements = []
    connect = sqlite3.connect

    def traced_connect(database, *args, **kwargs):
        connection = connect(database, *args, **kwargs)
        if Path(database) == tmp_path / "authority.sqlite3":
            connection.set_trace_callback(lambda sql: statements.append(" ".join(sql.upper().split())))
        return connection

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    with native_open():
        pass
    forbidden = (
        "PRAGMA QUICK_CHECK", "PRAGMA FOREIGN_KEY_CHECK",
        "FROM TRIAGE_WORK_ITEMS", "FROM TRIAGE_WORK_ITEM_VERSIONS ORDER BY",
        "FROM TRIAGE_PROPOSAL_VALIDATION_FINDINGS",
        "FROM TRIAGE_PROPOSAL_DISPOSITIONS",
        "FROM EVENT_HYPOTHESES_V2", "FROM EVENT_HYPOTHESIS_VERSIONS_V2 ORDER BY",
        "FROM PROJECTION_DELIVERY_ATTEMPTS ORDER BY",
    )
    sweeps = [sql for sql in statements if any(part in sql for part in forbidden) and " WHERE " not in sql]
    assert sweeps == []


def test_native_boot_does_not_enumerate_cas_orphans(native_open, monkeypatch) -> None:
    with native_open():
        pass

    def maintenance_only(*args, **kwargs):
        raise AssertionError("CAS orphan enumeration entered native boot")

    monkeypatch.setattr(_GovernedCAS, "cleanup_unreferenced_installed", maintenance_only)
    with native_open():
        pass


def test_native_source_read_proves_exact_security_closure(tmp_path, native_open) -> None:
    with native_open() as system:
        seed_check_lineage(system)
    with sqlite3.connect(tmp_path / "authority.sqlite3") as connection:
        event_id = connection.execute(
            "SELECT authority_event_id FROM source_definitions LIMIT 1"
        ).fetchone()[0]
        definition_id = connection.execute(
            "SELECT definition_id FROM source_definitions LIMIT 1"
        ).fetchone()[0]
    _change(
        tmp_path / "authority.sqlite3", "authentication_contexts",
        "UPDATE authentication_contexts SET principal_id=? WHERE authentication_context_id="
        "(SELECT authentication_context_id FROM ledger_events WHERE event_id=?)",
        ("changed-principal", event_id),
    )
    from newsroom.sources import SourceDefinitionId
    with native_open() as reopened:
        with pytest.raises(AuthorityPersistenceError):
            reopened.sources.definition(SourceDefinitionId.parse(definition_id), proof=proof())


def test_native_boot_sql_work_is_bounded_by_state_not_unused_events(
    tmp_path, native_open, monkeypatch, record_property,
) -> None:
    with native_open() as system:
        system.commands.execute(command(key="history-0"), proof=proof())
    counts = []
    connect = sqlite3.connect

    def counted_connect(database, *args, **kwargs):
        connection = connect(database, *args, **kwargs)
        if Path(database) == tmp_path / "authority.sqlite3":
            counts.append(0)
            index = len(counts) - 1
            def progress():
                counts[index] += 100
                return 0
            connection.set_progress_handler(progress, 100)
        return connection

    monkeypatch.setattr(sqlite3, "connect", counted_connect)
    with native_open():
        pass
    one = counts[-1]
    with native_open() as system:
        for number in range(1, 256):
            system.commands.execute(command(key=f"history-{number}"), proof=proof())
    with native_open():
        pass
    many = counts[-1]
    record_property("boot_with_one_unused_event_vm_steps", one)
    record_property("boot_with_256_unused_events_vm_steps", many)
    assert many <= one + 2000, (one, many)


def test_native_boot_defers_unselected_active_cas_but_rejects_its_use(tmp_path, native_open) -> None:
    with native_open() as system:
        admission = system.objects.admit(
            ObjectAdmissionRequest("source.capture", "live-cas"), b"current source bytes", proof=proof(),
        )
    path = next(p for p in (tmp_path / "objects" / "objects").rglob("*") if p.is_file())
    path.chmod(0o600)
    path.write_bytes(b"mutated source bytes")
    path.chmod(0o400)
    with native_open() as reopened:
        with pytest.raises(ObjectIntegrityError):
            reopened.objects.hydrate(
                HydrationRequest(admission.admission.admission_id, "project.discovery"), proof=proof(),
            )
        with pytest.raises(AuthorityPersistenceError, match="active authoritative blob"):
            reopened.validate_retained_history()
    with sqlite3.connect(tmp_path / "authority.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM object_access_decisions").fetchone()[0] == 0


def test_native_explicit_history_maintenance_still_accepts_valid_state(native_open) -> None:
    with native_open() as system:
        seed_check_lineage(system)
        system.validate_retained_history()


def test_native_pending_execution_lease_survives_reopen_without_new_effect(native_open) -> None:
    with native_open() as system:
        seed_check_lineage(system)
        admitted = system.discovery.admit_signal_to_lead(exact_admission_request(), proof=proof())
        decision = DecisionLeadBinding.from_authority(admitted.lead, admitted.initial_disposition)
        item = TriageWorkItem.create((decision,))
        version = replace(_version(item), retrieval=native_open.retrieval_binding)
        system.work_items.create_or_replay(item, version)
        batch = _batch(version)
        attempt = WorkerAttempt.create(
            member=batch.members[0], ordinal=1, worker_kind=FixtureWorkerKind.REPLAY,
            worker_version="startup-fixture-v1", input_digest=_digest(991),
        )
        system.executions.register_batch(batch, proof=proof())
        system.executions.register_attempt(batch.batch_id, attempt, proof=proof())
        lease = system.executions.claim(attempt.attempt_id, proof=proof())
    with native_open() as reopened:
        assert reopened.executions.register_attempt(batch.batch_id, attempt, proof=proof()) == attempt
        assert reopened.executions.claim(attempt.attempt_id, proof=proof()) == lease
        assert reopened.executions.release(lease.lease_id, proof=proof()).lifecycle is LeaseLifecycle.RELEASED


def test_native_current_work_item_rejects_source_security_tamper(tmp_path, native_open) -> None:
    with native_open() as system:
        seed_check_lineage(system)
        admitted = system.discovery.admit_signal_to_lead(exact_admission_request(), proof=proof())
        decision = DecisionLeadBinding.from_authority(admitted.lead, admitted.initial_disposition)
        item = TriageWorkItem.create((decision,))
        system.work_items.create_or_replay(
            item, replace(_version(item), retrieval=native_open.retrieval_binding),
        )
    _change(
        tmp_path / "authority.sqlite3", "authentication_contexts",
        "UPDATE authentication_contexts SET principal_id=? WHERE authentication_context_id="
        "(SELECT authentication_context_id FROM ledger_events WHERE event_id=?)",
        ("changed-principal", str(admitted.lead.event_id)),
    )
    from newsroom.increment6.work_items import WorkItemContractError
    with native_open() as reopened:
        with pytest.raises((AuthorityPersistenceError, WorkItemContractError)):
            reopened.work_items.current_version(item.work_item_id)


def test_native_selected_hydration_rechecks_retained_rights_bytes(tmp_path, native_open) -> None:
    with native_open() as system:
        admitted = system.objects.admit(
            ObjectAdmissionRequest("source.capture", "rights-on-use"), b"source", proof=proof(),
        )
    _change(
        tmp_path / "authority.sqlite3", "object_rights_decisions",
        "UPDATE object_rights_decisions SET canonical_bytes=?", (b"{}",),
    )
    with native_open() as reopened:
        with pytest.raises(AuthorityPersistenceError):
            reopened.objects.hydrate(
                HydrationRequest(admitted.admission.admission_id, "project.discovery"), proof=proof(),
            )


def test_native_object_backed_effect_proves_selected_rights_before_commit(tmp_path, native_open) -> None:
    from newsroom.tests.test_authority_a2b_object_commands import _command, _registries
    commands, schemas = _registries()
    kwargs = dict(registry=commands, payload_schemas=schemas)
    with native_open(**kwargs) as system:
        admitted = system.objects.admit(
            ObjectAdmissionRequest("source.capture", "effect-rights"), b"source", proof=proof(),
        )
    _change(
        tmp_path / "authority.sqlite3", "object_rights_decisions",
        "UPDATE object_rights_decisions SET canonical_bytes=?", (b"{}",),
    )
    with native_open(**kwargs) as reopened:
        with pytest.raises(AuthorityPersistenceError):
            reopened.commands.execute(_command(admitted.admission.admission_id, key="effect"), proof=proof())


def test_native_wal_recovery_does_not_turn_unused_corruption_into_authority(tmp_path, native_open) -> None:
    with native_open() as system:
        request = command(key="wal-history")
        retained = system.commands.execute(request, proof=proof())
    # Keep this fixture writer open so its committed WAL frames are present
    # when native SQLite opens. The native path never uses immutable=1 or an
    # unsafe main-file-only view that could hide those committed changes.
    connection = sqlite3.connect(tmp_path / "authority.sqlite3", isolation_level=None)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        trigger = "immutable_authority_payloads_update"
        definition = connection.execute("SELECT sql FROM sqlite_schema WHERE name=?", (trigger,)).fetchone()[0]
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute("UPDATE authority_payloads SET payload_bytes=?", (b"{}",))
        connection.execute(definition)
        connection.execute("COMMIT")
        assert (tmp_path / "authority.sqlite3-wal").stat().st_size > 0
        with native_open() as reopened:
            with pytest.raises(AuthorityPersistenceError):
                reopened.events.command_result(retained.command_id, proof=proof())
            fresh = reopened.commands.execute(command(key="wal-new"), proof=proof())
            assert reopened.events.after(retained.ledger_seq, limit=1, proof=proof())[0].event_id == fresh.event_id
    finally:
        connection.close()


def test_native_pending_deletion_still_requires_exact_bytes(tmp_path, native_open) -> None:
    with native_open() as system:
        admitted = system.objects.admit(
            ObjectAdmissionRequest("source.capture", "pending-deletion"), b"pending source", proof=proof(),
        ).admission
        system.objects.revoke(admitted.admission_id, reason_code="REVOKED", idempotency_key="revoke", proof=proof())
        system.objects.request_deletion(
            admitted.blob.blob_digest, reason_code="DELETE", idempotency_key="request-delete", proof=proof(),
        )
    path = next(p for p in (tmp_path / "objects" / "objects").rglob("*") if p.is_file())
    path.chmod(0o600)
    path.write_bytes(b"corrupt pending source")
    path.chmod(0o400)
    with pytest.raises(AuthorityPersistenceError, match="missing or corrupt"):
        native_open()


def test_native_current_projection_reopens_and_rejects_drifted_head(tmp_path, native_open) -> None:
    from newsroom.increment4 import Increment4Neo4jCurrentBuildRequest
    from newsroom.projection import ProjectionGenerationId, ProjectionGenerationState
    generation_id = ProjectionGenerationId.parse("00000000-0000-4000-8000-000000009001")
    with native_open() as system:
        system.commands.execute(command(key="projection-prefix"), proof=proof())
        system.increment4.build_current_and_promote(
            Increment4Neo4jCurrentBuildRequest(
                generation_id=generation_id, reason_code="CURRENT_STATE_FIXTURE", idempotency_key="build-current",
            ), proof=proof(),
        )
    with native_open() as reopened:
        assert reopened.increment4.generation_status(generation_id, proof=proof()).generation.state is ProjectionGenerationState.ACTIVE
    with sqlite3.connect(tmp_path / "authority.sqlite3") as connection:
        trigger = "projection_generations_update_guard"
        sql = connection.execute("SELECT sql FROM sqlite_schema WHERE name=?", (trigger,)).fetchone()
        if sql is not None:
            connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute(
            "UPDATE projection_generations SET authority_aggregate_version=authority_aggregate_version+1 WHERE generation_id=?",
            (str(generation_id),),
        )
        if sql is not None:
            connection.execute(sql[0])
    with pytest.raises(AuthorityPersistenceError, match="projection generation"):
        native_open()


def test_native_projection_use_rechecks_head_after_open(tmp_path, native_open) -> None:
    from newsroom.increment4 import Increment4Neo4jCurrentBuildRequest
    from newsroom.projection import ProjectionGenerationId
    generation_id = ProjectionGenerationId.parse("00000000-0000-4000-8000-000000009002")
    with native_open() as system:
        system.commands.execute(command(key="projection-prefix"), proof=proof())
        system.increment4.build_current_and_promote(
            Increment4Neo4jCurrentBuildRequest(
                generation_id=generation_id, reason_code="CURRENT_STATE_FIXTURE", idempotency_key="build-current",
            ), proof=proof(),
        )
        with sqlite3.connect(tmp_path / "authority.sqlite3") as connection:
            connection.execute(
                "UPDATE projection_generations SET authority_aggregate_version=authority_aggregate_version+1 WHERE generation_id=?",
                (str(generation_id),),
            )
        with pytest.raises(AuthorityPersistenceError, match="projection generation"):
            system.increment4.generation_status(generation_id, proof=proof())


def test_native_pending_projection_delivery_reopens_without_lifecycle_replay(native_open, monkeypatch) -> None:
    from newsroom.authority._increment4_neo4j_boundary import _Increment4Neo4jBoundary
    from newsroom.increment4 import Increment4Neo4jCurrentBuildRequest
    from newsroom.projection import ProjectionGenerationId, ProjectionGenerationState
    generation_id = ProjectionGenerationId.parse("00000000-0000-4000-8000-000000009003")
    request = Increment4Neo4jCurrentBuildRequest(
        generation_id=generation_id, reason_code="CURRENT_STATE_FIXTURE", idempotency_key="pending-build",
    )
    transition = _Increment4Neo4jBoundary._transition_to_validating

    def stop_before_validation(*args, **kwargs):
        raise RuntimeError("fixture interruption after retained delivery")

    monkeypatch.setattr(_Increment4Neo4jBoundary, "_transition_to_validating", stop_before_validation)
    with native_open() as system:
        system.commands.execute(command(key="projection-prefix"), proof=proof())
        with pytest.raises(RuntimeError, match="fixture interruption"):
            system.increment4.build_current_and_promote(request, proof=proof())
    monkeypatch.setattr(_Increment4Neo4jBoundary, "_transition_to_validating", transition)
    with native_open() as reopened:
        assert reopened.increment4.generation_status(generation_id, proof=proof()).generation.state is ProjectionGenerationState.BUILDING
        reopened.increment4.build_current_and_promote(request, proof=proof())
        assert reopened.increment4.generation_status(generation_id, proof=proof()).generation.state is ProjectionGenerationState.ACTIVE
