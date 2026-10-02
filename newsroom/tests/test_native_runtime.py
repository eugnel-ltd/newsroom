import logging
import sqlite3
from pathlib import Path

import pytest

from newsroom.authority import ObjectAdmissionRequest
from newsroom.authority.persistence import AuthorityWriterBusy
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.increment5.retrieval_context import RetrievalContextJournal
from newsroom.increment6.collision import (
    CurrentCollisionEffectEnforcer, TrustedCurrentCollisionAuthorityBoundary,
)
from newsroom.increment6.dispositions import (
    _create_current_candidate_citation_read_port,
)
from newsroom.increment6.work_items import RetrievalContextAuthority
from newsroom.tests.authority_helpers import FIXED_NOW
from newsroom.tests.discovery_3d_authority_helpers import seed_check_lineage
from newsroom.tests.increment5b2_helpers import config
from newsroom.tests.projection_b2_helpers import MemoryNeo4jAdapter


def _args(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "newsroom.authority._graphiti_increment4_system._open_structural_graph_adapter",
        lambda _: MemoryNeo4jAdapter(),
    )
    journal = RetrievalContextJournal(tmp_path / "retrieval.sqlite3")
    return dict(
        authority_path=tmp_path / "authority.sqlite3",
        object_root=tmp_path / "objects", workspace_root=tmp_path,
        intake_path=tmp_path / "intake.sqlite3", target_path=tmp_path / "serving.sqlite3",
        target_id="hermes-private-serving", credential="token-1",
        principal_id="principal.alpha", authority_domain="newsroom.authority",
        neo4j_config=config(), retrieval_authority=RetrievalContextAuthority(journal.path, {}),
        collision_enforcer=CurrentCollisionEffectEnforcer(
            current_authority_provider=lambda _: None,
            trusted_boundary=TrustedCurrentCollisionAuthorityBoundary(
                "fixture-scope", "fixture-profile", "sha256:" + "a" * 64,
                "sha256:" + "b" * 64, "fixture-port",
            ),
        ), clock=lambda: FIXED_NOW,
    )


def test_native_runtime_real_policy_composition_and_reopen(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        seed_check_lineage(runtime.authority)
        source = runtime.authority.objects.admit(
            ObjectAdmissionRequest("evidence.source", "source-1"), b"source bytes",
            proof=runtime.proof,
        ).admission
        assert source.allowed_use == "publication_evidence"
        assert runtime.ingress.receipt_count == 0
        assert runtime.publication is not None
        with pytest.raises(AuthorityWriterBusy):
            open_native_runtime(**args)
    with open_native_runtime(**args) as reopened:
        same = reopened.authority.objects.admit(
            ObjectAdmissionRequest("evidence.source", "source-1"), b"source bytes",
            proof=reopened.proof,
        ).admission
        assert same == source
        assert reopened.ingress.receipt_count == 0
    # No target row, provider invocation or fake retrieval success was created.


def test_existing_private_target_survives_licence_markup_normalisation(tmp_path, monkeypatch):
    from newsroom.control_plane import govuk_rights
    from newsroom.increment10.private_serving import PrivateServingError

    args = _args(tmp_path, monkeypatch)
    # Exact pre-#1037 permission contract, not a value derived from the new code.
    prior_policy = "sha256:e3f0f49394d325d0cf82178f3d55d0eea529e4eda4e0d95b1e385d9abe9bffec"
    with monkeypatch.context() as prior:
        prior.setattr(govuk_rights, "POLICY_DIGEST", prior_policy)
        with open_native_runtime(**args):
            pass
    with sqlite3.connect(args["target_path"]) as connection:
        original = connection.execute("SELECT * FROM private_serving_metadata").fetchall()
    with open_native_runtime(**args):
        pass
    with sqlite3.connect(args["target_path"]) as connection:
        assert connection.execute("SELECT * FROM private_serving_metadata").fetchall() == original
    # A genuinely different permission contract still cannot silently bind an
    # existing private destination; no metadata rewrite or blanket fallback.
    monkeypatch.setattr(govuk_rights, "POLICY_DIGEST", "sha256:" + "f" * 64)
    with pytest.raises(PrivateServingError, match="target binding differs"):
        open_native_runtime(**args)


def test_native_runtime_rejects_overlapping_store_identity_before_open(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    args["target_path"] = args["authority_path"]
    with pytest.raises(ValueError, match="must be distinct"):
        open_native_runtime(**args)
    assert not args["authority_path"].exists()


def test_native_runtime_builds_dependencies_from_opened_base_before_children(
    tmp_path, monkeypatch, caplog,
):
    caplog.set_level(logging.INFO, logger="newsroom.authority.open")
    args = _args(tmp_path, monkeypatch)
    retrieval = args.pop("retrieval_authority")
    collision = args.pop("collision_enforcer")
    citations = _create_current_candidate_citation_read_port(
        lambda *_: (_ for _ in ()).throw(LookupError("no retained citation"))
    )
    captured = []

    def build_dependencies(*, objects, extraction, commands, events):
        assert args["authority_path"].exists()
        captured.append((objects, extraction, commands, events))
        return retrieval, collision, citations

    args["native_dependency_factory"] = build_dependencies
    boot_records, history_records = [], []
    with open_native_runtime(**args) as runtime:
        assert len(captured) == 1
        objects, extraction, commands, events = captured[0]
        assert objects is runtime.authority.objects
        assert extraction is runtime.authority.extraction
        assert commands is runtime.authority.commands
        assert events is runtime.authority.events
        seed_check_lineage(runtime.authority)
        boot_records.extend(r for r in caplog.records if r.name == "newsroom.authority.open")
        before_history = len(caplog.records)
        runtime.authority.validate_retained_history()
        history_records.extend(r for r in caplog.records[before_history:] if r.name == "newsroom.authority.open")

    before_reopen = len(caplog.records)
    with open_native_runtime(**args) as reopened:
        assert len(captured) == 2
        assert captured[1][2] is reopened.authority.commands
        assert captured[1][3] is reopened.authority.events
        boot_records.extend(r for r in caplog.records[before_reopen:] if r.name == "newsroom.authority.open")
        before_history = len(caplog.records)
        reopened.authority.validate_retained_history()
        history_records.extend(r for r in caplog.records[before_history:] if r.name == "newsroom.authority.open")

    records = [r for r in caplog.records if r.name == "newsroom.authority.open"]
    historical_stages = (
        "extraction_integrity", "entity_integrity", "editorial_relation_integrity",
        "graphiti_integrity", "source_integrity", "check_integrity", "discovery_integrity",
    )
    startup_stages = ("projection_integrity", "cas_reconciliation", "native_dependencies", "native_semantic_stores")
    assert not any(r.args[0] in historical_stages for r in boot_records)
    for selected_records, stages in ((history_records, historical_stages), (boot_records, startup_stages)):
        for stage in stages:
            selected = [r for r in selected_records if r.args[0] == stage]
            assert [r.args[1] if len(r.args) > 1 else "STARTED" for r in selected] == [
                "STARTED", "COMPLETE", "STARTED", "COMPLETE",
            ]
    # Historical integrity still runs explicitly. Its nested phase totals
    # remain inclusive, and CURRENT boot does not silently run those sweeps.
    stack = []
    for record in records:
        assert record.levelno == logging.INFO
        stage = record.args[0]
        if len(record.args) == 1:
            stack.append(stage)
        else:
            assert stack.pop() == stage
            assert type(record.args[-1]) is int and record.args[-1] >= 0
    assert stack == []
    messages = [r.getMessage() for r in boot_records]
    validation_end = next(i for i, r in enumerate(boot_records) if r.args[:2] == ("validation", "COMPLETE"))
    # CURRENT projection integrity remains an actual boot boundary, not a
    # retired-history sweep. Do not waive it alongside historical children.
    assert messages.index("authority_open stage=projection_integrity status=STARTED") < validation_end
    assert messages.index("authority_open stage=cas_reconciliation status=STARTED") > validation_end
    assert messages.index("authority_open stage=native_semantic_stores status=STARTED") > validation_end


def test_native_runtime_factory_failure_closes_base_writer(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="newsroom.authority.open")
    args = _args(tmp_path, monkeypatch)
    retrieval = args.pop("retrieval_authority")
    collision = args.pop("collision_enforcer")

    failure = RuntimeError("dependency construction failed")

    def fail_factory(**_):
        raise failure

    args["native_dependency_factory"] = fail_factory
    with pytest.raises(RuntimeError, match="dependency construction failed") as raised:
        open_native_runtime(**args)
    assert raised.value is failure
    assert [r.args[0] for r in caplog.records if "status=FAILED" in r.getMessage()] == [
        "native_dependencies",
    ]

    args.pop("native_dependency_factory")
    args.update(retrieval_authority=retrieval, collision_enforcer=collision)
    with open_native_runtime(**args):
        pass


@pytest.mark.parametrize("failure", (None, "relationship", "lineage", "candidate"))
def test_explicit_native_history_shares_only_one_stable_validation_transaction(
    tmp_path, monkeypatch, failure, caplog
):
    from newsroom.authority import _hermes_native_system as native

    caplog.set_level(logging.INFO, logger="newsroom.authority.open")
    injected = ValueError("injected composed validation failure")
    args = _args(tmp_path, monkeypatch)
    connection = None
    shared_inputs = None
    visited = []
    statements = []
    original_relationship = native._SharedRelationshipStore._verify_relationships
    original_lineage = native._SharedLineageStore._verify
    original_candidate = native._SharedCandidateStore._verify

    def checkpoint(store, stage):
        nonlocal connection
        if connection is None:
            connection = store._connection
            connection.set_trace_callback(statements.append)
        assert store._connection is connection
        assert connection.in_transaction
        visited.append(stage)
        if stage == failure:
            raise injected

    def relationships(store, **kwargs):
        nonlocal shared_inputs
        checkpoint(store, "relationship")
        assert isinstance(kwargs["verified_versions"], dict)
        shared_inputs = original_relationship(store, **kwargs)
        return shared_inputs

    def lineage(store, **kwargs):
        checkpoint(store, "lineage")
        assert kwargs["relationship_inputs"] is shared_inputs
        return original_lineage(store, **kwargs)

    def candidate(store, **kwargs):
        checkpoint(store, "candidate")
        assert kwargs["relationship_receipts"] is shared_inputs[1]
        # A second SQLite writer cannot alter the shared baseline between checks.
        with sqlite3.connect(args["authority_path"], timeout=0) as other:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                other.execute("BEGIN IMMEDIATE")
        return original_candidate(store, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(native._SharedRelationshipStore, "_verify_relationships", relationships)
        scoped.setattr(native._SharedLineageStore, "_verify", lineage)
        scoped.setattr(native._SharedCandidateStore, "_verify", candidate)
        if failure is None:
            with open_native_runtime(**args) as runtime:
                assert visited == []
                runtime.authority.validate_retained_history()
                assert visited == ["relationship", "lineage", "candidate"]
                assert not connection.in_transaction
                assert "ROLLBACK" not in statements
                assert statements.count("COMMIT") == 1
        else:
            with pytest.raises(ValueError, match="injected composed validation failure") as raised:
                with open_native_runtime(**args) as runtime:
                    assert visited == []
                    runtime.authority.validate_retained_history()
            assert raised.value is injected
            assert visited == ["relationship", "lineage", "candidate"][:1 + ("relationship", "lineage", "candidate").index(failure)]
            assert statements[-1] == "ROLLBACK"
        # Success and every failure path release the root writer.
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")

    with open_native_runtime(**args):
        pass


def test_native_runtime_rejects_ambiguous_dependency_setup_before_open(
    tmp_path, monkeypatch,
):
    args = _args(tmp_path, monkeypatch)
    args["native_dependency_factory"] = lambda **_: (
        args["retrieval_authority"], args["collision_enforcer"],
    )
    with pytest.raises(TypeError, match="conflicts with explicit"):
        open_native_runtime(**args)
    assert not args["authority_path"].exists()


@pytest.mark.parametrize("phase", ("graphiti_integrity", "cas_reconciliation"))
def test_native_open_phase_failure_retains_exact_exception(tmp_path, monkeypatch, caplog, phase):
    from newsroom.authority._graphiti_adapter_store_integrity import _GraphitiAdapterIntegrityMixin
    from newsroom.authority._object_store_base import _ObjectStoreBase

    args = _args(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO, logger="newsroom.authority.open")
    failure = RuntimeError("private fault must not appear in phase log")
    calls = []

    def fail(*_):
        calls.append(phase)
        raise failure

    owner, method = (
        (_GraphitiAdapterIntegrityMixin, "_validate_graphiti_adapter_integrity")
        if phase == "graphiti_integrity" else (_ObjectStoreBase, "_reconcile_objects")
    )
    monkeypatch.setattr(owner, method, fail)
    with pytest.raises(RuntimeError) as raised:
        with open_native_runtime(**args) as runtime:
            if phase == "graphiti_integrity":
                assert calls == []
                runtime.authority.validate_retained_history()
    assert raised.value is failure
    assert calls == [phase]
    records = [r for r in caplog.records if r.name == "newsroom.authority.open"]
    failed = [r.args[0] for r in records if "status=FAILED" in r.getMessage()]
    assert failed == [phase]
    assert all(str(failure) not in r.getMessage() for r in records)
