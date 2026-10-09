from __future__ import annotations

import sqlite3
import json
import os
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from newsroom.authority import AuthorityCommands, AuthorityEvents, ObjectAdmissionRequest
from newsroom.authority.canonical import digest_bytes, digest_canonical
from newsroom.control_plane import (
    native_assessor, native_composition, native_embeddings, native_source_rights,
)
from newsroom.control_plane import govuk_rights
from newsroom.control_plane.govuk_rights import GovUkLicenceEvidence, POLICY_DIGEST
from newsroom.control_plane.model_usage import InvocationEfficiencyPolicy, WorkloadClass
from newsroom.control_plane.native_collision import NativeCollisionAuthority
from newsroom.control_plane.native_pipeline import NativePipeline, NativePipelineReport
from newsroom.control_plane.native_retrieval import NativeRetrievalContinuation
from newsroom.control_plane.writer import CONT_DISABLED_CAPABILITIES, CONT_PRIMARY_COMMAND_FLAGS
from newsroom.increment5.native_retrieval import NativeRetrievalDocuments, NativeRetrievalHold
from newsroom.increment9.proving import SOURCE_URLS
from newsroom.sources import SourceDefinitionId
from newsroom.tests.increment5b2_helpers import config
from newsroom.tests.projection_b2_helpers import MemoryNeo4jAdapter
from newsroom.tests.test_native_graphiti import _native


NOW = datetime(2026, 9, 8, 14, tzinfo=UTC)


def test_assessment_consumer_contract_binds_producer_and_rendering_policies():
    assert native_assessor.VERSION == "newsroom.native-evidence-assessor.v23"
    assert native_composition.ASSESSMENT_CONTRACT_VERSION == (
        "newsroom.native-evidence-assessor.v23+newsroom.named-entity.v16+"
        "newsroom.zh-hant-hk-shape.v14+newsroom.factual-localisation.v2+"
        "newsroom.qualification-relation.v5+newsroom.retained-assessment.v1+"
        "newsroom.native-assessor-spans.v2+newsroom.native-hko-qualification-clause.v1"
        "+newsroom.native-context-materialisation.v3+newsroom.source-qualification-consumer.v4"
        "+newsroom.source-qualification-resolution-consumer.v1"
        "+newsroom.source-qualification-resolution-consumer.v2+newsroom.source-qualification-replay.v1"
    )


def test_native_cursor_credential_loads_only_provisioned_key_and_restores_environment(
    tmp_path, monkeypatch,
):
    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    path = tmp_path / "Coding/newsroom/.env"
    path.parent.mkdir(parents=True)
    path.write_text("CURSOR_API_KEY='fixture-sdk-key'\nUNRELATED_SECRET=not-loaded\n")
    path.chmod(0o600)
    with native_composition._native_cursor_credential():
        assert os.environ["CURSOR_API_KEY"] == "fixture-sdk-key"
        assert "UNRELATED_SECRET" not in os.environ
    assert "CURSOR_API_KEY" not in os.environ
    path.write_text("CURSOR_API_KEY=\n")
    with pytest.raises(ValueError, match="credential is absent"):
        with native_composition._native_cursor_credential():
            pytest.fail("empty credential entered")
    monkeypatch.setenv("CURSOR_API_KEY", "already-provisioned")
    with native_composition._native_cursor_credential():
        assert os.environ["CURSOR_API_KEY"] == "already-provisioned"
    assert os.environ["CURSOR_API_KEY"] == "already-provisioned"


@pytest.mark.parametrize(
    "missing_workload",
    (WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING, WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
     "stale-assessor-contract", "stale-assessor-flags", "stale-assessor-reasoning",
     "stale-assessor-model", "stale-assessor-output-limit"),
)
def test_deployed_startup_rejects_unqualified_policy_before_credentials_or_io(
    tmp_path, monkeypatch, missing_workload,
):
    from newsroom.control_plane import broker, cycle, paths, writer

    root = tmp_path / "native"
    root.mkdir()
    for name in ("evidence-intake", "private-serving", "retrieval"):
        (root / f"{name}.sqlite3").touch()
    for constant, name in (
        ("CANONICAL_INCREMENT4_AUTHORITY_STORE", "authority.sqlite3"),
        ("CANONICAL_UNPUBLISHED_STORE", "private.sqlite3"),
        ("CANONICAL_PROVING_STORE", "proving.sqlite3"),
    ):
        path = tmp_path / name
        path.touch()
        monkeypatch.setattr(paths, constant, path)
    workspace = tmp_path / "graphiti-workspaces"
    workspace.mkdir()
    monkeypatch.setattr(paths, "CANONICAL_OBJECT_CAS_ROOT", tmp_path)
    monkeypatch.setattr(paths, "CANONICAL_GRAPHITI_WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(paths, "HOST_CONTROL_PLANE_STATE_ROOT", tmp_path)
    monkeypatch.setattr(cycle, "assert_no_owner_emergency_stop", lambda _: None)
    monkeypatch.setattr(writer, "cont_writer_implementation_identity", lambda: ("1" * 40, True))
    monkeypatch.setattr(native_composition.subprocess, "check_output", lambda *_a, **_k: "2" * 40)
    policies = {
        WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING: _embedding_policy(),
        WorkloadClass.NATIVE_EVIDENCE_ASSESSOR: _assessment_policy(),
    }
    def qualified_policy(**request):
        if request["workload_class"] is missing_workload:
            raise ValueError("qualification is absent")
        if (isinstance(missing_workload, str) and missing_workload.startswith("stale-assessor-")
                and request["workload_class"] is WorkloadClass.NATIVE_EVIDENCE_ASSESSOR):
            values = asdict(policies[request["workload_class"]])
            values.pop("canonical_digest")
            values.update(
                prompt_contract_version=("stale-contract" if missing_workload == "stale-assessor-contract"
                                         else native_assessor.VERSION),
                reasoning=("low" if missing_workload == "stale-assessor-reasoning"
                           else native_assessor.REASONING),
                command_flags=(CONT_PRIMARY_COMMAND_FLAGS if missing_workload == "stale-assessor-flags"
                               else native_assessor.COMMAND_FLAGS),
                model=("grok-4.6" if missing_workload == "stale-assessor-model"
                       else native_assessor.MODEL),
                max_output_tokens=(10_000 if missing_workload == "stale-assessor-output-limit"
                                   else None),
            )
            return InvocationEfficiencyPolicy.create(**values)
        return policies[request["workload_class"]]

    monkeypatch.setattr(native_composition, "ModelUsageService", lambda _: SimpleNamespace(
        qualified_policy=qualified_policy,
    ))

    def unexpected(*_args, **_kwargs):
        raise AssertionError("unqualified startup reached credentials or source/provider I/O")

    monkeypatch.setattr(broker, "neo4j_projector_config", unexpected)
    monkeypatch.setattr(broker, "openrouter_api_key", unexpected)
    monkeypatch.setattr(native_composition, "open_native_pipeline", unexpected)
    service = native_composition.deployed_native_service(SimpleNamespace(
        ledger=str(paths.CANONICAL_UNPUBLISHED_STORE), lock=str(root / "hermes.lock"),
        once=False, interval=300, failure_backoff=60,
    ))
    expected = ("profile differs before authority OPEN"
                if isinstance(missing_workload, str) and missing_workload.startswith("stale-assessor-")
                else "qualification is absent")
    with pytest.raises(ValueError, match=expected):
        service.run()


@pytest.mark.parametrize("replace_existing", (False, True))
def test_deployed_continuous_runtime_runs_without_history_qualification_gate(
    tmp_path, monkeypatch, replace_existing,
):
    from newsroom.control_plane import broker, cycle, native_qualification, paths, writer

    native_root = tmp_path / "native"
    native_root.mkdir()
    authority = tmp_path / "authority.sqlite3"
    with sqlite3.connect(authority) as connection:
        connection.execute(
            "CREATE TABLE source_definition_version_heads("
            "definition_id TEXT,current_version_id TEXT)"
        )
        connection.execute(
            "CREATE TABLE source_definition_versions("
            "version_id TEXT,locator TEXT)"
        )
    private = tmp_path / "private.sqlite3"
    private.touch()
    proving = tmp_path / "proving.sqlite3"
    proving.touch()
    cas = tmp_path / "objects"
    cas.mkdir()
    workspace = tmp_path / "workspaces"
    workspace.mkdir()
    for name, value in {
        "CANONICAL_INCREMENT4_AUTHORITY_STORE": authority,
        "CANONICAL_UNPUBLISHED_STORE": private,
        "CANONICAL_PROVING_STORE": proving,
        "CANONICAL_OBJECT_CAS_ROOT": cas,
        "CANONICAL_GRAPHITI_WORKSPACE_ROOT": workspace,
        "HOST_CONTROL_PLANE_STATE_ROOT": tmp_path,
    }.items():
        monkeypatch.setattr(paths, name, value)
    monkeypatch.setattr(cycle, "assert_no_owner_emergency_stop", lambda _: None)
    monkeypatch.setattr(
        writer, "cont_writer_implementation_identity", lambda: ("1" * 40, True)
    )
    monkeypatch.setattr(
        native_composition.subprocess, "check_output", lambda *_a, **_k: "2" * 40
    )
    policies = {
        WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING: _embedding_policy(),
        WorkloadClass.NATIVE_EVIDENCE_ASSESSOR: _assessment_policy(),
    }
    def fixture_qualified_policy(**request):
        from newsroom.control_plane.model_usage import ModelUsageAdmissionError
        if request['workload_class'] not in policies:
            raise ModelUsageAdmissionError('fixture semantic policy is not registered')
        return policies[request['workload_class']]

    monkeypatch.setattr(
        native_composition,
        "ModelUsageService",
        lambda _: SimpleNamespace(
            qualified_policy=fixture_qualified_policy
        ),
    )
    monkeypatch.setattr(
        native_composition, "_native_cursor_credential", lambda: nullcontext()
    )
    monkeypatch.setattr(broker, "neo4j_projector_config", lambda: object())
    monkeypatch.setattr(broker, "openrouter_api_key", lambda: "fixture-key")
    opens = []
    opened_arguments = []

    @contextmanager
    def opened_pipeline(**arguments):
        opens.append("open")
        opened_arguments.append(arguments)
        for name in ("intake_path", "serving_path", "retrieval_path"):
            Path(arguments[name]).touch()
        if replace_existing:
            replacement = authority.with_name("replacement.sqlite3")
            replacement.write_bytes(authority.read_bytes())
            replacement.replace(authority)
        try:
            yield SimpleNamespace(
                tick=lambda **_request: NativePipelineReport((), {}, 0),
                terminal_report=asdict,
            )
        finally:
            opens.append("close")

    monkeypatch.setattr(native_composition, "open_native_pipeline", opened_pipeline)
    monkeypatch.setattr(
        native_qualification,
        "validate_qualification",
        lambda *_a, **_k: pytest.fail("continuous startup required prior qualification"),
    )
    qualified = []
    monkeypatch.setattr(
        native_qualification,
        "record_qualification",
        lambda _connection, identity: qualified.append(identity),
    )

    service = native_composition.deployed_native_service(
        SimpleNamespace(
            ledger=str(private), lock=str(native_root / "hermes.lock"),
            once=False, interval=300, failure_backoff=60,
        )
    )
    service._wait = lambda _: True
    if replace_existing:
        with pytest.raises(ValueError, match="identity changed during open"):
            service.run()
        assert qualified == []
    else:
        report = service.run()
        assert report is not None and report.outcome == "COMPLETE"
        assert qualified == []
        assert service._qualify_once is None
    assert opens == ["open", "close"]
    assert opened_arguments[0]["reassessment_quantum_seconds"] == 300


def test_native_deployment_identity_binds_store_instance_not_changing_contents(tmp_path):
    store = tmp_path / "authority.sqlite3"
    store.write_bytes(b"first retained state")
    arguments = dict(
        revision="1" * 40, tree="2" * 40, paths={"authority": store},
        embedding_policy=_embedding_policy(), assessment_policy=_assessment_policy(),
    )
    identity = native_composition._deployment_identity(**arguments)
    store.write_bytes(b"next retained state")
    assert native_composition._deployment_identity(**arguments) == identity
    assert native_composition._deployment_identity(**{**arguments, "revision": "3" * 40}) != identity
    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(store.read_bytes())
    replacement.replace(store)
    assert native_composition._deployment_identity(**arguments) != identity


def test_deployed_service_rejects_symlink_store_before_any_mutation(tmp_path, monkeypatch):
    from newsroom.control_plane import broker, cycle, paths

    native_root = tmp_path / "native"
    native_root.mkdir()
    authority = tmp_path / "authority.sqlite3"
    proving = tmp_path / "proving.sqlite3"
    victim = tmp_path / "outside-private.sqlite3"
    authority.touch()
    proving.touch()
    victim.write_bytes(b"do not mutate")
    private = tmp_path / "private.sqlite3"
    private.symlink_to(victim)
    cas = tmp_path / "objects"
    workspace = tmp_path / "workspaces"
    cas.mkdir()
    workspace.mkdir()
    for name, value in {
        "CANONICAL_INCREMENT4_AUTHORITY_STORE": authority,
        "CANONICAL_UNPUBLISHED_STORE": private,
        "CANONICAL_PROVING_STORE": proving,
        "CANONICAL_OBJECT_CAS_ROOT": cas,
        "CANONICAL_GRAPHITI_WORKSPACE_ROOT": workspace,
        "HOST_CONTROL_PLANE_STATE_ROOT": tmp_path,
    }.items():
        monkeypatch.setattr(paths, name, value)
    monkeypatch.setattr(cycle, "assert_no_owner_emergency_stop", lambda _: None)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("failed preflight reached credentials or pipeline")

    monkeypatch.setattr(native_composition, "_native_cursor_credential", unexpected)
    monkeypatch.setattr(native_composition, "open_native_pipeline", unexpected)
    monkeypatch.setattr(broker, "openrouter_api_key", unexpected)
    lock = native_root / "hermes.lock"
    service = native_composition.deployed_native_service(SimpleNamespace(
        ledger=str(private), lock=str(lock), once=True,
        interval=300, failure_backoff=60,
    ))
    with pytest.raises(ValueError, match="path contains a symlink"):
        service.run(once=True)
    assert victim.read_bytes() == b"do not mutate"
    assert not lock.exists()


def test_deployment_preflight_rejects_creatable_file_below_symlink_parent(tmp_path):
    required_file = tmp_path / "authority.sqlite3"
    required_directory = tmp_path / "objects"
    actual_parent = tmp_path / "actual-native"
    redirected_parent = tmp_path / "native"
    required_file.touch()
    required_directory.mkdir()
    actual_parent.mkdir()
    redirected_parent.symlink_to(actual_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="path contains a symlink"):
        native_composition._native_deployment_preflight(
            supplied_ledger=required_file, supplied_lock=redirected_parent / "lock",
            expected_ledger=required_file, expected_lock=redirected_parent / "lock",
            required_files={"authority": required_file},
            required_directories={"Object CAS": required_directory},
            creatable_files={"singleton lock": redirected_parent / "lock"},
        )
    assert tuple(actual_parent.iterdir()) == ()


def _embedding_policy() -> InvocationEfficiencyPolicy:
    return InvocationEfficiencyPolicy.create(
        policy_id="native-composition-embedding", version="v1",
        workload_class=WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING,
        provider="openrouter", route=native_embeddings.ROUTE,
        model=native_embeddings.OPENROUTER_EMBEDDING_SLUG, reasoning="none",
        one_turn=True, exact_input=True, skills_enabled=False, tools_enabled=False,
        mcp_enabled=False, prior_message_count=0,
        command_semantic_version=native_embeddings.VERSION,
        command_flags=("POST=/embeddings",),
        context_manifest_schema_version=native_embeddings.VERSION,
        disabled_capabilities=("tools",),
        implementation_revision=native_embeddings.implementation_digest(),
        max_prompt_bytes=20_000, max_context_tokens=8_000,
        max_output_tokens=1, max_total_tokens=8_000,
        prompt_contract_version=native_embeddings.VERSION,
        output_schema_digest=native_embeddings.SCHEMA_DIGEST,
        allowed_context_identities=(native_embeddings.VERSION,),
        allowed_config_identities=(native_embeddings.VERSION,),
        hard_estimate_ceiling_tokens=None,
        evidence_digest=digest_canonical({"test": "native composition embedding"}),
        qualified=True,
    )


def _assessment_policy() -> InvocationEfficiencyPolicy:
    return InvocationEfficiencyPolicy.create(
        policy_id="native-composition-assessment", version="v1",
        workload_class=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
        provider="grok-build-cli", route=native_assessor.ROUTE,
        model=native_assessor.MODEL, reasoning=native_assessor.REASONING,
        one_turn=True, exact_input=True,
        skills_enabled=False, tools_enabled=False, mcp_enabled=False,
        prior_message_count=0, command_semantic_version="1.0.8",
        command_flags=native_assessor.COMMAND_FLAGS,
        context_manifest_schema_version=native_assessor.CONTEXT_MANIFEST_SCHEMA_VERSION,
        disabled_capabilities=CONT_DISABLED_CAPABILITIES,
        implementation_revision="1" * 40, max_prompt_bytes=1_000_000,
        max_context_tokens=100_000, max_output_tokens=None,
        max_total_tokens=100_000, prompt_contract_version=native_assessor.VERSION,
        output_schema_digest=native_assessor.PROVIDER_SCHEMA_DIGEST,
        allowed_context_identities=(native_assessor.CONTEXT_IDENTITY,),
        allowed_config_identities=(native_assessor.CONFIG_IDENTITY,),
        hard_estimate_ceiling_tokens=100_000,
        evidence_digest=digest_canonical({"test": "native composition assessment"}),
        qualified=True,
    )


class _RetrievalProjection:
    bootstraps = 0

    def __init__(self, *_args, **_kwargs) -> None:
        return None

    def bootstrap(self) -> None:
        type(self).bootstraps += 1

    def upsert(self, *_args, **_kwargs) -> None:
        raise AssertionError("test opener must not project a document")

    def retrieve(self, *_args, **_kwargs):
        raise AssertionError("test opener must not retrieve")

    def retrieve_vector(self, *_args, **_kwargs):
        raise AssertionError("test opener must not retrieve")


class _Reader:
    def close(self) -> None:
        return None


def _arguments(tmp_path):
    proving = tmp_path / "proving.sqlite3"
    sqlite3.connect(proving).close()
    return {
        "authority_path": tmp_path / "authority.sqlite3",
        "object_root": tmp_path / "objects",
        "workspace_root": tmp_path,
        "private_path": tmp_path / "private.sqlite3",
        "proving_path": proving,
        "intake_path": tmp_path / "intake.sqlite3",
        "serving_path": tmp_path / "serving.sqlite3",
        "retrieval_path": tmp_path / "retrieval.sqlite3",
        "neo4j_config": config(),
        "embedding_key": "test-key-never-dispatched",
        "embedding_policy": _embedding_policy(),
        "assessment_policy": _assessment_policy(),
        "source_definition_ids": {},
        "licence": None,
        "implementation_worktree_clean": True,
        "clock": lambda: NOW,
    }


def test_native_composition_opens_factory_once_reopens_and_has_no_pre_effect(
    tmp_path, monkeypatch,
) -> None:
    _RetrievalProjection.bootstraps = 0
    rights_observations = []
    monkeypatch.setattr(native_composition, "fetch_licensing_observations", lambda **_: {})

    def observe_terms(**_):
        rights_observations.append("observed")
        return {
            source: native_source_rights.SourceTermsEvidence(
                source, NOW.isoformat(), "SOURCE_TERMS_UNAVAILABLE", (),
            )
            for source in native_source_rights.TERMS
        }

    monkeypatch.setattr(
        native_composition, "observe_portfolio_terms", observe_terms,
    )
    monkeypatch.setattr(
        "newsroom.authority._graphiti_increment4_system._open_structural_graph_adapter",
        lambda _: MemoryNeo4jAdapter(),
    )
    monkeypatch.setattr(
        native_composition,
        "open_native_retrieval_neo4j_resources",
        lambda **_arguments: SimpleNamespace(
            projector=_RetrievalProjection(),
            fulltext=_Reader(),
            close=lambda: None,
        ),
    )
    terms = (
        b"<html><main>Reviewed GOV.UK reuse terms.</main></html>",
        b"<html><main>Reviewed Open Government Licence terms.</main></html>",
    )
    monkeypatch.setattr(
        govuk_rights,
        "REVIEWED_TEXT",
        {
            url: govuk_rights.licence_text_digest(raw)
            for url, raw in zip(
                (govuk_rights.REUSE_URL, govuk_rights.LICENCE_URL), terms,
                strict=True,
            )
        },
    )

    def retain_licence(*, objects, proof, dispatch_fence, clock, fetch=None):
        for _ in terms:
            with dispatch_fence():
                pass
        admissions = tuple(
            objects.admit(
                ObjectAdmissionRequest(
                    "evidence.source", f"native-composition-licence:{digest_bytes(raw)}"
                ),
                raw,
                proof=proof,
            ).admission
            for raw in terms
        )
        return GovUkLicenceEvidence(
            tuple(item.admission_id for item in admissions),
            tuple(item.blob.blob_digest for item in admissions),
            NOW.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            POLICY_DIGEST,
        )

    monkeypatch.setattr(
        native_composition, "retain_current_govuk_licence", retain_licence
    )
    stops = []
    fences = []

    def stop_check() -> None:
        stops.append("checked")

    @contextmanager
    def stop_fence():
        fences.append("entered")
        yield

    arguments = {
        **_arguments(tmp_path),
        "stop_check": stop_check,
        "stop_fence": stop_fence,
    }
    for expected_bootstraps in (1, 2):
        with native_composition.open_native_pipeline(**arguments) as pipeline:
            assert pipeline._publish.copy_correction_due({})
            assert pipeline._publish.copy_correction_due({"writer_id": "newsroom.offline-exact-copy.v3"})
            assert not pipeline._publish.copy_correction_due({"writer_id": "newsroom.native-story-writer.v1"})
            assert not pipeline._publish.copy_correction_due({"copy_correction_checked_version": "newsroom.native-story-writer.v1"})
            assert type(pipeline) is NativePipeline
            assert type(pipeline._runtime.authority.commands) is AuthorityCommands
            assert type(pipeline._runtime.authority.events) is AuthorityEvents
            assert type(pipeline._retrieval_for(())) is NativeRetrievalContinuation
            assert type(pipeline._retrieval_for(())._documents) is NativeRetrievalDocuments
            assert type(pipeline._collision) is NativeCollisionAuthority
            assert pipeline._runtime.authority.collision is pipeline._collision.enforcer
            assert pipeline._runtime.ingress.receipt_count == 0
            assert pipeline._journal.units == {}
            assert _RetrievalProjection.bootstraps == expected_bootstraps
            pipeline._refresh_rights()
            assert len(rights_observations) == expected_bootstraps
            with sqlite3.connect(arguments["serving_path"]) as serving:
                assert serving.execute(
                    "SELECT COUNT(*) FROM private_serving_payloads"
                ).fetchone()[0] == 0
            if expected_bootstraps == 2:
                interrupted = (_native("contract"), _native("unknown"))
                retained_ordinals = {}
                for unit, failure_class in zip(
                    interrupted, ("EvidencePackageError", "OSError"), strict=True
                ):
                    pipeline._journal.land((unit,))
                    pipeline._journal.advance(
                        unit.revision_id,
                        stage="ASSESSMENT_INTERRUPTED",
                        facts={
                            "candidate_version_id": "candidate:" + unit.item_key,
                            "failure_class": failure_class,
                            "reason": "ACQUISITION_RESULT_NOT_RETAINED",
                        },
                    )
                    retained_ordinals[unit.revision_id] = (
                        pipeline._journal.current(unit.revision_id)["ordinal"]
                    )

                def unexpected_hydration(**_arguments):
                    raise AssertionError("interrupted recovery hydrated source objects")

                recovery_sources = []

                class ProofOnlyContinuation:
                    def __init__(self, **arguments):
                        recovery_sources.append(arguments["sources"])

                    def advance(self, *, revision_id, candidate_version_id):
                        return SimpleNamespace(state="ASSESSMENT_INTERRUPTED")

                monkeypatch.setattr(
                    native_composition,
                    "native_evidence_sources",
                    unexpected_hydration,
                )
                monkeypatch.setattr(
                    native_composition,
                    "NativePublicationContinuation",
                    ProofOnlyContinuation,
                )
                for unit in interrupted:
                    pipeline._publish.advance(
                        revision_id=unit.revision_id,
                        candidate_version_id="candidate:" + unit.item_key,
                    )
                assert recovery_sources == [{}, {}]
                assert {
                    revision_id: pipeline._journal.current(revision_id)["ordinal"]
                    for revision_id in retained_ordinals
                } == retained_ordinals
                acknowledged = _native("copy-ack")
                pipeline._journal.land((acknowledged,))
                facts = {"candidate_version_id": "candidate:copy-ack", "graphiti_receipts": [{}]}
                pipeline._journal.advance(acknowledged.revision_id, stage="ACKNOWLEDGED", facts=facts)

                def unavailable_source(**_):
                    raise native_composition.NativeEvidenceHold("SOURCE_TEST_HOLD", acknowledged.source_id)

                monkeypatch.setattr(native_composition, "native_evidence_sources", unavailable_source)
                selected = ((acknowledged.revision_id, (acknowledged,)),)
                pipeline._advance_revisions(selected, work_deadline=float("inf"))
                assert recovery_sources == [{}, {}, {}]
                pipeline._journal.advance(acknowledged.revision_id, stage="ACKNOWLEDGED", facts={
                    **facts, "writer_id": "newsroom.offline-exact-copy.v3",
                })
                pipeline._advance_revisions(selected, work_deadline=float("inf"))
                assert recovery_sources == [{}, {}, {}, {}]
                pipeline._journal.advance(acknowledged.revision_id, stage="ACKNOWLEDGED", facts={
                    **facts, "writer_id": "newsroom.native-story-writer.v1",
                })
                pipeline._advance_revisions(selected, work_deadline=float("inf"))
                assert recovery_sources == [{}, {}, {}, {}]

    with pytest.raises(NativeRetrievalHold, match="NATIVE_EMBEDDING_POLICY_HOLD"):
        with native_composition.open_native_pipeline(
            **{
                **arguments,
                "embedding_policy": replace(
                    arguments["embedding_policy"], qualified=False
                ),
            }
        ):
            raise AssertionError("unqualified composition entered")
    assert _RetrievalProjection.bootstraps == 2
    # Three opens, eleven bounded observation/assessment stop checks per open,
    # and three bounded ACK turns. No network/provider stage skips its fence.
    assert stops == ["checked"] * 39
    assert fences == ["entered"] * 8


def test_rights_refresh_registers_and_binds_a_newly_permitted_source(
    tmp_path, monkeypatch,
) -> None:
    # This registration/retrieval-key fixture supplies raw reference identities,
    # not CAS bytes. The governed reader has its own real-runtime bundle tests.
    def current_observation(**arguments):
        entry = permitted[arguments["source_id"]]
        return {"observed_at": entry.observed_at, "reason": entry.reason,
                "observations": entry.observations}
    monkeypatch.setattr(native_composition, "read_rights_observation", current_observation)
    monkeypatch.setattr(native_composition, "require_rights_assessment", lambda **_: None)
    unavailable = {
        source_id: native_source_rights.SourceTermsEvidence(
            source_id, NOW.isoformat(), "SOURCE_TERMS_UNAVAILABLE", (),
        )
        for source_id in native_source_rights.TERMS
    }
    hk02_terms_url = native_source_rights.TERMS["HK-02"][0][0]
    permitted = dict(unavailable)
    permitted["HK-02"] = native_source_rights.SourceTermsEvidence(
        "HK-02", NOW.isoformat(), "REVIEWED_REUSE_PERMITTED",
        ((hk02_terms_url, "sha256:" + "1" * 64,
          "00000000-0000-4000-8000-000000000901", "access-hk02"),),
    )
    references = {
        source_id: native_source_rights.RightsSnapshotReference(
            f"00000000-0000-4000-8001-{index:012d}", "sha256:" + "2" * 64,
            f"00000000-0000-4000-8002-{index:012d}", "sha256:" + "3" * 64,
        )
        for index, source_id in enumerate(SOURCE_URLS, 1)
    }
    portfolio = native_source_rights.NativePortfolioRights(
        None, unavailable, refresh_current=lambda: (
            None, "GOVUK_LICENCE_REVIEW_HOLD", permitted, references,
        ),
    )
    monkeypatch.setattr(
        "newsroom.authority._graphiti_increment4_system._open_structural_graph_adapter",
        lambda _: MemoryNeo4jAdapter(),
    )
    monkeypatch.setattr(
        native_composition,
        "open_native_retrieval_neo4j_resources",
        lambda **_arguments: SimpleNamespace(
            projector=_RetrievalProjection(),
            fulltext=_Reader(),
            close=lambda: None,
        ),
    )
    registered = []
    register = native_composition.register_missing_native_source_definitions

    def counted_registration(**arguments):
        registered.append(tuple(arguments["rights_by_source"]))
        return register(**arguments)

    monkeypatch.setattr(
        native_composition, "register_missing_native_source_definitions",
        counted_registration,
    )
    arguments = {
        **_arguments(tmp_path), "licence": portfolio,
        "stop_check": lambda: None, "stop_fence": nullcontext,
    }
    with native_composition.open_native_pipeline(**arguments) as pipeline:
        held = pipeline._intake._poll_one("HK-02")
        assert held.status == "HOLD"
        assert held.reason_code == "SOURCE_TERMS_UNAVAILABLE"

        pipeline._refresh_rights()
        pipeline._intake._fetch = lambda _url: (200, b"{}")
        ready = pipeline._intake._poll_one("HK-02")
        assert ready.status == "READY"
        assert ready.units
        retrieval = pipeline._retrieval_for(ready.units)
        unit = ready.units[0]
        initial_key = retrieval._rights(unit)
        initial_assessment = portfolio.for_source(
            source_id=unit.source_id, definition_url=unit.source_definition_url,
        )
        # Fresh retained raw bytes/admission IDs may change outside the already
        # reviewed substantive terms. Permission/policy/scope are unchanged.
        permitted["HK-02"] = replace(
            permitted["HK-02"],
            observed_at="2026-09-08T15:05:00+00:00",
            observations=((hk02_terms_url, "sha256:" + "4" * 64,
                           "00000000-0000-4000-8000-000000000902", "access-hk02-new"),),
        )
        pipeline._refresh_rights()
        assert portfolio.for_source(
            source_id=unit.source_id, definition_url=unit.source_definition_url,
        ).record_id != initial_assessment.record_id
        assert retrieval._rights(unit) == initial_key
        stale = replace(unit, authority=replace(
            unit.authority,
            definition_version_id="00000000-0000-4000-8000-000000000999",
        ))
        with pytest.raises(NativeRetrievalHold, match="NATIVE_CURRENT_SOURCE_RIGHTS_HOLD"):
            retrieval._rights(stale)
        assert registered == [("HK-02",)]
        # A genuinely changed reviewed policy invalidates reuse.
        monkeypatch.setattr(native_source_rights, "POLICY_DIGEST", "sha256:" + "5" * 64)
        assert retrieval._rights(unit) != initial_key
        # Missing/changed terms still exclude the subject on this very tick.
        permitted["HK-02"] = replace(permitted["HK-02"], reason="SOURCE_TERMS_CHANGED")
        pipeline._refresh_rights()
        with pytest.raises(NativeRetrievalHold, match="NATIVE_CURRENT_SOURCE_RIGHTS_HOLD"):
            retrieval._rights(unit)
        with pytest.raises(ValueError, match="binding differs"):
            pipeline._intake.bind_definitions({"HK-02": SourceDefinitionId.new()})


@pytest.fixture
def composed_rights_cohort(tmp_path, monkeypatch):
    """Existing native opener and terms fixture, with real canonical CAS readers."""
    from newsroom.control_plane.native_evidence import NativeEvidenceHold
    from newsroom.tests.test_source_rights_bundle import _terms
    bodies = _terms(monkeypatch)
    monkeypatch.setattr(native_composition, "fetch_licensing_observations", lambda **_: bodies)
    now, stops, stop_requested = [NOW], [], [False]
    def no_govuk(**_):
        raise NativeEvidenceHold("GOVUK_LICENCE_REVIEW_HOLD", "UK-GOVUK")
    def observed_terms(**arguments):
        return native_source_rights.observe_portfolio_terms(**arguments)
    def stop():
        stops.append("checked")
        if stop_requested[0]:
            raise RuntimeError("signed owner stop")
    monkeypatch.setattr(native_composition, "retain_current_govuk_licence", no_govuk)
    monkeypatch.setattr(native_composition, "observe_portfolio_terms", observed_terms)
    monkeypatch.setattr("newsroom.authority._graphiti_increment4_system._open_structural_graph_adapter",
        lambda _: MemoryNeo4jAdapter())
    monkeypatch.setattr(native_composition, "open_native_retrieval_neo4j_resources", lambda **_: SimpleNamespace(
        projector=_RetrievalProjection(), fulltext=_Reader(), close=lambda: None))
    observations, assessments = [], []
    read, require = native_composition.read_rights_observation, native_composition.require_rights_assessment
    def checked_observation(**arguments):
        observations.append(arguments["source_id"])
        return read(**arguments)
    def checked_assessment(**arguments):
        assessments.append(arguments["assessment"].record_id)
        return require(**arguments)
    monkeypatch.setattr(native_composition, "read_rights_observation", checked_observation)
    monkeypatch.setattr(native_composition, "require_rights_assessment", checked_assessment)
    arguments = {**_arguments(tmp_path), "clock": lambda: now[0], "stop_check": stop, "stop_fence": nullcontext}
    with native_composition.open_native_pipeline(**arguments) as pipeline:
        pipeline._intake._fetch = lambda _: (200, b"{}")
        ready = pipeline._intake._poll_one("HK-02")
        assert ready.status == "READY" and ready.units
        observations.clear()
        assessments.clear()
        yield SimpleNamespace(pipeline=pipeline, retrieval=pipeline._retrieval_for(ready.units),
            unit=ready.units[0], now=now, stops=stops, observations=observations,
            assessments=assessments, arguments=arguments, stop_requested=stop_requested)


def test_composed_rights_cohort_reauthenticates_real_cas_once_per_key_and_each_operation(composed_rights_cohort):
    fixture = composed_rights_cohort
    expected = fixture.retrieval._rights(fixture.unit)
    fixture.observations.clear()
    fixture.assessments.clear()
    for operation in range(2):
        with fixture.retrieval._rights_cohort() as rights:
            assert rights(fixture.unit) == expected
            for index in range(30):
                # Different passage/body identity, exactly the same rights key.
                passage = replace(fixture.unit, body=f"Passage {index}")
                before = len(fixture.stops)
                assert rights(passage) == expected
                assert len(fixture.stops) == before + 1
        assert fixture.observations == ["HK-02"] * (2 * (operation + 1))
        assert len(fixture.assessments) == 2 * (operation + 1)
    assert fixture.pipeline._runtime.ingress.receipt_count == 0
    before = len(fixture.stops)
    with pytest.raises(NativeRetrievalHold, match="NATIVE_RIGHTS_COHORT_EXPIRED"):
        rights(fixture.unit)
    assert len(fixture.stops) == before
    assert len(fixture.observations) == 4


@pytest.mark.parametrize("changed", ("version", "locator", "definition", "source"))
def test_composed_rights_cohort_never_merges_distinct_source_keys(composed_rights_cohort, changed):
    fixture = composed_rights_cohort
    unit = fixture.unit
    if changed == "version":
        unit = replace(unit, authority=replace(unit.authority, definition_version_id="00000000-0000-4000-8000-000000000999"))
    elif changed == "definition":
        unit = replace(unit, authority=replace(unit.authority, definition_id="00000000-0000-4000-8000-000000000999"))
    elif changed == "locator":
        unit = replace(unit, source_definition_url=SOURCE_URLS["UK-01"])
    else:
        unit = replace(unit, source_id="HK-01")
    with pytest.raises((NativeRetrievalHold, LookupError)):
        with fixture.retrieval._rights_cohort() as rights:
            rights(fixture.unit)
            rights(unit)
    # A failed operation discards its permissions too.
    with fixture.retrieval._rights_cohort() as rights:
        assert rights(fixture.unit)
    assert fixture.observations.count("HK-02") == 3


@pytest.mark.parametrize("changed", ("current_version", "locator_column", "rights", "observation_revoked",
    "assessment_revoked", "cas_corrupt", "expired_authentication", "stop"))
def test_composed_rights_cohort_rechecks_end_mutations_before_return_or_port(composed_rights_cohort, monkeypatch, changed):
    from datetime import timedelta
    from newsroom.authority import AuthenticationError, ObjectAdmissionDenied, ObjectAdmissionId, ObjectIntegrityError, UtcTimestamp
    from newsroom.authority.auth import StaticAuthenticator
    from newsroom.authority.persistence import AuthorityPersistenceError
    from newsroom.sources import SourceDefinitionVersionId
    fixture = composed_rights_cohort
    runtime = fixture.pipeline._runtime
    snapshot = fixture.pipeline._intake._licence.snapshot_for("HK-02")
    returned = []
    mutation_completed = False
    with pytest.raises((NativeRetrievalHold, ObjectAdmissionDenied, ObjectIntegrityError,
        AuthenticationError, AuthorityPersistenceError, RuntimeError)) as denied:
        with fixture.retrieval._rights_cohort() as rights:
            assert rights(fixture.unit)
            if changed == "current_version":
                version = runtime.authority.sources.version_details(
                    SourceDefinitionVersionId.parse(fixture.unit.authority.definition_version_id), proof=runtime.proof)
                runtime.authority.sources.record_definition_version(replace(version.request,
                    version_id=SourceDefinitionVersionId.new(), version_number=2,
                    expected_previous_version_id=version.request.version_id,
                    extraction_scope=tuple(sorted((*version.request.extraction_scope, "cohort-fixture-new-scope"))),
                    idempotency_key="fixture-new-rights-source-version"), proof=runtime.proof)
            elif changed == "locator_column":
                with sqlite3.connect(fixture.arguments["authority_path"]) as retained:
                    # Corrupt this local fixture only, following the existing
                    # canonical-versus-normalised Source integrity tests.
                    retained.execute("DROP TRIGGER immutable_source_version_update")
                    retained.execute("UPDATE source_definition_versions SET locator=? WHERE version_id=?",
                        (SOURCE_URLS["UK-01"], fixture.unit.authority.definition_version_id))
            elif changed == "rights":
                portfolio = fixture.pipeline._intake._licence
                portfolio.evidence["HK-02"] = replace(portfolio.evidence["HK-02"], reason="SOURCE_TERMS_CHANGED")
            elif changed in {"observation_revoked", "assessment_revoked"}:
                target = snapshot.observation_admission_id if changed == "observation_revoked" else snapshot.assessment_admission_id
                runtime.authority.objects.revoke(ObjectAdmissionId.parse(target), reason_code="REVOKED",
                    idempotency_key="fixture-cohort-" + changed, proof=runtime.proof)
            elif changed == "cas_corrupt":
                digest = snapshot.observation_blob_digest.removeprefix("sha256:")
                path = fixture.arguments["object_root"] / "objects" / digest[:2] / digest
                path.chmod(0o600)
                path.write_bytes(b"corrupt current rights observation")
                path.chmod(0o400)
            elif changed == "expired_authentication":
                authenticate = StaticAuthenticator.authenticate
                monkeypatch.setattr(StaticAuthenticator, "authenticate",
                    lambda self, proof, *, now: authenticate(self, proof, now=UtcTimestamp(NOW)))
                fixture.now[0] += timedelta(minutes=6)
            else:
                fixture.stop_requested[0] = True
            mutation_completed = True
        returned.append("cohort returned before a retrieval port")
    assert mutation_completed, f"mutation setup failed: {type(denied.value).__name__}: {denied.value}"
    assert returned == []
    assert fixture.pipeline._runtime.ingress.receipt_count == 0


def test_native_composition_owner_stop_precedes_store_or_provider_effects(
    tmp_path,
) -> None:
    arguments = _arguments(tmp_path)

    def stopped() -> None:
        raise RuntimeError("signed owner stop")

    arguments.update(stop_check=stopped, stop_fence=nullcontext)

    with pytest.raises(RuntimeError, match="signed owner stop"):
        with native_composition.open_native_pipeline(**arguments):
            raise AssertionError("stopped composition entered")
    assert not arguments["authority_path"].exists()
    assert not arguments["private_path"].exists()


def _publication_caller_without_bootstrap(**bindings):
    """Execute the actual nested caller only; no runtime/source/provider bootstrap."""
    import ast
    tree = ast.parse(Path(native_composition.__file__).read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == 'Publication']
    assert len(nodes) == 1
    # The extracted legacy caller has the same dormant semantic closure as the
    # ordinary opener without a configured judgment credential.
    scope = {**vars(native_composition), 'judgment_api_key': None,
        'semantic_witness_disposition_reader': None, **bindings}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), native_composition.__file__, 'exec'), scope)
    return scope['Publication']()


@pytest.mark.parametrize('initial,defect', [
    ('old-interrupted', None), ('current-source-keyerror', None),
    ('old-interrupted', 'unresolved'), ('current-source-keyerror', 'current-allocation'),
    ('current-source-keyerror', 'unallocated-current-envelope'),
    ('current-source-keyerror', 'wrong-old-proof'), ('current-source-keyerror', 'raw-keyerror'),
    ('current-source-keyerror', 'unrelated-hold'), ('current-source-keyerror', 'exhausted'),
    ('current-source-keyerror', 'changed-source'), ('current-source-keyerror', 'stop'),
])
def test_exact_composition_caller_lazily_loads_selected_sources_after_accounted_failure_proof(tmp_path, monkeypatch, initial, defect):
    from newsroom.control_plane.native_evidence import NativeEvidenceController, NativeEvidenceHold
    from newsroom.control_plane.native_progress import NativeRevisionJournal
    from newsroom.control_plane.store import connect
    from newsroom.tests.test_native_assessor import _old_provider_failure, _empty_reference_result
    from newsroom.tests.test_native_publication_continuation import _source
    candidate_connection, candidate, base, service, usage, old_allocation, old_terminal = _old_provider_failure(tmp_path, monkeypatch)
    connection = connect(str(tmp_path / 'private.sqlite3'))
    journal = NativeRevisionJournal(connection)
    unit = _native('rich-6114-source-keyerror')
    journal.land((unit,))
    facts = {
        'candidate_id': candidate.candidate_id, 'candidate_version_id': candidate.version_id,
        'graphiti_receipts': [{}], 'intake_receipt_id': 'retained-intake',
        'assessment_contract_version': 'newsroom.native-evidence-assessor.v21+consumer.v1',
        'failure_class': 'CliTimeoutError', 'reason': 'ACQUISITION_RESULT_NOT_RETAINED',
        'acquisition_attempt_count': 1,
    }
    stage = 'ASSESSMENT_INTERRUPTED'
    if initial == 'current-source-keyerror':
        old = usage.retained_old_provider_failure(candidate)
        facts['assessment_superseded'] = {'contract_version': facts['assessment_contract_version'],
            'reason': facts['reason'], 'failure_class': facts['failure_class'],
            'provider_failure': {'outcome': old.outcome, **asdict(old.proof)}}
        facts.update(assessment_contract_version='newsroom.native-evidence-assessor.v23+consumer.v1',
                     failure_class='KeyError', assessment_started_at=None)
        stage = 'EVIDENCE_HOLD'
    if defect == 'wrong-old-proof': facts['assessment_superseded']['provider_failure']['terminal_digest'] = 'sha256:' + 'f' * 64
    if defect == 'raw-keyerror': facts.pop('assessment_superseded')
    if defect == 'unrelated-hold': facts['reason'] = 'SOURCE_POLICY_FACTS_HOLD'
    if defect == 'exhausted': facts['acquisition_attempt_count'] = 3
    if defect == 'changed-source': base = replace(base, passages=(base.passages[0] + ' changed',))
    if defect in {'current-allocation', 'unallocated-current-envelope'}:
        current = usage.begin(candidate, base, 'already admitted current',
            source_view=native_assessor.build_lossless_source_view(base.passages, base.source_ids))
        if defect == 'unallocated-current-envelope':
            with sqlite3.connect(service.path) as retained:
                retained.execute('DELETE FROM model_invocation_allocations WHERE invocation_id=?', (current.invocation_id,))
    if defect == 'unresolved':
        with sqlite3.connect(service.path) as retained: retained.execute('DELETE FROM model_invocation_terminals')
    journal.advance(unit.revision_id, stage=stage, facts=facts)
    before = journal.current(unit.revision_id)
    loaded, acquired, providers = [], [], []
    from newsroom.control_plane.native_assessor import AutonomousNativeEvidenceAssessor, NativeAssessmentExecution
    def provider(_request):
        providers.append('current-v23')
        return NativeAssessmentExecution(json.dumps(_empty_reference_result()), {
            'usage_basis': 'PROVIDER_REPORTED', 'input_tokens': 1, 'output_tokens': 1,
            'cached_read_tokens': 0, 'cached_write_tokens': 0, 'reasoning_tokens': 0,
            'context_tokens': 1, 'total_tokens': 2,
        })
    assessor = AutonomousNativeEvidenceAssessor(provider, usage=usage, dispatch_fence=nullcontext)
    def sources(**request):
        assert request['units'] == (unit,)
        # Source construction must follow an authenticated accounted-old failure.
        assert journal.current(unit.revision_id)['stage'] in {'ASSESSMENT_CONTRACT_REVALIDATION', 'ACQUISITION_STARTED'}
        loaded.append(unit.revision_id)
        if defect == 'stop':
            from newsroom.control_plane.veto import VetoError
            raise VetoError('owner stop')
        return (_source(unit),)
    def acquire(_self, **request):
        acquired.append(request['assessment_cached_only'])
        assert request['sources'][0].unit.revision_id == unit.revision_id
        assessor.assess_with_boundary(candidate, base, (), (), before_dispatch=request['before_assessment'],
            cached_only=request['assessment_cached_only'])
        raise NativeEvidenceHold('NO_QUALIFYING_NEW_INFORMATION', unit.source_id)
    monkeypatch.setattr(NativeEvidenceController, 'acquire_and_retain', acquire)
    def restore_current_publisher_output(current_journal, *, proof):
        assert current_journal is journal
        assert all(progress.get('stage') != 'ACKNOWLEDGED' for _,progress in journal.iter_summaries())
    publication = _publication_caller_without_bootstrap(journal=journal,
        runtime=SimpleNamespace(authority=SimpleNamespace(candidate_version=lambda _: candidate,
            sources=object(), objects=object()), ingress=object(),
            publication=SimpleNamespace(restore_current_publisher_output=restore_current_publisher_output),
            policies=object(), proof=object()),
        evidence=object.__new__(NativeEvidenceController), assessment_usage=usage,
        licence=object(), proof=object(), native_evidence_sources=sources,
        ASSESSMENT_CONTRACT_VERSION='newsroom.native-evidence-assessor.v23+consumer.v1',
        now=lambda: native_composition.UtcTimestamp.parse('2026-09-08T12:00:00Z'))
    from newsroom.tests.test_native_pipeline import _open
    pipeline, _journal, pipeline_connection, _units, pipeline_calls, dispositions = _open(tmp_path, monkeypatch)
    pipeline._journal = journal
    pipeline._publish = publication
    dispositions[0] = ()
    try:
        if defect == 'stop':
            from newsroom.control_plane.veto import VetoError
            with pytest.raises(VetoError): pipeline.tick(cycle_id='source-binding-stop')
            assert loaded == [unit.revision_id] and not acquired and not providers
            return
        report = pipeline.tick(cycle_id='source-binding-recovery')
        if defect not in {None, 'changed-source'}:
            assert not loaded and not acquired and not providers
            assert journal.current(unit.revision_id) == before
            return
        if defect == 'changed-source':
            assert loaded == [unit.revision_id] and acquired == [False] and not providers
            assert journal.current(unit.revision_id)['facts']['reason'] == 'ASSESSOR_REVALIDATION_INPUT_CHANGED_HOLD'
            return
        assert report.revision_states == {'EVIDENCE_HOLD': 1}
        pipeline.tick(cycle_id='source-binding-recovery-once')
        assert journal.current(unit.revision_id)['facts']['reason'] == 'NO_QUALIFYING_NEW_INFORMATION'
        assert loaded == [unit.revision_id] and acquired == [False] and providers == ['current-v23']
        settled = journal.current(unit.revision_id)['facts']
        assert settled['acquisition_attempt_count'] == (2 if initial == 'current-source-keyerror' else 1)
        assert settled['assessment_superseded']['contract_version'] == 'newsroom.native-evidence-assessor.v21+consumer.v1'
        with sqlite3.connect(service.path) as retained:
            assert retained.execute('SELECT count(*) FROM model_invocation_allocations').fetchone() == (2,)
            assert retained.execute('SELECT record_json FROM model_invocation_terminals WHERE invocation_id=?',
                (json.loads(old_terminal)['invocation_id'],)).fetchone() == (old_terminal,)
            assert retained.execute('SELECT record_json FROM model_invocation_allocations WHERE invocation_id=?',
                (json.loads(old_allocation)['invocation_id'],)).fetchone() == (old_allocation,)
    finally:
        with sqlite3.connect(service.path) as retained:
            if defect != 'unresolved':
                assert retained.execute('SELECT record_json FROM model_invocation_terminals WHERE invocation_id=?',
                    (json.loads(old_terminal)['invocation_id'],)).fetchone() == (old_terminal,)
            assert retained.execute('SELECT record_json FROM model_invocation_allocations WHERE invocation_id=?',
                (json.loads(old_allocation)['invocation_id'],)).fetchone() == (old_allocation,)
        candidate_connection.close()
        pipeline_connection.close()
        connection.close()


@pytest.mark.parametrize('case', ('older', 'older-after-regressed-story', 'current', 'newer', 'backdated-latest',
    'unknown-bound', 'unknown-latest', 'same-time-conflict', 'other-warning', 'other-issuer', 'wrong-endpoint', 'missing-latest'))
def test_hko_publisher_fence_uses_canonical_latest_not_latest_story(case):
    from types import SimpleNamespace
    from newsroom.authority import UtcTimestamp
    from newsroom.authority.canonical import canonical_json_bytes
    from newsroom.checks import deterministic_uuid4
    from newsroom.control_plane import native_composition as module
    from newsroom.control_plane.native_source_intake import VERSION as INTAKE_VERSION
    from newsroom.graphiti_adapter.identity import content_digest
    from newsroom.increment10.editorial import EditorialHold
    from newsroom.increment9.proving import SOURCE_URLS
    from newsroom.sources import SourceItemId, SourceTime
    from newsroom.tests.authority_helpers import proof

    definition_version = '00000000-0000-4000-8000-000000003102'
    key = 'WRAIN' if case == 'other-warning' else 'WTS'
    selected = '2026-10-03T00:35:00Z' if case == 'older' else '2026-10-03T01:00:00Z' if case == 'older-after-regressed-story' else '2026-10-03T01:50:00Z'
    latest_time = '2026-10-03T02:10:00Z' if case == 'newer' else '2026-10-03T00:35:00Z' if case == 'backdated-latest' else '2026-10-03T01:50:00Z'
    if case in ('newer', 'backdated-latest'): selected = latest_time
    def raw(at, action):
        return canonical_json_bytes({key: dict(code=key, name='雷暴警告', actionCode=action,
            issueTime='2026-10-02T07:55:00Z', updateTime=at)}).decode()
    selected_raw = raw(selected, 'EXTEND' if case in ('older','older-after-regressed-story','same-time-conflict') else 'CANCEL')
    source_id = 'UK-10' if case == 'other-issuer' else 'HK-02'
    package = SimpleNamespace(source_ids=(source_id,), passages=(selected_raw + '\n\nSource-bound facts.',))
    current = SimpleNamespace(source_id=source_id, currency_family='CURRENT_VERSION',
        version_reference='unparseable' if case == 'unknown-bound' else UtcTimestamp.parse(selected).to_text())
    item_id = deterministic_uuid4(SourceItemId, namespace=f'{INTAKE_VERSION}:item', semantic_value=[definition_version,'HK-02',key])
    latest = SimpleNamespace(request=SimpleNamespace(item_id=item_id, definition_version_id=definition_version,
        source_updated_time=SourceTime.unknown() if case == 'unknown-latest' else SourceTime.exact(UtcTimestamp.parse(latest_time)),
        permitted_state_digest=content_digest(headline='雷暴警告',body=raw(latest_time,'CANCEL'),canonical_url=SOURCE_URLS['HK-02']),
        prior_revision_id='prior'))
    previous = SimpleNamespace(request=SimpleNamespace(item_id=item_id,definition_version_id=definition_version,
        source_updated_time=SourceTime.exact(UtcTimestamp.parse('2026-10-03T01:50:00Z' if case == 'backdated-latest' else '2026-10-03T00:35:00Z'))))
    calls=[]
    def latest_revision(selected_id, **kwargs):
        calls.append(('latest',selected_id)); assert selected_id == item_id
        return None if case == 'missing-latest' else latest
    def revision(selected_id, **kwargs):
        calls.append(('prior',selected_id));assert selected_id=='prior';return previous
    sources=SimpleNamespace(latest_revision=latest_revision,revision=revision)
    arguments=dict(sources=sources,definition_version_id=definition_version,locator='https://other.test/api' if case=='wrong-endpoint' else SOURCE_URLS['HK-02'],proof=proof())
    if case in ('older','older-after-regressed-story','backdated-latest','unknown-bound','unknown-latest','same-time-conflict','wrong-endpoint','missing-latest'):
        with pytest.raises(EditorialHold,match='NATIVE_STORY_SOURCE_(SUPERSEDED|ORDER_UNKNOWN)'):
            module._require_hko_current_source(package,current,**arguments)
    else:
        module._require_hko_current_source(package,current,**arguments)
    if case=='other-issuer': assert calls==[]
    assert len(calls)<=2


def _source_binding_caller_without_bootstrap(**bindings):
    """Run the original method bytecode with its actual outer lexical closure."""
    from inspect import unwrap
    from types import CodeType, FunctionType, MethodType

    outer = unwrap(native_composition.open_native_pipeline).__code__
    publication = next(code for code in outer.co_consts
                       if isinstance(code, CodeType) and code.co_name == 'Publication')
    source_binding = next(code for code in publication.co_consts
                          if isinstance(code, CodeType) and code.co_name == 'sources_for')
    # Bind the real journal/licence/proof/runtime cells without opening a runtime.
    # Global names stay global; a conditional outer import must not be hidden
    # by compiling the extracted class alone in module scope.
    def cell(value):
        return (lambda: value).__closure__[0]
    closure = tuple(cell(bindings[name]) for name in source_binding.co_freevars)
    callback = FunctionType(source_binding, {**vars(native_composition), **bindings}, closure=closure)
    return SimpleNamespace(sources_for=MethodType(callback, object()))


@pytest.mark.parametrize('outcome', ('complete', 'source-hold', 'owner-stop'))
@pytest.mark.parametrize('diagnostic_failure', (False, True))
def test_source_binding_cost_diagnostic_preserves_one_call_result_or_failure(
    monkeypatch, outcome, diagnostic_failure,
):
    from newsroom.control_plane.native_evidence import NativeEvidenceHold
    from newsroom.control_plane.veto import VetoError

    revision_id = '00000000-0000-4000-8000-000000000611'
    units, observations = (object(),), object()
    selected_sources, selected_objects = object(), object()
    licence, proof = object(), object()
    result = (object(), object())
    failure = (NativeEvidenceHold('SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE', 'UK-01')
               if outcome == 'source-hold' else VetoError('signed owner stop'))
    loaded, calls, diagnostics = [], [], []

    class SelectedUnits:
        def __getitem__(self, selected):
            loaded.append(selected)
            return units

    def bind_sources(**request):
        calls.append(request)
        if outcome != 'complete':
            raise failure
        return result

    def emit(event, data):
        diagnostics.append((event, data))
        if diagnostic_failure:
            raise OSError('optional diagnostic storage unavailable')

    monkeypatch.setattr(native_composition, 'emit_diagnostic', emit, raising=False)
    wall, cpu = iter((1_000_000, 4_500_000)), iter((2_000_000, 3_250_000))
    monkeypatch.setattr(native_composition, 'perf_counter_ns', lambda: next(wall), raising=False)
    monkeypatch.setattr(native_composition, 'process_time_ns', lambda: next(cpu), raising=False)
    publication = _source_binding_caller_without_bootstrap(
        journal=SimpleNamespace(units=SelectedUnits(), observations=observations),
        runtime=SimpleNamespace(authority=SimpleNamespace(sources=selected_sources, objects=selected_objects)),
        licence=licence, proof=proof, native_evidence_sources=bind_sources,
    )
    if outcome == 'complete':
        assert publication.sources_for(revision_id) is result
    else:
        with pytest.raises(type(failure)) as raised:
            publication.sources_for(revision_id)
        assert raised.value is failure
    assert loaded == [revision_id]
    assert calls == [dict(units=units, sources=selected_sources, objects=selected_objects,
                          licence=licence, proof=proof, observations=observations)]
    assert diagnostics == [('native_source_binding_cost', {
        'revision_id': revision_id, 'wall_ms': 3.5, 'cpu_ms': 1.25,
        'status': 'COMPLETE' if outcome == 'complete' else 'HOLD' if outcome == 'source-hold' else 'FAILED',
        'failure_class': None if outcome == 'complete' else type(failure).__name__,
        'source_count': 2 if outcome == 'complete' else None,
    })]


@pytest.mark.parametrize('clock', ('perf_counter_ns', 'process_time_ns'))
@pytest.mark.parametrize('source_hold', (False, True))
def test_source_binding_cost_clock_failure_preserves_source_boundary(monkeypatch, clock, source_hold):
    from newsroom.control_plane.native_evidence import NativeEvidenceHold

    result, calls = (object(),), []
    failure = NativeEvidenceHold('SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE', 'UK-01')
    def bind_sources(**request):
        calls.append(request)
        if source_hold:
            raise failure
        return result
    def unavailable():
        raise OSError('optional diagnostic clock unavailable')
    monkeypatch.setattr(native_composition, clock, unavailable)
    monkeypatch.setattr(native_composition, 'emit_diagnostic',
        lambda *_: pytest.fail('no diagnostic is emitted without an initial clock sample'))
    publication = _source_binding_caller_without_bootstrap(
        journal=SimpleNamespace(units={'selected': (object(),)}, observations=object()),
        runtime=SimpleNamespace(authority=SimpleNamespace(sources=object(), objects=object())),
        licence=object(), proof=object(), native_evidence_sources=bind_sources,
    )
    if source_hold:
        with pytest.raises(NativeEvidenceHold) as raised:
            publication.sources_for('selected')
        assert raised.value is failure
    else:
        assert publication.sources_for('selected') is result
    assert len(calls) == 1


@pytest.mark.parametrize('changed', (None, 'definition-version', 'locator', 'stop'))
def test_actual_native_composition_rights_accepts_checked_immutable_headers(composed_rights_cohort, changed):
    from newsroom.control_plane.native_retrieval import NativeRetrievalHold
    fixture = composed_rights_cohort
    journal = fixture.pipeline._journal
    journal.land((fixture.unit,))
    reader = fixture.retrieval._unit_headers_for
    assert reader.__self__ is journal.units
    header = reader(fixture.unit.revision_id)[0]
    assert not hasattr(header, 'body') and not hasattr(header.authority, 'records')
    if changed == 'definition-version':
        header = replace(header, authority=replace(header.authority,
            definition_version_id='00000000-0000-4000-8000-000000000123'))
    elif changed == 'locator':
        header = replace(header, source_definition_url='https://wrong.example/source')
    elif changed == 'stop':
        fixture.stop_requested[0] = True
    if changed:
        with pytest.raises(RuntimeError if changed == 'stop' else NativeRetrievalHold):
            with fixture.retrieval._rights_cohort() as rights:
                rights(header)
    else:
        with fixture.retrieval._rights_cohort() as rights:
            assert rights(header) == rights(fixture.unit)


def test_combined_licensing_refresh_keeps_independent_portfolio_rights(composed_rights_cohort):
    fixture = composed_rights_cohort
    before = len(fixture.observations)
    fixture.retrieval._journal.sources(())
    fixture.pipeline._intake._licence.refresh()
    # The GOVUK fixture retains its original licence HOLD; independent HKO
    # observations still refresh through serial governed retention.
    assert len(fixture.observations) >= before
    assert fixture.pipeline._intake._licence.for_source(source_id='HK-02',
        definition_url=SOURCE_URLS['HK-02']).decision == 'PERMITTED'


def test_retained_witness_callback_uses_its_real_module_globals():
    """Do not manufacture canonical helpers in an extracted closure's globals."""
    import ast
    from newsroom.control_plane.native_source_qualification_consumer import current_source_passage
    from newsroom.control_plane.evidence import EvidencePackage
    from newsroom.authority.canonical import digest_bytes as expected_digest
    tree=ast.parse(Path(native_composition.__file__).read_text())
    callback=next(node for node in ast.walk(tree)if isinstance(node,ast.FunctionDef)and node.name=='semantic_witness_disposition_reader')
    observed=[]
    candidate=SimpleNamespace(candidate_id='candidate',governing_manifest=SimpleNamespace(hypothesis_id='hypothesis',
        lead_signal_bindings=(SimpleNamespace(lead_id='lead',signal_id='signal'),)))
    source=SimpleNamespace(unit=SimpleNamespace(source_id='UK-05',canonical_url='https://www.gov.uk/government/news/fixture',
        headline='Exact title',body='Exact body'))
    consumer=SimpleNamespace(read_current_disposition=lambda *args,**kwargs:observed.append((args,kwargs)))
    # These are the callback's actual lexical cells; digest_bytes is not one.
    namespace={**vars(native_composition),'current_source_passage':current_source_passage,
        'qualification_consumer':consumer,'proof':object()}
    exec(compile(ast.Module(body=[callback],type_ignores=[]),native_composition.__file__,'exec'),namespace)
    namespace['semantic_witness_disposition_reader'](candidate,(source,))
    assert len(observed)==1
    args,kwargs=observed[0];package=args[1]
    assert type(package)is EvidencePackage
    assert package.passages==('Exact title\n\nExact body',)
    assert package.observation_digests==(expected_digest(package.passages[0].encode()),)
    assert kwargs['source_passages']==package.passages and kwargs['proof']is namespace['proof']
