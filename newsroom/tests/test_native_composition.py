from __future__ import annotations

import sqlite3
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
    assert native_assessor.VERSION == "newsroom.native-evidence-assessor.v20"
    assert native_composition.ASSESSMENT_CONTRACT_VERSION == (
        "newsroom.native-evidence-assessor.v20+newsroom.named-entity.v15+"
        "newsroom.zh-hant-hk-shape.v14+newsroom.factual-localisation.v2+"
        "newsroom.qualification-relation.v3+newsroom.retained-assessment.v1"
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
def test_deployed_continuous_runtime_qualifies_without_prior_qualification_open(
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
    monkeypatch.setattr(
        native_composition,
        "ModelUsageService",
        lambda _: SimpleNamespace(
            qualified_policy=lambda **request: policies[request["workload_class"]]
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
        assert len(qualified) == 1
        assert qualified[0].startswith("sha256:")
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

    def retain_licence(*, objects, proof, dispatch_fence, clock):
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
            assert not pipeline._publish.copy_correction_due({"writer_id": "newsroom.offline-exact-copy.v3"})
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
                        pipeline._journal.progress[unit.revision_id]["ordinal"]
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
                    revision_id: pipeline._journal.progress[revision_id]["ordinal"]
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
                assert recovery_sources == [{}, {}, {}]

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
    assert stops == ["checked"] * 5  # Three opens and two bounded ACK turns.
    assert fences == ["entered"] * 8


def test_rights_refresh_registers_and_binds_a_newly_permitted_source(
    tmp_path, monkeypatch,
) -> None:
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
