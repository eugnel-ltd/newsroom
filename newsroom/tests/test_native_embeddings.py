import io
import json
import sqlite3
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

import pytest

from newsroom.authority.canonical import digest_bytes, digest_canonical
from newsroom.control_plane import native_embeddings as embedding
from newsroom.control_plane.model_usage import (
    InvocationAllocation, InvocationEfficiencyPolicy, InvocationTerminal,
    ModelUsageIntegrityError, ModelUsageService, UsageComponents, UsageStatus,
    WorkEnvelope, WorkloadClass,
)
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.native_progress import NativeRevisionJournal
from newsroom.control_plane.native_qualification import NativeQualificationError, _invocations
from newsroom.control_plane.store import append_ledger, connect
from newsroom.control_plane.veto import VetoError
from newsroom.increment5.native_retrieval import NativeRetrievalHold
from newsroom.tests.test_native_runtime import _args
from newsroom.tests.test_native_graphiti import _native

NOW = datetime(2026, 9, 8, 14, tzinfo=UTC)


def _policy():
    return InvocationEfficiencyPolicy.create(
        policy_id="test-native-embedding", version="v1", workload_class=WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING,
        provider="openrouter", route=embedding.ROUTE, model=embedding.OPENROUTER_EMBEDDING_SLUG,
        reasoning="none", one_turn=True, exact_input=True, skills_enabled=False,
        tools_enabled=False, mcp_enabled=False, prior_message_count=0,
        command_semantic_version=embedding.VERSION, command_flags=("POST=/embeddings",),
        context_manifest_schema_version=embedding.VERSION, disabled_capabilities=("tools",),
        implementation_revision=embedding.implementation_digest(), max_prompt_bytes=20_000,
        max_context_tokens=8_000, max_output_tokens=1, max_total_tokens=8_000,
        prompt_contract_version=embedding.VERSION, output_schema_digest=embedding.SCHEMA_DIGEST,
        allowed_context_identities=(embedding.VERSION,), allowed_config_identities=(embedding.VERSION,),
        hard_estimate_ceiling_tokens=None, evidence_digest=digest_canonical({"test-only": "bounded route"}), qualified=True,
    )


def _response():
    return {"id": "provider-request-1", "object": "list", "model": embedding.OPENROUTER_EMBEDDING_SLUG,
            "data": [{"index": 0, "embedding": [0.25] * 1024}],
            "usage": {"prompt_tokens": 4, "total_tokens": 4, "cost": 0.00001}}


def _embedding_started(connection, *, unit_id, passage_id, cycle_id, revision_id="revision-1"):
    facts = {
        "retrieval_embeddings": {
            unit_id: {
                "state": "STARTED",
                "passage_id": passage_id,
                "attempt_number": 1,
                "cycle_id": cycle_id,
            }
        }
    }
    NativeRevisionJournal(connection).advance(
        revision_id, stage='EMBEDDING_STARTED', facts=facts,
    )
    return facts


def _allocate_unresolved_embedding(engine, service):
    text, passage_id, cycle_id = "Unresolved passage.", "unresolved-passage", "unresolved-cycle"
    request = embedding._request(text)
    envelope = WorkEnvelope.create(
        cycle_id=cycle_id,
        workload_class=WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING,
        admitted_at=NOW,
        admission_decision_id=None,
        candidate_id=None,
        hypothesis_digest=None,
        evidence_package_digest=digest_bytes(text.encode()),
        ingest_id=passage_id,
        graphiti_attempt_id=None,
    )
    service.open_envelope(envelope)
    manifest = engine._manifest(request, text)
    service.retain_context_manifest(manifest)
    allocation = InvocationAllocation.create(
        envelope_id=envelope.envelope_id,
        cycle_id=cycle_id,
        leaf_ordinal=1,
        workload_class=engine._policy.workload_class,
        invocation_policy_digest=engine._policy.canonical_digest,
        provider=engine._policy.provider,
        route=engine._policy.route,
        model=engine._policy.model,
        reasoning="none",
        prompt_contract_version=embedding.VERSION,
        prompt_bytes=len(request),
        prompt_digest=digest_bytes(request),
        request_digest=manifest["request_digest"],
        output_schema_digest=embedding.SCHEMA_DIGEST,
        max_output_tokens=1,
        context_manifest_digest=manifest["context_manifest_digest"],
        context_identity=embedding.VERSION,
        config_identity=embedding.VERSION,
        one_turn=True,
        exact_input=True,
        skills_enabled=False,
        tools_enabled=False,
        mcp_enabled=False,
        prior_message_count=0,
        allocated_at=NOW,
        recovery_deadline_at=NOW + timedelta(seconds=embedding.TIMEOUT + 5),
        parent_invocation_id=None,
    )
    service.allocate(allocation, owner_emergency_stop=False)
    return allocation


@pytest.mark.parametrize("case", [
    "complete", "bare_model", "software_update", "unrelated_model", "bad_vector", "missing_usage",
    "transport_failed", "signed_stop",
])
def test_one_accounted_native_embedding_with_real_sqlite_and_governed_objects(tmp_path, monkeypatch, case):
    args = _args(tmp_path, monkeypatch)
    usage_path = str(tmp_path / "usage.sqlite3")
    service = ModelUsageService(usage_path)
    policy = _policy()
    if case == "software_update":
        policy = InvocationEfficiencyPolicy.create(**{
            **asdict(policy), "implementation_revision": "previous-code",
            "command_semantic_version": "previous-command",
        })
    service.register_policy(policy)
    calls = []
    value = _response()
    if case == "bare_model": value["model"] = "text-embedding-3-large"
    if case == "unrelated_model": value["model"] = "text-embedding-3-small"
    if case == "bad_vector": value["data"][0]["embedding"] = [0.25]
    if case == "missing_usage": value.pop("usage")
    class Response(io.BytesIO):
        status = 200
        def geturl(self): return embedding.URL
    class Opener:
        def open(self, request, timeout):
            calls.append((request.full_url, request.data, timeout))
            if case == "transport_failed": raise OSError("transport unavailable")
            return Response(json.dumps(value).encode())
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    @contextmanager
    def fence():
        if case == "signed_stop": raise VetoError("signed owner stop")
        yield
    with open_native_runtime(**args) as runtime:
        engine = embedding.NativePassageEmbedder(
            api_key="test-key-never-live", objects=runtime.authority.objects, usage=service,
            policy=policy, dispatch_fence=fence, implementation_worktree_clean=True, clock=lambda: NOW,
        )
        params = dict(text="Exact source passage.", passage_id="actual-passage-id", cycle_id="native-cycle-1", proof=runtime.proof)
        if case in {"complete", "bare_model", "software_update"}:
            reference = engine.retain(**params)
            assert reference.vector_admission_id != reference.receipt_admission_id
        elif case == "signed_stop":
            with pytest.raises(VetoError, match="signed owner stop"):
                engine.retain(**params)
        else:
            with pytest.raises(NativeRetrievalHold, match="RESULT_HOLD"):
                engine.retain(**params)
        assert engine.retryable_settled_attempt(
            text=params["text"], passage_id=params["passage_id"],
            cycle_id=params["cycle_id"],
        ) is (case in {"signed_stop", "unrelated_model", "bad_vector"})
    with sqlite3.connect(usage_path) as database:
        assert database.execute("SELECT COUNT(*) FROM model_invocation_allocations").fetchone()[0] == 1
        raw = database.execute("SELECT record_json FROM model_invocation_terminals").fetchone()[0]
        terminal = json.loads(raw)
        if case == "signed_stop":
            assert terminal["pre_dispatch_zero_proved"] is True
            assert terminal["components"]["total_tokens"] == 0
            assert not calls
        else:
            assert len(calls) == 1
            request = json.loads(calls[0][1])
            assert request == {"input": "Exact source passage.", "model": embedding.OPENROUTER_EMBEDDING_SLUG,
                               "dimensions": 1024, "encoding_format": "float"}
            assert "test-key" not in raw
            assert terminal["components"]["total_tokens"] == (
                4 if case in {"complete", "bare_model", "software_update", "unrelated_model", "bad_vector"} else None
            )
            assert terminal["dispatch_at"] is not None
        assert terminal["od_011_reference"] == "OD-011:NATIVE_RETRIEVAL_EMBEDDING"
        assert terminal["policy_breach"] is None
        assert terminal["usage_status"] == ("UNREPORTED" if case in {"missing_usage", "transport_failed"} else "REPORTED")
        telemetry = database.execute(
            "SELECT record_json FROM model_provider_telemetry"
        ).fetchone()
        if case == "bare_model":
            assert json.loads(telemetry[0])["provider_telemetry"]["model"] == "text-embedding-3-large"
        if case == "signed_stop":
            invocation_id = terminal["invocation_id"]
            terminal["invocation_id"] = "corrupt-invocation"
            database.execute(
                "UPDATE model_invocation_terminals SET record_json=?",
                (json.dumps(terminal, sort_keys=True, separators=(",", ":")),),
            )
            database.commit()
    if case == "signed_stop":
        with pytest.raises(ModelUsageIntegrityError):
            service.terminal(invocation_id)


def test_accounted_validation_failure_is_retryable_across_implementation_change(
    tmp_path, monkeypatch,
):
    args = _args(tmp_path, monkeypatch)
    service = ModelUsageService(str(tmp_path / "usage.sqlite3"))
    old_policy = _policy()
    value = _response()
    value["model"] = "text-embedding-3-small"

    class Response(io.BytesIO):
        status = 200

        def geturl(self):
            return embedding.URL

    class Opener:
        def open(self, request, timeout):
            return Response(json.dumps(value).encode())

    monkeypatch.setattr("urllib.request.build_opener", lambda *_: Opener())
    with open_native_runtime(**args) as runtime:
        old = embedding.NativePassageEmbedder(
            api_key="test-key", objects=runtime.authority.objects, usage=service,
            policy=old_policy, dispatch_fence=nullcontext,
            implementation_worktree_clean=True, clock=lambda: NOW,
        )
        with pytest.raises(NativeRetrievalHold, match="RESULT_HOLD"):
            old.retain(
                text="Exact source passage.", passage_id="actual-passage-id",
                cycle_id="native-cycle-1", proof=runtime.proof,
            )
        monkeypatch.setattr(embedding, "implementation_digest", lambda: "new-implementation")
        current = embedding.NativePassageEmbedder(
            api_key="test-key", objects=runtime.authority.objects, usage=service,
            policy=_policy(), dispatch_fence=nullcontext,
            implementation_worktree_clean=True, clock=lambda: NOW,
        )
        assert current.retryable_settled_attempt(
            text="Exact source passage.", passage_id="actual-passage-id",
            cycle_id="native-cycle-1",
        )
        assert not current.retryable_settled_attempt(
            text="Different passage.", passage_id="actual-passage-id",
            cycle_id="native-cycle-1",
        )

        with sqlite3.connect(service.path) as database:
            invocation_id = database.execute(
                "SELECT invocation_id FROM model_invocation_allocations"
            ).fetchone()[0]
        retained = service.terminal(invocation_id)
        monkeypatch.setattr(
            service, "terminal", lambda _invocation_id: replace(
                retained, usage_status=UsageStatus.AMBIGUOUS
            )
        )
        assert not current.retryable_settled_attempt(
            text="Exact source passage.", passage_id="actual-passage-id",
            cycle_id="native-cycle-1",
        )
        monkeypatch.setattr(service, "terminal", lambda _invocation_id: retained)
        with sqlite3.connect(service.path) as database:
            database.execute("DELETE FROM model_provider_telemetry")
        assert not current.retryable_settled_attempt(
            text="Exact source passage.", passage_id="actual-passage-id",
            cycle_id="native-cycle-1",
        )
        monkeypatch.setattr(
            service, "terminal", lambda _invocation_id: replace(
                retained, policy_breach="ACCOUNTING_POLICY_BREACH"
            )
        )
        assert not current.retryable_settled_attempt(
            text="Exact source passage.", passage_id="actual-passage-id",
            cycle_id="native-cycle-1",
        )


@pytest.mark.parametrize("case", ["immediate", "deferred", "other-unresolved"])
def test_post_dispatch_timeout_is_bounded_settled_and_retryable_after_restart(
    tmp_path, monkeypatch, case,
):
    args = _args(tmp_path, monkeypatch)
    usage_path = str(tmp_path / "usage.sqlite3")
    connection = connect(usage_path)
    service = ModelUsageService(usage_path)
    policy = _policy()
    passage_id = "actual-timeout-passage"
    unit = _native("embedding-timeout")
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    unit_id = unit.ingest_id
    cycle_id = f"native-passage:{unit_id}"
    text = "x" * (8_288 - len(embedding._request("")))
    assert len(embedding._request(text)) == 8_288
    facts = _embedding_started(
        connection,
        unit_id=unit_id,
        passage_id=passage_id,
        cycle_id=cycle_id,
        revision_id=unit.revision_id,
    )

    class Opener:
        def open(self, _request, timeout):
            raise TimeoutError("provider response deadline elapsed")

    monkeypatch.setattr("urllib.request.build_opener", lambda *_: Opener())
    with open_native_runtime(**args) as runtime:
        engine = embedding.NativePassageEmbedder(
            api_key="test-key",
            objects=runtime.authority.objects,
            usage=service,
            policy=policy,
            dispatch_fence=nullcontext,
            implementation_worktree_clean=True,
            clock=lambda: NOW,
        )
        unresolved = (
            _allocate_unresolved_embedding(engine, service)
            if case == "other-unresolved"
            else None
        )
        if case == "deferred":
            monkeypatch.setattr(
                service,
                "disposition_native_embedding_timeout",
                lambda **_values: {},
            )
        with pytest.raises(NativeRetrievalHold, match="RESULT_HOLD"):
            engine.retain(
                text=text,
                passage_id=passage_id,
                cycle_id=cycle_id,
                proof=runtime.proof,
            )
        assert connection.execute(
            "SELECT COUNT(*) FROM model_usage_conservative_dispositions"
        ).fetchone() == ((0,) if case == "deferred" else (1,))
        if case == "deferred":
            allocation = json.loads(connection.execute(
                "SELECT record_json FROM model_invocation_allocations"
            ).fetchone()[0])
            terminal = json.loads(connection.execute(
                "SELECT record_json FROM model_invocation_terminals"
            ).fetchone()[0])
            settle = {
                "invocation_id": allocation["invocation_id"],
                "expected_terminal_digest": terminal["terminal_digest"],
                "expected_allocation_digest": allocation["canonical_digest"],
                "expected_request_digest": allocation["request_digest"],
                "expected_passage_id": passage_id,
                "expected_cycle_id": cycle_id,
                "observed_at": NOW,
            }
            transport = connection.execute(
                "SELECT * FROM model_transport_observations"
            ).fetchone()
            connection.execute("DELETE FROM model_transport_observations")
            connection.commit()
            with pytest.raises(ModelUsageIntegrityError, match="dispatch"):
                ModelUsageService(usage_path).disposition_native_embedding_timeout(
                    **settle
                )
            original_transport = json.loads(transport[5])
            original_transport.pop("observation_digest")

            def insert_transport(*, observed_at, evidence_digest):
                record = {
                    **original_transport,
                    "observed_at": observed_at,
                    "evidence_digest": evidence_digest,
                }
                record_digest = digest_canonical(record)
                connection.execute(
                    "INSERT INTO model_transport_observations VALUES(?,?,?,?,?,?)",
                    (
                        record_digest,
                        allocation["invocation_id"],
                        observed_at,
                        "DISPATCH_STARTED",
                        evidence_digest,
                        json.dumps(
                            {**record, "observation_digest": record_digest},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ),
                )
                connection.commit()

            for observed_at, evidence_digest in (
                (transport[2], digest_canonical({"wrong": "request"})),
                ("2026-09-08T14:00:01.000000Z", allocation["request_digest"]),
            ):
                insert_transport(
                    observed_at=observed_at, evidence_digest=evidence_digest
                )
                with pytest.raises(ModelUsageIntegrityError, match="dispatch"):
                    ModelUsageService(usage_path).disposition_native_embedding_timeout(
                        **settle
                    )
                connection.execute("DELETE FROM model_transport_observations")
                connection.commit()
            connection.execute(
                "INSERT INTO model_transport_observations VALUES(?,?,?,?,?,?)",
                tuple(transport),
            )
            insert_transport(
                observed_at="2026-09-08T14:00:01.000000Z",
                evidence_digest=allocation["request_digest"],
            )
            with pytest.raises(ModelUsageIntegrityError, match="dispatch"):
                ModelUsageService(usage_path).disposition_native_embedding_timeout(
                    **settle
                )
            connection.execute(
                "DELETE FROM model_transport_observations WHERE observed_at!=?",
                (transport[2],),
            )
            telemetry = {"provider": "openrouter", "usage": "unknown"}
            telemetry_digest = digest_canonical(telemetry)
            telemetry_record = {
                "invocation_id": allocation["invocation_id"],
                "provider_telemetry_digest": telemetry_digest,
                "provider_telemetry": telemetry,
            }
            connection.execute(
                "INSERT INTO model_provider_telemetry VALUES(?,?,?,?)",
                (
                    digest_canonical(telemetry_record),
                    allocation["invocation_id"],
                    telemetry_digest,
                    json.dumps(
                        telemetry_record, sort_keys=True, separators=(",", ":")
                    ),
                ),
            )
            connection.commit()
            with pytest.raises(ModelUsageIntegrityError, match="telemetry"):
                ModelUsageService(usage_path).disposition_native_embedding_timeout(
                    **settle
                )
            connection.execute("DELETE FROM model_provider_telemetry")
            connection.commit()
        append_ledger(connection, "NATIVE_REVISION_PROGRESS", {
            "revision_id": unit.revision_id,
            "ordinal": 2,
            "stage": "RETRIEVAL_HOLD",
            "facts": {**facts, "reason": "NATIVE_EMBEDDING_RESULT_HOLD"},
        })
        connection.commit()
        restarted = embedding.NativePassageEmbedder(
            api_key="test-key",
            objects=runtime.authority.objects,
            usage=ModelUsageService(usage_path),
            policy=policy,
            dispatch_fence=nullcontext,
            implementation_worktree_clean=True,
            clock=lambda: NOW,
        )
        retryable = restarted.retryable_settled_attempt(
            text=text, passage_id=passage_id, cycle_id=cycle_id
        )
        assert retryable is (case != "other-unresolved")
        if unresolved is not None:
            assert service.route_state(embedding.ROUTE)["state"] == "OPEN"
            with pytest.raises(
                NativeQualificationError, match="invocation is in flight"
            ):
                _invocations(connection, NativeRevisionJournal(connection))
            service.complete(InvocationTerminal.create(
                invocation_id=unresolved.invocation_id,
                outcome="NATIVE_EMBEDDING_FAILED",
                failure_class="LOCAL_PRE_DISPATCH_STOP",
                usage_status=UsageStatus.REPORTED,
                components=UsageComponents(
                    total_tokens=0, provenance="CLI_DERIVED"
                ),
                dispatch_at=None,
                completed_at=NOW,
                observed_at=NOW,
                pre_dispatch_zero_proved=True,
                od_011_reference="OD-011:NATIVE_RETRIEVAL_EMBEDDING",
                subscription_cli_chat_not_cash_debited=False,
            ))
            assert restarted.retryable_settled_attempt(
                text=text, passage_id=passage_id, cycle_id=cycle_id
            )
        assert not restarted.retryable_settled_attempt(
            text=text + "changed", passage_id=passage_id, cycle_id=cycle_id
        )
        assert not restarted.retryable_settled_attempt(
            text=text, passage_id="other-passage", cycle_id=cycle_id
        )

    row = connection.execute(
        "SELECT record_json FROM model_usage_conservative_dispositions"
    ).fetchone()
    disposition = json.loads(row[0])
    assert disposition["components"]["total_tokens"] == 8_288
    assert disposition["qualified_policy_maximum_total_tokens"] == 8_000
    assert disposition["estimated_policy_ceiling_exceeded"] is True
    assert disposition["exact_policy_compliance_unknown"] is True
    assert disposition["cash_spend_known"] is False
    assert disposition["exact_usage_remains_unknown"] is True
    assert disposition["unknown_spend_released"] is False
    assert disposition["source_observation_digest"] == unit.observation_digest
    assert disposition["source_admission_id"] == unit.authority.admission_id
    assert disposition["source_access_decision_id"] == unit.authority.access_decision_id
    assert service.route_state(embedding.ROUTE)["state"] == "CLOSED"
    assert connection.execute(
        "SELECT COUNT(*) FROM model_usage_conservative_dispositions"
    ).fetchone() == (1,)
    allocation = json.loads(connection.execute(
        "SELECT a.record_json FROM model_invocation_allocations a JOIN "
        "model_usage_conservative_dispositions d "
        "ON d.invocation_id=a.invocation_id"
    ).fetchone()[0])
    terminal = json.loads(connection.execute(
        "SELECT record_json FROM model_invocation_terminals"
    ).fetchone()[0])
    assert _invocations(
        connection,
        NativeRevisionJournal(connection),
        (allocation["invocation_id"],),
    ) == (allocation["invocation_id"],)
    for field, changed in (
        ("authority_scope", "wrong-scope"),
        ("passage_id", "wrong-passage"),
        ("source_observation_digest", digest_canonical({"wrong": "source"})),
        ("estimated_policy_ceiling_exceeded", False),
    ):
        altered = {**disposition, field: changed}
        altered.pop("disposition_digest")
        altered_digest = digest_canonical(altered)
        altered["disposition_digest"] = altered_digest
        connection.execute(
            "UPDATE model_usage_conservative_dispositions SET "
            "disposition_digest=?,record_json=? WHERE invocation_id=?",
            (
                altered_digest,
                json.dumps(altered, sort_keys=True, separators=(",", ":")),
                allocation["invocation_id"],
            ),
        )
        connection.commit()
        with pytest.raises(NativeQualificationError):
            _invocations(
                connection,
                NativeRevisionJournal(connection),
                (allocation["invocation_id"],),
            )
        connection.execute(
            "UPDATE model_usage_conservative_dispositions SET "
            "disposition_digest=?,record_json=? WHERE invocation_id=?",
            (
                disposition["disposition_digest"],
                json.dumps(disposition, sort_keys=True, separators=(",", ":")),
                allocation["invocation_id"],
            ),
        )
        connection.commit()
    base = {
        "invocation_id": allocation["invocation_id"],
        "expected_terminal_digest": terminal["terminal_digest"],
        "expected_allocation_digest": allocation["canonical_digest"],
        "expected_request_digest": allocation["request_digest"],
        "expected_passage_id": passage_id,
        "expected_cycle_id": cycle_id,
        "observed_at": NOW,
    }
    for table, column, changed, original in (
        ("model_invocation_allocations", "leaf_ordinal", 7, 1),
        (
            "model_invocation_terminals",
            "outcome",
            "NATIVE_EMBEDDING_COMPLETE",
            "NATIVE_EMBEDDING_FAILED",
        ),
    ):
        connection.execute(
            f"UPDATE {table} SET {column}=? WHERE invocation_id=?",
            (changed, allocation["invocation_id"]),
        )
        connection.commit()
        with pytest.raises(ModelUsageIntegrityError, match="invocation binding"):
            ModelUsageService(usage_path).disposition_native_embedding_timeout(**base)
        with pytest.raises(NativeQualificationError):
            _invocations(
                connection,
                NativeRevisionJournal(connection),
                (allocation["invocation_id"],),
            )
        connection.execute(
            f"UPDATE {table} SET {column}=? WHERE invocation_id=?",
            (original, allocation["invocation_id"]),
        )
        connection.commit()
    policy_record = json.loads(connection.execute(
        "SELECT record_json FROM model_invocation_policies "
        "WHERE canonical_digest=?",
        (allocation["invocation_policy_digest"],),
    ).fetchone()[0])
    canonical_policy = json.dumps(
        policy_record, sort_keys=True, separators=(",", ":")
    )
    for changed_policy in (
        json.dumps(
            {**policy_record, "max_total_tokens": 9_000},
            sort_keys=True,
            separators=(",", ":"),
        ),
        json.dumps(
            {**policy_record, "qualified": 1},
            sort_keys=True,
            separators=(",", ":"),
        ),
        json.dumps(policy_record, sort_keys=True, indent=1),
    ):
        connection.execute(
            "UPDATE model_invocation_policies SET record_json=? "
            "WHERE canonical_digest=?",
            (changed_policy, allocation["invocation_policy_digest"]),
        )
        connection.commit()
        with pytest.raises(ModelUsageIntegrityError, match="policy binding"):
            ModelUsageService(usage_path).disposition_native_embedding_timeout(**base)
        with pytest.raises(NativeQualificationError):
            _invocations(
                connection,
                NativeRevisionJournal(connection),
                (allocation["invocation_id"],),
            )
    connection.execute(
        "UPDATE model_invocation_policies SET record_json=? "
        "WHERE canonical_digest=?",
        (canonical_policy, allocation["invocation_policy_digest"]),
    )
    connection.commit()
    for changed in (
        {"expected_request_digest": digest_canonical({"wrong": "request"})},
        {"expected_passage_id": "wrong-passage"},
        {"expected_cycle_id": "native-passage:wrong-unit"},
    ):
        with pytest.raises(ModelUsageIntegrityError):
            ModelUsageService(usage_path).disposition_native_embedding_timeout(
                **{**base, **changed}
            )
    connection.close()


def test_native_embedding_requires_qualified_output_contract_before_effects(tmp_path, monkeypatch):
    from dataclasses import asdict, replace
    args = _args(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        with pytest.raises(NativeRetrievalHold, match="POLICY_HOLD"):
            embedding.NativePassageEmbedder(api_key="test-key", objects=runtime.authority.objects,
                usage=ModelUsageService(str(tmp_path / "usage.sqlite3")),
                policy=replace(_policy(), output_schema_digest=digest_canonical({"different": "contract"})),
                dispatch_fence=nullcontext, implementation_worktree_clean=True)


def test_native_embedding_policy_resolution_uses_contract_not_software_revision(tmp_path):
    from dataclasses import asdict
    from newsroom.control_plane.model_usage import ModelUsageAdmissionError, InvocationEfficiencyPolicy
    service = ModelUsageService(str(tmp_path / "usage.sqlite3"))
    policy = _policy()
    old = InvocationEfficiencyPolicy.create(**{**asdict(policy), "implementation_revision": "retired-implementation", "version": "retired"})
    service.register_policy(old)
    service.register_policy(policy)
    query = dict(workload_class=policy.workload_class, provider=policy.provider,
                 route=policy.route, model=policy.model, reasoning=policy.reasoning,
                 output_schema_digest=policy.output_schema_digest)
    assert service.qualified_policy(**query, implementation_revision=policy.implementation_revision) == policy
    assert service.qualified_policy(**query) == policy
    assert service.qualified_policy(**query, implementation_revision="new-code") == policy
    with pytest.raises(ModelUsageAdmissionError):
        service.qualified_policy(**{**query, "output_schema_digest": digest_canonical({"wrong": "schema"})})
