"""Native inherited-negative receipts never impersonate a fresh provider call."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from newsroom.authority.canonical import digest_canonical
from newsroom.graphiti_adapter.combined_temporal_contract import (
    SourceRevisionInput, build_compact_prompt, SCHEMA,
)
from newsroom.graphiti_adapter.identity import configuration_digest


def test_current_empty_receipt_has_distinct_identity_and_no_fresh_model_call():
    from newsroom.graphiti_adapter.combined_temporal_extraction import inherited_empty_receipt
    old = SourceRevisionInput(
        body='Row 42: A="2018-02-10"; B=n:0', revision_id="old-revision",
        source_id="definition", item_key="old-item", representation_digest="old-representation",
        published_at="2023-02-01T00:00:00Z", updated_at="2026-10-02T14:13:45Z",
        observed_at="2026-10-02T14:20:12Z", ingested_at="2026-10-02T14:20:12Z",
        episode_uuid="old-episode",
    )
    current = replace(old, revision_id="current-revision", item_key="current-item",
        representation_digest="current-representation", updated_at="2026-10-03T13:05:34Z",
        observed_at="2026-10-03T13:18:46Z", ingested_at="2026-10-03T13:18:46Z",
        episode_uuid="current-episode")
    prompt = build_compact_prompt(old)
    retained = {
        "prompt_digest": digest_canonical({"prompt": prompt.text, "schema": SCHEMA}),
        "configuration_digest": configuration_digest(),
        "proposal_receipt": {"wire_payload": {"entities": [], "facts": []}},
        "zero_proposal_effect": "EXPLICIT",
    }
    source_refs = {"run_version_id": "old-run", "output_id": "old-output",
        "output_digest": "sha256:" + "a" * 64, "attempt_id": "old-attempt"}
    receipt = inherited_empty_receipt(current, retained, source_refs)
    assert receipt is not None
    assert receipt["execution_source"] == "INHERITED_NO_PROPOSALS"
    assert receipt["source_revision_id"] == "current-revision"
    assert receipt["invocation_count"] == 0
    assert receipt["transport_calls"] == []
    assert receipt["inherited_source"] == source_refs
    assert receipt["proposal_receipt"]["wire_payload"] == {"entities": [], "facts": []}
    assert retained["proposal_receipt"] == {"wire_payload": {"entities": [], "facts": []}}


def test_native_empty_callback_returns_before_sdk_or_provider_setup(tmp_path, monkeypatch):
    from newsroom.authority import UtcTimestamp
    from newsroom.graphiti_adapter import real
    from newsroom.graphiti_adapter.temporal import SOURCE_UPDATED
    from newsroom.graphiti_adapter.result_mapping import episode_body
    from newsroom.tests.test_graphiti_adapter_real_executor import _real_attempt
    attempt = replace(_real_attempt(tmp_path),
        reference_time=UtcTimestamp.parse("2026-10-03T13:05:34Z"),
        temporal_basis=SOURCE_UPDATED, episode_uuid="current-episode")
    revision = real._source_revision_input(attempt,
        body=episode_body(attempt), ingested_at=UtcTimestamp.parse("2026-10-03T13:18:46Z"))
    prompt = build_compact_prompt(revision)
    retained = {"prompt_digest": digest_canonical({"prompt": prompt.text, "schema": SCHEMA}),
        "configuration_digest": configuration_digest(), "zero_proposal_effect": "EXPLICIT",
        "proposal_receipt": {"wire_payload": {"entities": [], "facts": []}}}
    refs = {"run_version_id": "old-run", "output_id": "old-output",
        "output_digest": "sha256:" + "a" * 64, "attempt_id": "old-attempt"}
    observer = SimpleNamespace(inherited_empty_for=lambda actual: (retained, refs))
    monkeypatch.setattr(real, "_load_graphiti", lambda: pytest.fail("inherited empty loaded SDK"))
    monkeypatch.setattr(real, "openrouter_api_key", lambda: pytest.fail("inherited empty loaded provider credentials"))
    execution = real.RealGraphitiAdapter(invocation_observer=observer).execute(
        attempt=attempt, workspace_root=tmp_path / "workspace")
    assert execution.outcome.value == "COMPLETE"
    assert execution.produced.proposals == ()
    raw = execution.produced.raw_output_value
    assert raw["execution_source"] == "INHERITED_NO_PROPOSALS"
    assert raw["combined_temporal_receipt"]["inherited_source"] == refs
    assert raw["chat_invocation_count"] == 0
    assert raw["episode_uuid"] == "current-episode"


def _reader_fixture():
    import json
    import sqlite3
    from newsroom.authority.canonical import digest_bytes
    from newsroom.control_plane.model_usage import GraphitiIngestRetryEvidence
    from newsroom.control_plane.native_graphiti import NativeGraphitiProcessor
    from newsroom.extraction.types import ExtractionOutcome, ExtractionOutputValidation
    from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome
    from newsroom.tests.test_native_graphiti import _native
    unit = _native("inherited-empty")
    text = " ".join(unit.episode_body.split())
    passage = SimpleNamespace(text_digest=digest_bytes(text.encode()))
    current = SimpleNamespace(manifest=SimpleNamespace(revision_id=unit.revision_id,
        item_id=unit.authority.item_id, definition_version_id=unit.authority.definition_version_id,
        definition_id=unit.authority.definition_id, configuration_digest="config",
        extractor_contract_digest="contract"),
        extraction_request=SimpleNamespace(input_binding=SimpleNamespace(passages=(passage,))))
    old_manifest = SimpleNamespace(**vars(current.manifest), passages=(passage,))
    old = SimpleNamespace(outcome=GraphitiAdapterOutcome.COMPLETE, output_id="old-output",
        proposal_set_id=None, run_version_id="old-run", attempt_id="old-attempt", attempt_number=1)
    metadata = SimpleNamespace(outcome=ExtractionOutcome.SUCCESS, proposal_count=0,
        output=SimpleNamespace(validation=ExtractionOutputValidation.VALID))
    raw = {"entities": [], "relations": [], "proposals": [],
        "token_usage": {"unreported_chat_requests": 0},
        "combined_temporal_receipt": {"proposal_receipt": {"wire_payload": {"entities": [], "facts": []}}}}
    output = SimpleNamespace(canonical_bytes=json.dumps(raw).encode(),
        view=SimpleNamespace(canonical_digest="sha256:" + "a" * 64))
    empty = GraphitiIngestRetryEvidence((), (), (), None, ())
    settled = GraphitiIngestRetryEvidence((1,), (), (1,), 1, ())
    evidence = {unit.ingest_id: empty, "old-ingest": settled}
    connection = sqlite3.connect(":memory:")
    connection.executescript("CREATE TABLE unpublished_graphiti_ingest(ingest_id TEXT,source_id TEXT,outcome TEXT,at TEXT); CREATE TABLE unpublished_graphiti_receipts(ingest_id TEXT,receipt_json TEXT); CREATE TABLE model_work_envelopes(envelope_id TEXT,record_json TEXT); CREATE TABLE model_invocation_allocations(invocation_id TEXT,envelope_id TEXT); CREATE TABLE model_invocation_terminals(invocation_id TEXT,usage_status TEXT);")
    connection.execute("INSERT INTO unpublished_graphiti_ingest VALUES(?,?,?,?)", ("old-ingest", unit.source_id, "COMPLETE", "old"))
    connection.execute("INSERT INTO unpublished_graphiti_receipts VALUES(?,?)", ("old-ingest", json.dumps({"proposal_count":0,"passages":[{"text_digest":passage.text_digest}]})))
    reader = object.__new__(NativeGraphitiProcessor)
    reader._connection = connection
    reader._stop_check = lambda: None
    reader._proof = object()
    reader._usage = SimpleNamespace(
        graphiti_ingest_retry_evidence=lambda *,ingest_id,**_: evidence[ingest_id],
        graphiti_ingest_allocation_count=lambda *,ingest_id: len(evidence[ingest_id].attempt_numbers),
    )
    reader._system = SimpleNamespace(
        graphiti=SimpleNamespace(attempt_history=lambda *a,**k:(old,),manifest_for_attempt=lambda *a,**k:old_manifest),
        extraction=SimpleNamespace(metadata=lambda *a,**k:metadata,raw_output=lambda *a,**k:output),
        sources=SimpleNamespace(item=lambda *a,**k:SimpleNamespace(request=SimpleNamespace(source_native_id="observed|https://publisher.invalid/asset"))),
    )
    return reader,unit,current,old_manifest,metadata,raw,output,evidence


def test_native_reader_authenticates_selected_old_output_without_a_new_leaf():
    reader,unit,current,*_ = _reader_fixture()
    try:
        value = reader._inherited_empty_for(unit,current)
        assert value is not None
        assert value[1]["output_id"] == "old-output"
        assert value[1]["attempt_id"] == "old-attempt"
    finally:
        reader._connection.close()


@pytest.mark.parametrize("current_prior", ("ABSENT", "AMBIGUOUS_EFFECT", "IN_FLIGHT_WORKSPACE"))
def test_governed_native_inheritance_retains_current_receipt_and_real_accounting_on_reopen(tmp_path, monkeypatch, current_prior):
    """Real Source/CAS/attempt/output/usage boundaries; only old transport is fake."""
    import json
    from contextlib import nullcontext
    from datetime import datetime
    from newsroom.control_plane.native_graphiti import NativeGraphitiProcessor
    from newsroom.control_plane.model_usage import ModelUsageService
    from newsroom.control_plane.graphiti_admission import GraphitiAdmissionConsumerError
    from newsroom.control_plane.store import connect
    from newsroom.graphiti_adapter import real
    from newsroom.graphiti_adapter.combined_temporal_contract import CONTRACT_NAME
    from newsroom.graphiti_adapter.combined_temporal_extraction import inherited_empty_receipt
    from newsroom.extraction.types import ExtractionRunId
    from newsroom.graphiti_adapter.identity import typed_id
    from newsroom.tests.test_graphiti_operational_readiness import (
        _unit, _plan, _rights, _NOW, _PROOF, _open_operational_test_system,
        bootstrap_operational_authority,
    )
    from newsroom.control_plane import native_graphiti
    from newsroom.tests import test_graphiti_operational_readiness as readiness
    from newsroom.control_plane import graphiti_operational_readiness as bootstrap
    source_requests = bootstrap._source_requests
    def asset_source_requests(unit, rights, **kwargs):
        definition, version, item, revision, representation = source_requests(unit, rights, **kwargs)
        return definition, version, replace(item, source_native_id=unit.item_key), revision, representation
    monkeypatch.setattr(bootstrap, "_source_requests", asset_source_requests)
    # Fake transport models a qualified deployed tree, not the dirty test checkout.
    monkeypatch.setattr("newsroom.control_plane.graphiti._graphiti_implementation_identity",
        lambda: ("a" * 40, True))
    graph_policy = readiness.graphiti_read_policy()
    extract_policy = readiness.extraction_read_policy()
    source_policy = readiness.source_read_policy()
    principal = frozenset({"newsroom.control-plane"})
    monkeypatch.setattr(readiness, "graphiti_read_policy", lambda: replace(graph_policy,
        allowed_principal_ids=principal, attempt_required_scope="authority.graphiti.attempt.read",
        configuration_required_scope="authority.graphiti.configuration.read",
        replay_required_scope="authority.graphiti.replay.read"))
    monkeypatch.setattr(readiness, "extraction_read_policy", lambda: replace(extract_policy,
        allowed_principal_ids=principal, metadata_required_scope="authority.extraction.metadata.read",
        proposal_required_scope="authority.extraction.proposal.read", raw_output_required_scope="authority.extraction.raw.read"))
    monkeypatch.setattr(readiness, "source_read_policy", lambda: replace(source_policy, allowed_principal_ids=principal))

    class EmptyRecoveryDriver:
        async def execute_query(self, query, **kwargs):
            assert "RETURN m.episode_uuid AS marker_episode_uuid" in query
            return [], None, None
        async def close(self):
            pass
    monkeypatch.setattr(real, "_open_compensation_driver", EmptyRecoveryDriver)

    # Projection is a separate boundary: deliberately hold after the real extraction.
    def admission_hold(**_):
        raise GraphitiAdmissionConsumerError("fixture projection not requested")
    monkeypatch.setattr(native_graphiti, "compose_existing_graphiti_admission_consumer",
        lambda *a, **k: SimpleNamespace(enqueue_complete_receipts=admission_hold))
    connection = connect(str(tmp_path / "private.sqlite3"))
    usage = ModelUsageService(str(tmp_path / "private.sqlite3"))
    (tmp_path / "authority").mkdir(mode=0o700)
    system = _open_operational_test_system(tmp_path / "authority")
    clock = lambda: datetime.fromisoformat(_NOW.replace("Z", "+00:00"))

    def bind(key):
        unbound = _unit(item_key=key + "|https://example.test/retained.csv")
        _, binder = bootstrap_operational_authority(system, proof=_PROOF, plan=_plan(unbound))
        bound = binder(unbound)
        return replace(bound, proving_run_id="native-source:" + bound.observation_digest)

    def processor():
        return NativeGraphitiProcessor(system=system, connection=connection, usage=usage,
            proof=_PROOF, rights_for=lambda _: _rights(), stop_check=lambda: None,
            dispatch_fence=nullcontext, clock=clock)

    calls = []
    async def old_empty_transport(**values):
        calls.append(values["episode_id"])
        revision = values["revision"]
        prompt = build_compact_prompt(revision)
        observer = values["invocation_observer"]
        token = observer.before_cli_invocation(provider="cursor-agent-cli", model="composer-2.5",
            prompt=prompt.text, schema=json.dumps(SCHEMA), semantic_request_class=CONTRACT_NAME,
            max_tokens=1024)
        observer.transport_dispatch_started(token)
        usage_value = {"usage_basis": "PROVIDER_REPORTED", "input_tokens": 100,
            "output_tokens": 20, "total_tokens": 120, "cached_read_tokens": 0,
            "cached_write_tokens": 0, "reasoning_tokens": 0, "context_tokens": 100}
        refs = observer.after_cli_invocation(token, outcome="COMPLETE", usage=usage_value)
        values["telemetry"].chat_invocations = [{**refs, "provider":"cursor-agent-cli",
            "model":"composer-2.5", "outcome":"COMPLETE", "usage":usage_value}]
        combined = inherited_empty_receipt(revision, {
            "prompt_digest": digest_canonical({"prompt":prompt.text,"schema":SCHEMA}),
            "configuration_digest": configuration_digest(), "zero_proposal_effect":"EXPLICIT",
            "proposal_receipt":{"wire_payload":{"entities":[],"facts":[]}},
        }, {"run_version_id":"fixture", "output_id":"fixture", "output_digest":"fixture", "attempt_id":"fixture"})
        combined.pop("inherited_source")
        combined.pop("execution_source")
        combined["invocation_count"] = 1
        result = SimpleNamespace(episode=SimpleNamespace(uuid=values["episode_id"]),nodes=(),edges=())
        values["validate_result"](result,values["telemetry"],combined)
        return result

    monkeypatch.setattr(real,"_load_graphiti",lambda:SimpleNamespace())
    monkeypatch.setattr(real,"openrouter_api_key",lambda:"fixture")
    monkeypatch.setattr(real,"neo4j_community_password",lambda:"fixture")
    monkeypatch.setattr(real,"_add_episode",old_empty_transport)
    try:
        old = bind("old-observation")
        first = processor().advance((old,),cycle_id="old-native")
        assert first[0].state == "ADMISSION_HOLD", first
        assert connection.execute("SELECT outcome FROM unpublished_graphiti_ingest WHERE ingest_id=?",(old.ingest_id,)).fetchone() == ("COMPLETE",)
        old_head = system.graphiti.attempt_history(typed_id(ExtractionRunId,"run",old.ingest_id),limit=1,proof=_PROOF)[0]
        old_raw = system.extraction.raw_output(old_head.output_id,proof=_PROOF).canonical_bytes
        old_usage = connection.execute("SELECT record_json FROM model_invocation_terminals").fetchall()
        assert len(old_usage) == 1 and json.loads(old_usage[0][0])["usage_status"] == "REPORTED"
        current = bind("new-observation")
        assert old.revision_id != current.revision_id and old.ingest_id != current.ingest_id
        if current_prior == "AMBIGUOUS_EFFECT":
            # Authentic terminal before any allocation: its own mutation effects
            # remain ambiguous, even though a matching old empty result exists.
            pending_processor = processor()
            pending_processor._runner._inherited_empty_for = None
            async def ambiguous_before_allocation(**_):
                raise real.AmbiguousEpisodeEffect("fixture unresolved SDK mutation")
            monkeypatch.setattr(real, "_add_episode", ambiguous_before_allocation)
            pending_processor.advance((current,), cycle_id="current-ambiguous")
            pending_head = system.graphiti.attempt_history(typed_id(ExtractionRunId,"run",current.ingest_id),limit=1,proof=_PROOF)[0]
            assert pending_head.outcome.value == "AMBIGUOUS_EFFECT"
            assert usage.graphiti_ingest_allocation_count(ingest_id=current.ingest_id) == 0
        elif current_prior == "IN_FLIGHT_WORKSPACE":
            # Model the actual retained preallocation workspace of a process
            # killed before authority commit, not a invented journal status.
            pending_attempt = processor()._runner._attempt_for_unit(current)
            namespace = pending_attempt.configuration.workspace_policy.namespace_prefix + "-" + str(pending_attempt.workspace_id)
            (tmp_path / "authority" / namespace).mkdir(mode=0o700)
            assert usage.graphiti_ingest_allocation_count(ingest_id=current.ingest_id) == 0
        monkeypatch.setattr(real,"_load_graphiti",lambda:pytest.fail("inherited current loaded SDK"))
        monkeypatch.setattr(real,"openrouter_api_key",lambda:pytest.fail("inherited current loaded credentials"))
        monkeypatch.setattr(real,"_add_episode",lambda **_:pytest.fail("inherited current called provider"))
        current_processor = processor()
        if current_prior != "ABSENT":
            current_processor._runner._inherited_empty_for = lambda *a: pytest.fail("preallocation guard reached inheritance")
        second = current_processor.advance((current,),cycle_id="current-native")
        if current_prior != "ABSENT":
            assert second[0].state == "GRAPHITI_HOLD"
            assert connection.execute("SELECT outcome FROM unpublished_graphiti_ingest WHERE ingest_id=?",(current.ingest_id,)).fetchone() is None
            assert usage.graphiti_ingest_allocation_count(ingest_id=current.ingest_id) == 0
            assert connection.execute("SELECT record_json FROM model_invocation_terminals").fetchall() == old_usage
            assert system.extraction.raw_output(old_head.output_id,proof=_PROOF).canonical_bytes == old_raw
            if current_prior == "AMBIGUOUS_EFFECT":
                assert second[0].reason == "AMBIGUOUS_EFFECT:AMBIGUOUS_EFFECT"
            else:
                assert (tmp_path / "authority" / namespace).is_dir()
            return
        assert second[0].state == "ADMISSION_HOLD", second
        head = system.graphiti.attempt_history(typed_id(ExtractionRunId,"run",current.ingest_id),limit=1,proof=_PROOF)[0]
        retained = system.extraction.raw_output(head.output_id,proof=_PROOF)
        raw = json.loads(retained.canonical_bytes)
        assert raw["execution_source"] == "INHERITED_NO_PROPOSALS"
        assert raw["combined_temporal_receipt"]["inherited_source"]["output_id"] == str(old_head.output_id)
        assert raw["combined_temporal_receipt"]["source_revision_id"] == current.revision_id
        assert raw["chat_invocation_count"] == 0 and raw["proposal_count"] == 0
        assert connection.execute("SELECT record_json FROM model_invocation_terminals").fetchall() == old_usage
        assert usage.graphiti_ingest_allocation_count(ingest_id=current.ingest_id) == 0
        current_outcome = connection.execute("SELECT o.outcome FROM model_work_outcomes o JOIN model_work_envelopes e USING(envelope_id) WHERE json_extract(e.record_json,'$.ingest_id')=?",(current.ingest_id,)).fetchall()
        assert current_outcome == [("GRAPHITI_SUCCESS_ZERO_PROPOSALS",)]
        assert calls == [old.ingest_id]
        before = connection.execute("SELECT count(*) FROM unpublished_graphiti_attempt_receipts").fetchone()
        system.close()
        system = _open_operational_test_system(tmp_path / "authority")
        usage = ModelUsageService(str(tmp_path / "private.sqlite3"))
        processor().advance((current,),cycle_id="current-reopen")
        assert system.extraction.raw_output(head.output_id,proof=_PROOF).canonical_bytes == retained.canonical_bytes
        assert system.extraction.raw_output(old_head.output_id,proof=_PROOF).canonical_bytes == old_raw
        assert connection.execute("SELECT count(*) FROM unpublished_graphiti_attempt_receipts").fetchone() == before
        assert connection.execute("SELECT record_json FROM model_invocation_terminals").fetchall() == old_usage
    finally:
        system.close()
        connection.close()


def test_canonical_controller_envelope_without_leaf_is_not_a_provider_allocation(tmp_path):
    from newsroom.control_plane.model_usage import WorkEnvelope, WorkloadClass
    from newsroom.tests.test_graphiti_internal_requests import _service_fixture, T0
    from newsroom.tests.test_native_retry_evidence import _attempt
    from newsroom.tests.test_native_graphiti import _native
    service, _unused_envelope, policy, shape = _service_fixture(tmp_path)
    unit = _native("empty-envelope")
    envelope = WorkEnvelope.create(cycle_id="empty-envelope", workload_class=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        admitted_at=T0, admission_decision_id=None, candidate_id=None, hypothesis_digest=None,
        evidence_package_digest=None, ingest_id=unit.ingest_id, graphiti_attempt_id=f"{unit.ingest_id}:1")
    service.open_envelope(envelope)
    assert service.graphiti_ingest_retry_evidence(ingest_id=unit.ingest_id).unresolved_attempts == (1,)
    assert service.graphiti_ingest_allocation_count(ingest_id=unit.ingest_id) == 0
    # An actual allocated leaf remains denied even before dispatch/terminal.
    other = _native("active-envelope")
    _attempt(service, policy, shape, other, 1)
    assert service.graphiti_ingest_allocation_count(ingest_id=other.ingest_id) == 1


@pytest.mark.parametrize("mutation", ("current_active", "old_unknown", "old_estimated", "wrong_definition",
    "changed_body", "nonempty", "missing_output", "invalid_output", "rights_revoked", "stopped"))
def test_native_reader_denies_unknown_or_unqualified_inheritance(mutation):
    import json
    from newsroom.control_plane.model_usage import GraphitiIngestRetryEvidence
    from newsroom.extraction.types import ExtractionOutputValidation
    reader,unit,current,manifest,metadata,raw,output,evidence = _reader_fixture()
    try:
        if mutation == "current_active": evidence[unit.ingest_id] = GraphitiIngestRetryEvidence((1,),(),(),None,(1,))
        elif mutation == "old_unknown": evidence["old-ingest"] = GraphitiIngestRetryEvidence((1,),(),(),None,(1,))
        elif mutation == "old_estimated":
            reader._connection.execute("INSERT INTO model_work_envelopes VALUES(?,?)",("old-envelope",json.dumps({"ingest_id":"old-ingest"})))
            reader._connection.execute("INSERT INTO model_invocation_allocations VALUES(?,?)",("old-leaf","old-envelope"))
            reader._connection.execute("INSERT INTO model_invocation_terminals VALUES(?,?)",("old-leaf","ESTIMATED"))
        elif mutation == "wrong_definition": manifest.definition_version_id = "other-definition"
        elif mutation == "changed_body": current.extraction_request.input_binding.passages[0].text_digest = "sha256:" + "b"*64
        elif mutation == "nonempty": raw["entities"] = [{"name":"new entity"}]; output.canonical_bytes=json.dumps(raw).encode()
        elif mutation == "missing_output": metadata.output = None
        elif mutation == "invalid_output": metadata.output.validation = ExtractionOutputValidation.INVALID
        elif mutation == "rights_revoked": reader._system.extraction.raw_output = lambda *a,**k: (_ for _ in ()).throw(PermissionError("revoked"))
        elif mutation == "stopped": reader._stop_check = lambda: (_ for _ in ()).throw(RuntimeError("stopped"))
        if mutation in {"rights_revoked","stopped"}:
            with pytest.raises(PermissionError if mutation=="rights_revoked" else RuntimeError):
                reader._inherited_empty_for(unit,current)
        else:
            assert reader._inherited_empty_for(unit,current) is None
    finally:
        reader._connection.close()
