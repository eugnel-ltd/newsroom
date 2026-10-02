from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from newsroom.authority.auth import StaticAuthorizer
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority.evaluation_feedback_system import (
    open_evaluation_feedback_authority_system,
)
from newsroom.increment6.feedback import (
    EvaluationFeedbackOutcome,
    EvaluationFeedbackReason,
    FeedbackContractError,
    HandoffAcceptanceSnapshot,
    ReconciliationDispositionOutcome,
    ReconciliationDispositionReason,
    append_reconciliation_disposition,
    create_evaluation_feedback,
    create_reconciliation_obligation,
)
from newsroom.increment6.handoffs import (
    Acknowledgement,
    AcknowledgementOutcome,
    EvaluationHandoffStore,
    create_handoff,
)
from newsroom.tests import test_increment6e2_candidate_store as candidate_fixture


def _seed(tmp_path: Path, *, candidate_seed_snapshot=None):
    if candidate_seed_snapshot is None:
        adapter = candidate_fixture._Adapter(tmp_path)
    else:
        adapter = candidate_fixture._Adapter.__new__(candidate_fixture._Adapter)
        adapter.root = tmp_path
        adapter.snapshot = candidate_seed_snapshot
    location = adapter.create_location()
    handle = adapter.open_handle(location)
    handle.submit(candidate_fixture._generic("record-1"))
    row = handle._row("record-1")
    version = handle._opened().load_version(str(row[1]))
    handle.close()
    args = candidate_fixture._collaborators(location.seed)
    feedback, obligation = _feedback_for_version(
        location,
        args,
        version,
        suffix="v25-test",
        response_character="4",
        feedback_request_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
        obligation_request_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    )
    return location, args, feedback, obligation


def _feedback_for_version(
    location,
    args,
    version,
    *,
    suffix: str,
    response_character: str,
    feedback_request_id: str,
    obligation_request_id: str,
):
    store = EvaluationHandoffStore(
        sqlite3.connect(location.seed[1], isolation_level=None)
    )
    handoff = store.register(
        create_handoff(
            version.version_id,
            version.governing_manifest.canonical_digest,
            f"evaluation-sink:{suffix}",
            max_attempts=3,
        )
    )
    handoff = store.persist_attempt(handoff.handoff_id)
    handoff = store.mark_attempt_sent(
        handoff.handoff_id, handoff.attempts[0].attempt_id
    )
    acknowledgement = Acknowledgement.create(
        handoff_id=handoff.handoff_id,
        attempt_id=handoff.attempts[0].attempt_id,
        candidate_version_id=version.version_id,
        governing_manifest_digest=version.governing_manifest.canonical_digest,
        sink_id=handoff.sink_id,
        outcome=AcknowledgementOutcome.ACKNOWLEDGED,
        response_digest="sha256:" + response_character * 64,
    )
    handoff = store.correlate_acknowledgement(handoff.handoff_id, acknowledgement)
    store._connection.close()
    authentication = args["authenticator"].authenticate(
        location.seed[0][3], now=args["clock"]()
    )
    actor = digest_bytes(
        canonical_json_bytes(
            {
                "principal_id": authentication.principal_id,
                "credential_binding_digest": authentication.credential_binding_digest,
            }
        )
    )
    scopes = {
        item.required_scope for item in args["command_registry"].definitions()
    } | {"authority.evaluation-feedback.reconcile"}
    args["authorizer"] = StaticAuthorizer(
        policy_version="feedback-test-v1",
        grants_by_principal={"editor": frozenset(scopes)},
    )
    feedback = create_evaluation_feedback(
        handoff=handoff,
        attempt=handoff.attempts[0],
        acknowledgement=acknowledgement,
        candidate_version=version,
        source_feedback_id=f"evaluation-feedback:{suffix}",
        outcome=EvaluationFeedbackOutcome.ACCEPTED,
        reason=EvaluationFeedbackReason.INTAKE_ACCEPTED,
        detail_digest="sha256:" + "2" * 64,
        request_id=feedback_request_id,
        actor_identity_digest=actor,
        idempotency_key=f"feedback:{suffix}",
    )
    obligation = create_reconciliation_obligation(
        feedback,
        request_id=obligation_request_id,
        actor_identity_digest=actor,
        idempotency_key=f"obligation:{suffix}",
    )
    return feedback, obligation


def _advance_candidate_head(location, version):
    from newsroom.increment6.feedback import (
        merge_evaluation_feedback_authority_registries,
    )

    previous, advanced_hypothesis = candidate_fixture._advance_record_one(
        location, merge_evaluation_feedback_authority_registries
    )
    assert advanced_hypothesis.previous_version_id == previous.version_id
    binding = candidate_fixture.CandidateUseCollisionBinding(
        advanced_hypothesis.hypothesis_id,
        advanced_hypothesis.version_id,
        advanced_hypothesis.canonical_digest,
        candidate_fixture.CandidateUseOperation.USE_CURRENT_CANDIDATE,
        version.candidate_id,
        version.governing_manifest.collision_namespace,
        version.governing_manifest.collision_key_digest,
        "retrieval-generation-v2",
        candidate_fixture.QUERY_VALID,
        candidate_fixture.SERVING,
        42,
    )
    collision_request, collision = candidate_fixture._named_snapshot(
        location, binding, occupied_candidate_id=version.candidate_id
    )
    manifest = candidate_fixture._manifest(
        location,
        advanced_hypothesis,
        collision,
        merge_evaluation_feedback_authority_registries,
    )
    request = candidate_fixture.CandidateAdmissionRequest(
        "44444444-4444-4444-8444-444444444444",
        candidate_fixture._actor_digest(location.seed, "actor-1"),
        "candidate:feedback-head-advance",
        version.version_id,
        version.canonical_digest,
        version.ordinal,
        manifest.semantic_scope_digest,
        collision_request.request_digest,
        manifest.governing_state_binding.canonical_digest,
        None,
    )
    admission = candidate_fixture.evaluate_candidate_admission(
        request=request,
        manifest=manifest,
        collision=collision,
        current_version=version,
        governing_state=candidate_fixture.CandidateGoverningState(
            candidate_fixture.CandidateGoverningStateStatus.COMPLETE,
            manifest.governing_state_binding,
        ),
    )
    successor = candidate_fixture._Handle(
        location, merge_evaluation_feedback_authority_registries
    )
    try:
        advanced = successor._opened().admit(
            admission.canonical_bytes,
            collision_request=collision_request,
            proof=location.seed[0][3],
        )
    finally:
        successor.close()
    assert advanced.candidate_id == version.candidate_id
    assert advanced.previous_version_id == version.version_id
    assert advanced.ordinal == version.ordinal + 1
    return advanced


def test_accept_replay_snapshot_and_direct_tamper_fail_closed(tmp_path: Path) -> None:
    location, args, feedback, obligation = _seed(tmp_path)
    with pytest.raises(TypeError):
        open_evaluation_feedback_authority_system(
            location.seed[1],
            retrieval_authority=args["retrieval_authority"],
            authenticator=args["authenticator"],
            authorizer=args["authorizer"],
            command_registry=args["command_registry"],
            payload_schemas=args["payload_schemas"],
            _store_class=object,  # type: ignore[call-arg]
        )
    authority = open_evaluation_feedback_authority_system(
        location.seed[1],
        retrieval_authority=args["retrieval_authority"],
        authenticator=args["authenticator"],
        authorizer=args["authorizer"],
        command_registry=args["command_registry"],
        payload_schemas=args["payload_schemas"],
        clock=args["clock"],
    )
    accepted = authority.accept(
        feedback.canonical_bytes,
        obligation.canonical_bytes,
        candidate_proof=location.seed[0][3],
    )
    assert (
        HandoffAcceptanceSnapshot.from_canonical_bytes(
            accepted.handoff_snapshot.canonical_bytes
        )
        == accepted.handoff_snapshot
    )
    assert (
        authority.accept(
            feedback.canonical_bytes,
            obligation.canonical_bytes,
            candidate_proof=object(),
        )
        == accepted
    )
    connection = authority._EvaluationFeedbackAuthority__authority._connection
    handoff_guard = connection.execute(
        "SELECT sql FROM sqlite_master WHERE name='evaluation_handoff_identity_guard'"
    ).fetchone()[0]
    authority_counts = tuple(
        connection.execute(
            "SELECT (SELECT count(*) FROM evaluation_feedback),"
            "(SELECT count(*) FROM evaluation_reconciliation_dispositions),"
            "(SELECT count(*) FROM ledger_events)"
        ).fetchone()
    )
    connection.execute("DROP TRIGGER evaluation_handoff_identity_guard")
    connection.execute(
        "UPDATE evaluation_handoffs SET max_attempts=4 WHERE handoff_id=?",
        (feedback.handoff_id,),
    )
    connection.execute(handoff_guard)
    assert (
        tuple(
            connection.execute(
                "SELECT (SELECT count(*) FROM evaluation_feedback),"
                "(SELECT count(*) FROM evaluation_reconciliation_dispositions),"
                "(SELECT count(*) FROM ledger_events)"
            ).fetchone()
        )
        == authority_counts
    )
    assert (
        authority.accept(
            feedback.canonical_bytes,
            obligation.canonical_bytes,
            candidate_proof=object(),
        ).handoff_snapshot.observed_max_attempts
        == 3
    )
    trigger = connection.execute(
        "SELECT sql FROM sqlite_master WHERE name='immutable_evaluation_feedback'"
    ).fetchone()[0]
    connection.execute("DROP TRIGGER immutable_evaluation_feedback")
    connection.execute(
        "UPDATE evaluation_feedback SET recorded_at='2042-02-03T00:00:00.000000Z' "
        "WHERE feedback_id=?",
        (feedback.feedback_id,),
    )
    connection.execute(trigger)
    with pytest.raises(FeedbackContractError):
        authority.load(feedback.feedback_id)
    authority.close()


@pytest.fixture
def _generic_disposition_base(tmp_path: Path):
    # Each test may advance or corrupt this authority; never share its writes.
    location, args, feedback, obligation = _seed(tmp_path)
    authority = open_evaluation_feedback_authority_system(
        location.seed[1],
        retrieval_authority=args["retrieval_authority"],
        authenticator=args["authenticator"],
        authorizer=args["authorizer"],
        command_registry=args["command_registry"],
        payload_schemas=args["payload_schemas"],
        clock=args["clock"],
    )
    accepted = authority.accept(
        feedback.canonical_bytes,
        obligation.canonical_bytes,
        candidate_proof=location.seed[0][3],
    )
    authority.close()
    from newsroom.increment6.feedback import (
        merge_evaluation_feedback_authority_registries,
    )

    candidate = candidate_fixture._Handle(
        location, merge_evaluation_feedback_authority_registries
    )
    version = candidate._opened().load_version(feedback.candidate_version_id)
    candidate.close()
    advanced = _advance_candidate_head(location, version)
    assert advanced.version_id != feedback.candidate_version_id
    return location, args, feedback, obligation, accepted, version


def test_stale_candidate_feedback_is_rejected(
    _generic_disposition_base,
) -> None:
    location, args, _, _, _, version = _generic_disposition_base
    stale_feedback, stale_obligation = _feedback_for_version(
        location,
        args,
        version,
        suffix="stale-candidate-version",
        response_character="5",
        feedback_request_id="55555555-5555-4555-8555-555555555555",
        obligation_request_id="66666666-6666-4666-8666-666666666666",
    )
    authority = open_evaluation_feedback_authority_system(
        location.seed[1],
        retrieval_authority=args["retrieval_authority"],
        authenticator=args["authenticator"],
        authorizer=args["authorizer"],
        command_registry=args["command_registry"],
        payload_schemas=args["payload_schemas"],
        clock=args["clock"],
    )
    with pytest.raises(
        FeedbackContractError, match="fresh feedback requires current Candidate head"
    ):
        authority.accept(
            stale_feedback.canonical_bytes,
            stale_obligation.canonical_bytes,
            candidate_proof=location.seed[0][3],
        )
    authority.close()


@pytest.fixture
def _generic_disposition_state(_generic_disposition_base):
    location, args, feedback, obligation, accepted, _ = _generic_disposition_base
    authority = open_evaluation_feedback_authority_system(
        location.seed[1],
        retrieval_authority=args["retrieval_authority"],
        authenticator=args["authenticator"],
        authorizer=args["authorizer"],
        command_registry=args["command_registry"],
        payload_schemas=args["payload_schemas"],
        clock=args["clock"],
    )
    disposition = append_reconciliation_disposition(
        accepted.obligation,
        (),
        outcome=ReconciliationDispositionOutcome.UNRESOLVED,
        reason=ReconciliationDispositionReason.AWAITING_RECONCILIATION,
        resolution_digest="sha256:" + "6" * 64,
        request_id="11111111-1111-4111-8111-111111111111",
        actor_identity_digest=feedback.actor_identity_digest,
        idempotency_key="disposition:v25-test",
        expected_current_disposition_id=None,
        expected_current_disposition_digest=None,
        expected_current_ordinal=0,
    )
    assert (
        authority.append_disposition(
            disposition.canonical_bytes, candidate_proof=location.seed[0][3]
        )
        == disposition
    )
    assert (
        authority.append_disposition(
            disposition.canonical_bytes, candidate_proof=object()
        )
        == disposition
    )
    connection = authority._EvaluationFeedbackAuthority__authority._connection
    assert (
        connection.execute(
            "SELECT authority_aggregate_version FROM evaluation_reconciliation_dispositions"
        ).fetchone()[0]
        == 2
    )
    authority.close()
    return location, args, feedback, obligation, accepted, disposition


def test_disposition_is_generic_ledger_anchored_and_replay_precedes_ports(
    _generic_disposition_state,
) -> None:
    _, _, _, _, accepted, disposition = _generic_disposition_state
    assert disposition.obligation_id == accepted.obligation.obligation_id


def _blocked_disposition(obligation, feedback, disposition):
    return append_reconciliation_disposition(
        obligation,
        (disposition,),
        outcome=ReconciliationDispositionOutcome.BLOCKED,
        reason=ReconciliationDispositionReason.DEPENDENCY_UNAVAILABLE,
        resolution_digest="sha256:" + "7" * 64,
        request_id="22222222-2222-4222-8222-222222222222",
        actor_identity_digest=feedback.actor_identity_digest,
        idempotency_key="disposition:v25-blocked",
        expected_current_disposition_id=disposition.disposition_id,
        expected_current_disposition_digest=disposition.canonical_digest,
        expected_current_ordinal=1,
    )


def _fulfilled_disposition(obligation, feedback, disposition, blocked):
    return append_reconciliation_disposition(
        obligation,
        (disposition, blocked),
        outcome=ReconciliationDispositionOutcome.FULFILLED,
        reason=ReconciliationDispositionReason.FEEDBACK_RECORDED,
        resolution_digest="sha256:" + "8" * 64,
        request_id="33333333-3333-4333-8333-333333333333",
        actor_identity_digest=feedback.actor_identity_digest,
        idempotency_key="disposition:v25-fulfilled",
        expected_current_disposition_id=blocked.disposition_id,
        expected_current_disposition_digest=blocked.canonical_digest,
        expected_current_ordinal=2,
    )


def test_disposition_restart_loads_accepted_feedback(
    _generic_disposition_state,
) -> None:
    location, args, feedback, _, accepted, _ = _generic_disposition_state
    reopened = open_evaluation_feedback_authority_system(
        location.seed[1],
        retrieval_authority=args["retrieval_authority"],
        authenticator=args["authenticator"],
        authorizer=args["authorizer"],
        command_registry=args["command_registry"],
        payload_schemas=args["payload_schemas"],
        clock=args["clock"],
    )
    assert reopened.load(feedback.feedback_id) == accepted
    reopened.close()


def test_disposition_restart_appends_blocked_idempotently(
    _generic_disposition_state,
) -> None:
    location, args, feedback, obligation, _, disposition = _generic_disposition_state
    reopened = open_evaluation_feedback_authority_system(
        location.seed[1],
        retrieval_authority=args["retrieval_authority"],
        authenticator=args["authenticator"],
        authorizer=args["authorizer"],
        command_registry=args["command_registry"],
        payload_schemas=args["payload_schemas"],
        clock=args["clock"],
    )
    blocked = _blocked_disposition(obligation, feedback, disposition)
    assert (
        reopened.append_disposition(
            blocked.canonical_bytes, candidate_proof=location.seed[0][3]
        )
        == blocked
    )
    assert (
        reopened.append_disposition(blocked.canonical_bytes, candidate_proof=object())
        == blocked
    )
    reopened.close()


def test_disposition_restart_chain_is_terminal(
    _generic_disposition_state,
) -> None:
    location, args, feedback, obligation, _, disposition = _generic_disposition_state
    reopened = open_evaluation_feedback_authority_system(
        location.seed[1],
        retrieval_authority=args["retrieval_authority"],
        authenticator=args["authenticator"],
        authorizer=args["authorizer"],
        command_registry=args["command_registry"],
        payload_schemas=args["payload_schemas"],
        clock=args["clock"],
    )
    blocked = _blocked_disposition(obligation, feedback, disposition)
    assert (
        reopened.append_disposition(
            blocked.canonical_bytes, candidate_proof=location.seed[0][3]
        )
        == blocked
    )
    fulfilled = _fulfilled_disposition(obligation, feedback, disposition, blocked)
    assert (
        reopened.append_disposition(
            fulfilled.canonical_bytes, candidate_proof=location.seed[0][3]
        )
        == fulfilled
    )
    assert (
        reopened.append_disposition(fulfilled.canonical_bytes, candidate_proof=object())
        == fulfilled
    )
    with pytest.raises(FeedbackContractError, match="terminal"):
        append_reconciliation_disposition(
            obligation,
            (disposition, blocked, fulfilled),
            outcome=ReconciliationDispositionOutcome.UNRESOLVED,
            reason=ReconciliationDispositionReason.AWAITING_RECONCILIATION,
            resolution_digest="sha256:" + "9" * 64,
            request_id="77777777-7777-4777-8777-777777777777",
            actor_identity_digest=feedback.actor_identity_digest,
            idempotency_key="disposition:v25-after-terminal",
            expected_current_disposition_id=fulfilled.disposition_id,
            expected_current_disposition_digest=fulfilled.canonical_digest,
            expected_current_ordinal=3,
        )
    reopened.close()


def test_disposition_tamper_is_evident(
    _generic_disposition_state,
) -> None:
    location, args, feedback, _, _, disposition = _generic_disposition_state
    reopened = open_evaluation_feedback_authority_system(
        location.seed[1],
        retrieval_authority=args["retrieval_authority"],
        authenticator=args["authenticator"],
        authorizer=args["authorizer"],
        command_registry=args["command_registry"],
        payload_schemas=args["payload_schemas"],
        clock=args["clock"],
    )
    connection = reopened._EvaluationFeedbackAuthority__authority._connection
    trigger = connection.execute(
        "SELECT sql FROM sqlite_master WHERE name='immutable_evaluation_feedback'"
    ).fetchone()[0]
    event = connection.execute(
        "SELECT authority_event_id,recorded_at FROM "
        "evaluation_reconciliation_dispositions WHERE disposition_id=?",
        (disposition.disposition_id,),
    ).fetchone()
    connection.execute("DROP TRIGGER immutable_evaluation_feedback")
    connection.execute(
        "UPDATE evaluation_feedback SET authority_event_id=?,"
        "recorded_at=? WHERE feedback_id=?",
        (*event, feedback.feedback_id),
    )
    connection.execute(trigger)
    with pytest.raises(FeedbackContractError):
        reopened.load(feedback.feedback_id)
    reopened.close()


@pytest.mark.parametrize(
    ("table", "trigger", "column", "value"),
    (
        (
            "authority_commands",
            "immutable_authority_commands_update",
            "expected_aggregate_version",
            7,
        ),
        (
            "authority_audit_events",
            "immutable_authority_audit_events_update",
            "detail_digest",
            "sha256:" + "0" * 64,
        ),
        (
            "authority_audit_events",
            "immutable_authority_audit_events_update",
            "authorization_request_digest",
            "COPY",
        ),
        (
            "authority_aggregate_versions",
            "immutable_aggregate_versions_update",
            "trust_scope",
            "OBSERVED",
        ),
    ),
)
def test_generic_envelope_tamper_fails_with_restored_trigger(
    tmp_path: Path, table: str, trigger: str, column: str, value: object
) -> None:
    location, args, feedback, obligation = _seed(tmp_path)
    authority = open_evaluation_feedback_authority_system(
        location.seed[1],
        retrieval_authority=args["retrieval_authority"],
        authenticator=args["authenticator"],
        authorizer=args["authorizer"],
        command_registry=args["command_registry"],
        payload_schemas=args["payload_schemas"],
        clock=args["clock"],
    )
    authority.accept(
        feedback.canonical_bytes,
        obligation.canonical_bytes,
        candidate_proof=location.seed[0][3],
    )
    connection = authority._EvaluationFeedbackAuthority__authority._connection
    sql = connection.execute(
        "SELECT sql FROM sqlite_master WHERE name=?", (trigger,)
    ).fetchone()[0]
    connection.execute(f"DROP TRIGGER {trigger}")
    connection.execute("PRAGMA foreign_keys=OFF")
    if value == "COPY":
        connection.execute(
            "UPDATE authority_audit_events SET (authentication_context_id,authorization_request_digest,"
            "authorization_decision_id)=(SELECT authentication_context_id,authorization_request_digest,"
            "authorization_decision_id FROM authority_commands WHERE command_type!='evaluation-feedback.reconcile' LIMIT 1) "
            "WHERE command_id=(SELECT e.command_id FROM ledger_events e JOIN evaluation_feedback f "
            "ON f.authority_event_id=e.event_id WHERE f.feedback_id=?)",
            (feedback.feedback_id,),
        )
    else:
        identity_column = (
            "aggregate_id" if table == "authority_aggregate_versions" else "command_id"
        )
        local_column = (
            "authority_aggregate_id"
            if table == "authority_aggregate_versions"
            else "authority_event_id"
        )
        identity_expression = (
            f"SELECT {local_column} FROM evaluation_feedback WHERE feedback_id=?"
            if table == "authority_aggregate_versions"
            else "SELECT e.command_id FROM ledger_events e JOIN evaluation_feedback f "
            "ON f.authority_event_id=e.event_id WHERE f.feedback_id=?"
        )
        connection.execute(
            f"UPDATE {table} SET {column}=? WHERE {identity_column}="
            f"({identity_expression})",
            (value, feedback.feedback_id),
        )
    connection.execute(sql)
    connection.execute("PRAGMA foreign_keys=ON")
    with pytest.raises(FeedbackContractError):
        authority.load(feedback.feedback_id)
    authority.close()


@pytest.mark.parametrize("tamper", ("namespace", "result", "causation"))
def test_retained_namespace_result_and_causation_tamper_fail_closed(
    tmp_path: Path, tamper: str
) -> None:
    location, args, feedback, obligation = _seed(tmp_path)
    authority = open_evaluation_feedback_authority_system(
        location.seed[1],
        retrieval_authority=args["retrieval_authority"],
        authenticator=args["authenticator"],
        authorizer=args["authorizer"],
        command_registry=args["command_registry"],
        payload_schemas=args["payload_schemas"],
        clock=args["clock"],
    )
    authority.accept(
        feedback.canonical_bytes,
        obligation.canonical_bytes,
        candidate_proof=location.seed[0][3],
    )
    connection = authority._EvaluationFeedbackAuthority__authority._connection
    event_id, command_id = connection.execute(
        "SELECT e.event_id,e.command_id FROM ledger_events e "
        "JOIN evaluation_feedback f ON f.authority_event_id=e.event_id "
        "WHERE f.feedback_id=?",
        (feedback.feedback_id,),
    ).fetchone()
    if tamper in {"namespace", "result"}:
        trigger_name = "immutable_authority_commands_update"
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (trigger_name,)
        ).fetchone()[0]
        connection.execute(f"DROP TRIGGER {trigger_name}")
        if tamper == "namespace":
            connection.execute(
                "UPDATE authority_commands SET idempotency_namespace=? WHERE command_id=?",
                ("sha256:" + "f" * 64, command_id),
            )
        else:
            donor = connection.execute(
                "SELECT result_bytes,result_digest FROM authority_commands "
                "WHERE command_id!=? AND result_bytes IS NOT NULL LIMIT 1",
                (command_id,),
            ).fetchone()
            assert donor is not None
            connection.execute(
                "UPDATE authority_commands SET result_bytes=?,result_digest=? WHERE command_id=?",
                (*donor, command_id),
            )
        connection.execute(trigger_sql)
    else:
        trigger_name = "immutable_ledger_events_update"
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (trigger_name,)
        ).fetchone()[0]
        connection.execute(f"DROP TRIGGER {trigger_name}")
        connection.execute(
            "UPDATE ledger_events SET causation_kind='EXTERNAL',"
            "causation_identifier='forged-causation',"
            "causation_external_system='forged-system' WHERE event_id=?",
            (event_id,),
        )
        connection.execute(trigger_sql)
    with pytest.raises(FeedbackContractError):
        authority.load(feedback.feedback_id)
    authority.close()
