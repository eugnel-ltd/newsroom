import json
import sqlite3
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS

import pytest

from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.control_plane.model_usage import (
    CONSERVATIVE_DISPOSITION_SCHEMA_VERSION,
    InvocationAllocation,
    InvocationEfficiencyPolicy,
    UsageComponents,
    ModelUsageService,
    WorkEnvelope,
    WorkloadClass,
)
from newsroom.control_plane.native_progress import NativeRevisionJournal
from newsroom.control_plane.native_qualification import (
    NativeQualificationError,
    record_qualification,
    validate_qualification,
)
from newsroom.control_plane.store import append_ledger, connect
from newsroom.increment9.proving import SOURCE_IDS
from newsroom.tests.test_native_graphiti import _native

IDENTITY = "sha256:" + "a" * 64


def _open(path):
    ModelUsageService(str(path))
    return connect(str(path))


def _portfolio(journal, unit=None, *, first_reason=None, source_override=None):
    rights_holds = {
        "HK-01": "MEDIA_REUSE_PERMISSION_SCOPE_NOT_ESTABLISHED",
        "HK-04": "NON_COMMERCIAL_INTERNAL_USE_ONLY",
        "RAD-01": "AUTOMATED_REUSE_PERMISSION_NOT_RETAINED",
        "RAD-02": "COMPUTER_ANALYSIS_PERMISSION_NOT_RETAINED",
    }
    dispositions = tuple(
        NS(
            source_id=source_id, status="HOLD" if source_id in rights_holds else "READY",
            reason_code=(
                first_reason if source_id == SOURCE_IDS[0] and first_reason
                else rights_holds[source_id] if source_id in rights_holds
                else "NO_ACTIVE_WARNINGS_OBSERVED"
                if source_id in {"UK-10", "HK-02"}
                else "GOVERNED_REVISIONS_RETAINED"
            ),
            units=(unit,) if unit is not None and source_id == unit.source_id else (),
            observations=((
                f"https://terms.example/{source_id}",
                digest_canonical({"terms": source_id}),
                f"admission-{source_id}",
                f"access-{source_id}",
            ),) if source_id in rights_holds else (),
            item_holds=(),
        )
        for source_id in SOURCE_IDS
    )
    if source_override is not None:
        dispositions = (
            NS(**{**vars(dispositions[0]), **source_override}), *dispositions[1:],
        )
    journal.sources(dispositions)


def _cycle(
    connection,
    *,
    identity=IDENTITY,
    outcome="COMPLETE",
    queued=0,
    revision_state="EVIDENCE_HOLD",
    reason="SOURCE_LOCAL_EVIDENCE_HOLD",
    source_reason=None,
    source_override=None,
    facts=None,
):
    journal = NativeRevisionJournal(connection)
    unit = _native("qualification-hold")
    journal.land((unit,))
    journal.advance(unit.revision_id, stage=revision_state, facts={"reason": reason} if facts is None else facts)
    _portfolio(journal, unit, first_reason=source_reason, source_override=source_override)
    append_ledger(connection, "NATIVE_SERVICE_CYCLE_STARTED", {
        "cycle_id": "qualification-cycle", "runtime_identity_digest": identity,
    })
    connection.commit()
    append_ledger(connection, "NATIVE_SERVICE_CYCLE_TERMINAL", {
        "runtime_identity_digest": identity,
        "cycle_id": "qualification-cycle",
        "outcome": outcome,
        "failure_class": None if outcome == "COMPLETE" else "RuntimeError",
        "pipeline": {
            "sources": list(journal.portfolio),
            "revision_states": {
                revision_state: 1, **({"QUEUED": queued} if queued else {})
            },
            "unclassified_revisions": queued,
        },
    })
    connection.commit()
    return journal


def _allocation(
    connection,
    *,
    usage_status=None,
    workload=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
    provider="fixture",
    route="fixture",
    ingest_id=None,
):
    workload_value = workload.value
    typed_policy = InvocationEfficiencyPolicy.create(
        policy_id="qualification-policy", version="v1", workload_class=workload,
        provider=provider, route=route, model="fixture", reasoning="none",
        one_turn=True, exact_input=True, skills_enabled=False, tools_enabled=False,
        mcp_enabled=False, prior_message_count=0, max_prompt_bytes=10,
        max_context_tokens=10, max_output_tokens=1, max_total_tokens=10,
        prompt_contract_version="qualification-v1",
        output_schema_digest=digest_canonical({"schema": 1}),
        allowed_context_identities=("qualification-context",),
        allowed_config_identities=("qualification-config",),
        hard_estimate_ceiling_tokens=None, evidence_digest=digest_canonical({"evidence": 1}),
        qualified=True,
    )
    policy = typed_policy.canonical_digest
    allocated_at = datetime(2026, 9, 9, tzinfo=UTC)
    graphiti = workload in {
        WorkloadClass.GRAPHITI_CHAT_PRIMARY,
        WorkloadClass.GRAPHITI_CHAT_FALLBACK,
        WorkloadClass.GRAPHITI_EMBEDDING,
    }
    native_embedding = workload is WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING
    exact_ingest_id = ingest_id or (
        digest_canonical({"ingest": workload_value})
        if graphiti or native_embedding else None
    )
    envelope = WorkEnvelope.create(
        cycle_id="qualification-model-cycle",
        workload_class=workload,
        admitted_at=allocated_at,
        admission_decision_id=None,
        candidate_id="candidate-1" if not graphiti and not native_embedding else None,
        hypothesis_digest=(
            digest_canonical({"hypothesis": 1})
            if workload is WorkloadClass.NATIVE_EVIDENCE_ASSESSOR else None
        ),
        evidence_package_digest=(
            digest_canonical({"evidence": 1})
            if workload is WorkloadClass.NATIVE_EVIDENCE_ASSESSOR else None
        ),
        ingest_id=exact_ingest_id,
        graphiti_attempt_id=(
            f"{exact_ingest_id}:1" if graphiti else None
        ),
    )
    allocation = InvocationAllocation.create(
        envelope_id=envelope.envelope_id,
        cycle_id=envelope.cycle_id,
        leaf_ordinal=1,
        workload_class=workload,
        invocation_policy_digest=policy,
        provider=provider,
        route=route,
        model="fixture",
        reasoning="none",
        prompt_contract_version="qualification-v1",
        prompt_bytes=1,
        prompt_digest=digest_canonical({"prompt": 1}),
        request_digest=digest_canonical({"request": 1}),
        output_schema_digest=digest_canonical({"schema": 1}),
        max_output_tokens=1,
        context_manifest_digest=digest_canonical({"context": 1}),
        context_identity="qualification-context",
        config_identity="qualification-config",
        one_turn=True,
        exact_input=True,
        skills_enabled=False,
        tools_enabled=False,
        mcp_enabled=False,
        prior_message_count=0,
        allocated_at=allocated_at,
        recovery_deadline_at=allocated_at + timedelta(minutes=1),
        parent_invocation_id=None,
    )
    policy_record = canonical_json_bytes(typed_policy.as_record()).decode()
    envelope_value = envelope.as_record()
    allocation_value = allocation.as_record()
    assert allocation_value["invocation_policy_digest"] == policy
    assert "policy_digest" not in allocation_value
    envelope_record = canonical_json_bytes(envelope_value).decode()
    allocation_record = canonical_json_bytes(allocation_value).decode()
    connection.execute(
        "INSERT INTO model_invocation_policies VALUES(?,?,?,?,?,?,?,?,?)",
        (policy, "qualification-policy", "v1", workload_value,
         provider, route, "fixture", 1, policy_record),
    )
    connection.execute(
        "INSERT INTO model_work_envelopes VALUES(?,?,?,?,?,?)",
        (envelope.envelope_id, envelope.cycle_id, workload_value,
         envelope_value["admitted_at"], envelope.canonical_digest, envelope_record),
    )
    connection.execute(
        "INSERT INTO model_invocation_allocations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (allocation.invocation_id, envelope.envelope_id, envelope.cycle_id, 1,
         workload_value, policy, provider, route,
         "fixture", allocation.request_digest, None,
         allocation_value["allocated_at"], allocation.canonical_digest,
         allocation_record),
    )
    if usage_status is not None:
        terminal = {
            "schema_version": "newsroom.model-usage.v3",
            "terminal_digest": "",
            "invocation_id": allocation.invocation_id,
            "outcome": "COMPLETE",
            "failure_class": (
                "MISSING_PROVIDER_TELEMETRY"
                if usage_status == "UNREPORTED" else None
            ),
            "usage_status": usage_status,
            "components": UsageComponents(
                total_tokens=None if usage_status == "UNREPORTED" else 1,
                provenance="UNAVAILABLE" if usage_status == "UNREPORTED" else "PROVIDER_REPORTED",
            ).as_record(),
            "dispatch_at": "2026-09-09T00:00:01.000000Z",
            "completed_at": "2026-09-09T00:00:02.000000Z",
            "observed_at": "2026-09-09T00:00:02.000000Z",
            "provider_telemetry_digest": (
                None if usage_status == "UNREPORTED" else digest_canonical({"telemetry": 1})
            ),
            "raw_telemetry_pointer": None,
            "estimate_policy_digest": None,
            "estimate_calculation": None,
            "pre_dispatch_zero_proved": False,
            "od_011_reference": (
                "OD-011:FIXTURE" if native_embedding
                or workload is WorkloadClass.GRAPHITI_EMBEDDING else None
            ),
            "subscription_cli_chat_not_cash_debited": (
                not native_embedding and workload is not WorkloadClass.GRAPHITI_EMBEDDING
            ),
            "policy_breach": None,
        }
        terminal_digest = digest_canonical(terminal)
        terminal["terminal_digest"] = terminal_digest
        connection.execute(
            "INSERT INTO model_invocation_terminals VALUES(?,?,?,?,?,?,?)",
            (terminal_digest, allocation.invocation_id, usage_status, "COMPLETE",
             terminal["failure_class"], "2026-09-09T00:00:02.000000Z",
             canonical_json_bytes(terminal).decode()),
        )
    connection.commit()
    return allocation.invocation_id


def _conservative_disposition(connection, invocation):
    terminal_digest = connection.execute(
        "SELECT terminal_digest FROM model_invocation_terminals WHERE invocation_id=?",
        (invocation,),
    ).fetchone()[0]
    allocation_digest, policy_digest = connection.execute(
        "SELECT canonical_digest,policy_digest FROM model_invocation_allocations "
        "WHERE invocation_id=?", (invocation,),
    ).fetchone()
    record = {
        "schema_version": CONSERVATIVE_DISPOSITION_SCHEMA_VERSION,
        "invocation_id": invocation,
        "terminal_digest": terminal_digest,
        "allocation_digest": allocation_digest,
        "policy_digest": policy_digest,
        "approved_plan_digest": digest_canonical({"plan": 1}),
        "usage_status": "ESTIMATED",
        "components": {"total_tokens": 1, "provenance": "BOUNDED_ESTIMATE"},
        "estimate_policy_digest": policy_digest,
        "estimate_calculation": (
            "QUALIFIED_POLICY_MAX_TOTAL_TOKENS_CONSERVATIVE_UPPER_BOUND"
        ),
        "exact_usage_remains_unknown": True,
        "provider_dispatch_preserved": True,
        "unknown_spend_released": False,
        "authority_digest": digest_canonical({"authority": 1}),
        "approved_by": "github:fol2",
        "approval_reference": "retained:test",
        "approved_at": "2026-09-09T00:00:03.000000Z",
        "observed_at": "2026-09-09T00:00:03.000000Z",
    }
    digest = digest_canonical(record)
    record["disposition_digest"] = digest
    connection.execute(
        "INSERT INTO model_usage_conservative_dispositions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (digest, invocation, terminal_digest, allocation_digest, policy_digest,
         record["approved_plan_digest"], record["authority_digest"],
         record["approved_by"], record["approval_reference"], record["approved_at"],
         record["observed_at"], "ESTIMATED", canonical_json_bytes(record).decode()),
    )
    connection.commit()


def test_pending_usage_does_not_hide_a_later_policy_breach(tmp_path):
    from newsroom.control_plane.native_qualification import NativeQualificationPending

    connection = _open(tmp_path / "pending-and-breach.sqlite3")
    try:
        _cycle(connection)
        _allocation(connection, usage_status="UNREPORTED")
        _allocation(connection, usage_status="UNREPORTED",
                    workload=WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING)
        last = connection.execute(
            "SELECT invocation_id FROM model_invocation_allocations "
            "ORDER BY allocated_at DESC,invocation_id DESC LIMIT 1"
        ).fetchone()[0]
        record = json.loads(connection.execute(
            "SELECT record_json FROM model_invocation_terminals WHERE invocation_id=?",
            (last,),
        ).fetchone()[0])
        record["policy_breach"] = "INVOCATION_TOTAL_CEILING_EXCEEDED"
        record["terminal_digest"] = ""
        record["terminal_digest"] = digest_canonical(record)
        connection.execute(
            "UPDATE model_invocation_terminals SET terminal_digest=?,record_json=? "
            "WHERE invocation_id=?",
            (record["terminal_digest"], canonical_json_bytes(record).decode(), last),
        )
        connection.commit()
        with pytest.raises(NativeQualificationError, match="unresolved") as failure:
            record_qualification(connection, IDENTITY)
        assert not isinstance(failure.value, NativeQualificationPending)
        assert connection.execute(
            "SELECT count(*) FROM ledger WHERE kind='NATIVE_SERVICE_QUALIFICATION'"
        ).fetchone() == (0,)
    finally:
        connection.close()


def test_exact_identity_cycle_with_evidenced_hold_qualifies(tmp_path):
    path = tmp_path / "private.sqlite3"
    connection = _open(path)
    try:
        _cycle(connection)
        record_qualification(connection, IDENTITY)
        retained = validate_qualification(connection, IDENTITY)
        assert retained.cycle_id == "qualification-cycle"
        assert retained.started_seq < retained.terminal_seq
        assert retained.invocation_ids == ()
        assert retained.runtime_identity_digest == IDENTITY

        _cycle(connection, outcome="FAILED")
        _allocation(connection, workload=WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING)
        assert validate_qualification(connection, IDENTITY) == retained
    finally:
        connection.close()

    readonly = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        assert validate_qualification(readonly, IDENTITY) == retained
    finally:
        readonly.close()

    typed_hold = _open(tmp_path / "typed-hold.sqlite3")
    try:
        _cycle(
            typed_hold,
            revision_state="RETRIEVAL_HOLD",
            reason="CURRENT_SOURCE_REVISION_UNAVAILABLE",
        )
        record_qualification(typed_hold, IDENTITY)
    finally:
        typed_hold.close()


def test_retained_same_state_association_is_a_terminal_nonpublication(
    tmp_path,
) -> None:
    connection = _open(tmp_path / "same-state.sqlite3")
    try:
        journal = _cycle(
            connection,
            revision_state="SAME_STATE_ASSOCIATED",
            reason=None,
        )
        retained = record_qualification(connection, IDENTITY)
        assert validate_qualification(connection, IDENTITY) == retained
        assert {
            value["stage"] for value in (value for _, value in journal.iter_summaries())
        } == {"SAME_STATE_ASSOCIATED"}
    finally:
        connection.close()


def test_legacy_drift_failed_and_unfinished_cycles_never_qualify(tmp_path):
    for case in (
        "legacy", "drift", "failed", "queued", "invented", "generic",
        "source_missing",
    ):
        connection = _open(tmp_path / f"{case}.sqlite3")
        try:
            if case == "legacy":
                journal = NativeRevisionJournal(connection)
                _portfolio(journal)
                append_ledger(connection, "NATIVE_SERVICE_CYCLE_STARTED", {"cycle_id": "old"})
                append_ledger(connection, "NATIVE_SERVICE_CYCLE_TERMINAL", {
                    "cycle_id": "old", "outcome": "COMPLETE", "failure_class": None,
                    "pipeline": {"sources": list(journal.portfolio), "revision_states": {},
                                 "unclassified_revisions": 0},
                })
                connection.commit()
            else:
                _cycle(
                    connection,
                    identity="sha256:" + "b" * 64 if case == "drift" else IDENTITY,
                    outcome="FAILED" if case == "failed" else "COMPLETE",
                    queued=1 if case == "queued" else 0,
                    revision_state=(
                        "INVENTED_HOLD" if case == "invented" else "EVIDENCE_HOLD"
                    ),
                    reason="RuntimeError" if case == "generic" else "RETAINED_HOLD",
                    source_reason=(
                        "SOURCE_DEFINITION_MISSING"
                        if case == "source_missing" else None
                    ),
                )
            with pytest.raises(NativeQualificationError):
                record_qualification(connection, IDENTITY)
        finally:
            connection.close()


def _content_hold(reason="SOURCE_ITEM_NOT_YET_PUBLISHED"):
    return {
        "status": "HOLD", "reason_code": "SOURCE_ITEMS_HELD",
        "item_holds": (("https://www.gov.uk/held-item", reason),),
        "observations": ((
            "https://www.gov.uk/api/content/held-item",
            digest_canonical({"retained": "source content"}),
            "00000000-0000-4000-8000-000000000101",
            "00000000-0000-4000-8000-000000000102",
        ),),
    }


@pytest.mark.parametrize("reason", [
    "SOURCE_ITEM_NOT_YET_PUBLISHED",
    "SOURCE_ITEM_CHILD_COVERAGE_INCOMPLETE",
    "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE",
    "SOURCE_ITEM_RIGHTS_EXCLUSION_HOLD",
    "SOURCE_ITEM_METADATA_HOLD",
])
@pytest.mark.parametrize("repeated", [False, True])
def test_observation_bound_content_hold_qualifies_without_hiding_sibling(tmp_path, reason, repeated):
    path = tmp_path / "content-hold.sqlite3"
    connection = _open(path)
    try:
        held = _content_hold(reason)
        if repeated:
            # A feed item and its direct-child inventory can reach the same
            # retained observation and local HOLD; keep the original evidence.
            held["observations"] *= 2
            held["item_holds"] *= 2
        journal = _cycle(connection, source_override=held)
        retained = record_qualification(connection, IDENTITY)
        source = journal.portfolio[0]
        assert source["status"] == "HOLD"
        assert source["item_holds"] == [list(value) for value in held["item_holds"]]
        assert source["observations"] == [list(value) for value in held["observations"]]
        assert source["revision_ids"] == sorted(journal.units)
        assert validate_qualification(connection, IDENTITY) == retained
    finally:
        connection.close()
    readonly = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        assert validate_qualification(readonly, IDENTITY) == retained
    finally:
        readonly.close()


@pytest.mark.parametrize("case", [
    "unsupported", "transport", "invented", "missing_observation", "wrong_url",
    "conflicting_digest", "conflicting_admission", "conflicting_access",
    "bad_digest", "bad_admission", "bad_access", "bad_extra_observation",
    "no_items", "conflicting_items", "ready", "rights_reason", "other_route",
])
@pytest.mark.parametrize("held_reason", ["SOURCE_ITEM_NOT_YET_PUBLISHED", "SOURCE_ITEM_METADATA_HOLD"])
def test_unclassified_or_unbound_content_hold_does_not_qualify(tmp_path, case, held_reason):
    value = _content_hold(held_reason)
    if case in {"unsupported", "transport", "invented"}:
        reason = {
            "unsupported": "SOURCE_ITEM_DOCUMENT_TYPE_UNKNOWN",
            "transport": "SOURCE_ITEM_FETCH_INCOMPLETE",
            "invented": "INVENTED_HOLD",
        }[case]
        value = _content_hold(reason)
    elif case == "missing_observation":
        value["observations"] = ()
    elif case in {"wrong_url", "bad_digest", "bad_admission", "bad_access"}:
        parts = list(value["observations"][0])
        index = {"wrong_url": 0, "bad_digest": 1, "bad_admission": 2, "bad_access": 3}[case]
        parts[index] = "https://www.gov.uk/api/content/unrelated" if index == 0 else "invalid"
        value["observations"] = (tuple(parts),)
    elif case in {"conflicting_digest", "conflicting_admission", "conflicting_access", "bad_extra_observation"}:
        parts = list(value["observations"][0])
        index = {"conflicting_digest": 1, "conflicting_admission": 2, "conflicting_access": 3,
                 "bad_extra_observation": 1}[case]
        parts[index] = (
            "invalid" if case == "bad_extra_observation" else
            digest_canonical({"different": "source capture"}) if index == 1 else
            "00000000-0000-4000-8000-000000000103"
        )
        value["observations"] += (tuple(parts),)
    elif case == "no_items":
        value["item_holds"] = ()
    elif case == "conflicting_items":
        value["item_holds"] += ((value["item_holds"][0][0], "SOURCE_ITEM_RIGHTS_EXCLUSION_HOLD"),)
    elif case == "ready":
        value.update(status="READY", reason_code="GOVERNED_REVISIONS_RETAINED")
    elif case == "rights_reason":
        value["reason_code"] = "MEDIA_REUSE_PERMISSION_SCOPE_NOT_ESTABLISHED"
    elif case == "other_route":
        value["item_holds"] = (("https://example.org/held-item", value["item_holds"][0][1]),)
    connection = _open(tmp_path / f"{case}.sqlite3")
    try:
        _cycle(connection, source_override=value)
        with pytest.raises(NativeQualificationError):
            record_qualification(connection, IDENTITY)
    finally:
        connection.close()


def test_only_referenced_native_ledger_rows_enter_qualification(tmp_path):
    connection = _open(tmp_path / "bounded-ledger.sqlite3")
    try:
        append_ledger(connection, "UNRELATED_HISTORICAL_EVENT", {"value": "before"})
        connection.commit()
        _cycle(connection)
        record_qualification(connection, IDENTITY)

        connection.execute(
            "UPDATE ledger SET payload_json='{}' "
            "WHERE kind='UNRELATED_HISTORICAL_EVENT'"
        )
        connection.commit()
        assert validate_qualification(connection, IDENTITY).cycle_id == (
            "qualification-cycle"
        )

        connection.execute(
            "UPDATE ledger SET payload_json='{}' "
            "WHERE kind='NATIVE_REVISION_PROGRESS'"
        )
        connection.commit()
        # Ordinary continuation now lives in CURRENT rows, not STATE logs.
        assert validate_qualification(connection, IDENTITY).cycle_id == 'qualification-cycle'
        connection.execute("UPDATE native_current_heads SET state_json='{}'")
        connection.commit()
        with pytest.raises(NativeQualificationError, match='native revision|retained native qualification'):
            validate_qualification(connection, IDENTITY)
    finally:
        connection.close()


def test_qualification_retains_history_references_not_duplicate_payloads(tmp_path):
    from newsroom.control_plane.native_progress import LAND, PORTFOLIO, STATE
    from newsroom.control_plane.native_qualification import _ledger

    connection = _open(tmp_path / "history-references.sqlite3")
    try:
        _cycle(connection, facts={
            "reason": "SOURCE_LOCAL_EVIDENCE_HOLD", "retained_text": "x" * 262_144,
        })
        expected = connection.execute(
            "SELECT seq,kind,payload_digest,digest FROM ledger ORDER BY seq"
        ).fetchall()
        rows = _ledger(connection)
        assert [(row[0], row[2], row[3], row[6]) for row in rows] == expected
        assert all(row[4] is None for row in rows if row[2] in {LAND, STATE, PORTFOLIO})
        assert all(isinstance(row[4], str) for row in rows if row[2] not in {LAND, STATE, PORTFOLIO})
        retained = record_qualification(connection, IDENTITY)
        assert validate_qualification(connection, IDENTITY) == retained
    finally:
        connection.close()


@pytest.mark.parametrize("kind", ["NATIVE_REVISION_LANDED", "NATIVE_SOURCE_PORTFOLIO"])
def test_qualification_validates_payload_before_discarding_duplicate(kind, tmp_path):
    from newsroom.control_plane.native_qualification import _ledger

    connection = _open(tmp_path / "corrupt-history.sqlite3")
    try:
        _cycle(connection)
        connection.execute("UPDATE ledger SET payload_json='{}' WHERE kind=?", (kind,))
        connection.commit()
        with pytest.raises(NativeQualificationError, match="native ledger payload"):
            _ledger(connection)
    finally:
        connection.close()


def test_same_count_current_revision_fact_mutation_does_not_match_reference(tmp_path):
    connection = _open(tmp_path / 'current-revision-mutation.sqlite3')
    try:
        _cycle(connection)
        retained = record_qualification(connection, IDENTITY)
        connection.execute("UPDATE native_current_heads SET state_json='{}'")
        connection.commit()
        with pytest.raises(NativeQualificationError, match='native revision|retained native qualification'):
            validate_qualification(connection, IDENTITY)
        assert retained.revision_inventory_digest != digest_canonical({'EVIDENCE_HOLD': 1})
    finally:
        connection.close()


def test_native_invocation_must_have_resolved_retained_usage(tmp_path):
    unfinished = _open(tmp_path / "unfinished.sqlite3")
    try:
        _cycle(unfinished)
        _allocation(unfinished)
        with pytest.raises(NativeQualificationError, match="in flight"):
            record_qualification(unfinished, IDENTITY)
    finally:
        unfinished.close()

    resolved = _open(tmp_path / "resolved.sqlite3")
    try:
        _cycle(resolved)
        invocation = _allocation(resolved, usage_status="REPORTED")
        record_qualification(resolved, IDENTITY)
        assert validate_qualification(resolved, IDENTITY).invocation_ids == (invocation,)
    finally:
        resolved.close()

    unknown = _open(tmp_path / "unknown.sqlite3")
    try:
        _cycle(unknown)
        _allocation(unknown, usage_status="UNREPORTED")
        with pytest.raises(NativeQualificationError, match="unresolved"):
            record_qualification(unknown, IDENTITY)
    finally:
        unknown.close()

    reconciled = _open(tmp_path / "reconciled.sqlite3")
    try:
        journal = _cycle(reconciled)
        ingest_id = next(iter(journal.units.values()))[0].ingest_id
        invocation = _allocation(
            reconciled,
            usage_status="UNREPORTED",
            workload=WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            provider="cursor-agent-cli",
            route=WorkloadClass.GRAPHITI_CHAT_PRIMARY.value,
            ingest_id=ingest_id,
        )
        observation = digest_canonical({"dispatch": invocation})
        reconciled.execute(
            "INSERT INTO model_transport_observations VALUES(?,?,?,?,?,?)",
            (
                observation,
                invocation,
                "2026-09-09T00:00:01.000000Z",
                "DISPATCH_STARTED",
                observation,
                canonical_json_bytes({"dispatch": invocation}).decode(),
            ),
        )
        reconciled.commit()
        _conservative_disposition(reconciled, invocation)
        record_qualification(reconciled, IDENTITY)
        assert validate_qualification(reconciled, IDENTITY).invocation_ids == (
            invocation,
        )
    finally:
        reconciled.close()


@pytest.mark.parametrize("states,unclassified,ready", [
    ({}, 0, True),
    ({"ACKNOWLEDGED": 1, "EVIDENCE_HOLD": 2}, 0, True),
    ({"QUEUED": 1}, 1, False),
    ({"GRAPHITI_COMPLETE": 1}, 0, False),
    ({"QUEUED": 2, "GRAPHITI_COMPLETE": 1, "EVIDENCE_HOLD": 3}, 2, False),
])
def test_qualification_readiness_defers_only_valid_pending_reports(states, unclassified, ready):
    from newsroom.control_plane.native_qualification import qualification_report_ready

    assert qualification_report_ready(states, unclassified) is ready


@pytest.mark.parametrize("states,unclassified", [
    ({"UNKNOWN": 1}, 0), ({"ASSESSMENT_START": 1}, 0),
    ({"QUEUED": 0}, 0), ({"QUEUED": -1}, -1), ({"QUEUED": True}, 1),
    ({"GRAPHITI_COMPLETE": "1"}, 0), ({"QUEUED": 1}, 0),
    ({"ACKNOWLEDGED": 1}, 1), ({"ACKNOWLEDGED": 1}, False),
    ({1: 1}, 0), ([], 0),
])
def test_qualification_readiness_rejects_malformed_or_unknown_reports(states, unclassified):
    from newsroom.control_plane.native_qualification import qualification_report_ready

    with pytest.raises(NativeQualificationError, match="terminal inventory differs"):
        qualification_report_ready(states, unclassified)


# Exact checkpoints emitted by native_pipeline, native_retrieval and
# native_publication; these are continuation states, never qualification PASS.
_PENDING_CHECKPOINTS = (
    "QUEUED", "GRAPHITI_COMPLETE", "EMBEDDING_STARTED", "EMBEDDING_RETAINED",
    "DOCUMENT_RETAINED", "RETRIEVAL_COMPLETE", "CANDIDATE_ADMITTED",
    "ASSESSMENT_CONTRACT_REVALIDATION", "INTAKE_REQUESTED", "INTAKE_ACKNOWLEDGED",
    "ACQUISITION_STARTED", "ASSESSMENT_STARTED", "ASSESSMENT_INTERRUPTED",
    "EVIDENCE_RETAINED", "PUBLICATION_PREPARED", "PUBLICATION_STARTED",
)


@pytest.mark.parametrize("stage", _PENDING_CHECKPOINTS)
def test_durable_continuation_defers_readiness_but_never_qualifies(tmp_path, stage):
    from newsroom.control_plane.native_qualification import qualification_report_ready

    assert qualification_report_ready({stage: 1}, int(stage == "QUEUED")) is False
    connection = _open(tmp_path / "pending.sqlite3")
    try:
        _cycle(connection, revision_state=stage)
        with pytest.raises(NativeQualificationError, match="terminal inventory differs"):
            record_qualification(connection, IDENTITY)
    finally:
        connection.close()


def test_qualification_accepts_current_pairs_and_rejects_corrupt_current_root(tmp_path):
    from newsroom.tests.test_native_progress import _retrieval_facts
    connection = _open(tmp_path / 'current-pairs.sqlite3')
    journal = NativeRevisionJournal(connection)
    unit = _native('qualification-hold')
    journal.land((unit,))
    facts = _retrieval_facts()
    journal.advance(unit.revision_id, stage='RETRIEVAL_COMPLETE', facts=facts)
    _cycle(connection, facts=facts)
    assert connection.execute("SELECT count(*) FROM ledger WHERE kind='NATIVE_REVISION_PROGRESS'").fetchone()[0] == 0
    assert NativeRevisionJournal(connection).current(unit.revision_id)['facts'] == facts
    retained = record_qualification(connection, IDENTITY)
    assert validate_qualification(connection, IDENTITY) == retained
    connection.execute('PRAGMA foreign_keys=OFF')
    connection.execute("UPDATE native_current_heads SET pair_digest='sha256:' || printf('%064d', 0)")
    connection.commit()
    before = connection.total_changes
    with pytest.raises(NativeQualificationError, match='native revision|retained native qualification'):
        record_qualification(connection, IDENTITY)
    with pytest.raises(NativeQualificationError):
        validate_qualification(connection, IDENTITY)
    assert connection.total_changes == before
    connection.close()


def _compact_cycle(connection, journal, *, reference=None, states=None):
    cycle_id = f"compact-cycle-{connection.execute('SELECT max(seq) FROM ledger').fetchone()[0]}"
    append_ledger(connection, "NATIVE_SERVICE_CYCLE_STARTED", {
        "cycle_id": cycle_id, "runtime_identity_digest": IDENTITY,
    })
    pipeline = {
        "source_portfolio_ref": journal.portfolio_reference(journal.portfolio)
        if reference is None else reference,
        "revision_states": {"EVIDENCE_HOLD": 1} if states is None else states,
        "unclassified_revisions": 0,
    }
    append_ledger(connection, "NATIVE_SERVICE_CYCLE_TERMINAL", {
        "runtime_identity_digest": IDENTITY, "cycle_id": cycle_id,
        "outcome": "COMPLETE", "failure_class": None, "pipeline": pipeline,
    })
    connection.commit()


def test_compact_qualification_replays_after_restart_and_later_portfolio(tmp_path):
    path = tmp_path / "compact-reopen.sqlite3"
    connection = _open(path)
    journal = _cycle(connection)
    legacy = record_qualification(connection, IDENTITY)
    assert validate_qualification(connection, IDENTITY) == legacy
    _compact_cycle(connection, journal)
    compact = record_qualification(connection, IDENTITY)
    assert compact.source_inventory_digest == legacy.source_inventory_digest
    assert validate_qualification(connection, IDENTITY) == compact
    # Future source inventory does not rebind the original qualification.
    _portfolio(journal, next(iter(journal.units.values()))[0], source_override={
        "observations": (("https://fixture.example/new", "sha256:" + "b" * 64,
                          "new-admission", "new-access"),),
    })
    connection.close()
    connection = connect(str(path))
    try:
        assert validate_qualification(connection, IDENTITY) == compact
        reopened = NativeRevisionJournal(connection)
        _compact_cycle(connection, reopened)
        changed = record_qualification(connection, IDENTITY)
        assert changed.source_inventory_digest != compact.source_inventory_digest
        assert validate_qualification(connection, IDENTITY) == changed
    finally:
        connection.close()


@pytest.mark.parametrize("case", [
    "missing", "wrong-kind", "future", "stale", "wrong-digest", "bool-seq",
    "extra-field", "count-mismatch", "same-count-mutation",
])
def test_compact_qualification_rejects_inexact_portfolio_reference(tmp_path, case):
    connection = _open(tmp_path / "bad-reference.sqlite3")
    try:
        journal = _cycle(connection)
        reference = journal.portfolio_reference(journal.portfolio)
        if case == "wrong-kind":
            seq, digest = connection.execute(
                "SELECT seq,payload_digest FROM ledger WHERE kind='NATIVE_REVISION_LANDED'"
            ).fetchone()
            reference = {"seq": seq, "payload_digest": digest}
        elif case == "missing":
            reference = {"seq": 999999, "payload_digest": reference["payload_digest"]}
        elif case == "future":
            reference = {"seq": connection.execute("SELECT max(seq) FROM ledger").fetchone()[0] + 3,
                         "payload_digest": reference["payload_digest"]}
        elif case in {"stale", "same-count-mutation"}:
            sources = json.loads(canonical_json_bytes(journal.portfolio))
            if case == "same-count-mutation":
                sources[0]["observations"] = [["https://fixture.example/changed",
                    "sha256:" + "b" * 64, "admission", "access"]]
            # Even an equal-value older reference is not the latest portfolio.
            journal._retain("NATIVE_SOURCE_PORTFOLIO", {"sources": sources})
        elif case == "wrong-digest":
            reference = {**reference, "payload_digest": "sha256:" + "0" * 64}
        elif case == "bool-seq":
            reference = {**reference, "seq": True}
        elif case == "extra-field":
            reference = {**reference, "ordinal": 1}
        _compact_cycle(connection, journal, reference=reference,
                       states={"EVIDENCE_HOLD": 2} if case == "count-mismatch" else None)
        if case == "future":
            journal._retain("NATIVE_SOURCE_PORTFOLIO", {"sources": list(journal.portfolio)})
        with pytest.raises(NativeQualificationError):
            record_qualification(connection, IDENTITY)
    finally:
        connection.close()


@pytest.mark.parametrize("change", ["missing", "kind", "digest", "bytes", "noncanonical"])
def test_compact_reference_rechecks_exact_row_after_authenticated_inventory(tmp_path, change):
    from newsroom.control_plane.native_qualification import _ledger, _service_cycle, _terminal_portfolio

    connection = _open(tmp_path / "changed-reference.sqlite3")
    try:
        journal = _cycle(connection)
        _compact_cycle(connection, journal)
        rows = _ledger(connection)
        _, _, terminal, payload = _service_cycle(rows, IDENTITY)
        seq = payload["pipeline"]["source_portfolio_ref"]["seq"]
        if change == "missing":
            connection.execute("DELETE FROM ledger WHERE seq=?", (seq,))
        elif change == "kind":
            connection.execute("UPDATE ledger SET kind='OTHER' WHERE seq=?", (seq,))
        elif change == "digest":
            connection.execute("UPDATE ledger SET digest=? WHERE seq=?", ("sha256:" + "0" * 64, seq))
        elif change == "bytes":
            connection.execute("UPDATE ledger SET payload_json='{}' WHERE seq=?", (seq,))
        else:
            connection.execute("UPDATE ledger SET payload_json=payload_json || ' ' WHERE seq=?", (seq,))
        with pytest.raises(NativeQualificationError):
            _terminal_portfolio(connection, rows, terminal, payload["pipeline"])
    finally:
        connection.close()


@pytest.mark.parametrize("missing_observation", (False, True))
def test_compact_reference_preserves_full_source_hold_validation(tmp_path, missing_observation):
    connection = _open(tmp_path / "compact-hold.sqlite3")
    try:
        held = _content_hold("SOURCE_ITEM_METADATA_HOLD")
        if missing_observation:
            held["observations"] = ()
        journal = _cycle(connection, source_override=held)
        _compact_cycle(connection, journal)
        if missing_observation:
            with pytest.raises(NativeQualificationError):
                record_qualification(connection, IDENTITY)
        else:
            retained = record_qualification(connection, IDENTITY)
            assert validate_qualification(connection, IDENTITY) == retained
    finally:
        connection.close()



def test_terminal_inventory_checks_all_inline_holds_without_cold_pair_reads(tmp_path, monkeypatch):
    from newsroom.control_plane.native_qualification import _revision_inventory
    from newsroom.tests.test_native_progress import _retrieval_facts

    connection = connect(str(tmp_path / "terminal-inline.sqlite3"))
    journal = NativeRevisionJournal(connection)
    unit = _native("terminal-inline")
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts={
        **_retrieval_facts(), "reason": "NO_QUALIFYING_NEW_INFORMATION",
        "unknown_inline": {"future": True},
    })
    monkeypatch.setattr(NativeRevisionJournal, "current", lambda *_: pytest.fail("qualification expanded a cold pair"))
    try:
        retained, states = _revision_inventory(connection, (), {"EVIDENCE_HOLD": 1})
        assert states == {"EVIDENCE_HOLD": 1}
        assert retained.summary(unit.revision_id)["facts"]["unknown_inline"] == {"future": True}
    finally:
        connection.close()


def test_legitimate_new_current_state_invalidates_old_qualification_binding(tmp_path):
    connection = _open(tmp_path / 'new-current-state.sqlite3')
    try:
        _cycle(connection)
        record_qualification(connection, IDENTITY)
        journal = NativeRevisionJournal(connection)
        revision = next(iter(journal.units))
        current = journal.current(revision)
        journal.advance(revision, stage=current['stage'], facts={
            **current['facts'], 'reason': 'A_NEW_SOURCE_LOCAL_HOLD',
        })
        with pytest.raises(NativeQualificationError, match='qualification|inventory'):
            validate_qualification(connection, IDENTITY)
    finally:
        connection.close()
