from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from newsroom.authority import AuthorityPersistenceError, AuthoritySchemaError
from newsroom.authority._discovery_store import (
    _create_discovery_governing_producer_read_port,
)
from newsroom.discovery import (
    DecisionTerminality,
    DiscoveryContractError,
    DiscoveryGoverningProducerReadPort,
    DiscoverySemanticCollision,
    DiscoverySignalId,
    DiscoveryVersionConflict,
    GateDecisionId,
    GateOutcome,
    LeadDispositionDecisionId,
    NextAction,
    NextActionKind,
    ReasonReference,
    StructuredReason,
)
from newsroom.sources import (
    SourceDefinitionVersionId,
    SourceVersionConflict,
    VersionedPolicyRef,
)

from .check_3c_authority_helpers import proof, version_request
from .discovery_3d_authority_helpers import (
    DISPOSITION_ID,
    GATE_ID,
    LEAD_ID,
    SIGNAL_ID,
    exact_admission_request,
    exact_gate_request,
    exact_lead_request,
    exact_signal_request,
    open_discovery_system,
    scopes,
    seed_check_lineage,
)


def _seed_and_admit(database: Path):
    with open_discovery_system(database) as system:
        seed_check_lineage(system)
        return system.discovery.admit_signal_to_lead(
            exact_admission_request(),
            proof=proof(),
        )


def _trigger_sql(conn: sqlite3.Connection, name: str) -> str:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
        (name,),
    ).fetchone()
    assert row is not None and isinstance(row[0], str)
    return str(row[0])


def _transaction_connection(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("BEGIN IMMEDIATE")
    return connection


def test_governing_producer_read_port_is_private_and_transaction_bound(
    tmp_path: Path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    _seed_and_admit(database)
    with pytest.raises(DiscoveryContractError, match="authority-private"):
        DiscoveryGoverningProducerReadPort(object(), object())

    connection = _transaction_connection(database)
    try:
        port = _create_discovery_governing_producer_read_port(connection)
        connection.execute("COMMIT")
        with pytest.raises(DiscoveryContractError, match="transaction"):
            port.require_current_governing_producers((LEAD_ID,))
    finally:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        connection.close()


def test_governing_producer_read_port_returns_exact_ordered_closure(
    tmp_path: Path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    admitted = _seed_and_admit(database)
    connection = _transaction_connection(database)
    try:
        port = _create_discovery_governing_producer_read_port(connection)
        changes = connection.total_changes
        assert port.require_current_governing_producers((LEAD_ID,)) == (
            (admitted.lead, admitted.signal, admitted.gate),
        )
        assert connection.in_transaction and connection.total_changes == changes
        with pytest.raises(DiscoveryContractError):
            port.require_current_governing_producers((LEAD_ID, LEAD_ID))
    finally:
        connection.execute("ROLLBACK")
        connection.close()


def test_governing_producer_read_port_preserves_original_promoting_gate(
    tmp_path: Path,
) -> None:
    from newsroom.discovery import TimeValidity

    from .discovery_3d_helpers import reason

    database = tmp_path / "repromoted-governing-producer.sqlite3"
    admitted = _seed_and_admit(database)
    hold_id = GateDecisionId.parse("00000000-0000-4000-8000-000000007097")
    promoted_id = GateDecisionId.parse("00000000-0000-4000-8000-000000007098")
    with open_discovery_system(database) as system:
        system.discovery.decide_gate(
            replace(
                exact_gate_request(),
                decision_id=hold_id,
                decision_ordinal=2,
                previous_decision_id=GATE_ID,
                basis=replace(
                    exact_gate_request().basis,
                    policy_current=False,
                    operationally_executable=False,
                    time_validity=TimeValidity.CURRENT,
                ),
                outcome=GateOutcome.OPERATIONAL_HOLD,
                terminality=DecisionTerminality.PENDING_CONDITION,
                primary_reason=reason("OPS.POLICY_STALE"),
                next_action=NextAction(
                    NextActionKind.REVIEW,
                    "REVIEW_STALE_POLICY",
                    owner="discovery-operator",
                    instructions="Revalidate the current deterministic Gate policy.",
                ),
                idempotency_key="governing-port-repromotion-hold",
            ),
            proof=proof(),
        )
        system.discovery.decide_gate(
            replace(
                exact_gate_request(),
                decision_id=promoted_id,
                decision_ordinal=3,
                previous_decision_id=hold_id,
                idempotency_key="governing-port-repromotion-current",
            ),
            proof=proof(),
        )

    connection = _transaction_connection(database)
    try:
        port = _create_discovery_governing_producer_read_port(connection)
        assert port.require_current_governing_producers((LEAD_ID,)) == (
            (admitted.lead, admitted.signal, admitted.gate),
        )
    finally:
        connection.execute("ROLLBACK")
        connection.close()


def test_governing_producer_read_port_uses_exact_scoped_event_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from newsroom.authority import _discovery_store as private

    database = tmp_path / "scoped-events.sqlite3"
    admitted = _seed_and_admit(database)
    connection = _transaction_connection(database)
    event_ids: list[str] = []
    statements: list[str] = []
    connection.set_trace_callback(statements.append)
    monkeypatch.setattr(
        private._DiscoveryAuthorityStore,
        "_validate_relational_invariants",
        lambda *_: pytest.fail("scoped Discovery read used global relations"),
    )
    monkeypatch.setattr(
        private._DiscoveryAuthorityStore,
        "_validate_immutable_records",
        lambda *_: pytest.fail("scoped Discovery read used global records"),
    )
    try:
        port = _create_discovery_governing_producer_read_port(
            connection,
            validate_retained_event=event_ids.append,
        )
        assert port.require_current_governing_producers((LEAD_ID,)) == (
            (admitted.lead, admitted.signal, admitted.gate),
        )
        expected = {
            str(row[0])
            for table in (
                "discovery_signals",
                "discovery_gate_decisions",
                "news_leads",
                "discovery_watch_conditions",
                "lead_disposition_decisions",
            )
            for row in connection.execute(
                f"SELECT authority_event_id FROM {table}"
            )
        }
        assert set(event_ids) == expected
        traced = " ".join(statements).lower()
        assert "where e.aggregate_type=" in traced
        assert "pragma foreign_key_check" in traced
        assert "pragma foreign_key_check()" not in traced
    finally:
        connection.set_trace_callback(None)
        connection.execute("ROLLBACK")
        connection.close()


def test_governing_producer_read_port_propagates_exact_event_failure(
    tmp_path: Path,
) -> None:
    database = tmp_path / "exact-event-failure.sqlite3"
    _seed_and_admit(database)
    connection = _transaction_connection(database)
    signal_event_id = str(
        connection.execute(
            "SELECT authority_event_id FROM discovery_signals WHERE signal_id=?",
            (str(SIGNAL_ID),),
        ).fetchone()[0]
    )

    def validate(event_id: str) -> None:
        if event_id == signal_event_id:
            raise AuthorityPersistenceError("exact Discovery event differs")

    try:
        port = _create_discovery_governing_producer_read_port(
            connection,
            validate_retained_event=validate,
        )
        with pytest.raises(DiscoveryContractError, match="transaction read failed"):
            port.require_current_governing_producers((LEAD_ID,))
    finally:
        connection.execute("ROLLBACK")
        connection.close()


def test_governing_producer_read_port_rejects_currentness_and_offline_rewrite(
    tmp_path: Path,
) -> None:
    from newsroom.discovery import TimeValidity

    from .discovery_3d_helpers import reason

    current_db = tmp_path / "current.sqlite3"
    _seed_and_admit(current_db)
    hold = replace(
        exact_gate_request(),
        decision_id=GateDecisionId.parse(
            "00000000-0000-4000-8000-000000007096"
        ),
        decision_ordinal=2,
        previous_decision_id=GATE_ID,
        basis=replace(
            exact_gate_request().basis,
            policy_current=False,
            operationally_executable=False,
            time_validity=TimeValidity.CURRENT,
        ),
        outcome=GateOutcome.OPERATIONAL_HOLD,
        terminality=DecisionTerminality.PENDING_CONDITION,
        primary_reason=reason("OPS.POLICY_STALE"),
        next_action=NextAction(
            NextActionKind.REVIEW,
            "REVIEW_STALE_POLICY",
            owner="discovery-operator",
            instructions="Revalidate the current deterministic Gate policy.",
        ),
        idempotency_key="governing-port-current-gate",
    )
    with open_discovery_system(current_db) as system:
        system.discovery.decide_gate(hold, proof=proof())
    connection = _transaction_connection(current_db)
    try:
        port = _create_discovery_governing_producer_read_port(connection)
        with pytest.raises(DiscoveryContractError):
            port.require_current_governing_producers((LEAD_ID,))
    finally:
        connection.execute("ROLLBACK")
        connection.close()


    definition_db = tmp_path / "definition.sqlite3"
    _seed_and_admit(definition_db)
    with open_discovery_system(definition_db) as system:
        system.sources.record_definition_version(
            replace(
                version_request(),
                version_id=SourceDefinitionVersionId.parse(
                    "00000000-0000-4000-8000-000000006205"
                ),
                version_number=2,
                expected_previous_version_id=version_request().version_id,
                locator="fixture://increment-3d/governing-port-v2",
                change_reason="Exercise governing-producer currentness.",
                idempotency_key="governing-port-definition-v2",
            ),
            proof=proof(),
        )
    connection = _transaction_connection(definition_db)
    try:
        port = _create_discovery_governing_producer_read_port(connection)
        with pytest.raises(DiscoveryContractError):
            port.require_current_governing_producers((LEAD_ID,))
    finally:
        connection.execute("ROLLBACK")
        connection.close()

    tampered_db = tmp_path / "offline.sqlite3"
    _seed_and_admit(tampered_db)
    rewritten = replace(
        exact_signal_request(), purpose="SELF_CONSISTENT_OFFLINE_REWRITE"
    )
    with sqlite3.connect(tampered_db) as connection:
        triggers = {
            name: _trigger_sql(connection, name)
            for name in (
                "immutable_discovery_signals_update",
                "immutable_authority_payloads_update",
                "immutable_ledger_events_update",
            )
        }
        for name in triggers:
            connection.execute(f"DROP TRIGGER {name}")
        event = connection.execute(
            "SELECT authority_event_id FROM discovery_signals WHERE signal_id=?",
            (str(SIGNAL_ID),),
        ).fetchone()[0]
        payload_id = connection.execute(
            "SELECT payload_id FROM ledger_events WHERE event_id=?", (event,)
        ).fetchone()[0]
        connection.execute(
            "UPDATE discovery_signals SET purpose=?,semantic_digest=?,"
            "canonical_bytes=?,canonical_digest=? WHERE signal_id=?",
            (
                rewritten.purpose,
                rewritten.semantic_digest,
                rewritten.canonical_bytes,
                rewritten.digest,
                str(SIGNAL_ID),
            ),
        )
        connection.execute(
            "UPDATE authority_payloads SET payload_bytes=?,payload_digest=? "
            "WHERE payload_id=?",
            (rewritten.canonical_bytes, rewritten.digest, payload_id),
        )
        connection.execute(
            "UPDATE ledger_events SET payload_digest=? WHERE event_id=?",
            (rewritten.digest, event),
        )
        for sql in triggers.values():
            connection.execute(sql)
        connection.commit()
    connection = _transaction_connection(tampered_db)
    try:
        port = _create_discovery_governing_producer_read_port(connection)
        with pytest.raises(DiscoveryContractError):
            port.require_current_governing_producers((LEAD_ID,))
    finally:
        connection.execute("ROLLBACK")
        connection.close()

@pytest.mark.parametrize(
    ("trigger_name", "table", "column", "value"),
    (
        (
            "immutable_authority_commands_update",
            "authority_commands",
            "idempotency_key",
            "offline-command-rewrite",
        ),
        (
            "immutable_authority_commands_update",
            "authority_commands",
            "command_type",
            "offline.bypass",
        ),
        (
            "immutable_authority_audit_events_update",
            "authority_audit_events",
            "detail_digest",
            "sha256:" + "0" * 64,
        ),
    ),
)
def test_governing_producer_read_port_rejects_command_and_audit_tamper(
    tmp_path: Path,
    trigger_name: str,
    table: str,
    column: str,
    value: str,
) -> None:
    database = tmp_path / f"{table}.sqlite3"
    _seed_and_admit(database)
    with sqlite3.connect(database) as connection:
        trigger = _trigger_sql(connection, trigger_name)
        connection.execute(f"DROP TRIGGER {trigger_name}")
        connection.execute(
            f"UPDATE {table} SET {column}=? WHERE command_id=("
            "SELECT command_id FROM ledger_events WHERE event_id=("
            "SELECT authority_event_id FROM discovery_signals WHERE signal_id=?))",
            (value, str(SIGNAL_ID)),
        )
        connection.execute(trigger)
        connection.commit()
    connection = _transaction_connection(database)
    try:
        port = _create_discovery_governing_producer_read_port(connection)
        with pytest.raises(DiscoveryContractError):
            port.require_current_governing_producers((LEAD_ID,))
    finally:
        connection.execute("ROLLBACK")
        connection.close()


def test_governing_producer_read_port_rejects_split_security_binding(
    tmp_path: Path,
) -> None:
    database = tmp_path / "security-split.sqlite3"
    _seed_and_admit(database)
    with sqlite3.connect(database) as connection:
        trigger = _trigger_sql(connection, "immutable_authority_audit_events_update")
        connection.execute("DROP TRIGGER immutable_authority_audit_events_update")
        connection.execute(
            "UPDATE authority_audit_events SET "
            "authentication_context_id=(SELECT authentication_context_id FROM "
            "authority_audit_events WHERE event_type!='discovery.signal.admitted' LIMIT 1),"
            "authorization_request_digest=(SELECT authorization_request_digest FROM "
            "authority_audit_events WHERE event_type!='discovery.signal.admitted' LIMIT 1),"
            "authorization_decision_id=(SELECT authorization_decision_id FROM "
            "authority_audit_events WHERE event_type!='discovery.signal.admitted' LIMIT 1) "
            "WHERE command_id=(SELECT command_id FROM ledger_events WHERE event_id=("
            "SELECT authority_event_id FROM discovery_signals WHERE signal_id=?))",
            (str(SIGNAL_ID),),
        )
        connection.execute(trigger)
        connection.commit()
    connection = _transaction_connection(database)
    try:
        port = _create_discovery_governing_producer_read_port(connection)
        with pytest.raises(DiscoveryContractError):
            port.require_current_governing_producers((LEAD_ID,))
    finally:
        connection.execute("ROLLBACK")
        connection.close()



def test_signal_gate_lead_authority_replays_and_reopens(tmp_path: Path) -> None:
    database = tmp_path / "authority.sqlite3"
    with open_discovery_system(database) as system:
        seed_check_lineage(system)
        created = system.discovery.admit_signal_to_lead(
            exact_admission_request(), proof=proof()
        )
        replayed = system.discovery.admit_signal_to_lead(
            exact_admission_request(), proof=proof()
        )

        assert created.replayed is False
        assert replayed.replayed is True
        assert replayed.signal.event_id == created.signal.event_id
        assert replayed.gate.event_id == created.gate.event_id
        assert replayed.lead is not None and created.lead is not None
        assert replayed.lead.event_id == created.lead.event_id
        assert replayed.initial_disposition is not None
        assert created.initial_disposition is not None
        assert (
            replayed.initial_disposition.event_id
            == created.initial_disposition.event_id
        )
        assert system.discovery.signal(SIGNAL_ID, proof=proof()) == created.signal
        assert system.discovery.current_gate(SIGNAL_ID, proof=proof()) == created.gate
        assert system.discovery.lead(LEAD_ID, proof=proof()) == created.lead
        assert (
            system.discovery.current_disposition(LEAD_ID, proof=proof())
            == created.initial_disposition
        )
        status = system.discovery.current_status(SIGNAL_ID, proof=proof())
        assert status.lead == created.lead
        assert status.current_disposition == created.initial_disposition
        assert system.discovery.signals_for_revision(
            created.signal.request.revision_id,
            limit=10,
            proof=proof(),
        ) == (created.signal,)

    with open_discovery_system(database) as reopened:
        assert reopened.discovery.signal(SIGNAL_ID, proof=proof()).event_id == (
            created.signal.event_id
        )
        assert reopened.discovery.current_gate(
            SIGNAL_ID, proof=proof()
        ).request.decision_id == GATE_ID
        assert reopened.discovery.lead_for_signal(
            SIGNAL_ID, proof=proof()
        ).request.lead_id == LEAD_ID
        assert reopened.discovery.current_disposition(
            LEAD_ID, proof=proof()
        ).request.decision_id == DISPOSITION_ID


def test_duplicate_gate_cannot_collapse_distinct_signal_purpose(
    tmp_path: Path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    second_signal_id = DiscoverySignalId.parse(
        "00000000-0000-4000-8000-000000007098"
    )
    with open_discovery_system(database) as system:
        seed_check_lineage(system)
        system.discovery.admit_signal(exact_signal_request(), proof=proof())
        second_signal = replace(
            exact_signal_request(),
            signal_id=second_signal_id,
            purpose="SECONDARY_SOURCE_TRANSITION_PURPOSE",
            discriminator="SECONDARY_DETERMINISTIC_DISCRIMINATOR",
            idempotency_key="distinct-purpose-signal",
        )
        system.discovery.admit_signal(second_signal, proof=proof())

        duplicate_basis = replace(
            exact_gate_request().basis,
            duplicate_signal_id=SIGNAL_ID,
            duplicate_rule=VersionedPolicyRef("fixture-duplicate", "v1"),
        )
        duplicate_gate = replace(
            exact_gate_request(),
            decision_id=GateDecisionId.parse(
                "00000000-0000-4000-8000-000000007097"
            ),
            signal_id=second_signal_id,
            basis=duplicate_basis,
            outcome=GateOutcome.SUPPRESSED_DUPLICATE,
            primary_reason=StructuredReason(
                code="NOVELTY.EXACT_DUPLICATE",
                basis=exact_gate_request().primary_reason.basis,
                references=(
                    ReasonReference("DISCOVERY_SIGNAL", str(SIGNAL_ID)),
                ),
                explanation=(
                    "A distinct deterministic Signal purpose must not be "
                    "collapsed as an exact duplicate."
                ),
            ),
            next_action=NextAction(
                NextActionKind.CLOSE,
                "CLOSE_EXACT_DUPLICATE",
                instructions="Close only a genuinely equivalent Signal slot.",
            ),
            idempotency_key="distinct-purpose-duplicate-gate",
        )
        with pytest.raises(DiscoveryVersionConflict, match="distinct-purpose"):
            system.discovery.decide_gate(duplicate_gate, proof=proof())


def test_duplicate_gate_requires_an_earlier_signal_authority_event(
    tmp_path: Path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    later_signal_id = DiscoverySignalId.parse(
        "00000000-0000-4000-8000-000000007096"
    )
    with open_discovery_system(database) as system:
        seed_check_lineage(system)
        system.discovery.admit_signal(exact_signal_request(), proof=proof())
        later_signal = replace(
            exact_signal_request(),
            signal_id=later_signal_id,
            admission_policy=VersionedPolicyRef(
                "fixture-signal-admission",
                "v2",
            ),
            idempotency_key="later-equivalent-signal",
        )
        system.discovery.admit_signal(later_signal, proof=proof())

        reverse_duplicate_basis = replace(
            exact_gate_request().basis,
            duplicate_signal_id=later_signal_id,
            duplicate_rule=VersionedPolicyRef("fixture-duplicate", "v1"),
        )
        reverse_duplicate_gate = replace(
            exact_gate_request(),
            decision_id=GateDecisionId.parse(
                "00000000-0000-4000-8000-000000007095"
            ),
            basis=reverse_duplicate_basis,
            outcome=GateOutcome.SUPPRESSED_DUPLICATE,
            primary_reason=StructuredReason(
                code="NOVELTY.EXACT_DUPLICATE",
                basis=exact_gate_request().primary_reason.basis,
                references=(
                    ReasonReference(
                        "DISCOVERY_SIGNAL",
                        str(later_signal_id),
                    ),
                ),
                explanation=(
                    "An earlier Signal cannot be closed in favour of a later "
                    "Signal authority event."
                ),
            ),
            next_action=NextAction(
                NextActionKind.CLOSE,
                "CLOSE_EXACT_DUPLICATE",
                instructions="Duplicate direction must follow authority order.",
            ),
            idempotency_key="reverse-duplicate-gate",
        )
        with pytest.raises(DiscoveryVersionConflict, match="earlier retained Signal"):
            system.discovery.decide_gate(reverse_duplicate_gate, proof=proof())


def test_signal_semantic_collision_and_gate_head_conflict_roll_back(
    tmp_path: Path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    with open_discovery_system(database) as system:
        seed_check_lineage(system)
        system.discovery.admit_signal(exact_signal_request(), proof=proof())

        equivalent = replace(
            exact_signal_request(),
            signal_id=DiscoverySignalId.parse(
                "00000000-0000-4000-8000-000000007099"
            ),
            idempotency_key="equivalent-signal-different-id",
        )
        with pytest.raises(DiscoverySemanticCollision):
            system.discovery.admit_signal(equivalent, proof=proof())

        system.discovery.decide_gate(exact_gate_request(), proof=proof())
        wrong_head = replace(
            exact_gate_request(),
            decision_id=GateDecisionId.parse(
                "00000000-0000-4000-8000-000000007098"
            ),
            decision_ordinal=2,
            previous_decision_id=GateDecisionId.parse(
                "00000000-0000-4000-8000-000000007097"
            ),
            idempotency_key="wrong-gate-head",
        )
        with pytest.raises(DiscoveryVersionConflict, match="current head"):
            system.discovery.decide_gate(wrong_head, proof=proof())

    with sqlite3.connect(database) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM discovery_signals"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM discovery_gate_decisions"
        ).fetchone()[0] == 1


def test_discovery_write_and_read_scopes_are_separate(tmp_path: Path) -> None:
    missing_write = tmp_path / "missing-write.sqlite3"
    with open_discovery_system(
        missing_write,
        granted_scopes=scopes() - {"authority.discovery.signals.admit"},
    ) as system:
        seed_check_lineage(system)
        with pytest.raises(PermissionError):
            system.discovery.admit_signal(exact_signal_request(), proof=proof())

    database = tmp_path / "read-scopes.sqlite3"
    _seed_and_admit(database)
    with open_discovery_system(
        database,
        granted_scopes=scopes() - {"authority.discovery.read_sensitive"},
    ) as system:
        assert system.discovery.current_status(SIGNAL_ID, proof=proof()).lead is not None
        with pytest.raises(PermissionError):
            system.discovery.signal(SIGNAL_ID, proof=proof())
        with pytest.raises(PermissionError):
            system.discovery.latest_disposition(LEAD_ID, proof=proof())

    with open_discovery_system(
        database,
        granted_scopes=scopes() - {"authority.discovery.read"},
    ) as system:
        with pytest.raises(PermissionError):
            system.discovery.current_status(SIGNAL_ID, proof=proof())


def test_discovery_read_limits_are_policy_bounded(tmp_path: Path) -> None:
    database = tmp_path / "authority.sqlite3"
    _seed_and_admit(database)
    with open_discovery_system(database) as system:
        assert len(system.discovery.gates(SIGNAL_ID, limit=1, proof=proof())) == 1
        with pytest.raises(PermissionError):
            system.discovery.gates(SIGNAL_ID, limit=101, proof=proof())


def test_gate_revalidates_current_source_version(tmp_path: Path) -> None:
    database = tmp_path / "authority.sqlite3"
    with open_discovery_system(database) as system:
        seed_check_lineage(system)
        system.discovery.admit_signal(exact_signal_request(), proof=proof())
        second_version = replace(
            version_request(),
            version_id=SourceDefinitionVersionId.parse(
                "00000000-0000-4000-8000-000000006205"
            ),
            version_number=2,
            expected_previous_version_id=version_request().version_id,
            locator="fixture://increment-3d/maintained-guidance-v2",
            change_reason="Exercise stale Gate revalidation.",
            idempotency_key="fixture-check-source-version-v2",
        )
        system.sources.record_definition_version(second_version, proof=proof())
        with pytest.raises((DiscoveryVersionConflict, SourceVersionConflict), match="current"):
            system.discovery.decide_gate(exact_gate_request(), proof=proof())

        revalidated_gate = replace(
            exact_gate_request(),
            evaluated_definition_version_id=second_version.version_id,
            rights_decision_id=second_version.rights.rights_decision_id,
            rights_policy_version=second_version.rights.rights_policy_version,
            idempotency_key="fixture-gate-revalidated-v2",
        )
        current_lead = replace(
            exact_lead_request(),
            definition_version_id=second_version.version_id,
            idempotency_key="fixture-lead-current-v2",
        )
        result = system.discovery.admit_signal_to_lead(
            replace(
                exact_admission_request(),
                gate=revalidated_gate,
                lead=current_lead,
            ),
            proof=proof(),
        )
        assert result.gate.request.evaluated_definition_version_id == (
            second_version.version_id
        )
        assert result.lead is not None
        assert result.lead.request.definition_version_id == second_version.version_id


def test_signal_retains_exact_observation_after_source_version_advances(
    tmp_path: Path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    with open_discovery_system(database) as system:
        seed_check_lineage(system)
        second_version = replace(
            version_request(),
            version_id=SourceDefinitionVersionId.parse(
                "00000000-0000-4000-8000-000000006206"
            ),
            version_number=2,
            expected_previous_version_id=version_request().version_id,
            locator="fixture://increment-3d/maintained-guidance-v2-signal",
            change_reason="Advance source version before Signal admission.",
            idempotency_key="fixture-check-source-version-v2-signal",
        )
        system.sources.record_definition_version(second_version, proof=proof())

        signal = system.discovery.admit_signal(
            exact_signal_request(),
            proof=proof(),
        )
        assert signal.request.definition_version_id == version_request().version_id



def test_repromotion_requires_a_new_gate_bound_disposition(tmp_path: Path) -> None:
    from newsroom.discovery import TimeValidity
    from .discovery_3d_helpers import disposition_request, reason

    database = tmp_path / "authority.sqlite3"
    created = _seed_and_admit(database)
    assert created.lead is not None and created.initial_disposition is not None

    hold_id = GateDecisionId.parse(
        "00000000-0000-4000-8000-000000007097"
    )
    promote_id = GateDecisionId.parse(
        "00000000-0000-4000-8000-000000007098"
    )
    hold_gate = replace(
        exact_gate_request(),
        decision_id=hold_id,
        decision_ordinal=2,
        previous_decision_id=GATE_ID,
        basis=replace(
            exact_gate_request().basis,
            policy_current=False,
            operationally_executable=False,
            time_validity=TimeValidity.CURRENT,
        ),
        outcome=GateOutcome.OPERATIONAL_HOLD,
        terminality=DecisionTerminality.PENDING_CONDITION,
        primary_reason=reason("OPS.POLICY_STALE"),
        next_action=NextAction(
            NextActionKind.REVIEW,
            "REVIEW_STALE_POLICY",
            owner="discovery-operator",
            instructions="Revalidate the current deterministic Gate policy.",
        ),
        idempotency_key="gate-hold-before-repromotion",
    )
    revalidated_gate = replace(
        exact_gate_request(),
        decision_id=promote_id,
        decision_ordinal=3,
        previous_decision_id=hold_id,
        idempotency_key="gate-repromotion-v3",
    )
    replacement_disposition = replace(
        disposition_request(),
        decision_id=LeadDispositionDecisionId.parse(
            "00000000-0000-4000-8000-000000007099"
        ),
        gate_decision_id=promote_id,
        decision_ordinal=2,
        previous_decision_id=DISPOSITION_ID,
        idempotency_key="gate-bound-disposition-v2",
    )

    with open_discovery_system(database) as system:
        system.discovery.decide_gate(hold_gate, proof=proof())
        system.discovery.decide_gate(revalidated_gate, proof=proof())

        with pytest.raises(LookupError, match="current Lead Disposition"):
            system.discovery.current_disposition(LEAD_ID, proof=proof())
        assert system.discovery.latest_disposition(
            LEAD_ID, proof=proof()
        ) == created.initial_disposition

        prefix = system.discovery.current_status(SIGNAL_ID, proof=proof())
        assert prefix.lead == created.lead
        assert prefix.current_gate.request.decision_id == promote_id
        assert prefix.current_disposition is None
        assert prefix.action_source.value == "GATE_DECISION"
        assert prefix.phase.value == "LEAD_QUEUED"
        assert prefix.next_action == revalidated_gate.next_action

        retained = system.discovery.record_lead_disposition(
            replacement_disposition,
            proof=proof(),
        )
        completed = system.discovery.current_status(SIGNAL_ID, proof=proof())
        assert completed.current_disposition == retained
        assert system.discovery.latest_disposition(
            LEAD_ID, proof=proof()
        ) == retained
        assert completed.action_source.value == "LEAD_DISPOSITION"

def test_startup_rejects_signal_and_gate_head_tampering(tmp_path: Path) -> None:
    signal_db = tmp_path / "signal-tamper.sqlite3"
    _seed_and_admit(signal_db)
    with sqlite3.connect(signal_db) as conn:
        trigger = _trigger_sql(conn, "immutable_discovery_signals_update")
        conn.execute("DROP TRIGGER immutable_discovery_signals_update")
        conn.execute(
            "UPDATE discovery_signals SET purpose=? WHERE signal_id=?",
            ("TAMPERED_PURPOSE", str(SIGNAL_ID)),
        )
        conn.execute(trigger)
        conn.commit()
    with pytest.raises(AuthorityPersistenceError, match="Discovery Signal"):
        open_discovery_system(signal_db)

    head_db = tmp_path / "head-tamper.sqlite3"
    _seed_and_admit(head_db)
    with sqlite3.connect(head_db) as conn:
        trigger = _trigger_sql(conn, "gate_head_update_guard")
        conn.execute("DROP TRIGGER gate_head_update_guard")
        conn.execute(
            "UPDATE discovery_gate_decision_heads SET updated_at=?",
            ("2042-03-12T12:00:00.000000Z",),
        )
        conn.execute(trigger)
        conn.commit()
    with pytest.raises((AuthoritySchemaError, AuthorityPersistenceError)):
        open_discovery_system(head_db)


def test_crash_prefix_recovery_completes_missing_records(tmp_path: Path) -> None:
    database = tmp_path / "authority.sqlite3"
    plan = exact_admission_request()
    with open_discovery_system(database) as system:
        seed_check_lineage(system)
        signal = system.discovery.admit_signal(plan.signal, proof=proof())
        gate = system.discovery.decide_gate(plan.gate, proof=proof())
        assert signal.replayed is False and gate.replayed is False

        resumed = system.discovery.admit_signal_to_lead(plan, proof=proof())
        assert resumed.signal_state.value == "REPLAYED"
        assert resumed.gate_state.value == "REPLAYED"
        assert resumed.lead_state is not None
        assert resumed.lead_state.value == "CREATED"
        assert resumed.disposition_state is not None
        assert resumed.disposition_state.value == "CREATED"
        assert resumed.replayed is False


def test_later_operational_gate_blocks_new_watch_or_disposition(
    tmp_path: Path,
) -> None:
    from newsroom.discovery import TimeValidity
    from .discovery_3d_helpers import reason, watch_request, disposition_request

    database = tmp_path / "authority.sqlite3"
    created = _seed_and_admit(database)
    assert created.lead is not None and created.initial_disposition is not None

    hold_basis = replace(
        exact_gate_request().basis,
        policy_current=False,
        operationally_executable=False,
        time_validity=TimeValidity.CURRENT,
    )
    hold_gate = replace(
        exact_gate_request(),
        decision_id=GateDecisionId.parse(
            "00000000-0000-4000-8000-000000007096"
        ),
        decision_ordinal=2,
        previous_decision_id=GATE_ID,
        basis=hold_basis,
        outcome=GateOutcome.OPERATIONAL_HOLD,
        terminality=DecisionTerminality.PENDING_CONDITION,
        primary_reason=reason("OPS.POLICY_STALE"),
        next_action=NextAction(
            NextActionKind.REVIEW,
            "REVIEW_STALE_POLICY",
            owner="discovery-operator",
            instructions="Revalidate the current source and gate policy.",
        ),
        idempotency_key="second-gate-operational-hold",
    )
    with open_discovery_system(database) as system:
        system.discovery.decide_gate(hold_gate, proof=proof())
        with pytest.raises(DiscoveryVersionConflict, match="exact current promoting Gate"):
            system.discovery.record_watch_condition(watch_request(), proof=proof())
        later_disposition = replace(
            disposition_request(),
            decision_id=LeadDispositionDecisionId.parse(
                "00000000-0000-4000-8000-000000007095"
            ),
            decision_ordinal=2,
            previous_decision_id=DISPOSITION_ID,
            outcome=disposition_request().outcome,
            idempotency_key="later-disposition-after-hold",
        )
        with pytest.raises(DiscoveryVersionConflict, match="exact current promoting Gate"):
            system.discovery.record_lead_disposition(
                later_disposition, proof=proof()
            )


def test_discovery_read_boundary_rejects_untyped_identities_before_lookup(
    tmp_path: Path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    with open_discovery_system(database) as system:
        invalid_calls = (
            lambda: system.discovery.signal("signal", proof=proof()),
            lambda: system.discovery.current_gate("signal", proof=proof()),
            lambda: system.discovery.gates("signal", limit=1, proof=proof()),
            lambda: system.discovery.lead("lead", proof=proof()),
            lambda: system.discovery.lead_for_signal("signal", proof=proof()),
            lambda: system.discovery.watch_condition("watch", proof=proof()),
            lambda: system.discovery.disposition("disposition", proof=proof()),
            lambda: system.discovery.current_disposition("lead", proof=proof()),
            lambda: system.discovery.latest_disposition("lead", proof=proof()),
            lambda: system.discovery.dispositions("lead", limit=1, proof=proof()),
            lambda: system.discovery.current_status("signal", proof=proof()),
        )
        for call in invalid_calls:
            with pytest.raises(TypeError, match="identity must be typed"):
                call()


@pytest.mark.parametrize('historical_gates', [8, 64])
def test_current_producer_event_work_does_not_replay_historical_gates(
    tmp_path: Path, historical_gates: int,
) -> None:
    database = tmp_path / 'producer-cost.sqlite3'
    admitted = _seed_and_admit(database)
    previous = GATE_ID
    with open_discovery_system(database) as system:
        for ordinal in range(2, historical_gates + 2):
            decision_id = GateDecisionId.new()
            system.discovery.decide_gate(replace(
                exact_gate_request(), decision_id=decision_id,
                decision_ordinal=ordinal, previous_decision_id=previous,
                idempotency_key=f'producer-history-{ordinal}',
            ), proof=proof())
            previous = decision_id
    connection = _transaction_connection(database)
    try:
        event_ids = []
        port = _create_discovery_governing_producer_read_port(
            connection, validate_retained_event=event_ids.append,
        )
        result = port.require_current_governing_producers((LEAD_ID,))
        assert result == ((admitted.lead, admitted.signal, admitted.gate),)
        # Signal, Lead, immutable promoting Gate, current Gate and current
        # disposition; prior same-signal Gate history does not govern this read.
        assert len(event_ids) == len(set(event_ids)) == 5
        selected = {str(admitted.lead.event_id), str(admitted.signal.event_id), str(admitted.gate.event_id)}
        selected.add(str(connection.execute(
            'SELECT authority_event_id FROM discovery_gate_decisions WHERE decision_id=?',
            (str(previous),),
        ).fetchone()[0]))
        selected.add(str(connection.execute(
            'SELECT authority_event_id FROM lead_disposition_decisions WHERE decision_id=?',
            (str(DISPOSITION_ID),),
        ).fetchone()[0]))
        assert set(event_ids) == selected
    finally:
        connection.execute('ROLLBACK')
        connection.close()


@pytest.mark.parametrize('target', ['signal', 'lead', 'promoting_gate', 'current_gate', 'disposition', 'command', 'audit', 'head'])
def test_selected_producer_closure_rejects_corruption_with_real_event_validator(
    tmp_path: Path, target: str,
) -> None:
    database = tmp_path / f'producer-closure-{target}.sqlite3'
    _seed_and_admit(database)
    current_id = GateDecisionId.new()
    with open_discovery_system(database) as system:
        system.discovery.decide_gate(replace(
            exact_gate_request(), decision_id=current_id,
            decision_ordinal=2, previous_decision_id=GATE_ID,
            idempotency_key='producer-closure-current',
        ), proof=proof())
        store = system.discovery._GovernedDiscovery__signal.__self__._store
        connection = store._connection
        connection.execute('BEGIN IMMEDIATE')
        def offline_update(table, statement, parameters):
            triggers = connection.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='trigger' AND tbl_name=?",
                (table,),
            ).fetchall()
            for name, _ in triggers:
                connection.execute(f'DROP TRIGGER "{name}"')
            try:
                connection.execute(statement, parameters)
            finally:
                for _, sql in triggers:
                    connection.execute(sql)
        try:
            if target in {'command', 'audit'}:
                table = 'authority_commands' if target == 'command' else 'authority_audit_events'
                column = 'idempotency_key' if target == 'command' else 'detail_digest'
                value = 'changed-producer-command' if target == 'command' else 'sha256:' + '0' * 64
                where = 'command_id=(SELECT command_id FROM ledger_events WHERE event_id=(SELECT authority_event_id FROM discovery_signals WHERE signal_id=?))'
                key = str(SIGNAL_ID)
            elif target == 'head':
                offline_update('discovery_gate_decision_heads',
                    'UPDATE discovery_gate_decision_heads SET current_decision_ordinal=1 WHERE signal_id=?',
                    (str(SIGNAL_ID),),
                )
                table = None
            else:
                table, pk, key = {
                    'signal': ('discovery_signals', 'signal_id', str(SIGNAL_ID)),
                    'lead': ('news_leads', 'lead_id', str(LEAD_ID)),
                    'promoting_gate': ('discovery_gate_decisions', 'decision_id', str(GATE_ID)),
                    'current_gate': ('discovery_gate_decisions', 'decision_id', str(current_id)),
                    'disposition': ('lead_disposition_decisions', 'decision_id', str(DISPOSITION_ID)),
                }[target]
                column, value, where = 'canonical_digest', 'sha256:' + '0' * 64, f'{pk}=?'
            if table is not None:
                offline_update(table, f'UPDATE {table} SET {column}=? WHERE {where}', (value, key))
            port = _create_discovery_governing_producer_read_port(
                connection, validate_retained_event=store._validate_retained_event,
            )
            with pytest.raises(DiscoveryContractError):
                port.require_current_governing_producers((LEAD_ID,))
        finally:
            connection.execute('ROLLBACK')


@pytest.mark.parametrize('corrupt_watch', [False, True])
def test_selected_producer_includes_current_disposition_watch_closure(
    tmp_path: Path, corrupt_watch: bool,
) -> None:
    from newsroom.discovery import LeadDispositionOutcome
    from .discovery_3d_helpers import disposition_request, watch_request

    database = tmp_path / 'producer-watch.sqlite3'
    admitted = _seed_and_admit(database)
    with open_discovery_system(database) as system:
        watch = system.discovery.record_watch_condition(watch_request(), proof=proof())
        system.discovery.record_lead_disposition(replace(
            disposition_request(outcome=LeadDispositionOutcome.WATCH_DEFER),
            decision_id=LeadDispositionDecisionId.new(), decision_ordinal=2,
            previous_decision_id=DISPOSITION_ID, idempotency_key='producer-current-watch',
            urgency_route=exact_lead_request().urgency,
        ), proof=proof())
        store = system.discovery._GovernedDiscovery__signal.__self__._store
        connection = store._connection
        connection.execute('BEGIN IMMEDIATE')
        try:
            if corrupt_watch:
                trigger = _trigger_sql(connection, 'immutable_discovery_watch_conditions_update')
                connection.execute('DROP TRIGGER immutable_discovery_watch_conditions_update')
                connection.execute(
                    'UPDATE discovery_watch_conditions SET canonical_digest=? WHERE watch_condition_id=?',
                    ('sha256:' + '0' * 64, str(watch.request.watch_condition_id)),
                )
                connection.execute(trigger)
            seen = []
            def validate(event_id):
                seen.append(event_id)
                store._validate_retained_event(event_id)
            port = _create_discovery_governing_producer_read_port(connection, validate_retained_event=validate)
            if corrupt_watch:
                with pytest.raises(DiscoveryContractError):
                    port.require_current_governing_producers((LEAD_ID,))
            else:
                assert port.require_current_governing_producers((LEAD_ID,)) == (
                    (admitted.lead, admitted.signal, admitted.gate),
                )
                assert str(watch.event_id) in seen
                assert len(seen) == len(set(seen)) == 5
        finally:
            connection.execute('ROLLBACK')
