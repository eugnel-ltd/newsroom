"""Checked expiry preserves current authority and reserves obsolete command keys."""
import json
import sqlite3

import pytest

from newsroom.authority import AggregateId, InlinePayload, SemanticCommand
from newsroom.authority import audit_retention as retention
from newsroom.authority._event_store_read import _EventStoreReadMixin
from newsroom.authority.persistence import DiagnosticHistoryExpired, RetiredLedgerEventRecord
from newsroom.increment4.models import _stream_admitted_provenance
from .projection_b1_helpers import FAMILY_ID, open_projection_system, proof
from .test_retired_projection_audit import _seed


def _fixture(tmp_path, **kwargs):
    root = tmp_path / "newsroom"
    (root / "increment4/object_cas").mkdir(parents=True, mode=0o700)
    (root / "increment4").chmod(0o700)
    (root / "native").mkdir(mode=0o700)
    path = root / "increment4/authority.sqlite3"
    request, result = _seed(path, **kwargs)
    for name in retention._EXTERNAL_DATABASES:
        with sqlite3.connect(root / name) as conn:
            conn.execute("CREATE TABLE retained_receipts(payload BLOB)")
    return root, path, request, result


def _snapshot(path):
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        events = tuple(_EventStoreReadMixin._event_from_row(row) for row in conn.execute(
            "SELECT * FROM ledger_events ORDER BY ledger_seq"
        ))
        _, digest = _stream_admitted_provenance(
            entities=(), relations=(), events=events, through_ledger_seq=events[-1].ledger_seq,
        )
        return events, digest


def test_retirement_expires_chain_preserving_hash_keys_current_reads_and_reopen(tmp_path):
    root, path, request, result = _fixture(tmp_path)
    before, snapshot = _snapshot(path)
    with open_projection_system(path) as system:
        active_before = tuple(g for g in system.projections.generations(FAMILY_ID, proof=proof()) if g.state.value == "ACTIVE")
    report = retention.retire_native_projection_diagnostics(root, apply=True)
    assert report["committed"]
    assert report["retired_projection_chains"] == 1
    for table in ("authority_commands", "authority_payloads", "authority_audit_events", "authority_aggregate_versions", "authorization_requests"):
        assert report["deleted"][table] == 1
    after, digest = _snapshot(path)
    assert snapshot == digest
    assert [e.ledger_seq for e in before] == [e.ledger_seq for e in after]
    retired = next(e for e in after if e.event_id == str(result.authority_event_id))
    assert isinstance(retired, RetiredLedgerEventRecord)
    with open_projection_system(path) as system:
        assert active_before == tuple(g for g in system.projections.generations(FAMILY_ID, proof=proof()) if g.state.value == "ACTIVE")
        with pytest.raises(DiagnosticHistoryExpired, match="expired"):
            system.projections.record_delivery(request, proof=proof())
        with pytest.raises(DiagnosticHistoryExpired, match="expired"):
            system.events.provenance(str(result.authority_event_id), proof=proof())
        with pytest.raises(DiagnosticHistoryExpired, match="expired"):
            system.events.command_result(str(retired.command_id), proof=proof())
        assert any(isinstance(e, RetiredLedgerEventRecord) for e in system.events.after(0, limit=1000, proof=proof()))
    assert retention.retire_native_projection_diagnostics(root, apply=True)["retired_projection_chains"] == 0


@pytest.mark.parametrize("case", ["active", "building", "applied", "failure", "multiple"])
def test_retirement_excludes_non_disposable_delivery(tmp_path, case):
    root, path, _, _ = _fixture(tmp_path, retire=case not in {"active", "building"}, activate=case != "building", kind="ignored" if case in {"active", "building"} else case)
    before = _snapshot(path)
    assert retention.retire_native_projection_diagnostics(root, apply=True)["retired_projection_chains"] == 0
    assert _snapshot(path) == before
    with open_projection_system(path):
        pass


@pytest.mark.parametrize("reference", ["event", "command", "request", "attempt", "cas", "escaped"])
def test_retirement_honours_external_and_nested_reference_roots(tmp_path, reference):
    root, path, request, result = _fixture(tmp_path)
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT command_id,authorization_request_digest FROM ledger_events WHERE event_id=?", (str(result.authority_event_id),)).fetchone()
        attempt = conn.execute("SELECT delivery_attempt_id FROM projection_delivery_attempts WHERE generation_id=?", (str(request.generation_id),)).fetchone()[0]
    token = {"event": str(result.authority_event_id), "command": row[0], "request": row[1], "attempt": attempt, "cas": row[0], "escaped": row[0]}[reference]
    payload = json.dumps({"nested": [{"reference": token}]}).encode()
    if reference == "escaped":
        payload = payload.replace(token.encode(), "".join(f"\\u{ord(c):04x}" for c in token).encode())
    if reference == "cas":
        (root / "increment4/object_cas/receipt").write_bytes(payload)
    else:
        with sqlite3.connect(root / "unpublished_store.sqlite3") as conn:
            conn.execute("INSERT INTO retained_receipts VALUES (?)", (payload,))
    assert retention.retire_native_projection_diagnostics(root, apply=True)["retired_projection_chains"] == 0
    with open_projection_system(path) as system:
        assert system.projections.record_delivery(request, proof=proof()) == result


def test_reserved_namespace_key_denies_changed_generation_and_native_insert(tmp_path):
    root, path, request, result = _fixture(tmp_path)
    with sqlite3.connect(path) as conn:
        command = dict(zip([c[0] for c in conn.execute("SELECT * FROM authority_commands LIMIT 0").description], conn.execute("SELECT c.* FROM authority_commands c JOIN ledger_events e USING(command_id) WHERE e.event_id=?", (str(result.authority_event_id),)).fetchone(), strict=True))
    retention.retire_native_projection_diagnostics(root, apply=True)
    from dataclasses import replace
    with open_projection_system(path) as system:
        active = next(g for g in system.projections.generations(FAMILY_ID, proof=proof()) if g.state.value == "ACTIVE")
        with pytest.raises(DiagnosticHistoryExpired):
            system.projections.record_delivery(replace(request, generation_id=active.generation_id, expected_authority_version=active.authority_aggregate_version), proof=proof())
    command.update(command_id=str(AggregateId.new()), aggregate_id=str(AggregateId.new()))
    with sqlite3.connect(path) as conn, pytest.raises(sqlite3.IntegrityError, match="identity remains reserved"):
        conn.execute("INSERT INTO authority_commands(" + ",".join(command) + ") VALUES(" + ",".join("?" for _ in command) + ")", tuple(command.values()))


def test_retired_digest_corruption_is_rejected_on_reopen(tmp_path):
    root, path, _, _ = _fixture(tmp_path)
    retention.retire_native_projection_diagnostics(root, apply=True)
    with sqlite3.connect(path) as conn:
        sql = conn.execute("SELECT sql FROM sqlite_schema WHERE name='immutable_ledger_events_update'").fetchone()[0]
        conn.execute("DROP TRIGGER immutable_ledger_events_update")
        conn.execute("UPDATE ledger_events SET retired_record_digest=zeroblob(32) WHERE retired_header_digest IS NOT NULL")
        conn.execute(sql)
    from newsroom.authority import AuthorityPersistenceError
    with pytest.raises(AuthorityPersistenceError, match="retired.*digest"):
        open_projection_system(path)


def test_retirement_transaction_rolls_back_rows_and_guards(tmp_path, monkeypatch):
    root, path, request, result = _fixture(tmp_path)
    before = _snapshot(path)
    original = retention._fingerprint
    calls = 0
    def changed(paths):
        nonlocal calls
        calls += 1
        value = original(paths)
        return value if calls <= 2 else value + (("changed",),)
    monkeypatch.setattr(retention, "_fingerprint", changed)
    with pytest.raises(retention.AuditRetentionError, match="changed"):
        retention.retire_native_projection_diagnostics(root, apply=True)
    assert _snapshot(path) == before
    with open_projection_system(path) as system:
        assert system.projections.record_delivery(request, proof=proof()) == result


@pytest.mark.parametrize("kind", ["COMMAND", "EVENT"])
def test_retained_causation_pins_full_diagnostic_chain(tmp_path, kind):
    from newsroom.authority import CausationKind, CausationRef
    root, path, request, result = _fixture(tmp_path)
    with sqlite3.connect(path) as conn:
        command_id = conn.execute("SELECT command_id FROM ledger_events WHERE event_id=?", (str(result.authority_event_id),)).fetchone()[0]
    with open_projection_system(path) as system:
        system.commands.execute(SemanticCommand(
            command_type="candidate.fixture.write", aggregate_id=AggregateId.new(), expected_aggregate_version=0,
            payload=InlinePayload({"headline": "current causal root", "count": 1}), idempotency_key="causal-root",
            causation=CausationRef(CausationKind(kind), command_id if kind == "COMMAND" else str(result.authority_event_id)),
        ), proof=proof())
    assert retention.retire_native_projection_diagnostics(root, apply=True)["retired_projection_chains"] == 0
    with open_projection_system(path) as system:
        assert system.projections.record_delivery(request, proof=proof()) == result


def test_retained_foreign_key_consumer_pins_full_diagnostic_chain(tmp_path):
    from newsroom.authority.projection_retirement import select_candidates, protect_candidates
    _, path, _, result = _fixture(tmp_path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("CREATE TABLE fixture_business_consumer(event_id TEXT REFERENCES ledger_events(event_id))")
        conn.execute("INSERT INTO fixture_business_consumer VALUES(?)", (str(result.authority_event_id),))
        conn.execute("CREATE TEMP TABLE _audit_tokens(id TEXT PRIMARY KEY) WITHOUT ROWID")
        assert select_candidates(conn) == 1
        assert protect_candidates(conn) == 1
        assert conn.execute("SELECT count(*) FROM _retirement_candidates").fetchone()[0] == 0


def test_shared_authentication_parent_survives_chain_expiry(tmp_path, monkeypatch):
    from newsroom.authority import StaticAuthenticator, StaticPrincipal
    from . import test_retired_projection_audit as fixtures
    authenticator = StaticAuthenticator(credentials={"token-1": StaticPrincipal("principal.alpha")}, authority_domain="newsroom.authority")
    original_authenticate = authenticator.authenticate
    verified = None
    def shared(proof, *, now):
        nonlocal verified
        verified = verified or original_authenticate(proof, now=now)
        return verified
    monkeypatch.setattr(authenticator, "authenticate", shared)
    monkeypatch.setattr(fixtures, "open_projection_system", lambda path: open_projection_system(path, authenticator=authenticator))
    root, path, _, _ = _fixture(tmp_path)
    report = retention.retire_native_projection_diagnostics(root, apply=True)
    assert report["retired_projection_chains"] == 1
    assert report["deleted"]["authentication_contexts"] == 0
    with open_projection_system(path):
        pass


@pytest.mark.parametrize("fault", ["payload", "audit", "version", "authentication", "request", "decision", "source"])
def test_retirement_does_not_dispose_of_corrupt_candidate_authority(tmp_path, fault):
    from newsroom.authority import AuthorityPersistenceError
    root, path, _, result = _fixture(tmp_path)
    tables = {"payload": "authority_payloads", "audit": "authority_audit_events", "version": "authority_aggregate_versions",
              "authentication": "authentication_contexts", "request": "authorization_requests", "decision": "authorization_decisions", "source": "projection_delivery_states"}
    table = tables[fault]
    with sqlite3.connect(path) as conn:
        event = dict(zip([c[0] for c in conn.execute("SELECT * FROM ledger_events LIMIT 0").description], conn.execute("SELECT * FROM ledger_events WHERE event_id=?", (str(result.authority_event_id),)).fetchone(), strict=True))
        guards = conn.execute("SELECT name,sql FROM sqlite_schema WHERE type='trigger' AND tbl_name=? AND sql LIKE '%BEFORE UPDATE%'", (table,)).fetchall()
        for name, _ in guards:
            conn.execute(f'DROP TRIGGER "{name}"')
        key, value, field, new = {
            "payload": ("payload_id", event["payload_id"], "payload_bytes", b"{}"),
            "audit": ("command_id", event["command_id"], "detail_digest", "sha256:" + "0" * 64),
            "version": ("command_id", event["command_id"], "trust_scope", "PROPOSED"),
            "authentication": ("authentication_context_id", event["authentication_context_id"], "canonical_digest", "sha256:" + "0" * 64),
            "request": ("request_digest", event["authorization_request_digest"], "canonical_record_digest", "sha256:" + "0" * 64),
            "decision": ("authorization_decision_id", event["authorization_decision_id"], "canonical_digest", "sha256:" + "0" * 64),
            "source": ("last_authority_event_id", str(result.authority_event_id), "source_event_digest", "sha256:" + "0" * 64),
        }[fault]
        conn.execute(f"UPDATE {table} SET {field}=? WHERE {key}=?", (new, value))
        for _, sql in guards:
            conn.execute(sql)
    with pytest.raises((AuthorityPersistenceError, retention.AuditRetentionError)):
        retention.retire_native_projection_diagnostics(root, apply=True)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT retired_header_digest FROM ledger_events WHERE event_id=?", (str(result.authority_event_id),)).fetchone()[0] is None
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_dry_run_neither_expires_nor_reuses_identity(tmp_path):
    root, path, request, result = _fixture(tmp_path)
    before = _snapshot(path)
    report = retention.retire_native_projection_diagnostics(root)
    assert report["counts"]["retired_projection_chains"] == 1
    assert not report["committed"]
    assert _snapshot(path) == before
    with open_projection_system(path) as system:
        assert system.projections.record_delivery(request, proof=proof()) == result


@pytest.mark.parametrize("kind", ["COMMAND", "EVENT"])
def test_new_causation_requires_full_not_expired_provenance(tmp_path, kind):
    from newsroom.authority import CausationKind, CausationRef
    root, path, _, result = _fixture(tmp_path)
    with sqlite3.connect(path) as conn:
        command_id = conn.execute("SELECT command_id FROM ledger_events WHERE event_id=?", (str(result.authority_event_id),)).fetchone()[0]
    retention.retire_native_projection_diagnostics(root, apply=True)
    with open_projection_system(path) as system, pytest.raises(DiagnosticHistoryExpired):
        system.commands.execute(SemanticCommand(
            command_type="candidate.fixture.write", aggregate_id=AggregateId.new(), expected_aggregate_version=0,
            payload=InlinePayload({"headline": "new causal root", "count": 1}), idempotency_key="expired-causation",
            causation=CausationRef(CausationKind(kind), command_id if kind == "COMMAND" else str(result.authority_event_id)),
        ), proof=proof())


def test_expired_headers_never_substitute_for_required_mapping_or_watermark(tmp_path, monkeypatch):
    from newsroom.increment4 import models
    root, path, _, result = _fixture(tmp_path)
    retention.retire_native_projection_diagnostics(root, apply=True)
    events, _ = _snapshot(path)
    retired = next(e for e in events if e.event_id == str(result.authority_event_id))
    with pytest.raises(models.Increment4ProofContractError, match="watermark.*expired"):
        models._stream_admitted_provenance(entities=(), relations=(), events=(retired,), through_ledger_seq=retired.ledger_seq)
    monkeypatch.setattr(models, "_required_event_ids", lambda *_: {retired.event_id})
    with pytest.raises(models.Increment4ProofContractError, match="required mapping.*expired"):
        models._stream_admitted_provenance(entities=(), relations=(), events=events, through_ledger_seq=events[-1].ledger_seq)


def test_native_reservation_guard_uses_indexed_identity_lookups(tmp_path):
    _, path, _, _ = _fixture(tmp_path)
    with sqlite3.connect(path) as conn:
        trigger = conn.execute("SELECT sql FROM sqlite_schema WHERE name='authority_commands_retired_key_guard'").fetchone()[0]
        predicate = trigger.split("WHEN ", 1)[1].split("BEGIN ", 1)[0]
        for field in ("idempotency_namespace", "idempotency_key", "command_id"):
            predicate = predicate.replace("NEW." + field, "'fixture-value'")
        plans = [row[3] for row in conn.execute("EXPLAIN QUERY PLAN SELECT 1 WHERE " + predicate)]
        assert not any("SCAN ledger_events" in plan for plan in plans)
        assert any("idx_retired_command_key" in plan for plan in plans)
        assert sum("SEARCH ledger_events" in plan for plan in plans) == 2


def test_reference_scan_uses_actual_generated_and_virtual_selected_columns():
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE generated_fixture(value TEXT, size INTEGER GENERATED ALWAYS AS (length(value)) VIRTUAL)")
        conn.execute("INSERT INTO generated_fixture(value) VALUES('retained')")
        conn.execute("CREATE VIRTUAL TABLE external_fts_fixture USING fts5(body)")
        conn.execute("INSERT INTO external_fts_fixture(body) VALUES('current business')")
        first = retention._scan_business(conn)
        assert first["rows"] > 2
        assert retention._scan_business(conn) == first


def test_current_optional_delivery_can_consume_expired_routing_identity(tmp_path):
    from newsroom.projection import ProjectionDeliveryRequest, ProjectionDeliveryOutcome
    root, path, _, result = _fixture(tmp_path)
    retention.retire_native_projection_diagnostics(root, apply=True)
    events, _ = _snapshot(path)
    retired = next(e for e in events if e.event_id == str(result.authority_event_id))
    with open_projection_system(path) as system:
        active = next(g for g in system.projections.generations(FAMILY_ID, proof=proof()) if g.state.value == "ACTIVE")
        delivery = system.projections.record_delivery(ProjectionDeliveryRequest(
            active.generation_id, active.authority_aggregate_version, retired.ledger_seq,
            ProjectionDeliveryOutcome.IGNORED_OPTIONAL, "current-expired-route",
        ), proof=proof())
        assert str(delivery.source_event_id) == retired.event_id
        assert delivery.source_event_digest == retired.original_header_digest
    with open_projection_system(path):
        pass


def test_pinned_older_checkpoint_never_masquerades_as_current_expired_checkpoint(tmp_path):
    from newsroom.authority._projection_store import _ProjectionAuthorityStore
    root, path, request, _ = _fixture(tmp_path)
    # Creation remains a full provenance root, so its older checkpoint stays.
    retention.retire_native_projection_diagnostics(root, apply=True)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        assert conn.execute("SELECT 1 FROM projection_checkpoint_versions WHERE generation_id=?", (str(request.generation_id),)).fetchone() is not None
        with pytest.raises(DiagnosticHistoryExpired, match="checkpoint history expired"):
            _ProjectionAuthorityStore._checkpoint_seq(None, conn, str(request.generation_id))
