"""Retired ignored detail is dispensable only with exact retained authority."""
import sqlite3
import json
from datetime import timedelta

import pytest

from newsroom.authority import (
    AggregateId, AuthorityPersistenceError, InlinePayload, SemanticCommand, UtcTimestamp,
)
from .authority_helpers import FIXED_NOW
from newsroom.authority import audit_retention as retention
from newsroom.projection import (
    ProjectionDeliveryOutcome, ProjectionDeliveryRequest,
    ProjectionGenerationValidationRequest,
)
from .projection_b1_helpers import FAMILY_ID, open_projection_system, proof
from .test_projection_b3_authority import (
    GRAPH_DIGEST, SERVICE_DIGEST, _create, _promote, _register,
)


def _seed(path, *, retire=True, activate=True, kind="ignored", required=False, checkpoint_anchor=False):
    with open_projection_system(path) as system:
        _register(system)
        source_seq = 1
        if kind != "ignored":
            source_seq = system.commands.execute(SemanticCommand(
                command_type="source.item.write" if required else "candidate.fixture.write", aggregate_id=AggregateId.new(),
                expected_aggregate_version=0, payload=InlinePayload({"headline": "optional", "count": 1}),
                idempotency_key="optional-source",
            ), proof=proof()).ledger_seq
        outcomes = {
            "ignored": (ProjectionDeliveryOutcome.IGNORED_OPTIONAL,),
            "applied": (ProjectionDeliveryOutcome.APPLIED,),
            "failure": (ProjectionDeliveryOutcome.RETRYABLE_FAILURE,),
            "multiple": (ProjectionDeliveryOutcome.RETRYABLE_FAILURE, ProjectionDeliveryOutcome.IGNORED_OPTIONAL),
            "retried_applied": (ProjectionDeliveryOutcome.RETRYABLE_FAILURE, ProjectionDeliveryOutcome.APPLIED),
        }[kind]
        prior = None
        first_request = first_result = None
        for index in range(2 if retire else 1):
            generation = _create(system, f"retention-generation-{index}")
            def current():
                return next(g for g in system.projections.generations(FAMILY_ID, proof=proof())
                            if g.generation_id == generation.generation_id)
            for number, outcome in enumerate(outcomes):
                request = ProjectionDeliveryRequest(
                    generation.generation_id, current().authority_aggregate_version,
                    source_seq, outcome, f"delivery-{index}-{number}",
                    error_code="TRANSIENT" if outcome is ProjectionDeliveryOutcome.RETRYABLE_FAILURE else None,
                )
                result = system.projections.record_delivery(request, proof=proof())
                if first_request is None:
                    first_request, first_result = request, result
            if checkpoint_anchor:
                system.projections.record_delivery(ProjectionDeliveryRequest(
                    generation.generation_id, current().authority_aggregate_version,
                    1 if source_seq != 1 else 2, ProjectionDeliveryOutcome.IGNORED_OPTIONAL,
                    f"checkpoint-anchor-{index}",
                ), proof=proof())
            if not activate:
                continue
            with sqlite3.connect(path) as conn:
                checkpoint = conn.execute(
                    "SELECT contiguous_ledger_seq FROM projection_checkpoint_versions "
                    "WHERE generation_id=? ORDER BY checkpoint_version DESC LIMIT 1",
                    (str(generation.generation_id),),
                ).fetchone()[0]
            validation = system.projections.validate_generation(
                ProjectionGenerationValidationRequest(
                    generation.generation_id, current().authority_aggregate_version,
                    checkpoint, SERVICE_DIGEST, GRAPH_DIGEST, "RETENTION_TEST", f"validate-{index}",
                ), proof=proof(),
            )
            prior = _promote(system, current(), validation, f"promote-{index}", prior=prior).generation
    return first_request, first_result


def _remove_detail(path, request):
    with sqlite3.connect(path) as conn:
        trigger = conn.execute("SELECT sql FROM sqlite_master WHERE name='immutable_projection_delivery_attempt_delete'").fetchone()[0]
        conn.execute("DROP TRIGGER immutable_projection_delivery_attempt_delete")
        conn.execute("DELETE FROM projection_delivery_attempts WHERE generation_id=?",
                     (str(request.generation_id),))
        conn.execute(trigger)


def test_retired_single_ignored_delivery_reopens_and_replays_without_detail(tmp_path):
    path = tmp_path / "authority.sqlite3"
    request, result = _seed(path)
    _remove_detail(path, request)
    with open_projection_system(path) as system:
        assert system.projections.record_delivery(request, proof=proof()) == result


def test_active_delivery_still_requires_complete_attempt_detail(tmp_path):
    path = tmp_path / "authority.sqlite3"
    request, _ = _seed(path, retire=False)
    _remove_detail(path, request)
    with pytest.raises(AuthorityPersistenceError, match="not contiguous"):
        open_projection_system(path)


@pytest.mark.parametrize("fault", ["required", "count", "finalized", "error", "source", "payload"])
def test_reconstructed_retired_delivery_does_not_hide_corruption(tmp_path, fault):
    path = tmp_path / "authority.sqlite3"
    request, result = _seed(path)
    _remove_detail(path, request)
    with sqlite3.connect(path) as conn:
        triggers = conn.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' AND tbl_name IN ('projection_delivery_states','authority_payloads')").fetchall()
        for name, _ in triggers:
            conn.execute(f'DROP TRIGGER "{name}"')
        changes = {"required": "required=1", "count": "attempt_count=2",
                   "finalized": "finalized=0", "error": "last_error_code='FAILURE'",
                   "source": "source_event_digest='sha256:' || printf('%064d',0)"}
        if fault == "payload":
            conn.execute("UPDATE authority_payloads SET payload_bytes=? WHERE payload_id=(SELECT payload_id FROM ledger_events WHERE event_id=?)",
                         (b'{}', str(result.authority_event_id)))
        else:
            conn.execute("UPDATE projection_delivery_states SET " + changes[fault] + " WHERE generation_id=?",
                         (str(request.generation_id),))
        # Restore exact schema, so OPEN reaches data rather than DDL validation.
        for _, sql in triggers:
            conn.execute(sql)
    with pytest.raises(AuthorityPersistenceError):
        open_projection_system(path)


@pytest.mark.parametrize("referenced", [False, True])
def test_projection_detail_maintenance_preserves_summary_and_reference_closure(tmp_path, monkeypatch, referenced):
    root = tmp_path / "newsroom"
    (root / "increment4").mkdir(parents=True, mode=0o700)
    (root / "increment4/object_cas").mkdir(mode=0o700)
    (root / "native").mkdir(mode=0o700)
    path = root / "increment4/authority.sqlite3"
    request, result = _seed(path)
    with sqlite3.connect(path) as conn:
        old_id = conn.execute("SELECT delivery_attempt_id FROM projection_delivery_attempts WHERE generation_id=?",
                              (str(request.generation_id),)).fetchone()[0]
    for name in retention._EXTERNAL_DATABASES:
        with sqlite3.connect(root / name) as conn:
            conn.execute("CREATE TABLE retained_receipts(payload BLOB)")
            if referenced and name == "unpublished_store.sqlite3":
                conn.execute("INSERT INTO retained_receipts VALUES (?)", (json.dumps({"attempt": old_id}).encode(),))
    # This fixture exercises real projection authority. Native read profiles
    # have their separate full-runtime matrix; there are no read candidates here.
    monkeypatch.setattr(retention, "_policies", lambda conn: conn.execute(
        "CREATE TEMP TABLE _audit_policies(purpose TEXT PRIMARY KEY,class TEXT,digest TEXT)"
    ))
    report = retention.prune_native_diagnostic_audit(root, apply=True)
    assert report["projection_details_deleted"] == (0 if referenced else 1)
    with open_projection_system(path) as system:
        assert system.projections.record_delivery(request, proof=proof()) == result


@pytest.mark.parametrize("case,age_days,expected", [
    ("ignored", 6, 0), ("ignored", 7, 0), ("ignored", 8, 1),
    ("active", 8, 0), ("building", 8, 0),
    ("applied", 8, 0), ("failure", 8, 0), ("multiple", 8, 0),
])
def test_maintenance_selector_preserves_age_and_history_boundaries(tmp_path, case, age_days, expected):
    path = tmp_path / "authority.sqlite3"
    _seed(path, retire=case not in {"active", "building"},
          activate=case != "building", kind="ignored" if case in {"active", "building"} else case)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TEMP TABLE _audit_tokens(id TEXT PRIMARY KEY) WITHOUT ROWID")
        cutoff = UtcTimestamp(FIXED_NOW.value + timedelta(days=age_days - 7)).to_text()
        assert retention._projection_candidates(conn, cutoff=cutoff) == expected
