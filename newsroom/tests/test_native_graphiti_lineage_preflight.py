"""Fence divergent private attempts without changing their unknown accounting."""

from contextlib import contextmanager
from dataclasses import replace

import pytest

from newsroom.extraction.types import ExtractionRunId, ExtractionRunVersionId, FixtureExtractionCase
from newsroom.graphiti_adapter import GraphitiInputManifest, GraphitiInputManifestId
from newsroom.graphiti_adapter.identity import attempt_ids, typed_id
from newsroom.control_plane.model_usage import ModelUsageService
from newsroom.control_plane.store import (
    insert_graphiti_attempt_receipt, next_graphiti_attempt_number,
    reconcile_graphiti_spend, reserve_graphiti_spend,
)
from newsroom.tests.extraction_4a_helpers import extraction_proof, run_request
from newsroom.tests.graphiti_adapter_4d_authority_helpers import (
    fake_attempt, open_graphiti_system, seed_graphiti_authority_fixture,
)
from newsroom.tests.test_native_graphiti import _native, _open


@contextmanager
def _authority_head_five(tmp_path, unit):
    state = seed_graphiti_authority_fixture(
        tmp_path / "governed", fixture_case=FixtureExtractionCase.RETRYABLE_FAILURE)
    initial = fake_attempt(state, fixture_case=FixtureExtractionCase.RETRYABLE_FAILURE)
    with open_graphiti_system(state, workspace_root=tmp_path / "workspace") as system:
        system.graphiti.register_configuration(initial.configuration, proof=extraction_proof())
        previous = None
        for number in range(1, 6):
            request = run_request(state,
                run_id=typed_id(ExtractionRunId, "run", unit.ingest_id),
                run_version_id=typed_id(ExtractionRunVersionId, "lineage-run-version", number),
                version_number=number,
                previous=None if previous is None else previous.run_version_id,
                key=f"lineage-run:{number}")
            attempt_id, workspace_id, cleanup_id = attempt_ids(unit.ingest_id, number)
            attempt = replace(initial, attempt_id=attempt_id, workspace_id=workspace_id,
                cleanup_receipt_id=cleanup_id, attempt_number=number,
                expected_previous_attempt_id=None if previous is None else previous.attempt_id,
                extraction_request=request,
                manifest=GraphitiInputManifest.from_run_request(
                    manifest_id=typed_id(GraphitiInputManifestId, "lineage-manifest", number),
                    configuration=initial.configuration, contract=initial.extraction_contract,
                    request=request), idempotency_key=f"lineage-attempt:{number}")
            previous = system.graphiti.execute_attempt(attempt, proof=extraction_proof())
        assert previous.attempt_number == 5 and not previous.outcome.terminal
        yield system, state, previous


def _reserve(connection, unit, number):
    return reserve_graphiti_spend(connection,
        spend_id=f"{unit.ingest_id}:{number}", ingest_id=unit.ingest_id,
        attempt_number=number, proving_run_id=unit.proving_run_id,
        generation_id="lineage-fixture", reserved_gbp_microunits=500_000,
        ceiling_gbp_microunits=None)


def _private_prefix(connection, unit, count):
    for number in range(1, count + 1):
        _reserve(connection, unit, number)
        accounting = reconcile_graphiti_spend(connection,
            spend_id=f"{unit.ingest_id}:{number}", embedding_usage=None)
        insert_graphiti_attempt_receipt(connection,
            ingest_id=unit.ingest_id, attempt_number=number, outcome="FAILED",
            receipt={"ingest_id": unit.ingest_id, "attempt_number": number,
                     "outcome": "FAILED", "failure_code": "PRODUCER_INTERNAL_ERROR",
                     "binding_failure_type": "GraphitiAdapterVersionConflict",
                     "accounting": accounting})
    connection.commit()


def _snapshot(connection):
    return {table: connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            for table in ("ledger", "unpublished_graphiti_spend",
                          "unpublished_graphiti_attempt_receipts", "unpublished_graphiti_failures")}


def test_real_head_five_private_eight_hold_before_reservation_and_stays_read_only(
    tmp_path, monkeypatch,
):
    unit = _native("lineage-diverged")
    with _authority_head_five(tmp_path, unit) as (authority, _state, head):
        calls = []
        def ingest(connection, **values):
            calls.append(values["units"])
            # The old controller reaches this reservation although authority head is five.
            _reserve(connection, unit, next_graphiti_attempt_number(connection, unit.ingest_id))
            connection.commit()
        private = tmp_path / "private"
        private.mkdir()
        processor, connection, _ = _open(private, monkeypatch, ingest=ingest,
            rights=lambda _: pytest.fail("lineage hold must precede dispatch rights"))
        processor._system.graphiti = authority.graphiti
        processor._proof = extraction_proof()
        try:
            ModelUsageService(str(private / "private.sqlite3"))
            _private_prefix(connection, unit, 8)
            assert next_graphiti_attempt_number(connection, unit.ingest_id) == 9
            before = _snapshot(connection)
            for cycle in ("lineage:first", "lineage:second"):
                outcome, = processor.advance((unit,), cycle_id=cycle)
                assert outcome.state == "GRAPHITI_HOLD"
                assert outcome.reason == "SETTLEMENT_PENDING:LINEAGE_CONFLICT"
                assert calls == []
                assert _snapshot(connection) == before
                assert connection.execute("SELECT count(*) FROM model_invocation_allocations").fetchone()[0] == 0
            reread, = authority.graphiti.attempt_history(head.run_id, limit=1, proof=extraction_proof())
            assert reread == head
            latest = connection.execute(
                "SELECT usage_basis,status,actual_usd_microunits FROM unpublished_graphiti_spend "
                "WHERE ingest_id=? AND attempt_number=8", (unit.ingest_id,)).fetchone()
            assert tuple(latest) == ("UNREPORTED", "UNRECONCILED", None)
        finally:
            connection.close()


@pytest.mark.parametrize(("receipts", "reserved", "selected"), ((4, 5, 5), (5, None, 6)))
def test_exact_head_reentry_and_matched_successor_keep_the_existing_ingest_path(
    tmp_path, monkeypatch, receipts, reserved, selected,
):
    unit = _native("lineage-matched")
    with _authority_head_five(tmp_path, unit) as (authority, _state, _head):
        calls = []
        def ingest(connection, **_values):
            calls.append(next_graphiti_attempt_number(connection, unit.ingest_id))
        private = tmp_path / "private"
        private.mkdir()
        processor, connection, _ = _open(private, monkeypatch, ingest=ingest)
        processor._system.graphiti = authority.graphiti
        processor._proof = extraction_proof()
        try:
            _private_prefix(connection, unit, receipts)
            if reserved is not None:
                _reserve(connection, unit, reserved)
                connection.commit()
            before = _snapshot(connection)
            outcome, = processor.advance((unit,), cycle_id="matched-lineage")
            assert calls == [selected]
            assert outcome.reason != "SETTLEMENT_PENDING:LINEAGE_CONFLICT"
            assert _snapshot(connection) == before
        finally:
            connection.close()


def test_reserved_attempt_without_receipt_uses_actual_cycle_selector_and_remains_unknown(
    tmp_path, monkeypatch,
):
    unit = _native("lineage-unreceipted-reservation")
    with _authority_head_five(tmp_path, unit) as (authority, _state, _head):
        private = tmp_path / "private"
        private.mkdir()
        processor, connection, _ = _open(private, monkeypatch,
            ingest=lambda *_args, **_values: pytest.fail("divergent reservation must not enter ingest"))
        processor._system.graphiti = authority.graphiti
        processor._proof = extraction_proof()
        try:
            _private_prefix(connection, unit, 5)
            _reserve(connection, unit, 8)
            connection.commit()
            # Receipt-only MAX+1 would be six; cycle actually re-enters reserved eight.
            assert next_graphiti_attempt_number(connection, unit.ingest_id) == 8
            before = _snapshot(connection)
            outcome, = processor.advance((unit,), cycle_id="reserved-lineage")
            assert outcome.reason == "SETTLEMENT_PENDING:LINEAGE_CONFLICT"
            assert _snapshot(connection) == before
        finally:
            connection.close()
