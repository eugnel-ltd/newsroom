"""Selection must traverse each generation's state range, not once per event."""
from __future__ import annotations

import json
import sqlite3

import pytest

from newsroom.authority.projection_retirement import select_candidates, protect_candidates
from .test_retired_projection_audit import _seed


def _query(connection):
    """Capture the production staging statement, not a reimplemented predicate."""
    select_candidates(connection)
    return connection.selection.split(" AS\n", 1)[1]


class CaptureSelection(sqlite3.Connection):
    selection = ""
    def execute(self, sql, parameters=()):
        if sql.startswith("CREATE TEMP TABLE _retirement_candidates AS"):
            self.selection = sql
        return super().execute(sql, parameters)


def _plan(connection, query):
    return tuple(row[3] for row in connection.execute("EXPLAIN QUERY PLAN " + query))


def _assert_generation_state_event_order(plan):
    primary = [item for item in plan if item.startswith(("SCAN g", "SEARCH g", "SCAN s", "SEARCH s", "SCAN e", "SEARCH e"))]
    assert primary[0].startswith("SCAN g")
    assert primary[1].startswith("SEARCH s USING PRIMARY KEY (generation_id=?)")
    assert primary[2].startswith("SEARCH e ") and "(event_id=?)" in primary[2]
    assert not any("idx_ledger_events_aggregate" in item for item in primary)


def test_real_schema_selection_drives_generations_then_state_ranges_then_event_keys(tmp_path, record_property):
    path = tmp_path / "authority.sqlite3"
    _seed(path)
    with sqlite3.connect(path, factory=CaptureSelection) as connection:
        query = _query(connection)
        plan = _plan(connection, query)
        _assert_generation_state_event_order(plan)
        original = query.replace("CROSS JOIN", "JOIN")
        baseline = _plan(connection, original)
        assert any(item.startswith("SEARCH e ") and "idx_ledger_events_aggregate" in item for item in baseline)
        assert connection.execute(query).fetchall() == connection.execute(original).fetchall()
        record_property("real_schema_original_plan", json.dumps(baseline))
        record_property("real_schema_state_first_plan", json.dumps(plan))


def test_protection_reads_generation_heads_and_reuses_candidate_membership_indexes(tmp_path):
    path = tmp_path / "authority.sqlite3"
    _seed(path)
    with sqlite3.connect(path) as connection:
        select_candidates(connection)
        statements = []
        connection.set_trace_callback(statements.append)
        protect_candidates(connection)
        head_statement = next(sql for sql in statements if sql.startswith(
            "DELETE FROM _retirement_candidates WHERE event_id IN ("))
        plan = _plan(connection, head_statement)
        assert any("SEARCH a USING PRIMARY KEY (aggregate_type=?)" in item for item in plan)
        assert not any(item == "SCAN e" for item in plan)
        for table, key, column in (
            ("authority_payloads", "payload_id", "payload_id"),
            ("authentication_contexts", "authentication_context_id", "authentication_context_id"),
            ("authorization_requests", "request_digest", "authorization_request_digest"),
            ("authorization_decisions", "authorization_decision_id", "authorization_decision_id"),
        ):
            query = f"SELECT 1 FROM {table} WHERE NOT ({key} IN (SELECT {column} FROM _retirement_candidates))"
            assert any(f"_retirement_candidate_{column} FOR IN-OPERATOR" in item
                       for item in _plan(connection, query))


def _cohort(count, density):
    """SQL-access-path fixture only; full retention authority has its real tests."""
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.executescript('''
        CREATE TABLE projection_generations(generation_id TEXT PRIMARY KEY,family_id TEXT,state TEXT) STRICT;
        CREATE TABLE ledger_events(ledger_seq INTEGER PRIMARY KEY,event_id TEXT NOT NULL UNIQUE,
            command_id TEXT,payload_id TEXT,authentication_context_id TEXT,
            authorization_request_digest TEXT,authorization_decision_id TEXT,
            aggregate_type TEXT,aggregate_id TEXT,aggregate_version INTEGER,
            event_type TEXT,retired_header_digest BLOB) STRICT;
        CREATE INDEX idx_ledger_events_aggregate ON ledger_events(aggregate_type,aggregate_id,aggregate_version);
        CREATE TABLE projection_delivery_states(generation_id TEXT,ledger_seq INTEGER,last_authority_event_id TEXT,
            current_outcome TEXT,required INTEGER,finalized INTEGER,attempt_count INTEGER,last_error_code TEXT,
            PRIMARY KEY(generation_id,ledger_seq)) WITHOUT ROWID,STRICT;
        CREATE TABLE projection_delivery_attempts(generation_id TEXT,ledger_seq INTEGER,attempt_number INTEGER,
            PRIMARY KEY(generation_id,ledger_seq,attempt_number)) WITHOUT ROWID,STRICT;
        CREATE TABLE projection_families(family_id TEXT PRIMARY KEY,definition_digest TEXT) STRICT;
        CREATE TABLE projection_family_complete_contracts(definition_digest TEXT PRIMARY KEY) WITHOUT ROWID,STRICT;
    ''')
    connection.executemany("INSERT INTO projection_families VALUES(?,?)", [("structural", "structural-definition"), ("complete", "complete-definition")])
    connection.execute("INSERT INTO projection_family_complete_contracts VALUES('complete-definition')")
    generations = [(f"retired-{i}", "structural", "RETIRED") for i in range(4)] + [
        ("active", "structural", "ACTIVE"), ("building", "structural", "BUILDING"),
        ("failed", "structural", "FAILED"), ("complete", "complete", "RETIRED"),
    ]
    connection.executemany("INSERT INTO projection_generations VALUES(?,?,?)", generations)
    expected = set()
    sequence = 0
    for generation, family, state in generations:
        for ordinal in range(1, count + 1):
            sequence += 1
            event = f"{generation}:event:{ordinal}"
            outcome, required, finalized, attempts, error = "IGNORED_OPTIONAL", 0, 1, 1, None
            event_type, aggregate_type, aggregate_id, retired = "projection.delivery.recorded", "projection_generation", generation, None
            if density == "sparse" and ordinal % 16:
                event_type = "projection.generation.transitioned"
            # Distinct demonstrations of every exclusion in the same rowsets.
            if ordinal == 1: outcome = "APPLIED"
            elif ordinal == 2: required = 1
            elif ordinal == 3: finalized = 0
            elif ordinal == 4: attempts = 2
            elif ordinal == 5: error = "FAILURE"
            elif ordinal == 6: retired = bytes(32)
            elif ordinal == 7: event_type = "other.event"
            elif ordinal == 8: aggregate_type = "other_aggregate"
            elif ordinal == 9: aggregate_id = "other-generation"
            elif ordinal == 10: connection.execute("INSERT INTO projection_delivery_attempts VALUES(?,?,2)", (generation, ordinal))
            connection.execute("INSERT INTO ledger_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                sequence, event, "command:" + event, "payload:" + event, "authentication:" + event,
                "request:" + event, "decision:" + event, aggregate_type, aggregate_id, ordinal, event_type, retired,
            ))
            connection.execute("INSERT INTO projection_delivery_states VALUES(?,?,?,?,?,?,?,?)", (
                generation, ordinal, event, outcome, required, finalized, attempts, error,
            ))
            if (state == "RETIRED" and family == "structural" and ordinal > 10
                    and (density == "dense" or ordinal % 16 == 0)):
                expected.add(event)
    return connection, expected


def _select_steps(connection, query):
    steps = [0]
    connection.set_progress_handler(lambda: steps.__setitem__(0, steps[0] + 100) or 0, 100)
    try:
        rows = connection.execute(query).fetchall()
    finally:
        connection.set_progress_handler(None, 0)
    return rows, steps[0]


@pytest.mark.parametrize("density", ["sparse", "dense"])
def test_sparse_and_dense_cohort_selection_is_linear_with_identical_rowsets(tmp_path, record_property, density):
    path = tmp_path / "authority.sqlite3"
    _seed(path)
    with sqlite3.connect(path, factory=CaptureSelection) as real:
        repaired = _query(real)
    original = repaired.replace("CROSS JOIN", "JOIN")
    before, after, sizes = [], [], []
    for count in (256, 512):
        connection, expected = _cohort(count, density)
        try:
            baseline_rows, baseline_steps = _select_steps(connection, original)
            repaired_rows, repaired_steps = _select_steps(connection, repaired)
            assert sorted(baseline_rows) == sorted(repaired_rows)
            assert {row[0] for row in repaired_rows} == expected
            _assert_generation_state_event_order(_plan(connection, repaired))
            assert any(item.startswith("SEARCH e ") and "idx_ledger_events_aggregate" in item for item in _plan(connection, original))
            before.append(baseline_steps); after.append(repaired_steps); sizes.append(len(repaired_rows))
        finally:
            connection.close()
    assert before[1] > before[0] * 3
    assert after[1] < after[0] * 2.5
    assert after[1] * 8 < before[1]
    record_property(density + "_baseline_vm_steps", json.dumps(before))
    record_property(density + "_state_first_vm_steps", json.dumps(after))
    record_property(density + "_selected_row_counts", json.dumps(sizes))
