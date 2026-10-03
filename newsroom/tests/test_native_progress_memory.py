"""Replay shares immutable unit metadata, never authority containers or history."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane.corpus import units_from
from newsroom.control_plane.editorial import GroupedObservation
from newsroom.control_plane.items import SourceItem
from newsroom.control_plane.native_progress import LAND, NativeRevisionJournal, _landed_units, import_legacy_native_progress
from newsroom.control_plane.native_source_intake import NativeSourceDisposition
from newsroom.control_plane.store import append_ledger, connect
from newsroom.effective_revision import EffectiveRevisionIdentity
from newsroom.tests.test_native_progress import _retrieval_facts


def _complete_units():
    observed_at = "2026-09-26T12:00:00.000000Z"
    body = "Complete retained evidence from 香港🙂. " * 600
    observation = digest_bytes(body.encode())
    item = SourceItem("UK-01", "workbook", "Retained workbook headline", body,
                      "https://example.test/workbook")
    resolver = SimpleNamespace(
        resolve=lambda **kw: EffectiveRevisionIdentity(**kw, first_observed_at=observed_at),
        pull_first_observed_at=lambda **kw: observed_at,
    )
    return units_from(
        (GroupedObservation("UK-01", observation, item, observed_at),),
        proving_run_id="native-source:" + observation,
        effective_revision_resolver=resolver,
        rights_gate_reason="Retained source and current rights evidence",
    )


@pytest.mark.parametrize("encoding", ["legacy", "shared"])
def test_replay_shares_unit_strings_with_exact_bytes_and_isolated_authority(tmp_path, encoding):
    units = _complete_units()
    assert len(units) > 1
    value = {"revision_id": units[0].revision_id, "units": [asdict(unit) for unit in units]}
    if encoding == "shared":
        value["shared_body"] = units[0].body
        for unit in value["units"]:
            del unit["body"]
    connection = connect(str(tmp_path / "private.sqlite3"))
    try:
        connection.execute('DELETE FROM native_current_meta')
        append_ledger(connection, LAND, value)
        connection.commit()
        import_legacy_native_progress(connection)
        rows = connection.execute("SELECT * FROM ledger ORDER BY seq").fetchall()
        statements = []
        connection.set_trace_callback(statements.append)
        journal = NativeRevisionJournal(connection)
        connection.set_trace_callback(None)
        assert not any("FROM ledger" in statement for statement in statements)
        retained = journal.units[units[0].revision_id]
        assert retained == units
        assert _landed_units(json.loads(canonical_json_bytes(value))) == units
        first, second = retained[:2]
        assert first.headline is second.headline
        assert first.effective_revision.first_observed_at is second.effective_revision.first_observed_at
        assert first.authority.revision_id is second.authority.revision_id
        assert first.authority.records[0]["record_id"] is first.authority.definition_id
        assert first.authority.records[-1]["rights_gate_reason"] is second.authority.records[-1]["rights_gate_reason"]
        assert first.authority is not second.authority
        assert first.authority.records is not second.authority.records
        assert all(left is not right for left, right in zip(first.authority.records, second.authority.records))
        assert canonical_json_bytes([asdict(unit) for unit in retained]) == canonical_json_bytes([asdict(unit) for unit in units])
        first.authority.records[-1]["rights_gate_status"] = "HOLD"
        assert second.authority.records[-1]["rights_gate_status"] == "PASS"
        assert NativeRevisionJournal(connection).units[units[0].revision_id] == units
        assert connection.execute("SELECT * FROM ledger ORDER BY seq").fetchall() == rows
    finally:
        connection.close()


def test_shared_metadata_preserves_reference_closure_and_same_count_mutation_detection(tmp_path):
    units = _complete_units()
    revision = units[0].revision_id
    connection = connect(str(tmp_path / "private.sqlite3"))
    try:
        journal = NativeRevisionJournal(connection)
        journal.land(units)
        journal.advance(revision, stage="RETRIEVAL_COMPLETE", facts=_retrieval_facts())
        journal.advance(revision, stage="EVIDENCE_HOLD", facts=_retrieval_facts())
        reference = (units[0].canonical_url, units[0].observation_digest,
                     units[0].authority.admission_id, units[0].authority.access_decision_id)
        journal.sources((NativeSourceDisposition("UK-01", "HOLD", "CURRENT_RIGHTS_HOLD", observations=(reference,)),))
        journal.sources((NativeSourceDisposition("UK-01", "READY", "UNCHANGED"),))
        portfolio_reference = journal.portfolio_reference(journal.portfolio)
        rows = connection.execute("SELECT * FROM ledger ORDER BY seq").fetchall()
        reopened = NativeRevisionJournal(connection)
        assert reopened.current(revision) == journal.current(revision)
        assert reopened.observations == {reference[1]: reference}
        assert reopened.portfolio_reference(reopened.portfolio) == portfolio_reference
        assert reopened.units[revision][0].coverage_first_observed_at == units[0].coverage_first_observed_at
        changed = (replace(reopened.units[revision][0], body=units[0].body + " changed"), *reopened.units[revision][1:])
        with pytest.raises(ValueError, match="chunk coverage"):
            reopened.land(changed)
        assert connection.execute("SELECT * FROM ledger ORDER BY seq").fetchall() == rows
        facts = reopened.current(revision)["facts"]
        facts["retrieval_binding"]["request"]["nodes"][0] = "Changed same-count evidence"
        result = reopened.advance(revision, stage="EVIDENCE_HOLD", facts=facts)
        assert result["ordinal"] == 3
        encoded = json.loads(connection.execute("SELECT payload_json FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()[0])
        assert "retrieval_facts_ref" not in encoded
        assert NativeRevisionJournal(connection).current(revision) == reopened.current(revision)
    finally:
        connection.close()


def test_cold_current_retrieval_pairs_do_not_remain_in_journal_heap(tmp_path):
    import gc
    import tracemalloc
    from newsroom.control_plane.native_progress import STATE
    from newsroom.tests.test_native_graphiti import _native

    connection = connect(str(tmp_path / "cold-pairs.sqlite3"))
    inventory = [{
        "revision_id": f"revision-{number}", "ingest_id": f"ingest-{number}",
        "state": "INCLUDED", "reason": None,
        "document_digest": "sha256:" + f"{number:064x}",
        "current_rights_digest": "sha256:" + f"{number + 1000:064x}",
    } for number in range(360)]
    facts = {**_retrieval_facts(), "retrieval_rights_inventory": inventory}
    expected_pair_bytes = len(canonical_json_bytes(facts)) * 24
    revisions = []
    connection.execute('DELETE FROM native_current_meta')
    for number in range(24):
        unit = _native(f"cold-{number}")
        revisions.append(unit.revision_id)
        append_ledger(connection, LAND, {"revision_id": unit.revision_id, "units": [asdict(unit)]})
        append_ledger(connection, STATE, {
            "revision_id": unit.revision_id, "ordinal": 1,
            "stage": "EVIDENCE_HOLD", "facts": facts,
        })
    connection.commit()
    import_legacy_native_progress(connection)
    gc.collect()
    tracemalloc.start()
    try:
        journal = NativeRevisionJournal(connection)
        gc.collect()
        retained, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Only the journal replay allocation is measured, after its existing string
    # sharing. The input fixture and SQLite storage pre-date tracing.
    assert retained < expected_pair_bytes / 3
    for revision in revisions:
        assert journal.current(revision)["facts"] == facts
        assert "retrieval_rights_inventory" not in journal.summary(revision)["facts"]
    connection.close()


def test_equal_selected_bodies_preserve_bytes_and_detach_authority(tmp_path):
    from newsroom.tests.test_native_graphiti import _native

    connection = connect(str(tmp_path / "equal-bodies.sqlite3"))
    journal = NativeRevisionJournal(connection)
    first = _native("first-source-item")
    second = _native("second-source-item")
    first = replace(first, body=first.body.encode().decode())
    second = replace(second, body=second.body.encode().decode())
    assert first.body == second.body and first.body is not second.body
    assert first.revision_id != second.revision_id
    for unit in (first, second):
        journal.land((unit,))
    retained = [journal.units[unit.revision_id][0] for unit in (first, second)]
    assert retained[0].body == retained[1].body
    assert canonical_json_bytes([asdict(unit) for unit in retained]) == canonical_json_bytes([asdict(first), asdict(second)])
    assert retained[0].authority.item_id != retained[1].authority.item_id
    assert retained[0].authority.admission_id != retained[1].authority.admission_id
    assert retained[0].authority.access_decision_id != retained[1].authority.access_decision_id
    assert retained[0].authority.records is not retained[1].authority.records
    assert all(left is not right for left, right in zip(retained[0].authority.records, retained[1].authority.records, strict=True))
    rows = connection.execute("SELECT * FROM ledger ORDER BY seq").fetchall()
    retained[0].authority.records[-1]["rights_gate_reason"] = "changed caller metadata"
    assert retained[1].authority.records[-1].get("rights_gate_reason") != "changed caller metadata"
    reopened = NativeRevisionJournal(connection)
    assert reopened.units[first.revision_id][0].body == reopened.units[second.revision_id][0].body
    assert reopened._bodies == {}
    assert reopened.units[first.revision_id] == (first,)
    assert reopened.units[second.revision_id] == (second,)
    assert connection.execute("SELECT * FROM ledger ORDER BY seq").fetchall() == rows
    connection.close()


def test_current_journal_retains_headers_not_complete_source_bodies(tmp_path):
    import gc
    import tracemalloc
    from newsroom.tests.test_native_graphiti import _native

    connection = connect(str(tmp_path / 'selected-source-bodies.sqlite3'))
    expected = tuple(replace(_native(f'large-{index}'),
        body=('香港🙂字' * 20_000) + str(index)) for index in range(16))
    original = NativeRevisionJournal(connection)
    for unit in expected:
        original.land((unit,))
    del original
    gc.collect()
    tracemalloc.start()
    try:
        reopened = NativeRevisionJournal(connection)
        gc.collect()
        retained, _ = tracemalloc.get_traced_memory()
        # Fixture bodies pre-date tracing; only replay's persistent allocation
        # is measured, not whole engine RSS or Graphiti's full pending cohort.
        assert retained < 512_000
        assert len(reopened.units) == len(expected)
        for unit in expected:
            assert reopened.units[unit.revision_id] == (unit,)
        gc.collect()
        assert tracemalloc.get_traced_memory()[0] < 512_000
    finally:
        tracemalloc.stop()
        connection.close()


def test_selected_source_rechecks_same_count_ingest_index_drift(tmp_path):
    from newsroom.tests.test_native_graphiti import _native
    connection = connect(str(tmp_path / 'selected-index-drift.sqlite3'))
    unit = _native('selected-index')
    original = NativeRevisionJournal(connection)
    original.land((unit,))
    reopened = NativeRevisionJournal(connection)
    connection.execute('UPDATE native_current_units SET effective_revision_digest=? WHERE ingest_id=?',
                       ('sha256:' + 'f' * 64, unit.ingest_id))
    connection.commit()
    try:
        with pytest.raises(ValueError, match='selected source index'):
            reopened.units[unit.revision_id]
        assert not connection.in_transaction
    finally:
        connection.close()


@pytest.mark.parametrize('change', ['raw-body', 'resigned-body', 'land-digest', 'effective-revision'])
def test_selected_source_rechecks_original_byte_pin_after_reopen(tmp_path, change):
    from newsroom.tests.test_native_graphiti import _native
    connection = connect(str(tmp_path / 'selected-byte-pin.sqlite3'))
    unit = _native('selected-body')
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    journal = NativeRevisionJournal(connection)
    raw = connection.execute('SELECT content_json FROM native_current_sources WHERE revision_id=?',
                             (unit.revision_id,)).fetchone()[0]
    value = json.loads(raw)
    if change == 'effective-revision':
        value['units'][0]['effective_revision']['first_observed_at'] = '2026-09-03T12:00:00Z'
    else:
        value['units'][0]['body'] = value['units'][0]['body'].replace('Current', 'Changed')
    changed = canonical_json_bytes(value).decode()
    if change == 'land-digest':
        connection.execute('UPDATE native_current_sources SET land_digest=? WHERE revision_id=?',
                           ('sha256:' + 'f' * 64, unit.revision_id))
    else:
        connection.execute('UPDATE native_current_sources SET content_json=? WHERE revision_id=?',
                           (changed, unit.revision_id))
        if change != 'raw-body':
            connection.execute('UPDATE native_current_sources SET content_digest=? WHERE revision_id=?',
                               (digest_bytes(changed.encode()), unit.revision_id))
    connection.commit()
    try:
        with pytest.raises(ValueError, match='source'):
            journal.units[unit.revision_id]
        assert not connection.in_transaction
        assert journal._bodies == {}
    finally:
        connection.close()


def test_selected_source_preserves_caller_transaction_and_interrupts(tmp_path):
    import sqlite3
    from newsroom.tests.test_native_graphiti import _native
    connection = connect(str(tmp_path / 'selected-read-scope.sqlite3'))
    unit = _native('selected-scope')
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    connection.execute('BEGIN')
    try:
        assert journal.units[unit.revision_id] == (unit,)
        assert connection.in_transaction
        connection.set_progress_handler(lambda: 1, 1)
        with pytest.raises(sqlite3.OperationalError, match='interrupted'):
            journal.units[unit.revision_id]
        assert connection.in_transaction
    finally:
        connection.set_progress_handler(None, 0)
        connection.rollback()
        connection.close()
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        journal.units[unit.revision_id]


def test_selected_source_rejects_same_count_reassigned_ingest(tmp_path):
    from newsroom.tests.test_native_graphiti import _native
    connection = connect(str(tmp_path / 'selected-membership.sqlite3'))
    units = (_native('member-one'), _native('member-two'))
    journal = NativeRevisionJournal(connection)
    for unit in units:
        journal.land((unit,))
    journal = NativeRevisionJournal(connection)
    connection.execute('UPDATE native_current_units SET revision_id=? WHERE ingest_id=?',
                       (units[0].revision_id, units[1].ingest_id))
    connection.commit()
    try:
        with pytest.raises(ValueError, match='selected source index'):
            journal.units[units[0].revision_id]
        assert connection.execute('SELECT count(*) FROM native_current_units').fetchone()[0] == 2
        assert not connection.in_transaction
    finally:
        connection.close()


def test_normal_connection_indexes_exact_selected_revision_membership(tmp_path):
    connection = connect(str(tmp_path / 'source-membership-index.sqlite3'))
    try:
        plans = connection.execute('EXPLAIN QUERY PLAN SELECT count(*) FROM native_current_units WHERE revision_id=?',
                                   ('selected-revision',)).fetchall()
        assert any('COVERING INDEX native_current_units_revision' in row[3] for row in plans)
        assert not any(row[3].startswith('SCAN ') for row in plans)
    finally:
        connection.close()
