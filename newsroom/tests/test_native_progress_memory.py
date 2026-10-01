"""Replay shares immutable unit metadata, never authority containers or history."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane.corpus import units_from
from newsroom.control_plane.editorial import GroupedObservation
from newsroom.control_plane.items import SourceItem
from newsroom.control_plane.native_progress import LAND, NativeRevisionJournal, _landed_units
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
        append_ledger(connection, LAND, value)
        connection.commit()
        rows = connection.execute("SELECT * FROM ledger ORDER BY seq").fetchall()
        statements = []
        connection.set_trace_callback(statements.append)
        journal = NativeRevisionJournal(connection)
        connection.set_trace_callback(None)
        assert len(statements) == 1
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
        assert reopened.progress == journal.progress
        assert reopened.observations == {reference[1]: reference}
        assert reopened.portfolio_reference(reopened.portfolio) == portfolio_reference
        assert reopened.units[revision][0].coverage_first_observed_at == units[0].coverage_first_observed_at
        changed = (replace(reopened.units[revision][0], body=units[0].body + " changed"), *reopened.units[revision][1:])
        with pytest.raises(ValueError, match="chunk coverage"):
            reopened.land(changed)
        assert connection.execute("SELECT * FROM ledger ORDER BY seq").fetchall() == rows
        facts = reopened.progress[revision]["facts"]
        facts["retrieval_binding"]["request"]["nodes"][0] = "Changed same-count evidence"
        result = reopened.advance(revision, stage="EVIDENCE_HOLD", facts=facts)
        assert result["ordinal"] == 3
        encoded = json.loads(connection.execute("SELECT payload_json FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()[0])
        assert "retrieval_facts_ref" not in encoded
        assert NativeRevisionJournal(connection).progress == reopened.progress
    finally:
        connection.close()
