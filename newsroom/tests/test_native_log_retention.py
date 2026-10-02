from dataclasses import asdict
import json
import sqlite3

import pytest

from newsroom.control_plane.native_progress import NativeRevisionJournal, import_legacy_native_progress
from newsroom.control_plane.native_log_retention import prune_native_diagnostic_payloads
from newsroom.control_plane.store import append_ledger, connect
from newsroom.tests.test_native_graphiti import _native


def _fixture(tmp_path):
    connection = connect(str(tmp_path / 'private.sqlite3'))
    connection.execute('DELETE FROM native_current_meta')  # Legacy, before CURRENT cutover.
    unit = _native()
    append_ledger(connection, 'NATIVE_REVISION_LANDED', {'revision_id': unit.revision_id, 'units': [asdict(unit)]})
    pair = {'retrieval_binding': {'fixture': 'x' * 10000}, 'retrieval_rights_inventory': {'digest': 'rights'}}
    for ordinal, stage in enumerate(('GRAPHITI_COMPLETE', 'EVIDENCE_HOLD', 'EMBEDDING_STARTED', 'ACKNOWLEDGED'), 1):
        facts = {**pair, 'state': stage}
        if stage == 'EMBEDDING_STARTED':
            facts['retrieval_embeddings'] = {unit.ingest_id: {'state': 'STARTED', 'passage_id': unit.ingest_id, 'cycle_id': 'native-passage:'+unit.ingest_id, 'attempt_number': 1}}
        append_ledger(connection, 'NATIVE_REVISION_PROGRESS', {'revision_id': unit.revision_id, 'ordinal': ordinal, 'stage': stage, 'facts': facts})
    append_ledger(connection, 'NATIVE_SOURCE_PORTFOLIO', {'sources': []})
    append_ledger(connection, 'NATIVE_SOURCE_PORTFOLIO', {'sources': []})
    append_ledger(connection, 'NATIVE_SERVICE_CYCLE_TERMINAL', {'outcome': 'COMPLETE'})
    append_ledger(connection, 'MODEL_FIXTURE_BUSINESS', {'usage': 'UNKNOWN', 'retry_authorised': False})
    connection.commit()
    import_legacy_native_progress(connection)
    return connection, unit


def test_expire_stale_diagnostics_preserves_headers_current_pairs_and_business_proofs(tmp_path):
    connection, unit = _fixture(tmp_path)
    expected = NativeRevisionJournal(connection).current(unit.revision_id)
    headers = connection.execute('SELECT seq,at,kind,payload_digest,prev_digest,digest FROM ledger ORDER BY seq').fetchall()
    protected = connection.execute("SELECT seq,payload_json FROM ledger WHERE kind='MODEL_FIXTURE_BUSINESS' OR json_extract(payload_json,'$.stage') IN ('EMBEDDING_STARTED','ACKNOWLEDGED')").fetchall()
    report = prune_native_diagnostic_payloads(connection)
    assert report['expired_payloads'] == 4
    assert report['expired_bytes'] > 20000
    assert connection.execute('SELECT seq,at,kind,payload_digest,prev_digest,digest FROM ledger ORDER BY seq').fetchall() == headers
    assert all(connection.execute('SELECT payload_json FROM ledger WHERE seq=?', (seq,)).fetchone()[0] == raw for seq, raw in protected)
    assert NativeRevisionJournal(connection).current(unit.revision_id) == expected
    assert connection.execute('PRAGMA foreign_key_check').fetchall() == []
    assert prune_native_diagnostic_payloads(connection)['expired_payloads'] == 0
    connection.close()


def test_retention_requires_current_state_and_does_not_guess_from_logs(tmp_path):
    connection = connect(str(tmp_path / 'legacy.sqlite3'))
    connection.execute('DELETE FROM native_current_meta')
    append_ledger(connection, 'NATIVE_SERVICE_CYCLE_TERMINAL', {'outcome': 'COMPLETE'})
    connection.commit()
    with pytest.raises(ValueError, match='CURRENT'):
        prune_native_diagnostic_payloads(connection)
    assert connection.execute('SELECT payload_json FROM ledger').fetchone()[0] is not None
    connection.close()


def test_retention_failure_rolls_back_payload_expiry(tmp_path):
    connection, _ = _fixture(tmp_path)
    before = connection.execute('SELECT * FROM ledger ORDER BY seq').fetchall()
    connection.execute("CREATE TRIGGER deny_expiry BEFORE UPDATE OF payload_json ON ledger WHEN NEW.payload_json IS NULL AND OLD.kind='NATIVE_SOURCE_PORTFOLIO' BEGIN SELECT RAISE(ABORT,'fixture expiry failure'); END")
    connection.commit()
    with pytest.raises(sqlite3.IntegrityError, match='expiry failure'):
        prune_native_diagnostic_payloads(connection)
    assert connection.execute('SELECT * FROM ledger ORDER BY seq').fetchall() == before
    connection.close()
