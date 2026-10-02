"""Explicit diagnostic expiry is targeted SQL, never a historical re-audit."""
import json
import sqlite3

import pytest

from newsroom.authority import audit_retention as retention
from newsroom.authority.persistence import DiagnosticHistoryExpired
from .test_projection_chain_retirement import _fixture
from .projection_b1_helpers import open_projection_system,proof


def test_explicit_retirement_never_scans_business_cas_or_external_bytes(tmp_path,monkeypatch):
    root,path,request,result=_fixture(tmp_path)
    external=root/'unpublished_store.sqlite3'
    with sqlite3.connect(external) as connection:
        connection.execute('INSERT INTO retained_receipts VALUES (?)',(json.dumps({'old_diagnostic':str(result.authority_event_id)}).encode(),))
    before=external.read_bytes()
    def forbidden(*_args,**_kwargs):pytest.fail('historical scan called')
    monkeypatch.setattr(retention,'_scan_business',forbidden)
    monkeypatch.setattr(retention,'_scan_cas',forbidden)
    report=retention.retire_native_projection_diagnostics(root,apply=True)
    assert report['committed'] and report['retired_projection_chains']==1
    assert external.read_bytes()==before
    with open_projection_system(path) as system:
        with pytest.raises(DiagnosticHistoryExpired):
            system.events.provenance(str(result.authority_event_id),proof=proof())


def test_dry_run_keeps_authority_and_external_bytes(tmp_path):
    root,path,request,result=_fixture(tmp_path)
    before=path.read_bytes()
    report=retention.retire_native_projection_diagnostics(root)
    assert not report['committed'] and not report['compacted']
    assert report['counts']['retired_projection_chains']==1
    assert path.read_bytes()==before


def test_empty_retirement_does_not_build_indexes_or_walk_protection(tmp_path, monkeypatch):
    from newsroom.authority import projection_retirement

    root, _, _, _ = _fixture(tmp_path, retire=False)
    def unexpected(*args, **kwargs):
        pytest.fail("No candidate requires protection or expiry work")
    monkeypatch.setattr(projection_retirement, "protect_candidates", unexpected)
    monkeypatch.setattr(projection_retirement, "expire_candidates", unexpected)
    report = retention.retire_native_projection_diagnostics(root, apply=True)
    assert report["retired_projection_chains"] == 0
    assert report["deleted"] == {}


def test_native_fk_failure_rolls_back_expiry_and_keeps_guards(tmp_path,monkeypatch):
    from newsroom.authority import projection_retirement as projection
    root,path,request,result=_fixture(tmp_path)
    with sqlite3.connect(path) as connection:
        before=connection.execute('SELECT event_id,command_id,retired_header_digest FROM ledger_events ORDER BY ledger_seq').fetchall()
        guards=connection.execute("SELECT name,sql FROM sqlite_schema WHERE type='trigger' ORDER BY name").fetchall()
    original=projection.expire_candidates
    def fail(connection):
        assert connection.execute('PRAGMA foreign_keys').fetchone()==(1,)
        deleted=original(connection)
        connection.execute('CREATE TEMP TABLE fk_parent(id INTEGER PRIMARY KEY)')
        connection.execute('CREATE TEMP TABLE fk_child(id INTEGER REFERENCES fk_parent(id))')
        connection.execute('INSERT INTO fk_child VALUES(1)')
        return deleted
    monkeypatch.setattr(projection,'expire_candidates',fail)
    with pytest.raises(sqlite3.IntegrityError):retention.retire_native_projection_diagnostics(root,apply=True)
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT event_id,command_id,retired_header_digest FROM ledger_events ORDER BY ledger_seq').fetchall()==before
        assert connection.execute("SELECT name,sql FROM sqlite_schema WHERE type='trigger' ORDER BY name").fetchall()==guards


def test_committed_delete_is_reported_when_vacuum_fails(tmp_path,monkeypatch):
    root,path,request,result=_fixture(tmp_path)
    connect=retention.sqlite3.connect
    class VacuumFailure(sqlite3.Connection):
        def execute(self,sql,*args):
            if sql=='VACUUM':raise sqlite3.OperationalError('fixture compaction failure')
            return super().execute(sql,*args)
    monkeypatch.setattr(retention.sqlite3,'connect',lambda *a,**kw:connect(*a,**kw,factory=VacuumFailure))
    report=retention.retire_native_projection_diagnostics(root,apply=True)
    assert report['committed'] and not report['compacted']
    assert report['retired_projection_chains']==1
    assert 'Retirement committed; compaction failed' in report['compaction_error']
