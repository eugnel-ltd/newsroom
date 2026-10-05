"""A genuinely current source can retain bytes without replaying expired audit RPC."""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
import sqlite3

import pytest

from newsroom.authority import DiagnosticHistoryExpired, ObjectAdmissionRequest
from newsroom.authority.canonical import digest_bytes, digest_canonical
from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store
from newsroom.authority.native_current_rebuild import copy_selected_native_store
from newsroom.control_plane.graphiti_operational_readiness import OPERATIONAL_ADMISSION_TYPE
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.native_source_intake import NativeSourceIntake
from newsroom.increment9.proving import SOURCE_URLS
from newsroom.sources import SourceRevisionId
from newsroom.tests.test_native_current_checkpoint import source_fixture
from newsroom.tests.test_native_source_intake import ATOM, _document, _licence, _seed_uk01


@contextmanager
def _expired_passage(tmp_path, monkeypatch, *, enabled=True):
    with source_fixture(tmp_path, monkeypatch) as (args, runtime, _intake, first, before, root):
        unit, = first.units
        old_key = root._connection.execute('SELECT c.idempotency_key FROM authority_commands c JOIN object_admission_versions v ON v.event_id=(SELECT event_id FROM ledger_events WHERE command_id=c.command_id) WHERE v.admission_id=? AND v.lifecycle_version=1',
            (unit.authority.admission_id,)).fetchone()[0].removeprefix('lifecycle-activate:')
        roots = {'source_items': ((unit.authority.item_id,),),
            'source_revisions': ((unit.authority.revision_id,),),
            'discovery_representations': ((unit.authority.representation_id,),),
            'object_admissions': tuple((item[2],) for item in first.observations),
            'object_access_decisions': tuple((item[3],) for item in first.observations)}
        destination = tmp_path/'selected.sqlite3'
        with sqlite3.connect(destination, isolation_level=None) as selected:
            initialise_empty_checkpoint_store(selected)
            copy_selected_native_store(root, selected, roots=roots, dev_rebuild=True)
        destination.chmod(0o600)
        args = {**args, 'authority_path': destination}
    with open_native_runtime(**args) as runtime:
        definition = _seed_uk01(runtime)
        stat = destination.stat()
        epoch = digest_canonical({'path':str(destination.resolve()), 'device':stat.st_dev, 'inode':stat.st_ino})
        units, observations = {}, {item[1]:item for item in first.observations}
        intake = NativeSourceIntake(sources=runtime.authority.sources, objects=runtime.authority.objects,
            proof=runtime.proof, definition_ids={'UK-01':definition}, licence=_licence(),
            dispatch_fence=lambda *_:nullcontext(), retained_units=units, observations=observations,
            fetch=lambda url:(200, ATOM if url==SOURCE_URLS['UK-01'] else _document()),
            clock=lambda:datetime(2026,9,8,12,tzinfo=UTC), reobservation_epoch=epoch if enabled else None)
        # This is a real expired command reservation, not an exception-only mock.
        data = ' '.join(unit.episode_body.split()).encode()
        with sqlite3.connect(destination) as db:
            assert not db.execute('SELECT 1 FROM object_admissions WHERE admission_id=?', (unit.authority.admission_id,)).fetchone()
            reserved = db.execute('SELECT * FROM native_expired_command_keys WHERE key=?', ('lifecycle-activate:'+old_key,)).fetchone()
            assert reserved is not None
        with pytest.raises(DiagnosticHistoryExpired):
            runtime.authority.objects.admit(ObjectAdmissionRequest(OPERATIONAL_ADMISSION_TYPE, old_key), data, proof=runtime.proof)
        yield runtime, intake, unit, before, old_key, epoch, args, reserved


def test_expired_unreferenced_passage_has_one_current_intent_and_reopens(tmp_path, monkeypatch):
    from newsroom.control_plane.native_progress import NativeRevisionJournal
    from newsroom.control_plane.store import connect as connect_unpublished_store
    with _expired_passage(tmp_path, monkeypatch) as (runtime, intake, old, before, old_key, epoch, args, reserved):
        current = intake.poll()[0]
        assert current.status == 'READY', current
        unit, = current.units
        assert (unit.ingest_id,unit.revision_id,unit.authority.item_id,unit.authority.revision_id,unit.authority.representation_id) == (
            old.ingest_id,old.revision_id,old.authority.item_id,old.authority.revision_id,old.authority.representation_id)
        assert unit.authority.admission_id != old.authority.admission_id
        assert runtime.authority.sources.revision(SourceRevisionId.parse(unit.revision_id),proof=runtime.proof) == before
        suffix = old_key.removeprefix('native-source-passage:')
        request = ObjectAdmissionRequest(OPERATIONAL_ADMISSION_TYPE, f'native-source-passage-reobserve:{epoch}:{suffix}')
        assert str(runtime.authority.objects.committed_admission(request,proof=runtime.proof).admission.admission_id) == unit.authority.admission_id
        private = connect_unpublished_store(str(tmp_path/'private.sqlite3'))
        journal = NativeRevisionJournal(private);journal.land(current.units)
        intake._retained_units = journal.units
        assert intake.poll()[0].units == current.units
        private.close()
        with sqlite3.connect(args['authority_path']) as db:
            assert db.execute('SELECT * FROM native_expired_command_keys WHERE key=?', ('lifecycle-activate:'+old_key,)).fetchone() == reserved
            rows = db.execute('SELECT operation_id FROM object_lifecycle_operations WHERE idempotency_key=?', (request.idempotency_key,)).fetchall()
            assert len(rows) == 1
    with open_native_runtime(**args) as runtime:
        restored = runtime.authority.objects.committed_admission(request,proof=runtime.proof)
        assert str(restored.admission.admission_id) == unit.authority.admission_id
        private = connect_unpublished_store(str(tmp_path/'private.sqlite3'))
        assert NativeRevisionJournal(private).units[unit.revision_id] == current.units
        private.close()


def test_missing_epoch_keeps_expired_key_fail_closed(tmp_path, monkeypatch):
    with _expired_passage(tmp_path, monkeypatch, enabled=False) as (runtime, intake, old, _before, old_key, _epoch, args, reserved):
        assert intake.poll()[0].status == 'HOLD'
        with sqlite3.connect(args['authority_path']) as db:
            assert db.execute('SELECT * FROM native_expired_command_keys WHERE key=?', ('lifecycle-activate:'+old_key,)).fetchone() == reserved
            assert not db.execute("SELECT 1 FROM object_lifecycle_operations WHERE idempotency_key LIKE 'native-source-passage-reobserve:%'").fetchone()


def test_reobserved_admission_must_match_exact_fresh_passage_bytes(tmp_path, monkeypatch):
    with _expired_passage(tmp_path, monkeypatch) as (runtime, intake, old, _before, old_key, epoch, args, reserved):
        request = ObjectAdmissionRequest(OPERATIONAL_ADMISSION_TYPE,
            f'native-source-passage-reobserve:{epoch}:'+old_key.removeprefix('native-source-passage:'))
        forged = b'X'*len(' '.join(old.episode_body.split()).encode())
        poisoned = runtime.authority.objects.admit(request, forged, proof=runtime.proof).admission
        result = intake.poll()[0]
        assert result.status == 'HOLD' and result.units == ()
        assert str(runtime.authority.objects.committed_admission(request,proof=runtime.proof).admission.admission_id) == str(poisoned.admission_id)
        with sqlite3.connect(args['authority_path']) as db:
            assert db.execute('SELECT * FROM native_expired_command_keys WHERE key=?', ('lifecycle-activate:'+old_key,)).fetchone() == reserved
            assert db.execute('SELECT count(*) FROM object_lifecycle_operations WHERE idempotency_key=?', (request.idempotency_key,)).fetchone()[0] == 1


def test_revoked_current_rights_do_not_create_reobservation_intent(tmp_path, monkeypatch):
    from types import SimpleNamespace
    with _expired_passage(tmp_path, monkeypatch) as (_runtime, intake, _old, _before, _old_key, _epoch, args, _reserved):
        intake._licence = SimpleNamespace(for_source=lambda **_values:SimpleNamespace(decision='DENIED'))
        result = intake.poll()[0]
        assert result.status == 'HOLD' and result.reason_code == 'CURRENT_RIGHTS_HOLD'
        with sqlite3.connect(args['authority_path']) as db:
            assert not db.execute("SELECT 1 FROM object_lifecycle_operations WHERE idempotency_key LIKE 'native-source-passage-reobserve:%'").fetchone()


def test_owner_stop_recheck_denies_before_current_reobservation_admission(tmp_path, monkeypatch):
    from newsroom.control_plane.veto import VetoError
    with _expired_passage(tmp_path, monkeypatch) as (_runtime, intake, _old, _before, _old_key, _epoch, args, _reserved):
        roots = 0
        @contextmanager
        def stopped(source_id, url):
            nonlocal roots
            if source_id == 'UK-01' and url == SOURCE_URLS['UK-01']:
                roots += 1
                if roots == 2:
                    raise VetoError('fixture stopped before reobservation')
            yield
        intake._fence = stopped
        with pytest.raises(VetoError, match='stopped before reobservation'):
            intake.poll()
        with sqlite3.connect(args['authority_path']) as db:
            assert not db.execute("SELECT 1 FROM object_lifecycle_operations WHERE idempotency_key LIKE 'native-source-passage-reobserve:%'").fetchone()


def test_invalid_reobservation_epoch_denies_before_source_effects():
    with pytest.raises(ValueError):
        NativeSourceIntake(sources=None,objects=None,proof=None,definition_ids={},licence=None,
            dispatch_fence=lambda *_:nullcontext(),reobservation_epoch='not-a-digest')
