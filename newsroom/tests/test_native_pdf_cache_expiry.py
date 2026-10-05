"""Expired optional PDF metadata must not replace a fresh parser result."""
from contextlib import contextmanager, nullcontext
import sqlite3

import pytest

from newsroom.authority import ObjectAdmissionRequest, DiagnosticHistoryExpired
from newsroom.authority._graphiti_increment4_system import _AUTHORITY_COMPOSITION_TOKEN
from newsroom.authority.canonical import digest_canonical
from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store
from newsroom.authority.native_current_rebuild import copy_selected_native_store
from newsroom.control_plane.govuk_pdf import GovUkPdfHold, pdf_parse_binding
from newsroom.control_plane.graphiti_operational_readiness import OPERATOR_AUTHORITY_DOMAIN, OPERATOR_PRINCIPAL_ID
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.native_source_intake import NativeSourceIntake
from newsroom.control_plane.native_policies import NATIVE_SOURCE_OBSERVATION_ADMISSION_TYPE
from newsroom.tests.test_govuk_pdf import pdf_bytes, parent_bytes, ASSET, PARENT, NOW
from newsroom.tests.test_native_runtime import _args


@contextmanager
def _expired_cache(tmp_path, monkeypatch, *, negative=False):
    raw = pdf_bytes('Complete first page.', 'Complete second page.',
        **({'catalog':b'/AcroForm << /Fields [<< /FT /Tx >>] >>'} if negative else {}))
    parent = parent_bytes(raw)
    kwargs = dict(parent_url=PARENT, parent_raw=parent, asset_url=ASSET, raw=raw,
        retrieved_at=NOW, source_version='fixture-current-source-v1')
    args = _args(tmp_path,monkeypatch)
    args.update(principal_id=OPERATOR_PRINCIPAL_ID,authority_domain=OPERATOR_AUTHORITY_DOMAIN)
    with sqlite3.connect(args['authority_path'],isolation_level=None) as db:
        initialise_empty_checkpoint_store(db)
    args['authority_path'].chmod(0o600)
    def poller(runtime):
        return NativeSourceIntake(sources=runtime.authority.sources,objects=runtime.authority.objects,
            proof=runtime.proof,definition_ids={},licence=None,dispatch_fence=lambda *_:nullcontext())
    with open_native_runtime(**args) as runtime:
        original_hold = None
        original = None
        if negative:
            with pytest.raises(GovUkPdfHold) as held:
                poller(runtime)._parse_pdf(**kwargs)
            original_hold = held.value.reason_code
        else:
            original = poller(runtime)._parse_pdf(**kwargs)
        binding = pdf_parse_binding(**kwargs)
        request = ObjectAdmissionRequest(NATIVE_SOURCE_OBSERVATION_ADMISSION_TYPE,
            'native-pdf-parse:'+digest_canonical(binding))
        retained = runtime.authority.objects.committed_admission(request,proof=runtime.proof)
        assert retained is not None
        root = runtime.authority._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0]
        selected = tmp_path/'selected.sqlite3'
        with sqlite3.connect(selected,isolation_level=None) as db:
            initialise_empty_checkpoint_store(db)
            copy_selected_native_store(root,db,roots={},dev_rebuild=True)
        selected.chmod(0o600)
        args = {**args,'authority_path':selected}
    with open_native_runtime(**args) as runtime:
        # Exactly the live gap: original reservation, no activation/admission.
        with pytest.raises(DiagnosticHistoryExpired):
            runtime.authority.objects.committed_admission(request,proof=runtime.proof)
        with sqlite3.connect(selected) as db:
            assert not db.execute('SELECT 1 FROM object_admissions WHERE admission_id=?',(str(retained.admission.admission_id),)).fetchone()
            tombstone = db.execute('SELECT * FROM native_expired_command_keys WHERE key=?',('lifecycle-activate:'+request.idempotency_key,)).fetchone()
            assert tombstone is not None
        yield runtime,poller(runtime),kwargs,request,args,tombstone,original_hold,original


@pytest.mark.parametrize('negative',[False,True])
def test_expired_pdf_cache_returns_exact_fresh_document_or_original_hold(tmp_path,monkeypatch,negative):
    from newsroom.control_plane import native_source_intake as module
    with _expired_cache(tmp_path,monkeypatch,negative=negative) as (runtime,intake,kwargs,request,args,tombstone,reason,original):
        calls=[];parse=module.parse_govuk_pdf
        def counted(*values,**named):
            calls.append(1)
            return parse(*values,**named)
        monkeypatch.setattr(module,'parse_govuk_pdf',counted)
        if negative:
            with pytest.raises(GovUkPdfHold) as held:
                intake._parse_pdf(**kwargs)
            assert held.value.reason_code == reason
        else:
            document = intake._parse_pdf(**kwargs)
            assert document == original
            assert 'Complete first page.' in document.body_text and 'Complete second page.' in document.body_text
        assert calls == [1]
        with pytest.raises(DiagnosticHistoryExpired):
            runtime.authority.objects.committed_admission(request,proof=runtime.proof)
        with sqlite3.connect(args['authority_path']) as db:
            assert db.execute('SELECT * FROM native_expired_command_keys WHERE key=?',('lifecycle-activate:'+request.idempotency_key,)).fetchone()==tombstone
            assert not db.execute('SELECT 1 FROM object_lifecycle_operations WHERE idempotency_key=?',(request.idempotency_key,)).fetchone()


@pytest.mark.parametrize('negative',[False,True])
def test_expired_optional_cache_reparse_has_no_staging_or_security_regrowth(tmp_path,monkeypatch,negative):
    with _expired_cache(tmp_path,monkeypatch,negative=negative) as (_runtime,intake,kwargs,_request,args,_tombstone,reason,original):
        def counts():
            with sqlite3.connect(args['authority_path']) as db:
                tables=('object_staging_records','object_admission_preflights','authentication_contexts',
                        'authorization_requests','authorization_decisions')
                return {name:db.execute('SELECT count(*) FROM "'+name+'"').fetchone()[0]
                        for name in tables}
        before=counts()
        for _ in range(2):
            if negative:
                with pytest.raises(GovUkPdfHold) as held:intake._parse_pdf(**kwargs)
                assert held.value.reason_code==reason
            else:
                assert intake._parse_pdf(**kwargs)==original
        assert counts()==before


def test_expired_admission_denies_before_consuming_source_bytes(tmp_path,monkeypatch):
    with _expired_cache(tmp_path,monkeypatch) as (runtime,_intake,_kwargs,request,_args,_tombstone,_reason,_original):
        def source():
            pytest.fail('expired admission consumed source bytes')
            yield b'not read'
        with pytest.raises(DiagnosticHistoryExpired):
            runtime.authority.objects.admit(request,source(),proof=runtime.proof)


def test_expired_cache_does_not_waive_current_raw_pdf_identity(tmp_path,monkeypatch):
    with _expired_cache(tmp_path,monkeypatch) as (_runtime,intake,kwargs,_request,_args,_tombstone,_reason,_original):
        with pytest.raises(GovUkPdfHold,match='RAW_IDENTITY'):
            intake._parse_pdf(**{**kwargs,'raw':kwargs['raw'][:-5]})


def test_owner_stop_at_optional_cache_write_still_propagates(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from newsroom.control_plane.veto import VetoError
    with _expired_cache(tmp_path,monkeypatch) as (runtime,intake,kwargs,_request,_args,_tombstone,_reason,_original):
        def stopped(*_values,**_named):
            raise VetoError('fixture signed owner stop')
        intake._objects=SimpleNamespace(committed_admission=lambda *_args,**_kwargs:None,
            rehydrate=runtime.authority.objects.rehydrate,admit=stopped)
        with pytest.raises(VetoError,match='signed owner stop'):
            intake._parse_pdf(**kwargs)
