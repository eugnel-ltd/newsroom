"""Expired negative diagnostics must not suppress a fresh native HOLD."""
from datetime import UTC,datetime
import sqlite3

import pytest

from newsroom.authority import ObjectAdmissionRequest,DiagnosticHistoryExpired,HydrationRequest
from newsroom.authority.canonical import canonical_json_bytes,digest_bytes,digest_canonical
from newsroom.authority._graphiti_increment4_system import _AUTHORITY_COMPOSITION_TOKEN
from newsroom.authority.native_current_rebuild import copy_selected_native_store
from newsroom.authority.native_current_checkpoint_migrations import initialise_empty_checkpoint_store
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.native_evidence import PublicationRightsAssessment
from newsroom.control_plane import native_source_rights as rights
from newsroom.increment9.proving import SOURCE_URLS
from newsroom.tests.test_native_runtime import _args


def test_expired_original_hold_reobserves_once_per_owned_store_and_reopens(tmp_path,monkeypatch):
    args=_args(tmp_path,monkeypatch);target=tmp_path/'selected.sqlite3'
    assessment=PublicationRightsAssessment.create(decision='HOLD',permitted_use='PUBLICATION_EVIDENCE',
        policy_digest=rights.POLICY_DIGEST,evidence_digest=digest_bytes(b'transient current terms unavailable'))
    _,value=rights._rights_snapshot_values(source_id='HK-01',definition_url=SOURCE_URLS['HK-01'],
        assessment=assessment,observed_at='2026-09-08T12:00:00.000000Z',reason='SOURCE_TERMS_UNAVAILABLE',observations=())
    key='native-rights-assessment:'+assessment.record_id
    with open_native_runtime(**args) as runtime:
        retained=rights._retain_assessment(objects=runtime.authority.objects,proof=runtime.proof,value=value)
        original=runtime.authority._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0]
        event=original._connection.execute('SELECT event_id FROM object_admission_versions WHERE admission_id=? AND lifecycle_version=1',(str(retained.admission_id),)).fetchone()[0]
        with sqlite3.connect(target,isolation_level=None) as c:
            initialise_empty_checkpoint_store(c);copy_selected_native_store(original,c,roots={},dev_rebuild=True)
    target.chmod(0o600);args=dict(args,authority_path=target)
    metadata=target.stat();epoch=digest_canonical({'path':str(target.resolve()),'device':metadata.st_dev,'inode':metadata.st_ino})
    with open_native_runtime(**args) as runtime:
        objects,proof=runtime.authority.objects,runtime.proof
        root=runtime.authority._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0]
        with pytest.raises(DiagnosticHistoryExpired):objects.admit(ObjectAdmissionRequest('evidence.source',key),canonical_json_bytes(value),proof=proof)
        reservation=tuple(root._connection.execute('SELECT * FROM native_expired_command_keys WHERE event_id=?',(event,)).fetchone())
        first=rights._retain_assessment(objects=objects,proof=proof,value=value,reobservation_epoch=epoch)
        before={t:root._connection.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in ('object_admissions','object_access_decisions','object_admission_preflights','object_staging_records','authentication_contexts','authorization_requests','authorization_decisions')}
        assert rights._retain_assessment(objects=objects,proof=proof,value=value,reobservation_epoch=epoch)==first
        assert before=={t:root._connection.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in before}
        assert tuple(root._connection.execute('SELECT * FROM native_expired_command_keys WHERE event_id=?',(event,)).fetchone())==reservation
        assert value['decision']=='HOLD'
        assert objects.rehydrate(HydrationRequest(first.admission_id,'evidence.source'),proof=proof).data==canonical_json_bytes(value)
    with open_native_runtime(**args) as runtime:
        root=runtime.authority._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0]
        before=root._connection.total_changes
        assert rights._retain_assessment(objects=runtime.authority.objects,proof=runtime.proof,value=value,reobservation_epoch=epoch)==first
        assert root._connection.total_changes==before


@pytest.mark.parametrize('decision',['HOLD','PERMITTED'])
def test_current_permitted_and_normal_hold_keep_original_semantic_identity(tmp_path,monkeypatch,decision):
    args=_args(tmp_path,monkeypatch)
    assessment=PublicationRightsAssessment.create(decision=decision,permitted_use='PUBLICATION_EVIDENCE',policy_digest=rights.POLICY_DIGEST,evidence_digest=digest_bytes(decision.encode()))
    _,value=rights._rights_snapshot_values(source_id='HK-02',definition_url=SOURCE_URLS['HK-02'],assessment=assessment,
        observed_at='2026-09-08T12:00:00.000000Z',reason='FIXTURE_CURRENT',observations=())
    with open_native_runtime(**args) as runtime:
        first=rights._retain_assessment(objects=runtime.authority.objects,proof=runtime.proof,value=value)
        root=runtime.authority._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0]
        before=root._connection.total_changes
        assert rights._retain_assessment(objects=runtime.authority.objects,proof=runtime.proof,value=value,reobservation_epoch=digest_bytes(b'epoch'))==first
        assert root._connection.total_changes==before
        assert root._connection.execute("SELECT 1 FROM object_admission_idempotency WHERE idempotency_key LIKE '%:native-reobservation:%'").fetchone()is None


@pytest.mark.parametrize('fault',['forged','cas','revoked','permitted','stop'])
def test_reobservation_cannot_waive_corruption_current_permission_or_stop(tmp_path,monkeypatch,fault):
    from newsroom.authority import ObjectIntegrityError,ObjectAdmissionDenied,ObjectHydrationDenied
    from newsroom.control_plane.veto import VetoError
    args=_args(tmp_path,monkeypatch);target=tmp_path/'negative.sqlite3'
    assessment=PublicationRightsAssessment.create(decision='PERMITTED' if fault=='permitted' else 'HOLD',permitted_use='PUBLICATION_EVIDENCE',policy_digest=rights.POLICY_DIGEST,evidence_digest=digest_bytes(b'negative'))
    _,value=rights._rights_snapshot_values(source_id='HK-01',definition_url=SOURCE_URLS['HK-01'],assessment=assessment,
        observed_at='2026-09-08T12:00:00.000000Z',reason='FIXTURE',observations=())
    with open_native_runtime(**args) as runtime:
        rights._retain_assessment(objects=runtime.authority.objects,proof=runtime.proof,value=value)
        root=runtime.authority._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0]
        with sqlite3.connect(target,isolation_level=None) as c:
            initialise_empty_checkpoint_store(c);copy_selected_native_store(root,c,roots={},dev_rebuild=True)
    target.chmod(0o600);args=dict(args,authority_path=target);epoch=digest_bytes(b'fixture current epoch')
    with open_native_runtime(**args) as runtime:
        objects,proof=runtime.authority.objects,runtime.proof
        fallback=ObjectAdmissionRequest('evidence.source',f"native-rights-assessment:{assessment.record_id}:native-reobservation:{epoch}")
        restore=None
        if fault=='forged':objects.admit(fallback,b'{}',proof=proof)
        elif fault in ('cas','revoked'):
            admitted=rights._retain_assessment(objects=objects,proof=proof,value=value,reobservation_epoch=epoch)
            if fault=='revoked':objects.revoke(admitted.admission_id,reason_code='REVOKED',idempotency_key='revoke-fallback',proof=proof)
            else:
                blob=args['object_root']/'objects'/admitted.blob.blob_digest[7:9]/admitted.blob.blob_digest[7:]
                body=blob.read_bytes();mode=blob.stat().st_mode&0o777
                blob.chmod(0o600);blob.write_bytes(b'corrupt');blob.chmod(mode)
                def restore():blob.chmod(0o600);blob.write_bytes(body);blob.chmod(mode)
        elif fault=='stop':
            def veto(*_a,**_k):raise VetoError('owner stop')
            monkeypatch.setattr(type(objects),'committed_admission',veto)
        expected={'forged':ValueError,'cas':ObjectIntegrityError,'revoked':(ObjectAdmissionDenied,ObjectHydrationDenied),'permitted':DiagnosticHistoryExpired,'stop':VetoError}[fault]
        try:
            with pytest.raises(expected):rights._retain_assessment(objects=objects,proof=proof,value=value,reobservation_epoch=epoch)
        finally:
            if restore:restore()


def test_full_native_refresh_reobserves_transient_hold_without_repeat_growth(tmp_path,monkeypatch):
    import io
    from contextlib import nullcontext
    from types import SimpleNamespace
    from newsroom.tests import test_native_composition as fixture
    from newsroom.control_plane import native_composition,govuk_rights
    from newsroom.tests.projection_b2_helpers import MemoryNeo4jAdapter
    pages={govuk_rights.REUSE_URL:b'<main>Fixture reuse</main>',govuk_rights.LICENCE_URL:b'<main>Fixture licence</main>'}
    monkeypatch.setattr(govuk_rights,'REVIEWED_TEXT',{url:govuk_rights.licence_text_digest(raw) for url,raw in pages.items()})
    class Response(io.BytesIO):
        status=200
        def __init__(self,url):super().__init__(pages[url]);self.url=url
        def geturl(self):return self.url
    class Opener:
        def open(self,request,timeout):return Response(request.full_url)
    monkeypatch.setattr(govuk_rights.urllib.request,'build_opener',lambda *_:Opener())
    available=[False];observed_at=[fixture.NOW]
    def observe(*,objects,proof,**_):
        result={source:rights.SourceTermsEvidence(source,observed_at[0].isoformat(),'SOURCE_TERMS_UNAVAILABLE',()) for source in rights.TERMS}
        if available[0]:
            body=b'<main>Fixture restricted publisher terms</main>';digest=digest_bytes(body)
            admitted=objects.admit(ObjectAdmissionRequest('evidence.source','native-source-terms:HK-01:'+digest),body,proof=proof).admission
            access=objects.rehydrate(HydrationRequest(admitted.admission_id,'evidence.source'),proof=proof).decision
            result['HK-01']=rights.SourceTermsEvidence('HK-01',observed_at[0].isoformat(),rights.RESTRICTIONS['HK-01'],((rights.TERMS['HK-01'][0][0],digest,str(admitted.admission_id),str(access.access_decision_id)),))
        return result
    monkeypatch.setattr(native_composition,'observe_portfolio_terms',observe)
    monkeypatch.setattr('newsroom.authority._graphiti_increment4_system._open_structural_graph_adapter',lambda _:MemoryNeo4jAdapter())
    monkeypatch.setattr(native_composition,'open_native_retrieval_neo4j_resources',lambda **_:SimpleNamespace(projector=fixture._RetrievalProjection(),fulltext=fixture._Reader(),close=lambda:None))
    arguments={**fixture._arguments(tmp_path),'stop_check':lambda:None,'stop_fence':nullcontext}
    target=tmp_path/'pipeline-selected.sqlite3'
    with native_composition.open_native_pipeline(**arguments) as pipeline:
        portfolio=pipeline._intake._licence
        old=portfolio.snapshot_for('HK-01').assessment_admission_id
        available[0]=True;portfolio.refresh()
        root=pipeline._runtime.authority._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0]
        identities={str(a) for a in portfolio.govuk.admission_ids}
        for source in SOURCE_URLS:
            reference=portfolio.snapshot_for(source)
            identities.update((reference.assessment_admission_id,reference.observation_admission_id))
        current=rights.read_rights_observation(objects=pipeline._runtime.authority.objects,proof=pipeline._runtime.proof,
            reference=portfolio.snapshot_for('HK-01'),source_id='HK-01',definition_url=SOURCE_URLS['HK-01'])
        identities.update(v[2] for v in current['observations'])
        assert old not in identities
        with sqlite3.connect(target,isolation_level=None) as c:
            initialise_empty_checkpoint_store(c);copy_selected_native_store(root,c,roots={'object_admissions':tuple((identity,) for identity in sorted(identities))},dev_rebuild=True)
    from datetime import timedelta
    target.chmod(0o600);available[0]=False
    observed_at[0]=fixture.NOW+timedelta(seconds=1)
    arguments['clock']=lambda:observed_at[0]
    with native_composition.open_native_pipeline(**dict(arguments,authority_path=target)) as pipeline:
        portfolio=pipeline._intake._licence
        first=portfolio.snapshot_for('HK-01')
        assert first.assessment_admission_id!=old
        assert portfolio.for_source(source_id='HK-01',definition_url=SOURCE_URLS['HK-01']).decision=='HOLD'
        assert portfolio.for_source(source_id='UK-01',definition_url=SOURCE_URLS['UK-01']).decision=='PERMITTED'
        root=pipeline._runtime.authority._base._authority_composition(_AUTHORITY_COMPOSITION_TOKEN)[0]
        before=root._connection.total_changes
        portfolio.refresh();portfolio.refresh()
        assert portfolio.snapshot_for('HK-01')==first and root._connection.total_changes==before
    with native_composition.open_native_pipeline(**dict(arguments,authority_path=target)) as pipeline:
        assert pipeline._intake._licence.snapshot_for('HK-01')==first
