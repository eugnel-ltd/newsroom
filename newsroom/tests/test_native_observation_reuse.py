"""Fresh polls reuse only governed CURRENT raw observations, not admission work."""
from contextlib import nullcontext
from datetime import UTC,datetime,timedelta
import sqlite3

from newsroom.control_plane.native_source_intake import NativeSourceIntake
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.graphiti_operational_readiness import OPERATOR_PRINCIPAL_ID,OPERATOR_AUTHORITY_DOMAIN
from newsroom.increment9.proving import SOURCE_URLS
from newsroom.tests.test_native_runtime import _args
from newsroom.tests.test_native_source_intake import _seed_uk01,_licence,_document,ATOM


def counts(path):
    with sqlite3.connect(path) as c:
        return tuple(c.execute('SELECT count(*) FROM '+t).fetchone()[0]for t in
            ('object_admission_preflights','object_access_decisions','object_staging_records','object_admissions'))


def test_fresh_poll_repeat_and_reopen_uses_current_observation_without_preflight(tmp_path,monkeypatch):
    args=_args(tmp_path,monkeypatch);args.update(principal_id=OPERATOR_PRINCIPAL_ID,authority_domain=OPERATOR_AUTHORITY_DOMAIN)
    observations={};retained={};calls=[];observed=[];clock=[datetime(2026,9,8,12,tzinfo=UTC)]
    def fetch(url):calls.append(url);return 200,ATOM if url==SOURCE_URLS['UK-01']else _document()
    def intake(runtime,definition):
        result=NativeSourceIntake(sources=runtime.authority.sources,objects=runtime.authority.objects,
            proof=runtime.proof,definition_ids={'UK-01':definition},licence=_licence(),
            dispatch_fence=lambda *_:nullcontext(),retained_units=retained,observations=observations,
            fetch=fetch,clock=lambda:clock[0])
        original=result._settle_item
        def settle(*args,**kwargs):
            observed.append(args[7])
            return original(*args,**kwargs)
        result._settle_item=settle
        return result
    with open_native_runtime(**args) as runtime:
        definition=_seed_uk01(runtime);poll=intake(runtime,definition);first=poll.poll()[0]
        assert first.status=='READY'
        observations.update({r[1]:r for r in first.observations});retained.update({first.units[0].revision_id:first.units})
        before=counts(args['authority_path']);clock[0]+=timedelta(minutes=1)
        repeated=poll.poll()[0]
        assert repeated.status=='READY' and repeated.observations==first.observations
        assert counts(args['authority_path'])==before
    with open_native_runtime(**args) as runtime:
        before=counts(args['authority_path']);clock[0]+=timedelta(minutes=1)
        reopened=intake(runtime,definition).poll()[0]
        assert reopened.status=='READY' and reopened.observations==first.observations
        assert counts(args['authority_path'])==before
    assert len(calls)==6
    assert observed==[datetime(2026,9,8,12,minute,tzinfo=UTC)for minute in range(3)]


import pytest
from newsroom.authority import AuthenticationProof,ObjectAdmissionId
from newsroom.authority.canonical import digest_bytes


@pytest.fixture
def retained_poll(tmp_path,monkeypatch):
    args=_args(tmp_path,monkeypatch);args.update(principal_id=OPERATOR_PRINCIPAL_ID,authority_domain=OPERATOR_AUTHORITY_DOMAIN)
    raw=[_document()];observations={};retained={};calls=[]
    def fetch(url):calls.append(url);return 200,ATOM if url==SOURCE_URLS['UK-01']else raw[0]
    with open_native_runtime(**args) as runtime:
        definition=_seed_uk01(runtime)
        poll=NativeSourceIntake(sources=runtime.authority.sources,objects=runtime.authority.objects,
            proof=runtime.proof,definition_ids={'UK-01':definition},licence=_licence(),
            dispatch_fence=lambda *_:nullcontext(),retained_units=retained,observations=observations,
            fetch=fetch,clock=lambda:datetime(2026,9,8,12,tzinfo=UTC))
        first=poll.poll()[0];assert first.status=='READY'
        observations.update({r[1]:r for r in first.observations});retained[first.units[0].revision_id]=first.units
        yield args,runtime,poll,raw,observations,calls,first


def test_changed_bytes_admit_new_raw_and_record_new_revision(retained_poll):
    args,runtime,poll,raw,observations,calls,first=retained_poll
    before=counts(args['authority_path']);raw[0]=_document(body='Changed complete page.',updated='2026-09-08T11:30:00Z')
    changed=poll.poll()[0]
    assert changed.status=='READY' and changed.units[0].revision_id!=first.units[0].revision_id
    assert digest_bytes(raw[0])!=first.units[0].observation_digest
    assert counts(args['authority_path'])[0]>before[0]
    assert len(calls)==4


@pytest.mark.parametrize('damage',('tampered_bytes','revoked','bad_authentication','malformed_reference'))
def test_retained_observation_never_masks_permission_or_integrity_denial(retained_poll,damage):
    args,runtime,poll,raw,observations,calls,first=retained_poll
    reference=observations[digest_bytes(ATOM)]
    if damage=='tampered_bytes':
        h=reference[1].split(':')[1]
        path=args['object_root']/'objects'/h[:2]/h
        path.chmod(0o600);path.write_bytes(b'corrupt');path.chmod(0o400)
    elif damage=='revoked':
        runtime.authority.objects.revoke(ObjectAdmissionId.parse(reference[2]),reason_code='REVOKED',
            idempotency_key='fixture-revoke-observation',proof=runtime.proof)
    elif damage=='bad_authentication':poll._proof=AuthenticationProof(method='STATIC_TOKEN',credential='wrong')
    else:observations[reference[1]]=(reference[0],'sha256:'+'f'*64,reference[2],reference[3])
    before=counts(args['authority_path']);result=poll.poll()[0]
    assert result.status=='HOLD'
    assert counts(args['authority_path'])==before


def test_wrong_source_rights_denies_before_same_url_artifact_read(retained_poll,monkeypatch):
    from types import SimpleNamespace
    from newsroom.control_plane.native_evidence import PublicationRightsAssessment
    args,runtime,poll,raw,observations,calls,first=retained_poll
    rights=PublicationRightsAssessment.create(decision='HOLD',permitted_use='PUBLICATION_EVIDENCE',
        policy_digest='sha256:'+'a'*64,evidence_digest='sha256:'+'b'*64)
    poll._licence=SimpleNamespace(for_source=lambda **_:rights)
    monkeypatch.setattr(type(runtime.authority.objects),'rehydrate',lambda *a,**k:pytest.fail('rights-denied artifact consumed'))
    before=counts(args['authority_path']);result=poll.poll()[0]
    assert result.reason_code=='CURRENT_RIGHTS_HOLD' and len(calls)==2
    assert counts(args['authority_path'])==before
