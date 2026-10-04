"""Retained evidence reads reuse proofs without weakening fresh authority."""
from contextlib import contextmanager

import pytest

from newsroom.authority import AuthenticationProof,ObjectAdmissionRequest,StaticAuthorizer,AuthenticationError,AuthorizationDenied,ObjectAdmissionDenied,ObjectHydrationDenied,ObjectIntegrityError
from newsroom.authority.canonical import canonical_json_bytes
from newsroom.increment10.ingress import open_evidence_intake_ingress
from newsroom.tests.authority_helpers import proof
from newsroom.tests.test_increment10_evidence import _open_objects,_package_and_records,_facade
from newsroom.tests.test_increment10_ingress import _candidate,_receive


@contextmanager
def package_fixture(tmp_path):
    connection,port,version=_candidate(tmp_path)
    ingress=open_evidence_intake_ingress(tmp_path/'intake.sqlite3')
    received=_receive(ingress,connection,port,version,request_id='read-reuse')
    objects,policies=_open_objects(tmp_path/'objects.sqlite3')
    try:
        passage='The official deadline changed.'
        package,records=_package_and_records(version,passage)
        source=objects.objects.admit(ObjectAdmissionRequest('evidence.source','source'),passage.encode(),proof=proof()).admission
        ids=tuple(objects.objects.admit(ObjectAdmissionRequest('evidence.record',f'record-{n}'),canonical_json_bytes(row),proof=proof()).admission.admission_id for n,row in enumerate(records))
        facade=_facade(objects,ingress,policies)
        connection.execute('BEGIN IMMEDIATE')
        retained=facade.retain(package,receipt_id=received.receipt_id,candidate_port=port,
            source_admission_ids=(source.admission_id,),record_admission_ids=ids,proof=proof())
        yield objects,facade,port,retained,source
    finally:
        connection.rollback();objects.close();ingress.close();connection.close()


def counts(objects):
    store=objects.objects._GovernedObjects__hydrate.__self__._store
    return {t:store._connection.execute('SELECT count(*) FROM '+t).fetchone()[0]
            for t in ('object_access_decisions','authentication_contexts','authorization_requests','authorization_decisions')}


def test_same_credential_retained_package_read_has_no_diagnostic_growth(tmp_path,monkeypatch):
    with package_fixture(tmp_path) as (objects,facade,port,retained,_):
        boundary=objects.objects._GovernedObjects__hydrate.__self__
        authenticate=boundary._authenticate;calls=[]
        def fresh(proof):calls.append(True);return authenticate(proof)
        monkeypatch.setattr(boundary,'_authenticate',fresh)
        first=facade.read(retained.package_admission_id,candidate_port=port,proof=proof())
        before=counts(objects);called=len(calls)
        assert facade.read(retained.package_admission_id,candidate_port=port,proof=proof())==first==retained
        assert len(calls)>called # Fresh authentication is not a memoised proof.
        assert counts(objects)==before


@pytest.mark.parametrize('changed',['revoked','corrupt','permission','credential'])
def test_reused_package_read_rechecks_current_authority_and_cas(tmp_path,changed):
    with package_fixture(tmp_path) as (objects,facade,port,retained,source):
        facade.read(retained.package_admission_id,candidate_port=port,proof=proof())
        boundary=objects.objects._GovernedObjects__hydrate.__self__
        actual_proof=proof();restore=None
        if changed=='revoked':
            objects.objects.revoke(source.admission_id,reason_code='RIGHTS_REVOKED',idempotency_key='revoke-read',proof=proof())
        elif changed=='corrupt':
            blob=tmp_path/'objects.objects'/'objects'/source.blob.blob_digest[7:9]/source.blob.blob_digest[7:]
            raw=blob.read_bytes();mode=blob.stat().st_mode&0o777
            blob.chmod(0o600);blob.write_bytes(b'corrupt');blob.chmod(mode)
            def restore():blob.chmod(0o600);blob.write_bytes(raw);blob.chmod(mode)
        elif changed=='permission':
            boundary._authorizer=StaticAuthorizer(policy_version='authz-revoked',grants_by_principal={'principal.alpha':frozenset()})
        else:actual_proof=AuthenticationProof(method='STATIC_TOKEN',credential='unknown')
        try:
            expected={'revoked':(ObjectAdmissionDenied,ObjectHydrationDenied),'corrupt':ObjectIntegrityError,'permission':AuthorizationDenied,'credential':AuthenticationError}[changed]
            with pytest.raises(expected):facade.read(retained.package_admission_id,candidate_port=port,proof=actual_proof)
        finally:
            if restore:restore()


def test_reused_package_read_rejects_real_expired_rights(tmp_path,monkeypatch):
    from dataclasses import replace
    from datetime import timedelta
    from newsroom.tests import test_increment10_evidence as evidence_fixture
    from newsroom.tests.authority_a2b_helpers import MutableClock
    from newsroom.tests.authority_helpers import FIXED_NOW
    from newsroom.authority import RightsPolicyRegistry,ObjectAdmissionRegistry,UtcTimestamp
    clock=MutableClock(FIXED_NOW)
    policy_factory=evidence_fixture._policies
    def expiring():
        rights,hydration,admissions,contracts,definitions=policy_factory()
        changed=RightsPolicyRegistry(tuple(replace(item,validity_seconds=30) for item in rights.contracts()))
        mapping={old.contract_digest:new.contract_digest for old,new in zip(rights.contracts(),changed.contracts())}
        definitions=tuple(replace(item,rights_policy_contract_digest=mapping[item.rights_policy_contract_digest]) for item in definitions)
        admissions=ObjectAdmissionRegistry(definitions,rights_policies=changed,hydration_policies=hydration)
        return changed,hydration,admissions,contracts,definitions
    open_objects=evidence_fixture.open_object_system
    monkeypatch.setattr(evidence_fixture,'_policies',expiring)
    monkeypatch.setattr(evidence_fixture,'open_object_system',lambda path,**kwargs:open_objects(path,clock=clock,**kwargs))
    with package_fixture(tmp_path) as (objects,facade,port,retained,_):
        facade.read(retained.package_admission_id,candidate_port=port,proof=proof())
        before=counts(objects)
        clock.current=UtcTimestamp(FIXED_NOW.value+timedelta(seconds=30))
        with pytest.raises((ObjectAdmissionDenied,ObjectHydrationDenied),match='expired|EXPIRED'):
            facade.read(retained.package_admission_id,candidate_port=port,proof=proof())
        assert counts(objects)==before
