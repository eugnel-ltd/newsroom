"""Repeated retained ACK reads reuse proof without stale permission or CAS."""
import pytest
from newsroom.authority import ObjectAdmissionId
from newsroom.authority.objects import ObjectIntegrityError
from newsroom.tests import test_increment10_private_serving as serving
from newsroom.tests.test_native_publication import _bindings
from newsroom.tests.authority_helpers import proof
from newsroom.control_plane.native_publication import NativePublicationController, NativePublicationError


def test_repeated_exact_ack_read_reuses_access_and_still_denies_revocation(tmp_path):
    context=serving._context(tmp_path);delivery=serving._delivery(tmp_path,context);controller=None
    try:
        request=dict(story_receipt=context[-2],candidate_port=context[1],proof=proof())
        attempt,_=delivery.begin(context[-1],**request)
        delivery.apply(attempt,publication_receipt=context[-1],applied_at='2026-07-16T11:00:00Z',**request)
        observed=delivery.observe(attempt,publication_receipt=context[-1],observed_at='2026-07-16T11:30:00Z',**request)
        evidence=delivery.record(observed,attempt,expected_version=0,proof=proof())
        refs=dict(story_event_id=context[-2].event_id,publication_event_id=context[-1].event_id,
            delivery_attempt_event_id=attempt.event_id,delivery_evidence_event_id=evidence.event_id)
        controller=NativePublicationController(objects=context[3].objects,commands=context[3].commands,
            events=context[3].events,candidate_port=context[1],evidence_packages=context[8],bindings=_bindings(tmp_path,*context[4:8]))
        expected=controller.read_acknowledged(refs,proof=proof())
        root=context[3].objects._GovernedObjects__hydrate.__self__._store
        before=root._connection.execute('SELECT count(*) FROM object_access_decisions').fetchone()[0]
        for _ in range(3):assert controller.read_acknowledged(refs,proof=proof())==expected
        assert root._connection.execute('SELECT count(*) FROM object_access_decisions').fetchone()[0]==before
        # Normal current authority is checked on the next read, not cached.
        surface=root._connection.execute("SELECT admission_id FROM object_admissions WHERE object_class='surface_payload' LIMIT 1").fetchone()[0]
        context[3].objects.revoke(ObjectAdmissionId.parse(surface),reason_code='REVOKED',idempotency_key='reuse-proof-revoke',proof=proof())
        with pytest.raises(NativePublicationError):controller.read_acknowledged(refs,proof=proof())
    finally:
        if controller is not None:controller.close()
        serving._close(context,delivery)
