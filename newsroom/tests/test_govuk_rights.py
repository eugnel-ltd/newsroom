import io
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from datetime import UTC, datetime

import pytest

from newsroom.control_plane import govuk_rights as rights
from newsroom.control_plane.native_evidence import NativeEvidenceHold
from newsroom.control_plane.veto import VetoError
from newsroom.tests.test_native_runtime import _args
from newsroom.control_plane.native_runtime import open_native_runtime


def test_observed_licence_retained_once_and_unknown_scope_held(tmp_path, monkeypatch):
    # F1 fixture terms are pinned separately; never production licence evidence.
    responses = {
        rights.REUSE_URL: b"<main>Test GOV reuse policy</main>",
        rights.LICENCE_URL: b"<main>Test OGL terms</main>",
    }
    monkeypatch.setattr(rights, "REVIEWED_TEXT", {
        url: rights.licence_text_digest(raw) for url, raw in responses.items()
    })
    calls = []
    class Response(io.BytesIO):
        status = 200
        def __init__(self, url):
            super().__init__(responses[url]); self.url = url
        def geturl(self): return self.url
    class Opener:
        def open(self, request, timeout):
            calls.append((request.full_url, timeout))
            return Response(request.full_url)
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    args = _args(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        fence_calls = []
        params = dict(objects=runtime.authority.objects, proof=runtime.proof,
                      dispatch_fence=lambda: nullcontext(fence_calls.append(True)),
                      clock=lambda: datetime(2026, 9, 8, 10, tzinfo=UTC))
        first = rights.retain_current_govuk_licence(**params)
        second = rights.retain_current_govuk_licence(**params)
        assert first == second
        assert len(calls) == 4 and len(fence_calls) == 4
        assert len(first.admission_ids) == 2
        assert first.for_source(source_id="UK-01", definition_url="https://www.gov.uk/feed").decision == "PERMITTED"
        assert first.for_source(source_id="RAD-02", definition_url="https://www.gov.uk/feed").decision == "HOLD"
        assert first.for_source(source_id="UK-01", definition_url="https://www.gov.uk.evil.test/feed").decision == "HOLD"
        responses[rights.LICENCE_URL] = b"<main>Changed licence terms</main>"
        with pytest.raises(NativeEvidenceHold, match="LICENCE_REVIEW_HOLD"):
            rights.retain_current_govuk_licence(**params)
        assert len(fence_calls) == 6


def test_licence_substantive_text_changes_are_detected():
    first = rights.licence_text_digest(b"<main><p>Re-use is permitted.</p></main>")
    assert first == rights.licence_text_digest(b"<aside>menu</aside><main> Re-use  is permitted. </main>")
    assert first != rights.licence_text_digest(b"<main>Re-use is prohibited.</main>")


@pytest.mark.parametrize("metadata", [
    b'<div class="gem-c-published-dates"><h2>Updates to this page</h2>Last updated 9 December 2022</div>',
    b'<div class="gem-c-metadata"><dl><dt>Last updated:</dt><dd>9 December 2022</dd></dl></div>',
])
def test_licence_digest_canonicalises_only_recognised_publication_date_chrome(metadata):
    terms = b"<h1>Reuse policy</h1><p>Re-use is permitted with attribution.</p>"
    expected = rights.licence_text_digest(
        b"<main>" + terms + b"Updates to this page Last updated 9 December 2022</main>"
    )
    assert rights.licence_text_digest(b"<main>" + terms + metadata + b"</main>") == expected
    # Substantive text inside recognised chrome, following it, or elsewhere
    # remains part of the reviewed contract. A CSS class is not a bypass.
    for changed in (
        metadata.replace(b"2022", b"2022. Commercial use is prohibited."),
        metadata + b"Commercial use is prohibited.",
        metadata.replace(b"gem-c-", b"unknown-"),
        metadata.replace(b"2022", b"2023"),
    ):
        assert rights.licence_text_digest(b"<main>" + terms + changed + b"</main>") != expected


@pytest.mark.parametrize("stop_at", [1, 2])
def test_licence_fence_veto_propagates_before_partial_retention(monkeypatch, stop_at):
    raw = b"<main>Reviewed fixture terms.</main>"
    monkeypatch.setattr(rights, "REVIEWED_TEXT", {
        url: rights.licence_text_digest(raw)
        for url in (rights.REUSE_URL, rights.LICENCE_URL)
    })
    calls = []
    class Response(io.BytesIO):
        status = 200
        def __init__(self, url):
            super().__init__(raw)
            self.url = url
        def geturl(self):
            return self.url
    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            return Response(request.full_url)
    monkeypatch.setattr(rights.urllib.request, "build_opener", lambda *_: Opener())
    entered = 0
    @contextmanager
    def fence():
        nonlocal entered
        entered += 1
        if entered == stop_at:
            raise VetoError("owner stop during licence acquisition")
        yield
    objects = SimpleNamespace(admit=lambda *_a, **_k: pytest.fail("partial licence retention"))
    with pytest.raises(VetoError, match="owner stop"):
        rights.retain_current_govuk_licence(objects=objects, proof=object(), dispatch_fence=fence)
    assert len(calls) == stop_at - 1


def test_nonce_only_licence_refresh_reuses_semantic_assessment_but_retains_exact_observation(tmp_path, monkeypatch):
    import json
    from newsroom.authority import HydrationRequest
    from newsroom.authority.types import ObjectAdmissionId
    from newsroom.control_plane import native_source_rights

    responses = {url: b'<html><head><meta name="csp-nonce" content="one"></head><body><main>Reviewed fixture terms.</main></body></html>'
                 for url in (rights.REUSE_URL, rights.LICENCE_URL)}
    monkeypatch.setattr(rights, 'REVIEWED_TEXT', {url: rights.licence_text_digest(raw) for url, raw in responses.items()})
    fetches = []
    class Response(io.BytesIO):
        status = 200
        def __init__(self, url):
            super().__init__(responses[url]); self.url = url
        def geturl(self): return self.url
    class Opener:
        def open(self, request, timeout):
            fetches.append(request.full_url); return Response(request.full_url)
    monkeypatch.setattr(rights.urllib.request, 'build_opener', lambda *_: Opener())
    args = _args(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        objects, proof = runtime.authority.objects, runtime.proof
        params = dict(objects=objects, proof=proof, dispatch_fence=nullcontext)
        first = rights.retain_current_govuk_licence(**params, clock=lambda: datetime(2026, 9, 8, 10, tzinfo=UTC))
        source = dict(source_id='UK-01', definition_url='https://www.gov.uk/feed')
        first_assessment = first.for_source(**source)
        def snapshot(licence, assessment):
            return native_source_rights.retain_rights_snapshot(
                objects=objects, proof=proof, **source, assessment=assessment,
                observed_at=licence.observed_at, reason='REVIEWED_REUSE_PERMITTED',
                observations=tuple((url, digest, str(admission), '') for url, digest, admission in zip(
                    (rights.REUSE_URL, rights.LICENCE_URL), licence.raw_digests, licence.admission_ids, strict=True)),
                govuk_semantic_evidence=rights._govuk_semantic_evidence(**source),
            )
        one = snapshot(first, first_assessment)
        responses.update({url: raw.replace(b'content="one"', b'content="two"') for url, raw in responses.items()})
        second = rights.retain_current_govuk_licence(**params, clock=lambda: datetime(2026, 9, 8, 11, tzinfo=UTC))
        second_assessment = second.for_source(**source)
        two = snapshot(second, second_assessment)
        assert first.raw_digests != second.raw_digests and first.admission_ids != second.admission_ids
        assert first_assessment == second_assessment
        assert one.assessment_admission_id == two.assessment_admission_id
        assert one.assessment_blob_digest == two.assessment_blob_digest
        assert one.observation_admission_id != two.observation_admission_id
        assert len(fetches) == 4
        second.require_retained(objects=objects, proof=proof)
        data = objects.rehydrate(HydrationRequest(ObjectAdmissionId.parse(two.assessment_admission_id), 'evidence.source'), proof=proof).data
        document = json.loads(data)
        assert document['schema'] == 'hermes-native-rights-assessment-v2'
        assert document['record_id'] == second_assessment.record_id
        assert document['evidence_digest'] == second_assessment.evidence_digest
        assert 'evidence' not in document
        observed = json.loads(objects.rehydrate(HydrationRequest(ObjectAdmissionId.parse(two.observation_admission_id), 'evidence.source'), proof=proof).data)
        assert [item[1] for item in observed['observations']] == list(second.raw_digests)
        assert [item[2] for item in observed['observations']] == [str(x) for x in second.admission_ids]
        assert first.for_source(source_id='UK-02', definition_url=source['definition_url']).record_id != first_assessment.record_id
        assert first.for_source(source_id='UK-01', definition_url='https://www.gov.uk/other').record_id != first_assessment.record_id


@pytest.mark.parametrize('fault', ['current_credential', 'revoked_raw', 'cas_bytes', 'reviewed_terms', 'stale_policy'])
def test_semantic_identity_does_not_replace_current_retained_licence_checks(tmp_path, monkeypatch, fault):
    from dataclasses import replace
    from newsroom.authority import AuthenticationError, ObjectAdmissionDenied, ObjectAdmissionRequest, ObjectIntegrityError
    from newsroom.authority.canonical import digest_bytes

    raw = b'<main>Reviewed fixture terms.</main>'
    monkeypatch.setattr(rights, 'REVIEWED_TEXT', {url: rights.licence_text_digest(raw) for url in (rights.REUSE_URL, rights.LICENCE_URL)})
    args = _args(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        objects, proof = runtime.authority.objects, runtime.proof
        admissions = tuple(objects.admit(ObjectAdmissionRequest('evidence.source', f'fixture-raw-{index}'), raw, proof=proof).admission for index in range(2))
        licence = rights.GovUkLicenceEvidence(tuple(x.admission_id for x in admissions), (digest_bytes(raw),)*2, '2026-09-08T10:00:00Z', rights.POLICY_DIGEST)
        assessment = licence.for_source(source_id='UK-01', definition_url='https://www.gov.uk/feed')
        assert assessment.decision == 'PERMITTED'
        if fault == 'current_credential':
            proof = replace(proof, credential='not-the-current-credential')
            error = AuthenticationError
        elif fault == 'revoked_raw':
            objects.revoke(admissions[0].admission_id, reason_code='REVOKED', idempotency_key='revoke-fixture-licence', proof=proof)
            error = ObjectAdmissionDenied
        elif fault == 'cas_bytes':
            digest = admissions[0].blob.blob_digest.split(':', 1)[1]
            path = args['object_root'] / 'objects' / digest[:2] / digest
            path.chmod(0o600); path.write_bytes(raw.replace(b'Reviewed', b'Tampered')); path.chmod(0o400)
            error = ObjectIntegrityError
        elif fault == 'reviewed_terms':
            monkeypatch.setattr(rights, 'REVIEWED_TEXT', {url: 'sha256:'+'0'*64 for url in (rights.REUSE_URL, rights.LICENCE_URL)})
            error = NativeEvidenceHold
        else:
            licence = replace(licence, policy_digest='sha256:'+'0'*64)
            assert licence.for_source(source_id='UK-01', definition_url='https://www.gov.uk/feed').decision == 'HOLD'
            error = NativeEvidenceHold
        with pytest.raises(error):
            licence.require_retained(objects=objects, proof=proof)


@pytest.mark.parametrize('bad_url', (None, rights.REUSE_URL, rights.LICENCE_URL))
def test_prefetched_licence_preserves_order_atomic_validation_and_serial_retention(bad_url):
    import threading
    import dataclasses
    from newsroom.control_plane.native_source_rights import fetch_licensing_observations
    from newsroom.control_plane import native_source_rights
    owner, admitted, fences = threading.get_ident(), [], []
    raw = {rights.REUSE_URL:b'<main>fixture reuse</main>',rights.LICENCE_URL:b'<main>fixture ogl</main>'}
    def fetched(url):
        if url == bad_url:raise OSError('fixture unavailable')
        return raw.get(url,b'<main>fixture portfolio</main>')
    @contextmanager
    def fence():
        assert threading.get_ident() == owner
        fences.append(True);yield
    observed = fetch_licensing_observations(stop_check=lambda:None,stop_fence=fence,
        govuk_fetch=fetched,portfolio_fetch=fetched)
    def retained_fetch(url):
        value = observed[url]
        if isinstance(value,Exception):raise value
        return value
    class Objects:
        def admit(self, request, body, *, proof):
            assert threading.get_ident() == owner
            admitted.append(body)
            return SimpleNamespace(admission=SimpleNamespace(admission_id='fixture-'+str(len(admitted)),
                blob=SimpleNamespace(blob_digest=rights.digest_bytes(body))))
    expected = {url:rights.licence_text_digest(body) for url,body in raw.items()}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rights,'REVIEWED_TEXT',expected)
        if bad_url:
            with pytest.raises(NativeEvidenceHold,match='GOVUK_LICENCE_REVIEW_HOLD'):
                rights.retain_current_govuk_licence(objects=Objects(),proof=object(),dispatch_fence=fence,fetch=retained_fetch)
            assert admitted == []
            assert all(not isinstance(observed[url],Exception) for terms in native_source_rights.TERMS.values() for url,_ in terms)
        else:
            licence=rights.retain_current_govuk_licence(objects=Objects(),proof=object(),dispatch_fence=fence,fetch=retained_fetch)
            assert admitted == [raw[rights.REUSE_URL],raw[rights.LICENCE_URL]]
            assert licence.raw_digests == tuple(rights.digest_bytes(raw[url]) for url in (rights.REUSE_URL,rights.LICENCE_URL))


@pytest.mark.parametrize('damage', ('status','redirect','empty','oversize'))
def test_network_only_licence_transport_preserves_exact_bounds(monkeypatch, damage):
    class Response(io.BytesIO):
        status = 503 if damage == 'status' else 200
        def __init__(self):super().__init__(b'' if damage=='empty' else b'x'*(rights.MAX_BODY_BYTES+1) if damage=='oversize' else b'<main>terms</main>')
        def geturl(self):return 'https://wrong.example/' if damage=='redirect' else rights.REUSE_URL
    class Opener:
        def open(self,request,timeout):
            assert request.get_method()=='GET' and timeout==20
            assert request.get_header('User-agent')=='Newsroom-Hermes-Rights-Review/1.0'
            assert request.get_header('Accept-encoding')=='identity'
            return Response()
    monkeypatch.setattr('urllib.request.build_opener',lambda *_args:Opener())
    with pytest.raises(ValueError):rights._fetch_licence_observation(rights.REUSE_URL)
