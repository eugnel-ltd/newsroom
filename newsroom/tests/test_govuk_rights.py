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
