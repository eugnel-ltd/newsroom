import json
import sqlite3
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from newsroom.authority import (
    AuthenticationError, HydrationRequest, ObjectAdmissionDenied, ObjectAdmissionId,
    ObjectAdmissionRequest, ObjectIntegrityError,
)
from newsroom.authority.canonical import digest_bytes
from newsroom.control_plane import native_source_rights as rights
from newsroom.control_plane import cycle
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.veto import VetoError
from newsroom.increment9.proving import SOURCE_URLS
from newsroom.tests.test_native_runtime import _args
from newsroom.tests.test_native_source_intake import _licence, _source_read_audit_counts


def test_observed_rights_retain_real_terms_and_keep_source_local_holds(tmp_path, monkeypatch):
    bodies = {}
    terms = {}
    for source, entries in rights.TERMS.items():
        terms[source] = []
        for index, (url, _digest) in enumerate(entries):
            text = f"Reviewed test terms for {source} {index}"
            raw = (f'<div class="inner_page_content_container">{text}</div>' if source == "HK-04"
                   else f"<main>{text}</main>").encode()
            bodies[url] = raw
            terms[source].append((url, rights.terms_text_digest(source, raw)))
        terms[source] = tuple(terms[source])
    monkeypatch.setattr(rights, "TERMS", terms)
    calls = []
    def fetch(url):
        calls.append(url)
        return bodies[url]
    with open_native_runtime(**_args(tmp_path, monkeypatch)) as runtime:
        evidence = rights.observe_portfolio_terms(
            objects=runtime.authority.objects, proof=runtime.proof,
            stop_check=lambda: None, stop_fence=nullcontext, fetch=fetch,
            clock=lambda: datetime(2026, 9, 8, 15, tzinfo=UTC),
        )
        portfolio = rights.NativePortfolioRights(_licence(), evidence)
        assert len(calls) == 7 and set(evidence) == set(rights.TERMS)
        for source, entry in evidence.items():
            expected = "HOLD" if source in rights.RESTRICTIONS else "PERMITTED"
            assert portfolio.for_source(source_id=source, definition_url=SOURCE_URLS[source]).decision == expected
            for url, digest, admission, _access in entry.observations:
                retained = runtime.authority.objects.hydrate(
                    HydrationRequest(ObjectAdmissionId.parse(admission), "evidence.source"), proof=runtime.proof,
                )
                assert retained.data == bodies[url] and digest_bytes(retained.data) == digest
        assert portfolio.for_source(source_id="HK-02", definition_url=SOURCE_URLS["RAD-02"]).decision == "HOLD"
        with pytest.raises(ValueError, match="incomplete"):
            replace(evidence["HK-02"], observations=())
        with pytest.raises(ValueError, match="restriction"):
            replace(evidence["RAD-02"], reason="REVIEWED_REUSE_PERMITTED")
        hk02_url = rights.TERMS["HK-02"][0][0]
        bodies[hk02_url] = b"<main>Changed HK Observatory terms</main>"
        changed = rights.observe_portfolio_terms(
            objects=runtime.authority.objects, proof=runtime.proof,
            stop_check=lambda: None, stop_fence=nullcontext, fetch=fetch,
            clock=lambda: datetime(2026, 9, 8, 15, 5, tzinfo=UTC),
        )
        refreshed = rights.NativePortfolioRights(_licence(), changed)
        assert changed["HK-02"].reason == "SOURCE_TERMS_CHANGED"
        assert refreshed.for_source(
            source_id="HK-02", definition_url=SOURCE_URLS["HK-02"],
        ).decision == "HOLD"
        assert refreshed.for_source(
            source_id="UK-10", definition_url=SOURCE_URLS["UK-10"],
        ).decision == "PERMITTED"
        assessment = refreshed.for_source(
            source_id="HK-02", definition_url=SOURCE_URLS["HK-02"],
        )
        snapshot = rights.retain_rights_snapshot(
            objects=runtime.authority.objects, proof=runtime.proof,
            source_id="HK-02", definition_url=SOURCE_URLS["HK-02"],
            assessment=assessment, observed_at=changed["HK-02"].observed_at,
            reason=changed["HK-02"].reason,
            observations=changed["HK-02"].observations,
        )
        for admission, digest in (
            (snapshot.assessment_admission_id, snapshot.assessment_blob_digest),
            (snapshot.observation_admission_id, snapshot.observation_blob_digest),
        ):
            retained = runtime.authority.objects.hydrate(
                HydrationRequest(ObjectAdmissionId.parse(admission), "evidence.source"),
                proof=runtime.proof,
            )
            assert digest_bytes(retained.data) == digest
        later = rights.retain_rights_snapshot(
            objects=runtime.authority.objects, proof=runtime.proof,
            source_id="HK-02", definition_url=SOURCE_URLS["HK-02"],
            assessment=assessment, observed_at="2026-09-08T15:10:00+00:00",
            reason=changed["HK-02"].reason,
            observations=changed["HK-02"].observations,
        )
        assert later.assessment_admission_id == snapshot.assessment_admission_id
        assert later.assessment_blob_digest == snapshot.assessment_blob_digest
        assert later.observation_admission_id != snapshot.observation_admission_id
        assert later.observation_blob_digest != snapshot.observation_blob_digest
        observation = runtime.authority.objects.hydrate(
            HydrationRequest(
                ObjectAdmissionId.parse(later.observation_admission_id),
                "evidence.source",
            ), proof=runtime.proof,
        )
        assert json.loads(observation.data)["observed_at"] == (
            "2026-09-08T15:10:00+00:00"
        )


def test_terms_stop_and_unavailable_do_not_create_a_permission(tmp_path, monkeypatch):
    with open_native_runtime(**_args(tmp_path, monkeypatch)) as runtime:
        def fail(_url): raise OSError("test transport unavailable")
        evidence = rights.observe_portfolio_terms(objects=runtime.authority.objects, proof=runtime.proof,
                                                  stop_check=lambda: None, stop_fence=nullcontext, fetch=fail)
        portfolio = rights.NativePortfolioRights(_licence(), evidence)
        assert all(value.reason == "SOURCE_TERMS_UNAVAILABLE" for value in evidence.values())
        assert all(portfolio.for_source(source_id=source, definition_url=SOURCE_URLS[source]).decision == "HOLD" for source in evidence)
        def stop(): raise VetoError("owner stop")
        with pytest.raises(VetoError):
            rights.observe_portfolio_terms(objects=runtime.authority.objects, proof=runtime.proof,
                                            stop_check=stop, stop_fence=nullcontext,
                                            fetch=lambda _: pytest.fail("fetch after stop"))


def test_parallel_terms_fetches_share_one_real_owner_stop_fence(
    tmp_path, monkeypatch,
):
    bodies = {}
    terms = {}
    for source_id in rights.TERMS:
        url = rights.TERMS[source_id][0][0]
        text = f"Concurrent terms for {source_id}"
        raw = (
            f'<div class="inner_page_content_container">{text}</div>'
            if source_id == "HK-04" else f"<main>{text}</main>"
        ).encode()
        bodies[url] = raw
        terms[source_id] = ((url, rights.terms_text_digest(source_id, raw)),)
    monkeypatch.setattr(rights, "TERMS", terms)
    monkeypatch.setattr(cycle, "_PROVING_FENCE_TIMEOUT_SECONDS", 0.05)

    proving = tmp_path / "proving.sqlite3"
    with sqlite3.connect(proving) as database:
        database.executescript(
            """
            CREATE TABLE proving_runs(run_id TEXT PRIMARY KEY);
            CREATE TABLE proving_gates(
                run_id TEXT NOT NULL, gate_id TEXT NOT NULL, status TEXT NOT NULL
            );
            INSERT INTO proving_runs VALUES('run-1');
            INSERT INTO proving_gates VALUES(
                'run-1', 'NO_ACTIVE_HUMAN_EMERGENCY_STOP', 'PASS'
            );
            """
        )

    fence_entries = 0

    @contextmanager
    def owner_stop_fence():
        nonlocal fence_entries
        fence_entries += 1
        with cycle.owner_emergency_stop_fence(str(proving)):
            yield

    all_fetches_entered = threading.Event()
    state_lock = threading.Lock()
    active = maximum_active = 0

    def fetch(url):
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active == len(terms):
                all_fetches_entered.set()
        try:
            assert all_fetches_entered.wait(0.5)
            return bodies[url]
        finally:
            with state_lock:
                active -= 1

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(mode=0o700)
    with open_native_runtime(**_args(runtime_root, monkeypatch)) as runtime:
        evidence = rights.observe_portfolio_terms(
            objects=runtime.authority.objects, proof=runtime.proof,
            stop_check=lambda: None, stop_fence=owner_stop_fence, fetch=fetch,
            clock=lambda: datetime(2026, 9, 8, 15, tzinfo=UTC),
        )

    assert set(evidence) == set(terms)
    assert fence_entries == 1
    assert maximum_active == len(terms)


def test_owner_stop_fence_is_reentrant_only_on_the_holding_thread(
    tmp_path, monkeypatch,
):
    proving = tmp_path / "proving-fence.sqlite3"
    with sqlite3.connect(proving) as database:
        database.executescript(
            """
            CREATE TABLE proving_runs(run_id TEXT PRIMARY KEY);
            CREATE TABLE proving_gates(
                run_id TEXT NOT NULL, gate_id TEXT NOT NULL, status TEXT NOT NULL
            );
            INSERT INTO proving_runs VALUES('run-1');
            INSERT INTO proving_gates VALUES(
                'run-1', 'NO_ACTIVE_HUMAN_EMERGENCY_STOP', 'PASS'
            );
            """
        )
    monkeypatch.setattr(cycle, "_PROVING_FENCE_TIMEOUT_SECONDS", 0.05)
    writer_errors = []

    def activate_from_another_thread():
        try:
            with sqlite3.connect(proving, timeout=0.05) as database:
                database.execute(
                    "UPDATE proving_gates SET status='FAIL' "
                    "WHERE gate_id='NO_ACTIVE_HUMAN_EMERGENCY_STOP'"
                )
        except sqlite3.OperationalError as exc:
            writer_errors.append(exc)

    with pytest.raises(RuntimeError, match="unwind fence"):
        with cycle.owner_emergency_stop_fence(str(proving)):
            with cycle.owner_emergency_stop_fence(str(proving.resolve())):
                cycle.assert_no_owner_emergency_stop(str(proving))
            # A child inheriting thread-local state must acquire its own lock.
            parent_pid = cycle.os.getpid()
            with monkeypatch.context() as child:
                child.setattr(cycle.os, "getpid", lambda: parent_pid + 1)
                with pytest.raises(VetoError, match="authority is unavailable"):
                    with cycle.owner_emergency_stop_fence(str(proving)):
                        pytest.fail("another process reused the parent fence")
            writer = threading.Thread(target=activate_from_another_thread)
            writer.start()
            writer.join(timeout=0.5)
            assert not writer.is_alive()
            assert len(writer_errors) == 1
            raise RuntimeError("unwind fence")

    with sqlite3.connect(proving) as database:
        database.execute(
            "UPDATE proving_gates SET status='FAIL' "
            "WHERE gate_id='NO_ACTIVE_HUMAN_EMERGENCY_STOP'"
        )
    with pytest.raises(VetoError, match="owner emergency stop"):
        with cycle.owner_emergency_stop_fence(str(proving)):
            pytest.fail("active owner stop entered the protected effect")


def test_portfolio_refresh_replaces_permitted_snapshot_without_timestamp_churn():
    first = rights.SourceTermsEvidence(
        "HK-02", "2026-09-08T10:00:00+00:00", "REVIEWED_REUSE_PERMITTED",
        ((rights.TERMS["HK-02"][0][0], "sha256:" + "1" * 64,
          "00000000-0000-4000-8000-000000000001", "access-1"),),
    )
    same = replace(first, observed_at="2026-09-08T10:05:00+00:00")
    unavailable = {
        source: rights.SourceTermsEvidence(
            source, "2026-09-08T10:05:00+00:00", "SOURCE_TERMS_UNAVAILABLE", (),
        )
        for source in rights.TERMS
    }
    unavailable["HK-02"] = same
    refs = {
        source: rights.RightsSnapshotReference(
            f"00000000-0000-4000-8000-{index:012d}", "sha256:" + "2" * 64,
            f"00000000-0000-4000-8001-{index:012d}", "sha256:" + "3" * 64,
        )
        for index, source in enumerate(SOURCE_URLS, 1)
    }
    snapshots = iter(((None, "GOVUK_LICENCE_REVIEW_HOLD", unavailable, refs),))
    portfolio = rights.NativePortfolioRights(
        _licence(), {"HK-02": first}, refresh_current=lambda: next(snapshots),
    )
    initial = portfolio.for_source(
        source_id="HK-02", definition_url=SOURCE_URLS["HK-02"],
    )
    assert initial.decision == "PERMITTED"
    assert first.digest == same.digest
    govuk = _licence()
    assert govuk.for_source(
        source_id="UK-01", definition_url=SOURCE_URLS["UK-01"],
    ).record_id == replace(
        govuk, observed_at="2026-09-08T10:05:00.000000Z",
    ).for_source(
        source_id="UK-01", definition_url=SOURCE_URLS["UK-01"],
    ).record_id
    portfolio.refresh()
    assert portfolio.for_source(
        source_id="HK-02", definition_url=SOURCE_URLS["HK-02"],
    ).decision == "PERMITTED"
    assert portfolio.for_source(
        source_id="UK-01", definition_url=SOURCE_URLS["UK-01"],
    ).decision == "HOLD"
    assert portfolio.reason_for("HK-02") == "REVIEWED_REUSE_PERMITTED"
    assert portfolio.reason_for("UK-01") == "GOVUK_LICENCE_REVIEW_HOLD"


def _one_source_terms(monkeypatch):
    url = rights.TERMS["HK-02"][0][0]
    raw = b"<main>Reviewed unchanged Observatory terms</main>"
    monkeypatch.setattr(rights, "TERMS", {
        "HK-02": ((url, rights.terms_text_digest("HK-02", raw)),),
    })
    return url, raw


def test_rights_refresh_reuses_reads_but_retains_each_new_observation(
    tmp_path, monkeypatch,
):
    url, raw = _one_source_terms(monkeypatch)
    args = _args(tmp_path, monkeypatch)
    calls = []

    def fetch(selected_url):
        calls.append(selected_url)
        return raw

    with open_native_runtime(**args) as runtime:
        objects, proof = runtime.authority.objects, runtime.proof
        # An already admitted page without an access receipt needs its first
        # fully governed read, not an invented or missing historical receipt.
        admitted = objects.admit(ObjectAdmissionRequest(
            "evidence.source", f"native-source-terms:HK-02:{digest_bytes(raw)}",
        ), raw, proof=proof).admission
        before = _source_read_audit_counts(args["authority_path"])
        first = rights.observe_portfolio_terms(
            objects=objects, proof=proof, stop_check=lambda: None,
            stop_fence=nullcontext, fetch=fetch,
            clock=lambda: datetime(2026, 9, 8, 15, tzinfo=UTC),
        )["HK-02"]
        assert first.observations[0][2] == str(admitted.admission_id)
        after_first = _source_read_audit_counts(args["authority_path"])
        assert after_first == tuple(n + 1 for n in before)
        later = rights.observe_portfolio_terms(
            objects=objects, proof=proof, stop_check=lambda: None,
            stop_fence=nullcontext, fetch=fetch,
            clock=lambda: datetime(2026, 9, 8, 15, 5, tzinfo=UTC),
        )["HK-02"]
        assert calls == [url, url]  # Current acquisition is not cached.
        assert _source_read_audit_counts(args["authority_path"]) == after_first
        assert later.observations == first.observations
        assert later.observed_at != first.observed_at
        portfolio = rights.NativePortfolioRights(None, {"HK-02": later})
        for _ in range(2):
            portfolio.require_retained(objects=objects, proof=proof)
        assert _source_read_audit_counts(args["authority_path"]) == after_first
        assessment = portfolio.for_source(
            source_id="HK-02", definition_url=SOURCE_URLS["HK-02"],
        )
        request = dict(
            source_id="HK-02", definition_url=SOURCE_URLS["HK-02"],
            assessment=assessment, reason=later.reason,
            observations=later.observations,
        )
        snapshot = rights.retain_rights_snapshot(
            objects=objects, proof=proof, observed_at=first.observed_at, **request,
        )
        after_snapshot = _source_read_audit_counts(args["authority_path"])
        assert after_snapshot == tuple(n + delta for n, delta in zip(
            after_first, (2, 8, 8, 8), strict=True,
        ))
        latest = rights.retain_rights_snapshot(
            objects=objects, proof=proof, observed_at=later.observed_at, **request,
        )
        assert latest.assessment_admission_id == snapshot.assessment_admission_id
        assert latest.observation_admission_id != snapshot.observation_admission_id
        assert _source_read_audit_counts(args["authority_path"]) == tuple(
            n + delta for n, delta in zip(after_snapshot, (1, 4, 4, 4), strict=True)
        )
    with open_native_runtime(**args) as reopened:
        before = _source_read_audit_counts(args["authority_path"])
        portfolio.require_retained(objects=reopened.authority.objects, proof=reopened.proof)
        assert rights.retain_rights_snapshot(
            objects=reopened.authority.objects, proof=reopened.proof,
            observed_at=later.observed_at, **request,
        ) == latest
        assert _source_read_audit_counts(args["authority_path"]) == before


@pytest.mark.parametrize(("fault", "error"), (
    ("current_credential", AuthenticationError),
    ("revocation", ObjectAdmissionDenied),
    ("cas_bytes", ObjectIntegrityError),
    ("expected_digest", ValueError),
    ("reviewed_terms", ValueError),
))
def test_retained_rights_read_rechecks_current_authority_and_exact_content(
    tmp_path, monkeypatch, fault, error,
):
    url, raw = _one_source_terms(monkeypatch)
    args = _args(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        objects, proof = runtime.authority.objects, runtime.proof
        evidence = rights.observe_portfolio_terms(
            objects=objects, proof=proof, stop_check=lambda: None,
            stop_fence=nullcontext, fetch=lambda _: raw,
        )
        entry = evidence["HK-02"]
        admission_id = ObjectAdmissionId.parse(entry.observations[0][2])
        if fault == "current_credential":
            proof = replace(proof, credential="not-the-current-credential")
        elif fault == "revocation":
            objects.revoke(
                admission_id, reason_code="REVOKED", idempotency_key="revoke-terms",
                proof=proof,
            )
        elif fault == "cas_bytes":
            [path] = [path for path in (args["object_root"] / "objects").rglob("*")
                      if path.is_file()]
            path.chmod(0o600)
            path.write_bytes(raw.replace(b"Reviewed", b"Tampered"))
            path.chmod(0o400)
        elif fault == "expected_digest":
            _, _, admission, access = entry.observations[0]
            evidence["HK-02"] = replace(entry, observations=(
                (url, "sha256:" + "f" * 64, admission, access),
            ))
        else:
            monkeypatch.setattr(rights, "TERMS", {
                "HK-02": ((url, "sha256:" + "f" * 64),),
            })
        portfolio = rights.NativePortfolioRights(None, evidence)
        before = _source_read_audit_counts(args["authority_path"])
        with pytest.raises(error):
            portfolio.require_retained(objects=objects, proof=proof)
        assert _source_read_audit_counts(args["authority_path"]) == before


@pytest.mark.parametrize('fault', ['source_id', 'source_url', 'reviewed_text', 'assessment_digest'])
def test_govuk_v2_semantic_input_mismatch_refuses_before_retention(fault):
    from newsroom.control_plane import govuk_rights
    from newsroom.control_plane.native_evidence import PublicationRightsAssessment
    from types import SimpleNamespace
    from newsroom.authority.canonical import digest_canonical

    source = dict(source_id='UK-01', definition_url='https://www.gov.uk/feed')
    semantic = govuk_rights._govuk_semantic_evidence(**source)
    assessment = PublicationRightsAssessment.create(
        decision='PERMITTED', permitted_use='PUBLICATION_EVIDENCE', policy_digest=govuk_rights.POLICY_DIGEST,
        evidence_digest=digest_canonical(semantic),
    )
    supplied = dict(semantic)
    if fault == 'source_id': supplied['source_id'] = 'UK-02'
    elif fault == 'source_url': supplied['definition_url'] = 'https://www.gov.uk/other'
    elif fault == 'reviewed_text': supplied['reviewed_text'] = {}
    else: assessment = PublicationRightsAssessment.create(
        decision='PERMITTED', permitted_use='PUBLICATION_EVIDENCE', policy_digest=govuk_rights.POLICY_DIGEST,
        evidence_digest='sha256:'+'0'*64,
    )
    objects = SimpleNamespace(admit=lambda *a, **kw: pytest.fail('invalid semantic input must not persist'))
    with pytest.raises(ValueError, match='GOV.UK semantic rights binding differs'):
        rights.retain_rights_snapshot(
            objects=objects, proof=object(), **source, assessment=assessment,
            observed_at='2026-09-08T12:00:00Z', reason='REVIEWED_REUSE_PERMITTED', observations=(),
            govuk_semantic_evidence=supplied,
        )


def test_licensing_network_batches_are_four_bounded_fresh_and_owner_fenced(monkeypatch):
    import threading
    from contextlib import contextmanager
    owner = threading.get_ident()
    first_wave = threading.Event()
    lock = threading.Lock()
    state = {'active': 0, 'maximum': 0, 'fenced': False}
    calls, fences, stops = [], [], []
    @contextmanager
    def fence():
        assert threading.get_ident() == owner
        fences.append('enter');state['fenced'] = True
        try: yield
        finally:
            assert state['active'] == 0
            state['fenced'] = False
    def fetch(url):
        assert threading.get_ident() != owner and state['fenced']
        with lock:
            calls.append(url);state['active'] += 1
            state['maximum'] = max(state['maximum'], state['active'])
            initial_wave = len(calls) <= 4
            if state['active'] == 4:first_wave.set()
        try:
            if initial_wave:assert first_wave.wait(2)
            return url.encode()
        finally:
            with lock:state['active'] -= 1
    for _ in range(2):
        observed = rights.fetch_licensing_observations(stop_check=lambda: stops.append('check'),
            stop_fence=fence, govuk_fetch=fetch, portfolio_fetch=fetch)
        urls = [rights.REUSE_URL, rights.LICENCE_URL, *(url for terms in rights.TERMS.values() for url,_ in terms)]
        assert list(observed) == urls
        assert all(observed[url] == url.encode() for url in urls)
    assert state['maximum'] == 4
    assert len(calls) == 18 and len(fences) == len(stops) == 6


@pytest.mark.parametrize('failure', ('source', 'stop'))
def test_licensing_network_failure_settles_before_fence_exit(monkeypatch, failure):
    from contextlib import contextmanager
    from threading import Lock
    active, calls, checks = [0], [], [0]
    lock = Lock()
    @contextmanager
    def fence():
        try:yield
        finally:assert active[0] == 0
    def check():
        checks[0] += 1
        if failure == 'stop' and checks[0] == 2:raise VetoError('owner stop')
    def fetch(url):
        with lock:active[0] += 1;calls.append(url)
        try:
            if url == rights.REUSE_URL:raise OSError('unavailable')
            return url.encode()
        finally:
            with lock:active[0] -= 1
    if failure == 'stop':
        with pytest.raises(VetoError):
            rights.fetch_licensing_observations(stop_check=check, stop_fence=fence,
                govuk_fetch=fetch, portfolio_fetch=fetch)
        assert len(calls) == 4
    else:
        observed = rights.fetch_licensing_observations(stop_check=check, stop_fence=fence,
            govuk_fetch=fetch, portfolio_fetch=fetch)
        assert isinstance(observed[rights.REUSE_URL], OSError)
        assert len(observed) == 9 and len(calls) == 9


def test_licensing_worker_veto_propagates_after_current_batch_settles():
    from contextlib import contextmanager
    from threading import Lock
    active,calls=[0],[]
    lock=Lock()
    failure=VetoError('owner stop')
    @contextmanager
    def fence():
        try:yield
        finally:assert active[0]==0
    def fetch(url):
        with lock:active[0]+=1;calls.append(url)
        try:
            if url==rights.REUSE_URL:raise failure
            return url.encode()
        finally:
            with lock:active[0]-=1
    with pytest.raises(VetoError) as raised:
        rights.fetch_licensing_observations(stop_check=lambda:None,stop_fence=fence,
            govuk_fetch=fetch,portfolio_fetch=fetch)
    assert raised.value is failure and len(calls)==4
