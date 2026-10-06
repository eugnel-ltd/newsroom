"""Fresh portfolio observations share envelopes, never source permission."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
import json
import sqlite3

import pytest

from newsroom.authority import (
    AuthenticationError, HydrationRequest, ObjectAdmissionDenied,
    ObjectAdmissionId, ObjectAdmissionRequest, ObjectIntegrityError,
)
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane import native_source_rights as rights
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.veto import VetoError
from newsroom.increment9.proving import SOURCE_URLS
from newsroom.tests.test_native_runtime import _args
from newsroom.tests.test_native_source_intake import _source_read_audit_counts


def _terms(monkeypatch):
    bodies, terms = {}, {}
    for source, entries in rights.TERMS.items():
        terms[source] = []
        for ordinal, (url, _) in enumerate(entries):
            text = f"Current fixture terms {source} {ordinal}"
            raw = (f'<div class="inner_page_content_container">{text}</div>'
                   if source == "HK-04" else f"<main>{text}</main>").encode()
            bodies[url] = raw
            terms[source].append((url, rights.terms_text_digest(source, raw)))
        terms[source] = tuple(terms[source])
    monkeypatch.setattr(rights, "TERMS", terms)
    return bodies


def _inputs(portfolio, at):
    result = {}
    for source, url in SOURCE_URLS.items():
        evidence = portfolio.evidence.get(source)
        result[source] = dict(
            definition_url=url,
            assessment=portfolio.for_source(source_id=source, definition_url=url),
            observed_at=at if evidence is None else evidence.observed_at,
            reason=portfolio.reason_for(source),
            observations=() if evidence is None else evidence.observations,
        )
    return result


def _refresh(runtime, bodies, at, calls):
    def fetch(url):
        calls.append(url)
        return bodies[url]
    evidence = rights.observe_portfolio_terms(
        objects=runtime.authority.objects, proof=runtime.proof,
        stop_check=lambda: None, stop_fence=nullcontext, fetch=fetch,
        clock=lambda: datetime.fromisoformat(at.replace("Z", "+00:00")),
    )
    portfolio = rights.NativePortfolioRights(None, evidence)
    inputs = _inputs(portfolio, at)
    snapshots = rights.retain_rights_snapshot_bundle(
        objects=runtime.authority.objects, proof=runtime.proof,
        snapshots=inputs, stop_check=lambda: None,
    )
    return portfolio, inputs, snapshots


def _read(runtime, snapshot, source):
    return rights.read_rights_observation(
        objects=runtime.authority.objects, proof=runtime.proof,
        reference=snapshot, source_id=source, definition_url=SOURCE_URLS[source],
    )


def test_actual_refresh_bundles_ten_fresh_sources_and_reopens(tmp_path, monkeypatch, record_property):
    bodies, calls = _terms(monkeypatch), []
    monkeypatch.setattr(native_composition, "fetch_licensing_observations", lambda **_: bodies)
    args = _args(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        _, first_inputs, first = _refresh(runtime, bodies, "2026-09-08T15:00:00Z", calls)
        def inventory():
            with sqlite3.connect(args["authority_path"]) as connection:
                return (
                    connection.execute("SELECT count(*) FROM object_admissions").fetchone()[0],
                    connection.execute("SELECT count(*) FROM authority_commands WHERE command_type='object.admission.activate'").fetchone()[0],
                )
        first_inventory = inventory()
        assert first_inventory == (18, 18)  # Seven raw terms + ten assessments + one envelope.
        assert set(first) == set(SOURCE_URLS)
        assert len({ref.observation_admission_id for ref in first.values()}) == 1
        before = _source_read_audit_counts(args["authority_path"])
        _, latest_inputs, latest = _refresh(runtime, bodies, "2026-09-08T15:05:00Z", calls)
        assert inventory() == (19, 19)
        assert len(calls) == 14  # Every actual endpoint fetched again.
        assert _source_read_audit_counts(args["authority_path"])[0] == before[0] + 1
        assert len({ref.observation_admission_id for ref in latest.values()}) == 1
        for source in SOURCE_URLS:
            assert first[source].assessment_admission_id == latest[source].assessment_admission_id
            assert first[source].observation_admission_id != latest[source].observation_admission_id
            assert first[source].observation_source_id == source
            assert first[source].observation_member_digest != latest[source].observation_member_digest
            for inputs, snapshots in ((first_inputs, first), (latest_inputs, latest)):
                observed = _read(runtime, snapshots[source], source)
                assert observed["source_id"] == source
                assert observed["observed_at"] == inputs[source]["observed_at"]
                assert observed["reason"] == inputs[source]["reason"]
                assert observed["observations"] == json.loads(canonical_json_bytes(inputs[source]["observations"]))
                assert digest_bytes(canonical_json_bytes(observed)) == snapshots[source].observation_member_digest
        raw = runtime.authority.objects.rehydrate(HydrationRequest(
            ObjectAdmissionId.parse(latest["HK-02"].observation_admission_id), "evidence.source",
        ), proof=runtime.proof).data
        member_bytes = sum(len(canonical_json_bytes(item["member"])) for item in json.loads(raw)["members"])
        record_property("two_refresh_inventory", json.dumps({
            "first_admissions": first_inventory[0], "first_activation_commands": first_inventory[1],
            "second_admissions": inventory()[0], "second_activation_commands": inventory()[1],
            "observation_envelopes": 2, "legacy_observation_envelopes": 20,
            "bundle_bytes": len(raw), "individual_member_bytes": member_bytes,
            "bundle_wire_overhead_bytes": len(raw) - member_bytes,
        }))
    with open_native_runtime(**args) as reopened:
        before = _source_read_audit_counts(args["authority_path"])
        replay = rights.retain_rights_snapshot_bundle(
            objects=reopened.authority.objects, proof=reopened.proof,
            snapshots=latest_inputs, stop_check=lambda: None,
        )
        assert replay == latest
        assert _source_read_audit_counts(args["authority_path"]) == before
        assert _read(reopened, first["HK-02"], "HK-02")["observed_at"] == first_inputs["HK-02"]["observed_at"]


@pytest.mark.parametrize("fault", (
    "missing", "duplicate", "unknown", "swapped", "member_digest", "mutated_body",
))
def test_bundle_reader_rejects_invalid_member_inventory(tmp_path, monkeypatch, fault):
    bodies = _terms(monkeypatch)
    with open_native_runtime(**_args(tmp_path, monkeypatch)) as runtime:
        _, _, snapshots = _refresh(runtime, bodies, "2026-09-08T15:00:00Z", [])
        reference = snapshots["HK-02"]
        raw = runtime.authority.objects.rehydrate(HydrationRequest(
            ObjectAdmissionId.parse(reference.observation_admission_id), "evidence.source",
        ), proof=runtime.proof).data
        bundle = json.loads(raw)
        members = bundle["members"]
        selected = next(item for item in members if item["source_id"] == "HK-02")
        if fault == "missing":
            members.pop()
        elif fault == "duplicate":
            members.append(deepcopy(members[0]))
        elif fault == "unknown":
            members[0]["source_id"] = "UNKNOWN"
        elif fault == "swapped":
            selected["member"]["source_id"] = "UK-10"
            selected["member_digest"] = digest_bytes(canonical_json_bytes(selected["member"]))
        elif fault == "member_digest":
            selected["member_digest"] = "sha256:" + "f" * 64
        else:
            selected["member"]["reason"] = "SOURCE_TERMS_CHANGED"
        data = canonical_json_bytes(bundle)
        admission = runtime.authority.objects.admit(ObjectAdmissionRequest(
            "evidence.source", f"malformed-bundle:{fault}",
        ), data, proof=runtime.proof).admission
        malformed = replace(reference, observation_admission_id=str(admission.admission_id),
                            observation_blob_digest=digest_bytes(data))
        with pytest.raises(ValueError, match="rights observation"):
            _read(runtime, malformed, "HK-02")


@pytest.mark.parametrize("fault", ("source", "member", "missing_selector", "blob", "credential", "revocation", "cas"))
def test_bundle_reference_rechecks_binding_and_current_authority(tmp_path, monkeypatch, fault):
    bodies = _terms(monkeypatch)
    with open_native_runtime(**_args(tmp_path, monkeypatch)) as runtime:
        _, _, snapshots = _refresh(runtime, bodies, "2026-09-08T15:00:00Z", [])
        reference, proof = snapshots["HK-02"], runtime.proof
        error = ValueError
        if fault == "source":
            reference = replace(reference, observation_source_id="UK-10")
        elif fault == "member":
            reference = replace(reference, observation_member_digest=snapshots["UK-10"].observation_member_digest)
        elif fault == "missing_selector":
            reference = replace(reference, observation_source_id=None, observation_member_digest=None)
        elif fault == "blob":
            reference = replace(reference, observation_blob_digest="sha256:" + "f" * 64)
        elif fault == "credential":
            proof = replace(proof, credential="not-current")
            error = AuthenticationError
        elif fault == "revocation":
            runtime.authority.objects.revoke(ObjectAdmissionId.parse(reference.observation_admission_id),
                reason_code="REVOKED", idempotency_key="revoke-bundle", proof=proof)
            error = ObjectAdmissionDenied
        else:
            digest = reference.observation_blob_digest.split(":", 1)[1]
            path = tmp_path / "objects" / "objects" / digest[:2] / digest
            raw = path.read_bytes()
            path.chmod(0o600)
            path.write_bytes(b"[" + raw[1:])
            path.chmod(0o400)
            error = ObjectIntegrityError
        with pytest.raises(error):
            rights.read_rights_observation(objects=runtime.authority.objects, proof=proof,
                reference=reference, source_id="HK-02", definition_url=SOURCE_URLS["HK-02"])


def test_bundle_stop_and_changed_terms_remain_source_local(tmp_path, monkeypatch):
    bodies, calls = _terms(monkeypatch), []
    args = _args(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        portfolio, inputs, snapshots = _refresh(runtime, bodies, "2026-09-08T15:00:00Z", calls)
        before = _source_read_audit_counts(args["authority_path"])
        def stopped():
            raise VetoError("STOPPED")
        with pytest.raises(VetoError):
            rights.retain_rights_snapshot_bundle(objects=runtime.authority.objects,
                proof=runtime.proof, snapshots=inputs, stop_check=stopped)
        assert _source_read_audit_counts(args["authority_path"]) == before
        url = rights.TERMS["HK-02"][0][0]
        bodies[url] = b"<main>Changed source licence</main>"
        changed, _, latest = _refresh(runtime, bodies, "2026-09-08T15:05:00Z", calls)
        assert portfolio.for_source(source_id="HK-02", definition_url=SOURCE_URLS["HK-02"]).decision == "PERMITTED"
        assert changed.for_source(source_id="HK-02", definition_url=SOURCE_URLS["HK-02"]).decision == "HOLD"
        assert changed.for_source(source_id="UK-10", definition_url=SOURCE_URLS["UK-10"]).decision == "PERMITTED"
        assert snapshots["HK-02"].assessment_admission_id != latest["HK-02"].assessment_admission_id
        assert _read(runtime, latest["HK-02"], "HK-02")["reason"] == "SOURCE_TERMS_CHANGED"


@pytest.mark.parametrize("fault", ("missing", "unknown", "definition_url"))
def test_bundle_writer_rejects_bad_inventory_before_retention(tmp_path, monkeypatch, fault):
    bodies = _terms(monkeypatch)
    args = _args(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        _, inputs, _ = _refresh(runtime, bodies, "2026-09-08T15:00:00Z", [])
        if fault == "missing":
            inputs.pop("RAD-02")
        elif fault == "unknown":
            inputs["UNKNOWN"] = inputs.pop("RAD-02")
        else:
            inputs["HK-02"]["definition_url"] = SOURCE_URLS["UK-10"]
        before = _source_read_audit_counts(args["authority_path"])
        with pytest.raises(ValueError, match="rights observation"):
            rights.retain_rights_snapshot_bundle(objects=runtime.authority.objects,
                proof=runtime.proof, snapshots=inputs, stop_check=lambda: None)
        assert _source_read_audit_counts(args["authority_path"]) == before


def test_legacy_individual_observation_reads_without_bundle_selector(tmp_path, monkeypatch):
    bodies = _terms(monkeypatch)
    with open_native_runtime(**_args(tmp_path, monkeypatch)) as runtime:
        _, inputs, _ = _refresh(runtime, bodies, "2026-09-08T15:00:00Z", [])
        snapshot = rights.retain_rights_snapshot(objects=runtime.authority.objects,
            proof=runtime.proof, source_id="HK-02", **inputs["HK-02"])
        assert snapshot.observation_source_id is None
        assert _read(runtime, snapshot, "HK-02")["source_id"] == "HK-02"
        with pytest.raises(ValueError, match="rights observation"):
            _read(runtime, snapshot, "UK-10")


def test_composed_current_rights_checks_fresh_member_and_definition_version(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from newsroom.control_plane import native_composition
    from newsroom.control_plane.native_evidence import NativeEvidenceHold
    from newsroom.tests.test_native_composition import _arguments, _RetrievalProjection, _Reader
    from newsroom.tests.projection_b2_helpers import MemoryNeo4jAdapter

    bodies, calls = _terms(monkeypatch), []
    now = [datetime(2026, 9, 8, 14, tzinfo=UTC)]
    def no_govuk(**_):
        raise NativeEvidenceHold("GOVUK_LICENCE_REVIEW_HOLD", "UK-GOVUK")
    def observed_terms(**arguments):
        def fetch(url):
            calls.append(url)
            return bodies[url]
        arguments["fetch"] = fetch
        return rights.observe_portfolio_terms(**arguments)
    monkeypatch.setattr(native_composition, "retain_current_govuk_licence", no_govuk)
    monkeypatch.setattr(native_composition, "observe_portfolio_terms", observed_terms)
    monkeypatch.setattr("newsroom.authority._graphiti_increment4_system._open_structural_graph_adapter",
                        lambda _: MemoryNeo4jAdapter())
    monkeypatch.setattr(native_composition, "open_native_retrieval_neo4j_resources", lambda **_: SimpleNamespace(
        projector=_RetrievalProjection(), fulltext=_Reader(), close=lambda: None,
    ))
    arguments = {**_arguments(tmp_path), "clock": lambda: now[0],
                 "stop_check": lambda: None, "stop_fence": nullcontext}
    with native_composition.open_native_pipeline(**arguments) as pipeline:
        pipeline._intake._fetch = lambda _: (200, b"{}")
        ready = pipeline._intake._poll_one("HK-02")
        assert ready.status == "READY" and ready.units
        unit = ready.units[0]
        original = pipeline._graphiti._rights(unit)
        assert original["observation_source_id"] == "HK-02"
        assert original["observation_member_digest"]
        old_snapshot = pipeline._intake._licence.snapshot_for("HK-02")
        pipeline._refresh_rights()  # Opening complete snapshot is consumed once.
        now[0] = datetime(2026, 9, 8, 14, 5, tzinfo=UTC)
        pipeline._refresh_rights()
        latest = pipeline._graphiti._rights(unit)
        assert len(calls) == 14
        assert latest["observation_admission_id"] != original["observation_admission_id"]
        assert latest["observation_member_digest"] != original["observation_member_digest"]
        assert latest["assessment_admission_id"] == original["assessment_admission_id"]
        portfolio = pipeline._intake._licence
        current_snapshot = portfolio._snapshots["HK-02"]
        portfolio._snapshots["HK-02"] = old_snapshot
        with pytest.raises(ValueError, match="current fetched snapshot"):
            pipeline._graphiti._rights(unit)
        portfolio._snapshots["HK-02"] = current_snapshot
        stale = replace(unit, authority=replace(unit.authority,
                        definition_version_id="00000000-0000-4000-8000-000000000999"))
        assert pipeline._graphiti._rights(stale) is None
        pipeline._intake._objects.revoke(ObjectAdmissionId.parse(latest["assessment_admission_id"]),
            reason_code="REVOKED", idempotency_key="revoke-current-assessment",
            proof=pipeline._intake._proof)
        with pytest.raises(ObjectAdmissionDenied):
            pipeline._graphiti._rights(unit)


@pytest.mark.parametrize("bundle_members", ((True, True), (False, True)))
def test_provider_free_replay_keeps_fresh_bundle_selector(tmp_path, bundle_members):
    from newsroom.tests.test_graphiti_internal_requests import _service_fixture, T0
    from newsroom.tests.test_native_retry_evidence import _recovered_ambiguous_pattern
    from newsroom.tests.test_native_graphiti import _native
    from datetime import timedelta

    service, _, policy, shape = _service_fixture(tmp_path)
    unit = _native("bundle-provider-free-replay")
    _, _, _, rejected_digest = _recovered_ambiguous_pattern(
        service, policy, shape, unit, bundle_members=bundle_members,
    )
    digest = service.native_recovered_ambiguous_usage_evidence_digest(
        ingest_id=unit.ingest_id, authoritative_attempt_number=3,
        skipped_attempt_number=4, skipped_receipt_digest=rejected_digest,
        skipped_recorded_at=T0 + timedelta(seconds=3),
    )
    assert digest.startswith("sha256:")
    with sqlite3.connect(service.path) as connection:
        row = connection.execute("SELECT receipt_json FROM unpublished_graphiti_attempt_receipts "
                                 "WHERE ingest_id=? AND attempt_number=2", (unit.ingest_id,)).fetchone()
    replay = json.loads(row[0])
    assert replay["dispatch_rights"]["observation_source_id"] == unit.source_id
    assert replay["dispatch_rights"]["observation_member_digest"]


def test_gate_keeps_exact_member_reference_without_unchanged_regrowth(tmp_path):
    from newsroom.control_plane.native_discovery import NativeDiscovery
    from newsroom.discovery import ReasonReference
    from newsroom.tests.test_native_discovery import _seed, _current_rights, NOW, LATER
    from newsroom.tests.test_graphiti_operational_readiness import _unit
    from newsroom.tests.discovery_3d_authority_helpers import open_discovery_system
    from newsroom.tests.check_3c_authority_helpers import proof

    unit = _unit()
    packet = {**_current_rights(), "source_id": unit.source_id,
              "source_url": unit.source_definition_url,
              "observation_source_id": unit.source_id, "observation_member_digest": "sha256:" + "a" * 64}
    with sqlite3.connect(":memory:") as proving, open_discovery_system(tmp_path / "gate.sqlite3", clock=lambda: NOW) as system:
        _seed(system, unit)
        controller = NativeDiscovery(sources=system.sources, checks=system.checks,
            discovery=system.discovery, proving=proving, rights_for=lambda *_: packet)
        delivered = controller.deliver(unit, now=NOW, proof=proof())
        initial = controller.admit_lead(delivered, now=NOW, proof=proof())
        assert initial.current_gate.request.supporting_reasons[0].references[-1] == ReasonReference(
            "RIGHTS_OBSERVATION_MEMBER", unit.source_id, packet["observation_member_digest"])
        packet["observation_member_digest"] = "sha256:" + "b" * 64
        repeated = controller.admit_lead(delivered, now=LATER, proof=proof())
        assert repeated.current_gate == initial.current_gate
        assert repeated.current_gate.request.supporting_reasons[0].references[-1].digest == "sha256:" + "a" * 64
        for invalid in (
            {**packet, "observation_source_id": "HK-02"},
            {**packet, "source_url": "https://example.test/wrong-source"},
            {key: value for key, value in packet.items() if key != "observation_member_digest"},
            {**packet, "observation_member_digest": "not-a-digest"},
        ):
            packet.clear()
            packet.update(invalid)
            with pytest.raises(ValueError):
                controller.admit_lead(delivered, now=NOW, proof=proof())


@pytest.mark.parametrize("fault", ("missing", "swapped", "invalid_digest"))
def test_provider_free_replay_rejects_invalid_bundle_selector(tmp_path, fault):
    from newsroom.control_plane import model_usage as usage
    from newsroom.tests.test_graphiti_internal_requests import _service_fixture, T0
    from newsroom.tests.test_native_retry_evidence import _recovered_ambiguous_pattern
    from newsroom.tests.test_native_graphiti import _native
    from datetime import timedelta

    service, _, policy, shape = _service_fixture(tmp_path)
    unit = _native("invalid-bundle-provider-free-replay")
    _, _, _, rejected_digest = _recovered_ambiguous_pattern(
        service, policy, shape, unit, bundle_members=(True, True),
    )
    with sqlite3.connect(service.path) as connection:
        receipt = json.loads(connection.execute("SELECT receipt_json FROM unpublished_graphiti_attempt_receipts "
            "WHERE ingest_id=? AND attempt_number=2", (unit.ingest_id,)).fetchone()[0])
        packet = receipt["dispatch_rights"]
        if fault == "missing":
            packet.pop("observation_member_digest")
        elif fault == "swapped":
            packet["observation_source_id"] = "HK-02"
        else:
            packet["observation_member_digest"] = "not-a-digest"
        unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
        digest = digest_bytes(canonical_json_bytes(unsigned))
        receipt["receipt_digest"] = digest
        connection.execute("UPDATE unpublished_graphiti_attempt_receipts SET receipt_digest=?,receipt_json=? "
            "WHERE ingest_id=? AND attempt_number=2", (digest, json.dumps(receipt, sort_keys=True), unit.ingest_id))
    with pytest.raises(usage.ModelUsageIntegrityError):
        service.native_recovered_ambiguous_usage_evidence_digest(
            ingest_id=unit.ingest_id, authoritative_attempt_number=3,
            skipped_attempt_number=4, skipped_receipt_digest=rejected_digest,
            skipped_recorded_at=T0 + timedelta(seconds=3),
        )
