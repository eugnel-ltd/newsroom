import sqlite3
from dataclasses import replace

import pytest

from newsroom.authority import AggregateId, ObjectAdmissionRequest
from newsroom.authority import UtcTimestamp
from newsroom.authority.canonical import canonical_json_bytes
from newsroom.increment10.editorial import EditorialHold, StoryVersionRequest
from newsroom.control_plane.native_story_writer import write_native_story
from newsroom.tests.test_increment10_editorial import (
    _open_editorial_system, _candidate, _receive, _evidence_facade,
    _ready_package, _record_decision, _decision, _native,
)
from newsroom.increment10.ingress import open_evidence_intake_ingress
from newsroom.tests.authority_helpers import proof


def _fixture(tmp_path):
    candidate_connection, candidate_port, version = _candidate(tmp_path)
    ingress = open_evidence_intake_ingress(tmp_path / "intake.sqlite3")
    acknowledgement = _receive(ingress, candidate_connection, candidate_port, version, request_id="narrative-input")
    system, registries = _open_editorial_system(tmp_path / "authority.sqlite3")
    evidence = _evidence_facade(system, ingress, registries)
    passage, package, records = _ready_package(version)
    source = system.objects.admit(ObjectAdmissionRequest("evidence.source", "source-1"), passage.encode(), proof=proof()).admission
    ids = tuple(system.objects.admit(ObjectAdmissionRequest("evidence.record", f"record-{i}"),
        canonical_json_bytes(record), proof=proof()).admission.admission_id for i, record in enumerate(records))
    candidate_connection.execute("BEGIN IMMEDIATE")
    retained = evidence.retain(package, receipt_id=acknowledgement.receipt_id, candidate_port=candidate_port,
        source_admission_ids=(source.admission_id,), record_admission_ids=ids, proof=proof())
    reference = _record_decision(system, _decision(retained, source.admission_id))
    return system, candidate_connection, candidate_port, evidence, registries, retained, reference


def _writer(calls, *, supported=True, headline_body_link=False):
    def write(package, **identities):
        assert identities["candidate_id"] == package.candidate_id
        headline, substantive = package.governed_claims
        draft = {"title": "限期更改獲官方確認", "body": "官方確認限期已經更改，相關限期安排亦已更新。", "format": "BRIEF",
            "evidence_links": [
                {"governed_claim_id": headline.claim_id, "rendered_assertion": "限期更改獲官方確認"},
                {"governed_claim_id": headline.claim_id, "rendered_assertion": "官方確認限期已經更改"},
                {"governed_claim_id": substantive.claim_id, "rendered_assertion": "相關限期安排亦已更新"}]}
        if headline_body_link:
            draft["evidence_links"][0]["rendered_assertion"] = "官方確認限期已經更改"
        def generate(request):
            calls.append("draft")
            return draft
        def review(request):
            calls.append("review")
            return {"source_package_digest": package.digest, "draft_digest": request["draft_digest"],
                "verdict": "PASS" if supported else "HOLD",
                "covered_claim_ids": [headline.claim_id, substantive.claim_id],
                "sentence_support": [
                    {"sentence_index": 0, "claim_ids": [headline.claim_id], "verdict": "SUPPORTED"},
                    {"sentence_index": 1, "claim_ids": [headline.claim_id, substantive.claim_id], "verdict": "SUPPORTED"}],
                "factual_checks": {key: "PASS" for key in ("numbers", "entities", "modality", "quotations")}}
        return write_native_story(package, generate=generate, review=review)
    return write


def test_natural_story_admits_replays_and_reads_without_model_redispatch(tmp_path):
    system, candidates, port, evidence, registries, retained, reference = _fixture(tmp_path)
    native, calls = _native(system, evidence, registries), []
    native._story_writer = _writer(calls)
    request = StoryVersionRequest(AggregateId.new(), 0, "narrative-story")
    arguments = dict(package_admission_id=retained.package_admission_id,
                     decision_reference=reference, candidate_port=port, proof=proof())
    try:
        receipt, story = native.admit_story_version(request, **arguments)
        assert story.copy.title == "限期更改獲官方確認"
        assert story.story_format == "BRIEF" and story.writer_review["verdict"] == "PASS"
        from newsroom.increment10.publication import _render
        article, card = _render(story, ((retained.package.source_ids[0], "https://gov.example.test/update"),))
        assert article.headline == story.copy.title and article.body == story.copy.body
        assert article.renderer_version == "newsroom.source-grounded-brief.v1"
        assert card.kind == "FEED_CARD" and card.body == ""
        replay_receipt, replay = native.admit_story_version(request, **arguments)
        assert replay_receipt == receipt and replay.canonical_bytes() == story.canonical_bytes()
        read = native.read_story_version(receipt, candidate_port=port, proof=proof())
        assert read.canonical_bytes() == story.canonical_bytes()
        assert calls == ["draft", "review"]
    finally:
        candidates.close()
        system.close()


def test_unknown_source_review_has_no_story_effect(tmp_path):
    system, candidates, port, evidence, registries, retained, reference = _fixture(tmp_path)
    native, calls = _native(system, evidence, registries), []
    native._story_writer = _writer(calls, supported=False)
    before = sqlite3.connect(tmp_path / "authority.sqlite3").execute("SELECT count(*) FROM ledger_events WHERE event_type='editorial.story-version.admitted'").fetchone()[0]
    try:
        with pytest.raises(EditorialHold):
            native.admit_story_version(StoryVersionRequest(AggregateId.new(), 0, "rejected-story"),
                package_admission_id=retained.package_admission_id, decision_reference=reference,
                candidate_port=port, proof=proof())
        assert sqlite3.connect(tmp_path / "authority.sqlite3").execute("SELECT count(*) FROM ledger_events WHERE event_type='editorial.story-version.admitted'").fetchone()[0] == before
    finally:
        candidates.close()
        system.close()


def test_model_delay_cannot_admit_an_expired_observed_source(tmp_path):
    system, candidates, port, evidence, registries, retained, reference = _fixture(tmp_path)
    native = _native(system, evidence, registries)
    original = _decision(retained, retained.source_admission_ids[0])
    source = replace(original.currentness[0], currency_family="OBSERVED_STATE",
                     currency_window_seconds=300, version_reference=None,
                     supersession_evidence_digest=None)
    policy = type(original).create(**{name: (source,) if name == "currentness" else getattr(original, name)
        for name in original.__dataclass_fields__ if name != "decision_id"})
    native._clock = lambda: UtcTimestamp.parse("2026-09-08T12:07:00Z")
    def writer(package, **identities):
        identities["require_current"]()
        raise AssertionError("expired source reached drafting")
    native._story_writer = writer
    try:
        with pytest.raises(EditorialHold, match="CURRENCY"):
            native._build_story(StoryVersionRequest(AggregateId.new(), 0, "expiry"),
                                retained, policy, reference)
    finally:
        candidates.close()
        system.close()


_OLD_WRITE_POLICY = (
    "newsroom.write-admission.v9+newsroom.evid-012.v7+"
    "newsroom.evidence-approval.v8+newsroom.evidence-gates.v2+"
    "newsroom.governed-claim.v7+newsroom.governed-input.v10+"
    "newsroom.named-entity.v15+newsroom.cont-originality.v3+"
    "newsroom.zh-hant-hk-shape.v14+newsroom.factual-localisation.v2+"
    "newsroom.qualification-relation.v3"
)


def test_exact_retained_entity_v15_story_decodes_without_changing_bytes():
    from pathlib import Path
    from newsroom.increment10.editorial import StoryVersion
    from newsroom.authority.canonical import digest_bytes
    raw = (Path(__file__).parent / "fixtures/native_publication/retained-story-entity-v15.json").read_bytes()
    assert digest_bytes(raw) == "sha256:7ce68f793f46c326a7be4d39610d086624893f00b0c41de4e8e0c1930fc043a9"
    assert StoryVersion.from_bytes(raw).canonical_bytes() == raw


@pytest.mark.parametrize("field,value", [
    ("decision_id", "sha256:" + "0" * 64),
    ("policy_version", _OLD_WRITE_POLICY.replace("named-entity.v15", "named-entity.v999")),
])
def test_retained_story_policy_compatibility_does_not_accept_corruption(field, value):
    import json
    from pathlib import Path
    from newsroom.increment10.editorial import EditorialError, StoryVersion
    value_record = json.loads((Path(__file__).parent / "fixtures/native_publication/retained-story-entity-v15.json").read_bytes())
    value_record["write_admission"][field] = value
    with pytest.raises(EditorialError, match="Story Version fields differ"):
        StoryVersion.from_bytes(canonical_json_bytes(value_record))


@pytest.mark.parametrize("writer_id", [
    "newsroom.offline-exact-copy.v1", "newsroom.offline-exact-copy.v2",
    "newsroom.offline-exact-copy.v3", "newsroom.native-story-writer.v1",
])
def test_retained_policy_story_replays_and_reads_without_upgrading_identity(tmp_path, monkeypatch, writer_id):
    from newsroom.control_plane import admission
    system, candidates, port, evidence, registries, retained, reference = _fixture(tmp_path)
    native, calls = _native(system, evidence, registries), []
    native._story_writer = _writer(calls) if writer_id.endswith("writer.v1") else None
    request = StoryVersionRequest(AggregateId.new(), 0, "retained-policy")
    arguments = dict(package_admission_id=retained.package_admission_id,
                     decision_reference=reference, candidate_port=port, proof=proof())
    try:
        # Create immutable history under the formerly current policy, then upgrade.
        with monkeypatch.context() as old:
            old.setattr(admission, "WRITE_ADMISSION_POLICY_VERSION", _OLD_WRITE_POLICY)
            if writer_id != "newsroom.native-story-writer.v1":
                original_build = native._build_story
                def legacy_build(*args, **kwargs):
                    kwargs.setdefault("writer_id", writer_id)
                    return original_build(*args, **kwargs)
                old.setattr(native, "_build_story", legacy_build)
            receipt, story = native.admit_story_version(request, **arguments)
        before_calls = list(calls)
        assert native.read_story_version(receipt, candidate_port=port, proof=proof()).canonical_bytes() == story.canonical_bytes()
        replay_receipt, replay_story = native.admit_story_version(request, **arguments)
        assert replay_receipt == receipt and replay_story.canonical_bytes() == story.canonical_bytes()
        assert replay_story.write_admission.policy_version == _OLD_WRITE_POLICY
        assert calls == before_calls
        fresh = native.admit_story_version(StoryVersionRequest(AggregateId.new(), 0, "fresh-policy"), **arguments)[1]
        assert fresh.write_admission.policy_version == admission.WRITE_ADMISSION_POLICY_VERSION
        assert fresh.write_admission.decision_id != story.write_admission.decision_id
    finally:
        candidates.close()
        system.close()


def test_retained_admission_still_requires_current_readiness_and_exact_fields(tmp_path):
    from newsroom.control_plane.admission import WriteAdmissionDecision, _decision_id
    from newsroom.increment10.editorial import EditorialError
    system, candidates, port, evidence, registries, retained, reference = _fixture(tmp_path)
    native = _native(system, evidence, registries)
    request = StoryVersionRequest(AggregateId.new(), 0, "retained-fields")
    policy = native._read_policy_decision(reference, retained=retained, proof=proof())
    try:
        original = native._build_story(request, retained, policy, reference)
        record = original.write_admission.as_record()
        record["selection_rationale"] = "A different retained editorial selection"
        values = {key: value for key, value in record.items() if key not in {"decision_id", "decided_at"}}
        record["decision_id"] = _decision_id(**values)
        changed = WriteAdmissionDecision.from_record(record)
        with pytest.raises(EditorialError, match="object admission differs: retained write-admission"):
            native._build_story(request, retained, policy, reference, retained_admission=changed)
        held_policy = type(policy).create(**{name: (
            (("CLAIM_TRACEABILITY", "HOLD"), ("EVIDENCE_SUFFICIENCY", "PASS"),
             ("SOURCE_AUTHORITY", "PASS")) if name == "evidence_gate_results"
            else getattr(policy, name)) for name in policy.__dataclass_fields__ if name != "decision_id"})
        with pytest.raises(EditorialHold):
            native._build_story(request, retained, held_policy, reference,
                                retained_admission=original.write_admission)
    finally:
        candidates.close()
        system.close()
