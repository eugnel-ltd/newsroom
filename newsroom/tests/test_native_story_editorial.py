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


def _writer(calls, *, supported=True):
    def write(package, **identities):
        assert identities["candidate_id"] == package.candidate_id
        headline, substantive = package.governed_claims
        draft = {"title": "限期更改獲官方確認", "body": "官方確認限期已經更改，相關限期安排亦已更新。", "format": "BRIEF",
            "evidence_links": [
                {"governed_claim_id": headline.claim_id, "rendered_assertion": "限期更改獲官方確認"},
                {"governed_claim_id": headline.claim_id, "rendered_assertion": "官方確認限期已經更改"},
                {"governed_claim_id": substantive.claim_id, "rendered_assertion": "相關限期安排亦已更新"}]}
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
