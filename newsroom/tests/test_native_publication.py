from __future__ import annotations

import json
from datetime import timedelta
import sqlite3
from pathlib import Path

import pytest

from newsroom.authority import ObjectAdmissionRequest, UtcTimestamp
from newsroom.authority.canonical import canonical_json_bytes
from newsroom.control_plane.native_publication import (
    NativePublicationBindings,
    NativePublicationController,
    NativePublicationError,
)
from newsroom.increment10.editorial import EditorialHold
from newsroom.increment10.ingress import open_evidence_intake_ingress
from newsroom.increment10.private_serving import open_private_serving_read_port
from newsroom.tests.authority_helpers import proof
from newsroom.tests.test_increment10_editorial import (
    _decision,
    _evidence_facade,
    _ready_package,
)
from newsroom.tests.test_increment10_ingress import _candidate, _receive
from newsroom.tests.test_increment10_private_serving import _open


def _bindings(tmp_path: Path, registries, hydration, definitions, commands):
    return NativePublicationBindings(
        target_path=tmp_path / "private-serving.sqlite3",
        reader_principal_id="principal.alpha",
        authority_domain="newsroom.authority",
        editorial_controller_principal_id="principal.alpha",
        story_principal_id="principal.alpha",
        publication_controller_principal_id="principal.alpha",
        serving_adapter_principal_id="principal.alpha",
        editorial_policy_bundle_digest="sha256:" + "a" * 64,
        editorial_decision_hydration_policy_digest=(
            registries[3][0].contract_digest
        ),
        editorial_story_hydration_policy_digest=registries[3][1].contract_digest,
        editorial_decision_admission_definition_digest=registries[4][0].digest,
        editorial_decision_command_definition_digest=registries[5].digest,
        editorial_story_command_definition_digest=registries[6].digest,
        editorial_story_admission_definition_digest=registries[4][1].digest,
        publication_authorisation_policy_digest="sha256:" + "8" * 64,
        target_id="newsroom-app-serving-launch",
        target_policy_digest="sha256:" + "9" * 64,
        publication_surface_hydration_policy_digest=(
            registries[7][0].contract_digest
        ),
        publication_transaction_hydration_policy_digest=(
            registries[7][1].contract_digest
        ),
        publication_surface_admission_definition_digest=registries[8][0].digest,
        publication_transaction_admission_definition_digest=registries[8][1].digest,
        publication_command_definition_digest=registries[9].digest,
        target_context_digest="sha256:" + "a" * 64,
        serving_attempt_hydration_policy_digest=hydration[0].contract_digest,
        serving_evidence_hydration_policy_digest=hydration[1].contract_digest,
        serving_attempt_admission_definition_digest=definitions[0].digest,
        serving_evidence_admission_definition_digest=definitions[1].digest,
        serving_attempt_command_definition_digest=commands[0].digest,
        serving_evidence_command_definition_digest=commands[1].digest,
    )


@pytest.mark.parametrize("failure_boundary", (None, "before_apply", "after_apply", "after_record"))
@pytest.mark.parametrize("copy_correction", (False, True))
def test_native_publication_replays_to_exact_ack_only_rows(tmp_path: Path, monkeypatch, copy_correction, failure_boundary) -> None:
    candidate_connection, candidate_port, version = _candidate(tmp_path)
    ingress = open_evidence_intake_ingress(tmp_path / "intake.sqlite3")
    acknowledgement = _receive(
        ingress,
        candidate_connection,
        candidate_port,
        version,
        request_id="request-1",
    )
    instant = [UtcTimestamp.parse("2026-07-16T12:00:00Z").value]

    def effect_clock():
        sampled = UtcTimestamp(instant[0])
        instant[0] += timedelta(minutes=1)
        return sampled

    system, registries, hydration, definitions, commands = _open(
        tmp_path / "objects.sqlite3", clock=lambda: UtcTimestamp(instant[0]),
    )
    evidence_packages = _evidence_facade(system, ingress, registries)
    passage, package, records = _ready_package(version)
    source = system.objects.admit(
        ObjectAdmissionRequest("evidence.source", "source-1"),
        passage.encode(),
        proof=proof(),
    ).admission
    record_ids = tuple(
        system.objects.admit(
            ObjectAdmissionRequest("evidence.record", f"record-{index}"),
            canonical_json_bytes(record),
            proof=proof(),
        ).admission.admission_id
        for index, record in enumerate(records)
    )
    candidate_connection.execute("BEGIN IMMEDIATE")
    retained = evidence_packages.retain(
        package,
        receipt_id=acknowledgement.receipt_id,
        candidate_port=candidate_port,
        source_admission_ids=(source.admission_id,),
        record_admission_ids=record_ids,
        proof=proof(),
    )
    bindings = _bindings(tmp_path, registries, hydration, definitions, commands)
    controller = NativePublicationController(
        objects=system.objects,
        commands=system.commands,
        events=system.events,
        candidate_port=candidate_port,
        evidence_packages=evidence_packages,
        bindings=bindings, clock=effect_clock,
    )
    original_builder = controller._editorial._build_story
    if copy_correction:
        def old_copy(*args, **kwargs):
            kwargs["writer_id"] = "newsroom.offline-exact-copy.v2"
            return original_builder(*args, **kwargs)
        monkeypatch.setattr(controller._editorial, "_build_story", old_copy)
    request = {
        "expected_story_version": 0,
        "expected_publication_version": 0,
        "expected_delivery_evidence_version": 0,
        "proof": proof(),
    }
    with pytest.raises(EditorialHold):
        controller.advance(
            retained.package_admission_id,
            _decision(retained, source.admission_id, result="HOLD"),
            **request,
        )
    target = sqlite3.connect(bindings.target_path)
    count = target.execute(
        "SELECT COUNT(*) FROM private_serving_payloads"
    ).fetchone()[0]
    assert count == 0
    target.close()

    decision = _decision(retained, source.admission_id)
    if failure_boundary is not None:
        method = "record" if failure_boundary == "after_record" else "apply"
        original = getattr(controller._delivery, method)

        def interrupted(*args, **kwargs):
            if failure_boundary != "before_apply":
                original(*args, **kwargs)
            raise RuntimeError("simulated process interruption")

        monkeypatch.setattr(controller._delivery, method, interrupted)
        with pytest.raises(RuntimeError, match="simulated process interruption"):
            controller.advance(retained.package_admission_id, decision, **request)
        monkeypatch.setattr(controller._delivery, method, original)
        instant[0] = UtcTimestamp.parse("2026-07-16T13:00:00Z").value
    first = controller.advance(retained.package_admission_id, decision, **request)
    assert controller.retained_writer_id(first.story_receipt.event_id, proof=proof()) == first.writer_id
    replay = controller.advance(retained.package_admission_id, decision, **request)
    assert replay == first

    reader = open_private_serving_read_port(
        bindings.target_path,
        target_id=bindings.target_id,
        target_context_digest=bindings.target_context_digest,
        proof=first.read_proof,
    )
    acknowledged = reader.acknowledged_rows()
    assert acknowledged is not None
    assert tuple(row.surface_kind for row in acknowledged.rows) == (
        "ARTICLE",
        "FEED_CARD",
    )
    # Retry keeps the first real effect/ACK, never the pre-story intent time.
    expected_apply = "2026-07-16T13:00:00.000000Z" if failure_boundary == "before_apply" else "2026-07-16T12:00:00.000000Z"
    expected_ack = "2026-07-16T13:01:00.000000Z" if failure_boundary in {"before_apply", "after_apply"} else "2026-07-16T12:01:00.000000Z"
    assert {row.applied_at for row in acknowledged.rows} == {expected_apply}
    assert acknowledged.acknowledgement.first_private_effect_at == expected_apply
    assert acknowledged.acknowledgement.target_acknowledged_at == expected_ack
    assert reader._connection.total_changes == 0
    reader.close()

    if copy_correction:
        monkeypatch.setattr(controller._editorial, "_build_story", original_builder)
        facts = {
            "story_event_id": first.story_receipt.event_id,
            "publication_event_id": first.publication_receipt.event_id,
            "delivery_attempt_event_id": first.attempt_receipt.event_id,
            "delivery_evidence_event_id": first.evidence_receipt.event_id,
        }
        predecessor, old_story = controller.read_acknowledged(facts, proof=proof())
        assert predecessor == first
        assert old_story.copy.writer_id == "newsroom.offline-exact-copy.v2"
        corrected_request = {**request, "expected_story_version": 1,
                             "expected_publication_version": 2,
                             "correction_of": predecessor}
        corrected = controller.advance(retained.package_admission_id, decision, **corrected_request)
        assert corrected.story_receipt.aggregate_version == 2
        assert corrected.publication_receipt.aggregate_version == 3
        assert corrected.attempt_receipt.aggregate_version == 4
        assert controller.advance(retained.package_admission_id, decision, **corrected_request) == corrected
        for changed in (dict(expected_story_version=0), dict(expected_publication_version=1)):
            with pytest.raises(NativePublicationError, match="predecessor binding"):
                controller.advance(retained.package_admission_id, decision, **{**corrected_request, **changed})
        with pytest.raises(NativePublicationError, match="predecessor binding"):
            controller.advance(retained.package_admission_id,
                               _decision(retained, source.admission_id, result="HOLD"), **corrected_request)
        for result, status in ((first, "ORIGINAL"), (corrected, "CORRECTED")):
            port = open_private_serving_read_port(
                bindings.target_path, target_id=bindings.target_id,
                target_context_digest=bindings.target_context_digest, proof=result.read_proof,
            )
            try:
                rows = port.acknowledged_rows()
                assert rows is not None
                assert [json.loads(row.payload_bytes)["correction_status"] for row in rows.rows] == [status, status]
            finally:
                port.close()
        with sqlite3.connect(bindings.target_path) as target:
            assert target.execute("SELECT count(*) FROM private_serving_payloads").fetchone() == (4,)
            key, original_bytes = target.execute(
                "SELECT operation_key,payload_bytes FROM private_serving_payloads WHERE operation_key=?",
                (acknowledged.rows[0].operation_key,),
            ).fetchone()
            target.execute("UPDATE private_serving_payloads SET payload_bytes=? WHERE operation_key=?", (b"corrupt", key))
        with pytest.raises(NativePublicationError, match="predecessor validation"):
            controller.advance(retained.package_admission_id, decision, **corrected_request)
        with sqlite3.connect(bindings.target_path) as target:
            assert target.execute("SELECT count(*) FROM private_serving_payloads").fetchone() == (4,)
            target.execute("UPDATE private_serving_payloads SET payload_bytes=? WHERE operation_key=?", (original_bytes, key))

    controller.close()
    reopened = NativePublicationController(
        objects=system.objects,
        commands=system.commands,
        events=system.events,
        candidate_port=candidate_port,
        evidence_packages=evidence_packages,
        bindings=bindings, clock=effect_clock,
    )
    assert reopened.advance(retained.package_admission_id, decision, **request) == first
    reopened.close()
    candidate_connection.rollback()
    system.close()
    ingress.close()
    candidate_connection.close()
