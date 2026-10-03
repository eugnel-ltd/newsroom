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
@pytest.mark.parametrize("copy_correction", (False, True, "narrative", "reviewed_headline"))
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
        if copy_correction in {"narrative", "reviewed_headline"}:
            from newsroom.tests.test_native_story_editorial import _writer
            controller._editorial._story_writer = _writer([], headline_body_link=copy_correction == "reviewed_headline")
            controller.writer_contract_version = "newsroom.native-story-writer.v1"
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


@pytest.mark.parametrize('pair_mismatch', [None, 'before-publication-command', 'old-editorial-current-auth', 'current-editorial-old-auth', 'unknown-auth'])
def test_known_policy_pair_upgrade_reads_old_ack_and_corrects_once(tmp_path, monkeypatch, pair_mismatch):
    from contextlib import nullcontext
    from dataclasses import replace
    from types import SimpleNamespace
    from newsroom.control_plane import admission, native_story_model
    from newsroom.control_plane.native_policies import native_policy_components
    from newsroom.control_plane.native_story_writer import DRAFT_SCHEMA, REVIEW_SCHEMA, DRAFT_SYSTEM
    from newsroom.tests.test_native_assessor import _usage
    from newsroom.tests.test_native_story_editorial import _OLD_WRITE_POLICY

    candidates, port, version = _candidate(tmp_path)
    ingress = open_evidence_intake_ingress(tmp_path / 'pair-intake.sqlite3')
    acknowledgement = _receive(ingress, candidates, port, version, request_id='pair-upgrade')
    fixed_clock = lambda: UtcTimestamp.parse('2026-07-16T12:00:00Z')
    system, registries, hydration, definitions, commands = _open(tmp_path / 'pair-authority.sqlite3', clock=fixed_clock)
    evidence = _evidence_facade(system, ingress, registries)
    passage, package, records = _ready_package(version)
    source = system.objects.admit(ObjectAdmissionRequest('evidence.source', 'pair-source'), passage.encode(), proof=proof()).admission
    ids = tuple(system.objects.admit(ObjectAdmissionRequest('evidence.record', f'pair-record-{i}'),
        canonical_json_bytes(value), proof=proof()).admission.admission_id for i, value in enumerate(records))
    candidates.execute('BEGIN IMMEDIATE')
    retained = evidence.retain(package, receipt_id=acknowledgement.receipt_id, candidate_port=port,
        source_admission_ids=(source.admission_id,), record_admission_ids=ids, proof=proof())
    base = _bindings(tmp_path, registries, hydration, definitions, commands)
    qualified = native_policy_components(principal_id='principal.alpha', authority_domain='newsroom.authority',
        target_path=base.target_path, target_id=base.target_id).publication
    old_editorial, old_authorisation = qualified.retained_policy_pairs[0]
    current = replace(base, editorial_policy_bundle_digest=qualified.editorial_policy_bundle_digest,
        publication_authorisation_policy_digest=qualified.publication_authorisation_policy_digest,
        target_policy_digest=qualified.target_policy_digest, retained_policy_pairs=qualified.retained_policy_pairs)
    historical = replace(current, editorial_policy_bundle_digest=old_editorial,
        publication_authorisation_policy_digest=old_authorisation, retained_policy_pairs=())
    if pair_mismatch == 'old-editorial-current-auth':
        historical = replace(historical, publication_authorisation_policy_digest=current.publication_authorisation_policy_digest)
    elif pair_mismatch == 'current-editorial-old-auth':
        historical = replace(historical, editorial_policy_bundle_digest=current.editorial_policy_bundle_digest)
    elif pair_mismatch == 'unknown-auth':
        historical = replace(historical, publication_authorisation_policy_digest='sha256:' + '0' * 64)
    original = _decision(retained, source.admission_id)
    decision = type(original).create(**{name: historical.editorial_policy_bundle_digest if name == 'policy_bundle_digest' else getattr(original, name)
        for name in original.__dataclass_fields__ if name != 'decision_id'})
    arguments = dict(objects=system.objects, commands=system.commands, events=system.events,
        candidate_port=port, evidence_packages=evidence, clock=fixed_clock)
    request = dict(expected_story_version=0, expected_publication_version=0,
                   expected_delivery_evidence_version=0, proof=proof())
    old = NativePublicationController(**arguments, bindings=historical)
    with monkeypatch.context() as legacy:
        legacy.setattr(admission, 'WRITE_ADMISSION_POLICY_VERSION', _OLD_WRITE_POLICY)
        if pair_mismatch == 'before-publication-command':
            from newsroom.increment10.publication import PUBLICATION_COMMAND
            execute = type(system.commands).execute
            def interrupted(commands_port, command, **kwargs):
                if command.command_type == PUBLICATION_COMMAND:
                    raise RuntimeError('interrupted before publication command')
                return execute(commands_port, command, **kwargs)
            with monkeypatch.context() as fault:
                fault.setattr(type(system.commands), 'execute', interrupted)
                with pytest.raises(RuntimeError, match='before publication command'):
                    old.advance(retained.package_admission_id, decision, **request)
            old.close()
            old = NativePublicationController(**arguments, bindings=current)
        first = old.advance(retained.package_admission_id, decision, **request)
    references = lambda result: dict(story_event_id=result.story_receipt.event_id,
        publication_event_id=result.publication_receipt.event_id, delivery_attempt_event_id=result.attempt_receipt.event_id,
        delivery_evidence_event_id=result.evidence_receipt.event_id)
    old_bytes = old.read_acknowledged(references(first), proof=proof())[1].canonical_bytes()
    old.close()

    usage_root = tmp_path / 'usage'; usage_root.mkdir()
    usage, assessor = _usage(usage_root, monkeypatch)
    policies = {phase: native_story_model.story_model_policy(assessor._policy, phase=phase, schema=schema,
        revision='1' * 40, evidence_digest='sha256:' + 'a' * 64)
        for phase, schema in (('DRAFT', DRAFT_SCHEMA), ('REVIEW', REVIEW_SCHEMA))}
    for value in policies.values(): usage.register_policy(value)
    monkeypatch.setattr(native_story_model, 'cont_writer_implementation_identity', lambda: ('1' * 40, True))
    monkeypatch.setattr(native_story_model, 'read_grok_command_semantic_version', lambda: '1.0.8')
    calls = []
    def invoke(prompt, **kwargs):
        value = json.loads(prompt); claims = value['evidence']['approved_governed_claims']
        if kwargs['system_instruction'] == DRAFT_SYSTEM:
            calls.append('draft')
            result = dict(title=next(c['rendered_assertion'] for c in claims if c['claim_role'] == 'HEADLINE'),
                body='\n\n'.join(c['rendered_assertion'] for c in claims), format='BRIEF',
                evidence_links=[dict(governed_claim_id=c['governed_claim_id'], rendered_assertion=c['rendered_assertion']) for c in claims])
        else:
            calls.append('review')
            result = dict(source_package_digest=value['source_package_digest'], draft_digest=value['draft_digest'],
                verdict='PASS', covered_claim_ids=[c['governed_claim_id'] for c in claims],
                sentence_support=[dict(sentence_index=i, verdict='SUPPORTED', claim_ids=[link['governed_claim_id']
                    for link in value['draft']['evidence_links'] if link['rendered_assertion'] in sentence])
                    for i, sentence in enumerate(value['sentences'])],
                factual_checks={key: 'PASS' for key in ('numbers','entities','modality','quotations')})
        return SimpleNamespace(text=canonical_json_bytes(result).decode(), usage=dict(usage_basis='PROVIDER_REPORTED',
            input_tokens=1, output_tokens=1, context_tokens=1, total_tokens=2))
    model = native_story_model.NativeStoryModel(usage, policies, fence=nullcontext, stop_check=lambda: None, invoke=invoke)
    controller = NativePublicationController(**arguments, bindings=current, story_writer=model.write)
    try:
        if pair_mismatch not in (None, 'before-publication-command'):
            with pytest.raises(NativePublicationError, match='predecessor validation'):
                controller.read_acknowledged(references(first), proof=proof())
            assert calls == []
            with usage._connection() as c:
                assert c.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0] == 0
            return
        prior, frozen = controller.read_acknowledged(references(first), proof=proof())
        assert prior == first and frozen.canonical_bytes() == old_bytes
        assert controller.advance(retained.package_admission_id, decision, **request) == first
        assert calls == []
        from newsroom.authority import AggregateId
        from newsroom.increment10.editorial import DecisionReference, StoryVersionRequest
        from newsroom.increment10.publication import PublicationRequest, PublicationError
        old_reference = DecisionReference(frozen.policy_decision_event_id, frozen.policy_decision_admission_id)
        with pytest.raises(EditorialHold, match='CURRENT_EDITORIAL_POLICY_REQUIRED'):
            controller._editorial.admit_story_version(StoryVersionRequest(AggregateId.new(), 0, 'fresh-old-policy'),
                package_admission_id=retained.package_admission_id, decision_reference=old_reference,
                candidate_port=port, proof=proof())
        with pytest.raises(PublicationError, match='current editorial policy required'):
            controller._publication.decide(PublicationRequest(first.publication_receipt.publication_id, 2,
                'fresh-old-publication', 'AUTO_PUBLISH', decision.evidence_gate_results[0][:1], decision.evaluated_at),
                story_receipt=first.story_receipt, candidate_port=port, proof=proof())
        assert calls == []
        denied = type(decision).create(**{name: 'sha256:' + '0' * 64 if name == 'policy_bundle_digest' else getattr(decision, name)
            for name in decision.__dataclass_fields__ if name != 'decision_id'})
        with pytest.raises(NativePublicationError, match='predecessor binding'):
            controller.advance(retained.package_admission_id, denied, **{**request,
                'expected_story_version':1, 'expected_publication_version':2, 'correction_of':prior})
        corrected_request = {**request, 'expected_story_version':1, 'expected_publication_version':2, 'correction_of':prior}
        corrected = controller.advance(retained.package_admission_id, decision, **corrected_request)
        assert calls == ['draft','review'] and corrected.writer_id == 'newsroom.native-story-writer.v1'
        assert controller.read_acknowledged(references(first), proof=proof())[1].canonical_bytes() == old_bytes
        assert controller.advance(retained.package_admission_id, decision, **corrected_request) == corrected
        controller.close()
        controller = NativePublicationController(**arguments, bindings=current, story_writer=model.write)
        reopened_result, current_story = controller.read_acknowledged(references(corrected), proof=proof())
        assert reopened_result == corrected
        current_policy = controller._editorial._read_policy_decision(
            DecisionReference(current_story.policy_decision_event_id, current_story.policy_decision_admission_id),
            retained=retained, proof=proof())
        assert current_policy.policy_bundle_digest == current.editorial_policy_bundle_digest
        assert calls == ['draft','review']
        with usage._connection() as c:
            assert c.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0] == 2
            assert c.execute("SELECT count(*) FROM model_invocation_terminals WHERE outcome='COMPLETE' AND usage_status='REPORTED'").fetchone()[0] == 2
        assert controller._editorial.current_copy_decision(decision).policy_bundle_digest == current.editorial_policy_bundle_digest
        system.objects.revoke(source.admission_id, reason_code='SOURCE_NO_LONGER_CURRENT',
            idempotency_key='pair-source-revoked', proof=proof())
        with pytest.raises(NativePublicationError, match='predecessor validation'):
            controller.read_acknowledged(references(corrected), proof=proof())
        assert calls == ['draft','review']
    finally:
        controller.close(); candidates.rollback(); candidates.close(); system.close()
