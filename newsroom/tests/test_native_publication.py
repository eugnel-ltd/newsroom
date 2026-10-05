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


def test_fresh_story_checks_publisher_state_before_any_draft_or_decision(tmp_path):
    from types import SimpleNamespace
    from newsroom.authority import ObjectAdmissionId

    candidates, port, version = _candidate(tmp_path)
    try:
        _, package, _ = _ready_package(version)
        retained = SimpleNamespace(package=package, candidate_version_id=version.version_id,
            candidate_version_digest=version.canonical_digest,
            governing_manifest_digest=version.governing_manifest.canonical_digest,
            package_admission_id=ObjectAdmissionId.new())
        decision = _decision(retained, ObjectAdmissionId.new())
        controller = object.__new__(NativePublicationController)
        controller._candidate_port = port
        controller._evidence = SimpleNamespace(read=lambda *_args, **_kw: retained)
        controller._objects = SimpleNamespace(committed_admission=lambda *_args, **_kw: None)

        def superseded(actual_package, currentness):
            assert actual_package == package and currentness == decision.currentness
            raise EditorialHold(reason='NATIVE_STORY_SOURCE_SUPERSEDED')

        controller._source_currentness_fence = superseded
        controller._record_decision = lambda *_args, **_kw: pytest.fail('source check follows decision/model work')
        with pytest.raises(EditorialHold, match='NATIVE_STORY_SOURCE_SUPERSEDED'):
            controller.advance(ObjectAdmissionId.new(), decision, expected_story_version=0,
                               expected_publication_version=0, expected_delivery_evidence_version=0,
                               proof=proof())
    finally:
        candidates.close()


@pytest.mark.parametrize(('native_copy','failure'),
    [(native, failure) for native in (False,True) for failure in (None,'source','wrong-copy','wrong-version','bad-ack')]
    + [(True,'automatic'),(True,'derived-ack'),(True,'decision'),(True,'after-apply'),(True,'before-apply'),(True,'after-story'),(True,'later-sibling')])
def test_factual_correction_binds_current_and_superseded_ack_without_second_writer(tmp_path, monkeypatch, native_copy, failure):
    from dataclasses import replace
    from newsroom.tests.test_increment10_private_serving import _context, _close
    from newsroom.authority.canonical import digest_bytes
    from newsroom.increment10.evidence import _base_package
    from newsroom.increment10.editorial import EditorialError

    context = _context(tmp_path)
    candidates, port, ingress, system, registries, hydration, definitions, commands, evidence = context[:9]
    original_story = context[9].read_story_version(context[-2], candidate_port=port, proof=proof())
    current = evidence.read(original_story.package_admission_id, candidate_port=port, proof=proof())
    # Build a genuinely different, valid source-bound sibling; do not patch an
    # already admitted claim, payload, model result or ACK.
    passage, stale_package, stale_records = _ready_package(port.require_retained_version(current.candidate_version_id))
    passage = passage.replace('The deadline changed.', 'The deadline was extended.')
    head = replace(stale_package.governed_claims[0], claim='The deadline was extended.',
        supporting_excerpt='The deadline was extended.', rendered_assertion_zh_hant_hk='官方確認限期已經延長。')
    qualification = replace(stale_package.qualification_evidence[0], test_evidence=tuple(
        (key, head.claim if key in {'material_relation_span','reader_action'} else value)
        for key,value in stale_package.qualification_evidence[0].test_evidence))
    stale_package = replace(stale_package, passages=(passage,), observation_digests=(digest_bytes(passage.encode()),),
        governed_claims=(head,stale_package.governed_claims[1]), qualification_evidence=(qualification,),
        substantive_new_information=(head.claim,stale_package.governed_claims[1].claim))
    records = []
    for original in stale_records:
        record = {**original, 'base_package_digest':_base_package(stale_package).digest}
        if record.get('governed_claim_id') == head.claim_id:
            if 'claim_digest' in record: record['claim_digest'] = digest_bytes(head.claim.encode())
            if 'rendered_assertion_digest' in record: record['rendered_assertion_digest'] = digest_bytes(head.rendered_assertion_zh_hant_hk.encode())
            if 'evidence_span_digest' in record: record['evidence_span_digest'] = digest_bytes(head.supporting_excerpt.encode())
            if 'test_evidence' in record: record['test_evidence'] = [list(item)for item in qualification.test_evidence]
        if record['record_type']=='SOURCE_RECORD': record['originating_artefact_digest'] = digest_bytes(passage.encode())
        records.append(record)
    stale_source = system.objects.admit(ObjectAdmissionRequest('evidence.source','correction-stale-source'),passage.encode(),proof=proof()).admission
    stale_ids = tuple(system.objects.admit(ObjectAdmissionRequest('evidence.record',f'correction-stale-record:{index}'),
        canonical_json_bytes(record),proof=proof()).admission.admission_id for index,record in enumerate(records))
    older = evidence.retain(stale_package,
        receipt_id=current.receipt_id, candidate_port=port,
        source_admission_ids=(stale_source.admission_id,),
        record_admission_ids=stale_ids, proof=proof())
    bindings = _bindings(tmp_path, registries, hydration, definitions, commands)
    calls = []
    current_source = [True]
    writer_calls = []
    def story_writer(package, **_identities):
        from newsroom.control_plane.native_story_writer import write_native_story
        def draft(request):
            writer_calls.append('draft')
            claims = request['evidence']['approved_governed_claims']
            return dict(title=next(c['rendered_assertion'] for c in claims if c['claim_role']=='HEADLINE'),
                body='\n\n'.join(c['rendered_assertion'] for c in claims), format='BRIEF',
                evidence_links=[dict(governed_claim_id=c['governed_claim_id'],rendered_assertion=c['rendered_assertion'])for c in claims])
        def review(request):
            writer_calls.append('review')
            claims = request['evidence']['approved_governed_claims']
            return dict(source_package_digest=request['source_package_digest'],draft_digest=request['draft_digest'],
                verdict='PASS',covered_claim_ids=[c['governed_claim_id']for c in claims],
                sentence_support=[dict(sentence_index=i,verdict='SUPPORTED',claim_ids=[link['governed_claim_id']
                    for link in request['draft']['evidence_links']if link['rendered_assertion']in sentence])
                    for i,sentence in enumerate(request['sentences'])],
                factual_checks={key:'PASS'for key in ('numbers','entities','modality','quotations')})
        return write_native_story(package, generate=draft, review=review)
    def fence(package, currentness):
        calls.append(package.digest)
        if not current_source[0]: raise EditorialHold(reason='NATIVE_STORY_SOURCE_SUPERSEDED')
    controller = NativePublicationController(objects=system.objects, commands=system.commands, events=system.events,
        candidate_port=port, evidence_packages=evidence, bindings=bindings,
        clock=lambda: UtcTimestamp.parse('2026-07-16T12:00:00Z'),
        story_writer=story_writer if native_copy else None,
        source_currentness_fence=fence)
    source_id = current.source_admission_ids[0]
    request = dict(expected_story_version=0, expected_publication_version=0,
                   expected_delivery_evidence_version=0, proof=proof())
    try:
        current_ack = controller.advance(current.package_admission_id, _decision(current, source_id), **request)
        stale_ack = controller.advance(older.package_admission_id, _decision(older, stale_source.admission_id),
            **{**request, 'expected_story_version':1, 'expected_publication_version':2})
        prior_writer_calls = list(writer_calls)
        with sqlite3.connect(bindings.target_path)as target:
            original_rows = target.execute('SELECT *FROM private_serving_payloads ORDER BY operation_key').fetchall()
        correction = dict(**{**request, 'expected_story_version':2, 'expected_publication_version':4},
            factual_correction_of=stale_ack, reviewed_copy_from=current_ack)
        if failure == 'later-sibling':
            sibling = evidence.retain(replace(older.package,selection_rationale=older.package.selection_rationale+' A later sibling.'),
                receipt_id=current.receipt_id,candidate_port=port,source_admission_ids=(stale_source.admission_id,),
                record_admission_ids=stale_ids,proof=proof())
            controller.advance(sibling.package_admission_id,_decision(sibling,stale_source.admission_id),
                **{**request,'expected_story_version':2,'expected_publication_version':4})
            count = len(writer_calls)
            with pytest.raises((NativePublicationError,EditorialError)):
                controller.advance(current.package_admission_id,_decision(current,source_id),**correction)
            assert len(writer_calls)==count
            with sqlite3.connect(bindings.target_path)as target:
                assert target.execute('SELECT count(*)FROM private_serving_payloads').fetchone()[0]==6
            return
        if failure in {'automatic','derived-ack'}:
            from types import SimpleNamespace
            references = ('story_event_id','publication_event_id','delivery_attempt_event_id','delivery_evidence_event_id')
            snapshots = {}
            for revision, package, result in (('current',current,current_ack),('stale',older,stale_ack)):
                facts = dict(candidate_id=current.package.candidate_id, **dict(zip(references,(
                    result.story_receipt.event_id,result.publication_receipt.event_id,
                    result.attempt_receipt.event_id,result.evidence_receipt.event_id))))
                snapshots[revision] = dict(stage='ACKNOWLEDGED',facts=facts)
            class Journal:
                units = {revision:(SimpleNamespace(source_id='HK-02'),)for revision in snapshots}
                def iter_summaries(self): return iter(snapshots.items())
                def current(self,revision): return snapshots[revision]
                def advance(self,revision,*,stage,facts): snapshots[revision] = dict(stage=stage,facts=facts)
            # The production source fence alone diagnoses supersession; the
            # complete ACK, package and policy proof paths remain real here.
            controller._source_currentness_fence = lambda package,_: (_ for _ in ()).throw(
                EditorialHold(reason='NATIVE_STORY_SOURCE_SUPERSEDED')) if package.digest==older.package.digest else None
            journal = Journal()
            if failure=='derived-ack':
                snapshots['stale']['facts']['factual_correction_result'] = dict(zip(references,(
                    current_ack.story_receipt.event_id,current_ack.publication_receipt.event_id,
                    current_ack.attempt_receipt.event_id,current_ack.evidence_receipt.event_id)))
                controller.restore_current_publisher_output(journal,proof=proof())
                assert snapshots['stale']['facts']['factual_correction_hold']=='NativePublicationError'
                assert writer_calls==prior_writer_calls
                with sqlite3.connect(bindings.target_path)as target:
                    assert target.execute('SELECT count(*)FROM private_serving_payloads').fetchone()[0]==4
                return
            controller.restore_current_publisher_output(journal,proof=proof())
            assert snapshots['stale']['facts']['story_event_id']==stale_ack.story_receipt.event_id
            assert 'factual_correction_result'in snapshots['stale']['facts']
            assert writer_calls == prior_writer_calls
            original_refs = dict(snapshots['stale']['facts']['factual_correction_result'])
            controller.restore_current_publisher_output(journal,proof=proof())
            assert snapshots['stale']['facts']['factual_correction_result'] == original_refs
            assert writer_calls == prior_writer_calls
            with sqlite3.connect(bindings.target_path)as target:
                assert target.execute('SELECT count(*)FROM private_serving_payloads').fetchone()[0]==6
            return
        if failure in {'after-apply','before-apply','after-story'}:
            target = controller._publication if failure=='after-story' else controller._delivery
            name = 'decide' if failure=='after-story' else 'apply'
            original = getattr(target,name)
            def interrupted(*args,**kwargs):
                if failure=='after-apply': original(*args,**kwargs)
                raise RuntimeError('correction interrupted')
            with monkeypatch.context() as fault:
                fault.setattr(target,name,interrupted)
                with pytest.raises(RuntimeError,match='correction interrupted'):
                    controller.advance(current.package_admission_id,_decision(current,source_id),**correction)
            assert writer_calls == prior_writer_calls
            current_source[0] = False
            controller.close()
            controller = NativePublicationController(objects=system.objects,commands=system.commands,events=system.events,
                candidate_port=port,evidence_packages=evidence,bindings=bindings,
                clock=lambda:UtcTimestamp.parse('2026-07-16T12:00:00Z'),
                story_writer=story_writer if native_copy else None,source_currentness_fence=fence)
            if failure!='after-apply':
                with pytest.raises(EditorialHold,match='NATIVE_STORY_SOURCE_SUPERSEDED'):
                    controller.advance(current.package_admission_id,_decision(current,source_id),**correction)
                with sqlite3.connect(bindings.target_path)as target:
                    assert target.execute('SELECT count(*)FROM private_serving_payloads').fetchone()[0]==4
                return
        elif failure is not None:
            current_source[0] = failure != 'source'
            if failure == 'wrong-copy': correction['reviewed_copy_from'] = stale_ack
            if failure == 'wrong-version': correction['expected_story_version'] = 1
            if failure == 'bad-ack':
                correction['factual_correction_of'] = replace(stale_ack, evidence_receipt=current_ack.evidence_receipt)
            decision = _decision(current,source_id)
            if failure == 'decision':
                changed = replace(decision.currentness[0],source_definition_revision_digest='sha256:'+'0'*64)
                decision = type(decision).create(**{name:(tuple([changed]) if name=='currentness' else getattr(decision,name))
                    for name in decision.__dataclass_fields__ if name!='decision_id'})
            with pytest.raises((NativePublicationError, EditorialError)):
                controller.advance(current.package_admission_id, decision, **correction)
            assert writer_calls == prior_writer_calls
            with sqlite3.connect(bindings.target_path) as target:
                assert target.execute('SELECT count(*) FROM private_serving_payloads').fetchone()[0] == 4
            return
        corrected = controller.advance(current.package_admission_id, _decision(current, source_id),
            **{**request, 'expected_story_version':2, 'expected_publication_version':4},
            factual_correction_of=stale_ack, reviewed_copy_from=current_ack)
        assert corrected.story_receipt.aggregate_version == 3
        assert writer_calls == prior_writer_calls
        def refs(result):
            return dict(story_event_id=result.story_receipt.event_id,publication_event_id=result.publication_receipt.event_id,
                delivery_attempt_event_id=result.attempt_receipt.event_id,delivery_evidence_event_id=result.evidence_receipt.event_id)
        current_story = controller.read_acknowledged(refs(current_ack),proof=proof())[1]
        corrected_story = controller.read_acknowledged(refs(corrected),proof=proof())[1]
        stale_story = controller.read_acknowledged(refs(stale_ack),proof=proof())[1]
        assert corrected_story.copy == current_story.copy
        assert corrected_story.copy.body != stale_story.copy.body
        with sqlite3.connect(bindings.target_path)as target:
            assert all(target.execute('SELECT *FROM private_serving_payloads WHERE operation_key=?',(row[0],)).fetchone()==row
                       for row in original_rows)
        current_source[0] = False  # Original corrected rows settle/replay after a later source change.
        assert controller.advance(current.package_admission_id, _decision(current, source_id),
            **{**request, 'expected_story_version':2, 'expected_publication_version':4},
            factual_correction_of=stale_ack, reviewed_copy_from=current_ack) == corrected
        reader = open_private_serving_read_port(bindings.target_path, target_id=bindings.target_id,
            target_context_digest=bindings.target_context_digest, proof=corrected.read_proof)
        try:
            assert [json.loads(row.payload_bytes)['correction_status'] for row in reader.acknowledged_rows().rows] == ['CORRECTED','CORRECTED']
        finally:
            reader.close()
        assert controller.read_acknowledged(dict(story_event_id=stale_ack.story_receipt.event_id,
            publication_event_id=stale_ack.publication_receipt.event_id,
            delivery_attempt_event_id=stale_ack.attempt_receipt.event_id,
            delivery_evidence_event_id=stale_ack.evidence_receipt.event_id), proof=proof())[0] == stale_ack
    finally:
        controller.close()
        context[0].rollback(); context[3].close(); context[2].close(); context[0].close()


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
    source_current = [True]
    source_checks = []

    def current_source_fence(package, currentness):
        source_checks.append(package.digest)
        if not source_current[0]:
            raise EditorialHold(reason='NATIVE_STORY_SOURCE_SUPERSEDED')

    controller = NativePublicationController(
        objects=system.objects,
        commands=system.commands,
        events=system.events,
        candidate_port=candidate_port,
        evidence_packages=evidence_packages,
        bindings=bindings, clock=effect_clock,
        source_currentness_fence=current_source_fence,
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
        if failure_boundary in {'after_apply', 'after_record'}:
            source_current[0] = False  # Existing effects must settle, not republish.
    first = controller.advance(retained.package_admission_id, decision, **request)
    assert controller.retained_writer_id(first.story_receipt.event_id, proof=proof()) == first.writer_id
    source_current[0] = False
    before_replay_checks = len(source_checks)
    replay = controller.advance(retained.package_admission_id, decision, **request)
    assert replay == first
    assert len(source_checks) == before_replay_checks
    source_current[0] = True

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
    source_current[0] = False
    reopened = NativePublicationController(
        objects=system.objects,
        commands=system.commands,
        events=system.events,
        candidate_port=candidate_port,
        evidence_packages=evidence_packages,
        bindings=bindings, clock=effect_clock,
        source_currentness_fence=current_source_fence,
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


@pytest.mark.parametrize('boundary', ('complete', 'own-partial', 'incomplete', 'corrupt', 'revoked', 'legacy', 'unknown-policy', 'stop', 'empty-slot', 'second-sibling'))
def test_stale_story_slot_requires_exact_acknowledged_sibling(tmp_path, monkeypatch, boundary):
    from dataclasses import replace
    from newsroom.increment10.editorial import EditorialError

    candidates, port, version = _candidate(tmp_path)
    ingress = open_evidence_intake_ingress(tmp_path / 'stale-intake.sqlite3')
    acknowledgement = _receive(ingress, candidates, port, version, request_id='stale-slot')
    clock = lambda: UtcTimestamp.parse('2026-07-16T12:00:00Z')
    system, registries, hydration, definitions, commands = _open(tmp_path / 'stale-authority.sqlite3', clock=clock)
    evidence = _evidence_facade(system, ingress, registries)
    passage, package, records = _ready_package(version)
    source = system.objects.admit(ObjectAdmissionRequest('evidence.source', 'stale-source'), passage.encode(), proof=proof()).admission
    ids = tuple(system.objects.admit(ObjectAdmissionRequest('evidence.record', f'stale-record-{i}'),
        canonical_json_bytes(value), proof=proof()).admission.admission_id for i, value in enumerate(records))
    candidates.execute('BEGIN IMMEDIATE')
    retain = lambda value: evidence.retain(value, receipt_id=acknowledgement.receipt_id, candidate_port=port,
        source_admission_ids=(source.admission_id,), record_admission_ids=ids, proof=proof())
    retained = retain(package)
    incoming = retain(replace(package, selection_rationale=package.selection_rationale + ' Another retained selection.'))
    third = retain(replace(package, selection_rationale=package.selection_rationale + ' A third retained selection.')) if boundary == 'second-sibling' else None
    bindings = _bindings(tmp_path, registries, hydration, definitions, commands)
    if boundary in ('legacy', 'unknown-policy'):
        from newsroom.control_plane.native_policies import native_policy_components
        qualified = native_policy_components(principal_id='principal.alpha', authority_domain='newsroom.authority',
            target_path=bindings.target_path, target_id=bindings.target_id).publication
        bindings = replace(bindings, editorial_policy_bundle_digest=qualified.editorial_policy_bundle_digest,
            publication_authorisation_policy_digest=qualified.publication_authorisation_policy_digest,
            target_policy_digest=qualified.target_policy_digest, retained_policy_pairs=qualified.retained_policy_pairs)
    def decision_for(value, policy=None):
        original = _decision(value, source.admission_id)
        return type(original).create(**{name: (policy or bindings.editorial_policy_bundle_digest) if name == 'policy_bundle_digest' else getattr(original, name)
            for name in original.__dataclass_fields__ if name != 'decision_id'})
    controller = NativePublicationController(objects=system.objects, commands=system.commands, events=system.events,
        candidate_port=port, evidence_packages=evidence, bindings=bindings, clock=clock)
    request = dict(expected_story_version=0, expected_publication_version=0, expected_delivery_evidence_version=0, proof=proof())
    references = lambda result: dict(story_event_id=result.story_receipt.event_id,
        publication_event_id=result.publication_receipt.event_id, delivery_attempt_event_id=result.attempt_receipt.event_id,
        delivery_evidence_event_id=result.evidence_receipt.event_id)
    builder = controller._editorial._build_story
    def old_copy(*args, **kwargs):
        kwargs['writer_id'] = 'newsroom.offline-exact-copy.v2'
        return builder(*args, **kwargs)
    monkeypatch.setattr(controller._editorial, '_build_story', old_copy)
    first = controller.advance(retained.package_admission_id, decision_for(retained), **request)
    monkeypatch.setattr(controller._editorial, '_build_story', builder)
    stale = {**request, 'expected_story_version': 1, 'expected_publication_version': 2}
    corrected = controller.advance(retained.package_admission_id, decision_for(retained), **stale, correction_of=first)
    frozen = controller.read_acknowledged(references(corrected), proof=proof())[1].canonical_bytes()
    selected = retained if boundary == 'own-partial' else incoming
    sibling = references(corrected)
    if boundary == 'incomplete': sibling.pop('delivery_evidence_event_id')
    if boundary == 'corrupt': sibling['delivery_evidence_event_id'] = first.evidence_receipt.event_id
    if boundary == 'revoked':
        system.objects.revoke(source.admission_id, reason_code='SOURCE_NO_LONGER_CURRENT', idempotency_key='stale-revoke', proof=proof())
    if boundary == 'stop':
        from newsroom.control_plane.veto import OperatorDrainRequested
        def stopped(*args, **kwargs):
            raise OperatorDrainRequested('operator drain')
        monkeypatch.setattr(controller, 'read_acknowledged', stopped)
    try:
        if boundary in ('corrupt', 'revoked', 'stop'):
            with pytest.raises(Exception):
                controller.reconcile_stale_intent(selected.package_admission_id,
                    expected_story_version=1, expected_publication_version=2, acknowledged=(sibling,), proof=proof())
            return
        if boundary == 'empty-slot':
            assert controller.reconcile_stale_intent(incoming.package_admission_id,
                expected_story_version=2, expected_publication_version=4, acknowledged=(sibling,), proof=proof()) is None
            return
        reconciled = controller.reconcile_stale_intent(selected.package_admission_id,
            expected_story_version=1, expected_publication_version=2, acknowledged=(sibling,), proof=proof())
        if boundary in ('own-partial', 'incomplete'):
            assert reconciled is None
            return
        assert reconciled == (corrected, sibling)
        with pytest.raises(EditorialError, match='retained write-admission'):
            controller.advance(incoming.package_admission_id, decision_for(incoming), **stale)
        next_request = {**request, 'expected_story_version': 2, 'expected_publication_version': 4,
                        'reconciled_predecessor': sibling}
        incoming_decision = decision_for(incoming, bindings.retained_policy_pairs[0][0] if boundary == 'legacy' else None)
        if boundary == 'unknown-policy':
            with pytest.raises(EditorialError, match='policy'):
                controller.advance(incoming.package_admission_id, decision_for(incoming, 'sha256:' + '0' * 64), **next_request)
            return
        def interrupted(*args, **kwargs):
            raise RuntimeError('own partial publication')
        with monkeypatch.context() as fault:
            fault.setattr(controller._publication, 'decide', interrupted)
            with pytest.raises(RuntimeError, match='own partial'):
                controller.advance(incoming.package_admission_id, incoming_decision, **next_request)
        assert controller.reconcile_stale_intent(incoming.package_admission_id,
            expected_story_version=2, expected_publication_version=4, acknowledged=(sibling,), proof=proof()) is None
        controller.close()
        controller = NativePublicationController(objects=system.objects, commands=system.commands, events=system.events,
            candidate_port=port, evidence_packages=evidence, bindings=bindings, clock=clock)
        published = controller.advance(incoming.package_admission_id, incoming_decision, **next_request)
        assert (published.story_receipt.aggregate_version, published.attempt_receipt.aggregate_version) == (3, 6)
        assert controller.advance(incoming.package_admission_id, incoming_decision, **next_request) == published
        assert controller.read_acknowledged(references(corrected), proof=proof())[1].canonical_bytes() == frozen
        if third is not None:
            assert controller.reconcile_stale_intent(third.package_admission_id,
                expected_story_version=2, expected_publication_version=4, acknowledged=(sibling,), proof=proof()) is None
            with pytest.raises(EditorialError, match='retained write-admission'):
                controller.advance(third.package_admission_id, decision_for(third), **next_request)
            later, later_facts = controller.reconcile_stale_intent(third.package_admission_id,
                expected_story_version=2, expected_publication_version=4, acknowledged=(sibling, references(published)), proof=proof())
            assert (later.story_receipt.aggregate_version, later.attempt_receipt.aggregate_version) == (3, 6)
            final_request = {**request, 'expected_story_version': 3, 'expected_publication_version': 6,
                             'reconciled_predecessor': later_facts}
            final = controller.advance(third.package_admission_id, decision_for(third), **final_request)
            assert (final.story_receipt.aggregate_version, final.attempt_receipt.aggregate_version) == (4, 8)
            assert controller.advance(third.package_admission_id, decision_for(third), **final_request) == final
    finally:
        controller.close(); candidates.rollback(); candidates.close(); system.close()


def test_current_output_restoration_never_selects_unacknowledged_source_bodies():
    from types import SimpleNamespace
    class ColdSources(dict):
        def get(self, *_):
            pytest.fail('restoration selected an unrelated cold source')
    controller = object.__new__(NativePublicationController)
    controller._source_currentness_fence = lambda *_: None
    journal = SimpleNamespace(units=ColdSources(), iter_summaries=lambda: iter((
        ('held', {'stage': 'EVIDENCE_HOLD', 'facts': {}}),
        ('pending', {'stage': 'PUBLICATION_STARTED', 'facts': {}}),
    )))
    controller.restore_current_publisher_output(journal, proof=None)


@pytest.mark.parametrize('case', ['current', 'superseded', 'pending', 'corrupt-header', 'ambiguous'])
def test_current_first_restore_verifies_only_latest_ack_without_caching(case):
    from types import SimpleNamespace
    from newsroom.authority import ObjectAdmissionId
    from newsroom.control_plane.native_publication import _aggregate, STORY_EVENT
    controller=object.__new__(NativePublicationController)
    refs=('story_event_id','publication_event_id','delivery_attempt_event_id','delivery_evidence_event_id')
    def facts(n):
        return {'candidate_id':'candidate', **{key:f'{key}-{n}'for key in refs}}
    members=[('old',facts(1)),('current',facts(2))]
    if case=='pending':members[0][1]['factual_correction_intent']={'pending':True}
    events={f'story_event_id-{n}':SimpleNamespace(event_id=f'story_event_id-{n}',aggregate_id=str(_aggregate('story','candidate')),
        aggregate_type='story',aggregate_version=n,event_type=STORY_EVENT)for n in (1,2)}
    if case=='corrupt-header':events['story_event_id-1'].aggregate_type='other'
    if case=='ambiguous':events['story_event_id-1'].aggregate_version=2
    reads=[]
    controller._events=SimpleNamespace(provenance=lambda event_id,**_:SimpleNamespace(event=events[event_id]))
    def read_ack(values,**_):
        reads.append(values['story_event_id'])
        n=int(values['story_event_id'].rsplit('-',1)[1])
        return SimpleNamespace(story_receipt=SimpleNamespace(aggregate_version=n)),SimpleNamespace(
            copy=SimpleNamespace(writer_id='newsroom.native-story-writer.v1'), package_admission_id=f'package-{n}',
            policy_decision_event_id=f'00000000-0000-4000-8000-{n:012d}',
            policy_decision_admission_id=ObjectAdmissionId.parse(f'00000000-0000-4000-8000-{n+10:012d}'))
    controller.read_acknowledged=read_ack
    controller._candidate_port=object()
    controller._evidence=SimpleNamespace(read=lambda identity,**_:SimpleNamespace(package=identity))
    policy_reads=[]
    def policy(reference,**_):
        assert reference is not None
        policy_reads.append(reference)
        if case=='pending' or case=='superseded' and len(policy_reads)>1:
            raise RuntimeError('full-history-path')
        return SimpleNamespace(currentness=('current',))
    controller._editorial=SimpleNamespace(_read_policy_decision=policy)
    def source_fence(*_):
        if case=='superseded':raise EditorialHold(reason='NATIVE_STORY_SOURCE_SUPERSEDED')
    controller._source_currentness_fence=source_fence
    # The existing full repair path needs complete publication receipts. Reaching
    # this sentinel proves that unproved/pending cases did not use the shortcut.
    if case in {'superseded','pending'}:
        with pytest.raises(RuntimeError,match='full-history-path'):
            controller._restore_publisher_group(object(),members,refs,proof())
    elif case in {'corrupt-header','ambiguous'}:
        with pytest.raises(NativePublicationError,match='header differs|version is ambiguous'):
            controller._restore_publisher_group(object(),members,refs,proof())
        assert reads==[]
    else:
        for _ in range(2):controller._restore_publisher_group(object(),members,refs,proof())
        assert reads==['story_event_id-2','story_event_id-2']
