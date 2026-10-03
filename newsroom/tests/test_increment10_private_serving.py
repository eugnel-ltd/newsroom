from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from newsroom.authority import (
    AggregateId,
    CommandDefinition,
    CommandRegistry,
    HydrationPolicyContract,
    HydrationPolicyRegistry,
    ObjectAdmissionDefinition,
    ObjectAdmissionPayload,
    ObjectAdmissionRegistry,
    ObjectAdmissionRequest,
    ObjectLimits,
    PayloadMode,
    StaticAuthenticator,
    StaticAuthorizer,
    StaticPrincipal,
    SemanticCommand,
    TrustScope,
)
from newsroom.authority.canonical import canonical_json_bytes
from newsroom.increment10.editorial import StoryVersionRequest
from newsroom.increment10.ingress import open_evidence_intake_ingress
from newsroom.increment10.private_serving import (
    ATTEMPT_ADMISSION_TYPE,
    ATTEMPT_CLASS,
    ATTEMPT_COMMAND,
    ATTEMPT_EVENT,
    ATTEMPT_PURPOSE,
    ATTEMPT_USE,
    EVIDENCE_ADMISSION_TYPE,
    EVIDENCE_CLASS,
    EVIDENCE_COMMAND,
    EVIDENCE_EVENT,
    EVIDENCE_PURPOSE,
    EVIDENCE_USE,
    SERVING_RETENTION_SCOPE,
    SERVING_SECURITY_SCOPE,
    PrivateServingError,
    open_private_serving_delivery,
)
from newsroom.increment10.publication import (
    PUBLICATION_COMMAND,
    PUBLICATION_EVENT,
    PublicationRequest,
)
from newsroom.tests.authority_a2b_helpers import open_object_system
from newsroom.tests.authority_helpers import proof
from newsroom.tests.test_increment10_editorial import (
    _decision,
    _evidence_facade,
    _native,
    _ready_package,
    _record_decision,
)
from newsroom.tests.test_increment10_ingress import _candidate, _receive
from newsroom.tests.test_increment10_publication import (
    _publication,
    _publication_registries,
)


def _registries():
    base = _publication_registries()
    rights, hydration, admissions = base[0]
    hydration_contracts = tuple(
        HydrationPolicyContract(
            policy_id=f"{purpose}-read-v1",
            contract_version="hydration-v1",
            implementation_version="hydration-static-v1",
            purpose=purpose,
            required_scope="authority.objects.read",
            allowed_principal_ids=frozenset({"principal.alpha"}),
            allowed_authority_domains=frozenset({"newsroom.authority"}),
            allowed_object_classes=frozenset({object_class}),
            allowed_uses=frozenset({allowed_use}),
            allowed_security_scopes=frozenset({SERVING_SECURITY_SCOPE}),
            allowed_retention_scopes=frozenset({SERVING_RETENTION_SCOPE}),
            max_bytes=1024 * 1024,
        )
        for purpose, object_class, allowed_use in (
            (ATTEMPT_PURPOSE, ATTEMPT_CLASS, ATTEMPT_USE),
            (EVIDENCE_PURPOSE, EVIDENCE_CLASS, EVIDENCE_USE),
        )
    )
    all_hydration = HydrationPolicyRegistry(
        (*hydration.contracts(), *hydration_contracts)
    )
    definitions = tuple(
        ObjectAdmissionDefinition(
            admission_type=admission_type,
            definition_version="admission-v1",
            object_class=object_class,
            allowed_use=allowed_use,
            security_scope=SERVING_SECURITY_SCOPE,
            retention_scope=SERVING_RETENTION_SCOPE,
            required_write_scope=write_scope,
            required_read_scope="authority.objects.read",
            required_manage_scope="authority.objects.manage",
            rights_policy_contract_digest=rights.contracts()[0].contract_digest,
            hydration_policy_contract_digests=frozenset(
                {hydration_contract.contract_digest}
            ),
        )
        for (
            admission_type,
            object_class,
            allowed_use,
            write_scope,
        ), hydration_contract in zip(
            (
                (
                    ATTEMPT_ADMISSION_TYPE,
                    ATTEMPT_CLASS,
                    ATTEMPT_USE,
                    "authority.private-serving.attempt",
                ),
                (
                    EVIDENCE_ADMISSION_TYPE,
                    EVIDENCE_CLASS,
                    EVIDENCE_USE,
                    "authority.private-serving.evidence",
                ),
            ),
            hydration_contracts,
            strict=True,
        )
    )
    all_admissions = ObjectAdmissionRegistry(
        (*admissions.definitions(), *definitions),
        rights_policies=rights,
        hydration_policies=all_hydration,
    )
    payload = base[-1].contracts()[0]
    commands = tuple(
        CommandDefinition(
            command_type=command_type,
            definition_version="private-serving-command-v1",
            aggregate_type=aggregate_type,
            event_type=event_type,
            event_schema_version=1,
            payload_mode=PayloadMode.OBJECT_ADMISSION,
            payload_schema_version=payload.schema_version,
            payload_schema_contract_version=payload.contract_version,
            payload_schema_contract_digest=payload.contract_digest,
            payload_canonicalizer_version=payload.canonicalizer_implementation_version,
            trust_scope=TrustScope.ADMITTED,
            security_scope=SERVING_SECURITY_SCOPE,
            retention_scope=SERVING_RETENTION_SCOPE,
            required_scope=scope,
            required_object_class=object_class,
            required_allowed_use=allowed_use,
        )
        for command_type, aggregate_type, event_type, scope, object_class, allowed_use in (
            (
                ATTEMPT_COMMAND,
                "publication",
                ATTEMPT_EVENT,
                "authority.private-serving.attempt",
                ATTEMPT_CLASS,
                ATTEMPT_USE,
            ),
            (
                EVIDENCE_COMMAND,
                "private_serving_attempt",
                EVIDENCE_EVENT,
                "authority.private-serving.evidence",
                EVIDENCE_CLASS,
                EVIDENCE_USE,
            ),
        )
    )
    return (
        ((rights, all_hydration, all_admissions), *base[1:-2], CommandRegistry((*base[-2].definitions(), *commands)), base[-1]),
        hydration_contracts,
        definitions,
        commands,
    )


def _open(path: Path, *, clock=None):
    registries, hydration, definitions, commands = _registries()
    scopes = frozenset(
        {
            "authority.evidence.admit",
            "authority.editorial.decide",
            "authority.editorial.story.write",
            "authority.editorial.story.admit",
            "authority.publication.surface.write",
            "authority.publication.decide",
            "authority.private-serving.attempt",
            "authority.private-serving.evidence",
            "authority.objects.read",
            "authority.objects.manage",
            "authority.objects.lifecycle.write",
            "authority.fixture.events.read",
        }
    )
    system = open_object_system(
        path, clock=clock,
        policy_registries=registries[0],
        authenticator=StaticAuthenticator(
            credentials={"token-1": StaticPrincipal("principal.alpha")},
            authority_domain="newsroom.authority",
        ),
        authorizer=StaticAuthorizer(
            policy_version="authz-v1",
            grants_by_principal={"principal.alpha": scopes},
        ),
        command_registry=registries[-2],
        payload_schema_registry=registries[-1],
        object_limits=ObjectLimits(
            global_max_bytes=1024 * 1024,
            class_max_bytes={
                item.object_class: 1024 * 1024
                for item in registries[0][2].definitions()
            },
            max_read_bytes=1024 * 1024,
            min_free_bytes=0,
            io_chunk_bytes=64,
            max_staging_bytes=1024 * 1024,
            max_range_bytes=1024 * 1024,
        ),
    )
    return system, registries, hydration, definitions, commands


def _context(tmp_path: Path):
    candidate_connection, port, version = _candidate(tmp_path)
    ingress = open_evidence_intake_ingress(tmp_path / "intake.sqlite3")
    acknowledgement = _receive(
        ingress, candidate_connection, port, version, request_id="request-1"
    )
    system, registries, hydration, definitions, commands = _open(
        tmp_path / "objects.sqlite3"
    )
    evidence = _evidence_facade(system, ingress, registries)
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
    retained = evidence.retain(
        package,
        receipt_id=acknowledgement.receipt_id,
        candidate_port=port,
        source_admission_ids=(source.admission_id,),
        record_admission_ids=record_ids,
        proof=proof(),
    )
    decision_reference = _record_decision(
        system, _decision(retained, source.admission_id)
    )
    editorial = _native(system, evidence, registries)
    story_receipt, _ = editorial.admit_story_version(
        StoryVersionRequest(AggregateId.new(), 0, "story-1"),
        package_admission_id=retained.package_admission_id,
        decision_reference=decision_reference,
        candidate_port=port,
        proof=proof(),
    )
    publication = _publication(system, evidence, editorial, registries)
    publication_receipt, _ = publication.decide(
        PublicationRequest(
            AggregateId.new(),
            0,
            "publication-1",
            "AUTO_PUBLISH",
            ("AUTO_POLICY_APPROVED",),
            "2026-09-08T12:03:00Z",
        ),
        story_receipt=story_receipt,
        candidate_port=port,
        proof=proof(),
    )
    return (
        candidate_connection,
        port,
        ingress,
        system,
        registries,
        hydration,
        definitions,
        commands,
        evidence,
        editorial,
        publication,
        story_receipt,
        publication_receipt,
    )


def _delivery(tmp_path, context, *, filename="private-serving.sqlite3"):
    system, publication = context[3], context[10]
    hydration, definitions, commands = context[5:8]
    return open_private_serving_delivery(
        tmp_path / filename,
        objects=system.objects,
        commands=system.commands,
        events=system.events,
        publication=publication,
        adapter_principal_id="principal.alpha",
        authority_domain="newsroom.authority",
        target_id="newsroom-app-serving-launch",
        target_context_digest="sha256:" + "a" * 64,
        attempt_hydration_policy_digest=hydration[0].contract_digest,
        evidence_hydration_policy_digest=hydration[1].contract_digest,
        attempt_admission_definition_digest=definitions[0].digest,
        evidence_admission_definition_digest=definitions[1].digest,
        attempt_command_definition_digest=commands[0].digest,
        evidence_command_definition_digest=commands[1].digest,
    )


def _close(context, delivery) -> None:
    context[0].rollback()
    delivery.close()
    context[3].close()
    context[2].close()
    context[0].close()


def test_current_source_fence_precedes_new_rows_but_not_exact_effect_replay(tmp_path):
    context = _context(tmp_path)
    delivery = _delivery(tmp_path, context)
    request = dict(story_receipt=context[-2], candidate_port=context[1], proof=proof())
    calls = []

    def superseded():
        calls.append('source')
        raise ValueError('source superseded')

    try:
        attempt, batch = delivery.begin(context[-1], **request)
        with pytest.raises(ValueError, match='source superseded'):
            delivery.apply(attempt, publication_receipt=context[-1],
                           applied_at='2026-07-16T11:00:00Z',
                           before_new_effect=superseded, **request)
        assert calls == ['source']
        assert all(delivery.query(item.operation_key) is None for item in batch.attempts)
        assert not delivery._connection.in_transaction
        rows = delivery.apply(attempt, publication_receipt=context[-1],
                              applied_at='2026-07-16T11:00:00Z',
                              before_new_effect=lambda: calls.append('current'), **request)
        assert calls == ['source', 'current']
        # The original applied rows can settle after a later publisher update.
        assert delivery.apply(attempt, publication_receipt=context[-1],
                              applied_at='2026-07-16T11:10:00Z',
                              before_new_effect=superseded, **request) == rows
        assert calls == ['source', 'current']
        # A genuinely missing surface is a new effect, not an ACK replay.
        article, card = rows
        delivery._connection.execute('DELETE FROM private_serving_payloads WHERE operation_key=?',
                                     (card.operation_key,))
        delivery._connection.commit()
        with pytest.raises(ValueError, match='source superseded'):
            delivery.apply(attempt, publication_receipt=context[-1],
                           applied_at='2026-07-16T11:20:00Z',
                           before_new_effect=superseded, **request)
        assert delivery.query(article.operation_key) == article
        assert delivery.query(card.operation_key) is None
        assert not delivery._connection.in_transaction
    finally:
        _close(context, delivery)


def test_committed_publication_projects_and_acknowledges_exact_private_rows(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    connection, port, _, system, *_, story_receipt, publication_receipt = context
    delivery = _delivery(tmp_path, context)
    assert (tmp_path / "private-serving.sqlite3").stat().st_mode & 0o777 == 0o600
    attempt_receipt, batch = delivery.begin(
        publication_receipt,
        story_receipt=story_receipt,
        candidate_port=port,
        proof=proof(),
    )
    rows = delivery.apply(
        attempt_receipt,
        publication_receipt=publication_receipt,
        story_receipt=story_receipt,
        candidate_port=port,
        applied_at="2026-07-16T11:00:00Z",
        proof=proof(),
    )
    evidence = delivery.observe(
        attempt_receipt,
        publication_receipt=publication_receipt,
        story_receipt=story_receipt,
        candidate_port=port,
        observed_at="2026-07-16T11:30:00Z",
        proof=proof(),
    )
    evidence_receipt = delivery.record(
        evidence, attempt_receipt, expected_version=0, proof=proof()
    )

    assert tuple(row.surface_kind for row in rows) == ("ARTICLE", "FEED_CARD")
    assert evidence.outcome == "ACKNOWLEDGED"
    assert evidence.acknowledgement is not None
    acknowledged = delivery.acknowledged_rows(
        evidence_receipt,
        attempt_receipt,
        publication_receipt=publication_receipt,
        story_receipt=story_receipt,
        candidate_port=port,
        proof=proof(),
    )
    assert acknowledged is not None
    assert acknowledged.rows == rows
    assert acknowledged.primary_feed_published_at != rows[1].applied_at
    assert delivery.begin(
        publication_receipt,
        story_receipt=story_receipt,
        candidate_port=port,
        proof=proof(),
    )[0] == attempt_receipt
    assert delivery.apply(
        attempt_receipt,
        publication_receipt=publication_receipt,
        story_receipt=story_receipt,
        candidate_port=port,
        applied_at="2026-07-16T11:00:00Z",
        proof=proof(),
    ) == rows

    connection.rollback()
    delivery.close()
    reopened = _delivery(tmp_path, context)
    connection.execute("BEGIN IMMEDIATE")
    reopened_acknowledged = reopened.acknowledged_rows(
        evidence_receipt,
        attempt_receipt,
        publication_receipt=publication_receipt,
        story_receipt=story_receipt,
        candidate_port=port,
        proof=proof(),
    )
    assert reopened_acknowledged == acknowledged
    _close(context, reopened)


def test_lost_ack_observes_exact_effect_without_duplicate_and_conflicts_fail_closed(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    port, story_receipt, publication_receipt = context[1], context[-2], context[-1]
    delivery = _delivery(tmp_path, context)
    receipt, batch = delivery.begin(
        publication_receipt,
        story_receipt=story_receipt,
        candidate_port=port,
        proof=proof(),
    )
    missing = delivery.observe(
        receipt,
        publication_receipt=publication_receipt,
        story_receipt=story_receipt,
        candidate_port=port,
        observed_at="2026-07-16T10:30:00Z",
        proof=proof(),
    )
    missing_receipt = delivery.record(
        missing, receipt, expected_version=0, proof=proof()
    )
    rows = delivery.apply(
        receipt,
        publication_receipt=publication_receipt,
        story_receipt=story_receipt,
        candidate_port=port,
        applied_at="2026-07-16T11:00:00Z",
        proof=proof(),
    )
    delivery.close()
    delivery = _delivery(tmp_path, context)
    assert delivery.read_evidence(missing_receipt, receipt, proof=proof()) == missing
    matching = delivery.observe(
        receipt,
        publication_receipt=publication_receipt,
        story_receipt=story_receipt,
        candidate_port=port,
        observed_at="2026-07-16T11:30:00Z",
        proof=proof(),
    )
    matching_receipt = delivery.record(
        matching, receipt, expected_version=1, proof=proof()
    )
    assert matching.outcome == "ACKNOWLEDGED"
    acknowledged = delivery.acknowledged_rows(
        matching_receipt,
        receipt,
        publication_receipt=publication_receipt,
        story_receipt=story_receipt,
        candidate_port=port,
        proof=proof(),
    )
    assert acknowledged is not None and acknowledged.rows == rows
    later = delivery.observe(
        receipt,
        publication_receipt=publication_receipt,
        story_receipt=story_receipt,
        candidate_port=port,
        observed_at="2026-07-16T11:40:00Z",
        proof=proof(),
    )
    assert delivery.record(
        later, receipt, expected_version=1, proof=proof()
    ) == matching_receipt
    with pytest.raises(PrivateServingError, match="binding differs"):
        replace(matching, target_id="other-target")
    other_target = _delivery(tmp_path, context, filename="retargeted.sqlite3")
    with pytest.raises(PrivateServingError, match="publication differs"):
        other_target.apply(
            receipt,
            publication_receipt=publication_receipt,
            story_receipt=story_receipt,
            candidate_port=port,
            applied_at="2026-07-16T11:00:00Z",
            proof=proof(),
        )
    other_target.close()
    delivery._connection.execute(
        "UPDATE private_serving_payloads SET payload_bytes=? WHERE operation_key=?",
        (b"{}", batch.attempts[0].operation_key),
    )
    ambiguous = delivery.observe(
        receipt,
        publication_receipt=publication_receipt,
        story_receipt=story_receipt,
        candidate_port=port,
        observed_at="2026-07-16T11:30:00Z",
        proof=proof(),
    )
    evidence_receipt = delivery.record(
        ambiguous, receipt, expected_version=2, proof=proof()
    )
    assert ambiguous.outcome == "AMBIGUOUS"
    assert delivery.acknowledged_rows(
        evidence_receipt,
        receipt,
        publication_receipt=publication_receipt,
        story_receipt=story_receipt,
        candidate_port=port,
        proof=proof(),
    ) is None
    with pytest.raises(PrivateServingError, match="conflicts"):
        delivery.apply(
            receipt,
            publication_receipt=publication_receipt,
            story_receipt=story_receipt,
            candidate_port=port,
            applied_at="2026-07-16T11:00:00Z",
            proof=proof(),
        )
    assert len(rows) == 2
    _close(context, delivery)


def test_hold_or_stale_publication_cannot_create_a_target_effect(tmp_path: Path) -> None:
    context = _context(tmp_path)
    connection, port, _, system, *_, publication, story_receipt, publication_receipt = context
    delivery = _delivery(tmp_path, context)
    system.commands.execute(
        # A later publication command wins the aggregate fence before dispatch.
        SemanticCommand(
            PUBLICATION_COMMAND,
            publication_receipt.publication_id,
            publication_receipt.aggregate_version,
            ObjectAdmissionPayload(publication_receipt.admission_id),
            "newer-publication-state",
        ),
        proof=proof(),
    )
    with pytest.raises(PrivateServingError, match="fence rejected"):
        delivery.begin(
            publication_receipt,
            story_receipt=story_receipt,
            candidate_port=port,
            proof=proof(),
        )
    assert delivery.query("missing-operation") is None

    with pytest.raises(PrivateServingError, match="no private target operations"):
        hold_receipt, _ = publication.decide(
            PublicationRequest(
                AggregateId.new(),
                0,
                "publication-hold",
                "HOLD_FOR_REVIEW",
                ("REVIEW_REQUIRED",),
                "2026-09-08T12:06:00Z",
            ),
            story_receipt=story_receipt,
            candidate_port=port,
            proof=proof(),
        )
        delivery.begin(
            hold_receipt,
            story_receipt=story_receipt,
            candidate_port=port,
            proof=proof(),
        )
    _close(context, delivery)


def test_mutable_receipt_objects_are_rejected_before_authority_access() -> None:
    from newsroom.increment10.private_serving import PrivateServingDelivery

    class ChangingReceipt:
        @property
        def event_id(self):
            raise AssertionError("receipt property must not be read")

    delivery = object.__new__(PrivateServingDelivery)
    with pytest.raises(PrivateServingError, match="exact AttemptReceipt"):
        delivery._read_attempt_object(ChangingReceipt(), proof=None)
    with pytest.raises(PrivateServingError, match="exact EvidenceReceipt"):
        delivery.read_evidence(ChangingReceipt(), ChangingReceipt(), proof=None)
