from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from newsroom.authority import (
    AggregateId,
    CommandDefinition,
    CommandRegistry,
    HydrationPolicyContract,
    HydrationPolicyRegistry,
    HydrationRequest,
    ObjectAdmissionDefinition,
    ObjectAdmissionPayload,
    ObjectAdmissionRegistry,
    ObjectAdmissionRequest,
    ObjectLimits,
    PayloadMode,
    PayloadSchemaRegistry,
    SemanticCommand,
    StaticAuthenticator,
    StaticAuthorizer,
    StaticPrincipal,
    TrustScope,
)
from newsroom.authority.canonical import canonical_json_bytes
from newsroom.authority.persistence import ExpectedVersionConflict
from newsroom.increment10.editorial import STORY_EVENT, StoryVersionRequest
from newsroom.increment10.ingress import open_evidence_intake_ingress
from newsroom.increment10.publication import (
    LAUNCH_CAPABILITIES,
    PUBLICATION_COMMAND,
    PUBLICATION_EVENT,
    PUBLICATION_RETENTION_SCOPE,
    PUBLICATION_SECURITY_SCOPE,
    SURFACE_ADMISSION_TYPE,
    SURFACE_CLASS,
    SURFACE_PURPOSE,
    SURFACE_USE,
    TRANSACTION_ADMISSION_TYPE,
    TRANSACTION_CLASS,
    TRANSACTION_PURPOSE,
    TRANSACTION_USE,
    OfflinePublication,
    PublicationError,
    PublicationReceipt,
    PublicationRequest,
)
from newsroom.tests.authority_a2b_helpers import open_object_system
from newsroom.tests.authority_helpers import proof
from newsroom.tests.test_increment10_editorial import (
    _command_contract,
    _decision,
    _evidence_facade,
    _native,
    _ready_package,
    _record_decision,
    _registries,
)
from newsroom.tests.test_increment10_ingress import _candidate, _receive


def _publication_registries():
    base = _registries()
    rights, hydration, admissions = base[0]
    publication_hydration = tuple(
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
            allowed_security_scopes=frozenset({PUBLICATION_SECURITY_SCOPE}),
            allowed_retention_scopes=frozenset({PUBLICATION_RETENTION_SCOPE}),
            max_bytes=1024 * 1024,
        )
        for purpose, object_class, allowed_use in (
            (SURFACE_PURPOSE, SURFACE_CLASS, SURFACE_USE),
            (TRANSACTION_PURPOSE, TRANSACTION_CLASS, TRANSACTION_USE),
        )
    )
    all_hydration = HydrationPolicyRegistry(
        (*hydration.contracts(), *publication_hydration)
    )
    publication_definitions = tuple(
        ObjectAdmissionDefinition(
            admission_type=admission_type,
            definition_version="admission-v1",
            object_class=object_class,
            allowed_use=allowed_use,
            security_scope=PUBLICATION_SECURITY_SCOPE,
            retention_scope=PUBLICATION_RETENTION_SCOPE,
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
                    SURFACE_ADMISSION_TYPE,
                    SURFACE_CLASS,
                    SURFACE_USE,
                    "authority.publication.surface.write",
                ),
                (
                    TRANSACTION_ADMISSION_TYPE,
                    TRANSACTION_CLASS,
                    TRANSACTION_USE,
                    "authority.publication.decide",
                ),
            ),
            publication_hydration,
            strict=True,
        )
    )
    all_admissions = ObjectAdmissionRegistry(
        (*admissions.definitions(), *publication_definitions),
        rights_policies=rights,
        hydration_policies=all_hydration,
    )
    contract = _command_contract()
    command = CommandDefinition(
        command_type=PUBLICATION_COMMAND,
        definition_version="publication-command-v1",
        aggregate_type="publication",
        event_type=PUBLICATION_EVENT,
        event_schema_version=1,
        payload_mode=PayloadMode.OBJECT_ADMISSION,
        payload_schema_version=contract.schema_version,
        payload_schema_contract_version=contract.contract_version,
        payload_schema_contract_digest=contract.contract_digest,
        payload_canonicalizer_version=contract.canonicalizer_implementation_version,
        trust_scope=TrustScope.ADMITTED,
        security_scope=PUBLICATION_SECURITY_SCOPE,
        retention_scope=PUBLICATION_RETENTION_SCOPE,
        required_scope="authority.publication.decide",
        required_object_class=TRANSACTION_CLASS,
        required_allowed_use=TRANSACTION_USE,
    )
    return (
        (rights, all_hydration, all_admissions),
        *base[1:7],
        publication_hydration,
        publication_definitions,
        command,
        CommandRegistry((*base[-2].definitions(), command)),
        PayloadSchemaRegistry((contract,)),
    )


def _open_system(path: Path):
    registries = _publication_registries()
    scopes = frozenset(
        {
            "authority.evidence.admit",
            "authority.editorial.decide",
            "authority.editorial.story.write",
            "authority.editorial.story.admit",
            "authority.publication.surface.write",
            "authority.publication.decide",
            "authority.objects.read",
            "authority.objects.manage",
            "authority.objects.lifecycle.write",
            "authority.fixture.events.read",
        }
    )
    system = open_object_system(
        path,
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
                "evidence_source": 1024 * 1024,
                "evidence_record": 1024 * 1024,
                "evidence_package": 1024 * 1024,
                "editorial_policy_decision": 1024 * 1024,
                "story_version": 1024 * 1024,
                SURFACE_CLASS: 1024 * 1024,
                TRANSACTION_CLASS: 1024 * 1024,
            },
            max_read_bytes=1024 * 1024,
            min_free_bytes=0,
            io_chunk_bytes=64,
            max_staging_bytes=1024 * 1024,
            max_range_bytes=1024 * 1024,
        ),
    )
    return system, registries


def _publication(system, evidence, editorial, registries):
    publication_hydration = registries[7]
    publication_definitions = registries[8]
    command = registries[9]
    return OfflinePublication(
        objects=system.objects,
        commands=system.commands,
        events=system.events,
        editorial=editorial,
        evidence=evidence,
        reader_principal_id="principal.alpha",
        authority_domain="newsroom.authority",
        controller_principal_id="principal.alpha",
        authorisation_policy_digest="sha256:" + "8" * 64,
        target_id="newsroom-app-serving-launch",
        target_policy_digest="sha256:" + "9" * 64,
        target_capabilities=LAUNCH_CAPABILITIES,
        surface_hydration_policy_digest=publication_hydration[0].contract_digest,
        transaction_hydration_policy_digest=publication_hydration[1].contract_digest,
        surface_admission_definition_digest=publication_definitions[0].digest,
        transaction_admission_definition_digest=publication_definitions[1].digest,
        command_definition_digest=command.digest,
    )


def _context(tmp_path: Path):
    candidate_connection, candidate_port, version = _candidate(tmp_path)
    ingress = open_evidence_intake_ingress(tmp_path / "intake.sqlite3")
    acknowledgement = _receive(
        ingress, candidate_connection, candidate_port, version, request_id="request-1"
    )
    system, registries = _open_system(tmp_path / "objects.sqlite3")
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
        candidate_port=candidate_port,
        source_admission_ids=(source.admission_id,),
        record_admission_ids=record_ids,
        proof=proof(),
    )
    decision_reference = _record_decision(
        system, _decision(retained, source.admission_id)
    )
    editorial = _native(system, evidence, registries)
    story_receipt, story = editorial.admit_story_version(
        StoryVersionRequest(AggregateId.new(), 0, "story-1"),
        package_admission_id=retained.package_admission_id,
        decision_reference=decision_reference,
        candidate_port=candidate_port,
        proof=proof(),
    )
    return (
        candidate_connection,
        candidate_port,
        ingress,
        system,
        registries,
        evidence,
        editorial,
        story_receipt,
        story,
    )


def test_auto_publish_atomically_records_bundle_decision_and_two_operations(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    connection, port, ingress, system, registries, evidence, editorial, story_receipt, story = context
    publication = _publication(system, evidence, editorial, registries)
    class ChangingRequest:
        reads = 0

        @property
        def publication_id(self):
            self.reads += 1
            return AggregateId.new()

    changing = ChangingRequest()
    with pytest.raises(PublicationError, match="exact PublicationRequest"):
        publication.decide(
            changing, story_receipt=story_receipt, candidate_port=port, proof=proof()
        )
    assert changing.reads == 0
    request = PublicationRequest(
        AggregateId.new(),
        0,
        "publication-1",
        "AUTO_PUBLISH",
        ("AUTO_POLICY_APPROVED",),
        "2026-09-08T12:03:00Z",
    )
    receipt, transaction = publication.decide(
        request, story_receipt=story_receipt, candidate_port=port, proof=proof()
    )
    replay_receipt, replay = publication.decide(
        request, story_receipt=story_receipt, candidate_port=port, proof=proof()
    )

    assert transaction.bundle is not None
    original_bytes = transaction.canonical_bytes()
    exposed = transaction.value()
    exposed["audit"]["outcome"] = "MUTATED"
    assert transaction.canonical_bytes() == original_bytes
    assert isinstance(transaction.audit, tuple)
    with pytest.raises(PublicationError, match="exact PublicationReceipt"):
        publication.read(
            object(), story_receipt=story_receipt, candidate_port=port, proof=proof()
        )
    assert transaction.decision.bundle_id == transaction.bundle.bundle_id
    assert tuple(item.surface_kind for item in transaction.operations) == (
        "ARTICLE",
        "FEED_CARD",
    )
    assert all(item.state == "PENDING" for item in transaction.operations)
    assert transaction.bundle.story_version_digest == story.digest
    assert all(
        link[0] in {item[2] for item in transaction.bundle.surface_payloads}
        for link in transaction.bundle.claim_surface_manifest
    )
    rendered = tuple(
        json.loads(
            system.objects.hydrate(
                HydrationRequest(admission_id, SURFACE_PURPOSE), proof=proof()
            ).data
        )
        for _, admission_id, _, _ in transaction.bundle.surface_payloads
    )
    article, feed_card = rendered
    assert article["headline"] == story.copy.title.removeprefix("【未出版】")
    assert article["body"] == story.copy.body
    assert article["source_references"] == [
        ["source-1", "https://example.test/current"]
    ]
    assert article["claim_links"]
    for payload in rendered:
        for claim_id, field, assertion, materiality in payload["claim_links"]:
            assert materiality == "MATERIAL"
            assert payload[field.lower()].count(assertion) == 1
            assert [
                payload["payload_id"], claim_id, field, assertion, materiality
            ] in [list(item) for item in transaction.bundle.claim_surface_manifest]
    assert feed_card["body"] == ""
    assert all(link[1] == "HEADLINE" for link in feed_card["claim_links"])
    assert replay_receipt == receipt
    assert replay == transaction
    assert publication.read(
        receipt, story_receipt=story_receipt, candidate_port=port, proof=proof()
    ) == transaction
    before = tuple(
        event for event in system.events.after(0, limit=1000, proof=proof())
        if event.event_type == PUBLICATION_EVENT
    )
    with pytest.raises(ExpectedVersionConflict):
        publication.decide(
            replace(
                request, publication_id=AggregateId.new(),
                expected_aggregate_version=1, idempotency_key="wrong-version",
            ),
            story_receipt=story_receipt, candidate_port=port, proof=proof(),
        )
    assert tuple(
        event for event in system.events.after(0, limit=1000, proof=proof())
        if event.event_type == PUBLICATION_EVENT
    ) == before
    connection.rollback()
    system.close()

    reopened_system, reopened_registries = _open_system(tmp_path / "objects.sqlite3")
    reopened_evidence = _evidence_facade(
        reopened_system, ingress, reopened_registries
    )
    reopened_editorial = _native(
        reopened_system, reopened_evidence, reopened_registries
    )
    reopened_publication = _publication(
        reopened_system, reopened_evidence, reopened_editorial, reopened_registries
    )
    connection.execute("BEGIN IMMEDIATE")
    assert reopened_publication.read(
        receipt, story_receipt=story_receipt, candidate_port=port, proof=proof()
    ) == transaction
    with pytest.raises(PublicationError, match="publication replay outcome differs"):
        reopened_publication.decide(
            replace(
                request,
                outcome="HOLD_FOR_REVIEW",
                reason_codes=("REVIEW_REQUIRED",),
            ),
            story_receipt=story_receipt,
            candidate_port=port,
            proof=proof(),
        )
    with pytest.raises(PublicationError, match="authority event differs"):
        reopened_publication.read(
            replace(receipt, transaction_digest="sha256:" + "0" * 64),
            story_receipt=story_receipt,
            candidate_port=port,
            proof=proof(),
        )
    connection.rollback()
    reopened_system.close()
    ingress.close()
    connection.close()


def test_hold_commits_only_decision_and_audit(tmp_path: Path, monkeypatch) -> None:
    context = _context(tmp_path)
    connection, port, ingress, system, registries, evidence, editorial, story_receipt, _ = context
    publication = _publication(system, evidence, editorial, registries)
    before_story_events = tuple(
        item for item in system.events.after(0, limit=1000, proof=proof())
        if item.event_type == STORY_EVENT
    )
    receipt, transaction = publication.decide(
        PublicationRequest(
            AggregateId.new(),
            0,
            "publication-hold",
            "HOLD_FOR_REVIEW",
            ("AUTHORISATION_POLICY_MISSING",),
            "2026-09-08T12:03:00Z",
        ),
        story_receipt=story_receipt,
        candidate_port=port,
        proof=proof(),
    )

    assert transaction.bundle is None
    assert transaction.operations == ()
    assert transaction.decision.bundle_id is None
    assert publication.read(
        receipt, story_receipt=story_receipt, candidate_port=port, proof=proof()
    ) == transaction
    assert tuple(
        item for item in system.events.after(0, limit=1000, proof=proof())
        if item.event_type == STORY_EVENT
    ) == before_story_events
    def unexpected_effect(*args, **kwargs):
        pytest.fail("changed-outcome replay must not admit a surface or execute a command")

    monkeypatch.setattr(publication, "_admit_surface", unexpected_effect)
    monkeypatch.setattr(type(system.commands), "execute", unexpected_effect)
    monkeypatch.setattr(editorial, "_story_writer", unexpected_effect)
    with pytest.raises(PublicationError, match="publication replay outcome differs"):
        publication.decide(
            PublicationRequest(
                receipt.publication_id, 0, "publication-hold", "AUTO_PUBLISH",
                ("AUTHORISATION_POLICY_MISSING",), "2026-09-08T12:03:00Z",
            ),
            story_receipt=story_receipt, candidate_port=port, proof=proof(),
        )
    assert publication.read(
        receipt, story_receipt=story_receipt, candidate_port=port, proof=proof()
    ) == transaction
    connection.rollback()
    system.close()
    ingress.close()
    connection.close()


def test_generic_admitted_event_does_not_expose_an_unvalidated_operation(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    connection, port, ingress, system, registries, evidence, editorial, story_receipt, _ = context
    publication = _publication(system, evidence, editorial, registries)
    raw = canonical_json_bytes({"opaque": "not-a-publication-transaction"})
    admission = system.objects.admit(
        ObjectAdmissionRequest(TRANSACTION_ADMISSION_TYPE, "opaque-transaction"),
        raw,
        proof=proof(),
    ).admission
    publication_id = AggregateId.new()
    committed = system.commands.execute(
        SemanticCommand(
            PUBLICATION_COMMAND,
            publication_id,
            0,
            ObjectAdmissionPayload(admission.admission_id),
            "opaque-command",
        ),
        proof=proof(),
    )
    receipt = PublicationReceipt(
        committed.command_id,
        committed.event_id,
        publication_id,
        committed.aggregate_version,
        admission.admission_id,
        admission.blob.blob_digest,
        "sha256:" + "1" * 64,
        "sha256:" + "2" * 64,
        None,
        (),
    )

    with pytest.raises(PublicationError, match="transaction fields differ"):
        publication.read(
            receipt,
            story_receipt=story_receipt,
            candidate_port=port,
            proof=proof(),
        )

    connection.rollback()
    system.close()
    ingress.close()
    connection.close()
