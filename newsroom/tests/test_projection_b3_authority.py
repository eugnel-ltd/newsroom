from __future__ import annotations

from pathlib import Path
import json
import sqlite3

import pytest

from newsroom.authority import (
    AggregateId,
    InlinePayload,
    SemanticCommand,
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.projection import (
    ProjectionDeliveryOutcome,
    ProjectionDeliveryRequest,
    ProjectionFamilyRegistrationRequest,
    ProjectionGenerationCreateRequest,
    ProjectionGenerationId,
    ProjectionGenerationPromotionRequest,
    ProjectionGenerationState,
    ProjectionGenerationTransitionRequest,
    ProjectionGenerationValidationRequest,
    ProjectionStateError,
    ProjectionContractError,
)

from .projection_b1_helpers import FAMILY_ID, open_projection_system, proof


SERVICE_DIGEST = digest_canonical(
    {
        "neo4j_server": "2026.06.0",
        "neo4j_edition": "community",
        "driver": "6.2.0",
    }
)
GRAPH_DIGEST = digest_canonical({"generation": "exact", "state": "empty"})


def _register(system) -> None:
    system.projections.register_family(
        ProjectionFamilyRegistrationRequest(FAMILY_ID, "family-register-b3"),
        proof=proof(),
    )


def _create(system, key: str):
    return system.projections.create_generation(
        ProjectionGenerationCreateRequest(
            ProjectionGenerationId.new(),
            FAMILY_ID,
            "B3_BUILD",
            key,
        ),
        proof=proof(),
    )


def _validate(system, generation, key: str, *, graph_digest: str = GRAPH_DIGEST):
    return system.projections.validate_generation(
        ProjectionGenerationValidationRequest(
            generation.generation_id,
            generation.authority_aggregate_version,
            0,
            SERVICE_DIGEST,
            graph_digest,
            "B3_VALIDATE",
            key,
        ),
        proof=proof(),
    )


def _promote(
    system,
    generation,
    validation,
    key: str,
    *,
    prior=None,
):
    return system.projections.promote_generation(
        ProjectionGenerationPromotionRequest(
            generation.generation_id,
            generation.authority_aggregate_version,
            validation.checkpoint_ledger_seq,
            validation.validation_digest,
            "B3_PROMOTE",
            key,
            prior_generation_id=(
                None if prior is None else prior.generation_id
            ),
            expected_prior_authority_version=(
                None if prior is None else prior.authority_aggregate_version
            ),
        ),
        proof=proof(),
    )


def test_direct_active_transition_cannot_bypass_authority_promotion(
    tmp_path: Path,
) -> None:
    system = open_projection_system(tmp_path / "authority.sqlite3")
    try:
        _register(system)
        created = _create(system, "generation-create-direct-active")
        validating = system.projections.transition_generation(
            ProjectionGenerationTransitionRequest(
                created.generation_id,
                created.authority_aggregate_version,
                ProjectionGenerationState.VALIDATING,
                "LEGACY_VALIDATING",
                "generation-legacy-validating",
                validated_through_ledger_seq=0,
            ),
            proof=proof(),
        )
        before_events = system.events.after(0, limit=100, proof=proof())

        with pytest.raises(ProjectionStateError, match="authority promotion"):
            system.projections.transition_generation(
                ProjectionGenerationTransitionRequest(
                    validating.generation_id,
                    validating.authority_aggregate_version,
                    ProjectionGenerationState.ACTIVE,
                    "DIRECT_ACTIVE_FORBIDDEN",
                    "generation-direct-active-forbidden",
                    validated_through_ledger_seq=0,
                ),
                proof=proof(),
            )

        assert system.events.after(0, limit=100, proof=proof()) == before_events
        after_rejection = next(
            item
            for item in system.projections.generations(FAMILY_ID, proof=proof())
            if item.generation_id == created.generation_id
        )
        assert after_rejection.state is ProjectionGenerationState.VALIDATING
        assert after_rejection.authority_aggregate_version == (
            validating.authority_aggregate_version
        )

        validation = _validate(
            system,
            after_rejection,
            "generation-validate-after-direct-active",
        )
        validated = next(
            item
            for item in system.projections.generations(FAMILY_ID, proof=proof())
            if item.generation_id == created.generation_id
        )
        promotion = _promote(
            system,
            validated,
            validation,
            "generation-promote-after-direct-active",
        )
        assert promotion.generation.state is ProjectionGenerationState.ACTIVE
    finally:
        system.close()


def test_validation_evidence_is_exact_typed_and_replayable(tmp_path: Path) -> None:
    system = open_projection_system(tmp_path / "authority.sqlite3")
    try:
        _register(system)
        created = _create(system, "generation-create-validation")
        request = ProjectionGenerationValidationRequest(
            created.generation_id,
            created.authority_aggregate_version,
            0,
            SERVICE_DIGEST,
            GRAPH_DIGEST,
            "B3_VALIDATE",
            "generation-validate-replay",
        )
        validation = system.projections.validate_generation(request, proof=proof())
        replay = system.projections.validate_generation(request, proof=proof())

        assert replay == validation
        assert validation.checkpoint_ledger_seq == 0
        assert validation.service_compatibility_digest == SERVICE_DIGEST
        assert validation.projection_state_digest == GRAPH_DIGEST
        assert validation.lifecycle_version == 2
        current = system.projections.generations(FAMILY_ID, proof=proof())[0]
        assert current.state is ProjectionGenerationState.VALIDATING
        assert current.validated_through_ledger_seq == 0
        assert system.projections.validation(
            created.generation_id, proof=proof()
        ) == validation
        assert validation.source_snapshot_digest is None
    finally:
        system.close()


def test_source_bound_validation_is_exact_replayable_and_lookupable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.sqlite3"
    snapshot = digest_canonical({"source": "snapshot"})
    request_digest = digest_canonical({"source": "request"})
    system = open_projection_system(path)
    try:
        _register(system)
        created = _create(system, "generation-create-source-bound")
        request = ProjectionGenerationValidationRequest(
            created.generation_id,
            created.authority_aggregate_version,
            0,
            SERVICE_DIGEST,
            GRAPH_DIGEST,
            "B3_VALIDATE_CURRENT",
            "generation-validate-current",
            snapshot,
            0,
            request_digest,
        )
        with pytest.raises(ProjectionStateError, match="differs from required"):
            system.projections.validate_generation(
                request, proof=proof(), required_source_ledger_seq=1
            )
        validation = system.projections.validate_generation(request, proof=proof())
        assert system.projections.validate_generation(
            request, proof=proof(), required_source_ledger_seq=0
        ) == validation
        assert validation.source_snapshot_digest == snapshot
        assert validation.source_watermark_ledger_seq == 0
        assert validation.source_request_digest == request_digest
        boundary = system.projections._NativeProjections__validation.__self__
        namespace = boundary.validation_idempotency_namespace(proof())
        assert boundary._store.projection_generation_validation_for_key(
            namespace, request.idempotency_key
        ) == validation
        current = system.projections.generations(FAMILY_ID, proof=proof())[0]
        promoted = _promote(
            system, current, validation, "generation-promote-source-bound"
        ).generation
        active_request = ProjectionGenerationValidationRequest(
            promoted.generation_id,
            promoted.authority_aggregate_version,
            0,
            SERVICE_DIGEST,
            GRAPH_DIGEST,
            "B3_VALIDATE_CURRENT_ACTIVE",
            "generation-validate-current-active",
            snapshot,
            0,
            digest_canonical({"source": "active-request"}),
        )
        active_validation = system.projections.validate_generation(
            active_request, proof=proof()
        )
        assert system.projections.generations(FAMILY_ID, proof=proof())[0].state is (
            ProjectionGenerationState.ACTIVE
        )
    finally:
        system.close()

    reopened = open_projection_system(path)
    try:
        boundary = reopened.projections._NativeProjections__validation.__self__
        namespace = boundary.validation_idempotency_namespace(proof())
        assert boundary._store.projection_generation_validation_for_key(
            namespace, request.idempotency_key
        ) == validation
        assert boundary._store.projection_generation_validation_for_key(
            namespace, active_request.idempotency_key
        ) == active_validation
    finally:
        reopened.close()


def test_validation_source_binding_is_all_or_none_and_within_checkpoint() -> None:
    generation_id = ProjectionGenerationId.new()
    digest = digest_canonical({"source": "binding"})
    common = (
        generation_id,
        1,
        0,
        SERVICE_DIGEST,
        GRAPH_DIGEST,
        "B3_VALIDATE_CURRENT",
        "generation-validate-current-invalid",
    )
    with pytest.raises(ProjectionContractError, match="complete"):
        ProjectionGenerationValidationRequest(*common, digest, None, digest)
    with pytest.raises(ProjectionContractError, match="exceeds checkpoint"):
        ProjectionGenerationValidationRequest(*common, digest, 1, digest)
    for invalid_watermark in (True, "0", 0.0):
        with pytest.raises(ProjectionContractError):
            ProjectionGenerationValidationRequest(
                *common, digest, invalid_watermark, digest
            )


def test_source_bound_validation_reopen_rejects_self_consistent_command_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.sqlite3"
    digest = digest_canonical({"source": "original"})
    system = open_projection_system(path)
    try:
        _register(system)
        created = _create(system, "generation-create-source-tamper")
        system.projections.validate_generation(
            ProjectionGenerationValidationRequest(
                created.generation_id,
                created.authority_aggregate_version,
                0,
                SERVICE_DIGEST,
                GRAPH_DIGEST,
                "B3_VALIDATE_CURRENT",
                "generation-validate-current-tamper",
                digest,
                0,
                digest,
            ),
            proof=proof(),
        )
    finally:
        system.close()

    conn = sqlite3.connect(path)
    try:
        trigger = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND "
            "name='immutable_projection_generation_validation_update'"
        ).fetchone()[0]
        row = conn.execute(
            "SELECT validation_digest,canonical_bytes "
            "FROM projection_generation_validations"
        ).fetchone()
        value = json.loads(row[1])
        value["source_request_digest"] = digest_canonical({"source": "other"})
        canonical = canonical_json_bytes(value)
        conn.execute("DROP TRIGGER immutable_projection_generation_validation_update")
        conn.execute(
            "UPDATE projection_generation_validations SET validation_digest=?,"
            "canonical_bytes=? WHERE validation_digest=?",
            (digest_bytes(canonical), canonical, row[0]),
        )
        conn.execute(trigger)
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(Exception, match="command binding is inconsistent"):
        open_projection_system(path)


def test_source_bound_lookup_recomputes_validation_digest(tmp_path: Path) -> None:
    system = open_projection_system(tmp_path / "authority.sqlite3")
    try:
        _register(system)
        created = _create(system, "generation-create-source-digest-tamper")
        digest = digest_canonical({"source": "lookup"})
        request = ProjectionGenerationValidationRequest(
            created.generation_id,
            created.authority_aggregate_version,
            0,
            SERVICE_DIGEST,
            GRAPH_DIGEST,
            "B3_VALIDATE_CURRENT",
            "generation-validate-current-digest-tamper",
            digest,
            0,
            digest,
        )
        system.projections.validate_generation(request, proof=proof())
        boundary = system.projections._NativeProjections__validation.__self__
        conn = boundary._store._connection
        trigger = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND "
            "name='immutable_projection_generation_validation_update'"
        ).fetchone()[0]
        conn.execute("DROP TRIGGER immutable_projection_generation_validation_update")
        conn.execute(
            "UPDATE projection_generation_validations SET validation_digest=?",
            ("sha256:" + "0" * 64,),
        )
        conn.execute(trigger)
        conn.commit()
        namespace = boundary.validation_idempotency_namespace(proof())
        with pytest.raises(Exception, match="validation digest is inconsistent"):
            boundary._store.projection_generation_validation_for_key(
                namespace, request.idempotency_key
            )
    finally:
        system.close()


def test_first_promotion_is_authoritative_and_replayable(tmp_path: Path) -> None:
    system = open_projection_system(tmp_path / "authority.sqlite3")
    try:
        _register(system)
        created = _create(system, "generation-create-first-promotion")
        validation = _validate(system, created, "generation-validate-first")
        validating = system.projections.generations(FAMILY_ID, proof=proof())[0]
        request = ProjectionGenerationPromotionRequest(
            validating.generation_id,
            validating.authority_aggregate_version,
            0,
            validation.validation_digest,
            "B3_PROMOTE",
            "generation-promote-first",
        )
        promotion = system.projections.promote_generation(request, proof=proof())
        replay = system.projections.promote_generation(request, proof=proof())

        assert replay == promotion
        assert promotion.generation.state is ProjectionGenerationState.ACTIVE
        assert promotion.prior_generation is None
        assert promotion.generation.validated_through_ledger_seq == 0
        status = system.projections.status(FAMILY_ID, proof=proof())
        assert status.generation_id == created.generation_id
        assert status.generation_state is ProjectionGenerationState.ACTIVE
        assert system.projections.promotions(FAMILY_ID, proof=proof()) == (
            promotion,
        )
    finally:
        system.close()


def test_atomic_promotion_retires_prior_active_generation(tmp_path: Path) -> None:
    system = open_projection_system(tmp_path / "authority.sqlite3")
    try:
        _register(system)
        first_created = _create(system, "generation-create-prior")
        first_validation = _validate(
            system, first_created, "generation-validate-prior"
        )
        first_validating = next(
            item
            for item in system.projections.generations(FAMILY_ID, proof=proof())
            if item.generation_id == first_created.generation_id
        )
        first_promotion = _promote(
            system,
            first_validating,
            first_validation,
            "generation-promote-prior",
        )

        second_created = _create(system, "generation-create-replacement")
        second_validation = _validate(
            system, second_created, "generation-validate-replacement"
        )
        second_validating = next(
            item
            for item in system.projections.generations(FAMILY_ID, proof=proof())
            if item.generation_id == second_created.generation_id
        )
        request = ProjectionGenerationPromotionRequest(
            second_validating.generation_id,
            second_validating.authority_aggregate_version,
            0,
            second_validation.validation_digest,
            "B3_REPLACE",
            "generation-promote-replacement",
            prior_generation_id=first_promotion.generation.generation_id,
            expected_prior_authority_version=(
                first_promotion.generation.authority_aggregate_version
            ),
        )
        replacement = system.projections.promote_generation(request, proof=proof())
        replay = system.projections.promote_generation(request, proof=proof())

        assert replay == replacement
        assert replacement.generation.state is ProjectionGenerationState.ACTIVE
        assert replacement.prior_generation is not None
        assert replacement.prior_generation.state is ProjectionGenerationState.RETIRED
        generations = {
            item.generation_id: item
            for item in system.projections.generations(FAMILY_ID, proof=proof())
        }
        assert generations[second_created.generation_id].state is (
            ProjectionGenerationState.ACTIVE
        )
        assert generations[first_created.generation_id].state is (
            ProjectionGenerationState.RETIRED
        )
        assert system.projections.status(FAMILY_ID, proof=proof()).generation_id == (
            second_created.generation_id
        )
    finally:
        system.close()


def test_validation_and_promotion_require_exact_current_checkpoint(
    tmp_path: Path,
) -> None:
    system = open_projection_system(tmp_path / "authority.sqlite3")
    try:
        _register(system)
        created = _create(system, "generation-create-current-checkpoint")
        source_result = system.commands.execute(
            SemanticCommand(
                command_type="source.item.write",
                aggregate_id=AggregateId.new(),
                expected_aggregate_version=0,
                payload=InlinePayload({"headline": "B3 checkpoint", "count": 1}),
                idempotency_key="b3-current-checkpoint-source",
            ),
            proof=proof(),
        )
        source = next(
            event
            for event in system.events.after(0, limit=100, proof=proof())
            if event.command_id == source_result.command_id
        )
        current = next(
            item
            for item in system.projections.generations(FAMILY_ID, proof=proof())
            if item.generation_id == created.generation_id
        )
        system.projections.record_delivery(
            ProjectionDeliveryRequest(
                created.generation_id,
                current.authority_aggregate_version,
                source.ledger_seq,
                ProjectionDeliveryOutcome.APPLIED,
                "b3-current-checkpoint-delivery",
            ),
            proof=proof(),
        )
        checkpoint = system.projections.status(
            FAMILY_ID, proof=proof()
        ).contiguous_ledger_seq
        assert checkpoint > 0
        current = next(
            item
            for item in system.projections.generations(FAMILY_ID, proof=proof())
            if item.generation_id == created.generation_id
        )
        before = system.events.after(0, limit=1000, proof=proof())
        with pytest.raises(ProjectionStateError, match="exact current contiguous checkpoint"):
            system.projections.validate_generation(
                ProjectionGenerationValidationRequest(
                    current.generation_id,
                    current.authority_aggregate_version,
                    checkpoint - 1,
                    SERVICE_DIGEST,
                    GRAPH_DIGEST,
                    "B3_PARTIAL_VALIDATE",
                    "generation-partial-validation",
                ),
                proof=proof(),
            )
        assert system.events.after(0, limit=1000, proof=proof()) == before

        validation = system.projections.validate_generation(
            ProjectionGenerationValidationRequest(
                current.generation_id,
                current.authority_aggregate_version,
                checkpoint,
                SERVICE_DIGEST,
                GRAPH_DIGEST,
                "B3_COMPLETE_VALIDATE",
                "generation-complete-validation",
            ),
            proof=proof(),
        )
        validating = next(
            item
            for item in system.projections.generations(FAMILY_ID, proof=proof())
            if item.generation_id == created.generation_id
        )
        promotion = _promote(
            system,
            validating,
            validation,
            "generation-promote-current-checkpoint",
        )
        assert promotion.generation.state is ProjectionGenerationState.ACTIVE
        assert promotion.generation.validated_through_ledger_seq == checkpoint
    finally:
        system.close()


def test_promotion_rejects_stale_validation_after_checkpoint_advances(
    tmp_path: Path,
) -> None:
    system = open_projection_system(tmp_path / "authority.sqlite3")
    try:
        _register(system)
        created = _create(system, "generation-create-stale")
        validation = _validate(system, created, "generation-validate-stale")
        validating = system.projections.generations(FAMILY_ID, proof=proof())[0]
        source_result = system.commands.execute(
            SemanticCommand(
                command_type="source.item.write",
                aggregate_id=AggregateId.new(),
                expected_aggregate_version=0,
                payload=InlinePayload({"headline": "B3", "count": 1}),
                idempotency_key="b3-stale-source",
            ),
            proof=proof(),
        )
        source = next(
            event
            for event in system.events.after(0, limit=100, proof=proof())
            if event.command_id == source_result.command_id
        )
        delivered = system.projections.record_delivery(
            ProjectionDeliveryRequest(
                validating.generation_id,
                validating.authority_aggregate_version,
                source.ledger_seq,
                ProjectionDeliveryOutcome.APPLIED,
                "b3-stale-delivery",
            ),
            proof=proof(),
        )
        current = system.projections.generations(FAMILY_ID, proof=proof())[0]
        assert delivered.authority_event_id
        assert current.authority_aggregate_version > validating.authority_aggregate_version
        assert system.projections.status(FAMILY_ID, proof=proof()).contiguous_ledger_seq > 0

        with pytest.raises(ProjectionStateError, match="stale"):
            system.projections.promote_generation(
                ProjectionGenerationPromotionRequest(
                    current.generation_id,
                    current.authority_aggregate_version,
                    validation.checkpoint_ledger_seq,
                    validation.validation_digest,
                    "B3_PROMOTE_STALE",
                    "generation-promote-stale",
                ),
                proof=proof(),
            )
    finally:
        system.close()


def test_validation_and_promotion_rows_are_immutable_and_reopen_checked(
    tmp_path: Path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    system = open_projection_system(database)
    try:
        _register(system)
        created = _create(system, "generation-create-integrity")
        validation = _validate(system, created, "generation-validate-integrity")
        validating = system.projections.generations(FAMILY_ID, proof=proof())[0]
        _promote(
            system,
            validating,
            validation,
            "generation-promote-integrity",
        )
    finally:
        system.close()

    conn = sqlite3.connect(database)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE projection_generation_validations "
                "SET projector_version='tampered'"
            )
        trigger = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' "
            "AND name='immutable_projection_generation_promotion_update'"
        ).fetchone()[0]
        conn.execute("DROP TRIGGER immutable_projection_generation_promotion_update")
        conn.execute(
            "UPDATE projection_generation_promotions "
            "SET checkpoint_ledger_seq=checkpoint_ledger_seq+1"
        )
        conn.execute(str(trigger))
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(Exception, match="promotion digest is inconsistent"):
        open_projection_system(database)
