from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from newsroom.authority._capability import _CapabilityIssuer
from newsroom.authority._check_boundary import _CheckBoundary
from newsroom.authority._check_store import _CheckAuthorityStore
from newsroom.authority._check_facade import GovernedChecks
from newsroom.authority._proposal_admission import (
    _ProposalAdmissionBoundary,
)
from newsroom.authority._source_registry_system import (
    _SourceRegistryBoundary,
    GovernedSources,
)
from newsroom.authority.policy import CommandRegistry, PayloadSchemaRegistry
from newsroom.authority.service import CommandService
from newsroom.authority.types import UtcTimestamp
from newsroom.checks.policy import merge_discovery_check_authority_registries
from newsroom.checks.read_policy import DiscoveryCheckReadPolicy
from newsroom.sources import SourceRegistryReadPolicy
from newsroom.sources.policy import merge_source_registry_authority_registries


class GovernedCheckAuthoritySystem:
    __slots__ = ("sources", "checks", "__close")

    def __init__(
        self,
        *,
        sources: GovernedSources,
        checks: GovernedChecks,
        close: Callable[[], None],
    ) -> None:
        self.sources = sources
        self.checks = checks
        self.__close = close

    def close(self) -> None:
        self.__close()

    def __enter__(self) -> "GovernedCheckAuthoritySystem":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        tb: object,
    ) -> None:
        self.close()


def open_governed_check_authority_system(
    *,
    path: Path,
    registry: CommandRegistry,
    payload_schemas: PayloadSchemaRegistry,
    authenticator: Any,
    authorizer: Any,
    source_read_policy: SourceRegistryReadPolicy,
    check_read_policy: DiscoveryCheckReadPolicy,
    command_service_version: str = "authority-command-v1",
    busy_timeout_ms: int = 5_000,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
) -> GovernedCheckAuthoritySystem:
    source_registry, source_schemas = (
        merge_source_registry_authority_registries(
            command_registry=registry,
            payload_schemas=payload_schemas,
        )
    )
    merged_registry, merged_schemas = (
        merge_discovery_check_authority_registries(
            command_registry=source_registry,
            payload_schemas=source_schemas,
        )
    )
    issuer = _CapabilityIssuer(
        command_registry=merged_registry,
        payload_schemas=merged_schemas,
    )
    store: _CheckAuthorityStore | None = None
    try:
        store = _CheckAuthorityStore(
            path,
            issuer=issuer,
            command_registry=merged_registry,
            payload_schemas=merged_schemas,
            command_service_version=command_service_version,
            busy_timeout_ms=busy_timeout_ms,
            clock=clock,
        )
        command_service = CommandService(
            registry=merged_registry,
            payload_schemas=merged_schemas,
            authenticator=authenticator,
            authorizer=authorizer,
            committed_lookup=store,
            clock=clock,
            _issuer=issuer,
        )
        source_boundary = _SourceRegistryBoundary(
            store=store,
            command_service=command_service,
            authenticator=authenticator,
            authorizer=authorizer,
            read_policy=source_read_policy,
            clock=clock,
        )
        check_boundary = _CheckBoundary(
            store=store,
            command_service=command_service,
            authenticator=authenticator,
            authorizer=authorizer,
            read_policy=check_read_policy,
            clock=clock,
        )
        proposal_admission = _ProposalAdmissionBoundary(
            store=store,
            command_service=command_service,
        )
        closed = False

        def close() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            assert store is not None
            store.close()

        return GovernedCheckAuthoritySystem(
            sources=GovernedSources(
                register_definition=source_boundary.register_definition,
                record_version=source_boundary.record_version,
                register_item=source_boundary.register_item,
                decide_locator=source_boundary.decide_locator,
                record_revision=source_boundary.record_revision,
                record_representation=source_boundary.record_representation,
                record_occurrence=source_boundary.record_occurrence,
                definition=source_boundary.definition,
                current_summary=source_boundary.current_summary,
                version_details=source_boundary.version_details,
                item=source_boundary.item,
                revision=source_boundary.revision,
                latest_revision=source_boundary.latest_revision,
                representation=source_boundary.representation,
                occurrences=source_boundary.occurrences,
            ),
            checks=GovernedChecks(
                register_request=check_boundary.register_request,
                start_attempt=check_boundary.start_attempt,
                record_outcome=check_boundary.record_outcome,
                decide_baseline=check_boundary.decide_baseline,
                record_transition=check_boundary.record_transition,
                open_finding=check_boundary.open_finding,
                record_finding_occurrence=(
                    check_boundary.record_finding_occurrence
                ),
                admit_proposal=proposal_admission.admit,
                request=check_boundary.request,
                attempt=check_boundary.attempt,
                outcome=check_boundary.outcome,
                observed_prior_revision=check_boundary.observed_prior_revision,
                attempts=check_boundary.attempts,
                outcomes=check_boundary.outcomes,
                baseline=check_boundary.baseline,
                current_baseline=check_boundary.current_baseline,
                transition=check_boundary.transition,
                finding=check_boundary.finding,
                finding_occurrences=check_boundary.finding_occurrences,
            ),
            close=close,
        )
    except Exception:
        if store is not None:
            store.close()
        raise


__all__ = [
    "GovernedCheckAuthoritySystem",
    "GovernedChecks",
    "open_governed_check_authority_system",
]
