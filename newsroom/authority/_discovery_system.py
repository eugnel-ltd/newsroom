from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from newsroom.authority._capability import _CapabilityIssuer
from newsroom.authority._check_boundary import _CheckBoundary
from newsroom.authority._check_facade import GovernedChecks
from newsroom.authority._discovery_boundary import _DiscoveryBoundary
from newsroom.authority._discovery_facade import GovernedDiscovery
from newsroom.authority._discovery_store import _DiscoveryAuthorityStore
from newsroom.authority._proposal_admission import _ProposalAdmissionBoundary
from newsroom.authority._signal_lead_admission import _SignalLeadAdmissionBoundary
from newsroom.authority._source_registry_system import (
    _SourceRegistryBoundary,
    GovernedSources,
)
from newsroom.authority.policy import CommandRegistry, PayloadSchemaRegistry
from newsroom.authority.service import CommandService
from newsroom.authority.types import UtcTimestamp
from newsroom.checks.policy import merge_discovery_check_authority_registries
from newsroom.checks.read_policy import DiscoveryCheckReadPolicy
from newsroom.discovery.policy import merge_discovery_signal_lead_registries
from newsroom.discovery.types import DiscoveryReadPolicy
from newsroom.sources import SourceRegistryReadPolicy
from newsroom.sources.policy import merge_source_registry_authority_registries


class GovernedDiscoveryAuthoritySystem:
    __slots__ = ("sources", "checks", "discovery", "__close")

    def __init__(
        self,
        *,
        sources: GovernedSources,
        checks: GovernedChecks,
        discovery: GovernedDiscovery,
        close: Callable[[], None],
    ) -> None:
        self.sources = sources
        self.checks = checks
        self.discovery = discovery
        self.__close = close

    def close(self) -> None:
        self.__close()

    def __enter__(self) -> "GovernedDiscoveryAuthoritySystem":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def open_governed_discovery_authority_system(
    *,
    path: Path,
    registry: CommandRegistry,
    payload_schemas: PayloadSchemaRegistry,
    authenticator: Any,
    authorizer: Any,
    source_read_policy: SourceRegistryReadPolicy,
    check_read_policy: DiscoveryCheckReadPolicy,
    discovery_read_policy: DiscoveryReadPolicy,
    command_service_version: str = "authority-command-v1",
    busy_timeout_ms: int = 5_000,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
) -> GovernedDiscoveryAuthoritySystem:
    source_registry, source_schemas = merge_source_registry_authority_registries(
        command_registry=registry,
        payload_schemas=payload_schemas,
    )
    check_registry, check_schemas = merge_discovery_check_authority_registries(
        command_registry=source_registry,
        payload_schemas=source_schemas,
    )
    merged_registry, merged_schemas = merge_discovery_signal_lead_registries(
        command_registry=check_registry,
        payload_schemas=check_schemas,
    )
    issuer = _CapabilityIssuer(
        command_registry=merged_registry,
        payload_schemas=merged_schemas,
    )
    store: _DiscoveryAuthorityStore | None = None
    try:
        store = _DiscoveryAuthorityStore(
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
        discovery_boundary = _DiscoveryBoundary(
            store=store,
            command_service=command_service,
            authenticator=authenticator,
            authorizer=authorizer,
            read_policy=discovery_read_policy,
            clock=clock,
        )
        signal_lead_admission = _SignalLeadAdmissionBoundary(
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

        return GovernedDiscoveryAuthoritySystem(
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
                record_finding_occurrence=check_boundary.record_finding_occurrence,
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
            discovery=GovernedDiscovery(
                admit_signal=discovery_boundary.admit_signal,
                decide_gate=discovery_boundary.decide_gate,
                open_lead=discovery_boundary.open_lead,
                record_watch_condition=discovery_boundary.record_watch_condition,
                record_lead_disposition=discovery_boundary.record_lead_disposition,
                admit_signal_to_lead=signal_lead_admission.admit,
                signal=discovery_boundary.signal,
                gate=discovery_boundary.gate,
                current_gate=discovery_boundary.current_gate,
                gates=discovery_boundary.gates,
                lead=discovery_boundary.lead,
                lead_for_signal=discovery_boundary.lead_for_signal,
                watch_condition=discovery_boundary.watch_condition,
                disposition=discovery_boundary.disposition,
                current_disposition=discovery_boundary.current_disposition,
                latest_disposition=discovery_boundary.latest_disposition,
                dispositions=discovery_boundary.dispositions,
                signals_for_revision=discovery_boundary.signals_for_revision,
                current_status=discovery_boundary.current_status,
            ),
            close=close,
        )
    except Exception:
        if store is not None:
            store.close()
        raise


__all__ = [
    "GovernedDiscovery",
    "GovernedDiscoveryAuthoritySystem",
    "open_governed_discovery_authority_system",
]
