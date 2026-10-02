from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Iterable

from newsroom.authority.canonical import canonical_json_bytes, digest_canonical
from newsroom.authority.persistence import LedgerEventRecord, RetiredLedgerEventRecord
from newsroom.authority.types import TrustScope
from newsroom.entities.models import (
    CanonicalEntity,
    CanonicalEntityVersion,
    EntityAlias,
    EntityPreferredIdentity,
    EntityProjectionEvent,
)
from newsroom.entities.types import EntityProjectionAction
from newsroom.relations.editorial_models import (
    CanonicalEntityRelationEndpoint,
    EditorialRelationCurrentView,
    EditorialRelationProjectionEvent,
    RelationAssertionRelationEndpoint,
)
from newsroom.relations.editorial_types import (
    EditorialRelationAssertionLifecycle,
    EditorialRelationProjectionAction,
)


class Increment4ProofContractError(ValueError):
    """Raised when Increment 4 proof inputs are not exact admitted authority."""


def _event_digest(event: LedgerEventRecord | RetiredLedgerEventRecord) -> str:
    if isinstance(event, RetiredLedgerEventRecord):
        return event.original_header_digest
    if not isinstance(event, LedgerEventRecord):
        raise Increment4ProofContractError("proof event must be a retained ledger event")
    return digest_canonical(asdict(event))


@dataclass(frozen=True, slots=True)
class Increment4EntityProjectionState:
    entity: CanonicalEntity
    version: CanonicalEntityVersion
    preferred: EntityPreferredIdentity
    aliases: tuple[EntityAlias, ...]
    projection_event: EntityProjectionEvent

    def __post_init__(self) -> None:
        if not isinstance(self.entity, CanonicalEntity):
            raise Increment4ProofContractError("entity state requires a canonical entity")
        if not isinstance(self.version, CanonicalEntityVersion):
            raise Increment4ProofContractError("entity state requires a canonical version")
        if not isinstance(self.preferred, EntityPreferredIdentity):
            raise Increment4ProofContractError("entity state requires preferred authority")
        if not isinstance(self.projection_event, EntityProjectionEvent):
            raise Increment4ProofContractError("entity state requires a projection event")
        if self.projection_event.action is not EntityProjectionAction.UPSERT:
            raise Increment4ProofContractError("only admitted current entity UPSERT state can project")
        if (
            self.entity.entity_id != self.version.entity_id
            or self.entity.entity_id != self.preferred.entity_id
            or self.version.entity_version_id
            != self.preferred.current_entity_version_id
            or self.projection_event.entity_id != self.entity.entity_id
            or self.projection_event.entity_version_id != self.version.entity_version_id
            or self.projection_event.preferred_entity_id != self.preferred.preferred_entity_id
            or self.projection_event.lifecycle != self.version.lifecycle
            or self.preferred.lifecycle != self.version.lifecycle
        ):
            raise Increment4ProofContractError("entity current, version and projection identities differ")
        if not isinstance(self.aliases, tuple):
            raise Increment4ProofContractError("entity aliases must be an immutable tuple")
        alias_ids = tuple(str(item.alias_id) for item in self.aliases)
        if alias_ids != tuple(sorted(set(alias_ids))):
            raise Increment4ProofContractError("entity aliases must be sorted and unique")
        for alias in self.aliases:
            if not isinstance(alias, EntityAlias):
                raise Increment4ProofContractError("entity alias must be typed")
            if alias.entity_id != self.entity.entity_id:
                raise Increment4ProofContractError(
                    "entity alias targets another canonical entity"
                )

    @property
    def canonical_digest(self) -> str:
        return digest_canonical(
            {
                "entity": self.entity.canonical_value(),
                "version": self.version.canonical_value(),
                "preferred": {
                    "entity_id": str(self.preferred.entity_id),
                    "current_entity_version_id": str(
                        self.preferred.current_entity_version_id
                    ),
                    "preferred_entity_id": str(self.preferred.preferred_entity_id),
                    "lifecycle": self.preferred.lifecycle.value,
                    "decided_by_kind": (
                        None
                        if self.preferred.decided_by_kind is None
                        else self.preferred.decided_by_kind.value
                    ),
                    "decided_by_id": self.preferred.decided_by_id,
                    "projected_through_ledger_seq": (
                        self.preferred.projected_through_ledger_seq
                    ),
                },
                "aliases": [
                    {
                        "alias_id": str(item.alias_id),
                        "entity_id": str(item.entity_id),
                        "entity_version_id": str(item.entity_version_id),
                        "normalization_contract_digest": (
                            item.normalization_contract_digest
                        ),
                        "language": item.language,
                        "script": item.script.value,
                        "alias_kind": item.alias_kind.value,
                        "valid_from": (
                            None if item.valid_from is None else item.valid_from.to_text()
                        ),
                        "valid_until": (
                            None if item.valid_until is None else item.valid_until.to_text()
                        ),
                        "provenance_mention_id": str(item.provenance_mention_id),
                        "resolution_decision_id": str(item.resolution_decision_id),
                        "uncertainty_codes": list(item.uncertainty_codes),
                    }
                    for item in self.aliases
                ],
                "projection_event": self.projection_event.canonical_value(),
            }
        )


@dataclass(frozen=True, slots=True)
class Increment4RelationProjectionState:
    current: EditorialRelationCurrentView
    projection_event: EditorialRelationProjectionEvent

    def __post_init__(self) -> None:
        if not isinstance(self.current, EditorialRelationCurrentView):
            raise Increment4ProofContractError("relation state requires a current view")
        if not isinstance(self.projection_event, EditorialRelationProjectionEvent):
            raise Increment4ProofContractError("relation state requires a projection event")
        assertion = self.current.assertion
        canonical_entity_pair = isinstance(
            assertion.subject, CanonicalEntityRelationEndpoint
        ) and isinstance(assertion.object, CanonicalEntityRelationEndpoint)
        assertion_lineage_pair = isinstance(
            assertion.subject, RelationAssertionRelationEndpoint
        ) and isinstance(assertion.object, RelationAssertionRelationEndpoint)
        if not (canonical_entity_pair or assertion_lineage_pair):
            raise Increment4ProofContractError(
                "Increment 4 admitted proof supports canonical-entity or "
                "relation-assertion endpoint pairs"
            )
        if (
            self.current.lifecycle is not EditorialRelationAssertionLifecycle.ACTIVE
            or self.projection_event.action
            is not EditorialRelationProjectionAction.UPSERT
            or self.projection_event.lifecycle
            is not EditorialRelationAssertionLifecycle.ACTIVE
            or self.projection_event.assertion_id != assertion.assertion_id
            or self.projection_event.assertion != assertion
        ):
            raise Increment4ProofContractError("relation current and projection authority differ")
        if assertion.trust_scope is not TrustScope.ADMITTED:
            raise Increment4ProofContractError("only admitted relation assertions can project")

    @property
    def canonical_digest(self) -> str:
        assertion = self.current.assertion
        return digest_canonical(
            {
                "assertion_id": str(assertion.assertion_id),
                "assertion_digest": assertion.canonical_digest,
                "relation_key": assertion.relation_key,
                "lifecycle": self.current.lifecycle.value,
                "current_decision_id": str(self.current.current_decision_id),
                "current_decision_version": self.current.current_decision_version,
                "projection_event_id": str(self.projection_event.projection_event_id),
                "projection_event_digest": self.projection_event.canonical_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class Increment4AdmittedProjectionSnapshot:
    entities: tuple[Increment4EntityProjectionState, ...]
    relations: tuple[Increment4RelationProjectionState, ...]
    events: tuple[LedgerEventRecord, ...]
    through_ledger_seq: int

    def __post_init__(self) -> None:
        if not isinstance(self.entities, tuple):
            raise Increment4ProofContractError(
                "proof entities must be an immutable tuple"
            )
        if not isinstance(self.relations, tuple):
            raise Increment4ProofContractError("proof relations must be an immutable tuple")
        if not isinstance(self.events, tuple) or not self.events:
            raise Increment4ProofContractError("proof snapshot requires retained ledger events")
        entity_ids = tuple(str(item.entity.entity_id) for item in self.entities)
        if entity_ids != tuple(sorted(set(entity_ids))):
            raise Increment4ProofContractError("proof entities must be sorted and unique")
        assertion_ids = tuple(
            str(item.current.assertion.assertion_id) for item in self.relations
        )
        if assertion_ids != tuple(sorted(set(assertion_ids))):
            raise Increment4ProofContractError("proof relations must be sorted and unique")
        event_sequences = tuple(item.ledger_seq for item in self.events)
        if event_sequences != tuple(sorted(set(event_sequences))):
            raise Increment4ProofContractError("proof events must be sequence-sorted and unique")
        event_ids = {item.event_id for item in self.events}
        required_event_ids = _required_event_ids(self.entities, self.relations)
        missing = required_event_ids - event_ids
        if missing:
            raise Increment4ProofContractError(
                "proof snapshot lacks exact retained event provenance: "
                + ",".join(sorted(missing))
            )
        if (
            isinstance(self.through_ledger_seq, bool)
            or not isinstance(self.through_ledger_seq, int)
            or self.through_ledger_seq <= 0
            or self.through_ledger_seq < max(event_sequences)
        ):
            raise Increment4ProofContractError("proof cutoff must cover all retained events")
        current_versions = {
            str(item.version.entity_version_id): item for item in self.entities
        }
        current_entities = {str(item.entity.entity_id) for item in self.entities}
        for state in self.entities:
            preferred = str(state.preferred.preferred_entity_id)
            if preferred not in current_entities:
                raise Increment4ProofContractError(
                    "preferred entity is absent from the admitted projection snapshot"
                )
        for state in self.relations:
            assertion = state.current.assertion
            if isinstance(assertion.subject, CanonicalEntityRelationEndpoint):
                if not isinstance(assertion.object, CanonicalEntityRelationEndpoint):
                    raise Increment4ProofContractError(
                        "admitted relation endpoint kinds differ"
                    )
                subject = str(assertion.subject.entity_version_id)
                object_ = str(assertion.object.entity_version_id)
                if subject not in current_versions or object_ not in current_versions:
                    raise Increment4ProofContractError(
                        "admitted relation endpoint is absent from current entity authority"
                    )
            elif not (
                isinstance(assertion.subject, RelationAssertionRelationEndpoint)
                and isinstance(assertion.object, RelationAssertionRelationEndpoint)
            ):
                raise Increment4ProofContractError(
                    "admitted relation endpoint kinds are outside the proof contract"
                )

    @property
    def event_by_id(self) -> dict[str, LedgerEventRecord]:
        return {item.event_id: item for item in self.events}

    @property
    def canonical_digest(self) -> str:
        return digest_canonical(
            {
                "contract": "newsroom.increment4.admitted-snapshot.v1",
                "through_ledger_seq": self.through_ledger_seq,
                "entities": [item.canonical_digest for item in self.entities],
                "relations": [item.canonical_digest for item in self.relations],
                "events": [_event_digest(item) for item in self.events],
            }
        )


def _required_event_ids(entities, relations) -> set[str]:
    required_event_ids: set[str] = set()
    for state in entities:
        required_event_ids.update(
            {
                str(state.entity.authority_event_id),
                str(state.version.authority_event_id),
                str(state.projection_event.source_event_id),
            }
        )
        required_event_ids.update(str(alias.authority_event_id) for alias in state.aliases)
    for state in relations:
        required_event_ids.add(str(state.projection_event.source_event_id))
    return required_event_ids


def _stream_admitted_provenance(
    *,
    entities: tuple[Increment4EntityProjectionState, ...],
    relations: tuple[Increment4RelationProjectionState, ...],
    events: Iterable[LedgerEventRecord | RetiredLedgerEventRecord],
    through_ledger_seq: int,
    hash_history: bool = True,
) -> tuple[Increment4AdmittedProjectionSnapshot, str | None]:
    """Hash complete history; retain only mapping provenance and the final event.

    The returned tuple snapshot is only for the existing state validator/mapper.
    Its subset digest is NOT the full-history identity returned alongside it.
    """
    required = _required_event_ids(entities, relations)
    retained = []
    last = None
    digest = hashlib.sha256() if hash_history else None
    # Exactly canonical_json_bytes(snapshot digest value), in sorted key order,
    # but without retaining the full list of event digests or its JSON encoding.
    if digest is not None:
        digest.update(canonical_json_bytes({
            "contract": "newsroom.increment4.admitted-snapshot.v1",
            "entities": [item.canonical_digest for item in entities],
        })[:-1])
        digest.update(b',"events":[')
    for event in events:
        if last is not None:
            if event.ledger_seq <= last.ledger_seq:
                raise Increment4ProofContractError("proof events must be sequence-sorted and unique")
        if digest is not None:
            if last is not None:
                digest.update(b",")
            digest.update(canonical_json_bytes(_event_digest(event)))
        if event.event_id in required:
            if isinstance(event, RetiredLedgerEventRecord):
                raise Increment4ProofContractError("required mapping diagnostic provenance expired")
            retained.append(event)
        last = event
    if last is None or last.ledger_seq != through_ledger_seq:
        raise Increment4ProofContractError("proof watermark lacks an exact retained event")
    if isinstance(last, RetiredLedgerEventRecord):
        raise Increment4ProofContractError("proof watermark diagnostic provenance expired")
    if last.event_id not in required:
        retained.append(last)
    # Reuse all current entity, relation, endpoint and provenance checks.
    provenance = Increment4AdmittedProjectionSnapshot(
        entities, relations, tuple(retained), through_ledger_seq,
    )
    if digest is None:
        return provenance, None
    digest.update(b'],"relations":')
    digest.update(canonical_json_bytes([item.canonical_digest for item in relations]))
    digest.update(b',"through_ledger_seq":')
    digest.update(canonical_json_bytes(through_ledger_seq))
    digest.update(b"}")
    return provenance, "sha256:" + digest.hexdigest()


def sorted_snapshot(
    *,
    entities: Iterable[Increment4EntityProjectionState],
    relations: Iterable[Increment4RelationProjectionState],
    events: Iterable[LedgerEventRecord],
    through_ledger_seq: int,
) -> Increment4AdmittedProjectionSnapshot:
    return Increment4AdmittedProjectionSnapshot(
        entities=tuple(sorted(entities, key=lambda item: str(item.entity.entity_id))),
        relations=tuple(
            sorted(relations, key=lambda item: str(item.current.assertion.assertion_id))
        ),
        events=tuple(sorted(events, key=lambda item: item.ledger_seq)),
        through_ledger_seq=through_ledger_seq,
    )


__all__ = [
    "Increment4AdmittedProjectionSnapshot",
    "Increment4EntityProjectionState",
    "Increment4ProofContractError",
    "Increment4RelationProjectionState",
    "sorted_snapshot",
]
