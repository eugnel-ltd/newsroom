from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import sqlite3

from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.entities.types import (
    CanonicalEntityId,
    CanonicalEntityLifecycle,
    EntityCreationDecisionKind,
    EntityProjectionAction,
)
from newsroom.increment4.models import (
    Increment4AdmittedProjectionSnapshot,
    Increment4EntityProjectionState,
    Increment4RelationProjectionState,
    sorted_snapshot,
    _required_event_ids,
    _stream_admitted_provenance,
)
from newsroom.increment4.projection import build_increment4_admitted_batches
from newsroom.projection.models import (
    ProjectionFamilyDefinition, ProjectionGenerationId, ProjectionStateError,
)
from newsroom.projection.neo4j.models import StructuralBatch
from newsroom.relations.editorial_models import (
    CanonicalEntityRelationEndpoint,
)
from newsroom.relations.editorial_types import (
    EditorialRelationAssertionId,
    EditorialRelationStaleDecision,
)

from ._editorial_relation_store import _EditorialRelationAuthorityStore
from ._projection_store import _ProjectionAuthorityStore


@dataclass(frozen=True, slots=True)
class _Increment4CurrentBuildInputs:
    source_watermark: int
    snapshot_digest: str
    batches: tuple[StructuralBatch, ...]


class _Increment4ProjectionAuthorityStore(
    _EditorialRelationAuthorityStore,
    _ProjectionAuthorityStore,
):
    """Projection authority with exact governed Increment 4 snapshot reads."""

    def increment4_admitted_snapshot(self) -> Increment4AdmittedProjectionSnapshot:
        """Rederive current admitted graph input from retained SQLite authority.

        Callers may carry a snapshot only as an optimistic exact-value assertion.
        The returned object is reconstructed inside the authority boundary after
        current rights and retained canonical records have been revalidated.
        """

        with self._lock:
            conn = self._connection
            entity_states, relation_states, source_watermark = self._increment4_admitted_states()
            events = tuple(
                self._event_from_row(row)
                for row in conn.execute(
                    "SELECT * FROM ledger_events WHERE ledger_seq<=? ORDER BY ledger_seq",
                    (source_watermark,),
                )
            )
            if not events or events[-1].ledger_seq != source_watermark:
                raise AuthorityPersistenceError(
                    "Increment 4 source watermark lacks an exact retained event"
                )
            return sorted_snapshot(
                entities=entity_states,
                relations=relation_states,
                events=events,
                through_ledger_seq=source_watermark,
            )

    def _increment4_admitted_states(self):
        """Read current admitted state on the caller's locked connection."""
        conn = self._connection
        source_watermark = self.latest_projection_source_ledger_seq()
        if source_watermark <= 0:
            raise ProjectionStateError(
                "Increment 4 admitted projection has no retained source authority"
            )

        entity_states: list[Increment4EntityProjectionState] = []
        # Every lineage transition retains an UPSERT projection event. Split-
        # created successors stop belonging to current graph state when that
        # split is reversed, while a reversed merge successor remains required
        # to preserve the admitted merge/reversal lineage. Creation authority
        # therefore participates in current graph membership.
        entity_rows = conn.execute(
            "SELECT p.entity_id "
            "FROM entity_preferred_identities AS p "
            "JOIN canonical_entities AS c ON c.entity_id=p.entity_id "
            "JOIN entity_projection_events AS e "
            "ON e.entity_id=p.entity_id "
            "AND e.source_ledger_seq=p.projected_through_ledger_seq "
            "WHERE e.action=? "
            "AND NOT (p.lifecycle=? AND c.created_by_kind=?) "
            "ORDER BY p.entity_id",
            (
                EntityProjectionAction.UPSERT.value,
                CanonicalEntityLifecycle.REVERSED.value,
                EntityCreationDecisionKind.SPLIT.value,
            ),
        ).fetchall()
        for row in entity_rows:
            entity_id = CanonicalEntityId.parse(str(row["entity_id"]))
            try:
                entity = self.entity(entity_id)
                preferred = self.preferred_identity(entity_id)
                version = self.entity_version(
                    preferred.current_entity_version_id
                )
            except PermissionError:
                # Rights-invalid current state must disappear from derivative
                # authority rather than being copied from stale caller memory.
                continue

            # Alias evidence can have independent rights from the retained
            # entity creation decision. Decode every immutable alias row, but
            # retain only aliases whose own provenance remains currently
            # usable. One revoked alias must not remove an otherwise-current
            # entity or make relation endpoint membership inconsistent.
            alias_rows = conn.execute(
                "SELECT * FROM entity_aliases WHERE entity_id=? "
                "ORDER BY language,normalized_text,alias_id",
                (str(entity_id),),
            ).fetchall()
            admitted_aliases = []
            for alias_row in alias_rows:
                alias = self._alias_from_row(conn, alias_row)
                mention = self._mention_from_row(
                    conn,
                    self._mention_row(conn, alias.provenance_mention_id),
                    replayed=False,
                )
                try:
                    self._require_mention_current(conn, mention)
                except PermissionError:
                    continue
                admitted_aliases.append(alias)
            aliases = tuple(
                sorted(admitted_aliases, key=lambda item: str(item.alias_id))
            )
            projection_rows = conn.execute(
                "SELECT * FROM entity_projection_events "
                "WHERE entity_id=? AND source_ledger_seq=? "
                "ORDER BY projection_event_id",
                (
                    str(entity_id),
                    preferred.projected_through_ledger_seq,
                ),
            ).fetchall()
            if len(projection_rows) != 1:
                raise AuthorityPersistenceError(
                    "Increment 4 entity lacks one exact current projection event"
                )
            projection_event = self._projection_event_from_row(
                conn, projection_rows[0]
            )
            if projection_event.source_ledger_seq > source_watermark:
                raise AuthorityPersistenceError(
                    "Increment 4 entity projection exceeds source authority"
                )
            entity_states.append(
                Increment4EntityProjectionState(
                    entity=entity,
                    version=version,
                    preferred=preferred,
                    aliases=aliases,
                    projection_event=projection_event,
                )
            )

        # Current derivative authority must be dependency-closed. A merge or
        # split predecessor cannot remain when its preferred target was
        # excluded by current rights. Remove dangling states to a fixed point
        # so longer preferred-identity chains fail closed as one unit.
        entity_state_by_id = {
            str(item.entity.entity_id): item for item in entity_states
        }
        while True:
            retained_entity_ids = set(entity_state_by_id)
            dangling_entity_ids = tuple(
                sorted(
                    entity_id
                    for entity_id, item in entity_state_by_id.items()
                    if str(item.preferred.preferred_entity_id)
                    not in retained_entity_ids
                )
            )
            if not dangling_entity_ids:
                break
            for entity_id in dangling_entity_ids:
                del entity_state_by_id[entity_id]
        entity_states = [
            entity_state_by_id[entity_id]
            for entity_id in sorted(entity_state_by_id)
        ]
        current_entity_version_ids = {
            str(item.version.entity_version_id) for item in entity_states
        }

        relation_states: list[Increment4RelationProjectionState] = []
        relation_rows = conn.execute(
            "SELECT assertion_id FROM editorial_current_admitted_relations "
            "ORDER BY assertion_id"
        ).fetchall()
        for row in relation_rows:
            assertion_id = EditorialRelationAssertionId.parse(
                str(row["assertion_id"])
            )
            try:
                current = self.editorial_current(assertion_id)
            except (PermissionError, EditorialRelationStaleDecision):
                # Rights-invalid or endpoint-stale assertions remain immutable
                # history but cannot participate in the current graph snapshot.
                continue
            assertion = current.assertion
            if isinstance(
                assertion.subject,
                CanonicalEntityRelationEndpoint,
            ):
                if not isinstance(
                    assertion.object,
                    CanonicalEntityRelationEndpoint,
                ):
                    raise AuthorityPersistenceError(
                        "Increment 4 relation endpoint kinds differ"
                    )
                if (
                    str(assertion.subject.entity_version_id)
                    not in current_entity_version_ids
                    or str(assertion.object.entity_version_id)
                    not in current_entity_version_ids
                ):
                    # A relation can remain individually current while an
                    # endpoint was removed by preferred-identity closure.
                    # Preserve its immutable history but omit the derivative.
                    continue
            projection_row = conn.execute(
                "SELECT * FROM editorial_relation_projection_events "
                "WHERE assertion_id=? AND source_ledger_seq<=? "
                "ORDER BY source_ledger_seq DESC,projection_event_id DESC LIMIT 1",
                (str(assertion_id), source_watermark),
            ).fetchone()
            if projection_row is None:
                raise AuthorityPersistenceError(
                    "Increment 4 relation lacks current projection authority"
                )
            projection_event = self._editorial_projection_event_from_row(
                conn, projection_row
            )
            relation_states.append(
                Increment4RelationProjectionState(
                    current=current,
                    projection_event=projection_event,
                )
            )

        return (
            tuple(sorted(entity_states, key=lambda item: str(item.entity.entity_id))),
            tuple(sorted(relation_states, key=lambda item: str(item.current.assertion.assertion_id))),
            source_watermark,
        )

    @contextmanager
    def _increment4_projection_read(self):
        # Pin current rights/state and historical provenance together. A caller's
        # transaction belongs to the caller; never commit or roll it back here.
        with self._lock:
            conn = self._connection
            nested = conn.in_transaction
            conn.execute("SAVEPOINT increment4_current_build" if nested else "BEGIN")
            try:
                yield (conn, *self._increment4_admitted_states())
            finally:
                if nested:
                    conn.execute("ROLLBACK TO increment4_current_build")
                    conn.execute("RELEASE increment4_current_build")
                else:
                    conn.execute("ROLLBACK")

    def _increment4_current_build_inputs(
        self,
        *,
        generation_id: ProjectionGenerationId,
        family: ProjectionFamilyDefinition,
    ) -> _Increment4CurrentBuildInputs:
        with self._increment4_projection_read() as (conn, entities, relations, watermark):
            provenance, snapshot_digest = _stream_admitted_provenance(
                entities=entities,
                relations=relations,
                events=(
                    self._event_from_row(row)
                    for row in conn.execute(
                        "SELECT * FROM ledger_events WHERE ledger_seq<=? ORDER BY ledger_seq",
                        (watermark,),
                    )
                ),
                through_ledger_seq=watermark,
            )
            assert snapshot_digest is not None
            batches = build_increment4_admitted_batches(
                provenance, generation_id=generation_id, family=family,
            )
            return _Increment4CurrentBuildInputs(watermark, snapshot_digest, batches)

    def _increment4_current_batches(
        self,
        *,
        generation_id: ProjectionGenerationId,
        family: ProjectionFamilyDefinition,
    ) -> tuple[StructuralBatch, ...]:
        with self._increment4_projection_read() as (conn, entities, relations, watermark):
            # Reconciliation has no full-history digest. Read its exact current
            # provenance through the existing unique event index, not every
            # historical control/audit event. The build-input path stays full.
            required = sorted(_required_event_ids(entities, relations))
            events = []
            chunk_size = min(512, conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))
            for offset in range(0, len(required), chunk_size):
                chunk = required[offset:offset + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                events.extend(
                    self._event_from_row(row) for row in conn.execute(
                        f"SELECT * FROM ledger_events WHERE event_id IN ({placeholders})",
                        chunk,
                    )
                )
            if not any(event.ledger_seq == watermark for event in events):
                row = conn.execute(
                    "SELECT * FROM ledger_events WHERE ledger_seq=?", (watermark,),
                ).fetchone()
                if row is not None:
                    events.append(self._event_from_row(row))
            provenance, _ = _stream_admitted_provenance(
                entities=entities,
                relations=relations,
                events=sorted(events, key=lambda event: event.ledger_seq),
                through_ledger_seq=watermark,
                # Reconciliation requires exact provenance, not a new build's
                # full-history command identity. All state checks still run.
                hash_history=False,
            )
            return build_increment4_admitted_batches(
                provenance, generation_id=generation_id, family=family,
            )


__all__ = ["_Increment4ProjectionAuthorityStore"]
