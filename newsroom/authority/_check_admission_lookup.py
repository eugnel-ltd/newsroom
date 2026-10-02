from __future__ import annotations

from newsroom.authority.types import UtcTimestamp
from newsroom.checks.record_models import (
    ObservableTransition,
    OperationalFindingOccurrence,
)
from newsroom.checks.types import (
    CheckRequestId,
    CheckStateError,
    OperationalFindingOccurrenceId,
)
from newsroom.sources import (
    CheckOutcomeId,
    DiscoveryOccurrence,
    DiscoveryOccurrenceId,
    DiscoveryOccurrenceKind,
    DiscoveryRepresentation,
    SourceDefinitionId,
    SourceItem,
    SourceItemId,
    SourceRevision,
    SourceRevisionId,
)


class _CheckAdmissionLookupMixin:
    """Typed semantic lookups for deterministic Check observation lineage."""

    def observed_prior_revision(
        self, item_id: SourceItemId, *, request_id: CheckRequestId,
        outcome_id: CheckOutcomeId, completed_at: UtcTimestamp,
    ) -> SourceRevision | None:
        """Resolve observed lineage at an exact Check boundary, not ingest order."""
        with self._lock:
            item = self.source_item(item_id)
            request = self.check_request(request_id)
            if item is None or request is None:
                raise LookupError("observed prior requires a retained Item and Check Request")
            if (
                item.request.definition_id != request.request.definition_id
                or item.request.definition_version_id != request.request.definition_version_id
            ):
                raise CheckStateError("observed prior Item differs from the Check source version")
            current = self.check_outcome(outcome_id)
            if current is not None and (
                current.request.request_id != request_id
                or current.request.definition_id != request.request.definition_id
                or current.request.definition_version_id != request.request.definition_version_id
                or current.request.completed_at != completed_at
            ):
                raise CheckStateError("observed prior boundary differs from retained Check Outcome")
            if current is not None and self._connection.execute(
                "SELECT COUNT(*) FROM check_outcome_observed_items "
                "WHERE outcome_id=? AND item_id=?",
                (str(outcome_id), str(item_id)),
            ).fetchone()[0] != 1:
                raise CheckStateError("observed prior Item is not observed by its Check Outcome")
            if completed_at.value < request.request.requested_at.value:
                raise CheckStateError("observed prior boundary precedes its Check Request")
            if self.unresolved_prior_observed_outcome_for_item(
                item_id, completed_at=completed_at, exclude_outcome_id=outcome_id,
            ):
                raise CheckStateError("prior observed Check Outcome lacks retained source Occurrence")
            prior = self.latest_observed_source_revision(
                item_id, exclude_outcome_id=outcome_id,
                before_completed_at=completed_at,
            )
            predicate, parameters = self._prior_outcome_predicate(
                outcome_alias="o", event_alias="e",
                current_outcome_id=outcome_id, completed_at=completed_at,
            )
            observed = self._connection.execute(
                "SELECT i.outcome_id FROM check_outcome_observed_items i "
                "JOIN check_outcomes o ON o.outcome_id=i.outcome_id "
                "JOIN ledger_events e ON e.event_id=o.authority_event_id "
                f"WHERE i.item_id=? AND i.outcome_id<>? AND {predicate} "
                "ORDER BY o.completed_at DESC,e.ledger_seq DESC LIMIT 1",
                (str(item_id), str(outcome_id), *parameters),
            ).fetchone()
            if observed is None:
                if prior is not None:
                    raise CheckStateError("observed prior Occurrence lacks its observed Item")
                return None
            observed_id = CheckOutcomeId.parse(str(observed["outcome_id"]))
            observed_outcome = self.check_outcome(observed_id)
            occurrences = self._connection.execute(
                "SELECT d.* FROM discovery_occurrences d "
                "JOIN source_revisions r ON r.revision_id=d.revision_id "
                "WHERE d.check_outcome_id=? AND r.item_id=? LIMIT 2",
                (str(observed_id), str(item_id)),
            ).fetchall()
            if observed_outcome is None or len(occurrences) != 1:
                raise CheckStateError("prior observed Check Outcome lacks one exact source Occurrence")
            occurrence = self._occurrence_from_row(
                self._connection, occurrences[0], replayed=False,
            ).request
            if (
                prior is None or prior.request.revision_id != occurrence.revision_id
                or prior.request.item_id != item_id
                or observed_outcome.request.definition_id != item.request.definition_id
                or observed_outcome.request.definition_version_id != occurrence.definition_version_id
                or occurrence.observed_at != observed_outcome.request.completed_at
            ):
                raise CheckStateError("observed prior differs from exact Check source lineage")
            return prior

    def _prior_outcome_predicate(
        self,
        *,
        outcome_alias: str,
        event_alias: str,
        current_outcome_id,
        completed_at: UtcTimestamp,
    ) -> tuple[str, tuple[object, ...]]:
        if not isinstance(completed_at, UtcTimestamp):
            raise TypeError("semantic outcome boundary requires typed UTC")
        current = self._connection.execute(
            "SELECT o.completed_at,e.ledger_seq FROM check_outcomes o "
            "JOIN ledger_events e ON e.event_id=o.authority_event_id "
            "WHERE o.outcome_id=?",
            (str(current_outcome_id),),
        ).fetchone()
        if current is None:
            return (
                f"{outcome_alias}.completed_at<=?",
                (completed_at.to_text(),),
            )
        if str(current["completed_at"]) != completed_at.to_text():
            raise CheckStateError(
                "retained Check Outcome completion time differs from admission"
            )
        return (
            f"({outcome_alias}.completed_at<? OR "
            f"({outcome_alias}.completed_at=? AND "
            f"{event_alias}.ledger_seq<?))",
            (
                completed_at.to_text(),
                completed_at.to_text(),
                int(current["ledger_seq"]),
            ),
        )

    def source_item_by_identity_digest(
        self,
        definition_id: SourceDefinitionId,
        identity_digest: str,
    ) -> SourceItem | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM source_items WHERE definition_id=? "
                "AND identity_digest=?",
                (str(definition_id), identity_digest),
            ).fetchone()
            return (
                None
                if row is None
                else self._source_item_from_row(
                    self._connection,
                    row,
                    replayed=False,
                )
            )

    def latest_source_revision(
        self,
        item_id: SourceItemId,
    ) -> SourceRevision | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT r.* FROM source_revisions r "
                "JOIN ledger_events e ON e.event_id=r.authority_event_id "
                "WHERE r.item_id=? ORDER BY e.ledger_seq DESC LIMIT 1",
                (str(item_id),),
            ).fetchone()
            return (
                None
                if row is None
                else self._source_revision_from_row(
                    self._connection,
                    row,
                    replayed=False,
                )
            )


    def latest_observed_source_revision(
        self,
        item_id: SourceItemId,
        *,
        exclude_outcome_id=None,
        before_completed_at: UtcTimestamp | None = None,
    ) -> SourceRevision | None:
        with self._lock:
            where = "WHERE r.item_id=?"
            parameters: list[object] = [str(item_id)]
            if exclude_outcome_id is not None:
                where += " AND o.check_outcome_id<>?"
                parameters.append(str(exclude_outcome_id))
            if before_completed_at is not None:
                if exclude_outcome_id is None:
                    raise TypeError(
                        "bounded observed Revision lookup requires Outcome ID"
                    )
                predicate, boundary_parameters = self._prior_outcome_predicate(
                    outcome_alias="c",
                    event_alias="ce",
                    current_outcome_id=exclude_outcome_id,
                    completed_at=before_completed_at,
                )
                where += f" AND {predicate}"
                parameters.extend(boundary_parameters)
            row = self._connection.execute(
                "SELECT r.* FROM discovery_occurrences o "
                "JOIN source_revisions r ON r.revision_id=o.revision_id "
                "JOIN check_outcomes c ON c.outcome_id=o.check_outcome_id "
                "JOIN ledger_events ce ON ce.event_id=c.authority_event_id "
                "JOIN ledger_events oe ON oe.event_id=o.authority_event_id "
                f"{where} ORDER BY c.completed_at DESC,ce.ledger_seq DESC,"
                "oe.ledger_seq DESC LIMIT 1",
                tuple(parameters),
            ).fetchone()
            return (
                None
                if row is None
                else self._source_revision_from_row(
                    self._connection,
                    row,
                    replayed=False,
                )
            )

    def discovery_occurrence_count_for_revision(
        self,
        revision_id: SourceRevisionId,
        *,
        exclude_outcome_id=None,
        before_completed_at: UtcTimestamp | None = None,
    ) -> int:
        with self._lock:
            sql = (
                "SELECT COUNT(*) FROM discovery_occurrences d "
                "JOIN check_outcomes c ON c.outcome_id=d.check_outcome_id "
                "JOIN ledger_events ce ON ce.event_id=c.authority_event_id "
                "WHERE d.revision_id=?"
            )
            parameters: list[object] = [str(revision_id)]
            if exclude_outcome_id is not None:
                sql += " AND d.check_outcome_id<>?"
                parameters.append(str(exclude_outcome_id))
            if before_completed_at is not None:
                if exclude_outcome_id is None:
                    raise TypeError(
                        "bounded Revision occurrence count requires Outcome ID"
                    )
                predicate, boundary_parameters = self._prior_outcome_predicate(
                    outcome_alias="c",
                    event_alias="ce",
                    current_outcome_id=exclude_outcome_id,
                    completed_at=before_completed_at,
                )
                sql += f" AND {predicate}"
                parameters.extend(boundary_parameters)
            return int(self._connection.execute(sql, tuple(parameters)).fetchone()[0])

    def discovery_occurrence_count_for_item(
        self,
        item_id: SourceItemId,
        *,
        exclude_outcome_id=None,
        before_completed_at: UtcTimestamp | None = None,
    ) -> int:
        with self._lock:
            sql = (
                "SELECT COUNT(*) FROM discovery_occurrences o "
                "JOIN source_revisions r ON r.revision_id=o.revision_id "
                "JOIN check_outcomes c ON c.outcome_id=o.check_outcome_id "
                "JOIN ledger_events ce ON ce.event_id=c.authority_event_id "
                "WHERE r.item_id=?"
            )
            parameters: list[object] = [str(item_id)]
            if exclude_outcome_id is not None:
                sql += " AND o.check_outcome_id<>?"
                parameters.append(str(exclude_outcome_id))
            if before_completed_at is not None:
                if exclude_outcome_id is None:
                    raise TypeError(
                        "bounded Item occurrence count requires Outcome ID"
                    )
                predicate, boundary_parameters = self._prior_outcome_predicate(
                    outcome_alias="c",
                    event_alias="ce",
                    current_outcome_id=exclude_outcome_id,
                    completed_at=before_completed_at,
                )
                sql += f" AND {predicate}"
                parameters.extend(boundary_parameters)
            return int(self._connection.execute(sql, tuple(parameters)).fetchone()[0])

    def latest_observable_transition_for_item(
        self,
        item_id: SourceItemId,
    ) -> ObservableTransition | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT t.* FROM observable_transitions t "
                "JOIN check_outcomes c ON c.outcome_id=t.check_outcome_id "
                "JOIN ledger_events ce ON ce.event_id=c.authority_event_id "
                "JOIN ledger_events te ON te.event_id=t.authority_event_id "
                "WHERE t.item_id=? ORDER BY c.completed_at DESC,"
                "ce.ledger_seq DESC,te.ledger_seq DESC LIMIT 1",
                (str(item_id),),
            ).fetchone()
            return (
                None
                if row is None
                else self._observable_transition_from_row(
                    self._connection,
                    row,
                    replayed=False,
                )
            )

    def unresolved_prior_observed_outcome_for_item(
        self,
        item_id: SourceItemId,
        *,
        completed_at: UtcTimestamp,
        exclude_outcome_id,
    ) -> bool:
        """Return whether earlier observed authority lacks its source Occurrence.

        Proposal admission commits a Check Outcome before its source lineage.
        A later Check must not classify an item while an earlier observed Outcome
        for the same deterministic Source Item is still missing that lineage.
        """

        if not isinstance(item_id, SourceItemId):
            raise TypeError(
                "prior observed-outcome lookup requires Source Item ID"
            )
        if not isinstance(completed_at, UtcTimestamp):
            raise TypeError("prior observed-outcome lookup requires typed UTC")
        with self._lock:
            predicate, boundary_parameters = self._prior_outcome_predicate(
                outcome_alias="o",
                event_alias="e",
                current_outcome_id=exclude_outcome_id,
                completed_at=completed_at,
            )
            row = self._connection.execute(
                "SELECT 1 FROM check_outcome_observed_items i "
                "JOIN check_outcomes o ON o.outcome_id=i.outcome_id "
                "JOIN ledger_events e ON e.event_id=o.authority_event_id "
                "WHERE i.item_id=? AND i.outcome_id<>? "
                f"AND {predicate} AND NOT EXISTS("
                "SELECT 1 FROM discovery_occurrences d "
                "JOIN source_revisions r ON r.revision_id=d.revision_id "
                "WHERE d.check_outcome_id=i.outcome_id "
                "AND r.item_id=i.item_id) LIMIT 1",
                (
                    str(item_id),
                    str(exclude_outcome_id),
                    *boundary_parameters,
                ),
            ).fetchone()
            return row is not None

    def source_revision_by_identity_digest(
        self,
        item_id: SourceItemId,
        revision_identity_digest: str,
    ) -> SourceRevision | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM source_revisions WHERE item_id=? "
                "AND revision_identity_digest=?",
                (str(item_id), revision_identity_digest),
            ).fetchone()
            return (
                None
                if row is None
                else self._source_revision_from_row(
                    self._connection,
                    row,
                    replayed=False,
                )
            )

    def representation_by_producer_slot(
        self,
        revision_id: SourceRevisionId,
        producer_slot_digest: str,
    ) -> DiscoveryRepresentation | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM discovery_representations WHERE revision_id=? "
                "AND producer_slot_digest=?",
                (str(revision_id), producer_slot_digest),
            ).fetchone()
            return (
                None
                if row is None
                else self._representation_from_row(
                    self._connection,
                    row,
                    replayed=False,
                )
            )

    def latest_representation_for_revision(
        self,
        revision_id: SourceRevisionId,
    ) -> DiscoveryRepresentation | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT p.* FROM discovery_representations p "
                "JOIN ledger_events e ON e.event_id=p.authority_event_id "
                "WHERE p.revision_id=? ORDER BY p.produced_at DESC,"
                "e.ledger_seq DESC LIMIT 1",
                (str(revision_id),),
            ).fetchone()
            return (
                None
                if row is None
                else self._representation_from_row(
                    self._connection,
                    row,
                    replayed=False,
                )
            )

    def discovery_occurrence_by_identity(
        self,
        occurrence_id: DiscoveryOccurrenceId,
    ) -> DiscoveryOccurrence | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM discovery_occurrences WHERE occurrence_id=?",
                (str(occurrence_id),),
            ).fetchone()
            return (
                None
                if row is None
                else self._occurrence_from_row(
                    self._connection,
                    row,
                    replayed=False,
                )
            )

    def discovery_occurrence_for_outcome_revision_any(
        self,
        *,
        check_outcome_id,
        revision_id,
    ) -> DiscoveryOccurrence | None:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM discovery_occurrences WHERE check_outcome_id=? "
                "AND revision_id=? ORDER BY recorded_at,occurrence_id",
                (str(check_outcome_id), str(revision_id)),
            ).fetchall()
            if len(rows) > 1:
                raise CheckStateError(
                    "one Check Outcome retained multiple Occurrences for one Revision"
                )
            return (
                None
                if not rows
                else self._occurrence_from_row(
                    self._connection,
                    rows[0],
                    replayed=False,
                )
            )

    def finding_occurrence_by_identity(
        self,
        occurrence_id: OperationalFindingOccurrenceId,
    ) -> OperationalFindingOccurrence | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM operational_finding_occurrences "
                "WHERE occurrence_id=?",
                (str(occurrence_id),),
            ).fetchone()
            return (
                None
                if row is None
                else self._finding_occurrence_from_row(
                    self._connection,
                    row,
                    replayed=False,
                )
            )

    def has_discovery_occurrence_for_revision(
        self,
        revision_id: SourceRevisionId,
    ) -> bool:
        with self._lock:
            return self._connection.execute(
                "SELECT 1 FROM discovery_occurrences WHERE revision_id=? LIMIT 1",
                (str(revision_id),),
            ).fetchone() is not None



__all__ = ["_CheckAdmissionLookupMixin"]
