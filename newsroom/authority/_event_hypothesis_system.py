"""Private checked v21 Event Hypothesis authority composition."""

from __future__ import annotations

import fcntl
import os
import sqlite3
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from threading import Lock, get_ident
from typing import Self

from newsroom.authority.auth import AuthenticationProof, StaticAuthenticator
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority.migrations import (
    EXPECTED_MIGRATION_HISTORY,
    EXPECTED_SCHEMA_FINGERPRINT,
    SCHEMA_VERSION,
    apply_pending_migrations,
    prepare_pending_migration_backup,
    schema_fingerprint,
)
from newsroom.authority.types import UtcTimestamp
from newsroom.increment6.dispositions import (
    CurrentCandidateCitationReadPort,
    DispositionJudgement,
    ProposalDisposition,
    ProposalDispositionStore,
    _require_verification_snapshot,
    _verification_snapshot,
)
from newsroom.increment6.hypotheses import (
    EventHypothesis,
    EventHypothesisVersion,
    HypothesisContractError,
    HypothesisSourceBinding,
)
from newsroom.increment6.proposals import (
    HypothesisRelationship,
    LeadRecommendation,
    ProposalRoute,
    TriageProposal,
)
from newsroom.increment6.work_items import RetrievalContextAuthority

_REQUIRE_DISPOSITION = ProposalDispositionStore.require_current_in_transaction
_AUTHENTICATE_DISPOSITION = ProposalDispositionStore._authenticate
_VERIFY_DISPOSITION_INTEGRITY = (
    ProposalDispositionStore._verified_dispositions_in_transaction
)
_CREATE_ROUTES = {
    HypothesisRelationship.NO_ADEQUATE_PRIOR_MATCH,
    HypothesisRelationship.RELATED_DISTINCT,
    HypothesisRelationship.UNCERTAIN,
}
_APPEND_ROUTES = {
    HypothesisRelationship.SAME_STATE,
    HypothesisRelationship.DEVELOPMENT_OF,
    HypothesisRelationship.CORRECTION_REVERSAL_OF,
}
_ALLOWED_ROUTES = {
    ProposalRoute.ASSOCIATE_WITHOUT_CANDIDATE,
    ProposalRoute.NEW_EVENT_CANDIDATE,
    ProposalRoute.DEVELOPMENT_CANDIDATE,
    ProposalRoute.CORRECTION_CANDIDATE,
}


def _require_exact_proposal_authorisation(
    disposition: ProposalDisposition,
    recommendation: LeadRecommendation,
    proposal: TriageProposal,
    proposal_digest: str,
) -> None:
    if (
        disposition.judgement is not DispositionJudgement.ACCEPT
        or disposition.route not in _ALLOWED_ROUTES
        or disposition.proposal_id != proposal.proposal_id
        or disposition.proposal_content_identity != proposal.content_identity
        or disposition.proposal_canonical_digest != proposal_digest
        or disposition.route_binding != recommendation
        or disposition.decision_lead_id != recommendation.decision_lead_id
    ):
        raise HypothesisContractError(
            "disposition does not authorise the exact Proposal group"
        )


def _require_exact_proposal_provenance(
    version: EventHypothesisVersion,
    proposal: TriageProposal,
) -> None:
    if (
        version.work_item_id != proposal.work_item.work_item_id
        or version.work_item_version_id != proposal.work_item.work_item_version_id
        or version.work_item_version_digest
        != proposal.work_item.work_item_version_digest
        or version.retrieval_context_id != proposal.retrieval_context.context_id
        or version.retrieval_context_digest != proposal.retrieval_context.context_digest
    ):
        raise HypothesisContractError(
            "Hypothesis Version provenance differs from the exact Proposal"
        )


def _secure_directory(path: Path) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise HypothesisContractError(
                "authority database parent must be a real directory"
            )
    else:
        path.mkdir(parents=True, mode=0o700)
        os.chmod(path, 0o700)
    info = path.stat()
    if (hasattr(os, "getuid") and info.st_uid != os.getuid()) or stat.S_IMODE(
        info.st_mode
    ) != 0o700:
        raise HypothesisContractError("authority database parent ownership differs")


def _validate_owned_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise HypothesisContractError("authority database must be a regular file")
    info = path.stat()
    if (hasattr(os, "getuid") and info.st_uid != os.getuid()) or stat.S_IMODE(
        info.st_mode
    ) != 0o600:
        raise HypothesisContractError("authority database file ownership differs")


def _time(clock: Callable[[], UtcTimestamp]) -> str:
    return clock().to_text()


def _version_event_id(
    hypothesis_id: str,
    ordinal: int,
    previous_version_digest: str | None,
    proposal_digest: str,
    local_id: str,
    target_version_digest: str | None,
    bindings: tuple[HypothesisSourceBinding, ...],
    actor: str,
    recorded_at: str,
) -> str:
    identity = digest_bytes(
        canonical_json_bytes(
            {
                "ordinal": ordinal,
                "previous_version_digest": previous_version_digest,
                "proposal_digest": proposal_digest,
                "proposal_local_id": local_id,
                "target_version_digest": target_version_digest,
                "source_disposition_ids": [item.disposition_id for item in bindings],
                "actor_identity_digest": actor,
                "recorded_at": recorded_at,
            }
        )
    )
    return str(uuid.uuid5(uuid.UUID(hypothesis_id), f"authority-event:{identity}"))


def _creation_event_id(hypothesis_id: str, actor: str, recorded_at: str) -> str:
    identity = digest_bytes(
        canonical_json_bytes(
            {
                "hypothesis_id": hypothesis_id,
                "actor_identity_digest": actor,
                "recorded_at": recorded_at,
            }
        )
    )
    return str(uuid.uuid5(uuid.UUID(hypothesis_id), f"authority-create:{identity}"))


class _HypothesisStore:
    def __init__(
        self,
        connection: sqlite3.Connection,
        retrieval_authority: RetrievalContextAuthority,
        authenticator: StaticAuthenticator,
        clock: Callable[[], UtcTimestamp],
        current_candidate_citations: CurrentCandidateCitationReadPort | None = None,
        *,
        dispositions: ProposalDispositionStore | None = None,
        verified_dispositions: Mapping[str, ProposalDisposition] | None = None,
        verified_disposition_snapshot: tuple[int, int, int] | None = None,
        _open_verification: list[
            tuple[dict[str, EventHypothesisVersion], tuple[int, int, int]]
        ] | None = None,
        _validate_on_open: bool = True,
    ) -> None:
        if (
            type(connection) is not sqlite3.Connection
            or connection.in_transaction
            or type(retrieval_authority) is not RetrievalContextAuthority
            or type(authenticator) is not StaticAuthenticator
            or (
                dispositions is not None
                and (
                    type(dispositions) is not ProposalDispositionStore
                    or dispositions._connection is not connection
                    or dispositions._retrieval_authority is not retrieval_authority
                    or dispositions._authenticator is not authenticator
                    or dispositions._current_candidate_citations
                    is not current_candidate_citations
                )
            )
            or (
                verified_dispositions is not None
                and (
                    dispositions is None
                    or type(verified_dispositions) is not dict
                    or any(
                        type(value) is not ProposalDisposition
                        or key != value.disposition_id
                        for key, value in verified_dispositions.items()
                    )
                )
            )
            or (
                (verified_dispositions is None)
                != (verified_disposition_snapshot is None)
            )
            or (
                _open_verification is not None
                and (type(_open_verification) is not list or _open_verification)
            )
        ):
            raise HypothesisContractError("Hypothesis authority collaborators differ")
        self._connection = connection
        self._retrieval = retrieval_authority
        self._authenticator = authenticator
        self._clock = clock
        self._lock = Lock()
        self._owner: int | None = None
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            retrieval_authority.attach(connection)
            self._dispositions = dispositions or ProposalDispositionStore(
                connection, retrieval_authority, authenticator,
                current_candidate_citations,
            )
            self._begin()
            if verified_disposition_snapshot is not None:
                _require_verification_snapshot(
                    connection, verified_disposition_snapshot
                )
            if _validate_on_open:
                verified = self._verify(verified_dispositions=verified_dispositions)
                snapshot = _verification_snapshot(connection)
            self._commit()
            if _open_verification is not None and _validate_on_open:
                _open_verification.append((verified, snapshot))
        except BaseException as exc:
            self._rollback()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, HypothesisContractError):
                raise
            raise HypothesisContractError(
                "Hypothesis authority initialisation failed"
            ) from exc

    def _begin(self) -> None:
        self._lock.acquire()
        try:
            if self._connection.in_transaction:
                raise HypothesisContractError("connection has an active transaction")
            self._connection.execute("BEGIN IMMEDIATE")
            self._owner = get_ident()
        except BaseException:
            self._lock.release()
            raise

    def _commit(self) -> None:
        if self._owner != get_ident() or not self._connection.in_transaction:
            raise HypothesisContractError("transaction ownership differs")
        self._connection.execute("COMMIT")
        self._owner = None
        self._lock.release()

    def adopt_active_transaction(self) -> None:
        """Bind checked private helpers to an enclosing authority transaction."""
        if self._owner is not None or not self._connection.in_transaction:
            raise HypothesisContractError("active transaction adoption differs")
        self._owner = get_ident()

    def release_active_transaction(self) -> None:
        if self._owner != get_ident() or not self._connection.in_transaction:
            raise HypothesisContractError("active transaction ownership differs")
        self._owner = None

    def _rollback(self) -> None:
        if self._owner != get_ident():
            return
        try:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
        finally:
            self._owner = None
            self._lock.release()

    @staticmethod
    def _exact_dispositions(
        values: Sequence[ProposalDisposition],
    ) -> tuple[ProposalDisposition, ...]:
        if type(values) is not tuple or not values or len(values) > 32:
            raise HypothesisContractError(
                "dispositions must be a non-empty bounded exact sequence"
            )
        result = tuple(values)
        if any(type(value) is not ProposalDisposition for value in result):
            raise HypothesisContractError("dispositions must be exact retained values")
        if tuple(value.decision_lead_id for value in result) != tuple(
            sorted({value.decision_lead_id for value in result})
        ):
            raise HypothesisContractError(
                "dispositions must be sorted and unique by decision Lead"
            )
        return result

    @staticmethod
    def _bindings(
        values: tuple[ProposalDisposition, ...],
    ) -> tuple[HypothesisSourceBinding, ...]:
        return tuple(
            HypothesisSourceBinding(
                value.disposition_id,
                digest_bytes(value.canonical_bytes),
                value.finding_set_digest,
                value.route_binding_digest,
                value.decision_lead_id,
                value.lead_head.decision_lead_digest,
                value.lead_head.current_disposition_head_id,
                value.lead_head.current_disposition_head_digest,
            )
            for value in values
        )

    def retain(
        self,
        proposal_bytes: bytes,
        dispositions: Sequence[ProposalDisposition],
        *,
        proof: AuthenticationProof,
        expected_target_version: EventHypothesisVersion | None = None,
        proposal_local_id: str | None = None,
    ) -> EventHypothesisVersion:
        """Create or append exactly one complete proposal-local Hypothesis group."""
        try:
            proposal = TriageProposal.from_canonical_bytes(proposal_bytes)
            supplied = self._exact_dispositions(dispositions)
            local_ids = {
                value.route_binding.hypothesis.proposal_local_id
                for value in supplied
                if value.route_binding.hypothesis is not None
            }
            if proposal_local_id is not None:
                local_ids &= {proposal_local_id}
            if len(local_ids) != 1:
                raise HypothesisContractError(
                    "one exact proposal-local Hypothesis group is required"
                )
            local_id = next(iter(local_ids))
            recommendations = tuple(
                r
                for r in proposal.recommendations
                if r.hypothesis is not None
                and r.hypothesis.proposal_local_id == local_id
            )
            if not recommendations:
                raise HypothesisContractError("Proposal Hypothesis group is absent")
            required_leads = tuple(r.decision_lead_id for r in recommendations)
            if tuple(value.decision_lead_id for value in supplied) != required_leads:
                raise HypothesisContractError(
                    "Proposal Hypothesis group is partial or extra"
                )
            proposed = recommendations[0].hypothesis
            assert proposed is not None
            if any(r.hypothesis != proposed for r in recommendations):
                raise HypothesisContractError("Proposal Hypothesis group diverges")
            bindings = self._bindings(supplied)

            # Lost-response lookup is deliberately before currentness checks.
            # BEGIN IMMEDIATE also serialises a competing connection and the
            # local ownership lock prevents a thread observing uncommitted work.
            self._begin()
            row = self._connection.execute(
                "SELECT canonical_bytes,proposal_canonical_bytes FROM event_hypothesis_versions_v2 "
                "WHERE proposal_id=? AND proposal_local_id=?",
                (proposal.proposal_id, local_id),
            ).fetchone()
            if row is not None:
                retained = self._exact_version(
                    EventHypothesisVersion.from_canonical_bytes(bytes(row[0])).version_id
                )
                retained_dispositions = tuple(
                    self._dispositions._verify_requested_disposition_in_transaction(
                        binding.disposition_id
                    )
                    for binding in retained.source_bindings
                )
                if (
                    bytes(row[1]) != proposal_bytes
                    or retained.proposal_canonical_digest
                    != digest_bytes(proposal_bytes)
                    or retained.source_bindings != bindings
                    or tuple(value.canonical_bytes for value in retained_dispositions)
                    != tuple(value.canonical_bytes for value in supplied)
                ):
                    raise HypothesisContractError("semantic replay diverges")
                _, replay_actor = _AUTHENTICATE_DISPOSITION(self._dispositions, proof)
                if replay_actor != retained.actor_identity_digest:
                    raise HypothesisContractError("semantic replay actor differs")
                if retained.proposed_target_hypothesis_id is None:
                    if expected_target_version is not None:
                        raise HypothesisContractError(
                            "semantic replay comparator diverges"
                        )
                elif (
                    type(expected_target_version) is not EventHypothesisVersion
                    or expected_target_version.hypothesis_id
                    != retained.proposed_target_hypothesis_id
                    or expected_target_version.version_id != retained.target_version_id
                    or expected_target_version.canonical_digest
                    != retained.target_version_digest
                ):
                    raise HypothesisContractError("semantic replay comparator diverges")
                self._commit()
                return retained
            retained_values = tuple(
                self._require_current_disposition(value.disposition_id, proof)
                for value in supplied
            )
            if tuple(value.canonical_bytes for value in retained_values) != tuple(
                value.canonical_bytes for value in supplied
            ):
                raise HypothesisContractError(
                    "caller disposition differs from retained authority"
                )
            for value, recommendation in zip(
                retained_values, recommendations, strict=True
            ):
                _require_exact_proposal_authorisation(
                    value,
                    recommendation,
                    proposal,
                    digest_bytes(proposal_bytes),
                )
            actors = {
                value.validator_input.authenticated_context_identity
                for value in retained_values
            }
            if len(actors) != 1:
                raise HypothesisContractError(
                    "Hypothesis group requires one authenticated actor"
                )
            actor_identity_digest = next(iter(actors))

            target = proposed.target_hypothesis_id
            target_version: EventHypothesisVersion | None = None
            if target is not None:
                if type(expected_target_version) is not EventHypothesisVersion:
                    raise HypothesisContractError(
                        "target-bearing relationship requires the exact current target Version"
                    )
                if expected_target_version.hypothesis_id != target:
                    raise HypothesisContractError(
                        "relationship target differs from expected head"
                    )
                current_target = self._head(target)
                if (
                    current_target.version_id != expected_target_version.version_id
                    or current_target.canonical_digest
                    != expected_target_version.canonical_digest
                ):
                    raise HypothesisContractError("target Hypothesis head CAS differs")
                target_version = current_target
            if proposed.relationship_kind in _CREATE_ROUTES:
                hypothesis = EventHypothesis.allocate(proposal.proposal_id, local_id)
                ordinal, predecessor = 1, None
                if target is None and expected_target_version is not None:
                    raise HypothesisContractError(
                        "new Hypothesis does not accept a head token"
                    )
            elif proposed.relationship_kind in _APPEND_ROUTES:
                if target is None or target_version is None:
                    raise HypothesisContractError(
                        "append requires the exact expected current Version"
                    )
                hypothesis = EventHypothesis(target)
                ordinal, predecessor = target_version.ordinal + 1, target_version
            else:
                raise HypothesisContractError("Hypothesis relationship is unsupported")

            now = _time(self._clock)
            authority_event_id = _version_event_id(
                hypothesis.hypothesis_id,
                ordinal,
                None if predecessor is None else predecessor.canonical_digest,
                digest_bytes(proposal_bytes),
                local_id,
                None if target_version is None else target_version.canonical_digest,
                bindings,
                actor_identity_digest,
                now,
            )
            version = EventHypothesisVersion(
                str(
                    uuid.uuid5(
                        uuid.UUID(hypothesis.hypothesis_id), f"version:{ordinal}"
                    )
                ),
                hypothesis.hypothesis_id,
                ordinal,
                None if predecessor is None else predecessor.version_id,
                None if predecessor is None else predecessor.canonical_digest,
                proposed.summary,
                proposed.relationship_kind,
                proposed.target_hypothesis_id,
                None if target_version is None else target_version.version_id,
                None if target_version is None else target_version.canonical_digest,
                proposal.proposal_id,
                proposal.content_identity,
                digest_bytes(proposal_bytes),
                local_id,
                proposal.work_item.work_item_id,
                proposal.work_item.work_item_version_id,
                proposal.work_item.work_item_version_digest,
                proposal.retrieval_context.context_id,
                proposal.retrieval_context.context_digest,
                bindings,
                actor_identity_digest,
                authority_event_id,
                now,
            )
            _require_exact_proposal_provenance(version, proposal)
            self._connection.execute(
                "INSERT OR IGNORE INTO event_hypotheses_v2 VALUES(?,?,?,?,?,?)",
                (
                    hypothesis.hypothesis_id,
                    hypothesis.canonical_bytes,
                    hypothesis.canonical_digest,
                    actor_identity_digest,
                    _creation_event_id(
                        hypothesis.hypothesis_id, actor_identity_digest, now
                    ),
                    now,
                ),
            )
            existing_identity = self._connection.execute(
                "SELECT canonical_bytes FROM event_hypotheses_v2 WHERE hypothesis_id=?",
                (hypothesis.hypothesis_id,),
            ).fetchone()
            if (
                existing_identity is None
                or bytes(existing_identity[0]) != hypothesis.canonical_bytes
            ):
                raise HypothesisContractError("Hypothesis identity collision diverges")
            self._connection.execute(
                "INSERT INTO event_hypothesis_versions_v2("
                "version_id,hypothesis_id,ordinal,previous_version_id,previous_version_digest,"
                "proposal_id,proposal_local_id,proposal_content_identity,proposal_canonical_digest,"
                "proposal_canonical_bytes,proposed_relationship,proposed_target_hypothesis_id,"
                "target_version_id,target_version_digest,work_item_id,work_item_version_id,"
                "work_item_version_digest,retrieval_context_id,retrieval_context_digest,"
                "actor_identity_digest,authority_event_id,canonical_bytes,canonical_digest,recorded_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    version.version_id,
                    version.hypothesis_id,
                    version.ordinal,
                    version.previous_version_id,
                    version.previous_version_digest,
                    version.proposal_id,
                    version.proposal_local_id,
                    version.proposal_content_identity,
                    version.proposal_canonical_digest,
                    proposal_bytes,
                    version.proposed_relationship.value,
                    version.proposed_target_hypothesis_id,
                    version.target_version_id,
                    version.target_version_digest,
                    version.work_item_id,
                    version.work_item_version_id,
                    version.work_item_version_digest,
                    version.retrieval_context_id,
                    version.retrieval_context_digest,
                    version.actor_identity_digest,
                    version.authority_event_id,
                    version.canonical_bytes,
                    version.canonical_digest,
                    version.recorded_at,
                ),
            )
            if predecessor is None:
                self._connection.execute(
                    "INSERT INTO event_hypothesis_heads_v2 VALUES(?,?,?,?,?)",
                    (
                        version.hypothesis_id,
                        version.version_id,
                        version.ordinal,
                        version.canonical_digest,
                        now,
                    ),
                )
            else:
                changed = self._connection.execute(
                    "UPDATE event_hypothesis_heads_v2 SET version_id=?,ordinal=?,version_digest=?,updated_at=? "
                    "WHERE hypothesis_id=? AND version_id=? AND version_digest=?",
                    (
                        version.version_id,
                        version.ordinal,
                        version.canonical_digest,
                        now,
                        version.hypothesis_id,
                        predecessor.version_id,
                        predecessor.canonical_digest,
                    ),
                ).rowcount
                if changed != 1:
                    raise HypothesisContractError("Hypothesis head CAS lost")
            if self._exact_version(version.version_id) != version:
                raise HypothesisContractError("Hypothesis Version readback differs")
            if self._head(version.hypothesis_id) != version:
                raise HypothesisContractError("Hypothesis head readback differs")
            self._commit()
            return version
        except BaseException as exc:
            self._rollback()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, HypothesisContractError):
                raise
            raise HypothesisContractError("Hypothesis retention failed") from exc

    create_or_append = retain

    def load_version(self, version_id: str) -> EventHypothesisVersion:
        try:
            self._begin()
            self._verify()
            row = self._connection.execute(
                "SELECT canonical_bytes FROM event_hypothesis_versions_v2 WHERE version_id=?",
                (version_id,),
            ).fetchone()
            if row is None:
                raise HypothesisContractError("unknown Hypothesis Version")
            value = EventHypothesisVersion.from_canonical_bytes(bytes(row[0]))
            self._commit()
            return value
        except BaseException as exc:
            self._rollback()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, HypothesisContractError):
                raise
            raise HypothesisContractError("Hypothesis Version load failed") from exc

    def load_hypothesis(self, hypothesis_id: str) -> EventHypothesis:
        try:
            self._begin()
            self._verify()
            row = self._connection.execute(
                "SELECT canonical_bytes FROM event_hypotheses_v2 WHERE hypothesis_id=?",
                (hypothesis_id,),
            ).fetchone()
            if row is None:
                raise HypothesisContractError("unknown Hypothesis")
            value = EventHypothesis.from_canonical_bytes(bytes(row[0]))
            self._commit()
            return value
        except BaseException as exc:
            self._rollback()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, HypothesisContractError):
                raise
            raise HypothesisContractError("Hypothesis load failed") from exc

    def _head(self, hypothesis_id: str) -> EventHypothesisVersion:
        row = self._connection.execute(
            "SELECT version_id,ordinal,version_digest,updated_at "
            "FROM event_hypothesis_heads_v2 "
            "WHERE hypothesis_id=?",
            (hypothesis_id,),
        ).fetchone()
        if row is None:
            raise HypothesisContractError("unknown current Hypothesis")
        version = self._exact_version(str(row[0]))
        if (
            version.hypothesis_id != hypothesis_id
            or version.ordinal != int(row[1])
            or version.canonical_digest != str(row[2])
            or version.recorded_at != str(row[3])
            or self._connection.execute(
                "SELECT max(ordinal) FROM event_hypothesis_versions_v2 "
                "WHERE hypothesis_id=?",
                (hypothesis_id,),
            ).fetchone()[0]
            != version.ordinal
        ):
            raise HypothesisContractError("Hypothesis head binding differs")
        return version

    def _require_current_disposition(
        self, disposition_id: str, proof: AuthenticationProof
    ) -> ProposalDisposition:
        return _REQUIRE_DISPOSITION(
            self._dispositions, disposition_id, proof=proof
        )

    @staticmethod
    def _version_from_row(
        row: sqlite3.Row,
    ) -> tuple[EventHypothesisVersion, TriageProposal]:
        value = EventHypothesisVersion.from_canonical_bytes(bytes(row[21]))
        scalars = (
            value.version_id, value.hypothesis_id, value.ordinal,
            value.previous_version_id, value.previous_version_digest,
            value.proposal_id, value.proposal_local_id,
            value.proposal_content_identity, value.proposal_canonical_digest,
            value.proposed_relationship.value,
            value.proposed_target_hypothesis_id, value.target_version_id,
            value.target_version_digest, value.work_item_id,
            value.work_item_version_id, value.work_item_version_digest,
            value.retrieval_context_id, value.retrieval_context_digest,
            value.actor_identity_digest, value.authority_event_id,
            value.canonical_digest, value.recorded_at,
        )
        if scalars != tuple(
            row[index] for index in (*range(9), *range(10, 21), 22, 23)
        ):
            raise HypothesisContractError("retained Hypothesis Version scalars differ")
        proposal = TriageProposal.from_canonical_bytes(bytes(row[9]))
        if (
            digest_bytes(bytes(row[9])) != value.proposal_canonical_digest
            or proposal.proposal_id != value.proposal_id
            or proposal.content_identity != value.proposal_content_identity
        ):
            raise HypothesisContractError("retained Proposal differs")
        return value, proposal

    @staticmethod
    def _version_recommendations(
        value: EventHypothesisVersion, proposal: TriageProposal
    ) -> tuple[LeadRecommendation, ...]:
        recommendations = tuple(
            item
            for item in proposal.recommendations
            if item.hypothesis is not None
            and item.hypothesis.proposal_local_id == value.proposal_local_id
        )
        if not recommendations or any(
            item.hypothesis.summary != value.proposed_summary
            or item.hypothesis.relationship_kind is not value.proposed_relationship
            or item.hypothesis.target_hypothesis_id
            != value.proposed_target_hypothesis_id
            for item in recommendations
        ):
            raise HypothesisContractError("retained Proposal retargeted")
        if tuple(item.decision_lead_id for item in recommendations) != tuple(
            binding.decision_lead_id for binding in value.source_bindings
        ):
            raise HypothesisContractError(
                "retained Proposal group coverage differs"
            )
        return recommendations

    def _require_version_source_closure(
        self,
        value: EventHypothesisVersion,
        proposal: TriageProposal,
        dispositions: tuple[ProposalDisposition, ...],
    ) -> None:
        recommendations = self._version_recommendations(value, proposal)
        if len(recommendations) != len(dispositions):
            raise HypothesisContractError("retained source bindings differ")
        for disposition, recommendation in zip(
            dispositions, recommendations, strict=True
        ):
            _require_exact_proposal_authorisation(
                disposition,
                recommendation,
                proposal,
                value.proposal_canonical_digest,
            )
            if (
                disposition.proposal_id != value.proposal_id
                or disposition.proposal_content_identity
                != value.proposal_content_identity
                or disposition.proposal_canonical_digest
                != value.proposal_canonical_digest
                or disposition.work_item_id != value.work_item_id
                or disposition.work_item_version_id != value.work_item_version_id
                or disposition.work_item_version_digest
                != value.work_item_version_digest
                or disposition.retrieval_context_id != value.retrieval_context_id
                or disposition.retrieval_context_digest
                != value.retrieval_context_digest
            ):
                raise HypothesisContractError("retained source disposition retargeted")
            if (
                disposition.validator_input.authenticated_context_identity
                != value.actor_identity_digest
            ):
                raise HypothesisContractError("retained source actor differs")
        if self._bindings(dispositions) != value.source_bindings:
            raise HypothesisContractError("retained source bindings differ")

    def _require_creation_provenance(
        self, first: EventHypothesisVersion
    ) -> None:
        identity = self._connection.execute(
            "SELECT canonical_bytes,canonical_digest,actor_identity_digest,"
            "authority_event_id,recorded_at FROM event_hypotheses_v2 "
            "WHERE hypothesis_id=?",
            (first.hypothesis_id,),
        ).fetchone()
        expected = EventHypothesis(first.hypothesis_id)
        if (
            first.ordinal != 1
            or identity is None
            or bytes(identity[0]) != expected.canonical_bytes
            or str(identity[1]) != expected.canonical_digest
            or str(identity[2]) != first.actor_identity_digest
            or str(identity[3]) != _creation_event_id(
                first.hypothesis_id, first.actor_identity_digest, first.recorded_at
            )
            or str(identity[4]) != first.recorded_at
        ):
            raise HypothesisContractError("Hypothesis creation provenance differs")

    def _exact_version(
        self, version_id: str, _visited: frozenset[str] = frozenset()
    ) -> EventHypothesisVersion:
        if version_id in _visited:
            raise HypothesisContractError("Hypothesis target cycle differs")
        visited = _visited | {version_id}
        row = self._connection.execute(
            "SELECT version_id,hypothesis_id,ordinal,previous_version_id,"
            "previous_version_digest,proposal_id,proposal_local_id,"
            "proposal_content_identity,proposal_canonical_digest,"
            "proposal_canonical_bytes,proposed_relationship,"
            "proposed_target_hypothesis_id,target_version_id,target_version_digest,"
            "work_item_id,work_item_version_id,work_item_version_digest,"
            "retrieval_context_id,retrieval_context_digest,actor_identity_digest,"
            "authority_event_id,canonical_bytes,canonical_digest,recorded_at "
            "FROM event_hypothesis_versions_v2 WHERE version_id=?",
            (version_id,),
        ).fetchone()
        if row is None:
            raise HypothesisContractError("unknown retained Hypothesis Version")
        value, proposal = self._version_from_row(row)
        if (
            value.version_id != version_id
            or value.authority_event_id != _version_event_id(
                value.hypothesis_id, value.ordinal, value.previous_version_digest,
                value.proposal_canonical_digest, value.proposal_local_id,
                value.target_version_digest, value.source_bindings,
                value.actor_identity_digest, value.recorded_at,
            )
        ):
            raise HypothesisContractError("retained Hypothesis Version differs")
        _require_exact_proposal_provenance(value, proposal)
        retained_dispositions = tuple(
            self._dispositions._verify_requested_disposition_in_transaction(
                binding.disposition_id
            )
            for binding in value.source_bindings
        )
        self._require_version_source_closure(
            value, proposal, retained_dispositions
        )
        predecessor = None
        if value.previous_version_id is not None:
            predecessor_row = self._connection.execute(
                "SELECT hypothesis_id,ordinal,canonical_digest FROM "
                "event_hypothesis_versions_v2 WHERE version_id=?",
                (value.previous_version_id,),
            ).fetchone()
            predecessor = None if predecessor_row is None else tuple(predecessor_row)
        if (
            (value.ordinal == 1)
            != (value.previous_version_id is None)
            or (
                predecessor is not None
                and predecessor
                != (
                    value.hypothesis_id,
                    value.ordinal - 1,
                    value.previous_version_digest,
                )
            )
            or (value.previous_version_id is not None and predecessor is None)
        ):
            raise HypothesisContractError("Hypothesis predecessor binding differs")
        if value.previous_version_id is not None:
            self._exact_version(value.previous_version_id, visited)
        if value.target_version_id is not None:
            target = self._connection.execute(
                "SELECT hypothesis_id,canonical_digest FROM "
                "event_hypothesis_versions_v2 WHERE version_id=?",
                (value.target_version_id,),
            ).fetchone()
            if target is None or tuple(target) != (
                value.proposed_target_hypothesis_id,
                value.target_version_digest,
            ):
                raise HypothesisContractError("Hypothesis target binding differs")
            if value.target_version_id != value.previous_version_id:
                target_value = self._exact_version(value.target_version_id, visited)
                if (
                    target_value.hypothesis_id
                    != value.proposed_target_hypothesis_id
                    or target_value.canonical_digest != value.target_version_digest
                ):
                    raise HypothesisContractError("Hypothesis target binding differs")
        first_row = self._connection.execute(
            "SELECT version_id,hypothesis_id,ordinal,previous_version_id,"
            "previous_version_digest,proposal_id,proposal_local_id,"
            "proposal_content_identity,proposal_canonical_digest,"
            "proposal_canonical_bytes,proposed_relationship,"
            "proposed_target_hypothesis_id,target_version_id,target_version_digest,"
            "work_item_id,work_item_version_id,work_item_version_digest,"
            "retrieval_context_id,retrieval_context_digest,actor_identity_digest,"
            "authority_event_id,canonical_bytes,canonical_digest,recorded_at "
            "FROM event_hypothesis_versions_v2 WHERE hypothesis_id=? AND ordinal=1",
            (value.hypothesis_id,),
        ).fetchone()
        if first_row is None:
            raise HypothesisContractError("Hypothesis creation provenance differs")
        first, _ = self._version_from_row(first_row)
        self._require_creation_provenance(first)
        return value

    def require_retained_version_in_transaction(
        self, version_id: str
    ) -> EventHypothesisVersion:
        """Load one retained Version inside this store's checked transaction."""
        if self._owner != get_ident() or not self._connection.in_transaction:
            raise HypothesisContractError("transaction ownership differs")
        return self._exact_version(version_id)

    def require_current_version_in_transaction(
        self, version_id: str, *, proof: AuthenticationProof
    ) -> EventHypothesisVersion:
        """Recheck the exact Version head and its authenticated source chain."""
        if self._owner != get_ident() or not self._connection.in_transaction:
            raise HypothesisContractError("transaction ownership differs")
        verified = {version_id: self._exact_version(version_id)}
        versions, _ = self._require_current_versions_after_integrity_in_transaction(
            (version_id,), verified, proof=proof
        )
        return versions[0]

    def _exact_current_versions_in_transaction(
        self, version_ids: tuple[str, ...], *, proof: AuthenticationProof
    ) -> tuple[
        tuple[EventHypothesisVersion, ...], tuple[ProposalDisposition, ...]
    ]:
        if self._owner != get_ident() or not self._connection.in_transaction:
            raise HypothesisContractError("transaction ownership differs")
        verified = {
            version_id: self._exact_version(version_id)
            for version_id in version_ids
        }
        return self._require_current_versions_after_integrity_in_transaction(
            version_ids, verified, proof=proof
        )

    def _require_current_versions_after_integrity_in_transaction(
        self,
        version_ids: tuple[str, ...],
        verified: dict[str, EventHypothesisVersion],
        *,
        proof: AuthenticationProof,
    ) -> tuple[
        tuple[EventHypothesisVersion, ...],
        tuple[ProposalDisposition, ...],
    ]:
        """Check current heads after this transaction's full retained replay."""
        if (
            self._owner != get_ident()
            or not self._connection.in_transaction
            or type(version_ids) is not tuple
            or type(verified) is not dict
            or any(type(item) is not str for item in version_ids)
            or any(
                type(key) is not str
                or type(value) is not EventHypothesisVersion
                or key != value.version_id
                for key, value in verified.items()
            )
        ):
            raise HypothesisContractError("verified current Version batch differs")
        try:
            versions = tuple(verified[item] for item in version_ids)
        except KeyError as exc:
            raise HypothesisContractError(
                "unknown retained Hypothesis Version"
            ) from exc
        dispositions: dict[str, ProposalDisposition] = {}
        for version in versions:
            if self._head(version.hypothesis_id) != version:
                raise HypothesisContractError(
                    "Hypothesis Version is not the current head"
                )
            for binding in version.source_bindings:
                disposition = (
                    self._require_current_disposition(
                        binding.disposition_id, proof
                    )
                )
                if self._bindings((disposition,)) != (binding,):
                    raise HypothesisContractError("current source binding differs")
                retained = dispositions.setdefault(
                    disposition.disposition_id, disposition
                )
                if retained != disposition:
                    raise HypothesisContractError(
                        "current source disposition differs within batch"
                    )
        return versions, tuple(dispositions[item] for item in sorted(dispositions))

    def current(
        self, hypothesis_id: str, *, proof: AuthenticationProof
    ) -> EventHypothesisVersion:
        try:
            self._begin()
            self._verify()
            version = self._head(hypothesis_id)
            for binding in version.source_bindings:
                disposition = _REQUIRE_DISPOSITION(
                    self._dispositions, binding.disposition_id, proof=proof
                )
                if self._bindings((disposition,)) != (binding,):
                    raise HypothesisContractError("current source binding differs")
            self._commit()
            return version
        except BaseException as exc:
            self._rollback()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, HypothesisContractError):
                raise
            raise HypothesisContractError("Hypothesis currentness failed") from exc

    require_current = current

    def versions(self, hypothesis_id: str) -> tuple[EventHypothesisVersion, ...]:
        try:
            self._begin()
            self._verify()
            rows = self._connection.execute(
                "SELECT canonical_bytes FROM event_hypothesis_versions_v2 WHERE hypothesis_id=? ORDER BY ordinal",
                (hypothesis_id,),
            ).fetchall()
            values = tuple(
                EventHypothesisVersion.from_canonical_bytes(bytes(row[0]))
                for row in rows
            )
            self._commit()
            return values
        except BaseException as exc:
            self._rollback()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, HypothesisContractError):
                raise
            raise HypothesisContractError("Hypothesis history load failed") from exc

    def close(self, operation: Callable[[], None]) -> None:
        self._lock.acquire()
        try:
            operation()
        finally:
            self._lock.release()

    def _verify(
        self,
        *,
        verified_dispositions: Mapping[str, ProposalDisposition] | None = None,
    ) -> dict[str, EventHypothesisVersion]:
        dispositions_by_id = (
            _VERIFY_DISPOSITION_INTEGRITY(self._dispositions)
            if verified_dispositions is None
            else dict(verified_dispositions)
        )
        tables = {
            str(row[0])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {
            "event_hypotheses_v2",
            "event_hypothesis_versions_v2",
            "event_hypothesis_heads_v2",
        }
        if not required <= tables:
            raise HypothesisContractError("v21 Hypothesis schema is absent")
        identities: dict[str, EventHypothesis] = {}
        for row in self._connection.execute(
            "SELECT hypothesis_id,canonical_bytes,canonical_digest,actor_identity_digest,authority_event_id,recorded_at FROM event_hypotheses_v2"
        ):
            value = EventHypothesis.from_canonical_bytes(bytes(row[1]))
            if (
                value.hypothesis_id != row[0]
                or value.canonical_digest != row[2]
                or row[4]
                != _creation_event_id(value.hypothesis_id, str(row[3]), str(row[5]))
            ):
                raise HypothesisContractError("retained Hypothesis identity differs")
            identities[value.hypothesis_id] = value
        chains: dict[str, list[EventHypothesisVersion]] = {}
        versions_by_id: dict[str, EventHypothesisVersion] = {}
        semantic: set[tuple[str, str]] = set()
        for row in self._connection.execute(
            "SELECT version_id,hypothesis_id,ordinal,previous_version_id,previous_version_digest,proposal_id,proposal_local_id,proposal_content_identity,proposal_canonical_digest,proposal_canonical_bytes,proposed_relationship,proposed_target_hypothesis_id,target_version_id,target_version_digest,work_item_id,work_item_version_id,work_item_version_digest,retrieval_context_id,retrieval_context_digest,actor_identity_digest,authority_event_id,canonical_bytes,canonical_digest,recorded_at FROM event_hypothesis_versions_v2 ORDER BY hypothesis_id,ordinal"
        ):
            value, proposal = self._version_from_row(row)
            if value.authority_event_id != _version_event_id(
                value.hypothesis_id,
                value.ordinal,
                value.previous_version_digest,
                value.proposal_canonical_digest,
                value.proposal_local_id,
                value.target_version_digest,
                value.source_bindings,
                value.actor_identity_digest,
                value.recorded_at,
            ):
                raise HypothesisContractError(
                    "retained authority event identity differs"
                )
            _require_exact_proposal_provenance(value, proposal)
            retained_dispositions = []
            for binding in value.source_bindings:
                disposition = dispositions_by_id.get(binding.disposition_id)
                if disposition is None:
                    raise HypothesisContractError(
                        "retained source disposition is absent"
                    )
                retained_dispositions.append(disposition)
            self._require_version_source_closure(
                value, proposal, tuple(retained_dispositions)
            )
            if value.proposed_target_hypothesis_id is not None:
                if value.proposed_relationship in _APPEND_ROUTES and (
                    value.previous_version_id != value.target_version_id
                    or value.previous_version_digest != value.target_version_digest
                ):
                    raise HypothesisContractError(
                        "append comparator differs from predecessor"
                    )
            if (value.proposal_id, value.proposal_local_id) in semantic:
                raise HypothesisContractError("duplicate semantic source")
            semantic.add((value.proposal_id, value.proposal_local_id))
            chains.setdefault(value.hypothesis_id, []).append(value)
            versions_by_id[value.version_id] = value
        # Target hypotheses may sort after their dependants. Resolve pins only
        # after every Version has been decoded in this same transaction.
        for value in versions_by_id.values():
            if value.proposed_target_hypothesis_id is not None:
                target = versions_by_id.get(value.target_version_id)
                if target is None or (
                    target.hypothesis_id, target.canonical_digest
                ) != (value.proposed_target_hypothesis_id, value.target_version_digest):
                    raise HypothesisContractError("retained target Version pin differs")
        if set(chains) != set(identities):
            raise HypothesisContractError(
                "Hypothesis identity/version coverage differs"
            )
        heads = {
            str(row[0]): (
                str(row[1]),
                int(row[2]),
                str(row[3]),
                str(row[4]),
            )
            for row in self._connection.execute(
                "SELECT hypothesis_id,version_id,ordinal,version_digest,updated_at "
                "FROM event_hypothesis_heads_v2"
            )
        }
        if set(heads) != set(chains):
            raise HypothesisContractError("Hypothesis head coverage differs")
        for hypothesis_id, versions in chains.items():
            self._require_creation_provenance(versions[0])
            for index, version in enumerate(versions, 1):
                predecessor = None if index == 1 else versions[index - 2]
                if (
                    version.ordinal != index
                    or version.previous_version_id
                    != (None if predecessor is None else predecessor.version_id)
                    or version.previous_version_digest
                    != (None if predecessor is None else predecessor.canonical_digest)
                ):
                    raise HypothesisContractError("Hypothesis chain is not contiguous")
            last = versions[-1]
            if heads[hypothesis_id] != (
                last.version_id,
                last.ordinal,
                last.canonical_digest,
                last.recorded_at,
            ):
                raise HypothesisContractError("Hypothesis head is not max Version")
        for table in ("event_hypothesis_versions_v2", "event_hypothesis_heads_v2"):
            if self._connection.execute(
                f'PRAGMA foreign_key_check("{table}")'
            ).fetchone() is not None:
                raise HypothesisContractError("Hypothesis foreign keys differ")
        return versions_by_id


_AUTHORITY_TOKEN = object()


class EventHypothesisAuthority:
    __slots__ = ("__close", "__closed", "__store")

    def __init__(
        self,
        token: object,
        store: _HypothesisStore,
        close: Callable[[], None],
    ) -> None:
        if token is not _AUTHORITY_TOKEN or type(store) is not _HypothesisStore:
            raise HypothesisContractError(
                "Hypothesis authority construction is private"
            )
        self.__store = store
        self.__close = close
        self.__closed = False

    @classmethod
    def open(
        cls,
        database: str | Path,
        *,
        retrieval_authority: RetrievalContextAuthority,
        authenticator: StaticAuthenticator,
        clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
        busy_timeout_ms: int = 5000,
    ) -> Self:
        if (
            isinstance(busy_timeout_ms, bool)
            or type(busy_timeout_ms) is not int
            or busy_timeout_ms <= 0
        ):
            raise HypothesisContractError("busy timeout must be positive")
        path = Path(database).expanduser().absolute()
        if path.is_symlink():
            raise HypothesisContractError("authority database path cannot be a symlink")
        _secure_directory(path.parent)
        existed = path.exists()
        if existed:
            _validate_owned_file(path)
        lock_path = path.with_name(path.name + ".writer.lock")
        if lock_path.is_symlink():
            raise HypothesisContractError("writer lock path cannot be a symlink")
        if lock_path.exists():
            _validate_owned_file(lock_path)
        descriptor = os.open(
            lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0), 0o600
        )
        os.fchmod(descriptor, 0o600)
        lock_info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or (hasattr(os, "getuid") and lock_info.st_uid != os.getuid())
            or stat.S_IMODE(lock_info.st_mode) != 0o600
        ):
            os.close(descriptor)
            raise HypothesisContractError("writer lock ownership differs")
        connection: sqlite3.Connection | None = None
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise HypothesisContractError(
                    "another authority writer is active"
                ) from exc
            connection = sqlite3.connect(
                path,
                isolation_level=None,
                timeout=busy_timeout_ms / 1000,
                check_same_thread=False,
            )
            if not existed:
                os.chmod(path, 0o600)
            _validate_owned_file(path)
            connection.execute("PRAGMA foreign_keys=ON")
            if (
                str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
                != "wal"
            ):
                raise HypothesisContractError("SQLite WAL mode is unavailable")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            current = int(connection.execute("PRAGMA user_version").fetchone()[0])
            tables = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
            ).fetchone()
            if current == 0 and tables is not None:
                raise HypothesisContractError(
                    "refusing a non-empty unversioned authority database"
                )
            if current < SCHEMA_VERSION:
                prepare_pending_migration_backup(connection)
            apply_pending_migrations(connection, applied_at=_time(clock))
            history = connection.execute(
                "SELECT version,name,checksum FROM authority_migrations ORDER BY version"
            ).fetchall()
            if (
                int(connection.execute("PRAGMA user_version").fetchone()[0])
                != SCHEMA_VERSION
                or history != list(EXPECTED_MIGRATION_HISTORY)
                or schema_fingerprint(connection) != EXPECTED_SCHEMA_FINGERPRINT
                or connection.execute("PRAGMA quick_check").fetchone()[0] != "ok"
                or connection.execute("PRAGMA foreign_key_check").fetchone() is not None
                or connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1
                or connection.execute("PRAGMA synchronous").fetchone()[0] != 2
                or connection.execute("PRAGMA busy_timeout").fetchone()[0]
                != busy_timeout_ms
                or str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
                != "wal"
            ):
                raise HypothesisContractError("checked authority lifecycle differs")
            store = _HypothesisStore(
                connection, retrieval_authority, authenticator, clock
            )

            def close() -> None:
                assert connection is not None
                connection.close()
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

            return cls(_AUTHORITY_TOKEN, store, close)
        except Exception:
            if connection is not None:
                connection.close()
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
            raise

    def retain(self, *args: object, **kwargs: object) -> EventHypothesisVersion:
        return self.__store.retain(*args, **kwargs)  # type: ignore[arg-type]

    create_or_append = retain

    def current(
        self, hypothesis_id: str, *, proof: AuthenticationProof
    ) -> EventHypothesisVersion:
        return self.__store.current(hypothesis_id, proof=proof)

    require_current = current

    def load_version(self, version_id: str) -> EventHypothesisVersion:
        return self.__store.load_version(version_id)

    def load_hypothesis(self, hypothesis_id: str) -> EventHypothesis:
        return self.__store.load_hypothesis(hypothesis_id)

    def versions(self, hypothesis_id: str) -> tuple[EventHypothesisVersion, ...]:
        return self.__store.versions(hypothesis_id)

    def close(self) -> None:
        if not self.__closed:
            self.__closed = True
            self.__store.close(self.__close)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _compose_event_hypothesis_authority_for_test(
    store: object, close: Callable[[], None]
) -> EventHypothesisAuthority:
    """Compose an exact raw authority for focused in-memory store tests."""

    return EventHypothesisAuthority(
        _AUTHORITY_TOKEN,
        store,
        close,  # type: ignore[arg-type]
    )


__all__: list[str] = []
