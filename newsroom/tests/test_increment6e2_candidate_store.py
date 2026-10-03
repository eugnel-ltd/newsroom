from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import ClassVar

import pytest

from newsroom.authority._discovery_store import (
    _create_discovery_governing_producer_read_port,
)
from newsroom.authority._event_hypothesis_lineage_system import (
    _create_event_hypothesis_lineage_read_port,
)
from newsroom.authority.auth import AuthenticationProof, StaticAuthorizer
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority.migrations import (
    SCHEMA_VERSION,
    apply_pending_migrations,
    prepare_pending_migration_backup,
)
from newsroom.authority.persistence import (
    AuthorityPersistenceError,
    AuthoritySchemaError,
)
from newsroom.authority.story_candidate_system import (
    _create_story_candidate_read_port,
    _open_unlocked_story_candidate_authority_for_test,
)
from newsroom.authority.types import UtcTimestamp
from newsroom.discovery import NewsLeadId
from newsroom.increment6.candidates import (
    CandidateAdmission,
    CandidateAdmissionReason,
    CandidateAdmissionRequest,
    CandidateContractError,
    CandidateGoverningState,
    CandidateGoverningStateStatus,
    build_candidate_governing_manifest,
    candidate_command_definition,
    evaluate_candidate_admission,
    merge_candidate_authority_registries,
)
from newsroom.increment6.collision import (
    CandidateUseCollisionBinding,
    CandidateUseOperation,
    CurrentCollisionAuthoritySnapshot,
    CurrentCollisionEffectEnforcer,
    CurrentCollisionEligibilityRequest,
    CurrentCollisionReceiptEvidence,
    TrustedCurrentCollisionAuthorityBoundary,
    decide_current_collision_eligibility,
)
from newsroom.increment6.dispositions import ProposalDispositionStore
from newsroom.increment6.lineage import (
    lineage_command_definition,
    merge_lineage_authority_registries,
    open_event_hypothesis_lineage_authority,
)
from newsroom.increment6.outcomes import CanonicalOutcome
from newsroom.increment6.relationships import (
    ComparatorEvidence,
    ComparatorSetManifest,
    HypothesisVersionBinding,
    assess_relationships,
    merge_relationship_authority_registries,
    open_event_hypothesis_relationship_authority,
    relationship_command_definition,
)
from newsroom.tests import test_increment6d1_hypothesis_store as d1
from newsroom.tests import test_increment6d2_relationship_store as d2
from newsroom.tests import test_increment6d3_lineage as d3
from newsroom.tests.authority_store_conformance import (
    _SCENARIOS,
    CASE_INVENTORY,
    Applicability,
    AuthorityValue,
    BindingConflict,
    CaseId,
    IntegrityViolation,
    LostResponse,
    RollbackScope,
    StoredAuthorityState,
    TamperKind,
    WriteCommand,
    run_conformance,
)
from newsroom.tests.test_increment5c2_named_tool_authority_execution import (
    QUERY_VALID,
    SERVING,
    authority_database,
    authorize,
    collision_request,
    digest,
    executor,
)
from newsroom.tests.test_increment6e1_collision import _trusted_context

_RECORDS = (
    ("record-1", "alpha"),
    ("record-2", "beta"),
    ("record-a", "alpha"),
    ("record-b", "beta"),
    ("record-rollback-normal", "alpha"),
    ("record-rollback-abort", "alpha"),
)
_REQUESTS = ("different-request",) + tuple(f"request-{name}" for name, _ in _RECORDS)
_IDEMPOTENCIES = ("different-idempotency",) + tuple(
    f"idempotency-{name}" for name, _ in _RECORDS
)
_PREDECESSORS = ("different-cas_predecessor", "record-0", "record-1")
_UPSTREAMS = (
    (
        ("authority", "authority-head-1"),
        ("policy", "policy-head-1"),
    ),
)
_ACTOR_PROOFS = {
    "actor-1": AuthenticationProof(method="STATIC_TOKEN", credential="credential"),
    "different-actor": AuthenticationProof(method="STATIC_TOKEN", credential="other"),
}
_PRINCIPAL_ACTORS = {"editor": "actor-1", "other": "different-actor"}


def _generic(
    record_id: str, predecessor: str | None = None, value: str | None = None
) -> WriteCommand:
    predecessor = predecessor or ("record-1" if record_id == "record-2" else "record-0")
    value = value or ("beta" if record_id in {"record-2", "record-b"} else "alpha")
    return WriteCommand(
        record_id=record_id,
        canonical_bytes=f'{{"value":"{value}"}}'.encode(),
        scalar_columns={"value": value, "version": 1},
        identity_columns={"record_id": record_id, "authority": "fixture"},
        linked_rows=(
            {"child_id": f"child-{record_id}", "record_id": record_id, "ordinal": 0},
        ),
        actor="actor-1",
        request=f"request-{record_id}",
        idempotency=f"idempotency-{record_id}",
        cas_predecessor=predecessor,
        required_upstream_heads=_UPSTREAMS[0],
    )


def _code(values: tuple, value: object, field: str) -> int:
    try:
        return values.index(value)
    except ValueError as exc:
        raise CandidateContractError(
            f"unsupported conformance {field} binding"
        ) from exc


def _request_uuid(codes: tuple[int, ...]) -> str:
    raw = bytearray(hashlib.sha256(bytes(codes)).digest()[:16])
    raw[: len(codes)] = bytes(codes)
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(raw)))


def _request_codes(value: str) -> tuple[int, ...]:
    raw = uuid.UUID(value).bytes
    return tuple(raw[:6])


def _collision_digest(subject_version_id: str) -> str:
    return digest(f"candidate-conformance:{subject_version_id}")


def _database_was_locked(error: BaseException) -> bool:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, sqlite3.OperationalError) and "locked" in str(current):
            return True
        current = current.__cause__
    return False


@dataclass
class _Location:
    seed: tuple
    collision_root: Path

    def __post_init__(self) -> None:
        ids = tuple(item[0] for item in _RECORDS)
        versions = self.seed[3]
        mapped = (
            versions[0],
            versions[1],
            versions[0],
            versions[0],
            versions[0],
            versions[1],
        )
        self.subjects = dict(zip(ids, mapped, strict=True))
        self.relationships = {
            item.version_id: self.seed[8][item.version_id] for item in mapped
        }
        self.snapshots: dict[str, CurrentCollisionAuthoritySnapshot] = {}
        self.collision_results: dict[
            tuple[str, CandidateUseOperation, str | None], tuple[object, object]
        ] = {}
        self.collision_lock = Lock()


def _collaborators(seed) -> dict[str, object]:
    scopes = {
        relationship_command_definition().required_scope,
        lineage_command_definition().required_scope,
        candidate_command_definition().required_scope,
    }
    return {
        "database": seed[1],
        "retrieval_authority": seed[0][1],
        "authenticator": seed[0][2],
        "authorizer": StaticAuthorizer(
            policy_version="candidate-conformance-v1",
            grants_by_principal={
                "editor": frozenset(scopes),
                "other": frozenset(scopes),
            },
        ),
        "command_registry": seed[4],
        "payload_schemas": seed[5],
        "clock": lambda: UtcTimestamp.parse("2042-01-02T00:00:00.000000Z"),
        "busy_timeout_ms": 5_000,
    }


def _migrate_to_current(database: Path) -> None:
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        prepare_pending_migration_backup(connection)
        apply_pending_migrations(connection, applied_at="2042-01-02T00:00:00.000000Z")
        if connection.execute("PRAGMA user_version").fetchone() != (SCHEMA_VERSION,):
            raise AssertionError(
                "Candidate adapter migration did not reach current schema"
            )
    finally:
        connection.close()


def _build_seed(root: Path) -> tuple:
    seed = d2._seed_location(root, subjects=2)
    _migrate_to_current(seed[1])
    args = _collaborators(seed)
    retained = {}
    authority = open_event_hypothesis_relationship_authority(**args)
    try:
        for version in {item.version_id: item for item in seed[3]}.values():
            assessment, evidence = d3._decision(
                version, (), CanonicalOutcome.REL_NO_ADEQUATE_PRIOR_MATCH
            )
            authority.retain(
                assessment.canonical_bytes,
                tuple(item.canonical_bytes for item in evidence),
                proof=seed[0][3],
            )
            retained[version.version_id] = assessment.canonical_digest
    finally:
        authority.close()
    connection = sqlite3.connect(seed[1], isolation_level=None)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    return (*seed, retained)


@dataclass(frozen=True)
class _SeedSnapshot:
    database_bytes: bytes
    tail: tuple

    def fork(self) -> _SeedSnapshot:
        return pickle.loads(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    def clone(self, root: Path) -> tuple:
        root.mkdir(mode=0o700)
        database = root / "candidate-authority.sqlite3"
        database.write_bytes(self.database_bytes)
        os.chmod(database, 0o600)
        collaborators = pickle.loads(
            pickle.dumps(self.tail[0], protocol=pickle.HIGHEST_PROTOCOL)
        )
        tail = (collaborators, *self.tail[1:])
        retrieval = tail[0][1]
        retrieval_path = root / "retrieval-context.sqlite3"
        retrieval_path.write_bytes(Path(retrieval._path).read_bytes())
        os.chmod(retrieval_path, 0o600)
        retrieval._path = retrieval_path
        return (tail[0], database, *tail[1:])


def _seed_snapshot(root: Path) -> _SeedSnapshot:
    global_cache = d2._SEED_CACHE
    global_state = d2._seed_cache_state(global_cache)
    d2._SEED_CACHE = None
    try:
        seed = _build_seed(root)
        if len(seed[3]) != 2 or len(seed[8]) != 2:
            raise AssertionError("Candidate adapter seed snapshot shape")
        return _SeedSnapshot(seed[1].read_bytes(), (seed[0], *seed[2:]))
    finally:
        d2._SEED_CACHE = global_cache
        if d2._seed_cache_state(d2._SEED_CACHE) != global_state:
            raise AssertionError("global relationship seed cache changed")


_SHARED_SEED_SNAPSHOT: _SeedSnapshot | None = None


def _named_snapshot(
    location: _Location,
    binding: CandidateUseCollisionBinding,
    *,
    occupied_candidate_id: str | None = None,
) -> tuple[CurrentCollisionEligibilityRequest, object]:
    cache_key = (
        binding.subject_version_id,
        binding.operation,
        occupied_candidate_id,
    )
    with location.collision_lock:
        cached = location.collision_results.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        result = _create_named_snapshot(
            location, binding, occupied_candidate_id=occupied_candidate_id
        )
        location.collision_results[cache_key] = result
        return result  # type: ignore[return-value]


def _create_named_snapshot(
    location: _Location,
    binding: CandidateUseCollisionBinding,
    *,
    occupied_candidate_id: str | None,
) -> tuple[CurrentCollisionEligibilityRequest, object]:
    key = binding.collision_key_digest
    root = location.collision_root / str(uuid.uuid4())
    root.mkdir(parents=True)
    database = authority_database(root)
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM development_candidates_v2")
        if occupied_candidate_id is not None:
            connection.execute(
                "INSERT INTO development_candidates_v2 VALUES(?,?)",
                (occupied_candidate_id, key),
            )
    base = collision_request(
        idempotency_key=binding.idempotency_key,
        generation_id=binding.generation_id,
    )
    named = replace(base, collision_key_digest=key)
    result = executor(root, database).execute(named, authorize(root, named))
    evidence = CurrentCollisionReceiptEvidence(
        named,
        result.receipt.canonical_bytes,
        result.authority_receipt_bytes,
    )
    request = CurrentCollisionEligibilityRequest(binding, named.request_digest)
    context = _trusted_context(evidence)
    snapshot = CurrentCollisionAuthoritySnapshot(evidence, context)
    location.snapshots[request.request_digest] = snapshot
    return request, decide_current_collision_eligibility(
        request=request, evidence=evidence, trusted_context=context
    )


def _enforcer(location: _Location) -> CurrentCollisionEffectEnforcer:
    if not location.snapshots:
        # Establish the exact trusted boundary once; providers remain request keyed.
        version = next(iter(location.subjects.values()))
        binding = CandidateUseCollisionBinding(
            version.hypothesis_id,
            version.version_id,
            version.canonical_digest,
            CandidateUseOperation.ADMIT_NEW_CANDIDATE,
            None,
            "candidate-development",
            _collision_digest(version.version_id),
            "retrieval-generation-v2",
            QUERY_VALID,
            SERVING,
            42,
        )
        _named_snapshot(location, binding)
    context = next(iter(location.snapshots.values())).trusted_context
    return CurrentCollisionEffectEnforcer(
        current_authority_provider=lambda request: location.snapshots[
            request.request_digest
        ],
        trusted_boundary=TrustedCurrentCollisionAuthorityBoundary(
            context.authority_scope_id,
            context.authority_profile_id,
            context.adapter_config_digest,
            context.port_registry_digest,
            context.port_id,
        ),
    )


def _actor_digest(seed: tuple, actor: str) -> str:
    proof = _ACTOR_PROOFS[actor]
    authentication = seed[0][2].authenticate(
        proof, now=UtcTimestamp.parse("2042-01-02T00:00:00.000000Z")
    )
    return digest_bytes(
        canonical_json_bytes(
            {
                "principal_id": authentication.principal_id,
                "credential_binding_digest": authentication.credential_binding_digest,
            }
        )
    )


def _manifest(location: _Location, version, collision, registry_extender=None):
    connection = sqlite3.connect(location.seed[1], isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    commands, schemas = location.seed[4], location.seed[5]
    if registry_extender is not None:
        commands, schemas = registry_extender(commands, schemas)
    commands, schemas = merge_relationship_authority_registries(commands, schemas)
    commands, schemas = merge_lineage_authority_registries(commands, schemas)
    commands, schemas = merge_candidate_authority_registries(commands, schemas)
    lineage = _create_event_hypothesis_lineage_read_port(
        connection,
        retrieval_authority=location.seed[0][1],
        authenticator=location.seed[0][2],
        command_registry=commands,
        payload_schemas=schemas,
        clock=lambda: UtcTimestamp.parse("2042-01-02T00:00:00.000000Z"),
    )
    store = ProposalDispositionStore(
        connection, location.seed[0][1], location.seed[0][2]
    )
    connection.execute("BEGIN IMMEDIATE")
    try:
        snapshot = lineage.require_current_producers_in_transaction(
            version.version_id, proof=location.seed[0][3]
        )
        relationship = lineage.require_retained_relationship_in_transaction(
            location.relationships[version.version_id]
        )
        disposition_ids = tuple(
            sorted(item.disposition_id for item in version.source_bindings)
        )
        dispositions = tuple(
            store.require_current_in_transaction(item, proof=location.seed[0][3])
            for item in disposition_ids
        )
        lead_ids = tuple(
            sorted(
                {NewsLeadId.parse(item.decision_lead_id) for item in dispositions},
                key=str,
            )
        )
        discovery = _create_discovery_governing_producer_read_port(
            connection
        ).require_current_governing_producers(lead_ids)
        return build_candidate_governing_manifest(
            hypothesis_version=version,
            lineage_receipts=snapshot.receipts,
            lineage_initial_heads=snapshot.initial_heads,
            lineage_versions=snapshot.versions,
            lineage_relationship_proofs=snapshot.relationship_proofs,
            dispositions=dispositions,
            leads=tuple(item[0] for item in discovery),
            signals=tuple(item[1] for item in discovery),
            gates=tuple(item[2] for item in discovery),
            relationship=relationship,
            collision=collision,
        )
    finally:
        connection.rollback()
        connection.close()


def _admission(location: _Location, command: WriteCommand):
    record_code = _code(
        _RECORDS, (command.record_id, command.scalar_columns.get("value")), "record"
    )
    base = _generic(
        command.record_id, command.cas_predecessor, command.scalar_columns.get("value")
    )
    if (
        command.canonical_bytes != base.canonical_bytes
        or command.scalar_columns != base.scalar_columns
        or command.identity_columns != base.identity_columns
        or command.linked_rows != base.linked_rows
    ):
        raise CandidateContractError("unsupported conformance representation")
    request_code = _code(_REQUESTS, command.request, "request")
    idempotency_code = _code(_IDEMPOTENCIES, command.idempotency, "idempotency")
    predecessor_code = _code(_PREDECESSORS, command.cas_predecessor, "CAS predecessor")
    upstream_code = _code(_UPSTREAMS, command.required_upstream_heads, "upstream")
    actor_code = _code(tuple(_ACTOR_PROOFS), command.actor, "actor")
    version = location.subjects[command.record_id]
    binding = CandidateUseCollisionBinding(
        version.hypothesis_id,
        version.version_id,
        version.canonical_digest,
        CandidateUseOperation.ADMIT_NEW_CANDIDATE,
        None,
        "candidate-development",
        _collision_digest(version.version_id),
        "retrieval-generation-v2",
        QUERY_VALID,
        SERVING,
        42,
    )
    collision_request_value, collision = _named_snapshot(location, binding)
    manifest = _manifest(location, version, collision)
    codes = (
        record_code,
        request_code,
        idempotency_code,
        predecessor_code,
        upstream_code,
        actor_code,
    )
    request = CandidateAdmissionRequest(
        _request_uuid(codes),
        _actor_digest(location.seed, command.actor),
        "candidate:" + ":".join(str(item) for item in codes),
        None,
        None,
        0,
        manifest.semantic_scope_digest,
        collision_request_value.request_digest,
        manifest.governing_state_binding.canonical_digest,
        None,
    )
    admission = evaluate_candidate_admission(
        request=request,
        manifest=manifest,
        collision=collision,
        current_version=None,
        governing_state=CandidateGoverningState(
            CandidateGoverningStateStatus.COMPLETE,
            manifest.governing_state_binding,
        ),
    )
    return admission, collision_request_value


def _advance_record_one(location: _Location, registry_extender=None):
    previous = location.subjects["record-1"]
    connection = sqlite3.connect(location.seed[1], isolation_level=None)
    try:
        fixture = (connection, *location.seed[0][1:])
        proposal, dispositions = location.seed[7]["authority"]
        upstream = d1._open(fixture)
        try:
            advanced = upstream.retain(
                proposal,
                dispositions,
                proof=location.seed[0][3],
                expected_target_version=previous,
            )
        finally:
            upstream.close()
    finally:
        connection.close()
    subject = HypothesisVersionBinding.from_version(advanced)
    comparator = HypothesisVersionBinding.from_version(previous)
    evidence = ComparatorEvidence(subject, comparator, 59, 0, 80, 0, 0)
    assessment = assess_relationships(
        subject, ComparatorSetManifest.complete((comparator,)), (evidence,)
    )
    assert assessment.decision is CanonicalOutcome.REL_DEVELOPMENT_OF
    args = _collaborators(location.seed)
    if registry_extender is not None:
        args["command_registry"], args["payload_schemas"] = registry_extender(
            args["command_registry"], args["payload_schemas"]
        )
    commands, schemas = merge_relationship_authority_registries(
        args["command_registry"], args["payload_schemas"]
    )
    commands, schemas = merge_lineage_authority_registries(commands, schemas)
    args["command_registry"], args["payload_schemas"] = (
        merge_candidate_authority_registries(commands, schemas)
    )
    relationship = open_event_hypothesis_relationship_authority(**args)
    try:
        relationship.retain(
            assessment.canonical_bytes,
            (evidence.canonical_bytes,),
            proof=location.seed[0][3],
        )
    finally:
        relationship.close()
    location.subjects["record-1"] = advanced
    location.relationships[advanced.version_id] = assessment.canonical_digest
    return previous, advanced


class _Handle:
    def __init__(self, location: _Location, registry_extender=None) -> None:
        self.location = location
        self.authority = None
        self.open_error: Exception | None = None
        try:
            args = _collaborators(location.seed)
            if registry_extender is not None:
                args["command_registry"], args["payload_schemas"] = registry_extender(
                    args["command_registry"], args["payload_schemas"]
                )
            self.authority = _open_unlocked_story_candidate_authority_for_test(
                **args, collision_enforcer=_enforcer(location)
            )
        except Exception as exc:
            if not isinstance(exc, (AuthoritySchemaError, CandidateContractError)):
                raise
            self.open_error = exc

    def _opened(self):
        if self.open_error is not None:
            raise IntegrityViolation(
                "Candidate authority open failed"
            ) from self.open_error
        assert self.authority is not None
        return self.authority

    def _row(self, record_id: str, connection=None):
        owned = connection is None
        connection = connection or sqlite3.connect(self.location.seed[1])
        try:
            version = self.location.subjects[record_id]
            return connection.execute(
                "SELECT admission_digest,version_id FROM story_candidate_admission_receipts_v2 "
                "WHERE hypothesis_version_id=?",
                (version.version_id,),
            ).fetchone()
        finally:
            if owned:
                connection.close()

    def _decode(self, version_id: str, connection=None) -> WriteCommand:
        owned = connection is None
        connection = connection or sqlite3.connect(self.location.seed[1])
        try:
            row = connection.execute(
                "SELECT r.request_id,r.idempotency_key,a.principal_id "
                "FROM story_candidate_admission_receipts_v2 r "
                "JOIN ledger_events e ON e.event_id=r.authority_event_id "
                "JOIN authority_commands c ON c.command_id=e.command_id "
                "JOIN authentication_contexts a ON a.authentication_context_id="
                "c.authentication_context_id WHERE r.version_id=?",
                (version_id,),
            ).fetchone()
            if row is None:
                raise CandidateContractError("unknown conformance Candidate")
            codes = _request_codes(str(row[0]))
            encoded = tuple(int(item) for item in str(row[1]).split(":")[1:])
            if codes != encoded or len(codes) != 6:
                raise CandidateContractError(
                    "Candidate conformance request binding differs"
                )
            record_id, value = _RECORDS[codes[0]]
            actor = _PRINCIPAL_ACTORS[str(row[2])]
            return replace(
                _generic(record_id, _PREDECESSORS[codes[3]], value),
                request=_REQUESTS[codes[1]],
                idempotency=_IDEMPOTENCIES[codes[2]],
                actor=actor,
                required_upstream_heads=_UPSTREAMS[codes[4]],
            )
        except (IndexError, KeyError, ValueError) as exc:
            raise CandidateContractError(
                "Candidate conformance binding differs"
            ) from exc
        finally:
            if owned:
                connection.close()

    def submit(self, command: WriteCommand, *, lose_response: bool = False):
        existing = self._row(command.record_id)
        if existing is not None:
            try:
                retained_command = self._decode(str(existing[1]))
                if retained_command != command:
                    raise BindingConflict(command.record_id)
                connection = sqlite3.connect(self.location.seed[1])
                try:
                    row = connection.execute(
                        "SELECT admission_bytes,collision_request_bytes FROM "
                        "story_candidate_admission_receipts_v2 WHERE version_id=?",
                        (existing[1],),
                    ).fetchone()
                finally:
                    connection.close()
                admission = CandidateAdmission.from_canonical_bytes(bytes(row[0]))
                collision = CurrentCollisionEligibilityRequest.from_mapping(
                    json.loads(bytes(row[1]))
                )
                retained = self._opened().admit(
                    admission.canonical_bytes,
                    collision_request=collision,
                    proof=_ACTOR_PROOFS[command.actor],
                )
            except BindingConflict:
                raise
            except Exception as exc:
                raise IntegrityViolation(command.record_id) from exc
            if lose_response:
                raise LostResponse(command.record_id)
            return AuthorityValue.from_command(self._decode(retained.version_id))
        try:
            admission, collision = _admission(self.location, command)
            retained = self._opened().admit(
                admission.canonical_bytes,
                collision_request=collision,
                proof=_ACTOR_PROOFS[command.actor],
            )
        except Exception as exc:
            row = self._row(command.record_id)
            if row is not None:
                try:
                    self._opened().load_version(str(row[1]))
                except Exception as integrity_exc:
                    raise IntegrityViolation(command.record_id) from integrity_exc
                raise BindingConflict(command.record_id) from exc
            if _database_was_locked(exc):
                raise BindingConflict(command.record_id) from exc
            if isinstance(exc, (sqlite3.OperationalError, CandidateContractError)):
                raise BindingConflict(command.record_id) from exc
            raise IntegrityViolation(command.record_id) from exc
        if lose_response:
            raise LostResponse(command.record_id)
        return AuthorityValue.from_command(self._decode(retained.version_id))

    def observe(self, record_id: str):
        row = self._row(record_id)
        if row is None:
            return None
        try:
            self._opened().load_version(str(row[1]))
            command = self._decode(str(row[1]))
        except Exception as exc:
            raise IntegrityViolation(record_id) from exc
        return (
            StoredAuthorityState.from_command(command)
            if command.record_id == record_id
            else None
        )

    def history(self):
        connection = sqlite3.connect(self.location.seed[1])
        try:
            rows = connection.execute(
                "SELECT version_id FROM story_candidate_admission_receipts_v2 "
                "ORDER BY recorded_at,admission_digest"
            ).fetchall()
        finally:
            connection.close()
        try:
            values = []
            for row in rows:
                self._opened().load_version(str(row[0]))
                values.append(AuthorityValue.from_command(self._decode(str(row[0]))))
            return tuple(values)
        except Exception as exc:
            raise IntegrityViolation("history") from exc

    list_history = history

    def read(self, record_id: str):
        row = self._row(record_id)
        if row is None:
            raise KeyError(record_id)
        state = self.observe(record_id)
        if state is None:
            raise KeyError(record_id)
        return AuthorityValue.from_command(self._decode(str(row[1])))

    def set_upstream_head(self, authority: str, value: str) -> None:
        if value.endswith("head-1") or self._row("record-1") is None:
            return
        subject = self.location.subjects["record-1"]
        connection = sqlite3.connect(self.location.seed[1], isolation_level=None)
        try:
            if authority == "authority":
                fixture = (connection, *self.location.seed[0][1:])
                proposal, dispositions = self.location.seed[7]["authority"]
                upstream = d1._open(fixture)
                try:
                    upstream.retain(
                        proposal,
                        dispositions,
                        proof=self.location.seed[0][3],
                        expected_target_version=subject,
                    )
                finally:
                    upstream.close()
            elif authority == "policy":
                item, current = self.location.seed[7]["policy"]
                successor = replace(
                    d1.work_item_helpers._version(item, 2), retrieval=current.retrieval
                )
                from newsroom.increment6.work_items import TriageWorkItemStore

                TriageWorkItemStore(
                    connection, self.location.seed[0][1]
                ).append_version(
                    current.version_id, current.canonical_digest, successor
                )
            else:
                raise IntegrityViolation("unsupported upstream authority")
        finally:
            connection.close()

    def current_use(self, record_id: str):
        row = self._row(record_id)
        if row is None:
            raise KeyError(record_id)
        connection = sqlite3.connect(self.location.seed[1])
        try:
            candidate_id = str(
                connection.execute(
                    "SELECT candidate_id FROM story_candidate_admission_receipts_v2 "
                    "WHERE version_id=?",
                    (row[1],),
                ).fetchone()[0]
            )
        finally:
            connection.close()
        version = self.location.subjects[record_id]
        binding = CandidateUseCollisionBinding(
            version.hypothesis_id,
            version.version_id,
            version.canonical_digest,
            CandidateUseOperation.USE_CURRENT_CANDIDATE,
            candidate_id,
            "candidate-development",
            _collision_digest(version.version_id),
            "retrieval-generation-v2",
            QUERY_VALID,
            SERVING,
            42,
        )
        request, _ = _named_snapshot(
            self.location, binding, occupied_candidate_id=candidate_id
        )
        try:
            self._opened().current(
                candidate_id, collision_request=request, proof=self.location.seed[0][3]
            )
        except Exception as exc:
            raise IntegrityViolation(record_id) from exc
        return AuthorityValue.from_command(self._decode(str(row[1])))

    def tamper(self, record_id: str, kind: TamperKind) -> None:
        row = self._row(record_id)
        assert row is not None
        connection = sqlite3.connect(self.location.seed[1], isolation_level=None)
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute("DROP TRIGGER IF EXISTS immutable_candidate_receipt")
        column, value = {
            TamperKind.CANONICAL: ("admission_bytes", b"{}"),
            TamperKind.SCALAR: ("version_ordinal", 7),
            TamperKind.IDENTITY: ("semantic_scope_digest", "sha256:" + "0" * 64),
            TamperKind.LINKED_ROW: ("disposition_ids_bytes", b"[]"),
            TamperKind.DIGEST: ("version_digest", "sha256:" + "0" * 64),
            TamperKind.PROVENANCE: ("actor_identity_digest", "sha256:" + "0" * 64),
            TamperKind.OFFLINE_REWRITE: (
                "authority_aggregate_id",
                str(uuid.uuid4()),
            ),
        }[kind]
        connection.execute(
            f"UPDATE story_candidate_admission_receipts_v2 SET {column}=? "
            "WHERE admission_digest=?",
            (value, row[0]),
        )
        connection.close()

    def rollback_scope(self, operation: Callable[[RollbackScope], None]) -> None:
        store = self._opened()._store
        prepared = {
            record_id: _admission(self.location, _generic(record_id))
            for record_id in ("record-rollback-normal", "record-rollback-abort")
        }

        class Scope:
            def submit(_, command):
                try:
                    admission, collision = prepared[command.record_id]
                except KeyError as exc:
                    raise CandidateContractError(
                        "unsupported rollback conformance record"
                    ) from exc
                if admission.request.request_id != _request_uuid(
                    (
                        _code(
                            _RECORDS,
                            (command.record_id, command.scalar_columns.get("value")),
                            "record",
                        ),
                        _code(_REQUESTS, command.request, "request"),
                        _code(_IDEMPOTENCIES, command.idempotency, "idempotency"),
                        _code(
                            _PREDECESSORS, command.cas_predecessor, "CAS predecessor"
                        ),
                        _code(_UPSTREAMS, command.required_upstream_heads, "upstream"),
                        _code(tuple(_ACTOR_PROOFS), command.actor, "actor"),
                    )
                ):
                    raise CandidateContractError("rollback command binding differs")
                retained = transaction.admit(
                    admission.canonical_bytes,
                    collision_request=collision,
                    proof=_ACTOR_PROOFS[command.actor],
                )
                return AuthorityValue.from_command(
                    self._decode(retained.version_id, transaction._connection)
                )

            def observe(_, record_id):
                row = self._row(record_id, transaction._connection)
                if row is None:
                    return None
                command = self._decode(str(row[1]), transaction._connection)
                return StoredAuthorityState.from_command(command)

            def history(_):
                rows = transaction._connection.execute(
                    "SELECT version_id FROM story_candidate_admission_receipts_v2 "
                    "ORDER BY recorded_at,admission_digest"
                ).fetchall()
                return tuple(
                    AuthorityValue.from_command(
                        self._decode(str(row[0]), transaction._connection)
                    )
                    for row in rows
                )

        def inspect(scope):
            nonlocal transaction
            transaction = scope
            operation(Scope())

        transaction = None
        store.rollback_scope(inspect)

    def close(self) -> None:
        if self.authority is not None:
            self.authority.close()


class _Adapter:
    name = "story-candidate-v24-real"
    applicability: ClassVar = {
        case: Applicability.required() for case in CASE_INVENTORY
    }

    def __init__(self, root: Path) -> None:
        self.root = root
        self.snapshot = _SHARED_SEED_SNAPSHOT or _seed_snapshot(
            root / "capacity-2-seed"
        )

    def create_location(self) -> _Location:
        root = self.root / str(uuid.uuid4())
        return _Location(self.snapshot.clone(root), root / "collision")

    def open_handle(self, location: _Location, *, migrate: bool = False) -> _Handle:
        if migrate:
            _migrate_to_current(location.seed[1])
        return _Handle(location)


@pytest.mark.parametrize("case", CASE_INVENTORY, ids=lambda case: case.value)
def test_real_candidate_store_passes_required_conformance_case(
    tmp_path: Path, case: CaseId
) -> None:
    adapter = _Adapter(tmp_path)
    assert adapter.applicability[case] == Applicability.required()
    _SCENARIOS[case](adapter)


@pytest.mark.parametrize("mismatch", ("operation", "current-subject-key"))
def test_wrong_primary_collision_binding_rejects_before_candidate_producers(
    tmp_path: Path, mismatch: str
) -> None:
    adapter = _Adapter(tmp_path)
    location = adapter.create_location()
    handle = adapter.open_handle(location)
    command = _generic("record-1")
    admission, collision = _admission(location, command)
    binding = collision.binding
    if mismatch == "operation":
        binding = replace(
            binding,
            operation=CandidateUseOperation.USE_CURRENT_CANDIDATE,
            expected_candidate_id=str(uuid.uuid4()),
        )
    else:
        binding = replace(
            binding,
            collision_key_digest=digest("wrong-current-subject-key"),
        )
    divergent = replace(collision, binding=binding)
    store = handle._opened()._store
    producer_calls: list[object] = []
    original = store._producers
    store._producers = lambda *args, **kwargs: producer_calls.append((args, kwargs))
    connection = sqlite3.connect(location.seed[1])
    try:
        before = connection.execute(
            "SELECT count(*) FROM story_candidate_admission_receipts_v2"
        ).fetchone()
        with pytest.raises(CandidateContractError):
            handle._opened().admit(
                admission.canonical_bytes,
                collision_request=divergent,
                proof=location.seed[0][3],
            )
        after = connection.execute(
            "SELECT count(*) FROM story_candidate_admission_receipts_v2"
        ).fetchone()
    finally:
        store._producers = original
        connection.close()
        handle.close()
    assert producer_calls == []
    assert before == after == (0,)


def test_retained_collision_key_rejects_different_scope_before_producers_or_effect(
    tmp_path: Path,
) -> None:
    adapter = _Adapter(tmp_path)
    location = adapter.create_location()
    handle = adapter.open_handle(location)
    handle.submit(_generic("record-1"))
    row = handle._row("record-1")
    assert row is not None
    retained = handle._opened().load_version(str(row[1]))
    version = location.subjects["record-2"]
    binding = CandidateUseCollisionBinding(
        version.hypothesis_id,
        version.version_id,
        version.canonical_digest,
        CandidateUseOperation.ADMIT_NEW_CANDIDATE,
        None,
        retained.governing_manifest.collision_namespace,
        retained.governing_manifest.collision_key_digest,
        "retrieval-generation-v2",
        QUERY_VALID,
        SERVING,
        42,
    )
    collision_request_value, collision = _named_snapshot(location, binding)
    manifest = _manifest(location, version, collision)
    request = CandidateAdmissionRequest(
        str(uuid.uuid4()),
        _actor_digest(location.seed, "actor-1"),
        "candidate:retained-collision-key",
        None,
        None,
        0,
        manifest.semantic_scope_digest,
        collision_request_value.request_digest,
        manifest.governing_state_binding.canonical_digest,
        None,
    )
    admission = evaluate_candidate_admission(
        request=request,
        manifest=manifest,
        collision=collision,
        current_version=None,
        governing_state=CandidateGoverningState(
            CandidateGoverningStateStatus.COMPLETE,
            manifest.governing_state_binding,
        ),
    )
    assert admission.reason is CandidateAdmissionReason.NEW_CANDIDATE_PRE_EFFECT
    store = handle._opened()._store
    producer_calls: list[object] = []
    effect_calls: list[object] = []
    original_producers = store._producers
    original_provider = store._collision._current_authority_provider

    def observed_producers(*args, **kwargs):
        producer_calls.append((args, kwargs))
        return original_producers(*args, **kwargs)

    store._producers = observed_producers
    store._collision._current_authority_provider = lambda request: (
        effect_calls.append(request) or original_provider(request)
    )
    try:
        with pytest.raises(CandidateContractError):
            handle._opened().admit(
                admission.canonical_bytes,
                collision_request=collision_request_value,
                proof=location.seed[0][3],
            )
    finally:
        store._producers = original_producers
        store._collision._current_authority_provider = original_provider
        handle.close()
    assert producer_calls == effect_calls == []
    with sqlite3.connect(location.seed[1]) as connection:
        assert connection.execute(
            "SELECT count(*) FROM story_candidate_admission_receipts_v2"
        ).fetchone() == (1,)


@pytest.mark.parametrize("mismatch", ("operation", "subject", "key"))
def test_current_rejects_wrong_collision_binding_before_producers_or_effect(
    tmp_path: Path, mismatch: str
) -> None:
    adapter = _Adapter(tmp_path)
    location = adapter.create_location()
    handle = adapter.open_handle(location)
    command = _generic("record-1")
    handle.submit(command)
    row = handle._row(command.record_id)
    assert row is not None
    retained = handle._opened().load_version(str(row[1]))
    manifest = retained.governing_manifest
    binding = CandidateUseCollisionBinding(
        manifest.hypothesis_id,
        manifest.hypothesis_version_id,
        manifest.hypothesis_version_digest,
        CandidateUseOperation.USE_CURRENT_CANDIDATE,
        retained.candidate_id,
        manifest.collision_namespace,
        manifest.collision_key_digest,
        "retrieval-generation-v2",
        QUERY_VALID,
        SERVING,
        42,
    )
    if mismatch == "operation":
        binding = replace(
            binding,
            operation=CandidateUseOperation.ADMIT_NEW_CANDIDATE,
            expected_candidate_id=None,
        )
    elif mismatch == "subject":
        other = location.subjects["record-2"]
        binding = replace(
            binding,
            subject_id=other.hypothesis_id,
            subject_version_id=other.version_id,
            subject_version_digest=other.canonical_digest,
        )
    else:
        binding = replace(binding, collision_key_digest=digest("wrong-current-key"))
    collision_request_value, _ = _named_snapshot(
        location, binding, occupied_candidate_id=retained.candidate_id
    )
    store = handle._opened()._store
    producer_calls: list[object] = []
    effect_calls: list[object] = []
    original_producers = store._producers
    original_provider = store._collision._current_authority_provider
    store._producers = lambda *args, **kwargs: producer_calls.append((args, kwargs))
    store._collision._current_authority_provider = lambda request: (
        effect_calls.append(request) or original_provider(request)
    )
    try:
        with pytest.raises(CandidateContractError):
            handle._opened().current(
                retained.candidate_id,
                collision_request=collision_request_value,
                proof=location.seed[0][3],
            )
    finally:
        store._producers = original_producers
        store._collision._current_authority_provider = original_provider
        handle.close()
    assert producer_calls == effect_calls == []
    with sqlite3.connect(location.seed[1]) as connection:
        assert connection.execute(
            "SELECT count(*) FROM story_candidate_admission_receipts_v2"
        ).fetchone() == (1,)


def test_real_successor_commits_then_history_current_and_reopen_retain_both_versions(
    tmp_path: Path,
) -> None:
    adapter = _Adapter(tmp_path)
    location = adapter.create_location()
    first_handle = adapter.open_handle(location)
    first_admission, first_collision = _admission(location, _generic("record-1"))
    first = first_handle._opened().admit(
        first_admission.canonical_bytes,
        collision_request=first_collision,
        proof=location.seed[0][3],
    )
    first_handle.close()

    previous, advanced = _advance_record_one(location)
    assert advanced.previous_version_id == previous.version_id
    first_manifest = first.governing_manifest
    binding = CandidateUseCollisionBinding(
        advanced.hypothesis_id,
        advanced.version_id,
        advanced.canonical_digest,
        CandidateUseOperation.USE_CURRENT_CANDIDATE,
        first.candidate_id,
        first_manifest.collision_namespace,
        first_manifest.collision_key_digest,
        "retrieval-generation-v2",
        QUERY_VALID,
        SERVING,
        42,
    )
    collision_request_value, collision = _named_snapshot(
        location, binding, occupied_candidate_id=first.candidate_id
    )
    manifest = _manifest(location, advanced, collision)
    request = CandidateAdmissionRequest(
        str(uuid.uuid4()),
        _actor_digest(location.seed, "actor-1"),
        "candidate:successor-record-1",
        first.version_id,
        first.canonical_digest,
        first.ordinal,
        manifest.semantic_scope_digest,
        collision_request_value.request_digest,
        manifest.governing_state_binding.canonical_digest,
        None,
    )
    admission = evaluate_candidate_admission(
        request=request,
        manifest=manifest,
        collision=collision,
        current_version=first,
        governing_state=CandidateGoverningState(
            CandidateGoverningStateStatus.COMPLETE,
            manifest.governing_state_binding,
        ),
    )
    assert admission.reason is CandidateAdmissionReason.SUCCESSOR_VERSION_PRE_EFFECT

    second_handle = adapter.open_handle(location)
    second = second_handle._opened().admit(
        admission.canonical_bytes,
        collision_request=collision_request_value,
        proof=location.seed[0][3],
    )
    try:
        assert second.ordinal == 2
        assert second.previous_version_id == first.version_id
        assert second_handle._opened().load_version(first.version_id) == first
        assert second_handle._opened().versions(first.candidate_id) == (first, second)
        assert (
            second_handle._opened().current(
                first.candidate_id,
                collision_request=collision_request_value,
                proof=location.seed[0][3],
            )
            == second
        )
    finally:
        second_handle.close()

    reopened = adapter.open_handle(location)
    try:
        assert reopened._opened().load_version(first.version_id) == first
        assert reopened._opened().versions(first.candidate_id) == (first, second)
        assert (
            reopened._opened().current(
                first.candidate_id,
                collision_request=collision_request_value,
                proof=location.seed[0][3],
            )
            == second
        )
    finally:
        reopened.close()


@pytest.mark.parametrize("damage", ("upstream", "ancestor"))
def test_retained_successor_read_uses_only_exact_historical_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    from newsroom.authority import _event_hypothesis_lineage_system as lineage_system
    from newsroom.authority import (
        _event_hypothesis_relationship_system as relationship_system,
    )
    from newsroom.authority import _event_hypothesis_system as hypothesis_system

    adapter = _Adapter(tmp_path)
    location = adapter.create_location()
    first_handle = adapter.open_handle(location)
    first_admission, first_collision = _admission(location, _generic("record-1"))
    first = first_handle._opened().admit(
        first_admission.canonical_bytes,
        collision_request=first_collision,
        proof=location.seed[0][3],
    )
    other_admission, other_collision = _admission(location, _generic("record-2"))
    other = first_handle._opened().admit(
        other_admission.canonical_bytes, collision_request=other_collision,
        proof=location.seed[0][3],
    )
    first_handle.close()
    _, advanced = _advance_record_one(location)
    binding = CandidateUseCollisionBinding(
        advanced.hypothesis_id,
        advanced.version_id,
        advanced.canonical_digest,
        CandidateUseOperation.USE_CURRENT_CANDIDATE,
        first.candidate_id,
        first.governing_manifest.collision_namespace,
        first.governing_manifest.collision_key_digest,
        "retrieval-generation-v2",
        QUERY_VALID,
        SERVING,
        42,
    )
    collision_request_value, collision = _named_snapshot(
        location, binding, occupied_candidate_id=first.candidate_id
    )
    manifest = _manifest(location, advanced, collision)
    admission = evaluate_candidate_admission(
        request=CandidateAdmissionRequest(
            str(uuid.uuid4()),
            _actor_digest(location.seed, "actor-1"),
            "candidate:bounded-successor-record-1",
            first.version_id,
            first.canonical_digest,
            first.ordinal,
            manifest.semantic_scope_digest,
            collision_request_value.request_digest,
            manifest.governing_state_binding.canonical_digest,
            None,
        ),
        manifest=manifest,
        collision=collision,
        current_version=first,
        governing_state=CandidateGoverningState(
            CandidateGoverningStateStatus.COMPLETE,
            manifest.governing_state_binding,
        ),
    )
    handle = adapter.open_handle(location)
    second = handle._opened().admit(
        admission.canonical_bytes,
        collision_request=collision_request_value,
        proof=location.seed[0][3],
    )
    handle.close()

    connection = sqlite3.connect(location.seed[1], isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    port = _create_story_candidate_read_port(
        connection,
        retrieval_authority=location.seed[0][1],
        authenticator=location.seed[0][2],
        command_registry=location.seed[4],
        payload_schemas=location.seed[5],
        clock=lambda: UtcTimestamp.parse("2042-01-02T00:00:00.000000Z"),
    )

    def global_history(*_args, **_kwargs):
        raise AssertionError("retained Candidate read scanned unrelated history")

    monkeypatch.setattr(lineage_system._LineageStore, "_verify", global_history)
    monkeypatch.setattr(
        relationship_system,
        "_verify_relationship_reads_in_transaction",
        global_history,
    )
    monkeypatch.setattr(hypothesis_system._HypothesisStore, "_verify", global_history)
    monkeypatch.setattr(
        hypothesis_system, "_VERIFY_DISPOSITION_INTEGRITY", global_history
    )
    with pytest.raises(ValueError, match="active checked connection"):
        port.require_current_head_in_transaction(
            first.candidate_id, proof=location.seed[0][3]
        )
    assert not connection.in_transaction
    connection.execute("BEGIN")
    try:
        assert (
            port.require_retained_candidate_in_transaction(first.candidate_id)
            .candidate_id
            == first.candidate_id
        )
        from newsroom.authority.story_candidate_system import _CandidateStore
        original_verify_row = _CandidateStore._verify_row
        checked_versions = []

        def counted_row(store, digest):
            result = original_verify_row(store, digest)
            checked_versions.append(result[2].version_id)
            return result

        with monkeypatch.context() as selected:
            selected.setattr(_CandidateStore, "_verify_row", counted_row)
            assert port.require_retained_version_in_transaction(first.version_id) == first
        assert set(checked_versions) == {first.version_id, second.version_id}
        assert len(checked_versions) == 2
        assert other.version_id not in checked_versions
        assert port.require_retained_version_in_transaction(second.version_id) == second
        assert (
            port.require_current_head_in_transaction(
                first.candidate_id, proof=location.seed[0][3]
            )
            == second
        )
    finally:
        connection.execute("ROLLBACK")
    if damage == "ancestor":
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("DROP TRIGGER retained_candidate_receipt")
        connection.execute("DELETE FROM story_candidate_admission_receipts_v2 WHERE version_id=?",
                           (first.version_id,))
        connection.execute("PRAGMA foreign_keys=ON")
        requested = second.version_id
    else:
        trigger = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name="
            "'immutable_event_hypothesis_version_update'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER immutable_event_hypothesis_version_update")
        connection.execute(
            "UPDATE event_hypothesis_versions_v2 SET canonical_bytes=? WHERE version_id=?",
            (b"{}", first.governing_manifest.hypothesis_version_id),
        )
        connection.execute(trigger)
        requested = first.version_id
    connection.execute("BEGIN")
    try:
        with pytest.raises(ValueError, match="Candidate|Hypothesis|relationship"):
            port.require_retained_version_in_transaction(requested)
    finally:
        connection.execute("ROLLBACK")
        connection.close()


def test_exact_candidate_manifest_replays_bound_lineage_and_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from newsroom.authority import _event_hypothesis_lineage_system as lineage_system
    from newsroom.tests import test_increment6d3_lineage_store as lineage_store

    seed, args, receipt = lineage_store._seed(tmp_path)
    authority = open_event_hypothesis_lineage_authority(**args)
    try:
        authority.retain(receipt.canonical_bytes, proof=seed[0][3])
    finally:
        authority.close()
    commands, schemas = merge_relationship_authority_registries(
        args["command_registry"], args["payload_schemas"]
    )
    commands, schemas = merge_lineage_authority_registries(commands, schemas)
    connection = sqlite3.connect(seed[1], isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    port = _create_event_hypothesis_lineage_read_port(
        connection,
        retrieval_authority=args["retrieval_authority"],
        authenticator=args["authenticator"],
        command_registry=commands,
        payload_schemas=schemas,
        clock=args["clock"],
    )
    output = seed[3][0]
    assessment_digest = receipt.relationships[0].assessment_digest
    monkeypatch.setattr(
        lineage_system._LineageStore,
        "_history_rows",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("bound Candidate lineage scanned unrelated history")
        ),
    )
    connection.execute("BEGIN")
    try:
        retained = port._require_exact_candidate_manifest_in_transaction(
            output.version_id,
            assessment_digest,
            (receipt.canonical_digest,),
            1,
        )
        assert retained.assessment.canonical_digest == assessment_digest
        with pytest.raises(ValueError, match="lineage.*head"):
            port._require_exact_candidate_manifest_in_transaction(
                output.version_id,
                assessment_digest,
                (receipt.canonical_digest,),
                0,
            )
    finally:
        connection.execute("ROLLBACK")
    proposal, dispositions = seed[7]["authority"]
    upstream = d2.hypothesis_helpers._open((connection, *seed[0][1:]))
    try:
        successor = upstream.retain(
            proposal,
            dispositions,
            proof=seed[0][3],
            expected_target_version=output,
        )
        assert successor.previous_version_id == output.version_id
    finally:
        upstream.close()
    connection.execute("BEGIN")
    try:
        assert (
            port._require_exact_candidate_manifest_in_transaction(
                output.version_id,
                assessment_digest,
                (receipt.canonical_digest,),
                1,
            ).assessment.canonical_digest
            == assessment_digest
        )
    finally:
        connection.execute("ROLLBACK")
    trigger = connection.execute(
        "SELECT sql FROM sqlite_master WHERE name="
        "'immutable_event_hypothesis_lineage_update'"
    ).fetchone()[0]
    connection.execute("DROP TRIGGER immutable_event_hypothesis_lineage_update")
    connection.execute(
        "UPDATE event_hypothesis_lineage SET receipt_bytes=? WHERE lineage_id=?",
        (b"{}", receipt.lineage_id),
    )
    connection.execute(trigger)
    connection.execute("BEGIN")
    try:
        with pytest.raises(ValueError, match="lineage"):
            port._require_exact_candidate_manifest_in_transaction(
                output.version_id,
                assessment_digest,
                (receipt.canonical_digest,),
                1,
            )
    finally:
        connection.execute("ROLLBACK")
        connection.close()


def test_candidate_read_rejects_exact_authority_payload_tamper(tmp_path: Path) -> None:
    adapter = _Adapter(tmp_path)
    location = adapter.create_location()
    handle = adapter.open_handle(location)
    try:
        handle.submit(_generic("record-1"))
        row = handle._row("record-1")
        assert row is not None
        version_id = str(row[1])
        store = handle._opened()._store
        store._execute_test_sql("DROP TRIGGER immutable_authority_payloads_update")
        store._execute_test_sql(
            "UPDATE authority_payloads SET payload_bytes=? WHERE payload_id=("
            "SELECT e.payload_id FROM story_candidate_admission_receipts_v2 r "
            "JOIN ledger_events e ON e.event_id=r.authority_event_id "
            "WHERE r.version_id=?)",
            (b"{}", version_id),
        )

        with pytest.raises(AuthorityPersistenceError, match="payload digest"):
            handle._opened().load_version(version_id)
    finally:
        handle.close()


def test_candidate_reopen_rejects_unrelated_authority_corruption(
    tmp_path: Path,
) -> None:
    adapter = _Adapter(tmp_path)
    location = adapter.create_location()
    handle = adapter.open_handle(location)
    handle.submit(_generic("record-1"))
    store = handle._opened()._store
    trigger_sql = str(
        store._connection.execute(
            "SELECT sql FROM sqlite_schema WHERE type='trigger' AND name=?",
            ("immutable_authority_payloads_update",),
        ).fetchone()[0]
    )
    store._execute_test_sql("DROP TRIGGER immutable_authority_payloads_update")
    store._execute_test_sql(
        "UPDATE authority_payloads SET payload_bytes=? WHERE payload_id=("
        "SELECT p.payload_id FROM authority_payloads p "
        "WHERE p.payload_bytes IS NOT NULL AND NOT EXISTS("
        "SELECT 1 FROM story_candidate_admission_receipts_v2 r "
        "JOIN ledger_events e ON e.event_id=r.authority_event_id "
        "WHERE e.payload_id=p.payload_id) LIMIT 1)",
        (b"{}",),
    )
    store._execute_test_sql(trigger_sql)
    handle.close()

    with pytest.raises(AuthorityPersistenceError, match="payload digest"):
        adapter.open_handle(location)._opened()


def test_candidate_read_verifies_shared_upstream_once_for_all_relationships(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from newsroom.authority import _event_hypothesis_system as system
    from newsroom.increment6 import dispositions as disposition_system

    adapter = _Adapter(tmp_path)
    location = adapter.create_location()
    handle = adapter.open_handle(location)
    try:
        handle.submit(_generic("record-1"))
        handle.submit(_generic("record-2"))
        first = handle._row("record-1")
        assert first is not None

        disposition_calls = 0
        retrieval_calls = 0
        original_dispositions = system._VERIFY_DISPOSITION_INTEGRITY
        original_retrieval = disposition_system._RETRIEVAL_VERIFY_RETAINED

        def counted_dispositions(store):
            nonlocal disposition_calls
            disposition_calls += 1
            return original_dispositions(store)

        def counted_retrieval(*args: object) -> None:
            nonlocal retrieval_calls
            retrieval_calls += 1
            original_retrieval(*args)

        with sqlite3.connect(location.seed[1]) as connection:
            expected_retrievals = connection.execute(
                "SELECT count(DISTINCT work_item_version_id || ':' || proposal_id) "
                "FROM triage_proposal_dispositions"
            ).fetchone()[0]
        monkeypatch.setattr(
            system, "_VERIFY_DISPOSITION_INTEGRITY", counted_dispositions
        )
        monkeypatch.setattr(
            disposition_system, "_RETRIEVAL_VERIFY_RETAINED", counted_retrieval
        )
        handle._opened().load_version(str(first[1]))
        assert disposition_calls == 1
        assert retrieval_calls == expected_retrievals
    finally:
        handle.close()


def test_candidate_admission_uses_only_exact_producer_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from newsroom.authority import _event_hypothesis_lineage_system as lineage_system
    from newsroom.authority import (
        _event_hypothesis_relationship_system as relationship_system,
    )
    from newsroom.authority import _event_hypothesis_system as system

    adapter = _Adapter(tmp_path)
    location = adapter.create_location()
    handle = adapter.open_handle(location)
    try:
        command = _generic("record-a")
        admission, collision = _admission(location, command)
        def unqualified_history(*_args, **_kwargs):
            raise AssertionError("Candidate admission scanned unrelated history")

        monkeypatch.setattr(system._HypothesisStore, "_verify", unqualified_history)
        monkeypatch.setattr(
            system, "_VERIFY_DISPOSITION_INTEGRITY", unqualified_history
        )
        monkeypatch.setattr(
            relationship_system,
            "_verify_relationship_reads_in_transaction",
            unqualified_history,
        )
        monkeypatch.setattr(
            lineage_system._LineageStore, "_verify", unqualified_history
        )
        retained = handle._opened().admit(
            admission.canonical_bytes,
            collision_request=collision,
            proof=_ACTOR_PROOFS[command.actor],
        )

        assert (
            retained.governing_manifest.hypothesis_version_id
            == location.subjects["record-a"].version_id
        )
    finally:
        handle.close()


class _DefectiveAdapter(_Adapter):
    def __init__(self, root: Path, defect: str, case: CaseId) -> None:
        super().__init__(root)
        self.defect = defect
        self.applicability = {
            item: (
                Applicability.required()
                if item is case
                else Applicability.waived(
                    reason="focused mutation sensitivity",
                    waiver_reference="issue:401#adapter-sensitivity",
                )
            )
            for item in CASE_INVENTORY
        }

    def open_handle(self, location: _Location, *, migrate: bool = False) -> _Handle:
        handle = super().open_handle(location, migrate=migrate)
        if self.defect == "no-store":
            handle.submit = lambda command, **_: AuthorityValue.from_command(command)
        elif self.defect == "bypass":
            handle.read = lambda record_id: AuthorityValue.from_command(
                _generic(record_id)
            )
        return handle


@pytest.mark.parametrize(
    ("defect", "case"),
    (("no-store", CaseId.FRESH_REPLAY), ("bypass", CaseId.TAMPER_REJECTION)),
)
def test_conformance_sensitivity_rejects_candidate_adapter_defects(
    tmp_path: Path, defect: str, case: CaseId
) -> None:
    report = run_conformance(_DefectiveAdapter(tmp_path, defect, case))
    assert not report.passed
    assert any(failure.case is case for failure in report.failures)
