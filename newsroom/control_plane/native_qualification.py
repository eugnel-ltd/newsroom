"""Provider-free qualification of one retained private native service cycle."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass

from newsroom.authority import ObjectAccessDecisionId, ObjectAdmissionId
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
    validate_sha256_digest,
)
from newsroom.increment9.proving import SOURCE_IDS

from .govuk_evidence import _api_url
from .model_usage import (
    CONSERVATIVE_DISPOSITION_SCHEMA_VERSION,
    ModelUsageService,
    NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE,
    NATIVE_GRAPHITI_FALLBACK_CANCELLATION_USAGE_SCOPE,
    WorkloadClass,
    _policy_for_allocation,
    _retained_terminal_allocation,
    _valid_native_embedding_timeout_disposition_record,
    _valid_native_graphiti_embedding_cancellation_disposition_record,
    _valid_native_graphiti_fallback_cancellation_disposition_record,
)
from .native_progress import LAND, PORTFOLIO, STATE, NativeRevisionJournal
from .store import LEDGER_GENESIS, append_ledger

_STARTED = "NATIVE_SERVICE_CYCLE_STARTED"
_TERMINAL = "NATIVE_SERVICE_CYCLE_TERMINAL"
_QUALIFICATION = "NATIVE_SERVICE_QUALIFICATION"
_TERMINAL_REVISION_STATES = frozenset({
    "ACKNOWLEDGED",
    "COLLISION_HOLD",
    "DISCOVERY_HOLD",
    "EVIDENCE_HOLD",
    "GRAPHITI_HOLD",
    "NO_CANDIDATE",
    "OPERATIONAL_HOLD",
    "PUBLICATION_HOLD",
    "RETRIEVAL_HOLD",
    "SAME_STATE_ASSOCIATED",
    "SCHEDULING_HOLD",
})
_NATIVE_WORKLOADS = frozenset({
    WorkloadClass.NATIVE_EVIDENCE_ASSESSOR.value,
    WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING.value,
})
_GRAPHITI_WORKLOADS = frozenset({
    WorkloadClass.GRAPHITI_CHAT_PRIMARY.value,
    WorkloadClass.GRAPHITI_CHAT_FALLBACK.value,
    WorkloadClass.GRAPHITI_EMBEDDING.value,
})
_READY_SOURCE_REASONS = frozenset({
    "GOVERNED_REVISIONS_RETAINED", "NO_ACTIVE_WARNINGS_OBSERVED",
})
_RETAINED_RIGHTS_HOLDS = frozenset({
    "AUTOMATED_REUSE_PERMISSION_NOT_RETAINED",
    "COMPUTER_ANALYSIS_PERMISSION_NOT_RETAINED",
    "MEDIA_REUSE_PERMISSION_SCOPE_NOT_ESTABLISHED",
    "NON_COMMERCIAL_INTERNAL_USE_ONLY",
})
_RETAINED_CONTENT_HOLDS = frozenset({
    "SOURCE_ITEM_NOT_YET_PUBLISHED",
    "SOURCE_ITEM_CHILD_COVERAGE_INCOMPLETE",
    "SOURCE_ITEM_ATTACHMENT_COVERAGE_INCOMPLETE",
    "SOURCE_ITEM_RIGHTS_EXCLUSION_HOLD",
    "SOURCE_ITEM_METADATA_HOLD",
})


class NativeQualificationError(ValueError):
    """Raised when retained private-runtime evidence does not qualify."""


class NativeQualificationPending(NativeQualificationError):
    """Valid retained usage remains unknown; no qualification is issued."""


@dataclass(frozen=True, slots=True)
class RetainedNativeQualification:
    runtime_identity_digest: str
    cycle_id: str
    started_seq: int
    started_ledger_digest: str
    terminal_seq: int
    terminal_ledger_digest: str
    source_inventory_digest: str
    revision_inventory_digest: str
    invocation_ids: tuple[str, ...]


def _is_terminal_revision_state(value: object) -> bool:
    return type(value) is str and value in _TERMINAL_REVISION_STATES


def qualification_report_ready(states: object, unclassified: object) -> bool:
    """Defer durable continuations; malformed inventories still fail."""

    # Existing journal checkpoints emitted by native_pipeline, native_retrieval
    # and native_publication. Unknown effects remain pending, not qualified.
    pending = {
        "QUEUED", "GRAPHITI_COMPLETE", "EMBEDDING_STARTED", "EMBEDDING_RETAINED",
        "DOCUMENT_RETAINED", "RETRIEVAL_COMPLETE", "CANDIDATE_ADMITTED",
        "ASSESSMENT_CONTRACT_REVALIDATION", "INTAKE_REQUESTED", "INTAKE_ACKNOWLEDGED",
        "ACQUISITION_STARTED", "ASSESSMENT_STARTED", "ASSESSMENT_INTERRUPTED",
        "EVIDENCE_RETAINED", "PUBLICATION_PREPARED", "PUBLICATION_STARTED",
        "COPY_CORRECTION_PREPARED",
    }
    if (
        type(states) is not dict
        or any(
            type(name) is not str
            or (not _is_terminal_revision_state(name) and name not in pending)
            or type(count) is not int or count <= 0
            for name, count in states.items()
        )
        or type(unclassified) is not int
        or unclassified != states.get("QUEUED", 0)
    ):
        raise NativeQualificationError("native revision terminal inventory differs")
    return not any(name in pending for name in states)


def _document(raw: str, payload_digest: str) -> dict:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise NativeQualificationError("native ledger payload is invalid") from exc
    if (
        type(value) is not dict
        or canonical_json_bytes(value).decode() != raw
        or digest_bytes(raw.encode()) != payload_digest
    ):
        raise NativeQualificationError("native ledger payload differs")
    return value


def _ledger(connection: sqlite3.Connection) -> tuple[tuple, ...]:
    relevant_kinds = (_STARTED, _TERMINAL, _QUALIFICATION, LAND, STATE, PORTFOLIO)
    rows = []
    cursor = connection.execute(
        "SELECT current.seq,current.at,current.kind,current.payload_digest,"
        "current.payload_json,current.prev_digest,current.digest,prior.digest "
        "FROM ledger current LEFT JOIN ledger prior ON prior.seq=current.seq-1 "
        f"WHERE current.kind IN ({','.join('?' for _ in relevant_kinds)}) "
        "ORDER BY current.seq",
        relevant_kinds,
    )
    for row in cursor:
        seq, at, kind, payload_digest, raw, prev_digest, ledger_digest, prior_digest = row
        if (
            type(seq) is not int
            or prev_digest != (LEDGER_GENESIS if seq == 1 else prior_digest)
            or not isinstance(raw, str)
        ):
            raise NativeQualificationError("native ledger predecessor differs")
        _document(raw, payload_digest)
        expected = digest_bytes(canonical_json_bytes({
            "at": at,
            "kind": kind,
            "payload_digest": payload_digest,
            "prev": prev_digest,
        }))
        if ledger_digest != expected:
            raise NativeQualificationError("native ledger digest differs")
        # The journal separately reconstructs these logical records. Keep their
        # verified references, not another full history of bodies and bindings.
        rows.append(row[:4] + (None,) + row[5:] if kind in {LAND, STATE, PORTFOLIO} else row)
    return tuple(rows)


def _service_cycle(
    rows: tuple[tuple, ...], identity: str,
) -> tuple[tuple, dict, tuple, dict]:
    terminals = []
    for row in rows:
        if row[2] != _TERMINAL:
            continue
        payload = _document(row[4], row[3])
        if payload.get("runtime_identity_digest") == identity:
            terminals.append((row, payload))
    if not terminals:
        raise NativeQualificationError("identity-bound native terminal is absent")
    terminal, terminal_payload = terminals[-1]
    cycle_id = terminal_payload.get("cycle_id")
    starts = []
    for row in rows:
        if row[2] != _STARTED or row[0] >= terminal[0]:
            continue
        payload = _document(row[4], row[3])
        if payload.get("cycle_id") == cycle_id:
            starts.append((row, payload))
    if len(starts) != 1:
        raise NativeQualificationError("native cycle start binding differs")
    start, start_payload = starts[0]
    if (
        type(cycle_id) is not str
        or not cycle_id
        or start_payload != {
            "cycle_id": cycle_id,
            "runtime_identity_digest": identity,
        }
        or set(terminal_payload) != {
            "runtime_identity_digest", "cycle_id", "outcome", "failure_class", "pipeline",
        }
        or terminal_payload["outcome"] != "COMPLETE"
        or terminal_payload["failure_class"] is not None
        or type(terminal_payload["pipeline"]) is not dict
    ):
        raise NativeQualificationError("native cycle terminal differs")
    return start, start_payload, terminal, terminal_payload


def _terminal_portfolio(
    connection: sqlite3.Connection, rows: tuple[tuple, ...], terminal: tuple,
    pipeline: dict,
) -> tuple[tuple[dict, ...], dict[str, int]]:
    if "source_portfolio_ref" not in pipeline:
        return _portfolio(pipeline)  # Historical inline terminals remain exact.
    if set(pipeline) != {"source_portfolio_ref", "revision_states", "unclassified_revisions"}:
        raise NativeQualificationError("native pipeline report fields differ")
    reference = pipeline["source_portfolio_ref"]
    latest = next((row for row in reversed(rows)
                   if row[2] == PORTFOLIO and row[0] < terminal[0]), None)
    if (
        type(reference) is not dict or type(reference.get("seq")) is not int
        or latest is None
        or reference != {"seq": latest[0], "payload_digest": latest[3]}
    ):
        raise NativeQualificationError("native source portfolio reference differs")
    # _ledger has already authenticated the chain but deliberately discarded
    # portfolio bodies. Re-read only this exact immutable referenced record.
    stored = connection.execute(
        "SELECT seq,at,kind,payload_digest,payload_json,prev_digest,digest "
        "FROM ledger WHERE seq=?", (reference["seq"],),
    ).fetchone()
    if stored is None or stored[:4] + stored[5:] != latest[:4] + latest[5:7]:
        raise NativeQualificationError("native source portfolio ledger binding differs")
    document = _document(stored[4], stored[3])
    if set(document) != {"sources"}:
        raise NativeQualificationError("native source portfolio fields differ")
    return _portfolio({
        "sources": document["sources"],
        "revision_states": pipeline["revision_states"],
        "unclassified_revisions": pipeline["unclassified_revisions"],
    })


def _portfolio(pipeline: dict) -> tuple[tuple[dict, ...], dict[str, int]]:
    if set(pipeline) != {"sources", "revision_states", "unclassified_revisions"}:
        raise NativeQualificationError("native pipeline report fields differ")
    sources = pipeline["sources"]
    if type(sources) is not list or len(sources) != len(SOURCE_IDS):
        raise NativeQualificationError("native source disposition inventory differs")
    by_source = {}
    for item in sources:
        if type(item) is not dict or set(item) != {
            "source_id", "status", "reason_code", "revision_ids", "observations",
            "item_holds",
        }:
            raise NativeQualificationError("native source disposition differs")
        source_id = item["source_id"]
        if (
            source_id in by_source
            or source_id not in SOURCE_IDS
            or item["status"] not in {"READY", "HOLD"}
            or (
                item["status"] == "READY"
                and item["reason_code"] not in _READY_SOURCE_REASONS
            )
            or (
                item["status"] == "HOLD"
                and item["reason_code"] not in _RETAINED_RIGHTS_HOLDS
                and item["reason_code"] != "SOURCE_ITEMS_HELD"
            )
            or type(item["reason_code"]) is not str
            or not item["reason_code"]
            or type(item["revision_ids"]) is not list
            or len(item["revision_ids"]) != len(set(item["revision_ids"]))
            or item["revision_ids"] != sorted(item["revision_ids"])
            or any(type(value) is not str or not value for value in item["revision_ids"])
            or type(item["observations"]) is not list
            or (item["status"] == "HOLD" and not item["observations"])
            or any(
                type(value) is not list
                or len(value) != 4
                or any(type(part) is not str or not part for part in value)
                for value in item["observations"]
            )
            or type(item["item_holds"]) is not list
            or any(
                type(value) is not list
                or len(value) != 2
                or any(type(part) is not str or not part for part in value)
                for value in item["item_holds"]
            )
        ):
            raise NativeQualificationError("native source disposition differs")
        _source_item_holds(item)
        by_source[source_id] = item
    if tuple(item["source_id"] for item in sources) != SOURCE_IDS:
        raise NativeQualificationError("native source disposition order differs")
    states = pipeline["revision_states"]
    if not qualification_report_ready(states, pipeline["unclassified_revisions"]):
        raise NativeQualificationError("native revision terminal inventory differs")
    return tuple(sources), states


def _source_item_holds(item: dict) -> None:
    """Keep evidenced content holds local; never relabel them as ready revisions."""
    # Feed and child inventories can reach the same retained item twice. Only
    # exact repeats are equivalent; conflicting reasons or receipts still fail.
    # The original portfolio/ledger bytes remain unchanged and authenticated.
    holds = tuple(dict.fromkeys(tuple(value) for value in item["item_holds"]))
    if item["reason_code"] != "SOURCE_ITEMS_HELD":
        if holds:
            raise NativeQualificationError("native source item hold disposition differs")
        return
    if (
        item["status"] != "HOLD"
        or item["source_id"] not in {"UK-01", "UK-02", "UK-03", "UK-05"}
        or not holds
        or len({url for url, _ in holds}) != len(holds)
    ):
        raise NativeQualificationError("native source item hold inventory differs")
    for url, reason in holds:
        try:
            if reason not in _RETAINED_CONTENT_HOLDS:
                raise ValueError("unclassified content hold")
            endpoint = _api_url(url)
            observations = tuple(dict.fromkeys(
                tuple(value) for value in item["observations"] if value[0] == endpoint
            ))
            if len(observations) != 1:
                raise ValueError("exact source observation is absent")
            _, digest, admission, access = observations[0]
            validate_sha256_digest(digest)
            ObjectAdmissionId.parse(admission)
            ObjectAccessDecisionId.parse(access)
        except (TypeError, ValueError) as exc:
            raise NativeQualificationError("native source item hold evidence differs") from exc


def _revision_inventory(
    connection: sqlite3.Connection, sources: tuple[dict, ...], states: dict[str, int],
) -> tuple[NativeRevisionJournal, dict[str, int]]:
    try:
        journal = NativeRevisionJournal(connection)
    except (KeyError, TypeError, ValueError, sqlite3.DatabaseError) as exc:
        raise NativeQualificationError("native revision progress differs") from exc
    retained_states = Counter(
        journal.summary(revision_id).get("stage", "QUEUED")
        for revision_id in journal.units
    )
    if dict(retained_states) != states or journal.portfolio != sources:
        raise NativeQualificationError("native pipeline report is not retained authority")
    if any(not _is_terminal_revision_state(state) for state in retained_states):
        raise NativeQualificationError("native revision is not terminal")
    for _, value in journal.iter_summaries():
        stage = value.get("stage")
        reason = value.get("facts", {}).get("reason")
        if stage.endswith("_HOLD") and (
            type(reason) is not str
            or not reason
            or reason.endswith(("Error", "Exception"))
        ):
            raise NativeQualificationError("native revision hold is not evidenced")
    return journal, dict(retained_states)


def _revision_inventory_digest(rows: tuple[tuple, ...], terminal_seq: int) -> str:
    references = [
        {
            "seq": row[0], "kind": row[2], "payload_digest": row[3],
            "ledger_digest": row[6],
        }
        for row in rows
        if row[0] <= terminal_seq and row[2] in {LAND, STATE}
    ]
    return digest_bytes(canonical_json_bytes(references))


def _canonical_record(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise NativeQualificationError("model usage record is invalid") from exc
    if type(value) is not dict or canonical_json_bytes(value).decode() != raw:
        raise NativeQualificationError("model usage record differs")
    return value


def _invocations(
    connection: sqlite3.Connection,
    journal: NativeRevisionJournal,
    retained_ids: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if retained_ids == ():
        return ()
    ingest_ids = {
        unit.ingest_id for units in journal.units.values() for unit in units
    }
    selected_workloads = tuple(sorted(_NATIVE_WORKLOADS | _GRAPHITI_WORKLOADS))
    values = retained_ids if retained_ids is not None else selected_workloads
    field = "a.invocation_id" if retained_ids is not None else "a.workload_class"
    rows = tuple(connection.execute(
        "SELECT a.invocation_id,a.workload_class,a.policy_digest,a.canonical_digest,"
        "a.record_json,e.record_json "
        "FROM model_invocation_allocations a JOIN model_work_envelopes e "
        "ON e.envelope_id=a.envelope_id "
        f"WHERE {field} IN ({','.join('?' for _ in values)}) "
        "ORDER BY a.allocated_at,a.invocation_id",
        values,
    ))
    selected = []
    allocations = {}
    for (
        invocation_id, workload, policy_digest, allocation_digest,
        allocation_raw, envelope_raw,
    ) in rows:
        allocation = _canonical_record(allocation_raw)
        envelope = _canonical_record(envelope_raw)
        if (
            allocation.get("invocation_id") != invocation_id
            or allocation.get("workload_class") != workload
            or allocation.get("canonical_digest") != allocation_digest
            or allocation.get("invocation_policy_digest") != policy_digest
            or envelope.get("envelope_id") != allocation.get("envelope_id")
        ):
            raise NativeQualificationError("model usage allocation binding differs")
        if workload in _NATIVE_WORKLOADS or (
            workload in _GRAPHITI_WORKLOADS and envelope.get("ingest_id") in ingest_ids
        ):
            selected.append(invocation_id)
            allocations[invocation_id] = allocation
    if retained_ids is not None:
        if tuple(sorted(set(retained_ids))) != retained_ids:
            raise NativeQualificationError("native invocation inventory differs")
        selected = [value for value in selected if value in retained_ids]
        if tuple(sorted(selected)) != retained_ids:
            raise NativeQualificationError("native invocation inventory differs")
    usage_pending = False
    for invocation_id in selected:
        terminal_row = connection.execute(
            "SELECT terminal_digest,usage_status,record_json "
            "FROM model_invocation_terminals WHERE invocation_id=?",
            (invocation_id,),
        ).fetchone()
        if terminal_row is None:
            raise NativeQualificationError("native model invocation is in flight")
        terminal = _canonical_record(terminal_row[2])
        unsigned = dict(terminal)
        retained_digest = unsigned.get("terminal_digest")
        unsigned["terminal_digest"] = ""
        if (
            terminal.get("invocation_id") != invocation_id
            or terminal.get("usage_status") != terminal_row[1]
            or retained_digest != terminal_row[0]
            or digest_canonical(unsigned) != retained_digest
        ):
            raise NativeQualificationError("native model terminal binding differs")
        if terminal.get("usage_status") == "UNREPORTED":
            # Unknown usage is pending only after the same retained semantics
            # used by accounting; a rehashed but impossible total is corruption.
            try:
                allocation, typed_terminal = _retained_terminal_allocation(
                    connection, invocation_id,
                )
                breach = ModelUsageService._validate_terminal(
                    typed_terminal, allocation.workload_class,
                    _policy_for_allocation(connection, allocation),
                    requested_max_output_tokens=allocation.max_output_tokens,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise NativeQualificationError("native model terminal semantics differ") from exc
            if breach is not None:
                raise NativeQualificationError("native model usage is unresolved")
        reconciliation_rows = tuple(connection.execute(
            "SELECT reconciliation_digest,record_json FROM model_usage_reconciliations "
            "WHERE invocation_id=? ORDER BY observed_at,reconciliation_digest",
            (invocation_id,),
        ))
        effective = terminal
        if reconciliation_rows:
            digest, raw = reconciliation_rows[-1]
            effective = _canonical_record(raw)
            unsigned = dict(effective)
            if (
                unsigned.pop("reconciliation_digest", None) != digest
                or effective.get("invocation_id") != invocation_id
                or digest_canonical(unsigned) != digest
            ):
                raise NativeQualificationError("native usage reconciliation differs")
        disposition_row = connection.execute(
            "SELECT disposition_digest,terminal_digest,usage_status,record_json,approved_by "
            "FROM model_usage_conservative_dispositions WHERE invocation_id=?",
            (invocation_id,),
        ).fetchone()
        if effective.get("usage_status") != "REPORTED" and disposition_row is not None:
            disposition = _canonical_record(disposition_row[3])
            unsigned = dict(disposition)
            retained_digest = unsigned.pop("disposition_digest", None)
            embedding_timeout = _valid_native_embedding_timeout_disposition_record(
                connection,
                allocation_record=allocations[invocation_id],
                terminal_record=terminal,
                disposition_record=disposition,
            )
            if (
                disposition_row[4] == NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE
                or disposition.get("authority_scope")
                == NATIVE_GRAPHITI_EMBEDDING_CANCELLATION_USAGE_SCOPE
            ) and not _valid_native_graphiti_embedding_cancellation_disposition_record(
                connection, allocation_record=allocations[invocation_id],
                terminal_record=terminal, disposition_record=disposition,
            ):
                raise NativeQualificationError("native embedding cancellation disposition differs")
            # Scope markers are mutable: the immutable cancelled leaf must
            # still select this proof after a coherently rehashed downgrade.
            if (
                disposition_row[4] == NATIVE_GRAPHITI_FALLBACK_CANCELLATION_USAGE_SCOPE
                or disposition.get("authority_scope")
                == NATIVE_GRAPHITI_FALLBACK_CANCELLATION_USAGE_SCOPE
                or (
                    allocations[invocation_id].get("workload_class")
                    == WorkloadClass.GRAPHITI_CHAT_FALLBACK.value
                    and terminal.get("outcome") == "CANCELLED"
                )
            ) and not _valid_native_graphiti_fallback_cancellation_disposition_record(
                connection, allocation_record=allocations[invocation_id],
                terminal_record=terminal, disposition_record=disposition,
            ):
                raise NativeQualificationError("native fallback cancellation disposition differs")
            if (
                retained_digest != disposition_row[0]
                or digest_canonical(unsigned) != retained_digest
                or disposition.get("invocation_id") != invocation_id
                or disposition.get("terminal_digest") != disposition_row[1]
                or disposition.get("allocation_digest")
                != allocations[invocation_id].get("canonical_digest")
                or disposition.get("policy_digest")
                != allocations[invocation_id].get("invocation_policy_digest")
                or disposition.get("schema_version")
                != CONSERVATIVE_DISPOSITION_SCHEMA_VERSION
                or disposition.get("usage_status") != disposition_row[2]
                or disposition.get("usage_status") != "ESTIMATED"
                or disposition.get("estimate_policy_digest")
                != disposition.get("policy_digest")
                or disposition.get("exact_usage_remains_unknown") is not True
                or disposition.get("provider_dispatch_preserved") is not True
                or disposition.get("unknown_spend_released") is not False
                or terminal.get("usage_status") != "UNREPORTED"
                or (
                    terminal.get("failure_class")
                    != "MISSING_PROVIDER_TELEMETRY"
                    and not embedding_timeout
                )
            ):
                raise NativeQualificationError("native usage disposition differs")
            effective = disposition
        if effective.get("policy_breach") is not None:
            raise NativeQualificationError("native model usage is unresolved")
        if effective.get("usage_status") == "UNREPORTED":
            usage_pending = True
        elif effective.get("usage_status") not in {"REPORTED", "ESTIMATED"}:
            raise NativeQualificationError("native model usage is unresolved")
    # Validate every selected record before deferring: an earlier unknown usage
    # must not conceal a later corrupt binding or policy breach.
    if usage_pending:
        raise NativeQualificationPending("native model usage is unresolved")
    return tuple(sorted(selected))


def _candidate_qualification(
    connection: sqlite3.Connection, identity: str, rows: tuple[tuple, ...],
) -> RetainedNativeQualification:
    start, _start_payload, terminal, terminal_payload = _service_cycle(rows, identity)
    sources, states = _terminal_portfolio(connection, rows, terminal, terminal_payload["pipeline"])
    journal, _retained_states = _revision_inventory(connection, sources, states)
    invocation_ids = _invocations(connection, journal)
    return RetainedNativeQualification(
        identity,
        terminal_payload["cycle_id"],
        start[0],
        start[6],
        terminal[0],
        terminal[6],
        digest_bytes(canonical_json_bytes(sources)),
        _revision_inventory_digest(rows, terminal[0]),
        invocation_ids,
    )


def record_qualification(
    connection: sqlite3.Connection, current_identity_digest: str,
) -> RetainedNativeQualification:
    """Validate and retain the completed ``--once`` qualification reference."""

    validate_sha256_digest(current_identity_digest)
    retained = _candidate_qualification(
        connection, current_identity_digest, _ledger(connection),
    )
    payload = {
        "runtime_identity_digest": retained.runtime_identity_digest,
        "cycle_id": retained.cycle_id,
        "started_seq": retained.started_seq,
        "started_ledger_digest": retained.started_ledger_digest,
        "terminal_seq": retained.terminal_seq,
        "terminal_ledger_digest": retained.terminal_ledger_digest,
        "source_inventory_digest": retained.source_inventory_digest,
        "revision_inventory_digest": retained.revision_inventory_digest,
        "invocation_ids": list(retained.invocation_ids),
    }
    append_ledger(connection, _QUALIFICATION, payload)
    connection.commit()
    return retained


def validate_qualification(
    connection: sqlite3.Connection, current_identity_digest: str,
) -> RetainedNativeQualification:
    """Reconstruct the latest exact private qualification without any effect."""

    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("native qualification requires a SQLite connection")
    try:
        validate_sha256_digest(current_identity_digest)
        rows = _ledger(connection)
        references = [
            (row, _document(row[4], row[3]))
            for row in rows if row[2] == _QUALIFICATION
            and _document(row[4], row[3]).get("runtime_identity_digest")
            == current_identity_digest
        ]
        if not references:
            raise NativeQualificationError("identity-bound native qualification is absent")
        _reference_row, reference = references[-1]
        if set(reference) != {
            "runtime_identity_digest", "cycle_id", "started_seq",
            "started_ledger_digest", "terminal_seq", "terminal_ledger_digest",
            "source_inventory_digest", "revision_inventory_digest", "invocation_ids",
        }:
            raise NativeQualificationError("native qualification reference differs")
        by_seq = {row[0]: row for row in rows}
        start = by_seq.get(reference["started_seq"])
        terminal = by_seq.get(reference["terminal_seq"])
        if (
            start is None or start[2] != _STARTED
            or terminal is None or terminal[2] != _TERMINAL
            or start[6] != reference["started_ledger_digest"]
            or terminal[6] != reference["terminal_ledger_digest"]
        ):
            raise NativeQualificationError("native qualification ledger binding differs")
        start_payload = _document(start[4], start[3])
        terminal_payload = _document(terminal[4], terminal[3])
        sources, states = _terminal_portfolio(connection, rows, terminal, terminal_payload["pipeline"])
        if (
            start_payload != {
                "cycle_id": reference["cycle_id"],
                "runtime_identity_digest": current_identity_digest,
            }
            or terminal_payload.get("cycle_id") != reference["cycle_id"]
            or terminal_payload.get("runtime_identity_digest")
            != current_identity_digest
            or terminal_payload.get("outcome") != "COMPLETE"
            or terminal_payload.get("failure_class") is not None
            or digest_bytes(canonical_json_bytes(sources))
            != reference["source_inventory_digest"]
            or _revision_inventory_digest(rows, terminal[0])
            != reference["revision_inventory_digest"]
        ):
            raise NativeQualificationError("native qualification evidence differs")
        journal = NativeRevisionJournal(connection)
        invocation_ids = _invocations(
            connection, journal, tuple(reference["invocation_ids"]),
        )
    except NativeQualificationError:
        raise
    except (KeyError, TypeError, ValueError, sqlite3.DatabaseError) as exc:
        raise NativeQualificationError("retained native qualification differs") from exc
    return RetainedNativeQualification(
        current_identity_digest,
        reference["cycle_id"],
        reference["started_seq"],
        reference["started_ledger_digest"],
        reference["terminal_seq"],
        reference["terminal_ledger_digest"],
        reference["source_inventory_digest"],
        reference["revision_inventory_digest"],
        invocation_ids,
    )


__all__ = [
    "NativeQualificationError",
    "NativeQualificationPending",
    "RetainedNativeQualification",
    "record_qualification",
    "validate_qualification",
]
