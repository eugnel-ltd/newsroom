"""Provider-free, read-only Graphiti steady-state evidence packet."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
    validate_sha256_digest,
)
from newsroom.authority.migrations import MIGRATIONS
from newsroom.control_plane.graphiti_admission import (
    GRAPHITI_ADMISSION_COHORT_SCHEMA_VERSION,
    GRAPHITI_ADMISSION_GENERATION_IDENTITY_VERSION,
    GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION,
    GraphitiAdmissionConsumerError,
    graphiti_admission_telemetry,
    graphiti_decided_cohort_generation_identity,
    graphiti_governed_decision_from_json,
    graphiti_projection_receipt_from_json,
    graphiti_projection_receipt_matches_binding,
    graphiti_projection_reconciliation_from_json,
)
from newsroom.control_plane.corpus import CorpusIngestUnit
from newsroom.control_plane.cycle import load_graphiti_units_from_connection
from newsroom.control_plane.graphiti_events import (
    GRAPHITI_EVENT_STATES,
    GraphitiRevisionEvent,
    graphiti_unit_binding_reason,
)
from newsroom.control_plane.graphiti_event_reconciliation import (
    GraphitiEventRepairDecision,
    GraphitiEventRepairDisposition,
    GraphitiEventReconciliationError,
    classify_graphiti_event_gaps,
)
from newsroom.control_plane.graphiti_spend_reconciliation import (
    GraphitiSpendReconciliationError,
    validate_retained_graphiti_spend_dispositions,
)
from newsroom.control_plane.issue_790_canary import graphiti_excluded_event_ids
from newsroom.control_plane.read_only_snapshot import read_only_snapshot
from newsroom.increment9.proving import (
    ProvingError,
    PROVING_GATES,
    RIGHTS_GATE_BY_SOURCE,
    SOURCE_IDS,
    SOURCE_URLS,
    resolve_observation_body,
)
from newsroom.graphiti_adapter.evaluation_packet import GRAPHITI_EXTRACTION_TIMEOUT_MS
from newsroom.graphiti_adapter.identity import attempt_ids
from newsroom.increment4.contracts import (
    INCREMENT4_ADMITTED_FAMILY_ID,
    INCREMENT4_ADMITTED_MAPPING_ID,
    INCREMENT4_ADMITTED_MAPPING_VERSION,
    INCREMENT4_ADMITTED_ONTOLOGY_ID,
    INCREMENT4_ADMITTED_ONTOLOGY_VERSION,
    INCREMENT4_ADMITTED_PROJECTOR_VERSION,
    increment4_admitted_family_v1,
    increment4_admitted_mapping_v1,
    increment4_admitted_ontology_v1,
)
from newsroom.projection import ProjectionGenerationId
from newsroom.projection.neo4j import StructuralReconciliationView

SCHEMA_VERSION = "newsroom.graphiti-steady-state-packet.v4"

CAMPAIGN_SCHEMA_VERSION = "newsroom.graphiti-bounded-campaign-input.v4"

CAMPAIGN_REQUIRED_STOP_CONDITIONS = frozenset(
    {
        "CAP_REACHED",
        "CONFIG_DRIFT",
        "EXACT_RECEIPT_DRIFT",
        "GRAPH_IDENTITY_DRIFT",
        "IDENTITY_DRIFT",
        "INTEGRITY_FAILURE",
        "PROVIDER_FAILURE",
        "PROVIDER_USAGE_DRIFT",
        "PROJECTION_GENERATION_DRIFT",
        "RATE_CAP_REACHED",
        "RECONCILIATION_FAILURE",
        "RECONCILIATION_DRIFT",
        "RIGHTS_DRIFT",
        "SNAPSHOT_DRIFT",
        "SPEND_ACCOUNTING_DRIFT",
        "CIRCUIT_OPEN",
        "WALL_TIME_CAP_REACHED",
    }
)

CAMPAIGN_RAMP_ENTRY_CONDITIONS = frozenset(
    {"EXACT_SNAPSHOT_AND_IDENTITY_RECONFIRMED", "OWNER_F4_GO_RETAINED"}
)
CAMPAIGN_RAMP_ADVANCE_CONDITIONS = frozenset(
    {
        "ALL_EXACT_RECEIPTS_RECONCILED",
        "CAPS_AND_ACCOUNTING_RECONCILED",
        "NO_STOP_CONDITION_OBSERVED",
    }
)

CAMPAIGN_SUCCESS_OBJECTIVE_BASE = MappingProxyType(
    {
        "watermark": "selected cohort terminal",
        "backlog": 0,
        "velocity": "service_at_least_arrival",
        "reconciliation": "exact",
    }
)

CAMPAIGN_PER_EVENT_SPEND_GBP_MICROUNITS = 500_000
_CAMPAIGN_COUNT_CAP_NAMES = (
    "proposals",
    "entity_admits",
    "relation_admits",
    "effects",
)


def campaign_event_limits(event_count: int) -> tuple[int, ...]:
    """Return the automatic 1 → 10 → N ramp, collapsing duplicates."""

    if event_count <= 0:
        raise ValueError("campaign event count must be positive")
    return tuple(
        dict.fromkeys(
            value
            for value in (1, min(10, event_count), event_count)
            if value > 0
        )
    )


def _campaign_wall_time_seconds(event_count: int) -> int:
    return max(600, event_count * (GRAPHITI_EXTRACTION_TIMEOUT_MS // 1000))


def _sorted_ramp_conditions(value: object) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(item, str) and item for item in value):
        return None
    unique = sorted(set(value))
    if value != unique:
        return None
    return unique


def _copy_ramp_phase(
    raw: Mapping[str, object], *, event_limit: int | None = None
) -> dict[str, object]:
    phase = dict(raw)
    if event_limit is not None:
        phase["event_limit"] = event_limit
    entry = raw.get("entry_conditions")
    advance = raw.get("advance_conditions")
    if isinstance(entry, list):
        phase["entry_conditions"] = list(entry)
    if isinstance(advance, list):
        phase["advance_conditions"] = list(advance)
    return phase


def _campaign_ramp_is_closed(
    phases: object, *, original_event_cap: int
) -> bool:
    if not isinstance(phases, list) or not phases:
        return False
    prior_limit = 0
    for raw in phases:
        if not isinstance(raw, Mapping):
            return False
        phase_id = raw.get("phase_id")
        if not isinstance(phase_id, str) or not phase_id.strip():
            return False
        limit = raw.get("event_limit")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            return False
        if limit <= prior_limit:
            return False
        entry = _sorted_ramp_conditions(raw.get("entry_conditions"))
        advance = _sorted_ramp_conditions(raw.get("advance_conditions"))
        if (
            entry is None
            or advance is None
            or not CAMPAIGN_RAMP_ENTRY_CONDITIONS.issubset(entry)
            or not CAMPAIGN_RAMP_ADVANCE_CONDITIONS.issubset(advance)
        ):
            return False
        prior_limit = limit
    return prior_limit == original_event_cap


def _narrow_campaign_ramp_to_selected_cohort(
    ramp: object,
    *,
    selected_event_count: int,
    original_event_cap: int,
) -> object:
    """Truncate a closed ramp; leave empty or malformed input unchanged."""

    if not isinstance(ramp, Mapping):
        return ramp
    phases = ramp.get("phases")
    if not isinstance(phases, list) or not _campaign_ramp_is_closed(
        phases, original_event_cap=original_event_cap
    ):
        return dict(ramp)
    adapted: list[dict[str, object]] = []
    remaining: list[Mapping[str, object]] = []
    truncated = False
    for raw in phases:
        if not isinstance(raw, Mapping):
            return dict(ramp)
        limit = int(raw["event_limit"])
        if truncated:
            remaining.append(raw)
            continue
        if limit < selected_event_count:
            adapted.append(_copy_ramp_phase(raw))
            continue
        adapted.append(_copy_ramp_phase(raw, event_limit=selected_event_count))
        truncated = True
    if not truncated:
        return dict(ramp)
    if remaining:
        final = dict(adapted[-1])
        entry = list(final["entry_conditions"])
        advance = list(final["advance_conditions"])
        for raw in remaining:
            extra_entry = raw.get("entry_conditions")
            extra_advance = raw.get("advance_conditions")
            if not isinstance(extra_entry, list) or not isinstance(extra_advance, list):
                return dict(ramp)
            entry = sorted(set(entry).union(extra_entry))
            advance = sorted(set(advance).union(extra_advance))
        final["entry_conditions"] = entry
        final["advance_conditions"] = advance
        adapted[-1] = final
    return {**dict(ramp), "phases": adapted}


def _narrow_campaign_input_to_selected_cohort(
    campaign: Mapping[str, object],
    *,
    selected_event_count: int,
) -> dict[str, object]:
    """Bind machine campaign totals to the sealed post-exclusion cohort.

    Planning may freeze ``caps.total.events`` from an earlier bootstrap
    count. The selected cohort is the later, exclusion-consistent dispatch
    set. Narrowing never adds events, never retries excluded ``RETRY_HELD``
    identities, and never raises a smaller supplied cap.
    """

    caps = campaign.get("caps")
    if not isinstance(caps, Mapping):
        return dict(campaign)
    per_event = caps.get("per_event")
    total = caps.get("total")
    rate = caps.get("rate")
    if not isinstance(per_event, Mapping) or not isinstance(total, Mapping):
        return dict(campaign)
    supplied_events = total.get("events")
    if (
        isinstance(supplied_events, bool)
        or not isinstance(supplied_events, int)
        or supplied_events <= selected_event_count
    ):
        return dict(campaign)

    ramp = campaign.get("ramp")
    if not _campaign_ramp_is_closed(
        ramp.get("phases") if isinstance(ramp, Mapping) else None,
        original_event_cap=supplied_events,
    ):
        return dict(campaign)

    aligned_total = dict(total)
    aligned_total["events"] = selected_event_count
    for name in _CAMPAIGN_COUNT_CAP_NAMES:
        per_event_cap = per_event.get(name)
        total_cap = total.get(name)
        if (
            isinstance(per_event_cap, int)
            and not isinstance(per_event_cap, bool)
            and isinstance(total_cap, int)
            and not isinstance(total_cap, bool)
            and total_cap == supplied_events * per_event_cap
        ):
            aligned_total[name] = selected_event_count * per_event_cap
    spend = total.get("spend_gbp_microunits")
    if (
        isinstance(spend, int)
        and not isinstance(spend, bool)
        and spend == supplied_events * CAMPAIGN_PER_EVENT_SPEND_GBP_MICROUNITS
    ):
        aligned_total["spend_gbp_microunits"] = (
            selected_event_count * CAMPAIGN_PER_EVENT_SPEND_GBP_MICROUNITS
        )
    wall_time = total.get("wall_time_seconds")
    if (
        isinstance(wall_time, int)
        and not isinstance(wall_time, bool)
        and wall_time == _campaign_wall_time_seconds(supplied_events)
    ):
        aligned_total["wall_time_seconds"] = _campaign_wall_time_seconds(
            selected_event_count
        )
    aligned_caps = {
        "per_event": dict(per_event),
        "total": aligned_total,
        "rate": dict(rate) if isinstance(rate, Mapping) else rate,
    }
    return {
        **dict(campaign),
        "caps": aligned_caps,
        "ramp": _narrow_campaign_ramp_to_selected_cohort(
            campaign.get("ramp"),
            selected_event_count=selected_event_count,
            original_event_cap=supplied_events,
        ),
    }


HISTORICAL_PARTITION_CATEGORIES = (
    "VERIFIED_TERMINAL",
    "CURRENT_DISPATCH_PREFLIGHT_CANDIDATE",
    "RIGHTS_OR_INPUT_HELD",
    "NON_REPLAYABLE_OR_AMBIGUOUS_EFFECT_HOLD",
    "UNCLASSIFIED",
)
PRE_FRONTIER_BACKLOG_HOLD_REASON = "PRE_FRONTIER_BACKLOG_NOT_ACTIONABLE"
_CURRENT_PREFLIGHT_REASON = "CURRENT_RIGHTS_INPUT_AND_BINDING_VERIFIED"


class GraphitiCampaignRuntime:
    """Already-composed governed worker capabilities for a bounded campaign."""

    __slots__ = (
        "__graphiti",
        "__admission_factory",
        "__bind_unit_authority",
        "__graph_state_fence",
        "__authority_store_source_path",
        "__authority_store_descriptor_digest",
        "__graph_destination_id",
        "__construction_token",
    )

    def __init__(
        self,
        *,
        graphiti: object,
        admission_factory: Callable[..., object],
        bind_unit_authority: Callable[[CorpusIngestUnit], CorpusIngestUnit],
        graph_state_fence: Callable[[Mapping[str, object]], Mapping[str, object]],
        authority_store_source_path: str,
        authority_store_descriptor_digest: str,
        graph_destination_id: str,
        _construction_token: object,
    ) -> None:
        if _construction_token is not _CAMPAIGN_RUNTIME_CONSTRUCTION_TOKEN:
            raise TypeError("campaign runtimes require the governed worker composer")
        if (
            not callable(admission_factory)
            or not callable(bind_unit_authority)
            or not callable(graph_state_fence)
        ):
            raise TypeError("campaign runtime capabilities must be callable")
        if (
            not isinstance(authority_store_source_path, str)
            or not authority_store_source_path
        ):
            raise ValueError("campaign runtime authority path is invalid")
        validate_sha256_digest(
            authority_store_descriptor_digest,
            field="campaign runtime authority descriptor digest",
        )
        validate_sha256_digest(
            graph_destination_id,
            field="campaign runtime graph destination id",
        )
        self.__graphiti = graphiti
        self.__admission_factory = admission_factory
        self.__bind_unit_authority = bind_unit_authority
        self.__graph_state_fence = graph_state_fence
        self.__authority_store_source_path = authority_store_source_path
        self.__authority_store_descriptor_digest = authority_store_descriptor_digest
        self.__graph_destination_id = graph_destination_id
        self.__construction_token = _construction_token

    @property
    def graphiti(self) -> object:
        return self.__graphiti

    @property
    def admission_factory(self) -> Callable[..., object]:
        return self.__admission_factory

    @property
    def bind_unit_authority(
        self,
    ) -> Callable[[CorpusIngestUnit], CorpusIngestUnit]:
        return self.__bind_unit_authority

    @property
    def graph_state_fence(
        self,
    ) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
        return self.__graph_state_fence

    @property
    def authority_store_source_path(self) -> str:
        return self.__authority_store_source_path

    @property
    def authority_store_descriptor_digest(self) -> str:
        return self.__authority_store_descriptor_digest

    @property
    def graph_destination_id(self) -> str:
        return self.__graph_destination_id


_CAMPAIGN_RUNTIME_CONSTRUCTION_TOKEN = object()


def _mint_graphiti_campaign_runtime(
    *,
    graphiti: object,
    admission_factory: Callable[..., object],
    bind_unit_authority: Callable[[CorpusIngestUnit], CorpusIngestUnit],
    graph_state_fence: Callable[[Mapping[str, object]], Mapping[str, object]],
    authority_store_source_path: str,
    authority_store_descriptor_digest: str,
    graph_destination_id: str,
) -> GraphitiCampaignRuntime:
    """Mint one opaque runtime after the governed worker composer has wired it."""

    return GraphitiCampaignRuntime(
        graphiti=graphiti,
        admission_factory=admission_factory,
        bind_unit_authority=bind_unit_authority,
        graph_state_fence=graph_state_fence,
        authority_store_source_path=authority_store_source_path,
        authority_store_descriptor_digest=authority_store_descriptor_digest,
        graph_destination_id=graph_destination_id,
        _construction_token=_CAMPAIGN_RUNTIME_CONSTRUCTION_TOKEN,
    )


def _is_minted_graphiti_campaign_runtime(value: object) -> bool:
    return isinstance(value, GraphitiCampaignRuntime) and (
        getattr(
            value,
            "_GraphitiCampaignRuntime__construction_token",
            None,
        )
        is _CAMPAIGN_RUNTIME_CONSTRUCTION_TOKEN
    )


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _schema_fingerprint(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
    ).fetchall()
    return digest_canonical(
        [
            {
                "type": str(row[0]),
                "name": str(row[1]),
                "table": str(row[2]),
                "sql": str(row[3]),
            }
            for row in rows
        ]
    )


_SERIALIZE_MAX_BYTES = 256 * 1024 * 1024


def _watermark(connection: sqlite3.Connection) -> int:
    tables = _tables(connection)
    if "ledger_events" in tables:
        return int(
            connection.execute(
                "SELECT COALESCE(MAX(ledger_seq),0) FROM ledger_events"
            ).fetchone()[0]
        )
    if "ledger" in tables:
        return int(
            connection.execute(
                "SELECT COALESCE(MAX(seq),0) FROM ledger"
            ).fetchone()[0]
        )
    return 0


def _database_byte_size(connection: sqlite3.Connection) -> int:
    page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
    page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
    return page_count * page_size


def _uncopied_logical_digest(connection: sqlite3.Connection) -> str:
    """Content-sensitive identity without serialize() or a tempfile copy.

    Counts and watermarks miss same-count payload replacements. Stream every
    user-table row inside the caller's read transaction so WAL checkpointing
    does not change the digest and a same-count UPDATE does.
    """

    hasher = hashlib.sha256()
    tables = sorted(
        name
        for name in _tables(connection)
        if not name.startswith("sqlite_") and '"' not in name
    )
    for table in tables:
        hasher.update(b"T")
        hasher.update(table.encode("utf-8"))
        cursor = connection.execute(f'SELECT * FROM "{table}"')
        columns = tuple(str(item[0]) for item in cursor.description)
        hasher.update(repr(columns).encode("utf-8"))
        for row in cursor:
            for value in row:
                if value is None:
                    hasher.update(b"N")
                elif isinstance(value, bool):
                    hasher.update(b"B")
                    hasher.update(b"1" if value else b"0")
                elif isinstance(value, int):
                    hasher.update(b"I")
                    hasher.update(str(value).encode("ascii"))
                elif isinstance(value, float):
                    hasher.update(b"F")
                    hasher.update(repr(value).encode("ascii"))
                elif isinstance(value, (bytes, memoryview)):
                    payload = bytes(value)
                    hasher.update(b"Y")
                    hasher.update(len(payload).to_bytes(8, "big"))
                    hasher.update(payload)
                else:
                    encoded = str(value).encode("utf-8")
                    hasher.update(b"S")
                    hasher.update(len(encoded).to_bytes(8, "big"))
                    hasher.update(encoded)
            hasher.update(b"R")
    return validate_sha256_digest(
        f"sha256:{hasher.hexdigest()}", field="streamed logical digest"
    )


def _logical_content_digest(
    snapshot: object, connection: sqlite3.Connection
) -> str:
    """Bind store identity without a tempfile copy or a ≳2 GiB serialize.

    A planted ``snapshot_files`` sha256 remains authoritative for tests.
    Small images still use ``Connection.serialize()``. Larger stores use
    schema fingerprint, watermark and append-only row counts.
    """

    files = getattr(snapshot, "snapshot_files", ())
    if files:
        recorded = files[0]
        digest = recorded.get("sha256") if isinstance(recorded, dict) else None
        if isinstance(digest, str):
            return validate_sha256_digest(
                f"sha256:{digest}", field="copied snapshot digest"
            )
    if _database_byte_size(connection) <= _SERIALIZE_MAX_BYTES:
        try:
            return digest_bytes(connection.serialize())
        except sqlite3.OperationalError:
            if not _tables(connection) and _database_byte_size(connection) == 0:
                return digest_bytes(b"")
    return _uncopied_logical_digest(connection)


def _store_descriptor(snapshot: object) -> dict[str, object]:
    connection = snapshot.connection
    tables = _tables(connection)
    logical_content_digest = _logical_content_digest(snapshot, connection)
    migrations = (
        [
            {"version": int(row[0]), "name": str(row[1]), "checksum": str(row[2])}
            for row in connection.execute(
                "SELECT version,name,checksum FROM authority_migrations ORDER BY version"
            )
        ]
        if "authority_migrations" in tables
        else []
    )
    watermark = _watermark(connection)
    # SQLite WAL checkpointing changes physical files without changing the
    # database.  Bind the authority identity to the exact logical image so a
    # sealed packet remains usable after the preparation process closes and a
    # later F4 process reopens the same store.  Physical file observations are
    # retained as evidence, but are deliberately not authority inputs.
    identity: dict[str, object] = {
        "source_path": snapshot.source_path,
        "logical_content_digest": logical_content_digest,
        "schema_fingerprint": _schema_fingerprint(connection),
        "migration_identity": migrations,
        "watermark": watermark,
    }
    return {
        **identity,
        "source_files": list(snapshot.source_files),
        "snapshot_files": list(snapshot.snapshot_files),
        "descriptor_digest": digest_canonical(identity),
    }


def graphiti_store_snapshot_digests(
    *,
    proving_store: str | Path,
    unpublished_store: str | Path,
    authority_store: str | Path,
) -> dict[str, str]:
    """Recompute the three exact read-only store identities used by a packet."""

    with ExitStack() as stack:
        snapshots = {
            "proving": stack.enter_context(read_only_snapshot(proving_store)),
            "unpublished": stack.enter_context(
                read_only_snapshot(unpublished_store)
            ),
            "authority": stack.enter_context(read_only_snapshot(authority_store)),
        }
        return {
            name: str(_store_descriptor(snapshot)["descriptor_digest"])
            for name, snapshot in snapshots.items()
        }


def graphiti_graph_destination_readback(
    *,
    destination_id: str,
    reconciliation: StructuralReconciliationView,
) -> dict[str, object]:
    """Serialise an authenticated existing-facade reconciliation result."""

    if not isinstance(destination_id, str) or not destination_id.strip():
        raise ValueError("graph destination identity is invalid")
    if not isinstance(reconciliation, StructuralReconciliationView):
        raise TypeError("graph readback requires typed structural reconciliation")
    return {
        "destination_id": destination_id,
        "family_id": reconciliation.family_id,
        "generation_id": str(reconciliation.generation_id),
        "checkpoint_ledger_seq": reconciliation.checkpoint_ledger_seq,
        "projection_state_digest": reconciliation.projection_state_digest,
        "serving_time": reconciliation.serving_time.to_text(),
    }


def graphiti_graph_destination_identity(
    readback: Mapping[str, object],
) -> dict[str, object]:
    """Return durable graph identity without its observation timestamp."""

    identity = dict(readback)
    if "serving_time" not in identity:
        raise ValueError("graph readback serving-time observation is missing")
    del identity["serving_time"]
    return identity


def _authority_snapshot_evidence(
    connection: sqlite3.Connection,
) -> tuple[dict[str, object], list[str]]:
    tables = _tables(connection)
    required = {
        "authority_migrations",
        "ledger_events",
        "extraction_proposals",
        "graphiti_adapter_attempts",
        "entity_resolution_decisions",
        "editorial_relation_decisions",
        "projection_ontology_contracts",
        "projection_mapping_contracts",
        "projection_family_definitions",
        "projection_families",
        "projection_generations",
    }
    if not required.issubset(tables):
        return {"schema_present": False}, ["AUTHORITY_STORE_SCHEMA_INCOMPLETE"]
    actual_migrations = [
        (int(row[0]), str(row[1]), str(row[2]))
        for row in connection.execute(
            "SELECT version,name,checksum FROM authority_migrations ORDER BY version"
        )
    ]
    expected_migrations = [
        (int(item.version), str(item.name), str(item.checksum)) for item in MIGRATIONS
    ]
    migration_valid = bool(actual_migrations) and actual_migrations == expected_migrations[
        : len(actual_migrations)
    ]
    max_version = max((item[0] for item in actual_migrations), default=0)
    user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    # Full-file page walks exceed the one-minute operational seal bound.
    # Schema and migration history remain.
    integrity = "omitted"
    blockers: list[str] = []
    if not migration_valid or user_version != max_version or max_version < 16:
        blockers.append("AUTHORITY_MIGRATION_HISTORY_INVALID")

    graph_rows = connection.execute(
        "SELECT g.generation_id,g.state,g.validated_through_ledger_seq,"
        "g.family_id,d.projector_version,o.ontology_id,o.ontology_version,"
        "m.mapping_id,m.mapping_version,o.contract_digest,m.contract_digest,"
        "d.definition_digest FROM projection_generations AS g "
        "JOIN projection_families AS f ON f.family_id=g.family_id "
        "JOIN projection_family_definitions AS d "
        "ON d.definition_digest=f.definition_digest "
        "JOIN projection_ontology_contracts AS o "
        "ON o.contract_digest=d.ontology_contract_digest "
        "JOIN projection_mapping_contracts AS m "
        "ON m.contract_digest=d.mapping_contract_digest "
        "WHERE g.family_id=? AND g.state='ACTIVE'",
        (INCREMENT4_ADMITTED_FAMILY_ID,),
    ).fetchall()
    graph_readback: dict[str, object] | None = None
    if len(graph_rows) != 1:
        blockers.append("ACTIVE_GRAPH_GENERATION_READBACK_INVALID")
    else:
        row = graph_rows[0]
        validated_through = row[2]
        if (
            isinstance(validated_through, bool)
            or not isinstance(validated_through, int)
            or validated_through < 0
        ):
            blockers.append("ACTIVE_GRAPH_WATERMARK_INVALID")
            validated_through = -1
        graph_readback = {
            "generation_id": str(row[0]),
            "state": str(row[1]),
            "validated_through_ledger_seq": validated_through,
            "family_id": str(row[3]),
            "projector_version": str(row[4]),
            "ontology_id": str(row[5]),
            "ontology_version": str(row[6]),
            "mapping_id": str(row[7]),
            "mapping_version": str(row[8]),
            "ontology_contract_digest": str(row[9]),
            "mapping_contract_digest": str(row[10]),
            "family_definition_digest": str(row[11]),
        }
        expected_graph = {
            "family_id": INCREMENT4_ADMITTED_FAMILY_ID,
            "projector_version": INCREMENT4_ADMITTED_PROJECTOR_VERSION,
            "ontology_id": INCREMENT4_ADMITTED_ONTOLOGY_ID,
            "ontology_version": INCREMENT4_ADMITTED_ONTOLOGY_VERSION,
            "mapping_id": INCREMENT4_ADMITTED_MAPPING_ID,
            "mapping_version": INCREMENT4_ADMITTED_MAPPING_VERSION,
        }
        if any(graph_readback.get(key) != value for key, value in expected_graph.items()):
            blockers.append("ACTIVE_GRAPH_IDENTITY_INVALID")
        ontology = increment4_admitted_ontology_v1()
        mapping = increment4_admitted_mapping_v1(ontology)
        family = increment4_admitted_family_v1(ontology, mapping)
        if (
            graph_readback["ontology_contract_digest"] != ontology.contract_digest
            or graph_readback["mapping_contract_digest"] != mapping.contract_digest
            or graph_readback["family_definition_digest"] != family.digest
        ):
            blockers.append("ACTIVE_GRAPH_CONTRACT_DIGEST_INVALID")
        try:
            ProjectionGenerationId.parse(str(graph_readback["generation_id"]))
        except (TypeError, ValueError):
            blockers.append("ACTIVE_GRAPH_GENERATION_ID_INVALID")
        if (
            graph_readback["validated_through_ledger_seq"] < 0
            or graph_readback["validated_through_ledger_seq"]
            > int(
                connection.execute(
                    "SELECT COALESCE(MAX(ledger_seq),0) FROM ledger_events"
                ).fetchone()[0]
            )
        ):
            blockers.append("ACTIVE_GRAPH_WATERMARK_INVALID")
        for field in (
            "ontology_contract_digest",
            "mapping_contract_digest",
            "family_definition_digest",
        ):
            try:
                validate_sha256_digest(graph_readback[field], field=field)
            except (TypeError, ValueError):
                blockers.append("ACTIVE_GRAPH_CONTRACT_DIGEST_INVALID")
                break
    value = {
        "schema_present": True,
        "migration_history_digest": digest_canonical(actual_migrations),
        "migration_history_valid": migration_valid,
        "user_version": user_version,
        "watermark": int(
            connection.execute(
                "SELECT COALESCE(MAX(ledger_seq),0) FROM ledger_events"
            ).fetchone()[0]
        ),
        "integrity_check": integrity,
        "active_projection_authority": graph_readback,
    }
    return value, blockers


def _has_exact_durable_proposal_authority(
    authority: sqlite3.Connection | None,
    unpublished: sqlite3.Connection,
    ingest_ids: tuple[str, ...],
) -> bool:
    retained_rows = [
        unpublished.execute(
            "SELECT ingest.proposal_count,receipt.receipt_json "
            "FROM unpublished_graphiti_ingest AS ingest "
            "JOIN unpublished_graphiti_receipts AS receipt USING(ingest_id) "
            "WHERE ingest.ingest_id=?",
            (ingest_id,),
        ).fetchone()
        for ingest_id in ingest_ids
    ]
    if any(row is None for row in retained_rows):
        return False
    if all(int(row[0]) == 0 for row in retained_rows if row is not None):
        return True
    if authority is None:
        return False
    authority_tables = _tables(authority)
    if not {
        "graphiti_adapter_attempts",
        "extraction_outputs",
        "extraction_proposals",
    }.issubset(authority_tables):
        return False
    for ingest_id, retained in zip(ingest_ids, retained_rows, strict=True):
        assert retained is not None
        proposal_count = int(retained[0])
        if proposal_count == 0:
            continue
        try:
            receipt = json.loads(str(retained[1]))
            attempt_number = receipt["attempt_number"]
            proposals = receipt["proposals"]
            if (
                isinstance(attempt_number, bool)
                or not isinstance(attempt_number, int)
                or attempt_number <= 0
                or not isinstance(proposals, list)
                or len(proposals) != proposal_count
            ):
                return False
            attempt_id = str(attempt_ids(ingest_id, attempt_number)[0])
            attempt = authority.execute(
                "SELECT outcome,proposal_count,proposal_set_id,extraction_output_id,"
                "run_id,run_version_id FROM graphiti_adapter_attempts "
                "WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if (
                attempt is None
                or str(attempt[0]) != "COMPLETE"
                or receipt.get("outcome") != str(attempt[0])
                or int(attempt[1]) != proposal_count
                or attempt[2] is None
                or attempt[3] is None
            ):
                return False
            output = authority.execute(
                "SELECT run_id,run_version_id,canonical_bytes,canonical_digest "
                "FROM extraction_outputs WHERE output_id=?",
                (str(attempt[3]),),
            ).fetchone()
            if (
                output is None
                or str(output[0]) != str(attempt[4])
                or str(output[1]) != str(attempt[5])
            ):
                return False
            raw_bytes = bytes(output[2])
            raw_value = json.loads(raw_bytes)
            if (
                not isinstance(raw_value, dict)
                or canonical_json_bytes(raw_value) != raw_bytes
                or digest_bytes(raw_bytes) != str(output[3])
            ):
                return False
            retained_raw_digest = raw_value.pop("raw_output_digest", None)
            terminal_raw_digest = receipt.get("raw_output_digest")
            if (
                retained_raw_digest != terminal_raw_digest
                or retained_raw_digest
                != digest_bytes(canonical_json_bytes(raw_value))
            ):
                return False
            exact_raw_fields = (
                "attempt_number",
                "provider_attempt_number",
                "generation_id",
                "episode_uuid",
                "temporal_basis",
                "reference_time",
                "proposals",
                "passages",
                "entities",
                "relations",
                "proposal_count",
                "entity_count",
                "relation_count",
            )
            if any(
                receipt.get(field) != raw_value.get(field)
                for field in exact_raw_fields
            ):
                return False
            envelopes = authority.execute(
                "SELECT local_id,semantic_digest,output_id,run_id,run_version_id "
                "FROM extraction_proposals WHERE proposal_set_id=? "
                "ORDER BY local_id",
                (str(attempt[2]),),
            ).fetchall()
            raw_by_local_id = {
                str(item["local_id"]): digest_canonical(item)
                for item in proposals
                if isinstance(item, Mapping) and isinstance(item.get("local_id"), str)
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, sqlite3.Error):
            return False
        if len(raw_by_local_id) != proposal_count or len(envelopes) != proposal_count:
            return False
        if any(
            raw_by_local_id.get(str(row[0])) != str(row[1])
            or str(row[2]) != str(attempt[3])
            or str(row[3]) != str(attempt[4])
            or str(row[4]) != str(attempt[5])
            for row in envelopes
        ):
            return False
    return True


def _proving_accounting(
    connection: sqlite3.Connection,
) -> tuple[dict[str, object], list[str]]:
    required = {
        "proving_runs",
        "proving_gates",
        "proving_observations",
        "proving_source_health",
    }
    if not required.issubset(_tables(connection)):
        return {"schema_present": False}, ["PROVING_ACCOUNTABILITY_SCHEMA_MISSING"]
    run = connection.execute(
        "SELECT run_id,started_at,publication,public_dispatch,openrouter_invoked,"
        "spend_gbp_minor FROM proving_runs ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    if run is None:
        return {"schema_present": True, "latest_run_id": None}, [
            "PROVING_RUN_MISSING"
        ]
    run_id = str(run[0])
    gate_rows = connection.execute(
        "SELECT gate_id,status FROM proving_gates WHERE run_id=? ORDER BY gate_id",
        (run_id,),
    ).fetchall()
    gate_statuses = {str(gate_id): str(status) for gate_id, status in gate_rows}
    source_rows = connection.execute(
        "SELECT source_id,status,endpoint,reason FROM proving_source_health "
        "WHERE run_id=? ORDER BY source_id",
        (run_id,),
    ).fetchall()
    retained_source_ids = {str(row[0]) for row in source_rows}
    missing_source_ids = sorted(set(SOURCE_IDS) - retained_source_ids)
    unexpected_source_ids = sorted(retained_source_ids - set(SOURCE_IDS))
    sources: list[dict[str, object]] = []
    unaccounted: list[str] = []
    successful = held = 0
    for source_id, status, endpoint, reason in source_rows:
        source_id = str(source_id)
        status = str(status)
        endpoint = str(endpoint)
        observation = connection.execute(
            "SELECT fetched_at,url,status_code,body_digest,item_count,error "
            "FROM proving_observations WHERE run_id=? AND source_id=? "
            "ORDER BY fetched_at DESC LIMIT 1",
            (run_id, source_id),
        ).fetchone()
        expected_endpoint = SOURCE_URLS.get(source_id)
        body_digest_valid = False
        if observation is not None:
            try:
                resolve_observation_body(connection, observation[3])
                body_digest_valid = True
            except (ProvingError, sqlite3.Error):
                # Preserve typed failed readiness for corrupt or missing bodies.
                pass
        rights_gate_status = gate_statuses.get(
            RIGHTS_GATE_BY_SOURCE.get(source_id, "")
        )
        observation_success = bool(
            status == "ACTIVE"
            and observation is not None
            and int(observation[2]) == 200
            and observation[5] is None
            and str(observation[1]) == endpoint
            and endpoint == expected_endpoint
            and body_digest_valid
            and rights_gate_status == "PASS"
        )
        typed_hold = bool(
            status in {"DEGRADED", "HELD", "BLOCKED"}
            and reason
            and endpoint == expected_endpoint
        )
        if observation_success:
            successful += 1
        elif typed_hold:
            held += 1
        else:
            unaccounted.append(source_id)
        sources.append(
            {
                "source_id": source_id,
                "status": status,
                "endpoint": endpoint,
                "reason": None if reason is None else str(reason),
                "observation_success": observation_success,
                "typed_hold": typed_hold,
                "rights_gate_status": rights_gate_status,
                "observed_at": None if observation is None else str(observation[0]),
                "body_digest": None if observation is None else str(observation[3]),
                "body_digest_valid": body_digest_valid,
                "item_count": None if observation is None else int(observation[4]),
            }
        )
    missing_gate_ids = sorted(set(PROVING_GATES) - set(gate_statuses))
    non_pass_gates = sorted(
        gate_id for gate_id, status in gate_statuses.items() if status != "PASS"
    )
    external_effects = {
        "publication": int(run[2]),
        "public_dispatch": int(run[3]),
        "openrouter_invoked": int(run[4]),
        "spend_gbp_minor": int(run[5]),
    }
    blockers: list[str] = []
    if missing_source_ids or unexpected_source_ids:
        blockers.append("PROVING_SOURCE_MANIFEST_DIFFERS")
    if unaccounted:
        blockers.append("PROVING_SOURCE_UNACCOUNTED")
    if missing_gate_ids:
        blockers.append("PROVING_GATE_MISSING")
    if non_pass_gates:
        blockers.append("PROVING_GATE_NOT_PASS")
    if any(external_effects.values()):
        blockers.append("PROVING_RUN_EXTERNAL_EFFECT_PRESENT")
    return {
        "schema_present": True,
        "latest_run_id": run_id,
        "started_at": str(run[1]),
        "source_count": len(source_rows),
        "expected_source_ids": list(SOURCE_IDS),
        "source_manifest_digest": digest_canonical(
            {
                "source_ids": list(SOURCE_IDS),
                "source_urls": SOURCE_URLS,
            }
        ),
        "missing_source_ids": missing_source_ids,
        "unexpected_source_ids": unexpected_source_ids,
        "successful_observation_count": successful,
        "typed_hold_count": held,
        "unaccounted_source_ids": unaccounted,
        "sources": sources,
        "gate_status_counts": dict(sorted(Counter(gate_statuses.values()).items())),
        "missing_gate_ids": missing_gate_ids,
        "non_pass_gate_ids": non_pass_gates,
        "external_effects": external_effects,
    }, blockers


def _event_accounting(
    connection: sqlite3.Connection,
    *,
    gap_decisions: tuple[GraphitiEventRepairDecision, ...] = (),
) -> tuple[dict[str, object], list[str]]:
    blockers: list[str] = []
    required = {
        "unpublished_effective_revision_landed",
        "unpublished_graphiti_revision_events",
    }
    if not required.issubset(_tables(connection)):
        return {
            "landed_revision_count": 0,
            "event_count": 0,
            "missing_event_ledger_sequences": [],
            "orphan_event_ledger_sequences": [],
            "one_to_one": False,
        }, ["EVENT_ACCOUNTING_SCHEMA_MISSING"]
    landed = {
        int(row[0]): str(row[1])
        for row in connection.execute(
            "SELECT ledger.seq, landed.ledger_digest "
            "FROM unpublished_effective_revision_landed AS landed "
            "JOIN ledger ON ledger.digest=landed.ledger_digest "
            "WHERE NOT (landed.legacy_v10=1 AND EXISTS ("
            "SELECT 1 FROM unpublished_effective_revision_landed AS marker "
            "WHERE marker.legacy_v10=0 "
            "AND marker.source_id=landed.source_id "
            "AND marker.item_key=landed.item_key "
            "AND marker.revision_digest=landed.revision_digest "
            "AND marker.first_observed_at=landed.first_observed_at "
            "AND (marker.published_at<>'' OR marker.updated_at<>'')))"
        )
    }
    events = {
        int(row[0]): str(row[1])
        for row in connection.execute(
            "SELECT ledger_seq, ledger_digest FROM unpublished_graphiti_revision_events"
        )
    }
    missing = sorted(set(landed) - set(events))
    orphan = sorted(set(events) - set(landed))
    contradictions = sorted(
        seq
        for seq in set(landed) & set(events)
        if landed[seq] != events[seq]
    )
    decisions_by_seq = {item.ledger_seq: item for item in gap_decisions}
    classified_sequences = set(decisions_by_seq)
    missing_sequences = set(missing)
    classification_complete = classified_sequences == missing_sequences
    projection_candidates = sorted(
        seq
        for seq, item in decisions_by_seq.items()
        if item.disposition is GraphitiEventRepairDisposition.PROJECT_EVENT
    )
    held = sorted(
        seq
        for seq, item in decisions_by_seq.items()
        if item.disposition is GraphitiEventRepairDisposition.HOLD
    )
    unclassified = sorted(
        seq
        for seq, item in decisions_by_seq.items()
        if item.disposition is GraphitiEventRepairDisposition.UNCLASSIFIED
    )
    if not classification_complete:
        blockers.append("EVENT_GAP_CLASSIFICATION_INCOMPLETE")
    if projection_candidates:
        blockers.append("LANDED_REVISION_EVENT_MISSING")
    if unclassified:
        blockers.append("LANDED_REVISION_EVENT_UNCLASSIFIED")
    if orphan:
        blockers.append("GRAPHITI_EVENT_ORPHANED")
    if contradictions:
        blockers.append("LANDED_EVENT_IDENTITY_CONTRADICTION")
    return {
        "landed_revision_count": len(landed),
        "event_count": len(events),
        "missing_event_ledger_sequences": missing,
        "projection_candidate_ledger_sequences": projection_candidates,
        "held_missing_event_ledger_sequences": held,
        "unclassified_missing_event_ledger_sequences": unclassified,
        "gap_classification_complete": classification_complete,
        "orphan_event_ledger_sequences": orphan,
        "contradictory_ledger_sequences": contradictions,
        "one_to_one": not (missing or orphan or contradictions),
        "eligible_one_to_one": not (
            projection_candidates
            or unclassified
            or orphan
            or contradictions
            or not classification_complete
        ),
    }, blockers


def _events_and_receipts(
    connection: sqlite3.Connection,
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    blockers: list[str] = []
    tables = _tables(connection)
    required = {
        "unpublished_graphiti_revision_events",
        "unpublished_graphiti_ingest",
        "unpublished_graphiti_receipts",
        "unpublished_graphiti_attempt_receipts",
    }
    if not required.issubset(tables):
        return (
            {"state_counts": {}, "integrity_failures": []},
            {"terminal_ingest_count": 0, "integrity_failures": []},
            ["TERMINAL_RECEIPT_SCHEMA_MISSING"],
        )
    rows = connection.execute(
        "SELECT event_id,ledger_seq,ledger_digest,state,proposal_count,unit_count,"
        "manifest_json,manifest_digest "
        "FROM unpublished_graphiti_revision_events ORDER BY ledger_seq"
    ).fetchall()
    states = Counter(str(row[3]) for row in rows)
    unknown = sorted(set(states) - set(GRAPHITI_EVENT_STATES))
    if unknown:
        blockers.append("UNKNOWN_EVENT_STATE")
    event_failures: list[dict[str, str]] = []
    receipt_failures: list[dict[str, str]] = []
    terminal_ingests: set[str] = set()
    zero_proposal = 0
    for (
        event_id,
        ledger_seq,
        ledger_digest,
        state,
        event_proposals,
        unit_count,
        manifest_json,
        manifest_digest,
    ) in rows:
        try:
            manifest = json.loads(str(manifest_json))
            unit_refs = manifest.get("unit_refs")
            landed_ingest_ids = manifest.get("landed_ingest_ids")
            if (
                not isinstance(manifest, dict)
                or digest_canonical(manifest) != str(manifest_digest)
                or manifest.get("ledger_seq") != int(ledger_seq)
                or manifest.get("ledger_digest") != str(ledger_digest)
                or str(event_id) != str(ledger_digest)
                or not isinstance(unit_refs, list)
                or not all(isinstance(item, dict) for item in unit_refs)
                or not isinstance(landed_ingest_ids, list)
                or not all(isinstance(item, str) for item in landed_ingest_ids)
            ):
                raise ValueError
            ingest_ids = [item.get("ingest_id") for item in unit_refs]
            if (
                len(unit_refs) != int(unit_count)
                or not all(isinstance(item, str) and item for item in ingest_ids)
                or len(set(ingest_ids)) != len(ingest_ids)
                or len(set(landed_ingest_ids)) != len(landed_ingest_ids)
                or (
                    landed_ingest_ids
                    and ingest_ids
                    and tuple(ingest_ids) != tuple(landed_ingest_ids)
                )
            ):
                raise ValueError
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            event_failures.append(
                {"event_id": str(event_id), "reason": "MALFORMED_EVENT_MANIFEST"}
            )
            continue
        if state != "TERMINAL":
            continue
        if not ingest_ids:
            event_failures.append(
                {
                    "event_id": str(event_id),
                    "reason": "TERMINAL_EVENT_HAS_NO_RESOLVED_UNITS",
                }
            )
            continue
        proposal_sum = 0
        for ingest_id in ingest_ids:
            assert isinstance(ingest_id, str)
            if ingest_id in terminal_ingests:
                receipt_failures.append(
                    {
                        "ingest_id": ingest_id,
                        "reason": "INGEST_BOUND_TO_MULTIPLE_TERMINAL_EVENTS",
                    }
                )
            terminal_ingests.add(ingest_id)
            row = connection.execute(
                "SELECT ingest.outcome,ingest.proposal_count,ingest.entity_count,"
                "ingest.relation_count,ingest.receipt_digest,receipt.receipt_json,"
                "attempt.receipt_digest,attempt.receipt_json "
                "FROM unpublished_graphiti_ingest "
                "AS ingest LEFT JOIN unpublished_graphiti_receipts AS receipt "
                "USING(ingest_id) LEFT JOIN unpublished_graphiti_attempt_receipts "
                "AS attempt ON attempt.ingest_id=ingest.ingest_id "
                "AND attempt.receipt_digest=ingest.receipt_digest "
                "WHERE ingest.ingest_id=?",
                (ingest_id,),
            ).fetchone()
            if row is None or row[0] not in {"COMPLETE", "PARTIAL"} or row[5] is None:
                receipt_failures.append(
                    {
                        "ingest_id": ingest_id,
                        "reason": "TERMINAL_INGEST_OR_RECEIPT_MISSING",
                    }
                )
                continue
            try:
                receipt = json.loads(str(row[5]))
            except json.JSONDecodeError:
                receipt_failures.append(
                    {"ingest_id": ingest_id, "reason": "RECEIPT_JSON_INVALID"}
                )
                continue
            proposal_sum += int(row[1])
            proposals = receipt.get("proposals") if isinstance(receipt, dict) else None
            if not isinstance(proposals, list) or len(proposals) != int(row[1]):
                receipt_failures.append(
                    {
                        "ingest_id": ingest_id,
                        "reason": "RECEIPT_PROPOSAL_COUNT_CONTRADICTION",
                    }
                )
                continue
            unsigned_receipt = dict(receipt)
            supplied_digest = unsigned_receipt.pop("receipt_digest", None)
            if (
                supplied_digest != str(row[4])
                or digest_canonical(unsigned_receipt) != str(row[4])
            ):
                receipt_failures.append(
                    {
                        "ingest_id": ingest_id,
                        "reason": "RECEIPT_ENVELOPE_DIGEST_CONTRADICTION",
                    }
                )
                continue
            if row[6] != row[4] or row[7] != row[5]:
                receipt_failures.append(
                    {
                        "ingest_id": ingest_id,
                        "reason": "ATTEMPT_RECEIPT_COPY_CONTRADICTION",
                    }
                )
                continue
            raw_digest = receipt.get("raw_output_digest")
            try:
                validate_sha256_digest(
                    raw_digest,
                    field="Graphiti receipt raw output digest",
                )
            except (TypeError, ValueError):
                receipt_failures.append(
                    {
                        "ingest_id": ingest_id,
                        "reason": "RAW_OUTPUT_DIGEST_INVALID",
                    }
                )
                continue
            if (
                receipt.get("ingest_id") != ingest_id
                or receipt.get("outcome") != row[0]
                or receipt.get("proposal_count") != int(row[1])
                or receipt.get("entity_count") != int(row[2])
                or receipt.get("relation_count") != int(row[3])
            ):
                receipt_failures.append(
                    {
                        "ingest_id": ingest_id,
                        "reason": "RECEIPT_INGEST_BINDING_CONTRADICTION",
                    }
                )
                continue
            if int(row[1]) == 0:
                zero_proposal += 1
        if event_proposals is None or int(event_proposals) != proposal_sum:
            receipt_failures.append(
                {
                    "event_id": str(event_id),
                    "reason": "EVENT_PROPOSAL_COUNT_CONTRADICTION",
                }
            )
    nonterminal = sum(count for state, count in states.items() if state != "TERMINAL")
    if event_failures:
        blockers.append("EVENT_MANIFEST_INTEGRITY_FAILURE")
    if receipt_failures:
        blockers.append("TERMINAL_RECEIPT_INTEGRITY_FAILURE")
    return {
        "state_counts": {
            key: states.get(key, 0) for key in sorted(GRAPHITI_EVENT_STATES)
        },
        "terminal_event_count": states.get("TERMINAL", 0),
        "nonterminal_event_count": nonterminal,
        "integrity_failures": event_failures,
    }, {
        "terminal_ingest_count": len(terminal_ingests),
        "zero_proposal_success_count": zero_proposal,
        "zero_proposal_is_success": True,
        "integrity_failures": receipt_failures,
    }, blockers


def _historical_partition(
    proving: sqlite3.Connection,
    unpublished: sqlite3.Connection,
    *,
    authority: sqlite3.Connection | None,
    observed_at: datetime,
    event_evidence: Mapping[str, object],
    receipt_evidence: Mapping[str, object],
    event_gap_decisions: tuple[GraphitiEventRepairDecision, ...] = (),
    resolved_units: tuple[CorpusIngestUnit, ...] = (),
    unit_resolution_available: bool = True,
    excluded_event_ids: frozenset[str] = frozenset(),
    pre_frontier_hold_watermark: int | None = None,
) -> tuple[dict[str, object], list[str]]:
    """Partition every effective landed revision without provider effects."""

    required = {
        "ledger",
        "unpublished_effective_revision_landed",
        "unpublished_graphiti_revision_events",
    }
    if not required.issubset(_tables(unpublished)):
        return {
            "universe_count": 0,
            "partitioned_count": 0,
            "disjoint": False,
            "total": False,
            "categories": {},
        }, ["HISTORICAL_PARTITION_SCHEMA_MISSING"]

    landed_rows = unpublished.execute(
        "SELECT ledger.seq,ledger.digest,landed.source_id,landed.item_key,"
        "landed.revision_digest,landed.published_at,landed.updated_at "
        "FROM unpublished_effective_revision_landed AS landed "
        "JOIN ledger ON ledger.digest=landed.ledger_digest "
        "WHERE NOT (landed.legacy_v10=1 AND EXISTS ("
        "SELECT 1 FROM unpublished_effective_revision_landed AS marker "
        "WHERE marker.legacy_v10=0 "
        "AND marker.source_id=landed.source_id "
        "AND marker.item_key=landed.item_key "
        "AND marker.revision_digest=landed.revision_digest "
        "AND marker.first_observed_at=landed.first_observed_at "
        "AND (marker.published_at<>'' OR marker.updated_at<>''))) "
        "ORDER BY ledger.seq"
    ).fetchall()
    event_rows = unpublished.execute(
        "SELECT event_id,ledger_seq,ledger_digest,source_id,item_key,"
        "revision_digest,published_at,updated_at,unit_count,manifest_json,"
        "manifest_digest,state,attempt_count,available_at,claim_owner,"
        "claim_expires_at,provider_dispatched "
        "FROM unpublished_graphiti_revision_events ORDER BY ledger_seq"
    ).fetchall()
    events_by_ledger = {int(row[1]): row for row in event_rows}
    gap_decisions_by_ledger = {
        item.ledger_seq: item for item in event_gap_decisions
    }
    terminal_receipt_schema_present = {
        "unpublished_graphiti_ingest",
        "unpublished_graphiti_receipts",
        "unpublished_graphiti_attempt_receipts",
    }.issubset(_tables(unpublished))

    event_integrity_failures = {
        str(item.get("event_id"))
        for item in event_evidence.get("integrity_failures", [])
        if isinstance(item, Mapping) and item.get("event_id") is not None
    }
    receipt_integrity_failures = {
        str(item.get(key))
        for item in receipt_evidence.get("integrity_failures", [])
        if isinstance(item, Mapping)
        for key in ("event_id", "ingest_id")
        if item.get(key) is not None
    }

    units_by_revision: dict[
        tuple[str, str, str, str, str], list[CorpusIngestUnit]
    ] = {}
    for unit in resolved_units:
        key = (
            unit.source_id,
            unit.item_key,
            unit.revision_digest,
            unit.published_at or "",
            unit.updated_at or "",
        )
        units_by_revision.setdefault(key, []).append(unit)

    assignments: dict[str, list[int]] = {
        category: [] for category in HISTORICAL_PARTITION_CATEGORIES
    }
    reason_counts: dict[str, Counter[str]] = {
        category: Counter() for category in HISTORICAL_PARTITION_CATEGORIES
    }
    candidate_events: list[dict[str, object]] = []
    candidate_source_semantics: dict[int, tuple[str, str]] = {}
    now_text = observed_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    for landed in landed_rows:
        ledger_seq = int(landed[0])
        ledger_digest = str(landed[1])
        event_row = events_by_ledger.get(ledger_seq)
        category: str
        reason: str
        if event_row is None:
            gap = gap_decisions_by_ledger.get(ledger_seq)
            if gap is None or gap.disposition is GraphitiEventRepairDisposition.UNCLASSIFIED:
                category = "UNCLASSIFIED"
                reason = (
                    "EVENT_GAP_CLASSIFICATION_MISSING"
                    if gap is None
                    else str(gap.reason or "EVENT_GAP_UNCLASSIFIED")
                )
            else:
                category = "RIGHTS_OR_INPUT_HELD"
                reason = (
                    "EVENT_PROJECTION_MISSING"
                    if gap.disposition is GraphitiEventRepairDisposition.PROJECT_EVENT
                    else str(gap.reason or "CURRENT_RIGHTS_OR_INPUT_HELD")
                )
        else:
            event_id = str(event_row[0])
            try:
                manifest = json.loads(str(event_row[9]))
                unit_refs = manifest.get("unit_refs")
                landed_ingest_ids = manifest.get("landed_ingest_ids")
                if (
                    not isinstance(manifest, dict)
                    or digest_canonical(manifest) != str(event_row[10])
                    or event_id != ledger_digest
                    or str(event_row[2]) != ledger_digest
                    or tuple(str(value) for value in event_row[3:8])
                    != tuple(str(value) for value in landed[2:7])
                    or manifest.get("ledger_seq") != ledger_seq
                    or manifest.get("ledger_digest") != ledger_digest
                    or not isinstance(unit_refs, list)
                    or not all(isinstance(item, dict) for item in unit_refs)
                    or not isinstance(landed_ingest_ids, list)
                    or not all(isinstance(item, str) for item in landed_ingest_ids)
                ):
                    raise ValueError
                resolved_ingest_ids = tuple(
                    item.get("ingest_id") for item in unit_refs
                )
                if (
                    len(unit_refs) != int(event_row[8])
                    or not all(
                        isinstance(item, str) and item
                        for item in resolved_ingest_ids
                    )
                    or len(set(resolved_ingest_ids)) != len(resolved_ingest_ids)
                    or len(set(landed_ingest_ids)) != len(landed_ingest_ids)
                    or (
                        resolved_ingest_ids
                        and landed_ingest_ids
                        and resolved_ingest_ids != tuple(landed_ingest_ids)
                    )
                ):
                    raise ValueError
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                category = "UNCLASSIFIED"
                reason = "EVENT_MANIFEST_UNVERIFIED"
            else:
                state = str(event_row[11])
                if state == "TERMINAL":
                    terminal_integrity_keys = {event_id, *resolved_ingest_ids}
                    if not terminal_receipt_schema_present:
                        category = "UNCLASSIFIED"
                        reason = "TERMINAL_RECEIPT_SCHEMA_UNAVAILABLE"
                    elif event_id in event_integrity_failures or (
                        terminal_integrity_keys & receipt_integrity_failures
                    ):
                        category = "UNCLASSIFIED"
                        reason = "TERMINAL_OUTCOME_UNVERIFIED"
                    elif not _has_exact_durable_proposal_authority(
                        authority,
                        unpublished,
                        tuple(str(item) for item in resolved_ingest_ids),
                    ):
                        category = "NON_REPLAYABLE_OR_AMBIGUOUS_EFFECT_HOLD"
                        reason = "IMMUTABLE_RAW_PROPOSAL_WITHOUT_DURABLE_ENVELOPE"
                    else:
                        category = "VERIFIED_TERMINAL"
                        reason = "TERMINAL_EVENT_AND_RECEIPTS_VERIFIED"
                elif state not in GRAPHITI_EVENT_STATES:
                    category = "UNCLASSIFIED"
                    reason = "EVENT_STATE_UNRECOGNISED"
                elif (
                    event_id in excluded_event_ids
                    or bool(event_row[16])
                    or state in {"CONFIGURATION_HELD", "DEAD_LETTER"}
                ):
                    category = "NON_REPLAYABLE_OR_AMBIGUOUS_EFFECT_HOLD"
                    reason = "EVENT_HISTORY_OR_EFFECT_REQUIRES_ADJUDICATION"
                elif (
                    state != "QUEUED"
                    or int(event_row[12]) != 0
                    or str(event_row[13]) > now_text
                    or event_row[14] is not None
                    or event_row[15] is not None
                ):
                    category = "RIGHTS_OR_INPUT_HELD"
                    reason = "EVENT_NOT_FRESH_AND_CLAIMABLE"
                else:
                    event = GraphitiRevisionEvent(
                        event_id=event_id,
                        ledger_seq=ledger_seq,
                        source_id=str(event_row[3]),
                        item_key=str(event_row[4]),
                        revision_digest=str(event_row[5]),
                        published_at=str(event_row[6]),
                        updated_at=str(event_row[7]),
                        expected_unit_count=int(event_row[8]),
                        landed_ingest_ids=tuple(landed_ingest_ids),
                        landed_payload_digest=str(
                            manifest.get("landed_payload_digest") or ""
                        ),
                        unit_refs=tuple(unit_refs),
                        state=state,
                        attempt_count=int(event_row[12]),
                        units=(),
                    )
                    revision_key = (
                        event.source_id,
                        event.item_key,
                        event.revision_digest,
                        event.published_at,
                        event.updated_at,
                    )
                    current_units = tuple(units_by_revision.get(revision_key, ()))
                    binding_reason = graphiti_unit_binding_reason(
                        event,
                        current_units,
                    )
                    if binding_reason is None:
                        category = "CURRENT_DISPATCH_PREFLIGHT_CANDIDATE"
                        reason = _CURRENT_PREFLIGHT_REASON
                        current_ingest_ids = [
                            item.ingest_id
                            for item in sorted(
                                current_units,
                                key=lambda item: item.chunk_ordinal,
                            )
                        ]
                        candidate_events.append(
                            {
                                "event_id": event_id,
                                "ledger_seq": ledger_seq,
                                "manifest_digest": str(event_row[10]),
                                "ingest_ids": current_ingest_ids,
                            }
                        )
                        authorities = {
                            (
                                digest_canonical(
                                    {
                                        "item_id": item.authority.item_id,
                                        "source_native_revision_token": None,
                                        "permitted_state_digest": item.revision_digest,
                                    }
                                ),
                                item.authority.revision_id,
                            )
                            for item in current_units
                            if item.authority is not None
                        }
                        if len(authorities) == 1 and all(
                            item.authority is not None for item in current_units
                        ):
                            candidate_source_semantics[ledger_seq] = authorities.pop()
                    else:
                        category = "RIGHTS_OR_INPUT_HELD"
                        reason = binding_reason
        assignments[category].append(ledger_seq)
        reason_counts[category][reason] += 1

    retained_revision_by_semantics: dict[str, str] = {}
    if authority is not None and authority.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_revisions'"
    ).fetchone():
        retained_revision_by_semantics = {
            str(row[1]): str(row[0])
            for row in authority.execute(
                "SELECT revision_id,revision_identity_digest "
                "FROM source_revisions"
            ).fetchall()
        }

    candidates_by_semantics: dict[str, list[tuple[int, str]]] = {}
    for ledger_seq, (revision_semantics, revision_id) in (
        candidate_source_semantics.items()
    ):
        candidates_by_semantics.setdefault(revision_semantics, []).append(
            (ledger_seq, revision_id)
        )
    held_semantic_duplicates: set[int] = set()
    for semantic_key, semantic_candidates in candidates_by_semantics.items():
        ordered = sorted(semantic_candidates)
        retained_revision_id = retained_revision_by_semantics.get(semantic_key)
        if retained_revision_id is None:
            held_semantic_duplicates.update(seq for seq, _revision_id in ordered[1:])
            continue
        exact = [
            seq
            for seq, revision_id in ordered
            if revision_id == retained_revision_id
        ]
        keep = exact[0] if exact else None
        held_semantic_duplicates.update(
            seq for seq, _revision_id in ordered if seq != keep
        )

    candidate_events = _hold_current_preflight_candidates(
        assignments=assignments,
        reason_counts=reason_counts,
        candidate_events=candidate_events,
        held_sequences=held_semantic_duplicates,
        hold_reason="UNCHANGED_CONTENT_REOBSERVATION_REQUIRES_OCCURRENCE",
    )
    frontier_ledger_seq = max((int(row[0]) for row in landed_rows), default=None)
    candidate_sequences = {
        int(candidate["ledger_seq"]) for candidate in candidate_events
    }
    pre_frontier_held: set[int] = set()
    if pre_frontier_hold_watermark is not None:
        if (
            type(pre_frontier_hold_watermark) is not int
            or pre_frontier_hold_watermark not in events_by_ledger
        ):
            raise GraphitiEventReconciliationError(
                "sealed frontier hold watermark is invalid"
            )
        frontier_ledger_seq = pre_frontier_hold_watermark
    if (
        pre_frontier_hold_watermark is not None
        or frontier_ledger_seq in candidate_sequences
    ):
        pre_frontier_held = {
            seq for seq in candidate_sequences if seq < frontier_ledger_seq
        }
        candidate_events = _hold_current_preflight_candidates(
            assignments=assignments,
            reason_counts=reason_counts,
            candidate_events=candidate_events,
            held_sequences=pre_frontier_held,
            hold_reason=PRE_FRONTIER_BACKLOG_HOLD_REASON,
        )

    all_sequences = [seq for values in assignments.values() for seq in values]
    disjoint = len(all_sequences) == len(set(all_sequences))
    universe_sequences = [int(row[0]) for row in landed_rows]
    total = sorted(all_sequences) == sorted(universe_sequences)
    categories = {
        category: {
            "count": len(assignments[category]),
            "ledger_sequences": assignments[category],
            "reason_counts": dict(sorted(reason_counts[category].items())),
            **(
                {"dispatch_authorised": False}
                if category == "CURRENT_DISPATCH_PREFLIGHT_CANDIDATE"
                else {}
            ),
        }
        for category in HISTORICAL_PARTITION_CATEGORIES
    }
    evidence_without_digest: dict[str, object] = {
        "universe": "LEGACY_FILTERED_EFFECTIVE_REVISION_LANDED",
        "universe_count": len(universe_sequences),
        "partitioned_count": len(all_sequences),
        "disjoint": disjoint,
        "total": total,
        "categories": categories,
        "source_semantic_collision_holds": sorted(held_semantic_duplicates),
        **(
            {"pre_frontier_backlog_holds": sorted(pre_frontier_held)}
            if pre_frontier_held
            else {}
        ),
        "current_preflight_candidates": candidate_events,
        "current_preflight_candidate_manifest_digest": digest_canonical(
            candidate_events
        ),
    }
    blockers: list[str] = []
    if not unit_resolution_available:
        blockers.append("CURRENT_RIGHTS_INPUT_RESOLUTION_UNAVAILABLE")
    if not disjoint or not total:
        blockers.append("HISTORICAL_PARTITION_NOT_TOTAL_AND_DISJOINT")
    if assignments["UNCLASSIFIED"]:
        blockers.append("HISTORICAL_OBLIGATION_UNCLASSIFIED")
    return {
        **evidence_without_digest,
        "partition_digest": digest_canonical(evidence_without_digest),
    }, blockers


def _hold_current_preflight_candidates(
    *,
    assignments: dict[str, list[int]],
    reason_counts: dict[str, Counter[str]],
    candidate_events: list[dict[str, object]],
    held_sequences: set[int],
    hold_reason: str,
) -> list[dict[str, object]]:
    """Move selected bindable FRESH candidates onto a hold without draining them."""

    if not held_sequences:
        return candidate_events
    candidate_assignments = assignments["CURRENT_DISPATCH_PREFLIGHT_CANDIDATE"]
    assignments["CURRENT_DISPATCH_PREFLIGHT_CANDIDATE"] = [
        seq for seq in candidate_assignments if seq not in held_sequences
    ]
    assignments["RIGHTS_OR_INPUT_HELD"].extend(sorted(held_sequences))
    assignments["RIGHTS_OR_INPUT_HELD"].sort()
    reason_counts["CURRENT_DISPATCH_PREFLIGHT_CANDIDATE"][
        _CURRENT_PREFLIGHT_REASON
    ] -= len(held_sequences)
    if not reason_counts["CURRENT_DISPATCH_PREFLIGHT_CANDIDATE"][
        _CURRENT_PREFLIGHT_REASON
    ]:
        del reason_counts["CURRENT_DISPATCH_PREFLIGHT_CANDIDATE"][
            _CURRENT_PREFLIGHT_REASON
        ]
    reason_counts["RIGHTS_OR_INPUT_HELD"][hold_reason] += len(held_sequences)
    return [
        candidate
        for candidate in candidate_events
        if int(candidate["ledger_seq"]) not in held_sequences
    ]


def graphiti_operational_partition_snapshot(
    proving: sqlite3.Connection,
    unpublished: sqlite3.Connection,
    *,
    authority: sqlite3.Connection,
    observed_at: datetime,
    pre_frontier_hold_watermark: int | None = None,
) -> dict[str, object]:
    """Return the existing exact dispatch partition as bounded campaign evidence.

    Bindable pre-frontier QUEUED stays held while the frontier is FRESH.
    A completing campaign pins that same cut-off; later arrivals stay visible.
    """

    if observed_at.tzinfo is None:
        raise GraphitiEventReconciliationError(
            "operational partition observation must be timezone-aware"
        )
    resolved_units = load_graphiti_units_from_connection(
        proving,
        evaluated_at=observed_at,
    )
    gap_decisions = classify_graphiti_event_gaps(
        proving,
        unpublished,
        evaluated_at=observed_at,
        resolved_units=resolved_units,
    )
    excluded_event_ids = graphiti_excluded_event_ids(unpublished)
    event_evidence, receipt_evidence, receipt_blockers = _events_and_receipts(
        unpublished
    )
    accounting, accounting_blockers = _event_accounting(
        unpublished,
        gap_decisions=gap_decisions,
    )
    partition, partition_blockers = _historical_partition(
        proving,
        unpublished,
        authority=authority,
        observed_at=observed_at,
        event_evidence=event_evidence,
        receipt_evidence=receipt_evidence,
        event_gap_decisions=gap_decisions,
        resolved_units=resolved_units,
        excluded_event_ids=excluded_event_ids,
        pre_frontier_hold_watermark=pre_frontier_hold_watermark,
    )

    accepted_accounting_blockers = {
        "LANDED_REVISION_EVENT_MISSING",
        "LANDED_REVISION_EVENT_UNCLASSIFIED",
    }
    unexpected_accounting_blockers = sorted(
        set(accounting_blockers) - accepted_accounting_blockers
    )
    gap_unclassified = {
        item.ledger_seq
        for item in gap_decisions
        if item.disposition is GraphitiEventRepairDisposition.UNCLASSIFIED
    }
    categories = partition.get("categories")
    if not isinstance(categories, Mapping):
        raise GraphitiEventReconciliationError(
            "operational partition categories are unavailable"
        )
    unclassified_category = categories.get("UNCLASSIFIED")
    unclassified_sequences = (
        set(unclassified_category.get("ledger_sequences", ()))
        if isinstance(unclassified_category, Mapping)
        else set()
    )
    unexpected_partition_blockers = sorted(
        blocker
        for blocker in partition_blockers
        if not (
            blocker == "HISTORICAL_OBLIGATION_UNCLASSIFIED"
            and unclassified_sequences == gap_unclassified
        )
    )
    if (
        receipt_blockers
        or unexpected_accounting_blockers
        or unexpected_partition_blockers
        or unclassified_sequences != gap_unclassified
        or accounting.get("gap_classification_complete") is not True
    ):
        blockers = sorted(
            {
                *receipt_blockers,
                *unexpected_accounting_blockers,
                *unexpected_partition_blockers,
                *(
                    ("HISTORICAL_OBLIGATION_UNCLASSIFIED",)
                    if unclassified_sequences != gap_unclassified
                    else ()
                ),
                *(
                    ("EVENT_GAP_CLASSIFICATION_INCOMPLETE",)
                    if accounting.get("gap_classification_complete") is not True
                    else ()
                ),
            }
        )
        raise GraphitiEventReconciliationError(
            "operational partition integrity differs: " + ",".join(blockers)
        )

    landed_rows = unpublished.execute(
        "SELECT ledger.seq,ledger.digest,landed.at "
        "FROM unpublished_effective_revision_landed AS landed "
        "JOIN ledger ON ledger.digest=landed.ledger_digest "
        "WHERE NOT (landed.legacy_v10=1 AND EXISTS ("
        "SELECT 1 FROM unpublished_effective_revision_landed AS marker "
        "WHERE marker.legacy_v10=0 "
        "AND marker.source_id=landed.source_id "
        "AND marker.item_key=landed.item_key "
        "AND marker.revision_digest=landed.revision_digest "
        "AND marker.first_observed_at=landed.first_observed_at "
        "AND (marker.published_at<>'' OR marker.updated_at<>''))) "
        "ORDER BY ledger.seq"
    ).fetchall()
    landed = {
        int(ledger_seq): {
            "event_id": str(event_id),
            "landed_at": str(landed_at),
        }
        for ledger_seq, event_id, landed_at in landed_rows
    }
    for landing in landed.values():
        try:
            landed_at = datetime.fromisoformat(
                str(landing["landed_at"]).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise GraphitiEventReconciliationError(
                "operational landing time is invalid"
            ) from exc
        if landed_at.tzinfo is None or landed_at.astimezone(UTC) > observed_at:
            raise GraphitiEventReconciliationError(
                "operational landing time differs from the observation"
            )
    candidate_by_sequence = {
        int(item["ledger_seq"]): item
        for item in partition.get("current_preflight_candidates", ())
        if isinstance(item, Mapping)
    }
    actionable: list[dict[str, object]] = []
    for ledger_seq, candidate in candidate_by_sequence.items():
        landing = landed.get(ledger_seq)
        if landing is None or landing["event_id"] != candidate.get("event_id"):
            raise GraphitiEventReconciliationError(
                "operational candidate landing identity differs"
            )
        actionable.append(
            {
                "ledger_seq": ledger_seq,
                "event_id": str(candidate["event_id"]),
                "landed_at": str(landing["landed_at"]),
                "kind": "FRESH_EVENT",
                "manifest_digest": str(candidate["manifest_digest"]),
                "ingest_ids": list(candidate["ingest_ids"]),
            }
        )
    for decision in gap_decisions:
        if decision.disposition not in {
            GraphitiEventRepairDisposition.PROJECT_EVENT,
            GraphitiEventRepairDisposition.UNCLASSIFIED,
        }:
            continue
        landing = landed.get(decision.ledger_seq)
        if landing is None or landing["event_id"] != decision.event_id:
            raise GraphitiEventReconciliationError(
                "operational event-gap landing identity differs"
            )
        actionable.append(
            {
                "ledger_seq": decision.ledger_seq,
                "event_id": decision.event_id,
                "landed_at": str(landing["landed_at"]),
                "kind": (
                    "PROJECT_EVENT_GAP"
                    if decision.disposition
                    is GraphitiEventRepairDisposition.PROJECT_EVENT
                    else "UNCLASSIFIED_GAP"
                ),
            }
        )

    projectable_or_unclassified_gaps = {
        item.ledger_seq
        for item in gap_decisions
        if item.disposition
        in {
            GraphitiEventRepairDisposition.PROJECT_EVENT,
            GraphitiEventRepairDisposition.UNCLASSIFIED,
        }
    }
    hold_reasons = {
        item.ledger_seq: str(item.reason or "CURRENT_RIGHTS_OR_INPUT_HELD")
        for item in gap_decisions
        if item.disposition is GraphitiEventRepairDisposition.HOLD
    }
    hold_reasons.update(
        {
            int(ledger_seq): "UNCHANGED_CONTENT_REOBSERVATION_REQUIRES_OCCURRENCE"
            for ledger_seq in partition.get("source_semantic_collision_holds", ())
        }
    )
    hold_reasons.update(
        {
            int(ledger_seq): PRE_FRONTIER_BACKLOG_HOLD_REASON
            for ledger_seq in partition.get("pre_frontier_backlog_holds", ())
        }
    )
    for category in (
        "RIGHTS_OR_INPUT_HELD",
        "NON_REPLAYABLE_OR_AMBIGUOUS_EFFECT_HOLD",
    ):
        category_value = categories.get(category)
        if not isinstance(category_value, Mapping):
            raise GraphitiEventReconciliationError(
                "operational hold partition is unavailable"
            )
        for ledger_seq in category_value.get("ledger_sequences", ()):
            sequence = int(ledger_seq)
            if sequence in projectable_or_unclassified_gaps:
                continue
            hold_reasons.setdefault(sequence, category)
    holds = []
    for ledger_seq, reason in sorted(hold_reasons.items()):
        landing = landed.get(ledger_seq)
        if landing is None:
            raise GraphitiEventReconciliationError(
                "operational hold landing identity differs"
            )
        holds.append(
            {
                "ledger_seq": ledger_seq,
                "event_id": str(landing["event_id"]),
                "reason": reason,
            }
        )

    value: dict[str, object] = {
        "observed_at": observed_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "partition_digest": partition["partition_digest"],
        "actionable": sorted(
            actionable,
            key=lambda item: (int(item["ledger_seq"]), str(item["event_id"])),
        ),
        "holds": holds,
    }
    return {**value, "snapshot_digest": digest_canonical(value)}


def _exact_admission_reconciliation(
    connection: sqlite3.Connection,
) -> dict[str, object]:
    """Verify total, disjoint exact-cohort reconciliation membership."""

    queue_ingest_ids = tuple(
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT ingest_id FROM unpublished_graphiti_admission_queue "
            "ORDER BY ingest_id"
        )
    )
    reconciliation_rows = connection.execute(
        "SELECT receipt_digest,projector_family_id,generation_id,"
        "authority_watermark,receipt_json FROM "
        "unpublished_graphiti_projection_reconciliations "
        "ORDER BY reconciled_at,receipt_digest"
    ).fetchall()
    if not queue_ingest_ids and not reconciliation_rows:
        return {
            "schema_version": GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION,
            "cohort_count": 0,
            "covered_ingest_count": 0,
            "queue_ingest_count": 0,
            "latest_generation_id": None,
            "total": True,
            "disjoint": True,
            "cohorts": [],
        }

    covered_ingest_ids: list[str] = []
    generations: list[dict[str, object]] = []
    for (
        receipt_digest,
        family_id,
        generation_id,
        authority_watermark,
        receipt_json,
    ) in reconciliation_rows:
        receipt, binding = graphiti_projection_reconciliation_from_json(
            str(receipt_json)
        )
        if binding is None:
            raise GraphitiAdmissionConsumerError(
                "projection reconciliation lacks exact cohort membership"
            )
        ingest_ids = tuple(str(item) for item in binding["ingest_ids"])
        cohort_digest, _expected_generation_id = (
            graphiti_decided_cohort_generation_identity(
                connection,
                ingest_ids=ingest_ids,
                require_terminal_states=True,
            )
        )
        if (
            receipt.receipt_digest != str(receipt_digest)
            or receipt.projector_family_id != str(family_id)
            or receipt.generation_id != str(generation_id)
            or receipt.authority_watermark != int(authority_watermark)
            or binding["cohort_digest"] != cohort_digest
        ):
            raise GraphitiAdmissionConsumerError(
                "projection reconciliation SQL or cohort identity differs"
            )

        placeholders = ",".join("?" for _ in ingest_ids)
        decision_rows = connection.execute(
            "SELECT decision.action,decision.authority_ledger_seq,"
            "decision.decision_json FROM "
            "unpublished_graphiti_admission_decisions AS decision "
            "JOIN unpublished_graphiti_admission_queue AS queue "
            "USING(proposal_key) WHERE queue.ingest_id IN ("
            + placeholders
            + ") ORDER BY queue.queue_seq",
            ingest_ids,
        ).fetchall()
        admitted_effect_ids: list[str] = []
        decision_watermarks: list[int] = []
        for action, decision_watermark, decision_json in decision_rows:
            decision = graphiti_governed_decision_from_json(str(decision_json))
            if (
                decision.action.value != str(action)
                or decision.authority_ledger_seq != int(decision_watermark)
            ):
                raise GraphitiAdmissionConsumerError(
                    "projection reconciliation decision or terminal state differs"
                )
            decision_watermarks.append(int(decision_watermark))
            if decision.action.value == "ADMIT":
                assert decision.admitted_authority_id is not None
                admitted_effect_ids.append(decision.admitted_authority_id)
        expected_effect_ids = tuple(sorted(admitted_effect_ids))

        projection_rows = connection.execute(
            "SELECT projection.proposal_key,decision.decision_id,"
            "projection.effect_id,projection.authority_watermark,"
            "projection.projector_family_id,projection.generation_id,"
            "projection.schema_version,projection.trust_scope,"
            "projection.receipt_json,projection.receipt_digest "
            "FROM unpublished_graphiti_projection_receipts AS projection "
            "JOIN unpublished_graphiti_admission_queue AS queue "
            "USING(proposal_key) JOIN unpublished_graphiti_admission_decisions "
            "AS decision USING(proposal_key) WHERE queue.ingest_id IN ("
            + placeholders
            + ") ORDER BY projection.effect_id",
            ingest_ids,
        ).fetchall()
        projected_effect_ids: list[str] = []
        generation_bindings: set[tuple[str, str, str, str]] = set()
        for projection_row in projection_rows:
            projection_json = str(projection_row[8])
            projection = graphiti_projection_receipt_from_json(
                projection_json
            )
            projection_value = json.loads(projection_json)
            projection_unsigned = dict(projection_value)
            supplied_projection_digest = projection_unsigned.pop(
                "receipt_digest", None
            )
            if (
                not isinstance(projection_value, dict)
                or canonical_json_bytes(projection_value).decode("utf-8")
                != projection_json
                or supplied_projection_digest
                != digest_canonical(projection_unsigned)
                or projection.proposal_key != str(projection_row[0])
                or projection.decision_id != str(projection_row[1])
                or projection.effect_id != str(projection_row[2])
                or projection.authority_watermark != int(projection_row[3])
                or projection.projector_family_id != str(projection_row[4])
                or projection.generation_id != str(projection_row[5])
                or projection.schema_version != str(projection_row[6])
                or projection.trust_scope != str(projection_row[7])
                or projection.receipt_digest != str(projection_row[9])
                or projection.generation_id != receipt.generation_id
                or not graphiti_projection_receipt_matches_binding(
                    projection, binding
                )
            ):
                raise GraphitiAdmissionConsumerError(
                    "projection receipt differs from exact cohort reconciliation"
                )
            projected_effect_ids.append(projection.effect_id)
            generation_bindings.add(
                (
                    str(projection.source_snapshot_digest),
                    str(projection.validation_digest),
                    str(projection.promotion_digest),
                    str(projection.generation_result_digest),
                )
            )
        if (
            not decision_watermarks
            or len(generation_bindings) > 1
            or receipt.authority_watermark < max(decision_watermarks)
            or receipt.expected_effect_ids != expected_effect_ids
            or receipt.actual_effect_ids != expected_effect_ids
            or tuple(projected_effect_ids) != expected_effect_ids
        ):
            raise GraphitiAdmissionConsumerError(
                "projection effects or authority watermark differ from exact cohort"
            )
        covered_ingest_ids.extend(ingest_ids)
        generations.append(
            {
                "cohort_digest": cohort_digest,
                "generation_id": receipt.generation_id,
                "ingest_ids": list(ingest_ids),
                "effect_count": len(expected_effect_ids),
                "authority_watermark": receipt.authority_watermark,
            }
        )

    disjoint = len(covered_ingest_ids) == len(set(covered_ingest_ids))
    total = set(queue_ingest_ids).issubset(covered_ingest_ids)
    if not disjoint or not total:
        raise GraphitiAdmissionConsumerError(
            "projection reconciliation cohort membership is not total and disjoint"
        )
    return {
        "schema_version": GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION,
        "cohort_count": len(generations),
        "covered_ingest_count": len(covered_ingest_ids),
        "queue_ingest_count": len(queue_ingest_ids),
        "latest_generation_id": generations[-1]["generation_id"],
        "total": total,
        "disjoint": disjoint,
        "cohorts": generations,
    }


def _admission(
    connection: sqlite3.Connection, *, observed_at: datetime
) -> tuple[dict[str, object], list[str]]:
    required = {
        "unpublished_graphiti_ingest",
        "unpublished_graphiti_admission_queue",
        "unpublished_graphiti_admission_decisions",
        "unpublished_graphiti_projection_receipts",
        "unpublished_graphiti_projection_tombstones",
        "unpublished_graphiti_projection_reconciliations",
        "unpublished_graphiti_admission_receipt_failures",
    }
    if not required.issubset(_tables(connection)):
        return {"schema_present": False}, ["ADMISSION_TELEMETRY_SCHEMA_MISSING"]
    value = graphiti_admission_telemetry(
        connection, now=observed_at
    ).canonical_value()
    value["schema_present"] = True
    try:
        exact_reconciliation = _exact_admission_reconciliation(connection)
    except (
        GraphitiAdmissionConsumerError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        sqlite3.Error,
    ):
        exact_reconciliation = {
            "schema_version": GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION,
            "latest_generation_id": None,
            "total": False,
            "disjoint": False,
            "cohorts": [],
        }
    value["telemetry_projection_reconciled"] = value["projection_reconciled"]
    value["exact_cohort_reconciliation"] = exact_reconciliation
    value["projection_reconciled"] = bool(
        value["projection_reconciled"]
        and exact_reconciliation.get("total") is True
        and exact_reconciliation.get("disjoint") is True
    )
    blockers = []
    if value["dead_letter_count"]:
        blockers.append("ADMISSION_DEAD_LETTER_PRESENT")
    if value["integrity_hold_receipt_count"]:
        blockers.append("ADMISSION_INTEGRITY_HOLD_PRESENT")
    if value["admission_backlog"]:
        blockers.append("ADMISSION_BACKLOG_PRESENT")
    if (
        exact_reconciliation.get("total") is not True
        or exact_reconciliation.get("disjoint") is not True
        or value["proposal_denominator"]
        and not value["projection_reconciled"]
    ):
        blockers.append("ADMISSION_PROJECTION_UNRECONCILED")
    return value, blockers


def _spend(connection: sqlite3.Connection) -> tuple[dict[str, object], list[str]]:
    if "unpublished_graphiti_spend" not in _tables(connection):
        return {"schema_present": False}, ["GRAPHITI_SPEND_SCHEMA_MISSING"]
    disposition_integrity_valid = True
    try:
        validate_retained_graphiti_spend_dispositions(connection)
    except (GraphitiSpendReconciliationError, sqlite3.Error):
        disposition_integrity_valid = False
    rows = connection.execute(
        "SELECT status,COUNT(*),COALESCE(SUM(reserved_gbp_microunits),0),"
        "COALESCE(SUM(actual_gbp_microunits),0),"
        "COALESCE(SUM(actual_usd_microunits),0),"
        "SUM(CASE WHEN provider_usage_json IS NOT NULL THEN 1 ELSE 0 END) "
        "FROM unpublished_graphiti_spend GROUP BY status"
    ).fetchall()
    counts = {str(row[0]): int(row[1]) for row in rows}
    reserved_by_status = {str(row[0]): int(row[2]) for row in rows}
    disposition_counts: dict[str, int] = {}
    retained_typed_hold_count = 0
    retained_typed_hold_reserved = 0
    if (
        disposition_integrity_valid
        and "unpublished_graphiti_spend_dispositions" in _tables(connection)
    ):
        retained_rows = connection.execute(
            "SELECT d.disposition,COUNT(*),"
            "COALESCE(SUM(s.reserved_gbp_microunits),0) "
            "FROM unpublished_graphiti_spend s "
            "JOIN unpublished_graphiti_spend_dispositions d "
            "ON d.spend_id=s.spend_id "
            "WHERE s.status IN ('RESERVED','UNRECONCILED') "
            "GROUP BY d.disposition"
        ).fetchall()
        disposition_counts = {str(row[0]): int(row[1]) for row in retained_rows}
        retained_typed_hold_count = sum(int(row[1]) for row in retained_rows)
        retained_typed_hold_reserved = sum(int(row[2]) for row in retained_rows)
        undispositioned = connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(s.reserved_gbp_microunits),0) "
            "FROM unpublished_graphiti_spend s "
            "LEFT JOIN unpublished_graphiti_spend_dispositions d "
            "ON d.spend_id=s.spend_id "
            "WHERE s.status IN ('RESERVED','UNRECONCILED') "
            "AND d.spend_id IS NULL"
        ).fetchone()
    else:
        undispositioned = connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(reserved_gbp_microunits),0) "
            "FROM unpublished_graphiti_spend "
            "WHERE status IN ('RESERVED','UNRECONCILED')"
        ).fetchone()
    undispositioned_count = int(undispositioned[0])
    undispositioned_reserved = int(undispositioned[1])
    return {
        "schema_present": True,
        "status_counts": {
            key: counts.get(key, 0)
            for key in ("RECONCILED", "RESERVED", "UNRECONCILED")
        },
        "reserved_gbp_microunits": sum(
            int(row[2])
            for row in rows
            if row[0] in {"RESERVED", "UNRECONCILED"}
        ),
        "reserved_gbp_microunits_by_status": {
            key: reserved_by_status.get(key, 0)
            for key in ("RECONCILED", "RESERVED", "UNRECONCILED")
        },
        "actual_gbp_microunits": sum(
            int(row[3]) for row in rows if row[0] == "RECONCILED"
        ),
        "actual_usd_microunits": sum(
            int(row[4]) for row in rows if row[0] == "RECONCILED"
        ),
        "provider_usage_record_count": sum(int(row[5]) for row in rows),
        "retained_disposition_integrity_valid": disposition_integrity_valid,
        "unreconciled_attempt_count": (
            counts.get("UNRECONCILED", 0) + counts.get("RESERVED", 0)
        ),
        "authenticated_retained_typed_hold_count": retained_typed_hold_count,
        "authenticated_retained_typed_hold_reserved_gbp_microunits": (
            retained_typed_hold_reserved
        ),
        "retained_typed_hold_disposition_counts": disposition_counts,
        "undispositioned_unresolved_attempt_count": undispositioned_count,
        "undispositioned_unresolved_reserved_gbp_microunits": (
            undispositioned_reserved
        ),
    }, (
        ([] if disposition_integrity_valid else ["GRAPHITI_SPEND_EVIDENCE_INTEGRITY_FAILURE"])
        + (["GRAPHITI_SPEND_UNRECONCILED"] if undispositioned_count else [])
    )


def _campaign_evidence(
    campaign: Mapping[str, object] | None,
    *,
    head_sha: str,
    tree_sha: str,
    observed_at: str,
    store_descriptors: Mapping[str, Mapping[str, object]],
    historical_partition: Mapping[str, object],
    authority_evidence: Mapping[str, object],
    graph_destination_reconciliation: StructuralReconciliationView | None,
    runtime_composed: bool,
    runtime_graph_destination_id: str | None,
) -> tuple[dict[str, object], list[str]]:
    if campaign is None:
        return {"configured": False, "campaign_authorised": False}, [
            "CAMPAIGN_INPUT_MISSING"
        ]

    blockers: list[str] = []

    expected_campaign_fields = {
        "schema_version",
        "focus_gate",
        "selection_policy",
        "provider",
        "graph",
        "caps",
        "ramp",
        "recovery",
        "immediate_stop_conditions",
        "success_objectives",
        "campaign_authorised",
    }
    if set(campaign) != expected_campaign_fields:
        blockers.append("CAMPAIGN_FIELDS_INVALID")

    def mapping(value: object, field: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            blockers.append(f"{field}_INVALID")
            return {}
        return value

    def token(value: object, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            blockers.append(f"{field}_INVALID")
            return ""
        return value

    def finite(value: object, field: str, *, positive: bool = False) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (1 if positive else 0)
        ):
            blockers.append(f"{field}_INVALID")
            return 0
        return value

    if campaign.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        blockers.append("CAMPAIGN_SCHEMA_INVALID")
    focus = mapping(campaign.get("focus_gate"), "FOCUS_GATE_EVIDENCE")
    if set(focus) != {"head_sha", "tree_sha", "conclusion", "manifest_digest"}:
        blockers.append("FOCUS_GATE_FIELDS_INVALID")
    focus_digest = token(focus.get("manifest_digest"), "FOCUS_GATE_MANIFEST_DIGEST")
    try:
        validate_sha256_digest(focus_digest, field="Focus Gate manifest digest")
    except (TypeError, ValueError):
        blockers.append("FOCUS_GATE_MANIFEST_DIGEST_INVALID")
    if (
        focus.get("head_sha") != head_sha
        or focus.get("tree_sha") != tree_sha
        or focus.get("conclusion") != "SUCCESS"
    ):
        blockers.append("EXACT_HEAD_FOCUS_GATE_NOT_PROVEN")

    actual_snapshot_digests = {
        name: descriptor["descriptor_digest"]
        for name, descriptor in store_descriptors.items()
    }

    candidates = historical_partition.get("current_preflight_candidates")
    if not isinstance(candidates, list):
        candidates = []
        blockers.append("CURRENT_PREFLIGHT_COHORT_UNAVAILABLE")
    derived_event_ids = [
        str(item.get("event_id"))
        for item in candidates
        if isinstance(item, Mapping)
    ]
    derived_events = [
        {
            "event_id": str(item.get("event_id")),
            "ledger_seq": item.get("ledger_seq"),
            "manifest_digest": str(item.get("manifest_digest")),
            "ingest_ids": list(item.get("ingest_ids", [])),
        }
        for item in candidates
        if isinstance(item, Mapping)
    ]
    if derived_event_ids:
        campaign = _narrow_campaign_input_to_selected_cohort(
            campaign, selected_event_count=len(derived_event_ids)
        )

    selection = mapping(campaign.get("selection_policy"), "SELECTION_POLICY")
    if set(selection) != {"policy_id", "policy_version"}:
        blockers.append("SELECTION_POLICY_FIELDS_INVALID")
    selection_value = {
        "policy_id": token(selection.get("policy_id"), "SELECTION_POLICY_ID"),
        "policy_version": token(
            selection.get("policy_version"), "SELECTION_POLICY_VERSION"
        ),
    }
    selection_digest = digest_canonical(selection_value)

    provider = mapping(campaign.get("provider"), "PROVIDER_IDENTITIES")
    if set(provider) != {
        "transport_id",
        "provider_id",
        "model_id",
        "embedding_provider_id",
        "embedding_model_id",
    }:
        blockers.append("PROVIDER_IDENTITY_FIELDS_INVALID")
    provider_value = {
        key: token(provider.get(key), f"{key.upper()}_IDENTITY")
        for key in (
            "transport_id",
            "provider_id",
            "model_id",
            "embedding_provider_id",
            "embedding_model_id",
        )
    }
    graph = mapping(campaign.get("graph"), "GRAPH_IDENTITIES")
    if set(graph) != {"destination_id"}:
        blockers.append("GRAPH_IDENTITY_FIELDS_INVALID")
    destination_id = token(graph.get("destination_id"), "GRAPH_DESTINATION_ID")
    if (
        runtime_graph_destination_id is not None
        and destination_id != runtime_graph_destination_id
    ):
        blockers.append("GRAPH_DESTINATION_RUNTIME_BINDING_INVALID")
    graph_readback = authority_evidence.get("active_projection_authority")
    if not isinstance(graph_readback, Mapping):
        blockers.append("ACTIVE_GRAPH_GENERATION_READBACK_INVALID")
        graph_value = {
            "destination_id": destination_id,
            "family_id": "",
            "ontology_id": "",
            "ontology_version": "",
            "ontology_contract_digest": "",
            "mapping_id": "",
            "mapping_version": "",
            "mapping_contract_digest": "",
            "projector_version": "",
            "current_generation_id": "",
            "generation_identity_version": (
                GRAPHITI_ADMISSION_GENERATION_IDENTITY_VERSION
            ),
            "generation_cohort_schema_version": (
                GRAPHITI_ADMISSION_COHORT_SCHEMA_VERSION
            ),
        }
    else:
        graph_value = {
            "destination_id": destination_id,
            "family_id": str(graph_readback.get("family_id", "")),
            "ontology_id": str(graph_readback.get("ontology_id", "")),
            "ontology_version": str(graph_readback.get("ontology_version", "")),
            "ontology_contract_digest": str(
                graph_readback.get("ontology_contract_digest", "")
            ),
            "mapping_id": str(graph_readback.get("mapping_id", "")),
            "mapping_version": str(graph_readback.get("mapping_version", "")),
            "mapping_contract_digest": str(
                graph_readback.get("mapping_contract_digest", "")
            ),
            "projector_version": str(graph_readback.get("projector_version", "")),
            "current_generation_id": str(graph_readback.get("generation_id", "")),
            "generation_identity_version": (
                GRAPHITI_ADMISSION_GENERATION_IDENTITY_VERSION
            ),
            "generation_cohort_schema_version": (
                GRAPHITI_ADMISSION_COHORT_SCHEMA_VERSION
            ),
        }

    authenticated_graph_readback: dict[str, object] | None = None
    if graph_destination_reconciliation is None:
        blockers.append("GRAPH_DESTINATION_READBACK_UNAVAILABLE")
    elif not isinstance(
        graph_destination_reconciliation, StructuralReconciliationView
    ):
        blockers.append("GRAPH_DESTINATION_READBACK_INVALID")
    else:
        supplied_readback = graphiti_graph_destination_readback(
            destination_id=destination_id,
            reconciliation=graph_destination_reconciliation,
        )
        readback_valid = (
            isinstance(graph_readback, Mapping)
            and bool(graph_value["current_generation_id"])
            and supplied_readback.get("family_id") == graph_value["family_id"]
            and supplied_readback.get("generation_id")
            == graph_value["current_generation_id"]
            and supplied_readback.get("checkpoint_ledger_seq")
            == graph_readback.get("validated_through_ledger_seq")
            and str(supplied_readback.get("serving_time")) <= observed_at
        )
        if readback_valid:
            authenticated_graph_readback = supplied_readback
        else:
            blockers.append("GRAPH_DESTINATION_READBACK_INVALID")
    if not runtime_composed:
        blockers.append("CANONICAL_OPERATOR_RUNTIME_UNCONFIGURED")
    caps = mapping(campaign.get("caps"), "CAMPAIGN_CAPS")
    per_event = mapping(caps.get("per_event"), "PER_EVENT_CAPS")
    total = mapping(caps.get("total"), "TOTAL_CAPS")
    rate = mapping(caps.get("rate"), "RATE_CAPS")
    count_names = (
        "proposals",
        "entity_admits",
        "relation_admits",
        "effects",
        "retries",
        "fallbacks",
    )
    caps_value = {
        "per_event": {
            name: finite(per_event.get(name), f"PER_EVENT_{name.upper()}_CAP")
            for name in count_names
        },
        "total": {
            "events": finite(total.get("events"), "TOTAL_EVENT_CAP", positive=True),
            **{
                name: finite(total.get(name), f"TOTAL_{name.upper()}_CAP")
                for name in count_names
            },
            "wall_time_seconds": finite(
                total.get("wall_time_seconds"), "TOTAL_WALL_TIME_CAP", positive=True
            ),
            "spend_gbp_microunits": finite(
                total.get("spend_gbp_microunits"), "TOTAL_SPEND_CAP"
            ),
        },
        "rate": {
            "events_per_minute": finite(
                rate.get("events_per_minute"), "EVENT_RATE_CAP", positive=True
            )
        },
    }
    if caps_value["total"]["events"] != len(derived_event_ids):
        blockers.append("TOTAL_EVENT_CAP_DIFFERS_FROM_SELECTED_COHORT")
    if (
        caps_value["per_event"]["fallbacks"] != 0
        or caps_value["total"]["fallbacks"] != 0
    ):
        blockers.append("FALLBACK_CAP_MUST_BE_ZERO")

    ramp = mapping(campaign.get("ramp"), "RAMP_ENTRY")
    phases = ramp.get("phases")
    ramp_phases: list[dict[str, object]] = []
    if not isinstance(phases, list) or not phases:
        blockers.append("RAMP_PHASES_INVALID")
    else:
        prior_limit = 0
        for index, raw_phase in enumerate(phases):
            phase = mapping(raw_phase, f"RAMP_PHASE_{index + 1}")
            phase_id = token(phase.get("phase_id"), f"RAMP_PHASE_{index + 1}_ID")
            event_limit = finite(
                phase.get("event_limit"),
                f"RAMP_PHASE_{index + 1}_EVENT_LIMIT",
                positive=True,
            )
            entry = phase.get("entry_conditions")
            advance = phase.get("advance_conditions")
            if (
                not isinstance(entry, list)
                or not entry
                or not all(isinstance(item, str) and item for item in entry)
                or entry != sorted(set(entry))
                or not CAMPAIGN_RAMP_ENTRY_CONDITIONS.issubset(entry)
            ):
                blockers.append(f"RAMP_PHASE_{index + 1}_ENTRY_INVALID")
                entry = []
            if (
                not isinstance(advance, list)
                or not advance
                or not all(isinstance(item, str) and item for item in advance)
                or advance != sorted(set(advance))
                or not CAMPAIGN_RAMP_ADVANCE_CONDITIONS.issubset(advance)
            ):
                blockers.append(f"RAMP_PHASE_{index + 1}_ADVANCE_INVALID")
                advance = []
            if event_limit <= prior_limit:
                blockers.append("RAMP_PHASE_LIMITS_NOT_STRICTLY_INCREASING")
            prior_limit = event_limit
            ramp_phases.append(
                {
                    "phase_id": phase_id,
                    "event_limit": event_limit,
                    "entry_conditions": entry,
                    "advance_conditions": advance,
                }
            )
        if prior_limit != caps_value["total"]["events"]:
            blockers.append("RAMP_FINAL_PHASE_DIFFERS_FROM_EVENT_CAP")
    ramp_value = {"phases": ramp_phases}

    recovery = mapping(campaign.get("recovery"), "RECOVERY_BINDINGS")
    recovery_value = {
        key: token(recovery.get(key), f"{key.upper()}_IDENTITY")
        for key in (
            "backup_identity",
            "rollback_procedure_id",
            "reconciliation_procedure_id",
        )
    }
    stops = campaign.get("immediate_stop_conditions")
    if (
        not isinstance(stops, list)
        or not all(isinstance(item, str) and item for item in stops)
        or len(stops) != len(set(item for item in stops if isinstance(item, str)))
        or not CAMPAIGN_REQUIRED_STOP_CONDITIONS.issubset(
            item for item in stops if isinstance(item, str)
        )
    ):
        blockers.append("IMMEDIATE_STOP_CONDITIONS_INCOMPLETE")
        stops = [] if not isinstance(stops, list) else stops

    objectives = mapping(campaign.get("success_objectives"), "SUCCESS_OBJECTIVES")
    lag_objective = mapping(objectives.get("lag"), "SUCCESS_OBJECTIVE_LAG")
    max_oldest_eligible_seconds = finite(
        lag_objective.get("max_oldest_eligible_seconds"),
        "MAX_OLDEST_ELIGIBLE_LAG",
        positive=True,
    )
    objectives_value = {
        **CAMPAIGN_SUCCESS_OBJECTIVE_BASE,
        "lag": {
            "max_oldest_eligible_seconds": max_oldest_eligible_seconds,
        },
    }
    if (
        set(objectives) != {*CAMPAIGN_SUCCESS_OBJECTIVE_BASE, "lag"}
        or any(
            objectives.get(key) != expected
            for key, expected in CAMPAIGN_SUCCESS_OBJECTIVE_BASE.items()
        )
        or set(lag_objective) != {"max_oldest_eligible_seconds"}
    ):
        blockers.append("SUCCESS_OBJECTIVES_INCOMPLETE")

    if campaign.get("campaign_authorised") is not False:
        blockers.append("CAMPAIGN_AUTHORITY_BOUNDARY_INVALID")
    value = {
        "configured": True,
        "campaign_input_digest": digest_canonical(campaign),
        "code_identity": {"head_sha": head_sha, "tree_sha": tree_sha},
        "focus_gate": dict(focus),
        "source_snapshot_digests": actual_snapshot_digests,
        "cohort": {
            "event_ids": derived_event_ids,
            "manifest_digest": historical_partition.get(
                "current_preflight_candidate_manifest_digest"
            ),
            "events": derived_events,
            "dispatch_authorised": False,
            "claim_performed": False,
        },
        "selection_policy": {**selection_value, "digest": selection_digest},
        "provider": provider_value,
        "graph": graph_value,
        "graph_destination_readback": authenticated_graph_readback,
        "caps": caps_value,
        "ramp": ramp_value,
        "recovery": recovery_value,
        "immediate_stop_conditions": stops,
        "success_objectives": objectives_value,
        "objectives_are_prospective": True,
        "campaign_authorised": False,
    }
    return value, blockers


def build_graphiti_steady_state_packet(
    *,
    proving_store: str | Path,
    unpublished_store: str | Path,
    head_sha: str,
    tree_sha: str,
    observed_at: datetime,
    authority_store: str | Path | None = None,
    campaign_input: Mapping[str, object] | None = None,
    graph_destination_reconciliation: StructuralReconciliationView | None = None,
    governed_runtime: GraphitiCampaignRuntime | None = None,
) -> dict[str, object]:
    """Build a stable report without invoking a provider or mutating either store."""

    if observed_at.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    observed_text = observed_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    with ExitStack() as stack:
        proving = stack.enter_context(read_only_snapshot(proving_store))
        unpublished = stack.enter_context(read_only_snapshot(unpublished_store))
        authority = (
            stack.enter_context(read_only_snapshot(authority_store))
            if authority_store is not None
            else None
        )
        proving_accounting, proving_blockers = _proving_accounting(
            proving.connection
        )
        try:
            resolved_units = load_graphiti_units_from_connection(
                proving.connection,
                evaluated_at=observed_at,
            )
            unit_resolution_failure = None
        except (sqlite3.Error, ValueError):
            resolved_units = ()
            unit_resolution_failure = "CURRENT_UNIT_RESOLUTION_UNAVAILABLE"
        try:
            event_gap_decisions = classify_graphiti_event_gaps(
                proving.connection,
                unpublished.connection,
                evaluated_at=observed_at,
                resolved_units=resolved_units,
                unit_resolution_failure=unit_resolution_failure,
            )
        except (GraphitiEventReconciliationError, sqlite3.Error, ValueError):
            event_gap_decisions = ()
            event_gap_classification_valid = False
        else:
            event_gap_classification_valid = True
        try:
            # This validates every authenticated event-repair receipt and its
            # retained queue/hold effects before any exclusion is trusted.
            excluded_event_ids = graphiti_excluded_event_ids(
                unpublished.connection
            )
        except (GraphitiEventReconciliationError, sqlite3.Error, ValueError):
            excluded_event_ids = frozenset()
            event_repair_evidence_valid = False
        else:
            event_repair_evidence_valid = True
        accounting, accounting_blockers = _event_accounting(
            unpublished.connection,
            gap_decisions=event_gap_decisions,
        )
        if not event_gap_classification_valid:
            accounting_blockers.append("EVENT_GAP_CLASSIFICATION_INVALID")
        if not event_repair_evidence_valid:
            accounting_blockers.append("EVENT_REPAIR_EVIDENCE_INTEGRITY_FAILURE")
        events, receipts, receipt_blockers = _events_and_receipts(
            unpublished.connection
        )
        historical_partition, partition_blockers = _historical_partition(
            proving.connection,
            unpublished.connection,
            authority=None if authority is None else authority.connection,
            observed_at=observed_at,
            event_evidence=events,
            receipt_evidence=receipts,
            event_gap_decisions=event_gap_decisions,
            resolved_units=resolved_units,
            unit_resolution_available=unit_resolution_failure is None,
            excluded_event_ids=excluded_event_ids,
        )
        admission, admission_blockers = _admission(
            unpublished.connection, observed_at=observed_at
        )
        spend, spend_blockers = _spend(unpublished.connection)
        store_descriptors = {
            "proving": _store_descriptor(proving),
            "unpublished": _store_descriptor(unpublished),
        }
        authority_blockers: list[str] = []
        authority_evidence: dict[str, object] = {"configured": False}
        if authority is None:
            authority_blockers.append("AUTHORITY_STORE_UNCONFIGURED")
        else:
            store_descriptors["authority"] = _store_descriptor(authority)
            authority_evidence, authority_blockers = _authority_snapshot_evidence(
                authority.connection
            )
        authority_descriptor = store_descriptors.get("authority")
        runtime_is_typed = _is_minted_graphiti_campaign_runtime(governed_runtime)
        runtime_authority_path = (
            governed_runtime.authority_store_source_path
            if runtime_is_typed
            else None
        )
        runtime_authority_digest = (
            governed_runtime.authority_store_descriptor_digest
            if runtime_is_typed
            else None
        )
        runtime_graph_destination_id = (
            governed_runtime.graph_destination_id if runtime_is_typed else None
        )
        runtime_composed = (
            runtime_is_typed
            and authority_descriptor is not None
            and runtime_authority_path == authority_descriptor["source_path"]
            and runtime_authority_digest
            == authority_descriptor["descriptor_digest"]
        )
        campaign, campaign_blockers = _campaign_evidence(
            campaign_input,
            head_sha=head_sha,
            tree_sha=tree_sha,
            observed_at=observed_text,
            store_descriptors=store_descriptors,
            historical_partition=historical_partition,
            authority_evidence=authority_evidence,
            graph_destination_reconciliation=(
                graph_destination_reconciliation
            ),
            runtime_composed=runtime_composed,
            runtime_graph_destination_id=runtime_graph_destination_id,
        )
        campaign_graph = campaign.get("graph")
        exact_reconciliation = admission.get("exact_cohort_reconciliation")
        latest_exact_generation = (
            exact_reconciliation.get("latest_generation_id")
            if isinstance(exact_reconciliation, Mapping)
            else None
        )
        active_projection = authority_evidence.get("active_projection_authority")
        authenticated_readback = campaign.get("graph_destination_readback")
        if latest_exact_generation is not None and (
            not isinstance(active_projection, Mapping)
            or active_projection.get("generation_id") != latest_exact_generation
            or not isinstance(authenticated_readback, Mapping)
            or authenticated_readback.get("generation_id")
            != latest_exact_generation
        ):
            campaign_blockers.append("ADMISSION_ACTIVE_GENERATION_DRIFT")
        runtime_campaign_graph_bound = (
            runtime_composed
            and isinstance(campaign_graph, Mapping)
            and campaign_graph.get("destination_id")
            == runtime_graph_destination_id
        )
        blockers = (
            proving_blockers
            + accounting_blockers
            + receipt_blockers
            + partition_blockers
            + admission_blockers
            + spend_blockers
            + authority_blockers
            + campaign_blockers
        )
        blocker_values = sorted(set(blockers))
        ready = not blocker_values
        body: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "code_identity": {"head_sha": head_sha, "tree_sha": tree_sha},
            "observed_at": observed_text,
            "store_snapshots": store_descriptors,
            "proving_accountability": proving_accounting,
            "authority_snapshot_evidence": authority_evidence,
            "runtime_composition": {
                "state": (
                    "DORMANT_GOVERNED_RUNTIME_COMPOSED"
                    if runtime_composed
                    else "CANONICAL_OPERATOR_RUNTIME_UNCONFIGURED"
                ),
                "authority_store_configured": authority is not None,
                "authority_store_source_path": runtime_authority_path,
                "authority_store_descriptor_digest": runtime_authority_digest,
                "graph_destination_id": runtime_graph_destination_id,
                "durable_proposal_envelope_binding": runtime_composed,
                "admission_policy_configured": runtime_composed,
                "full_generation_projector_configured": runtime_composed,
                "dormant_worker_path_wired": runtime_composed,
                "campaign_packet_enforced": runtime_campaign_graph_bound,
                "actual_graph_readback_observed": (
                    campaign.get("graph_destination_readback") is not None
                ),
                "campaign_authorised": False,
            },
            "bounded_campaign": campaign,
            "landed_event_accounting": accounting,
            "events": events,
            "terminal_receipts": receipts,
            "historical_partition": historical_partition,
            "admission": admission,
            "usage_and_spend": spend,
            "non_effects": {
                "provider_calls": 0,
                "store_mutations": 0,
                "service_loads": 0,
                "publication_effects": 0,
                "production_admission_effects": 0,
            },
            "blockers": blocker_values,
            "verdict": "READY_FOR_OWNER_DECISION" if ready else "NO_GO",
            "readiness": (
                "F4_CAMPAIGN_READY_FOR_OWNER_DECISION"
                if ready
                else "ENGINEERING_PREPARATION_ONLY"
            ),
        }
        return {**body, "packet_digest": digest_canonical(body)}


def validate_graphiti_campaign_packet(
    packet: Mapping[str, object],
) -> dict[str, object]:
    """Validate one sealed, ready, still unauthorised campaign packet."""

    value = dict(packet)
    supplied_digest = value.get("packet_digest")
    unsigned = {key: item for key, item in value.items() if key != "packet_digest"}
    if supplied_digest != digest_canonical(unsigned):
        raise ValueError("campaign packet canonical digest differs")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("campaign packet schema differs")
    if (
        value.get("verdict") != "READY_FOR_OWNER_DECISION"
        or value.get("readiness") != "F4_CAMPAIGN_READY_FOR_OWNER_DECISION"
        or value.get("blockers") != []
    ):
        raise ValueError("campaign packet is not ready")

    runtime = value.get("runtime_composition")
    if not isinstance(runtime, Mapping) or (
        set(runtime)
        != {
            "state",
            "authority_store_configured",
            "authority_store_source_path",
            "authority_store_descriptor_digest",
            "graph_destination_id",
            "durable_proposal_envelope_binding",
            "admission_policy_configured",
            "full_generation_projector_configured",
            "dormant_worker_path_wired",
            "campaign_packet_enforced",
            "actual_graph_readback_observed",
            "campaign_authorised",
        }
        or runtime.get("state") != "DORMANT_GOVERNED_RUNTIME_COMPOSED"
        or runtime.get("authority_store_configured") is not True
        or runtime.get("durable_proposal_envelope_binding") is not True
        or runtime.get("admission_policy_configured") is not True
        or runtime.get("full_generation_projector_configured") is not True
        or runtime.get("dormant_worker_path_wired") is not True
        or runtime.get("campaign_packet_enforced") is not True
        or runtime.get("actual_graph_readback_observed") is not True
        or runtime.get("campaign_authorised") is not False
    ):
        raise ValueError("campaign packet runtime composition differs")
    try:
        validate_sha256_digest(
            runtime.get("graph_destination_id"),
            field="campaign runtime graph destination id",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("campaign packet runtime composition differs") from exc

    campaign = value.get("bounded_campaign")
    if not isinstance(campaign, Mapping) or (
        campaign.get("configured") is not True
        or campaign.get("campaign_authorised") is not False
    ):
        raise ValueError("campaign packet authority boundary differs")
    if set(campaign) != {
        "configured",
        "campaign_input_digest",
        "code_identity",
        "focus_gate",
        "source_snapshot_digests",
        "cohort",
        "selection_policy",
        "provider",
        "graph",
        "graph_destination_readback",
        "caps",
        "ramp",
        "recovery",
        "immediate_stop_conditions",
        "success_objectives",
        "objectives_are_prospective",
        "campaign_authorised",
    } or campaign.get("objectives_are_prospective") is not True:
        raise ValueError("campaign packet fields differ")
    if campaign.get("code_identity") != value.get("code_identity"):
        raise ValueError("campaign packet code identity differs")
    focus = campaign.get("focus_gate")
    if not isinstance(focus, Mapping) or set(focus) != {
        "head_sha",
        "tree_sha",
        "conclusion",
        "manifest_digest",
    }:
        raise ValueError("campaign packet Focus Gate differs")
    try:
        validate_sha256_digest(
            focus.get("manifest_digest"),
            field="campaign Focus Gate manifest digest",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("campaign packet Focus Gate differs") from exc
    code_identity = value.get("code_identity")
    if (
        not isinstance(code_identity, Mapping)
        or focus.get("head_sha") != code_identity.get("head_sha")
        or focus.get("tree_sha") != code_identity.get("tree_sha")
        or focus.get("conclusion") != "SUCCESS"
    ):
        raise ValueError("campaign packet Focus Gate differs")

    stores = value.get("store_snapshots")
    if not isinstance(stores, Mapping):
        raise ValueError("campaign packet store snapshots differ")
    authority_store = stores.get("authority")
    if (
        not isinstance(authority_store, Mapping)
        or runtime.get("authority_store_source_path")
        != authority_store.get("source_path")
        or runtime.get("authority_store_descriptor_digest")
        != authority_store.get("descriptor_digest")
    ):
        raise ValueError("campaign packet runtime authority binding differs")
    snapshot_digests = {
        str(name): descriptor.get("descriptor_digest")
        for name, descriptor in stores.items()
        if isinstance(descriptor, Mapping)
    }
    if (
        len(snapshot_digests) != len(stores)
        or campaign.get("source_snapshot_digests") != snapshot_digests
    ):
        raise ValueError("campaign packet store snapshot binding differs")

    cohort = campaign.get("cohort")
    if not isinstance(cohort, Mapping):
        raise ValueError("campaign packet cohort differs")
    events = cohort.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("campaign packet cohort differs")
    exact_event_ids: list[str] = []
    for event in events:
        if not isinstance(event, Mapping) or set(event) != {
            "event_id",
            "ledger_seq",
            "manifest_digest",
            "ingest_ids",
        }:
            raise ValueError("campaign packet cohort differs")
        event_id = event.get("event_id")
        manifest_digest = event.get("manifest_digest")
        ledger_seq = event.get("ledger_seq")
        ingest_ids = event.get("ingest_ids")
        try:
            validate_sha256_digest(event_id, field="campaign event id")
            validate_sha256_digest(
                manifest_digest, field="campaign event manifest digest"
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("campaign packet cohort differs") from exc
        if (
            isinstance(ledger_seq, bool)
            or not isinstance(ledger_seq, int)
            or ledger_seq < 1
            or not isinstance(ingest_ids, list)
            or not ingest_ids
            or not all(isinstance(item, str) and item for item in ingest_ids)
            or len(set(ingest_ids)) != len(ingest_ids)
        ):
            raise ValueError("campaign packet cohort differs")
        exact_event_ids.append(str(event_id))
    if (
        cohort.get("event_ids") != exact_event_ids
        or cohort.get("manifest_digest") != digest_canonical(events)
        or cohort.get("dispatch_authorised") is not False
        or cohort.get("claim_performed") is not False
    ):
        raise ValueError("campaign packet cohort differs")

    selection = campaign.get("selection_policy")
    if not isinstance(selection, Mapping) or set(selection) != {
        "policy_id",
        "policy_version",
        "digest",
    }:
        raise ValueError("campaign packet selection policy differs")
    selection_value = {
        "policy_id": selection.get("policy_id"),
        "policy_version": selection.get("policy_version"),
    }
    if (
        any(not isinstance(item, str) or not item for item in selection_value.values())
        or selection.get("digest") != digest_canonical(selection_value)
    ):
        raise ValueError("campaign packet selection policy differs")

    provider = campaign.get("provider")
    if (
        not isinstance(provider, Mapping)
        or set(provider)
        != {
            "transport_id",
            "provider_id",
            "model_id",
            "embedding_provider_id",
            "embedding_model_id",
        }
        or any(not isinstance(item, str) or not item for item in provider.values())
    ):
        raise ValueError("campaign packet provider identities differ")

    caps = campaign.get("caps")
    if not isinstance(caps, Mapping) or set(caps) != {
        "per_event",
        "total",
        "rate",
    }:
        raise ValueError("campaign packet caps differ")
    per_event_caps = caps.get("per_event")
    total_caps = caps.get("total")
    rate_caps = caps.get("rate")
    count_names = {
        "proposals",
        "entity_admits",
        "relation_admits",
        "effects",
        "retries",
        "fallbacks",
    }
    if (
        not isinstance(per_event_caps, Mapping)
        or set(per_event_caps) != count_names
        or not isinstance(total_caps, Mapping)
        or set(total_caps)
        != {
            "events",
            *count_names,
            "wall_time_seconds",
            "spend_gbp_microunits",
        }
        or not isinstance(rate_caps, Mapping)
        or set(rate_caps) != {"events_per_minute"}
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in (*per_event_caps.values(), *total_caps.values())
        )
        or isinstance(rate_caps.get("events_per_minute"), bool)
        or not isinstance(rate_caps.get("events_per_minute"), int)
        or int(rate_caps["events_per_minute"]) <= 0
        or total_caps.get("wall_time_seconds", 0) <= 0
        or total_caps.get("events") != len(events)
        or per_event_caps.get("fallbacks") != 0
        or total_caps.get("fallbacks") != 0
    ):
        raise ValueError("campaign packet caps differ")

    ramp = campaign.get("ramp")
    phases = ramp.get("phases") if isinstance(ramp, Mapping) else None
    if (
        not isinstance(ramp, Mapping)
        or set(ramp) != {"phases"}
        or not isinstance(phases, list)
        or not phases
        or not isinstance(phases[-1], Mapping)
        or phases[-1].get("event_limit") != len(events)
    ):
        raise ValueError("campaign packet cohort cap differs")
    prior_limit = 0
    for phase in phases:
        if not isinstance(phase, Mapping) or set(phase) != {
            "phase_id",
            "event_limit",
            "entry_conditions",
            "advance_conditions",
        }:
            raise ValueError("campaign packet ramp differs")
        event_limit = phase.get("event_limit")
        entry = phase.get("entry_conditions")
        advance = phase.get("advance_conditions")
        if (
            not isinstance(phase.get("phase_id"), str)
            or not phase.get("phase_id")
            or isinstance(event_limit, bool)
            or not isinstance(event_limit, int)
            or event_limit <= prior_limit
            or not isinstance(entry, list)
            or entry != sorted(set(entry))
            or not CAMPAIGN_RAMP_ENTRY_CONDITIONS.issubset(entry)
            or not isinstance(advance, list)
            or advance != sorted(set(advance))
            or not CAMPAIGN_RAMP_ADVANCE_CONDITIONS.issubset(advance)
        ):
            raise ValueError("campaign packet ramp differs")
        prior_limit = event_limit

    recovery = campaign.get("recovery")
    if (
        not isinstance(recovery, Mapping)
        or set(recovery)
        != {
            "backup_identity",
            "rollback_procedure_id",
            "reconciliation_procedure_id",
        }
        or any(not isinstance(item, str) or not item for item in recovery.values())
    ):
        raise ValueError("campaign packet recovery bindings differ")
    stops = campaign.get("immediate_stop_conditions")
    if (
        not isinstance(stops, list)
        or any(not isinstance(item, str) or not item for item in stops)
        or len(stops) != len(set(stops))
        or not CAMPAIGN_REQUIRED_STOP_CONDITIONS.issubset(stops)
    ):
        raise ValueError("campaign packet stop conditions differ")
    objectives = campaign.get("success_objectives")
    lag_objective = objectives.get("lag") if isinstance(objectives, Mapping) else None
    if (
        not isinstance(objectives, Mapping)
        or set(objectives) != {*CAMPAIGN_SUCCESS_OBJECTIVE_BASE, "lag"}
        or any(
            objectives.get(key) != expected
            for key, expected in CAMPAIGN_SUCCESS_OBJECTIVE_BASE.items()
        )
        or not isinstance(lag_objective, Mapping)
        or set(lag_objective) != {"max_oldest_eligible_seconds"}
        or isinstance(lag_objective.get("max_oldest_eligible_seconds"), bool)
        or not isinstance(lag_objective.get("max_oldest_eligible_seconds"), int)
        or int(lag_objective["max_oldest_eligible_seconds"]) <= 0
    ):
        raise ValueError("campaign packet objectives differ")

    graph = campaign.get("graph")
    readback = campaign.get("graph_destination_readback")
    authority_evidence = value.get("authority_snapshot_evidence")
    active_authority = (
        authority_evidence.get("active_projection_authority")
        if isinstance(authority_evidence, Mapping)
        else None
    )
    expected_graph_fields = {
        "destination_id",
        "family_id",
        "ontology_id",
        "ontology_version",
        "ontology_contract_digest",
        "mapping_id",
        "mapping_version",
        "mapping_contract_digest",
        "projector_version",
        "current_generation_id",
        "generation_identity_version",
        "generation_cohort_schema_version",
    }
    if (
        not isinstance(graph, Mapping)
        or set(graph) != expected_graph_fields
        or any(not isinstance(item, str) or not item for item in graph.values())
        or graph.get("generation_identity_version")
        != GRAPHITI_ADMISSION_GENERATION_IDENTITY_VERSION
        or graph.get("generation_cohort_schema_version")
        != GRAPHITI_ADMISSION_COHORT_SCHEMA_VERSION
        or not isinstance(readback, Mapping)
        or runtime.get("graph_destination_id") != graph.get("destination_id")
    ):
        raise ValueError("campaign packet graph readback differs")
    if (
        set(readback)
        != {
            "destination_id",
            "family_id",
            "generation_id",
            "checkpoint_ledger_seq",
            "projection_state_digest",
            "serving_time",
        }
        or readback.get("destination_id") != graph.get("destination_id")
        or readback.get("family_id") != graph.get("family_id")
        or readback.get("generation_id") != graph.get("current_generation_id")
        or not isinstance(active_authority, Mapping)
        or readback.get("generation_id") != active_authority.get("generation_id")
        or readback.get("checkpoint_ledger_seq")
        != active_authority.get("validated_through_ledger_seq")
        or isinstance(readback.get("checkpoint_ledger_seq"), bool)
        or not isinstance(readback.get("checkpoint_ledger_seq"), int)
        or int(readback["checkpoint_ledger_seq"]) < 0
        or not isinstance(readback.get("serving_time"), str)
        or str(readback["serving_time"]) > str(value.get("observed_at"))
    ):
        raise ValueError("campaign packet graph readback differs")
    try:
        validate_sha256_digest(
            readback.get("projection_state_digest"),
            field="campaign graph projection state digest",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("campaign packet graph readback differs") from exc
    admission = value.get("admission")
    exact_reconciliation = (
        admission.get("exact_cohort_reconciliation")
        if isinstance(admission, Mapping)
        else None
    )
    cohorts = (
        exact_reconciliation.get("cohorts")
        if isinstance(exact_reconciliation, Mapping)
        else None
    )
    latest_exact_generation = (
        exact_reconciliation.get("latest_generation_id")
        if isinstance(exact_reconciliation, Mapping)
        else None
    )
    if (
        not isinstance(exact_reconciliation, Mapping)
        or exact_reconciliation.get("total") is not True
        or exact_reconciliation.get("disjoint") is not True
        or not isinstance(cohorts, list)
    ) or (
        cohorts
        and (
            not isinstance(cohorts[-1], Mapping)
            or not isinstance(latest_exact_generation, str)
            or not latest_exact_generation
            or cohorts[-1].get("generation_id") != latest_exact_generation
            or latest_exact_generation != graph.get("current_generation_id")
            or latest_exact_generation != readback.get("generation_id")
            or latest_exact_generation != active_authority.get("generation_id")
        )
    ) or (not cohorts and latest_exact_generation is not None):
        raise ValueError(
            "campaign packet active generation differs from exact admission"
        )
    if value.get("non_effects") != {
        "provider_calls": 0,
        "store_mutations": 0,
        "service_loads": 0,
        "publication_effects": 0,
        "production_admission_effects": 0,
    }:
        raise ValueError("campaign packet non-effects differ")
    return dict(campaign)


def write_content_addressed_packet(
    packet: Mapping[str, object], directory: str | Path
) -> Path:
    digest = str(packet.get("packet_digest") or "")
    if not digest.startswith("sha256:"):
        raise ValueError("packet has no canonical digest")
    body = {key: value for key, value in packet.items() if key != "packet_digest"}
    if digest_canonical(body) != digest:
        raise ValueError("packet canonical digest differs")
    output = Path(directory) / (
        f"graphiti-steady-state-{digest.removeprefix('sha256:')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
