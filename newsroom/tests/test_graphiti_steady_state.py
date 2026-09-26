from __future__ import annotations

import inspect
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from newsroom.authority.auth import (
    AuthenticationProof,
    StaticAuthenticator,
    StaticPrincipal,
)
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.authority.migrations import MIGRATIONS
from newsroom.authority.types import UtcTimestamp
from newsroom.control_plane.command_service import ControlPlaneCommandService
from newsroom.control_plane.graphiti_admission import (
    GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION,
    GRAPHITI_ADMISSION_GENERATION_IDENTITY_VERSION,
    GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION,
    GraphitiAdmissionConsumerError,
    GraphitiGovernedDecision,
    GraphitiProjectionReconciliationReceipt,
    GraphitiProjectionGenerationResult,
)
from newsroom.control_plane.graphiti_event_reconciliation import GraphitiEventReconciliationError
from newsroom.control_plane.graphiti_events import (
    GRAPHITI_EVENT_PROJECTION_GENERATION,
    GRAPHITI_EVENT_PROJECTOR_VERSION,
)
from newsroom.control_plane.graphiti_operational_readiness import (
    build_operational_campaign_input,
)
from newsroom.control_plane.graphiti_spend_reconciliation import (
    plan_graphiti_spend_reconciliation,
)
from newsroom.control_plane.graphiti_steady_state import (
    GraphitiCampaignRuntime,
    PRE_FRONTIER_BACKLOG_HOLD_REASON,
    _authority_snapshot_evidence,
    _exact_admission_reconciliation,
    _mint_graphiti_campaign_runtime,
    _spend,
    _store_descriptor,
    build_graphiti_steady_state_packet,
    graphiti_graph_destination_identity,
    graphiti_operational_partition_snapshot,
    graphiti_store_snapshot_digests,
    validate_graphiti_campaign_packet,
    write_content_addressed_packet,
)
from newsroom.control_plane.read_only_snapshot import read_only_snapshot
from newsroom.control_plane.store import (
    connect,
    effective_revision_landed_payload,
    insert_graphiti_attempt_receipt,
    reconcile_graphiti_spend,
    reserve_graphiti_spend,
)
from newsroom.effective_revision import EffectiveRevisionIdentity
from newsroom.extraction.types import ExtractionProposalKind
from newsroom.graphiti_adapter.admission import GraphitiProposalAdmissionAction
from newsroom.graphiti_adapter.identity import attempt_ids, typed_id
from newsroom.increment9.proving import PROVING_GATES, SOURCE_IDS, SOURCE_URLS
from newsroom.increment4.contracts import (
    increment4_admitted_family_v1,
    increment4_admitted_mapping_v1,
    increment4_admitted_ontology_v1,
)
from newsroom.projection import ProjectionGenerationId
from newsroom.projection.neo4j import StructuralReconciliationView
from scripts import graphiti_steady_state_report

from .test_graphiti_admission_consumer import (
    _Authority as _AdmissionAuthority,
    _Projector as _AdmissionProjector,
    _Rights as _AdmissionRights,
    _consumer as _admission_consumer,
    _draft as _admission_draft,
    _seed_receipt as _seed_admission_receipt,
)

GRAPH_DESTINATION_ID = "sha256:" + "9" * 64
NOW = datetime(2026, 9, 1, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    ("field", "drifted_value"),
    (
        ("destination_id", "sha256:" + "8" * 64),
        ("family_id", "graph.increment4.other"),
        ("generation_id", "00000000-0000-4000-8000-000000000896"),
        ("checkpoint_ledger_seq", 8),
        ("projection_state_digest", "sha256:" + "7" * 64),
    ),
)
def test_graph_destination_identity_excludes_only_serving_time(
    field: str,
    drifted_value: object,
) -> None:
    sealed = {
        "destination_id": GRAPH_DESTINATION_ID,
        "family_id": "graph.increment4.admitted",
        "generation_id": "00000000-0000-4000-8000-000000000895",
        "checkpoint_ledger_seq": 7,
        "projection_state_digest": "sha256:" + "6" * 64,
        "serving_time": "2026-09-01T12:00:00.000000Z",
    }
    later_observation = {
        **sealed,
        "serving_time": "2026-09-01T12:05:00.000000Z",
    }
    assert graphiti_graph_destination_identity(
        later_observation
    ) == graphiti_graph_destination_identity(sealed)

    later_observation[field] = drifted_value
    assert graphiti_graph_destination_identity(
        later_observation
    ) != graphiti_graph_destination_identity(sealed)


def _stores(tmp_path: Path) -> tuple[Path, Path, sqlite3.Connection]:
    proving = tmp_path / "proving.sqlite3"
    sqlite3.connect(proving).execute(
        "CREATE TABLE proof(value TEXT)"
    ).connection.close()
    unpublished = tmp_path / "unpublished.sqlite3"
    connection = connect(str(unpublished))
    return proving, unpublished, connection


def test_spend_accepts_authenticated_retained_hold_but_not_unclassified_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unpublished.sqlite3"
    connection = connect(str(path))
    spend_id = "retained-hold:1"
    assert reserve_graphiti_spend(
        connection,
        spend_id=spend_id,
        ingest_id="retained-hold",
        attempt_number=1,
        proving_run_id="run-1",
        generation_id="generation",
        reserved_gbp_microunits=500_000,
        ceiling_gbp_microunits=1_000_000,
    )
    accounting = reconcile_graphiti_spend(
        connection, spend_id=spend_id, embedding_usage=None
    )
    insert_graphiti_attempt_receipt(
        connection,
        ingest_id="retained-hold",
        attempt_number=1,
        outcome="FAILED",
        receipt={
            "ingest_id": "retained-hold",
            "attempt_number": 1,
            "outcome": "FAILED",
            "provider_attempt_number": 1,
            "chat_invocations": [],
            "embedding_usage": None,
            "accounting": accounting,
        },
    )
    connection.commit()
    connection.close()
    evaluated_at = datetime(2026, 9, 1, 12, tzinfo=UTC)
    plan = plan_graphiti_spend_reconciliation(str(path), evaluated_at=evaluated_at)
    service = ControlPlaneCommandService(
        authenticator=StaticAuthenticator(
            credentials={
                "operator-token": StaticPrincipal(principal_id="newsroom.hermes")
            },
            authority_domain="newsroom.control-plane",
        )
    )
    service.reconcile_graphiti_spend(
        unpublished_store=str(path),
        dry_run_plan=plan.as_dict(),
        evaluated_at=evaluated_at,
        idempotency_key="retained-hold",
        expected_plan_digest=plan.plan_digest,
        proof=AuthenticationProof(method="STATIC_TOKEN", credential="operator-token"),
    )

    connection = sqlite3.connect(path)
    spend, blockers = _spend(connection)
    connection.close()

    assert blockers == []
    assert spend["status_counts"]["UNRECONCILED"] == 1
    assert spend["reserved_gbp_microunits_by_status"]["UNRECONCILED"] == 500_000
    assert spend["authenticated_retained_typed_hold_count"] == 1
    assert spend["retained_typed_hold_disposition_counts"] == {
        "UNRECONCILED_REPORTED_MISSING": 1
    }
    assert spend["undispositioned_unresolved_attempt_count"] == 0

    connection = connect(str(path))
    assert reserve_graphiti_spend(
        connection,
        spend_id="unclassified:1",
        ingest_id="unclassified",
        attempt_number=1,
        proving_run_id="run-1",
        generation_id="generation",
        reserved_gbp_microunits=250_000,
        ceiling_gbp_microunits=1_000_000,
    )
    connection.commit()
    spend, blockers = _spend(connection)
    connection.close()

    assert blockers == ["GRAPHITI_SPEND_UNRECONCILED"]
    assert spend["authenticated_retained_typed_hold_count"] == 1
    assert spend["undispositioned_unresolved_attempt_count"] == 1
    assert spend["undispositioned_unresolved_reserved_gbp_microunits"] == 250_000

    connection = connect(str(path))
    connection.execute(
        "UPDATE unpublished_graphiti_spend_dispositions SET evidence_json='{}' "
        "WHERE spend_id=?",
        (spend_id,),
    )
    connection.commit()
    connection.close()
    proving = tmp_path / "proving.sqlite3"
    sqlite3.connect(proving).execute(
        "CREATE TABLE proof(value TEXT)"
    ).connection.close()

    packet = _packet(proving, path)

    assert packet["verdict"] == "NO_GO"
    assert "GRAPHITI_SPEND_EVIDENCE_INTEGRITY_FAILURE" in packet["blockers"]
    assert packet["usage_and_spend"]["retained_disposition_integrity_valid"] is False
    assert packet["usage_and_spend"]["authenticated_retained_typed_hold_count"] == 0


def test_spend_incomplete_retained_receipt_is_machine_readable_no_go(
    tmp_path: Path,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    connection.close()
    plan = plan_graphiti_spend_reconciliation(
        str(unpublished), evaluated_at=NOW
    )
    service = ControlPlaneCommandService(
        authenticator=StaticAuthenticator(
            credentials={
                "operator-token": StaticPrincipal(principal_id="newsroom.hermes")
            },
            authority_domain="newsroom.control-plane",
        )
    )
    service.reconcile_graphiti_spend(
        unpublished_store=str(unpublished),
        dry_run_plan=plan.as_dict(),
        evaluated_at=NOW,
        idempotency_key="incomplete-retained-receipt",
        expected_plan_digest=plan.plan_digest,
        proof=AuthenticationProof(method="STATIC_TOKEN", credential="operator-token"),
    )
    connection = sqlite3.connect(unpublished)
    connection.execute(
        "UPDATE unpublished_graphiti_spend_reconciliation_receipts "
        "SET receipt_json=?",
        (canonical_json_bytes({"disposition_counts": {}}).decode("utf-8"),),
    )
    connection.commit()
    connection.close()

    packet = _packet(proving, unpublished)

    assert packet["verdict"] == "NO_GO"
    assert "GRAPHITI_SPEND_EVIDENCE_INTEGRITY_FAILURE" in packet["blockers"]
    assert packet["usage_and_spend"]["retained_disposition_integrity_valid"] is False
    assert packet["usage_and_spend"]["authenticated_retained_typed_hold_count"] == 0


def _terminal_zero_proposal(connection: sqlite3.Connection) -> None:
    at = "2026-09-01T12:00:00.000000Z"
    ledger_digest = "sha256:" + "1" * 64
    payload_digest = "sha256:" + "2" * 64
    manifest = {
        "event_type": "EFFECTIVE_SOURCE_REVISION_LANDED",
        "ledger_seq": 1,
        "ledger_digest": ledger_digest,
        # Legacy projected events retained resolved units before this field
        # was populated; terminal receipt accounting must use unit_refs.
        "landed_ingest_ids": [],
        "landed_payload_digest": payload_digest,
        "unit_refs": [{"ingest_id": "ingest-1"}],
    }
    receipt = {
        "ingest_id": "ingest-1",
        "outcome": "COMPLETE",
        "proposal_count": 0,
        "entity_count": 0,
        "relation_count": 0,
        "raw_output_digest": "sha256:" + "3" * 64,
        "proposals": [],
    }
    receipt_digest = digest_canonical(receipt)
    receipt = {**receipt, "receipt_digest": receipt_digest}
    connection.execute(
        "INSERT INTO ledger(seq,at,kind,payload_digest,payload_json,prev_digest,"
        "digest) VALUES(1,?,?,?,?,?,?)",
        (
            at,
            "EFFECTIVE_REVISION_LANDED",
            payload_digest,
            "{}",
            "GENESIS",
            ledger_digest,
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_effective_revision_landed("
        "source_id,item_key,revision_digest,published_at,updated_at,"
        "first_observed_at,ingest_ids_json,legacy_v10,payload_digest,"
        "ledger_digest,at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            "source",
            "item",
            "revision",
            "",
            "",
            at,
            '["ingest-1"]',
            0,
            payload_digest,
            ledger_digest,
            at,
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_revision_events("
        "event_id,ledger_seq,ledger_digest,source_id,item_key,revision_digest,"
        "published_at,updated_at,landed_at,manifest_json,manifest_digest,"
        "unit_count,projector_version,projection_generation,state,attempt_count,"
        "available_at,provider_dispatched,terminal_at,proposal_count) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            ledger_digest,
            1,
            ledger_digest,
            "source",
            "item",
            "revision",
            "",
            "",
            at,
            json.dumps(manifest, sort_keys=True),
            digest_canonical(manifest),
            1,
            GRAPHITI_EVENT_PROJECTOR_VERSION,
            GRAPHITI_EVENT_PROJECTION_GENERATION,
            "TERMINAL",
            1,
            at,
            1,
            at,
            0,
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_ingest("
        "ingest_id,source_id,item_key,outcome,proposal_count,entity_count,"
        "relation_count,failure_code,temporal_basis,reference_time,generation_id,"
        "receipt_digest,at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "ingest-1",
            "source",
            "item",
            "COMPLETE",
            0,
            0,
            0,
            "",
            "PUBLISHED_AT",
            at,
            "generation",
            receipt_digest,
            at,
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_attempt_receipts("
        "ingest_id,attempt_number,outcome,receipt_digest,receipt_json,at) "
        "VALUES(?,?,?,?,?,?)",
        (
            "ingest-1",
            1,
            "COMPLETE",
            receipt_digest,
            json.dumps(receipt, sort_keys=True),
            at,
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_receipts(ingest_id,receipt_json) VALUES(?,?)",
        ("ingest-1", json.dumps(receipt, sort_keys=True)),
    )
    connection.commit()


def _turn_terminal_receipt_into_historical_raw_only(
    connection: sqlite3.Connection,
) -> None:
    row = connection.execute(
        "SELECT receipt_json FROM unpublished_graphiti_receipts WHERE ingest_id='ingest-1'"
    ).fetchone()
    assert row is not None
    receipt = json.loads(str(row[0]))
    receipt.pop("receipt_digest")
    receipt["proposal_count"] = 1
    receipt["proposals"] = [{"local_id": "historical-raw-only"}]
    receipt_digest = digest_canonical(receipt)
    retained = {**receipt, "receipt_digest": receipt_digest}
    retained_json = json.dumps(retained, sort_keys=True)
    connection.execute(
        "UPDATE unpublished_graphiti_ingest SET proposal_count=1,receipt_digest=? "
        "WHERE ingest_id='ingest-1'",
        (receipt_digest,),
    )
    connection.execute(
        "UPDATE unpublished_graphiti_receipts SET receipt_json=? "
        "WHERE ingest_id='ingest-1'",
        (retained_json,),
    )
    connection.execute(
        "UPDATE unpublished_graphiti_attempt_receipts "
        "SET receipt_digest=?,receipt_json=? WHERE ingest_id='ingest-1'",
        (receipt_digest, retained_json),
    )
    connection.execute(
        "UPDATE unpublished_graphiti_revision_events SET proposal_count=1"
    )
    connection.commit()


def _bind_terminal_raw_proposal_to_authority(
    unpublished: Path,
    authority: Path,
) -> None:
    unpublished_connection = sqlite3.connect(unpublished)
    row = unpublished_connection.execute(
        "SELECT receipt_json FROM unpublished_graphiti_receipts "
        "WHERE ingest_id='ingest-1'"
    ).fetchone()
    assert row is not None
    receipt = json.loads(str(row[0]))
    receipt.pop("receipt_digest")
    receipt["attempt_number"] = 1
    raw = dict(receipt)
    raw.pop("raw_output_digest", None)
    raw["raw_output_digest"] = digest_bytes(canonical_json_bytes(raw))
    receipt["raw_output_digest"] = raw["raw_output_digest"]
    raw_bytes = canonical_json_bytes(raw)
    receipt_digest = digest_canonical(receipt)
    retained_json = json.dumps(
        {**receipt, "receipt_digest": receipt_digest}, sort_keys=True
    )
    unpublished_connection.execute(
        "UPDATE unpublished_graphiti_ingest SET receipt_digest=? "
        "WHERE ingest_id='ingest-1'",
        (receipt_digest,),
    )
    unpublished_connection.execute(
        "UPDATE unpublished_graphiti_receipts SET receipt_json=? "
        "WHERE ingest_id='ingest-1'",
        (retained_json,),
    )
    unpublished_connection.execute(
        "UPDATE unpublished_graphiti_attempt_receipts "
        "SET receipt_digest=?,receipt_json=? WHERE ingest_id='ingest-1'",
        (receipt_digest, retained_json),
    )
    unpublished_connection.commit()
    unpublished_connection.close()

    proposal = receipt["proposals"][0]
    authority_connection = sqlite3.connect(authority)
    authority_connection.execute(
        "INSERT INTO graphiti_adapter_attempts VALUES(?,?,?,?,?,?,?)",
        (
            str(attempt_ids("ingest-1", 1)[0]),
            "COMPLETE",
            1,
            "proposal-set",
            "output",
            "run",
            "run-version",
        ),
    )
    authority_connection.execute(
        "INSERT INTO extraction_outputs VALUES(?,?,?,?,?)",
        (
            "output",
            "run",
            "run-version",
            raw_bytes,
            digest_bytes(raw_bytes),
        ),
    )
    authority_connection.execute(
        "INSERT INTO extraction_proposals VALUES(?,?,?,?,?,?,?)",
        (
            "proposal-id",
            "proposal-set",
            "output",
            "run",
            "run-version",
            proposal["local_id"],
            digest_canonical(proposal),
        ),
    )
    authority_connection.commit()
    authority_connection.close()


def _nonterminal_obligation(
    connection: sqlite3.Connection,
    *,
    ledger_seq: int,
    item_key: str,
    ingest_id: str,
    with_event: bool = True,
    provider_dispatched: int = 0,
    revision_digest: str | None = None,
    published_at: str = "",
    updated_at: str = "",
    revision_id: str | None = None,
) -> None:
    at = "2026-09-01T12:00:00.000000Z"
    ledger_digest = f"sha256:{ledger_seq:064x}"
    revision_digest = revision_digest or f"revision-{item_key}"
    revision_id = revision_id or f"revision-id-{item_key}"
    identity = EffectiveRevisionIdentity(
        source_id="source",
        item_key=item_key,
        revision_digest=revision_digest,
        first_observed_at=at,
    )
    payload = effective_revision_landed_payload(
        identity,
        published_at=published_at,
        updated_at=updated_at,
        ingest_ids=(ingest_id,),
        first_observed_at=at,
    )
    payload_digest = digest_canonical(payload)
    previous = connection.execute(
        "SELECT digest FROM ledger ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    connection.execute(
        "INSERT INTO ledger(seq,at,kind,payload_digest,payload_json,prev_digest,"
        "digest) VALUES(?,?,?,?,?,?,?)",
        (
            ledger_seq,
            at,
            "EFFECTIVE_REVISION_LANDED",
            payload_digest,
            canonical_json_bytes(payload).decode("utf-8"),
            "GENESIS" if previous is None else str(previous[0]),
            ledger_digest,
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_effective_revision_landed("
        "source_id,item_key,revision_digest,published_at,updated_at,"
        "first_observed_at,ingest_ids_json,legacy_v10,payload_digest,"
        "ledger_digest,at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            "source",
            item_key,
            revision_digest,
            published_at,
            updated_at,
            at,
            json.dumps([ingest_id]),
            0,
            payload_digest,
            ledger_digest,
            at,
        ),
    )
    if not with_event:
        return
    manifest = {
        "event_type": "EFFECTIVE_SOURCE_REVISION_LANDED",
        "ledger_seq": ledger_seq,
        "ledger_digest": ledger_digest,
        "landed_ingest_ids": [ingest_id],
        "landed_payload_digest": payload_digest,
        "unit_refs": [
            {
                "ingest_id": ingest_id,
                "revision_id": revision_id,
                "representation_digest": f"representation-{item_key}",
                "chunk_digest": f"chunk-{item_key}",
                "chunk_ordinal": 1,
                "predecessor_ingest_id": None,
            }
        ],
    }
    connection.execute(
        "INSERT INTO unpublished_graphiti_revision_events("
        "event_id,ledger_seq,ledger_digest,source_id,item_key,revision_digest,"
        "published_at,updated_at,landed_at,manifest_json,manifest_digest,"
        "unit_count,projector_version,projection_generation,state,attempt_count,"
        "available_at,provider_dispatched) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            ledger_digest,
            ledger_seq,
            ledger_digest,
            "source",
            item_key,
            revision_digest,
            published_at,
            updated_at,
            at,
            json.dumps(manifest, sort_keys=True),
            digest_canonical(manifest),
            1,
            GRAPHITI_EVENT_PROJECTOR_VERSION,
            GRAPHITI_EVENT_PROJECTION_GENERATION,
            "QUEUED",
            0,
            at,
            provider_dispatched,
        ),
    )


def _current_unit(
    *,
    item_key: str,
    ingest_id: str,
    revision_digest: str | None = None,
    published_at: str | None = None,
    updated_at: str | None = None,
    revision_id: str | None = None,
    authority_item_id: str | None = None,
) -> SimpleNamespace:
    revision_id = revision_id or f"revision-id-{item_key}"
    return SimpleNamespace(
        source_id="source",
        item_key=item_key,
        revision_digest=revision_digest or f"revision-{item_key}",
        published_at=published_at,
        updated_at=updated_at,
        ingest_id=ingest_id,
        proving_run_id="run",
        observation_digest="observation",
        revision_id=revision_id,
        representation_digest=f"representation-{item_key}",
        digest=f"chunk-{item_key}",
        chunk_ordinal=1,
        chunk_count=1,
        predecessor_ingest_id=None,
        observed_at="2026-09-01T12:00:00.000000Z",
        effective_pull_first_observed_at="2026-09-01T12:00:00.000000Z",
        authority=(
            None
            if authority_item_id is None
            else SimpleNamespace(
                item_id=authority_item_id,
                revision_id=revision_id,
                records=(),
            )
        ),
    )


def _packet(proving: Path, unpublished: Path, **kwargs: object) -> dict[str, object]:
    return build_graphiti_steady_state_packet(
        proving_store=proving,
        unpublished_store=unpublished,
        head_sha="head",
        tree_sha="tree",
        observed_at=NOW,
        **kwargs,
    )


def _seed_proving_accountability(
    proving: Path,
    *,
    held_source_id: str | None = None,
) -> None:
    from newsroom.increment9.proving import _store_body
    from newsroom.increment9.proving_store_schema import create_proving_schema

    with sqlite3.connect(proving) as initial:
        initial.execute("DROP TABLE proof")
    # Metadata fixture setup is not a proving writer dispatch. Pytest directory
    # names may contain "production"; the real writer path guard stays intact.
    connection = sqlite3.connect(proving)
    connection.execute("PRAGMA foreign_keys=ON")
    create_proving_schema(connection)
    connection.execute(
        "INSERT INTO proving_runs VALUES(?,?,?,?,?,?)",
        ("run-1", "2026-09-01T12:00:00Z", 0, 0, 0, 0),
    )
    connection.executemany(
        "INSERT INTO proving_gates VALUES(?,?,?,?)",
        (("run-1", gate_id, "PASS", "fixture") for gate_id in PROVING_GATES),
    )
    for source_id in SOURCE_IDS:
        body = source_id.encode()
        _store_body(connection, digest_bytes(body), body)
        connection.execute(
            "INSERT INTO proving_observations VALUES(?,?,?,?,?,?,?,?)",
            (
                source_id,
                "run-1",
                "2026-09-01T12:00:00Z",
                SOURCE_URLS[source_id],
                200,
                digest_bytes(body),
                1,
                None,
            ),
        )
        held = source_id == held_source_id
        connection.execute(
            "INSERT INTO proving_source_health VALUES(?,?,?,?,?,?,?,?)",
            (
                source_id,
                "run-1",
                "HELD" if held else "ACTIVE",
                SOURCE_URLS[source_id],
                1,
                "EXPLICIT_FIXTURE_HOLD" if held else None,
                None,
                None,
            ),
        )
    connection.commit()
    connection.close()


def _authority_store(tmp_path: Path) -> Path:
    path = tmp_path / "authority.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE authority_migrations(
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL
        );
        CREATE TABLE ledger_events(ledger_seq INTEGER PRIMARY KEY);
        CREATE TABLE extraction_proposals(
            proposal_id TEXT PRIMARY KEY,
            proposal_set_id TEXT NOT NULL,
            output_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            run_version_id TEXT NOT NULL,
            local_id TEXT NOT NULL,
            semantic_digest TEXT NOT NULL
        );
        CREATE TABLE extraction_outputs(
            output_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            run_version_id TEXT NOT NULL,
            canonical_bytes BLOB NOT NULL,
            canonical_digest TEXT NOT NULL
        );
        CREATE TABLE entity_resolution_decisions(decision_id TEXT PRIMARY KEY);
        CREATE TABLE editorial_relation_decisions(decision_id TEXT PRIMARY KEY);
        CREATE TABLE graphiti_adapter_attempts(
            attempt_id TEXT PRIMARY KEY,
            outcome TEXT NOT NULL,
            proposal_count INTEGER NOT NULL,
            proposal_set_id TEXT,
            extraction_output_id TEXT,
            run_id TEXT NOT NULL,
            run_version_id TEXT NOT NULL
        );
        CREATE TABLE projection_ontology_contracts(
            contract_digest TEXT PRIMARY KEY,
            ontology_id TEXT NOT NULL,
            ontology_version TEXT NOT NULL
        );
        CREATE TABLE projection_mapping_contracts(
            contract_digest TEXT PRIMARY KEY,
            mapping_id TEXT NOT NULL,
            mapping_version TEXT NOT NULL,
            ontology_contract_digest TEXT NOT NULL
        );
        CREATE TABLE projection_family_definitions(
            definition_digest TEXT PRIMARY KEY,
            family_id TEXT NOT NULL,
            definition_version TEXT NOT NULL,
            projector_version TEXT NOT NULL,
            ontology_contract_digest TEXT NOT NULL,
            mapping_contract_digest TEXT NOT NULL
        );
        CREATE TABLE projection_families(
            family_id TEXT PRIMARY KEY,
            definition_digest TEXT NOT NULL
        );
        CREATE TABLE projection_generations(
            generation_id TEXT PRIMARY KEY,
            family_id TEXT NOT NULL,
            state TEXT NOT NULL,
            validated_through_ledger_seq INTEGER NOT NULL
        );
        INSERT INTO ledger_events VALUES(7);
        """
    )
    ontology = increment4_admitted_ontology_v1()
    mapping = increment4_admitted_mapping_v1(ontology)
    family = increment4_admitted_family_v1(ontology, mapping)
    connection.executemany(
        "INSERT INTO authority_migrations VALUES(?,?,?,?)",
        tuple(
            (int(item.version), str(item.name), str(item.checksum), NOW.isoformat())
            for item in MIGRATIONS
        ),
    )
    connection.execute(f"PRAGMA user_version={max(int(item.version) for item in MIGRATIONS)}")
    connection.execute(
        "INSERT INTO projection_ontology_contracts VALUES(?,?,?)",
        (ontology.contract_digest, ontology.ontology_id, ontology.ontology_version),
    )
    connection.execute(
        "INSERT INTO projection_mapping_contracts VALUES(?,?,?,?)",
        (
            mapping.contract_digest,
            mapping.mapping_id,
            mapping.mapping_version,
            mapping.ontology_contract_digest,
        ),
    )
    connection.execute(
        "INSERT INTO projection_family_definitions VALUES(?,?,?,?,?,?)",
        (
            family.digest,
            family.family_id,
            family.definition_version,
            family.projector_version,
            family.ontology_contract_digest,
            family.mapping_contract_digest,
        ),
    )
    connection.execute(
        "INSERT INTO projection_families VALUES(?,?)",
        (family.family_id, family.digest),
    )
    connection.execute(
        "INSERT INTO projection_generations VALUES(?,?,?,?)",
        (
            "00000000-0000-4000-8000-000000000895",
            family.family_id,
            "ACTIVE",
            7,
        ),
    )
    connection.commit()
    connection.close()
    return path


def _campaign_input(packet: dict[str, object]) -> dict[str, object]:
    partition = packet["historical_partition"]
    assert isinstance(partition, dict)
    candidates = partition["current_preflight_candidates"]
    assert isinstance(candidates, list)
    selection = {
        "policy_id": "issue-895-current-preflight",
        "policy_version": "v1",
    }
    return {
        "schema_version": "newsroom.graphiti-bounded-campaign-input.v4",
        "focus_gate": {
            "head_sha": "head",
            "tree_sha": "tree",
            "conclusion": "SUCCESS",
            "manifest_digest": "sha256:" + "f" * 64,
        },
        "selection_policy": selection,
        "provider": {
            "transport_id": "CURSOR_SDK",
            "provider_id": "provider",
            "model_id": "model",
            "embedding_provider_id": "embedding-provider",
            "embedding_model_id": "embedding",
        },
        "graph": {"destination_id": GRAPH_DESTINATION_ID},
        "caps": {
            "per_event": {
                "proposals": 20,
                "entity_admits": 10,
                "relation_admits": 10,
                "effects": 20,
                "retries": 1,
                "fallbacks": 0,
            },
            "total": {
                "events": len(candidates),
                "proposals": 20,
                "entity_admits": 10,
                "relation_admits": 10,
                "effects": 20,
                "retries": 1,
                "fallbacks": 0,
                "wall_time_seconds": 600,
                "spend_gbp_microunits": 1000000,
            },
            "rate": {"events_per_minute": 1},
        },
        "ramp": {
            "phases": [
                {
                    "phase_id": "phase-1",
                    "event_limit": len(candidates),
                    "entry_conditions": [
                        "EXACT_SNAPSHOT_AND_IDENTITY_RECONFIRMED",
                        "OWNER_F4_GO_RETAINED",
                    ],
                    "advance_conditions": [
                        "ALL_EXACT_RECEIPTS_RECONCILED",
                        "CAPS_AND_ACCOUNTING_RECONCILED",
                        "NO_STOP_CONDITION_OBSERVED",
                    ],
                }
            ]
        },
        "recovery": {
            "backup_identity": "backup-before-campaign",
            "rollback_procedure_id": "restore-active-generation",
            "reconciliation_procedure_id": "increment4-full-reconciliation",
        },
        "immediate_stop_conditions": [
            "CAP_REACHED",
            "CIRCUIT_OPEN",
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
            "WALL_TIME_CAP_REACHED",
        ],
        "success_objectives": {
            "watermark": "selected cohort terminal",
            "backlog": 0,
            "velocity": "service_at_least_arrival",
            "lag": {"max_oldest_eligible_seconds": 300},
            "reconciliation": "exact",
        },
        "campaign_authorised": False,
    }


def _graph_destination_reconciliation(
    packet: dict[str, object],
    *,
    family_id: str | None = None,
    generation_id: str | None = None,
    checkpoint_ledger_seq: int | None = None,
    serving_time: str | None = None,
) -> StructuralReconciliationView:
    authority_evidence = packet["authority_snapshot_evidence"]
    assert isinstance(authority_evidence, dict)
    graph = authority_evidence["active_projection_authority"]
    assert isinstance(graph, dict)
    return StructuralReconciliationView(
        family_id=family_id or str(graph["family_id"]),
        generation_id=ProjectionGenerationId.parse(
            generation_id or str(graph["generation_id"])
        ),
        checkpoint_ledger_seq=(
            int(graph["validated_through_ledger_seq"])
            if checkpoint_ledger_seq is None
            else checkpoint_ledger_seq
        ),
        projection_state_digest="sha256:" + "d" * 64,
        serving_time=UtcTimestamp.parse(
            serving_time or str(packet["observed_at"])
        ),
    )


def _governed_runtime(packet: dict[str, object]) -> GraphitiCampaignRuntime:
    stores = packet["store_snapshots"]
    assert isinstance(stores, dict)
    authority = stores["authority"]
    assert isinstance(authority, dict)
    return _mint_graphiti_campaign_runtime(
        graphiti=object(),
        admission_factory=lambda _connection: object(),
        bind_unit_authority=lambda unit: unit,
        graph_state_fence=lambda _campaign: {},
        graph_destination_id=GRAPH_DESTINATION_ID,
        authority_store_source_path=str(authority["source_path"]),
        authority_store_descriptor_digest=str(authority["descriptor_digest"]),
    )


def _bounded_candidate_packet(
    proving: Path,
    unpublished: Path,
    authority: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(item_key="candidate", ingest_id="candidate-ingest"),
        ),
    )
    preparation = _packet(
        proving,
        unpublished,
        authority_store=authority,
    )
    return _packet(
        proving,
        unpublished,
        authority_store=authority,
        campaign_input=_campaign_input(preparation),
        graph_destination_reconciliation=(
            _graph_destination_reconciliation(preparation)
        ),
        governed_runtime=_governed_runtime(preparation),
    )


def _seed_admission_obligation(
    connection: sqlite3.Connection,
    *,
    action: str | None,
) -> None:
    at = "2026-09-01T12:00:00.000000Z"
    proposal_key = "proposal-1"
    connection.execute(
        "INSERT INTO unpublished_graphiti_admission_queue("
        "proposal_key,ingest_id,source_revision_id,source_receipt_digest,"
        "proposal_digest,proposal_kind,request_json,request_digest,state,"
        "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            proposal_key,
            "proposal-ingest",
            "proposal-revision",
            "sha256:" + "1" * 64,
            "sha256:" + "2" * 64,
            "ENTITY_MENTION",
            "{}",
            "sha256:" + "3" * 64,
            "READY" if action is None else "TERMINAL",
            at,
            at,
        ),
    )
    if action is not None:
        decision = GraphitiGovernedDecision(
            proposal_key=proposal_key,
            proposal_digest="sha256:" + "2" * 64,
            proposal_kind=ExtractionProposalKind.ENTITY_MENTION,
            proposal_local_id="proposal-local-1",
            action=GraphitiProposalAdmissionAction(action),
            decision_id="decision-1",
            authority_ledger_seq=1,
            reason_code="FIXTURE_POLICY",
            authority_receipt_digest="sha256:" + "4" * 64,
        )
        decision_json = canonical_json_bytes(decision.canonical_value()).decode(
            "utf-8"
        )
        connection.execute(
            "INSERT INTO unpublished_graphiti_admission_decisions("
            "proposal_key,action,decision_id,authority_ledger_seq,reason_code,"
            "authority_receipt_digest,decision_json,decision_digest,decided_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                proposal_key,
                action,
                "decision-1",
                1,
                "FIXTURE_POLICY",
                "sha256:" + "4" * 64,
                decision_json,
                digest_bytes(decision_json.encode("utf-8")),
                at,
            ),
        )
    connection.commit()


def test_campaign_runtime_rejects_non_governed_construction_token() -> None:
    with pytest.raises(
        TypeError, match="campaign runtimes require the governed worker composer"
    ):
        GraphitiCampaignRuntime(
            graphiti=object(),
            admission_factory=lambda _connection: object(),
            bind_unit_authority=lambda unit: unit,
            graph_state_fence=lambda _campaign: {},
            authority_store_source_path="/authority.sqlite3",
            authority_store_descriptor_digest="sha256:" + "a" * 64,
            graph_destination_id=GRAPH_DESTINATION_ID,
            _construction_token=object(),
        )


def test_read_only_snapshot_does_not_copy_the_store(tmp_path: Path) -> None:
    from newsroom.control_plane import read_only_snapshot as snapshot_module

    path = tmp_path / "store.sqlite3"
    sqlite3.connect(path).close()
    source = inspect.getsource(snapshot_module.read_only_snapshot)
    assert "backup(" not in source
    assert "TemporaryDirectory" not in source
    with read_only_snapshot(path) as snapshot:
        assert snapshot.snapshot_files == (
            {
                "name": "store.sqlite3",
                "size": path.stat().st_size,
                "copy_omitted": True,
            },
        )
        assert snapshot.connection.execute("PRAGMA query_only").fetchone() == (1,)


def test_authority_snapshot_evidence_omits_integrity_check() -> None:
    source = inspect.getsource(_authority_snapshot_evidence)
    assert 'execute("PRAGMA integrity_check")' not in source
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE authority_migrations (version INTEGER, name TEXT, checksum TEXT)"
    )
    connection.execute("CREATE TABLE ledger_events (ledger_seq INTEGER)")
    for name in (
        "extraction_proposals",
        "graphiti_adapter_attempts",
        "entity_resolution_decisions",
        "editorial_relation_decisions",
    ):
        connection.execute(f"CREATE TABLE {name} (x INTEGER)")
    connection.execute(
        "CREATE TABLE projection_generations ("
        "generation_id TEXT, state TEXT, validated_through_ledger_seq INTEGER,"
        "family_id TEXT)"
    )
    connection.execute(
        "CREATE TABLE projection_families (family_id TEXT, definition_digest TEXT)"
    )
    connection.execute(
        "CREATE TABLE projection_family_definitions ("
        "definition_digest TEXT, projector_version TEXT,"
        "ontology_contract_digest TEXT, mapping_contract_digest TEXT)"
    )
    connection.execute(
        "CREATE TABLE projection_ontology_contracts ("
        "contract_digest TEXT, ontology_id TEXT, ontology_version TEXT)"
    )
    connection.execute(
        "CREATE TABLE projection_mapping_contracts ("
        "contract_digest TEXT, mapping_id TEXT, mapping_version TEXT)"
    )
    evidence, blockers = _authority_snapshot_evidence(connection)
    assert evidence["integrity_check"] == "omitted"
    assert "AUTHORITY_STORE_INTEGRITY_FAILURE" not in blockers


def test_uncopied_store_identity_detects_same_count_payload_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import newsroom.control_plane.graphiti_steady_state as module

    monkeypatch.setattr(module, "_SERIALIZE_MAX_BYTES", 0)
    proving = tmp_path / "proving.sqlite3"
    unpublished = tmp_path / "unpublished.sqlite3"
    authority = tmp_path / "authority.sqlite3"
    with sqlite3.connect(proving) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            "CREATE TABLE proving_observations(run_id TEXT, body BLOB NOT NULL)"
        )
        connection.execute(
            "INSERT INTO proving_observations VALUES ('run', x'01')"
        )
    with sqlite3.connect(unpublished) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            "CREATE TABLE unpublished_graphiti_ingest("
            "ingest_id TEXT PRIMARY KEY, receipt_digest TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO unpublished_graphiti_ingest VALUES ('ing', 'state-a')"
        )
    with sqlite3.connect(authority) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            "CREATE TABLE ledger_events("
            "ledger_seq INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
        )
        connection.execute("INSERT INTO ledger_events VALUES (1, 'payload-a')")
    first = graphiti_store_snapshot_digests(
        proving_store=proving,
        unpublished_store=unpublished,
        authority_store=authority,
    )
    with sqlite3.connect(authority) as connection:
        assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    assert (
        graphiti_store_snapshot_digests(
            proving_store=proving,
            unpublished_store=unpublished,
            authority_store=authority,
        )
        == first
    )
    with sqlite3.connect(proving) as connection:
        connection.execute("UPDATE proving_observations SET body=x'02'")
    after_proving = graphiti_store_snapshot_digests(
        proving_store=proving,
        unpublished_store=unpublished,
        authority_store=authority,
    )
    assert after_proving["proving"] != first["proving"]
    with sqlite3.connect(unpublished) as connection:
        connection.execute(
            "UPDATE unpublished_graphiti_ingest SET receipt_digest='state-b'"
        )
    after_ingest = graphiti_store_snapshot_digests(
        proving_store=proving,
        unpublished_store=unpublished,
        authority_store=authority,
    )
    assert after_ingest["unpublished"] != after_proving["unpublished"]
    with sqlite3.connect(authority) as connection:
        connection.execute("UPDATE ledger_events SET payload='payload-b'")
    after_payload = graphiti_store_snapshot_digests(
        proving_store=proving,
        unpublished_store=unpublished,
        authority_store=authority,
    )
    assert after_payload["authority"] != after_ingest["authority"]


def test_wal_snapshot_is_logical_and_query_only(tmp_path: Path) -> None:
    path = tmp_path / "wal.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE evidence(value TEXT)")
    connection.execute("INSERT INTO evidence VALUES('retained')")
    connection.commit()
    with read_only_snapshot(path) as snapshot:
        assert snapshot.connection.execute(
            "SELECT value FROM evidence"
        ).fetchone() == ("retained",)
        assert snapshot.connection.execute("PRAGMA query_only").fetchone() == (1,)

    assert connection.execute("SELECT value FROM evidence").fetchone() == ("retained",)
    connection.close()


def test_store_identity_survives_wal_checkpoint_but_detects_data_change(
    tmp_path: Path,
) -> None:
    proving = tmp_path / "proving.sqlite3"
    unpublished = tmp_path / "unpublished.sqlite3"
    authority = tmp_path / "authority.sqlite3"
    connections = []
    for path in (proving, unpublished, authority):
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE evidence(value TEXT)")
        connection.execute("INSERT INTO evidence VALUES('retained')")
        connection.commit()
        connections.append(connection)

    while_open = graphiti_store_snapshot_digests(
        proving_store=proving,
        unpublished_store=unpublished,
        authority_store=authority,
    )
    for connection in connections:
        connection.close()
    after_checkpoint = graphiti_store_snapshot_digests(
        proving_store=proving,
        unpublished_store=unpublished,
        authority_store=authority,
    )

    assert after_checkpoint == while_open

    with sqlite3.connect(authority) as connection:
        connection.execute("UPDATE evidence SET value='changed'")
    assert graphiti_store_snapshot_digests(
        proving_store=proving,
        unpublished_store=unpublished,
        authority_store=authority,
    )["authority"] != after_checkpoint["authority"]


def test_store_identity_binds_copied_snapshot_digest_not_serialize(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE evidence(value TEXT)")
    connection.execute("INSERT INTO evidence VALUES('retained')")
    connection.commit()
    planted = "ab" * 32
    with read_only_snapshot(path) as snapshot:
        assert snapshot.snapshot_files[0].get("copy_omitted") is True
        assert "sha256" not in snapshot.snapshot_files[0]
        planted_snapshot = SimpleNamespace(
            connection=snapshot.connection,
            source_path=snapshot.source_path,
            source_files=snapshot.source_files,
            snapshot_files=(
                {
                    "name": "authority.sqlite3",
                    "size": 1,
                    "sha256": planted,
                },
            ),
        )
        descriptor = _store_descriptor(planted_snapshot)
        assert descriptor["logical_content_digest"] == f"sha256:{planted}"
        assert descriptor["logical_content_digest"] != digest_bytes(
            snapshot.connection.serialize()
        )
    connection.close()


def test_snapshot_accepts_checkpoint_churn_without_logical_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wal.sqlite3"
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE evidence(value TEXT)")
    writer.execute("INSERT INTO evidence VALUES('retained')")
    writer.commit()

    with read_only_snapshot(path) as snapshot:
        writer.execute("PRAGMA wal_checkpoint(PASSIVE)")
        path.touch()
        assert snapshot.connection.execute(
            "SELECT value FROM evidence"
        ).fetchone() == ("retained",)
    assert writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0

    writer.close()


def test_snapshot_is_transaction_consistent_across_logical_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wal.sqlite3"
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE evidence(value TEXT)")
    writer.execute("INSERT INTO evidence VALUES('retained')")
    writer.commit()

    with read_only_snapshot(path) as snapshot:
        writer.execute("UPDATE evidence SET value='changed'")
        writer.commit()
        assert snapshot.connection.execute(
            "SELECT value FROM evidence"
        ).fetchone() == ("retained",)

    with read_only_snapshot(path) as snapshot:
        assert snapshot.connection.execute(
            "SELECT value FROM evidence"
        ).fetchone() == ("changed",)

    writer.close()


def test_missing_campaign_inputs_are_finite_no_go(tmp_path: Path) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    connection.close()

    packet = _packet(proving, unpublished)

    assert packet["verdict"] == "NO_GO"
    assert packet["readiness"] == "ENGINEERING_PREPARATION_ONLY"
    assert "AUTHORITY_STORE_UNCONFIGURED" in packet["blockers"]
    assert "CAMPAIGN_INPUT_MISSING" in packet["blockers"]
    assert packet["runtime_composition"]["durable_proposal_envelope_binding"] is False
    assert packet["non_effects"] == {
        "provider_calls": 0,
        "store_mutations": 0,
        "service_loads": 0,
        "publication_effects": 0,
        "production_admission_effects": 0,
    }


def test_zero_proposal_terminal_receipt_is_success(tmp_path: Path) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _terminal_zero_proposal(connection)
    connection.close()

    packet = _packet(proving, unpublished)

    assert packet["terminal_receipts"]["zero_proposal_success_count"] == 1
    assert packet["terminal_receipts"]["integrity_failures"] == []
    assert packet["landed_event_accounting"]["one_to_one"] is True


def test_historical_partition_is_total_disjoint_and_candidates_are_unauthorised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _terminal_zero_proposal(connection)
    _nonterminal_obligation(
        connection,
        ledger_seq=2,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    legacy_manifest = json.loads(
        connection.execute(
            "SELECT manifest_json FROM unpublished_graphiti_revision_events "
            "WHERE ledger_seq=2"
        ).fetchone()[0]
    )
    legacy_manifest["landed_ingest_ids"] = []
    legacy_manifest["unit_refs"] = []
    connection.execute(
        "UPDATE unpublished_graphiti_revision_events "
        "SET unit_count=0,manifest_json=?,manifest_digest=? WHERE ledger_seq=2",
        (
            json.dumps(legacy_manifest, sort_keys=True),
            digest_canonical(legacy_manifest),
        ),
    )
    _nonterminal_obligation(
        connection,
        ledger_seq=3,
        item_key="rights-held",
        ingest_id="held-ingest",
    )
    _nonterminal_obligation(
        connection,
        ledger_seq=4,
        item_key="changed",
        ingest_id="landed-ingest",
    )
    _nonterminal_obligation(
        connection,
        ledger_seq=5,
        item_key="no-event",
        ingest_id="missing-event-ingest",
        with_event=False,
    )
    _nonterminal_obligation(
        connection,
        ledger_seq=6,
        item_key="ambiguous-effect",
        ingest_id="ambiguous-ingest",
        provider_dispatched=1,
    )
    connection.commit()
    connection.close()
    calls = 0

    def load_once(_connection, *, evaluated_at):
        nonlocal calls
        calls += 1
        assert evaluated_at == NOW
        return (
            _current_unit(
                item_key="candidate",
                ingest_id="hydrated-current-ingest",
            ),
            _current_unit(
                item_key="changed",
                ingest_id="current-ingest",
            ),
        )

    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        load_once,
    )

    partition = _packet(proving, unpublished)["historical_partition"]

    assert calls == 1
    assert partition["universe_count"] == 6
    assert partition["partitioned_count"] == 6
    assert partition["disjoint"] is True
    assert partition["total"] is True
    assert partition["categories"]["VERIFIED_TERMINAL"]["ledger_sequences"] == [1]
    candidate = partition["categories"][
        "CURRENT_DISPATCH_PREFLIGHT_CANDIDATE"
    ]
    assert candidate["ledger_sequences"] == [2]
    assert candidate["dispatch_authorised"] is False
    assert partition["current_preflight_candidates"][0]["ingest_ids"] == [
        "hydrated-current-ingest"
    ]
    assert partition["categories"]["RIGHTS_OR_INPUT_HELD"][
        "ledger_sequences"
    ] == [3, 4, 5]
    assert partition["categories"]["RIGHTS_OR_INPUT_HELD"]["reason_counts"][
        "RESOLVED_INGEST_IDS_DIFFER_FROM_LANDED"
    ] == 1
    assert partition["categories"][
        "NON_REPLAYABLE_OR_AMBIGUOUS_EFFECT_HOLD"
    ]["ledger_sequences"] == [6]


@pytest.mark.parametrize(
    ("retained_revision_id", "candidate_sequences", "held_sequences"),
    ((None, [1], [2]), ("revision-later", [2], [1]), ("revision-other", [], [1, 2])),
)
def test_unchanged_content_reobserved_at_changed_timestamp_converges_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retained_revision_id: str | None,
    candidate_sequences: list[int],
    held_sequences: list[int],
) -> None:
    proving, _unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="same-item",
        ingest_id="earlier-ingest",
        revision_digest="same-content",
        published_at="2026-09-01T09:00:00Z",
        revision_id="revision-earlier",
    )
    _nonterminal_obligation(
        connection,
        ledger_seq=2,
        item_key="same-item",
        ingest_id="later-ingest",
        revision_digest="same-content",
        published_at="2026-09-01T10:00:00Z",
        updated_at="2026-09-01T11:00:00Z",
        revision_id="revision-later",
    )
    connection.commit()
    units = (
        _current_unit(
            item_key="same-item",
            ingest_id="earlier-ingest",
            revision_digest="same-content",
            published_at="2026-09-01T09:00:00Z",
            revision_id="revision-earlier",
            authority_item_id="canonical-item",
        ),
        _current_unit(
            item_key="same-item",
            ingest_id="later-ingest",
            revision_digest="same-content",
            published_at="2026-09-01T10:00:00Z",
            updated_at="2026-09-01T11:00:00Z",
            revision_id="revision-later",
            authority_item_id="canonical-item",
        ),
    )
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: units,
    )
    authority = sqlite3.connect(":memory:")
    authority.execute(
        "CREATE TABLE source_revisions("
        "revision_id TEXT PRIMARY KEY,revision_identity_digest TEXT NOT NULL)"
    )
    if retained_revision_id is not None:
        authority.execute(
            "INSERT INTO source_revisions VALUES(?,?)",
            (
                retained_revision_id,
                digest_canonical(
                    {
                        "item_id": "canonical-item",
                        "source_native_revision_token": None,
                        "permitted_state_digest": "same-content",
                    }
                ),
            ),
        )

    snapshot = graphiti_operational_partition_snapshot(
        sqlite3.connect(proving),
        connection,
        authority=authority,
        observed_at=NOW,
    )

    assert [
        item["ledger_seq"] for item in snapshot["actionable"]
    ] == candidate_sequences
    assert [item["ledger_seq"] for item in snapshot["holds"]] == held_sequences
    assert {item["reason"] for item in snapshot["holds"]} == (
        {"UNCHANGED_CONTENT_REOBSERVATION_REQUIRES_OCCURRENCE"}
        if held_sequences
        else set()
    )


def test_operational_partition_reuses_fresh_gap_and_hold_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, _unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    _nonterminal_obligation(
        connection,
        ledger_seq=2,
        item_key="projectable-gap",
        ingest_id="gap-ingest",
        with_event=False,
    )
    _nonterminal_obligation(
        connection,
        ledger_seq=3,
        item_key="claimed-history",
        ingest_id="claimed-ingest",
    )
    connection.execute(
        "UPDATE unpublished_graphiti_revision_events SET state='CLAIMED',"
        "claim_owner='historical-worker',claim_expires_at=? WHERE ledger_seq=3",
        ("2026-09-01T12:05:00.000000Z",),
    )
    _nonterminal_obligation(
        connection,
        ledger_seq=4,
        item_key="rights-held-gap",
        ingest_id="held-ingest",
        with_event=False,
    )
    connection.commit()
    proving_connection = sqlite3.connect(proving)
    authority = sqlite3.connect(":memory:")
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(item_key="candidate", ingest_id="candidate-ingest"),
            _current_unit(item_key="projectable-gap", ingest_id="gap-ingest"),
        ),
    )

    snapshot = graphiti_operational_partition_snapshot(
        proving_connection,
        connection,
        authority=authority,
        observed_at=NOW,
    )

    assert [item["kind"] for item in snapshot["actionable"]] == [
        "FRESH_EVENT",
        "PROJECT_EVENT_GAP",
    ]
    assert [item["ledger_seq"] for item in snapshot["holds"]] == [3, 4]
    assert snapshot["snapshot_digest"] == digest_canonical(
        {key: value for key, value in snapshot.items() if key != "snapshot_digest"}
    )
    authority.close()
    proving_connection.close()
    connection.close()


def _queued_rows(connection: sqlite3.Connection) -> list[tuple[object, ...]]:
    return [
        tuple(row)
        for row in connection.execute(
            "SELECT ledger_seq,state,attempt_count FROM "
            "unpublished_graphiti_revision_events ORDER BY ledger_seq"
        )
    ]


@pytest.mark.parametrize("frontier_state", ["CONFIGURATION_HELD", "DEAD_LETTER"])
def test_empty_actionable_then_frontier_supply_excludes_old_bindable_queued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frontier_state: str,
) -> None:
    proving, _unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="old-backlog",
        ingest_id="old-ingest",
    )
    connection.commit()
    proving_connection = sqlite3.connect(proving)
    authority = sqlite3.connect(":memory:")
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (),
    )

    empty = graphiti_operational_partition_snapshot(
        proving_connection,
        connection,
        authority=authority,
        observed_at=NOW,
    )

    assert empty["actionable"] == []
    assert [item["ledger_seq"] for item in empty["holds"]] == [1]
    assert _queued_rows(connection) == [(1, "QUEUED", 0)]

    _nonterminal_obligation(
        connection,
        ledger_seq=2,
        item_key="frontier",
        ingest_id="frontier-ingest",
    )
    connection.commit()
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(item_key="old-backlog", ingest_id="old-ingest"),
            _current_unit(item_key="frontier", ingest_id="frontier-ingest"),
        ),
    )

    supplied = graphiti_operational_partition_snapshot(
        proving_connection,
        connection,
        authority=authority,
        observed_at=NOW,
    )

    assert [(item["ledger_seq"], item["kind"]) for item in supplied["actionable"]] == [
        (2, "FRESH_EVENT")
    ]
    assert supplied["actionable"][0]["ingest_ids"] == ["frontier-ingest"]
    assert [item["ledger_seq"] for item in supplied["holds"]] == [1]
    assert supplied["holds"][0]["reason"] == PRE_FRONTIER_BACKLOG_HOLD_REASON
    assert _queued_rows(connection) == [(1, "QUEUED", 0), (2, "QUEUED", 0)]
    connection.execute(
        "UPDATE unpublished_graphiti_revision_events SET state=?,attempt_count=1 WHERE ledger_seq=2",
        (frontier_state,),
    )
    connection.commit()
    after = graphiti_operational_partition_snapshot(
        proving_connection, connection, authority=authority, observed_at=NOW,
        pre_frontier_hold_watermark=2,
    )
    assert after["actionable"] == []
    assert after["holds"][0]["reason"] == PRE_FRONTIER_BACKLOG_HOLD_REASON
    assert _queued_rows(connection)[0] == (1, "QUEUED", 0)
    for invalid_pin in (True, "2", 999):
        with pytest.raises(GraphitiEventReconciliationError, match="watermark is invalid"):
            graphiti_operational_partition_snapshot(
                proving_connection, connection, authority=authority, observed_at=NOW,
                pre_frontier_hold_watermark=invalid_pin,
            )
    # The original cut-off does not suppress arrivals or missing-event gaps.
    _nonterminal_obligation(connection, ledger_seq=3, item_key="arrival", ingest_id="arrival-ingest")
    _nonterminal_obligation(connection, ledger_seq=4, item_key="later-arrival", ingest_id="later-ingest")
    _nonterminal_obligation(connection, ledger_seq=5, item_key="gap", ingest_id="gap-ingest", with_event=False)
    connection.commit()
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state.load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: tuple(
            _current_unit(item_key=key, ingest_id=ingest)
            for key, ingest in (
                ("old-backlog", "old-ingest"), ("frontier", "frontier-ingest"),
                ("arrival", "arrival-ingest"), ("later-arrival", "later-ingest"), ("gap", "gap-ingest"),
            )
        ),
    )
    with_arrivals = graphiti_operational_partition_snapshot(
        proving_connection, connection, authority=authority, observed_at=NOW,
        pre_frontier_hold_watermark=2,
    )
    assert [(item["ledger_seq"], item["kind"]) for item in with_arrivals["actionable"]] == [
        (3, "FRESH_EVENT"), (4, "FRESH_EVENT"), (5, "PROJECT_EVENT_GAP"),
    ]
    assert with_arrivals["holds"][0]["reason"] == PRE_FRONTIER_BACKLOG_HOLD_REASON

    authority.close()
    proving_connection.close()
    connection.close()


def test_pre_frontier_bindable_queued_stay_held_when_frontier_is_fresh_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, _unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="old-backlog",
        ingest_id="old-ingest",
    )
    _nonterminal_obligation(
        connection,
        ledger_seq=2,
        item_key="projectable-gap",
        ingest_id="gap-ingest",
        with_event=False,
    )
    _nonterminal_obligation(
        connection,
        ledger_seq=3,
        item_key="frontier",
        ingest_id="frontier-ingest",
    )
    connection.commit()
    proving_connection = sqlite3.connect(proving)
    authority = sqlite3.connect(":memory:")
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(item_key="old-backlog", ingest_id="old-ingest"),
            _current_unit(item_key="projectable-gap", ingest_id="gap-ingest"),
            _current_unit(item_key="frontier", ingest_id="frontier-ingest"),
        ),
    )

    snapshot = graphiti_operational_partition_snapshot(
        proving_connection,
        connection,
        authority=authority,
        observed_at=NOW,
    )

    assert [(item["ledger_seq"], item["kind"]) for item in snapshot["actionable"]] == [
        (2, "PROJECT_EVENT_GAP"),
        (3, "FRESH_EVENT"),
    ]
    assert snapshot["actionable"][1]["ingest_ids"] == ["frontier-ingest"]
    assert snapshot["holds"] == [
        {
            "ledger_seq": 1,
            "event_id": "sha256:" + f"{1:064x}",
            "reason": PRE_FRONTIER_BACKLOG_HOLD_REASON,
        }
    ]
    assert _queued_rows(connection) == [(1, "QUEUED", 0), (3, "QUEUED", 0)]
    authority.close()
    proving_connection.close()
    connection.close()


def test_complete_packet_remains_no_go_without_graph_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    _nonterminal_obligation(
        connection,
        ledger_seq=2,
        item_key="historical-hold",
        ingest_id="held-ingest",
        provider_dispatched=1,
    )
    connection.commit()
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(item_key="candidate", ingest_id="candidate-ingest"),
        ),
    )

    preparation = _packet(proving, unpublished, authority_store=authority)
    campaign = _campaign_input(preparation)
    packet = _packet(
        proving,
        unpublished,
        authority_store=authority,
        campaign_input=campaign,
    )

    assert packet["verdict"] == "NO_GO"
    assert packet["readiness"] == "ENGINEERING_PREPARATION_ONLY"
    assert "GRAPH_DESTINATION_READBACK_UNAVAILABLE" in packet["blockers"]
    assert packet["bounded_campaign"]["campaign_authorised"] is False
    assert packet["bounded_campaign"]["cohort"]["dispatch_authorised"] is False
    assert packet["bounded_campaign"]["cohort"]["claim_performed"] is False
    assert packet["historical_partition"]["categories"][
        "NON_REPLAYABLE_OR_AMBIGUOUS_EFFECT_HOLD"
    ]["count"] == 1
    assert packet["runtime_composition"] == {
        "state": "CANONICAL_OPERATOR_RUNTIME_UNCONFIGURED",
        "authority_store_configured": True,
        "graph_destination_id": None,
        "authority_store_source_path": None,
        "authority_store_descriptor_digest": None,
        "durable_proposal_envelope_binding": False,
        "admission_policy_configured": False,
        "full_generation_projector_configured": False,
        "dormant_worker_path_wired": False,
        "campaign_packet_enforced": False,
        "actual_graph_readback_observed": False,
        "campaign_authorised": False,
    }


def test_owner_input_rejects_extra_derived_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    connection.commit()
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(item_key="candidate", ingest_id="candidate-ingest"),
        ),
    )
    preparation = _packet(proving, unpublished, authority_store=authority)
    campaign = _campaign_input(preparation)
    campaign["code_identity"] = {"head_sha": "head", "tree_sha": "tree"}
    campaign["cohort"] = {
        "event_ids": ["self-addressed-event"],
        "manifest_digest": "sha256:" + "0" * 64,
    }
    campaign["source_snapshot_digests"] = {}

    packet = _packet(
        proving,
        unpublished,
        authority_store=authority,
        campaign_input=campaign,
    )

    assert packet["verdict"] == "NO_GO"
    assert "CAMPAIGN_FIELDS_INVALID" in packet["blockers"]


def test_authenticated_graph_readback_builds_valid_ready_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    connection.commit()
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(item_key="candidate", ingest_id="candidate-ingest"),
        ),
    )
    preparation = _packet(proving, unpublished, authority_store=authority)
    campaign = _campaign_input(preparation)
    reconciliation = _graph_destination_reconciliation(preparation)
    runtime = _governed_runtime(preparation)

    packet = _packet(
        proving,
        unpublished,
        authority_store=authority,
        campaign_input=campaign,
        graph_destination_reconciliation=reconciliation,
        governed_runtime=runtime,
    )

    assert packet["verdict"] == "READY_FOR_OWNER_DECISION"
    assert packet["readiness"] == "F4_CAMPAIGN_READY_FOR_OWNER_DECISION"
    assert packet["blockers"] == []
    assert packet["bounded_campaign"]["cohort"] == {
        "event_ids": ["sha256:" + f"{1:064x}"],
        "manifest_digest": packet["historical_partition"][
            "current_preflight_candidate_manifest_digest"
        ],
        "events": packet["historical_partition"]["current_preflight_candidates"],
        "dispatch_authorised": False,
        "claim_performed": False,
    }
    assert packet["bounded_campaign"]["graph_destination_readback"] == {
        "destination_id": GRAPH_DESTINATION_ID,
        "family_id": reconciliation.family_id,
        "generation_id": str(reconciliation.generation_id),
        "checkpoint_ledger_seq": reconciliation.checkpoint_ledger_seq,
        "projection_state_digest": reconciliation.projection_state_digest,
        "serving_time": reconciliation.serving_time.to_text(),
    }
    assert packet["runtime_composition"]["actual_graph_readback_observed"] is True
    assert packet["runtime_composition"]["authority_store_source_path"] == (
        packet["store_snapshots"]["authority"]["source_path"]
    )
    assert packet["runtime_composition"][
        "authority_store_descriptor_digest"
    ] == packet["store_snapshots"]["authority"]["descriptor_digest"]
    assert packet["runtime_composition"]["graph_destination_id"] == (
        GRAPH_DESTINATION_ID
    )
    assert validate_graphiti_campaign_packet(packet) == packet["bounded_campaign"]

    cap_drift = json.loads(json.dumps(packet))
    cap_drift["bounded_campaign"]["caps"]["total"]["events"] = 2
    cap_drift["packet_digest"] = digest_canonical(
        {key: value for key, value in cap_drift.items() if key != "packet_digest"}
    )
    with pytest.raises(ValueError, match="caps differ"):
        validate_graphiti_campaign_packet(cap_drift)

    ramp_drift = json.loads(json.dumps(packet))
    ramp_drift["bounded_campaign"]["ramp"]["phases"][-1]["event_limit"] = 2
    ramp_drift["packet_digest"] = digest_canonical(
        {key: value for key, value in ramp_drift.items() if key != "packet_digest"}
    )
    with pytest.raises(ValueError, match="cohort cap differs"):
        validate_graphiti_campaign_packet(ramp_drift)

    stop_drift = json.loads(json.dumps(packet))
    stop_drift["bounded_campaign"]["immediate_stop_conditions"].remove(
        "RIGHTS_DRIFT"
    )
    stop_drift["packet_digest"] = digest_canonical(
        {key: value for key, value in stop_drift.items() if key != "packet_digest"}
    )
    with pytest.raises(ValueError, match="stop conditions differ"):
        validate_graphiti_campaign_packet(stop_drift)

    runtime_graph_drift = json.loads(json.dumps(packet))
    runtime_graph_drift["runtime_composition"]["graph_destination_id"] = (
        "sha256:" + "8" * 64
    )
    runtime_graph_drift["packet_digest"] = digest_canonical(
        {
            key: value
            for key, value in runtime_graph_drift.items()
            if key != "packet_digest"
        }
    )
    with pytest.raises(ValueError, match="graph readback differs"):
        validate_graphiti_campaign_packet(runtime_graph_drift)


def test_bootstrap_overcount_narrows_to_selected_cohort_after_retry_held_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate-a",
        ingest_id="candidate-a-ingest",
    )
    _nonterminal_obligation(
        connection,
        ledger_seq=2,
        item_key="candidate-b",
        ingest_id="candidate-b-ingest",
    )
    _nonterminal_obligation(
        connection,
        ledger_seq=3,
        item_key="retry-held",
        ingest_id="retry-held-ingest",
    )
    held_event_id = "sha256:" + f"{3:064x}"
    connection.execute(
        "UPDATE unpublished_graphiti_revision_events SET state='RETRY_HELD',"
        "attempt_count=1 WHERE ledger_seq=3"
    )
    connection.execute(
        "CREATE TABLE issue_790_graphiti_retry_exclusions(event_id TEXT PRIMARY KEY)"
    )
    connection.execute(
        "INSERT INTO issue_790_graphiti_retry_exclusions(event_id) VALUES(?)",
        (held_event_id,),
    )
    connection.commit()
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(
                item_key="candidate-a", ingest_id="candidate-a-ingest"
            ),
            _current_unit(
                item_key="candidate-b", ingest_id="candidate-b-ingest"
            ),
        ),
    )
    preparation = _packet(proving, unpublished, authority_store=authority)
    campaign = build_operational_campaign_input(
        head_sha="head",
        tree_sha="tree",
        focus_manifest_digest="sha256:" + "f" * 64,
        graph_destination_id=GRAPH_DESTINATION_ID,
        candidate_event_count=3,
        recovery_identity="sha256:" + "a" * 64,
    )
    assert campaign["caps"]["total"]["events"] == 3
    packet = _packet(
        proving,
        unpublished,
        authority_store=authority,
        campaign_input=campaign,
        graph_destination_reconciliation=_graph_destination_reconciliation(
            preparation
        ),
        governed_runtime=_governed_runtime(preparation),
    )

    selected = packet["bounded_campaign"]["cohort"]["event_ids"]
    assert packet["verdict"] == "READY_FOR_OWNER_DECISION"
    assert packet["blockers"] == []
    assert "TOTAL_EVENT_CAP_DIFFERS_FROM_SELECTED_COHORT" not in packet["blockers"]
    assert held_event_id not in selected
    assert packet["historical_partition"]["categories"][
        "NON_REPLAYABLE_OR_AMBIGUOUS_EFFECT_HOLD"
    ]["ledger_sequences"] == [3]
    assert selected == [
        "sha256:" + f"{1:064x}",
        "sha256:" + f"{2:064x}",
    ]
    assert packet["bounded_campaign"]["caps"]["total"]["events"] == 2
    assert packet["bounded_campaign"]["caps"]["total"]["spend_gbp_microunits"] == (
        1_000_000
    )
    assert [
        phase["event_limit"]
        for phase in packet["bounded_campaign"]["ramp"]["phases"]
    ] == [1, 2]
    assert campaign["caps"]["total"]["events"] == 3
    assert validate_graphiti_campaign_packet(packet) == packet["bounded_campaign"]


def _retry_held_overcount_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    selected_event_count: int,
    campaign_event_count: int,
    mutate_campaign: object = None,
) -> tuple[dict[str, object], dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    proving, unpublished, connection = _stores(tmp_path)
    units = []
    for seq in range(1, selected_event_count + 1):
        item_key = f"candidate-{seq}"
        ingest_id = f"{item_key}-ingest"
        _nonterminal_obligation(
            connection,
            ledger_seq=seq,
            item_key=item_key,
            ingest_id=ingest_id,
        )
        units.append(_current_unit(item_key=item_key, ingest_id=ingest_id))
    held_seq = selected_event_count + 1
    _nonterminal_obligation(
        connection,
        ledger_seq=held_seq,
        item_key="retry-held",
        ingest_id="retry-held-ingest",
    )
    held_event_id = "sha256:" + f"{held_seq:064x}"
    connection.execute(
        "UPDATE unpublished_graphiti_revision_events SET state='RETRY_HELD',"
        "attempt_count=1 WHERE ledger_seq=?",
        (held_seq,),
    )
    connection.execute(
        "CREATE TABLE issue_790_graphiti_retry_exclusions(event_id TEXT PRIMARY KEY)"
    )
    connection.execute(
        "INSERT INTO issue_790_graphiti_retry_exclusions(event_id) VALUES(?)",
        (held_event_id,),
    )
    connection.commit()
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: tuple(units),
    )
    preparation = _packet(proving, unpublished, authority_store=authority)
    campaign = build_operational_campaign_input(
        head_sha="head",
        tree_sha="tree",
        focus_manifest_digest="sha256:" + "f" * 64,
        graph_destination_id=GRAPH_DESTINATION_ID,
        candidate_event_count=campaign_event_count,
        recovery_identity="sha256:" + "a" * 64,
    )
    if mutate_campaign is not None:
        mutate_campaign(campaign)
    packet = _packet(
        proving,
        unpublished,
        authority_store=authority,
        campaign_input=campaign,
        graph_destination_reconciliation=_graph_destination_reconciliation(
            preparation
        ),
        governed_runtime=_governed_runtime(preparation),
    )
    return packet, campaign


def test_narrowing_preserves_stricter_intermediate_checkpoint_and_later_phase_conditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extra_entry = "PHASE_SPECIFIC_CHECKPOINT"

    def mutate(campaign: dict[str, object]) -> None:
        campaign["ramp"] = {
            "phases": [
                {
                    "phase_id": "phase-1",
                    "event_limit": 1,
                    "entry_conditions": [
                        "EXACT_SNAPSHOT_AND_IDENTITY_RECONFIRMED",
                        "OWNER_F4_GO_RETAINED",
                    ],
                    "advance_conditions": [
                        "ALL_EXACT_RECEIPTS_RECONCILED",
                        "CAPS_AND_ACCOUNTING_RECONCILED",
                        "NO_STOP_CONDITION_OBSERVED",
                    ],
                },
                {
                    "phase_id": "phase-2",
                    "event_limit": 2,
                    "entry_conditions": [
                        "EXACT_SNAPSHOT_AND_IDENTITY_RECONFIRMED",
                        "OWNER_F4_GO_RETAINED",
                    ],
                    "advance_conditions": [
                        "ALL_EXACT_RECEIPTS_RECONCILED",
                        "CAPS_AND_ACCOUNTING_RECONCILED",
                        "NO_STOP_CONDITION_OBSERVED",
                    ],
                },
                {
                    "phase_id": "phase-3",
                    "event_limit": 20,
                    "entry_conditions": [
                        "EXACT_SNAPSHOT_AND_IDENTITY_RECONFIRMED",
                        "OWNER_F4_GO_RETAINED",
                        extra_entry,
                    ],
                    "advance_conditions": [
                        "ALL_EXACT_RECEIPTS_RECONCILED",
                        "CAPS_AND_ACCOUNTING_RECONCILED",
                        "NO_STOP_CONDITION_OBSERVED",
                    ],
                },
            ]
        }

    packet, campaign = _retry_held_overcount_packet(
        tmp_path,
        monkeypatch,
        selected_event_count=15,
        campaign_event_count=20,
        mutate_campaign=mutate,
    )
    phases = packet["bounded_campaign"]["ramp"]["phases"]
    limits = [phase["event_limit"] for phase in phases]

    assert packet["verdict"] == "READY_FOR_OWNER_DECISION"
    assert packet["blockers"] == []
    assert limits == [1, 2, 15]
    assert extra_entry in phases[-1]["entry_conditions"]
    assert campaign["ramp"]["phases"][1]["event_limit"] == 2
    assert campaign["ramp"]["phases"][-1]["event_limit"] == 20
    assert campaign["caps"]["total"]["events"] == 20
    assert validate_graphiti_campaign_packet(packet) == packet["bounded_campaign"]


def test_narrowing_preserves_default_213_to_212_ramp_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet, campaign = _retry_held_overcount_packet(
        tmp_path,
        monkeypatch,
        selected_event_count=212,
        campaign_event_count=213,
    )
    limits = [
        phase["event_limit"]
        for phase in packet["bounded_campaign"]["ramp"]["phases"]
    ]

    assert packet["verdict"] == "READY_FOR_OWNER_DECISION"
    assert packet["blockers"] == []
    assert [
        phase["event_limit"] for phase in campaign["ramp"]["phases"]
    ] == [1, 10, 213]
    assert limits == [1, 10, 212]
    assert campaign["caps"]["total"]["events"] == 213
    assert packet["bounded_campaign"]["caps"]["total"]["events"] == 212
    assert validate_graphiti_campaign_packet(packet) == packet["bounded_campaign"]


def test_narrowing_one_event_collapse_retains_later_phase_conditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extra_advance = "PHASE_SPECIFIC_ADVANCE"

    def mutate(campaign: dict[str, object]) -> None:
        campaign["ramp"]["phases"][-1]["advance_conditions"] = sorted(
            [
                "ALL_EXACT_RECEIPTS_RECONCILED",
                "CAPS_AND_ACCOUNTING_RECONCILED",
                extra_advance,
                "NO_STOP_CONDITION_OBSERVED",
            ]
        )

    packet, campaign = _retry_held_overcount_packet(
        tmp_path,
        monkeypatch,
        selected_event_count=1,
        campaign_event_count=2,
        mutate_campaign=mutate,
    )
    phases = packet["bounded_campaign"]["ramp"]["phases"]

    assert packet["verdict"] == "READY_FOR_OWNER_DECISION"
    assert packet["blockers"] == []
    assert [phase["event_limit"] for phase in phases] == [1]
    assert extra_advance in phases[0]["advance_conditions"]
    assert extra_advance in campaign["ramp"]["phases"][-1]["advance_conditions"]
    assert campaign["caps"]["total"]["events"] == 2
    assert validate_graphiti_campaign_packet(packet) == packet["bounded_campaign"]


def test_narrowing_does_not_heal_empty_or_invalid_ramp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_packet, empty_campaign = _retry_held_overcount_packet(
        tmp_path / "empty-ramp",
        monkeypatch,
        selected_event_count=2,
        campaign_event_count=3,
        mutate_campaign=lambda campaign: campaign["ramp"].update({"phases": []}),
    )

    assert empty_packet["verdict"] == "NO_GO"
    assert "RAMP_PHASES_INVALID" in empty_packet["blockers"]
    assert empty_packet["bounded_campaign"]["ramp"]["phases"] == []
    assert empty_campaign["ramp"]["phases"] == []
    assert empty_campaign["caps"]["total"]["events"] == 3

    def invalidate_later_phase(campaign: dict[str, object]) -> None:
        campaign["ramp"]["phases"][-1]["entry_conditions"] = ["OWNER_F4_GO_RETAINED"]
        campaign["ramp"]["phases"][-1]["advance_conditions"] = ["not-a-contract"]

    invalid_packet, invalid_campaign = _retry_held_overcount_packet(
        tmp_path / "invalid-ramp",
        monkeypatch,
        selected_event_count=2,
        campaign_event_count=3,
        mutate_campaign=invalidate_later_phase,
    )
    invalid_phases = invalid_packet["bounded_campaign"]["ramp"]["phases"]

    assert invalid_packet["verdict"] == "NO_GO"
    assert "RAMP_PHASE_2_ENTRY_INVALID" in invalid_packet["blockers"]
    assert "RAMP_PHASE_2_ADVANCE_INVALID" in invalid_packet["blockers"]
    assert [phase["event_limit"] for phase in invalid_phases] != [1, 10, 2]
    assert invalid_campaign["ramp"]["phases"][-1]["entry_conditions"] == [
        "OWNER_F4_GO_RETAINED"
    ]


def test_narrowing_does_not_heal_invalid_remaining_on_collapse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalidate_later_phase(campaign: dict[str, object]) -> None:
        campaign["ramp"]["phases"][-1]["entry_conditions"] = ["OWNER_F4_GO_RETAINED"]
        campaign["ramp"]["phases"][-1]["advance_conditions"] = ["not-a-contract"]

    packet, campaign = _retry_held_overcount_packet(
        tmp_path,
        monkeypatch,
        selected_event_count=1,
        campaign_event_count=2,
        mutate_campaign=invalidate_later_phase,
    )
    phases = packet["bounded_campaign"]["ramp"]["phases"]

    assert packet["verdict"] == "NO_GO"
    assert "RAMP_PHASE_2_ENTRY_INVALID" in packet["blockers"]
    assert "RAMP_PHASE_2_ADVANCE_INVALID" in packet["blockers"]
    assert [phase["event_limit"] for phase in phases] == [1, 2]
    assert campaign["caps"]["total"]["events"] == 2
    assert campaign["ramp"]["phases"][-1]["advance_conditions"] == ["not-a-contract"]


def test_narrowing_preserves_duplicate_boundary_later_phase_conditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extra_entry = "PHASE_SPECIFIC_CHECKPOINT"

    def mutate(campaign: dict[str, object]) -> None:
        campaign["ramp"]["phases"][-1]["entry_conditions"] = [
            "EXACT_SNAPSHOT_AND_IDENTITY_RECONFIRMED",
            "OWNER_F4_GO_RETAINED",
            extra_entry,
        ]

    packet, campaign = _retry_held_overcount_packet(
        tmp_path,
        monkeypatch,
        selected_event_count=10,
        campaign_event_count=20,
        mutate_campaign=mutate,
    )
    phases = packet["bounded_campaign"]["ramp"]["phases"]

    assert packet["verdict"] == "READY_FOR_OWNER_DECISION"
    assert packet["blockers"] == []
    assert [phase["event_limit"] for phase in phases] == [1, 10]
    assert extra_entry in phases[-1]["entry_conditions"]
    assert extra_entry in campaign["ramp"]["phases"][-1]["entry_conditions"]
    assert [
        phase["event_limit"] for phase in campaign["ramp"]["phases"]
    ] == [1, 10, 20]
    assert validate_graphiti_campaign_packet(packet) == packet["bounded_campaign"]


@pytest.mark.parametrize("invalid_phase_id", ["", " ", 1, None])
def test_narrowing_keeps_invalid_discarded_phase_id_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_phase_id: object,
) -> None:
    def mutate(campaign: dict[str, object]) -> None:
        if invalid_phase_id is None:
            del campaign["ramp"]["phases"][-1]["phase_id"]
        else:
            campaign["ramp"]["phases"][-1]["phase_id"] = invalid_phase_id

    packet, campaign = _retry_held_overcount_packet(
        tmp_path,
        monkeypatch,
        selected_event_count=1,
        campaign_event_count=2,
        mutate_campaign=mutate,
    )
    phases = packet["bounded_campaign"]["ramp"]["phases"]

    assert packet["verdict"] == "NO_GO"
    assert "RAMP_PHASE_2_ID_INVALID" in packet["blockers"]
    assert [phase["event_limit"] for phase in phases] == [1, 2]
    assert len(campaign["ramp"]["phases"]) == 2
    if invalid_phase_id is None:
        assert "phase_id" not in campaign["ramp"]["phases"][-1]
    else:
        assert campaign["ramp"]["phases"][-1]["phase_id"] == invalid_phase_id
    assert campaign["caps"]["total"]["events"] == 2


def test_narrowing_keeps_original_final_limit_mismatch_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(campaign: dict[str, object]) -> None:
        campaign["ramp"]["phases"][-1]["event_limit"] = 20

    packet, campaign = _retry_held_overcount_packet(
        tmp_path,
        monkeypatch,
        selected_event_count=2,
        campaign_event_count=3,
        mutate_campaign=mutate,
    )
    phases = packet["bounded_campaign"]["ramp"]["phases"]

    assert packet["verdict"] == "NO_GO"
    assert "RAMP_FINAL_PHASE_DIFFERS_FROM_EVENT_CAP" in packet["blockers"]
    assert [phase["event_limit"] for phase in phases] != [1, 2]
    assert [phase["event_limit"] for phase in campaign["ramp"]["phases"]] == [1, 20]
    assert campaign["caps"]["total"]["events"] == 3
    assert packet["bounded_campaign"]["caps"]["total"]["events"] == 3


def test_narrowing_keeps_original_cap_mismatch_when_final_equals_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(campaign: dict[str, object]) -> None:
        campaign["ramp"]["phases"][-1]["event_limit"] = 2

    packet, campaign = _retry_held_overcount_packet(
        tmp_path,
        monkeypatch,
        selected_event_count=2,
        campaign_event_count=3,
        mutate_campaign=mutate,
    )
    phases = packet["bounded_campaign"]["ramp"]["phases"]

    assert packet["verdict"] == "NO_GO"
    assert "RAMP_FINAL_PHASE_DIFFERS_FROM_EVENT_CAP" in packet["blockers"]
    assert [phase["event_limit"] for phase in phases] == [1, 2]
    assert campaign["caps"]["total"]["events"] == 3
    assert packet["bounded_campaign"]["caps"]["total"]["events"] == 3
    assert [phase["event_limit"] for phase in campaign["ramp"]["phases"]] == [1, 2]


def test_undercounted_campaign_cap_does_not_widen_to_selected_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate-a",
        ingest_id="candidate-a-ingest",
    )
    _nonterminal_obligation(
        connection,
        ledger_seq=2,
        item_key="candidate-b",
        ingest_id="candidate-b-ingest",
    )
    _nonterminal_obligation(
        connection,
        ledger_seq=3,
        item_key="retry-held",
        ingest_id="retry-held-ingest",
    )
    connection.execute(
        "UPDATE unpublished_graphiti_revision_events SET state='RETRY_HELD',"
        "attempt_count=1 WHERE ledger_seq=3"
    )
    connection.commit()
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(
                item_key="candidate-a", ingest_id="candidate-a-ingest"
            ),
            _current_unit(
                item_key="candidate-b", ingest_id="candidate-b-ingest"
            ),
        ),
    )
    preparation = _packet(proving, unpublished, authority_store=authority)
    campaign = _campaign_input(preparation)
    campaign["caps"]["total"]["events"] = 1
    campaign["ramp"]["phases"][-1]["event_limit"] = 1
    packet = _packet(
        proving,
        unpublished,
        authority_store=authority,
        campaign_input=campaign,
        graph_destination_reconciliation=_graph_destination_reconciliation(
            preparation
        ),
        governed_runtime=_governed_runtime(preparation),
    )

    assert packet["verdict"] == "NO_GO"
    assert "TOTAL_EVENT_CAP_DIFFERS_FROM_SELECTED_COHORT" in packet["blockers"]
    assert packet["bounded_campaign"]["caps"]["total"]["events"] == 1
    assert len(packet["bounded_campaign"]["cohort"]["event_ids"]) == 2


def test_admission_backlog_blocks_owner_decision_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    _seed_admission_obligation(connection, action=None)
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)

    packet = _bounded_candidate_packet(
        proving,
        unpublished,
        authority,
        monkeypatch,
    )

    assert packet["verdict"] == "NO_GO"
    assert packet["admission"]["proposal_denominator"] == 1
    assert packet["admission"]["admission_backlog"] == 1
    assert packet["admission"]["projection_reconciled"] is False
    assert "ADMISSION_BACKLOG_PRESENT" in packet["blockers"]
    assert "ADMISSION_PROJECTION_UNRECONCILED" in packet["blockers"]
    assert "ADMISSION_DEAD_LETTER_PRESENT" not in packet["blockers"]
    assert "ADMISSION_INTEGRITY_HOLD_PRESENT" not in packet["blockers"]


def test_all_hold_admission_rejects_unbound_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    _seed_admission_obligation(connection, action="HOLD")
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)

    unreconciled = _bounded_candidate_packet(
        proving,
        unpublished,
        authority,
        monkeypatch,
    )

    assert unreconciled["verdict"] == "NO_GO"
    assert unreconciled["admission"]["proposal_denominator"] == 1
    assert unreconciled["admission"]["admitted_count"] == 0
    assert unreconciled["admission"]["held_count"] == 1
    assert unreconciled["admission"]["admission_backlog"] == 0
    assert unreconciled["admission"]["projection_reconciled"] is False
    assert "ADMISSION_PROJECTION_UNRECONCILED" in unreconciled["blockers"]
    assert "ADMISSION_BACKLOG_PRESENT" not in unreconciled["blockers"]

    connection = sqlite3.connect(unpublished)
    receipt = GraphitiProjectionReconciliationReceipt(
        generation_id="00000000-0000-4000-8000-000000000895",
        expected_effect_ids=(),
        actual_effect_ids=(),
        authority_watermark=1,
        receipt_digest="sha256:" + "5" * 64,
    ).canonical_value()
    connection.execute(
        "INSERT INTO unpublished_graphiti_projection_reconciliations("
        "receipt_digest,projector_family_id,generation_id,"
        "authority_watermark,receipt_json,reconciled_at) VALUES(?,?,?,?,?,?)",
        (
            receipt["receipt_digest"],
            receipt["projector_family_id"],
            receipt["generation_id"],
            1,
            canonical_json_bytes(receipt).decode("utf-8"),
            "2026-09-01T12:00:00.000000Z",
        ),
    )
    connection.commit()
    connection.close()

    unbound = _bounded_candidate_packet(
        proving,
        unpublished,
        authority,
        monkeypatch,
    )

    assert unbound["verdict"] == "NO_GO"
    assert unbound["admission"]["telemetry_projection_reconciled"] is True
    assert unbound["admission"]["projection_reconciled"] is False
    assert unbound["admission"]["exact_cohort_reconciliation"]["total"] is False
    assert "ADMISSION_PROJECTION_UNRECONCILED" in unbound["blockers"]


def test_orphan_reconciliation_blocks_without_a_proposal_denominator(
    tmp_path: Path,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    receipt = GraphitiProjectionReconciliationReceipt(
        generation_id="00000000-0000-4000-8000-000000000895",
        expected_effect_ids=(),
        actual_effect_ids=(),
        authority_watermark=1,
        receipt_digest="sha256:" + "5" * 64,
    ).canonical_value()
    connection.execute(
        "INSERT INTO unpublished_graphiti_projection_reconciliations("
        "receipt_digest,projector_family_id,generation_id,"
        "authority_watermark,receipt_json,reconciled_at) VALUES(?,?,?,?,?,?)",
        (
            receipt["receipt_digest"],
            receipt["projector_family_id"],
            receipt["generation_id"],
            1,
            canonical_json_bytes(receipt).decode("utf-8"),
            "2026-09-01T12:00:00.000000Z",
        ),
    )
    connection.commit()
    connection.close()

    packet = _packet(proving, unpublished)

    assert packet["admission"]["proposal_denominator"] == 0
    assert packet["admission"]["exact_cohort_reconciliation"]["total"] is False
    assert "ADMISSION_PROJECTION_UNRECONCILED" in packet["blockers"]


def test_exact_all_hold_generation_reconciliation_is_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    _seed_admission_obligation(connection, action="HOLD")
    cohort_digest = "sha256:" + "6" * 64
    generation_id = str(
        typed_id(
            ProjectionGenerationId,
            GRAPHITI_ADMISSION_GENERATION_IDENTITY_VERSION,
            cohort_digest,
        )
    )
    raw_receipt = GraphitiProjectionReconciliationReceipt(
        generation_id=generation_id,
        expected_effect_ids=(),
        actual_effect_ids=(),
        authority_watermark=1,
        receipt_digest="sha256:" + "7" * 64,
    )
    binding = {
        "schema_version": GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION,
        "cohort_digest": cohort_digest,
        "ingest_ids": ["proposal-ingest"],
        "raw_receipt": raw_receipt.canonical_value(),
    }
    connection.execute(
        "INSERT INTO unpublished_graphiti_projection_reconciliations("
        "receipt_digest,projector_family_id,generation_id,"
        "authority_watermark,receipt_json,reconciled_at) VALUES(?,?,?,?,?,?)",
        (
            raw_receipt.receipt_digest,
            raw_receipt.projector_family_id,
            raw_receipt.generation_id,
            raw_receipt.authority_watermark,
            canonical_json_bytes(binding).decode("utf-8"),
            "2026-09-01T12:00:00.000000Z",
        ),
    )
    connection.commit()
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)
    def exact_identity(
        exact_connection: sqlite3.Connection,
        *,
        ingest_ids: tuple[str, ...],
        require_terminal_states: bool,
    ) -> tuple[str, str]:
        assert ingest_ids == ("proposal-ingest",)
        assert require_terminal_states is True
        state = exact_connection.execute(
            "SELECT state FROM unpublished_graphiti_admission_queue "
            "WHERE proposal_key='proposal-1'"
        ).fetchone()[0]
        if state != "TERMINAL":
            raise GraphitiAdmissionConsumerError(
                "exact generation decision integrity differs"
            )
        return cohort_digest, generation_id

    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "graphiti_decided_cohort_generation_identity",
        exact_identity,
    )

    mismatched = _bounded_candidate_packet(
        proving,
        unpublished,
        authority,
        monkeypatch,
    )

    assert mismatched["verdict"] == "NO_GO"
    assert mismatched["admission"]["exact_cohort_reconciliation"][
        "latest_generation_id"
    ] == generation_id
    assert "ADMISSION_ACTIVE_GENERATION_DRIFT" in mismatched["blockers"]

    authority_connection = sqlite3.connect(authority)
    authority_connection.execute(
        "UPDATE projection_generations SET generation_id=? WHERE state='ACTIVE'",
        (generation_id,),
    )
    authority_connection.commit()
    authority_connection.close()

    packet = _bounded_candidate_packet(
        proving,
        unpublished,
        authority,
        monkeypatch,
    )

    assert packet["verdict"] == "READY_FOR_OWNER_DECISION"
    assert packet["admission"]["projection_reconciled"] is True
    assert packet["admission"]["exact_cohort_reconciliation"]["total"] is True
    assert packet["admission"]["exact_cohort_reconciliation"]["disjoint"] is True
    assert packet["blockers"] == []
    assert validate_graphiti_campaign_packet(packet) == packet["bounded_campaign"]

    generation_drift = json.loads(json.dumps(packet))
    generation_drift["admission"]["exact_cohort_reconciliation"][
        "latest_generation_id"
    ] = "00000000-0000-4000-8000-000000000895"
    generation_drift["packet_digest"] = digest_canonical(
        {
            key: value
            for key, value in generation_drift.items()
            if key != "packet_digest"
        }
    )
    with pytest.raises(ValueError, match="active generation differs"):
        validate_graphiti_campaign_packet(generation_drift)

    connection = sqlite3.connect(unpublished)
    connection.execute(
        "UPDATE unpublished_graphiti_admission_queue SET state='READY' "
        "WHERE proposal_key='proposal-1'"
    )
    connection.commit()
    connection.close()

    corrupt_state = _bounded_candidate_packet(
        proving,
        unpublished,
        authority,
        monkeypatch,
    )

    assert corrupt_state["admission"]["admission_backlog"] == 0
    assert corrupt_state["admission"]["telemetry_projection_reconciled"] is True
    assert corrupt_state["admission"]["projection_reconciled"] is False
    assert corrupt_state["verdict"] == "NO_GO"
    assert "ADMISSION_PROJECTION_UNRECONCILED" in corrupt_state["blockers"]


def test_active_extension_inventory_counts_cohorts_not_generation_ids(
    tmp_path: Path,
) -> None:
    connection = connect(str(tmp_path / "active-extension-inventory.sqlite3"))
    first_id = "00000000-0000-4000-8000-0000000075f1"
    second_id = "00000000-0000-4000-8000-0000000075f2"
    first = _admission_draft(
        "entity.0001", ExtractionProposalKind.ENTITY_MENTION
    )
    second = _admission_draft(
        "entity.0002", ExtractionProposalKind.ENTITY_MENTION, subject="Bob"
    )
    _seed_admission_receipt(connection, first, ingest_id=first_id)
    _seed_admission_receipt(connection, second, ingest_id=second_id)
    active_generation_id = "00000000-0000-4000-8000-000000007599"

    class StableProjector(_AdmissionProjector):
        def build_and_promote_increment4_cohort(
            self, requests, *, cohort_digest, generation_id, idempotency_key
        ):
            effect_ids = tuple(
                sorted(
                    str(item.decision.admitted_authority_id)
                    for item in requests
                    if item.decision.action
                    is GraphitiProposalAdmissionAction.ADMIT
                )
            )
            return self.generation_effects.setdefault(
                idempotency_key,
                GraphitiProjectionGenerationResult(
                    cohort_digest=cohort_digest,
                    generation_id=active_generation_id,
                    source_snapshot_digest=digest_canonical(
                        {"snapshot": cohort_digest}
                    ),
                    authority_watermark=max(
                        item.decision.authority_ledger_seq for item in requests
                    ),
                    validation_digest=digest_canonical(
                        {"validation": cohort_digest}
                    ),
                    promotion_digest="sha256:" + "b2" * 32,
                    reconciliation_digest=digest_canonical(
                        {"projection": cohort_digest}
                    ),
                    admitted_authority_ids=effect_ids,
                    schema_version=GRAPHITI_PROJECTION_EXTENSION_SCHEMA_VERSION,
                    source_request_digest=digest_canonical(
                        {
                            "generation_id": generation_id,
                            "reason_code": "GRAPHITI_ADMISSION_COHORT",
                            "idempotency_key": idempotency_key,
                            "purge_retired_generation": True,
                            "allow_active_extension": True,
                        }
                    ),
                ),
            )

    consumer = _admission_consumer(
        connection,
        _AdmissionAuthority(
            {
                first.local_id: GraphitiProposalAdmissionAction.ADMIT,
                second.local_id: GraphitiProposalAdmissionAction.ADMIT,
            }
        ),
        StableProjector(),
        _AdmissionRights(),
    )
    for ingest_id in (first_id, second_id):
        consumer.enqueue_complete_receipts(ingest_ids=(ingest_id,))
        consumer.drain(
            worker_id="worker-a", limit=1, ingest_ids=(ingest_id,)
        )
        consumer.finalise_decided_cohort(ingest_ids=(ingest_id,))

    inventory = _exact_admission_reconciliation(connection)
    assert inventory["cohort_count"] == 2
    assert inventory["covered_ingest_count"] == 2
    assert inventory["queue_ingest_count"] == 2
    assert inventory["latest_generation_id"] == active_generation_id
    assert [item["effect_count"] for item in inventory["cohorts"]] == [1, 1]
    assert inventory["total"] is True
    assert inventory["disjoint"] is True
    connection.close()


def test_untyped_caller_graph_readback_is_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    connection.commit()
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(item_key="candidate", ingest_id="candidate-ingest"),
        ),
    )
    preparation = _packet(proving, unpublished, authority_store=authority)
    packet = _packet(
        proving,
        unpublished,
        authority_store=authority,
        campaign_input=_campaign_input(preparation),
        graph_destination_reconciliation={  # type: ignore[arg-type]
            "destination_id": GRAPH_DESTINATION_ID,
            "generation_id": "00000000-0000-4000-8000-000000000895",
            "readback_digest": "sha256:" + "0" * 64,
        },
        governed_runtime=_governed_runtime(preparation),
    )

    assert packet["verdict"] == "NO_GO"
    assert "GRAPH_DESTINATION_READBACK_INVALID" in packet["blockers"]
    assert packet["runtime_composition"]["actual_graph_readback_observed"] is False


def test_runtime_authority_descriptor_drift_is_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    connection.commit()
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(item_key="candidate", ingest_id="candidate-ingest"),
        ),
    )
    preparation = _packet(proving, unpublished, authority_store=authority)
    runtime = _governed_runtime(preparation)
    drifted_runtime = _mint_graphiti_campaign_runtime(
        graphiti=runtime.graphiti,
        admission_factory=runtime.admission_factory,
        bind_unit_authority=runtime.bind_unit_authority,
        graph_state_fence=runtime.graph_state_fence,
        graph_destination_id=runtime.graph_destination_id,
        authority_store_source_path=runtime.authority_store_source_path,
        authority_store_descriptor_digest="sha256:" + "0" * 64,
    )

    packet = _packet(
        proving,
        unpublished,
        authority_store=authority,
        campaign_input=_campaign_input(preparation),
        graph_destination_reconciliation=(
            _graph_destination_reconciliation(preparation)
        ),
        governed_runtime=drifted_runtime,
    )

    assert packet["verdict"] == "NO_GO"
    assert "CANONICAL_OPERATOR_RUNTIME_UNCONFIGURED" in packet["blockers"]
    assert packet["runtime_composition"]["authority_store_descriptor_digest"] == (
        "sha256:" + "0" * 64
    )
    assert packet["runtime_composition"]["campaign_packet_enforced"] is False


def test_runtime_graph_destination_drift_is_deterministic_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    connection.commit()
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(item_key="candidate", ingest_id="candidate-ingest"),
        ),
    )
    preparation = _packet(proving, unpublished, authority_store=authority)
    campaign = _campaign_input(preparation)
    campaign["graph"] = {"destination_id": "sha256:" + "8" * 64}

    packet = _packet(
        proving,
        unpublished,
        authority_store=authority,
        campaign_input=campaign,
        graph_destination_reconciliation=(
            _graph_destination_reconciliation(preparation)
        ),
        governed_runtime=_governed_runtime(preparation),
    )

    assert packet["verdict"] == "NO_GO"
    assert "GRAPH_DESTINATION_RUNTIME_BINDING_INVALID" in packet["blockers"]
    assert packet["runtime_composition"]["campaign_packet_enforced"] is False
    assert packet["runtime_composition"]["graph_destination_id"] == (
        GRAPH_DESTINATION_ID
    )


def test_unminted_runtime_instance_cannot_seal_ready_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    connection.commit()
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(item_key="candidate", ingest_id="candidate-ingest"),
        ),
    )
    preparation = _packet(proving, unpublished, authority_store=authority)
    unminted_runtime = object.__new__(GraphitiCampaignRuntime)
    object.__setattr__(
        unminted_runtime,
        "_GraphitiCampaignRuntime__construction_token",
        object(),
    )

    packet = _packet(
        proving,
        unpublished,
        authority_store=authority,
        campaign_input=_campaign_input(preparation),
        graph_destination_reconciliation=(
            _graph_destination_reconciliation(preparation)
        ),
        governed_runtime=unminted_runtime,
    )

    assert packet["verdict"] == "NO_GO"
    assert "CANONICAL_OPERATOR_RUNTIME_UNCONFIGURED" in packet["blockers"]
    assert packet["runtime_composition"]["state"] == (
        "CANONICAL_OPERATOR_RUNTIME_UNCONFIGURED"
    )
    assert packet["runtime_composition"]["campaign_packet_enforced"] is False


@pytest.mark.parametrize(
    "reconciliation_overrides",
    (
        {"family_id": "graph.different"},
        {"generation_id": "00000000-0000-4000-8000-000000000000"},
        {"checkpoint_ledger_seq": 8},
        {"serving_time": "2026-09-01T13:00:00Z"},
    ),
)
def test_graph_reconciliation_identity_drift_is_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reconciliation_overrides: dict[str, object],
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    connection.commit()
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(item_key="candidate", ingest_id="candidate-ingest"),
        ),
    )
    preparation = _packet(proving, unpublished, authority_store=authority)
    reconciliation = _graph_destination_reconciliation(
        preparation,
        **reconciliation_overrides,  # type: ignore[arg-type]
    )

    packet = _packet(
        proving,
        unpublished,
        authority_store=authority,
        campaign_input=_campaign_input(preparation),
        graph_destination_reconciliation=reconciliation,
        governed_runtime=_governed_runtime(preparation),
    )

    assert packet["verdict"] == "NO_GO"
    assert "GRAPH_DESTINATION_READBACK_INVALID" in packet["blockers"]


def test_typed_graph_readback_without_operator_runtime_is_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    connection.commit()
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(item_key="candidate", ingest_id="candidate-ingest"),
        ),
    )
    preparation = _packet(proving, unpublished, authority_store=authority)
    packet = _packet(
        proving,
        unpublished,
        authority_store=authority,
        campaign_input=_campaign_input(preparation),
        graph_destination_reconciliation=(
            _graph_destination_reconciliation(preparation)
        ),
    )

    assert packet["verdict"] == "NO_GO"
    assert "CANONICAL_OPERATOR_RUNTIME_UNCONFIGURED" in packet["blockers"]
    assert packet["runtime_composition"] == {
        "state": "CANONICAL_OPERATOR_RUNTIME_UNCONFIGURED",
        "authority_store_configured": True,
        "graph_destination_id": None,
        "authority_store_source_path": None,
        "authority_store_descriptor_digest": None,
        "durable_proposal_envelope_binding": False,
        "admission_policy_configured": False,
        "full_generation_projector_configured": False,
        "dormant_worker_path_wired": False,
        "campaign_packet_enforced": False,
        "actual_graph_readback_observed": True,
        "campaign_authorised": False,
    }


def test_campaign_packet_validator_rejects_digest_drift(
    tmp_path: Path,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    connection.close()
    packet = _packet(proving, unpublished)
    packet["packet_digest"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match="canonical digest differs"):
        validate_graphiti_campaign_packet(packet)


def test_unverified_terminal_is_unclassified_in_historical_partition(
    tmp_path: Path,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _terminal_zero_proposal(connection)
    connection.execute(
        "UPDATE unpublished_graphiti_receipts SET receipt_json=? "
        "WHERE ingest_id='ingest-1'",
        (json.dumps({"raw_output_digest": "tampered", "proposals": []}),),
    )
    connection.commit()
    connection.close()

    partition = _packet(proving, unpublished)["historical_partition"]

    assert partition["categories"]["VERIFIED_TERMINAL"]["count"] == 0
    assert partition["categories"]["UNCLASSIFIED"]["ledger_sequences"] == [1]
    assert partition["categories"]["UNCLASSIFIED"]["reason_counts"] == {
        "TERMINAL_OUTCOME_UNVERIFIED": 1
    }


def test_historical_raw_only_proposal_is_immutable_non_blocking_hold(
    tmp_path: Path,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _terminal_zero_proposal(connection)
    _turn_terminal_receipt_into_historical_raw_only(connection)
    connection.close()

    packet = _packet(proving, unpublished)
    hold = packet["historical_partition"]["categories"][
        "NON_REPLAYABLE_OR_AMBIGUOUS_EFFECT_HOLD"
    ]

    assert hold["ledger_sequences"] == [1]
    assert hold["reason_counts"] == {
        "IMMUTABLE_RAW_PROPOSAL_WITHOUT_DURABLE_ENVELOPE": 1
    }
    assert "HISTORICAL_EFFECT_ADJUDICATION_REQUIRED" not in packet["blockers"]


def test_historical_proposal_is_verified_only_from_exact_4a_4d_snapshot(
    tmp_path: Path,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _terminal_zero_proposal(connection)
    _turn_terminal_receipt_into_historical_raw_only(connection)
    connection.close()
    authority = _authority_store(tmp_path)
    _bind_terminal_raw_proposal_to_authority(unpublished, authority)

    packet = _packet(proving, unpublished, authority_store=authority)

    assert packet["historical_partition"]["categories"]["VERIFIED_TERMINAL"][
        "ledger_sequences"
    ] == [1]
    assert packet["historical_partition"]["categories"][
        "NON_REPLAYABLE_OR_AMBIGUOUS_EFFECT_HOLD"
    ]["count"] == 0


def test_non_terminal_4d_partial_attempt_is_not_future_admission_authority(
    tmp_path: Path,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _terminal_zero_proposal(connection)
    _turn_terminal_receipt_into_historical_raw_only(connection)
    connection.close()
    authority = _authority_store(tmp_path)
    _bind_terminal_raw_proposal_to_authority(unpublished, authority)
    authority_connection = sqlite3.connect(authority)
    authority_connection.execute(
        "UPDATE graphiti_adapter_attempts SET outcome='PARTIAL'"
    )
    authority_connection.commit()
    authority_connection.close()

    packet = _packet(proving, unpublished, authority_store=authority)

    hold = packet["historical_partition"]["categories"][
        "NON_REPLAYABLE_OR_AMBIGUOUS_EFFECT_HOLD"
    ]
    assert hold["ledger_sequences"] == [1]
    assert hold["reason_counts"] == {
        "IMMUTABLE_RAW_PROPOSAL_WITHOUT_DURABLE_ENVELOPE": 1
    }


def test_internally_valid_raw_output_drift_keeps_historical_proposal_on_hold(
    tmp_path: Path,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _terminal_zero_proposal(connection)
    _turn_terminal_receipt_into_historical_raw_only(connection)
    connection.close()
    authority = _authority_store(tmp_path)
    _bind_terminal_raw_proposal_to_authority(unpublished, authority)
    authority_connection = sqlite3.connect(authority)
    row = authority_connection.execute(
        "SELECT canonical_bytes FROM extraction_outputs WHERE output_id='output'"
    ).fetchone()
    assert row is not None
    raw = json.loads(bytes(row[0]))
    raw.pop("raw_output_digest")
    raw["proposals"][0]["local_id"] = "different-retained-proposal"
    raw["raw_output_digest"] = digest_bytes(canonical_json_bytes(raw))
    raw_bytes = canonical_json_bytes(raw)
    authority_connection.execute(
        "UPDATE extraction_outputs SET canonical_bytes=?,canonical_digest=? "
        "WHERE output_id='output'",
        (raw_bytes, digest_bytes(raw_bytes)),
    )
    authority_connection.commit()
    authority_connection.close()

    packet = _packet(proving, unpublished, authority_store=authority)

    hold = packet["historical_partition"]["categories"][
        "NON_REPLAYABLE_OR_AMBIGUOUS_EFFECT_HOLD"
    ]
    assert hold["ledger_sequences"] == [1]
    assert hold["reason_counts"] == {
        "IMMUTABLE_RAW_PROPOSAL_WITHOUT_DURABLE_ENVELOPE": 1
    }


def test_forged_admission_queue_does_not_create_durable_4a_4d_authority(
    tmp_path: Path,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _terminal_zero_proposal(connection)
    _turn_terminal_receipt_into_historical_raw_only(connection)
    receipt_digest = connection.execute(
        "SELECT receipt_digest FROM unpublished_graphiti_ingest "
        "WHERE ingest_id='ingest-1'"
    ).fetchone()[0]
    forged_request = {
        "proposal_authority_binding": {
            "proposal_envelope": {"local_id": "historical-raw-only"}
        }
    }
    connection.execute(
        "INSERT INTO unpublished_graphiti_admission_queue("
        "proposal_key,ingest_id,source_revision_id,source_receipt_digest,"
        "proposal_digest,proposal_kind,request_json,request_digest,state,"
        "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            "forged-key",
            "ingest-1",
            "revision",
            str(receipt_digest),
            "sha256:" + "1" * 64,
            "ENTITY_MENTION",
            json.dumps(forged_request, sort_keys=True),
            digest_canonical(forged_request),
            "READY",
            NOW.isoformat(),
            NOW.isoformat(),
        ),
    )
    connection.commit()
    connection.close()
    authority = _authority_store(tmp_path)

    packet = _packet(proving, unpublished, authority_store=authority)

    hold = packet["historical_partition"]["categories"][
        "NON_REPLAYABLE_OR_AMBIGUOUS_EFFECT_HOLD"
    ]
    assert hold["reason_counts"] == {
        "IMMUTABLE_RAW_PROPOSAL_WITHOUT_DURABLE_ENVELOPE": 1
    }


def test_arbitrary_migration_checksum_and_owner_supplied_generation_are_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    connection.commit()
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(item_key="candidate", ingest_id="candidate-ingest"),
        ),
    )
    preparation = _packet(proving, unpublished, authority_store=authority)
    campaign = _campaign_input(preparation)
    campaign["graph"]["current_generation_id"] = "self-addressed-generation"
    authority_connection = sqlite3.connect(authority)
    authority_connection.execute(
        "UPDATE authority_migrations SET checksum=? WHERE version=16",
        ("sha256:" + "0" * 64,),
    )
    authority_connection.commit()
    authority_connection.close()

    packet = _packet(
        proving,
        unpublished,
        authority_store=authority,
        campaign_input=campaign,
    )

    assert "AUTHORITY_MIGRATION_HISTORY_INVALID" in packet["blockers"]
    assert "GRAPH_IDENTITY_FIELDS_INVALID" in packet["blockers"]


def test_fallback_stop_and_ramp_contracts_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    connection.commit()
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(tmp_path)
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda _connection, *, evaluated_at: (
            _current_unit(item_key="candidate", ingest_id="candidate-ingest"),
        ),
    )
    preparation = _packet(proving, unpublished, authority_store=authority)
    campaign = _campaign_input(preparation)
    campaign["caps"]["per_event"]["fallbacks"] = 1
    campaign["immediate_stop_conditions"].remove("EXACT_RECEIPT_DRIFT")
    phase = campaign["ramp"]["phases"][0]
    phase["entry_conditions"] = ["OWNER_F4_GO_RETAINED", "OWNER_F4_GO_RETAINED"]

    packet = _packet(
        proving,
        unpublished,
        authority_store=authority,
        campaign_input=campaign,
    )

    assert "FALLBACK_CAP_MUST_BE_ZERO" in packet["blockers"]
    assert "IMMEDIATE_STOP_CONDITIONS_INCOMPLETE" in packet["blockers"]
    assert "RAMP_PHASE_1_ENTRY_INVALID" in packet["blockers"]


def test_event_manifest_rejects_resolved_landed_identity_mismatch(
    tmp_path: Path,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _terminal_zero_proposal(connection)
    row = connection.execute(
        "SELECT manifest_json FROM unpublished_graphiti_revision_events"
    ).fetchone()
    assert row is not None
    manifest = json.loads(str(row[0]))
    manifest["landed_ingest_ids"] = ["different-ingest"]
    connection.execute(
        "UPDATE unpublished_graphiti_revision_events "
        "SET manifest_json=?,manifest_digest=?",
        (json.dumps(manifest, sort_keys=True), digest_canonical(manifest)),
    )
    connection.commit()
    connection.close()

    packet = _packet(proving, unpublished)

    assert "EVENT_MANIFEST_INTEGRITY_FAILURE" in packet["blockers"]
    assert packet["events"]["integrity_failures"] == [
        {
            "event_id": "sha256:" + "1" * 64,
            "reason": "MALFORMED_EVENT_MANIFEST",
        }
    ]


def test_explicit_source_hold_is_not_counted_as_successful_observation(
    tmp_path: Path,
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    connection.close()
    _seed_proving_accountability(proving, held_source_id=SOURCE_IDS[0])

    packet = _packet(proving, unpublished)
    accountability = packet["proving_accountability"]

    assert accountability["successful_observation_count"] == len(SOURCE_IDS) - 1
    assert accountability["typed_hold_count"] == 1
    assert accountability["unaccounted_source_ids"] == []
    assert "PROVING_SOURCE_UNACCOUNTED" not in packet["blockers"]


def test_tampered_receipt_is_machine_readable_no_go(tmp_path: Path) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    _terminal_zero_proposal(connection)
    connection.execute(
        "UPDATE unpublished_graphiti_receipts SET receipt_json=? "
        "WHERE ingest_id='ingest-1'",
        (json.dumps({"raw_output_digest": "tampered", "proposals": []}),),
    )
    connection.commit()
    connection.close()

    packet = _packet(proving, unpublished)

    assert packet["verdict"] == "NO_GO"
    assert "TERMINAL_RECEIPT_INTEGRITY_FAILURE" in packet["blockers"]
    assert packet["terminal_receipts"]["integrity_failures"] == [
        {
            "ingest_id": "ingest-1",
            "reason": "RECEIPT_ENVELOPE_DIGEST_CONTRADICTION",
        }
    ]


def test_digest_is_stable_and_output_is_create_exclusive(tmp_path: Path) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    connection.close()
    first = _packet(proving, unpublished)
    second = _packet(proving, unpublished)

    assert first == second
    output = write_content_addressed_packet(first, tmp_path / "packets")
    assert output.name.endswith(
        f"{str(first['packet_digest']).removeprefix('sha256:')}.json"
    )
    with pytest.raises(FileExistsError):
        write_content_addressed_packet(first, tmp_path / "packets")


def test_content_addressed_output_is_not_left_partial_when_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proving, unpublished, connection = _stores(tmp_path)
    connection.close()
    packet = _packet(proving, unpublished)
    output_dir = tmp_path / "packets"

    def fail_publish(_source, _destination) -> None:
        raise OSError("fixture publish failure")

    monkeypatch.setattr("newsroom.control_plane.graphiti_steady_state.os.link", fail_publish)
    with pytest.raises(OSError, match="fixture publish failure"):
        write_content_addressed_packet(packet, output_dir)

    assert list(output_dir.iterdir()) == []


def test_report_requires_clean_exact_main(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
        ("rev-parse", "HEAD"): "head",
        ("rev-parse", "origin/main"): "head",
        ("rev-parse", "HEAD^{tree}"): "tree",
    }
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "_git",
        lambda *args: values[args],
    )

    assert graphiti_steady_state_report._exact_main_identity() == ("head", "tree")

    values[("status", "--porcelain=v1", "--untracked-files=all")] = " M file"
    with pytest.raises(RuntimeError, match="clean worktree"):
        graphiti_steady_state_report._exact_main_identity()

    values[("status", "--porcelain=v1", "--untracked-files=all")] = ""
    values[("rev-parse", "origin/main")] = "other"
    with pytest.raises(RuntimeError, match="exact origin/main"):
        graphiti_steady_state_report._exact_main_identity()


def test_report_git_commands_are_anchored_to_executing_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class Result:
        stdout = "value\n"

    def run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return Result()

    monkeypatch.setattr(graphiti_steady_state_report.subprocess, "run", run)

    assert graphiti_steady_state_report._git("rev-parse", "HEAD") == "value"
    assert observed["cwd"] == graphiti_steady_state_report.REPOSITORY_ROOT


@pytest.mark.parametrize(
    ("verdict", "expected_exit"),
    (("READY_FOR_OWNER_DECISION", 0), ("NO_GO", 2)),
)
def test_report_exit_zero_only_for_owner_decision_ready(
    verdict: str,
    expected_exit: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "_exact_main_identity",
        lambda: ("head", "tree"),
    )
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "build_graphiti_steady_state_packet",
        lambda **_kwargs: {"verdict": verdict},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "graphiti_steady_state_report.py",
            "--proving",
            "proving.sqlite3",
            "--unpublished",
            "unpublished.sqlite3",
        ],
    )

    assert graphiti_steady_state_report.main() == expected_exit
    assert json.loads(capsys.readouterr().out) == {"verdict": verdict}


@pytest.mark.parametrize("corruption", ["missing", "non-blob", "mismatch"])
def test_proving_body_corruption_is_typed_failed_readiness(tmp_path, corruption):
    proving, unpublished, connection = _stores(tmp_path)
    connection.close()
    _seed_proving_accountability(proving)
    source_id = SOURCE_IDS[0]
    with sqlite3.connect(proving) as connection:
        digest = connection.execute(
            "SELECT body_digest FROM proving_observations WHERE source_id=?",
            (source_id,),
        ).fetchone()[0]
        if corruption == "missing":
            connection.execute("DELETE FROM proving_bodies WHERE body_digest=?", (digest,))
        else:
            connection.execute(
                "UPDATE proving_bodies SET body=? WHERE body_digest=?",
                ("text" if corruption == "non-blob" else b"wrong", digest),
            )
    packet = _packet(proving, unpublished)
    accountability = packet["proving_accountability"]
    assert accountability["unaccounted_source_ids"] == [source_id]
    assert "PROVING_SOURCE_UNACCOUNTED" in packet["blockers"]
    source = next(row for row in accountability["sources"] if row["source_id"] == source_id)
    assert source["body_digest_valid"] is False
    assert source["observation_success"] is False
