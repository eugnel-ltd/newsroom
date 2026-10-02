from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import newsroom.authority.migrations as authority_migrations
import newsroom.increment7.readiness as readiness_module
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority.migrations import apply_pending_migrations, schema_fingerprint
from newsroom.increment7 import (
    INCREMENT_7_READINESS,
    INCREMENT_7_READINESS_DIGEST,
    READINESS_CONTRACT_PATH,
    GateTier,
    Increment7ReadinessError,
    load_increment7_readiness_contract,
    validate_interface_inventory,
)

_BASE_COMMIT = "ddd77f7e96ebe8df42861631dc47005c30048662"
_BASE_TREE = "f5109a81962db2d4206426abfc152c890ef5d461"
_BASE_FINGERPRINT = (
    "sha256:353900bf5804f0b770489982541f3cff4fd30ea36fc75d19b9c63315d1b6ec06"
)
# Historical readiness anchors stay immutable. Current implementations are
# verified by the exact commit's Focus Gate and review, not duplicate hash pins
# that require editing this unrelated readiness test after every code change.
_NATIVE_SUCCESSORS = frozenset({
    "newsroom.increment6.candidates", "newsroom.increment6.work_items",
})


def _assert_reviewed_native_successor_inventory() -> None:
    assert validate_interface_inventory(INCREMENT_7_READINESS) == tuple(
        f"{name}: module source differs" for name in sorted(_NATIVE_SUCCESSORS)
    )


_CHILD_ISSUES = tuple(range(435, 447))
_MIGRATIONS = {437: 26, 439: 27, 443: 28, 445: 29}
_WAVES = {
    0: (435,),
    1: (436, 438, 440),
    2: (437, 439, 441, 442),
    3: (443,),
    4: (444,),
    5: (445,),
    6: (446,),
}


def _record() -> dict[str, object]:
    value = json.loads(READINESS_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_readiness_is_bound_to_exact_final_increment6_head_and_v25_schema() -> None:
    readiness = INCREMENT_7_READINESS
    assert readiness.issue_number == 435
    assert readiness.parent_issue_number == 147
    assert readiness.accepted_base_commit == _BASE_COMMIT
    assert readiness.accepted_base_tree == _BASE_TREE
    assert readiness.accepted_schema_version == 25
    assert readiness.accepted_schema_fingerprint == _BASE_FINGERPRINT
    assert (
        readiness.accepted_migration_history
        == (authority_migrations.EXPECTED_MIGRATION_HISTORY[:25])
    )
    assert readiness.effective_when == "PRESENT_ON_MAIN_AFTER_REVIEWED_7R_MERGE"
    assert readiness.production_activation_authorised is False
    assert readiness.contract_digest == INCREMENT_7_READINESS_DIGEST
    assert readiness.contract_digest == digest_bytes(
        READINESS_CONTRACT_PATH.read_bytes()
    )


def test_inherited_interfaces_preserve_historical_anchors_and_checked_native_successors() -> None:
    readiness = INCREMENT_7_READINESS
    assert tuple(item.interface_id for item in readiness.interface_inventory) == (
        "increment6.final-closeout",
        "increment6.supplemental-discovery-reentry",
        "increment6.candidate-authority",
        "increment6.handoff-authority",
        "accepted.agenda-observation-semantics",
        "accepted.agenda-miss-guard",
        "qualification.fixture-profile",
        "qualification.evaluation-plan",
        "authority.checked-schema",
    )
    _assert_reviewed_native_successor_inventory()
    assert readiness.interface_inventory[1].symbol_values == {
        "SUPPLEMENTAL_DISCOVERY_REENTRY": "NEW_GOVERNED_DISCOVERY_LINEAGE_ONLY"
    }


def test_all_atomic_owners_are_disjoint_and_transaction_ports_are_explicit() -> None:
    readiness = INCREMENT_7_READINESS
    assert tuple(item.issue_number for item in readiness.allocations) == _CHILD_ISSUES
    for attribute in (
        "public_modules",
        "schema_ids",
        "table_names",
        "interface_ownership",
    ):
        values = [
            value
            for item in readiness.allocations
            for value in getattr(item, attribute)
        ]
        assert len(values) == len(set(values))
    assert readiness.allocation_by_issue[437].public_modules == (
        "newsroom.increment7.agenda_authority",
    )
    assert readiness.allocation_by_issue[439].public_modules == (
        "newsroom.increment7.search_authority",
    )
    assert readiness.allocation_by_issue[443].public_modules == (
        "newsroom.increment7.coverage_authority",
    )
    assert readiness.allocation_by_issue[445].public_modules == (
        "newsroom.increment7.local_watch_authority",
    )
    for issue in (437, 439, 443, 445):
        owned = readiness.allocation_by_issue[issue].interface_ownership
        assert any(name.endswith("AUTHORITY") for name in owned)
        assert any("READ_PORT" in name for name in owned)


def test_migrations_v26_to_v29_are_contiguous_and_singly_owned() -> None:
    readiness = INCREMENT_7_READINESS
    observed = {
        item.issue_number: item.migration_version
        for item in readiness.allocations
        if item.migration_version is not None
    }
    assert observed == _MIGRATIONS
    assert tuple(sorted(observed.values())) == tuple(range(26, 30))
    assert readiness.migration_policy["reservation_is_not_a_migration"] is True
    assert readiness.migration_policy["current_schema_version"] == 25
    assert authority_migrations.SCHEMA_VERSION >= 25


def test_dependency_graph_is_acyclic_and_g_is_last() -> None:
    readiness = INCREMENT_7_READINESS
    assert readiness.parallel_waves == _WAVES
    wave_by_issue = {
        issue: wave
        for wave, issues in readiness.parallel_waves.items()
        for issue in issues
    }
    assert set(wave_by_issue) == set(_CHILD_ISSUES)
    for allocation in readiness.allocations:
        for dependency in allocation.dependencies:
            if dependency in wave_by_issue:
                assert (
                    wave_by_issue[dependency] < wave_by_issue[allocation.issue_number]
                )
    assert readiness.allocation_by_issue[443].dependencies == (435, 439, 441, 442)
    assert readiness.allocation_by_issue[446].dependencies == tuple(range(435, 446))


def test_gate_tiers_and_disabled_runtime_boundary_are_explicit() -> None:
    readiness = INCREMENT_7_READINESS
    assert readiness.allocation_by_issue[435].gate_tier is GateTier.S
    for issue in (436, 438, 440, 441, 442, 444):
        assert readiness.allocation_by_issue[issue].gate_tier is GateTier.L
    for issue in (437, 439, 443, 445):
        assert readiness.allocation_by_issue[issue].gate_tier is GateTier.S
    assert readiness.allocation_by_issue[446].gate_tier is GateTier.M
    assert set(readiness.gate_requirements) == set(GateTier)
    assert (
        "AGENDA_SEARCH_AUDIT_WATCH_REENTRY_PATH"
        in readiness.gate_requirements[GateTier.M]
    )
    assert {
        "LIVE_SEARCH_PROVIDER",
        "RECURRING_QUERY",
        "EXTERNAL_NETWORK_EGRESS",
        "PERMANENT_LOCALITY_SELECTION",
        "MODEL_OR_EMBEDDING_EXECUTION",
        "PRODUCTION_ACTIVATION",
    } <= set(readiness.exclusions)


def test_contract_is_canonical_and_changed_or_duplicate_bytes_fail_closed(
    tmp_path: Path,
) -> None:
    raw = READINESS_CONTRACT_PATH.read_bytes()
    assert raw == canonical_json_bytes(json.loads(raw.decode("utf-8")))
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(_record(), indent=2), encoding="utf-8")
    with pytest.raises(Increment7ReadinessError, match="exact canonical JSON"):
        load_increment7_readiness_contract(pretty)
    duplicate = raw.decode().replace(
        '"schema_version":"newsroom.increment7.readiness.v1"',
        '"schema_version":"newsroom.increment7.readiness.v1","schema_version":"newsroom.increment7.readiness.v1"',
        1,
    )
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(Increment7ReadinessError, match="duplicate object name"):
        load_increment7_readiness_contract(duplicate_path)
    changed = _record()
    payload = changed["payload"]
    assert isinstance(payload, dict)
    payload["production_activation_authorised"] = True
    changed_path = tmp_path / "changed.json"
    changed_path.write_bytes(canonical_json_bytes(changed))
    with pytest.raises(Increment7ReadinessError, match="reviewed v1"):
        load_increment7_readiness_contract(changed_path)


def test_inherited_module_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    item = INCREMENT_7_READINESS.interface_inventory[1]
    monkeypatch.setattr(
        item := readiness_module._INTERFACE_MODULES[item.module],
        "SUPPLEMENTAL_DISCOVERY_REENTRY",
        "CLOCK_CAN_CREATE_CANDIDATE",
    )
    findings = validate_interface_inventory(INCREMENT_7_READINESS)
    assert any(
        "SUPPLEMENTAL_DISCOVERY_REENTRY: value differs" in finding
        for finding in findings
    )


def test_reserved_additive_schema_suffix_is_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checksum = "sha256:" + "a" * 64
    _assert_reviewed_native_successor_inventory()
    wrong_name = list(authority_migrations.EXPECTED_MIGRATION_HISTORY)
    wrong_name[26] = (27, "wrong_v27", checksum)
    monkeypatch.setattr(
        authority_migrations,
        "EXPECTED_MIGRATION_HISTORY",
        tuple(wrong_name),
    )
    assert any(
        "reserved v27 name differs" in finding
        for finding in validate_interface_inventory(INCREMENT_7_READINESS)
    )

    malformed_history = (*INCREMENT_7_READINESS.accepted_migration_history, 26)
    monkeypatch.setattr(
        authority_migrations,
        "EXPECTED_MIGRATION_HISTORY",
        malformed_history,
    )
    findings = validate_interface_inventory(INCREMENT_7_READINESS)
    assert "newsroom.authority.migrations: suffix entry is malformed" in findings
    assert "newsroom.authority.migrations: live history/version differ" in findings

    v29 = tuple(
        (version, f"reserved_v{version}", "sha256:" + "c" * 64)
        for version in range(27, 30)
    )
    v30 = (30, "increment8_authorised_v30", "sha256:" + "d" * 64)
    monkeypatch.setattr(
        authority_migrations,
        "EXPECTED_MIGRATION_HISTORY",
        (
            *INCREMENT_7_READINESS.accepted_migration_history,
            (26, "planned_agenda_authority_v26", checksum),
            *v29,
            v30,
        ),
    )
    monkeypatch.setattr(authority_migrations, "SCHEMA_VERSION", 30)
    findings = validate_interface_inventory(INCREMENT_7_READINESS)
    assert "newsroom.authority.migrations: v30 is outside 7R allocation" not in findings

    monkeypatch.setattr(
        authority_migrations,
        "EXPECTED_MIGRATION_HISTORY",
        (
            *INCREMENT_7_READINESS.accepted_migration_history,
            (26.0, "planned_agenda_authority_v26", checksum),
            (27, "bounded_search_authority_v27", checksum),
        ),
    )
    monkeypatch.setattr(authority_migrations, "SCHEMA_VERSION", 27)
    findings = validate_interface_inventory(INCREMENT_7_READINESS)
    assert (
        "newsroom.authority.migrations: central suffix versions differ" in findings
    )

    monkeypatch.setattr(
        authority_migrations,
        "apply_pending_migrations",
        lambda connection: None,
    )
    findings = validate_interface_inventory(INCREMENT_7_READINESS)
    assert any(
        "apply_pending_migrations: signature differs" in item for item in findings
    )

    monkeypatch.setattr(
        authority_migrations,
        "EXPECTED_SCHEMA_FINGERPRINT",
        INCREMENT_7_READINESS.accepted_schema_fingerprint,
    )
    findings = validate_interface_inventory(INCREMENT_7_READINESS)
    assert (
        "newsroom.authority.migrations: successor fingerprint did not advance"
        in findings
    )
    monkeypatch.setattr(
        authority_migrations,
        "EXPECTED_SCHEMA_FINGERPRINT",
        None,
    )
    findings = validate_interface_inventory(INCREMENT_7_READINESS)
    assert "newsroom.authority.migrations: live fingerprint is malformed" in findings

    monkeypatch.delattr(
        authority_migrations,
        "EXPECTED_MIGRATION_HISTORY",
    )
    monkeypatch.delattr(authority_migrations, "SCHEMA_VERSION")
    monkeypatch.delattr(
        authority_migrations,
        "EXPECTED_SCHEMA_FINGERPRINT",
    )
    findings = validate_interface_inventory(INCREMENT_7_READINESS)
    assert any("EXPECTED_MIGRATION_HISTORY: missing" in item for item in findings)
    assert any("SCHEMA_VERSION: missing" in item for item in findings)
    assert any("EXPECTED_SCHEMA_FINGERPRINT: missing" in item for item in findings)
    assert "newsroom.authority.migrations: accepted history prefix differs" in findings
    assert "newsroom.authority.migrations: schema version regressed" in findings


def test_7r_accepts_the_exact_v25_prefix_without_applying_a_migration() -> None:
    assert (
        INCREMENT_7_READINESS.accepted_migration_history
        == (authority_migrations.EXPECTED_MIGRATION_HISTORY[:25])
    )
    if authority_migrations.SCHEMA_VERSION == 25:
        connection = sqlite3.connect(":memory:", isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            apply_pending_migrations(
                connection,
                applied_at="1970-01-01T00:00:00.000000Z",
            )
            assert connection.execute("PRAGMA user_version").fetchone() == (25,)
            assert schema_fingerprint(connection) == _BASE_FINGERPRINT
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            assert not any(
                name.startswith(
                    (
                        "planned_agenda_",
                        "search_",
                        "coverage_",
                        "event_scoped_local_watch_",
                    )
                )
                for name in tables
            )
        finally:
            connection.close()
