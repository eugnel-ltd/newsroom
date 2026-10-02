from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path

import pytest

from newsroom.authority import migrations as authority_migrations
from newsroom.authority.canonical import digest_canonical
from newsroom.tests.authority_migration_compatibility import (
    BACKUP_PREDECESSOR_VERSIONS,
    CURRENT_VERSION,
    MIGRATION_REGISTRY,
    NEWER_VERSION,
    PINNED_MIGRATION_HISTORY,
    PREDECESSOR_VERSION,
    RETAINED_MIN_VERSION,
    RETAINED_VERSIONS,
    UPGRADE_PREDECESSOR_VERSIONS,
    MigrationCompatibilityError,
    build_exact_prefix,
    canonical_cell,
    history_through,
    inspect_exact_prefix,
    migration_for_version,
    prepare_default_connection_backup,
    render_compatibility_matrix,
    statement_symbol_for_version,
    statements_for_version,
)

_EXPECTED_NAMES = {
    13: "extraction_run_authority_v13",
    14: "entity_resolution_authority_v14",
    15: "editorial_relation_authority_v15",
    16: "graphiti_proposal_adapter_v16",
    17: "evaluation_handoff_authority_v17",
    18: "triage_work_item_authority_v18",
    19: "triage_proposal_disposition_authority_v19",
    20: "triage_execution_authority_v20",
    21: "event_hypothesis_authority_v21",
    22: "event_hypothesis_relationship_authority_v22",
    23: "event_hypothesis_lineage_authority_v23",
    24: "story_candidate_authority_v24",
    25: "evaluation_feedback_authority_v25",
    26: "planned_agenda_authority_v26",
    27: "bounded_search_authority_v27",
    28: "coverage_audit_authority_v28",
    29: "event_scoped_local_watch_authority_v29",
    30: "increment8_evaluation_authority_v30",
    31: "increment8_operational_authority_v31",
    32: "increment8_recovery_authority_v32",
    34: "graphiti_evaluation_extraction_authority_v34",
    35: "graphiti_accounted_zero_proposal_authority_v35",
    36: "authorisation_shared_scope_content_v36",
    37: "authentication_context_compaction_v37",
    38: "authorization_request_residual_storage_v38",
    39: "graphiti_recovered_ambiguous_progression_v39",
    40: "relationship_open_index_v40",
    41: "command_bound_request_result_storage_v41",
    42: "retired_projection_diagnostic_reservations_v42",
}
_EXPECTED_CHECKSUMS = {
    13: "sha256:c3e5ae627dda1c04bebc50952786413d977bd399e67b7f5b87452794f08f49ab",
    14: "sha256:546e81c2419ecb895a1eea7f9c9556931a3e8ad85efe61e878b2fcc25ad72ee9",
    15: "sha256:946a697524cd1ce84546208c21948ec29c59df79410c5eafef196c344f2d8587",
    16: "sha256:ffd44aa70e65e7a2c69a48b3b652160ccc33285d9282c7eed202d206133ba991",
    17: "sha256:c15b3a3fc90833048938b591291d16f59ee1f36b54a6d72dbd04b63877682e7f",
    18: "sha256:f815499c103fed95fbff0c25528331b2483b7c01687f8742394faa92a538bb88",
    19: "sha256:d5f9702d359836e3b564ba1cadbad27e5fc17ba79e5155e2b34382ec30681177",
    20: "sha256:6eb04f981f650bbb4956f148d11f1656bcd2b7c510117db96602dd9d83ab9bd3",
    21: "sha256:42009475669a475af8e3e24bbcd02e6fcd9fbb71a800e18d83624e34e79e5e21",
    22: "sha256:e59eb222a95e2901ccaae29ce1b9e8eded797306e9796718a6d2c4fa505a6636",
    23: "sha256:6c24d402f246f4e82a49a9772d70677d922282aae3b6dde93c62c0ef9b1b7a72",
    24: "sha256:1eea25005483de124e0add0100f4805ed5a537852fc70916f17a209c633e0ca0",
    25: "sha256:59fe3bd40a2e22e874b4e5b02448501deffc23597e11b442e35b18e39ead0496",
    26: "sha256:55e6e8878140714dc6fc6c8149357e1f15e4683fcb7ee0b31b168a737bfd3d4c",
    27: "sha256:ee5679aba6ceb3e95ba925febbbb7853369f93d055563ac85402d380377672b0",
    28: "sha256:c923daf18aed10bb9c197bfd588d816223d978668bda56c157438d1a4b7cc487",
    29: "sha256:ca57c62c9bfadc2ea0a09a3bf762f95854e413aa71d324a296b4c867c90dec7b",
    30: "sha256:764306cbc8fced0b50657c87c2c8735aa07b6ed6b02b1d7ceec84afd9db7dc15",
    31: "sha256:b3a9535516836d7a0023cc0c030926edd8036b0fd8b31b9647342a9612152342",
    32: "sha256:513d983ce8f21f576c08b6a99337f3164025b73e588867d8dde4d500805f79ee",
    34: "sha256:ff4d4c7fdb3d9ebe8354002fe0c71739edd549f46f685782821038ab535f350f",
    35: "sha256:5618c2a4392aabc196687b1fccbb47deee20bcdccc52cc21759cb46e32277829",
    36: "sha256:a91546af0a81e4dbc5c1fb2aaa215a455e36b8e28014991238d597c9ef1b15f9",
    37: "sha256:8ac6c775a376d1787f645ef6526a7aff8e1c70bbb7d22489d06a810d28cfd1be",
    38: "sha256:18c4ef2179dfeea90d8ffec815e89190b1053b87dc4a06410d8b00ee7a789b38",
    39: "sha256:40a2d6a969e6759ad76fa950d122dda1552a81f44f6a65c30c4b1e8389ccc3ed",
    40: "sha256:1f126c29773eded31cb36abcf6f6101dd827e0032c9e379adabbbaffd06c1c79",
    41: "sha256:e2beb20eb921e69a8892234f2e4cae2349369700de2f615ffc9d744e8833fd22",
    42: "sha256:6d8074762801dbb74e966ea4a441ad035a7eebea178372523dab69e5e91fd2e5",
}

_EXPECTED_MATRIX = """version | migration | objects | history fingerprint | schema fingerprint | object fingerprint
--- | --- | ---: | --- | --- | ---
v13 | extraction_run_authority_v13 | 720 | sha256:415a157b19c4ba351c0a257c06d215c081e9348bc846aedfecb2441a4120687e | sha256:62eb9596a324b75a3ec96cc0db6e182217fa30fc6b64fd5b801cf784dfdea9b4 | sha256:ee51d8754920b8dd818a39d464c59106ff55e0024d341dfdfc17b51e45dd66a5
v14 | entity_resolution_authority_v14 | 879 | sha256:5134c423d25fb1f1b3d19974a1fcbc6f8c35966618c35f825552430f48a8aec3 | sha256:47a0421affa099c11cc478220c26d8ce6164cb621057fe9760f410c7dcea8233 | sha256:d52ca75ac8207b3a3abde1a7cd55e6e006e2c62f5b1160b9544d3ba165f4be1c
v15 | editorial_relation_authority_v15 | 985 | sha256:ef45c22301c6c68c81e8fe0327a73623e78ec405474da042ed4868d527a9acda | sha256:5b113904c4ab06452f792078b32bee1752640bb821dc98fc3fdeeb747274efca | sha256:115ee7a7fc0632bf1cf5afa4ac133a120b103f523837e955d4a3c7e0f6ad0c1f
v16 | graphiti_proposal_adapter_v16 | 1068 | sha256:f6368810467de4600769e7213d2aeb9e29bf17fe2988968f4df60d4905ce0cbc | sha256:b5a6d2afc78838cdeb648e7cd34b66452f2e0a0f7dab4773dd17a4cc28e3b5d8 | sha256:6a25ce2721c07b6e90f44e81e1b396a8ef5c89f8f53890121af74cea736a040f
v17 | evaluation_handoff_authority_v17 | 1085 | sha256:0966aacc7bee4d80486f701e3c4a525f1df0a98a6e122f18a7bc1433bddac7a8 | sha256:aaa9544bc6f90dce5831452cffb227967175901e4f7b085e17056ac4194109f5 | sha256:6cff28a00ceb9864c1df97e7fabfc98bf2c25c0793cfdab3afc112482c5e1a7d
v18 | triage_work_item_authority_v18 | 1108 | sha256:7813e4ac4a260c5601b4632fc0aeb020064f9ad9e2edc0700871ad10980aad34 | sha256:7a33005d06998ffd7c438e352ffce2c2c4da008deaa2b0d1171fe3f7599798ea | sha256:08108c0e68585b6236aa4622b46612cc4a5da1265dd5be4865b319dbc2126ca5
v19 | triage_proposal_disposition_authority_v19 | 1123 | sha256:b117a350a73c136444a9a30398618614d6883fc40cdc734fde36303ba5168965 | sha256:542bd9c351094cf4d56905fa92aa042924b5dab877cc04374a097c48fe6b6003 | sha256:082f473aabb994e1941436adacbaa5ede7944349eb2a599c27b784e6be056d5d
v20 | triage_execution_authority_v20 | 1147 | sha256:01aaf90aef4a4e7e5d7946ab944af3d7a19481f8213162cb337f75f0afdf8274 | sha256:36a7c9910775ede9c29113a43e08bba261a5a98c4fab5225dd2cae9448689389 | sha256:60cf06c18f743984c14464793ac579f524c3ef56766441cc6f0bc0aa04fc5d6c
v21 | event_hypothesis_authority_v21 | 1170 | sha256:7404a1b6ffb14aacff8d3e9bb1ddff7f751287100003bfa268d55722c4e34ab0 | sha256:d314d06118a25f8a32a0f9d8acb1af5383abd6b30be682cb5f65943ae15c213f | sha256:d5c9fe7ac19900901f4ccb64545bf9defca4db9e6d82de7e71bb66f9c8d9aaff
v22 | event_hypothesis_relationship_authority_v22 | 1179 | sha256:69acb590abbd8cbb3e6acdad8e6a0c0f31e13e1ed0a718098b64d545c343c1ed | sha256:2118fa893fb7fd2911bbde3056b79b1d0e26ccd6903e1c4228616f342898eaad | sha256:117392ff7fd1160034ec0792ab9f0d94e3a4643dc6fc280b44855ac67efff77a
v23 | event_hypothesis_lineage_authority_v23 | 1192 | sha256:cd4750c9e0c44e3a91b6e6ecbb45ba382dc96fa77fbf2d87d3670801fb6bc9bc | sha256:c341333cf54d724bb4d2092bb9da81e9f3a434ddb03e6ddc14a51fdf2c6c1b52 | sha256:a23a710567b0cdf97f753e4412f1ca0a16b2d85c939a82f32a181f8efa583cce
v24 | story_candidate_authority_v24 | 1221 | sha256:87fbc9d10bfea4239e9105cf851827404afb12012cc91ccf358ae9def233f6ff | sha256:abf8430bfd676a9b0e574847cde9375d90aa1e32680725a08b30c0657d567a7c | sha256:42452161bfddc32e5553bb0aab38c325dda223ec811f47cfae195d642fe06926
v25 | evaluation_feedback_authority_v25 | 1253 | sha256:bea793377d065d3073e6dfa8d40139fedfd377d5e24d9812d12cdb1ad52e9a0f | sha256:353900bf5804f0b770489982541f3cff4fd30ea36fc75d19b9c63315d1b6ec06 | sha256:002a7c3cb59690568a8779e05c46a751ef57f91ee9070dc9095b0bfb953cf3f8
v26 | planned_agenda_authority_v26 | 1285 | sha256:d2cb21a981674513b4495ca19012e49b93c8dc30cbe61165b5462c391f989004 | sha256:9c4d5b94b10b34d3b9ef2f140dbfbcc85b2c01fb6d8660879403a21fef701374 | sha256:0724fb8b7796f7581123d17658613baf488090dfcc7d5320f9dd804364d5a542
v27 | bounded_search_authority_v27 | 1347 | sha256:f4afb25765da01175b272333daaaef80b21416d9258a5ec4e09ffba59f878936 | sha256:d7fc1557bd02588969efdd53c749a2f125ab5bab146395c4cf8f7d51b1e32719 | sha256:ebd8c2d5739a09103f526fa682dadf66efa7646304b25f266265b91b70e55cbd
v28 | coverage_audit_authority_v28 | 1381 | sha256:81792dba56950b37325ccdffe06b64f322379218a10264e949d57e7e0dec58d6 | sha256:a613b28a765b36fa9110bcdc2b9bc565c6e2bc0ed8b8381d77f5fcd734c39c48 | sha256:a8330e1d8ae8c9e181f0e2d1ebbde88a1386e934e93eaac3c3d292722c7fff35
v29 | event_scoped_local_watch_authority_v29 | 1417 | sha256:02e531a9279e316e7f131beabfef5b2f5d02f6b825b7312f4c6296028ffee4ff | sha256:68194825ecc7c429b283204dbc1332a43481e04ca2681fcbf75886a984ea6f55 | sha256:75fc30052c67d91d4a439a3baadc6c3410f65d470d535906a0624d051a0f32cd
v30 | increment8_evaluation_authority_v30 | 1456 | sha256:abcca56b09c0f49ea8315f21f66d252153c63fd572ba1eaa2bcd9a4215214808 | sha256:cf9a5ee83f6d3396d8d9fff4aa234ba252037dd4056e88f85afcb27c6c45bfd9 | sha256:ec469b35837aaf07827e9cde89aca928201e129f85ffd1f0f8046b0cd08ba26c
v31 | increment8_operational_authority_v31 | 1486 | sha256:7a1592121ba3f2c399f7fcdefdf8d618b6d7f08ecc94ffa96ca918e8213e0830 | sha256:8a8f2aafc484a4d0270b0fbc582c2c22fc83e545570e3117b0b2be8eee874bc5 | sha256:097fde371d437c41a9493a95f788857cfae0b53dd42f1ef466b713a9022a4964
v32 | increment8_recovery_authority_v32 | 1511 | sha256:5a48fd76cd11f266e19a4b48174d0c009f320a8d00d3eeb281a558fc2d561910 | sha256:3439b82ec6d212116e54765d50cace4d7f147b6ecc3e6ff84146b523c6fd5676 | sha256:ca9ce0c8b304f7d7bccb2f1e3796f02ff4d0c024a6fa78dc4d2098478afe4fae
v34 | graphiti_evaluation_extraction_authority_v34 | 1516 | sha256:f589854a5241991459ba5052be2cb3804c75da2742ca52d035c679292c8c8d9f | sha256:8b38a4c2279363ed4105c272370bfdc733591c30032aed4cbab5e83ef92b7065 | sha256:cc181b84140510c0239bc812b690c66cd085cdce314381b08bdad78244b0f6b4
v35 | graphiti_accounted_zero_proposal_authority_v35 | 1516 | sha256:eb02cf288b626cbd2a895d972a8ebd69ab66a2b3e243445edfc7241bca546b20 | sha256:e6f107455a75986a977008073e3882780155d51b73660b1a2ed780a2e573455a | sha256:38a2ffa11cfd76250cf9782102093708cdf64570fe201d68a1ef3fdd64d3c7a0
v36 | authorisation_shared_scope_content_v36 | 1522 | sha256:cddffffe87f4c5123c3f5f501bb246668dd7b13077e1e4b0cbced4d7aab2b1d1 | sha256:df4cd39f154791d3e5680ac4fa501c2a076427d0ea18caff40145319c08647d0 | sha256:66ecc87b40ee20a59a0c67ddc0ade6c6a6ca670b9c883939b84b7faa0b6d183c
v37 | authentication_context_compaction_v37 | 1523 | sha256:4478eb4b5d8ea85fa26a72e02830c71d5fafff20d62e72e56e4be1edb9c26ad4 | sha256:4003bc1eb0124845189a50e561b39da33bfde75ab8eabd19ddf9c7f807417d3d | sha256:fe1f5f6d2a109496751dd035da536c7ce9833c19075d7f2ec0a8c3a985c9b505
v38 | authorization_request_residual_storage_v38 | 1524 | sha256:28c196a875fd3553b758064fd90c9fa2be1f0f08931b69806dd3c84a7b2e6a70 | sha256:fd12ca767bc53236efb0a7500dae2bc2dbaee70bfc78cdc99a5ad4c421cc2a29 | sha256:89bd6d2ad8f81bfff8ee0cbad72fa1f266038b363a15a4f4e9fb450f8992a016
v39 | graphiti_recovered_ambiguous_progression_v39 | 1524 | sha256:fa2d3fc80a04d5df4c5213e44f95c095279256a3bf3d31e41e465c38986a7b96 | sha256:d88defccf440c0c5a89ae9a3e2e900bb4acdc64fb9781c9483add6d0ad79a6c8 | sha256:a36bec5b6d0f281e2ee05274700d1ed18a0f38688ae5731a1f8688cd57c1e429
v40 | relationship_open_index_v40 | 1525 | sha256:864fd85321ff4cfec55870a3d3737e94582f22cab32e9adcea9cf66bea6f3f7f | sha256:1e6a22fbc1b755d1eebd957378dc691b4e41e0ca405f2ef00d7ddcbd629206c7 | sha256:bff1d52da6dc88a67095977d8a737c710891af420efd07238ed16423040a466c
v41 | command_bound_request_result_storage_v41 | 1525 | sha256:71dcf0dd7c21734f7d44261451c99c244bb6cdadb9e13c54a805a589089e8773 | sha256:94014382880d8d9cda0a7b727dea90ca522c65299345e5ea6f80cf857311a3e9 | sha256:2159817f0d9e5a0d7b11d00820eb0485f88168a140393c1c98c13ef2dd8f46de
v42 | retired_projection_diagnostic_reservations_v42 | 1530 | sha256:f9b00420523752d7a736693da3288d922ec2895c79e21565e96596fefb9d4d9c | sha256:30bbcbc39e452773ae141b19794601c7ff8d294b09fa47624e561dc9de5ba104 | sha256:79901ff4764b2faf118d9aab2c2a1ed5a0a308f1bff6dfa69bc54e66feee99d7
"""


def _upgrade_to_current(database: Path, starting_version: int) -> None:
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        if starting_version in BACKUP_PREDECESSOR_VERSIONS:
            prepare_default_connection_backup(connection)
        authority_migrations.apply_pending_migrations(
            connection, applied_at="1970-01-02T00:00:00.000000Z"
        )
    finally:
        connection.close()


def test_registry_history_and_statement_pins_are_complete_and_named() -> None:
    assert CURRENT_VERSION == authority_migrations.SCHEMA_VERSION
    assert PREDECESSOR_VERSION == tuple(
        record.version for record in MIGRATION_REGISTRY
    )[-2]
    assert NEWER_VERSION == CURRENT_VERSION + 1
    assert UPGRADE_PREDECESSOR_VERSIONS == RETAINED_VERSIONS[:-1]
    assert (
        BACKUP_PREDECESSOR_VERSIONS[-1]
        == authority_migrations.GRAPHITI_EVALUATION_SCHEMA_VERSION
    )
    assert PREDECESSOR_VERSION not in BACKUP_PREDECESSOR_VERSIONS
    assert tuple(authority_migrations.EXPECTED_MIGRATION_HISTORY) == (
        PINNED_MIGRATION_HISTORY
    )
    assert RETAINED_MIN_VERSION == 13
    assert RETAINED_VERSIONS == tuple(_EXPECTED_NAMES)
    assert tuple(record.version for record in MIGRATION_REGISTRY) == tuple(
        (*range(1, 33), 34, 35, 36, 37, 38, 39, 40, 41, 42)
    )
    assert (
        tuple(
            (record.version, record.name, record.checksum)
            for record in MIGRATION_REGISTRY
        )
        == authority_migrations.EXPECTED_MIGRATION_HISTORY
    )

    for version, name, checksum in PINNED_MIGRATION_HISTORY:
        record = migration_for_version(version)
        assert (record.name, record.checksum) == (name, checksum)
        assert history_through(version)[-1] == (
            record.version,
            record.name,
            record.checksum,
        )
        assert (
            digest_canonical(
                {
                    "version": record.version,
                    "name": record.name,
                    "statements": list(statements_for_version(version)),
                }
            )
            == record.checksum
        )

    for version, expected_name in _EXPECTED_NAMES.items():
        record = migration_for_version(version)
        assert record.name == expected_name
        assert record.checksum == _EXPECTED_CHECKSUMS[version]

    with pytest.raises(MigrationCompatibilityError, match="found 0"):
        migration_for_version(NEWER_VERSION)
    with pytest.raises(MigrationCompatibilityError, match="exact integer"):
        migration_for_version(True)


@pytest.mark.parametrize("version", RETAINED_VERSIONS)
def test_exact_prefix_cells_are_deterministic(tmp_path: Path, version: int) -> None:
    database = tmp_path / f"v{version}.sqlite3"
    built = build_exact_prefix(database, version)
    inspected = inspect_exact_prefix(database, expected_version=version)

    assert built == inspected == canonical_cell(version)
    assert inspected.history == history_through(version)
    assert inspected.foreign_key_check == ()
    assert inspected.quick_check == ("ok",)


def test_fresh_current_migrator_equals_direct_exact_current_prefix(
    tmp_path: Path,
) -> None:
    direct_path = tmp_path / f"direct-v{CURRENT_VERSION}.sqlite3"
    fresh_path = tmp_path / "fresh-current.sqlite3"
    direct = build_exact_prefix(direct_path, CURRENT_VERSION)

    connection = sqlite3.connect(fresh_path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        authority_migrations.apply_pending_migrations(
            connection, applied_at="1970-01-01T00:00:00.000000Z"
        )
    finally:
        connection.close()

    assert inspect_exact_prefix(fresh_path, expected_version=CURRENT_VERSION) == direct


@pytest.mark.parametrize(
    ("version", "expected_calls"),
    ((35, 0), (36, 1)),
)
def test_exact_prefix_invokes_v36_procedure_once_only_for_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: int,
    expected_calls: int,
) -> None:
    calls = 0
    original = authority_migrations.migrate_authorisation_scope_content

    def counted(connection: sqlite3.Connection, *, expected_history) -> None:
        nonlocal calls
        calls += 1
        assert connection.in_transaction
        original(connection, expected_history=expected_history)

    monkeypatch.setattr(
        authority_migrations, "migrate_authorisation_scope_content", counted
    )
    build_exact_prefix(tmp_path / f"procedural-v{version}.sqlite3", version)
    assert calls == expected_calls


@pytest.mark.parametrize(("version", "expected_calls"), ((41, 0), (42, 1)))
def test_exact_prefix_invokes_v42_procedure_only_at_its_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: int,
    expected_calls: int,
) -> None:
    calls = 0
    original = authority_migrations.migrate_projection_retirement

    def counted(connection: sqlite3.Connection, *, expected_history) -> None:
        nonlocal calls
        calls += 1
        assert connection.in_transaction
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert expected_history == history_through(41)
        original(connection, expected_history=expected_history)

    monkeypatch.setattr(authority_migrations, "migrate_projection_retirement", counted)
    build_exact_prefix(tmp_path / f"procedural-v{version}.sqlite3", version)
    assert calls == expected_calls


@pytest.mark.parametrize("predecessor", UPGRADE_PREDECESSOR_VERSIONS)
def test_exact_predecessors_upgrade_to_current(
    tmp_path: Path, predecessor: int
) -> None:
    database = tmp_path / f"upgrade-v{predecessor}.sqlite3"
    build_exact_prefix(database, predecessor)
    _upgrade_to_current(database, predecessor)
    assert inspect_exact_prefix(
        database, expected_version=CURRENT_VERSION
    ) == canonical_cell(CURRENT_VERSION)


def test_multihop_upgrade_retains_each_exact_backup_and_digest(tmp_path: Path) -> None:
    database = tmp_path / "multihop.sqlite3"
    build_exact_prefix(database, RETAINED_MIN_VERSION)
    _upgrade_to_current(database, RETAINED_MIN_VERSION)

    for predecessor in BACKUP_PREDECESSOR_VERSIONS:
        successor = next(
            record.version
            for record in MIGRATION_REGISTRY
            if record.version > predecessor
        )
        backup = database.with_name(database.name + f".pre-v{successor}.sqlite3")
        digest = backup.with_name(backup.name + ".sha256")
        assert inspect_exact_prefix(
            backup, expected_version=predecessor
        ) == canonical_cell(predecessor)
        assert digest.read_text(encoding="ascii") == (
            "sha256:" + hashlib.sha256(backup.read_bytes()).hexdigest() + "\n"
        )


@pytest.mark.parametrize("predecessor", BACKUP_PREDECESSOR_VERSIONS)
def test_default_connection_backup_leaves_exclusive_upgrade_available(
    tmp_path: Path, predecessor: int
) -> None:
    database = tmp_path / f"default-v{predecessor}.sqlite3"
    build_exact_prefix(database, predecessor)

    connection = sqlite3.connect(database)
    try:
        assert connection.isolation_level == ""
        receipt = prepare_default_connection_backup(connection)
        assert Path(receipt.backup_path).is_file()
        assert not connection.in_transaction
        connection.execute("BEGIN EXCLUSIVE")
        connection.execute("ROLLBACK")
        authority_migrations.apply_pending_migrations(
            connection, applied_at="1970-01-02T00:00:00.000000Z"
        )
    finally:
        connection.close()

    inspect_exact_prefix(database, expected_version=CURRENT_VERSION)


@pytest.mark.parametrize(
    ("predecessor", "fail_after", "message"),
    (
        (40, "ALTER TABLE authorization_requests DROP COLUMN storage_request_marker",
         "injected after v41 marker removal"),
        (41, "DROP TABLE ledger_events", "injected after v42 parent removal"),
    ),
)
def test_failed_upgrade_rolls_back_to_exact_predecessor(
    tmp_path: Path, predecessor: int, fail_after: str, message: str,
) -> None:
    database = tmp_path / f"rollback-v{predecessor}.sqlite3"
    before = build_exact_prefix(database, predecessor)

    class FailAfterMigrationStatement(sqlite3.Connection):
        def execute(self, sql: str, parameters=(), /):
            cursor = super().execute(sql, parameters)
            if sql == fail_after:
                raise sqlite3.OperationalError(message)
            return cursor

    connection = sqlite3.connect(database, factory=FailAfterMigrationStatement)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.DatabaseError, match=message):
            authority_migrations.apply_pending_migrations(
                connection, applied_at="1970-01-02T00:00:00.000000Z"
            )
        assert not connection.in_transaction
    finally:
        connection.close()

    assert inspect_exact_prefix(database, expected_version=predecessor) == before


def test_successful_upgrade_can_restore_exact_backup_then_reupgrade(
    tmp_path: Path,
) -> None:
    backup_predecessor = BACKUP_PREDECESSOR_VERSIONS[-1]
    database = tmp_path / f"restore-v{backup_predecessor}.sqlite3"
    exact_predecessor = build_exact_prefix(database, backup_predecessor)

    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        receipt = prepare_default_connection_backup(connection)
        authority_migrations.apply_pending_migrations(
            connection, applied_at="1970-01-02T00:00:00.000000Z"
        )
    finally:
        connection.close()

    backup = Path(receipt.backup_path)
    digest = Path(receipt.digest_path)
    assert inspect_exact_prefix(
        database, expected_version=CURRENT_VERSION
    ) == canonical_cell(CURRENT_VERSION)
    assert (
        inspect_exact_prefix(backup, expected_version=backup_predecessor)
        == exact_predecessor
    )
    assert digest.read_text(encoding="ascii") == (
        "sha256:" + hashlib.sha256(backup.read_bytes()).hexdigest() + "\n"
    )

    database.unlink()
    shutil.copy2(backup, database)
    assert (
        inspect_exact_prefix(database, expected_version=backup_predecessor)
        == exact_predecessor
    )

    _upgrade_to_current(database, backup_predecessor)
    assert inspect_exact_prefix(
        database, expected_version=CURRENT_VERSION
    ) == canonical_cell(CURRENT_VERSION)


def test_exact_builder_statement_failure_rolls_back_atomically(tmp_path: Path) -> None:
    database = tmp_path / "builder-failure.sqlite3"
    failure_version = BACKUP_PREDECESSOR_VERSIONS[2]

    def fail_at_v18(
        connection: sqlite3.Connection, version: int, index: int, statement: str
    ) -> None:
        if version == failure_version and index == 0:
            raise sqlite3.OperationalError("injected statement failure")
        connection.execute(statement)

    with pytest.raises(sqlite3.OperationalError, match="injected"):
        build_exact_prefix(database, failure_version, statement_executor=fail_at_v18)

    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert (
            connection.execute("SELECT type,name FROM sqlite_master").fetchall() == []
        )
    finally:
        connection.close()


def test_newer_schema_is_rejected_before_inventory_inspection(tmp_path: Path) -> None:
    database = tmp_path / "newer.sqlite3"
    build_exact_prefix(database, CURRENT_VERSION)
    connection = sqlite3.connect(database)
    try:
        connection.execute(f"PRAGMA user_version={NEWER_VERSION}")
    finally:
        connection.close()

    with pytest.raises(MigrationCompatibilityError, match="outside retained"):
        inspect_exact_prefix(database)


def test_production_migrator_rejects_newer_without_changing_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / f"central-newer-v{NEWER_VERSION}.sqlite3"
    build_exact_prefix(database, CURRENT_VERSION)
    connection = sqlite3.connect(database)
    try:
        connection.execute(f"PRAGMA user_version={NEWER_VERSION}")
    finally:
        connection.close()

    before_bytes = database.read_bytes()
    connection = sqlite3.connect(database)
    try:
        before_version = connection.execute("PRAGMA user_version").fetchone()[0]
        before_history = connection.execute(
            "SELECT version,name,checksum FROM authority_migrations ORDER BY version"
        ).fetchall()
        before_inventory = connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master "
            "ORDER BY type,name,tbl_name"
        ).fetchall()
        with pytest.raises(sqlite3.DatabaseError, match="newer than supported"):
            authority_migrations.apply_pending_migrations(
                connection, applied_at="1970-01-02T00:00:00.000000Z"
            )
        assert connection.execute("PRAGMA user_version").fetchone()[0] == before_version
        assert (
            connection.execute(
                "SELECT version,name,checksum FROM authority_migrations ORDER BY version"
            ).fetchall()
            == before_history
        )
        assert (
            connection.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_master "
                "ORDER BY type,name,tbl_name"
            ).fetchall()
            == before_inventory
        )
    finally:
        connection.close()

    assert database.read_bytes() == before_bytes


@pytest.mark.parametrize("tamper", ("checksum", "missing", "extra"))
def test_history_tamper_is_rejected(tmp_path: Path, tamper: str) -> None:
    database = tmp_path / f"history-{tamper}.sqlite3"
    build_exact_prefix(database, CURRENT_VERSION)
    connection = sqlite3.connect(database)
    try:
        if tamper == "checksum":
            connection.execute("DROP TRIGGER immutable_authority_migrations_update")
            connection.execute(
                "UPDATE authority_migrations SET checksum='sha256:tampered' "
                f"WHERE version={BACKUP_PREDECESSOR_VERSIONS[2]}"
            )
        elif tamper == "missing":
            connection.execute("DROP TRIGGER immutable_authority_migrations_delete")
            connection.execute(
                "DELETE FROM authority_migrations WHERE version=?",
                (BACKUP_PREDECESSOR_VERSIONS[2],),
            )
        else:
            connection.execute(
                "INSERT INTO authority_migrations VALUES(999,'extra','extra','now')"
            )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(MigrationCompatibilityError, match="migration history"):
        inspect_exact_prefix(database, expected_version=CURRENT_VERSION)


@pytest.mark.parametrize("object_type", ("table", "trigger", "index"))
def test_sqlite_master_object_leaks_are_rejected(
    tmp_path: Path, object_type: str
) -> None:
    database = tmp_path / f"leak-{object_type}.sqlite3"
    build_exact_prefix(database, CURRENT_VERSION)
    statements = {
        "table": "CREATE TABLE leaked_table(value TEXT)",
        "trigger": (
            "CREATE TRIGGER leaked_trigger AFTER INSERT ON authority_migrations "
            "BEGIN SELECT 1; END"
        ),
        "index": "CREATE INDEX leaked_index ON authority_migrations(applied_at)",
    }
    connection = sqlite3.connect(database)
    try:
        connection.execute(statements[object_type])
    finally:
        connection.close()

    with pytest.raises(MigrationCompatibilityError, match="sqlite_master inventory"):
        inspect_exact_prefix(database, expected_version=CURRENT_VERSION)


def test_missing_sqlite_master_object_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "missing-index.sqlite3"
    build_exact_prefix(database, CURRENT_VERSION)
    connection = sqlite3.connect(database)
    try:
        row = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND sql IS NOT NULL ORDER BY name LIMIT 1"
        ).fetchone()
        assert row is not None
        index_name = str(row[0]).replace('"', '""')
        connection.execute(f'DROP INDEX "{index_name}"')
    finally:
        connection.close()

    with pytest.raises(MigrationCompatibilityError, match="sqlite_master inventory"):
        inspect_exact_prefix(database, expected_version=CURRENT_VERSION)


def test_quoted_literal_whitespace_drift_bypasses_normalised_fingerprint_only(
    tmp_path: Path,
) -> None:
    database = tmp_path / "quoted-literal-space.sqlite3"
    build_exact_prefix(database, CURRENT_VERSION)
    connection = sqlite3.connect(database)
    try:
        production_fingerprint = authority_migrations.schema_fingerprint(connection)
        row = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='trigger' AND name='immutable_authority_migrations_update'"
        ).fetchone()
        assert row is not None
        original_sql = str(row[0])
        changed_sql = original_sql.replace(
            "immutable migration history", "immutable  migration history", 1
        )
        assert changed_sql != original_sql
        connection.execute("DROP TRIGGER immutable_authority_migrations_update")
        connection.execute(changed_sql)
        connection.commit()
        assert (
            authority_migrations.schema_fingerprint(connection)
            == production_fingerprint
        )
    finally:
        connection.close()

    with pytest.raises(MigrationCompatibilityError, match="sqlite_master inventory"):
        inspect_exact_prefix(database, expected_version=CURRENT_VERSION)


def test_automatic_index_null_sql_changed_to_empty_text_is_rejected(
    tmp_path: Path,
) -> None:
    database = tmp_path / "automatic-index-empty-sql.sqlite3"
    build_exact_prefix(database, CURRENT_VERSION)
    connection = sqlite3.connect(database)
    try:
        row = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND name LIKE 'sqlite_autoindex_%' "
            "AND sql IS NULL ORDER BY name LIMIT 1"
        ).fetchone()
        assert row is not None
        connection.execute("PRAGMA writable_schema=ON")
        try:
            connection.execute(
                "UPDATE sqlite_master SET sql='' WHERE type='index' AND name=?",
                (str(row[0]),),
            )
            connection.commit()
        finally:
            connection.execute("PRAGMA writable_schema=OFF")
    finally:
        connection.close()

    reopened = sqlite3.connect(database)
    try:
        assert reopened.execute("PRAGMA writable_schema").fetchone()[0] == 0
        assert reopened.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert (
            reopened.execute(
                "SELECT typeof(sql) FROM sqlite_master WHERE name=?", (str(row[0]),)
            ).fetchone()[0]
            == "text"
        )
    finally:
        reopened.close()

    with pytest.raises(MigrationCompatibilityError, match="sqlite_master inventory"):
        inspect_exact_prefix(database, expected_version=CURRENT_VERSION)


def test_matrix_render_is_stable_and_pinned() -> None:
    assert render_compatibility_matrix() == _EXPECTED_MATRIX
