import json
import os
import stat
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane.corpus import CorpusIngestUnit
from newsroom.control_plane.cycle import CycleReport
from newsroom.control_plane.cycle import run_cycle as _run_cycle
from newsroom.control_plane.editorial import StoryCandidateRecord
from newsroom.control_plane.evidence import (
    ClaimAuthorityClass,
    Evid012QualificationTest,
    EvidenceGateEvidence,
    EvidencePackage,
    GovernedClaimEvidence,
    GovernedClaimStatus,
    QualificationEvidence,
    bounded_named_entities,
    package_for,
)
from newsroom.control_plane.intake import run_intake
from newsroom.control_plane.items import parse_observation
from newsroom.control_plane.reports import news_report
from newsroom.control_plane.store import connect, list_payloads, mark_public_dispatch
from newsroom.control_plane.surface import UnpublishedSurfacePayload
from newsroom.control_plane.veto import VetoError, refuse_public_effect
from newsroom.control_plane.writer import (
    CliChainWriter,
    FixtureWriter,
    WriterCopy,
    WriterDispatchError,
    cont_writer_implementation_identity,
    run_cursor_agent_cli,
    run_grok_cli,
)
from newsroom.effective_revision import (
    retain_observation_revision_first_seen,
)
from newsroom.graphiti_adapter.evaluation_packet import (
    EVALUATION_GRAPHITI_PACKET,
    EVALUATION_WORKSPACE_POLICY,
    GRAPHITI_CHAT_MODEL,
    GRAPHITI_CORE_RELEASE,
    GRAPHITI_EMBEDDING_MODEL,
    GRAPHITI_EVALUATION_DESTINATION_TOKENS,
    OPENROUTER_API,
    WRITER_FALLBACK,
    WRITER_MODEL,
)
from newsroom.graphiti_adapter.models import REAL_GRAPHITI_RUNTIME_ENABLED
from newsroom.graphiti_adapter.temporal import OBSERVED_FALLBACK
from newsroom.graphiti_adapter.types import (
    GraphitiCredentialClass,
    GraphitiEgressPolicy,
    GraphitiRuntimeNotAuthorized,
)
from newsroom.increment9.proving import (
    FETCH_MAX_ATTEMPTS,
    PROVING_GATES,
    SOURCE_URLS,
)
from newsroom.increment9.rights import (
    BINDINGS,
    FIXTURE_DESTINATIONS,
    FIXTURE_NOW,
    fixture_inventory,
)


def run_cycle(*args: Any, **kwargs: Any) -> CycleReport:
    kwargs.setdefault("clock", lambda: datetime(2026, 8, 20, tzinfo=UTC))
    kwargs.setdefault("evidence_package_builder", _fixture_evidence_package)
    return _run_cycle(*args, **kwargs)


def _fixture_evidence_package(candidate: StoryCandidateRecord) -> EvidencePackage:
    package = package_for(candidate)
    headline = f"Official action changed: {candidate.headline}"
    claim = f"Official action changed: {candidate.items[0].body}"
    governed_claims = (
        GovernedClaimEvidence(
            claim_id=f"claim:{candidate.candidate_id}:headline",
            claim=headline,
            passage_index=0,
            supporting_excerpt=headline,
            source_ids=(candidate.items[0].source_id,),
            source_record_ids=(f"source-record:{candidate.items[0].source_id}",),
            source_authority_decision_ids=(
                f"source-authority:{candidate.items[0].source_id}:headline",
            ),
            rights_decision_ids=(f"rights:{candidate.items[0].source_id}",),
            dependency_evidence_ids=(f"dependency:{candidate.items[0].source_id}",),
            evidential_origin_ids=(f"origin:{candidate.items[0].source_id}",),
            authority_class=ClaimAuthorityClass.RESPONSIBLE_PRIMARY,
            authority_scope="Fixture responsible source for its own update.",
            status=GovernedClaimStatus.CONFIRMED_FACT,
            attribution=candidate.items[0].source_id,
            rendered_assertion_zh_hant_hk="官方公布咗最新安排",
            claim_role="HEADLINE",
            semantic_relation_evidence_id=f"semantic:{candidate.candidate_id}:headline",
        ),
        GovernedClaimEvidence(
            claim_id=f"claim:{candidate.candidate_id}:substantive",
            claim=claim,
            passage_index=0,
            supporting_excerpt=claim,
            source_ids=(candidate.items[0].source_id,),
            source_record_ids=(f"source-record:{candidate.items[0].source_id}",),
            source_authority_decision_ids=(
                f"source-authority:{candidate.items[0].source_id}:substantive",
            ),
            rights_decision_ids=(f"rights:{candidate.items[0].source_id}",),
            dependency_evidence_ids=(f"dependency:{candidate.items[0].source_id}",),
            evidential_origin_ids=(f"origin:{candidate.items[0].source_id}",),
            authority_class=ClaimAuthorityClass.RESPONSIBLE_PRIMARY,
            authority_scope="Fixture responsible source for its own update.",
            status=GovernedClaimStatus.CONFIRMED_FACT,
            attribution=candidate.items[0].source_id,
            rendered_assertion_zh_hant_hk="相關官方資料確認安排已經更新",
            claim_role="SUBSTANTIVE",
            semantic_relation_evidence_id=f"semantic:{candidate.candidate_id}:substantive",
        ),
    )
    governed_claims = tuple(_bind_fixture_entities(item) for item in governed_claims)
    claim_ids = tuple(item.claim_id for item in governed_claims)
    return replace(
        package,
        passages=(f"{package.passages[0]}\n{headline}\n{claim}",),
        substantive_new_information=(headline, claim),
        governed_claims=governed_claims,
        qualification_evidence=tuple(
            QualificationEvidence(
                Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE,
                item.claim_id,
                f"qualification:{item.claim_id}",
                (
                    ("action_class", "OFFICIAL_DEADLINE"),
                    ("event_polarity", "AFFIRMED"),
                    ("action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"),
                    ("material_relation_span", item.claim),
                    ("reader_action", item.claim),
                ),
            )
            for item in governed_claims
        ),
        selection_rationale="Explicit private-beta fixture qualification.",
        geography=("Hong Kong" if "HK" in candidate.items[0].source_id else "UK",),
        categories=("Politics and law",),
        evidence_gate_results=(
            ("CLAIM_TRACEABILITY", "PASS"),
            ("EVIDENCE_SUFFICIENCY", "PASS"),
            ("SOURCE_AUTHORITY", "PASS"),
        ),
        evidence_gate_evidence=tuple(
            EvidenceGateEvidence(gate, "PASS", claim_ids)  # type: ignore[arg-type]
            for gate in (
                "CLAIM_TRACEABILITY",
                "EVIDENCE_SUFFICIENCY",
                "SOURCE_AUTHORITY",
            )
        ),
        freshness_result="PASS",
        integrity_result="PASS",
        resolved_evidence_records=(
            ("fixture-evidence-record", f"digest:{candidate.candidate_id}"),
            *(
                (
                    item.semantic_relation_evidence_id,
                    f"semantic-digest:{item.claim_id}",
                )
                for item in governed_claims
            ),
            *(
                (record_id, f"entity-digest:{record_id}")
                for item in governed_claims
                for _text, _entity_type, record_id in item.named_entity_evidence
            ),
            *(
                (
                    f"qualification:{item.claim_id}",
                    f"qualification-digest:{item.claim_id}",
                )
                for item in governed_claims
            ),
        ),
    )


def _bind_fixture_entities(claim: GovernedClaimEvidence) -> GovernedClaimEvidence:
    entities = tuple(
        sorted(bounded_named_entities(f"{claim.claim}\n{claim.supporting_excerpt}"))
    )
    if not entities:
        return claim
    texts = tuple(text for text, _entity_type in entities)
    return replace(
        claim,
        rendered_assertion_zh_hant_hk=(
            "、".join(texts) + "：" + claim.rendered_assertion_zh_hant_hk
        ),
        named_entity_evidence=tuple(
            (text, entity_type, f"entity:{claim.claim_id}:{index}")
            for index, (text, entity_type) in enumerate(entities)
        ),
        named_entities=texts,
        rendered_named_entities=texts,
    )


ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>urn:example:1</id>
    <title>Home Office update</title>
    <link href="https://www.gov.uk/example-1"/>
    <summary>A retained proving item.</summary>
  </entry>
</feed>
"""
RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item>
    <guid>hk-1</guid>
    <title>香港政府新聞</title>
    <link>https://www.news.gov.hk/a</link>
    <description>保留來源正文。</description>
  </item>
</channel></rss>
""".encode("utf-8")
JSON_DOC = b'{"title":"BNO visa","base_path":"/british-national-overseas-bno-visa","content_id":"abc","description":"Apply for a visa."}'
SAME_URL_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item>
    <guid>rad-1</guid>
    <title>Same event from radio</title>
    <link>https://www.gov.uk/example-1</link>
    <description>Corroborating retained item.</description>
  </item>
</channel></rss>
""".encode("utf-8")


def _evaluation_cycle_destinations() -> tuple[str, ...]:
    return tuple(
        sorted({*FIXTURE_DESTINATIONS, *GRAPHITI_EVALUATION_DESTINATION_TOKENS})
    )


def _cycle_rights_inventory(
    gate: str,
    destinations: tuple[str, ...] | None = None,
) -> dict[str, object]:
    return fixture_inventory(
        gate=gate,
        destinations=(
            _evaluation_cycle_destinations() if destinations is None else destinations
        ),
        now="2026-08-20T00:00:00.000000Z",
        issued_at="2026-01-01T00:00:00.000000Z",
        expires_at="2099-01-01T00:00:00.000000Z",
    )


def _proving(tmp_path: Path, extra: tuple[tuple[str, bytes], ...] = ()) -> Path:
    path = tmp_path / "proving_store.sqlite3"
    from newsroom.increment9.proving import _connect, _store_body

    connection = _connect(str(path))
    connection.execute(
        """
        INSERT INTO proving_runs(
            run_id, started_at, publication, public_dispatch,
            openrouter_invoked, spend_gbp_minor
        ) VALUES('run-1','2026-08-16T21:41:34.000000Z',0,0,0,0)
        """
    )
    rows = (
        (
            "UK-01",
            SOURCE_URLS["UK-01"],
            ATOM,
        ),
        (
            "HK-01",
            "https://www.news.gov.hk/tc/common/html/topstories.rss.xml",
            RSS,
        ),
        (
            "UK-02",
            "https://www.gov.uk/api/content/british-national-overseas-bno-visa",
            JSON_DOC,
        ),
        *tuple(
            (
                source_id,
                BINDINGS[f"RIGHTS_{source_id}"][2],
                body,
            )
            for source_id, body in extra
        ),
    )
    for source_id, url, body in rows:
        digest = digest_bytes(body)
        _store_body(connection, digest, body)
        connection.execute(
            "INSERT INTO proving_observations VALUES(?,?,?,?,?,?,?,?)",
            (
                source_id,
                "run-1",
                "2026-08-16T21:41:34.000000Z",
                url,
                200,
                digest,
                1,
                None,
            ),
        )
        retain_observation_revision_first_seen(
            connection,
            source_id=source_id,
            url=url,
            body=body,
            observed_at="2026-08-16T21:41:34.000000Z",
        )
    for gate_id in PROVING_GATES:
        if gate_id.startswith("RIGHTS_"):
            continue
        connection.execute(
            "INSERT INTO proving_gates VALUES(?,?,?,?)",
            ("run-1", gate_id, "PASS", "fixture"),
        )
    for source_id, _url, _body in rows:
        gate_id = f"RIGHTS_{source_id}"
        connection.execute(
            "INSERT INTO proving_gates VALUES(?,?,?,?)",
            ("run-1", gate_id, "PASS", "fixture"),
        )
        packet = _cycle_rights_inventory(gate_id)
        packet_bytes = canonical_json_bytes(packet)
        connection.execute(
            "INSERT INTO proving_rights_packets VALUES(?,?,?,?,?)",
            (
                "run-1",
                gate_id,
                digest_bytes(packet_bytes),
                packet_bytes.decode("utf-8"),
                "2026-08-20T00:00:00.000000Z",
            ),
        )
    connection.commit()
    connection.close()
    return path


def test_real_graphiti_cycle_requires_canonical_shared_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.hermes_control_plane as hermes
    from newsroom.control_plane import paths as control_paths

    anchored_default = Path(hermes.DEFAULT_UNPUBLISHED)
    anchored_proving = Path(hermes.DEFAULT_PROVING)
    assert anchored_default.is_absolute()
    assert anchored_proving.is_absolute()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert Path(hermes.DEFAULT_UNPUBLISHED) == anchored_default
    assert Path(hermes.DEFAULT_PROVING) == anchored_proving

    canonical = tmp_path / "canonical" / "unpublished.sqlite3"
    canonical.parent.mkdir()
    canonical.touch()
    alias = tmp_path / "canonical-alias.sqlite3"
    alias.symlink_to(canonical)
    canonical_proving = tmp_path / "canonical" / "proving.sqlite3"
    canonical_proving.touch()
    proving_alias = tmp_path / "proving-alias.sqlite3"
    proving_alias.symlink_to(canonical_proving)
    monkeypatch.setattr(control_paths, "CANONICAL_PROVING_STORE", canonical_proving)
    monkeypatch.setattr(control_paths, "CANONICAL_UNPUBLISHED_STORE", canonical)
    control_paths.require_canonical_proving_store(str(proving_alias))
    control_paths.require_canonical_unpublished_store(str(alias))
    with pytest.raises(ValueError, match="canonical proving store"):
        control_paths.require_canonical_proving_store(
            str(tmp_path / "other-proving.sqlite3")
        )
    with pytest.raises(ValueError, match="canonical unpublished store"):
        control_paths.require_canonical_unpublished_store(
            str(tmp_path / "other.sqlite3")
        )


def test_control_plane_state_root_preserves_legacy_pair_or_bootstraps_fresh(
    tmp_path: Path,
) -> None:
    from newsroom.control_plane.paths import (
        _canonical_state_root,
        _ensure_private_directory,
    )

    fresh_home = tmp_path / "fresh"
    assert _canonical_state_root(fresh_home) == (
        fresh_home / ".local" / "share" / "newsroom"
    )
    for store_name in ("proving_store.sqlite3", "unpublished_store.sqlite3"):
        intermediate_home = tmp_path / f"fresh-{store_name}"
        intermediate = intermediate_home / ".local" / "share" / "newsroom"
        intermediate.mkdir(parents=True)
        (intermediate / store_name).touch()
        assert _canonical_state_root(intermediate_home) == intermediate

    legacy = tmp_path / "legacy" / "Coding" / "newsroom" / "data" / "newsroom"
    legacy.mkdir(parents=True)
    (legacy / "proving_store.sqlite3").write_bytes(b"proving")
    with pytest.raises(RuntimeError, match="store pair is incomplete"):
        _canonical_state_root(tmp_path / "legacy")
    (legacy / "unpublished_store.sqlite3").write_bytes(b"unpublished")
    assert _canonical_state_root(tmp_path / "legacy") == legacy

    conflict_home = tmp_path / "conflict"
    conflict_legacy = conflict_home / "Coding" / "newsroom" / "data" / "newsroom"
    conflict_fresh = conflict_home / ".local" / "share" / "newsroom"
    conflict_legacy.mkdir(parents=True)
    conflict_fresh.mkdir(parents=True)
    for name in ("proving_store.sqlite3", "unpublished_store.sqlite3"):
        (conflict_legacy / name).write_bytes(b"legacy")
        (conflict_fresh / name).write_bytes(b"fresh")
    with pytest.raises(RuntimeError, match="ambiguous authority"):
        _canonical_state_root(conflict_home)

    mixed_home = tmp_path / "legacy-with-fresh-intermediate"
    mixed_legacy = mixed_home / "Coding" / "newsroom" / "data" / "newsroom"
    mixed_fresh = mixed_home / ".local" / "share" / "newsroom"
    mixed_legacy.mkdir(parents=True)
    mixed_fresh.mkdir(parents=True)
    for name in ("proving_store.sqlite3", "unpublished_store.sqlite3"):
        (mixed_legacy / name).touch()
    (mixed_fresh / "proving_store.sqlite3").touch()
    with pytest.raises(RuntimeError, match="refusing a split authority"):
        _canonical_state_root(mixed_home)

    for name in ("proving_store.sqlite3", "unpublished_store.sqlite3"):
        (conflict_fresh / name).unlink()
        (conflict_fresh / name).symlink_to(conflict_legacy / name)
    assert _canonical_state_root(conflict_home) == conflict_legacy

    private_root = tmp_path / "fresh-private"
    _ensure_private_directory(private_root)
    assert stat.S_IMODE(private_root.stat().st_mode) == 0o700


def test_default_clis_bootstrap_fresh_stores_across_two_invocations(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "HOME": str(tmp_path)}
    fresh = tmp_path / ".local" / "share" / "newsroom"

    status = subprocess.run(
        [sys.executable, "scripts/hermes_control_plane.py", "status"],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert status.returncode == 0, status.stderr
    assert (fresh / "unpublished_store.sqlite3").exists()
    assert not (fresh / "proving_store.sqlite3").exists()

    proving_list = subprocess.run(
        [sys.executable, "scripts/increment9_proving_store.py", "list"],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proving_list.returncode == 0, proving_list.stderr
    assert (fresh / "proving_store.sqlite3").exists()


def test_real_graphiti_run_cycle_rejects_alternate_authority_or_ledger_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from newsroom.control_plane import paths as control_paths
    from newsroom.control_plane.graphiti import EvaluationGraphitiRunner

    canonical_proving = tmp_path / "canonical-proving.sqlite3"
    canonical_unpublished = tmp_path / "canonical-unpublished.sqlite3"
    monkeypatch.setattr(control_paths, "CANONICAL_PROVING_STORE", canonical_proving)
    monkeypatch.setattr(
        control_paths, "CANONICAL_UNPUBLISHED_STORE", canonical_unpublished
    )
    runner = EvaluationGraphitiRunner()

    with pytest.raises(ValueError, match="canonical proving store"):
        run_cycle(
            proving_store=str(tmp_path / "stale-pass.sqlite3"),
            unpublished_store=str(canonical_unpublished),
            writer=FixtureWriter(),
            graphiti=runner,
        )
    with pytest.raises(ValueError, match="canonical unpublished store"):
        run_cycle(
            proving_store=str(canonical_proving),
            unpublished_store=str(tmp_path / "alternate-ledger.sqlite3"),
            writer=FixtureWriter(),
            graphiti=runner,
        )


def test_veto_refuses_public_intents() -> None:
    with pytest.raises(VetoError, match="AUTO_PUBLISH"):
        refuse_public_effect("AUTO_PUBLISH")
    with pytest.raises(VetoError, match="IAP"):
        refuse_public_effect("IAP")
    refuse_public_effect("PRIVATE_CYCLE_START")


def test_parsers_extract_atom_rss_and_json() -> None:
    atom = parse_observation(
        source_id="UK-01", url="https://www.gov.uk/search/all.atom", body=ATOM
    )
    assert atom[0].headline == "Home Office update"
    assert atom[0].canonical_url == "https://www.gov.uk/example-1"
    rss = parse_observation(
        source_id="HK-01",
        url="https://www.news.gov.hk/tc/common/html/topstories.rss.xml",
        body=RSS,
    )
    assert rss[0].headline == "香港政府新聞"
    json_item = parse_observation(
        source_id="UK-02",
        url="https://www.gov.uk/api/content/british-national-overseas-bno-visa",
        body=JSON_DOC,
    )
    assert json_item[0].headline == "BNO visa"
    assert json_item[0].canonical_url.endswith("/british-national-overseas-bno-visa")


def test_cycle_mints_unpublished_payloads_with_evidence(tmp_path: Path) -> None:
    proving = _proving(tmp_path)
    unpublished = tmp_path / "unpublished_store.sqlite3"
    writer = FixtureWriter()
    first = run_cycle(
        proving_store=str(proving),
        unpublished_store=str(unpublished),
        writer=writer,
        max_writes=10,
    )
    assert first.proving_run_id == "run-1"
    assert first.minted == 3
    assert first.duplicate == 0
    assert first.candidates == 3
    assert first.writer_id == writer.writer_id
    payloads = list_payloads(str(unpublished))
    assert len(payloads) == 3
    assert {item.status for item in payloads} == {"UNPUBLISHED"}
    assert {item.language for item in payloads} == {"ZH_HANT_HK"}
    assert all(item.evidence_package_digest.startswith("sha256:") for item in payloads)
    assert all(item.publication_bundle is False for item in payloads)
    assert all(item.auto_publish is False for item in payloads)
    assert all(not item.body.startswith("London —") for item in payloads)
    second = run_cycle(
        proving_store=str(proving),
        unpublished_store=str(unpublished),
        writer=writer,
        max_writes=10,
    )
    assert second.minted == 0
    assert second.duplicate == 3
    assert len(list_payloads(str(unpublished))) == 3


def test_cycle_runs_configured_graphiti_admission_atom(tmp_path: Path) -> None:
    proving = _proving(tmp_path)
    unpublished = tmp_path / "unpublished_store.sqlite3"
    calls: list[object] = []

    class AdmissionAtom:
        def enqueue_complete_receipts(self) -> int:
            calls.append("enqueue")
            return 0

        def drain(self, *, worker_id: str, limit: int = 100) -> object:
            calls.append(("drain", worker_id, limit))
            return object()

        def reconcile_rights(self, *, limit: int = 100) -> int:
            calls.append(("rights", limit))
            return 0

    def admission_factory(connection: object) -> AdmissionAtom:
        assert connection is not None
        calls.append("factory")
        return AdmissionAtom()

    run_cycle(
        proving_store=str(proving),
        unpublished_store=str(unpublished),
        writer=FixtureWriter(),
        max_writes=0,
        graphiti_admission_factory=admission_factory,
        max_graphiti_admissions=7,
    )

    assert calls == [
        "factory",
        "enqueue",
        ("drain", "hermes-cycle:run-1", 7),
        ("rights", 7),
    ]


def test_cycle_stops_backlog_walk_after_unknown_writer_failure(tmp_path: Path) -> None:
    proving = _proving(tmp_path)
    unpublished = tmp_path / "unpublished_store.sqlite3"

    class FlakyWriter:
        writer_id = "test-flaky-writer"

        def __init__(self) -> None:
            self.calls = 0

        def write(
            self, candidate: StoryCandidateRecord, package: EvidencePackage
        ) -> WriterCopy:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("grok writer timed out")
            return FixtureWriter().write(candidate, package)

    writer = FlakyWriter()
    report = run_cycle(
        proving_store=str(proving),
        unpublished_store=str(unpublished),
        writer=writer,
        max_writes=10,
    )
    assert writer.calls == 1
    assert report.minted == 0
    assert report.candidates == 3
    assert report.candidate_attempts == 1
    assert report.no_useful_output_circuit_open is True
    payloads = list_payloads(str(unpublished))
    assert payloads == ()


def test_same_event_url_consolidates_to_one_candidate(tmp_path: Path) -> None:
    proving = _proving(tmp_path, extra=(("RAD-01", SAME_URL_RSS),))
    unpublished = tmp_path / "unpublished_store.sqlite3"
    report = run_cycle(
        proving_store=str(proving),
        unpublished_store=str(unpublished),
        writer=FixtureWriter(),
        max_writes=10,
    )
    assert report.candidates == 3
    payloads = list_payloads(str(unpublished))
    home = [item for item in payloads if "UK-01" in item.source_lineage]
    assert len(home) == 1
    assert "RAD-01" in home[0].source_lineage


def test_cycle_reserves_graphiti_spend_before_stub_extract(tmp_path: Path) -> None:
    from newsroom.control_plane.graphiti import GraphitiCycleResult

    proving = _proving(tmp_path)
    unpublished = tmp_path / "unpublished_store.sqlite3"
    calls: list[str] = []

    class StubGraphiti:
        def ingest(self, unit: CorpusIngestUnit) -> GraphitiCycleResult:
            calls.append(unit.ingest_id)
            return GraphitiCycleResult(
                ingest_id=unit.ingest_id,
                source_id=unit.source_id,
                item_key=unit.item_key,
                outcome="COMPLETE",
                proposal_count=2,
                entity_count=2,
                relation_count=0,
                failure_code="NONE",
                temporal_basis=OBSERVED_FALLBACK,
                reference_time=unit.observed_at,
                attempt_number=unit.attempt_number,
                provider_attempt_number=unit.attempt_number,
            )

    first = run_cycle(
        proving_store=str(proving),
        unpublished_store=str(unpublished),
        writer=FixtureWriter(),
        max_writes=10,
        graphiti=StubGraphiti(),
        max_graphiti=1,
    )
    assert first.graphiti == 1
    assert first.minted == 3
    assert len(calls) == 1
    connection = __import__("sqlite3").connect(unpublished)
    kinds = [
        row[0] for row in connection.execute("SELECT kind FROM ledger ORDER BY seq")
    ]
    connection.close()
    assert kinds.count("GRAPHITI_SPEND_RESERVE") == 1
    assert kinds.count("GRAPHITI_SPEND_RECONCILE") == 1
    assert kinds.count("GRAPHITI_EVALUATION_ATTEMPT") == 1
    assert kinds.index("GRAPHITI_SPEND_RESERVE") < kinds.index(
        "GRAPHITI_EVALUATION_ATTEMPT"
    )
    second = run_cycle(
        proving_store=str(proving),
        unpublished_store=str(unpublished),
        writer=FixtureWriter(),
        max_writes=10,
        graphiti=StubGraphiti(),
        max_graphiti=1,
    )
    assert second.graphiti == 1
    assert len(calls) == 2
    connection = __import__("sqlite3").connect(unpublished)
    kinds = [
        row[0] for row in connection.execute("SELECT kind FROM ledger ORDER BY seq")
    ]
    connection.close()
    assert kinds.count("GRAPHITI_SPEND_RESERVE") == 2
    assert kinds.count("GRAPHITI_SPEND_RECONCILE") == 2
    assert kinds.count("GRAPHITI_EVALUATION_ATTEMPT") == 2


def test_cycle_retries_failed_graphiti_extract(tmp_path: Path) -> None:
    from newsroom.control_plane.graphiti import GraphitiCycleResult
    from newsroom.tests.test_graphiti_corpus_ingest import _complete

    proving = _proving(tmp_path)
    unpublished = tmp_path / "unpublished_store.sqlite3"
    calls: list[str] = []

    class FlakyGraphiti:
        def ingest(self, unit: CorpusIngestUnit) -> GraphitiCycleResult:
            calls.append(unit.ingest_id)
            if len(calls) == 1:
                raise RuntimeError("provider failed before returning a result")
            return _complete(unit)

    first = run_cycle(
        proving_store=str(proving),
        unpublished_store=str(unpublished),
        writer=FixtureWriter(),
        max_writes=10,
        graphiti=FlakyGraphiti(),
        max_graphiti=1,
    )
    assert first.graphiti == 1
    connection = __import__("sqlite3").connect(unpublished)
    stored = connection.execute(
        "SELECT COUNT(*) FROM unpublished_graphiti_ingest"
    ).fetchone()[0]
    connection.close()
    assert stored == 0
    second = run_cycle(
        proving_store=str(proving),
        unpublished_store=str(unpublished),
        writer=FixtureWriter(),
        max_writes=10,
        graphiti=FlakyGraphiti(),
        max_graphiti=1,
    )
    assert second.graphiti == 1
    assert calls[0] == calls[1]
    connection = __import__("sqlite3").connect(unpublished)
    stored = connection.execute(
        "SELECT outcome, proposal_count FROM unpublished_graphiti_ingest"
    ).fetchone()
    connection.close()
    assert stored == ("COMPLETE", 1)


def test_surface_payload_refuses_dateline_dump_and_publication_bundle() -> None:
    with pytest.raises(ValueError, match="dateline dump"):
        UnpublishedSurfacePayload(
            payload_kind="unpublished_surface_payload",
            publication_bundle=False,
            auto_publish=False,
            language="ZH_HANT_HK",
            title="x",
            body="London — Home Office and UKVI, 2026-08-19 — copied",
            evidence_package_digest="sha256:" + ("a" * 64),
            story_candidate_id="c",
            event_hypothesis_id="h",
            source_lineage=("UK-01",),
            generated_at="2026-08-20T00:00:00.000000Z",
            status="UNPUBLISHED",
            writer_id="w",
        )
    with pytest.raises(VetoError, match="PUBLICATION_BUNDLE"):
        UnpublishedSurfacePayload(
            payload_kind="unpublished_surface_payload",
            publication_bundle=True,
            auto_publish=False,
            language="ZH_HANT_HK",
            title="x",
            body="【未出版原創評核稿】正文",
            evidence_package_digest="sha256:" + ("a" * 64),
            story_candidate_id="c",
            event_hypothesis_id="h",
            source_lineage=("UK-01",),
            generated_at="2026-08-20T00:00:00.000000Z",
            status="UNPUBLISHED",
            writer_id="w",
        )


def test_store_refuses_production_alias_and_public_dispatch(tmp_path: Path) -> None:
    with pytest.raises(VetoError, match="production"):
        connect(str(tmp_path / "production" / "unpublished.sqlite3"))
    unpublished = tmp_path / "unpublished_store.sqlite3"
    connection = connect(str(unpublished))
    with pytest.raises(VetoError, match="PUBLIC_DISPATCH"):
        mark_public_dispatch(connection, "missing")


def test_news_report_dateline_is_not_the_cycle_writer() -> None:
    items = parse_observation(
        source_id="UK-01", url="https://www.gov.uk/search/all.atom", body=ATOM
    )
    report = news_report(items[0], observed_at="2026-08-19T22:00:00.000000Z")
    assert report.startswith("London — Home Office and UKVI, 2026-08-19 — ")


def test_evaluation_packet_authorises_evaluation_and_refuses_production() -> None:
    from newsroom.authority.canonical import digest_canonical
    from newsroom.extraction.types import VersionedExtractionComponent
    from newsroom.graphiti_adapter.contracts import (
        GRAPHITI_ADAPTER_CODE_COMPONENT,
        GRAPHITI_ADAPTER_NORMALISATION_COMPONENT,
        GRAPHITI_ADAPTER_OUTPUT_SCHEMA_COMPONENT,
        GRAPHITI_ADAPTER_POLICY_COMPONENT,
        GRAPHITI_ADAPTER_TEMPORAL_COMPONENT,
        GRAPHITI_PROMPT_COMPONENT,
    )
    from newsroom.graphiti_adapter.models import GraphitiAdapterConfiguration
    from newsroom.graphiti_adapter.types import (
        GraphitiExecutionProfile,
        GraphitiRuntimeMode,
    )
    from newsroom.tests.extraction_4a_helpers import contract_request
    from newsroom.tests.graphiti_adapter_4d_helpers import FAKE_CONFIGURATION_ID

    assert REAL_GRAPHITI_RUNTIME_ENABLED is True
    assert GRAPHITI_CORE_RELEASE == "graphiti-core-0.29.3"
    assert OPENROUTER_API == "OPENROUTER_API"
    assert GRAPHITI_CHAT_MODEL == "cursor-agent-cli:composer-2.5"
    assert GRAPHITI_EMBEDDING_MODEL == "openrouter:openai.text-embedding-3-large"
    assert WRITER_MODEL == "grok-build-cli:grok-4.6"
    assert WRITER_FALLBACK == "cursor-agent-cli"
    assert EVALUATION_GRAPHITI_PACKET.model_release == GRAPHITI_CHAT_MODEL
    assert "placeholder" not in EVALUATION_GRAPHITI_PACKET.framework_release
    assert (
        EVALUATION_WORKSPACE_POLICY.egress_policy
        is GraphitiEgressPolicy.APPROVED_PROVIDER_ONLY
    )
    assert (
        EVALUATION_WORKSPACE_POLICY.credential_class
        is GraphitiCredentialClass.PROPOSAL_WORKSPACE_ONLY
    )
    assert EVALUATION_GRAPHITI_PACKET.framework_release == GRAPHITI_CORE_RELEASE
    contract = contract_request()
    digest = digest_canonical({"contract": "evaluation-component"})
    configuration = GraphitiAdapterConfiguration(
        configuration_id=FAKE_CONFIGURATION_ID,
        runtime_mode=GraphitiRuntimeMode.REAL_GRAPHITI,
        execution_profile=GraphitiExecutionProfile.EVALUATION,
        framework=VersionedExtractionComponent(
            "graphiti.framework", GRAPHITI_CORE_RELEASE, digest
        ),
        model=VersionedExtractionComponent(
            "graphiti.model", GRAPHITI_CHAT_MODEL, digest
        ),
        embedding=VersionedExtractionComponent(
            "graphiti.embedding", GRAPHITI_EMBEDDING_MODEL, digest
        ),
        prompt=GRAPHITI_PROMPT_COMPONENT,
        output_schema=GRAPHITI_ADAPTER_OUTPUT_SCHEMA_COMPONENT,
        code=GRAPHITI_ADAPTER_CODE_COMPONENT,
        normalisation=GRAPHITI_ADAPTER_NORMALISATION_COMPONENT,
        temporal_policy=GRAPHITI_ADAPTER_TEMPORAL_COMPONENT,
        adapter_policy=GRAPHITI_ADAPTER_POLICY_COMPONENT,
        extractor_contract_id=contract.contract_id,
        extractor_contract_digest=contract.digest,
        workspace_policy=EVALUATION_WORKSPACE_POLICY,
        fixture_case=None,
        real_runtime_authority=EVALUATION_GRAPHITI_PACKET,
        idempotency_key="evaluation-packet-authorises-evaluation",
    )
    configuration.require_execution_authorized()
    production = GraphitiAdapterConfiguration(
        configuration_id=FAKE_CONFIGURATION_ID,
        runtime_mode=GraphitiRuntimeMode.REAL_GRAPHITI,
        execution_profile=GraphitiExecutionProfile.PRODUCTION,
        framework=VersionedExtractionComponent(
            "graphiti.framework", GRAPHITI_CORE_RELEASE, digest
        ),
        model=VersionedExtractionComponent(
            "graphiti.model", GRAPHITI_CHAT_MODEL, digest
        ),
        embedding=VersionedExtractionComponent(
            "graphiti.embedding", GRAPHITI_EMBEDDING_MODEL, digest
        ),
        prompt=GRAPHITI_PROMPT_COMPONENT,
        output_schema=GRAPHITI_ADAPTER_OUTPUT_SCHEMA_COMPONENT,
        code=GRAPHITI_ADAPTER_CODE_COMPONENT,
        normalisation=GRAPHITI_ADAPTER_NORMALISATION_COMPONENT,
        temporal_policy=GRAPHITI_ADAPTER_TEMPORAL_COMPONENT,
        adapter_policy=GRAPHITI_ADAPTER_POLICY_COMPONENT,
        extractor_contract_id=contract.contract_id,
        extractor_contract_digest=contract.digest,
        workspace_policy=EVALUATION_WORKSPACE_POLICY,
        fixture_case=None,
        real_runtime_authority=EVALUATION_GRAPHITI_PACKET,
        idempotency_key="evaluation-packet-refuses-production",
    )
    with pytest.raises(GraphitiRuntimeNotAuthorized, match="EVALUATION"):
        production.require_execution_authorized()


def test_intake_fetches_when_gates_pass(tmp_path: Path) -> None:
    proving = tmp_path / "proving_store.sqlite3"

    def fetch(url: str) -> tuple[int, bytes]:
        if "atom" in url:
            return 200, ATOM
        if url.endswith(".xml") or "rss" in url.lower() or "WarningsRSS" in url:
            return 200, RSS
        return 200, JSON_DOC

    report = run_intake(
        proving_store=str(proving),
        fetch=fetch,
        clock=lambda: datetime.fromisoformat(FIXTURE_NOW.replace("Z", "+00:00")),
    )
    assert report.authorised
    assert report.sources == 10
    assert report.ok == 10
    assert report.health == "ACTIVE"
    assert report.active == 10
    assert report.degraded == 0
    assert report.held == 0
    assert report.blocked == 0
    assert report.proving_run_id.startswith("proving-9p-private-beta-")
    second = run_intake(
        proving_store=str(proving),
        fetch=fetch,
        clock=lambda: datetime.fromisoformat(FIXTURE_NOW.replace("Z", "+00:00")),
    )
    assert second.proving_run_id != report.proving_run_id


@pytest.mark.parametrize(
    ("source_id", "bad_body"),
    [
        ("UK-02", b""),
        ("UK-02", b"{bad"),
        ("UK-01", b"<rss><open"),
        ("UK-01", b"<html><body>error</body></html>"),
    ],
)
def test_intake_degrades_unusable_content_without_global_block(
    tmp_path: Path, source_id: str, bad_body: bytes
) -> None:
    proving = tmp_path / "proving_store.sqlite3"
    target_url = SOURCE_URLS[source_id]
    attempts: dict[str, int] = {}

    def fetch(url: str) -> tuple[int, bytes]:
        attempts[url] = attempts.get(url, 0) + 1
        if url == target_url:
            return 200, bad_body
        if "atom" in url:
            return 200, ATOM
        if url.endswith(".xml") or "rss" in url.lower() or "WarningsRSS" in url:
            return 200, RSS
        return 200, JSON_DOC

    report = run_intake(
        proving_store=str(proving),
        fetch=fetch,
        clock=lambda: datetime.fromisoformat(FIXTURE_NOW.replace("Z", "+00:00")),
    )

    assert report.authorised
    assert report.sources == 10
    assert report.ok == 9
    assert report.health == "DEGRADED"
    assert report.active == 9
    assert report.degraded == 1
    assert report.held == 0
    assert report.blocked == 0
    assert not report.complete
    assert attempts[target_url] == FETCH_MAX_ATTEMPTS


def test_intake_counts_only_usable_200_observations_as_ok(tmp_path: Path) -> None:
    proving = tmp_path / "proving_store.sqlite3"
    target_url = SOURCE_URLS["UK-02"]
    attempts: dict[str, int] = {}

    def fetch(url: str) -> tuple[int, bytes]:
        attempts[url] = attempts.get(url, 0) + 1
        if url == target_url and attempts[url] < 3:
            return 200, b"{bad"
        if "atom" in url:
            return 200, ATOM
        if url.endswith(".xml") or "rss" in url.lower() or "WarningsRSS" in url:
            return 200, RSS
        return 200, JSON_DOC

    report = run_intake(
        proving_store=str(proving),
        fetch=fetch,
        clock=lambda: datetime.fromisoformat(FIXTURE_NOW.replace("Z", "+00:00")),
    )

    assert report.authorised
    assert report.ok == 10
    assert report.complete
    assert report.health == "ACTIVE"
    assert report.active == 10
    assert report.degraded == 0
    assert attempts[target_url] == 3


def _sample_candidate_package() -> tuple[StoryCandidateRecord, EvidencePackage]:
    from newsroom.control_plane.editorial import (
        DiscoverySignalRecord,
        NewsLeadRecord,
        StoryCandidateRecord,
    )
    from newsroom.control_plane.evidence import EvidencePackage

    candidate = StoryCandidateRecord(
        candidate_id="c1",
        hypothesis_id="h1",
        headline="Home Office update",
        items=parse_observation(
            source_id="UK-01", url="https://www.gov.uk/search/all.atom", body=ATOM
        ),
        signals=(
            DiscoverySignalRecord(
                signal_id="s1",
                source_id="UK-01",
                item_key="k",
                observation_digest="sha256:" + ("b" * 64),
            ),
        ),
        leads=(
            NewsLeadRecord(lead_id="l1", signal_id="s1", headline="Home Office update"),
        ),
    )
    package = EvidencePackage(
        candidate_id="c1",
        hypothesis_id="h1",
        signal_ids=("s1",),
        lead_ids=("l1",),
        source_ids=("UK-01",),
        observation_digests=("sha256:" + ("b" * 64),),
        passages=("UK-01: retained",),
    )
    return candidate, package


def test_cli_writer_uses_grok_then_falls_back_to_cursor_agent() -> None:
    def grok(_prompt: str) -> str:
        raise RuntimeError("grok writer failed")

    def cursor(_prompt: str) -> str:
        return json.dumps(
            {
                "title": "【未出版】測試稿",
                "body": "【未出版原創】根據證據包改寫，唔係來源標題複本。",
            }
        )

    copy = CliChainWriter(primary=grok, fallback=cursor).write(
        *_sample_candidate_package()
    )
    assert copy.writer_id == "cursor-agent-cli-cont-writer"
    assert copy.title.startswith("【未出版】")


def test_cli_writer_prefers_grok_build_cli() -> None:
    def grok(_prompt: str) -> str:
        return json.dumps(
            {
                "text": "",
                "structured_output": {
                    "title": "【未出版】Grok稿",
                    "body": "【未出版原創】Grok Build CLI 正文。",
                },
            }
        )

    def cursor(_prompt: str) -> str:
        raise AssertionError("fallback must not run")

    copy = CliChainWriter(primary=grok, fallback=cursor).write(
        *_sample_candidate_package()
    )
    assert copy.writer_id == "grok-build-cli-cont-writer"
    assert "Grok" in copy.title


def test_cli_writer_rejects_planning_residue_and_falls_back() -> None:
    def grok(_prompt: str) -> str:
        return json.dumps(
            {
                "title": "正在核實幼稚園收生與家長智Net來源",
                "body": "先查 CONT 記者稿例。",
            }
        )

    def cursor(_prompt: str) -> str:
        return json.dumps(
            {
                "title": "【未出版】立法會教育議案",
                "body": "【未出版原創】立法會通過教育議案，本報根據證據包改寫。",
            },
            ensure_ascii=False,
        )

    copy = CliChainWriter(primary=grok, fallback=cursor).write(
        *_sample_candidate_package()
    )
    assert copy.writer_id == "cursor-agent-cli-cont-writer"
    assert copy.title == "【未出版】立法會教育議案"
    assert not copy.title.startswith("正在")
    assert not copy.body.startswith("先查")


def test_cli_writer_accepts_finished_copy_that_mentions_verification() -> None:
    def grok(_prompt: str) -> str:
        return json.dumps(
            {
                "title": "英國入境規則公開更新",
                "body": "申請人須按現行版本核實資格，本報根據證據包改寫。",
            },
            ensure_ascii=False,
        )

    def cursor(_prompt: str) -> str:
        raise AssertionError("fallback must not run")

    copy = CliChainWriter(primary=grok, fallback=cursor).write(
        *_sample_candidate_package()
    )
    assert copy.writer_id == "grok-build-cli-cont-writer"
    assert "核實資格" in copy.body


def test_grok_cli_uses_hermetic_single_turn_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "https://auth.x.ai::fixture": {
                    "auth_mode": "oauth",
                    "expires_at": "2099-01-01T00:00:00Z",
                    "key": "fixture-access-token",
                    "refresh_token": "fixture-refresh-token",
                }
            }
        )
    )
    auth_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    monkeypatch.setattr("newsroom.control_plane.writer.GROK_AUTH_FILE", str(auth_file))
    monkeypatch.setenv("AMBIENT_WRITER_CANARY", "must-not-enter")
    calls: list[dict[str, object]] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command: tuple[str, ...], **kwargs: object) -> Result:
        result = Result()
        cwd = kwargs.get("cwd")
        environment = kwargs.get("env")
        home = (
            Path(environment["HOME"])
            if isinstance(environment, dict)
            and isinstance(environment.get("HOME"), str)
            else None
        )
        calls.append(
            {
                "command": command,
                "timeout": kwargs.get("timeout"),
                "cwd": cwd,
                "entries": os.listdir(cwd) if isinstance(cwd, str) else [],
                "environment": dict(environment) if isinstance(environment, dict) else {},
                "grok_home_entries": (
                    sorted(path.name for path in (home / ".grok").iterdir())
                    if home is not None and (home / ".grok").is_dir()
                    else []
                ),
            }
        )
        if "inspect" in command:
            result.stdout = json.dumps(
                {
                    "grokVersion": "1.0.8",
                    "projectRoot": None,
                    "projectInstructions": [],
                    "permissions": {"loaded": 0},
                    "skills": [],
                    "plugins": [],
                    "marketplaces": [],
                    "mcpServers": [],
                    "hooks": [],
                    "lspServers": [],
                    "configSources": {"layers": []},
                }
            )
        elif "version" in command:
            result.stdout = "grok 1.0.8 (fixture) [alpha]\n"
        else:
            result.stdout = json.dumps({"title": "t", "body": "b"})
        return result

    monkeypatch.setattr("newsroom.control_plane.writer.subprocess.run", fake_run)
    run_grok_cli("prompt")
    provider_call = calls[-1]
    command = provider_call["command"]
    assert isinstance(command, tuple)
    assert provider_call["timeout"] == 300
    assert command[command.index("--max-turns") + 1] == "1"
    assert command[command.index("--reasoning-effort") + 1] == "low"
    assert "--json-schema" in command
    assert "--system-prompt-override" in command
    assert "--verbatim" in command
    assert "--no-plan" in command
    assert "--disable-web-search" in command
    assert "--no-subagents" in command
    assert "--disallowed-tools" in command
    denied = command[command.index("--disallowed-tools") + 1]
    assert isinstance(denied, str)
    assert "run_terminal_command" in denied.split(",")
    assert "spawn_subagent" in denied.split(",")
    assert "search_tool" in denied.split(",")
    assert "--always-approve" not in command
    assert "--force" not in command
    assert "--yolo" not in command
    assert provider_call["entries"] == []
    environment = provider_call["environment"]
    assert isinstance(environment, dict)
    assert "AMBIENT_WRITER_CANARY" not in environment
    assert set(environment) == {
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
    }
    isolated_home = Path(environment["HOME"])
    assert isolated_home != Path.home()
    assert provider_call["grok_home_entries"] == ["auth.json"]
    cwd = provider_call["cwd"]
    assert isinstance(cwd, str)
    assert cwd != os.getcwd()


def test_grok_writer_parser_reads_1_0_10_streaming_json_text_events() -> None:
    from newsroom.control_plane.writer import _parse_grok_writer_output

    copy = json.dumps(
        {
            "title": "【未出版】官方公布咗最新安排",
            "body": "本報根據已核實證據報道：相關官方資料確認安排已經更新",
            "evidence_links": [],
        },
        ensure_ascii=False,
    )
    raw = "\n".join(
        (
            json.dumps({"type": "text", "data": copy}),
            json.dumps(
                {
                    "type": "end",
                    "stopReason": "end_turn",
                    "usage": {
                        "input_tokens": 800,
                        "output_tokens": 40,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "reasoning_tokens": 0,
                        "total_tokens": 840,
                        "context_tokens": 900,
                    },
                }
            ),
        )
    )
    execution = _parse_grok_writer_output(raw)
    assert json.loads(execution.text)["title"].startswith("【未出版】")
    assert execution.usage["usage_basis"] == "PROVIDER_REPORTED"
    assert execution.usage["input_tokens"] == 800
    assert execution.usage["context_tokens"] == 900


def test_grok_writer_parser_reads_json_schema_wrapper() -> None:
    from newsroom.control_plane.writer import _parse_grok_writer_output

    raw = json.dumps(
        {
            "text": "ignored prose",
            "structured_output": {
                "title": "【未出版】官方公布咗最新安排",
                "body": "本報根據已核實證據報道：相關官方資料確認安排已經更新",
                "evidence_links": [],
            },
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
        ensure_ascii=False,
    )
    execution = _parse_grok_writer_output(raw)
    assert json.loads(execution.text)["title"].startswith("【未出版】")
    assert execution.usage["usage_basis"] == "PROVIDER_REPORTED"
    assert execution.usage["total_tokens"] == 120
    assert execution.usage["context_tokens"] == 100


def test_grok_cli_accepts_matching_unpinned_semantic_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from newsroom.control_plane.writer import run_grok_cli

    monkeypatch.setattr(
        "newsroom.control_plane.writer.read_grok_command_semantic_version",
        lambda: "1.0.10",
    )
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "https://auth.x.ai::fixture": {
                    "auth_mode": "oauth",
                    "expires_at": "2099-01-01T00:00:00Z",
                    "key": "fixture-access-token",
                    "refresh_token": "fixture-refresh-token",
                }
            }
        ),
        encoding="utf-8",
    )
    auth_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    monkeypatch.setattr("newsroom.control_plane.writer.GROK_AUTH_FILE", str(auth_file))

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command: tuple[str, ...], **kwargs: object) -> Result:
        del kwargs
        result = Result()
        if "inspect" in command:
            result.stdout = json.dumps(
                {
                    "grokVersion": "1.0.10",
                    "projectRoot": None,
                    "projectInstructions": [],
                    "permissions": {"loaded": 0},
                    "skills": [],
                    "plugins": [],
                    "marketplaces": [],
                    "mcpServers": [],
                    "hooks": [],
                    "lspServers": [],
                    "configSources": {"layers": []},
                }
            )
        elif "version" in command:
            result.stdout = "grok 1.0.10 (fixture) [alpha]\n"
        else:
            result.stdout = json.dumps({"title": "t", "body": "b"})
        return result

    monkeypatch.setattr("newsroom.control_plane.writer.subprocess.run", fake_run)
    execution = run_grok_cli("prompt")
    assert "title" in execution.text


def test_grok_cli_rejects_inspect_command_version_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from newsroom.control_plane.writer import WriterDispatchError, run_grok_cli

    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "https://auth.x.ai::fixture": {
                    "auth_mode": "oauth",
                    "expires_at": "2099-01-01T00:00:00Z",
                    "key": "fixture-access-token",
                    "refresh_token": "fixture-refresh-token",
                }
            }
        ),
        encoding="utf-8",
    )
    auth_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    monkeypatch.setattr("newsroom.control_plane.writer.GROK_AUTH_FILE", str(auth_file))

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command: tuple[str, ...], **kwargs: object) -> Result:
        del kwargs
        result = Result()
        if "inspect" in command:
            result.stdout = json.dumps(
                {
                    "grokVersion": "1.0.8",
                    "projectRoot": None,
                    "projectInstructions": [],
                    "permissions": {"loaded": 0},
                    "skills": [],
                    "plugins": [],
                    "marketplaces": [],
                    "mcpServers": [],
                    "hooks": [],
                    "lspServers": [],
                    "configSources": {"layers": []},
                }
            )
        elif "version" in command:
            result.stdout = "grok 1.0.10 (fixture) [alpha]\n"
        else:
            result.stdout = json.dumps({"title": "t", "body": "b"})
        return result

    monkeypatch.setattr("newsroom.control_plane.writer.subprocess.run", fake_run)
    with pytest.raises(WriterDispatchError, match="zero-skill") as caught:
        run_grok_cli("prompt")
    assert caught.value.reason_code == "HERMETIC_CAPABILITY_UNPROVABLE"
    assert caught.value.provider_dispatched is False


def test_writer_head_proof_rejects_hidden_index_flags_and_git_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(("/usr/bin/git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(
        ("/usr/bin/git", "config", "user.email", "fixture@example.invalid"),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("/usr/bin/git", "config", "user.name", "Fixture"),
        cwd=repository,
        check=True,
    )
    tracked = repository / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(("/usr/bin/git", "add", "tracked.py"), cwd=repository, check=True)
    subprocess.run(
        ("/usr/bin/git", "commit", "-q", "-m", "fixture"),
        cwd=repository,
        check=True,
    )
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "poison"))

    revision, clean = cont_writer_implementation_identity(str(repository))
    assert len(revision) == 40
    assert clean is True
    monkeypatch.delenv("GIT_DIR")

    subprocess.run(
        ("/usr/bin/git", "update-index", "--assume-unchanged", "tracked.py"),
        cwd=repository,
        check=True,
    )
    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    assert subprocess.run(
        ("/usr/bin/git", "status", "--porcelain"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout == ""

    assert cont_writer_implementation_identity(str(repository)) == (revision, False)


def test_grok_cli_missing_minimal_auth_fails_before_provider_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing-auth.json"
    monkeypatch.setattr("newsroom.control_plane.writer.GROK_AUTH_FILE", str(missing))
    calls = 0

    def fake_run(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("pre-dispatch authentication failure must not run CLI")

    monkeypatch.setattr("newsroom.control_plane.writer.subprocess.run", fake_run)
    with pytest.raises(WriterDispatchError) as caught:
        run_grok_cli("prompt")
    assert caught.value.reason_code == "MINIMAL_AUTHENTICATION_UNAVAILABLE"
    assert caught.value.provider_dispatched is False
    assert calls == 0


def test_cursor_cli_fails_before_dispatch_when_zero_tools_cannot_be_proved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    workspaces: list[tuple[list[str], dict[str, str], bool]] = []

    class Result:
        returncode = 0
        stdout = "Usage: agent [options]\n--mode <mode>\n--sandbox <mode>\n"
        stderr = ""

    def fake_run(command: tuple[str, ...], **kwargs: object) -> Result:
        calls.append(command)
        cwd = kwargs.get("cwd")
        environment = kwargs.get("env")
        assert isinstance(cwd, str)
        assert isinstance(environment, dict)
        home = Path(environment["HOME"])
        workspaces.append((os.listdir(cwd), dict(environment), (home / ".grok").exists()))
        return Result()

    monkeypatch.setattr("newsroom.control_plane.writer.subprocess.run", fake_run)
    with pytest.raises(WriterDispatchError) as caught:
        run_cursor_agent_cli("prompt")
    assert caught.value.reason_code == "HERMETIC_CAPABILITY_UNPROVABLE"
    assert caught.value.provider_dispatched is False
    assert calls == [("/Users/jamesto/.local/bin/cursor-agent", "--help")]
    entries, environment, grok_home_exists = workspaces[0]
    assert entries == []
    assert Path(environment["HOME"]) != Path.home()
    assert grok_home_exists is False


def test_cli_writer_reads_title_from_grok_text_envelope() -> None:
    def grok(_prompt: str) -> str:
        return json.dumps(
            {
                "text": json.dumps(
                    {"title": "【未出版】信封稿", "body": "【未出版原創】text 欄。"},
                    ensure_ascii=False,
                )
            }
        )

    def cursor(_prompt: str) -> str:
        raise AssertionError("fallback must not run")

    copy = CliChainWriter(primary=grok, fallback=cursor).write(
        *_sample_candidate_package()
    )
    assert copy.title == "【未出版】信封稿"


def test_broker_error_does_not_include_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from newsroom.control_plane import broker

    class Result:
        returncode = 1
        stdout = "super-secret-openrouter-key\n"
        stderr = "not found"

    monkeypatch.setattr(
        broker.subprocess,
        "run",
        lambda *args, **kwargs: Result(),
    )
    with pytest.raises(broker.BrokerError, match="Keychain class OPENROUTER_API lookup unavailable") as caught:
        broker.openrouter_api_key()
    assert "super-secret-openrouter-key" not in str(caught.value)


def test_keychain_timeout_is_a_broker_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from newsroom.control_plane import broker

    def timeout(*_args: object, **_values: object) -> object:
        raise broker.subprocess.TimeoutExpired("security", 10)

    monkeypatch.setattr(broker.subprocess, "run", timeout)
    with pytest.raises(broker.BrokerError, match="OPENROUTER_API lookup timed out"):
        broker.openrouter_api_key()
