from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest

from newsroom.increment9.proving import _store_body

from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.control_plane.admission import (
    _EARLIEST_CURRENT_SHAPE_WRITE_ADMISSION_POLICY_VERSION,
    _LATEST_LEGACY_WRITE_ADMISSION_POLICY_VERSION,
    _OLDEST_WRITE_ADMISSION_POLICY_VERSION,
    _EARLIER_WRITE_ADMISSION_POLICY_VERSION,
    _PENULTIMATE_WRITE_ADMISSION_POLICY_VERSION,
    _PREVIOUS_WRITE_ADMISSION_POLICY_VERSION,
    WRITE_ADMISSION_POLICY_VERSION,
    DeterministicWriteAdmission,
    WriteAdmissionDecision,
    WriteSelectionRecord,
    _decision_id,
    _duration_is_exactly_supported,
    select_write_ready,
)
from newsroom.control_plane.child_environment import unprivileged_child_environment
from newsroom.control_plane.cycle import run_cycle
from newsroom.control_plane.drafting import DraftOutcomeRecord
from newsroom.control_plane.editorial import (
    DiscoverySignalRecord,
    GroupedObservation,
    NewsLeadRecord,
    StoryCandidateRecord,
    form_candidates,
)
from newsroom.control_plane.evidence import (
    EVID_012_POLICY_VERSION,
    EVIDENCE_APPROVAL_POLICY_VERSION,
    EVIDENCE_GATE_POLICY_VERSION,
    FACTUAL_LOCALISATION_POLICY_VERSION,
    GOVERNED_CLAIM_POLICY_VERSION,
    GOVERNED_INPUT_SCHEMA_VERSION,
    NAMED_ENTITY_POLICY_VERSION,
    ORIGINALITY_POLICY_VERSION,
    ClaimAuthorityClass,
    Evid012QualificationTest,
    EvidenceGateEvidence,
    EvidencePackage,
    GovernedClaimEvidence,
    GovernedClaimStatus,
    QualificationEvidence,
    _has_valid_origin_independence,
    _resolve_governed_records,
    bounded_named_entities,
    package_for,
)
from newsroom.control_plane.items import SourceItem, parse_observation
from newsroom.control_plane.model_usage import (
    InvocationEfficiencyPolicy,
    ModelUsageAdmissionError,
    ModelUsageService,
    WorkloadClass,
)
from newsroom.control_plane.store import (
    LEDGER_GENESIS,
    append_ledger,
    connect,
    list_payloads,
    reserve_write_candidate_attempt,
    reserve_writer_provider_attempt,
    retain_draft_outcome,
    retain_write_admission_decision,
    retain_write_selection,
)
from newsroom.control_plane.veto import VetoError
from newsroom.control_plane.writer import (
    CONT_CONTEXT_MANIFEST_SCHEMA_VERSION,
    CONT_DISABLED_CAPABILITIES,
    CONT_FALLBACK_COMMAND_FLAGS,
    CONT_FALLBACK_MODEL,
    CONT_FALLBACK_CONFIG_IDENTITY,
    CONT_FALLBACK_PROVIDER,
    CONT_FALLBACK_REASONING,
    CONT_FALLBACK_ROUTE,
    CONT_PRIMARY_MODEL,
    CONT_PRIMARY_CONFIG_IDENTITY,
    CONT_PRIMARY_PROVIDER,
    CONT_PRIMARY_REASONING,
    CONT_PRIMARY_ROUTE,
    CONT_PRIMARY_COMMAND_FLAGS,
    CONT_WRITER_CONTEXT_IDENTITY,
    CONT_WRITER_OUTPUT_SCHEMA_DIGEST,
    CONT_WRITER_PROMPT_CONTRACT_VERSION,
    CliChainWriter,
    FixtureWriter,
    WriterCliExecution,
    WriterCopy,
    WriterDispatchError,
    WriterEvidenceLink,
    GROK_COMMAND_SEMANTIC_VERSION,
    CURSOR_COMMAND_SEMANTIC_VERSION,
    required_surface_copy,
    validate_writer_copy,
)
from newsroom.control_plane.zh_hant import (
    ZH_HANT_HK_SHAPE_POLICY_VERSION,
    contains_discourse_filler,
    contains_simplified_variant,
)
from newsroom.effective_revision import retain_observation_revision_first_seen
from newsroom.tests.test_control_plane_private_beta import _proving

_CLOCK = lambda: datetime(2026, 8, 20, tzinfo=UTC)
_WRITER_REVISION = "a" * 40


@pytest.fixture(autouse=True)
def _exact_writer_head(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "newsroom.control_plane.writer.cont_writer_implementation_identity",
        lambda: (_WRITER_REVISION, True),
    )


def test_cont_context_manifest_is_canonical_zero_capability_evidence() -> None:
    candidate, package = _candidate_package()
    manifest = CliChainWriter(
        primary=lambda _prompt: "unused", fallback=lambda _prompt: "unused"
    ).invocation_manifest(candidate, package, route="PRIMARY")
    record = manifest.as_record()
    unsigned = dict(record)
    unsigned.pop("context_manifest_digest")

    assert digest_canonical(unsigned) == manifest.context_manifest_digest
    assert manifest.evidence_package_digest == package.digest
    assert manifest.working_directory_inventory == ()
    assert manifest.prior_message_count == 0
    assert manifest.skill_count == 0
    assert manifest.tool_count == 0
    assert manifest.mcp_server_count == 0
    assert manifest.mcp_tool_count == 0
    assert manifest.tools_enabled is False
    assert manifest.skills_enabled is False
    assert manifest.mcp_enabled is False
    retained = json.dumps(record, ensure_ascii=False)
    assert candidate.headline not in retained
    assert all(passage not in retained for passage in package.passages)


def test_context_manifest_with_one_tool_fails_route_gate(tmp_path: Path) -> None:
    candidate, package = _candidate_package()
    manifest = CliChainWriter(
        primary=lambda _prompt: "unused", fallback=lambda _prompt: "unused"
    ).invocation_manifest(candidate, package, route="PRIMARY")
    record = manifest.as_record()
    record["tool_count"] = 1
    unsigned = dict(record)
    unsigned.pop("context_manifest_digest")
    record["context_manifest_digest"] = digest_canonical(unsigned)

    with pytest.raises(ModelUsageAdmissionError, match="ambient capability"):
        ModelUsageService(str(tmp_path / "usage.sqlite3")).retain_context_manifest(
            record
        )


def test_non_controller_children_do_not_receive_evidence_approval_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "NEWSROOM_EVIDENCE_APPROVAL_KEY",
        "GITHUB_TOKEN",
        "NEO4J_PASSWORD",
        "CUSTOM_SECRET",
    ):
        monkeypatch.setenv(name, "secret" * 8)
    environment = unprivileged_child_environment()
    assert (
        not {
            "NEWSROOM_EVIDENCE_APPROVAL_KEY",
            "GITHUB_TOKEN",
            "NEO4J_PASSWORD",
            "CUSTOM_SECRET",
        }
        & environment.keys()
    )


@pytest.mark.parametrize(
    "text",
    (
        "官方說明最新安排",
        "衛生署與群組公佈安排",
        "人群聚集",
        "群眾散去",
        "衞生署公佈安排",
        "了解安排",
        "高峯期安排",
        "着數安排",
        "穿着校服安排",
        "嘉峯臺最新安排",
        "麪粉供應安排",
        "皇后出席活動",
        "路程全長十公里",
        "當局不會干預安排",
        "內容獲准先至可以公布",
        "平台准許已批准嘅社群進行核查",
        "人才群體公布安排",
        "人工智能高峰會將於周四舉行",
        "三十周年搜集證據安排",
        "人口回流香港",
        "高峰期每周更新查核資料",
        "港人回流英國",
        "山峰地區有暴雨",
        "峰值",
        "巔峰",
        "金融峰會",
        "官方宣布新安排",
        "家具供應增加",
        "數據分布情況",
        "舞台表演最新安排",
        "月台最新安排",
        "港台最新安排",
    ),
)
def test_hong_kong_traditional_variants_are_not_simplified(text: str) -> None:
    assert not contains_simplified_variant(text)


@pytest.mark.parametrize(
    "text",
    (
        "整體而言，官方公布最新安排",
        "簡而言之，官方公布最新安排",
        "總的來說，官方公布最新安排",
        "由此可見，官方公布最新安排",
        "值得留意，官方公布最新安排",
        "說到底，官方公布最新安排",
        "歸根究底，官方公布最新安排",
        "眾所周知，官方公布最新安排",
    ),
)
def test_discourse_filler_is_rejected(text: str) -> None:
    assert contains_discourse_filler(text)


def test_official_programme_terms_have_a_structured_entity_type() -> None:
    for text in ("EUSS", "Universal Credit", "Nationality and Borders Act"):
        assert bounded_named_entities(f"{text} changed") == frozenset(
            {(text, "OFFICIAL_TERM")}
        )


@pytest.mark.parametrize(
    "text",
    (
        "香港政府公布未來安排",
        "香港政府公布交通安排",
        "香港政府公布防疫安排",
        "香港政府公布服務安排",
        "香港政府公布學生安排",
        "香港政府公布弱勢安排",
        "香港政府公布申請安排",
        "香港政府公布房屋安排",
        "香港政府公布就業安排",
        "香港政府公布長者安排",
        "香港政府公布基層安排",
        "香港政府公布醫療安排",
        "香港政府公布福利安排",
        "香港政府公布政策安排",
        "香港政府公布措施安排",
        "香港政府公布計劃安排",
        "方針明確公布安排",
        "方表示支持新安排",
        "政府任命任務安排",
        "政府任命任務安排為項目",
        "機構邀請相關安排",
        "機構邀請方表示",
        "政府公布政策將實施",
        "政府公布措施將推行",
        "政府公布計劃將啟用",
        "政府公布工作小組出任項目角色",
        "政府公布專責小組出任項目角色",
        "政府公布方代表出任項目角色",
        "政府公布新中心將設於初步階段",
        "政府公布新中心將設於測試階段",
        "政府公布新中心將設於學校附近",
        "政府公布新中心將設於社會層面",
    ),
)
def test_common_chinese_words_are_not_invented_as_people_or_places(text: str) -> None:
    assert not {
        entity
        for entity in bounded_named_entities(text)
        if entity[1] in {"PERSON", "PLACE"}
    }


@pytest.mark.parametrize(
    ("text", "person"),
    (
        ("警方拘捕李小明", "李小明"),
        ("法院起訴王小明", "王小明"),
        ("機構邀請李小明", "李小明"),
        ("政府會見劉志偉", "劉志偉"),
    ),
)
def test_action_context_retains_arbitrary_chinese_person(
    text: str, person: str
) -> None:
    assert (person, "PERSON") in bounded_named_entities(text)


def test_owner_approved_hong_kong_charter_has_no_simplified_shapes() -> None:
    charter = (
        Path(__file__).parents[2]
        / "docs/reference/editorial/product-editorial-charter.zh-HK.md"
    )

    violations = tuple(
        (line_number, line)
        for line_number, line in enumerate(charter.read_text().splitlines(), start=1)
        if contains_simplified_variant(line)
    )

    assert violations == ()


def test_unambiguous_simplified_shape_is_rejected() -> None:
    for text in (
        "官方说新安排",
        "里面安排不变",
        "后台程序更新",
        "后天公布安排",
        "后年公布安排",
        "后半段安排",
        "干部公布安排",
        "干事公布安排",
        "心里已有安排",
        "屋里等候安排",
        "其余安排已經確認",
        "定于明日公布安排",
        "一只警犬參與行動",
        "万事如意",
        "一百万元",
        "一叶知秋",
        "一碗面",
        "不适合",
        "不准确",
        "不明确",
        "一伙人",
        "不舍得",
        "三角巾包扎法",
        "制造商公布安排",
        "占用道路",
        "征收安排",
        "放松限制",
        "涂改文件",
        "谷物供應",
        "手表展示",
        "云端系統",
        "系上安全帶",
        "官方划定範圍",
        "企業托管資產",
        "部門占据場地",
        "公開征集意見",
        "這是范例",
        "面包供應增加",
        "柜台服務恢復",
        "方案适用全港",
        "种植面積增加",
        "党派發表聲明",
        "几乎全部完成",
        "夸大影響",
        "願景成為愿望",
        "群众散去",
        "警方驱散群众",
    ):
        assert contains_simplified_variant(text)


@pytest.mark.parametrize(
    "entity", ("里斯本", "里約熱內盧", "后海灣", "干邑", "干德道", "干諾道中")
)
def test_approved_proper_name_is_exempt_from_contextual_shape_gate(
    entity: str,
) -> None:
    candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    source_claim = f"{entity}公布最新安排"
    changed_headline = replace(
        headline,
        claim=source_claim,
        supporting_excerpt=source_claim,
        named_entity_evidence=((entity, "PLACE", f"entity:{entity}"),),
        named_entities=(entity,),
        rendered_named_entities=(entity,),
        rendered_assertion_zh_hant_hk=f"{entity}最新安排",
    )
    guarded_package = replace(
        package,
        passages=(f"{source_claim}\n{substantive.claim}",),
        substantive_new_information=(source_claim, substantive.claim),
        governed_claims=(changed_headline, substantive),
        qualification_evidence=(
            replace(
                package.qualification_evidence[0],
                test_evidence=(
                    ("action_class", "OFFICIAL_DEADLINE"),
                    ("event_polarity", "AFFIRMED"),
                    ("action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"),
                    ("material_relation_span", source_claim),
                    ("reader_action", source_claim),
                ),
            ),
            package.qualification_evidence[1],
        ),
        resolved_evidence_records=(
            *package.resolved_evidence_records,
            (f"entity:{entity}", "fixture-digest"),
        ),
    )

    decision = DeterministicWriteAdmission().decide(
        candidate, guarded_package, decided_at="2026-08-20T00:00:00.000000Z"
    )

    assert decision.decision == "WRITE_READY"


@pytest.mark.parametrize(
    ("source_claim", "rendered_claim", "source_entity", "rendered_entity"),
    (
        (
            "香港政府公布荃灣新安排",
            "香港政府最新公告涉及沙田安排",
            ("荃灣", "PLACE"),
            ("沙田", "PLACE"),
        ),
        (
            "政府任命王小明出任局長並公布新安排",
            "政府任命李小明出任局長並公布新安排",
            ("王小明", "PERSON"),
            ("李小明", "PERSON"),
        ),
        (
            "政府任命王小龍出任局長並公布新安排",
            "政府任命李小宇出任局長並公布新安排",
            ("王小龍", "PERSON"),
            ("李小宇", "PERSON"),
        ),
        (
            "香港政府公布旺角新安排",
            "香港政府最新公告涉及尖沙咀安排",
            ("旺角", "PLACE"),
            ("尖沙咀", "PLACE"),
        ),
        (
            "香港政府公布王小龍獲委任為局長",
            "香港政府最新公告指李小宇獲委任做局長",
            ("王小龍", "PERSON"),
            ("李小宇", "PERSON"),
        ),
        (
            "香港政府公布：北角將實施新安排",
            "香港政府最新公告指：太子將實施新安排",
            ("北角", "PLACE"),
            ("太子", "PLACE"),
        ),
        (
            "香港政府公布劉志偉獲委任為局長",
            "香港政府最新公告指郭志強獲委任做局長",
            ("劉志偉", "PERSON"),
            ("郭志強", "PERSON"),
        ),
        (
            "香港政府公布：銅鑼灣將設立新中心",
            "香港政府最新公告指：佐敦將設立新中心",
            ("銅鑼灣", "PLACE"),
            ("佐敦", "PLACE"),
        ),
        (
            "香港政府公布新人事安排由劉志偉出任局長",
            "香港政府最新公告指新人事由郭志強出任局長",
            ("劉志偉", "PERSON"),
            ("郭志強", "PERSON"),
        ),
        (
            "香港政府公布新人事安排由劉志偉接任局長",
            "香港政府最新公告指新人事由郭志強升任局長",
            ("劉志偉", "PERSON"),
            ("郭志強", "PERSON"),
        ),
        (
            "香港政府公布新中心將設於銅鑼灣",
            "香港政府最新公告顯示新中心設於佐敦",
            ("銅鑼灣", "PLACE"),
            ("佐敦", "PLACE"),
        ),
    ),
)
def test_changed_chinese_identity_fails_closed_before_writer(
    source_claim: str,
    rendered_claim: str,
    source_entity: tuple[str, str],
    rendered_entity: tuple[str, str],
) -> None:
    assert source_entity in bounded_named_entities(source_claim)
    assert rendered_entity in bounded_named_entities(rendered_claim)
    candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    source_entities = tuple(sorted(bounded_named_entities(source_claim)))
    changed_headline = replace(
        headline,
        claim=source_claim,
        supporting_excerpt=source_claim,
        rendered_assertion_zh_hant_hk=rendered_claim,
        named_entity_evidence=tuple(
            (text, entity_type, f"entity:identity:{index}")
            for index, (text, entity_type) in enumerate(source_entities)
        ),
        named_entities=tuple(text for text, _entity_type in source_entities),
        rendered_named_entities=tuple(text for text, _entity_type in source_entities),
    )
    guarded = replace(
        package,
        passages=(f"{source_claim}\n{substantive.claim}",),
        substantive_new_information=(source_claim, substantive.claim),
        governed_claims=(changed_headline, substantive),
        qualification_evidence=(
            replace(
                package.qualification_evidence[0],
                test_evidence=(
                    ("action_class", "OFFICIAL_DEADLINE"),
                    ("event_polarity", "AFFIRMED"),
                    ("action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"),
                    ("material_relation_span", source_claim),
                    ("reader_action", source_claim),
                ),
            ),
            package.qualification_evidence[1],
        ),
        resolved_evidence_records=(
            *package.resolved_evidence_records,
            *(
                (f"entity:identity:{index}", f"digest:identity:{index}")
                for index, _item in enumerate(source_entities)
            ),
        ),
    )
    decision = DeterministicWriteAdmission().decide(
        candidate, guarded, decided_at="2026-08-20T00:00:00Z"
    )
    assert decision.decision == "HOLD"
    assert "INVALID_GOVERNED_CLAIM_EVIDENCE" in decision.stable_reason_codes


def test_short_service_delay_cannot_masquerade_as_law_change() -> None:
    candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    service_noise = "服務公布新增十分鐘延誤"
    changed_headline = replace(
        headline,
        claim=service_noise,
        supporting_excerpt=service_noise,
        rendered_assertion_zh_hant_hk="服務新增十分鐘延誤安排",
    )
    guarded = replace(
        package,
        passages=(f"{service_noise}\n{substantive.claim}",),
        substantive_new_information=(service_noise, substantive.claim),
        governed_claims=(changed_headline, substantive),
        qualification_evidence=(
            QualificationEvidence(
                Evid012QualificationTest.LAW_RIGHT_STATUS_POLICY,
                changed_headline.claim_id,
                package.qualification_evidence[0].qualification_record_id,
                (
                    ("change_kind", "LAW"),
                    ("event_polarity", "AFFIRMED"),
                    ("change_relation", "NEW_OR_CHANGED_STATE"),
                    ("material_relation_span", service_noise),
                    ("new_state", service_noise),
                ),
            ),
            package.qualification_evidence[1],
        ),
    )
    decision = DeterministicWriteAdmission().decide(
        candidate, guarded, decided_at="2026-08-20T00:00:00Z"
    )
    assert decision.decision == "HOLD"
    assert decision.stable_reason_codes == ("QUALIFICATION_EVIDENCE_NOT_EXACT",)


@pytest.mark.parametrize(
    ("service_noise", "action_class"),
    (
        ("服務公布新增十分鐘延誤", "INSTRUCTION"),
        ("服務公布新增半小時延誤", "INSTRUCTION"),
        ("服務公布零分鐘延誤", "INSTRUCTION"),
        ("青松學校公布校巴新增半小時延誤", "INSTRUCTION"),
        ("服務公布半小時延誤後乘客要求賠償", "INSTRUCTION"),
        ("服務公布半小時延誤後乘客認為必須賠償", "INSTRUCTION"),
        ("服務公布半小時延誤後工會表示乘客應該獲賠償", "INSTRUCTION"),
        ("服務公布新增半小時延誤後乘客表示必須獲賠償", "INSTRUCTION"),
        ("政府公布服務新增半小時延誤後乘客表示必須獲賠償", "INSTRUCTION"),
        ("服務重申申請截止日期並公布新增半小時延誤", "OFFICIAL_DEADLINE"),
        ("服務重申申請限期延長安排並公布新增半小時延誤", "OFFICIAL_DEADLINE"),
    ),
)
def test_service_delay_cannot_masquerade_as_official_instruction(
    service_noise: str, action_class: str
) -> None:
    candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    entities = tuple(sorted(bounded_named_entities(service_noise)))
    changed_headline = replace(
        headline,
        claim=service_noise,
        supporting_excerpt=service_noise,
        rendered_assertion_zh_hant_hk=f"{service_noise.replace('公布', '公布咗')}安排",
        named_entity_evidence=tuple(
            (text, entity_type, f"entity:service:{index}")
            for index, (text, entity_type) in enumerate(entities)
        ),
        named_entities=tuple(text for text, _entity_type in entities),
        rendered_named_entities=tuple(text for text, _entity_type in entities),
    )
    guarded = replace(
        package,
        passages=(f"{service_noise}\n{substantive.claim}",),
        substantive_new_information=(service_noise, substantive.claim),
        governed_claims=(changed_headline, substantive),
        qualification_evidence=(
            QualificationEvidence(
                Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE,
                changed_headline.claim_id,
                package.qualification_evidence[0].qualification_record_id,
                (
                    ("action_class", action_class),
                    ("event_polarity", "AFFIRMED"),
                    ("action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"),
                    ("material_relation_span", service_noise),
                    ("reader_action", service_noise),
                ),
            ),
            package.qualification_evidence[1],
        ),
        resolved_evidence_records=(
            *package.resolved_evidence_records,
            *(
                (f"entity:service:{index}", f"digest:service:{index}")
                for index, _entity in enumerate(entities)
            ),
        ),
    )
    decision = DeterministicWriteAdmission().decide(
        candidate, guarded, decided_at="2026-08-20T00:00:00Z"
    )
    assert decision.decision == "HOLD"
    assert decision.stable_reason_codes == ("QUALIFICATION_EVIDENCE_NOT_EXACT",)


def test_explicit_reader_instruction_survives_disruption_noise_gate() -> None:
    candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    instruction = "政府公布鐵路停駛期間乘客必須改乘巴士"
    changed_headline = replace(
        headline,
        claim=instruction,
        supporting_excerpt=instruction,
        rendered_assertion_zh_hant_hk="政府宣布鐵路停駛期間乘客必須改搭巴士",
    )
    guarded = replace(
        package,
        passages=(f"{instruction}\n{substantive.claim}",),
        substantive_new_information=(instruction, substantive.claim),
        governed_claims=(changed_headline, substantive),
        qualification_evidence=(
            QualificationEvidence(
                Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE,
                changed_headline.claim_id,
                package.qualification_evidence[0].qualification_record_id,
                (
                    ("action_class", "INSTRUCTION"),
                    ("event_polarity", "AFFIRMED"),
                    ("action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"),
                    ("material_relation_span", instruction),
                    ("reader_action", instruction),
                ),
            ),
            package.qualification_evidence[1],
        ),
    )
    decision = DeterministicWriteAdmission().decide(
        candidate, guarded, decided_at="2026-08-20T00:00:00Z"
    )
    assert decision.decision == "WRITE_READY"


def test_governed_records_reject_cross_source_claim_link() -> None:
    items = (
        SourceItem(
            "A", "a", "Discovery A", "Background A", "https://example.test/same"
        ),
        SourceItem(
            "B",
            "b",
            "Claim from B",
            "Deadline is 30 September.",
            "https://example.test/same",
        ),
    )
    candidate = StoryCandidateRecord(
        "candidate-cross",
        "hypothesis-cross",
        "Discovery A",
        items,
        (DiscoverySignalRecord("signal-a", "A", "a", "digest-a"),),
        (NewsLeadRecord("lead-a", "signal-a", "Discovery A"),),
    )
    base = package_for(candidate)
    claim = GovernedClaimEvidence(
        "claim-cross",
        "Claim from B",
        1,
        "Claim from B",
        ("A",),
        ("source-record-B",),
        ("authority-A",),
        ("rights-A",),
        ("dependency-A",),
        ("origin-A",),
        ClaimAuthorityClass.RESPONSIBLE_PRIMARY,
        "Own deadline",
        GovernedClaimStatus.CONFIRMED_FACT,
        "A",
        "官方公布最新限期安排",
        "HEADLINE",
        "semantic-A",
    )
    package = replace(base, governed_claims=(claim,))
    records = (
        {
            "record_id": "source-record-B",
            "record_type": "SOURCE_RECORD",
            "candidate_id": candidate.candidate_id,
            "base_package_digest": base.digest,
            "status": "CURRENT",
            "source_id": "B",
            "canonical_url": "https://example.test/same",
            "publisher": "B",
            "responsible_body": "B",
            "source_type": "PRIMARY_OFFICIAL",
            "authority_class": "RESPONSIBLE_PRIMARY",
            "publication_time": "2026-01-01",
            "retrieval_time": "2026-01-01",
            "geography": "UK",
            "language": "en",
            "extraction_status": "COMPLETE",
            "rights_decision_id": "rights-A",
            "originating_report_id": "origin-A",
            "originating_artefact_digest": "sha256:source-a",
            "dependency_evidence_ids": ["dependency-A"],
        },
        {
            "record_id": "authority-A",
            "record_type": "SOURCE_AUTHORITY_DECISION",
            "candidate_id": candidate.candidate_id,
            "base_package_digest": base.digest,
            "status": "CURRENT",
            "source_id": "A",
            "decision": "ADMITTED",
            "authority_class": "RESPONSIBLE_PRIMARY",
            "authority_scope": "Own deadline",
            "governed_claim_id": claim.claim_id,
            "claim_digest": digest_bytes(claim.claim.encode()),
        },
        {
            "record_id": "rights-A",
            "record_type": "RIGHTS_DECISION",
            "candidate_id": candidate.candidate_id,
            "base_package_digest": base.digest,
            "status": "CURRENT",
            "source_id": "A",
            "decision": "PERMITTED",
            "permitted_use": "PUBLICATION_EVIDENCE",
        },
        {
            "record_id": "dependency-A",
            "record_type": "DEPENDENCY_EVIDENCE",
            "candidate_id": candidate.candidate_id,
            "base_package_digest": base.digest,
            "status": "CURRENT",
            "source_id": "A",
            "dependency_status": "RESOLVED",
            "evidential_origin_id": "origin-A",
            "originating_report_id": "origin-A",
        },
        {
            "record_id": "semantic-A",
            "record_type": "SEMANTIC_RELATION_EVIDENCE",
            "candidate_id": candidate.candidate_id,
            "base_package_digest": base.digest,
            "status": "CURRENT",
            "governed_claim_id": claim.claim_id,
            "source_modality": "ASSERTED",
            "rendered_modality": "ASSERTED",
            "source_polarity": "AFFIRMED",
            "rendered_polarity": "AFFIRMED",
            "relation": "SEMANTICALLY_EQUIVALENT",
            "claim_digest": digest_bytes(claim.claim.encode()),
            "rendered_assertion_digest": digest_bytes(
                claim.rendered_assertion_zh_hant_hk.encode()
            ),
        },
    )
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE proving_write_evidence_records("
        "record_id TEXT PRIMARY KEY, record_type TEXT, "
        "record_json TEXT, record_digest TEXT)"
    )
    for record in records:
        raw = canonical_json_bytes(record).decode()
        connection.execute(
            "INSERT INTO proving_write_evidence_records VALUES(?,?,?,?)",
            (
                record["record_id"],
                record["record_type"],
                raw,
                digest_bytes(raw.encode()),
            ),
        )

    assert _resolve_governed_records(connection, candidate, base, package) is None
    connection.close()


@pytest.mark.parametrize(
    "source_overrides",
    (
        {"source_type": "LEAD_ONLY"},
        {
            "publisher": 123,
            "responsible_body": ["fixture"],
            "publication_time": True,
            "retrieval_time": {"at": "2026-01-01"},
            "geography": 42,
            "language": 99,
        },
        {"publication_time": "not-a-time", "retrieval_time": "tomorrow-ish"},
        {"geography": "Mars", "language": "definitely-not-bcp47"},
    ),
)
def test_governed_records_reject_lead_only_or_untyped_source_records(
    source_overrides: dict[str, object],
) -> None:
    item = SourceItem(
        "A", "a", "Official deadline", "Deadline changed", "https://example.test/a"
    )
    candidate = StoryCandidateRecord(
        "candidate-source-schema",
        "hypothesis-source-schema",
        item.headline,
        (item,),
        (DiscoverySignalRecord("signal-a", "A", "a", "digest-a"),),
        (NewsLeadRecord("lead-a", "signal-a", item.headline),),
    )
    base = package_for(candidate)
    claim = GovernedClaimEvidence(
        "claim-source-schema",
        item.headline,
        0,
        item.headline,
        ("A",),
        ("source-record-A",),
        ("authority-A",),
        ("rights-A",),
        ("dependency-A",),
        ("origin-A",),
        ClaimAuthorityClass.RESPONSIBLE_PRIMARY,
        "Own deadline",
        GovernedClaimStatus.CONFIRMED_FACT,
        "A",
        "官方公布最新限期安排",
        "HEADLINE",
        "semantic-A",
    )
    package = replace(base, governed_claims=(claim,))
    source_record: dict[str, object] = {
        "record_id": "source-record-A",
        "record_type": "SOURCE_RECORD",
        "candidate_id": candidate.candidate_id,
        "base_package_digest": base.digest,
        "status": "CURRENT",
        "source_id": "A",
        "canonical_url": item.canonical_url,
        "publisher": "Fixture authority",
        "responsible_body": "Fixture authority",
        "source_type": "PRIMARY_OFFICIAL",
        "authority_class": "RESPONSIBLE_PRIMARY",
        "publication_time": "2026-01-01",
        "retrieval_time": "2026-01-01",
        "geography": "UK",
        "language": "en",
        "extraction_status": "COMPLETE",
        "rights_decision_id": "rights-A",
        "originating_report_id": "origin-A",
        "originating_artefact_digest": "sha256:source-a",
        "dependency_evidence_ids": ["dependency-A"],
    }
    records: tuple[dict[str, object], ...] = (
        source_record,
        {
            "record_id": "authority-A",
            "record_type": "SOURCE_AUTHORITY_DECISION",
            "candidate_id": candidate.candidate_id,
            "base_package_digest": base.digest,
            "status": "CURRENT",
            "source_id": "A",
            "decision": "ADMITTED",
            "authority_class": "RESPONSIBLE_PRIMARY",
            "authority_scope": "Own deadline",
            "governed_claim_id": claim.claim_id,
            "claim_digest": digest_bytes(claim.claim.encode()),
        },
        {
            "record_id": "rights-A",
            "record_type": "RIGHTS_DECISION",
            "candidate_id": candidate.candidate_id,
            "base_package_digest": base.digest,
            "status": "CURRENT",
            "source_id": "A",
            "decision": "PERMITTED",
            "permitted_use": "PUBLICATION_EVIDENCE",
        },
        {
            "record_id": "dependency-A",
            "record_type": "DEPENDENCY_EVIDENCE",
            "candidate_id": candidate.candidate_id,
            "base_package_digest": base.digest,
            "status": "CURRENT",
            "source_id": "A",
            "dependency_status": "RESOLVED",
            "evidential_origin_id": "origin-A",
            "originating_report_id": "origin-A",
        },
        {
            "record_id": "semantic-A",
            "record_type": "SEMANTIC_RELATION_EVIDENCE",
            "candidate_id": candidate.candidate_id,
            "base_package_digest": base.digest,
            "status": "CURRENT",
            "governed_claim_id": claim.claim_id,
            "source_modality": "ASSERTED",
            "rendered_modality": "ASSERTED",
            "source_polarity": "AFFIRMED",
            "rendered_polarity": "AFFIRMED",
            "relation": "SEMANTICALLY_EQUIVALENT",
            "claim_digest": digest_bytes(claim.claim.encode()),
            "rendered_assertion_digest": digest_bytes(
                claim.rendered_assertion_zh_hant_hk.encode()
            ),
        },
    )

    def resolve(overrides: dict[str, object]) -> object:
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE proving_write_evidence_records("
            "record_id TEXT PRIMARY KEY, record_type TEXT, "
            "record_json TEXT, record_digest TEXT)"
        )
        for original in records:
            record = dict(original)
            if record["record_type"] == "SOURCE_RECORD":
                record.update(overrides)
            raw = canonical_json_bytes(record).decode()
            connection.execute(
                "INSERT INTO proving_write_evidence_records VALUES(?,?,?,?)",
                (
                    record["record_id"],
                    record["record_type"],
                    raw,
                    digest_bytes(raw.encode()),
                ),
            )
        resolved = _resolve_governed_records(connection, candidate, base, package)
        connection.close()
        return resolved

    assert resolve({}) is not None
    assert resolve(source_overrides) is None


def test_independent_authority_collapses_shared_report_and_url_origins() -> None:
    sources = [
        {
            "source_id": "A",
            "canonical_url": "https://example.test/shared",
            "originating_report_id": "same-report",
            "originating_artefact_digest": "sha256:shared",
        },
        {
            "source_id": "B",
            "canonical_url": "https://example.test/shared",
            "originating_report_id": "same-report",
            "originating_artefact_digest": "sha256:shared",
        },
    ]
    dependencies = [
        {"originating_report_id": "same-report", "evidential_origin_id": "origin-1"},
        {"originating_report_id": "same-report", "evidential_origin_id": "origin-2"},
    ]

    assert not _has_valid_origin_independence(
        ClaimAuthorityClass.INDEPENDENT_RELIABLE,
        sources,
        dependencies,
        ("origin-1", "origin-2"),
    )
    sources[1]["originating_report_id"] = "report-2"
    dependencies[1]["originating_report_id"] = "report-2"
    assert not _has_valid_origin_independence(
        ClaimAuthorityClass.INDEPENDENT_RELIABLE,
        sources,
        dependencies,
        ("origin-1", "origin-2"),
    )
    sources[1]["canonical_url"] = "https://example.test/independent"
    assert not _has_valid_origin_independence(
        ClaimAuthorityClass.INDEPENDENT_RELIABLE,
        sources,
        dependencies,
        ("origin-1", "origin-2"),
    )
    sources[1]["originating_artefact_digest"] = "sha256:independent"
    assert _has_valid_origin_independence(
        ClaimAuthorityClass.INDEPENDENT_RELIABLE,
        sources,
        dependencies,
        ("origin-1", "origin-2"),
    )


def _admit_package(
    candidate: StoryCandidateRecord,
    package: EvidencePackage,
    *,
    authority_class: ClaimAuthorityClass = ClaimAuthorityClass.RESPONSIBLE_PRIMARY,
    origin_ids: tuple[str, ...] | None = None,
) -> EvidencePackage:
    headline = "Official action changed: a new official arrangement."
    substantive = "Official action changed: an official deadline."
    origins = origin_ids or (f"origin:{candidate.items[0].source_id}",)
    claims = (
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
            evidential_origin_ids=origins,
            authority_class=authority_class,
            authority_scope="Responsible source for its own published update.",
            status=GovernedClaimStatus.CONFIRMED_FACT,
            attribution=candidate.items[0].source_id,
            rendered_assertion_zh_hant_hk="官方公布咗最新安排",
            claim_role="HEADLINE",
            semantic_relation_evidence_id=f"semantic:{candidate.candidate_id}:headline",
        ),
        GovernedClaimEvidence(
            claim_id=f"claim:{candidate.candidate_id}:substantive",
            claim=substantive,
            passage_index=0,
            supporting_excerpt=substantive,
            source_ids=(candidate.items[0].source_id,),
            source_record_ids=(f"source-record:{candidate.items[0].source_id}",),
            source_authority_decision_ids=(
                f"source-authority:{candidate.items[0].source_id}:substantive",
            ),
            rights_decision_ids=(f"rights:{candidate.items[0].source_id}",),
            dependency_evidence_ids=(f"dependency:{candidate.items[0].source_id}",),
            evidential_origin_ids=origins,
            authority_class=authority_class,
            authority_scope="Responsible source for its own published update.",
            status=GovernedClaimStatus.CONFIRMED_FACT,
            attribution=candidate.items[0].source_id,
            rendered_assertion_zh_hant_hk="相關官方資料確認安排已經更新",
            claim_role="SUBSTANTIVE",
            semantic_relation_evidence_id=f"semantic:{candidate.candidate_id}:substantive",
        ),
    )
    claims = tuple(_bind_fixture_entities(claim) for claim in claims)
    claim_ids = tuple(claim.claim_id for claim in claims)
    return replace(
        package,
        passages=(f"{package.passages[0]}\n{headline}\n{substantive}",),
        substantive_new_information=(headline, substantive),
        governed_claims=claims,
        qualification_evidence=tuple(
            QualificationEvidence(
                Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE,
                claim.claim_id,
                f"qualification:{claim.claim_id}",
                (
                    ("action_class", "OFFICIAL_DEADLINE"),
                    ("event_polarity", "AFFIRMED"),
                    ("action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"),
                    ("material_relation_span", claim.claim),
                    ("reader_action", claim.claim),
                ),
            )
            for claim in claims
        ),
        selection_rationale="Retained governed evidence proves a material official development.",
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
                    claim.semantic_relation_evidence_id,
                    f"semantic-digest:{claim.claim_id}",
                )
                for claim in claims
            ),
            *(
                (record_id, f"entity-digest:{record_id}")
                for claim in claims
                for _text, _entity_type, record_id in claim.named_entity_evidence
            ),
            *(
                (
                    f"qualification:{claim.claim_id}",
                    f"qualification-digest:{claim.claim_id}",
                )
                for claim in claims
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


def _candidate_package(
    *, substantive: tuple[str, ...] = ("The deadline changed to 30 September.",)
) -> tuple[StoryCandidateRecord, EvidencePackage]:
    item = SourceItem(
        source_id="HK-01",
        item_key="item-1",
        headline="申請期限改至9月30日",
        body="The deadline changed to 30 September.",
        canonical_url="https://example.test/item-1",
    )
    signal = DiscoverySignalRecord("signal-1", "HK-01", "item-1", "sha256:body")
    lead = NewsLeadRecord("lead-1", "signal-1", item.headline)
    candidate = StoryCandidateRecord(
        "candidate-1", "hypothesis-1", item.headline, (item,), (signal,), (lead,)
    )
    package = EvidencePackage(
        candidate_id=candidate.candidate_id,
        hypothesis_id=candidate.hypothesis_id,
        signal_ids=(signal.signal_id,),
        lead_ids=(lead.lead_id,),
        source_ids=(item.source_id,),
        observation_digests=(signal.observation_digest,),
        passages=(f"HK-01: {item.headline}\n{item.body}",),
    )
    package = _admit_package(candidate, package)
    if not substantive:
        package = replace(package, substantive_new_information=())
    return candidate, package


def _qualified_builder(
    ready_sources: frozenset[str],
):
    def build(candidate: StoryCandidateRecord) -> EvidencePackage:
        package = package_for(candidate)
        if not ({item.source_id for item in candidate.items} & ready_sources):
            return package
        return _admit_package(candidate, package)

    return build


class RecordingFixtureWriter(FixtureWriter):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def write(
        self, candidate: StoryCandidateRecord, package: EvidencePackage
    ) -> WriterCopy:
        self.calls.append(candidate.candidate_id)
        return super().write(candidate, package)


class CountingWriter:
    writer_id = "counting-writer"

    def __init__(self) -> None:
        self.calls = 0

    def write(
        self, candidate: StoryCandidateRecord, package: EvidencePackage
    ) -> WriterCopy:
        self.calls += 1
        raise AssertionError("an unqualified candidate reached the writer")


def test_zero_qualifying_candidates_succeed_without_writer_or_filler(
    tmp_path: Path,
) -> None:
    proving = _proving(tmp_path)
    unpublished = tmp_path / "unpublished_store.sqlite3"
    writer = CountingWriter()

    report = run_cycle(
        proving_store=str(proving),
        unpublished_store=str(unpublished),
        writer=writer,
        max_writes=5,
        clock=_CLOCK,
    )

    assert writer.calls == 0
    assert report.candidates_considered == 3
    assert report.write_ready == 0
    assert report.selected_write_ready == 0
    assert report.candidate_attempts == 0
    assert report.provider_dispatches == 0
    assert report.accepted_payload_count == 0
    assert report.minted == 0
    assert list_payloads(str(unpublished)) == ()

    connection = sqlite3.connect(unpublished)
    decisions = connection.execute(
        "SELECT decision, COUNT(*) FROM unpublished_write_admission_decisions "
        "GROUP BY decision ORDER BY decision"
    ).fetchall()
    close_kind = connection.execute(
        "SELECT COUNT(*) FROM ledger WHERE kind='PRIVATE_CYCLE_CLOSE'"
    ).fetchone()[0]
    connection.close()
    assert decisions == [("HOLD", 3)]
    assert close_kind == 1


def test_non_empty_passage_without_substantive_fact_is_rejected() -> None:
    candidate, package = _candidate_package(substantive=())

    decision = DeterministicWriteAdmission().decide(
        candidate, package, decided_at="2026-08-20T00:00:00.000000Z"
    )

    assert decision.decision == "REJECT"
    assert decision.stable_reason_codes == ("NO_SUBSTANTIVE_NEW_INFORMATION",)


def test_evid_012_is_a_closed_versioned_domain() -> None:
    with pytest.raises(ValueError, match="not in EVID-012"):
        QualificationEvidence(
            cast(Evid012QualificationTest, "TOTALLY_NOT_EVID_012"),
            "claim-1",
            "qualification:claim-1",
            (("invalid", "invalid"),),
        )


def test_stale_originality_policy_version_fails_closed() -> None:
    _candidate, package = _candidate_package()
    with pytest.raises(ValueError, match="originality policy version"):
        replace(
            package.governed_claims[0],
            originality_policy_version="newsroom.cont-originality.v1",
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"certainty": "BOGUS"},
        {"originality_basis": "BOGUS"},
        {"claim_role": "BOGUS"},
        {"named_entities": (1,)},
    ),
)
def test_malformed_governed_claim_literals_fail_closed(
    changes: dict[str, object],
) -> None:
    _candidate, package = _candidate_package()
    with pytest.raises(ValueError):
        replace(package.governed_claims[0], **changes)


def test_unknown_evidence_gate_fails_closed() -> None:
    _candidate, package = _candidate_package()
    with pytest.raises(ValueError, match="gate or result"):
        EvidenceGateEvidence(
            "BOGUS",  # type: ignore[arg-type]
            "PASS",
            tuple(item.claim_id for item in package.governed_claims),
        )


def test_duplicate_qualification_and_substantive_inventories_fail_closed() -> None:
    _candidate, package = _candidate_package()
    qualification = package.qualification_evidence[0]
    with pytest.raises(ValueError, match="qualification evidence must be unique"):
        replace(package, qualification_evidence=(qualification, qualification))
    fact = package.substantive_new_information[0]
    with pytest.raises(ValueError, match="governed inventories must be unique"):
        replace(package, substantive_new_information=(fact, fact))


def test_ordinary_short_delay_cannot_satisfy_material_disruption_evidence() -> None:
    with pytest.raises(ValueError, match="does not satisfy EVID-012"):
        QualificationEvidence(
            Evid012QualificationTest.ESSENTIAL_SERVICE_DISRUPTION,
            "claim-short-delay",
            "qualification:claim-short-delay",
            (
                ("service_kind", "TRANSPORT"),
                ("event_polarity", "AFFIRMED"),
                ("duration_relation", "DISRUPTION_DURATION"),
                ("material_effect_class", "MINOR_DELAY"),
                ("affected_group", "Two passengers for two minutes"),
            ),
        )


def test_material_duration_classifier_must_be_exact_in_retained_fact() -> None:
    candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    short_delay = "Train delayed by 2 minutes."
    changed_substantive = replace(
        substantive,
        claim=short_delay,
        supporting_excerpt=short_delay,
        rendered_assertion_zh_hant_hk="列車只係短暫延誤咗兩分鐘",
    )
    unsupported = replace(
        package,
        passages=(f"HK-01: {headline.claim}\n{short_delay}",),
        substantive_new_information=(headline.claim, short_delay),
        governed_claims=(headline, changed_substantive),
        qualification_evidence=(
            package.qualification_evidence[0],
            QualificationEvidence(
                Evid012QualificationTest.ESSENTIAL_SERVICE_DISRUPTION,
                changed_substantive.claim_id,
                f"qualification:{changed_substantive.claim_id}",
                (
                    ("service_kind", "TRANSPORT"),
                    ("event_polarity", "AFFIRMED"),
                    ("duration_relation", "DISRUPTION_DURATION"),
                    ("duration_minutes", "60"),
                    ("affected_group", short_delay),
                ),
            ),
        ),
    )

    decision = DeterministicWriteAdmission().decide(
        candidate, unsupported, decided_at="2026-08-20T00:00:00.000000Z"
    )
    assert decision.decision == "HOLD"
    assert decision.stable_reason_codes == ("QUALIFICATION_EVIDENCE_NOT_EXACT",)


def test_admission_policy_identity_binds_all_admission_subpolicies() -> None:
    assert _OLDEST_WRITE_ADMISSION_POLICY_VERSION == (
        "newsroom.write-admission.v3+newsroom.evid-012.v7+"
        "newsroom.evidence-approval.v8+newsroom.evidence-gates.v2+"
        "newsroom.governed-claim.v7+newsroom.governed-input.v10+"
        "newsroom.named-entity.v8+newsroom.cont-originality.v3+"
        "newsroom.zh-hant-hk-shape.v13"
    )
    assert _EARLIER_WRITE_ADMISSION_POLICY_VERSION == (
        "newsroom.write-admission.v4+"
        "newsroom.evid-012.v7+newsroom.evidence-approval.v8+"
        "newsroom.evidence-gates.v2+newsroom.governed-claim.v7+"
        "newsroom.governed-input.v10+newsroom.named-entity.v8+"
        "newsroom.cont-originality.v3+newsroom.zh-hant-hk-shape.v13"
    )
    assert _PREVIOUS_WRITE_ADMISSION_POLICY_VERSION == (
        "newsroom.write-admission.v5+"
        "newsroom.evid-012.v7+newsroom.evidence-approval.v8+"
        "newsroom.evidence-gates.v2+newsroom.governed-claim.v7+"
        "newsroom.governed-input.v10+newsroom.named-entity.v8+"
        "newsroom.cont-originality.v3+newsroom.zh-hant-hk-shape.v13"
    )
    assert _PENULTIMATE_WRITE_ADMISSION_POLICY_VERSION == (
        "newsroom.write-admission.v6+"
        "newsroom.evid-012.v7+newsroom.evidence-approval.v8+"
        "newsroom.evidence-gates.v2+newsroom.governed-claim.v7+"
        "newsroom.governed-input.v10+newsroom.named-entity.v8+"
        "newsroom.cont-originality.v3+newsroom.zh-hant-hk-shape.v13"
    )
    assert _EARLIEST_CURRENT_SHAPE_WRITE_ADMISSION_POLICY_VERSION == (
        "newsroom.write-admission.v7+"
        "newsroom.evid-012.v7+newsroom.evidence-approval.v8+"
        "newsroom.evidence-gates.v2+newsroom.governed-claim.v7+"
        "newsroom.governed-input.v10+newsroom.named-entity.v9+"
        "newsroom.cont-originality.v3+newsroom.zh-hant-hk-shape.v14"
    )
    assert _LATEST_LEGACY_WRITE_ADMISSION_POLICY_VERSION == (
        "newsroom.write-admission.v8+"
        "newsroom.evid-012.v7+newsroom.evidence-approval.v8+"
        "newsroom.evidence-gates.v2+newsroom.governed-claim.v7+"
        "newsroom.governed-input.v10+newsroom.named-entity.v10+"
        "newsroom.cont-originality.v3+newsroom.zh-hant-hk-shape.v14"
    )
    assert WRITE_ADMISSION_POLICY_VERSION == (
        "newsroom.write-admission.v10+"
        f"{EVID_012_POLICY_VERSION}+{EVIDENCE_APPROVAL_POLICY_VERSION}+"
        f"{EVIDENCE_GATE_POLICY_VERSION}+"
        f"{GOVERNED_CLAIM_POLICY_VERSION}+{GOVERNED_INPUT_SCHEMA_VERSION}+"
        f"{NAMED_ENTITY_POLICY_VERSION}+{ORIGINALITY_POLICY_VERSION}+"
        f"{ZH_HANT_HK_SHAPE_POLICY_VERSION}+{FACTUAL_LOCALISATION_POLICY_VERSION}+"
        "newsroom.qualification-relation.v3"
    )


@pytest.mark.parametrize("entity_version, qualification", (
    (11, False), (12, False), (13, False), (13, True), (14, True), (14, 2), (15, 2),
))
def test_prior_v9_subpolicy_decisions_replay_but_are_not_current(entity_version, qualification, tmp_path):
    candidate, package = _candidate_package()
    current = DeterministicWriteAdmission().decide(
        candidate, package, decided_at="2026-09-19T12:00:00Z",
    )
    values = asdict(current)
    values.pop("decision_id")
    decided_at = values.pop("decided_at")
    values["policy_version"] = (
        "newsroom.write-admission.v9+newsroom.evid-012.v7+"
        "newsroom.evidence-approval.v8+newsroom.evidence-gates.v2+"
        "newsroom.governed-claim.v7+newsroom.governed-input.v10+"
        f"newsroom.named-entity.v{entity_version}+newsroom.cont-originality.v3+"
        "newsroom.zh-hant-hk-shape.v14"
    ) + ("+newsroom.factual-localisation.v1" if entity_version >= 13 else "")
    if qualification:
        values["policy_version"] += f"+newsroom.qualification-relation.v{int(qualification)}"
    old = WriteAdmissionDecision(
        decision_id=_decision_id(**values), decided_at=decided_at, **values,
    )
    assert WriteAdmissionDecision.from_record(old.as_record()) == old
    assert old.decision_id != current.decision_id
    with pytest.raises(ValueError, match="policy is not current"):
        select_write_ready(((candidate, package, old),), limit=1, selected_at=decided_at)
    with pytest.raises(ValueError, match="unsupported write-admission policy version"):
        replace(old, policy_version=old.policy_version.replace(
            f"named-entity.v{entity_version}", "named-entity.v999",
        ))
    connection = connect(str(tmp_path / "v9-replay.sqlite3"))
    retain_write_admission_decision(connection, old)
    retain_write_admission_decision(connection, old)
    retain_write_admission_decision(connection, current)
    assert connection.execute(
        "SELECT count(*) FROM unpublished_write_admission_decisions"
    ).fetchone() == (2,)
    connection.close()


def test_changed_admission_semantics_replay_the_exact_previous_policy(
    tmp_path: Path,
) -> None:
    candidate, package = _candidate_package()
    original = package.governed_claims[0]
    claim = "The Home Office published changes"
    excerpt = "The Home Office published changes to the Skilled Worker Visa."
    legacy_entity_records = (
        ("Home Office", "ORGANISATION", "legacy-entity:home-office"),
        (
            "Skilled Worker Visa",
            "OFFICIAL_TERM",
            "legacy-entity:skilled-worker-visa",
        ),
    )
    legacy_claim = replace(
        original,
        claim=claim,
        supporting_excerpt=excerpt,
        rendered_assertion_zh_hant_hk=(
            "Home Office 已公布 Skilled Worker Visa 的修訂。"
        ),
        named_entity_evidence=legacy_entity_records,
        named_entities=("Home Office", "Skilled Worker Visa"),
        rendered_named_entities=("Home Office", "Skilled Worker Visa"),
    )
    legacy_package = replace(
        package,
        passages=(excerpt,),
        substantive_new_information=(claim,),
        governed_claims=(legacy_claim,),
        qualification_evidence=(),
        evidence_gate_evidence=tuple(
            replace(item, governed_claim_ids=(legacy_claim.claim_id,))
            for item in package.evidence_gate_evidence
        ),
        resolved_evidence_records=(
            *package.resolved_evidence_records,
            *(
                (record_id, "legacy-record")
                for _text, _type, record_id in legacy_entity_records
            ),
        ),
    )
    current = DeterministicWriteAdmission().decide(
        candidate, legacy_package, decided_at="2026-09-12T12:00:00Z"
    )
    assert current.decision == "HOLD"
    assert current.stable_reason_codes == (
        "INVALID_GOVERNED_CLAIM_EVIDENCE",
        "UNQUALIFIED_HEADLINE_CLAIM",
    )

    values = {
        "candidate_id": legacy_package.candidate_id,
        "evidence_package_digest": legacy_package.digest,
        "decision": "HOLD",
        "substantive_new_information": legacy_package.substantive_new_information,
        "qualification_tests": (),
        "selection_rationale": legacy_package.selection_rationale,
        "geography": legacy_package.geography,
        "categories": legacy_package.categories,
        "evidence_gate_results": legacy_package.evidence_gate_results,
        "freshness_result": legacy_package.freshness_result,
        "integrity_result": legacy_package.integrity_result,
        "stable_reason_codes": (
            "INVALID_SUBSTANTIVE_CLAIM_INVENTORY",
            "UNQUALIFIED_HEADLINE_CLAIM",
        ),
        "policy_version": _PREVIOUS_WRITE_ADMISSION_POLICY_VERSION,
    }
    legacy = WriteAdmissionDecision(
        decision_id=_decision_id(**values),
        decided_at="2026-09-11T12:00:00Z",
        **values,
    )
    assert WriteAdmissionDecision.from_record(legacy.as_record()) == legacy
    penultimate_values = {
        **values,
        "policy_version": _PENULTIMATE_WRITE_ADMISSION_POLICY_VERSION,
    }
    penultimate = WriteAdmissionDecision(
        decision_id=_decision_id(**penultimate_values),
        decided_at="2026-09-11T18:00:00Z",
        **penultimate_values,
    )
    assert WriteAdmissionDecision.from_record(penultimate.as_record()) == penultimate
    current_shape_legacy_values = {
        **values,
        "policy_version": _EARLIEST_CURRENT_SHAPE_WRITE_ADMISSION_POLICY_VERSION,
    }
    current_shape_legacy = WriteAdmissionDecision(
        decision_id=_decision_id(**current_shape_legacy_values),
        decided_at="2026-09-11T21:00:00Z",
        **current_shape_legacy_values,
    )
    assert WriteAdmissionDecision.from_record(
        current_shape_legacy.as_record()
    ) == current_shape_legacy
    latest_legacy_values = {
        **values,
        "policy_version": _LATEST_LEGACY_WRITE_ADMISSION_POLICY_VERSION,
    }
    latest_legacy = WriteAdmissionDecision(
        decision_id=_decision_id(**latest_legacy_values),
        decided_at="2026-09-11T23:00:00Z",
        **latest_legacy_values,
    )
    assert WriteAdmissionDecision.from_record(latest_legacy.as_record()) == latest_legacy
    with pytest.raises(ValueError, match="policy is not current"):
        select_write_ready(
            ((candidate, legacy_package, latest_legacy),),
            limit=1,
            selected_at="2026-09-12T12:00:00Z",
        )
    earlier_values = {
        **values,
        "policy_version": _EARLIER_WRITE_ADMISSION_POLICY_VERSION,
    }
    earlier = WriteAdmissionDecision(
        decision_id=_decision_id(**earlier_values),
        decided_at="2026-09-10T12:00:00Z",
        **earlier_values,
    )
    assert WriteAdmissionDecision.from_record(earlier.as_record()) == earlier
    oldest_values = {
        **values,
        "policy_version": _OLDEST_WRITE_ADMISSION_POLICY_VERSION,
    }
    oldest = WriteAdmissionDecision(
        decision_id=_decision_id(**oldest_values),
        decided_at="2026-09-09T12:00:00Z",
        **oldest_values,
    )
    assert WriteAdmissionDecision.from_record(oldest.as_record()) == oldest
    with pytest.raises(ValueError, match="policy is not current"):
        select_write_ready(
            ((candidate, legacy_package, legacy),),
            limit=1,
            selected_at="2026-09-12T12:00:00Z",
        )

    connection = connect(str(tmp_path / "write-admission-policy-replay.sqlite3"))
    retain_write_admission_decision(connection, oldest)
    retain_write_admission_decision(connection, earlier)
    retain_write_admission_decision(connection, legacy)
    retain_write_admission_decision(connection, penultimate)
    retain_write_admission_decision(connection, current_shape_legacy)
    retain_write_admission_decision(connection, latest_legacy)
    retain_write_admission_decision(connection, legacy)
    retain_write_admission_decision(connection, current)
    assert connection.execute(
        "SELECT policy_version,decision FROM unpublished_write_admission_decisions "
        "ORDER BY policy_version"
    ).fetchall() == sorted([
        (_OLDEST_WRITE_ADMISSION_POLICY_VERSION, "HOLD"),
        (_EARLIER_WRITE_ADMISSION_POLICY_VERSION, "HOLD"),
        (_PREVIOUS_WRITE_ADMISSION_POLICY_VERSION, "HOLD"),
        (_PENULTIMATE_WRITE_ADMISSION_POLICY_VERSION, "HOLD"),
        (_EARLIEST_CURRENT_SHAPE_WRITE_ADMISSION_POLICY_VERSION, "HOLD"),
        (_LATEST_LEGACY_WRITE_ADMISSION_POLICY_VERSION, "HOLD"),
        (WRITE_ADMISSION_POLICY_VERSION, "HOLD"),
    ])

    relabelled = legacy.as_record()
    relabelled["policy_version"] = WRITE_ADMISSION_POLICY_VERSION
    with pytest.raises(ValueError, match="identity is not canonical"):
        WriteAdmissionDecision.from_record(relabelled)
    relabelled_latest = latest_legacy.as_record()
    relabelled_latest["policy_version"] = WRITE_ADMISSION_POLICY_VERSION
    with pytest.raises(ValueError, match="identity is not canonical"):
        WriteAdmissionDecision.from_record(relabelled_latest)
    forged = legacy.as_record()
    forged["decision"] = "REJECT"
    with pytest.raises(ValueError, match="identity is not canonical"):
        WriteAdmissionDecision.from_record(forged)
    unknown = legacy.as_record()
    unknown["policy_version"] = "newsroom.write-admission.unknown"
    with pytest.raises(ValueError, match="unsupported write-admission policy"):
        WriteAdmissionDecision.from_record(unknown)
    connection.execute(
        "UPDATE unpublished_write_admission_decisions SET record_digest=? "
        "WHERE decision_id=?",
        ("sha256:" + "0" * 64, legacy.decision_id),
    )
    with pytest.raises(sqlite3.IntegrityError, match="conflicting write-admission"):
        retain_write_admission_decision(connection, legacy)
    connection.close()


def test_route_number_cannot_masquerade_as_material_disruption_duration() -> None:
    candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    short_delay = "Route 60 train delayed by two minutes."
    changed_substantive = replace(
        substantive,
        claim=short_delay,
        supporting_excerpt=short_delay,
        rendered_assertion_zh_hant_hk="六十號線列車只係延誤咗兩分鐘",
    )
    unsupported = replace(
        package,
        passages=(f"HK-01: {headline.claim}\n{short_delay}",),
        substantive_new_information=(headline.claim, short_delay),
        governed_claims=(headline, changed_substantive),
        qualification_evidence=(
            package.qualification_evidence[0],
            QualificationEvidence(
                Evid012QualificationTest.ESSENTIAL_SERVICE_DISRUPTION,
                changed_substantive.claim_id,
                f"qualification:{changed_substantive.claim_id}",
                (
                    ("service_kind", "TRANSPORT"),
                    ("event_polarity", "AFFIRMED"),
                    ("duration_relation", "DISRUPTION_DURATION"),
                    ("duration_minutes", "60"),
                    ("affected_group", short_delay),
                ),
            ),
        ),
    )

    decision = DeterministicWriteAdmission().decide(
        candidate, unsupported, decided_at="2026-08-20T00:00:00.000000Z"
    )

    assert decision.decision == "HOLD"
    assert decision.stable_reason_codes == ("QUALIFICATION_EVIDENCE_NOT_EXACT",)


def test_scheduled_journey_duration_cannot_masquerade_as_disruption_duration() -> None:
    candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    short_delay = (
        "Route 60 train delayed by two minutes; its scheduled journey takes 60 minutes."
    )
    changed_substantive = replace(
        substantive,
        claim=short_delay,
        supporting_excerpt=short_delay,
        rendered_assertion_zh_hant_hk="六十號線列車只係延誤咗兩分鐘",
    )
    unsupported = replace(
        package,
        passages=(f"HK-01: {headline.claim}\n{short_delay}",),
        substantive_new_information=(headline.claim, short_delay),
        governed_claims=(headline, changed_substantive),
        qualification_evidence=(
            package.qualification_evidence[0],
            QualificationEvidence(
                Evid012QualificationTest.ESSENTIAL_SERVICE_DISRUPTION,
                changed_substantive.claim_id,
                f"qualification:{changed_substantive.claim_id}",
                (
                    ("service_kind", "TRANSPORT"),
                    ("event_polarity", "AFFIRMED"),
                    ("duration_relation", "DISRUPTION_DURATION"),
                    ("duration_minutes", "60"),
                    ("affected_group", short_delay),
                ),
            ),
        ),
    )

    decision = DeterministicWriteAdmission().decide(
        candidate, unsupported, decided_at="2026-08-20T00:00:00.000000Z"
    )

    assert decision.decision == "HOLD"
    assert decision.stable_reason_codes == ("QUALIFICATION_EVIDENCE_NOT_EXACT",)


def test_material_disruption_duration_relation_is_admitted() -> None:
    candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    material_delay = "Train services were delayed for 60 minutes."
    changed_substantive = replace(
        substantive,
        claim=material_delay,
        supporting_excerpt=material_delay,
        rendered_assertion_zh_hant_hk="列車服務延誤咗六十分鐘",
    )
    supported = replace(
        package,
        passages=(f"HK-01: {headline.claim}\n{material_delay}",),
        substantive_new_information=(headline.claim, material_delay),
        governed_claims=(headline, changed_substantive),
        qualification_evidence=(
            package.qualification_evidence[0],
            QualificationEvidence(
                Evid012QualificationTest.ESSENTIAL_SERVICE_DISRUPTION,
                changed_substantive.claim_id,
                f"qualification:{changed_substantive.claim_id}",
                (
                    ("service_kind", "TRANSPORT"),
                    ("event_polarity", "AFFIRMED"),
                    ("duration_relation", "DISRUPTION_DURATION"),
                    ("duration_minutes", "60"),
                    ("affected_group", material_delay),
                ),
            ),
        ),
    )

    decision = DeterministicWriteAdmission().decide(
        candidate, supported, decided_at="2026-08-20T00:00:00.000000Z"
    )

    assert decision.decision == "WRITE_READY"


@pytest.mark.parametrize(
    ("fact", "duration_minutes", "expected"),
    (
        ("Train service was delayed for two hours.", "120", "WRITE_READY"),
        ("列車服務延誤兩小時。", "120", "WRITE_READY"),
        ("Train service was delayed for twenty two hours.", "120", "HOLD"),
        ("列車服務延誤二十二小時。", "120", "HOLD"),
        ("Train service was delayed for two hours.", "60", "HOLD"),
        ("Train service was delayed for two minutes.", "120", "HOLD"),
    ),
)
def test_localised_material_duration_is_exact(
    fact: str, duration_minutes: str, expected: str
) -> None:
    candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    changed_substantive = replace(
        substantive,
        claim=fact,
        supporting_excerpt=fact,
        rendered_assertion_zh_hant_hk="列車服務證實出現達兩小時延誤",
    )
    checked = replace(
        package,
        passages=(f"HK-01: {headline.claim}\n{fact}",),
        substantive_new_information=(headline.claim, fact),
        governed_claims=(headline, changed_substantive),
        qualification_evidence=(
            package.qualification_evidence[0],
            QualificationEvidence(
                Evid012QualificationTest.ESSENTIAL_SERVICE_DISRUPTION,
                changed_substantive.claim_id,
                f"qualification:{changed_substantive.claim_id}",
                (
                    ("service_kind", "TRANSPORT"),
                    ("event_polarity", "AFFIRMED"),
                    ("duration_relation", "DISRUPTION_DURATION"),
                    ("duration_minutes", duration_minutes),
                    ("affected_group", fact),
                ),
            ),
        ),
    )

    decision = DeterministicWriteAdmission().decide(
        candidate, checked, decided_at="2026-08-20T00:00:00.000000Z"
    )

    assert decision.decision == expected


@pytest.mark.parametrize(
    ("fact", "duration_minutes", "expected"),
    (
        ("Train service had a twenty two hours delay.", "120", False),
        ("Train service had a twenty-two-hour delay.", "120", False),
        ("A 2.5-hour delay affected the train service.", "300", False),
        ("A 1/2-hour delay affected the train service.", "120", False),
        ("A 2,500-hour delay affected the train service.", "30000", False),
        ("Train service had a two hours delay.", "120", True),
        ("A two-hour delay followed a 0.5-hour inspection.", "120", True),
        ("A twenty-two-hour delay ended before a two-hour delay began.", "120", True),
    ),
)
def test_duration_does_not_slice_a_compound_number(
    fact: str, duration_minutes: str, expected: bool
) -> None:
    claim = SimpleNamespace(claim=fact, supporting_excerpt=fact)
    assert _duration_is_exactly_supported(claim, duration_minutes) is expected


@pytest.mark.parametrize(
    ("fact", "expected"),
    (
        (
            (
                "No 60-minute service disruption occurred; "
                "Route 60 train was delayed by two minutes."
            ),
            "HOLD",
        ),
        ("Train services face delays of up to 60 minutes.", "WRITE_READY"),
        ("Officials denied train delays of 60 minutes.", "HOLD"),
        ("Officials denied reports of delays of up to 60 minutes.", "HOLD"),
        ("There were zero 60-minute service disruptions.", "HOLD"),
        ("Officials ruled out delays of up to 60 minutes.", "HOLD"),
        ("Officials dismissed reports of delays of up to 60 minutes.", "HOLD"),
        ("The route may be closed for 120 minutes.", "HOLD"),
        ("The route could be closed for 120 minutes.", "HOLD"),
        ("The route is expected to be closed for 120 minutes.", "HOLD"),
        ("當局澄清並不存在60分鐘延誤。", "HOLD"),
        ("當局排除會延誤60分鐘。", "HOLD"),
        ("The claim of a 60-minute delay is false.", "HOLD"),
        ("Reports of train delays of 60 minutes were incorrect.", "HOLD"),
        ("列車只延誤兩分鐘，原定車程為60分鐘。", "HOLD"),
    ),
)
def test_material_disruption_duration_requires_positive_exact_relation(
    fact: str, expected: str
) -> None:
    candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    changed_substantive = replace(
        substantive,
        claim=fact,
        supporting_excerpt=fact,
        rendered_assertion_zh_hant_hk="列車服務最新延誤安排已經確認",
    )
    checked = replace(
        package,
        passages=(f"HK-01: {headline.claim}\n{fact}",),
        substantive_new_information=(headline.claim, fact),
        governed_claims=(headline, changed_substantive),
        qualification_evidence=(
            package.qualification_evidence[0],
            QualificationEvidence(
                Evid012QualificationTest.ESSENTIAL_SERVICE_DISRUPTION,
                changed_substantive.claim_id,
                f"qualification:{changed_substantive.claim_id}",
                (
                    ("service_kind", "TRANSPORT"),
                    ("event_polarity", "AFFIRMED"),
                    ("duration_relation", "DISRUPTION_DURATION"),
                    ("duration_minutes", "60"),
                    ("affected_group", fact),
                ),
            ),
        ),
    )

    decision = DeterministicWriteAdmission().decide(
        candidate, checked, decided_at="2026-08-20T00:00:00.000000Z"
    )

    assert decision.decision == expected


def test_self_reported_pass_gates_without_governed_claims_fail_closed() -> None:
    candidate, package = _candidate_package()
    ungoverned = replace(
        package,
        governed_claims=(),
        evidence_gate_evidence=(),
    )

    decision = DeterministicWriteAdmission().decide(
        candidate, ungoverned, decided_at="2026-08-20T00:00:00.000000Z"
    )

    assert decision.decision == "HOLD"
    assert "MISSING_GOVERNED_CLAIMS" in decision.stable_reason_codes
    assert "CLAIM_TRACEABILITY_PROVENANCE_NOT_PASS" in decision.stable_reason_codes


def test_simplified_rendering_fails_versioned_zh_hant_hk_gate() -> None:
    candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    simplified = replace(
        headline,
        rendered_assertion_zh_hant_hk="官方说新安排",
    )
    decision = DeterministicWriteAdmission().decide(
        candidate,
        replace(package, governed_claims=(simplified, substantive)),
        decided_at="2026-08-20T00:00:00.000000Z",
    )
    assert decision.decision == "HOLD"
    assert "INVALID_GOVERNED_CLAIM_EVIDENCE" in decision.stable_reason_codes


def test_qualification_detail_must_be_exact_in_bound_governed_claim() -> None:
    candidate, package = _candidate_package()
    unsupported = replace(
        package,
        qualification_evidence=(
            package.qualification_evidence[0],
            QualificationEvidence(
                Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE,
                package.governed_claims[1].claim_id,
                f"qualification:{package.governed_claims[1].claim_id}",
                (
                    ("action_class", "OFFICIAL_DEADLINE"),
                    ("event_polarity", "AFFIRMED"),
                    ("action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"),
                    (
                        "material_relation_span",
                        package.governed_claims[1].claim,
                    ),
                    ("reader_action", "An unsupported reader action."),
                ),
            ),
        ),
    )

    decision = DeterministicWriteAdmission().decide(
        candidate, unsupported, decided_at="2026-08-20T00:00:00.000000Z"
    )
    assert decision.decision == "HOLD"
    assert decision.stable_reason_codes == ("QUALIFICATION_EVIDENCE_NOT_EXACT",)


@pytest.mark.parametrize(
    ("test", "claim_text", "test_evidence"),
    (
        (
            Evid012QualificationTest.LAW_RIGHT_STATUS_POLICY,
            "Officials confirmed the policy has not changed.",
            (
                ("change_kind", "PUBLIC_POLICY"),
                ("event_polarity", "AFFIRMED"),
                ("change_relation", "NEW_OR_CHANGED_STATE"),
                ("new_state", "Officials confirmed the policy has not changed."),
            ),
        ),
        (
            Evid012QualificationTest.SAFETY_OR_PUBLIC_HEALTH,
            "Officials confirmed there is no public-health risk.",
            (
                ("effect_class", "PUBLIC_HEALTH_WARNING"),
                ("event_polarity", "AFFIRMED"),
                ("effect_relation", "MATERIAL_EFFECT"),
                (
                    "affected_group",
                    "Officials confirmed there is no public-health risk.",
                ),
            ),
        ),
        (
            Evid012QualificationTest.HOUSEHOLD_PRACTICAL_EFFECT,
            "Officials confirmed there is no effect on household money.",
            (
                ("domain", "MONEY"),
                ("event_polarity", "AFFIRMED"),
                ("effect_relation", "MATERIAL_PRACTICAL_EFFECT"),
                (
                    "practical_effect",
                    "Officials confirmed there is no effect on household money.",
                ),
            ),
        ),
        (
            Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE,
            "The deadline did not change from 30 September.",
            (
                ("action_class", "OFFICIAL_DEADLINE"),
                ("event_polarity", "AFFIRMED"),
                ("action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"),
                ("reader_action", "The deadline did not change from 30 September."),
            ),
        ),
        (
            Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE,
            "The deadline remains the same.",
            (
                ("action_class", "OFFICIAL_DEADLINE"),
                ("event_polarity", "AFFIRMED"),
                ("action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"),
                ("reader_action", "The deadline remains the same."),
            ),
        ),
        (
            Evid012QualificationTest.LAW_RIGHT_STATUS_POLICY,
            "Officials said the policy remains the same.",
            (
                ("change_kind", "PUBLIC_POLICY"),
                ("event_polarity", "AFFIRMED"),
                ("change_relation", "NEW_OR_CHANGED_STATE"),
                ("new_state", "Officials said the policy remains the same."),
            ),
        ),
        (
            Evid012QualificationTest.SAFETY_OR_PUBLIC_HEALTH,
            "Officials said public-health risk is absent.",
            (
                ("effect_class", "PUBLIC_HEALTH_WARNING"),
                ("event_polarity", "AFFIRMED"),
                ("effect_relation", "MATERIAL_EFFECT"),
                ("affected_group", "Officials said public-health risk is absent."),
            ),
        ),
        (
            Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE,
            "政府並未更改申請限期",
            (
                ("action_class", "OFFICIAL_DEADLINE"),
                ("event_polarity", "AFFIRMED"),
                ("action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"),
                ("reader_action", "政府並未更改申請限期"),
            ),
        ),
        (
            Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE,
            "政府未曾更改申請限期",
            (
                ("action_class", "OFFICIAL_DEADLINE"),
                ("event_polarity", "AFFIRMED"),
                ("action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"),
                ("reader_action", "政府未曾更改申請限期"),
            ),
        ),
        (
            Evid012QualificationTest.SAFETY_OR_PUBLIC_HEALTH,
            "Officials disputed reports of a public-health risk.",
            (
                ("effect_class", "PUBLIC_HEALTH_WARNING"),
                ("event_polarity", "AFFIRMED"),
                ("effect_relation", "MATERIAL_EFFECT"),
                (
                    "affected_group",
                    "Officials disputed reports of a public-health risk.",
                ),
            ),
        ),
        *(
            (
                Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE,
                claim,
                (
                    ("action_class", "OFFICIAL_DEADLINE"),
                    ("event_polarity", "AFFIRMED"),
                    ("action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"),
                    ("reader_action", claim),
                ),
            )
            for claim in (
                "Officials failed to change the deadline.",
                "Officials declined to change the deadline.",
                "Officials are unlikely to change the deadline.",
                "Officials refuted claims that the deadline changed.",
                "Officials rebutted reports that the deadline changed.",
                "Officials are investigating whether the deadline changed.",
                "Officials asked whether the deadline changed.",
                "The authority is reviewing a new deadline.",
                "Officials discussed reports that the deadline changed.",
                "The deadline allegedly changed.",
                "網傳申請限期已改",
                "Sources suggested that the deadline changed.",
                "It is unclear if the deadline changed.",
                "It is doubtful that the deadline changed.",
                "There is scant evidence the deadline changed.",
                "The unverified claim says the deadline changed.",
                "Sources speculated that the deadline changed.",
                "政府拒絕更改申請限期",
                "政府放棄更改申請限期",
                "政府擱置更改申請限期",
                "政府駁斥申請限期已改",
            )
        ),
    ),
)
def test_negated_or_unchanged_fact_cannot_satisfy_qualification(
    test: Evid012QualificationTest,
    claim_text: str,
    test_evidence: tuple[tuple[str, str], ...],
) -> None:
    candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    changed = replace(
        substantive,
        claim=claim_text,
        supporting_excerpt=claim_text,
        rendered_assertion_zh_hant_hk="官方確認相關情況維持不變",
    )
    test_evidence = (*test_evidence, ("material_relation_span", claim_text))
    checked = replace(
        package,
        passages=(f"{headline.claim}\n{claim_text}",),
        substantive_new_information=(headline.claim, claim_text),
        governed_claims=(headline, changed),
        qualification_evidence=(
            package.qualification_evidence[0],
            QualificationEvidence(
                test,
                changed.claim_id,
                f"qualification:{changed.claim_id}",
                test_evidence,
            ),
        ),
    )

    decision = DeterministicWriteAdmission().decide(
        candidate, checked, decided_at="2026-08-20T00:00:00Z"
    )

    assert decision.decision == "HOLD"
    assert decision.stable_reason_codes == ("QUALIFICATION_EVIDENCE_NOT_EXACT",)


def test_positive_material_clause_is_not_tainted_by_unrelated_negative_clause() -> None:
    candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    full_claim = "政府公布新限期，但現有申請不受影響"
    material_clause = "政府公布新限期"
    changed = replace(
        substantive,
        claim=full_claim,
        supporting_excerpt=full_claim,
        rendered_assertion_zh_hant_hk="政府公布咗新限期",
    )
    checked = replace(
        package,
        passages=(f"{headline.claim}\n{full_claim}",),
        substantive_new_information=(headline.claim, full_claim),
        governed_claims=(headline, changed),
        qualification_evidence=(
            package.qualification_evidence[0],
            QualificationEvidence(
                Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE,
                changed.claim_id,
                f"qualification:{changed.claim_id}",
                (
                    ("action_class", "OFFICIAL_DEADLINE"),
                    ("event_polarity", "AFFIRMED"),
                    ("action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"),
                    ("material_relation_span", material_clause),
                    ("reader_action", material_clause),
                ),
            ),
        ),
    )

    decision = DeterministicWriteAdmission().decide(
        candidate, checked, decided_at="2026-08-20T00:00:00Z"
    )

    assert decision.decision == "WRITE_READY"


def test_admission_identity_binds_exact_evidence_package_digest() -> None:
    candidate, package = _candidate_package()
    changed = replace(package, passages=(package.passages[0] + " Corrected version.",))
    policy = DeterministicWriteAdmission()

    first = policy.decide(candidate, package, decided_at="2026-08-20T00:00:00.000000Z")
    replay = policy.decide(candidate, package, decided_at="2026-08-21T00:00:00.000000Z")
    drifted = policy.decide(
        candidate, changed, decided_at="2026-08-20T00:00:00.000000Z"
    )

    assert first.decision == "WRITE_READY"
    assert replay.decision_id == first.decision_id
    assert drifted.decision_id != first.decision_id
    assert drifted.evidence_package_digest != first.evidence_package_digest


def test_two_write_ready_candidates_never_trigger_three_replacements(
    tmp_path: Path,
) -> None:
    proving = _proving(tmp_path)
    unpublished = tmp_path / "unpublished_store.sqlite3"
    writer = RecordingFixtureWriter()

    report = run_cycle(
        proving_store=str(proving),
        unpublished_store=str(unpublished),
        writer=writer,
        evidence_package_builder=_qualified_builder(frozenset({"HK-01", "UK-01"})),
        max_writes=5,
        clock=_CLOCK,
    )

    assert report.write_ready == 2
    assert report.admission_hold == 1
    assert report.selected_write_ready == 2
    assert report.candidate_attempts == 2
    assert report.provider_dispatches == 2
    assert report.minted == 2
    assert len(writer.calls) == 2


def test_five_ready_candidates_with_valid_primary_results_insert_five(
    tmp_path: Path,
) -> None:
    proving = _proving(tmp_path)
    expanded = b"""<feed>
      <entry><id>uk-1</id><title>Rule one changed</title><link href="https://example.test/uk-1"/><summary>Rule one took effect.</summary></entry>
      <entry><id>uk-2</id><title>Rule two changed</title><link href="https://example.test/uk-2"/><summary>Rule two took effect.</summary></entry>
      <entry><id>uk-3</id><title>Rule three changed</title><link href="https://example.test/uk-3"/><summary>Rule three took effect.</summary></entry>
    </feed>"""
    connection = sqlite3.connect(proving)
    _store_body(connection, digest_bytes(expanded), expanded)
    connection.execute(
        "UPDATE proving_observations SET body_digest=?, item_count=3 "
        "WHERE source_id='UK-01'",
        (digest_bytes(expanded),),
    )
    connection.commit()
    connection.close()
    unpublished = tmp_path / "unpublished_store.sqlite3"
    writer = RecordingFixtureWriter()

    report = run_cycle(
        proving_store=str(proving),
        unpublished_store=str(unpublished),
        writer=writer,
        evidence_package_builder=_qualified_builder(
            frozenset({"HK-01", "UK-01", "UK-02"})
        ),
        max_writes=5,
        clock=_CLOCK,
    )

    assert report.selected_write_ready == 5
    assert report.candidate_attempts == 5
    assert report.provider_dispatches == 5
    assert report.minted == 5
    assert len(writer.calls) == 5
    assert len(list_payloads(str(unpublished))) == 5


def _valid_cli_json(prompt: str) -> str:
    lines = prompt.splitlines()
    title = next(
        line.removeprefix("必須輸出嘅 title：")
        for line in lines
        if line.startswith("必須輸出嘅 title：")
    )
    body = next(
        line.removeprefix("必須輸出嘅 body：")
        for line in lines
        if line.startswith("必須輸出嘅 body：")
    )
    raw_links = next(
        line.removeprefix("必須輸出嘅 evidence_links：")
        for line in lines
        if line.startswith("必須輸出嘅 evidence_links：")
    )
    return json.dumps(
        {
            "title": title,
            "body": body,
            "evidence_links": json.loads(raw_links),
        },
        ensure_ascii=False,
    )


def test_malformed_primary_then_one_valid_fallback_consumes_two_leaf_calls(
    tmp_path: Path,
) -> None:
    calls = {"primary": 0, "fallback": 0}

    def primary(_prompt: str) -> str:
        calls["primary"] += 1
        return "not-json"

    def fallback(prompt: str) -> str:
        calls["fallback"] += 1
        return _valid_cli_json(prompt)

    report = run_cycle(
        proving_store=str(_proving(tmp_path)),
        unpublished_store=str(tmp_path / "unpublished_store.sqlite3"),
        writer=CliChainWriter(primary=primary, fallback=fallback),
        evidence_package_builder=_qualified_builder(frozenset({"HK-01"})),
        clock=_CLOCK,
    )

    assert calls == {"primary": 1, "fallback": 1}
    assert report.provider_dispatches == 2
    assert report.primary_dispatches == 1
    assert report.fallback_dispatches == 1
    assert report.minted == 1


def test_second_candidate_cannot_consume_second_cycle_fallback(tmp_path: Path) -> None:
    calls = {"primary": 0, "fallback": 0}

    def primary(_prompt: str) -> str:
        calls["primary"] += 1
        return "malformed"

    def fallback(prompt: str) -> str:
        calls["fallback"] += 1
        return _valid_cli_json(prompt)

    report = run_cycle(
        proving_store=str(_proving(tmp_path)),
        unpublished_store=str(tmp_path / "unpublished_store.sqlite3"),
        writer=CliChainWriter(primary=primary, fallback=fallback),
        evidence_package_builder=_qualified_builder(frozenset({"HK-01", "UK-01"})),
        clock=_CLOCK,
    )

    assert calls == {"primary": 2, "fallback": 1}
    assert report.candidate_attempts == 2
    assert report.provider_dispatches == 3
    assert report.fallback_dispatches == 1
    assert report.minted == 1
    assert report.no_useful_output_circuit_open is True


def test_exhausted_candidate_routes_open_no_useful_output_circuit(
    tmp_path: Path,
) -> None:
    calls = 0

    def unusable(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "malformed"

    report = run_cycle(
        proving_store=str(_proving(tmp_path)),
        unpublished_store=str(tmp_path / "unpublished_store.sqlite3"),
        writer=CliChainWriter(primary=unusable, fallback=unusable),
        evidence_package_builder=_qualified_builder(
            frozenset({"HK-01", "UK-01", "UK-02"})
        ),
        clock=_CLOCK,
    )

    assert report.selected_write_ready == 3
    assert report.candidate_attempts == 1
    assert report.provider_dispatches == 2
    assert calls == 2
    assert report.no_useful_output_circuit_open is True
    assert report.minted == 0


def test_systemic_authentication_failure_opens_route_circuit_immediately(
    tmp_path: Path,
) -> None:
    class AuthFailureWriter(FixtureWriter):
        def __init__(self) -> None:
            self.calls = 0

        def dispatch(self, *args: object, **kwargs: object) -> WriterCopy:
            self.calls += 1
            raise WriterDispatchError(
                "provider authentication failed",
                failure_class="SYSTEMIC",
                reason_code="PROVIDER_AUTHENTICATION_FAILURE",
            )

    writer = AuthFailureWriter()
    report = run_cycle(
        proving_store=str(_proving(tmp_path)),
        unpublished_store=str(tmp_path / "unpublished_store.sqlite3"),
        writer=writer,
        evidence_package_builder=_qualified_builder(
            frozenset({"HK-01", "UK-01", "UK-02"})
        ),
        clock=_CLOCK,
    )

    assert writer.calls == 1
    assert report.provider_dispatches == 1
    assert report.writer_circuit_open is True
    assert report.writer_circuit_open_reason == "PROVIDER_AUTHENTICATION_FAILURE"
    assert report.minted == 0


@pytest.mark.parametrize(
    "message",
    (
        "401 Unauthorized",
        "login required",
        "invalid API key",
        "permission denied",
        "invalid model configuration",
        "HTTP 429 Too Many Requests",
        "402 Payment Required",
        "API Error: 429",
        "Request failed [429]",
        "Error 402",
        "APIError: 429",
        "HTTPError: 429",
        '{"statusCode":429}',
        '{"status":429}',
        '{"code":402}',
        "429 rate_limit_exceeded",
        '{"type":"rate_limit_error"}',
        "grok writer failed: 429",
    ),
)
def test_actual_cli_classifier_marks_provider_control_failures_systemic(
    message: str,
) -> None:
    candidate, package = _candidate_package()

    def primary(_prompt: str) -> str:
        raise RuntimeError(message)

    writer = CliChainWriter(
        primary=primary,
        fallback=lambda _prompt: pytest.fail(
            "systemic failure must not reach fallback"
        ),
    )
    with pytest.raises(WriterDispatchError) as caught:
        writer.dispatch(candidate, package, route="PRIMARY")
    assert caught.value.failure_class == "SYSTEMIC"
    assert caught.value.reason_code == "SYSTEMIC_PROVIDER_FAILURE"


@pytest.mark.parametrize("offset", (401, 403, 429))
def test_json_character_offset_remains_fallback_eligible(offset: int) -> None:
    candidate, package = _candidate_package()

    def primary(_prompt: str) -> str:
        raise RuntimeError(f"JSON parse error at column {offset + 1} (char {offset})")

    writer = CliChainWriter(primary=primary, fallback=lambda _prompt: "unused")
    with pytest.raises(WriterDispatchError) as caught:
        writer.dispatch(candidate, package, route="PRIMARY")
    assert caught.value.failure_class == "FALLBACK_ELIGIBLE"


def test_required_surface_copy_passes_and_one_extra_character_fails() -> None:
    candidate, package = _candidate_package()
    title, body, links = required_surface_copy(package)
    copy = WriterCopy(
        title=title,
        body=body,
        writer_id="exact-assembly",
        evidence_package_digest=package.digest,
        evidence_links=links,
    )
    extra = WriterCopy(
        title=title,
        body=body + "。",
        writer_id="exact-assembly",
        evidence_package_digest=package.digest,
        evidence_links=links,
    )

    exact = validate_writer_copy(copy, package)
    drifted = validate_writer_copy(extra, package)

    assert all(item.result == "PASS" for item in exact)
    assert any(
        item.result == "FAIL"
        and item.reason_code
        in {
            "DUPLICATE_OR_MISPLACED_GOVERNED_CLAIM",
            "UNSUPPORTED_CLAIM_RESIDUE",
        }
        for item in drifted
    )


def test_writer_prompt_is_claim_assembly_not_journalism() -> None:
    from newsroom.control_plane.writer import _prompt

    candidate, package = _candidate_package()
    title, body, _links = required_surface_copy(package)
    prompt = _prompt(candidate, package)

    assert "必須原創改寫" not in prompt
    assert "原創記者" not in prompt
    assert "claim 拼裝器" in prompt
    assert f"必須輸出嘅 title：{title}" in prompt
    assert f"必須輸出嘅 body：{body}" in prompt
    assert "來源（禁止入稿）" not in prompt
    assert all(passage not in prompt for passage in package.passages)
    assert CONT_WRITER_PROMPT_CONTRACT_VERSION.endswith(".v3")


def test_rejected_copy_text_is_retained_on_draft_outcome(tmp_path: Path) -> None:
    class ExtraCharacterWriter(FixtureWriter):
        def dispatch(
            self,
            candidate: StoryCandidateRecord,
            package: EvidencePackage,
            *,
            route: str,
        ) -> WriterCopy:
            del route
            copy = super().write(candidate, package)
            return WriterCopy(
                title=copy.title,
                body=copy.body + "。",
                writer_id=copy.writer_id,
                evidence_package_digest=copy.evidence_package_digest,
                evidence_links=copy.evidence_links,
            )

    unpublished = tmp_path / "unpublished_store.sqlite3"
    report = run_cycle(
        proving_store=str(_proving(tmp_path)),
        unpublished_store=str(unpublished),
        writer=ExtraCharacterWriter(),
        evidence_package_builder=_qualified_builder(frozenset({"HK-01"})),
        clock=_CLOCK,
    )
    connection = sqlite3.connect(unpublished)
    retained = json.loads(
        connection.execute(
            "SELECT record_json FROM unpublished_draft_outcomes"
        ).fetchone()[0]
    )
    connection.close()

    assert report.minted == 0
    assert retained["outcome"] == "REJECT"
    assert retained["rejected_title"]
    assert retained["rejected_body"].endswith("。")


def test_filler_output_is_rejected_and_never_inserted(tmp_path: Path) -> None:
    class FillerWriter(FixtureWriter):
        def dispatch(
            self,
            candidate: StoryCandidateRecord,
            package: EvidencePackage,
            *,
            route: str,
        ) -> WriterCopy:
            rendered = package.governed_claims[0].rendered_assertion_zh_hant_hk
            return WriterCopy(
                title=f"【未出版】{rendered}",
                body=f"值得注意的是，{rendered}",
                writer_id=f"filler-{route}",
                evidence_package_digest=package.digest,
                evidence_links=(
                    WriterEvidenceLink(
                        f"claim:{candidate.candidate_id}:headline",
                        rendered,
                    ),
                ),
            )

    unpublished = tmp_path / "unpublished_store.sqlite3"
    report = run_cycle(
        proving_store=str(_proving(tmp_path)),
        unpublished_store=str(unpublished),
        writer=FillerWriter(),
        evidence_package_builder=_qualified_builder(frozenset({"HK-01"})),
        clock=_CLOCK,
    )

    assert report.provider_dispatches == 2
    assert report.draft_reject == 1
    assert report.minted == 0
    assert list_payloads(str(unpublished)) == ()


def test_syntactically_valid_unmapped_claim_is_rejected(tmp_path: Path) -> None:
    class UnsupportedWriter(FixtureWriter):
        def dispatch(
            self,
            candidate: StoryCandidateRecord,
            package: EvidencePackage,
            *,
            route: str,
        ) -> WriterCopy:
            rendered = package.governed_claims[0].rendered_assertion_zh_hant_hk
            return WriterCopy(
                title=f"【未出版】{rendered}",
                body=f"本報根據已核實證據報道：{rendered}。市民明天必須撤離。",
                writer_id=f"unsupported-{route}",
                evidence_package_digest=package.digest,
                evidence_links=(
                    WriterEvidenceLink(
                        f"claim:{candidate.candidate_id}:headline",
                        rendered,
                    ),
                ),
            )

    unpublished = tmp_path / "unpublished_store.sqlite3"
    report = run_cycle(
        proving_store=str(_proving(tmp_path)),
        unpublished_store=str(unpublished),
        writer=UnsupportedWriter(),
        evidence_package_builder=_qualified_builder(frozenset({"HK-01"})),
        clock=_CLOCK,
    )

    assert report.draft_reject == 1
    assert (
        "DUPLICATE_OR_MISPLACED_GOVERNED_CLAIM",
        1,
    ) in report.draft_reason_counts
    assert list_payloads(str(unpublished)) == ()


def test_unrelated_claim_in_same_sentence_fails_governed_entailment() -> None:
    _candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    copy = WriterCopy(
        title=f"【未出版】{headline.rendered_assertion_zh_hant_hk}",
        body=(
            "本報根據已核實證據報道："
            f"{substantive.rendered_assertion_zh_hant_hk}；李家超親自拍板"
        ),
        writer_id="adversarial-writer",
        evidence_package_digest=package.digest,
        evidence_links=tuple(
            WriterEvidenceLink(item.claim_id, item.rendered_assertion_zh_hant_hk)
            for item in package.governed_claims
        ),
    )

    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, package)
        if item.result == "FAIL"
    }
    assert "UNSUPPORTED_CLAIM_RESIDUE" in failed


@pytest.mark.parametrize(
    "variant",
    ("duplicate-body", "duplicate-title", "duplicate-title-scaffold"),
)
def test_repeated_governed_content_fails_exact_once_structure(variant: str) -> None:
    _candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    title = f"【未出版】{headline.rendered_assertion_zh_hant_hk}"
    body = f"本報根據已核實證據報道：{substantive.rendered_assertion_zh_hant_hk}"
    if variant == "duplicate-body":
        body += f"；{substantive.rendered_assertion_zh_hant_hk}"
    elif variant == "duplicate-title":
        title += headline.rendered_assertion_zh_hant_hk
    else:
        title = f"【未出版】【未出版】{headline.rendered_assertion_zh_hant_hk}"
    copy = WriterCopy(
        title=title,
        body=body,
        writer_id="repetitive-writer",
        evidence_package_digest=package.digest,
        evidence_links=tuple(
            WriterEvidenceLink(item.claim_id, item.rendered_assertion_zh_hant_hk)
            for item in package.governed_claims
        ),
    )

    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, package)
        if item.result == "FAIL"
    }

    assert "DUPLICATE_OR_MISPLACED_GOVERNED_CLAIM" in failed


def test_claim_link_cannot_rebind_another_governed_claim_identity() -> None:
    _candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    copy = WriterCopy(
        title=f"【未出版】{headline.rendered_assertion_zh_hant_hk}",
        body=(f"本報根據已核實證據報道：{substantive.rendered_assertion_zh_hant_hk}"),
        writer_id="adversarial-writer",
        evidence_package_digest=package.digest,
        evidence_links=(
            WriterEvidenceLink(
                substantive.claim_id,
                headline.rendered_assertion_zh_hant_hk,
            ),
            WriterEvidenceLink(
                substantive.claim_id,
                substantive.rendered_assertion_zh_hant_hk,
            ),
        ),
    )

    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, package)
        if item.result == "FAIL"
    }
    assert "UNSUPPORTED_MATERIAL_CLAIM" in failed


def test_headline_only_draft_cannot_omit_substantive_claim() -> None:
    _candidate, package = _candidate_package()
    headline = package.governed_claims[0]
    copy = WriterCopy(
        title=f"【未出版】{headline.rendered_assertion_zh_hant_hk}",
        body=f"本報根據已核實證據報道：{headline.rendered_assertion_zh_hant_hk}",
        writer_id="headline-only-writer",
        evidence_package_digest=package.digest,
        evidence_links=(
            WriterEvidenceLink(
                headline.claim_id,
                headline.rendered_assertion_zh_hant_hk,
            ),
        ),
    )

    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, package)
        if item.result == "FAIL"
    }
    assert "REQUIRED_GOVERNED_CLAIM_MISSING" in failed


def test_verbatim_source_expression_fails_originality_boundary() -> None:
    _candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    copy = WriterCopy(
        title=f"【未出版】{headline.rendered_assertion_zh_hant_hk}",
        body=(
            "本報根據已核實證據報道："
            f"{substantive.rendered_assertion_zh_hant_hk}\n{package.passages[0]}"
        ),
        writer_id="copying-writer",
        evidence_package_digest=package.digest,
        evidence_links=tuple(
            WriterEvidenceLink(item.claim_id, item.rendered_assertion_zh_hant_hk)
            for item in package.governed_claims
        ),
    )

    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, package)
        if item.result == "FAIL"
    }
    assert "VERBATIM_SOURCE_EXPRESSION" in failed
    assert "NOT_COMPLETED_ZH_HANT_HK_REPORT" in failed


def test_punctuation_cannot_split_copied_source_sequence() -> None:
    _candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    source = package.passages[0]
    punctuated_copy = "，".join(
        source[index : index + 11] for index in range(0, len(source), 11)
    )
    copy = WriterCopy(
        title=f"【未出版】{headline.rendered_assertion_zh_hant_hk}",
        body=(
            "本報根據已核實證據報道："
            f"{substantive.rendered_assertion_zh_hant_hk}\n{punctuated_copy}"
        ),
        writer_id="punctuation-copying-writer",
        evidence_package_digest=package.digest,
        evidence_links=tuple(
            WriterEvidenceLink(item.claim_id, item.rendered_assertion_zh_hant_hk)
            for item in package.governed_claims
        ),
    )
    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, package)
        if item.result == "FAIL"
    }
    assert "VERBATIM_SOURCE_EXPRESSION" in failed


def test_inserted_characters_cannot_break_source_sequence_alignment() -> None:
    _candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    source = package.passages[0]
    interrupted_copy = "的".join(
        source[index : index + 11] for index in range(0, len(source), 11)
    )
    copy = WriterCopy(
        title=f"【未出版】{headline.rendered_assertion_zh_hant_hk}",
        body=(
            "本報根據已核實證據報道："
            f"{substantive.rendered_assertion_zh_hant_hk}\n{interrupted_copy}"
        ),
        writer_id="interrupted-copying-writer",
        evidence_package_digest=package.digest,
        evidence_links=tuple(
            WriterEvidenceLink(item.claim_id, item.rendered_assertion_zh_hant_hk)
            for item in package.governed_claims
        ),
    )
    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, package)
        if item.result == "FAIL"
    }
    assert "VERBATIM_SOURCE_EXPRESSION" in failed


def test_short_block_insertions_cannot_break_source_sequence_alignment() -> None:
    _candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    source = package.passages[0]
    interrupted_copy = "的".join(
        source[index : index + 3] for index in range(0, len(source), 3)
    )
    copy = WriterCopy(
        title=f"【未出版】{headline.rendered_assertion_zh_hant_hk}",
        body=(
            "本報根據已核實證據報道："
            f"{substantive.rendered_assertion_zh_hant_hk}\n{interrupted_copy}"
        ),
        writer_id="short-block-copying-writer",
        evidence_package_digest=package.digest,
        evidence_links=tuple(
            WriterEvidenceLink(item.claim_id, item.rendered_assertion_zh_hant_hk)
            for item in package.governed_claims
        ),
    )
    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, package)
        if item.result == "FAIL"
    }
    assert "VERBATIM_SOURCE_EXPRESSION" in failed


def test_short_source_sentence_cannot_be_copied_with_punctuation_only() -> None:
    _candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    source_sentence = "區議會今日復會"
    changed_substantive = replace(
        substantive,
        claim=source_sentence,
        supporting_excerpt=source_sentence,
        rendered_assertion_zh_hant_hk=f"{source_sentence}。",
    )
    guarded_package = replace(
        package,
        passages=(source_sentence,),
        substantive_new_information=(source_sentence,),
        governed_claims=(headline, changed_substantive),
    )
    copy = WriterCopy(
        title=f"【未出版】{headline.rendered_assertion_zh_hant_hk}",
        body=f"本報根據已核實證據報道：{changed_substantive.rendered_assertion_zh_hant_hk}",
        writer_id="short-sentence-copying-writer",
        evidence_package_digest=guarded_package.digest,
        evidence_links=tuple(
            WriterEvidenceLink(item.claim_id, item.rendered_assertion_zh_hant_hk)
            for item in guarded_package.governed_claims
        ),
    )
    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, guarded_package)
        if item.result == "FAIL"
    }
    assert "VERBATIM_SOURCE_EXPRESSION" in failed


def test_approved_named_entity_is_removed_before_originality_overlap() -> None:
    _candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    entity = "Hong Kong Monetary Authority"
    changed_headline = replace(
        headline,
        named_entity_evidence=(
            (entity, "ORGANISATION", "entity:hong-kong-monetary-authority"),
        ),
        named_entities=(entity,),
        rendered_named_entities=(entity,),
        rendered_assertion_zh_hant_hk=f"{entity}公布咗最新安排",
    )
    guarded_package = replace(
        package,
        passages=(f"{package.passages[0]}\n{entity}",),
        governed_claims=(changed_headline, substantive),
    )
    copy = WriterCopy(
        title=f"【未出版】{changed_headline.rendered_assertion_zh_hant_hk}",
        body=(f"本報根據已核實證據報道：{substantive.rendered_assertion_zh_hant_hk}"),
        writer_id="approved-entity-writer",
        evidence_package_digest=guarded_package.digest,
        evidence_links=tuple(
            WriterEvidenceLink(item.claim_id, item.rendered_assertion_zh_hant_hk)
            for item in guarded_package.governed_claims
        ),
    )
    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, guarded_package)
        if item.result == "FAIL"
    }
    assert "VERBATIM_SOURCE_EXPRESSION" not in failed


@pytest.mark.parametrize(
    ("claim_text", "entity"),
    (
        ("The deadline is now 30 September", "The deadline is now 30 September"),
        ("香港政府公布新限期安排", "香港政府公布新限期安排"),
        ("香港機場客運量創新高引起關注", "香港機場客運量創新高"),
        (
            "Update: Hong Kong Authority Launches New Payment System",
            "Hong Kong Authority Launches New Payment System",
        ),
        (
            "Government Opens New Housing Authority",
            "Government Opens New Housing Authority",
        ),
        (
            "Government Plans New Housing Authority",
            "Government Plans New Housing Authority",
        ),
        (
            "Government Creates New Housing Authority",
            "Government Creates New Housing Authority",
        ),
        (
            "Government Backs New Housing Authority",
            "Government Backs New Housing Authority",
        ),
    ),
)
def test_named_entity_cannot_exempt_an_entire_source_sentence(
    claim_text: str, entity: str
) -> None:
    _candidate, package = _candidate_package()
    substantive = package.governed_claims[1]

    with pytest.raises(ValueError, match="exact typed retained evidence"):
        replace(
            substantive,
            claim=claim_text,
            supporting_excerpt=claim_text,
            rendered_assertion_zh_hant_hk=entity,
            named_entity_evidence=((entity, "ORGANISATION", "entity:sentence"),),
            named_entities=(entity,),
            rendered_named_entities=(entity,),
        )


@pytest.mark.parametrize(
    ("source_claim", "rendered_assertion"),
    (
        ("香港政府推出新措施", "英國當局實施全新政策"),
        ("李家超公布新措施", "陳茂波宣布另一政策"),
        ("John Smith announced a new measure", "Jane Jones announced another policy"),
        ("行政長官李家超公布新措施", "財政司司長陳茂波公布新政策"),
        ("運輸署公布新安排", "教育局公布另一安排"),
        ("李家超出席會議", "陳茂波會見代表"),
        ("李家超簽署文件", "陳茂波視察工地"),
        ("深圳公布新安排", "北京公布另一安排"),
        ("上海公布新安排", "澳門公布另一安排"),
        ("廣州推出新安排", "巴黎推出新安排"),
        (
            "Department for Work and Pensions announced a change",
            "NHS England confirmed another change",
        ),
        (
            "Transport for London announced a change",
            "Northern Transport Directorate confirmed another change",
        ),
    ),
)
def test_named_entity_inventory_cannot_be_omitted_or_replaced(
    source_claim: str, rendered_assertion: str
) -> None:
    candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    changed = replace(
        headline,
        claim=source_claim,
        supporting_excerpt=source_claim,
        rendered_assertion_zh_hant_hk=rendered_assertion,
    )
    guarded = replace(
        package,
        passages=(f"{source_claim}\n{substantive.claim}",),
        governed_claims=(changed, substantive),
    )

    decision = DeterministicWriteAdmission().decide(
        candidate, guarded, decided_at="2026-08-20T00:00:00.000000Z"
    )

    assert decision.decision == "HOLD"
    assert "INVALID_GOVERNED_CLAIM_EVIDENCE" in decision.stable_reason_codes


def test_named_entity_evidence_binds_one_canonical_identity() -> None:
    _candidate, package = _candidate_package()
    substantive = package.governed_claims[1]

    with pytest.raises(ValueError, match="exact typed retained evidence"):
        replace(
            substantive,
            claim="李家超公布新措施",
            supporting_excerpt="李家超公布新措施",
            rendered_assertion_zh_hant_hk="陳茂波宣布另一政策",
            named_entity_evidence=(("李家超", "PERSON", "entity:chief-executive"),),
            named_entities=("李家超",),
            rendered_named_entities=("陳茂波",),
        )


def test_may_to_will_requires_retained_semantic_relation_evidence() -> None:
    candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    changed = replace(
        substantive,
        claim="The route may close.",
        supporting_excerpt="The route may close.",
        rendered_assertion_zh_hant_hk="相關路線將會關閉",
        semantic_relation_evidence_id="semantic:unretained-may-to-will",
    )
    guarded = replace(
        package,
        passages=(f"{headline.claim}\nThe route may close.",),
        governed_claims=(headline, changed),
    )
    copy = FixtureWriter().write(candidate, guarded)

    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, guarded)
        if item.result == "FAIL"
    }
    assert "CERTAINTY_EXCEEDS_EVIDENCE" in failed


@pytest.mark.parametrize(
    ("source_claim", "rendered_assertion", "source_fact", "rendered_fact"),
    (
        (
            "會議定於2026年12月31日上午9時30分舉行",
            "活動安排喺2026年12月31日上午9時30分開始",
            "2026年12月31日上午9時30分",
            "2026年12月31日上午9時30分",
        ),
        (
            "會議定於二零二六年十二月三十一日上午九時三十分舉行",
            "活動安排喺二零二六年十二月三十一日上午九時三十分開始",
            "二零二六年十二月三十一日上午九時三十分",
            "二零二六年十二月三十一日上午九時三十分",
        ),
        (
            "資助額增至一億二千三百四十五萬六千七百八十九元",
            "新安排提供一億二千三百四十五萬六千七百八十九元資助",
            "一億二千三百四十五萬六千七百八十九元",
            "一億二千三百四十五萬六千七百八十九元",
        ),
    ),
)
def test_approved_date_is_removed_before_originality_overlap(
    source_claim: str,
    rendered_assertion: str,
    source_fact: str,
    rendered_fact: str,
) -> None:
    _candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    changed_substantive = replace(
        substantive,
        claim=source_claim,
        supporting_excerpt=source_claim,
        rendered_assertion_zh_hant_hk=rendered_assertion,
        localised_factual_expressions=((source_fact, rendered_fact),),
    )
    guarded_package = replace(
        package,
        passages=(source_claim,),
        substantive_new_information=(source_claim,),
        governed_claims=(headline, changed_substantive),
    )
    copy = WriterCopy(
        title=f"【未出版】{headline.rendered_assertion_zh_hant_hk}",
        body=(
            f"本報根據已核實證據報道：{changed_substantive.rendered_assertion_zh_hant_hk}"
        ),
        writer_id="approved-date-writer",
        evidence_package_digest=guarded_package.digest,
        evidence_links=tuple(
            WriterEvidenceLink(item.claim_id, item.rendered_assertion_zh_hant_hk)
            for item in guarded_package.governed_claims
        ),
    )
    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, guarded_package)
        if item.result == "FAIL"
    }
    assert "VERBATIM_SOURCE_EXPRESSION" not in failed


@pytest.mark.parametrize(
    ("source_claim", "rendered_assertion", "source_fact", "rendered_fact"),
    (
        (
            "The meeting is on 31 December 2026 at 9:30.",
            "會議定喺2026年12月31日上午9時30分",
            "31 December 2026 at 9:30",
            "2026年12月31日上午9時30分",
        ),
        (
            "The grant is HK$100,000,000.",
            "資助額係一億元",
            "HK$100,000,000",
            "一億元",
        ),
        (
            "The delay lasted one hour.",
            "延誤持續一小時",
            "one hour",
            "一小時",
        ),
        (
            "The delay lasted 60 minutes.",
            "延誤持續六十分鐘",
            "60 minutes",
            "六十分鐘",
        ),
    ),
)
def test_controller_bound_localised_fact_passes_numeric_fidelity(
    source_claim: str,
    rendered_assertion: str,
    source_fact: str,
    rendered_fact: str,
) -> None:
    _candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    changed_substantive = replace(
        substantive,
        claim=source_claim,
        supporting_excerpt=source_claim,
        rendered_assertion_zh_hant_hk=rendered_assertion,
        localised_factual_expressions=((source_fact, rendered_fact),),
    )
    guarded_package = replace(
        package,
        passages=(source_claim,),
        substantive_new_information=(source_claim,),
        governed_claims=(headline, changed_substantive),
    )
    copy = WriterCopy(
        title=f"【未出版】{headline.rendered_assertion_zh_hant_hk}",
        body=f"本報根據已核實證據報道：{rendered_assertion}",
        writer_id="localised-fact-writer",
        evidence_package_digest=guarded_package.digest,
        evidence_links=tuple(
            WriterEvidenceLink(item.claim_id, item.rendered_assertion_zh_hant_hk)
            for item in guarded_package.governed_claims
        ),
    )

    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, guarded_package)
        if item.result == "FAIL"
    }

    assert "UNSUPPORTED_NUMBER_OR_DATE" not in failed


def test_unbound_localised_word_unit_fails_numeric_fidelity() -> None:
    _candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    changed_substantive = replace(
        substantive,
        claim="The delay lasted one hour.",
        supporting_excerpt="The delay lasted one hour.",
        rendered_assertion_zh_hant_hk="延誤持續一小時",
    )
    guarded_package = replace(
        package,
        passages=(changed_substantive.claim,),
        substantive_new_information=(changed_substantive.claim,),
        governed_claims=(headline, changed_substantive),
    )
    copy = WriterCopy(
        title=f"【未出版】{headline.rendered_assertion_zh_hant_hk}",
        body="本報根據已核實證據報道：延誤持續一小時",
        writer_id="unbound-localised-fact-writer",
        evidence_package_digest=guarded_package.digest,
        evidence_links=tuple(
            WriterEvidenceLink(item.claim_id, item.rendered_assertion_zh_hant_hk)
            for item in guarded_package.governed_claims
        ),
    )

    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, guarded_package)
        if item.result == "FAIL"
    }

    assert "UNSUPPORTED_NUMBER_OR_DATE" in failed


def test_localised_date_mapping_must_preserve_canonical_value() -> None:
    _candidate, package = _candidate_package()
    substantive = package.governed_claims[1]

    with pytest.raises(ValueError, match="equivalent exact claim facts"):
        replace(
            substantive,
            claim="The deadline is now 30 September.",
            supporting_excerpt="The deadline is now 30 September.",
            rendered_assertion_zh_hant_hk="新限期定喺十月三十日",
            localised_factual_expressions=(("30 September", "十月三十日"),),
        )


def test_localised_large_chinese_units_do_not_collide() -> None:
    _candidate, package = _candidate_package()
    substantive = package.governed_claims[1]

    with pytest.raises(ValueError, match="equivalent exact claim facts"):
        replace(
            substantive,
            claim="資助額增至一萬億元",
            supporting_excerpt="資助額增至一萬億元",
            rendered_assertion_zh_hant_hk="新資助額係一億零一萬元",
            localised_factual_expressions=(("一萬億元", "一億零一萬元"),),
        )


@pytest.mark.parametrize(
    ("source_fact", "rendered_fact"),
    (
        ("一百二元", "一百零二元"),
        ("一千二元", "一千零二元"),
        ("一萬二元", "一萬零二元"),
        ("萬億元", "零元"),
        ("一百二元", "一百三元"),
        ("一千二元", "一千三元"),
        ("一萬二元", "一萬三元"),
        ("萬億元", "億元"),
    ),
)
def test_ambiguous_chinese_number_shorthand_fails_closed(
    source_fact: str, rendered_fact: str
) -> None:
    _candidate, package = _candidate_package()
    substantive = package.governed_claims[1]

    with pytest.raises(ValueError, match="equivalent exact claim facts"):
        replace(
            substantive,
            claim=f"資助額增至{source_fact}",
            supporting_excerpt=f"資助額增至{source_fact}",
            rendered_assertion_zh_hant_hk=f"新資助額係{rendered_fact}",
            localised_factual_expressions=((source_fact, rendered_fact),),
        )


@pytest.mark.parametrize(
    ("source_fact", "rendered_fact"),
    (
        ("九月三十日上午十二時零分", "九月三十日下午十二時零分"),
        ("二月三十一日", "二月三十一日"),
    ),
)
def test_invalid_or_polarity_changed_date_fails_closed(
    source_fact: str, rendered_fact: str
) -> None:
    _candidate, package = _candidate_package()
    substantive = package.governed_claims[1]

    with pytest.raises(ValueError, match="equivalent exact claim facts"):
        replace(
            substantive,
            claim=f"會議定於{source_fact}",
            supporting_excerpt=f"會議定於{source_fact}",
            rendered_assertion_zh_hant_hk=f"會議改喺{rendered_fact}",
            localised_factual_expressions=((source_fact, rendered_fact),),
        )


@pytest.mark.parametrize(
    ("source_claim", "rendered_assertion"),
    (
        ("資助額增至一億元", "新安排提供二億元資助"),
        (
            "會議定於二零二六年十二月三十一日舉行",
            "活動改喺二零二七年十二月三十一日開始",
        ),
        ("計劃進入第三階段", "相關工作展開第四階段"),
        ("項目涉及三間學校", "工作涵蓋四間學校"),
        ("申請分三批處理", "安排改為四批處理"),
        ("活動改於星期三舉行", "會議定喺星期四進行"),
        ("支持率為百分之三", "最新支持率係百分之四"),
        ("氣溫升至三十度", "預測錄得四十度高溫"),
        ("三歲兒童可參加", "四歲小童符合資格"),
        ("提供三張門票", "另有四張入場券"),
        ("涉及三條路線", "涵蓋四條巴士線"),
        ("面積為三公頃", "面積達三英畝"),
        ("東區有三間學校，西區有四間學校", "東區有四間學校，西區有三間學校"),
        ("有三部巴士", "涉及三條道路"),
        ("計劃進入第三階段", "相關工作進入第Ⅳ期"),
        ("計劃進入第三階段", "相關工作進入第肆期"),
        ("面積為3公頃", "面積達3英畝"),
        ("服務涉及3間學校", "服務涉及3間醫院"),
        ("用水量係3公升", "用水量係3毫升"),
        ("溫度係3攝氏度", "溫度係3華氏度"),
        ("速度係3海里", "速度係3英里"),
        ("數量係3打", "數量係3箱"),
        ("用水量係3 公升", "用水量係3 毫升"),
        ("計劃進入第③階段", "相關工作進入第④階段"),
        ("計劃進入第㉑階段", "相關工作進入第㉒階段"),
        ("溫度係+3度", "溫度係−3度"),
        ("有3宗，涉及學校", "有3宗，涉及醫院"),
        ("資助額係£3", "資助額係$3"),
        ("資助額係€3", "資助額係¥3"),
    ),
)
def test_changed_chinese_figure_or_date_fails_fidelity(
    source_claim: str, rendered_assertion: str
) -> None:
    _candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    changed_substantive = replace(
        substantive,
        claim=source_claim,
        supporting_excerpt=source_claim,
        rendered_assertion_zh_hant_hk=rendered_assertion,
    )
    guarded_package = replace(
        package,
        passages=(source_claim,),
        substantive_new_information=(source_claim,),
        governed_claims=(headline, changed_substantive),
    )
    copy = WriterCopy(
        title=f"【未出版】{headline.rendered_assertion_zh_hant_hk}",
        body=f"本報根據已核實證據報道：{rendered_assertion}",
        writer_id="changed-chinese-number-writer",
        evidence_package_digest=guarded_package.digest,
        evidence_links=tuple(
            WriterEvidenceLink(item.claim_id, item.rendered_assertion_zh_hant_hk)
            for item in guarded_package.governed_claims
        ),
    )
    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, guarded_package)
        if item.result == "FAIL"
    }
    assert "UNSUPPORTED_NUMBER_OR_DATE" in failed


@pytest.mark.parametrize(
    ("source_claim", "rendered_assertion"),
    (
        ("申請本月截止", "申請下月截止"),
        ("活動今早開始", "活動今晚開始"),
        ("限期係後日", "限期係月底"),
        ("申請即日截止", "申請翌日截止"),
        ("申請本季截止", "申請下季截止"),
        ("The event is next month", "活動喺今年底舉行"),
    ),
)
def test_changed_relative_time_fails_fidelity(
    source_claim: str, rendered_assertion: str
) -> None:
    _candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    changed_substantive = replace(
        substantive,
        claim=source_claim,
        supporting_excerpt=source_claim,
        rendered_assertion_zh_hant_hk=rendered_assertion,
    )
    guarded_package = replace(
        package,
        passages=(source_claim,),
        substantive_new_information=(source_claim,),
        governed_claims=(headline, changed_substantive),
    )
    copy = WriterCopy(
        title=f"【未出版】{headline.rendered_assertion_zh_hant_hk}",
        body=f"本報根據已核實證據報道：{rendered_assertion}",
        writer_id="changed-relative-time-writer",
        evidence_package_digest=guarded_package.digest,
        evidence_links=tuple(
            WriterEvidenceLink(item.claim_id, item.rendered_assertion_zh_hant_hk)
            for item in guarded_package.governed_claims
        ),
    )

    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, guarded_package)
        if item.result == "FAIL"
    }

    assert "UNSUPPORTED_NUMBER_OR_DATE" in failed


def test_approved_quotation_must_retain_exact_attribution() -> None:
    _candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    attributed_claim = "署方說會繼續跟進"
    changed_substantive = replace(
        substantive,
        claim=attributed_claim,
        supporting_excerpt=attributed_claim,
        attribution="署方",
        quotations=("會繼續跟進",),
        rendered_assertion_zh_hant_hk="「會繼續跟進」",
    )
    guarded_package = replace(
        package,
        passages=(f"{headline.claim}\n{attributed_claim}",),
        substantive_new_information=(attributed_claim,),
        governed_claims=(headline, changed_substantive),
    )
    copy = WriterCopy(
        title=f"【未出版】{headline.rendered_assertion_zh_hant_hk}",
        body=f"本報根據已核實證據報道：{changed_substantive.rendered_assertion_zh_hant_hk}",
        writer_id="unattributed-quote-writer",
        evidence_package_digest=guarded_package.digest,
        evidence_links=tuple(
            WriterEvidenceLink(item.claim_id, item.rendered_assertion_zh_hant_hk)
            for item in guarded_package.governed_claims
        ),
    )
    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, guarded_package)
        if item.result == "FAIL"
    }
    assert "UNSUPPORTED_OR_UNATTRIBUTED_QUOTATION" in failed
    assert "REQUIRED_ATTRIBUTION_MISSING" in failed


def test_long_passage_cannot_hide_interrupted_sentence_copy() -> None:
    _candidate, package = _candidate_package()
    headline, substantive = package.governed_claims
    copied_sentence = (
        "The retained authority announced a distinctive material deadline."
    )
    long_source = (
        "Background context contains many unrelated words and details. "
        f"{copied_sentence} "
        "Further context continues with many unrelated words and details."
    )
    interrupted_copy = "的".join(
        copied_sentence[index : index + 11]
        for index in range(0, len(copied_sentence), 11)
    )
    guarded_package = replace(package, passages=(long_source,))
    copy = WriterCopy(
        title=f"【未出版】{headline.rendered_assertion_zh_hant_hk}",
        body=(
            "本報根據已核實證據報道："
            f"{substantive.rendered_assertion_zh_hant_hk}\n{interrupted_copy}"
        ),
        writer_id="long-passage-copying-writer",
        evidence_package_digest=guarded_package.digest,
        evidence_links=tuple(
            WriterEvidenceLink(item.claim_id, item.rendered_assertion_zh_hant_hk)
            for item in guarded_package.governed_claims
        ),
    )
    failed = {
        item.reason_code
        for item in validate_writer_copy(copy, guarded_package)
        if item.result == "FAIL"
    }
    assert "VERBATIM_SOURCE_EXPRESSION" in failed


def test_default_package_builder_can_admit_explicit_governed_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving = _proving(tmp_path)
    body = json.dumps(
        {
            "title": "Official deadline changed",
            "description": "The deadline is now 30 September.",
            "url": "https://example.test/governed-deadline",
        }
    ).encode()
    item = parse_observation(
        source_id="UK-02",
        url="https://www.gov.uk/api/content/governed-deadline",
        body=body,
    )[0]
    governed_candidate = form_candidates(
        (
            GroupedObservation(
                "UK-02",
                digest_bytes(body),
                item,
                "2026-08-20T00:00:00.000000Z",
            ),
        )
    )[0]
    base_package = package_for(governed_candidate)
    claim_ids = ("governed-headline", "governed-development")
    evidence = {
        "schema_version": "newsroom.governed-input.v10",
        "candidate_id": governed_candidate.candidate_id,
        "hypothesis_id": governed_candidate.hypothesis_id,
        "base_package_digest": base_package.digest,
        "governed_claims": [
            {
                "claim_id": "governed-headline",
                "claim": "Official deadline changed",
                "passage_index": 0,
                "supporting_excerpt": "Official deadline changed",
                "source_ids": ["UK-02"],
                "source_record_ids": ["source-record:UK-02"],
                "source_authority_decision_ids": ["source-authority:UK-02:headline"],
                "rights_decision_ids": ["rights:UK-02"],
                "dependency_evidence_ids": ["dependency:UK-02"],
                "evidential_origin_ids": ["official-release-1"],
                "authority_class": "RESPONSIBLE_PRIMARY",
                "authority_scope": "Responsible body for its own deadline.",
                "status": "CONFIRMED_FACT",
                "attribution": "UK-02",
                "rendered_assertion_zh_hant_hk": "官方公布咗最新限期安排",
                "claim_role": "HEADLINE",
                "semantic_relation_evidence_id": "semantic:governed-headline",
                "localised_factual_expressions": [],
                "named_entity_evidence": [],
                "named_entities": [],
                "rendered_named_entities": [],
                "quotations": [],
                "certainty": "CONFIRMED",
                "originality_basis": "FACTUAL_REWRITE_REQUIRED",
                "originality_policy_version": "newsroom.cont-originality.v3",
                "admitted_use": "PUBLICATION_EVIDENCE",
                "policy_version": "newsroom.governed-claim.v7",
            },
            {
                "claim_id": "governed-development",
                "claim": "The deadline is now 30 September.",
                "passage_index": 0,
                "supporting_excerpt": "The deadline is now 30 September.",
                "source_ids": ["UK-02"],
                "source_record_ids": ["source-record:UK-02"],
                "source_authority_decision_ids": ["source-authority:UK-02:substantive"],
                "rights_decision_ids": ["rights:UK-02"],
                "dependency_evidence_ids": ["dependency:UK-02"],
                "evidential_origin_ids": ["official-release-1"],
                "authority_class": "RESPONSIBLE_PRIMARY",
                "authority_scope": "Responsible body for its own deadline.",
                "status": "CONFIRMED_FACT",
                "attribution": "UK-02",
                "rendered_assertion_zh_hant_hk": "新限期定喺九月三十日",
                "claim_role": "SUBSTANTIVE",
                "semantic_relation_evidence_id": "semantic:governed-development",
                "localised_factual_expressions": [["30 September", "九月三十日"]],
                "named_entity_evidence": [],
                "named_entities": [],
                "rendered_named_entities": [],
                "quotations": [],
                "certainty": "CONFIRMED",
                "originality_basis": "FACTUAL_REWRITE_REQUIRED",
                "originality_policy_version": "newsroom.cont-originality.v3",
                "admitted_use": "PUBLICATION_EVIDENCE",
                "policy_version": "newsroom.governed-claim.v7",
            },
        ],
        "substantive_new_information": [
            "Official deadline changed",
            "The deadline is now 30 September.",
        ],
        "qualification_evidence": [
            {
                "test": "OFFICIAL_ACTION_OR_DEADLINE",
                "governed_claim_id": "governed-headline",
                "qualification_record_id": "qualification:governed-headline",
                "test_evidence": [
                    ["action_class", "OFFICIAL_DEADLINE"],
                    ["event_polarity", "AFFIRMED"],
                    ["action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"],
                    ["material_relation_span", "Official deadline changed"],
                    ["reader_action", "Official deadline changed"],
                ],
                "policy_version": "newsroom.evid-012.v7",
            },
            {
                "test": "OFFICIAL_ACTION_OR_DEADLINE",
                "governed_claim_id": "governed-development",
                "qualification_record_id": "qualification:governed-development",
                "test_evidence": [
                    ["action_class", "OFFICIAL_DEADLINE"],
                    ["event_polarity", "AFFIRMED"],
                    ["action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"],
                    [
                        "material_relation_span",
                        "The deadline is now 30 September.",
                    ],
                    ["reader_action", "The deadline is now 30 September."],
                ],
                "policy_version": "newsroom.evid-012.v7",
            },
        ],
        "selection_rationale": "Readers need the retained official deadline.",
        "geography": ["UK"],
        "categories": ["Politics and law"],
        "evidence_gate_results": [
            ["CLAIM_TRACEABILITY", "PASS"],
            ["EVIDENCE_SUFFICIENCY", "PASS"],
            ["SOURCE_AUTHORITY", "PASS"],
        ],
        "evidence_gate_evidence": [
            {
                "gate": gate,
                "result": "PASS",
                "governed_claim_ids": list(claim_ids),
                "policy_version": "newsroom.evidence-gates.v2",
            }
            for gate in (
                "CLAIM_TRACEABILITY",
                "EVIDENCE_SUFFICIENCY",
                "SOURCE_AUTHORITY",
            )
        ],
        "freshness_result": "PASS",
        "integrity_result": "PASS",
        "explicit_exclusions": [],
    }
    connection = sqlite3.connect(proving)
    _store_body(connection, digest_bytes(body), body)
    connection.execute(
        "UPDATE proving_observations SET body_digest=?, item_count=1 "
        "WHERE source_id='UK-02'",
        (digest_bytes(body),),
    )
    retain_observation_revision_first_seen(
        connection,
        source_id="UK-02",
        url="https://www.gov.uk/api/content/governed-deadline",
        body=body,
        observed_at="2026-08-20T00:00:00.000000Z",
    )
    raw_evidence = canonical_json_bytes(evidence).decode()
    package_json_digest = digest_bytes(raw_evidence.encode())
    records = (
        {
            "record_id": "source-record:UK-02",
            "record_type": "SOURCE_RECORD",
            "candidate_id": governed_candidate.candidate_id,
            "base_package_digest": base_package.digest,
            "status": "CURRENT",
            "source_id": "UK-02",
            "canonical_url": "https://example.test/governed-deadline",
            "publisher": "Fixture authority",
            "responsible_body": "Fixture authority",
            "source_type": "PRIMARY_OFFICIAL",
            "authority_class": "RESPONSIBLE_PRIMARY",
            "publication_time": "2026-08-20T00:00:00Z",
            "retrieval_time": "2026-08-20T00:00:00Z",
            "geography": "UK",
            "language": "en",
            "extraction_status": "COMPLETE",
            "rights_decision_id": "rights:UK-02",
            "originating_report_id": "official-release-1",
            "originating_artefact_digest": digest_bytes(body),
            "dependency_evidence_ids": ["dependency:UK-02"],
        },
        {
            "record_id": "source-authority:UK-02:headline",
            "record_type": "SOURCE_AUTHORITY_DECISION",
            "candidate_id": governed_candidate.candidate_id,
            "base_package_digest": base_package.digest,
            "status": "CURRENT",
            "source_id": "UK-02",
            "decision": "ADMITTED",
            "authority_class": "RESPONSIBLE_PRIMARY",
            "authority_scope": "Responsible body for its own deadline.",
            "governed_claim_id": "governed-headline",
            "claim_digest": digest_bytes(b"Official deadline changed"),
        },
        {
            "record_id": "source-authority:UK-02:substantive",
            "record_type": "SOURCE_AUTHORITY_DECISION",
            "candidate_id": governed_candidate.candidate_id,
            "base_package_digest": base_package.digest,
            "status": "CURRENT",
            "source_id": "UK-02",
            "decision": "ADMITTED",
            "authority_class": "RESPONSIBLE_PRIMARY",
            "authority_scope": "Responsible body for its own deadline.",
            "governed_claim_id": "governed-development",
            "claim_digest": digest_bytes(b"The deadline is now 30 September."),
        },
        {
            "record_id": "rights:UK-02",
            "record_type": "RIGHTS_DECISION",
            "candidate_id": governed_candidate.candidate_id,
            "base_package_digest": base_package.digest,
            "status": "CURRENT",
            "source_id": "UK-02",
            "decision": "PERMITTED",
            "permitted_use": "PUBLICATION_EVIDENCE",
        },
        {
            "record_id": "dependency:UK-02",
            "record_type": "DEPENDENCY_EVIDENCE",
            "candidate_id": governed_candidate.candidate_id,
            "base_package_digest": base_package.digest,
            "status": "CURRENT",
            "source_id": "UK-02",
            "dependency_status": "RESOLVED",
            "evidential_origin_id": "official-release-1",
            "originating_report_id": "official-release-1",
        },
        {
            "record_id": "semantic:governed-headline",
            "record_type": "SEMANTIC_RELATION_EVIDENCE",
            "candidate_id": governed_candidate.candidate_id,
            "base_package_digest": base_package.digest,
            "status": "CURRENT",
            "governed_claim_id": "governed-headline",
            "source_modality": "ASSERTED",
            "rendered_modality": "ASSERTED",
            "source_polarity": "AFFIRMED",
            "rendered_polarity": "AFFIRMED",
            "relation": "SEMANTICALLY_EQUIVALENT",
            "claim_digest": digest_bytes(b"Official deadline changed"),
            "rendered_assertion_digest": digest_bytes(
                "官方公布咗最新限期安排".encode()
            ),
        },
        {
            "record_id": "semantic:governed-development",
            "record_type": "SEMANTIC_RELATION_EVIDENCE",
            "candidate_id": governed_candidate.candidate_id,
            "base_package_digest": base_package.digest,
            "status": "CURRENT",
            "governed_claim_id": "governed-development",
            "source_modality": "ASSERTED",
            "rendered_modality": "ASSERTED",
            "source_polarity": "AFFIRMED",
            "rendered_polarity": "AFFIRMED",
            "relation": "SEMANTICALLY_EQUIVALENT",
            "claim_digest": digest_bytes(b"The deadline is now 30 September."),
            "rendered_assertion_digest": digest_bytes("新限期定喺九月三十日".encode()),
        },
        {
            "record_id": "qualification:governed-headline",
            "record_type": "QUALIFICATION_EVIDENCE",
            "candidate_id": governed_candidate.candidate_id,
            "base_package_digest": base_package.digest,
            "status": "CURRENT",
            "governed_claim_id": "governed-headline",
            "test": "OFFICIAL_ACTION_OR_DEADLINE",
            "test_evidence": [
                ["action_class", "OFFICIAL_DEADLINE"],
                ["event_polarity", "AFFIRMED"],
                ["action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"],
                ["material_relation_span", "Official deadline changed"],
                ["reader_action", "Official deadline changed"],
            ],
            "policy_version": "newsroom.evid-012.v7",
            "evidence_span_digest": digest_bytes(b"Official deadline changed"),
            "source_record_ids": ["source-record:UK-02"],
        },
        {
            "record_id": "qualification:governed-development",
            "record_type": "QUALIFICATION_EVIDENCE",
            "candidate_id": governed_candidate.candidate_id,
            "base_package_digest": base_package.digest,
            "status": "CURRENT",
            "governed_claim_id": "governed-development",
            "test": "OFFICIAL_ACTION_OR_DEADLINE",
            "test_evidence": [
                ["action_class", "OFFICIAL_DEADLINE"],
                ["event_polarity", "AFFIRMED"],
                ["action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"],
                [
                    "material_relation_span",
                    "The deadline is now 30 September.",
                ],
                ["reader_action", "The deadline is now 30 September."],
            ],
            "policy_version": "newsroom.evid-012.v7",
            "evidence_span_digest": digest_bytes(b"The deadline is now 30 September."),
            "source_record_ids": ["source-record:UK-02"],
        },
    )
    retained_records = []
    connection.execute(
        "CREATE TABLE proving_write_evidence_records("
        "record_id TEXT PRIMARY KEY, record_type TEXT NOT NULL, "
        "record_json TEXT NOT NULL, record_digest TEXT NOT NULL)"
    )
    for record in records:
        record_raw = canonical_json_bytes(record).decode()
        record_digest = digest_bytes(record_raw.encode())
        retained_records.append((record["record_id"], record_digest))
        connection.execute(
            "INSERT INTO proving_write_evidence_records VALUES(?,?,?,?)",
            (record["record_id"], record["record_type"], record_raw, record_digest),
        )
    evidence_record_set_digest = digest_bytes(
        canonical_json_bytes(
            {"records": [list(item) for item in sorted(retained_records)]}
        )
    )
    approval = {
        "base_package_digest": base_package.digest,
        "candidate_id": governed_candidate.candidate_id,
        "controller_principal": "HERMES_EVIDENCE_CONTROLLER",
        "decision": "APPROVED",
        "evidence_record_set_digest": evidence_record_set_digest,
        "hypothesis_id": governed_candidate.hypothesis_id,
        "package_json_digest": package_json_digest,
        "policy_version": "newsroom.evidence-approval.v8",
    }
    approval_raw = canonical_json_bytes(approval).decode()
    approval_key = b"fixture-evidence-controller-key-32-bytes"
    monkeypatch.setenv("NEWSROOM_EVIDENCE_APPROVAL_KEY", approval_key.decode())
    signature = hmac.new(
        approval_key,
        approval_raw.encode(),
        hashlib.sha256,
    ).hexdigest()
    connection.execute(
        "CREATE TABLE proving_write_evidence_packages("
        "candidate_id TEXT PRIMARY KEY, package_json TEXT NOT NULL, "
        "package_json_digest TEXT NOT NULL, approval_status TEXT NOT NULL, "
        "approval_record_json TEXT NOT NULL, approval_signature TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO proving_write_evidence_packages VALUES(?,?,?,?,?,?)",
        (
            governed_candidate.candidate_id,
            raw_evidence,
            package_json_digest,
            "APPROVED",
            approval_raw,
            signature,
        ),
    )
    connection.commit()
    connection.close()

    report = run_cycle(
        proving_store=str(proving),
        unpublished_store=str(tmp_path / "unpublished_store.sqlite3"),
        writer=FixtureWriter(),
        max_writes=5,
        clock=_CLOCK,
    )

    assert report.write_ready == 1
    assert report.provider_dispatches == 1
    assert report.minted == 1

    connection = sqlite3.connect(proving)
    connection.execute(
        "DELETE FROM proving_write_evidence_records WHERE record_id='dependency:UK-02'"
    )
    connection.commit()
    connection.close()
    dangling_report = run_cycle(
        proving_store=str(proving),
        unpublished_store=str(tmp_path / "dangling-unpublished.sqlite3"),
        writer=CountingWriter(),
        max_writes=5,
        clock=_CLOCK,
    )
    assert dangling_report.write_ready == 0
    assert dangling_report.provider_dispatches == 0
    dependency_record = next(
        record for record in records if record["record_id"] == "dependency:UK-02"
    )
    dependency_raw = canonical_json_bytes(dependency_record).decode()
    connection = sqlite3.connect(proving)
    connection.execute(
        "INSERT INTO proving_write_evidence_records VALUES(?,?,?,?)",
        (
            dependency_record["record_id"],
            dependency_record["record_type"],
            dependency_raw,
            digest_bytes(dependency_raw.encode()),
        ),
    )
    connection.commit()
    connection.close()

    drifted_body = json.dumps(
        {
            "title": "Official deadline changed",
            "description": "The deadline changed again after approval.",
            "url": "https://example.test/governed-deadline",
        }
    ).encode()
    connection = sqlite3.connect(proving)
    _store_body(connection, digest_bytes(drifted_body), drifted_body)
    connection.execute(
        "UPDATE proving_observations SET body_digest=? WHERE source_id='UK-02'",
        (digest_bytes(drifted_body),),
    )
    retain_observation_revision_first_seen(
        connection,
        source_id="UK-02",
        url="https://www.gov.uk/api/content/governed-deadline",
        body=drifted_body,
        observed_at="2026-08-20T00:01:00.000000Z",
    )
    connection.commit()
    connection.close()

    drifted_report = run_cycle(
        proving_store=str(proving),
        unpublished_store=str(tmp_path / "drifted-unpublished.sqlite3"),
        writer=CountingWriter(),
        max_writes=5,
        clock=_CLOCK,
    )
    assert drifted_report.write_ready == 0
    assert drifted_report.provider_dispatches == 0


def test_insert_race_does_not_refund_reserved_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    unpublished = tmp_path / "unpublished_store.sqlite3"
    monkeypatch.setattr(
        "newsroom.control_plane.cycle.insert_payload", lambda *_args: False
    )

    report = run_cycle(
        proving_store=str(_proving(tmp_path)),
        unpublished_store=str(unpublished),
        writer=FixtureWriter(),
        evidence_package_builder=_qualified_builder(frozenset({"HK-01"})),
        clock=_CLOCK,
    )

    assert report.candidate_attempts == 1
    assert report.provider_dispatches == 1
    assert report.duplicate == 1
    assert report.draft_hold == 1
    assert report.minted == 0
    connection = sqlite3.connect(unpublished)
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM unpublished_write_candidate_attempts"
        ).fetchone()[0]
        == 1
    )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM unpublished_writer_provider_attempts"
        ).fetchone()[0]
        == 1
    )
    assert connection.execute(
        "SELECT outcome, reason_codes_json FROM unpublished_draft_outcomes"
    ).fetchone() == ("HOLD", '["DUPLICATE_INSERT_RACE"]')
    connection.close()


def test_final_ledger_counters_equal_retained_attempts_outcomes_and_inserts(
    tmp_path: Path,
) -> None:
    unpublished = tmp_path / "unpublished_store.sqlite3"
    fence_calls = 0

    def writer_dispatch_fence() -> None:
        nonlocal fence_calls
        fence_calls += 1

    report = run_cycle(
        proving_store=str(_proving(tmp_path)),
        unpublished_store=str(unpublished),
        writer=RecordingFixtureWriter(),
        evidence_package_builder=_qualified_builder(frozenset({"HK-01", "UK-01"})),
        clock=_CLOCK,
        writer_dispatch_fence=writer_dispatch_fence,
    )

    connection = sqlite3.connect(unpublished)
    close = json.loads(
        connection.execute(
            "SELECT payload_json FROM ledger WHERE kind='PRIVATE_CYCLE_CLOSE'"
        ).fetchone()[0]
    )
    start = json.loads(
        connection.execute(
            "SELECT payload_json FROM ledger WHERE kind='PRIVATE_CYCLE_START'"
        ).fetchone()[0]
    )
    attempt_cycle_ids = {
        str(row[0])
        for row in connection.execute(
            "SELECT cycle_execution_id FROM unpublished_write_candidate_attempts"
        )
    }
    retained = {
        "candidate_attempts": connection.execute(
            "SELECT COUNT(*) FROM unpublished_write_candidate_attempts"
        ).fetchone()[0],
        "provider_dispatches": connection.execute(
            "SELECT COUNT(*) FROM unpublished_writer_provider_attempts"
        ).fetchone()[0],
        "accepted": connection.execute(
            "SELECT COUNT(*) FROM unpublished_draft_outcomes WHERE outcome='ACCEPTED'"
        ).fetchone()[0],
        "payloads": connection.execute(
            "SELECT COUNT(*) FROM unpublished_surface_payloads"
        ).fetchone()[0],
    }
    connection.close()
    assert (
        close["candidate_attempts"]
        == retained["candidate_attempts"]
        == report.candidate_attempts
    )
    assert (
        close["provider_dispatches"]
        == retained["provider_dispatches"]
        == report.provider_dispatches
    )
    assert (
        close["draft_outcomes"]["ACCEPTED"]
        == retained["accepted"]
        == report.draft_accepted
    )
    assert close["accepted_payload_count"] == retained["payloads"] == report.minted
    assert str(UUID(report.cycle_id)) == report.cycle_id
    assert start["cycle_id"] == close["cycle_id"] == report.cycle_id
    assert attempt_cycle_ids == {report.cycle_id}
    assert fence_calls == report.provider_dispatches


def test_owner_emergency_stop_is_rechecked_at_writer_dispatch(
    tmp_path: Path,
) -> None:
    proving = _proving(tmp_path)
    unpublished = tmp_path / "owner-stop-before-dispatch.sqlite3"
    usage = ModelUsageService(str(unpublished))
    fence_calls = 0

    def activate_owner_stop() -> None:
        nonlocal fence_calls
        fence_calls += 1
        connection = sqlite3.connect(proving)
        connection.execute(
            "UPDATE proving_gates SET status='FAIL' "
            "WHERE run_id='run-1' "
            "AND gate_id='NO_ACTIVE_HUMAN_EMERGENCY_STOP'"
        )
        connection.commit()
        connection.close()

    with pytest.raises(VetoError, match="owner emergency stop"):
        run_cycle(
            proving_store=str(proving),
            unpublished_store=str(unpublished),
            writer=RecordingFixtureWriter(),
            evidence_package_builder=_qualified_builder(frozenset({"HK-01"})),
            clock=_CLOCK,
            writer_dispatch_fence=activate_owner_stop,
            model_usage=usage,
        )

    connection = sqlite3.connect(unpublished)
    assert fence_calls == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_writer_provider_attempts"
    ).fetchone() == (0,)
    assert connection.execute(
        "SELECT outcome,reason_codes_json FROM unpublished_draft_outcomes"
    ).fetchone() == ("HOLD", '["OWNER_EMERGENCY_STOP"]')
    assert connection.execute(
        "SELECT outcome,json_extract(record_json,'$.stable_reason_codes') "
        "FROM model_work_outcomes"
    ).fetchone() == ("HOLD", '["OWNER_EMERGENCY_STOP"]')
    assert connection.execute(
        "SELECT COUNT(*) FROM model_invocation_allocations"
    ).fetchone() == (0,)
    connection.close()
    query = usage.query(
        start=datetime(2026, 8, 19, tzinfo=UTC),
        end=datetime(2026, 8, 21, tzinfo=UTC),
    )
    assert query["leaves"] == []
    assert len(query["envelopes"]) == 1
    assert query["envelopes"][0]["outcome"] == "HOLD"  # type: ignore[index]
    assert query["envelopes"][0]["stable_reason_codes"] == [  # type: ignore[index]
        "OWNER_EMERGENCY_STOP"
    ]


def test_owner_emergency_stop_after_reservation_vetoes_writer_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from newsroom.control_plane import cycle

    proving = _proving(tmp_path)
    unpublished = tmp_path / "owner-stop-after-reservation.sqlite3"
    usage = ModelUsageService(str(unpublished))
    _register_cont_usage_policy(
        usage,
        workload_class=WorkloadClass.CONT_WRITER_PRIMARY,
        provider=CONT_PRIMARY_PROVIDER,
        route=CONT_PRIMARY_ROUTE,
        model=CONT_PRIMARY_MODEL,
        reasoning=CONT_PRIMARY_REASONING,
    )
    provider_calls = 0

    def primary(_prompt: str) -> str:
        nonlocal provider_calls
        provider_calls += 1
        return json.dumps({"title": "must not dispatch", "body": "must not dispatch"})

    writer = CliChainWriter(primary=primary, fallback=primary)
    reserve = cycle.reserve_writer_provider_attempt

    def reserve_then_stop(*args: object, **kwargs: object) -> str:
        provider_attempt_id = reserve(*args, **kwargs)  # type: ignore[arg-type]
        connection = sqlite3.connect(proving)
        connection.execute(
            "UPDATE proving_gates SET status='FAIL' "
            "WHERE run_id='run-1' "
            "AND gate_id='NO_ACTIVE_HUMAN_EMERGENCY_STOP'"
        )
        connection.commit()
        connection.close()
        return provider_attempt_id

    monkeypatch.setattr(cycle, "reserve_writer_provider_attempt", reserve_then_stop)

    with pytest.raises(VetoError, match="owner emergency stop"):
        run_cycle(
            proving_store=str(proving),
            unpublished_store=str(unpublished),
            writer=writer,
            evidence_package_builder=_qualified_builder(frozenset({"HK-01"})),
            clock=_CLOCK,
            model_usage=usage,
        )

    assert provider_calls == 0
    connection = sqlite3.connect(unpublished)
    assert connection.execute(
        "SELECT status,reason_code FROM unpublished_writer_provider_attempts"
    ).fetchone() == ("FAILED", "OWNER_EMERGENCY_STOP")
    assert connection.execute(
        "SELECT outcome,reason_codes_json FROM unpublished_draft_outcomes"
    ).fetchone() == ("HOLD", '["OWNER_EMERGENCY_STOP"]')
    assert connection.execute(
        "SELECT outcome,json_extract(record_json,'$.stable_reason_codes') "
        "FROM model_work_outcomes"
    ).fetchone() == ("HOLD", '["OWNER_EMERGENCY_STOP"]')
    assert connection.execute(
        "SELECT outcome,failure_class,usage_status FROM model_invocation_terminals"
    ).fetchone() == (
        "VETOED_BEFORE_PROVIDER_DISPATCH",
        "OWNER_EMERGENCY_STOP",
        "REPORTED",
    )
    connection.close()


def test_hold_and_reject_candidates_never_reach_injected_writer(
    tmp_path: Path,
) -> None:
    def builder(candidate: StoryCandidateRecord) -> EvidencePackage:
        qualified = _qualified_builder(frozenset({candidate.items[0].source_id}))(
            candidate
        )
        if candidate.items[0].source_id == "UK-01":
            return package_for(candidate)
        if candidate.items[0].source_id == "UK-02":
            return replace(
                qualified,
                substantive_new_information=(),
            )
        return qualified

    writer = RecordingFixtureWriter()
    report = run_cycle(
        proving_store=str(_proving(tmp_path)),
        unpublished_store=str(tmp_path / "unpublished_store.sqlite3"),
        writer=writer,
        evidence_package_builder=builder,
        clock=_CLOCK,
    )

    assert report.write_ready == 1
    assert report.admission_hold == 1
    assert report.admission_reject == 1
    assert len(writer.calls) == 1


def test_selection_uses_retained_quality_before_candidate_id() -> None:
    candidate, _package = _candidate_package()
    low_candidate = replace(candidate, candidate_id="a-candidate")
    low = _admit_package(
        low_candidate,
        package_for(low_candidate),
        authority_class=ClaimAuthorityClass.INDEPENDENT_RELIABLE,
        origin_ids=("wire-origin", "independent-origin"),
    )
    wire_candidate = replace(candidate, candidate_id="z-wire-copies")
    wire = replace(
        low,
        candidate_id=wire_candidate.candidate_id,
        source_ids=("HK-01", "WIRE-COPY-1", "WIRE-COPY-2"),
    )
    policy = DeterministicWriteAdmission()
    low_decision = policy.decide(low_candidate, low, decided_at="2026-08-20T00:00:00Z")
    wire_decision = policy.decide(
        wire_candidate, wire, decided_at="2026-08-20T00:00:00Z"
    )

    selected = select_write_ready(
        (
            (low_candidate, low, low_decision),
            (wire_candidate, wire, wire_decision),
        ),
        limit=2,
        selected_at="2026-08-20T00:00:00Z",
    )

    assert [item[0].candidate_id for item in selected] == [
        "a-candidate",
        "z-wire-copies",
    ]
    assert selected[0][3].ordering_evidence[2] == "independent_evidential_origins=2"

    primary_candidate = replace(candidate, candidate_id="z-primary")
    primary = _admit_package(primary_candidate, package_for(primary_candidate))
    primary_decision = policy.decide(
        primary_candidate, primary, decided_at="2026-08-20T00:00:00Z"
    )
    selected = select_write_ready(
        (
            (low_candidate, low, low_decision),
            (primary_candidate, primary, primary_decision),
        ),
        limit=2,
        selected_at="2026-08-20T00:00:00Z",
    )
    assert selected[0][0].candidate_id == "z-primary"
    assert selected[0][3].ordering_evidence[1] == "claim_authority_score=4"


def test_connect_upgrades_legacy_ledger_for_retained_counter_payloads(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-unpublished.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE ledger(seq INTEGER PRIMARY KEY, at TEXT NOT NULL, kind TEXT NOT NULL, "
        "payload_digest TEXT NOT NULL, prev_digest TEXT NOT NULL, digest TEXT NOT NULL)"
    )
    legacy_digest = digest_bytes(b"legacy payload bytes are unavailable")
    connection.execute(
        "INSERT INTO ledger VALUES(1,?,?,?,?,?)",
        (
            "2026-08-20T00:00:00Z",
            "LEGACY_EVENT",
            legacy_digest,
            LEDGER_GENESIS,
            "sha256:" + ("a" * 64),
        ),
    )
    connection.commit()
    connection.close()

    upgraded = connect(str(path))
    migrated_schema = upgraded.execute("PRAGMA table_info(ledger)").fetchall()
    migrated = upgraded.execute(
        "SELECT payload_digest, payload_json FROM ledger WHERE seq=1"
    ).fetchone()
    upgraded.close()

    assert migrated == (legacy_digest, None)
    fresh = connect(str(tmp_path / "fresh-unpublished.sqlite3"))
    fresh_schema = fresh.execute("PRAGMA table_info(ledger)").fetchall()
    fresh.close()
    assert migrated_schema == fresh_schema
    reopened = connect(str(path))
    assert reopened.execute("SELECT COUNT(*) FROM ledger").fetchone() == (1,)
    assert not reopened.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ledger_pre_v12'"
    ).fetchone()
    reopened.close()


def test_concurrent_ledger_appends_form_one_contiguous_chain(tmp_path: Path) -> None:
    path = tmp_path / "concurrent-unpublished.sqlite3"
    connection = connect(str(path))
    connection.commit()
    connection.close()

    def append(value: int) -> None:
        worker = connect(str(path))
        append_ledger(worker, "CONCURRENT_FIXTURE", {"value": value})
        time.sleep(0.05)
        worker.commit()
        worker.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in (pool.submit(append, 1), pool.submit(append, 2)):
            future.result()

    connection = connect(str(path))
    rows = connection.execute(
        "SELECT prev_digest, digest FROM ledger ORDER BY seq"
    ).fetchall()
    connection.close()
    assert len(rows) == 2
    assert rows[0][0] == LEDGER_GENESIS
    assert rows[1][0] == rows[0][1]


def test_write_attempt_relations_fail_closed_on_orphans_and_cross_binding(
    tmp_path: Path,
) -> None:
    connection = connect(str(tmp_path / "fk-unpublished.sqlite3"))
    with pytest.raises(sqlite3.IntegrityError):
        reserve_writer_provider_attempt(
            connection,
            candidate_attempt_id="missing",
            route="PRIMARY",
            ordinal=1,
        )
    candidate, package = _candidate_package()
    decision = DeterministicWriteAdmission().decide(
        candidate, package, decided_at="2026-08-20T00:00:00Z"
    )
    retain_write_admission_decision(connection, decision)
    with pytest.raises(sqlite3.IntegrityError):
        reserve_write_candidate_attempt(
            connection,
            cycle_execution_id="cycle-1",
            decision_id=decision.decision_id,
            candidate_id="another-candidate",
            evidence_package_digest=decision.evidence_package_digest,
            ordinal=1,
        )
    connection.close()


def test_conflicting_admission_for_same_package_is_not_silently_ignored(
    tmp_path: Path,
) -> None:
    connection = connect(str(tmp_path / "conflict-unpublished.sqlite3"))
    candidate, package = _candidate_package()
    decision = DeterministicWriteAdmission().decide(
        candidate, package, decided_at="2026-08-20T00:00:00Z"
    )
    retain_write_admission_decision(connection, decision)
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO unpublished_write_admission_decisions "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                "sha256:" + ("f" * 64),
                decision.candidate_id,
                decision.evidence_package_digest,
                decision.policy_version,
                "HOLD",
                '["CONFLICT"]',
                "{}",
                "2026-08-20T00:00:01Z",
                "2026-08-20T00:00:01Z",
                "sha256:" + ("0" * 64),
            ),
        )
    connection.close()


def test_write_selection_replay_requires_exact_canonical_record(tmp_path: Path) -> None:
    connection = connect(str(tmp_path / "selection-replay.sqlite3"))
    candidate, package = _candidate_package()
    decision = DeterministicWriteAdmission().decide(
        candidate, package, decided_at="2026-08-20T00:00:00Z"
    )
    retain_write_admission_decision(connection, decision)
    selection = select_write_ready(
        ((candidate, package, decision),),
        limit=1,
        selected_at="2026-08-20T00:00:00Z",
    )[0][3]
    retain_write_selection(connection, selection)
    retain_write_selection(
        connection,
        replace(selection, selected_at="2026-08-20T00:00:01Z"),
    )

    with pytest.raises(ValueError, match="selection identity"):
        WriteSelectionRecord(
            selection_id=selection.selection_id,
            decision_id=selection.decision_id,
            candidate_id=selection.candidate_id,
            evidence_package_digest=selection.evidence_package_digest,
            rank=selection.rank,
            quality_score=(0, *selection.quality_score[1:]),
            ordering_evidence=selection.ordering_evidence,
            policy_version=selection.policy_version,
            selected_at=selection.selected_at,
        )
    with pytest.raises(ValueError, match="unsupported write selection policy"):
        WriteSelectionRecord(
            selection_id=digest_bytes(
                canonical_json_bytes(
                    {
                        "decision_id": selection.decision_id,
                        "candidate_id": selection.candidate_id,
                        "evidence_package_digest": selection.evidence_package_digest,
                        "rank": selection.rank,
                        "quality_score": selection.quality_score,
                        "policy_version": "evil.policy",
                    }
                )
            ),
            decision_id=selection.decision_id,
            candidate_id=selection.candidate_id,
            evidence_package_digest=selection.evidence_package_digest,
            rank=selection.rank,
            quality_score=selection.quality_score,
            ordering_evidence=selection.ordering_evidence,
            policy_version="evil.policy",
            selected_at=selection.selected_at,
        )

    original_record = selection.as_record()
    poisoned_records = (
        {},
        {**original_record, "ordering_evidence": ["FORGED"]},
        {**original_record, "evil": "value"},
        {**original_record, "selected_at": "attacker-time"},
    )
    for poisoned in poisoned_records:
        connection.execute(
            "UPDATE unpublished_write_selections SET record_json=? "
            "WHERE selection_id=?",
            (canonical_json_bytes(poisoned).decode(), selection.selection_id),
        )
        with pytest.raises(sqlite3.IntegrityError, match="conflicting write-selection"):
            retain_write_selection(connection, selection)
        connection.execute(
            "UPDATE unpublished_write_selections SET record_json=? "
            "WHERE selection_id=?",
            (canonical_json_bytes(original_record).decode(), selection.selection_id),
        )
    time_poison = {**original_record, "selected_at": "attacker-time"}
    connection.execute(
        "UPDATE unpublished_write_selections SET record_json=?, selected_at=? "
        "WHERE selection_id=?",
        (
            canonical_json_bytes(time_poison).decode(),
            "attacker-time",
            selection.selection_id,
        ),
    )
    with pytest.raises(sqlite3.IntegrityError, match="conflicting write-selection"):
        retain_write_selection(connection, selection)
    connection.close()


def test_admission_replay_rejects_tampered_canonical_record(tmp_path: Path) -> None:
    connection = connect(str(tmp_path / "admission-replay.sqlite3"))
    candidate, package = _candidate_package()
    decision = DeterministicWriteAdmission().decide(
        candidate, package, decided_at="2026-08-20T00:00:00Z"
    )
    retain_write_admission_decision(connection, decision)
    connection.execute(
        "UPDATE unpublished_write_admission_decisions "
        "SET reason_codes_json='[\"TAMPERED\"]', record_json='{}' "
        "WHERE decision_id=?",
        (decision.decision_id,),
    )
    with pytest.raises(sqlite3.IntegrityError, match="conflicting write-admission"):
        retain_write_admission_decision(connection, decision)
    poisoned = {**decision.as_record(), "decided_at": "attacker-time"}
    connection.execute(
        "UPDATE unpublished_write_admission_decisions "
        "SET reason_codes_json=?, record_json=?, decided_at=? WHERE decision_id=?",
        (
            canonical_json_bytes(list(decision.stable_reason_codes)).decode(),
            canonical_json_bytes(poisoned).decode(),
            "attacker-time",
            decision.decision_id,
        ),
    )
    with pytest.raises(sqlite3.IntegrityError, match="conflicting write-admission"):
        retain_write_admission_decision(connection, decision)
    connection.close()


def test_draft_outcome_retains_and_enforces_attempt_identity(tmp_path: Path) -> None:
    connection = connect(str(tmp_path / "outcome-attempts.sqlite3"))
    candidate, package = _candidate_package()
    decision = DeterministicWriteAdmission().decide(
        candidate, package, decided_at="2026-08-20T00:00:00Z"
    )
    retain_write_admission_decision(connection, decision)
    candidate_attempt_id = reserve_write_candidate_attempt(
        connection,
        cycle_execution_id="cycle-1",
        decision_id=decision.decision_id,
        candidate_id=decision.candidate_id,
        evidence_package_digest=decision.evidence_package_digest,
        ordinal=1,
    )
    provider_attempt_id = reserve_writer_provider_attempt(
        connection,
        candidate_attempt_id=candidate_attempt_id,
        route="PRIMARY",
        ordinal=1,
    )
    outcome = DraftOutcomeRecord.create(
        write_admission_decision_id=decision.decision_id,
        candidate_id=decision.candidate_id,
        evidence_package_digest=decision.evidence_package_digest,
        provider_attempt_ids=(provider_attempt_id,),
        outcome="HOLD",
        validator_results=(),
        stable_reason_codes=("FIXTURE_HOLD",),
        payload_digest=None,
        recorded_at="2026-08-20T00:00:00Z",
        candidate_attempt_id=candidate_attempt_id,
    )
    retain_draft_outcome(connection, outcome)
    retained = json.loads(
        connection.execute(
            "SELECT record_json FROM unpublished_draft_outcomes WHERE outcome_id=?",
            (outcome.outcome_id,),
        ).fetchone()[0]
    )
    assert retained["candidate_attempt_id"] == candidate_attempt_id

    orphan = DraftOutcomeRecord.create(
        write_admission_decision_id=decision.decision_id,
        candidate_id=decision.candidate_id,
        evidence_package_digest=decision.evidence_package_digest,
        provider_attempt_ids=(),
        outcome="HOLD",
        validator_results=(),
        stable_reason_codes=("ORPHAN",),
        payload_digest=None,
        recorded_at="2026-08-20T00:00:01Z",
        candidate_attempt_id="missing-attempt",
    )
    with pytest.raises(sqlite3.IntegrityError, match="candidate attempt"):
        retain_draft_outcome(connection, orphan)

    second_attempt_id = reserve_write_candidate_attempt(
        connection,
        cycle_execution_id="cycle-2",
        decision_id=decision.decision_id,
        candidate_id=decision.candidate_id,
        evidence_package_digest=decision.evidence_package_digest,
        ordinal=1,
    )
    cross_provider_id = reserve_writer_provider_attempt(
        connection,
        candidate_attempt_id=second_attempt_id,
        route="PRIMARY",
        ordinal=1,
    )
    cross_bound = DraftOutcomeRecord.create(
        write_admission_decision_id=decision.decision_id,
        candidate_id=decision.candidate_id,
        evidence_package_digest=decision.evidence_package_digest,
        provider_attempt_ids=(cross_provider_id,),
        outcome="HOLD",
        validator_results=(),
        stable_reason_codes=("CROSS_BOUND",),
        payload_digest=None,
        recorded_at="2026-08-20T00:00:02Z",
        candidate_attempt_id=candidate_attempt_id,
    )
    with pytest.raises(sqlite3.IntegrityError, match="provider attempts"):
        retain_draft_outcome(connection, cross_bound)
    connection.close()


def test_admission_is_durable_when_later_package_construction_fails(
    tmp_path: Path,
) -> None:
    calls = 0

    def build(candidate: StoryCandidateRecord) -> EvidencePackage:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("later governed package failed")
        return package_for(candidate)

    unpublished = tmp_path / "durable-admission.sqlite3"
    with pytest.raises(RuntimeError, match="later governed package failed"):
        run_cycle(
            proving_store=str(_proving(tmp_path)),
            unpublished_store=str(unpublished),
            evidence_package_builder=build,
            writer=FixtureWriter(),
            clock=_CLOCK,
        )
    connection = sqlite3.connect(unpublished)
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_write_admission_decisions"
    ).fetchone() == (1,)
    connection.close()


def test_validated_payload_and_outcome_survive_later_coverage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unpublished = tmp_path / "durable-outcome.sqlite3"

    def fail_coverage(*_args: object, **_kwargs: object) -> dict[str, int]:
        raise RuntimeError("coverage closeout failed")

    monkeypatch.setattr("newsroom.control_plane.cycle.graphiti_coverage", fail_coverage)
    with pytest.raises(RuntimeError, match="coverage closeout failed"):
        run_cycle(
            proving_store=str(_proving(tmp_path)),
            unpublished_store=str(unpublished),
            evidence_package_builder=_qualified_builder(frozenset({"HK-01"})),
            writer=RecordingFixtureWriter(),
            clock=_CLOCK,
        )
    connection = sqlite3.connect(unpublished)
    assert connection.execute(
        "SELECT status FROM unpublished_writer_provider_attempts"
    ).fetchall() == [("COMPLETE",)]
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_surface_payloads"
    ).fetchone() == (1,)
    assert connection.execute(
        "SELECT outcome FROM unpublished_draft_outcomes"
    ).fetchall() == [("ACCEPTED",)]
    connection.close()


def _register_cont_usage_policy(
    usage: ModelUsageService,
    *,
    workload_class: WorkloadClass,
    provider: str,
    route: str,
    model: str,
    reasoning: str,
    max_prompt_bytes: int = 100_000,
) -> None:
    usage.register_policy(
        InvocationEfficiencyPolicy.create(
            policy_id=f"issue-728-{route.lower()}",
            version="fixture-v1",
            workload_class=workload_class,
            provider=provider,
            route=route,
            model=model,
            reasoning=reasoning,
            one_turn=True,
            exact_input=True,
            skills_enabled=False,
            tools_enabled=False,
            mcp_enabled=False,
            prior_message_count=0,
            command_semantic_version=(
                GROK_COMMAND_SEMANTIC_VERSION
                if route == CONT_PRIMARY_ROUTE
                else CURSOR_COMMAND_SEMANTIC_VERSION
            ),
            command_flags=(
                CONT_PRIMARY_COMMAND_FLAGS
                if route == CONT_PRIMARY_ROUTE
                else CONT_FALLBACK_COMMAND_FLAGS
            ),
            context_manifest_schema_version=CONT_CONTEXT_MANIFEST_SCHEMA_VERSION,
            disabled_capabilities=CONT_DISABLED_CAPABILITIES,
            implementation_revision=_WRITER_REVISION,
            max_prompt_bytes=max_prompt_bytes,
            max_context_tokens=10_000,
            max_output_tokens=2_000,
            max_total_tokens=12_000,
            prompt_contract_version=CONT_WRITER_PROMPT_CONTRACT_VERSION,
            output_schema_digest=CONT_WRITER_OUTPUT_SCHEMA_DIGEST,
            allowed_context_identities=(CONT_WRITER_CONTEXT_IDENTITY,),
            allowed_config_identities=(
                CONT_PRIMARY_CONFIG_IDENTITY
                if route == CONT_PRIMARY_ROUTE
                else CONT_FALLBACK_CONFIG_IDENTITY,
            ),
            hard_estimate_ceiling_tokens=12_000,
            evidence_digest=digest_canonical({"issue": 728, "route": route}),
            qualified=True,
        )
    )


def test_controller_holds_writer_before_dispatch_without_exact_usage_policy(
    tmp_path: Path,
) -> None:
    unpublished = tmp_path / "usage-policy-held.sqlite3"
    usage = ModelUsageService(str(unpublished))
    calls = 0

    def primary(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "must-not-dispatch"

    report = run_cycle(
        proving_store=str(_proving(tmp_path)),
        unpublished_store=str(unpublished),
        writer=CliChainWriter(primary=primary, fallback=primary),
        evidence_package_builder=_qualified_builder(frozenset({"HK-01"})),
        clock=_CLOCK,
        model_usage=usage,
    )

    retained = usage.report(
        start=datetime(2026, 8, 19, tzinfo=UTC),
        end=datetime(2026, 8, 21, tzinfo=UTC),
    )
    assert calls == 0
    assert report.provider_dispatches == 0
    assert report.draft_hold == 1
    assert retained["envelope_count"] == 1
    assert retained["leaf_dispatch_count"] == 0


def test_controller_allocates_every_primary_and_fallback_leaf_before_dispatch(
    tmp_path: Path,
) -> None:
    unpublished = tmp_path / "usage-primary-fallback.sqlite3"
    usage = ModelUsageService(str(unpublished))
    _register_cont_usage_policy(
        usage,
        workload_class=WorkloadClass.CONT_WRITER_PRIMARY,
        provider=CONT_PRIMARY_PROVIDER,
        route=CONT_PRIMARY_ROUTE,
        model=CONT_PRIMARY_MODEL,
        reasoning=CONT_PRIMARY_REASONING,
    )
    _register_cont_usage_policy(
        usage,
        workload_class=WorkloadClass.CONT_WRITER_FALLBACK,
        provider=CONT_FALLBACK_PROVIDER,
        route=CONT_FALLBACK_ROUTE,
        model=CONT_FALLBACK_MODEL,
        reasoning=CONT_FALLBACK_REASONING,
    )

    report = run_cycle(
        proving_store=str(_proving(tmp_path)),
        unpublished_store=str(unpublished),
        writer=CliChainWriter(
            primary=lambda _prompt: WriterCliExecution(
                text="malformed",
                usage={
                    "usage_basis": "PROVIDER_REPORTED",
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cached_read_tokens": 5,
                    "cached_write_tokens": 0,
                    "reasoning_tokens": 0,
                    "context_tokens": 80,
                    "total_tokens": 125,
                },
            ),
            fallback=lambda prompt: WriterCliExecution(
                text=_valid_cli_json(prompt),
                usage={
                    "usage_basis": "PROVIDER_REPORTED",
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cached_read_tokens": 5,
                    "cached_write_tokens": 0,
                    "reasoning_tokens": 0,
                    "context_tokens": 80,
                    "total_tokens": 125,
                },
            ),
        ),
        evidence_package_builder=_qualified_builder(frozenset({"HK-01"})),
        clock=_CLOCK,
        model_usage=usage,
    )

    retained = usage.query(
        start=datetime(2026, 8, 19, tzinfo=UTC),
        end=datetime(2026, 8, 21, tzinfo=UTC),
    )
    leaves = retained["leaves"]
    assert report.provider_dispatches == 2
    assert report.minted == 1
    assert [leaf["workload_class"] for leaf in leaves] == [
        "CONT_WRITER_PRIMARY",
        "CONT_WRITER_FALLBACK",
    ]
    assert leaves[1]["parent_invocation_id"] == leaves[0]["invocation_id"]
    assert all(leaf["admission_decision_id"] for leaf in leaves)
    assert all(leaf["work_outcome_record_id"] for leaf in leaves)
    assert len({leaf["provider_attempt_id"] for leaf in leaves}) == 2
    assert {leaf["usage_status"] for leaf in leaves} == {"REPORTED"}
    assert sum(leaf["total_tokens"] for leaf in leaves) == 250
    assert {leaf["work_outcome"] for leaf in leaves} == {"ACCEPTED"}
    assert all(leaf["context_manifest"] is not None for leaf in leaves)
    assert all(leaf["context_manifest"]["tool_count"] == 0 for leaf in leaves)
    assert all(leaf["context_manifest"]["mcp_server_count"] == 0 for leaf in leaves)
    assert all(leaf["context_manifest"]["prior_message_count"] == 0 for leaf in leaves)
    assert all(
        leaf["context_manifest_observation"]["provider_context_tokens"] == 80
        for leaf in leaves
    )


def test_oversized_exact_evidence_is_held_before_dispatch_without_truncation(
    tmp_path: Path,
) -> None:
    unpublished = tmp_path / "usage-oversized-exact-input.sqlite3"
    usage = ModelUsageService(str(unpublished))
    _register_cont_usage_policy(
        usage,
        workload_class=WorkloadClass.CONT_WRITER_PRIMARY,
        provider=CONT_PRIMARY_PROVIDER,
        route=CONT_PRIMARY_ROUTE,
        model=CONT_PRIMARY_MODEL,
        reasoning=CONT_PRIMARY_REASONING,
        max_prompt_bytes=1,
    )
    calls = 0

    def primary(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "must-not-dispatch"

    report = run_cycle(
        proving_store=str(_proving(tmp_path)),
        unpublished_store=str(unpublished),
        writer=CliChainWriter(primary=primary),
        evidence_package_builder=_qualified_builder(frozenset({"HK-01"})),
        clock=_CLOCK,
        model_usage=usage,
    )

    assert calls == 0
    assert report.provider_dispatches == 0
    assert report.draft_hold == 1
    assert (
        "EXACT_INPUT_EXCEEDS_QUALIFIED_BOUND",
        1,
    ) in report.draft_reason_counts
    connection = sqlite3.connect(unpublished)
    manifest = json.loads(
        connection.execute(
            "SELECT record_json FROM model_invocation_context_manifests"
        ).fetchone()[0]
    )
    connection.close()
    assert manifest["prompt_bytes"] > 1
    assert manifest["evidence_package_digest"]


def test_dirty_writer_implementation_is_held_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unpublished = tmp_path / "usage-dirty-writer.sqlite3"
    usage = ModelUsageService(str(unpublished))
    _register_cont_usage_policy(
        usage,
        workload_class=WorkloadClass.CONT_WRITER_PRIMARY,
        provider=CONT_PRIMARY_PROVIDER,
        route=CONT_PRIMARY_ROUTE,
        model=CONT_PRIMARY_MODEL,
        reasoning=CONT_PRIMARY_REASONING,
    )
    monkeypatch.setattr(
        "newsroom.control_plane.writer.cont_writer_implementation_identity",
        lambda: (_WRITER_REVISION, False),
    )
    calls = 0

    def primary(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "must-not-dispatch"

    report = run_cycle(
        proving_store=str(_proving(tmp_path)),
        unpublished_store=str(unpublished),
        writer=CliChainWriter(primary=primary),
        evidence_package_builder=_qualified_builder(frozenset({"HK-01"})),
        clock=_CLOCK,
        model_usage=usage,
    )

    assert calls == 0
    assert report.provider_dispatches == 0
    assert report.draft_hold == 1
    assert ("MODEL_USAGE_ADMISSION_HELD", 1) in report.draft_reason_counts


def test_controller_retains_explicit_zero_when_writer_executable_is_missing(
    tmp_path: Path,
) -> None:
    unpublished = tmp_path / "usage-pre-dispatch-zero.sqlite3"
    usage = ModelUsageService(str(unpublished))
    _register_cont_usage_policy(
        usage,
        workload_class=WorkloadClass.CONT_WRITER_PRIMARY,
        provider=CONT_PRIMARY_PROVIDER,
        route=CONT_PRIMARY_ROUTE,
        model=CONT_PRIMARY_MODEL,
        reasoning=CONT_PRIMARY_REASONING,
    )

    def missing_executable(_prompt: str) -> str:
        raise WriterDispatchError(
            "writer executable not found",
            failure_class="SYSTEMIC",
            reason_code="EXECUTABLE_NOT_FOUND",
            provider_dispatched=False,
        )

    report = run_cycle(
        proving_store=str(_proving(tmp_path)),
        unpublished_store=str(unpublished),
        writer=CliChainWriter(primary=missing_executable),
        evidence_package_builder=_qualified_builder(frozenset({"HK-01"})),
        clock=_CLOCK,
        model_usage=usage,
    )

    retained = usage.query(
        start=datetime(2026, 8, 19, tzinfo=UTC),
        end=datetime(2026, 8, 21, tzinfo=UTC),
    )
    assert report.provider_dispatches == 0
    assert ("EXECUTABLE_NOT_FOUND", 1) in report.draft_reason_counts
    assert len(retained["leaves"]) == 1
    assert retained["leaves"][0]["usage_status"] == "REPORTED"
    assert retained["leaves"][0]["total_tokens"] == 0
    assert retained["leaves"][0]["dispatch_at"] is None
    assert retained["leaves"][0]["pre_dispatch_zero_proved"] is True


def test_timeout_and_cancellation_retain_context_manifest_before_cleanup(
    tmp_path: Path,
) -> None:
    for suffix, failure in (
        ("timeout", RuntimeError("grok writer timed out")),
        ("cancelled", KeyboardInterrupt()),
    ):
        fixture_root = tmp_path / suffix
        fixture_root.mkdir()
        unpublished = tmp_path / f"usage-{suffix}.sqlite3"
        usage = ModelUsageService(str(unpublished))
        _register_cont_usage_policy(
            usage,
            workload_class=WorkloadClass.CONT_WRITER_PRIMARY,
            provider=CONT_PRIMARY_PROVIDER,
            route=CONT_PRIMARY_ROUTE,
            model=CONT_PRIMARY_MODEL,
            reasoning=CONT_PRIMARY_REASONING,
        )

        def fail(_prompt: str, *, error: BaseException = failure) -> str:
            raise error

        if isinstance(failure, KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                run_cycle(
                    proving_store=str(_proving(fixture_root)),
                    unpublished_store=str(unpublished),
                    writer=CliChainWriter(primary=fail),
                    evidence_package_builder=_qualified_builder(frozenset({"HK-01"})),
                    clock=_CLOCK,
                    model_usage=usage,
                )
        else:
            run_cycle(
                proving_store=str(_proving(fixture_root)),
                unpublished_store=str(unpublished),
                writer=CliChainWriter(primary=fail),
                evidence_package_builder=_qualified_builder(frozenset({"HK-01"})),
                clock=_CLOCK,
                model_usage=usage,
            )
        leaves = usage.query(
            start=datetime(2026, 8, 19, tzinfo=UTC),
            end=datetime(2026, 8, 21, tzinfo=UTC),
        )["leaves"]
        assert len(leaves) == 1
        assert leaves[0]["context_manifest"] is not None
        assert leaves[0]["context_manifest"]["working_directory_inventory"] == []


def test_previous_v9_reader_preserves_exact_record_without_accepting_unknown_policy():
    from newsroom.control_plane.admission import _PRIOR_CURRENT_WRITE_ADMISSION_POLICY_VERSION, _decision_id
    assert _PRIOR_CURRENT_WRITE_ADMISSION_POLICY_VERSION == (
        "newsroom.write-admission.v9+newsroom.evid-012.v7+"
        "newsroom.evidence-approval.v8+newsroom.evidence-gates.v2+"
        "newsroom.governed-claim.v7+newsroom.governed-input.v10+"
        "newsroom.named-entity.v16+newsroom.cont-originality.v3+"
        "newsroom.zh-hant-hk-shape.v14+newsroom.factual-localisation.v2+"
        "newsroom.qualification-relation.v3")
    candidate, package = _candidate_package()
    decision = DeterministicWriteAdmission().decide(candidate, package, decided_at="2026-09-12T12:00:00Z")
    record = decision.as_record()
    record["policy_version"] = _PRIOR_CURRENT_WRITE_ADMISSION_POLICY_VERSION
    values = {key: getattr(decision, key) for key in record if key not in {"decision_id", "decided_at"}}
    values["policy_version"] = record["policy_version"]
    record["decision_id"] = _decision_id(**values)
    assert WriteAdmissionDecision.from_record(record).as_record() == record
    bad = {**record, "policy_version": record["policy_version"].replace("v9+", "v99+")}
    with pytest.raises(ValueError, match="unsupported write-admission"):
        WriteAdmissionDecision.from_record(bad)
