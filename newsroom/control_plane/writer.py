"""CONT writer: Grok Build CLI, then cursor-agent CLI. Graphiti is never the writer."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal, Protocol, cast

from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.control_plane.child_environment import unprivileged_child_environment
from newsroom.control_plane.diagnostic_logging import emit_diagnostic
from newsroom.control_plane.editorial import StoryCandidateRecord
from newsroom.control_plane.evidence import EvidencePackage, GovernedClaimEvidence
from newsroom.control_plane.governed_context import GovernedContextStatus
from newsroom.control_plane.zh_hant import (
    contains_discourse_filler,
    contains_non_han_letter,
    contains_simplified_variant,
)
from newsroom.graphiti_adapter.cli_process import (
    timeout_deadline_after, timeout_diagnostic, validated_timeout_diagnostics,
)
from newsroom.graphiti_adapter.usage_meter import (
    cursor_cli_usage,
    grok_cli_usage,
    unreported_cli_usage,
)

GROK_BIN = os.environ.get("NEWSROOM_GROK_BIN", "/Users/jamesto/.grok/bin/grok")
GROK_AUTH_FILE = os.environ.get(
    "NEWSROOM_GROK_AUTH_FILE", os.path.expanduser("~/.grok/auth.json")
)
CURSOR_AGENT_BIN = os.environ.get(
    "NEWSROOM_CURSOR_AGENT_BIN", "/Users/jamesto/.local/bin/cursor-agent"
)
# Fixture-record default only. Live dispatch observes the installed binary and
# does not compare it against this string.
GROK_COMMAND_SEMANTIC_VERSION = "1.0.8"
CURSOR_COMMAND_SEMANTIC_VERSION = "2026.08.11-e8db854"
_GROK_COMMAND_SEMANTIC_VERSION_CACHE: str | None = None
CONT_PRIMARY_CONFIG_IDENTITY = "cont-writer-grok-hermetic-command-v3"
CONT_FALLBACK_CONFIG_IDENTITY = "cont-writer-cursor-hermetic-command-v2"
CONT_WRITER_SYSTEM_INSTRUCTION = (
    "你係一個單次、無工具、無工作區嘅 Newsroom 寫作轉換器。"
    "只按用戶提供嘅 CONT 合約、Story Candidate 同 Evidence Package 回覆。"
)
WRITER_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "body": {"type": "string"},
        "evidence_links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "governed_claim_id": {"type": "string"},
                    "rendered_assertion": {"type": "string"},
                },
                "required": [
                    "governed_claim_id",
                    "rendered_assertion",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "body", "evidence_links"],
    "additionalProperties": False,
}
CONT_WRITER_PROMPT_CONTRACT_VERSION = "newsroom.cont-writer.prompt.v3"
CONT_WRITER_CONTEXT_IDENTITY = "cont-evidence-package-only-v1"
CONT_WRITER_OUTPUT_SCHEMA_DIGEST = digest_canonical(WRITER_SCHEMA)
CONT_PRIMARY_PROVIDER = "grok-build-cli"
CONT_PRIMARY_ROUTE = "CONT_PRIMARY"
CONT_PRIMARY_MODEL = "grok-4.6"
CONT_PRIMARY_REASONING = "low"
CONT_FALLBACK_PROVIDER = "cursor-agent-cli"
CONT_FALLBACK_ROUTE = "CONT_FALLBACK"
CONT_FALLBACK_MODEL = "cursor-pinned"
CONT_FALLBACK_REASONING = "provider-default"
CONT_CONTEXT_MANIFEST_SCHEMA_VERSION = "newsroom.cont-writer.context-manifest.v2"
_TRUSTED_GIT_EXECUTABLE = "/usr/bin/git"
_HERMETIC_ENVIRONMENT_KEYS = frozenset(
    {
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
)


def cont_writer_implementation_identity(
    repository: str | None = None,
) -> tuple[str, bool]:
    """Return the Git revision and exact clean-tree state for calibration binding."""

    repository = repository or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/var/empty",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    git = (
        _TRUSTED_GIT_EXECUTABLE,
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.trustctime=true",
        "-c",
        "core.checkStat=default",
        "-c",
        "core.ignoreStat=false",
    )
    try:
        revision = subprocess.run(
            (*git, "rev-parse", "HEAD"),
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            (*git, "status", "--porcelain=v1", "--untracked-files=all"),
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        flags = subprocess.run(
            (*git, "ls-files", "-v", "-z"),
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "UNVERSIONED", False
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        return "UNVERSIONED", False
    index_flags_clean = all(
        record.startswith(b"H ") for record in flags.split(b"\0") if record
    )
    return revision, status == "" and index_flags_clean


def _required_title_and_body(package: EvidencePackage) -> tuple[str, str] | None:
    headlines = tuple(
        claim for claim in package.governed_claims if claim.claim_role == "HEADLINE"
    )
    if len(headlines) != 1:
        return None
    body_claims = tuple(
        claim
        for claim in package.governed_claims
        if claim.claim_role in {"SUBSTANTIVE", "CONTEXT"}
    )
    return (
        f"【未出版】{headlines[0].rendered_assertion_zh_hant_hk}",
        "本報根據已核實證據報道："
        + "；".join(claim.rendered_assertion_zh_hant_hk for claim in body_claims),
    )


_PROMPT = (
    "你係一個單次、無工具嘅 Newsroom claim 拼裝器，唔係記者，亦唔係 Graphiti。"
    "只輸出一個 JSON 物件，欄位 title、body、evidence_links。"
    "title 同 body 必須同下方「必須輸出」完全相同，一個字、標點、空白都唔准改。"
    "唔好抄來源標題或 dateline。"
    "唔准加事實、名字、數字、日期、引句、因果、肯定程度或任何額外字。"
    "唔准 AUTO_PUBLISH，唔准當公開發行。"
    "evidence_links 必須用 approved_governed_claims 入面完全相同嘅 "
    "governed_claim_id 同 rendered_assertion，唔准改寫。"
)
_TITLE_RESIDUE_PREFIXES = ("正在", "搜集", "查核", "先查", "草稿：")
_TITLE_RESIDUE_EXACT = frozenset({"新聞稿任務", "Newsroom 原創稿"})
_BODY_RESIDUE_PREFIXES = ("先查", "先核", "正在核")
_FILLER_MARKERS = (
    "總括而言",
    "總而言之",
    "值得注意的是",
    "放眼未來",
    "時間會證明",
    "草稿：",
)
_CHINESE_NUMERAL_FACT = re.compile(
    r"(?:百分之|星期|第)?[零〇一二三四五六七八九十百千萬万億亿兆兩两"
    r"壹貳贰參叁肆伍陸陆柒捌玖拾佰仟ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+"
    r"(?:[\u3400-\u9fff])?"
)
_QUANTIFIED_FACT = re.compile(
    r"(?P<number>[+\-−]?\d+(?:[.,]\d+)*(?:%|％)?|"
    r"[零〇一二三四五六七八九十百千萬万億亿兆兩两壹貳贰參叁肆伍陸陆柒捌玖拾佰仟"
    r"ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ①-⑳⑴-⒇㉑-㊿]+)\s*"
    r"(?P<unit>平方公里|平方米|公頃|英畝|港元|公噸|公斤|公里|分鐘|分钟|"
    r"小時|小时|星期|個月|个月|階段|阶段|百分比|%|％|元|噸|吨|米|日|天|"
    r"月|年|期|級|级|批|次|成|倍|間|间|部|條|条|所|座|架|輛|辆|艘|層|层|"
    r"項|项|個|个|名|位|戶|户|宗|件|℃|℉|°C|°F)"
    r"(?P<object>[\u3400-\u9fff]{1,8})?(?=$|[，,。；;：:\s])"
)
_NUMBER_ADJACENT_HAN_FACT = re.compile(
    r"(?P<number>第?(?:[+\-−]?\d+(?:[.,]\d+)*(?:%|％)?|"
    r"[零〇一二三四五六七八九十百千萬万億亿兆兩两壹貳贰參叁肆伍陸陆柒捌玖拾佰仟"
    r"ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ①-⑳⑴-⒇㉑-㊿]+))"
    r"\s*(?P<unit>[\u3400-\u9fff]{1,8})(?=$|[，,。；;：:\s])"
)
_CURRENCY_FACT = re.compile(
    r"(?P<currency>HK\$|US\$|£|€|¥|￥|\$)\s*"
    r"(?P<number>[+\-−]?\d+(?:[.,]\d+)*)|"
    r"(?P<number_after>[+\-−]?\d+(?:[.,]\d+)*)\s*"
    r"(?P<currency_after>港元|英鎊|歐元|美元|日圓|人民幣)"
)
_RELATIVE_TIME_FACT = re.compile(
    r"\b(?:today|tomorrow|yesterday|tonight|this\s+(?:morning|afternoon|evening|"
    r"week|month|year)|next\s+(?:week|month|year)|last\s+(?:week|month|year)|"
    r"day\s+after\s+tomorrow|end\s+of\s+(?:the\s+)?(?:month|year)|year[ -]end)\b|"
    r"今日|今天|明日|聽日|听日|昨日|尋日|寻日|下星期|下週|下周|本星期|"
    r"本週|本周|上星期|上週|上周|本月|下月|上月|今年|明年|去年|今早|"
    r"今朝|今晚|今午|後日|后日|月底|月尾|年底|年尾|即日|當日|当日|"
    r"翌日|翌晨|翌晚|本季|今季|下季|上季|本季度|下季度|上季度|清晨|"
    r"早上|上午|中午|下午|傍晚|黃昏|黄昏|晚間|晚间|深夜",
    re.IGNORECASE,
)


def _remove_exact_expressions(text: str, expressions: tuple[str, ...]) -> str:
    for expression in sorted(expressions, key=len, reverse=True):
        text = text.replace(expression, "")
    return text


def _quantified_relations(text: str) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            match.group("number"),
            match.group("unit"),
            match.group("object") or "",
        )
        for match in _QUANTIFIED_FACT.finditer(text)
    )


def _number_adjacent_han_relations(text: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (match.group("number"), match.group("unit"))
        for match in _NUMBER_ADJACENT_HAN_FACT.finditer(text)
    )


def _currency_relations(text: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            match.group("currency") or match.group("currency_after"),
            match.group("number") or match.group("number_after"),
        )
        for match in _CURRENCY_FACT.finditer(text)
    )


def _unicode_currency_relations(text: str) -> tuple[tuple[str, str], ...]:
    relations: list[tuple[str, str]] = []
    for index, character in enumerate(text):
        if unicodedata.category(character) != "Sc":
            continue
        before = re.search(r"[+\-−]?\d+(?:[.,]\d+)*\s*$", text[:index])
        after = re.match(r"\s*([+\-−]?\d+(?:[.,]\d+)*)", text[index + 1 :])
        if before is not None:
            relations.append((character, before.group(0).strip()))
        elif after is not None:
            relations.append((character, after.group(1)))
    return tuple(relations)


def _unicode_number_relations(text: str) -> tuple[tuple[str, float], ...]:
    return tuple(
        (character, unicodedata.numeric(character))
        for character in text
        if unicodedata.category(character).startswith("N")
    )


def _has_unicode_quote_delimiter(text: str) -> bool:
    for character in text:
        name = unicodedata.name(character, "")
        if (
            unicodedata.category(character) in {"Pi", "Pf"}
            or "QUOTATION MARK" in name
            or "ANGLE BRACKET ORNAMENT" in name
        ):
            return True
    return False


def _unicode_quoted_contents(text: str) -> tuple[str, ...]:
    positions = tuple(
        index
        for index, character in enumerate(text)
        if _has_unicode_quote_delimiter(character)
    )
    return tuple(
        text[start + 1 : end]
        for start, end in zip(positions[::2], positions[1::2], strict=False)
        if start + 1 < end
    )


def _unicode_quotes_are_balanced(text: str) -> bool:
    return sum(_has_unicode_quote_delimiter(character) for character in text) % 2 == 0


def _signed_number_relations(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[+\-−]\d+(?:[.,]\d+)*", text))


def _document_year_localisation(claim: GovernedClaimEvidence) -> tuple[tuple[str, str], ...]:
    """Prove one year label on an evidenced named annual document, not a count.

    The immutable claim/rendering stays unchanged. Bare numbers, form numbers,
    durations, extra precision and additional numeric facts receive no exemption.
    """
    source, target = claim.claim, claim.rendered_assertion_zh_hant_hk
    numbers = re.findall(r"\d+(?:[.,]\d+)*", source)
    if len(numbers) != 1 or re.findall(r"\d+(?:[.,]\d+)*", target) != numbers:
        return ()
    year = numbers[0]
    if not re.fullmatch(r"(?:19|20)\d{2}", year):
        return ()
    if any(unicodedata.category(c) == "Sc" or c in "%％" for c in source + target):
        return ()
    if re.search(r"[+\-−]\s*" + year, source + "\n" + target):
        return ()
    match = re.fullmatch(
        r"The (?:deadline|closing date) for the " + year
        + r" (?P<title>(?:[A-Za-z]+[ -]){1,9}(?:return|report)) "
        r"\((?P<term>[A-Z][A-Za-z]{1,9})\) has now (?:passed|expired)\.?", source,
    )
    if match is None or not re.search(r"\b(?:annual|budget|financial|academic)\b", match["title"], flags=re.IGNORECASE):
        return ()
    term = match["term"]
    if (term not in claim.rendered_named_entities
            or not any(text == term and kind == "OFFICIAL_TERM" for text, kind, _ in claim.named_entity_evidence)):
        return ()
    rendered = re.fullmatch(
        re.escape(year) + r"年(?P<title>[^0-9零〇一二三四五六七八九十百千萬億年月日時分秒%％]{1,80})[（(]"
        + re.escape(term) + r"[)）]的(?:截止日期|限期)現已(?:過|屆滿)\。?", target,
    )
    if rendered is None:
        return ()
    if re.search(
        r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
        r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
        r"billion|trillion|half|quarter|dozen|first|second|third|fourth|fifth|sixth|"
        r"seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|"
        r"fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|"
        r"thirtieth|fortieth|fiftieth|sixtieth|seventieth|eightieth|ninetieth|"
        r"hundredth|thousandth|millionth|billionth|trillionth)\b", match["title"],
        flags=re.IGNORECASE,
    ) or re.search(r"[半數数幾几首第廿卅]", rendered["title"]):
        return ()
    return ((year, year + "年"),)


def _writer_numeric_localisations(claim: GovernedClaimEvidence) -> tuple[tuple[str, str], ...]:
    pairs = (*claim.localised_factual_expressions, *_document_year_localisation(claim))
    # Cantonese's indefinite measure phrase is not an invented exact duration.
    # A month/day/hour or any additional numeric fact still faces the ordinary
    # fidelity checks; only this exact source/target phrase is removed.
    short_period = re.fullmatch(
        r"The form will remain open (?P<period>for a short period)"
        r"(?P<late> for late submissions)?\.?", claim.claim,
    )
    rendered_period = re.fullmatch(
        r"表格會繼續開放一段短時間(?P<late>，供逾期提交)?。?",
        claim.rendered_assertion_zh_hant_hk,
    )
    if (short_period is not None and rendered_period is not None
            and bool(short_period["late"]) == bool(rendered_period["late"])):
        pairs += ((short_period["period"], "一段短時間"),)
    return pairs


def _numeric_han_context(text: str) -> str:
    match = re.search(
        r"\d|[零〇一二三四五六七八九十百千萬万億亿兆兩两壹貳贰參叁肆伍陸陆"
        r"柒捌玖拾佰仟ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ①-⑳⑴-⒇㉑-㊿]",
        text,
    )
    if match is None or not re.search(r"[\u3400-\u9fff]", text):
        return ""
    return "".join(
        character
        for character in text
        if character.isdigit() or "\u3400" <= character <= "\u9fff"
    )


_SYSTEMIC_MARKERS = (
    "authentication",
    "not logged in",
    "login required",
    "unauthorized",
    "forbidden",
    "invalid api key",
    "api key invalid",
    "permission denied",
    "invalid model",
    "invalid command",
    "configuration",
    "quota",
    "rate limit",
    "rate-limit",
    "rate_limit_exceeded",
    "rate_limit_error",
    "too many requests",
    "payment required",
    "billing required",
    "no such file",
    "not found",
)

WriterRoute = Literal["PRIMARY", "FALLBACK"]
WriterFailureClass = Literal[
    "FALLBACK_ELIGIBLE", "SYSTEMIC", "CANDIDATE_LOCAL", "UNKNOWN"
]
ORIGINALITY_ALIGNMENT_MIN_SOURCE_COVERAGE = 0.8


@dataclass(frozen=True, slots=True)
class WriterEvidenceLink:
    governed_claim_id: str
    rendered_assertion: str


def _source_ordered_story_claims(
    package: EvidencePackage,
) -> tuple[GovernedClaimEvidence, ...]:
    ordered: list[tuple[int, int, GovernedClaimEvidence]] = []
    for claim in package.governed_claims:
        if claim.claim_role not in {"HEADLINE", "SUBSTANTIVE", "CONTEXT"}:
            continue
        if claim.passage_index >= len(package.passages):
            raise ValueError("governed claim source span order is unavailable")
        passage = package.passages[claim.passage_index]
        starts: list[int] = []
        offset = 0
        while True:
            start = passage.find(claim.supporting_excerpt, offset)
            if start < 0:
                break
            starts.append(start)
            offset = start + 1
        claim_offset = claim.supporting_excerpt.find(claim.claim)
        if (
            len(starts) != 1
            or claim_offset < 0
            or claim.supporting_excerpt.find(claim.claim, claim_offset + 1) >= 0
        ):
            raise ValueError("governed claim source span order is ambiguous")
        ordered.append(
            (claim.passage_index, starts[0] + claim_offset, claim)
        )
    if len({(passage, start) for passage, start, _claim in ordered}) != len(ordered):
        raise ValueError("governed claim source span order is ambiguous")
    return tuple(
        claim
        for _passage, _start, claim in sorted(
            ordered, key=lambda item: (item[0], item[1])
        )
    )


def required_surface_copy(
    package: EvidencePackage,
    *, paragraphs: bool = False,
    context_preserving: bool = False,
) -> tuple[str, str, tuple[WriterEvidenceLink, ...]]:
    required = _required_title_and_body(package)
    if required is None:
        raise ValueError("Evidence Package requires exactly one HEADLINE claim")
    title, body = required
    headline = next(
        claim for claim in package.governed_claims if claim.claim_role == "HEADLINE"
    )
    body_claims = tuple(
        claim
        for claim in package.governed_claims
        if claim.claim_role in {"SUBSTANTIVE", "CONTEXT"}
    )
    linked_claims = (headline, *body_claims)
    if context_preserving:
        if not paragraphs:
            raise ValueError("context-preserving copy requires paragraph rendering")
        linked_claims = _source_ordered_story_claims(package)
        body = "\n\n".join(
            claim.rendered_assertion_zh_hant_hk for claim in linked_claims
        )
    elif paragraphs:
        body = "\n\n".join(claim.rendered_assertion_zh_hant_hk for claim in body_claims)
    return (
        title,
        body,
        tuple(
            WriterEvidenceLink(
                governed_claim_id=claim.claim_id,
                rendered_assertion=claim.rendered_assertion_zh_hant_hk,
            )
            for claim in linked_claims
        ),
    )


@dataclass(frozen=True, slots=True)
class WriterValidatorResult:
    validator: str
    result: Literal["PASS", "FAIL"]
    reason_code: str


@dataclass(frozen=True, slots=True)
class WriterCopy:
    title: str
    body: str
    writer_id: str
    evidence_package_digest: str = ""
    evidence_links: tuple[WriterEvidenceLink, ...] = ()
    usage: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class WriterInvocationManifest:
    schema_version: str
    provider: str
    route: str
    model: str
    reasoning: str
    command_semantic_version: str
    command_flags: tuple[str, ...]
    implementation_revision: str
    implementation_worktree_clean: bool
    prompt_contract_version: str
    system_bytes: int
    system_digest: str
    prompt_bytes: int
    prompt_digest: str
    schema_bytes: int
    schema_digest: str
    request_digest: str
    output_schema_digest: str
    context_manifest_digest: str
    context_identity: str
    config_identity: str
    allowed_config_digests: tuple[str, ...]
    working_directory_inventory: tuple[str, ...]
    working_directory_inventory_digest: str
    disabled_capabilities: tuple[str, ...]
    evidence_package_digest: str
    evidence_package_bytes: int
    one_turn: bool
    exact_input: bool
    skills_enabled: bool
    tools_enabled: bool
    mcp_enabled: bool
    prior_message_count: int
    skill_count: int
    tool_count: int
    mcp_server_count: int
    mcp_tool_count: int

    @classmethod
    def create(cls, **values: object) -> WriterInvocationManifest:
        """Build one canonical manifest from the typed serialisation contract."""

        values.pop("context_manifest_digest", None)
        constructor = cast(Callable[..., WriterInvocationManifest], cls)
        draft = constructor(**values, context_manifest_digest="")
        canonical = draft.as_record()
        canonical.pop("context_manifest_digest")
        return constructor(
            **values,
            context_manifest_digest=digest_canonical(canonical),
        )

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "context_manifest_digest": self.context_manifest_digest,
            "provider": self.provider,
            "route": self.route,
            "model": self.model,
            "reasoning": self.reasoning,
            "command_semantic_version": self.command_semantic_version,
            "command_flags": list(self.command_flags),
            "implementation_revision": self.implementation_revision,
            "implementation_worktree_clean": self.implementation_worktree_clean,
            "prompt_contract_version": self.prompt_contract_version,
            "system_bytes": self.system_bytes,
            "system_digest": self.system_digest,
            "prompt_bytes": self.prompt_bytes,
            "prompt_digest": self.prompt_digest,
            "schema_bytes": self.schema_bytes,
            "schema_digest": self.schema_digest,
            "request_digest": self.request_digest,
            "output_schema_digest": self.output_schema_digest,
            "context_identity": self.context_identity,
            "config_identity": self.config_identity,
            "allowed_config_digests": list(self.allowed_config_digests),
            "working_directory_inventory": list(self.working_directory_inventory),
            "working_directory_inventory_digest": (
                self.working_directory_inventory_digest
            ),
            "disabled_capabilities": list(self.disabled_capabilities),
            "evidence_package_digest": self.evidence_package_digest,
            "evidence_package_bytes": self.evidence_package_bytes,
            "one_turn": self.one_turn,
            "exact_input": self.exact_input,
            "skills_enabled": self.skills_enabled,
            "tools_enabled": self.tools_enabled,
            "mcp_enabled": self.mcp_enabled,
            "prior_message_count": self.prior_message_count,
            "skill_count": self.skill_count,
            "tool_count": self.tool_count,
            "mcp_server_count": self.mcp_server_count,
            "mcp_tool_count": self.mcp_tool_count,
            "provider_context_tokens": None,
        }


@dataclass(frozen=True, slots=True)
class WriterCliExecution:
    text: str
    usage: dict[str, object]


@dataclass(frozen=True, slots=True)
class WriterRouteProbeResult:
    executable_ok: bool
    authentication_ok: bool
    configuration_ok: bool
    provider_available: bool
    provider_dispatched: bool
    provider_receipt_reference: str | None


class WriterDispatchError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failure_class: WriterFailureClass,
        reason_code: str,
        usage: dict[str, object] | None = None,
        provider_dispatched: bool = True,
    ) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.reason_code = reason_code
        self.usage = usage
        self.provider_dispatched = provider_dispatched


class CliProcessError(RuntimeError):
    def __init__(self, message: str, *, provider_status: int | None) -> None:
        super().__init__(message)
        self.provider_status = provider_status


class CliTimeoutError(RuntimeError):
    """The CLI wall deadline expired; provider completion and usage stay unknown."""

    def __init__(self, message: str, *, evidence: dict | None = None) -> None:
        super().__init__(message)
        self.evidence = (None if evidence is None
                         else validated_timeout_diagnostics([evidence])[0])
        self.diagnostic_reference: dict | None = None


def _timeout_output_bytes(value: object) -> bytes | None:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8", errors="surrogatepass")
    return None  # Unexpected output types remain unobserved, never fabricated.


class CliEmptyOutputError(RuntimeError):
    """The CLI exited successfully without a retained provider response."""


@dataclass(frozen=True, slots=True)
class _HermeticWorkspace:
    root: str
    cwd: str
    home: str
    request: str
    environment: dict[str, str]


def _hermetic_workspace(root: str, *, binary: str) -> _HermeticWorkspace:
    paths = {
        "cwd": os.path.join(root, "workspace"),
        "home": os.path.join(root, "home"),
        "request": os.path.join(root, "request"),
        "tmp": os.path.join(root, "tmp"),
        "config": os.path.join(root, "xdg-config"),
        "data": os.path.join(root, "xdg-data"),
        "cache": os.path.join(root, "xdg-cache"),
        "state": os.path.join(root, "xdg-state"),
    }
    for path in paths.values():
        os.mkdir(path, mode=0o700)
    binary_dirs = tuple(
        dict.fromkeys(
            (
                os.path.dirname(binary),
                "/usr/bin",
                "/bin",
                "/usr/sbin",
                "/sbin",
            )
        )
    )
    environment = {
        "HOME": paths["home"],
        "LANG": "en_GB.UTF-8",
        "LC_ALL": "en_GB.UTF-8",
        "PATH": os.pathsep.join(binary_dirs),
        "TMPDIR": paths["tmp"],
        "XDG_CACHE_HOME": paths["cache"],
        "XDG_CONFIG_HOME": paths["config"],
        "XDG_DATA_HOME": paths["data"],
        "XDG_STATE_HOME": paths["state"],
    }
    if set(environment) != _HERMETIC_ENVIRONMENT_KEYS:
        raise RuntimeError("hermetic writer environment contract drifted")
    return _HermeticWorkspace(
        root=root,
        cwd=paths["cwd"],
        home=paths["home"],
        request=paths["request"],
        environment=environment,
    )


def _minimal_grok_auth_bytes() -> bytes:
    try:
        descriptor = os.open(
            GROK_AUTH_FILE,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise WriterDispatchError(
            "minimal Grok authentication is unavailable",
            failure_class="SYSTEMIC",
            reason_code="MINIMAL_AUTHENTICATION_UNAVAILABLE",
            provider_dispatched=False,
        ) from exc
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        raw = handle.read(65_537)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o077
        or metadata.st_size > 65_536
        or len(raw) > 65_536
    ):
        raise WriterDispatchError(
            "minimal Grok authentication has unsafe file permissions",
            failure_class="SYSTEMIC",
            reason_code="MINIMAL_AUTHENTICATION_UNAVAILABLE",
            provider_dispatched=False,
        )
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WriterDispatchError(
            "minimal Grok authentication is malformed",
            failure_class="SYSTEMIC",
            reason_code="MINIMAL_AUTHENTICATION_UNAVAILABLE",
            provider_dispatched=False,
        ) from exc
    identities = values.values() if isinstance(values, dict) else ()
    if not any(
        isinstance(identity, dict)
        and any(
            isinstance(identity.get(field), str) and bool(identity[field].strip())
            for field in ("key", "refresh_token")
        )
        for identity in identities
    ):
        raise WriterDispatchError(
            "minimal Grok authentication lacks a permitted login credential",
            failure_class="SYSTEMIC",
            reason_code="MINIMAL_AUTHENTICATION_UNAVAILABLE",
            provider_dispatched=False,
        )
    return raw


def _install_minimal_grok_auth(workspace: _HermeticWorkspace, raw: bytes) -> None:
    grok_home = os.path.join(workspace.home, ".grok")
    os.mkdir(grok_home, mode=0o700)
    auth_path = os.path.join(grok_home, "auth.json")
    descriptor = os.open(auth_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)


def parse_grok_command_semantic_version(text: str) -> str | None:
    match = re.search(r"\bgrok\s+(\d+\.\d+\.\d+)\b", text)
    return None if match is None else match.group(1)


def read_grok_command_semantic_version() -> str:
    """Return the installed Grok command version. Not a qualification pin."""

    global _GROK_COMMAND_SEMANTIC_VERSION_CACHE
    if _GROK_COMMAND_SEMANTIC_VERSION_CACHE is not None:
        return _GROK_COMMAND_SEMANTIC_VERSION_CACHE
    with tempfile.TemporaryDirectory(prefix="newsroom-grok-version-") as root:
        workspace = _hermetic_workspace(root, binary=GROK_BIN)
        result = _run_predispatch((GROK_BIN, "version"), workspace=workspace)
    parsed = parse_grok_command_semantic_version(result.stdout)
    if result.returncode != 0 or parsed is None:
        raise WriterDispatchError(
            "Grok command semantic version is not readable",
            failure_class="SYSTEMIC",
            reason_code="HERMETIC_COMMAND_VERSION_UNQUALIFIED",
            provider_dispatched=False,
        )
    _GROK_COMMAND_SEMANTIC_VERSION_CACHE = parsed
    return parsed


def _run_predispatch(
    command: tuple[str, ...], *, workspace: _HermeticWorkspace
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            cwd=workspace.cwd,
            env=workspace.environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WriterDispatchError(
            "hermetic writer preflight failed",
            failure_class="SYSTEMIC",
            reason_code="HERMETIC_PREFLIGHT_FAILED",
            provider_dispatched=False,
        ) from exc


def _prove_grok_hermetic_capabilities(auth: bytes) -> None:
    with tempfile.TemporaryDirectory(prefix="newsroom-grok-preflight-") as root:
        workspace = _hermetic_workspace(root, binary=GROK_BIN)
        _install_minimal_grok_auth(workspace, auth)
        version = _run_predispatch((GROK_BIN, "version"), workspace=workspace)
        observed = parse_grok_command_semantic_version(version.stdout)
        if version.returncode != 0 or observed is None:
            raise WriterDispatchError(
                "Grok command semantic version is not readable",
                failure_class="SYSTEMIC",
                reason_code="HERMETIC_COMMAND_VERSION_UNQUALIFIED",
                provider_dispatched=False,
            )
        global _GROK_COMMAND_SEMANTIC_VERSION_CACHE
        _GROK_COMMAND_SEMANTIC_VERSION_CACHE = observed
        inspection = _run_predispatch(
            (GROK_BIN, "--cwd", workspace.cwd, "inspect", "--json"),
            workspace=workspace,
        )
        try:
            manifest = json.loads(inspection.stdout)
        except json.JSONDecodeError as exc:
            raise WriterDispatchError(
                "Grok capability inspection is not machine-readable",
                failure_class="SYSTEMIC",
                reason_code="HERMETIC_CAPABILITY_UNPROVABLE",
                provider_dispatched=False,
            ) from exc
        if not isinstance(manifest, dict):
            raise WriterDispatchError(
                "Grok capability inspection has the wrong shape",
                failure_class="SYSTEMIC",
                reason_code="HERMETIC_CAPABILITY_UNPROVABLE",
                provider_dispatched=False,
            )
        permissions = manifest.get("permissions", {})
        config_sources = manifest.get("configSources", {})
        empty_fields = (
            "projectInstructions",
            "skills",
            "plugins",
            "marketplaces",
            "mcpServers",
            "hooks",
            "lspServers",
        )
        proved = (
            inspection.returncode == 0
            and manifest.get("grokVersion") == observed
            and manifest.get("projectRoot") is None
            and all(manifest.get(field) == [] for field in empty_fields)
            and isinstance(permissions, dict)
            and permissions.get("loaded") == 0
            and isinstance(config_sources, dict)
            and config_sources.get("layers") == []
        )
        if not proved:
            raise WriterDispatchError(
                "Grok zero-skill, zero-MCP context cannot be proved",
                failure_class="SYSTEMIC",
                reason_code="HERMETIC_CAPABILITY_UNPROVABLE",
                provider_dispatched=False,
            )


def require_permitted_context(
    candidate: StoryCandidateRecord,
    package: EvidencePackage,
) -> None:
    context = package.admitted_context
    candidate_context = candidate.governed_context
    if (context is None) != (candidate_context is None) or (
        context is not None
        and candidate_context is not None
        and context.digest != candidate_context.digest
    ):
        raise WriterDispatchError(
            "candidate and evidence governed context differ",
            failure_class="CANDIDATE_LOCAL",
            reason_code="GOVERNED_CONTEXT_DRIFT",
        )
    if context is not None and context.status is GovernedContextStatus.HOLD:
        raise WriterDispatchError(
            "admitted structured context is held",
            failure_class="CANDIDATE_LOCAL",
            reason_code="GOVERNED_CONTEXT_HELD",
        )
    if context is not None and (
        context.stale or context.degraded or context.projection_gap_count != 0
    ):
        raise WriterDispatchError(
            "admitted structured context is not current and gap-free",
            failure_class="CANDIDATE_LOCAL",
            reason_code="GOVERNED_CONTEXT_NOT_CURRENT",
        )
    if context is not None and not context.currency_consistent:
        raise WriterDispatchError(
            "admitted structured context currency differs from its items",
            failure_class="CANDIDATE_LOCAL",
            reason_code="GOVERNED_CONTEXT_CURRENCY_DRIFT",
        )


class WriterPort(Protocol):
    writer_id: str

    def write(
        self, candidate: StoryCandidateRecord, package: EvidencePackage
    ) -> WriterCopy: ...


class DispatchWriterPort(Protocol):
    writer_id: str

    def dispatch(
        self,
        candidate: StoryCandidateRecord,
        package: EvidencePackage,
        *,
        route: WriterRoute,
    ) -> WriterCopy: ...


class FixtureWriter:
    writer_id = "evaluation-fixture-writer-v1"

    def write(
        self, candidate: StoryCandidateRecord, package: EvidencePackage
    ) -> WriterCopy:
        require_permitted_context(candidate, package)
        title, body, links = required_surface_copy(package)
        return WriterCopy(
            title=title,
            body=body,
            writer_id=self.writer_id,
            evidence_package_digest=package.digest,
            evidence_links=links,
        )


def _writer_evidence_value(package: EvidencePackage) -> dict[str, object]:
    approved_claims = [
        {
            "governed_claim_id": claim.claim_id,
            "claim": claim.claim,
            "supporting_excerpt": claim.supporting_excerpt,
            "rendered_assertion": claim.rendered_assertion_zh_hant_hk,
            "claim_role": claim.claim_role,
            "status": claim.status.value,
            "attribution": claim.attribution,
            "named_entities": list(claim.named_entities),
            "rendered_named_entities": list(claim.rendered_named_entities),
            "quotations": list(claim.quotations),
            "certainty": claim.certainty,
            "originality_basis": claim.originality_basis,
            "originality_policy_version": claim.originality_policy_version,
        }
        for claim in package.governed_claims
    ]
    admitted_context = (
        None
        if package.admitted_context is None
        else package.admitted_context.canonical_value()
    )
    return {
        "approved_governed_claims": approved_claims,
        "permitted_admitted_structured_context": admitted_context,
        "passages": list(package.passages),
    }


def _prompt(candidate: StoryCandidateRecord, package: EvidencePackage) -> str:
    del candidate
    evidence = _writer_evidence_value(package)
    parts = [_PROMPT]
    required = _required_title_and_body(package)
    if required is not None:
        title, body, links = required_surface_copy(package)
        links_json = json.dumps(
            [
                {
                    "governed_claim_id": link.governed_claim_id,
                    "rendered_assertion": link.rendered_assertion,
                }
                for link in links
            ],
            ensure_ascii=False,
        )
        parts.extend(
            (
                f"必須輸出嘅 title：{title}",
                f"必須輸出嘅 body：{body}",
                f"必須輸出嘅 evidence_links：{links_json}",
            )
        )
        return "\n".join(parts)
    parts.extend(
        (
            "approved_governed_claims："
            + json.dumps(evidence["approved_governed_claims"], ensure_ascii=False),
            "permitted_admitted_structured_context："
            + json.dumps(
                evidence["permitted_admitted_structured_context"],
                ensure_ascii=False,
            ),
            "來源（禁止入稿）：\n" + "\n---\n".join(package.passages),
        )
    )
    return "\n".join(parts)


def _extract_json(raw: str) -> str:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("writer returned no JSON object")
    return raw[start : end + 1]


def _copy_fields(
    payload: object,
) -> tuple[str, str, tuple[WriterEvidenceLink, ...]] | None:
    if not isinstance(payload, dict):
        return None
    title = payload.get("title")
    body = payload.get("body")
    if isinstance(title, str) and isinstance(body, str):
        raw_links = payload.get("evidence_links", [])
        if not isinstance(raw_links, list):
            return None
        links: list[WriterEvidenceLink] = []
        for item in raw_links:
            if not isinstance(item, dict):
                return None
            claim_id = item.get("governed_claim_id")
            rendered = item.get("rendered_assertion")
            if not isinstance(claim_id, str) or not isinstance(rendered, str):
                return None
            links.append(WriterEvidenceLink(claim_id.strip(), rendered.strip()))
        return title.strip(), body.strip(), tuple(links)
    return None


def _finished_copy(
    title: str,
    body: str,
    links: tuple[WriterEvidenceLink, ...] = (),
) -> tuple[str, str, tuple[WriterEvidenceLink, ...]]:
    if not title or not body:
        raise RuntimeError("writer JSON missing title or body")
    if title.startswith(_TITLE_RESIDUE_PREFIXES) or title in _TITLE_RESIDUE_EXACT:
        raise RuntimeError("writer returned planning residue, not unpublished copy")
    if body.startswith(_BODY_RESIDUE_PREFIXES):
        raise RuntimeError("writer returned planning residue, not unpublished copy")
    return title, body, links


def _parse_copy(raw: str) -> tuple[str, str, tuple[WriterEvidenceLink, ...]]:
    payload = json.loads(_extract_json(raw))
    found = _copy_fields(payload)
    if found:
        return _finished_copy(*found)
    for key in ("structured_output", "structuredOutput"):
        found = _copy_fields(payload.get(key))
        if found:
            return _finished_copy(*found)
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        found = _copy_fields(json.loads(_extract_json(text)))
        if found:
            return _finished_copy(*found)
    raise RuntimeError("writer JSON missing title or body")


def validate_writer_copy(
    copy: WriterCopy, package: EvidencePackage
) -> tuple[WriterValidatorResult, ...]:
    text = f"{copy.title}\n{copy.body}"
    results: list[WriterValidatorResult] = []

    def check(name: str, passed: bool, reason: str) -> None:
        results.append(
            WriterValidatorResult(name, "PASS" if passed else "FAIL", reason)
        )

    check(
        "EVIDENCE_PACKAGE_BINDING",
        copy.evidence_package_digest == package.digest,
        "EVIDENCE_PACKAGE_DRIFT",
    )
    text_without_approved_entities = text
    for claim in package.governed_claims:
        for entity in claim.named_entities:
            text_without_approved_entities = text_without_approved_entities.replace(
                entity, ""
            )
    check(
        "COMPLETED_ORIGINAL_ZH_HANT_HK_REPORT",
        bool(copy.title.strip() and copy.body.strip())
        and any("\u3400" <= character <= "\u9fff" for character in text)
        and not contains_non_han_letter(text_without_approved_entities)
        and not contains_simplified_variant(text_without_approved_entities),
        "NOT_COMPLETED_ZH_HANT_HK_REPORT",
    )
    check(
        "NO_PLANNING_RESIDUE_OR_FILLER",
        not copy.title.removeprefix("【未出版】").startswith(_TITLE_RESIDUE_PREFIXES)
        and copy.title.removeprefix("【未出版】") not in _TITLE_RESIDUE_EXACT
        and not copy.body.startswith(_BODY_RESIDUE_PREFIXES)
        and not any(marker in text for marker in _FILLER_MARKERS)
        and not contains_discourse_filler(text),
        "PLANNING_RESIDUE_OR_FILLER",
    )
    check(
        "NO_RAW_PROPOSED_GRAPHITI_CONTEXT",
        "PROPOSED Graphiti" not in text and "graphiti_workspace" not in text,
        "RAW_PROPOSED_GRAPHITI_CONTEXT",
    )
    governed_claims = {claim.claim_id: claim for claim in package.governed_claims}
    exact_links = bool(copy.evidence_links) and all(
        (governed := governed_claims.get(link.governed_claim_id)) is not None
        and link.rendered_assertion == governed.rendered_assertion_zh_hant_hk
        and link.rendered_assertion in text
        for link in copy.evidence_links
    )
    check(
        "CLAIM_EVIDENCE_LINKS",
        exact_links,
        "UNSUPPORTED_MATERIAL_CLAIM",
    )
    linked_ids = tuple(link.governed_claim_id for link in copy.evidence_links)
    headline_claims = tuple(
        claim for claim in package.governed_claims if claim.claim_role == "HEADLINE"
    )
    substantive_claims = tuple(
        claim
        for claim in package.governed_claims
        if claim.claim_role == "SUBSTANTIVE"
        and claim.claim in package.substantive_new_information
    )
    rendered_assertions = tuple(
        claim.rendered_assertion_zh_hant_hk for claim in package.governed_claims
    )
    context_preserving_claims = (
        _source_ordered_story_claims(package)
        if copy.writer_id == "newsroom.offline-exact-copy.v3"
        else ()
    )
    exact_role_structure = (
        len(rendered_assertions) == len(set(rendered_assertions))
        and len(headline_claims) == 1
        and copy.title
        == f"【未出版】{headline_claims[0].rendered_assertion_zh_hant_hk}"
        and (
            copy.body == "\n\n".join(
                claim.rendered_assertion_zh_hant_hk
                for claim in context_preserving_claims
            ) if copy.writer_id == "newsroom.offline-exact-copy.v3" else
            copy.body == "\n\n".join(
                claim.rendered_assertion_zh_hant_hk for claim in package.governed_claims
                if claim.claim_role in {"SUBSTANTIVE", "CONTEXT"}
            ) if copy.writer_id == "newsroom.offline-exact-copy.v2" else
            copy.body.startswith("本報根據已核實證據報道：")
            and copy.body.count("本報根據已核實證據報道：") == 1
        )
        and all(
            copy.title.count(claim.rendered_assertion_zh_hant_hk) == 1
            and copy.body.count(claim.rendered_assertion_zh_hant_hk)
            == (1 if copy.writer_id == "newsroom.offline-exact-copy.v3" else 0)
            if claim.claim_role == "HEADLINE"
            else copy.title.count(claim.rendered_assertion_zh_hant_hk) == 0
            and copy.body.count(claim.rendered_assertion_zh_hant_hk) == 1
            for claim in package.governed_claims
        )
    )
    check(
        "ROLE_SPECIFIC_EXACT_ONCE_STRUCTURE",
        exact_role_structure,
        "DUPLICATE_OR_MISPLACED_GOVERNED_CLAIM",
    )
    check(
        "REQUIRED_GOVERNED_CLAIM_COVERAGE",
        len(linked_ids) == len(set(linked_ids))
        and set(linked_ids) == set(governed_claims)
        and len(headline_claims) == 1
        and headline_claims[0].rendered_assertion_zh_hant_hk in copy.title
        and bool(substantive_claims)
        and all(
            claim.rendered_assertion_zh_hant_hk in copy.body
            for claim in substantive_claims
        ),
        "REQUIRED_GOVERNED_CLAIM_MISSING",
    )
    narrative_segments = tuple(
        segment.strip() for segment in re.split(r"\n+", text) if segment.strip()
    )
    check(
        "CENTRAL_CLAIM_COVERAGE",
        bool(copy.evidence_links)
        and all(
            any(link.rendered_assertion in segment for link in copy.evidence_links)
            for segment in narrative_segments
        ),
        "UNMAPPED_CENTRAL_CLAIM",
    )
    allowed_scaffolding = (
        "【未出版】",
        "本報根據已核實證據報道：",
    )
    bounded_segments = True
    for segment in narrative_segments:
        residue = segment
        for link in sorted(
            copy.evidence_links, key=lambda item: -len(item.rendered_assertion)
        ):
            residue = residue.replace(link.rendered_assertion, "")
        for scaffold in allowed_scaffolding:
            residue = residue.replace(scaffold, "")
        residue = re.sub(r"[\s，,；;：:、（）()【】《》〈〉—-]+", "", residue)
        if residue:
            bounded_segments = False
            break
    check(
        "GOVERNED_CLAIM_ENTAILMENT_BOUNDARY",
        exact_links and exact_role_structure and bounded_segments,
        "UNSUPPORTED_CLAIM_RESIDUE",
    )
    source_expressions = (*package.passages,) + tuple(
        value.strip()
        for item in package.passages
        for value in re.split(r"(?<=[.!?。！？；;])|\n+", item)
        if value.strip()
    )
    numeric_expression_patterns = (
        re.compile(r"\d+(?:(?:[年月日時时分秒號号點点])|(?:[.,:/-]\d+)|\d+)+"),
        re.compile(
            r"(?:[零〇一二三四五六七八九十百千萬万億亿兆兩两]+"
            r"(?:年|月|日|號|号|時|时|分|秒|點|点))+"
        ),
        re.compile(
            r"[零〇一二三四五六七八九十百千萬万億亿兆兩两]+\s*"
            r"(?:港元|元|人|宗|件|公噸|噸|吨|公斤|公里|米|分鐘|分钟|小時|"
            r"小时|日|天|星期|個月|个月|年|戶|户|名|位|%|％)"
        ),
        re.compile(
            r"第?[零〇一二三四五六七八九十百千萬万億亿兆兩两]+\s*"
            r"(?:階段|阶段|間|间|批|次|項|项|個|个|所|座|架|輛|辆|艘|"
            r"層|层|期|級|级|成|倍)"
        ),
        re.compile(
            r"\d+(?:[.,]\d+)*\s*(?:%|％|minutes?|mins?|hours?|hrs?|days?|"
            r"weeks?|months?|years?|million|billion|trillion|people|cases?|"
            r"tonnes?|kilograms?|kilometres?|公里|米|分鐘|分钟|小時|小时|日|天|"
            r"星期|個月|个月|年|港元|元|人|宗|件|公噸|噸|吨|公斤|戶|户|名|位)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|"
            r"August|September|October|November|December)(?:\s+\d{1,2})?"
            r"(?:,?\s+\d{4})?",
            re.IGNORECASE,
        ),
        re.compile(r"[零〇一二三四五六七八九十百千萬万億亿兆兩两]+"),
    )
    numeric_localisations = {
        claim.claim_id: _writer_numeric_localisations(claim)
        for claim in package.governed_claims
    }
    approved_numeric_expressions = tuple(
        match.group(0)
        for claim in package.governed_claims
        for evidence_text in (claim.claim, claim.supporting_excerpt)
        for pattern in numeric_expression_patterns
        for match in pattern.finditer(evidence_text)
    ) + tuple(
        match.group(0)
        for claim in package.governed_claims
        for _source, target in numeric_localisations[claim.claim_id]
        for pattern in numeric_expression_patterns
        for match in pattern.finditer(target)
    )
    approved_overlap = (
        tuple(
            value
            for claim in package.governed_claims
            for value in (*claim.named_entities, *claim.quotations)
            if value
        )
        + approved_numeric_expressions
    )

    def normalise_originality(value: str) -> str:
        for approved in sorted(approved_overlap, key=len, reverse=True):
            value = value.replace(approved, "")
        return "".join(
            character.casefold() for character in value if character.isalnum()
        )

    normalised_draft = normalise_originality(text)
    copied_source_expression = any(
        sequence in normalised_draft
        for expression in source_expressions
        for normalised_expression in (normalise_originality(expression),)
        for sequence in (
            normalised_expression[index : index + 12]
            for index in range(max(0, len(normalised_expression) - 11))
        )
    )
    aligned_source_expression = any(
        len(normalised_expression) >= 4
        and sum(
            block.size
            for block in SequenceMatcher(
                None,
                normalised_expression,
                normalised_segment,
                autojunk=False,
            ).get_matching_blocks()
        )
        / len(normalised_expression)
        >= ORIGINALITY_ALIGNMENT_MIN_SOURCE_COVERAGE
        for expression in source_expressions
        for normalised_expression in (normalise_originality(expression),)
        for segment in narrative_segments
        for normalised_segment in (normalise_originality(segment),)
    )
    check(
        "ORIGINALITY_BOUNDARY",
        not copied_source_expression and not aligned_source_expression,
        "VERBATIM_SOURCE_EXPRESSION",
    )
    governed_text = "\n".join(
        value
        for claim in package.governed_claims
        for value in (claim.claim, claim.supporting_excerpt)
    )
    numbers = set(re.findall(r"\d+(?:[.,]\d+)*(?:%|％)?", text))
    governed_numbers = set(re.findall(r"\d+(?:[.,]\d+)*(?:%|％)?", governed_text))
    governed_numbers.update(
        number
        for claim in package.governed_claims
        for _source, target in numeric_localisations[claim.claim_id]
        for number in re.findall(r"\d+(?:[.,]\d+)*(?:%|％)?", target)
    )
    draft_numeric_expressions = {
        match.group(0)
        for pattern in numeric_expression_patterns
        for match in pattern.finditer(text)
    }
    claim_numeric_relations = all(
        _quantified_relations(source_without_localised)
        == _quantified_relations(rendered_without_localised)
        and _number_adjacent_han_relations(source_without_localised)
        == _number_adjacent_han_relations(rendered_without_localised)
        and _currency_relations(source_without_localised)
        == _currency_relations(rendered_without_localised)
        and _unicode_currency_relations(source_without_localised)
        == _unicode_currency_relations(rendered_without_localised)
        and _unicode_number_relations(source_without_localised)
        == _unicode_number_relations(rendered_without_localised)
        and _signed_number_relations(source_without_localised)
        == _signed_number_relations(rendered_without_localised)
        and _numeric_han_context(source_without_localised)
        == _numeric_han_context(rendered_without_localised)
        and tuple(
            match.group(0)
            for match in _CHINESE_NUMERAL_FACT.finditer(source_without_localised)
        )
        == tuple(
            match.group(0)
            for match in _CHINESE_NUMERAL_FACT.finditer(rendered_without_localised)
        )
        for claim in package.governed_claims
        for source_without_localised, rendered_without_localised in (
            (
                _remove_exact_expressions(
                    claim.claim,
                    tuple(
                        source
                        for source, _target in numeric_localisations[claim.claim_id]
                    ),
                ),
                _remove_exact_expressions(
                    claim.rendered_assertion_zh_hant_hk,
                    tuple(
                        target
                        for _source, target in numeric_localisations[claim.claim_id]
                    ),
                ),
            ),
        )
    )
    claim_relative_time_relations = all(
        tuple(
            match.group(0).casefold()
            for match in _RELATIVE_TIME_FACT.finditer(source_without_localised)
        )
        == tuple(
            match.group(0).casefold()
            for match in _RELATIVE_TIME_FACT.finditer(rendered_without_localised)
        )
        for claim in package.governed_claims
        for source_without_localised, rendered_without_localised in (
            (
                _remove_exact_expressions(
                    claim.claim,
                    tuple(
                        source
                        for source, _target in claim.localised_factual_expressions
                    ),
                ),
                _remove_exact_expressions(
                    claim.rendered_assertion_zh_hant_hk,
                    tuple(
                        target
                        for _source, target in claim.localised_factual_expressions
                    ),
                ),
            ),
        )
    )
    check(
        "NUMERIC_AND_DATE_FIDELITY",
        numbers.issubset(governed_numbers)
        and draft_numeric_expressions.issubset(set(approved_numeric_expressions))
        and claim_numeric_relations
        and claim_relative_time_relations,
        "UNSUPPORTED_NUMBER_OR_DATE",
    )
    quoted = {
        match
        for pattern in (
            r'"([^"\n]+)"',
            r"“([^”\n]+)”",
            r"「([^」\n]+)」",
            r"『([^』\n]+)』",
            r"‘([^’\n]+)’",
            r"〝([^〞\n]+)〞",
            r"﹁([^﹂\n]+)﹂",
            r"❝([^❞\n]+)❞",
            r"﹃([^﹄\n]+)﹄",
            r"«([^»\n]+)»",
            r"‹([^›\n]+)›",
            r"(?<![A-Za-z])'([^'\n]+)'(?![A-Za-z])",
        )
        for match in re.findall(pattern, text)
    }
    quoted.update(_unicode_quoted_contents(text))
    check(
        "QUOTE_FIDELITY",
        _unicode_quotes_are_balanced(text)
        and all(
            any(
                value in claim.quotations
                and claim.attribution in claim.rendered_assertion_zh_hant_hk
                and any(
                    value in segment and claim.attribution in segment
                    for segment in narrative_segments
                )
                for claim in package.governed_claims
            )
            for value in quoted
        ),
        "UNSUPPORTED_OR_UNATTRIBUTED_QUOTATION",
    )
    attribution_bound = all(
        claim.attribution in claim.rendered_assertion_zh_hant_hk
        and any(
            claim.rendered_assertion_zh_hant_hk in segment
            and claim.attribution in segment
            for segment in narrative_segments
        )
        for claim in package.governed_claims
        if claim.quotations or claim.status.value == "ATTRIBUTED_CLAIM_OR_OPINION"
    )
    check(
        "ATTRIBUTION_FIDELITY",
        attribution_bound,
        "REQUIRED_ATTRIBUTION_MISSING",
    )
    resolved_record_ids = {
        record_id for record_id, _digest in package.resolved_evidence_records
    }
    check(
        "CERTAINTY_FIDELITY",
        all(
            claim.certainty == "CONFIRMED"
            and claim.semantic_relation_evidence_id in resolved_record_ids
            for claim in package.governed_claims
        ),
        "CERTAINTY_EXCEEDS_EVIDENCE",
    )
    return tuple(results)


def _provider_control_status(message: str) -> int | None:
    lowered = message.lower()
    status = re.search(
        r"\b(?:http(?:\s*(?:status|error))?|status(?:\s+code)?|api\s*error|"
        r"provider\s+error|request\s+failed|writer\s+failed|error)"
        r"\s*[:=\[(]?\s*(401|402|403|429)\b",
        lowered,
    )
    if status:
        return int(status.group(1))
    machine_status = re.search(
        r'["\']?(?:status(?:\s*_?\s*code)?|code)["\']?\s*:\s*'
        r"(401|402|403|429)\b",
        lowered,
    )
    if machine_status:
        return int(machine_status.group(1))
    if re.fullmatch(r"\s*(?:401|402|403|429)\s*", lowered):
        return int(lowered.strip())
    return None


def _failure(
    message: str,
    *,
    provider_status: int | None = None,
    usage: dict[str, object] | None = None,
) -> WriterDispatchError:
    lowered = message.lower()
    if (
        provider_status in {401, 402, 403, 429}
        or _provider_control_status(message) in {401, 402, 403, 429}
        or any(marker in lowered for marker in _SYSTEMIC_MARKERS)
    ):
        return WriterDispatchError(
            message,
            failure_class="SYSTEMIC",
            reason_code="SYSTEMIC_PROVIDER_FAILURE",
            usage=usage,
        )
    return WriterDispatchError(
        message,
        failure_class="FALLBACK_ELIGIBLE",
        reason_code="PRIMARY_OUTPUT_UNUSABLE",
        usage=usage,
    )


def _run(
    command: tuple[str, ...],
    *,
    timeout: int,
    cwd: str | None = None,
    environment: dict[str, str] | None = None,
) -> str:
    name = os.path.basename(command[0])
    deadline = timeout_deadline_after(timeout)
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=environment or unprivileged_child_environment(),
        )
    except FileNotFoundError as exc:
        raise WriterDispatchError(
            f"{name} executable not found",
            failure_class="SYSTEMIC",
            reason_code="EXECUTABLE_NOT_FOUND",
            provider_dispatched=False,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = _timeout_output_bytes(exc.output), _timeout_output_bytes(exc.stderr)
        _observe_cli_timeout_stream(stdout)
        progress = ("UNOBSERVED" if stdout is None or stderr is None
                    else "OUTPUT_OBSERVED" if stdout or stderr else "NO_OUTPUT_OBSERVED")
        evidence = timeout_diagnostic(
            boundary="CONTROLLER_DEADLINE", phase="CLI_TRANSPORT",
            cause="CONFIGURED_TIMEOUT_EXPIRED", configured_timeout_ms=round(timeout * 1000),
            elapsed_ms=max(0, round((time.monotonic() - started) * 1000)),
            deadline_at=deadline, last_progress=progress, termination="UNOBSERVED",
            process="CLI_CHILD", stdout=stdout, stderr=stderr,
        )
        raise CliTimeoutError(f"{name} writer timed out", evidence=evidence) from None
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CliProcessError(
            f"{name} writer failed: {detail}".rstrip(),
            provider_status=_provider_control_status(detail),
        )
    if not result.stdout.strip():
        raise CliEmptyOutputError("writer returned empty stdout")
    return result.stdout


def _parse_cursor_writer_output(raw: str) -> WriterCliExecution:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return WriterCliExecution(raw, unreported_cli_usage())
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), str):
        return WriterCliExecution(raw, unreported_cli_usage())
    return WriterCliExecution(
        str(payload["result"]), cursor_cli_usage(payload.get("usage"))
    )


def _grok_writer_update(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    params = value.get("params")
    if isinstance(params, dict) and isinstance(params.get("update"), dict):
        return params["update"]
    update = value.get("update")
    return update if isinstance(update, dict) else value


def _observe_cli_timeout_stream(raw: bytes | None) -> None:
    """Bounded framing observation only; completion and usage remain unknown."""
    if raw is None:
        return
    try:
        data = {"thought": [0, 0], "text": [0, 0], "usage": 0, "end": 0,
                "end_seen": False, "stop": "UNOBSERVED", "other": 0,
                "invalid": 0, "truncated": len(raw) > 131_072}
        lines = raw[:131_072].split(b"\n")
        data["truncated"] |= len(lines) > 1024
        for line in lines[:1024]:
            if not line.strip():
                continue
            if len(line) > 32_768:
                data["truncated"] = True
                continue
            try:
                update = _grok_writer_update(json.loads(line.decode("utf-8")))
                if update is None:
                    data["invalid"] += 1
                    continue
                kind = update.get("sessionUpdate") or update.get("type")
                if kind in {"thought", "agent_thought_chunk", "text", "agent_message_chunk", "assistant_message_chunk"}:
                    content = update.get("content")
                    text = (update.get("data") if kind in {"thought", "text"}
                            else content.get("text") if isinstance(content, dict) else content)
                    size = len(text.encode("utf-8")) if isinstance(text, str) else 0
                    key = "thought" if kind in {"thought", "agent_thought_chunk"} else "text"
                    data[key][0] += 1
                    data[key][1] += size
                elif kind == "usage":
                    data["usage"] += 1
                elif kind in {"end", "turn_completed", "turnEnded"}:
                    data["end"] += 1
                    data["end_seen"] = True
                    stop = update.get("stopReason") or update.get("stop_reason")
                    if stop in {"end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"}:
                        data["stop"] = stop
                    elif stop == "EndTurn":
                        data["stop"] = "end_turn"
                else:
                    data["other"] += 1
            except (UnicodeError, ValueError, RecursionError, TypeError):
                data["invalid"] += 1
        if len(json.dumps(data, separators=(",", ":")).encode()) <= 256:
            emit_diagnostic("CLI_TIMEOUT_STREAM_OBSERVATION", data)
    except Exception:
        pass  # Optional observations never replace the original timeout.


def _retain_grok_usage(current: dict[str, object], value: object) -> dict[str, object]:
    parsed = grok_cli_usage(value)
    if parsed.get("usage_basis") == "PROVIDER_REPORTED":
        return parsed
    return current


def _parse_grok_writer_output(raw: str) -> WriterCliExecution:
    chunks: list[str] = []
    usage = unreported_cli_usage()
    recognised = False
    structured_output: dict[str, object] | None = None
    for line in raw.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        update = _grok_writer_update(value)
        if update is None:
            continue
        kind = update.get("sessionUpdate") or update.get("type")
        if kind in {"agent_message_chunk", "assistant_message_chunk"}:
            content = update.get("content")
            text = content.get("text") if isinstance(content, dict) else content
            if isinstance(text, str):
                chunks.append(text)
                recognised = True
        elif kind == "text":
            data = update.get("data")
            if isinstance(data, str):
                chunks.append(data)
                recognised = True
        elif kind in {"turn_completed", "turnEnded", "usage", "end"}:
            usage = _retain_grok_usage(usage, update.get("usage"))
            candidate = update.get("structured_output")
            if isinstance(candidate, dict):
                structured_output = candidate
            recognised = True
    if recognised and structured_output is not None:
        return WriterCliExecution(
            json.dumps(structured_output, ensure_ascii=False), usage
        )
    if recognised and chunks:
        return WriterCliExecution("".join(chunks), usage)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return WriterCliExecution(raw, usage)
    if not isinstance(payload, dict):
        return WriterCliExecution(raw, usage)
    usage = _retain_grok_usage(usage, payload.get("usage"))
    structured = payload.get("structured_output")
    if isinstance(structured, dict):
        return WriterCliExecution(json.dumps(structured, ensure_ascii=False), usage)
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return WriterCliExecution(text, usage)
    return WriterCliExecution(raw, usage)


# Grok 1.0.10 still advertises the stock toolset when `--tools` is empty.
# `--disallowed-tools` removes those schemas from the model prompt.
GROK_WRITER_DISALLOWED_TOOLS = (
    "run_terminal_command",
    "run_terminal_cmd",
    "bash",
    "read_file",
    "search_replace",
    "list_dir",
    "grep",
    "kill_command_or_subagent",
    "todo_write",
    "get_command_or_subagent_output",
    "spawn_subagent",
    "scheduler_create",
    "scheduler_delete",
    "scheduler_list",
    "monitor",
    "search_tool",
    "use_tool",
    "workflow",
    "enter_plan_mode",
    "exit_plan_mode",
    "ask_user_question",
    "image_gen",
    "image_edit",
    "image_to_video",
    "reference_to_video",
    "write",
    "Agent",
)
GROK_WRITER_DISALLOWED_TOOLS_FLAG = ",".join(GROK_WRITER_DISALLOWED_TOOLS)
_GROK_WRITER_SEMANTIC_FLAGS = (
    "--prompt-file",
    "REQUEST",
    "-m",
    CONT_PRIMARY_MODEL,
    "--json-schema",
    "SCHEMA",
    "--disable-web-search",
    "--sandbox",
    "read-only",
    "--permission-mode",
    "dontAsk",
    "--tools",
    "",
    "--disallowed-tools",
    GROK_WRITER_DISALLOWED_TOOLS_FLAG,
    "--deny",
    "*",
    "--no-plan",
    "--max-turns",
    "1",
    "--no-subagents",
    "--reasoning-effort",
    CONT_PRIMARY_REASONING,
    "--system-prompt-override",
    "SYSTEM",
    "--verbatim",
    "--output-format",
    "streaming-json",
)
_CURSOR_WRITER_REQUIRED_FLAGS = (
    "--print",
    "--mode",
    "ask",
    "--output-format",
    "json",
    "--sandbox",
    "enabled",
    "--disable-tools",
    "--disable-mcp",
)
CONT_PRIMARY_COMMAND_FLAGS = _GROK_WRITER_SEMANTIC_FLAGS
CONT_FALLBACK_COMMAND_FLAGS = _CURSOR_WRITER_REQUIRED_FLAGS
CONT_DISABLED_CAPABILITIES = (
    "REPOSITORY_DISCOVERY",
    "INSTALLED_SKILLS",
    "MCP_SERVERS",
    "PLANNING",
    "PRIOR_MESSAGES",
    "SHELL_EXECUTION",
    "SUBAGENTS",
    "TOOLS",
    "WEB_SEARCH",
)


def _grok_command_flags(reasoning_effort: str = CONT_PRIMARY_REASONING, *,
                        model: str = CONT_PRIMARY_MODEL) -> tuple[str, ...]:
    """Use the declared model/effort without changing the hermetic controls."""
    return tuple(
        reasoning_effort if index and _GROK_WRITER_SEMANTIC_FLAGS[index - 1] == "--reasoning-effort"
        else model if index and _GROK_WRITER_SEMANTIC_FLAGS[index - 1] == "-m"
        else value for index, value in enumerate(_GROK_WRITER_SEMANTIC_FLAGS)
    )


def _grok_json_command(
    path: str, schema: str, system_instruction: str, *,
    reasoning_effort: str = CONT_PRIMARY_REASONING,
    model: str = CONT_PRIMARY_MODEL,
) -> tuple[str, ...]:
    replacements = {
        "REQUEST": path,
        "SCHEMA": schema,
        "SYSTEM": system_instruction,
    }
    return (
        GROK_BIN,
        *(replacements.get(value, value) for value in _grok_command_flags(reasoning_effort, model=model)),
    )


def _run_grok_json(
    prompt: str,
    *,
    schema: dict[str, object],
    system_instruction: str,
    temporary_prefix: str,
    reasoning_effort: str = CONT_PRIMARY_REASONING,
    model: str = CONT_PRIMARY_MODEL,
) -> WriterCliExecution:
    auth = _minimal_grok_auth_bytes()
    _prove_grok_hermetic_capabilities(auth)
    schema_text = canonical_json_bytes(schema).decode("utf-8")
    with tempfile.TemporaryDirectory(prefix=temporary_prefix) as root:
        workspace = _hermetic_workspace(root, binary=GROK_BIN)
        _install_minimal_grok_auth(workspace, auth)
        path = os.path.join(workspace.request, "prompt.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(prompt)
        raw = _run(
            _grok_json_command(path, schema_text, system_instruction, reasoning_effort=reasoning_effort, model=model),
            timeout=300,
            cwd=workspace.cwd,
            environment=workspace.environment,
        )
        return _parse_grok_writer_output(raw)


def run_grok_cli(prompt: str) -> WriterCliExecution:
    return _run_grok_json(
        prompt,
        schema=WRITER_SCHEMA,
        system_instruction=CONT_WRITER_SYSTEM_INSTRUCTION,
        temporary_prefix="newsroom-grok-writer-",
    )


def run_cursor_agent_cli(prompt: str) -> WriterCliExecution:
    del prompt
    with tempfile.TemporaryDirectory(prefix="newsroom-cursor-preflight-") as root:
        workspace = _hermetic_workspace(root, binary=CURSOR_AGENT_BIN)
        help_result = _run_predispatch(
            (CURSOR_AGENT_BIN, "--help"), workspace=workspace
        )
        required_controls = ("--disable-tools", "--disable-mcp")
        if help_result.returncode != 0 or not all(
            control in help_result.stdout for control in required_controls
        ):
            raise WriterDispatchError(
                "Cursor CLI cannot prove a zero-tool, zero-MCP ask",
                failure_class="SYSTEMIC",
                reason_code="HERMETIC_CAPABILITY_UNPROVABLE",
                provider_dispatched=False,
            )
    raise WriterDispatchError(
        "Cursor CLI hermetic command semantics are not qualified",
        failure_class="SYSTEMIC",
        reason_code="HERMETIC_COMMAND_VERSION_UNQUALIFIED",
        provider_dispatched=False,
    )


class CliChainWriter:
    """Primary: Grok Build CLI. Fallback: cursor-agent CLI."""

    writer_id = "grok-build-cli-cont-writer"

    def __init__(
        self,
        *,
        primary: Callable[[str], str | WriterCliExecution] | None = None,
        fallback: Callable[[str], str | WriterCliExecution] | None = None,
    ) -> None:
        self._primary = primary or run_grok_cli
        self._fallback = fallback or run_cursor_agent_cli

    def write(
        self, candidate: StoryCandidateRecord, package: EvidencePackage
    ) -> WriterCopy:
        try:
            return self.dispatch(candidate, package, route="PRIMARY")
        except WriterDispatchError as exc:
            if exc.failure_class == "SYSTEMIC":
                raise
            return self.dispatch(candidate, package, route="FALLBACK")

    def dispatch(
        self,
        candidate: StoryCandidateRecord,
        package: EvidencePackage,
        *,
        route: WriterRoute,
    ) -> WriterCopy:
        require_permitted_context(candidate, package)
        prompt = _prompt(candidate, package)
        invoke = self._primary if route == "PRIMARY" else self._fallback
        writer_id = (
            self.writer_id if route == "PRIMARY" else "cursor-agent-cli-cont-writer"
        )
        execution: WriterCliExecution | None = None
        try:
            raw = invoke(prompt)
            execution = (
                raw
                if isinstance(raw, WriterCliExecution)
                else WriterCliExecution(
                    text=raw, usage={"usage_basis": "UNREPORTED"}
                )
            )
            title, body, links = _parse_copy(execution.text)
        except WriterDispatchError:
            raise
        except CliProcessError as exc:
            raise _failure(
                str(exc),
                provider_status=exc.provider_status,
                usage=(None if execution is None else execution.usage),
            ) from exc
        except (
            RuntimeError,
            json.JSONDecodeError,
            OSError,
            subprocess.TimeoutExpired,
        ) as exc:
            raise _failure(
                str(exc), usage=(None if execution is None else execution.usage)
            ) from exc
        return WriterCopy(
            title=title,
            body=body,
            writer_id=writer_id,
            evidence_package_digest=package.digest,
            evidence_links=links,
            usage=execution.usage,
        )

    def invocation_manifest(
        self,
        candidate: StoryCandidateRecord,
        package: EvidencePackage,
        *,
        route: WriterRoute,
    ) -> WriterInvocationManifest:
        """Return the non-secret exact dispatch contract before provider I/O."""

        prompt = _prompt(candidate, package)
        prompt_bytes = prompt.encode("utf-8")
        system_bytes = CONT_WRITER_SYSTEM_INSTRUCTION.encode("utf-8")
        schema_bytes = canonical_json_bytes(WRITER_SCHEMA)
        evidence_package_bytes = canonical_json_bytes(
            _writer_evidence_value(package)
        )
        implementation_revision, implementation_worktree_clean = (
            cont_writer_implementation_identity()
        )
        if route == "PRIMARY":
            provider = CONT_PRIMARY_PROVIDER
            provider_route = CONT_PRIMARY_ROUTE
            model = CONT_PRIMARY_MODEL
            reasoning = CONT_PRIMARY_REASONING
            command_semantic_version = read_grok_command_semantic_version()
            command_flags = _GROK_WRITER_SEMANTIC_FLAGS
            config_identity = CONT_PRIMARY_CONFIG_IDENTITY
            try:
                auth_bytes = _minimal_grok_auth_bytes()
            except WriterDispatchError:
                allowed_config_digests: tuple[str, ...] = ()
            else:
                allowed_config_digests = (digest_bytes(auth_bytes),)
        else:
            provider = CONT_FALLBACK_PROVIDER
            provider_route = CONT_FALLBACK_ROUTE
            model = CONT_FALLBACK_MODEL
            reasoning = CONT_FALLBACK_REASONING
            command_semantic_version = CURSOR_COMMAND_SEMANTIC_VERSION
            command_flags = _CURSOR_WRITER_REQUIRED_FLAGS
            config_identity = CONT_FALLBACK_CONFIG_IDENTITY
            allowed_config_digests = ()
        request_digest = digest_canonical(
            {
                "provider": provider,
                "route": provider_route,
                "model": model,
                "reasoning": reasoning,
                "command_semantic_version": command_semantic_version,
                "command_flags": list(command_flags),
                "implementation_revision": implementation_revision,
                "system_digest": digest_bytes(system_bytes),
                "prompt_digest": digest_bytes(prompt_bytes),
                "output_schema_digest": CONT_WRITER_OUTPUT_SCHEMA_DIGEST,
            }
        )
        working_directory_inventory: tuple[str, ...] = ()
        disabled_capabilities = CONT_DISABLED_CAPABILITIES
        return WriterInvocationManifest.create(
            schema_version=CONT_CONTEXT_MANIFEST_SCHEMA_VERSION,
            provider=provider,
            route=provider_route,
            model=model,
            reasoning=reasoning,
            command_semantic_version=command_semantic_version,
            command_flags=command_flags,
            implementation_revision=implementation_revision,
            implementation_worktree_clean=implementation_worktree_clean,
            prompt_contract_version=CONT_WRITER_PROMPT_CONTRACT_VERSION,
            system_bytes=len(system_bytes),
            system_digest=digest_bytes(system_bytes),
            prompt_bytes=len(prompt_bytes),
            prompt_digest=digest_bytes(prompt_bytes),
            schema_bytes=len(schema_bytes),
            schema_digest=digest_bytes(schema_bytes),
            request_digest=request_digest,
            output_schema_digest=CONT_WRITER_OUTPUT_SCHEMA_DIGEST,
            context_identity=CONT_WRITER_CONTEXT_IDENTITY,
            config_identity=config_identity,
            allowed_config_digests=allowed_config_digests,
            working_directory_inventory=working_directory_inventory,
            working_directory_inventory_digest=digest_canonical(
                list(working_directory_inventory)
            ),
            disabled_capabilities=disabled_capabilities,
            evidence_package_digest=package.digest,
            evidence_package_bytes=len(evidence_package_bytes),
            one_turn=True,
            exact_input=True,
            skills_enabled=False,
            tools_enabled=False,
            mcp_enabled=False,
            prior_message_count=0,
            skill_count=0,
            tool_count=0,
            mcp_server_count=0,
            mcp_tool_count=0,
        )


def grok_cli_ready() -> bool:
    return shutil.which(GROK_BIN) is not None or os.path.isfile(GROK_BIN)


def cursor_agent_cli_ready() -> bool:
    return shutil.which(CURSOR_AGENT_BIN) is not None or os.path.isfile(
        CURSOR_AGENT_BIN
    )


def prove_grok_cli() -> None:
    result = subprocess.run(
        (GROK_BIN, "version"),
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        env=unprivileged_child_environment(),
    )
    if result.returncode != 0 or "grok" not in result.stdout.lower():
        raise RuntimeError("Grok Build CLI is not logged in or not runnable")


def probe_grok_writer_route() -> WriterRouteProbeResult:
    """Run the no-content CONT route probe against the pinned Grok model list."""

    executable_ok = grok_cli_ready()
    if not executable_ok:
        return WriterRouteProbeResult(False, False, False, False, False, None)
    try:
        result = subprocess.run(
            (GROK_BIN, "models"),
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            env=unprivileged_child_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        receipt = digest_bytes(
            canonical_json_bytes(
                {
                    "probe": "GROK_MODELS_NO_CONTENT",
                    "exception_class": type(exc).__name__,
                }
            )
        )
        return WriterRouteProbeResult(True, False, False, False, True, receipt)
    receipt = digest_bytes(
        canonical_json_bytes(
            {
                "probe": "GROK_MODELS_NO_CONTENT",
                "return_code": result.returncode,
                "stdout_digest": digest_bytes(result.stdout.encode()),
                "stderr_digest": digest_bytes(result.stderr.encode()),
            }
        )
    )
    available = result.returncode == 0
    configured = available and "grok-4.6" in result.stdout
    return WriterRouteProbeResult(
        executable_ok=True,
        authentication_ok=available,
        configuration_ok=configured,
        provider_available=available,
        provider_dispatched=True,
        provider_receipt_reference=receipt,
    )


def prove_cursor_agent_cli() -> None:
    result = subprocess.run(
        (CURSOR_AGENT_BIN, "--list-models"),
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        env=unprivileged_child_environment(),
    )
    if result.returncode != 0 or "Available models" not in result.stdout:
        raise RuntimeError("cursor-agent CLI is not logged in or not runnable")


def default_writer() -> WriterPort:
    name = os.environ.get("NEWSROOM_WRITER", "grok").strip().lower()
    if name == "fixture":
        return FixtureWriter()
    return CliChainWriter()
