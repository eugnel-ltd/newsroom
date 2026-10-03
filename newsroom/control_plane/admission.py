"""Deterministic pre-write admission for retained Evidence Packages."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane.editorial import StoryCandidateRecord
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
    EvidencePackage,
    GovernedClaimEvidence,
    GovernedClaimStatus,
    QualificationEvidence,
    _canonical_localised_fact,
    bounded_named_entities,
    rendered_named_entities,
)
from newsroom.control_plane.zh_hant import (
    ZH_HANT_HK_SHAPE_POLICY_VERSION,
    contains_discourse_filler,
    contains_non_han_letter,
    contains_simplified_variant,
)

_OLDEST_WRITE_ADMISSION_POLICY_VERSION = (
    "newsroom.write-admission.v3+newsroom.evid-012.v7+"
    "newsroom.evidence-approval.v8+newsroom.evidence-gates.v2+"
    "newsroom.governed-claim.v7+newsroom.governed-input.v10+"
    "newsroom.named-entity.v8+newsroom.cont-originality.v3+"
    "newsroom.zh-hant-hk-shape.v13"
)
_EARLIER_WRITE_ADMISSION_POLICY_VERSION = (
    "newsroom.write-admission.v4+"
    "newsroom.evid-012.v7+newsroom.evidence-approval.v8+"
    "newsroom.evidence-gates.v2+newsroom.governed-claim.v7+"
    "newsroom.governed-input.v10+newsroom.named-entity.v8+"
    "newsroom.cont-originality.v3+newsroom.zh-hant-hk-shape.v13"
)
_PREVIOUS_WRITE_ADMISSION_POLICY_VERSION = (
    "newsroom.write-admission.v5+"
    "newsroom.evid-012.v7+newsroom.evidence-approval.v8+"
    "newsroom.evidence-gates.v2+newsroom.governed-claim.v7+"
    "newsroom.governed-input.v10+newsroom.named-entity.v8+"
    "newsroom.cont-originality.v3+newsroom.zh-hant-hk-shape.v13"
)
_PENULTIMATE_WRITE_ADMISSION_POLICY_VERSION = (
    "newsroom.write-admission.v6+"
    "newsroom.evid-012.v7+newsroom.evidence-approval.v8+"
    "newsroom.evidence-gates.v2+newsroom.governed-claim.v7+"
    "newsroom.governed-input.v10+newsroom.named-entity.v8+"
    "newsroom.cont-originality.v3+newsroom.zh-hant-hk-shape.v13"
)
_EARLIEST_CURRENT_SHAPE_WRITE_ADMISSION_POLICY_VERSION = (
    "newsroom.write-admission.v7+"
    "newsroom.evid-012.v7+newsroom.evidence-approval.v8+"
    "newsroom.evidence-gates.v2+newsroom.governed-claim.v7+"
    "newsroom.governed-input.v10+newsroom.named-entity.v9+"
    "newsroom.cont-originality.v3+newsroom.zh-hant-hk-shape.v14"
)
_LATEST_LEGACY_WRITE_ADMISSION_POLICY_VERSION = (
    "newsroom.write-admission.v8+"
    "newsroom.evid-012.v7+newsroom.evidence-approval.v8+"
    "newsroom.evidence-gates.v2+newsroom.governed-claim.v7+"
    "newsroom.governed-input.v10+newsroom.named-entity.v10+"
    "newsroom.cont-originality.v3+newsroom.zh-hant-hk-shape.v14"
)
_PRIOR_V9_WRITE_ADMISSION_POLICY_VERSIONS = frozenset(
    "newsroom.write-admission.v9+newsroom.evid-012.v7+"
    "newsroom.evidence-approval.v8+newsroom.evidence-gates.v2+"
    "newsroom.governed-claim.v7+newsroom.governed-input.v10+"
    f"newsroom.named-entity.v{version}+newsroom.cont-originality.v3+"
    "newsroom.zh-hant-hk-shape.v14"
    for version in (11, 12)
) | {
    "newsroom.write-admission.v9+newsroom.evid-012.v7+"
    "newsroom.evidence-approval.v8+newsroom.evidence-gates.v2+"
    "newsroom.governed-claim.v7+newsroom.governed-input.v10+"
    "newsroom.named-entity.v13+newsroom.cont-originality.v3+"
    "newsroom.zh-hant-hk-shape.v14+newsroom.factual-localisation.v1"
} | {
    "newsroom.write-admission.v9+newsroom.evid-012.v7+"
    "newsroom.evidence-approval.v8+newsroom.evidence-gates.v2+"
    "newsroom.governed-claim.v7+newsroom.governed-input.v10+"
    f"newsroom.named-entity.v{version}+newsroom.cont-originality.v3+"
    "newsroom.zh-hant-hk-shape.v14+newsroom.factual-localisation.v1+"
    "newsroom.qualification-relation.v1"
    for version in (13, 14)
} | {
    "newsroom.write-admission.v9+newsroom.evid-012.v7+"
    "newsroom.evidence-approval.v8+newsroom.evidence-gates.v2+"
    "newsroom.governed-claim.v7+newsroom.governed-input.v10+"
    f"newsroom.named-entity.v{version}+newsroom.cont-originality.v3+"
    "newsroom.zh-hant-hk-shape.v14+newsroom.factual-localisation.v1+"
    "newsroom.qualification-relation.v2"
    for version in (14, 15)
} | {
    "newsroom.write-admission.v9+newsroom.evid-012.v7+"
    "newsroom.evidence-approval.v8+newsroom.evidence-gates.v2+"
    "newsroom.governed-claim.v7+newsroom.governed-input.v10+"
    "newsroom.named-entity.v15+newsroom.cont-originality.v3+"
    "newsroom.zh-hant-hk-shape.v14+newsroom.factual-localisation.v1+"
    "newsroom.qualification-relation.v3"
} | {
    "newsroom.write-admission.v9+newsroom.evid-012.v7+"
    "newsroom.evidence-approval.v8+newsroom.evidence-gates.v2+"
    "newsroom.governed-claim.v7+newsroom.governed-input.v10+"
    "newsroom.named-entity.v15+newsroom.cont-originality.v3+"
    "newsroom.zh-hant-hk-shape.v14+newsroom.factual-localisation.v2+"
    "newsroom.qualification-relation.v3"
}
QUALIFICATION_RELATION_POLICY_VERSION = "newsroom.qualification-relation.v3"
# Known historical identity; later consumer subpolicy changes must not relabel it.
_PRIOR_CURRENT_WRITE_ADMISSION_POLICY_VERSION = (
    "newsroom.write-admission.v9+newsroom.evid-012.v7+"
    "newsroom.evidence-approval.v8+newsroom.evidence-gates.v2+"
    "newsroom.governed-claim.v7+newsroom.governed-input.v10+"
    "newsroom.named-entity.v16+newsroom.cont-originality.v3+"
    "newsroom.zh-hant-hk-shape.v14+newsroom.factual-localisation.v2+"
    "newsroom.qualification-relation.v3"
)
WRITE_ADMISSION_POLICY_VERSION = (
    "newsroom.write-admission.v10+"
    f"{EVID_012_POLICY_VERSION}+{EVIDENCE_APPROVAL_POLICY_VERSION}+"
    f"{EVIDENCE_GATE_POLICY_VERSION}+"
    f"{GOVERNED_CLAIM_POLICY_VERSION}+{GOVERNED_INPUT_SCHEMA_VERSION}+"
    f"{NAMED_ENTITY_POLICY_VERSION}+{ORIGINALITY_POLICY_VERSION}+"
    f"{ZH_HANT_HK_SHAPE_POLICY_VERSION}+{FACTUAL_LOCALISATION_POLICY_VERSION}+"
    f"{QUALIFICATION_RELATION_POLICY_VERSION}"
)
WRITE_SELECTION_POLICY_VERSION = "newsroom.write-selection.v1"

WriteAdmissionResult = Literal["WRITE_READY", "HOLD", "REJECT"]

_REQUIRED_EVIDENCE_GATES = frozenset(
    {"CLAIM_TRACEABILITY", "EVIDENCE_SUFFICIENCY", "SOURCE_AUTHORITY"}
)
_CANONICAL_PASS_GATES = tuple(
    (gate, "PASS") for gate in sorted(_REQUIRED_EVIDENCE_GATES)
)
_APPROVED_GEOGRAPHIES = frozenset({"UK", "Hong Kong", "Global"})
_APPROVED_CATEGORIES = frozenset(
    {
        "Politics and law",
        "Immigration and status",
        "Safety and crime",
        "Weather and disasters",
        "Transport and infrastructure",
        "Health and healthcare",
        "Education and campuses",
        "Tax and welfare",
        "Work and employment",
        "Housing and local life",
        "Economy and finance",
        "Consumer rights and scams",
        "Technology and cyber security",
        "War and international affairs",
        "Community and public services",
    }
)
_QUALIFICATION_CLASSIFIER_FIELDS = frozenset(
    {
        "change_kind",
        "effect_class",
        "service_kind",
        "domain",
        "action_class",
        "importance_class",
        "event_polarity",
        "duration_relation",
        "duration_minutes",
        "change_relation",
        "effect_relation",
        "action_relation",
        "importance_relation",
    }
)


def _qualification_text_is_negative(text: str) -> bool:
    return bool(re.search(
        r"\b(?:no|not|without|never|zero|unchanged|absent|unlikely|"
        r"may|might|could|propos\w*|consider\w*|plan\w*|intend\w*|"
        r"expect\w*|forecast\w*|"
        r"fail\w*|declin\w*|refus\w*|abandon\w*|postpon\w*|shelv\w*|"
        r"defer\w*|withhold\w*|remain\w*|disput\w*|den\w*|refut\w*|"
        r"rebut\w*|investigat\w*|review\w*|assess\w*|examin\w*|study\w*|"
        r"question\w*|whether|asked?|inquir\w*|discuss\w*|alleged(?:ly)?|"
        r"reported(?:ly)?|rumou?r\w*|reports?|suggest\w*|speculat\w*|"
        r"doubt\w*|unclear|uncertain|unverified|unconfirmed|scant\s+evidence|"
        r"little\s+evidence|insufficient\s+evidence|false|incorrect|untrue|"
        r"purported(?:ly)?|supposed(?:ly)?|ostensibly|hoax|according\s+to\s+"
        r"(?:unnamed|anonymous)\s+sources?|was\s+said\s+to|"
        r"rule(?:d)?\s+out|dismiss\w*|reject\w*)\b|"
        r"\bsame\s+(?:policy|status|rules?|arrangements?|deadline|action)\b|"
        r"[未不無无沒没否非莫勿]|排除|澄清|維持|维持|拒絕|拒绝|放棄|"
        r"放弃|擱置|搁置|延後|延后|暫緩|暂缓|駁斥|驳斥|反駁|反驳|"
        r"可能|或會|或会|擬|拟|建議|建议|考慮|考虑|計劃|计划|預計|预计|"
        r"調查|调查|檢視|检视|審視|审视|研究中|查詢|查询|網傳|网传|"
        r"傳聞|传闻|傳言|传言|據報|据报|據稱|据称|聲稱|声称|疑似|似乎",
        text,
        flags=re.IGNORECASE,
    ))


def _qualification_text_is_affirmative(text: str) -> bool:
    if _qualification_text_is_negative(text):
        return False
    return bool(
        re.match(
            r"^(?:Official (?:action|deadline|policy|status) changed\b|"
            r"(?:(?:The\s+)?(?:authority|government|officials?|department|agency|"
            r"service|route|train services?|deadline|policy|law|risk|grant)\s+"
            r"(?:has\s+|have\s+|had\s+|is\s+|are\s+|was\s+|were\s+)?"
            r"(?:announc\w*|confirm\w*|chang\w*|introduc\w*|launch\w*|"
            r"extend\w*|shorten\w*|move\w*|grant\w*|revoke\w*|declar\w*|"
            r"detect\w*|issu\w*|clos\w*|disrupt\w*|delay\w*|suspend\w*|"
            r"now\b|effective\b|present\b|increas\w*|decreas\w*|"
            r"face\s+delays?))|"
            r"(?:[A-Z][A-Za-z.-]+(?:\s+[A-Z][A-Za-z.-]+){0,5})\s+"
            r"(?:announc\w*|confirm\w*|extend\w*|launch\w*|introduc\w*|"
            r"shorten\w*|revoke\w*|grant\w*)|"
            r"(?:[\u3400-\u9fff]{1,24})(?:公布|宣佈|宣布|確認|證實|证实|"
            r"推出|延長|延长|縮短|缩短|生效|批出|撤銷|撤销|發出|发出|"
            r"發現|发现|關閉|关闭|停運|停駛|停驶|中斷|中断|延誤|延误))",
            text.strip(),
            flags=re.IGNORECASE,
        )
    )


_MATERIAL_RELATION_SPAN_PATTERNS = {
    Evid012QualificationTest.ESSENTIAL_SERVICE_DISRUPTION: re.compile(
        r"closed?|closure|disrupt(?:ed|ion)?|delay(?:ed)?|suspend(?:ed|sion)?|"
        r"停運|停驶|停駛|關閉|关闭|中斷|中断|延誤|延误|暫停|暂停",
        re.IGNORECASE,
    ),
    Evid012QualificationTest.LAW_RIGHT_STATUS_POLICY: re.compile(
        r"changed|changes|new|now|introduced?|effective|granted?|revoked?|"
        r"已(?:經|经)?更?改|更?改(?:為|为|至)|新增|新|現時|现时|生效|"
        r"批出|撤銷|撤销",
        re.IGNORECASE,
    ),
    Evid012QualificationTest.SAFETY_OR_PUBLIC_HEALTH: re.compile(
        r"present|increased?|issued?|confirmed?|detected?|declared?|"
        r"存在|上升|發出|发出|證實|证实|發現|发现|宣布",
        re.IGNORECASE,
    ),
    Evid012QualificationTest.HOUSEHOLD_PRACTICAL_EFFECT: re.compile(
        r"changed|changes|new|now|increased?|decreased?|rose|fell|cut|"
        r"已(?:經|经)?更?改|更?改(?:為|为|至)|新增|新|現時|现时|上升|"
        r"下降|增加|減少|减少",
        re.IGNORECASE,
    ),
    Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE: re.compile(
        r"changed|changes|new|now|introduced?|announced?|launched?|extended?|"
        r"shortened?|moved?|effective|已(?:經|经)?更?改|更?改(?:為|为|至)|"
        r"新增|新|現時|现时|推出|公布|延長|延长|縮短|缩短|生效",
        re.IGNORECASE,
    ),
    Evid012QualificationTest.EXCEPTIONAL_PUBLIC_IMPORTANCE: re.compile(
        r"declared?|confirmed?|unprecedented|record|emergency|"
        r"宣布|證實|证实|史無前例|史无前例|紀錄|纪录|緊急|紧急",
        re.IGNORECASE,
    ),
}

_MATERIAL_SUBJECT_SPAN_PATTERNS = {
    Evid012QualificationTest.ESSENTIAL_SERVICE_DISRUPTION: re.compile(
        r"service|route|rail|train|bus|ferry|flight|road|power|water|"
        r"服務|路线|路線|鐵路|铁路|列車|列车|巴士|渡輪|渡轮|航班|道路|"
        r"電力|电力|供水",
        re.IGNORECASE,
    ),
    Evid012QualificationTest.LAW_RIGHT_STATUS_POLICY: re.compile(
        r"law|regulation|rule|policy|right|status|visa|benefit|tax|"
        r"法例|法律|規例|规例|規則|规则|政策|權利|权利|身份|簽證|签证|"
        r"福利|稅|税",
        re.IGNORECASE,
    ),
    Evid012QualificationTest.SAFETY_OR_PUBLIC_HEALTH: re.compile(
        r"safety|risk|warning|health|disease|infection|medicine|"
        r"安全|風險|风险|警告|健康|疾病|感染|藥物|药物",
        re.IGNORECASE,
    ),
    Evid012QualificationTest.HOUSEHOLD_PRACTICAL_EFFECT: re.compile(
        r"household|rent|mortgage|bill|price|money|school|childcare|"
        r"家庭|住戶|住户|租金|按揭|帳單|账单|價格|价格|金錢|金钱|學校|"
        r"学校|託兒|托儿",
        re.IGNORECASE,
    ),
    Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE: re.compile(
        r"official|authority|government|department|agency|deadline|application|"
        r"官方|當局|当局|政府|部門|部门|機構|机构|限期|截止|申請|申请|"
        r"公布|宣佈|宣布|推出|延長|延长|縮短|缩短",
        re.IGNORECASE,
    ),
    Evid012QualificationTest.EXCEPTIONAL_PUBLIC_IMPORTANCE: re.compile(
        r"emergency|unprecedented|national|constitutional|war|disaster|"
        r"緊急|紧急|史無前例|史无前例|全國|全国|憲制|宪制|戰爭|战争|"
        r"災難|灾难",
        re.IGNORECASE,
    ),
}


def _operational_replacement_is_proven(
    span: str, claim: GovernedClaimEvidence, *, source_context: str, new_state: str
) -> bool:
    # This complete sentence carries one operational contrast; appositive commas
    # are not separate claims. The ordinary clause/negation rules stay unchanged.
    sentences = {
        sentence.strip().rstrip(".")
        for text in (claim.claim, claim.supporting_excerpt)
        for sentence in re.split(r"[\n.;!?。；！？]+", text)
    }
    source_sentences = {
        sentence.strip()
        for sentence in re.split(r"[.!?。！？]+", source_context)
    }
    # A model-selected excerpt or a source line wrap must not remove a denial,
    # condition or proposal from the complete authoritative sentence.
    if (
        span.strip().rstrip(".") not in sentences
        or span.strip().rstrip(".") not in source_sentences
    ):
        return False
    match = re.fullmatch(
        r"(?:These changes|This (?:change|policy|reform)|The (?:policy|reforms?)) "
        r"replace(?:s|d)? "
        r"(?P<old>[^,.;!?\n]{1,120}) with "
        r"(?P<new>a new [^,.;!?\n]{1,120})"
        r"(?:, (?:now )?called (?P<name>[^,.;!?\n]{1,120}))?(?:, which allows "
        r"(?P<operation>[a-z]+) to take place throughout "
        r"(?P<scope>the [a-z]+(?: [a-z]+){0,5}) rather than only at the end)?\.?",
        span, flags=re.IGNORECASE,
    )
    if match is None or re.search(
        r"\b(?:unknown|unspecified|subject to|pending|if|unless)\b", span,
        flags=re.IGNORECASE,
    ):
        return False
    if new_state not in (
        span, match["new"], match["name"], span[match.start("new"):].rstrip("."),
    ):
        return False
    # The retained past-tense form names the same assessment operation in both
    # object slots without repeating the optional operational-effect clause.
    operation = match["operation"] or "assessment"
    operation_pattern = rf"\b{re.escape(operation)}\b"
    # Without an effect clause, the new object itself must name the operation:
    # a generic approach plus a new label does not establish a changed model.
    if match["operation"] is None and not re.search(
        operation_pattern, match["new"], flags=re.IGNORECASE,
    ):
        return False
    if not all(re.search(operation_pattern, label, flags=re.IGNORECASE) for label in (
        match["old"], match["name"] or match["new"],
    )):
        return False
    negative_text = span
    if operation.lower() == "assessment":
        # Only the repeated operation noun in the proved old/new/effect slots is
        # exempt from assess\w*. Assessing a proposal elsewhere remains a HOLD.
        for group in ("operation", "name", "new", "old"):
            start, end = match.span(group)
            if start >= 0:
                negative_text = (
                    negative_text[:start]
                    + re.sub(operation_pattern, "operation", negative_text[start:end], flags=re.IGNORECASE)
                    + negative_text[end:]
                )
    return not _qualification_text_is_negative(negative_text)


def _elapsed_official_deadline_is_proven(
    span: str, claim: GovernedClaimEvidence, *, source_context: str,
) -> bool:
    """Prove an elapsed deadline, not a predicted/changed date or clipped denial."""
    match = re.fullmatch(
        r"The (?:deadline|closing date) for (?P<subject>[^.;!?\n]{1,180}) "
        r"has now (?:passed|expired)\.?", span,
    )
    if match is None:
        return False
    exact = span.rstrip(".")
    claim_sentences = {
        part.strip().rstrip(".") for text in (claim.claim, claim.supporting_excerpt)
        for part in re.split(r"[.!?。！？]+", text)
    }
    if exact not in claim_sentences:
        return False
    source_sentences = [part.strip() for part in re.split(r"[.!?。！？]+", source_context)]
    containing = [index for index, part in enumerate(source_sentences) if exact in part]
    if not containing:
        return False
    subject = re.sub(r"^(?:the )?(?:[0-9]{4} )?", "", match["subject"], flags=re.IGNORECASE)
    headings = {subject.casefold(), re.sub(r" \([A-Z0-9]{2,12}\)$", "", subject).casefold()}
    for index in containing:
        sentence = source_sentences[index]
        # Canonical page text can prepend its heading without punctuation.
        # Keep that full prefix (including line wraps/semicolons) when checking
        # denial, condition and modality; an excerpt must not excise them.
        if not sentence.endswith(exact):
            return False
        prefix = sentence[:-len(exact)].strip()
        if prefix:
            lines = [line.strip().casefold() for line in prefix.splitlines() if line.strip()]
            if len(lines) > 2 or any(line not in headings for line in lines):
                return False
        for neighbour in source_sentences[max(0, index - 1):index] + source_sentences[index + 1:index + 2]:
            if re.search(r"\b(?:deadline|closing date|statement|claim|assertion|sentence)\b", neighbour, flags=re.IGNORECASE) and (
                _qualification_text_is_negative(neighbour)
                or re.search(r"\b(?:if|unless|subject to|conditional(?:ly)?|pending)\b", neighbour, flags=re.IGNORECASE)
            ):
                return False
        # This proved official form name is a noun, not a prediction. No other
        # occurrence of forecast/assessment/proposal is exempted.
        checked = re.sub(r"\bbudget forecast return\b", "return", sentence, flags=re.IGNORECASE)
        if _qualification_text_is_negative(checked) or re.search(
            r"\b(?:if|unless|subject to|conditional(?:ly)?|pending)\b", checked,
            flags=re.IGNORECASE,
        ):
            return False
    return True


def _qualification_relation_is_proven(
    qualification: QualificationEvidence, claim: GovernedClaimEvidence, *, source_context: str
) -> bool:
    spans = tuple(
        value
        for field, value in qualification.test_evidence
        if field == "material_relation_span"
        or (
            qualification.test is Evid012QualificationTest.ESSENTIAL_SERVICE_DISRUPTION
            and field == "affected_group"
        )
    )
    pattern = _MATERIAL_RELATION_SPAN_PATTERNS.get(qualification.test)
    subject_pattern = _MATERIAL_SUBJECT_SPAN_PATTERNS.get(qualification.test)
    clauses = {
        clause.strip().strip(".,，。;；!?！？")
        for clause in re.split(
            r"[\n,，.;；。!?！？]+", f"{claim.claim}\n{claim.supporting_excerpt}"
        )
        if clause.strip()
    }
    span = spans[0] if len(spans) == 1 else ""
    disruption_relation = bool(
        _MATERIAL_RELATION_SPAN_PATTERNS[
            Evid012QualificationTest.ESSENTIAL_SERVICE_DISRUPTION
        ].search(span)
    )
    qualification_fields = dict(qualification.test_evidence)
    deadline_state = (
        qualification.test is Evid012QualificationTest.LAW_RIGHT_STATUS_POLICY
        and qualification_fields.get("change_kind") == "OFFICIAL_DEADLINE"
        and qualification_fields.get("new_state") == span
    ) or (
        qualification.test is Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE
        and qualification_fields.get("action_class") == "OFFICIAL_DEADLINE"
        and qualification_fields.get("reader_action") == span
    )
    if len(spans) == 1 and deadline_state and _elapsed_official_deadline_is_proven(
        span, claim, source_context=source_context,
    ):
        return True
    authority_instruction = re.search(
        r"(?:official|authority|government|department|agency|"
        r"政府|當局|当局|部門|部门|機構|机构|署|局).{0,48}"
        r"(?:must|need(?:s)? to|required? to|should|instruct(?:ed|ion)?|"
        r"必須|必须|需要|應該|应该|改乘|改搭)",
        span,
        flags=re.IGNORECASE,
    )
    competing_attribution = re.search(
        r"(?:passengers?|users?|residents?|unions?|patients?|parents?|students?|"
        r"乘客|工會|工会|居民|市民|旅客|用戶|用户|患者|家長|家长|學生|学生)"
        r".{0,8}(?:said|says|stated|argued|believed|demanded|表示|認為|认为|"
        r"要求|聲稱|声称|稱|说|說).{0,24}"
        r"(?:must|required?|should|必須|必须|需要|應該|应该)",
        span,
        flags=re.IGNORECASE,
    )
    explicit_reader_instruction = bool(
        qualification.test is Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE
        and qualification_fields.get("action_class") == "INSTRUCTION"
        and authority_instruction
        and not competing_attribution
    )
    explicit_official_deadline = bool(
        qualification.test is Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE
        and qualification_fields.get("action_class") == "OFFICIAL_DEADLINE"
        and not re.search(
            r"reiterat(?:e|ed|es)|restate(?:d|s)?|unchanged|重申|再次確認|再次确认|"
            r"維持不變|维持不变",
            span,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"(?:deadline|closing date)\s+(?:changed|extended|shortened|moved)|"
            r"(?:changed|extended|shortened|moved)\s+(?:the\s+)?"
            r"(?:deadline|closing date)|"
            r"(?:限期|截止日期)(?:已|將|会|會)?(?:更改|改為|延長|縮短|押後|提前)|"
            r"(?:更改|延長|縮短|押後|提前)(?:申請)?(?:限期|截止日期)",
            span,
            flags=re.IGNORECASE,
        )
    )
    if (
        disruption_relation
        and qualification.test
        is not Evid012QualificationTest.ESSENTIAL_SERVICE_DISRUPTION
        and not explicit_reader_instruction
        and not explicit_official_deadline
    ):
        return False
    if (
        qualification.test is Evid012QualificationTest.LAW_RIGHT_STATUS_POLICY
        and qualification_fields.get("change_kind") == "PUBLIC_POLICY"
        and _operational_replacement_is_proven(
            span, claim, source_context=source_context,
            new_state=qualification_fields["new_state"],
        )
    ):
        return True
    if qualification.test is Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE:
        action_pattern = {
            "INSTRUCTION": re.compile(
                r"must|need(?:s)? to|required? to|should|instruct(?:ed|ion)?|"
                r"必須|必须|需要|應該|应该|指示|要求",
                re.IGNORECASE,
            ),
            "PROCESS": re.compile(
                r"process|procedure|apply|application|register|submit|form|"
                r"程序|流程|申請|申请|登記|登记|提交|表格",
                re.IGNORECASE,
            ),
            "OFFICIAL_DEADLINE": re.compile(
                r"official action|deadline|closing date|expires?|by\s+\d|"
                r"官方行動|官方行动|限期|截止|屆滿|届满|公布|宣佈|宣布",
                re.IGNORECASE,
            ),
        }[qualification_fields["action_class"]]
        if qualification_fields["reader_action"] != span or not action_pattern.search(
            span
        ):
            return False
    return (
        len(spans) == 1
        and pattern is not None
        and subject_pattern is not None
        and spans[0].strip().strip(".,，。;；!?！？") in clauses
        and _qualification_text_is_affirmative(spans[0])
        and bool(pattern.search(spans[0]))
        and bool(subject_pattern.search(spans[0]))
    )


def _duration_is_exactly_supported(
    claim: GovernedClaimEvidence, duration_minutes: str
) -> bool:
    try:
        minutes = int(duration_minutes)
    except ValueError:
        return False
    evidence_text = f"{claim.claim}\n{claim.supporting_excerpt}"
    minute_pattern = (
        rf"(?<![\d.,/]){minutes}(?!\d)\s*(?:-|–|—)?\s*"
        r"(?:minutes?|mins?|分鐘|分钟)"
    )
    duration_patterns = [minute_pattern]
    if minutes % 60 == 0:
        hours = minutes // 60
        hour_values = {str(hours)}
        for value in (
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "一",
            "二",
            "三",
            "四",
            "五",
            "六",
            "七",
            "八",
            "九",
            "十",
            "兩",
            "两",
        ):
            expression = f"{value} hours" if value.isascii() else f"{value}小時"
            if _canonical_localised_fact(expression) == (
                "DURATION_MINUTES",
                minutes,
            ):
                hour_values.add(value)
        if hours == 1:
            hour_values.add("an")
        hour_value = "|".join(sorted(hour_values, key=len, reverse=True))
        duration_patterns.append(
            rf"(?<![A-Za-z0-9.,/零〇一二三四五六七八九十百千萬万億亿兩两])"
            rf"(?:{hour_value})"
            rf"(?![A-Za-z0-9零〇一二三四五六七八九十百千萬万億亿兩两])"
            rf"\s*(?:-|–|—)?\s*(?:hours?|hrs?|小時|小时)"
        )
    duration = rf"({'|'.join(duration_patterns)})"
    english_disruption = (
        r"(?:delays?|delayed|disruption|suspend(?:ed|sion)?|clos(?:ed|ure)|"
        r"outage|interrupt(?:ed|ion)?|unavailable)"
    )
    chinese_disruption = (
        r"(?:延誤|延误|停駛|停驶|中斷|中断|暫停|暂停|關閉|关闭|停電|停电)"
    )
    chinese_duration_modifier = r"(?:長達|长达|達|达|約|约|最多)?"
    relation = re.compile(
        rf"(?:{english_disruption}\s*(?:(?:for|by|lasting|lasted|"
        rf"of(?:\s+up\s+to)?)\s+)?"
        rf"{duration}|{duration}\s*(?:-|–|—)?\s*(?:service\s+)?"
        rf"{english_disruption}|{chinese_disruption}\s*"
        rf"{chinese_duration_modifier}\s*{duration}|"
        rf"{chinese_duration_modifier}\s*{duration}\s*(?:的|嘅)?\s*"
        rf"{chinese_disruption})",
        re.IGNORECASE,
    )
    compound_prefix = re.compile(
        r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
        r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
        r"eighty|ninety|hundred|thousand|million|billion)"
        r"(?:[ -]+and)?[ -]+$|\d+\s*[.,/]\s*$",
        re.IGNORECASE,
    )
    for clause in re.split(
        r"[\n，;；。!?！？]+|(?<!\d)[.,]|[.,](?!\d)", evidence_text
    ):
        # Validate the matched quantity, not unrelated durations in the excerpt.
        if not any(
            not compound_prefix.search(clause[:match.start(index)])
            for match in relation.finditer(clause)
            for index in range(1, 5)
            if match.start(index) != -1
        ):
            continue
        polarity_text = re.sub(
            r"\b(?:no|not)\s+less\s+than\b|不少於|不少于",
            "",
            clause,
            flags=re.IGNORECASE,
        )
        if re.search(
            r"\b(?:no|not|without|never|zero|den(?:y|ies|ied|ying)|false|incorrect|"
            r"inaccurate|untrue|baseless|refut(?:e|es|ed)|disput(?:e|es|ed))\b|"
            r"\b(?:rule(?:d)?\s+out|dismiss(?:es|ed)?|reject(?:s|ed)?|"
            r"disprov(?:e|es|ed))\b|未有|沒有|没有|並無|并无|不存在|"
            r"否認|否认|不實|不实|錯誤|错误|排除|澄清",
            polarity_text,
            flags=re.IGNORECASE,
        ):
            continue
        return True
    return False


def _valid_zh_hant_hk_rendering(claim: GovernedClaimEvidence) -> bool:
    rendered = claim.rendered_assertion_zh_hant_hk
    without_entities = rendered
    for entity in claim.named_entities:
        without_entities = without_entities.replace(entity, "")
    return (
        any("\u3400" <= character <= "\u9fff" for character in rendered)
        and not contains_non_han_letter(without_entities)
        and not contains_simplified_variant(without_entities)
        and not contains_discourse_filler(rendered)
        and all(
            len(value) < 8 or value not in rendered
            for value in (claim.claim, claim.supporting_excerpt)
        )
    )


@dataclass(frozen=True, slots=True)
class WriteAdmissionDecision:
    decision_id: str
    candidate_id: str
    evidence_package_digest: str
    decision: WriteAdmissionResult
    substantive_new_information: tuple[str, ...]
    qualification_tests: tuple[str, ...]
    selection_rationale: str
    geography: tuple[str, ...]
    categories: tuple[str, ...]
    evidence_gate_results: tuple[tuple[str, str], ...]
    freshness_result: str
    integrity_result: str
    stable_reason_codes: tuple[str, ...]
    policy_version: str
    decided_at: str

    def __post_init__(self) -> None:
        if self.decision not in {"WRITE_READY", "HOLD", "REJECT"}:
            raise ValueError("invalid write-admission result")
        if self.policy_version not in {
            WRITE_ADMISSION_POLICY_VERSION,
            _LATEST_LEGACY_WRITE_ADMISSION_POLICY_VERSION,
            _EARLIEST_CURRENT_SHAPE_WRITE_ADMISSION_POLICY_VERSION,
            _PENULTIMATE_WRITE_ADMISSION_POLICY_VERSION,
            _PREVIOUS_WRITE_ADMISSION_POLICY_VERSION,
            _EARLIER_WRITE_ADMISSION_POLICY_VERSION,
            _OLDEST_WRITE_ADMISSION_POLICY_VERSION,
            *_PRIOR_V9_WRITE_ADMISSION_POLICY_VERSIONS,
            _PRIOR_CURRENT_WRITE_ADMISSION_POLICY_VERSION,
        }:
            raise ValueError("unsupported write-admission policy version")
        expected = _decision_id(
            candidate_id=self.candidate_id,
            evidence_package_digest=self.evidence_package_digest,
            decision=self.decision,
            substantive_new_information=self.substantive_new_information,
            qualification_tests=self.qualification_tests,
            selection_rationale=self.selection_rationale,
            geography=self.geography,
            categories=self.categories,
            evidence_gate_results=self.evidence_gate_results,
            freshness_result=self.freshness_result,
            integrity_result=self.integrity_result,
            stable_reason_codes=self.stable_reason_codes,
            policy_version=self.policy_version,
        )
        if self.decision_id != expected:
            raise ValueError("write-admission decision identity is not canonical")

    def as_record(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "evidence_package_digest": self.evidence_package_digest,
            "decision": self.decision,
            "substantive_new_information": list(self.substantive_new_information),
            "qualification_tests": list(self.qualification_tests),
            "selection_rationale": self.selection_rationale,
            "geography": list(self.geography),
            "categories": list(self.categories),
            "evidence_gate_results": [
                list(item) for item in self.evidence_gate_results
            ],
            "freshness_result": self.freshness_result,
            "integrity_result": self.integrity_result,
            "stable_reason_codes": list(self.stable_reason_codes),
            "policy_version": self.policy_version,
            "decided_at": self.decided_at,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> WriteAdmissionDecision:
        expected_keys = {
            "decision_id",
            "candidate_id",
            "evidence_package_digest",
            "decision",
            "substantive_new_information",
            "qualification_tests",
            "selection_rationale",
            "geography",
            "categories",
            "evidence_gate_results",
            "freshness_result",
            "integrity_result",
            "stable_reason_codes",
            "policy_version",
            "decided_at",
        }
        if set(record) != expected_keys:
            raise ValueError("write-admission record fields are not exact")
        string_fields = (
            "decision_id",
            "candidate_id",
            "evidence_package_digest",
            "decision",
            "freshness_result",
            "integrity_result",
            "policy_version",
            "decided_at",
        )
        for field in string_fields:
            value = record[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError("write-admission string fields are required")
        if not isinstance(record["selection_rationale"], str):
            raise ValueError("write-admission selection rationale must be a string")

        def string_tuple(field: str) -> tuple[str, ...]:
            value = record[field]
            if not isinstance(value, list) or any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                raise ValueError(f"write-admission {field} is invalid")
            return tuple(value)

        gate_results = record["evidence_gate_results"]
        if not isinstance(gate_results, list) or any(
            not isinstance(item, list)
            or len(item) != 2
            or any(not isinstance(value, str) or not value.strip() for value in item)
            for item in gate_results
        ):
            raise ValueError("write-admission evidence gates are invalid")
        return cls(
            decision_id=str(record["decision_id"]),
            candidate_id=str(record["candidate_id"]),
            evidence_package_digest=str(record["evidence_package_digest"]),
            decision=str(record["decision"]),  # type: ignore[arg-type]
            substantive_new_information=string_tuple("substantive_new_information"),
            qualification_tests=string_tuple("qualification_tests"),
            selection_rationale=str(record["selection_rationale"]),
            geography=string_tuple("geography"),
            categories=string_tuple("categories"),
            evidence_gate_results=tuple(
                (str(item[0]), str(item[1])) for item in gate_results
            ),
            freshness_result=str(record["freshness_result"]),
            integrity_result=str(record["integrity_result"]),
            stable_reason_codes=string_tuple("stable_reason_codes"),
            policy_version=str(record["policy_version"]),
            decided_at=str(record["decided_at"]),
        )


class WriteAdmissionPort(Protocol):
    def decide(
        self,
        candidate: StoryCandidateRecord,
        package: EvidencePackage,
        *,
        decided_at: str,
    ) -> WriteAdmissionDecision: ...


@dataclass(frozen=True, slots=True)
class WriteSelectionRecord:
    selection_id: str
    decision_id: str
    candidate_id: str
    evidence_package_digest: str
    rank: int
    quality_score: tuple[int, int, int, int]
    ordering_evidence: tuple[str, ...]
    policy_version: str
    selected_at: str

    def __post_init__(self) -> None:
        identity = {
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "evidence_package_digest": self.evidence_package_digest,
            "rank": self.rank,
            "quality_score": self.quality_score,
            "policy_version": self.policy_version,
        }
        if self.policy_version != WRITE_SELECTION_POLICY_VERSION:
            raise ValueError("unsupported write selection policy version")
        if self.selection_id != digest_bytes(canonical_json_bytes(identity)):
            raise ValueError("write selection identity does not match retained fields")
        expected_ordering = (
            f"qualification_tests={self.quality_score[0]}",
            f"claim_authority_score={self.quality_score[1]}",
            f"independent_evidential_origins={self.quality_score[2]}",
            f"substantive_new_information={self.quality_score[3]}",
        )
        if self.ordering_evidence != expected_ordering:
            raise ValueError("write selection ordering evidence does not match score")

    def as_record(self) -> dict[str, object]:
        return {
            "selection_id": self.selection_id,
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "evidence_package_digest": self.evidence_package_digest,
            "rank": self.rank,
            "quality_score": list(self.quality_score),
            "ordering_evidence": list(self.ordering_evidence),
            "policy_version": self.policy_version,
            "selected_at": self.selected_at,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> WriteSelectionRecord:
        expected_keys = {
            "selection_id",
            "decision_id",
            "candidate_id",
            "evidence_package_digest",
            "rank",
            "quality_score",
            "ordering_evidence",
            "policy_version",
            "selected_at",
        }
        if set(record) != expected_keys:
            raise ValueError("write selection record fields are not exact")
        string_fields = (
            "selection_id",
            "decision_id",
            "candidate_id",
            "evidence_package_digest",
            "policy_version",
            "selected_at",
        )
        for field in string_fields:
            value = record[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError("write selection string fields are required")
        rank = record["rank"]
        quality_score = record["quality_score"]
        ordering_evidence = record["ordering_evidence"]
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError("write selection rank is invalid")
        if (
            not isinstance(quality_score, list)
            or len(quality_score) != 4
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in quality_score
            )
        ):
            raise ValueError("write selection quality score is invalid")
        if (
            not isinstance(ordering_evidence, list)
            or len(ordering_evidence) != 4
            or any(
                not isinstance(value, str) or not value.strip()
                for value in ordering_evidence
            )
        ):
            raise ValueError("write selection ordering evidence is invalid")
        return cls(
            selection_id=str(record["selection_id"]),
            decision_id=str(record["decision_id"]),
            candidate_id=str(record["candidate_id"]),
            evidence_package_digest=str(record["evidence_package_digest"]),
            rank=rank,
            quality_score=tuple(quality_score),  # type: ignore[arg-type]
            ordering_evidence=tuple(ordering_evidence),  # type: ignore[arg-type]
            policy_version=str(record["policy_version"]),
            selected_at=str(record["selected_at"]),
        )


def _decision_id(**record: object) -> str:
    return digest_bytes(canonical_json_bytes(record))


def _make_decision(
    package: EvidencePackage,
    *,
    decision: WriteAdmissionResult,
    reason_codes: tuple[str, ...],
    decided_at: str,
) -> WriteAdmissionDecision:
    values: dict[str, object] = {
        "candidate_id": package.candidate_id,
        "evidence_package_digest": package.digest,
        "decision": decision,
        "substantive_new_information": package.substantive_new_information,
        "qualification_tests": tuple(
            sorted({item.test.value for item in package.qualification_evidence})
        ),
        "selection_rationale": package.selection_rationale,
        "geography": package.geography,
        "categories": package.categories,
        "evidence_gate_results": package.evidence_gate_results,
        "freshness_result": package.freshness_result,
        "integrity_result": package.integrity_result,
        "stable_reason_codes": reason_codes,
        "policy_version": WRITE_ADMISSION_POLICY_VERSION,
    }
    return WriteAdmissionDecision(
        decision_id=_decision_id(**values),
        decided_at=decided_at,
        **values,  # type: ignore[arg-type]
    )


class DeterministicWriteAdmission:
    """Fail-closed admission over exact, already-retained package fields."""

    def decide(
        self,
        candidate: StoryCandidateRecord,
        package: EvidencePackage,
        *,
        decided_at: str,
    ) -> WriteAdmissionDecision:
        return self.decide_candidate_identity(
            candidate_id=candidate.candidate_id,
            hypothesis_id=candidate.hypothesis_id,
            package=package,
            decided_at=decided_at,
        )

    def decide_candidate_identity(
        self,
        *,
        candidate_id: str,
        hypothesis_id: str,
        package: EvidencePackage,
        decided_at: str,
    ) -> WriteAdmissionDecision:
        """Apply the existing policy to an authenticated native identity."""
        if (candidate_id, hypothesis_id) != (
            package.candidate_id,
            package.hypothesis_id,
        ):
            raise ValueError("candidate and Evidence Package identity differ")
        if package.explicit_exclusions:
            return _make_decision(
                package,
                decision="REJECT",
                reason_codes=tuple(
                    sorted(
                        f"EXPLICIT_EXCLUSION_{item}"
                        for item in package.explicit_exclusions
                    )
                ),
                decided_at=decided_at,
            )
        missing: list[str] = []
        if not package.geography:
            missing.append("MISSING_GEOGRAPHY")
        elif not set(package.geography).issubset(_APPROVED_GEOGRAPHIES):
            missing.append("UNRECOGNISED_GEOGRAPHY")
        if not package.categories:
            missing.append("MISSING_CATEGORY")
        elif not set(package.categories).issubset(_APPROVED_CATEGORIES):
            missing.append("UNRECOGNISED_CATEGORY")
        if not package.selection_rationale.strip():
            missing.append("MISSING_SELECTION_RATIONALE")
        if not package.resolved_evidence_records:
            missing.append("UNRESOLVED_GOVERNED_EVIDENCE_RECORDS")
        elif any(
            record_id not in {item[0] for item in package.resolved_evidence_records}
            for record_id in (
                *(
                    item.qualification_record_id
                    for item in package.qualification_evidence
                ),
                *(
                    record_id
                    for claim in package.governed_claims
                    for _text, _entity_type, record_id in claim.named_entity_evidence
                ),
                *(
                    claim.semantic_relation_evidence_id
                    for claim in package.governed_claims
                ),
            )
        ):
            missing.append("UNRESOLVED_SEMANTIC_EVIDENCE")
        governed_claims = {item.claim_id: item for item in package.governed_claims}
        invalid_claims = tuple(
            item
            for item in package.governed_claims
            if item.passage_index >= len(package.passages)
            or item.claim not in package.passages[item.passage_index]
            or item.supporting_excerpt not in package.passages[item.passage_index]
            or not set(item.source_ids).issubset(package.source_ids)
            or (
                item.status is not GovernedClaimStatus.CONFIRMED_FACT
                and not (
                    item.claim_role == "CONTEXT"
                    and item.status in {
                        GovernedClaimStatus.EXPRESSLY_PROVISIONAL_FACT,
                        GovernedClaimStatus.CONTEXTUAL_BACKGROUND,
                    }
                )
            )
            or not _valid_zh_hant_hk_rendering(item)
            or any(
                entity not in item.claim
                for entity in item.named_entities
            )
            or not bounded_named_entities(
                item.claim,
                source_context=package.passages[item.passage_index],
            ) <= bounded_named_entities(
                item.supporting_excerpt,
                source_context=package.passages[item.passage_index],
            )
            or bounded_named_entities(
                item.claim,
                source_context=package.passages[item.passage_index],
            )
            != frozenset(
                (text, entity_type)
                for text, entity_type, _record_id in item.named_entity_evidence
            )
            or rendered_named_entities(
                item.rendered_assertion_zh_hant_hk,
                bounded_named_entities(
                    item.claim,
                    source_context=package.passages[item.passage_index],
                ),
            )
            != frozenset(
                (text, entity_type)
                for text, (_source, entity_type, _record_id) in zip(
                    item.rendered_named_entities,
                    item.named_entity_evidence,
                    strict=True,
                )
            )
            or any(
                quotation not in item.supporting_excerpt
                for quotation in item.quotations
            )
            or (
                item.authority_class is ClaimAuthorityClass.INDEPENDENT_RELIABLE
                and len(item.evidential_origin_ids) < 2
            )
        )
        if not governed_claims:
            missing.append("MISSING_GOVERNED_CLAIMS")
        elif invalid_claims:
            missing.append("INVALID_GOVERNED_CLAIM_EVIDENCE")
        if sum(item.claim_role == "HEADLINE" for item in package.governed_claims) != 1:
            missing.append("INVALID_HEADLINE_CLAIM_INVENTORY")
        else:
            headline_claim = next(
                item
                for item in package.governed_claims
                if item.claim_role == "HEADLINE"
            )
            qualified_claim_ids = {
                item.governed_claim_id for item in package.qualification_evidence
            }
            if package.substantive_new_information and (
                headline_claim.claim not in package.substantive_new_information
                or headline_claim.claim_id not in qualified_claim_ids
            ):
                missing.append("UNQUALIFIED_HEADLINE_CLAIM")
        if any(
            not any(
                claim.claim == fact and claim.claim_role in {"HEADLINE", "SUBSTANTIVE"}
                for claim in package.governed_claims
            )
            for fact in package.substantive_new_information
        ):
            # The independently qualified headline is itself substantive news.
            # Background must not be promoted or duplicated to fill another role.
            missing.append("INVALID_SUBSTANTIVE_CLAIM_INVENTORY")
        expected_claim_ids = frozenset(governed_claims)
        gate_evidence = {item.gate: item for item in package.evidence_gate_evidence}
        if (
            len(gate_evidence) != len(package.evidence_gate_evidence)
            or set(gate_evidence) != _REQUIRED_EVIDENCE_GATES
        ):
            missing.append("INVALID_EVIDENCE_GATE_INVENTORY")
        for gate in sorted(_REQUIRED_EVIDENCE_GATES):
            provenance = gate_evidence.get(gate)
            if (
                provenance is None
                or provenance.result != "PASS"
                or provenance.policy_version != EVIDENCE_GATE_POLICY_VERSION
                or frozenset(provenance.governed_claim_ids) != expected_claim_ids
            ):
                missing.append(f"{gate}_PROVENANCE_NOT_PASS")
        if package.evidence_gate_results != _CANONICAL_PASS_GATES:
            missing.append("EVIDENCE_GATE_RESULTS_NOT_COMPUTED")
        if package.freshness_result != "PASS":
            missing.append("FRESHNESS_NOT_PASS")
        if package.integrity_result != "PASS":
            missing.append("INTEGRITY_NOT_PASS")
        if missing:
            return _make_decision(
                package,
                decision="HOLD",
                reason_codes=tuple(sorted(set(missing))),
                decided_at=decided_at,
            )
        if not package.substantive_new_information:
            return _make_decision(
                package,
                decision="REJECT",
                reason_codes=("NO_SUBSTANTIVE_NEW_INFORMATION",),
                decided_at=decided_at,
            )
        if any(
            fact not in {item.claim for item in package.governed_claims}
            for fact in package.substantive_new_information
        ):
            return _make_decision(
                package,
                decision="HOLD",
                reason_codes=("SUBSTANTIVE_INFORMATION_NOT_EXACT",),
                decided_at=decided_at,
            )
        if not package.qualification_evidence:
            return _make_decision(
                package,
                decision="REJECT",
                reason_codes=("NO_QUALIFICATION_TEST",),
                decided_at=decided_at,
            )
        unsupported = tuple(
            item
            for item in package.qualification_evidence
            if item.governed_claim_id not in governed_claims
            or not _qualification_relation_is_proven(
                item, governed_claims[item.governed_claim_id],
                source_context=package.passages[
                    governed_claims[item.governed_claim_id].passage_index
                ],
            )
            or any(
                field not in _QUALIFICATION_CLASSIFIER_FIELDS
                and value not in governed_claims[item.governed_claim_id].claim
                and value
                not in governed_claims[item.governed_claim_id].supporting_excerpt
                for field, value in item.test_evidence
            )
        )
        unsupported_duration = tuple(
            item
            for item in package.qualification_evidence
            if item.test.value == "ESSENTIAL_SERVICE_DISRUPTION"
            and (
                item.governed_claim_id not in governed_claims
                or not _duration_is_exactly_supported(
                    governed_claims[item.governed_claim_id],
                    dict(item.test_evidence)["duration_minutes"],
                )
            )
        )
        if unsupported or unsupported_duration:
            return _make_decision(
                package,
                decision="HOLD",
                reason_codes=("QUALIFICATION_EVIDENCE_NOT_EXACT",),
                decided_at=decided_at,
            )
        return _make_decision(
            package,
            decision="WRITE_READY",
            reason_codes=("QUALIFIED_WRITE_READY",),
            decided_at=decided_at,
        )


def validate_admission_binding(
    decision: WriteAdmissionDecision,
    candidate: StoryCandidateRecord,
    package: EvidencePackage,
) -> None:
    if decision.candidate_id != candidate.candidate_id:
        raise ValueError("write admission binds another candidate")
    if decision.evidence_package_digest != package.digest:
        raise ValueError("write admission binds another Evidence Package")
    if decision.substantive_new_information != package.substantive_new_information:
        raise ValueError("write admission substantive facts differ from package")
    if decision.qualification_tests != tuple(
        sorted({item.test.value for item in package.qualification_evidence})
    ):
        raise ValueError("write admission qualification differs from package")
    if (
        decision.selection_rationale,
        decision.geography,
        decision.categories,
        decision.evidence_gate_results,
        decision.freshness_result,
        decision.integrity_result,
    ) != (
        package.selection_rationale,
        package.geography,
        package.categories,
        package.evidence_gate_results,
        package.freshness_result,
        package.integrity_result,
    ):
        raise ValueError("write admission governed fields differ from package")


def select_write_ready(
    admitted: tuple[
        tuple[StoryCandidateRecord, EvidencePackage, WriteAdmissionDecision], ...
    ],
    *,
    limit: int,
    selected_at: str,
) -> tuple[
    tuple[
        StoryCandidateRecord,
        EvidencePackage,
        WriteAdmissionDecision,
        WriteSelectionRecord,
    ],
    ...,
]:
    """Select by retained evidence quality; candidate ID is only a final tie-break."""

    if limit < 0:
        raise ValueError("write-ready selection limit must be non-negative")
    if any(
        decision.policy_version != WRITE_ADMISSION_POLICY_VERSION
        for _candidate, _package, decision in admitted
    ):
        raise ValueError("write admission policy is not current")

    def quality(
        item: tuple[StoryCandidateRecord, EvidencePackage, WriteAdmissionDecision],
    ) -> tuple[int, int, int, int]:
        _candidate, package, _decision = item
        origins = {
            origin
            for claim in package.governed_claims
            for origin in claim.evidential_origin_ids
        }
        authority_score = sum(
            2 if claim.authority_class is ClaimAuthorityClass.RESPONSIBLE_PRIMARY else 1
            for claim in package.governed_claims
        )
        return (
            len(package.qualification_evidence),
            authority_score,
            len(origins),
            len(package.substantive_new_information),
        )

    ready = [item for item in admitted if item[2].decision == "WRITE_READY"]
    ready.sort(
        key=lambda item: (
            tuple(-value for value in quality(item)) + (item[0].candidate_id,)
        )
    )
    selected = []
    for rank, item in enumerate(ready[:limit], start=1):
        candidate, package, decision = item
        score = quality(item)
        identity = {
            "decision_id": decision.decision_id,
            "candidate_id": candidate.candidate_id,
            "evidence_package_digest": package.digest,
            "rank": rank,
            "quality_score": score,
            "policy_version": WRITE_SELECTION_POLICY_VERSION,
        }
        record = WriteSelectionRecord(
            selection_id=digest_bytes(canonical_json_bytes(identity)),
            decision_id=decision.decision_id,
            candidate_id=candidate.candidate_id,
            evidence_package_digest=package.digest,
            rank=rank,
            quality_score=score,
            ordering_evidence=(
                f"qualification_tests={score[0]}",
                f"claim_authority_score={score[1]}",
                f"independent_evidential_origins={score[2]}",
                f"substantive_new_information={score[3]}",
            ),
            policy_version=WRITE_SELECTION_POLICY_VERSION,
            selected_at=selected_at,
        )
        selected.append((*item, record))
    return tuple(selected)


def write_admission_revalidation_due(facts: dict) -> bool:
    """Reconsider only a retained inventory HOLD under the changed consumer."""
    return (facts.get("reason") == "INVALID_SUBSTANTIVE_CLAIM_INVENTORY"
            and facts.get("write_admission_policy_version") != WRITE_ADMISSION_POLICY_VERSION
            and type(facts.get("package_admission_id")) is str
            and isinstance(facts.get("editorial_decision"), dict))
