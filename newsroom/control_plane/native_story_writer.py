"""Natural native copy with a distinct, source-bound review; no model or store IO."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import re

from jsonschema import ValidationError, validate

from newsroom.authority.canonical import canonical_json_bytes, digest_canonical
from .evidence import EvidencePackage, bounded_named_entities
from .native_story_entities import story_entity_names_are_bound, declared_source_speakers_are_retained
from .writer import WriterCopy, WriterEvidenceLink, WriterValidatorResult, _writer_evidence_value, _writer_numeric_localisations
from .zh_hant import contains_discourse_filler, contains_non_han_letter, contains_simplified_variant

WRITER_ID = "newsroom.native-story-writer.v1"
CONSUMER_VERSION = "newsroom.native-story-support.v3"
LEGACY_DRAFT_SYSTEM = (
    "Write original Hong Kong Traditional Chinese news, not claim declarations. Use only approved facts "
    "and supporting source windows: a natural headline, attributed lead, detail and available context. "
    "Preserve numbers, entities, attribution, quotations and provisional/uncertain meaning. "
    "Map verbatim draft spans to approved claim IDs. Rich material deserves coherent paragraphs; "
    "sparse warnings may be BRIEF, never padded to a word quota. No external knowledge or planning prose."
)
DRAFT_SYSTEM = LEGACY_DRAFT_SYSTEM + (
    " Write reader-facing official actions, not decoder or classifier preambles such as 'official status changed'. "
    "When approved issue and update timestamps are exactly equal, state the time once and combine the actions "
    "without losing either approved fact. One verbatim draft span may map to several approved claim IDs. "
    "Keep exact approved dates and factual meaning; do not introduce relative time such as 'tonight' unless approved."
)
REVIEW_SYSTEM = (
    "You are the separate source-support reviewer, not the writer. Review the exact draft against the "
    "approved claims and immutable source windows, ignoring the writer's own assurances. Every supplied "
    "sentence, including the headline, needs approved claim support. Check numbers, entities, modality "
    "and quotations, and cover all required claims. Return HOLD or UNKNOWN for uncertainty or missing "
    "support; PASS means source support, not an editorial quality grade. Echo both exact digests."
)
_TEXT = {"type": "string", "minLength": 1, "maxLength": 12000}
_IDS = {"type": "array", "minItems": 1, "uniqueItems": True, "items": _TEXT}
_FACTS = ("numbers", "entities", "modality", "quotations")
DRAFT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["title", "body", "format", "evidence_links"],
    "properties": {"title": _TEXT, "body": _TEXT, "format": {"enum": ["ARTICLE", "BRIEF"]},
        "evidence_links": {"type": "array", "minItems": 1, "maxItems": 200, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["governed_claim_id", "rendered_assertion"],
            "properties": {"governed_claim_id": _TEXT, "rendered_assertion": _TEXT}}}},
}
REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["source_package_digest", "draft_digest", "verdict", "covered_claim_ids", "sentence_support", "factual_checks"],
    "properties": {"source_package_digest": _TEXT, "draft_digest": _TEXT,
        "verdict": {"enum": ["PASS", "HOLD", "UNKNOWN"]}, "covered_claim_ids": _IDS,
        "sentence_support": {"type": "array", "minItems": 1, "maxItems": 200, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["sentence_index", "claim_ids", "verdict"],
            "properties": {"sentence_index": {"type": "integer", "minimum": 0}, "claim_ids": _IDS,
                "verdict": {"enum": ["SUPPORTED", "UNSUPPORTED", "UNKNOWN"]}}}},
        "factual_checks": {"type": "object", "additionalProperties": False, "required": list(_FACTS),
            "properties": {key: {"enum": ["PASS", "FAIL", "UNKNOWN"]} for key in _FACTS}}},
}


class NativeStoryWriterHold(ValueError):
    """Unproved copy is held; transport usage remains the caller's responsibility."""
    def __init__(self, reason, *, failures=()):
        super().__init__(reason)
        self.stable_reason_codes=tuple(failures)or(reason,)


@dataclass(frozen=True)
class SourceSupportReview:
    canonical_record: bytes

    def as_record(self) -> dict:
        return json.loads(self.canonical_record)


@dataclass(frozen=True)
class NativeStoryResult:
    copy: WriterCopy
    review: SourceSupportReview
    validators: tuple[WriterValidatorResult, ...]
    format: str


def _object(value, schema):
    result = json.loads(value) if isinstance(value, (str, bytes)) else value
    validate(result, schema)
    return json.loads(canonical_json_bytes(result))


def _draft(copy, format):
    return {"title": copy.title, "body": copy.body, "format": format,
            "evidence_links": [{"governed_claim_id": link.governed_claim_id,
                                "rendered_assertion": link.rendered_assertion} for link in copy.evidence_links]}


def _sentences(copy):
    return [copy.title, *(part.strip() for part in re.findall(r"[^。！？!?\n]+(?:[。！？!?][」』”’\"]*)?", copy.body) if part.strip())]


def _sentence_claim_links(copy):
    """Intersect verbatim draft evidence spans with each body sentence."""
    spans=[]
    for link in copy.evidence_links:
        start=0
        while (start:=copy.body.find(link.rendered_assertion,start))>=0:
            spans.append((start,start+len(link.rendered_assertion),link.governed_claim_id))
            start+=1
    return [
        {identity for start,end,identity in spans
         if start<match.end() and end>match.start()
         and re.search(r'[\w\u3400-\u9fff]',copy.body[max(start,match.start()):min(end,match.end())])}
        for match in re.finditer(r"[^。！？!?\n]+(?:[。！？!?][」』”’\"]*)?",copy.body)
        if match.group().strip()
    ]


def validate_retained_story(copy: WriterCopy, package: EvidencePackage, review_record: Mapping, format: str,
                            *, source_currentness=(), source_records=()):
    """Validate retained bindings without calling either model again."""
    if 'date_derivation' in review_record:
        from .native_story_dates import derive_and_verify
        if copy.writer_id!=WRITER_ID or copy.evidence_package_digest!=package.digest:
            return (WriterValidatorResult('NATIVE_STORY_PACKAGE_BINDING','FAIL','SOURCE_BINDING_DIFFERS'),)
        try:
            proof=review_record['date_derivation']
            original=_object(proof['original_draft'],DRAFT_SCHEMA)
            original_copy=WriterCopy(original['title'],original['body'],WRITER_ID,package.digest,
                tuple(WriterEvidenceLink(**link)for link in original['evidence_links']))
            review={key:review_record[key]for key in REVIEW_SCHEMA['required']}
            checks=validate_retained_story(original_copy,package,review,original['format'])
            if any(check.result!='PASS'for check in checks):
                return checks
            derive_and_verify(original,review,package,source_currentness,
                source_records=source_records,
                final_draft=_draft(copy,format),date_derivation=proof)
        except (KeyError,TypeError,ValueError,ValidationError):
            return (WriterValidatorResult('NATIVE_STORY_DATE_DERIVATION','FAIL','UNPROVEN_SOURCE_DATE'),)
        return (*checks,WriterValidatorResult('NATIVE_STORY_DATE_DERIVATION','PASS','SOURCE_BOUND_DATE'))
    checks = []

    def check(name, passed):
        checks.append(WriterValidatorResult(name, "PASS" if passed else "FAIL", name if not passed else "SOURCE_BOUND"))

    try:
        _object(_draft(copy, format), DRAFT_SCHEMA)
        review = _object({key: review_record.get(key) for key in REVIEW_SCHEMA["required"]}, REVIEW_SCHEMA)
    except (ValueError, TypeError, ValidationError, AttributeError):
        check("NATIVE_STORY_OUTPUT_CONTRACT", False)
        return tuple(checks)
    claims = {claim.claim_id: claim for claim in package.governed_claims}
    sentences, text = _sentences(copy), copy.title + "\n" + copy.body
    sentence_links=_sentence_claim_links(copy)
    links = copy.evidence_links
    support = review["sentence_support"]
    headline_ids = {identity for identity, claim in claims.items() if claim.claim_role == "HEADLINE"}
    reviewed_headline = next((item for item in support if item["sentence_index"] == 0), None)
    headline_supported = (reviewed_headline is not None
                          and reviewed_headline["verdict"] == "SUPPORTED"
                          and bool(set(reviewed_headline["claim_ids"]) & headline_ids))
    check("NATIVE_STORY_PACKAGE_BINDING", copy.writer_id == WRITER_ID and copy.evidence_package_digest == package.digest
          and review["source_package_digest"] == package.digest)
    check("NATIVE_STORY_DRAFT_BINDING", review["draft_digest"] == digest_canonical(_draft(copy, format)))
    check("NATIVE_STORY_CLAIM_COVERAGE", bool(claims) and {link.governed_claim_id for link in links} == set(claims)
          and set(review["covered_claim_ids"]) == set(claims)
          and all(link.rendered_assertion in text for link in links)
          and all((headline_supported and identity in reviewed_headline["claim_ids"])
                  if claim.claim_role == "HEADLINE" else
                  any(link.governed_claim_id == identity and link.rendered_assertion in copy.body for link in links)
                  for identity, claim in claims.items()))
    check("NATIVE_STORY_SENTENCE_SUPPORT", [item["sentence_index"] for item in support] == list(range(len(sentences)))
          and all(item["verdict"] == "SUPPORTED" and set(item["claim_ids"]) <= set(claims) for item in support)
          and all(headline_supported if item["sentence_index"] == 0 else
                  bool(set(item["claim_ids"]) & sentence_links[item["sentence_index"]-1])
                  for item in support))
    check("NATIVE_STORY_SEPARATE_REVIEW", review["verdict"] == "PASS" and all(review["factual_checks"][key] == "PASS" for key in _FACTS))
    number = re.compile(r"\d+(?:[.,]\d+)*|[零〇一二三四五六七八九十百千萬億兆兩廿卅]+(?:年|月|日|時|分|秒|人|名|個|間|所|座|公里|元|英鎊|%|％)")
    approved = "\n".join(value for claim in claims.values() for value in
                         (claim.claim, claim.rendered_assertion_zh_hant_hk, *(target for _, target in _writer_numeric_localisations(claim))))
    check("NATIVE_STORY_FACTUAL_NUMBERS", set(number.findall(text)) <= set(number.findall(approved)))
    entities = {name for claim in claims.values() for name in (*claim.named_entities, *claim.rendered_named_entities)}
    check("NATIVE_STORY_FACTUAL_ENTITIES", story_entity_names_are_bound(text, claims.values()))
    check("NATIVE_STORY_SOURCE_SPEAKERS", declared_source_speakers_are_retained(claims.values(), links))
    modality = ("可能", "或會", "預計", "預料", "暫定", "初步", "尚未", "未確定", "至少", "最多")
    check("NATIVE_STORY_FACTUAL_MODALITY", all(not any(word in claim.rendered_assertion_zh_hant_hk for word in modality)
          or any(word in "\n".join(link.rendered_assertion for link in links if link.governed_claim_id == identity) for word in modality)
          for identity, claim in claims.items()))
    quotes = lambda value: set(re.findall(r"[「“\"]([^」”\"]+)[」”\"]", value))
    check("NATIVE_STORY_FACTUAL_QUOTATIONS", quotes(text) <= quotes(approved) | {quote for claim in claims.values() for quote in claim.quotations})
    language = text
    for entity in entities:
        language = language.replace(entity, "")
    check("NATIVE_STORY_NARRATIVE", any("\u3400" <= character <= "\u9fff" for character in text)
          and not contains_simplified_variant(language) and not contains_non_han_letter(language)
          and not contains_discourse_filler(text) and "已核實證據報道" not in text)
    return tuple(checks)


def write_native_story(package: EvidencePackage, *, generate: Callable, review: Callable,
                       source_currentness=(), source_records=()) -> NativeStoryResult:
    if generate is review:
        raise NativeStoryWriterHold("NATIVE_STORY_SEPARATE_REVIEW_REQUIRED")
    evidence = _writer_evidence_value(package)
    evidence.pop("passages")
    evidence.pop("permitted_admitted_structured_context")
    for item, claim in zip(evidence["approved_governed_claims"], package.governed_claims, strict=True):
        item.update(source_ids=list(claim.source_ids), passage_index=claim.passage_index,
                    localised_factual_expressions=[list(pair) for pair in claim.localised_factual_expressions])
    request = {"source_package_digest": package.digest, "evidence": evidence}
    generated = generate(json.loads(canonical_json_bytes(request)))
    try:
        draft = _object(generated, DRAFT_SCHEMA)
        copy = WriterCopy(draft["title"], draft["body"], WRITER_ID, package.digest,
                          tuple(WriterEvidenceLink(**link) for link in draft["evidence_links"]))
        request.update(draft=draft, draft_digest=digest_canonical(draft), sentences=_sentences(copy))
    except (ValueError, TypeError, ValidationError) as exc:
        raise NativeStoryWriterHold("NATIVE_STORY_OUTPUT_CONTRACT_HOLD") from exc
    reviewed = review(json.loads(canonical_json_bytes(request)))
    try:
        record = _object(reviewed, REVIEW_SCHEMA)
    except (ValueError, TypeError, ValidationError) as exc:
        raise NativeStoryWriterHold("NATIVE_STORY_OUTPUT_CONTRACT_HOLD") from exc
    validators = validate_retained_story(copy, package, record, draft["format"])
    failures=tuple(check.validator for check in validators if check.result!='PASS')
    if failures:
        raise NativeStoryWriterHold(failures[0],failures=failures)
    from .native_story_dates import derive_and_verify
    final,proof=derive_and_verify(draft,record,package,source_currentness,source_records=source_records)
    if proof is not None:
        copy=WriterCopy(final['title'],final['body'],WRITER_ID,package.digest,
            tuple(WriterEvidenceLink(**link)for link in final['evidence_links']))
        record={**record,'date_derivation':proof}
        validators=validate_retained_story(copy,package,record,final['format'],
            source_currentness=source_currentness,source_records=source_records)
        if any(check.result!='PASS'for check in validators):
            raise NativeStoryWriterHold('NATIVE_STORY_DATE_DERIVATION')
    return NativeStoryResult(copy, SourceSupportReview(canonical_json_bytes(record)), validators, draft["format"])
