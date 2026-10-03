"""Pure, lossless source references for the native assessor provider boundary."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.control_plane.admission import _QUALIFICATION_CLASSIFIER_FIELDS
from newsroom.control_plane.evidence import _entity_pattern, bounded_named_entities, NAMED_ENTITY_POLICY_VERSION_V15

VERSION = "newsroom.native-assessor-references.v1"
MAX_RESULT_BYTES = 256 * 1024
MAX_SOURCES = 32
MAX_SEGMENTS = 8192
MAX_CLAIMS = 32
MAX_FRAGMENT_LENGTH = 4096
_QUALIFICATION_WITNESSES = {
    "LAW_RIGHT_STATUS_POLICY": {"change_kind", "event_polarity", "change_relation", "material_relation_span", "new_state"},
    "SAFETY_OR_PUBLIC_HEALTH": {"effect_class", "event_polarity", "effect_relation", "material_relation_span", "affected_group"},
    "ESSENTIAL_SERVICE_DISRUPTION": {"service_kind", "event_polarity", "duration_relation", "duration_minutes", "affected_group"},
    "HOUSEHOLD_PRACTICAL_EFFECT": {"domain", "event_polarity", "effect_relation", "material_relation_span", "practical_effect"},
    "OFFICIAL_ACTION_OR_DEADLINE": {"action_class", "event_polarity", "action_relation", "material_relation_span", "reader_action"},
    "EXCEPTIONAL_PUBLIC_IMPORTANCE": {"importance_class", "event_polarity", "importance_relation", "material_relation_span", "affected_group"},
}
_SPAN_ID = re.compile(r"S([1-9][0-9]*)L([1-9][0-9]*)\Z")
_LINE_SEPARATORS = ("\r\n", "\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")
_RANGE = {
    "type": "object", "properties": {
        "first_span_id": {"type": "string", "pattern": r"^S[1-9][0-9]*L[1-9][0-9]*$"},
        "last_span_id": {"type": "string", "pattern": r"^S[1-9][0-9]*L[1-9][0-9]*$"},
    }, "required": ["first_span_id", "last_span_id"], "additionalProperties": False,
}


class SourceReferenceError(ValueError):
    """A wire reference differs from the immutable acquired source view."""

    reason_code = "ASSESSOR_SOURCE_REFERENCE_HOLD"


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise SourceReferenceError("duplicate reference result key")
        value[key] = item
    return value


@dataclass(frozen=True, slots=True)
class SourceSegment:
    span_id: str
    source_id: str
    passage_index: int
    ordinal: int
    text: str
    start_byte: int
    content_end_byte: int
    end_byte: int
    digest: str
    entities: tuple[tuple[str, str], ...]

    def request_record(self) -> dict[str, object]:
        return {"span_id": self.span_id, "source_id": self.source_id,
                "text": self.text, "entities": [list(item) for item in self.entities]}

    def manifest_record(self) -> dict[str, object]:
        return {"span_id": self.span_id, "source_id": self.source_id,
                "passage_index": self.passage_index, "ordinal": self.ordinal,
                "start_byte": self.start_byte, "content_end_byte": self.content_end_byte,
                "end_byte": self.end_byte,
                "text_digest": self.digest, "entities": [list(item) for item in self.entities]}


@dataclass(frozen=True, slots=True)
class SourceView:
    passages: tuple[str, ...]
    source_ids: tuple[str, ...]
    segments: tuple[SourceSegment, ...]
    body_digests: tuple[str, ...]
    source_entities: tuple[tuple[tuple[str, str], ...], ...]
    manifest_digest: str
    entity_policy_version: str = NAMED_ENTITY_POLICY_VERSION_V15

    @property
    def request_segments(self) -> tuple[dict[str, object], ...]:
        return tuple(segment.request_record() for segment in self.segments)

    @property
    def manifest(self) -> dict[str, object]:
        return _manifest(self.source_ids, self.body_digests, self.segments, policy_version=self.entity_policy_version)

    def resolve_range(self, reference: object) -> tuple[str, int, str]:
        if (type(reference) is not dict or set(reference) != {"first_span_id", "last_span_id"}
                or any(type(value) is not str or _SPAN_ID.fullmatch(value) is None
                       for value in reference.values())):
            raise SourceReferenceError("source range shape differs")
        by_id = {segment.span_id: index for index, segment in enumerate(self.segments)}
        first = by_id.get(reference["first_span_id"])
        last = by_id.get(reference["last_span_id"])
        if first is None or last is None or first > last:
            raise SourceReferenceError("source range identity or order differs")
        selected = self.segments[first:last + 1]
        if any(item.passage_index != selected[0].passage_index for item in selected):
            raise SourceReferenceError("source range crosses an acquired body")
        if tuple(item.ordinal for item in selected) != tuple(range(
            selected[0].ordinal, selected[-1].ordinal + 1,
        )):
            raise SourceReferenceError("source range is not contiguous")
        body = self.passages[selected[0].passage_index]
        encoded = body.encode("utf-8")
        if encoded[selected[0].start_byte:selected[-1].end_byte] != "".join(
            item.text for item in selected
        ).encode("utf-8"):
            raise SourceReferenceError("source range bytes differ")
        # Preserve every internal separator, but the selected last line's
        # separator is framing, not part of its literal content.
        text = encoded[selected[0].start_byte:selected[-1].content_end_byte].decode("utf-8")
        return text, selected[0].passage_index, selected[0].source_id


def _manifest(source_ids: tuple[str, ...], body_digests: tuple[str, ...],
              segments: tuple[SourceSegment, ...], *, policy_version: str = NAMED_ENTITY_POLICY_VERSION_V15) -> dict[str, object]:
    return {"version": VERSION, "source_ids": list(source_ids),
            "body_digests": list(body_digests),
            "segments": [segment.manifest_record() for segment in segments],
            **({"entity_policy_version": policy_version} if policy_version != NAMED_ENTITY_POLICY_VERSION_V15 else {})}


def build_source_view(passages: tuple[str, ...], source_ids: tuple[str, ...], *, policy_version: str = NAMED_ENTITY_POLICY_VERSION_V15) -> SourceView:
    if (type(passages) is not tuple or type(source_ids) is not tuple
            or not 0 < len(passages) == len(source_ids) <= MAX_SOURCES
            or len(set(source_ids)) != len(source_ids)
            or any(type(item) is not str or not item for item in (*passages, *source_ids))):
        raise SourceReferenceError("source view identity differs")
    segments = []
    body_digests = []
    all_entities = []
    for passage_index, (body, source_id) in enumerate(zip(passages, source_ids, strict=True)):
        body_bytes = body.encode("utf-8")
        if len(body_bytes) > 1_048_576:
            raise SourceReferenceError("source body exceeds acquired content bound")
        body_digests.append(digest_bytes(body_bytes))
        seen_entities: set[tuple[str, str]] = set()
        body_entities: list[tuple[str, str]] = []
        offset = 0
        for ordinal, text in enumerate(body.splitlines(keepends=True), 1):
            if len(segments) >= MAX_SEGMENTS:
                raise SourceReferenceError("source segment count exceeds bound")
            raw = text.encode("utf-8")
            separator = next((ending for ending in _LINE_SEPARATORS if text.endswith(ending)), "")
            content_end = offset + len(text[:-len(separator)].encode("utf-8")) if separator else offset + len(raw)
            found = _claim_entities(text, body, policy_version=policy_version)
            for item in found:
                if item not in seen_entities:
                    seen_entities.add(item)
                    body_entities.append(item)
            segments.append(SourceSegment(
                f"S{passage_index + 1}L{ordinal}", source_id, passage_index,
                ordinal, text, offset, content_end, offset + len(raw), digest_bytes(raw), found,
            ))
            offset += len(raw)
        if offset != len(body_bytes):
            raise SourceReferenceError("source segmentation lost exact bytes")
        all_entities.append(tuple(body_entities))
    exact = tuple(segments)
    digests = tuple(body_digests)
    manifest = _manifest(source_ids, digests, exact, policy_version=policy_version)
    return SourceView(passages, source_ids, exact, digests, tuple(all_entities),
                      digest_canonical(manifest), policy_version)


def make_provider_schema(existing_internal_schema: dict[str, object]) -> dict[str, object]:
    """Derive one closed static wire schema from the retained internal contract."""
    schema = json.loads(canonical_json_bytes(existing_internal_schema))
    package = schema["properties"]["package"]
    properties = package["properties"]
    claim = properties["governed_claims"]["items"]
    claim_props = claim["properties"]
    for key in ("claim_id", "claim", "passage_index", "supporting_excerpt", "source_ids",
                "rendered_assertion_zh_hant_hk"):
        claim_props.pop(key)
    claim_props.update({
        "claim_range": _RANGE, "support_range": _RANGE,
        "rendered_fragments": {"type": "array", "items": {"type": "string", "maxLength": MAX_FRAGMENT_LENGTH},
                               "minItems": 1, "maxItems": 65},
    })
    claim_props["localised_factual_expressions"]["maxItems"] = 32
    claim_props["localised_factual_expressions"]["items"]["items"]["maxLength"] = 256
    claim_props["quotations"]["maxItems"] = 32
    claim_props["quotations"]["items"]["maxLength"] = 256
    claim["required"] = list(claim_props)
    properties["governed_claims"]["maxItems"] = MAX_CLAIMS
    properties["substantive_new_information"] = {
        "type": "array", "items": {"type": "integer", "minimum": 0, "maximum": MAX_CLAIMS - 1},
        "maxItems": MAX_CLAIMS,
    }
    properties["substantive_claim_indexes"] = properties.pop("substantive_new_information")
    for qualification in properties["qualification_evidence"]["items"]["oneOf"]:
        qualification["properties"].pop("governed_claim_id")
        qualification["properties"]["claim_index"] = {
            "type": "integer", "minimum": 0, "maximum": MAX_CLAIMS - 1,
        }
        qualification["required"] = list(qualification["properties"])
        for key, value in qualification["properties"]["test_evidence"]["properties"].items():
            if value.get("type") == "string" and "const" not in value and "enum" not in value:
                value["maxLength"] = 256
    properties["qualification_evidence"]["maxItems"] = 32
    properties["selection_rationale"]["maxLength"] = 4096
    for key in ("geography", "categories", "explicit_exclusions"):
        properties[key]["maxItems"] = 32
    properties["explicit_exclusions"]["items"]["maxLength"] = 256
    package["required"] = list(properties)
    return schema


def _exact_source_key(key: object, claim: str, support: str) -> str:
    if type(key) is not str or not 0 < len(key.encode("utf-8")) <= 256:
        raise SourceReferenceError("source lookup key is unbounded")
    for source in (claim, support):
        offset = source.find(key)
        if offset >= 0:
            return source[offset:offset + len(key)]
    raise SourceReferenceError("source lookup key is not verbatim")


def _claim_entities(claim: str, body: str, *, policy_version: str = NAMED_ENTITY_POLICY_VERSION_V15) -> tuple[tuple[str, str], ...]:
    found = []
    offset = 0
    for line in claim.splitlines(keepends=True):
        names = bounded_named_entities(line, source_context=body, policy_version=policy_version)
        for name, kind in names:
            for match in re.finditer(_entity_pattern(name), line):
                found.append((offset + match.start(), offset + match.end(), name, kind))
        offset += len(line)
    selected = []
    for item in sorted(found, key=lambda value: (value[0], -(value[1] - value[0]))):
        if not selected or item[0] >= selected[-1][1]:
            selected.append(item)
    return tuple((name, kind) for _, _, name, kind in selected)


def materialise(raw_wire: bytes | str | dict[str, object], view: SourceView,
                request_identity: str, *, provider_schema: dict[str, object] | None = None,
                ) -> tuple[dict[str, object], dict[str, object]]:
    """Resolve source choices into the unchanged governed-package shape."""
    if type(request_identity) is not str or not request_identity:
        raise SourceReferenceError("request identity is absent")
    if (digest_canonical(view.manifest) != view.manifest_digest
            or tuple(digest_bytes(body.encode("utf-8")) for body in view.passages)
            != view.body_digests
            or any("".join(item.text for item in view.segments
                            if item.passage_index == index) != body
                   for index, body in enumerate(view.passages))):
        raise SourceReferenceError("source view or acquired bytes differ")
    if type(raw_wire) is bytes:
        raw = raw_wire
    elif type(raw_wire) is str:
        raw = raw_wire.encode("utf-8")
    elif type(raw_wire) is dict:
        raw = canonical_json_bytes(raw_wire)
    else:
        raise SourceReferenceError("reference result type differs")
    if len(raw) > MAX_RESULT_BYTES:
        raise SourceReferenceError("reference result exceeds bound")
    try:
        wire = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise SourceReferenceError("reference result JSON differs") from exc
    if type(wire) is not dict or set(wire) != {"package"} or type(wire["package"]) is not dict:
        raise SourceReferenceError("reference result shape differs")
    if provider_schema is not None:
        from jsonschema import Draft202012Validator, ValidationError
        try:
            Draft202012Validator(provider_schema).validate(wire)
        except (TypeError, ValueError, ValidationError, RecursionError) as exc:
            raise SourceReferenceError("reference result violates static schema") from exc
    payload = wire["package"]
    expected = {"substantive_claim_indexes", "governed_claims", "qualification_evidence",
                "selection_rationale", "geography", "categories", "explicit_exclusions"}
    if set(payload) != expected:
        raise SourceReferenceError("reference package fields differ")
    if (type(payload["selection_rationale"]) is not str
            or any(type(payload[key]) is not list or len(payload[key]) > 32
                   or any(type(value) is not str for value in payload[key])
                   for key in ("geography", "categories", "explicit_exclusions"))):
        raise SourceReferenceError("reference package value type differs")
    claims = payload["governed_claims"]
    indexes = payload["substantive_claim_indexes"]
    qualifications = payload["qualification_evidence"]
    if (type(claims) is not list or len(claims) > MAX_CLAIMS or type(indexes) is not list
            or len(indexes) > MAX_CLAIMS or type(qualifications) is not list
            or len(qualifications) > 32):
        raise SourceReferenceError("reference result array bound differs")
    materialised = []
    inserted_entity_orders = []
    seen_ranges = set()
    by_id = {segment.span_id: segment for segment in view.segments}
    for ordinal, item in enumerate(claims):
        if type(item) is not dict:
            raise SourceReferenceError("reference claim differs")
        required = {"claim_role", "claim_range", "support_range", "rendered_fragments",
                    "localised_factual_expressions", "quotations", "status",
                    "semantic_relation", "certainty", "originality_basis",
                    "originality_policy_version", "admitted_use", "policy_version"}
        if (set(item) != required
                or type(item["claim_role"]) is not str
                or item["claim_role"] not in {"HEADLINE", "SUBSTANTIVE", "CONTEXT"}
                or any(type(item[key]) is not str for key in (
                    "status", "certainty", "originality_basis",
                    "originality_policy_version", "admitted_use", "policy_version",
                ))
                or type(item["semantic_relation"]) is not dict
                or set(item["semantic_relation"]) != {
                    "source_modality", "rendered_modality", "source_polarity",
                    "rendered_polarity", "relation",
                }
                or any(type(value) is not str for value in item["semantic_relation"].values())):
            raise SourceReferenceError("reference claim fields differ")
        claim, passage_index, source_id = view.resolve_range(item["claim_range"])
        support, support_index, _ = view.resolve_range(item["support_range"])
        claim_first = by_id[item["claim_range"]["first_span_id"]]
        claim_last = by_id[item["claim_range"]["last_span_id"]]
        support_first = by_id[item["support_range"]["first_span_id"]]
        support_last = by_id[item["support_range"]["last_span_id"]]
        if (passage_index != support_index
                or support_first.start_byte > claim_first.start_byte
                or support_last.content_end_byte < claim_last.content_end_byte
                or claim not in support):
            raise SourceReferenceError("claim is not contained in its support")
        range_identity = (source_id, item["claim_range"]["first_span_id"],
                          item["claim_range"]["last_span_id"])
        if range_identity in seen_ranges:
            raise SourceReferenceError("duplicate claim range")
        seen_ranges.add(range_identity)
        entities = _claim_entities(claim, view.passages[passage_index], policy_version=view.entity_policy_version)
        inserted_entity_orders.append([list(item) for item in entities])
        fragments = item["rendered_fragments"]
        if (type(fragments) is not list or len(fragments) != len(entities) + 1
                or len(fragments) > 65 or any(type(part) is not str
                or len(part) > MAX_FRAGMENT_LENGTH for part in fragments)):
            raise SourceReferenceError("rendered fragment count differs")
        # Only these names will be inserted. Unselected source inventories
        # must not turn ordinary HK/date wording into an extra rejection gate.
        if any(re.search(_entity_pattern(name), part)
               for part in fragments for name, _kind in entities):
            raise SourceReferenceError("rendered fragment copied a source entity")
        rendered = "".join(part + (entities[index][0] if index < len(entities) else "")
                           for index, part in enumerate(fragments))
        pairs = item["localised_factual_expressions"]
        quotations = item["quotations"]
        if (type(pairs) is not list or len(pairs) > 32 or type(quotations) is not list
                or len(quotations) > 32):
            raise SourceReferenceError("source key array bound differs")
        localised = []
        for pair in pairs:
            if (type(pair) is not list or len(pair) != 2 or type(pair[1]) is not str
                    or len(pair[1].encode("utf-8")) > 256):
                raise SourceReferenceError("localisation pair differs")
            localised.append([_exact_source_key(pair[0], claim, support), pair[1]])
        quoted = [_exact_source_key(value, claim, support) for value in quotations]
        claim_id = digest_canonical({"request_identity": request_identity,
                                     "role": item["claim_role"], "range": range_identity,
                                     "ordinal": ordinal})
        materialised.append({
            "claim_id": claim_id, "claim": claim, "passage_index": passage_index,
            "supporting_excerpt": support, "source_ids": [source_id],
            "status": item["status"], "rendered_assertion_zh_hant_hk": rendered,
            "claim_role": item["claim_role"],
            "semantic_relation": item["semantic_relation"],
            "localised_factual_expressions": localised, "quotations": quoted,
            "certainty": item["certainty"], "originality_basis": item["originality_basis"],
            "originality_policy_version": item["originality_policy_version"],
            "admitted_use": item["admitted_use"], "policy_version": item["policy_version"],
        })
    if (any(type(index) is not int or not 0 <= index < len(materialised) for index in indexes)
            or len(indexes) != len(set(indexes))):
        raise SourceReferenceError("substantive claim indexes differ")
    resolved_qualifications = []
    for item in qualifications:
        if (type(item) is not dict
                or set(item) != {"test", "claim_index", "test_evidence", "policy_version"}
                or type(item.get("claim_index")) is not int
                or type(item.get("test")) is not str
                or type(item.get("policy_version")) is not str):
            raise SourceReferenceError("qualification claim index differs")
        index = item["claim_index"]
        if not 0 <= index < len(materialised) or type(item.get("test_evidence")) is not dict:
            raise SourceReferenceError("qualification claim binding differs")
        if set(item["test_evidence"]) != _QUALIFICATION_WITNESSES.get(item["test"]):
            raise SourceReferenceError("qualification witness keys differ")
        evidence = {}
        for key, value in item["test_evidence"].items():
            if key in _QUALIFICATION_CLASSIFIER_FIELDS:
                if type(value) is not str:
                    raise SourceReferenceError("qualification classifier type differs")
                evidence[key] = value
            else:
                evidence[key] = _exact_source_key(
                    value, materialised[index]["claim"], materialised[index]["supporting_excerpt"],
                )
        resolved_qualifications.append({
            "test": item.get("test"), "governed_claim_id": materialised[index]["claim_id"],
            "test_evidence": evidence, "policy_version": item.get("policy_version"),
        })
    package = {"package": {
        "substantive_new_information": [materialised[index]["claim"] for index in indexes],
        "governed_claims": materialised, "qualification_evidence": resolved_qualifications,
        "selection_rationale": payload["selection_rationale"],
        "geography": payload["geography"], "categories": payload["categories"],
        "explicit_exclusions": payload["explicit_exclusions"],
    }}
    package_bytes = canonical_json_bytes(package)
    if len(package_bytes) > MAX_RESULT_BYTES:
        raise SourceReferenceError("materialised package exceeds bound")
    receipt = {"version": VERSION, "request_identity": request_identity,
               "raw_digest": digest_bytes(raw), "manifest_digest": view.manifest_digest,
               "provider_schema_digest": (
                   digest_bytes(canonical_json_bytes(provider_schema))
                   if provider_schema is not None else None
               ),
               "body_digests": list(view.body_digests),
               "ordered_entities": [[list(item) for item in names] for names in view.source_entities],
               "claim_entity_order": inserted_entity_orders,
               "package_digest": digest_bytes(package_bytes),
               "materialised_text": package_bytes.decode("utf-8")}
    receipt["receipt_digest"] = digest_canonical(receipt)
    if len(canonical_json_bytes(receipt)) > 2 * MAX_RESULT_BYTES:
        raise SourceReferenceError("materialisation receipt exceeds bound")
    return package, receipt
