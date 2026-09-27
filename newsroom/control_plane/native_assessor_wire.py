"""Small v18 provider wire mapped to the frozen v17 source-reference consumer."""

from __future__ import annotations

import json

from jsonschema import Draft202012Validator, ValidationError

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.control_plane.admission import _QUALIFICATION_CLASSIFIER_FIELDS

from .native_assessor_references import (
    MAX_RESULT_BYTES, SourceReferenceError, SourceView, _unique_object,
    materialise as materialise_v17,
)

VERSION = "newsroom.native-assessor-wire.v1"
_CLAIM_CONSTANTS = (
    "semantic_relation", "certainty", "originality_basis",
    "originality_policy_version", "admitted_use", "policy_version",
)


def _claim_properties(schema: dict) -> dict:
    return schema["properties"]["package"]["properties"]["governed_claims"]["items"]["properties"]


def _qualification_variants(schema: dict) -> list[dict]:
    return schema["properties"]["package"]["properties"]["qualification_evidence"]["items"]["oneOf"]


def make_provider_schema(v17_schema: dict[str, object]) -> dict[str, object]:
    """Derive one static closed schema without changing the frozen v17 codec."""
    schema = json.loads(canonical_json_bytes(v17_schema))
    claim = schema["properties"]["package"]["properties"]["governed_claims"]["items"]
    fields = claim["properties"]
    for key in _CLAIM_CONSTANTS:
        fields.pop(key)
    fields.pop("support_range")
    fields["source_range"] = fields.pop("claim_range")
    fields["rendered_assertion_zh_hant_hk_fragments"] = fields.pop("rendered_fragments")
    pair = fields.pop("localised_factual_expressions")
    fields["factual_localisations"] = {
        "type": "array", "maxItems": pair["maxItems"], "items": {
            "type": "object", "properties": {
                "source_lookup_key": {"type": "string", "maxLength": 256},
                "rendered_expression": {"type": "string", "maxLength": 256},
            },
            "required": ["source_lookup_key", "rendered_expression"],
            "additionalProperties": False,
        },
    }
    fields["quotation_source_keys"] = fields.pop("quotations")
    claim["required"] = list(fields)
    for qualification in _qualification_variants(schema):
        fields = qualification["properties"]
        fields.pop("policy_version")
        witnesses = fields["test_evidence"]
        witness_fields = witnesses["properties"]
        for key in tuple(witness_fields):
            if key not in _QUALIFICATION_CLASSIFIER_FIELDS:
                witness_fields[key + "_source_lookup_key"] = witness_fields.pop(key)
        witnesses["required"] = list(witness_fields)
        qualification["required"] = list(fields)
    return schema


def _constant(schema: dict, key: str) -> object:
    value = schema.get("const")
    if "const" in schema:
        return value
    if schema.get("type") == "object":
        props = schema.get("properties")
        if type(props) is dict and all("const" in item for item in props.values()):
            return {name: item["const"] for name, item in props.items()}
    raise SourceReferenceError(f"v17 {key} is not a schema constant")


def materialise(
    raw: bytes | str | dict[str, object], view: SourceView, request_identity: str,
    *, provider_schema: dict[str, object], v17_schema: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate v18, restore v17 constants, then use the frozen materialiser."""
    try:
        if type(raw) is bytes:
            raw_bytes = raw
        elif type(raw) is str:
            raw_bytes = raw.encode("utf-8")
        elif type(raw) is dict:
            raw_bytes = canonical_json_bytes(raw)
        else:
            raise SourceReferenceError("v18 result type differs")
    except (UnicodeError, TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise SourceReferenceError("v18 raw result encoding differs") from exc
    if len(raw_bytes) > MAX_RESULT_BYTES:
        raise SourceReferenceError("v18 raw result exceeds bound")
    try:
        wire = json.loads(raw_bytes, object_pairs_hook=_unique_object)
        Draft202012Validator(provider_schema).validate(wire)
    except (UnicodeDecodeError, ValueError, TypeError, RecursionError, ValidationError) as exc:
        raise SourceReferenceError("v18 result violates static schema") from exc
    claim_constants = {
        key: _constant(_claim_properties(v17_schema)[key], key)
        for key in _CLAIM_CONSTANTS
    }
    qualification_constants = {
        variant["properties"]["test"]["const"]: _constant(
            variant["properties"]["policy_version"], "qualification policy_version",
        )
        for variant in _qualification_variants(v17_schema)
    }
    source = wire["package"]
    claims = []
    for item in source["governed_claims"]:
        claims.append({
            "claim_range": item["source_range"],
            "support_range": item["source_range"],
            "rendered_fragments": item["rendered_assertion_zh_hant_hk_fragments"],
            "localised_factual_expressions": [
                [record["source_lookup_key"], record["rendered_expression"]]
                for record in item["factual_localisations"]
            ],
            "quotations": item["quotation_source_keys"],
            "status": item["status"], "claim_role": item["claim_role"],
            **claim_constants,
        })
    qualifications = []
    for item in source["qualification_evidence"]:
        test = item["test"]
        if test not in qualification_constants:
            raise SourceReferenceError("v18 qualification classifier differs")
        witnesses = {
            (key.removesuffix("_source_lookup_key")
             if key.endswith("_source_lookup_key") else key): value
            for key, value in item["test_evidence"].items()
        }
        qualifications.append({
            "test": test, "claim_index": item["claim_index"],
            "test_evidence": witnesses,
            "policy_version": qualification_constants[test],
        })
    v17_wire = {"package": {
        **{key: source[key] for key in (
            "substantive_claim_indexes", "selection_rationale", "geography",
            "categories", "explicit_exclusions",
        )},
        "governed_claims": claims, "qualification_evidence": qualifications,
    }}
    package, receipt = materialise_v17(
        v17_wire, view, request_identity, provider_schema=v17_schema,
    )
    receipt = dict(receipt)
    receipt["raw_digest"] = digest_bytes(raw_bytes)
    receipt["provider_schema_digest"] = digest_bytes(canonical_json_bytes(provider_schema))
    receipt.pop("receipt_digest")
    receipt["receipt_digest"] = digest_canonical(receipt)
    if len(canonical_json_bytes(receipt)) > 2 * MAX_RESULT_BYTES:
        raise SourceReferenceError("v18 materialisation receipt exceeds bound")
    return package, receipt
