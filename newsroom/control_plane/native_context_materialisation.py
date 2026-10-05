"""Pure context composition; callers authenticate receipts and fresh Source fences.

A matching digest is a codec binding, never evidence of permission, accounting
or model truth. Only an already read support/localisation result belongs here.
"""
from __future__ import annotations

from copy import deepcopy
import json

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical, validate_sha256_digest
from .native_assessor import NativeAssessmentExecution, _materialise_reference_result, _reference_binding, VERSION as CODEC
from .native_assessor_judgments import JudgedAssessment, VERSION as JUDGMENT_VERSION
from .native_assessor_references import SourceView, MAX_CLAIMS, VERSION as REFERENCE_VERSION
from .native_source_qualification import VERSION as QUALIFICATION_VERSION

VERSION = 'newsroom.native-context-materialisation.v1'


class ContextCompositionError(ValueError):
    pass


def _original_receipt(original, view):
    if isinstance(original, JudgedAssessment):
        decision = json.loads(original.decision_record)
        if (canonical_json_bytes(decision) != original.decision_record
                or decision.get('schema') not in {JUDGMENT_VERSION, QUALIFICATION_VERSION}
                or 'composition_proof' in decision):
            raise ContextCompositionError('CONTEXT_ORIGINAL_NESTING_OR_SCHEMA')
        receipt = decision.get('materialisation_receipt')
        if (type(receipt) is not dict or not isinstance(original.execution, NativeAssessmentExecution)
                or original.execution.text != receipt.get('materialised_text')):
            raise ContextCompositionError('CONTEXT_ORIGINAL_EXECUTION_BINDING')
    elif type(original) is dict:
        receipt = original
    else:
        raise ContextCompositionError('CONTEXT_ORIGINAL_TYPE')
    if type(receipt) is not dict:
        raise ContextCompositionError('CONTEXT_ORIGINAL_RECEIPT')
    unsigned = dict(receipt)
    digest = unsigned.pop('receipt_digest', None)
    text = receipt.get('materialised_text')
    if (digest_canonical(unsigned) != digest or type(text) is not str
            or digest_bytes(text.encode()) != receipt.get('package_digest')
            or receipt.get('version') != REFERENCE_VERSION
            or receipt.get('manifest_digest') != view.manifest_digest
            or receipt.get('body_digests') != list(view.body_digests)):
        raise ContextCompositionError('CONTEXT_ORIGINAL_RECEIPT_BINDING')
    document = json.loads(text)
    if canonical_json_bytes(document).decode() != text:
        raise ContextCompositionError('CONTEXT_ORIGINAL_CANONICAL_TEXT')
    package = document['package']
    if (sum(c['claim_role'] == 'HEADLINE' for c in package['governed_claims']) != 1
            or not package['substantive_new_information'] or not package['qualification_evidence']):
        raise ContextCompositionError('CONTEXT_ORIGINAL_HEADLINE_UNQUALIFIED')
    return receipt, document


def compose_context_execution(original, context_wire, localisation_receipt, *, binding,
                              view: SourceView, support_receipt):
    """Keep original expanded decisions; derive only new CONTEXT claim identities.

    ``support_receipt`` is an authenticated Typesafe.read result with four flat
    judgments per included span. The caller also authenticates localisation and
    original CAS/usage records. This pure function does not replace those reads.
    """
    if not isinstance(view, SourceView) or type(binding) is not dict:
        raise ContextCompositionError('CONTEXT_SOURCE_VIEW')
    validate_sha256_digest(binding.get('content_digest'))
    receipt, document = _original_receipt(original, view)
    if (binding.get('context_purpose') != 'newsroom.native-context-package.v1'
            or binding.get('context_original_receipt_digest') != digest_canonical(receipt)
            or binding.get('source_reference_binding') != _reference_binding(view)):
        raise ContextCompositionError('CONTEXT_PURPOSE_OR_SOURCE_BINDING')
    if isinstance(original, JudgedAssessment):
        original_binding = json.loads(original.decision_record)['source_binding']
        if any(original_binding.get(key) != binding.get(key) for key in (
                'content_digest', 'candidate_id', 'candidate_version_id', 'hypothesis_digest',
                'source_reference_binding', 'source_currentness')):
            raise ContextCompositionError('CONTEXT_ORIGINAL_SCOPE_BINDING')
    ranges = binding.get('context_ranges')
    if type(ranges) is not dict or not 0 < len(ranges) <= MAX_CLAIMS:
        raise ContextCompositionError('CONTEXT_RANGE_INVENTORY')
    if (support_receipt.get('snapshot', {}).get('source_binding') != binding
            or localisation_receipt.get('source_binding') != binding):
        raise ContextCompositionError('CONTEXT_READER_BINDING')
    expected = {f'{identity}:{field}' for identity in ranges
                for field in ('support', 'modality', 'attribution', 'entities')}
    answers = support_receipt.get('answers')
    if type(answers) is not dict or set(answers) != expected:
        raise ContextCompositionError('CONTEXT_SUPPORT_INVENTORY')
    if any(type(answers[key]) is not dict or answers[key].get('choice') !=
           ('SUPPORTED' if key.endswith(':support') else 'YES') for key in expected):
        raise ContextCompositionError('CONTEXT_SUPPORT_UNPROVEN')
    for value in (support_receipt, localisation_receipt):
        for field in ('invocation_id', 'terminal_digest'):
            validate_sha256_digest(value.get(field))
    wire = deepcopy(context_wire)
    package = wire['package']
    claims = package['governed_claims']
    if (package['select_new_information'] is not False or package['qualification_evidence']
            or any(package[key] for key in ('geography', 'categories', 'explicit_exclusions'))
            or len(claims) != len(ranges)
            or any(claim.get('claim_role') != 'CONTEXT' for claim in claims)
            or len(document['package']['governed_claims']) + len(claims) > MAX_CLAIMS):
        raise ContextCompositionError('CONTEXT_ONLY_WIRE_REQUIRED')
    by_range = {canonical_json_bytes(value): identity for identity, value in ranges.items()}
    if len(by_range) != len(ranges):
        raise ContextCompositionError('CONTEXT_DUPLICATE_RANGE')
    included = set()
    for claim in claims:
        identity = by_range.get(canonical_json_bytes(claim['source_range']))
        if identity is None or identity in included:
            raise ContextCompositionError('CONTEXT_WIRE_RANGE_BINDING')
        view.resolve_range(claim['source_range'])
        included.add(identity)
        rendering = {key: claim[key] for key in ('rendered_assertion_zh_hant_hk_fragments',
                     'factual_localisations', 'quotation_source_keys')}
        if localisation_receipt.get('renderings', {}).get(identity) != rendering:
            raise ContextCompositionError('CONTEXT_LOCALISATION_WIRE_BINDING')
    if set(localisation_receipt.get('renderings', {})) != included:
        raise ContextCompositionError('CONTEXT_LOCALISATION_INVENTORY')
    context, context_receipt = _materialise_reference_result(canonical_json_bytes(wire), view,
                                                          digest_canonical(binding), CODEC)
    combined = deepcopy(document)
    combined['package']['governed_claims'].extend(context['package']['governed_claims'])
    ids = [c['claim_id'] for c in combined['package']['governed_claims']]
    if len(ids) != len(set(ids)):
        raise ContextCompositionError('CONTEXT_DUPLICATE_CLAIM')
    text = canonical_json_bytes(combined).decode()
    proof = {'version': VERSION, 'source_binding_digest': digest_canonical(binding),
        'original_receipt_digest': digest_canonical(receipt),
        'original_materialisation_digest': receipt['receipt_digest'],
        'context_receipt_digest': context_receipt['receipt_digest'],
        'support_receipt_digest': digest_canonical(support_receipt),
        'localisation_receipt_digest': digest_canonical(localisation_receipt),
        'combined_digest': digest_bytes(text.encode())}
    return NativeAssessmentExecution(text, {}), proof
