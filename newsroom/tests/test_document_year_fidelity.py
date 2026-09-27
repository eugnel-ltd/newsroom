"""Retained pre-story publication failure; no provider or live store is opened."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from newsroom.control_plane.admission import DeterministicWriteAdmission
from newsroom.control_plane.evidence import EVIDENCE_GATE_POLICY_VERSION, EvidenceGateEvidence
from newsroom.control_plane.writer import WriterCopy, required_surface_copy, validate_writer_copy
from newsroom.increment10.editorial import EditorialPolicyDecision
from newsroom.increment10.evidence import _package_from_value

FIXTURES = Path(__file__).parent / 'fixtures/native_publication'


def package():
    value = _package_from_value(json.loads((FIXTURES/'deadline-package.json').read_bytes())['package'])
    policy = EditorialPolicyDecision.from_bytes((FIXTURES/'deadline-decision.json').read_bytes())
    claims = tuple(c.claim_id for c in value.governed_claims)
    return replace(value, evidence_gate_results=policy.evidence_gate_results,
        evidence_gate_evidence=tuple(EvidenceGateEvidence(gate, result, claims, EVIDENCE_GATE_POLICY_VERSION)
                                    for gate, result in policy.evidence_gate_results),
        freshness_result='PASS', integrity_result='PASS')


def checks(value):
    title, body, links = required_surface_copy(value, paragraphs=True, context_preserving=True)
    copy = WriterCopy(title, body, 'newsroom.offline-exact-copy.v3', value.digest, links)
    return {x.validator: x.result for x in validate_writer_copy(copy, value)}


def test_retained_source_bound_document_year_passes_without_rewriting_claims():
    value = package(); original = value.digest
    assert value.governed_claims[0].localised_factual_expressions == ()
    admission = DeterministicWriteAdmission().decide_candidate_identity(
        candidate_id=value.candidate_id, hypothesis_id=value.hypothesis_id,
        package=value, decided_at='2026-09-27T03:22:42.987906Z',
    )
    assert admission.decision == 'WRITE_READY'
    assert set(checks(value).values()) == {'PASS'}
    assert value.digest == original and value.governed_claims[0].localised_factual_expressions == ()


@pytest.mark.parametrize('change', ['different-year','invented-month','invented-day','duplicate-number','count','duration','identifier','no-term-proof','currency','signed','percent'])
def test_document_year_inference_never_approves_changed_or_unproved_quantities(change):
    value = package(); claim = value.governed_claims[0]; original_claim = claim.claim
    if change == 'different-year': claim = replace(claim, rendered_assertion_zh_hant_hk=claim.rendered_assertion_zh_hant_hk.replace('2026','2027'))
    elif change == 'invented-month': claim = replace(claim, rendered_assertion_zh_hant_hk=claim.rendered_assertion_zh_hant_hk.replace('2026年','2026年5月'))
    elif change == 'invented-day': claim = replace(claim, rendered_assertion_zh_hant_hk=claim.rendered_assertion_zh_hant_hk.replace('2026年','2026年五月一日'))
    elif change == 'duplicate-number': claim = replace(claim, rendered_assertion_zh_hant_hk=claim.rendered_assertion_zh_hant_hk+'涉及2026所學園。')
    elif change == 'count': claim = replace(claim, claim='The 2026 academies submitted returns.', supporting_excerpt='The 2026 academies submitted returns.')
    elif change == 'duration': claim = replace(claim, claim='The return lasts 2026 years.', supporting_excerpt='The return lasts 2026 years.')
    elif change == 'identifier': claim = replace(claim, claim='The form identifier is BFR 2026.', supporting_excerpt='The form identifier is BFR 2026.')
    elif change == 'currency': claim = replace(claim, rendered_assertion_zh_hant_hk='£'+claim.rendered_assertion_zh_hant_hk)
    elif change == 'signed': claim = replace(claim, rendered_assertion_zh_hant_hk='+ '+claim.rendered_assertion_zh_hant_hk)
    elif change == 'percent': claim = replace(claim, rendered_assertion_zh_hant_hk=claim.rendered_assertion_zh_hant_hk.replace('2026年','2026年%'))
    else: claim = replace(claim, named_entities=(), named_entity_evidence=(), rendered_named_entities=())
    value = replace(value, governed_claims=(claim, *value.governed_claims[1:]),
                    substantive_new_information=tuple(claim.claim if item == original_claim else item for item in value.substantive_new_information),
                    passages=tuple(item.replace(original_claim, claim.claim) for item in value.passages))
    assert checks(value)['NUMERIC_AND_DATE_FIDELITY'] == 'FAIL'


@pytest.mark.parametrize('rendered', [
    '表格會繼續開放一個月，供逾期提交。',
    '表格會繼續開放一段短時間（一個月），供逾期提交。',
    '表格會繼續開放一段短時間（30日），供逾期提交。',
])
def test_indefinite_short_period_does_not_grant_exact_duration(rendered):
    value = package(); original = value.governed_claims[1]
    claim = replace(original, rendered_assertion_zh_hant_hk=rendered)
    value = replace(value, governed_claims=(value.governed_claims[0], claim, *value.governed_claims[2:]))
    assert checks(value)['NUMERIC_AND_DATE_FIDELITY'] == 'FAIL'


def test_exact_source_duration_cannot_be_removed_as_an_indefinite_period():
    value = package(); old = value.governed_claims[1]
    claim = replace(old, claim=old.claim.replace('a short period','30 days'),
                    supporting_excerpt=old.supporting_excerpt.replace('a short period','30 days'))
    value = replace(value, governed_claims=(value.governed_claims[0], claim, *value.governed_claims[2:]),
        passages=tuple(p.replace(old.claim, claim.claim) for p in value.passages),
        substantive_new_information=tuple(claim.claim if x == old.claim else x for x in value.substantive_new_information))
    assert checks(value)['NUMERIC_AND_DATE_FIDELITY'] == 'FAIL'


@pytest.mark.parametrize('prefix', ['−', '﹣', '＋', '⁺', '£', 'HK$'])
def test_document_year_does_not_erase_number_signs_or_currency(prefix):
    value = package(); claim = value.governed_claims[0]
    claim = replace(claim, rendered_assertion_zh_hant_hk=prefix + claim.rendered_assertion_zh_hant_hk)
    value = replace(value, governed_claims=(claim, *value.governed_claims[1:]))
    assert checks(value)['NUMERIC_AND_DATE_FIDELITY'] == 'FAIL'


@pytest.mark.parametrize('source', [
    'The form will remain open for a short period of one month for late submissions.',
    'The form will remain open for a short period of a year for late submissions.',
    'The form will remain open for a short period lasting one week for late submissions.',
    'For one month, the form will remain open for a short period.',
    'The form will remain open for a short period until May.',
    'From June, the form will remain open for a short period.',
])
def test_short_period_does_not_discard_source_precision_or_bounds(source):
    value = package(); old = value.governed_claims[1]
    claim = replace(old, claim=source, supporting_excerpt=source)
    value = replace(value, governed_claims=(value.governed_claims[0], claim, *value.governed_claims[2:]),
        passages=tuple(p.replace(old.claim, claim.claim) for p in value.passages),
        substantive_new_information=tuple(claim.claim if x == old.claim else x for x in value.substantive_new_information))
    assert checks(value)['NUMERIC_AND_DATE_FIDELITY'] == 'FAIL'


@pytest.mark.parametrize('extra', [' The deadline expired one day ago.', ' The deadline applies to two schools.'])
def test_year_equivalence_never_masks_a_second_source_numeric_fact(extra):
    value = package(); old = value.governed_claims[0]
    claim = replace(old, claim=old.claim + extra, supporting_excerpt=old.supporting_excerpt + extra)
    value = replace(value, governed_claims=(claim, *value.governed_claims[1:]),
        passages=tuple(p.replace(old.claim, claim.claim) for p in value.passages),
        substantive_new_information=tuple(claim.claim if x == old.claim else x for x in value.substantive_new_information))
    assert checks(value)['NUMERIC_AND_DATE_FIDELITY'] == 'FAIL'


def test_short_period_equivalence_cannot_hide_an_invented_first_period():
    value = package(); old = value.governed_claims[1]
    claim = replace(old, rendered_assertion_zh_hant_hk=old.rendered_assertion_zh_hant_hk.replace('一段', '第一段'))
    value = replace(value, governed_claims=(value.governed_claims[0], claim, *value.governed_claims[2:]))
    assert checks(value)['NUMERIC_AND_DATE_FIDELITY'] == 'FAIL'
