"""Retained deadline facts, not permission to invent or reclassify model claims."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from newsroom.control_plane.admission import _qualification_relation_is_proven
from newsroom.control_plane.evidence import QualificationEvidence

WITNESS = json.loads((Path(__file__).parent / 'fixtures/native_assessor/deadline-43452.json').read_text())
SPAN = WITNESS['governed_claims'][0]['claim']


def qualification(span=SPAN, *, kind='LAW_RIGHT_STATUS_POLICY'):
    fields = ({'change_kind': 'OFFICIAL_DEADLINE', 'change_relation': 'NEW_OR_CHANGED_STATE',
               'new_state': span} if kind == 'LAW_RIGHT_STATUS_POLICY' else
              {'action_class': 'OFFICIAL_DEADLINE', 'action_relation': 'NEW_OR_CHANGED_OFFICIAL_ACTION',
               'reader_action': span})
    return QualificationEvidence(kind, 'c1', 'deadline-proof', tuple({
        **fields, 'event_polarity': 'AFFIRMED', 'material_relation_span': span,
    }.items()))


def proven(span=SPAN, *, source=None, kind='LAW_RIGHT_STATUS_POLICY'):
    return _qualification_relation_is_proven(
        qualification(span, kind=kind), SimpleNamespace(claim=span, supporting_excerpt=span),
        source_context=span if source is None else source,
    )


@pytest.mark.parametrize('kind', ['LAW_RIGHT_STATUS_POLICY', 'OFFICIAL_ACTION_OR_DEADLINE'])
def test_retained_elapsed_deadline_fits_existing_deadline_classifiers(kind):
    assert proven(source=WITNESS['source_text'], kind=kind)


@pytest.mark.parametrize('span', [
    'The deadline for applications has now passed.',
    'The closing date for applications has now expired.',
])
def test_exact_affirmed_deadline_completion_is_supported(span):
    assert proven(span)


@pytest.mark.parametrize('prefix', [
    'It is false that ', 'Officials deny that ', 'Officials denied that\n',
    'If approved; ', 'Subject to approval, ', 'Unless conditions are met, ',
    'Officials are forecasting that ', 'The authority is considering whether ',
])
def test_model_excerpt_cannot_remove_source_denial_condition_or_prediction(prefix):
    assert not proven(source=prefix + SPAN)


@pytest.mark.parametrize('suffix', [' if approved.', ' unless extended.', ' is incorrect.', ' according to unverified reports.'])
def test_model_excerpt_cannot_remove_source_suffix_qualification(suffix):
    assert not proven(source=SPAN[:-1] + suffix)


@pytest.mark.parametrize('span', [
    SPAN.replace('has now passed', 'has not yet passed'),
    SPAN.replace('has now passed', 'may now have passed'),
    SPAN.replace('has now passed', 'is expected to pass'),
    SPAN.replace('has now passed', 'has now been extended'),
    SPAN.replace('budget forecast return', 'forecast that the return deadline'),
    'The deadline remains unchanged.',
])
def test_non_expiry_and_uncertain_facts_are_not_relabelled_as_completion(span):
    assert not proven(span)


def test_elapsed_deadline_cannot_prove_other_typed_change_or_missing_source():
    claim = SimpleNamespace(claim=SPAN, supporting_excerpt=SPAN)
    q = qualification()
    from dataclasses import replace
    q = replace(q, test_evidence=tuple((k, 'LAW' if k == 'change_kind' else v) for k, v in q.test_evidence))
    assert not _qualification_relation_is_proven(q, claim, source_context=WITNESS['source_text'])
    assert not proven(source='Only the current return guidance is available.')
