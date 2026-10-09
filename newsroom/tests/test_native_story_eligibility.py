"""Initial news eligibility is separate from source validity and retained context."""
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import pytest

from newsroom.sources.types import BaselinePolicy, BaselinePolicyKind, VersionedPolicyRef
from newsroom.control_plane.native_evidence import EvidenceAssessor, IndependentEvidenceAssessment, NativeEvidenceHold

FIRST_OBSERVED = '2026-10-09T10:46:46.345370Z'
INITIAL = {'newness': 'SOURCE_DECLARED_FIRST_PUBLICATION', 'prior_scope': None}


def case(*, published='2026-10-09T10:41:52.000000Z', observed=FIRST_OBSERVED,
         kind=BaselinePolicyKind.BOUNDED_BACKFILL, window=604800):
    policy = BaselinePolicy(VersionedPolicyRef('approved-fixture-baseline', 'v1'), kind,
        freshness_window_seconds=window if kind is BaselinePolicyKind.BOUNDED_BACKFILL else None)
    source = SimpleNamespace(unit=SimpleNamespace(source_id='UK-05', coverage_first_observed_at=observed,
        effective_revision=SimpleNamespace(first_observed_at=observed), observed_at='2030-10-09T12:00:00Z'),
        source_version=SimpleNamespace(request=SimpleNamespace(baseline_policy=policy)))
    acquired = SimpleNamespace(publication_time=published, source_updated_time='2026-10-09T12:00:00Z',
        retrieval_time='2030-10-09T12:00:00Z')
    return source, acquired


def test_actual_old_2025_teaching_release_is_reference_not_new_story():
    from newsroom.control_plane.native_story_eligibility import require_news_candidate_eligibility
    source, acquired = case(published='2025-07-02T13:37:48.000000Z', observed='2026-10-07T12:32:22.905193Z')
    before = deepcopy((vars(source.unit), vars(acquired)))
    with pytest.raises(NativeEvidenceHold, match='NEWS_CANDIDATE_OUTSIDE_BACKFILL_WINDOW'):
        require_news_candidate_eligibility((source,), (acquired,), scope=INITIAL)
    assert (vars(source.unit), vars(acquired)) == before


@pytest.mark.parametrize('mode', [{}, {'cached_only': True},
    {'cached_only': True, 'qualification_cached_only': True}, {'semantic_only': True}, {'context_only': True}])
def test_old_initial_news_is_denied_before_cache_origin_or_any_assessment(mode):
    from newsroom.control_plane.native_story_eligibility import require_news_candidate_eligibility
    source, acquired = case(published='2025-07-02T13:37:48.000000Z')
    calls = []
    def forbidden(*_args, **_kwargs):
        calls.append('work')
        pytest.fail('baseline exclusion reached cache/provider work')
    forbidden.assess_with_boundary = forbidden
    def eligibility(_candidate, _package, sources, acquired):
        calls.append('eligibility')
        require_news_candidate_eligibility(sources, acquired, scope=INITIAL)
    adapter = EvidenceAssessor(forbidden, cached_qualification_origin=forbidden,
        news_candidate_eligibility=eligibility)
    with pytest.raises(NativeEvidenceHold, match='NEWS_CANDIDATE_OUTSIDE_BACKFILL_WINDOW'):
        adapter.assess(object(), object(), (source,), (acquired,),
            before_assessment=forbidden, **mode)
    assert calls == ['eligibility']


@pytest.mark.parametrize('scope,reason', [
    ({'newness': 'KNOWN_UNCHANGED', 'prior_scope': {'sources': ['same retained state']}}, 'NEWS_CANDIDATE_KNOWN_UNCHANGED'),
    ({'newness': 'UNKNOWN', 'prior_scope': {'sources': []}}, 'NEWS_CANDIDATE_NEWNESS_UNPROVEN'),
])
def test_unchanged_or_partial_prior_does_not_open_new_provider_work(scope, reason):
    from newsroom.control_plane.native_story_eligibility import require_news_candidate_eligibility
    source, acquired = case()
    with pytest.raises(NativeEvidenceHold, match=reason):
        require_news_candidate_eligibility((source,), (acquired,), scope=scope)


def test_fresh_basis_can_use_old_valid_maintained_rule_as_context():
    from newsroom.control_plane.native_story_eligibility import require_news_candidate_eligibility
    fresh, current = case()
    old, reference = case(published='2021-01-29T15:49:33Z', kind=BaselinePolicyKind.MAINTAINED_DOCUMENT)
    assert require_news_candidate_eligibility((old, fresh), (reference, current), scope=INITIAL) == frozenset({1})
    assert require_news_candidate_eligibility((fresh, old), (current, reference), scope=INITIAL) == frozenset({0})


@pytest.mark.parametrize('published', ['', None, 'not-a-date', '2026-10-09', '2026-10-09T11:00:00Z'])
def test_fresh_basis_does_not_hide_invalid_or_future_reference_time(published):
    from newsroom.control_plane.native_story_eligibility import require_news_candidate_eligibility
    fresh, current = case()
    old, reference = case(published=published, kind=BaselinePolicyKind.MAINTAINED_DOCUMENT)
    with pytest.raises(NativeEvidenceHold, match='NEWS_CANDIDATE_INITIAL_(TIME_UNPROVEN|PUBLICATION_FUTURE)'):
        require_news_candidate_eligibility((fresh, old), (current, reference), scope=INITIAL)


@pytest.mark.parametrize('headline', [0, 1])
def test_fresh_context_cannot_launder_an_old_headline(headline):
    from newsroom.control_plane.native_story_eligibility import require_news_candidate_eligibility
    fresh, current = case()
    old, reference = case(published='2021-01-29T15:49:33Z', kind=BaselinePolicyKind.MAINTAINED_DOCUMENT)
    result = IndependentEvidenceAssessment((), (), ('An asserted news item',),
        (SimpleNamespace(claim_role='HEADLINE', passage_index=headline),
         SimpleNamespace(claim_role='CONTEXT', passage_index=1 - headline)), (), (), 'fixture', (), ())
    def assess(*_args, **_kwargs):
        return result
    assess.assess_with_boundary = assess
    adapter = EvidenceAssessor(assess, news_candidate_eligibility=lambda _candidate, _package, sources, acquired:
        require_news_candidate_eligibility(sources, acquired, scope=INITIAL))
    if headline == 1:
        assert adapter.assess(object(), object(), (old, fresh), (reference, current)) is result
    else:
        with pytest.raises(NativeEvidenceHold, match='NEWS_CANDIDATE_INITIAL_HEADLINE_OUTSIDE_BASIS'):
            adapter.assess(object(), object(), (old, fresh), (reference, current))


def test_actual_today_visa_remains_eligible_after_reopen_and_later_retrieval():
    from newsroom.control_plane.native_story_eligibility import require_news_candidate_eligibility
    source, acquired = case()  # Actual 9 October visa publication/first observation.
    assert require_news_candidate_eligibility((source,), (acquired,), scope=INITIAL) == frozenset({0})
    source.unit.observed_at = acquired.retrieval_time = '2040-10-09T12:00:00Z'
    assert require_news_candidate_eligibility((source,), (acquired,), scope=INITIAL) == frozenset({0})


def test_old_rule_with_real_current_prior_change_has_no_initial_age_cutoff():
    from newsroom.control_plane.native_story_eligibility import require_news_candidate_eligibility
    source, acquired = case(published='2021-01-29T15:49:33Z', kind=BaselinePolicyKind.MAINTAINED_DOCUMENT)
    scope = {'newness': 'KNOWN_CHANGE', 'prior_scope': {'sources': [{'body': 'Prior rule without child exception'}]}}
    assert require_news_candidate_eligibility((source,), (acquired,), scope=scope) is None


@pytest.mark.parametrize('extra_seconds', [0, 1])
def test_source_policy_boundary_is_inclusive_not_a_moving_clock(extra_seconds):
    from newsroom.control_plane.native_story_eligibility import require_news_candidate_eligibility
    observed = datetime.fromisoformat(FIRST_OBSERVED.replace('Z', '+00:00'))
    published = (observed - timedelta(seconds=604800 + extra_seconds)).isoformat()
    source, acquired = case(published=published)
    if extra_seconds:
        with pytest.raises(NativeEvidenceHold, match='OUTSIDE_BACKFILL_WINDOW'):
            require_news_candidate_eligibility((source,), (acquired,), scope=INITIAL)
    else:
        assert require_news_candidate_eligibility((source,), (acquired,), scope=INITIAL) == frozenset({0})


@pytest.mark.parametrize('window,allowed', [(86400, False), (345600, True)])
def test_each_exact_source_definition_owns_its_window(window, allowed):
    from newsroom.control_plane.native_story_eligibility import require_news_candidate_eligibility
    observed = datetime.fromisoformat(FIRST_OBSERVED.replace('Z', '+00:00'))
    source, acquired = case(published=(observed - timedelta(days=2)).isoformat(), window=window)
    if allowed:
        assert require_news_candidate_eligibility((source,), (acquired,), scope=INITIAL) == frozenset({0})
    else:
        with pytest.raises(NativeEvidenceHold, match='OUTSIDE_BACKFILL_WINDOW'):
            require_news_candidate_eligibility((source,), (acquired,), scope=INITIAL)


def test_only_stable_revision_time_can_backfill_a_missing_pull_property():
    from newsroom.control_plane.native_story_eligibility import require_news_candidate_eligibility
    source, acquired = case()
    del source.unit.coverage_first_observed_at
    assert require_news_candidate_eligibility((source,), (acquired,), scope=INITIAL) == frozenset({0})
    source.unit.effective_revision.first_observed_at = ''
    with pytest.raises(NativeEvidenceHold, match='INITIAL_TIME_UNPROVEN'):
        require_news_candidate_eligibility((source,), (acquired,), scope=INITIAL)


def test_fresh_anchor_does_not_hide_a_missing_policy():
    from newsroom.control_plane.native_story_eligibility import require_news_candidate_eligibility
    fresh, current = case()
    other, reference = case()
    del other.source_version.request.baseline_policy
    with pytest.raises(NativeEvidenceHold, match='BASELINE_POLICY_HOLD'):
        require_news_candidate_eligibility((fresh, other), (current, reference), scope=INITIAL)


@pytest.mark.parametrize('kind,reason', [
    (BaselinePolicyKind.MAINTAINED_DOCUMENT, 'INITIAL_BASELINE_ONLY'),
    (BaselinePolicyKind.MANUAL_ONLY, 'BASELINE_POLICY_HOLD'),
])
def test_initial_reference_or_non_news_policy_does_not_create_new_story(kind, reason):
    from newsroom.control_plane.native_story_eligibility import require_news_candidate_eligibility
    source, acquired = case(kind=kind)
    with pytest.raises(NativeEvidenceHold, match=reason):
        require_news_candidate_eligibility((source,), (acquired,), scope=INITIAL)


def test_source_approved_first_observed_active_policy_remains_separate():
    from newsroom.control_plane.native_story_eligibility import require_news_candidate_eligibility
    source, acquired = case(kind=BaselinePolicyKind.COMPLETE_STATE_FIRST_OBSERVED_ACTIVE, published='')
    assert require_news_candidate_eligibility((source,), (acquired,), scope={'newness': 'UNKNOWN', 'prior_scope': None}) == frozenset({0})


def test_legitimate_no_news_result_needs_no_headline_or_new_bounded_work():
    from newsroom.control_plane.native_story_eligibility import require_news_candidate_eligibility
    source, acquired = case()
    result = IndependentEvidenceAssessment((), (), (), (), (), (), 'No new information', (), ())
    adapter = EvidenceAssessor(lambda *_: result,
        news_candidate_eligibility=lambda _candidate, _package, sources, acquired:
            require_news_candidate_eligibility(sources, acquired, scope=INITIAL))
    assert adapter.assess(object(), object(), (source,), (acquired,)) is result
