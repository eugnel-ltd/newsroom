"""Source-specific initial news eligibility; never source validity or model truth."""
from datetime import timedelta

from newsroom.authority import UtcTimestamp
from newsroom.sources.types import BaselinePolicy, BaselinePolicyKind
from .native_evidence import NativeEvidenceHold

VERSION = 'newsroom.native-news-candidate-eligibility.v1'


def require_news_candidate_eligibility(sources, acquired, *, scope):
    """Keep old initial captures as reference without reclassifying retained work.

    The caller supplies the authenticated current/prior scope and exact approved
    Source Definition Versions. This check adds no qualification or retry credit.
    """
    source_id = sources[0].unit.source_id if sources else 'UNKNOWN_SOURCE'
    if not sources or len(sources) != len(acquired) or type(scope) is not dict:
        raise NativeEvidenceHold('NEWS_CANDIDATE_BASELINE_SCOPE_HOLD', source_id)
    # A source's publication age cannot veto a real later revision/delta.
    if scope.get('newness') == 'KNOWN_CHANGE':
        return
    if scope.get('newness') == 'KNOWN_UNCHANGED':
        raise NativeEvidenceHold('NEWS_CANDIDATE_KNOWN_UNCHANGED', source_id)
    if scope.get('prior_scope') is not None:
        raise NativeEvidenceHold('NEWS_CANDIDATE_NEWNESS_UNPROVEN', source_id)
    eligible = set()
    exclusions = []
    for index, (source, result) in enumerate(zip(sources, acquired, strict=True)):
        source_id = source.unit.source_id
        policy = getattr(getattr(getattr(source, 'source_version', None), 'request', None), 'baseline_policy', None)
        if not isinstance(policy, BaselinePolicy):
            raise NativeEvidenceHold('NEWS_CANDIDATE_BASELINE_POLICY_HOLD', source_id)
        if policy.kind is BaselinePolicyKind.COMPLETE_STATE_FIRST_OBSERVED_ACTIVE:
            eligible.add(index)
            continue  # Active-state authority remains a separate existing gate.
        if policy.kind not in {BaselinePolicyKind.BOUNDED_BACKFILL, BaselinePolicyKind.MAINTAINED_DOCUMENT}:
            raise NativeEvidenceHold('NEWS_CANDIDATE_BASELINE_POLICY_HOLD', source_id)
        first_observed = getattr(source.unit, 'coverage_first_observed_at', '') or getattr(
            getattr(source.unit, 'effective_revision', None), 'first_observed_at', '')
        try:
            observed = UtcTimestamp.parse(first_observed).value
            published = UtcTimestamp.parse(result.publication_time).value
        except (AttributeError, TypeError, ValueError):
            raise NativeEvidenceHold('NEWS_CANDIDATE_INITIAL_TIME_UNPROVEN', source_id) from None
        if published > observed:
            raise NativeEvidenceHold('NEWS_CANDIDATE_INITIAL_PUBLICATION_FUTURE', source_id)
        if policy.kind is BaselinePolicyKind.MAINTAINED_DOCUMENT:
            exclusions.append(('NEWS_CANDIDATE_INITIAL_BASELINE_ONLY', source_id))
        elif observed - published > timedelta(seconds=policy.freshness_window_seconds):
            exclusions.append(('NEWS_CANDIDATE_OUTSIDE_BACKFILL_WINDOW', source_id))
        else:
            eligible.add(index)
    # A valid old reference is not a news basis, nor a veto on a fresh anchor.
    # All policy/time validity checks above still run before admitting that basis.
    if not eligible:
        raise NativeEvidenceHold(*exclusions[0])
    return frozenset(eligible)
