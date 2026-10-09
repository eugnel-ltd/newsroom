"""The existing real opener composes qualified semantic services without dispatch."""
import pytest

from newsroom.authority.canonical import digest_bytes
from newsroom.control_plane import native_assessor_judgments
from newsroom.control_plane.native_claim_localisation import localisation_policy
from newsroom.control_plane.typesafe_judgment import judgment_policy
from newsroom.tests import test_native_composition as existing


def test_real_semantic_composition_opens_and_reopens_without_model_calls(tmp_path, monkeypatch):
    from newsroom.control_plane import native_composition
    original_arguments = existing._arguments
    original_consumer = native_assessor_judgments.NativeAssessorJudgments
    original_graphiti = native_composition.NativeGraphitiProcessor
    consumers = []
    processors = []

    def arguments(path):
        return {**original_arguments(path),
            'judgment_api_key': lambda: pytest.fail('opening must not read credentials'),
            'judgment_policy': judgment_policy(evidence_digest=digest_bytes(b'qualified semantic fixture'), qualified=True),
            'localisation_policy': localisation_policy(evidence_digest=digest_bytes(b'qualified rendering fixture'), qualified=True)}

    def consumer(**values):
        result = original_consumer(**values)
        consumers.append(result)
        return result

    def graphiti(**values):
        result = original_graphiti(**values)
        processors.append(result)
        return result

    monkeypatch.setattr(existing, '_arguments', arguments)
    monkeypatch.setattr(native_assessor_judgments, 'NativeAssessorJudgments', consumer)
    monkeypatch.setattr(native_composition, 'NativeGraphitiProcessor', graphiti)
    existing.test_native_composition_opens_factory_once_reopens_and_has_no_pre_effect(tmp_path, monkeypatch)
    assert len(consumers) == 2
    assert consumers[0].judgments.policy.route == 'TYPESAFE_JUDGMENT'
    assert consumers[0].localise is not None and consumers[0].read_localisation is not None
    assert len(processors) == 2
    assert all(callable(processor._runner._typed_proposal_verifier) for processor in processors)


def _source(*, revision='current', body='The authority approved a new programme.',
            headline='Programme approved', observed='2026-10-05T10:00:00Z',
            definition='definition-v1', source_id='UK03', item_key='programme',
            canonical_url='https://www.gov.uk/programme', authorised=True):
    from types import SimpleNamespace
    authority = SimpleNamespace(definition_id='definition-id', definition_version_id=definition,
                                revision_id=revision) if authorised else None
    return SimpleNamespace(unit=SimpleNamespace(source_id=source_id, item_key=item_key,
        canonical_url=canonical_url, authority=authority, headline=headline, body=body,
        observed_at=observed, published_at=observed, updated_at=observed,
        ingest_id='ingest-'+revision, effective_revision=SimpleNamespace(),
        revision_digest=digest_bytes(body.encode())),
        source_version=SimpleNamespace(request=SimpleNamespace(roles=())))


def _scope(monkeypatch, current, prior=(), *, acquired_count=None):
    from types import SimpleNamespace
    from newsroom.control_plane import native_composition
    units = {str(source.unit.authority.revision_id if source.unit.authority else index): (source.unit,)
             for index, source in enumerate(prior)}
    # A header-only scan must not read every historical body.
    def header(values, revision):
        value = values[revision][0]
        return SimpleNamespace(source_id=value.source_id, item_key=value.item_key,
                               canonical_url=value.canonical_url, observed_ats=(value.observed_at,))
    monkeypatch.setattr(native_composition, 'source_header', header)
    acquired = [SimpleNamespace(body=source.unit.body.encode(), publication_time=source.unit.published_at,
                                source_updated_time=source.unit.updated_at,
                                canonical_url=source.unit.canonical_url, source_type='PRIMARY_OFFICIAL',
                                currentness_basis='AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT',
                                retrieval_time=source.unit.observed_at,
                                receipt_digest=digest_bytes(source.unit.body.encode())) for source in current]
    if acquired_count is not None:
        acquired = acquired[:acquired_count]
    return native_composition._judgment_scope(SimpleNamespace(units=units), current, acquired)


@pytest.mark.parametrize('prior, expected', [
    ((), 'UNKNOWN'),
    ((_source(revision='old', observed='2026-10-04T10:00:00Z'),), 'KNOWN_UNCHANGED'),
    ((_source(revision='old', observed='2026-10-04T10:00:00Z', body='The programme was proposed.'),), 'KNOWN_CHANGE'),
    ((_source(revision='later', observed='2026-10-06T10:00:00Z'),), 'UNKNOWN'),
    ((_source(revision='current', observed='2026-10-04T10:00:00Z'),), 'UNKNOWN'),
    ((_source(revision='old', observed='2026-10-04T10:00:00Z', body=''),), 'UNKNOWN'),
    ((_source(revision='old', observed='2026-10-04T10:00:00Z', authorised=False),), 'UNKNOWN'),
    ((_source(revision='other-item', observed='2026-10-04T10:00:00Z', item_key='other'),), 'UNKNOWN'),
])
def test_semantic_scope_needs_exact_retained_prior(monkeypatch, prior, expected):
    scope = _scope(monkeypatch, (_source(),), prior)
    assert scope['newness'] == expected
    assert scope['coverage'] == 'COMPLETE'
    assert scope['source_currentness'][0]['definition_version_id'] == 'definition-v1'
    if expected == 'UNKNOWN':
        assert scope['prior_scope'] is None


def test_semantic_scope_incomplete_acquisition_is_partial_not_zip_failure(monkeypatch):
    scope = _scope(monkeypatch, (_source(), _source(source_id='UK05')), acquired_count=1)
    assert scope == {'coverage': 'PARTIAL', 'newness': 'UNKNOWN', 'prior_scope': None}


def test_semantic_scope_unrelated_feed_hold_is_not_editorial_veto(monkeypatch):
    current = _source()
    old = _source(revision='old', observed='2026-10-04T10:00:00Z', body='Earlier programme.')
    unrelated = _source(revision='held-other', observed='2026-10-04T11:00:00Z', item_key='held-other')
    scope = _scope(monkeypatch, (current,), (old, unrelated))
    assert scope['coverage'] == 'COMPLETE'
    assert scope['newness'] == 'KNOWN_CHANGE'
    assert scope['prior_scope']['sources'][0]['body'] == 'Earlier programme.'


@pytest.mark.parametrize('stale_profile', [None, 'judgment', 'rendering'])
def test_deployed_factory_requires_registered_profiles_and_defers_credential_read(tmp_path, monkeypatch, stale_profile):
    from pathlib import Path
    from types import SimpleNamespace
    from newsroom.control_plane import native_composition
    from newsroom.control_plane.model_usage import WorkloadClass
    from newsroom.control_plane import typesafe_judgment, native_claim_localisation

    home = tmp_path / 'home'
    key = home / '.config/newsroom/credentials/typesafe-jev-evaluation.key'
    key.parent.mkdir(parents=True)
    # Deliberately invalid credential: opening composes, dispatch reads it.
    key.write_text('not-a-key')
    monkeypatch.setattr(Path, 'home', lambda: home)
    actual_setattr = monkeypatch.setattr
    captured = []
    requests = []
    semantic = judgment_policy(evidence_digest=digest_bytes(b'semantic acceptance'), qualified=True)
    rendering = localisation_policy(evidence_digest=digest_bytes(b'rendering acceptance'), qualified=True)
    if stale_profile is not None:
        from dataclasses import asdict
        from newsroom.control_plane.model_usage import InvocationEfficiencyPolicy
        selected = semantic if stale_profile == 'judgment' else rendering
        values = asdict(selected)
        values.pop('canonical_digest')
        values['implementation_revision'] = digest_bytes(b'stale implementation')
        selected = InvocationEfficiencyPolicy.create(**values)
        if stale_profile == 'judgment':
            semantic = selected
        else:
            rendering = selected

    def patched_setattr(target, name, value, *args, **kwargs):
        if target is native_composition and name == 'ModelUsageService':
            original_factory = value
            def factory(path):
                original = original_factory(path)
                def qualified(**request):
                    requests.append(request)
                    if request.get('route') == semantic.route:
                        assert request['workload_class'] is WorkloadClass.TYPESAFE_JUDGMENT
                        assert request['implementation_revision'] == typesafe_judgment.implementation_digest()
                        return semantic
                    if request.get('route') == rendering.route:
                        assert request['implementation_revision'] == digest_bytes(Path(native_claim_localisation.__file__).read_bytes())
                        return rendering
                    return original.qualified_policy(**request)
                return SimpleNamespace(qualified_policy=qualified)
            value = factory
        elif target is native_composition and name == 'open_native_pipeline':
            original_open = value
            def opening(**arguments):
                captured.append(arguments)
                return original_open(**arguments)
            value = opening
        return actual_setattr(target, name, value, *args, **kwargs)

    monkeypatch.setattr = patched_setattr
    existing.test_deployed_continuous_runtime_runs_without_history_qualification_gate(tmp_path, monkeypatch, False)
    if stale_profile is not None:
        assert 'judgment_policy' not in captured[0]
        assert 'localisation_policy' not in captured[0]
        return
    assert captured[0]['judgment_policy'] == semantic
    assert captured[0]['localisation_policy'] == rendering
    # The credential is still not read on factory open; the dispatch boundary
    # will reject the unsafe file before a provider call.
    with pytest.raises(ValueError, match='credential'):
        captured[0]['judgment_api_key']()
    assert [r['route'] for r in requests if r['route'] in {semantic.route, rendering.route}] == [semantic.route, rendering.route]


@pytest.mark.parametrize('changed, permitted', [(False, True), (True, True), (False, False)])
def test_shared_semantic_dispatch_fence_rechecks_source_definition_and_rights(changed, permitted):
    from types import SimpleNamespace
    from newsroom.control_plane.native_composition import _require_semantic_current_sources
    from newsroom.sources import SourceDefinitionId
    proof = object()
    identity = '00000000-0000-4000-8000-000000003201'
    calls = []
    def current(selected, **request):
        assert selected == SourceDefinitionId.parse(identity) and request['proof'] is proof
        return SimpleNamespace(version_id='changed' if changed else 'current-version')
    def details(selected, **request):
        calls.append(('definition', selected))
        return SimpleNamespace(request=SimpleNamespace(locator='https://www.gov.uk/approved'))
    def rights(source_id, locator):
        calls.append(('rights', source_id, locator))
        return {'permission': 'PERMITTED'} if permitted else None
    arguments = dict(proof=proof, rights_for=rights)
    binding = {'source_currentness': [{'source_id': 'UK-03', 'definition_id': identity,
                                      'definition_version_id': 'current-version'}]}
    sources = SimpleNamespace(current_summary=current, version_details=details)
    if changed or not permitted:
        with pytest.raises(ValueError, match='definition changed|rights unavailable'):
            _require_semantic_current_sources(sources, binding, **arguments)
    else:
        _require_semantic_current_sources(sources, binding, **arguments)
    assert len(calls) == (0 if changed else 2)


@pytest.mark.parametrize('case,expected', [
    ('declared', 'SOURCE_DECLARED_FIRST_PUBLICATION'),
    ('updated-only', 'UNKNOWN'),
    ('not-originating', 'UNKNOWN'),
    ('non-govuk', 'UNKNOWN'),
    ('known-prior-missing-body', 'UNKNOWN'),
])
def test_first_publication_is_publisher_provenance_not_observation_or_change(monkeypatch, case, expected):
    from types import SimpleNamespace
    current = _source(source_id='UK-01')
    current.source_version.request.roles = (SimpleNamespace(role=SimpleNamespace(value='ORIGINATING_AUTHORITY')),)
    prior = ()
    if case == 'updated-only':
        current.unit.published_at = ''
    elif case == 'not-originating':
        current.source_version.request.roles = ()
    elif case == 'non-govuk':
        current.unit.canonical_url = 'https://example.invalid/news'
    elif case == 'known-prior-missing-body':
        prior = (_source(revision='old', source_id='UK-01', body='', observed='2026-10-04T10:00:00Z'),)
    scope = _scope(monkeypatch, (current,), prior)
    assert scope['newness'] == expected
    assert scope['prior_scope'] is None
    if expected == 'SOURCE_DECLARED_FIRST_PUBLICATION':
        assert scope['first_publication'][0] == {
            'source_id': 'UK-01', 'definition_id': 'definition-id',
            'definition_version_id': 'definition-v1', 'source_revision_digest': current.unit.revision_digest,
            'acquisition_receipt_digest': digest_bytes(current.unit.body.encode()),
            'first_published_at': current.unit.published_at}
    else:
        assert 'first_publication' not in scope


def test_real_qualification_composition_opens_without_any_provider_dispatch(tmp_path, monkeypatch):
    from newsroom.control_plane import native_composition
    original_arguments = existing._arguments
    from newsroom.control_plane import native_assessor
    from newsroom.control_plane.native_source_qualification import qualification_policy
    original_assessor = native_assessor.AutonomousNativeEvidenceAssessor
    original_consumer = native_assessor_judgments.NativeAssessorJudgments
    original_graphiti = native_composition.NativeGraphitiProcessor
    consumers = []
    processors = []
    assessors = []
    pipelines = []
    original_pipeline = native_composition.NativePipeline
    def pipeline(**values):
        result = original_pipeline(**values)
        pipelines.append(result)
        return result

    def arguments(path):
        return {**original_arguments(path),
            'judgment_api_key': lambda: pytest.fail('opening must not read credentials'),
            'judgment_policy': judgment_policy(evidence_digest=digest_bytes(b'qualified semantic fixture'), qualified=True),
            'localisation_policy': localisation_policy(evidence_digest=digest_bytes(b'qualified rendering fixture'), qualified=True),
            'source_qualification_policy': qualification_policy(evidence_digest=digest_bytes(b'qualified exception fixture'), qualified=True)}

    def consumer(**values):
        result = original_consumer(**values)
        consumers.append(result)
        return result

    def graphiti(**values):
        result = original_graphiti(**values)
        processors.append(result)
        return result

    def assessor(**values):
        result = original_assessor(**values)
        assessors.append(result)
        return result

    monkeypatch.setattr(native_composition, 'AutonomousNativeEvidenceAssessor', assessor)
    monkeypatch.setattr(native_composition, 'NativePipeline', pipeline)
    monkeypatch.setattr(existing, '_arguments', arguments)
    monkeypatch.setattr(native_assessor_judgments, 'NativeAssessorJudgments', consumer)
    monkeypatch.setattr(native_composition, 'NativeGraphitiProcessor', graphiti)
    existing.test_native_composition_opens_factory_once_reopens_and_has_no_pre_effect(tmp_path, monkeypatch)
    assert len(consumers) == 2
    assert consumers[0].judgments.policy.route == 'TYPESAFE_JUDGMENT'
    assert consumers[0].localise is not None and consumers[0].read_localisation is not None
    assert len(processors) == 2
    assert all(callable(processor._runner._typed_proposal_verifier) for processor in processors)

    assert len(assessors) == 2 and all(callable(item._qualification) for item in assessors)
    assert all(item._judgments.semantic_witness_reader.__self__.resolution_reader.__self__.resolve_disagreements
               for item in assessors)
    assert all(item._judgments.semantic_witness_reader.__self__.resolution_reader.__self__.resolution_contract ==
               'newsroom.qualification-semantic-resolution.v2' for item in assessors)
    assert len(pipelines) == 2 and all(callable(item._publish.qualification_resolution_due) for item in pipelines)
    assert all(item._publish.qualification_resolution_due({}) is False for item in pipelines)
    from newsroom.control_plane.native_composition import ASSESSMENT_CONTRACT_VERSION
    from newsroom.control_plane.native_source_qualification_consumer import REPLAY_CONSUMER_VERSION
    from newsroom.control_plane.native_publication import NativePublicationContinuation
    monkeypatch.setattr(native_composition,'NativePublicationContinuation',NativePublicationContinuation)
    prior=ASSESSMENT_CONTRACT_VERSION.removesuffix('+'+REPLAY_CONSUMER_VERSION)
    assert prior!=ASSESSMENT_CONTRACT_VERSION
    recipe={'reason':'QUALIFICATION_ORIGINAL_RECIPE_UNSUPPORTED','failure_class':'QualificationHold',
        'candidate_id':'candidate','candidate_version_id':'version','graphiti_receipts':[{}],
        'intake_receipt_id':'retained-intake','assessment_contract_version':prior}
    assert all(item._publish.qualification_resolution_due(recipe) is True for item in pipelines)
