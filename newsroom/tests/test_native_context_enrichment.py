"""An explicit context purpose never redispatches the original qualification."""
from copy import deepcopy
from datetime import UTC,datetime
import json
from types import SimpleNamespace as N
from contextlib import contextmanager

import pytest
from newsroom.authority import ObjectAdmissionRequest
from newsroom.authority.canonical import canonical_json_bytes,digest_bytes
from newsroom.control_plane.model_usage import ModelUsageService
from newsroom.control_plane.native_context_enrichment import NativeContextEnricher,ContextEnrichmentHold
from newsroom.control_plane.native_assessor_judgments import JudgedAssessment
from newsroom.control_plane.typesafe_judgment import TypesafeJudgment,judgment_policy
from newsroom.control_plane.native_claim_localisation import NativeClaimLocaliser,localisation_policy
from newsroom.control_plane.native_assessor import NativeAssessmentExecution
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.tests.test_native_runtime import _args
from newsroom.tests.test_native_context_materialisation import _case


@contextmanager
def _services(tmp_path,monkeypatch,*,uncertain=False,failed_support=False,timeout=False,email=False,localisation_version=None):
    from newsroom.tests.test_native_context_materialisation import BODY
    body = BODY.replace('2 chemicals under UK law.', '2 chemicals under UK law; contact help@example.org.') if email else BODY
    original,wire,rendering,binding,view,support,base,source,acquired=_case(body)
    candidate=N(candidate_id=base.candidate_id,version_id='version',
        governing_manifest=N(canonical_digest=binding['hypothesis_digest'], hypothesis_id=base.hypothesis_id))
    scope={'coverage':'COMPLETE','newness':'KNOWN_CHANGE','prior_scope':None,
        'current_scope':{'sources':[{'source_id':base.source_ids[0],'body':base.passages[0],
            'published_at':acquired.publication_time,'updated_at':acquired.source_updated_time,
            'retrieved_at':acquired.retrieval_time}]},'source_currentness':binding['source_currentness']}
    usage=ModelUsageService(str(tmp_path/'usage.sqlite3'));calls=[];local_calls=[]
    @contextmanager
    def fence(current,proof):
        assert current['content_digest']==base.digest
        yield
    def transport(request,**_kwargs):
        value=json.loads(request.data);calls.append(value);answers={}
        if timeout:raise TimeoutError('fixture context transport')
        for identity,question in value['questions'].items():
            choice=('UNCERTAIN'if uncertain else'INCLUDE')if 'INCLUDE'in question['criteria']else(
                'NO'if failed_support and identity.endswith(':entities')else
                'SUPPORTED'if identity.endswith(':support')else'YES')
            answers[identity]={'type':'choice','choice':choice,'confidence':1,
                'probabilities':{key:int(key==choice)for key in question['criteria']}}
        raw=canonical_json_bytes({'model':'jev-1.13.0','answers':answers,
            'usage':{'input_tokens':100,'output_tokens':30}})
        return 200,request.full_url,raw
    def runner(prompt):
        local_calls.append(json.loads(prompt))
        if email:
            rendering['renderings']['S1L2']['rendered_assertion_zh_hant_hk_fragments'] = ['政府正提出按', '法律管制2種化學物質；請聯絡', '。']
        return NativeAssessmentExecution(canonical_json_bytes({'renderings':[{'span_id':identity,**item} for identity,item in rendering['renderings'].items()]}).decode(),
            {'usage_basis':'PROVIDER_REPORTED','input_tokens':40,'output_tokens':20,'total_tokens':60})
    with open_native_runtime(**_args(tmp_path,monkeypatch))as runtime:
        admitted=runtime.authority.objects.admit(ObjectAdmissionRequest('evidence.record','original-source-qualification'),
            original.decision_record,proof=runtime.proof).admission
        original=JudgedAssessment(original.execution,original.decision_record,admitted.admission_id)
        judgments=TypesafeJudgment(usage=usage,objects=runtime.authority.objects,
            policy=judgment_policy(evidence_digest=digest_bytes(b'qualified fixture'),qualified=True),
            api_key=lambda:'fixture-not-live',source_fence=fence,transport=transport,
            implementation_worktree_clean=True,clock=lambda:datetime(2026,10,5,tzinfo=UTC))
        localiser=NativeClaimLocaliser(usage=usage,objects=runtime.authority.objects,
            policy=localisation_policy(evidence_digest=digest_bytes(b'qualified fixture'),qualified=True,
                **({} if localisation_version is None else {'version': localisation_version})),
            source_fence=fence,runner=runner,implementation_worktree_clean=True,
            clock=lambda:datetime(2026,10,5,tzinfo=UTC))
        consumer=NativeContextEnricher(judgments=judgments,localiser=localiser,
            objects=runtime.authority.objects,proof=runtime.proof,require_current=lambda:None)
        yield consumer,original,candidate,base,source,acquired,scope,usage,calls,local_calls


@pytest.mark.parametrize('version', ('newsroom.native-claim-localisation.v3', 'newsroom.native-claim-localisation.v4'))
def test_context_accounting_replays_after_get_clock_change_without_new_calls(tmp_path,monkeypatch,version):
    with _services(tmp_path,monkeypatch,localisation_version=version)as(c,original,candidate,base,source,acquired,scope,usage,calls,local):
        first=c.enrich(original,candidate,base,(source,),(acquired,),scope=scope)
        assert len(calls)==2 and len(local)==1
        later=deepcopy(scope);later['current_scope']['sources'][0]['retrieved_at']='2026-10-05T20:00:00Z'
        second=c.enrich(original,candidate,base,(source,),(acquired,),scope=later)
        assert first==second and len(calls)==2 and len(local)==1
        assert json.loads(first.execution.text)['package']['governed_claims'][0]==json.loads(original.execution.text)['package']['governed_claims'][0]
        with usage._connection()as db:
            assert db.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]==3


@pytest.mark.parametrize('uncertain,failed_support',[(True,False),(False,True)])
def test_unproved_context_is_retained_not_a_new_qualification_retry(tmp_path,monkeypatch,uncertain,failed_support):
    with _services(tmp_path,monkeypatch,uncertain=uncertain,failed_support=failed_support)as(c,o,ca,b,s,a,scope,usage,calls,local):
        for _ in range(2):
            with pytest.raises(ContextEnrichmentHold):c.enrich(o,ca,b,(s,),(a,),scope=scope)
        assert len(calls)==(1 if uncertain else 2)
        assert len(local)==(0 if uncertain else 1)


def test_unknown_context_attempt_never_dispatches_again(tmp_path,monkeypatch):
    with _services(tmp_path,monkeypatch,timeout=True)as(c,o,ca,b,s,a,scope,usage,calls,local):
        for _ in range(2):
            with pytest.raises(Exception):c.enrich(o,ca,b,(s,),(a,),scope=scope)
        assert len(calls)==1 and local==[]
        with usage._connection()as db:
            assert db.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]==1


def test_context_source_change_cannot_replace_retained_intent(tmp_path,monkeypatch):
    with _services(tmp_path,monkeypatch)as(c,o,ca,b,s,a,scope,usage,calls,local):
        c.enrich(o,ca,b,(s,),(a,),scope=scope)
        changed=N(**{**vars(a),'body':b'Changed Source'})
        with pytest.raises(ContextEnrichmentHold,match='SOURCE_BYTES'):
            c.enrich(o,ca,b,(s,),(changed,),scope=scope)
        assert len(calls)==2 and len(local)==1


def test_context_support_receives_actual_materialised_entity_surface(tmp_path, monkeypatch):
    with _services(tmp_path, monkeypatch) as (consumer, original, candidate, base, source, acquired, scope, usage, calls, local):
        consumer.enrich(original, candidate, base, (source,), (acquired,), scope=scope)
        support = calls[-1]['state']
        assert support['rendered_assertions']['S1L2'] == '政府正提出按UK法律管制2種化學物質。'
        assert 'Dan Jarvis' in support['rendered_assertions']['S1L3']
        assert all('rendered_assertions' not in call['state'] for call in calls[:-1])
        assert len(local) == 1


def test_closed_old_support_is_preserved_and_only_one_assembled_batch_is_new(tmp_path, monkeypatch):
    from copy import deepcopy
    from newsroom.control_plane.native_context_enrichment import SUPPORT_CONTRACT
    with _services(tmp_path, monkeypatch, failed_support=True) as (consumer, original, candidate, base, source, acquired, scope, usage, calls, local):
        batch = consumer._batch
        def legacy_batch(phase, state, questions, binding, candidate):
            if phase == SUPPORT_CONTRACT:
                state = {key:value for key,value in state.items() if key not in {'rendered_assertions','support_contract'}}
                questions = deepcopy(questions)
                for key in questions:
                    if key.endswith(':entities'):
                        identity = key.rsplit(':',1)[0]
                        questions[key]['instructions'] = f'Verify entities for {identity}: All rendered entities are exactly the supplied Source identities; no translated or added alias.'
                phase = 'CONTEXT_SUPPORT'
            return batch(phase, state, questions, binding, candidate)
        monkeypatch.setattr(consumer, '_batch', legacy_batch)
        with pytest.raises(ContextEnrichmentHold, match='SUPPORT_UNPROVEN'):
            consumer.enrich(original, candidate, base, (source,), (acquired,), scope=scope)
        with usage._connection() as db:
            old = db.execute("SELECT record_json FROM model_invocation_terminals ORDER BY invocation_id").fetchall()
        assert len(calls) == 2 and len(local) == 1
        monkeypatch.setattr(consumer, '_batch', batch)
        # The fixture still says NO: it is a new input contract, not a claimed
        # semantic pass. Replay must not ask either batch or renderer again.
        for _ in range(2):
            with pytest.raises(ContextEnrichmentHold, match='SUPPORT_UNPROVEN'):
                consumer.enrich(original, candidate, base, (source,), (acquired,), scope=scope)
        assert len(calls) == 3 and len(local) == 1
        with usage._connection() as db:
            retained = db.execute('SELECT record_json FROM model_invocation_terminals').fetchall()
            assert all(row in retained for row in old)


def test_unknown_old_support_never_receives_new_assembled_purpose(tmp_path, monkeypatch):
    from copy import deepcopy
    from newsroom.control_plane.native_context_enrichment import SUPPORT_CONTRACT
    with _services(tmp_path, monkeypatch) as (consumer, original, candidate, base, source, acquired, scope, usage, calls, local):
        batch = consumer._batch
        transport = consumer.judgments.transport
        def fail_support(request, **kwargs):
            if calls:
                calls.append(json.loads(request.data))
                raise TimeoutError('fixture unknown legacy support')
            return transport(request, **kwargs)
        monkeypatch.setattr(consumer.judgments, 'transport', fail_support)
        def old_batch(phase, state, questions, binding, candidate):
            if phase == SUPPORT_CONTRACT:
                state = {key:value for key,value in state.items() if key not in {'rendered_assertions','support_contract'}}
                questions = deepcopy(questions)
                for key in questions:
                    if key.endswith(':entities'):
                        identity = key.rsplit(':',1)[0]
                        questions[key]['instructions'] = f'Verify entities for {identity}: All rendered entities are exactly the supplied Source identities; no translated or added alias.'
                phase = 'CONTEXT_SUPPORT'
            return batch(phase, state, questions, binding, candidate)
        monkeypatch.setattr(consumer, '_batch', old_batch)
        with pytest.raises(Exception):
            consumer.enrich(original, candidate, base, (source,), (acquired,), scope=scope)
        monkeypatch.setattr(consumer, '_batch', batch)
        with pytest.raises(ContextEnrichmentHold, match='PRIOR_SUPPORT_UNSETTLED'):
            consumer.enrich(original, candidate, base, (source,), (acquired,), scope=scope)
        assert len(calls) == 2 and len(local) == 1
        with usage._connection() as db:
            assert db.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0] == 3


def test_typed_context_projects_literals_but_keeps_original_codec_and_authenticated_proof(tmp_path, monkeypatch):
    from newsroom.control_plane.native_assessor import AutonomousNativeEvidenceAssessor
    from newsroom.control_plane.native_assessor_judgments import NativeSemanticWitnesses
    with _services(tmp_path, monkeypatch, email=True) as (c, original, candidate, base, source, acquired, scope, usage, calls, local):
        first = c.enrich(original, candidate, base, (source,), (acquired,), scope=scope)
        witnesses = NativeSemanticWitnesses(judgments=c.judgments, proof=c.proof,
            candidate_for=lambda _identity: candidate, require_current=lambda: None)
        witnesses.rendering_reader = c.localiser.read_localisation
        assessment = AutonomousNativeEvidenceAssessor._validated_execution(first.execution, candidate, base,
            (source,), (acquired,), source_renderings=first.source_renderings, semantic_witness_reader=witnesses.read)
        claim = next(row for row in assessment.governed_claims if 'help@example.org' in row.claim)
        assert 'help@example.org' in claim.named_entities
        assert any(name == 'help@example.org' and kind == 'SOURCE_LITERAL' for name, kind, _ref in claim.named_entity_evidence)
        assert c.enrich(original, candidate, base, (source,), (acquired,), scope=scope) == first
        assert len(local) == 1 and len(calls) == 2
        assert json.loads(first.execution.text)['package']['governed_claims'][0] == json.loads(original.execution.text)['package']['governed_claims'][0]


def test_v4_context_reuses_retained_v3_rendering_without_new_calls(tmp_path, monkeypatch):
    with _services(tmp_path, monkeypatch, localisation_version='newsroom.native-claim-localisation.v3') as (
            c, original, candidate, base, source, acquired, scope, usage, calls, local):
        first = c.enrich(original, candidate, base, (source,), (acquired,), scope=scope)
        with usage._connection() as db:
            before = db.execute('SELECT record_json FROM model_invocation_terminals ORDER BY invocation_id').fetchall()
        old = c.localiser
        c.localiser = NativeClaimLocaliser(usage=usage, objects=old.objects,
            policy=localisation_policy(evidence_digest=digest_bytes(b'qualified v4 fixture'), qualified=True),
            source_fence=old.fence, runner=old.runner, implementation_worktree_clean=True, clock=old.clock)
        assert c.enrich(original, candidate, base, (source,), (acquired,), scope=scope) == first
        assert len(calls) == 2 and len(local) == 1
        with usage._connection() as db:
            assert db.execute('SELECT record_json FROM model_invocation_terminals ORDER BY invocation_id').fetchall() == before
