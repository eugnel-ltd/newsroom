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
def _services(tmp_path,monkeypatch,*,uncertain=False,failed_support=False,timeout=False):
    original,wire,rendering,binding,view,support,base,source,acquired=_case()
    candidate=N(candidate_id=base.candidate_id,version_id='version',
        governing_manifest=N(canonical_digest=binding['hypothesis_digest']))
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
        return NativeAssessmentExecution(canonical_json_bytes({'renderings':rendering['renderings']}).decode(),
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
            policy=localisation_policy(evidence_digest=digest_bytes(b'qualified fixture'),qualified=True),
            source_fence=fence,runner=runner,implementation_worktree_clean=True,
            clock=lambda:datetime(2026,10,5,tzinfo=UTC))
        consumer=NativeContextEnricher(judgments=judgments,localiser=localiser,
            objects=runtime.authority.objects,proof=runtime.proof,require_current=lambda:None)
        yield consumer,original,candidate,base,source,acquired,scope,usage,calls,local_calls


def test_context_accounting_replays_after_get_clock_change_without_new_calls(tmp_path,monkeypatch):
    with _services(tmp_path,monkeypatch)as(c,original,candidate,base,source,acquired,scope,usage,calls,local):
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
