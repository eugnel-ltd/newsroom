"""Provider-free Source-bound semantic witness and transitive admission contracts."""
import json
from dataclasses import replace
import pytest
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane.evidence import QualificationEvidence, Evid012QualificationTest, evidence_package_value
from newsroom.tests.test_increment10_editorial import _ready_package
from newsroom.tests.assessor_fixture_support import candidate_fixture


def test_semantic_witness_reference_is_optional_and_preserves_legacy_canonical_bytes(tmp_path):
    connection, _port, candidate = candidate_fixture(tmp_path)
    try:
        package = _ready_package(candidate)[1]
        old = evidence_package_value(package)
        assert all('semantic_witness_ref' not in q for q in old['qualification_evidence'])
        qualification = package.qualification_evidence[0]
        ref = tuple(sorted({'contract':'newsroom.qualification-semantic-witness.v1',
            'invocation_id':'sha256:'+'a'*64,'raw_admission_id':'11111111-1111-4111-8111-111111111111',
            'receipt_admission_id':'22222222-2222-4222-8222-222222222222','question_id':'criterion'}.items()))
        verified = replace(qualification, semantic_witness_ref=ref)
        assert dict(verified.semantic_witness_ref)['question_id'] == 'criterion'
        changed = evidence_package_value(replace(package, qualification_evidence=(verified,)))
        assert changed['qualification_evidence'][0]['semantic_witness_ref'] == dict(ref)
        restored = replace(verified, semantic_witness_ref=())
        assert evidence_package_value(replace(package, qualification_evidence=(restored,))) == old
    finally:
        connection.close()


from contextlib import contextmanager
from types import SimpleNamespace
from newsroom.control_plane.native_assessor_judgments import NativeSemanticWitnesses, _semantic_witness_inputs
from newsroom.control_plane.admission import qualification_relation_is_admitted, DeterministicWriteAdmission
from newsroom.tests.test_typesafe_judgment import _case as typesafe_case


@contextmanager
def _witness_case(tmp_path, monkeypatch, *, choice='YES', fault=None):
    connection, _port, candidate = candidate_fixture(tmp_path)
    package = _ready_package(candidate)[1]
    claim = package.governed_claims[0]
    qualification = package.qualification_evidence[0]
    binding={'candidate_id':candidate.candidate_id,'candidate_version_id':candidate.version_id,
        'hypothesis_digest':candidate.governing_manifest.canonical_digest,
        'content_digest':__import__('newsroom.increment10.evidence',fromlist=['_base_package'])._base_package(package).digest,
        'coverage':'COMPLETE','newness':'KNOWN_CHANGE','current_scope':{'sources':[
            {'source_id':sid,'body':body}for sid,body in zip(package.source_ids,package.passages,strict=True)]},
        'prior_scope':{'sources':[{'source_id':package.source_ids[0],'body':'An older complete Source statement.'}]}}
    binding['evidence_package_digest']=binding['content_digest']
    with typesafe_case(tmp_path,monkeypatch) as (engine,_inputs,usage,calls,_raw):
        def transport(request,**_kw):
            calls.append(request)
            if fault=='timeout':raise TimeoutError('synthetic unknown transport')
            criteria=json.loads(request.data)['questions']['criterion']['criteria']
            value={'model':'jev-1.13.0','answers':{'criterion':{'type':'choice','choice':choice,
                'confidence':1,'probabilities':{key:int(key==choice)for key in criteria}},
                },'usage':{'input_tokens':40,'output_tokens':10}}
            if fault=='answers':value['answers']={}
            if fault=='unreported':value.pop('usage')
            return 200,request.full_url,json.dumps(value).encode()
        engine.transport=transport
        stopped=[False]
        def current():
            if stopped[0]:raise ValueError('current Source/rights/stop no longer holds')
        verifier=NativeSemanticWitnesses(judgments=engine,candidate_for=lambda _:candidate,proof=_inputs['proof'],require_current=current)
        try:yield verifier,qualification,claim,package,binding,usage,calls,stopped
        finally:connection.close()


@pytest.mark.parametrize('scenario',['read-replay','write-admission','long-parent','legacy','lexical-false-negative'])
def test_authenticated_semantic_witness_positive_boundaries(tmp_path,monkeypatch,scenario):
    with _witness_case(tmp_path,monkeypatch)as(verifier,q,c,p,b,usage,calls,stopped):
        if scenario=='legacy':
            assert qualification_relation_is_admitted(q,c,p)
            assert not calls
            return
        if scenario=='lexical-false-negative':
            from newsroom.increment10.evidence import _base_package
            statement='All schools now receive fully funded practical education materials.'
            c=replace(c,claim=statement,supporting_excerpt=statement)
            p=replace(p,passages=(statement,),observation_digests=(digest_bytes(statement.encode()),))
            q=QualificationEvidence(Evid012QualificationTest.HOUSEHOLD_PRACTICAL_EFFECT,c.claim_id,'qualification-1',
                (('domain','EDUCATION'),('event_polarity','AFFIRMED'),('effect_relation','MATERIAL_PRACTICAL_EFFECT'),
                 ('material_relation_span',statement),('practical_effect',statement)))
            b={**b,'current_scope':{'sources':[{'source_id':p.source_ids[0],'body':statement}]},
                'content_digest':_base_package(p).digest,'evidence_package_digest':_base_package(p).digest}
            assert not qualification_relation_is_admitted(q,c,p)
        if scenario=='long-parent':
            body=p.passages[0]+'\n'+'Full parent context preserved. '*30
            from newsroom.increment10.evidence import _base_package
            p=replace(p,passages=(body,),observation_digests=(digest_bytes(body.encode()),))
            b={**b,'current_scope':{'sources':[{'source_id':p.source_ids[0],'body':body}]},
                'content_digest':_base_package(p).digest,'evidence_package_digest':_base_package(p).digest}
        ref=verifier.evaluate(q,c,p,b);verified=replace(q,semantic_witness_ref=ref)
        assert qualification_relation_is_admitted(verified,c,p,semantic_witness_reader=verifier.read)
        assert qualification_relation_is_admitted(verified,c,p,semantic_witness_reader=verifier.read)
        assert verifier.evaluate(q,c,p,b)==ref and len(calls)==1
        if scenario=='write-admission':
            from newsroom.control_plane.evidence import EvidenceGateEvidence, EVIDENCE_GATE_POLICY_VERSION
            _passage, _original, records = _ready_package(__import__('newsroom.tests.assessor_fixture_support',fromlist=['candidate_value']).candidate_value())
            # The surrounding readiness gates are synthetic fixture evidence,
            # not a live Source/currentness assertion.
            changed=replace(p,qualification_evidence=(verified,), freshness_result='PASS',integrity_result='PASS',
                resolved_evidence_records=tuple((r['record_id'],digest_bytes(canonical_json_bytes(r)))for r in records),
                evidence_gate_results=tuple((gate,'PASS')for gate in ('CLAIM_TRACEABILITY','EVIDENCE_SUFFICIENCY','SOURCE_AUTHORITY')),
                evidence_gate_evidence=tuple(EvidenceGateEvidence(gate,'PASS',tuple(c.claim_id for c in p.governed_claims),EVIDENCE_GATE_POLICY_VERSION)
                    for gate in ('CLAIM_TRACEABILITY','EVIDENCE_SUFFICIENCY','SOURCE_AUTHORITY')))
            no_reader=DeterministicWriteAdmission().decide_candidate_identity(candidate_id=p.candidate_id,
                hypothesis_id=p.hypothesis_id,package=changed,decided_at='2026-10-05T12:00:00Z')
            assert no_reader.decision=='HOLD'
            decision=DeterministicWriteAdmission(semantic_witness_reader=verifier.read).decide_candidate_identity(
                candidate_id=p.candidate_id,hypothesis_id=p.hypothesis_id,package=changed,decided_at='2026-10-05T12:00:00Z')
            assert decision.decision=='WRITE_READY'


@pytest.mark.parametrize('fault',['NO','UNCERTAIN','timeout','unreported','answers','candidate','source','fields',
    'question','mock-bool','current-stop','partial-parent'])
def test_semantic_witness_negative_boundaries_are_fail_closed(tmp_path,monkeypatch,fault):
    with _witness_case(tmp_path,monkeypatch,choice=fault if fault in {'NO','UNCERTAIN'}else'YES',
            fault=fault)as(verifier,q,c,p,b,usage,calls,stopped):
        if fault in {'NO','UNCERTAIN','timeout','unreported','answers'}:
            for _ in range(2):
                with pytest.raises((ValueError,RuntimeError,TimeoutError)):
                    verifier.evaluate(q,c,p,b)
            assert len(calls)==1
            return
        if fault=='partial-parent':
            c=replace(c,claim=c.claim[4:],supporting_excerpt=c.claim[4:])
            with pytest.raises(ValueError):verifier.evaluate(q,c,p,b)
            assert calls==[]
            return
        ref=verifier.evaluate(q,c,p,b);verified=replace(q,semantic_witness_ref=ref)
        if fault=='candidate':p=replace(p,candidate_id='foreign-candidate')
        elif fault=='source':p=replace(p,passages=('Altered source body.',))
        elif fault=='fields':verified=replace(verified,test_evidence=tuple((k,v+' altered')if k=='material_relation_span'else(k,v)for k,v in q.test_evidence))
        elif fault=='question':verified=replace(verified,semantic_witness_ref=tuple((k,'wrong')if k=='question_id'else(k,v)for k,v in ref))
        elif fault=='current-stop':stopped[0]=True
        if fault=='mock-bool':
            assert not qualification_relation_is_admitted(verified,c,p,semantic_witness_reader=lambda *_:True)
        else:
            with pytest.raises(ValueError):verifier.read(verified,c,p)
        assert len(calls)==1


@contextmanager
def _selected_qualification_case(tmp_path, monkeypatch, *, malformed_rendering=False, fault=None, candidate_binding=None):
    """Genuine disposable QA/TypeSafe/localiser ledgers and CAS; synthetic answers."""
    from newsroom.tests.test_native_assessor_judgments import _case as typed_case
    from newsroom.control_plane.native_assessor_judgments import JudgmentFallback
    from newsroom.control_plane.native_source_qualification import NativeSourceQualifier, qualification_policy
    from newsroom.control_plane.native_claim_localisation import NativeClaimLocaliser, localisation_policy
    from newsroom.control_plane.native_assessor import NativeAssessmentExecution
    from newsroom.control_plane.writer import WriterCliExecution
    from newsroom.authority.canonical import digest_canonical
    from datetime import UTC, datetime
    headline={'actor':'Schools now receive SEND practical education materials.',
        'date':'Schools could receive practical education materials next year.',
        'value':'Schools now receive £0.50 for practical education materials.',
        'terms':'Schools now receive SEND training.',
        'quoted-terms':'Schools now receive SEND training.',
        'year':'Schools will receive fully funded education materials next year.'}.get(fault,
        'Schools now receive fully funded practical education materials.')
    supporting = ('The SEND programme is delivered by Ambition Institute and charity Dingley’s Promise.'
        if fault=='terms' else 'The SEND programme is delivered by charity Dingley’s Promise, under the name “Dingley’s Promise”.'
        if fault=='quoted-terms' else 'The materials are now available to households.')
    body=headline+'\n'+supporting
    with typed_case(tmp_path,monkeypatch,max_prompt_bytes=7000,body=body) as (consumer,service,candidate,base,source,acquired,usage,jev_calls):
        if candidate_binding is not None:
            candidate=candidate_binding
            base=replace(base,candidate_id=candidate.candidate_id,hypothesis_id=candidate.governing_manifest.hypothesis_id,
                signal_ids=tuple(row.signal_id for row in candidate.governing_manifest.lead_signal_bindings),
                lead_ids=tuple(row.lead_id for row in candidate.governing_manifest.lead_signal_bindings))
            @contextmanager
            def current_fence(binding,proof):
                assert binding['content_digest']==base.digest
                yield
            service.fence=current_fence
            acquired.publication_time=acquired.source_updated_time='2026-09-08T12:00:00Z'
            acquired.retrieval_time='2026-09-08T12:01:00Z'
        else:
            candidate.governing_manifest.hypothesis_id=base.hypothesis_id
        acquired.geography='Hong Kong';acquired.language='en-GB';acquired.body_origin=''
        scope={'coverage':'COMPLETE','newness':'KNOWN_CHANGE','prior_scope':{'revision_digest':digest_bytes(b'prior')},
            'current_scope':{'sources':[{'source_id':source.unit.source_id,'body':body,
                'published_at':acquired.publication_time,'updated_at':acquired.source_updated_time,'retrieved_at':acquired.retrieval_time}]}}
        consumer.scope_for=lambda *_:scope
        fallback=consumer.assess(candidate,base,(source,),(acquired,))
        assert isinstance(fallback,JudgmentFallback) and fallback.reason=='JUDGMENT_INPUT_BOUND',fallback
        wire={'package':{'select_new_information':True,'governed_claims':[
            {'claim_role':role,'status':'CONFIRMED_FACT','source_range':{'first_span_id':span,'last_span_id':span},
             'rendered_assertion_zh_hant_hk_fragments':[render], 'factual_localisations':[],'quotation_source_keys':[]}
            for role,span,render in [('HEADLINE','S1L1','Teaching materials available.' if malformed_rendering else '學校現時獲提供全額資助嘅實用教育教材。'),
                                    ('SUBSTANTIVE','S1L2','教材現時可供家庭使用。')]],
            'qualification_evidence':[{'test':'HOUSEHOLD_PRACTICAL_EFFECT','claim_index':0,
                'test_evidence':{'domain':'EDUCATION','event_polarity':'AFFIRMED','effect_relation':'MATERIAL_PRACTICAL_EFFECT',
                    'material_relation_span_source_lookup_key':body.split('\n')[0], 'practical_effect_source_lookup_key':body.split('\n')[0]}}],
            'selection_rationale':'A new practical education provision.','geography':['Hong Kong'],'categories':['Education and campuses'],'explicit_exclusions':[]}}
        if fault=='qa-NO':
            from newsroom.tests.test_native_source_qualification import WIRE
            wire=WIRE
        if fault=='quoted-terms':
            acquired.publisher='Dingley’s Promise'
            wire['package']['governed_claims'][1]['quotation_source_keys']=['Dingley’s Promise']
        qa_calls=[];render_calls=[]
        def qa_runner(_prompt):
            qa_calls.append(_prompt)
            return NativeAssessmentExecution(canonical_json_bytes(wire).decode(),{'usage_basis':'PROVIDER_REPORTED','input_tokens':40,'output_tokens':10,'total_tokens':50})
        old_transport=service.transport
        def transport(request,**kwargs):
            data=json.loads(request.data)
            if 'criterion' not in data['questions']:return old_transport(request,**kwargs)
            jev_calls.append(data)
            choice='NO' if fault=='NO' else 'YES'
            return 200,request.full_url,json.dumps({'model':'jev-1.13.0','answers':{'criterion':{'type':'choice','choice':choice,
                'confidence':1,'probabilities':{key:int(key==choice)for key in data['questions']['criterion']['criteria']} }},
                'usage':{'input_tokens':40,'output_tokens':10}}).encode()
        service.transport=transport
        def render_runner(prompt):
            render_calls.append(prompt)
            if fault in {'terms','year','quoted-terms'}:
                requested=json.loads(prompt)['claims']
                rows=[]
                for index,row in requested.items():
                    names=row['entities']
                    fragments=((['學校現時獲提供']+['培訓。']*len(names))if index=='0'else(['計劃由']+['提供支援。']*len(names))) if names else ['學校將於2027年獲提供全額資助教育教材。'if index=='0'else'教材現時可供家庭使用。']
                    if fault=='quoted-terms' and index=='1':
                        fragments=['','計劃由','提供。名稱為「','」。']
                    rows.append({'span_id':index,'rendered_assertion_zh_hant_hk_fragments':fragments,
                        'factual_localisations':[{'source_lookup_key':a,'rendered_expression':b}for a,b in row.get('source_derived_facts',[])],
                        'quotation_source_keys':['Dingley’s Promise']if fault=='quoted-terms'and index=='1'else[]})
                return WriterCliExecution(json.dumps({'renderings':rows},ensure_ascii=False),
                    {'usage_basis':'PROVIDER_REPORTED','input_tokens':40,'output_tokens':10,'total_tokens':50})
            return WriterCliExecution(json.dumps({'renderings':[
                {'span_id':str(i),'rendered_assertion_zh_hant_hk_fragments':[render],
                 'factual_localisations':[],'quotation_source_keys':[]}for i,render in enumerate(
                    ('學校現時獲提供全額資助嘅實用教育教材。','教材現時可供家庭使用。'))]},ensure_ascii=False),
                {'usage_basis':'PROVIDER_REPORTED','input_tokens':40,'output_tokens':10,'total_tokens':50})
        localiser=NativeClaimLocaliser(usage=usage,objects=service.objects,policy=localisation_policy(
            evidence_digest=digest_bytes(b'synthetic qualified localiser'),qualified=True),source_fence=service.fence,
            runner=render_runner,implementation_worktree_clean=True,clock=lambda:datetime(2026,10,5,tzinfo=UTC))
        identities=dict(candidate_id=candidate.candidate_id,hypothesis_digest=candidate.governing_manifest.canonical_digest,
            evidence_package_digest=base.digest,proof=consumer.proof)
        qualifier=NativeSourceQualifier(usage=usage,objects=service.objects,policy=qualification_policy(
            evidence_digest=digest_bytes(b'synthetic qualified source QA'),qualified=True),source_fence=service.fence,
            judgments=service,runner=qa_runner,implementation_worktree_clean=True,clock=lambda:datetime(2026,10,5,tzinfo=UTC))
        original=qualifier.assess(candidate,base,(source,),(acquired,),fallback,scope=scope,proof=consumer.proof)
        witness=NativeSemanticWitnesses(judgments=service,candidate_for=lambda _:candidate,proof=consumer.proof,require_current=lambda:None)
        from newsroom.control_plane.native_source_qualification_consumer import NativeQualifiedSourceConsumer
        post=NativeQualifiedSourceConsumer(qualifier,semantic_witnesses=witness,
            localise=lambda request:localiser.localise(request,**identities),
            read_localisation=lambda reference,request:localiser.read_localisation(reference,request,**identities))
        witness.parent_reader=post.read_semantic_parent
        yield post,witness,original,candidate,base,source,acquired,scope,consumer.proof,usage,qa_calls,jev_calls,render_calls


@pytest.mark.parametrize('malformed_rendering',[False,True])
def test_selected_sourceqa_composes_authenticated_witness_then_optional_rendering_once(tmp_path,monkeypatch,malformed_rendering):
    from newsroom.control_plane.native_assessor import AutonomousNativeEvidenceAssessor
    from newsroom.control_plane.native_source_qualification_replay import read_current_result as original_read
    def read_current_result(q,c,b,s,a,*,scope,proof):
        return q.compose_selected(original_read(q.qualifier,c,b,s,a,scope=scope,proof=proof),c,b,s,a,proof=proof)
    with _selected_qualification_case(tmp_path,monkeypatch,malformed_rendering=malformed_rendering)as(q,w,old,c,b,s,a,scope,proof,usage,qa,jev,render):
        old_bytes=old.decision_record
        selected=read_current_result(q,c,b,(s,),(a,),scope=scope,proof=proof)
        result=AutonomousNativeEvidenceAssessor._validated_execution(selected.execution,c,b,(s,),(a,),
            semantic_witnesses=selected.semantic_witnesses,semantic_witness_reader=w.read,source_renderings=selected.source_renderings)
        assert result.qualification_evidence[0].semantic_witness_ref
        assert len(result.governed_claims)==2 and len(qa)==1 and len(jev)==2
        assert len(render)==int(malformed_rendering)
        assert read_current_result(q,c,b,(s,),(a,),scope=scope,proof=proof)==selected
        assert len(qa)==1 and len(jev)==2 and len(render)==int(malformed_rendering)
        assert old.decision_record==old_bytes
        # Original producer raw/schema/purpose is immutable; no QA redispatch.
        import sqlite3
        with sqlite3.connect(usage.path)as db:
            assert db.execute("SELECT COUNT(*)FROM model_invocation_allocations WHERE route='NATIVE_SOURCE_QUALIFICATION'").fetchone()[0]==1


def test_semantic_parent_forgery_denies_before_new_witness_or_rendering(tmp_path,monkeypatch):
    with _selected_qualification_case(tmp_path,monkeypatch)as(q,w,old,c,b,s,a,scope,proof,usage,qa,jev,render):
        value=json.loads(old.decision_record)
        value['qualification_reference']['invocation_id']='sha256:'+'0'*64
        forged=replace(old,decision_record=canonical_json_bytes(value))
        with pytest.raises((ValueError,RuntimeError)):
            q.compose_selected(forged,c,b,(s,),(a,),proof=proof)
        assert len(qa)==len(jev)==1 and render==[]


def test_governed_package_retention_read_and_final_admission_reauthenticate_semantic_reference(tmp_path,monkeypatch):
    from newsroom.tests.test_increment10_editorial import _open_editorial_system,_evidence_facade,_native,_decision,_record_decision
    from newsroom.tests.test_increment10_ingress import _candidate,_receive
    from newsroom.increment10.ingress import open_evidence_intake_ingress
    from newsroom.tests.authority_helpers import proof
    from newsroom.increment10.evidence import _base_package,EvidencePackageError
    from newsroom.authority import ObjectAdmissionRequest, AggregateId
    from newsroom.increment10.editorial import StoryVersionRequest
    with _witness_case(tmp_path,monkeypatch)as(w,_q,_c,_p,b,usage,calls,stopped):
        candidate_path=tmp_path/'selected';candidate_path.mkdir()
        connection,port,candidate=_candidate(candidate_path)
        ingress=open_evidence_intake_ingress(tmp_path/'selected-ingress.sqlite3')
        ack=_receive(ingress,connection,port,candidate,request_id='semantic-witness')
        system,registries=_open_editorial_system(tmp_path/'selected-objects.sqlite3')
        evidence=_evidence_facade(system,ingress,registries)
        passage,package,records=_ready_package(candidate)
        claim=package.governed_claims[0];qualification=package.qualification_evidence[0]
        w.candidate_for=lambda _:candidate
        binding={**b,'candidate_id':candidate.candidate_id,'candidate_version_id':candidate.version_id,
            'hypothesis_digest':candidate.governing_manifest.canonical_digest,
            'content_digest':_base_package(package).digest,'evidence_package_digest':_base_package(package).digest}
        ref=w.evaluate(qualification,claim,package,binding)
        package=replace(package,qualification_evidence=(replace(qualification,semantic_witness_ref=ref),))
        records=tuple({**r,'semantic_witness_ref':dict(ref)}if r['record_type']=='QUALIFICATION_EVIDENCE' else r for r in records)
        source=system.objects.admit(ObjectAdmissionRequest('evidence.source','source'),passage.encode(),proof=proof()).admission
        ids=tuple(system.objects.admit(ObjectAdmissionRequest('evidence.record','record-'+str(i)),canonical_json_bytes(r),proof=proof()).admission.admission_id
            for i,r in enumerate(records))
        connection.execute('BEGIN')
        try:
            with pytest.raises(EvidencePackageError,match='semantic witness'):
                evidence.retain(package,receipt_id=ack.receipt_id,candidate_port=port,source_admission_ids=(source.admission_id,),record_admission_ids=ids,proof=proof())
            evidence.semantic_witness_reader=w.read
            retained=evidence.retain(package,receipt_id=ack.receipt_id,candidate_port=port,source_admission_ids=(source.admission_id,),record_admission_ids=ids,proof=proof())
            assert evidence.read(retained.package_admission_id,candidate_port=port,proof=proof())==retained
            # Complete governed package -> controller decision -> final Story
            # admission, entirely within disposable authority fixtures.
            native=_native(system,evidence,registries)
            decision_ref=_record_decision(system,_decision(retained,source.admission_id))
            story_receipt,story=native.admit_story_version(StoryVersionRequest(AggregateId.new(),0,'semantic-story'),
                package_admission_id=retained.package_admission_id,decision_reference=decision_ref,
                candidate_port=port,proof=proof())
            assert story.copy.body and story_receipt.event_id
            evidence.semantic_witness_reader=lambda *_:True
            with pytest.raises(EvidencePackageError,match='semantic witness'):
                evidence.read(retained.package_admission_id,candidate_port=port,proof=proof())
            evidence.semantic_witness_reader=w.read;stopped[0]=True
            with pytest.raises(ValueError,match='current Source'):
                evidence.read(retained.package_admission_id,candidate_port=port,proof=proof())
            assert len(calls)==1
        finally:
            connection.rollback();connection.close();ingress.close();system.close()


@pytest.mark.parametrize('fault',['qa-NO','NO','actor','date','value'])
def test_sourceqa_consumer_never_requalifies_no_or_pays_for_unsupported_rendering(tmp_path,monkeypatch,fault):
    from newsroom.control_plane.native_source_qualification import QualificationHold
    from newsroom.control_plane.native_source_qualification_replay import read_current_result as original_read
    def read_current_result(q,c,b,s,a,*,scope,proof):
        return q.compose_selected(original_read(q.qualifier,c,b,s,a,scope=scope,proof=proof),c,b,s,a,proof=proof)
    with _selected_qualification_case(tmp_path,monkeypatch,malformed_rendering=True,fault=fault)as(q,w,old,c,b,s,a,scope,proof,usage,qa,jev,render):
        original=old.decision_record
        if fault=='qa-NO':
            assert read_current_result(q,c,b,(s,),(a,),scope=scope,proof=proof)==old
            assert read_current_result(q,c,b,(s,),(a,),scope=scope,proof=proof)==old
            assert len(qa)==len(jev)==1
        else:
            for _ in range(2):
                with pytest.raises(ValueError):
                    read_current_result(q,c,b,(s,),(a,),scope=scope,proof=proof)
            assert len(qa)==1 and len(jev)==2
        assert render==[] and old.decision_record==original


@pytest.mark.parametrize('capability',['terms','year'])
def test_source_rendering_capabilities_are_source_bound_and_reauthenticated(tmp_path,monkeypatch,capability):
    from newsroom.control_plane.native_assessor import AutonomousNativeEvidenceAssessor
    from newsroom.control_plane.native_source_qualification_replay import read_current_result as original_read
    def read_current_result(q,c,b,s,a,*,scope,proof):
        return q.compose_selected(original_read(q.qualifier,c,b,s,a,scope=scope,proof=proof),c,b,s,a,proof=proof)
    from newsroom.control_plane.admission import source_rendering_is_admitted
    with _selected_qualification_case(tmp_path,monkeypatch,malformed_rendering=True,fault=capability)as(q,w,old,c,b,s,a,scope,proof,usage,qa,jev,render):
        selected=read_current_result(q,c,b,(s,),(a,),scope=scope,proof=proof)
        result=AutonomousNativeEvidenceAssessor._validated_execution(selected.execution,c,b,(s,),(a,),
            semantic_witnesses=selected.semantic_witnesses,source_renderings=selected.source_renderings,semantic_witness_reader=w.read)
        result=replace(b,governed_claims=result.governed_claims,qualification_evidence=result.qualification_evidence,
            substantive_new_information=result.substantive_new_information)
        assert len(qa)==1 and len(jev)==2 and len(render)==1
        affected=[claim for claim in result.governed_claims if claim.source_rendering_ref]
        assert affected and all(source_rendering_is_admitted(claim,result,semantic_witness_reader=w.read)for claim in affected)
        assert not source_rendering_is_admitted(affected[0],result)
        assert not source_rendering_is_admitted(affected[0],result,semantic_witness_reader=lambda *_:True)
        if capability=='terms':
            assert {'SEND','Ambition Institute','Dingley’s Promise'}=={name for claim in affected for name in claim.named_entities}
        else:
            assert ('next year','2027年') in affected[0].localised_factual_expressions
            wrong=replace(affected[0],rendered_assertion_zh_hant_hk=affected[0].rendered_assertion_zh_hant_hk.replace('2027','2028'),
                localised_factual_expressions=(('next year','2028年'),))
            with pytest.raises(ValueError,match='derivation'):
                w.read(None,wrong,result)
        forged_ref=tuple((key,'sha256:'+'0'*64)if key=='invocation_id'else(key,value)for key,value in affected[0].source_rendering_ref)
        with pytest.raises(ValueError):w.read(None,replace(affected[0],source_rendering_ref=forged_ref),result)
        assert read_current_result(q,c,b,(s,),(a,),scope=scope,proof=proof)==selected
        assert len(qa)==1 and len(jev)==2 and len(render)==1


@pytest.mark.parametrize('capability',['terms','year'])
def test_source_rendering_full_governed_retention_and_final_write_chain(tmp_path,monkeypatch,capability):
    from newsroom.tests.test_increment10_editorial import _open_editorial_system,_evidence_facade,_native,_decision,_record_decision
    from newsroom.tests.test_increment10_ingress import _candidate,_receive
    from newsroom.increment10.ingress import open_evidence_intake_ingress
    from newsroom.tests.authority_helpers import proof
    from newsroom.authority import ObjectAdmissionRequest,AggregateId
    from newsroom.increment10.editorial import StoryVersionRequest,EditorialPolicyDecision
    from newsroom.increment10.evidence import EvidencePackageError
    from newsroom.control_plane.native_assessor import AutonomousNativeEvidenceAssessor
    from newsroom.control_plane.native_evidence import NativeEvidenceController
    from newsroom.control_plane.native_source_qualification_replay import read_current_result
    candidate_path=tmp_path/'selected';candidate_path.mkdir()
    connection,port,candidate=_candidate(candidate_path)
    ingress=open_evidence_intake_ingress(tmp_path/'selected-ingress.sqlite3')
    ack=_receive(ingress,connection,port,candidate,request_id='typed-rendering')
    with _selected_qualification_case(tmp_path,monkeypatch,malformed_rendering=True,fault=capability,candidate_binding=candidate)as(q,w,old,c,b,s,a,scope,auth,usage,qa,jev,render):
        original=read_current_result(q.qualifier,c,b,(s,),(a,),scope=scope,proof=auth)
        selected=q.compose_selected(original,c,b,(s,),(a,),proof=auth)
        assessment=AutonomousNativeEvidenceAssessor._validated_execution(selected.execution,c,b,(s,),(a,),
            semantic_witnesses=selected.semantic_witnesses,source_renderings=selected.source_renderings,semantic_witness_reader=w.read)
        package=replace(b,governed_claims=assessment.governed_claims,qualification_evidence=assessment.qualification_evidence,
            substantive_new_information=assessment.substantive_new_information,selection_rationale=assessment.selection_rationale,
            geography=assessment.geography,categories=assessment.categories)
        records=NativeEvidenceController._records(b,package,(s,),(a,),assessment)
        system,registries=_open_editorial_system(tmp_path/'selected-objects.sqlite3')
        evidence=_evidence_facade(system,ingress,registries)
        source=system.objects.admit(ObjectAdmissionRequest('evidence.source','source'),a.body,proof=proof()).admission
        record_ids=tuple(system.objects.admit(ObjectAdmissionRequest('evidence.record',str(index)),canonical_json_bytes(r),proof=proof()).admission.admission_id
            for index,r in enumerate(records))
        connection.execute('BEGIN')
        try:
            evidence.semantic_witness_reader=w.read
            retained=evidence.retain(package,receipt_id=ack.receipt_id,candidate_port=port,source_admission_ids=(source.admission_id,),record_admission_ids=record_ids,proof=proof())
            assert evidence.read(retained.package_admission_id,candidate_port=port,proof=proof())==retained
            template=_decision(retained,source.admission_id)
            # Synthetic complete currentness/integrity decisions for this exact
            # admitted Source; no live Source or rights acceptance is claimed.
            decision=EditorialPolicyDecision.create(**{key:getattr(template,key)for key in
                ('candidate_version_id','candidate_version_digest','governing_manifest_digest','package_admission_id','package_digest','policy_bundle_digest','evaluated_at','evidence_gate_results')},
                currentness=tuple(replace(row,source_id=s.unit.source_id)for row in template.currentness),
                integrity=tuple(replace(row,source_id=s.unit.source_id,source_digest=a.body_digest)for row in template.integrity))
            reference=_record_decision(system,decision)
            receipt,story=_native(system,evidence,registries).admit_story_version(StoryVersionRequest(AggregateId.new(),0,'typed-rendering-story'),
                package_admission_id=retained.package_admission_id,decision_reference=reference,candidate_port=port,proof=proof())
            assert story.copy.body and receipt.event_id
            if capability=='terms':
                from newsroom.control_plane.native_source_qualification_consumer import validate_source_literal_copy
                from newsroom.control_plane.writer import validate_writer_copy
                original_results=validate_writer_copy(story.copy,retained.package)
                corrected=validate_source_literal_copy(story.copy,retained.package,semantic_witness_reader=w.read)
                assert [row for row in corrected if row.validator!='QUOTE_FIDELITY']==[row for row in original_results if row.validator!='QUOTE_FIDELITY']
                assert next(row for row in validate_source_literal_copy(story.copy,retained.package)if row.validator=='QUOTE_FIDELITY').result=='FAIL'
                for suffix in ('\n“unattributed quotation”','\n“unbalanced quotation','\npolicy’s invention'):
                    wrong=replace(story.copy,body=story.copy.body+suffix)
                    assert next(row for row in validate_source_literal_copy(wrong,retained.package,semantic_witness_reader=w.read)
                        if row.validator=='QUOTE_FIDELITY').result=='FAIL'
            evidence.semantic_witness_reader=lambda *_:True
            with pytest.raises(EvidencePackageError):evidence.read(retained.package_admission_id,candidate_port=port,proof=proof())
            assert len(qa)==1 and len(jev)==2 and len(render)==1
        finally:
            connection.rollback();connection.close();ingress.close();system.close()


def test_source_literal_quote_wrapper_preserves_supported_outer_quote_and_attribution(tmp_path,monkeypatch):
    from newsroom.control_plane.native_source_qualification_replay import read_current_result
    from newsroom.control_plane.native_assessor import AutonomousNativeEvidenceAssessor
    from newsroom.control_plane.native_source_qualification_consumer import validate_source_literal_copy
    from newsroom.control_plane.writer import WriterCopy,validate_writer_copy,required_surface_copy
    with _selected_qualification_case(tmp_path,monkeypatch,malformed_rendering=True,fault='quoted-terms')as(q,w,old,c,b,s,a,scope,proof,usage,qa,jev,render):
        original=read_current_result(q.qualifier,c,b,(s,),(a,),scope=scope,proof=proof)
        selected=q.compose_selected(original,c,b,(s,),(a,),proof=proof)
        assessment=AutonomousNativeEvidenceAssessor._validated_execution(selected.execution,c,b,(s,),(a,),
            semantic_witnesses=selected.semantic_witnesses,source_renderings=selected.source_renderings,semantic_witness_reader=w.read)
        package=replace(b,governed_claims=assessment.governed_claims,qualification_evidence=assessment.qualification_evidence,
            substantive_new_information=assessment.substantive_new_information)
        title,body,links=required_surface_copy(package,paragraphs=True,context_preserving=True)
        copy=WriterCopy(title,body,'newsroom.offline-exact-copy.v3',package.digest,links)
        before=canonical_json_bytes(evidence_package_value(package));copy_before=repr(copy)
        original_results=validate_writer_copy(copy,package)
        corrected=validate_source_literal_copy(copy,package,semantic_witness_reader=w.read)
        assert next(row for row in corrected if row.validator=='QUOTE_FIDELITY').result=='PASS'
        assert [row for row in corrected if row.validator!='QUOTE_FIDELITY']==[row for row in original_results if row.validator!='QUOTE_FIDELITY']
        assert canonical_json_bytes(evidence_package_value(package))==before and repr(copy)==copy_before
        assert next(row for row in validate_source_literal_copy(copy,package)if row.validator=='QUOTE_FIDELITY').result=='FAIL'
        for wrong in (replace(copy,body=copy.body.replace('名稱為「','名稱為「未經證實')),replace(copy,body=copy.body+'\n“unattributed quotation”'),
                replace(copy,body=copy.body+'\n“unbalanced'),replace(copy,body=copy.body+'\npolicy’s invention')):
            assert next(row for row in validate_source_literal_copy(wrong,package,semantic_witness_reader=w.read)if row.validator=='QUOTE_FIDELITY').result=='FAIL'
        assert len(qa)==1 and len(jev)==2 and len(render)==1


def test_authenticated_witness_rejects_replacement_context_instead_of_rescuing_stale_source(tmp_path,monkeypatch):
    with _witness_case(tmp_path,monkeypatch)as(w,q,c,p,b,usage,calls,stopped):
        verified=replace(q,semantic_witness_ref=w.evaluate(q,c,p,b))
        assert qualification_relation_is_admitted(verified,c,p,semantic_witness_reader=w.read,source_context=p.passages[0])
        assert not qualification_relation_is_admitted(verified,c,p,semantic_witness_reader=w.read,
            source_context=p.passages[0]+'\nThe original assertion is withdrawn.')
        assert len(calls)==1


# Exact public Source bytes and two original QA selections from the retained
# 821 proposal. These fixtures prove framing only, never criterion YES.
_ACTUAL_821_SOURCE = 'Biggest ever overhaul of SEND teacher training begins\n\nEvery single teacher in England will be empowered to better support children and young people with SEND, as government kickstarts the most comprehensive teacher training offer in a generation. Developed by training experts Ambition Institute, and building on expertise from the sector, the first-of-their-kind free, adaptable and practical training materials for schools and post-16 settings are now available to be used in the way that works best for them, throughout the year or on existing training days. The materials will put inclusion at the heart of every classroom and workshop, giving teachers practical techniques such as breaking instructions into digestible steps, pre-teaching specialist vocabulary, and carving out time for children and young people to practise it. They will go hand in hand with online training for early years staff, developed and delivered by early years charity Dingley’s Promise, supporting staff to use age-appropriate inclusive practices from day one - like using visual timetables to help children with transitions between activities, or adapting environments to reduce unnecessary noise that can be a barrier to communication and emotional regulation. The move is the first step as part of the government’s wider £200 million investment into teacher training to transform support for children and young people with SEND, ending a postcode lottery that has left too many families fighting for support. Minister for School Standards, Georgia Gould, said: Every child and young person deserves the chance to thrive, whatever their needs, and that starts with brilliant teaching. I’ve heard from teachers, parents and young people across the country about the need for more investment in teacher training and today we are responding to those concerns. We are backing staff with the tools to spot and support additional needs from the earliest years, making sure all children and young people feel included. One in three teachers in a recent survey said they want more training on supporting children with SEND. Alongside this, a quarter of surveyed teaching assistants in schools reported that a lack of sufficient training was a barrier to them effectively providing support for these pupils. This is just start of reforms to change that, with further resources due to published in January and April next year. New fully funded training courses will then follow in September next year available to teachers, leaders and support staff across the country. The offer fits into the government’s wider work to ensure there is a pathway for every child, no matter their background or needs. As research finds that young people with SEND are around 80 per cent more likely to be NEET, than average, it is more important than ever to deliver the right support earlier so every child is set up to achieve later in life. Claire Heywood, Vice Principal at Kidderminster College, who provided feedback on the materials, said: The strongest feature of these materials is that they place inclusive practice at the heart of high-quality teaching, recognising that meeting the needs of learners with SEND is an integral part of effective teaching and learning. The attention given to vocational and technical learning, learner independence, progression and preparation for adulthood means the resources feel both evidence-informed and genuinely relevant to the further education sector. Crucially, they provide staff with practical approaches and tools that can be readily applied in the classroom and workshops to support the success of all learners. Hilary Spencer, Chief Executive of Ambition Institute, said: Every child and young person deserves to feel included and able to achieve their potential. We know that educators across the country want to feel more confident and well-equipped to support children and young people with SEND. Our aim throughout has been to create practical, adaptable materials that help teachers and support staff to respond to a wide range of needs and strengthen their inclusive teaching practice. At Ambition Institute, our mission is to help educators serving children from disadvantaged backgrounds to keep getting better. We hope these materials contribute to helping every child and young person thrive, whatever their starting point. Catherine Mole, CEO of Dingley’s Promise, said: We are delighted to have been selected to lead on this vital training programme. This programme offers early years educators the opportunity to upskill and gain more confidence to support children with SEND at no cost to themselves or the setting and shows the increasing value being placed on early years education and its impact. This government investment is a key step towards ensuring that every child has access to the right support at the earliest point possible to give many more children the best start in life. We would encourage all early years settings to access this training and take a whole setting approach to inclusion as we know first-hand how transformational this can be”.'
_ACTUAL_821_SELECTED = [({'admitted_use': 'PUBLICATION_EVIDENCE', 'certainty': 'CONFIRMED', 'claim': 'Every single teacher in England will be empowered to better support children and young people with SEND, as government kickstarts the most comprehensive teacher training offer in a generation. ', 'claim_id': 'sha256:ca7fac8c4b78607d99c82ffee4abd9b8e5abeed5cb63458d81389ad8ac4c2166', 'claim_role': 'HEADLINE', 'localised_factual_expressions': [], 'originality_basis': 'FACTUAL_REWRITE_REQUIRED', 'originality_policy_version': 'newsroom.cont-originality.v3', 'passage_index': 0, 'policy_version': 'newsroom.governed-claim.v7', 'quotations': [], 'rendered_assertion_zh_hant_hk': '隨著政府啟動一代人以來最全面的教師培訓安排，英格蘭每一名教師將更有能力支援有SEND的兒童及青少年。', 'semantic_relation': {'relation': 'SEMANTICALLY_EQUIVALENT', 'rendered_modality': 'ASSERTED', 'rendered_polarity': 'AFFIRMED', 'source_modality': 'ASSERTED', 'source_polarity': 'AFFIRMED'}, 'source_ids': ['UK-05'], 'status': 'CONFIRMED_FACT', 'supporting_excerpt': 'Every single teacher in England will be empowered to better support children and young people with SEND, as government kickstarts the most comprehensive teacher training offer in a generation. '}, {'governed_claim_id': 'sha256:ca7fac8c4b78607d99c82ffee4abd9b8e5abeed5cb63458d81389ad8ac4c2166', 'policy_version': 'newsroom.evid-012.v7', 'test': 'LAW_RIGHT_STATUS_POLICY', 'test_evidence': {'change_kind': 'PUBLIC_POLICY', 'change_relation': 'NEW_OR_CHANGED_STATE', 'event_polarity': 'AFFIRMED', 'material_relation_span': 'government kickstarts the most comprehensive teacher training offer in a generation', 'new_state': 'the most comprehensive teacher training offer in a generation'}}), ({'admitted_use': 'PUBLICATION_EVIDENCE', 'certainty': 'CONFIRMED', 'claim': 'Developed by training experts Ambition Institute, and building on expertise from the sector, the first-of-their-kind free, adaptable and practical training materials for schools and post-16 settings are now available to be used in the way that works best for them, throughout the year or on existing training days. ', 'claim_id': 'sha256:dca69485329626afe4a0753dab82766aedc9496218c6e1216af4feb167b139c1', 'claim_role': 'SUBSTANTIVE', 'localised_factual_expressions': [], 'originality_basis': 'FACTUAL_REWRITE_REQUIRED', 'originality_policy_version': 'newsroom.cont-originality.v3', 'passage_index': 0, 'policy_version': 'newsroom.governed-claim.v7', 'quotations': [], 'rendered_assertion_zh_hant_hk': '這些材料由培訓專家Ambition Institute開發，並建基於界別的專業經驗；供學校及16歲後教育場所使用的首創、免費、可調適及實用培訓材料現已可供使用，讓它們按最合適的方式，於全年或現有培訓日採用。', 'semantic_relation': {'relation': 'SEMANTICALLY_EQUIVALENT', 'rendered_modality': 'ASSERTED', 'rendered_polarity': 'AFFIRMED', 'source_modality': 'ASSERTED', 'source_polarity': 'AFFIRMED'}, 'source_ids': ['UK-05'], 'status': 'CONFIRMED_FACT', 'supporting_excerpt': 'Developed by training experts Ambition Institute, and building on expertise from the sector, the first-of-their-kind free, adaptable and practical training materials for schools and post-16 settings are now available to be used in the way that works best for them, throughout the year or on existing training days. '}, {'governed_claim_id': 'sha256:dca69485329626afe4a0753dab82766aedc9496218c6e1216af4feb167b139c1', 'policy_version': 'newsroom.evid-012.v7', 'test': 'LAW_RIGHT_STATUS_POLICY', 'test_evidence': {'change_kind': 'PUBLIC_POLICY', 'change_relation': 'NEW_OR_CHANGED_STATE', 'event_polarity': 'AFFIRMED', 'material_relation_span': 'the first-of-their-kind free, adaptable and practical training materials for schools and post-16 settings are now available to be used in the way that works best for them, throughout the year or on existing training days.', 'new_state': 'are now available to be used'}})]


@pytest.mark.parametrize('index',[0,1])
def test_retained_821_trailing_space_preserves_exact_witness_frame(index):
    from newsroom.authority.canonical import digest_canonical
    from newsroom.control_plane.evidence import EvidencePackage
    from newsroom.increment10.evidence import _base_package
    row, proposed = _ACTUAL_821_SELECTED[index]
    claim=SimpleNamespace(**{**row,'source_ids':tuple(row['source_ids'])})
    qualification=QualificationEvidence(Evid012QualificationTest(proposed['test']),claim.claim_id,
        'retained-qualification',tuple(proposed['test_evidence'].items()),proposed['policy_version'])
    base=EvidencePackage(candidate_id='candidate-fixture',hypothesis_id='hypothesis-fixture',signal_ids=('signal-fixture',),lead_ids=('lead-fixture',),
        source_ids=('UK-05',),observation_digests=(digest_bytes(_ACTUAL_821_SOURCE.encode()),),passages=(_ACTUAL_821_SOURCE,))
    binding={'candidate_id':base.candidate_id,'candidate_version_id':'retained-version','hypothesis_digest':digest_bytes(b'hypothesis'),
        'content_digest':base.digest,'evidence_package_digest':base.digest,'coverage':'COMPLETE','newness':'KNOWN_CHANGE',
        'current_scope':{'sources':[{'source_id':'UK-05','body':_ACTUAL_821_SOURCE}]},'prior_scope':{'sources':[]}}
    original=claim.claim.encode();assert original[-1:]==b' '
    input_before=canonical_json_bytes({'qualification':proposed,'claim':row,'binding':binding})
    inputs=_semantic_witness_inputs(qualification,claim,base,binding)
    start=_ACTUAL_821_SOURCE.encode().find(original);end=start+len(original)
    assert inputs['state']['claim']['range']=={'start_byte':start,'end_byte':end}
    assert _ACTUAL_821_SOURCE.encode()[start:end]==original
    assert inputs['state']['sources'][0]['text']==_ACTUAL_821_SOURCE
    assert inputs['state']['fields']==proposed['test_evidence']
    assert canonical_json_bytes({'qualification':proposed,'claim':row,'binding':binding})==input_before


@pytest.mark.parametrize('fault',['fragment','spaced-substring','excerpt','digest','condition','negation'])
def test_witness_sentence_boundary_rejects_partial_or_mismatched_parent(fault):
    from newsroom.control_plane.evidence import EvidencePackage
    statement='Schools now receive practical education materials. '
    body=statement+'The programme remains supported.'
    if fault=='fragment':statement='Schools now receive practical education materials ';body=statement+'after approval.'
    if fault=='spaced-substring':body='The notice states: '+body
    if fault=='condition':body='If approved; '+body
    if fault=='negation':body='It is false that '+body
    base=EvidencePackage(candidate_id='candidate',hypothesis_id='hypothesis',signal_ids=('signal',),lead_ids=('lead',),
        source_ids=('source',),observation_digests=(digest_bytes(body.encode()),),passages=(body,))
    claim=SimpleNamespace(claim_id='claim',claim=statement,supporting_excerpt=statement if fault!='excerpt'else statement.rstrip(),
        source_ids=('source',),passage_index=0,claim_role='HEADLINE',status='CONFIRMED_FACT')
    qualification=QualificationEvidence(Evid012QualificationTest.HOUSEHOLD_PRACTICAL_EFFECT,'claim','qualification',
        (('domain','EDUCATION'),('event_polarity','AFFIRMED'),('effect_relation','MATERIAL_PRACTICAL_EFFECT'),
         ('material_relation_span',statement),('practical_effect',statement)))
    binding={'candidate_id':'candidate','candidate_version_id':'version','hypothesis_digest':digest_bytes(b'hypothesis'),
        'content_digest':base.digest if fault!='digest'else digest_bytes(b'wrong'),'evidence_package_digest':base.digest,
        'coverage':'COMPLETE','newness':'KNOWN_CHANGE','current_scope':{'sources':[{'source_id':'source','body':body}]}}
    with pytest.raises(ValueError):_semantic_witness_inputs(qualification,claim,base,binding)
