"""Source-bound closed judgments; localisation/generative exception are separate."""
from __future__ import annotations
from dataclasses import dataclass,field
import re
import json

from newsroom.authority import ObjectAdmissionRequest, ObjectAdmissionId, HydrationRequest

from newsroom.authority.canonical import canonical_json_bytes,digest_canonical
from newsroom.authority.types import UtcTimestamp
from .native_assessor_spans import build_lossless_source_view
from .qualification_rubrics import witness_inventory,question as qualification_question

VERSION='newsroom.native-assessor-judgments.v2'
ROLES=('MATERIAL','SUPPORTING','BACKGROUND','UNCERTAIN')


@dataclass(frozen=True)
class JudgedAssessment:
    execution: object
    decision_record: bytes
    decision_admission_id: ObjectAdmissionId
    judgment_inputs: tuple=()


@dataclass(frozen=True)
class JudgmentFallback:
    reason: str
    references: tuple=()
    details: dict=field(default_factory=dict)


def source_role_questions(candidates):
    """Exact current producer role questions, also used for retained proof reads."""
    return {identity:{'type':'choice','instructions':f'Classify exact span {identity} in full source context; future announced change can be material, but is not already in force. Keep supporting facts and times even if not novel.',
            'criteria':{'MATERIAL':'Affirmed material event/policy/action claim.', 'SUPPORTING':'Affirmed supported detail, timestamp or explanatory fact; preserve exact modality and negation.',
                        'BACKGROUND':'Administrative/irrelevant framing or separator.', 'UNCERTAIN':'Unresolved factual role/support, attributed allegation or provisional rather than confirmed fact.'}}for identity in candidates}


class NativeAssessorJudgments:
    """Two accounted batches over one exact source view, not a backend router."""
    def __init__(self,*,judgments,scope_for,proof,require_current=lambda:None,
                 localise=None,read_localisation=None):
        self.judgments,self.scope_for,self.proof=judgments,scope_for,proof
        self.require_current=require_current
        self.localise,self.read_localisation=localise,read_localisation

    @staticmethod
    def _binding(candidate,base,scope,view):
        from .native_assessor import _reference_binding
        binding={'content_digest':base.digest,'source_reference_binding':None if view is None else _reference_binding(view),
                 'candidate_version_id':candidate.version_id,'candidate_id':candidate.candidate_id,
                 'hypothesis_digest':candidate.governing_manifest.canonical_digest,'evidence_package_digest':base.digest,'current_scope':scope.get('current_scope'),
                 'prior_scope':scope.get('prior_scope'),'newness':scope.get('newness'),'coverage':'COMPLETE'}
        if 'source_currentness' in scope:
            binding['source_currentness']=scope['source_currentness']
        if scope.get('newness')=='SOURCE_DECLARED_FIRST_PUBLICATION':
            binding['first_publication']=scope.get('first_publication')
        return binding

    @staticmethod
    def _decision_key(binding):
        return 'judgment-decision:'+digest_canonical([VERSION,binding])

    @staticmethod
    def _first_publication_proven(scope,sources,acquired):
        if scope.get('prior_scope') is not None:
            return False
        try:
            expected=[]
            for source,item in zip(sources,acquired,strict=True):
                if (source.unit.source_id not in {'UK-01','UK-02','UK-03','UK-05'}
                        or not item.canonical_url.startswith('https://www.gov.uk/')
                        or item.source_type!='PRIMARY_OFFICIAL'
                        or item.currentness_basis!='AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT'
                        or not any(role.role.value=='ORIGINATING_AUTHORITY' for role in source.source_version.request.roles)
                        or UtcTimestamp.parse(item.publication_time).value>UtcTimestamp.parse(item.retrieval_time).value):
                    return False
                expected.append({'source_id':source.unit.source_id,'definition_id':str(source.unit.authority.definition_id),
                    'definition_version_id':str(source.unit.authority.definition_version_id),'source_revision_digest':source.unit.revision_digest,
                    'acquisition_receipt_digest':item.receipt_digest,'first_published_at':item.publication_time})
            return bool(expected) and scope.get('first_publication')==expected
        except (AttributeError,TypeError,ValueError):
            return False

    def get_decision_ref(self,candidate,base,sources,acquired):
        scope=self.scope_for(candidate,base,sources,acquired)
        if type(scope)is not dict or scope.get('coverage')!='COMPLETE' or scope.get('newness')not in ('KNOWN_CHANGE','SOURCE_DECLARED_FIRST_PUBLICATION'):
            return None
        if scope['newness']=='SOURCE_DECLARED_FIRST_PUBLICATION' and not self._first_publication_proven(scope,sources,acquired):
            return None
        if base.source_ids!=tuple(source.unit.source_id for source in sources) or base.passages!=tuple(item.body.decode('utf-8') for item in acquired):
            raise ValueError('judgment acquired source bytes differ')
        view=build_lossless_source_view(base.passages,base.source_ids)
        binding=self._binding(candidate,base,scope,view)
        self.require_current()
        existing=self.judgments.objects.committed_admission(ObjectAdmissionRequest('evidence.record',self._decision_key(binding)),proof=self.proof)
        return None if existing is None else existing.admission.admission_id

    def assess(self,candidate,base,sources,acquired):
        return self._assess(candidate,base,sources,acquired)

    def read(self,admission_id,candidate,base,sources,acquired):
        self.require_current()
        raw=self.judgments.objects.rehydrate(HydrationRequest(admission_id,'evidence.record'),proof=self.proof).data
        record=json.loads(raw)
        if canonical_json_bytes(record)!=raw or record.get('schema')!=VERSION:
            raise ValueError('judgment decision canonical record differs')
        result=self._assess(candidate,base,sources,acquired,retained=record,admission_id=admission_id)
        if type(result)is not JudgedAssessment or result.decision_record!=raw:
            raise ValueError('judgment decision current binding differs')
        return result

    def validation_failure(self,result,candidate,base,sources,acquired,reason):
        """Only a caller-proved static materialisation failure enters this seam."""
        checked=self.read(result.decision_admission_id,candidate,base,sources,acquired)
        decision=json.loads(checked.decision_record)
        inputs=list(checked.judgment_inputs)
        from .typesafe_judgment import JudgmentReference
        refs=tuple(JudgmentReference(row['invocation_id'],ObjectAdmissionId.parse(row['raw_admission_id']),
            ObjectAdmissionId.parse(row['receipt_admission_id']))for row in decision['judgments'])
        return JudgmentFallback('MATERIALISATION_VALIDATION_FAILED',refs,{'source_binding':decision['source_binding'],
            'state':inputs[-1]['state'],'judgment_inputs':inputs,'failed_questions':[{'reason':reason}],
            'prior_decision_admission_id':str(result.decision_admission_id),
            'render_provenance':decision['render_provenance']})

    def _finish(self,wire,view,binding,references,judgment_inputs,render_proof,admission_id):
        from .native_assessor import NativeAssessmentExecution,_materialise_reference_result,VERSION as CODEC
        _package,receipt=_materialise_reference_result(canonical_json_bytes(wire),view,digest_canonical(binding),CODEC)
        self.require_current()
        decision={'schema':VERSION,'source_binding':binding,'materialisation_receipt':receipt,
            'judgments':[{'invocation_id':r.invocation_id,'raw_admission_id':str(r.raw_admission_id),
                'receipt_admission_id':str(r.receipt_admission_id)}for r in references],
            'render_provenance':render_proof}
        raw=canonical_json_bytes(decision)
        if admission_id is None:
            admission_id=self.judgments.objects.admit(ObjectAdmissionRequest('evidence.record',self._decision_key(binding)),raw,proof=self.proof).admission.admission_id
        return JudgedAssessment(NativeAssessmentExecution(receipt['materialised_text'],{}),raw,admission_id,tuple(judgment_inputs))

    def _assess(self,candidate,base,sources,acquired,*,retained=None,admission_id=None):
        from .native_assessor import NativeAssessmentExecution,_materialise_reference_result,VERSION as CODEC,PROVIDER_SCHEMA
        scope=self.scope_for(candidate,base,sources,acquired)
        def early(reason):
            view=(build_lossless_source_view(base.passages,base.source_ids)
                if type(scope)is dict and scope.get('coverage')=='COMPLETE' else None)
            binding=self._binding(candidate,base,scope if type(scope)is dict else {},view)
            return JudgmentFallback(reason,(),{'source_binding':binding,
                'state':{'sources':[{'source_id':identity,'text':body}for identity,body in zip(base.source_ids,base.passages,strict=True)]},
                'judgment_inputs':[],'failed_questions':[{'reason':reason}]})
        if type(scope)is not dict or scope.get('coverage')!='COMPLETE':return early('SOURCE_COVERAGE_UNPROVEN')
        if scope.get('newness')not in ('KNOWN_CHANGE','KNOWN_UNCHANGED','SOURCE_DECLARED_FIRST_PUBLICATION'):return early('NEWNESS_BASELINE_UNKNOWN')
        if scope['newness']=='KNOWN_UNCHANGED':return early('EXISTING_REASONING_NEWNESS_REQUIRED')
        if scope['newness']=='SOURCE_DECLARED_FIRST_PUBLICATION' and not self._first_publication_proven(scope,sources,acquired):
            return early('FIRST_PUBLICATION_PROVENANCE_UNPROVEN')
        if base.source_ids!=tuple(source.unit.source_id for source in sources) or base.passages!=tuple(item.body.decode('utf-8') for item in acquired):
            raise ValueError('judgment acquired source bytes differ')
        view=build_lossless_source_view(base.passages,base.source_ids)
        if not view.segments:return early('SOURCE_HAS_NO_CLAUSES')
        if len(view.segments)>253:return early('CANDIDATE_COVERAGE_LIMIT')
        candidates={s.span_id:{'source_id':s.source_id,'text':s.text.encode('utf-8')[:s.content_end_byte-s.start_byte].decode('utf-8'),'entities':[list(e) for e in s.entities],
            'rendering_fragment_count':len(s.entities)+1,'source_range':{'first_span_id':s.span_id,'last_span_id':s.span_id}}for s in view.segments}
        binding=self._binding(candidate,base,scope,view)
        stages=[]
        judgment_inputs=[]
        def fallback(reason,*references,failed=()):
            return JudgmentFallback(reason,tuple(references),{'source_binding':binding,'state':state,
                'judgment_inputs':judgment_inputs,'failed_questions':list(failed)or[{'reason':reason}],
                'witness_inventory':state.get('witness_inventory',{})})
        def batch(phase,state,questions):
            self.require_current()
            inputs={'state':state,'questions':questions,'source_binding':binding,
                'cycle_id':digest_canonical([VERSION,phase,binding,state,questions]),'caller_identity':'NATIVE_ASSESSOR',
                'candidate_id':candidate.candidate_id,'hypothesis_digest':candidate.governing_manifest.canonical_digest,
                'proof':self.proof}
            if retained is None:
                from .typesafe_judgment import MODEL,_json
                if len(_json({'model':MODEL,'state':state,'questions':questions}))>self.judgments.policy.max_prompt_bytes:
                    return fallback('JUDGMENT_INPUT_BOUND',*stages,failed=({'stage':phase,'reason':'INPUT_BOUND'},))
                reference=self.judgments.evaluate(**inputs)
            else:
                from .typesafe_judgment import JudgmentReference
                value=retained['judgments'][len(stages)]
                reference=JudgmentReference(value['invocation_id'],ObjectAdmissionId.parse(value['raw_admission_id']),ObjectAdmissionId.parse(value['receipt_admission_id']))
            record=self.judgments.read(reference,**inputs)
            judgment_inputs.append({key:value for key,value in inputs.items()if key!='proof'})
            stages.append(reference)
            if type(record)is not dict or set(record.get('answers',{}))!=set(questions):raise ValueError('judgment answer inventory differs')
            return reference,record['answers']
        state={'candidates':candidates,'current_scope':binding['current_scope'],'prior_scope':binding['prior_scope']}
        if scope['newness']=='SOURCE_DECLARED_FIRST_PUBLICATION':
            state['publication_basis']={'mode':scope['newness'],'published_at':{row['source_id']:row['first_published_at']for row in scope['first_publication']}}
        questions=source_role_questions(candidates)
        result=batch('SOURCE_ROLES',state,questions)
        if type(result)is JudgmentFallback:return result
        first,roles=result
        if any(a.get('choice')not in ROLES for a in roles.values()):raise ValueError('judgment role differs')
        if any(a['choice']=='UNCERTAIN'for a in roles.values()):return fallback('UNCERTAIN_SOURCE_ROLE',first)
        selected=[identity for identity,a in roles.items()if a['choice']in ('MATERIAL','SUPPORTING')]
        material=[identity for identity in selected if roles[identity]['choice']=='MATERIAL']
        if not material:
            wire={'package':{'select_new_information':False,'governed_claims':[],'qualification_evidence':[],
                'selection_rationale':'Complete source-role judgment inventory identifies no material new assertion.',
                'geography':[],'categories':[],'explicit_exclusions':[]}}
            return self._finish(wire,view,binding,(first,),judgment_inputs,{'mode':'NO_MATERIAL_CLAIMS'},admission_id)
        if len(selected)>32:return fallback('MISSING_OR_UNBOUNDED_MATERIAL_CLAIMS',first)
        inventory=witness_inventory(view)
        state={**state,'witness_inventory':inventory}
        if any(len(inventory[identity]['candidates'])>254 or not inventory[identity]['candidates'] or inventory[identity]['uncovered_clause_ids'] for identity in material):
            return fallback('QUALIFICATION_WITNESS_COVERAGE_UNPROVEN',first)

        variants=PROVIDER_SCHEMA['properties']['package']['properties']['qualification_evidence']['items']['oneOf']
        rules={v['properties']['test']['const']:v['properties']['test_evidence']['properties']for v in variants}
        questions={'headline':{'type':'choice','instructions':'Choose the strongest material headline; use UNCERTAIN for unresolved support.',
            'criteria':{**{identity:candidates[identity]['text']for identity in material},'UNCERTAIN':'Unresolved headline.'}}}
        for identity in material:
            if scope['newness']=='SOURCE_DECLARED_FIRST_PUBLICATION':
                questions[identity+':announced_event']={'type':'choice','instructions':
                    f'Does exact span {identity} affirm a newly announced event or official action at the source-declared first publication? '
                    'No prior baseline exists: do not infer comparative change, novelty from fetched/updated time, or in-force status from a future announcement.',
                    'criteria':{'YES':'Source affirms a newly announced material event/action, with exact modality and timing.',
                        'NO':'Only administrative metadata, older/background facts or an amendment needing an unavailable comparison.',
                        'UNCERTAIN':'Newly announced status is not established.'}}
            for test,fields in rules.items():
                prefix=identity+':'+test
                questions[prefix]=qualification_question(identity,test,fields)
                for field,schema in fields.items():
                    if 'enum'in schema:choices={value:value for value in schema['enum']}
                    elif field.endswith('_source_lookup_key'):
                        choices={key:value['text']for key,value in inventory[identity]['candidates'].items()}
                    elif field=='duration_minutes':
                        # Exact literal minute inventory only; existing local validator owns arithmetic.
                        choices={m.group(1):m.group(1) for m in re.finditer(r'(?<![\d.,])([0-9]+)\s*(?:minutes?|mins?|分鐘)(?![A-Za-z])',candidates[identity]['text'],re.I)}
                    else:continue
                    questions[prefix+':'+field]={'type':'choice','instructions':f'For {identity}/{test}, select source-supported {field}; NONE if not established.',
                        'criteria':{**choices,'NONE':'No established value.'}}
        result=batch('SELECTED_QUALIFICATION',state,questions)
        if type(result)is JudgmentFallback:return result
        second,answers=result
        headline=answers['headline'].get('choice')
        if headline not in material:return fallback('UNCERTAIN_HEADLINE',first,second)
        if scope['newness']=='SOURCE_DECLARED_FIRST_PUBLICATION' and any(answers[identity+':announced_event'].get('choice')!='YES' for identity in material):
            return fallback('FIRST_PUBLICATION_ANNOUNCEMENT_UNPROVEN',first,second)
        qualification=[]
        ordered=[headline,*[identity for identity in selected if identity!=headline]]
        for index,identity in enumerate(ordered):
            if identity not in material:
                continue
            for test,fields in rules.items():
                prefix=identity+':'+test;choice=answers[prefix].get('choice')
                if choice=='UNCERTAIN':
                    if index==0:return fallback('UNCERTAIN_QUALIFICATION',first,second,failed=({'question_id':prefix,'reason':'UNCERTAIN'},))
                    continue
                if choice!='YES':continue
                witnesses={}
                for field,schema in fields.items():
                    if 'const'in schema:witnesses[field]=schema['const'];continue
                    value=answers[prefix+':'+field].get('choice')
                    if value=='NONE':
                        if index==0:return fallback('QUALIFICATION_WITNESS_MISSING',first,second,failed=({'question_id':prefix+':'+field,'reason':'NONE'},))
                        witnesses=None
                        break
                    if field.endswith('_source_lookup_key'):
                        from .admission import _qualification_text_is_negative
                        parent=inventory[identity]['parent_text'].strip()
                        chosen=inventory[identity]['candidates'][value]['text']
                        if chosen!=parent and _qualification_text_is_negative(parent):
                            if index==0:return fallback('PARENT_MODALITY_REQUIRES_REASONING',first,second,failed=({'question_id':prefix+':'+field,'reason':'PARENT_NEGATION_OR_MODALITY'},))
                            witnesses=None
                            break
                    witnesses[field]=inventory[identity]['candidates'][value]['text']if field.endswith('_source_lookup_key')else value
                if witnesses is not None:
                    qualification.append({'claim_index':index,'test':test,'test_evidence':witnesses})
        if not any(q['claim_index']==0 for q in qualification):return fallback('HEADLINE_QUALIFICATION_UNPROVEN',first,second)
        if self.localise and self.read_localisation:
            request={'source_binding':binding,'claims':{identity:candidates[identity]for identity in ordered}}
            if retained is None:
                reference=self.localise(request)
            else:
                from .native_claim_localisation import LocalisationReference
                value=retained['render_provenance']
                reference=LocalisationReference(value['invocation_id'],ObjectAdmissionId.parse(value['raw_admission_id']),ObjectAdmissionId.parse(value['receipt_admission_id']))
            record=self.read_localisation(reference,request)
            if record.get('source_binding')!=binding or not record.get('invocation_id')or not record.get('terminal_digest'):raise ValueError('localisation provenance differs')
            if set(record.get('renderings',{}))!=set(ordered):raise ValueError('localisation claim inventory differs')
            renderings=record['renderings'];render_proof={'mode':'QUALIFIED_LOCALISATION','invocation_id':record['invocation_id'],
                'terminal_digest':record['terminal_digest'],'raw_admission_id':str(reference.raw_admission_id),
                'receipt_admission_id':str(reference.receipt_admission_id)}
        else:return fallback('QUALIFIED_LOCALISATION_REQUIRED',first,second)
        claims=[{'claim_role':'HEADLINE'if identity==headline else 'SUBSTANTIVE'if roles[identity]['choice']=='MATERIAL'else 'CONTEXT',
            'status':'CONFIRMED_FACT','source_range':candidates[identity]['source_range'],
            'rendered_assertion_zh_hant_hk_fragments':renderings[identity]['rendered_assertion_zh_hant_hk_fragments'],
            'factual_localisations':renderings[identity].get('factual_localisations',[]),
            'quotation_source_keys':renderings[identity].get('quotation_source_keys',[])}for identity in ordered]
        wire={'package':{'select_new_information':True,'governed_claims':claims,'qualification_evidence':qualification,
            'selection_rationale':'Source-bound staged judgment selection.','geography':[],'categories':[],'explicit_exclusions':[]}}
        return self._finish(wire,view,binding,(first,second),judgment_inputs,render_proof,admission_id)
