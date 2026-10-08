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
    semantic_witnesses: object = None
    source_renderings: object = None


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


def _packed_support_candidates(view, candidates, roles):
    """Lossless contiguous context, never fewer material facts or Source bytes."""
    from .admission import _qualification_text_is_negative
    from .native_assessor_references import _claim_entities, MAX_FRAGMENT_LENGTH
    from .native_source_context_ranges import _FIRST_PERSON, _SPEAKER
    from .writer import _has_unicode_quote_delimiter

    def quoted(text):
        if text.count("'") >= 2 or re.search(r"\w+’\w+’\w+",text):return True  # Ambiguous paired quotes.
        for i,char in enumerate(text):
            if char in {"'",'’'}and 0<i<len(text)-1 and text[i-1].isalnum()and text[i+1].isalnum():
                continue  # Apostrophe inside a Source word, not a quotation boundary.
            if char in {'"',"'"}or _has_unicode_quote_delimiter(char):return True
        return False

    def barrier(text):
        return (_qualification_text_is_negative(text) or _FIRST_PERSON.search(text)
            or _SPEAKER.match(text) or quoted(text) or re.search(r"\b[A-Za-z]+n['’]t\b",text,re.I)
            or re.search(r'\b(?:if|unless|subject to|conditional(?:ly)?|pending|said|says|stated|announced)\b',text,re.I))

    def candidate(first,last):
        reference={'first_span_id':first.span_id,'last_span_id':last.span_id}
        text,passage,source_id=view.resolve_range(reference)
        entities=_claim_entities(text,view.passages[passage],policy_version=view.entity_policy_version)
        if len(entities)>64 or len(text)>MAX_FRAGMENT_LENGTH:return None
        return {'source_id':source_id,'text':text,'entities':[list(item)for item in entities],
            'rendering_fragment_count':len(entities)+1,'source_range':reference}

    packed={};previous=None;start=None;count=0
    for segment in view.segments:
        role=roles[segment.span_id]['choice']
        if role not in {'MATERIAL','SUPPORTING'}:
            previous=None;start=None;count=0
            continue
        single=candidates[segment.span_id]
        can_join=(role=='SUPPORTING'and previous is not None and start is not None and count<32
            and previous.passage_index==segment.passage_index and previous.ordinal+1==segment.ordinal
            and not barrier(single['text']))
        merged=candidate(start,segment)if can_join else None
        if merged is not None:
            packed[start.span_id]=merged;previous=segment;count+=1
            continue
        if role=='MATERIAL':packed[segment.span_id]=single
        else:
            single=candidate(segment,segment)
            if single is None:return None
            packed[segment.span_id]=single
        if role=='SUPPORTING'and not barrier(single['text']):
            start=previous=segment;count=1
        else:previous=None;start=None;count=0
    return packed


class NativeAssessorJudgments:
    """Two accounted batches over one exact source view, not a backend router."""
    def __init__(self,*,judgments,scope_for,proof,require_current=lambda:None,
                 localise=None,read_localisation=None):
        self.judgments,self.scope_for,self.proof=judgments,scope_for,proof
        self.require_current=require_current
        self.localise,self.read_localisation=localise,read_localisation
        self.semantic_witness_reader = None

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

    def _finish(self,wire,view,binding,references,judgment_inputs,render_proof,admission_id,rendering_reference=None):
        from .native_assessor import NativeAssessmentExecution,_materialise_reference_result,VERSION as CODEC
        _package,receipt=_materialise_reference_result(canonical_json_bytes(wire),view,digest_canonical(binding),CODEC)
        self.require_current()
        decision={'schema':VERSION,'source_binding':binding,'materialisation_receipt':receipt,
            'judgments':[{'invocation_id':r.invocation_id,'raw_admission_id':str(r.raw_admission_id),
                'receipt_admission_id':str(r.receipt_admission_id)}for r in references],
            'render_provenance':render_proof}
        source_renderings = None
        if rendering_reference is not None:
            from .evidence import SOURCE_RENDERING_CONTRACT_V2
            ref = tuple(sorted({'contract': SOURCE_RENDERING_CONTRACT_V2, 'operation': 'SOURCE_RENDERING',
                                **rendering_reference}.items()))
            rows = json.loads(receipt['materialised_text'])['package']['governed_claims']
            source_renderings = SourceRenderingMetadata(tuple((row['claim_id'], ref) for row in rows))
            decision['source_renderings'] = [[key, dict(value)] for key, value in source_renderings.references]
        raw=canonical_json_bytes(decision)
        if admission_id is None:
            admission_id=self.judgments.objects.admit(ObjectAdmissionRequest('evidence.record',self._decision_key(binding)),raw,proof=self.proof).admission.admission_id
        return JudgedAssessment(NativeAssessmentExecution(receipt['materialised_text'],{}),raw,admission_id,tuple(judgment_inputs),
                                source_renderings=source_renderings)

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
        render_candidates=candidates
        if len(selected)>32:
            render_candidates=_packed_support_candidates(view,candidates,roles)
            if render_candidates is None or len(render_candidates)>32:
                return fallback('MISSING_OR_UNBOUNDED_MATERIAL_CLAIMS',first)
            selected=list(render_candidates)
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
        rendering_reference = None
        if self.localise and self.read_localisation:
            request={'source_binding':binding,'claims':{identity:render_candidates[identity]for identity in ordered}}
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
            from .native_claim_localisation import TYPED_VERSION
            if record.get('version') == TYPED_VERSION:
                if record.get('original_state') != request or record.get('projected_state') != source_rendering_projection(request):
                    raise ValueError('typed localisation projection differs')
                renderings = {identity: original_rendering_slots(request['claims'][identity],
                    record['projected_state']['claims'][identity], renderings[identity]) for identity in ordered}
                rendering_reference = {key: render_proof[key] for key in ('invocation_id', 'raw_admission_id', 'receipt_admission_id')}
        else:return fallback('QUALIFIED_LOCALISATION_REQUIRED',first,second)
        claims=[{'claim_role':'HEADLINE'if identity==headline else 'SUBSTANTIVE'if roles[identity]['choice']=='MATERIAL'else 'CONTEXT',
            'status':'CONFIRMED_FACT','source_range':render_candidates[identity]['source_range'],
            'rendered_assertion_zh_hant_hk_fragments':renderings[identity]['rendered_assertion_zh_hant_hk_fragments'],
            'factual_localisations':renderings[identity].get('factual_localisations',[]),
            'quotation_source_keys':renderings[identity].get('quotation_source_keys',[])}for identity in ordered]
        wire={'package':{'select_new_information':True,'governed_claims':claims,'qualification_evidence':qualification,
            'selection_rationale':'Source-bound staged judgment selection.','geography':[],'categories':[],'explicit_exclusions':[]}}
        return self._finish(wire,view,binding,(first,second),judgment_inputs,render_proof,admission_id,
                            rendering_reference=rendering_reference)


SEMANTIC_WITNESS_CONSUMER_VERSION = 'newsroom.semantic-witness-consumer.v1'


@dataclass(frozen=True, slots=True)
class SemanticWitnessMetadata:
    """Application side-channel, never part of SourceQA producer JSON."""
    references: tuple

    def __post_init__(self):
        from .evidence import semantic_witness_reference
        if type(self.references) is not tuple or len(dict(self.references)) != len(self.references):
            raise ValueError('semantic witness metadata differs')
        for key, ref in self.references:
            if type(key) is not tuple or len(key) != 2 or any(type(p) is not str or not p for p in key):
                raise ValueError('semantic witness key differs')
            semantic_witness_reference(ref)


@dataclass(frozen=True, slots=True)
class SourceRenderingMetadata:
    references: tuple

    def __post_init__(self):
        from .evidence import source_rendering_reference
        if type(self.references) is not tuple or len(dict(self.references)) != len(self.references):
            raise ValueError('Source rendering metadata differs')
        for claim_id, ref in self.references:
            if type(claim_id) is not str or not claim_id:
                raise ValueError('Source rendering claim differs')
            source_rendering_reference(ref)


def source_rendering_details(claim, body, chronology, *, contract=None):
    from .native_source_term_bindings import source_term_bindings, derive_relative_year, VERSION as TERM_VERSION
    from .evidence import SOURCE_RENDERING_CONTRACT, SOURCE_RENDERING_CONTRACT_V2
    from .native_source_term_bindings import VERSION_V2
    contract = SOURCE_RENDERING_CONTRACT if contract is None else contract
    term_version = VERSION_V2 if contract == SOURCE_RENDERING_CONTRACT_V2 else TERM_VERSION
    if contract not in {SOURCE_RENDERING_CONTRACT, SOURCE_RENDERING_CONTRACT_V2} or not contract.endswith(term_version):
        raise ValueError('Source term consumer identity differs')
    from newsroom.authority.canonical import digest_bytes
    raw, selected = body.encode(), claim.claim.encode()
    start = raw.find(selected)
    if start < 0 or raw.find(selected, start + 1) >= 0:
        raise ValueError('Source rendering selected range is ambiguous')
    args = dict(body_digest=digest_bytes(raw),start_byte=start,end_byte=start+len(selected))
    terms = source_term_bindings(body,claim.claim,**args, version=term_version)
    year = (derive_relative_year(body,claim.claim,**args,
        publication_time=chronology['published_at'],source_updated_time=chronology['updated_at'])
        if re.search(r'\bnext year\b',claim.claim,re.I) else None)
    return terms, year


def source_rendering_names(claim, body, *, contract=None):
    from .native_source_term_bindings import source_term_bindings
    from .evidence import bounded_named_entities, SOURCE_RENDERING_CONTRACT, SOURCE_RENDERING_CONTRACT_V2
    from .native_source_term_bindings import VERSION, VERSION_V2
    contract = SOURCE_RENDERING_CONTRACT if contract is None else contract
    if contract not in {SOURCE_RENDERING_CONTRACT, SOURCE_RENDERING_CONTRACT_V2}:
        raise ValueError('Source rendering contract differs')
    from newsroom.authority.canonical import digest_bytes
    raw, selected = body.encode(), claim.claim.encode()
    start = raw.find(selected)
    if start < 0 or raw.find(selected,start+1) >= 0:
        raise ValueError('Source rendering selected range is ambiguous')
    known = bounded_named_entities(claim.claim,source_context=body)
    terms = source_term_bindings(body,claim.claim,body_digest=digest_bytes(raw),start_byte=start,end_byte=start+len(selected),
        version=VERSION_V2 if contract == SOURCE_RENDERING_CONTRACT_V2 else VERSION)
    names = {name for name,_kind in known}
    return known | frozenset((name,'SOURCE_LITERAL')for name,_kind,_start,_end in terms if name not in names)


def source_rendering_projection(original_state):
    """New literal slots over the same selected bytes, never a paid view rewrite."""
    from copy import deepcopy
    from .native_source_term_bindings import source_term_bindings, VERSION_V2
    from .evidence import bounded_named_entities
    from newsroom.authority.canonical import digest_bytes
    projected = deepcopy(original_state)
    sources = original_state['source_binding'].get('current_scope', {}).get('sources', ())
    view = None
    if sources:
        view = build_lossless_source_view(tuple(row['body'] for row in sources), tuple(row['source_id'] for row in sources),
            version=original_state['source_binding'].get('source_reference_binding', {}).get('partition_version',
                'newsroom.native-assessor-spans.v2'))
        if original_state['source_binding'].get('source_reference_binding'):
            from .native_assessor import _reference_binding
            if _reference_binding(view) != original_state['source_binding']['source_reference_binding']:
                raise ValueError('Source rendering projection binding differs')
    for identity, original in original_state['claims'].items():
        text = original['text']
        if view is None:
            terms = source_term_bindings(text, text, body_digest=digest_bytes(text.encode()),
                start_byte=0, end_byte=len(text.encode()), version=VERSION_V2)
            if (any(kind == 'SOURCE_LITERAL' for _name, kind in original.get('entities', ()))
                    or any(name not in {name for name, _kind in original.get('entities', ())} for name, *_ in terms)):
                raise ValueError('Source rendering projection requires current Source bytes')
            continue
        if original.get('source_range'):
            selected, passage, source_id = view.resolve_range(original['source_range'])
            if selected != text or source_id != original['source_id']:
                raise ValueError('Source rendering projection range differs')
            first = next(segment for segment in view.segments if segment.span_id == original['source_range']['first_span_id'])
            start = first.start_byte
        else:
            matching = [index for index, row in enumerate(sources) if row['source_id'] == original['source_id']]
            if len(matching) != 1:
                raise ValueError('Source rendering projection source differs')
            passage = matching[0]
            raw = sources[passage]['body'].encode(); start = raw.find(text.encode())
            if start < 0 or raw.find(text.encode(), start + 1) >= 0:
                raise ValueError('Source rendering projection selected range is ambiguous')
        body = sources[passage]['body']
        from .native_source_term_bindings import VERSION
        legacy_terms = source_term_bindings(body, text, body_digest=digest_bytes(body.encode()), start_byte=start,
            end_byte=start+len(text.encode()), version=VERSION)
        recognised = bounded_named_entities(text, source_context=body)
        old_literals = {name for name, _kind, _start, _end in legacy_terms}
        if any((name, kind) not in recognised and not (kind == 'SOURCE_LITERAL' and name in old_literals)
               for name, kind in original.get('entities', ())):
            raise ValueError('Source rendering original literal/entity differs')
        terms = source_term_bindings(body, text, body_digest=digest_bytes(body.encode()), start_byte=start,
            end_byte=start+len(text.encode()), version=VERSION_V2)
        kinds = dict(bounded_named_entities(text, source_context=body))
        entities = [[name, kinds.get(name, 'SOURCE_LITERAL')] for name, _kind, _start, _end in terms]
        projected['claims'][identity] = {**original, 'entities': entities, 'rendering_fragment_count': len(entities)+1}
    return projected


def original_rendering_slots(original, projected, rendering):
    """Re-express new slots in the immutable Source codec's original slots."""
    result = dict(rendering)
    fragments = rendering['rendered_assertion_zh_hant_hk_fragments']
    text = fragments[0] + ''.join(name+fragment for (name, _kind), fragment in
        zip(projected['entities'], fragments[1:], strict=True))
    restored = []
    for name, _kind in original['entities']:
        before, found, text = text.partition(name)
        if not found:
            raise ValueError('Source rendering original entity omitted')
        restored.append(before)
    result['rendered_assertion_zh_hant_hk_fragments'] = [*restored, text]
    return result


def _semantic_witness_inputs(qualification, claim, package, binding):
    """Full Source plus exact ranges; never shortened lexical witness keys."""
    from .evidence import SEMANTIC_WITNESS_CONTRACT
    from newsroom.increment10.evidence import _base_package
    base = _base_package(package)
    if (binding.get('candidate_id') != base.candidate_id
            or binding.get('content_digest') != base.digest
            or binding.get('evidence_package_digest') != base.digest
            or binding.get('coverage') != 'COMPLETE'
            or binding.get('newness') not in {'KNOWN_CHANGE', 'SOURCE_DECLARED_FIRST_PUBLICATION'}
            or not binding.get('candidate_version_id') or not binding.get('hypothesis_digest')
            or not 0 <= claim.passage_index < len(base.passages)
            or claim.source_ids != (base.source_ids[claim.passage_index],)
            or qualification.governed_claim_id != claim.claim_id):
        raise ValueError('semantic witness Source binding differs')
    current = binding.get('current_scope')
    if (type(current) is not dict or tuple(row.get('source_id') for row in current.get('sources', ())) != base.source_ids
            or tuple(row.get('body') for row in current['sources']) != base.passages):
        raise ValueError('semantic witness complete current Source differs')
    source = base.passages[claim.passage_index]
    raw, selected = source.encode(), claim.claim.encode()
    first = raw.find(selected)
    preceding = raw[:first].rstrip(b' \t') if first >= 0 else b''
    if (first < 0 or raw.find(selected, first + 1) >= 0 or claim.supporting_excerpt != claim.claim
            or preceding and preceding[-1:] not in {b'\n', b'.', b'!', b'?'}
            or first + len(selected) < len(raw) and raw[first + len(selected):].lstrip(b' \t')[:1] != b'\n'
                and selected.rstrip(b' \t\r\n\v\f')[-1:] not in {b'.', b'!', b'?'}):
        raise ValueError('semantic witness exact range differs')
    fields = dict(qualification.test_evidence)
    from .admission import _QUALIFICATION_CLASSIFIER_FIELDS
    if any(value not in claim.claim for key, value in fields.items() if key not in _QUALIFICATION_CLASSIFIER_FIELDS):
        raise ValueError('semantic witness field is outside selected Source')
    # Rendering is deliberately absent: localisation never changes qualification.
    state = {'sources': [{'source_id': identity, 'text': text}
                         for identity, text in zip(base.source_ids, base.passages, strict=True)],
        'claim': {'claim_id': claim.claim_id, 'source_id': claim.source_ids[0],
            'range': {'start_byte': first, 'end_byte': first + len(selected)},
            'parent_range': {'start_byte': 0, 'end_byte': len(raw)},
            'claim_role': claim.claim_role, 'status': str(claim.status)},
        'test': qualification.test.value, 'fields': fields,
        'newness': binding['newness'], 'current': {'sources': [{k:v for k,v in row.items() if k != 'body'}for row in current['sources']]},
        'prior': binding.get('prior_scope'), 'first_publication': binding.get('first_publication')}
    question_id = 'criterion'
    from .qualification_rubrics import RUBRICS, TEMPORAL_RULES
    rubric = RUBRICS[qualification.test.value]
    questions = {question_id: {'type': 'choice', 'instructions': {
        'task': 'Verify the proposed qualification criterion and every supplied field against the exact selected range in complete Source/parent context. '
                'No fixed English subject/verb vocabulary is required. Do not remove negation, conditions or future modality. '
                'A confirmed announcement is not an already-effective rule; another category requires a separate answer.',
        **rubric, 'temporal_and_source_rules': TEMPORAL_RULES,
        'required_witness_fields': list(fields)},
        'criteria': {'YES': 'Every required field and the exact selected assertion establish this criterion with newly confirmed material information.',
            'NO': 'Wrong criterion, missing field, unsupported relation, hypothetical intention or contradicted/incomplete parent.',
            'UNCERTAIN': 'Semantic support, newness, parent condition or required field remains unresolved.'}}}
    inputs = {'state': state, 'questions': questions, 'source_binding': binding,
        'cycle_id': digest_canonical([SEMANTIC_WITNESS_CONTRACT, binding, state, questions]),
        'caller_identity': 'NATIVE_ASSESSOR', 'candidate_id': base.candidate_id,
        'hypothesis_digest': binding['hypothesis_digest']}
    return inputs


class NativeSemanticWitnesses:
    """One existing TypeSafe phase; authenticated reads are required at every use."""
    def __init__(self, *, judgments, candidate_for, proof, require_current, parent_reader=None):
        from .typesafe_judgment import TypesafeJudgment
        if type(judgments) is not TypesafeJudgment or not callable(candidate_for) or not callable(require_current):
            raise TypeError('concrete semantic witness authority required')
        self.judgments, self.candidate_for, self.proof, self.require_current = judgments, candidate_for, proof, require_current
        self.parent_reader = parent_reader
        self.rendering_reader = None

    def evaluate(self, qualification, claim, package, binding):
        from .evidence import SEMANTIC_WITNESS_CONTRACT
        self.require_current()
        inputs = _semantic_witness_inputs(qualification, claim, package, binding)
        self._read_parent(qualification, claim, package, binding, self.candidate_for(binding['candidate_version_id']))
        try:
            ref = self.judgments.evaluate(**inputs, proof=self.proof)
        except Exception as exc:
            self._usage_hold(exc,claim)
            raise
        value = tuple(sorted({'contract': SEMANTIC_WITNESS_CONTRACT, 'question_id': 'criterion',
            'invocation_id': ref.invocation_id, 'raw_admission_id': str(ref.raw_admission_id),
            'receipt_admission_id': str(ref.receipt_admission_id)}.items()))
        from dataclasses import replace
        checked = replace(qualification, semantic_witness_ref=value)
        if not self.read(checked, claim, package):
            raise ValueError('semantic witness is not affirmatively verified')
        return value

    def read_existing(self, qualification, claim, package, binding):
        """Resolve one exact existing paid-v1 receipt; never evaluate or allocate."""
        import sqlite3,time
        from pathlib import Path
        from .typesafe_judgment import JudgmentReference
        from .evidence import SEMANTIC_WITNESS_CONTRACT
        self.require_current()
        inputs = _semantic_witness_inputs(qualification,claim,package,binding)
        with sqlite3.connect(Path(self.judgments.usage.path).resolve().as_uri()+'?mode=ro',uri=True)as c:
            c.execute('PRAGMA query_only=ON')
            deadline=time.monotonic()+5;c.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
            rows=c.execute('SELECT a.invocation_id FROM model_invocation_allocations a JOIN model_work_envelopes e USING(envelope_id) '
                'WHERE e.cycle_id=? AND a.route=? LIMIT 2',(inputs['cycle_id'],self.judgments.policy.route)).fetchall()
        if len(rows)!=1:return None
        admitted=self.judgments.objects.committed_admission(ObjectAdmissionRequest('evidence.record','typesafe-receipt:'+rows[0][0]),proof=self.proof)
        if admitted is None:return None
        raw=self.judgments.objects.rehydrate(HydrationRequest(admitted.admission.admission_id,'evidence.record'),proof=self.proof).data
        receipt=json.loads(raw)
        if canonical_json_bytes(receipt)!=raw:raise ValueError('retained semantic witness receipt differs')
        ref=tuple(sorted({'contract':SEMANTIC_WITNESS_CONTRACT,'question_id':'criterion','invocation_id':rows[0][0],
            'raw_admission_id':receipt['raw_admission_id'],'receipt_admission_id':str(admitted.admission.admission_id)}.items()))
        from dataclasses import replace
        return self.read(replace(qualification,semantic_witness_ref=ref),claim,package)

    @staticmethod
    def _usage_hold(error, claim):
        from .typesafe_judgment import TypesafeJudgmentError
        from .native_evidence import NativeEvidenceHold
        unresolved = {'TYPESAFE_EXISTING_INTENT_UNSETTLED','TYPESAFE_UNSETTLED_OR_INVALID_TRANSPORT',
            'TYPESAFE_REPLAY_USAGE_HOLD','TYPESAFE_RESULT_HOLD'}
        if isinstance(error, TimeoutError) or isinstance(error, TypesafeJudgmentError) and str(error) in unresolved:
            raise NativeEvidenceHold('QUALIFICATION_SEMANTIC_WITNESS_UNKNOWN_HOLD',claim.source_ids[0]) from error

    def _read_parent(self, qualification, claim, package, binding, candidate):
        from newsroom.increment10.evidence import _base_package
        if 'source_qualification_reference' in binding:
            from .native_source_qualification_consumer import NativeQualifiedSourceConsumer
            if (getattr(self.parent_reader, '__func__', None) is not NativeQualifiedSourceConsumer.read_semantic_parent
                    or type(getattr(self.parent_reader, '__self__', None)) is not NativeQualifiedSourceConsumer):
                raise ValueError('semantic witness original qualification reader absent')
            parent = self.parent_reader(binding, candidate, _base_package(package), proof=self.proof)
            original = json.loads(parent['materialised_text'])['package']
            selected = next((row for row in original['governed_claims'] if row['claim_id'] == claim.claim_id), None)
            proposed = next((row for row in original['qualification_evidence']
                if row['governed_claim_id'] == claim.claim_id and row['test'] == qualification.test.value), None)
            if (not original['substantive_new_information'] or selected is None or proposed is None
                    or selected['claim'] != claim.claim or selected['claim_role'] != claim.claim_role
                    or selected['status'] != str(claim.status) or selected['source_ids'] != list(claim.source_ids)
                    or proposed['test_evidence'] != dict(qualification.test_evidence)):
                raise ValueError('semantic witness original qualification differs')

    def read(self, qualification, claim, package):
        if qualification is None:
            return self._read_source_rendering(claim, package)
        from .evidence import semantic_witness_reference
        from .typesafe_judgment import JudgmentReference
        from newsroom.authority import HydrationRequest
        self.require_current()
        value = semantic_witness_reference(qualification.semantic_witness_ref)
        ref = JudgmentReference(value['invocation_id'], ObjectAdmissionId.parse(value['raw_admission_id']),
            ObjectAdmissionId.parse(value['receipt_admission_id']))
        raw = self.judgments.objects.rehydrate(HydrationRequest(ref.receipt_admission_id, 'evidence.record'), proof=self.proof).data
        record = json.loads(raw)
        if canonical_json_bytes(record) != raw or record.get('invocation_id') != ref.invocation_id:
            raise ValueError('semantic witness receipt differs')
        binding = record['snapshot']['source_binding']
        candidate = self.candidate_for(binding['candidate_version_id'])
        if (candidate.version_id != binding['candidate_version_id']
                or candidate.candidate_id != package.candidate_id
                or candidate.governing_manifest.hypothesis_id != package.hypothesis_id
                or candidate.governing_manifest.canonical_digest != binding.get('hypothesis_digest')):
            raise ValueError('semantic witness current Candidate differs')
        self._read_parent(qualification, claim, package, binding, candidate)
        inputs = _semantic_witness_inputs(qualification, claim, package, binding)
        if value['question_id'] != 'criterion':
            raise ValueError('semantic witness question differs')
        try:
            verified = self.judgments.read(ref, **inputs, proof=self.proof)
        except Exception as exc:
            self._usage_hold(exc,claim)
            raise
        from .native_evidence import NativeEvidenceHold
        if verified.get('outcome') != 'TYPESAFE_COMPLETE':
            raise NativeEvidenceHold('QUALIFICATION_SEMANTIC_WITNESS_UNKNOWN_HOLD',claim.source_ids[0])
        answer = verified['answers']['criterion']
        choice = answer.get('choice')
        if choice in {'NO','UNCERTAIN'}:
            held = NativeEvidenceHold('QUALIFICATION_SEMANTIC_WITNESS_'+choice,claim.source_ids[0])
            held.semantic_witness_disposition = {'reference':value,'confidence_ppm':answer['confidence_ppm'],
                'probabilities_ppm':answer['probabilities_ppm']}
            raise held
        return choice == 'YES'

    def _read_source_rendering(self, claim, package):
        from .evidence import source_rendering_reference, _localised_fact_is_bound, SOURCE_RENDERING_CONTRACT_V2
        from .native_source_qualification import VERSION as QA_VERSION
        from .native_source_qualification_consumer import NativeQualifiedSourceConsumer
        from newsroom.increment10.evidence import _base_package
        self.require_current()
        value = source_rendering_reference(claim.source_rendering_ref)
        if value['contract'] == SOURCE_RENDERING_CONTRACT_V2:
            return self._read_typed_source_rendering(value, claim, package)
        if (getattr(self.parent_reader,'__func__',None) is not NativeQualifiedSourceConsumer.read_semantic_parent
                or type(getattr(self.parent_reader,'__self__',None)) is not NativeQualifiedSourceConsumer):
            raise ValueError('Source rendering parent reader absent')
        raw = self.judgments.objects.rehydrate(HydrationRequest(ObjectAdmissionId.parse(value['receipt_admission_id']),
            'evidence.record'),proof=self.proof).data
        receipt = json.loads(raw)
        if canonical_json_bytes(receipt) != raw or receipt.get('version') != QA_VERSION:
            raise ValueError('Source rendering parent receipt differs')
        binding = {**receipt['source_binding'],'source_qualification_reference':{key:value[key]for key in
            ('invocation_id','raw_admission_id','receipt_admission_id')}}
        candidate = self.candidate_for(binding['candidate_version_id'])
        if candidate.governing_manifest.hypothesis_id != package.hypothesis_id:
            raise ValueError('Source rendering Candidate differs')
        parent = self.parent_reader(binding,candidate,_base_package(package),proof=self.proof)
        document = json.loads(parent['materialised_text'])['package']
        original = next((row for row in document['governed_claims'] if row['claim_id'] == claim.claim_id),None)
        if (not document['substantive_new_information'] or original is None or any(original[key] != getattr(claim,key)
                for key in ('claim','supporting_excerpt','claim_role','passage_index'))
                or original['status'] != str(claim.status) or original['source_ids'] != list(claim.source_ids)
                or original['quotations'] != list(claim.quotations)):
            raise ValueError('Source rendering original claim differs')
        body = package.passages[claim.passage_index]
        chronology = binding['current_scope']['sources'][claim.passage_index]
        terms, year = source_rendering_details(claim,body,chronology)
        expected_names = source_rendering_names(claim,body)
        if frozenset((name,kind)for name,kind,_ref in claim.named_entity_evidence) != expected_names:
            raise ValueError('Source rendering literal identities differ')
        for source,target in claim.localised_factual_expressions:
            if not _localised_fact_is_bound(source,target,claim.claim,claim.supporting_excerpt,claim.rendered_assertion_zh_hant_hk):
                if year is None or (source,target) != year[:2] or target not in claim.rendered_assertion_zh_hant_hk:
                    raise ValueError('Source rendering factual derivation differs')
        if year is not None and (year[1] not in claim.rendered_assertion_zh_hant_hk
                or year[:2] not in claim.localised_factual_expressions):
            raise ValueError('Source rendering anchored year omitted')
        return True


    def _read_typed_source_rendering(self, value, claim, package):
        """Authenticate literal/fact rendering only; never certify selection or YES."""
        from .native_claim_localisation import NativeClaimLocaliser, LocalisationReference, TYPED_VERSION
        from .evidence import factual_rendering_is_bound_v2
        from newsroom.increment10.evidence import _base_package
        if (getattr(self.rendering_reader, '__func__', None) is not NativeClaimLocaliser.read_localisation
                or type(getattr(self.rendering_reader, '__self__', None)) is not NativeClaimLocaliser):
            raise ValueError('typed Source rendering reader absent')
        raw = self.judgments.objects.rehydrate(HydrationRequest(ObjectAdmissionId.parse(value['receipt_admission_id']),
            'evidence.record'), proof=self.proof).data
        receipt = json.loads(raw)
        if canonical_json_bytes(receipt) != raw or receipt.get('version') != TYPED_VERSION:
            raise ValueError('typed Source rendering receipt differs')
        state = {'source_binding': receipt['source_binding'], 'claims': receipt['original_claims']}
        binding = state['source_binding']; base = _base_package(package)
        current = binding['current_scope']['sources']
        candidate = self.candidate_for(binding['candidate_version_id'])
        if (binding['candidate_id'] != base.candidate_id or binding['content_digest'] != base.digest
                or binding['evidence_package_digest'] != base.digest
                or candidate.candidate_id != base.candidate_id
                or candidate.governing_manifest.hypothesis_id != package.hypothesis_id
                or tuple(row['body'] for row in current) != base.passages
                or tuple(row['source_id'] for row in current) != base.source_ids):
            raise ValueError('typed Source rendering complete Source differs')
        ref = LocalisationReference(value['invocation_id'], ObjectAdmissionId.parse(value['raw_admission_id']),
                                    ObjectAdmissionId.parse(value['receipt_admission_id']))
        checked = self.rendering_reader(ref, state, proof=self.proof,
            **{key: binding[key] for key in ('candidate_id', 'hypothesis_digest', 'evidence_package_digest')})
        if checked.get('original_state') != state or checked.get('projected_state') != source_rendering_projection(state):
            raise ValueError('typed Source rendering authenticated projection differs')
        view = build_lossless_source_view(base.passages, base.source_ids,
            version=binding.get('source_reference_binding', {}).get('partition_version', 'newsroom.native-assessor-spans.v2'))
        for identity, selected in state['claims'].items():
            if selected['text'] != claim.claim or selected['source_id'] != claim.source_ids[0]:
                continue
            if selected.get('source_range'):
                text, passage, source_id = view.resolve_range(selected['source_range'])
                if text != claim.claim or passage != claim.passage_index or source_id != claim.source_ids[0]:
                    continue
            elif current[claim.passage_index]['source_id'] != selected['source_id'] or selected['text'] not in current[claim.passage_index]['body']:
                continue
            projected = checked['projected_state']['claims'][identity]; item = checked['renderings'][identity]
            fragments = item['rendered_assertion_zh_hant_hk_fragments']
            rendered = fragments[0] + ''.join(name+fragment for (name, _kind), fragment in
                zip(projected['entities'], fragments[1:], strict=True))
            expected = frozenset(tuple(entity) for entity in projected['entities'])
            pairs = tuple((p['source_lookup_key'], p['rendered_expression']) for p in item['factual_localisations'])
            if (rendered == claim.rendered_assertion_zh_hant_hk
                    and expected == frozenset((name, kind) for name, kind, _ref in claim.named_entity_evidence)
                    and pairs == tuple(claim.localised_factual_expressions)
                    and tuple(item['quotation_source_keys']) == tuple(claim.quotations)
                    and factual_rendering_is_bound_v2(claim.claim, rendered, pairs,
                        literals=tuple(name for name, _kind in expected),
                        derived_pairs=tuple(map(tuple, selected.get('source_derived_facts', ()))))):
                return True
        raise ValueError('typed Source rendering claim/literals differ')


def semantic_witness_reader_is_bound(reader):
    return (getattr(reader, '__func__', None) is NativeSemanticWitnesses.read
            and type(getattr(reader, '__self__', None)) is NativeSemanticWitnesses)
