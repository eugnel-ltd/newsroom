"""One Source-bound context purpose; original headline qualification is immutable."""
from __future__ import annotations

from copy import deepcopy
import json

from newsroom.authority import HydrationRequest, ObjectAdmissionRequest
from newsroom.authority.canonical import canonical_json_bytes, digest_canonical
from .native_assessor_judgments import JudgedAssessment, NativeAssessorJudgments
from .native_assessor_spans import build_lossless_source_view

VERSION = 'newsroom.native-context-package.v2'
SUPPORT_CONTRACT = 'newsroom.native-context-support.assembled.v1'


class ContextEnrichmentHold(ValueError):
    pass


def _stable_binding(binding):
    value=deepcopy(binding)
    for source in value.get('current_scope',{}).get('sources',[]):
        source.pop('retrieved_at',None)
    for source in value.get('first_publication',[]):
        source.pop('acquisition_receipt_digest',None)
    return value


def _reference(reference):
    return {name:str(getattr(reference,name))for name in
        ('invocation_id','raw_admission_id','receipt_admission_id')}


def _candidates(view, original):
    package=json.loads(original.execution.text)['package']
    covered={claim['claim']for claim in package['governed_claims']}
    from .native_source_context_ranges import context_candidates
    return {identity: candidate for identity, candidate in context_candidates(view).items()
            if not any(candidate['text'].strip() in claim for claim in covered)}


class NativeContextEnricher:
    """Reuse existing accounted judgments, rendering and governed object ports."""
    def __init__(self, *, judgments, localiser, objects, proof, require_current):
        self.judgments,self.localiser,self.objects=judgments,localiser,objects
        self.proof,self.require_current=proof,require_current

    def _read(self, admission):
        raw=self.objects.rehydrate(HydrationRequest(admission,'evidence.record'),proof=self.proof).data
        record=json.loads(raw)
        if canonical_json_bytes(record)!=raw:
            raise ContextEnrichmentHold('CONTEXT_RECORD_CANONICAL_HOLD')
        return record

    def _batch(self, phase, state, questions, binding, candidate):
        inputs=dict(state=state,questions=questions,source_binding=binding,
            cycle_id=digest_canonical([VERSION,phase,binding,state,questions]),
            caller_identity='NATIVE_ASSESSOR',candidate_id=candidate.candidate_id,
            hypothesis_digest=candidate.governing_manifest.canonical_digest,proof=self.proof)
        self.require_current()
        reference=self.judgments.evaluate(**inputs)
        record=self.judgments.read(reference,**inputs)
        if set(record.get('answers',{}))!=set(questions):
            raise ContextEnrichmentHold('CONTEXT_ANSWER_INVENTORY_HOLD')
        return reference,record

    def enrich(self, original, candidate, base, sources, acquired, *, scope):
        if type(original)is not JudgedAssessment or original.decision_admission_id is None:
            raise ContextEnrichmentHold('CONTEXT_ORIGINAL_RESULT_HOLD')
        self.require_current()
        original_record=self._read(original.decision_admission_id)
        if canonical_json_bytes(original_record)!=original.decision_record:
            raise ContextEnrichmentHold('CONTEXT_ORIGINAL_RECORD_HOLD')
        if (base.source_ids!=tuple(source.unit.source_id for source in sources)
                or base.passages!=tuple(item.body.decode()for item in acquired)):
            raise ContextEnrichmentHold('CONTEXT_SOURCE_BYTES_HOLD')
        view=build_lossless_source_view(base.passages,base.source_ids)
        binding=NativeAssessorJudgments._binding(candidate,base,scope,view)
        intent_key='native-context-input:'+digest_canonical([VERSION,
            str(original.decision_admission_id),_stable_binding(binding)])
        request=ObjectAdmissionRequest('evidence.record',intent_key)
        retained=self.objects.committed_admission(request,proof=self.proof)
        if retained is None:
            candidates=_candidates(view,original)
            if not candidates or len(candidates)>32:
                raise ContextEnrichmentHold('CONTEXT_CANDIDATE_COUNT_HOLD')
            state={'version':VERSION,'source_binding':binding,
                'original_admission_id':str(original.decision_admission_id),
                'original_record_digest':digest_canonical(original_record),
                'original_execution_digest':digest_canonical(json.loads(original.execution.text)),
                'candidates':candidates}
            self.require_current()
            admitted=self.objects.admit(request,canonical_json_bytes(state),proof=self.proof).admission
            state=self._read(admitted.admission_id)
        else:
            state=self._read(retained.admission.admission_id)
        if (state.get('version')!=VERSION
                or state.get('original_record_digest')!=digest_canonical(original_record)
                or state.get('original_execution_digest')!=digest_canonical(json.loads(original.execution.text))
                or _stable_binding(state.get('source_binding',{}))!=_stable_binding(binding)
                or state.get('candidates')!=_candidates(view,original)):
            raise ContextEnrichmentHold('CONTEXT_INTENT_SOURCE_DRIFT_HOLD')
        binding=state['source_binding'];candidates=state['candidates']
        public={'sources':list(base.passages),'headline':[
            {key:claim[key]for key in ('claim','supporting_excerpt','claim_role','status',
                'rendered_assertion_zh_hant_hk')}
            for claim in json.loads(original.execution.text)['package']['governed_claims']],
            'candidates':candidates,
            'publication':[{key:value for key,value in source.items()
                if key in {'source_id','published_at','updated_at'}}
                for source in binding.get('current_scope',{}).get('sources',[])]}
        questions={identity:{'type':'choice',
            'instructions':f'For exact candidate {identity}, select useful context for the supplied qualified headline. '
                'Omit any candidate marked speaker_parent_hold. Require an affirmed source assertion, full parent support, exact attribution and preserved '
                'proposal/future/conditional/quoted status. A proposal is not in force. Exclude administration and filler.',
            'criteria':{'INCLUDE':'Useful, source-supported topic, detail or attribution with exact modality.',
                'OMIT':'Irrelevant, duplicated, administrative or unsupported context.',
                'UNCERTAIN':'Support, relevance, attribution or modality is unresolved.'}}
            for identity in candidates}
        selection_ref,selection=self._batch('CONTEXT_SELECTION',public,questions,binding,candidate)
        if any(answer.get('choice')=='UNCERTAIN'for answer in selection['answers'].values()):
            raise ContextEnrichmentHold('CONTEXT_SELECTION_UNCERTAIN_HOLD')
        selected={identity:item for identity,item in candidates.items()
            if selection['answers'][identity].get('choice')=='INCLUDE'}
        if not selected:
            raise ContextEnrichmentHold('CONTEXT_NOT_ESTABLISHED_HOLD')
        if any(item.get('speaker_parent_hold') for item in selected.values()):
            raise ContextEnrichmentHold('CONTEXT_SOURCE_SPEAKER_UNRESOLVED_HOLD')
        context_binding={**binding,'context_purpose':VERSION,
            'context_original_receipt_digest':digest_canonical(original_record['materialisation_receipt']),
            'context_ranges':{identity:item['source_range']for identity,item in selected.items()}}
        localisation_input={'source_binding':context_binding,'claims':selected}
        identities={'candidate_id':candidate.candidate_id,
            'hypothesis_digest':candidate.governing_manifest.canonical_digest,'evidence_package_digest':base.digest}
        self.require_current()
        rendering_ref=self.localiser.localise(localisation_input,proof=self.proof,**identities)
        rendering=self.localiser.read_localisation(rendering_ref,localisation_input,proof=self.proof,**identities)
        from .native_context_materialisation import context_renderings
        renderings = context_renderings(rendering)
        wire={'package':{'select_new_information':False,'governed_claims':[
            {'claim_role':'CONTEXT','status':'CONFIRMED_FACT','source_range':item['source_range'],
                **renderings[identity]}for identity,item in selected.items()],
            'qualification_evidence':[],'selection_rationale':'Source-bound supporting context.',
            'geography':[],'categories':[],'explicit_exclusions':[]}}
        from .native_assessor import _materialise_reference_result, VERSION as CODEC
        materialised, _ = _materialise_reference_result(canonical_json_bytes(wire), view,
            digest_canonical(context_binding), CODEC)
        assertions = {identity: claim['rendered_assertion_zh_hant_hk']
            for identity, claim in zip(selected, materialised['package']['governed_claims'], strict=True)}
        verification_state={**public,'selected':selected,'renderings':renderings,
            'rendered_assertions':assertions,'support_contract':SUPPORT_CONTRACT}

        criteria={'support':'The exact complete Source supports the asserted context, not an inferred fact.',
            'modality':'Rendering preserves negation, proposal, future, conditional and provisional meaning.',
            'attribution':'Quoted or first-person assertions preserve the exact Source speaker and parent, never replace it with the publisher or become confirmed outcomes.',
            'entities':'Verify the rendered_assertions full application-materialised text, not the separate assembly fragments. All entity slots have already been filled with their exact Source names; no translated or added alias.'}
        checks={identity+':'+kind:{'type':'choice','instructions':f'Verify {kind} for {identity}: {instruction}',
            'criteria':{('SUPPORTED'if kind=='support'else'YES'):'Established in exact full Source and rendering.',
                ('UNSUPPORTED'if kind=='support'else'NO'):'Contradicted, changed or unsupported.','UNCERTAIN':'Unresolved.'}}
            for identity in selected for kind,instruction in criteria.items()}
        # A new input contract is not credit to repeat an unknown old batch.
        legacy_state={**public,'selected':selected,'renderings':renderings}
        legacy_checks=deepcopy(checks)
        for identity in selected:
            legacy_checks[identity+':entities']['instructions']=(
                f'Verify entities for {identity}: All rendered entities are exactly the supplied Source identities; no translated or added alias.')
        legacy_cycle=digest_canonical([VERSION,'CONTEXT_SUPPORT',context_binding,legacy_state,legacy_checks])
        from .model_usage import _retained_terminal_allocation, UsageStatus
        with self.judgments.usage._connection() as c:
            prior=c.execute('SELECT a.invocation_id FROM model_invocation_allocations a '
                'JOIN model_work_envelopes e ON e.envelope_id=a.envelope_id '
                'WHERE e.cycle_id=? AND e.workload_class=?', (legacy_cycle,'TYPESAFE_JUDGMENT')).fetchall()
            if len(prior)>1:
                raise ContextEnrichmentHold('CONTEXT_PRIOR_SUPPORT_AMBIGUOUS_HOLD')
            if prior:
                _allocation,terminal=_retained_terminal_allocation(c,prior[0][0])
                if terminal.usage_status is not UsageStatus.REPORTED or terminal.outcome!='TYPESAFE_COMPLETE' or terminal.policy_breach:
                    raise ContextEnrichmentHold('CONTEXT_PRIOR_SUPPORT_UNSETTLED_HOLD')
        support_ref,support=self._batch(SUPPORT_CONTRACT,verification_state,checks,context_binding,candidate)
        if any(answer.get('choice')!=('SUPPORTED'if identity.endswith(':support')else'YES')
                for identity,answer in support['answers'].items()):
            raise ContextEnrichmentHold('CONTEXT_SUPPORT_UNPROVEN_HOLD')
        from .native_context_materialisation import compose_context_execution
        execution,composition=compose_context_execution(original,wire,rendering,
            binding=context_binding,view=view,support_receipt=support)
        record={'version':VERSION,'source_binding':context_binding,'intent_key':intent_key,
            'original_admission_id':str(original.decision_admission_id),
            'selection_reference':_reference(selection_ref),
            'support_reference':_reference(support_ref),
            'localisation_reference':_reference(rendering_ref),
            'composition':composition,'execution':json.loads(execution.text)}
        from .native_claim_localisation import TYPED_VERSION
        from .native_assessor_judgments import SourceRenderingMetadata
        source_renderings = original.source_renderings
        if rendering.get('version') == TYPED_VERSION:
            from .evidence import SOURCE_RENDERING_CONTRACT_V2
            ref = tuple(sorted({'contract': SOURCE_RENDERING_CONTRACT_V2,
                'operation': 'SOURCE_RENDERING', **_reference(rendering_ref)}.items()))
            original_ids = {row['claim_id'] for row in json.loads(original.execution.text)['package']['governed_claims']}
            context_refs = tuple((row['claim_id'], ref) for row in record['execution']['package']['governed_claims']
                                 if row['claim_id'] not in original_ids)
            source_renderings = SourceRenderingMetadata(
                (() if source_renderings is None else source_renderings.references) + context_refs)
            record['source_renderings'] = [[key, dict(value)] for key, value in source_renderings.references]
        self.require_current()
        admitted=self.objects.admit(ObjectAdmissionRequest('evidence.record',
            'native-context-package:'+digest_canonical(record)),canonical_json_bytes(record),proof=self.proof).admission
        return JudgedAssessment(execution,canonical_json_bytes(record),admitted.admission_id,
            semantic_witnesses=original.semantic_witnesses, source_renderings=source_renderings)
