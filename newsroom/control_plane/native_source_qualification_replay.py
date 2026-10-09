"""Read one exact accounted Source qualification result under fresh Source proof."""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
from newsroom.authority import ObjectAdmissionId, ObjectAdmissionRequest, HydrationRequest
from newsroom.authority.canonical import canonical_json_bytes, digest_canonical
from .model_usage import UsageStatus, ModelUsageIntegrityError, _retained_terminal_allocation
from .native_source_qualification import QualificationHold, QualificationReference, VERSION, ROUTE
from .native_assessor import NativeAssessmentExecution
from .native_assessor_spans import build_lossless_source_view


def original_qualification_reference(qualifier, candidate, base, *, proof, optional=False, include_resolution=False):
    """One original plus at most one authenticated resolution, with overflow HOLD."""
    with sqlite3.connect(Path(qualifier.usage.path).resolve().as_uri()+'?mode=ro', uri=True) as c:
        c.execute('PRAGMA query_only=ON')
        deadline = time.monotonic() + 5
        c.set_progress_handler(lambda: time.monotonic() > deadline, 1000)
        rows = c.execute("SELECT a.invocation_id FROM model_invocation_allocations a JOIN model_work_envelopes e USING(envelope_id) "
            "JOIN model_invocation_policies p ON p.canonical_digest=a.policy_digest "
            "WHERE a.route=? AND json_extract(e.record_json,'$.candidate_id')=? "
            "AND json_extract(e.record_json,'$.hypothesis_digest')=? "
            "AND json_extract(e.record_json,'$.evidence_package_digest')=? "
            "AND json_extract(p.record_json,'$.prompt_contract_version')=? LIMIT 3",
            (ROUTE, candidate.candidate_id, candidate.governing_manifest.canonical_digest, base.digest, VERSION)).fetchall()
        if not rows and optional:
            return None
        if not rows or len(rows) > 2:
            raise QualificationHold('QUALIFICATION_KNOWN_RESULT_ABSENT_OR_AMBIGUOUS')
        values = []
        for (invocation,) in rows:
            try:
                _allocation, terminal = _retained_terminal_allocation(c, invocation)
            except ModelUsageIntegrityError as error:
                if len(rows) > 1:
                    raise QualificationHold('QUALIFICATION_KNOWN_RESULT_ABSENT_OR_AMBIGUOUS') from error
                raise
            if terminal is None or terminal.usage_status is not UsageStatus.REPORTED or terminal.outcome != 'QUALIFICATION_COMPLETE' or terminal.policy_breach:
                if len(rows) > 1:
                    raise QualificationHold('QUALIFICATION_KNOWN_RESULT_ABSENT_OR_AMBIGUOUS')
                raise QualificationHold('QUALIFICATION_REPLAY_USAGE_HOLD')
            admitted = qualifier.objects.committed_admission(ObjectAdmissionRequest('evidence.record',
                'source-qualification-receipt:'+invocation), proof=proof)
            if admitted is None:
                raise QualificationHold('QUALIFICATION_PRIOR_RESULT_UNAVAILABLE')
            raw = qualifier.objects.rehydrate(HydrationRequest(admitted.admission.admission_id, 'evidence.record'), proof=proof).data
            receipt = json.loads(raw)
            if canonical_json_bytes(receipt) != raw or receipt.get('invocation_id') != invocation:
                raise QualificationHold('QUALIFICATION_REPLAY_BINDING_HOLD')
            ref = QualificationReference(invocation, ObjectAdmissionId.parse(receipt['raw_admission_id']), admitted.admission.admission_id)
            values.append((ref, receipt))
    originals = [(ref, row) for ref, row in values if 'semantic_resolution' not in row['source_binding']]
    if len(originals) != 1:
        raise QualificationHold('QUALIFICATION_KNOWN_RESULT_ABSENT_OR_AMBIGUOUS')
    reference, receipt = originals[0]
    resolution = None
    for ref, row in values:
        if ref == reference:
            continue
        from .native_source_qualification_consumer import _resolution_state, _require_resolution_result, _corrected_resolution_package
        from .evidence import SEMANTIC_RESOLUTION_CONTRACT, SEMANTIC_RESOLUTION_CONTRACT_V2
        marker = row['source_binding']['semantic_resolution']
        parent = {'invocation_id': reference.invocation_id, 'raw_admission_id': str(reference.raw_admission_id),
                  'receipt_admission_id': str(reference.receipt_admission_id)}
        if set(marker) != {'contract', 'parent', 'witnesses'} or marker['contract'] not in {
                SEMANTIC_RESOLUTION_CONTRACT, SEMANTIC_RESOLUTION_CONTRACT_V2} or marker['parent'] != parent:
            raise QualificationHold('QUALIFICATION_RESOLUTION_PARENT_HOLD')
        state, package = _resolution_state(qualifier, candidate, base, parent, marker['witnesses'], proof=proof,
            contract=marker['contract'])
        result = qualifier.read_qualification(ref, state, proof=proof, candidate_id=candidate.candidate_id,
            hypothesis_digest=candidate.governing_manifest.canonical_digest, evidence_package_digest=base.digest)
        if marker['contract'] == SEMANTIC_RESOLUTION_CONTRACT_V2:
            _corrected_resolution_package(result, state, base)
        else:
            _require_resolution_result(result, package)
        resolution = (ref, row)
    if include_resolution:
        return reference, receipt, resolution
    return reference, receipt

def read_current_result(qualifier, candidate, base, sources, acquired, *, scope, proof):
    """Read one proved original retained result; never allocate or call."""
    from copy import deepcopy
    from .native_assessor_judgments import NativeAssessorJudgments, JudgedAssessment
    if (base.source_ids != tuple(source.unit.source_id for source in sources)
            or base.passages != tuple(item.body.decode('utf-8') for item in acquired)):
        raise QualificationHold('QUALIFICATION_ACQUIRED_BYTES_HOLD')
    ids = dict(candidate_id=candidate.candidate_id,
        hypothesis_digest=candidate.governing_manifest.canonical_digest, evidence_package_digest=base.digest)
    reference, receipt = original_qualification_reference(qualifier, candidate, base, proof=proof)
    binding = receipt['source_binding']
    view = build_lossless_source_view(base.passages, base.source_ids)
    current = NativeAssessorJudgments._binding(candidate,base,scope,view)
    original = {key:value for key,value in binding.items() if key not in {'qualification_contract','prior_judgments','failure_inventory'}}
    def stable(value):
        value=deepcopy(value)
        for row in value.get('current_scope',{}).get('sources',[]):
            row.pop('retrieved_at',None)
        for row in value.get('first_publication',[]):
            row.pop('acquisition_receipt_digest',None)
        return value
    if stable(current) != stable(original):
        raise QualificationHold('QUALIFICATION_CURRENT_SNAPSHOT_HOLD')
    if scope.get('newness')=='SOURCE_DECLARED_FIRST_PUBLICATION' and not NativeAssessorJudgments._first_publication_proven(scope,sources,acquired):
        raise QualificationHold('QUALIFICATION_FIRST_PUBLICATION_HOLD')
    state = original_qualification_state(qualifier, candidate, base, binding, proof=proof)
    checked=qualifier.read_qualification(reference,state,proof=proof,**ids)
    expected={'schema':VERSION,'source_binding':binding,'materialisation_receipt':checked['materialisation'],
        'qualification_reference':{'invocation_id':reference.invocation_id,'raw_admission_id':str(reference.raw_admission_id),
            'receipt_admission_id':str(reference.receipt_admission_id)}}
    decision=qualifier.objects.committed_admission(ObjectAdmissionRequest('evidence.record','source-qualification-decision:'+digest_canonical(state)),proof=proof)
    if decision is None:
        raise QualificationHold('QUALIFICATION_PRIOR_DECISION_UNAVAILABLE')
    raw=qualifier.objects.rehydrate(HydrationRequest(decision.admission.admission_id,'evidence.record'),proof=proof).data
    if raw!=canonical_json_bytes(expected):
        raise QualificationHold('QUALIFICATION_PRIOR_DECISION_CHANGED')
    with qualifier.fence(current,proof):
        original = JudgedAssessment(NativeAssessmentExecution(checked['materialisation']['materialised_text'],{}),raw,decision.admission.admission_id)
    return original


def original_qualification_state(qualifier, candidate, base, binding, *, proof):
    """Reconstruct the supported original paid recipe, with no producer dispatch."""
    import re
    from .native_assessor_judgments import source_role_questions, VERSION as JUDGMENT_VERSION
    from .qualification_rubrics import witness_inventory, question as qualification_question
    from .native_assessor import PROVIDER_SCHEMA
    from .typesafe_judgment import JudgmentReference
    view = build_lossless_source_view(base.passages, base.source_ids)
    from .native_assessor import _reference_binding
    if (binding.get('candidate_id') != candidate.candidate_id
            or binding.get('candidate_version_id') != candidate.version_id
            or binding.get('hypothesis_digest') != candidate.governing_manifest.canonical_digest
            or binding.get('content_digest') != base.digest
            or binding.get('evidence_package_digest') != base.digest
            or binding.get('source_reference_binding') != _reference_binding(view)):
        raise QualificationHold('QUALIFICATION_PARENT_INPUT_HOLD')
    current = binding.get('current_scope', {})
    if (tuple(row.get('source_id')for row in current.get('sources',())) != base.source_ids
            or tuple(row.get('body')for row in current.get('sources',())) != base.passages):
        raise QualificationHold('QUALIFICATION_PARENT_SOURCE_HOLD')
    ids = dict(candidate_id=candidate.candidate_id, hypothesis_digest=candidate.governing_manifest.canonical_digest)
    original = {key:value for key,value in binding.items()
                if key not in {'qualification_contract','prior_judgments','failure_inventory'}}
    refs=binding.get('prior_judgments',[])
    inventory=binding.get('failure_inventory')
    if (binding.get('qualification_contract')!=VERSION or type(refs)is not list
            or len(refs)not in {0,1,2} or type(inventory)is not list or len(inventory)!=1
            or type(inventory[0])is not dict
            or any(type(row)is not dict or set(row)!={'invocation_id','raw_admission_id','receipt_admission_id'}
                   or any(type(value)is not str for value in row.values())for row in refs)
            or len({row['invocation_id']for row in refs})!=len(refs)):
        raise QualificationHold('QUALIFICATION_ORIGINAL_RECIPE_UNSUPPORTED')
    failure=inventory[0]
    if failure=={'stage':'SELECTED_QUALIFICATION','reason':'INPUT_BOUND'} and len(refs)==1:
        reason='JUDGMENT_INPUT_BOUND'
    elif failure=={'reason':'QUALIFICATION_WITNESS_COVERAGE_UNPROVEN'} and len(refs)==1:
        reason='QUALIFICATION_WITNESS_COVERAGE_UNPROVEN'
    elif failure=={'reason':'FIRST_PUBLICATION_ANNOUNCEMENT_UNPROVEN'} and len(refs)==2:
        reason='FIRST_PUBLICATION_ANNOUNCEMENT_UNPROVEN'
    elif failure=={'reason':'TYPED_OUTPUT_CONTRACT_UNPROVEN'} and len(refs)==2:
        reason='MATERIALISATION_VALIDATION_FAILED'
    elif (set(failure)=={'question_id','reason'} and failure['reason']=='NONE'
            and type(failure['question_id'])is str and len(refs)==2):
        reason='QUALIFICATION_WITNESS_MISSING'
    elif set(failure)=={'reason'} and type(failure['reason'])is str and failure['reason']:
        # Plain-reason fallbacks use the same frozen recipe, including the
        # zero-judgment early exit. The original request/manifest authenticates
        # the label and full reconstructed input; it is not a new model request.
        reason=failure['reason']
    else:
        raise QualificationHold('QUALIFICATION_ORIGINAL_RECIPE_UNSUPPORTED')
    candidates={segment.span_id:{'source_id':segment.source_id,
        'text':segment.text.encode('utf-8')[:segment.content_end_byte-segment.start_byte].decode('utf-8'),
        'entities':[list(item)for item in segment.entities],'rendering_fragment_count':len(segment.entities)+1,
        'source_range':{'first_span_id':segment.span_id,'last_span_id':segment.span_id}}for segment in view.segments}
    role_state={'candidates':candidates,'current_scope':binding['current_scope'],'prior_scope':binding['prior_scope']}
    if binding['newness']=='SOURCE_DECLARED_FIRST_PUBLICATION':
        role_state['publication_basis']={'mode':binding['newness'],
            'published_at':{row['source_id']:row['first_published_at']for row in binding['first_publication']}}
    def read_batch(row, phase, state, questions):
        ref=JudgmentReference(row['invocation_id'],ObjectAdmissionId.parse(row['raw_admission_id']),
                              ObjectAdmissionId.parse(row['receipt_admission_id']))
        record=qualifier.judgments.read(ref,state=state,questions=questions,source_binding=original,
            caller_identity='NATIVE_ASSESSOR',cycle_id=digest_canonical([JUDGMENT_VERSION,phase,original,state,questions]),
            candidate_id=candidate.candidate_id,hypothesis_digest=ids['hypothesis_digest'],proof=proof)
        return {'questions':questions,'answers':record['answers'],'outcome':record['outcome']}
    judgments=([read_batch(refs[0],'SOURCE_ROLES',role_state,source_role_questions(candidates))]
               if refs else [])
    if len(refs)==2:
        material=[identity for identity,answer in judgments[0]['answers'].items()
                  if answer['choice']=='MATERIAL']
        witnesses=witness_inventory(view)
        selected_state={**role_state,'witness_inventory':witnesses}
        variants=PROVIDER_SCHEMA['properties']['package']['properties']['qualification_evidence']['items']['oneOf']
        rules={v['properties']['test']['const']:v['properties']['test_evidence']['properties']for v in variants}
        questions={'headline':{'type':'choice','instructions':'Choose the strongest material headline; use UNCERTAIN for unresolved support.',
            'criteria':{**{identity:candidates[identity]['text']for identity in material},'UNCERTAIN':'Unresolved headline.'}}}
        # Reproduce the frozen selected-batch recipe; the authenticated reader
        # checks its original state/question snapshots before exposing answers.
        for identity in material:
            if binding['newness']=='SOURCE_DECLARED_FIRST_PUBLICATION':
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
                    if 'enum'in schema:
                        choices={value:value for value in schema['enum']}
                    elif field.endswith('_source_lookup_key'):
                        choices={key:value['text']for key,value in witnesses[identity]['candidates'].items()}
                    elif field=='duration_minutes':
                        choices={m.group(1):m.group(1)for m in re.finditer(r'(?<![\d.,])([0-9]+)\s*(?:minutes?|mins?|分鐘)(?![A-Za-z])',candidates[identity]['text'],re.I)}
                    else:
                        continue
                    questions[prefix+':'+field]={'type':'choice','instructions':f'For {identity}/{test}, select source-supported {field}; NONE if not established.',
                        'criteria':{**choices,'NONE':'No established value.'}}
        judgments.append(read_batch(refs[1],'SELECTED_QUALIFICATION',selected_state,questions))
    source_rows=binding['current_scope']['sources']
    state={'source_binding':binding,'source_view':{'passages':list(base.passages),'source_ids':list(base.source_ids),
        'sources':[{'source_id':row['source_id'],'publication_time':row['published_at'],
            'source_updated_time':row['updated_at'],'retrieval_time':row['retrieved_at'],
            'segments':[{**segment.request_record(),'rendering_fragment_count':len(segment.entities)+1}
                for segment in view.segments if segment.source_id==row['source_id']]}for row in source_rows]},
        'issue':{'reason':reason,'failed_questions':inventory,'newness':binding['newness'],'prior_scope':binding['prior_scope']},
        'judgments':judgments}
    return state
