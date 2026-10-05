"""Read one exact accounted Source qualification result under fresh Source proof."""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
from newsroom.authority import ObjectAdmissionId, ObjectAdmissionRequest, HydrationRequest
from newsroom.authority.canonical import canonical_json_bytes, digest_canonical
from .model_usage import UsageStatus, _retained_terminal_allocation
from .native_source_qualification import QualificationHold, QualificationReference, VERSION, ROUTE
from .native_assessor import NativeAssessmentExecution
from .native_assessor_spans import build_lossless_source_view

def read_current_result(qualifier, candidate, base, sources, acquired, *, scope, proof):
    """Read one proved original role/input-bound result; never allocate or call."""
    from copy import deepcopy
    from .native_assessor_judgments import NativeAssessorJudgments, JudgedAssessment, source_role_questions, VERSION as JUDGMENT_VERSION
    from .typesafe_judgment import JudgmentReference
    if (base.source_ids != tuple(source.unit.source_id for source in sources)
            or base.passages != tuple(item.body.decode('utf-8') for item in acquired)):
        raise QualificationHold('QUALIFICATION_ACQUIRED_BYTES_HOLD')
    ids = dict(candidate_id=candidate.candidate_id,
        hypothesis_digest=candidate.governing_manifest.canonical_digest, evidence_package_digest=base.digest)
    with sqlite3.connect(Path(qualifier.usage.path).resolve().as_uri()+'?mode=ro', uri=True) as c:
        c.execute('PRAGMA query_only=ON')
        deadline=time.monotonic()+5
        c.set_progress_handler(lambda: time.monotonic()>deadline,1000)
        rows = c.execute(
            "SELECT a.invocation_id FROM model_invocation_allocations a JOIN model_work_envelopes e USING(envelope_id) "
            "JOIN model_invocation_policies p ON p.canonical_digest=a.policy_digest "
            "WHERE a.route=? AND json_extract(e.record_json,'$.candidate_id')=? "
            "AND json_extract(e.record_json,'$.hypothesis_digest')=? "
            "AND json_extract(e.record_json,'$.evidence_package_digest')=? "
            "AND json_extract(p.record_json,'$.prompt_contract_version')=? LIMIT 2",
            (ROUTE, ids['candidate_id'], ids['hypothesis_digest'], base.digest, VERSION)).fetchall()
        if len(rows) != 1:
            raise QualificationHold('QUALIFICATION_KNOWN_RESULT_ABSENT_OR_AMBIGUOUS')
        allocation, terminal = _retained_terminal_allocation(c, rows[0][0])
        if terminal is None or terminal.usage_status is not UsageStatus.REPORTED or terminal.outcome != 'QUALIFICATION_COMPLETE' or terminal.policy_breach:
            raise QualificationHold('QUALIFICATION_REPLAY_USAGE_HOLD')
    receipt_admission = qualifier.objects.committed_admission(ObjectAdmissionRequest('evidence.record',
        'source-qualification-receipt:'+allocation.invocation_id), proof=proof)
    if receipt_admission is None:
        raise QualificationHold('QUALIFICATION_PRIOR_RESULT_UNAVAILABLE')
    receipt_raw = qualifier.objects.rehydrate(HydrationRequest(receipt_admission.admission.admission_id,'evidence.record'),proof=proof).data
    receipt = json.loads(receipt_raw)
    if canonical_json_bytes(receipt) != receipt_raw:
        raise QualificationHold('QUALIFICATION_REPLAY_BINDING_HOLD')
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
    refs=binding.get('prior_judgments',[])
    inventory=binding.get('failure_inventory')
    if (binding.get('qualification_contract')!=VERSION or len(refs)!=1
            or type(inventory)is not list or len(inventory)!=1
            or inventory[0].get('stage')!='SELECTED_QUALIFICATION' or inventory[0].get('reason')!='INPUT_BOUND'):
        raise QualificationHold('QUALIFICATION_ORIGINAL_RECIPE_UNSUPPORTED')
    candidates={segment.span_id:{'source_id':segment.source_id,
        'text':segment.text.encode('utf-8')[:segment.content_end_byte-segment.start_byte].decode('utf-8'),
        'entities':[list(item)for item in segment.entities],'rendering_fragment_count':len(segment.entities)+1,
        'source_range':{'first_span_id':segment.span_id,'last_span_id':segment.span_id}}for segment in view.segments}
    role_state={'candidates':candidates,'current_scope':binding['current_scope'],'prior_scope':binding['prior_scope']}
    if binding['newness']=='SOURCE_DECLARED_FIRST_PUBLICATION':
        role_state['publication_basis']={'mode':binding['newness'],
            'published_at':{row['source_id']:row['first_published_at']for row in binding['first_publication']}}
    ref=JudgmentReference(refs[0]['invocation_id'],ObjectAdmissionId.parse(refs[0]['raw_admission_id']),ObjectAdmissionId.parse(refs[0]['receipt_admission_id']))
    questions=source_role_questions(candidates)
    role=qualifier.judgments.read(ref,state=role_state,questions=questions,source_binding=original,
        caller_identity='NATIVE_ASSESSOR',cycle_id=digest_canonical([JUDGMENT_VERSION,'SOURCE_ROLES',original,role_state,questions]),
        candidate_id=candidate.candidate_id,hypothesis_digest=ids['hypothesis_digest'],proof=proof)
    source_rows=binding['current_scope']['sources']
    state={'source_binding':binding,'source_view':{'passages':list(base.passages),'source_ids':list(base.source_ids),
        'sources':[{'source_id':row['source_id'],'publication_time':row['published_at'],
            'source_updated_time':row['updated_at'],'retrieval_time':row['retrieved_at'],
            'segments':[{**segment.request_record(),'rendering_fragment_count':len(segment.entities)+1}
                for segment in view.segments if segment.source_id==row['source_id']]}for row in source_rows]},
        'issue':{'reason':'JUDGMENT_INPUT_BOUND','failed_questions':inventory,'newness':binding['newness'],'prior_scope':binding['prior_scope']},
        'judgments':[{'questions':questions,'answers':role['answers'],'outcome':role['outcome']}]}
    reference=QualificationReference(allocation.invocation_id,ObjectAdmissionId.parse(receipt['raw_admission_id']),receipt_admission.admission.admission_id)
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
        return JudgedAssessment(NativeAssessmentExecution(checked['materialisation']['materialised_text'],{}),raw,decision.admission.admission_id)
