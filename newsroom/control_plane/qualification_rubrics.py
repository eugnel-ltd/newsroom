"""EVID-011/012/013 meaning and lossless Source-bound witness candidates."""
from __future__ import annotations
import re

RUBRICS = {
    'LAW_RIGHT_STATUS_POLICY': {
        'definition': 'A verified new or changed law, right, official status, deadline or public policy.',
        'requires': 'Identify the legal/policy/status change and its exact affirmed source clause. A confirmed announcement is not proof the rule is already in force.',
        'excludes': 'Standing guidance, first observation, publication/update metadata alone, or a merely hypothetical change.'},
    'SAFETY_OR_PUBLIC_HEALTH': {
        'definition': 'A credible material effect on safety or public health.',
        'requires': 'An affirmed injury risk, public-health warning, evacuation or material exposure, with the affected group and the evidence linking the event to that effect.',
        'excludes': 'Generic dangerousness, conditional possible harm, or a regulatory announcement with no established material safety effect.'},
    'ESSENTIAL_SERVICE_DISRUPTION': {
        'definition': 'Material disruption of an essential service, route, school, workplace or locality.',
        'requires': 'The affected service/group and source-supported material duration or daily-life consequence. Exact quantities remain code-validated.',
        'excludes': 'Routine transport/weather/utility status, record-update noise, or an unconfirmed future interruption.'},
    'HOUSEHOLD_PRACTICAL_EFFECT': {
        'definition': 'A practical effect on household money, work, housing, education, healthcare or UK–Hong Kong travel.',
        'requires': 'An affirmed practical change for affected readers and its exact source-supported consequence.',
        'excludes': 'General background, speculative consequences or a topic match without a material practical effect.'},
    'OFFICIAL_ACTION_OR_DEADLINE': {
        'definition': 'An official instruction, process or deadline that affected readers may need to act on.',
        'requires': 'An affirmed official action/process/deadline and the specific reader action. Opening a consultation is a confirmed process; its proposed rule is not thereby enacted.',
        'excludes': 'Publication activity alone, internal administration, an unannounced intention or a missing reader action.'},
    'EXCEPTIONAL_PUBLIC_IMPORTANCE': {
        'definition': 'Exceptional public importance in Hong Kong or internationally.',
        'requires': 'Source evidence of a Hong Kong-wide material event, international emergency or constitutional change, and the affected public scope.',
        'excludes': 'Generic interest, ordinary international coverage, dramatic wording or an unsupported importance claim.'},
}
TEMPORAL_RULES = ('Require newly confirmed substantive information, not first-observation novelty. '
    'The six tests are independent alternatives. Preserve proposals, negation, quotation, attribution and effective dates. '
    'A confirmed announcement/process and an already-effective rule are different facts. Missing prior evidence cannot prove a comparative change. '
    'Use the full parent assertion and source context; a clause fragment cannot remove its condition or contradiction.')
_CLAUSE = re.compile(r'[^\n,，.;；。!?！？]+')


def witness_inventory(view):
    """Keep all parent text; exact clauses are selectable, never generated prose.

    The existing codec limits resolved Source lookup keys to 256 UTF-8 bytes.
    Oversized clauses stay visible and trigger the caller's bounded escalation.
    """
    result = {}
    for segment in view.segments:
        text = segment.text.encode('utf-8')[:segment.content_end_byte-segment.start_byte].decode('utf-8')
        ranges = [(segment.span_id,0,len(text))]
        if not re.match(r'Row [1-9][0-9]*: ',text):
            ranges.extend((f'{segment.span_id}C{n}',m.start(),m.end())for n,m in enumerate(_CLAUSE.finditer(text),1))
        candidates,seen,oversized,uncovered = {},set(),[],[]
        for identity,start,end in ranges:
            raw = text[start:end]
            first = start+len(raw)-len(raw.lstrip());last=end-(len(raw)-len(raw.rstrip()))
            value=text[first:last]
            if not value or value in seen:continue
            seen.add(value)
            item={'text':value,'parent_span_id':segment.span_id,
                'start_byte':segment.start_byte+len(text[:first].encode('utf-8')),
                'end_byte':segment.start_byte+len(text[:last].encode('utf-8'))}
            if len(value.encode('utf-8'))>256:
                oversized.append(identity)
                if identity!=segment.span_id or len(ranges)==1:uncovered.append(identity)
            else:candidates[identity]=item
        result[segment.span_id]={'parent_text':text,'candidates':candidates,'overbound_ids':oversized,'uncovered_clause_ids':uncovered}
    return result


def question(span_id,test,fields):
    rubric=RUBRICS[test]
    return {'type':'choice','instructions':{'task':f'Does assertion {span_id} satisfy {test}?',
        **rubric,'temporal_and_source_rules':TEMPORAL_RULES,'required_witness_fields':list(fields)},
        'criteria':{'YES':rubric['requires'],'NO':rubric['excludes'],
                    'UNCERTAIN':'Evidence is incomplete or contradicts the proposed criterion; reasoning is required.'}}
