"""Bounded read-only CURRENT assessment inventory; never a retry grant.

Groups observed reasons and authenticated retained metadata. A cached candidate
still requires the normal consumer's full source/current-rights validation.
Unknown/missing usage is never interpreted as zero spend or safe redispatch.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import time

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.control_plane.native_progress_state import checked_json
from newsroom.control_plane.model_usage import (
    _envelope_from_record, _retained_terminal_allocation, _policy_for_allocation,
    _require_reported_telemetry, ModelUsageService, WorkloadClass, UsageStatus,
)
from newsroom.control_plane.native_assessor import (
    VERSION, same_assessment_producer, _retained_context, _materialisation_record,
    _ASSESSMENT_RESULT_SCHEMA_VERSION, _REFERENCE_PRODUCERS, _source_view_for_binding,
)

MAX_RECORD_BYTES=2*1024*1024


def _label(value,limit):
    text=str(value)
    return text[:limit],len(text)>limit,digest_bytes(text.encode())


def _json(raw):
    if type(raw) is not str or len(raw.encode())>MAX_RECORD_BYTES:raise ValueError('record exceeds bound')
    value=json.loads(raw)
    if type(value) is not dict or canonical_json_bytes(value).decode()!=raw:raise ValueError('record is not canonical')
    return value


def _indexed(c,sql,parameters=()):
    if any('SCAN ' in r[3] for r in c.execute('EXPLAIN QUERY PLAN '+sql,parameters)):
        raise ValueError('selected lookup has no index')
    return c.execute(sql,parameters)


def _retained(c,kind,invocation):
    if kind not in ('NATIVE_ASSESSMENT_RESULT','NATIVE_ASSESSMENT_MATERIALISATION'):raise ValueError('retained kind differs')
    rows=_indexed(c,"SELECT CASE WHEN length(CAST(payload_json AS BLOB))<=2097152 THEN payload_json END,payload_digest FROM ledger WHERE kind='"+kind+"' "
        "AND json_extract(payload_json,'$.invocation_id')=? LIMIT 2",(invocation,)).fetchall()
    if not rows:return None
    if len(rows)!=1:raise ValueError('retained result cardinality differs')
    return checked_json(rows[0][0],rows[0][1],label=kind)


def _assessment(c,envelope,facts,check,source=None):
    availability={key:'UNKNOWN' for key in ('envelope','accounting','retained_raw','materialisation','source_binding')}
    availability['envelope']='VERIFIED'
    ids=_indexed(c,'SELECT invocation_id FROM model_invocation_allocations WHERE envelope_id=? LIMIT 3',
        (envelope.envelope_id,)).fetchall()
    if len(ids)!=1:return availability,None
    check()
    sizes=_indexed(c,'SELECT length(CAST(a.record_json AS BLOB)),length(CAST(t.record_json AS BLOB)) '
        'FROM model_invocation_allocations a LEFT JOIN model_invocation_terminals t USING(invocation_id) '
        'WHERE a.invocation_id=?',(ids[0][0],)).fetchone()
    if sizes is None or any(size is None or size>MAX_RECORD_BYTES for size in sizes):return availability,None
    allocation,terminal=_retained_terminal_allocation(c,ids[0][0])
    if allocation.envelope_id!=envelope.envelope_id or allocation.workload_class is not WorkloadClass.NATIVE_EVIDENCE_ASSESSOR:
        raise ValueError('allocation envelope differs')
    for table,key,identity in (('model_invocation_policies','canonical_digest',allocation.invocation_policy_digest),
        ('model_invocation_context_manifests','context_manifest_digest',allocation.context_manifest_digest)):
        size=_indexed(c,'SELECT length(CAST(record_json AS BLOB)) FROM '+table+' WHERE '+key+'=?',(identity,)).fetchone()
        if size is None or size[0]>MAX_RECORD_BYTES:return availability,None
    policy=_policy_for_allocation(c,allocation)
    breach=ModelUsageService._validate_terminal(terminal,allocation.workload_class,policy,
        requested_max_output_tokens=allocation.max_output_tokens)
    if terminal.usage_status is not UsageStatus.REPORTED or terminal.policy_breach or breach:
        return availability,None
    _require_reported_telemetry(c,terminal)
    availability['accounting']='REPORTED_SETTLED'
    context=_retained_context(c,allocation)
    if context.get('evidence_package_digest')!=envelope.evidence_package_digest:raise ValueError('context envelope differs')
    result=_retained(c,'NATIVE_ASSESSMENT_RESULT',allocation.invocation_id)
    availability['retained_raw']='MISSING'
    if result is None:return availability,None
    if (result.get('schema_version')!=_ASSESSMENT_RESULT_SCHEMA_VERSION
        or result.get('allocation_digest')!=allocation.canonical_digest
        or result.get('invocation_policy_digest')!=allocation.invocation_policy_digest
        or result.get('request_digest')!=allocation.request_digest):raise ValueError('result allocation differs')
    text=result.get('result_text')
    if result.get('retention_outcome')!='RETAINED' or type(text)is not str:return availability,None
    if len(text.encode())!=result.get('result_bytes') or digest_bytes(text.encode())!=result.get('result_digest'):
        raise ValueError('retained raw hash differs')
    if terminal.dispatch_at is None or datetime.fromisoformat(result.get('dispatch_at',''))!=terminal.dispatch_at:
        raise ValueError('result dispatch binding differs')
    availability['retained_raw']='HASH_VERIFIED'
    document=json.loads(text)
    package=document.get('package',{}) if type(document) is dict else {}
    negative=(type(package) is dict and (
        package.get('select_new_information') is False
        or package.get('substantive_claim_indexes')==[]
        or package.get('substantive_new_information')==[])
        and type(package.get('selection_rationale')) is str and bool(package['selection_rationale'].strip()))
    proof={'invocation_id':allocation.invocation_id,'producer':allocation.prompt_contract_version,
        'completed_at':terminal.completed_at.isoformat(),'cached_candidate':False,
        'model_decision_reported':'NO_NEW_INFORMATION'if negative else 'OTHER_OR_UNKNOWN',
        'editorial_acceptance':'UNASSESSED'}
    materialised=_retained(c,'NATIVE_ASSESSMENT_MATERIALISATION',allocation.invocation_id)
    availability['materialisation']='NOT_REQUIRED'
    if allocation.prompt_contract_version in _REFERENCE_PRODUCERS:
        availability['materialisation']='MISSING'
        if materialised is None:return availability,proof
        if materialised!=_materialisation_record(allocation,context,result['result_digest'],materialised.get('receipt')):
            raise ValueError('materialisation binding differs')
        availability['materialisation']='HASH_BOUND'
    # This is only a candidate: exact base/current-source reproduction happens
    # in the existing consumer, not through this observational CLI.
    availability['source_binding']='RETAINED_CONTEXT_ONLY'
    if source is not None and type(context.get('source_reference_binding')) is dict:
        units=source.get('units',[])
        if units and type(source.get('shared_body',units[0].get('body'))) is str:
            # Only the exact CURRENT body can establish this observational
            # binding. Different acquired formatting remains UNKNOWN, not equal.
            body=source.get('shared_body',units[0].get('body'))
            if 'shared_body' in source or all(u.get('body')==body for u in units):
                try:
                    _source_view_for_binding((body,),(units[0]['source_id'],),context['source_reference_binding'])
                    availability['source_binding']='EXACT_CURRENT_BODY_MANIFEST'
                except ValueError:pass
    check()
    candidate=(availability['source_binding']=='EXACT_CURRENT_BODY_MANIFEST'
        and terminal.outcome in ('ASSESSOR_ACCEPTED','ASSESSOR_VALIDATION_FAILED')
        and same_assessment_producer(allocation.prompt_contract_version,VERSION)
        and facts.get('assessment_failure_allocation_digest')==allocation.canonical_digest
        and facts.get('assessment_failure_terminal_digest')==terminal.terminal_digest
        and facts.get('assessment_failure_context_manifest_digest')==allocation.context_manifest_digest)
    proof['cached_candidate']=candidate
    return availability,proof


def inventory(path:Path,*,limit=1000,seconds=10):
    if type(limit)is not int or not 1<=limit<=5000 or not 0<seconds<=60:raise ValueError('inventory bounds differ')
    path=Path(path).resolve(strict=True);deadline=time.monotonic()+seconds
    def check():
        if time.monotonic()>deadline:raise TimeoutError('inventory deadline exceeded')
    groups={};selected=0;truncated=False;deadline_exceeded=False;pending=[]
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=min(seconds,5))) as c:
        c.execute('PRAGMA query_only=ON');c.set_progress_handler(lambda:int(time.monotonic()>deadline),1000);c.execute('BEGIN')
        tables={r[0]for r in c.execute("SELECT name FROM sqlite_schema WHERE type='table'")}
        envelopes=defaultdict(list);envelopes_complete=True;scanned_envelopes=0
        # Thin CURRENT metadata first; source bodies are authenticated only for
        # bounded representatives, not expanded for every revision.
        sql="""SELECT s.revision_id,json_extract(s.content_json,'$.revision_id'),
            json_extract(s.content_json,'$.units[0].source_id'),
            json_extract(s.content_json,'$.units[0].published_at'),
            json_extract(s.content_json,'$.units[0].updated_at'),
            json_extract(s.content_json,'$.units[0].effective_revision.first_observed_at'),
            h.ordinal,CASE WHEN length(CAST(h.state_json AS BLOB))<=2097152 THEN h.state_json END,
            h.state_digest,h.pair_digest FROM native_current_sources s
            LEFT JOIN native_current_heads h USING(revision_id) ORDER BY s.revision_id LIMIT ?"""
        try:
            for revision,source_revision,source_id,published,updated,observed,ordinal,head_raw,head_digest,pair in c.execute(sql,(limit+1,)):
                check()
                if selected==limit:truncated=True;break
                selected+=1;reason='CURRENT_PROOF_INVALID';kind='UNKNOWN';contract='UNKNOWN';stage='UNKNOWN';facts={}
                freshness={'published_at':published,'updated_at':updated,'first_observed_at':observed}
                availability={k:'UNKNOWN'for k in ('current_source','candidate','envelope','accounting','retained_raw','materialisation','source_binding')};proof=None
                try:
                    state=_json(head_raw)
                    if source_revision!=revision or state.get('revision_id')!=revision or state.get('ordinal')!=ordinal or digest_canonical({'state':state,'pair_digest':pair})!=head_digest:
                        raise ValueError('head binding differs')
                    facts=state['facts'];stage=state['stage'];reason=str(facts.get('reason')or stage);kind=str(source_id)
                    contract=str(facts.get('assessment_contract_version')or 'UNKNOWN')
                    availability['candidate']='CURRENT_BOUND'if facts.get('candidate_id')and facts.get('candidate_version_id')else 'MISSING'
                except (ValueError,KeyError,TypeError):pass
                reason,reason_truncated,reason_digest=_label(reason,512)
                kind,kind_truncated,kind_digest=_label(kind,128)
                contract,contract_truncated,contract_digest=_label(contract,128)
                not_assessor=stage in ('ACKNOWLEDGED','SAME_STATE_ASSOCIATED')and not facts.get('candidate_id')
                key=(reason_digest,kind_digest,contract_digest,not_assessor)
                if key not in groups:groups[key]={'reason':reason,'source_kind':kind,'consumer_contract':contract,
                    'disposition':'NOT_ASSESSOR_WORK'if not_assessor else 'UNKNOWN_OR_NO_RETRY','reason_digest':reason_digest,'reason_truncated':reason_truncated,
                    'source_kind_truncated':kind_truncated,'consumer_contract_truncated':contract_truncated,'count':0,'revision_ids':[],'representatives':[],
                    'proof_coverage':{},'model_decisions_reported':{},'editorial_acceptance':'UNASSESSED'}
                group=groups[key];group['count']+=1;group['revision_ids'].append(revision)
                representative=len(group['representatives'])<3
                if representative:
                    rep={'revision_id':revision,'stage':stage,'freshness':freshness,'availability':availability,
                        'invocation_id':None,'model_decision_reported':None,'editorial_acceptance':'UNASSESSED',
                        'disposition':'NOT_ASSESSOR_WORK'if not_assessor else 'UNKNOWN_OR_NO_RETRY',
                        'proof_checked':False}
                    if rep['disposition']=='NOT_ASSESSOR_WORK':
                        for field in ('candidate','envelope','accounting','retained_raw','materialisation','source_binding'):availability[field]='NOT_APPLICABLE'
                    group['representatives'].append(rep);pending.append((group,rep,facts))
        except (TimeoutError,sqlite3.OperationalError):
            if time.monotonic()<=deadline:raise
            deadline_exceeded=True;truncated=True
        try:
            if 'model_work_envelopes' in tables:
                # Single bounded thin-header pass: no prompt/result/history payloads.
                rows=c.execute("SELECT envelope_id,CASE WHEN length(CAST(record_json AS BLOB))<=2097152 THEN record_json END FROM model_work_envelopes WHERE workload_class='NATIVE_EVIDENCE_ASSESSOR' ORDER BY envelope_id LIMIT ?",(limit*4+1,)).fetchall() if 'workload_class' in {r[1]for r in c.execute('PRAGMA table_info(model_work_envelopes)')} else []
                scanned_envelopes=min(len(rows),limit*4)
                envelopes_complete=len(rows)<=limit*4
                for identity,raw in rows[:limit*4]:
                    check()
                    try:
                        envelope=_envelope_from_record(_json(raw))
                        if envelope.envelope_id!=identity:raise ValueError('envelope PK differs')
                        envelopes[envelope.candidate_id].append(envelope)
                    except (ValueError,KeyError,TypeError):envelopes_complete=False
        except (TimeoutError,sqlite3.OperationalError):
            if time.monotonic()<=deadline:raise
            deadline_exceeded=True;envelopes_complete=False
        # Counts and all machine-readable revision IDs are now captured. Only
        # representatives receive proof; their disposition is never extrapolated.
        for group,rep,facts in pending:
            if time.monotonic()>deadline:deadline_exceeded=True;break
            availability=rep['availability'];proof=None
            try:
                check()
                raw,digest=c.execute('SELECT CASE WHEN length(CAST(content_json AS BLOB))<=2097152 THEN content_json END,content_digest FROM native_current_sources WHERE revision_id=?',(rep['revision_id'],)).fetchone()
                source=checked_json(raw,digest,label='source')
                if source.get('revision_id')!=rep['revision_id']:raise ValueError('source identity differs')
                availability['current_source']='HASH_VERIFIED'
                candidates=envelopes.get(facts.get('candidate_id'),[])
                if rep['disposition']!='NOT_ASSESSOR_WORK' and envelopes_complete and len(candidates)==1:
                    found,proof=_assessment(c,candidates[0],facts,check,source);availability.update(found);rep['proof_checked']=True
            except TimeoutError:
                deadline_exceeded=True;break
            except (ValueError,KeyError,TypeError,sqlite3.DatabaseError):
                if time.monotonic()>deadline:deadline_exceeded=True;break
                availability['proof']='INVALID_OR_MISSING';rep['proof_checked']=True
            if proof:
                rep.update(invocation_id=proof['invocation_id'],model_decision_reported=proof['model_decision_reported'],
                    disposition='CACHED_REVALIDATION_CANDIDATE'if proof['cached_candidate']else 'UNKNOWN_OR_NO_RETRY')
        for group in groups.values():
            group['source_representatives_verified']=sum(rep['availability'].get('current_source')=='HASH_VERIFIED'for rep in group['representatives'])
            group['proof_representatives_checked']=sum(rep['proof_checked']for rep in group['representatives'])
            group['unverified_revisions']=group['count']-group['source_representatives_verified']
            for rep in group['representatives']:
                for field,value in rep['availability'].items():
                    bucket=group['proof_coverage'].setdefault(field,{})
                    bucket[value]=bucket.get(value,0)+1
                model=rep['model_decision_reported']or 'UNKNOWN'
                group['model_decisions_reported'][model]=group['model_decisions_reported'].get(model,0)+1
    return {'schema':'newsroom.native-assessment-inventory.v1','selected':selected,'truncated':truncated,'deadline_exceeded':deadline_exceeded,
        'envelope_inventory_complete':envelopes_complete,'envelopes_scanned':scanned_envelopes,
        'envelopes_truncated':not envelopes_complete,'groups_complete':not truncated,'deadline_seconds':seconds,
        'scope':'CURRENT source/head hashes plus selected retained accounting metadata; no rights/CAS/editorial acceptance or retry authority','retry_authorised':False,'groups':[groups[k]for k in sorted(groups)]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal',type=Path,required=True);parser.add_argument('--limit',type=int,default=1000)
    parser.add_argument('--seconds',type=float,default=10)
    args=parser.parse_args()
    try:result=inventory(args.journal,limit=args.limit,seconds=args.seconds)
    except (ValueError,sqlite3.DatabaseError,TimeoutError,OSError) as exc:
        result={'schema':'newsroom.native-assessment-inventory.v1','availability':'UNKNOWN','error_class':type(exc).__name__,'retry_authorised':False}
    print(json.dumps(result,sort_keys=True,separators=(',',':')))

if __name__=='__main__':main()
