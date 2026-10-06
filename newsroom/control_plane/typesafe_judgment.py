"""One accounted Typesafe batch; model judgments never grant source authority."""
from __future__ import annotations

import json
from copy import deepcopy
import re
import sqlite3
import ssl
import urllib.request
import urllib.error
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Callable, Mapping

from newsroom.authority import AuthenticationProof, GovernedObjects, ObjectAdmissionRequest
from newsroom.authority.objects import HydrationRequest, ObjectAdmissionId
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical, validate_sha256_digest
from .govuk_evidence import _NoRedirect, _unique_object
from .model_usage import (InvocationAllocation, InvocationEfficiencyPolicy, InvocationTerminal,
    ModelUsageService, UsageComponents, UsageStatus, WorkEnvelope, WorkloadClass,
    _allocation_from_record, _retained_terminal_allocation, _policy_for_allocation,
    _require_reported_telemetry, ModelUsageIntegrityError)

VERSION = 'newsroom.typesafe-judgment.v1'
CONSUMER_VERSION = 'newsroom.typesafe-judgment.answers-consumer.v2'
ROUTE = 'TYPESAFE_JUDGMENT'
MODEL = 'jev-latest'
URL = 'https://api.typesafe.ai/v1/systemone'
TIMEOUT = 10
MAX_RESPONSE_BYTES = 1_048_576
PPM = 1_000_000
FLAGS = ('POST=/v1/systemone', 'RETRIES=0')
SCHEMA_DIGEST = digest_canonical({'version': VERSION, 'answers': ['choice', 'noul', 'score'], 'projection': 'integer-ppm-round-half-even'})


class TypesafeJudgmentError(RuntimeError):
    def __init__(self, code, *, reference=None):
        super().__init__(code)
        self.reference = reference


@dataclass(frozen=True, slots=True)
class JudgmentReference:
    invocation_id: str
    raw_admission_id: ObjectAdmissionId
    receipt_admission_id: ObjectAdmissionId


def implementation_digest():
    return digest_bytes(Path(__file__).read_bytes())


def judgment_policy(*, evidence_digest: str, qualified: bool) -> InvocationEfficiencyPolicy:
    return InvocationEfficiencyPolicy.create(
        policy_id=VERSION, version=VERSION, workload_class=WorkloadClass.TYPESAFE_JUDGMENT,
        provider='typesafe', route=ROUTE, model=MODEL, reasoning='none', one_turn=True,
        exact_input=True, skills_enabled=False, tools_enabled=False, mcp_enabled=False,
        prior_message_count=0, command_semantic_version=VERSION, command_flags=FLAGS,
        context_manifest_schema_version=VERSION, disabled_capabilities=('tools', 'prior-messages'),
        implementation_revision=implementation_digest(), max_prompt_bytes=131072,
        max_context_tokens=32768, max_output_tokens=16384, max_total_tokens=65536,
        prompt_contract_version=VERSION, output_schema_digest=SCHEMA_DIGEST,
        allowed_context_identities=(VERSION,), allowed_config_identities=(VERSION,),
        hard_estimate_ceiling_tokens=None, evidence_digest=evidence_digest, qualified=qualified)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _reject_constant(_value):
    raise ValueError('nonfinite JSON number')


def _decode(raw):
    return json.loads(raw.decode('utf-8'), parse_float=Decimal,
                      parse_constant=_reject_constant, object_pairs_hook=_unique_object)


def _ppm(value, *, maximum=1):
    if type(value) not in (int, Decimal) or not Decimal(value).is_finite() or not 0 <= value <= maximum:
        raise ValueError('judgment number outside its finite range')
    return int((Decimal(value) * PPM).to_integral_value(rounding=ROUND_HALF_EVEN))


def _questions(questions):
    if type(questions) is not dict or not questions:
        raise ValueError('question map is absent')
    for key, q in questions.items():
        if type(key) is not str or not key or type(q) is not dict or q.get('type') not in {'choice', 'noul', 'score'}:
            raise ValueError('question identity/type differs')
        if type(q.get('instructions')) not in (str, dict, list):
            raise ValueError('question instructions are absent')
        kind, criteria = q['type'], q.get('criteria')
        if set(q) - {'type', 'instructions', 'criteria'}:
            raise ValueError('question fields differ')
        if kind == 'choice' and (type(criteria) is not dict or not 1 <= len(criteria) <= 255 or any(type(k) is not str or not k for k in criteria)):
            raise ValueError('choice criteria differ')
        if kind == 'score' and (type(criteria) is not list or not 2 <= len(criteria) <= 10):
            raise ValueError('score criteria differ')
        if kind == 'noul' and criteria is not None and (type(criteria) is not dict or set(criteria) != {'true', 'false'}):
            raise ValueError('noul criteria differ')
    _json(questions)


def _answers(value, questions):
    if type(value) is not dict or set(value) != set(questions):
        raise ValueError('answer IDs differ')
    result = {}
    for key, q in questions.items():
        answer, kind = value[key], q['type']
        if type(answer) is not dict or answer.get('type') != kind:
            raise ValueError('answer type differs')
        if kind == 'noul':
            if set(answer) != {'type', 'noul'}:
                raise ValueError('noul fields differ')
            result[key] = {'type': kind, 'noul_ppm': _ppm(answer['noul'])}
            continue
        expected = {'type', 'probabilities', 'confidence', 'choice'} if kind == 'choice' else {'type', 'probabilities', 'confidence', 'score', 'legend'}
        probabilities = answer.get('probabilities')
        keys = set(q['criteria']) if kind == 'choice' else {str(i) for i in range(len(q['criteria']))}
        if set(answer) != expected or type(probabilities) is not dict or set(probabilities) != keys:
            raise ValueError('answer fields/distribution differ')
        projected = {k: _ppm(v) for k, v in probabilities.items()}
        tolerance = Decimal('0.01') if kind == 'choice' else Decimal('0.000001')
        if abs(sum(Decimal(v) for v in probabilities.values()) - 1) > tolerance:
            raise ValueError('probabilities do not sum to one')
        result[key] = {'type': kind, 'confidence_ppm': _ppm(answer['confidence']), 'probabilities_ppm': projected}
        if kind == 'choice':
            if answer['choice'] not in keys or probabilities[answer['choice']] != max(probabilities.values()):
                raise ValueError('choice differs from distribution')
            result[key]['choice'] = answer['choice']
        else:
            legend = {str(i): v for i, v in enumerate(q['criteria'])}
            if answer['legend'] != legend:
                raise ValueError('score legend differs')
            score = _ppm(answer['score'], maximum=len(keys) - 1)
            expectation = sum(Decimal(int(k)) * Decimal(v) for k, v in probabilities.items())
            if abs(Decimal(answer['score']) - expectation) > Decimal('0.000001'):
                raise ValueError('score differs from distribution')
            result[key].update(score_ppm=score, legend=legend)
    return result


def _http(request, *, timeout, max_bytes):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    with opener.open(request, timeout=timeout) as response:
        return response.status, response.geturl(), response.read(max_bytes + 1)


class TypesafeJudgment:
    def __init__(self, *, usage: ModelUsageService, objects: GovernedObjects,
                 policy: InvocationEfficiencyPolicy, api_key: Callable[[], str],
                 source_fence: Callable[[dict, AuthenticationProof], AbstractContextManager],
                 transport=_http, clock=lambda: datetime.now(UTC), implementation_worktree_clean=False):
        if (policy.workload_class is not WorkloadClass.TYPESAFE_JUDGMENT or
            (policy.provider, policy.route, policy.model, policy.reasoning) != ('typesafe', ROUTE, MODEL, 'none') or
            policy.prompt_contract_version != VERSION or policy.output_schema_digest != SCHEMA_DIGEST or not policy.qualified):
            raise TypesafeJudgmentError('TYPESAFE_POLICY_HOLD')
        if implementation_worktree_clean is not True or policy.implementation_revision != implementation_digest():
            raise TypesafeJudgmentError('TYPESAFE_IMPLEMENTATION_HOLD')
        if not all(callable(f) for f in (api_key, source_fence, transport, clock)):
            raise TypeError('judgment dependencies must be callable')
        usage.register_policy(policy)
        self.usage, self.objects, self.policy = usage, objects, policy
        self.key, self.fence, self.transport, self.clock = api_key, source_fence, transport, clock
        self.implementation_worktree_clean = implementation_worktree_clean

    def _input(self, *, state, questions, source_binding, cycle_id, caller_identity, candidate_id=None,
               hypothesis_digest=None, ingest_id=None, graphiti_attempt_id=None,
               proof=None, parent_invocation_id=None):
        if parent_invocation_id is not None:
            raise TypesafeJudgmentError('CROSS_ENVELOPE_PARENT_REQUIRES_RECEIPT_LINKAGE')
        _questions(questions)
        if type(source_binding) is not dict or not source_binding:
            raise ValueError('source binding is absent')
        if caller_identity == 'NATIVE_ASSESSOR':
            if not candidate_id or not hypothesis_digest or graphiti_attempt_id is not None:
                raise ValueError('native judgment caller binding differs')
        elif caller_identity == 'GRAPHITI_VERIFIER':
            prefix, separator, number = str(graphiti_attempt_id or '').rpartition(':')
            if not ingest_id or candidate_id is not None or prefix != ingest_id or not separator or not number.isdigit() or int(number) <= 0:
                raise ValueError('graphiti judgment caller binding differs')
        else:
            raise ValueError('judgment caller identity differs')
        if type(source_binding.get('content_digest')) is not str:
            raise ValueError('judgment caller binding is incomplete')
        validate_sha256_digest(source_binding['content_digest'])
        request = _json({'model': MODEL, 'state': state, 'questions': questions})
        snapshot = {'caller_identity': caller_identity, 'source_binding': source_binding, 'candidate_id': candidate_id,
                    'hypothesis_digest': hypothesis_digest, 'ingest_id': ingest_id,
                    'graphiti_attempt_id': graphiti_attempt_id, 'state_digest': digest_bytes(_json(state)),
                    'question_digest': digest_bytes(_json(questions))}
        snapshot_digest = digest_canonical(snapshot)
        envelope = WorkEnvelope.create(cycle_id=cycle_id, workload_class=self.policy.workload_class,
            admitted_at=self.clock(), admission_decision_id=None, candidate_id=candidate_id,
            hypothesis_digest=hypothesis_digest, evidence_package_digest=snapshot_digest,
            ingest_id=ingest_id, graphiti_attempt_id=graphiti_attempt_id)
        value = dict(schema_version=VERSION, provider='typesafe', route=ROUTE, model=MODEL,
            reasoning='none', command_semantic_version=VERSION, command_flags=list(FLAGS),
            disabled_capabilities=list(self.policy.disabled_capabilities), implementation_revision=self.policy.implementation_revision,
            implementation_worktree_clean=self.implementation_worktree_clean, prompt_contract_version=VERSION,
            prompt_bytes=len(request), prompt_digest=digest_bytes(request), schema_digest=SCHEMA_DIGEST,
            output_schema_digest=SCHEMA_DIGEST, system_digest=digest_bytes(_json(questions)),
            evidence_package_digest=snapshot_digest, evidence_package_bytes=len(canonical_json_bytes(snapshot)),
            context_identity=VERSION, config_identity=VERSION, one_turn=True, exact_input=True,
            skills_enabled=False, tools_enabled=False, mcp_enabled=False, prior_message_count=0,
            skill_count=0, tool_count=0, mcp_server_count=0, mcp_tool_count=0)
        value['caller_identity'] = caller_identity
        value['caller_ingest_id'] = ingest_id
        value['caller_graphiti_attempt_id'] = graphiti_attempt_id
        value['request_digest'] = digest_canonical({k: value[k] for k in ('provider','route','model','reasoning',
            'command_semantic_version','command_flags','implementation_revision','system_digest','prompt_digest','output_schema_digest')})
        value['context_manifest_digest'] = digest_canonical(value)
        return request, snapshot, envelope, value

    def evaluate(self, **inputs):
        inputs = {**inputs, **{key: deepcopy(inputs[key]) for key in ('state', 'questions', 'source_binding')}}
        request, snapshot, envelope, manifest = self._input(**inputs)
        proof = inputs['proof']
        with self.fence(inputs['source_binding'], proof):
            connection = sqlite3.connect(Path(self.usage.path).resolve().as_uri()+'?mode=ro', uri=True)
            try:
                rows = connection.execute('SELECT record_json FROM model_invocation_allocations WHERE envelope_id=?',
                    (envelope.envelope_id,)).fetchall()
            finally:
                connection.close()
            if rows:
                if len(rows) != 1:
                    raise TypesafeJudgmentError('TYPESAFE_ALLOCATION_AMBIGUOUS')
                allocation = _allocation_from_record(json.loads(rows[0][0]))
                prior = self.objects.committed_admission(ObjectAdmissionRequest('evidence.record', 'typesafe-receipt:'+allocation.invocation_id), proof=proof)
                if prior is None:
                    raise TypesafeJudgmentError('TYPESAFE_EXISTING_INTENT_UNSETTLED')
                receipt = json.loads(self.objects.rehydrate(HydrationRequest(prior.admission.admission_id, 'evidence.record'), proof=proof).data)
                reference = JudgmentReference(allocation.invocation_id, ObjectAdmissionId.parse(receipt['raw_admission_id']), prior.admission.admission_id)
                self._read_bound(reference, inputs, snapshot, envelope, manifest)
                return reference
        if len(request) > self.policy.max_prompt_bytes:
            raise TypesafeJudgmentError('TYPESAFE_INPUT_BOUND')
        envelope = self.usage.resume_or_open_typesafe_envelope(envelope)
        self.usage.retain_context_manifest(manifest)
        allocation = InvocationAllocation.create(envelope_id=envelope.envelope_id, cycle_id=inputs['cycle_id'], leaf_ordinal=1,
            workload_class=self.policy.workload_class, invocation_policy_digest=self.policy.canonical_digest,
            provider='typesafe', route=ROUTE, model=MODEL, reasoning='none', prompt_contract_version=VERSION,
            prompt_bytes=len(request), prompt_digest=digest_bytes(request), request_digest=manifest['request_digest'],
            output_schema_digest=SCHEMA_DIGEST, max_output_tokens=self.policy.max_output_tokens,
            context_manifest_digest=manifest['context_manifest_digest'], context_identity=VERSION, config_identity=VERSION,
            one_turn=True, exact_input=True, skills_enabled=False, tools_enabled=False, mcp_enabled=False,
            prior_message_count=0, allocated_at=self.clock(), recovery_deadline_at=self.clock()+timedelta(seconds=TIMEOUT+5), parent_invocation_id=None)
        self.usage.allocate(allocation, owner_emergency_stop=False)
        dispatch, usage, raw, value, answers, error = None, None, None, None, None, None
        transport_failure = None
        try:
            with self.fence(inputs['source_binding'], proof):
                key = self.key()
                if type(key) is not str or not key:
                    raise ValueError('credential absent')
                http = urllib.request.Request(URL, data=request, method='POST', headers={'Authorization':'Bearer '+key, 'Content-Type':'application/json'})
                dispatch = self.clock()
                self.usage.observe_transport(invocation_id=allocation.invocation_id, observed_at=dispatch, state='DISPATCH_STARTED', evidence_digest=manifest['request_digest'])
                status, url, raw = self.transport(http, timeout=TIMEOUT, max_bytes=MAX_RESPONSE_BYTES)
            if status != 200 or url != URL or type(raw) is not bytes or len(raw)>MAX_RESPONSE_BYTES:
                raise ValueError('HTTP response envelope differs')
            value = _decode(raw)
            u = value.get('usage') if type(value) is dict else None
            if type(u) is dict and set(u)=={'input_tokens','output_tokens'} and all(type(u[k]) is int and u[k]>=0 for k in u):
                usage = u
            if type(value) is not dict or set(value)!={'model','answers','usage'} or type(value['model']) is not str or not re.fullmatch(r'jev-\d+\.\d+(?:\.\d+)?', value['model']):
                raise ValueError('response fields/model differ')
            answers = _answers(value['answers'], inputs['questions'])
            if usage is None:
                raise ValueError('reported usage is absent')
        except Exception as exc:
            error = type(exc).__name__
            if isinstance(exc, urllib.error.HTTPError):
                # Preserve the actual response for diagnosis, not a zero-cost or
                # provider-completion inference. No Authorization headers/logs.
                error = (f'HTTPError:{exc.code}' if type(exc.code) is int and 100 <= exc.code <= 599
                         and exc.geturl() == URL else 'HTTPError:INVALID_TRANSPORT')
                transport_failure = {'status': exc.code, 'endpoint_matches': exc.geturl() == URL}
                if exc.headers is not None:
                    request_id = exc.headers.get('x-typesafe-request-id')
                    if type(request_id) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', request_id):
                        transport_failure['provider_request_id'] = request_id
                    retry_after = exc.headers.get('Retry-After')
                    if type(exc.code) is int and 500 <= exc.code <= 599 and exc.geturl() == URL and retry_after is not None:
                        error = f'HTTPError:{exc.code}:RETRY_DELAY_HOLD'
                    if type(retry_after) is str and re.fullmatch(r'\d{1,7}', retry_after):
                        transport_failure['retry_after_seconds'] = int(retry_after)

                try:
                    raw = exc.read(MAX_RESPONSE_BYTES + 1)
                except Exception as read_error:
                    transport_failure['response_read_failure'] = type(read_error).__name__
        known, zero = usage is not None, dispatch is None
        telemetry = None if not known else {'provider':'typesafe','model_alias':MODEL, 'model_returned':value.get('model'), **usage}
        components = UsageComponents(input_tokens=usage['input_tokens'], output_tokens=usage['output_tokens'],
            total_tokens=sum(usage.values()), provenance='PROVIDER_REPORTED') if known else UsageComponents(total_tokens=0, provenance='CLI_DERIVED') if zero else UsageComponents(provenance='UNAVAILABLE')
        terminal = self.usage.complete(InvocationTerminal.create(invocation_id=allocation.invocation_id,
            outcome='TYPESAFE_COMPLETE' if error is None else 'TYPESAFE_FAILED', failure_class=error,
            usage_status=UsageStatus.REPORTED if known or zero else UsageStatus.UNREPORTED, components=components,
            dispatch_at=dispatch, completed_at=self.clock(), observed_at=self.clock(), pre_dispatch_zero_proved=zero,
            provider_telemetry_digest=None if telemetry is None else digest_canonical(telemetry),
            raw_telemetry_pointer=None if telemetry is None else digest_bytes(raw),
            od_011_reference='OD-011:TYPESAFE_JUDGMENT', subscription_cli_chat_not_cash_debited=False), provider_telemetry=telemetry)
        if transport_failure is not None:
            failure_record = {'schema_version': VERSION, 'invocation_id': allocation.invocation_id,
                'allocation_digest': allocation.canonical_digest, 'terminal_digest': terminal.terminal_digest,
                'request_digest': manifest['request_digest'], 'snapshot': snapshot,
                'transport_failure': transport_failure, 'outcome': terminal.outcome}
            self.objects.admit(ObjectAdmissionRequest('evidence.record', 'typesafe-transport-failure:'+allocation.invocation_id),
                canonical_json_bytes(failure_record), proof=proof)
        if raw is None or type(raw) is not bytes or len(raw)>MAX_RESPONSE_BYTES:
            raise TypesafeJudgmentError('TYPESAFE_UNSETTLED_OR_INVALID_TRANSPORT')
        raw_admission = self.objects.admit(ObjectAdmissionRequest('evidence.record','typesafe-raw:'+allocation.invocation_id), raw, proof=proof).admission
        receipt = {'schema_version':VERSION,'invocation_id':allocation.invocation_id,'allocation_digest':allocation.canonical_digest,
            'request_digest':manifest['request_digest'],'snapshot':snapshot,'model_alias':MODEL,
            'model_returned':None if type(value) is not dict else value.get('model'),'raw_admission_id':str(raw_admission.admission_id),
            'raw_response_digest':digest_bytes(raw),'terminal_digest':terminal.terminal_digest,'outcome':terminal.outcome,
            'answers':answers,'usage':usage,'tariff':{'input_microUSD_per_1000_tokens':42,'output_microUSD_per_1000_tokens':0,
            'basis':'CALCULATED_FROM_QUALIFIED_TARIFF','calculated_usd_microunits':None if not known else (usage['input_tokens']*42+999)//1000}}
        if transport_failure is not None:
            receipt['transport_failure'] = transport_failure
        admitted = self.objects.admit(ObjectAdmissionRequest('evidence.record','typesafe-receipt:'+allocation.invocation_id), canonical_json_bytes(receipt), proof=proof).admission
        reference = JudgmentReference(allocation.invocation_id, raw_admission.admission_id, admitted.admission_id)
        if error is not None or terminal.policy_breach:
            raise TypesafeJudgmentError('TYPESAFE_RESULT_HOLD', reference=reference)
        return reference

    def read(self, reference, **inputs):
        inputs = {**inputs, **{key: deepcopy(inputs[key]) for key in ('state', 'questions', 'source_binding')}}
        _request, snapshot, envelope, manifest = self._input(**inputs)
        with self.fence(inputs['source_binding'], inputs['proof']):
            return self._read_bound(reference, inputs, snapshot, envelope, manifest)

    def _read_bound(self, reference, inputs, snapshot, envelope, manifest):
        connection = sqlite3.connect(Path(self.usage.path).resolve().as_uri()+'?mode=ro', uri=True)
        try:
            connection.execute('PRAGMA query_only=ON')
            connection.execute('BEGIN')
            allocation, terminal = _retained_terminal_allocation(connection, reference.invocation_id)
            revalidate = terminal.outcome == 'TYPESAFE_FAILED' and terminal.failure_class == 'ValueError'
            if (terminal.usage_status is not UsageStatus.REPORTED
                    or (terminal.outcome != 'TYPESAFE_COMPLETE' and not revalidate) or terminal.policy_breach):
                raise TypesafeJudgmentError('TYPESAFE_REPLAY_USAGE_HOLD', reference=reference)
            original_policy = _policy_for_allocation(connection, allocation)
            _require_reported_telemetry(connection, terminal)
            old, current = asdict(original_policy), asdict(self.policy)
            for name in ('canonical_digest', 'implementation_revision', 'evidence_digest'):
                old.pop(name, None)
                current.pop(name, None)
            if old != current or not original_policy.qualified:
                raise TypesafeJudgmentError('TYPESAFE_REPLAY_POLICY_HOLD')
            row = connection.execute('SELECT provider,route,evidence_package_digest,record_json FROM model_invocation_context_manifests WHERE context_manifest_digest=?',
                (allocation.context_manifest_digest,)).fetchone()
            original_manifest = dict(manifest)
            original_manifest.pop('context_manifest_digest')
            original_manifest['implementation_revision'] = original_policy.implementation_revision
            original_manifest['request_digest'] = digest_canonical({k: original_manifest[k] for k in (
                'provider','route','model','reasoning','command_semantic_version','command_flags',
                'implementation_revision','system_digest','prompt_digest','output_schema_digest')})
            original_manifest['context_manifest_digest'] = digest_canonical(original_manifest)
            if (row is None or tuple(row[:3]) != (original_manifest['provider'], original_manifest['route'], original_manifest['evidence_package_digest'])
                    or canonical_json_bytes(original_manifest).decode() != row[3]
                    or allocation.invocation_id != reference.invocation_id
                    or allocation.envelope_id != envelope.envelope_id
                    or allocation.request_digest != original_manifest['request_digest']
                    or allocation.context_manifest_digest != original_manifest['context_manifest_digest']):
                raise TypesafeJudgmentError('TYPESAFE_REPLAY_ALLOCATION_HOLD')
        finally:
            connection.close()
        receipt_raw = self.objects.rehydrate(HydrationRequest(reference.receipt_admission_id,'evidence.record'), proof=inputs['proof']).data
        receipt = json.loads(receipt_raw)
        raw = self.objects.rehydrate(HydrationRequest(reference.raw_admission_id,'evidence.record'), proof=inputs['proof']).data
        if (canonical_json_bytes(receipt)!=receipt_raw or receipt.get('schema_version')!=VERSION or
            receipt.get('invocation_id')!=reference.invocation_id or receipt.get('snapshot')!=snapshot or
            receipt.get('request_digest')!=original_manifest['request_digest'] or receipt.get('raw_admission_id')!=str(reference.raw_admission_id) or
            receipt.get('allocation_digest')!=allocation.canonical_digest or
            receipt.get('raw_response_digest')!=digest_bytes(raw) or terminal is None or
            receipt.get('terminal_digest')!=terminal.terminal_digest or receipt.get('outcome')!=terminal.outcome
            or terminal.usage_status is not UsageStatus.REPORTED or terminal.policy_breach):
            raise TypesafeJudgmentError('TYPESAFE_REPLAY_BINDING_HOLD')
        try:
            decoded = _decode(raw)
            usage = decoded.get('usage') if type(decoded) is dict else None
            if (type(decoded) is not dict or set(decoded)!={'model','answers','usage'}
                    or type(decoded['model']) is not str or not re.fullmatch(r'jev-\d+\.\d+(?:\.\d+)?',decoded['model'])
                    or type(usage) is not dict or set(usage)!={'input_tokens','output_tokens'}
                    or any(type(value)is not int or value<0 for value in usage.values())):
                raise ValueError('retained response envelope differs')
            answers = _answers(decoded['answers'], inputs['questions'])
        except (ValueError,TypeError,KeyError,UnicodeError) as exc:
            raise TypesafeJudgmentError('TYPESAFE_REPLAY_USAGE_HOLD' if revalidate else 'TYPESAFE_REPLAY_VALUE_HOLD',reference=reference) from exc
        expected_answers = None if revalidate else answers
        if (receipt.get('answers')!=expected_answers or receipt.get('usage')!=usage
                or receipt.get('model_returned')!=decoded['model']):
            raise TypesafeJudgmentError('TYPESAFE_REPLAY_VALUE_HOLD',reference=reference)
        if (receipt.get('model_alias') != MODEL or receipt.get('tariff') != {
                'input_microUSD_per_1000_tokens':42, 'output_microUSD_per_1000_tokens':0,
                'basis':'CALCULATED_FROM_QUALIFIED_TARIFF',
                'calculated_usd_microunits':(decoded['usage']['input_tokens']*42+999)//1000}
                or terminal.components.input_tokens != decoded['usage']['input_tokens']
                or terminal.components.output_tokens != decoded['usage']['output_tokens']
                or terminal.components.total_tokens != sum(decoded['usage'].values())):
            raise TypesafeJudgmentError('TYPESAFE_REPLAY_USAGE_HOLD')
        if revalidate:
            return {**receipt,'answers':answers,'consumer_revalidation':{
                'consumer_contract':CONSUMER_VERSION,'original_outcome':terminal.outcome,
                'original_terminal_digest':terminal.terminal_digest,'raw_response_digest':digest_bytes(raw)}}
        return receipt
