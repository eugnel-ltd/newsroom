"""One accounted rendering of already selected source claims; never select facts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import re
from pathlib import Path
import sqlite3

from jsonschema import ValidationError, validate

from newsroom.authority import HydrationRequest, ObjectAdmissionId, ObjectAdmissionRequest
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical, validate_sha256_digest
from .cycle import _complete_writer_usage
from .govuk_evidence import _unique_object
from .model_usage import (
    InvocationAllocation, InvocationEfficiencyPolicy, ModelUsageIntegrityError, ModelUsageService,
    UsageStatus, WorkEnvelope, WorkloadClass, _policy_for_allocation,
    _require_reported_telemetry, _retained_terminal_allocation, _has_exact_dispatch, _envelope_from_record, _utc_text,
)
from .native_embeddings import _retained_allocation
from .writer import _run_grok_json, CONT_DISABLED_CAPABILITIES, _grok_command_flags

LEGACY_VERSION = 'newsroom.native-claim-localisation.v1'
VERSION = 'newsroom.native-claim-localisation.v2'
CONSUMER_VERSION = 'newsroom.native-claim-localisation-consumer.v1'
ROUTE = 'NATIVE_CLAIM_LOCALISATION'
MODEL = 'grok-4.7'
COMMAND_FLAGS = _grok_command_flags('high', model=MODEL)
LEGACY_SYSTEM = ('Render only the supplied factual source assertions in natural Hong Kong Traditional Chinese. '
          'Do not choose news, add facts, change negation, proposal/effective status, dates, quantities or attribution. '
          'Return one entry for every supplied span ID and no others. Return exactly entities+1 fragments: '
          'the application inserts the original ordered entity names between them. '
          'Record exact source_lookup_key/rendered_expression pairs for translated factual expressions, '
          'and exact source quotation keys. No explanation or text outside the JSON object.')
ITEM = {'type': 'object', 'additionalProperties': False, 'required': [
    'rendered_assertion_zh_hant_hk_fragments', 'factual_localisations', 'quotation_source_keys'], 'properties': {
    'rendered_assertion_zh_hant_hk_fragments': {'type': 'array', 'minItems': 1, 'maxItems': 65,
        'items': {'type': 'string', 'maxLength': 4096}},
    'factual_localisations': {'type': 'array', 'maxItems': 32, 'items': {'type': 'object',
        'additionalProperties': False, 'required': ['source_lookup_key', 'rendered_expression'],
        'properties': {'source_lookup_key': {'type': 'string', 'maxLength': 256},
            'rendered_expression': {'type': 'string', 'maxLength': 256}}}},
    'quotation_source_keys': {'type': 'array', 'maxItems': 32, 'items': {'type': 'string', 'maxLength': 256}},
}}
LEGACY_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['renderings'],
    'properties': {'renderings': {'type': 'object', 'additionalProperties': ITEM}}}
LEGACY_SCHEMA_DIGEST = digest_canonical(LEGACY_SCHEMA)
SYSTEM = (LEGACY_SYSTEM + ' Each rendering is an array entry with the exact span_id. '
          'Source lookup and quotation keys must be exact verbatim source fragments of at most 256 UTF-8 bytes; '
          'use separate sentence-sized verbatim keys for a long quotation. Do not paraphrase source keys.')
ENTRY = {**ITEM, 'required': ['span_id', *ITEM['required']],
    'properties': {'span_id': {'type': 'string', 'maxLength': 256}, **ITEM['properties']}}
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['renderings'],
    'properties': {'renderings': {'type': 'array', 'minItems': 1, 'maxItems': 32, 'items': ENTRY}}}
SCHEMA_DIGEST = digest_canonical(SCHEMA)


class LocalisationHold(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LocalisationReference:
    invocation_id: str
    raw_admission_id: ObjectAdmissionId
    receipt_admission_id: ObjectAdmissionId


def localisation_policy(*, evidence_digest, qualified):
    return InvocationEfficiencyPolicy.create(policy_id=VERSION, version=VERSION,
        workload_class=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR, provider='grok-build-cli', route=ROUTE,
        model=MODEL, reasoning='high', one_turn=True, exact_input=True, skills_enabled=False,
        tools_enabled=False, mcp_enabled=False, prior_message_count=0, command_semantic_version=VERSION,
        command_flags=COMMAND_FLAGS, context_manifest_schema_version=VERSION,
        disabled_capabilities=CONT_DISABLED_CAPABILITIES, implementation_revision=digest_bytes(Path(__file__).read_bytes()),
        max_prompt_bytes=131072, max_context_tokens=500000, max_output_tokens=None, max_total_tokens=300000,
        prompt_contract_version=VERSION, output_schema_digest=SCHEMA_DIGEST,
        allowed_context_identities=(VERSION,), allowed_config_identities=(VERSION,),
        hard_estimate_ceiling_tokens=300000, evidence_digest=evidence_digest, qualified=qualified)


def _prompt(state, *, version=VERSION):
    if type(state) is not dict or set(state) != {'source_binding', 'claims'}:
        raise LocalisationHold('LOCALISATION_INPUT_HOLD')
    validate_sha256_digest(state['source_binding']['content_digest'])
    claims = state['claims']
    if type(claims) is not dict or not 0 < len(claims) <= 32:
        raise LocalisationHold('LOCALISATION_CLAIM_COUNT_HOLD')
    for identity, claim in claims.items():
        if type(identity) is not str or type(claim) is not dict or type(claim.get('text')) is not str:
            raise LocalisationHold('LOCALISATION_CLAIM_INPUT_HOLD')
        if len(claim.get('entities', ())) >= 65 or claim.get('rendering_fragment_count') != len(claim.get('entities', ())) + 1:
            raise LocalisationHold('LOCALISATION_FRAGMENT_INVENTORY_HOLD')
    # Authority, rights and caller IDs stay in the local manifest, not the model prompt.
    return canonical_json_bytes({'contract': version, 'claims': claims}).decode()


def _source_span_aliases(state):
    """Only an unambiguous single Source span may identify a rendering slot."""
    aliases = {}
    for identity, claim in state['claims'].items():
        source_range = claim.get('source_range')
        if type(source_range) is not dict or set(source_range) != {'first_span_id', 'last_span_id'}:
            continue
        span = source_range['first_span_id']
        if (span != source_range['last_span_id'] or type(span) is not str
                or re.fullmatch(r'S[1-9][0-9]*L[1-9][0-9]*', span) is None):
            continue
        if span == identity:
            continue
        if span in aliases or span in state['claims']:
            raise LocalisationHold('LOCALISATION_SPAN_PARTITION_HOLD')
        aliases[span] = identity
    return aliases


def _renderings(raw, state, *, version=VERSION):
    value = json.loads(raw.decode(), object_pairs_hook=_unique_object)
    validate(value, LEGACY_SCHEMA if version == LEGACY_VERSION else SCHEMA)
    if version == LEGACY_VERSION:
        renderings = value['renderings']
    else:
        renderings = {}
        for item in value['renderings']:
            identity = item['span_id']
            if identity in renderings:
                raise LocalisationHold('LOCALISATION_DUPLICATE_SPAN_HOLD')
            renderings[identity] = {key: value for key, value in item.items() if key != 'span_id'}
        if set(renderings) != set(state['claims']):
            aliases = _source_span_aliases(state)
            mapped = {}
            for identity, item in renderings.items():
                slot = aliases.get(identity, identity)
                if slot in mapped:
                    raise LocalisationHold('LOCALISATION_DUPLICATE_SPAN_HOLD')
                mapped[slot] = item
            renderings = mapped
    if set(renderings) != set(state['claims']):
        raise LocalisationHold('LOCALISATION_SPAN_PARTITION_HOLD')
    for identity, item in renderings.items():
        if len(item['rendered_assertion_zh_hant_hk_fragments']) != state['claims'][identity]['rendering_fragment_count']:
            raise LocalisationHold('LOCALISATION_FRAGMENT_COUNT_HOLD')
        # v1 reads retain their original character-based contract; v2 source keys are byte bounded.
        if version != LEGACY_VERSION and any(len(text.encode()) > 256 for text in (
                *item['quotation_source_keys'],
                *(pair['source_lookup_key'] for pair in item['factual_localisations']),
                *(pair['rendered_expression'] for pair in item['factual_localisations']))):
            raise LocalisationHold('LOCALISATION_SOURCE_KEY_BOUND_HOLD')
    return renderings


def _failure_diagnostic(error):
    """Record structure and digests, never exception messages or source text."""
    result = {'failure_class': type(error).__name__}
    if isinstance(error, ValidationError):
        value = json.dumps(error.instance, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
        result.update(validator=error.validator,
            instance_path=[str(part)[:64] for part in list(error.absolute_path)[:8]],
            schema_path=[str(part)[:64] for part in list(error.absolute_schema_path)[:8]],
            value_bytes=len(value), value_digest=digest_bytes(value))
    elif isinstance(error, json.JSONDecodeError):
        result.update(position=error.pos, line=error.lineno, column=error.colno)
    elif isinstance(error, LocalisationHold) and str(error) in {
            'LOCALISATION_DUPLICATE_SPAN_HOLD', 'LOCALISATION_SPAN_PARTITION_HOLD',
            'LOCALISATION_FRAGMENT_COUNT_HOLD', 'LOCALISATION_SOURCE_KEY_BOUND_HOLD',
            'LOCALISATION_RESULT_BOUND_HOLD'}:
        result['reason'] = str(error)
    return result


class NativeClaimLocaliser:
    def __init__(self, *, usage: ModelUsageService, objects, policy, source_fence, runner=None,
                 implementation_worktree_clean=False, clock=lambda: datetime.now(UTC)):
        if (not policy.qualified or (policy.provider, policy.route, policy.model, policy.reasoning)
                != ('grok-build-cli', ROUTE, MODEL, 'high') or policy.output_schema_digest != SCHEMA_DIGEST
                or policy.prompt_contract_version != VERSION or implementation_worktree_clean is not True
                or policy.implementation_revision != digest_bytes(Path(__file__).read_bytes())):
            raise LocalisationHold('LOCALISATION_POLICY_HOLD')
        usage.register_policy(policy)
        self.usage, self.objects, self.policy, self.fence, self.clock = usage, objects, policy, source_fence, clock
        self.implementation_worktree_clean = implementation_worktree_clean
        self.runner = runner or (lambda prompt: _run_grok_json(prompt, schema=SCHEMA,
            system_instruction=SYSTEM, temporary_prefix='newsroom-claim-localisation-',
            reasoning_effort='high', model=MODEL))

    def _input(self, state, *, candidate_id, hypothesis_digest, evidence_package_digest,
               version=VERSION, repair_of=None):
        prompt = _prompt(state, version=version)
        if len(prompt.encode()) > self.policy.max_prompt_bytes:
            raise LocalisationHold('LOCALISATION_INPUT_BOUND_HOLD')
        snapshot = {'state': state, 'candidate_id': candidate_id, 'hypothesis_digest': hypothesis_digest,
                    'evidence_package_digest': evidence_package_digest}
        digest = digest_canonical(snapshot)
        purpose = 'claim-localisation:' if version == LEGACY_VERSION else 'claim-localisation-v2:'
        if repair_of is not None:
            purpose = 'claim-localisation-v2-repair:'+repair_of+':'
        envelope = WorkEnvelope.create(cycle_id=purpose+digest,
            workload_class=self.policy.workload_class, admitted_at=self.clock(), admission_decision_id=None,
            candidate_id=candidate_id, hypothesis_digest=hypothesis_digest,
            evidence_package_digest=evidence_package_digest, ingest_id=None, graphiti_attempt_id=None)
        p = self.policy
        manifest = dict(schema_version=VERSION, provider=p.provider, route=ROUTE, model=MODEL, reasoning='high',
            command_semantic_version=VERSION, command_flags=list(p.command_flags), disabled_capabilities=list(p.disabled_capabilities),
            implementation_revision=p.implementation_revision, implementation_worktree_clean=self.implementation_worktree_clean,
            prompt_contract_version=VERSION, prompt_bytes=len(prompt.encode()), prompt_digest=digest_bytes(prompt.encode()),
            schema_digest=SCHEMA_DIGEST, output_schema_digest=SCHEMA_DIGEST, system_digest=digest_bytes(SYSTEM.encode()),
            evidence_package_digest=evidence_package_digest, evidence_package_bytes=len(canonical_json_bytes(snapshot)),
            context_identity=VERSION, config_identity=VERSION, one_turn=True, exact_input=True, skills_enabled=False,
            tools_enabled=False, mcp_enabled=False, prior_message_count=0, skill_count=0, tool_count=0, mcp_server_count=0,
            mcp_tool_count=0, source_snapshot_digest=digest)
        manifest['request_digest'] = digest_canonical({k: manifest[k] for k in ('provider', 'route', 'model', 'reasoning',
            'command_semantic_version', 'command_flags', 'implementation_revision', 'system_digest', 'prompt_digest', 'output_schema_digest')})
        manifest['context_manifest_digest'] = digest_canonical(manifest)
        return prompt, digest, envelope, manifest

    def _terminal(self, invocation_id, *, expected_snapshot=None):
        with sqlite3.connect(Path(self.usage.path).resolve().as_uri()+'?mode=ro', uri=True) as c:
            if c.execute('SELECT 1 FROM model_invocation_terminals WHERE invocation_id=?',
                         (invocation_id,)).fetchone() is None:
                raise LocalisationHold('LOCALISATION_PRIOR_ACTIVE_HOLD')
            allocation, terminal = _retained_terminal_allocation(c, invocation_id)
            policy = _policy_for_allocation(c, allocation)
            if terminal.usage_status is not UsageStatus.REPORTED or terminal.policy_breach:
                raise LocalisationHold('LOCALISATION_PRIOR_USAGE_HOLD')
            _require_reported_telemetry(c, terminal)
            envelope_row = c.execute('SELECT envelope_id,cycle_id,workload_class,admitted_at,canonical_digest,record_json '
                'FROM model_work_envelopes WHERE envelope_id=?', (allocation.envelope_id,)).fetchone()
            if envelope_row is None:
                raise LocalisationHold('LOCALISATION_REPLAY_SCOPE_HOLD')
            retained_envelope = _envelope_from_record(json.loads(envelope_row[5]))
            if (tuple(envelope_row[:5]) != (retained_envelope.envelope_id, retained_envelope.cycle_id,
                    retained_envelope.workload_class.value, retained_envelope.as_record()['admitted_at'], retained_envelope.canonical_digest)
                    or any(getattr(retained_envelope, key) != expected_snapshot[key]
                           for key in ('candidate_id','hypothesis_digest','evidence_package_digest'))):
                raise LocalisationHold('LOCALISATION_REPLAY_SCOPE_HOLD')
            row = c.execute('SELECT provider,route,evidence_package_digest,record_json FROM model_invocation_context_manifests '
                'WHERE context_manifest_digest=?', (allocation.context_manifest_digest,)).fetchone()
            if row is None:
                raise LocalisationHold('LOCALISATION_REPLAY_MANIFEST_HOLD')
            manifest = json.loads(row[3])
            unsigned = {key: value for key, value in manifest.items() if key != 'context_manifest_digest'}
            version = allocation.prompt_contract_version
            system = LEGACY_SYSTEM if version == LEGACY_VERSION else SYSTEM
            if (tuple(row[:3]) != (manifest.get('provider'), manifest.get('route'), manifest.get('evidence_package_digest'))
                    or manifest.get('evidence_package_digest') != expected_snapshot['evidence_package_digest']
                    or manifest.get('source_snapshot_digest') != digest_canonical(expected_snapshot)
                    or version not in {LEGACY_VERSION, VERSION}
                    or manifest.get('context_manifest_digest') != allocation.context_manifest_digest
                    or digest_canonical(unsigned) != allocation.context_manifest_digest
                    or manifest.get('system_digest') != digest_bytes(system.encode())
                    or manifest.get('prompt_digest') != allocation.prompt_digest
                    or manifest.get('output_schema_digest') != allocation.output_schema_digest
                    or manifest.get('implementation_revision') != policy.implementation_revision
                    or tuple(manifest.get('command_flags', ())) != COMMAND_FLAGS
                    or allocation.model != MODEL or allocation.reasoning != 'high'):
                raise LocalisationHold('LOCALISATION_REPLAY_MANIFEST_HOLD')
        return allocation, terminal, policy

    def _reference(self, invocation_id, *, proof):
        admitted = self.objects.committed_admission(ObjectAdmissionRequest('evidence.record',
            'claim-localisation-receipt:'+invocation_id), proof=proof)
        if admitted is None:
            raise LocalisationHold('LOCALISATION_PRIOR_RESULT_UNAVAILABLE')
        retained = json.loads(self.objects.rehydrate(HydrationRequest(admitted.admission.admission_id,
            'evidence.record'), proof=proof).data)
        return LocalisationReference(invocation_id, ObjectAdmissionId.parse(retained['raw_admission_id']),
            admitted.admission.admission_id)

    def localise(self, state, *, proof, **scope):
        # A version upgrade is not a retry credit for unknown, active or breached old work.
        legacy_prompt, _, legacy_envelope, _ = self._input(state, version=LEGACY_VERSION, **scope)
        legacy = _retained_allocation(self.usage, envelope=legacy_envelope,
            prompt_digest=digest_bytes(legacy_prompt.encode()), policy=self.policy)
        repair_of = None
        if legacy is not None:
            allocation, terminal, policy = self._terminal(legacy.invocation_id, expected_snapshot={
                'state':state, **{key:scope[key] for key in ('candidate_id','hypothesis_digest','evidence_package_digest')}})
            if (not policy.qualified or policy.prompt_contract_version != LEGACY_VERSION
                    or policy.output_schema_digest != LEGACY_SCHEMA_DIGEST
                    or allocation.prompt_contract_version != LEGACY_VERSION
                    or allocation.output_schema_digest != LEGACY_SCHEMA_DIGEST):
                raise LocalisationHold('LOCALISATION_LEGACY_POLICY_HOLD')
            if terminal.outcome == 'LOCALISATION_COMPLETE':
                reference = self._reference(legacy.invocation_id, proof=proof)
                self.read_localisation(reference, state, proof=proof, **scope)
                return reference
            if terminal.outcome != 'LOCALISATION_FAILED' or terminal.failure_class != 'ValidationError':
                raise LocalisationHold('LOCALISATION_LEGACY_FAILURE_HOLD')
            if terminal.components.provenance != 'PROVIDER_REPORTED' or terminal.pre_dispatch_zero_proved:
                raise LocalisationHold('LOCALISATION_LEGACY_FAILURE_HOLD')
            with sqlite3.connect(Path(self.usage.path).resolve().as_uri()+'?mode=ro', uri=True) as c:
                if not _has_exact_dispatch(c, terminal):
                    raise LocalisationHold('LOCALISATION_LEGACY_FAILURE_HOLD')
                dispatches = c.execute('SELECT observed_at,evidence_digest FROM model_transport_observations '
                    "WHERE invocation_id=? AND state='DISPATCH_STARTED'", (allocation.invocation_id,)).fetchall()
                if len(dispatches) != 1 or tuple(dispatches[0]) != (_utc_text(terminal.dispatch_at), allocation.request_digest):
                    raise LocalisationHold('LOCALISATION_LEGACY_FAILURE_HOLD')
            if ModelUsageService._validate_terminal(terminal, WorkloadClass.NATIVE_EVIDENCE_ASSESSOR, policy,
                    requested_max_output_tokens=allocation.max_output_tokens) is not None:
                raise LocalisationHold('LOCALISATION_LEGACY_FAILURE_HOLD')
            repair_of = legacy.invocation_id
        prompt, snapshot, envelope, manifest = self._input(state, repair_of=repair_of, **scope)
        prior = _retained_allocation(self.usage, envelope=envelope,
            prompt_digest=digest_bytes(prompt.encode()), policy=self.policy)
        if prior is not None:
            reference = self._reference(prior.invocation_id, proof=proof)
            self.read_localisation(reference, state, proof=proof, **scope)
            return reference
        envelope = self.usage.resume_or_open_native_assessor_envelope(envelope)
        self.usage.retain_context_manifest(manifest)
        allocation = InvocationAllocation.create(envelope_id=envelope.envelope_id, cycle_id=envelope.cycle_id,
            leaf_ordinal=1, workload_class=self.policy.workload_class, invocation_policy_digest=self.policy.canonical_digest,
            provider=self.policy.provider, route=ROUTE, model=MODEL, reasoning='high', prompt_contract_version=VERSION,
            prompt_bytes=len(prompt.encode()), prompt_digest=digest_bytes(prompt.encode()), request_digest=manifest['request_digest'],
            output_schema_digest=SCHEMA_DIGEST, max_output_tokens=None, context_manifest_digest=manifest['context_manifest_digest'],
            context_identity=VERSION, config_identity=VERSION, one_turn=True, exact_input=True, skills_enabled=False,
            tools_enabled=False, mcp_enabled=False, prior_message_count=0, allocated_at=self.clock(),
            recovery_deadline_at=self.clock()+timedelta(seconds=305), parent_invocation_id=None)
        self.usage.allocate(allocation, owner_emergency_stop=False)
        dispatched = None
        execution = None
        failure = None
        raw = None
        try:
            with self.fence(state['source_binding'], proof):
                dispatched = self.clock()
                self.usage.observe_transport(invocation_id=allocation.invocation_id, state='DISPATCH_STARTED',
                    observed_at=dispatched, evidence_digest=allocation.request_digest)
                execution = self.runner(prompt)
            raw = execution.text.encode()
            if len(raw) > 262144:
                raise LocalisationHold('LOCALISATION_RESULT_BOUND_HOLD')
            _renderings(raw, state)
        except BaseException as error:
            failure = error
        _complete_writer_usage(self.usage, allocation, outcome='LOCALISATION_COMPLETE' if failure is None else 'LOCALISATION_FAILED',
            failure_class=None if failure is None else type(failure).__name__,
            usage=None if execution is None else execution.usage, dispatch_at=dispatched,
            completed_at=self.clock(), provider_dispatched=dispatched is not None, policy=self.policy)
        terminal = self.usage.terminal(allocation.invocation_id)
        reference = None
        if raw is not None and len(raw) <= 262144:
            with self.fence(state['source_binding'], proof):
                raw_admission = self.objects.admit(ObjectAdmissionRequest('evidence.record',
                    'claim-localisation-raw:'+allocation.invocation_id), raw, proof=proof).admission
                receipt = {'version': VERSION, 'invocation_id': allocation.invocation_id,
                    'allocation_digest': allocation.canonical_digest, 'terminal_digest': terminal.terminal_digest,
                    'source_snapshot_digest': snapshot, 'source_binding': state['source_binding'],
                    'raw_admission_id': str(raw_admission.admission_id), 'raw_digest': digest_bytes(raw),
                    'schema_digest': SCHEMA_DIGEST, 'outcome': terminal.outcome, 'repair_of': repair_of,
                    'diagnostic': None if failure is None else _failure_diagnostic(failure)}
                admitted = self.objects.admit(ObjectAdmissionRequest('evidence.record',
                    'claim-localisation-receipt:'+allocation.invocation_id),
                    canonical_json_bytes(receipt), proof=proof).admission
                reference = LocalisationReference(allocation.invocation_id, raw_admission.admission_id,
                    admitted.admission_id)
        if failure is not None:
            raise failure
        if terminal.usage_status is not UsageStatus.REPORTED or terminal.policy_breach:
            raise LocalisationHold('LOCALISATION_USAGE_HOLD')
        return reference

    def read_localisation(self, reference, state, *, proof, **scope):
        with self.fence(state['source_binding'], proof):
            allocation, terminal, policy = self._terminal(reference.invocation_id, expected_snapshot={
                'state':state, **{key:scope[key] for key in ('candidate_id','hypothesis_digest','evidence_package_digest')}})
            raw = self.objects.rehydrate(HydrationRequest(reference.raw_admission_id, 'evidence.record'), proof=proof).data
            receipt_raw = self.objects.rehydrate(HydrationRequest(reference.receipt_admission_id, 'evidence.record'), proof=proof).data
        receipt = json.loads(receipt_raw)
        version = receipt.get('version')
        if version not in {LEGACY_VERSION, VERSION}:
            raise LocalisationHold('LOCALISATION_REPLAY_VERSION_HOLD')
        repair_of = receipt.get('repair_of') if version == VERSION else None
        _prompt_value, snapshot, envelope, _manifest = self._input(state, version=version,
            repair_of=repair_of, **scope)
        schema_digest = LEGACY_SCHEMA_DIGEST if version == LEGACY_VERSION else SCHEMA_DIGEST
        revalidate = (version == VERSION and terminal.outcome == 'LOCALISATION_FAILED'
            and terminal.failure_class == 'LocalisationHold'
            and receipt.get('diagnostic') == {'failure_class': 'LocalisationHold',
                                             'reason': 'LOCALISATION_SPAN_PARTITION_HOLD'})
        expected_outcome = 'LOCALISATION_FAILED' if revalidate else 'LOCALISATION_COMPLETE'
        if (canonical_json_bytes(receipt) != receipt_raw or not policy.qualified
                or allocation.envelope_id != envelope.envelope_id or allocation.prompt_digest != digest_bytes(_prompt_value.encode())
                or allocation.route != ROUTE or allocation.provider != 'grok-build-cli'
                or policy.prompt_contract_version != version or policy.output_schema_digest != schema_digest
                or allocation.prompt_contract_version != version or allocation.output_schema_digest != schema_digest
                or (version == VERSION and (receipt.get('schema_digest') != schema_digest
                    or receipt.get('outcome') != expected_outcome or (not revalidate and receipt.get('diagnostic') is not None)))
                or terminal is None or terminal.outcome != expected_outcome or terminal.usage_status is not UsageStatus.REPORTED
                or terminal.policy_breach or receipt['invocation_id'] != reference.invocation_id
                or receipt['allocation_digest'] != allocation.canonical_digest or receipt['terminal_digest'] != terminal.terminal_digest
                or receipt['source_snapshot_digest'] != snapshot or receipt['source_binding'] != state['source_binding']
                or receipt['raw_admission_id'] != str(reference.raw_admission_id) or receipt['raw_digest'] != digest_bytes(raw)):
            raise LocalisationHold('LOCALISATION_REPLAY_BINDING_HOLD')
        renderings = _renderings(raw, state, version=version)
        revalidation = {}
        if revalidate:
            aliases = _source_span_aliases(state)
            applied = {item['span_id']: aliases[item['span_id']]
                       for item in json.loads(raw)['renderings'] if item['span_id'] in aliases}
            if (not applied or len(raw) > 262144 or terminal.pre_dispatch_zero_proved
                    or terminal.dispatch_at is None or terminal.components.provenance != 'PROVIDER_REPORTED'
                    or ModelUsageService._validate_terminal(terminal, WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
                        policy, requested_max_output_tokens=allocation.max_output_tokens) is not None):
                raise LocalisationHold('LOCALISATION_REPLAY_USAGE_HOLD')
            with sqlite3.connect(Path(self.usage.path).resolve().as_uri()+'?mode=ro', uri=True) as connection:
                if not _has_exact_dispatch(connection, terminal):
                    raise LocalisationHold('LOCALISATION_REPLAY_USAGE_HOLD')
                dispatches = connection.execute('SELECT observed_at,evidence_digest FROM model_transport_observations '
                    "WHERE invocation_id=? AND state='DISPATCH_STARTED'", (allocation.invocation_id,)).fetchall()
                if len(dispatches) != 1 or tuple(dispatches[0]) != (_utc_text(terminal.dispatch_at), allocation.request_digest):
                    raise LocalisationHold('LOCALISATION_REPLAY_USAGE_HOLD')
            revalidation['consumer_revalidation'] = {'consumer_contract': CONSUMER_VERSION,
                'original_outcome': terminal.outcome, 'original_terminal_digest': terminal.terminal_digest,
                'raw_response_digest': digest_bytes(raw), 'source_span_aliases': applied}
        return {**receipt, **revalidation, 'renderings': renderings,
                'receipt_admission_id': str(reference.receipt_admission_id)}
