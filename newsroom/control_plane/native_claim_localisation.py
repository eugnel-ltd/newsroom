"""One accounted rendering of already selected source claims; never select facts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sqlite3

from jsonschema import validate

from newsroom.authority import HydrationRequest, ObjectAdmissionId, ObjectAdmissionRequest
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical, validate_sha256_digest
from .cycle import _complete_writer_usage
from .govuk_evidence import _unique_object
from .model_usage import (
    InvocationAllocation, InvocationEfficiencyPolicy, ModelUsageIntegrityError, ModelUsageService,
    UsageStatus, WorkEnvelope, WorkloadClass, _policy_for_allocation,
    _require_reported_telemetry, _retained_terminal_allocation,
)
from .native_embeddings import _retained_allocation
from .writer import _run_grok_json, CONT_DISABLED_CAPABILITIES, _grok_command_flags

VERSION = 'newsroom.native-claim-localisation.v1'
ROUTE = 'NATIVE_CLAIM_LOCALISATION'
MODEL = 'grok-4.7'
COMMAND_FLAGS = _grok_command_flags('high', model=MODEL)
SYSTEM = ('Render only the supplied factual source assertions in natural Hong Kong Traditional Chinese. '
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
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['renderings'],
    'properties': {'renderings': {'type': 'object', 'additionalProperties': ITEM}}}
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


def _prompt(state):
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
    return canonical_json_bytes({'contract': VERSION, 'claims': claims}).decode()


def _renderings(raw, state):
    value = json.loads(raw.decode(), object_pairs_hook=_unique_object)
    validate(value, SCHEMA)
    if set(value['renderings']) != set(state['claims']):
        raise LocalisationHold('LOCALISATION_SPAN_PARTITION_HOLD')
    for identity, item in value['renderings'].items():
        if len(item['rendered_assertion_zh_hant_hk_fragments']) != state['claims'][identity]['rendering_fragment_count']:
            raise LocalisationHold('LOCALISATION_FRAGMENT_COUNT_HOLD')
    return value['renderings']


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

    def _input(self, state, *, candidate_id, hypothesis_digest, evidence_package_digest):
        prompt = _prompt(state)
        if len(prompt.encode()) > self.policy.max_prompt_bytes:
            raise LocalisationHold('LOCALISATION_INPUT_BOUND_HOLD')
        snapshot = {'state': state, 'candidate_id': candidate_id, 'hypothesis_digest': hypothesis_digest,
                    'evidence_package_digest': evidence_package_digest}
        digest = digest_canonical(snapshot)
        envelope = WorkEnvelope.create(cycle_id='claim-localisation:'+digest,
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

    def localise(self, state, *, proof, **scope):
        prompt, snapshot, envelope, manifest = self._input(state, **scope)
        prior = _retained_allocation(self.usage, envelope=envelope, prompt_digest=digest_bytes(prompt.encode()), policy=self.policy)
        if prior is not None:
            admitted = self.objects.committed_admission(ObjectAdmissionRequest('evidence.record',
                'claim-localisation-receipt:'+prior.invocation_id), proof=proof)
            if admitted is None:
                raise LocalisationHold('LOCALISATION_PRIOR_RESULT_UNAVAILABLE')
            retained = json.loads(self.objects.rehydrate(HydrationRequest(admitted.admission.admission_id,
                'evidence.record'), proof=proof).data)
            reference = LocalisationReference(prior.invocation_id, ObjectAdmissionId.parse(retained['raw_admission_id']),
                admitted.admission.admission_id)
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
        if failure is not None:
            raise failure
        terminal = self.usage.terminal(allocation.invocation_id)
        if terminal.usage_status is not UsageStatus.REPORTED or terminal.policy_breach:
            raise LocalisationHold('LOCALISATION_USAGE_HOLD')
        with self.fence(state['source_binding'], proof):
            raw_admission = self.objects.admit(ObjectAdmissionRequest('evidence.record', 'claim-localisation-raw:'+allocation.invocation_id), raw, proof=proof).admission
            receipt = {'version': VERSION, 'invocation_id': allocation.invocation_id, 'allocation_digest': allocation.canonical_digest,
                'terminal_digest': terminal.terminal_digest, 'source_snapshot_digest': snapshot, 'source_binding': state['source_binding'],
                'raw_admission_id': str(raw_admission.admission_id), 'raw_digest': digest_bytes(raw)}
            admitted = self.objects.admit(ObjectAdmissionRequest('evidence.record', 'claim-localisation-receipt:'+allocation.invocation_id),
                canonical_json_bytes(receipt), proof=proof).admission
        return LocalisationReference(allocation.invocation_id, raw_admission.admission_id, admitted.admission_id)

    def read_localisation(self, reference, state, *, proof, **scope):
        _prompt_value, snapshot, envelope, _manifest = self._input(state, **scope)
        with self.fence(state['source_binding'], proof):
            with sqlite3.connect(Path(self.usage.path).resolve().as_uri()+'?mode=ro', uri=True) as c:
                allocation, terminal = _retained_terminal_allocation(c, reference.invocation_id)
                policy = _policy_for_allocation(c, allocation)
                if terminal is None or terminal.usage_status is not UsageStatus.REPORTED:
                    raise LocalisationHold('LOCALISATION_REPLAY_USAGE_HOLD')
                _require_reported_telemetry(c, terminal)
            raw = self.objects.rehydrate(HydrationRequest(reference.raw_admission_id, 'evidence.record'), proof=proof).data
            receipt_raw = self.objects.rehydrate(HydrationRequest(reference.receipt_admission_id, 'evidence.record'), proof=proof).data
        receipt = json.loads(receipt_raw)
        if (canonical_json_bytes(receipt) != receipt_raw or receipt['version'] != VERSION
                or allocation.envelope_id != envelope.envelope_id or allocation.prompt_digest != digest_bytes(_prompt_value.encode())
                or allocation.route != ROUTE or allocation.provider != 'grok-build-cli'
                or policy.prompt_contract_version != VERSION or policy.output_schema_digest != SCHEMA_DIGEST
                or terminal is None or terminal.outcome != 'LOCALISATION_COMPLETE' or terminal.usage_status is not UsageStatus.REPORTED
                or terminal.policy_breach or receipt['invocation_id'] != reference.invocation_id
                or receipt['allocation_digest'] != allocation.canonical_digest or receipt['terminal_digest'] != terminal.terminal_digest
                or receipt['source_snapshot_digest'] != snapshot or receipt['source_binding'] != state['source_binding']
                or receipt['raw_admission_id'] != str(reference.raw_admission_id) or receipt['raw_digest'] != digest_bytes(raw)):
            raise LocalisationHold('LOCALISATION_REPLAY_BINDING_HOLD')
        return {**receipt, 'renderings': _renderings(raw, state), 'receipt_admission_id': str(reference.receipt_admission_id)}
