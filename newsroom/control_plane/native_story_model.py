"""Accounted, once-only Grok calls for native drafting and source review."""
from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime, timedelta
import json
from typing import Callable

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical

from .cycle import _complete_writer_usage
from .model_usage import (
    InvocationAllocation, InvocationEfficiencyPolicy, ModelUsageAdmissionError,
    ModelUsageService, UsageStatus, WorkEnvelope, WorkloadClass, _policy_from_record,
    _envelope_from_record, _retained_terminal_allocation,
    _allocation_from_record,
)
from .writer import (
    CONT_DISABLED_CAPABILITIES, _grok_command_flags, _run_grok_json,
    cont_writer_implementation_identity, read_grok_command_semantic_version,
)

VERSION = "newsroom.native-story-model.v1"
CONTEXT = "native-approved-claim-inventory-v1"
CONFIG = "native-story-grok-hermetic-command-v1"
MANIFEST = "newsroom.native-story-model-context.v1"
ROUTES = {"DRAFT": "NATIVE_STORY_DRAFT", "REVIEW": "NATIVE_STORY_REVIEW"}
FLAGS = _grok_command_flags("high", model="grok-4.7")


def story_model_policy(template, *, phase, schema, revision, evidence_digest):
    """Reuse the qualified transport controls, with an explicit authoring contract."""
    values = asdict(template)
    values.pop("canonical_digest")
    values.update(
        policy_id="native-story-" + phase.lower(), version=revision,
        workload_class=WorkloadClass.NATIVE_STORY_WRITER,
        route=ROUTES[phase], model="grok-4.7", reasoning="high",
        implementation_revision=revision, prompt_contract_version=VERSION + ":" + phase,
        output_schema_digest=digest_canonical(schema), command_flags=FLAGS,
        context_manifest_schema_version=MANIFEST,
        allowed_context_identities=(CONTEXT,), allowed_config_identities=(CONFIG,),
        max_prompt_bytes=65_536, max_output_tokens=None,
        evidence_digest=evidence_digest, calibration_only=False,
        allowed_candidate_ids=(), hard_estimate_ceiling_tokens=None, qualified=True,
    )
    return InvocationEfficiencyPolicy.create(**values)


class NativeStoryModel:
    def __init__(self, service: ModelUsageService, policies: dict, *, fence,
                 stop_check: Callable[[], None], clock=lambda: datetime.now(UTC),
                 invoke=_run_grok_json, cached_only=False):
        if type(cached_only) is not bool:
            raise ValueError('native story cache mode differs')
        self.cached_only = cached_only
        self.service, self.policies = service, policies
        self.fence, self.stop_check, self.clock, self.invoke = fence, stop_check, clock, invoke
        self.receipts = {}
        with service._connection() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS native_story_model_results(
                request_key TEXT PRIMARY KEY, invocation_id TEXT NOT NULL UNIQUE,
                package_digest TEXT NOT NULL, phase TEXT NOT NULL,
                input_digest TEXT NOT NULL, response_text TEXT NOT NULL,
                response_digest TEXT NOT NULL, terminal_digest TEXT NOT NULL)""")

    def call(self, request: dict, *, phase: str, schema: dict, system: str,
             candidate_id: str, hypothesis_digest: str, admission_decision_id: str):
        self.stop_check()
        package_digest = request["source_package_digest"]
        prompt = canonical_json_bytes(request).decode()
        input_digest = digest_canonical({"request": request, "system": system, "schema": schema})
        key = digest_canonical({"version": VERSION, "package": package_digest, "phase": phase})
        with self.service._connection() as connection:
            row = connection.execute("SELECT * FROM native_story_model_results WHERE request_key=?", (key,)).fetchone()
            binding = None if row is None else connection.execute("""SELECT e.record_json,e.envelope_id,
                e.cycle_id,e.workload_class,e.admitted_at,e.canonical_digest
                FROM model_invocation_allocations a JOIN model_work_envelopes e USING(envelope_id)
                WHERE a.invocation_id=?""", (row[1],)).fetchone()
            authority = None if row is None else _retained_terminal_allocation(connection, row[1])
        if row is not None:
            _, invocation, source, retained_phase, retained_input, text, response_digest, terminal_digest = row
            allocation, terminal = authority
            if (source != package_digest or retained_phase != phase or retained_input != input_digest
                    or digest_bytes(text.encode()) != response_digest or terminal is None
                    or terminal.terminal_digest != terminal_digest):
                raise ModelUsageAdmissionError("retained story model response differs", reason_code="NATIVE_STORY_RESULT_DIFFERS")
            envelope = None if binding is None else _envelope_from_record(json.loads(binding[0]))
            envelope_record = {} if envelope is None else envelope.as_record()
            if (binding is None or tuple(binding[1:]) != (
                    envelope.envelope_id, envelope.cycle_id, envelope.workload_class.value,
                    envelope_record["admitted_at"], envelope.canonical_digest)
                    or allocation.route != ROUTES[phase]
                    or allocation.workload_class is not WorkloadClass.NATIVE_STORY_WRITER
                    or allocation.envelope_id != envelope.envelope_id
                    or any(envelope_record.get(name) != expected for name, expected in (
                        ("candidate_id", candidate_id), ("hypothesis_digest", hypothesis_digest),
                        ("admission_decision_id", admission_decision_id), ("evidence_package_digest", package_digest)))):
                raise ModelUsageAdmissionError("retained story editorial identity differs", reason_code="NATIVE_STORY_IDENTITY_DIFFERS")
            self._require_terminal(terminal)
            self.receipts[phase] = {"invocation_id": invocation, "terminal_digest": terminal_digest,
                                    "response_digest": response_digest}
            return json.loads(text)

        if self.cached_only:
            raise ModelUsageAdmissionError('retained story response unavailable',
                reason_code='NATIVE_STORY_RESULT_NOT_RETAINED')
        self.stop_check()
        policy = self.policies[phase]
        if (policy.route != ROUTES[phase] or policy.output_schema_digest != digest_canonical(schema)
                or policy.workload_class is not WorkloadClass.NATIVE_STORY_WRITER
                or policy.max_output_tokens is not None):
            raise ModelUsageAdmissionError("story model policy differs", reason_code="NATIVE_STORY_POLICY_DIFFERS")
        revision, clean = cont_writer_implementation_identity()
        # Software revisions are retained audit facts, not route qualifications.
        if not clean:
            raise ModelUsageAdmissionError("story model implementation differs", reason_code="NATIVE_STORY_IMPLEMENTATION_DIFFERS")
        now = self.clock().astimezone(UTC)
        envelope = WorkEnvelope.create(
            cycle_id=key, workload_class=WorkloadClass.NATIVE_STORY_WRITER,
            admitted_at=now, admission_decision_id=admission_decision_id,
            candidate_id=candidate_id, hypothesis_digest=hypothesis_digest,
            evidence_package_digest=package_digest, ingest_id=None, graphiti_attempt_id=None,
        )
        envelope = self.service.resume_or_open_native_story_envelope(envelope)
        manifest = {
            "schema_version": MANIFEST, "provider": policy.provider,
            "route": policy.route, "model": policy.model, "reasoning": policy.reasoning,
            "prompt_contract_version": policy.prompt_contract_version,
            "prompt_digest": digest_bytes(prompt.encode()), "prompt_bytes": len(prompt.encode()),
            "system_digest": digest_bytes(system.encode()), "output_schema_digest": policy.output_schema_digest,
            "schema_digest": policy.output_schema_digest,
            "evidence_package_bytes": len(canonical_json_bytes(request.get("evidence", request))),
            "command_semantic_version": read_grok_command_semantic_version(),
            "command_flags": list(FLAGS), "implementation_revision": revision,
            "implementation_worktree_clean": clean, "context_identity": CONTEXT,
            "config_identity": CONFIG, "evidence_package_digest": package_digest,
            "one_turn": True, "exact_input": True, "skills_enabled": False,
            "tools_enabled": False, "mcp_enabled": False, "prior_message_count": 0,
            "skill_count": 0, "tool_count": 0, "mcp_server_count": 0, "mcp_tool_count": 0,
            "disabled_capabilities": list(CONT_DISABLED_CAPABILITIES),
        }
        manifest["request_digest"] = digest_canonical({name: manifest[name] for name in (
            "provider", "route", "model", "reasoning", "command_semantic_version",
            "command_flags", "implementation_revision", "system_digest", "prompt_digest", "output_schema_digest")})
        manifest["context_manifest_digest"] = digest_canonical(manifest)
        self.service.retain_context_manifest(manifest)
        allocation = InvocationAllocation.create(
            envelope_id=envelope.envelope_id, cycle_id=key, leaf_ordinal=1,
            workload_class=WorkloadClass.NATIVE_STORY_WRITER,
            invocation_policy_digest=policy.canonical_digest, provider=policy.provider,
            route=policy.route, model=policy.model, reasoning=policy.reasoning,
            prompt_contract_version=policy.prompt_contract_version,
            prompt_bytes=len(prompt.encode()), prompt_digest=manifest["prompt_digest"],
            request_digest=manifest["request_digest"], output_schema_digest=policy.output_schema_digest,
            max_output_tokens=None, context_manifest_digest=manifest["context_manifest_digest"],
            context_identity=CONTEXT, config_identity=CONFIG, one_turn=True, exact_input=True,
            skills_enabled=False, tools_enabled=False, mcp_enabled=False, prior_message_count=0,
            allocated_at=now, recovery_deadline_at=now + timedelta(minutes=6), parent_invocation_id=None,
        )
        self.service.allocate(allocation, owner_emergency_stop=False)
        dispatch_at = None
        try:
            with self.fence():
                self.stop_check()
                dispatch_at = self.clock().astimezone(UTC)
                self.service.observe_transport(invocation_id=allocation.invocation_id,
                    observed_at=dispatch_at, state="DISPATCH_STARTED", evidence_digest=allocation.request_digest)
                execution = self.invoke(prompt, schema=schema, system_instruction=system,
                    temporary_prefix="newsroom-native-story-", reasoning_effort="high", model="grok-4.7")
        except Exception as exc:
            _complete_writer_usage(self.service, allocation, outcome="FAILED",
                failure_class=type(exc).__name__, usage=getattr(exc, "usage", None),
                dispatch_at=dispatch_at, completed_at=self.clock(), policy=policy,
                provider_dispatched=dispatch_at is not None)
            raise
        _complete_writer_usage(self.service, allocation, outcome="COMPLETE", failure_class=None,
            usage=execution.usage, dispatch_at=dispatch_at, completed_at=self.clock(), policy=policy)
        terminal = self.service.terminal(allocation.invocation_id)
        # Preserve the exact response before interpreting it; a malformed answer is
        # a settled result, not permission to dispatch the same package again.
        with self.service._connection() as connection:
            connection.execute("INSERT INTO native_story_model_results VALUES(?,?,?,?,?,?,?,?)", (
                key, allocation.invocation_id, package_digest, phase, input_digest,
                execution.text, digest_bytes(execution.text.encode()), terminal.terminal_digest,
            ))
        self.receipts[phase] = {"invocation_id": allocation.invocation_id,
                                "terminal_digest": terminal.terminal_digest,
                                "response_digest": digest_bytes(execution.text.encode())}
        self._require_terminal(terminal)
        return json.loads(execution.text)

    def _draft_system(self, package_digest, *, candidate_id, hypothesis_digest, admission_decision_id):
        """Replay the original known prompt for allocated intent; never new spend."""
        from .native_story_writer import DRAFT_SYSTEM, LEGACY_DRAFT_SYSTEM
        from .native_assessor import _retained_context
        key = digest_canonical({"version": VERSION, "package": package_digest, "phase": "DRAFT"})
        envelope = WorkEnvelope.create(
            cycle_id=key, workload_class=WorkloadClass.NATIVE_STORY_WRITER,
            admitted_at=self.clock().astimezone(UTC), admission_decision_id=admission_decision_id,
            candidate_id=candidate_id, hypothesis_digest=hypothesis_digest,
            evidence_package_digest=package_digest, ingest_id=None, graphiti_attempt_id=None,
        )
        with self.service._connection() as connection:
            rows = connection.execute("SELECT invocation_id,record_json FROM model_invocation_allocations "
                "WHERE envelope_id=? ORDER BY leaf_ordinal LIMIT 2", (envelope.envelope_id,)).fetchall()
            if not rows:
                return DRAFT_SYSTEM
            if len(rows) != 1:
                raise ModelUsageAdmissionError("retained draft prompt allocation differs", reason_code="NATIVE_STORY_PROMPT_DIFFERS")
            try:
                allocation = _allocation_from_record(json.loads(rows[0][1]))
                if (allocation.invocation_id != rows[0][0] or allocation.envelope_id != envelope.envelope_id
                        or allocation.cycle_id != key or allocation.route != ROUTES["DRAFT"]
                        or allocation.workload_class is not WorkloadClass.NATIVE_STORY_WRITER):
                    raise ModelUsageAdmissionError("retained draft prompt identity differs", reason_code="NATIVE_STORY_PROMPT_DIFFERS")
                context = _retained_context(connection, allocation)
            except (ValueError, KeyError, TypeError) as exc:
                raise ModelUsageAdmissionError("retained draft prompt binding differs", reason_code="NATIVE_STORY_PROMPT_DIFFERS") from exc
            for system in (LEGACY_DRAFT_SYSTEM, DRAFT_SYSTEM):
                if context.get("system_digest") == digest_bytes(system.encode()):
                    return system
        raise ModelUsageAdmissionError("retained draft prompt is unknown", reason_code="NATIVE_STORY_PROMPT_UNKNOWN")

    def write(self, package, *, require_current=lambda: None, source_currentness=(), **identities):
        from dataclasses import replace
        from .native_story_writer import (
            DRAFT_SCHEMA, REVIEW_SCHEMA, REVIEW_SYSTEM,
            SourceSupportReview, write_native_story,
        )
        def generate(request):
            require_current()
            system = self._draft_system(package.digest, **identities)
            return self.call(request, phase="DRAFT", schema=DRAFT_SCHEMA, system=system, **identities)
        def review(request):
            require_current()
            return self.call(request, phase="REVIEW", schema=REVIEW_SCHEMA, system=REVIEW_SYSTEM, **identities)
        result = write_native_story(package, generate=generate, review=review,
            source_currentness=source_currentness)
        require_current()
        record = result.review.as_record()
        record["model_receipts"] = self.receipts
        return replace(result, review=SourceSupportReview(canonical_json_bytes(record)))

    @staticmethod
    def _require_terminal(terminal):
        if (terminal is None or terminal.outcome != "COMPLETE"
                or terminal.usage_status not in {UsageStatus.REPORTED, UsageStatus.ESTIMATED}
                or terminal.policy_breach is not None):
            raise ModelUsageAdmissionError("story model usage requires settlement", reason_code="NATIVE_STORY_MODEL_USAGE_HOLD")


def load_story_model_policies(service):
    """Resolve only the current reviewed policy for each explicit native route."""
    from .native_story_writer import DRAFT_SCHEMA, REVIEW_SCHEMA
    result = {}
    with service._connection() as connection:
        for phase, schema in (("DRAFT", DRAFT_SCHEMA), ("REVIEW", REVIEW_SCHEMA)):
            row = connection.execute("""SELECT record_json FROM model_invocation_policies
                WHERE workload_class=? AND route=? AND qualified=1 ORDER BY rowid DESC LIMIT 1""",
                (WorkloadClass.NATIVE_STORY_WRITER.value, ROUTES[phase])).fetchone()
            if row is None:
                raise ModelUsageAdmissionError("native writer policy requires registration",
                                               reason_code="NATIVE_STORY_POLICY_UNAVAILABLE")
            policy = _policy_from_record(json.loads(row[0]))
            if (policy.prompt_contract_version != VERSION + ":" + phase
                    or policy.output_schema_digest != digest_canonical(schema)
                    or policy.command_flags != FLAGS or policy.allowed_context_identities != (CONTEXT,)
                    or policy.allowed_config_identities != (CONFIG,)):
                raise ModelUsageAdmissionError("native writer policy differs", reason_code="NATIVE_STORY_POLICY_DIFFERS")
            result[phase] = policy
    return result
