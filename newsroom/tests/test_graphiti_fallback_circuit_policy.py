from __future__ import annotations

import pytest

from newsroom.control_plane.graphiti_fallback_policy import (
    FallbackEligibility,
    GraphitiFallbackClass,
    classify_graphiti_fallback,
    load_checked_graphiti_fallback_circuit_policy,
    load_checked_native_graphiti_fallback_circuit_policy,
)
from newsroom.control_plane.graphiti_requests import (
    GraphitiLeafClass,
    load_checked_graphiti_call_shape_policy,
    load_checked_native_graphiti_call_shape_policy,
)
from newsroom.graphiti_adapter import cli_client, cursor_transport


@pytest.mark.parametrize(
    ("outcome", "eligibility", "outcome_class"),
    [
        (
            "MALFORMED_OUTPUT",
            FallbackEligibility.ELIGIBLE,
            GraphitiFallbackClass.MALFORMED_OUTPUT,
        ),
        (
            "COMPLETE",
            FallbackEligibility.NOT_NEEDED,
            GraphitiFallbackClass.COMPLETE,
        ),
        (
            "EXECUTABLE_NOT_FOUND",
            FallbackEligibility.INELIGIBLE,
            GraphitiFallbackClass.EXECUTABLE,
        ),
        (
            "PREDISPATCH_REFUSED",
            FallbackEligibility.INELIGIBLE,
            GraphitiFallbackClass.CONFIGURATION,
        ),
        (
            "FAILED",
            FallbackEligibility.INELIGIBLE,
            GraphitiFallbackClass.SYSTEMIC_TRANSPORT,
        ),
        (
            "TIMEOUT",
            FallbackEligibility.INELIGIBLE,
            GraphitiFallbackClass.TIMEOUT,
        ),
        (
            "CANCELLED",
            FallbackEligibility.INELIGIBLE,
            GraphitiFallbackClass.CANCELLATION,
        ),
        (
            "DISPATCH_FENCE_REFUSED",
            FallbackEligibility.INELIGIBLE,
            GraphitiFallbackClass.DISPATCH_FENCE,
        ),
        (
            "OUTPUT_LIMIT_EXCEEDED",
            FallbackEligibility.INELIGIBLE,
            GraphitiFallbackClass.CONTEXT_OUTPUT_BREACH,
        ),
        (
            "AUTHENTICATION_FAILED",
            FallbackEligibility.INELIGIBLE,
            GraphitiFallbackClass.AUTHENTICATION,
        ),
        (
            "QUOTA_EXCEEDED",
            FallbackEligibility.INELIGIBLE,
            GraphitiFallbackClass.QUOTA,
        ),
        (
            "UNRECOGNISED_FAILURE",
            FallbackEligibility.INELIGIBLE,
            GraphitiFallbackClass.UNKNOWN,
        ),
    ],
)
def test_public_fallback_classifier_is_fail_closed(
    outcome: str,
    eligibility: FallbackEligibility,
    outcome_class: GraphitiFallbackClass,
) -> None:
    decision = classify_graphiti_fallback(outcome)

    assert decision.outcome == outcome
    assert decision.eligibility is eligibility
    assert decision.outcome_class is outcome_class


def test_checked_fallback_policy_is_bound_to_call_shape_and_729_release_order() -> None:
    policy = load_checked_native_graphiti_fallback_circuit_policy()
    call_shape = load_checked_native_graphiti_call_shape_policy()

    assert policy.version == "issue-981-grok-build-fallback-v1"
    assert policy.call_shape_policy_digest == call_shape.canonical_digest
    assert policy.eligible_outcomes == ("MALFORMED_OUTPUT",)
    assert policy.max_fallback_leaves_per_primary == 1
    assert policy.circuit_release_preference == (
        "DETERMINISTIC_HEALTH_PROBE",
        "AUTHORISED_OPERATOR_RESET",
    )
    assert {route.route for route in call_shape.qualified_routes} == {
        "GRAPHITI_CHAT_PRIMARY",
        "GRAPHITI_CHAT_FALLBACK",
        "GRAPHITI_EMBEDDING",
    }
    for leaf_class in GraphitiLeafClass:
        # Qualified software identity is retained provenance, not a runtime pin.
        assert call_shape.route_for(leaf_class).implementation_revision.startswith(
            "sha256:"
        )
    primary = call_shape.route_for(GraphitiLeafClass.PRIMARY)
    assert primary.config_identity == "cursor-sdk-api-key-composer-floor-v2"
    # Check current capabilities here; exact v12/v13 identity and replay are
    # covered by the dedicated native SDK response-contract tests.
    assert "REPORTED_OUTPUT_TOKENS=ADVISORY_INCLUDES_REASONING_V1" in primary.command_flags
    assert "REPORTED_TOTAL_TOKENS=ADVISORY_CUMULATIVE_CONSUMPTION_V1" in primary.command_flags
    assert f"sdk={cursor_transport.MIN_SDK_REQUIREMENT}" in primary.command_flags
    assert "auth=CURSOR_API_KEY" in primary.command_flags
    assert "composer_floor>=2.5" in primary.command_flags
    assert "selected_model=RUNTIME_SELECTED" in primary.command_flags
    assert "tools=EMPTY" in primary.command_flags
    assert "disallowed_tools=shell,mcp,task" in primary.command_flags
    assert "mcp_servers=EMPTY" in primary.command_flags
    assert "custom_tools=EMPTY" in primary.command_flags
    assert "subagents=EMPTY" in primary.command_flags
    assert "setting_sources=OMITTED" in primary.command_flags
    assert "cwd=EMPTY_NON_GIT" in primary.command_flags
    assert "store=EPHEMERAL_LOCAL_ISOLATED" in primary.command_flags
    assert "fresh_run=TRUE" in primary.command_flags
    assert "resume=FALSE" in primary.command_flags
    assert "max_retries=0" in primary.command_flags
    assert "CONTROLLER_TIMEOUT_MS=160000" in primary.command_flags
    assert (
        "TIMEOUT_DIAGNOSTIC_SCHEMA=newsroom.graphiti-timeout-diagnostic.v1"
        in primary.command_flags
    )
    assert (
        f"CONTROLLER_OUTPUT_CONTRACT={cursor_transport.CURSOR_OUTPUT_LIMIT_IDENTITY}"
        in primary.command_flags
    )
    assert "binary=" not in " ".join(primary.command_flags)
    assert "AUTHENTICATION_BRIDGE=" not in " ".join(primary.command_flags)
    assert "cursor-agent" not in " ".join(primary.command_flags)
    fallback = call_shape.route_for(GraphitiLeafClass.FALLBACK)
    assert fallback.config_identity == "graphiti-grok-hermetic-user-config-v2"
    assert fallback.command_semantic_version == (
        "newsroom.graphiti-provider-dispatch.v8"
    )
    assert "--max-output-tokens=REQUEST_MAX_TOKENS" not in fallback.command_flags
    assert (
        "USER_CONFIG_MODEL_MAX_COMPLETION_TOKENS=REQUEST_MAX_TOKENS"
        in fallback.command_flags
    )
    assert cli_client.GROK_COMPLETION_LIMIT_IDENTITY.endswith(
        'model."grok-4.6".max_completion_tokens=REQUEST_MAX_TOKENS'
    )
    assert "USER_CONFIG_SCOPE=ISOLATED_HOME" in fallback.command_flags
    assert "AUTH_STORAGE=GROK_AUTH_PATH" in fallback.command_flags
    assert "--verbatim" in fallback.command_flags
    assert "USER_CONFIG_TITLE_REFRESH=FALSE" in fallback.command_flags
    assert "USER_CONFIG_TURN_SUMMARY=FALSE" in fallback.command_flags
    assert "USER_CONFIG_MAX_RETRIES=0" in fallback.command_flags
    assert (
        "ACP_SEQUENCE=initialize,session/new,_x.ai/session/rename,session/close"
        in fallback.command_flags
    )
    assert "ACP_INFERENCE_REQUESTS=0" in fallback.command_flags
    assert "--resume=ACP_SESSION_UUID" in fallback.command_flags
    assert "CONTROLLER_TIMEOUT_MS=160000" in fallback.command_flags
    assert (
        "TIMEOUT_DIAGNOSTIC_SCHEMA=newsroom.graphiti-timeout-diagnostic.v1"
        in fallback.command_flags
    )
    assert (
        f"CONTROLLER_STDOUT_CONTRACT={cli_client.GROK_STDOUT_LIMIT_IDENTITY}"
        in fallback.command_flags
    )


def test_native_fallback_policy_does_not_rewrite_the_checked_campaign_policy() -> None:
    policy = load_checked_graphiti_fallback_circuit_policy()
    call_shape = load_checked_graphiti_call_shape_policy()

    assert policy.version == "issue-816-v2"
    assert call_shape.version == "issue-816-v2"
    assert policy.call_shape_policy_digest == call_shape.canonical_digest
