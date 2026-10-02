"""Checked fallback eligibility and route-circuit policy for Graphiti leaves."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from newsroom.authority.canonical import digest_canonical

GRAPHITI_FALLBACK_CIRCUIT_SCHEMA_VERSION = (
    "newsroom.graphiti-fallback-circuit-policy.v1"
)
_POLICY_PATH = Path(__file__).with_name("graphiti_fallback_circuit_policy_v1.json")
_NATIVE_POLICY_PATH = Path(__file__).with_name(
    "native_graphiti_fallback_circuit_policy_v2.json"
)


class FallbackEligibility(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    NOT_NEEDED = "NOT_NEEDED"


class GraphitiFallbackClass(StrEnum):
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"
    COMPLETE = "COMPLETE"
    EXECUTABLE = "EXECUTABLE"
    CONFIGURATION = "CONFIGURATION"
    SYSTEMIC_TRANSPORT = "SYSTEMIC_TRANSPORT"
    TIMEOUT = "TIMEOUT"
    CANCELLATION = "CANCELLATION"
    DISPATCH_FENCE = "DISPATCH_FENCE"
    CONTEXT_OUTPUT_BREACH = "CONTEXT_OUTPUT_BREACH"
    AUTHENTICATION = "AUTHENTICATION"
    QUOTA = "QUOTA"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class GraphitiFallbackDecision:
    outcome: str
    eligibility: FallbackEligibility
    outcome_class: GraphitiFallbackClass


@dataclass(frozen=True, slots=True)
class GraphitiFallbackCircuitPolicy:
    policy_id: str
    version: str
    call_shape_policy_digest: str
    outcome_classes: dict[str, GraphitiFallbackClass]
    eligible_outcomes: tuple[str, ...]
    no_fallback_needed_outcomes: tuple[str, ...]
    circuit_open_classes: tuple[GraphitiFallbackClass, ...]
    max_fallback_leaves_per_primary: int
    circuit_release_preference: tuple[str, ...]
    canonical_digest: str

    def classify(self, outcome: str) -> GraphitiFallbackDecision:
        outcome_class = self.outcome_classes.get(
            outcome, GraphitiFallbackClass.UNKNOWN
        )
        eligibility = (
            FallbackEligibility.ELIGIBLE
            if outcome in self.eligible_outcomes
            else FallbackEligibility.NOT_NEEDED
            if outcome in self.no_fallback_needed_outcomes
            else FallbackEligibility.INELIGIBLE
        )
        return GraphitiFallbackDecision(outcome, eligibility, outcome_class)


def _load_checked_graphiti_fallback_circuit_policy(
    path: Path,
) -> GraphitiFallbackCircuitPolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("checked Graphiti fallback policy is not an object")
    expected_digest = payload.pop("canonical_digest", None)
    if payload.get("schema_version") != GRAPHITI_FALLBACK_CIRCUIT_SCHEMA_VERSION:
        raise ValueError("checked Graphiti fallback policy schema differs")
    canonical_digest = digest_canonical(payload)
    if expected_digest != canonical_digest:
        raise ValueError("checked Graphiti fallback policy digest changed")
    outcome_classes = payload.get("outcome_classes")
    if not isinstance(outcome_classes, dict):
        raise ValueError("checked Graphiti outcome classes are absent")
    policy = GraphitiFallbackCircuitPolicy(
        policy_id=str(payload["policy_id"]),
        version=str(payload["version"]),
        call_shape_policy_digest=str(payload["call_shape_policy_digest"]),
        outcome_classes={
            str(outcome): GraphitiFallbackClass(str(outcome_class))
            for outcome, outcome_class in outcome_classes.items()
        },
        eligible_outcomes=tuple(str(item) for item in payload["eligible_outcomes"]),
        no_fallback_needed_outcomes=tuple(
            str(item) for item in payload["no_fallback_needed_outcomes"]
        ),
        circuit_open_classes=tuple(
            GraphitiFallbackClass(str(item))
            for item in payload["circuit_open_classes"]
        ),
        max_fallback_leaves_per_primary=int(
            payload["max_fallback_leaves_per_primary"]
        ),
        circuit_release_preference=tuple(
            str(item) for item in payload["circuit_release_preference"]
        ),
        canonical_digest=canonical_digest,
    )
    if policy.eligible_outcomes != ("MALFORMED_OUTPUT",):
        raise ValueError("checked Graphiti fallback eligibility changed")
    if policy.max_fallback_leaves_per_primary != 1:
        raise ValueError("checked Graphiti fallback leaf bound changed")
    if policy.circuit_release_preference != (
        "DETERMINISTIC_HEALTH_PROBE",
        "AUTHORISED_OPERATOR_RESET",
    ):
        raise ValueError("checked Graphiti circuit release preference changed")
    return policy


@lru_cache(maxsize=1)
def load_checked_graphiti_fallback_circuit_policy() -> GraphitiFallbackCircuitPolicy:
    return _load_checked_graphiti_fallback_circuit_policy(_POLICY_PATH)


@lru_cache(maxsize=1)
def load_checked_native_graphiti_fallback_circuit_policy() -> GraphitiFallbackCircuitPolicy:
    return _load_checked_graphiti_fallback_circuit_policy(_NATIVE_POLICY_PATH)


def classify_graphiti_fallback(outcome: str) -> GraphitiFallbackDecision:
    """Classify a retained runtime outcome without changing its public spelling."""

    return load_checked_graphiti_fallback_circuit_policy().classify(outcome)


__all__ = [
    "FallbackEligibility",
    "GRAPHITI_FALLBACK_CIRCUIT_SCHEMA_VERSION",
    "GraphitiFallbackCircuitPolicy",
    "GraphitiFallbackClass",
    "GraphitiFallbackDecision",
    "classify_graphiti_fallback",
    "load_checked_graphiti_fallback_circuit_policy",
    "load_checked_native_graphiti_fallback_circuit_policy",
]
