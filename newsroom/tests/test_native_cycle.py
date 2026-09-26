from __future__ import annotations

import sqlite3
from contextlib import nullcontext
from dataclasses import replace

from newsroom.control_plane.native_cycle import (
    advance_native_cycle,
    advance_native_revisions,
)
from newsroom.control_plane.native_discovery import NativeDiscovery
from newsroom.tests.discovery_3d_authority_helpers import (
    exact_admission_request,
    proof,
    seed_check_lineage,
)
from newsroom.tests.test_native_triage import (
    _actor_digest,
    _candidate_collision,
    _no_match_retrieval,
    _shared_system,
)
from newsroom.tests.test_graphiti_operational_readiness import _unit
from newsroom.tests.test_native_discovery import NOW, _current_rights, _seed


class _Retrieval:
    def __init__(self, binding):
        self.binding = binding

    def retrieve(self, lead, *, proof):
        return self.binding


class _Collision:
    def __init__(self, request=None, *, hold_once=False):
        self.value = request
        self.hold_once = hold_once

    def request(self, triage, retrieval, *, proof):
        if self.hold_once:
            self.hold_once = False
            return None
        return self.value


def test_revision_failure_is_visible_without_blocking_unrelated_native_work(
    tmp_path, monkeypatch
) -> None:
    retrieval_authority, binding = _no_match_retrieval(tmp_path)
    monkeypatch.setattr(
        "newsroom.control_plane.cycle._dispatch_rights_decision",
        lambda *args, **kwargs: _current_rights(),
    )
    with sqlite3.connect(":memory:") as proving, _shared_system(
        tmp_path, monkeypatch, retrieval_authority
    ) as system:
        unit = _unit()
        _seed(system, unit)
        controller = NativeDiscovery(
            sources=system.sources,
            checks=system.checks,
            discovery=system.discovery,
            proving=proving,
        )
        outcomes = advance_native_revisions(
            controller,
            system,
            (replace(unit, body="tampered"), unit),
            now=NOW,
            retrieval=_Retrieval(binding),
            collision_requests=_Collision(),
            actor_identity_digest=_actor_digest(),
            proof=proof(),
            owner_stop_check=lambda: None,
            owner_stop_fence=nullcontext,
        )

        assert [item.state for item in outcomes] == [
            "DISCOVERY_HOLD",
            "COLLISION_HOLD",
        ]
        assert outcomes[1].triage is not None
        assert outcomes[1].triage.state == "CANDIDATE_READY"


def test_native_cycle_isolates_hold_then_admits_and_replays_after_restart(
    tmp_path, monkeypatch
) -> None:
    retrieval_authority, binding = _no_match_retrieval(tmp_path)
    with _shared_system(tmp_path, monkeypatch, retrieval_authority) as system:
        seed_check_lineage(system)
        admitted = system.discovery.admit_signal_to_lead(
            exact_admission_request(), proof=proof()
        )
        assert admitted.lead is not None
        status = system.discovery.current_status(
            admitted.lead.request.signal_id, proof=proof()
        )
        held = advance_native_cycle(
            system,
            (status,),
            retrieval=_Retrieval(binding),
            collision_requests=_Collision(),
            actor_identity_digest=_actor_digest(),
            proof=proof(),
            owner_stop_check=lambda: None,
            owner_stop_fence=nullcontext,
        )[0]
        assert held.state == "COLLISION_HOLD"
        assert held.triage is not None
        assert held.triage.hypothesis is not None

    request, _, enforcer = _candidate_collision(tmp_path, held.triage.hypothesis)
    fences = []

    def owner_stop_fence():
        fences.append("entered")
        return nullcontext()

    from newsroom.control_plane import native_cycle

    preparations = []
    original_prepare = native_cycle.advance_native_triage

    def counted_prepare(*args, **kwargs):
        preparations.append(kwargs["work"].version.version_id)
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(native_cycle, "advance_native_triage", counted_prepare)

    with _shared_system(
        tmp_path, monkeypatch, retrieval_authority, collision=enforcer
    ) as reopened:
        outcomes = advance_native_cycle(
            reopened,
            (status, status),
            retrieval=_Retrieval(binding),
            collision_requests=_Collision(request, hold_once=True),
            actor_identity_digest=_actor_digest(),
            proof=proof(),
            owner_stop_check=lambda: None,
            owner_stop_fence=owner_stop_fence,
        )
        assert [item.state for item in outcomes] == [
            "COLLISION_HOLD",
            "CANDIDATE_ADMITTED",
        ]
        candidate = outcomes[1].triage.candidate
        assert candidate is not None
        assert fences == ["entered"]
        assert preparations == [
            held.triage.work.version.version_id,
            held.triage.work.version.version_id,
        ]

    with _shared_system(
        tmp_path, monkeypatch, retrieval_authority, collision=enforcer
    ) as restarted:
        def unbounded_history(*_args, **_kwargs):
            raise AssertionError("native replay used an unbounded history read")

        monkeypatch.setattr(type(restarted.candidates), "versions", unbounded_history)
        monkeypatch.setattr(type(restarted.hypotheses), "current", unbounded_history)
        replay = advance_native_cycle(
            restarted,
            (status,),
            retrieval=_Retrieval(binding),
            collision_requests=_Collision(request),
            actor_identity_digest=_actor_digest(),
            proof=proof(),
            owner_stop_check=lambda: None,
            owner_stop_fence=nullcontext,
        )[0]
        assert replay.state == "CANDIDATE_ADMITTED"
        assert replay.triage is not None
        assert replay.triage.candidate == candidate
