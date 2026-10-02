from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from newsroom.authority import UtcTimestamp
from newsroom.checks import TriggerKind, ObservableTransitionKind
from newsroom.control_plane.native_discovery import NativeDiscovery
from newsroom.control_plane.graphiti_operational_readiness import _source_requests
from newsroom.discovery import GateOutcome, ReasonReference
from newsroom.sources import BaselinePolicy, BaselinePolicyKind, ObservationModel
from newsroom.tests.check_3c_authority_helpers import proof
from newsroom.tests.discovery_3d_authority_helpers import open_discovery_system
from newsroom.tests.test_graphiti_operational_readiness import _unit, _rights, _next_revision

NOW = UtcTimestamp.parse("2026-09-02T12:02:00.000000Z")
LATER = UtcTimestamp.parse("2026-09-02T12:03:00.000000Z")
RIGHTS_ADMISSION_ID = "4971eb89-f97d-46f1-b6ab-496cc315efff"
RIGHTS_BLOB_DIGEST = "sha256:" + "b" * 64
RIGHTS_OBSERVATION_ID = "da0b7059-e6a3-42b3-bf43-5d9632cfc193"
RIGHTS_OBSERVATION_DIGEST = "sha256:" + "e" * 64


def _current_rights():
    return {
        **_rights(),
        "assessment_admission_id": RIGHTS_ADMISSION_ID,
        "assessment_blob_digest": RIGHTS_BLOB_DIGEST,
        "observation_admission_id": RIGHTS_OBSERVATION_ID,
        "observation_blob_digest": RIGHTS_OBSERVATION_DIGEST,
    }


def _seed(system, unit, prior=None, *, baseline=None, observation_model=None):
    requests = _source_requests(unit, _rights(), prior_revision_id=prior)
    if baseline is not None or observation_model is not None:
        version = requests[1]
        requests = (
            requests[0],
            replace(
                version,
                baseline_policy=baseline or version.baseline_policy,
                observation_model=observation_model or version.observation_model,
            ),
            *requests[2:],
        )
    methods = (
        system.sources.register_definition, system.sources.record_definition_version,
        system.sources.register_item, system.sources.record_revision,
        system.sources.record_representation,
    )
    for method, request in zip(methods, requests, strict=True):
        method(request, proof=proof())


def _timeless_baseline(kind: BaselinePolicyKind) -> BaselinePolicy:
    baseline = _source_requests(_unit(), _rights())[1].baseline_policy
    return replace(baseline, kind=kind, freshness_window_seconds=None)


def _controller(system, proving):
    return NativeDiscovery(
        sources=system.sources, checks=system.checks, discovery=system.discovery,
        proving=proving,
    )


def test_retained_revision_enters_native_discovery_and_replays_after_reopen(tmp_path, monkeypatch):
    database = tmp_path / "authority.sqlite3"
    unit = _unit()
    monkeypatch.setattr(
        "newsroom.control_plane.cycle._dispatch_rights_decision",
        lambda *args, **kwargs: _current_rights(),
    )
    with sqlite3.connect(":memory:") as proving:
        with open_discovery_system(database, clock=lambda: NOW) as system:
            _seed(system, unit)
            controller = _controller(system, proving)
            delivered = controller.deliver(unit, now=NOW, proof=proof())
            assert system.checks.request(delivered.outcome.request.request_id, proof=proof()).request.trigger.kind is TriggerKind.DELIVERED_INPUT
            assert delivered.transition.request.kind is ObservableTransitionKind.FIRST_OBSERVED
            assert delivered.outcome.request.completed_at == NOW
            assert system.sources.revision(delivered.transition.request.current_revision_id, proof=proof()).request.observed_at != NOW
            status = controller.admit_lead(delivered, now=NOW, proof=proof())
            assert status.current_gate.request.outcome is GateOutcome.PROMOTED_TO_LEAD
            assert status.lead is not None
            lead_id = status.lead.request.lead_id
        with sqlite3.connect(database) as connection:
            counts = tuple(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in ("ledger_events", "discovery_occurrences", "check_outcomes", "news_leads"))
        with open_discovery_system(database, clock=lambda: LATER) as system:
            controller = _controller(system, proving)
            resumed = controller.deliver(unit, now=LATER, proof=proof())
            assert resumed.outcome.request.completed_at == NOW
            assert controller.admit_lead(resumed, now=LATER, proof=proof()).lead.request.lead_id == lead_id
        with sqlite3.connect(database) as connection:
            assert tuple(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in ("ledger_events", "discovery_occurrences", "check_outcomes", "news_leads")) == counts


def test_missing_current_rights_hold_and_recovery_is_automatic(tmp_path, monkeypatch):
    rights = [None]
    monkeypatch.setattr("newsroom.control_plane.cycle._dispatch_rights_decision", lambda *args, **kwargs: rights[0])
    with sqlite3.connect(":memory:") as proving, open_discovery_system(tmp_path / "authority.sqlite3", clock=lambda: NOW) as system:
        unit = _unit()
        _seed(system, unit)
        controller = _controller(system, proving)
        delivered = controller.deliver(unit, now=NOW, proof=proof())
        held = controller.admit_lead(delivered, now=NOW, proof=proof())
        assert held.current_gate.request.outcome is GateOutcome.OPERATIONAL_HOLD
        assert held.lead is None
        rights[0] = _current_rights()
        ready = controller.admit_lead(delivered, now=LATER, proof=proof())
        assert ready.current_gate.request.decision_ordinal == 2
        assert ready.lead is not None
        rights[0] = None
        held_again = controller.admit_lead(delivered, now=LATER, proof=proof())
        assert held_again.current_gate.request.outcome is GateOutcome.OPERATIONAL_HOLD
        assert held_again.lead is None


def test_tampered_delivered_fields_fail_before_check_writes(tmp_path):
    with sqlite3.connect(":memory:") as proving, open_discovery_system(tmp_path / "authority.sqlite3", clock=lambda: NOW) as system:
        unit = _unit()
        _seed(system, unit)
        with pytest.raises(ValueError, match="exact retained"):
            _controller(system, proving).deliver(replace(unit, body="tampered"), now=NOW, proof=proof())
    with sqlite3.connect(tmp_path / "authority.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM check_requests").fetchone()[0] == 0


def test_changed_revision_retains_predecessor_and_other_items_continue(tmp_path, monkeypatch):
    monkeypatch.setattr("newsroom.control_plane.cycle._dispatch_rights_decision", lambda *args, **kwargs: _current_rights())
    later = UtcTimestamp.parse("2026-09-02T12:40:00.000000Z")
    with sqlite3.connect(":memory:") as proving, open_discovery_system(tmp_path / "authority.sqlite3", clock=lambda: later) as system:
        unit = _unit()
        _seed(system, unit)
        controller = _controller(system, proving)
        first = controller.deliver(unit, now=NOW, proof=proof())
        second = _next_revision(unit)
        _seed(system, second, first.transition.request.current_revision_id)
        changed = controller.deliver(second, now=later, proof=proof())
        assert changed.transition.request.kind is ObservableTransitionKind.REVISED
        assert changed.transition.request.prior_revision_id == first.transition.request.current_revision_id
        other = _unit(item_key="unrelated")
        _seed(system, other)
        fresh = controller.deliver(other, now=later, proof=proof())
        assert controller.admit_lead(fresh, now=later, proof=proof()).lead is not None
        assert controller.admit_lead(changed, now=later, proof=proof()).lead is not None


def test_parser_reobservation_is_suppressed_and_old_delivery_stays_replayable(tmp_path, monkeypatch):
    from newsroom.sources import DiscoveryRepresentationId
    monkeypatch.setattr("newsroom.control_plane.cycle._dispatch_rights_decision", lambda *args, **kwargs: _current_rights())
    with sqlite3.connect(":memory:") as proving, open_discovery_system(tmp_path / "authority.sqlite3", clock=lambda: NOW) as system:
        unit = _unit()
        _seed(system, unit)
        controller = _controller(system, proving)
        first = controller.deliver(unit, now=NOW, proof=proof())
        controller.admit_lead(first, now=NOW, proof=proof())
        request = _source_requests(unit, _rights())[-1]
        reparsed = replace(
            request, representation_id=DiscoveryRepresentationId.new(),
            parser_version="retained-parser-v2", idempotency_key="reparsed-representation",
        )
        system.sources.record_representation(reparsed, proof=proof())
        new_unit = replace(unit, authority=replace(unit.authority, representation_id=str(reparsed.representation_id)))
        second = controller.deliver(new_unit, now=LATER, proof=proof())
        assert second.transition.request.kind is ObservableTransitionKind.REOBSERVED
        assert controller.admit_lead(second, now=LATER, proof=proof()).current_gate.request.outcome is GateOutcome.SUPPRESSED_NON_CHANGE
        assert controller.deliver(unit, now=LATER, proof=proof()).outcome.request == first.outcome.request
    with sqlite3.connect(tmp_path / "authority.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM news_leads").fetchone()[0] == 1


def test_delivered_history_does_not_become_fresh_from_native_check_time(tmp_path, monkeypatch):
    monkeypatch.setattr("newsroom.control_plane.cycle._dispatch_rights_decision", lambda *args, **kwargs: _current_rights())
    old = UtcTimestamp.parse("2026-09-12T12:00:00.000000Z")
    with sqlite3.connect(":memory:") as proving, open_discovery_system(tmp_path / "authority.sqlite3", clock=lambda: old) as system:
        unit = _unit()
        _seed(system, unit)
        controller = _controller(system, proving)
        delivered = controller.deliver(unit, now=old, proof=proof())
        status = controller.admit_lead(delivered, now=old, proof=proof())
        assert status.lead is None
        assert status.current_gate.request.basis.time_validity.value == "STALE"


@pytest.mark.parametrize(
    ("observation_model", "baseline_kind"),
    (
        (ObservationModel.MUTABLE_ITEM, BaselinePolicyKind.MAINTAINED_DOCUMENT),
        (
            ObservationModel.COMPLETE_CURRENT_STATE,
            BaselinePolicyKind.COMPLETE_STATE_FIRST_OBSERVED_ACTIVE,
        ),
    ),
)
def test_current_retained_timeless_source_is_current_until_a_successor_exists(
    tmp_path, monkeypatch, observation_model, baseline_kind
):
    monkeypatch.setattr(
        "newsroom.control_plane.cycle._dispatch_rights_decision",
        lambda *args, **kwargs: _current_rights(),
    )
    much_later = UtcTimestamp.parse("2027-09-02T12:02:00.000000Z")
    baseline = _timeless_baseline(baseline_kind)
    clock = [NOW]
    with sqlite3.connect(":memory:") as proving, open_discovery_system(
        tmp_path / "authority.sqlite3", clock=lambda: clock[0]
    ) as system:
        first_unit = _unit()
        _seed(
            system,
            first_unit,
            baseline=baseline,
            observation_model=observation_model,
        )
        controller = _controller(system, proving)
        first = controller.deliver(first_unit, now=NOW, proof=proof())
        clock[0] = much_later
        current = controller.admit_lead(first, now=much_later, proof=proof())
        assert current.lead is not None
        assert current.current_gate.request.basis.time_validity.value == "CURRENT"

        second_unit = _next_revision(first_unit)
        _seed(
            system,
            second_unit,
            first.transition.request.current_revision_id,
            baseline=baseline,
            observation_model=observation_model,
        )
        stale = controller.admit_lead(first, now=much_later, proof=proof())
        assert stale.lead is None
        assert stale.current_gate.request.basis.time_validity.value == "STALE"
        second = controller.deliver(second_unit, now=much_later, proof=proof())
        assert controller.admit_lead(
            second, now=much_later, proof=proof()
        ).lead is not None


def test_future_retained_revision_is_not_delivered_as_current(tmp_path):
    before_observation = UtcTimestamp.parse("2026-09-02T11:59:00.000000Z")
    with sqlite3.connect(":memory:") as proving, open_discovery_system(
        tmp_path / "authority.sqlite3", clock=lambda: NOW
    ) as system:
        unit = _unit()
        _seed(system, unit)
        with pytest.raises(ValueError, match="exact retained"):
            _controller(system, proving).deliver(
                unit, now=before_observation, proof=proof()
            )


def test_native_rights_lookup_does_not_mint_or_consume_beta_fixture_packets(tmp_path, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("legacy fixture rights route was invoked")
    monkeypatch.setattr("newsroom.control_plane.cycle._dispatch_rights_decision", forbid)
    with sqlite3.connect(":memory:") as proving, open_discovery_system(tmp_path / "authority.sqlite3", clock=lambda: NOW) as system:
        unit = _unit()
        _seed(system, unit)
        calls = []
        rights_snapshot = [_current_rights()]
        def current_rights(source_id, locator, now):
            calls.append((source_id, locator, now))
            return rights_snapshot[0]
        controller = NativeDiscovery(sources=system.sources, checks=system.checks,
                                     discovery=system.discovery, proving=proving,
                                     rights_for=current_rights)
        status = controller.admit_lead(controller.deliver(unit, now=NOW, proof=proof()), now=NOW, proof=proof())
        assert status.lead is not None
        assert calls == [(unit.source_id, unit.source_definition_url, NOW)]
        reason = status.current_gate.request.supporting_reasons[0]
        assert reason.references == (
            ReasonReference(
                "RIGHTS_ASSESSMENT", RIGHTS_ADMISSION_ID, RIGHTS_BLOB_DIGEST
            ),
            ReasonReference(
                "RIGHTS_OBSERVATION", RIGHTS_OBSERVATION_ID, RIGHTS_OBSERVATION_DIGEST
            ),
        )
        assert (
            status.current_gate.request.rights_decision_id
            == system.sources.version_details(
                status.current_gate.request.evaluated_definition_version_id,
                proof=proof(),
            ).request.rights.rights_decision_id
        )
        refreshed_admission = "121d3431-fbdd-49c9-aa69-67a59024da56"
        refreshed_digest = "sha256:" + "c" * 64
        # A fresh observation of identical rights does not change the Gate.
        # Its original provenance remains immutable; the new observation is
        # independently retained by the rights service.
        rights_snapshot[0] = {
            **rights_snapshot[0],
            "observation_admission_id": refreshed_admission,
            "observation_blob_digest": refreshed_digest,
        }
        refreshed = controller.admit_lead(
            controller.deliver(unit, now=LATER, proof=proof()),
            now=LATER,
            proof=proof(),
        )
        assert refreshed.current_gate == status.current_gate
        assert calls[-1] == (unit.source_id, unit.source_definition_url, LATER)
        assert refreshed.current_disposition is not None
        assert (
            refreshed.current_disposition.request.gate_decision_id
            == refreshed.current_gate.request.decision_id
        )
        assert refreshed.current_disposition == status.current_disposition
        assert refreshed.current_gate.request.supporting_reasons[0].references == (
            ReasonReference(
                "RIGHTS_ASSESSMENT", RIGHTS_ADMISSION_ID, RIGHTS_BLOB_DIGEST
            ),
            ReasonReference(
                "RIGHTS_OBSERVATION", RIGHTS_OBSERVATION_ID, RIGHTS_OBSERVATION_DIGEST
            ),
        )
        before_replay = system.discovery.dispositions(
            refreshed.lead.request.lead_id, limit=10, proof=proof()
        )
        replay = controller.admit_lead(
            controller.deliver(unit, now=LATER, proof=proof()),
            now=LATER,
            proof=proof(),
        )
        assert replay.current_disposition == refreshed.current_disposition
        assert system.discovery.dispositions(
            refreshed.lead.request.lead_id, limit=10, proof=proof()
        ) == before_replay
        rights_snapshot[0] = None
        held = controller.admit_lead(
            controller.deliver(unit, now=LATER, proof=proof()),
            now=LATER,
            proof=proof(),
        )
        assert held.current_gate.request.outcome is GateOutcome.OPERATIONAL_HOLD
        assert system.discovery.dispositions(
            refreshed.lead.request.lead_id, limit=10, proof=proof()
        ) == before_replay
        assert proving.execute("SELECT name FROM sqlite_master").fetchall() == []


def test_reopen_repairs_a_current_gate_missing_its_queued_disposition(tmp_path):
    database = tmp_path / "authority.sqlite3"
    unit = _unit()
    rights = [_current_rights()]

    class InterruptDisposition:
        def __init__(self, discovery):
            self._discovery = discovery

        def __getattr__(self, name):
            return getattr(self._discovery, name)

        def record_lead_disposition(self, request, *, proof):
            raise RuntimeError("crash after Gate authority")

    with sqlite3.connect(":memory:") as proving:
        with open_discovery_system(database, clock=lambda: NOW) as system:
            _seed(system, unit)
            controller = NativeDiscovery(
                sources=system.sources,
                checks=system.checks,
                discovery=system.discovery,
                proving=proving,
                rights_for=lambda *_: rights[0],
            )
            delivered = controller.deliver(unit, now=NOW, proof=proof())
            initial = controller.admit_lead(delivered, now=NOW, proof=proof())
            assert initial.current_disposition is not None
            rights[0] = {
                **rights[0],
                "assessment_blob_digest": "sha256:" + "d" * 64,
                "observation_admission_id": (
                    "121d3431-fbdd-49c9-aa69-67a59024da56"
                ),
                "observation_blob_digest": "sha256:" + "c" * 64,
            }
            interrupted = NativeDiscovery(
                sources=system.sources,
                checks=system.checks,
                discovery=InterruptDisposition(system.discovery),
                proving=proving,
                rights_for=lambda *_: rights[0],
            )
            with pytest.raises(RuntimeError, match="crash after Gate"):
                interrupted.admit_lead(delivered, now=LATER, proof=proof())
            prefix = system.discovery.current_status(
                initial.signal.request.signal_id, proof=proof()
            )
            assert prefix.current_gate.request.decision_ordinal == 2
            assert prefix.current_disposition is None
            assert system.discovery.latest_disposition(
                initial.lead.request.lead_id, proof=proof()
            ) == initial.current_disposition

        with open_discovery_system(database, clock=lambda: LATER) as system:
            controller = NativeDiscovery(
                sources=system.sources,
                checks=system.checks,
                discovery=system.discovery,
                proving=proving,
                rights_for=lambda *_: rights[0],
            )
            delivered = controller.deliver(unit, now=LATER, proof=proof())
            repaired = controller.admit_lead(
                delivered, now=LATER, proof=proof()
            )
            assert repaired.current_disposition is not None
            assert repaired.current_disposition.request.decision_ordinal == 2
            assert (
                repaired.current_disposition.request.previous_decision_id
                == initial.current_disposition.request.decision_id
            )
            before_replay = system.discovery.dispositions(
                repaired.lead.request.lead_id, limit=10, proof=proof()
            )
            assert controller.admit_lead(
                delivered, now=LATER, proof=proof()
            ).current_disposition == repaired.current_disposition
            assert system.discovery.dispositions(
                repaired.lead.request.lead_id, limit=10, proof=proof()
            ) == before_replay


@pytest.mark.parametrize("fresh", (True, False))
def test_fresh_rights_observations_do_not_repeat_gate_commands(tmp_path, fresh):
    database = tmp_path / "authority.sqlite3"
    clock = [NOW if fresh else UtcTimestamp.parse("2026-09-12T12:00:00.000000Z")]
    rights = _current_rights()
    calls = []

    def current_rights(*args):
        calls.append(args)
        return rights

    def command_count():
        with sqlite3.connect(database) as connection:
            return connection.execute("SELECT count(*) FROM authority_commands").fetchone()[0]

    with sqlite3.connect(":memory:") as proving, open_discovery_system(
        database, clock=lambda: clock[0]
    ) as system:
        unit = _unit()
        _seed(system, unit)
        controller = NativeDiscovery(
            sources=system.sources, checks=system.checks, discovery=system.discovery,
            proving=proving, rights_for=current_rights,
        )
        delivered = controller.deliver(unit, now=clock[0], proof=proof())
        initial = controller.admit_lead(delivered, now=clock[0], proof=proof())
        before = command_count()
        rights.update(observation_admission_id="121d3431-fbdd-49c9-aa69-67a59024da56",
                      observation_blob_digest="sha256:" + "c" * 64)
        for _ in range(2):
            current = controller.admit_lead(delivered, now=clock[0], proof=proof())
            assert current.current_gate == initial.current_gate
            assert current.current_disposition == initial.current_disposition
        assert len(calls) == 3
        assert command_count() == before

        # Changed terms/policy/expiry change the retained assessment, so reuse ends.
        rights["assessment_blob_digest"] = "sha256:" + "d" * 64
        changed = controller.admit_lead(delivered, now=clock[0], proof=proof())
        assert changed.current_gate.request.decision_ordinal == 2
        assert command_count() > before
        assert changed.current_gate.request.supporting_reasons[0].references[1] == (
            ReasonReference("RIGHTS_OBSERVATION", rights["observation_admission_id"],
                            rights["observation_blob_digest"])
        )


        # No current observation must still hold, even if the assessment exists.
        del rights["observation_admission_id"]
        held = controller.admit_lead(delivered, now=clock[0], proof=proof())
        assert held.current_gate.request.decision_ordinal == 3
        assert held.current_gate.request.outcome is GateOutcome.OPERATIONAL_HOLD
        before = command_count()
        assert controller.admit_lead(
            delivered, now=clock[0], proof=proof()
        ).current_gate == held.current_gate
        assert command_count() == before


def test_gate_reuse_rechecks_freshness_and_compares_effective_policy(tmp_path, monkeypatch):
    import newsroom.control_plane.native_discovery as native

    clock = [NOW]
    with sqlite3.connect(":memory:") as proving, open_discovery_system(
        tmp_path / "authority.sqlite3", clock=lambda: clock[0]
    ) as system:
        unit = _unit()
        _seed(system, unit)
        controller = NativeDiscovery(
            sources=system.sources, checks=system.checks, discovery=system.discovery,
            proving=proving, rights_for=lambda *_: _current_rights(),
        )
        delivered = controller.deliver(unit, now=NOW, proof=proof())
        initial = controller.admit_lead(delivered, now=NOW, proof=proof())
        original_policy = native.policy
        monkeypatch.setattr(native, "policy", lambda name: (
            native.VersionedPolicyRef("hermes-delivered-gate", "v2")
            if name == "gate" else original_policy(name)
        ))
        changed = controller.admit_lead(delivered, now=NOW, proof=proof())
        assert changed.current_gate.request.decision_ordinal == 2
        assert changed.current_gate.request.basis == initial.current_gate.request.basis
        assert changed.current_gate.request.gate_policy == native.policy("gate")

        clock[0] = UtcTimestamp.parse("2026-09-12T12:00:00.000000Z")
        stale = controller.admit_lead(delivered, now=clock[0], proof=proof())
        assert stale.current_gate.request.decision_ordinal == 3
        assert stale.current_gate.request.basis.time_validity.value == "STALE"
        assert stale.current_gate.request.outcome is GateOutcome.OPERATIONAL_HOLD


def test_backlog_delivery_uses_observed_predecessor_not_canonical_ingestion(tmp_path):
    from newsroom.sources import SourceRevisionId

    at = UtcTimestamp.parse("2026-09-02T13:00:00.000000Z")
    later = UtcTimestamp.parse("2026-09-02T13:01:00.000000Z")
    with sqlite3.connect(":memory:") as proving, open_discovery_system(
        tmp_path / "authority.sqlite3", clock=lambda: later
    ) as system:
        older = _unit()
        newer = _next_revision(older)
        _seed(system, older)
        _seed(system, newer, SourceRevisionId.parse(older.authority.revision_id))
        controller = _controller(system, proving)
        first = controller.deliver(newer, now=at, proof=proof())
        assert first.transition.request.kind is ObservableTransitionKind.FIRST_OBSERVED
        historical = controller.deliver(older, now=later, proof=proof())
        assert historical.transition.request.kind is ObservableTransitionKind.REVISED
        assert historical.transition.request.prior_revision_id == first.transition.request.current_revision_id
        assert system.sources.latest_revision(first.transition.request.item_id, proof=proof()).request.revision_id == first.transition.request.current_revision_id


def _reparsed_unit(system, unit, parser_version):
    from newsroom.sources import DiscoveryRepresentationId

    request = _source_requests(unit, _rights())[-1]
    representation = replace(
        request, representation_id=DiscoveryRepresentationId.new(),
        parser_version=parser_version, idempotency_key=f"reparse:{parser_version}",
    )
    system.sources.record_representation(representation, proof=proof())
    return replace(
        unit, authority=replace(unit.authority, representation_id=str(representation.representation_id)),
    )


def test_seen_revision_after_different_observed_state_is_changed_not_reobserved(tmp_path):
    from newsroom.checks import CheckOutcomeKind

    later = UtcTimestamp.parse("2026-09-02T13:00:00.000000Z")
    with sqlite3.connect(":memory:") as proving, open_discovery_system(
        tmp_path / "authority.sqlite3", clock=lambda: later
    ) as system:
        older = _unit()
        _seed(system, older)
        controller = _controller(system, proving)
        first = controller.deliver(older, now=NOW, proof=proof())
        newer = _next_revision(older)
        _seed(system, newer, first.transition.request.current_revision_id)
        changed = controller.deliver(newer, now=later, proof=proof())
        historical = controller.deliver(_reparsed_unit(system, older, "switch-back"), now=later, proof=proof())
        assert historical.outcome.request.kind is CheckOutcomeKind.SUCCESS_CHANGED
        assert historical.transition.request.kind is ObservableTransitionKind.REVISED
        assert historical.transition.request.prior_revision_id == changed.transition.request.current_revision_id
        assert historical.transition.request.current_revision_id == first.transition.request.current_revision_id


def test_partial_delivery_repaired_as_of_original_outcome_and_replays_after_reopen(tmp_path, monkeypatch):
    from newsroom.authority._check_facade import GovernedChecks
    from newsroom.checks import CheckVersionConflict

    database = tmp_path / "authority.sqlite3"
    at = UtcTimestamp.parse("2026-09-02T13:00:00.000000Z")
    later = UtcTimestamp.parse("2026-09-02T13:01:00.000000Z")
    with sqlite3.connect(":memory:") as proving:
        with open_discovery_system(database, clock=lambda: later) as system:
            older = _unit()
            _seed(system, older)
            controller = _controller(system, proving)
            first = controller.deliver(older, now=NOW, proof=proof())
            newer = _next_revision(older)
            _seed(system, newer, first.transition.request.current_revision_id)
            def interrupt(*args, **kwargs):
                raise CheckVersionConflict("fixture transition interruption")
            with monkeypatch.context() as patch:
                patch.setattr(GovernedChecks, "record_transition", interrupt)
                with pytest.raises(CheckVersionConflict, match="fixture transition interruption"):
                    controller.deliver(newer, now=at, proof=proof())
            later_delivery = controller.deliver(_reparsed_unit(system, older, "after-interruption"), now=later, proof=proof())
            assert later_delivery.transition.request.prior_revision_id != first.transition.request.current_revision_id
        with sqlite3.connect(database) as connection:
            tables = ("check_requests", "check_attempts", "check_outcomes", "discovery_occurrences", "observable_transitions")
            counts = tuple(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables)
            original_outcomes = connection.execute("SELECT outcome_id,canonical_bytes FROM check_outcomes ORDER BY outcome_id").fetchall()
        with open_discovery_system(database, clock=lambda: later) as system:
            controller = _controller(system, proving)
            repaired = controller.deliver(newer, now=later, proof=proof())
            assert repaired.outcome.request.completed_at == at
            assert repaired.transition.request.prior_revision_id == first.transition.request.current_revision_id
            assert repaired.transition.request.kind is ObservableTransitionKind.REVISED
            assert controller.deliver(newer, now=later, proof=proof()).transition.request == repaired.transition.request
            assert controller.deliver(older, now=later, proof=proof()).transition.request == first.transition.request
            assert system.sources.latest_revision(first.transition.request.item_id, proof=proof()).request.revision_id == repaired.transition.request.current_revision_id
        with sqlite3.connect(database) as connection:
            assert tuple(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables) == (*counts[:-1], counts[-1] + 1)
            assert connection.execute("SELECT outcome_id,canonical_bytes FROM check_outcomes ORDER BY outcome_id").fetchall() == original_outcomes


def test_equal_completion_time_uses_ledger_order_and_replays_earlier_boundary(tmp_path):
    later = UtcTimestamp.parse("2026-09-02T13:00:00.000000Z")
    with sqlite3.connect(":memory:") as proving, open_discovery_system(
        tmp_path / "authority.sqlite3", clock=lambda: later
    ) as system:
        older = _unit()
        _seed(system, older)
        controller = _controller(system, proving)
        first = controller.deliver(older, now=later, proof=proof())
        newer = _next_revision(older)
        _seed(system, newer, first.transition.request.current_revision_id)
        second = controller.deliver(newer, now=later, proof=proof())
        assert second.transition.request.prior_revision_id == first.transition.request.current_revision_id
        assert system.checks.observed_prior_revision(
            first.transition.request.item_id, request_id=first.outcome.request.request_id,
            outcome_id=first.outcome.request.outcome_id, completed_at=later, proof=proof(),
        ) is None
        assert controller.deliver(older, now=later, proof=proof()).transition.request == first.transition.request


def test_unresolved_earlier_outcome_stops_before_new_outcome_or_occurrence(tmp_path, monkeypatch):
    from newsroom.authority._source_registry_system import GovernedSources
    from newsroom.checks import CheckStateError
    from newsroom.sources import SourceRevisionId

    database = tmp_path / "authority.sqlite3"
    later = UtcTimestamp.parse("2026-09-02T13:00:00.000000Z")
    with sqlite3.connect(":memory:") as proving, open_discovery_system(database, clock=lambda: later) as system:
        older = _unit()
        _seed(system, older)
        controller = _controller(system, proving)
        def interrupt(*args, **kwargs):
            raise RuntimeError("fixture occurrence interruption")
        with monkeypatch.context() as patch:
            patch.setattr(GovernedSources, "record_occurrence", interrupt)
            with pytest.raises(RuntimeError, match="fixture occurrence interruption"):
                controller.deliver(older, now=NOW, proof=proof())
        newer = _next_revision(older)
        _seed(system, newer, SourceRevisionId.parse(older.authority.revision_id))
        with pytest.raises(CheckStateError, match="prior observed Check Outcome lacks"):
            controller.deliver(newer, now=later, proof=proof())
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT COUNT(*) FROM check_outcomes").fetchone()[0] == 1
            assert connection.execute("SELECT COUNT(*) FROM discovery_occurrences").fetchone()[0] == 0


def test_inconsistent_retained_unchanged_outcome_is_not_rewritten_on_recovery(tmp_path, monkeypatch):
    from newsroom.authority._check_facade import GovernedChecks
    from newsroom.checks import CheckContractError, CheckVersionConflict

    database = tmp_path / "authority.sqlite3"
    later = UtcTimestamp.parse("2026-09-02T13:00:00.000000Z")
    with sqlite3.connect(":memory:") as proving, open_discovery_system(database, clock=lambda: later) as system:
        older = _unit()
        _seed(system, older)
        controller = _controller(system, proving)
        first = controller.deliver(older, now=NOW, proof=proof())
        newer = _next_revision(older)
        _seed(system, newer, first.transition.request.current_revision_id)
        controller.deliver(newer, now=later, proof=proof())
        historical = _reparsed_unit(system, older, "legacy-unchanged")
        older_record = system.sources.revision(first.transition.request.current_revision_id, proof=proof())
        # Reproduce the old producer's "ever observed" classification while
        # leaving the real transition guard in place.
        with monkeypatch.context() as patch:
            patch.setattr(GovernedChecks, "observed_prior_revision", lambda *args, **kwargs: older_record.request.revision_id)
            with pytest.raises(CheckVersionConflict, match="latest observed source state"):
                controller.deliver(historical, now=later, proof=proof())
        with sqlite3.connect(database) as connection:
            original_outcomes = connection.execute("SELECT outcome_id,canonical_bytes FROM check_outcomes ORDER BY outcome_id").fetchall()
            occurrences = connection.execute("SELECT COUNT(*) FROM discovery_occurrences").fetchone()[0]
        with pytest.raises(CheckContractError, match="same Source Revision"):
            controller.deliver(historical, now=later, proof=proof())
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT outcome_id,canonical_bytes FROM check_outcomes ORDER BY outcome_id").fetchall() == original_outcomes
            assert connection.execute("SELECT COUNT(*) FROM discovery_occurrences").fetchone()[0] == occurrences
