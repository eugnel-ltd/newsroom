"""Operation-local identities do not replace retained accounting validation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from newsroom.control_plane import corpus, native_graphiti
from newsroom.control_plane.model_usage import _SCHEMA
from newsroom.graphiti_adapter.types import GraphitiAdapterOutcome
from newsroom.tests.test_native_graphiti import _native, _open


@pytest.mark.parametrize("count", (8, 64))
def test_accounting_reuses_predispatch_keys_then_rederives_current_identities(
    tmp_path, monkeypatch, count,
):
    processor, connection, calls = _open(
        tmp_path, monkeypatch,
        ingest=lambda *_args, **_values: pytest.fail("terminal work retried"),
    )
    # Real indexed selectors, but no retained liabilities or provider work.
    connection.executescript(_SCHEMA)
    processor._usage = SimpleNamespace()
    monkeypatch.setattr(
        native_graphiti, "graphiti_required_route_holds", lambda *_a, **_k: (),
    )
    history = []
    processor._system.graphiti = SimpleNamespace(attempt_history=lambda *args, **_kw: (
        history.append(args[0]) or SimpleNamespace(
            outcome=GraphitiAdapterOutcome.AMBIGUOUS_EFFECT,
            failure_code="AMBIGUOUS_EFFECT",
        ),
    ))
    first = tuple(
        replace(_native(f"identity-cost-{number}"), body="Retained source. " * 64)
        for number in range(count)
    )
    second = (*first[:-1], replace(first[-1], body=first[-1].body + "Changed."))
    expected = tuple(tuple(unit.ingest_id for unit in units) for units in (first, second))
    assert expected[0][-1] != expected[1][-1]
    identity = corpus.CorpusIngestUnit.ingest_id.fget
    identities = []

    def counted_identity(unit):
        identities.append(unit.item_key)
        return identity(unit)

    monkeypatch.setattr(corpus.CorpusIngestUnit, "ingest_id", property(counted_identity))
    queries = []
    connection.set_trace_callback(queries.append)
    try:
        for number, units in enumerate((first, second)):
            identities.clear()
            queries.clear()
            outcomes = processor.advance(units, cycle_id=f"identity-cost:{number}")
            assert tuple(item.ingest_id for item in outcomes) == expected[number]
            assert all(
                item.state == "GRAPHITI_HOLD"
                and item.reason == "AMBIGUOUS_EFFECT:AMBIGUOUS_EFFECT"
                for item in outcomes
            )
            assert len(identities) == count * 2
            # Both accounting phases still select every unit independently.
            assert sum(
                "INDEXED BY model_usage_native_graphiti_ingest" in query
                for query in queries
            ) == count * 8
            assert len(history) == (number + 1) * count
        assert not calls
        assert connection.execute("SELECT count(*) FROM ledger").fetchone() == (0,)
    finally:
        connection.close()


@pytest.mark.parametrize("mutation", ("body", "authority"))
def test_postdispatch_accounting_rederives_mutated_frozen_input(
    tmp_path, monkeypatch, mutation,
):
    unit = _native("phase-boundary")
    before = unit.ingest_id
    dispatched = []

    def ingest(_connection, **values):
        assert values["units"] == (unit,)
        dispatched.append(before)
        # Model an adversarial external-call boundary, not ordinary dataclass use.
        if mutation == "body":
            object.__setattr__(unit, "body", unit.body + "Changed at dispatch.")
        else:
            object.__setattr__(unit, "authority", replace(
                unit.authority, revision_id="00000000-0000-4000-8000-000000008202",
            ))

    processor, connection, _ = _open(tmp_path, monkeypatch, ingest=ingest)
    connection.executescript(_SCHEMA)
    processor._usage = SimpleNamespace()
    monkeypatch.setattr(
        native_graphiti, "graphiti_required_route_holds", lambda *_a, **_k: (),
    )
    queries = []
    connection.set_trace_callback(queries.append)
    try:
        processor.advance((unit,), cycle_id="changed-at-dispatch")
        after = unit.ingest_id
        assert before != after and dispatched == [before]
        selectors = tuple(
            query for query in queries
            if "INDEXED BY model_usage_native_graphiti_ingest" in query
        )
        assert len(selectors) == 8
        assert all(before in query and after not in query for query in selectors[:4])
        assert all(after in query and before not in query for query in selectors[4:])
    finally:
        connection.close()
