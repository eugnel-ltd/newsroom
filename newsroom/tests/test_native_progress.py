from dataclasses import replace

import pytest

from newsroom.control_plane.native_progress import NativeRevisionJournal
from newsroom.control_plane.store import connect
from newsroom.tests.test_native_graphiti import _native


def test_native_journal_reopens_progress_without_repeating_landing_or_state(tmp_path):
    path = str(tmp_path / "private.sqlite3")
    unit = _native()
    connection = connect(path)
    journal = NativeRevisionJournal(connection)
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="GRAPHITI_HOLD", facts={"reason": "RIGHTS_HOLD"})
    connection.close()
    connection = connect(path)
    journal = NativeRevisionJournal(connection)
    assert journal.units[unit.revision_id] == (unit,)
    journal.land((unit,))
    journal.advance(unit.revision_id, stage="GRAPHITI_HOLD", facts={"reason": "RIGHTS_HOLD"})
    assert connection.execute("SELECT count(*) FROM ledger").fetchone()[0] == 2
    journal.advance(unit.revision_id, stage="GRAPHITI_COMPLETE", facts={"receipt": unit.digest})
    assert journal.current(unit.revision_id)["ordinal"] == 2
    connection.close()


def test_native_journal_rejects_missing_chunk_without_writing(tmp_path):
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    with pytest.raises(ValueError, match="chunk coverage"):
        journal.land((replace(_native(), chunk_count=2),))
    assert connection.execute("SELECT count(*) FROM ledger").fetchone()[0] == 0
    connection.close()


def test_native_journal_retains_old_binding_on_unchanged_reobservation(tmp_path):
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    unit = _native()
    journal.land((unit,))
    journal.land((replace(unit, observation_digest="sha256:" + "e" * 64),))
    assert journal.units[unit.revision_id] == (unit,)
    assert connection.execute("SELECT count(*) FROM ledger").fetchone()[0] == 1
    connection.close()


@pytest.mark.parametrize("field", ["headline", "body", "canonical_url"])
@pytest.mark.parametrize("ordinal", [1, 3])
def test_native_journal_rechecks_changed_chunk_content_before_reobservation(
    tmp_path, field, ordinal,
):
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    unit = replace(_native(), body="Complete retained text. " * 750, chunk_count=3)
    units = tuple(replace(unit, chunk_ordinal=index) for index in range(1, 4))
    journal.land(units)
    changed = tuple(
        replace(item, **{field: getattr(item, field) + " changed"})
        if item.chunk_ordinal == ordinal else item
        for item in units
    )
    with pytest.raises(ValueError, match="chunk coverage"):
        journal.land(changed)
    assert journal.units[unit.revision_id] == units
    assert connection.execute("SELECT count(*) FROM ledger").fetchone()[0] == 1
    assert NativeRevisionJournal(connection).units[unit.revision_id] == units
    connection.close()


def test_native_journal_reopen_retains_one_body_per_multichunk_revision(tmp_path):
    connection = connect(str(tmp_path / "private.sqlite3"))
    unit = replace(_native(), body="A complete retained paragraph. " * 1000, chunk_count=3)
    units = tuple(replace(unit, chunk_ordinal=ordinal) for ordinal in range(1, 4))
    NativeRevisionJournal(connection).land(units)
    reopened = NativeRevisionJournal(connection)
    retained = reopened.units[unit.revision_id]
    assert retained == units
    # Every chunk owns its identity, not a second copy of the full source body.
    assert len({id(item.body) for item in retained}) == 1
    connection.close()


def test_native_journal_rejects_tampered_payload_on_reopen(tmp_path):
    connection = connect(str(tmp_path / "private.sqlite3"))
    NativeRevisionJournal(connection).land((_native(),))
    connection.execute("UPDATE ledger SET payload_json='{}'")
    connection.commit()
    with pytest.raises(ValueError, match="payload differs"):
        NativeRevisionJournal(connection)
    connection.close()


def test_native_journal_retains_page_references_and_per_item_holds_across_poll(tmp_path):
    from newsroom.control_plane.native_source_intake import NativeSourceDisposition
    path = str(tmp_path / "private.sqlite3")
    connection = connect(path)
    journal = NativeRevisionJournal(connection)
    reference = ("https://www.gov.uk/api/content/item", "sha256:" + "d" * 64, "admission", "access")
    journal.sources((NativeSourceDisposition(
        "UK-01", "HOLD", "SOURCE_ITEMS_HELD", observations=(reference,),
        item_holds=(("https://www.gov.uk/other", "UNSUPPORTED_TYPE"),),
    ),))
    assert journal.portfolio[0]["item_holds"] == [["https://www.gov.uk/other", "UNSUPPORTED_TYPE"]]
    journal.sources((NativeSourceDisposition("UK-01", "READY", "UNCHANGED"),))
    connection.close()
    connection = connect(path)
    reopened = NativeRevisionJournal(connection)
    assert reopened.observations[reference[1]] == reference
    connection.close()


def _retrieval_facts():
    return {
        "retrieval_binding": {"request": {"nodes": ["Retained graph record. " * 200]}},
        "retrieval_rights_inventory": [{"source_id": "UK-01", "rights": "retained"}],
        "retrieval_embeddings": {"passage": {"state": "STARTED", "cycle_id": "cycle"}},
        "reason": "EVIDENCE_NOT_READY",
    }


def test_unchanged_retrieval_pair_references_previous_record_but_returns_full_facts(tmp_path):
    import json

    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    revision = _native().revision_id
    journal.land((_native(),))
    facts = _retrieval_facts()
    journal.advance(revision, stage="RETRIEVAL_COMPLETE", facts=facts)
    previous = connection.execute("SELECT seq,payload_digest FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
    result = journal.advance(revision, stage="EVIDENCE_HOLD", facts=facts)
    raw = json.loads(connection.execute("SELECT payload_json FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()[0])
    assert raw["retrieval_facts_ref"] == {"seq": previous[0], "payload_digest": previous[1], "ordinal": 1}
    assert "retrieval_binding" not in raw["facts"]
    assert "retrieval_rights_inventory" not in raw["facts"]
    assert raw["facts"]["retrieval_embeddings"] == facts["retrieval_embeddings"]
    assert result["facts"] == facts
    assert result == NativeRevisionJournal(connection).current(revision)
    connection.close()


def test_mutated_previous_pair_is_not_mistaken_for_committed_content(tmp_path):
    import json

    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    revision = _native().revision_id
    journal.land((_native(),))
    retained = journal.advance(revision, stage="RETRIEVAL_COMPLETE", facts=_retrieval_facts())
    retained["facts"]["retrieval_binding"]["request"]["nodes"].append("new record")
    result = journal.advance(revision, stage="RETRIEVAL_COMPLETE", facts=retained["facts"])
    assert result["ordinal"] == 2
    raw = json.loads(connection.execute("SELECT payload_json FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()[0])
    assert "retrieval_facts_ref" not in raw
    assert result == NativeRevisionJournal(connection).current(revision)
    connection.close()


def test_failed_append_rolls_back_without_advancing_reference(tmp_path, monkeypatch):
    import newsroom.control_plane.native_progress as progress

    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    revision = _native().revision_id
    journal.land((_native(),))
    before = journal.advance(revision, stage="RETRIEVAL_COMPLETE", facts=_retrieval_facts())
    original = progress.append_ledger
    def fail_after_insert(*args):
        original(*args)
        raise RuntimeError("injected append failure")
    monkeypatch.setattr(progress, "append_ledger", fail_after_insert)
    with pytest.raises(RuntimeError, match="injected"):
        journal.advance(revision, stage="EVIDENCE_HOLD", facts=_retrieval_facts())
    assert not connection.in_transaction
    assert connection.execute("SELECT count(*) FROM ledger").fetchone()[0] == 2
    assert journal.current(revision) == before
    assert NativeRevisionJournal(connection).current(revision) == before
    connection.close()


@pytest.mark.parametrize("facts", [[], [["reason", "HOLD"]], None, "facts"])
def test_progress_requires_object_facts_before_append(tmp_path, facts):
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((_native(),))
    with pytest.raises(ValueError, match="facts"):
        journal.advance(_native().revision_id, stage="EVIDENCE_HOLD", facts=facts)
    assert connection.execute("SELECT count(*) FROM ledger").fetchone()[0] == 1
    assert tuple(journal.iter_summaries()) == ()
    connection.close()


@pytest.mark.parametrize("facts", [[], [["reason", "HOLD"]], None, "facts"])
def test_replay_rejects_non_object_facts(tmp_path, facts):
    from newsroom.control_plane.native_progress import STATE
    from newsroom.control_plane.store import append_ledger

    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    journal.land((_native(),))
    append_ledger(connection, STATE, {
        "revision_id": _native().revision_id, "ordinal": 1,
        "stage": "EVIDENCE_HOLD", "facts": facts,
    })
    connection.commit()
    with pytest.raises(ValueError, match="facts"):
        NativeRevisionJournal(connection)
    connection.close()


@pytest.mark.parametrize("case", [
    "absent", "forward", "cross_revision", "digest", "ordinal", "older",
    "no_prior_pair", "first", "null", "list", "extra", "bool_seq",
    "bool_ordinal", "inline_binding", "inline_rights",
])
def test_replay_rejects_invalid_retrieval_reference(tmp_path, case):
    from newsroom.control_plane.native_progress import STATE
    from newsroom.control_plane.store import append_ledger

    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    unit, other = _native(), _native("other")
    journal.land((unit,))
    journal.land((other,))
    journal.advance(other.revision_id, stage="RETRIEVAL_COMPLETE", facts=_retrieval_facts())
    other_record = journal._records[other.revision_id].reference()
    if case != "first":
        journal.advance(unit.revision_id, stage="RETRIEVAL_COMPLETE", facts=(
            {} if case == "no_prior_pair" else _retrieval_facts()
        ))
    reference = (other_record if case == "first"
                 else journal._records[unit.revision_id].reference())
    ordinal = 1 if case == "first" else 2
    facts = {"reason": "EVIDENCE_NOT_READY"}
    if case == "absent":
        reference["seq"] = 0
    elif case == "forward":
        reference["seq"] += 1
    elif case == "cross_revision":
        reference = other_record
    elif case == "digest":
        reference["payload_digest"] = "sha256:" + "0" * 64
    elif case == "ordinal":
        reference["ordinal"] += 1
    elif case == "older":
        journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts=_retrieval_facts())
        ordinal = 3
    elif case == "null":
        reference = None
    elif case == "list":
        reference = list(reference.items())
    elif case == "extra":
        reference["extra"] = "not part of the reference"
    elif case == "bool_seq":
        reference["seq"] = True
    elif case == "bool_ordinal":
        reference["ordinal"] = True
    elif case == "inline_binding":
        facts["retrieval_binding"] = {}
    elif case == "inline_rights":
        facts["retrieval_rights_inventory"] = []
    append_ledger(connection, STATE, {
        "revision_id": unit.revision_id, "ordinal": ordinal,
        "stage": "EVIDENCE_HOLD", "facts": facts, "retrieval_facts_ref": reference,
    })
    connection.commit()
    before = connection.total_changes
    with pytest.raises(ValueError, match="retrieval facts reference"):
        NativeRevisionJournal(connection)
    assert connection.total_changes == before
    connection.close()


@pytest.mark.parametrize("field", ["retrieval_binding", "retrieval_rights_inventory"])
@pytest.mark.parametrize("remove", [False, True])
def test_changed_or_absent_pair_is_retained_inline(tmp_path, field, remove):
    import json

    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    revision = _native().revision_id
    journal.land((_native(),))
    journal.advance(revision, stage="RETRIEVAL_COMPLETE", facts=_retrieval_facts())
    facts = _retrieval_facts()
    if remove:
        del facts[field]
    else:
        facts[field] = {"changed": True}
    result = journal.advance(revision, stage="EVIDENCE_HOLD", facts=facts)
    raw = json.loads(connection.execute(
        "SELECT payload_json FROM ledger ORDER BY seq DESC LIMIT 1"
    ).fetchone()[0])
    assert "retrieval_facts_ref" not in raw
    assert raw["facts"] == facts
    assert NativeRevisionJournal(connection).current(revision) == result
    connection.close()


@pytest.mark.parametrize("changed_stage", [False, True])
def test_mutated_returned_pair_does_not_rebind_original_input(tmp_path, changed_stage):
    import json

    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    revision = _native().revision_id
    journal.land((_native(),))
    returned = journal.advance(revision, stage="RETRIEVAL_COMPLETE", facts=_retrieval_facts())
    returned["facts"]["retrieval_rights_inventory"].append({"source_id": "not committed"})
    returned["ordinal"] = 100
    stage = "EVIDENCE_HOLD" if changed_stage else "RETRIEVAL_COMPLETE"
    result = journal.advance(revision, stage=stage, facts=_retrieval_facts())
    assert result["ordinal"] == (2 if changed_stage else 1)
    assert result["facts"] == _retrieval_facts()
    assert result == NativeRevisionJournal(connection).current(revision)
    raw = json.loads(connection.execute(
        "SELECT payload_json FROM ledger ORDER BY seq DESC LIMIT 1"
    ).fetchone()[0])
    # Mutating a detached result never mutates the retained pair root.
    assert ("retrieval_facts_ref" in raw) is changed_stage
    assert connection.execute("SELECT count(*) FROM ledger").fetchone()[0] == (3 if changed_stage else 2)
    connection.close()


def test_input_mutation_cannot_change_committed_or_live_facts(tmp_path):
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    revision = _native().revision_id
    journal.land((_native(),))
    for stage in ("RETRIEVAL_COMPLETE", "EVIDENCE_HOLD"):
        facts = _retrieval_facts()
        result = journal.advance(revision, stage=stage, facts=facts)
        facts["retrieval_binding"]["request"]["nodes"].append("not committed")
        facts["retrieval_embeddings"].clear()
        assert result["facts"] == _retrieval_facts()
        assert result == NativeRevisionJournal(connection).current(revision)
    connection.close()


def test_reference_chain_replays_in_one_query_and_preserves_original_records(tmp_path):
    from newsroom.control_plane.native_progress import STATE
    from newsroom.control_plane.store import append_ledger

    path = str(tmp_path / "private.sqlite3")
    connection = connect(path)
    journal = NativeRevisionJournal(connection)
    units = (_native(), _native("other"))
    for unit in units:
        journal.land((unit,))
        # An old inline record is already sufficient; history is not rewritten.
        append_ledger(connection, STATE, {
            "revision_id": unit.revision_id, "ordinal": 1,
            "stage": "RETRIEVAL_COMPLETE", "facts": _retrieval_facts(),
        })
        connection.commit()
    history = connection.execute("SELECT * FROM ledger ORDER BY seq").fetchall()
    journal = NativeRevisionJournal(connection)
    for number in range(40):
        unit = units[number % 2]
        facts = {**_retrieval_facts(), "reason": f"HELD_{number}"}
        last = journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts=facts)
    assert connection.execute("SELECT * FROM ledger ORDER BY seq LIMIT ?", (len(history),)).fetchall() == history
    expected = {unit.revision_id: journal.current(unit.revision_id) for unit in units}
    connection.close()
    connection = connect(path)
    statements = []
    connection.set_trace_callback(statements.append)
    reopened = NativeRevisionJournal(connection)
    connection.set_trace_callback(None)
    assert len(statements) == 1
    assert all(reopened.current(unit.revision_id) == expected[unit.revision_id] for unit in units)
    before = connection.total_changes
    assert reopened.advance(units[1].revision_id, stage="EVIDENCE_HOLD", facts=facts) == last
    assert connection.total_changes == before
    connection.close()


def test_embedding_consumer_reads_inline_facts_from_shared_progress(tmp_path):
    from types import SimpleNamespace
    from newsroom.control_plane.model_usage import _native_embedding_progress_binding

    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    unit = _native()
    journal.land((unit,))
    facts = _retrieval_facts()
    journal.advance(unit.revision_id, stage="RETRIEVAL_COMPLETE", facts=facts)
    cycle_id = f"native-passage:{unit.ingest_id}"
    facts["retrieval_embeddings"] = {unit.ingest_id: {
        "state": "STARTED", "passage_id": "passage", "cycle_id": cycle_id,
        "attempt_number": 1,
    }}
    journal.advance(unit.revision_id, stage="EMBEDDING_STARTED", facts=facts)
    envelope = SimpleNamespace(ingest_id="passage", cycle_id=cycle_id)
    binding = _native_embedding_progress_binding(connection, envelope=envelope)
    assert binding["revision_id"] == unit.revision_id
    assert binding["progress_seq"] == 3
    assert _native_embedding_progress_binding(
        connection, envelope=envelope, retained_progress=binding,
    ) == binding
    connection.close()


def test_commit_failure_rolls_back_reference_and_allows_retry(tmp_path, monkeypatch):
    import sqlite3
    import newsroom.control_plane.native_progress as progress

    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    revision = _native().revision_id
    journal.land((_native(),))
    before = journal.advance(revision, stage="RETRIEVAL_COMPLETE", facts=_retrieval_facts())
    connection.execute("CREATE TABLE test_parent(id INTEGER PRIMARY KEY)")
    connection.execute(
        "CREATE TABLE test_child(parent_id INTEGER REFERENCES test_parent(id) "
        "DEFERRABLE INITIALLY DEFERRED)"
    )
    append = progress.append_ledger

    def fail_at_commit(*args):
        append(*args)
        connection.execute("INSERT INTO test_child VALUES(1)")

    with monkeypatch.context() as patch:
        patch.setattr(progress, "append_ledger", fail_at_commit)
        with pytest.raises(sqlite3.IntegrityError):
            journal.advance(revision, stage="EVIDENCE_HOLD", facts=_retrieval_facts())
    assert not connection.in_transaction
    assert connection.execute("SELECT count(*) FROM ledger").fetchone()[0] == 2
    assert connection.execute("SELECT count(*) FROM test_child").fetchone()[0] == 0
    assert journal.current(revision) == before
    after = journal.advance(revision, stage="EVIDENCE_HOLD", facts=_retrieval_facts())
    assert after["ordinal"] == 2
    assert NativeRevisionJournal(connection).current(revision) == after
    connection.close()


def test_portfolio_reference_reopens_and_changes_only_with_committed_inventory(tmp_path, monkeypatch):
    from copy import deepcopy
    from newsroom.control_plane import native_progress
    from newsroom.control_plane.native_source_intake import NativeSourceDisposition

    path = str(tmp_path / "portfolio-reference.sqlite3")
    connection = connect(path)
    journal = NativeRevisionJournal(connection)
    with pytest.raises(ValueError, match="portfolio reference"):
        journal.portfolio_reference(())
    first = NativeSourceDisposition("UK-01", "READY", "UNCHANGED")
    journal.sources((first,))
    reference = journal.portfolio_reference(journal.portfolio)
    journal.sources((first,))
    assert journal.portfolio_reference(journal.portfolio) == reference
    original = native_progress.append_ledger

    def interrupted(*args):
        original(*args)
        raise RuntimeError("append interrupted")

    second = NativeSourceDisposition("UK-01", "HOLD", "CURRENT_RIGHTS_HOLD")
    with monkeypatch.context() as patch:
        patch.setattr(native_progress, "append_ledger", interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            journal.sources((second,))
    assert journal.portfolio_reference(journal.portfolio) == reference
    assert connection.execute("SELECT count(*) FROM ledger").fetchone()[0] == 1
    connection.close()

    connection = connect(path)
    try:
        journal = NativeRevisionJournal(connection)
        assert journal.portfolio_reference(journal.portfolio) == reference
        forged = deepcopy(journal.portfolio)
        forged[0]["reason_code"] = "FORGED"
        with pytest.raises(ValueError, match="portfolio reference"):
            journal.portfolio_reference(forged)
        journal.sources((second,))
        changed = journal.portfolio_reference(journal.portfolio)
        assert changed["seq"] > reference["seq"]
        assert changed["payload_digest"] != reference["payload_digest"]
        # Mutating the shared logical object also fails its committed-byte proof.
        journal.portfolio[0]["reason_code"] = "FORGED"
        with pytest.raises(ValueError, match="portfolio reference"):
            journal.portfolio_reference(journal.portfolio)
    finally:
        connection.close()


def test_reopen_shares_equal_text_without_aliasing_mutable_facts_or_reordering(tmp_path):
    from newsroom.authority.canonical import canonical_json_bytes

    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    units = tuple(_native(name) for name in ("one", "two", "three"))
    text = "sha256:" + "d" * 64
    for unit in units:
        journal.land((unit,))
        journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts={
            "later_revision": units[-1].revision_id,
            "items": [{"digest": text, "label": "同一份香港證據🙂"}],
        })
    expected_order = tuple(revision for revision, _ in journal.iter_summaries())
    expected = canonical_json_bytes(dict(journal.iter_summaries()))
    rows = connection.execute("SELECT * FROM ledger ORDER BY seq").fetchall()
    changes = connection.total_changes
    reopened = NativeRevisionJournal(connection)
    facts = [reopened.current(unit.revision_id)["facts"] for unit in units]
    assert canonical_json_bytes(dict(reopened.iter_summaries())) == expected
    assert tuple(revision for revision, _ in reopened.iter_summaries()) == expected_order
    assert facts[0]["items"][0]["digest"] is facts[1]["items"][0]["digest"]
    assert facts[0]["items"][0]["label"] is facts[1]["items"][0]["label"]
    assert facts[0]["items"] is not facts[1]["items"]
    assert facts[0]["items"][0] is not facts[1]["items"][0]
    facts[0]["items"][0]["label"] = "changed locally"
    assert facts[1]["items"][0]["label"] == "同一份香港證據🙂"
    assert connection.total_changes == changes
    assert connection.execute("SELECT * FROM ledger ORDER BY seq").fetchall() == rows
    connection.close()


def test_multichunk_land_persists_body_once_with_exact_legacy_replay(tmp_path):
    import json
    from dataclasses import asdict
    from newsroom.control_plane.native_progress import LAND
    from newsroom.control_plane.store import append_ledger

    connection = connect(str(tmp_path / 'shared.sqlite3'))
    unit = replace(_native(), body='A complete paragraph. ' * 1000, chunk_count=3)
    units = tuple(replace(unit, chunk_ordinal=i) for i in range(1, 4))
    journal = NativeRevisionJournal(connection)
    journal.land(units)
    raw = connection.execute('SELECT payload_json FROM ledger').fetchone()[0]
    value = json.loads(raw)
    assert value['shared_body'] == unit.body
    assert all('body' not in value for value in value['units'])
    old = {'revision_id': unit.revision_id, 'units': [asdict(item) for item in units]}
    assert len(raw) < len(json.dumps(old)) * 0.6
    assert NativeRevisionJournal(connection).units[unit.revision_id] == units
    journal.land(units)
    assert connection.execute('SELECT count(*) FROM ledger').fetchone()[0] == 1
    legacy = connect(str(tmp_path / 'legacy.sqlite3'))
    append_ledger(legacy, LAND, old); legacy.commit()
    assert NativeRevisionJournal(legacy).units == journal.units
    legacy.close(); connection.close()


@pytest.mark.parametrize('mutation', ['mixed', 'missing', 'type'])
def test_shared_landing_rejects_ambiguous_or_corrupted_body(tmp_path, mutation):
    import json
    from newsroom.control_plane.native_progress import LAND
    from newsroom.control_plane.store import append_ledger
    connection = connect(str(tmp_path / 'private.sqlite3'))
    unit = replace(_native(), body='Source text. ' * 1000, chunk_count=2)
    units = tuple(replace(unit, chunk_ordinal=i) for i in (1, 2))
    NativeRevisionJournal(connection).land(units)
    value = json.loads(connection.execute('SELECT payload_json FROM ledger').fetchone()[0])
    assert 'shared_body' in value
    if mutation == 'mixed': value['units'][0]['body'] = unit.body
    elif mutation == 'missing': value.pop('shared_body')
    elif mutation == 'type': value['shared_body'] = 123
    # A correct ledger hash does not excuse an ambiguous storage encoding.
    other = connect(str(tmp_path / 'bad.sqlite3'))
    append_ledger(other, LAND, value); other.commit()
    with pytest.raises((ValueError, KeyError, TypeError)):
        NativeRevisionJournal(other)
    other.close(); connection.close()


def test_summary_and_current_are_detached_and_preserve_unknown_and_partial_facts(tmp_path):
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    unit = _native()
    assert journal.summary(unit.revision_id) == journal.current(unit.revision_id) == {}
    assert tuple(journal.iter_summaries()) == ()
    journal.land((unit,))
    facts = {"retrieval_binding": {"partial": ["香港🙂"]}, "future_fact": [None, {"active": True}]}
    expected = journal.advance(unit.revision_id, stage="FUTURE_UNKNOWN_STAGE", facts=facts)
    summary = journal.summary(unit.revision_id)
    assert summary == journal.current(unit.revision_id) == expected
    assert type(summary) is dict and type(summary["facts"]["future_fact"]) is list
    summary["facts"]["future_fact"][1]["active"] = False
    expected["facts"]["retrieval_binding"]["partial"].append("caller only")
    assert journal.current(unit.revision_id)["facts"] == facts
    assert NativeRevisionJournal(connection).current(unit.revision_id)["facts"] == facts
    connection.close()


@pytest.mark.parametrize("mutation", (
    "missing_root", "foreign_root", "obsolete_same_revision", "bool_seq",
    "missing_row", "kind", "payload_header", "same_count_body",
    "self_consistent_unknown_body", "inline_state",
))
def test_selected_current_and_unchanged_pair_advance_deny_tampering_before_write(tmp_path, mutation):
    import json
    from newsroom.authority.canonical import canonical_json_bytes, digest_bytes

    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    unit, other = _native(), _native("other")
    for candidate in (unit, other):
        journal.land((candidate,))
        journal.advance(candidate.revision_id, stage="RETRIEVAL_COMPLETE", facts=_retrieval_facts())
        journal.advance(candidate.revision_id, stage="EVIDENCE_HOLD", facts=_retrieval_facts())
    original_root = journal._pair_roots[unit.revision_id]
    facts = journal.current(unit.revision_id)["facts"]
    if mutation == "missing_root":
        del journal._pair_roots[unit.revision_id]
    elif mutation == "foreign_root":
        journal._pair_roots[unit.revision_id] = journal._pair_roots[other.revision_id]
    elif mutation == "obsolete_same_revision":
        facts["retrieval_binding"]["request"]["nodes"][0] = "Changed same-count canonical context"
        journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts=facts)
        journal._pair_roots[unit.revision_id] = original_root
    elif mutation == "bool_seq":
        journal._pair_roots[unit.revision_id] = replace(original_root, seq=True)
    elif mutation == "missing_row":
        connection.execute("DELETE FROM ledger WHERE seq=?", (original_root.seq,))
    elif mutation == "kind":
        connection.execute("UPDATE ledger SET kind='OTHER' WHERE seq=?", (original_root.seq,))
    elif mutation == "payload_header":
        connection.execute("UPDATE ledger SET payload_digest=? WHERE seq=?", ("sha256:" + "0" * 64, original_root.seq))
    elif mutation == "inline_state":
        journal._summaries[unit.revision_id]["facts"]["reason"] = "Changed same-count live metadata"
    else:
        raw = connection.execute("SELECT payload_json FROM ledger WHERE seq=?", (original_root.seq,)).fetchone()[0]
        value = json.loads(raw)
        if mutation == "self_consistent_unknown_body":
            value["facts"]["reason"] = "Changed same-count original inline fact"
        else:
            value["facts"]["retrieval_rights_inventory"][0]["rights"] = "revoked"
        altered = canonical_json_bytes(value).decode()
        altered_digest = digest_bytes(altered.encode())
        connection.execute("UPDATE ledger SET payload_json=?,payload_digest=? WHERE seq=?", (altered, altered_digest, original_root.seq))
        if mutation == "self_consistent_unknown_body":
            # A coherent byte/header hash is still bound to the original root's
            # complete logical state, not just to its unchanged retrieval pair.
            journal._pair_roots[unit.revision_id] = replace(original_root, payload_digest=altered_digest)
    connection.commit()
    before = connection.total_changes
    with pytest.raises(ValueError, match="(pair root|logical state)"):
        journal.current(unit.revision_id)
    with pytest.raises(ValueError, match="(pair root|logical state)"):
        journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts=facts)
    with pytest.raises(ValueError, match="(pair root|logical state)"):
        journal.advance(unit.revision_id, stage="PUBLICATION_HOLD", facts=facts)
    assert connection.total_changes == before
    connection.close()


def test_metadata_iteration_is_bounded_and_selected_current_uses_one_root_pk(tmp_path):
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    first, second, late = _native(), _native("second"), _native("late")
    for unit in (first, second):
        journal.land((unit,))
        journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts=_retrieval_facts())
    summaries = journal.iter_summaries()
    journal.land((late,))
    journal.advance(late.revision_id, stage="EVIDENCE_HOLD", facts=_retrieval_facts())
    statements = []
    connection.set_trace_callback(statements.append)
    snapshots = list(summaries)
    assert [revision for revision, _ in snapshots] == [first.revision_id, second.revision_id]
    assert all("retrieval_binding" not in value["facts"] for _, value in snapshots)
    assert statements == []
    first_current = journal.current(first.revision_id)
    assert len(statements) == 1 and "WHERE seq=" in statements[0]
    second_current = journal.current(first.revision_id)
    assert len(statements) == 2
    first_current["facts"]["retrieval_rights_inventory"][0]["rights"] = "caller only"
    snapshots[0][1]["facts"]["retrieval_embeddings"].clear()
    assert second_current["facts"] == _retrieval_facts()
    assert journal.summary(first.revision_id)["facts"]["retrieval_embeddings"] == _retrieval_facts()["retrieval_embeddings"]
    connection.set_trace_callback(None)
    connection.close()


@pytest.mark.parametrize("missing", ("summary", "record"))
def test_known_state_missing_its_metadata_denies_reads_and_writes(tmp_path, missing):
    connection = connect(str(tmp_path / "private.sqlite3"))
    journal = NativeRevisionJournal(connection)
    unit = _native()
    journal.land((unit,))
    facts = _retrieval_facts()
    journal.advance(unit.revision_id, stage="EVIDENCE_HOLD", facts=facts)
    if missing == "summary":
        del journal._summaries[unit.revision_id]
    else:
        del journal._records[unit.revision_id]
    before = connection.total_changes
    with pytest.raises(ValueError, match="native progress current"):
        journal.summary(unit.revision_id)
    with pytest.raises(ValueError, match="native progress current"):
        journal.current(unit.revision_id)
    with pytest.raises(ValueError, match="native progress current"):
        tuple(journal.iter_summaries())
    with pytest.raises(ValueError, match="native progress current"):
        journal.advance(unit.revision_id, stage="PUBLICATION_HOLD", facts=facts)
    assert connection.total_changes == before
    connection.close()
