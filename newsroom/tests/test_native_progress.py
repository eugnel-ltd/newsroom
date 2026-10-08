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
    assert connection.execute("SELECT count(*) FROM ledger").fetchone()[0] == 1
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


def test_source_header_projects_immutable_content_observation_on_restart(tmp_path, monkeypatch):
    from dataclasses import FrozenInstanceError
    from newsroom.control_plane.native_progress import _CurrentUnits, source_header

    connection = connect(str(tmp_path / "source-header.sqlite3"))
    first = replace(_native("bno-content-change"), chunk_count=2,
                    canonical_url="https://www.gov.uk/british-national-overseas-bno-visa",
                    updated_at="2024-10-31T17:00:36Z", observed_at="2026-10-08T15:56:04Z",
                    effective_revision=replace(_native().effective_revision,
                                               first_observed_at="2026-10-08T15:56:04Z"))
    units = (first, replace(first, chunk_ordinal=2, predecessor_ingest_id=first.ingest_id))
    journal = NativeRevisionJournal(connection)
    journal.land(units)
    journal.land(tuple(replace(unit, observed_at="2026-10-09T16:00:00Z") for unit in units))
    restarted = NativeRevisionJournal(connection)
    monkeypatch.setattr(_CurrentUnits, "__getitem__", lambda *_: pytest.fail("header selected a body"))
    try:
        header = source_header(restarted.units, first.revision_id)
        assert header.first_observed_at == "2026-10-08T15:56:04Z"
        assert header.updated_at == "2024-10-31T17:00:36Z"
        assert header.observed_ats == ("2026-10-08T15:56:04Z",) * 2
        assert not hasattr(header, "body")
        with pytest.raises(FrozenInstanceError):
            header.first_observed_at = "2026-10-09T16:00:00Z"
    finally:
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
    connection.execute("UPDATE native_current_sources SET content_json='{}'")
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


def test_unchanged_retrieval_pair_is_content_addressed_and_returns_complete_facts(tmp_path):
    connection = connect(str(tmp_path / 'pairs.sqlite3'))
    journal = NativeRevisionJournal(connection)
    unit = _native()
    journal.land((unit,))
    facts = _retrieval_facts()
    journal.advance(unit.revision_id, stage='RETRIEVAL_COMPLETE', facts=facts)
    result = journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=facts)
    assert connection.execute('SELECT count(*) FROM native_current_pairs').fetchone()[0] == 1
    assert connection.execute('SELECT count(*) FROM ledger').fetchone()[0] == 1
    assert 'retrieval_binding' not in journal.summary(unit.revision_id)['facts']
    assert result['facts'] == facts
    assert NativeRevisionJournal(connection).current(unit.revision_id) == result
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


def test_failed_current_write_rolls_back_without_advancing_state(tmp_path, monkeypatch):
    from newsroom.control_plane import native_progress_state as state
    connection = connect(str(tmp_path / 'rollback.sqlite3'))
    journal = NativeRevisionJournal(connection)
    unit = _native()
    journal.land((unit,))
    before = journal.advance(unit.revision_id, stage='RETRIEVAL_COMPLETE', facts=_retrieval_facts())
    original = state.retain_head
    def fail_after_insert(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('injected current-write failure')
    monkeypatch.setattr(state, 'retain_head', fail_after_insert)
    with pytest.raises(RuntimeError, match='injected'):
        journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=_retrieval_facts())
    assert not connection.in_transaction
    assert journal.current(unit.revision_id) == before
    assert NativeRevisionJournal(connection).current(unit.revision_id) == before
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
def test_explicit_legacy_import_rejects_non_object_facts(tmp_path, facts):
    from dataclasses import asdict
    from newsroom.control_plane.native_progress import LAND, STATE, import_legacy_native_progress
    from newsroom.control_plane.store import append_ledger
    connection = connect(str(tmp_path / 'legacy-invalid.sqlite3'))
    connection.execute('DELETE FROM native_current_meta')
    unit = _native()
    append_ledger(connection, LAND, {'revision_id': unit.revision_id, 'units': [asdict(unit)]})
    append_ledger(connection, STATE, {'revision_id': unit.revision_id, 'ordinal': 1, 'stage': 'EVIDENCE_HOLD', 'facts': facts})
    connection.commit()
    with pytest.raises(ValueError, match='facts'):
        import_legacy_native_progress(connection)
    assert connection.execute('SELECT count(*) FROM native_current_sources').fetchone()[0] == 0
    assert connection.execute('SELECT count(*) FROM native_current_meta').fetchone()[0] == 0
    connection.close()


@pytest.mark.parametrize("case", [
    "absent", "forward", "cross_revision", "digest", "ordinal", "older",
    "no_prior_pair", "first", "null", "list", "extra", "bool_seq",
    "bool_ordinal", "inline_binding", "inline_rights",
])
def test_explicit_legacy_import_rejects_invalid_retrieval_reference(tmp_path, case):
    from dataclasses import asdict
    from newsroom.control_plane.native_progress import LAND, STATE, import_legacy_native_progress
    from newsroom.control_plane.store import append_ledger
    connection = connect(str(tmp_path / 'legacy-reference.sqlite3'))
    connection.execute('DELETE FROM native_current_meta')
    unit, other = _native(), _native('other')
    for candidate in (unit, other):
        append_ledger(connection, LAND, {'revision_id': candidate.revision_id, 'units': [asdict(candidate)]})
    append_ledger(connection, STATE, {'revision_id': other.revision_id, 'ordinal': 1, 'stage': 'RETRIEVAL_COMPLETE', 'facts': _retrieval_facts()})
    if case != 'first':
        append_ledger(connection, STATE, {'revision_id': unit.revision_id, 'ordinal': 1, 'stage': 'RETRIEVAL_COMPLETE', 'facts': {} if case == 'no_prior_pair' else _retrieval_facts()})
    connection.commit()
    legacy = NativeRevisionJournal(connection, _legacy=True)
    reference = legacy._records[other.revision_id if case == 'first' else unit.revision_id].reference()
    ordinal = 1 if case == 'first' else 2
    facts = {'reason': 'EVIDENCE_NOT_READY'}
    if case == 'absent': reference['seq'] = 0
    elif case == 'forward': reference['seq'] += 1
    elif case == 'cross_revision': reference = legacy._records[other.revision_id].reference()
    elif case == 'digest': reference['payload_digest'] = 'sha256:' + '0' * 64
    elif case == 'ordinal': reference['ordinal'] += 1
    elif case == 'older':
        append_ledger(connection, STATE, {'revision_id': unit.revision_id, 'ordinal': 2, 'stage': 'EVIDENCE_HOLD', 'facts': _retrieval_facts()})
        ordinal = 3
    elif case == 'null': reference = None
    elif case == 'list': reference = list(reference.items())
    elif case == 'extra': reference['extra'] = 'not part of reference'
    elif case == 'bool_seq': reference['seq'] = True
    elif case == 'bool_ordinal': reference['ordinal'] = True
    elif case == 'inline_binding': facts['retrieval_binding'] = {}
    elif case == 'inline_rights': facts['retrieval_rights_inventory'] = []
    append_ledger(connection, STATE, {'revision_id': unit.revision_id, 'ordinal': ordinal, 'stage': 'EVIDENCE_HOLD', 'facts': facts, 'retrieval_facts_ref': reference})
    connection.commit()
    with pytest.raises(ValueError, match='retrieval facts reference'):
        import_legacy_native_progress(connection)
    assert connection.execute('SELECT count(*) FROM native_current_meta').fetchone()[0] == 0
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
    assert result['facts'] == facts
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
    assert connection.execute('SELECT count(*) FROM native_current_pairs').fetchone()[0] == 1
    assert connection.execute("SELECT count(*) FROM ledger").fetchone()[0] == 1
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


def test_current_boot_cost_is_independent_of_large_obsolete_history(tmp_path):
    from newsroom.control_plane.store import append_ledger
    connection = connect(str(tmp_path / 'bounded-boot.sqlite3'))
    journal = NativeRevisionJournal(connection)
    unit = _native()
    journal.land((unit,))
    expected = journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=_retrieval_facts())
    for number in range(1000):
        append_ledger(connection, 'NATIVE_REVISION_PROGRESS', {'old_debug': 'obsolete body ' * 500, 'ordinal': number})
    connection.commit()
    statements = []
    steps = [0]
    connection.set_trace_callback(statements.append)
    def count_steps():
        steps[0] += 1
        return steps[0] > 500
    connection.set_progress_handler(count_steps, 100)
    reopened = NativeRevisionJournal(connection)
    connection.set_trace_callback(None)
    connection.set_progress_handler(None, 0)
    assert not any('FROM ledger' in statement for statement in statements)
    assert steps[0] < 50
    assert reopened.current(unit.revision_id) == expected
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
    assert binding["progress_seq"] == 2
    assert _native_embedding_progress_binding(
        connection, envelope=envelope, retained_progress=binding,
    ) == binding
    connection.close()


def test_current_commit_failure_rolls_back_and_allows_retry(tmp_path, monkeypatch):
    import sqlite3
    from newsroom.control_plane import native_progress_state as state
    connection = connect(str(tmp_path / 'commit-failure.sqlite3'))
    connection.execute('CREATE TABLE test_parent(id INTEGER PRIMARY KEY)')
    connection.execute('CREATE TABLE test_child(parent_id INTEGER REFERENCES test_parent(id) DEFERRABLE INITIALLY DEFERRED)')
    journal = NativeRevisionJournal(connection)
    unit = _native()
    journal.land((unit,))
    before = journal.advance(unit.revision_id, stage='RETRIEVAL_COMPLETE', facts=_retrieval_facts())
    original = state.retain_head
    def fail_at_commit(*args, **kwargs):
        original(*args, **kwargs)
        connection.execute('INSERT INTO test_child VALUES(1)')
    with monkeypatch.context() as patch:
        patch.setattr(state, 'retain_head', fail_at_commit)
        with pytest.raises(sqlite3.IntegrityError):
            journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=_retrieval_facts())
    assert not connection.in_transaction
    assert connection.execute('SELECT count(*) FROM test_child').fetchone()[0] == 0
    assert journal.current(unit.revision_id) == before
    after = journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=_retrieval_facts())
    assert after['ordinal'] == 2
    assert NativeRevisionJournal(connection).current(unit.revision_id) == after
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
    from newsroom.control_plane.native_progress import import_legacy_native_progress
    legacy.execute('DELETE FROM native_current_meta')
    append_ledger(legacy, LAND, old); legacy.commit()
    import_legacy_native_progress(legacy)
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
    from newsroom.control_plane.native_progress import import_legacy_native_progress
    other.execute('DELETE FROM native_current_meta')
    append_ledger(other, LAND, value); other.commit()
    with pytest.raises((ValueError, KeyError, TypeError)):
        import_legacy_native_progress(other)
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
    "missing_row", "same_count_body", "inline_state", "foreign_digest", "malformed_json",
))
def test_selected_current_pair_and_live_metadata_deny_tampering_before_write(tmp_path, mutation):
    from newsroom.authority.canonical import canonical_json_bytes
    connection = connect(str(tmp_path / 'selected-pair.sqlite3'))
    journal = NativeRevisionJournal(connection)
    unit = _native()
    journal.land((unit,))
    facts = _retrieval_facts()
    journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=facts)
    digest = journal._records[unit.revision_id].pair_digest
    if mutation == 'missing_row':
        connection.execute('PRAGMA foreign_keys=OFF')
        connection.execute('DELETE FROM native_current_pairs WHERE pair_digest=?', (digest,))
    elif mutation == 'inline_state':
        journal._summaries[unit.revision_id]['facts']['reason'] = 'Changed live metadata'
    elif mutation == 'foreign_digest':
        journal._records[unit.revision_id] = replace(journal._records[unit.revision_id], pair_digest='sha256:' + 'f' * 64)
    elif mutation == 'malformed_json':
        connection.execute("UPDATE native_current_pairs SET pair_json='not JSON' WHERE pair_digest=?", (digest,))
    else:
        changed = {key: facts[key] for key in ('retrieval_binding', 'retrieval_rights_inventory')}
        changed['retrieval_rights_inventory'][0]['rights'] = 'revoked'
        connection.execute('UPDATE native_current_pairs SET pair_json=? WHERE pair_digest=?', (canonical_json_bytes(changed).decode(), digest))
    connection.commit()
    before = connection.total_changes
    for call in (lambda: journal.current(unit.revision_id), lambda: journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=_retrieval_facts()), lambda: journal.advance(unit.revision_id, stage='PUBLICATION_HOLD', facts=_retrieval_facts())):
        with pytest.raises(ValueError, match='(pair|logical state)'):
            call()
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
    assert len(statements) == 4 and "WHERE pair_digest=" in statements[0]
    assert all("FROM ledger" not in statement for statement in statements)
    second_current = journal.current(first.revision_id)
    assert len(statements) == 8
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


@pytest.mark.parametrize("referenced", (False, True))
def test_equal_digest_aba_pair_reuses_immutable_content_not_old_diagnostics(tmp_path, referenced):
    connection = connect(str(tmp_path / 'aba.sqlite3'))
    journal = NativeRevisionJournal(connection)
    unit = _native()
    journal.land((unit,))
    first = {**_retrieval_facts(), 'reason': 'A1'}
    journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=first)
    changed = {**_retrieval_facts(), 'retrieval_binding': {'request': {'nodes': ['B2 different context']}}, 'reason': 'B2'}
    journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=changed)
    latest = {**_retrieval_facts(), 'reason': 'A3'}
    journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=latest)
    if referenced:
        latest = {**latest, 'reason': 'A4'}
        journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=latest)
    assert connection.execute('SELECT count(*) FROM native_current_pairs').fetchone()[0] == 1
    assert connection.execute('SELECT count(*) FROM ledger').fetchone()[0] == 1
    assert NativeRevisionJournal(connection).current(unit.revision_id)['facts'] == latest
    assert journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=latest)['facts'] == latest
    connection.close()


def test_current_journal_boot_does_not_read_or_depend_on_debug_history(tmp_path):
    connection = connect(str(tmp_path / "current.sqlite3"))
    journal = NativeRevisionJournal(connection)
    unit = _native()
    journal.land((unit,))
    expected = journal.advance(unit.revision_id, stage="ACKNOWLEDGED", facts=_retrieval_facts())
    # Debug payload loss/corruption is independent of durable CURRENT state.
    connection.execute("UPDATE ledger SET payload_json=NULL WHERE kind='NATIVE_REVISION_PROGRESS'")
    connection.commit()
    statements = []
    connection.set_trace_callback(statements.append)
    reopened = NativeRevisionJournal(connection)
    connection.set_trace_callback(None)
    assert reopened.current(unit.revision_id) == expected
    assert not any("FROM ledger" in statement for statement in statements)
    assert reopened.advance(unit.revision_id, stage="ACKNOWLEDGED", facts=_retrieval_facts()) == expected
    connection.close()


def test_ordinary_progress_diagnostic_has_bounded_metadata_not_full_facts(tmp_path, caplog):
    import json
    connection = connect(str(tmp_path / "compact.sqlite3"))
    journal = NativeRevisionJournal(connection)
    unit = _native()
    journal.land((unit,))
    facts = _retrieval_facts()
    facts['large_other_fact'] = 'complete current state ' * 2000
    expected = journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=facts)
    assert connection.execute('SELECT count(*) FROM ledger').fetchone()[0] == 1
    assert NativeRevisionJournal(connection).current(unit.revision_id) == expected
    connection.close()


def test_current_state_and_pair_corruption_fail_closed_without_history_fallback(tmp_path):
    connection = connect(str(tmp_path / 'corrupt-current.sqlite3'))
    journal = NativeRevisionJournal(connection)
    unit = _native()
    journal.land((unit,))
    journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=_retrieval_facts())
    connection.execute("UPDATE native_current_heads SET state_json='{}'")
    connection.commit()
    with pytest.raises(ValueError, match='CURRENT'):
        NativeRevisionJournal(connection)
    connection.close()


def test_explicit_legacy_import_is_required_and_preserves_source_state_and_pairs(tmp_path):
    from dataclasses import asdict
    from newsroom.control_plane.native_progress import LAND, STATE, import_legacy_native_progress
    from newsroom.control_plane.store import append_ledger
    connection = connect(str(tmp_path / 'legacy.sqlite3'))
    connection.execute('DELETE FROM native_current_meta')
    unit = _native()
    append_ledger(connection, LAND, {'revision_id': unit.revision_id, 'units': [asdict(unit)]})
    expected = {'revision_id': unit.revision_id, 'ordinal': 1, 'stage': 'EVIDENCE_HOLD', 'facts': _retrieval_facts()}
    append_ledger(connection, STATE, expected)
    connection.commit()
    with pytest.raises(ValueError, match='explicit legacy import'):
        NativeRevisionJournal(connection)
    receipt = import_legacy_native_progress(connection)
    assert not receipt['already_current']
    assert NativeRevisionJournal(connection).current(unit.revision_id) == expected
    assert NativeRevisionJournal(connection).units[unit.revision_id] == (unit,)
    assert import_legacy_native_progress(connection)['already_current']
    connection.execute('UPDATE ledger SET payload_json=NULL WHERE kind=?', (STATE,))
    connection.commit()
    assert NativeRevisionJournal(connection).current(unit.revision_id) == expected
    connection.close()


def test_selected_business_landing_uses_current_ingest_key_and_exact_ledger_pin(tmp_path):
    from newsroom.control_plane.model_usage import _native_landed_source_unit, ModelUsageIntegrityError
    connection = connect(str(tmp_path / 'selected-source.sqlite3'))
    journal = NativeRevisionJournal(connection)
    unit = _native()
    journal.land((unit,))
    statements = []
    connection.set_trace_callback(statements.append)
    assert _native_landed_source_unit(connection, ingest_id=unit.ingest_id) == unit
    connection.set_trace_callback(None)
    assert not any("FROM ledger WHERE kind" in statement for statement in statements)
    connection.execute("UPDATE ledger SET payload_json='{}' WHERE kind='NATIVE_REVISION_LANDED'")
    connection.commit()
    assert NativeRevisionJournal(connection).units[unit.revision_id] == (unit,)
    with pytest.raises(ModelUsageIntegrityError, match='source landing'):
        _native_landed_source_unit(connection, ingest_id=unit.ingest_id)
    connection.close()


def test_selected_embedding_business_pin_survives_ordinary_diagnostic_loss(tmp_path):
    from types import SimpleNamespace
    from newsroom.control_plane.model_usage import _native_embedding_progress_binding
    connection = connect(str(tmp_path / 'selected-embedding.sqlite3'))
    journal = NativeRevisionJournal(connection)
    unit = _native()
    journal.land((unit,))
    cycle = f'native-passage:{unit.ingest_id}'
    facts = _retrieval_facts()
    facts['retrieval_embeddings'] = {unit.ingest_id: {
        'state': 'STARTED', 'passage_id': 'passage', 'cycle_id': cycle, 'attempt_number': 1,
    }}
    journal.advance(unit.revision_id, stage='EMBEDDING_STARTED', facts=facts)
    journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=facts)
    envelope = SimpleNamespace(ingest_id='passage', cycle_id=cycle)
    statements = []
    connection.set_trace_callback(statements.append)
    binding = _native_embedding_progress_binding(connection, envelope=envelope)
    connection.set_trace_callback(None)
    assert binding['progress_seq'] == 2
    assert not any('payload_json LIKE' in statement for statement in statements)
    assert _native_embedding_progress_binding(connection, envelope=envelope, retained_progress=binding) == binding
    connection.close()


@pytest.mark.parametrize('table', ['native_current_sources', 'native_current_heads', 'native_current_pairs', 'native_current_units', 'native_current_observations', 'native_current_portfolio'])
def test_current_inventory_detects_missing_rows_before_boot(tmp_path, table):
    from newsroom.control_plane.native_source_intake import NativeSourceDisposition
    connection = connect(str(tmp_path / f'missing-{table}.sqlite3'))
    journal = NativeRevisionJournal(connection)
    unit = _native()
    journal.land((unit,))
    journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=_retrieval_facts())
    reference = ('https://example.test/item', 'sha256:' + 'e' * 64, 'admission', 'access')
    journal.sources((NativeSourceDisposition('UK-01', 'READY', 'UNCHANGED', observations=(reference,)),))
    connection.execute('PRAGMA foreign_keys=OFF')
    connection.execute(f'DELETE FROM {table}')
    connection.commit()
    with pytest.raises(ValueError, match='CURRENT inventory'):
        NativeRevisionJournal(connection)
    connection.close()


def test_missing_current_marker_is_not_silently_reimported_or_reinitialised(tmp_path):
    from newsroom.control_plane.native_progress import import_legacy_native_progress
    path = str(tmp_path / 'missing-marker.sqlite3')
    connection = connect(path)
    journal = NativeRevisionJournal(connection)
    journal.land((_native(),))
    connection.execute('DELETE FROM native_current_meta')
    connection.commit()
    with pytest.raises(ValueError, match='CURRENT'):
        NativeRevisionJournal(connection)
    with pytest.raises(ValueError, match='CURRENT.*recovery'):
        import_legacy_native_progress(connection)
    connection.close()
    connection = connect(path)
    with pytest.raises(ValueError, match='CURRENT'):
        NativeRevisionJournal(connection)
    connection.close()


def test_legacy_import_failure_is_atomic_and_leaves_old_business_records_untouched(tmp_path, monkeypatch):
    from dataclasses import asdict
    from newsroom.control_plane import native_progress_state as state
    from newsroom.control_plane.native_progress import LAND, STATE, import_legacy_native_progress
    from newsroom.control_plane.store import append_ledger
    connection = connect(str(tmp_path / 'import-rollback.sqlite3'))
    connection.execute('DELETE FROM native_current_meta')
    unit = _native()
    append_ledger(connection, LAND, {'revision_id': unit.revision_id, 'units': [asdict(unit)]})
    append_ledger(connection, STATE, {'revision_id': unit.revision_id, 'ordinal': 1, 'stage': 'EVIDENCE_HOLD', 'facts': _retrieval_facts()})
    connection.commit()
    before = connection.execute('SELECT * FROM ledger ORDER BY seq').fetchall()
    original = state.retain_head
    def fail_after_write(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('injected import failure')
    with monkeypatch.context() as patch:
        patch.setattr(state, 'retain_head', fail_after_write)
        with pytest.raises(RuntimeError, match='injected'):
            import_legacy_native_progress(connection)
    for table in (*state.TABLES, 'native_current_units', 'native_current_meta'):
        assert connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0] == 0
    assert connection.execute('SELECT * FROM ledger ORDER BY seq').fetchall() == before
    import_legacy_native_progress(connection)
    assert NativeRevisionJournal(connection).current(unit.revision_id)['ordinal'] == 1
    connection.close()


def test_legacy_embedding_pin_remains_exact_after_current_import_and_diagnostic_retirement(tmp_path):
    from dataclasses import asdict
    from types import SimpleNamespace
    from newsroom.control_plane.native_progress import LAND, STATE, import_legacy_native_progress
    from newsroom.control_plane.model_usage import _native_embedding_progress_binding, ModelUsageIntegrityError
    from newsroom.control_plane.native_progress_state import current_state_retained_sequences
    from newsroom.control_plane.store import append_ledger
    connection = connect(str(tmp_path / 'legacy-pins.sqlite3'))
    connection.execute('DELETE FROM native_current_meta')
    unit = _native()
    cycle = f'native-passage:{unit.ingest_id}'
    facts = _retrieval_facts()
    facts['retrieval_embeddings'] = {unit.ingest_id: {'state': 'STARTED', 'passage_id': 'passage', 'cycle_id': cycle, 'attempt_number': 1}}
    append_ledger(connection, LAND, {'revision_id': unit.revision_id, 'units': [asdict(unit)]})
    append_ledger(connection, STATE, {'revision_id': unit.revision_id, 'ordinal': 1, 'stage': 'EMBEDDING_STARTED', 'facts': facts})
    append_ledger(connection, STATE, {'revision_id': unit.revision_id, 'ordinal': 2, 'stage': 'EVIDENCE_HOLD', 'facts': facts})
    connection.commit()
    envelope = SimpleNamespace(ingest_id='passage', cycle_id=cycle)
    # Immutable selected witness exists before CURRENT conversion as well.
    expected = {'revision_id': unit.revision_id, 'unit_ingest_id': unit.ingest_id, 'passage_id': 'passage', 'embedding_cycle_id': cycle, 'embedding_attempt_number': 1, 'progress_seq': 2}
    import_legacy_native_progress(connection)
    actual = _native_embedding_progress_binding(connection, envelope=envelope)
    assert all(actual[key] == value for key, value in expected.items())
    assert current_state_retained_sequences(connection) == frozenset({1, 2})
    connection.execute('UPDATE ledger SET payload_json=NULL WHERE seq=3')
    connection.commit()
    assert NativeRevisionJournal(connection).current(unit.revision_id)['ordinal'] == 2
    assert _native_embedding_progress_binding(connection, envelope=envelope, retained_progress=actual) == actual
    connection.execute("UPDATE ledger SET payload_json='{}' WHERE seq=2")
    connection.commit()
    with pytest.raises(ModelUsageIntegrityError, match='progress'):
        _native_embedding_progress_binding(connection, envelope=envelope, retained_progress=actual)
    connection.close()


def test_current_source_index_rebinding_fails_closed_even_with_equal_counts(tmp_path):
    connection = connect(str(tmp_path / 'source-index.sqlite3'))
    journal = NativeRevisionJournal(connection)
    first, second = _native(), _native('other')
    journal.land((first,))
    journal.land((second,))
    connection.execute('UPDATE native_current_units SET revision_id=? WHERE ingest_id=?', (second.revision_id, first.ingest_id))
    connection.commit()
    with pytest.raises(ValueError, match='CURRENT source.*binding'):
        NativeRevisionJournal(connection)
    connection.close()


def test_optional_diagnostic_failure_does_not_undo_committed_current_state(tmp_path, monkeypatch):
    from newsroom.control_plane import diagnostic_logging
    connection = connect(str(tmp_path / 'log-failure.sqlite3'))
    journal = NativeRevisionJournal(connection)
    unit = _native()
    journal.land((unit,))
    def broken_sink(*_args):
        raise RuntimeError('optional diagnostic failure')
    monkeypatch.setattr(diagnostic_logging, 'emit_diagnostic', broken_sink)
    expected = journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=_retrieval_facts())
    assert journal.current(unit.revision_id) == expected
    assert NativeRevisionJournal(connection).current(unit.revision_id) == expected
    assert connection.execute('SELECT count(*) FROM ledger').fetchone()[0] == 1
    connection.close()


def test_current_pair_roots_track_heads_not_transition_history(tmp_path):
    connection = connect(str(tmp_path / 'current-pair-roots.sqlite3'))
    journal = NativeRevisionJournal(connection)
    first, second = _native(), _native('other')
    original = _retrieval_facts()
    changed = {**_retrieval_facts(), 'retrieval_binding': {'request': {'nodes': ['Changed exact context']}}}
    for unit in (first, second):
        journal.land((unit,))
        journal.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts=original)
    assert connection.execute('SELECT count(*) FROM native_current_pairs').fetchone()[0] == 1
    journal.advance(first.revision_id, stage='EVIDENCE_HOLD', facts=changed)
    assert connection.execute('SELECT count(*) FROM native_current_pairs').fetchone()[0] == 2
    assert journal.current(second.revision_id)['facts'] == original
    journal.advance(second.revision_id, stage='EVIDENCE_HOLD', facts=changed)
    assert connection.execute('SELECT count(*) FROM native_current_pairs').fetchone()[0] == 1
    journal.advance(first.revision_id, stage='EVIDENCE_HOLD', facts={'reason': 'No current pair'})
    journal.advance(second.revision_id, stage='EVIDENCE_HOLD', facts={'reason': 'No current pair'})
    assert connection.execute('SELECT count(*) FROM native_current_pairs').fetchone()[0] == 0
    assert NativeRevisionJournal(connection).current(first.revision_id)['facts'] == {'reason': 'No current pair'}
    connection.close()


def test_stale_current_writer_does_not_overwrite_a_newer_committed_ordinal(tmp_path):
    connection = connect(str(tmp_path / 'writer-conflict.sqlite3'))
    first = NativeRevisionJournal(connection)
    unit = _native()
    first.land((unit,))
    first.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts={'reason': 'original'})
    stale = NativeRevisionJournal(connection)
    expected = first.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts={'reason': 'newer'})
    with pytest.raises(ValueError, match='CURRENT.*ordinal'):
        stale.advance(unit.revision_id, stage='EVIDENCE_HOLD', facts={'reason': 'stale overwrite'})
    assert not connection.in_transaction
    assert NativeRevisionJournal(connection).current(unit.revision_id) == expected
    connection.close()


def test_current_boot_does_not_commit_or_rollback_callers_transaction(tmp_path):
    connection = connect(str(tmp_path / 'caller-transaction.sqlite3'))
    journal = NativeRevisionJournal(connection)
    journal.land((_native(),))
    connection.execute('CREATE TABLE caller_work(value TEXT)')
    connection.execute("INSERT INTO caller_work VALUES('pending')")
    assert connection.in_transaction
    assert NativeRevisionJournal(connection).units == journal.units
    assert connection.in_transaction
    connection.rollback()
    assert connection.execute('SELECT count(*) FROM caller_work').fetchone()[0] == 0
    connection.close()


def test_unrelated_current_write_does_not_mask_missing_ack_head(tmp_path):
    first = _native('missing-ack')
    second = _native('other-ack')
    connection = connect(str(tmp_path / 'missing-current.sqlite3'))
    journal = NativeRevisionJournal(connection)
    for unit in (first, second):
        journal.land((unit,))
        journal.advance(unit.revision_id, stage='ACKNOWLEDGED', facts={'receipt': unit.ingest_id})
    inventory = connection.execute('SELECT * FROM native_current_meta').fetchall()
    second_before = connection.execute('SELECT * FROM native_current_heads WHERE revision_id=?', (second.revision_id,)).fetchone()
    connection.execute('DELETE FROM native_current_heads WHERE revision_id=?', (first.revision_id,))
    connection.commit()
    with pytest.raises(ValueError, match='CURRENT inventory'):
        journal.advance(second.revision_id, stage='ACKNOWLEDGED', facts={'receipt': second.ingest_id, 'changed': True})
    assert connection.execute('SELECT * FROM native_current_meta').fetchall() == inventory
    assert connection.execute('SELECT * FROM native_current_heads WHERE revision_id=?', (second.revision_id,)).fetchone() == second_before
    with pytest.raises(ValueError, match='CURRENT inventory'):
        NativeRevisionJournal(connection)
    connection.close()


@pytest.mark.parametrize('damage', (None, 'body', 'rights', 'index-digest', 'missing-index', 'pin'))
def test_retrieval_rights_headers_validate_current_bytes_without_full_unit_reconstruction(tmp_path, monkeypatch, damage):
    import json
    from dataclasses import FrozenInstanceError
    from newsroom.control_plane import native_progress as module
    connection = connect(str(tmp_path / 'rights-headers.sqlite3'))
    journal = NativeRevisionJournal(connection)
    first = replace(_native('headers'), chunk_count=2)
    second = replace(first, chunk_ordinal=2, authority=replace(first.authority,
        definition_version_id='00000000-0000-4000-8000-000000000123'))
    units = (first, second)
    journal.land(units)
    if damage in {'body', 'rights'}:
        raw = connection.execute('SELECT content_json FROM native_current_sources WHERE revision_id=?', (first.revision_id,)).fetchone()[0]
        value = json.loads(raw)
        if damage == 'body':
            key = 'shared_body' if 'shared_body' in value else None
            if key: value[key] = 'X' + value[key][1:]
            else: value['units'][0]['body'] = 'X' + value['units'][0]['body'][1:]
        else:
            old = value['units'][1]['authority']['definition_version_id']
            value['units'][1]['authority']['definition_version_id'] = old[:-1] + '4'
        mutated = module.canonical_json_bytes(value).decode()
        assert len(mutated.encode()) == len(raw.encode())
        connection.execute('UPDATE native_current_sources SET content_json=? WHERE revision_id=?', (mutated, first.revision_id))
    elif damage == 'index-digest':
        connection.execute('UPDATE native_current_units SET effective_revision_digest=? WHERE ingest_id=?', ('sha256:'+'f'*64, first.ingest_id))
    elif damage == 'missing-index':
        connection.execute('DELETE FROM native_current_units WHERE ingest_id=?', (second.ingest_id,))
    elif damage == 'pin':
        connection.execute('UPDATE native_current_sources SET land_digest=? WHERE revision_id=?', ('sha256:'+'f'*64, first.revision_id))
    connection.commit()
    monkeypatch.setattr(module, '_landed_units', lambda *_args, **_kwargs: pytest.fail('metadata reader rebuilt full units'))
    try:
        if damage:
            with pytest.raises(ValueError, match='CURRENT'):
                journal.units.retrieval_headers(first.revision_id)
        else:
            headers = journal.units.retrieval_headers(first.revision_id)
            assert [h.ingest_id for h in headers] == [u.ingest_id for u in units]
            assert [h.authority.definition_version_id for h in headers] == [u.authority.definition_version_id for u in units]
            assert all(h.headline == first.headline and h.source_definition_url == first.source_definition_url for h in headers)
            assert all(not hasattr(h, 'body') and not hasattr(h.authority, 'records') for h in headers)
            with pytest.raises(FrozenInstanceError): headers[0].source_id = 'other'
            with pytest.raises(FrozenInstanceError): headers[0].authority.definition_id = 'other'
    finally:
        connection.close()


def _scoped_digital_age_disposition():
    from pathlib import Path
    from datetime import UTC,datetime
    from newsroom.control_plane.govuk_evidence import parse_govuk_content_document,_api_url
    from newsroom.control_plane.native_source_intake import NativeSourceDisposition
    from newsroom.authority.canonical import digest_bytes
    raw=(Path(__file__).parent/'fixtures/govuk/digital-age-0be.json').read_bytes()
    url='https://www.gov.uk/government/news/new-rules-pave-the-way-for-businesses-to-adopt-digital-proof-of-age-for-alcohol-sales'
    document=parse_govuk_content_document(url,raw,retrieved_at=datetime(2026,10,6,tzinfo=UTC),
        extraction_scope=('body','canonical_url','headline','published_at','updated_at'))
    observation=(_api_url(url),digest_bytes(raw),'admission','access')
    return NativeSourceDisposition('UK-01','READY','GOVERNED_REVISIONS_RETAINED',observations=(observation,),
        scope_excluded_assets=document.scope_excluded_assets)


def test_current_portfolio_preserves_exact_text_scope_exclusions_without_repeat_writes(tmp_path):
    from dataclasses import replace
    path=str(tmp_path/'scope-portfolio.sqlite3');connection=connect(path);journal=NativeRevisionJournal(connection)
    disposition=_scoped_digital_age_disposition();journal.sources((disposition,));reference=journal.portfolio_reference(journal.portfolio)
    record=journal.portfolio[0]['scope_excluded_assets'][0]
    assert record['disposition']=='SOURCE_SCOPE_EXCLUDED'and record['mime']=='image/jpeg'
    assert record['raw_root_digest']=='sha256:72e9e48805c3140acde256535c45f80500bc1db6ed937e625f60722094a6e52c'
    assert record['definition_scope']==['body','canonical_url','headline','published_at','updated_at']
    rows=connection.execute('SELECT COUNT(*)FROM ledger').fetchone()[0]
    journal.sources((disposition,));assert connection.execute('SELECT COUNT(*)FROM ledger').fetchone()[0]==rows
    connection.close();connection=connect(path)
    try:
        reopened=NativeRevisionJournal(connection)
        assert reopened.portfolio[0]['scope_excluded_assets'][0]==record
        assert reopened.portfolio_reference(reopened.portfolio)==reference
        changed=replace(disposition,scope_excluded_assets=(replace(disposition.scope_excluded_assets[0],asset_url=disposition.scope_excluded_assets[0].asset_url.replace('Digital-Proof-of-Age-Image.jpg','another.jpg')),))
        reopened.sources((changed,));assert reopened.portfolio_reference(reopened.portfolio)['payload_digest']!=reference['payload_digest']
        reopened.sources((replace(disposition,scope_excluded_assets=()),))
        assert 'scope_excluded_assets'not in reopened.portfolio[0]
    finally:connection.close()


@pytest.mark.parametrize('fault',['root','mime','scope','policy','host','dict'])
def test_current_scope_exclusion_projection_rejects_unbound_metadata(tmp_path,fault):
    from dataclasses import asdict,replace
    disposition=_scoped_digital_age_disposition();record=disposition.scope_excluded_assets[0]
    if fault=='root':record=replace(record,raw_root_digest='sha256:'+'f'*64)
    elif fault=='mime':record=replace(record,mime='application/pdf')
    elif fault=='scope':record=replace(record,definition_scope=('all_assets',))
    elif fault=='policy':record=replace(record,policy_version='unapproved')
    elif fault=='host':record=replace(record,asset_url='https://example.invalid/picture.jpg')
    else:record=asdict(record)
    c=connect(str(tmp_path/'invalid-scope.sqlite3'))
    try:
        journal=NativeRevisionJournal(c)
        with pytest.raises(ValueError,match='scope exclusion'):
            journal.sources((replace(disposition,scope_excluded_assets=(record,)),))
        assert c.execute('SELECT COUNT(*)FROM ledger').fetchone()[0]==0
    finally:c.close()


def test_empty_scope_exclusion_keeps_original_current_portfolio_bytes(tmp_path):
    from newsroom.control_plane.native_source_intake import NativeSourceDisposition
    from newsroom.authority.canonical import canonical_json_bytes,digest_bytes
    c=connect(str(tmp_path/'empty-scope.sqlite3'))
    try:
        journal=NativeRevisionJournal(c);journal.sources((NativeSourceDisposition('UK-01','READY','UNCHANGED'),))
        expected={'sources':[{'source_id':'UK-01','status':'READY','reason_code':'UNCHANGED','revision_ids':[],
            'observations':[],'item_holds':[]}]}
        raw,digest=c.execute('SELECT portfolio_json,portfolio_digest FROM native_current_portfolio WHERE singleton=1').fetchone()
        assert raw==canonical_json_bytes(expected).decode()
        assert digest==digest_bytes(canonical_json_bytes(expected))
        assert 'scope_excluded_assets'not in raw
    finally:c.close()
