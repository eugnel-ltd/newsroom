"""Authenticated Check observation order remains distinct from ingestion order."""
from dataclasses import replace
import sqlite3

import pytest

from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.checks import CheckOutcomeId, CheckRequestId, CheckStateError
from newsroom.sources import SourceItemId, SourceRevisionId
from newsroom.tests.check_3c_authority_helpers import open_check_system, proof, scopes, item_request
from newsroom.tests.check_3c_helpers import ITEM_ID, REQUEST_ID, OUTCOME_ID, NOW, LATER
from newsroom.tests.test_check_3c_authority_store import seed_complete_fixture


def _read(system, **changes):
    arguments = dict(
        request_id=REQUEST_ID, outcome_id=CheckOutcomeId.new(),
        completed_at=LATER, proof=proof(),
    )
    arguments.update(changes)
    return system.checks.observed_prior_revision(arguments.pop("item_id", ITEM_ID), **arguments)


def test_check_only_reader_receives_lineage_id_not_sensitive_source_record(tmp_path):
    database = tmp_path / "authority.sqlite3"
    seed_complete_fixture(database)
    with open_check_system(database, granted_scopes=frozenset({
        "authority.checks.read", "authority.checks.read_sensitive",
    })) as system:
        prior = _read(system)
        assert isinstance(prior, SourceRevisionId)
        assert not hasattr(prior, "request")
        with pytest.raises(PermissionError):
            system.sources.revision(prior, proof=proof())


def test_observed_prior_read_is_typed_authenticated_and_has_no_write_effect(tmp_path):
    database = tmp_path / "authority.sqlite3"
    seed_complete_fixture(database)
    with sqlite3.connect(database) as connection:
        count = connection.execute("SELECT COUNT(*) FROM ledger_events").fetchone()[0]
    with open_check_system(database) as system:
        prior = _read(system)
        assert isinstance(prior, SourceRevisionId)
        assert _read(system, outcome_id=OUTCOME_ID) is None
        with pytest.raises(TypeError, match="typed"):
            _read(system, item_id=str(ITEM_ID))
        with pytest.raises(LookupError, match="retained Item and Check Request"):
            _read(system, request_id=CheckRequestId.new())
        with pytest.raises(CheckStateError, match="boundary differs"):
            _read(system, outcome_id=OUTCOME_ID, completed_at=NOW)
    with open_check_system(database, granted_scopes=scopes() - {"authority.checks.read_sensitive"}) as system:
        with pytest.raises(PermissionError):
            _read(system)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM ledger_events").fetchone()[0] == count


def test_observed_prior_read_rejects_other_item_for_retained_outcome(tmp_path):
    database = tmp_path / "authority.sqlite3"
    seed_complete_fixture(database)
    with open_check_system(database) as system:
        other = replace(
            item_request(), item_id=SourceItemId.new(),
            identity_components=tuple(replace(component, value="other") for component in item_request().identity_components),
            idempotency_key="other-item",
        )
        system.sources.register_item(other, proof=proof())
        with pytest.raises(CheckStateError, match="not observed by its Check Outcome"):
            _read(system, item_id=other.item_id, outcome_id=OUTCOME_ID)


@pytest.mark.parametrize("mutation", ("outcome_header", "occurrence_header", "missing_occurrence", "duplicate_occurrence", "wrong_observed_item"))
def test_observed_prior_read_rejects_corrupt_or_ambiguous_selected_lineage(tmp_path, mutation):
    database = tmp_path / "authority.sqlite3"
    seed_complete_fixture(database)
    with open_check_system(database) as system:
        assert _read(system) is not None
        with sqlite3.connect(database) as connection:
            triggers = connection.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='trigger' "
                "AND tbl_name IN ('check_outcomes','discovery_occurrences','check_outcome_observed_items')"
            ).fetchall()
            for name, _ in triggers:
                connection.execute(f'DROP TRIGGER "{name}"')
            if mutation == "outcome_header":
                connection.execute("UPDATE check_outcomes SET kind='SUCCESS_PARTIAL',incomplete=1")
            elif mutation == "occurrence_header":
                connection.execute("UPDATE discovery_occurrences SET receipt_digest=?", ("sha256:" + "a" * 64,))
            elif mutation == "missing_occurrence":
                connection.execute("DELETE FROM discovery_occurrences")
            elif mutation == "wrong_observed_item":
                connection.execute("UPDATE check_outcome_observed_items SET item_id=?", (str(SourceItemId.new()),))
            else:
                columns = [row[1] for row in connection.execute("PRAGMA table_info(discovery_occurrences)")]
                replacements = {
                    "occurrence_id": "'00000000-0000-4000-8000-000000006999'",
                    "occurrence_kind": "'REOBSERVED'", "semantic_digest": "?",
                    "authority_event_id": "(SELECT event_id FROM ledger_events WHERE event_id NOT IN (SELECT authority_event_id FROM discovery_occurrences) ORDER BY ledger_seq DESC LIMIT 1)",
                }
                selected = [replacements.get(name, name) for name in columns]
                connection.execute(
                    f"INSERT INTO discovery_occurrences ({','.join(columns)}) SELECT {','.join(selected)} FROM discovery_occurrences",
                    ("sha256:" + "a" * 64,),
                )
            for _, sql in triggers:
                connection.execute(sql)
        with pytest.raises((AuthorityPersistenceError, CheckStateError)):
            _read(system)


def test_observed_prior_read_rejects_item_from_previous_source_version(tmp_path):
    from newsroom.sources import SourceDefinitionVersionId
    from newsroom.tests.check_3c_authority_helpers import version_request
    from newsroom.tests.check_3c_helpers import check_request

    database = tmp_path / "authority.sqlite3"
    seed_complete_fixture(database)
    with open_check_system(database) as system:
        original = version_request()
        version = replace(
            original, version_id=SourceDefinitionVersionId.new(), version_number=2,
            expected_previous_version_id=original.version_id, locator=original.locator + "/v2",
            change_reason="Fixture version successor.",
            idempotency_key="source-version-successor",
        )
        system.sources.record_definition_version(version, proof=proof())
        request = replace(
            check_request(), request_id=CheckRequestId.new(),
            definition_version_id=version.version_id,
            idempotency_key="version-successor-check",
        )
        system.checks.register_request(request, proof=proof())
        with pytest.raises(CheckStateError, match="Check source version"):
            _read(system, request_id=request.request_id)
