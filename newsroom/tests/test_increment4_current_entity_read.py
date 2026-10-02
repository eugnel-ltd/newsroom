"""A projection read shares one current entity proof, never across operations."""
from collections import Counter

import pytest

from newsroom.authority._entity_store_read import _EntityReadMixin
from newsroom.authority._entity_store_common import _EntityStoreSupport
from newsroom.authority._increment4_projection_store import _Increment4ProjectionAuthorityStore
from .increment4e_governed_path_helpers import (
    seed_increment4_graphiti_path, admit_increment4_graphiti_path,
    open_graphiti_path_increment4_neo4j_system,
)
from .projection_b2_helpers import MemoryNeo4jAdapter
from .extraction_4a_helpers import extraction_proof
from .test_increment4_active_extension import request, add_suffix, G2


@pytest.mark.parametrize("current_mode", [False, True])
def test_projection_current_entity_pass_uses_one_public_entity_traversal(tmp_path, monkeypatch, record_property, current_mode):
    state = seed_increment4_graphiti_path(tmp_path)
    admit_increment4_graphiti_path(state)
    original = _Increment4ProjectionAuthorityStore._increment4_admitted_states
    entity = _EntityReadMixin.entity
    calls = Counter()
    captured = []
    def counted(self, entity_id):
        calls[str(entity_id)] += 1
        return entity(self, entity_id)
    def observed(self):
        self._current_state_only = current_mode
        calls.clear()
        result = original(self)
        current_calls = dict(calls)
        assert result[0]
        # Existing public surface still repeats its independent checks. It is
        # the record/negative reference, not a new trusted caller switch.
        for current in result[0]:
            calls.clear()
            expected_entity = self.entity(current.entity.entity_id)
            expected_preferred = self.preferred_identity(current.entity.entity_id)
            expected_version = self.entity_version(expected_preferred.current_entity_version_id)
            assert expected_entity == current.entity
            assert expected_preferred == current.preferred
            assert expected_version == current.version
            assert calls[str(current.entity.entity_id)] == 3
            assert current_calls[str(current.entity.entity_id)] == 1
        captured.append(current_calls)
        return result
    monkeypatch.setattr(_EntityReadMixin, 'entity', counted)
    monkeypatch.setattr(_Increment4ProjectionAuthorityStore, '_increment4_admitted_states', observed)
    adapter = MemoryNeo4jAdapter()
    with open_graphiti_path_increment4_neo4j_system(state.relation, adapter) as system:
        system.increment4.build_current_and_promote(request(), proof=extraction_proof())
    assert len(captured) == 1
    add_suffix(state)
    with open_graphiti_path_increment4_neo4j_system(state.relation, adapter) as system:
        system.increment4.build_current_and_promote(request(G2, "counted-suffix"), proof=extraction_proof())
    assert len(captured) >= 3  # Active extension still rechecks after writes.
    record_property('entity_calls_per_member_public_reference', 3)
    record_property('entity_calls_per_member_projection', 1)


def test_combined_current_read_does_not_cache_a_later_rights_denial(tmp_path, monkeypatch):
    state = seed_increment4_graphiti_path(tmp_path)
    admit_increment4_graphiti_path(state)
    original = _Increment4ProjectionAuthorityStore._increment4_admitted_states
    observed = []
    def checked(self):
        valid = original(self)
        assert valid[0]
        def denied(_self, _conn, _entity):
            raise PermissionError('fixture current rights revoked')
        with monkeypatch.context() as patch:
            patch.setattr(_EntityStoreSupport, '_require_entity_current', denied)
            invalid = original(self)
            assert invalid[0] == () and invalid[1] == ()
        observed.append(True)
        return valid
    monkeypatch.setattr(_Increment4ProjectionAuthorityStore, '_increment4_admitted_states', checked)
    with open_graphiti_path_increment4_neo4j_system(state.relation, MemoryNeo4jAdapter()) as system:
        system.increment4.build_current_and_promote(request(), proof=extraction_proof())
    assert observed


@pytest.mark.parametrize("drift", ["preferred_version", "preferred_watermark", "head_version", "projection_authority"])
def test_combined_current_mode_rejects_preferred_head_and_authority_drift(tmp_path, monkeypatch, drift):
    from newsroom.authority.persistence import AuthorityPersistenceError
    state = seed_increment4_graphiti_path(tmp_path)
    admit_increment4_graphiti_path(state)
    combined = _EntityReadMixin._current_entity_projection_records
    captured = []
    def capture(self, entity_id):
        captured.append((self, entity_id))
        return combined(self, entity_id)
    monkeypatch.setattr(_EntityReadMixin, "_current_entity_projection_records", capture)
    with open_graphiti_path_increment4_neo4j_system(state.relation, MemoryNeo4jAdapter()) as system:
        system.increment4.build_current_and_promote(request(), proof=extraction_proof())
        store, entity_id = captured[0]
        store._current_state_only = True
        assert combined(store, entity_id)[1] == store.preferred_identity(entity_id)
        conn = store._connection
        table = "entity_preferred_identities"
        if drift == "head_version": table = "canonical_entity_heads"
        elif drift == "projection_authority": table = "entity_projection_events"
        # Deliberately corrupt only this isolated fixture, including a broken
        # composite FK; CURRENT reads must reject without a global boot sweep.
        conn.execute("PRAGMA foreign_keys=OFF")
        guards = conn.execute("SELECT name,sql FROM sqlite_schema WHERE type='trigger' AND tbl_name=? AND sql LIKE '%BEFORE UPDATE%'", (table,)).fetchall()
        for name, _ in guards: conn.execute('DROP TRIGGER "' + name + '"')
        if drift == "preferred_version":
            other = conn.execute("SELECT entity_version_id FROM canonical_entity_versions WHERE entity_id<>? LIMIT 1", (str(entity_id),)).fetchone()[0]
            conn.execute("UPDATE entity_preferred_identities SET current_entity_version_id=? WHERE entity_id=?", (other, str(entity_id)))
        elif drift == "preferred_watermark":
            conn.execute("UPDATE entity_preferred_identities SET projected_through_ledger_seq=projected_through_ledger_seq+1 WHERE entity_id=?", (str(entity_id),))
        elif drift == "head_version":
            conn.execute("UPDATE canonical_entity_heads SET current_version_number=current_version_number+1 WHERE entity_id=?", (str(entity_id),))
        else:
            other = conn.execute("SELECT event_id FROM ledger_events ORDER BY ledger_seq LIMIT 1").fetchone()[0]
            conn.execute("UPDATE entity_projection_events SET source_event_id=? WHERE projection_event_id=(SELECT projection_event_id FROM entity_projection_events WHERE entity_id=? ORDER BY source_ledger_seq DESC LIMIT 1)", (other, str(entity_id)))
        for _, sql in guards: conn.execute(sql)
        conn.execute("PRAGMA foreign_keys=ON")
        # The public getter is the exact CURRENT-mode rejection reference.
        with pytest.raises(AuthorityPersistenceError): store.preferred_identity(entity_id)
        with pytest.raises(AuthorityPersistenceError): combined(store, entity_id)
