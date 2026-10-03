from __future__ import annotations

from datetime import timedelta
import sqlite3

import pytest

from newsroom.authority import AuthenticationError, HydrationRequest, ObjectAdmissionDenied, ObjectIntegrityError
from .authority_a2b_helpers import MutableClock, admit, open_object_system
from .authority_helpers import FIXED_NOW, proof


def _counts(path):
    with sqlite3.connect(path) as connection:
        return tuple(connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
                     for table in ('object_access_decisions', 'authentication_contexts',
                                   'authorization_requests', 'authorization_decisions'))


def test_rehydrate_reuses_exact_receipt_without_recording_diagnostic_reads(tmp_path):
    path = tmp_path / 'authority.sqlite3'
    with open_object_system(path) as system:
        admission = admit(system, data=b'retained source bytes').admission
        request = HydrationRequest(admission.admission_id, 'project.discovery')
        first = system.objects.rehydrate(request, proof=proof())
        before = _counts(path)
        for _ in range(4):
            retained = system.objects.rehydrate(request, proof=proof())
            assert retained == first
        assert _counts(path) == before
        fresh = system.objects.hydrate(request, proof=proof())
        assert fresh.decision.access_decision_id != first.decision.access_decision_id
        assert _counts(path) == tuple(n + 1 for n in before)
        ranged = system.objects.rehydrate(
            HydrationRequest(admission.admission_id, 'project.discovery', offset=1, length=3), proof=proof())
        assert ranged.data == b'eta'
        assert ranged.decision.access_decision_id != fresh.decision.access_decision_id
    with open_object_system(path) as reopened:
        before = _counts(path)
        assert reopened.objects.rehydrate(request, proof=proof()) == fresh
        assert _counts(path) == before


def test_historical_access_receipt_remains_bound_after_a_fresh_read(tmp_path):
    with open_object_system(tmp_path / "authority.sqlite3") as system:
        admission = admit(system, data=b"retained source bytes").admission
        request = HydrationRequest(admission.admission_id, "project.discovery")
        first = system.objects.hydrate(request, proof=proof()).decision
        latest = system.objects.hydrate(request, proof=proof()).decision
        assert latest.access_decision_id != first.access_decision_id
        assert system.objects.access_decision(
            first.access_decision_id,
            admission_id=admission.admission_id,
            purpose="project.discovery",
            proof=proof(),
        ) == first


@pytest.mark.parametrize(("column", "replacement"), [
    ("principal_id", "different-principal"),
    ("authority_domain", "different-authority"),
    ("purpose", "different-purpose"),
    ("admission_id", None),
    ("object_class", "different.object"),
    ("allowed_use", "different.use"),
    ("security_scope", "different.security"),
    ("retention_scope", "different.retention"),
    ("byte_offset", 1),
    ("allowed_bytes", 9),
    ("decided_at", "2026-01-01T00:00:00.000000Z"),
])
def test_historical_access_receipt_rejects_index_mismatch(
    tmp_path, column, replacement,
):
    path = tmp_path / "authority.sqlite3"
    with open_object_system(path) as system:
        first = admit(system, key="first", data=b"same-size-a").admission
        second = admit(system, key="second", data=b"same-size-b").admission
        receipt = system.objects.hydrate(
            HydrationRequest(first.admission_id, "project.discovery"), proof=proof(),
        ).decision
    with sqlite3.connect(path) as connection:
        trigger, = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE name='immutable_object_access_decisions_update'"
        ).fetchone()
        connection.execute("DROP TRIGGER immutable_object_access_decisions_update")
        connection.execute(
            f"UPDATE object_access_decisions SET {column}=? "
            "WHERE access_decision_id=?",
            (
                str(second.admission_id) if replacement is None else replacement,
                str(receipt.access_decision_id),
            ),
        )
        connection.execute(trigger)
    from newsroom.authority import AuthorityPersistenceError
    with open_object_system(path) as reopened:
        with pytest.raises(AuthorityPersistenceError, match="indexed fields"):
            reopened.objects.access_decision(
                receipt.access_decision_id,
                admission_id=(
                    second.admission_id if column == "admission_id"
                    else first.admission_id
                ),
                purpose="project.discovery",
                proof=proof(),
            )


def test_historical_access_receipt_rejects_rebound_security_chain(tmp_path):
    import json
    from newsroom.authority import AuthorityPersistenceError, canonical_json_bytes
    from newsroom.authority.canonical import digest_bytes

    path = tmp_path / "authority.sqlite3"
    with open_object_system(path) as system:
        first = admit(system, key="first", data=b"same-size-a").admission
        second = admit(system, key="second", data=b"same-size-b").admission
        first_read = system.objects.hydrate(
            HydrationRequest(first.admission_id, "project.discovery"), proof=proof(),
        ).decision
        second_read = system.objects.hydrate(
            HydrationRequest(second.admission_id, "project.discovery"), proof=proof(),
        ).decision
    fields = (
        "authentication_context_id", "authorization_request_digest",
        "authorization_decision_id",
    )
    changed = tuple(str(getattr(second_read, field)) for field in fields)
    with sqlite3.connect(path) as connection:
        trigger, = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE name='immutable_object_access_decisions_update'"
        ).fetchone()
        raw, = connection.execute(
            "SELECT canonical_bytes FROM object_access_decisions "
            "WHERE access_decision_id=?",
            (str(first_read.access_decision_id),),
        ).fetchone()
        record = json.loads(raw)
        record.update(zip(fields, changed))
        raw = canonical_json_bytes(record)
        connection.execute("DROP TRIGGER immutable_object_access_decisions_update")
        connection.execute(
            "UPDATE object_access_decisions SET authentication_context_id=?,"
            "authorization_request_digest=?,authorization_decision_id=?,"
            "canonical_bytes=?,canonical_digest=? WHERE access_decision_id=?",
            (*changed, raw, digest_bytes(raw), str(first_read.access_decision_id)),
        )
        connection.execute(trigger)
    with open_object_system(path) as reopened:
        with pytest.raises(AuthorityPersistenceError, match="binding differs"):
            reopened.objects.access_decision(
                first_read.access_decision_id,
                admission_id=first.admission_id,
                purpose="project.discovery",
                proof=proof(),
            )


def test_historical_access_receipt_rejects_altered_credential_provenance(tmp_path):
    from newsroom.authority import AuthorityPersistenceError, canonical_json_bytes
    from newsroom.authority.canonical import digest_bytes

    path = tmp_path / "authority.sqlite3"
    with open_object_system(path) as system:
        admission = admit(system, data=b"retained source bytes").admission
        receipt = system.objects.hydrate(
            HydrationRequest(admission.admission_id, "project.discovery"), proof=proof(),
        ).decision
    with sqlite3.connect(path) as connection:
        trigger, = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE name='immutable_authentication_contexts_update'"
        ).fetchone()
        from newsroom.authority.security_record_migrations import AUTHENTICATION_CONTEXT_FIELDS
        columns = ','.join(AUTHENTICATION_CONTEXT_FIELDS)
        values = connection.execute(
            f"SELECT {columns} FROM authentication_contexts "
            "WHERE authentication_context_id=?",
            (str(receipt.authentication_context_id),),
        ).fetchone()
        record = dict(zip(AUTHENTICATION_CONTEXT_FIELDS, values, strict=True))
        record["credential_binding_digest"] = "sha256:" + "f" * 64
        raw = canonical_json_bytes(record)
        connection.execute("DROP TRIGGER immutable_authentication_contexts_update")
        connection.execute(
            "UPDATE authentication_contexts SET credential_binding_digest=?,"
            "canonical_digest=? "
            "WHERE authentication_context_id=?",
            (
                record["credential_binding_digest"], digest_bytes(raw),
                str(receipt.authentication_context_id),
            ),
        )
        connection.execute(trigger)
    with open_object_system(path) as reopened:
        with pytest.raises(AuthorityPersistenceError, match="security binding"):
            reopened.objects.access_decision(
                receipt.access_decision_id,
                admission_id=admission.admission_id,
                purpose="project.discovery",
                proof=proof(),
            )


def test_historical_access_receipt_rejects_post_expiry_access_time(tmp_path):
    import json
    from newsroom.authority import AuthorityPersistenceError, canonical_json_bytes
    from newsroom.authority.canonical import digest_bytes

    path = tmp_path / "authority.sqlite3"
    with open_object_system(path) as system:
        admission = admit(system, data=b"retained source bytes").admission
        receipt = system.objects.hydrate(
            HydrationRequest(admission.admission_id, "project.discovery"), proof=proof(),
        ).decision
    with sqlite3.connect(path) as connection:
        trigger, = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE name='immutable_object_access_decisions_update'"
        ).fetchone()
        raw, = connection.execute(
            "SELECT canonical_bytes FROM object_access_decisions "
            "WHERE access_decision_id=?",
            (str(receipt.access_decision_id),),
        ).fetchone()
        record = json.loads(raw)
        record["decided_at"] = "2099-01-01T00:00:00.000000Z"
        raw = canonical_json_bytes(record)
        connection.execute("DROP TRIGGER immutable_object_access_decisions_update")
        connection.execute(
            "UPDATE object_access_decisions SET decided_at=?,canonical_bytes=?,"
            "canonical_digest=? WHERE access_decision_id=?",
            (
                record["decided_at"], raw, digest_bytes(raw),
                str(receipt.access_decision_id),
            ),
        )
        connection.execute(trigger)
    with open_object_system(path) as reopened:
        with pytest.raises(AuthorityPersistenceError, match="security binding"):
            reopened.objects.access_decision(
                receipt.access_decision_id,
                admission_id=admission.admission_id,
                purpose="project.discovery",
                proof=proof(),
            )


def test_historical_access_receipt_rejects_post_rights_expiry_time(tmp_path):
    import json
    from newsroom.authority import AuthorityPersistenceError, canonical_json_bytes
    from newsroom.authority.canonical import digest_bytes

    path = tmp_path / "authority.sqlite3"
    with open_object_system(path) as system:
        admission = admit(
            system, data=b"retained source bytes", admission_type="source.short",
        ).admission
        receipt = system.objects.hydrate(
            HydrationRequest(admission.admission_id, "project.discovery"), proof=proof(),
        ).decision
    with sqlite3.connect(path) as connection:
        trigger, = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE name='immutable_object_access_decisions_update'"
        ).fetchone()
        raw, = connection.execute(
            "SELECT canonical_bytes FROM object_access_decisions "
            "WHERE access_decision_id=?",
            (str(receipt.access_decision_id),),
        ).fetchone()
        record = json.loads(raw)
        record["decided_at"] = "2026-07-16T12:01:00.000000Z"
        raw = canonical_json_bytes(record)
        connection.execute("DROP TRIGGER immutable_object_access_decisions_update")
        connection.execute(
            "UPDATE object_access_decisions SET decided_at=?,canonical_bytes=?,"
            "canonical_digest=? WHERE access_decision_id=?",
            (
                record["decided_at"], raw, digest_bytes(raw),
                str(receipt.access_decision_id),
            ),
        )
        connection.execute(trigger)
    with open_object_system(path) as reopened:
        with pytest.raises(AuthorityPersistenceError, match="admission binding"):
            reopened.objects.access_decision(
                receipt.access_decision_id,
                admission_id=admission.admission_id,
                purpose="project.discovery",
                proof=proof(),
            )


def test_historical_access_receipt_rejects_truncated_read_to_end(tmp_path):
    import json
    from newsroom.authority import AuthorityPersistenceError, canonical_json_bytes
    from newsroom.authority.canonical import digest_bytes

    path = tmp_path / "authority.sqlite3"
    with open_object_system(path) as system:
        admission = admit(system, data=b"0123456789").admission
        receipt = system.objects.hydrate(
            HydrationRequest(admission.admission_id, "project.discovery"), proof=proof(),
        ).decision
    with sqlite3.connect(path) as connection:
        trigger, = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE name='immutable_object_access_decisions_update'"
        ).fetchone()
        raw, = connection.execute(
            "SELECT canonical_bytes FROM object_access_decisions "
            "WHERE access_decision_id=?",
            (str(receipt.access_decision_id),),
        ).fetchone()
        record = json.loads(raw)
        record["allowed_bytes"] = 4
        record["state_cutoff"]["length"] = 4
        cutoff = canonical_json_bytes(record["state_cutoff"])
        record["state_cutoff_digest"] = digest_bytes(cutoff)
        raw = canonical_json_bytes(record)
        connection.execute("DROP TRIGGER immutable_object_access_decisions_update")
        connection.execute(
            "UPDATE object_access_decisions SET allowed_bytes=4,"
            "state_cutoff_bytes=?,state_cutoff_digest=?,canonical_bytes=?,"
            "canonical_digest=? WHERE access_decision_id=?",
            (
                cutoff, record["state_cutoff_digest"], raw, digest_bytes(raw),
                str(receipt.access_decision_id),
            ),
        )
        connection.execute(trigger)
    with open_object_system(path) as reopened:
        with pytest.raises(AuthorityPersistenceError, match="admission binding"):
            reopened.objects.access_decision(
                receipt.access_decision_id,
                admission_id=admission.admission_id,
                purpose="project.discovery",
                proof=proof(),
            )


@pytest.mark.parametrize("case", ["cross_admission_use", "rebound_rights_cutoff"])
def test_historical_access_receipt_rejects_admission_authority_rebind(
    tmp_path, case,
):
    import json
    from newsroom.authority import AuthorityPersistenceError, canonical_json_bytes
    from newsroom.authority.canonical import digest_bytes

    path = tmp_path / "authority.sqlite3"
    with open_object_system(path) as system:
        first = admit(system, key="first", data=b"same-size-a").admission
        second = admit(system, key="second", data=b"same-size-b").admission
        receipt = system.objects.hydrate(
            HydrationRequest(first.admission_id, "project.discovery"), proof=proof(),
        ).decision
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        trigger, = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE name='immutable_object_access_decisions_update'"
        ).fetchone()
        row = connection.execute(
            "SELECT * FROM object_access_decisions WHERE access_decision_id=?",
            (str(receipt.access_decision_id),),
        ).fetchone()
        record = json.loads(row["canonical_bytes"])
        updates = {}
        if case == "cross_admission_use":
            record["allowed_use"] = "publish.article"
            updates["allowed_use"] = record["allowed_use"]
        else:
            other = connection.execute(
                "SELECT a.rights_decision_id,r.canonical_digest "
                "FROM object_admissions a JOIN object_rights_decisions r "
                "ON r.rights_decision_id=a.rights_decision_id "
                "WHERE a.admission_id=?",
                (str(second.admission_id),),
            ).fetchone()
            record["state_cutoff"].update({
                "rights_decision_id": other["rights_decision_id"],
                "rights_decision_digest": other["canonical_digest"],
            })
        cutoff = canonical_json_bytes(record["state_cutoff"])
        record["state_cutoff_digest"] = digest_bytes(cutoff)
        raw = canonical_json_bytes(record)
        updates.update({
            "state_cutoff_bytes": cutoff,
            "state_cutoff_digest": record["state_cutoff_digest"],
            "canonical_bytes": raw,
            "canonical_digest": digest_bytes(raw),
        })
        connection.execute("DROP TRIGGER immutable_object_access_decisions_update")
        connection.execute(
            "UPDATE object_access_decisions SET "
            + ",".join(f"{field}=?" for field in updates)
            + " WHERE access_decision_id=?",
            (*updates.values(), str(receipt.access_decision_id)),
        )
        connection.execute(trigger)
    with open_object_system(path) as reopened:
        with pytest.raises(AuthorityPersistenceError, match="admission binding"):
            reopened.objects.access_decision(
                receipt.access_decision_id,
                admission_id=first.admission_id,
                purpose="project.discovery",
                proof=proof(),
            )


def test_historical_access_receipt_rejects_rights_canonical_index_split(tmp_path):
    import json
    from newsroom.authority import AuthorityPersistenceError, canonical_json_bytes
    from newsroom.authority.canonical import digest_bytes

    path = tmp_path / "authority.sqlite3"
    with open_object_system(path) as system:
        first = admit(system, key="first", data=b"same-size-a").admission
        second = admit(system, key="second", data=b"same-size-b").admission
        receipt = system.objects.hydrate(
            HydrationRequest(first.admission_id, "project.discovery"), proof=proof(),
        ).decision
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        access_trigger, = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE name='immutable_object_access_decisions_update'"
        ).fetchone()
        rights_trigger, = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE name='immutable_object_rights_decisions_update'"
        ).fetchone()
        access = connection.execute(
            "SELECT * FROM object_access_decisions WHERE access_decision_id=?",
            (str(receipt.access_decision_id),),
        ).fetchone()
        admission = connection.execute(
            "SELECT * FROM object_admissions WHERE admission_id=?",
            (str(first.admission_id),),
        ).fetchone()
        other_blob, = connection.execute(
            "SELECT blob_digest FROM object_admissions WHERE admission_id=?",
            (str(second.admission_id),),
        ).fetchone()
        rights = connection.execute(
            "SELECT * FROM object_rights_decisions WHERE rights_decision_id=?",
            (admission["rights_decision_id"],),
        ).fetchone()
        rights_value = json.loads(rights["canonical_bytes"])
        rights_value["blob"]["blob_digest"] = other_blob
        rights_raw = canonical_json_bytes(rights_value)
        rights_digest = digest_bytes(rights_raw)
        access_value = json.loads(access["canonical_bytes"])
        access_value["state_cutoff"]["rights_decision_digest"] = rights_digest
        cutoff = canonical_json_bytes(access_value["state_cutoff"])
        access_value["state_cutoff_digest"] = digest_bytes(cutoff)
        access_raw = canonical_json_bytes(access_value)
        connection.execute("DROP TRIGGER immutable_object_access_decisions_update")
        connection.execute("DROP TRIGGER immutable_object_rights_decisions_update")
        connection.execute(
            "UPDATE object_rights_decisions SET canonical_bytes=?,canonical_digest=? "
            "WHERE rights_decision_id=?",
            (rights_raw, rights_digest, admission["rights_decision_id"]),
        )
        connection.execute(
            "UPDATE object_access_decisions SET state_cutoff_bytes=?,"
            "state_cutoff_digest=?,canonical_bytes=?,canonical_digest=? "
            "WHERE access_decision_id=?",
            (
                cutoff, access_value["state_cutoff_digest"], access_raw,
                digest_bytes(access_raw), str(receipt.access_decision_id),
            ),
        )
        connection.execute(rights_trigger)
        connection.execute(access_trigger)
    with open_object_system(path) as reopened:
        with pytest.raises(AuthorityPersistenceError, match="rights indexed fields"):
            reopened.objects.access_decision(
                receipt.access_decision_id,
                admission_id=first.admission_id,
                purpose="project.discovery",
                proof=proof(),
            )


def test_rehydrate_still_authenticates_and_checks_revocation(tmp_path):
    path = tmp_path / 'authority.sqlite3'
    with open_object_system(path) as system:
        admission = admit(system).admission
        request = HydrationRequest(admission.admission_id, 'project.discovery')
        system.objects.rehydrate(request, proof=proof())
        with pytest.raises(AuthenticationError):
            system.objects.rehydrate(request, proof=proof(credential='wrong-token'))
        system.objects.revoke(admission.admission_id, reason_code='REVOKED', idempotency_key='revoke', proof=proof())
        before = _counts(path)
        with pytest.raises(ObjectAdmissionDenied):
            system.objects.rehydrate(request, proof=proof())
        assert _counts(path) == before


def test_rehydrate_rechecks_current_rights_expiry(tmp_path):
    clock = MutableClock(FIXED_NOW)
    with open_object_system(tmp_path / 'authority.sqlite3', clock=clock) as system:
        admission = admit(system, admission_type='source.short').admission
        request = HydrationRequest(admission.admission_id, 'project.discovery')
        system.objects.rehydrate(request, proof=proof())
        clock.current = type(FIXED_NOW)(FIXED_NOW.value + timedelta(seconds=31))
        with pytest.raises(ObjectAdmissionDenied):
            system.objects.rehydrate(request, proof=proof())


@pytest.mark.parametrize('fault_type', ('expiry', 'bytes'))
def test_rehydrate_rechecks_rights_and_pinned_bytes_after_read(tmp_path, fault_type):
    clock = MutableClock(FIXED_NOW)
    root = tmp_path / 'objects'
    armed = False

    def fault(checkpoint):
        if not armed or checkpoint != 'after_range_read_before_rehash':
            return
        if fault_type == 'expiry':
            clock.current = type(FIXED_NOW)(FIXED_NOW.value + timedelta(seconds=31))
        else:
            [path] = [path for path in (root / 'objects').rglob('*') if path.is_file()]
            path.chmod(0o600)
            path.write_bytes(b'tampered source bytes')
            path.chmod(0o400)

    path = tmp_path / 'authority.sqlite3'
    with open_object_system(path, object_root=root, clock=clock, fault_hook=fault) as system:
        admission = admit(system, admission_type='source.short').admission
        request = HydrationRequest(admission.admission_id, 'project.discovery')
        system.objects.rehydrate(request, proof=proof())
        before = _counts(path)
        armed = True
        with pytest.raises(ObjectAdmissionDenied if fault_type == 'expiry' else ObjectIntegrityError):
            system.objects.rehydrate(request, proof=proof())
        assert _counts(path) == before


def test_rehydrate_rejects_rebound_request_for_another_admission(tmp_path):
    import json
    from newsroom.authority import AuthorityPersistenceError, canonical_json_bytes
    from newsroom.authority.canonical import digest_bytes

    path = tmp_path / 'authority.sqlite3'
    with open_object_system(path) as system:
        first = admit(system, key='one', data=b'one').admission
        second = admit(system, key='two', data=b'two').admission
        request = HydrationRequest(first.admission_id, 'project.discovery')
        first_read = system.objects.hydrate(request, proof=proof())
        second_read = system.objects.hydrate(HydrationRequest(second.admission_id, 'project.discovery'), proof=proof())
        with sqlite3.connect(path) as conn:
            trigger, = conn.execute("SELECT sql FROM sqlite_master WHERE name='immutable_object_access_decisions_update'").fetchone()
            raw, = conn.execute('SELECT canonical_bytes FROM object_access_decisions WHERE access_decision_id=?',
                                (str(first_read.decision.access_decision_id),)).fetchone()
            record = json.loads(raw)
            fields = ('authentication_context_id', 'authorization_request_digest', 'authorization_decision_id')
            changed = tuple(str(getattr(second_read.decision, field)) for field in fields)
            record.update(zip(fields, changed))
            raw = canonical_json_bytes(record)
            conn.execute('DROP TRIGGER immutable_object_access_decisions_update')
            conn.execute('UPDATE object_access_decisions SET authentication_context_id=?,authorization_request_digest=?,authorization_decision_id=?,canonical_bytes=?,canonical_digest=? WHERE access_decision_id=?',
                         (*changed, raw, digest_bytes(raw), str(first_read.decision.access_decision_id)))
            conn.execute(trigger)
        with pytest.raises(AuthorityPersistenceError, match='request semantics'):
            system.objects.rehydrate(request, proof=proof())


def test_current_rehydrate_reuses_exact_event_only_within_its_transaction(tmp_path, monkeypatch):
    with open_object_system(tmp_path / 'current-rehydrate.sqlite3') as system:
        admission = admit(system, data=b'retained source bytes').admission
        store = system.objects._GovernedObjects__rehydrate.__self__._store
        store._current_state_only = True
        request = HydrationRequest(admission.admission_id, 'project.discovery')
        first = system.objects.rehydrate(request, proof=proof())
        event_checks, current_checks = [], []
        exact = store._validate_retained_event_exact
        current = store._current_admission_row
        def check_event(event_id):
            event_checks.append(event_id)
            return exact(event_id)
        def check_current(*args, **kwargs):
            current_checks.append(kwargs['now'])
            return current(*args, **kwargs)
        monkeypatch.setattr(store, '_validate_retained_event_exact', check_event)
        monkeypatch.setattr(store, '_current_admission_row', check_current)
        for turn in (1, 2):
            assert system.objects.rehydrate(request, proof=proof()) == first
            assert len(event_checks) == turn
            assert len(current_checks) == 3 * turn
            assert store._exact_event_scope is None
        with pytest.raises(AuthenticationError):
            system.objects.rehydrate(request, proof=proof(credential='wrong'))
        system.objects.revoke(admission.admission_id, reason_code='REVOKED', idempotency_key='revoke-scoped', proof=proof())
        with pytest.raises(ObjectAdmissionDenied, match='ACTIVE'):
            system.objects.rehydrate(request, proof=proof())
        assert store._exact_event_scope is None
