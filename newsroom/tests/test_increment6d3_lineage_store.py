from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import scripts.sdlc.workflow_lane as sdlc_lane
from newsroom.authority.auth import StaticAuthorizer
from newsroom.authority.persistence import AuthoritySchemaError
from newsroom.increment6.lineage import (
    EventHypothesisLineageAuthority,
    HypothesisLineageHead,
    HypothesisLineageReceipt,
    HypothesisLineageRelationshipProof,
    lineage_command_definition,
    open_event_hypothesis_lineage_authority,
)
from newsroom.increment6.outcomes import CanonicalOutcome
from newsroom.increment6.relationships import (
    open_event_hypothesis_relationship_authority,
    relationship_command_definition,
)
from newsroom.tests import test_increment6d2_relationship_store as d2
from newsroom.tests import test_increment6d3_lineage as d3

REPO_ROOT = Path(__file__).parents[2]


def _seed(tmp_path):
    seed = d2._seed_location(tmp_path / "lineage-store")
    relationship_scope = relationship_command_definition().required_scope
    lineage_scope = lineage_command_definition().required_scope
    authorizer = StaticAuthorizer(
        policy_version="lineage-test-v1",
        grants_by_principal={
            "editor": frozenset({relationship_scope, lineage_scope}),
            "other": frozenset({relationship_scope, lineage_scope}),
        },
    )
    args = d2._open_arguments(seed)
    args["authorizer"] = authorizer
    output, left, right = seed[3][:3]
    inputs = tuple(sorted((left, right), key=lambda item: item.version_id))
    assessment, evidence = d3._decision(output, inputs, CanonicalOutcome.REL_SAME_STATE)
    relationship = open_event_hypothesis_relationship_authority(**args)
    try:
        relationship.retain(
            assessment.canonical_bytes,
            tuple(item.canonical_bytes for item in evidence),
            proof=seed[0][3],
        )
    finally:
        relationship.close()
    receipt = HypothesisLineageReceipt.consolidation(
        expected_generation=0,
        inputs=inputs,
        output=output,
        relationship_proofs=(
            HypothesisLineageRelationshipProof.from_assessment(assessment, evidence),
        ),
    )
    return seed, args, receipt


class _OpeningSignal(BaseException):
    pass


@pytest.mark.parametrize("failure", ("port", "service", "verify"))
def test_post_super_open_failure_releases_writer_for_corrected_retry(
    tmp_path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from newsroom.authority import _event_hypothesis_lineage_system as private

    _, args, _ = _seed(tmp_path)
    with monkeypatch.context() as scoped:
        if failure == "port":
            scoped.setattr(
                private,
                "_create_event_hypothesis_relationship_read_port",
                lambda *args, **kwargs: (_ for _ in ()).throw(_OpeningSignal()),
            )
        elif failure == "service":
            scoped.setattr(
                private,
                "CommandService",
                lambda *args, **kwargs: (_ for _ in ()).throw(_OpeningSignal()),
            )
        else:
            scoped.setattr(
                private._LineageStore,
                "_verify",
                lambda self: (_ for _ in ()).throw(_OpeningSignal()),
            )
        with pytest.raises(_OpeningSignal):
            open_event_hypothesis_lineage_authority(**args)

    corrected = open_event_hypothesis_lineage_authority(**args)
    corrected.close()


def _rewrite_retained_relationship_evidence(database) -> None:
    connection = sqlite3.connect(database, isolation_level=None)
    trigger_sql = str(
        connection.execute(
            "SELECT sql FROM sqlite_schema WHERE type='trigger' AND name=?",
            ("immutable_event_hypothesis_relationship_update",),
        ).fetchone()[0]
    )
    row = connection.execute(
        "SELECT decision_id,assessment_bytes,evidence_bytes "
        "FROM event_hypothesis_relationship_decisions"
    ).fetchone()
    assessment = d2.RelationshipAssessment.from_canonical_bytes(bytes(row[1]))
    values = d2.json.loads(bytes(row[2]))
    evidence = tuple(d2.ComparatorEvidence.from_value(value) for value in values)
    tampered = replace(evidence[0], score=evidence[0].score + 1)
    rewritten_evidence = (tampered, *evidence[1:])
    rewritten = d2.assess_relationships(
        assessment.subject,
        assessment.comparator_manifest,
        rewritten_evidence,
    )
    connection.execute("DROP TRIGGER immutable_event_hypothesis_relationship_update")
    connection.execute(
        "UPDATE event_hypothesis_relationship_decisions SET "
        "decision_id=?,decision=?,assessment_bytes=?,assessment_digest=?,"
        "evidence_bytes=?,evidence_digest=? WHERE decision_id=?",
        (
            rewritten.canonical_digest,
            rewritten.decision.value,
            rewritten.canonical_bytes,
            rewritten.canonical_digest,
            d2.canonical_json_bytes(
                [item.canonical_value for item in rewritten_evidence]
            ),
            rewritten.evidence_digest,
            str(row[0]),
        ),
    )
    connection.execute(trigger_sql)
    connection.close()


def test_empty_lineage_verifies_retained_relationship_history(tmp_path) -> None:
    (tmp_path / "clean").mkdir()
    seed, args, _ = _seed(tmp_path / "clean")
    authority = open_event_hypothesis_lineage_authority(**args)
    try:
        assert authority.history() == ()
        assert authority.current_heads(proof=seed[0][3]) == ()
    finally:
        authority.close()

    (tmp_path / "pre-open-tamper").mkdir()
    _, tampered_args, _ = _seed(tmp_path / "pre-open-tamper")
    _rewrite_retained_relationship_evidence(tampered_args["database"])
    with pytest.raises(ValueError, match="relationship|lineage"):
        open_event_hypothesis_lineage_authority(**tampered_args)


def test_open_lineage_reads_reverify_retained_relationship_history(tmp_path) -> None:
    seed, args, _ = _seed(tmp_path)
    authority = open_event_hypothesis_lineage_authority(**args)
    try:
        _rewrite_retained_relationship_evidence(args["database"])
        with pytest.raises(ValueError, match="relationship|lineage"):
            authority.history()
        with pytest.raises(ValueError, match="relationship|lineage"):
            authority.current_heads(proof=seed[0][3])
    finally:
        authority.close()


def test_rollback_scope_reverifies_retained_relationship_history(tmp_path) -> None:
    from newsroom.authority._event_hypothesis_lineage_system import (
        _open_unlocked_lineage_authority_for_test,
    )

    _, args, _ = _seed(tmp_path)
    authority = _open_unlocked_lineage_authority_for_test(**args)
    observed: list[tuple[HypothesisLineageReceipt, ...]] = []
    try:
        authority.rollback_scope(
            lambda transaction: observed.append(transaction.history())
        )
        assert observed == [()]

        _rewrite_retained_relationship_evidence(args["database"])
        with pytest.raises(ValueError, match="relationship|lineage"):
            authority.rollback_scope(lambda transaction: transaction.history())
    finally:
        authority.close()


def test_composed_lineage_reuses_exact_transaction_inputs_but_replays_local_rows(
    tmp_path, monkeypatch
) -> None:
    from newsroom.authority import _event_hypothesis_relationship_system as relations

    seed, args, receipt = _seed(tmp_path)
    authority = open_event_hypothesis_lineage_authority(**args)
    try:
        authority.retain(receipt.canonical_bytes, proof=seed[0][3])
        store = authority._EventHypothesisLineageAuthority__authority._store
        calls = []
        original = relations._verify_relationship_reads_in_transaction

        def capture_inputs(*args, **kwargs):
            values = original(*args, **kwargs)
            calls.append(values)
            return values

        monkeypatch.setattr(
            relations, "_verify_relationship_reads_in_transaction", capture_inputs
        )
        with store._transaction():
            expected = store._verify()
            assert len(calls) == 1
            inputs = calls[0]
            reused = store._verify(relationship_inputs=inputs)
            assert (*reused[:3], *reused[4:]) == (*expected[:3], *expected[4:])
            assert tuple((p.assessment, p.evidence) for p in reused[3]) == tuple(
                (p.assessment, p.evidence) for p in expected[3]
            )
            assert expected[0] == (receipt,)
            assert len(calls) == 1

            # Missing upstream records do not become empty/default lineage inputs.
            versions, relationships = inputs
            missing_versions = dict(versions)
            del missing_versions[receipt.inputs[0].version_id]
            missing_relationships = dict(relationships)
            del missing_relationships[receipt.relationships[0].assessment_digest]
            for incomplete in (
                (missing_versions, relationships),
                (versions, missing_relationships),
            ):
                with pytest.raises(ValueError, match="relationship input is absent"):
                    store._verify(relationship_inputs=incomplete)

            # Reuse covers upstream values only: local replay/head checks still run.
            store._connection.execute("SAVEPOINT head_tamper")
            try:
                store._connection.execute(
                    "UPDATE event_hypothesis_lineage_heads SET generation=99"
                )
                with pytest.raises(AuthoritySchemaError, match="materialised heads"):
                    store._verify(relationship_inputs=inputs)
            finally:
                store._connection.execute("ROLLBACK TO head_tamper")
                store._connection.execute("RELEASE head_tamper")

        # Independent calls never reuse the previous transaction's values.
        assert authority.history() == (receipt,)
        assert authority.history() == (receipt,)
        assert len(calls) == 3
        _rewrite_retained_relationship_evidence(args["database"])
        with pytest.raises(ValueError, match="relationship|lineage"):
            authority.history()
    finally:
        authority.close()


def test_retain_replay_reopen_and_guarded_heads(tmp_path) -> None:
    seed, args, receipt = _seed(tmp_path)
    authority = open_event_hypothesis_lineage_authority(**args)
    assert type(authority) is EventHypothesisLineageAuthority
    try:
        retained = authority.retain(receipt.canonical_bytes, proof=seed[0][3])
        assert retained == receipt
        connection = sqlite3.connect(seed[1])
        aggregate_id, idempotency_key = connection.execute(
            "SELECT l.authority_aggregate_id,c.idempotency_key "
            "FROM event_hypothesis_lineage l JOIN ledger_events e "
            "ON e.event_id=l.authority_event_id JOIN authority_commands c "
            "ON c.command_id=e.command_id WHERE l.lineage_id=?",
            (receipt.lineage_id,),
        ).fetchone()
        connection.close()
        from newsroom.authority._event_hypothesis_lineage_system import (
            _lineage_aggregate_id,
        )

        assert aggregate_id == str(_lineage_aggregate_id(receipt.lineage_id))
        assert idempotency_key == receipt.lineage_id
        assert (
            authority.create_or_replay(receipt.canonical_bytes, proof=seed[0][3])
            == receipt
        )
        assert authority.load(receipt.lineage_id) == receipt
        assert authority.history() == (receipt,)
        heads = authority.current_heads(proof=seed[0][3])
        assert tuple(head.node.version_id for head in heads) == (
            receipt.outputs[0].version_id,
        )
    finally:
        authority.close()

    reopened = open_event_hypothesis_lineage_authority(**args)
    try:
        assert reopened.history() == (receipt,)
    finally:
        reopened.close()

    connection = sqlite3.connect(seed[1], isolation_level=None)
    row = connection.execute("SELECT * FROM event_hypothesis_lineage_heads").fetchone()
    connection.execute("DELETE FROM event_hypothesis_lineage_heads")
    connection.execute(
        "INSERT INTO event_hypothesis_lineage_heads VALUES(?,?,?,?,?,?)",
        (row[0], row[1], "sha256:" + "f" * 64, row[3], row[4], row[5]),
    )
    connection.close()
    with pytest.raises((AuthoritySchemaError, ValueError), match="heads|lineage"):
        open_event_hypothesis_lineage_authority(**args)


def test_lineage_read_rejects_exact_authority_provenance_tamper(tmp_path) -> None:
    seed, args, receipt = _seed(tmp_path)
    authority = open_event_hypothesis_lineage_authority(**args)
    authority.retain(receipt.canonical_bytes, proof=seed[0][3])
    connection = sqlite3.connect(seed[1], isolation_level=None)
    try:
        connection.execute("DROP TRIGGER immutable_authorization_requests_update")
        connection.execute(
            "UPDATE authorization_requests SET storage_request_residual=? WHERE "
            "request_digest=(SELECT e.authorization_request_digest FROM "
            "event_hypothesis_lineage l JOIN ledger_events e "
            "ON e.event_id=l.authority_event_id WHERE l.lineage_id=?)",
            (b"{}", receipt.lineage_id),
        )
    finally:
        connection.close()
    try:
        with pytest.raises(ValueError, match="lineage load failed"):
            authority.load(receipt.lineage_id)
    finally:
        authority.close()


def test_lineage_reopen_rejects_unrelated_authority_corruption(tmp_path) -> None:
    seed, args, receipt = _seed(tmp_path)
    authority = open_event_hypothesis_lineage_authority(**args)
    authority.retain(receipt.canonical_bytes, proof=seed[0][3])
    authority.close()
    connection = sqlite3.connect(seed[1], isolation_level=None)
    try:
        trigger_sql = str(
            connection.execute(
                "SELECT sql FROM sqlite_schema WHERE type='trigger' AND name=?",
                ("immutable_authority_payloads_update",),
            ).fetchone()[0]
        )
        connection.execute("DROP TRIGGER immutable_authority_payloads_update")
        connection.execute(
            "UPDATE authority_payloads SET payload_bytes=? WHERE payload_id=("
            "SELECT p.payload_id FROM authority_payloads p "
            "WHERE p.payload_bytes IS NOT NULL AND NOT EXISTS("
            "SELECT 1 FROM event_hypothesis_lineage l JOIN ledger_events e "
            "ON e.event_id=l.authority_event_id WHERE e.payload_id=p.payload_id) "
            "LIMIT 1)",
            (b"{}",),
        )
        connection.execute(trigger_sql)
    finally:
        connection.close()

    with pytest.raises(ValueError, match="lineage authority open failed"):
        open_event_hypothesis_lineage_authority(**args)


def test_trigger_preserving_fk_clean_aggregate_rewrite_fails_closed(
    tmp_path,
) -> None:
    from newsroom.authority._event_hypothesis_lineage_system import (
        _lineage_aggregate_id,
    )

    seed, args, receipt = _seed(tmp_path)
    authority = open_event_hypothesis_lineage_authority(**args)
    authority.retain(receipt.canonical_bytes, proof=seed[0][3])
    authority.close()

    connection = sqlite3.connect(seed[1], isolation_level=None)
    connection.row_factory = sqlite3.Row
    expected = str(_lineage_aggregate_id(receipt.lineage_id))
    rewritten = str(uuid.uuid4())
    lineage = connection.execute(
        "SELECT * FROM event_hypothesis_lineage WHERE lineage_id=?",
        (receipt.lineage_id,),
    ).fetchone()
    event = connection.execute(
        "SELECT * FROM ledger_events WHERE event_id=?",
        (lineage["authority_event_id"],),
    ).fetchone()
    command = connection.execute(
        "SELECT * FROM authority_commands WHERE command_id=?",
        (event["command_id"],),
    ).fetchone()
    version = connection.execute(
        "SELECT * FROM authority_aggregate_versions WHERE command_id=?",
        (event["command_id"],),
    ).fetchone()
    aggregate = connection.execute(
        "SELECT * FROM authority_aggregates WHERE aggregate_type=? AND aggregate_id=?",
        (event["aggregate_type"], expected),
    ).fetchone()
    triggers = connection.execute(
        "SELECT name,sql FROM sqlite_schema WHERE type='trigger' ORDER BY name"
    ).fetchall()
    from newsroom.authority.command_bound_storage import command_request_bytes, validated_command_result_bytes
    from newsroom.authority import AuthorityPersistenceError
    original_result_bytes = validated_command_result_bytes(connection, command)
    original_request_row = connection.execute(
        "SELECT * FROM authorization_requests WHERE request_digest=?",
        (command["authorization_request_digest"],),
    ).fetchone()
    original_request_bytes = command_request_bytes(connection, original_request_row)

    def rewritten_row(row: sqlite3.Row, **changes: object) -> tuple[object, ...]:
        return tuple(
            changes.get(key, row[key])
            for key in row.keys()  # noqa: SIM118
        )

    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("BEGIN IMMEDIATE")
    connection.execute("PRAGMA defer_foreign_keys=ON")
    connection.execute(
        "CREATE UNIQUE INDEX aggregate_rewrite_gate "
        "ON authority_aggregates(aggregate_type) "
        "WHERE aggregate_type='event_hypothesis_lineage'"
    )
    connection.execute(
        "INSERT OR REPLACE INTO authority_aggregates VALUES(?,?,?,?,?)",
        rewritten_row(aggregate, aggregate_id=rewritten),
    )
    connection.execute(
        "INSERT OR REPLACE INTO authority_aggregate_versions VALUES(?,?,?,?,?,?,?)",
        rewritten_row(version, aggregate_id=rewritten),
    )
    connection.execute(
        "INSERT OR REPLACE INTO ledger_events VALUES("
        + ",".join("?" for _ in event.keys())  # noqa: SIM118
        + ")",
        rewritten_row(event, aggregate_id=rewritten),
    )
    connection.execute(
        "INSERT OR REPLACE INTO event_hypothesis_lineage VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        rewritten_row(lineage, authority_aggregate_id=rewritten),
    )
    connection.execute("DROP INDEX aggregate_rewrite_gate")
    connection.execute("COMMIT")

    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert (
        connection.execute(
            "SELECT name,sql FROM sqlite_schema WHERE type='trigger' ORDER BY name"
        ).fetchall()
        == triggers
    )
    assert command["aggregate_id"] == expected
    assert expected.encode() in original_result_bytes
    request = connection.execute(
        "SELECT storage_request_residual FROM authorization_requests "
        "WHERE request_digest=?",
        (command["authorization_request_digest"],),
    ).fetchone()
    assert bytes(request[0]) == bytes(original_request_row["storage_request_residual"])
    assert expected.encode() in original_request_bytes
    connection.close()

    with pytest.raises((AuthorityPersistenceError, ValueError), match="aggregate|lineage"):
        open_event_hypothesis_lineage_authority(**args)


def test_private_rollback_scope_exposes_real_uncommitted_v23_rows(tmp_path) -> None:
    from newsroom.authority._event_hypothesis_lineage_system import (
        _open_unlocked_lineage_authority_for_test,
    )

    seed, args, receipt = _seed(tmp_path)
    authority = _open_unlocked_lineage_authority_for_test(**args)
    observed: list[tuple[HypothesisLineageReceipt, ...]] = []
    try:

        def inspect(transaction: object) -> None:
            transaction.retain(receipt.canonical_bytes, proof=seed[0][3])  # type: ignore[attr-defined]
            observed.append(transaction.history())  # type: ignore[attr-defined]

        authority.rollback_scope(inspect)
        assert observed == [(receipt,)]
        assert authority.history() == ()
    finally:
        authority.close()


def test_v22_to_v23_preserves_retained_relationship_before_lineage_use(
    tmp_path,
) -> None:
    from newsroom.authority.event_hypothesis_lineage_migrations import (
        event_hypothesis_lineage_backup_paths,
    )
    from newsroom.authority.migrations import (
        SCHEMA_VERSION,
        apply_pending_migrations,
        prepare_pending_migration_backup,
    )
    from newsroom.authority.story_candidate_migrations import (
        story_candidate_backup_paths,
    )
    from newsroom.tests.graphiti_adapter_4d_migration_helpers import (
        drop_empty_v23_lineage_schema,
    )

    seed, args, receipt = _seed(tmp_path)
    connection = sqlite3.connect(seed[1], isolation_level=None)
    before = (
        connection.execute(
            "SELECT * FROM event_hypothesis_relationship_decisions ORDER BY decision_id"
        ).fetchall(),
        connection.execute("SELECT * FROM ledger_events ORDER BY event_id").fetchall(),
    )
    drop_empty_v23_lineage_schema(connection)
    assert connection.execute("PRAGMA user_version").fetchone() == (22,)
    for path in event_hypothesis_lineage_backup_paths(seed[1]):
        path.unlink(missing_ok=True)
    for path in story_candidate_backup_paths(seed[1]):
        path.unlink(missing_ok=True)
    from newsroom.authority.bounded_search_migrations import (
        bounded_search_backup_paths,
    )
    from newsroom.authority.coverage_audit_migrations import (
        coverage_audit_backup_paths,
    )
    from newsroom.authority.evaluation_feedback_migrations import (
        evaluation_feedback_backup_paths,
    )
    from newsroom.authority.increment8_evaluation_migrations import (
        increment8_evaluation_backup_paths,
    )
    from newsroom.authority.increment8_operational_migrations import (
        increment8_operational_backup_paths,
    )
    from newsroom.authority.increment8_recovery_migrations import (
        increment8_recovery_backup_paths,
    )
    from newsroom.authority.graphiti_accounted_zero_migrations import (
        graphiti_accounted_zero_backup_paths,
    )
    from newsroom.authority.graphiti_evaluation_migrations import (
        graphiti_evaluation_backup_paths,
    )
    from newsroom.authority.local_watch_migrations import local_watch_backup_paths
    from newsroom.authority.planned_agenda_migrations import (
        planned_agenda_backup_paths,
    )

    for path in evaluation_feedback_backup_paths(seed[1]):
        path.unlink(missing_ok=True)
    for path in planned_agenda_backup_paths(seed[1]):
        path.unlink(missing_ok=True)
    for path in bounded_search_backup_paths(seed[1]):
        path.unlink(missing_ok=True)
    for path in coverage_audit_backup_paths(seed[1]):
        path.unlink(missing_ok=True)
    for path in local_watch_backup_paths(seed[1]):
        path.unlink(missing_ok=True)
    for path in increment8_evaluation_backup_paths(seed[1]):
        path.unlink(missing_ok=True)
    for path in increment8_operational_backup_paths(seed[1]):
        path.unlink(missing_ok=True)
    for path in increment8_recovery_backup_paths(seed[1]):
        path.unlink(missing_ok=True)
    for path in graphiti_evaluation_backup_paths(seed[1]):
        path.unlink(missing_ok=True)
    for path in graphiti_accounted_zero_backup_paths(seed[1]):
        path.unlink(missing_ok=True)
    assert prepare_pending_migration_backup(connection) is not None
    apply_pending_migrations(connection, applied_at="2042-01-03T00:00:00.000000Z")
    assert connection.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
    assert (
        connection.execute(
            "SELECT * FROM event_hypothesis_relationship_decisions ORDER BY decision_id"
        ).fetchall(),
        connection.execute("SELECT * FROM ledger_events ORDER BY event_id").fetchall(),
    ) == before
    connection.close()
    authority = open_event_hypothesis_lineage_authority(**args)
    try:
        assert authority.retain(receipt.canonical_bytes, proof=seed[0][3]) == receipt
    finally:
        authority.close()


def test_disjoint_lineages_preserve_each_head_producer_time(tmp_path) -> None:
    seed, args, first = _seed(tmp_path)
    output, left, right = seed[3][3:6]
    inputs = tuple(sorted((left, right), key=lambda item: item.version_id))
    assessment, evidence = d3._decision(output, inputs, CanonicalOutcome.REL_SAME_STATE)
    relationship = open_event_hypothesis_relationship_authority(**args)
    try:
        relationship.retain(
            assessment.canonical_bytes,
            tuple(item.canonical_bytes for item in evidence),
            proof=seed[0][3],
        )
    finally:
        relationship.close()
    second = HypothesisLineageReceipt.consolidation(
        expected_generation=0,
        inputs=inputs,
        output=output,
        relationship_proofs=(
            HypothesisLineageRelationshipProof.from_assessment(assessment, evidence),
        ),
    )
    authority = open_event_hypothesis_lineage_authority(**args)
    try:
        authority.retain(first.canonical_bytes, proof=seed[0][3])
    finally:
        authority.close()
    args["clock"] = lambda: d2.UtcTimestamp.parse("2042-01-03T00:00:00.000000Z")
    authority = open_event_hypothesis_lineage_authority(**args)
    try:
        authority.retain(second.canonical_bytes, proof=seed[0][3])
        assert {
            head.node.version_id for head in authority.current_heads(proof=seed[0][3])
        } == {
            first.outputs[0].version_id,
            second.outputs[0].version_id,
        }
    finally:
        authority.close()


def test_candidate_closure_keeps_untouched_split_sibling_current(tmp_path) -> None:
    from newsroom.authority._event_hypothesis_lineage_system import (
        _create_event_hypothesis_lineage_read_port,
    )
    from newsroom.increment6.lineage import merge_lineage_authority_registries
    from newsroom.increment6.relationships import (
        merge_relationship_authority_registries,
    )

    cached_seed = d2._SEED_CACHE
    d2._SEED_CACHE = None
    try:
        seed = d2._seed_location(tmp_path / "partial-component", subjects=7)
    finally:
        d2._SEED_CACHE = cached_seed
    args = d2._open_arguments(seed)
    args["authorizer"] = _seed_authorizer()
    commands, schemas = merge_relationship_authority_registries(
        args["command_registry"], args["payload_schemas"]
    )
    commands, schemas = merge_lineage_authority_registries(commands, schemas)
    checked = {**args, "command_registry": commands, "payload_schemas": schemas}
    source, left, untouched, other_source, extra, other_untouched, combined = (
        seed[3][:7]
    )
    relationships = open_event_hypothesis_relationship_authority(**checked)
    try:
        split_receipts = []
        split_assessment = None
        for split_source, outputs in (
            (source, (left, untouched)),
            (other_source, (extra, other_untouched)),
        ):
            split_proofs = []
            for output in outputs:
                other = outputs[1] if output is outputs[0] else outputs[0]
                assessment, evidence = d3._decision(
                    output,
                    tuple(
                        sorted(
                            (split_source, other),
                            key=lambda item: item.version_id,
                        )
                    ),
                    CanonicalOutcome.REL_RELATED_DISTINCT,
                )
                relationships.retain(
                    assessment.canonical_bytes,
                    tuple(item.canonical_bytes for item in evidence),
                    proof=seed[0][3],
                )
                split_proofs.append(
                    HypothesisLineageRelationshipProof.from_assessment(
                        assessment, evidence
                    )
                )
                if output is untouched:
                    split_assessment = assessment
            split_receipts.append(
                HypothesisLineageReceipt.split(
                    expected_generation=0,
                    source=split_source,
                    outputs=outputs,
                    relationship_proofs=tuple(split_proofs),
                )
            )
        merge_inputs = tuple(sorted((left, extra), key=lambda item: item.version_id))
        merge_assessment, merge_evidence = d3._decision(
            combined, merge_inputs, CanonicalOutcome.REL_SAME_STATE
        )
        relationships.retain(
            merge_assessment.canonical_bytes,
            tuple(item.canonical_bytes for item in merge_evidence),
            proof=seed[0][3],
        )
    finally:
        relationships.close()
    consolidation = HypothesisLineageReceipt.consolidation(
        expected_generation=1,
        inputs=merge_inputs,
        output=combined,
        relationship_proofs=(
            HypothesisLineageRelationshipProof.from_assessment(
                merge_assessment, merge_evidence
            ),
        ),
    )
    authority = open_event_hypothesis_lineage_authority(**checked)
    try:
        for split in split_receipts:
            authority.retain(split.canonical_bytes, proof=seed[0][3])
        authority.retain(consolidation.canonical_bytes, proof=seed[0][3])
    finally:
        authority.close()
    assert split_assessment is not None
    connection = sqlite3.connect(seed[1], isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    port = _create_event_hypothesis_lineage_read_port(
        connection,
        retrieval_authority=checked["retrieval_authority"],
        authenticator=checked["authenticator"],
        command_registry=commands,
        payload_schemas=schemas,
        clock=checked["clock"],
    )
    connection.execute("BEGIN IMMEDIATE")
    try:
        snapshot, _, _ = port._require_candidate_inputs_in_transaction(
            untouched.version_id,
            split_assessment.canonical_digest,
            proof=seed[0][3],
        )
        assert snapshot.subject == untouched
        assert {head.node.version_id for head in snapshot.replay.active_heads} == {
            untouched.version_id,
            other_untouched.version_id,
            combined.version_id,
        }
        with pytest.raises(ValueError, match="lineage|stale"):
            port._require_candidate_inputs_in_transaction(
                source.version_id,
                split_assessment.canonical_digest,
                proof=seed[0][3],
            )
    finally:
        connection.execute("ROLLBACK")
    trigger = connection.execute(
        "SELECT sql FROM sqlite_master WHERE name="
        "'event_hypothesis_lineage_head_update_guard'"
    ).fetchone()[0]
    connection.execute("DROP TRIGGER event_hypothesis_lineage_head_update_guard")
    connection.execute(
        "UPDATE event_hypothesis_lineage_heads SET producing_lineage_id=? "
        "WHERE version_id=?",
        (split_receipts[1].lineage_id, untouched.version_id),
    )
    connection.execute(trigger)
    connection.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(ValueError, match="lineage|stale"):
            port._require_candidate_inputs_in_transaction(
                untouched.version_id,
                split_assessment.canonical_digest,
                proof=seed[0][3],
            )
    finally:
        connection.execute("ROLLBACK")
        connection.close()


def test_semantic_replay_rejects_divergent_output_and_actor(tmp_path) -> None:
    seed, args, receipt = _seed(tmp_path)
    divergent_output = seed[3][3]
    inputs = tuple(sorted(seed[3][1:3], key=lambda item: item.version_id))
    assessment, evidence = d3._decision(
        divergent_output,
        inputs,
        CanonicalOutcome.REL_SAME_STATE,
    )
    relationship = open_event_hypothesis_relationship_authority(**args)
    try:
        relationship.retain(
            assessment.canonical_bytes,
            tuple(item.canonical_bytes for item in evidence),
            proof=seed[0][3],
        )
    finally:
        relationship.close()
    divergent = HypothesisLineageReceipt.consolidation(
        expected_generation=0,
        inputs=inputs,
        output=divergent_output,
        relationship_proofs=(
            HypothesisLineageRelationshipProof.from_assessment(assessment, evidence),
        ),
    )
    assert divergent.lineage_id == receipt.lineage_id
    authority = open_event_hypothesis_lineage_authority(**args)
    try:
        authority.retain(receipt.canonical_bytes, proof=seed[0][3])
        with pytest.raises(ValueError, match="diverge|retention"):
            authority.retain(divergent.canonical_bytes, proof=seed[0][3])
        with pytest.raises(ValueError, match="diverge|retention"):
            authority.retain(
                receipt.canonical_bytes,
                proof=d2.AuthenticationProof(method="STATIC_TOKEN", credential="other"),
            )
        assert authority.history() == (receipt,)
    finally:
        authority.close()


def test_lineage_replay_rejects_changed_exact_input_version(tmp_path) -> None:
    seed, args, receipt = _seed(tmp_path)
    authority = open_event_hypothesis_lineage_authority(**args)
    authority.retain(receipt.canonical_bytes, proof=seed[0][3])
    connection = sqlite3.connect(seed[1], isolation_level=None)
    trigger = connection.execute(
        "SELECT sql FROM sqlite_master WHERE name="
        "'immutable_event_hypothesis_version_update'"
    ).fetchone()[0]
    connection.execute("DROP TRIGGER immutable_event_hypothesis_version_update")
    connection.execute(
        "UPDATE event_hypothesis_versions_v2 SET canonical_bytes=? WHERE version_id=?",
        (b"{}", receipt.inputs[0].version_id),
    )
    connection.execute(trigger)
    connection.close()
    with pytest.raises(ValueError, match="lineage|retention"):
        authority.retain(receipt.canonical_bytes, proof=seed[0][3])
    authority.close()


def test_lineage_rejects_second_consumption_of_generation_zero_input(tmp_path) -> None:
    seed, args, first = _seed(tmp_path)
    consumed = next(
        version
        for version in seed[3]
        if version.version_id == first.inputs[0].version_id
    )
    other = seed[3][3]
    output = seed[3][4]
    inputs = tuple(sorted((consumed, other), key=lambda item: item.version_id))
    assessment, evidence = d3._decision(
        output, inputs, CanonicalOutcome.REL_SAME_STATE
    )
    relationships = open_event_hypothesis_relationship_authority(**args)
    try:
        relationships.retain(
            assessment.canonical_bytes,
            tuple(item.canonical_bytes for item in evidence),
            proof=seed[0][3],
        )
    finally:
        relationships.close()
    second = HypothesisLineageReceipt.consolidation(
        expected_generation=0,
        inputs=inputs,
        output=output,
        relationship_proofs=(
            HypothesisLineageRelationshipProof.from_assessment(assessment, evidence),
        ),
    )
    authority = open_event_hypothesis_lineage_authority(**args)
    try:
        authority.retain(first.canonical_bytes, proof=seed[0][3])
        with pytest.raises(ValueError, match="lineage|current|stale"):
            authority.retain(second.canonical_bytes, proof=seed[0][3])
    finally:
        authority.close()


def test_empty_component_rejects_forged_generation_zero_materialised_head(
    tmp_path,
) -> None:
    from newsroom.authority._event_hypothesis_lineage_system import _LineageStore

    seed, _, receipt = _seed(tmp_path)
    node = receipt.inputs[0]
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.execute(
        "CREATE TABLE event_hypothesis_lineage_heads("
        "hypothesis_id,version_id,version_digest,generation,"
        "producing_lineage_id,updated_at)"
    )
    connection.execute(
        "INSERT INTO event_hypothesis_lineage_heads VALUES(?,?,?,?,?,?)",
        (
            node.hypothesis_id,
            node.version_id,
            node.version_digest,
            0,
            "forged-lineage",
            "2042-01-01T00:00:00.000000Z",
        ),
    )
    store = object.__new__(_LineageStore)
    store._conn = connection
    store._closed = False
    replay = type(
        "Replay", (), {"active_heads": (HypothesisLineageHead(node, 0),)}
    )()
    with pytest.raises(ValueError, match="materialised heads"):
        store._require_materialised_heads((), replay)
    connection.close()


@pytest.mark.parametrize("field", ("generation", "timestamp"))
def test_head_scalar_and_timestamp_tamper_fail_closed(tmp_path, field: str) -> None:
    root = tmp_path / field
    root.mkdir()
    seed, args, receipt = _seed(root)
    authority = open_event_hypothesis_lineage_authority(**args)
    authority.retain(receipt.canonical_bytes, proof=seed[0][3])
    authority.close()
    connection = sqlite3.connect(seed[1], isolation_level=None)
    row = connection.execute("SELECT * FROM event_hypothesis_lineage_heads").fetchone()
    connection.execute("DELETE FROM event_hypothesis_lineage_heads")
    values = list(row)
    values[3 if field == "generation" else 5] = (
        int(row[3]) + 1 if field == "generation" else "2042-01-09T00:00:00.000000Z"
    )
    connection.execute(
        "INSERT INTO event_hypothesis_lineage_heads VALUES(?,?,?,?,?,?)", values
    )
    connection.close()
    with pytest.raises((AuthoritySchemaError, ValueError), match="heads|lineage"):
        open_event_hypothesis_lineage_authority(**args)


def test_historical_reads_and_exact_replay_ignore_advanced_input_currentness(
    tmp_path,
) -> None:
    seed, args, receipt = _seed(tmp_path)
    authority = open_event_hypothesis_lineage_authority(**args)
    authority.retain(receipt.canonical_bytes, proof=seed[0][3])
    authority.close()
    connection = sqlite3.connect(seed[1], isolation_level=None)
    fixture = (connection, *seed[0][1:])
    proposal, dispositions = seed[7]["authority"]
    upstream = d2.hypothesis_helpers._open(fixture)
    try:
        upstream.retain(
            proposal,
            dispositions,
            proof=seed[0][3],
            expected_target_version=seed[3][0],
        )
    finally:
        upstream.close()
        connection.close()
    reopened = open_event_hypothesis_lineage_authority(**args)
    try:
        assert reopened.load(receipt.lineage_id) == receipt
        assert reopened.history() == (receipt,)
        assert reopened.retain(receipt.canonical_bytes, proof=seed[0][3]) == receipt
        with pytest.raises(ValueError, match="current heads failed"):
            reopened.current_heads(proof=seed[0][3])
    finally:
        reopened.close()


def test_reversal_target_retarget_fails_closed(tmp_path) -> None:
    from newsroom.increment6.lineage import merge_lineage_authority_registries
    from newsroom.increment6.relationships import (
        merge_relationship_authority_registries,
    )

    cached_seed = d2._SEED_CACHE
    d2._SEED_CACHE = None
    try:
        seed = d2._seed_location(tmp_path / "reversal-store", subjects=3)
    finally:
        d2._SEED_CACHE = cached_seed
    args = d2._open_arguments(seed)
    args["authorizer"] = _seed_authorizer()
    commands, schemas = merge_relationship_authority_registries(
        args["command_registry"], args["payload_schemas"]
    )
    commands, schemas = merge_lineage_authority_registries(commands, schemas)
    checked_args = {**args, "command_registry": commands, "payload_schemas": schemas}
    source = seed[3][0]
    outputs = (seed[3][1], seed[3][2])
    split_proofs = []
    relationship = open_event_hypothesis_relationship_authority(**checked_args)
    try:
        for output in outputs:
            comparators = tuple(
                sorted(
                    (source, *(item for item in outputs if item != output)),
                    key=lambda item: item.version_id,
                )
            )
            assessment, evidence = d3._decision(
                output, comparators, CanonicalOutcome.REL_RELATED_DISTINCT
            )
            relationship.retain(
                assessment.canonical_bytes,
                tuple(item.canonical_bytes for item in evidence),
                proof=seed[0][3],
            )
            split_proofs.append(
                HypothesisLineageRelationshipProof.from_assessment(assessment, evidence)
            )
    finally:
        relationship.close()
    split = HypothesisLineageReceipt.split(
        expected_generation=0,
        source=source,
        outputs=outputs,
        relationship_proofs=tuple(split_proofs),
    )
    authority = open_event_hypothesis_lineage_authority(**checked_args)
    try:
        authority.retain(split.canonical_bytes, proof=seed[0][3])

        connection = sqlite3.connect(seed[1], isolation_level=None)
        upstream = d2.hypothesis_helpers._open((connection, *seed[0][1:]))
        proposal, dispositions = seed[7]["authority"]
        try:
            restored = upstream.retain(
                proposal,
                dispositions,
                proof=seed[0][3],
                expected_target_version=source,
            )
        finally:
            upstream.close()
            connection.close()

        comparators = tuple(sorted(outputs, key=lambda item: item.version_id))
        assessment, evidence = d3._decision(
            restored, comparators, CanonicalOutcome.REL_CORRECTION_REVERSAL_OF
        )
        relationship = d2._open_unlocked_relationship_authority_for_test(**checked_args)
        try:
            relationship.retain(
                assessment.canonical_bytes,
                tuple(item.canonical_bytes for item in evidence),
                proof=seed[0][3],
            )
        finally:
            relationship.close()
        reversal = HypothesisLineageReceipt.reversal(
            expected_generation=1,
            target=split,
            outputs=(restored,),
            relationship_proofs=(
                HypothesisLineageRelationshipProof.from_assessment(
                    assessment, evidence
                ),
            ),
        )
        authority.retain(reversal.canonical_bytes, proof=seed[0][3])
    finally:
        authority.close()

    connection = sqlite3.connect(seed[1], isolation_level=None)
    connection.execute("PRAGMA foreign_keys=OFF")
    connection.execute("DROP TRIGGER immutable_event_hypothesis_lineage_update")
    connection.execute(
        "UPDATE event_hypothesis_lineage SET reversal_target_lineage_id=? "
        "WHERE lineage_id=?",
        (str(uuid.uuid4()), reversal.lineage_id),
    )
    connection.close()
    with pytest.raises((AuthoritySchemaError, ValueError), match="lineage"):
        open_event_hypothesis_lineage_authority(**checked_args)


def test_public_facade_normalises_forged_results_and_ordinary_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from newsroom.increment6 import lineage

    class Raw:
        retained: object = object()

        def retain(self, *_: object, **__: object) -> object:
            return self.retained

        def load(self, *_: object, **__: object) -> object:
            raise RuntimeError("ordinary")

        def current_heads(self, *_: object, **__: object) -> object:
            return (object.__new__(lineage.HypothesisLineageHead),)

    raw = Raw()
    facade = lineage._compose_event_hypothesis_lineage_authority(raw)
    with pytest.raises(lineage.HypothesisLineageContractError):
        facade.retain(b"{}", proof=object())
    raw.retained = object.__new__(lineage.HypothesisLineageReceipt)
    with pytest.raises(lineage.HypothesisLineageContractError):
        facade.retain(b"{}", proof=object())
    with pytest.raises(ValueError, match="load failed"):
        facade.load("lineage")
    with pytest.raises(lineage.HypothesisLineageContractError):
        facade.current_heads(proof=object())

    from newsroom.authority import event_hypothesis_lineage_system as system

    monkeypatch.setattr(
        system,
        "open_event_hypothesis_lineage_authority_system",
        lambda *a, **k: object(),
    )
    with pytest.raises(ValueError, match="forged facade"):
        lineage.open_event_hypothesis_lineage_authority(
            "fixture.sqlite3",
            retrieval_authority=object(),
            authenticator=object(),
            authorizer=object(),
            command_registry=object(),  # type: ignore[arg-type]
            payload_schemas=object(),  # type: ignore[arg-type]
        )


def test_candidate_read_port_reconstructs_exact_producer_snapshot() -> None:
    from newsroom.increment6 import lineage

    class Raw:
        def require_producers_in_transaction(self, *_: object, **__: object):
            return object.__new__(lineage.HypothesisLineageProducerSnapshot)

    port = lineage._compose_event_hypothesis_lineage_read_port(Raw())
    with pytest.raises(
        lineage.HypothesisLineageContractError,
        match="lineage require_producers_in_transaction transaction read failed",
    ):
        port.require_current_producers_in_transaction("version", proof=object())

    class CandidateRaw:
        def require_candidate_inputs_in_transaction(
            self, *_: object, **__: object
        ) -> object:
            return (
                object.__new__(lineage.HypothesisLineageProducerSnapshot),
                object(),
                [],
            )

    candidate_port = lineage._compose_event_hypothesis_lineage_read_port(
        CandidateRaw()
    )
    with pytest.raises(
        lineage.HypothesisLineageContractError,
        match="lineage Candidate input transaction read differs",
    ):
        candidate_port._require_candidate_inputs_in_transaction(
            "version", "sha256:" + "0" * 64, proof=object()
        )


def test_candidate_relationship_batch_rejects_untyped_or_wrong_count_results(
) -> None:
    from newsroom.increment6 import lineage

    class Raw:
        value: object = ()

        def require_retained_relationships_in_transaction(self, *_: object):
            return self.value

    raw = Raw()
    port = lineage._compose_event_hypothesis_lineage_read_port(raw)
    with pytest.raises(lineage.HypothesisLineageContractError):
        port.require_retained_relationships_in_transaction([])  # type: ignore[arg-type]
    digest = "sha256:" + "0" * 64
    with pytest.raises(lineage.HypothesisLineageContractError):
        port.require_retained_relationships_in_transaction((digest,))
    raw.value = (object(),)
    with pytest.raises(lineage.HypothesisLineageContractError):
        port.require_retained_relationships_in_transaction((digest,))


def test_second_public_writer_fails_closed_until_first_closes(tmp_path) -> None:
    _, args, _ = _seed(tmp_path)
    first = open_event_hypothesis_lineage_authority(**args)
    try:
        with pytest.raises(ValueError, match="open failed"):
            open_event_hypothesis_lineage_authority(**args)
    finally:
        first.close()
    reopened = open_event_hypothesis_lineage_authority(**args)
    reopened.close()


# #384 adapter: command-field encodings live in retained D2 evidence and every
# observable record/history value is reconstructed through a real v23 receipt.
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import ClassVar

from newsroom.increment6.relationships import (
    ComparatorEvidence,
    ComparatorSetManifest,
    HypothesisVersionBinding,
    assess_relationships,
)
from newsroom.tests.authority_store_conformance import (
    _CURRENT_USE_AUTHORITIES,
    _HISTORICAL_TAMPERS,
    _REPRESENTATION_TAMPERS,
    _REQUEST_BINDING_FIELDS,
    _SCENARIOS,
    _TAMPER_REJECTION_TAMPERS,
    CASE_INVENTORY,
    Applicability,
    AuthorityValue,
    BindingConflict,
    CaseId,
    IntegrityViolation,
    LostResponse,
    StoredAuthorityState,
    TamperKind,
    WriteCommand,
    _current_use_authority,
    _exact_replay,
    _fresh_reopen_digest,
    _fresh_reopen_persistence,
    _fresh_submission,
    _historical_retained_operation,
    _historical_tamper_operation,
    _representation_retained,
    _representation_tamper,
    _request_binding_field,
    _restart_or_migrate,
    _rollback_kind,
    _rollback_sequence,
    _tamper_rejection_kind,
)

_CONFORMANCE_TEMPLATES: dict[tuple[str, ...], tuple[bytes, tuple]] = {}
_CONFORMANCE_BASE_CAPACITY = 6
_CONFORMANCE_BASE_SNAPSHOT: tuple[bytes, tuple] | None = None
_XDIST_SHARED_CACHE_ACTIVE = False
_SHARED_CONFORMANCE_TEMPLATE_KEYS = (
    ("record-1",),
    ("record-1", "record-2"),
    ("record-1", "record-b"),
    ("record-a", "record-b"),
    ("record-rollback-normal", "record-rollback-abort"),
)
_CONFORMANCE_RECORDS_BY_CASE = {
    CaseId.HISTORICAL_READ: ("record-1", "record-2"),
    CaseId.COMPETING_WRITERS: ("record-a", "record-b"),
    CaseId.TRANSACTION_ROLLBACK: (
        "record-rollback-normal",
        "record-rollback-abort",
    ),
    CaseId.TAMPER_REJECTION: ("record-1", "record-b"),
}


def _records_for_conformance_case(case: CaseId) -> tuple[str, ...]:
    return _CONFORMANCE_RECORDS_BY_CASE.get(case, ("record-1",))


def _validated_template_subset(
    template_keys: object,
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(template_keys, tuple) or not template_keys:
        raise ValueError("d3 shared cache template subset")
    if any(not isinstance(key, tuple) for key in template_keys):
        raise ValueError("d3 shared cache template subset")
    expected = tuple(
        key for key in _SHARED_CONFORMANCE_TEMPLATE_KEYS if key in template_keys
    )
    if expected != template_keys or len(expected) != len(set(expected)):
        raise ValueError("d3 shared cache template subset")
    return expected


def _clone_conformance_base(root: Path) -> tuple:
    global _CONFORMANCE_BASE_SNAPSHOT
    if _CONFORMANCE_BASE_SNAPSHOT is None:
        d2._SEED_CACHE = None
        build_root = root.parent / f"base-{uuid.uuid4()}"
        seed = d2._seed_location(
            build_root,
            subjects=_CONFORMANCE_BASE_CAPACITY,
        )
        assert len(seed[3]) == _CONFORMANCE_BASE_CAPACITY
        connection = sqlite3.connect(seed[1], isolation_level=None)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
        _CONFORMANCE_BASE_SNAPSHOT = (seed[1].read_bytes(), seed)
    payload, template_seed = _CONFORMANCE_BASE_SNAPSHOT
    assert len(template_seed[3]) == _CONFORMANCE_BASE_CAPACITY
    root.mkdir(mode=0o700)
    database = root / "relationship-authority.sqlite3"
    database.write_bytes(payload)
    database.chmod(0o600)
    return (template_seed[0], database, *template_seed[2:])


class _ConformanceLocation:
    def __init__(self, seed, records: tuple[str, ...]) -> None:
        self.seed = seed
        self.records = records
        self.outputs = dict(zip(records, seed[3][: len(records)], strict=True))
        root_count = 4 if "record-2" in records else 2
        roots = seed[3][len(records) : len(records) + root_count]
        assert len(self.outputs) == len(records) and len(roots) == root_count
        self.roots = {
            record_id: tuple(
                sorted(
                    roots[:2],
                    key=lambda item: item.version_id,
                )
            )
            for record_id in records
        }
        if "record-2" in records:
            self.roots["record-2"] = tuple(
                sorted(roots[2:4], key=lambda item: item.version_id)
            )


def _seed_conformance_relationships(location: _ConformanceLocation) -> None:
    args = d2._open_arguments(location.seed)
    args["authorizer"] = _seed_authorizer()
    relationship = d2._open_unlocked_relationship_authority_for_test(**args)
    try:
        for record_id in location.records:
            command = d2._generic(record_id)
            assessment, evidence, _ = _conformance_domain(location, command)
            relationship.retain(
                assessment.canonical_bytes,
                tuple(item.canonical_bytes for item in evidence),
                proof=d2._ACTOR_PROOFS[command.actor],
            )
    finally:
        relationship.close()


def _build_conformance_template(root: Path, records: tuple[str, ...]):
    location = _ConformanceLocation(_clone_conformance_base(root), records)
    _seed_conformance_relationships(location)
    connection = sqlite3.connect(location.seed[1], isolation_level=None)
    assert connection.execute(
        "SELECT COUNT(*) FROM event_hypothesis_relationship_decisions"
    ).fetchone() == (len(records),)
    assert connection.execute(
        "SELECT COUNT(*) FROM event_hypothesis_lineage"
    ).fetchone() == (0,)
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    return (location.seed[1].read_bytes(), location.seed)


def _prepare_shared_conformance_cache(
    artifact_root: Path,
    template_keys: tuple[tuple[str, ...], ...],
) -> dict[str, object]:
    global _CONFORMANCE_BASE_SNAPSHOT
    selected = _validated_template_subset(template_keys)
    artifact_root.mkdir(mode=0o700, exist_ok=True)
    d2._SEED_CACHE = None
    _CONFORMANCE_BASE_SNAPSHOT = None
    _CONFORMANCE_TEMPLATES.clear()
    _clone_conformance_base(artifact_root / "base-clone")
    for index, records in enumerate(selected):
        _CONFORMANCE_TEMPLATES[records] = _build_conformance_template(
            artifact_root / f"template-{index}", records
        )
    return {
        "artifact_root": artifact_root,
        "capacity": _CONFORMANCE_BASE_CAPACITY,
        "template_keys": selected,
        "d2_seed_cache": d2._SEED_CACHE,
        "base_snapshot": _CONFORMANCE_BASE_SNAPSHOT,
        "templates": dict(_CONFORMANCE_TEMPLATES),
    }


def _install_shared_conformance_cache(
    bundle: object,
    *,
    expected_template_keys: tuple[tuple[str, ...], ...],
    hydrate_root: Path | None = None,
) -> None:
    global _CONFORMANCE_BASE_SNAPSHOT, _XDIST_SHARED_CACHE_ACTIVE
    selected = _validated_template_subset(expected_template_keys)
    if not isinstance(bundle, dict):
        raise TypeError("d3 shared cache bundle")
    if set(bundle) != {
        "artifact_root",
        "base_snapshot",
        "capacity",
        "d2_seed_cache",
        "template_keys",
        "templates",
    }:
        raise ValueError("d3 shared cache bundle fields")
    if bundle.get("capacity") != _CONFORMANCE_BASE_CAPACITY:
        raise ValueError("d3 shared cache capacity")
    if bundle.get("template_keys") != selected:
        raise ValueError("d3 shared cache template keys")
    seed_cache = bundle.get("d2_seed_cache")
    base_snapshot = bundle.get("base_snapshot")
    templates = bundle.get("templates")
    artifact_root = bundle.get("artifact_root")
    if not isinstance(artifact_root, Path) or not artifact_root.is_absolute():
        raise ValueError("d3 shared cache artifact root")
    if not isinstance(seed_cache, tuple) or len(seed_cache) != 8:
        raise ValueError("d3 shared cache seed")
    if len(seed_cache[3]) != _CONFORMANCE_BASE_CAPACITY:
        raise ValueError("d3 shared cache seed capacity")
    if not isinstance(base_snapshot, tuple) or len(base_snapshot) != 2:
        raise ValueError("d3 shared cache base snapshot")
    if len(base_snapshot[1][3]) != _CONFORMANCE_BASE_CAPACITY:
        raise ValueError("d3 shared cache base capacity")
    if not isinstance(templates, dict) or tuple(templates) != selected:
        raise ValueError("d3 shared cache templates")
    for records, template in templates.items():
        if (
            records not in selected
            or not isinstance(template, tuple)
            or len(template) != 2
            or not isinstance(template[0], bytes)
            or len(template[1][3]) != _CONFORMANCE_BASE_CAPACITY
        ):
            raise ValueError("d3 shared cache template")
    seeds = (base_snapshot[1], *(template[1] for template in templates.values()))
    paths = [seed[1] for seed in seeds]
    retrieval = seed_cache[1][1]
    paths.append(retrieval._path)
    if any(
        not isinstance(path, Path)
        or not path.is_absolute()
        or not path.is_relative_to(artifact_root)
        for path in paths
    ):
        raise ValueError("d3 shared cache ancillary path")
    if hydrate_root is not None:
        hydrate_root.mkdir(mode=0o700)
        hydrated_retrieval = hydrate_root / retrieval._path.name
        hydrated_retrieval.write_bytes(retrieval._path.read_bytes())
        hydrated_retrieval.chmod(0o600)
        retrieval._path = hydrated_retrieval
    d2._SEED_CACHE = seed_cache
    _CONFORMANCE_BASE_SNAPSHOT = base_snapshot
    _CONFORMANCE_TEMPLATES.clear()
    _CONFORMANCE_TEMPLATES.update(templates)
    _XDIST_SHARED_CACHE_ACTIVE = hydrate_root is not None


def _conformance_domain(location: _ConformanceLocation, command: WriteCommand):
    value = command.scalar_columns.get("value")
    record_code = d2._code(d2._RECORD_VALUES, (command.record_id, value), "record")
    base = d2._generic(command.record_id, command.cas_predecessor, value)
    if (
        command.canonical_bytes,
        command.scalar_columns,
        command.identity_columns,
        command.linked_rows,
    ) != (
        base.canonical_bytes,
        base.scalar_columns,
        base.identity_columns,
        base.linked_rows,
    ):
        raise ValueError("unsupported conformance representation")
    request_code = d2._code(d2._REQUEST_VALUES, command.request, "request")
    idempotency_code = d2._code(
        d2._IDEMPOTENCY_VALUES, command.idempotency, "idempotency"
    )
    cas_code = d2._code(d2._CAS_VALUES, command.cas_predecessor, "CAS predecessor")
    upstream_code = d2._code(
        d2._UPSTREAM_VALUES, command.required_upstream_heads, "upstream"
    )
    subject = HypothesisVersionBinding.from_version(location.outputs[command.record_id])
    comparators = tuple(
        HypothesisVersionBinding.from_version(item)
        for item in location.roots[command.record_id]
    )
    evidence = (
        ComparatorEvidence(
            subject,
            comparators[0],
            60 + request_code,
            idempotency_code,
            cas_code,
            80 + record_code,
            upstream_code,
        ),
        ComparatorEvidence(subject, comparators[1], 60, 0, 0, 80, 0),
    )
    assessment = assess_relationships(
        subject, ComparatorSetManifest.complete(comparators), evidence
    )
    proof = HypothesisLineageRelationshipProof.from_assessment(assessment, evidence)
    receipt = HypothesisLineageReceipt.consolidation(
        expected_generation=0,
        inputs=location.roots[command.record_id],
        output=location.outputs[command.record_id],
        relationship_proofs=(proof,),
    )
    return assessment, evidence, receipt


class _ConformanceHandle:
    def __init__(self, location: _ConformanceLocation) -> None:
        from newsroom.authority._event_hypothesis_lineage_system import (
            _open_unlocked_lineage_authority_for_test,
        )

        self.location = location
        self.args = d2._open_arguments(location.seed)
        self.args["authorizer"] = _seed_authorizer()
        try:
            self.authority = _open_unlocked_lineage_authority_for_test(**self.args)
        except (LookupError, RuntimeError, ValueError, sqlite3.DatabaseError) as exc:
            failure = exc

            class FailedAuthority:
                def __getattr__(_, name: str):
                    if name == "close":
                        return lambda: None

                    def failed(*args: object, **kwargs: object) -> object:
                        raise IntegrityViolation("lineage open failed") from failure

                    return failed

            self.authority = FailedAuthority()

    def _row(self, record_id: str, connection=None):
        owned = connection is None
        connection = connection or sqlite3.connect(self.location.seed[1])
        try:
            return connection.execute(
                "SELECT lineage_id FROM event_hypothesis_lineage WHERE lineage_id=?",
                (
                    _conformance_domain(self.location, d2._generic(record_id))[
                        2
                    ].lineage_id,
                ),
            ).fetchone()
        finally:
            if owned:
                connection.close()

    def _decode(self, lineage_id: str, connection=None) -> WriteCommand:
        owned = connection is None
        connection = connection or sqlite3.connect(self.location.seed[1])
        try:
            row = connection.execute(
                "SELECT l.receipt_bytes,a.principal_id FROM event_hypothesis_lineage l JOIN ledger_events e ON e.event_id=l.authority_event_id JOIN authority_commands c ON c.command_id=e.command_id JOIN authentication_contexts a ON a.authentication_context_id=c.authentication_context_id WHERE l.lineage_id=?",
                (lineage_id,),
            ).fetchone()
            if row is None:
                raise ValueError("unknown lineage")
            receipt = HypothesisLineageReceipt.from_canonical_bytes(bytes(row[0]))
            binding = receipt.relationships[0]
            evidence_row = connection.execute(
                "SELECT evidence_bytes FROM event_hypothesis_relationship_decisions WHERE decision_id=?",
                (binding.assessment_digest,),
            ).fetchone()
            values = json.loads(bytes(evidence_row[0]))
            evidence = ComparatorEvidence.from_value(values[0])
            record_code = evidence.same_state_score - 80
            record_id, value = d2._RECORD_VALUES[record_code]
            command = replace(
                d2._generic(
                    record_id, d2._CAS_VALUES[evidence.development_score], value
                ),
                actor=d2._PRINCIPAL_ACTORS[str(row[1])],
                request=d2._REQUEST_VALUES[evidence.score - 60],
                idempotency=d2._IDEMPOTENCY_VALUES[evidence.correction_reversal_score],
                required_upstream_heads=d2._UPSTREAM_VALUES[
                    evidence.related_distinct_score
                ],
            )
            if (
                receipt.outputs[0].version_id
                != self.location.outputs[record_id].version_id
            ):
                raise ValueError("lineage output differs")
            return command
        finally:
            if owned:
                connection.close()

    def submit(
        self, command: WriteCommand, *, lose_response: bool = False
    ) -> AuthorityValue:
        try:
            _, _, receipt = _conformance_domain(self.location, command)
            proof = d2._ACTOR_PROOFS[command.actor]
            retained = self.authority.retain(receipt.canonical_bytes, proof=proof)
            value = AuthorityValue.from_command(self._decode(retained.lineage_id))
        except Exception as exc:
            row = self._row(command.record_id)
            if row is not None:
                try:
                    self.authority.load(str(row[0]))
                except Exception as integrity:
                    raise IntegrityViolation(command.record_id) from integrity
                raise BindingConflict(command.record_id) from exc
            raise BindingConflict(command.record_id) from exc
        if lose_response:
            raise LostResponse(command.record_id)
        return value

    def observe(self, record_id: str):
        row = self._row(record_id)
        if row is None:
            return None
        try:
            self.authority.load(str(row[0]))
            command = self._decode(str(row[0]))
        except Exception as exc:
            raise IntegrityViolation(record_id) from exc
        return (
            StoredAuthorityState.from_command(command)
            if command.record_id == record_id
            else None
        )

    def history(self):
        try:
            receipts = self.authority.history()
        except Exception as exc:
            raise IntegrityViolation("history") from exc
        return tuple(
            AuthorityValue.from_command(self._decode(item.lineage_id))
            for item in receipts
        )

    def read(self, record_id: str):
        state = self.observe(record_id)
        if state is None:
            raise KeyError(record_id)
        return AuthorityValue.from_command(self._decode(str(self._row(record_id)[0])))

    list_history = history

    def set_upstream_head(self, authority: str, value: str) -> None:
        if value.endswith("head-1") or self._row("record-1") is None:
            return
        subject = self.location.outputs["record-1"]
        connection = sqlite3.connect(self.location.seed[1], isolation_level=None)
        try:
            if authority == "authority":
                fixture = (connection, *self.location.seed[0][1:])
                proposal, dispositions = self.location.seed[7]["authority"]
                upstream = d2.hypothesis_helpers._open(fixture)
                try:
                    upstream.retain(
                        proposal,
                        dispositions,
                        proof=self.location.seed[0][3],
                        expected_target_version=subject,
                    )
                finally:
                    upstream.close()
            elif authority == "policy":
                item, current = self.location.seed[7]["policy"]
                successor = replace(
                    d2.hypothesis_helpers.work_item_helpers._version(item, 2),
                    retrieval=current.retrieval,
                )
                d2.TriageWorkItemStore(
                    connection, self.location.seed[0][1]
                ).append_version(
                    current.version_id, current.canonical_digest, successor
                )
        finally:
            connection.close()

    def current_use(self, record_id: str):
        try:
            self.authority.current_heads(proof=self.location.seed[0][3])
        except Exception as exc:
            raise IntegrityViolation(record_id) from exc
        return self.read(record_id)

    def tamper(self, record_id: str, kind: TamperKind) -> None:
        lineage_id = str(self._row(record_id)[0])
        connection = sqlite3.connect(self.location.seed[1], isolation_level=None)
        connection.execute(
            "DROP TRIGGER IF EXISTS immutable_event_hypothesis_lineage_update"
        )
        if kind is TamperKind.CANONICAL:
            connection.execute(
                "UPDATE event_hypothesis_lineage SET receipt_bytes=? WHERE lineage_id=?",
                (b"{}", lineage_id),
            )
        elif kind is TamperKind.SCALAR:
            connection.execute(
                "UPDATE event_hypothesis_lineage SET expected_generation=expected_generation+1 WHERE lineage_id=?",
                (lineage_id,),
            )
        elif kind is TamperKind.IDENTITY:
            connection.execute(
                "UPDATE event_hypothesis_lineage SET kind='HYPOTHESIS_SPLIT' WHERE lineage_id=?",
                (lineage_id,),
            )
        elif kind is TamperKind.LINKED_ROW:
            connection.execute("DELETE FROM event_hypothesis_lineage_heads")
        elif kind is TamperKind.DIGEST:
            connection.execute(
                "UPDATE event_hypothesis_lineage SET receipt_digest=? WHERE lineage_id=?",
                ("sha256:" + "0" * 64, lineage_id),
            )
        elif kind is TamperKind.PROVENANCE:
            connection.execute(
                "UPDATE event_hypothesis_lineage SET actor_identity_digest=? WHERE lineage_id=?",
                ("sha256:" + "0" * 64, lineage_id),
            )
        elif kind is TamperKind.OFFLINE_REWRITE:
            _, _, divergent = _conformance_domain(
                self.location, d2._generic("record-b")
            )
            assert divergent.lineage_id == lineage_id
            assert divergent.canonical_digest != str(
                connection.execute(
                    "SELECT receipt_digest FROM event_hypothesis_lineage "
                    "WHERE lineage_id=?",
                    (lineage_id,),
                ).fetchone()[0]
            )
            updated_at = str(
                connection.execute(
                    "SELECT updated_at FROM event_hypothesis_lineage_heads"
                ).fetchone()[0]
            )
            connection.execute(
                "UPDATE event_hypothesis_lineage SET kind=?,expected_generation=?,"
                "receipt_bytes=?,receipt_digest=?,reversal_target_lineage_id=?,"
                "reversal_target_lineage_digest=? WHERE lineage_id=?",
                (
                    divergent.kind.value,
                    divergent.expected_generation,
                    divergent.canonical_bytes,
                    divergent.canonical_digest,
                    None,
                    None,
                    lineage_id,
                ),
            )
            connection.execute("DELETE FROM event_hypothesis_lineage_heads")
            output = divergent.outputs[0]
            connection.execute(
                "INSERT INTO event_hypothesis_lineage_heads VALUES(?,?,?,?,?,?)",
                (
                    output.hypothesis_id,
                    output.version_id,
                    output.version_digest,
                    divergent.expected_generation + 1,
                    lineage_id,
                    updated_at,
                ),
            )
            ledger_payload = bytes(
                connection.execute(
                    "SELECT p.payload_bytes FROM event_hypothesis_lineage l "
                    "JOIN ledger_events e ON e.event_id=l.authority_event_id "
                    "JOIN authority_payloads p ON p.payload_id=e.payload_id "
                    "WHERE l.lineage_id=?",
                    (lineage_id,),
                ).fetchone()[0]
            )
            assert ledger_payload != divergent.canonical_bytes
        else:  # pragma: no cover - conformance enum exhaustiveness guard
            raise AssertionError(kind)
        connection.close()

    def rollback_scope(self, operation) -> None:
        outer = self

        class Scope:
            def submit(_, command):
                _, _, receipt = _conformance_domain(outer.location, command)
                proof = d2._ACTOR_PROOFS[command.actor]
                retained = transaction.retain(receipt.canonical_bytes, proof=proof)
                return AuthorityValue.from_command(
                    outer._decode(retained.lineage_id, transaction._store._connection)
                )

            def observe(_, record_id):
                row = outer._row(record_id, transaction._store._connection)
                return (
                    None
                    if row is None
                    else StoredAuthorityState.from_command(
                        outer._decode(str(row[0]), transaction._store._connection)
                    )
                )

            def history(_):
                return tuple(
                    AuthorityValue.from_command(
                        outer._decode(item.lineage_id, transaction._store._connection)
                    )
                    for item in transaction.history()
                )

        def inspect(scope):
            nonlocal transaction
            transaction = scope
            operation(Scope())

        transaction = None
        self.authority.rollback_scope(inspect)

    def close(self) -> None:
        self.authority.close()


def _seed_authorizer():
    relationship_scope = relationship_command_definition().required_scope
    lineage_scope = lineage_command_definition().required_scope
    return StaticAuthorizer(
        policy_version="lineage-conformance-v1",
        grants_by_principal={
            "editor": frozenset({relationship_scope, lineage_scope}),
            "other": frozenset({relationship_scope, lineage_scope}),
        },
    )


def _conformance_template(root: Path, records: tuple[str, ...]) -> tuple[bytes, tuple]:
    template = _CONFORMANCE_TEMPLATES.get(records)
    if template is not None:
        return template
    if _XDIST_SHARED_CACHE_ACTIVE:
        raise ValueError("xdist d3 shared cache template miss")
    template = _build_conformance_template(root, records)
    _CONFORMANCE_TEMPLATES[records] = template
    return template


class _LineageAdapter:
    name = "lineage-v23-real"
    applicability: ClassVar = {
        case: Applicability.required() for case in CASE_INVENTORY
    }

    def __init__(self, root: Path, case: CaseId) -> None:
        self.root = root
        self.records = _records_for_conformance_case(case)

    def create_location(self):
        template = _conformance_template(self.root / str(uuid.uuid4()), self.records)
        root = self.root / str(uuid.uuid4())
        root.mkdir(mode=0o700)
        database = root / "relationship-authority.sqlite3"
        database.write_bytes(template[0])
        database.chmod(0o600)
        template_seed = template[1]
        location = _ConformanceLocation(
            (template_seed[0], database, *template_seed[2:]), self.records
        )
        return location

    def open_handle(self, location, *, migrate: bool = False):
        if migrate:
            from newsroom.authority.migrations import (
                SCHEMA_VERSION,
                apply_pending_migrations,
                prepare_pending_migration_backup,
            )

            connection = sqlite3.connect(location.seed[1], isolation_level=None)
            try:
                assert prepare_pending_migration_backup(connection) is None
                apply_pending_migrations(
                    connection, applied_at="2042-01-03T00:00:00.000000Z"
                )
                assert connection.execute("PRAGMA user_version").fetchone() == (
                    SCHEMA_VERSION,
                )
                assert connection.execute(
                    "SELECT MAX(version) FROM authority_migrations"
                ).fetchone() == (SCHEMA_VERSION,)
            finally:
                connection.close()
        return _ConformanceHandle(location)


@dataclass(frozen=True)
class _ConformanceProbe:
    probe_id: str
    case: CaseId
    operation: Callable[[object], None]


_CONFORMANCE_PROBES = (
    _ConformanceProbe(
        "fresh_replay-submission",
        CaseId.FRESH_REPLAY,
        _fresh_submission,
    ),
    _ConformanceProbe(
        "fresh_replay-replay",
        CaseId.FRESH_REPLAY,
        _exact_replay,
    ),
    _ConformanceProbe(
        "fresh_reopen-persistence",
        CaseId.FRESH_REOPEN,
        _fresh_reopen_persistence,
    ),
    _ConformanceProbe(
        "fresh_reopen-digest",
        CaseId.FRESH_REOPEN,
        _fresh_reopen_digest,
    ),
    _ConformanceProbe(
        "representation_binding-retained",
        CaseId.REPRESENTATION_BINDING,
        _representation_retained,
    ),
    *(
        _ConformanceProbe(
            f"representation_binding-{kind.value}",
            CaseId.REPRESENTATION_BINDING,
            partial(_representation_tamper, kind=kind),
        )
        for kind in _REPRESENTATION_TAMPERS
    ),
    *(
        _ConformanceProbe(
            f"request_binding-{field_name}",
            CaseId.REQUEST_BINDING,
            partial(_request_binding_field, field_name=field_name),
        )
        for field_name in _REQUEST_BINDING_FIELDS
    ),
    _ConformanceProbe(
        CaseId.LOST_RESPONSE_REPLAY.value,
        CaseId.LOST_RESPONSE_REPLAY,
        _SCENARIOS[CaseId.LOST_RESPONSE_REPLAY],
    ),
    *(
        _ConformanceProbe(
            f"historical_read-{kind.value}-{phase}-{noun}",
            CaseId.HISTORICAL_READ,
            partial(
                _historical_tamper_operation,
                kind=kind,
                reopened=reopened,
                listing=listing,
            ),
        )
        for kind in _HISTORICAL_TAMPERS
        for phase, reopened in (("fresh", False), ("reopened", True))
        for noun, listing in (
            ("read", False),
            ("list", True),
        )
    ),
    *(
        _ConformanceProbe(
            f"historical_read-retained-{noun}",
            CaseId.HISTORICAL_READ,
            partial(_historical_retained_operation, listing=listing),
        )
        for noun, listing in (("read", False), ("list", True))
    ),
    *(
        _ConformanceProbe(
            f"current_use_revalidation-{authority}",
            CaseId.CURRENT_USE_REVALIDATION,
            partial(_current_use_authority, authority=authority),
        )
        for authority in _CURRENT_USE_AUTHORITIES
    ),
    *(
        _ConformanceProbe(
            f"tamper_rejection-{kind.value}",
            CaseId.TAMPER_REJECTION,
            partial(_tamper_rejection_kind, kind=kind),
        )
        for kind in _TAMPER_REJECTION_TAMPERS
    ),
    _ConformanceProbe(
        CaseId.COMPETING_WRITERS.value,
        CaseId.COMPETING_WRITERS,
        _SCENARIOS[CaseId.COMPETING_WRITERS],
    ),
    *(
        _ConformanceProbe(
            f"transaction_rollback-{label}",
            CaseId.TRANSACTION_ROLLBACK,
            operation,
        )
        for label, operation in (
            ("normal", partial(_rollback_kind, abort=False)),
            ("abort", _rollback_sequence),
        )
    ),
    _ConformanceProbe(
        "restart_migration-restart",
        CaseId.RESTART_MIGRATION,
        partial(_restart_or_migrate, migrate=False),
    ),
    _ConformanceProbe(
        "restart_migration-migrate",
        CaseId.RESTART_MIGRATION,
        partial(_restart_or_migrate, migrate=True),
    ),
)


def _selected_shared_template_keys(
    arguments: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    probe_function = "test_real_v23_store_passes_required_conformance_probe"
    probes = {probe.probe_id: probe for probe in _CONFORMANCE_PROBES}
    selected_cases: set[CaseId] = set()
    conservative = False
    saw_d3 = False
    for argument in arguments:
        if "test_increment6d3_lineage_store.py" not in argument:
            continue
        saw_d3 = True
        tail = argument.split("test_increment6d3_lineage_store.py", 1)[1]
        if not tail.startswith("::"):
            conservative = True
            continue
        node = tail[2:]
        if not node.startswith(probe_function):
            continue
        suffix = node[len(probe_function) :]
        if not suffix:
            conservative = True
            continue
        if not suffix.startswith("[") or not suffix.endswith("]"):
            raise ValueError("unknown exact d3 conformance probe")
        probe_id = suffix[1:-1]
        probe = probes.get(probe_id)
        if probe is None:
            raise ValueError("unknown exact d3 conformance probe")
        selected_cases.add(probe.case)
    if not saw_d3:
        raise ValueError("d3 conformance selection missing")
    if conservative or not selected_cases:
        return _SHARED_CONFORMANCE_TEMPLATE_KEYS
    required = {_records_for_conformance_case(case) for case in selected_cases}
    selected = tuple(
        key for key in _SHARED_CONFORMANCE_TEMPLATE_KEYS if key in required
    )
    return _validated_template_subset(selected)


_EXPECTED_PROBE_IDS = frozenset(
    {
        "fresh_replay-submission",
        "fresh_replay-replay",
        "fresh_reopen-persistence",
        "fresh_reopen-digest",
        "representation_binding-retained",
        "representation_binding-scalar",
        "representation_binding-canonical",
        "representation_binding-identity",
        "representation_binding-linked_row",
        "request_binding-actor",
        "request_binding-request",
        "request_binding-idempotency",
        "request_binding-cas_predecessor",
        "lost_response_replay",
        "historical_read-retained-read",
        "historical_read-retained-list",
        "historical_read-identity-fresh-read",
        "historical_read-identity-fresh-list",
        "historical_read-identity-reopened-read",
        "historical_read-identity-reopened-list",
        "historical_read-digest-fresh-read",
        "historical_read-digest-fresh-list",
        "historical_read-digest-reopened-read",
        "historical_read-digest-reopened-list",
        "historical_read-provenance-fresh-read",
        "historical_read-provenance-fresh-list",
        "historical_read-provenance-reopened-read",
        "historical_read-provenance-reopened-list",
        "current_use_revalidation-authority",
        "current_use_revalidation-policy",
        "tamper_rejection-canonical",
        "tamper_rejection-linked_row",
        "tamper_rejection-offline_rewrite",
        "competing_writers",
        "transaction_rollback-normal",
        "transaction_rollback-abort",
        "restart_migration-restart",
        "restart_migration-migrate",
    }
)


def test_real_v23_conformance_probe_inventory_is_exact_and_unique() -> None:
    probe_ids = tuple(probe.probe_id for probe in _CONFORMANCE_PROBES)
    assert len(probe_ids) == len(set(probe_ids))
    assert frozenset(probe_ids) == _EXPECTED_PROBE_IDS
    assert {probe.case for probe in _CONFORMANCE_PROBES} == set(CASE_INVENTORY)


def test_real_v23_conformance_base_is_immutable_max_capacity(tmp_path) -> None:
    first = _clone_conformance_base(tmp_path / "first")
    assert _CONFORMANCE_BASE_SNAPSHOT is not None
    payload, template_seed = _CONFORMANCE_BASE_SNAPSHOT
    assert len(first[3]) == _CONFORMANCE_BASE_CAPACITY == 6
    assert len(template_seed[3]) == _CONFORMANCE_BASE_CAPACITY

    first[1].write_bytes(b"local clone mutation")
    second = _clone_conformance_base(tmp_path / "second")
    assert second[1] != first[1]
    assert second[1].read_bytes() == payload
    assert len(second[3]) == _CONFORMANCE_BASE_CAPACITY


class _FakeSharedD3:
    def __init__(self, marker: Path) -> None:
        self.marker = marker
        self.installed: list[object] = []

    def _prepare_shared_conformance_cache(
        self, artifact: Path, template_keys: tuple[tuple[str, ...], ...]
    ) -> dict[str, object]:
        with self.marker.open("ab") as stream:
            stream.write(b"x")
        ancillary = artifact / "context.sqlite"
        ancillary.write_bytes(b"context")
        return {"fake": 1, "template_keys": template_keys}

    def _install_shared_conformance_cache(
        self,
        bundle: object,
        *,
        expected_template_keys: tuple[tuple[str, ...], ...],
        hydrate_root: Path | None = None,
    ) -> None:
        assert bundle == {"fake": 1, "template_keys": expected_template_keys}
        assert hydrate_root is None
        self.installed.append(bundle)


def test_d3_shared_cache_has_one_producer_across_four_processes(tmp_path) -> None:
    root = tmp_path / "shared"
    marker = tmp_path / "producers"
    script = """
from pathlib import Path
import sys
from newsroom.tests import conftest as cache

class Fake:
    def _prepare_shared_conformance_cache(self, artifact, template_keys):
        with Path(sys.argv[3]).open("ab") as stream:
            stream.write(b"x")
        (artifact / "context.sqlite").write_bytes(b"context")
        return {"fake": 1, "template_keys": template_keys}
    def _install_shared_conformance_cache(self, bundle, *, expected_template_keys):
        assert bundle == {"fake": 1, "template_keys": expected_template_keys}

cache._ensure_d3_cache(Path(sys.argv[1]), sys.argv[2], Fake())
"""
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                os.fspath(root),
                "run-shared",
                os.fspath(marker),
            ]
        )
        for _ in range(4)
    ]
    assert [process.wait(timeout=30) for process in processes] == [0, 0, 0, 0]
    assert marker.read_bytes() == b"x"
    assert len(tuple(root.glob("artifact-*"))) == 1


def test_d3_shared_cache_run_uid_isolation_and_incomplete_stage_rebuild(
    tmp_path,
) -> None:
    from newsroom.tests import conftest as cache

    marker = tmp_path / "producers"
    first_root = tmp_path / "run-one"
    first_root.mkdir(mode=0o700)
    partial = first_root / "bundle-incomplete.stage"
    partial.write_bytes(b"partial")
    partial.chmod(0o600)
    cache._ensure_d3_cache(first_root, "run-one", _FakeSharedD3(marker))
    second_root = tmp_path / "run-two"
    cache._ensure_d3_cache(second_root, "run-two", _FakeSharedD3(marker))

    assert marker.read_bytes() == b"xx"
    assert (first_root / "manifest.json").is_file()
    assert (second_root / "manifest.json").is_file()
    assert first_root != second_root


def _bound_or_collected_core_inventory() -> tuple[str, ...]:
    inventory_path = os.environ.get(sdlc_lane._CORE_NODE_INVENTORY_PATH_ENV)
    inventory_digest = os.environ.get(sdlc_lane._CORE_NODE_INVENTORY_DIGEST_ENV)
    if (inventory_path is None) != (inventory_digest is None):
        raise ValueError("incomplete bound core inventory")
    return (
        sdlc_lane._collect_core_node_ids(REPO_ROOT)
        if inventory_path is None
        else sdlc_lane._load_core_node_inventory(
            REPO_ROOT, inventory_path, inventory_digest
        )
    )


def test_ten_core_shard_probe_selections_derive_exact_template_subsets() -> None:
    prefix = (
        "newsroom/tests/test_increment6d3_lineage_store.py::"
        "test_real_v23_store_passes_required_conformance_probe"
    )
    inventory = _bound_or_collected_core_inventory()
    shards = sdlc_lane._core_node_shards(inventory)
    probe_ids_by_shard = tuple(
        tuple(
            node_id[len(prefix) + 1 : -1]
            for node_id in shard
            if node_id.startswith(f"{prefix}[")
        )
        for shard in shards
    )
    flattened = tuple(probe_id for shard in probe_ids_by_shard for probe_id in shard)
    assert len(flattened) == len(set(flattened))
    assert frozenset(flattened) == _EXPECTED_PROBE_IDS

    probes = {probe.probe_id: probe for probe in _CONFORMANCE_PROBES}
    records_by_case = {
        CaseId.HISTORICAL_READ: ("record-1", "record-2"),
        CaseId.COMPETING_WRITERS: ("record-a", "record-b"),
        CaseId.TRANSACTION_ROLLBACK: (
            "record-rollback-normal",
            "record-rollback-abort",
        ),
        CaseId.TAMPER_REJECTION: ("record-1", "record-b"),
    }
    d3_shards = tuple(
        shard
        for shard in shards
        if any("test_increment6d3_lineage_store.py" in node_id for node_id in shard)
    )
    selected = tuple(
        _selected_shared_template_keys(tuple(shard)) for shard in d3_shards
    )
    expected = tuple(
        tuple(
            key
            for key in _SHARED_CONFORMANCE_TEMPLATE_KEYS
            if key
            in {
                records_by_case.get(probes[probe_id].case, ("record-1",))
                for probe_id in probe_ids
            }
        )
        if probe_ids
        else _SHARED_CONFORMANCE_TEMPLATE_KEYS
        for shard, probe_ids in zip(shards, probe_ids_by_shard, strict=True)
        if any("test_increment6d3_lineage_store.py" in node_id for node_id in shard)
    )
    assert selected == expected


def test_whole_module_selection_is_conservative_and_unknown_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = "newsroom/tests/test_increment6d3_lineage_store.py"
    assert _selected_shared_template_keys((module,)) == (
        _SHARED_CONFORMANCE_TEMPLATE_KEYS
    )
    probe = "test_real_v23_store_passes_required_conformance_probe"
    with pytest.raises(ValueError, match="unknown exact d3 conformance probe"):
        _selected_shared_template_keys((f"{module}::{probe}[unknown-probe]",))

    monkeypatch.setenv(
        sdlc_lane._CORE_NODE_INVENTORY_PATH_ENV,
        os.fspath(tmp_path / "missing.json"),
    )
    monkeypatch.delenv(sdlc_lane._CORE_NODE_INVENTORY_DIGEST_ENV, raising=False)
    with pytest.raises(ValueError, match="incomplete bound core inventory"):
        _bound_or_collected_core_inventory()


@pytest.mark.parametrize(
    "template_keys",
    (
        (),
        (("record-1",), ("record-1",)),
        (("record-1", "record-2"), ("record-1",)),
        (("record-unknown",),),
        (*_SHARED_CONFORMANCE_TEMPLATE_KEYS, ("record-unknown",)),
    ),
    ids=("missing", "duplicate", "reordered", "unknown", "superset"),
)
def test_d3_template_subset_contract_fails_closed(template_keys) -> None:
    with pytest.raises(ValueError, match="d3 shared cache template subset"):
        _validated_template_subset(template_keys)


def test_expected_template_subset_mismatch_fails_before_unpickle(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from newsroom.tests import conftest as cache

    root = tmp_path / "mismatch"
    selected = (("record-1",),)
    cache._ensure_d3_cache(
        root,
        "run-mismatch",
        _FakeSharedD3(tmp_path / "producer-mismatch"),
        template_keys=selected,
    )
    monkeypatch.setattr(
        cache.pickle,
        "loads",
        lambda _: (_ for _ in ()).throw(AssertionError("unpickle called")),
    )
    with pytest.raises(ValueError, match="d3 cache manifest contract"):
        cache._load_d3_cache(root, (("record-1", "record-2"),))


def test_xdist_template_miss_fails_closed_and_serial_miss_builds(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = sys.modules[__name__]
    records = ("record-1",)
    monkeypatch.setattr(module, "_CONFORMANCE_TEMPLATES", {})
    monkeypatch.setattr(module, "_XDIST_SHARED_CACHE_ACTIVE", True)
    with pytest.raises(ValueError, match="xdist d3 shared cache template miss"):
        _conformance_template(tmp_path / "xdist", records)

    expected = (b"template", ("seed",))
    monkeypatch.setattr(module, "_XDIST_SHARED_CACHE_ACTIVE", False)
    monkeypatch.setattr(module, "_build_conformance_template", lambda *_: expected)
    assert _conformance_template(tmp_path / "serial", records) == expected
    assert _CONFORMANCE_TEMPLATES == {records: expected}


@pytest.mark.parametrize(
    "corruption", ("digest", "mode", "symlink", "capacity", "template")
)
def test_published_d3_shared_cache_corruption_fails_closed(
    tmp_path, corruption: str
) -> None:
    from newsroom.tests import conftest as cache

    root = tmp_path / corruption
    marker = tmp_path / f"producer-{corruption}"
    fake = _FakeSharedD3(marker)
    cache._ensure_d3_cache(root, f"run-{corruption}", fake)
    bundle = root / "bundle.pickle"
    manifest_path = root / "manifest.json"
    if corruption == "digest":
        bundle.write_bytes(bundle.read_bytes() + b"tamper")
    elif corruption == "mode":
        bundle.chmod(0o644)
    elif corruption == "symlink":
        target = tmp_path / "symlink-target"
        target.write_bytes(bundle.read_bytes())
        target.chmod(0o600)
        bundle.unlink()
        bundle.symlink_to(target)
    else:
        manifest = json.loads(manifest_path.read_bytes())
        if corruption == "capacity":
            manifest["capacity"] = 5
        else:
            manifest["template_keys"] = [["record-corrupt"]]
        manifest_path.write_bytes(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        manifest_path.chmod(0o600)
    with pytest.raises(ValueError, match="d3 cache|unsafe d3 cache"):
        cache._ensure_d3_cache(root, f"run-{corruption}", fake)
    assert marker.read_bytes() == b"x"


@pytest.mark.parametrize("corruption", ("capacity", "keys", "templates"))
def test_d3_shared_bundle_contract_corruption_fails_closed(
    tmp_path, corruption: str
) -> None:
    _clone_conformance_base(tmp_path / "base")
    assert d2._SEED_CACHE is not None
    assert _CONFORMANCE_BASE_SNAPSHOT is not None
    template = _CONFORMANCE_BASE_SNAPSHOT
    bundle = {
        "artifact_root": tmp_path,
        "capacity": _CONFORMANCE_BASE_CAPACITY,
        "template_keys": _SHARED_CONFORMANCE_TEMPLATE_KEYS,
        "d2_seed_cache": d2._SEED_CACHE,
        "base_snapshot": _CONFORMANCE_BASE_SNAPSHOT,
        "templates": {
            records: template for records in _SHARED_CONFORMANCE_TEMPLATE_KEYS
        },
    }
    if corruption == "capacity":
        bundle["capacity"] = 5
    elif corruption == "keys":
        bundle["template_keys"] = (("record-corrupt",),)
    else:
        bundle["templates"] = {("record-corrupt",): template}
    with pytest.raises(ValueError, match="d3 shared cache"):
        _install_shared_conformance_cache(
            bundle, expected_template_keys=_SHARED_CONFORMANCE_TEMPLATE_KEYS
        )


@pytest.mark.parametrize(
    "probe",
    _CONFORMANCE_PROBES,
    ids=lambda probe: probe.probe_id,
)
def test_real_v23_store_passes_required_conformance_probe(tmp_path, probe) -> None:
    adapter = _LineageAdapter(tmp_path, probe.case)
    assert adapter.applicability[probe.case] == Applicability.required()
    probe.operation(adapter)
