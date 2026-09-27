from __future__ import annotations

import io
import json
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from email.message import Message
from types import SimpleNamespace

import pytest

from newsroom.authority import ObjectAdmissionRequest, UtcTimestamp
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority.neo4j_fulltext_reader import (
    Neo4jFullTextReadPhase,
    Neo4jFullTextReadResult,
    Neo4jFullTextReader,
)
from newsroom.control_plane import govuk_rights, native_assessor, native_composition, native_source_rights
from newsroom.control_plane.govuk_rights import GovUkLicenceEvidence, POLICY_DIGEST
from newsroom.control_plane.native_assessor import NativeAssessmentExecution
from newsroom.control_plane.evidence import (
    EVID_012_POLICY_VERSION,
    GOVERNED_CLAIM_POLICY_VERSION,
    ORIGINALITY_POLICY_VERSION,
)
from newsroom.graphiti_adapter.real import RealGraphitiAdapter
from newsroom.increment5.fulltext_contracts import (
    FULLTEXT_COMPONENT_DIGEST,
    NORMALIZATION_COMPONENT_DIGEST,
    FullTextIndexState,
    FullTextProfile,
    FullTextProjectionSnapshot,
)
from newsroom.projection.models import ProjectionGenerationId, ProjectionGenerationState
from newsroom.projection.neo4j.models import (
    NEO4J_B2_DRIVER_VERSION,
    NEO4J_B2_SERVER_VERSION,
)
from newsroom.tests.increment5b2_helpers import component_row, index_row
from newsroom.tests.projection_b2_helpers import MemoryNeo4jAdapter
from newsroom.tests.test_graphiti_adapter_4d_outcomes import (
    _production_shaped_execution,
)
from newsroom.tests.test_native_composition import _arguments
from newsroom.tests.test_native_assessor import _v18_wire_from_v17
from newsroom.tests.test_native_embeddings import _response as embedding_response
from newsroom.tests.test_native_source_intake import ATOM, _document, _seed_uk01


@pytest.fixture(autouse=True)
def _graphiti_test_deployment_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    # These fake-provider observers model a deployed tree, not pytest artefacts.
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti._graphiti_implementation_identity",
        lambda: ("a" * 40, True),
    )


NOW = datetime(2042, 3, 12, 12, tzinfo=UTC)


class _Projection:
    latest_snapshot = None

    def __init__(
        self, _driver, *, database, generation_id, fulltext_index, vector_index
    ):
        self.generation_id = generation_id
        self.document_label = "NewsroomNativeRetrievalDocument_vertical"
        self.fulltext_index = fulltext_index
        self.rows = []

    def bootstrap(self):
        return None

    def upsert(self, receipt, document, vector):
        existing = [row for row in self.rows if row[1].passage_id == document.passage_id]
        if existing:
            assert existing == [(receipt, document, vector)]
        else:
            self.rows.append((receipt, document, vector))

    def reconcile_membership(self, receipts):
        self.rows = [row for row in self.rows if row[0] in receipts]
        retained = {row[0] for row in self.rows}
        return tuple(receipt for receipt in receipts if receipt not in retained)

    def retrieve(self, *, query_text, query_vector):
        rows = tuple(
            {**receipt.projection_value(), "score": 1.0}
            for receipt, _document, _vector in self.rows
        )
        return (), rows

    def retrieve_vector(self, *, query_vector):
        return self.retrieve(query_text="unused", query_vector=query_vector)[1]

    def snapshot(
        self,
        *,
        generation_identity_digest,
        rights_manifest_digest,
        contiguous_ledger_seq,
        expected_document_count,
        clock,
    ):
        assert len(self.rows) == expected_document_count
        recorded = clock()
        snapshot = FullTextProjectionSnapshot(
            generation_id=ProjectionGenerationId.parse(self.generation_id),
            generation_state=ProjectionGenerationState.ACTIVE,
            generation_identity_digest=generation_identity_digest,
            document_label=self.document_label,
            index_name=self.fulltext_index,
            index_state=FullTextIndexState.ONLINE,
            fulltext_component_digest=FULLTEXT_COMPONENT_DIGEST,
            normalization_component_digest=NORMALIZATION_COMPONENT_DIGEST,
            rights_manifest_digest=rights_manifest_digest,
            profile=FullTextProfile.NATIVE_RUNTIME,
            contiguous_ledger_seq=contiguous_ledger_seq,
            open_gap_count=0,
            dead_letter_count=0,
            validation_recorded_at=recorded,
            freshness_deadline=UtcTimestamp(recorded.value + timedelta(hours=1)),
            index_document_count=expected_document_count,
            server_version=NEO4J_B2_SERVER_VERSION,
            driver_version=NEO4J_B2_DRIVER_VERSION,
        )
        type(self).latest_snapshot = snapshot
        return snapshot


def _fulltext_reader():
    def read(request):
        snapshot = _Projection.latest_snapshot
        if request.phase is Neo4jFullTextReadPhase.COMPONENT:
            return Neo4jFullTextReadResult(
                request.phase,
                NEO4J_B2_DRIVER_VERSION,
                component=component_row(version=NEO4J_B2_SERVER_VERSION),
            )
        if request.phase is Neo4jFullTextReadPhase.INDEX:
            return Neo4jFullTextReadResult(
                request.phase,
                NEO4J_B2_DRIVER_VERSION,
                indexes=(index_row(snapshot),),
            )
        return Neo4jFullTextReadResult(
            request.phase, NEO4J_B2_DRIVER_VERSION, rows=()
        )

    return Neo4jFullTextReader(
        driver_version=NEO4J_B2_DRIVER_VERSION, read=read
    )


class _Response(io.BytesIO):
    def __init__(self, body, *, url, content_type="application/json"):
        super().__init__(body)
        self.status = 200
        self._url = url
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def geturl(self):
        return self._url


def _install_boundaries(monkeypatch, counters):
    monkeypatch.setattr(
        "newsroom.authority._graphiti_increment4_system._open_structural_graph_adapter",
        lambda _: MemoryNeo4jAdapter(),
    )
    monkeypatch.setattr(
        native_composition,
        "open_native_retrieval_neo4j_resources",
        lambda **arguments: SimpleNamespace(
            projector=_Projection(
                None,
                database=arguments["config"].database,
                generation_id=arguments["generation_id"],
                fulltext_index=arguments["fulltext_index"],
                vector_index=arguments["vector_index"],
            ),
            fulltext=_fulltext_reader(),
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(
        RealGraphitiAdapter,
        "execute",
        lambda _self, *, attempt, workspace_root: (
            counters.__setitem__("graphiti", counters["graphiti"] + 1)
            or _production_shaped_execution(attempt)
        ),
    )

    terms = (
        b"<html><main>Reviewed GOV.UK reuse terms.</main></html>",
        b"<html><main>Reviewed Open Government Licence terms.</main></html>",
    )
    monkeypatch.setattr(
        govuk_rights,
        "REVIEWED_TEXT",
        {
            url: govuk_rights.licence_text_digest(raw)
            for url, raw in zip(
                (govuk_rights.REUSE_URL, govuk_rights.LICENCE_URL),
                terms,
                strict=True,
            )
        },
    )

    def retain_licence(*, objects, proof, dispatch_fence, clock):
        admissions = tuple(
            objects.admit(
                ObjectAdmissionRequest(
                    "evidence.source", f"native-vertical-licence:{digest_bytes(raw)}"
                ),
                raw,
                proof=proof,
            ).admission
            for raw in terms
        )
        return GovUkLicenceEvidence(
            tuple(item.admission_id for item in admissions),
            tuple(item.blob.blob_digest for item in admissions),
            clock().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            POLICY_DIGEST,
        )

    monkeypatch.setattr(
        native_composition, "retain_current_govuk_licence", retain_licence
    )
    monkeypatch.setattr(
        native_composition, "observe_portfolio_terms", lambda **kwargs: {
            source: native_source_rights.SourceTermsEvidence(
                source, kwargs["clock"]().isoformat(), "SOURCE_TERMS_UNAVAILABLE", (),
            ) for source in native_source_rights.TERMS
        }
    )

    class _Opener:
        def open(self, request, timeout):
            url = request.full_url
            if url.endswith("/embeddings"):
                counters["embedding"] += 1
                return _Response(json.dumps(embedding_response()).encode(), url=url)
            counters["source"] += 1
            successor = counters.get("document_body") != "Official deadline changed."
            return _Response(
                (
                    ATOM.replace(
                        b"2026-09-08T11:00:00Z", b"2026-09-08T11:05:00Z"
                    )
                    if successor else ATOM
                )
                if ".atom" in url
                else _document(
                    body=counters.get("document_body", "Official deadline changed."),
                    updated=(
                        "2026-09-08T11:05:00Z"
                        if successor else "2026-09-08T11:00:00Z"
                    ),
                ),
                url=url,
                content_type=("application/atom+xml" if ".atom" in url else "application/json"),
            )

    monkeypatch.setattr("urllib.request.build_opener", lambda *_args: _Opener())
    monkeypatch.setattr(
        native_assessor,
        "read_grok_command_semantic_version",
        lambda: "1.0.8",
    )
    monkeypatch.setattr(
        native_assessor,
        "cont_writer_implementation_identity",
        lambda: ("1" * 40, True),
    )

    def assess(prompt):
        counters["assessor"] += 1
        request = json.loads(prompt)
        source = request["sources"][0]
        assert "body" not in source
        assert "passages" not in request["base_package"]
        assert "".join(item["text"] for item in source["segments"])
        headline = counters.get("document_body", "Official deadline changed.")
        selected = {item["text"].removesuffix("\n"): item["span_id"]
                    for item in source["segments"]}
        assert headline in selected
        assert "Visa rules updated" in selected
        rendered_headline = (
            "官方限期已經更改。" if headline == "Official deadline changed."
            else "官方限期已更改，截止日期延後。"
        )
        rendered = "簽證規則已更新"
        qualification_span = headline.split(".", 1)[0]
        qualification_facts = {
            "action_class": "OFFICIAL_DEADLINE",
            "event_polarity": "AFFIRMED",
            "action_relation": "NEW_OR_CHANGED_OFFICIAL_ACTION",
            "material_relation_span": qualification_span,
            "reader_action": qualification_span,
        }

        def governed_claim(*, span_id, rendered, role):
            return {
                "claim_range": {"first_span_id": span_id, "last_span_id": span_id},
                "support_range": {"first_span_id": span_id, "last_span_id": span_id},
                "status": "CONFIRMED_FACT",
                "rendered_fragments": [rendered],
                "claim_role": role,
                "semantic_relation": {
                    "source_modality": "ASSERTED",
                    "rendered_modality": "ASSERTED",
                    "source_polarity": "AFFIRMED",
                    "rendered_polarity": "AFFIRMED",
                    "relation": "SEMANTICALLY_EQUIVALENT",
                },
                "localised_factual_expressions": [],
                "quotations": [],
                "certainty": "CONFIRMED",
                "originality_basis": "FACTUAL_REWRITE_REQUIRED",
                "originality_policy_version": ORIGINALITY_POLICY_VERSION,
                "admitted_use": "PUBLICATION_EVIDENCE",
                "policy_version": GOVERNED_CLAIM_POLICY_VERSION,
            }

        package = {
            "substantive_claim_indexes": [0, 1],
            "governed_claims": [
                governed_claim(
                    span_id=selected[headline], rendered=rendered_headline,
                    role="HEADLINE",
                ),
                governed_claim(
                    span_id=selected["Visa rules updated"], rendered=rendered,
                    role="SUBSTANTIVE",
                ),
            ],
            "qualification_evidence": [{
                "test": "OFFICIAL_ACTION_OR_DEADLINE",
                "claim_index": 0,
                "test_evidence": qualification_facts,
                "policy_version": EVID_012_POLICY_VERSION,
            }],
            "selection_rationale": "A verified official deadline changed.",
            "geography": ["UK"],
            "categories": ["Politics and law"],
            "explicit_exclusions": [],
        }
        return NativeAssessmentExecution(
            canonical_json_bytes(
                _v18_wire_from_v17({"package": package})
            ).decode(),
            {
                "usage_basis": "PROVIDER_REPORTED",
                "input_tokens": 1,
                "output_tokens": 1,
                "cached_read_tokens": 0,
                "cached_write_tokens": 0,
                "reasoning_tokens": 0,
                "context_tokens": 1,
                "total_tokens": 2,
            },
        )

    monkeypatch.setattr(native_assessor, "_dispatch_grok", assess)


def _tick_without_shared_history_rescans(pipeline, *, cycle_id, monkeypatch):
    from newsroom.authority._discovery_store import _DiscoveryAuthorityStore
    from newsroom.authority._event_hypothesis_lineage_system import _LineageStore
    from newsroom.authority._event_hypothesis_relationship_system import _RelationshipEventStore
    from newsroom.authority.story_candidate_system import _CandidateStore

    connection = pipeline._runtime.authority.work_items._authority._connection
    global_foreign_keys = []

    def trace(statement):
        if statement.strip().rstrip(";").upper() == "PRAGMA FOREIGN_KEY_CHECK":
            global_foreign_keys.append(statement)

    def global_scan(*_args, **_kwargs):
        raise AssertionError("native domain operation rescanned shared authority history")

    # The real opener has already run. Domain operations must retain their
    # complete exact-event checks, not repeat the full shared-store verifier.
    with monkeypatch.context() as scoped:
        for store in (_CandidateStore, _LineageStore, _DiscoveryAuthorityStore):
            for method in (
                "_validate_relational_invariants", "_validate_immutable_records",
                "_validate_registry_coverage",
            ):
                scoped.setattr(store, method, global_scan)
        scoped.setattr(_RelationshipEventStore, "_validate_relational_invariants", global_scan)
        connection.set_trace_callback(trace)
        try:
            report = pipeline.tick(cycle_id=cycle_id)
        finally:
            connection.set_trace_callback(None)
    assert global_foreign_keys == []
    return report


def test_native_vertical_reaches_private_ack_and_reopens_without_provider_repeat(
    tmp_path, monkeypatch
):
    counters = {
        "source": 0,
        "graphiti": 0,
        "embedding": 0,
        "assessor": 0,
        "document_body": "Official deadline changed.",
    }
    _install_boundaries(monkeypatch, counters)
    clock = [NOW]
    arguments = {
        **_arguments(tmp_path),
        "clock": lambda: clock[0],
        "stop_check": lambda: None,
        "stop_fence": nullcontext,
    }
    with native_composition.open_native_pipeline(**arguments) as setup:
        definition_id = _seed_uk01(setup._runtime)
    arguments["source_definition_ids"] = {"UK-01": definition_id}

    with native_composition.open_native_pipeline(**arguments) as pipeline:
        report = _tick_without_shared_history_rescans(
            pipeline, cycle_id="native-vertical-1", monkeypatch=monkeypatch,
        )
        assert report.revision_states == {"ACKNOWLEDGED": 2}, pipeline._journal.progress
        assert pipeline._collision._journal == arguments["private_path"]
        assert pipeline._runtime.ingress.receipt_count == 2

        first_versions = {
            progress["facts"]["candidate_version_id"]:
            pipeline._runtime.authority.candidates.load_version(
                progress["facts"]["candidate_version_id"]
            )
            for progress in pipeline._journal.progress.values()
            if "candidate_version_id" in progress["facts"]
        }
        first_candidates = {
            candidate_id: pipeline._runtime.authority.candidates.versions(candidate_id)
            for candidate_id in {version.candidate_id for version in first_versions.values()}
        }

    clock[0] = NOW + timedelta(minutes=5)
    counters["document_body"] = "Official deadline changed. It now has a later date."
    with native_composition.open_native_pipeline(**arguments) as successor:
        report = _tick_without_shared_history_rescans(
            successor, cycle_id="native-vertical-successor", monkeypatch=monkeypatch,
        )
        assert report.revision_states == {"ACKNOWLEDGED": 4}, successor._journal.progress
        successor_versions = {
            progress["facts"]["candidate_version_id"]:
            successor._runtime.authority.candidates.load_version(
                progress["facts"]["candidate_version_id"]
            )
            for progress in successor._journal.progress.values()
            if "candidate_version_id" in progress["facts"]
        }
        successor_candidates = {
            candidate_id: successor._runtime.authority.candidates.versions(candidate_id)
            for candidate_id in {version.candidate_id for version in successor_versions.values()}
        }
        assert successor_candidates.keys() == first_candidates.keys()
        assert all(
            len(successor_candidates[candidate_id])
            == len(first_candidates[candidate_id]) + 1
            and successor_candidates[candidate_id][-1].ordinal
            == first_candidates[candidate_id][-1].ordinal + 1
            for candidate_id in first_candidates
        )

    dispatched = dict(counters)
    with native_composition.open_native_pipeline(**arguments) as reopened:
        report = _tick_without_shared_history_rescans(
            reopened, cycle_id="native-vertical-replay", monkeypatch=monkeypatch,
        )
        assert report.revision_states == {"ACKNOWLEDGED": 4}
        assert reopened._runtime.ingress.receipt_count == 4
    assert counters["graphiti"] == dispatched["graphiti"]
    assert counters["embedding"] == dispatched["embedding"]
    assert counters["assessor"] == dispatched["assessor"]
