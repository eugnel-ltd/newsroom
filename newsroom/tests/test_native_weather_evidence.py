from __future__ import annotations

from contextlib import contextmanager, nullcontext

import io
import json
from dataclasses import replace
from datetime import UTC, datetime
from email.message import Message

import pytest

from newsroom.authority import ObjectAdmissionRequest, UtcTimestamp
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.control_plane.graphiti_operational_readiness import (
    OPERATOR_AUTHORITY_DOMAIN,
    OPERATOR_PRINCIPAL_ID,
    _source_requests,
)
from newsroom.control_plane.native_evidence import (
    EvidenceAcquisitionRequest,
    NativeEvidenceHold,
)
from newsroom.control_plane.evidence import validate_governed_evidence_records
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.control_plane.native_source_rights import (
    NativePortfolioRights,
    SourceTermsEvidence,
)
from newsroom.control_plane.native_source_definitions import (
    native_source_definition_requests,
)
from newsroom.control_plane.native_weather_evidence import (
    HK02_ATTRIBUTION,
    POLICY_DIGEST,
    NativeWeatherEvidenceAcquisition,
    hko_completed_event_claim,
)
from newsroom.increment9.proving import SOURCE_URLS
from newsroom.tests.test_graphiti_operational_readiness import _rights, _unit
from newsroom.tests.test_native_runtime import _args
from newsroom.tests.test_native_source_intake import _licence
from newsroom.graphiti_adapter.identity import content_digest
from newsroom.sources import (
    IdentityComponent,
    SourceItemId,
    SourceItemIdentityKind,
    SourceItemRequest,
    SourceRevisionId,
    SourceRevisionRequest,
    SourceTime,
)

NOW = datetime(2026, 9, 8, 15, tzinfo=UTC)
CONTROLLER_POLICY_DIGEST = digest_canonical(
    {"govuk": "component-policy", "weather": POLICY_DIGEST}
)


def _assert_governed_records(base, package, source, acquired, records):
    retained = tuple(
        (record["record_id"], record["record_type"],
         canonical_json_bytes(record).decode(), digest_canonical(record))
        for record in records
    )
    assert validate_governed_evidence_records(
        candidate_id=base.candidate_id,
        source_inventory=((source.unit.source_id, acquired.canonical_url),),
        base_package_digest=base.digest, package=package, retained_records=retained,
    ) is not None
    assert acquired.language == "zh-Hant-HK"
HKO_WARNING = {
    "name": "雷暴警告",
    "code": "WTS",
    "actionCode": "ISSUE",
    "issueTime": "2026-09-08T11:00:00+08:00",
    "updateTime": "2026-09-08T12:00:00+08:00",
    "expireTime": "2026-09-08T23:00:00+08:00",
    "additional_field": "retained",
}
HKO_RAW = json.dumps(
    {"WTS": HKO_WARNING}, ensure_ascii=False, separators=(",", ":"),
).encode()
RSS_ENTRY = (
    b"<item><title>Rain warning</title><guid>one</guid>"
    b"<link>https://weather.metoffice.gov.uk/warnings</link>"
    b"<pubDate>Tue, 08 Sep 2026 12:00:00 GMT</pubDate>"
    b"<description>Full supplied summary</description>"
    b"<extra>Not discarded</extra></item>"
)
RSS_RAW = (
    b'<rss version="2.0"><channel><title>Met Office warnings for UK</title>'
    + RSS_ENTRY
    + b"</channel></rss>"
)


class _Response(io.BytesIO):
    status = 200

    def __init__(self, raw: bytes, url: str, content_type: str) -> None:
        super().__init__(raw)
        self._url = url
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def geturl(self) -> str:
        return self._url


def _seed(runtime, source_id: str, rights):
    if source_id == "HK-02":
        item_key = "WTS"
        headline = HKO_WARNING["name"]
        body = json.dumps(
            {"WTS": HKO_WARNING}, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True,
        )
        canonical_url = SOURCE_URLS[source_id]
        published = "2026-09-08T03:00:00.000000Z"
        updated = "2026-09-08T04:00:00.000000Z"
    else:
        item_key = "one"
        headline = "Rain warning"
        body = RSS_ENTRY.decode()
        canonical_url = "https://weather.metoffice.gov.uk/warnings"
        published = "2026-09-08T12:00:00.000000Z"
        updated = None
    unit = replace(
        _unit(),
        source_id=source_id,
        item_key=item_key,
        headline=headline,
        body=body,
        canonical_url=canonical_url,
        published_at=published,
        updated_at=updated,
        source_definition_url=SOURCE_URLS[source_id],
    )
    if source_id == "HK-02":
        native = native_source_definition_requests(
            source_id=source_id,
            rights=rights.for_source(
                source_id=source_id, definition_url=SOURCE_URLS[source_id]
            ),
        )
        runtime.authority.sources.register_definition(
            native.definition, proof=runtime.proof
        )
        runtime.authority.sources.record_definition_version(
            native.version, proof=runtime.proof
        )
        item_id = SourceItemId.new()
        item = SourceItemRequest(
            item_id,
            native.definition.definition_id,
            native.version.version_id,
            SourceItemIdentityKind.COMPOSITE,
            native.version.item_identity_policy,
            item_key,
            (
                IdentityComponent("item_key", item_key),
                IdentityComponent("source_id", source_id),
            ),
            (),
            f"weather-test-item:{item_id}",
        )
        revision_id = SourceRevisionId.new()
        revision = SourceRevisionRequest(
            revision_id,
            item_id,
            native.version.version_id,
            None,
            updated,
            content_digest(
                headline=headline, body=body, canonical_url=canonical_url
            ),
            native.version.revision_policy,
            "weather-test-v1",
            SourceTime.exact(UtcTimestamp.parse(published)),
            SourceTime.exact(UtcTimestamp.parse(updated)),
            UtcTimestamp.parse("2026-09-08T05:00:00.000000Z"),
            f"weather-test-revision:{revision_id}",
        )
        runtime.authority.sources.register_item(item, proof=runtime.proof)
        runtime.authority.sources.record_revision(revision, proof=runtime.proof)
        requests = (native.definition, native.version, item, revision)
    else:
        requests = _source_requests(unit, _rights())
        for method, request in zip(
            (
                runtime.authority.sources.register_definition,
                runtime.authority.sources.record_definition_version,
                runtime.authority.sources.register_item,
                runtime.authority.sources.record_revision,
                runtime.authority.sources.record_representation,
            ),
            requests,
            strict=True,
        ):
            method(request, proof=runtime.proof)
    version = runtime.authority.sources.version_details(
        requests[1].version_id, proof=runtime.proof
    )
    return unit, EvidenceAcquisitionRequest(
        source_id,
        str(requests[0].definition_id),
        str(requests[1].version_id),
        version.canonical_digest,
        str(requests[3].revision_id),
        canonical_url,
        CONTROLLER_POLICY_DIGEST,
    )


def _portfolio(runtime, monkeypatch):
    from newsroom.control_plane import native_source_rights as source_rights

    terms = {}
    evidence = {}
    observations = {}
    for source_id in ("HK-02", "UK-10"):
        entries = []
        retained = []
        for index, (url, _old_digest) in enumerate(source_rights.TERMS[source_id]):
            raw = (
                f"<main>Fixture retained terms for {source_id} {index}</main>"
            ).encode()
            expected = source_rights.terms_text_digest(source_id, raw)
            entries.append((url, expected))
            admission = runtime.authority.objects.admit(
                ObjectAdmissionRequest(
                    "evidence.source", f"weather-rights:{source_id}:{expected}"
                ),
                raw,
                proof=runtime.proof,
            ).admission
            access = runtime.authority.objects.hydrate(
                source_rights.HydrationRequest(
                    admission.admission_id, "evidence.source"
                ),
                proof=runtime.proof,
            ).decision
            retained.append((
                url, digest_bytes(raw), str(admission.admission_id),
                str(access.access_decision_id),
            ))
        terms[source_id] = tuple(entries)
        observations[source_id] = tuple(retained)
    monkeypatch.setattr(source_rights, "TERMS", terms)
    for source_id in ("HK-02", "UK-10"):
        evidence[source_id] = SourceTermsEvidence(
            source_id,
            "2026-09-08T09:00:00+00:00",
            "REVIEWED_REUSE_PERMITTED",
            observations[source_id],
        )
    # The weather route verifies only its exact retained portfolio terms; the
    # GOV.UK licence member is neither consulted nor relabelled for this source.
    return NativePortfolioRights(_licence(), evidence)


def _acquisition(runtime, rights, fences, **retained):
    return NativeWeatherEvidenceAcquisition(
        sources=runtime.authority.sources,
        objects=runtime.authority.objects,
        proof=runtime.proof,
        rights=rights,
        transport_policy_digest=CONTROLLER_POLICY_DIGEST,
        dispatch_fence=lambda request: nullcontext(fences.append(request)),
        **retained,
        clock=lambda: NOW,
    )


def test_hko_acquires_exact_retained_warning_with_current_rights(
    tmp_path, monkeypatch,
):
    args = _args(tmp_path, monkeypatch)
    args["principal_id"] = OPERATOR_PRINCIPAL_ID
    args["authority_domain"] = OPERATOR_AUTHORITY_DOMAIN
    calls = []
    with open_native_runtime(**args) as runtime:
        rights = _portfolio(runtime, monkeypatch)
        unit_value, request = _seed(runtime, "HK-02", rights)

        held = []
        class FencedResponse(_Response):
            def read(self, *args):
                assert held == [request]
                return super().read(*args)
        class _Opener:
            def open(self, http, timeout):
                assert held == [request]
                calls.append((http.full_url, timeout, http.get_method()))
                return FencedResponse(HKO_RAW, http.full_url, "application/json")

        monkeypatch.setattr("urllib.request.build_opener", lambda *_: _Opener())
        fences = []
        @contextmanager
        def fence(request):
            held.append(request)
            fences.append(request)
            try:
                yield
            finally:
                held.pop()
        acquisition = _acquisition(runtime, rights, fences)
        acquisition._fence = fence
        result = acquisition(request)
        assert held == []
        assert calls == [(SOURCE_URLS["HK-02"], 20, "GET")]
        assert fences == [request]
        assert json.loads(result.body.split(b"\n\n", 1)[0]) == {"WTS": HKO_WARNING}
        assert b"Official status changed" in result.body
        assert result.publication_time == "2026-09-08T03:00:00.000000Z"
        assert result.source_updated_time == "2026-09-08T04:00:00.000000Z"
        assert result.publisher == "Hong Kong Observatory"
        assert result.licence_attribution == HK02_ATTRIBUTION
        assert result.currentness_basis == "AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT"
        assert result.rights_eligibility_digest
        # Exercise the actual assessor admission boundary with transport output,
        # not an independently constructed digest fixture. No model is called.
        from newsroom.control_plane.native_assessor import AutonomousNativeEvidenceAssessor
        from newsroom.control_plane.native_evidence import (
            DependencyAssessment, NativeEvidenceSource, PublicationRightsAssessment,
        )
        from newsroom.sources import SourceDefinitionVersionId
        from newsroom.tests.test_increment10_ingress import _candidate
        from newsroom.tests.test_increment10_editorial import _ready_package
        from newsroom.increment10.evidence import _base_package

        candidate_root = tmp_path / "assessor-candidate"
        candidate_root.mkdir()
        connection, _, candidate = _candidate(candidate_root)
        try:
            current_rights = rights.for_source(source_id="HK-02", definition_url=SOURCE_URLS["HK-02"])
            source = NativeEvidenceSource(
                unit_value,
                runtime.authority.sources.version_details(SourceDefinitionVersionId.parse(request.source_definition_version_id), proof=runtime.proof),
                current_rights,
                DependencyAssessment.create(
                    dependency_status="RESOLVED", evidential_origin_id="HKO",
                    originating_report_id=request.source_revision_id,
                    evidence_digest=result.transport_evidence_digest,
                ),
            )
            class AssessmentReached(Exception):
                pass
            def dispatch(prompt):
                request_value = json.loads(prompt)
                request_source = request_value["sources"][0]
                assert request_source["acquisition_receipt_id"] == result.receipt_digest
                assert "body" not in request_source
                assert "passages" not in request_value["base_package"]
                assert "".join(
                    segment["text"] for segment in request_source["segments"]
                ) == result.body.decode("utf-8")
                raise AssessmentReached
            assessor = AutonomousNativeEvidenceAssessor(dispatch=dispatch)
            base = _base_package(_ready_package(candidate)[1])
            base = replace(
                base,
                source_ids=("HK-02",),
                passages=(result.body.decode("utf-8"),),
                observation_digests=(result.body_digest,),
            )
            with pytest.raises(AssessmentReached):
                assessor(candidate, base, (source,), (result,))
            # Exercise actual structured-field prose through the existing model
            # output validator and writer; no provider stage or invented fact.
            from newsroom.tests.test_native_assessor import _model_package_value
            from newsroom.control_plane.native_assessor import NativeAssessmentExecution
            from newsroom.control_plane.admission import _qualification_relation_is_proven
            from newsroom.control_plane.writer import WriterCopy, required_surface_copy, validate_writer_copy
            package = _model_package_value(_ready_package(candidate)[1])
            sentences = result.body.decode().split("\n\n", 1)[1].splitlines()
            claims = package["governed_claims"]
            for claim, span, rendering in zip(claims, (sentences[0], sentences[2]), (
                "香港天文台已發出雷暴警告。",
                "雷暴警告紀錄於香港時間2026年9月8日12時00分更新。",
            ), strict=True):
                claim.update(claim=span, supporting_excerpt=span, source_ids=["HK-02"],
                             rendered_assertion_zh_hant_hk=rendering)
            claims[1]["localised_factual_expressions"] = [[
                "8 September 2026 at 12:00", "2026年9月8日12時00分",
            ]]
            package.update(substantive_new_information=[c["claim"] for c in claims],
                           geography=["Hong Kong"], categories=["Weather and disasters"],
                           qualification_evidence=[{
                               "test": "LAW_RIGHT_STATUS_POLICY", "governed_claim_id": claims[0]["claim_id"],
                               "policy_version": "newsroom.evid-012.v7", "test_evidence": {
                                   "change_kind": "STATUS", "event_polarity": "AFFIRMED",
                                   "change_relation": "NEW_OR_CHANGED_STATE",
                                   "material_relation_span": sentences[0], "new_state": "issued",
                               },
                           }])
            base = replace(base, source_ids=("HK-02",), passages=(result.body.decode(),),
                           observation_digests=(result.body_digest,))
            output = AutonomousNativeEvidenceAssessor._validated_execution(
                NativeAssessmentExecution(json.dumps({"package": package}), {}),
                candidate, base, (source,), (result,),
            )
            assert _qualification_relation_is_proven(
                output.qualification_evidence[0], output.governed_claims[0],
                source_context=result.body.decode(),
            )
            copy_package = replace(base, governed_claims=output.governed_claims,
                                   substantive_new_information=output.substantive_new_information,
                                   qualification_evidence=output.qualification_evidence)
            from newsroom.control_plane.native_evidence import NativeEvidenceController
            records = NativeEvidenceController._records(base, copy_package, (source,), (result,), output)
            _assert_governed_records(base, copy_package, source, result, records)
            copy_package = replace(copy_package, resolved_evidence_records=tuple(
                sorted((record["record_id"], digest_canonical(record)) for record in records)
            ))
            title, body, links = required_surface_copy(copy_package, paragraphs=True)
            copy = WriterCopy(title, body, "newsroom.offline-exact-copy.v2", copy_package.digest, links)
            failed = [r.reason_code for r in validate_writer_copy(copy, copy_package) if r.result != "PASS"]
            assert failed == []
            for changed in ("policy_digest", "evidence_digest"):
                values = {name: getattr(current_rights, name) for name in (
                    "decision", "permitted_use", "policy_digest", "evidence_digest",
                )}
                values[changed] = digest_canonical({"changed": changed})
                altered = replace(source, rights=PublicationRightsAssessment.create(**values))
                with pytest.raises(NativeEvidenceHold, match="SOURCE_POLICY_FACTS_HOLD"):
                    assessor(candidate, base, (altered,), (result,))
        finally:
            connection.close()


def test_met_office_pubdate_does_not_become_an_asserted_version_time(
    tmp_path, monkeypatch,
):
    args = _args(tmp_path, monkeypatch)
    args["principal_id"] = OPERATOR_PRINCIPAL_ID
    args["authority_domain"] = OPERATOR_AUTHORITY_DOMAIN
    with open_native_runtime(**args) as runtime:
        rights = _portfolio(runtime, monkeypatch)
        _unit_value, request = _seed(runtime, "UK-10", rights)

        class _Opener:
            def open(self, http, timeout):
                return _Response(RSS_RAW, http.full_url, "application/rss+xml")

        monkeypatch.setattr("urllib.request.build_opener", lambda *_: _Opener())
        fences = []
        with pytest.raises(
            NativeEvidenceHold, match="SOURCE_VERSION_TIME_NOT_ASSERTED:UK-10"
        ):
            _acquisition(runtime, rights, fences)(request)
        assert fences == [request]


def test_binding_mismatch_and_empty_inventory_fail_before_evidence_emission(
    tmp_path, monkeypatch,
):
    args = _args(tmp_path, monkeypatch)
    args["principal_id"] = OPERATOR_PRINCIPAL_ID
    args["authority_domain"] = OPERATOR_AUTHORITY_DOMAIN
    calls = []
    with open_native_runtime(**args) as runtime:
        rights = _portfolio(runtime, monkeypatch)
        _unit_value, request = _seed(runtime, "HK-02", rights)
        acquire = _acquisition(runtime, rights, [])
        with pytest.raises(NativeEvidenceHold, match="TRANSPORT_POLICY_MISMATCH"):
            acquire(replace(request, transport_policy_digest=POLICY_DIGEST))
        with pytest.raises(NativeEvidenceHold, match="WEATHER_SOURCE_BINDING_HOLD"):
            acquire(replace(
                request,
                source_definition_version_digest="sha256:" + "0" * 64,
            ))
        assert calls == []

        class _Opener:
            def open(self, http, timeout):
                calls.append(http.full_url)
                return _Response(b"{}", http.full_url, "application/json")

        monkeypatch.setattr("urllib.request.build_opener", lambda *_: _Opener())
        with pytest.raises(NativeEvidenceHold, match="WEATHER_EVIDENCE_METADATA_HOLD"):
            acquire(request)
        assert calls == [SOURCE_URLS["HK-02"]]
        with pytest.raises(ValueError, match="configuration differs"):
            NativeWeatherEvidenceAcquisition(
                sources=runtime.authority.sources,
                objects=runtime.authority.objects,
                proof=runtime.proof,
                rights=rights,
                transport_policy_digest="not-a-digest",
                dispatch_fence=lambda _: nullcontext(),
            )


def test_hko_absent_current_item_rehydrates_exact_retained_cancellation(
    tmp_path, monkeypatch,
):
    from newsroom.control_plane.native_source_intake import native_evidence_sources
    from newsroom.tests.test_native_weather_sources import _poll

    args = _args(tmp_path, monkeypatch)
    args["principal_id"] = OPERATOR_PRINCIPAL_ID
    args["authority_domain"] = OPERATOR_AUTHORITY_DOMAIN
    warning = {
        **HKO_WARNING, "name": "火災危險警告", "code": "WFIREY",
        "type": "黃色", "actionCode": "CANCEL",
        "issueTime": "2026-09-08T11:00:00+08:00",
        "updateTime": "2026-09-08T12:00:00+08:00",
    }
    raw = canonical_json_bytes({"WFIRE": warning, "WTS": HKO_WARNING})
    with open_native_runtime(**args) as runtime:
        rights = _portfolio(runtime, monkeypatch)
        _intake, disposition = _poll(runtime, "HK-02", raw, rights, [])
        unit = next(item for item in disposition.units if item.item_key == "WFIRE")
        observations = {item[1]: item for item in disposition.observations}
        (source,) = native_evidence_sources(
            units=(unit,), sources=runtime.authority.sources,
            objects=runtime.authority.objects, observations=observations,
            licence=rights, proof=runtime.proof,
        )
        request = EvidenceAcquisitionRequest(
            unit.source_id, unit.authority.definition_id,
            unit.authority.definition_version_id,
            source.source_version.canonical_digest,
            unit.authority.revision_id, unit.canonical_url,
            CONTROLLER_POLICY_DIGEST,
        )

        class _Opener:
            def open(self, http, timeout):
                return _Response(b"{}", http.full_url, "application/json")

        monkeypatch.setattr("urllib.request.build_opener", lambda *_: _Opener())
        result = _acquisition(
            runtime, rights, [], retained_units={unit.revision_id: (unit,)},
            observations=observations,
        )(request)

        assert result.currentness_basis == "RETAINED_AUTHORITATIVE_COMPLETED_EVENT"
        assert result.source_observed_time == unit.observed_at
        assert result.retrieval_time == "2026-09-08T15:00:00.000000Z"
        selected_raw = unit.body.encode("utf-8")
        assert result.body.startswith(selected_raw + b"\n\n")
        assert digest_bytes(raw) == unit.observation_digest
        assert digest_bytes(selected_raw) != unit.observation_digest
        source_span, rendered_headline, source_date, rendered_date = (
            hko_completed_event_claim(selected_raw)
        )
        sentence = next(
            line for line in result.body.decode().splitlines()
            if line.startswith(
                "Official status changed for completed historical event: "
            )
        )
        assert sentence == source_span
        assert source_date == "8 September 2026 at 12:00"
        assert rendered_date == "2026年9月8日12時00分"
        assert result.body.decode().splitlines()[-1] == (
            "The timestamp above is the record update time and no exact "
            "cancellation time is asserted."
        )
        assert result.rights_eligibility_digest

        sources = runtime.authority.sources
        class _RepresentationMismatch:
            def __getattr__(self, name):
                return getattr(sources, name)

            def representation(self, *args, **kwargs):
                retained = sources.representation(*args, **kwargs)
                return replace(
                    retained,
                    request=replace(
                        retained.request,
                        permitted_fields_digest=digest_canonical(
                            {"wrong": "representation fields"}
                        ),
                    ),
                )

        mismatched = _acquisition(
            runtime, rights, [], retained_units={unit.revision_id: (unit,)},
            observations=observations,
        )
        mismatched._sources = _RepresentationMismatch()
        with pytest.raises(
            NativeEvidenceHold, match="WEATHER_RETAINED_COMPLETED_EVENT_HOLD"
        ):
            mismatched(request)

        observation = observations[unit.observation_digest]
        observations[unit.observation_digest] = (
            observation[0], digest_canonical({"wrong": "observation"}),
            observation[2], observation[3],
        )
        with pytest.raises(
            NativeEvidenceHold, match="WEATHER_RETAINED_COMPLETED_EVENT_HOLD"
        ):
            _acquisition(
                runtime, rights, [], retained_units={unit.revision_id: (unit,)},
                observations=observations,
            )(request)

        from newsroom.control_plane.admission import (
            _qualification_relation_is_proven,
        )
        from newsroom.control_plane.native_assessor import (
            AutonomousNativeEvidenceAssessor, NativeAssessmentExecution,
        )
        from newsroom.control_plane.writer import (
            WriterCopy, required_surface_copy, validate_writer_copy,
        )
        from newsroom.increment10.evidence import _base_package
        from newsroom.tests.test_increment10_editorial import _ready_package
        from newsroom.tests.test_increment10_ingress import _candidate
        from newsroom.tests.test_native_assessor import _model_package_value

        candidate_root = tmp_path / "historical-assessor-candidate"
        candidate_root.mkdir()
        candidate_connection, _, candidate = _candidate(candidate_root)
        try:
            base = _base_package(_ready_package(candidate)[1])
            package = _model_package_value(_ready_package(candidate)[1])
            lines = result.body.decode().splitlines()
            historical = next(
                line for line in lines
                if line.startswith(
                    "Official status changed for completed historical event: "
                )
            )
            updated = next(line for line in lines if "record was updated" in line)
            claims = package["governed_claims"]
            for claim, span, rendering in zip(
                claims, (historical, updated),
                (
                    rendered_headline,
                    "該紀錄於香港時間2026年9月8日12時00分更新。",
                ), strict=True,
            ):
                claim.update(
                    claim=span, supporting_excerpt=span, source_ids=["HK-02"],
                    rendered_assertion_zh_hant_hk=rendering,
                )
            for claim in claims:
                claim["localised_factual_expressions"] = [[
                    "8 September 2026 at 12:00", "2026年9月8日12時00分",
                ]]
            package.update(
                substantive_new_information=[item["claim"] for item in claims],
                geography=["Hong Kong"], categories=["Weather and disasters"],
                qualification_evidence=[{
                    "test": "LAW_RIGHT_STATUS_POLICY",
                    "governed_claim_id": claims[0]["claim_id"],
                    "policy_version": "newsroom.evid-012.v7",
                    "test_evidence": {
                        "change_kind": "STATUS", "event_polarity": "AFFIRMED",
                        "change_relation": "NEW_OR_CHANGED_STATE",
                        "material_relation_span": historical,
                        "new_state": "cancelled",
                    },
                }],
            )
            base = replace(
                base, source_ids=("HK-02",), passages=(result.body.decode(),),
                observation_digests=(result.body_digest,),
            )
            output = AutonomousNativeEvidenceAssessor._validated_execution(
                NativeAssessmentExecution(json.dumps({"package": package}), {}),
                candidate, base, (source,), (result,),
            )
            assert _qualification_relation_is_proven(
                output.qualification_evidence[0], output.governed_claims[0],
                source_context=result.body.decode(),
            )
            # Prove the v18 provider boundary reconstructs the exact
            # completed-event headline without retaining its terminal newline.
            from newsroom.control_plane.native_assessor import (
                PROVIDER_SCHEMA, _V17_PROVIDER_SCHEMA,
            )
            from newsroom.control_plane.native_assessor_references import (
                build_source_view,
            )
            from newsroom.control_plane.native_assessor_wire import materialise
            from newsroom.tests.test_native_assessor import _v18_wire_from_v17

            view = build_source_view((result.body.decode(),), ("HK-02",))
            by_text = {
                view.resolve_range({
                    "first_span_id": segment.span_id,
                    "last_span_id": segment.span_id,
                })[0]: segment
                for segment in view.segments
            }

            def fragments(target, segment):
                values = []
                offset = 0
                for name, _kind in segment.entities:
                    index = target.find(name, offset)
                    assert index >= offset
                    values.append(target[offset:index])
                    offset = index + len(name)
                values.append(target[offset:])
                return values

            reference_claims = []
            for role, span, rendering in (
                ("HEADLINE", source_span, rendered_headline),
                (
                    "SUBSTANTIVE",
                    updated,
                    "該紀錄於香港時間2026年9月8日12時00分更新。",
                ),
            ):
                segment = by_text[span]
                reference_claims.append({
                    "claim_role": role,
                    "claim_range": {
                        "first_span_id": segment.span_id,
                        "last_span_id": segment.span_id,
                    },
                    "support_range": {
                        "first_span_id": segment.span_id,
                        "last_span_id": segment.span_id,
                    },
                    "rendered_fragments": fragments(rendering, segment),
                    "status": "CONFIRMED_FACT",
                    "semantic_relation": {
                        "source_modality": "ASSERTED",
                        "rendered_modality": "ASSERTED",
                        "source_polarity": "AFFIRMED",
                        "rendered_polarity": "AFFIRMED",
                        "relation": "SEMANTICALLY_EQUIVALENT",
                    },
                    "localised_factual_expressions": [[
                        source_date, rendered_date,
                    ]],
                    "quotations": [],
                    "certainty": "CONFIRMED",
                    "originality_basis": "FACTUAL_REWRITE_REQUIRED",
                    "originality_policy_version": "newsroom.cont-originality.v3",
                    "admitted_use": "PUBLICATION_EVIDENCE",
                    "policy_version": "newsroom.governed-claim.v7",
                })
            wire = {"package": {
                "substantive_claim_indexes": [0, 1],
                "governed_claims": reference_claims,
                "qualification_evidence": [{
                    "test": "LAW_RIGHT_STATUS_POLICY",
                    "claim_index": 0,
                    "policy_version": "newsroom.evid-012.v7",
                    "test_evidence": {
                        "change_kind": "STATUS",
                        "event_polarity": "AFFIRMED",
                        "change_relation": "NEW_OR_CHANGED_STATE",
                        "material_relation_span": source_span,
                        "new_state": "cancelled",
                    },
                }],
                "selection_rationale": "Exact completed-event source spans selected.",
                "geography": ["Hong Kong"],
                "categories": ["Weather and disasters"],
                "explicit_exclusions": [],
            }}
            wire = _v18_wire_from_v17(wire)
            materialised, receipt = materialise(
                canonical_json_bytes(wire), view, "historical-weather-request",
                provider_schema=PROVIDER_SCHEMA,
                v17_schema=_V17_PROVIDER_SCHEMA,
            )
            assert receipt["materialised_text"] == canonical_json_bytes(
                materialised
            ).decode()
            reference_output = AutonomousNativeEvidenceAssessor._validated_execution(
                NativeAssessmentExecution(receipt["materialised_text"], {}),
                candidate, base, (source,), (result,),
            )
            assert reference_output.governed_claims[0].claim == source_span
            assert (
                reference_output.governed_claims[0].rendered_assertion_zh_hant_hk
                == rendered_headline
            )
            assert reference_output.governed_claims[0].localised_factual_expressions == (
                (source_date, rendered_date),
            )
            assert _qualification_relation_is_proven(
                reference_output.qualification_evidence[0],
                reference_output.governed_claims[0],
                source_context=result.body.decode(),
            )
            from copy import deepcopy
            for fault in ("undated", "cancellation-time", "present-warning"):
                altered = deepcopy(package)
                head = altered["governed_claims"][0]
                if fault == "undated":
                    plain = next(line for line in lines if line.startswith("Official status changed:"))
                    head.update(claim=plain, supporting_excerpt=plain,
                                rendered_assertion_zh_hant_hk="香港天文台已取消黃色火災危險警告。",
                                localised_factual_expressions=[])
                    altered["substantive_new_information"][0] = plain
                elif fault == "cancellation-time":
                    head["rendered_assertion_zh_hant_hk"] = (
                        "香港天文台於香港時間2026年9月8日12時00分取消黃色火災危險警告。"
                    )
                else:
                    head["rendered_assertion_zh_hant_hk"] = rendered_headline + "目前沒有火災危險。"
                with pytest.raises(NativeEvidenceHold, match="HISTORICAL_TIME_RELATION_HOLD"):
                    AutonomousNativeEvidenceAssessor._validated_execution(
                        NativeAssessmentExecution(json.dumps({"package": altered}), {}),
                        candidate, base, (source,), (result,),
                    )
            copy_package = replace(
                base, governed_claims=output.governed_claims,
                substantive_new_information=output.substantive_new_information,
                qualification_evidence=output.qualification_evidence,
            )
            from newsroom.control_plane.native_evidence import (
                NativeEvidenceController,
            )
            records = NativeEvidenceController._records(
                base, copy_package, (source,), (result,), output,
            )
            _assert_governed_records(base, copy_package, source, result, records)
            copy_package = replace(
                copy_package, resolved_evidence_records=tuple(sorted(
                    (record["record_id"], digest_canonical(record))
                    for record in records
                )),
            )
            title, body, links = required_surface_copy(
                copy_package,
                paragraphs=True,
                context_preserving=True,
            )
            copy = WriterCopy(
                title, body, "newsroom.offline-exact-copy.v3",
                copy_package.digest, links,
            )
            assert body == "\n\n".join((
                "該紀錄於香港時間2026年9月8日12時00分更新。",
                rendered_headline,
            ))
            assert body.count(rendered_headline) == 1
            failed = [
                item.reason_code for item in validate_writer_copy(
                    copy, copy_package
                ) if item.result != "PASS"
            ]
            assert failed == []
        finally:
            candidate_connection.close()


def test_hko_absent_current_item_does_not_relabel_retained_active_warning(
    tmp_path, monkeypatch,
):
    from newsroom.control_plane.native_source_intake import native_evidence_sources
    from newsroom.tests.test_native_weather_sources import _poll

    args = _args(tmp_path, monkeypatch)
    args["principal_id"] = OPERATOR_PRINCIPAL_ID
    args["authority_domain"] = OPERATOR_AUTHORITY_DOMAIN
    with open_native_runtime(**args) as runtime:
        rights = _portfolio(runtime, monkeypatch)
        _intake, disposition = _poll(runtime, "HK-02", HKO_RAW, rights, [])
        unit = next(item for item in disposition.units if item.item_key == "WTS")
        observations = {item[1]: item for item in disposition.observations}
        (source,) = native_evidence_sources(
            units=(unit,), sources=runtime.authority.sources,
            objects=runtime.authority.objects, observations=observations,
            licence=rights, proof=runtime.proof,
        )
        request = EvidenceAcquisitionRequest(
            unit.source_id, unit.authority.definition_id,
            unit.authority.definition_version_id,
            source.source_version.canonical_digest,
            unit.authority.revision_id, unit.canonical_url,
            CONTROLLER_POLICY_DIGEST,
        )

        class _Opener:
            def open(self, http, timeout):
                return _Response(b"{}", http.full_url, "application/json")

        monkeypatch.setattr("urllib.request.build_opener", lambda *_: _Opener())
        with pytest.raises(
            NativeEvidenceHold, match="WEATHER_RETAINED_COMPLETED_EVENT_HOLD"
        ):
            _acquisition(
                runtime, rights, [], retained_units={unit.revision_id: (unit,)},
                observations=observations,
            )(request)


@pytest.mark.parametrize(("action", "verb"), (("ISSUE", "issued"), ("REISSUE", "reissued"), ("CANCEL", "cancelled"), ("EXTEND", "extended")))
def test_hko_structured_fields_have_exact_readable_fact_spans(action, verb):
    from newsroom.control_plane.native_weather_evidence import (
        hko_completed_event_body, hko_evidence_body, legacy_hko_body,
    )
    warning = {**HKO_WARNING, "name": "火災危險警告", "type": "黃色", "actionCode": action}
    raw = canonical_json_bytes({"WFIRE": warning})
    body = hko_evidence_body(raw)
    assert body.startswith(raw + b"\n\n")
    assert f"Official status changed: 香港天文台 {verb} the 黃色火災危險警告." in body.decode()
    assert "8 September 2026 at 12:00" in body.decode()
    assert legacy_hko_body(body) == raw
    if action == "CANCEL":
        historical = hko_completed_event_body(raw)
        assert legacy_hko_body(historical) == body
    with pytest.raises(ValueError, match="normalised"):
        legacy_hko_body(raw + b"\n\n" + body.split(b"\n\n", 1)[1].replace(b"12:00", b"13:00"))


def test_hko_update_or_absence_does_not_invent_warning_cancellation():
    from newsroom.control_plane.native_weather_evidence import hko_evidence_body
    raw = json.dumps({"WTS": {**HKO_WARNING, "actionCode": "UPDATE"}}, ensure_ascii=False).encode()
    body = hko_evidence_body(raw)
    assert b"Official status changed" not in body
    assert b"cancelled" not in body
    for value in ({}, {"WTS": {**HKO_WARNING, "actionCode": "UNKNOWN"}}, {"WTS": {**HKO_WARNING, "type": {}}}):
        with pytest.raises(ValueError):
            hko_evidence_body(json.dumps(value).encode())


def test_hko_completed_event_claim_preserves_exact_update_precision():
    warning = {
        **HKO_WARNING, "name": "火災危險警告", "type": "黃色",
        "actionCode": "CANCEL",
        "updateTime": "2026-09-20T22:20:30.123456+08:00",
    }
    raw = canonical_json_bytes({"WFIRE": warning})
    source, rendered, source_date, rendered_date = hko_completed_event_claim(raw)
    assert source == (
        "Official status changed for completed historical event: 香港天文台 "
        "cancelled the 黃色火災危險警告 (record updated on 20 September 2026 "
        "at 22:20:30.123456 (香港時間))."
    )
    assert rendered == (
        "香港天文台已取消黃色火災危險警告；官方紀錄於香港時間"
        "2026年9月20日22時20分30.123456秒更新。"
    )
    assert source_date == "20 September 2026 at 22:20:30.123456"
    assert rendered_date == "2026年9月20日22時20分30.123456秒"
    for invalid in (
        canonical_json_bytes({"WFIRE": {**warning, "actionCode": "ISSUE"}}),
        canonical_json_bytes({"WFIRE": warning, "WTS": HKO_WARNING}),
        canonical_json_bytes({
            "WFIRE": {**warning, "updateTime": "2026-09-20T22:20:30"}
        }),
    ):
        with pytest.raises(ValueError):
            hko_completed_event_claim(invalid)


@pytest.mark.parametrize(("text", "invalid"), (("香港天文台", False), ("香港天文台发布消息", True), ("台湾气象台", True)))
def test_observatory_hong_kong_spelling_is_not_simplified(text, invalid):
    from newsroom.control_plane.zh_hant import contains_simplified_variant
    assert contains_simplified_variant(text) is invalid
