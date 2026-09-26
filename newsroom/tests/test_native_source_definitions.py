import pytest

from newsroom.authority.canonical import digest_canonical
from newsroom.control_plane.native_evidence import PublicationRightsAssessment
from newsroom.control_plane.native_source_definitions import (
    MISSING_SOURCE_IDS, native_source_definition_requests,
    register_missing_native_source_definitions,
)
from newsroom.control_plane.graphiti_operational_readiness import (
    OPERATOR_AUTHORITY_DOMAIN, OPERATOR_PRINCIPAL_ID,
)
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.increment9.proving import SOURCE_URLS
from newsroom.sources import (
    BaselinePolicyKind, CoverageContribution, ObservationModel,
    SourceDependencyKind,
)
from newsroom.tests.test_native_runtime import _args


def _rights(
    source_id: str, *, decision: str = "PERMITTED", evidence_tag: str = "",
):
    return PublicationRightsAssessment.create(
        decision=decision,
        permitted_use="PUBLICATION_EVIDENCE",
        policy_digest=digest_canonical({"policy": source_id}),
        evidence_digest=digest_canonical({
            "observation": source_id, "tag": evidence_tag,
        }),
    )


def test_missing_native_source_definitions_are_typed_exact_and_replayable():
    expected = {
        "UK-02": (ObservationModel.MUTABLE_ITEM, BaselinePolicyKind.MAINTAINED_DOCUMENT,
                  CoverageContribution.REVISION_VISIBILITY),
        "UK-03": (ObservationModel.MUTABLE_ITEM, BaselinePolicyKind.MAINTAINED_DOCUMENT,
                  CoverageContribution.REVISION_VISIBILITY),
        "UK-05": (ObservationModel.ROLLING_LIST, BaselinePolicyKind.BOUNDED_BACKFILL,
                  CoverageContribution.DETECTION_PATH),
        "HK-02": (ObservationModel.COMPLETE_CURRENT_STATE,
                  BaselinePolicyKind.COMPLETE_STATE_FIRST_OBSERVED_ACTIVE,
                  CoverageContribution.URGENT_FAST_PATH),
    }
    assert tuple(expected) == MISSING_SOURCE_IDS
    for source_id, semantics in expected.items():
        first = native_source_definition_requests(source_id=source_id, rights=_rights(source_id))
        replay = native_source_definition_requests(source_id=source_id, rights=_rights(source_id))
        assert first == replay
        assert first.version.definition_id == first.definition.definition_id
        assert first.version.locator == SOURCE_URLS[source_id]
        assert (first.version.observation_model, first.version.baseline_policy.kind,
                first.version.coverage_mappings[0].contribution) == semantics
        assert first.version.dependencies[0].kind is SourceDependencyKind.ORIGINATING_MATERIAL
        assert first.rights == _rights(source_id)
        assert first.version.rights.rights_decision_id
        assert first.version.rights.rights_policy_version == (
            "hermes-observed-portfolio-rights-v1"
            if source_id == "HK-02" else "hermes-govuk-text-ogl-v1"
        )


def test_native_source_definition_needs_real_current_rights():
    with pytest.raises(ValueError, match="current permitted rights"):
        native_source_definition_requests(source_id="UK-02", rights=_rights("UK-02", decision="HOLD"))
    with pytest.raises(ValueError, match="does not need"):
        native_source_definition_requests(source_id="UK-01", rights=_rights("UK-01"))

    assert register_missing_native_source_definitions(
        sources=None, proof=None, rights_by_source={},
    ) == {}
    with pytest.raises(ValueError, match="exceeds missing sources"):
        register_missing_native_source_definitions(
            sources=None, proof=None, rights_by_source={"UK-01": _rights("UK-01")},
        )


def test_register_missing_native_source_definitions_replays_on_disposable_authority(
    tmp_path, monkeypatch,
):
    args = _args(tmp_path, monkeypatch)
    args["principal_id"] = OPERATOR_PRINCIPAL_ID
    args["authority_domain"] = OPERATOR_AUTHORITY_DOMAIN
    rights = {source_id: _rights(source_id) for source_id in MISSING_SOURCE_IDS}
    with open_native_runtime(**args) as runtime:
        first = register_missing_native_source_definitions(
            sources=runtime.authority.sources, proof=runtime.proof,
            rights_by_source={"UK-02": rights["UK-02"]},
        )
        original_version_id = runtime.authority.sources.current_summary(
            first["UK-02"], proof=runtime.proof,
        ).version_id
        replay = register_missing_native_source_definitions(
            sources=runtime.authority.sources, proof=runtime.proof,
            rights_by_source={"UK-02": _rights("UK-02", evidence_tag="new")},
        )
        assert first == replay
        assert set(first) == {"UK-02"}
        assert runtime.authority.sources.current_summary(
            first["UK-02"], proof=runtime.proof,
        ).version_id == original_version_id
        remaining = register_missing_native_source_definitions(
            sources=runtime.authority.sources, proof=runtime.proof,
            rights_by_source={
                source_id: rights[source_id]
                for source_id in MISSING_SOURCE_IDS if source_id != "UK-02"
            },
        )
        for source_id, definition_id in (first | remaining).items():
            current = runtime.authority.sources.current_summary(
                definition_id, proof=runtime.proof,
            )
            version = runtime.authority.sources.version_details(
                current.version_id, proof=runtime.proof,
            )
            assert version.request.locator == SOURCE_URLS[source_id]
